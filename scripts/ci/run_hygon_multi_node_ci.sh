#!/usr/bin/env bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT

set -Eeuo pipefail

activate_dtk() {
    export USER=root
    set +u
    # shellcheck disable=SC1091
    source /opt/dtk/env.sh
    set -u
    export PATH="/opt/mpi/bin:${PATH}"
    export LD_LIBRARY_PATH="/opt/mpi/lib:/opt/hwloc/lib:${LD_LIBRARY_PATH:-}"
}

generate_topology() {
    local nic_name="$1"
    local nic_index="$2"

    [[ -d "/sys/class/infiniband/${nic_name}" ]] || {
        echo "ERROR: HCA ${nic_name} does not exist on $(hostname)" >&2
        exit 1
    }
    grep -q '^4: ACTIVE' "/sys/class/infiniband/${nic_name}/ports/1/state" || {
        echo "ERROR: HCA ${nic_name} is not ACTIVE on $(hostname)" >&2
        cat "/sys/class/infiniband/${nic_name}/ports/1/state" >&2
        exit 1
    }

    {
        for node_path in /sys/class/kfd/kfd/topology/nodes/*; do
            local gpu_id location_raw domain_raw location_id domain_id
            local bus device function

            gpu_id="$(cat "${node_path}/gpu_id" 2>/dev/null || echo 0)"
            [[ "$gpu_id" != 0 ]] || continue

            location_raw="$(awk '$1 == "location_id" { print $2; exit }' "${node_path}/properties")"
            domain_raw="$(awk '$1 == "domain" { print $2; exit }' "${node_path}/properties")"
            [[ -n "$location_raw" ]] || continue
            domain_raw="${domain_raw:-0}"

            if [[ "$location_raw" == 0x* ]]; then
                location_id=$((location_raw))
            else
                location_id=$((10#$location_raw))
            fi
            if [[ "$domain_raw" == 0x* ]]; then
                domain_id=$((domain_raw))
            else
                domain_id=$((10#$domain_raw))
            fi

            bus=$(((location_id >> 8) & 0xff))
            device=$(((location_id >> 3) & 0x1f))
            function=$((location_id & 0x7))
            printf '%04x:%02x:%02x.%x %s %s\n' \
                "$domain_id" "$bus" "$device" "$function" \
                "$nic_name" "$nic_index"
        done
    } | sort -u > /home/topo.config

    [[ "$(wc -l < /home/topo.config)" -eq 8 ]] || {
        echo "ERROR: expected 8 GPUs in /home/topo.config" >&2
        cat /home/topo.config >&2
        exit 1
    }
    while read -r gpu_bdf configured_nic configured_index; do
        [[ -e "/sys/bus/pci/devices/${gpu_bdf}" ]] || {
            echo "ERROR: topology contains missing GPU ${gpu_bdf}" >&2
            exit 1
        }
        [[ "$configured_nic" == "$nic_name" && "$configured_index" == "$nic_index" ]] || exit 1
    done < /home/topo.config
    cat /home/topo.config
}

install_dtk() {
    local dtk_source="$1"
    local tarball="$dtk_source"
    local extracted

    if [[ ! -f /opt/dtk/env.sh ]]; then
        if [[ "$dtk_source" =~ ^https?:// ]]; then
            tarball="/opt/$(basename "$dtk_source")"
            curl --fail --location --retry 3 "$dtk_source" --output "$tarball"
        else
            [[ -f "$tarball" ]] || {
                echo "ERROR: DTK archive is missing in the container: ${tarball}" >&2
                exit 1
            }
        fi
        tar -xzf "$tarball" -C /opt
        extracted="$(find /opt -mindepth 1 -maxdepth 1 -type d -name 'dtk-*' | sort | tail -n 1)"
        [[ -n "$extracted" ]] || { echo "ERROR: DTK directory was not created" >&2; exit 1; }
        ln -sfn "$extracted" /opt/dtk
    fi
    test -f /opt/dtk/env.sh
}

upgrade_rdma_core() {
    dnf --refresh install -y \
        rdma-core rdma-core-devel \
        libibverbs libibverbs-devel libibverbs-utils

    local rdma_version
    rdma_version="$(rpm -q --qf '%{VERSION}' rdma-core)"
    if [[ "$(printf '%s\n' 45 "$rdma_version" | sort -V | head -n 1)" != 45 ]]; then
        echo "ERROR: rdma-core >= 45 is required; installed version is ${rdma_version}" >&2
        exit 1
    fi
    rpm -q rdma-core libibverbs
}

prepare_container() {
    local source_tar="$1"
    local dtk_source="$2"
    local nic_name="$3"
    local nic_index="$4"
    local ssh_port="$5"
    local primary_host="$6"
    local secondary_host="$7"
    local primary_ip="$8"
    local secondary_ip="$9"

    dnf install -y curl openssh-clients openssh-server python3-pip iproute procps-ng
    upgrade_rdma_core
    install_dtk "$dtk_source"
    activate_dtk

    python3 -m pip install pyyaml
    python3 -c 'import yaml; print("PyYAML", yaml.__version__)'

    grep -Eq "^${primary_ip}[[:space:]]+${primary_host}([[:space:]]|$)" /etc/hosts || \
        printf '%s %s\n' "$primary_ip" "$primary_host" >> /etc/hosts
    grep -Eq "^${secondary_ip}[[:space:]]+${secondary_host}([[:space:]]|$)" /etc/hosts || \
        printf '%s %s\n' "$secondary_ip" "$secondary_host" >> /etc/hosts

    ssh-keygen -A
    install -d -m 0755 /run/sshd
    install -d -m 0700 /root/.ssh
    install -m 0600 /work/ssh/id_ed25519 /root/.ssh/id_ed25519
    install -m 0644 /work/ssh/id_ed25519.pub /root/.ssh/id_ed25519.pub
    install -m 0600 /work/ssh/id_ed25519.pub /root/.ssh/authorized_keys
    cat > /root/.ssh/config <<EOF
Host ${primary_host} ${secondary_host}
    User root
    Port ${ssh_port}
    IdentityFile /root/.ssh/id_ed25519
    IdentitiesOnly yes
    BatchMode yes
    ConnectTimeout 10
    StrictHostKeyChecking accept-new
    CheckHostIP no
EOF
    chmod 0600 /root/.ssh/config
    /usr/sbin/sshd -t
    /usr/sbin/sshd -p "$ssh_port"

    generate_topology "$nic_name" "$nic_index"

    local gid_type gid_value
    gid_type="$(cat "/sys/class/infiniband/${nic_name}/ports/1/gid_attrs/types/${ROCSHMEM_CI_GID_INDEX:-3}")"
    gid_value="$(cat "/sys/class/infiniband/${nic_name}/ports/1/gids/${ROCSHMEM_CI_GID_INDEX:-3}")"
    [[ "$gid_type" == 'RoCE v2' ]] || {
        echo "ERROR: ${nic_name} GID ${ROCSHMEM_CI_GID_INDEX:-3} is ${gid_type}, expected RoCE v2" >&2
        exit 1
    }
    [[ "$gid_value" != '0000:0000:0000:0000:0000:0000:0000:0000' ]] || {
        echo "ERROR: ${nic_name} GID ${ROCSHMEM_CI_GID_INDEX:-3} is empty" >&2
        exit 1
    }
    printf 'HCA=%s GID_INDEX=%s GID=%s TYPE=%s\n' \
        "$nic_name" "${ROCSHMEM_CI_GID_INDEX:-3}" "$gid_value" "$gid_type"

    rm -rf /home/rocSHMEM-das /opt/rocshmem
    install -d /home/rocSHMEM-das
    tar -xf "$source_tar" -C /home/rocSHMEM-das
    install -d /home/rocSHMEM-das/build
    cd /home/rocSHMEM-das/build
    ../scripts/build_configs/ipc_ro_mlx5

    test -x /opt/rocshmem/share/rocshmem/run_ctest.sh
    test -x /opt/rocshmem/share/rocshmem/rocshmem_functional_tests
    ctest --test-dir /opt/rocshmem/bin/rocshmem -N | tail -n 3
}

run_suite() {
    local suite="$1"
    local primary_host="$2"
    local secondary_host="$3"
    local gid_index="$4"

    activate_dtk
    export ROCSHMEM_TEST_DIR=/opt/rocshmem/bin/rocshmem
    export ROCSHMEM_TEST_LOG_DIR=/patch/test_log

    /opt/rocshmem/share/rocshmem/run_ctest.sh "$suite" \
        --host "${primary_host},${secondary_host}" \
        --topo /home/topo.config \
        --ib-gid-index "$gid_index"
}

case "${1:-}" in
    __prepare)
        shift
        prepare_container "$@"
        exit
        ;;
    __test)
        shift
        run_suite "$@"
        exit
        ;;
esac

required=(
    ROCSHMEM_CI_IMAGE ROCSHMEM_CI_SUITE
    ROCSHMEM_CI_PRIMARY_HOST ROCSHMEM_CI_SECONDARY_HOST
)
for name in "${required[@]}"; do
    [[ -n "${!name:-}" ]] || { echo "ERROR: ${name} is required" >&2; exit 2; }
done

PRIMARY_HOST="$ROCSHMEM_CI_PRIMARY_HOST"
SECONDARY_HOST="$ROCSHMEM_CI_SECONDARY_HOST"
REMOTE_USER="${ROCSHMEM_CI_REMOTE_USER:-github}"
GID_INDEX="${ROCSHMEM_CI_GID_INDEX:-3}"
PRIMARY_IP="${ROCSHMEM_CI_PRIMARY_IP:-10.17.1.1}"
SECONDARY_IP="${ROCSHMEM_CI_SECONDARY_IP:-10.17.1.2}"
PRIMARY_NIC="${ROCSHMEM_CI_PRIMARY_NIC:-mlx5_6}"
SECONDARY_NIC="${ROCSHMEM_CI_SECONDARY_NIC:-mlx5_0}"
DTK_HOST_PATH="${ROCSHMEM_CI_DTK_HOST_PATH:-/public/opendas/ArchivedFile/dtk-pkg/dtk26.04/DTK-26.04-rc4-centos8-x86_64.tar.gz}"
DTK_CONTAINER_PATH="/opt/$(basename "$DTK_HOST_PATH")"
PRIMARY_NIC_INDEX="${ROCSHMEM_CI_PRIMARY_NIC_INDEX:-6}"
SECONDARY_NIC_INDEX="${ROCSHMEM_CI_SECONDARY_NIC_INDEX:-0}"
SETUP_TIMEOUT="${ROCSHMEM_CI_SETUP_TIMEOUT:-3600}"
TEST_TIMEOUT="${ROCSHMEM_CI_TEST_TIMEOUT:-21600}"
LOCK_FILE="${ROCSHMEM_CI_LOCK_FILE:-/tmp/mooncake-hcu-cross-node.lock}"
LOCK_TIMEOUT="${ROCSHMEM_CI_LOCK_TIMEOUT:-900}"
PORT_MIN="${ROCSHMEM_CI_PORT_RANGE_MIN:-20000}"
PORT_MAX="${ROCSHMEM_CI_PORT_RANGE_MAX:-29999}"
REMOTE="${REMOTE_USER}@${SECONDARY_HOST}"
RUN_TOKEN="${GITHUB_RUN_ID:-manual}-${GITHUB_RUN_ATTEMPT:-0}-$$"
LOCAL_ROOT="${RUNNER_TEMP:-/tmp}/rocshmem-${RUN_TOKEN}"
REMOTE_ROOT="/tmp/rocshmem-${RUN_TOKEN}"
LOG_ROOT="${ROCSHMEM_CI_LOG_DIR:-${LOCAL_ROOT}/logs}"
SOURCE_TAR="${LOCAL_ROOT}/source.tar"
SSH_DIR="${LOCAL_ROOT}/ssh"
SCRIPT_PATH="$(realpath "${BASH_SOURCE[0]}")"
PRIMARY_CONTAINER="rocshmem-hygon-primary-${RUN_TOKEN}"
SECONDARY_CONTAINER="rocshmem-hygon-secondary-${RUN_TOKEN}"
PRIMARY_LOCK_PID=""
SECONDARY_LOCK_PID=""
SSH_PORT=""

case "$ROCSHMEM_CI_SUITE" in standard|full) ;; *) echo "ERROR: suite must be standard or full" >&2; exit 2;; esac
for command_name in awk cksum docker flock git grep realpath scp ssh ss timeout; do
    command -v "$command_name" >/dev/null 2>&1 || {
        echo "ERROR: runner command is missing: ${command_name}" >&2
        exit 1
    }
done
[[ "$LOCK_TIMEOUT" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: invalid lock timeout" >&2; exit 2; }
[[ "$PORT_MIN" =~ ^[0-9]+$ && "$PORT_MAX" =~ ^[0-9]+$ ]] || { echo "ERROR: invalid port range" >&2; exit 2; }
((PORT_MIN >= 1024 && PORT_MAX <= 65535 && PORT_MAX > PORT_MIN)) || { echo "ERROR: invalid port range" >&2; exit 2; }

SSH_OPTIONS=(-o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=yes)
mkdir -p "$LOCAL_ROOT" "$LOG_ROOT" "$SSH_DIR"
git archive --format=tar --output="$SOURCE_TAR" HEAD
ssh-keygen -q -t ed25519 -N '' -f "$SSH_DIR/id_ed25519"

collect_remote_logs() {
    set +e
    mkdir -p "$LOG_ROOT/secondary"
    scp -r "${SSH_OPTIONS[@]}" "${REMOTE}:${REMOTE_ROOT}/logs/." "$LOG_ROOT/secondary/" >/dev/null 2>&1 || true
}

cleanup() {
    local rc=$?
    trap - EXIT
    set +e

    # Preserve diagnostics before removing either container. This also runs
    # when a test is terminated by the outer timeout.
    docker cp "${PRIMARY_CONTAINER}:/patch/test_log/." "$LOG_ROOT/" >/dev/null 2>&1 || true
    ssh "${SSH_OPTIONS[@]}" "$REMOTE" \
        "docker cp '${SECONDARY_CONTAINER}:/patch/test_log/.' '$REMOTE_ROOT/logs/' >/dev/null 2>&1 || true" || true
    collect_remote_logs
    docker rm -f "$PRIMARY_CONTAINER" >/dev/null 2>&1 || true
    ssh "${SSH_OPTIONS[@]}" "$REMOTE" "docker rm -f '$SECONDARY_CONTAINER' >/dev/null 2>&1 || true" || true
    ssh "${SSH_OPTIONS[@]}" "$REMOTE" "case '$REMOTE_ROOT' in /tmp/rocshmem-*) rm -rf -- '$REMOTE_ROOT';; esac" || true
    if [[ -n "$SECONDARY_LOCK_PID" ]]; then
        kill "$SECONDARY_LOCK_PID" >/dev/null 2>&1 || true
        wait "$SECONDARY_LOCK_PID" >/dev/null 2>&1 || true
    fi
    if [[ -n "$PRIMARY_LOCK_PID" ]]; then
        kill "$PRIMARY_LOCK_PID" >/dev/null 2>&1 || true
        wait "$PRIMARY_LOCK_PID" >/dev/null 2>&1 || true
    fi
    exit "$rc"
}
trap cleanup EXIT

ssh "${SSH_OPTIONS[@]}" "$REMOTE" 'command -v docker >/dev/null'

wait_for_lock() {
    local log_file="$1" process_id="$2" node_name="$3"
    local attempt
    for ((attempt = 1; attempt <= LOCK_TIMEOUT + 10; attempt++)); do
        grep -q '^LOCK_ACQUIRED$' "$log_file" && return 0
        kill -0 "$process_id" 2>/dev/null || {
            echo "ERROR: failed to acquire HCU lock on ${node_name}" >&2
            cat "$log_file" >&2 || true
            return 1
        }
        sleep 1
    done
    echo "ERROR: timed out acquiring HCU lock on ${node_name}" >&2
    return 1
}

PARENT_PID="$$"
(
    exec 9>"$LOCK_FILE"
    flock -w "$LOCK_TIMEOUT" 9 || exit 75
    echo LOCK_ACQUIRED
    while kill -0 "$PARENT_PID" 2>/dev/null; do sleep 5; done
) >"$LOG_ROOT/primary-lock.log" 2>&1 &
PRIMARY_LOCK_PID=$!
wait_for_lock "$LOG_ROOT/primary-lock.log" "$PRIMARY_LOCK_PID" "$PRIMARY_HOST"

ssh "${SSH_OPTIONS[@]}" "$REMOTE" bash -s -- "$LOCK_FILE" "$LOCK_TIMEOUT" \
    >"$LOG_ROOT/secondary-lock.log" 2>&1 <<'REMOTE_LOCK' &
set -Eeuo pipefail
lock_file="$1"
lock_timeout="$2"
session_parent="$PPID"
command -v flock >/dev/null
exec 9>"$lock_file"
flock -w "$lock_timeout" 9 || exit 75
echo LOCK_ACQUIRED
while kill -0 "$session_parent" 2>/dev/null; do sleep 5; done
REMOTE_LOCK
SECONDARY_LOCK_PID=$!
wait_for_lock "$LOG_ROOT/secondary-lock.log" "$SECONDARY_LOCK_PID" "$SECONDARY_HOST"

port_is_free_on_both_nodes() {
    local candidate="$1"
    if ss -H -ltn | grep -Eq "[:.]${candidate}[[:space:]]"; then
        return 1
    fi
    ssh "${SSH_OPTIONS[@]}" "$REMOTE" \
        "if ss -H -ltn | grep -Eq '[:.]${candidate}[[:space:]]'; then exit 1; fi"
}

port_span=$((PORT_MAX - PORT_MIN + 1))
port_seed="$(printf '%s' "$RUN_TOKEN" | cksum | awk '{print $1}')"
for ((port_attempt = 0; port_attempt < port_span; port_attempt++)); do
    candidate=$((PORT_MIN + ((port_seed + port_attempt) % port_span)))
    if port_is_free_on_both_nodes "$candidate"; then
        SSH_PORT="$candidate"
        break
    fi
done
[[ -n "$SSH_PORT" ]] || { echo "ERROR: no common free container SSH port in ${PORT_MIN}-${PORT_MAX}" >&2; exit 1; }
echo "Cross-node locks acquired; selected container SSH port ${SSH_PORT}."

[[ -f "$DTK_HOST_PATH" ]] || {
    echo "ERROR: DTK archive is missing on ${PRIMARY_HOST}: ${DTK_HOST_PATH}" >&2
    exit 1
}
ssh "${SSH_OPTIONS[@]}" "$REMOTE" "test -f '$DTK_HOST_PATH'" || {
    echo "ERROR: DTK archive is missing on ${SECONDARY_HOST}: ${DTK_HOST_PATH}" >&2
    exit 1
}

ssh "${SSH_OPTIONS[@]}" "$REMOTE" "mkdir -p '$REMOTE_ROOT/ssh' '$REMOTE_ROOT/logs'"
scp "${SSH_OPTIONS[@]}" "$SOURCE_TAR" "$SCRIPT_PATH" "$SSH_DIR/id_ed25519" "$SSH_DIR/id_ed25519.pub" "${REMOTE}:${REMOTE_ROOT}/"
ssh "${SSH_OPTIONS[@]}" "$REMOTE" "mv '$REMOTE_ROOT/id_ed25519' '$REMOTE_ROOT/id_ed25519.pub' '$REMOTE_ROOT/ssh/'"

docker run --name "$PRIMARY_CONTAINER" -u root \
    --ulimit memlock=-1:-1 --shm-size=32g --privileged \
    --device=/dev/kfd --device=/dev/mkfd --device=/dev/dri/ \
    -v /opt/hyhal:/opt/hyhal:ro --network=host --ipc=host \
    -v "${DTK_HOST_PATH}:${DTK_CONTAINER_PATH}:ro" \
    --group-add video -d "$ROCSHMEM_CI_IMAGE" tail -f /dev/null

ssh "${SSH_OPTIONS[@]}" "$REMOTE" docker run --name "$SECONDARY_CONTAINER" -u root \
    --ulimit memlock=-1:-1 --shm-size=32g --privileged \
    --device=/dev/kfd --device=/dev/mkfd --device=/dev/dri/ \
    -v /opt/hyhal:/opt/hyhal:ro --network=host --ipc=host \
    -v "${DTK_HOST_PATH}:${DTK_CONTAINER_PATH}:ro" \
    --group-add video -d "$ROCSHMEM_CI_IMAGE" tail -f /dev/null

docker exec "$PRIMARY_CONTAINER" mkdir -p /work /patch/test_log
ssh "${SSH_OPTIONS[@]}" "$REMOTE" \
    "docker exec '$SECONDARY_CONTAINER' mkdir -p /work /patch/test_log"

docker cp "$SOURCE_TAR" "${PRIMARY_CONTAINER}:/work/source.tar"
docker cp "$SCRIPT_PATH" "${PRIMARY_CONTAINER}:/work/run_hygon_multi_node_ci.sh"
docker cp "$SSH_DIR" "${PRIMARY_CONTAINER}:/work/ssh"
ssh "${SSH_OPTIONS[@]}" "$REMOTE" \
    "docker cp '$REMOTE_ROOT/source.tar' '${SECONDARY_CONTAINER}:/work/source.tar' && \
     docker cp '$REMOTE_ROOT/run_hygon_multi_node_ci.sh' '${SECONDARY_CONTAINER}:/work/run_hygon_multi_node_ci.sh' && \
     docker cp '$REMOTE_ROOT/ssh' '${SECONDARY_CONTAINER}:/work/ssh'"

PREPARE_COMMON=(
    "$DTK_CONTAINER_PATH" "$SSH_PORT" "$PRIMARY_HOST" "$SECONDARY_HOST"
    "$PRIMARY_IP" "$SECONDARY_IP"
)

timeout "$SETUP_TIMEOUT" docker exec \
    -e "PIP_INDEX_URL=${PIP_INDEX_URL:-}" \
    -e "PIP_TRUSTED_HOST=${PIP_TRUSTED_HOST:-}" \
    -e "ROCSHMEM_CI_GID_INDEX=${GID_INDEX}" \
    "$PRIMARY_CONTAINER" bash /work/run_hygon_multi_node_ci.sh __prepare \
    /work/source.tar "${PREPARE_COMMON[0]}" "$PRIMARY_NIC" "$PRIMARY_NIC_INDEX" \
    "${PREPARE_COMMON[@]:1}" >"$LOG_ROOT/build-${PRIMARY_HOST}.log" 2>&1 &
PRIMARY_PID=$!

ssh "${SSH_OPTIONS[@]}" "$REMOTE" \
    "timeout '$SETUP_TIMEOUT' docker exec \
       -e 'PIP_INDEX_URL=${PIP_INDEX_URL:-}' -e 'PIP_TRUSTED_HOST=${PIP_TRUSTED_HOST:-}' \
       -e 'ROCSHMEM_CI_GID_INDEX=${GID_INDEX}' \
       '$SECONDARY_CONTAINER' bash /work/run_hygon_multi_node_ci.sh __prepare \
       /work/source.tar '$DTK_CONTAINER_PATH' '$SECONDARY_NIC' '$SECONDARY_NIC_INDEX' \
       '$SSH_PORT' '$PRIMARY_HOST' '$SECONDARY_HOST' '$PRIMARY_IP' '$SECONDARY_IP'" \
    >"$LOG_ROOT/build-${SECONDARY_HOST}.log" 2>&1 &
SECONDARY_PID=$!

set +e
wait "$PRIMARY_PID"; PRIMARY_RC=$?
wait "$SECONDARY_PID"; SECONDARY_RC=$?
set -e
if [[ "$PRIMARY_RC" -ne 0 || "$SECONDARY_RC" -ne 0 ]]; then
    tail -n 100 "$LOG_ROOT/build-${PRIMARY_HOST}.log" || true
    tail -n 100 "$LOG_ROOT/build-${SECONDARY_HOST}.log" || true
    echo "ERROR: build failed: ${PRIMARY_HOST}=${PRIMARY_RC}, ${SECONDARY_HOST}=${SECONDARY_RC}" >&2
    exit 1
fi

# The validated manual run used the same functional-test executable on both
# nodes. Preserve that invariant even if separate builds contain timestamps or
# other non-reproducible sections.
CANONICAL_BINARY=/opt/rocshmem/share/rocshmem/rocshmem_functional_tests
docker cp "${PRIMARY_CONTAINER}:${CANONICAL_BINARY}" "${LOCAL_ROOT}/rocshmem_functional_tests"
scp "${SSH_OPTIONS[@]}" "${LOCAL_ROOT}/rocshmem_functional_tests" "${REMOTE}:${REMOTE_ROOT}/"
ssh "${SSH_OPTIONS[@]}" "$REMOTE" \
    "docker cp '$REMOTE_ROOT/rocshmem_functional_tests' '${SECONDARY_CONTAINER}:/tmp/rocshmem_functional_tests' && \
     docker exec '$SECONDARY_CONTAINER' install -m 0755 /tmp/rocshmem_functional_tests '$CANONICAL_BINARY'"

for host in "$PRIMARY_HOST" "$SECONDARY_HOST"; do
    docker exec "$PRIMARY_CONTAINER" ssh "$host" 'hostname; id -u'
done
docker exec "$PRIMARY_CONTAINER" bash -lc \
    "source /opt/dtk/env.sh && /opt/mpi/bin/mpirun --host '${PRIMARY_HOST},${SECONDARY_HOST}' \
     -np 2 --allow-run-as-root --map-by ppr:1:node -x PATH -x LD_LIBRARY_PATH hostname"

set -o pipefail
timeout --signal=TERM --kill-after=30s "$TEST_TIMEOUT" \
    docker exec "$PRIMARY_CONTAINER" bash /work/run_hygon_multi_node_ci.sh __test \
    "$ROCSHMEM_CI_SUITE" "$PRIMARY_HOST" "$SECONDARY_HOST" "$GID_INDEX" \
    2>&1 | tee "$LOG_ROOT/${ROCSHMEM_CI_SUITE}-console.log"

docker cp "${PRIMARY_CONTAINER}:/patch/test_log/." "$LOG_ROOT/"
collect_remote_logs
echo "rocSHMEM ${ROCSHMEM_CI_SUITE} suite passed on ${PRIMARY_HOST},${SECONDARY_HOST}."
