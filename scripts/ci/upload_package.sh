#!/usr/bin/env bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT

set -Eeuo pipefail

validate_run_package() {
    local installer="$1" destination="$2" payload_line
    test -s "$installer" && test -x "$installer" || return 1
    payload_line="$(awk '/^# __ROCSHMEM_TGZ_BELOW__$/ { print NR + 1; exit }' "$installer")"
    [[ "$payload_line" =~ ^[0-9]+$ ]] || {
        echo 'ERROR: RUN installer payload is missing' >&2; return 1;
    }
    tail -n +"$payload_line" "$installer" | gzip -t || return 1
    sh "$installer" --help || return 1
    # Smoke-install only inside the disposable build container; no hardware tests.
    sh "$installer" --prefix "$destination" || return 1
    test -s "$destination/lib/librocshmem.a" &&
        test -s "$destination/lib/cmake/rocshmem/rocshmem-config.cmake" &&
        test -x "$destination/bin/rocshmem_functional_tests" &&
        test -x "$destination/share/rocshmem/run_ctest.sh" &&
        test -s "$destination/share/rocshmem/rocshmem-env.sh" &&
        test -x "$destination/bin/rocshmem-uninstall" &&
        grep -qxF "install_prefix=$destination" "$destination/.rocshmem-install"
}

package_image_id() {
    local image="$1" id
    # The DTK toolchain is baked into this image, so a locally cached tag is
    # not trustworthy: any job with access to this Docker daemon could have
    # re-tagged it. Always fetch from the registry before packaging and then
    # build from the resolved image ID (content-addressed), never from the tag.
    echo "Pulling build image: $image" >&2
    docker pull "$image" >&2 || return 1
    id="$(docker image inspect --format '{{.Id}}' "$image")" || return 1
    [[ -n "$id" ]] || return 1
    printf '%s\n' "$id"
}

build_package() {
    # This entry point is ONLY for the disposable container created below.
    [[ -f /.dockerenv && "${ROCSHMEM_PACKAGE_CONTAINER:-}" == 1 ]]
    [[ "$(uname -m)" == x86_64 ]]
    local dtk_dir
    case "$ROCSHMEM_PACKAGE_VARIANT" in
        mlx5)
            # shellcheck source=/dev/null
            source /etc/os-release
            [[ "$ID" == ubuntu && "$VERSION_ID" == 22.04 ]]
            export DEBIAN_FRONTEND=noninteractive
            apt-get update
            apt-get install -y --no-install-recommends \
                git curl ca-certificates python3-pip binutils \
                libnuma-dev rdma-core libibverbs-dev ibverbs-utils
            ;;
        shca)
            # Same user-space SHCA setup as Mooncake, in the disposable container.
            # shellcheck source=/dev/null
            source /etc/os-release
            [[ "$ID" == ubuntu && "$VERSION_ID" == 22.04 ]]
            export DEBIAN_FRONTEND=noninteractive
            apt-get update
            apt-get install -y --no-install-recommends git curl ca-certificates python3-pip binutils libnuma-dev
            mkdir /tmp/rocshmem-shca-setup
            cd /tmp/rocshmem-shca-setup
            curl --fail --location --retry 3 -o mlxtoshca.sh \
                "${RESOURCE_SERVER_URL%/}/Jenkins/CompileDep/mooncake/mlxtoshca.sh"
            curl --fail --location --retry 3 -o shca-tools_2.500.4.B068-Ubuntu22.04_amd64.deb \
                "${RESOURCE_SERVER_URL%/}/Jenkins/CompileDep/mooncake/shca-tools_2.500.4.B068-Ubuntu22.04_amd64.deb"
            bash ./mlxtoshca.sh
            [[ "$(dpkg-query -W -f='${Status}' shca-tools)" == 'install ok installed' ]]
            test -s /usr/include/infiniband/shca_dv.h
            test -s /usr/lib/x86_64-linux-gnu/libshca.so
            ldconfig
            ldd /usr/lib/x86_64-linux-gnu/libshca.so > shca-ldd.log
            if grep -F 'not found' shca-ldd.log; then
                echo 'ERROR: SHCA shared library dependencies are missing' >&2; return 1;
            fi
            ;;
        *) echo 'ERROR: unknown package variant' >&2; return 1 ;;
    esac
    dtk_dir="$(readlink -f /opt/dtk)"
    if [[ -f /opt/dtk/.dtk_version ]]; then
        grep -Eq "^DTK-${ROCSHMEM_PACKAGE_DTK_VERSION:0:2}\.${ROCSHMEM_PACKAGE_DTK_VERSION:2:2}([^0-9]|$)" /opt/dtk/.dtk_version || {
            echo "ERROR: image DTK version does not match $ROCSHMEM_PACKAGE_DTK_VERSION" >&2; return 1;
        }
    fi
    export USER=root
    set +u
    # shellcheck disable=SC1091
    source /opt/dtk/env.sh
    set -u
    [[ "$(readlink -f "$(command -v hipcc)")" == "$dtk_dir/"* ]] || {
        echo 'ERROR: hipcc does not resolve to the DTK in /opt/dtk' >&2; return 1;
    }
    export PATH="/opt/dtk/llvm/bin:/opt/dtk/bin:/opt/mpi/bin:${PATH}"
    command -v llvm-ar
    /opt/mpi/bin/mpicxx --version
    export LD_LIBRARY_PATH="/opt/mpi/lib:/opt/hwloc/lib:${LD_LIBRARY_PATH:-}"
    python3 -m pip install pyyaml

    # Keep git metadata for ROCm's version detection, but never copy credentials.
    git clone /tmp/rocshmem-source.bundle /home/rocshmem-package-source
    cd /home/rocshmem-package-source
    git checkout --detach "$ROCSHMEM_PACKAGE_SHA"
    git remote remove origin
    [[ "$(git rev-parse HEAD)" == "$ROCSHMEM_PACKAGE_SHA" ]]
    # This prefix is NOT mounted from the host. Remove any image-baked build.
    [[ ! -L /opt/rocshmem ]]
    rm -rf -- /opt/rocshmem
    mkdir build
    cd build
    export ASAN=OFF BUILD_TYPE=Release INSTALL_PREFIX=/opt/rocshmem TZ=UTC-8
    bash "../scripts/build_configs/gda_${ROCSHMEM_PACKAGE_VARIANT}"
    grep -qx 'USE_GDA:BOOL=ON' CMakeCache.txt
    if [[ "$ROCSHMEM_PACKAGE_VARIANT" == shca ]]; then
        grep -qx 'GDA_SHCA:BOOL=ON' CMakeCache.txt
        grep -qx 'GDA_MLX5:BOOL=OFF' CMakeCache.txt
    else
        grep -qx 'GDA_MLX5:BOOL=ON' CMakeCache.txt
        grep -qx 'GDA_SHCA:BOOL=OFF' CMakeCache.txt
    fi

    local upstream_version expected_filename
    upstream_version="$(sed -n 's/^set(CPACK_PACKAGE_VERSION "\([^"]*\)")$/\1/p' CPackConfig.cmake)"
    expected_filename="$(sed -n 's/^set(CPACK_PACKAGE_FILE_NAME "\([^"]*\)")$/\1/p' CPackConfig.cmake)"
    [[ -n "$upstream_version" && -n "$expected_filename" ]] || {
        echo 'ERROR: generated CPack package identity is missing' >&2; return 1;
    }
    expected_filename+='.run'
    test -s MakeRunPackage.cmake
    test -s rocshmem.run.stub
    mkdir -p /tmp/rocshmem-run-output
    # The project's post-build hook appends its installer stub to the TGZ.
    cpack --config "$PWD/CPackConfig.cmake" -G TGZ \
        -D "CPACK_POST_BUILD_SCRIPTS=$PWD/MakeRunPackage.cmake" \
        -D CPACK_PACKAGING_INSTALL_PREFIX=/opt/rocshmem \
        -D CPACK_INCLUDE_TOPLEVEL_DIRECTORY=ON \
        -D CPACK_MONOLITHIC_INSTALL=ON \
        -D CPACK_STRIP_FILES=OFF \
        -B /tmp/rocshmem-run-output

    local run_files installer inspect
    shopt -s nullglob
    run_files=(/tmp/rocshmem-run-output/*.run)
    [[ "${#run_files[@]}" -eq 1 ]] || { echo 'ERROR: expected one monolithic RUN'; return 1; }
    installer="${run_files[0]}"
    [[ "$(basename "$installer")" == "$expected_filename" ]]
    inspect="$(mktemp -d /tmp/rocshmem-run-inspect.XXXXXX)"
    validate_run_package "$installer" "$inspect/install"

    local py_version py_tag
    py_version="$(python3 -c 'import sys; print("%d.%d.%d" % sys.version_info[:3])' 2>/dev/null || echo unknown)"
    py_tag="$(python3 -c 'import sys; v = "%d%d" % sys.version_info[:2]; print("-cp" + v + "-cp" + v)' 2>/dev/null || echo unknown)"
    mkdir /tmp/rocshmem-run-artifacts
    cp "$installer" /tmp/rocshmem-run-artifacts/
    chmod 0755 "/tmp/rocshmem-run-artifacts/${expected_filename}"
    {
        printf 'commit=%s\nrun_id=%s\nrun_attempt=%s\n' \
            "$ROCSHMEM_PACKAGE_SHA" "$GITHUB_RUN_ID" "$GITHUB_RUN_ATTEMPT"
        printf 'package_format=run\npackage_filename=%s\nupstream_version=%s\n' "$expected_filename" "$upstream_version"
        printf 'dtk_version=%s\ndtk_directory=%s\n' "$ROCSHMEM_PACKAGE_DTK_VERSION" "$dtk_dir"
        printf 'python_tag=%s\npython_version=%s\npython_executable=%s\n' \
            "$py_tag" "$py_version" "$(command -v python3 || echo unknown)"
        printf 'build_timezone=UTC+08:00\n'
        printf 'build_config=scripts/build_configs/gda_%s\nstrip=OFF\narchitecture=%s\n' \
            "$ROCSHMEM_PACKAGE_VARIANT" "$(uname -m)"
        cat /etc/os-release
        hipcc --version
        /opt/mpi/bin/mpiexec --version
        cmake --version
        if [[ "$ROCSHMEM_PACKAGE_VARIANT" == shca ]]; then
            dpkg-query -W shca-tools
            sha256sum /tmp/rocshmem-shca-setup/mlxtoshca.sh /tmp/rocshmem-shca-setup/*.deb
        else
            dpkg-query -W rdma-core libibverbs-dev ibverbs-utils
        fi
    } > /tmp/rocshmem-run-artifacts/build-info.txt
    cd /tmp/rocshmem-run-artifacts
    sha256sum ./*.run > SHA256SUMS
    sha256sum --check SHA256SUMS
}

# Allow the focused local tests to load helpers without starting CI or Docker.
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
    return 0
fi

if [[ "${1:-}" == __build ]]; then
    build_package
    exit 0
fi

for required in ROCSHMEM_CI_IMAGE PIP_INDEX_URL PIP_TRUSTED_HOST \
    ROCSHMEM_PACKAGE_VARIANT ROCSHMEM_PACKAGE_DTK_VERSION ROCSHMEM_PACKAGE_SHA \
    ROCSHMEM_PACKAGE_ROOT ROCSHMEM_PACKAGE_HOST_DIR \
    GITHUB_RUN_ID GITHUB_RUN_ATTEMPT; do
    [[ -n "${!required:-}" ]] || { echo "ERROR: missing ${required}" >&2; exit 1; }
done
[[ "$(hostname)" == github-nmz1 ]]
[[ "$ROCSHMEM_PACKAGE_HOST_DIR" == /ci/github_ci_packages/rocshmem ]] || {
    echo 'ERROR: unexpected host package directory' >&2; exit 1;
}
[[ "$ROCSHMEM_PACKAGE_SHA" =~ ^[0-9a-f]{40}$ ]]
case "$ROCSHMEM_PACKAGE_VARIANT" in
    mlx5|shca) ;;
    *) echo 'ERROR: unknown package variant' >&2; exit 1 ;;
esac
[[ "$(git rev-parse HEAD)" == "$ROCSHMEM_PACKAGE_SHA" ]]
[[ "$(git rev-parse --is-shallow-repository)" == false ]]
mkdir -p "$ROCSHMEM_PACKAGE_ROOT/logs"
exec > >(tee "$ROCSHMEM_PACKAGE_ROOT/logs/package.log") 2>&1

# Same primary-node lock as the existing two-node CI / Mooncake jobs.
exec 9>/tmp/mooncake-hcu-cross-node.lock
flock -w 900 9 || { echo 'ERROR: timed out waiting for nmz1 CI lock'; exit 1; }
task_dir="$(mktemp -d "${RUNNER_TEMP:-/tmp}/rocshmem-package.XXXXXX")"
container_name="rocshmem-run-${ROCSHMEM_PACKAGE_VARIANT}-${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT}-$$"
container_created=false
host_copy_temp=""
cleanup() {
    local rc=$?
    trap - EXIT
    if [[ "$container_created" == true ]]; then
        docker logs "$container_name" > "$ROCSHMEM_PACKAGE_ROOT/logs/container.log" 2>&1 || true
        docker rm -f "$container_name" || true
    fi
    if [[ -n "$host_copy_temp" ]]; then
        case "$host_copy_temp" in
            "$ROCSHMEM_PACKAGE_HOST_DIR"/.*.tmp-*) rm -f -- "$host_copy_temp" || true ;;
        esac
    fi
    # Only remove this invocation's known file and then its empty directory.
    rm -f -- "$task_dir/source.bundle" || true
    rmdir -- "$task_dir" || true
    echo "Packaging exit code: $rc"
    exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
git bundle create "$task_dir/source.bundle" HEAD
image_id="$(package_image_id "$ROCSHMEM_CI_IMAGE")"
docker image inspect --format '{{json .RepoDigests}}' "$image_id" > "$ROCSHMEM_PACKAGE_ROOT/logs/image-digests.json"
docker create --name "$container_name" --user root \
    --ulimit memlock=-1:-1 --shm-size=32g --privileged \
    --device=/dev/kfd --device=/dev/mkfd --device=/dev/dri/ \
    --network=host --ipc=host --group-add video \
    -v /opt/hyhal:/opt/hyhal:ro \
    -e ROCSHMEM_PACKAGE_VARIANT -e ROCSHMEM_PACKAGE_DTK_VERSION -e RESOURCE_SERVER_URL \
    -e ROCSHMEM_PACKAGE_CONTAINER=1 \
    -e ROCSHMEM_PACKAGE_SHA \
    -e GITHUB_RUN_ID -e GITHUB_RUN_ATTEMPT -e PIP_INDEX_URL -e PIP_TRUSTED_HOST \
    "$image_id" tail -f /dev/null
container_created=true
docker start "$container_name"
docker cp "$task_dir/source.bundle" "$container_name:/tmp/rocshmem-source.bundle"
docker cp scripts/ci/upload_package.sh "$container_name:/tmp/upload_package.sh"
timeout --signal=TERM --kill-after=30s 6000s \
    docker exec "$container_name" bash /tmp/upload_package.sh __build
mkdir -p "$ROCSHMEM_PACKAGE_ROOT/packages"
docker cp "$container_name:/tmp/rocshmem-run-artifacts/." "$ROCSHMEM_PACKAGE_ROOT/packages/"
printf 'image=%s\nimage_id=%s\n' "$ROCSHMEM_CI_IMAGE" "$image_id" >> "$ROCSHMEM_PACKAGE_ROOT/packages/build-info.txt"
cp "$ROCSHMEM_PACKAGE_ROOT/logs/image-digests.json" "$ROCSHMEM_PACKAGE_ROOT/packages/"
(cd "$ROCSHMEM_PACKAGE_ROOT/packages"; sha256sum --check SHA256SUMS)

# Publish the verified RUN onto the nmz1 host. Copy to a hidden temporary file
# first so consumers never observe a partially copied package.
shopt -s nullglob
host_runs=("$ROCSHMEM_PACKAGE_ROOT"/packages/*.run)
[[ "${#host_runs[@]}" -eq 1 ]] || {
    echo 'ERROR: expected exactly one verified RUN on the host' >&2; exit 1;
}
mkdir -p -- "$ROCSHMEM_PACKAGE_HOST_DIR"
host_package="$ROCSHMEM_PACKAGE_HOST_DIR/$(basename "${host_runs[0]}")"
host_copy_temp="$ROCSHMEM_PACKAGE_HOST_DIR/.$(basename "${host_runs[0]}").tmp-${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT}-$$"
[[ ! -e "$host_package" && ! -e "$host_copy_temp" ]] || {
    echo "ERROR: refusing to overwrite existing host package: $host_package" >&2; exit 1;
}
cp -- "${host_runs[0]}" "$host_copy_temp"
chmod 0755 "$host_copy_temp"
cmp -s -- "${host_runs[0]}" "$host_copy_temp" || {
    echo 'ERROR: host package copy verification failed' >&2; exit 1;
}
mv -- "$host_copy_temp" "$host_package"
host_copy_temp=""
printf 'host_package_path=%s\n' "$host_package" >> "$ROCSHMEM_PACKAGE_ROOT/packages/build-info.txt"
sha256sum "$host_package"
echo "Host package published: $host_package"
