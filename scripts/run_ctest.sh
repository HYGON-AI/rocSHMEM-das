#!/bin/bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT

set -euo pipefail

usage() {
    cat << EOF
Usage: $(basename "$0") <label> --host <host_spec>

Labels:
  quick|smoke    Quick smoke tests (IPC + GDA)
  standard       Standard tests (IPC + GDA)
  pr             pr tests (IPC + GDA + RO)
  comprehensive  Comprehensive tests (IPC + GDA + RO)
  nightly        nightly tests (IPC + GDA + RO)
  full           Full tests (IPC + GDA + RO)

Options:
  --host         MPI host specification for multi-node tests (required)
                 Example: --host host1,host2

Environment Variables:
  ROCSHMEM_TEST_DIR          ctest directory (Default: /home/rocshmem/rocshmem-install/bin/rocshmem)
  ROCSHMEM_TEST_LOG_DIR      Log directory for both ctest and per-test logs (Default: $(pwd)/test_logs)

Examples:
  $(basename "$0") quick --host host1,host2
  $(basename "$0") standard --host host1,host2
  ROCSHMEM_TEST_LOG_DIR=/tmp/rocshmem_logs $(basename "$0") quick --host host1,host2
  ROCSHMEM_IB_GID_INDEX=1 UCX_IB_GID_INDEX=1 $(basename "$0") quick --host host1,host2
EOF
    exit 1
}

if [[ $# -lt 1 ]]; then
    usage
fi

LABEL=$(echo "$1" | tr '[:upper:]' '[:lower:]')
shift

# Validate label early to prevent path traversal in LOG_PREFIX
case "$LABEL" in
    quick|smoke|standard|pr|comprehensive|nightly|full) ;;
    *) echo "Error: Unknown label '$LABEL'" >&2; usage ;;
esac

HOST_SPEC=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --host)
            [[ -z "${2:-}" ]] && { echo "Error: --host requires an argument" >&2; usage; }
            HOST_SPEC="$2"
            shift 2
            ;;
        *)
            echo "Error: Unknown option: $1" >&2
            usage
            ;;
    esac
done

if [[ -z "$HOST_SPEC" ]]; then
    echo "Error: --host is required (GDA tests need multi-node MPI)" >&2
    usage
fi

TEST_DIR=${ROCSHMEM_TEST_DIR:-/home/rocshmem/rocshmem-install/bin/rocshmem}
LOG_DIR=${ROCSHMEM_TEST_LOG_DIR:-$(pwd)/test_logs}
mkdir -p "$LOG_DIR"
# Convert to absolute path: ctest --test-dir changes CWD, so test_wrapper.sh
# would resolve relative LOG_DIR against the install dir, not the caller's CWD
LOG_DIR=$(cd "$LOG_DIR" && pwd)
export ROCSHMEM_TEST_LOG_DIR="$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_PREFIX="$LOG_DIR/test_${LABEL}_${TIMESTAMP}"

echo "=========================================="
echo "rocSHMEM CTest Runner"
echo "Label: $LABEL"
echo "Test Dir: $TEST_DIR"
echo "Host Spec: $HOST_SPEC"
echo "Log Dir: $LOG_DIR"
echo "Timestamp: $(date)"
echo "=========================================="

# MPI host spec (required for GDA multi-node tests)
HOST_PARAM="--host $HOST_SPEC"

# IB GID index: forward via -x for multi-node propagation; default to 1 if unset
# Validate to prevent MPI param injection (must be numeric)
ROCSHMEM_IB_GID="${ROCSHMEM_IB_GID_INDEX:-1}"
UCX_IB_GID="${UCX_IB_GID_INDEX:-1}"
[[ "$ROCSHMEM_IB_GID" =~ ^[0-9]+$ ]] || { echo "Error: ROCSHMEM_IB_GID_INDEX must be numeric, got '$ROCSHMEM_IB_GID'" >&2; exit 1; }
[[ "$UCX_IB_GID" =~ ^[0-9]+$ ]] || { echo "Error: UCX_IB_GID_INDEX must be numeric, got '$UCX_IB_GID'" >&2; exit 1; }
IB_GID_PARAMS="-x ROCSHMEM_IB_GID_INDEX=$ROCSHMEM_IB_GID -x UCX_IB_GID_INDEX=$UCX_IB_GID"

# Common MPI params shared by all backends
COMMON_MPI="--allow-run-as-root -x ROCSHMEM_HEAP_SIZE=10737418240 -x ROCSHMEM_MAX_NUM_CONTEXTS=40 --mca coll_hcoll_enable 0 -x UCX_WARN_UNUSED_ENV_VARS=n -x LD_LIBRARY_PATH -x PATH $IB_GID_PARAMS --map-by numa"

# Per-backend MPI params
MPI_PARAMS_IPC="$COMMON_MPI -x ROCSHMEM_BACKEND=ipc"
MPI_PARAMS_GDA="$COMMON_MPI -x ROCSHMEM_BACKEND=gda $HOST_PARAM"
MPI_PARAMS_RO="$COMMON_MPI -mca pml ucx -mca osc ucx -x ROCSHMEM_BACKEND=ro $HOST_PARAM"

# Label config: CTEST_LABEL, BACKENDS, EXTRA_ARGS array
parse_label() {
    CTEST_LABEL="$1"
    case "$1" in
        quick|smoke|standard)
            BACKENDS="ipc gda";     EXTRA_ARGS=(-R '_uuid$' -E '^unit') ;;
        pr|comprehensive|nightly|full)
            BACKENDS="ipc gda ro"; EXTRA_ARGS=(-E '^unit') ;;
        *)
            echo "Error: Unknown label '$1'" >&2; usage ;;
    esac
}

# Run one backend: run_backend <backend> <mpi_params> <ctest_label> <extra_args...>
run_backend() {
    local backend=$1 mpi_params=$2 ctest_label=$3
    shift 3
    local -a extra=("$@")
    local log="${LOG_PREFIX}_${backend}_${ctest_label}.log"

    echo ""
    echo "====== [${backend^^}] Running ${ctest_label^} Tests ======"
    echo "[CONFIG] ROCSHMEM_TEST_MPI_PARAMS=$mpi_params"
    echo "[COMMAND] ROCSHMEM_TEST_MPI_PARAMS=\"$mpi_params\" ctest -L $ctest_label ${extra[*]} --test-dir $TEST_DIR --output-on-failure"

    local result=0
    ROCSHMEM_BACKEND="$backend" \
    ROCSHMEM_TEST_MPI_PARAMS="$mpi_params" \
        ctest -L "$ctest_label" "${extra[@]}" --test-dir "$TEST_DIR" --output-on-failure 2>&1 \
        | tee "$log" || result=$?

    if [[ $result -eq 0 ]]; then
        echo "✅ ${backend^^} ${ctest_label^} PASSED"
    else
        echo "❌ ${backend^^} ${ctest_label^} FAILED (exit code: $result)"
    fi
    return $result
}

# Print summary: print_summary <title> <name1> <result1> [name2 result2 ...]
print_summary() {
    local title=$1 overall=0; shift
    echo ""
    echo "=========================================="
    echo "Test Summary ($title)"
    echo "=========================================="
    while [[ $# -gt 0 ]]; do
        printf "%-14s: %s\n" "$1" "$([ $2 -eq 0 ] && echo '✅ PASSED' || echo '❌ FAILED')"
        [[ $2 -ne 0 ]] && overall=1
        shift 2
    done
    echo "=========================================="
    return $overall
}

parse_label "$LABEL"

echo ""
echo "[INFO] Running ${CTEST_LABEL^^} suite (backends: ${BACKENDS// /, })"
echo ""

declare -A MPI_PARAMS=( [ipc]="$MPI_PARAMS_IPC" [gda]="$MPI_PARAMS_GDA" [ro]="$MPI_PARAMS_RO" )
declare -A RESULTS

for be in $BACKENDS; do
    # run test
    run_backend "$be" "${MPI_PARAMS[$be]}" "$CTEST_LABEL" "${EXTRA_ARGS[@]}" || RESULTS[$be]=$?
    [[ -z "${RESULTS[$be]:-}" ]] && RESULTS[$be]=0
done

# Build summary args
SUMMARY_ARGS=""
for be in $BACKENDS; do
    SUMMARY_ARGS+=" ${be^^} ${RESULTS[$be]}"
done

print_summary "$CTEST_LABEL" $SUMMARY_ARGS
exit $?
