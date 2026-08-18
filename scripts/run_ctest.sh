#!/bin/bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT

set -eo pipefail

usage() {
    cat << EOF
Usage: $(basename "$0") <label> --host <host_spec>

Labels:
  quick|smoke    Quick smoke tests (IPC + GDA)
  standard       Standard tests (IPC + GDA + RO)
  full           Full tests (IPC + GDA + RO)

Options:
  --host         MPI host specification for multi-node tests (required)
                 Example: --host bw1,bw2

Environment Variables:
  ROCSHMEM_TEST_DIR          ctest directory (Default: /home/rocshmem/rocshmem-install/bin/rocshmem)
  ROCSHMEM_TEST_LOG_DIR      Log directory for both ctest and per-test logs (Default: $(pwd)/test_logs)

Examples:
  $(basename "$0") quick --host bw1,bw2
  $(basename "$0") standard --host bw1,bw2
  ROCSHMEM_TEST_LOG_DIR=/tmp/rocshmem_logs $(basename "$0") quick --host bw1,bw2
  ROCSHMEM_IB_GID_INDEX=1 UCX_IB_GID_INDEX=1 $(basename "$0") quick --host bw1,bw2
EOF
    exit 1
}

if [[ $# -lt 1 ]]; then
    usage
fi

LABEL=$(echo "$1" | tr '[:upper:]' '[:lower:]')
shift

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
export ROCSHMEM_TEST_LOG_DIR="$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_PREFIX="$LOG_DIR/test_${LABEL}_${TIMESTAMP}"

echo "=========================================="
echo "rocSHMEM CTest Runner"
echo "Label: $LABEL"
echo "Test Dir: $TEST_DIR"
echo "Host Spec: $HOST_SPEC"
echo "Log Prefix: $LOG_PREFIX"
echo "Timestamp: $(date)"
echo "=========================================="

# Generic runner: execute ctest suite for a given backend
# Usage: run_backend <backend> <tier> <mpi_params> [ctest_extra_args...]
run_backend() {
    local backend=$1 tier=$2 mpi_params=$3
    shift 3
    local label=$([[ "$tier" == "quick" ]] && echo quick || echo standard)
    local log="${LOG_PREFIX}_${backend}_${tier}.log"

    echo ""
    echo "====== [${backend^^}] Running ${tier^} Tests ======"
    echo "[CONFIG] ROCSHMEM_TEST_MPI_PARAMS=$mpi_params"
    echo "[COMMAND] ROCSHMEM_TEST_MPI_PARAMS=\"$mpi_params\" ctest -L $label $* --test-dir $TEST_DIR --output-on-failure"

    # pipefail makes the pipeline return ctest's exit code; || result=$? suppresses set -e for result printing
    local result=0
    ROCSHMEM_BACKEND="$backend" \
    ROCSHMEM_TEST_MPI_PARAMS="$mpi_params" \
        ctest -L "$label" "$@" --test-dir "$TEST_DIR" --output-on-failure 2>&1 \
        | tee "$log" || result=$?

    if [[ $result -eq 0 ]]; then
        echo "✅ ${backend^^} ${tier^} Tests PASSED"
    else
        echo "❌ ${backend^^} ${tier^} Tests FAILED (exit code: $result)"
    fi
    return $result
}

# MPI host spec (required for GDA multi-node tests)
HOST_PARAM="--host $HOST_SPEC"

# IB GID index: forward via -x for multi-node propagation; default to 1 if unset
ROCSHMEM_IB_GID="${ROCSHMEM_IB_GID_INDEX:-1}"
UCX_IB_GID="${UCX_IB_GID_INDEX:-1}"
IB_GID_PARAMS="-x ROCSHMEM_IB_GID_INDEX=$ROCSHMEM_IB_GID -x UCX_IB_GID_INDEX=$UCX_IB_GID"

# Common MPI params shared by all backends
COMMON_MPI="--allow-run-as-root -x ROCSHMEM_HEAP_SIZE=21474836480 --mca coll_hcoll_enable 0 -x UCX_WARN_UNUSED_ENV_VARS=n -x LD_LIBRARY_PATH -x PATH $IB_GID_PARAMS --map-by numa"

# Per-backend MPI params
MPI_PARAMS_IPC="$COMMON_MPI -x ROCSHMEM_BACKEND=ipc"
MPI_PARAMS_GDA="$COMMON_MPI -x ROCSHMEM_BACKEND=gda $HOST_PARAM"
MPI_PARAMS_RO="$COMMON_MPI -mca pml ucx -mca osc ucx -x ROCSHMEM_BACKEND=ro $HOST_PARAM"

IPC_RESULT=0
GDA_RESULT=0
RO_RESULT=0

case "$LABEL" in
    quick|smoke)
        echo ""
        echo "[INFO] Running QUICK/SMOKE test suite (IPC + GDA)"
        echo ""

        run_backend ipc quick "$MPI_PARAMS_IPC" -R "_uuid$" -E "^unit" || IPC_RESULT=$?
        run_backend gda quick "$MPI_PARAMS_GDA" -R "_uuid$" -E "^unit" || GDA_RESULT=$?

        echo ""
        echo "=========================================="
        echo "Test Summary (Quick/Smoke)"
        echo "=========================================="
        echo "IPC Quick : $([ $IPC_RESULT -eq 0 ] && echo '✅ PASSED' || echo '❌ FAILED')"
        echo "GDA Quick : $([ $GDA_RESULT -eq 0 ] && echo '✅ PASSED' || echo '❌ FAILED')"
        echo "=========================================="

        exit $([ $IPC_RESULT -ne 0 ] || [ $GDA_RESULT -ne 0 ])
        ;;

    standard)
        echo ""
        echo "[INFO] Running STANDARD test suite (IPC + GDA + RO) with -L standard"
        echo ""

        run_backend ipc standard "$MPI_PARAMS_IPC" -E "^unit" || IPC_RESULT=$?
        run_backend gda standard "$MPI_PARAMS_GDA" -E "^unit" || GDA_RESULT=$?
        run_backend ro  standard "$MPI_PARAMS_RO" -E "^unit" || RO_RESULT=$?

        echo ""
        echo "=========================================="
        echo "Test Summary (Standard Full)"
        echo "=========================================="
        echo "IPC Standard   : $([ $IPC_RESULT -eq 0 ] && echo '✅ PASSED' || echo '❌ FAILED')"
        echo "GDA Standard   : $([ $GDA_RESULT -eq 0 ] && echo '✅ PASSED' || echo '❌ FAILED')"
        echo "RO Standard    : $([ $RO_RESULT -eq 0 ] && echo '✅ PASSED' || echo '❌ FAILED')"
        echo "=========================================="

        exit $([ $IPC_RESULT -ne 0 ] || [ $GDA_RESULT -ne 0 ] || [ $RO_RESULT -ne 0 ])
        ;;
    full)
        echo ""
        echo "[INFO] Running FULL test suite (IPC + GDA + RO) with -L full"
        echo ""

        run_backend ipc full "$MPI_PARAMS_IPC" -E "^unit" || IPC_RESULT=$?
        run_backend gda full "$MPI_PARAMS_GDA" -E "^unit" || GDA_RESULT=$?
        run_backend ro  full "$MPI_PARAMS_RO" -E "^unit" || RO_RESULT=$?    

        echo ""
        echo "=========================================="
        echo "Test Summary (Full)"
        echo "=========================================="
        echo "IPC Full   : $([ $IPC_RESULT -eq 0 ] && echo '✅ PASSED' || echo '❌ FAILED')"
        echo "GDA Full   : $([ $GDA_RESULT -eq 0 ] && echo '✅ PASSED' || echo '❌ FAILED')"
        echo "RO Full    : $([ $RO_RESULT -eq 0 ] && echo '✅ PASSED' || echo '❌ FAILED')"
        echo "=========================================="

        exit $([ $IPC_RESULT -ne 0 ] || [ $GDA_RESULT -ne 0 ] || [ $RO_RESULT -ne 0 ])
        ;;

    *)
        echo "Error: Unknown label '$LABEL'"
        echo ""
        usage
        ;;
esac
