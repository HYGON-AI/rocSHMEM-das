#!/bin/bash
###############################################################################
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################

# CTest wrapper for rocshmem functional tests
# This script handles conditional test skipping based on backend type
# and other runtime conditions, matching the behavior of driver.sh

# CTest SKIP return code
SKIP_CODE=125

prepare_mpi_launch_command() {
  local -n cmd=$1
  local -n num_ranks=$2

  # Parse original command to extract key info
  LAUNCHER="${cmd[0]}"

  # Find --host/-H in original command to detect multi-node config
  host_list=""
  node_count=0
  for ((i=0; i<${#cmd[@]}; i++)); do
    if [[ "${cmd[$i]}" == "-H" ]] || [[ "${cmd[$i]}" == "--host" ]]; then
      if [[ $((i+1)) -lt ${#cmd[@]} ]]; then
        host_list="${cmd[$((i+1))]}"
        IFS=',' read -ra hosts <<< "$host_list"
        node_count=${#hosts[@]}
      fi
      break
    elif [[ "${cmd[$i]}" == "--hostfile" ]]; then
      if [[ $((i+1)) -lt ${#cmd[@]} ]]; then
        hostfile="${cmd[$((i+1))]}"
        if [[ -f "$hostfile" ]]; then
          node_count=$(grep -v '^#' "$hostfile" | grep -v '^$' | wc -l)
          host_list="from_file:$hostfile"
        fi
      fi
      break
    fi
  done

  # If no custom MPI params and no multi-node host, skip reconstruction
  if [[ -z "${ROCSHMEM_TEST_MPI_PARAMS:-}" ]] && [[ $node_count -le 1 ]]; then
    return
  fi

  TEST_EXEC=""
  TEST_ARGS=()
  FOUND_APP=false
  UUID_ARG=""
  SKIP_NEXT_X=false

  # Parse original command to extract key info (drop CMake defaults; keep only UUID + test exec/args)
  for arg in "${cmd[@]:1}"; do
    if ! $FOUND_APP; then
      if [[ "$arg" == "-x" ]]; then
        SKIP_NEXT_X=true
        continue
      elif $SKIP_NEXT_X; then
        # Preserve ROCSHMEM_TEST_UUID env var (important!)
        if [[ "$arg" == ROCSHMEM_TEST_UUID=* ]]; then
          UUID_ARG="-x $arg"
        fi
        SKIP_NEXT_X=false
      elif [[ "$arg" == *"rocshmem"* ]] || [[ -x "$arg" ]]; then
        TEST_EXEC="$arg"
        FOUND_APP=true
      fi
    else
      TEST_ARGS+=("$arg")
    fi
  done

  # Parse custom MPI params (if any) for optimization
  mpi_params=()
  if [[ -n "${ROCSHMEM_TEST_MPI_PARAMS:-}" ]]; then
    read -ra mpi_params <<< "$ROCSHMEM_TEST_MPI_PARAMS"
  fi

  # If ROCSHMEM_TEST_MPI_PARAMS is set, re-scan it for host config (overrides cmd scan)
  if [[ -n "${ROCSHMEM_TEST_MPI_PARAMS:-}" ]]; then
    for ((i=0; i<${#mpi_params[@]}; i++)); do
      param="${mpi_params[$i]}"
      if [[ "$param" == "-H" ]] || [[ "$param" == "--host" ]]; then
        if [[ $((i+1)) -lt ${#mpi_params[@]} ]]; then
          host_list="${mpi_params[$((i+1))]}"
          IFS=',' read -ra hosts <<< "$host_list"
          node_count=${#hosts[@]}
        fi
        break
      elif [[ "$param" == "--hostfile" ]]; then
        if [[ $((i+1)) -lt ${#mpi_params[@]} ]]; then
          hostfile="${mpi_params[$((i+1))]}"
          if [[ -f "$hostfile" ]]; then
            node_count=$(grep -v '^#' "$hostfile" | grep -v '^$' | wc -l)
            host_list="from_file:$hostfile"
          fi
        fi
        break
      elif [[ "$param" == "-hosts" ]]; then
        if [[ $((i+1)) -lt ${#mpi_params[@]} ]]; then
          host_list="${mpi_params[$((i+1))]}"
          IFS=',' read -ra hosts <<< "$host_list"
          node_count=${#hosts[@]}
        fi
        break
      fi
    done
  fi

  # Auto-distribute ranks across nodes for multi-node runs
  if [[ $node_count -gt 1 ]] && [[ $num_ranks -ge $node_count ]]; then
    ranks_per_node=$((num_ranks / node_count))
    remainder=$((num_ranks % node_count))

    # Mode A: when using --hostfile
    if [[ "$host_list" == from_file:* ]]; then
      hostfile_path="${host_list#from_file:}"

      if [[ -f "$hostfile_path" ]] && [[ -w "$hostfile_path" ]]; then
        # Backup and modify hostfile to set ranks-per-node
        cp "$hostfile_path" "${hostfile_path}.bak"

        awk -v ppn="$ranks_per_node" '
          /^[[:space:]]*#/ || /^[[:space:]]*$/ { print; next }
          { print $1 " slots=" ppn }
        ' "$hostfile_path".bak > "$hostfile_path"

        # Auto-restore original hostfile on exit
        restore_hostfile() { cp "${hostfile_path}.bak" "$hostfile_path" 2>/dev/null; rm -f "${hostfile_path}.bak"; }
        trap restore_hostfile EXIT
      else
        # If hostfile not writable, use -npernode instead
        if [[ ${#mpi_params[@]} -gt 0 ]]; then
          mpi_params+=("-npernode" "$ranks_per_node")
        else
          cmd+=("-npernode" "$ranks_per_node")
        fi
      fi
    else
      # Mode B: rebuild host spec string when using -H/--host
      new_host_spec=""
      idx=0
      IFS=',' read -ra host_array <<< "$host_list"

      for host in "${host_array[@]}"; do
        node_ranks=$ranks_per_node
        # Spread remainder: first REMAINDER nodes get 1 extra rank
        if [[ $idx -lt $remainder ]]; then
          node_ranks=$((node_ranks + 1))
        fi

        if [[ "$host" == *":"* ]]; then
          hostname="${host%%:*}"
          new_host_spec="${new_host_spec}${hostname}:${node_ranks},"
        else
          new_host_spec="${new_host_spec}${host}:${node_ranks},"
        fi
        idx=$((idx + 1))
      done
      new_host_spec="${new_host_spec%,}"

      if [[ ${#mpi_params[@]} -gt 0 ]]; then
        # Update host spec in mpi_params array
        for ((i=0; i<${#mpi_params[@]}; i++)); do
          if [[ "${mpi_params[$i]}" == "-H" ]] || [[ "${mpi_params[$i]}" == "--host" ]]; then
            if [[ $((i+1)) -lt ${#mpi_params[@]} ]]; then
              mpi_params[$((i+1))]="$new_host_spec"
            fi
            break
          fi
        done
      else
        # Update host spec directly in cmd array
        for ((i=0; i<${#cmd[@]}; i++)); do
          if [[ "${cmd[$i]}" == "-H" ]] || [[ "${cmd[$i]}" == "--host" ]]; then
            if [[ $((i+1)) -lt ${#cmd[@]} ]]; then
              cmd[$((i+1))]="$new_host_spec"
            fi
            break
          fi
        done
      fi
    fi
  fi

  # Rebuild final command
  if [[ ${#mpi_params[@]} -gt 0 ]]; then
    # Custom MPI params provided: rebuild from scratch (drop CMake defaults, keep only UUID)
    cmd=("$LAUNCHER" -n "$num_ranks" "${mpi_params[@]}" $UUID_ARG "$TEST_EXEC" "${TEST_ARGS[@]}")
  fi
  # If no custom MPI params but multi-node was detected, cmd is already updated in-place above
}

# Extract test name (first argument)
TEST_NAME=$1
shift

# Detect launcher mode: MPI or SLR
# - If ROCSHMEM_SLR_NP is set, we're in SLR mode (direct execution)
# - Otherwise, we're in MPI mode (command contains mpirun/mpiexec)
if [[ -n "${ROCSHMEM_SLR_NP}" ]]; then
    LAUNCHER_MODE="SLR"
else
    LAUNCHER_MODE="MPI"
fi

# Find the test executable path from remaining arguments
# For MPI: The command is: mpirun/mpiexec [mpi_args] <executable> [test_args]
# For SLR: The command is: <executable> [test_args]
# We need to find the rocshmem test executable (not mpirun)
TEST_EXECUTABLE=""
SKIP_NEXT=false
for arg in "$@"; do
    # Skip argument values that follow MPI flags
    if $SKIP_NEXT; then
        SKIP_NEXT=false
        continue
    fi
    # MPI flags that take arguments - skip their values
    if [[ "$arg" == "-n" || "$arg" == "-np" || "$arg" == "-mca" || "$arg" == "-x" || "$arg" == "--timeout" || "$arg" == "--map-by" ]]; then
        SKIP_NEXT=true
        continue
    fi
    # Skip other MPI flags
    if [[ "$arg" == -* ]]; then
        continue
    fi
    # Look for rocshmem test executable (not mpirun/mpiexec)
    if [[ "$arg" == *"rocshmem"* ]] && [[ -x "$arg" ]]; then
        TEST_EXECUTABLE="$arg"
        break
    fi
done

# Get backend type from environment or rocshmem_info
# Strategy:
# 1. If ROCSHMEM_BACKEND is explicitly set, use that value
# 2. Otherwise, check rocshmem_info:
#    a. If exactly ONE backend is compiled, use that backend
#    b. If MULTIPLE backends are compiled, don't set backend (let rocshmem decide)
if [[ -n "$ROCSHMEM_BACKEND" ]]; then
    # User explicitly requested a specific backend
    BACKEND="$ROCSHMEM_BACKEND"
    echo "Using explicitly set backend: $BACKEND"
elif [[ -n "$ROCSHMEM_BACKEND_TYPE" ]]; then
    # Legacy compatibility
    BACKEND="$ROCSHMEM_BACKEND_TYPE"
    echo "Using ROCSHMEM_BACKEND_TYPE: $BACKEND"
elif [[ -n "$TEST" ]]; then
    # For compatibility with driver.sh TEST variable (e.g., "ro", "gda")
    BACKEND="$TEST"
    echo "Using TEST variable: $BACKEND"
else
    # Auto-detect from rocshmem_info
    # Find rocshmem_info (use test executable path)
    if [[ -n "$TEST_EXECUTABLE" ]]; then
        ROCSHMEM_INFO="$(dirname "$TEST_EXECUTABLE")/rocshmem_info"
        if [[ ! -x "$ROCSHMEM_INFO" ]]; then
            # Try alternate location (builddir case)
            ROCSHMEM_INFO="$(dirname "$TEST_EXECUTABLE")/../../tools/rocshmem_info"
        fi
        if [[ -x "$ROCSHMEM_INFO" ]]; then
            # Check which backends are compiled in
            # Format: # USE_RO                      : ON                                             #
            INFO_OUTPUT=$("$ROCSHMEM_INFO")
            USE_RO=$(echo "$INFO_OUTPUT" | grep "USE_RO" | awk -F ':' '{print $2}' | awk '{print $1}')
            USE_IPC=$(echo "$INFO_OUTPUT" | grep "USE_IPC" | awk -F ':' '{print $2}' | awk '{print $1}')
            USE_GDA=$(echo "$INFO_OUTPUT" | grep "USE_GDA" | awk -F ':' '{print $2}' | awk '{print $1}')

            # Count how many backends are enabled
            BACKEND_COUNT=0
            AVAILABLE_BACKEND=""
            if [[ "$USE_RO" == "ON" ]]; then
                BACKEND_COUNT=$((BACKEND_COUNT + 1))
                AVAILABLE_BACKEND="ro"
            fi
            if [[ "$USE_IPC" == "ON" ]]; then
                BACKEND_COUNT=$((BACKEND_COUNT + 1))
                AVAILABLE_BACKEND="ipc"
            fi
            if [[ "$USE_GDA" == "ON" ]]; then
                BACKEND_COUNT=$((BACKEND_COUNT + 1))
                AVAILABLE_BACKEND="gda"
            fi

            if [[ $BACKEND_COUNT -eq 1 ]]; then
                # Exactly one backend compiled - use it and apply skip logic
                BACKEND="$AVAILABLE_BACKEND"
                echo "Single backend detected: $BACKEND (will apply backend-specific skip logic)"
            elif [[ $BACKEND_COUNT -gt 1 ]]; then
                # Multiple backends compiled - let rocshmem decide, don't skip tests
                BACKEND="multi"
                echo "Multiple backends detected (RO=$USE_RO, IPC=$USE_IPC, GDA=$USE_GDA) - letting rocshmem choose, no skip logic"
            else
                # No backends found (shouldn't happen)
                BACKEND="unknown"
                echo "Warning: No backends detected in rocshmem_info"
            fi
        else
            BACKEND="unknown"
            echo "Warning: rocshmem_info not found at $ROCSHMEM_INFO"
        fi
    else
        BACKEND="unknown"
        echo "Warning: Could not find test executable"
    fi
fi

echo "Test: $TEST_NAME (Launcher: $LAUNCHER_MODE, Backend: $BACKEND, Executable: ${TEST_EXECUTABLE:-<not found>})"

# Apply skip conditions based on backend type and known issues
# These match the skip logic from driver.sh
# Note: Skip logic only applies when backend is known (not "multi" or "unknown")

# Host non-MPI IPC tests require IPC backend
if [[ "$BACKEND" == "ro" || "$BACKEND" == "gda" ]]; then
    case "$TEST_NAME" in
        host_putmem_*|host_getmem_*|host_amo_*|host_ctx_*|host_int_*)
            echo "Skip: $TEST_NAME (host non-MPI IPC tests require IPC backend)"
            exit $SKIP_CODE
            ;;
    esac
fi

# AIROCSHMEM-120: RO get tests abort
if [[ "$BACKEND" == "ro" ]]; then
    case "$TEST_NAME" in
        get_*|getnbi_*|defaultctxget_*|defaultctxgetnbi_*|teamctxget_*|teamctxgetnbi_*|wgget_*|wggetnbi_*|waveget_*|wavegetnbi_*)
            echo "Skip: $TEST_NAME (AIROCSHMEM-120: RO get tests abort)"
            exit $SKIP_CODE
            ;;
    esac
fi

# AIROCSHMEM-162: GDA _g not implemented
if [[ "$BACKEND" == "gda" ]]; then
    case "$TEST_NAME" in
        g_*|defaultctxg_*|flood_g_*)
            echo "Skip: $TEST_NAME (AIROCSHMEM-162: GDA _g not implemented)"
            exit $SKIP_CODE
            ;;
    esac
fi

# AIROCSHMEM-211: RO AMO operations abort
if [[ "$BACKEND" == "ro" ]]; then
    case "$TEST_NAME" in
        amo_add_*|amo_fadd_*|amo_inc_*|amo_finc_*)
            echo "Skip: $TEST_NAME (AIROCSHMEM-211: RO amo abort)"
            exit $SKIP_CODE
            ;;
    esac
fi

# AIROCSHMEM-217: RO putmem_signal_on_stream sometimes abort
if [[ "$BACKEND" == "ro" ]]; then
    case "$TEST_NAME" in
        putmem_signal_on_stream_*)
            echo "Skip: $TEST_NAME (AIROCSHMEM-217: RO sometimes abort)"
            exit $SKIP_CODE
            ;;
    esac
fi

# AIROCSHMEM-324: RO flood tests fail in UCX
if [[ "$BACKEND" == "ro" ]]; then
    case "$TEST_NAME" in
        flood_*)
            echo "Skip: $TEST_NAME (AIROCSHMEM-324: RO flood tests fail in UCX)"
            exit $SKIP_CODE
            ;;
    esac
fi

# AIROCSHMEM-418: fence tests not supported on RO
if [[ "$BACKEND" == "ro" ]]; then
    case "$TEST_NAME" in
        fence_*)
            echo "Skip: $TEST_NAME (AIROCSHMEM-418: fence tests not supported on RO)"
            exit $SKIP_CODE
            ;;
    esac
fi

# Check GPU availability
if command -v amd-smi >/dev/null && amd-smi version 2>&1 >/dev/null; then
    NUM_GPUS=$(amd-smi list | grep GPU | wc -l)
elif command -v rocm-smi >/dev/null && rocm-smi --version 2>&1 >/dev/null; then
    NUM_GPUS=$(rocm-smi --showserial | grep GPU | wc -l)
else
    NUM_GPUS=0
fi
NUM_GPUS=$((NUM_GPUS > 0 ? NUM_GPUS : 8))

# Extract number of ranks from test name (format: testname_n<ranks>_w<wg>_z<threads>)
# Or from ROCSHMEM_SLR_NP if in SLR mode
if [[ "$LAUNCHER_MODE" == "SLR" ]]; then
    NUM_RANKS=${ROCSHMEM_SLR_NP}
elif [[ "$TEST_NAME" =~ _n([0-9]+)_ ]]; then
    NUM_RANKS=${BASH_REMATCH[1]}
fi

# Skip if not enough GPUs
if [[ -n "$NUM_RANKS" ]]; then
    # For SLR mode, always check GPU availability (no hostfile support)
    if [[ "$LAUNCHER_MODE" == "SLR" ]] && [[ $NUM_GPUS -lt $NUM_RANKS ]]; then
        echo "Skip: $TEST_NAME (SLR requires $NUM_RANKS GPUs, only $NUM_GPUS available)"
        exit $SKIP_CODE
    fi
    # For MPI mode, skip only if no hostfile and insufficient GPUs
    if [[ "$LAUNCHER_MODE" == "MPI" ]] && [[ -z "$HOSTFILE" ]] && [[ $NUM_GPUS -lt $NUM_RANKS ]]; then
        echo "Skip: $TEST_NAME ($NUM_RANKS ranks required but only $NUM_GPUS GPUs available)"
        exit $SKIP_CODE
    fi
fi

# Setup log directory and file (matching driver.sh behavior)
# Use environment variable LOG_DIR if set, otherwise use current directory
LOG_DIR=${ROCSHMEM_TEST_LOG_DIR:-${LOG_DIR:-.}}
mkdir -p "$LOG_DIR/$BACKEND"

LOG_FILE="$LOG_DIR/$BACKEND/$TEST_NAME.log"

BUILD_COMMAND=("$@")

if [[ "$LAUNCHER_MODE" == "MPI" ]]; then
  prepare_mpi_launch_command BUILD_COMMAND NUM_RANKS
fi

echo "# ${BUILD_COMMAND[*]}" > "$LOG_FILE"

"${BUILD_COMMAND[@]}" >> "$LOG_FILE" 2>&1
TEST_EXIT_CODE=$?

# If test failed, show the log content (for CTest output)
if [ $TEST_EXIT_CODE -ne 0 ]; then
    echo "Test failed - see log: $LOG_FILE"
    cat "$LOG_FILE"
fi

exit $TEST_EXIT_CODE