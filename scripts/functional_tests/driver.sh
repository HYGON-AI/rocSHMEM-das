###############################################################################
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to
# deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
###############################################################################

#!/bin/bash
if true || tty -s; then
  PRETTY_FAILED="\033[1;31mFAILED\033[0m"
  PRETTY_PASSED="\033[1;32mPASSED\033[0m"
else
  PRETTY_FAILED="FAILED"
  PRETTY_PASSED="PASSED"
fi

# This names/values should match the TestType enum in rocSHMEM/tests/functional_tests/tester.hpp
# Self-developed test cases start from 200 to avoid community conflicts.
declare -A TEST_NUMBERS=(
  ["get"]="0"
  ["getnbi"]="1"
  ["put"]="2"
  ["putnbi"]="3"
  ["amo_fadd"]="4"
  ["amo_finc"]="5"
  ["amo_fetch"]="6"
  ["amo_fcswap"]="7"
  ["amo_add"]="8"
  ["amo_inc"]="9"
  ["amo_cswap"]="10"
  ["init"]="11"
  ["pingpong"]="12"
  ["randomaccess"]="13"
  ["barrierall"]="14"
  ["syncall"]="15"
  ["teamsync"]="16"
  ["collect"]="17"
  ["fcollect"]="18"
  ["alltoall"]="19"
  ["alltoallv"]="20"
  ["shmemptr"]="21"
  ["p"]="22"
  ["g"]="23"
  ["wgget"]="24"
  ["wggetnbi"]="25"
  ["wgput"]="26"
  ["wgputnbi"]="27"
  ["waveget"]="28"
  ["wavegetnbi"]="29"
  ["waveput"]="30"
  ["waveputnbi"]="31"
  ["teambroadcast"]="32"
  ["teamreduction"]="33"
  ["teamctxget"]="34"
  ["teamctxgetnbi"]="35"
  ["teamctxput"]="36"
  ["teamctxputnbi"]="37"
  ["teamctxinfra"]="38"
  ["putnbimr"]="39"
  ["amo_set"]="40"
  ["amo_swap"]="41"
  ["amo_fetchand"]="42"
  ["amo_fetchor"]="43"
  ["amo_fetchxor"]="44"
  ["amo_and"]="45"
  ["amo_or"]="46"
  ["amo_xor"]="47"
  ["pingall"]="48"
  ["putsignal"]="49"
  ["wgputsignal"]="50"
  ["waveputsignal"]="51"
  ["putsignalnbi"]="52"
  ["wgputsignalnbi"]="53"
  ["waveputsignalnbi"]="54"
  ["signalfetch"]="55"
  ["wgsignalfetch"]="56"
  ["wavesignalfetch"]="57"
  ["teamwgbarrier"]="58"
  ["defaultctxget"]="59"
  ["defaultctxgetnbi"]="60"
  ["defaultctxput"]="61"
  ["defaultctxputnbi"]="62"
  ["defaultctxp"]="63"
  ["defaultctxg"]="64"
  ["wavebarrierall"]="65"
  ["wgbarrierall"]="66"
  ["wavesyncall"]="67"
  ["wgsyncall"]="68"
  ["teambarrier"]="69"
  ["teamwavebarrier"]="70"
  ["teamwavesync"]="71"
  ["teamwgsync"]="72"
  ["teamctxsingleinfra"]="73"
  ["teamctxblockinfra"]="74"
  ["teamctxoddeveninfra"]="75"
  ["alltoallmem_on_stream"]="76"
  ["barrier_all_on_stream"]="77"
  ["broadcastmem_on_stream"]="78"
  ["getmem_on_stream"]="79"
  ["putmem_on_stream"]="80"
  ["putmem_signal_on_stream"]="81"
  ["signal_wait_until_on_stream"]="82"
  ["flood_put"]="83"
  ["flood_putnbi"]="84"
  ["flood_p"]="85"
  ["flood_get"]="86"
  ["flood_getnbi"]="87"
  ["flood_g"]="88"
  ["hipmodule_init"]="89"
  ["flood_add"]="90"
  ["flood_fadd"]="91"
  ["flood_waitadd"]="92"
  ["device_bitcode"]="93"
  ["library_info"]="94"
  ["teamctxsharedinfra"]="95"
  ["quiet_on_stream"]="96"
  ["sync_all_on_stream"]="97"
  ["teamctxsubsetparentinfra"]="98"

  ["defaultctx_waveputnbi_dp"]="200"
  ["defaultctx_amo_add_dp"]="201"
  ["waveputnbi_dp"]="202"
  ["amo_add_dp"]="203"
)

ExecTest() {
  TEST_NAME=$1
  NUM_RANKS=$2
  NUM_WG=$3
  NUM_THREADS=$4
  MAX_MSG_SIZE=$5
  IS_RETRY=${6:-0}  # Optional 6th parameter to indicate if this is a retry

  if [[ "" == "$NOTIMEOUT" ]]; then
    TIMEOUT=$((5 * 60)) # Timeout in seconds
  fi
  HEAP_SIZE=$((6*1024*1024*1024))

  if command -v amd-smi >/dev/null && amd-smi version 2>&1 >/dev/null
  then
    NUM_GPUS=${NUM_GPUS:-$(amd-smi list | grep GPU | wc -l)}
  elif command -v rocm-smi >/dev/null && rocm-smi --version 2>&1 >/dev/null
  then
    NUM_GPUS=${NUM_GPUS:-$(rocm-smi --showserial | grep GPU | wc -l)}
  fi
  NUM_GPUS=${NUM_GPUS:-0}
  NUM_GPUS=$(($NUM_GPUS > 0? $NUM_GPUS: 8))

  TEST_NUM=${TEST_NUMBERS[$TEST_NAME]}

  if [[ "" == "$TEST_NUM" ]]
  then
    echo "Test $TEST_NAME does not exist" >&2
    DRIVER_RETURN_STATUS=1
    return
  fi

  if [[ "" == "$ROCSHMEM_MAX_NUM_CONTEXTS" ]]
  then
    ROCSHMEM_MAX_NUM_CONTEXTS=$NUM_WG
  fi

  # MPI Parameters
  LAUNCHER=mpirun

  if [[ "" != "$ROCSHMEM_TEST_USE_DEFAULT_STREAM" ]]
  then
    OPTIONS+=" -x ROCSHMEM_TEST_USE_DEFAULT_STREAM=$ROCSHMEM_TEST_USE_DEFAULT_STREAM"
  fi

  if [[ "" != "$HOSTFILE" ]]
  then
    OPTIONS+=" --hostfile $HOSTFILE"
  fi

  # Build command as an array to avoid command injection with eval
  local -a cmd
  cmd=( "$LAUNCHER"
        -n "$NUM_RANKS"
        -mca pml "${OMPI_MCA_pml:-ucx}"
        -mca osc "${OMPI_MCA_osc:-ucx}"
        -x "ROCSHMEM_MAX_NUM_CONTEXTS=$ROCSHMEM_MAX_NUM_CONTEXTS"
        -x "UCX_ROCM_IPC_SIGPOOL_MAX_ELEMS=16384"
        -x "ROCSHMEM_HEAP_SIZE=$HEAP_SIZE"
        ${ROCSHMEM_TEST_USE_DEFAULT_STREAM:+-x "ROCSHMEM_TEST_USE_DEFAULT_STREAM=$ROCSHMEM_TEST_USE_DEFAULT_STREAM"}
        ${ROCSHMEM_TEST_UUID:+-x "ROCSHMEM_TEST_UUID=$ROCSHMEM_TEST_UUID"}
        ${TIMEOUT:+--timeout "$TIMEOUT"}
        ${HOSTFILE:+--hostfile "$HOSTFILE"}
        --map-by numa
      )

  # Use MPI Parameters when provided via environment variable (overrides all default launcher params)
  if [[ -n "${ROCSHMEM_TEST_MPI_PARAMS:-}" ]]; then
    cmd=( "$LAUNCHER" -n "$NUM_RANKS" $ROCSHMEM_TEST_MPI_PARAMS )
  fi

  # Construct Test Command
  TEST_LOG_NAME="$TEST_NAME"_n"$NUM_RANKS"_w"$NUM_WG"_z"$NUM_THREADS"
  cmd+=( "$APP" -a "$TEST_NUM" -w "$NUM_WG" -z "$NUM_THREADS" ${NOVERIF:+-noverif} -localbuftype ${LOCALBUFTYPE:-heap} )
  if [[ "" != "$MAX_MSG_SIZE" ]]
  then
    # Check if in volume mode
    if [[ $MAX_MSG_SIZE == v* ]]; then
      cmd+=( -v "${MAX_MSG_SIZE#v}" )
    else
      cmd+=( -s "$MAX_MSG_SIZE" )
    fi
    TEST_LOG_NAME+=_"$MAX_MSG_SIZE"B
  fi
  # Create a human-readable representation of the command for logging purposes
  CMD="${cmd[@]}"

  # Determine log file name based on whether this is a retry
  if [ $IS_RETRY -eq 1 ]; then
    LOG_FILE="$LOG_DIR/$TEST_LOG_NAME.retry.log"
    echo "Retry:  $TEST_LOG_NAME"
  else
    LOG_FILE="$LOG_DIR/$TEST_LOG_NAME.log"
    echo "Test:   $TEST_LOG_NAME"
  fi

  # Run Test
  if [ $NUM_GPUS -ge $NUM_RANKS ] || [[ "" != "$HOSTFILE" ]] || [[ "${cmd[*]}" == *-H* ]] || [[ "${cmd[*]}" == *--host* ]]; then
    echo "# $CMD >> $LOG_FILE" >"$LOG_FILE"
    "${cmd[@]}" >>"$LOG_FILE" 2>&1
  else
    echo "Skip:   $TEST_LOG_NAME ($NUM_RANKS greater than $NUM_GPUS)"
  fi

  # Validate Test
  if [ $? -ne 0 ]
  then
    echo -e "$PRETTY_FAILED: $TEST_LOG_NAME" >&2
    cat "$LOG_FILE"
    DRIVER_RETURN_STATUS=1
    if [ $IS_RETRY -eq 0 ]; then
      # Track failed tests with their parameters for potential retry
      # Capture environment/config state to ensure retry runs under same conditions
      FAILED_LIST="$FAILED_LIST $TEST_LOG_NAME"
      FAILED_TESTS+=("$TEST_NAME|$NUM_RANKS|$NUM_WG|$NUM_THREADS|$MAX_MSG_SIZE|${ROCSHMEM_TEST_USE_DEFAULT_STREAM:-}|${ROCSHMEM_MAX_NUM_CONTEXTS:-}|${NOTIMEOUT:-}|${NOVERIF:-}")
    else
      # Track tests that failed even after retry
      RETRY_FAILED_LIST="$RETRY_FAILED_LIST $TEST_LOG_NAME"
    fi
  else
    # If this was a retry and it passed, remove from failed list
    if [ $IS_RETRY -eq 1 ]; then
      echo -e "$PRETTY_PASSED: $TEST_LOG_NAME (passed on retry)"
      RETRY_PASSED_LIST="$RETRY_PASSED_LIST $TEST_LOG_NAME"
    fi
  fi

  unset ROCSHMEM_MAX_NUM_CONTEXTS
}

TestRMAPut() {
  ##############################################################################
  #       | Name             | Ranks | Workgroups | Threads | Max Message Size #
  ##############################################################################
  ExecTest  "put"              2       1            1         1048576
  ExecTest  "put"              2       1            1024      512
  ExecTest  "put"              2       8            1         1048576
  ExecTest  "put"              2       16           128       8
  ExecTest  "put"              2       32           256       512
  ExecTest  "put"              2       64           1024      8

  ExecTest  "defaultctxput"    2       4            128       1024
  ExecTest  "teamctxput"       2       4            128       1024
  ExecTest  "teamctxput"       2       16           256       1024

  ExecTest  "defaultctxp"      2       4            128       1024
  ExecTest  "defaultctxg"      2       4            128       1024

  ExecTest  "wgput"            2       1            64        1048576
  ExecTest  "wgput"            2       2            64        1048576
  ExecTest  "wgput"            2       16           64        8

  ExecTest  "waveput"          2       1            64        1048576
  ExecTest  "waveput"          2       2            64        1048576
  ExecTest  "waveput"          2       2            128       1048576
  ExecTest  "waveput"          2       16           128       8

  ################################ Non-Blocking ################################
  ExecTest  "p"                2       1            1         128
  ExecTest  "p"                2       1            1024      2
  ExecTest  "p"                2       8            1         32
  ExecTest  "p"                2       16           128       4

  ExecTest  "putnbi"           2       1            1         1048576
  ExecTest  "putnbi"           2       1            1024      512
  ExecTest  "putnbi"           2       8            1         1048576
  ExecTest  "putnbi"           2       16           128       8
  ExecTest  "putnbi"           2       32           256       512
  ExecTest  "putnbi"           2       64           1024      8

  ExecTest  "putnbimr"         2       16           128       1048576

  ExecTest  "defaultctxputnbi" 2       4            128       1024
  ExecTest  "teamctxputnbi"    2       4            128       1024
  ExecTest  "teamctxputnbi"    2       16           256       1024

  ExecTest  "wgputnbi"         2       1            64        1048576
  ExecTest  "wgputnbi"         2       2            64        1048576
  ExecTest  "wgputnbi"         2       16           64        8

  ExecTest  "waveputnbi"       2       1            64        1048576
  ExecTest  "waveputnbi"       2       2            64        1048576
  ExecTest  "waveputnbi"       2       2            128       1048576
  ExecTest  "waveputnbi"       2       16           128       8

  ################################ User Buffer Tests ################################
  if [[ $TEST != gda* ]]; then # AIROCSHMEM-383
    export LOCALBUFTYPE=host
    ExecTest  "putnbi"           2       32           128       512
    unset LOCALBUFTYPE

    export LOCALBUFTYPE=device
    ExecTest  "putnbi"           2       32           128       512
    unset LOCALBUFTYPE

    export LOCALBUFTYPE=fine
    ExecTest  "putnbi"           2       32           128       512
    unset LOCALBUFTYPE

    export LOCALBUFTYPE=uncached
    ExecTest  "putnbi"           2       32           128       512
    unset LOCALBUFTYPE

    export LOCALBUFTYPE=managed
    ExecTest  "putnbi"           2       32           128       512
    unset LOCALBUFTYPE
  fi
}

TestRMAGet() {
  ##############################################################################
  #       | Name             | Ranks | Workgroups | Threads | Max Message Size #
  ##############################################################################
  if [[ $TEST != ro* ]]; then #AIROCSHMEM-120
  ExecTest  "get"              2       1            1         1048576
  ExecTest  "get"              2       1            1024      512
  ExecTest  "get"              2       8            1         1048576
  ExecTest  "get"              2       16           128       8
  ExecTest  "get"              2       32           256       512
  ExecTest  "get"              2       64           1024      8

  ExecTest  "defaultctxget"    2       4            128       1024
  ExecTest  "teamctxget"       2       4            128       1024
  ExecTest  "teamctxget"       2       16           256       1024

  ExecTest  "wgget"            2       1            64        1048576
  ExecTest  "wgget"            2       2            64        1048576
  ExecTest  "wgget"            2       16           64        8

  ExecTest  "waveget"          2       1            64        1048576
  ExecTest  "waveget"          2       2            64        1048576
  ExecTest  "waveget"          2       2            128       1048576
  ExecTest  "waveget"          2       16           128       8

  if [[ $TEST != gda* ]]; then #AIROCSHMEM-162
  ExecTest  "g"                2       1            1         128
  ExecTest  "g"                2       1            1024      1
  ExecTest  "g"                2       8            1         32
  ExecTest  "g"                2       16           128       4
  else echo "Skip:   g_* (AIROCSHMEM-162: GDA _g not implemented)"; fi

  ################################ Non-Blocking ################################
  ExecTest  "getnbi"           2       1            1         1048576
  ExecTest  "getnbi"           2       1            1024      512
  ExecTest  "getnbi"           2       8            1         1048576
  ExecTest  "getnbi"           2       16           128       8
  ExecTest  "getnbi"           2       32           256       512
  ExecTest  "getnbi"           2       64           1024      8

  ExecTest  "defaultctxgetnbi" 2       4            128       1024
  ExecTest  "teamctxgetnbi"    2       4            128       1024
  ExecTest  "teamctxgetnbi"    2       16           256       1024

  ExecTest  "wggetnbi"         2       1            64        1048576
  ExecTest  "wggetnbi"         2       2            64        1048576
  ExecTest  "wggetnbi"         2       16           64        8

  ExecTest  "wavegetnbi"       2       1            64        1048576
  ExecTest  "wavegetnbi"       2       2            64        1048576
  ExecTest  "wavegetnbi"       2       2            128       1048576
  ExecTest  "wavegetnbi"       2       16           128       8
  else echo "Skip:   get_* (AIROCSHMEM-120: RO get tests abort)"; fi

  ################################ User Buffer Tests ################################
  # AIROCSHMEM-383 for GDA
  # AIROCSHMEM-120 for RO
  if [[ $TEST != gda* && $TEST != ro* ]]; then
    export LOCALBUFTYPE=host
    ExecTest  "getnbi"           2       32           128       512
    unset LOCALBUFTYPE

    export LOCALBUFTYPE=device
    ExecTest  "getnbi"           2       32           128       512
    unset LOCALBUFTYPE

    export LOCALBUFTYPE=fine
    ExecTest  "getnbi"           2       32           128       512
    unset LOCALBUFTYPE

    export LOCALBUFTYPE=uncached
    ExecTest  "getnbi"           2       32           128       512
    unset LOCALBUFTYPE

    export LOCALBUFTYPE=managed
    ExecTest  "getnbi"           2       32           128       512
    unset LOCALBUFTYPE
  fi
}

TestRMA() {
  TestRMAPut
  TestRMAGet
}

TestAMO() {
  ##############################################################################
  #       | Name             | Ranks | Workgroups | Threads | Max Message Size #
  ##############################################################################
  if [[ $TEST != ro* ]]; then #AIROCSHMEM-211
  ExecTest  "amo_add"          2       1            1
  ExecTest  "amo_add"          2       1            1024
  ExecTest  "amo_add"          2       8            1
  ExecTest  "amo_add"          2       32           128

  ExecTest  "amo_fadd"         2       1            1
  ExecTest  "amo_fadd"         2       1            1024
  ExecTest  "amo_fadd"         2       8            1
  ExecTest  "amo_fadd"         2       32           128

  ExecTest  "amo_inc"          2       1            1
  ExecTest  "amo_inc"          2       1            1024
  ExecTest  "amo_inc"          2       8            1
  ExecTest  "amo_inc"          2       32           128

  ExecTest  "amo_finc"         2       1            1
  ExecTest  "amo_finc"         2       1            1024
  ExecTest  "amo_finc"         2       8            1
  ExecTest  "amo_finc"         2       32           128
  else echo "Skip:   amo_add* (AIROCSHMEM-211: ro amo abort)"; fi

  ExecTest  "amo_set"          2       1            1
  ExecTest  "amo_set"          2       8            1
  ExecTest  "amo_set"          2       32           1

  ExecTest  "amo_fetch"        2       1            1
  ExecTest  "amo_fetch"        2       1            1024
  ExecTest  "amo_fetch"        2       8            1
  ExecTest  "amo_fetch"        2       32           128

  ExecTest  "amo_fcswap"       2       1            1
  ExecTest  "amo_fcswap"       2       32           1
  ExecTest  "amo_fcswap"       2       8            1

  ExecTest  "amo_and"          2       1            1

  ExecTest  "amo_fetchand"     2       1            1

  ExecTest  "amo_xor"          2       1            1

  ExecTest  "amo_swap"         2       1            1
  ExecTest  "amo_fetchor"      2       1            1
  ExecTest  "amo_fetchxor"     2       1            1
  ExecTest  "amo_or"           2       1            1
}

TestSigOps() {
  ##############################################################################
  #       | Name             | Ranks | Workgroups | Threads | Max Message Size #
  ##############################################################################
  ExecTest  "putsignal"        2       1            1         1048576
  ExecTest  "putsignal"        2       2            32        1048576
  ExecTest  "wgputsignal"      2       2            32        1048576
  ExecTest  "waveputsignal"    2       1            32        1048576
  ExecTest  "waveputsignal"    2       2            64        1048576

  ExecTest  "putsignalnbi"     2       1            1         1048576
  ExecTest  "putsignalnbi"     2       2            32        1048576
  ExecTest  "wgputsignalnbi"   2       2            32        1048576
  ExecTest  "waveputsignalnbi" 2       1            32        1048576
  ExecTest  "waveputsignalnbi" 2       2            64        1048576

  ExecTest  "signalfetch"      2       1            1
  ExecTest  "wgsignalfetch"    2       2            32
  ExecTest  "wavesignalfetch"  2       1            32
  ExecTest  "wavesignalfetch"  2       1            64
}

TestColl() {
  ##############################################################################
  #       | Name             | Ranks | Workgroups | Threads | Max Message Size #
  ##############################################################################
  ExecTest  "syncall"          2       1            1

  ExecTest  "wavesyncall"      2       1            1

  ExecTest  "wgsyncall"        2       1            1

  ExecTest  "teamsync"         2       1            1
  ExecTest  "teamsync"         2       16           64
  ExecTest  "teamsync"         2       32           256
  ExecTest  "teamsync"         2       39           1024

  ExecTest  "teamwavesync"     2       1            1
  ExecTest  "teamwavesync"     2       16           64
  ExecTest  "teamwavesync"     2       32           256
  ExecTest  "teamwavesync"     2       39           1024

  ExecTest  "teamwgsync"       2       1            1
  ExecTest  "teamwgsync"       2       16           64
  ExecTest  "teamwgsync"       2       32           256
  ExecTest  "teamwgsync"       2       39           1024

  ExecTest  "barrierall"       2       1            1

  ExecTest  "wavebarrierall"   2       1            1

  ExecTest  "wgbarrierall"     2       1            1

  ExecTest  "teambarrier"      2       1            1
  ExecTest  "teambarrier"      2       16           64
  ExecTest  "teambarrier"      2       32           256
  ExecTest  "teambarrier"      2       39           1024

  ExecTest  "teamwavebarrier"  2       1            1
  ExecTest  "teamwavebarrier"  2       16           64
  ExecTest  "teamwavebarrier"  2       32           256
  ExecTest  "teamwavebarrier"  2       39           1024

  ExecTest  "teamwgbarrier"    2       1            1
  ExecTest  "teamwgbarrier"    2       16           64
  ExecTest  "teamwgbarrier"    2       32           256
  ExecTest  "teamwgbarrier"    2       39           1024

  ExecTest  "alltoall"         2       1            64        512

  ExecTest  "teambroadcast"    2       1            64        32768

  ExecTest  "fcollect"         2       1            64        32768

  ExecTest  "teamreduction"    2       1            64        32768
}

TestOnStream() {
  ##############################################################################
  #       | Name             | Ranks | Workgroups | Threads | Max Message Size #
  ##############################################################################
  ExecTest  "putmem_on_stream" 2       1            1         1048576
  export ROCSHMEM_TEST_USE_DEFAULT_STREAM=1
  ExecTest  "putmem_on_stream" 2       1            1         1048576
  unset ROCSHMEM_TEST_USE_DEFAULT_STREAM

  ExecTest  "getmem_on_stream" 2       1            1         1048576

  ExecTest  "signal_wait_until_on_stream" 2  1      1
  if [[ $TEST != ro* ]]; then #AIROCSHMEM-217
  ExecTest  "putmem_signal_on_stream" 2  1          1         1048576
  else echo "Skip:   putmem_signal_on_stream (AIROCSHMEM-217: RO sometimes abort)"; fi

  ExecTest  "barrier_all_on_stream"  2  1           1
  ExecTest  "quiet_on_stream"        2  1           1
  ExecTest  "sync_all_on_stream"     2  1           1
  ExecTest  "alltoallmem_on_stream"  2  1           64        1048576
  ExecTest  "broadcastmem_on_stream" 2  1           64        1048576
}

TestOther() {
  ##############################################################################
  #       | Name             | Ranks | Workgroups | Threads | Max Message Size #
  ##############################################################################
  ExecTest  "init"             2       1            1
  ExecTest  "library_info"     2       1            1
  ExecTest  "hipmodule_init"   2       1            1
  ExecTest  "device_bitcode"   2       1            1
  ExecTest  "device_bitcode"   2       32           1024
  ExecTest  "device_bitcode"   4       16           256
  ExecTest  "device_bitcode"   8       16           128

  ExecTest  "pingpong"         2       1            1
  ExecTest  "pingpong"         2       8            1
  ExecTest  "pingpong"         2       32           1

  ExecTest  "pingall"          2       1            1
  ExecTest  "pingall"          2       8            1
  ExecTest  "pingall"          2       32           1

  ExecTest  "randomaccess"     2       8            128       1048576

  ################################ Flood test ##################################
  if [[ $TEST != ro* ]]; then #AIROCSHMEM-324
  ExecTest  "flood_put"        2       64           1024
  ExecTest  "flood_put"        8       64           1024
  ExecTest  "flood_putnbi"     8       64           1024
  ExecTest  "flood_p"          8       64           1024

  ExecTest  "flood_get"        2       64           1024
  ExecTest  "flood_get"        8       64           1024
  ExecTest  "flood_getnbi"     8       64           1024

  if [[ $TEST == gda-mlx5* ]]; then  
  ExecTest  "defaultctx_waveputnbi_dp"    2       1            64        1048576
  ExecTest  "waveputnbi_dp"    2       1            64        1048576
  ExecTest  "defaultctx_amo_add_dp"   2       1            1
  ExecTest  "amo_add_dp"       2       1            1
  else echo "Skip:   *_dp (AIROCSHMEM: GDA *_dp not implemented)"; fi

  if [[ $TEST != gda* ]]; then #AIROCSHMEM-162
  ExecTest  "flood_g"          8       64           1024
  else echo "Skip:   flood_g (AIROCSHMEM-162: GDA _g not implemented)"; fi

  ExecTest  "flood_add"        2       64           1024
  ExecTest  "flood_add"        8       64           1024
  ExecTest  "flood_fadd"       8       64           1024
  ExecTest  "flood_waitadd"    8       64           1024
  else echo "Skip:   flood_* (AIROCSHMEM-324: RO flood tests fail in UCX)"; fi

  # This test requires more contexts than workgroups
  export ROCSHMEM_MAX_NUM_CONTEXTS=1024
  ExecTest  "teamctxinfra"        2       1            1
  ExecTest  "teamctxsingleinfra"  2       1            1
  ExecTest  "teamctxblockinfra"   4       1            1
  ExecTest  "teamctxblockinfra"   5       1            1
  ExecTest  "teamctxoddeveninfra" 4       1            1
  ExecTest  "teamctxoddeveninfra" 5       1            1
  ExecTest  "teamctxsharedinfra"  2       1            1
  ExecTest  "teamctxsharedinfra"  5       1            1
  ExecTest  "teamctxsubsetparentinfra" 4  1            1
  ExecTest  "teamctxsubsetparentinfra" 5  1            1
  unset ROCSHMEM_MAX_NUM_CONTEXTS

  ExecTest  "shmemptr"         2       1            1         8
  ExecTest  "shmemptr"         2       1            1024      8
  ExecTest  "shmemptr"         2       8            1         8
  ExecTest  "shmemptr"         2       16           128       8
}

TestHeatMapRMA() {
  NOTIMEOUT=1
  NOVERIF=1
  ##############################################################################
  #       | Name             | Ranks | Workgroups | Threads | Max Message Size #
  ##############################################################################
  ExecTest  "get"              2       1            1         v1048576
  ExecTest  "get"              2       32           1024      v1073741824
  ExecTest  "waveget"          2       1            64        v1073741824
  ExecTest  "waveget"          2       2            64        v1073741824
  ExecTest  "waveget"          2       16           1024      v1073741824
  ExecTest  "wgget"            2       1            1024      v1073741824
  ExecTest  "wgget"            2       16           1024      v1073741824
  #ExecTest  "wgget"            2       32           1024      v1073741824

  ExecTest  "put"              2       1            1         v1048576
  ExecTest  "put"              2       32           1024      v1073741824
  ExecTest  "waveput"          2       1            64        v1073741824
  ExecTest  "waveput"          2       2            64        v1073741824
  ExecTest  "waveput"          2       16           1024      v1073741824
  ExecTest  "wgput"            2       1            1024      v1073741824
  ExecTest  "wgput"            2       16           1024      v1073741824
  #ExecTest  "wgput"            2       32           1024      v1073741824
}

TestHeatMapColl() {
  NOTIMEOUT=1
  NOVERIF=1
  ExecTest  "alltoall"         2       1            256        v1073741824
  ExecTest  "alltoall"         4       1            256        v1073741824
  ExecTest  "alltoall"         8       1            256        v1073741824
  ExecTest  "alltoall"         16      1            256        v1073741824
  ExecTest  "alltoall"         32      1            256        v1073741824
  ExecTest  "alltoall"         64      1            256        v1073741824
}

TestPerfColl() {
  NOTIMEOUT=1
  NOVERIF=1

  ##############################################################################
  #       | Name                     | Ranks               |  Max Message Size #
  ##############################################################################
  ExecPerfTest  "alltoall"                $RANKS       $MAX_MESSAGE_SIZE
  ExecPerfTest  "alltoallv"               $RANKS       $MAX_MESSAGE_SIZE
  ExecPerfTest  "teambroadcast"           $RANKS       $MAX_MESSAGE_SIZE
  ExecPerfTest  "fcollect"                $RANKS       $MAX_MESSAGE_SIZE
  # ExecPerfTest  "teamreduction"           $RANKS       $MAX_MESSAGE_SIZE
  ExecPerfTest  "alltoallmem_on_stream"   $RANKS       $MAX_MESSAGE_SIZE
  ExecPerfTest  "broadcastmem_on_stream"  $RANKS       $MAX_MESSAGE_SIZE
}

TestRMAPerf() {
  NOTIMEOUT=1
  NOVERIF=1

  local rma_ranks=$RANKS
  if [ $rma_ranks -gt 2 ]; then    
    rma_ranks=2  
  fi

  ##############################################################################
  #       | Name                    | Ranks                 | Max Message Size #
  ##############################################################################
  ExecPerfTest  "put"                     $rma_ranks         $MAX_MESSAGE_SIZE
  ExecPerfTest  "wgput"                   $rma_ranks         $MAX_MESSAGE_SIZE
  ExecPerfTest  "waveput"                 $rma_ranks         $MAX_MESSAGE_SIZE
  
  ExecPerfTest  "putnbi"                  $rma_ranks         $MAX_MESSAGE_SIZE
  ExecPerfTest  "wgputnbi"                $rma_ranks         $MAX_MESSAGE_SIZE
  ExecPerfTest  "waveputnbi"              $rma_ranks         $MAX_MESSAGE_SIZE
  
  ExecPerfTest  "get"                     $rma_ranks         $MAX_MESSAGE_SIZE
  ExecPerfTest  "wgget"                   $rma_ranks         $MAX_MESSAGE_SIZE
  ExecPerfTest  "waveget"                 $rma_ranks         $MAX_MESSAGE_SIZE
  
  ExecPerfTest  "getnbi"                  $rma_ranks         $MAX_MESSAGE_SIZE
  ExecPerfTest  "wggetnbi"                $rma_ranks         $MAX_MESSAGE_SIZE
  ExecPerfTest  "wavegetnbi"              $rma_ranks         $MAX_MESSAGE_SIZE
  
  ExecPerfTest  "putmem_on_stream"        $rma_ranks         $MAX_MESSAGE_SIZE
  ExecPerfTest  "getmem_on_stream"        $rma_ranks         $MAX_MESSAGE_SIZE
  
  if [[ $TEST == perf-mlx5* ]]; then  
  ExecPerfTest  "defaultctx_waveputnbi_dp"  $rma_ranks       $MAX_MESSAGE_SIZE
  ExecPerfTest  "waveputnbi_dp"             $rma_ranks       $MAX_MESSAGE_SIZE
  else echo "Skip:   *_dp (AIROCSHMEM: GDA *_dp not implemented)"; fi
}

TestPerf() {
  TestRMAPerf
  TestPerfColl
}

ExecPerfTest() {
  NOTIMEOUT=1
  NOVERIF=1
  
  local test_name=$1
  local num_ranks=$2
  local max_msg_size=$3
  
  local -a wg_options=(${ROCSHMEM_TEST_WGS:-16})
  local -a thread_options=(${ROCSHMEM_TEST_THDS:-128 256})

  if [[ "$test_name" == "alltoallv" ]]; then
    # alltoallv only supports one workgroup
    wg_options=(1)
  fi
  
  declare -A best_latency
  declare -A best_bandwidth
  declare -A best_latency_config
  declare -A best_bandwidth_config
  
  echo "--------------------------------------------------------------------------------------"
  echo "Performance Test: $test_name"
  echo "Ranks: $num_ranks, Max Msg Size: $max_msg_size"
  echo "Workgroups: ${wg_options[@]}"
  echo "Threads: ${thread_options[@]}"
  echo "--------------------------------------------------------------------------------------"
  
  for wg in "${wg_options[@]}"; do
    for threads in "${thread_options[@]}"; do
      if [ $((wg * threads)) -gt 65536 ]; then
        continue
      fi
      
      ExecTest "$test_name" "$num_ranks" "$wg" "$threads" "$max_msg_size"
      
      local log_file="$LOG_DIR/${test_name}_n${num_ranks}_w${wg}_z${threads}_${max_msg_size}B.log"
      
      if [ ! -f "$log_file" ]; then
        continue
      fi
      
      local results=$(awk -v max="$max_msg_size" '
        /Msg Size.*Latency.*Bandwidth/ { in_table=1; next }
        in_table && /^[0-9]/ && $2 <= max { print $2 ":" $4 ":" $5 }
      ' "$log_file")
      
      while IFS= read -r line; do
        [ -z "$line" ] && continue
        
        IFS=':' read -r msg_size latency bandwidth <<< "$line"
        
        if [ -z "${best_latency[$msg_size]}" ] || \
           awk -v a="$latency" -v b="${best_latency[$msg_size]}" 'BEGIN {exit !(a < b)}'; then
          best_latency[$msg_size]="$latency"
          best_latency_config[$msg_size]="${wg}:${threads}"
        fi
        
        if [ -z "${best_bandwidth[$msg_size]}" ] || \
           awk -v a="$bandwidth" -v b="${best_bandwidth[$msg_size]}" 'BEGIN {exit !(a > b)}'; then
          best_bandwidth[$msg_size]="$bandwidth"
          best_bandwidth_config[$msg_size]="${wg}:${threads}"
        fi
      done <<< "$results"
    done
  done
  
  local result_file="$LOG_DIR/${test_name}_n${num_ranks}_${max_msg_size}B_best.log"
  
  {
    echo ""
    echo "======================================================================================"
    printf "  %-15s   %-15s   %-15s   %-15s   %-15s\n" \
      "Msg Size" "Best Lat(us)" "Lat(WG:Threads)" "Best BW(GB/s)" "BW(WG:Threads)"
    echo "--------------------------------------------------------------------------------------"
    
    for msg_size in $(echo "${!best_latency[@]}" | tr ' ' '\n' | sort -n); do
      printf "  %-15s   %-15s   %-15s   %-15s   %-15s\n" \
        "$msg_size" \
        "${best_latency[$msg_size]}" \
        "${best_latency_config[$msg_size]}" \
        "${best_bandwidth[$msg_size]}" \
        "${best_bandwidth_config[$msg_size]}"
    done
    
    echo "======================================================================================"
  } | tee "$result_file"
}

ValidateInput() {
  INPUT_COUNT=$1
  ALL_ARGS=("$@") 

  if [[ "$*" == *"--show-cases"* ]]; then
    echo "Functional Test Cases:"
    for test_case in "${!TEST_NUMBERS[@]}"; do
      echo "$test_case"
    done
    exit 1
  fi
  
  if [ $INPUT_COUNT -lt 3 ] ; then
    echo "This script must be run with at least 3 arguments."
    echo "Usage: ${0} <executable> <test_suite | test_name | test_config> <log_dir> [hostfile] [--show-cases]"
    echo
    echo "    <executable>  : path to the tester executable"
    echo "    <test_suite>  : test suite to run, e.g. 'all', 'rma', or 'put'"
    echo "    <test_name>   : name of test to run, e.g. 'putnbi' or 'amo_fadd'"
    echo "    <test_config> : quoted test configuration to run, in the format"
    echo '                    "<test_name> <ranks> <workgroups> <threads> [max_msg_size]"'
    echo '                    e.g. "putnbi 2 8 1024 65536" or "amo_fadd 2 1 64"'
    echo "        <ranks>        : number of PEs/ranks to use for test"
    echo "        <workgroups>   : number of workgroups per PE"
    echo "        <threads>      : number of threads per workgroup"
    echo "        [max_msg_size] : maximum message size to test"
    echo
    echo "        Performance Test Mode:"
    echo "            Syntax: 'perf/perf-mlx5 <test_name> <ranks> [max_msg_size]'"
    echo "            <test_name>      : 'all', 'coll', 'rma', or specific test (put, get, alltoall...)"
    echo "            <ranks>          : number of PEs/ranks to use for test"
    echo "            [max_msg_size]   : maximum message size to test, default 8MB"
    echo "            Examples:"
    echo '                "perf all 2 1048576"      # All tests, 2 ranks, max 1MB'
    echo '                "perf put 2 8388608"      # Put test, 2 ranks, max 8MB'
    echo '                "perf-mlx5 alltoall 2 65536" # Alltoall on mlx5, 2 ranks, max 64KB'
    echo
    echo "    <log_dir>     : path to output log directory"
    echo "    [hostfile]    : path to hostfile"
    echo "    [--show-cases] : show all available test case names"
    echo
    echo "Environment Vars:"
    echo "     ROCSHMEM_TEST_MPI_PARAMS: MPI parameters(excluding -np/-n) to override default, e.g. '-x LD_LIBRARY_PATH -x ROCSHMEM_BACKEND=gda'"
    echo "     ROCSHMEM_TEST_WGS: workgroup options per PE in performance test, default '8 16 32'"
    echo "     ROCSHMEM_TEST_THDS: thread options per workgroup in performance test, default '64 128 256'" 
    exit 1
  fi
}

ValidateLogDir() {
  if [ ! -d $1 ]; then
    echo "LOG_DIR=$1 does not exist"
    mkdir -p $1
    echo "Created $1"
  fi
}

RerunFailedTests() {
  if [ ${#FAILED_TESTS[@]} -eq 0 ]; then
    return
  fi

  echo ""
  echo "========================================================================"
  echo "Rerunning ${#FAILED_TESTS[@]} failed test(s)..."
  echo "========================================================================"
  echo ""

  # Clear the driver return status for retry
  DRIVER_RETURN_STATUS=0
  RETRY_FAILED_LIST=""
  RETRY_PASSED_LIST=""

  # Rerun each failed test with the same environment/config state
  for test_params in "${FAILED_TESTS[@]}"; do
    IFS='|' read -r test_name num_ranks num_wg num_threads max_msg_size use_default_stream max_contexts notimeout noverif <<< "$test_params"

    # Restore environment state from original test run
    if [[ -n "$use_default_stream" ]]; then
      export ROCSHMEM_TEST_USE_DEFAULT_STREAM="$use_default_stream"
    fi
    if [[ -n "$max_contexts" ]]; then
      export ROCSHMEM_MAX_NUM_CONTEXTS="$max_contexts"
    fi
    if [[ -n "$notimeout" ]]; then
      NOTIMEOUT="$notimeout"
    fi
    if [[ -n "$noverif" ]]; then
      NOVERIF="$noverif"
    fi

    ExecTest "$test_name" "$num_ranks" "$num_wg" "$num_threads" "$max_msg_size" 1

    # Clean up environment state after retry
    unset ROCSHMEM_TEST_USE_DEFAULT_STREAM
    unset ROCSHMEM_MAX_NUM_CONTEXTS
    unset NOTIMEOUT
    unset NOVERIF
  done

  echo ""
  echo "========================================================================"
  echo "Retry Summary"
  echo "========================================================================"

  if [[ -n "$RETRY_PASSED_LIST" ]]; then
    echo -e "$PRETTY_PASSED on retry:$RETRY_PASSED_LIST"
  fi

  if [[ -n "$RETRY_FAILED_LIST" ]]; then
    echo -e "$PRETTY_FAILED even after retry:$RETRY_FAILED_LIST"
  fi

  echo ""
}

PrintfNetWorkInfo() {
  has_ip_cmd=false; command -v ip &>/dev/null && has_ip_cmd=true
  has_ifconfig_cmd=false; command -v ifconfig &>/dev/null && has_ifconfig_cmd=true
  has_ibstat_cmd=false; command -v ibstat &>/dev/null && has_ibstat_cmd=true
  has_ethtool_cmd=false; command -v ethtool &>/dev/null && has_ethtool_cmd=true

  for nic in $(ls /sys/class/net 2>/dev/null | grep -v '^lo$'); do
    ip_info="Unknown!"
    if $has_ip_cmd; then
      ip_info=$(ip -4 addr show $nic 2>/dev/null | grep -oP 'inet \K[\d.]+')
    elif $has_ifconfig_cmd; then
      ip_info=$(ifconfig $nic 2>/dev/null | grep -oP 'inet \K[\d.]+')
    fi
    [[ -z "$ip_info" ]] && ip_info="Unknown!"

    driver_info="Unknown!"
    if [ -f "/sys/class/net/$nic/device/uevent" ]; then
      driver_info=$(grep "^DRIVER=" /sys/class/net/$nic/device/uevent 2>/dev/null | cut -d'=' -f2)
      [[ -z "$driver_info" ]] && driver_info="Unknown!"
    fi

    speed_info="Unknown!"
    port_info="Unknown!"
    ib_info=""

    # 检查是否为 IB/RoCE 设备
    ib_device=""
    if [ -d "/sys/class/net/$nic/device/infiniband" ]; then
      ib_device=$(ls /sys/class/net/$nic/device/infiniband 2>/dev/null | head -n 1)
    fi

    # 缓存命令输出，避免重复执行
    ibstat_out=""
    ethtool_out=""
    [[ -n "$ib_device" ]] && $has_ibstat_cmd && ibstat_out=$(ibstat $ib_device 2>/dev/null)
    $has_ethtool_cmd && ethtool_out=$(ethtool $nic 2>/dev/null)

    # 获取物理端口类型
    if [[ -n "$ethtool_out" ]]; then
      current_port=$(echo "$ethtool_out" | grep -i "Port:" | awk -F': ' '{print $2}' | tr -d '[:space:]')
      supported_ports=$(echo "$ethtool_out" | grep -i "Supported ports:" | awk -F': ' '{print $2}' | tr -d '[:space:]')
      if [[ -n "$current_port" ]]; then
        port_info="$current_port"
        [[ -n "$supported_ports" ]] && port_info+=" (Supported: $supported_ports)"
      fi
    fi

    if [[ -n "$ibstat_out" ]]; then
      # IB/RoCE设备处理
      rate=$(echo "$ibstat_out" | grep "Rate:" | awk '{print $2}')
      link_layer=$(echo "$ibstat_out" | grep "Link layer:" | awk -F': ' '{print $2}')
      state=$(echo "$ibstat_out" | grep "State:" | head -1 | awk -F': ' '{print $2}')
      phys_state=$(echo "$ibstat_out" | grep "Physical state:" | head -1 | awk -F': ' '{print $2}')
      ca_type=$(echo "$ibstat_out" | grep "CA type:" | awk -F': ' '{print $2}')
      fw_ver=$(echo "$ibstat_out" | grep "Firmware version:" | awk -F': ' '{print $2}')
      node_guid=$(echo "$ibstat_out" | grep "Node GUID:" | awk -F': ' '{print $2}')
      base_lid=$(echo "$ibstat_out" | grep "Base lid:" | awk -F': ' '{print $2}')

      if [[ -n "$rate" ]]; then
        if [[ "$link_layer" == "Ethernet" ]]; then
          speed_info="${rate}Gbps (RoCE)"
        else
          speed_info="${rate}Gbps (IB)"
        fi
      fi

      # 组装IB协议栈信息
      ib_info="IB State($ib_device): ${state:-Unknown}, Phys: ${phys_state:-Unknown}, Link layer: ${link_layer:-Unknown}"
      [[ -n "$ca_type" ]] && ib_info+=", CA: $ca_type"
      [[ -n "$fw_ver" ]] && ib_info+=", FW: $fw_ver"
      [[ -n "$node_guid" ]] && ib_info+=", GUID: $node_guid"
      [[ -n "$base_lid" ]] && ib_info+=", LID: $base_lid"
      
    elif [[ -n "$ethtool_out" ]]; then
      # 普通以太网设备处理
      speed_info=$(echo "$ethtool_out" | grep -i "Speed:" | awk -F': ' '{print $2}' | tr -d '[:space:]')
      [[ -z "$speed_info" ]] && speed_info="Unknown!"
    fi

    # 格式化打印
    echo "$nic: IP: $ip_info, Speed: $speed_info"
    echo "      Port: $port_info, Driver: $driver_info"
    [[ -n "$ib_info" ]] && echo "      $ib_info"
  done
}

PrintEnvInfo() {
  local env_log="$LOG_DIR/env_info.log"
  local timestamp=$(date "+%Y-%m-%d %H:%M:%S")
  
  # 确保日志目录存在
  mkdir -p "$LOG_DIR" || { echo "Error: Failed to create log directory $LOG_DIR"; return 1; }
  
  {
    echo -e "================================================================================================"
    echo -e "Environment info: $timestamp"
    echo -e "================================================================================================"

    # build information and environment variables
    echo ""
    echo -e "================================== Build Info and Env Vars ====================================="
    [ -x "$ROCSHMEM_INFO" ] && "$ROCSHMEM_INFO" --env:all 2>&1

    # DTK
    echo ""
    echo -e "============================================ DTK ==============================================="
    cat /opt/dtk/.dtk_version 2>/dev/null

    # System info
    echo ""
    echo -e "=========================================== System ============================================="
    uname -a 2>/dev/null
    cat /etc/os-release 2>/dev/null | grep -E "^(NAME|VERSION)="

    # CPU info
    echo ""
    echo -e "=========================================== CPU ================================================"
    lscpu 2>/dev/null | grep -v "Flags:"

    # GPU info
    echo ""
    echo -e "============================================ GPU ==============================================="
    hy-smi --showid --showproductname --showdriverversion 2>&1

    # Network info
    echo ""
    echo -e "=========================================== Network ============================================"
    PrintfNetWorkInfo

    echo ""
    echo -e "================================================================================================"
    echo -e "Environment info collection completed: $timestamp"
    echo -e "================================================================================================"
  } > "$env_log" 2>&1
  
  if [ $? -eq 0 ]; then
    echo "Environment info saved to: $env_log"
  else
    echo "Error: Failed to collect environment info"
    return 1
  fi
}

APP=$1
TEST=$2
LOG_DIR=$3
HOSTFILE=$4

DRIVER_RETURN_STATUS=0
FAILED_TESTS=()  # Array to store failed test parameters
RETRY_THRESHOLD=${RETRY_THRESHOLD:-5}  # Maximum number of failed tests to retry (can be overridden via env var)

ValidateInput $# "$@"
ValidateLogDir $LOG_DIR

# Print build info and environment variables before running tests
ROCSHMEM_INFO="$(dirname "$APP")/rocshmem_info"
if [ ! -x "$ROCSHMEM_INFO" ]; then
  # builddir case
  ROCSHMEM_INFO="$(dirname "$APP")/../../tools/rocshmem_info"
fi
if [ -x "$ROCSHMEM_INFO" ]; then
  "$ROCSHMEM_INFO"
fi

PrintEnvInfo

case $TEST in
  "heatmaprma")
    TestHeatMapRMA
    ;;
  "heatmapcoll")
    TestHeatMapColl
    ;;
  "heatmap")
    TestHeatMapRMA
    TestHeatMapColl
    ;;
  "all"|"gda"|"gda-mlx5"|"gda-bnxt"|"gda-ionic"|"ro"|"all-ro")
    TEST=${TEST#all-} #convert all-ro used in CI scripts into simple ro prefix
    TestRMA
    TestAMO
    TestSigOps
    TestColl
    TestOther
    TestOnStream
    ;;
  *"rma")
    TestRMA
    ;;
  *"put")
    TestRMAPut
    ;;
  *"get")
    TestRMAGet
    ;;
  *"amo")
    TestAMO
    ;;
  *"sigops")
    TestSigOps
    ;;
  *"coll")
    TestColl
    ;;
  *"stream")
    TestOnStream
    ;;
  *"other")
    TestOther
    ;;
  "perf"*|"perf-mlx5"*)
    TEST_OPTS=($TEST)
    NAME=${TEST_OPTS[1]}
    if [ ${#TEST_OPTS[@]} -ge 3 ]; then
      RANKS=${TEST_OPTS[2]}
      MAX_MESSAGE_SIZE=${TEST_OPTS[3]:-8388608}
    fi

    if [ "$NAME" == "all" ]; then
      TestPerf
    elif [ "$NAME" == "rma" ]; then
      TestRMAPerf
    elif [ "$NAME" == "coll" ]; then
      TestPerfColl
    else
      ExecPerfTest  "${NAME}"  "${RANKS}"  "${MAX_MESSAGE_SIZE}"
    fi
    ;;
  *)
    #######################################################################################
    #        |   Name   |   Ranks   |   Workgroups   |   Threads   |   Max Message Size   #
    #######################################################################################
    # Allow passing in a test config as "<test_name> <ranks> <workgroups> <threads> [max_msg_size]"
    # e.g. "putnbi 2 8 1024 65536" or "amo_fadd 2 1 64"
    TEST_OPTS=($TEST)
    NAME=${TEST_OPTS[0]}
    if [ ${#TEST_OPTS[@]} -eq 4 ] || [ ${#TEST_OPTS[@]} -eq 5 ]; then
      RANKS=${TEST_OPTS[1]}
      WORKGROUPS=${TEST_OPTS[2]}
      THREADS=${TEST_OPTS[3]}
      MAX_MESSAGE_SIZE=${TEST_OPTS[4]}
    else
      RANKS=2
      WORKGROUPS=1
      THREADS=1
      MAX_MESSAGE_SIZE=8
    fi

    ExecTest  "${NAME}"  "${RANKS}"  "${WORKGROUPS}"  "${THREADS}"  "${MAX_MESSAGE_SIZE}"
    ;;
esac

EXIT_STATUS=$DRIVER_RETURN_STATUS

# If there were failures, try to rerun them once (only if below threshold)
if [ $EXIT_STATUS -ne 0 ] && [ ${#FAILED_TESTS[@]} -gt 0 ]; then
  if [ ${#FAILED_TESTS[@]} -le $RETRY_THRESHOLD ]; then
    RerunFailedTests
    EXIT_STATUS=$DRIVER_RETURN_STATUS
  else
    echo ""
    echo "========================================================================"
    echo "Too many test failures (${#FAILED_TESTS[@]} > $RETRY_THRESHOLD threshold)"
    echo "Skipping retry - this may indicate a systemic issue"
    echo "========================================================================"
    echo ""
  fi
fi

# Final summary
echo ""
echo "========================================================================"
if [ $EXIT_STATUS -eq 0 ]; then
  echo -e "TESTS PASSED"
else
  if [[ -n "$RETRY_FAILED_LIST" ]]; then
    echo -e "TESTS FAILED (even after retry): $RETRY_FAILED_LIST"
  else
    echo -e "TESTS FAILED: $FAILED_LIST"
  fi
fi
echo "========================================================================"
exit $EXIT_STATUS
