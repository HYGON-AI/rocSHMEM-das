#!/usr/bin/env bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT

set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 <full-archive> <output-dir> [expected-member-count]" >&2
  exit 2
fi

full_archive=$(realpath "$1")
output_dir=$2
expected_count=${3:-}

toolchain_root=${ROCSHMEM_DEVICE_ARCHIVE_TOOLCHAIN_ROOT:-/opt/dtk/llvm/bin}
ar_tool=${ROCSHMEM_DEVICE_ARCHIVE_AR:-"${toolchain_root}/llvm-ar"}
bundler=${ROCSHMEM_DEVICE_ARCHIVE_BUNDLER:-"${toolchain_root}/clang-offload-bundler"}

# Use the compiler toolchain's archive and offload tools so the bundle format
# exactly matches the aicc/DTK version used by the main build.
for tool in "$ar_tool" "$bundler"; do
  if [[ ! -x "$tool" ]]; then
    echo "required tool is not executable: $tool" >&2
    exit 1
  fi
done

if [[ ! -f "$full_archive" ]]; then
  echo "full archive does not exist: $full_archive" >&2
  exit 1
fi

mkdir -p "$output_dir"
# Intermediate unbundled images are kept outside the final member list.
work_dir="${output_dir}/.extract"
mkdir -p "$work_dir"

archive_index=0
device_index=0
# Process members independently to preserve the full library's HIP TU topology.
while IFS= read -r member; do
  [[ -n "$member" ]] || continue
  archive_index=$((archive_index + 1))

  # device_globals.cpp must remain a normal HIP translation unit.  Its host
  # half registers module-local device-global setters and carries the hidden
  # archive anchor, so CMake compiles it separately into the final archive.
  if [[ "$member" == "device_globals.cpp.o" ]]; then
    continue
  fi

  extracted="${work_dir}/${archive_index}_${member}"
  "$ar_tool" p "$full_archive" "$member" >"$extracted"

  # Select device targets only; deliberately omit the host target when the
  # member is bundled again below.
  mapfile -t targets < <("$bundler" --list --type=o --input="$extracted" 2>/dev/null \
    | awk '/^(hip|hipv4)-/ { print }')
  if [[ ${#targets[@]} -eq 0 ]]; then
    echo "archive member has no HIP device bundle: $member" >&2
    exit 1
  fi

  device_index=$((device_index + 1))
  target_csv=$(IFS=,; echo "${targets[*]}")
  unbundled_outputs=()
  for target_index in "${!targets[@]}"; do
    unbundled_outputs+=("${work_dir}/${device_index}_${target_index}.o")
  done
  output_csv=$(IFS=,; echo "${unbundled_outputs[*]}")

  "$bundler" --unbundle --type=o --targets="$target_csv" \
    --input="$extracted" --outputs="$output_csv"

  # Recreate one device-only fat object per original archive member.
  rebundled="${output_dir}/${device_index}_${member}"
  "$bundler" --type=o --targets="$target_csv" \
    --inputs="$output_csv" --output="$rebundled"
done < <("$ar_tool" t "$full_archive")

# Catch source-list/archive drift before CMake attempts to create device.a.
if [[ -n "$expected_count" && "$device_index" -ne "$expected_count" ]]; then
  echo "device member count mismatch: expected $expected_count, produced $device_index" >&2
  exit 1
fi

echo "device_only_members=$device_index"
