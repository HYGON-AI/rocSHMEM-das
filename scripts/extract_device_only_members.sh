#!/usr/bin/env bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT

set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 <full-archive> <output-dir> [expected-member-count]" >&2
  exit 2
fi

if [[ ! -f "$1" ]]; then
  echo "full archive does not exist: $1" >&2
  exit 1
fi
full_archive=$(realpath "$1")
output_dir=$2
expected_count=${3:-}

toolchain_root=${ROCSHMEM_DEVICE_ARCHIVE_TOOLCHAIN_ROOT:-/opt/dtk/llvm/bin}
ar_tool=${ROCSHMEM_DEVICE_ARCHIVE_AR:-"${toolchain_root}/llvm-ar"}
bundler=${ROCSHMEM_DEVICE_ARCHIVE_BUNDLER:-"${toolchain_root}/clang-offload-bundler"}
host_cxx=${ROCSHMEM_DEVICE_ARCHIVE_HOST_CXX:-"${toolchain_root}/clang++"}

for tool in "$ar_tool" "$bundler" "$host_cxx"; do
  if [[ ! -x "$tool" ]]; then
    echo "required tool is not executable: $tool" >&2
    exit 1
  fi
done

mkdir -p "$output_dir"
# Intermediate unbundled images are kept outside the final member list.
work_dir="${output_dir}/.extract"
mkdir -p "$work_dir"

# Build one minimal empty host stub object.  clang-offload-bundler --type=o
# emits a valid ELF only when the target list contains at least one host
# target; otherwise it produces a bare bundle container that the host linker
# rejects with "not an ELF file".  The stub has no symbols so it cannot
# collide with the host-only companion library.
host_stub_src="${work_dir}/host_stub.cpp"
host_stub_obj="${work_dir}/host_stub.o"
: >"$host_stub_src"
"$host_cxx" -fPIC -fvisibility=hidden -c "$host_stub_src" -o "$host_stub_obj"

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

  # Discover the host triple for this member.  Each TU may carry its own
  # host triple, so extract it per-member rather than assuming uniformity.
  member_host_triple=$("$bundler" --list --type=o --input="$extracted" 2>/dev/null \
    | awk '/^host-/ { gsub(/[[:space:]]+$/, ""); print; exit }')
  if [[ -z "$member_host_triple" ]]; then
    echo "archive member has no host bundle: $member" >&2
    exit 1
  fi

  # Select device targets only; host code is replaced by the empty stub below.
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

  output_flags=()
  for f in "${unbundled_outputs[@]}"; do
    output_flags+=("--output=$f")
  done
  "$bundler" --unbundle --type=o --targets="$target_csv" \
    --input="$extracted" "${output_flags[@]}"

  # Rebundle with an empty host stub alongside the device images so the result
  # is a well-formed ELF that the host archiver and linker can ingest.
  # clang-offload-bundler uses singular --input/--output (repeatable) flags;
  # the plural CSV forms are deprecated and inconsistent across LLVM versions.
  rebundle_targets="${member_host_triple},${target_csv}"
  input_flags=("--input=$host_stub_obj")
  for f in "${unbundled_outputs[@]}"; do
    input_flags+=("--input=$f")
  done
  rebundled="${output_dir}/${device_index}_${member}"
  "$bundler" --type=o --targets="$rebundle_targets" \
    "${input_flags[@]}" --output="$rebundled"

  # Verify the rebundled output is a real ELF, not a bare bundle container.
  if [[ ! -s "$rebundled" ]]; then
    echo "rebundle produced empty output: $member" >&2
    exit 1
  fi
  magic=$(od -An -tx1 -N4 "$rebundled" | tr -d ' \n')
  if [[ "$magic" != "7f454c46" ]]; then
    echo "rebundle output is not ELF: $member (magic=$magic)" >&2
    exit 1
  fi
done < <("$ar_tool" t "$full_archive")

# Catch source-list/archive drift before CMake attempts to create device.a.
if [[ -n "$expected_count" && "$device_index" -ne "$expected_count" ]]; then
  echo "device member count mismatch: expected $expected_count, produced $device_index" >&2
  exit 1
fi

echo "device_only_members=$device_index"
