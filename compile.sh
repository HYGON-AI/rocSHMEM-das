#!/bin/bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT

set -eux

ROCSHMEM_INSTALL_PREFIX=${ROCSHMEM_INSTALL_PREFIX:=/opt/rocshmem}

# Use one DTK installation for compiler detection and package discovery.
ROCM_PATH=${ROCM_PATH:-${ROCM_HOME:-/opt/dtk}}
export ROCM_PATH
if [[ ! -x "${ROCM_PATH}/bin/hipcc" ]]; then
    printf 'ERROR: hipcc not found in %s/bin; set ROCM_PATH to your DTK root.\n' "${ROCM_PATH}" >&2
    exit 1
fi
export PATH="${ROCM_PATH}/bin:${PATH}"

mkdir -p build
cd build

INSTALL_PREFIX="${ROCSHMEM_INSTALL_PREFIX}" bash ../scripts/build_configs/gda_mlx5 \
    -DROCM_PATH="${ROCM_PATH}" \
    -DCMAKE_CXX_COMPILER="${ROCM_PATH}/bin/hipcc" \
    "$@"
