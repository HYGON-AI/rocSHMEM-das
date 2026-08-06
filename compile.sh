#!/bin/bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT

set -eux

ROCSHMEM_INSTALL_PREFIX=${ROCSHMEM_INSTALL_PREFIX:=$(pwd)/rocshmem_dir}

mkdir -p build
cd build

INSTALL_PREFIX=${ROCSHMEM_INSTALL_PREFIX} bash ../scripts/build_configs/gda_mlx5
