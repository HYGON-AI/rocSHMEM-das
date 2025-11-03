#!/bin/bash
set -eux

ROCSHMEM_INSTALL_PREFIX=${ROCSHMEM_INSTALL_PREFIX:=$(pwd)/rocshmem_dir}

mkdir -p build
cd build

INSTALL_PREFIX=${ROCSHMEM_INSTALL_PREFIX} bash ../scripts/build_configs/gda_mlx5
