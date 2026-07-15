###############################################################################
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################

include_guard(GLOBAL)

if(NOT TARGET rocshmem)
  message(FATAL_ERROR "RocshmemSplitLibraries.cmake requires the rocshmem target")
endif()
if(NOT TARGET rocshmem_device_bitcode)
  message(FATAL_ERROR
    "RocshmemSplitLibraries.cmake requires rocshmem device bitcode targets")
endif()

###############################################################################
# DEVICE-ONLY STATIC LIBRARY
###############################################################################
# Bundle all architecture-specific bitcode produced by DeviceBitcode.cmake into
# a HIP offload object, then combine it with the one device-state definition TU.
set(_bundle_targets "")
foreach(_arch ${BITCODE_GPU_ARCHS})
  list(APPEND _bundle_targets "hip-amdgcn-amd-amdhsa--${_arch}")
endforeach()
list(JOIN _bundle_targets "," _bundle_targets_arg)
list(JOIN ALL_BITCODE_OUTPUTS "," _bundle_inputs_arg)

find_program(CLANG_OFFLOAD_BUNDLER clang-offload-bundler
  PATHS ${ROCM_PATH}/llvm/bin ${THEROCK_TOOLCHAIN_ROOT}/lib/llvm/bin
  NO_DEFAULT_PATH REQUIRED)

set(ROCSHMEM_DEVICE_OBJECT "${CMAKE_BINARY_DIR}/librocshmem_device.o")
add_custom_command(
  OUTPUT ${ROCSHMEM_DEVICE_OBJECT}
  COMMAND ${CLANG_OFFLOAD_BUNDLER} -type=o
          -targets=${_bundle_targets_arg}
          -inputs=${_bundle_inputs_arg}
          -outputs=${ROCSHMEM_DEVICE_OBJECT}
  DEPENDS ${ALL_BITCODE_OUTPUTS}
  COMMENT "Bundling rocSHMEM device bitcode"
  VERBATIM)

add_library(rocshmem_device STATIC
  ${ROCSHMEM_DEVICE_OBJECT}
  ${CMAKE_SOURCE_DIR}/src/device_globals.cpp)
add_library(roc::rocshmem_device ALIAS rocshmem_device)

set_source_files_properties(${ROCSHMEM_DEVICE_OBJECT} PROPERTIES
  GENERATED TRUE
  EXTERNAL_OBJECT TRUE)
add_dependencies(rocshmem_device rocshmem_device_bitcode)

target_compile_options(rocshmem_device PRIVATE -fgpu-rdc)
target_include_directories(rocshmem_device PRIVATE
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_BINARY_DIR}/include
  ${CMAKE_BINARY_DIR}/include/rocshmem)
target_link_libraries(rocshmem_device PRIVATE
  hip::device
  $<$<BOOL:${HAVE_EXTERNAL_MPI}>:MPI::MPI_CXX>)

###############################################################################
# HOST-ONLY SHARED LIBRARY
###############################################################################
# Derive the host source set from the fully populated rocshmem target so that
# conditional backend sources cannot drift from this library.
get_target_property(ROCSHMEM_ALL_SOURCES rocshmem SOURCES)
list(REMOVE_ITEM ROCSHMEM_ALL_SOURCES
  device_globals.cpp
  ${CMAKE_SOURCE_DIR}/src/device_globals.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/device_globals.cpp)

add_library(rocshmem_host SHARED ${ROCSHMEM_ALL_SOURCES})
add_library(roc::rocshmem_host ALIAS rocshmem_host)

foreach(_prop IN ITEMS
    COMPILE_DEFINITIONS COMPILE_FEATURES COMPILE_OPTIONS
    INCLUDE_DIRECTORIES LINK_LIBRARIES LINK_OPTIONS
    INTERFACE_COMPILE_DEFINITIONS INTERFACE_COMPILE_FEATURES
    INTERFACE_COMPILE_OPTIONS INTERFACE_INCLUDE_DIRECTORIES
    INTERFACE_LINK_LIBRARIES INTERFACE_LINK_OPTIONS)
  get_target_property(_value rocshmem ${_prop})
  if(_value AND NOT _value STREQUAL "_value-NOTFOUND")
    set_property(TARGET rocshmem_host PROPERTY ${_prop} "${_value}")
  endif()
endforeach()

target_compile_options(rocshmem_host PRIVATE --cuda-host-only -fPIC)
set_target_properties(rocshmem_host PROPERTIES
  OUTPUT_NAME rocshmem_host
  POSITION_INDEPENDENT_CODE ON)
target_link_options(rocshmem_host PRIVATE -Wl,-Bsymbolic)

# Host on-stream APIs launch kernels whose stubs are compiled into
# rocshmem_host.  Embed the matching device archive in the same DSO so HIP can
# register those stubs against a fatbin containing the actual kernels.
target_link_libraries(rocshmem_host PRIVATE
  -Wl,--whole-archive
  rocshmem_device
  -Wl,--no-whole-archive)

###############################################################################
# INSTALL
###############################################################################
rocm_install(TARGETS rocshmem_host rocshmem_device)
