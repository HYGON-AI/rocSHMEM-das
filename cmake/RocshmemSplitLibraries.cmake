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
# Preserve the full target's per-translation-unit HIP member topology.  aicc's
# final GPU image (including scratch/private-segment behaviour) depends on this
# topology.  Strip only the host bundle from every member; device_globals.cpp
# remains one normal HIP TU because its host constructor registers the module's
# device-global setters and defines the hidden force-link anchor.
get_target_property(ROCSHMEM_ALL_SOURCES rocshmem SOURCES)
set(_device_member_dir "${CMAKE_CURRENT_BINARY_DIR}/rocshmem_device_members")
set(_device_member_outputs "")
set(_device_member_count 0)
foreach(_source IN LISTS ROCSHMEM_ALL_SOURCES)
  get_filename_component(_source_name "${_source}" NAME)
  if(_source_name STREQUAL "device_globals.cpp")
    continue()
  endif()
  math(EXPR _device_member_count "${_device_member_count} + 1")
  list(APPEND _device_member_outputs
    "${_device_member_dir}/${_device_member_count}_${_source_name}.o")
endforeach()

find_program(BASH_EXECUTABLE bash REQUIRED)
add_custom_command(
  OUTPUT ${_device_member_outputs}
  COMMAND ${CMAKE_COMMAND} -E make_directory ${_device_member_dir}
  COMMAND ${CMAKE_COMMAND} -E env
          ROCSHMEM_DEVICE_ARCHIVE_TOOLCHAIN_ROOT=${ROCM_PATH}/llvm/bin
          ${BASH_EXECUTABLE}
          ${CMAKE_SOURCE_DIR}/scripts/extract_device_only_members.sh
          $<TARGET_FILE:rocshmem>
          ${_device_member_dir}
          ${_device_member_count}
  DEPENDS rocshmem
          ${CMAKE_SOURCE_DIR}/scripts/extract_device_only_members.sh
  COMMENT "Extracting per-TU rocSHMEM device-only archive members"
  VERBATIM)

set_source_files_properties(${_device_member_outputs} PROPERTIES
  GENERATED TRUE
  EXTERNAL_OBJECT TRUE)

add_library(rocshmem_device STATIC
  ${_device_member_outputs}
  ${CMAKE_SOURCE_DIR}/src/device_globals.cpp)
add_library(roc::rocshmem_device ALIAS rocshmem_device)

target_compile_options(rocshmem_device PRIVATE -fgpu-rdc)
target_include_directories(rocshmem_device PRIVATE
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_BINARY_DIR}/include
  ${CMAKE_BINARY_DIR}/include/rocshmem)
target_link_libraries(rocshmem_device PRIVATE
  hip::device
  $<$<BOOL:${HAVE_EXTERNAL_MPI}>:MPI::MPI_CXX>)
target_link_options(rocshmem_device INTERFACE
  -Wl,-u,rocshmem_force_link_device_module)

###############################################################################
# HOST-ONLY SHARED LIBRARY
###############################################################################
# Derive the host source set from the fully populated rocshmem target so that
# conditional backend sources cannot drift from this library.
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
