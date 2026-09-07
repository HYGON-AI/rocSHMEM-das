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

cmake_minimum_required(VERSION 3.16.3 FATAL_ERROR)

###############################################################################
# PROJECT
###############################################################################
include(${CMAKE_SOURCE_DIR}/cmake/setup_project.cmake)
project(rocshmem_functional_tests VERSION 1.0.0 LANGUAGES CXX)

# ASAN option (if not already defined by parent project)
option(ASAN "Enable AddressSanitizer" ${ASAN})

###############################################################################
# SOURCES
###############################################################################
add_executable(${PROJECT_NAME} "")

# In split-library mode, also build a second executable that links against the
# separate rocshmem_host (shared) and rocshmem_device (static) libraries so
# both link scenarios are exercised. The original executable always links the
# combined roc::rocshmem target.
set(FUNCTIONAL_TEST_TARGETS ${PROJECT_NAME})
set(FUNCTIONAL_TEST_SPLIT OFF)
if(TARGET roc::rocshmem_host AND TARGET roc::rocshmem_device)
  set(FUNCTIONAL_TEST_SPLIT_TARGET ${PROJECT_NAME}_splitlibs)
  add_executable(${FUNCTIONAL_TEST_SPLIT_TARGET} "")
  list(APPEND FUNCTIONAL_TEST_TARGETS ${FUNCTIONAL_TEST_SPLIT_TARGET})
  set(FUNCTIONAL_TEST_SPLIT ON)
endif()

set(FUNCTIONAL_TEST_SOURCES
    barrier_all_tester.cpp
    sync_all_tester.cpp
    test_driver.cpp
    tester.cpp
    tester_arguments.cpp
    ping_pong_tester.cpp
    ping_all_tester.cpp
    primitive_tester.cpp
    default_ctx_primitive_tester.cpp
    team_ctx_primitive_tester.cpp
    team_ctx_infra_tester.cpp
    amo_bitwise_tester.cpp
    amo_extended_tester.cpp
    amo_standard_tester.cpp
    random_access_tester.cpp
    shmem_ptr_tester.cpp
    signaling_operations_tester.cpp
    signaling_operations_tester.hpp
    workgroup_primitives.cpp
    empty_tester.cpp
    wavefront_primitives.cpp
    wavefront_dp_primitives.cpp
    amo_dp_standard_tester.cpp
    flood_tester.cpp
    flood_amo_tester.cpp
    device_bitcode_tester.cpp
    library_info_tester.cpp
    fence_ordering_tester.cpp
    tile_rma_tester.cpp
    host_team_sync_barrier_tester.cpp
    host_rma_tester.cpp
    tile_broadcast_tester.cpp
    tile_allgather_tester.cpp
)

foreach(_t IN LISTS FUNCTIONAL_TEST_TARGETS)
  target_include_directories(
    ${_t}
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
  )
  target_sources(
    ${_t}
    PRIVATE
      ${FUNCTIONAL_TEST_SOURCES}
  )
endforeach()

###############################################################################
# ROCSHMEM
###############################################################################
if (BUILD_TESTS_ONLY)
  #TODO these find_packages should be performed as-needed in rocshmem-config.cmake
  find_package(hip REQUIRED)
  find_package(MPI)
  find_package(Threads REQUIRED)
  find_package(rocshmem REQUIRED)

  foreach(_t IN LISTS FUNCTIONAL_TEST_TARGETS)
    target_include_directories(
      ${_t}
      PRIVATE
        $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../src>
    )
  endforeach()
endif()

find_package(PMIx)

foreach(_t IN LISTS FUNCTIONAL_TEST_TARGETS)
  target_compile_definitions(
    ${_t}
    PRIVATE
      $<$<TARGET_EXISTS:PMIx::pmix>:HAVE_PMIX=1>
      CMAKE_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
      CMAKE_BINARY_DIR="${CMAKE_BINARY_DIR}"
  )
endforeach()

configure_file(../../scripts/functional_tests/driver.sh rocshmem_functional_driver.sh COPYONLY)
rocm_install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/rocshmem_functional_driver.sh COMPONENT tests DESTINATION ${CMAKE_INSTALL_DATADIR}/rocshmem)
foreach(_t IN LISTS FUNCTIONAL_TEST_TARGETS)
  install(TARGETS ${_t} COMPONENT tests RUNTIME DESTINATION ${CMAKE_INSTALL_DATADIR}/rocshmem)
endforeach()

# Common link libraries shared by all functional test executables.
set(FUNCTIONAL_TEST_COMMON_LINK_LIBS
  $<TARGET_NAME_IF_EXISTS:MPI::MPI_CXX>
  $<TARGET_NAME_IF_EXISTS:PMIx::pmix>
  hiprtc
)

# ${PROJECT_NAME} always links the combined roc::rocshmem library.
target_link_libraries(
  ${PROJECT_NAME}
  PRIVATE
    roc::rocshmem
    ${FUNCTIONAL_TEST_COMMON_LINK_LIBS}
)

# In split-library mode, the _splitlibs executable links the separate
# rocshmem_host (shared) and rocshmem_device (static) libraries.
if(FUNCTIONAL_TEST_SPLIT)
  target_link_libraries(
    ${FUNCTIONAL_TEST_SPLIT_TARGET}
    PRIVATE
      roc::rocshmem_host
      -Wl,--whole-archive
      roc::rocshmem_device
      -Wl,--no-whole-archive
      ${FUNCTIONAL_TEST_COMMON_LINK_LIBS}
  )
  target_link_options(${FUNCTIONAL_TEST_SPLIT_TARGET} PRIVATE -Wl,--export-dynamic)
endif()

if(${ROCM_MAJOR_VERSION} LESS 7)
  # ROCm 6.x requires us to explicitly enable warp sync builtins
  foreach(_t IN LISTS FUNCTIONAL_TEST_TARGETS)
    target_compile_definitions(${_t} PRIVATE HIP_ENABLE_WARP_SYNC_BUILTINS=1)
  endforeach()
endif()

###############################################################################
# ASAN SETTINGS
###############################################################################
foreach(_t IN LISTS FUNCTIONAL_TEST_TARGETS)
  target_compile_options(
    ${_t}
    PRIVATE
      $<$<BOOL:${ASAN}>:-fsanitize=address -g>
  )
  target_link_options(
    ${_t}
    PRIVATE
      $<$<BOOL:${ASAN}>:-fsanitize=address -g -shared-libsan>
  )
endforeach()

###############################################################################
# DEVICE BITCODE TESTER (put/get via HSACO + rocshmem_hipmodule_init)
###############################################################################
include(${CMAKE_CURRENT_SOURCE_DIR}/CMakeDeviceBitcodeTester.cmake)

###############################################################################
# CTEST INTEGRATION (replaces driver.sh with native CTest tests)
###############################################################################
# Enable testing
enable_testing()

# Install the wrapper script and retry helper
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/test_wrapper.sh
               ${CMAKE_CURRENT_BINARY_DIR}/test_wrapper.sh COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/ctest_with_retry.sh
               ${CMAKE_CURRENT_BINARY_DIR}/ctest_with_retry.sh COPYONLY)
rocm_install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/test_wrapper.sh
             ${CMAKE_CURRENT_BINARY_DIR}/ctest_with_retry.sh
             COMPONENT tests DESTINATION ${CMAKE_INSTALL_DATADIR}/rocshmem)

# Include CTest-based functional tests
# Set BUILD_CTESTS=ON to register all tests (can be slow for large test suites)
# By default, only the shell script is installed for compatibility
if(BUILD_CTESTS)
    message(STATUS "Registering CTest-based functional tests")
    include(${CMAKE_CURRENT_SOURCE_DIR}/CTestFunctionalTests.cmake OPTIONAL)

    # Check if CTest generation functions are available
    if(COMMAND init_install_ctest_generation AND COMMAND register_all_functional_tests)
        # Initialize install-time test generation
        set(INSTALL_CTEST_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/CTestFunctionalInstall.cmake")
        init_install_ctest_generation("${INSTALL_CTEST_OUTPUT}")

        # Configure template header
        configure_file(
            ${CMAKE_CURRENT_SOURCE_DIR}/CTestFunctionalInstall.cmake.in
            ${INSTALL_CTEST_OUTPUT}
            @ONLY)

        # Register all tests (this will populate both build-time and install-time)
        register_all_functional_tests()

        # Apply YAML-based tier labels (build-time + install-time)
        set(_test_categories_yaml "${CMAKE_SOURCE_DIR}/tests/test_categories.yaml")
        include(${CMAKE_SOURCE_DIR}/tests/ApplyInstallTierLabels.cmake OPTIONAL)
        apply_tier_labels_build_and_install("${_test_categories_yaml}" "${INSTALL_CTEST_OUTPUT}")

        # Install the generated install-time CTest file
        rocm_install(FILES "${INSTALL_CTEST_OUTPUT}"
                     COMPONENT tests
                     DESTINATION ${CMAKE_INSTALL_BINDIR}/rocshmem/tests/functional
                     RENAME CTestTestfile.cmake)

        # Print summary
        get_directory_property(all_tests TESTS)
        list(LENGTH all_tests num_tests)
        message(STATUS "Registered ${num_tests} functional tests")
        message(STATUS "Run with: ctest -L functional")
        message(STATUS "Install-time CTest file: ${INSTALL_CTEST_OUTPUT}")
    else()
        message(WARNING "CTest generation files not found in sparse checkout - skipping CTest registration")
        set(BUILD_CTESTS OFF CACHE BOOL "CTests disabled due to missing files" FORCE)
    endif()
endif()

# For backward compatibility and development, also install the shell script
# Usage: rocshmem_functional_driver.sh <executable> <test_suite> <log_dir> [hostfile]