# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT

# Use one traceable filename identity for every release artifact:
set(ROCSHMEM_PACKAGE_DTK_VERSION "" CACHE STRING
  "DTK filename version without separators (for example, 2604 or 26042)")
if(NOT ROCSHMEM_PACKAGE_DTK_VERSION)
  if(DEFINED ENV{ROCSHMEM_PACKAGE_DTK_VERSION} AND
     NOT "$ENV{ROCSHMEM_PACKAGE_DTK_VERSION}" STREQUAL "")
    set(ROCSHMEM_PACKAGE_DTK_VERSION "$ENV{ROCSHMEM_PACKAGE_DTK_VERSION}")
  elseif(EXISTS "${ROCM_PATH}/.dtk_version")
    file(STRINGS "${ROCM_PATH}/.dtk_version" _dtk_version_text LIMIT_COUNT 1)
    if(_dtk_version_text MATCHES "^DTK-([0-9]+\\.[0-9]+(\\.[0-9]+)?)")
      string(REPLACE "." "" ROCSHMEM_PACKAGE_DTK_VERSION "${CMAKE_MATCH_1}")
    endif()
  endif()
endif()
if(NOT ROCSHMEM_PACKAGE_DTK_VERSION MATCHES "^[0-9]+$")
  message(FATAL_ERROR
    "Unable to determine the DTK package version. Set "
    "-DROCSHMEM_PACKAGE_DTK_VERSION=<major><minor>[patch] "
    "(for example, 2604 or 26042).")
endif()

string(TIMESTAMP ROCSHMEM_PACKAGE_TIMESTAMP "%y%m%d%H%M")
if(ROCSHMEM_GIT_HASH STREQUAL "unknown")
  set(ROCSHMEM_PACKAGE_GIT_SHORT "unknown")
else()
  string(SUBSTRING "${ROCSHMEM_GIT_HASH}" 0 8 ROCSHMEM_PACKAGE_GIT_SHORT)
endif()

string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" ROCSHMEM_PACKAGE_ARCH)
if(ROCSHMEM_PACKAGE_ARCH MATCHES "^(amd64|x64)$")
  set(ROCSHMEM_PACKAGE_ARCH "x86_64")
elseif(ROCSHMEM_PACKAGE_ARCH STREQUAL "arm64")
  set(ROCSHMEM_PACKAGE_ARCH "aarch64")
endif()
if(ROCSHMEM_PACKAGE_ARCH STREQUAL "x86_64")
  set(ROCSHMEM_PACKAGE_DEB_ARCH "amd64")
elseif(ROCSHMEM_PACKAGE_ARCH STREQUAL "aarch64")
  set(ROCSHMEM_PACKAGE_DEB_ARCH "arm64")
else()
  set(ROCSHMEM_PACKAGE_DEB_ARCH "${ROCSHMEM_PACKAGE_ARCH}")
endif()

set(_rocshmem_package_name "rocshmem")
if(GDA_SHCA)
  set(_rocshmem_package_name "rocshmem_shca")
endif()
string(CONCAT ROCSHMEM_PACKAGE_FILE_BASE
  "${_rocshmem_package_name}-${VERSION_STRING}+dtk${ROCSHMEM_PACKAGE_DTK_VERSION}."
  "${ROCSHMEM_PACKAGE_TIMESTAMP}.g${ROCSHMEM_PACKAGE_GIT_SHORT}")
set(CPACK_PACKAGE_FILE_NAME
  "${ROCSHMEM_PACKAGE_FILE_BASE}_${ROCSHMEM_PACKAGE_ARCH}")
configure_file(
  "${CMAKE_SOURCE_DIR}/cmake/CPackProjectConfig.cmake.in"
  "${CMAKE_BINARY_DIR}/CPackProjectConfig.cmake"
  @ONLY
)
set(CPACK_PROJECT_CONFIG_FILE
  "${CMAKE_BINARY_DIR}/CPackProjectConfig.cmake")

# Append a self-extracting .run installer after CPack creates the TGZ.
string(REGEX REPLACE "^/+" "" _run_install_prefix "${CMAKE_INSTALL_PREFIX}")
string(REPLACE "/" ";" _run_install_prefix_parts "${_run_install_prefix}")
list(FILTER _run_install_prefix_parts EXCLUDE REGEX "^$")
list(LENGTH _run_install_prefix_parts _run_install_prefix_depth)
# CPack's TGZ contains one package-name directory followed by the absolute
# install prefix (for example: package-name/opt/rocshmem/...).
math(EXPR ROCSHMEM_TGZ_STRIP_COMPONENTS "${_run_install_prefix_depth} + 1")
configure_file(
  "${CMAKE_SOURCE_DIR}/cmake/rocshmem.run.in"
  "${CMAKE_BINARY_DIR}/rocshmem.run.stub"
  @ONLY
  NEWLINE_STYLE UNIX
)
configure_file(
  "${CMAKE_SOURCE_DIR}/cmake/MakeRunPackage.cmake.in"
  "${CMAKE_BINARY_DIR}/MakeRunPackage.cmake"
  @ONLY
)
list(APPEND CPACK_POST_BUILD_SCRIPTS
  "${CMAKE_BINARY_DIR}/MakeRunPackage.cmake"
)
