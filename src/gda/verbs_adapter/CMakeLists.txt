# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT

# Compile each SDK separately; embed host-only objects in the existing libraries.
foreach(provider mlx5 shca)
  set(adapter rocshmem_verbs_${provider})
  add_library(${adapter} OBJECT verbs_adapter_${provider}.cpp)
  target_include_directories(${adapter} PRIVATE "${CMAKE_BINARY_DIR}/include")
  target_compile_options(${adapter} PRIVATE
    --offload-host-only -fvisibility=hidden -fvisibility-inlines-hidden -fno-lto)
  set_target_properties(${adapter} PROPERTIES POSITION_INDEPENDENT_CODE ON)
  target_sources(${PROJECT_NAME} PRIVATE $<TARGET_OBJECTS:${adapter}>)
endforeach()
