# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
target_sources(
  ${PROJECT_NAME}
  PRIVATE
    backend_gda_shca.cpp
    queue_pair_shca.cpp
    segment_builder_shca.cpp
    queue_pair_shca_multiqp.cpp
)
