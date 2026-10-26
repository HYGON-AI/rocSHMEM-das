# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT

if(GDA_ZXDH)
target_sources(
  ${PROJECT_NAME}
  PRIVATE
    backend_gda_zxdh.cpp
    queue_pair_zxdh.cpp
    queue_pair_zxdh_multiqp.cpp
    segment_builder_zxdh.cpp
)
endif()
