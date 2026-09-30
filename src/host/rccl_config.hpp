// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#ifndef LIBRARY_SRC_HOST_RCCL_CONFIG_HPP_
#define LIBRARY_SRC_HOST_RCCL_CONFIG_HPP_

#include "rocshmem/rocshmem_config.h"

#if defined(USE_RCCL)
#include <array>
#include <cstddef>
#include <cstdint>

namespace rocshmem {

enum class RcclCollective : size_t {
  BROADCAST = 0,
  ALL_REDUCE = 1,
  REDUCE_SCATTER = 2,
  ALL_GATHER = 3,
  REDUCE = 4,
  GATHER = 5,
  SCATTER = 6,
  ALL_TO_ALL = 7,
  ALL_TO_ALL_V = 8,
  NUM_COLLECTIVES = 9
};

inline constexpr size_t kRcclCollectiveCount =
    static_cast<size_t>(RcclCollective::NUM_COLLECTIVES);

enum class RcclScope : size_t { INTRA_NODE = 0, INTER_NODE = 1 };

struct RcclRuntimeConfig {
  std::array<std::array<uint64_t, kRcclCollectiveCount>, 2> min_sizes{};
  int disabled{0};
  int delay_init{0};
};

const RcclRuntimeConfig& rccl_runtime_config();

}  // namespace rocshmem
#endif  // USE_RCCL

#endif  // LIBRARY_SRC_HOST_RCCL_CONFIG_HPP_

