// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#pragma once

#if defined(USE_DAS)

namespace rocshmem {

namespace proxy {

// An epoch identifies one complete synchronization phase, not one IPC domain or PE.
constexpr int kWorldBroadcastSyncPhases = 3;

// The same staged-transfer/fan-out protocol serves all WG teams.
__device__ inline int TeamBroadcastPhaseCount([[maybe_unused]] const GDATeam *team) {
  return kWorldBroadcastSyncPhases;
}

}  // namespace proxy

// Convert a byte policy to an element count, bounded by the caller's range or
// scratch capacity. Budgeted policies preserve single-WG and IPC paths.
template <typename T>
__device__ __forceinline__ int GDAContext::collective_chunk_elems(int capacity, CollectiveChunkPolicy policy,
                                                                  bool uses_network) {
  size_t chunk_bytes = policy.max_bytes;

  if (policy.concurrent_bytes != 0) {
    const size_t wg_count = static_cast<size_t>(gridDim.x) * gridDim.y * gridDim.z;
    if (wg_count <= 1 || !uses_network) return capacity;
    chunk_bytes = min(chunk_bytes, policy.concurrent_bytes / wg_count);
  }

  return min(capacity, static_cast<int>(max(size_t{1}, chunk_bytes / sizeof(T))));
}

}  // namespace rocshmem

#endif  // USE_DAS
