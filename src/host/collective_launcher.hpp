// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#ifndef LIBRARY_SRC_HOST_COLLECTIVE_LAUNCHER_HPP_
#define LIBRARY_SRC_HOST_COLLECTIVE_LAUNCHER_HPP_

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "rocshmem/rocshmem_COLL.hpp"
#include "rocshmem/rocshmem.hpp"
#include "../team.hpp"
#include "../util.hpp"

namespace rocshmem {

/**
 * Launches collective kernels on user-supplied streams and manages the
 * duplicate-team resources used by the multi-workgroup path.
 *
 * Single-block collective kernels are launched directly on the user stream so
 * that kernels targeting *different* teams (or different user streams) run
 * concurrently — they synchronize internally via per-team pSync and do not
 * need a shared progress stream.
 */
class CollectiveLauncher {
 public:
  template <typename TeamDestroyer>
  static void release_team(rocshmem_team_t team, TeamDestroyer destroy_team) {
    std::lock_guard<std::recursive_mutex> lock(resources_mutex());
    auto& resources = resources_map();
    auto it = resources.find(team);
    if (it == resources.end()) {
      return;
    }

    // Team destruction is an explicit lifecycle boundary. Wait here so no
    // asynchronous collective can retain a duplicate-team handle after it is
    // destroyed; normal collective enqueue remains asynchronous.
    CHECK_HIP(hipDeviceSynchronize());
    destroy_wg_teams(it->second, destroy_team);
    resources.erase(it);
  }

  template <typename TeamDestroyer>
  static void release_all(TeamDestroyer destroy_team) {
    std::lock_guard<std::recursive_mutex> lock(resources_mutex());
    auto& resources = resources_map();
    if (resources.empty()) {
      return;
    }

    CHECK_HIP(hipDeviceSynchronize());
    for (auto& entry : resources) {
      destroy_wg_teams(entry.second, destroy_team);
    }
    resources.clear();
  }

  /**
   * Enqueue a multi-workgroup collective using one duplicate team per
   * workgroup. The workgroup count is derived from `message_bytes`.
   *
   * The caller-supplied `launch` callback receives the device team handles,
   * the stream, and the computed `num_wgs`:
   *
   *   launch(dev_wg_teams, stream, num_wgs)
   *
   * Returns false if sub-team creation failed; caller should fall back.
   *
   * Resource initialization and launch are serialized so the resource map,
   * team vector, and device handle array cannot be modified while an
   * asynchronous kernel is being enqueued from another host thread.
   */
  template <typename Launcher>
  static bool enqueue_multi_wg(rocshmem_team_t team, size_t message_bytes,
                               hipStream_t stream, Launcher launch) {
    size_t num_wgs = compute_num_wgs(message_bytes);
    size_t team_size = static_cast<size_t>(rocshmem_team_n_pes(team));

    std::lock_guard<std::recursive_mutex> lock(resources_mutex());
    ParallelResources& resources = parallel_resources_for(team);
    ensure_wg_teams(team, team_size, num_wgs, resources);

    if (resources.wg_teams.empty() ||
        resources.wg_teams[0] == ROCSHMEM_TEAM_INVALID) {
      return false;
    }

    launch(resources.dev_wg_teams, stream, num_wgs);
    return true;
  }

  /**
   * Compute the workgroup count from the collective message size. Each
   * workgroup targets roughly kChunkSize bytes, subject to configured limits.
   */
  static size_t compute_num_wgs(size_t message_bytes) {
    size_t chunk_count = (message_bytes + kChunkSize - 1) / kChunkSize;
    size_t num_wgs = chunk_count;

    // Optional override: ROCSHMEM_NUM_WGS_ON_STREAM forces a specific workgroup count.
    // It takes precedence but remains subject to the limits below.
    if (const char* env_wgs = std::getenv("ROCSHMEM_NUM_WGS_ON_STREAM")) {
      size_t user_num_wgs = static_cast<size_t>(atoi(env_wgs));
      if (user_num_wgs > 0) {
        num_wgs = user_num_wgs;
      }
    }

    // Cap by available teams (ROCSHMEM_MAX_NUM_TEAMS env var, default 40).
    // Leave one configured slot for TEAM_WORLD. The team pool is shared by
    // all parent teams, so creation can still fail and trigger fallback.
    const char* env = std::getenv("ROCSHMEM_MAX_NUM_TEAMS");
    size_t max_teams = env ? static_cast<size_t>(atoi(env)) : 40;
    if (max_teams > 1) max_teams -= 1;
    num_wgs = std::min(num_wgs, max_teams);
    num_wgs = std::min(num_wgs, kMaxParallelWgs);
    if (num_wgs < 1) num_wgs = 1;
    return num_wgs;
  }

  static bool use_single_wg(rocshmem_team_t team, size_t size) {
    if (get_internal_team(team)->type == BackendType::RO_BACKEND) {
      return true;
    }
    if (size < kMultiBlockThreshold) {
      return true;
    }
    const char* env_wgs = std::getenv("ROCSHMEM_NUM_WGS_ON_STREAM");
    return env_wgs != nullptr && atoi(env_wgs) == 1;
  }

  /**
   * Common one-block collective path.
   *
   * Launches the kernel directly on the user stream.  Single-block
   * collective kernels (alltoallmem, broadcastmem, ...) are self-contained:
   * they synchronize internally via per-team pSync, so kernels targeting
   * *different* teams can run concurrently on different streams.  This
   * preserves the multi-team concurrency that tests like
   * team_alltoallmem_on_stream_tester rely on.
   */
  template <auto Kernel, typename DestPtr, typename SrcPtr, typename... Args>
  static void enqueue_single_wg(rocshmem_team_t team, DestPtr dest,
                                SrcPtr source, size_t size,
                                hipStream_t stream, Args... args) {
    int optimal_block_size = 0;
    int unused_grid_size = 0;
    CHECK_HIP(hipOccupancyMaxPotentialBlockSize(&unused_grid_size, &optimal_block_size, Kernel, 0, 0));

    int threads = std::min(optimal_block_size, kMaxThreads);
    if (size < static_cast<size_t>(threads)) {
      threads = static_cast<int>(size);
    }
    threads = std::max(threads, 1);
    Kernel<<<dim3(1), dim3(threads), 0, stream>>>(team, dest, source, size, args...);
    CHECK_HIP(hipGetLastError());
  }

  static constexpr int kMaxThreads = 256;
  static constexpr size_t kMaxParallelWgs = 32;
  static constexpr size_t kMultiBlockThreshold = 1024 * 1024;  // 1MB
  static constexpr size_t kChunkSize = 256 * 1024;

  struct ParallelResources {
    ~ParallelResources() {
      if (dev_wg_teams != nullptr) {
        hipFree(dev_wg_teams);
      }
    }

    // Pre-split sub-teams, one per workgroup, for multi-wg kernels.
    std::vector<rocshmem_team_t> wg_teams;
    rocshmem_team_t* dev_wg_teams{};
    size_t uploaded_wg_teams{};
  };

 private:
  using ResourcesMap = std::unordered_map<rocshmem_team_t, ParallelResources>;

  static ResourcesMap& resources_map() {
    static ResourcesMap resources;
    return resources;
  }

  template <typename TeamDestroyer>
  static void destroy_wg_teams(ParallelResources& resources, TeamDestroyer destroy_team) {
    for (rocshmem_team_t team : resources.wg_teams) {
      if (team != ROCSHMEM_TEAM_INVALID) {
        destroy_team(team);
      }
    }
  }

  static ParallelResources& parallel_resources_for(rocshmem_team_t team) {
    return resources_map().try_emplace(team).first->second;
  }

  static std::recursive_mutex& resources_mutex() {
    static std::recursive_mutex mutex;
    return mutex;
  }

  /**
   * Ensure sub-teams are created (one per workgroup) and their device
   * handles are uploaded. Safe to call repeatedly: existing teams are reused,
   * and device storage is allocated once at the fixed maximum capacity.
   */
  static void ensure_wg_teams(rocshmem_team_t parent_team,
                              size_t team_size, size_t required_num_wgs,
                              ParallelResources& resources) {
    // Create host-side sub-teams if needed.
    if (resources.wg_teams.size() < required_num_wgs) {
      resources.wg_teams.resize(required_num_wgs, ROCSHMEM_TEAM_INVALID);
      for (size_t i = 0; i < required_num_wgs; ++i) {
        if (resources.wg_teams[i] == ROCSHMEM_TEAM_INVALID) {
          int status = rocshmem_team_split_strided(
              parent_team, 0, 1, static_cast<int>(team_size), nullptr, 0,
              &resources.wg_teams[i]);
          if (status != 0 ||
              resources.wg_teams[i] == ROCSHMEM_TEAM_INVALID) {
            // Team split failed — free any successfully created sub-teams
            // so they can be retried on the next call.
            for (size_t j = 0; j <= i; ++j) {
              if (resources.wg_teams[j] != ROCSHMEM_TEAM_INVALID) {
                rocshmem_team_destroy(resources.wg_teams[j]);
                resources.wg_teams[j] = ROCSHMEM_TEAM_INVALID;
              }
            }
            resources.wg_teams.resize(0);
            resources.uploaded_wg_teams = 0;
            return;  // Caller will see an empty team list and may fall back.
          }
        }
      }
    }
    // Allocate the fixed maximum once. The pointer is never replaced while
    // asynchronous kernels may reference it, so no device-wide synchronize is
    // required on the on-stream path.
    if (resources.dev_wg_teams == nullptr) {
      CHECK_HIP(hipMalloc(&resources.dev_wg_teams, sizeof(rocshmem_team_t) * kMaxParallelWgs));
    }
    if (resources.uploaded_wg_teams < required_num_wgs) {
      size_t first = resources.uploaded_wg_teams;
      size_t count = required_num_wgs - first;
      CHECK_HIP(hipMemcpy(resources.dev_wg_teams + first,
                          resources.wg_teams.data() + first,
                          sizeof(rocshmem_team_t) * count,
                          hipMemcpyHostToDevice));
      resources.uploaded_wg_teams = required_num_wgs;
    }
  }
};

}  // namespace rocshmem

#endif  // LIBRARY_SRC_HOST_COLLECTIVE_LAUNCHER_HPP_
