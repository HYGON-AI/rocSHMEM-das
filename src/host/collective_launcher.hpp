// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#ifndef LIBRARY_SRC_HOST_COLLECTIVE_LAUNCHER_HPP_
#define LIBRARY_SRC_HOST_COLLECTIVE_LAUNCHER_HPP_

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <mutex>
#include <limits>
#include <unordered_map>
#include <vector>

#include "rocshmem/rocshmem_COLL.hpp"
#include "rocshmem/rocshmem.hpp"
#include "../team.hpp"
#include "../envvar.hpp"
#include "../util.hpp"

namespace rocshmem {

/**
 * Launches collective kernels on user-supplied streams and manages the
 * duplicate-team resources used by the multi-workgroup path.
 *
 * Calls on the same team are ordered through a completion event. Different
 * teams retain independent execution on user streams.
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
   * workgroup. Message size determines the count within configured limits.
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
   * Select one workgroup per chunk, subject to configured WG/team/context limits.
   */
  static size_t compute_num_wgs(size_t message_bytes) {
    const size_t bytes_per_wg = kBytesPerWg;
    const size_t chunk_count = message_bytes / bytes_per_wg +
                              (message_bytes % bytes_per_wg != 0);
    const size_t configured_wgs = static_cast<size_t>(std::max(1, envvar::max_num_wgs_on_stream.get_value()));
    // Reserve TEAM_WORLD's slot. The shared team pool can still exhaust,
    // in which case team creation triggers the single-WG fallback.
    const size_t max_teams = envvar::max_num_teams.get_value();
    return std::max(size_t{1}, std::min({chunk_count, configured_wgs,
                                         max_teams > 1 ? max_teams - 1 : size_t{1},
                                         envvar::max_num_contexts.get_value()}));
  }

  static bool use_single_wg(rocshmem_team_t team, size_t size) {
    if (get_internal_team(team)->type == BackendType::RO_BACKEND) {
      return true;
    }
    if (size < kMultiBlockThreshold) {
      return true;
    }
    return compute_num_wgs(size) <= 1;
  }

  // Device counts are deliberately not copied to the host for alltoallv.
  static constexpr size_t kUnknownWorkBytes = std::numeric_limits<size_t>::max();

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
  template <auto Kernel, typename... Args>
  static void enqueue_single_wg(size_t launch_threads_size,
                                hipStream_t stream, int min_threads,
                                Args... args) {
    const int limit = std::max(1, envvar::num_threads_on_stream.get_value());
    const int wavefront = wavefront_threads();
    int threads = limit;
    if (launch_threads_size < static_cast<size_t>(threads)) {
      threads = static_cast<int>(launch_threads_size);
    }
    min_threads = std::clamp(min_threads, 1, limit);
    threads = std::max(threads, min_threads);
    threads = std::min(limit, std::max(wavefront,
                       ((threads + wavefront - 1) / wavefront) * wavefront));
    Kernel<<<dim3(1), dim3(threads), 0, stream>>>(args...);
    CHECK_HIP(hipGetLastError());
  }

  static constexpr size_t kMultiBlockThreshold = 256 * 1024;
  static constexpr size_t kBytesPerWg = 32 * 1024;

  // Algorithm chunking is independent of launch selection granularity.
  static constexpr size_t kChunkSize = 256 * 1024;

  static int wavefront_threads() {
    int device_id = 0;
    CHECK_HIP(hipGetDevice(&device_id));
    return get_wf_size(device_id);
  }

  struct ParallelResources {
    ~ParallelResources() {
      if (completion != nullptr) {
        CHECK_HIP(hipEventDestroy(completion));
      }
      if (dev_wg_teams != nullptr) {
        hipFree(dev_wg_teams);
      }
    }

    hipEvent_t completion{};
    bool completion_recorded{};

    // Pre-split sub-teams, one per workgroup, for multi-wg kernels.
    std::vector<rocshmem_team_t> wg_teams;
    rocshmem_team_t* dev_wg_teams{};
    size_t uploaded_wg_teams{};
    // Team capacity is shared globally. Once creation fails for this parent,
    // use the single-WG fallback until the parent team is destroyed instead
    // of repeating the same expensive split attempts on every API call.
    bool wg_team_creation_failed{};
  };

  // Order on-stream collectives, including barriers and synchronization.
  // Keep wait, enqueue (including fallback/RCCL), and record atomic on the host.
  // All PEs must submit collectives on a team in the same logical order.
  class TeamStreamOrder {
   public:
    TeamStreamOrder(rocshmem_team_t team, hipStream_t stream)
        : lock_(resources_mutex()), stream_(stream),
          resources_(parallel_resources_for(team)) {
      hipStreamCaptureStatus capture;
      CHECK_HIP(hipStreamIsCapturing(stream, &capture));
      if (capture != hipStreamCaptureStatusNone) {
        CHECK_HIP(hipErrorStreamCaptureUnsupported);
      }
      if (resources_.completion == nullptr) {
        CHECK_HIP(hipEventCreateWithFlags(&resources_.completion,
                                          hipEventDisableTiming));
      }
      if (resources_.completion_recorded) {
        // Also safe when a destroyed stream's handle is reused by HIP.
        CHECK_HIP(hipStreamWaitEvent(stream_, resources_.completion, 0));
      }
    }

    ~TeamStreamOrder() {
      CHECK_HIP(hipEventRecord(resources_.completion, stream_));
      resources_.completion_recorded = true;
    }

    TeamStreamOrder(const TeamStreamOrder&) = delete;
    TeamStreamOrder& operator=(const TeamStreamOrder&) = delete;

   private:
    std::unique_lock<std::recursive_mutex> lock_;
    hipStream_t stream_;
    ParallelResources& resources_;
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
   * handles are uploaded. Existing teams are reused, and device storage is
   * allocated once at the configured maximum capacity.
   */
  static void ensure_wg_teams(rocshmem_team_t parent_team,
                              size_t team_size, size_t required_num_wgs,
                              ParallelResources& resources) {
    if (resources.wg_team_creation_failed) {
      return;
    }

    // Expansion may fail and destroy existing children. Drain the previous
    // launch before changing their lifetime, including across user streams.
    if (resources.wg_teams.size() < required_num_wgs) {
      if (!resources.wg_teams.empty() && resources.completion_recorded) {
        CHECK_HIP(hipEventSynchronize(resources.completion));
      }
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
            resources.wg_team_creation_failed = true;
            return;  // Caller will see an empty team list and may fall back.
          }
        }
      }
    }
    // Environment values are cached, so the capacity stays fixed for the
    // lifetime of these resources. Outstanding kernels retain a stable pointer.
    if (resources.dev_wg_teams == nullptr) {
      CHECK_HIP(hipMalloc(&resources.dev_wg_teams,
                         sizeof(rocshmem_team_t) * static_cast<size_t>(
                             std::max(1, envvar::max_num_wgs_on_stream.get_value()))));
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
