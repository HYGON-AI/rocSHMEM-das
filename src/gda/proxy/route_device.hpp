// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#ifndef LIBRARY_SRC_GDA_PROXY_ROUTE_DEVICE_HPP_
#define LIBRARY_SRC_GDA_PROXY_ROUTE_DEVICE_HPP_

namespace rocshmem::proxy {

/** @brief Match WORLD membership, including teams with the same members. */
__device__ __forceinline__ bool IsWorldMembership(int pe_start, int stride, int pe_size) {
  return pe_start == 0 && stride == 1 && pe_size == constmem.num_pes;
}

/** @brief Return whether a team has one selected proxy in multiple IPC domains. */
__device__ __forceinline__ bool HasInTeamProxyRoute(const GDATeam *team) {
  return team != nullptr && team->in_team_proxy_ipc_domain_count > 1;
}

/**
 * @brief Check whether the calling PE can use WORLD proxy synchronization.
 * @return true when WORLD membership and IPC layout match the host plan.
 */
__device__ __forceinline__ bool IsWorldProxySyncEligible(int pe, int pe_start, int stride, int pe_size) {
  const int local_pe_count = constmem.ipc_shm_size;
  return constmem.world_proxy_sync_available && pe_start == 0 && stride == 1 && pe_size == constmem.num_pes &&
         local_pe_count > 0 && (local_pe_count == 1 || constmem.ipc_stride == 1) &&
         constmem.world_proxy_ipc_domain_count > 1 && pe >= constmem.ipc_first_pe &&
         pe < constmem.ipc_first_pe + local_pe_count;
}

/**
 * @brief Check whether aligned WORLD data routing is safe for this team.
 * @return true when corresponding IPC slots own matching selected NICs.
 */
__device__ __forceinline__ bool IsAlignedWorldProxyDataEligible(int pe_start, int stride, int pe_size) {
  const int local_pe_count = constmem.ipc_shm_size;
  return constmem.world_proxy_aligned_data_available && IsWorldMembership(pe_start, stride, pe_size) &&
         local_pe_count > 1 && constmem.ipc_stride == 1 && pe_size % local_pe_count == 0 &&
         constmem.ipc_first_pe == (constmem.my_pe / local_pe_count) * local_pe_count;
}

enum class ProxyCollectiveLayout {
  None,              // No eligible PXN route; not a non-PXN mode.
  TeamProxyRoute,    // Partial team: one same-NIC team proxy per IPC domain.
  WorldAlignedData,  // WORLD: corresponding local PEs have matching NICs.
  WorldSyncOnly,     // WORLD: sync route exists; data NICs may not align.
};

/**
 * @brief Classify PXN routing guarantees without choosing an operation algorithm.
 *
 * @param team Team metadata, or nullptr for an internal WORLD operation.
 * @param pe_start First world PE in the operation.
 * @param stride World-rank stride between operation members.
 * @param pe_size Number of operation members.
 * @return The strongest available proxy layout; algorithms may impose
 *         additional message-size and scratch-capacity requirements.
 */
__device__ __forceinline__ ProxyCollectiveLayout SelectProxyCollectiveLayout(const GDATeam *team, int pe_start,
                                                                             int stride, int pe_size) {
  if (IsWorldMembership(pe_start, stride, pe_size)) {
    if (IsAlignedWorldProxyDataEligible(pe_start, stride, pe_size)) {
      return ProxyCollectiveLayout::WorldAlignedData;
    }

    if (IsWorldProxySyncEligible(constmem.my_pe, pe_start, stride, pe_size)) {
      return ProxyCollectiveLayout::WorldSyncOnly;
    }
    return ProxyCollectiveLayout::None;
  }
  return HasInTeamProxyRoute(team) ? ProxyCollectiveLayout::TeamProxyRoute : ProxyCollectiveLayout::None;
}

// Membership and routing capabilities are independent of the collective algorithm.
struct ProxyTeamView {
  const GDATeam *team;
  bool is_world;
  int ipc_domain_count;

  __device__ int domain_first(int domain) const {
    return is_world ? constmem.world_proxy_ipc_domain_first_pes[domain]
                    : team->in_team_proxy_ipc_domain_first_pes[domain];
  }
};

__device__ inline ProxyTeamView GetProxyTeamView(const GDATeam *team, int pe_start, int stride, int pe_size) {
  const bool world = IsWorldMembership(pe_start, stride, pe_size);
  return {team, world,
          world ? constmem.world_proxy_ipc_domain_count : (team != nullptr ? team->in_team_proxy_ipc_domain_count : 0)};
}

struct ProxyDataRoute {
  int source_pe{-1};
  int target_pe{-1};
};

// WORLD and partial teams differ only in candidate membership. Fixed endpoints
// preserve tasks whose input or final output already belongs to a particular PE.
struct ProxyDataRouteCandidates {
  const GDATeam *team;
  int source_first;
  int source_end;
  int target_first;
  int target_end;
  bool require_matching_nic;
  int fixed_source{-1};
  int fixed_target{-1};
};

__device__ __forceinline__ bool IsProxyDataMember(const GDATeam *team, int pe) {
  if (team == nullptr) {
    return true;
  }

  const auto *info = team->tinfo_wrt_world;
  const int distance = pe - info->pe_start;

  return info->stride > 0 && distance >= 0 && distance % info->stride == 0 &&
         distance / info->stride < info->size;
}

__device__ inline int CountProxyDataEndpoints(const GDATeam *team, int first, int end, int nic, int fixed_pe) {
  int count = 0;

  for (int pe = first; pe < end; pe++) {
    count += (fixed_pe < 0 || fixed_pe == pe) && IsProxyDataMember(team, pe) &&
             (nic < 0 || constmem.world_proxy_pe_nic_name_ids[pe] == nic);
  }

  return count;
}

__device__ inline int SelectProxyDataEndpoint(const GDATeam *team, int first, int end,
                                              int nic, int fixed_pe, uint64_t ordinal) {
  for (int pe = first; pe < end; pe++) {
    if ((fixed_pe < 0 || fixed_pe == pe) && IsProxyDataMember(team, pe) &&
        (nic < 0 || constmem.world_proxy_pe_nic_name_ids[pe] == nic)) {
      if (ordinal == 0) {
        return pe;
      }
      ordinal--;
    }
  }

  return -1;
}

// Equal turns per NIC, then equal turns among that NIC's endpoint PEs. The
// index identifies an existing task/operation, never a PE-local counter.
__device__ __forceinline__ ProxyDataRoute SelectProxyDataRouteImpl(const GDATeam *team, int source_first,
                                                                   int source_end, int target_first, int target_end,
                                                                   bool require_matching_nic, int fixed_source,
                                                                   int fixed_target, uint64_t transfer_index) {
  if (!constmem.world_proxy_nic_layout_available || source_first < 0 || target_first < 0 ||
      source_end > constmem.num_pes || target_end > constmem.num_pes) {
    return {};
  }

  if (require_matching_nic && fixed_target >= 0) {
    if (fixed_target < target_first || fixed_target >= target_end || !IsProxyDataMember(team, fixed_target)) {
      return {};
    }

    const int nic = constmem.world_proxy_pe_nic_name_ids[fixed_target];
    const int sources = CountProxyDataEndpoints(team, source_first, source_end, nic, fixed_source);
    if (sources == 0) {
      return {};
    }

    return {SelectProxyDataEndpoint(team, source_first, source_end, nic, fixed_source, transfer_index % sources),
            fixed_target};
  }

  const int nics = constmem.world_proxy_nic_name_count;
  const int pairs = require_matching_nic ? nics : nics * nics;
  int groups = 0;

  for (int pair = 0; pair < pairs; pair++) {
    const int source_nic = require_matching_nic ? pair : pair / nics;
    const int target_nic = require_matching_nic ? pair : pair % nics;
    const int sources = CountProxyDataEndpoints(team, source_first, source_end, source_nic, fixed_source);
    const int targets = CountProxyDataEndpoints(team, target_first, target_end, target_nic, fixed_target);
    groups += sources > 0 && targets > 0;
  }

  if (groups == 0) {
    return {};
  }

  int selected = transfer_index % groups;
  const uint64_t round = transfer_index / groups;

  for (int pair = 0; pair < pairs; pair++) {
    const int source_nic = require_matching_nic ? pair : pair / nics;
    const int target_nic = require_matching_nic ? pair : pair % nics;
    const int sources = CountProxyDataEndpoints(team, source_first, source_end, source_nic, fixed_source);
    const int targets = CountProxyDataEndpoints(team, target_first, target_end, target_nic, fixed_target);

    if (sources == 0 || targets == 0) {
      continue;
    }

    if (selected-- == 0) {
      return {SelectProxyDataEndpoint(team, source_first, source_end, source_nic, fixed_source, round % sources),
              SelectProxyDataEndpoint(team, target_first, target_end, target_nic, fixed_target, round % targets)};
    }
  }

  return {};
}

__device__ __forceinline__ ProxyDataRoute SelectProxyDataRoute(const ProxyDataRouteCandidates &candidates,
                                                               uint64_t transfer_index) {
  return SelectProxyDataRouteImpl(candidates.team, candidates.source_first, candidates.source_end,
                                  candidates.target_first, candidates.target_end, candidates.require_matching_nic,
                                  candidates.fixed_source, candidates.fixed_target, transfer_index);
}

/** @brief Return the exclusive world-PE end of an IPC domain. */
__device__ __forceinline__ int WorldProxyIpcDomainEnd(int ipc_domain_first_pe) {
  for (int ipc_domain = 0; ipc_domain < constmem.world_proxy_ipc_domain_count; ipc_domain++) {
    if (constmem.world_proxy_ipc_domain_first_pes[ipc_domain] == ipc_domain_first_pe) {
      return ipc_domain + 1 < constmem.world_proxy_ipc_domain_count
                 ? constmem.world_proxy_ipc_domain_first_pes[ipc_domain + 1]
                 : constmem.num_pes;
    }
  }
  return -1;
}

// Build data candidates without modifying the fixed synchronization layout.
__device__ __forceinline__ ProxyDataRoute SelectTeamProxyDataRoute(const GDATeam *team, int source_first,
                                                                   int target_first, uint64_t transfer_index) {
  const int source_end = WorldProxyIpcDomainEnd(source_first);
  const int target_end = WorldProxyIpcDomainEnd(target_first);
  if (source_end > source_first && target_end > target_first && constmem.world_proxy_nic_layout_available) {
    return SelectProxyDataRoute({team, source_first, source_end, target_first, target_end, constmem.pxn_enabled},
                                transfer_index);
  }
  // Multi-NIC-per-PE configurations still need an explicit NIC-to-QP map.
  // Retain their host-validated single route instead of guessing a QP's NIC.
  ProxyDataRoute route;

  const bool world = team == nullptr || IsWorldMembership(team->tinfo_wrt_world->pe_start,
                                                          team->tinfo_wrt_world->stride,
                                                          team->tinfo_wrt_world->size);
  if (world && constmem.world_proxy_sync_available && source_end > source_first && target_end > target_first) {
    return {source_first, target_first};
  }

  if (team != nullptr) {
    for (int domain = 0; domain < team->in_team_proxy_ipc_domain_count; domain++) {
      if (team->in_team_proxy_ipc_domain_first_pes[domain] == source_first) {
        route.source_pe = team->in_team_proxy_pes[domain];
      }

      if (team->in_team_proxy_ipc_domain_first_pes[domain] == target_first) {
        route.target_pe = team->in_team_proxy_pes[domain];
      }
    }
  }
  return route;
}

// Packing/reduction owners are data proxies, independent of fixed sync proxies.
__device__ inline int SelectCollectiveDomainProxy(const ProxyTeamView &view, int domain, uint64_t operation) {
  const int first = view.domain_first(0);
  const int peer = view.domain_first(domain == 0 ? 1 : domain);
  const auto route = SelectTeamProxyDataRoute(view.team, first, peer, operation);
  return domain == 0 ? route.source_pe : route.target_pe;
}

/** @brief Count PEs owning a NIC ID within a world-PE interval. */
__device__ __forceinline__ int CountNicProxies(int first, int end, int nic_name_id) {
  int count = 0;

  for (int pe = first; pe < end; pe++) {
    count += constmem.world_proxy_pe_nic_name_ids[pe] == nic_name_id;
  }

  return count;
}

/** @brief Source proxy and scratch batch assigned to an all-to-all target. */
struct WorldAlltoallTargetAssignment {
  int source_proxy_pe{-1};  // Same-NIC source PE, or -1 if unreachable.
  int scratch_batch{-1};    // Reused scratch slot group, or -1 if unreachable.
};

/**
 * @brief Select the same-NIC source proxy for one all-to-all target.
 * @return Source proxy and scratch batch, or invalid values when unreachable.
 */
__device__ __forceinline__ WorldAlltoallTargetAssignment SelectWorldAlltoallTargetAssignment(
    int source_ipc_domain_first_pe, int target_ipc_domain_first_pe, int target_pe, uint64_t operation_index = 0) {
  const int nic_name_id = constmem.world_proxy_pe_nic_name_ids[target_pe];
  const int source_end = WorldProxyIpcDomainEnd(source_ipc_domain_first_pe);
  const int source_count = CountNicProxies(source_ipc_domain_first_pe, source_end, nic_name_id);
  if (source_count == 0) {
    return {};
  }

  int target_ordinal = 0;
  for (int pe = target_ipc_domain_first_pe; pe < target_pe; pe++) {
    target_ordinal += constmem.world_proxy_pe_nic_name_ids[pe] == nic_name_id;
  }

  // The destination owns the output: keep that endpoint fixed to avoid an
  // extra receive buffer/copy, and rotate eligible same-NIC source proxies.
  const auto route = SelectProxyDataRoute({nullptr, source_ipc_domain_first_pe, source_end, target_ipc_domain_first_pe,
                                           WorldProxyIpcDomainEnd(target_ipc_domain_first_pe), true, -1, target_pe},
                                          operation_index + target_ordinal);
  return {route.source_pe, target_ordinal / source_count};
}

}  // namespace rocshmem::proxy

#if defined(USE_DAS)

namespace rocshmem {

// Shared proxy route plans.
// Share one type-independent plan across fcollect and both reduction outputs.
__device__ inline __noinline__ const int *GDAContext::internal_build_collective_proxy_plan_wg(GDATeam *team_obj,
                                                                                              uint64_t operation) {
  const auto *info = team_obj->tinfo_wrt_world;
  const auto view = proxy::GetProxyTeamView(team_obj, info->pe_start, info->stride, team_obj->num_pes);
  __shared__ int proxies[ROCSHMEM_BARRIER_SYNC_SIZE];

  if (is_thread_zero_in_block()) {
    for (int domain = 0; domain < view.ipc_domain_count; domain++) {
      proxies[domain] = proxy::SelectCollectiveDomainProxy(view, domain, operation);
      if (proxies[domain] < 0) {
        LOGD_ERROR_ABORT("PXN collective has no usable domain data proxy");
      }
    }
  }
  __syncthreads();
  return proxies;
}

// Reuse one owner table for all reduction tiles. Every lane has one distinct
// owner per IPC domain, so simultaneous transfers cannot overwrite scratch.
__device__ inline __noinline__ const int *GDAContext::internal_build_reduction_proxy_plan_wg(GDATeam *team_obj,
                                                                                             uint64_t operation,
                                                                                             int requested_lanes,
                                                                                             int &lanes) {
  const auto *info = team_obj->tinfo_wrt_world;
  const auto view = proxy::GetProxyTeamView(team_obj, info->pe_start, info->stride, team_obj->num_pes);
  __shared__ int owners[ROCSHMEM_BARRIER_SYNC_SIZE];
  __shared__ int lane_count;

  if (is_thread_zero_in_block()) {
    lane_count = 0;
    if (constmem.world_proxy_nic_layout_available) {
      const int nics = constmem.world_proxy_nic_name_count;
      // Round-robin NICs first, then distinct endpoint PEs within each NIC.
      for (int round = 0; round < constmem.num_pes && lane_count < requested_lanes; round++) {
        for (int turn = 0; turn < nics && lane_count < requested_lanes; turn++) {
          const int nic = (operation + turn) % nics;
          int endpoints = constmem.num_pes;
          for (int domain = 0; domain < view.ipc_domain_count; domain++) {
            const int first = view.domain_first(domain);
            endpoints = min(endpoints, proxy::CountProxyDataEndpoints(team_obj, first,
                                                          proxy::WorldProxyIpcDomainEnd(first), nic, -1));
          }

          if (round >= endpoints || (lane_count + 1) * view.ipc_domain_count > ROCSHMEM_BARRIER_SYNC_SIZE) {
            continue;
          }

          for (int domain = 0; domain < view.ipc_domain_count; domain++) {
            const int first = view.domain_first(domain);
            const int count = proxy::CountProxyDataEndpoints(team_obj, first,
                                         proxy::WorldProxyIpcDomainEnd(first), nic, -1);
            owners[domain * requested_lanes + lane_count] = proxy::SelectProxyDataEndpoint(
                team_obj, first, proxy::WorldProxyIpcDomainEnd(first), nic, -1, (operation / nics + round) % count);
          }
          lane_count++;
        }
      }
    }

    if (lane_count == 0) {
      for (int domain = 0; domain < view.ipc_domain_count; domain++) {
        owners[domain * requested_lanes] = proxy::SelectCollectiveDomainProxy(view, domain, operation);
      }
      lane_count = 1;
    }
  }

  __syncthreads();
  lanes = lane_count;
  return owners;
}

// Cache target assignments once per WG using type-independent LDS storage.
__device__ inline __noinline__ int *GDAContext::internal_build_alltoall_routes_wg(GDATeam *team_obj,
                                                                                  int source_ipc_domain_first_pe) {
  __shared__ int entries[2 * ROCSHMEM_BARRIER_SYNC_SIZE];
  int *source_proxies = entries;
  int *scratch_batches = entries + ROCSHMEM_BARRIER_SYNC_SIZE;

  if (is_thread_zero_in_block()) {
    for (int domain = 0; domain < constmem.world_proxy_ipc_domain_count; domain++) {
      const int first = constmem.world_proxy_ipc_domain_first_pes[domain];
      if (first == source_ipc_domain_first_pe) {
        continue;
      }

      for (int peer = first; peer < proxy::WorldProxyIpcDomainEnd(first); peer++) {
        const auto assignment = proxy::SelectWorldAlltoallTargetAssignment(
            source_ipc_domain_first_pe, first, peer, team_obj->proxy_data_route_sequence + team_obj->pool_index_);
        source_proxies[peer] = assignment.source_proxy_pe;
        scratch_batches[peer] = assignment.scratch_batch;
      }
    }
  }
  __syncthreads();

  return entries;
}

}  // namespace rocshmem

#endif  // USE_DAS

#endif  // LIBRARY_SRC_GDA_PROXY_ROUTE_DEVICE_HPP_
