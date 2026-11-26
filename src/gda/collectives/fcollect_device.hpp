// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#pragma once

#if defined(USE_DAS)

namespace rocshmem {

/*
 * Pack contributions inside each IPC domain, exchange one packed tile per
 * batch directly into team-rank output slots. WORLD is just a team
 * containing all PEs. Tiling bounds scratch use; it does not select an algorithm.
 */
template <typename T>
__device__ void GDAContext::internal_fcollect_wg_ipc_domain_pxn(GDATeam *team_obj, T *dst, const T *src, int nelems,
                                                                  int elem_offset, int elem_count) {
  const auto *info = team_obj->tinfo_wrt_world;
  const auto view = proxy::GetProxyTeamView(team_obj, info->pe_start, info->stride, team_obj->num_pes);
  const uint64_t operation = team_obj->proxy_data_route_sequence + team_obj->pool_index_;
  const int *domain_proxies = internal_build_collective_proxy_plan_wg(team_obj, operation);
  constexpr size_t scratch_elements = ROCSHMEM_REDUCE_MIN_WRKDATA_SIZE * sizeof(double) / sizeof(T);
  T *packed = reinterpret_cast<T *>(team_obj->pWrk);
  int max_members = 0;

  for (int domain = 0; domain < view.ipc_domain_count; domain++) {
    const int first = view.domain_first(domain);
    const int end = proxy::WorldProxyIpcDomainEnd(first);
    int members = 0;
    for (int rank = 0; rank < team_obj->num_pes; rank++) {
      const int peer = team_obj->get_pe_in_world(rank);
      members += peer >= first && peer < end;
    }
    max_members = max(max_members, members);
  }

  if (max_members <= 0) {
    LOGD_ERROR_ABORT("PXN fcollect has no IPC members");
  }

  // Use the same pack/broadcast protocol for every size. A complete domain
  // pack is contiguous in the final output. Otherwise one member per batch
  // keeps each tile contiguous too, avoiding receive-and-scatter copies.
  const bool pack_domain =
      elem_offset == 0 && elem_count == nelems && static_cast<size_t>(nelems) <= scratch_elements / max_members;
  const int tile_capacity = static_cast<int>(scratch_elements / (pack_domain ? max_members : 1));
  const int qp_row = team_obj->pool_index_ % num_qps_per_pe;
  long *p_sync = team_obj->alltoall_pSync;
  int64_t epoch = static_cast<int64_t>(team_obj->proxy_fcollect_sync_epoch);
  ActiveWFInfo wf_info(ctx_id_, ThreadScope::wg);

  for (int offset = elem_offset; offset < elem_offset + elem_count; offset += tile_capacity) {
    const int count = min(tile_capacity, elem_offset + elem_count - offset);

    // Capture every input tile before any in-place output can overwrite it.
    if (!pack_domain) {
      internal_proxy_getmem_wg(packed, src + offset, static_cast<size_t>(count) * sizeof(T),
                               constmem.my_pe, 0, wf_info);
    } else {
      for (int domain = 0; domain < view.ipc_domain_count; domain++) {
        if (constmem.my_pe != domain_proxies[domain]) {
          continue;
        }
        int slot = 0;
        const int first = view.domain_first(domain);
        for (int rank = 0; rank < team_obj->num_pes; rank++) {
          const int peer = team_obj->get_pe_in_world(rank);
          if (peer >= first && peer < proxy::WorldProxyIpcDomainEnd(first)) {
            internal_proxy_getmem_wg(packed + slot * count, src + offset, static_cast<size_t>(count) * sizeof(T),
                                     peer, 0, wf_info);
            slot++;
          }
        }
      }
    }

    internal_proxy_sync_wg(team_obj, p_sync, wf_info, epoch++);

    for (int domain = 0; domain < view.ipc_domain_count; domain++) {
      const int first = view.domain_first(domain);
      int first_rank = -1;
      int members = 0;
      for (int rank = 0; rank < team_obj->num_pes; rank++) {
        const int peer = team_obj->get_pe_in_world(rank);
        if (peer >= first && peer < proxy::WorldProxyIpcDomainEnd(first)) {
          if (first_rank < 0) {
            first_rank = rank;
          }
          members++;
        }
      }
      const int batch_members = pack_domain ? max(1, members) : 1;
      for (int member = 0; member < members; member += batch_members) {
        const int rank = first_rank + member;
        const int root = pack_domain ? domain_proxies[domain] : team_obj->get_pe_in_world(rank);
        internal_broadcast_wg(dst + static_cast<size_t>(rank) * nelems + offset, packed, batch_members * count, root,
                              info->pe_start, info->stride, team_obj->num_pes, p_sync, epoch, qp_row, team_obj);
        epoch += proxy::TeamBroadcastPhaseCount(team_obj);
      }
    }
    // All readers must finish before the next tile overwrites packed inputs.
    internal_proxy_sync_wg(team_obj, p_sync, wf_info, epoch++);
  }

  if (elem_count == 0) {
    internal_proxy_sync_wg(team_obj, p_sync, wf_info, epoch++);
  }

  if (is_thread_zero_in_block()) {
    team_obj->proxy_fcollect_sync_epoch = epoch;
  }
  __syncthreads();
}

template <typename T>
__device__ void GDAContext::fcollect_wg_das(rocshmem_team_t team, T *dst, const T *src,
                                                int nelems, int elem_offset, int elem_count) {
  GDATeam *team_obj = reinterpret_cast<GDATeam *>(team);

  int pe_start = team_obj->tinfo_wrt_world->pe_start;
  int pe_size = team_obj->num_pes;
  int stride = team_obj->tinfo_wrt_world->stride;
  long *pSync = team_obj->alltoall_pSync;
  int my_pe_in_team = team_obj->my_pe;
  if (elem_count < 0) elem_count = nelems;

  if (internal_try_collective_pxn(team_obj, pe_start, stride, pe_size, "fcollect",
      [&] __device__() {
        if (proxy::SelectProxyCollectiveLayout(team_obj, pe_start, stride, pe_size) ==
            proxy::ProxyCollectiveLayout::None) {
          return false;
        }
        internal_try_sync_pxn(team_obj, team_obj->barrier_pSync, ThreadScope::wg);
        internal_fcollect_wg_ipc_domain_pxn(team_obj, dst, src, nelems, elem_offset, elem_count);
        return true;
      })) {
    return;
  }

  ActiveWFInfo wf_info(ctx_id_, ThreadScope::wg);

  if (!constmem.pxn_enabled && try_fcollectmem_same_slot_wg_nopxn(
          team_obj, dst, src, static_cast<size_t>(nelems) * sizeof(T),
          static_cast<size_t>(elem_offset) * sizeof(T),
          static_cast<size_t>(elem_count) * sizeof(T))) {
    return;
  }

  // Chunk the linear path below instead of selecting the synchronization-heavy
  // hierarchical path solely because more workgroups are active.
  if (pe_size >= 4 && static_cast<size_t>(elem_count) * sizeof(T) >= kHierarchicalMinBytes) {
    // FCollect is a concatenation of one broadcast per source rank.  Reusing
    // the hierarchical broadcast path makes each source cross a node boundary
    // once, followed by IPC fan-out inside the destination IPC domain, instead of
    // sending the same source segment independently to every remote PE.
    for (int rank = 0; rank < pe_size; rank++) {
      const int root = team_obj->get_pe_in_world(rank);
      internal_broadcastmem_wg_das(dst + static_cast<size_t>(rank) * nelems + elem_offset, src + elem_offset,
                                   elem_count * sizeof(T),
                                   root, pe_start, stride, pe_size, pSync, team_obj);
    }
    return;
  }

  // Bound aggregate network traffic across WGs without introducing one team-wide
  // broadcast per source. Single-WG and IPC transfers retain their size.
  const int chunk_elems = collective_chunk_elems<T>(elem_count, kFcollectChunks, pe_size > constmem.ipc_shm_size);
  for (int processed = 0; processed < elem_count;) {
    const int count = min(chunk_elems, elem_count - processed);
    const int offset = elem_offset + processed;
    for (int rank = 0; rank < pe_size; rank++) {
      int dest_pe = team_obj->get_pe_in_world(rank);
      internal_putmem_nbi_wg(&dst[my_pe_in_team * nelems + offset], src + offset, count * sizeof(T),
                             dest_pe, dest_pe, wf_info);
    }

    if (is_thread_zero_in_block()) {
      for (int rank = 0; rank < pe_size; rank++) {
        int dest_pe = team_obj->get_pe_in_world(rank);
        int local_pe{-1};
        if (!ipcImpl_.isIpcAvailable(constmem.my_pe, dest_pe, &local_pe)) {
          qps[dest_pe].quiet(wf_info);
        }
      }
    }
    processed += count;
  }
  internal_sync_wg(constmem.my_pe, pe_start, stride, pe_size, pSync, wf_info);
}

__device__ inline bool GDAContext::fcollectmem_wave_pxn(rocshmem_team_t team, void *dst,
                                                                         const void *src, int bytes) {
  GDATeam *team_obj = reinterpret_cast<GDATeam *>(team);
  const auto *info = team_obj->tinfo_wrt_world;
  return internal_try_collective_pxn(team_obj, info->pe_start, info->stride, team_obj->num_pes, "fcollect_wave",
      [&] __device__() {
        if (proxy::SelectProxyCollectiveLayout(team_obj, info->pe_start, info->stride, team_obj->num_pes) ==
            proxy::ProxyCollectiveLayout::None) {
          return false;
        }

        constexpr size_t capacity = ROCSHMEM_REDUCE_MIN_WRKDATA_SIZE * sizeof(double);
        char *stage = static_cast<char *>(team_obj->pWrk);
        ActiveWFInfo wf_info(ctx_id_, ThreadScope::wave);
        int64_t epoch = static_cast<int64_t>(team_obj->proxy_fcollect_sync_epoch);
        long *p_sync = team_obj->alltoall_pSync;
        if (bytes == 0) {
          internal_proxy_sync_wave(team_obj, p_sync, wf_info, epoch++);
        }

        for (size_t offset = 0; offset < static_cast<size_t>(bytes); offset += capacity) {
          const size_t count = min(capacity, static_cast<size_t>(bytes) - offset);
          memcpy_wave<MemcpyKind::Get>(stage, const_cast<char *>(static_cast<const char *>(src) + offset), count);
          internal_proxy_sync_wave(team_obj, p_sync, wf_info, epoch++);
          for (int rank = 0; rank < team_obj->num_pes; rank++) {
            if (!internal_broadcastmem_wave_pxn(
                    team_obj, static_cast<char *>(dst) + static_cast<size_t>(rank) * bytes + offset, stage, count,
                    team_obj->get_pe_in_world(rank), p_sync, wf_info, epoch)) {
              return false;
            }
          }
        }

        if (is_thread_zero_in_wave()) {
          team_obj->proxy_fcollect_sync_epoch = epoch;
        }
        __threadfence_system();
        __builtin_amdgcn_wave_barrier();
        return true;
      });
}

}  // namespace rocshmem

#endif  // USE_DAS
