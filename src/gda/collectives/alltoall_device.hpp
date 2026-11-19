// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#pragma once

#if defined(USE_DAS)

namespace rocshmem {

namespace proxy {

/**
 * @brief Check whether every target has a proxy and the scratch fits.
 *
 * Direct means target-selected proxy routing, not bypassing the proxy.
 * @param elem_bytes Size of one element in bytes.
 * @return true when the direct WORLD proxy all-to-all path is usable.
 */
__device__ __forceinline__ bool CanUseDirectWorldProxyAlltoall(size_t elem_bytes) {
  if (!constmem.world_proxy_nic_layout_available) {
    return false;
  }

  int max_local_pe_count = 0;
  int max_batches = 0;

  for (int source = 0; source < constmem.world_proxy_ipc_domain_count; source++) {
    const int source_ipc_domain_first_pe = constmem.world_proxy_ipc_domain_first_pes[source];
    max_local_pe_count =
        max(max_local_pe_count, WorldProxyIpcDomainEnd(source_ipc_domain_first_pe) - source_ipc_domain_first_pe);
    for (int target = 0; target < constmem.world_proxy_ipc_domain_count; target++) {
      if (source == target) {
        continue;
      }
      const int target_ipc_domain_first_pe = constmem.world_proxy_ipc_domain_first_pes[target];
      const int target_end = WorldProxyIpcDomainEnd(target_ipc_domain_first_pe);
      for (int pe = target_ipc_domain_first_pe; pe < target_end; pe++) {
        const int batch = SelectWorldAlltoallTargetAssignment(source_ipc_domain_first_pe,
                                            target_ipc_domain_first_pe, pe).scratch_batch;
        if (batch < 0) {
          return false;
        }
        max_batches = max(max_batches, batch + 1);
      }
    }
  }

  const size_t scratch_elements = ROCSHMEM_REDUCE_MIN_WRKDATA_SIZE * sizeof(double) / elem_bytes;
  return max_batches > 0 && max_local_pe_count <= scratch_elements / max_batches;
}

/** @brief Return scratch batches required by a target IPC domain. */
__device__ __forceinline__ int WorldAlltoallScratchBatchCount(int target_ipc_domain_first_pe) {
  int batches = 0;
  const int target_end = WorldProxyIpcDomainEnd(target_ipc_domain_first_pe);

  for (int source = 0; source < constmem.world_proxy_ipc_domain_count; source++) {
    const int source_ipc_domain_first_pe = constmem.world_proxy_ipc_domain_first_pes[source];
    if (source_ipc_domain_first_pe == target_ipc_domain_first_pe) {
      continue;
    }
    for (int pe = target_ipc_domain_first_pe; pe < target_end; pe++) {
      const auto assignment = SelectWorldAlltoallTargetAssignment(source_ipc_domain_first_pe,
                                                                  target_ipc_domain_first_pe, pe);
      batches = max(batches, assignment.scratch_batch + 1);
    }
  }

  return batches;
}

}  // namespace proxy

/**
 * @brief Try the eligible PXN all-to-all algorithm before the original path.
 *
 * @param team_obj Team and proxy layout metadata.
 * @param dst Destination buffer.
 * @param src Source buffer.
 * @param nelems Elements per peer.
 * @param elem_offset First element in this tile.
 * @param elem_count Number of elements in this tile.
 * @param wf_info Active workgroup metadata.
 * @return true when a PXN path completed this tile.
 */
template <typename T>
__device__ bool GDAContext::internal_alltoall_wg_pxn(GDATeam *team_obj, T *dst, const T *src, int nelems,
                                                               int elem_offset, int elem_count, ActiveWFInfo &wf_info) {
  if (elem_count <= 0) {
    return false;
  }

  const auto *team_info = team_obj->tinfo_wrt_world;
  const auto layout = proxy::SelectProxyCollectiveLayout(team_obj, team_info->pe_start,
                                                         team_info->stride, team_obj->num_pes);
  if (layout == proxy::ProxyCollectiveLayout::None) {
    return false;
  }

  // None was rejected above; reuse the classified membership instead of checking it again.
  const bool world =
      layout == proxy::ProxyCollectiveLayout::WorldAlignedData || layout == proxy::ProxyCollectiveLayout::WorldSyncOnly;
  // Full WORLD mappings permit directed destination-owned transfers. The
  // same row-streaming fallback handles every team without that capability.
  if (world && proxy::CanUseDirectWorldProxyAlltoall(sizeof(T))) {
    internal_alltoall_wg_routed_world_proxy_pxn(team_obj, dst, src, nelems, elem_offset, elem_count, wf_info);
  } else {
    internal_alltoall_wg_rows_pxn(team_obj, dst, src, nelems, elem_offset, elem_count);
  }

  return true;
}

/**
 * @brief Stream each source row for any team, retaining the local column.
 *
 * @tparam T Element type.
 * @param team_obj Team routing and scratch state.
 * @param dst Destination buffer.
 * @param src Source buffer.
 * @param nelems Elements per peer.
 * @param elem_offset First element to transfer.
 * @param elem_count Number of elements to transfer.
 */
template <typename T>
__device__ void GDAContext::internal_alltoall_wg_rows_pxn(GDATeam *team_obj, T *dst, const T *src, int nelems,
                                                            int elem_offset, int elem_count) {
  T *scratch = reinterpret_cast<T *>(team_obj->pWrk);
  long *p_sync = team_obj->alltoall_pSync;
  const int team_size = team_obj->num_pes;
  const int tile_capacity = static_cast<int>(ROCSHMEM_REDUCE_MIN_WRKDATA_SIZE * sizeof(double) / sizeof(T));
  const int tid = get_flat_block_id();
  const int block_size = get_flat_block_size();
  const int qp_row = team_obj->pool_index_ % num_qps_per_pe;
  const int broadcast_phases = proxy::TeamBroadcastPhaseCount(team_obj);
  int64_t epoch = static_cast<int64_t>(team_obj->proxy_alltoall_sync_epoch);
  ActiveWFInfo wf_info(ctx_id_, ThreadScope::wg);
  const size_t row_elements = static_cast<size_t>(team_size) * nelems;
  const size_t column_begin = static_cast<size_t>(team_obj->my_pe) * nelems;
  const size_t wanted_begin = column_begin + elem_offset;
  const size_t wanted_end = wanted_begin + elem_count;

  for (int source_rank = 0; source_rank < team_size; source_rank++) {
    const int source_pe = team_obj->get_pe_in_world(source_rank);
    // All members broadcast the same contiguous row tiles. Each receiver keeps
    // only its requested column interval, including workgroup subranges.
    // This replaces one tiny broadcast per target with one stream per source.
    for (size_t offset = 0; offset < row_elements; offset += tile_capacity) {
      const int count = static_cast<int>(min(static_cast<size_t>(tile_capacity), row_elements - offset));
      internal_broadcast_wg(scratch, src + offset, count, source_pe, team_obj->tinfo_wrt_world->pe_start,
                            team_obj->tinfo_wrt_world->stride, team_size, p_sync, epoch, qp_row, team_obj);
      const size_t begin = max(offset, wanted_begin);
      const size_t end = min(offset + count, wanted_end);
      for (size_t i = begin + tid; i < end; i += block_size) {
        dst[static_cast<size_t>(source_rank) * nelems + i - column_begin] = uncached_load(scratch + i - offset);
      }
      __syncthreads();
      // No PE may overwrite the receive tile before every consumer finishes.
      internal_proxy_sync_wg(team_obj, p_sync, wf_info, epoch + broadcast_phases);
      epoch += broadcast_phases + 1;
    }
  }

  if (is_thread_zero_in_block()) {
    team_obj->proxy_alltoall_sync_epoch = epoch;
  }
  __syncthreads();
}

/**
 * @brief Send WORLD all-to-all tiles through same-NIC source proxies.
 *
 * @tparam T Element type.
 * @param team_obj WORLD team routing and scratch state.
 * @param dst Destination buffer.
 * @param src Source buffer.
 * @param nelems Elements per peer.
 * @param elem_offset First element to transfer.
 * @param elem_count Number of elements to transfer.
 * @param wf_info Active workgroup metadata.
 */
template <typename T>
__device__ __noinline__ void GDAContext::internal_alltoall_wg_routed_world_proxy_pxn(GDATeam *team_obj, T *dst,
                                                                                 const T *src, int nelems,
                                                                                 int elem_offset, int elem_count,
                                                                                 ActiveWFInfo &wf_info) {
  const int my_pe = constmem.my_pe;
  const int source_ipc_domain_first_pe = constmem.ipc_first_pe;
  const int source_size = constmem.ipc_shm_size;
  T *scratch = reinterpret_cast<T *>(team_obj->pWrk);
  long *p_sync = team_obj->alltoall_pSync;
  int max_local_pe_count = 0;
  int max_batches = 0;
  for (int ipc_domain = 0; ipc_domain < constmem.world_proxy_ipc_domain_count; ipc_domain++) {
    const int first = constmem.world_proxy_ipc_domain_first_pes[ipc_domain];
    max_local_pe_count = max(max_local_pe_count, proxy::WorldProxyIpcDomainEnd(first) - first);
    max_batches = max(max_batches, proxy::WorldAlltoallScratchBatchCount(first));
  }

  const size_t scratch_bytes = ROCSHMEM_REDUCE_MIN_WRKDATA_SIZE * sizeof(double);
  const int tile_capacity = static_cast<int>(scratch_bytes / (max_local_pe_count * max_batches * sizeof(T)));
  int64_t sync_epoch = static_cast<int64_t>(team_obj->proxy_alltoall_sync_epoch + 1);

  const int *source_proxies = internal_build_alltoall_routes_wg(team_obj, source_ipc_domain_first_pe);
  const int *scratch_batches = source_proxies + ROCSHMEM_BARRIER_SYNC_SIZE;

  // Within an IPC domain, the source writes directly to each local target.
  for (int slot = 0; slot < source_size; slot++) {
    const int target_pe = source_ipc_domain_first_pe + slot;
    T *target = dst + my_pe * nelems + elem_offset;
    const T *source = src + target_pe * nelems + elem_offset;
    const size_t bytes = static_cast<size_t>(elem_count) * sizeof(T);
    if (target_pe == my_pe) {
      memcpy_wg<MemcpyKind::Put>(target, const_cast<T *>(source), bytes);
    } else {
      internal_proxy_putmem_wg(target, source, bytes, target_pe, 0, wf_info);
    }
  }

  // Every PE follows the same target-domain, batch and tile order, even when
  // its own IPC domain is the target. This keeps world proxy sync collective.
  for (int ipc_domain = 0; ipc_domain < constmem.world_proxy_ipc_domain_count; ipc_domain++) {
    const int target_ipc_domain_first_pe = constmem.world_proxy_ipc_domain_first_pes[ipc_domain];
    const int target_end = proxy::WorldProxyIpcDomainEnd(target_ipc_domain_first_pe);
    const int batches = proxy::WorldAlltoallScratchBatchCount(target_ipc_domain_first_pe);
    for (int batch = 0; batch < batches; batch++) {
      for (int offset = elem_offset; offset < elem_offset + elem_count; offset += tile_capacity) {
        const int count = min(tile_capacity, elem_offset + elem_count - offset);
        const size_t bytes = static_cast<size_t>(count) * sizeof(T);

        if (source_ipc_domain_first_pe != target_ipc_domain_first_pe) {
          for (int target_pe = target_ipc_domain_first_pe; target_pe < target_end; target_pe++) {
            const proxy::WorldAlltoallTargetAssignment assignment{source_proxies[target_pe],
                                                                  scratch_batches[target_pe]};
            if (assignment.scratch_batch != batch || assignment.source_proxy_pe != my_pe) {
              continue;
            }
            for (int slot = 0; slot < source_size; slot++) {
              const int source_pe = source_ipc_domain_first_pe + slot;
              T *stage = scratch + (batch * max_local_pe_count + slot) * tile_capacity;
              const T *source = src + target_pe * nelems + offset;
              if (source_pe == my_pe) {
                memcpy_wg<MemcpyKind::Put>(stage, const_cast<T *>(source), bytes);
              } else {
                internal_proxy_getmem_wg(stage, source, bytes, source_pe, 0, wf_info);
              }
            }
          }
        }
        __threadfence_system();
        internal_world_proxy_sync_wg(my_pe, constmem.num_pes, p_sync, wf_info, sync_epoch++);

        if (source_ipc_domain_first_pe != target_ipc_domain_first_pe) {
          for (int target_pe = target_ipc_domain_first_pe; target_pe < target_end; target_pe++) {
            const proxy::WorldAlltoallTargetAssignment assignment{source_proxies[target_pe],
                                                                  scratch_batches[target_pe]};
            if (assignment.scratch_batch != batch || assignment.source_proxy_pe != my_pe) {
              continue;
            }

            // The staging owner sends its tile, so the transport publishes
            // this GPU's local writes before its NIC reads the source buffer.
            for (int slot = 0; slot < source_size; slot++) {
              const int source_pe = source_ipc_domain_first_pe + slot;
              T *target = dst + source_pe * nelems + offset;
              const T *stage = scratch + (batch * max_local_pe_count + slot) * tile_capacity;
              internal_proxy_putmem_wg(target, stage, bytes, target_pe, 0, wf_info);
            }
          }
        }
        internal_world_proxy_sync_wg(my_pe, constmem.num_pes, p_sync, wf_info, sync_epoch++);
      }
    }
  }

  if (is_thread_zero_in_block()) {
    team_obj->proxy_alltoall_sync_epoch = sync_epoch - 1;
    team_obj->proxy_data_route_sequence++;
  }
  __syncthreads();
}

__device__ __forceinline__ bool gda_copy_if_self(int pe, void *dst, const void *src, size_t bytes) {
  if (pe != constmem.my_pe) return false;
  if (bytes != 0) {
    memcpy_lane<MemcpyKind::Put>(dst, const_cast<void *>(src), bytes);
  }
  return true;
}

template <typename T>
__device__ __noinline__ void GDAContext::alltoall_linear_wg_nopxn(rocshmem_team_t team, T *dst, const T *src,
                                                                  int nelems, int elem_offset, int elem_count) {
  GDATeam *team_obj = reinterpret_cast<GDATeam *>(team);

  int pe_start = team_obj->tinfo_wrt_world->pe_start;
  int pe_size = team_obj->num_pes;
  int stride = team_obj->tinfo_wrt_world->stride;
  long *pSync = team_obj->alltoall_pSync;
  int my_pe_in_team = team_obj->my_pe;

  // Normalize: elem_count == -1 means "process the full [0, nelems) range".
  if (elem_count < 0) elem_count = nelems;

  int wf_id = get_flat_block_id() / WF_SIZE;
  int wf_count = (int)ceil((double)get_flat_block_size() / (double)WF_SIZE);

  ActiveWFInfo wf_info(ctx_id_, ThreadScope::wg);
  // Empty slices still participate in the team synchronization below, but
  // must not submit a zero-byte RDMA operation. alltoallv_on_stream may launch
  // more workgroups than elements because its counts reside in device memory.
  if (elem_count > 0) {
    const int chunk_elems = collective_chunk_elems<T>(elem_count, kAlltoallChunks, pe_size > constmem.ipc_shm_size);
    for (int processed = 0; processed < elem_count;) {
      const int count = min(chunk_elems, elem_count - processed);
      const int offset = elem_offset + processed;
      // Have each PE put their designated data to the other PEs.
      for (int j = wf_id; j < pe_size; j += wf_count) {
        int dest_pe = team_obj->get_pe_in_world(j);
        internal_putmem_nbi_wave(&dst[my_pe_in_team * nelems + offset], &src[j * nelems + offset],
                                 count * sizeof(T), dest_pe, dest_pe, wf_info);
      }
      // Drain each chunk before allowing the next burst of peer transfers.
      for (int j = wf_id; j < pe_size; j += wf_count) {
        int dest_pe = team_obj->get_pe_in_world(j);
        internal_quiet(dest_pe, wf_info);
      }
      processed += count;
    }
  }

  // wait until everyone has obtained their designated data
  internal_sync_wg(constmem.my_pe, pe_start, stride, pe_size, pSync, wf_info);
}

template <typename T>
__device__ void GDAContext::alltoall_linear_thread_puts_nopxn(rocshmem_team_t team, T *dst, const T *src, int nelems,
                                                              int elem_offset, int elem_count) {
  GDATeam *team_obj = reinterpret_cast<GDATeam *>(team);

  int pe_size = team_obj->num_pes;
  long *pSync = team_obj->alltoall_pSync;
  int my_pe_in_team = team_obj->my_pe;
  uint64_t a2a_sn = team_obj->alltoall_sequence_number;
  uint64_t alltoall_pSync_offset = (a2a_sn % 2) * pe_size;
  long completion_count = a2a_sn / 2 + 1;

  // Normalize: elem_count == -1 means "process the full [0, nelems) range".
  if (elem_count < 0) elem_count = nelems;

  int tid = get_flat_block_id();
  int step_size = min(get_flat_block_size(), WF_SIZE);

  int *qp_indices = reinterpret_cast<int *>(team_obj->pWrk);

  // Have each PE put their designated data to the other PEs
  for (int j = tid; j < pe_size; j += step_size) {
    int dest_pe = team_obj->get_pe_in_world(j);
    ActiveWFInfo wf_info(dest_pe);
    int qp_index = get_qp_index(dest_pe, wf_info);
    qp_indices[j] = qp_index;
    if (gda_copy_if_self(dest_pe, &dst[my_pe_in_team * nelems + elem_offset],
                         &src[j * nelems + elem_offset], elem_count * sizeof(T))) {
      pSync[alltoall_pSync_offset + my_pe_in_team] = completion_count;
      continue;
    }

    // Submit payload before the completion notification.
    put_nbi_shared_qp(qp_index, &dst[my_pe_in_team * nelems + elem_offset],
                      &src[j * nelems + elem_offset], elem_count * sizeof(T), true);
    publish_completion_shared_qp(qp_index, &pSync[alltoall_pSync_offset + my_pe_in_team], completion_count);
  }

  // wait until everyone has obtained their designated data
  for (int j = tid; j < pe_size; j += step_size) {
    long *sync_flags = &pSync[alltoall_pSync_offset + j];
    while (uncached_load(sync_flags) < completion_count) {
    }

    quiet_shared_qp(qp_indices[j]);
  }

  if (is_thread_zero_in_block()) {
    team_obj->alltoall_sequence_number++;
  }

  __syncthreads();
}

__device__ inline bool GDAContext::alltoallmem_wave_pxn(rocshmem_team_t team, void *dst,
                                                                         const void *src, int bytes) {
  GDATeam *team_obj = reinterpret_cast<GDATeam *>(team);
  const auto *info = team_obj->tinfo_wrt_world;
  return internal_try_collective_pxn(team_obj, info->pe_start, info->stride, team_obj->num_pes, "alltoall_wave",
      [&] __device__() {
        if (proxy::SelectProxyCollectiveLayout(team_obj, info->pe_start, info->stride, team_obj->num_pes) ==
            proxy::ProxyCollectiveLayout::None) {
          return false;
        }
        constexpr size_t capacity = ROCSHMEM_REDUCE_MIN_WRKDATA_SIZE * sizeof(double) / 2;
        char *stage = static_cast<char *>(team_obj->pWrk);
        char *received = stage + capacity;
        ActiveWFInfo wf_info(ctx_id_, ThreadScope::wave);
        int64_t epoch = static_cast<int64_t>(team_obj->proxy_alltoall_sync_epoch);
        long *p_sync = team_obj->alltoall_pSync;
        if (bytes == 0) {
          internal_proxy_sync_wave(team_obj, p_sync, wf_info, epoch++);
        }
        for (int target = 0; target < team_obj->num_pes; target++) {
          for (size_t offset = 0; offset < static_cast<size_t>(bytes); offset += capacity) {
            const size_t count = min(capacity, static_cast<size_t>(bytes) - offset);
            memcpy_wave<MemcpyKind::Get>(stage,
                const_cast<char *>(static_cast<const char *>(src) + static_cast<size_t>(target) * bytes + offset),
                count);
            internal_proxy_sync_wave(team_obj, p_sync, wf_info, epoch++);
            for (int rank = 0; rank < team_obj->num_pes; rank++) {
              if (!internal_broadcastmem_wave_pxn(team_obj, received, stage, count, team_obj->get_pe_in_world(rank),
                                                     p_sync, wf_info, epoch)) {
                return false;
              }
              if (team_obj->my_pe == target) {
                char *output = static_cast<char *>(dst) + static_cast<size_t>(rank) * bytes + offset;
                for (size_t i = get_flat_block_id() % WF_SIZE; i < count; i += WF_SIZE) {
                  output[i] = uncached_load(received + i);
                }
              }
              internal_proxy_sync_wave(team_obj, p_sync, wf_info, epoch++);
            }
          }
        }
        if (is_thread_zero_in_wave()) {
          team_obj->proxy_alltoall_sync_epoch = epoch;
        }
        __threadfence_system();
        __builtin_amdgcn_wave_barrier();
        return true;
      });
}

// Owns path dispatch and the complete collective operation.
template <typename T>
__device__ __noinline__ void GDAContext::alltoall_wg_das(rocshmem_team_t team, T *dst, const T *src, int nelems,
                                                         int elem_offset, int elem_count) {
  GDATeam *team_obj = reinterpret_cast<GDATeam *>(team);
  const auto *info = team_obj->tinfo_wrt_world;
  const int count = elem_count < 0 ? nelems : elem_count;
  if (internal_try_collective_pxn(team_obj, info->pe_start, info->stride, team_obj->num_pes, "alltoall",
      [&] __device__() {
        ActiveWFInfo wf_info(ctx_id_, ThreadScope::wg);
        if (count == 0) {
          if (proxy::SelectProxyCollectiveLayout(team_obj, info->pe_start, info->stride, team_obj->num_pes) ==
              proxy::ProxyCollectiveLayout::None) {
            return false;
          }

          const int64_t epoch = static_cast<int64_t>(team_obj->proxy_alltoall_sync_epoch);
          internal_proxy_sync_wg(team_obj, team_obj->alltoall_pSync, wf_info, epoch);
          if (is_thread_zero_in_block()) {
            team_obj->proxy_alltoall_sync_epoch = epoch + 1;
          }
          __syncthreads();
          return true;
        }
        internal_try_sync_pxn(team_obj, team_obj->barrier_pSync, ThreadScope::wg);
        return internal_alltoall_wg_pxn(team_obj, dst, src, nelems, elem_offset, count, wf_info);
      })) {
    return;
  }

  alltoall_linear_wg_nopxn(team, dst, src, nelems, elem_offset, elem_count);
}

}  // namespace rocshmem

#endif  // USE_DAS
