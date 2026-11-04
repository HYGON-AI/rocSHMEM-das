// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#pragma once

#if defined(USE_DAS)

namespace rocshmem {

// Every PE must agree before a locally regular layout can use alltoall.
__device__ inline __noinline__ void GDAContext::internal_alltoallv_wg_world_proxy_check_uniform_layout_pxn(
    GDATeam *team_obj, const size_t dest_nelems[], const size_t dest_displs[], const size_t source_nelems[],
    const size_t source_displs[], ActiveWFInfo &wf_info, int64_t epoch, ProxyAlltoallvLayout *layout) {
  const int pe_size = constmem.num_pes;
  const int proxy_qp_row = team_obj->pool_index_ % num_qps_per_pe;
  // Select the regular fast path collectively. A locally regular PE may
  // belong to a globally asymmetric exchange, so local checks alone are unsafe.
  size_t *control = reinterpret_cast<size_t *>(team_obj->pWrk);
  const size_t local_count = source_nelems[0];
  bool local_regular = local_count <= 0x7fffffff;

  for (int peer = 0; peer < pe_size && local_regular; peer++) {
    local_regular = source_nelems[peer] == local_count && dest_nelems[peer] == local_count &&
                    source_displs[peer] == static_cast<size_t>(peer) * local_count &&
                    dest_displs[peer] == static_cast<size_t>(peer) * local_count;
  }

  if (is_thread_zero_in_block()) {
    control[0] = local_regular;
    control[1] = local_count;
  }
  __threadfence_system();
  __syncthreads();
  internal_world_proxy_sync_wg(constmem.my_pe, pe_size, team_obj->alltoall_pSync, wf_info, epoch++, proxy_qp_row);

  // Aggregate over local IPC first; only one descriptor per IPC domain crosses
  // the network. Slots 0/2/4/6 hold local, fetched, aggregate and received pairs.
  bool regular = true;
  size_t common_count = 0;
  for (int ipc_domain = 0; ipc_domain < constmem.world_proxy_ipc_domain_count; ipc_domain++) {
    const int first = constmem.world_proxy_ipc_domain_first_pes[ipc_domain];
    if (constmem.my_pe == first) {
      if (is_thread_zero_in_block()) {
        control[4] = control[0];
        control[5] = control[1];
      }
      __syncthreads();
      for (int peer = first + 1; peer < proxy::WorldProxyIpcDomainEnd(first); peer++) {
        internal_proxy_getmem_wg(control + 2, control, 2 * sizeof(size_t), peer, proxy_qp_row, wf_info);
        if (is_thread_zero_in_block()) {
          control[4] = control[4] && uncached_load(control + 2) && control[5] == uncached_load(control + 3);
        }
        __syncthreads();
      }
    }
    __threadfence_system();
    __syncthreads();
    // Root-side proxies must not fetch the descriptor before aggregation ends.
    internal_world_proxy_sync_wg(constmem.my_pe, pe_size, team_obj->alltoall_pSync, wf_info, epoch++, proxy_qp_row);
    internal_broadcast_wg_dispatch(control + 6, control + 4, 2, first, 0, 1, pe_size, team_obj->alltoall_pSync, epoch,
                          proxy_qp_row, team_obj);
    epoch += proxy::kWorldBroadcastSyncPhases;
    const size_t ipc_domain_regular = __hip_atomic_load(control + 6, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
    const size_t ipc_domain_element_count = __hip_atomic_load(control + 7, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
    if (ipc_domain == 0) {
      common_count = ipc_domain_element_count;
    }
    regular = regular && ipc_domain_regular && ipc_domain_element_count == common_count;
    // Every lane must consume this descriptor before its receive slot is reused.
    __syncthreads();
    internal_world_proxy_sync_wg(constmem.my_pe, pe_size, team_obj->alltoall_pSync, wf_info, epoch++, proxy_qp_row);
  }

  // Publish one decision for the whole workgroup, including the next free
  // synchronization epoch. No helper writes through a caller-private reference.
  if (is_thread_zero_in_block()) {
    layout->count = common_count;
    layout->next_epoch = epoch;
    layout->uniform = regular;
  }
  __syncthreads();
}

// Broadcast each packed source row; keep its count table outside the payload tile.
template <typename T>
__device__ __noinline__ int64_t GDAContext::internal_alltoallv_wg_world_proxy_packed_rows_pxn(
    GDATeam *team_obj, T *dest, const size_t dest_nelems[], const size_t dest_displs[], T *source,
    const size_t source_nelems[], const size_t source_displs[], int wg_id, int wg_count, ActiveWFInfo &wf_info,
    int64_t epoch) {
  char *workspace = reinterpret_cast<char *>(team_obj->pWrk);
  constexpr size_t kWorkspaceBytes = ROCSHMEM_REDUCE_MIN_WRKDATA_SIZE * sizeof(double);
  const size_t half_bytes = kWorkspaceBytes / 2;
  T *stage = reinterpret_cast<T *>(workspace);
  T *received = reinterpret_cast<T *>(workspace + half_bytes);
  size_t *stage_count = reinterpret_cast<size_t *>(stage);
  size_t *received_count = reinterpret_cast<size_t *>(received);
  const int pe_size = constmem.num_pes;
  const int proxy_qp_row = team_obj->pool_index_ % num_qps_per_pe;
  const size_t metadata_bytes = static_cast<size_t>(pe_size) * sizeof(size_t);
  // The host limits WORLD proxy synchronization to this many PEs.
  // Even the largest supported count table leaves room for a payload element.
  static_assert(ROCSHMEM_BARRIER_SYNC_SIZE * sizeof(size_t) + sizeof(T) <= kWorkspaceBytes / 2,
                "WORLD alltoallv needs room for counts and payload");
  size_t *row_counts = reinterpret_cast<size_t *>(workspace + kWorkspaceBytes - metadata_bytes);
  const size_t row_capacity = (half_bytes - metadata_bytes) / sizeof(T);

  for (int source_pe = 0; source_pe < pe_size; source_pe++) {
    if (constmem.my_pe == source_pe) {
      for (int peer = get_flat_block_id(); peer < pe_size; peer += get_flat_block_size()) {
        const size_t total = source_nelems[peer];
        stage_count[peer] = total / wg_count + (static_cast<size_t>(wg_id) < total % wg_count);
      }
    }

    __threadfence_system();
    __syncthreads();
    internal_world_proxy_sync_wg(constmem.my_pe, pe_size, team_obj->alltoall_pSync, wf_info, epoch++, proxy_qp_row);
    internal_broadcast_wg_dispatch(received_count, stage_count, pe_size, source_pe, 0, 1, pe_size, team_obj->alltoall_pSync,
                          epoch, proxy_qp_row, team_obj);
    epoch += proxy::kWorldBroadcastSyncPhases;

    for (int peer = get_flat_block_id(); peer < pe_size; peer += get_flat_block_size()) {
      row_counts[peer] = __hip_atomic_load(received_count + peer, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
    }
    __syncthreads();
    internal_world_proxy_sync_wg(constmem.my_pe, pe_size, team_obj->alltoall_pSync, wf_info, epoch++, proxy_qp_row);

    size_t row_elements = 0;
    size_t wanted_begin = 0;
    for (int peer = 0; peer < pe_size; peer++) {
      if (peer < constmem.my_pe) {
        wanted_begin += row_counts[peer];
      }
      row_elements += row_counts[peer];
    }
    const size_t total = dest_nelems[source_pe];
    const size_t local_elements = total / wg_count + (static_cast<size_t>(wg_id) < total % wg_count);
    const size_t wanted_elements = min(row_counts[constmem.my_pe], local_elements);
    const size_t local_begin = (total / wg_count) * wg_id + min(static_cast<size_t>(wg_id), total % wg_count);

    for (size_t offset = 0; offset < row_elements; offset += row_capacity) {
      const size_t tile = min(row_capacity, row_elements - offset);
      if (constmem.my_pe == source_pe) {
        size_t peer_begin = 0;
        for (int peer = 0; peer < pe_size; peer++) {
          const size_t begin = max(offset, peer_begin);
          const size_t end = min(offset + tile, peer_begin + row_counts[peer]);
          if (begin < end) {
            const size_t send_total = source_nelems[peer];
            const size_t send_begin =
                (send_total / wg_count) * wg_id + min(static_cast<size_t>(wg_id), send_total % wg_count);
            memcpy_wg<MemcpyKind::Put>(stage + begin - offset,
                                       source + source_displs[peer] + send_begin + begin - peer_begin,
                                       (end - begin) * sizeof(T));
          }
          peer_begin += row_counts[peer];
        }
      }
      __threadfence_system();
      __syncthreads();
      internal_world_proxy_sync_wg(constmem.my_pe, pe_size, team_obj->alltoall_pSync, wf_info, epoch++, proxy_qp_row);
      internal_broadcast_wg_dispatch(received, stage, static_cast<int>(tile), source_pe, 0, 1, pe_size, team_obj->alltoall_pSync,
                            epoch, proxy_qp_row, team_obj);
      epoch += proxy::kWorldBroadcastSyncPhases;
      const size_t begin = max(offset, wanted_begin);
      const size_t end = min(offset + tile, wanted_begin + wanted_elements);
      if (begin < end) {
        for (size_t i = get_flat_block_id(); i < end - begin; i += get_flat_block_size()) {
          dest[dest_displs[source_pe] + local_begin + begin - wanted_begin + i] =
              uncached_load(received + begin - offset + i);
        }
      }
      __syncthreads();
      internal_world_proxy_sync_wg(constmem.my_pe, pe_size, team_obj->alltoall_pSync, wf_info, epoch++, proxy_qp_row);
    }
  }

  return epoch;
}

/**
 * @brief Select uniform all-to-all or packed-row WORLD all-to-allv transfer.
 *
 * @tparam T Element type.
 * @param team_obj WORLD team routing and scratch state.
 * @param dest Destination buffer and per-source metadata.
 * @param dest_nelems Destination element counts.
 * @param dest_displs Destination element displacements.
 * @param source Source buffer and per-target metadata.
 * @param source_nelems Source element counts.
 * @param source_displs Source element displacements.
 */
template <typename T>
__device__ __noinline__ void GDAContext::internal_alltoallv_wg_world_proxy_pxn(
    GDATeam *team_obj, T *dest, const size_t dest_nelems[], const size_t dest_displs[], T *source,
    const size_t source_nelems[], const size_t source_displs[], int wg_id, int wg_count) {
  ActiveWFInfo wf_info(ctx_id_, ThreadScope::wg);
  int64_t epoch = static_cast<int64_t>(team_obj->proxy_alltoall_sync_epoch);
  __shared__ ProxyAlltoallvLayout layout;
  internal_alltoallv_wg_world_proxy_check_uniform_layout_pxn(team_obj, dest_nelems, dest_displs, source_nelems,
                                                         source_displs, wf_info, epoch, &layout);
  const size_t common_count = layout.count;
  const bool regular = layout.uniform;
  epoch = layout.next_epoch;
  if (is_thread_zero_in_block()) {
    team_obj->proxy_alltoall_sync_epoch = epoch;
  }
  __syncthreads();
  if (regular) {
    size_t begin = 0;
    size_t count = 0;
    split_wg_range(common_count, wg_id, wg_count, &begin, &count);
    alltoall_wg_das(reinterpret_cast<rocshmem_team_t>(team_obj), dest, source, static_cast<int>(common_count),
                    static_cast<int>(begin), static_cast<int>(count));
    return;
  }
  epoch = internal_alltoallv_wg_world_proxy_packed_rows_pxn(team_obj, dest, dest_nelems, dest_displs, source, source_nelems,
                                                        source_displs, wg_id, wg_count, wf_info, epoch);
  if (is_thread_zero_in_block()) {
    team_obj->proxy_alltoall_sync_epoch = epoch;
  }
  __syncthreads();
}

template <typename T>
__device__ __noinline__ void GDAContext::alltoallv_copy_nopxn(rocshmem_team_t team, T *dest, const size_t dest_nelems[],
                                                              const size_t dest_displs[], T *source,
                                                              const size_t source_nelems[],
                                                              const size_t source_displs[], int wg_id, int wg_count) {
  GDATeam *team_obj = reinterpret_cast<GDATeam *>(team);
  int pe_size = team_obj->num_pes;
  long *pSync = team_obj->alltoall_pSync;
  int my_pe_in_team = team_obj->my_pe;
  uint64_t a2a_sn = team_obj->alltoall_sequence_number;
  uint64_t alltoall_pSync_offset = (a2a_sn % 2) * pe_size;
  long completion_count = a2a_sn / 2 + 1;
  T *tmp_buf = reinterpret_cast<T *>(team_obj->pWrk);
  int tmp_buf_off = (ROCSHMEM_REDUCE_MIN_WRKDATA_SIZE * sizeof(double)) / (pe_size * sizeof(T));

  int tid = get_flat_block_id();
  int step_size = min(get_flat_block_size(), WF_SIZE);

  if (tid == 0) {
    for (int j = 0; j < pe_size; j++) {
      size_t offset = 0;
      size_t count = 0;
      split_wg_range(source_nelems[j], wg_id, wg_count, &offset, &count);
      if (count > static_cast<size_t>(tmp_buf_off)) {
        LOGD_ERROR_ABORT("GDA alltoallv COPY source exceeds pWrk slot");
      }
      split_wg_range(dest_nelems[j], wg_id, wg_count, &offset, &count);
      if (count > static_cast<size_t>(tmp_buf_off)) {
        LOGD_ERROR_ABORT("GDA alltoallv COPY destination exceeds pWrk slot");
      }
    }
  }
  __syncthreads();

  // Have each PE put their designated data to the other PEs
  for (int j = tid; j < pe_size; j += step_size) {
    int dest_pe = team_obj->get_pe_in_world(j);
    int qp_index = get_team_qp_index(team_obj, dest_pe);
    const size_t total = source_nelems[j];
    size_t elem_offset = 0;
    size_t elem_count = 0;
    split_wg_range(total, wg_id, wg_count, &elem_offset, &elem_count);
    size_t nelems = elem_count * sizeof(T);

    if (nelems != 0) {
      T *src = source + source_displs[j] + elem_offset;
      T *dst = &tmp_buf[my_pe_in_team * tmp_buf_off];
      put_nbi_shared_qp(qp_index, dst, src, nelems, true);
    }

    publish_completion_shared_qp(qp_index, &pSync[alltoall_pSync_offset + my_pe_in_team], completion_count);
  }

  // wait until everyone has obtained their designated data
  for (int j = tid; j < pe_size; j += step_size) {
    int dest_pe = team_obj->get_pe_in_world(j);
    int qp_index = get_team_qp_index(team_obj, dest_pe);

    long *sync_flags = &pSync[alltoall_pSync_offset + j];
    while (uncached_load(sync_flags) < completion_count) {
    }

    quiet_shared_qp(qp_index);
  }

  // Copy out of staging buffer
  __syncthreads();

  for (int j = 0; j < pe_size; j++) {
    const size_t total = dest_nelems[j];
    size_t elem_offset = 0;
    size_t elem_count = 0;
    split_wg_range(total, wg_id, wg_count, &elem_offset, &elem_count);
    size_t nelems = elem_count * sizeof(T);

    if (nelems != 0) {
      T *dst = dest + dest_displs[j] + elem_offset;
      T *src = (T *)((char *)&tmp_buf[j * tmp_buf_off]);
      memcpy_wg<MemcpyKind::Put>(dst, src, nelems);
    }
  }

  __syncthreads();

  if (is_thread_zero_in_block()) {
    team_obj->alltoall_sequence_number++;
  }
  __syncthreads();
}

template <typename T>
__device__ __noinline__ void GDAContext::alltoallv_get_nopxn(rocshmem_team_t team, T *dest, const size_t dest_nelems[],
                                                             const size_t dest_displs[], T *source,
                                                             const size_t source_nelems[], const size_t source_displs[],
                                                             int wg_id, int wg_count) {
  GDATeam *team_obj = reinterpret_cast<GDATeam *>(team);
  int pe_size = team_obj->num_pes;
  long *pSync = team_obj->alltoall_pSync;
  int my_pe_in_team = team_obj->my_pe;
  uint64_t a2a_sn = team_obj->alltoall_sequence_number;
  uint64_t alltoall_pSync_offset = (a2a_sn % 2) * pe_size;
  long completion_count = a2a_sn / 2 + 1;
  uint64_t *tmp_buf = (uint64_t *)team_obj->pWrk;

  const uint64_t displs_mask = 0x0000'FFFF'FFFF'FFFF;
  const uint64_t seq_mask = 0xFFFF;
  const uint64_t seq_shift = 48;

  int tid = get_flat_block_id();
  int step_size = min(get_flat_block_size(), WF_SIZE);

  /* Phase 1: publish displacement control messages to every peer. */
  for (int j = tid; j < pe_size; j += step_size) {
    uint64_t *src;
    uint64_t *dst;
    uint64_t seq_bits;
    uint64_t displ_bits;

    int dest_pe = team_obj->get_pe_in_world(j);
    int qp_index = get_team_qp_index(team_obj, dest_pe);

    /* Pack Ctrl Message * 16 bits seq | 48bit displ */
    seq_bits = (seq_mask & (a2a_sn + 1)) << seq_shift;
    const size_t send_total = source_nelems[j];
    size_t send_offset = 0;
    [[maybe_unused]] size_t send_count = 0;
    split_wg_range(send_total, wg_id, wg_count, &send_offset, &send_count);
    displ_bits = (displs_mask & (source_displs[j] + send_offset));
    uint64_t ctrl_msg = seq_bits | displ_bits;

    /* Prepare Ctrl Message */
    src = (uint64_t *)&ctrl_msg;
    dst = &tmp_buf[constmem.my_pe];

    put_nbi_shared_qp(qp_index, dst, src, sizeof(uint64_t), true);
    quiet_shared_qp(qp_index);
  }

  __syncthreads();

  /* Phase 2: consume every control message and issue the corresponding GET. */
  for (int j = tid; j < pe_size; j += step_size) {
    int dest_pe = team_obj->get_pe_in_world(j);
    int qp_index = get_team_qp_index(team_obj, dest_pe);
    uint64_t ctrl_value;
    uint64_t *vol_ctrl = &tmp_buf[dest_pe];
    uint64_t seq_bits;
    uint64_t displ_bits;

    do {
      ctrl_value = uncached_load(vol_ctrl);
      seq_bits = (ctrl_value >> seq_shift) & seq_mask;
      displ_bits = ctrl_value & displs_mask;
    } while (seq_bits != ((a2a_sn + 1) & seq_mask));

    /* Get data */
    const size_t recv_total = dest_nelems[j];
    size_t recv_offset = 0;
    size_t recv_count = 0;
    split_wg_range(recv_total, wg_id, wg_count, &recv_offset, &recv_count);
    size_t nelems = recv_count * sizeof(T);
    const void *src = reinterpret_cast<const char *>(source) + (displ_bits * sizeof(T));
    void *dst = dest + dest_displs[j] + recv_offset;

    // Most workgroups have an empty slice for small variable-count messages.
    // They still signal completion, but must not submit a zero-byte GET.
    if (nelems != 0) {
      get_nbi_shared_qp(qp_index, dst, src, nelems);
    }

    /* Put Completion */
    publish_completion_shared_qp(qp_index, &pSync[alltoall_pSync_offset + my_pe_in_team], completion_count);
  }

  __syncthreads();

  /* Phase 3: wait for all peers and drain the QPs used by this team. */
  for (int j = tid; j < pe_size; j += step_size) {
    int dest_pe = team_obj->get_pe_in_world(j);
    int qp_index = get_team_qp_index(team_obj, dest_pe);
    long *sync_flags = &pSync[alltoall_pSync_offset + j];
    while (uncached_load(sync_flags) < completion_count) {
    }

    quiet_shared_qp(qp_index);
  }

  if (is_thread_zero_in_block()) {
    team_obj->alltoall_sequence_number++;
  }

  __syncthreads();
}

// Owns path dispatch and the complete collective operation.
template <typename T>
__device__ __noinline__ void GDAContext::alltoallv_das(rocshmem_team_t team, T *dest, const size_t dest_nelems[],
                                                       const size_t dest_displs[], T *source,
                                                       const size_t source_nelems[], const size_t source_displs[],
                                                       int wg_id, int wg_count) {
  // The on-stream launcher supplies one independent team per workgroup.
  // Partition every PE pair exactly as the latest COPY/GET implementations do.
  GDATeam *pxn_team = reinterpret_cast<GDATeam *>(team);
  if (internal_try_collective_pxn(pxn_team, pxn_team->tinfo_wrt_world->pe_start, pxn_team->tinfo_wrt_world->stride,
      pxn_team->num_pes, "alltoallv",
      [&] __device__() {
        if (!proxy::IsWorldProxySyncEligible(
                constmem.my_pe, pxn_team->tinfo_wrt_world->pe_start,
                pxn_team->tinfo_wrt_world->stride, pxn_team->num_pes)) {
          return false;
        }
        internal_alltoallv_wg_world_proxy_pxn(pxn_team, dest, dest_nelems, dest_displs, source,
                                          source_nelems, source_displs, wg_id, wg_count);
        return true;
      })) {
    return;
  }

  GDATeam *team_obj = reinterpret_cast<GDATeam *>(team);
  const int pe_size = team_obj->num_pes;
  const size_t nelems = source_nelems[0];
  bool regular_layout = nelems <= 0x7fffffff;

  for (int pe = 0; pe < pe_size && regular_layout; pe++) {
    regular_layout = source_nelems[pe] == nelems && dest_nelems[pe] == nelems &&
                     source_displs[pe] == static_cast<size_t>(pe) * nelems &&
                     dest_displs[pe] == static_cast<size_t>(pe) * nelems;
  }

  if (regular_layout) {
    size_t elem_offset = 0;
    size_t elem_count = 0;
    split_wg_range(nelems, wg_id, wg_count, &elem_offset, &elem_count);
    alltoall_wg_das(team, dest, source, static_cast<int>(nelems), static_cast<int>(elem_offset),
                    static_cast<int>(elem_count));
    return;
  }

  if (constmem.alltoall_wg_algo == gda::ALLTOALLV_WG_ALGO_COPY) {
    alltoallv_copy_nopxn(team, dest, dest_nelems, dest_displs, source, source_nelems, source_displs, wg_id, wg_count);
  } else {
    alltoallv_get_nopxn(team, dest, dest_nelems, dest_displs, source, source_nelems, source_displs, wg_id, wg_count);
  }
}

}  // namespace rocshmem

#endif  // USE_DAS
