// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#pragma once

#if defined(USE_DAS)

namespace rocshmem {

namespace proxy {

// Preserve the original full-vector traversal; sliced reduce-scatter visits
// only this WG's interval within every destination block.
struct ProxyReductionRange {
  size_t block_elements;
  size_t total_elements;
  size_t element_offset;
  size_t element_count;
  bool is_partitioned_scatter;

  __device__ static ProxyReductionRange Make(int nelems, int peers, bool scatter, int elem_offset, int elem_count) {
    const size_t element_count = elem_count < 0 ? nelems : elem_count;
    return {static_cast<size_t>(nelems), scatter ? static_cast<size_t>(peers) * nelems : nelems,
            static_cast<size_t>(elem_offset), element_count,
            scatter && (elem_offset != 0 || element_count != static_cast<size_t>(nelems))};
  }

  __device__ int segment_count(int peers) const { return is_partitioned_scatter ? peers : 1; }

  __device__ size_t segment_begin(int segment) const {
    return is_partitioned_scatter ? segment * block_elements + element_offset : 0;
  }

  __device__ size_t segment_end(int segment) const {
    return is_partitioned_scatter ? segment_begin(segment) + element_count : total_elements;
  }
};

}  // namespace proxy

/*
 * Non-PXN allreduce: reduce local proxy segments, GET remote partials on
 * the root domain, then GET the completed segments and distribute via IPC.
 * Keep the original proxy ownership, QP rows and synchronization phases.
 */
template <typename T, ROCSHMEM_OP Op>
__device__ __noinline__ void GDAContext::internal_reduce_wg_proxy_nopxn(T *dst, const T *src, int nelems, GDATeam *team_obj,
                                                     ActiveWFInfo &wf_info) {
  const int pe_start = team_obj->tinfo_wrt_world->pe_start;
  const int pe_size = team_obj->tinfo_wrt_world->size;
  const int local_size = constmem.ipc_shm_size;
  const int local_slot = ipcImpl_.shm_rank;
  const int node_count = pe_size / local_size;
  const int root_node_first = pe_start;
  const bool on_root_node = constmem.ipc_first_pe == root_node_first;
  long *p_sync = team_obj->reduce_pSync;
  T *p_wrk = reinterpret_cast<T *>(team_obj->pWrk);
  const int wg_id = get_flat_block_id();
  const int wg_size = get_flat_block_size();
  constexpr size_t MIN_BYTES_PER_PROXY = 64 * 1024;
  const size_t nbytes = static_cast<size_t>(nelems) * sizeof(T);
  const size_t useful_proxies = (nbytes + MIN_BYTES_PER_PROXY - 1) / MIN_BYTES_PER_PROXY;
  const int proxies = static_cast<int>(min(static_cast<size_t>(local_size), useful_proxies));
  const int proxy_chunk = (nelems + proxies - 1) / proxies;

  if (local_slot < proxies) {
    const int begin = local_slot * proxy_chunk;
    const int count = min(proxy_chunk, nelems - begin);
    internal_fused_ipc_reduce<T, Op>(dst + begin, src, begin, count, local_size, local_slot, wg_id, wg_size);
  }

  // Publish local partials before a remote proxy fetches them.
  threadfence_system();
  __syncthreads();
  internal_sync_wg(constmem.my_pe, pe_start, 1, pe_size, p_sync, wf_info);

  // The first node's proxies reduce matching partials from all other nodes.
  if (on_root_node && local_slot < proxies) {
    const int begin = local_slot * proxy_chunk;
    const int count = min(proxy_chunk, nelems - begin);
    const int tile_elems = max(1, static_cast<int>(ROCSHMEM_REDUCE_MIN_WRKDATA_SIZE * sizeof(double) / sizeof(T)));
    for (int node = 1; node < node_count; node++) {
      const int remote_proxy = pe_start + node * local_size + local_slot;
      for (int offset = 0; offset < count; offset += tile_elems) {
        const int tile = min(tile_elems, count - offset);
        const uint32_t qp_row = static_cast<uint32_t>(blockIdx.x) % num_qps_per_pe;
        const int qp_index = qp_row * constmem.num_pes + remote_proxy;
        internal_getmem_wg(p_wrk, dst + begin + offset, static_cast<size_t>(tile) * sizeof(T), remote_proxy, qp_index, wf_info);
        gda_compute_reduce<T, Op>(p_wrk, dst + begin + offset, tile, wg_id, wg_size);
      }
    }
  }

  internal_sync_wg(constmem.my_pe, pe_start, 1, pe_size, p_sync, wf_info);

  // Matching proxies fetch their final segment from the reduction node.
  if (!on_root_node && local_slot < proxies) {
    const int begin = local_slot * proxy_chunk;
    const int count = min(proxy_chunk, nelems - begin);
    const int root_proxy = root_node_first + local_slot;
    const uint32_t qp_row = static_cast<uint32_t>(blockIdx.x) % num_qps_per_pe;
    const int qp_index = qp_row * constmem.num_pes + root_proxy;
    internal_getmem_wg(dst + begin, dst + begin, static_cast<size_t>(count) * sizeof(T), root_proxy, qp_index, wf_info);
  }

  internal_sync_wg(constmem.my_pe, pe_start, 1, pe_size, p_sync, wf_info);

  // Assemble the complete result from local proxy segments.
  for (int slot = 0; slot < proxies; slot++) {
    if (slot == local_slot) continue;
    const int begin = slot * proxy_chunk;
    const int count = min(proxy_chunk, nelems - begin);
    const int local_proxy = constmem.ipc_first_pe + slot;
    internal_getmem_wg(dst + begin, dst + begin,
                       static_cast<size_t>(count) * sizeof(T), local_proxy,
                       local_proxy, wf_info);
  }

  internal_sync_wg(constmem.my_pe, pe_start, 1, pe_size, p_sync, wf_info);
}

/**
 * @brief Try the proxy allreduce path selected for the current NIC policy.
 *
 * @param team_obj Team and proxy layout metadata.
 * @param dest Reduction output buffer.
 * @param source Local contribution buffer.
 * @param nelems Number of elements.
 * @param wf_info Active workgroup metadata.
 * @return true when a proxy path completed the reduction and team barrier.
 */
template <typename T, ROCSHMEM_OP Op>
__device__ bool GDAContext::internal_reduce_wg_pxn(GDATeam *team_obj, T *dest, const T *source,
                                                            int nelems, ActiveWFInfo &wf_info) {
  if (nelems <= 0) {
    return false;
  }

  const auto *info = team_obj->tinfo_wrt_world;
  if (!constmem.pxn_enabled) {
    // Preserve the validated unrestricted segmented algorithm and thresholds.
    const int local_size = constmem.ipc_shm_size;
    if (static_cast<size_t>(nelems) * sizeof(T) < kHierarchicalMinBytes || info->stride != 1 || local_size <= 1 ||
        constmem.ipc_stride != 1 || info->pe_start % local_size != 0 || info->size % local_size != 0 ||
        info->size <= local_size) {
      return false;
    }
    internal_reduce_wg_proxy_nopxn<T, Op>(dest, source, nelems, team_obj, wf_info);
  } else {
    if (proxy::SelectProxyCollectiveLayout(team_obj, info->pe_start, info->stride, team_obj->num_pes) ==
        proxy::ProxyCollectiveLayout::None) {
      return false;
    }
    internal_reduce_wg_ipc_domain_pxn<T, Op, false>(team_obj, dest, source, nelems);
  }
  barrier_wg(reinterpret_cast<rocshmem_team_t>(team_obj));
  return true;
}

// PXN allreduce and reduce-scatter share local reduction and domain exchange;
// only the output interval differs. Native reduce-scatter retains its ring.
template <typename T, ROCSHMEM_OP Op>
__device__ bool GDAContext::internal_reduce_scatter_wg_pxn(GDATeam *team_obj, T *dest, const T *source,
                                                                 int nelems, int elem_offset, int elem_count) {
  if (nelems <= 0) {
    return false;
  }

  const auto *info = team_obj->tinfo_wrt_world;
  if (!constmem.pxn_enabled) {
    const int local_size = constmem.ipc_shm_size;
    int local_slot = -1;
    if (!ipcImpl_.isIpcAvailable(constmem.my_pe, constmem.my_pe, &local_slot) || local_size <= 1 || info->stride != 1 ||
        constmem.ipc_stride != 1 || info->pe_start % local_size != 0 || team_obj->num_pes <= local_size ||
        team_obj->num_pes % local_size != 0 || nelems <= 1024) {
      return false;
    }
    if (hybrid_reduce_scatter_wg_nopxn<T, Op>(team_obj, dest, source, nelems, elem_offset, elem_count, info->pe_start,
                                        info->stride, team_obj->num_pes, team_obj->my_pe) != ROCSHMEM_SUCCESS) {
      LOGD_ERROR_ABORT("Non-PXN reduce-scatter ring failed");
    }
    // The original ring completes its own synchronization.
    return true;
  } else {
    if (proxy::SelectProxyCollectiveLayout(team_obj, info->pe_start, info->stride, team_obj->num_pes) ==
        proxy::ProxyCollectiveLayout::None) {
      return false;
    }
    internal_reduce_wg_ipc_domain_pxn<T, Op, true>(team_obj, dest, source, nelems, elem_offset, elem_count);
  }
  barrier_wg(reinterpret_cast<rocshmem_team_t>(team_obj));
  return true;
}

/**
 * @brief Reduce local team contributions before exchanging domain partials.
 *
 * Distinct owners reduce disjoint tiles concurrently in every IPC domain.
 * Root-domain owners fold incoming tiles and return the results to their peers.
 * Reduce-scatter copies only the caller's output interval through IPC.
 */
template <typename T, ROCSHMEM_OP Op, bool ReduceScatter>
__device__ __noinline__ void GDAContext::internal_reduce_wg_ipc_domain_pxn(GDATeam *team_obj, T *dest,
                                                                             const T *source, int nelems,
                                                                             int elem_offset, int elem_count) {
  constexpr size_t tile_elements = ROCSHMEM_REDUCE_MIN_WRKDATA_SIZE * sizeof(double) / (2 * sizeof(T));
  T *partial = reinterpret_cast<T *>(team_obj->pWrk);
  T *received = partial + tile_elements;
  const auto *info = team_obj->tinfo_wrt_world;
  const auto view = proxy::GetProxyTeamView(team_obj, info->pe_start, info->stride, team_obj->num_pes);
  const auto range =
      proxy::ProxyReductionRange::Make(nelems, team_obj->num_pes, ReduceScatter, elem_offset, elem_count);
  const size_t wanted_begin = ReduceScatter ? static_cast<size_t>(team_obj->my_pe) * nelems : 0;
  const size_t wanted_end = wanted_begin + nelems;
  const uint64_t operation = team_obj->proxy_data_route_sequence + team_obj->pool_index_;
  const size_t total_bytes = range.total_elements * sizeof(T);
  const int requested = min(min(8, ROCSHMEM_BARRIER_SYNC_SIZE / view.ipc_domain_count),
                            static_cast<int>(max(size_t{1}, (total_bytes + 65535) / 65536)));
  int lanes = 0;
  const int *owners = internal_build_reduction_proxy_plan_wg(team_obj, operation, requested, lanes);
  int local_domain = -1;
  int local_lane = -1;

  for (int domain = 0; domain < view.ipc_domain_count; domain++) {
    if (view.domain_first(domain) == constmem.ipc_first_pe) {
      local_domain = domain;
      for (int lane = 0; lane < lanes; lane++) {
        if (owners[domain * requested + lane] == constmem.my_pe) {
          local_lane = lane;
        }
      }
    }
  }
  if (local_domain < 0) {
    LOGD_ERROR_ABORT("PXN reduction has no local IPC domain");
  }

  const int tid = get_flat_block_id();
  const int block_size = get_flat_block_size();
  const int qp_row = team_obj->pool_index_ % num_qps_per_pe;
  long *p_sync = team_obj->reduce_pSync;
  int64_t epoch = static_cast<int64_t>(team_obj->proxy_reduce_sync_epoch);
  ActiveWFInfo wf_info(ctx_id_, ThreadScope::wg);

  // One protocol for WORLD/partial teams and both reduction outputs:
  // parallel local segments -> fold at root-domain owners -> distribute results.
  // Increasing tile rounds keep in-place scatter stores behind consumed input.
  for (int segment = 0; segment < range.segment_count(team_obj->num_pes); segment++) {
    const size_t finish = range.segment_end(segment);
    const size_t length = finish - range.segment_begin(segment);
    const size_t alignment = max(size_t{1}, size_t{16} / sizeof(T));
    const size_t tile_count =
        min(tile_elements, ((length + lanes - 1) / lanes + alignment - 1) / alignment * alignment);
    for (size_t base = range.segment_begin(segment); base < finish; base += lanes * tile_count) {
      const size_t offset = base + static_cast<size_t>(max(0, local_lane)) * tile_count;
      const int count = local_lane >= 0 && offset < finish ? static_cast<int>(min(tile_count, finish - offset)) : 0;
      if (count > 0 && view.is_world) {
        // Contiguous IPC membership permits the existing vectorized local fold.
        // This changes only memory access, not the segment/exchange protocol.
        internal_fused_ipc_reduce<T, Op>(partial, source + offset, 0, count, constmem.ipc_shm_size,
                                         ipcImpl_.shm_rank, tid, block_size);
      } else if (count > 0) {
        bool first = true;
        for (int rank = 0; rank < team_obj->num_pes; rank++) {
          const int member = team_obj->get_pe_in_world(rank);
          if (member < constmem.ipc_first_pe || member >= proxy::WorldProxyIpcDomainEnd(constmem.ipc_first_pe)) {
            continue;
          }
          T *input = const_cast<T *>(source + offset);
          if (member != constmem.my_pe) {
            int local_slot = -1;
            if (!ipcImpl_.isIpcAvailable(constmem.my_pe, member, &local_slot)) {
              LOGD_ERROR_ABORT("PXN reduction member is not IPC accessible");
            }
            input = reinterpret_cast<T *>(get_local_ptr(source + offset, local_slot));
          }
          for (int i = tid; i < count; i += block_size) {
            T value = uncached_load(input + i);
            if (first) {
              partial[i] = value;
            } else {
              OpWrap<Op>::Calc(&value, partial + i, 0);
            }
          }
          __syncthreads();
          first = false;
        }
      }
      internal_proxy_sync_wg(team_obj, p_sync, wf_info, epoch++);

      for (int domain = 1; domain < view.ipc_domain_count; domain++) {
        if (local_domain == domain && count > 0) {
          internal_proxy_putmem_wg(received, partial, static_cast<size_t>(count) * sizeof(T),
                                   owners[local_lane], qp_row, wf_info);
        }
        internal_proxy_sync_wg(team_obj, p_sync, wf_info, epoch++);
        if (local_domain == 0 && count > 0) {
          for (int i = tid; i < count; i += block_size) {
            T value = uncached_load(received + i);
            OpWrap<Op>::Calc(&value, partial + i, 0);
          }
          __syncthreads();
        }
        internal_proxy_sync_wg(team_obj, p_sync, wf_info, epoch++);
      }
      if (local_domain == 0 && count > 0) {
        for (int domain = 1; domain < view.ipc_domain_count; domain++) {
          internal_proxy_putmem_wg(partial, partial, static_cast<size_t>(count) * sizeof(T),
                                   owners[domain * requested + local_lane], qp_row, wf_info);
        }
      }
      internal_proxy_sync_wg(team_obj, p_sync, wf_info, epoch++);

      // Allreduce reads every segment; reduce-scatter reads only its own output.
      for (int lane = 0; lane < lanes; lane++) {
        const size_t lane_begin = base + lane * tile_count;
        const size_t begin = max(lane_begin, wanted_begin);
        const size_t end = min(min(lane_begin + tile_count, finish), wanted_end);
        if (begin < end) {
          internal_proxy_getmem_wg(dest + begin - wanted_begin, partial + begin - lane_begin, (end - begin) * sizeof(T),
                                   owners[local_domain * requested + lane], 0, wf_info);
        }
      }
      internal_proxy_sync_wg(team_obj, p_sync, wf_info, epoch++);
    }
  }

  if (is_thread_zero_in_block()) {
    team_obj->proxy_reduce_sync_epoch = epoch;
    team_obj->proxy_data_route_sequence++;
  }
  __syncthreads();
}

// Wave reductions share one tiled broadcast-and-fold algorithm. Every source
// tile is captured before any in-place output is changed.
template <typename T, ROCSHMEM_OP Op, bool ReduceScatter>
__device__ bool GDAContext::reduce_wave_pxn(rocshmem_team_t team, T *dst, const T *src, int nelems) {
  GDATeam *team_obj = reinterpret_cast<GDATeam *>(team);
  const auto *info = team_obj->tinfo_wrt_world;
  return internal_try_collective_pxn(team_obj, info->pe_start, info->stride, team_obj->num_pes, "reduce_wave",
      [&] __device__() {
        if (proxy::SelectProxyCollectiveLayout(team_obj, info->pe_start, info->stride, team_obj->num_pes) ==
            proxy::ProxyCollectiveLayout::None) {
          return false;
        }

        constexpr size_t capacity = ROCSHMEM_REDUCE_MIN_WRKDATA_SIZE * sizeof(double) / (2 * sizeof(T));
        T *stage = static_cast<T *>(team_obj->pWrk);
        T *received = stage + capacity;
        const size_t total = ReduceScatter ? static_cast<size_t>(team_obj->num_pes) * nelems : nelems;
        const size_t wanted_begin = ReduceScatter ? static_cast<size_t>(team_obj->my_pe) * nelems : 0;
        const size_t wanted_end = wanted_begin + nelems;
        ActiveWFInfo wf_info(ctx_id_, ThreadScope::wave);
        int64_t epoch = static_cast<int64_t>(team_obj->proxy_reduce_sync_epoch);
        long *p_sync = team_obj->reduce_pSync;
        if (total == 0) {
          internal_proxy_sync_wave(team_obj, p_sync, wf_info, epoch++);
        }

        for (size_t offset = 0; offset < total; offset += capacity) {
          const size_t count = min(capacity, total - offset);
          memcpy_wave<MemcpyKind::Get>(stage, const_cast<T *>(src + offset), count * sizeof(T));
          internal_proxy_sync_wave(team_obj, p_sync, wf_info, epoch++);
          const size_t begin = max(offset, wanted_begin);
          const size_t end = min(offset + count, wanted_end);
          for (int rank = 0; rank < team_obj->num_pes; rank++) {
            if (!internal_broadcastmem_wave_pxn(team_obj, received, stage, count * sizeof(T),
                                                   team_obj->get_pe_in_world(rank), p_sync, wf_info, epoch)) {
              return false;
            }
            for (size_t i = begin + get_flat_block_id() % WF_SIZE; i < end; i += WF_SIZE) {
              T value = uncached_load(received + i - offset);
              if (rank == 0) {
                dst[i - wanted_begin] = value;
              } else {
                OpWrap<Op>::Calc(&value, dst + i - wanted_begin, 0);
              }
            }
            internal_proxy_sync_wave(team_obj, p_sync, wf_info, epoch++);
          }
        }
        if (is_thread_zero_in_wave()) {
          team_obj->proxy_reduce_sync_epoch = epoch;
        }
        __threadfence_system();
        __builtin_amdgcn_wave_barrier();
        return true;
      });
}

/**
 * @brief Fused IPC-aware vectorized reduction with single write optimization
 *
 * Performs an optimized reduction operation across IPC (Inter-Process Communication)
 * peers within the same node. This function reduces data from all local GPUs
 * (sharing memory via IPC) and writes the final result exactly once to minimize
 * memory bandwidth consumption.
 */
template <typename T, ROCSHMEM_OP Op>
__device__ __noinline__ void GDAContext::internal_fused_ipc_reduce(T *dst, const T *src, int source_offset, int count,
                                                                   int local_size, int local_slot, int wg_id,
                                                                   int wg_size) {
  using VecT = std::conditional_t<
      std::is_same_v<T, float>, float4,
      std::conditional_t<std::is_same_v<T, double>, double2,
                         std::conditional_t<std::is_same_v<T, int>, int4,
                                            std::conditional_t<std::is_same_v<T, unsigned int>, uint4, T>>>>;
  constexpr int vec_width = sizeof(VecT) / sizeof(T);
  constexpr bool vectorizable = !std::is_same_v<VecT, T>;
  bool used_vector_path = false;

  if constexpr (vectorizable) {
    const T *segment_src = src + source_offset;
    const bool aligned = (reinterpret_cast<uintptr_t>(segment_src) % alignof(VecT)) == 0 &&
                         (reinterpret_cast<uintptr_t>(dst) % alignof(VecT)) == 0;
    if (aligned && (source_offset % vec_width) == 0 && (count % vec_width) == 0) {
      const int vectors = count / vec_width;
      for (int i = wg_id; i < vectors; i += wg_size) {
        VecT result = reinterpret_cast<const VecT *>(segment_src)[i];
        T *result_elem = reinterpret_cast<T *>(&result);
        for (int slot = 0; slot < local_size; ++slot) {
          if (slot == local_slot) continue;
          const T *peer_src = reinterpret_cast<const T *>(get_local_ptr(src, slot)) + source_offset;
          VecT value = reinterpret_cast<const VecT *>(peer_src)[i];
          T *value_elem = reinterpret_cast<T *>(&value);
          for (int v = 0; v < vec_width; ++v) {
            OpWrap<Op>::Calc(value_elem + v, result_elem + v, 0);
          }
        }
        reinterpret_cast<VecT *>(dst)[i] = result;
      }
      used_vector_path = true;
    }
  }

  // Scalar fallback path for unaligned or unsupported types
  // Processes one element at a time with correct semantics but lower throughput
  if (!used_vector_path) {
    for (int i = wg_id; i < count; i += wg_size) {
      T result = src[source_offset + i];
      for (int slot = 0; slot < local_size; ++slot) {
        if (slot == local_slot) continue;
        const T *peer_src = reinterpret_cast<const T *>(get_local_ptr(src, slot)) + source_offset;
        OpWrap<Op>::Calc(const_cast<T *>(peer_src + i), &result, 0);
      }
      dst[i] = result;
    }
  }
  __syncthreads();
}

/*
 * Visual representation of the ring_allreduce algorithm below
 * assuming 4 PEs and a single segment.
 *
 *         Initial state
 *  PE#     0              1             2              3
 *        [00]           [10]          [20]           [30]
 *        [01]           [11]          [21]           [31]
 *        [02]           [12]          [22]           [32]
 *        [03]           [13]          [23]           [33]
 *
 * Loop 1:
 *        iter 0
 *  PE#     0              1             2              3
 *        [00+30]        [10]          [20]           [30]
 *        [01]           [01+11]       [21]           [31]
 *        [02]           [12]          [12+22]        [32]
 *        [03]           [13]          [23]           [23+33]
 *
 *        iter 1
 *  PE#     0              1             2              3
 *        [00+30]        [00+10+30]    [20]           [30]
 *        [01]           [01+11]       [01+11+21]     [31]
 *        [02]           [12]          [12+22]        [12+22+32]
 *        [03+23+33]     [13]          [23]           [23+33]
 *
 *        iter 2
 *  PE#     0              1             2              3
 *        [00+30]        [00+10+30]    [00+10+20+30]  [30]
 *        [01]           [01+11]       [01+11+21]     [01+11+21+31]
 *        [02+12+22+32]  [12]          [12+22]        [12+22+32]
 *        [03+23+33]     [03+13+23+33] [23]           [23+33]
 *
 * Loop 2:
 *
 *       iter 3
 *  PE#     0              1             2              3
 *        [00+30]        [00+10+30]    [00+10+20+30]  [00+10+20+30]
 *        [01+11+21+31]  [01+11]       [01+11+21]     [01+11+21+31]
 *        [02+12+22+32]  [02+12+22+32] [12+22]        [12+22+32]
 *        [03+23+33]     [03+13+23+33] [03+13+23+33]  [23+33]
 *
 *       iter 4
 *  PE#     0              1             2              3
 *        [00+10+20+30]  [00+10+30]    [00+10+20+30]  [00+10+20+30]
 *        [01+11+21+31]  [01+11+21+31] [01+11+21]     [01+11+21+31]
 *        [02+12+22+32]  [02+12+22+32] [02+12+22+32]  [12+22+32]
 *        [03+23+33]     [03+13+23+33] [03+13+23+33]  [03+13+23+33]
 *
 *        iter 5
 *  PE#     0              1             2              3
 *        [00+10+20+30]  [00+10+20+30] [00+10+20+30]  [00+10+20+30]
 *        [01+11+21+31]  [01+11+21+31] [01+11+21+31]  [01+11+21+31]
 *        [02+12+22+32]  [02+12+22+32] [02+12+22+32]  [02+12+22+32]
 *        [03+13+23+33]  [03+13+23+33] [03+13+23+33]  [03+13+23+33]
 */
template <typename T, ROCSHMEM_OP Op>
__device__ void GDAContext::internal_ring_allreduce_das(T *dst, const T *src, int nelems,
                                                        GDATeam *team_obj,  // NOLINT(runtime/int)
                                                        int n_seg, int seg_size, int chunk_size,
                                                        ActiveWFInfo &wf_info) {
  int PE_size = team_obj->tinfo_wrt_world->size;
  long *pSync = team_obj->reduce_pSync;
  T *pWrk = reinterpret_cast<T *>(team_obj->pWrk);
  int my_pe_in_team = team_obj->my_pe;
  long sequence_base = team_obj->reduce_sequence_number;

  int off_seg, off_send, off_recv;
  int send_pe = (my_pe_in_team + 1) % PE_size;
  // send_pe is relative to team, convert it relative to team world
  send_pe = team_obj->get_pe_in_world(send_pe);
  long wait_val;  // NOLINT(runtime/int)

  int wg_size = get_flat_block_size();
  int wg_id = get_flat_block_id();

  for (int i = wg_id; i < nelems; i += wg_size) {
    dst[i] = src[i];
  }
  __syncthreads();

  for (int seg = 0; seg < n_seg; seg++) {
    off_seg = seg * seg_size;
    // Loop 2 in the algorithm above
    for (int iter = 0; iter < PE_size - 1; iter++) {
      off_send = (((my_pe_in_team + 1 - iter + 2 * PE_size) % PE_size) * chunk_size);
      off_recv = (((my_pe_in_team - iter + 2 * PE_size) % PE_size) * chunk_size);

      // Keep the payload and its completion signal on the same RC QP.  RC
      // ordering makes the payload remotely visible before the signal, and
      // avoids quieting every QP in the context at every ring step.
      int qp_index = get_qp_index(send_pe, wf_info);
      internal_putmem_nbi_wg(reinterpret_cast<void *>(&pWrk[off_send]),
                             reinterpret_cast<void *>(&dst[off_send + off_seg]), chunk_size * sizeof(T),
                             send_pe, qp_index, wf_info);

      if (is_thread_zero_in_block()) {
        wait_val = sequence_base + seg + 1;
        internal_putmem(&pSync[iter], &wait_val, sizeof(*pSync), send_pe, qp_index, wf_info);
#if ROCSHMEM_ARCH_HYGON
        __threadfence_system();
#endif  // ROCSHMEM_ARCH_HYGON
        wait_until(&pSync[iter], ROCSHMEM_CMP_EQ, wait_val);
      }
      __syncthreads();
      gda_compute_reduce<T, Op>(&pWrk[off_recv], &dst[off_seg + off_recv], chunk_size, wg_id, wg_size);
    }

    // Loop 2 in the example above
    for (int iter = PE_size - 1; iter < 2 * PE_size - 2; iter++) {
      off_send = (((my_pe_in_team + 1 - iter + 2 * PE_size) % PE_size) * chunk_size);
      int qp_index = get_qp_index(send_pe, wf_info);
      internal_putmem_nbi_wg(reinterpret_cast<void *>(&dst[off_send + off_seg]),
                             reinterpret_cast<void *>(&dst[off_send + off_seg]), chunk_size * sizeof(T),
                             send_pe, qp_index, wf_info);

      if (is_thread_zero_in_block()) {
        wait_val = sequence_base + seg + 1;
        internal_putmem(&pSync[iter], &wait_val, sizeof(*pSync), send_pe, qp_index, wf_info);
#if ROCSHMEM_ARCH_HYGON
        __threadfence_system();
#endif  // ROCSHMEM_ARCH_HYGON
        wait_until(&pSync[iter], ROCSHMEM_CMP_EQ, wait_val);
      }
      __syncthreads();
    }
  }
  __syncthreads();

  if (is_thread_zero_in_block()) {
    team_obj->reduce_sequence_number += n_seg;
  }
  __syncthreads();
}

template <typename T, ROCSHMEM_OP Op>
__device__ int GDAContext::reduce_wg_das(rocshmem_team_t team, T *dest, const T *source, int nreduce) {
  GDATeam *team_obj = reinterpret_cast<GDATeam *>(team);

  int PE_size = team_obj->tinfo_wrt_world->size;

  size_t direct_pWrk = PE_size * nreduce;
  size_t direct_pSync = PE_size;
  size_t ring_pSync = 2 * PE_size;
  size_t provided_pWrk = ROCSHMEM_REDUCE_MIN_WRKDATA_SIZE;
  size_t provided_pSync = ROCSHMEM_REDUCE_SYNC_SIZE;

  ActiveWFInfo wf_info(ctx_id_, ThreadScope::wg);

  const int pe_start = team_obj->tinfo_wrt_world->pe_start;
  const int stride = team_obj->tinfo_wrt_world->stride;
  if (internal_try_collective_pxn(team_obj, pe_start, stride, PE_size, "allreduce",
      [&] __device__() {
        if (proxy::SelectProxyCollectiveLayout(team_obj, pe_start, stride, PE_size) ==
            proxy::ProxyCollectiveLayout::None) {
          return false;
        }
        internal_try_sync_pxn(team_obj, team_obj->barrier_pSync, ThreadScope::wg);
        return nreduce == 0 || internal_reduce_wg_pxn<T, Op>(team_obj, dest, source, nreduce, wf_info);
      })) {
    return ROCSHMEM_SUCCESS;
  }

  // Both NIC policies use the shared proxy dispatcher and completion barrier.
  if (!constmem.pxn_enabled && internal_reduce_wg_pxn<T, Op>(team_obj, dest, source, nreduce, wf_info)) {
    return ROCSHMEM_SUCCESS;
  }

  // Messages above DIRECT_MAX use ring. default 8192 (8KB).
  constexpr int DIRECT_MAX_NELEMS = 8192;

  bool use_direct =
      (provided_pWrk >= direct_pWrk) && (provided_pSync >= direct_pSync) && (nreduce <= DIRECT_MAX_NELEMS);
  if (use_direct) {
    internal_direct_allreduce_wg<T, Op>(dest, source, nreduce, team_obj, wf_info);
  } else {
    if (ring_pSync <= ROCSHMEM_REDUCE_SYNC_SIZE) {
      size_t ring_pWrk = provided_pWrk;
      // integer division truncating value
      int chunk_size = ring_pWrk / PE_size;
      int seg_size = chunk_size * PE_size;

      // integer division truncating value
      int n_seg = nreduce / seg_size;
      // integer division rounding up
      int n_seg_up = (nreduce - 1) / seg_size + 1;
      // recalculate chunk_size
      chunk_size = seg_size / PE_size;

      if (n_seg > 0) {
        internal_ring_allreduce_das<T, Op>(dest, source, nreduce, team_obj, n_seg, seg_size, chunk_size, wf_info);
      }
      if (n_seg_up > n_seg) {
        T *p_dst = (dest + (n_seg * seg_size));
        const T *p_src = (source + (n_seg * seg_size));
        int p_count = nreduce - (n_seg * seg_size);
        int p_chunk = p_count / PE_size;

        if (p_chunk > 0) {
          internal_ring_allreduce_das<T, Op>(p_dst, p_src, (p_chunk * PE_size), team_obj, 1, (p_chunk * PE_size),
                                             p_chunk, wf_info);
        }

        if ((p_chunk * PE_size) < p_count) {
          // Final elements need to use direct_allreduce
          p_count -= (p_chunk * PE_size);
          p_dst += (p_chunk * PE_size);
          const T *p_src2 = p_src + (p_chunk * PE_size);

          internal_direct_allreduce_wg<T, Op>(p_dst, p_src2, p_count, team_obj, wf_info);
        }
      }
    } else {
      LOGD_WARN("Unsupported reduction size for GDA conduit.");
      return ROCSHMEM_ERROR;
    }
  }
  barrier_wg(team);
  return ROCSHMEM_SUCCESS;
}

/*
 * Reduce-scatter: PE r receives the element-wise reduction of
 * source[r*nreduce .. (r+1)*nreduce - 1] across all PEs into dest[0..nreduce-1].
 *
 * The operation uses team-owned pWrk and pSync. Concurrent workgroups must
 * use distinct team instances and input/output buffers. All threads in the
 * invoking workgroup must participate in the operation and matching barriers.
 */
template <typename T, ROCSHMEM_OP Op>
__device__ int GDAContext::reduce_scatter_wg_das(rocshmem_team_t team, T *dest, const T *source, int nreduce,
                                                 int elem_offset, int elem_count) {
  GDATeam *team_obj = reinterpret_cast<GDATeam *>(team);
  int PE_size = team_obj->tinfo_wrt_world->size;
  int PE_start = team_obj->tinfo_wrt_world->pe_start;
  int stride = team_obj->tinfo_wrt_world->stride;
  int my_pe = constmem.my_pe;
  int team_rank = (my_pe - PE_start) / stride;

  if (elem_count < 0) elem_count = nreduce;
  if (internal_try_collective_pxn(team_obj, PE_start, stride, PE_size, "reduce-scatter",
      [&] __device__() {
        if (proxy::SelectProxyCollectiveLayout(team_obj, PE_start, stride, PE_size) ==
            proxy::ProxyCollectiveLayout::None) {
          return false;
        }
        // Publish all inputs before a proxy reads another PE's IPC mapping.
        internal_try_sync_pxn(team_obj, team_obj->barrier_pSync, ThreadScope::wg);
        return elem_count == 0 ||
               internal_reduce_scatter_wg_pxn<T, Op>(team_obj, dest, source, nreduce, elem_offset, elem_count);
      })) {
    return ROCSHMEM_SUCCESS;
  }

  if (!constmem.pxn_enabled &&
      internal_reduce_scatter_wg_pxn<T, Op>(team_obj, dest, source, nreduce, elem_offset, elem_count)) {
    return ROCSHMEM_SUCCESS;
  }

  long *pSync = team_obj->reduce_pSync;
  T *pWrk = reinterpret_cast<T *>(team_obj->pWrk);
  ActiveWFInfo wf_info(ctx_id_, ThreadScope::wg);
  int wg_id = get_flat_block_id();
  int wg_size = get_flat_block_size();
  int pWrk_elems = static_cast<int>(ROCSHMEM_REDUCE_MIN_WRKDATA_SIZE * sizeof(double) / sizeof(T));
  int chunk_size = max(1, pWrk_elems / PE_size);
  int n_chunks = (elem_count + chunk_size - 1) / chunk_size;
  int64_t flag_val = 1;
  int finish = PE_start + stride * PE_size;

  // pWrk holds one incoming contribution per PE, so process large inputs in chunks.
  for (int c = 0; c < n_chunks; c++) {
    int offset = elem_offset + c * chunk_size;
    int count = min(chunk_size, elem_offset + elem_count - offset);

    // In-place output aliases source block zero. Preserve that contribution
    // in our own scratch slot before seeding the output from another block.
    if (dest == source) {
      for (int j = wg_id; j < count; j += wg_size) {
        pWrk[team_rank * chunk_size + j] = source[offset + j];
      }
      __syncthreads();
    }

    // Seed this PE's output block with its local contribution.
    for (int j = wg_id; j < count; j += wg_size) {
      dest[offset + j] = source[team_rank * nreduce + offset + j];
    }
    __threadfence_system();
    __syncthreads();

    /*
     * Send each remote PE this PE's contribution to that remote PE's output
     * block. The payload and ready flag use the same RC QP, so the receiver
     * cannot observe the flag before the payload is remotely visible.
     */
    for (int pe = PE_start; pe < finish; pe += stride) {
      if (pe != my_pe) {
        int remote_rank = (pe - PE_start) / stride;
        int qp_index = get_qp_index(pe, wf_info);
        const T *contribution = source + remote_rank * nreduce + offset;
        if (dest == source && remote_rank == 0) {
          contribution = pWrk + team_rank * chunk_size;
        }
        internal_putmem_nbi_wg(pWrk + team_rank * chunk_size, contribution, count * sizeof(T), pe, qp_index, wf_info);
        if (is_thread_zero_in_block()) {
          pSync[team_rank] = flag_val;
          __threadfence_system();
          internal_putmem(pSync + team_rank, pSync + team_rank, sizeof(*pSync), pe, qp_index, wf_info);
          pSync[team_rank] = ROCSHMEM_SYNC_VALUE;
        }
      }
    }
    threadfence_system();
    __syncthreads();

    // Wait for every remote contribution, then reduce it into dest.
    for (int pe = PE_start; pe < finish; pe += stride) {
      if (pe != my_pe) {
        int remote_rank = (pe - PE_start) / stride;
        if (is_thread_zero_in_block()) {
          wait_until(&pSync[remote_rank], ROCSHMEM_CMP_EQ, flag_val);
        }
        __syncthreads();
        for (int j = wg_id; j < count; j += wg_size) {
          T value = uncached_load(pWrk + remote_rank * chunk_size + j);
          OpWrap<Op>::Calc(&value, dest + offset + j, 0);
        }
        __syncthreads();
        threadfence_system();
      }
    }
    __syncthreads();

    // Reset synchronization slots before the next chunk reuses them.
    for (int j = wg_id; j < PE_size; j += wg_size) {
      pSync[j] = ROCSHMEM_SYNC_VALUE;
    }
    threadfence_system();
    __syncthreads();

    // Ensure all PEs finish this chunk before pWrk and pSync are reused.
    barrier_wg(team);
  }

  return ROCSHMEM_SUCCESS;
}

/*
 * Hierarchical reduce-scatter for teams spanning multiple IPC domains.
 * Each local slot first reduces matching source blocks from its IPC peers
 * into pWrk, then matching slots exchange and reduce those partial blocks
 * through a ring connecting IPC domains. Each ring transfer uses one QP and at most
 * 64 KiB of payload.
 *
 * The caller must provide a contiguous team composed of equally sized IPC
 * IPC domains. Concurrent workgroups must use distinct team-owned pWrk/pSync and
 * non-overlapping output ranges. Source blocks retain their full nreduce
 * stride; each workgroup processes only its assigned element range.
 * All threads in the workgroup must participate.
 */
template <typename T, ROCSHMEM_OP Op>
__device__ int GDAContext::hybrid_reduce_scatter_wg_nopxn(
    GDATeam *team_obj, T *dest, const T *source, int nreduce,
    int elem_offset, int elem_count, int PE_start, int stride, int PE_size,
    int team_rank) {

  const int local_size = constmem.ipc_shm_size;
  const int local_slot = ipcImpl_.shm_rank;
  const int node_id = team_rank / local_size;
  const int num_nodes = PE_size / local_size;

  T *p_wrk = reinterpret_cast<T *>(team_obj->pWrk);
  long *p_sync = team_obj->reduce_pSync;
  ActiveWFInfo wf_info(ctx_id_, ThreadScope::wg);
  const int tid = get_flat_block_id();
  const int wg_size = get_flat_block_size();
  const int scratch_elems = static_cast<int>(ROCSHMEM_REDUCE_MIN_WRKDATA_SIZE * sizeof(double) / sizeof(T));
  const int scratch_chunk_size = max(1, scratch_elems / num_nodes);
  const int chunk_size = collective_chunk_elems<T>(scratch_chunk_size,
                                                  kReduceScatterChunks);
  rocshmem_team_t team = reinterpret_cast<rocshmem_team_t>(team_obj);

  const int left_node = (node_id - 1 + num_nodes) % num_nodes;
  const int right_node = (node_id + 1) % num_nodes;
  const int right_pe = PE_start + (right_node * local_size + local_slot) * stride;

  // Iterate relative to the assigned range, keeping source PE-block strides
  // and destination offsets in the original collective's coordinates.
  for (int processed = 0; processed < elem_count;) {
    const int offset = elem_offset + processed;
    const int count = min(chunk_size, elem_count - processed);

    // Stage 1: slot s reduces, through IPC, one output block per node.  The
    // num_nodes partial blocks form this slot's input to the node-level ring.
    for (int target_node = 0; target_node < num_nodes; ++target_node) {
      const int target_rank = target_node * local_size + local_slot;
      T *partial = p_wrk + target_node * chunk_size;
      const int source_offset = target_rank * nreduce + offset;
      internal_fused_ipc_reduce<T, Op>(partial, source, source_offset, count, local_size, local_slot, tid, wg_size);
    }

    // Stage 2: run the inter-node ring between matching local slots.
    for (int step = 0; step < num_nodes - 1; ++step) {
      const int send_idx = (node_id - step - 1 + num_nodes) % num_nodes;
      const int recv_idx = (node_id - step - 2 + num_nodes) % num_nodes;
      const size_t send_bytes = static_cast<size_t>(count) * sizeof(T);
      const int signal_qp = get_qp_index(right_pe, wf_info);

      internal_putmem_nbi_wg(dest + offset,
                             p_wrk + send_idx * chunk_size, send_bytes,
                             right_pe, signal_qp, wf_info);
      __syncthreads();
      if (is_thread_zero_in_block()) {
        // The payload and ready signal use the same QP.
        // Use registered team storage for the ready signal.
        p_sync[node_id] = 1;
        __threadfence_system();
        internal_putmem(p_sync + node_id, p_sync + node_id, sizeof(*p_sync), right_pe, signal_qp, wf_info);
        p_sync[node_id] = ROCSHMEM_SYNC_VALUE;
        wait_until(p_sync + left_node, ROCSHMEM_CMP_EQ, 1L);
      }
      __syncthreads();

      // Accumulate the received block before forwarding it.
      gda_compute_reduce<T, Op>(dest + offset, p_wrk + recv_idx * chunk_size, count, tid, wg_size);

      // The last ring step produces this node's final block; publish it before the existing team barrier.
      if (step == num_nodes - 2) {
        for (int i = tid; i < count; i += wg_size) {
          dest[offset + i] = p_wrk[node_id * chunk_size + i];
        }
      }
      if (is_thread_zero_in_block()) {
        p_sync[left_node] = ROCSHMEM_SYNC_VALUE;
      }
      threadfence_system();
      __syncthreads();
      barrier_wg(team);
    }
    processed += count;
  }

  return ROCSHMEM_SUCCESS;
}

}  // namespace rocshmem

#endif  // USE_DAS
