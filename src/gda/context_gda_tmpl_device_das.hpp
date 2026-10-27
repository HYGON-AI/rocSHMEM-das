// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

/******************************************************************************
 * 1. 较大的DAS自研改动在本文件中维护，尽量保留原文件中的社区代码，
 *    方便后续维护及同步上游更新。
 * 2. 修改已有函数时，在此实现对应的_das函数；启用USE_DAS后(默认开启)，原函数
 *    在入口调用_das函数并返回，未启用时执行原来的社区实现。
 * 3. 新增辅助函数也在此维护，公共逻辑尽量复用原文件中的函数；
 *****************************************************************************/

#ifndef LIBRARY_SRC_GDA_CONTEXT_TMPL_DEVICE_DAS_HPP_
#define LIBRARY_SRC_GDA_CONTEXT_TMPL_DEVICE_DAS_HPP_

#include "device_arch.hpp"

#include "rocshmem/rocshmem_config.h"

#if defined(USE_DAS)
namespace rocshmem {

// Convert a byte policy to an element count, bounded by the caller's range or
// scratch capacity. Budgeted policies preserve single-WG and intra-node paths.
template <typename T>
__device__ __forceinline__ int GDAContext::collective_chunk_elems(
    int capacity, CollectiveChunkPolicy policy, bool inter_node) {
  size_t chunk_bytes = policy.max_bytes;
  if (policy.concurrent_bytes != 0) {
    const size_t wg_count = static_cast<size_t>(gridDim.x) * gridDim.y * gridDim.z;
    if (wg_count <= 1 || !inter_node) return capacity;
    chunk_bytes = min(chunk_bytes, policy.concurrent_bytes / wg_count);
  }
  return min(capacity, static_cast<int>(max(size_t{1}, chunk_bytes / sizeof(T))));
}

__device__ __forceinline__ uint32_t GDAContext::get_team_qp_index(
    const GDATeam *team_obj, int pe) {
  const uint32_t qp_slot =
      static_cast<uint32_t>(team_obj->pool_index_) % num_qps_per_pe;
  return qp_slot * constmem.num_pes + pe;
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
__device__ void GDAContext::internal_fused_ipc_reduce(
    T *dst, const T *src, int source_offset, int count, int local_size,
    int local_slot, int wg_id, int wg_size) {
  using VecT = std::conditional_t<std::is_same_v<T, float>, float4,
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
 * Hierarchical allreduce for teams spanning multiple IPC islands.
 * Local proxies first reduce disjoint segments across their IPC peers. The
 * first node then combines matching segments from every node, remote proxies
 * fetch the final segments, and each PE assembles the complete result locally.
 *
 * The team must contain contiguous, equally sized IPC islands. Concurrent
 * workgroups must use distinct team-owned pWrk/pSync and input/output buffers.
 */
template <typename T, ROCSHMEM_OP Op>
__device__ void GDAContext::internal_proxy_allreduce(T *dst, const T *src, int nelems, GDATeam *team_obj,
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
__device__ void GDAContext::internal_ring_allreduce_das(T *dst, const T *src,
    int nelems, GDATeam *team_obj,  // NOLINT(runtime/int)
    int n_seg, int seg_size, int chunk_size, ActiveWFInfo &wf_info) {

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
        reinterpret_cast<void *>(&dst[off_send + off_seg]),
        chunk_size * sizeof(T), send_pe, qp_index, wf_info);

      if (is_thread_zero_in_block()) {
        wait_val = sequence_base + seg + 1;
        internal_putmem(&pSync[iter], &wait_val, sizeof(*pSync), send_pe, qp_index, wf_info);
#if ROCSHMEM_ARCH_HYGON
        __threadfence_system();
#endif  // ROCSHMEM_ARCH_HYGON
        wait_until(&pSync[iter], ROCSHMEM_CMP_EQ, wait_val);
      }
      __syncthreads();
      gda_compute_reduce<T, Op>(&pWrk[off_recv], &dst[off_seg + off_recv],
                                chunk_size, wg_id, wg_size);
    }

    // Loop 2 in the example above
    for (int iter = PE_size - 1; iter < 2 * PE_size - 2; iter++) {
      off_send = (((my_pe_in_team + 1 - iter + 2 * PE_size) % PE_size) * chunk_size);
      int qp_index = get_qp_index(send_pe, wf_info);
      internal_putmem_nbi_wg(reinterpret_cast<void *>(&dst[off_send + off_seg]),
        reinterpret_cast<void *>(&dst[off_send + off_seg]),
        chunk_size * sizeof(T), send_pe, qp_index, wf_info);

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
__device__ int GDAContext::reduce_das(rocshmem_team_t team, T *dest,
                                  const T *source, int nreduce) {
  GDATeam *team_obj = reinterpret_cast<GDATeam *>(team);

  int PE_size = team_obj->tinfo_wrt_world->size;

  size_t direct_pWrk = PE_size * nreduce;
  size_t direct_pSync = PE_size;
  size_t ring_pSync = 2 * PE_size;
  size_t provided_pWrk = ROCSHMEM_REDUCE_MIN_WRKDATA_SIZE;
  size_t provided_pSync = ROCSHMEM_REDUCE_SYNC_SIZE;

  ActiveWFInfo wf_info(ctx_id_, ThreadScope::wg);

  constexpr size_t PROXY_ALLREDUCE_MIN_BYTES = 256 * 1024;
  const int local_size = constmem.ipc_shm_size;
  const int pe_start = team_obj->tinfo_wrt_world->pe_start;
  const int stride = team_obj->tinfo_wrt_world->stride;
  const bool use_proxy_allreduce = static_cast<size_t>(nreduce) * sizeof(T) >= PROXY_ALLREDUCE_MIN_BYTES &&
      stride == 1 && local_size > 1 && constmem.ipc_stride == 1 &&
      pe_start % local_size == 0 && PE_size % local_size == 0 &&
      PE_size > local_size;

  if (use_proxy_allreduce) {
    internal_proxy_allreduce<T, Op>(dest, source, nreduce, team_obj, wf_info);
    barrier_wg(team);
    return ROCSHMEM_SUCCESS;
  }

  // Messages above DIRECT_MAX use ring. default 8192 (8KB).
  constexpr int DIRECT_MAX_NELEMS = 8192;

  bool use_direct = (provided_pWrk >= direct_pWrk) &&
                    (provided_pSync >= direct_pSync) &&
                    (nreduce <= DIRECT_MAX_NELEMS);
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
        internal_ring_allreduce_das<T, Op>(dest, source, nreduce, team_obj, n_seg,
          seg_size, chunk_size, wf_info);
      }
      if (n_seg_up > n_seg) {
        T *p_dst = (dest + (n_seg * seg_size));
        const T *p_src = (source + (n_seg * seg_size));
        int p_count = nreduce - (n_seg * seg_size);
        int p_chunk = p_count / PE_size;

        if (p_chunk > 0) {
          internal_ring_allreduce_das<T, Op>(p_dst, p_src, (p_chunk * PE_size),
            team_obj, 1, (p_chunk * PE_size), p_chunk, wf_info);
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
__device__ int GDAContext::reduce_scatter_wg_das(rocshmem_team_t team, T *dest,
                                             const T *source, int nreduce,
                                             int elem_offset, int elem_count) {
  GDATeam *team_obj = reinterpret_cast<GDATeam *>(team);
  int PE_size = team_obj->tinfo_wrt_world->size;
  int PE_start = team_obj->tinfo_wrt_world->pe_start;
  int stride = team_obj->tinfo_wrt_world->stride;
  int my_pe = constmem.my_pe;
  int team_rank = (my_pe - PE_start) / stride;

  const int local_size = constmem.ipc_shm_size;
  int local_target_pe{-1};
  const bool has_ipc = ipcImpl_.isIpcAvailable(my_pe, my_pe, &local_target_pe);
  if (elem_count < 0) elem_count = nreduce;
  const bool use_hybrid = has_ipc && local_size > 1 && stride == 1 &&
      constmem.ipc_stride == 1 && PE_start % local_size == 0 &&
      PE_size > local_size && PE_size % local_size == 0 && nreduce > 1024;

  if (use_hybrid) {
    return hybrid_reduce_scatter_wg<T, Op>(team_obj, dest, source, nreduce,
                                           elem_offset, elem_count,
                                           PE_start, stride, PE_size, team_rank);
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

    // Seed this PE's output block with its local contribution.
    for (int j = wg_id; j < count; j += wg_size) {
      dest[offset + j] = source[team_rank * nreduce + offset + j];
    }
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
        internal_putmem_nbi_wg(&pWrk[team_rank * chunk_size],
                           source + remote_rank * nreduce + offset,
                           count * sizeof(T), pe, qp_index, wf_info);
        if (is_thread_zero_in_block()) {
          internal_putmem(&pSync[team_rank], &flag_val, sizeof(*pSync),
                          pe, qp_index, wf_info);
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
        gda_compute_reduce<T, Op>(&pWrk[remote_rank * chunk_size],
                                  dest + offset, count, wg_id, wg_size);
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
 * Hierarchical reduce-scatter for teams spanning multiple IPC islands.
 * Each local slot first reduces matching source blocks from its IPC peers
 * into pWrk, then matching slots exchange and reduce those partial blocks
 * through an inter-node ring. Each ring transfer uses one QP and at most
 * 64 KiB of payload.
 *
 * The caller must provide a contiguous team composed of equally sized IPC
 * islands. Concurrent workgroups must use distinct team-owned pWrk/pSync and
 * non-overlapping output ranges. Source blocks retain their full nreduce
 * stride; each workgroup processes only its assigned element range.
 * All threads in the workgroup must participate.
 */
template <typename T, ROCSHMEM_OP Op>
__device__ int GDAContext::hybrid_reduce_scatter_wg(
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
        const int64_t ready = 1;
        internal_putmem(p_sync + node_id, &ready, sizeof(*p_sync), right_pe, signal_qp, wf_info);
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

/**
 * @brief Select the next same-QP lane group for wave-safe submission.
 * @note Default-context QPs may be shared across workgroups. Processing one
 * QP group at a time avoids a wave blocking on multiple SQ locks.
 */
__device__ __forceinline__ uint64_t GDAContext::get_next_qp_group(
    int qp_index, uint64_t pending_lanes) {
  // Use the first pending lane to select the next QP.
  const int first_lane = __builtin_ctzll(pending_lanes);
  const int selected_qp = __shfl(qp_index, first_lane);
  const int lane_id = get_flat_block_id() % WF_SIZE;
  // Collect all pending lanes mapped to the selected QP.
  return __ballot((pending_lanes & (uint64_t{1} << lane_id)) &&
                  qp_index == selected_qp);
}

/** @brief Submit a PUT safely through a QP shared across workgroups. */
__device__ __forceinline__ void GDAContext::put_nbi_shared_qp(
    int qp_index, void *dest, const void *source, size_t length, bool ring_db) {
  if (ctx_id_ != 0) {
    qps[qp_index].put_nbi_single(dest, source, length, ring_db);
    return;
  }

  uint64_t pending_lanes = get_active_lane_mask();
  const int lane_id = get_flat_block_id() % WF_SIZE;
  // Submit each QP group through the existing wave-batched path.
  while (pending_lanes) {
    const uint64_t qp_group = get_next_qp_group(qp_index, pending_lanes);
    if (qp_group & (uint64_t{1} << lane_id)) {
      ActiveWFInfo wf_info(qp_index);
      qps[qp_index].put_nbi(dest, source, length, wf_info, ring_db);
    }
    // Continue with the remaining QP groups.
    pending_lanes &= ~qp_group;
  }
}

/** @brief Submit a GET safely through a QP shared across workgroups. */
__device__ __forceinline__ void GDAContext::get_nbi_shared_qp(
    int qp_index, void *dest, const void *source, size_t length) {
  if (ctx_id_ != 0) {
    qps[qp_index].get_nbi_single(dest, source, length, true);
    return;
  }

  uint64_t pending_lanes = get_active_lane_mask();
  const int lane_id = get_flat_block_id() % WF_SIZE;
  // Submit each QP group through the existing wave-batched path.
  while (pending_lanes) {
    const uint64_t qp_group = get_next_qp_group(qp_index, pending_lanes);
    if (qp_group & (uint64_t{1} << lane_id)) {
      ActiveWFInfo wf_info(qp_index);
      qps[qp_index].get_nbi(dest, source, length, wf_info);
    }
    pending_lanes &= ~qp_group;
  }
}

/** @brief Submit a non-fetching atomic safely through a shared QP. */
__device__ __forceinline__ void GDAContext::atomic_nofetch_shared_qp(
    int qp_index, void *dest, int64_t value) {
  if (ctx_id_ != 0) {
    qps[qp_index].atomic_add_single(dest, value);
    return;
  }

  uint64_t pending_lanes = get_active_lane_mask();
  const int lane_id = get_flat_block_id() % WF_SIZE;
  // Submit each QP group through the existing wave-batched path.
  while (pending_lanes) {
    const uint64_t qp_group = get_next_qp_group(qp_index, pending_lanes);
    if (qp_group & (uint64_t{1} << lane_id)) {
      ActiveWFInfo wf_info(qp_index);
      qps[qp_index].atomic_add(dest, value, 0, wf_info);
    }
    pending_lanes &= ~qp_group;
  }
}

__device__ __forceinline__ bool gda_copy_if_self(int pe, void* dst,
                                                 const void* src,
                                                 size_t bytes) {
  if (pe != constmem.my_pe) return false;
  if (bytes != 0) {
    memcpy_lane<MemcpyKind::Put>(dst, const_cast<void*>(src), bytes);
  }
  return true;
}

template <typename T>
__device__ void GDAContext::alltoall_linear_wg_das(rocshmem_team_t team, T *dst,
                                            const T *src, int nelems,
                                            int elem_offset, int elem_count) {
  GDATeam *team_obj = reinterpret_cast<GDATeam *>(team);

  int pe_start = team_obj->tinfo_wrt_world->pe_start;
  int pe_size = team_obj->num_pes;
  int stride = team_obj->tinfo_wrt_world->stride;
  long *pSync = team_obj->alltoall_pSync;
  int my_pe_in_team = team_obj->my_pe;

  // Normalize: elem_count == -1 means "process the full [0, nelems) range".
  if (elem_count < 0) elem_count = nelems;

  int wf_id = get_flat_block_id() / WF_SIZE;
  int wf_count = (int) ceil((double)get_flat_block_size() / (double)WF_SIZE);

  ActiveWFInfo wf_info(ctx_id_, ThreadScope::wg);
  // Empty slices still participate in the team synchronization below, but
  // must not submit a zero-byte RDMA operation. alltoallv_on_stream may launch
  // more workgroups than elements because its counts reside in device memory.
  if (elem_count > 0) {
    const int chunk_elems = collective_chunk_elems<T>(
        elem_count, kAlltoallChunks, pe_size > constmem.ipc_shm_size);
    for (int processed = 0; processed < elem_count;) {
      const int count = min(chunk_elems, elem_count - processed);
      const int offset = elem_offset + processed;
      // Have each PE put their designated data to the other PEs.
      for (int j = wf_id; j < pe_size; j += wf_count) {
        int dest_pe = team_obj->get_pe_in_world(j);
        internal_putmem_nbi_wave(&dst[my_pe_in_team * nelems + offset],
          &src[j * nelems + offset],
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
__device__ void GDAContext::alltoall_linear_thread_puts_das(rocshmem_team_t team,
    T *dst, const T *src, int nelems, int elem_offset, int elem_count) {
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

  int* qp_indices = reinterpret_cast<int*>(team_obj->pWrk);

  // Have each PE put their designated data to the other PEs
  for (int j = tid; j < pe_size; j += step_size) {
    int dest_pe = team_obj->get_pe_in_world(j);
    ActiveWFInfo wf_info(dest_pe);
    int qp_index = get_qp_index(dest_pe, wf_info);
    qp_indices[j] = qp_index;
    uint64_t base_heap_offset = base_heap[dest_pe] - base_heap[constmem.my_pe];
    if (gda_copy_if_self(
          dest_pe, &dst[my_pe_in_team * nelems + elem_offset],
          &src[j * nelems + elem_offset], elem_count * sizeof(T))) {
      pSync[alltoall_pSync_offset + my_pe_in_team] = completion_count;
      continue;
    }

    put_nbi_shared_qp(qp_index,
      reinterpret_cast<char*>(&dst[my_pe_in_team * nelems + elem_offset]) + base_heap_offset,
      &src[j * nelems + elem_offset], elem_count * sizeof(T), false);
    atomic_nofetch_shared_qp(qp_index,
      reinterpret_cast<char *>(&pSync[alltoall_pSync_offset + my_pe_in_team]) +
      base_heap_offset, 1);
  }

  // wait until everyone has obtained their designated data
  for (int j = tid; j < pe_size; j+= step_size) {
    int dest_pe = team_obj->get_pe_in_world(j);

    long *sync_flags = &pSync[alltoall_pSync_offset + dest_pe];
    while (uncached_load(sync_flags) < completion_count) { }

    qps[qp_indices[j]].quiet_single();

  }

  if (is_thread_zero_in_block()) {
    team_obj->alltoall_sequence_number++;
  }

  __syncthreads();
}

template <typename T>
__device__ void GDAContext::alltoall_wg_das(rocshmem_team_t team, T *dst,
                                     const T *src, int nelems,
                                     int elem_offset, int elem_count) {
  alltoall_linear_wg_das(team, dst, src, nelems, elem_offset, elem_count);
}

template <typename T>
__device__ void GDAContext::alltoallv_das(rocshmem_team_t team,
                                      T *dest, const size_t dest_nelems[],
                                      const size_t dest_displs[],
                                      T *source, const size_t source_nelems[],
                                      const size_t source_displs[],
                                      int wg_id, int wg_count) {
  GDATeam *team_obj = reinterpret_cast<GDATeam *>(team);
  const int pe_size = team_obj->num_pes;
  const size_t nelems = source_nelems[0];
  bool regular_layout = nelems <= 0x7fffffff;

  for (int pe = 0; pe < pe_size && regular_layout; pe++) {
    regular_layout = source_nelems[pe] == nelems &&
                     dest_nelems[pe] == nelems &&
                     source_displs[pe] == static_cast<size_t>(pe) * nelems &&
                     dest_displs[pe] == static_cast<size_t>(pe) * nelems;
  }

  if (regular_layout) {
    size_t elem_offset = 0;
    size_t elem_count = 0;
    split_wg_range(nelems, wg_id, wg_count, &elem_offset, &elem_count);
    alltoall_wg_das(team, dest, source, static_cast<int>(nelems),
             static_cast<int>(elem_offset), static_cast<int>(elem_count));
    return;
  }

  if (constmem.alltoall_wg_algo == gda::ALLTOALLV_WG_ALGO_COPY) {
    alltoallv_copy_das(team, dest, dest_nelems, dest_displs, source, source_nelems,
                   source_displs, wg_id, wg_count);
  } else {
    alltoallv_get_das(team,
                  dest, dest_nelems, dest_displs,
                  source, source_nelems, source_displs,
                  wg_id, wg_count);
  }
}

template <typename T>
__device__ void GDAContext::alltoallv_copy_das(rocshmem_team_t team, T *dest,
    const size_t dest_nelems[], const size_t dest_displs[], T *source,
    const size_t source_nelems[], const size_t source_displs[],
    int wg_id, int wg_count) {
  GDATeam *team_obj = reinterpret_cast<GDATeam *>(team);
  int pe_size = team_obj->num_pes;
  long *pSync = team_obj->alltoall_pSync;
  int my_pe_in_team = team_obj->my_pe;
  uint64_t a2a_sn = team_obj->alltoall_sequence_number;
  uint64_t alltoall_pSync_offset = (a2a_sn % 2) * pe_size;
  long completion_count = a2a_sn / 2 + 1;
  T *tmp_buf = reinterpret_cast<T*>(team_obj->pWrk);
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
    uint64_t base_heap_offset = base_heap[dest_pe] - base_heap[constmem.my_pe];
    const size_t total = source_nelems[j];
    size_t elem_offset = 0;
    size_t elem_count = 0;
    split_wg_range(total, wg_id, wg_count, &elem_offset, &elem_count);
    size_t nelems = elem_count * sizeof(T);
    char* amo_dst = ((char*)&pSync[alltoall_pSync_offset + my_pe_in_team] + base_heap_offset);

    if (nelems != 0) {
      T* src = source + source_displs[j] + elem_offset;
      T* dst = (T*)((char*)&tmp_buf[my_pe_in_team * tmp_buf_off] + base_heap_offset);
      put_nbi_shared_qp(qp_index, dst, src, nelems, false);
    }

    atomic_nofetch_shared_qp(qp_index, amo_dst, 1);
  }

  // wait until everyone has obtained their designated data
  for (int j = tid; j < pe_size; j += step_size) {
    int dest_pe = team_obj->get_pe_in_world(j);
    int qp_index = get_team_qp_index(team_obj, dest_pe);

    long *sync_flags = &pSync[alltoall_pSync_offset + j];
    while (uncached_load(sync_flags) < completion_count) { }

    qps[qp_index].quiet_single();

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
        T* dst = dest + dest_displs[j] + elem_offset;
        T* src = (T*)((char*) &tmp_buf[j * tmp_buf_off]);
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
__device__ void GDAContext::alltoallv_get_das(rocshmem_team_t team, T *dest,
    const size_t dest_nelems[], const size_t dest_displs[], T *source,
    const size_t source_nelems[], const size_t source_displs[],
    int wg_id, int wg_count) {
  GDATeam *team_obj = reinterpret_cast<GDATeam *>(team);
  int pe_size = team_obj->num_pes;
  long *pSync = team_obj->alltoall_pSync;
  int my_pe_in_team = team_obj->my_pe;
  uint64_t a2a_sn   = team_obj->alltoall_sequence_number;
  uint64_t alltoall_pSync_offset = (a2a_sn % 2) * pe_size;
  long completion_count = a2a_sn / 2 + 1;
  uint64_t *tmp_buf = (uint64_t*)team_obj->pWrk;

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
    uint64_t base_heap_offset = base_heap[dest_pe] - base_heap[constmem.my_pe];

    /* Pack Ctrl Message * 16 bits seq | 48bit displ */
    seq_bits = (seq_mask & (a2a_sn + 1)) << seq_shift;
    const size_t send_total = source_nelems[j];
    size_t send_offset = 0;
    [[maybe_unused]] size_t send_count = 0;
    split_wg_range(send_total, wg_id, wg_count, &send_offset, &send_count);
    displ_bits = (displs_mask & (source_displs[j] + send_offset));
    uint64_t ctrl_msg = seq_bits | displ_bits;

    /* Prepare Ctrl Message */
    src = (uint64_t*)&ctrl_msg;
    dst = (uint64_t*)((char*)&tmp_buf[constmem.my_pe] + base_heap_offset);

    put_nbi_shared_qp(qp_index, dst, src, sizeof(uint64_t), true);
    qps[qp_index].quiet_single();
  }

  __syncthreads();

  /* Phase 2: consume every control message and issue the corresponding GET. */
  for (int j = tid; j < pe_size; j+= step_size) {
    int dest_pe = team_obj->get_pe_in_world(j);
    int qp_index = get_team_qp_index(team_obj, dest_pe);
    uint64_t base_heap_offset = base_heap[dest_pe] - base_heap[constmem.my_pe];
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
    const void *src = reinterpret_cast<const char *>(source) +
                      (displ_bits * sizeof(T)) + base_heap_offset;
    void *dst = dest + dest_displs[j] + recv_offset;

    // Most workgroups have an empty slice for small variable-count messages.
    // They still signal completion, but must not submit a zero-byte GET.
    if (nelems != 0) {
      get_nbi_shared_qp(qp_index, dst, src, nelems);
    }

    /* Put Completion */
    char* amo_dst = ((char*)&pSync[alltoall_pSync_offset + my_pe_in_team] + base_heap_offset);
    atomic_nofetch_shared_qp(qp_index, amo_dst, 1);
  }

  __syncthreads();

  /* Phase 3: wait for all peers and drain the QPs used by this team. */
  for (int j = tid; j < pe_size; j+= step_size) {
    int dest_pe = team_obj->get_pe_in_world(j);
    int qp_index = get_team_qp_index(team_obj, dest_pe);
    long *sync_flags = &pSync[alltoall_pSync_offset + j];
    while (uncached_load(sync_flags) < completion_count) { }

    qps[qp_index].quiet_single();
  }

  if (is_thread_zero_in_block()) {
    team_obj->alltoall_sequence_number++;
  }

  __syncthreads();
}

template <typename T>
__device__ void GDAContext::fcollect_linear_das(rocshmem_team_t team, T *dst,
                                            const T *src, int nelems,
                                            int elem_offset, int elem_count) {
  GDATeam *team_obj = reinterpret_cast<GDATeam *>(team);

  int pe_start = team_obj->tinfo_wrt_world->pe_start;
  int pe_size = team_obj->num_pes;
  int stride = team_obj->tinfo_wrt_world->stride;
  long *pSync = team_obj->alltoall_pSync;
  int my_pe_in_team = team_obj->my_pe;
  if (elem_count < 0) elem_count = nelems;

  ActiveWFInfo wf_info(ctx_id_, ThreadScope::wg);

  // Chunk the linear path below instead of selecting the synchronization-heavy
  // hierarchical path solely because more workgroups are active.
  if (pe_size >= 4 &&
      static_cast<size_t>(elem_count) * sizeof(T) >= kHierarchicalMinBytes) {
    // FCollect is a concatenation of one broadcast per source rank.  Reusing
    // the hierarchical broadcast path makes each source cross a node boundary
    // once, followed by IPC fan-out inside the destination node, instead of
    // sending the same source segment independently to every remote PE.
    for (int rank = 0; rank < pe_size; rank++) {
      const int root = team_obj->get_pe_in_world(rank);
      internal_broadcastmem_wg(dst + rank * nelems + elem_offset, src + elem_offset,
                         elem_count * sizeof(T), root, pe_start, stride, pe_size, pSync);
    }
    return;
  }

  // Bound the cross-node burst across WGs without introducing one team-wide
  // broadcast per source. Single-WG and intra-node transfers retain their size.
  const int chunk_elems = collective_chunk_elems<T>(
      elem_count, kFcollectChunks, pe_size > constmem.ipc_shm_size);
  for (int processed = 0; processed < elem_count;) {
    const int count = min(chunk_elems, elem_count - processed);
    const int offset = elem_offset + processed;
    for (int rank = 0; rank < pe_size; rank++) {
      int dest_pe = team_obj->get_pe_in_world(rank);
      internal_putmem_nbi_wg(&dst[my_pe_in_team * nelems + offset],
                             src + offset, count * sizeof(T),
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

}
#endif  // defined(USE_DAS)
#endif  // LIBRARY_SRC_GDA_CONTEXT_TMPL_DEVICE_DAS_HPP_
