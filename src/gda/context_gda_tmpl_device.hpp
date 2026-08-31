/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 * Copyright (c) 2026 Hygon Information Technology Co., Ltd.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *****************************************************************************/

#ifndef LIBRARY_SRC_GDA_CONTEXT_TMPL_DEVICE_HPP_
#define LIBRARY_SRC_GDA_CONTEXT_TMPL_DEVICE_HPP_

#include "rocshmem/rocshmem_config.h"  // NOLINT(build/include_subdir)
#include "rocshmem/rocshmem.hpp"
#include "constmem.hpp"
#include "log.hpp"
#include "util.hpp"
#include "context_gda_device.hpp"
#include "gda_team.hpp"
#include "queue_pair.hpp"
#include "rocshmem_calc.hpp"
#include "backend_gda.hpp"

#include <hip/hip_runtime.h>

namespace rocshmem {

/******************************************************************************
 ************************** TEMPLATE SPECIALIZATIONS **************************
 *****************************************************************************/
template <typename T>
__device__ void GDAContext::p(T *dest, T value, int pe) {
  int local_pe{-1};
  if (ipcImpl_.isIpcAvailable(constmem.my_pe, pe, &local_pe)) {
    long L_offset{reinterpret_cast<char *>(dest) - ipcImpl_.ipc_bases[ipcImpl_.shm_rank]};
    ipcImpl_.ipcCopy<MemcpyKind::Put>(ipcImpl_.ipc_bases[local_pe] + L_offset, reinterpret_cast<void *>(&value), sizeof(T), local_pe);
    return;
  }
  putmem_nbi(dest, &value, sizeof(T), pe);
}

template <typename T>
__device__ void GDAContext::put(T *dest, const T *source, size_t nelems, int pe) {
  putmem(dest, source, nelems * sizeof(T), pe);
}

template <typename T>
__device__ void GDAContext::put_nbi(T *dest, const T *source, size_t nelems, int pe) {
  putmem_nbi(dest, source, sizeof(T) * nelems, pe);
}

template <typename T>
__device__ T GDAContext::g(const T *source, int pe) {
  T ret{};
  int local_pe{-1};
  if (ipcImpl_.isIpcAvailable(constmem.my_pe, pe, &local_pe)) {
    const char *src_typed{reinterpret_cast<const char *>(source)};
    long L_offset{const_cast<char *>(src_typed) - ipcImpl_.ipc_bases[ipcImpl_.shm_rank]};
    ipcImpl_.ipcCopy<MemcpyKind::Get>(&ret, ipcImpl_.ipc_bases[local_pe] + L_offset, sizeof(T), local_pe);
    return ret;
  }
  LOGD_ERROR_ABORT("gda::g not implemented");
  //TODO the following is incorrect because ret is not ibv registered memory
  //getmem(&ret, source, sizeof(T), pe);
  return ret;
}

template <typename T>
__device__ void GDAContext::get(T *dest, const T *source, size_t nelems, int pe) {
  getmem(dest, source, sizeof(T) * nelems, pe);
}

template <typename T>
__device__ void GDAContext::get_nbi(T *dest, const T *source, size_t nelems, int pe) {
  getmem_nbi(dest, source, sizeof(T) * nelems, pe);
}

// Atomics
template <typename T>
__device__ void GDAContext::amo_add(void *dst, T value, int pe) {
  if constexpr (sizeof(T) != 8) { LOGD_ERROR_ABORT("gda::amo_add not implemented for non-64bit types"); }//TODO:support for non-uint64t
  int local_pe{-1};
  if (ipcImpl_.isIpcAvailable(constmem.my_pe, pe, &local_pe)) {
    ipcImpl_.ipcAMOAdd(reinterpret_cast<T *>(get_local_ptr(dst, local_pe)), value);
    return;
  }
  
  ActiveWFInfo wf_info(pe);
  int qp_index = get_qp_index(pe, wf_info);
  bool need_turn {true};
  uint64_t turns = __ballot(need_turn);
  while (turns) {
    uint8_t lane = __ffsll((unsigned long long)turns) - 1;
    int pe_turn = __shfl(pe, lane);
    if (pe_turn == pe) {
      qps[qp_index].atomic_nofetch(get_remote_ptr(dst, pe), value, 0, wf_info);
      need_turn = false;
    }
    turns = __ballot(need_turn);
  }
}

template <typename T>
__device__ void GDAContext::amo_add_dp(void *dst, T value, int qp_idx, int pe) {
  if constexpr (sizeof(T) != 8) { printf("rocshmem::gda:amo_add_dp not implemented for non-64bit types.\n"); abort(); }//TODO:support for non-uint64t
  // int local_pe{-1};
  // if (ipcImpl_.isIpcAvailable(constmem.my_pe, pe, &local_pe)) {
  //   ipcImpl_.ipcAMOAdd(reinterpret_cast<T *>(get_local_ptr(dst, local_pe)), value);
  //   return;
  // }

  bool need_turn {true};
  uint64_t turns = __ballot(need_turn);
  while (turns) {
    uint8_t lane = __ffsll((unsigned long long)turns) - 1;  // 返回 x 中最低位 1 的位置（从 1 开始计数）。
    int pe_turn = __shfl(pe, lane);  // 广播到整个 warp 
    if (pe_turn == pe) {
      qps[qp_idx].atomic_nofetch_dp(get_remote_ptr(dst, pe), value, 0, pe);
      need_turn = false;
    }
    turns = __ballot(need_turn);
  }
}

template <typename T>
__device__ void GDAContext::amo_set(void *dst, T value, int pe) {
  amo_swap(dst, value, pe);
}

template <typename T>
__device__ T GDAContext::amo_swap(void *dst, T value, int pe) {
  if constexpr (sizeof(T) != 8) { LOGD_ERROR_ABORT("gda::amo_swap not implemented for non-64bit types"); }//TODO:support for non-uint64t
  T ret_val;
  int local_pe{-1};
  if (ipcImpl_.isIpcAvailable(constmem.my_pe, pe, &local_pe)) {
    ret_val = ipcImpl_.ipcAMOSwap(reinterpret_cast<T *>(get_local_ptr(dst, local_pe)), value);
    return ret_val;
  }

  ActiveWFInfo wf_info(pe);
  int qp_index = get_qp_index(pe, wf_info);
  bool need_turn {true};
  uint64_t turns = __ballot(need_turn);
  T cond = 0;
  while (turns) {
    uint8_t lane = __ffsll((unsigned long long)turns) - 1;
    int pe_turn = __shfl(pe, lane);
    if (pe_turn == pe) {
      /**
       * Guess that the remote memory is zero by setting condition to zero.
       * The compare-and-swap loop will execute at least twice if wrong.
       * It may run additional times if contention on memory location.
       */
      while (wf_info.update(pe), (ret_val = qps[qp_index].atomic_cas(
             get_remote_ptr(dst, pe), value, cond, wf_info)) != cond) {
        cond = ret_val;
      }
      need_turn = false;
    }
    turns = __ballot(need_turn);
  }
  return ret_val;
}

template <typename T>
__device__ T GDAContext::amo_fetch_and(void *dst, T value, int pe) {
  if constexpr (sizeof(T) != 8) { LOGD_ERROR_ABORT("gda::amo_fetch_and not implemented for non-64bit types"); }//TODO:support for non-uint64t
  T ret_val;
  int local_pe{-1};
  if (ipcImpl_.isIpcAvailable(constmem.my_pe, pe, &local_pe)) {
    ret_val = ipcImpl_.ipcAMOFetchAnd(reinterpret_cast<T *>(get_local_ptr(dst, local_pe)), value);
    return ret_val;
  }
  
  ActiveWFInfo wf_info(pe);
  int qp_index = get_qp_index(pe, wf_info);
  bool need_turn {true};
  uint64_t turns = __ballot(need_turn);
  T cond = 0;
  T desired_val = cond & value;
  while (turns) {
    uint8_t lane = __ffsll((unsigned long long)turns) - 1;
    int pe_turn = __shfl(pe, lane);
    if (pe_turn == pe) {
      while (wf_info.update(pe), (ret_val = qps[qp_index].atomic_cas(
             get_remote_ptr(dst, pe), desired_val, cond, wf_info)) != cond) {
        cond = ret_val;
        desired_val = ret_val & value;
      }
      need_turn = false;
    }
    turns = __ballot(need_turn);
  }
  return ret_val;
}

template <typename T>
__device__ void GDAContext::amo_and(void *dst, T value, int pe) {
  amo_fetch_and(dst, value, pe);
}

template <typename T>
__device__ T GDAContext::amo_fetch_or(void *dst, T value, int pe) {
  if constexpr (sizeof(T) != 8) { LOGD_ERROR_ABORT("gda::amo_fetch_or not implemented for non-64bit types"); }//TODO:support for non-uint64t
  T ret_val;
  int local_pe{-1};
  if (ipcImpl_.isIpcAvailable(constmem.my_pe, pe, &local_pe)) {
    ret_val = ipcImpl_.ipcAMOFetchOr(reinterpret_cast<T *>(get_local_ptr(dst, local_pe)), value);
    return ret_val;
  }

  ActiveWFInfo wf_info(pe);
  int qp_index = get_qp_index(pe, wf_info);
  bool need_turn {true};
  uint64_t turns = __ballot(need_turn);
  T cond = 0;
  T desired_val = cond | value;
  while (turns) {
    uint8_t lane = __ffsll((unsigned long long)turns) - 1;
    int pe_turn = __shfl(pe, lane);
    if (pe_turn == pe) {
      while (wf_info.update(pe), (ret_val = qps[qp_index].atomic_cas(
             get_remote_ptr(dst, pe), desired_val, cond, wf_info)) != cond) {
        cond = ret_val;
        desired_val = ret_val | value;
      }
      need_turn = false;
    }
    turns = __ballot(need_turn);
  }
  return ret_val;
}

template <typename T>
__device__ void GDAContext::amo_or(void *dst, T value, int pe) {
  amo_fetch_or(dst, value, pe);
}

template <typename T>
__device__ T GDAContext::amo_fetch_xor(void *dst, T value, int pe) {
  if constexpr (sizeof(T) != 8) { LOGD_ERROR_ABORT("gda::amo_fetch_xor not implemented for non-64bit types"); }//TODO:support for non-uint64t
  T ret_val;
  int local_pe{-1};
  if (ipcImpl_.isIpcAvailable(constmem.my_pe, pe, &local_pe)) {
    ret_val = ipcImpl_.ipcAMOFetchXor(reinterpret_cast<T *>(get_local_ptr(dst, local_pe)), value);
    return ret_val;
  }
  
  ActiveWFInfo wf_info(pe);
  int qp_index = get_qp_index(pe, wf_info);
  bool need_turn {true};
  uint64_t turns = __ballot(need_turn);
  T cond = 0;
  T desired_val = cond ^ value;
  while (turns) {
    uint8_t lane = __ffsll((unsigned long long)turns) - 1;
    int pe_turn = __shfl(pe, lane);
    if (pe_turn == pe) {
      while (wf_info.update(pe), (ret_val = qps[qp_index].atomic_cas(
             get_remote_ptr(dst, pe), desired_val, cond, wf_info)) != cond) {
        cond = ret_val;
        desired_val = ret_val ^ value;
      }
      need_turn = false;
    }
    turns = __ballot(need_turn);
  }
  return ret_val;
}

template <typename T>
__device__ void GDAContext::amo_xor(void *dst, T value, int pe) {
  amo_fetch_xor(dst, value, pe);
}

template <typename T>
__device__ void GDAContext::amo_cas(void *dst, T value, T cond, int pe) {
  if constexpr (sizeof(T) != 8) { LOGD_ERROR_ABORT("gda::amo_cas not implemented for non-64bit types"); }//TODO:support for non-uint64t
  int local_pe{-1};
  if (ipcImpl_.isIpcAvailable(constmem.my_pe, pe, &local_pe)) {
    ipcImpl_.ipcAMOCas(reinterpret_cast<T *>(get_local_ptr(dst, local_pe)), value);
    return;
  }
  
  ActiveWFInfo wf_info(pe);
  int qp_index = get_qp_index(pe, wf_info);
  bool need_turn {true};
  uint64_t turns = __ballot(need_turn);
  while (turns) {
    uint8_t lane = __ffsll((unsigned long long)turns) - 1;
    int pe_turn = __shfl(pe, lane);
    if (pe_turn == pe) {
      qps[qp_index].atomic_cas_nofetch(get_remote_ptr(dst, pe), value, cond, wf_info);
      need_turn = false;
    }
    turns = __ballot(need_turn);
  }
}

template <typename T>
__device__ T GDAContext::amo_fetch_add(void *dst, T value, int pe) {
  if constexpr (sizeof(T) != 8) { LOGD_ERROR_ABORT("gda::amo_fetch_add not implemented for non-64bit types"); }//TODO:support for non-uint64t
  T ret_val = 0;
  int local_pe{-1};
  if (ipcImpl_.isIpcAvailable(constmem.my_pe, pe, &local_pe)) {
    ret_val = ipcImpl_.ipcAMOFetchAdd(reinterpret_cast<T *>(get_local_ptr(dst, local_pe)), value);
    return ret_val;
  }
  
  ActiveWFInfo wf_info(pe);
  int qp_index = get_qp_index(pe, wf_info);
  bool need_turn {true};
  uint64_t turns = __ballot(need_turn);
  while (turns) {
    uint8_t lane = __ffsll((unsigned long long)turns) - 1;
    int pe_turn = __shfl(pe, lane);
    if (pe_turn == pe) {
      ret_val =  qps[qp_index].atomic_fetch(get_remote_ptr(dst, pe), value, 0, wf_info);
      need_turn = false;
    }
    turns = __ballot(need_turn);
  }
  return ret_val;
}

template <typename T>
__device__ T GDAContext::amo_fetch_cas(void *dst, T value, T cond, int pe) {
  if constexpr (sizeof(T) != 8) { LOGD_ERROR_ABORT("gda::amo_fetch_cas not implemented for non-64bit types"); }//TODO:support for non-uint64t
  T ret_val;
  int local_pe{-1};
  if (ipcImpl_.isIpcAvailable(constmem.my_pe, pe, &local_pe)) {
    ret_val = ipcImpl_.ipcAMOFetchCas(reinterpret_cast<T *>(get_local_ptr(dst, local_pe)), cond, value);
    return ret_val;
  }
  
  ActiveWFInfo wf_info(pe);
  int qp_index = get_qp_index(pe, wf_info);
  bool need_turn {true};
  uint64_t turns = __ballot(need_turn);
  while (turns) {
    uint8_t lane = __ffsll((unsigned long long)turns) - 1;
    int pe_turn = __shfl(pe, lane);
    if (pe_turn == pe) {
      ret_val = qps[qp_index].atomic_cas(get_remote_ptr(dst, pe), value, cond, wf_info);
      need_turn = false;
    }
    turns = __ballot(need_turn);
  }
  return ret_val;
}

// Collectives TODO: loosely adapted from IPC, needs review
template <typename T, ROCSHMEM_OP Op>
__device__ void gda_compute_reduce(T *src, T *dst, int size, int wg_id, int wg_size) {
  for (int i = wg_id; i < size; i += wg_size) {
    OpWrap<Op>::Calc(src, dst, i);
  }
  __syncthreads();
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

template <typename T, ROCSHMEM_OP Op>
__device__ void GDAContext::internal_direct_allreduce(T *dst, const T *src,
    int nelems, GDATeam *team_obj, ActiveWFInfo &wf_info) {  // NOLINT(runtime/int)

  int stride = team_obj->tinfo_wrt_world->stride;
  int PE_start = team_obj->tinfo_wrt_world->pe_start;
  int PE_size = team_obj->tinfo_wrt_world->size;
  long *pSync = team_obj->reduce_pSync;
  T *pWrk = reinterpret_cast<T *>(team_obj->pWrk);

  int finish = PE_start + stride * PE_size;
  int pe = constmem.my_pe;

  int wg_id = get_flat_block_id();
  int wg_size = get_flat_block_size();
  int64_t flag_val = 1;

  for (int i = wg_id; i < nelems; i += wg_size) {
    dst[i] = src[i];
  }
  __syncthreads();

  for (int i = PE_start; i < finish; i += stride) {
    if (i != pe) {
      internal_putmem_wg(&pWrk[pe * nelems], reinterpret_cast<const void *>(src),
        nelems * sizeof(T), i, i, wf_info);

      if (is_thread_zero_in_block()) {
        fence();
        internal_putmem(&pSync[pe], &flag_val, sizeof(*pSync), i, i, wf_info);
      }
    }
  }
  threadfence_system();
  __syncthreads();

  // Do the compute and pSync reset in parallel.
  for (int i = PE_start; i < finish; i += stride) {
    if (i != pe) {
      // Wait for leader thread to see that the buffer is ready.
      if (is_thread_zero_in_block()) {
        wait_until(&pSync[i], ROCSHMEM_CMP_EQ, flag_val);
      }
      __syncthreads();

      T *ptr = &pWrk[i * nelems];
      gda_compute_reduce<T, Op>(ptr, dst, nelems, wg_id, wg_size);
      threadfence_system();
    }
  }

  __syncthreads();

  for (int i = wg_id; i < constmem.num_pes; i += wg_size) {
    pSync[i] = ROCSHMEM_SYNC_VALUE;
  }
  threadfence_system();
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
__device__ void GDAContext::internal_ring_allreduce(T *dst, const T *src,
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
#if defined(__gfx936__) || defined (__gfx938__)
        __threadfence_system();
#endif /* __gfx936__ */
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
#if defined(__gfx936__) || defined (__gfx938__)
        __threadfence_system();
#endif /* __gfx936__ */
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
__device__ int GDAContext::reduce(rocshmem_team_t team, T *dest,
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
    internal_direct_allreduce<T, Op>(dest, source, nreduce, team_obj, wf_info);
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
        internal_ring_allreduce<T, Op>(dest, source, nreduce, team_obj, n_seg,
          seg_size, chunk_size, wf_info);
      }
      if (n_seg_up > n_seg) {
        T *p_dst = (dest + (n_seg * seg_size));
        const T *p_src = (source + (n_seg * seg_size));
        int p_count = nreduce - (n_seg * seg_size);
        int p_chunk = p_count / PE_size;

        if (p_chunk > 0) {
          internal_ring_allreduce<T, Op>(p_dst, p_src, (p_chunk * PE_size),
            team_obj, 1, (p_chunk * PE_size), p_chunk, wf_info);
        }

        if ((p_chunk * PE_size) < p_count) {
          // Final elements need to use direct_allreduce
          p_count -= (p_chunk * PE_size);
          p_dst += (p_chunk * PE_size);
          const T *p_src2 = p_src + (p_chunk * PE_size);

          internal_direct_allreduce<T, Op>(p_dst, p_src2, p_count, team_obj, wf_info);
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
__device__ int GDAContext::reduce_scatter_wg(rocshmem_team_t team, T *dest,
                                             const T *source, int nreduce) {
  GDATeam *team_obj = reinterpret_cast<GDATeam *>(team);
  int PE_size = team_obj->tinfo_wrt_world->size;
  int PE_start = team_obj->tinfo_wrt_world->pe_start;
  int stride = team_obj->tinfo_wrt_world->stride;
  int my_pe = constmem.my_pe;
  int team_rank = (my_pe - PE_start) / stride;

  const int local_size = constmem.ipc_shm_size;
  int local_target_pe{-1};
  const bool has_ipc = ipcImpl_.isIpcAvailable(my_pe, my_pe, &local_target_pe);
  const bool use_hybrid = has_ipc && local_size > 1 && stride == 1 &&
      constmem.ipc_stride == 1 && PE_start % local_size == 0 &&
      PE_size > local_size && PE_size % local_size == 0 && nreduce > 1024;

  if (use_hybrid) {
    return hybrid_reduce_scatter_wg<T, Op>(team_obj, dest, source, nreduce,
                                           PE_start, stride, PE_size, team_rank);
  }

  long *pSync = team_obj->reduce_pSync;
  T *pWrk = reinterpret_cast<T *>(team_obj->pWrk);
  ActiveWFInfo wf_info(ctx_id_, ThreadScope::wg);
  int wg_id = get_flat_block_id();
  int wg_size = get_flat_block_size();
  int pWrk_elems = static_cast<int>(ROCSHMEM_REDUCE_MIN_WRKDATA_SIZE * sizeof(double) / sizeof(T));
  int chunk_size = max(1, pWrk_elems / PE_size);
  int n_chunks = (nreduce + chunk_size - 1) / chunk_size;
  int64_t flag_val = 1;
  int finish = PE_start + stride * PE_size;

  // pWrk holds one incoming contribution per PE, so process large inputs in chunks.
  for (int c = 0; c < n_chunks; c++) {
    int offset = c * chunk_size;
    int count = min(chunk_size, nreduce - offset);

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
 * through an inter-node ring. Large ring blocks are striped over two rails.
 *
 * The caller must provide a contiguous team composed of equally sized IPC
 * islands. Concurrent workgroups must use distinct team-owned pWrk/pSync and
 * input/output buffers. All threads in the workgroup must participate.
 */
template <typename T, ROCSHMEM_OP Op>
__device__ int GDAContext::hybrid_reduce_scatter_wg(
    GDATeam *team_obj, T *dest, const T *source, int nreduce,
    int PE_start, int stride, int PE_size, int team_rank) {

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
  const int chunk_size = max(1, scratch_elems / num_nodes);
  rocshmem_team_t team = reinterpret_cast<rocshmem_team_t>(team_obj);

  const int left_node = (node_id - 1 + num_nodes) % num_nodes;
  const int right_node = (node_id + 1) % num_nodes;
  const int right_pe = PE_start + (right_node * local_size + local_slot) * stride;

  for (int offset = 0; offset < nreduce; offset += chunk_size) {
    const int count = min(chunk_size, nreduce - offset);

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

      // Stripe large blocks across at most two rails.
      constexpr size_t min_bytes_per_rail = 128 * 1024;
      constexpr uint32_t max_collective_rails = 2;
      uint32_t rails = static_cast<uint32_t>((send_bytes + min_bytes_per_rail - 1) / min_bytes_per_rail);
      rails = max(1U, min(rails, min(num_qps_per_pe, max_collective_rails)));

      if (rails == 1) {
        internal_putmem_nbi_wg(dest + offset,
                               p_wrk + send_idx * chunk_size, send_bytes,
                               right_pe, signal_qp, wf_info);
      } else if (is_thread_zero_in_block()) {
        const size_t rail_bytes = (send_bytes + rails - 1) / rails;
        const uint64_t remote_offset = reinterpret_cast<char *>(dest + offset) - base_heap[constmem.my_pe];
        const char *send_src = reinterpret_cast<const char *>(p_wrk + send_idx * chunk_size);
        for (uint32_t rail = 0; rail < rails; ++rail) {
          const size_t begin = static_cast<size_t>(rail) * rail_bytes;
          if (begin >= send_bytes) break;
          const size_t length = min(rail_bytes, send_bytes - begin);
          const uint32_t qp_row = (static_cast<uint32_t>(blockIdx.x) + rail) % num_qps_per_pe;
          const uint32_t qp_index = qp_row * constmem.num_pes + right_pe;
          qps[qp_index].put_nbi(base_heap[right_pe] + remote_offset + begin,
                                send_src + begin, length, wf_info);
        }
        for (uint32_t rail = 0; rail < rails; ++rail) {
          const size_t begin = static_cast<size_t>(rail) * rail_bytes;
          if (begin >= send_bytes) break;
          const uint32_t qp_row = (static_cast<uint32_t>(blockIdx.x) + rail) % num_qps_per_pe;
          const uint32_t qp_index = qp_row * constmem.num_pes + right_pe;
          qps[qp_index].quiet(wf_info);
        }
      }
      __syncthreads();
      if (is_thread_zero_in_block()) {
        // Publish ready only after every payload rail completes.
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
  }

  return ROCSHMEM_SUCCESS;
}

template <typename T>
__device__ void GDAContext::internal_put_broadcast(T *dst, const T *src,
    int nelems, int pe_root, int pe_start, int stride, int pe_size,
    ActiveWFInfo &wf_info) {  // NOLINT(runtime/int)
  if (constmem.my_pe == pe_root) {
    int finish = pe_start + stride * pe_size;
    for (int i = pe_start; i < finish; i += stride) {
      if (constmem.my_pe != i)
        internal_putmem_nbi_wg(dst, src, nelems * sizeof(T), i, i, wf_info);
    }
    memcpy_wg<MemcpyKind::Put>(dst, const_cast<T *>(src), nelems * sizeof(T));
  }
}

template <typename T>
__device__ void GDAContext::internal_get_broadcast(T *dst, const T *src,
    int nelems, int pe_root, ActiveWFInfo &wf_info) {  // NOLINT(runtime/int)
  if (constmem.my_pe == pe_root) {
    memcpy_wg<MemcpyKind::Put>(dst, const_cast<T *>(src), nelems * sizeof(T));
    return;
  }

  const size_t nbytes = static_cast<size_t>(nelems) * sizeof(T);
  int local_pe{-1};
  if (ipcImpl_.isIpcAvailable(constmem.my_pe, pe_root, &local_pe)) {
    const uint64_t offset = reinterpret_cast<const char *>(src) - ipcImpl_.ipc_bases[ipcImpl_.shm_rank];
    ipcImpl_.ipcCopy_wg<MemcpyKind::GetBlocking>(dst, ipcImpl_.ipc_bases[local_pe] + offset, nbytes, local_pe);
    return;
  }

  // QP rows are mapped round-robin across merged NICs.  Stripe a large
  // inter-node message over the useful QP rows, submit every chunk first,
  // and only then wait for completion.
  if (is_thread_zero_in_block()) {
    constexpr size_t MIN_BYTES_PER_RAIL = 64 * 1024;
    uint32_t desired_rails = static_cast<uint32_t>((nbytes + MIN_BYTES_PER_RAIL - 1) / MIN_BYTES_PER_RAIL);
    if (desired_rails == 0) desired_rails = 1;
    desired_rails = min(desired_rails, num_qps_per_pe);

    // Share the available rows among concurrently active workgroups to avoid
    // multiplying the outstanding reads by gridDim.x.
    const uint32_t active_wgs = static_cast<uint32_t>(gridDim.x);
    uint32_t rails = (desired_rails + active_wgs - 1) / active_wgs;
    if (rails == 0) rails = 1;
    rails = min(rails, num_qps_per_pe);

    const size_t chunk = (nbytes + rails - 1) / rails;
    const uint64_t src_offset = reinterpret_cast<const char *>(src) - base_heap[constmem.my_pe];
    char *dst_bytes = reinterpret_cast<char *>(dst);

    for (uint32_t rail = 0; rail < rails; rail++) {
      const size_t begin = static_cast<size_t>(rail) * chunk;
      if (begin >= nbytes) break;
      const size_t length = ((nbytes - begin) < chunk) ? (nbytes - begin) : chunk;
      const uint32_t qp_row = (static_cast<uint32_t>(blockIdx.x) + rail * active_wgs) % num_qps_per_pe;
      const uint32_t qp_index = qp_row * constmem.num_pes + pe_root;
      qps[qp_index].get_nbi(dst_bytes + begin,
                            base_heap[pe_root] + src_offset + begin,
                            length, wf_info);
    }

    for (uint32_t rail = 0; rail < rails; rail++) {
      const size_t begin = static_cast<size_t>(rail) * chunk;
      if (begin >= nbytes) break;
      const uint32_t qp_row = (static_cast<uint32_t>(blockIdx.x) + rail * active_wgs) % num_qps_per_pe;
      const uint32_t qp_index = qp_row * constmem.num_pes + pe_root;
      qps[qp_index].quiet(wf_info);
    }
  }
}

template <typename T>
__device__ void GDAContext::broadcast(rocshmem_team_t team, T *dst,
    const T *src, int nelems, int pe_root) {
  GDATeam *team_obj = reinterpret_cast<GDATeam *>(team);

  int stride = team_obj->tinfo_wrt_world->stride;
  int pe_start = team_obj->tinfo_wrt_world->pe_start;
  int pe_size = team_obj->tinfo_wrt_world->size;
  long *p_sync = team_obj->bcast_pSync;

  // Passed pe_root is relative to team, convert to world root
  int pe_root_world = team_obj->get_pe_in_world(pe_root);
  internal_broadcast<T>(dst, src, nelems, pe_root_world, pe_start, stride, pe_size, p_sync);
}

template <typename T>
__device__ void GDAContext::internal_broadcast(T *dst, const T *src,
    int nelems, int pe_root, int pe_start, int stride, int pe_size,
    long *p_sync) {  // NOLINT(runtime/int)
  ActiveWFInfo wf_info(ctx_id_, ThreadScope::wg);
  if (constmem.num_pes < 4) {
    internal_put_broadcast(dst, src, nelems, pe_root, pe_start, stride, pe_size, wf_info);
    internal_sync_wg(constmem.my_pe, pe_start, stride, pe_size, p_sync, wf_info);
    return;
  }

  // Attempt multi-proxy broadcast for large messages with multiple nodes
  if (internal_proxy_broadcast<T>(dst, src, nelems, pe_root, pe_start,
                                  stride, pe_size, p_sync, wf_info)) {
    return;  // Proxy broadcast handled the operation
  }

  // Hierarchical broadcast: choose one leader in each IPC island.  Only the
  // leader fetches the message from the root (striped across merged NICs for
  // an inter-node root); all other local PEs consume the leader's copy over
  // IPC.  This avoids sending the full message once per GPU across the fabric.
  int node_leader = constmem.my_pe;
  for (int rank = 0; rank < pe_size; rank++) {
    const int peer = pe_start + rank * stride;
    int local_pe{-1};
    if (peer == constmem.my_pe ||
        ipcImpl_.isIpcAvailable(constmem.my_pe, peer, &local_pe)) {
      node_leader = peer;
      break;
    }
  }

  const size_t nbytes = static_cast<size_t>(nelems) * sizeof(T);

  if (constmem.my_pe == pe_root) {
    memcpy_wg<MemcpyKind::Put>(dst, const_cast<T *>(src), nbytes);
  } else if (constmem.my_pe == node_leader) {
    internal_get_broadcast(dst, src, nelems, pe_root, wf_info);
  }

  // Node leaders must finish the inter-node phase before local IPC readers
  // consume their destination buffers.
  internal_sync_wg(constmem.my_pe, pe_start, stride, pe_size, p_sync, wf_info);

  if (constmem.my_pe != node_leader && constmem.my_pe != pe_root) {
    internal_getmem_wg(dst, dst, nbytes, node_leader, node_leader, wf_info);
  }

  internal_sync_wg(constmem.my_pe, pe_start, stride, pe_size, p_sync, wf_info);
}

template <typename T>
__device__ bool GDAContext::internal_proxy_broadcast(T *dst, const T *src,
    int nelems, int pe_root, int pe_start, int stride, int pe_size,
    long *p_sync, ActiveWFInfo &wf_info) {  // NOLINT(runtime/int)

  constexpr size_t MIN_BYTES_PER_PROXY = 64 * 1024;
  const size_t nbytes = static_cast<size_t>(nelems) * sizeof(T);
  const int local_shm_size = constmem.ipc_shm_size;

  // Multi-proxy hierarchical broadcast.  One PE per local GPU/NIC transfers
  // one segment between nodes; IPC stages the root data and fans the segments
  // out locally.  Restrict this path to complete, contiguous node groups so
  // proxy slot N names the same local GPU/NIC on every node.
  const bool use_proxy_bcast =
      nbytes >= kHierarchicalMinBytes && stride == 1 && local_shm_size > 1 &&
      constmem.ipc_stride == 1 && pe_start % local_shm_size == 0 &&
      pe_size % local_shm_size == 0 && pe_size > local_shm_size;
  if (!use_proxy_bcast) {
    return false;
  }

  // Calculate optimal proxy count: balance parallelism vs overhead per proxy
  const size_t useful_proxies = (nbytes + MIN_BYTES_PER_PROXY - 1) / MIN_BYTES_PER_PROXY;
  const int proxies = min(local_shm_size, static_cast<int>(useful_proxies));
  const int local_slot = constmem.my_pe - constmem.ipc_first_pe;
  const int root_rank = pe_root - pe_start;
  const int root_node_first = pe_start + (root_rank / local_shm_size) * local_shm_size;
  const bool on_root_node = constmem.ipc_first_pe == root_node_first;
  const size_t chunk = (nbytes + proxies - 1) / proxies;
  char *dst_bytes = reinterpret_cast<char *>(dst);
  const char *src_bytes = reinterpret_cast<const char *>(src);

  // Root keeps its complete result.  The other root-node proxies stage only
  // their segment in dst, making a symmetric source address available to
  // the corresponding proxy on every remote node.
  if (constmem.my_pe == pe_root) {
    memcpy_wg<MemcpyKind::Put>(dst_bytes, const_cast<char *>(src_bytes), nbytes);
  } else if (on_root_node && local_slot < proxies) {
    const size_t begin = static_cast<size_t>(local_slot) * chunk;
    if (begin < nbytes) {
      const size_t length = min(chunk, nbytes - begin);
      internal_getmem_wg(dst_bytes + begin, src_bytes + begin, length,
                         pe_root, pe_root, wf_info);
    }
  }

  internal_sync_wg(constmem.my_pe, pe_start, stride, pe_size, p_sync, wf_info);

  // Every remote-node proxy fetches one disjoint segment.  Since NIC
  // selection is unique per local PE, the node drives its available NICs in
  // parallel instead of funneling all traffic through one leader GPU.
  if (!on_root_node && local_slot < proxies) {
    const size_t begin = static_cast<size_t>(local_slot) * chunk;
    if (begin < nbytes) {
      const size_t length = min(chunk, nbytes - begin);
      const int source_proxy = root_node_first + local_slot;
      const uint32_t qp_row = static_cast<uint32_t>(blockIdx.x) % num_qps_per_pe;
      const int qp_index = qp_row * constmem.num_pes + source_proxy;
      internal_getmem_wg(dst_bytes + begin, dst_bytes + begin, length,
                         source_proxy, qp_index, wf_info);
    }
  }

  internal_sync_wg(constmem.my_pe, pe_start, stride, pe_size, p_sync, wf_info);

  // Assemble the complete result from this node's local proxy buffers.
  for (int slot = 0; slot < proxies; slot++) {
    if (slot == local_slot) continue;
    const size_t begin = static_cast<size_t>(slot) * chunk;
    if (begin >= nbytes) break;
    const size_t length = min(chunk, nbytes - begin);
    const int local_proxy = constmem.ipc_first_pe + slot;
    internal_getmem_wg(dst_bytes + begin, dst_bytes + begin, length,
                       local_proxy, local_proxy, wf_info);
  }

  internal_sync_wg(constmem.my_pe, pe_start, stride, pe_size, p_sync, wf_info);

  return true;  // Proxy broadcast completed successfully
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
    qps[qp_index].atomic_nofetch_single(dest, value);
    return;
  }

  uint64_t pending_lanes = get_active_lane_mask();
  const int lane_id = get_flat_block_id() % WF_SIZE;
  // Submit each QP group through the existing wave-batched path.
  while (pending_lanes) {
    const uint64_t qp_group = get_next_qp_group(qp_index, pending_lanes);
    if (qp_group & (uint64_t{1} << lane_id)) {
      ActiveWFInfo wf_info(qp_index);
      qps[qp_index].atomic_nofetch(dest, value, 0, wf_info);
    }
    pending_lanes &= ~qp_group;
  }
}

template <typename T>
__device__ void GDAContext::alltoall(rocshmem_team_t team, T *dst,
                                     const T *src, int nelems,
                                     int elem_offset, int elem_count) {
  alltoall_linear_thread_puts(team, dst, src, nelems, elem_offset, elem_count);
}

template <typename T>
__device__ void GDAContext::alltoallv(rocshmem_team_t team,
                                      T *dest, const size_t dest_nelems[],
                                      const size_t dest_displs[],
                                      T *source, const size_t source_nelems[],
                                      const size_t source_displs[]) {
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
    alltoall(team, dest, source, static_cast<int>(nelems), 0,
             static_cast<int>(nelems));
    return;
  }

  if (constmem.alltoall_wg_algo == gda::ALLTOALLV_WG_ALGO_COPY) {
    alltoallv_copy(team,
                   dest, dest_nelems, dest_displs,
                   source, source_nelems, source_displs);
  } else {
    alltoallv_get(team,
                  dest, dest_nelems, dest_displs,
                  source, source_nelems, source_displs);
  }
}

template <typename T>
__device__ void GDAContext::alltoallv_copy(rocshmem_team_t team, T *dest,
    const size_t dest_nelems[], const size_t dest_displs[], T *source,
    const size_t source_nelems[], const size_t source_displs[]) {
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

  // Have each PE put their designated data to the other PEs
  for (int j = tid; j < pe_size; j+= step_size) {
    int dest_pe = team_obj->get_pe_in_world(j);
    int qp_index = get_team_qp_index(team_obj, dest_pe);
    uint64_t base_heap_offset = base_heap[dest_pe] - base_heap[constmem.my_pe];
    size_t nelems = source_nelems[dest_pe] * sizeof(T);
    char* amo_dst = ((char*)&pSync[alltoall_pSync_offset + my_pe_in_team] + base_heap_offset);

    if (nelems != 0) {
      T* src = (T*)((char*)source + (source_displs[j] * sizeof(T)));
      T* dst = (T*)((char*)&tmp_buf[constmem.my_pe * tmp_buf_off] + base_heap_offset);
      put_nbi_shared_qp(qp_index, dst, src, nelems, false);
    }

    atomic_nofetch_shared_qp(qp_index, amo_dst, 1);
  }

  // wait until everyone has obtained their designated data
  for (int j = tid; j < pe_size; j+= step_size) {
    int dest_pe = team_obj->get_pe_in_world(j);
    int qp_index = get_team_qp_index(team_obj, dest_pe);

    long *sync_flags = &pSync[alltoall_pSync_offset + dest_pe];
    while (uncached_load(sync_flags) < completion_count) { }

    qps[qp_index].quiet_single();

  }

  // Copy out of staging buffer
  __syncthreads();

  for (int j = 0; j < pe_size; j++) {
    size_t nelems = dest_nelems[j] * sizeof(T);

    if (nelems != 0) {
      T* dst = (T*)((char*) dest + dest_displs[j] * sizeof(T));
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
__device__ void GDAContext::alltoallv_get(rocshmem_team_t team, T *dest,
    [[maybe_unused]] const size_t dest_nelems[], const size_t dest_displs[], T *source,
    [[maybe_unused]] const size_t source_nelems[], const size_t source_displs[]) {
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
  for (int j = tid; j < pe_size; j+= step_size) {
    uint64_t *src;
    uint64_t *dst;
    uint64_t seq_bits;
    uint64_t displ_bits;

    int dest_pe = team_obj->get_pe_in_world(j);
    int qp_index = get_team_qp_index(team_obj, dest_pe);
    uint64_t base_heap_offset = base_heap[dest_pe] - base_heap[constmem.my_pe];

    /* Pack Ctrl Message * 16 bits seq | 48bit displ */
    seq_bits = (seq_mask & (a2a_sn + 1)) << seq_shift;
    displ_bits = (displs_mask & source_displs[dest_pe]);
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
    size_t nelems = dest_nelems[dest_pe] * sizeof(T);
    uint64_t *src = (uint64_t*)((char*)source +
                               (displ_bits * sizeof(T)) + base_heap_offset);
    uint64_t *dst = (uint64_t*)((char*)dest +
                               (dest_displs[j] * sizeof(T)));

    get_nbi_shared_qp(qp_index, dst, src, nelems);

    /* Put Completion */
    char* amo_dst = ((char*)&pSync[alltoall_pSync_offset + my_pe_in_team] + base_heap_offset);
    atomic_nofetch_shared_qp(qp_index, amo_dst, 1);
  }

  __syncthreads();

  /* Phase 3: wait for all peers and drain the QPs used by this team. */
  for (int j = tid; j < pe_size; j+= step_size) {
    int dest_pe = team_obj->get_pe_in_world(j);
    int qp_index = get_team_qp_index(team_obj, dest_pe);
    long *sync_flags = &pSync[alltoall_pSync_offset + dest_pe];
    while (uncached_load(sync_flags) < completion_count) { }

    qps[qp_index].quiet_single();
  }

  if (is_thread_zero_in_block()) {
    team_obj->alltoall_sequence_number++;
  }

  __syncthreads();
}

template <typename T>
__device__ void GDAContext::alltoall_linear(rocshmem_team_t team, T *dst,
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
  // Have each PE put their designated data to the other PEs
  for (int j = wf_id; j < pe_size; j+= wf_count) {
    int dest_pe = team_obj->get_pe_in_world(j);
    internal_putmem_nbi_wave(&dst[my_pe_in_team * nelems + elem_offset],
      &src[j * nelems + elem_offset],
      elem_count * sizeof(T), dest_pe, dest_pe, wf_info);
  }

  for (int j = wf_id; j < pe_size; j+= wf_count) {
    int dest_pe = team_obj->get_pe_in_world(j);
    qps[dest_pe].quiet(wf_info);
  }

  // wait until everyone has obtained their designated data
  internal_sync_wg(constmem.my_pe, pe_start, stride, pe_size, pSync, wf_info);
}

template <typename T>
__device__ void GDAContext::alltoall_linear_thread_puts(rocshmem_team_t team,
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
__device__ void GDAContext::fcollect(rocshmem_team_t team, T *dst,
                                     const T *src, int nelems) {
  fcollect_linear(team, dst, src, nelems);
}

template <typename T>
__device__ void GDAContext::fcollect_linear(rocshmem_team_t team, T *dst,
                                            const T *src, int nelems) {
  GDATeam *team_obj = reinterpret_cast<GDATeam *>(team);

  int pe_start = team_obj->tinfo_wrt_world->pe_start;
  int pe_size = team_obj->num_pes;
  int stride = team_obj->tinfo_wrt_world->stride;
  long *pSync = team_obj->alltoall_pSync;
  int my_pe_in_team = team_obj->my_pe;

  ActiveWFInfo wf_info(ctx_id_, ThreadScope::wg);

  if (pe_size >= 4 && static_cast<size_t>(nelems) * sizeof(T) >= kHierarchicalMinBytes) {
    // FCollect is a concatenation of one broadcast per source rank.  Reusing
    // the hierarchical broadcast path makes each source cross a node boundary
    // once, followed by IPC fan-out inside the destination node, instead of
    // sending the same source segment independently to every remote PE.
    for (int rank = 0; rank < pe_size; rank++) {
      const int root = team_obj->get_pe_in_world(rank);
      internal_broadcast(dst + rank * nelems, src, nelems, root, pe_start, stride, pe_size, pSync);
    }
    return;
  }

  // Small messages avoid the extra two synchronization phases per source.
  for (int rank = 0; rank < pe_size; rank++) {
    int dest_pe = team_obj->get_pe_in_world(rank);
    internal_putmem_nbi_wg(&dst[my_pe_in_team * nelems], src,
                           nelems * sizeof(T), dest_pe, dest_pe, wf_info);
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
  internal_sync_wg(constmem.my_pe, pe_start, stride, pe_size, pSync, wf_info);
}

// Block/wave functions
template <typename T>
__device__ void GDAContext::put_wg(T *dest, const T *source, size_t nelems, int pe) {
  putmem_wg(dest, source, nelems * sizeof(T), pe);
}

template <typename T>
__device__ void GDAContext::put_nbi_wg(T *dest, const T *source, size_t nelems, int pe) {
  putmem_nbi_wg(dest, source, nelems * sizeof(T), pe);
}

  template <typename T>
__device__ void GDAContext::put_wave(T *dest, const T *source, size_t nelems, int pe) {
  putmem_wave(dest, source, nelems * sizeof(T), pe);
}

template <typename T>
__device__ void GDAContext::put_nbi_wave(T *dest, const T *source, size_t nelems, int pe) {
  putmem_nbi_wave(dest, source, nelems * sizeof(T), pe);
}

template <typename T>
__device__ void GDAContext::put_nbi_wave_dp(T *dest, const T *source, size_t nelems, int qp_idx, int pe) {
  putmem_nbi_wave_dp(dest, source, nelems * sizeof(T), qp_idx, pe);
}


template <typename T>
__device__ void GDAContext::get_wg(T *dest, const T *source, size_t nelems, int pe) {
  getmem_wg(dest, source, nelems * sizeof(T), pe);
}

template <typename T>
__device__ void GDAContext::get_nbi_wg(T *dest, const T *source, size_t nelems, int pe) {
  getmem_nbi_wg(dest, source, nelems * sizeof(T), pe);
}

template <typename T>
__device__ void GDAContext::get_wave(T *dest, const T *source, size_t nelems, int pe) {
  getmem_wave(dest, source, nelems * sizeof(T), pe);
}

template <typename T>
__device__ void GDAContext::get_nbi_wave(T *dest, const T *source, size_t nelems, int pe) {
  getmem_nbi_wave(dest, source, nelems * sizeof(T), pe);
}

#define GDA_CONTEXT_PUT_SIGNAL_DEF(SUFFIX)                                                            \
  template <typename T>                                                                               \
  __device__ void GDAContext::put_signal##SUFFIX(T *dest, const T *source, size_t nelems,             \
                                                 uint64_t *sig_addr, uint64_t signal, int sig_op,     \
                                                 int pe) {                                            \
    putmem_signal##SUFFIX(dest, source, nelems * sizeof(T), sig_addr, signal, sig_op, pe);            \
  }                                                                                                   \
                                                                                                      \
  template <typename T>                                                                               \
  __device__ void GDAContext::put_signal_nbi##SUFFIX(T *dest, const T *source, size_t nelems,         \
                                                     uint64_t *sig_addr, uint64_t signal, int sig_op, \
                                                     int pe) {                                        \
    putmem_signal##SUFFIX(dest, source, nelems * sizeof(T), sig_addr, signal, sig_op, pe);            \
  }

GDA_CONTEXT_PUT_SIGNAL_DEF()
GDA_CONTEXT_PUT_SIGNAL_DEF(_wg)
GDA_CONTEXT_PUT_SIGNAL_DEF(_wave)

// Internal functions used by collective and signal operations
template <typename T>
__device__ void GDAContext::internal_amo_add(void *dst, T value, int pe,
    int qp_index, ActiveWFInfo &wf_info) {
  if constexpr (sizeof(T) != 8) { LOGD_ERROR_ABORT("gda::amo_add not implemented for non-64bit types"); }//TODO:support for non-uint64t
  int local_pe{-1};
  if (ipcImpl_.isIpcAvailable(constmem.my_pe, pe, &local_pe)) {
    ipcImpl_.ipcAMOAdd(reinterpret_cast<T *>(get_local_ptr(dst, local_pe)), value);
    return;
  }

  uint64_t L_offset = reinterpret_cast<char *>(dst) - base_heap[constmem.my_pe];
  bool need_turn {true};
  uint64_t turns = __ballot(need_turn);
  while (turns) {
    uint8_t lane = __ffsll((unsigned long long)turns) - 1;
    int pe_turn = __shfl(pe, lane);
    if (pe_turn == pe) {
      qps[qp_index].atomic_nofetch(base_heap[pe] + L_offset, value, 0, wf_info);
      need_turn = false;
    }
    turns = __ballot(need_turn);
  }
}

template <typename T>
__device__ T GDAContext::internal_amo_fetch_add(void *dst, T value, int pe,
    int qp_index, ActiveWFInfo &wf_info) {
  if constexpr (sizeof(T) != 8) { LOGD_ERROR_ABORT("gda::amo_fadd not implemented for non-64bit types"); }//TODO:support for non-uint64t
  int local_pe{-1};
  if (ipcImpl_.isIpcAvailable(constmem.my_pe, pe, &local_pe)) {
    return ipcImpl_.ipcAMOFetchAdd(reinterpret_cast<T *>(get_local_ptr(dst, local_pe)), value);
  }

  uint64_t L_offset = reinterpret_cast<char *>(dst) - base_heap[constmem.my_pe];
  T ret_val = 0;
  bool need_turn {true};
  uint64_t turns = __ballot(need_turn);
  while (turns) {
    uint8_t lane = __ffsll((unsigned long long)turns) - 1;
    int pe_turn = __shfl(pe, lane);
    if (pe_turn == pe) {
      ret_val =  qps[qp_index].atomic_fetch(base_heap[pe] + L_offset, value, 0, wf_info);
      need_turn = false;
    }
    turns = __ballot(need_turn);
  }
  return ret_val;
}

template <typename T>
__device__ T GDAContext::internal_amo_swap(void *dst, T value, int pe,
    int qp_index, ActiveWFInfo &wf_info) {
  if constexpr (sizeof(T) != 8) { LOGD_ERROR_ABORT("gda::amo_set not implemented for non-64bit types"); }//TODO:support for non-uint64t
  int local_pe{-1};
  if (ipcImpl_.isIpcAvailable(constmem.my_pe, pe, &local_pe)) {
    return ipcImpl_.ipcAMOSwap(reinterpret_cast<T *>(get_local_ptr(dst, local_pe)), value);
  }

  uint64_t L_offset = reinterpret_cast<char *>(dst) - base_heap[constmem.my_pe];
  bool need_turn {true};
  uint64_t turns = __ballot(need_turn);
  T ret_val;
  T cond = 0;
  while (turns) {
    uint8_t lane = __ffsll((unsigned long long)turns) - 1;
    int pe_turn = __shfl(pe, lane);
    if (pe_turn == pe) {
      /**
       * Guess that the remote memory is zero by setting condition to zero.
       * The compare-and-swap loop will execute at least twice if wrong.
       * It may run additional times if contention on memory location.
       */
      while (wf_info.update(pe), (ret_val = qps[qp_index].atomic_cas(
             base_heap[pe] + L_offset, value, cond, wf_info)) != cond) {
        cond = ret_val;
      }
      need_turn = false;
    }
    turns = __ballot(need_turn);
  }
  return ret_val;
}

/******************************************************************************
 ****************************** INLINE FUNCTIONS ******************************
 *****************************************************************************/

/**
 * @brief Get the Queue Pair index for a given PE based on a atomic counter
 *        This ensures even distribution of requests across multiple QPs
 *        allocated per PE.
 * @param pe The target PE
 * @return The Queue Pair index
 *
 * Explanation of QP indexing scheme:
 *  num_qps_per_pe = 4
 *  num_pes        = 3
 *
 *  Layout of QPs per PE:
 *
 *             PE0          PE1          PE2
 *           ───────      ───────      ───────
 *  QP0  ─> [ QP0,0 ]    [ QP0,1 ]    [ QP0,2 ]
 *  QP1  ─> [ QP1,0 ]    [ QP1,1 ]    [ QP1,2 ]
 *  QP2  ─> [ QP2,0 ]    [ QP2,1 ]  **[ QP2,2 ]** <-- highlighted (3rd QP of PE2)
 *  QP3  ─> [ QP3,0 ]    [ QP3,1 ]    [ QP3,2 ]
 *
 *  Legend:
 *    - num_qps_per_pe = 4  →  Four Queue Pairs per PE
 *    - num_pes = 3         →  Three Processing Elements (PE0–PE2)
 *    - QP[i,j]             →  i-th QP of PE j
 *    - **[ QP2,2 ]**       →  The 3rd QP (QP index 2) of PE2
 */
__device__ __forceinline__ uint32_t GDAContext::get_qp_index(int pe,
    ActiveWFInfo wf_info) {

  uint32_t qp_index   {0};

  if(wf_info.pe_group_logical_lane_id == 0) {
    // Only the leader lane updates the counter (Does it require atomics?)
    // uint32_t local_qp_counter = __hip_atomic_fetch_add(&qp_counter[pe], 1,
    //                                        __ATOMIC_RELAXED,
    //                                        __HIP_MEMORY_SCOPE_AGENT);
    // local_qp_counter %= num_qps_per_pe;
    // qp_index = (local_qp_counter * num_pes) + pe;
    qp_index = (qp_counter[pe]++ % num_qps_per_pe) * constmem.num_pes + pe;
  }

  // Broadcast the qp_index value to other lanes in the wavefront
  // that are targeting the same PE
  qp_index = __shfl_sync(wf_info.pe_group_mask, qp_index, wf_info.pe_group_first_phys_lane_id);

  return qp_index;
}

__device__ __forceinline__ uint32_t GDAContext::get_team_qp_index(
    const GDATeam *team_obj, int pe) {
  const uint32_t qp_slot =
      static_cast<uint32_t>(team_obj->pool_index_) % num_qps_per_pe;
  return qp_slot * constmem.num_pes + pe;
}

/******************************************************************************
 **************** TILE API STUB IMPLEMENTATIONS (NOT IMPLEMENTED) *************
 *****************************************************************************/

// RMA PUT operations - Type-erased interface
__device__ inline int GDAContext::tile_put([[maybe_unused]] void* dst_data, [[maybe_unused]] const void* src_data,
                                    [[maybe_unused]] const size_t* dst_strides, [[maybe_unused]] const size_t* src_strides,
                                    [[maybe_unused]] const size_t* start_coord, [[maybe_unused]] const size_t* boundary,
                                    [[maybe_unused]] int ndim, [[maybe_unused]] size_t element_size,
                                    [[maybe_unused]] int pe, [[maybe_unused]] uint64_t flags) {
  LOGD_WARN("Tile API not implemented for GDA backend");
  return ROCSHMEM_ERROR;
}

__device__ inline int GDAContext::tile_put_wave([[maybe_unused]] void* dst_data, [[maybe_unused]] const void* src_data,
                                         [[maybe_unused]] const size_t* dst_strides, [[maybe_unused]] const size_t* src_strides,
                                         [[maybe_unused]] const size_t* start_coord, [[maybe_unused]] const size_t* boundary,
                                         [[maybe_unused]] int ndim, [[maybe_unused]] size_t element_size,
                                         [[maybe_unused]] int pe, [[maybe_unused]] uint64_t flags) {
  LOGD_WARN("Tile API not implemented for GDA backend");
  return ROCSHMEM_ERROR;
}

__device__ inline int GDAContext::tile_put_wg([[maybe_unused]] void* dst_data, [[maybe_unused]] const void* src_data,
                                       [[maybe_unused]] const size_t* dst_strides, [[maybe_unused]] const size_t* src_strides,
                                       [[maybe_unused]] const size_t* start_coord, [[maybe_unused]] const size_t* boundary,
                                       [[maybe_unused]] int ndim, [[maybe_unused]] size_t element_size,
                                       [[maybe_unused]] int pe, [[maybe_unused]] uint64_t flags) {
  LOGD_WARN("Tile API not implemented for GDA backend");
  return ROCSHMEM_ERROR;
}

// RMA GET operations - Type-erased interface
__device__ inline int GDAContext::tile_get([[maybe_unused]] void* dst_data, [[maybe_unused]] const void* src_data,
                                    [[maybe_unused]] const size_t* dst_strides, [[maybe_unused]] const size_t* src_strides,
                                    [[maybe_unused]] const size_t* start_coord, [[maybe_unused]] const size_t* boundary,
                                    [[maybe_unused]] int ndim, [[maybe_unused]] size_t element_size,
                                    [[maybe_unused]] int pe, [[maybe_unused]] uint64_t flags) {
  LOGD_WARN("Tile API not implemented for GDA backend");
  return ROCSHMEM_ERROR;
}

__device__ inline int GDAContext::tile_get_wave([[maybe_unused]] void* dst_data, [[maybe_unused]] const void* src_data,
                                         [[maybe_unused]] const size_t* dst_strides, [[maybe_unused]] const size_t* src_strides,
                                         [[maybe_unused]] const size_t* start_coord, [[maybe_unused]] const size_t* boundary,
                                         [[maybe_unused]] int ndim, [[maybe_unused]] size_t element_size,
                                         [[maybe_unused]] int pe, [[maybe_unused]] uint64_t flags) {
  LOGD_WARN("Tile API not implemented for GDA backend");
  return ROCSHMEM_ERROR;
}

__device__ inline int GDAContext::tile_get_wg([[maybe_unused]] void* dst_data, [[maybe_unused]] const void* src_data,
                                       [[maybe_unused]] const size_t* dst_strides, [[maybe_unused]] const size_t* src_strides,
                                       [[maybe_unused]] const size_t* start_coord, [[maybe_unused]] const size_t* boundary,
                                       [[maybe_unused]] int ndim, [[maybe_unused]] size_t element_size,
                                       [[maybe_unused]] int pe, [[maybe_unused]] uint64_t flags) {
  LOGD_WARN("Tile API not implemented for GDA backend");
  return ROCSHMEM_ERROR;
}

// Allgather operations - Type-erased interface
__device__ inline int GDAContext::tile_allgather([[maybe_unused]] rocshmem_team_t team,
                                          [[maybe_unused]] void* dst_data, [[maybe_unused]] const void* src_data,
                                          [[maybe_unused]] const size_t* dst_strides, [[maybe_unused]] const size_t* src_strides,
                                          [[maybe_unused]] const size_t* start_coord, [[maybe_unused]] const size_t* boundary,
                                          [[maybe_unused]] int ndim, [[maybe_unused]] size_t element_size,
                                          [[maybe_unused]] uint64_t flags) {
  LOGD_WARN("Tile API not implemented for GDA backend");
  return ROCSHMEM_ERROR;
}

__device__ inline int GDAContext::tile_allgather_wave([[maybe_unused]] rocshmem_team_t team,
                                               [[maybe_unused]] void* dst_data, [[maybe_unused]] const void* src_data,
                                               [[maybe_unused]] const size_t* dst_strides, [[maybe_unused]] const size_t* src_strides,
                                               [[maybe_unused]] const size_t* start_coord, [[maybe_unused]] const size_t* boundary,
                                               [[maybe_unused]] int ndim, [[maybe_unused]] size_t element_size,
                                               [[maybe_unused]] uint64_t flags) {
  LOGD_WARN("Tile API not implemented for GDA backend");
  return ROCSHMEM_ERROR;
}

__device__ inline int GDAContext::tile_allgather_wg([[maybe_unused]] rocshmem_team_t team,
                                             [[maybe_unused]] void* dst_data, [[maybe_unused]] const void* src_data,
                                             [[maybe_unused]] const size_t* dst_strides, [[maybe_unused]] const size_t* src_strides,
                                             [[maybe_unused]] const size_t* start_coord, [[maybe_unused]] const size_t* boundary,
                                             [[maybe_unused]] int ndim, [[maybe_unused]] size_t element_size,
                                             [[maybe_unused]] uint64_t flags) {
  LOGD_WARN("Tile API not implemented for GDA backend");
  return ROCSHMEM_ERROR;
}

// Broadcast operations - Type-erased interface
__device__ inline int GDAContext::tile_broadcast([[maybe_unused]] rocshmem_team_t team,
                                          [[maybe_unused]] void* dst_data, [[maybe_unused]] const void* src_data,
                                          [[maybe_unused]] const size_t* dst_strides, [[maybe_unused]] const size_t* src_strides,
                                          [[maybe_unused]] const size_t* start_coord, [[maybe_unused]] const size_t* boundary,
                                          [[maybe_unused]] int ndim, [[maybe_unused]] size_t element_size,
                                          [[maybe_unused]] int pe_root, [[maybe_unused]] uint64_t flags) {
  LOGD_WARN("Tile API not implemented for GDA backend");
  return ROCSHMEM_ERROR;
}

__device__ inline int GDAContext::tile_broadcast_wave([[maybe_unused]] rocshmem_team_t team,
                                               [[maybe_unused]] void* dst_data, [[maybe_unused]] const void* src_data,
                                               [[maybe_unused]] const size_t* dst_strides, [[maybe_unused]] const size_t* src_strides,
                                               [[maybe_unused]] const size_t* start_coord, [[maybe_unused]] const size_t* boundary,
                                               [[maybe_unused]] int ndim, [[maybe_unused]] size_t element_size,
                                               [[maybe_unused]] int pe_root, [[maybe_unused]] uint64_t flags) {
  LOGD_WARN("Tile API not implemented for GDA backend");
  return ROCSHMEM_ERROR;
}

__device__ inline int GDAContext::tile_broadcast_wg([[maybe_unused]] rocshmem_team_t team,
                                             [[maybe_unused]] void* dst_data, [[maybe_unused]] const void* src_data,
                                             [[maybe_unused]] const size_t* dst_strides, [[maybe_unused]] const size_t* src_strides,
                                             [[maybe_unused]] const size_t* start_coord, [[maybe_unused]] const size_t* boundary,
                                             [[maybe_unused]] int ndim, [[maybe_unused]] size_t element_size,
                                             [[maybe_unused]] int pe_root, [[maybe_unused]] uint64_t flags) {
  LOGD_WARN("Tile API not implemented for GDA backend");
  return ROCSHMEM_ERROR;
}

// SUM Reduction operations - Type-erased interface
__device__ inline int GDAContext::tile_sum_reduce([[maybe_unused]] rocshmem_team_t team,
                                           [[maybe_unused]] void* dst_data, [[maybe_unused]] const void* src_data,
                                           [[maybe_unused]] const size_t* dst_strides, [[maybe_unused]] const size_t* src_strides,
                                           [[maybe_unused]] const size_t* start_coord, [[maybe_unused]] const size_t* boundary,
                                           [[maybe_unused]] int ndim, [[maybe_unused]] size_t element_size,
                                           [[maybe_unused]] int root, [[maybe_unused]] uint64_t flags) {
  LOGD_WARN("Tile API not implemented for GDA backend");
  return ROCSHMEM_ERROR;
}

__device__ inline int GDAContext::tile_sum_reduce_wave([[maybe_unused]] rocshmem_team_t team,
                                                [[maybe_unused]] void* dst_data, [[maybe_unused]] const void* src_data,
                                                [[maybe_unused]] const size_t* dst_strides, [[maybe_unused]] const size_t* src_strides,
                                                [[maybe_unused]] const size_t* start_coord, [[maybe_unused]] const size_t* boundary,
                                                [[maybe_unused]] int ndim, [[maybe_unused]] size_t element_size,
                                                [[maybe_unused]] int root, [[maybe_unused]] uint64_t flags) {
  LOGD_WARN("Tile API not implemented for GDA backend");
  return ROCSHMEM_ERROR;
}

__device__ inline int GDAContext::tile_sum_reduce_wg([[maybe_unused]] rocshmem_team_t team,
                                              [[maybe_unused]] void* dst_data, [[maybe_unused]] const void* src_data,
                                              [[maybe_unused]] const size_t* dst_strides, [[maybe_unused]] const size_t* src_strides,
                                              [[maybe_unused]] const size_t* start_coord, [[maybe_unused]] const size_t* boundary,
                                              [[maybe_unused]] int ndim, [[maybe_unused]] size_t element_size,
                                              [[maybe_unused]] int root, [[maybe_unused]] uint64_t flags) {
  LOGD_WARN("Tile API not implemented for GDA backend");
  return ROCSHMEM_ERROR;
}

// MAX Reduction operations - Type-erased interface
__device__ inline int GDAContext::tile_max_reduce([[maybe_unused]] rocshmem_team_t team,
                                                   [[maybe_unused]] void* dst_data,
                                                   [[maybe_unused]] const void* src_data,
                                                   [[maybe_unused]] const size_t* dst_strides,
                                                   [[maybe_unused]] const size_t* src_strides,
                                                   [[maybe_unused]] const size_t* start_coord,
                                                   [[maybe_unused]] const size_t* boundary,
                                                   [[maybe_unused]] int ndim,
                                                   [[maybe_unused]] size_t element_size,
                                                   [[maybe_unused]] int root,
                                                   [[maybe_unused]] uint64_t flags) {
  LOGD_WARN("Tile API not implemented for GDA backend");
  return ROCSHMEM_ERROR;
}

__device__ inline int GDAContext::tile_max_reduce_wave([[maybe_unused]] rocshmem_team_t team,
                                                        [[maybe_unused]] void* dst_data,
                                                        [[maybe_unused]] const void* src_data,
                                                        [[maybe_unused]] const size_t* dst_strides,
                                                        [[maybe_unused]] const size_t* src_strides,
                                                        [[maybe_unused]] const size_t* start_coord,
                                                        [[maybe_unused]] const size_t* boundary,
                                                        [[maybe_unused]] int ndim,
                                                        [[maybe_unused]] size_t element_size,
                                                        [[maybe_unused]] int root,
                                                        [[maybe_unused]] uint64_t flags) {
  LOGD_WARN("Tile API not implemented for GDA backend");
  return ROCSHMEM_ERROR;
}

__device__ inline int GDAContext::tile_max_reduce_wg([[maybe_unused]] rocshmem_team_t team,
                                                      [[maybe_unused]] void* dst_data,
                                                      [[maybe_unused]] const void* src_data,
                                                      [[maybe_unused]] const size_t* dst_strides,
                                                      [[maybe_unused]] const size_t* src_strides,
                                                      [[maybe_unused]] const size_t* start_coord,
                                                      [[maybe_unused]] const size_t* boundary,
                                                      [[maybe_unused]] int ndim,
                                                      [[maybe_unused]] size_t element_size,
                                                      [[maybe_unused]] int root,
                                                      [[maybe_unused]] uint64_t flags) {
  LOGD_WARN("Tile API not implemented for GDA backend");
  return ROCSHMEM_ERROR;
}

// MIN Reduction operations - Type-erased interface
__device__ inline int GDAContext::tile_min_reduce([[maybe_unused]] rocshmem_team_t team,
                                                   [[maybe_unused]] void* dst_data,
                                                   [[maybe_unused]] const void* src_data,
                                                   [[maybe_unused]] const size_t* dst_strides,
                                                   [[maybe_unused]] const size_t* src_strides,
                                                   [[maybe_unused]] const size_t* start_coord,
                                                   [[maybe_unused]] const size_t* boundary,
                                                   [[maybe_unused]] int ndim,
                                                   [[maybe_unused]] size_t element_size,
                                                   [[maybe_unused]] int root,
                                                   [[maybe_unused]] uint64_t flags) {
  LOGD_WARN("Tile API not implemented for GDA backend");
  return ROCSHMEM_ERROR;
}

__device__ inline int GDAContext::tile_min_reduce_wave([[maybe_unused]] rocshmem_team_t team,
                                                        [[maybe_unused]] void* dst_data,
                                                        [[maybe_unused]] const void* src_data,
                                                        [[maybe_unused]] const size_t* dst_strides,
                                                        [[maybe_unused]] const size_t* src_strides,
                                                        [[maybe_unused]] const size_t* start_coord,
                                                        [[maybe_unused]] const size_t* boundary,
                                                        [[maybe_unused]] int ndim,
                                                        [[maybe_unused]] size_t element_size,
                                                        [[maybe_unused]] int root,
                                                        [[maybe_unused]] uint64_t flags) {
  LOGD_WARN("Tile API not implemented for GDA backend");
  return ROCSHMEM_ERROR;
}

__device__ inline int GDAContext::tile_min_reduce_wg([[maybe_unused]] rocshmem_team_t team,
                                                      [[maybe_unused]] void* dst_data,
                                                      [[maybe_unused]] const void* src_data,
                                                      [[maybe_unused]] const size_t* dst_strides,
                                                      [[maybe_unused]] const size_t* src_strides,
                                                      [[maybe_unused]] const size_t* start_coord,
                                                      [[maybe_unused]] const size_t* boundary,
                                                      [[maybe_unused]] int ndim,
                                                      [[maybe_unused]] size_t element_size,
                                                      [[maybe_unused]] int root,
                                                      [[maybe_unused]] uint64_t flags) {
  LOGD_WARN("Tile API not implemented for GDA backend");
  return ROCSHMEM_ERROR;
}

// Rooted SUM Reduction operations
// Rooted MAX Reduction operations
// Rooted MIN Reduction operations
}  // namespace rocshmem

#endif  // LIBRARY_SRC_GDA_CONTEXT_TMPL_DEVICE_HPP_
