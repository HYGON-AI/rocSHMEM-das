// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

/******************************************************************************
 * 1. 较大的DAS自研改动在本文件中维护，尽量保留原文件中的社区代码，
 *    方便后续维护及同步上游更新。
 * 2. 修改已有函数时，在此实现对应的_das函数；启用USE_DAS后(默认开启)，原函数
 *    在入口调用_das函数并返回，未启用时执行原来的社区实现。
 * 3. 新增辅助函数也在此维护，公共逻辑尽量复用原文件中的函数；
 *****************************************************************************/

#ifndef LIBRARY_SRC_IPC_CONTEXT_TMPL_DEVICE_DAS_HPP_
#define LIBRARY_SRC_IPC_CONTEXT_TMPL_DEVICE_DAS_HPP_

#include "rocshmem/rocshmem_config.h"

#if defined(USE_DAS)

namespace rocshmem {

// Pipeline chunk size for large messages (256KB)
constexpr size_t ROCSHMEM_CHUNK_BYTES = 262144;

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
__device__ void IPCContext::internal_ring_allreduce_das(
    T *dst, const T *src, int nelems, IPCTeam *team_obj,  // NOLINT(runtime/int)
    int n_seg, int seg_size, int chunk_size) {

  int PE_size = team_obj->tinfo_wrt_world->size;
  long *pSync = team_obj->reduce_pSync;
  T *pWrk = reinterpret_cast<T *>(team_obj->pWrk);
  int my_pe_in_team = team_obj->my_pe;

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
    // Loop 1 in the algorithm above
    for (int iter = 0; iter < PE_size - 1; iter++) {
      off_send = (((my_pe_in_team + 1 - iter + 2 * PE_size) % PE_size) * chunk_size);
      off_recv = (((my_pe_in_team - iter + 2 * PE_size) % PE_size) * chunk_size);

      internal_putmem_wg(reinterpret_cast<void *>(&pWrk[off_send]),
                    reinterpret_cast<void *>(&dst[off_send + off_seg]),
                    chunk_size * sizeof(T), send_pe);

      if (is_thread_zero_in_block()) {
        wait_val = seg + 100;
        fence(send_pe);
        internal_putmem(&pSync[iter], &wait_val, sizeof(*pSync), send_pe);
        wait_until(&pSync[iter], ROCSHMEM_CMP_EQ, wait_val);
      }
      __syncthreads();
      ipc_compute_reduce<T, Op>(&pWrk[off_recv], &dst[off_seg + off_recv],
                            chunk_size, wg_id, wg_size);
    }

    // Loop 2 in the example above
    for (int iter = PE_size - 1; iter < 2 * PE_size - 2; iter++) {
      off_send = (((my_pe_in_team + 1 - iter + 2 * PE_size) % PE_size) * chunk_size);
      putmem_nbi_wg(reinterpret_cast<void *>(&dst[off_send + off_seg]),
                    reinterpret_cast<void *>(&dst[off_send + off_seg]),
                    chunk_size * sizeof(T), send_pe);

      if (is_thread_zero_in_block()) {
        wait_val = seg + 10;
        fence(send_pe);
        internal_putmem(&pSync[iter], &wait_val, sizeof(*pSync), send_pe);
        wait_until(&pSync[iter], ROCSHMEM_CMP_EQ, wait_val);
      }
      __syncthreads();
    }
  }

  for (int i = wg_id; i < 2 * PE_size - 2; i += wg_size) {
    pSync[i] = ROCSHMEM_SYNC_VALUE;
  }
  __syncthreads();
}

template <typename T, ROCSHMEM_OP Op>
__device__ int IPCContext::reduce_das(rocshmem_team_t team, T *dest,
                                  const T *source, int nreduce) {
  IPCTeam *team_obj = reinterpret_cast<IPCTeam *>(team);

  int PE_size = team_obj->tinfo_wrt_world->size;

  size_t direct_pWrk = PE_size * nreduce;
  size_t direct_pSync = PE_size;
  size_t ring_pSync = 2 * PE_size;
  size_t provided_pWrk = ROCSHMEM_REDUCE_MIN_WRKDATA_SIZE;
  size_t provided_pSync = ROCSHMEM_REDUCE_SYNC_SIZE;

  // Messages above DIRECT_MAX use ring. default 8192 (8KB).
  constexpr int DIRECT_MAX_NELEMS = 8192;

  bool use_direct = (provided_pWrk >= direct_pWrk) &&
                    (provided_pSync >= direct_pSync) &&
                    (nreduce <= DIRECT_MAX_NELEMS);
  const bool use_reduce_scatter_allgather =
      !use_direct && (nreduce % PE_size == 0) &&
      (nreduce / PE_size >= 1024);

  if (use_reduce_scatter_allgather) {
    const int block_count = nreduce / PE_size;
    const int team_rank = team_obj->my_pe;
    T *local_block = dest + team_rank * block_count;
    int status = reduce_scatter_wg<T, Op>(team, local_block, source,
                                          block_count);
    if (status != ROCSHMEM_SUCCESS) {
      return status;
    }

    // Specialized allgather for the allreduce result. local_block already
    // occupies its final slot in dest, so avoid a redundant self-copy and
    // send the complete block once to each remote PE.
    const int wave_id = get_flat_block_id() / WF_SIZE;
    const int num_waves = (get_flat_block_size() + WF_SIZE - 1) / WF_SIZE;
    for (int remote_rank = wave_id; remote_rank < PE_size; remote_rank += num_waves) {
      if (remote_rank != team_rank) {
        const int remote_pe = team_obj->get_pe_in_world(remote_rank);
        put_nbi_wave(dest + team_rank * block_count, local_block, block_count, remote_pe);
      }
    }
    __syncthreads();
    internal_sync_wg(constmem.my_pe,
                     team_obj->tinfo_wrt_world->pe_start,
                     team_obj->tinfo_wrt_world->stride, PE_size,
                     team_obj->alltoall_pSync);
    return ROCSHMEM_SUCCESS;
  }

  if (use_direct) {
    internal_direct_allreduce<T, Op>(dest, source, nreduce, team_obj);
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
                                       seg_size, chunk_size);
      }
      if (n_seg_up > n_seg) {
        T *p_dst = (dest + (n_seg * seg_size));
        const T *p_src = (source + (n_seg * seg_size));
        int p_count = nreduce - (n_seg * seg_size);
        int p_chunk = p_count / PE_size;
        if (p_chunk > 0) {
          internal_ring_allreduce<T, Op>(p_dst, p_src,
                                         (p_chunk * PE_size), team_obj, 1,
                                         (p_chunk * PE_size), p_chunk);
        }

        if ((p_chunk * PE_size) < p_count) {
          // Final elements need to use direct_allreduce
          p_count -= (p_chunk * PE_size);
          p_dst += (p_chunk * PE_size);
          const T *p_src2 = p_src + (p_chunk * PE_size);

          internal_direct_allreduce<T, Op>(p_dst, p_src2, p_count, team_obj);
        }
      }
    } else {
      LOGD_WARN("Unsupported reduction size for IPC conduit.");
      return ROCSHMEM_ERROR;
    }
  }
  barrier_wg(team);
  return ROCSHMEM_SUCCESS;
}

/* Reduce-scatter through wave-parallel peer PUTs into team scratch. */
template <typename T, ROCSHMEM_OP Op>
__device__ int IPCContext::reduce_scatter_wg_das(rocshmem_team_t team, T *dest,
                                             const T *source, int nreduce,
                                             int elem_offset, int elem_count) {
  IPCTeam *team_obj = reinterpret_cast<IPCTeam *>(team);

  const int pe_size = team_obj->tinfo_wrt_world->size;
  const int pe_start = team_obj->tinfo_wrt_world->pe_start;
  const int stride = team_obj->tinfo_wrt_world->stride;
  const int team_rank = (constmem.my_pe - pe_start) / stride;

  T *pWrk = reinterpret_cast<T *>(team_obj->pWrk);
  long *pSync = team_obj->reduce_pSync;
  const int thread_id = get_flat_block_id();
  const int block_size = get_flat_block_size();
  const int scratch_elems = static_cast<int>(ROCSHMEM_REDUCE_MIN_WRKDATA_SIZE * sizeof(double) / sizeof(T));
  const int scratch_slots = max(1, pe_size - 1);
  const int chunk_size = max(1, scratch_elems / scratch_slots);
  const int wave_id = thread_id / WF_SIZE;
  const int num_waves = (block_size + WF_SIZE - 1) / WF_SIZE;

  // Each WG carries its own team (independent pWrk / reduce_pSync pool slot)
  // and its own dest/source buffers, so all WGs run the collective in
  // parallel without any cross-WG serialisation.

  if (elem_count < 0) elem_count = nreduce;
  const int num_chunks = (elem_count + chunk_size - 1) / chunk_size;

  // Reserve one generation per chunk. Reusing a generation within a call lets
  // a previous chunk's data/ack flag satisfy the next chunk's wait early.
  if (is_thread_zero_in_block()) {
    pSync[2 * pe_size] += num_chunks;
  }
  __syncthreads();
  const int64_t first_flag = pSync[2 * pe_size] - num_chunks + 1;

  const int range_end = elem_offset + elem_count;
  int chunk_index = 0;
  for (int offset = elem_offset; offset < range_end;
       offset += chunk_size, ++chunk_index) {
    const int64_t flag_val = first_flag + chunk_index;
    const int count = min(chunk_size, range_end - offset);

    // ------------------------------------------------------------------
    // Phase 1: send block-[remote_rank] to every peer.  Wave-level
    // parallelism keeps all peer links busy at the same time.
    // ------------------------------------------------------------------
    if (pe_size - 1 <= num_waves) {
      if (wave_id < pe_size - 1) {
        const int remote_rank = wave_id + (wave_id >= team_rank);
        const int pe = pe_start + remote_rank * stride;
        const int scratch_slot = team_rank - (team_rank > remote_rank);
        T *scratch = pWrk + scratch_slot * chunk_size;
        const T *send = source + remote_rank * nreduce + offset;
        internal_putmem_wave(scratch, send, count * sizeof(T), pe);
        if (is_thread_zero_in_wave()) {
          fence(pe);
          internal_putmem(pSync + team_rank, &flag_val, sizeof(*pSync), pe);
        }
      }
    } else {
      for (int pe = pe_start; pe < pe_start + stride * pe_size; pe += stride) {
        if (pe != constmem.my_pe) {
          const int remote_rank = (pe - pe_start) / stride;
          const int scratch_slot = team_rank - (team_rank > remote_rank);
          internal_putmem_wg(pWrk + scratch_slot * chunk_size,
                             source + remote_rank * nreduce + offset,
                             count * sizeof(T), pe);
          if (is_thread_zero_in_block()) {
            fence(pe);
            internal_putmem(pSync + team_rank, &flag_val, sizeof(*pSync), pe);
          }
        }
      }
    }
    __syncthreads();

    // ------------------------------------------------------------------
    // Phase 2: wait for all peers to deliver their data to my scratch.
    // ------------------------------------------------------------------
    if (pe_size - 1 <= num_waves) {
      if (wave_id < pe_size - 1) {
        const int remote_rank = wave_id + (wave_id >= team_rank);
        if (is_thread_zero_in_wave()) {
          wait_until(pSync + remote_rank, ROCSHMEM_CMP_EQ, flag_val);
        }
      }
      __syncthreads();
    } else {
      if (is_thread_zero_in_block()) {
        for (int remote_rank = 0; remote_rank < pe_size; remote_rank++) {
          if (remote_rank != team_rank) {
            wait_until(pSync + remote_rank, ROCSHMEM_CMP_EQ, flag_val);
          }
        }
      }
      __syncthreads();
    }

    // ------------------------------------------------------------------
    // Phase 3: fuse local + peer scratch contributions into dest.
    // Use vectorised load/store to reduce loop iterations, while
    // OpWrap<Op>::Calc keeps the reduction generic for all Op types.
    // ------------------------------------------------------------------
    // Vector type selector: float->float4 (128-bit), double->double2 (128-bit), int->int4 (128-bit), uint->uint4 (128-bit)
    using VecT = std::conditional_t<std::is_same_v<T, float>, float4,
                 std::conditional_t<std::is_same_v<T, double>, double2,
                 std::conditional_t<std::is_same_v<T, int>, int4,
                 std::conditional_t<std::is_same_v<T, unsigned int>, uint4, T>>>>;

    static_assert(sizeof(VecT) % sizeof(T) == 0, "Vector type must be a multiple of scalar type");
    constexpr int VEC_WIDTH = sizeof(VecT) / sizeof(T);
    constexpr bool IS_VECTORIZABLE = !std::is_same_v<VecT, T>;
    bool vectorized_ok = false;

    if constexpr (IS_VECTORIZABLE) {
      if ((nreduce % VEC_WIDTH) == 0 && (offset % VEC_WIDTH) == 0 && (count % VEC_WIDTH) == 0) {
        const int vectors = count / VEC_WIDTH;
        const VecT *local = reinterpret_cast<const VecT *>(source + team_rank * nreduce + offset);
        VecT *output = reinterpret_cast<VecT *>(dest + offset);
        for (int i = thread_id; i < vectors; i += block_size) {
          VecT result = local[i];
          for (int remote_rank = 0; remote_rank < pe_size; remote_rank++) {
            if (remote_rank != team_rank) {
              const int scratch_slot = remote_rank - (remote_rank > team_rank);
              const VecT value = reinterpret_cast<const VecT *>(pWrk + scratch_slot * chunk_size)[i];
              T *res_elem = reinterpret_cast<T *>(&result);
              const T *val_elem = reinterpret_cast<const T *>(&value);
              for (int v = 0; v < VEC_WIDTH; ++v) {
                OpWrap<Op>::Calc(const_cast<T *>(val_elem + v), res_elem + v, 0);
              }
            }
          }
          output[i] = result;
        }
        vectorized_ok = true;
      }
    }

    if (!vectorized_ok) {
      for (int i = thread_id; i < count; i += block_size) {
        T result = source[team_rank * nreduce + offset + i];
        for (int remote_rank = 0; remote_rank < pe_size; remote_rank++) {
          if (remote_rank != team_rank) {
            const int scratch_slot =
                remote_rank - (remote_rank > team_rank);
            OpWrap<Op>::Calc(pWrk + scratch_slot * chunk_size + i, &result, 0);
          }
        }
        dest[offset + i] = result;
      }
    }
    __syncthreads();

    // ------------------------------------------------------------------
    // Phase 4: point-to-point completion ACK. Receiver rank R notifies every
    // sender through slot [pe_size + R] after consuming its scratch payload.
    // A sender waits for all receiver ACKs before reusing pWrk in the next
    // chunk/call. This provides the safety of a team barrier without forcing
    // unrelated peers through the barrier protocol.
    // ------------------------------------------------------------------
    if (is_thread_zero_in_block()) {
      for (int remote_rank = 0; remote_rank < pe_size; ++remote_rank) {
        if (remote_rank == team_rank) continue;
        const int pe = pe_start + remote_rank * stride;
        internal_putmem(pSync + pe_size + team_rank, &flag_val,
                        sizeof(*pSync), pe);
      }
      for (int remote_rank = 0; remote_rank < pe_size; ++remote_rank) {
        if (remote_rank == team_rank) continue;
        wait_until(pSync + pe_size + remote_rank, ROCSHMEM_CMP_EQ, flag_val);
      }
    }
    __syncthreads();
  }

  return ROCSHMEM_SUCCESS;
}

template <typename T>
__device__ void IPCContext::alltoall_das(rocshmem_team_t team, T *dst,
                                     const T *src, int nelems,
                                     int elem_offset, int elem_count) {
  // Normalize: elem_count == -1 means "process the full [0, nelems) range".
  if (elem_count < 0) elem_count = nelems;
#if defined(USE_SDMA)
  if (sizeof(T) * elem_count < 512 || ipcImpl_.sdmaImpl_.sdmaEnabled)
#else
  if (sizeof(T) * elem_count < 512)
#endif
    alltoall_linear_thread_puts(team, dst, src, nelems, elem_offset, elem_count);
  else
    alltoall_linear(team, dst, src, nelems, elem_offset, elem_count);
}

template <typename T>
__device__ void IPCContext::alltoallv_das([[maybe_unused]] rocshmem_team_t team,
                                      [[maybe_unused]] T *dest, [[maybe_unused]] const size_t dest_nelems[],
                                      [[maybe_unused]] const size_t dest_displs[],
                                      [[maybe_unused]] T *source, [[maybe_unused]] const size_t source_nelems[],
                                      [[maybe_unused]] const size_t source_displs[],
                                      int wg_id, int wg_count) {
  IPCTeam *team_obj = reinterpret_cast<IPCTeam *>(team);
  const int pe_size = team_obj->num_pes;
  const int my_rank = team_obj->my_pe;
  const int wave_id = get_flat_block_id() / WF_SIZE;
  const int num_waves = (get_flat_block_size() + WF_SIZE - 1) / WF_SIZE;

  // Uniform alltoallv is common in communication benchmarks and frameworks;
  // route it through the vectorized alltoall path. Nonuniform counts and
  // displacements continue through the fully general path below.
  const size_t uniform_count = source_nelems[0];
  bool uniform = uniform_count == dest_nelems[0];
  for (int rank = 0; rank < pe_size && uniform; rank++) {
    uniform = source_nelems[rank] == uniform_count &&
              dest_nelems[rank] == uniform_count &&
              source_displs[rank] == static_cast<size_t>(rank) * uniform_count &&
              dest_displs[rank] == static_cast<size_t>(rank) * uniform_count;
  }
  if (uniform) {
    // Uniform alltoallv is exactly alltoall. Split its per-peer element range
    // across duplicate teams so single- and multi-WG launches share the same
    // optimized alltoall implementation.
    const int total = static_cast<int>(uniform_count);
    size_t elem_offset = 0;
    size_t elem_count = 0;
    split_wg_range(uniform_count, wg_id, wg_count, &elem_offset, &elem_count);
    alltoall(team, dest, source, total, elem_offset, elem_count);
    return;
  }

  // Every WG runs the same alltoallv algorithm over all peers, restricted to
  // its own slice of each peer's element range.  wg_id=0,wg_count=1 naturally
  // covers the complete single-WG operation.
  for (int remote_rank = wave_id; remote_rank < pe_size;
       remote_rank += num_waves) {
    const size_t total = source_nelems[remote_rank];
    size_t elem_offset = 0;
    size_t elem_count = 0;
    split_wg_range(total, wg_id, wg_count, &elem_offset, &elem_count);
    if (elem_count == 0) continue;
    const int pe = team_obj->get_pe_in_world(remote_rank);
    put_nbi_wave(dest + dest_displs[my_rank] + elem_offset,
                 source + source_displs[remote_rank] + elem_offset,
                 elem_count, pe);
  }

  internal_sync_wg(constmem.my_pe,
                   team_obj->tinfo_wrt_world->pe_start,
                   team_obj->tinfo_wrt_world->stride, pe_size,
                   team_obj->alltoall_pSync);
}

template <typename T>
__device__ void IPCContext::alltoall_linear_das(rocshmem_team_t team, T *dst,
                                            const T *src, int nelems,
                                            int elem_offset, int elem_count) {
  IPCTeam *team_obj = reinterpret_cast<IPCTeam *>(team);

  int pe_start = team_obj->tinfo_wrt_world->pe_start;
  int pe_size = team_obj->num_pes;
  int stride = team_obj->tinfo_wrt_world->stride;
  long *pSync = team_obj->alltoall_pSync;
  int my_pe_in_team = team_obj->my_pe;

  // Normalize: elem_count == -1 means "process the full [0, nelems) range".
  if (elem_count < 0) elem_count = nelems;

  // pipeline chunking for large messages
  constexpr int CHUNK_NELEMS = ROCSHMEM_CHUNK_BYTES / sizeof(T);
  const bool use_pipeline = (elem_count > CHUNK_NELEMS);

  int wf_id = get_flat_block_id() / WF_SIZE;
  int wf_count = (get_flat_block_size() + WF_SIZE - 1) / WF_SIZE;

  if (!use_pipeline) {
    for (int j = wf_id; j < pe_size; j += wf_count) {
      int dest_pe = team_obj->get_pe_in_world(j);
      put_nbi_wave(&dst[my_pe_in_team * nelems + elem_offset],
                   &src[j * nelems + elem_offset], elem_count, dest_pe);
    }
  } else {
    for (int c_start = 0; c_start < elem_count; c_start += CHUNK_NELEMS) {
      int c_end = min(c_start + CHUNK_NELEMS, elem_count);
      int c_nelems = c_end - c_start;

      for (int j = wf_id; j < pe_size; j += wf_count) {
        int dest_pe = team_obj->get_pe_in_world(j);
        put_nbi_wave(&dst[my_pe_in_team * nelems + elem_offset + c_start],
                     &src[j * nelems + elem_offset + c_start],
                     c_nelems, dest_pe);
      }
    }
  }

  /* internal_sync_wg has threadfence_system */
  //  if (is_thread_zero_in_block()) {
  //    quiet();
  //  }

  // wait until everyone has obtained their designated data
  internal_sync_wg(constmem.my_pe, pe_start, stride, pe_size, pSync);
}

template <typename T>
__device__ void IPCContext::alltoall_linear_thread_puts_das(rocshmem_team_t team,
    T *dst, const T *src, int nelems, int elem_offset, int elem_count) {
  IPCTeam *team_obj = reinterpret_cast<IPCTeam *>(team);

  int pe_size = team_obj->num_pes;
  long *pSync = team_obj->alltoall_pSync;
  int my_pe_in_team = team_obj->my_pe;
  size_t alltoall_pSync_offset = (team_obj->alltoall_sequence_number % 2) * pe_size;

  // Normalize: elem_count == -1 means "process the full [0, nelems) range".
  if (elem_count < 0) elem_count = nelems;

  int tid = get_flat_block_id();
  int step_size = min(get_flat_block_size(), WF_SIZE);
//  printf("my_pe=%d\ttid=%d, pSync=%p, tnpes=%d, tmype=%d, offset=%lu, step=%d\n", my_pe_in_team, tid, pSync, pe_size, my_pe_in_team, alltoall_pSync_offset, step_size);

  // Have each PE put their designated data to the other PEs
  for (int j = tid; j < pe_size; j += step_size) {
    int dest_pe = team_obj->get_pe_in_world(j);
    put_nbi(
      &dst[my_pe_in_team * nelems + elem_offset],
      &src[j * nelems + elem_offset], elem_count, dest_pe);
  }
  for (int j = tid; j < pe_size; j += step_size) {
    int dest_pe = team_obj->get_pe_in_world(j);
    fence(dest_pe);
    ptrdiff_t L_offset = reinterpret_cast<char*>(&pSync[alltoall_pSync_offset + my_pe_in_team]) - wrk_sync_pool_bases_[constmem.my_pe];
    ipcImpl_.ipcAMOAdd(reinterpret_cast<long*>(wrk_sync_pool_bases_[dest_pe] + L_offset), 1L);
  }

  // wait until everyone has obtained their designated data
  for (int j = tid; j < pe_size; j+= step_size) {
    int dest_pe = team_obj->get_pe_in_world(j);

    long *sync_flag = &pSync[alltoall_pSync_offset + dest_pe];
    while (uncached_load(sync_flag) != 1) { }

    //quiet(dest_pe);// needed to quiet add when it is nbi in gda, it is not nbi in ipc

    pSync[alltoall_pSync_offset + dest_pe] = ROCSHMEM_SYNC_VALUE;
  }

  if (is_thread_zero_in_block()) {
    team_obj->alltoall_sequence_number++;
  }

  __syncthreads();
}

template <typename T>
__device__ void IPCContext::fcollect_linear_das(rocshmem_team_t team, T *dst,
                                            const T *src, int nelems,
                                            int elem_offset, int elem_count) {
  IPCTeam *team_obj = reinterpret_cast<IPCTeam *>(team);

  int pe_start = team_obj->tinfo_wrt_world->pe_start;
  int pe_size = team_obj->num_pes;
  int stride = team_obj->tinfo_wrt_world->stride;
  long *pSync = team_obj->alltoall_pSync;
  int my_pe_in_team = team_obj->my_pe;

  if (elem_count < 0) elem_count = nelems;
  // pipeline chunking for large messages
  constexpr int CHUNK_NELEMS = ROCSHMEM_CHUNK_BYTES / sizeof(T);
  const bool use_pipeline = (elem_count > CHUNK_NELEMS);

  int wf_id = get_flat_block_id() / WF_SIZE;
  int wf_count = (get_flat_block_size() + WF_SIZE - 1) / WF_SIZE;

  if (!use_pipeline) {
    for (int j = wf_id; j < pe_size; j += wf_count) {
      int dest_pe = team_obj->get_pe_in_world(j);
      put_nbi_wave(&dst[my_pe_in_team * nelems + elem_offset],
                   src + elem_offset, elem_count, dest_pe);
    }
  } else {
    for (int c_start = 0; c_start < elem_count; c_start += CHUNK_NELEMS) {
      int c_end = min(c_start + CHUNK_NELEMS, elem_count);
      int c_nelems = c_end - c_start;

      for (int j = wf_id; j < pe_size; j += wf_count) {
        int dest_pe = team_obj->get_pe_in_world(j);
        put_nbi_wave(&dst[my_pe_in_team * nelems + elem_offset + c_start],
                     &src[elem_offset + c_start],
                     c_nelems, dest_pe);
      }
    }
  }

  // wait until everyone has obtained their designated data
  internal_sync_wg(constmem.my_pe, pe_start, stride, pe_size, pSync);
}

}
#endif  // defined(USE_DAS)

#endif  // LIBRARY_SRC_IPC_CONTEXT_TMPL_DEVICE_DAS_HPP_
