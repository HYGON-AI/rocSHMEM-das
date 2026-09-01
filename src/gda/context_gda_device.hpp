/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef LIBRARY_SRC_GDA_CONTEXT_DEVICE_HPP_
#define LIBRARY_SRC_GDA_CONTEXT_DEVICE_HPP_

#include "context.hpp"
#include "team.hpp"
#include "queue_pair.hpp"
#include "constmem.hpp"
#include "gda/gda_symm_table.hpp"

namespace rocshmem {

class QueuePair;

class GDAContext : public Context {
 public:

#if defined(USE_DAS)
  // DAS implementations called by the original collective entry points.
  __device__ void internal_get_broadcastmem_wg_das(
      void *dst, const void *src, int nelems, int pe_root, ActiveWFInfo &wf_info);

  template <typename T>
  __device__ void alltoall_wg_das(rocshmem_team_t team, T *dest, const T *source,
                           int nelems, int elem_offset = 0,
                           int elem_count = -1);

  template <typename T>
  __device__ void alltoall_linear_wg_das(rocshmem_team_t team, T *dest,
    const T *source, int nelems, int elem_offset = 0, int elem_count = -1);

  template <typename T>
  __device__ void alltoall_linear_thread_puts_das(rocshmem_team_t team, T *dest,
                                              const T *source, int nelems,
                                              int elem_offset = 0,
                                              int elem_count = -1);

  template <typename T>
  __device__ void alltoallv_das(rocshmem_team_t team,
                            T *dest, const size_t dest_nelems[],
                            const size_t dest_displs[],
                            T *source, const size_t source_nelems[],
                            const size_t source_displs[], int wg_id = 0,
                            int wg_count = 1);

  template <typename T>
  __device__ void alltoallv_copy_das(rocshmem_team_t team,
                                 T *dest, const size_t dest_nelems[],
                                 const size_t dest_displs[],
                                 T *source, const size_t source_nelems[],
                                 const size_t source_displs[], int wg_id,
                                 int wg_count);

  template <typename T>
  __device__ void alltoallv_get_das(rocshmem_team_t team,
                                T *dest, const size_t dest_nelems[],
                                const size_t dest_displs[],
                                T *source, const size_t source_nelems[],
                                const size_t source_displs[], int wg_id,
                                int wg_count);

  __device__ void broadcastmem_wg_das(rocshmem_team_t team, void *dest, const void* source,
                                  int nelement, int PE_root);

  template <typename T>
  __device__ void fcollect_linear_das(rocshmem_team_t team, T *dest,
      const T *source, int nelems, int elem_offset, int elem_count);

  __device__ void internal_broadcastmem_wg_das(void *dest, const void *source, int nelems,
      int pe_root, int pe_start, int stride, int pe_size, long *p_sync);

  template <typename T, ROCSHMEM_OP Op>
  __device__ void internal_ring_allreduce_das(T *dst, const T *src, int nelems,
      GDATeam *team_obj, int n_seg, int seg_size, int chunk_size,
      ActiveWFInfo &wf_info);

  template <typename T, ROCSHMEM_OP Op>
  __device__ void internal_proxy_allreduce(T *dst, const T *src, int nelems,
      GDATeam *team_obj, ActiveWFInfo &wf_info);

  template <typename T, ROCSHMEM_OP Op>
  __device__ void internal_fused_ipc_reduce(T *dst, const T *src,
      int source_offset, int count, int local_size, int local_slot,
      int wg_id, int wg_size);

  template <typename T, ROCSHMEM_OP Op>
  __device__ int reduce_das(rocshmem_team_t team, T *dest, const T *source, int nreduce);

  template <typename T, ROCSHMEM_OP Op>
  __device__ int reduce_scatter_wg_das(rocshmem_team_t team, T *dest, const T *source,
                                   int nreduce, int elem_offset = 0,
                                   int elem_count = -1);

  template <typename T, ROCSHMEM_OP Op>
  __device__ int hybrid_reduce_scatter_wg(GDATeam *team_obj, T *dest, const T *source,
                                          int nreduce, int elem_offset,
                                          int elem_count, int PE_start, int stride,
                                          int PE_size, int team_rank);

#endif  // USE_DAS

  __host__ GDAContext(Backend *b, unsigned int ctx_id);

  __host__ ~GDAContext();

  __device__ GDAContext(Backend *b, unsigned int ctx_id); //TODO is this used?

  __device__ void putmem(void *dest, const void *source, size_t nelems, int pe);

  __device__ void getmem(void *dest, const void *source, size_t nelems, int pe);

  __device__ void putmem_nbi(void *dest, const void *source, size_t nelems,
                             int pe);

  __device__ void getmem_nbi(void *dest, const void *source, size_t nelems,
                             int pe);

  __device__ void fence();

  __device__ void fence(int pe);

  __device__ void quiet();

  __device__ void pe_quiet(size_t pe);

  __device__ void qp_quiet(size_t qp_idx);

  __device__ void *shmem_ptr(const void *dest, int pe);

  __device__ void barrier_all();

  __device__ void barrier_all_wave();

  __device__ void barrier_all_wg();

  __device__ void barrier(rocshmem_team_t team);

  __device__ void barrier_wave(rocshmem_team_t team);

  __device__ void barrier_wg(rocshmem_team_t team);

  __device__ void sync_all();

  __device__ void sync_all_wave();

  __device__ void sync_all_wg();

  __device__ void sync(rocshmem_team_t team);

  __device__ void sync_wave(rocshmem_team_t team);

  __device__ void sync_wg(rocshmem_team_t team);

  template <typename T>
  __device__ void p(T *dest, T value, int pe);

  template <typename T>
  __device__ void put(T *dest, const T *source, size_t nelems, int pe);

  template <typename T>
  __device__ void put_nbi(T *dest, const T *source, size_t nelems, int pe);

  template <typename T>
  __device__ T g(const T *source, int pe);

  template <typename T>
  __device__ void get(T *dest, const T *source, size_t nelems, int pe);

  template <typename T>
  __device__ void get_nbi(T *dest, const T *source, size_t nelems, int pe);

  // Atomic operations
  template <typename T>
  __device__ void amo_add(void *dst, T value, int pe);

  template <typename T>
  __device__ void amo_add_dp(void *dst, T value, int qp_idx, int pe);

  template <typename T>
  __device__ void amo_set(void *dst, T value, int pe);

  template <typename T>
  __device__ T amo_swap(void *dst, T value, int pe);

  template <typename T>
  __device__ T amo_fetch_and(void *dst, T value, int pe);

  template <typename T>
  __device__ void amo_and(void *dst, T value, int pe);

  template <typename T>
  __device__ T amo_fetch_or(void *dst, T value, int pe);

  template <typename T>
  __device__ void amo_or(void *dst, T value, int pe);

  template <typename T>
  __device__ T amo_fetch_xor(void *dst, T value, int pe);

  template <typename T>
  __device__ void amo_xor(void *dst, T value, int pe);

  template <typename T>
  __device__ void amo_cas(void *dst, T value, T cond, int pe);

  template <typename T>
  __device__ T amo_fetch_add(void *dst, T value, int pe);

  template <typename T>
  __device__ T amo_fetch_cas(void *dst, T value, T cond, int pe);

  // Collectives
  template <typename T, ROCSHMEM_OP Op>
  __device__ int reduce_wg(rocshmem_team_t team, T *dest, const T *source, int nreduce);

  template <typename T, ROCSHMEM_OP Op>
  __device__ int reduce_scatter_wg(rocshmem_team_t team, T *dest, const T *source,
                                   int nreduce, int elem_offset = 0,
                                   int elem_count = -1);

  template <typename T, ROCSHMEM_OP Op>
  __device__ int reduce_wave(rocshmem_team_t team, T *dest, const T *source, int nreduce);
  
  template <typename T, ROCSHMEM_OP Op>
  __device__ int reduce_scatter_wave(rocshmem_team_t team, T *dest, const T *source,
                                     int nreduce);

  template <typename T>
  __device__ void broadcast_wg(rocshmem_team_t team, T *dest, const T *source,
                            int nelems, int pe_root);

  __device__ void broadcastmem_wg(rocshmem_team_t team, void *dest, const void* source,
                                  int nelems, int PE_root);

  template <typename T>
  __device__ int broadcast_wave(rocshmem_team_t team,
                                T *dest, const T* source, int nelems, int PE_root);

  __device__ int broadcastmem_wave(rocshmem_team_t team,
                                void *dest, const void* source, int nelems, int PE_root);

  template <typename T>
  __device__ void alltoall_wg(rocshmem_team_t team, T *dest, const T *source,
                           int nelems, int elem_offset = 0,
                           int elem_count = -1);

  __device__ void alltoallmem_wg(rocshmem_team_t team, void *dest, const void *source,
                           int nelems);

  template <typename T>
  __device__ void alltoallv(rocshmem_team_t team,
                            T *dest, const size_t dest_nelems[],
                            const size_t dest_displs[],
                            T *source, const size_t source_nelems[],
                            const size_t source_displs[], int wg_id = 0,
                            int wg_count = 1);

  template <typename T>
  __device__ void alltoallv_copy(rocshmem_team_t team,
                                 T *dest, const size_t dest_nelems[],
                                 const size_t dest_displs[],
                                 T *source, const size_t source_nelems[],
                                 const size_t source_displs[], int wg_id,
                                 int wg_count);

  template <typename T>
  __device__ void alltoallv_get(rocshmem_team_t team,
                                T *dest, const size_t dest_nelems[],
                                const size_t dest_displs[],
                                T *source, const size_t source_nelems[],
                                const size_t source_displs[], int wg_id,
                                int wg_count);

  template <typename T>
  __device__ int alltoall_wave(rocshmem_team_t team, T* dest, 
                                  const T* source, int nelems);

  __device__ int alltoallmem_wave(rocshmem_team_t team, void* dest, 
                                  const void* source, int nelems);

  template <typename T>
  __device__ void fcollect_wg(rocshmem_team_t team, T *dest, const T *source,
                           int nelems, int elem_offset = 0,
                           int elem_count = -1);

  __device__ void fcollectmem_wg(rocshmem_team_t team, void *dest, const void *source,
                           int nelems);

  template <typename T>
  __device__ int fcollect_wave(rocshmem_team_t team, T *dest, const T *source,
                           int nelems);

  __device__ int fcollectmem_wave(rocshmem_team_t team, void *dest, const void *source,
                           int nelems);

  // Block/wave functions
  __device__ void putmem_wg(void *dest, const void *source, size_t nelems,
                            int pe);

  __device__ void getmem_wg(void *dest, const void *source, size_t nelems,
                            int pe);

  __device__ void putmem_nbi_wg(void *dest, const void *source, size_t nelems,
                                int pe);

  __device__ void getmem_nbi_wg(void *dest, const void *source, size_t nelems,
                                int pe);

  __device__ void putmem_wave(void *dest, const void *source, size_t nelems,
                              int pe);

  __device__ void getmem_wave(void *dest, const void *source, size_t nelems,
                              int pe);

  __device__ void putmem_nbi_wave(void *dest, const void *source, size_t nelems,
                                  int pe);

  __device__ void putmem_nbi_wave_dp(void *dest, const void *source, size_t nelems,
                                  int qp_idx, int pe);
                                  
  __device__ void getmem_nbi_wave(void *dest, const void *source, size_t nelems,
                                  int pe);

  template <typename T>
  __device__ void put_wg(T *dest, const T *source, size_t nelems, int pe);

  template <typename T>
  __device__ void put_nbi_wg(T *dest, const T *source, size_t nelems, int pe);

  template <typename T>
  __device__ void put_wave(T *dest, const T *source, size_t nelems, int pe);

  template <typename T>
  __device__ void put_nbi_wave(T *dest, const T *source, size_t nelems, int pe);

  // add multiqp func
  template <typename T>
  __device__ void put_nbi_wave_dp(T *dest, const T *source, size_t nelems, int qp_idx, int pe);

  template <typename T>
  __device__ void get_wg(T *dest, const T *source, size_t nelems, int pe);

  template <typename T>
  __device__ void get_nbi_wg(T *dest, const T *source, size_t nelems, int pe);


  template <typename T>
  __device__ void get_wave(T *dest, const T *source, size_t nelems, int pe);

  template <typename T>
  __device__ void get_nbi_wave(T *dest, const T *source, size_t nelems, int pe);

#define GDA_CONTEXT_PUT_SIGNAL_DEC(SUFFIX)                                               \
  template <typename T>                                                                  \
  __device__ void put_signal##SUFFIX(T *dest, const T *source, size_t nelems,            \
                                     uint64_t *sig_addr, uint64_t signal, int sig_op,    \
                                     int pe);                                            \
                                                                                         \
  __device__ void putmem_signal##SUFFIX(void *dest, const void *source, size_t nelems,   \
                                        uint64_t *sig_addr, uint64_t signal, int sig_op, \
                                        int pe);

  GDA_CONTEXT_PUT_SIGNAL_DEC()
  GDA_CONTEXT_PUT_SIGNAL_DEC(_wg)
  GDA_CONTEXT_PUT_SIGNAL_DEC(_wave)
  GDA_CONTEXT_PUT_SIGNAL_DEC(_nbi)
  GDA_CONTEXT_PUT_SIGNAL_DEC(_nbi_wg)
  GDA_CONTEXT_PUT_SIGNAL_DEC(_nbi_wave)

  __device__ uint64_t signal_fetch(const uint64_t *sig_addr);
  __device__ uint64_t signal_fetch_wg(const uint64_t *sig_addr);
  __device__ uint64_t signal_fetch_wave(const uint64_t *sig_addr);

  /**
   * @brief get local pointer to remote data
   */
  __device__ uint64_t get_p2p_ptr(void *dest, int rank, int dst_rank);

 private:

  static constexpr size_t kHierarchicalMinBytes = 256 * 1024;  // 256 KB: minimum size for hierarchical/proxy algorithms
  struct CollectiveChunkPolicy {
    size_t max_bytes;
    // A nonzero budget caps aggregate per-peer traffic across cross-node WGs.
    // Zero applies max_bytes unconditionally, e.g. to scratch-buffer algorithms.
    size_t concurrent_bytes;
  };
  static constexpr CollectiveChunkPolicy kAlltoallChunks{16 * 1024, 256 * 1024};
  static constexpr CollectiveChunkPolicy kFcollectChunks{16 * 1024, 128 * 1024};
  static constexpr CollectiveChunkPolicy kReduceScatterChunks{64 * 1024, 0};

  template <typename T>
  __device__ __forceinline__ static int collective_chunk_elems(
      int capacity, CollectiveChunkPolicy policy, bool inter_node = true);

  //internal functions used by collective operations
  __device__ void internal_broadcastmem_wg(void *dest, const void *source, int nelems,
      int pe_root, int pe_start, int stride, int pe_size, long *p_sync);  // NOLINT(runtime/int)

  __device__ void internal_put_broadcastmem_wg(void *dst, const void *src, int nelems,
      int pe_root, int PE_start, int logPE_stride, int PE_size,
      ActiveWFInfo &wf_info);  // NOLINT(runtime/int)

  __device__ void internal_get_broadcastmem_wg(void *dst, const void *src, int nelems,
      int pe_root, ActiveWFInfo &wf_info);  // NOLINT(runtime/int)

  template <typename T>
  __device__ void internal_broadcast_wave(T *dest, const T *source, int nelems,
      int pe_root, int pe_start, int stride, int pe_size, long *p_sync);  // NOLINT(runtime/int)

  __device__ bool internal_proxy_broadcast(void *dst, const void *src, int nelems,
      int pe_root, int pe_start, int stride, int pe_size, long *p_sync,
      ActiveWFInfo &wf_info);  // NOLINT(runtime/int)

  template <typename T>
  __device__ void internal_put_broadcast_wave(T *dst, const T *src, int nelems,
      int pe_root, int PE_start, int logPE_stride, int PE_size,
      ActiveWFInfo &wf_info);  // NOLINT(runtime/int)

  template <typename T>
  __device__ void internal_get_broadcast_wave(T *dst, const T *src, int nelems,
      int pe_root, ActiveWFInfo &wf_info);  // NOLINT(runtime/int)

  __device__ void internal_broadcastmem_wave(void *dst, const void *src,
    int nelems, int pe_root, int pe_start, int stride, int pe_size,
    long *p_sync);

  __device__ void internal_get_broadcastmem_wave(void *dst, const void *src,
    int nelems, int pe_root, ActiveWFInfo &wf_info);

  __device__ void internal_put_broadcastmem_wave(void *dst, const void *src,
    int nelems, int pe_root, int pe_start, int stride, int pe_size,
    ActiveWFInfo &wf_info);
      
  __device__ void fcollectmem_linear_wg(rocshmem_team_t team, void *dest,
      const void *source, int nelems);

  __device__ void fcollectmem_linear_wave(rocshmem_team_t team, void *dest,
      const void *source, int nelems);

  template <typename T>
  __device__ void alltoall_linear_wg(rocshmem_team_t team, T *dest,
    const T *source, int nelems, int elem_offset = 0, int elem_count = -1);

  __device__ void alltoallmem_linear_thread_puts_wg(rocshmem_team_t team, void *dest,
                                              const void *source, int nelems);

  __device__ __forceinline__ uint64_t get_next_qp_group(
      int qp_index, uint64_t pending_lanes);

  __device__ __forceinline__ void put_nbi_shared_qp(int qp_index, void *dest,
      const void *source, size_t length, bool ring_db);

  __device__ __forceinline__ void get_nbi_shared_qp(int qp_index, void *dest,
      const void *source, size_t length);

  __device__ __forceinline__ void atomic_nofetch_shared_qp(int qp_index, void *dest,
      int64_t value);

  __device__ void alltoallmem_linear_wave(rocshmem_team_t team, void *dst,
                                          const void *src, int nelems);

  __device__ void alltoallmem_linear_thread_puts_wave(rocshmem_team_t team,
    void *dst, const void *src, int nelems);

  __device__ void internal_sync(int pe, int PE_start, int stride, int PE_size,
      int64_t *pSync, ActiveWFInfo &wf_info);

  __device__ void internal_sync_wave(int pe, int PE_start, int stride,
      int PE_size, int64_t *pSync, ActiveWFInfo &wf_info);

  __device__ void internal_sync_wg(int pe, int PE_start, int stride,
    int PE_size, int64_t *pSync, ActiveWFInfo &wf_info);

  __device__ void internal_direct_barrier(int pe, int PE_start, int stride,
      int n_pes, int64_t *pSync, ActiveWFInfo &wf_info);

  __device__ void internal_direct_barrier_wg(int pe, int PE_start, int stride,
      int n_pes, int64_t *pSync, ActiveWFInfo &wf_info);

  __device__ void internal_atomic_barrier(int pe, int PE_start, int stride,
      int n_pes, int64_t *pSync, ActiveWFInfo &wf_info);

  template <typename T, ROCSHMEM_OP Op>
  __device__ void internal_direct_allreduce_wg(T *dst, const T *src, int nelems,
      GDATeam *team_obj, ActiveWFInfo &wf_info);

  template <typename T, ROCSHMEM_OP Op>
  __device__ void internal_direct_allreduce_wave(T *dst, const T *src, int nelems,
      GDATeam *team_obj, ActiveWFInfo &wf_info);

  /**
   * @brief Get the destination pointer for a given PE
   */
  __device__ char* get_remote_ptr(const void* addr, int pe);
  __device__ char* get_local_ptr(const void* addr, int pe);

  template <typename T, ROCSHMEM_OP Op>
  __device__ void internal_ring_allreduce_wg(T *dst, const T *src, int nelems,
      GDATeam *team_obj, int n_seg, int seg_size, int chunk_size,
      ActiveWFInfo &wf_info);

  template <typename T, ROCSHMEM_OP Op>
  __device__ void internal_ring_allreduce_wave(T *dst, const T *src, int nelems,
      GDATeam *team_obj, int n_seg, int seg_size, int chunk_size,
      ActiveWFInfo &wf_info);

  __device__ void internal_putmem(void *dest, const void *source,
      size_t nelems, int pe, int qp_index, ActiveWFInfo &wf_info);

  __device__ void internal_getmem(void *dest, const void *source,
      size_t nelems, int pe, int qp_index, ActiveWFInfo &wf_info);

  __device__ void internal_putmem_wg(void *dest, const void *source,
      size_t nelems, int pe, int qp_index, ActiveWFInfo &wf_info);

  __device__ void internal_getmem_wg(void *dest, const void *source,
      size_t nelems, int pe, int qp_index, ActiveWFInfo &wf_info);

  __device__ void internal_putmem_wave(void *dest, const void *source,
      size_t nelems, int pe, int qp_index, ActiveWFInfo &wf_info);

  __device__ void internal_getmem_wave(void *dest, const void *source,
      size_t nelems, int pe, int qp_index, ActiveWFInfo &wf_info);

  __device__ void internal_putmem_nbi(void *dest, const void *source,
      size_t nelems, int pe, int qp_index, ActiveWFInfo &wf_info);

  __device__ void internal_getmem_nbi(void *dest, const void *source,
      size_t nelems, int pe, int qp_index, ActiveWFInfo &wf_info);

  __device__ void internal_putmem_nbi_wg(void *dest, const void *source,
      size_t nelems, int pe, int qp_index, ActiveWFInfo &wf_info);

  __device__ void internal_getmem_nbi_wg(void *dest, const void *source,
      size_t nelems, int pe, int qp_index, ActiveWFInfo &wf_info);

  __device__ void internal_putmem_nbi_wave(void *dest, const void *source,
      size_t nelems, int pe, int qp_index, ActiveWFInfo &wf_info);

  __device__ void internal_getmem_nbi_wave(void *dest, const void *source,
      size_t nelems, int pe, int qp_index, ActiveWFInfo &wf_info);

  __device__ void tile_finish_put(int pe, int qp_index, ActiveWFInfo &wf_info);

  __device__ void tile_finish_get(int pe, int qp_index, ActiveWFInfo &wf_info);

  /**
   * @brief Post a single NBI put/get chunk from every participating lane.
   *
   * All lanes of the wave that reach the call share @p qp_index, so they post
   * through the wave-collective path (put_nbi/get_nbi with an ActiveWFInfo
   * built from the current execution mask). That path elects one lane to take
   * the SQ lock and reserve slots for the whole group, then each lane fills its
   * own WQE. Lanes that already fell out of a striped loop are simply absent
   * from the group, so unequal per-lane chunk counts are fine.
   *
   * These must not use the *_single posters: those take the SQ lock per lane
   * and advance the SQ producer non-atomically, which is only valid when every
   * active lane holds a *different* QP. With one QP per wave the lock holder
   * cannot make progress until its peers leave the spin loop, so the wave
   * deadlocks.
   *
   * @p pe must be wave-uniform here (it is, since it also selects the shared
   * QP), so the collective group spans every active lane.
   */
  __device__ void tile_put_chunk_nbi(char *dst, const char *src, size_t bytes,
                                     int pe, int qp_index);
  __device__ void tile_get_chunk_nbi(char *dst, const char *src, size_t bytes,
                                     int pe, int qp_index);
  __device__ int tile_qp_index_for_worker(int pe, int worker_id,
                                          int worker_count);
  __device__ void tile_quiet_gda_workers(int pe, int worker_id, int worker_count,
                                         int wave_qp_index);
  __device__ void tile_put_contig_slices_nbi(char *dst, const char *src,
                                             size_t bytes, int pe, int qp_index,
                                             int worker_id, int worker_count);
  __device__ void tile_get_contig_slices_nbi(char *dst, const char *src,
                                             size_t bytes, int pe, int qp_index,
                                             int worker_id, int worker_count);

  /**
   * @brief Post NBI puts for contiguous rows, striped across workers.
   *
   * Workers may diverge when num_rows is not a multiple of worker_count; each
   * round posts as a collective over whichever lanes are still in the loop.
   * Does not quiet; caller must quiet_single (or tile_finish_put) after a
   * wave/block barrier.
   * Strides are in bytes between consecutive rows.
   */
  __device__ void tile_put_rows_nbi(char *dst_base, const char *src_base,
                                    size_t dst_row_stride_bytes,
                                    size_t src_row_stride_bytes,
                                    size_t num_rows, size_t row_bytes,
                                    int pe, int qp_index, int worker_id,
                                    int worker_count);

  /**
   * @brief Post NBI puts for contiguous columns, striped across workers.
   * Strides are in bytes between consecutive columns.
   */
  __device__ void tile_put_cols_nbi(char *dst_base, const char *src_base,
                                    size_t dst_col_stride_bytes,
                                    size_t src_col_stride_bytes,
                                    size_t num_cols, size_t col_bytes,
                                    int pe, int qp_index, int worker_id,
                                    int worker_count);

  /**
   * @brief Post NBI gets for contiguous rows, striped across workers.
   */
  __device__ void tile_get_rows_nbi(char *dst_base, const char *src_base,
                                    size_t dst_row_stride_bytes,
                                    size_t src_row_stride_bytes,
                                    size_t num_rows, size_t row_bytes,
                                    int pe, int qp_index, int worker_id,
                                    int worker_count);

  /**
   * @brief Post NBI gets for contiguous columns, striped across workers.
   */
  __device__ void tile_get_cols_nbi(char *dst_base, const char *src_base,
                                    size_t dst_col_stride_bytes,
                                    size_t src_col_stride_bytes,
                                    size_t num_cols, size_t col_bytes,
                                    int pe, int qp_index, int worker_id,
                                    int worker_count);

  __device__ void tile_put_strided_2d_nbi(char *dst_base, const char *src_base,
                                          size_t dst_s0, size_t dst_s1,
                                          size_t src_s0, size_t src_s1,
                                          size_t extent0, size_t extent1,
                                          size_t element_size, int pe,
                                          int qp_index, int worker_id,
                                          int worker_count);
  __device__ void tile_get_strided_2d_nbi(char *dst_base, const char *src_base,
                                          size_t dst_s0, size_t dst_s1,
                                          size_t src_s0, size_t src_s1,
                                          size_t extent0, size_t extent1,
                                          size_t element_size, int pe,
                                          int qp_index, int worker_id,
                                          int worker_count);

  __device__ void tile_put_gda_workers(void *dst_data, const void *src_data,
                                       const size_t *dst_strides,
                                       const size_t *src_strides,
                                       const size_t *start_coord,
                                       const size_t *boundary, int ndim,
                                       size_t element_size, int pe,
                                       int worker_id, int worker_count);
  __device__ void tile_get_gda_workers(void *dst_data, const void *src_data,
                                       const size_t *dst_strides,
                                       const size_t *src_strides,
                                       const size_t *start_coord,
                                       const size_t *boundary, int ndim,
                                       size_t element_size, int pe,
                                       int worker_id, int worker_count);

  __device__
  void internal_quiet(ActiveWFInfo &wf_info);

  __device__
  void internal_quiet(int qp_index, ActiveWFInfo &wf_info);

  template <typename T>
  __device__ void internal_amo_add(void *dst, T value, int pe, int qp_index,
      ActiveWFInfo &wf_info);

  template <typename T>
  __device__ T internal_amo_fetch_add(void *dst, T value, int pe, int qp_index,
      ActiveWFInfo &wf_info);

  template <typename T>
  __device__ T internal_amo_swap(void *dst, T value, int pe, int qp_index,
      ActiveWFInfo &wf_info);

  /**
   * @brief Get the Queue Pair index to use for a given PE
   */
  __device__ __forceinline__ uint32_t get_qp_index(
      int pe, ActiveWFInfo wf_info, bool force_single_qp = false);

  __device__ __forceinline__ uint32_t get_team_qp_index(
      const GDATeam *team_obj, int pe);

  //Temporary scratchpad memory used by internal barrier algorithms.
  int64_t *barrier_sync{nullptr};

  /**
   * @brief Array containing the addresses of the work/sync buffer bases
   * of other PEs
  */
  char **wrk_sync_pool_bases_{nullptr};

  /**
   * @brief Device context Id
   */
  unsigned int ctx_id_{};

  /**
   * @brief Number of Queue Pairs allocated per PE
   */
  uint32_t num_qps_per_pe {1};

  /**
   * @brief Total number of Queue Pairs allocated = num_qps_per_pe * num_pes
   */
  uint32_t num_qps {1};

  /**
   * @brief Device pointer to the qp_counter variable to pick next qp index
   */
  uint32_t *qp_counter {nullptr};

 public:
  /**************************************************************************
   ****************** TILE API METHODS **************************************
   *************************************************************************/

  // RMA PUT operations - Type-erased interface
  __device__ int tile_put(void* dst_data, const void* src_data,
                          const size_t* dst_strides, const size_t* src_strides,
                          const size_t* start_coord, const size_t* boundary,
                          int ndim, size_t element_size, int pe, uint64_t flags);

  __device__ int tile_put_wave(void* dst_data, const void* src_data,
                               const size_t* dst_strides, const size_t* src_strides,
                               const size_t* start_coord, const size_t* boundary,
                               int ndim, size_t element_size, int pe, uint64_t flags);

  __device__ int tile_put_wg(void* dst_data, const void* src_data,
                             const size_t* dst_strides, const size_t* src_strides,
                             const size_t* start_coord, const size_t* boundary,
                             int ndim, size_t element_size, int pe, uint64_t flags);

  // RMA GET operations - Type-erased interface
  __device__ int tile_get(void* dst_data, const void* src_data,
                          const size_t* dst_strides, const size_t* src_strides,
                          const size_t* start_coord, const size_t* boundary,
                          int ndim, size_t element_size, int pe, uint64_t flags);

  __device__ int tile_get_wave(void* dst_data, const void* src_data,
                               const size_t* dst_strides, const size_t* src_strides,
                               const size_t* start_coord, const size_t* boundary,
                               int ndim, size_t element_size, int pe, uint64_t flags);

  __device__ int tile_get_wg(void* dst_data, const void* src_data,
                             const size_t* dst_strides, const size_t* src_strides,
                             const size_t* start_coord, const size_t* boundary,
                             int ndim, size_t element_size, int pe, uint64_t flags);

  // Collective operations - Allgather - Type-erased interface
  __device__ int tile_allgather(rocshmem_team_t team, void* dst_data, const void* src_data,
                                const size_t* dst_strides, const size_t* src_strides,
                                const size_t* start_coord, const size_t* boundary,
                                int ndim, size_t element_size, uint64_t flags);

  __device__ int tile_allgather_wave(rocshmem_team_t team, void* dst_data, const void* src_data,
                                     const size_t* dst_strides, const size_t* src_strides,
                                     const size_t* start_coord, const size_t* boundary,
                                     int ndim, size_t element_size, uint64_t flags);

  __device__ int tile_allgather_wg(rocshmem_team_t team, void* dst_data, const void* src_data,
                                   const size_t* dst_strides, const size_t* src_strides,
                                   const size_t* start_coord, const size_t* boundary,
                                   int ndim, size_t element_size, uint64_t flags);

  // Collective operations - Broadcast - Type-erased interface
  __device__ int tile_broadcast(rocshmem_team_t team, void* dst_data, const void* src_data,
                                const size_t* dst_strides, const size_t* src_strides,
                                const size_t* start_coord, const size_t* boundary,
                                int ndim, size_t element_size, int pe_root, uint64_t flags);

  __device__ int tile_broadcast_wave(rocshmem_team_t team, void* dst_data, const void* src_data,
                                     const size_t* dst_strides, const size_t* src_strides,
                                     const size_t* start_coord, const size_t* boundary,
                                     int ndim, size_t element_size, int pe_root, uint64_t flags);

  __device__ int tile_broadcast_wg(rocshmem_team_t team, void* dst_data, const void* src_data,
                                   const size_t* dst_strides, const size_t* src_strides,
                                   const size_t* start_coord, const size_t* boundary,
                                   int ndim, size_t element_size, int pe_root, uint64_t flags);

  // Collective wait - No change needed
  __device__ int tile_collective_wait(rocshmem_team_t team, uint64_t flags);

  // SUM Reduction operations - Type-erased interface
  __device__ int tile_sum_reduce(rocshmem_team_t team, void* dst_data, const void* src_data,
                                 const size_t* dst_strides, const size_t* src_strides,
                                 const size_t* start_coord, const size_t* boundary,
                                 int ndim, size_t element_size, int root, uint64_t flags);

  __device__ int tile_sum_reduce_wave(rocshmem_team_t team, void* dst_data, const void* src_data,
                                      const size_t* dst_strides, const size_t* src_strides,
                                      const size_t* start_coord, const size_t* boundary,
                                      int ndim, size_t element_size, int root, uint64_t flags);

  __device__ int tile_sum_reduce_wg(rocshmem_team_t team, void* dst_data, const void* src_data,
                                    const size_t* dst_strides, const size_t* src_strides,
                                    const size_t* start_coord, const size_t* boundary,
                                    int ndim, size_t element_size, int root, uint64_t flags);

  // MAX Reduction operations - Type-erased interface
  __device__ int tile_max_reduce(rocshmem_team_t team, void* dst_data, const void* src_data,
                                 const size_t* dst_strides, const size_t* src_strides,
                                 const size_t* start_coord, const size_t* boundary,
                                 int ndim, size_t element_size, int root, uint64_t flags);

  __device__ int tile_max_reduce_wave(rocshmem_team_t team, void* dst_data, const void* src_data,
                                      const size_t* dst_strides, const size_t* src_strides,
                                      const size_t* start_coord, const size_t* boundary,
                                      int ndim, size_t element_size, int root, uint64_t flags);

  __device__ int tile_max_reduce_wg(rocshmem_team_t team, void* dst_data, const void* src_data,
                                    const size_t* dst_strides, const size_t* src_strides,
                                    const size_t* start_coord, const size_t* boundary,
                                    int ndim, size_t element_size, int root, uint64_t flags);

  // MIN Reduction operations - Type-erased interface
  __device__ int tile_min_reduce(rocshmem_team_t team, void* dst_data, const void* src_data,
                                 const size_t* dst_strides, const size_t* src_strides,
                                 const size_t* start_coord, const size_t* boundary,
                                 int ndim, size_t element_size, int root, uint64_t flags);

  __device__ int tile_min_reduce_wave(rocshmem_team_t team, void* dst_data, const void* src_data,
                                      const size_t* dst_strides, const size_t* src_strides,
                                      const size_t* start_coord, const size_t* boundary,
                                      int ndim, size_t element_size, int root, uint64_t flags);

  __device__ int tile_min_reduce_wg(rocshmem_team_t team, void* dst_data, const void* src_data,
                                    const size_t* dst_strides, const size_t* src_strides,
                                    const size_t* start_coord, const size_t* boundary,
                                    int ndim, size_t element_size, int root, uint64_t flags);

  template <typename T, ROCSHMEM_OP Op>
  __device__ int tile_reduce_typed_impl(rocshmem_team_t team,
                                        const void* src_data,
                                        const size_t* src_strides,
                                        const size_t* start_coord,
                                        const size_t* boundary, int ndim,
                                        int root, size_t segment_start,
                                        size_t segment_elems,
                                        size_t segment_capacity,
                                        int worker_id, int worker_count);

  template <typename T, ROCSHMEM_OP Op>
  __device__ int tile_reduce_typed(rocshmem_team_t team, void* dst_data,
                                   const void* src_data,
                                   const size_t* dst_strides,
                                   const size_t* src_strides,
                                   const size_t* start_coord,
                                   const size_t* boundary, int ndim, int root);

  template <typename T, ROCSHMEM_OP Op>
  __device__ int tile_reduce_typed_wave(rocshmem_team_t team, void* dst_data,
                                        const void* src_data,
                                        const size_t* dst_strides,
                                        const size_t* src_strides,
                                        const size_t* start_coord,
                                        const size_t* boundary, int ndim,
                                        int root);

  template <typename T, ROCSHMEM_OP Op>
  __device__ int tile_reduce_typed_wg(rocshmem_team_t team, void* dst_data,
                                      const void* src_data,
                                      const size_t* dst_strides,
                                      const size_t* src_strides,
                                      const size_t* start_coord,
                                      const size_t* boundary, int ndim,
                                      int root);

  // Rooted SUM Reduction operations
  // Rooted MAX Reduction operations
  // Rooted MIN Reduction operations
 public:
  QueuePair *qps{nullptr};

  /**
   * @brief Base heap pointers of all PEs
   */
  char *const *base_heap{nullptr};

  //TODO(Avinash):
  //Make tinfo private variable, it requires changes to the context
  //creation API in backend

  //Team information for the team associated with the context
  TeamInfo *tinfo{nullptr};

  __device__ uint32_t get_num_qps_per_pe() { return num_qps_per_pe; }
};

}  // namespace rocshmem

#endif // LIBRARY_SRC_GDA_CONTEXT_DEVICE_HPP_
