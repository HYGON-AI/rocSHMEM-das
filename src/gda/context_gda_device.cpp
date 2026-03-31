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

#include <hip/hip_runtime.h>
#include <hip/amd_detail/amd_device_functions.h>

#include "rocshmem/rocshmem_config.h"  // NOLINT(build/include_subdir)
#include "rocshmem/rocshmem.hpp"
#include "backend_gda.hpp"
#include "context_gda_device.hpp"
#include "context_gda_tmpl_device.hpp"
#include "queue_pair.hpp"

namespace rocshmem {

__host__ GDAContext::GDAContext(Backend *b, unsigned int ctx_id, int gda_provider)
    : Context(b) {
  GDABackend *backend{static_cast<GDABackend *>(b)};
  base_heap = backend->heap.get_heap_bases().data();

  barrier_sync = backend->barrier_sync;
  wrk_sync_pool_bases_ = backend->get_wrk_sync_bases();

  uint32_t num_qps_per_pe_default_ctx = max(1, 1 + envvar::gda::num_qps_default_ctx.get_value() / num_pes);
  uint32_t num_qps_per_pe_usr_ctx = max(1, envvar::gda::num_qps_per_pe_usr_ctx.get_value());
  num_qps_per_pe = ctx_id ? num_qps_per_pe_usr_ctx : num_qps_per_pe_default_ctx;
  num_qps = num_qps_per_pe * num_pes;

  CHECK_HIP(hipMalloc(&qps, sizeof(QueuePair) * num_qps));
  CHECK_HIP(hipMemset(qps, 0, sizeof(QueuePair) * num_qps));

  // Calculate offset into the backend's GPU QP array
  int offset = num_pes * (ctx_id > 0) * (num_qps_per_pe_default_ctx + num_qps_per_pe_usr_ctx * (ctx_id - 1));
  CHECK_HIP(hipMemcpy(qps, &backend->gpu_qps[offset], num_qps * sizeof(QueuePair), hipMemcpyDefault));

  for (int i = 0; i < num_qps; i++) {
    qps[i].base_heap = base_heap;
  }

  ipcImpl_.ipc_bases = backend->ipcImpl.ipc_bases;
  ipcImpl_.shm_size = backend->ipcImpl.shm_size;
  ipcImpl_.shm_rank = backend->ipcImpl.shm_rank;
  ipcImpl_.pes_with_ipc_avail = backend->ipcImpl.pes_with_ipc_avail;

  ctx_id_ = ctx_id;
  gda_provider_ = gda_provider;
}

__device__ char* GDAContext::get_remote_ptr(const void* addr, int pe) {
  const char* addr_ = reinterpret_cast<const char*>(addr);
  uint64_t L_offset = addr_ - base_heap[my_pe];
  return base_heap[pe] + L_offset;
}

__device__ char* GDAContext::get_local_ptr(const void* addr, int pe) {
  const char* addr_ = reinterpret_cast<const char*>(addr);
  uint64_t L_offset = addr_ - ipcImpl_.ipc_bases[ipcImpl_.shm_rank];
  return ipcImpl_.ipc_bases[pe] + L_offset;
}

__device__ uint64_t GDAContext::get_p2p_ptr(void *dest, int rank, int dst_rank){
  if (rank == dst_rank) // 相同 GPU 的地址可以直接返回
    return reinterpret_cast<uint64_t>(dest);

  int local_ranks = ipcImpl_.shm_size; // 获取本节点的rank数量
  int node_src = rank / local_ranks;
  int node_dst = dst_rank / local_ranks;

  if (node_src != node_dst) { // 先将 RDMA 的请求返回
    return 0;
  } else {  // 同一计算节点上的 不同 GPU 需要 IPC 映射后将指针返回
    uint64_t L_offset = reinterpret_cast<char *>(dest) - ipcImpl_.ipc_bases[rank % local_ranks];
    char* dst_ptr = ipcImpl_.ipc_bases[dst_rank % local_ranks] + L_offset;
    return reinterpret_cast<uint64_t>(dst_ptr);
  }
}

__host__ GDAContext::~GDAContext() {
  CHECK_HIP(hipFree(qps));
}

__device__ void GDAContext::ctx_create() {
}

__device__ void GDAContext::ctx_destroy(){
}

__device__ void GDAContext::putmem(void *dest, const void *source, size_t nelems,
                                   int pe) {
  int local_pe{-1};
  if (ipcImpl_.isIpcAvailable(my_pe, pe, &local_pe)) {
    ipcImpl_.ipcCopy(get_local_ptr(dest, local_pe), const_cast<void *>(source), nelems);
    return;
  }
  qps[pe].put_nbi(get_remote_ptr(dest, pe), source, nelems, pe);
  qps[pe].quiet();
}

__device__ void GDAContext::getmem(void *dest, const void *source, size_t nelems,
                                   int pe) {
  int local_pe{-1};
  if (ipcImpl_.isIpcAvailable(my_pe, pe, &local_pe)) {
    ipcImpl_.ipcCopy(dest, get_local_ptr(source, local_pe), nelems);
    return;
  }
  qps[pe].get_nbi(dest, get_remote_ptr(source, pe), nelems, pe);
  qps[pe].quiet();
}

__device__ void GDAContext::putmem_nbi(void *dest, const void *source,
                                       size_t nelems, int pe) {
  int local_pe{-1};
  if (ipcImpl_.isIpcAvailable(my_pe, pe, &local_pe)) {
    ipcImpl_.ipcCopy(get_local_ptr(dest, local_pe), const_cast<void *>(source), nelems);
    return;
  }
  qps[pe].put_nbi(get_remote_ptr(dest, pe), source, nelems, pe);
}

__device__ void GDAContext::getmem_nbi(void *dest, const void *source,
                                       size_t nelems, int pe) {
  int local_pe{-1};
  if (ipcImpl_.isIpcAvailable(my_pe, pe, &local_pe)) {
    ipcImpl_.ipcCopy(dest, get_local_ptr(source, local_pe), nelems);
    return;
  }
  qps[pe].get_nbi(dest, get_remote_ptr(source, pe), nelems, pe);
}

__device__ void GDAContext::fence() { //TODO: optimize
  for (int i = 0; i < num_qps; i++) {
    qps[i].quiet();
  }
  __threadfence_system();
}

__device__ void GDAContext::fence([[maybe_unused]] int pe) {
  fence(); //TODO: optimize
}

__device__ void GDAContext::quiet() {
  for (int i = 0; i < num_qps; i++) {
    qps[i].quiet();
  }
}

__device__ void GDAContext::qp_quiet(size_t qp_idx) {
  qps[qp_idx].quiet_dp_single_lane();
}

__device__ void GDAContext::quiet_wave() {
  for (int i = 0; i < num_qps; i++) {
    qps[i].quiet(QueuePair::WAVE);
  }
}

__device__ void GDAContext::pe_quiet(size_t pe) {
  for(int i = 0; i < num_qps_per_pe; i++) {
    int qp_index = i * num_pes + pe;
    qps[qp_index].quiet();
  }
}

__device__ void GDAContext::pe_quiet_single(size_t pe) {
  qps[pe].quiet_single();
}

__device__ void *GDAContext::shmem_ptr(const void *dest, int pe) {
  void *ret = nullptr;
  int local_pe{-1};
  if (ipcImpl_.isIpcAvailable(my_pe, pe, &local_pe)) {
    void *dst = const_cast<void *>(dest);
    uint64_t L_offset = reinterpret_cast<char *>(dst) - ipcImpl_.ipc_bases[ipcImpl_.shm_rank];
    ret = ipcImpl_.ipc_bases[local_pe] + L_offset;
  }
  return ret;
}

__device__ void GDAContext::putmem_wg(void *dest, const void *source,
                                      size_t nelems, int pe) {
  int local_pe{-1};
  if (ipcImpl_.isIpcAvailable(my_pe, pe, &local_pe)) {
    ipcImpl_.ipcCopy_wg(get_local_ptr(dest, local_pe), const_cast<void *>(source), nelems);
    return;
  }
  if (is_wave_zero_in_block()) {
    qps[pe].put_nbi(get_remote_ptr(dest, pe), source, nelems, pe, QueuePair::WAVE);
    qps[pe].quiet();
  }
}

__device__ void GDAContext::getmem_wg(void *dest, const void *source,
                                      size_t nelems, int pe) {
  const char *src_typed = reinterpret_cast<const char *>(source);
  int local_pe{-1};
  if (ipcImpl_.isIpcAvailable(my_pe, pe, &local_pe)) {
    ipcImpl_.ipcCopy_wg(dest, get_local_ptr(source, local_pe), nelems);
    return;
  }
  if (is_wave_zero_in_block()) {
    qps[pe].get_nbi(dest, get_remote_ptr(source, pe), nelems, pe, QueuePair::WAVE);
    qps[pe].quiet();
  }
}

__device__ void GDAContext::putmem_nbi_wg(void *dest, const void *source,
                                          size_t nelems, int pe) {
  int local_pe{-1};
  if (ipcImpl_.isIpcAvailable(my_pe, pe, &local_pe)) {
    ipcImpl_.ipcCopy_wg(get_local_ptr(dest, local_pe), const_cast<void *>(source), nelems);
    return;
  }
  if (is_wave_zero_in_block()) {
    qps[pe].put_nbi(get_remote_ptr(dest, pe), source, nelems, pe, QueuePair::WAVE);
  }
}

__device__ void GDAContext::getmem_nbi_wg(void *dest, const void *source,
                                          size_t nelems, int pe) {
  const char *src_typed = reinterpret_cast<const char *>(source);
  int local_pe{-1};
  if (ipcImpl_.isIpcAvailable(my_pe, pe, &local_pe)) {
    ipcImpl_.ipcCopy_wg(dest, get_local_ptr(source, local_pe), nelems);
    return;
  }
  if (is_wave_zero_in_block()) {
    qps[pe].get_nbi(dest, get_remote_ptr(source, pe), nelems, pe, QueuePair::WAVE);
  }
}

__device__ void GDAContext::putmem_wave(void *dest, const void *source,
                                        size_t nelems, int pe) {
  int local_pe{-1};
  if (ipcImpl_.isIpcAvailable(my_pe, pe, &local_pe)) {
    ipcImpl_.ipcCopy_wave(get_local_ptr(dest, local_pe), const_cast<void *>(source), nelems);
    return;
  }
  qps[pe].put_nbi(get_remote_ptr(dest, pe), source, nelems, pe, QueuePair::WAVE);
  qps[pe].quiet();
}

__device__ void GDAContext::getmem_wave(void *dest, const void *source,
                                        size_t nelems, int pe) {
  const char *src_typed = reinterpret_cast<const char *>(source);
  int local_pe{-1};
  if (ipcImpl_.isIpcAvailable(my_pe, pe, &local_pe)) {
    ipcImpl_.ipcCopy_wave(dest, get_local_ptr(source, local_pe), nelems);
    return;
  }
  qps[pe].get_nbi(dest, get_remote_ptr(source, pe), nelems, pe, QueuePair::WAVE);
  qps[pe].quiet();
}

__device__ void GDAContext::putmem_nbi_wave(void *dest, const void *source,
                                            size_t nelems, int pe) {
  int local_pe{-1};
  if (ipcImpl_.isIpcAvailable(my_pe, pe, &local_pe)) {
    ipcImpl_.ipcCopy_wave(get_local_ptr(dest, local_pe), const_cast<void *>(source), nelems);
    return;
  }
  uint64_t L_offset = reinterpret_cast<char*>(dest) - base_heap[my_pe];
  qps[pe].put_nbi(get_remote_ptr(dest, pe), source, nelems, pe, QueuePair::WAVE);
}

__device__ void GDAContext::putmem_nbi_wave_dp(void *dest, const void *source,
                                            size_t nelems, int qp_idx, int pe) {
  // int local_pe{-1};
  // if (ipcImpl_.isIpcAvailable(my_pe, pe, &local_pe)) {
  //   ipcImpl_.ipcCopy_wave(get_local_ptr(dest, local_pe), const_cast<void *>(source), nelems);
  //   return;
  // }
  if (is_thread_zero_in_wave()) {
     qps[qp_idx].put_nbi_dp(get_remote_ptr(dest, pe), source, nelems);
  }
}

__device__ void GDAContext::getmem_nbi_wave(void *dest, const void *source,
                                            size_t nelems, int pe) {
  const char *src_typed = reinterpret_cast<const char *>(source);
  int local_pe{-1};
  if (ipcImpl_.isIpcAvailable(my_pe, pe, &local_pe)) {
    ipcImpl_.ipcCopy_wave(dest, get_local_ptr(source, local_pe), nelems);
    return;
  }
  qps[pe].get_nbi(dest, get_remote_ptr(source, pe), nelems, pe, QueuePair::WAVE);
}


//TODO: copied from IPC, needs review
__device__ void GDAContext::putmem_signal(void *dest, const void *source, size_t nelems,
                                          uint64_t *sig_addr, uint64_t signal, int sig_op,
                                          int pe) {
  putmem(dest, source, nelems, pe);
  fence();

  switch (sig_op) {
  case ROCSHMEM_SIGNAL_SET:
    amo_set<uint64_t>(static_cast<void*>(sig_addr), signal, pe);
    break;
  case ROCSHMEM_SIGNAL_ADD:
    amo_add<uint64_t>(static_cast<void*>(sig_addr), signal, pe);
    break;
  default:
    DPRINTF("[%s] Invalid sig_op value (%d)\n", __func__, sig_op);
    break;
  }
  //TODO: missing quiet_pe?
}

__device__ void GDAContext::putmem_signal_wg(void *dest, const void *source, size_t nelems,
                                             uint64_t *sig_addr, uint64_t signal, int sig_op,
                                             int pe) {
  putmem_wg(dest, source, nelems, pe);
  fence();

  if (is_thread_zero_in_block()) {
    switch (sig_op) {
    case ROCSHMEM_SIGNAL_SET:
      amo_set<uint64_t>(static_cast<void*>(sig_addr), signal, pe);
      break;
    case ROCSHMEM_SIGNAL_ADD:
      amo_add<uint64_t>(static_cast<void*>(sig_addr), signal, pe);
      break;
    default:
      DPRINTF("[%s] Invalid sig_op value (%d)\n", __func__, sig_op);
      break;
    }
    //TODO: missing quiet_pe?
  }
}

__device__ void GDAContext::putmem_signal_wave(void *dest, const void *source, size_t nelems,
                                               uint64_t *sig_addr, uint64_t signal, int sig_op,
                                               int pe) {
  putmem_wave(dest, source, nelems, pe);
  fence();

  if (is_thread_zero_in_wave()) {
    switch (sig_op) {
    case ROCSHMEM_SIGNAL_SET:
      amo_set<uint64_t>(static_cast<void*>(sig_addr), signal, pe);
      break;
    case ROCSHMEM_SIGNAL_ADD:
      amo_add<uint64_t>(static_cast<void*>(sig_addr), signal, pe);
      break;
    default:
      DPRINTF("[%s] Invalid sig_op value (%d)\n", __func__, sig_op);
      break;
    }
    //TODO: missing quiet_pe?
  }
}

__device__ void GDAContext::putmem_signal_nbi(void *dest, const void *source, size_t nelems,
                                              uint64_t *sig_addr, uint64_t signal, int sig_op,
                                              int pe) {
  putmem_signal(dest, source, nelems, sig_addr, signal, sig_op, pe); //TODO: optimize
}

__device__ void GDAContext::putmem_signal_nbi_wg(void *dest, const void *source, size_t nelems,
                                                 uint64_t *sig_addr, uint64_t signal, int sig_op,
                                                 int pe) {
  putmem_signal_wg(dest, source, nelems, sig_addr, signal, sig_op, pe); //TODO: optimize
}

__device__ void GDAContext::putmem_signal_nbi_wave(void *dest, const void *source, size_t nelems,
                                                   uint64_t *sig_addr, uint64_t signal, int sig_op,
                                                   int pe) {
  putmem_signal_wave(dest, source, nelems, sig_addr, signal, sig_op, pe); //TODO: optimize
}

__device__ uint64_t GDAContext::signal_fetch(const uint64_t *sig_addr) {
  uint64_t *dst = const_cast<uint64_t*>(sig_addr);
  return amo_fetch_add<uint64_t>(static_cast<void*>(dst), 0, my_pe);
}

__device__ uint64_t GDAContext::signal_fetch_wg(const uint64_t *sig_addr) {
  __shared__ uint64_t value;
  if (is_thread_zero_in_block()) {
    uint64_t *dst = const_cast<uint64_t*>(sig_addr);
    value = amo_fetch_add<uint64_t>(static_cast<void*>(dst), 0, my_pe);
  }
  __threadfence_block();
  return value;
}

__device__ uint64_t GDAContext::signal_fetch_wave(const uint64_t *sig_addr) {
  uint64_t value = 0;
  if (is_thread_zero_in_wave()) {
    uint64_t *dst = const_cast<uint64_t*>(sig_addr);
    value = amo_fetch_add<uint64_t>(static_cast<void*>(dst), 0, my_pe);
  }
  __threadfence_block();
  value = __shfl(value, 0);
  return value;
}

}  // namespace rocshmem
