// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

/******************************************************************************
 * 1. 较大的DAS自研改动在本文件中维护，尽量保留原文件中的社区代码，
 *    方便后续维护及同步上游更新。
 * 2. 修改已有函数时，在此实现对应的_das函数；启用USE_DAS后(默认开启)，原函数
 *    在入口调用_das函数并返回，未启用时执行原来的社区实现。
 * 3. 新增辅助函数也在此维护，公共逻辑尽量复用原文件中的函数；
 *****************************************************************************/

#include "rocshmem/rocshmem.hpp"
#include "constmem.hpp"
#include "context_incl.hpp"
#include "context_ipc_tmpl_device.hpp"
#include "util.hpp"
#include "ipc_team.hpp"

#if defined(USE_DAS)

namespace rocshmem {

/*
 * Chunk-pipelined chain broadcast for large messages.
 *
 * Ranks are rotated so that the root has virtual rank zero.  The message is
 * split into chunks and forwarded along the virtual-rank chain.  In a given
 * round, each link carries a different chunk, so all active links can make
 * progress concurrently.  A team-wide synchronization closes each round and
 * guarantees that a received chunk is visible before it is forwarded.
 *
 * Unlike a full-message binomial tree, the root sends each byte only once.
 */
__device__ void IPCContext::internal_chain_broadcast(
    void *dst, const void *src, int nelems, int pe_root, int pe_start, int stride,
    int pe_size, long *p_sync) {  // NOLINT(runtime/int)
  // Compute virtual rank: root=0, enables circular chain topology
  int my_rank = (constmem.my_pe - pe_start) / stride;
  int root_rank = (pe_root - pe_start) / stride;

  int virtual_rank = my_rank - root_rank;
  if (virtual_rank < 0) {
    virtual_rank += pe_size;
  }

  // Root fetches data from source into local dst buffer
  if (virtual_rank == 0) {
    getmem_wg(dst, src, nelems, pe_root);
  }

  // Split message into chunks for pipelined forwarding
  constexpr int CHUNK_NELEMS = ROCSHMEM_CHUNK_BYTES;
  int num_chunks = (nelems + CHUNK_NELEMS - 1) / CHUNK_NELEMS;

  // Determine downstream neighbor in the chain (-1 if last node)
  int next_pe = -1;
  if (virtual_rank < pe_size - 1) {
    int next_rank = virtual_rank + 1 + root_rank;
    if (next_rank >= pe_size) {
      next_rank -= pe_size;
    }
    next_pe = pe_start + next_rank * stride;
  }

  // Pipeline: forward chunks along the chain with overlap
  for (int chunk = 0; chunk < num_chunks; chunk++) {
    // Progress signal: chunk N is ready when p_sync[0] >= N+1
    int64_t ready = chunk + 1;

    // Non-root nodes wait for upstream to send this chunk
    if (virtual_rank > 0) {
      if (is_thread_zero_in_block()) {
        wait_until(&p_sync[0], ROCSHMEM_CMP_GE, ready);
      }
      __syncthreads();
    }

    // Compute current chunk boundaries
    int offset = chunk * CHUNK_NELEMS;
    int count = min(CHUNK_NELEMS, nelems - offset);

    // Forward this chunk to downstream neighbor
    if (next_pe != -1) {
      // Root reads from src, other nodes forward from received dst
      const void *forward_src = (virtual_rank == 0) ? static_cast<const char *>(src) + offset
                                                    : static_cast<const char *>(dst) + offset;
      putmem_wg(static_cast<char *>(dst) + offset, forward_src, count, next_pe);

      // Signal completion to downstream node
      if (is_thread_zero_in_block()) {
        fence(next_pe);
        internal_putmem(&p_sync[0], &ready, sizeof(ready), next_pe);
      }
      __syncthreads();
    }
  }

  // Global barrier ensures all PEs have completed reception
  internal_sync_wg(constmem.my_pe, pe_start, stride, pe_size, p_sync + 1);

  // The chunk-ready flag is reused by the next broadcast on this team.  Clear
  // it after the completion barrier so a subsequent invocation cannot mistake
  // the previous operation's progress for the current operation's progress.
  if (is_thread_zero_in_block()) {
    *p_sync = 0;
    threadfence_system();
  }
  __syncthreads();
}

__device__ void IPCContext::internal_broadcastmem_wg_das(void *dst, const void *src, int nelems,
                                      int pe_root, int pe_start,
                                      int stride, int pe_size,
                                      long *p_sync) {  // NOLINT(runtime/int)
  constexpr size_t DIRECT_MAX_BYTES = 8 * 1024;
  size_t message_bytes = static_cast<size_t>(nelems);

  if (pe_size <= 2 || message_bytes <= DIRECT_MAX_BYTES) {
    internal_put_broadcastmem_wg(dst, src, nelems, pe_root, pe_start, stride,
                           pe_size);
    internal_sync_wg(constmem.my_pe, pe_start, stride, pe_size, p_sync);
  } else {
    internal_chain_broadcast(dst, src, nelems, pe_root, pe_start, stride, pe_size, p_sync);
  }
}

}  // namespace rocshmem
#endif  // defined(USE_DAS)
