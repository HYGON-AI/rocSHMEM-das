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

#include "rocshmem/rocshmem.hpp"
#include "constmem.hpp"
#include "context_incl.hpp"
#include "context_ipc_tmpl_device.hpp"
#include "util.hpp"
#include "ipc_team.hpp"

namespace rocshmem {

__device__ void IPCContext::internal_direct_barrier(int pe, int PE_start,
                                                    int stride, int n_pes,
                                                    int64_t *pSync) {
  int64_t flag_val = 1;
  if (pe == PE_start) {
    // Go through all PE offsets (except current offset = 0)
    // and wait until they all reach
#if defined(__gfx936__) || defined (__gfx938__)
    __threadfence_system();
#endif /* __gfx936__ */
    for (int i = 1; i < n_pes; i++) {
      wait_until(&pSync[i], ROCSHMEM_CMP_EQ, flag_val);
      pSync[i] = ROCSHMEM_SYNC_VALUE;
    }
    threadfence_system();

    // Announce to other PEs that all have reached
    for (int i = 1, j = PE_start + stride; i < n_pes; ++i, j += stride) {
      internal_putmem(&pSync[0], &flag_val, sizeof(*pSync), j);
#if defined(__gfx936__) || defined (__gfx938__)
        __threadfence_system();
#endif /* __gfx936__ */
    }
  } else {
    // Mark current PE offset as reached
    size_t pe_offset = (pe - PE_start) / stride;
    internal_putmem(&pSync[pe_offset], &flag_val, sizeof(*pSync), PE_start);
#if defined(__gfx936__) || defined (__gfx938__)
    __threadfence_system();
#endif /* __gfx936__ */
    wait_until(&pSync[0], ROCSHMEM_CMP_EQ, flag_val);
    pSync[0] = ROCSHMEM_SYNC_VALUE;
    threadfence_system();
  }
}

__device__ void IPCContext::internal_atomic_barrier(int pe, int PE_start,
                                                    int stride, int n_pes,
                                                    int64_t *pSync) {
  int64_t flag_val = 1;
  if (pe == PE_start) {
    wait_until(&pSync[0], ROCSHMEM_CMP_EQ, (int64_t)(n_pes - 1));
    pSync[0] = ROCSHMEM_SYNC_VALUE;
    threadfence_system();

    for (int i = 1, j = PE_start + stride; i < n_pes; ++i, j += stride) {
      internal_putmem(&pSync[0], &flag_val, sizeof(*pSync), j);
    }
  } else {
    amo_add<int64_t>(&pSync[0], flag_val, PE_start);
    wait_until(&pSync[0], ROCSHMEM_CMP_EQ, flag_val);
    pSync[0] = ROCSHMEM_SYNC_VALUE;
    threadfence_system();
  }
}

__device__ void IPCContext::internal_sync(int pe, int PE_start, int stride,
                                          int PE_size, int64_t *pSync) {
  if (PE_size < 64) {
    internal_direct_barrier(pe, PE_start, stride, PE_size, pSync);
  } else {
    internal_atomic_barrier(pe, PE_start, stride, PE_size, pSync);
  }
}

__device__ void IPCContext::internal_sync_wave(int pe, int PE_start, int stride,
                                               int PE_size, int64_t *pSync) {
  if (is_thread_zero_in_wave()) {
    if (PE_size < 64) {
      internal_direct_barrier(pe, PE_start, stride, PE_size, pSync);
    } else {
      internal_atomic_barrier(pe, PE_start, stride, PE_size, pSync);
    }
  }
}

// Uses PE values that are relative to world
__device__ void IPCContext::internal_sync_wg(int pe, int PE_start, int stride,
                                             int PE_size, int64_t *pSync) {
  __syncthreads();
  if (is_thread_zero_in_block()) {
    if (PE_size < 64) {
      internal_direct_barrier(pe, PE_start, stride, PE_size, pSync);
    } else {
      internal_atomic_barrier(pe, PE_start, stride, PE_size, pSync);
    }
  }
  __syncthreads();
}

__device__ void IPCContext::sync(rocshmem_team_t team) {
  IPCTeam *team_obj = reinterpret_cast<IPCTeam *>(team);

  int pe = team_obj->my_pe_in_world;
  int pe_start = team_obj->tinfo_wrt_world->pe_start;
  int pe_stride = team_obj->tinfo_wrt_world->stride;
  int pe_size = team_obj->num_pes;
  long *p_sync = team_obj->barrier_pSync;

  internal_sync(pe, pe_start, pe_stride, pe_size, p_sync);
}

__device__ void IPCContext::sync_wave(rocshmem_team_t team) {
  IPCTeam *team_obj = reinterpret_cast<IPCTeam *>(team);

  int pe = team_obj->my_pe_in_world;
  int pe_start = team_obj->tinfo_wrt_world->pe_start;
  int pe_stride = team_obj->tinfo_wrt_world->stride;
  int pe_size = team_obj->num_pes;
  long *p_sync = team_obj->barrier_pSync;

  internal_sync_wave(pe, pe_start, pe_stride, pe_size, p_sync);
}

__device__ void IPCContext::sync_wg(rocshmem_team_t team) {
  IPCTeam *team_obj = reinterpret_cast<IPCTeam *>(team);

  int pe = team_obj->my_pe_in_world;
  int pe_start = team_obj->tinfo_wrt_world->pe_start;
  int pe_stride = team_obj->tinfo_wrt_world->stride;
  int pe_size = team_obj->num_pes;
  long *p_sync = team_obj->barrier_pSync;

  internal_sync_wg(pe, pe_start, pe_stride, pe_size, p_sync);
}

__device__ void IPCContext::sync_all() {
  internal_sync(constmem.my_pe, 0, 1, constmem.num_pes, barrier_sync);
}

__device__ void IPCContext::sync_all_wave() {
  internal_sync_wave(constmem.my_pe, 0, 1, constmem.num_pes, barrier_sync);
}

__device__ void IPCContext::sync_all_wg() {
  internal_sync_wg(constmem.my_pe, 0, 1, constmem.num_pes, barrier_sync);
}

__device__ void IPCContext::barrier_all() {
  quiet();
  sync_all();
}

__device__ void IPCContext::barrier_all_wave() {
  if (is_thread_zero_in_wave()) {
    quiet();
  }
  sync_all_wave();
}

__device__ void IPCContext::barrier_all_wg() {
  if (is_thread_zero_in_block()) {
    quiet();
  }
  sync_all_wg();
  __syncthreads();
}

__device__ void IPCContext::barrier(rocshmem_team_t team) {
  IPCTeam *team_obj = reinterpret_cast<IPCTeam *>(team);

  int pe = team_obj->my_pe_in_world;
  int pe_start = team_obj->tinfo_wrt_world->pe_start;
  int pe_stride = team_obj->tinfo_wrt_world->stride;
  int pe_size = team_obj->num_pes;
  long *p_sync = team_obj->barrier_pSync;

  quiet();
  internal_sync(pe, pe_start, pe_stride, pe_size, p_sync);
}

__device__ void IPCContext::barrier_wave(rocshmem_team_t team) {
  IPCTeam *team_obj = reinterpret_cast<IPCTeam *>(team);

  int pe = team_obj->my_pe_in_world;
  int pe_start = team_obj->tinfo_wrt_world->pe_start;
  int pe_stride = team_obj->tinfo_wrt_world->stride;
  int pe_size = team_obj->num_pes;
  long *p_sync = team_obj->barrier_pSync;

  if (is_thread_zero_in_wave()) {
    quiet();
  }
  internal_sync_wave(pe, pe_start, pe_stride, pe_size, p_sync);
}

__device__ void IPCContext::barrier_wg(rocshmem_team_t team) {
  IPCTeam *team_obj = reinterpret_cast<IPCTeam *>(team);

  int pe = team_obj->my_pe_in_world;
  int pe_start = team_obj->tinfo_wrt_world->pe_start;
  int pe_stride = team_obj->tinfo_wrt_world->stride;
  int pe_size = team_obj->num_pes;
  long *p_sync = team_obj->barrier_pSync;

  if (is_thread_zero_in_block()) {
    quiet();
  }
  internal_sync_wg(pe, pe_start, pe_stride, pe_size, p_sync);
  __syncthreads();
}

__device__ void IPCContext::internal_put_broadcastmem_wave(
    void *dst, const void *src, int nelems, int pe_root, int pe_start,
    int stride, int pe_size) {  // NOLINT(runtime/int)
  if (my_pe == pe_root) {
    int finish = pe_start + stride * pe_size;
    for (int i = pe_start; i < finish; i += stride) {
        putmem_nbi_wave(dst, src, nelems, i);
    }
  }
}

__device__ void IPCContext::internal_get_broadcastmem_wave(
  void *dst, const void *src, int nelems, int pe_root) {
    getmem_wave(dst, src, nelems, pe_root);
}

__device__ void IPCContext::internal_broadcastmem_wave(void *dst, const void *src, int nelems,
                                      int pe_root, int pe_start,
                                      int stride, int pe_size,
                                      long *p_sync) {  // NOLINT(runtime/int)
  if (num_pes < 4) {
    internal_put_broadcastmem_wave(dst, src, nelems, pe_root, pe_start, stride, pe_size);
  } else {
    internal_get_broadcastmem_wave(dst, src, nelems, pe_root);
  }

  // Synchronize on completion of broadcast
  internal_sync_wave(my_pe, pe_start, stride, pe_size, p_sync);
}

__device__ int IPCContext::broadcastmem_wave(rocshmem_team_t team,
                              void *dest, const void *source, int nelement, int PE_root) {

  if (dest == nullptr || source == nullptr || team == ROCSHMEM_TEAM_INVALID)
    return ROCSHMEM_ERROR;

  IPCTeam *team_obj = reinterpret_cast<IPCTeam *>(team);

  int stride = team_obj->tinfo_wrt_world->stride;
  int pe_start = team_obj->tinfo_wrt_world->pe_start;
  int pe_size = team_obj->tinfo_wrt_world->size;
  long *p_sync = team_obj->bcast_pSync;

  // Passed pe_root is relative to team, convert to world root
  int pe_root_world = team_obj->get_pe_in_world(PE_root);

  internal_broadcastmem_wave(dest, source, nelement, pe_root_world, 
                              pe_start, stride, pe_size, p_sync);
  return ROCSHMEM_SUCCESS;
}

__device__ void IPCContext::internal_put_broadcastmem_wg(
    void *dst, const void *src, int nelems, int pe_root, int pe_start,
    int stride, int pe_size) {  // NOLINT(runtime/int)
  if (my_pe == pe_root) {
    int finish = pe_start + stride * pe_size;
    for (int i = pe_start; i < finish; i += stride) {
        putmem_nbi_wg(dst, src, nelems, i);
    }
  }
}

__device__ void IPCContext::internal_get_broadcastmem_wg(
  void *dst, const void *src, int nelems, int pe_root) {
    getmem_wg(dst, src, nelems, pe_root);
}

__device__ void IPCContext::internal_broadcastmem_wg(void *dst, const void *src, int nelems,
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

  // if (constmem.num_pes < 4) {
  //   internal_put_broadcastmem_wg(dst, src, nelems, pe_root, pe_start, stride, pe_size);
  // } else {
  //   internal_get_broadcastmem_wg(dst, src, nelems, pe_root);
  // }

  // // Synchronize on completion of broadcast
  // internal_sync_wg(my_pe, pe_start, stride, pe_size, p_sync);

}

__device__ void IPCContext::broadcastmem_wg(rocshmem_team_t team,
                              void *dest, const void *source, int nelement, int PE_root) {
  IPCTeam *team_obj = reinterpret_cast<IPCTeam *>(team);

  int stride = team_obj->tinfo_wrt_world->stride;
  int pe_start = team_obj->tinfo_wrt_world->pe_start;
  int pe_size = team_obj->tinfo_wrt_world->size;
  long *p_sync = team_obj->bcast_pSync;

  // Passed pe_root is relative to team, convert to world root
  int pe_root_world = team_obj->get_pe_in_world(PE_root);

  internal_broadcastmem_wg(dest, source, nelement, pe_root_world, 
                              pe_start, stride, pe_size, p_sync);
}

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

}  // namespace rocshmem
