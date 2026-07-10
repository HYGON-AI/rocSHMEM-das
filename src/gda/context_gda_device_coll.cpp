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
#include "context_incl.hpp"
#include "context_gda_tmpl_device.hpp"
#include "util.hpp"
#include "gda_team.hpp"

namespace rocshmem {

__device__ void GDAContext::internal_direct_barrier(int pe, int PE_start,
    int stride, int n_pes, int64_t *pSync, ActiveWFInfo &wf_info) {
  int64_t flag_val{1};
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
    __threadfence_system();

    // Announce to other PEs that all have reached
    for (int i = 1, j = PE_start + stride; i < n_pes; ++i, j += stride) {
      pSync[0] = flag_val;
      internal_putmem(&pSync[0], &pSync[0], sizeof(*pSync), j, j, wf_info);
#if defined(__gfx936__) || defined (__gfx938__)
      __threadfence_system();
#endif /* __gfx936__ */
    }
    pSync[0] = ROCSHMEM_SYNC_VALUE;
  } else {
    // Mark current PE offset as reached
    size_t pe_offset = (pe - PE_start) / stride;
    pSync[pe_offset] = flag_val;
    internal_putmem(&pSync[pe_offset], &pSync[pe_offset], sizeof(*pSync),
      PE_start, PE_start, wf_info);
#if defined(__gfx936__) || defined (__gfx938__)
    __threadfence_system();
#endif /* __gfx936__ */
    wait_until(&pSync[0], ROCSHMEM_CMP_EQ, flag_val);
    pSync[0] = ROCSHMEM_SYNC_VALUE;
    pSync[pe_offset] = ROCSHMEM_SYNC_VALUE;
    __threadfence_system();
  }
}

__device__ void GDAContext::internal_direct_barrier_wg(int pe, int PE_start,
    int stride, int n_pes, int64_t *pSync, ActiveWFInfo &wf_info) {
  int64_t flag_val{1};

  if (pe == PE_start) {
    int wf_id = get_flat_block_id() / WF_SIZE;
    int wf_count = (int) ceil((double)get_flat_block_size() / (double)WF_SIZE);
    bool wf_leader = is_first_active_lane();

    // Go through all PE offsets (except current offset = 0)
    // and wait until they all reach
    if (wf_leader) {
      for (int j = wf_id + 1; j < n_pes; j+= wf_count) {
        wait_until(&pSync[j], ROCSHMEM_CMP_EQ, flag_val);
        pSync[j] = ROCSHMEM_SYNC_VALUE;
      }
    }

    __syncthreads();

    // Announce to other PEs that all have reached
    for (int i = wf_id + 1, j = PE_start + stride + wf_id;
             i < n_pes;
             i+= wf_count, j += (wf_count * stride)) {
      internal_putmem_nbi_wave(&pSync[0], &flag_val, sizeof(*pSync), j,
        j, wf_info);
    }

    for (int i = wf_id + 1, j = PE_start + stride + wf_id;
             i < n_pes;
             i+= wf_count, j += (wf_count * stride)) {
      if (is_thread_zero_in_wave()) {
        qps[j].quiet(wf_info);
      }
    }

    __syncthreads();

    if (is_thread_zero_in_block()) {
      pSync[0] = ROCSHMEM_SYNC_VALUE;
    }
  } else {
    if (is_thread_zero_in_block()) {
      // Mark current PE offset as reached
      size_t pe_offset = (pe - PE_start) / stride;
      internal_putmem(&pSync[pe_offset], &flag_val, sizeof(*pSync), PE_start,
        PE_start, wf_info);
      wait_until(&pSync[0], ROCSHMEM_CMP_EQ, flag_val);
      pSync[0] = ROCSHMEM_SYNC_VALUE;
      __threadfence_system();
    }
  }
}

__device__ void GDAContext::internal_atomic_barrier(int pe, int PE_start,
    int stride, int n_pes, int64_t *pSync, ActiveWFInfo &wf_info) {
  int64_t flag_val{1};
  if (pe == PE_start) {
    wait_until(&pSync[0], ROCSHMEM_CMP_EQ, (int64_t)(n_pes - 1));
    pSync[0] = ROCSHMEM_SYNC_VALUE;
    __threadfence_system();

    pSync[0] = flag_val;
    for (int i = 1, j = PE_start + stride; i < n_pes; ++i, j += stride) {
      internal_putmem_nbi(&pSync[0], &pSync[0], sizeof(*pSync), j, j, wf_info);
    }
    quiet();
    pSync[0] = ROCSHMEM_SYNC_VALUE;
  } else {
    internal_amo_add<int64_t>(&pSync[0], flag_val, PE_start,
      PE_start, wf_info);
    wait_until(&pSync[0], ROCSHMEM_CMP_EQ, flag_val);
    pSync[0] = ROCSHMEM_SYNC_VALUE;
    __threadfence_system();
  }
}

__device__ void GDAContext::internal_sync(int pe, int PE_start, int stride,
    int PE_size, int64_t *pSync, ActiveWFInfo &wf_info) {
  if (PE_size < 64) {
    internal_direct_barrier(pe, PE_start, stride, PE_size, pSync, wf_info);
  } else {
    internal_atomic_barrier(pe, PE_start, stride, PE_size, pSync, wf_info);
  }
}

__device__ void GDAContext::internal_sync_wave(int pe, int PE_start,
    int stride, int PE_size, int64_t *pSync, ActiveWFInfo &wf_info) {
  if (is_thread_zero_in_wave()) {
    if (PE_size < 64) {
      internal_direct_barrier(pe, PE_start, stride, PE_size, pSync, wf_info);
    } else {
      internal_atomic_barrier(pe, PE_start, stride, PE_size, pSync, wf_info);
    }
  }
}

__device__ void GDAContext::internal_sync_wg(int pe, int PE_start, int stride,
    int PE_size, int64_t *pSync, ActiveWFInfo &wf_info) {
  __syncthreads();
  if (PE_size < 64) {
    internal_direct_barrier_wg(pe, PE_start, stride, PE_size, pSync, wf_info);
  } else {
    if (is_thread_zero_in_block()) {
      internal_atomic_barrier(pe, PE_start, stride, PE_size, pSync, wf_info);
    }
  }
  __threadfence_system();
  __syncthreads();
}

__device__ void GDAContext::sync(rocshmem_team_t team) {
  GDATeam *team_obj = reinterpret_cast<GDATeam *>(team);

  int pe = team_obj->my_pe_in_world;
  int pe_start = team_obj->tinfo_wrt_world->pe_start;
  int pe_stride = team_obj->tinfo_wrt_world->stride;
  int pe_size = team_obj->num_pes;
  long *p_sync = team_obj->barrier_pSync;

  ActiveWFInfo wf_info(ctx_id_);
  internal_sync(pe, pe_start, pe_stride, pe_size, p_sync, wf_info);
}

__device__ void GDAContext::sync_wave(rocshmem_team_t team) {
  GDATeam *team_obj = reinterpret_cast<GDATeam *>(team);

  int pe = team_obj->my_pe_in_world;
  int pe_start = team_obj->tinfo_wrt_world->pe_start;
  int pe_stride = team_obj->tinfo_wrt_world->stride;
  int pe_size = team_obj->num_pes;
  long *p_sync = team_obj->barrier_pSync;

  ActiveWFInfo wf_info(ctx_id_, ThreadScope::wave);
  internal_sync_wave(pe, pe_start, pe_stride, pe_size, p_sync, wf_info);
}

__device__ void GDAContext::sync_wg(rocshmem_team_t team) {
  GDATeam *team_obj = reinterpret_cast<GDATeam *>(team);

  int pe = team_obj->my_pe_in_world;
  int pe_start = team_obj->tinfo_wrt_world->pe_start;
  int pe_stride = team_obj->tinfo_wrt_world->stride;
  int pe_size = team_obj->num_pes;
  long *p_sync = team_obj->barrier_pSync;

  ActiveWFInfo wf_info(ctx_id_, ThreadScope::wg);
  internal_sync_wg(pe, pe_start, pe_stride, pe_size, p_sync, wf_info);
}

__device__ void GDAContext::sync_all() {
  ActiveWFInfo wf_info(ctx_id_);
  internal_sync(constmem.my_pe, 0, 1, constmem.num_pes, barrier_sync, wf_info);
}

__device__ void GDAContext::sync_all_wave() {
  ActiveWFInfo wf_info(ctx_id_, ThreadScope::wave);
  internal_sync_wave(constmem.my_pe, 0, 1, constmem.num_pes, barrier_sync, wf_info);
}

__device__ void GDAContext::sync_all_wg() {
  ActiveWFInfo wf_info(ctx_id_, ThreadScope::wg);
  internal_sync_wg(constmem.my_pe, 0, 1, constmem.num_pes, barrier_sync, wf_info);
}

__device__ void GDAContext::barrier_all() {
  quiet();
  sync_all();
}

__device__ void GDAContext::barrier_all_wave() {
  quiet();
  sync_all_wave();
}

__device__ void GDAContext::barrier_all_wg() {
  if (is_thread_zero_in_block()) {
    quiet();
  }
  sync_all_wg();
  __syncthreads();
}

__device__ void GDAContext::barrier(rocshmem_team_t team) {
  GDATeam *team_obj = reinterpret_cast<GDATeam *>(team);

  int pe = team_obj->my_pe_in_world;
  int pe_start = team_obj->tinfo_wrt_world->pe_start;
  int pe_stride = team_obj->tinfo_wrt_world->stride;
  int pe_size = team_obj->num_pes;
  long *p_sync = team_obj->barrier_pSync;

  ActiveWFInfo wf_info(ctx_id_);
  internal_quiet(wf_info);
  internal_sync(pe, pe_start, pe_stride, pe_size, p_sync, wf_info);
}

__device__ void GDAContext::barrier_wave(rocshmem_team_t team) {
  GDATeam *team_obj = reinterpret_cast<GDATeam *>(team);

  int pe = team_obj->my_pe_in_world;
  int pe_start = team_obj->tinfo_wrt_world->pe_start;
  int pe_stride = team_obj->tinfo_wrt_world->stride;
  int pe_size = team_obj->num_pes;
  long *p_sync = team_obj->barrier_pSync;

  ActiveWFInfo wf_info(ctx_id_, ThreadScope::wave);
  internal_quiet(wf_info);
  internal_sync_wave(pe, pe_start, pe_stride, pe_size, p_sync, wf_info);
}

__device__ void GDAContext::barrier_wg(rocshmem_team_t team) {
  GDATeam *team_obj = reinterpret_cast<GDATeam *>(team);

  int pe = team_obj->my_pe_in_world;
  int pe_start = team_obj->tinfo_wrt_world->pe_start;
  int pe_stride = team_obj->tinfo_wrt_world->stride;
  int pe_size = team_obj->num_pes;
  long *p_sync = team_obj->barrier_pSync;

  ActiveWFInfo wf_info(ctx_id_, ThreadScope::wg);
  if (is_thread_zero_in_block()) {
    internal_quiet(wf_info);
  }
  internal_sync_wg(pe, pe_start, pe_stride, pe_size, p_sync, wf_info);
  __syncthreads();
}

__device__ void GDAContext::internal_put_broadcastmem_wave(void *dst, const void *src,
    int nelems, int pe_root, int pe_start, int stride, int pe_size,
    ActiveWFInfo &wf_info) {  // NOLINT(runtime/int)
  if (constmem.my_pe == pe_root) {
    int finish = pe_start + stride * pe_size;
    for (int i = pe_start; i < finish; i += stride) {
      if (i != constmem.my_pe) {
        internal_putmem_nbi_wave(dst, src, nelems, i, i, wf_info);
      }
    }
    memcpy_wave<MemcpyKind::Put>(dst, const_cast<void *>(src), nelems);
  }
}

__device__ void GDAContext::internal_get_broadcastmem_wave(void *dst, const void *src,
    int nelems, int pe_root, ActiveWFInfo &wf_info) {  // NOLINT(runtime/int)
  if (constmem.my_pe != pe_root) {
    internal_getmem_wave(dst, src, nelems, pe_root, pe_root, wf_info);
  } else {
    memcpy_wave<MemcpyKind::Get>(dst, const_cast<void *>(src), nelems);
  }
}

__device__ int GDAContext::broadcastmem_wave(rocshmem_team_t team, void *dest, 
    const void* source, int nelement, int PE_root) {
  if (dest == nullptr || 
    source == nullptr || 
    team == ROCSHMEM_TEAM_INVALID)
    return ROCSHMEM_ERROR;

  GDATeam *team_obj = reinterpret_cast<GDATeam *>(team);

  int stride = team_obj->tinfo_wrt_world->stride;
  int pe_start = team_obj->tinfo_wrt_world->pe_start;
  int pe_size = team_obj->tinfo_wrt_world->size;
  long *p_sync = team_obj->bcast_pSync;

  // Passed pe_root is relative to team, convert to world root
  int pe_root_world = team_obj->get_pe_in_world(PE_root);
  internal_broadcastmem_wave(dest, source, nelement, pe_root_world, pe_start, stride,
               pe_size, p_sync);
  return ROCSHMEM_SUCCESS;
}

__device__ void GDAContext::internal_broadcastmem_wave(void *dst, const void *src,
    int nelems, int pe_root, int pe_start, int stride, int pe_size,
    long *p_sync) {  // NOLINT(runtime/int)

  ActiveWFInfo wf_info(ctx_id_, ThreadScope::wg);
  if (constmem.num_pes < 4) { //TODO: optimized for IPC
    internal_put_broadcastmem_wave(dst, src, nelems, pe_root, pe_start, stride,
      pe_size, wf_info);
  } else {
    internal_get_broadcastmem_wave(dst, src, nelems, pe_root, wf_info);
  }

  // Synchronize on completion of broadcast
  internal_sync_wave(constmem.my_pe, pe_start, stride, pe_size, p_sync, wf_info);
}

__device__ void GDAContext::internal_put_broadcastmem_wg(void *dst, const void *src,
    int nelems, int pe_root, int pe_start, int stride, int pe_size,
    ActiveWFInfo &wf_info) {  // NOLINT(runtime/int)
  if (constmem.my_pe == pe_root) {
    int finish = pe_start + stride * pe_size;
    for (int i = pe_start; i < finish; i += stride) {
      if (constmem.my_pe != i)
        internal_putmem_nbi_wg(dst, src, nelems, i, i, wf_info);
    }
    memcpy_wg<MemcpyKind::Put>(dst, const_cast<void *>(src), nelems);
  }
}

__device__ void GDAContext::internal_get_broadcastmem_wg(void *dst, const void *src, int nelems,
    int pe_root, ActiveWFInfo &wf_info) {
  if (constmem.my_pe == pe_root) {
    memcpy_wg<MemcpyKind::Put>(dst, const_cast<void *>(src), nelems);
    return;
  }

  const size_t nbytes = static_cast<size_t>(nelems);
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

__device__ void GDAContext::broadcastmem_wg(rocshmem_team_t team, void *dst,
    const void *src, int nelems, int pe_root) {
  GDATeam *team_obj = reinterpret_cast<GDATeam *>(team);

  int stride = team_obj->tinfo_wrt_world->stride;
  int pe_start = team_obj->tinfo_wrt_world->pe_start;
  int pe_size = team_obj->tinfo_wrt_world->size;
  long *p_sync = team_obj->bcast_pSync;

  // Passed pe_root is relative to team, convert to world root
  int pe_root_world = team_obj->get_pe_in_world(pe_root);
  internal_broadcastmem_wg(dst, src, nelems, pe_root_world, pe_start, stride,
               pe_size, p_sync);
}

__device__ void GDAContext::internal_broadcastmem_wg(void *dst, const void *src,
    int nelems, int pe_root, int pe_start, int stride, int pe_size,
    long *p_sync) {  // NOLINT(runtime/int)
  ActiveWFInfo wf_info(ctx_id_, ThreadScope::wg);
  if (constmem.num_pes < 4) {
    internal_put_broadcastmem_wg(dst, src, nelems, pe_root, pe_start, stride, pe_size, wf_info);
    internal_sync_wg(constmem.my_pe, pe_start, stride, pe_size, p_sync, wf_info);
    return;
  }

  // Attempt multi-proxy broadcast for large messages with multiple nodes
  if (internal_proxy_broadcast(dst, src, nelems, pe_root, pe_start,
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

  const size_t nbytes = static_cast<size_t>(nelems);
  if (constmem.my_pe == pe_root) {
    memcpy_wg<MemcpyKind::Put>(dst, const_cast<void *>(src), nbytes);
  } else if (constmem.my_pe == node_leader) {
    internal_get_broadcastmem_wg(dst, src, nelems, pe_root, wf_info);
  }

  // Node leaders must finish the inter-node phase before local IPC readers
  // consume their destination buffers.
  internal_sync_wg(constmem.my_pe, pe_start, stride, pe_size, p_sync, wf_info);

  if (constmem.my_pe != node_leader && constmem.my_pe != pe_root) {
    internal_getmem_wg(dst, dst, nbytes, node_leader, node_leader, wf_info);
  }

  internal_sync_wg(constmem.my_pe, pe_start, stride, pe_size, p_sync, wf_info);
}

__device__ bool GDAContext::internal_proxy_broadcast(void *dst, const void *src,
    int nelems, int pe_root, int pe_start, int stride, int pe_size,
    long *p_sync, ActiveWFInfo &wf_info) {  // NOLINT(runtime/int)

  constexpr size_t MIN_BYTES_PER_PROXY = 64 * 1024;
  const size_t nbytes = static_cast<size_t>(nelems);
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

}  // namespace rocshmem
