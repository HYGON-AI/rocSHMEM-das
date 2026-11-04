// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#include "rocshmem/rocshmem.hpp"
#include "context_incl.hpp"
#include "gda/context_gda_tmpl_device.hpp"
#include "util.hpp"
#include "gda/gda_team.hpp"

#if defined(USE_DAS)

namespace rocshmem {

// A wave uses one selected same-NIC proxy per IPC domain. Data is pushed
// between proxies; the remaining team members consume it through local IPC.
__device__ __noinline__ bool GDAContext::internal_broadcastmem_wave_pxn(GDATeam *team_obj, void *dst,
                                                                           const void *src, size_t bytes, int root,
                                                                           int64_t *p_sync, ActiveWFInfo &wf_info,
                                                                           int64_t &epoch) {
  const auto *info = team_obj->tinfo_wrt_world;

  if (proxy::SelectProxyCollectiveLayout(team_obj, info->pe_start, info->stride, team_obj->num_pes) ==
      proxy::ProxyCollectiveLayout::None) {
    return false;
  }

  const bool world = info->pe_start == 0 && info->stride == 1 && team_obj->num_pes == constmem.num_pes;
  const int ipc_domain_count = world ? constmem.world_proxy_ipc_domain_count : team_obj->in_team_proxy_ipc_domain_count;
  int root_first = -1;
  int local_domain = -1;

  for (int domain = 0; domain < ipc_domain_count; domain++) {
    const int first = world ? constmem.world_proxy_ipc_domain_first_pes[domain]
                            : team_obj->in_team_proxy_ipc_domain_first_pes[domain];
    const int end = world                           ? proxy::WorldProxyIpcDomainEnd(first)
                    : domain + 1 < ipc_domain_count ? team_obj->in_team_proxy_ipc_domain_first_pes[domain + 1]
                                                    : constmem.num_pes;
    if (first == constmem.ipc_first_pe) {
      local_domain = domain;
    }
    if (root >= first && root < end) {
      root_first = first;
    }
  }

  if (root_first < 0 || local_domain < 0) {
    return false;
  }

  const int qp_row = world && !constmem.world_proxy_nic_layout_available ? 0 : team_obj->pool_index_ % num_qps_per_pe;
  constexpr size_t tile_bytes = 64 * 1024;
  if (bytes == 0) {
    internal_proxy_sync_wave(team_obj, p_sync, wf_info, epoch++);
  }

  for (size_t offset = 0; offset < bytes; offset += tile_bytes) {
    const size_t count = min(tile_bytes, bytes - offset);
    char *target = static_cast<char *>(dst) + offset;
    const char *source = static_cast<const char *>(src) + offset;
    // Four existing phases per tile; every member derives the same route.
    const uint64_t operation = team_obj->proxy_data_route_sequence + team_obj->pool_index_;

    if (constmem.my_pe == root) {
      memcpy_wave<MemcpyKind::Get>(target, const_cast<char *>(source), count);
    }
    internal_proxy_sync_wave(team_obj, p_sync, wf_info, epoch++);
    for (int domain = 0; domain < ipc_domain_count; domain++) {
      const int first = world ? constmem.world_proxy_ipc_domain_first_pes[domain]
                              : team_obj->in_team_proxy_ipc_domain_first_pes[domain];
      if (first == root_first) {
        continue;
      }
      const auto route = proxy::SelectTeamProxyDataRoute(team_obj, root_first, first, operation + domain);
      assert(route.source_pe >= 0 && route.target_pe >= 0);
      if (constmem.my_pe == route.source_pe && route.source_pe != root) {
        internal_getmem_wave(target, target, count, root, root, wf_info);
        memcpy_wave<MemcpyKind::Put>(target, target, count);
      }
    }
    internal_proxy_sync_wave(team_obj, p_sync, wf_info, epoch++);
    for (int domain = 0; domain < ipc_domain_count; domain++) {
      const int first = world ? constmem.world_proxy_ipc_domain_first_pes[domain]
                              : team_obj->in_team_proxy_ipc_domain_first_pes[domain];
      if (first == root_first) {
        continue;
      }
      const auto route = proxy::SelectTeamProxyDataRoute(team_obj, root_first, first, operation + domain);
      if (constmem.my_pe == route.source_pe && is_thread_zero_in_wave()) {
        internal_putmem(target, target, count, route.target_pe, qp_row * constmem.num_pes + route.target_pe, wf_info);
      }
    }
    internal_proxy_sync_wave(team_obj, p_sync, wf_info, epoch++);

    const int local_first = world ? constmem.world_proxy_ipc_domain_first_pes[local_domain]
                                  : team_obj->in_team_proxy_ipc_domain_first_pes[local_domain];
    const int local_proxy = local_first == root_first
            ? root
            : proxy::SelectTeamProxyDataRoute(team_obj, root_first, local_first, operation + local_domain).target_pe;
    if (constmem.my_pe != local_proxy) {
      internal_getmem_wave(target, target, count, local_proxy, local_proxy, wf_info);
    }
    internal_proxy_sync_wave(team_obj, p_sync, wf_info, epoch++);

    if (is_thread_zero_in_wave()) {
      team_obj->proxy_data_route_sequence++;
    }
    __builtin_amdgcn_wave_barrier();
  }

  return true;
}

__device__ bool GDAContext::broadcastmem_wave_pxn(rocshmem_team_t team, void *dst, const void *src,
                                                                   int bytes, int root) {
  GDATeam *team_obj = reinterpret_cast<GDATeam *>(team);
  const auto *info = team_obj->tinfo_wrt_world;
  return internal_try_collective_pxn(team_obj, info->pe_start, info->stride, team_obj->num_pes, "broadcast_wave",
      [&] __device__() {
        ActiveWFInfo wf_info(ctx_id_, ThreadScope::wave);
        int64_t epoch = static_cast<int64_t>(team_obj->proxy_broadcast_sync_epoch);
        if (!internal_broadcastmem_wave_pxn(team_obj, dst, src, bytes, team_obj->get_pe_in_world(root),
                                               team_obj->bcast_pSync, wf_info, epoch)) {
          return false;
        }
        if (is_thread_zero_in_wave()) {
          team_obj->proxy_broadcast_sync_epoch = epoch;
        }
        __threadfence_system();
        __builtin_amdgcn_wave_barrier();
        return true;
      });
}

__device__ void GDAContext::internal_get_broadcastmem_wg_das(void *dst, const void *src, int nelems, int pe_root,
                                                             ActiveWFInfo &wf_info) {
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
  // message sent between IPC domains over the useful QP rows, submit every chunk first,
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
    char *dst_bytes = reinterpret_cast<char *>(dst);

    for (uint32_t rail = 0; rail < rails; rail++) {
      const size_t begin = static_cast<size_t>(rail) * chunk;
      if (begin >= nbytes) break;
      const size_t length = ((nbytes - begin) < chunk) ? (nbytes - begin) : chunk;
      const uint32_t qp_row = (static_cast<uint32_t>(blockIdx.x) + rail * active_wgs) % num_qps_per_pe;
      const uint32_t qp_index = qp_row * constmem.num_pes + pe_root;
      qps[qp_index].get_nbi(dst_bytes + begin, reinterpret_cast<const char *>(src) + begin, length, wf_info);
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

// Shared byte transport: one LDS route table for all typed broadcast APIs.
__device__ __noinline__ bool GDAContext::internal_broadcastmem_wg_pxn(GDATeam *team_obj, void *dst, const void *src,
                                                                        size_t bytes, int pe_root, int pe_start,
                                                                        int stride, int pe_size, long *p_sync,
                                                                        ActiveWFInfo &wf_info, int64_t proxy_epoch_base,
                                                                        int proxy_qp_row) {  // NOLINT(runtime/int)
  if (proxy::SelectProxyCollectiveLayout(team_obj, pe_start, stride, pe_size) == proxy::ProxyCollectiveLayout::None) {
    return false;
  }

  const auto view = proxy::GetProxyTeamView(team_obj, pe_start, stride, pe_size);
  int root_first = -1;
  int local_domain = -1;
  for (int domain = 0; domain < view.ipc_domain_count; domain++) {
    const int first = view.domain_first(domain);
    if (first == constmem.ipc_first_pe) {
      local_domain = domain;
    }
    // IPC membership, rather than gaps in partial-team domain indices,
    // determines which domain contains the root.
    if (pe_root >= first && pe_root < proxy::WorldProxyIpcDomainEnd(first)) {
      root_first = first;
    }
  }

  if (root_first < 0 || local_domain < 0) {
    LOGD_ERROR_ABORT("PXN broadcast has invalid IPC-domain membership");
  }

  const uint64_t operation = team_obj != nullptr 
                           ? team_obj->proxy_data_route_sequence + team_obj->pool_index_ 
                           : proxy_epoch_base;
  // A single protocol uses one lane for small messages and independent NIC
  // lanes for large messages. Build the table once, then reuse it in all phases.
  __shared__ int source_proxies[ROCSHMEM_BARRIER_SYNC_SIZE];
  __shared__ int target_proxies[ROCSHMEM_BARRIER_SYNC_SIZE];
  __shared__ int lane_counts[ROCSHMEM_BARRIER_SYNC_SIZE];
  __shared__ size_t segment_bytes[ROCSHMEM_BARRIER_SYNC_SIZE];
  constexpr size_t min_lane_bytes = 64 * 1024;

  if (is_thread_zero_in_block()) {
    for (int domain = 0; domain < view.ipc_domain_count; domain++) {
      const int first = view.domain_first(domain);
      if (first == root_first) {
        continue;
      }
      int groups = 0;
      int endpoint_lanes = 0;
      if (constmem.world_proxy_nic_layout_available) {
        for (int nic = 0; nic < constmem.world_proxy_nic_name_count; nic++) {
          const int sources = proxy::CountProxyDataEndpoints(team_obj,
                                     root_first, proxy::WorldProxyIpcDomainEnd(root_first), nic, -1);
          const int targets = proxy::CountProxyDataEndpoints(team_obj,
                                     first, proxy::WorldProxyIpcDomainEnd(first), nic, -1);
          groups += sources > 0 && targets > 0;
          endpoint_lanes += min(sources, targets);
        }
      }
      // Missing PE/NIC mapping retains the host-validated fixed route.
      const int useful = max(1, static_cast<int>((bytes + min_lane_bytes - 1) / min_lane_bytes));
      const int lanes = groups > 0 ? min(endpoint_lanes, useful) : 1;
      lane_counts[domain] = lanes;
      segment_bytes[domain] = ((bytes + lanes - 1) / lanes + 15) & ~size_t{15};
      int lane = 0;
      for (int round = 0; lane < lanes; round++) {
        for (int turn = 0; groups > 0 && turn < constmem.world_proxy_nic_name_count && lane < lanes; turn++) {
          const int nic = (operation + domain + turn) % constmem.world_proxy_nic_name_count;
          const int sources = proxy::CountProxyDataEndpoints(team_obj,
                                     root_first, proxy::WorldProxyIpcDomainEnd(root_first), nic, -1);
          const int targets = proxy::CountProxyDataEndpoints(team_obj,
                                     first, proxy::WorldProxyIpcDomainEnd(first), nic, -1);
          if (round >= min(sources, targets)) {
            continue;
          }
          const uint64_t ordinal = operation / constmem.world_proxy_nic_name_count + round;
          source_proxies[first + lane] = proxy::SelectProxyDataEndpoint(team_obj,
                                     root_first, proxy::WorldProxyIpcDomainEnd(root_first), nic, -1, ordinal % sources);
          target_proxies[first + lane] = proxy::SelectProxyDataEndpoint(team_obj,
                                     first, proxy::WorldProxyIpcDomainEnd(first), nic, -1, ordinal % targets);
          lane++;
        }
        if (groups == 0) {
          const auto route = proxy::SelectTeamProxyDataRoute(team_obj, root_first, first, operation);
          if (route.source_pe < 0 || route.target_pe < 0) {
            LOGD_ERROR_ABORT("PXN broadcast has no usable same-NIC data route");
          }
          source_proxies[first] = route.source_pe;
          target_proxies[first] = route.target_pe;
          lane = lanes;
        }
      }
    }
  }
  __syncthreads();

  if (constmem.my_pe == pe_root) {
    memcpy_wg<MemcpyKind::Put>(dst, const_cast<void *>(src), bytes);
  }
  // Byte chunks permit any element type without truncating a final lane.
  char *target = reinterpret_cast<char *>(dst);
  const char *source = reinterpret_cast<const char *>(src);
  for (int domain = 0; domain < view.ipc_domain_count; domain++) {
    const int first = view.domain_first(domain);
    if (first == root_first) {
      continue;
    }
    for (int lane = 0; lane < lane_counts[domain]; lane++) {
      const size_t begin = lane * segment_bytes[domain];
      if (source_proxies[first + lane] == constmem.my_pe && constmem.my_pe != pe_root) {
        internal_proxy_getmem_wg(target + begin, source + begin, min(segment_bytes[domain], bytes - begin),
                                 pe_root, 0, wf_info);
      }
    }
  }

  auto synchronize = [&] __device__(int phase) {
    if (view.is_world) {
      internal_world_proxy_sync_wg(constmem.my_pe, pe_size, p_sync, wf_info, proxy_epoch_base + phase, proxy_qp_row);
    } else {
      internal_team_proxy_sync_wg(team_obj, p_sync, wf_info, proxy_epoch_base + phase);
    }
  };

  synchronize(0);

  const int qp_row = proxy_qp_row >= 0 ? proxy_qp_row % num_qps_per_pe : static_cast<int>(blockIdx.x) % num_qps_per_pe;
  for (int domain = 0; domain < view.ipc_domain_count; domain++) {
    const int first = view.domain_first(domain);
    if (first == root_first) {
      continue;
    }
    for (int lane = 0; lane < lane_counts[domain]; lane++) {
      const size_t begin = lane * segment_bytes[domain];
      if (source_proxies[first + lane] == constmem.my_pe) {
        internal_proxy_putmem_wg(target + begin, target + begin, min(segment_bytes[domain], bytes - begin),
                                 target_proxies[first + lane], qp_row, wf_info);
      }
    }
  }
  synchronize(1);

  if (constmem.ipc_first_pe == root_first) {
    // Source proxies already hold their staged segments. Fan out from these
    // owners too, instead of concentrating every full-message IPC read on root.
    const int reference = local_domain == 0 ? 1 : 0;
    const int first = view.domain_first(reference);
    if (constmem.my_pe != pe_root) {
      for (int lane = 0; lane < lane_counts[reference]; lane++) {
        const int owner = source_proxies[first + lane];
        const size_t begin = lane * segment_bytes[reference];
        if (constmem.my_pe != owner) {
          internal_proxy_getmem_wg(target + begin, target + begin, min(segment_bytes[reference], bytes - begin),
                                 owner, 0, wf_info);
        }
      }
    }
  } else {
    const int first = view.domain_first(local_domain);
    for (int lane = 0; lane < lane_counts[local_domain]; lane++) {
      const int local_proxy = target_proxies[first + lane];
      const size_t begin = lane * segment_bytes[local_domain];
      if (constmem.my_pe != local_proxy) {
        internal_proxy_getmem_wg(target + begin, target + begin, min(segment_bytes[local_domain], bytes - begin),
                                 local_proxy, 0, wf_info);
      }
    }
  }
  synchronize(2);
  if (team_obj != nullptr && is_thread_zero_in_block()) {
    team_obj->proxy_data_route_sequence++;
  }
  __syncthreads();
  return true;
}

// Preserve the original unrestricted MLX5 broadcast algorithm.
__device__ void GDAContext::internal_get_broadcastmem_wg_nopxn(void *dst, const void *src, int nelems, int pe_root,
                                                               ActiveWFInfo &wf_info) {
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
    char *dst_bytes = reinterpret_cast<char *>(dst);

    for (uint32_t rail = 0; rail < rails; rail++) {
      const size_t begin = static_cast<size_t>(rail) * chunk;
      if (begin >= nbytes) break;
      const size_t length = ((nbytes - begin) < chunk) ? (nbytes - begin) : chunk;
      const uint32_t qp_row = (static_cast<uint32_t>(blockIdx.x) + rail * active_wgs) % num_qps_per_pe;
      const uint32_t qp_index = qp_row * constmem.num_pes + pe_root;
      qps[qp_index].get_nbi(dst_bytes + begin, reinterpret_cast<const char *>(src) + begin, length, wf_info);
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

__device__ void GDAContext::internal_broadcastmem_wg_nopxn(void *dst, const void *src, int nelems, int pe_root,
                                                           int pe_start, int stride, int pe_size,
                                                           long *p_sync) {  // NOLINT(runtime/int)
  ActiveWFInfo wf_info(ctx_id_, ThreadScope::wg);
  if (constmem.num_pes < 4) {
    internal_put_broadcastmem_wg(dst, src, nelems, pe_root, pe_start, stride, pe_size, wf_info);
    internal_sync_wg(constmem.my_pe, pe_start, stride, pe_size, p_sync, wf_info);
    return;
  }

  // Attempt multi-proxy broadcast for large messages with multiple nodes
  if (internal_broadcastmem_wg_proxy_nopxn(dst, src, nelems, pe_root, pe_start, stride, pe_size, p_sync, wf_info)) {
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
    if (peer == constmem.my_pe || ipcImpl_.isIpcAvailable(constmem.my_pe, peer, &local_pe)) {
      node_leader = peer;
      break;
    }
  }

  const size_t nbytes = static_cast<size_t>(nelems);
  if (constmem.my_pe == pe_root) {
    memcpy_wg<MemcpyKind::Put>(dst, const_cast<void *>(src), nbytes);
  } else if (constmem.my_pe == node_leader) {
    internal_get_broadcastmem_wg_nopxn(dst, src, nelems, pe_root, wf_info);
  }

  // Node leaders must finish the inter-node phase before local IPC readers
  // consume their destination buffers.
  internal_sync_wg(constmem.my_pe, pe_start, stride, pe_size, p_sync, wf_info);

  if (constmem.my_pe != node_leader && constmem.my_pe != pe_root) {
    internal_getmem_wg(dst, dst, nbytes, node_leader, node_leader, wf_info);
  }

  internal_sync_wg(constmem.my_pe, pe_start, stride, pe_size, p_sync, wf_info);
}

__device__ bool GDAContext::internal_broadcastmem_wg_proxy_nopxn(void *dst, const void *src, int nelems, int pe_root,
                                                              int pe_start, int stride, int pe_size, long *p_sync,
                                                              ActiveWFInfo &wf_info) {  // NOLINT(runtime/int)

  constexpr size_t MIN_BYTES_PER_PROXY = 64 * 1024;
  const size_t nbytes = static_cast<size_t>(nelems);
  const int local_shm_size = constmem.ipc_shm_size;

  // Multi-proxy hierarchical broadcast.  One PE per local GPU/NIC transfers
  // one segment between nodes; IPC stages the root data and fans the segments
  // out locally.  Restrict this path to complete, contiguous node groups so
  // proxy slot N names the same local GPU/NIC on every node.
  const bool use_proxy_bcast = nbytes >= kHierarchicalMinBytes && stride == 1 && local_shm_size > 1 &&
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
      internal_getmem_wg(dst_bytes + begin, src_bytes + begin, length, pe_root, pe_root, wf_info);
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
      internal_getmem_wg(dst_bytes + begin, dst_bytes + begin, length, source_proxy, qp_index, wf_info);
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
    internal_getmem_wg(dst_bytes + begin, dst_bytes + begin, length, local_proxy, local_proxy, wf_info);
  }

  internal_sync_wg(constmem.my_pe, pe_start, stride, pe_size, p_sync, wf_info);

  return true;  // Proxy broadcast completed successfully
}

__device__ void GDAContext::broadcastmem_wg_das(rocshmem_team_t team, void *dst, const void *src, int nelems,
                                                int pe_root) {
  auto *team_obj = reinterpret_cast<GDATeam *>(team);
  const auto *info = team_obj->tinfo_wrt_world;
  const int root = team_obj->get_pe_in_world(pe_root);
  long *p_sync = team_obj->bcast_pSync;

  if (internal_try_collective_pxn(team_obj, info->pe_start, info->stride, team_obj->num_pes, "broadcast",
      [&] __device__() {
        if (proxy::SelectProxyCollectiveLayout(team_obj, info->pe_start, info->stride, team_obj->num_pes) ==
            proxy::ProxyCollectiveLayout::None) {
          return false;
        }

        // This collective dispatcher publishes input and owns epoch advancement. The proxy
        // implementation below only stages, transfers and distributes the payload.
        internal_try_sync_pxn(team_obj, team_obj->barrier_pSync, ThreadScope::wg);
        ActiveWFInfo wf_info(ctx_id_, ThreadScope::wg);
        const int64_t epoch = static_cast<int64_t>(team_obj->proxy_broadcast_sync_epoch);
        if (nelems == 0) {
          internal_proxy_sync_wg(team_obj, p_sync, wf_info, epoch);
        } else if (!internal_broadcast_wg_pxn(team_obj, static_cast<char *>(dst),
                                                        static_cast<const char *>(src), nelems, root,
                                                        info->pe_start, info->stride, team_obj->num_pes, p_sync,
                                                        wf_info, epoch, team_obj->pool_index_ % num_qps_per_pe)) {
          LOGD_ERROR_ABORT("PXN broadcast has no usable proxy route");
        }
        if (is_thread_zero_in_block()) {
          team_obj->proxy_broadcast_sync_epoch =
              epoch + (nelems == 0 ? 1 : proxy::TeamBroadcastPhaseCount(team_obj));
        }
        __syncthreads();
        return true;
      })) {
    return;
  }

  internal_broadcastmem_wg_nopxn(dst, src, nelems, root, info->pe_start, info->stride, team_obj->num_pes, p_sync);
}

__device__ void GDAContext::internal_broadcastmem_wg_das(void *dst, const void *src, int nelems, int pe_root,
                                                         int pe_start, int stride, int pe_size, long *p_sync,
                                                         GDATeam *team_obj) {  // NOLINT(runtime/int)
  // Typed and byte APIs share the same task routing and completion protocol.
  internal_broadcast_wg_dispatch(static_cast<char *>(dst), static_cast<const char *>(src), nelems, pe_root, pe_start, stride,
                        pe_size, p_sync, 1, -1, team_obj);
}

}  // namespace rocshmem

#endif  // USE_DAS
