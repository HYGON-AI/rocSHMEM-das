// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#include "rocshmem/rocshmem.hpp"
#include "context_incl.hpp"
#include "gda/context_gda_tmpl_device.hpp"
#include "util.hpp"
#include "gda/gda_team.hpp"

#if defined(USE_DAS)

namespace rocshmem {

namespace {

// Payload policy is private to the non-PXN proxy broadcast below. Each call
// is bounded by its caller; native helpers handle alignment and byte tails.
__device__ __forceinline__ void broadcast_proxy_copy_root_wg(char *target, const char *source, size_t length) {
  // Exact in-place root input is already its complete output. Avoid
  // rewriting it while other root-node proxies read the same src.
  if (length == 0 || target == source) return;
#if defined(__HIP_DEVICE_COMPILE__) && ROCSHMEM_ARCH_HYGON && (defined(__gfx936__) || defined(__gfx938__))
  if (((reinterpret_cast<uintptr_t>(target) | reinterpret_cast<uintptr_t>(source)) & 15) == 0) {
    const size_t tid = static_cast<size_t>(get_flat_block_id());
    const size_t threads = static_cast<size_t>(get_flat_block_size());
    const size_t group_bytes = 4 * threads * sizeof(int4);
    const size_t bulk = (length / group_bytes) * group_bytes;
    const auto *input = reinterpret_cast<const int4 *>(source);
    auto *output = reinterpret_cast<int4 *>(target);
    for (size_t offset = 0; offset < bulk; offset += group_bytes) {
      const size_t first = offset / sizeof(int4) + tid;
      // Four independent local vectors per thread. This source-level
      // grouping still requires final ISA/resource and runtime checks.
      const int4 v0 = input[first];
      const int4 v1 = input[first + threads];
      const int4 v2 = input[first + 2 * threads];
      const int4 v3 = input[first + 3 * threads];
      // Match the original Put system-visible store policy, issuing
      // all four stores together; the stage phase fences every writer.
      asm volatile(
          "flat_store_dwordx4 %0, %4, glc slc\n"
          "flat_store_dwordx4 %1, %5, glc slc\n"
          "flat_store_dwordx4 %2, %6, glc slc\n"
          "flat_store_dwordx4 %3, %7, glc slc"
          :
          : "v"(output + first), "v"(output + first + threads),
            "v"(output + first + 2 * threads), "v"(output + first + 3 * threads),
            "v"(__builtin_bit_cast(__int128_t, v0)), "v"(__builtin_bit_cast(__int128_t, v1)),
            "v"(__builtin_bit_cast(__int128_t, v2)), "v"(__builtin_bit_cast(__int128_t, v3))
          : "memory");
    }
    if (bulk < length) {
      memcpy_wg<MemcpyKind::Put>(target + bulk, const_cast<char *>(source + bulk), length - bulk);
    }
    return;
  }
#endif
  memcpy_wg<MemcpyKind::Put>(target, const_cast<char *>(source), length);
}

// Called only by complete consumer waves. Return the grouped prefix length
// so the caller can use its native IPC helper for alignment and byte tails.
__device__ __forceinline__ size_t broadcast_proxy_copy_ipc_wave(char *target, const char *source, size_t length) {
#if defined(__HIP_DEVICE_COMPILE__) && ROCSHMEM_ARCH_HYGON && (defined(__gfx936__) || defined(__gfx938__))
  static_assert(WF_SIZE == 64, "Immediate offsets require 64-lane waves");
  const int lane = get_flat_block_id() % WF_SIZE;
  if (((reinterpret_cast<uintptr_t>(target) | reinterpret_cast<uintptr_t>(source)) & 15) == 0) {
    constexpr size_t wave_bytes = WF_SIZE * 16;
    constexpr size_t group_bytes = 8 * wave_bytes;
    const size_t bulk = (length / group_bytes) * group_bytes;
    for (size_t offset = 0; offset < bulk; offset += group_bytes) {
      const size_t first = offset + static_cast<size_t>(lane) * 16;
      const char *base0 = source + first;
      const char *base1 = base0 + 4 * wave_bytes;
      __int128_t v0, v1, v2, v3, v4, v5, v6, v7;
      // Early-clobber keeps both address pairs distinct from every
      // output tuple until all eight instructions have read them.
      asm volatile(
          "flat_load_dwordx4 %0, %8 offset:0 glc slc\n"
          "flat_load_dwordx4 %1, %8 offset:1024 glc slc\n"
          "flat_load_dwordx4 %2, %8 offset:2048 glc slc\n"
          "flat_load_dwordx4 %3, %8 offset:3072 glc slc\n"
          "flat_load_dwordx4 %4, %9 offset:0 glc slc\n"
          "flat_load_dwordx4 %5, %9 offset:1024 glc slc\n"
          "flat_load_dwordx4 %6, %9 offset:2048 glc slc\n"
          "flat_load_dwordx4 %7, %9 offset:3072 glc slc\n"
          "s_waitcnt vmcnt(0)"
          : "=&v"(v0), "=&v"(v1), "=&v"(v2), "=&v"(v3), "=&v"(v4), "=&v"(v5), "=&v"(v6), "=&v"(v7)
          : "v"(base0), "v"(base1)
          : "memory");
      *reinterpret_cast<int4 *>(target + first + 0 * wave_bytes) = __builtin_bit_cast(int4, v0);
      *reinterpret_cast<int4 *>(target + first + 1 * wave_bytes) = __builtin_bit_cast(int4, v1);
      *reinterpret_cast<int4 *>(target + first + 2 * wave_bytes) = __builtin_bit_cast(int4, v2);
      *reinterpret_cast<int4 *>(target + first + 3 * wave_bytes) = __builtin_bit_cast(int4, v3);
      *reinterpret_cast<int4 *>(target + first + 4 * wave_bytes) = __builtin_bit_cast(int4, v4);
      *reinterpret_cast<int4 *>(target + first + 5 * wave_bytes) = __builtin_bit_cast(int4, v5);
      *reinterpret_cast<int4 *>(target + first + 6 * wave_bytes) = __builtin_bit_cast(int4, v6);
      *reinterpret_cast<int4 *>(target + first + 7 * wave_bytes) = __builtin_bit_cast(int4, v7);
    }
    return bulk;
  }
#endif
  return 0;
}

struct ProxyBroadcastSlots {
  // 0..246 are payload-ready flags written by a segment's receiving GPU.
  // 247 caches the team vote for the lifetime of this team's ready pool slot.
  // 248 is a source owner's GPU flag or its remote counterpart's NIC inbox.
  // 249 holds the registered source notice, or a receiving GPU's IPC relay.
  // 250 supplies a registered Done epoch or indexed-barrier signal.
  // 251 publishes the root's input.
  // 252 is each PE's release inbox; 253 is its local completion ACK.
  // 254/255 are alternating Done inboxes, so a fast peer may advance once.
  // Bootstrap borrows 248..250 before the first epoch; all network sources
  // stay in registered storage until the issuing QP has completed the write.
  static constexpr int topology_slot = 247;
  static constexpr size_t payload_capacity = topology_slot;
  static constexpr int source_ready_slot = 248;
  static constexpr int notification_source_slot = 249;
  static constexpr int done_source_slot = 250;
  static constexpr int root_input_slot = 251;
  static constexpr int completion_release_slot = 252;
  static constexpr int completion_ack_slot = 253;
  static constexpr int first_done_slot = 254;
  static_assert(first_done_slot + 2 <= ROCSHMEM_BARRIER_SYNC_SIZE, "broadcast ready table too small");
};

constexpr size_t kProxyBroadcastLargeThreshold = 1024 * 1024;
constexpr size_t kProxyBroadcastCopyCallLimit = size_t{1} << 30;

// Invocation references and team properties; no shared or persistent storage.
struct ProxyBroadcastTeamState {
  int pe_start, stride, pe_size, pe_root;
  long *p_sync;
  GDATeam *ready_team;
  long *ready;
  int local_shm_size;
  uint32_t num_qps_per_pe;
  QueuePair *qps;
  IpcImpl &ipcImpl_;
  ActiveWFInfo &wf_info;
};

struct ProxyBroadcastPayload {
  size_t nbytes;
  char *dst_bytes;
  const char *src_bytes;
  int domain_count, local_slot, root_node_first, root_slot;
  bool on_root_node;
  int proxies;
  size_t chunk;
  long epoch;
};

struct ProxyBroadcastThreads {
  int tid, wave, lane, full_waves;
  bool parallel;
  int consumer, consumers;
  bool consume;
};

struct ProxyBroadcastLargeGeometry {
  int payload_proxies, source_count;
  size_t segment_bytes;
  bool power2_geometry;
  size_t global_tile_bytes;
  unsigned tile_shift, segment_shift;
  size_t global_tiles, pieces_per_segment;
};

struct ProxyBroadcastSegmentGeometry {
  bool fine_tiles;
  size_t tile_bytes, tiles;
  bool root_early_ready;
};

enum class ProxyBroadcastPhaseKind { PairedDone, LeaderSync };
enum class ProxyBroadcastNoticeOp { Post, Drain };

// Participants: thread zero of each indexed-barrier member only; no WG barrier.
template <typename NativeOps>
__device__ __forceinline__ void proxy_broadcast_indexed_sync_leader(
    const ProxyBroadcastTeamState &s, const NativeOps &ops, int member_stride, int members) {
  constexpr long signal = 1;
  __hip_atomic_store(&s.ready[ProxyBroadcastSlots::done_source_slot], signal, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
  __threadfence_system();
  if (constmem.my_pe == s.pe_start) {
    for (int rank = 1; rank < members; ++rank) {
      while (__hip_atomic_load(&s.p_sync[rank], __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) != signal) {}
      __hip_atomic_store(&s.p_sync[rank], 0L, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
    }
    __threadfence_system();
    for (int rank = 1; rank < members; ++rank) {
      const int peer = s.pe_start + rank * member_stride;
      ops.put(&s.p_sync[0], &s.ready[ProxyBroadcastSlots::done_source_slot], sizeof(long), peer, peer);
    }
  } else {
    const int rank = (constmem.my_pe - s.pe_start) / member_stride;
    ops.put(&s.p_sync[rank], &s.ready[ProxyBroadcastSlots::done_source_slot], sizeof(long), s.pe_start, s.pe_start);
    __threadfence_system();
    while (__hip_atomic_load(&s.p_sync[0], __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) != signal) {}
    __hip_atomic_store(&s.p_sync[0], 0L, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
  }
  // Complete this lane's IPC releases or local inbox reset before reuse.
  __threadfence_system();
}

// Participants: every thread of every team WG; indexed work remains thread zero only.
template <typename NativeOps>
__device__ __forceinline__ void proxy_broadcast_team_sync_wg(
    const ProxyBroadcastTeamState &s, const NativeOps &ops) {
  if (s.pe_size < 64) {
    ops.sync_wg(constmem.my_pe, s.pe_start, 1, s.pe_size, s.p_sync);
    return;
  }
  __syncthreads();
  __threadfence_system();
  __syncthreads();
  if (is_thread_zero_in_block()) proxy_broadcast_indexed_sync_leader(s, ops, 1, s.pe_size);
  __syncthreads();
  __threadfence_system();
  __syncthreads();
}

// Participants: every team WG thread, including unsupported GPUs; both bootstrap joins precede rejection.
template <typename NativeOps>
__device__ __forceinline__ bool proxy_broadcast_validate_team_wg(
    const ProxyBroadcastTeamState &s, const NativeOps &ops) {
  #if defined(__HIP_DEVICE_COMPILE__) && ROCSHMEM_ARCH_HYGON && (defined(__gfx936__) || defined(__gfx938__))
    constexpr bool has_pipeline_copy_helpers = true;
  #else
    constexpr bool has_pipeline_copy_helpers = false;
  #endif

  // A nonzero counter follows positive certification and the final WG join.
  // Slot 247 is immutable and reused, not current payload publication.
  // Zero (including wraparound) retains the conservative original read.
  long width;
  if (s.ready_team->bcast_sequence_number == 0) {
    width = __hip_atomic_load(&s.ready[ProxyBroadcastSlots::topology_slot], __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
  } else {
    width = __hip_atomic_load(&s.ready[ProxyBroadcastSlots::topology_slot], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  }
  if (width == 0) {
    // Each PE votes on its actual IPC membership. Stride one and the
    // self-slot check place it in a complete team-aligned interval. If all
    // widths match, those intervals partition the team into equal domains.
    // Slot-based non-PXN QPs do not require the PXN WORLD routing tables.
    if (is_thread_zero_in_block()) {
      const int first = constmem.ipc_first_pe;
      int self_slot = -1;
      // An unsupported GPU votes no and still joins both bootstrap barriers.
      long vote = -1;
      if (has_pipeline_copy_helpers &&
          s.local_shm_size > 1 && s.local_shm_size <= static_cast<int>(ProxyBroadcastSlots::payload_capacity) &&
          constmem.ipc_stride == 1 && first >= s.pe_start &&
          (first - s.pe_start) % s.local_shm_size == 0 && s.pe_size % s.local_shm_size == 0 &&
          first + s.local_shm_size <= s.pe_start + s.pe_size && s.pe_size > s.local_shm_size &&
          constmem.gda_provider == GDAProvider::MLX5 && warpSize == WF_SIZE &&
          s.ipcImpl_.isIpcAvailable(constmem.my_pe, constmem.my_pe, &self_slot) &&
          self_slot == constmem.my_pe - first) {
        vote = s.local_shm_size;
      }
      __hip_atomic_store(&s.ready[ProxyBroadcastSlots::source_ready_slot], vote, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
    }
    proxy_broadcast_team_sync_wg(s, ops);
    if (constmem.my_pe == s.pe_start && is_thread_zero_in_block()) {
      long decision = __hip_atomic_load(&s.ready[ProxyBroadcastSlots::source_ready_slot], __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
      for (int peer = s.pe_start + 1; decision > 0 && peer < s.pe_start + s.pe_size; ++peer) {
        ops.get(&s.ready[ProxyBroadcastSlots::notification_source_slot], &s.ready[ProxyBroadcastSlots::source_ready_slot],
                        sizeof(long), peer, peer);
        if (__hip_atomic_load(&s.ready[ProxyBroadcastSlots::notification_source_slot], __ATOMIC_ACQUIRE,
                              __HIP_MEMORY_SCOPE_SYSTEM) != decision) decision = -1;
      }
      __hip_atomic_store(&s.ready[ProxyBroadcastSlots::done_source_slot], decision, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
      for (int peer = s.pe_start + 1; peer < s.pe_start + s.pe_size; ++peer) {
        ops.put(&s.ready[ProxyBroadcastSlots::topology_slot], &s.ready[ProxyBroadcastSlots::done_source_slot], sizeof(long), peer, peer);
      }
      __hip_atomic_store(&s.ready[ProxyBroadcastSlots::topology_slot], decision, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
    }
    proxy_broadcast_team_sync_wg(s, ops);
    width = __hip_atomic_load(&s.ready[ProxyBroadcastSlots::topology_slot], __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
  }
  // A positive team vote certifies this PE's immutable IPC width.
  if (width <= 0) return false;

  return true;
}

// Participants: any thread; pure per-invocation geometry, after positive width validation.
__device__ __forceinline__ ProxyBroadcastPayload proxy_broadcast_make_payload(
    const ProxyBroadcastTeamState &s, void *dst, const void *src, size_t nbytes, bool pipeline) {
  const int domain_count = s.pe_size / s.local_shm_size;
  const int local_slot = constmem.my_pe - constmem.ipc_first_pe;
  const int root_node_first = s.pe_start + ((s.pe_root - s.pe_start) / s.local_shm_size) * s.local_shm_size;
  const int root_slot = s.pe_root - root_node_first;
  const bool on_root_node = constmem.ipc_first_pe == root_node_first;
  const size_t segment_target = pipeline ? 32 * 1024 : 64 * 1024;
  const int proxies = min(s.local_shm_size, static_cast<int>((nbytes + segment_target - 1) / segment_target));
  const size_t chunk = (nbytes + proxies - 1) / proxies;
  char *dst_bytes = reinterpret_cast<char *>(dst);
  const char *src_bytes = reinterpret_cast<const char *>(src);
  return {nbytes, dst_bytes, src_bytes, domain_count, local_slot, root_node_first, root_slot,
          on_root_node, proxies, chunk, 0};
}

// Participants: every WG thread on the existing PXN fallback branch; preserve all three team joins.
template <typename NativeOps>
__device__ __forceinline__ void proxy_broadcast_legacy_wg(
    const ProxyBroadcastTeamState &s, const ProxyBroadcastPayload &p, const NativeOps &ops) {
  if (constmem.my_pe == s.pe_root) {
    memcpy_wg<MemcpyKind::Put>(p.dst_bytes, const_cast<char *>(p.src_bytes), p.nbytes);
  } else if (p.on_root_node && p.local_slot < p.proxies) {
    const size_t begin = static_cast<size_t>(p.local_slot) * p.chunk;
    if (begin < p.nbytes) {
      ops.get_wg(p.dst_bytes + begin, p.src_bytes + begin, min(p.chunk, p.nbytes - begin), s.pe_root, s.pe_root);
    }
  }
  ops.sync_wg(constmem.my_pe, s.pe_start, s.stride, s.pe_size, s.p_sync);
  if (!p.on_root_node && p.local_slot < p.proxies) {
    const size_t begin = static_cast<size_t>(p.local_slot) * p.chunk;
    if (begin < p.nbytes) {
      const size_t length = min(p.chunk, p.nbytes - begin);
      const int source_proxy = p.root_node_first + p.local_slot;
      const uint32_t qp_row = static_cast<uint32_t>(blockIdx.x) % s.num_qps_per_pe;
      const int qp_index = qp_row * constmem.num_pes + source_proxy;
      ops.get_wg(p.dst_bytes + begin, p.dst_bytes + begin, length, source_proxy, qp_index);
    }
  }

  ops.sync_wg(constmem.my_pe, s.pe_start, s.stride, s.pe_size, s.p_sync);
  for (int slot = 0; slot < p.proxies; ++slot) {
    if (slot == p.local_slot) continue;
    const size_t begin = static_cast<size_t>(slot) * p.chunk;
    if (begin >= p.nbytes) break;
    const int peer = constmem.ipc_first_pe + slot;
    ops.get_wg(p.dst_bytes + begin, p.dst_bytes + begin, min(p.chunk, p.nbytes - begin), peer, peer);
  }
  ops.sync_wg(constmem.my_pe, s.pe_start, s.stride, s.pe_size, s.p_sync);
}

// Participants: every WG thread; the first epoch joins the team before and after cooperative clearing.
template <typename NativeOps>
__device__ __forceinline__ void proxy_broadcast_begin_epoch_wg(
    const ProxyBroadcastTeamState &s, const ProxyBroadcastPayload &p, const NativeOps &ops) {
  if (p.epoch == 1) {
    proxy_broadcast_team_sync_wg(s, ops);
    long zero = 0;
    for (int i = get_flat_block_id(); i < ROCSHMEM_BARRIER_SYNC_SIZE; i += get_flat_block_size()) {
      if (i != ProxyBroadcastSlots::topology_slot) {
        put_asm(reinterpret_cast<uint8_t *>(&zero), reinterpret_cast<uint8_t *>(&s.ready[i]), sizeof(zero));
      }
    }
    __threadfence_system();
    proxy_broadcast_team_sync_wg(s, ops);
  }
}

// Participants: each thread; complete waves alone become parallel consumers.
__device__ __forceinline__ ProxyBroadcastThreads proxy_broadcast_make_threads() {
  const int tid = get_flat_block_id();
  const int wave = tid / WF_SIZE;
  const int lane = tid % WF_SIZE;
  const int full_waves = get_flat_block_size() / WF_SIZE;
  const bool parallel = full_waves >= 2;
  const int consumer = parallel ? wave - 1 : 0;
  const int consumers = parallel ? full_waves - 1 : 1;
  const bool consume = !parallel || (wave > 0 && wave < full_waves);
  return {tid, wave, lane, full_waves, parallel, consumer, consumers, consume};
}

// Participants: every WG thread on every PE, after local producers and consumers join.
template <typename NativeOps>
__device__ __forceinline__ void proxy_broadcast_finish_wg(
    const ProxyBroadcastTeamState &s, const ProxyBroadcastPayload &p, const ProxyBroadcastThreads &w, const
    NativeOps &ops, ProxyBroadcastPhaseKind phase) {
  // Every writer fences its own completed work before thread zero publishes
  // this PE's ACK. A thread-zero fence alone does not publish other writers.
  __syncthreads();
  __threadfence_system();
  __syncthreads();
  if (is_thread_zero_in_block()) {
    if (phase == ProxyBroadcastPhaseKind::PairedDone) {
      // This PE's producers and IPC readers have all joined and fenced.
      // Pair the same local slot across the two domains before local ACK.
      const size_t done_slot = ProxyBroadcastSlots::first_done_slot + static_cast<size_t>(p.epoch & 1);
      long done_epoch = p.epoch;
      put_asm(reinterpret_cast<uint8_t *>(&done_epoch),
              reinterpret_cast<uint8_t *>(&s.ready[ProxyBroadcastSlots::done_source_slot]), sizeof(done_epoch));
      __threadfence_system();
      const int counterpart = constmem.ipc_first_pe == s.pe_start
                                  ? s.pe_start + s.local_shm_size + p.local_slot : s.pe_start + p.local_slot;
      const uint32_t base_row = static_cast<uint32_t>(blockIdx.x) % s.num_qps_per_pe;
      const uint32_t done_qp = base_row * constmem.num_pes + counterpart;
      s.qps[done_qp].put_nbi(&s.ready[done_slot], &s.ready[ProxyBroadcastSlots::done_source_slot], sizeof(long), s.wf_info);
      // Even an inline eight-byte write must finish on exactly this QP.
      s.qps[done_qp].quiet(s.wf_info);
      // Two slots preserve an unconsumed epoch if the peer advances once.
      while (__hip_atomic_load(&s.ready[done_slot], __ATOMIC_ACQUIRE,
                                __HIP_MEMORY_SCOPE_SYSTEM) != p.epoch) {}
    }
    __hip_atomic_store(&s.ready[ProxyBroadcastSlots::completion_ack_slot], p.epoch, __ATOMIC_RELEASE,
                       __HIP_MEMORY_SCOPE_SYSTEM);
  }
  if (p.local_slot == 0) {
    const int leaders = max(w.full_waves, 1);
    if (w.wave < leaders && w.lane == 0) {
      for (int slot = w.wave; slot < s.local_shm_size; slot += leaders) {
        auto *ack = reinterpret_cast<long *>(ops.local_ptr(&s.ready[ProxyBroadcastSlots::completion_ack_slot], slot));
        while (__hip_atomic_load(ack, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) != p.epoch) {}
      }
    }
    __syncthreads();
    if (phase == ProxyBroadcastPhaseKind::LeaderSync) {
      if (w.tid == 0) proxy_broadcast_indexed_sync_leader(s, ops, s.local_shm_size, p.domain_count);
      __syncthreads();
    }
    if (w.wave < leaders && w.lane == 0) {
      long release_epoch = p.epoch;
      __threadfence_system();
      for (int slot = w.wave; slot < s.local_shm_size; slot += leaders) {
        auto *inbox = reinterpret_cast<uint8_t *>(ops.local_ptr(&s.ready[ProxyBroadcastSlots::completion_release_slot], slot));
        put_asm(reinterpret_cast<uint8_t *>(&release_epoch), inbox, sizeof(release_epoch));
      }
      __threadfence_system();
    }
  } else if (w.tid == 0) {
    while (__hip_atomic_load(&s.ready[ProxyBroadcastSlots::completion_release_slot], __ATOMIC_ACQUIRE,
                             __HIP_MEMORY_SCOPE_SYSTEM) != p.epoch) {}
  }
  __syncthreads();
  if (w.tid == 0) ++s.ready_team->bcast_sequence_number;
  __syncthreads();
}

// Participants: the original source-owner thread zero only; exact Post/Drain QPs.
__device__ __forceinline__ void proxy_broadcast_notify_domains(
    const ProxyBroadcastTeamState &s, const ProxyBroadcastPayload &p, ProxyBroadcastNoticeOp op) {
  const uint32_t row = static_cast<uint32_t>(blockIdx.x) % s.num_qps_per_pe;
  if (op == ProxyBroadcastNoticeOp::Post) {
    long notice = p.epoch;
    put_asm(reinterpret_cast<uint8_t *>(&notice),
            reinterpret_cast<uint8_t *>(&s.ready[ProxyBroadcastSlots::notification_source_slot]), sizeof(notice));
    __threadfence_system();
  }
  for (int first = s.pe_start; first < s.pe_start + s.pe_size; first += s.local_shm_size) {
    if (first == p.root_node_first) continue;
    const uint32_t qp = row * constmem.num_pes + first + p.local_slot;
    if (op == ProxyBroadcastNoticeOp::Post) {
      s.qps[qp].put_nbi(&s.ready[ProxyBroadcastSlots::source_ready_slot], &s.ready[ProxyBroadcastSlots::notification_source_slot], sizeof(long), s.wf_info);
    } else {
      s.qps[qp].quiet(s.wf_info);
    }
  }
}

// Participants: parallel: one complete consumer wave; nonparallel: every WG thread.
__device__ __forceinline__ void proxy_broadcast_wait_consumer(
    const ProxyBroadcastPayload &p, const ProxyBroadcastThreads &w, long *flag) {
  if (w.parallel) {
    int arrived = 0;
    do {
      if (w.lane == 0) arrived = __hip_atomic_load(flag, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) == p.epoch;
    } while (!__shfl(arrived, 0));
  } else {
    if (w.tid == 0) {
      while (__hip_atomic_load(flag, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) != p.epoch) {}
    }
    __syncthreads();
  }
}

// Participants: every WG thread; remote domains do nothing, root-domain joins stay inside this helper.
template <typename NativeOps>
__device__ __forceinline__ void proxy_broadcast_publish_root_input_wg(
    const ProxyBroadcastTeamState &s, const ProxyBroadcastPayload &p, const NativeOps &ops) {
  if (p.on_root_node) {
    if (constmem.my_pe == s.pe_root) {
      // Publish all WG writers' input before the root's copy or any source
      // proxy GET. Thread zero alone cannot fence the other input writers.
      __syncthreads();
      __threadfence_system();
      __syncthreads();
      if (is_thread_zero_in_block()) {
        long input_epoch = p.epoch;
        put_asm(reinterpret_cast<uint8_t *>(&input_epoch),
                reinterpret_cast<uint8_t *>(&s.ready[ProxyBroadcastSlots::root_input_slot]), sizeof(input_epoch));
        __threadfence_system();
      }
      __syncthreads();
    } else {
      if (is_thread_zero_in_block()) {
        auto *input_ready = reinterpret_cast<long *>(ops.local_ptr(&s.ready[ProxyBroadcastSlots::root_input_slot], p.root_slot));
        while (__hip_atomic_load(input_ready, __ATOMIC_ACQUIRE,
                                  __HIP_MEMORY_SCOPE_SYSTEM) != p.epoch) {}
      }
      __syncthreads();
    }
  }
}

// Participants: every WG thread; bounded calls retain zero/exact-in-place skips.
__device__ __forceinline__ void proxy_broadcast_copy_root_range_wg(
    char *target, const char *source, size_t length) {
  // Exact in-place root input is already its complete output. Avoid
  // rewriting it while other root-node proxies read the same src.
  if (length == 0 || target == source) return;
  // The existing helper handles alignment and byte tails.
  for (size_t offset = 0; offset < length;) {
    const size_t count = min(kProxyBroadcastCopyCallLimit, length - offset);
    broadcast_proxy_copy_root_wg(target + offset, source + offset, count);
    offset += count;
  }
}

// Participants: parallel: complete consumer wave; nonparallel: every WG thread.
__device__ __forceinline__ void proxy_broadcast_copy_ipc_tile(
    const ProxyBroadcastTeamState &s, const ProxyBroadcastThreads &w, char *target, char *source, size_t
    length, int slot) {
  // Each consumer is a complete wave; without parallelism the whole WG copies.
  for (size_t offset = 0; offset < length;) {
    const size_t count = min(kProxyBroadcastCopyCallLimit, length - offset);
    if (w.parallel) {
      const size_t bulk = broadcast_proxy_copy_ipc_wave(target + offset, source + offset, count);
      if (bulk < count) {
        s.ipcImpl_.ipcCopy_wave<MemcpyKind::GetBlocking>(target + offset + bulk, source + offset + bulk,
                                                     count - bulk, slot);
      }
    } else {
      s.ipcImpl_.ipcCopy_wg<MemcpyKind::GetBlocking>(target + offset, source + offset, count, slot);
    }
    offset += count;
  }
}

// Participants: any thread; unchanged general and power-of-two arithmetic.
__device__ __forceinline__ ProxyBroadcastLargeGeometry proxy_broadcast_make_large_geometry(
    const ProxyBroadcastTeamState &s, const ProxyBroadcastPayload &p) {
  const int payload_proxies = s.local_shm_size;
  const int source_count = s.local_shm_size - 1;
  const size_t piece_capacity = ProxyBroadcastSlots::payload_capacity / payload_proxies;
  constexpr size_t requested_global_tile = 64 * 1024;
  const size_t segment_bytes = ((p.nbytes + payload_proxies - 1) / payload_proxies + 15) & ~size_t{15};
  // Unaligned segment ends can cost one extra piece. A one-flag segment
  // uses its own width as the tile, making both boundaries coincide.
  const size_t tile_min = piece_capacity == 1 ? segment_bytes
      : ((segment_bytes + piece_capacity - 2) / (piece_capacity - 1) + 15) & ~size_t{15};
  const size_t tile_cap = ((p.nbytes + source_count * 4 - 1) / (source_count * 4) + 15) & ~size_t{15};
  const bool power2_geometry = (p.nbytes & (p.nbytes - 1)) == 0 &&
                               (payload_proxies & (payload_proxies - 1)) == 0;
  size_t global_tile_bytes = max(tile_min, min(requested_global_tile, tile_cap));
  unsigned tile_shift = 0;
  unsigned segment_shift = 0;
  if (power2_geometry) {
    // The aligned segment is nbytes / L, a power of two. Both rounded
    // bounds fit in the segment, so a power-of-two tile divides it exactly.
    const size_t min_power2 = tile_min <= 1 ? size_t{1}
        : size_t{1} << (64 - __builtin_clzll(tile_min - 1));
    const size_t cap_power2 = size_t{1} << (63 - __builtin_clzll(tile_cap));
    global_tile_bytes = max(min_power2, min(requested_global_tile, cap_power2));
    tile_shift = static_cast<unsigned>(__builtin_ctzll(global_tile_bytes));
    segment_shift = static_cast<unsigned>(__builtin_ctzll(segment_bytes));
  }
  const size_t global_tiles = power2_geometry ? p.nbytes >> tile_shift
      : (p.nbytes + global_tile_bytes - 1) / global_tile_bytes;
  const size_t pieces_per_segment = power2_geometry ? segment_bytes >> tile_shift : 0;
  return {payload_proxies, source_count, segment_bytes, power2_geometry, global_tile_bytes,
          tile_shift, segment_shift, global_tiles, pieces_per_segment};
}

// Participants: every WG thread on every PE; all-writer fence, source notice, then WG join.
template <typename NativeOps>
__device__ __forceinline__ void proxy_broadcast_stage_large_sources_wg(
    const ProxyBroadcastTeamState &s, const ProxyBroadcastPayload &p, const ProxyBroadcastLargeGeometry &g,
    const NativeOps &ops) {
  if (p.on_root_node) {
    if (constmem.my_pe == s.pe_root) {
      proxy_broadcast_copy_root_range_wg(p.dst_bytes, p.src_bytes, p.nbytes);
    } else {
      const int owner_ordinal = p.local_slot - (p.local_slot > p.root_slot);
      for (size_t tile = static_cast<size_t>(owner_ordinal); tile < g.global_tiles; tile += g.source_count) {
        const size_t begin = g.power2_geometry ? tile << g.tile_shift : tile * g.global_tile_bytes;
        ops.get_wg(p.dst_bytes + begin, p.src_bytes + begin,
            g.power2_geometry ? g.global_tile_bytes : min(g.global_tile_bytes, p.nbytes - begin),
            s.pe_root, s.pe_root);
      }
    }
  }
  // Finish ALL owned global tiles before publishing this owner's epoch.
  // Every WG writer fences; a thread-zero fence alone is insufficient.
  __syncthreads();
  __threadfence_system();
  __syncthreads();
  if (p.on_root_node && constmem.my_pe != s.pe_root && is_thread_zero_in_block()) {
    __hip_atomic_store(&s.ready[ProxyBroadcastSlots::source_ready_slot], p.epoch, __ATOMIC_RELEASE,
                       __HIP_MEMORY_SCOPE_SYSTEM);
    proxy_broadcast_notify_domains(s, p, ProxyBroadcastNoticeOp::Post);
  }
  // No local fanout starts before this PE has posted its owner notification.
  __syncthreads();
}

// Participants: every root-domain WG thread; only eligible complete waves consume when parallel.
template <typename NativeOps>
__device__ __forceinline__ void proxy_broadcast_fanout_large_root_wg(
    const ProxyBroadcastTeamState &s, const ProxyBroadcastPayload &p, const ProxyBroadcastThreads &w, const
    ProxyBroadcastLargeGeometry &g, const NativeOps &ops) {
  // Root already owns a complete result. Every other PE keeps its owned
  // tiles intact while copying only disjoint tiles from the other owners.
  if (constmem.my_pe != s.pe_root && w.consume) {
    unsigned acquired_owners = 0;
    int next_owner = w.consumer % g.source_count;
    const int owner_step = w.consumers % g.source_count;
    for (size_t tile = static_cast<size_t>(w.consumer); tile < g.global_tiles; tile += w.consumers) {
      const int owner_ordinal = next_owner;
      next_owner += owner_step;
      if (next_owner >= g.source_count) next_owner -= g.source_count;
      const int owner_slot = owner_ordinal + (owner_ordinal >= p.root_slot);
      if (owner_slot == p.local_slot) continue;
      // Owners above the mask width are acquired on every use.
      const unsigned owner_bit = owner_ordinal < 32 ? (1u << owner_ordinal) : 0u;
      if (!(acquired_owners & owner_bit)) {
        auto *owner_ready = reinterpret_cast<long *>(
            ops.local_ptr(&s.ready[ProxyBroadcastSlots::source_ready_slot], owner_slot));
        proxy_broadcast_wait_consumer(p, w, owner_ready);
        acquired_owners |= owner_bit;
      }
      const size_t begin = g.power2_geometry ? tile << g.tile_shift : tile * g.global_tile_bytes;
      char *source = ops.local_ptr(p.dst_bytes + begin, owner_slot);
      proxy_broadcast_copy_ipc_tile(s, w, p.dst_bytes + begin, source,
          g.power2_geometry ? g.global_tile_bytes : min(g.global_tile_bytes, p.nbytes - begin), owner_slot);
    }
  }
}

// Participants: remote producer thread zero only; acquire GPU relay249, including repeated owners above bit31.
template <typename NativeOps>
__device__ __forceinline__ void proxy_broadcast_acquire_large_owner_leader(
    const ProxyBroadcastTeamState &s, const ProxyBroadcastPayload &p, const NativeOps &ops, unsigned
    &acquired_owners, int owner_ordinal) {
  // Owners above the mask width are acquired on every use.
  const unsigned owner_bit = owner_ordinal < 32 ? (1u << owner_ordinal) : 0u;
  if (!(acquired_owners & owner_bit)) {
    const int owner_slot = owner_ordinal + (owner_ordinal >= p.root_slot);
    // Peer IPC only observes GPU-written relay249, never NIC inbox248.
    auto *relay = reinterpret_cast<long *>(ops.local_ptr(&s.ready[ProxyBroadcastSlots::notification_source_slot], owner_slot));
    while (__hip_atomic_load(relay, __ATOMIC_ACQUIRE,
                              __HIP_MEMORY_SCOPE_SYSTEM) != p.epoch) {}
    acquired_owners |= owner_bit;
  }
}

// Participants: the remote producer thread; unchanged circular owner arithmetic.
__device__ __forceinline__ int proxy_broadcast_advance_owner(
    int owner, int source_count) {
  return owner + 1 == source_count ? 0 : owner + 1;
}

// Participants: only wave0/lane0 in remote-domain WGs; relay own inbox before any peer or segment guard.
template <typename NativeOps>
__device__ __forceinline__ void proxy_broadcast_receive_large_segment_leader(
    const ProxyBroadcastTeamState &s, const ProxyBroadcastPayload &p, const ProxyBroadcastLargeGeometry &g,
    const NativeOps &ops) {
  // Relay our OWN NIC inbox before any segment guard, GET, or peer wait.
  // The root physical slot has no source owner and no incoming notice.
  if (p.local_slot != p.root_slot) {
    while (__hip_atomic_load(&s.ready[ProxyBroadcastSlots::source_ready_slot], __ATOMIC_ACQUIRE,
                              __HIP_MEMORY_SCOPE_SYSTEM) != p.epoch) {}
    __threadfence_system();
    __hip_atomic_store(&s.ready[ProxyBroadcastSlots::notification_source_slot], p.epoch, __ATOMIC_RELEASE,
                       __HIP_MEMORY_SCOPE_SYSTEM);
  }
  unsigned acquired_owners = 0;

  const size_t segment = g.power2_geometry ? static_cast<size_t>(p.local_slot) << g.segment_shift
      : static_cast<size_t>(p.local_slot) * g.segment_bytes;
  const size_t end = min(segment + g.segment_bytes, p.nbytes);
  if (segment < end) {
    const size_t first_tile = g.power2_geometry ? segment >> g.tile_shift : segment / g.global_tile_bytes;
    const size_t pieces = g.power2_geometry ? g.pieces_per_segment
        : (end + g.global_tile_bytes - 1) / g.global_tile_bytes - first_tile;
    const uint32_t row = static_cast<uint32_t>(blockIdx.x) % s.num_qps_per_pe;
    const uint32_t windows = s.num_qps_per_pe >= 4 ? 4 : (s.num_qps_per_pe >= 2 ? 2 : 1);
    // Pieces visit consecutive global tiles even when the first piece is partial.

    int completing_owner = static_cast<int>(first_tile % g.source_count);
    int posting_owner = completing_owner;
    // Each window has at most one outstanding piece. Source ownership
    // may change on refill, so completion uses that piece's exact QP.
    for (uint32_t window = 0; window < windows && window < pieces; window++) {
      const size_t begin = g.power2_geometry ? segment + (static_cast<size_t>(window) << g.tile_shift)
          : (window == 0 ? segment : (first_tile + window) * g.global_tile_bytes);
      const int owner_ordinal = posting_owner;
      posting_owner = proxy_broadcast_advance_owner(posting_owner, g.source_count);
      const int source_pe = p.root_node_first + owner_ordinal + (owner_ordinal >= p.root_slot);
      const uint32_t qp = ((row + window) % s.num_qps_per_pe) * constmem.num_pes + source_pe;
      const size_t length = g.power2_geometry ? g.global_tile_bytes
          : min(end - begin, g.global_tile_bytes - begin % g.global_tile_bytes);
      proxy_broadcast_acquire_large_owner_leader(s, p, ops, acquired_owners, owner_ordinal);
      s.qps[qp].get_nbi(p.dst_bytes + begin, p.dst_bytes + begin, length, s.wf_info);
    }
    for (size_t piece = 0; piece < pieces; piece++) {
      const int owner_ordinal = completing_owner;
      completing_owner = proxy_broadcast_advance_owner(completing_owner, g.source_count);
      const int source_pe = p.root_node_first + owner_ordinal + (owner_ordinal >= p.root_slot);
      const uint32_t window = static_cast<uint32_t>(piece % windows);
      const uint32_t qp = ((row + window) % s.num_qps_per_pe) * constmem.num_pes + source_pe;
      s.qps[qp].quiet(s.wf_info);
      __threadfence_system();
      __hip_atomic_store(&s.ready[piece * g.payload_proxies + p.local_slot], p.epoch,
                         __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
      const size_t next_piece = piece + windows;
      if (next_piece < pieces) {
        const size_t next = g.power2_geometry ? segment + (next_piece << g.tile_shift)
            : (first_tile + next_piece) * g.global_tile_bytes;
        const int next_ordinal = posting_owner;
        posting_owner = proxy_broadcast_advance_owner(posting_owner, g.source_count);
        const int next_source = p.root_node_first + next_ordinal + (next_ordinal >= p.root_slot);
        const uint32_t next_qp = ((row + window) % s.num_qps_per_pe) * constmem.num_pes + next_source;
        const size_t length = g.power2_geometry ? g.global_tile_bytes
            : min(end - next, g.global_tile_bytes - next % g.global_tile_bytes);
        proxy_broadcast_acquire_large_owner_leader(s, p, ops, acquired_owners, next_ordinal);
        s.qps[next_qp].get_nbi(p.dst_bytes + next, p.dst_bytes + next, length, s.wf_info);
      }
    }
  }
}

// Participants: every remote-domain WG thread; nonparallel WG joins after production, parallel waves overlap.
template <typename NativeOps>
__device__ __forceinline__ void proxy_broadcast_fanout_large_remote_wg(
    const ProxyBroadcastTeamState &s, const ProxyBroadcastPayload &p, const ProxyBroadcastThreads &w, const
    ProxyBroadcastLargeGeometry &g, const NativeOps &ops) {
  // With fewer than two full waves the WG first produces all its tiles;
  // every thread then participates in the ordinary WG IPC copy.
  if (!w.parallel) __syncthreads();
  if (w.consume) {
    for (int ordinal = w.consumer; ordinal < g.payload_proxies - 1; ordinal += w.consumers) {
      const int slot = ordinal + (ordinal >= p.local_slot);
      const size_t segment = g.power2_geometry ? static_cast<size_t>(slot) << g.segment_shift
          : static_cast<size_t>(slot) * g.segment_bytes;
      const size_t end = min(segment + g.segment_bytes, p.nbytes);
      if (segment >= end) continue;
      const size_t first_tile = g.power2_geometry ? segment >> g.tile_shift : segment / g.global_tile_bytes;
      const size_t pieces = g.power2_geometry ? g.pieces_per_segment
          : (end + g.global_tile_bytes - 1) / g.global_tile_bytes - first_tile;
      for (size_t piece = 0; piece < pieces; piece++) {
        const size_t begin = g.power2_geometry ? segment + (piece << g.tile_shift)
            : (piece == 0 ? segment : (first_tile + piece) * g.global_tile_bytes);
        auto *flag = reinterpret_cast<long *>(ops.local_ptr(&s.ready[piece * g.payload_proxies + slot], slot));
        proxy_broadcast_wait_consumer(p, w, flag);
        char *source = ops.local_ptr(p.dst_bytes + begin, slot);
        proxy_broadcast_copy_ipc_tile(s, w, p.dst_bytes + begin, source,
            g.power2_geometry ? g.global_tile_bytes
                : min(end - begin, g.global_tile_bytes - begin % g.global_tile_bytes), slot);
      }
    }
  }
}

// Participants: any thread; team geometry stays independent of local QP-window availability.
__device__ __forceinline__ ProxyBroadcastSegmentGeometry proxy_broadcast_make_segment_geometry(
    const ProxyBroadcastTeamState &s, const ProxyBroadcastPayload &p) {
  // Segment geometry is a team decision. Each producer independently limits
  // its in-flight tile count to the QP rows available on that PE.
  const bool fine_tiles = p.proxies == s.local_shm_size && p.nbytes > kProxyBroadcastLargeThreshold / 2;
  const size_t requested_tile_bytes = fine_tiles ? 32 * 1024 : 64 * 1024;
  const size_t max_tiles = ProxyBroadcastSlots::payload_capacity / p.proxies;
  const size_t capacity_tile = (p.chunk + max_tiles - 1) / max_tiles;
  const size_t tile_bytes = max(requested_tile_bytes, (capacity_tile + 15) & ~size_t{15});
  const size_t tiles = (p.chunk + tile_bytes - 1) / tile_bytes;
  const bool root_early_ready = p.proxies == s.local_shm_size && constmem.my_pe == s.pe_root;
  return {fine_tiles, tile_bytes, tiles, root_early_ready};
}

// Participants: every WG thread; early root publication and every original all-writer join stay together.
template <typename NativeOps>
__device__ __forceinline__ void proxy_broadcast_stage_segment_sources_wg(
    const ProxyBroadcastTeamState &s, const ProxyBroadcastPayload &p, const ProxyBroadcastThreads &w, const
    ProxyBroadcastSegmentGeometry &g, const NativeOps &ops) {
  if (g.root_early_ready) {
    const size_t own_begin = static_cast<size_t>(p.local_slot) * p.chunk;
    const size_t own_end = min(own_begin + p.chunk, p.nbytes);
    proxy_broadcast_copy_root_range_wg(p.dst_bytes + own_begin, p.src_bytes + own_begin, own_end - own_begin);
    // Every own-segment writer joins and fences before tile publication.
    __syncthreads();
    __threadfence_system();
    __syncthreads();
    if (is_thread_zero_in_block()) {
      for (size_t tile = 0; tile < g.tiles; tile++) {
        if (own_begin + tile * g.tile_bytes >= own_end) break;
        __hip_atomic_store(&s.ready[tile * p.proxies + p.local_slot], p.epoch,
                           __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
      }
      proxy_broadcast_notify_domains(s, p, ProxyBroadcastNoticeOp::Post);
    }
    // Join the post without quiet; remote GET may overlap the remaining copy.
    // Both ranges are disjoint from the published own segment.
    __syncthreads();
    if (p.dst_bytes != p.src_bytes) {
      proxy_broadcast_copy_root_range_wg(p.dst_bytes, p.src_bytes, own_begin);
      proxy_broadcast_copy_root_range_wg(p.dst_bytes + own_end, p.src_bytes + own_end, p.nbytes - own_end);
    }
  } else if (constmem.my_pe == s.pe_root) {
    proxy_broadcast_copy_root_range_wg(p.dst_bytes, p.src_bytes, p.nbytes);
  } else if (p.on_root_node && p.local_slot < p.proxies) {
    const size_t begin = static_cast<size_t>(p.local_slot) * p.chunk;
    if (begin < p.nbytes) {
      const size_t length = min(p.chunk, p.nbytes - begin);
      ops.get_wg(p.dst_bytes + begin, p.src_bytes + begin, length, s.pe_root, s.pe_root);
    }
  }

  // Every staging writer finishes before thread zero publishes its segment.
  __syncthreads();
  __threadfence_system();
  __syncthreads();
  if (p.on_root_node && p.local_slot < p.proxies && !g.root_early_ready && w.tid == 0) proxy_broadcast_notify_domains(s, p, ProxyBroadcastNoticeOp::Post);
}

// Participants: only tid0 where local_slot < proxies; retain NIC wait, exact-QP quiet/refill, and root drain.
__device__ __forceinline__ void proxy_broadcast_receive_segment_leader(
    const ProxyBroadcastTeamState &s, const ProxyBroadcastPayload &p, const ProxyBroadcastSegmentGeometry &g) {
  const size_t segment = static_cast<size_t>(p.local_slot) * p.chunk;
  const int source_proxy = p.root_node_first + p.local_slot;
  const uint32_t row = static_cast<uint32_t>(blockIdx.x) % s.num_qps_per_pe;
  const uint32_t windows = (g.fine_tiles && s.num_qps_per_pe >= 4) ? 4 : (s.num_qps_per_pe >= 2 ? 2 : 1);
  const size_t end = min(segment + p.chunk, p.nbytes);
  // A QP contains at most one tile from this operation. Prime the
  // independent rows before waiting; completion on one row must not
  // wait for the tile on the other row.
  if (!p.on_root_node) {
    // Only our GPU polls its own NIC inbox.
    while (__hip_atomic_load(&s.ready[ProxyBroadcastSlots::source_ready_slot], __ATOMIC_ACQUIRE,
                             __HIP_MEMORY_SCOPE_SYSTEM) != p.epoch) {}
    for (uint32_t window = 0; window < windows; window++) {
      const size_t begin = segment + static_cast<size_t>(window) * g.tile_bytes;
      if (begin >= end) break;
      const uint32_t qp = ((row + window) % s.num_qps_per_pe) * constmem.num_pes + source_proxy;
      s.qps[qp].get_nbi(p.dst_bytes + begin, p.dst_bytes + begin, min(g.tile_bytes, end - begin), s.wf_info);
    }
  }
  for (size_t tile = 0; !g.root_early_ready && tile < g.tiles; tile++) {
    const size_t begin = segment + tile * g.tile_bytes;
    if (begin >= end) break;
    const uint32_t window = static_cast<uint32_t>(tile % windows);
    const uint32_t qp = ((row + window) % s.num_qps_per_pe) * constmem.num_pes + source_proxy;
    if (!p.on_root_node) s.qps[qp].quiet(s.wf_info);
    __threadfence_system();
    __hip_atomic_store(&s.ready[tile * p.proxies + p.local_slot], p.epoch,
                       __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
    // Refill only the row whose tile has completed and been published.
    const size_t next = segment + (tile + windows) * g.tile_bytes;
    if (!p.on_root_node && next < end) {
      s.qps[qp].get_nbi(p.dst_bytes + next, p.dst_bytes + next, min(g.tile_bytes, end - next), s.wf_info);
    }
  }
  if (p.on_root_node) proxy_broadcast_notify_domains(s, p, ProxyBroadcastNoticeOp::Drain);
}

// Participants: every WG thread; preserve nonparallel join and root-domain whole-segment/remote per-tile waits.
template <typename NativeOps>
__device__ __forceinline__ void proxy_broadcast_fanout_segments_wg(
    const ProxyBroadcastTeamState &s, const ProxyBroadcastPayload &p, const ProxyBroadcastThreads &w, const
    ProxyBroadcastSegmentGeometry &g, const NativeOps &ops) {
  if (!w.parallel) __syncthreads();
  if (w.consume && constmem.my_pe != s.pe_root) {
    const int own = p.local_slot < p.proxies ? p.local_slot : -1;
    const int peers = p.proxies - (own >= 0);
    for (int ordinal = w.consumer; ordinal < peers; ordinal += w.consumers) {
      const int slot = ordinal + (own >= 0 && ordinal >= own);
      const size_t segment = static_cast<size_t>(slot) * p.chunk;
      if (p.on_root_node) {
        // Root-domain sources stage their complete segment before flag zero.
        // Receiver-NIC flags prove only one tile and must keep the loop below.
        const size_t end = min(segment + p.chunk, p.nbytes);
        if (segment < end) {
          auto *flag = reinterpret_cast<long *>(ops.local_ptr(&s.ready[slot], slot));
          proxy_broadcast_wait_consumer(p, w, flag);
          char *source = ops.local_ptr(p.dst_bytes + segment, slot);
          proxy_broadcast_copy_ipc_tile(s, w, p.dst_bytes + segment, source, end - segment, slot);
        }
      } else {
        for (size_t tile = 0; tile < g.tiles; tile++) {
          const size_t begin = segment + tile * g.tile_bytes;
          if (begin >= min(segment + p.chunk, p.nbytes)) break;
          auto *flag = reinterpret_cast<long *>(ops.local_ptr(&s.ready[tile * p.proxies + slot], slot));
          proxy_broadcast_wait_consumer(p, w, flag);
          char *source = ops.local_ptr(p.dst_bytes + begin, slot);
          proxy_broadcast_copy_ipc_tile(s, w, p.dst_bytes + begin, source,
              min(g.tile_bytes, min(segment + p.chunk, p.nbytes) - begin), slot);
        }
      }
    }
  }
}

}  // namespace

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
                                                           long *p_sync, GDATeam *ready_team) {  // NOLINT(runtime/int)
  ActiveWFInfo wf_info(ctx_id_, ThreadScope::wg);
  if (constmem.num_pes < 4) {
    internal_put_broadcastmem_wg(dst, src, nelems, pe_root, pe_start, stride, pe_size, wf_info);
    internal_sync_wg(constmem.my_pe, pe_start, stride, pe_size, p_sync, wf_info);
    return;
  }

  // Attempt multi-proxy broadcast for large messages with multiple nodes
  if (internal_broadcastmem_wg_proxy_nopxn(dst, src, nelems, pe_root, pe_start, stride, pe_size,
                                         p_sync, wf_info, ready_team)) {
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
                                                              ActiveWFInfo &wf_info, GDATeam *ready_team) {  // NOLINT(runtime/int)

  const size_t nbytes = static_cast<size_t>(nelems);
  if (nbytes < kHierarchicalMinBytes || stride != 1) return false;
  const bool pipeline = !constmem.pxn_enabled;
  if (pipeline && (!ready_team || !ready_team->bcast_ready)) return false;
  long *ready = pipeline ? ready_team->bcast_ready : nullptr;
  if (pipeline && pe_size > ROCSHMEM_BARRIER_SYNC_SIZE) return false;

  // Keep access to native private context operations inside member scope.
  // The stages have one concrete adapter type and no indirect device calls.
  struct NativeOps {
    GDAContext &ctx;
    ActiveWFInfo &wf;
    __device__ __forceinline__ char *local_ptr(const void *address, int slot) const {
      return ctx.get_local_ptr(address, slot);
    }
    __device__ __forceinline__ void get(void *target, const void *source, size_t bytes, int pe, int qp) const {
      ctx.internal_getmem(target, source, bytes, pe, qp, wf);
    }
    __device__ __forceinline__ void put(void *target, const void *source, size_t bytes, int pe, int qp) const {
      ctx.internal_putmem(target, source, bytes, pe, qp, wf);
    }
    __device__ __forceinline__ void get_wg(void *target, const void *source, size_t bytes, int pe, int qp) const {
      ctx.internal_getmem_wg(target, source, bytes, pe, qp, wf);
    }
    __device__ __forceinline__ void sync_wg(int pe, int first, int step, int members, long *sync) const {
      ctx.internal_sync_wg(pe, first, step, members, sync, wf);
    }
  };
  const NativeOps ops{*this, wf_info};
  const ProxyBroadcastTeamState s{pe_start, stride, pe_size, pe_root, p_sync, ready_team, ready,
                                constmem.ipc_shm_size, num_qps_per_pe, qps, ipcImpl_, wf_info};
  if (pipeline) {
    if (!proxy_broadcast_validate_team_wg(s, ops)) return false;
  } else {
    // Retain the existing PXN fallback eligibility contract.
    if (s.local_shm_size <= 1 || (ready_team && s.local_shm_size > ROCSHMEM_BARRIER_SYNC_SIZE) ||
        constmem.ipc_stride != 1 || pe_start % s.local_shm_size != 0 ||
        pe_size % s.local_shm_size != 0 || pe_size <= s.local_shm_size) return false;
  }
  auto p = proxy_broadcast_make_payload(s, dst, src, nbytes, pipeline);
  if (!pipeline) {
    proxy_broadcast_legacy_wg(s, p, ops);
    return true;
  }

  p.epoch = static_cast<long>(ready_team->bcast_sequence_number % INT64_MAX) + 1;
  proxy_broadcast_begin_epoch_wg(s, p, ops);
  const auto w = proxy_broadcast_make_threads();
  const auto completion = p.domain_count == 2 ? ProxyBroadcastPhaseKind::PairedDone
                                             : ProxyBroadcastPhaseKind::LeaderSync;
  proxy_broadcast_publish_root_input_wg(s, p, ops);

  if (nbytes > kProxyBroadcastLargeThreshold) {
    const auto g = proxy_broadcast_make_large_geometry(s, p);
    proxy_broadcast_stage_large_sources_wg(s, p, g, ops);
    if (p.on_root_node) {
      proxy_broadcast_fanout_large_root_wg(s, p, w, g, ops);
    } else {
      if (w.wave == 0 && w.lane == 0) proxy_broadcast_receive_large_segment_leader(s, p, g, ops);
      proxy_broadcast_fanout_large_remote_wg(s, p, w, g, ops);
    }
    if (p.on_root_node && p.local_slot != p.root_slot && w.tid == 0)
      proxy_broadcast_notify_domains(s, p, ProxyBroadcastNoticeOp::Drain);
    proxy_broadcast_finish_wg(s, p, w, ops, completion);
    return true;
  }

  const auto g = proxy_broadcast_make_segment_geometry(s, p);
  proxy_broadcast_stage_segment_sources_wg(s, p, w, g, ops);
  if (w.tid == 0 && p.local_slot < p.proxies) proxy_broadcast_receive_segment_leader(s, p, g);
  proxy_broadcast_fanout_segments_wg(s, p, w, g, ops);
  proxy_broadcast_finish_wg(s, p, w, ops, completion);
  return true;
}

__device__ bool GDAContext::try_broadcast_ready_fanout_wg_nopxn(const BroadcastFanout &fanout,
    GDATeam *ready_team, int pe_size, ActiveWFInfo &wf_info) {
  if (constmem.pxn_enabled || !ready_team || !ready_team->bcast_ready) return false;

  long *ready = ready_team->bcast_ready;
  const long epoch = static_cast<long>(ready_team->bcast_sequence_number % INT64_MAX) + 1;
  if (fanout.local_slot < fanout.proxies) {
    // All staging writers and the thread waiting on GET completion precede publication.
    __syncthreads();
    if (is_thread_zero_in_block()) {
      __threadfence_system();
      __hip_atomic_store(&ready[fanout.local_slot], epoch, __ATOMIC_RELEASE,
                         __HIP_MEMORY_SCOPE_SYSTEM);
    }
    __syncthreads();
  }

  constexpr size_t wave_fanout_min_bytes = 512 * 1024;
  if (fanout.nbytes < wave_fanout_min_bytes && fanout.proxies <= 64 &&
      pe_size < ROCSHMEM_BCAST_SYNC_SIZE) {
    broadcast_gather_ready_segments_wg_nopxn(fanout, ready, epoch, wf_info);
  } else {
    broadcast_gather_ready_segments_wave_nopxn(fanout, ready, epoch);
  }
  return true;
}

__device__ void GDAContext::broadcast_gather_ready_segments_wg_nopxn(const BroadcastFanout &fanout,
    long *ready, long epoch, ActiveWFInfo &wf_info) {
  // This control value belongs to the WG, never to any collective's pSync.
  __shared__ int selected_slot;
  uint64_t pending = fanout.proxies == 64 ? ~uint64_t{0} : (uint64_t{1} << fanout.proxies) - 1;
  if (fanout.local_slot < fanout.proxies) pending &= ~(uint64_t{1} << fanout.local_slot);

  int scan_start = 0;
  while (pending) {
    if (is_thread_zero_in_block()) {
      int slot = scan_start;
      for (;;) {
        if (pending & (uint64_t{1} << slot)) {
          // Proxy admission guarantees that contiguous IPC slots name these owners.
          auto *flag = reinterpret_cast<long *>(get_local_ptr(&ready[slot], slot));
          if (__hip_atomic_load(flag, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) == epoch) break;
        }
        if (++slot == fanout.proxies) slot = 0;
      }
      selected_slot = slot;
    }
    __syncthreads();
    const int slot = selected_slot;
    const size_t begin = static_cast<size_t>(slot) * fanout.chunk;
    char *segment = fanout.dst + begin;
    const int peer = constmem.ipc_first_pe + slot;
    internal_getmem_wg(segment, segment, min(fanout.chunk, fanout.nbytes - begin), peer, peer, wf_info);
    pending &= ~(uint64_t{1} << slot);
    scan_start = (slot + 1 == fanout.proxies) ? 0 : slot + 1;
    __syncthreads();
  }
}

__device__ void GDAContext::broadcast_gather_ready_segments_wave_nopxn(const BroadcastFanout &fanout,
    long *ready, long epoch) {
  constexpr int max_ipc_waves = 8;
  const int tid = get_flat_block_id();
  const int lane = tid % warpSize;
  // ipcCopy_wave indexes lanes using WF_SIZE; mismatches use a uniform WG copy.
  const bool wave_copy = warpSize == WF_SIZE;
  const int group = wave_copy ? tid / WF_SIZE : 0;
  const int available_waves = (get_flat_block_size() + WF_SIZE - 1) / WF_SIZE;
  const int groups = wave_copy ? min(max_ipc_waves, available_waves) : 1;
  if (group < groups) {
    for (int slot = group; slot < fanout.proxies; slot += groups) {
      if (slot == fanout.local_slot) continue;
      auto *flag = reinterpret_cast<long *>(get_local_ptr(&ready[slot], slot));
      int arrived = 0;
      do {
        if (lane == 0) {
          arrived = __hip_atomic_load(flag, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) == epoch;
        }
      } while (!__shfl(arrived, 0));

      const size_t begin = static_cast<size_t>(slot) * fanout.chunk;
      char *segment = fanout.dst + begin;
      char *source = get_local_ptr(segment, slot);
      const size_t length = min(fanout.chunk, fanout.nbytes - begin);
      if (wave_copy) {
        ipcImpl_.ipcCopy_wave<MemcpyKind::GetBlocking>(segment, source, length, slot);
      } else {
        ipcImpl_.ipcCopy_wg<MemcpyKind::GetBlocking>(segment, source, length, slot);
      }
    }
  }
  __syncthreads();
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

  internal_broadcastmem_wg_nopxn(dst, src, nelems, root, info->pe_start, info->stride,
                                team_obj->num_pes, p_sync, team_obj);
}

__device__ void GDAContext::internal_broadcastmem_wg_das(void *dst, const void *src, int nelems, int pe_root,
                                                         int pe_start, int stride, int pe_size, long *p_sync,
                                                         GDATeam *team_obj) {  // NOLINT(runtime/int)
  // Typed and byte APIs share the same task routing and completion protocol.
  internal_broadcast_wg(static_cast<char *>(dst), static_cast<const char *>(src), nelems, pe_root, pe_start, stride,
                        pe_size, p_sync, 1, -1, team_obj);
}

}  // namespace rocshmem

#endif  // USE_DAS
