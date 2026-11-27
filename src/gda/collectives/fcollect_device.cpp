// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#include "rocshmem/rocshmem.hpp"
#include "context_incl.hpp"
#include "gda/context_gda_tmpl_device.hpp"
#include "gda/gda_team.hpp"
#include "util.hpp"
#include <climits>

#if defined(USE_DAS)
namespace rocshmem {
namespace {

// All slots are symmetric, registered, and private to non-PXN WG fcollect.
struct FcollectControl {
  enum { vote = 0, scratch = 1, decision = 2, release = 3,
         signal = 4, sync_sequence = 5 };
};

__device__ __forceinline__ long fcollect_load(const long *p) {
  return __hip_atomic_load(p, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ void fcollect_store(long *p, long value) {
  __hip_atomic_store(p, value, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}

// Every WG thread participates. A peer cannot publish its next arrival until
// it has consumed this release, so exact epochs also safely wrap LONG_MAX to1.
// Blocking puts keep the registered signal stable until the NIC is finished.
template <typename Ops>
__device__ __forceinline__ void fcollect_join_wg(GDATeam *team, const Ops &ops) {
  __syncthreads();
  __threadfence_system();
  __syncthreads();
  if (is_thread_zero_in_block()) {
    long *control = team->fcollect_control;
    const long previous = control[FcollectControl::sync_sequence];
    const long epoch = previous == LONG_MAX ? 1 : previous + 1;
    const int first = team->tinfo_wrt_world->pe_start;
    const int stride = team->tinfo_wrt_world->stride;
    long *arrivals = team->fcollect_pSync + (epoch & 1) * ROCSHMEM_BARRIER_SYNC_SIZE;
    fcollect_store(&control[FcollectControl::signal], epoch);
    if (team->my_pe == 0) {
      for (int rank = 1; rank < team->num_pes; ++rank) {
        while (fcollect_load(&arrivals[rank]) != epoch) {}
      }
      for (int rank = 1; rank < team->num_pes; ++rank) {
        ops.put(&control[FcollectControl::release], &control[FcollectControl::signal],
                sizeof(long), first + rank * stride);
      }
    } else {
      ops.put(&arrivals[team->my_pe], &control[FcollectControl::signal], sizeof(long), first);
      while (fcollect_load(&control[FcollectControl::release]) != epoch) {}
    }
    control[FcollectControl::sync_sequence] = epoch;
  }
  __syncthreads();
  __threadfence_system();
  __syncthreads();
}

__device__ __forceinline__ bool fcollect_heap_range(const void *ptr, size_t bytes) {
  const uintptr_t offset = reinterpret_cast<uintptr_t>(ptr) - constmem.heap_base;
  return offset < constmem.heap_size && bytes <= constmem.heap_size - offset;
}

// Per-call predicates are voted on even when one PE cannot use IPC. A positive
// width certifies equal team-aligned domains and every actual local slot map.
template <typename Ops>
__device__ __forceinline__ bool fcollect_admit_wg(
    GDATeam *team, long vote, const Ops &ops) {
  long *control = team->fcollect_control;
  if (is_thread_zero_in_block()) fcollect_store(&control[FcollectControl::vote], vote);
  fcollect_join_wg(team, ops);
  if (team->my_pe == 0 && is_thread_zero_in_block()) {
    long decision = fcollect_load(&control[FcollectControl::vote]);
    for (int rank = 1; decision > 0 && rank < team->num_pes; ++rank) {
      ops.get(&control[FcollectControl::scratch], &control[FcollectControl::vote], sizeof(long),
              team->get_pe_in_world(rank));
      if (fcollect_load(&control[FcollectControl::scratch]) != decision) decision = -1;
    }
    fcollect_store(&control[FcollectControl::decision], decision);
    for (int rank = 1; rank < team->num_pes; ++rank) {
      ops.put(&control[FcollectControl::decision], &control[FcollectControl::decision],
              sizeof(long), team->get_pe_in_world(rank));
    }
  }
  fcollect_join_wg(team, ops);
  return fcollect_load(&control[FcollectControl::decision]) > 0;
}

}  // namespace

__device__ __noinline__ bool GDAContext::try_fcollectmem_same_slot_wg_nopxn(
    GDATeam *team, void *dst, const void *src, size_t rank_bytes,
    size_t offset_bytes, size_t count_bytes) {
  // Only team-uniform API properties can reject before the collective vote.
  if (constmem.pxn_enabled || team->num_pes > ROCSHMEM_BARRIER_SYNC_SIZE ||
      team->tinfo_wrt_world->stride != 1 || count_bytes == 0 ||
      rank_bytes == 0 || offset_bytes > rank_bytes || count_bytes > rank_bytes - offset_bytes) return false;

  ActiveWFInfo wf_info(ctx_id_, ThreadScope::wg);
  struct NativeOps {
    GDAContext &ctx;
    ActiveWFInfo &wf;
    __device__ __forceinline__ void put(void *dst, const void *src, size_t bytes, int peer) const {
      ctx.internal_putmem(dst, src, bytes, peer, ctx.get_team_qp_index(team, peer), wf);
    }
    __device__ __forceinline__ void get(void *dst, const void *src, size_t bytes, int peer) const {
      ctx.internal_getmem(dst, src, bytes, peer, ctx.get_team_qp_index(team, peer), wf);
    }
    GDATeam *team;
  } ops{*this, wf_info, team};

  const int width = constmem.ipc_shm_size;
  const int first = team->tinfo_wrt_world->pe_start;
  const int threads = get_flat_block_size();
  bool eligible = false;
#if defined(__HIP_DEVICE_COMPILE__) && ROCSHMEM_ARCH_HYGON && (defined(__gfx936__) || defined(__gfx938__))
  eligible = constmem.gda_provider == GDAProvider::MLX5 && warpSize == WF_SIZE &&
      threads >= WF_SIZE && width > 1 && team->num_pes > width &&
      team->num_pes % width == 0 && constmem.ipc_stride == 1 &&
      constmem.ipc_first_pe >= first && (constmem.ipc_first_pe - first) % width == 0 &&
      constmem.ipc_first_pe + width <= first + team->num_pes;
#endif
  const size_t pes = static_cast<size_t>(team->num_pes);
  eligible = eligible && rank_bytes <= SIZE_MAX / pes;
  const size_t output_bytes = eligible ? rank_bytes * pes : 0;
  const uintptr_t da = reinterpret_cast<uintptr_t>(dst);
  const uintptr_t sa = reinterpret_cast<uintptr_t>(src);
  eligible = eligible && fcollect_heap_range(dst, output_bytes) && fcollect_heap_range(src, rank_bytes) &&
      (da <= sa ? output_bytes <= sa - da : rank_bytes <= da - sa);
  if (eligible) {
    for (int slot = 0; slot < width; ++slot) {
      int mapped = -1;
      if (!ipcImpl_.isIpcAvailable(constmem.my_pe, constmem.ipc_first_pe + slot, &mapped) || mapped != slot) {
        eligible = false;
      }
    }
  }
  if (!fcollect_admit_wg(team, eligible ? width : -1, ops)) {
    if (is_thread_zero_in_block()) LOGD_TRACE("fcollect same-slot fallback");
    return false;
  }

  constexpr size_t chunk_limit = 256 * 1024;
  const size_t chunk_bytes = min(count_bytes, chunk_limit);
  const size_t chunks = count_bytes / chunk_bytes + (count_bytes % chunk_bytes != 0);
  const int bank = team->fcollect_sequence_number & 1;
  // Rebase before converting a cumulative ready threshold to signed long.
  if (team->fcollect_ready_base[bank] > static_cast<uint64_t>(LONG_MAX) - chunks) {
    fcollect_join_wg(team, ops);
    for (int i = get_flat_block_id(); i < 2 * ROCSHMEM_BARRIER_SYNC_SIZE; i += threads) {
      fcollect_store(&team->fcollect_ready[i], 0);
    }
    if (is_thread_zero_in_block()) team->fcollect_ready_base[0] = team->fcollect_ready_base[1] = 0;
    fcollect_join_wg(team, ops);
  }
  long *ready = team->fcollect_ready + bank * ROCSHMEM_BARRIER_SYNC_SIZE;
  const long base = static_cast<long>(team->fcollect_ready_base[bank]);
  const int local_slot = team->my_pe % width;
  const int my_domain = team->my_pe / width;
  const int domains = team->num_pes / width;
  const int consumers = count_bytes <= chunk_limit ? min(8, threads / WF_SIZE) : 1;
  const size_t depth = count_bytes <= 2 * 1024 * 1024 ? 1 : 2;
  if (is_thread_zero_in_block()) LOGD_TRACE("fcollect same-slot bytes=%zu offset=%zu count=%zu waves=%d",
                                          rank_bytes, offset_bytes, count_bytes, consumers);

  // Every caller lane publishes its source writes before a NIC can read them.
  __threadfence_system();
  __syncthreads();
  auto publish = [&] __device__(size_t chunk) {
    const size_t offset = offset_bytes + chunk * chunk_bytes;
    const size_t length = min(chunk_bytes, count_bytes - chunk * chunk_bytes);
    char *target = static_cast<char *>(dst) + static_cast<size_t>(team->my_pe) * rank_bytes + offset;
    const char *source = static_cast<const char *>(src) + offset;
    memcpy_wg<MemcpyKind::Put>(target, const_cast<char *>(source), length);
    for (int domain = 0; domain < domains; ++domain) {
      if (domain == my_domain) continue;
      const int peer = first + domain * width + local_slot;
      internal_putmem_nbi_wg(target, source, length, peer, get_team_qp_index(team, peer), wf_info);
    }
    __syncthreads();
    __threadfence_system();
    __syncthreads();
    if (is_thread_zero_in_block()) {
      for (int domain = 0; domain < domains; ++domain) {
        if (domain == my_domain) continue;
        const int peer = first + domain * width + local_slot;
        const int qp = get_team_qp_index(team, peer);
        // Confirm payload completion even for relaxed-ordering registrations.
        qps[qp].quiet(wf_info);
        auto [address, key] = qps[qp].get_raddr_info(&ready[team->my_pe]);
        qps[qp].atomic_add(reinterpret_cast<void *>(address), key, 1, wf_info);
      }
      fcollect_store(&ready[team->my_pe], base + chunk + 1);
    }
    __syncthreads();
  };

  for (size_t chunk = 0; chunk < min(depth, chunks); ++chunk) publish(chunk);
  for (size_t chunk = 0; chunk < chunks; ++chunk) {
    const size_t offset = offset_bytes + chunk * chunk_bytes;
    const size_t length = min(chunk_bytes, count_bytes - chunk * chunk_bytes);
    const long expected = base + chunk + 1;
    const int wave = get_flat_block_id() / WF_SIZE;
    // Complete waves partition source ranks with exactly the same stride.
    const int begin = consumers > 1 ? wave : 0;
    if (consumers == 1 || wave < consumers) {
      for (int rank = begin; rank < team->num_pes; rank += consumers) {
        const int owner = rank % width;
        auto *flag = reinterpret_cast<long *>(get_local_ptr(&ready[rank], owner));
        if (consumers > 1) {
          if (is_thread_zero_in_wave()) while (fcollect_load(flag) < expected) {}
          __builtin_amdgcn_wave_barrier();
          __threadfence_system();
        } else {
          if (is_thread_zero_in_block()) while (fcollect_load(flag) < expected) {}
          __syncthreads();
          __threadfence_system();
        }
        if (owner == local_slot) continue;
        char *target = static_cast<char *>(dst) + static_cast<size_t>(rank) * rank_bytes + offset;
        char *source = get_local_ptr(target, owner);
        if (consumers > 1) memcpy_wave<MemcpyKind::Get>(target, source, length);
        else memcpy_wg<MemcpyKind::Get>(target, source, length);
      }
    }
    __syncthreads();
    if (chunk + depth < chunks) publish(chunk + depth);
  }
  __syncthreads();
  __threadfence_system();
  __syncthreads();
  if (is_thread_zero_in_block()) {
    for (int domain = 0; domain < domains; ++domain) {
      if (domain != my_domain) qps[get_team_qp_index(team, first + domain * width + local_slot)].quiet(wf_info);
    }
  }
  fcollect_join_wg(team, ops);
  if (is_thread_zero_in_block()) {
    team->fcollect_ready_base[bank] += chunks;
    ++team->fcollect_sequence_number;
  }
  __syncthreads();
  return true;
}
}  // namespace rocshmem
#endif  // USE_DAS
