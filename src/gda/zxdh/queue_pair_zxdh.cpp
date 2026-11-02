// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#include "gda/queue_pair.hpp"
#include "segment_builder_zxdh.hpp"

namespace rocshmem {

namespace zxdh {

constexpr uint32_t kCqWqeShift = 6;
constexpr uint64_t kPollTimeout = 4000000000ULL;

constexpr uint64_t kCqValid = 1ULL << 5;
constexpr uint64_t kCqError = 1ULL << 39;
constexpr uint64_t kCqMinorErr = ((1ULL << 39) - 1) & ~((1ULL << 7) - 1);
constexpr uint64_t kCqMajorErr = ((1ULL << 39) - 1) & ~((1ULL << 23) - 1);
constexpr uint64_t kCqWqeIdx = ((1ULL << 58) - 1) & ~((1ULL << 40) - 1);

__device__ __forceinline__ uint64_t field_get(uint64_t mask,
                                              uint64_t value) {
  return (value & mask) >> (__ffsll(static_cast<long long>(mask)) - 1);
}

__device__ __forceinline__ uint8_t cqe_polarity(uint64_t cqe_idx,
                                                 uint32_t depth) {
  return !((cqe_idx >> (__ffs(depth) - 1)) & 1);
}

__device__ __forceinline__ uint64_t global_time() {
  uint64_t ret;
  asm volatile("s_memtime %0\n"
               "s_waitcnt lgkmcnt(0)\n"
               : "=s"(ret)::"memory");
  return ret;
}

}  // namespace zxdh

__device__ __forceinline__ void QueuePair::zxdh_ring_doorbell() {
  __hip_atomic_store(zxdh_dbreg, zxdh_qpn, __ATOMIC_RELAXED,
                     __HIP_MEMORY_SCOPE_AGENT);
}

__device__ void QueuePair::zxdh_poll_cq(uint64_t target) {
  if (target == 0 || zxdh_cq_cnt == 0) {
    return;
  }

  const uint64_t start = zxdh::global_time();
  while (true) {
    if (__hip_atomic_load(&zxdh_cq_consumed, __ATOMIC_RELAXED,
                          __HIP_MEMORY_SCOPE_AGENT) >= target) {
      return;
    }

    if (atomicCAS(&zxdh_poll_lock, 0, 1) == 0) {
      uint64_t consumed = __hip_atomic_load(&zxdh_cq_consumed, __ATOMIC_RELAXED,
                                            __HIP_MEMORY_SCOPE_AGENT);
      if (consumed < target) {
        uint64_t *cqe = reinterpret_cast<uint64_t*>(
            reinterpret_cast<uintptr_t>(zxdh_cq_buf) +
            ((consumed & (zxdh_cq_cnt - 1)) << zxdh::kCqWqeShift));
        const uint8_t expected = zxdh::cqe_polarity(consumed, zxdh_cq_cnt);
        uint64_t qword0;
        do {
          qword0 = __hip_atomic_load(&cqe[0], __ATOMIC_RELAXED,
                                     __HIP_MEMORY_SCOPE_AGENT);
        } while (zxdh::field_get(zxdh::kCqValid, qword0) != expected);

        if (zxdh::field_get(zxdh::kCqError, qword0)) {
          printf("ZXDH CQE error: minor=0x%llx major=0x%llx wqe=0x%llx\n",
                 static_cast<unsigned long long>(
                     zxdh::field_get(zxdh::kCqMinorErr, qword0)),
                 static_cast<unsigned long long>(
                     zxdh::field_get(zxdh::kCqMajorErr, qword0)),
                 static_cast<unsigned long long>(
                     zxdh::field_get(zxdh::kCqWqeIdx, qword0)));
          abort();
        }

        uint64_t wqe_desc = zxdh::field_get(zxdh::kCqWqeIdx, qword0);
        uint32_t wqe_in_cqe = static_cast<uint32_t>(wqe_desc & 0x7fff);
        uint8_t polarity_in_cqe =
            static_cast<uint8_t>((wqe_desc >> 15) & 1);
        uint32_t expected_wqe =
            (static_cast<uint32_t>(consumed) * 2u) &
            (zxdh_sq_wqe_cnt - 1);
        uint8_t expected_wqe_polarity =
            !((expected_wqe >> (__ffs(zxdh_sq_wqe_cnt) - 1)) & 1);
        if (wqe_in_cqe != expected_wqe ||
            polarity_in_cqe != expected_wqe_polarity) {
          printf("ZXDH CQE/WQE mismatch: consumed=%llu cqe_wqe=%u "
                 "expected_wqe=%u cqe_pol=%u expected_pol=%u\n",
                 static_cast<unsigned long long>(consumed), wqe_in_cqe,
                 expected_wqe, polarity_in_cqe, expected_wqe_polarity);
          abort();
        }
        __hip_atomic_store(&zxdh_cq_consumed, consumed + 1, __ATOMIC_RELAXED,
                           __HIP_MEMORY_SCOPE_AGENT);
      }
      __threadfence();
      __hip_atomic_store(&zxdh_poll_lock, 0, __ATOMIC_RELEASE,
                         __HIP_MEMORY_SCOPE_AGENT);
    }

    if (zxdh::global_time() - start > zxdh::kPollTimeout) {
      printf("ZXDH CQ poll timeout: target=%llu consumed=%llu\n",
             static_cast<unsigned long long>(target),
             static_cast<unsigned long long>(
                 __hip_atomic_load(&zxdh_cq_consumed, __ATOMIC_RELAXED,
                                   __HIP_MEMORY_SCOPE_AGENT)));
      abort();
    }
  }
}

__device__ void QueuePair::zxdh_quiet_raw() {
  zxdh_poll_cq(__hip_atomic_load(&zxdh_sq_ready_head, __ATOMIC_RELAXED,
                                 __HIP_MEMORY_SCOPE_AGENT) /
               2);
}

__device__ void QueuePair::zxdh_quiet_with_cst() {
  zxdh_quiet_raw();
  if (zxdh_loopback_qps && this != &zxdh_loopback_qps[0] &&
      this != &zxdh_loopback_qps[1]) {
    zxdh_loopback_qps[0].zxdh_post_loopback_cst();
  }
}

__device__ void QueuePair::zxdh_wait_for_sq_space(uint64_t base,
                                                  uint32_t count) {
  while (base + count >
         __hip_atomic_load(&zxdh_cq_consumed, __ATOMIC_RELAXED,
                           __HIP_MEMORY_SCOPE_AGENT) *
                 2 +
             zxdh_sq_wqe_cnt) {
    zxdh_poll_cq(
        __hip_atomic_load(&zxdh_sq_ready_head, __ATOMIC_RELAXED,
                          __HIP_MEMORY_SCOPE_AGENT) /
        2);
  }
}

__device__ void QueuePair::zxdh_submit_requests(uint64_t base,
                                                uint32_t count) {
  const uint64_t new_ready = base + count;
  while (atomicCAS(reinterpret_cast<unsigned long long*>(&zxdh_sq_ready_head),
                   static_cast<unsigned long long>(base),
                   static_cast<unsigned long long>(new_ready)) != base) {
  }

  __threadfence();
  if (gpuHdpReg) {
    __hip_atomic_store(gpuHdpReg, 1u, __ATOMIC_SEQ_CST,
                       __HIP_MEMORY_SCOPE_SYSTEM);
  }
  zxdh_ring_doorbell();
}

__device__ void QueuePair::zxdh_build_rma_wqe(uint64_t wqe_idx,
                                              uintptr_t laddr,
                                              uintptr_t raddr,
                                              uint32_t size,
                                              uint8_t opcode,
                                              uint32_t selected_lkey) {
  zxdh_build_rma_wqe_with_key(wqe_idx, laddr, raddr, size, opcode,
                              selected_lkey, rkey);
}

__device__ void QueuePair::zxdh_build_rma_wqe_with_key(
    uint64_t wqe_idx, uintptr_t laddr, uintptr_t raddr, uint32_t size,
    uint8_t opcode, uint32_t selected_lkey, uint32_t remote_key) {
  SegmentBuilder_ZXDH seg_build(wqe_idx, zxdh_sq_buf, zxdh_sq_wqe_cnt);
  seg_build.update_rma_ctrl_seg(opcode, remote_key);
  seg_build.update_raddr_seg(raddr);
  seg_build.update_data_seg(laddr, size, selected_lkey);
  seg_build.update_nop_seg();
  seg_build.commit();
}

__device__ void QueuePair::zxdh_post_wqe_rma(int32_t size, uintptr_t laddr,
                                             uintptr_t raddr, uint8_t opcode,
                                             int flag) {
  if (size == 0) {
    return;
  }

  const uint64_t activemask = get_active_lane_mask();
  const uint8_t lanes = get_active_lane_count(activemask);
  const uint8_t leader = get_first_active_lane_id(activemask);
  const uint32_t count = 2u * lanes;

  uint64_t base = 0;
  if (is_first_active_lane(activemask)) {
    base = atomicAdd(
        reinterpret_cast<unsigned long long*>(&zxdh_sq_resv_head),
        static_cast<unsigned long long>(count));
  }
  base = __shfl(base, leader);

  zxdh_wait_for_sq_space(base, count);
  const uint64_t my_wqe = base + 2u * get_active_lane_num(activemask);
  const uint32_t lkey = flag == 0 ? lkey_hdp : lkey;
  zxdh_build_rma_wqe(my_wqe, laddr, raddr, static_cast<uint32_t>(size),
                     opcode, lkey);

  __atomic_signal_fence(__ATOMIC_SEQ_CST);
  if (get_active_lane_num(activemask) == lanes - 1) {
    zxdh_submit_requests(base, count);
  }
}

__device__ uint64_t QueuePair::zxdh_reserve_amo_slot() {
  uint64_t slot_idx =
      atomicAdd(reinterpret_cast<unsigned long long *>(&zxdh_amo_head), 1ULL);

  while (slot_idx + 1 -
             __hip_atomic_load(&zxdh_amo_tail, __ATOMIC_RELAXED,
                               __HIP_MEMORY_SCOPE_AGENT) >
         FETCHING_ATOMIC_CNT) {
  }
  return slot_idx;
}

__device__ void QueuePair::zxdh_release_amo_slot(uint64_t slot_idx) {
  __threadfence();
  while (atomicCAS(reinterpret_cast<unsigned long long *>(&zxdh_amo_tail),
                   slot_idx, slot_idx + 1) != slot_idx) {
  }
}

__device__ uint64_t QueuePair::zxdh_post_wqe_amo(
    uintptr_t raddr, uint8_t opcode, int64_t atomic_data, int64_t atomic_cmp,
    bool fetch) {
  const uint64_t slot_idx = zxdh_reserve_amo_slot();
  uint64_t *scratch =
      fetching_atomic + (slot_idx & (FETCHING_ATOMIC_CNT - 1));

  constexpr uint32_t count = 2;
  uint64_t base = atomicAdd(
      reinterpret_cast<unsigned long long *>(&zxdh_sq_resv_head),
      static_cast<unsigned long long>(count));
  zxdh_wait_for_sq_space(base, count);
  zxdh_build_rma_wqe_with_key(
      base, reinterpret_cast<uintptr_t>(scratch), raddr, sizeof(uint64_t),
      gda_op_rdma_read, fetching_atomic_lkey, rkey);
  __threadfence();
  zxdh_submit_requests(base, count);
  if (fetch) {
    zxdh_quiet_with_cst();
  } else {
    zxdh_quiet_raw();
  }

  uint64_t old_value =
      __hip_atomic_load(scratch, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  bool should_write = true;
  uint64_t new_value = old_value;
  if (opcode == gda_op_atomic_cs) {
    should_write =
        old_value == static_cast<uint64_t>(atomic_cmp);
    new_value = static_cast<uint64_t>(atomic_data);
  } else {
    new_value = old_value + static_cast<uint64_t>(atomic_data);
  }

  if (should_write) {
    *scratch = new_value;
    __threadfence();
    base = atomicAdd(
        reinterpret_cast<unsigned long long *>(&zxdh_sq_resv_head),
        static_cast<unsigned long long>(count));
    zxdh_wait_for_sq_space(base, count);
    zxdh_build_rma_wqe_with_key(
        base, reinterpret_cast<uintptr_t>(scratch), raddr, sizeof(uint64_t),
        gda_op_rdma_write, fetching_atomic_lkey, rkey);
    __threadfence();
    zxdh_submit_requests(base, count);
    zxdh_quiet_raw();
  }

  zxdh_release_amo_slot(slot_idx);
  return fetch ? old_value : 0;
}

__device__ void QueuePair::zxdh_post_loopback_cst() {
  if (!zxdh_loopback_qps) {
    return;
  }

  QueuePair *peer = zxdh_loopback_qps + 1;
  if (this == peer) {
    peer = zxdh_loopback_qps;
  }

  constexpr uint32_t count = 2;
  uint64_t base = atomicAdd(
      reinterpret_cast<unsigned long long *>(&zxdh_sq_resv_head),
      static_cast<unsigned long long>(count));
  zxdh_wait_for_sq_space(base, count);
  zxdh_build_rma_wqe_with_key(
      base, reinterpret_cast<uintptr_t>(nonfetching_atomic),
      reinterpret_cast<uintptr_t>(peer->nonfetching_atomic), 1,
      gda_op_rdma_read, nonfetching_atomic_lkey,
      peer->nonfetching_atomic_rkey);
  __threadfence();
  zxdh_submit_requests(base, count);
  zxdh_quiet_raw();
}

#if defined(GDA_ZXDH)
__device__ void QueuePair::get_nbi_with_cst(
    void *dest, const void *source, size_t nelems, int pe, Collectivity cy) {
  if (zxdh_loopback_qps && this != &zxdh_loopback_qps[0] &&
      this != &zxdh_loopback_qps[1] &&
      (cy == THREAD || is_thread_zero_in_wave())) {
    zxdh_loopback_qps[0].zxdh_post_loopback_cst();
  }
  get_nbi(dest, source, nelems, pe, cy);
}

__device__ void QueuePair::ensure_consistency_at_target() {
  if (zxdh_loopback_qps) {
    zxdh_loopback_qps[0].zxdh_post_loopback_cst();
  }
}
#endif

}  // namespace rocshmem
