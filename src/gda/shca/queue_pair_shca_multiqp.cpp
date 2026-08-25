// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#include "gda/queue_pair.hpp"
#include "containers/free_list_impl.hpp"

namespace rocshmem {

__device__ void QueuePair::shca_post_wqe_rma_single_dp(
    int32_t length, uintptr_t laddr, uint32_t local_key,
    uintptr_t raddr, uint32_t remote_key, uint8_t opcode, bool ring_db) {
  acquire_lock(&shca_sq_lock);
  uint64_t sq_counter = shca_sq_posted++;
  shca_wait_for_free_sq_slots(sq_counter, 1);
  uint64_t sq_index = sq_counter % shca_sq_wqe_cnt;
  shca_build_rma_wqe(sq_counter, sq_index, laddr, local_key,
                     raddr, remote_key, length, opcode);
  __atomic_signal_fence(__ATOMIC_SEQ_CST);
  if (ring_db) shca_ring_doorbell(sq_counter, static_cast<uint8_t>(1));
  release_lock(&shca_sq_lock);
}

__device__ uint64_t QueuePair::shca_post_wqe_amo_single(
    uintptr_t raddr, uint32_t remote_key, uint8_t opcode,
    int64_t atomic_data, int64_t atomic_cmp, bool fetching, bool fence) {
  (void)fence;
  acquire_lock(&shca_sq_lock);
  uint64_t sq_counter = shca_sq_posted++;
  shca_wait_for_free_sq_slots(sq_counter, 1);
  uint64_t sq_index = sq_counter % shca_sq_wqe_cnt;
  uint64_t *atomic_addr = nonfetching_atomic;
  if (fetching) {
    auto res = fetching_atomic_freelist->pop_front();
    while (!res.success) res = fetching_atomic_freelist->pop_front();
    atomic_addr = res.value;
  }
  shca_build_amo_wqe(sq_counter, sq_index, raddr, remote_key, opcode,
                     atomic_data, atomic_cmp, fetching, atomic_addr);
  __atomic_signal_fence(__ATOMIC_SEQ_CST);
  shca_ring_doorbell(sq_counter, static_cast<uint8_t>(1));
  release_lock(&shca_sq_lock);
  uint64_t result = 0;
  if (fetching) {
    shca_quiet_single();
    result = *atomic_addr;
    fetching_atomic_freelist->push_back(atomic_addr);
  }
  return result;
}

__device__ void QueuePair::shca_post_wqe_rma_single_lane_dp(
    int32_t size, uintptr_t laddr, uintptr_t raddr, uint8_t opcode) {
  shca_post_wqe_rma_single_dp(size, laddr, lkey, raddr, rkey, opcode, true);
}

__device__ void QueuePair::shca_post_wqe_amo_single_lane_dp(
    int32_t size, uintptr_t raddr, uint8_t opcode, int64_t atomic_data,
    int64_t atomic_cmp, bool fetching) {
  (void)size;
  shca_post_wqe_amo_single(raddr, rkey, opcode, atomic_data, atomic_cmp,
                           fetching, false);
}

__device__ void QueuePair::shca_quiet_dp_single_lane() { shca_quiet_single(); }

}  // namespace rocshmem
