// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#include "gda/queue_pair.hpp"
#include "segment_builder_zxdh.hpp"

namespace rocshmem {

__device__ void QueuePair::zxdh_build_inline_wqe(uint64_t wqe_idx,
                                                 uintptr_t raddr,
                                                 uint64_t value,
                                                 uint32_t remote_key) {
  SegmentBuilder_ZXDH seg_build(wqe_idx, zxdh_sq_buf, zxdh_sq_wqe_cnt);
  seg_build.update_inline_ctrl_seg(sizeof(value), remote_key);
  seg_build.update_raddr_seg(raddr);
  seg_build.update_inl_data_seg(&value, sizeof(value));
  seg_build.commit();
}

__device__ void QueuePair::zxdh_post_value_dp(uintptr_t raddr,
                                              uint64_t value) {
  constexpr uint32_t count = 4;
  uint64_t base = atomicAdd(
      reinterpret_cast<unsigned long long *>(&zxdh_sq_resv_head),
      static_cast<unsigned long long>(count));
  zxdh_wait_for_sq_space(base, count);

  zxdh_build_rma_wqe_with_key(
      base, reinterpret_cast<uintptr_t>(nonfetching_atomic), raddr,
      sizeof(uint64_t), gda_op_rdma_read, nonfetching_atomic_lkey, rkey);
  zxdh_build_inline_wqe(base + 2, raddr, value, rkey);
  __threadfence();
  zxdh_submit_requests(base, count);
}

__device__ void QueuePair::zxdh_quiet_dp_single_lane() {
  zxdh_quiet_raw();
}

__device__ void QueuePair::zxdh_post_wqe_rma_dp_single_lane(
    int32_t size, uintptr_t laddr, uintptr_t raddr, uint8_t opcode) {
  zxdh_post_wqe_rma(size, laddr, raddr, opcode, 1);
}

__device__ uint64_t QueuePair::zxdh_post_wqe_amo_dp_single_lane(
    uintptr_t raddr, uint8_t opcode, int64_t atomic_data, int64_t atomic_cmp,
    bool fetch) {
  return zxdh_post_wqe_amo(raddr, opcode, atomic_data, atomic_cmp, fetch);
}

}  // namespace rocshmem
