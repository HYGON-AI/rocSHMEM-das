// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#ifndef LIBRARY_SRC_GDA_ZXDH_SEGMENT_BUILDER_HPP_
#define LIBRARY_SRC_GDA_ZXDH_SEGMENT_BUILDER_HPP_

#include <cstdint>
#include <hip/hip_runtime.h>

namespace rocshmem {

class SegmentBuilder_ZXDH {
 public:
  __device__ SegmentBuilder_ZXDH(uint64_t wqe_idx, void *base,
                                 uint32_t depth);

  __device__ void update_rma_ctrl_seg(uint8_t opcode, uint32_t rkey);
  __device__ void update_inline_ctrl_seg(uint32_t size, uint32_t rkey);
  __device__ void update_raddr_seg(uint64_t raddr);
  __device__ void update_data_seg(uint64_t laddr, uint32_t size,
                                  uint32_t lkey);
  __device__ void update_inl_data_seg(const void *source, uint32_t size);
  __device__ void update_nop_seg();
  __device__ void commit();

 private:
  uint64_t local_wqe_[8]{};
  uint64_t *wqe_{nullptr};
  uint8_t polarity_{0};
};

}  // namespace rocshmem

#endif  // LIBRARY_SRC_GDA_ZXDH_SEGMENT_BUILDER_HPP_
