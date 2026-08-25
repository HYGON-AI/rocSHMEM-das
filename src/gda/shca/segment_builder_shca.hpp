// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#ifndef LIBRARY_SRC_GDA_SHCA_SEGMENT_BUILDER_HPP_
#define LIBRARY_SRC_GDA_SHCA_SEGMENT_BUILDER_HPP_

#include "gda/shca/provider_gda_shca.hpp"

#include "util.hpp"

namespace rocshmem {

class SegmentBuilder_SHCA {
  public:
    __device__ SegmentBuilder_SHCA(uint64_t wqe_idx, void *base);

    __device__ void update_ctrl_seg(uint16_t pi, uint8_t opcode, uint8_t opmod, uint32_t qp_num,
                                    uint8_t fm_ce_se, uint8_t ds, uint8_t signature, uint32_t imm);

    __device__ void update_raddr_seg(uint64_t raddr, uint32_t rkey);

    __device__ void update_data_seg(uint64_t laddr, uint32_t size, uint32_t lkey);

    __device__ void update_inl_data_seg(const void* laddr, int32_t size);

    __device__ void update_atomic_seg(uint64_t atomic_data, uint64_t atomic_cmp);

  private:
    const int SEGMENTS_PER_WQE = 4;

    union shca_segment {
      shca_wqe_ctrl_seg ctrl_seg;
      shca_wqe_raddr_seg raddr_seg;
      shca_wqe_data_seg data_seg;
      shca_wqe_inline_data_seg inl_data_seg;
      shca_wqe_atomic_seg atomic_seg;
    }__attribute__((__aligned__(16)));

    shca_segment *segp;
};

}  // namespace rocshmem

#endif  // LIBRARY_SRC_GDA_SHCA_SEGMENT_BUILDER_HPP_
