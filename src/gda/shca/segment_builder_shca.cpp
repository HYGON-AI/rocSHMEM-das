// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#include "segment_builder_shca.hpp"

#include "util.hpp"

namespace rocshmem {

__device__ SegmentBuilder_SHCA::SegmentBuilder_SHCA(uint64_t wqe_idx, void *base) {
  shca_segment *base_ptr = static_cast<shca_segment*>(base);
  size_t segment_offset = wqe_idx * SEGMENTS_PER_WQE;
  segp = &base_ptr[segment_offset];
}

/*
 * Control segment - contains some control information for the current WQE.
 *
 * Output:
 *      seg       - control segment to be filled
 * Input:
 *      pi        - WQEBB number of the first block of this WQE.
 *                  This number should wrap at 0xffff, regardless of
 *                  size of the WQ.
 *      opcode    - Opcode of this WQE. Encodes the type of operation
 *                  to be executed on the QP.
 *      opmod     - Opcode modifier.
 *      qp_num    - QP/SQ number this WQE is posted to.
 *      fm_ce_se  - FM (fence mode), CE (completion and event mode)
 *                  and SE (solicited event).
 *      ds        - WQE size in octowords (16-byte units). DS accounts for all
 *                  the segments in the WQE as summarized in WQE construction.
 *      signature - WQE signature.
 *      imm       - Immediate data/Invalidation key/UMR mkey.
 */
__device__ void SegmentBuilder_SHCA::update_ctrl_seg(uint16_t pi, uint8_t opcode, uint8_t opmod, uint32_t qp_num, uint8_t fm_ce_se, uint8_t ds, uint8_t signature, uint32_t imm) {
  segp->ctrl_seg = {0};
  segp->ctrl_seg.opcode = opcode;
  segp->ctrl_seg.wrid = pi;
  segp->ctrl_seg.bf_flag = 0;
  segp->ctrl_seg.qpn = qp_num;
  segp->ctrl_seg.ds = ds;
  segp->ctrl_seg.fm_ce_se = fm_ce_se;
  segp->ctrl_seg.signature = signature;
  segp->ctrl_seg.imm = imm;
  segp++;
}

__device__ void SegmentBuilder_SHCA::update_raddr_seg(uint64_t raddr, uint32_t rkey) {
  segp->raddr_seg = {0};
  segp->raddr_seg.raddr = raddr;
  segp->raddr_seg.rkey = rkey;
  segp++;
}

/*
 * Data Segments - contain pointers and a byte count for the scatter/gather list.
 * They can optionally contain data, which will save a memory read access for
 * gather Work Requests.
 */
__device__ void SegmentBuilder_SHCA::update_data_seg(uint64_t laddr, uint32_t size, uint32_t lkey) {
  segp->data_seg = {0};
  segp->data_seg.byte_count = size;
  segp->data_seg.lkey = lkey;
  segp->data_seg.addr = laddr;
  segp++;
}

__device__ void SegmentBuilder_SHCA::update_inl_data_seg(const void* laddr, int32_t size) {
  segp->inl_data_seg.length = size;
  segp->inl_data_seg.is_inline = SHCA_WQE_SEND_INLINE;
  memcpy(&segp->inl_data_seg + 1, laddr, size);
  segp++;
}

__device__ void SegmentBuilder_SHCA::update_atomic_seg(uint64_t atomic_data, uint64_t atomic_cmp) {
  segp->atomic_seg = {0};
  segp->atomic_seg.swap_add = atomic_data;
  segp->atomic_seg.compare = atomic_cmp;
  segp++;
}

}  // namespace rocshmem
