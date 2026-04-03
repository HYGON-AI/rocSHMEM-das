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

#include "segment_builder.hpp"

#include "util.hpp"

namespace rocshmem {

__device__ SegmentBuilder::SegmentBuilder(uint64_t wqe_idx, void *base) {
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
__device__ void SegmentBuilder::update_ctrl_seg(uint16_t pi, uint8_t opcode, uint8_t opmod, uint32_t qp_num, uint8_t fm_ce_se, uint8_t ds, uint8_t signature, uint32_t imm) {
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

__device__ void SegmentBuilder::update_raddr_seg(uint64_t raddr, uint32_t rkey) {
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
__device__ void SegmentBuilder::update_data_seg(uint64_t laddr, uint32_t size, uint32_t lkey) {
  segp->data_seg = {0};
  segp->data_seg.byte_count = size;
  segp->data_seg.lkey = lkey;
  segp->data_seg.addr = laddr;
  segp++;
}

__device__ void SegmentBuilder::update_inl_data_seg(const void* laddr, int32_t size) {
  segp->inl_data_seg.length = size;
  segp->inl_data_seg.is_inline = SHCA_WQE_SEND_INLINE;
  memcpy(&segp->inl_data_seg + 1, laddr, size);
  segp++;
}

__device__ void SegmentBuilder::update_atomic_seg(uint64_t atomic_data, uint64_t atomic_cmp) {
  segp->atomic_seg = {0};
  segp->atomic_seg.swap_add = atomic_data;
  segp->atomic_seg.compare = atomic_cmp;
  segp++;
}

}  // namespace rocshmem
