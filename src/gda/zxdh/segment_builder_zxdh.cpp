// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#include "segment_builder_zxdh.hpp"

#include <hip/hip_runtime.h>

#include "gda/zxdh/provider_gda_zxdh.hpp"

namespace rocshmem {

namespace {

constexpr uint32_t kSqWqeShift = 5;
constexpr uint32_t kOpNop = 0x00;
constexpr uint64_t kSqOpcode = ((1ULL << 63) - 1) & ~((1ULL << 57) - 1);
constexpr uint64_t kSqSigCompl = 1ULL << 56;
constexpr uint64_t kSqValid = 1ULL << 63;
constexpr uint64_t kSqInlineFlag = 1ULL << 48;
constexpr uint64_t kSqInlineLen = ((1ULL << 48) - 1) & ~((1ULL << 40) - 1);
constexpr uint64_t kSqFirstFragValid = 1ULL;
constexpr uint64_t kSqFirstFragLen = ((1ULL << 32) - 1) & ~1ULL;
constexpr uint64_t kSqFirstFragStag = ~((1ULL << 32) - 1);
constexpr uint64_t kSqFragTo = ~0ULL;
constexpr uint64_t kSqRemoteStag = ((1ULL << 32) - 1);

__device__ __forceinline__ uint64_t field_prep(uint64_t mask,
                                               uint64_t value) {
  return (value << (__ffsll(static_cast<long long>(mask)) - 1)) & mask;
}

__device__ __forceinline__ uint8_t swqe_polarity(uint64_t wqe_idx,
                                                  uint32_t depth) {
  return !((wqe_idx >> (__ffs(depth) - 1)) & 1);
}

__device__ __forceinline__ void write_wqe(uint64_t local_wqe[8],
                                          uint64_t *wqe) {
  for (int i = 1; i < 8; ++i) {
    __hip_atomic_store(&wqe[i], local_wqe[i], __ATOMIC_RELAXED,
                       __HIP_MEMORY_SCOPE_AGENT);
  }
  __hip_atomic_store(&wqe[0], local_wqe[0], __ATOMIC_RELEASE,
                     __HIP_MEMORY_SCOPE_AGENT);
}

__device__ __forceinline__ void copy_inline_data(uint8_t *dest,
                                                 const uint8_t *source,
                                                 uint32_t size,
                                                 uint8_t polarity) {
  if (size <= 7) {
    for (uint32_t i = 0; i < size; ++i) {
      dest[i] = source[i];
    }
    dest[7] = polarity << 7;
  } else {
    for (uint32_t i = 0; i < 7; ++i) {
      dest[i] = source[i];
    }
    dest[7] = polarity << 7;
    dest[8] = source[7];
  }
}

}  // namespace

__device__ SegmentBuilder_ZXDH::SegmentBuilder_ZXDH(uint64_t wqe_idx,
                                                    void *base,
                                                    uint32_t depth) {
  uint32_t index = wqe_idx & (depth - 1);
  wqe_ = reinterpret_cast<uint64_t *>(
      reinterpret_cast<uintptr_t>(base) +
      (static_cast<uintptr_t>(index) << kSqWqeShift));
  polarity_ = swqe_polarity(wqe_idx, depth);
}

__device__ void SegmentBuilder_ZXDH::update_rma_ctrl_seg(uint8_t opcode,
                                                         uint32_t rkey) {
  local_wqe_[0] = field_prep(kSqValid, polarity_) |
                  field_prep(kSqOpcode, opcode) |
                  field_prep(kSqSigCompl, 1) |
                  field_prep(kSqRemoteStag, rkey);
}

__device__ void SegmentBuilder_ZXDH::update_inline_ctrl_seg(
    uint32_t size, uint32_t rkey) {
  local_wqe_[0] =
      field_prep(kSqValid, polarity_) |
      field_prep(kSqOpcode, ROCSHMEM_ZXDH_OP_WRITE) |
      field_prep(kSqSigCompl, 1) |
      field_prep(kSqInlineFlag, 1) |
      field_prep(kSqInlineLen, size) |
      field_prep(kSqRemoteStag, rkey);
}

__device__ void SegmentBuilder_ZXDH::update_raddr_seg(uint64_t raddr) {
  local_wqe_[3] = field_prep(kSqFragTo, raddr);
}

__device__ void SegmentBuilder_ZXDH::update_data_seg(uint64_t laddr,
                                                     uint32_t size,
                                                     uint32_t lkey) {
  local_wqe_[1] = field_prep(kSqFragTo, laddr);
  local_wqe_[2] = field_prep(kSqFirstFragValid, 0) |
                  field_prep(kSqFirstFragLen, size) |
                  field_prep(kSqFirstFragStag, lkey);
}

__device__ void SegmentBuilder_ZXDH::update_inl_data_seg(
    const void *source, uint32_t size) {
  copy_inline_data(reinterpret_cast<uint8_t *>(&local_wqe_[4]),
                   reinterpret_cast<const uint8_t *>(source), size,
                   polarity_);
}

__device__ void SegmentBuilder_ZXDH::update_nop_seg() {
  local_wqe_[5] = 0;
  local_wqe_[6] = 0;
  local_wqe_[7] = 0;
  local_wqe_[4] = field_prep(kSqOpcode, kOpNop) |
                  field_prep(kSqValid, polarity_);
}

__device__ void SegmentBuilder_ZXDH::commit() {
  write_wqe(local_wqe_, wqe_);
}

}  // namespace rocshmem
