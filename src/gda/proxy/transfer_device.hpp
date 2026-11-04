// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#pragma once

#if defined(USE_DAS)

namespace rocshmem {

// Proxy transfer boundaries.
// Proxy routing supplies a world PE and a QP row. Reuse the GDA transport's
// IPC selection and completion handling for every collective.
__device__ inline __noinline__ void GDAContext::internal_proxy_putmem_wg(void *dst, const void *src, size_t bytes,
                                                                         int peer, int qp_row, ActiveWFInfo &wf_info) {
  const int qp_index = qp_row * constmem.num_pes + peer;
  internal_putmem_wg(dst, src, bytes, peer, qp_index, wf_info);
}

// Reading uses the same PE/QP-row contract as the proxy send entry.
__device__ inline __noinline__ void GDAContext::internal_proxy_getmem_wg(void *dst, const void *src, size_t bytes,
                                                                         int peer, int qp_row, ActiveWFInfo &wf_info) {
  const int qp_index = qp_row * constmem.num_pes + peer;
  internal_getmem_wg(dst, src, bytes, peer, qp_index, wf_info);
}

}  // namespace rocshmem

#endif  // USE_DAS
