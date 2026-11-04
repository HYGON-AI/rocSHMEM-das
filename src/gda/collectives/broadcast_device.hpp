// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#pragma once

#if defined(USE_DAS)

namespace rocshmem {

/*
 * One broadcast protocol for WORLD and partial teams:
 * stage at source proxies -> PUT to destination proxies -> local IPC fan-out.
 * Routes are selected once per operation. Fixed synchronization proxies are
 * independent of these data proxies. All three phases reserve one epoch each.
 */
template <typename T>
__device__ __noinline__ bool GDAContext::internal_broadcast_wg_pxn(GDATeam *team_obj, T *dst, const T *src,
                                                                             int nelems, int pe_root, int pe_start,
                                                                             int stride, int pe_size, long *p_sync,
                                                                             ActiveWFInfo &wf_info,
                                                                             int64_t proxy_epoch_base,
                                                                             int proxy_qp_row) {  // NOLINT(runtime/int)
  return internal_broadcastmem_wg_pxn(team_obj, dst, src, static_cast<size_t>(nelems) * sizeof(T), pe_root, pe_start,
                                        stride, pe_size, p_sync, wf_info, proxy_epoch_base, proxy_qp_row);
}

template <typename T>
__device__ __noinline__ void GDAContext::internal_broadcast_wg_dispatch(T *dst, const T *src, int nelems, int pe_root,
                                                               int pe_start, int stride, int pe_size, long *p_sync,
                                                               int64_t proxy_epoch_base, int proxy_qp_row,
                                                               GDATeam *team_obj) {  // NOLINT(runtime/int)
  if (internal_try_collective_pxn(team_obj, pe_start, stride, pe_size, "broadcast",
      [&] __device__() {
        ActiveWFInfo wf_info(ctx_id_, ThreadScope::wg);
        return internal_broadcast_wg_pxn(team_obj, dst, src, nelems, pe_root, pe_start, stride, pe_size,
                                                   p_sync, wf_info, proxy_epoch_base, proxy_qp_row);
      })) {
    return;
  }

  internal_broadcastmem_wg_nopxn(dst, src, nelems * sizeof(T), pe_root, pe_start, stride, pe_size, p_sync);
}

}  // namespace rocshmem

#endif  // USE_DAS
