// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#pragma once

#if defined(USE_DAS)
#include "gda/proxy/route_device.hpp"

namespace rocshmem {
__device__ inline void GDAContext::internal_require_pxn_route(GDATeam *team, int pe_start, int stride,
                                                              int pe_size, const char *operation) {
  const bool world = pe_start == 0 && stride == 1 && pe_size == constmem.num_pes;
  const bool cross_ipc_domain = world ? constmem.world_proxy_ipc_domain_count > 1 : proxy::HasInTeamProxyRoute(team);
  if (cross_ipc_domain) {
    LOGD_ERROR_ABORT("PXN %s has no supported proxy algorithm for this team", operation);
  }
}

template <typename TryProxy>
__device__ inline bool GDAContext::internal_try_collective_pxn(GDATeam *team, int pe_start, int stride, int pe_size,
                                                               const char *operation, TryProxy try_proxy) {
  if (!constmem.pxn_enabled) {
    return false;
  }

  if (try_proxy()) {
    return true;
  }

  internal_require_pxn_route(team, pe_start, stride, pe_size, operation);

  return false;  // Only fallback within one IPC domain is permitted in PXN mode.
}

}

#endif  // USE_DAS
