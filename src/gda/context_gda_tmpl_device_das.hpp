// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

/******************************************************************************
 * Shared GDA device helpers and inclusion point for DAS collectives.
 *
 * Collective entry points and algorithms are defined in gda/collectives/.
 * Proxy routing, synchronization and transfer helpers reside in gda/proxy/.
 *****************************************************************************/

#pragma once

#include "device_arch.hpp"
#include "rocshmem/rocshmem_config.h"

#if defined(USE_DAS)
#include "gda/proxy/entry_device.hpp"
#include "gda/proxy/transfer_device.hpp"
#include "gda/collectives/common_device.hpp"
#include "gda/collectives/broadcast_device.hpp"
#include "gda/collectives/fcollect_device.hpp"
#include "gda/collectives/alltoall_device.hpp"
#include "gda/collectives/alltoallv_device.hpp"
#include "gda/collectives/reduce_device.hpp"

namespace rocshmem {

// Common GDA QP helpers, shared by native and proxy collective paths.
__device__ __forceinline__ uint32_t GDAContext::get_team_qp_index(
    const GDATeam *team_obj, int pe) {
  const uint32_t qp_slot =
      static_cast<uint32_t>(team_obj->pool_index_) % num_qps_per_pe;
  return qp_slot * constmem.num_pes + pe;
}

/**
 * @brief Select the next same-QP lane group for wave-safe submission.
 * @note Default-context QPs may be shared across workgroups. Processing one
 * QP group at a time avoids a wave blocking on multiple SQ locks.
 */
__device__ __forceinline__ uint64_t GDAContext::get_next_qp_group(
    int qp_index, uint64_t pending_lanes) {
  // Use the first pending lane to select the next QP.
  const int first_lane = __builtin_ctzll(pending_lanes);
  const int selected_qp = __shfl(qp_index, first_lane);
  const int lane_id = get_flat_block_id() % WF_SIZE;
  // Collect all pending lanes mapped to the selected QP.
  return __ballot((pending_lanes & (uint64_t{1} << lane_id)) &&
                  qp_index == selected_qp);
}

/** @brief Submit a PUT to a local symmetric destination through a shared QP. */
__device__ __forceinline__ void GDAContext::put_nbi_shared_qp(
    int qp_index, void *dest, const void *source, size_t length, bool ring_db) {
  uint64_t pending_lanes = get_active_lane_mask();
  const int lane_id = get_flat_block_id() % WF_SIZE;
  // Submit each QP group through the existing wave-batched path.
  while (pending_lanes) {
    const uint64_t qp_group = get_next_qp_group(qp_index, pending_lanes);
    if (qp_group & (uint64_t{1} << lane_id)) {
      ActiveWFInfo wf_info(qp_index);
      qps[qp_index].put_nbi(dest, source, length, wf_info, ring_db);
    }
    // Continue with the remaining QP groups.
    pending_lanes &= ~qp_group;
  }
}

/** @brief Submit a GET safely through a QP shared across workgroups. */
__device__ __forceinline__ void GDAContext::get_nbi_shared_qp(
    int qp_index, void *dest, const void *source, size_t length) {
  uint64_t pending_lanes = get_active_lane_mask();
  const int lane_id = get_flat_block_id() % WF_SIZE;
  // Submit each QP group through the existing wave-batched path.
  while (pending_lanes) {
    const uint64_t qp_group = get_next_qp_group(qp_index, pending_lanes);
    if (qp_group & (uint64_t{1} << lane_id)) {
      ActiveWFInfo wf_info(qp_index);
      qps[qp_index].get_nbi(dest, source, length, wf_info);
    }
    pending_lanes &= ~qp_group;
  }
}

/** @brief Drain shared QPs using the provider's concurrent CQ path. */
__device__ __forceinline__ void GDAContext::quiet_shared_qp(int qp_index) {
  uint64_t pending_lanes = get_active_lane_mask();
  const int lane_id = get_flat_block_id() % WF_SIZE;
  while (pending_lanes) {
    const uint64_t qp_group = get_next_qp_group(qp_index, pending_lanes);
    if (qp_group & (uint64_t{1} << lane_id)) {
      ActiveWFInfo wf_info(qp_index);
      qps[qp_index].quiet(wf_info);
    }
    pending_lanes &= ~qp_group;
  }
}

/** @brief Publish the completion count to a slot owned by this sender. */
__device__ __forceinline__ void GDAContext::publish_completion_shared_qp(
    int qp_index, void *dest, int64_t completion_count) {
  // Each peer owns a separate slot; overwrite its completion count instead of
  // requiring a remote atomic increment. dest is a local symmetric address.
  put_nbi_shared_qp(qp_index, dest, &completion_count, sizeof(completion_count), true);
}

}  // namespace rocshmem

#endif  // USE_DAS
