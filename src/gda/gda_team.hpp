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

#ifndef LIBRARY_SRC_GDA_TEAM_HPP_
#define LIBRARY_SRC_GDA_TEAM_HPP_

#include "team.hpp"

namespace rocshmem {

class GDATeam : public Team {
 public:
  GDATeam(Backend* handle, const TeamInfo& team_info_wrt_parent,
          const TeamInfo& team_info_wrt_world, int num_pes, int my_pe,
          MPI_Comm team_comm, int pool_index);

  virtual ~GDATeam();

  long* barrier_pSync{nullptr};
  long* reduce_pSync{nullptr};
  long* bcast_pSync{nullptr};
  long* alltoall_pSync{nullptr};
  void* pWrk{nullptr};
  void* pAta{nullptr};

  int pool_index_{-1};
  uint64_t reduce_sequence_number = 0;
  uint64_t alltoall_sequence_number = 0;
  uint64_t proxy_alltoall_sync_epoch = 1;
  uint64_t proxy_broadcast_sync_epoch = 1;
  uint64_t proxy_fcollect_sync_epoch = 1;
  uint64_t proxy_reduce_sync_epoch = 1;
  uint64_t in_team_proxy_sync_epoch = 1;

  // Partial-team PXN layout. Entries are ordered by IPC domain first PE.
  // A zero count is valid only for teams contained in one IPC domain when PXN is enabled.
  int in_team_proxy_ipc_domain_count = 0;
  int in_team_proxy_ipc_domain_first_pes[ROCSHMEM_BARRIER_SYNC_SIZE]{};
  int in_team_proxy_pes[ROCSHMEM_BARRIER_SYNC_SIZE]{};

  // One deterministic sequence for completed data-routing tasks. Synchronization
  // epochs have algorithm-dependent strides and must not drive round-robin turns.
  // Concurrent operations must use distinct teams, as required by pWrk/pSync.
  uint64_t proxy_data_route_sequence = 0;
#if defined(USE_DAS)
  // Only ordinary non-PXN WG broadcasts consume this ready state.
  long *bcast_ready{nullptr};
  uint64_t bcast_sequence_number = 0;
  // Non-PXN WG fcollect never borrows another collective's synchronization.
  long *fcollect_ready{nullptr};
  long *fcollect_pSync{nullptr};
  long *fcollect_control{nullptr};
  uint64_t fcollect_sequence_number = 0;
  uint64_t fcollect_ready_base[2] = {0, 0};
#endif
};

}  // namespace rocshmem

#endif  // LIBRARY_SRC_GDA_TEAM_HPP_
