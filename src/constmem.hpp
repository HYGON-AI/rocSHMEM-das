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

#ifndef LIBRARY_SRC_CONSTMEM_HPP_
#define LIBRARY_SRC_CONSTMEM_HPP_

#include <array>
#include <cstddef>
#include <cstdint>

#include "gda/gda_enums.hpp"
#include "rocshmem/rocshmem_common.hpp"
#include "util.hpp"

namespace rocshmem {

struct constmem_t {
  BackendType backend_type;
  GDAProvider gda_provider;
  uint64_t alltoall_wg_algo;
  int my_pe;
  int num_pes;
  int ipc_first_pe;
  int ipc_stride;    // 0 = pattern invalid (use fallback linear scan)
  int ipc_shm_size;
  /*
   * An IPC domain contains node-local PEs with direct IPC connectivity.
   * A physical node may contain multiple IPC domains.
   */
  bool pxn_enabled;  // Enforce same-name NIC proxy routes when DISABLE_PXN=0.
  bool world_proxy_sync_available;  // Host-validated WORLD proxy sync layout is available.
  // Corresponding PE slots in equal-sized IPC domains own matching NICs.
  bool world_proxy_aligned_data_available;
  int world_proxy_ipc_domain_count;  // Number of IPC domains in the WORLD proxy layout.
  // [IPC domain index] -> first world PE; this is not a physical node ID.
  int world_proxy_ipc_domain_first_pes[ROCSHMEM_BARRIER_SYNC_SIZE];
  bool world_proxy_nic_layout_available;  // WORLD PE-to-NIC-name mapping is available.
  int world_proxy_nic_name_count;  // Number of distinct NIC names in that mapping.
  /*
   * [world PE] -> NIC-name ID, shared across nodes and assigned by sorted name.
   * This ID is not a node-local IB device index. Valid only when the mapping is enabled.
   */
  int world_proxy_pe_nic_name_ids[ROCSHMEM_BARRIER_SYNC_SIZE];
  uintptr_t heap_base;  // Local symmetric heap base
  size_t heap_size;     // Local symmetric heap size in bytes
} __attribute__ ((aligned (16)));

extern __constant__ constmem_t constmem;
void init_constant_memory(void);

}  // namespace rocshmem

#endif  // LIBRARY_SRC_CONSTMEM_HPP_
