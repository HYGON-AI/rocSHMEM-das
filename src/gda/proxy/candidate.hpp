// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#ifndef LIBRARY_SRC_GDA_PROXY_CANDIDATE_HPP_
#define LIBRARY_SRC_GDA_PROXY_CANDIDATE_HPP_

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "gda/proxy/route_plan_host.hpp"

namespace rocshmem::proxy {

/**
 * @brief Build this PE's proxy candidates from the NICs selected by GDA.
 *
 * @param selected_nics NIC names in the priority order returned by select_nics().
 * @param my_pe This PE's world rank.
 * @param ipc_domain_first_pe First world PE in this PE's IPC domain.
 * @return One candidate for each selected NIC.
 */
inline std::vector<ProxyNicCandidate> BuildLocalNicCandidates(const std::vector<std::string> &selected_nics,
                                                              int my_pe, int ipc_domain_first_pe) {
  std::vector<ProxyNicCandidate> candidates;

  for (size_t nic_index = 0; nic_index < selected_nics.size(); nic_index++) {
    // Use the same order as select_nics(): it is the actual QP/NIC set and
    // already reflects OAM auto-topology for the default single-NIC path.
    candidates.push_back({my_pe, ipc_domain_first_pe, selected_nics[nic_index], static_cast<int>(nic_index)});
  }

  return candidates;
}

/**
 * @brief Exchange variable-length proxy candidate lists across all PEs.
 *
 * @tparam AllGather Callback type with signature (buffer, candidate_bytes_per_pe).
 * @param local_candidates Candidates advertised by this PE.
 * @param my_pe This PE's world rank.
 * @param num_pes Number of world PEs.
 * @param all_gather In-place fixed-size all-gather callback.
 * @return Validated candidates from every PE.
 */
template <typename AllGather>
std::vector<ProxyNicCandidate> GatherProxyNicCandidates(const std::vector<ProxyNicCandidate> &local_candidates,
                                                        int my_pe, int num_pes, AllGather all_gather) {
  if (num_pes <= 0 || my_pe < 0 || my_pe >= num_pes ||
      local_candidates.size() > static_cast<size_t>(std::numeric_limits<int>::max())) {
    throw std::invalid_argument("invalid proxy candidate count or PE");
  }

  std::vector<int> candidate_counts(num_pes, 0);
  candidate_counts[my_pe] = static_cast<int>(local_candidates.size());
  all_gather(candidate_counts.data(), sizeof(int));

  const int max_candidates_per_pe = *std::max_element(candidate_counts.begin(), candidate_counts.end());
  if (max_candidates_per_pe <= 0) {
    return {};
  }

  struct ProxyNicCandidateWire {
    int proxy_pe;
    int ipc_domain_first_pe;
    int nic_priority;
    char nic_name[64];
  };

  const size_t candidate_bytes_per_pe = static_cast<size_t>(max_candidates_per_pe) * sizeof(ProxyNicCandidateWire);
  if (candidate_bytes_per_pe > static_cast<size_t>(std::numeric_limits<int>::max())) {
    throw std::overflow_error("proxy candidate exchange exceeds MPI count");
  }

  std::vector<ProxyNicCandidateWire> gathered(static_cast<size_t>(num_pes) * max_candidates_per_pe);

  for (size_t i = 0; i < local_candidates.size(); i++) {
    const auto &candidate = local_candidates[i];
    auto &wire = gathered[static_cast<size_t>(my_pe) * max_candidates_per_pe + i];
    if (candidate.nic_name.size() >= sizeof(wire.nic_name)) {
      throw std::invalid_argument("NIC name exceeds proxy wire capacity");
    }
    wire.proxy_pe = candidate.proxy_pe;
    wire.ipc_domain_first_pe = candidate.ipc_domain_first_pe;
    wire.nic_priority = candidate.nic_priority;
    std::memcpy(wire.nic_name, candidate.nic_name.c_str(), candidate.nic_name.size() + 1);
  }

  all_gather(gathered.data(), candidate_bytes_per_pe);

  std::vector<ProxyNicCandidate> all_candidates;

  for (int pe = 0; pe < num_pes; pe++) {
    if (candidate_counts[pe] < 0 || candidate_counts[pe] > max_candidates_per_pe) {
      throw std::runtime_error("invalid proxy candidate exchange count");
    }
    for (int i = 0; i < candidate_counts[pe]; i++) {
      const auto &wire = gathered[static_cast<size_t>(pe) * max_candidates_per_pe + i];
      if (wire.proxy_pe != pe || std::memchr(wire.nic_name, '\0', sizeof(wire.nic_name)) == nullptr) {
        throw std::runtime_error("invalid proxy candidate exchange record");
      }
      all_candidates.push_back({wire.proxy_pe, wire.ipc_domain_first_pe, wire.nic_name, wire.nic_priority});
    }
  }

  return all_candidates;
}

}  // namespace rocshmem::proxy

#endif  // LIBRARY_SRC_GDA_PROXY_CANDIDATE_HPP_