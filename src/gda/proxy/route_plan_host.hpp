// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#ifndef LIBRARY_SRC_GDA_PROXY_ROUTE_PLAN_HOST_HPP_
#define LIBRARY_SRC_GDA_PROXY_ROUTE_PLAN_HOST_HPP_

#include <algorithm>
#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace rocshmem::proxy {

// A PE may advertise more than one selected NIC.
struct ProxyNicCandidate {
  int proxy_pe;             // PE that owns this NIC and can act as its proxy.
  int ipc_domain_first_pe;  // World PE of this IPC domain's first member.
  std::string nic_name;
  int nic_priority;  // Position in select_nics() output; zero is preferred.
};

struct ProxyRoute {
  int source_proxy_pe = -1;  // Sender IPC domain's selected NIC owner.
  int target_proxy_pe = -1;  // Receiver IPC domain's same-name NIC owner.
  std::string nic_name;

  explicit operator bool() const { return source_proxy_pe >= 0; }
};

struct ProxyNicGroup {
  std::string nic_name;
  std::vector<ProxyNicCandidate> source_candidates;
  std::vector<ProxyNicCandidate> target_candidates;
};

struct ProxyRouteSchedule {
  std::vector<ProxyNicGroup> nic_groups;

  explicit operator bool() const { return !nic_groups.empty(); }
};

/**
 * @brief Build all load-balanceable same-name routes between two IPC domains.
 *
 * @param candidates Globally exchanged NIC candidates.
 * @param source_ipc_domain_first_pe First world PE in the source IPC domain.
 * @param target_ipc_domain_first_pe First world PE in the target IPC domain.
 * @return NIC groups ordered by topology priority.
 */
inline ProxyRouteSchedule BuildProxyRouteSchedule(const std::vector<ProxyNicCandidate> &candidates,
                                                  int source_ipc_domain_first_pe, int target_ipc_domain_first_pe) {
  ProxyRouteSchedule schedule;
  if (source_ipc_domain_first_pe == target_ipc_domain_first_pe) {
    return schedule;
  }

  std::map<std::string, ProxyNicGroup> by_nic;

  for (const auto &candidate : candidates) {
    if (candidate.proxy_pe < 0 || candidate.nic_name.empty() || candidate.nic_priority < 0) {
      continue;
    }

    auto &group = by_nic[candidate.nic_name];
    group.nic_name = candidate.nic_name;
    if (candidate.ipc_domain_first_pe == source_ipc_domain_first_pe) {
      group.source_candidates.push_back(candidate);
    } else if (candidate.ipc_domain_first_pe == target_ipc_domain_first_pe) {
      group.target_candidates.push_back(candidate);
    }
  }

  auto by_priority = [](const ProxyNicCandidate &a, const ProxyNicCandidate &b) {
    if (a.nic_priority != b.nic_priority) {
      return a.nic_priority < b.nic_priority;
    }
    return a.proxy_pe < b.proxy_pe;
  };

  for (auto &entry : by_nic) {
    auto &group = entry.second;
    if (group.source_candidates.empty() || group.target_candidates.empty()) {
      continue;
    }

    std::sort(group.source_candidates.begin(), group.source_candidates.end(), by_priority);
    std::sort(group.target_candidates.begin(), group.target_candidates.end(), by_priority);
    schedule.nic_groups.push_back(std::move(group));
  }

  std::sort(schedule.nic_groups.begin(), schedule.nic_groups.end(), [](const ProxyNicGroup &a, const ProxyNicGroup &b) {
    const int a_priority = a.source_candidates.front().nic_priority + a.target_candidates.front().nic_priority;
    const int b_priority = b.source_candidates.front().nic_priority + b.target_candidates.front().nic_priority;
    return a_priority == b_priority ? a.nic_name < b.nic_name : a_priority < b_priority;
  });

  return schedule;
}

/**
 * @brief Select one balanced route for a collective transfer index.
 *
 * @param schedule Route schedule shared by all PEs.
 * @param transfer_index Globally consistent collective transfer index.
 * @return A route rotating across NICs and eligible endpoint PEs.
 */
inline ProxyRoute SelectScheduledProxyRoute(const ProxyRouteSchedule &schedule, std::uint64_t transfer_index) {
  if (!schedule) {
    return {};
  }

  const std::uint64_t nic_group_count = schedule.nic_groups.size();
  const auto &group = schedule.nic_groups[transfer_index % nic_group_count];
  const std::uint64_t candidate_round = transfer_index / nic_group_count;
  const auto &source = group.source_candidates[candidate_round % group.source_candidates.size()];
  const auto &target = group.target_candidates[candidate_round % group.target_candidates.size()];

  return {source.proxy_pe, target.proxy_pe, group.nic_name};
}

/**
 * @brief Host-owned PXN discovery result shared by backend setup stages.
 *
 * All layouts derive from the same globally exchanged candidate set.
 */
struct ProxyRoutingState {
  std::vector<ProxyNicCandidate> candidates;

  bool world_sync_available = false;
  bool world_aligned_data_available = false;
  std::vector<int> world_ipc_domain_first_pes;
  std::vector<int> world_pe_nic_name_ids;
  int world_nic_name_count = 0;

  bool has_world_nic_layout() const { return !world_pe_nic_name_ids.empty(); }
};

}  // namespace rocshmem::proxy

#endif  // LIBRARY_SRC_GDA_PROXY_ROUTE_PLAN_HOST_HPP_
