// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#ifndef LIBRARY_SRC_GDA_PROXY_LAYOUT_HPP_
#define LIBRARY_SRC_GDA_PROXY_LAYOUT_HPP_

#include <limits>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "gda/proxy/route_plan_host.hpp"

namespace rocshmem::proxy {

/*
 * An IPC domain is a node-local set of PEs that can communicate directly through IPC.
 * Domain identifiers are world-PE based and do not represent physical host IDs.
 */

struct WorldProxySyncLayout {
  std::vector<int> ipc_domain_first_pes;

  explicit operator bool() const { return ipc_domain_first_pes.size() > 1; }
};

/**
 * @brief Validate the first-PE proxy layout used by WORLD synchronization.
 *
 * @param candidates Globally exchanged NIC candidates.
 * @param routes Selected directional routes between IPC domains.
 * @param num_pes Number of world PEs.
 * @param max_sync_slots Available world-rank synchronization slots.
 * @return Ordered IPC domain first PEs, or an invalid layout.
 */
inline WorldProxySyncLayout BuildWorldProxySyncLayout(const std::vector<ProxyNicCandidate> &candidates,
                                                      const std::map<std::pair<int, int>, ProxyRoute> &routes,
                                                      int num_pes, int max_sync_slots) {
  WorldProxySyncLayout layout;
  if (num_pes < 2 || num_pes > max_sync_slots) {
    return {};
  }

  // Key is the first world PE in an IPC domain, not a hardware node ID.
  std::map<int, std::set<int>> ipc_domain_members;
  std::map<int, std::string> first_pe_nics;
  for (const auto &candidate : candidates) {
    if (candidate.ipc_domain_first_pe < 0 || candidate.proxy_pe < 0 || candidate.proxy_pe >= num_pes ||
        candidate.ipc_domain_first_pe > candidate.proxy_pe) {
      return {};
    }
    ipc_domain_members[candidate.ipc_domain_first_pe].insert(candidate.proxy_pe);
    if (candidate.proxy_pe == candidate.ipc_domain_first_pe && candidate.nic_priority == 0) {
      first_pe_nics[candidate.proxy_pe] = candidate.nic_name;
    }
  }

  if (first_pe_nics[0].empty()) {
    return {};
  }

  int next_pe = 0;

  for (const auto &ipc_domain : ipc_domain_members) {
    const int ipc_domain_first_pe = ipc_domain.first;
    const int local_pe_count = static_cast<int>(ipc_domain.second.size());
    if (ipc_domain_first_pe != next_pe || local_pe_count < 1 || local_pe_count > num_pes - next_pe) {
      return {};
    }

    for (int pe = next_pe; pe < next_pe + local_pe_count; pe++) {
      if (!ipc_domain.second.count(pe)) {
        return {};
      }
    }

    layout.ipc_domain_first_pes.push_back(ipc_domain_first_pe);
    next_pe += local_pe_count;
    if (ipc_domain_first_pe == 0) {
      continue;
    }

    auto forward = routes.find({0, ipc_domain_first_pe});
    auto reverse = routes.find({ipc_domain_first_pe, 0});
    // The QPs connecting IPC domains must belong to the proxy PEs that run the
    // device protocol, in both directions, on the same named NIC.
    if (forward == routes.end() || reverse == routes.end() || forward->second.source_proxy_pe != 0 ||
        forward->second.target_proxy_pe != ipc_domain_first_pe ||
        reverse->second.source_proxy_pe != ipc_domain_first_pe || reverse->second.target_proxy_pe != 0 ||
        forward->second.nic_name != reverse->second.nic_name ||
        first_pe_nics[ipc_domain_first_pe] != first_pe_nics[0] || forward->second.nic_name != first_pe_nics[0]) {
      return {};
    }
  }

  if (next_pe != num_pes || layout.ipc_domain_first_pes.size() < 2) {
    return {};
  }

  return layout;
}

/**
 * @brief Check whether equal local PE slots form valid WORLD data routes.
 *
 * @param candidates Globally exchanged NIC candidates.
 * @param num_pes Number of world PEs.
 * @param local_pe_count Number of PEs in each required equal-size IPC domain.
 * @return true when every corresponding slot owns the same named NIC.
 */
inline bool CanUseAlignedWorldProxyData(const std::vector<ProxyNicCandidate> &candidates, int num_pes,
                                        int local_pe_count) {
  if (local_pe_count <= 1 || num_pes <= local_pe_count || num_pes % local_pe_count != 0) {
    return false;
  }

  std::map<int, std::set<int>> ipc_domain_members;
  std::map<int, std::string> selected_nics;
  for (const auto &candidate : candidates) {
    ipc_domain_members[candidate.ipc_domain_first_pe].insert(candidate.proxy_pe);
    if (candidate.nic_priority == 0) {
      selected_nics[candidate.proxy_pe] = candidate.nic_name;
    }
  }

  if (static_cast<int>(selected_nics.size()) != num_pes ||
      static_cast<int>(ipc_domain_members.size()) != num_pes / local_pe_count) {
    return false;
  }

  for (int ipc_domain_first_pe = 0; ipc_domain_first_pe < num_pes; ipc_domain_first_pe += local_pe_count) {
    auto ipc_domain = ipc_domain_members.find(ipc_domain_first_pe);
    if (ipc_domain == ipc_domain_members.end() || static_cast<int>(ipc_domain->second.size()) != local_pe_count) {
      return false;
    }

    for (int pe = ipc_domain_first_pe; pe < ipc_domain_first_pe + local_pe_count; pe++) {
      if (!ipc_domain->second.count(pe)) {
        return false;
      }
    }
  }

  for (int pe = local_pe_count; pe < num_pes; pe++) {
    const auto &local_nic = selected_nics[pe];
    const auto &first_ipc_domain_nic = selected_nics[pe % local_pe_count];
    if (local_nic.empty() || local_nic != first_ipc_domain_nic) {
      return false;
    }
  }

  return true;
}

struct WorldNicLayout {
  std::vector<std::string> nic_names;
  std::vector<int> pe_nic_name_ids;

  explicit operator bool() const { return !nic_names.empty(); }
};

/**
 * @brief Build deterministic NIC IDs for WORLD multi-lane data routing.
 *
 * This layout accepts exactly one candidate per world PE, all with priority
 * zero. NIC names are sorted so every PE assigns the same ID to a given name.
 *
 * @param candidates Globally exchanged NIC candidates.
 * @param world_pe_count Number of world PEs.
 * @param device_layout_capacity Capacity of the device-side PE/NIC arrays.
 * @return NIC names and one NIC ID per PE, or an invalid layout.
 */
inline WorldNicLayout BuildWorldNicLayout(const std::vector<ProxyNicCandidate> &candidates,
                                          int world_pe_count, int device_layout_capacity) {
  if (world_pe_count < 2 || world_pe_count > device_layout_capacity) {
    return {};
  }

  std::vector<std::string> nic_name_by_pe(world_pe_count);
  std::vector<int> candidate_count_by_pe(world_pe_count, 0);
  std::set<std::string> unique_nic_names;

  for (const auto &candidate : candidates) {
    if (candidate.proxy_pe < 0 || candidate.proxy_pe >= world_pe_count || candidate.nic_priority != 0 ||
        candidate.nic_name.empty()) {
      return {};
    }

    nic_name_by_pe[candidate.proxy_pe] = candidate.nic_name;
    candidate_count_by_pe[candidate.proxy_pe]++;
    unique_nic_names.insert(candidate.nic_name);
  }

  // The PE count is already capacity-checked, and there is one NIC per PE.
  for (int count : candidate_count_by_pe) {
    if (count != 1) {
      return {};
    }
  }

  WorldNicLayout layout;
  layout.nic_names.assign(unique_nic_names.begin(), unique_nic_names.end());
  std::map<std::string, int> nic_id_by_name;

  for (size_t i = 0; i < layout.nic_names.size(); i++) {
    nic_id_by_name[layout.nic_names[i]] = static_cast<int>(i);
  }

  for (const auto &nic_name : nic_name_by_pe) {
    layout.pe_nic_name_ids.push_back(nic_id_by_name[nic_name]);
  }

  return layout;
}

// A partial team can use PXN without external progress when every
// participating IPC domain has a team member that owns the same NIC.
struct InTeamProxyLayout {
  std::string nic_name;
  std::map<int, int> proxy_pe_by_ipc_domain_first_pe;

  explicit operator bool() const { return !nic_name.empty() && !proxy_pe_by_ipc_domain_first_pe.empty(); }
};

/**
 * @brief Test whether a world PE belongs to a strided team.
 *
 * @param pe_start First world PE in the team.
 * @param stride World-rank stride between team members.
 * @param team_size Number of team members.
 * @param world_pe World PE to test.
 * @return true when world_pe is a member of the team.
 */
inline bool TeamContainsPe(int pe_start, int stride, int team_size, int world_pe) {
  if (stride <= 0 || team_size <= 0 || world_pe < pe_start) {
    return false;
  }

  const int distance = world_pe - pe_start;
  return distance % stride == 0 && distance / stride < team_size;
}

/**
 * @brief Select one common in-team NIC proxy in each participating IPC domain.
 *
 * @param candidates Globally exchanged NIC candidates.
 * @param pe_start First world PE in the team.
 * @param stride World-rank stride between team members.
 * @param team_size Number of team members.
 * @return The lowest-priority-sum common-NIC layout, or an invalid layout.
 */
inline InTeamProxyLayout BuildInTeamProxyLayout(const std::vector<ProxyNicCandidate> &candidates,
                                                int pe_start, int stride, int team_size) {
  if (pe_start < 0 || stride <= 0 || team_size <= 0) {
    return {};
  }

  std::set<int> participating_ipc_domains;
  std::set<std::string> nic_names;
  for (const auto &candidate : candidates) {
    if (!TeamContainsPe(pe_start, stride, team_size, candidate.proxy_pe)) {
      continue;
    }

    participating_ipc_domains.insert(candidate.ipc_domain_first_pe);
    if (!candidate.nic_name.empty() && candidate.nic_priority >= 0) {
      nic_names.insert(candidate.nic_name);
    }
  }

  if (participating_ipc_domains.empty()) {
    return {};
  }

  InTeamProxyLayout best;
  int best_priority = std::numeric_limits<int>::max();

  for (const auto &nic_name : nic_names) {
    std::map<int, int> proxies;
    int total_priority = 0;
    for (int ipc_domain_first_pe : participating_ipc_domains) {
      const ProxyNicCandidate *selected = nullptr;
      for (const auto &candidate : candidates) {
        if (candidate.ipc_domain_first_pe != ipc_domain_first_pe || candidate.nic_name != nic_name ||
            candidate.nic_priority < 0 || !TeamContainsPe(pe_start, stride, team_size, candidate.proxy_pe)) {
          continue;
        }

        if (selected == nullptr || candidate.nic_priority < selected->nic_priority ||
            (candidate.nic_priority == selected->nic_priority && candidate.proxy_pe < selected->proxy_pe)) {
          selected = &candidate;
        }
      }

      if (selected == nullptr) {
        proxies.clear();
        break;
      }
      proxies[ipc_domain_first_pe] = selected->proxy_pe;
      total_priority += selected->nic_priority;
    }

    if (proxies.size() != participating_ipc_domains.size()) {
      continue;
    }

    if (!best || total_priority < best_priority) {
      best.nic_name = nic_name;
      best.proxy_pe_by_ipc_domain_first_pe = std::move(proxies);
      best_priority = total_priority;
    }
  }

  return best;
}

}  // namespace rocshmem::proxy

#endif  // LIBRARY_SRC_GDA_PROXY_LAYOUT_HPP_
