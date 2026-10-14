// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#include "rccl_config.hpp"

#if defined(USE_RCCL)
#include <cerrno>
#include <cctype>
#include <cstdlib>
#include <limits>
#include <optional>
#include <sstream>
#include <string>

#include "envvar.hpp"
#include "log.hpp"

namespace rocshmem {
namespace {

constexpr uint64_t kDefaultRcclMinSize = 1ULL * 1024 * 1024;

std::string trim(std::string value) {
  const auto first = value.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) return {};
  const auto last = value.find_last_not_of(" \t\r\n");
  return value.substr(first, last - first + 1);
}

std::string lowercase(std::string value) {
  for (char& ch : value) {
    ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  }
  return value;
}

bool parse_size(const std::string& input, uint64_t* value) {
  const std::string text = lowercase(trim(input));
  if (text.empty() || text.front() == '-' || value == nullptr) return false;

  errno = 0;
  char* suffix_start = nullptr;
  const unsigned long long number =
      std::strtoull(text.c_str(), &suffix_start, 10);
  if (errno == ERANGE || suffix_start == text.c_str()) return false;

  const std::string suffix = trim(suffix_start);
  uint64_t multiplier = 1;
  if (suffix.empty() || suffix == "b") {
    multiplier = 1;
  } else if (suffix == "k" || suffix == "kb" || suffix == "kib") {
    multiplier = 1024ULL;
  } else if (suffix == "m" || suffix == "mb" || suffix == "mib") {
    multiplier = 1024ULL * 1024;
  } else if (suffix == "g" || suffix == "gb" || suffix == "gib") {
    multiplier = 1024ULL * 1024 * 1024;
  } else {
    return false;
  }
  if (number > std::numeric_limits<uint64_t>::max() / multiplier) {
    return false;
  }
  *value = static_cast<uint64_t>(number) * multiplier;
  return true;
}

struct RcclCollectiveAlias {
  const char* alias;
  RcclCollective collective;
};

constexpr RcclCollectiveAlias kRcclCollectiveAliases[] = {
    {"broadcast", RcclCollective::BROADCAST},
    {"allreduce", RcclCollective::ALL_REDUCE},
    {"all_reduce", RcclCollective::ALL_REDUCE},
    {"reduce_scatter", RcclCollective::REDUCE_SCATTER},
    {"reducescatter", RcclCollective::REDUCE_SCATTER},
    {"allgather", RcclCollective::ALL_GATHER},
    {"all_gather", RcclCollective::ALL_GATHER},
    {"fcollect", RcclCollective::ALL_GATHER},
    {"reduce", RcclCollective::REDUCE},
    {"gather", RcclCollective::GATHER},
    {"scatter", RcclCollective::SCATTER},
    {"alltoall", RcclCollective::ALL_TO_ALL},
    {"all_to_all", RcclCollective::ALL_TO_ALL},
};

bool parse_collective(const std::string& input,
                      RcclCollective* out_collective) {
  const std::string alias = lowercase(trim(input));
  if (out_collective == nullptr) return false;
  for (const auto& entry : kRcclCollectiveAliases) {
    if (alias == entry.alias) {
      *out_collective = entry.collective;
      return true;
    }
  }
  return false;
}

struct ParsedRcclMinSizePolicy {
  uint64_t default_size{kDefaultRcclMinSize};
  std::array<std::optional<uint64_t>, kRcclCollectiveCount> overrides{};
};

void parse_min_size_entry(const std::string& entry,
                          ParsedRcclMinSizePolicy& policy) {
  const size_t separator = entry.find(':');
  uint64_t size = 0;
  if (separator == std::string::npos ||
      !parse_size(entry.substr(separator + 1), &size)) {
    LOG_WARN("Invalid RCCL minimum-size policy entry '%s'; ignoring it",
             entry.c_str());
    return;
  }

  const std::string alias = lowercase(trim(entry.substr(0, separator)));
  if (alias == "default") {
    policy.default_size = size;
    return;
  }
  if (alias == "alltoallv" || alias == "all_to_all_v") {
    LOG_WARN("RCCL alltoallv has device-resident counts; its size threshold "
             "is ignored");
    return;
  }

  RcclCollective collective{};
  if (!parse_collective(alias, &collective)) {
    LOG_WARN("Unknown RCCL collective '%s' in minimum-size policy; "
             "ignoring it",
             alias.c_str());
    return;
  }
  policy.overrides[static_cast<size_t>(collective)] = size;
}

ParsedRcclMinSizePolicy parse_min_size_policy(const std::string& value, const char* name) {
  ParsedRcclMinSizePolicy result;
  const std::string policy = trim(value);

  // A scalar sets the threshold for every collective in this scope.
  if (policy.find(':') == std::string::npos &&
      policy.find(',') == std::string::npos) {
    if (!parse_size(policy, &result.default_size)) {
      LOG_WARN("Invalid %s value '%s'; using 1M",
               name, policy.c_str());
    }
    return result;
  }

  std::istringstream stream(policy);
  for (std::string entry; std::getline(stream, entry, ',');) {
    entry = trim(entry);
    if (!entry.empty()) parse_min_size_entry(entry, result);
  }
  return result;
}

RcclRuntimeConfig make_runtime_config() {
  RcclRuntimeConfig config{};
  config.disabled = envvar::disable_rccl.get_value() ? 1 : 0;
  config.delay_init = envvar::rccl_delay_init.get_value() ? 1 : 0;

  const std::string values[] = {envvar::rccl_min_size_intra.get_value(),
                                envvar::rccl_min_size_inter.get_value()};
  const char* names[] = {"ROCSHMEM_RCCL_MIN_SIZE_INTRA",
                         "ROCSHMEM_RCCL_MIN_SIZE_INTER"};
  for (size_t scope = 0; scope < 2; ++scope) {
    const auto policy = parse_min_size_policy(values[scope], names[scope]);
    auto& thresholds = config.min_sizes[scope];
    thresholds.fill(policy.default_size);
    for (size_t i = 0; i < kRcclCollectiveCount; ++i) {
      if (policy.overrides[i]) thresholds[i] = *policy.overrides[i];
    }
  }
  return config;
}

}  // namespace

const RcclRuntimeConfig& rccl_runtime_config() {
  static const RcclRuntimeConfig config = make_runtime_config();
  return config;
}

}  // namespace rocshmem
#endif  // USE_RCCL
