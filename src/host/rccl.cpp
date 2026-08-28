// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#include "rccl.hpp"

#if defined(USE_RCCL)
#include <dlfcn.h>
#include <atomic>
#include <cstdint>
#include <limits>
#include <mutex>
#include <string>
#include <vector>

#include "bootstrap/bootstrap.hpp"
#include "envvar.hpp"
#include "log.hpp"
#include "mpi_instance.hpp"
#include "team.hpp"

namespace rocshmem {

RcclFunctionTable rccl_ftable{};

enum class RcclCommState { UNINITIALIZED, PREPARED, READY, FAILED };

namespace {
struct RcclRuntimeConfig {
  uint64_t min_size;
  int disabled;
  int delay_init;
};

/** @brief Parse and return the process-wide RCCL runtime configuration. */
const RcclRuntimeConfig& runtime_config() {
  static const RcclRuntimeConfig config{
      static_cast<uint64_t>(envvar::rccl_min_size.get_value()),
      envvar::disable_rccl.get_value() ? 1 : 0,
      envvar::rccl_delay_init.get_value() ? 1 : 0};
  return config;
}
}  // namespace

class RcclCommContext {
 public:
  std::atomic<RcclCommState> state{RcclCommState::UNINITIALIZED};
  std::mutex init_mutex;
  ncclComm_t comm{nullptr};
  ncclUniqueId unique_id{};
  uint64_t capabilities{0};
};

namespace {
void* rccl_handle{nullptr};
std::once_flag rccl_load_once;
bool rccl_loaded{false};
std::atomic<bool> rccl_config_validated{false};

enum RcclCapability : uint64_t {
  RCCL_CAP_BROADCAST = 1ULL << 0,
  RCCL_CAP_ALL_REDUCE = 1ULL << 1,
  RCCL_CAP_REDUCE_SCATTER = 1ULL << 2,
  RCCL_CAP_ALL_GATHER = 1ULL << 3,
  RCCL_CAP_REDUCE = 1ULL << 4,
  RCCL_CAP_GATHER = 1ULL << 5,
  RCCL_CAP_SCATTER = 1ULL << 6,
  RCCL_CAP_ALL_TO_ALL = 1ULL << 7,
  RCCL_CAP_ALL_TO_ALL_V = 1ULL << 8,
};

enum class RcclSizePolicy { USE_THRESHOLD, IGNORE_THRESHOLD };

/** @brief Build the bitmap of collective operations supported by this PE. */
uint64_t local_capabilities() {
  uint64_t capabilities = 0;
  if (rccl_ftable.broadcast) capabilities |= RCCL_CAP_BROADCAST;
  if (rccl_ftable.all_reduce) capabilities |= RCCL_CAP_ALL_REDUCE;
  if (rccl_ftable.reduce_scatter) capabilities |= RCCL_CAP_REDUCE_SCATTER;
  if (rccl_ftable.all_gather) capabilities |= RCCL_CAP_ALL_GATHER;
  if (rccl_ftable.reduce) capabilities |= RCCL_CAP_REDUCE;
  if (rccl_ftable.gather) capabilities |= RCCL_CAP_GATHER;
  if (rccl_ftable.scatter) capabilities |= RCCL_CAP_SCATTER;
  if (rccl_ftable.all_to_all) capabilities |= RCCL_CAP_ALL_TO_ALL;
  if (rccl_ftable.all_to_all_v) capabilities |= RCCL_CAP_ALL_TO_ALL_V;
  return capabilities;
}

/** @brief Return the size in bytes of an RCCL data type. */
size_t rccl_type_size(ncclDataType_t type) {
  switch (type) {
    case ncclInt8:
    case ncclUint8:
      return 1;
    case ncclFloat16:
      return 2;
    case ncclInt32:
    case ncclUint32:
    case ncclFloat32:
      return 4;
    case ncclInt64:
    case ncclUint64:
    case ncclFloat64:
      return 8;
    default:
      return 0;
  }
}

/** @brief Check whether a message is large enough to use RCCL. */
bool rccl_size_enabled(size_t bytes) {
  return bytes > runtime_config().min_size;
}

/** @brief Convert an element count to bytes with overflow checking. */
bool rccl_count_bytes(size_t count, ncclDataType_t type, size_t* bytes) {
  const size_t type_size = rccl_type_size(type);
  if (type_size == 0 || bytes == nullptr ||
      count > std::numeric_limits<size_t>::max() / type_size) {
    return false;
  }
  *bytes = count * type_size;
  return true;
}

/** @brief Load a required RCCL symbol. */
template <typename T>
bool load_symbol(const char* name, T* target) {
  *target = reinterpret_cast<T>(dlsym(rccl_handle, name));
  if (*target == nullptr) {
    LOG_WARN("RCCL symbol %s is unavailable: %s", name, dlerror());
    return false;
  }
  return true;
}

/** @brief Load an optional RCCL symbol used by one capability. */
template <typename T>
void load_optional_symbol(const char* name, T* target) {
  *target = reinterpret_cast<T>(dlsym(rccl_handle, name));
  if (*target == nullptr) {
    LOG_INFO("Optional RCCL symbol %s is unavailable; the corresponding "
             "feature will use fallback", name);
  }
}

/** @brief Load the RCCL runtime and populate the process-wide function table. */
void load_rccl() {
  const char* libraries[] = {"librccl.so", "libnccl.so.2"};
  for (const char* library : libraries) {
    rccl_handle = dlopen(library, RTLD_NOW | RTLD_LOCAL);
    if (rccl_handle != nullptr) break;
  }
  if (rccl_handle == nullptr) {
    LOG_INFO("RCCL library not found; using rocSHMEM collective fallback");
    return;
  }

  bool ok = true;
#define ROCSHMEM_RCCL_LOAD(member, symbol) \
  ok = load_symbol(symbol, &rccl_ftable.member) && ok
  ROCSHMEM_RCCL_LOAD(get_version, "ncclGetVersion");
  ROCSHMEM_RCCL_LOAD(get_error_string, "ncclGetErrorString");
  ROCSHMEM_RCCL_LOAD(get_unique_id, "ncclGetUniqueId");
  ROCSHMEM_RCCL_LOAD(comm_init_rank, "ncclCommInitRank");
  ROCSHMEM_RCCL_LOAD(comm_destroy, "ncclCommDestroy");
#undef ROCSHMEM_RCCL_LOAD

  if (!ok) {
    dlclose(rccl_handle);
    rccl_handle = nullptr;
    rccl_ftable = {};
    return;
  }

  load_optional_symbol("ncclCommAbort", &rccl_ftable.comm_abort);
  load_optional_symbol("ncclCommGetAsyncError", &rccl_ftable.comm_get_async_error);
  load_optional_symbol("ncclGetLastError", &rccl_ftable.get_last_error);
  load_optional_symbol("ncclAllReduce", &rccl_ftable.all_reduce);
  load_optional_symbol("ncclReduceScatter", &rccl_ftable.reduce_scatter);
  load_optional_symbol("ncclBroadcast", &rccl_ftable.broadcast);
  load_optional_symbol("ncclAllGather", &rccl_ftable.all_gather);
  load_optional_symbol("ncclReduce", &rccl_ftable.reduce);
  load_optional_symbol("ncclGather", &rccl_ftable.gather);
  load_optional_symbol("ncclScatter", &rccl_ftable.scatter);
  load_optional_symbol("ncclAllToAll", &rccl_ftable.all_to_all);
  load_optional_symbol("ncclAllToAllv", &rccl_ftable.all_to_all_v);

  int version = 0;
  if (rccl_ftable.get_version(&version) == ncclSuccess) {
    LOG_INFO("RCCL runtime loaded (version %d); communicator initialization pending", version);
  } else {
    LOG_INFO("RCCL runtime loaded; communicator initialization pending");
  }
  rccl_loaded = true;
}

struct RcclInitSnapshot {
  RcclRuntimeConfig runtime;
  uint64_t capabilities;
  ncclUniqueId unique_id;  // Root PE generated RCCL communicator ID.
  int runtime_available;
  int unique_id_valid;  // Whether unique_id was generated successfully.
};

/** @brief Build the local snapshot used to initialize a Team communicator. */
RcclInitSnapshot make_init_snapshot(bool root) {
  const auto& config = runtime_config();
  RcclInitSnapshot snapshot{config, local_capabilities(), {}, rccl_loaded ? 1 : 0, 0};

  if (root && rccl_loaded) {
    snapshot.unique_id_valid =
        rccl_check(rccl_ftable.get_unique_id(&snapshot.unique_id), "ncclGetUniqueId")
            ? 1
            : 0;
  }
  return snapshot;
}

/** @brief Validate PE initialization snapshots and apply common capabilities. */
bool apply_init_snapshots(RcclCommContext* context,
                          const std::vector<RcclInitSnapshot>& snapshots) {
  if (snapshots.empty()) return false;

  const auto& reference = snapshots[0];
  uint64_t global_capabilities = ~uint64_t{0};
  int all_runtimes_available = 1;
  for (const auto& snapshot : snapshots) {
    if (snapshot.runtime.min_size != reference.runtime.min_size ||
        snapshot.runtime.disabled != reference.runtime.disabled ||
        snapshot.runtime.delay_init != reference.runtime.delay_init) {
      LOG_ERROR_ABORT("RCCL configuration is inconsistent across PEs");
    }
    global_capabilities &= snapshot.capabilities;
    all_runtimes_available &= snapshot.runtime_available;
  }
  rccl_config_validated.store(true, std::memory_order_release);
  if (reference.runtime.disabled || !all_runtimes_available ||
      !reference.unique_id_valid) {
    context->state.store(RcclCommState::FAILED, std::memory_order_release);
    return false;
  }
  context->unique_id = reference.unique_id;
  context->capabilities = global_capabilities;
  context->state.store(RcclCommState::PREPARED, std::memory_order_release);
  return true;
}

/** @brief Configure an RCCL context with one MPI initialization exchange. */
bool configure_mpi_context(RcclCommContext* context, MPI_Comm mpi_comm) {
  if (context == nullptr || mpi_comm == MPI_COMM_NULL) return false;
  auto current = context->state.load(std::memory_order_acquire);
  if (current != RcclCommState::UNINITIALIZED) {
    return current != RcclCommState::FAILED;
  }
  std::lock_guard<std::mutex> lock(context->init_mutex);
  if (context->state.load(std::memory_order_relaxed) != RcclCommState::UNINITIALIZED) {
    return context->state.load(std::memory_order_relaxed) != RcclCommState::FAILED;
  }

  const auto& config = runtime_config();
  if (!config.disabled) std::call_once(rccl_load_once, load_rccl);
  int rank = 0;
  if (mpilib_ftable_.Comm_rank(mpi_comm, &rank) != MPI_SUCCESS) {
    context->state.store(RcclCommState::FAILED, std::memory_order_release);
    LOG_WARN("MPI_Comm_rank failed during RCCL context configuration");
    return false;
  }
  RcclInitSnapshot local_snapshot = make_init_snapshot(rank == 0);
  int comm_size = 0;
  if (mpilib_ftable_.Comm_size(mpi_comm, &comm_size) != MPI_SUCCESS ||
      comm_size <= 0) {
    context->state.store(RcclCommState::FAILED, std::memory_order_release);
    LOG_WARN("MPI_Comm_size failed during RCCL context configuration");
    return false;
  }
  std::vector<RcclInitSnapshot> snapshots(comm_size);
  if (mpilib_ftable_.Allgather(
          &local_snapshot, sizeof(RcclInitSnapshot), MPI_CHAR,
          snapshots.data(), sizeof(RcclInitSnapshot), MPI_CHAR, mpi_comm) !=
      MPI_SUCCESS) {
    context->state.store(RcclCommState::FAILED, std::memory_order_release);
    LOG_WARN("MPI_Allgather failed during RCCL context configuration");
    return false;
  }
  return apply_init_snapshots(context, snapshots);
}

/** @brief Configure an RCCL context with one bootstrap initialization exchange. */
bool configure_bootstrap_context(RcclCommContext* context, Team* team, Bootstrap* bootstrap) {
  if (context == nullptr || team == nullptr || bootstrap == nullptr) {
    return false;
  }
  auto current = context->state.load(std::memory_order_acquire);
  if (current != RcclCommState::UNINITIALIZED) {
    return current != RcclCommState::FAILED;
  }
  std::lock_guard<std::mutex> lock(context->init_mutex);
  if (context->state.load(std::memory_order_relaxed) != RcclCommState::UNINITIALIZED) {
    return context->state.load(std::memory_order_relaxed) != RcclCommState::FAILED;
  }

  const auto& config = runtime_config();
  if (!config.disabled) std::call_once(rccl_load_once, load_rccl);
  const TeamInfo* info = team->tinfo_wrt_world;
  std::vector<RcclInitSnapshot> snapshots(team->num_pes);
  snapshots[team->my_pe] = make_init_snapshot(team->my_pe == 0);
  bootstrap->groupAllGather(snapshots.data(), sizeof(RcclInitSnapshot),
                            info->pe_start, info->stride, team->num_pes);
  return apply_init_snapshots(context, snapshots);
}

/** @brief Configure a Team context through its available bootstrap path. */
bool configure_team_context(Team* team, Bootstrap* bootstrap) {
  if (team == nullptr || team->rccl_context == nullptr) return false;
  return team->mpi_comm != MPI_COMM_NULL
             ? configure_mpi_context(team->rccl_context, team->mpi_comm)
             : configure_bootstrap_context(team->rccl_context, team, bootstrap);
}
}  // namespace

/** @brief Return whether RCCL host collectives are enabled at runtime. */
bool rccl_runtime_enabled() { return !runtime_config().disabled; }

/** @brief Return whether Team construction should initialize RCCL eagerly. */
bool rccl_eager_init_enabled() {
  const auto& config = runtime_config();
  return !config.disabled && !config.delay_init;
}

/** @brief Return whether runtime policy selects RCCL for a message. */
bool should_use_rccl(size_t bytes) {
  return rccl_runtime_enabled() && rccl_size_enabled(bytes);
}

/** @brief Report an RCCL operation result. */
bool rccl_check(ncclResult_t result, const char* operation) {
  if (result == ncclSuccess) return true;
  const char* error = rccl_ftable.get_error_string != nullptr
                          ? rccl_ftable.get_error_string(result)
                          : "unknown RCCL error";
  LOG_WARN("%s failed: %s", operation, error);
  return false;
}

namespace {
/** @brief Initialize one communicator with its previously exchanged unique ID. */
bool rccl_initialize_comm(RcclCommContext* context, int rank, int size) {
  if (context->state.load(std::memory_order_acquire) == RcclCommState::READY) {
    return true;
  }

  std::lock_guard<std::mutex> lock(context->init_mutex);
  const auto state = context->state.load(std::memory_order_relaxed);
  if (state == RcclCommState::READY) return true;
  if (state != RcclCommState::PREPARED) return false;

  if (!rccl_check(rccl_ftable.comm_init_rank(&context->comm, size,
                                             context->unique_id, rank),
                  "ncclCommInitRank")) {
    context->state.store(RcclCommState::FAILED, std::memory_order_release);
    return false;
  }
  context->state.store(RcclCommState::READY, std::memory_order_release);
  return true;
}
}  // namespace

/** @brief Allocate an RCCL communicator context. */
RcclCommContext* rccl_comm_context_create() { return new RcclCommContext(); }

/**
 * @brief Destroy a context and release its RCCL communicator.
 * @pre The owner has stopped all submissions that reference this context.
 */
void rccl_comm_context_destroy(RcclCommContext* context) {
  if (context == nullptr) return;
  {
    std::lock_guard<std::mutex> lock(context->init_mutex);
    context->state.store(RcclCommState::FAILED, std::memory_order_release);
    if (context->comm != nullptr && rccl_ftable.comm_get_async_error != nullptr) {
      ncclResult_t async_result = ncclSuccess;
      const ncclResult_t query_result = rccl_ftable.comm_get_async_error(
          context->comm, &async_result);
      if ((query_result != ncclSuccess || async_result != ncclSuccess) &&
          rccl_ftable.comm_abort != nullptr) {
        rccl_check(query_result != ncclSuccess ? query_result : async_result, "ncclCommGetAsyncError");
        rccl_ftable.comm_abort(context->comm);
        context->comm = nullptr;
      }
    }

    if (context->comm != nullptr && rccl_ftable.comm_destroy != nullptr) {
      rccl_check(rccl_ftable.comm_destroy(context->comm), "ncclCommDestroy");
    }
    context->comm = nullptr;
    context->state.store(RcclCommState::FAILED, std::memory_order_release);
  }
  delete context;
}

/** @brief Initialize an RCCL communicator over an MPI communicator. */
bool rccl_mpi_comm_init(MPI_Comm mpi_comm, int rank, int size,
                        RcclCommContext* context) {
  if (!configure_mpi_context(context, mpi_comm)) return false;
  return rccl_initialize_comm(context, rank, size);
}

/** @brief Prepare a Team context, optionally creating its communicator eagerly. */
bool rccl_team_prepare(Team* team, Bootstrap* bootstrap) {
  if (team == nullptr || team->rccl_context == nullptr) return false;
  if (!rccl_config_validated.load(std::memory_order_acquire) &&
      !configure_team_context(team, bootstrap)) {
    return runtime_config().disabled ||
           rccl_config_validated.load(std::memory_order_acquire);
  }
  const auto& config = runtime_config();
  if (config.disabled || config.delay_init) return true;
  return rccl_team_init(team, bootstrap);
}

/** @brief Lazily initialize the RCCL communicator associated with a Team. */
bool rccl_team_init(Team* team, Bootstrap* bootstrap) {
  if (team == nullptr) return false;
  if (team->rccl_context == nullptr) {
    team->rccl_context = rccl_comm_context_create();
  }
  RcclCommContext* context = team->rccl_context;
  if (context->state.load(std::memory_order_acquire) == RcclCommState::FAILED) {
    return false;
  }
  if (context->state.load(std::memory_order_acquire) == RcclCommState::UNINITIALIZED &&
      !configure_team_context(team, bootstrap)) {
    return false;
  }
  return rccl_initialize_comm(context, team->my_pe, team->num_pes);
}

/** @brief Destroy the RCCL context associated with a Team. */
void rccl_team_destroy(Team* team) {
  if (team == nullptr) return;
  rccl_comm_context_destroy(team->rccl_context);
  team->rccl_context = nullptr;
}

namespace {
/** @brief Return communicator-specific RCCL error details when available. */
const char* rccl_get_last_error(RcclCommContext* context) {
  if (context == nullptr || context->comm == nullptr ||
      rccl_ftable.get_last_error == nullptr) {
    return "unavailable";
  }
  const char* detail = rccl_ftable.get_last_error(context->comm);
  return detail != nullptr && detail[0] != '\0' ? detail : "unavailable";
}

/** @brief Abort a communicator after RCCL reports an unrecoverable error. */
void rccl_abort_comm(RcclCommContext* context) {
  if (context->comm != nullptr) {
    ncclResult_t result = rccl_ftable.comm_abort != nullptr
                              ? rccl_ftable.comm_abort(context->comm)
                              : rccl_ftable.comm_destroy(context->comm);
    rccl_check(result, rccl_ftable.comm_abort != nullptr ? "ncclCommAbort" : "ncclCommDestroy");
  }
  context->comm = nullptr;
  context->state.store(RcclCommState::FAILED, std::memory_order_release);
}

/** @brief Report a fatal RCCL-path error with communicator details. */
[[noreturn]] void rccl_error_abort(RcclCommContext* context,
                                   const char* operation, const char* stage,
                                   const char* error, int error_code) {
  const std::string last_error{rccl_get_last_error(context)};
  LOG_ERROR("%s %s failed: %s (code %d), last RCCL error: %s", operation,
            stage, error, error_code, last_error.c_str());
  rccl_abort_comm(context);
  LOG_ERROR_ABORT("RCCL fallback is unsafe after %s failure", operation);
}

/** @brief Query the deferred RCCL error when an explicit check is required. */
bool rccl_check_async_error(RcclCommContext* context, const char* operation) {
  if (rccl_ftable.comm_get_async_error == nullptr) return true;
  if (context == nullptr || context->comm == nullptr) return false;
  ncclResult_t async_result = ncclSuccess;
  ncclResult_t query_result = rccl_ftable.comm_get_async_error(context->comm, &async_result);
  if (query_result == ncclSuccess && async_result == ncclSuccess) return true;
  const ncclResult_t error = query_result != ncclSuccess ? query_result : async_result;
  rccl_error_abort(context, operation, "asynchronous operation",
                   rccl_ftable.get_error_string(error),
                   static_cast<int>(error));
}

/** @brief Finish a collective and classify submission or synchronization errors. */
RcclStatus rccl_finish_collective(ncclResult_t result,
                                  RcclCommContext* context,
                                  const char* operation, hipStream_t stream,
                                  bool blocking) {
  if (result != ncclSuccess) {
    rccl_error_abort(context, operation, "submission",
                     rccl_ftable.get_error_string(result),
                     static_cast<int>(result));
  }
  if (!blocking) return RcclStatus::SUCCESS;
  hipError_t status = hipStreamSynchronize(stream);
  if (status != hipSuccess) {
    rccl_error_abort(context, operation, "stream synchronization",
                     hipGetErrorString(status), static_cast<int>(status));
  }
  if (!rccl_check_async_error(context, operation)) {
    return RcclStatus::FATAL;
  }
  return RcclStatus::SUCCESS;
}

/** @brief Launch a byte-count collective using a Team context. */
template <typename Launch>
RcclStatus run_rccl_collective(Team* team, Bootstrap* bootstrap, size_t bytes,
                               uint64_t capability, const char* operation,
                               hipStream_t stream, bool blocking, Launch launch,
                               RcclSizePolicy size_policy = RcclSizePolicy::USE_THRESHOLD) {
  if (team == nullptr) return RcclStatus::FALLBACK_SAFE;
  if (!rccl_runtime_enabled() ||
      (size_policy == RcclSizePolicy::USE_THRESHOLD &&
       !rccl_size_enabled(bytes))) {
    return RcclStatus::FALLBACK_SAFE;
  }
  if (team->rccl_context == nullptr) {
    team->rccl_context = rccl_comm_context_create();
  }
  RcclCommContext* context = team->rccl_context;
  const auto state = context->state.load(std::memory_order_acquire);
  if (state == RcclCommState::FAILED) return RcclStatus::FALLBACK_SAFE;
  if (state == RcclCommState::UNINITIALIZED &&
      !configure_team_context(team, bootstrap)) {
    return RcclStatus::FALLBACK_SAFE;
  }
  if ((context->capabilities & capability) == 0) {
    return RcclStatus::FALLBACK_SAFE;
  }
  if (!rccl_team_init(team, bootstrap)) return RcclStatus::FALLBACK_SAFE;
  return rccl_finish_collective(launch(context->comm), context, operation,
                                stream, blocking);
}

/** @brief Launch a byte-count collective using an existing context. */
template <typename Launch>
RcclStatus run_rccl_collective(RcclCommContext* context, size_t bytes,
                               uint64_t capability, const char* operation,
                               hipStream_t stream, bool blocking,
                               Launch launch) {
  if (context == nullptr || !should_use_rccl(bytes) ||
      (context->capabilities & capability) == 0) {
    return RcclStatus::FALLBACK_SAFE;
  }
  if (context->state.load(std::memory_order_acquire) != RcclCommState::READY) {
    return RcclStatus::FALLBACK_SAFE;
  }
  return rccl_finish_collective(launch(context->comm), context, operation, stream, blocking);
}

/** @brief Convert an element count and launch a Team collective. */
template <typename Launch>
RcclStatus run_rccl_collective(Team* team, Bootstrap* bootstrap, size_t count,
                               ncclDataType_t type, uint64_t capability,
                               const char* operation, hipStream_t stream,
                               bool blocking, Launch launch) {
  size_t bytes = 0;
  if (!rccl_count_bytes(count, type, &bytes)) return RcclStatus::FALLBACK_SAFE;
  return run_rccl_collective(team, bootstrap, bytes, capability, operation,
                             stream, blocking, launch);
}

/** @brief Convert an element count and launch a context collective. */
template <typename Launch>
RcclStatus run_rccl_collective(RcclCommContext* context, size_t count,
                               ncclDataType_t type, uint64_t capability,
                               const char* operation, hipStream_t stream,
                               bool blocking, Launch launch) {
  size_t bytes = 0;
  if (!rccl_count_bytes(count, type, &bytes)) return RcclStatus::FALLBACK_SAFE;
  return run_rccl_collective(context, bytes, capability, operation, stream, blocking, launch);
}

/** @brief Validate a reduction operation and launch a Team collective. */
template <typename Launch>
RcclStatus run_rccl_collective(Team* team, Bootstrap* bootstrap, size_t count,
                               ncclDataType_t type, ncclRedOp_t op,
                               uint64_t capability, const char* operation,
                               hipStream_t stream, bool blocking,
                               Launch launch) {
  if (op == ncclNumOps) return RcclStatus::FALLBACK_SAFE;
  return run_rccl_collective(team, bootstrap, count, type, capability,
                             operation, stream, blocking, launch);
}

/** @brief Validate a reduction operation and launch a context collective. */
template <typename Launch>
RcclStatus run_rccl_collective(RcclCommContext* context, size_t count,
                               ncclDataType_t type, ncclRedOp_t op,
                               uint64_t capability, const char* operation,
                               hipStream_t stream, bool blocking,
                               Launch launch) {
  if (op == ncclNumOps) return RcclStatus::FALLBACK_SAFE;
  return run_rccl_collective(context, count, type, capability, operation,
                             stream, blocking, launch);
}

}  // namespace

/** @brief Launch RCCL broadcast for a Team. */
bool rccl_broadcast(Team* team, Bootstrap* bootstrap, const void* source,
                    void* dest, size_t bytes, int root, hipStream_t stream,
                    bool blocking) {
  return run_rccl_collective(team, bootstrap, bytes, RCCL_CAP_BROADCAST,
      "ncclBroadcast", stream, blocking,
      [=](ncclComm_t comm) {
        return rccl_ftable.broadcast(source, dest, bytes, ncclUint8, root, comm, stream);
      }) == RcclStatus::SUCCESS;
}

/** @brief Launch RCCL broadcast for an existing context. */
bool rccl_broadcast(RcclCommContext* context, const void* source, void* dest,
                    size_t bytes, int root, hipStream_t stream, bool blocking) {
  return run_rccl_collective(context, bytes, RCCL_CAP_BROADCAST,
      "ncclBroadcast", stream, blocking,
      [=](ncclComm_t comm) {
        return rccl_ftable.broadcast(source, dest, bytes, ncclUint8, root, comm, stream);
      }) == RcclStatus::SUCCESS;
}

/** @brief Launch RCCL all-to-all for a Team. */
bool rccl_alltoall(Team* team, Bootstrap* bootstrap, const void* source,
                   void* dest, size_t bytes_per_pe, hipStream_t stream) {
  return run_rccl_collective(team, bootstrap, bytes_per_pe, RCCL_CAP_ALL_TO_ALL,
      "ncclAllToAll", stream, false,
      [=](ncclComm_t comm) {
        return rccl_ftable.all_to_all(source, dest, bytes_per_pe, ncclUint8, comm, stream);
      }) == RcclStatus::SUCCESS;
}

/** @brief Launch RCCL variable-count all-to-all for a Team. */
bool rccl_alltoallv(Team* team, Bootstrap* bootstrap, const void* source,
                    const size_t* send_counts, const size_t* send_displs,
                    void* dest, const size_t* recv_counts,
                    const size_t* recv_displs, ncclDataType_t type,
                    hipStream_t stream, bool blocking) {
  if (send_counts == nullptr || send_displs == nullptr ||
      recv_counts == nullptr || recv_displs == nullptr ||
      type == ncclNumTypes || team == nullptr) {
    return false;
  }
  if (!rccl_runtime_enabled()) return false;
  return run_rccl_collective(
      team, bootstrap, 0, RCCL_CAP_ALL_TO_ALL_V, "ncclAllToAllv", stream,
      blocking,
      [=](ncclComm_t comm) {
        return rccl_ftable.all_to_all_v(
            source, send_counts, send_displs, dest, recv_counts, recv_displs,
            type, comm, stream);
      },
      RcclSizePolicy::IGNORE_THRESHOLD) == RcclStatus::SUCCESS;
}

/** @brief Launch RCCL all-gather for a Team. */
bool rccl_all_gather(Team* team, Bootstrap* bootstrap, const void* source,
                     void* dest, size_t send_count, ncclDataType_t type,
                     hipStream_t stream, bool blocking) {
  return run_rccl_collective(team, bootstrap, send_count, type, RCCL_CAP_ALL_GATHER,
      "ncclAllGather", stream, blocking,
      [=](ncclComm_t comm) {
        return rccl_ftable.all_gather(source, dest, send_count, type, comm, stream);
      }) == RcclStatus::SUCCESS;
}

/** @brief Launch RCCL reduce for a Team. */
bool rccl_reduce(Team* team, Bootstrap* bootstrap, const void* source,
                 void* dest, size_t count, ncclDataType_t type,
                 ncclRedOp_t op, int root, hipStream_t stream, bool blocking) {
  return run_rccl_collective(team, bootstrap, count, type, op, RCCL_CAP_REDUCE,
      "ncclReduce", stream, blocking,
      [=](ncclComm_t comm) {
        return rccl_ftable.reduce(source, dest, count, type, op, root, comm, stream);
      }) == RcclStatus::SUCCESS;
}

/** @brief Launch RCCL gather for a Team. */
bool rccl_gather(Team* team, Bootstrap* bootstrap, const void* source,
                 void* dest, size_t send_count, ncclDataType_t type, int root,
                 hipStream_t stream, bool blocking) {
  return run_rccl_collective(team, bootstrap, send_count, type, RCCL_CAP_GATHER,
      "ncclGather", stream, blocking,
      [=](ncclComm_t comm) {
        return rccl_ftable.gather(source, dest, send_count, type, root, comm, stream);
      }) == RcclStatus::SUCCESS;
}

/** @brief Launch RCCL scatter for a Team. */
bool rccl_scatter(Team* team, Bootstrap* bootstrap, const void* source,
                  void* dest, size_t recv_count, ncclDataType_t type, int root,
                  hipStream_t stream, bool blocking) {
  return run_rccl_collective(team, bootstrap, recv_count, type, RCCL_CAP_SCATTER,
      "ncclScatter", stream, blocking,
      [=](ncclComm_t comm) {
        return rccl_ftable.scatter(source, dest, recv_count, type, root, comm, stream);
      }) == RcclStatus::SUCCESS;
}

/** @brief Launch RCCL all-reduce for an existing context. */
bool rccl_all_reduce(RcclCommContext* context, const void* source, void* dest,
                     size_t count, ncclDataType_t type, ncclRedOp_t op,
                     hipStream_t stream, bool blocking) {
  return run_rccl_collective(context, count, type, op, RCCL_CAP_ALL_REDUCE,
      "ncclAllReduce", stream, blocking,
      [=](ncclComm_t comm) {
        return rccl_ftable.all_reduce(source, dest, count, type, op, comm, stream);
      }) == RcclStatus::SUCCESS;
}

/** @brief Launch RCCL all-reduce for a Team. */
bool rccl_all_reduce(Team* team, Bootstrap* bootstrap, const void* source,
                     void* dest, size_t count, ncclDataType_t type,
                     ncclRedOp_t op, hipStream_t stream, bool blocking) {
  return run_rccl_collective(team, bootstrap, count, type, op, RCCL_CAP_ALL_REDUCE,
      "ncclAllReduce", stream, blocking,
      [=](ncclComm_t comm) {
        return rccl_ftable.all_reduce(source, dest, count, type, op, comm, stream);
      }) == RcclStatus::SUCCESS;
}

/** @brief Launch RCCL reduce-scatter for a Team. */
bool rccl_reduce_scatter(Team* team, Bootstrap* bootstrap, const void* source,
                         void* dest, size_t count, ncclDataType_t type,
                         ncclRedOp_t op, hipStream_t stream, bool blocking) {
  return run_rccl_collective(team, bootstrap, count, type, op, RCCL_CAP_REDUCE_SCATTER,
      "ncclReduceScatter", stream, blocking,
      [=](ncclComm_t comm) {
        return rccl_ftable.reduce_scatter(source, dest, count, type, op, comm, stream);
      }) == RcclStatus::SUCCESS;
}

}  // namespace rocshmem
#endif  // USE_RCCL
