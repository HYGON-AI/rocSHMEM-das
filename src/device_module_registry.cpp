#include "device_module_registry.hpp"

#include "backend_bc.hpp"
#include "util.hpp"

#include <algorithm>
#include <mutex>
#include <vector>

// Keep this registry process-wide and separate from device_globals.cpp.
// device_globals.cpp may be instantiated once in the host DSO and again in
// the executable through the device archive.  All of those module-local
// setters must register here so initialization can broadcast the final
// context, team, backend, constant-memory, and logging state to every module.

namespace rocshmem {
namespace {
// Function-local statics avoid cross-translation-unit initialization-order
// dependencies when device modules register from their static constructors.
std::mutex& registry_mutex() {
  static std::mutex value;
  return value;
}

// Serialize snapshot publication, replay, and invalidation.  It is separate
// from registry_mutex so registry container access never holds that mutex
// across HIP runtime calls.  Recursive locking also permits HIP/loader work to
// register another module on the same thread while a snapshot is being applied.
std::recursive_mutex& lifecycle_mutex() {
  static std::recursive_mutex value;
  return value;
}

// Stores one module-specific setter table for every registered HIP module.
std::vector<const DeviceModuleSetters *>& device_modules() {
  static std::vector<const DeviceModuleSetters *> value;
  return value;
}

// Last fully initialized process-wide device state.  A module can be loaded
// after rocshmem_init(), so registration must be able to replay this state
// without asking the application to initialize rocSHMEM a second time.
struct DeviceStateSnapshot {
  rocshmem_ctx_t ctx{};
  rocshmem_team_t team_world{};
  rocshmem_team_t team_shared{};
  constmem_t constmem_values{};
  Backend *backend{};
  struct logd_constants log_values{};
  bool valid{false};
};

DeviceStateSnapshot& device_state_snapshot() {
  static DeviceStateSnapshot value;
  return value;
}

void apply_snapshot(const DeviceModuleSetters *module,
                    const DeviceStateSnapshot& snapshot) {
  module->set_ctx(&snapshot.ctx);
  module->set_team_world(snapshot.team_world);
  module->set_team_shared(snapshot.team_shared);
  module->set_constmem(&snapshot.constmem_values);
  module->set_backend(snapshot.backend);
  module->set_log(&snapshot.log_values);
}
}  // namespace

// Called by each device module's static constructor to make that module
// reachable from the single process-wide synchronization point.
extern "C" void rocshmem_register_device_module(
    const DeviceModuleSetters *setters) {
  {
    std::lock_guard<std::mutex> lock(registry_mutex());
    auto& modules = device_modules();
    if (std::find(modules.begin(), modules.end(), setters) != modules.end())
      return;
    modules.push_back(setters);
  }

  std::lock_guard<std::recursive_mutex> lifecycle_lock(lifecycle_mutex());
  DeviceStateSnapshot snapshot;
  {
    std::lock_guard<std::mutex> lock(registry_mutex());
    snapshot = device_state_snapshot();
  }
  if (snapshot.valid) apply_snapshot(setters, snapshot);
}

// Read the finalized runtime state from the calling library's source HIP
// module, then broadcast the same state through every registered module's
// setters.  This runs after backend, context, team, and constant-memory setup.
void sync_device_modules() {
  std::lock_guard<std::recursive_mutex> lifecycle_lock(lifecycle_mutex());
  DeviceStateSnapshot snapshot;

  // Capture one consistent snapshot from the source module.  -Bsymbolic keeps
  // these HIP tokens bound to the host DSO's own symbols in split-link mode.
  CHECK_HIP(hipMemcpyFromSymbol(&snapshot.ctx,
                                HIP_SYMBOL(ROCSHMEM_CTX_DEFAULT),
                                sizeof(snapshot.ctx)));
  CHECK_HIP(hipMemcpyFromSymbol(&snapshot.team_world,
                                HIP_SYMBOL(device::ROCSHMEM_TEAM_WORLD),
                                sizeof(snapshot.team_world)));
  CHECK_HIP(hipMemcpyFromSymbol(&snapshot.team_shared,
                                HIP_SYMBOL(device::ROCSHMEM_TEAM_SHARED),
                                sizeof(snapshot.team_shared)));
  CHECK_HIP(hipMemcpyFromSymbol(&snapshot.constmem_values,
                                HIP_SYMBOL(constmem),
                                sizeof(snapshot.constmem_values)));
  CHECK_HIP(hipMemcpyFromSymbol(&snapshot.backend,
                                HIP_SYMBOL(device_backend_proxy),
                                sizeof(snapshot.backend)));
  CHECK_HIP(hipMemcpyFromSymbol(&snapshot.log_values,
                                HIP_SYMBOL(logd_constants),
                                sizeof(snapshot.log_values)));
  snapshot.valid = true;

  std::vector<const DeviceModuleSetters *> modules;
  {
    std::lock_guard<std::mutex> lock(registry_mutex());
    device_state_snapshot() = snapshot;
    modules = device_modules();
  }

  // Apply the snapshot to both the host DSO's embedded device module and any
  // device module linked into the executable.  A full-static build normally
  // traverses only one entry, making this an idempotent rewrite.
  for (const auto *module : modules) apply_snapshot(module, snapshot);
}

void clear_device_module_state() {
  std::lock_guard<std::recursive_mutex> lifecycle_lock(lifecycle_mutex());
  std::lock_guard<std::mutex> lock(registry_mutex());
  device_state_snapshot() = DeviceStateSnapshot{};
}
}  // namespace rocshmem
