#include "device_module_registry.hpp"

#include "backend_bc.hpp"
#include "util.hpp"

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

// Stores one module-specific setter table for every registered HIP module.
std::vector<const DeviceModuleSetters *>& device_modules() {
  static std::vector<const DeviceModuleSetters *> value;
  return value;
}

// Serialize registration and synchronization so the callback list remains
// stable for the complete traversal.
template <typename F>
void for_each_module(F &&fn) {
  std::lock_guard<std::mutex> lock(registry_mutex());
  for (const auto *module : device_modules()) fn(module);
}
}  // namespace

// Called by each device module's static constructor to make that module
// reachable from the single process-wide synchronization point.
extern "C" void rocshmem_register_device_module(
    const DeviceModuleSetters *setters) {
  std::lock_guard<std::mutex> lock(registry_mutex());
  device_modules().push_back(setters);
}

// Read the finalized runtime state from the calling library's source HIP
// module, then broadcast the same state through every registered module's
// setters.  This runs after backend, context, team, and constant-memory setup.
void sync_device_modules() {
  rocshmem_ctx_t ctx{};
  rocshmem_team_t team_world{};
  rocshmem_team_t team_shared{};
  constmem_t constmem_values{};
  Backend *backend{};
  struct logd_constants log_values{};

  // Capture one consistent snapshot from the source module.  -Bsymbolic keeps
  // these HIP tokens bound to the host DSO's own symbols in split-link mode.
  CHECK_HIP(hipMemcpyFromSymbol(&ctx, HIP_SYMBOL(ROCSHMEM_CTX_DEFAULT),
                                sizeof(ctx)));
  CHECK_HIP(hipMemcpyFromSymbol(&team_world,
                                HIP_SYMBOL(device::ROCSHMEM_TEAM_WORLD),
                                sizeof(team_world)));
  CHECK_HIP(hipMemcpyFromSymbol(&team_shared,
                                HIP_SYMBOL(device::ROCSHMEM_TEAM_SHARED),
                                sizeof(team_shared)));
  CHECK_HIP(hipMemcpyFromSymbol(&constmem_values, HIP_SYMBOL(constmem),
                                sizeof(constmem_values)));
  CHECK_HIP(hipMemcpyFromSymbol(&backend, HIP_SYMBOL(device_backend_proxy),
                                sizeof(backend)));
  CHECK_HIP(hipMemcpyFromSymbol(&log_values, HIP_SYMBOL(logd_constants),
                                sizeof(log_values)));

  // Apply the snapshot to both the host DSO's embedded device module and any
  // device module linked into the executable.  A full-static build normally
  // traverses only one entry, making this an idempotent rewrite.
  for_each_module([&](const auto *module) {
    module->set_ctx(&ctx);
    module->set_team_world(team_world);
    module->set_team_shared(team_shared);
    module->set_constmem(&constmem_values);
    module->set_backend(backend);
    module->set_log(&log_values);
  });
}
}  // namespace rocshmem
