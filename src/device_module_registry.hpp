#ifndef ROCSHMEM_DEVICE_MODULE_REGISTRY_HPP
#define ROCSHMEM_DEVICE_MODULE_REGISTRY_HPP

#include "rocshmem/rocshmem.hpp"
#include "constmem.hpp"
#include "log.hpp"

namespace rocshmem {

class Backend;

struct DeviceModuleSetters {
  void (*set_ctx)(const rocshmem_ctx_t *);
  void (*set_team_world)(rocshmem_team_t);
  void (*set_team_shared)(rocshmem_team_t);
  void (*set_constmem)(const constmem_t *);
  void (*set_backend)(Backend *);
  void (*set_log)(const struct logd_constants *);
};

extern "C" void rocshmem_register_device_module(
    const DeviceModuleSetters *setters);

void sync_device_modules();

}  // namespace rocshmem

#endif
