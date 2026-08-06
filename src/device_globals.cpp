// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#include "rocshmem/rocshmem.hpp"
#include "backend_bc.hpp"
#include "constmem.hpp"
#include "log.hpp"
#include "device_module_registry.hpp"
#include "util.hpp"

// Static consumers reference this hidden symbol with -Wl,-u.  This forces the
// one host-bearing archive member into the final link so its module-local
// device-global registration constructor runs.  All other members of the
// split device archive contain device bundles only.
extern "C" __attribute__((visibility("hidden"), used)) void
rocshmem_force_link_device_module() {
  return;
}

// This file is intentionally module-local: every HIP device module needs its
// own device globals and setters whose HIP symbol tokens refer to that module.
// The setters are registered with the process-wide registry implemented in
// device_module_registry.cpp.  Do not move that registry implementation here,
// otherwise split host/device linking could create one isolated registry per
// ELF module and leave their device state unsynchronized.

// Device-side runtime state owned by this HIP module.  Kernels resolve these
// symbols within the module that contains them, so each copy must receive the
// same host-initialized values before device APIs are used.
namespace rocshmem {
__device__ rocshmem_ctx_t
__attribute__((visibility("default"))) ROCSHMEM_CTX_DEFAULT{};
__constant__ rocshmem_ctx_t* rocshmem_ctx_array;
__constant__ Backend* device_backend_proxy;
__constant__ constmem_t constmem;
__constant__ rocshmem_ctx_t ROCSHMEM_CTX_INVALID = {nullptr, nullptr};
__constant__ struct logd_constants logd_constants;
namespace device {
extern "C" __constant__ rocshmem_team_t
__attribute__((visibility("default"))) ROCSHMEM_TEAM_WORLD = nullptr;
extern "C" __constant__ rocshmem_team_t
__attribute__((visibility("default"), used)) ROCSHMEM_TEAM_SHARED = nullptr;
}

namespace {
// These setters deliberately use the HIP symbol tokens compiled in this file.
// Consequently, each callback updates this module's copy rather than whichever
// same-named symbol happens to be visible from another ELF module.
void set_ctx(const rocshmem_ctx_t* value) {
  CHECK_HIP(hipMemcpyToSymbol(HIP_SYMBOL(ROCSHMEM_CTX_DEFAULT), value,
                              sizeof(*value)));
}
void set_team_world(rocshmem_team_t value) {
  CHECK_HIP(hipMemcpyToSymbol(HIP_SYMBOL(device::ROCSHMEM_TEAM_WORLD), &value,
                              sizeof(value)));
}
void set_team_shared(rocshmem_team_t value) {
  CHECK_HIP(hipMemcpyToSymbol(HIP_SYMBOL(device::ROCSHMEM_TEAM_SHARED), &value,
                              sizeof(value)));
}
void set_constmem(const constmem_t* value) {
  CHECK_HIP(hipMemcpyToSymbol(HIP_SYMBOL(constmem), value, sizeof(*value)));
}
void set_backend(Backend* value) {
  CHECK_HIP(hipMemcpyToSymbol(HIP_SYMBOL(device_backend_proxy), &value,
                              sizeof(value)));
}
void set_log(const struct logd_constants* value) {
  CHECK_HIP(hipMemcpyToSymbol(HIP_SYMBOL(logd_constants), value, sizeof(*value)));
}

// Callback table exported to the process-wide registry.  Keeping the table
// module-local preserves the association between a callback and its HIP symbol
// tokens.
const DeviceModuleSetters setters{set_ctx, set_team_world, set_team_shared,
                                  set_constmem, set_backend, set_log};

// Register this module during ELF/static initialization, before rocSHMEM's
// runtime initialization calls sync_device_modules().
struct RegisterDeviceModule {
  RegisterDeviceModule() { rocshmem_register_device_module(&setters); }
} register_device_module;
}  // namespace

}
