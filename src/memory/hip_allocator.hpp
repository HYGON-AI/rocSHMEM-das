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

#ifndef LIBRARY_SRC_MEMORY_HIP_ALLOCATOR_HPP_
#define LIBRARY_SRC_MEMORY_HIP_ALLOCATOR_HPP_

/**
 * @file hip_allocator.hpp
 *
 * @brief Contains HIP wrapper class for memory allocator
 */

#include "rocshmem/rocshmem_config.h"  // NOLINT(build/include_subdir)
#include "memory_allocator.hpp"

#include <hip/hip_runtime_api.h>
#include <hip/hip_version.h>
#include <hsa/hsa_ext_amd.h>

#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <limits>
#include <map>
#include <vector>
#include <unistd.h>
#include <sys/syscall.h>

// 安全检查：确保宏已定义
#ifndef HIP_VERSION_PATCH  
    #error "HIP_VERSION_PATCH not defined! Check your HIP installation."
#endif

namespace rocshmem {

enum HIPIpcHandleType {
  HandleTypeLegacy = 0,
  HandleTypePosix,
  HandleTypeFabric,
  HandleTypeLast
};

enum HIPAllocatorType {
  AllocatorTypeCoarsegrained = 0,
  AllocatorTypeFinegrained,
  AllocatorTypeUncached,
  AllocatorTypeVMM,
  AllocatorTypeLast
};

#if HIP_VERSION >= 70000000
struct hipIpcMemHandlePosix_t {
  uint64_t fd;
  uint32_t pid;
  size_t size;
};
#endif

class HIPIpcHandleVec {
public:
  virtual ~HIPIpcHandleVec() = default;

  virtual HIPIpcHandleType GetIpcHandleType() = 0;
  virtual void* GetHandleVecElem(int elem) = 0;
};

class HIPIpcHandleLegacyVec : public HIPIpcHandleVec {
public:
  friend class HIPAllocator;

  HIPIpcHandleType GetIpcHandleType() { return HandleTypeLegacy; }

  void* GetHandleVecElem(int elem)
  {
    return reinterpret_cast<void*> (&this->handle[elem]);
  }

protected:
  std::vector<hipIpcMemHandle_t> handle;

};

#if HIP_VERSION >= 70000000
class HIPIpcHandlePosixVec : public HIPIpcHandleVec {
public:
  friend class HIPAllocatorVMMPosixFd;

  HIPIpcHandleType GetIpcHandleType() { return HandleTypePosix; }

  void* GetHandleVecElem(int elem)
  {
    return reinterpret_cast<void*> (&this->handle[elem]);
  }

protected:
  std::vector<hipIpcMemHandlePosix_t> handle;

};
#endif

class HIPAllocator : public MemoryAllocator {
 public:

  HIPAllocator() : MemoryAllocator(hipMalloc, hipFree) {}

  HIPAllocator(hipError_t (*hip_alloc_fn)(void**, size_t),
               hipError_t (*hip_free_fn)(void*)) :
      MemoryAllocator (hip_alloc_fn, hip_free_fn) {}

  HIPAllocator (hipError_t (*hip_alloc_fn)(void**, size_t, unsigned),
                hipError_t (*hip_free_fn)(void*), unsigned flags) :
    MemoryAllocator (hip_alloc_fn, hip_free_fn, flags) {}

  virtual ~HIPAllocator() = default;

  HIPAllocatorType type = AllocatorTypeCoarsegrained;

  virtual hipError_t GetIpcHandle(void *dev_ptr, void *handle)
  {
    return hipIpcGetMemHandle(reinterpret_cast<hipIpcMemHandle_t *>(handle), dev_ptr);
  }

  virtual hipError_t OpenIpcHandle(void **dev_ptr, void *handle)
  {
    return hipIpcOpenMemHandle(dev_ptr, *(reinterpret_cast<hipIpcMemHandle_t *>(handle)),
                               hipIpcMemLazyEnablePeerAccess);
  }

  virtual hipError_t CloseIpcHandle(void *dev_ptr)
  {
    return hipIpcCloseMemHandle(dev_ptr);
  }

  virtual size_t GetIpcHandleSize()
  {
    return sizeof(hipIpcMemHandle_t);
  }

  virtual HIPIpcHandleVec* AllocateIpcHandleVec(int num_elems)
  {
    HIPIpcHandleLegacyVec* vec = new HIPIpcHandleLegacyVec();
    vec->handle.resize(num_elems);
    return vec;
  }

  virtual hipError_t GetDmabufHandle(void *dev_ptr, size_t size, int *dmabuf_fd, uint64_t *dmabuf_offset)
  {
    if (dev_ptr == nullptr || dmabuf_fd == nullptr || dmabuf_offset == nullptr) {
      return hipErrorInvalidValue;
    }

    // Use HSA API to export dmabuf from device pointer
    uint64_t offset = 0;
    int fd = -1;
    hsa_status_t status = hsa_amd_portable_export_dmabuf(dev_ptr, size, &fd, &offset);

    if (status != HSA_STATUS_SUCCESS) {
      *dmabuf_fd = -1;
      *dmabuf_offset = 0;
      return hipErrorInvalidValue;
    }

    *dmabuf_fd = fd;
    *dmabuf_offset = offset;
    return hipSuccess;
  }
};

using HIPAllocatorCoarsegrained = HIPAllocator;

class HIPAllocatorFinegrained : public HIPAllocator {
public:
  HIPAllocatorFinegrained()
      : HIPAllocator(
            malloc_with_flags,  // 静态函数指针，可以被转换
            hipFree,
            get_malloc_flags()) {
      type = AllocatorTypeFinegrained;
  }

private:
  static bool is_xdp_enabled() {
    // 检查环境变量 ROCSHMEM_GDR_DISABLE_XDP
    char* env_disable_xdp = getenv("ROCSHMEM_GDR_DISABLE_XDP");
    if (env_disable_xdp != NULL) {
      // 如果环境变量设置为"1"、"true"、"on"、"yes"等，则禁用XDP
      char disable_value[16];
      strncpy(disable_value, env_disable_xdp, sizeof(disable_value) - 1);
      disable_value[sizeof(disable_value) - 1] = '\0';

      // 转换为小写比较
      for (char* p = disable_value; *p; ++p) {
        *p = tolower(*p);
      }

      if (strcmp(disable_value, "1") == 0 ||
          strcmp(disable_value, "true") == 0 ||
          strcmp(disable_value, "on") == 0 ||
          strcmp(disable_value, "yes") == 0) {
        return false;  // 环境变量强制禁用XDP
      }
    }

    // 读取系统参数
    FILE* fp = fopen("/sys/module/hycu/parameters/xdp_size", "r");
    if (!fp) {
      return false;  // 文件不存在，默认为普通细粒度内存
    }

    unsigned long xdp_size;
    int result = fscanf(fp, "%lu", &xdp_size);
    fclose(fp);

    if (result != 1) {
      return false;  // 读取失败
    }

    // 根据 xdp_size 判断是否启用 XDP
    // 假设规则：如果 xdp_size > 0，则使用 XDP 内存
    return xdp_size > 0;
  }

  static unsigned int get_malloc_flags() {
  #if defined(HIP_VERSION_PATCH) && (HIP_VERSION_PATCH >= 25521)
    if (is_xdp_enabled()) {
      return hipDeviceMallocUncachedXdp;
    } else {
      return hipDeviceMallocFinegrained;
    }
  #else
    return hipDeviceMallocFinegrained;
  #endif 
  }

  // 静态函数替代 lambda
  static hipError_t malloc_with_flags(void** ptr, size_t size, unsigned int /*flags*/) {
    // 这里的 flags 参数被忽略，因为我们使用静态标志
    return hipExtMallocWithFlags(ptr, size, get_malloc_flags());
  }
};  

#if defined HAVE_DEVICE_MALLOC_UNCACHED
class HIPAllocatorUncached : public HIPAllocator {
 public:
  HIPAllocatorUncached()
      : HIPAllocator(hipExtMallocWithFlags, hipFree,
                     hipDeviceMallocUncached) {
    type = AllocatorTypeUncached;
  }
};
#endif

#if HIP_VERSION >= 70000000
class HIPAllocatorVMMPosixFd : public HIPAllocator {
 private:
  struct VMMAllocationInfo {
    hipMemGenericAllocationHandle_t handle;
    size_t size;
    int exported_fd;  // File descriptor exported for IPC, -1 if not exported
  };
  static std::map<void*, VMMAllocationInfo> allocations_;
  static std::map<void*, VMMAllocationInfo> imported_allocations_;

  static hipError_t VMMAlloc(void** ptr, size_t size);
  static hipError_t VMMFree(void* ptr);

 public:
  HIPAllocatorVMMPosixFd();

  hipError_t GetIpcHandle(void *dev_ptr, void *handle) override;
  hipError_t OpenIpcHandle(void **dev_ptr, void *handle) override;
  hipError_t CloseIpcHandle(void *dev_ptr) override;
  size_t GetIpcHandleSize() override;
  HIPIpcHandleVec* AllocateIpcHandleVec(int num_elems) override;
  hipError_t GetDmabufHandle(void *dev_ptr, size_t size, int *dmabuf_fd, uint64_t *dmabuf_offset) override;
};
#endif

class HIPHostAllocator : public MemoryAllocator {
 public:
  HIPHostAllocator()
      : MemoryAllocator(hipHostMalloc, hipFree, hipHostMallocCoherent) {}
};

class PosixAligned64Allocator : public MemoryAllocator {
 public:
  PosixAligned64Allocator() : MemoryAllocator(posix_memalign, std::free, 64) {}
};

using HostAllocator = PosixAligned64Allocator;
}  // namespace rocshmem

#endif  // LIBRARY_SRC_MEMORY_HIP_ALLOCATOR_HPP_
