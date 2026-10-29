// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#ifndef LIBRARY_SRC_DEVICE_ARCH_HPP_
#define LIBRARY_SRC_DEVICE_ARCH_HPP_

// Identify supported GPU targets for architecture-specific code.
#if defined(__gfx936__) || defined(__gfx938__)
#define ROCSHMEM_ARCH_HYGON 1
#else
#define ROCSHMEM_ARCH_HYGON 0
#endif

#endif  // LIBRARY_SRC_DEVICE_ARCH_HPP_
