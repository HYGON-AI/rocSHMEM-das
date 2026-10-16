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

#ifndef _VERIFY_RESULTS_KERNELS_HPP_
#define _VERIFY_RESULTS_KERNELS_HPP_

#include <hip/hip_runtime.h>
#include <stdexcept>

namespace rocshmem {

[[maybe_unused]] static __global__ void verify_results_kernel_char(
    char *dest, size_t size, size_t stride,
    size_t concurrency, int loop, int skip, int batch,
    bool *verification_error) {

  int start_slot = (batch - (skip % batch)) % batch;
  int verify_iters = min(batch, loop + skip);

  size_t idx = get_flat_id();
  size_t check_per_buf = size * verify_iters;
  size_t total = check_per_buf * concurrency;

  if (idx >= total) {
    return;
  }

  size_t b = idx / check_per_buf;
  size_t local = idx % check_per_buf;
  size_t iter = local / size;
  size_t i = local % size;
  size_t slot = (start_slot + iter) % batch;

  if (dest[b * stride + slot * size + i] != 'a') {
    *verification_error = true;
  }
}

namespace collective_verifier {

template <typename T>
struct VerificationResult {
  unsigned int failed{};
  size_t index{};
  T expected{};
  T actual{};
};

using Result = VerificationResult<unsigned int>;
using IntResult = VerificationResult<int>;

[[maybe_unused]] static __global__ void verifyIntKernel(
    const int *buffer, size_t count, int expected, IntResult *result) {
  const size_t step = static_cast<size_t>(gridDim.x) * blockDim.x;
  for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < count;) {
    const int actual = buffer[i];
    if (actual != expected) {
      if (atomicCAS(&result->failed, 0U, 1U) == 0U) {
        result->index = i;
        result->expected = expected;
        result->actual = actual;
      }
      return;
    }
    if (count - i <= step) break;
    i += step;
  }
}

// Each grid row checks one sender's segment, including nonuniform layouts.
[[maybe_unused]] static __global__ void verifyCharKernel(const char *buffer, size_t uniform_bytes,
                                    const size_t *counts, const size_t *displs,
                                    size_t sender_base, size_t pattern_stride, size_t pattern_bias,
                                    Result *result) {
  const size_t sender = sender_base + blockIdx.y;
  const size_t count = counts ? counts[sender] : uniform_bytes;
  const size_t offset = displs ? displs[sender] : sender * uniform_bytes;
  const unsigned char expected = static_cast<unsigned char>(sender * pattern_stride + pattern_bias);
  const size_t step = static_cast<size_t>(gridDim.x) * blockDim.x;
  for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < count;) {
    const unsigned char actual = static_cast<unsigned char>(buffer[offset + i]);
    if (actual != expected) {
      if (atomicCAS(&result->failed, 0U, 1U) == 0U) {
        result->index = offset + i;
        result->expected = expected;
        result->actual = actual;
      }
      return;
    }
    if (count - i <= step) break;
    i += step;
  }
}

// The caller synchronizes the collective before verification. Only the small
// error record is copied back; all payload reads stay on the GPU.
[[maybe_unused]] static Result verifyChar(const char *buffer, size_t uniform_bytes, int n_pes,
                     const size_t *counts, const size_t *displs,
                     size_t pattern_stride, size_t pattern_bias,
                     hipStream_t stream) {
  const auto check = [](hipError_t status) {
    if (status != hipSuccess) throw std::runtime_error(hipGetErrorString(status));
  };
  Result *device_result = nullptr;
  check(hipMalloc(&device_result, sizeof(Result)));
  check(hipMemsetAsync(device_result, 0, sizeof(Result), stream));
  // Batch teams larger than grid.y's limit.
  for (size_t sender = 0; sender < static_cast<size_t>(n_pes); sender += 65535) {
    const unsigned int rows = static_cast<unsigned int>(
        static_cast<size_t>(n_pes) - sender < 65535 ?
        static_cast<size_t>(n_pes) - sender : 65535);
    verifyCharKernel<<<dim3(256, rows), 256, 0, stream>>>(
        buffer, uniform_bytes, counts, displs, sender,
        pattern_stride, pattern_bias,
        device_result);
    check(hipGetLastError());
  }
  check(hipStreamSynchronize(stream));
  Result result{};
  check(hipMemcpy(&result, device_result, sizeof(Result), hipMemcpyDeviceToHost));
  check(hipFree(device_result));
  return result;
}

[[maybe_unused]] static IntResult verifyInt(const int *buffer, size_t count,
                                         int expected, hipStream_t stream) {
  const auto check = [](hipError_t status) {
    if (status != hipSuccess) throw std::runtime_error(hipGetErrorString(status));
  };
  IntResult *device_result = nullptr;
  check(hipMalloc(&device_result, sizeof(IntResult)));
  check(hipMemsetAsync(device_result, 0, sizeof(IntResult), stream));
  verifyIntKernel<<<256, 256, 0, stream>>>(buffer, count, expected, device_result);
  check(hipGetLastError());
  check(hipStreamSynchronize(stream));
  IntResult result{};
  check(hipMemcpy(&result, device_result, sizeof(IntResult), hipMemcpyDeviceToHost));
  check(hipFree(device_result));
  return result;
}

}  // namespace collective_verifier

}  // namespace rocshmem

#endif /* _VERIFY_RESULTS_KERNELS_HPP_ */
