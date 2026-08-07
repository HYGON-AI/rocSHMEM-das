/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 * Copyright (c) 2026 Hygon Information Technology Co., Ltd.
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

#ifndef LIBRARY_SRC_MEMORY_SINGLE_HEAP_HPP_
#define LIBRARY_SRC_MEMORY_SINGLE_HEAP_HPP_

#include "envvar.hpp"
#include "heap_memory.hpp"
#include "heap_type.hpp"
#if defined USE_ALLOC_DLMALLOC
#include "dlmalloc.hpp"
#elif defined USE_ALLOC_POW2BINS
#include "address_record.hpp"
#include "pow2_bins.hpp"
#else
#error "You need to have one of USE_ALLOC_DLMALLOC, USE_ALLOC_POW2BINS set to ON"
#endif

/**
 * @file single_heap.hpp
 *
 * @brief Contains a single heap
 *
 * The single heap implements local processing element allocations. The
 * symmetric heap delegates allocations to this class.
 */

namespace rocshmem {

class SingleHeap {
#if defined USE_ALLOC_DLMALLOC
  /**
   * @brief Helper type for allocation strategy
   */
  using STRAT_T = DLAllocatorStrategy<HEAP_T>;
  using STRAT_T_HDP = DLAllocatorStrategy<HEAP_T_HDP>;
#elif defined USE_ALLOC_POW2BINS
  /**
   * @brief Helper type for address records
   */
  using AR_T = AddressRecord;
  /**
   * @brief Helper type for allocation strategy
   */
  using STRAT_T = Pow2Bins<AR_T, HEAP_T>;
#endif // defined USE_ALLOC_POW2BINS

 public:
  /**
   * @brief Primary constructor
   */
  SingleHeap();

  /**
   * @brief Allocates memory from the heap
   *
   * @param[in,out] A pointer to memory handle
   * @param[in] Size in bytes of memory allocation
   */
  void malloc(void** ptr, size_t size);
  void malloc_hdp(void** ptr, size_t size);

  /**
   * @brief Allocates memory from the heap
   *
   * @param[in,out] A pointer to memory handle
   * @param[in] Size in bytes of memory allocation
   *
   * @note Not implemented
   */
  __device__ void malloc(void** ptr, size_t size);
  __device__ void malloc_hdp(void** ptr, size_t size);

  /**
   * @brief Frees memory from the heap
   *
   * @param[in] Raw pointer to heap memory
   */
  void free(void* ptr);
  void free_hdp(void* ptr);

  /**
   * @brief Frees memory from the heap
   *
   * @param[in] Raw pointer to heap memory
   *
   * @note Not implemented
   */
  __device__ void free(void* ptr);
  __device__ void free_hdp(void* ptr);

  /**
   * @brief
   *
   * @param[in]
   * @param[in]
   *
   * @return
   */
  void* realloc(void* ptr, size_t size);

  /**
   * @brief
   *
   * @param[in]
   * @param[in]
   *
   * @return
   */
  void* malign(size_t alignment, size_t size);

  /**
   * @brief Accessor for heap base ptr
   *
   * @return Pointer to base of my heap
   */
  char* get_base_ptr();
  char* get_base_ptr_hdp();

  /**
   * @brief Accessor for heap size
   *
   * @return Amount of bytes in heap
   */
  size_t get_size();
  size_t get_size_hdp();

  /**
   * @brief Accessor for heap usage
   *
   * @return Amount of used bytes in heap
   */
  size_t get_used();
  size_t get_used_hdp();

  /**
   * @brief Accessor for heap available
   *
   * @return Amount of available bytes in heap
   */
  size_t get_avail();
  size_t get_avail_hdp();

  /**
   * @brief Returns is the heap is allocated with managed memory
   *
   * @return bool
   */
  bool is_managed() { return heap_mem_.is_managed(); }

  bool is_managed_hdp() { return heap_mem_hdp_.is_managed(); }

 private:
  /**
   * @brief Heap memory object
   */
  HEAP_T heap_mem_{envvar::heap_size};
  HEAP_T_HDP heap_mem_hdp_{envvar::heap_size};

  /**
   * @brief Allocation strategy object
   */
  STRAT_T strat_{&heap_mem_};
  STRAT_T_HDP strat_hdp_{&heap_mem_hdp_};
};

}  // namespace rocshmem

#endif  // LIBRARY_SRC_MEMORY_SINGLE_HEAP_HPP_
