// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#include "amo_dp_standard_tester.hpp"
#include "tester.hpp"

#include <iostream>
#include <rocshmem/rocshmem.hpp>

using namespace rocshmem;

/* Declare the global kernel template with a generic implementation */
template <typename T>
__global__ void AMODpStandardTest(int loop, int skip, long long int *start_time,
                                long long int *end_time, T *dest,
                                T *ret_val, AddrMode addr_mode,
                                TestType type, ShmemContextType ctx_type,
                                int wf_size, int num_pes, long long* done_flags) {
  return;
}

template <class T>
__device__ inline T* compute_target_ptr(T* base_ptr, AddrMode addr_mode,
                                        int wg_idx, int itr, int n_wgs) {
  // PerBlock: element = wg_idx, with n_wgs elements per loop
  // PerGrid : single element shared by the whole grid per loop
  if (addr_mode == AddrMode::PerBlock) {
    size_t offset = wg_idx + itr * n_wgs;
    return base_ptr + offset;
  } else { // PerGrid
    return base_ptr + itr;
  }
}

/******************************************************************************
 * HOST TESTER CLASS METHODS
 *****************************************************************************/
template <typename T>
AMODpStandardTester<T>::AMODpStandardTester(TesterArguments args) : Tester(args) {
  n_out   = (args.addr_mode == AddrMode::PerBlock) ? args.num_wgs : 1;
  n_in    = args.num_wgs * args.wg_size;
  n_loops = args.loop + args.skip;

  // One return per *thread* per loop
  CHECK_HIP(hipMalloc((void **)&ret_val, sizeof(T) * n_in * n_loops));

  dest = (T *)rocshmem_malloc(sizeof(T) * n_out * n_loops);
  done_flags = (long long*)rocshmem_malloc(sizeof(long long) * args.num_wgs * num_warps);
  if (dest == nullptr || done_flags == nullptr) {
    std::cerr << "Error allocating memory from symmetric heap" << std::endl;
    std::cerr << "dest: " << dest << ", done_flags: " << done_flags << std::endl;

    if (dest) {
      rocshmem_free(dest);
    }

    if (done_flags) {
      rocshmem_free(done_flags);
    }
  }

  check_id = (_type == DefaultCtx_AMO_Add_DPTestType || _type == AMO_Add_DPTestType)
              ? 1
              : 0;
}

template <typename T>
AMODpStandardTester<T>::~AMODpStandardTester() {
  CHECK_HIP(hipFree(ret_val));
  rocshmem_free(dest);
  rocshmem_free(done_flags);
}

template <typename T>
void AMODpStandardTester<T>::resetBuffers(size_t size) {
  memset(ret_val, 0, sizeof(T) * n_in  * n_loops);
  memset(dest,    0, sizeof(T) * n_out * n_loops);
  memset(done_flags, 0, sizeof(long long) * args.num_wgs * num_warps);
}

static __device__ void verify_and_set_flags(long long *done_flags, int idx, bool is_skip) {
  long long done_flags_t = 0;
  long long *done_flags_ptr = done_flags + idx;
  long long expected_flag = is_skip ? 1 : 2;                       
  do {
    done_flags_t = __hip_atomic_load(done_flags_ptr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  } while (done_flags_t != expected_flag);

  // set done_flags of PE 0 to 1 to signal that the kernel is done
  rocshmem_longlong_atomic_add_dp(done_flags_ptr, 1, 0, 0);        
}

static __global__ void wave_verify_and_set_flags(long long *done_flags, int wf_size) {
  if (is_thread_zero_in_wave()) {
    int wg_id = get_flat_grid_id();
    int wf_id = get_flat_block_id() / wf_size;
    int wg_offset = wg_id * ((get_flat_block_size() - 1 ) / wf_size + 1);   
    int idx = wf_id + wg_offset; 
   
    // skip
    verify_and_set_flags(done_flags, idx, true);

    // loop
    verify_and_set_flags(done_flags, idx, false);
  } 
}

template <typename T>
void AMODpStandardTester<T>::launchKernel(dim3 gridsize, dim3 blocksize, int loop,
                                        size_t size) {
  size_t shared_bytes = 0;
  if (args.myid == 0) {
    hipLaunchKernelGGL(AMODpStandardTest, gridsize, blocksize, shared_bytes, stream,
                      args.loop, args.skip, start_time, end_time, dest, ret_val,
                      args.addr_mode, _type, _shmem_context, wf_size, args.numprocs, done_flags);
  } else if (args.myid == check_id) {
    hipLaunchKernelGGL(wave_verify_and_set_flags, gridsize, blocksize, shared_bytes, stream, done_flags, wf_size);
  }

  if (DefaultCtx_AMO_Add_DPTestType == _type ||
      AMO_Add_DPTestType == _type) {
    num_msgs       = n_loops   * gridsize.x  * num_warps;
    num_timed_msgs = args.loop * gridsize.x  * num_warps;
  } else {
    num_msgs       = n_loops   * gridsize.x * blocksize.x;
    num_timed_msgs = args.loop * gridsize.x * blocksize.x;
  }
}


template <typename G>
void fail_eq(const G& got, const G& exp, int i) {
  std::cerr << "Data validation error at idx " << i << "\n"
            << "got " << got << ", expected " << exp << std::endl;
  std::exit(-1);
}

template <typename G>
void fail_nonzero(const G& got) {
  std::cerr << "Data validation error\n"
            << "got " << got << ", expected non-zero" << std::endl;
  std::exit(-1);
}

// Map (loop, elem_idx) -> dest[] index for current address mode.
template <typename T>
int AMODpStandardTester<T>::destIndex(int l, int elem_idx) const {
  return (args.addr_mode == AddrMode::PerBlock)
          ? l * args.num_wgs + elem_idx
          : l; // PerGrid has a single element per loop
}

// Number of output elements to check per loop for current address mode.
template <typename T>
int AMODpStandardTester<T>::numElems() const {
  return (args.addr_mode == AddrMode::PerBlock)
          ? static_cast<int>(args.num_wgs)
          : 1; // PerGrid
}

template <typename T>
void AMODpStandardTester<T>::verifyDestValues() {
  const int loops   = static_cast<int>(n_loops);
  const int n_elems = numElems();

  auto check_equal_all = [&](T expected) {
    for (int l = 0; l < loops; ++l) {
      for (int elem = 0; elem < n_elems; ++elem) {
        const int idx = destIndex(l, elem);
        if (dest[idx] != expected) fail_eq(dest[idx], expected, idx);
      }
    }
  };

  switch (_type) {
    case DefaultCtx_AMO_Add_DPTestType:
    case AMO_Add_DPTestType: {
      const T expected = (args.addr_mode == AddrMode::PerBlock)
        ? static_cast<T>(2*num_warps)
        : static_cast<T>(num_warps * args.num_wgs * 2);
      check_equal_all(expected);
      break;
    }
    default:
      break;
  }
}

template <typename T>
void AMODpStandardTester<T>::verifyResults(size_t size) {
  if (args.myid == check_id) {
    verifyDestValues();
  } 
}

__device__ void wave_amo_quite_dp(long long *done_flags, int idx, bool is_default_ctx, rocshmem_ctx_t ctx, int qp_idx, int dst_pe, bool is_skip) {
  long long *done_flags_ptr = done_flags + idx;
  int expected_flag = is_skip ? 1 : 2;

  if (is_default_ctx) {
    rocshmem_longlong_atomic_add_dp(done_flags_ptr, 1, qp_idx, dst_pe);
  } else {   
    rocshmem_ctx_longlong_atomic_add_dp(ctx, done_flags_ptr, 1, qp_idx, dst_pe);
  }

  long long done_flags_t = 0;                                           
  do {
    done_flags_t = __hip_atomic_load(done_flags_ptr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  } while (done_flags_t != expected_flag);
}

#define AMO_STANDARD_DEF_GEN(T, TNAME)                                         \
  template <>                                                                  \
  __global__ void AMODpStandardTest<T>(                                        \
      int loop, int skip, long long int *start_time,                           \
      long long int *end_time, T *dest, T *ret_val,                            \
      AddrMode addr_mode, TestType type, ShmemContextType ctx_type,            \
      int wf_size, int num_pes, long long* done_flags) {                       \
    __shared__ rocshmem_ctx_t ctx;                                             \
    int wg_id     = get_flat_grid_id();                                        \
    int wf_id = get_flat_block_id() / wf_size;                                 \
    int wg_offset = wg_id * ((get_flat_block_size() - 1 ) / wf_size + 1);      \
    int idx = wf_id + wg_offset;                                               \
    int n_wgs     = get_grid_num_blocks();                                     \
    bool is_default_ctx = true;                                                \
    switch (type) {                                                            \
      case AMO_Add_DPTestType:                                                 \
        rocshmem_wg_ctx_create(ctx_type, &ctx);                                \
        is_default_ctx = false;                                                \
        break;                                                                 \
      default:                                                                 \
        break;                                                                 \
    }                                                                          \
    int dst_pe = 1;                                                            \
    uint32_t num_qps_per_pe = 0;                                               \
    if (is_default_ctx) {                                                      \
      num_qps_per_pe = rocshmem_ctx_num_qps_per_pe();                          \
    } else {                                                                   \
      num_qps_per_pe = rocshmem_ctx_num_qps_per_pe(ctx);                       \
    }                                                                          \
    if (num_qps_per_pe == 0) {                                                 \
      return;                                                                  \
    }                                                                          \
    int wf_qp_index = idx % num_qps_per_pe;                                    \
    int qp_idx =  wf_qp_index * num_pes + dst_pe;                              \
    for (int i = 0; i < loop + skip; i++) {                                    \
      T *ptr = compute_target_ptr<T>(dest, addr_mode, wg_id, i, n_wgs);        \
      T ret = 0;                                                               \
      if (i == skip) {                                                         \
        __syncthreads();                                                       \
        if (is_thread_zero_in_wave()) {                                        \
          wave_amo_quite_dp(done_flags, idx, is_default_ctx, ctx, qp_idx, dst_pe, true); \
          start_time[idx] = wall_clock64();                                    \
        }                                                                      \
      }                                                                        \
      if (!is_thread_zero_in_wave()) {                                         \
        break;                                                                 \
      }                                                                        \
      switch (type) {                                                          \
        case DefaultCtx_AMO_Add_DPTestType:                                    \
          rocshmem_##TNAME##_atomic_add_dp((T *)ptr, 2, qp_idx, dst_pe);              \
          break;                                                                      \
        case AMO_Add_DPTestType:                                                      \
          rocshmem_ctx_##TNAME##_atomic_add_dp(ctx, (T *)ptr, 2, qp_idx, dst_pe);     \
          break;                                                               \
        default:                                                               \
          break;                                                               \
      }                                                                        \
    }                                                                          \
    __syncthreads();                                                           \
    if (is_thread_zero_in_wave()) {                                            \
      wave_amo_quite_dp(done_flags, idx, is_default_ctx, ctx, qp_idx, dst_pe, false); \
      end_time[idx] = wall_clock64();                                          \
    }                                                                          \
    __syncthreads();                                                           \
    if (!is_default_ctx) {                                                     \
      rocshmem_wg_ctx_destroy(&ctx);                                           \
    }                                                                          \
    return;                                                                    \
  }                                                                            \
  template class AMODpStandardTester<T>;

AMO_STANDARD_DEF_GEN(long long, longlong)