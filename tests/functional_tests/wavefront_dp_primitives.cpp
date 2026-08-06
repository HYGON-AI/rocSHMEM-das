// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#include "wavefront_dp_primitives.hpp"

#include <rocshmem/rocshmem.hpp>

#include <numeric>

using namespace rocshmem;

/******************************************************************************
 * DEVICE TEST KERNEL
 *****************************************************************************/
template <typename T>
 __global__ void WaveDpPrimitiveTest(int loop, int skip,
                                       long long int *start_time,
                                       long long int *end_time, T *source,
                                       T *dest, size_t size, TestType type,
                                       ShmemContextType ctx_type,
                                       int wf_size, int num_pes,
                                       signed char *dest_flags, signed char *src_flags) {
  return;
}

/******************************************************************************
 * HOST TESTER CLASS METHODS
 *****************************************************************************/
template <typename T>
WaveDpPrimitiveTester<T>::WaveDpPrimitiveTester(TesterArguments args)
    : Tester(args) {
  buff_size = max_msg_size * sizeof(T) * args.num_wgs * num_warps;
  source = (T *)rocshmem_malloc(buff_size);
  dest = (T *)rocshmem_malloc(buff_size);
  src_flags = (signed char *)rocshmem_malloc(sizeof(signed char) * args.num_wgs * num_warps);
  dest_flags = (signed char *)rocshmem_malloc(sizeof(signed char) * args.num_wgs * num_warps);

  if (source == nullptr || dest == nullptr || src_flags == nullptr || dest_flags == nullptr) {
    std::cerr << "Error allocating memory from symmetric heap" << std::endl;
    std::cerr << "source: " << source << ", dest: " << dest << ", src_flags: " << src_flags << ", dest_flags: " << dest_flags << std::endl;
    if (source) {
      rocshmem_free(source);
    }
    if (dest) {
      rocshmem_free(dest);
    }
    if (src_flags) {
      rocshmem_free(src_flags);
    }
    if (dest_flags) {
      rocshmem_free(dest_flags);
    }
    rocshmem_global_exit(1);
  }

  for(size_t i = 0; i < buff_size; i++) {
    source[i] = static_cast<T>('a' + i % 26);
  }

  check_id = (_type == DefaultCtx_WAVEPutNBI_DPTestType || _type == WAVEPutNBI_DPTestType)
              ? 1
              : 0;
}

template <typename T>
WaveDpPrimitiveTester<T>::~WaveDpPrimitiveTester() {
  rocshmem_free(source);
  rocshmem_free(dest);
  rocshmem_free(src_flags);
  rocshmem_free(dest_flags);
}

template <typename T>
void WaveDpPrimitiveTester<T>::resetBuffers(size_t size) {
  memset(dest, '1', buff_size);
  memset(src_flags, '0', sizeof(signed char) * args.num_wgs * num_warps);
  memset(dest_flags, '0', sizeof(signed char) * args.num_wgs * num_warps);
}

template <typename T>
static __global__ void verify_results_kernel(T *source, T *dest, size_t buf_size,
                                                  bool *verification_error) {
  int idx = get_flat_id();

  if (idx >= buf_size) {
    return;
  }

  if (dest[idx] != source[idx]) {
    *verification_error = true;
  }
}

static __device__ void verify_and_set_flags(signed char *dest_flags, signed char *src_flags, int idx, signed char expected_flag) {
  signed char dest_flags_t = '0';
  do {
    dest_flags_t = __hip_atomic_load(dest_flags + idx, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  } while (dest_flags_t != expected_flag);

  // set src_flags of PE 0 to 1 to signal that the kernel is done
  signed char *src_flags_ptr = src_flags + idx;                       
  dest_flags[idx] = DEST_RETURN_FLAG;
  rocshmem_schar_put_nbi_wave_dp(src_flags_ptr, &dest_flags[idx], 1, 0, 0);        
}

static __global__ void wave_verify_and_set_flags(signed char *dest_flags, signed char *src_flags, int wf_size) {
  if (is_thread_zero_in_wave()) {
    int wg_id = get_flat_grid_id();
    int wf_id = get_flat_block_id() / wf_size;
    int wg_offset = wg_id * ((get_flat_block_size() - 1 ) / wf_size + 1);   
    int idx = wf_id + wg_offset; 
   
    // skip
    verify_and_set_flags(dest_flags, src_flags, idx, SKIP_DONE_FLAG);

    // loop
    verify_and_set_flags(dest_flags, src_flags, idx, LOOP_DONE_FLAG);
  } 
}

template <typename T>
void WaveDpPrimitiveTester<T>::launchKernel(dim3 gridSize, dim3 blockSize,
                                           int loop, size_t size) {
  size_t shared_bytes = 0;

  /* PE 0 performs data transmission and sets dest_flags, 
   * PE 1 verifies dest_flags and sets src_flags
   */
  if (args.myid == 0) {
    hipLaunchKernelGGL(WaveDpPrimitiveTest, gridSize, blockSize, shared_bytes,
                      stream, loop, args.skip, start_time, end_time,
                      source, dest, size, _type, _shmem_context,
                      wf_size, args.numprocs, dest_flags, src_flags);
  } else if (args.myid == check_id) {
    // wait amo done flags before verifying results
    hipLaunchKernelGGL(wave_verify_and_set_flags, gridSize, blockSize, shared_bytes, stream, dest_flags, src_flags, wf_size);
  }

  num_msgs = (loop + args.skip) * gridSize.x * num_warps;
  num_timed_msgs = loop * gridSize.x * num_warps;
}

template <typename T>
void WaveDpPrimitiveTester<T>::verifyResults(size_t size) {
  if (args.myid == check_id) {
    size_t buf_size = size * args.num_wgs * num_warps;
    size_t verify_wg_size = std::min((size_t) 1024, buf_size);
    size_t verify_num_wgs = buf_size / verify_wg_size;

    hipLaunchKernelGGL(verify_results_kernel, verify_num_wgs, verify_wg_size, 0, stream,
                       source, dest, buf_size, verification_error);
    CHECK_HIP(hipStreamSynchronize(stream));
    if (*verification_error) {
      for (size_t i = 0; i < buf_size; i++) {
        if (dest[i] != source[i]) {
          std::cerr << "Data validation error at idx " << i << std::endl;
          std::cerr << " Got " << dest[i] << ", Expected "
                    << source[i] << std::endl;
          exit(-1);
        }
      }
      *verification_error = false;
    }
  }
}

__device__ void wave_quite_dp(signed char *dest_flags, signed char *src_flags, int idx, bool is_default_ctx, rocshmem_ctx_t ctx, int qp_idx, int dst_pe, signed char flag_set) {
  signed char *dest_tag_ptr = dest_flags + idx;
  src_flags[idx] = flag_set;

  if (is_default_ctx) {
    rocshmem_schar_put_nbi_wave_dp(dest_tag_ptr, &src_flags[idx], 1, qp_idx, dst_pe);
  } else {   
    rocshmem_ctx_schar_put_nbi_wave_dp(ctx, dest_tag_ptr, &src_flags[idx], 1, qp_idx, dst_pe);
  }
  signed char src_flags_t = '0';                                           
  do {
    src_flags_t = __hip_atomic_load(src_flags + idx, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  } while (src_flags_t != DEST_RETURN_FLAG);
}

#define RMA_DEF_GEN(T, TNAME)                                                 \
  template <>                                                                 \
  __global__ void WaveDpPrimitiveTest<T>(int loop, int skip,                  \
                                      long long int *start_time,              \
                                      long long int *end_time, T *source,     \
                                      T *dest, size_t size, TestType type,    \
                                      ShmemContextType ctx_type,              \
                                      int wf_size, int num_pes,               \
                                      signed char *dest_flags, signed char *src_flags) {          \
    __shared__ rocshmem_ctx_t ctx;                                            \
    int wg_id = get_flat_grid_id();                                           \
    bool is_default_ctx = true;                                               \
    switch (type) {                                                           \
      case WAVEPutNBI_DPTestType:                                             \
        rocshmem_wg_ctx_create(ctx_type, &ctx);                               \
        is_default_ctx = false;                                               \
        break;                                                                \
      default:                                                                \
        break;                                                                \
    }                                                                         \
    int wf_id = get_flat_block_id() / wf_size;                                \
    int wg_offset = wg_id * ((get_flat_block_size() - 1 ) / wf_size + 1);     \
    int idx = wf_id + wg_offset;                                              \
    size_t offset = size * idx;                                               \
    source += offset;                                                         \
    dest += offset;                                                           \
    int dst_pe = 1;                                                           \
    uint32_t num_qps_per_pe = 0;                                              \
    if (is_default_ctx) {                                                     \
      num_qps_per_pe = rocshmem_ctx_num_qps_per_pe();                         \
    } else {                                                                  \
      num_qps_per_pe = rocshmem_ctx_num_qps_per_pe(ctx);                      \
    }                                                                         \
    if (num_qps_per_pe == 0) {                                                \
      return;                                                                 \
    }                                                                         \
    int wf_qp_index = idx % num_qps_per_pe;                                   \
    int qp_idx =  wf_qp_index * num_pes + dst_pe;                             \
    for (int i = 0; i < loop + skip; i++) {                                   \
      if (i == skip) {                                                        \
        __syncthreads();                                                      \
        if (is_thread_zero_in_wave()) {                                       \
          wave_quite_dp(dest_flags, src_flags, idx, is_default_ctx, ctx, qp_idx, dst_pe, SKIP_DONE_FLAG); \
          start_time[idx] = wall_clock64();                                   \
        }                                                                     \
      }                                                                       \
      switch (type) {                                                                      \
        case DefaultCtx_WAVEPutNBI_DPTestType:                                             \
          rocshmem_##TNAME##_put_nbi_wave_dp(dest, source, size, qp_idx, dst_pe);          \
          break;                                                                           \
        case WAVEPutNBI_DPTestType:                                                        \
          rocshmem_ctx_##TNAME##_put_nbi_wave_dp(ctx, dest, source, size, qp_idx, dst_pe); \
          break;                                                                           \
        default:                                                              \
          break;                                                              \
      }                                                                       \
    }                                                                         \
    __syncthreads();                                                          \
    if (is_thread_zero_in_wave()) {                                           \
      /*__threadfence(); */                                                   \
      wave_quite_dp(dest_flags, src_flags, idx, is_default_ctx, ctx, qp_idx, dst_pe, LOOP_DONE_FLAG); \
      end_time[idx] = wall_clock64();                                          \
    }                                                                         \
    __syncthreads();                                                          \
    if (!is_default_ctx) {                                                    \
      rocshmem_wg_ctx_destroy(&ctx);                                          \
    }                                                                         \
    return;                                                                   \
  }                                                                           \
  template class WaveDpPrimitiveTester<T>;                                      

RMA_DEF_GEN(signed char, schar)
