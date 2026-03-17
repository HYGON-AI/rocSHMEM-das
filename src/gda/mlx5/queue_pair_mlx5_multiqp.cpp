#include "gda/queue_pair.hpp"
#include "util.hpp"
#include "containers/free_list_impl.hpp"
#include "gda/endian.hpp"

namespace rocshmem {

__device__ void QueuePair::mlx5_post_wqe_rma_dp_single_lane(int32_t size, uintptr_t laddr, uintptr_t raddr, uint8_t opcode) {
  // get SQ lock
  acquire_lock(&mlx5_sq.lock);

  // wqe_idx is the logical WQE id that wraps at 0xFFFF, sq_idx is the index into the actual SQ
  uint16_t wqe_idx = mlx5_sq.tail;
  uint16_t sq_idx = wqe_idx % mlx5_sq.depth;

  // can we inline the data into the WQE?
  bool send_inline = gda_mlx5_wqe_rma::can_inline(opcode, size, inline_threshold);

  // construct the WQE on the stack
  gda_mlx5_wqe wqe{wqe_idx, opcode, qp_num, 0,
                   raddr, rkey, laddr, lkey, static_cast<uint32_t>(size), send_inline};

  // copy to SQ
  mlx5_sq.buf[sq_idx] = wqe;

  // increment tail counter
  mlx5_sq.tail += 1;
  mlx5_sq.post += 1;

  if (gpuHdpReg != nullptr) {
    __hip_atomic_store(reinterpret_cast<uint32_t*>(gpuHdpReg), (uint32_t)0x1, __ATOMIC_SEQ_CST, __HIP_MEMORY_SCOPE_SYSTEM);
  }

  uint64_t wqe_writed {0};
  do {
    wqe_writed = __hip_atomic_load(&sq_wqe_writed, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  } while (wqe_writed != (mlx5_sq.post - 1));

  // ring doorbell for this WQE
  mlx5_ring_doorbell(mlx5_sq.tail, wqe, false);

  __hip_atomic_store(&sq_wqe_writed, mlx5_sq.post, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);

  // release SQ lock
  release_lock(&mlx5_sq.lock);
}

__device__ void QueuePair::mlx5_post_wqe_amo_dp_single_lane(int32_t size, uintptr_t raddr, uint8_t opcode,
                                                            int64_t atomic_data, int64_t atomic_cmp, bool fetching) {
  // get SQ lock
  acquire_lock(&mlx5_sq.lock);

  uint64_t* atomic_laddr = nonfetching_atomic;
  uint32_t atomic_lkey = nonfetching_atomic_lkey;
  

  // wqe_idx is the logical WQE id that wraps at 0xFFFF, sq_idx is the index into the actual SQ
  uint16_t wqe_idx = mlx5_sq.tail;
  uint16_t sq_idx = wqe_idx % mlx5_sq.depth;

  // construct the WQE on the stack
  gda_mlx5_wqe wqe{wqe_idx, opcode, qp_num, 0,
                   raddr, rkey,
                   static_cast<uint64_t>(atomic_data), static_cast<uint64_t>(atomic_cmp),
                   reinterpret_cast<uintptr_t>(atomic_laddr), atomic_lkey};

  // copy to SQ
  mlx5_sq.buf[sq_idx] = wqe;

  // increment tail counter
  mlx5_sq.tail += 1;
  mlx5_sq.post += 1;

  if (gpuHdpReg != nullptr) {
    __hip_atomic_store(reinterpret_cast<uint32_t*>(gpuHdpReg), (uint32_t)0x1, __ATOMIC_SEQ_CST, __HIP_MEMORY_SCOPE_SYSTEM);
  }

  uint64_t wqe_writed {0};
  do {
    wqe_writed = __hip_atomic_load(&sq_wqe_writed, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  } while (wqe_writed != (mlx5_sq.post - 1));

  // ring doorbell for this WQE (note: need to check this for correctness)
  mlx5_ring_doorbell(mlx5_sq.tail, wqe, false);

  __hip_atomic_store(&sq_wqe_writed, mlx5_sq.post, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);

  // release SQ lock
  release_lock(&mlx5_sq.lock);
}

__device__ void QueuePair::mlx5_quiet_dp_single_lane() {
  mlx5_poll_cq_until(mlx5_sq.depth);
}

}

