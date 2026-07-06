#include "gda/queue_pair.hpp"
#include "util.hpp"
#include "containers/free_list_impl.hpp"
#include "gda/endian.hpp"
#include "segment_builder_shca.hpp"

namespace rocshmem {

__device__ void QueuePair::shca_ring_doorbell_dp(uint64_t db_val, uint64_t my_sq_counter) {
  *dbrec = (uint32_t)my_sq_counter;
  __atomic_signal_fence(__ATOMIC_SEQ_CST);

  __hip_atomic_store(shca_db.ptr, db_val, __ATOMIC_SEQ_CST, __HIP_MEMORY_SCOPE_SYSTEM);
  
  if (shca_fwb_bufsize > 0) {
        uint64_t db_uint = __hip_atomic_load(&shca_db.uint, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        db_uint ^= 0x100;
      __hip_atomic_store(&shca_db.uint, db_uint, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  }
}

__device__ void QueuePair::shca_post_wqe_rma_dp_single_lane(int32_t size, uintptr_t laddr, uintptr_t raddr, uint8_t opcode) {
  uint64_t my_sq_counter = __hip_atomic_fetch_add(&sq_posted, 1, __ATOMIC_SEQ_CST, __HIP_MEMORY_SCOPE_AGENT);
  uint64_t my_sq_index = my_sq_counter & (sq_wqe_cnt - 1);

  SegmentBuilder_SHCA seg_build(my_sq_index, sq_buf);
  seg_build.update_ctrl_seg(my_sq_counter, opcode, 0, qp_num, 0, 3, 0, 0);
  seg_build.update_raddr_seg(raddr, rkey);       
  seg_build.update_data_seg(laddr, size, lkey_hdp); 

  __atomic_signal_fence(__ATOMIC_SEQ_CST);

  if (gpuHdpReg != nullptr) {
    __hip_atomic_store(reinterpret_cast<uint32_t*>(gpuHdpReg), (uint32_t)0x1, __ATOMIC_SEQ_CST, __HIP_MEMORY_SCOPE_SYSTEM);
  }

  uint64_t wqe_writed {0};
  do {
    wqe_writed = __hip_atomic_load(&sq_wqe_writed, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  } while (wqe_writed != my_sq_counter);

  uint8_t *base_ptr = reinterpret_cast<uint8_t*>(sq_buf);
  uint64_t* ctrl_wqe_8B_for_db = reinterpret_cast<uint64_t*>(&base_ptr[64 * my_sq_index]);
  shca_ring_doorbell_dp(*ctrl_wqe_8B_for_db, my_sq_counter + 1); 

  __hip_atomic_store(&sq_wqe_writed, my_sq_counter + 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__ void QueuePair::shca_post_wqe_amo_dp_single_lane(int32_t size, uintptr_t raddr, uint8_t opcode,
                                                            int64_t atomic_data, int64_t atomic_cmp, bool fetching) {
  uint64_t my_sq_counter = __hip_atomic_fetch_add((uint64_t*)&sq_posted, 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  uint64_t my_sq_index = my_sq_counter & (sq_wqe_cnt - 1);

  SegmentBuilder_SHCA seg_build(my_sq_index, sq_buf);
  seg_build.update_ctrl_seg(my_sq_counter, opcode, 0, qp_num, 0, 4, 0, 0);
  seg_build.update_raddr_seg(raddr, rkey);
  seg_build.update_atomic_seg(atomic_data, atomic_cmp);
  seg_build.update_data_seg(reinterpret_cast<uintptr_t>(nonfetching_atomic), 8, nonfetching_atomic_lkey);
  
  __atomic_signal_fence(__ATOMIC_SEQ_CST);

  // if (gpuHdpReg != nullptr) {
  //   __hip_atomic_store(reinterpret_cast<uint32_t*>(gpuHdpReg), (uint32_t)0x1, __ATOMIC_SEQ_CST, __HIP_MEMORY_SCOPE_SYSTEM);
  // }

  uint64_t wqe_writed {0};
  do {
    wqe_writed = __hip_atomic_load(&sq_wqe_writed, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  } while (wqe_writed != my_sq_counter);

  uint8_t *base_ptr = reinterpret_cast<uint8_t*>(sq_buf);
  uint64_t* ctrl_wqe_8B_for_db = reinterpret_cast<uint64_t*>(&base_ptr[64 * my_sq_index]);
  shca_ring_doorbell_dp(*ctrl_wqe_8B_for_db, my_sq_counter + 1);

  __hip_atomic_store(&sq_wqe_writed, my_sq_counter + 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__ void QueuePair::shca_quiet_dp_single_lane() {
  uint64_t posted = __hip_atomic_load(&quiet_posted, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);
  uint64_t completed = __hip_atomic_load(&quiet_completed, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);

  uint64_t wqe_id_max = 0; 
  uint64_t processed = 0;

  for(uint64_t cq_consumer = completed;cq_consumer < posted; ++cq_consumer){
    uint64_t cq_index = cq_consumer & (cq_cnt - 1); 
    volatile shca_cqe64 *cqe_entry = &shca_cq_buf[cq_index];
    uint8_t own_se_op = *((volatile uint8_t*)&cqe_entry->own_se_op);
    uint8_t owner_bit = (cq_consumer >> cq_log_cnt) & 1;

    if ((own_se_op & 1) == owner_bit || (own_se_op >> 4) == SHCA_CQE_INVALID) {
      break; 
    }

    uint16_t wqe_counter = *((volatile uint16_t*)&cqe_entry->wqe_counter);

    uint64_t wqe_id =  outstanding_wqes[wqe_counter];
    if(wqe_id > wqe_id_max)     wqe_id_max = wqe_id;

    uint8_t shca_invld_bits = SHCA_CQE_INVALID << 4 | owner_bit;
    *((volatile uint8_t*)&cqe_entry->own_se_op) = shca_invld_bits;
    __atomic_signal_fence(__ATOMIC_SEQ_CST);
    processed++;
  }

  if (processed > 0) {
    uint64_t new_completed = completed + processed;

    uint32_t db_val = static_cast<uint32_t>(new_completed);
    __atomic_store_n(cq_dbrec, db_val, __ATOMIC_SEQ_CST);
    __atomic_signal_fence(__ATOMIC_SEQ_CST);

    __hip_atomic_store(&quiet_completed, new_completed, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
  }
}

}
