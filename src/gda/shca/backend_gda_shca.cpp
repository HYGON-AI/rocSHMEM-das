// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#include "gda/backend_gda.hpp"
#include "util.hpp"

namespace rocshmem {

void* GDABackend::shca_dv_dlopen() {
  void* dv_handle{nullptr};
  dv_handle = dlopen("libshca.so", RTLD_NOW);
  if (!dv_handle) {
    DPRINTF("Could not open libshca.so. Returning\n");
  }
  return dv_handle;
}

int GDABackend::shca_dv_dl_init() {
  shcadv_handle_ = shca_dv_dlopen();
  if (!shcadv_handle_)
    return ROCSHMEM_ERROR;

  DLSYM_HELPER(shcadv, shca_dv_, shcadv_handle_, init_obj);
  return ROCSHMEM_SUCCESS;
}

void GDABackend::shca_initialize_gpu_qp(QueuePair* gpu_qp, int conn_num) {
  shca_dv_cq cq_out;
  shca_dv_obj shca_obj;
  shca_obj.cq.in = cqs[conn_num];
  shca_obj.cq.out = &cq_out;
  shcadv.init_obj(&shca_obj, SHCA_DV_OBJ_CQ);
  dump_shcadv_cq(&cq_out, conn_num);

  gpu_qp->shca_cq_buf = reinterpret_cast<shca_cqe64*>(cq_out.buf);
  gpu_qp->cq_cnt = cq_out.cqe_cnt;
  gpu_qp->cq_log_cnt = log2(cq_out.cqe_cnt);
  gpu_qp->cq_dbrec = reinterpret_cast<uint32_t*>(cq_out.dbrec);

  shca_dv_qp qp_out;
  shca_obj.qp.in = qps[conn_num];
  shca_obj.qp.out = &qp_out;
  shcadv.init_obj(&shca_obj, SHCA_DV_OBJ_QP);
  dump_shcadv_qp(&qp_out, conn_num);

  gpu_qp->dbrec = reinterpret_cast<uint32_t*>(&qp_out.dbrec[1]); // points to two pointers: 0 -> SHCA_RCV_DBR, 1 -> SHCA_SND_DBR
  gpu_qp->sq_buf = reinterpret_cast<uint64_t*>(qp_out.sq.buf);
  gpu_qp->sq_wqe_cnt = qp_out.sq.wqe_cnt;
  gpu_qp->rkey = heap_rkey[conn_num % num_pes];
  gpu_qp->lkey = heap_mr->lkey;
  gpu_qp->rkey_hdp = heap_rkey_hdp[conn_num % num_pes];
  gpu_qp->lkey_hdp = heap_mr_hdp->lkey;
  gpu_qp->qp_num = qps[conn_num]->qp_num;
  gpu_qp->inline_threshold = inline_threshold;
  // The 2 in qp_out.fwb.size * 2 below facilitates the switching between fast wqe buffer registers

  int hip_dev_id{-1};
  CHECK_HIP(hipGetDevice(&hip_dev_id));
  void* gpu_ptr{nullptr};
  if (qp_out.fwb.size > 0) {
      rocm_memory_lock_to_fine_grain(qp_out.fwb.reg, qp_out.fwb.size * 2, &gpu_ptr, hip_dev_id);
  } else {
      rocm_memory_lock_to_fine_grain(qp_out.fwb.reg, 8, &gpu_ptr, hip_dev_id);
  }
  gpu_qp->shca_fwb_bufsize = qp_out.fwb.size;
  gpu_qp->shca_db.ptr = reinterpret_cast<uint64_t*>(gpu_ptr);
}

}  // namespace rocshmem
