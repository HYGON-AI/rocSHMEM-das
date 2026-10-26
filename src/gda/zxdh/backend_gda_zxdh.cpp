// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#include <dlfcn.h>

#include <hip/hip_runtime.h>

#include "gda/backend_gda.hpp"
#include "util.hpp"

namespace rocshmem {

namespace {

uint32_t round_up_pow2(uint32_t value) {
  uint32_t result = 1;
  while (result < value) {
    result <<= 1;
  }
  return result;
}

template <typename T>
bool load_symbol(void *handle, const char *name, T &target) {
  target = reinterpret_cast<T>(dlsym(handle, name));
  return target != nullptr;
}

}  // namespace

void *GDABackend::zxdh_dv_dlopen() {
  void *handle = dlopen("libzrdma.so", RTLD_NOW);
  if (!handle) {
    handle = dlopen("libzrdma.so.1", RTLD_NOW);
  }
  if (!handle) {
    DPRINTF("Could not open libzrdma.so[.1]. Returning\n");
  }
  return handle;
}

int GDABackend::zxdh_dv_dl_init() {
  zxdh_dv_handle_ = zxdh_dv_dlopen();
  if (!zxdh_dv_handle_) {
    return ROCSHMEM_ERROR;
  }

  bool ok = true;
  ok &= load_symbol(zxdh_dv_handle_, "zxdh_devx_get_cq_init_size",
                    zxdh_dv.get_cq_init_size);
  ok &= load_symbol(zxdh_dv_handle_, "zxdh_devx_get_qp_init_size",
                    zxdh_dv.get_qp_init_size);
  ok &= load_symbol(zxdh_dv_handle_, "zxdh_devx_create_cq",
                    zxdh_dv.create_cq);
  ok &= load_symbol(zxdh_dv_handle_, "zxdh_devx_create_qp",
                    zxdh_dv.create_qp);
  ok &= load_symbol(zxdh_dv_handle_, "zxdh_devx_modify_qp",
                    zxdh_dv.modify_qp);
  ok &= load_symbol(zxdh_dv_handle_, "zxdh_devx_destroy_qp",
                    zxdh_dv.destroy_qp);
  ok &= load_symbol(zxdh_dv_handle_, "zxdh_devx_destroy_cq",
                    zxdh_dv.destroy_cq);

  if (!ok) {
    dlclose(zxdh_dv_handle_);
    zxdh_dv_handle_ = nullptr;
    return ROCSHMEM_ERROR;
  }
  return ROCSHMEM_SUCCESS;
}

void GDABackend::zxdh_create_cqs(int ncqes) {
  struct zxdh_devx_cq_init_attr cq_init_attr{};
  struct zxdh_devx_init_cq_resp cq_init_resp{};
  cq_init_attr.cqe_num = round_up_pow2(ncqes);

  int status = zxdh_dv.get_cq_init_size(context, &cq_init_attr, &cq_init_resp);
  CHECK_ZERO(status, "zxdh_devx_get_cq_init_size");

  zxdh_qps.clear();
  zxdh_cqs.clear();
  zxdh_qp_bufs.clear();
  zxdh_cq_bufs.clear();
  zxdh_qps.resize(total_num_qps, nullptr);
  zxdh_cqs.resize(total_num_qps, nullptr);
  zxdh_qp_bufs.resize(total_num_qps, nullptr);
  zxdh_cq_bufs.resize(total_num_qps, nullptr);

  for (size_t i = 0; i < total_num_qps; ++i) {
    void *cq_buf = nullptr;
    CHECK_HIP(hipExtMallocWithFlags(reinterpret_cast<void **>(&cq_buf),
                                    cq_init_resp.total_size,
                                    hipDeviceMallocFinegrained));
    CHECK_HIP(hipMemset(cq_buf, 0, cq_init_resp.total_size));

    struct zxdh_devx_cq_buffers buffers{};
    buffers.cq_base = cq_buf;
    buffers.total_size = cq_init_resp.total_size;

    struct zxdh_devx_cq *cq =
        zxdh_dv.create_cq(context, cq_init_attr.cqe_num, nullptr, 0, &buffers);
    CHECK_NNULL(cq, "zxdh_devx_create_cq");

    zxdh_cq_bufs[i] = cq_buf;
    zxdh_cqs[i] = cq;
    cqs[i] = &cq->verbs_cq.cq;
  }
}

void GDABackend::zxdh_create_qps(int sq_length) {
  struct zxdh_devx_qp_init_attr qp_init_attr{};
  struct zxdh_devx_init_qp_resp qp_init_resp{};
  qp_init_attr.qp_type = IBV_QPT_RC;
  qp_init_attr.cap.max_send_wr = round_up_pow2(sq_length);
  qp_init_attr.cap.max_recv_wr = round_up_pow2(sq_length);
  qp_init_attr.cap.max_send_sge = 1;
  qp_init_attr.cap.max_recv_sge = 1;
  qp_init_attr.cap.max_inline_data = 8;

  int status =
      zxdh_dv.get_qp_init_size(context, &qp_init_attr, &qp_init_resp);
  CHECK_ZERO(status, "zxdh_devx_get_qp_init_size");

  for (size_t i = 0; i < total_num_qps; ++i) {
    void *qp_buf = nullptr;
    CHECK_HIP(hipExtMallocWithFlags(reinterpret_cast<void **>(&qp_buf),
                                    qp_init_resp.total_size,
                                    hipDeviceMallocFinegrained));
    CHECK_HIP(hipMemset(qp_buf, 0, qp_init_resp.total_size));

    struct zxdh_devx_qp_buffers buffers{};
    buffers.qp_base = qp_buf;
    buffers.total_size = qp_init_resp.total_size;

    struct ibv_qp_init_attr attr{};
    attr.qp_type = IBV_QPT_RC;
    attr.send_cq = cqs[i];
    attr.recv_cq = cqs[i];
    attr.cap.max_send_wr = qp_init_attr.cap.max_send_wr;
    attr.cap.max_recv_wr = qp_init_attr.cap.max_recv_wr;
    attr.cap.max_send_sge = qp_init_attr.cap.max_send_sge;
    attr.cap.max_recv_sge = qp_init_attr.cap.max_recv_sge;
    attr.cap.max_inline_data = qp_init_attr.cap.max_inline_data;

    struct zxdh_devx_qp *qp =
        zxdh_dv.create_qp(pd_orig, &attr, &buffers);
    CHECK_NNULL(qp, "zxdh_devx_create_qp");

    if (qp->shadow_size) {
      void *shadow =
          static_cast<char *>(qp_buf) + qp_init_resp.total_size -
          qp->shadow_size;
      CHECK_HIP(hipMemset(shadow, qp->shadow_init_value, qp->shadow_size));
    }

    if (!zxdh_uar_hptr) {
      hipError_t hip_status =
          hipHostRegister(qp->wqe_alloc_db, 4, hipHostRegisterIoMemory);
      if (hip_status == hipSuccess) {
        CHECK_HIP(hipHostGetDevicePointer(&zxdh_uar_dptr,
                                          qp->wqe_alloc_db, 0));
        zxdh_uar_hptr = qp->wqe_alloc_db;
      } else if (hip_status == hipErrorHostMemoryAlreadyRegistered) {
        CHECK_HIP(hipHostGetDevicePointer(&zxdh_uar_dptr,
                                          qp->wqe_alloc_db, 0));
        zxdh_uar_hptr = qp->wqe_alloc_db;
      } else {
        CHECK_HIP(hip_status);
      }
    }

    zxdh_qp_bufs[i] = qp_buf;
    zxdh_qps[i] = qp;
    qps[i] = &qp->verbs_qp.qp;
  }

  zxdh_create_loopback_qps(round_up_pow2(sq_length),
                           round_up_pow2(sq_length));
}

void GDABackend::zxdh_create_loopback_qps(int ncqes, int sq_length) {
  struct zxdh_devx_cq_init_attr cq_init_attr{};
  struct zxdh_devx_init_cq_resp cq_init_resp{};
  cq_init_attr.cqe_num = round_up_pow2(ncqes);
  int status = zxdh_dv.get_cq_init_size(context, &cq_init_attr, &cq_init_resp);
  CHECK_ZERO(status, "zxdh_devx_get_cq_init_size (loopback)");

  struct zxdh_devx_qp_init_attr qp_init_attr{};
  struct zxdh_devx_init_qp_resp qp_init_resp{};
  qp_init_attr.qp_type = IBV_QPT_RC;
  qp_init_attr.cap.max_send_wr = round_up_pow2(sq_length);
  qp_init_attr.cap.max_recv_wr = round_up_pow2(sq_length);
  qp_init_attr.cap.max_send_sge = 1;
  qp_init_attr.cap.max_recv_sge = 1;
  qp_init_attr.cap.max_inline_data = 8;
  status =
      zxdh_dv.get_qp_init_size(context, &qp_init_attr, &qp_init_resp);
  CHECK_ZERO(status, "zxdh_devx_get_qp_init_size (loopback)");

  zxdh_loopback_devx_qps.assign(2, nullptr);
  zxdh_loopback_devx_cqs.assign(2, nullptr);
  zxdh_loopback_qp_bufs.assign(2, nullptr);
  zxdh_loopback_cq_bufs.assign(2, nullptr);

  for (int i = 0; i < 2; ++i) {
    void *cq_buf = nullptr;
    CHECK_HIP(hipExtMallocWithFlags(reinterpret_cast<void **>(&cq_buf),
                                    cq_init_resp.total_size,
                                    hipDeviceMallocFinegrained));
    CHECK_HIP(hipMemset(cq_buf, 0, cq_init_resp.total_size));

    struct zxdh_devx_cq_buffers cq_buffers{};
    cq_buffers.cq_base = cq_buf;
    cq_buffers.total_size = cq_init_resp.total_size;
    struct zxdh_devx_cq *cq = zxdh_dv.create_cq(
        context, cq_init_attr.cqe_num, nullptr, 0, &cq_buffers);
    CHECK_NNULL(cq, "zxdh_devx_create_cq (loopback)");

    void *qp_buf = nullptr;
    CHECK_HIP(hipExtMallocWithFlags(reinterpret_cast<void **>(&qp_buf),
                                    qp_init_resp.total_size,
                                    hipDeviceMallocFinegrained));
    CHECK_HIP(hipMemset(qp_buf, 0, qp_init_resp.total_size));

    struct zxdh_devx_qp_buffers qp_buffers{};
    qp_buffers.qp_base = qp_buf;
    qp_buffers.total_size = qp_init_resp.total_size;
    struct ibv_qp_init_attr attr{};
    attr.qp_type = IBV_QPT_RC;
    attr.send_cq = &cq->verbs_cq.cq;
    attr.recv_cq = &cq->verbs_cq.cq;
    attr.cap.max_send_wr = qp_init_attr.cap.max_send_wr;
    attr.cap.max_recv_wr = qp_init_attr.cap.max_recv_wr;
    attr.cap.max_send_sge = 1;
    attr.cap.max_recv_sge = 1;
    attr.cap.max_inline_data = 8;
    struct zxdh_devx_qp *qp =
        zxdh_dv.create_qp(pd_orig, &attr, &qp_buffers);
    CHECK_NNULL(qp, "zxdh_devx_create_qp (loopback)");

    if (qp->shadow_size) {
      void *shadow = static_cast<char *>(qp_buf) +
                     qp_init_resp.total_size - qp->shadow_size;
      CHECK_HIP(hipMemset(shadow, qp->shadow_init_value, qp->shadow_size));
    }

    zxdh_loopback_devx_qps[i] = qp;
    zxdh_loopback_devx_cqs[i] = cq;
    zxdh_loopback_qp_bufs[i] = qp_buf;
    zxdh_loopback_cq_bufs[i] = cq_buf;
  }

  struct ibv_qp_attr attr{};
  int attr_mask = 0;
  memset(&attr, 0, sizeof(attr));
  attr.qp_state = IBV_QPS_INIT;
  attr.pkey_index = 0;
  attr.port_num = port;
  attr.qp_access_flags =
      IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ |
      IBV_ACCESS_REMOTE_WRITE;
  attr_mask = IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT |
              IBV_QP_ACCESS_FLAGS;
  for (int i = 0; i < 2; ++i) {
    CHECK_ZERO(zxdh_dv.modify_qp(zxdh_loopback_devx_qps[i], &attr,
                                 attr_mask),
               "zxdh_devx_modify_qp (loopback INIT)");
  }

  for (int i = 0; i < 2; ++i) {
    struct zxdh_devx_qp *peer = zxdh_loopback_devx_qps[1 - i];
    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_RTR;
    attr.path_mtu = portinfo.active_mtu;
    attr.dest_qp_num = peer->qpn;
    attr.rq_psn = 0;
    attr.max_dest_rd_atomic = 1;
    attr.min_rnr_timer = 0x12;
    attr.ah_attr.port_num = port;
    if (portinfo.link_layer == IBV_LINK_LAYER_ETHERNET) {
      attr.ah_attr.is_global = 1;
      attr.ah_attr.grh.dgid = gid;
      attr.ah_attr.grh.sgid_index = gid_index;
      attr.ah_attr.grh.hop_limit = 1;
      attr.ah_attr.grh.traffic_class = envvar::gda::traffic_class;
    } else {
      attr.ah_attr.dlid = portinfo.lid;
    }
    attr_mask = IBV_QP_STATE | IBV_QP_PATH_MTU | IBV_QP_RQ_PSN |
                IBV_QP_DEST_QPN | IBV_QP_AV | IBV_QP_MAX_DEST_RD_ATOMIC |
                IBV_QP_MIN_RNR_TIMER;
    CHECK_ZERO(zxdh_dv.modify_qp(zxdh_loopback_devx_qps[i], &attr,
                                 attr_mask),
               "zxdh_devx_modify_qp (loopback RTR)");
  }

  for (int i = 0; i < 2; ++i) {
    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_RTS;
    attr.timeout = 14;
    attr.retry_cnt = 7;
    attr.rnr_retry = 7;
    attr.sq_psn = 0;
    attr.max_rd_atomic = 1;
    attr_mask = IBV_QP_STATE | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC |
                IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY;
    CHECK_ZERO(zxdh_dv.modify_qp(zxdh_loopback_devx_qps[i], &attr,
                                 attr_mask),
               "zxdh_devx_modify_qp (loopback RTS)");
  }
}

void GDABackend::zxdh_initialize_gpu_qp(QueuePair *gpu_qp, int conn_num) {
  struct zxdh_devx_qp *qp = zxdh_qps[conn_num];
  struct zxdh_devx_cq *cq = zxdh_cqs[conn_num];

  gpu_qp->zxdh_sq_buf = zxdh_qp_bufs[conn_num];
  gpu_qp->zxdh_sq_wqe_cnt = qp->sqdepth;
  gpu_qp->zxdh_qpn = qp->qpn;
  gpu_qp->zxdh_dbreg =
      reinterpret_cast<volatile uint32_t *>(zxdh_uar_dptr);
  gpu_qp->zxdh_cq_buf = zxdh_cq_bufs[conn_num];
  gpu_qp->zxdh_cq_cnt = cq->cqe_num;
  gpu_qp->zxdh_sq_resv_head = 0;
  gpu_qp->zxdh_sq_ready_head = 0;
  gpu_qp->zxdh_cq_consumed = 0;
  gpu_qp->zxdh_poll_lock = 0;
  gpu_qp->zxdh_amo_head = 0;
  gpu_qp->zxdh_amo_tail = 0;

  gpu_qp->rkey = heap_rkey[conn_num % num_pes];
  gpu_qp->lkey = heap_mr->lkey;
  gpu_qp->rkey_hdp = heap_rkey_hdp[conn_num % num_pes];
  gpu_qp->lkey_hdp = heap_mr_hdp->lkey;
  gpu_qp->qp_num = qp->qpn;
  gpu_qp->inline_threshold = inline_threshold;
}

void GDABackend::zxdh_initialize_loopback_gpu_qp(QueuePair *gpu_qp,
                                                 int index) {
  struct zxdh_devx_qp *qp = zxdh_loopback_devx_qps[index];
  struct zxdh_devx_cq *cq = zxdh_loopback_devx_cqs[index];

  gpu_qp->zxdh_sq_buf = zxdh_loopback_qp_bufs[index];
  gpu_qp->zxdh_sq_wqe_cnt = qp->sqdepth;
  gpu_qp->zxdh_qpn = qp->qpn;
  gpu_qp->zxdh_dbreg =
      reinterpret_cast<volatile uint32_t *>(zxdh_uar_dptr);
  gpu_qp->zxdh_cq_buf = zxdh_loopback_cq_bufs[index];
  gpu_qp->zxdh_cq_cnt = cq->cqe_num;
  gpu_qp->zxdh_sq_resv_head = 0;
  gpu_qp->zxdh_sq_ready_head = 0;
  gpu_qp->zxdh_cq_consumed = 0;
  gpu_qp->zxdh_poll_lock = 0;
  gpu_qp->zxdh_amo_head = 0;
  gpu_qp->zxdh_amo_tail = 0;
  gpu_qp->zxdh_loopback_qps = zxdh_loopback_gpu_qps;
  gpu_qp->inline_threshold = inline_threshold;
}

void GDABackend::zxdh_setup_loopback_qps() {
  CHECK_HIP(hipMalloc(&zxdh_loopback_gpu_qps,
                      sizeof(QueuePair) * 2));
  zxdh_loopback_host_qps =
      static_cast<QueuePair *>(malloc(sizeof(QueuePair) * 2));
  CHECK_NNULL(zxdh_loopback_host_qps, "malloc (zxdh_loopback_host_qps)");

  for (int i = 0; i < 2; ++i) {
    new (&zxdh_loopback_host_qps[i]) QueuePair(pd_orig, gda_provider);
    zxdh_initialize_loopback_gpu_qp(&zxdh_loopback_host_qps[i], i);
    CHECK_HIP(hipMemcpy(&zxdh_loopback_gpu_qps[i],
                        &zxdh_loopback_host_qps[i], sizeof(QueuePair),
                        hipMemcpyDefault));
  }

  for (size_t i = 0; i < total_num_qps; ++i) {
    gpu_qps[i].zxdh_loopback_qps = zxdh_loopback_gpu_qps;
  }
}

void GDABackend::zxdh_cleanup_loopback_qps() {
  if (zxdh_loopback_host_qps) {
    for (int i = 0; i < 2; ++i) {
      zxdh_loopback_host_qps[i].~QueuePair();
    }
    free(zxdh_loopback_host_qps);
    zxdh_loopback_host_qps = nullptr;
  }
  if (zxdh_loopback_gpu_qps) {
    CHECK_HIP(hipFree(zxdh_loopback_gpu_qps));
    zxdh_loopback_gpu_qps = nullptr;
  }
}

void GDABackend::zxdh_cleanup() {
  for (auto *qp : zxdh_qps) {
    if (qp) {
      zxdh_dv.destroy_qp(qp);
    }
  }
  for (auto *cq : zxdh_cqs) {
    if (cq) {
      zxdh_dv.destroy_cq(cq);
    }
  }
  for (auto *qp : zxdh_loopback_devx_qps) {
    if (qp) {
      zxdh_dv.destroy_qp(qp);
    }
  }
  for (auto *cq : zxdh_loopback_devx_cqs) {
    if (cq) {
      zxdh_dv.destroy_cq(cq);
    }
  }
  for (void *buf : zxdh_qp_bufs) {
    if (buf) {
      CHECK_HIP(hipFree(buf));
    }
  }
  for (void *buf : zxdh_cq_bufs) {
    if (buf) {
      CHECK_HIP(hipFree(buf));
    }
  }
  for (void *buf : zxdh_loopback_qp_bufs) {
    if (buf) {
      CHECK_HIP(hipFree(buf));
    }
  }
  for (void *buf : zxdh_loopback_cq_bufs) {
    if (buf) {
      CHECK_HIP(hipFree(buf));
    }
  }
  if (zxdh_uar_hptr) {
    CHECK_HIP(hipHostUnregister(const_cast<void *>(zxdh_uar_hptr)));
    zxdh_uar_hptr = nullptr;
    zxdh_uar_dptr = nullptr;
  }

  zxdh_qps.clear();
  zxdh_cqs.clear();
  zxdh_qp_bufs.clear();
  zxdh_cq_bufs.clear();
  zxdh_loopback_devx_qps.clear();
  zxdh_loopback_devx_cqs.clear();
  zxdh_loopback_qp_bufs.clear();
  zxdh_loopback_cq_bufs.clear();
}

}  // namespace rocshmem
