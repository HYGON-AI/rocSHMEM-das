// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

// Compiled separately for each SDK; only the value interface crosses this boundary.
#include <cerrno>
#include <cstddef>
#include <cstring>
#include <dlfcn.h>
#include <pthread.h>
#if defined(ROCSHMEM_MULTI_PROVIDER_SHCA_ADAPTER)
#include <infiniband/verbs.h>
#elif defined(ROCSHMEM_MULTI_PROVIDER_MLX5_ADAPTER)
#include "../ibv_core.hpp"
#else
#error "Select the MLX5 or SHCA multi-provider verbs adapter"
#endif
#include "verbs_adapter.hpp"

namespace {
using namespace rocshmem::verbs_adapter;
int (*query_port_fn)(ibv_context*, uint8_t, ibv_port_attr*);
int (*modify_qp_fn)(ibv_qp*, ibv_qp_attr*, int);
ibv_ah* (*create_ah_fn)(ibv_pd*, ibv_ah_attr*);
ibv_qp* (*create_qp_fn)(ibv_pd*, ibv_qp_init_attr*);

ContextInfo context_info(void* ptr) {
  auto* ctx = static_cast<ibv_context*>(ptr);
  return {ctx->device, ctx->cmd_fd, ctx->async_fd,
          ctx->num_comp_vectors, ctx->abi_compat};
}
int query_port(void* ptr, uint8_t port, PortAttr* out) {
  auto* ctx = static_cast<ibv_context*>(ptr);
  ibv_port_attr attr{};
  auto* ext = verbs_get_ctx_op(ctx, query_port);
  int rc = ext ? ext->query_port(ctx, port, &attr, sizeof(attr))
               : query_port_fn(ctx, port, &attr);
  if (rc) return rc;
#define COPY(field) out->field = attr.field
  COPY(state); COPY(max_mtu); COPY(active_mtu); COPY(gid_tbl_len);
  COPY(port_cap_flags); COPY(max_msg_sz); COPY(bad_pkey_cntr); COPY(qkey_viol_cntr);
  COPY(pkey_tbl_len); COPY(port_cap_flags2); COPY(lmc); COPY(max_vl_num);
  COPY(sm_sl); COPY(subnet_timeout); COPY(init_type_reply); COPY(active_width);
  COPY(active_speed); COPY(phys_state); COPY(link_layer); COPY(flags);
#if defined(ROCSHMEM_MULTI_PROVIDER_SHCA_ADAPTER)
  out->lid = u17_to_32(attr.lid);
  out->sm_lid = u17_to_32(attr.sm_lid);
#else
  COPY(lid); COPY(sm_lid);
#endif
#undef COPY
  return 0;
}
void* alloc_parent_domain(void* ptr, void* attr) {
  return ibv_alloc_parent_domain(static_cast<ibv_context*>(ptr),
      static_cast<ibv_parent_domain_init_attr*>(attr));
}
void* create_cq_ex(void* ptr, void* attr) {
  return ibv_create_cq_ex(static_cast<ibv_context*>(ptr),
      static_cast<ibv_cq_init_attr_ex*>(attr));
}
void* create_qp_ex(void* ptr, void* value) {
  auto* ctx = static_cast<ibv_context*>(ptr);
  auto* attr = static_cast<ibv_qp_init_attr_ex*>(value);
  if (attr->comp_mask == IBV_QP_INIT_ATTR_PD)
    return create_qp_fn(attr->pd, reinterpret_cast<ibv_qp_init_attr*>(attr));
  auto* ext = verbs_get_ctx_op(ctx, create_qp_ex);
  if (!ext) { errno = EOPNOTSUPP; return nullptr; }
  return ext->create_qp_ex(ctx, attr);
}
int to_ibv_ah_attr(const AhAttr& src, ibv_ah_attr* dst) {
#if defined(ROCSHMEM_MULTI_PROVIDER_SHCA_ADAPTER)
  if (src.dlid > 0x1ffffu) return EOVERFLOW;
  dst->dlid = u32_to_17(src.dlid);
#else
  if (src.dlid > UINT16_MAX) return EOVERFLOW;
  dst->dlid = static_cast<uint16_t>(src.dlid);
#endif
  std::memcpy(&dst->grh.dgid, src.dgid, sizeof(src.dgid));
  dst->grh.flow_label = src.flow_label;
  dst->grh.sgid_index = src.sgid_index;
  dst->grh.hop_limit = src.hop_limit;
  dst->grh.traffic_class = src.traffic_class;
#define COPY(field) dst->field = src.field
  COPY(sl); COPY(src_path_bits); COPY(static_rate); COPY(is_global); COPY(port_num);
#undef COPY
  return 0;
}
int modify_qp(void* ptr, const QpAttr* src, int mask) {
  ibv_qp_attr attr{};
#define COPY(flag, field) if (mask & flag) attr.field = static_cast<decltype(attr.field)>(src->field)
  COPY(IBV_QP_STATE, qp_state); COPY(IBV_QP_CUR_STATE, cur_qp_state);
  COPY(IBV_QP_PATH_MTU, path_mtu); COPY(IBV_QP_PATH_MIG_STATE, path_mig_state);
  COPY(IBV_QP_QKEY, qkey); COPY(IBV_QP_RQ_PSN, rq_psn); COPY(IBV_QP_SQ_PSN, sq_psn);
  COPY(IBV_QP_DEST_QPN, dest_qp_num); COPY(IBV_QP_ACCESS_FLAGS, qp_access_flags);
  COPY(IBV_QP_PKEY_INDEX, pkey_index); COPY(IBV_QP_EN_SQD_ASYNC_NOTIFY, en_sqd_async_notify);
  COPY(IBV_QP_MAX_QP_RD_ATOMIC, max_rd_atomic);
  COPY(IBV_QP_MAX_DEST_RD_ATOMIC, max_dest_rd_atomic);
  COPY(IBV_QP_MIN_RNR_TIMER, min_rnr_timer); COPY(IBV_QP_PORT, port_num);
  COPY(IBV_QP_TIMEOUT, timeout); COPY(IBV_QP_RETRY_CNT, retry_cnt);
  COPY(IBV_QP_RNR_RETRY, rnr_retry); COPY(IBV_QP_RATE_LIMIT, rate_limit);
#undef COPY
  if (mask & IBV_QP_CAP) {
    attr.cap = {src->max_send_wr, src->max_recv_wr, src->max_send_sge,
                src->max_recv_sge, src->max_inline_data};
  }
  if (mask & IBV_QP_AV) {
    int rc = to_ibv_ah_attr(src->ah_attr, &attr.ah_attr);
    if (rc) { errno = rc; return rc; }
  }
  if (mask & IBV_QP_ALT_PATH) {
    int rc = to_ibv_ah_attr(src->alt_ah_attr, &attr.alt_ah_attr);
    if (rc) { errno = rc; return rc; }
    attr.alt_pkey_index = src->alt_pkey_index;
    attr.alt_port_num = src->alt_port_num;
    attr.alt_timeout = src->alt_timeout;
  }
  return modify_qp_fn(static_cast<ibv_qp*>(ptr), &attr, mask);
}
void* create_ah(void* ptr, const AhAttr* src) {
  ibv_ah_attr attr{};
  int rc = to_ibv_ah_attr(*src, &attr);
  if (rc) { errno = rc; return nullptr; }
  return create_ah_fn(static_cast<ibv_pd*>(ptr), &attr);
}
const Api api{context_info, query_port, alloc_parent_domain, create_cq_ex,
              create_qp_ex, modify_qp, create_ah};
} // namespace

namespace rocshmem::verbs_adapter {
#if defined(ROCSHMEM_MULTI_PROVIDER_SHCA_ADAPTER)
const Api* init_shca_api(void* verbs_library) {
#else
const Api* init_mlx5_api(void* verbs_library) {
#endif
#define LOAD(name) name##_fn = reinterpret_cast<decltype(name##_fn)>(dlsym(verbs_library, "ibv_" #name)); if (!name##_fn) return nullptr
  LOAD(query_port); LOAD(modify_qp); LOAD(create_ah); LOAD(create_qp);
#undef LOAD
  return &api;
}
} // namespace rocshmem::verbs_adapter
