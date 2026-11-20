// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

// Private implementation included only by ibv_wrapper.cpp in namespace rocshmem.
namespace {
verbs_adapter::AhAttr to_adapter_ah_attr(const ibv_ah_attr& src) {
  verbs_adapter::AhAttr out{};
  std::memcpy(out.dgid, &src.grh.dgid, sizeof(out.dgid));
  out.flow_label = src.grh.flow_label;
  out.sgid_index = src.grh.sgid_index;
  out.hop_limit = src.grh.hop_limit;
  out.traffic_class = src.grh.traffic_class;
  out.dlid = u17_to_32(src.dlid);
#define COPY(field) out.field = src.field
  COPY(sl); COPY(src_path_bits); COPY(static_rate); COPY(is_global); COPY(port_num);
#undef COPY
  return out;
}
verbs_adapter::QpAttr to_adapter_qp_attr(const ibv_qp_attr& src, int mask) {
  verbs_adapter::QpAttr out{};
#define COPY(flag, field) if (mask & flag) out.field = src.field
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
    out.max_send_wr = src.cap.max_send_wr;
    out.max_recv_wr = src.cap.max_recv_wr;
    out.max_send_sge = src.cap.max_send_sge;
    out.max_recv_sge = src.cap.max_recv_sge;
    out.max_inline_data = src.cap.max_inline_data;
  }
  if (mask & IBV_QP_AV) out.ah_attr = to_adapter_ah_attr(src.ah_attr);
  if (mask & IBV_QP_ALT_PATH) {
    out.alt_ah_attr = to_adapter_ah_attr(src.alt_ah_attr);
    out.alt_pkey_index = src.alt_pkey_index;
    out.alt_port_num = src.alt_port_num;
    out.alt_timeout = src.alt_timeout;
  }
  return out;
}
void to_ibv_port_attr(const verbs_adapter::PortAttr& src, ibv_port_attr* dst) {
  *dst = {};
#define COPY(field) dst->field = static_cast<decltype(dst->field)>(src.field)
  COPY(state); COPY(max_mtu); COPY(active_mtu); COPY(gid_tbl_len);
  COPY(port_cap_flags); COPY(max_msg_sz); COPY(bad_pkey_cntr); COPY(qkey_viol_cntr);
  COPY(pkey_tbl_len); COPY(port_cap_flags2); COPY(lmc); COPY(max_vl_num);
  COPY(sm_sl); COPY(subnet_timeout); COPY(init_type_reply); COPY(active_width);
  COPY(active_speed); COPY(phys_state); COPY(link_layer); COPY(flags);
#undef COPY
  dst->lid = u32_to_17(src.lid);
  dst->sm_lid = u32_to_17(src.sm_lid);
}
} // namespace

const verbs_adapter::Api* IBVWrapper::adapter_api(ibv_context* context) const {
  // Device attributes have the common public layout. Do not dereference the
  // SDK-dependent context before entering its matching SDK implementation.
  ibv_device_attr device{};
  if (ibv.query_device(context, &device))
    LOG_ERROR_EXIT("Could not identify verbs adapter device");
  const unsigned index = device.vendor_id == 0x1cb8 ? 1 : 0;
  if (adapter_apis_[index]) return adapter_apis_[index];
  adapter_apis_[index] = index ? verbs_adapter::init_shca_api(ibv_handle)
                              : verbs_adapter::init_mlx5_api(ibv_handle);
  if (!adapter_apis_[index])
    LOG_ERROR_EXIT("Could not initialize verbs adapter");
  LOG_INFO("Using verbs adapter %s", index ? "shca" : "mlx5");
  return adapter_apis_[index];
}
