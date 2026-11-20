// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

// Private SDK boundary. These are values, not copies of a vendor ABI.
#pragma once
#include <cstdint>

namespace rocshmem::verbs_adapter {
struct ContextInfo {
  void* device;
  int cmd_fd, async_fd, num_comp_vectors;
  void* abi_compat;
};
struct PortAttr {
  int state, max_mtu, active_mtu, gid_tbl_len;
  uint32_t port_cap_flags, max_msg_sz, bad_pkey_cntr, qkey_viol_cntr;
  uint32_t lid, sm_lid;
  uint16_t pkey_tbl_len, port_cap_flags2;
  uint8_t lmc, max_vl_num, sm_sl, subnet_timeout, init_type_reply;
  uint8_t active_width, active_speed, phys_state, link_layer, flags;
};
struct AhAttr {
  uint8_t dgid[16];
  uint32_t flow_label, dlid;
  uint8_t sgid_index, hop_limit, traffic_class;
  uint8_t sl, src_path_bits, static_rate, is_global, port_num;
};
struct QpAttr {
  int qp_state, cur_qp_state, path_mtu, path_mig_state;
  uint32_t qkey, rq_psn, sq_psn, dest_qp_num, qp_access_flags;
  uint32_t max_send_wr, max_recv_wr, max_send_sge, max_recv_sge, max_inline_data;
  AhAttr ah_attr, alt_ah_attr;
  uint16_t pkey_index, alt_pkey_index;
  uint8_t en_sqd_async_notify, max_rd_atomic, max_dest_rd_atomic;
  uint8_t min_rnr_timer, port_num, timeout, retry_cnt, rnr_retry;
  uint8_t alt_port_num, alt_timeout;
  uint32_t rate_limit;
};
struct Api {
  ContextInfo (*context_info)(void*);
  int (*query_port)(void*, uint8_t, PortAttr*);
  void* (*alloc_parent_domain)(void*, void*);
  void* (*create_cq_ex)(void*, void*);
  void* (*create_qp_ex)(void*, void*);
  int (*modify_qp)(void*, const QpAttr*, int);
  void* (*create_ah)(void*, const AhAttr*);
};
__attribute__((visibility("hidden"))) const Api* init_mlx5_api(void* verbs_library);
__attribute__((visibility("hidden"))) const Api* init_shca_api(void* verbs_library);
} // namespace rocshmem::verbs_adapter
