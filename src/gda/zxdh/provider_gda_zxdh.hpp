// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#ifndef LIBRARY_SRC_GDA_ZXDH_GDA_PROVIDER_HPP_
#define LIBRARY_SRC_GDA_ZXDH_GDA_PROVIDER_HPP_

#include "gda/zxdh/zxdh_dv.h"

inline constexpr uint8_t ROCSHMEM_ZXDH_OP_WRITE = 0x04;
inline constexpr uint8_t ROCSHMEM_ZXDH_OP_READ = 0x06;

struct zxdh_dv_funcs_t {
  int (*get_cq_init_size)(struct ibv_context *,
                          struct zxdh_devx_cq_init_attr *,
                          struct zxdh_devx_init_cq_resp *);
  int (*get_qp_init_size)(struct ibv_context *,
                          struct zxdh_devx_qp_init_attr *,
                          struct zxdh_devx_init_qp_resp *);
  struct zxdh_devx_cq *(*create_cq)(struct ibv_context *,
                                    int,
                                    struct ibv_comp_channel *,
                                    int,
                                    struct zxdh_devx_cq_buffers *);
  struct zxdh_devx_qp *(*create_qp)(struct ibv_pd *,
                                    struct ibv_qp_init_attr *,
                                    struct zxdh_devx_qp_buffers *);
  int (*modify_qp)(struct zxdh_devx_qp *, struct ibv_qp_attr *, int);
  int (*destroy_qp)(struct zxdh_devx_qp *);
  int (*destroy_cq)(struct zxdh_devx_cq *);
};

#endif  // LIBRARY_SRC_GDA_ZXDH_GDA_PROVIDER_HPP_
