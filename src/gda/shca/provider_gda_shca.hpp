// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#ifndef LIBRARY_SRC_GDA_SHCA_GDA_PROVIDER_HPP_
#define LIBRARY_SRC_GDA_SHCA_GDA_PROVIDER_HPP_

extern "C" {
#include <infiniband/shca_dv.h>
}

typedef union shca_db_reg {
  uint64_t *ptr;
  uintptr_t uint;
} shca_db_reg_t;

struct shcadv_funcs_t {
  int (*init_obj)(struct shca_dv_obj *obj, uint64_t obj_type);
};

#endif  //LIBRARY_SRC_GDA_SHCA_GDA_PROVIDER_HPP_
