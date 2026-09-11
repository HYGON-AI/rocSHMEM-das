// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#ifndef LIBRARY_SRC_HOST_RCCL_HPP_
#define LIBRARY_SRC_HOST_RCCL_HPP_

#include "rocshmem/rocshmem_config.h"
#include "rocshmem/rocshmem.hpp"

#if defined(USE_RCCL)
#include <climits>
#include <rccl.h>
#include <type_traits>

namespace rocshmem {

class Team;
class Bootstrap;
class RcclCommContext;

enum class RcclStatus { SUCCESS, FALLBACK_SAFE, FATAL };

struct RcclFunctionTable {
  ncclResult_t (*get_version)(int*);
  const char* (*get_error_string)(ncclResult_t);
  const char* (*get_last_error)(ncclComm_t);
  ncclResult_t (*get_unique_id)(ncclUniqueId*);
  ncclResult_t (*comm_init_rank)(ncclComm_t*, int, ncclUniqueId, int);
  ncclResult_t (*comm_destroy)(ncclComm_t);
  ncclResult_t (*comm_abort)(ncclComm_t);
  ncclResult_t (*comm_get_async_error)(ncclComm_t, ncclResult_t*);
  ncclResult_t (*all_reduce)(const void*, void*, size_t, ncclDataType_t,
                             ncclRedOp_t, ncclComm_t, hipStream_t);
  ncclResult_t (*reduce_scatter)(const void*, void*, size_t, ncclDataType_t,
                                ncclRedOp_t, ncclComm_t, hipStream_t);
  ncclResult_t (*broadcast)(const void*, void*, size_t, ncclDataType_t, int,
                            ncclComm_t, hipStream_t);
  ncclResult_t (*all_gather)(const void*, void*, size_t, ncclDataType_t,
                             ncclComm_t, hipStream_t);
  ncclResult_t (*reduce)(const void*, void*, size_t, ncclDataType_t,
                         ncclRedOp_t, int, ncclComm_t, hipStream_t);
  ncclResult_t (*gather)(const void*, void*, size_t, ncclDataType_t, int,
                         ncclComm_t, hipStream_t);
  ncclResult_t (*scatter)(const void*, void*, size_t, ncclDataType_t, int,
                          ncclComm_t, hipStream_t);
  ncclResult_t (*all_to_all)(const void*, void*, size_t, ncclDataType_t,
                             ncclComm_t, hipStream_t);
  ncclResult_t (*all_to_all_v)(const void*, const size_t*, const size_t*,
                               void*, const size_t*, const size_t*,
                               ncclDataType_t, ncclComm_t, hipStream_t);
};

extern RcclFunctionTable rccl_ftable;

bool rccl_runtime_enabled();
bool rccl_eager_init_enabled();
bool should_use_rccl(size_t bytes);
RcclCommContext* rccl_comm_context_create();
void rccl_comm_context_destroy(RcclCommContext* context);
bool rccl_team_prepare(Team* team, Bootstrap* bootstrap = nullptr);
bool rccl_team_init(Team* team, Bootstrap* bootstrap = nullptr);
void rccl_team_destroy(Team* team);
bool rccl_mpi_comm_init(MPI_Comm mpi_comm, int rank, int size,
                        RcclCommContext* context);
bool rccl_check(ncclResult_t result, const char* operation);
bool rccl_broadcast(Team* team, Bootstrap* bootstrap, const void* source,
                    void* dest, size_t bytes, int root, hipStream_t stream,
                    bool blocking = false);
bool rccl_broadcast(RcclCommContext* context, const void* source, void* dest,
                    size_t bytes, int root, hipStream_t stream, bool blocking);
bool rccl_alltoall(Team* team, Bootstrap* bootstrap, const void* source,
                   void* dest, size_t bytes_per_pe, hipStream_t stream);
bool rccl_alltoallv(Team* team, Bootstrap* bootstrap, const void* source,
                    const size_t* send_counts, const size_t* send_displs,
                    void* dest, const size_t* recv_counts,
                    const size_t* recv_displs, ncclDataType_t type,
                    hipStream_t stream, bool blocking = false);
bool rccl_all_gather(Team* team, Bootstrap* bootstrap, const void* source,
                     void* dest, size_t send_count, ncclDataType_t type,
                     hipStream_t stream, bool blocking = false);
bool rccl_reduce(Team* team, Bootstrap* bootstrap, const void* source,
                 void* dest, size_t count, ncclDataType_t type,
                 ncclRedOp_t op, int root, hipStream_t stream,
                 bool blocking = false);
bool rccl_gather(Team* team, Bootstrap* bootstrap, const void* source,
                 void* dest, size_t send_count, ncclDataType_t type, int root,
                 hipStream_t stream, bool blocking = false);
bool rccl_scatter(Team* team, Bootstrap* bootstrap, const void* source,
                  void* dest, size_t recv_count, ncclDataType_t type, int root,
                  hipStream_t stream, bool blocking = false);
bool rccl_all_reduce(RcclCommContext* context, const void* source, void* dest,
                     size_t count, ncclDataType_t type, ncclRedOp_t op,
                     hipStream_t stream, bool blocking);
bool rccl_all_reduce(Team* team, Bootstrap* bootstrap, const void* source,
                     void* dest, size_t count, ncclDataType_t type,
                     ncclRedOp_t op, hipStream_t stream, bool blocking);
bool rccl_reduce_scatter(Team* team, Bootstrap* bootstrap, const void* source,
                         void* dest, size_t count, ncclDataType_t type,
                         ncclRedOp_t op, hipStream_t stream,
                         bool blocking = false);

template <ROCSHMEM_OP Op>
constexpr ncclRedOp_t to_rccl_reduce_op() {
  if constexpr (Op == ROCSHMEM_SUM) return ncclSum;
  if constexpr (Op == ROCSHMEM_PROD) return ncclProd;
  if constexpr (Op == ROCSHMEM_MIN) return ncclMin;
  if constexpr (Op == ROCSHMEM_MAX) return ncclMax;
  return ncclNumOps;
}

template <typename T>
constexpr ncclDataType_t to_rccl_data_type() {
  if constexpr (std::is_same_v<T, char>) {
    return CHAR_MIN == 0 ? ncclUint8 : ncclInt8;
  } else if constexpr (std::is_same_v<T, signed char>) {
    return ncclInt8;
  } else if constexpr (std::is_same_v<T, unsigned char>) {
    return ncclUint8;
  } else if constexpr (std::is_same_v<T, int>) {
    return ncclInt32;
  } else if constexpr (std::is_same_v<T, unsigned int>) {
    return ncclUint32;
  } else if constexpr (std::is_same_v<T, long> ||
                       std::is_same_v<T, long long>) {
    return ncclInt64;
  } else if constexpr (std::is_same_v<T, unsigned long> ||
                       std::is_same_v<T, unsigned long long>) {
    return ncclUint64;
  } else if constexpr (std::is_same_v<T, float>) {
    return ncclFloat32;
  } else if constexpr (std::is_same_v<T, double>) {
    return ncclFloat64;
  }
  return ncclNumTypes;
}

template <typename T, ROCSHMEM_OP Op>
bool rccl_all_reduce(Team* team, Bootstrap* bootstrap, const T* source,
                     T* dest, size_t count, hipStream_t stream,
                     bool blocking = false) {
  constexpr ncclDataType_t type = to_rccl_data_type<T>();
  constexpr ncclRedOp_t op = to_rccl_reduce_op<Op>();
  return type != ncclNumTypes && op != ncclNumOps &&
         rccl_all_reduce(team, bootstrap, source, dest, count, type, op,
                         stream, blocking);
}

template <typename T>
bool rccl_all_gather(Team* team, Bootstrap* bootstrap, const T* source,
                     T* dest, size_t send_count, hipStream_t stream,
                     bool blocking = false) {
  constexpr ncclDataType_t type = to_rccl_data_type<T>();
  return type != ncclNumTypes &&
         rccl_all_gather(team, bootstrap, source, dest, send_count, type,
                         stream, blocking);
}

template <typename T, ROCSHMEM_OP Op>
bool rccl_reduce(Team* team, Bootstrap* bootstrap, const T* source, T* dest,
                 size_t count, int root, hipStream_t stream,
                 bool blocking = false) {
  constexpr ncclDataType_t type = to_rccl_data_type<T>();
  constexpr ncclRedOp_t op = to_rccl_reduce_op<Op>();
  return type != ncclNumTypes && op != ncclNumOps &&
         rccl_reduce(team, bootstrap, source, dest, count, type, op, root,
                     stream, blocking);
}

template <typename T>
bool rccl_gather(Team* team, Bootstrap* bootstrap, const T* source, T* dest,
                 size_t send_count, int root, hipStream_t stream,
                 bool blocking = false) {
  constexpr ncclDataType_t type = to_rccl_data_type<T>();
  return type != ncclNumTypes &&
         rccl_gather(team, bootstrap, source, dest, send_count, type, root,
                     stream, blocking);
}

template <typename T>
bool rccl_scatter(Team* team, Bootstrap* bootstrap, const T* source, T* dest,
                  size_t recv_count, int root, hipStream_t stream,
                  bool blocking = false) {
  constexpr ncclDataType_t type = to_rccl_data_type<T>();
  return type != ncclNumTypes &&
         rccl_scatter(team, bootstrap, source, dest, recv_count, type, root,
                      stream, blocking);
}

template <typename T>
bool rccl_alltoallv(Team* team, Bootstrap* bootstrap, const T* source,
                    const size_t* send_counts, const size_t* send_displs,
                    T* dest, const size_t* recv_counts,
                    const size_t* recv_displs, hipStream_t stream,
                    bool blocking = false) {
  constexpr ncclDataType_t type = to_rccl_data_type<T>();
  return type != ncclNumTypes &&
         rccl_alltoallv(team, bootstrap, source, send_counts, send_displs,
                        dest, recv_counts, recv_displs, type, stream,
                        blocking);
}

template <typename T, ROCSHMEM_OP Op>
bool rccl_reduce_scatter(Team* team, Bootstrap* bootstrap, const T* source,
                         T* dest, size_t count, hipStream_t stream,
                         bool blocking = false) {
  constexpr ncclDataType_t type = to_rccl_data_type<T>();
  constexpr ncclRedOp_t op = to_rccl_reduce_op<Op>();
  return type != ncclNumTypes && op != ncclNumOps &&
         rccl_reduce_scatter(team, bootstrap, source, dest, count, type, op,
                             stream, blocking);
}

}  // namespace rocshmem
#endif  // USE_RCCL

#endif  // LIBRARY_SRC_HOST_RCCL_HPP_
