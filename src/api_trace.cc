/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *****************************************************************************/

#include "rocshmem/api_trace.h"
#include "rocshmem/rocshmem_common.hpp"
#include <cstddef>
#include <cstdint>
#include <limits>

#if defined(ROCSHMEM_ROCPROFILER_REGISTER)
#    include <rocprofiler-register/rocprofiler-register.h>
#    include "rocshmem/rocshmem_config.h"

#    define ROCP_REG_VERSION                                                             \
        ROCPROFILER_REGISTER_COMPUTE_VERSION_3(ROCSHMEM_VENDOR_MAJOR_VERSION,            \
                                               ROCSHMEM_VENDOR_MINOR_VERSION,            \
                                               ROCSHMEM_VENDOR_PATCH_VERSION)

ROCPROFILER_REGISTER_DEFINE_IMPORT(rocshmem, ROCP_REG_VERSION)
#endif

#include "context.hpp"
#include "log.hpp"
#include "rocshmem/rocshmem_COLL.hpp"

// Forward declarations for implementation functions with C linkage
extern "C" {
__host__ void rocshmem_barrier_all_on_stream_impl(hipStream_t stream);

__host__ void rocshmem_quiet_on_stream_impl(hipStream_t stream);

__host__ void rocshmem_sync_all_on_stream_impl(hipStream_t stream);

__host__ void rocshmem_alltoallmem_on_stream_impl(rocshmem_team_t team, void *dest,
                                                   const void *source, size_t size,
                                                   hipStream_t stream);

__host__ void rocshmem_broadcastmem_on_stream_impl(rocshmem_team_t team, void *dest,
                                                    const void *source, size_t nelems,
                                                    int pe_root, hipStream_t stream);

__host__ void rocshmem_getmem_on_stream_impl(void *dest, const void *source, size_t nelems,
                                              int pe, hipStream_t stream);

__host__ void rocshmem_putmem_on_stream_impl(void *dest, const void *source, size_t nelems,
                                              int pe, hipStream_t stream);

__host__ void rocshmem_putmem_signal_on_stream_impl(void *dest, const void *source,
                                                     size_t nelems, uint64_t *sig_addr,
                                                     uint64_t signal, int sig_op, int pe,
                                                     hipStream_t stream);

__host__ void rocshmem_signal_wait_until_on_stream_impl(uint64_t *sig_addr, int cmp,
                                                         uint64_t cmp_value,
                                                         hipStream_t stream);

__host__ int rocshmem_alltoallvmem_on_stream_impl(
    rocshmem_team_t, void*, const size_t[], const size_t[], void*,
    const size_t[], const size_t[], hipStream_t);
__host__ int rocshmem_reduce_scatter_on_stream_impl(
    rocshmem_team_t, void*, const void*, int, hipStream_t,
    rocshmem_datatype_t, rocshmem_reduce_op_t);
__host__ int rocshmem_fcollectmem_on_stream_impl(
    rocshmem_team_t, void*, const void*, size_t, hipStream_t);
}  // extern "C"

namespace rocshmem
{
extern rocshmem_ctx_t ROCSHMEM_HOST_CTX_DEFAULT;

namespace
{
constexpr size_t
compute_table_offset(size_t n)
{
    return (sizeof(uint64_t) + (n * sizeof(void*)));
}

constexpr size_t
compute_table_size(size_t nmembers)
{
    return (sizeof(uint64_t) + (nmembers * sizeof(void*)));
}

#define ROCSHMEM_ASSERT_OFFSET(TABLE, MEMBER, IDX)                                           \
    static_assert(offsetof(TABLE, MEMBER) == compute_table_offset(IDX),                  \
                  "Do not re-arrange the table members")


// DO NOT REORDER, ADD NEW ITEMS TO BOTTOM
ROCSHMEM_ASSERT_OFFSET(rocshmemApiFuncTable, barrier_all_on_stream_fn, 0);
ROCSHMEM_ASSERT_OFFSET(rocshmemApiFuncTable, quiet_on_stream_fn, 1);
ROCSHMEM_ASSERT_OFFSET(rocshmemApiFuncTable, sync_all_on_stream_fn, 2);
ROCSHMEM_ASSERT_OFFSET(rocshmemApiFuncTable, alltoallmem_on_stream_fn, 3);
ROCSHMEM_ASSERT_OFFSET(rocshmemApiFuncTable, broadcastmem_on_stream_fn, 4);
ROCSHMEM_ASSERT_OFFSET(rocshmemApiFuncTable, getmem_on_stream_fn, 5);
ROCSHMEM_ASSERT_OFFSET(rocshmemApiFuncTable, putmem_on_stream_fn, 6);
ROCSHMEM_ASSERT_OFFSET(rocshmemApiFuncTable, putmem_signal_on_stream_fn, 7);
ROCSHMEM_ASSERT_OFFSET(rocshmemApiFuncTable, signal_wait_until_on_stream_fn, 8);
ROCSHMEM_ASSERT_OFFSET(rocshmemApiFuncTable, alltoallvmem_on_stream_fn, 9);
ROCSHMEM_ASSERT_OFFSET(rocshmemApiFuncTable, reduce_scatter_on_stream_fn, 10);
ROCSHMEM_ASSERT_OFFSET(rocshmemApiFuncTable, fcollectmem_on_stream_fn, 11);

#undef ROCSHMEM_ASSERT_OFFSET

static_assert(sizeof(rocshmemApiFuncTable) == compute_table_size(12),
              "Update table major/step version and add a new offset assertion if this "
              "fails to compile");

// Use alignas to ensure proper alignment for placement new
alignas(rocshmemApiFuncTable) std::array<unsigned char, sizeof(rocshmemApiFuncTable)> m_buffer = {};

rocshmemApiFuncTable*
RocshmemGetFunctionTable_impl()
{
    static auto* tbl =
        new(m_buffer.data()) rocshmemApiFuncTable{
            sizeof(rocshmemApiFuncTable),
            &rocshmem_barrier_all_on_stream_impl,
            &rocshmem_quiet_on_stream_impl,
            &rocshmem_sync_all_on_stream_impl,
            &rocshmem_alltoallmem_on_stream_impl,
            &rocshmem_broadcastmem_on_stream_impl,
            &rocshmem_getmem_on_stream_impl,
            &rocshmem_putmem_on_stream_impl,
            &rocshmem_putmem_signal_on_stream_impl,
            &rocshmem_signal_wait_until_on_stream_impl,
            &rocshmem_alltoallvmem_on_stream_impl,
            &rocshmem_reduce_scatter_on_stream_impl,
            &rocshmem_fcollectmem_on_stream_impl
        };

#if defined(ROCSHMEM_ROCPROFILER_REGISTER) && ROCSHMEM_ROCPROFILER_REGISTER > 0
    std::array<void*, 1>                       table_array{ tbl };
    rocprofiler_register_library_indentifier_t lib_id =
        rocprofiler_register_library_indentifier_t{};
    rocprofiler_register_error_code_t rocp_reg_status =
        rocprofiler_register_library_api_table(
            "rocshmem", &ROCPROFILER_REGISTER_IMPORT_FUNC(rocshmem), ROCP_REG_VERSION,
            table_array.data(), table_array.size(), &lib_id);

    if(rocp_reg_status != ROCP_REG_SUCCESS && rocp_reg_status != ROCP_REG_NO_TOOLS) {
        fprintf(stderr, "[rocprofiler-sdk-rocshmem][%d] rocprofiler-register failed with error code %d : %s\n",
                getpid(), rocp_reg_status,
                rocprofiler_register_error_string(rocp_reg_status));
    }
#endif

    return tbl;
}

}  // end of anonymous namespace

}  // end of namespace rocshmem

extern "C" {
const rocshmemApiFuncTable*
RocshmemGetFunctionTable()
{
    static const auto* tbl = rocshmem::RocshmemGetFunctionTable_impl();
    return tbl;
}
}


// =============================================================================
// IMPLEMENTATION FUNCTIONS WITH ROCPROFILER TRACING
// =============================================================================

extern "C" {

__host__ void rocshmem_barrier_all_on_stream_impl(hipStream_t stream)
{
    LOG_API("host::barrier_all_on_stream ()");
    rocshmem::Context* ctx = reinterpret_cast<rocshmem::Context*>(rocshmem::ROCSHMEM_HOST_CTX_DEFAULT.ctx_opaque);
    ctx->barrier_all_on_stream(stream);
}

__host__ void rocshmem_quiet_on_stream_impl(hipStream_t stream)
{
    LOG_API("rocshmem_quiet_on_stream");
    rocshmem::Context* ctx = reinterpret_cast<rocshmem::Context*>(rocshmem::ROCSHMEM_HOST_CTX_DEFAULT.ctx_opaque);
    ctx->quiet_on_stream(stream);
}

__host__ void rocshmem_sync_all_on_stream_impl(hipStream_t stream)
{
    LOG_API("rocshmem_sync_all_on_stream");
    rocshmem::Context* ctx = reinterpret_cast<rocshmem::Context*>(rocshmem::ROCSHMEM_HOST_CTX_DEFAULT.ctx_opaque);
    ctx->sync_all_on_stream(stream);
}

__host__ void rocshmem_alltoallmem_on_stream_impl(rocshmem_team_t team, void *dest,
                                                   const void *source, size_t size,
                                                   hipStream_t stream)
{
     LOG_API("host::alltoallmem_on_stream (dest=%p, source=%p, size=%zd)", dest, source, size);
    rocshmem::Context* ctx = reinterpret_cast<rocshmem::Context*>(rocshmem::ROCSHMEM_HOST_CTX_DEFAULT.ctx_opaque);
    ctx->alltoallmem_on_stream(team, dest, source, size, stream);
}

__host__ int rocshmem_alltoallvmem_on_stream_impl(
    rocshmem_team_t team, void *dest, const size_t dn[], const size_t dd[],
    void *source, const size_t sn[], const size_t sd[], hipStream_t stream) {
    return rocshmem::rocshmem_ctx_char_alltoallv_on_stream(
        rocshmem::ROCSHMEM_HOST_CTX_DEFAULT, team, static_cast<char *>(dest),
        dn, dd, static_cast<char *>(source), sn, sd, stream);
}

__host__ int rocshmem_fcollectmem_on_stream_impl(
    rocshmem_team_t team, void *dest, const void *source, size_t size,
    hipStream_t stream) {
    if (size > static_cast<size_t>(std::numeric_limits<int>::max())) {
        return hipErrorInvalidValue;
    }
    return rocshmem::rocshmem_ctx_char_fcollect_on_stream(
        rocshmem::ROCSHMEM_HOST_CTX_DEFAULT, team, static_cast<char *>(dest),
        static_cast<const char *>(source), static_cast<int>(size), stream);
}

#define ROCSHMEM_DISPATCH_RS_OP(TNAME, TYPE, OPNAME, OPENUM)                \
    case ROCSHMEM_REDUCE_##OPENUM:                                              \
        return rocshmem::rocshmem_ctx_##TNAME##_##OPNAME##_reduce_scatter_on_stream(\
            rocshmem::ROCSHMEM_HOST_CTX_DEFAULT, team,                       \
            static_cast<TYPE*>(dest),                                        \
            static_cast<const TYPE*>(source), nreduce, stream)
#define ROCSHMEM_DISPATCH_RS_ARITH(TNAME, TYPE)                              \
    switch (op) {                                                            \
        ROCSHMEM_DISPATCH_RS_OP(TNAME, TYPE, sum, SUM);                      \
        ROCSHMEM_DISPATCH_RS_OP(TNAME, TYPE, min, MIN);                      \
        ROCSHMEM_DISPATCH_RS_OP(TNAME, TYPE, max, MAX);                      \
        ROCSHMEM_DISPATCH_RS_OP(TNAME, TYPE, prod, PROD);                    \
        default: return hipErrorInvalidValue;                                \
    }
#define ROCSHMEM_DISPATCH_RS_INT(TNAME, TYPE)                                \
    switch (op) {                                                            \
        ROCSHMEM_DISPATCH_RS_OP(TNAME, TYPE, sum, SUM);                      \
        ROCSHMEM_DISPATCH_RS_OP(TNAME, TYPE, min, MIN);                      \
        ROCSHMEM_DISPATCH_RS_OP(TNAME, TYPE, max, MAX);                      \
        ROCSHMEM_DISPATCH_RS_OP(TNAME, TYPE, prod, PROD);                    \
        ROCSHMEM_DISPATCH_RS_OP(TNAME, TYPE, or, OR);                        \
        ROCSHMEM_DISPATCH_RS_OP(TNAME, TYPE, and, AND);                      \
        ROCSHMEM_DISPATCH_RS_OP(TNAME, TYPE, xor, XOR);                      \
        default: return hipErrorInvalidValue;                                \
    }

__host__ int rocshmem_reduce_scatter_on_stream_impl(
    rocshmem_team_t team, void *dest, const void *source, int nreduce,
    hipStream_t stream, rocshmem_datatype_t datatype,
    rocshmem_reduce_op_t op) {
    switch (datatype) {
        case ROCSHMEM_DATATYPE_FLOAT: { ROCSHMEM_DISPATCH_RS_ARITH(float, float); }
        case ROCSHMEM_DATATYPE_DOUBLE: { ROCSHMEM_DISPATCH_RS_ARITH(double, double); }
        case ROCSHMEM_DATATYPE_SHORT: { ROCSHMEM_DISPATCH_RS_INT(short, short); }
        case ROCSHMEM_DATATYPE_INT: { ROCSHMEM_DISPATCH_RS_INT(int, int); }
        case ROCSHMEM_DATATYPE_LONG: { ROCSHMEM_DISPATCH_RS_INT(long, long); }
        case ROCSHMEM_DATATYPE_LONGLONG: { ROCSHMEM_DISPATCH_RS_INT(longlong, long long); }
        default: return hipErrorInvalidValue;
    }
}

#undef ROCSHMEM_DISPATCH_RS_INT
#undef ROCSHMEM_DISPATCH_RS_ARITH
#undef ROCSHMEM_DISPATCH_RS_OP

__host__ void rocshmem_broadcastmem_on_stream_impl(rocshmem_team_t team, void *dest,
                                                    const void *source, size_t nelems,
                                                    int pe_root, hipStream_t stream)
{
    LOG_API("host::broadcastmem_on_stream (dest=%p, source=%p, nelems=%zd, pe_root=%d)", dest, source, nelems, pe_root);
    rocshmem::Context* ctx = reinterpret_cast<rocshmem::Context*>(rocshmem::ROCSHMEM_HOST_CTX_DEFAULT.ctx_opaque);
    ctx->broadcastmem_on_stream(team, dest, source, nelems, pe_root, stream);
}

__host__ void rocshmem_getmem_on_stream_impl(void *dest, const void *source, size_t nelems,
                                              int pe, hipStream_t stream)
{
    LOG_API("host::getmem_on_stream (dest=%p, source=%p, nelems=%zd, pe=%d)", dest, source, nelems, pe);
    rocshmem::Context* ctx = reinterpret_cast<rocshmem::Context*>(rocshmem::ROCSHMEM_HOST_CTX_DEFAULT.ctx_opaque);
    ctx->getmem_on_stream(dest, source, nelems, pe, stream);
}

__host__ void rocshmem_putmem_on_stream_impl(void *dest, const void *source, size_t nelems,
                                              int pe, hipStream_t stream)
{
    LOG_API("host::putmem_on_stream (dest=%p, source=%p, nelems=%zd, pe=%d)", dest, source, nelems, pe);
    rocshmem::Context* ctx = reinterpret_cast<rocshmem::Context*>(rocshmem::ROCSHMEM_HOST_CTX_DEFAULT.ctx_opaque);
    ctx->putmem_on_stream(dest, source, nelems, pe, stream);
}

__host__ void rocshmem_putmem_signal_on_stream_impl(void *dest, const void *source,
                                                     size_t nelems, uint64_t *sig_addr,
                                                     uint64_t signal, int sig_op, int pe,
                                                     hipStream_t stream)
{
    LOG_API("host::putmem_signal_on_stream (dest=%p, source=%p, nelems=%zd, pe=%d)", dest, source, nelems, pe);
    rocshmem::Context* ctx = reinterpret_cast<rocshmem::Context*>(rocshmem::ROCSHMEM_HOST_CTX_DEFAULT.ctx_opaque);
    ctx->putmem_signal_on_stream(dest, source, nelems, sig_addr, signal, sig_op, pe, stream);
}

__host__ void rocshmem_signal_wait_until_on_stream_impl(uint64_t *sig_addr, int cmp,
                                                         uint64_t cmp_value,
                                                         hipStream_t stream)
{
    LOG_API("host::signal_wait_until_on_stream (sig_addr=%p, cmp=%d)", sig_addr, cmp);
    rocshmem::Context* ctx = reinterpret_cast<rocshmem::Context*>(rocshmem::ROCSHMEM_HOST_CTX_DEFAULT.ctx_opaque);
    ctx->signal_wait_until_on_stream(sig_addr, cmp, cmp_value, stream);
}

}  // extern "C"
