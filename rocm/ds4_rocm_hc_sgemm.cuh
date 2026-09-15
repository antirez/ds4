#pragma once
/* HC-only F32 SGEMM plan. The graph lends full heads storage on stream0.
 * This plan owns no device allocation and never changes the global handle. */
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <new>

static hipError_t v41_hc_widen(float *, const uint16_t *, uint64_t);

static constexpr uint64_t v41_hc_wide_bytes = UINT64_C(20480) * 24u * 4u;
static constexpr uint64_t v41_hc_workspace_cap = UINT64_C(64) * 1024u * 1024u;
static constexpr uint64_t v41_hc_scratch_bytes = v41_hc_wide_bytes + v41_hc_workspace_cap;
struct ds4_gpu_dsv41_hc_plan {
    rocblas_handle handle = nullptr;
    void *scratch = nullptr;
    size_t workspace_bytes = 0;
    bool pending = false, failed = false, unsupported = false;
};
static void v41_hc_drain(ds4_gpu_dsv41_hc_plan *p) {
    if (!p || !p->pending) return;
    const hipError_t status = hipStreamSynchronize(nullptr);
    if (status != hipSuccess) {
        std::fprintf(stderr, "ds4: HC SGEMM cannot prove stream drained: %s\n", hipGetErrorString(status));
        std::abort();
    }
    p->pending = false;
}
static int v41_hc_error(ds4_gpu_dsv41_hc_plan *p, const char *what, int status) {
    std::fprintf(stderr, "ds4: HC SGEMM %s failed: %d\n", what, status);
    p->failed = true;
    p->pending = true;
    v41_hc_drain(p);
    return -1;
}
static void v41_hc_plan_destroy(ds4_gpu_dsv41_hc_plan *p) {
    if (!p) return;
    v41_hc_drain(p);
    if (p->handle) {
        const rocblas_status status = rocblas_destroy_handle(p->handle);
        if (status != rocblas_status_success) {
            std::fprintf(stderr, "ds4: HC SGEMM handle teardown failed: %d\n", int(status));
            std::abort();
        }
    }
    delete p;
}
static rocblas_status v41_hc_submit(ds4_gpu_dsv41_hc_plan *p,
        float *out, const float *wide, const float *input) {
    const float alpha = 1.0f, beta = 0.0f;
    return rocblas_sgemm(p->handle, rocblas_operation_transpose, rocblas_operation_none,
        24, 2048, 20480, &alpha, wide, 20480, input, 20480, &beta, out, 24);
}
/*1 enqueued,0 unsupported before projection submission,-1 failed and drained.
 * Inputs are the existing RMS-normalized F32 values; no precision boundary is added. */
static int v41_hc_plan_run(ds4_gpu_dsv41_hc_plan **owner,
        float *out, const uint16_t *weight, const float *input, void *scratch) {
    if (!*owner) *owner = new (std::nothrow) ds4_gpu_dsv41_hc_plan;
    ds4_gpu_dsv41_hc_plan *p = *owner;
    if (!p || p->failed) return -1;
    if (p->unsupported) return 0;
#define V41_HC_BLAS(call) do { const rocblas_status st_ = (call); if (st_ != rocblas_status_success) return v41_hc_error(p, #call, int(st_)); } while (0)
    if (!p->handle) {
        V41_HC_BLAS(rocblas_create_handle(&p->handle));
        V41_HC_BLAS(rocblas_set_stream(p->handle, nullptr));
        V41_HC_BLAS(rocblas_set_pointer_mode(p->handle, rocblas_pointer_mode_host));
        V41_HC_BLAS(rocblas_set_atomics_mode(p->handle, rocblas_atomics_not_allowed));
        V41_HC_BLAS(rocblas_set_math_mode(p->handle, rocblas_default_math));
    }
    if (p->scratch != scratch) {
        v41_hc_drain(p);
        float *wide = static_cast<float *>(scratch);
        void *workspace = static_cast<unsigned char *>(scratch) + v41_hc_wide_bytes;
        V41_HC_BLAS(rocblas_set_workspace(p->handle, workspace, 256));
        V41_HC_BLAS(rocblas_start_device_memory_size_query(p->handle));
        const rocblas_status query = v41_hc_submit(p, out, wide, input);
        size_t required = 0;
        const rocblas_status stop = rocblas_stop_device_memory_size_query(p->handle, &required);
        if (query != rocblas_status_success && query != rocblas_status_size_increased && query != rocblas_status_size_unchanged)
            return v41_hc_error(p, "workspace query", int(query));
        if (stop != rocblas_status_success) return v41_hc_error(p, "workspace query stop", int(stop));
        if (required > v41_hc_workspace_cap) { p->unsupported = true; return 0; }
        const size_t capacity = required ? required : 256;
        V41_HC_BLAS(rocblas_set_workspace(p->handle, workspace, capacity));
        rocblas_math_mode math; rocblas_atomics_mode atomics;
        rocblas_pointer_mode pointer; hipStream_t stream; size_t actual = 0;
        V41_HC_BLAS(rocblas_get_math_mode(p->handle, &math));
        V41_HC_BLAS(rocblas_get_atomics_mode(p->handle, &atomics));
        V41_HC_BLAS(rocblas_get_pointer_mode(p->handle, &pointer));
        V41_HC_BLAS(rocblas_get_stream(p->handle, &stream));
        V41_HC_BLAS(rocblas_get_device_memory_size(p->handle, &actual));
        if (math != rocblas_default_math || atomics != rocblas_atomics_not_allowed ||
            pointer != rocblas_pointer_mode_host || stream != nullptr || actual != capacity ||
            rocblas_is_managing_device_memory(p->handle)) return v41_hc_error(p, "precision/workspace contract", -1);
        p->scratch = scratch; p->workspace_bytes = capacity;
    }
    p->pending = true;
    const hipError_t converted = v41_hc_widen(static_cast<float *>(scratch), weight, UINT64_C(20480) * 24u);
    if (converted != hipSuccess) return v41_hc_error(p, "weight widening", int(converted));
    V41_HC_BLAS(v41_hc_submit(p, out, static_cast<float *>(scratch), input));
#undef V41_HC_BLAS
    return 1;
}
