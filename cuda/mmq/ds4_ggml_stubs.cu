// SPDX-License-Identifier: MIT
// Implementations of the ggml-API stubs declared in ds4_ggml_stubs.h plus
// the bodies of ggml_backend_cuda_context / ggml_cuda_info /
// new_pool_for_device that the vendored common.cuh declares without
// defining.
//
// Phase 0: pool is plain cudaMallocAsync / cudaFreeAsync. Phase 4 swaps
// in ds4's existing cuda_tmp_alloc slab allocator.

#include "common.cuh"   // pulls in ds4_ggml_stubs.h via redirect headers

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>

// ----------------------------------------------------------------------------
// Device info singleton.
//
// Common.cuh declares `const ggml_cuda_device_info & ggml_cuda_info();` -
// we provide the body. The struct layout (common.cuh:1091) is:
//   { int device_count; cuda_device_info devices[GGML_CUDA_MAX_DEVICES];
//     std::array<float, GGML_CUDA_MAX_DEVICES> default_tensor_split; }
// where cuda_device_info has { cc, nsm, smpb, smpbo, integrated, vmm,
// vmm_granularity, total_vram, warp_size, supports_cooperative_launch }.
// ----------------------------------------------------------------------------

const ggml_cuda_device_info & ggml_cuda_info() {
    static ggml_cuda_device_info info;
    static std::once_flag once;
    std::call_once(once, []{
        int count = 0;
        cudaError_t err = cudaGetDeviceCount(&count);
        if (err != cudaSuccess) {
            fprintf(stderr, "ggml_cuda_info: cudaGetDeviceCount failed: %s\n", cudaGetErrorString(err));
            count = 0;
        }
        if (count > GGML_CUDA_MAX_DEVICES) count = GGML_CUDA_MAX_DEVICES;
        info.device_count = count;
        for (int i = 0; i < count; i++) {
            cudaDeviceProp p;
            CUDA_CHECK(cudaGetDeviceProperties(&p, i));
            info.devices[i].cc                          = p.major * 100 + p.minor * 10;
            info.devices[i].nsm                         = p.multiProcessorCount;
            info.devices[i].smpb                        = p.sharedMemPerBlock;
            info.devices[i].smpbo                       = p.sharedMemPerBlockOptin;
            info.devices[i].integrated                  = p.integrated != 0;
            info.devices[i].vmm                         = false;
            info.devices[i].vmm_granularity             = 0;
            info.devices[i].total_vram                  = p.totalGlobalMem;
            info.devices[i].warp_size                   = p.warpSize;
            info.devices[i].supports_cooperative_launch = p.cooperativeLaunch != 0;
        }
    });
    return info;
}

// ggml_cuda_get_device / ggml_cuda_set_device are declared (not defined) in
// common.cuh. We provide thin wrappers.

int ggml_cuda_get_device() {
    int dev = 0;
    cudaGetDevice(&dev);
    return dev;
}

void ggml_cuda_set_device(int device) {
    int cur = -1;
    cudaGetDevice(&cur);
    if (cur != device) {
        CUDA_CHECK(cudaSetDevice(device));
    }
}

int64_t ggml_time_us() {
    using clock = std::chrono::steady_clock;
    static const auto t0 = clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(clock::now() - t0).count();
}

// ----------------------------------------------------------------------------
// ggml_cuda_error: invoked by the CUDA_CHECK / CUBLAS_CHECK macros defined
// in common.cuh on the error path. Marked [[noreturn]] in the declaration
// (common.cuh:155) - abort() satisfies that contract.
// ----------------------------------------------------------------------------

[[noreturn]] void ggml_cuda_error(const char * stmt, const char * func, const char * file, int line, const char * msg) {
    fprintf(stderr, "CUDA error: %s\n  call: %s\n  in: %s at %s:%d\n", msg, stmt, func, file, line);
    fflush(stderr);
    abort();
}

// ----------------------------------------------------------------------------
// Concrete pool wrapping cudaMallocAsync / cudaFreeAsync.
// ----------------------------------------------------------------------------

namespace {

struct ds4_naive_pool : public ggml_cuda_pool {
    int device;

    explicit ds4_naive_pool(int device) : device(device) {}

    void * alloc(size_t size, size_t * actual_size) override {
        ggml_cuda_set_device(device);
        void * ptr = nullptr;
        CUDA_CHECK(cudaMallocAsync(&ptr, size, cudaStreamPerThread));
        if (actual_size) *actual_size = size;
        return ptr;
    }

    void free(void * ptr, size_t /*size*/) override {
        if (!ptr) return;
        ggml_cuda_set_device(device);
        CUDA_CHECK(cudaFreeAsync(ptr, cudaStreamPerThread));
    }
};

} // anonymous namespace

std::unique_ptr<ggml_cuda_pool> ggml_backend_cuda_context::new_pool_for_device(int device, int /*stream_no*/) {
    return std::unique_ptr<ggml_cuda_pool>(new ds4_naive_pool(device));
}

ggml_backend_cuda_context::~ggml_backend_cuda_context() {
    if (copy_event) {
        cudaEventDestroy(copy_event);
        copy_event = nullptr;
    }
    // streams[][], cublas_handles[], and pools[][] are owned-by-value
    // (cudaStream_t and cublasHandle_t are opaque handles - destroying the
    // context "should" tear them down, but in our shim ds4 manages streams
    // externally and we leave them alone. The pools auto-destruct via
    // unique_ptr.).
}
