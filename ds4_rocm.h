#pragma once

#include <hip/hip_runtime.h>

#include <rocwmma/rocwmma.hpp>

#include <hipblas/hipblas.h>
#include <hip/hip_fp16.h>

#include <hipcub/block/block_radix_sort.hpp>

namespace cub = hipcub;
namespace wmma = rocwmma;

#define cudaError_t hipError_t
#define cudaStream_t hipStream_t
#define cudaEvent_t hipEvent_t
#define cudaDeviceProp hipDeviceProp_t
#define cudaMemLocation hipMemLocation

#define cudaSuccess hipSuccess
#define cudaErrorNotSupported hipErrorNotSupported
#define cudaErrorInvalidValue hipErrorInvalidValue
#define cudaGetLastError hipGetLastError
#define cudaGetErrorString hipGetErrorString

#define cudaGetDevice hipGetDevice
#define cudaSetDevice hipSetDevice
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaDeviceGetAttribute hipDeviceGetAttribute
#define cudaGetDeviceProperties hipGetDeviceProperties
#define cudaDevAttrPageableMemoryAccess hipDeviceAttributePageableMemoryAccess
#define cudaMemLocationTypeDevice hipMemLocationTypeDevice

#define cudaMalloc hipMalloc
#define cudaMallocHost(p1, p2) hipHostMalloc(p1, p2, hipHostMallocDefault)
#define cudaMallocManaged hipMallocManaged
#define cudaFree hipFree
#define cudaFreeHost hipFreeHost
#define cudaMemset hipMemset
#define cudaMemcpy hipMemcpy
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define cudaMemGetInfo hipMemGetInfo
#define cudaMemsetAsync hipMemsetAsync

#define cudaHostRegister hipHostRegister
#define cudaHostUnregister hipHostUnregister
#define cudaHostGetDevicePointer hipHostGetDevicePointer
#define cudaHostRegisterMapped hipHostRegisterMapped
#define cudaHostRegisterReadOnly hipHostRegisterReadOnly

#define cudaMemAdviseSetReadMostly hipMemAdviseSetReadMostly
#define cudaMemAdviseSetPreferredLocation hipMemAdviseSetPreferredLocation

#define cudaStreamCreateWithFlags hipStreamCreateWithFlags
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaStreamDestroy hipStreamDestroy
#define cudaStreamNonBlocking hipStreamNonBlocking

#define cudaEventCreate hipEventCreate
#define cudaEventCreateWithFlags hipEventCreateWithFlags
#define cudaEventDestroy hipEventDestroy
#define cudaEventRecord hipEventRecord
#define cudaEventSynchronize hipEventSynchronize
#define cudaEventElapsedTime hipEventElapsedTime
#define cudaEventDisableTiming hipEventDisableTiming

#define cudaDevAttrMaxSharedMemoryPerBlockOptin hipDeviceAttributeSharedMemPerBlockOptin
#define cudaFuncAttributeMaxDynamicSharedMemorySize hipFuncAttributeMaxDynamicSharedMemorySize

#define cublasHandle_t hipblasHandle_t
#define cublasStatus_t hipblasStatus_t
#define cublasMath_t hipblasMath_t

#define CUBLAS_STATUS_SUCCESS HIPBLAS_STATUS_SUCCESS
#define CUBLAS_OP_N HIPBLAS_OP_N
#define CUBLAS_OP_T HIPBLAS_OP_T
#define CUBLAS_GEMM_DEFAULT HIPBLAS_GEMM_DEFAULT
#define CUBLAS_DEFAULT_MATH HIPBLAS_DEFAULT_MATH
#define CUBLAS_COMPUTE_32F HIPBLAS_COMPUTE_32F
#define CUBLAS_TF32_TENSOR_OP_MATH HIPBLAS_TF32_TENSOR_OP_MATH

#define CUDA_R_16F HIP_R_16F
#define CUDA_R_32F HIP_R_32F

#define cublasCreate hipblasCreate
#define cublasDestroy hipblasDestroy
#define cublasSetMathMode hipblasSetMathMode
#define cublasSgemm hipblasSgemm
#define cublasSgemmStridedBatched hipblasSgemmStridedBatched
#define cublasGemmEx hipblasGemmEx
#define cublasGemmStridedBatchedEx hipblasGemmStridedBatchedEx


template<typename T1, typename T2, typename T3>
__forceinline__ decltype(auto) ds4_hipFuncSetAttribute(T1&& p1, T2&& p2, T3&& p3) {
    return hipFuncSetAttribute(reinterpret_cast<const void*>(p1), std::forward<T2>(p2), std::forward<T3>(p3));
}
#define cudaFuncSetAttribute ds4_hipFuncSetAttribute

template<typename T1, typename T2, typename T3, typename T4>
__forceinline__ decltype(auto) ds4_hipMemAdvise(T1&& p1, T2&& p2, T3&& p3, T4&& p4) {
    return hipMemAdvise(std::forward<T1>(p1), std::forward<T2>(p2), std::forward<T3>(p3), p4.id);
}
#define cudaMemAdvise ds4_hipMemAdvise

template<typename T1, typename T2, typename T3, typename T4, typename T5>
__forceinline__ decltype(auto) ds4_hipMemPrefetchAsync(T1&& p1, T2&& p2, T3&& p3, T4&& /* p4 */, T5&& p5) {
    return hipMemPrefetchAsync(std::forward<T1>(p1), std::forward<T2>(p2), p3.id, std::forward<T5>(p5));
}
#define cudaMemPrefetchAsync ds4_hipMemPrefetchAsync


static __device__ __forceinline__ __half ds4_float2half(float x) {
    if (__builtin_isnan(x)) return __ushort_as_half(0x7E00u);
    x = fminf(x, 65504.0f);
    x = fmaxf(x, -65504.0f);
    return __float2half_rn(x);  // explicit round-to-nearest-even
}
#define __float2half ds4_float2half

static __device__ __forceinline__ float ds4_fminf(float a, float b) {
    if (__builtin_isnan(a)) return b;
    if (__builtin_isnan(b)) return a;
    return fminf(a, b);
}
#define fminf ds4_fminf

static __device__ __forceinline__ float ds4_fmaxf(float a, float b) {
    if (__builtin_isnan(a)) return b;
    if (__builtin_isnan(b)) return a;
    return fmaxf(a, b);
}
#define fmaxf ds4_fmaxf

#define rsqrtf(x) (1.0f / sqrtf(x))

typedef int8_t int8x4_t __attribute__((ext_vector_type(4)));
typedef uint8_t uint8x4_t __attribute__((ext_vector_type(4)));

static __device__ __forceinline__ unsigned int __vcmpne4(unsigned int a, unsigned int b) {
    const uint8x4_t& va = reinterpret_cast<const uint8x4_t&>(a);
    const uint8x4_t& vb = reinterpret_cast<const uint8x4_t&>(b);
    unsigned int c;
    uint8x4_t& vc = reinterpret_cast<uint8x4_t&>(c);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        vc[i] = va[i] == vb[i] ? 0x00 : 0xff;
    }
    return c;
}

static __device__ __forceinline__ uint32_t __vsub4(uint32_t a, uint32_t b) {
    return ((a | 0x80808080u) - (b & 0x7F7F7F7Fu)) ^ ((a ^ ~b) & 0x80808080u);
}

static __device__ __forceinline__ int32_t __dp4a(int32_t a, int32_t b, int32_t c) {
    const int8_t *a_bytes = reinterpret_cast<const int8_t*>(&a);
    const int8_t *b_bytes = reinterpret_cast<const int8_t*>(&b);
    return c + (int32_t)a_bytes[0] * b_bytes[0]
             + (int32_t)a_bytes[1] * b_bytes[1]
             + (int32_t)a_bytes[2] * b_bytes[2]
             + (int32_t)a_bytes[3] * b_bytes[3];
}

__device__ static float warp_sum_f32(float v);
template <uint32_t ROWS_PER_BLOCK>
__global__ static void matmul_f16_pair_warp_kernel(
        float *out0,
        float *out1,
        const __half *w0,
        const __half *w1,
        const float *x,
        uint64_t in_dim,
        uint64_t out0_dim,
        uint64_t out1_dim) {

    const uint64_t row_base = (uint64_t)blockIdx.x * ROWS_PER_BLOCK;
    const uint32_t tid = threadIdx.x;
    const uint32_t warp = tid >> 5u;
    const uint32_t lane = tid & 31u;

    const uint64_t row = row_base + warp;
    const bool valid0 = row < out0_dim;
    const bool valid1 = row < out1_dim;
    if (!valid0 && !valid1) {
        return;
    }

    float sum0 = 0.0f;
    float sum1 = 0.0f;

    const __half *wr0 = valid0 ? w0 + row * in_dim : w0;
    const __half *wr1 = valid1 ? w1 + row * in_dim : w1;

    for (uint64_t i = lane; i < in_dim; i += 32u) {

        const float xv = x[i];
        if (valid0) sum0 += __half2float(wr0[i]) * xv;
        if (valid1) sum1 += __half2float(wr1[i]) * xv;
    }

    sum0 = warp_sum_f32(sum0);
    sum1 = warp_sum_f32(sum1);

    if (lane == 0) {
        if (valid0) out0[row] = sum0;
        if (valid1) out1[row] = sum1;
    }
}

