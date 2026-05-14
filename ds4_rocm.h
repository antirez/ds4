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
#define cudaMallocHost hipHostMalloc
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

#define __shfl_sync(mask, var, laneMask, width) __shfl_sync(static_cast<uint64_t>(mask), var, laneMask, width)
#define __shfl_down_sync(mask, var, laneMask, width) __shfl_down_sync(static_cast<uint64_t>(mask), var, laneMask, width)
#define __shfl_up_sync(mask, var, laneMask, width) __shfl_up_sync(static_cast<uint64_t>(mask), var, laneMask, width)
#define __shfl_xor_sync(mask, var, laneMask, width) __shfl_xor_sync(static_cast<uint64_t>(mask), var, laneMask, width)


template<typename T1, typename T2, typename T3>
__forceinline__ decltype(auto) myHipFuncSetAttribute(T1&& p1, T2&& p2, T3&& p3) {
    return hipFuncSetAttribute(reinterpret_cast<const void*>(p1), std::forward<T2>(p2), std::forward<T3>(p3));
}
#define cudaFuncSetAttribute myHipFuncSetAttribute

template<typename T1, typename T2, typename T3, typename T4>
__forceinline__ decltype(auto) myHipMemAdvise(T1&& p1, T2&& p2, T3&& p3, T4&& p4) {
    return hipMemAdvise(std::forward<T1>(p1), std::forward<T2>(p2), std::forward<T3>(p3), p4.id);
}
#define cudaMemAdvise myHipMemAdvise

template<typename T1, typename T2, typename T3, typename T4, typename T5>
__forceinline__ decltype(auto) myHipMemPrefetchAsync(T1&& p1, T2&& p2, T3&& p3, T4&& /* p4 */, T5&& p5) {
    return hipMemPrefetchAsync(std::forward<T1>(p1), std::forward<T2>(p2), p3.id, std::forward<T5>(p5));
}
#define cudaMemPrefetchAsync myHipMemPrefetchAsync

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

static __device__ __forceinline__ int32_t __vsub4(int32_t a, int32_t b) {
    // Per-byte subtraction (wrapping, not saturating)
    uint32_t ua = (uint32_t)a, ub = (uint32_t)b;
    // Trick: subtract bytes in parallel avoiding cross-byte borrows
    uint32_t diff = ((ua | 0x80808080u) - (ub & 0x7F7F7F7Fu)) ^ ((ua ^ ~ub) & 0x80808080u);
    return (int32_t)diff;
}

// __dp4a: dot product of 4 signed int8s packed in an int32
static __device__ __forceinline__ int32_t __dp4a(int32_t a, int32_t b, int32_t c) {
    const int8_t *a_bytes = reinterpret_cast<const int8_t*>(&a);
    const int8_t *b_bytes = reinterpret_cast<const int8_t*>(&b);
    return c + (int32_t)a_bytes[0] * b_bytes[0]
             + (int32_t)a_bytes[1] * b_bytes[1]
             + (int32_t)a_bytes[2] * b_bytes[2]
             + (int32_t)a_bytes[3] * b_bytes[3];
}

static __device__ __forceinline__ uint32_t __dp4a(uint32_t a, uint32_t b, uint32_t c) {
    const uint8_t *a_bytes = reinterpret_cast<const uint8_t*>(&a);
    const uint8_t *b_bytes = reinterpret_cast<const uint8_t*>(&b);
    return c + (uint32_t)a_bytes[0] * b_bytes[0]
             + (uint32_t)a_bytes[1] * b_bytes[1]
             + (uint32_t)a_bytes[2] * b_bytes[2]
             + (uint32_t)a_bytes[3] * b_bytes[3];
}

