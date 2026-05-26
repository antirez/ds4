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

