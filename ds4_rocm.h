#pragma once

#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <hip/hip_fp16.h>
#include <rocwmma/rocwmma-version.hpp>

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

#define cudaMemAdvise(p1, p2, p3, p4) hipMemAdvise(p1, p2, p3, p4.id)
#define cudaMemPrefetchAsync(devPtr, count, location, flags, stream) hipMemPrefetchAsync(devPtr, count, location.id, stream)
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
#define CUDA_R_16F HIPBLAS_R_16F
#define CUDA_R_32F HIPBLAS_R_32F

#define cublasCreate hipblasCreate
#define cublasDestroy hipblasDestroy
#define cublasSetMathMode hipblasSetMathMode
#define cublasSgemm hipblasSgemm
#define cublasSgemmStridedBatched hipblasSgemmStridedBatched
#define cublasGemmEx hipblasGemmEx
#define cublasGemmStridedBatchedEx hipblasGemmStridedBatchedEx

static __device__ __forceinline__ int32_t __vcmpne4(uint32_t a, uint32_t b) {
    // For each byte: 0xFF if a != b, 0x00 if a == b
    uint32_t diff = a ^ b;
    // Spread any set bit in each byte to fill the whole byte
    diff |= (diff >> 1); diff |= (diff >> 2); diff |= (diff >> 4);
    diff &= 0x01010101u;
    diff *= 0xFFu; // 0x01 -> 0xFF per byte
    return (int32_t)diff;
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

