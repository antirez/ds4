#include <hip/hip_runtime.h>

#include <stdint.h>
#include <stdlib.h>

static __device__ __forceinline__ unsigned char alias_pattern_byte(
        uint32_t direction,
        uint64_t generation,
        uint64_t message,
        uint64_t offset) {
    uint64_t value = offset * UINT64_C(0x9e3779b185ebca87);
    value ^= (uint64_t)direction * UINT64_C(0xd6e8feb86659fd93);
    value ^= generation * UINT64_C(0xe7037ed1a0b428db);
    value ^= (message + 1u) * UINT64_C(0xa0761d6478bd642f);
    value ^= value >> 29;
    value *= UINT64_C(0xbf58476d1ce4e5b9);
    value ^= value >> 32;
    return (unsigned char)value;
}

static __global__ void alias_fill_kernel(unsigned char *payload,
                                         uint64_t bytes,
                                         uint32_t direction,
                                         uint64_t generation,
                                         uint64_t message) {
    uint64_t offset = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (offset < bytes)
        payload[offset] = alias_pattern_byte(direction, generation, message,
                                             offset);
}

static __global__ void alias_verify_kernel(const unsigned char *payload,
                                           uint64_t bytes,
                                           uint32_t direction,
                                           uint64_t generation,
                                           uint64_t message,
                                           unsigned long long *first_bad) {
    uint64_t offset = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (offset < bytes &&
        payload[offset] != alias_pattern_byte(direction, generation, message,
                                              offset)) {
        atomicMin(first_bad, (unsigned long long)offset);
    }
}

extern "C" int ds4_test_nhi_gpu_alias_fill(void *device_ptr,
                                             uint64_t bytes,
                                             uint32_t direction,
                                             uint64_t generation,
                                             uint64_t message) {
    if (!device_ptr || bytes == 0) return 0;
    const unsigned int threads = 256u;
    const unsigned int blocks = (unsigned int)((bytes + threads - 1u) / threads);
    alias_fill_kernel<<<blocks, threads>>>((unsigned char *)device_ptr,
                                           bytes,
                                           direction,
                                           generation,
                                           message);
    return hipGetLastError() == hipSuccess &&
           hipDeviceSynchronize() == hipSuccess;
}

extern "C" int ds4_test_nhi_gpu_alias_verify(const void *device_ptr,
                                               uint64_t bytes,
                                               uint32_t direction,
                                               uint64_t generation,
                                               uint64_t message,
                                               uint64_t *first_bad) {
    if (first_bad) *first_bad = UINT64_MAX;
    if (!device_ptr || bytes == 0) return 0;

    unsigned long long *device_bad = NULL;
    if (hipMalloc((void **)&device_bad, sizeof(*device_bad)) != hipSuccess)
        return 0;
    int ok = hipMemset(device_bad, 0xff, sizeof(*device_bad)) == hipSuccess;
    if (ok) {
        const unsigned int threads = 256u;
        const unsigned int blocks =
            (unsigned int)((bytes + threads - 1u) / threads);
        alias_verify_kernel<<<blocks, threads>>>(
            (const unsigned char *)device_ptr,
            bytes,
            direction,
            generation,
            message,
            device_bad);
        ok = hipGetLastError() == hipSuccess &&
             hipDeviceSynchronize() == hipSuccess;
    }

    unsigned long long host_bad = UINT64_MAX;
    if (ok) {
        ok = hipMemcpy(&host_bad, device_bad, sizeof(host_bad),
                       hipMemcpyDeviceToHost) == hipSuccess;
    }
    if (hipFree(device_bad) != hipSuccess) ok = 0;
    if (first_bad) *first_bad = (uint64_t)host_bad;
    return ok && host_bad == UINT64_MAX;
}
