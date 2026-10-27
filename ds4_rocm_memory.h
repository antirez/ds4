#ifndef DS4_ROCM_MEMORY_H
#define DS4_ROCM_MEMORY_H

#include <hip/hip_runtime.h>
#include "ds4_linux_memory.h"

static inline uint64_t ds4_rocm_model_arena_bytes(uint64_t need) {
    /* Large tensor spans should not strand hundreds of MiB each. Small
     * tensors still share an arena to avoid thousands of driver allocations. */
    return need >= (256ull << 20) ? need : (1792ull << 20);
}

static inline bool ds4_rocm_uses_host_ram(void) {
    int device;
    hipDeviceProp_t prop;
    return hipGetDevice(&device) == hipSuccess &&
           hipGetDeviceProperties(&prop, device) == hipSuccess && prop.integrated;
}

static inline bool ds4_rocm_allocation_fits(size_t bytes, bool host = false) {
    if (!host && !ds4_rocm_uses_host_ram()) return true;
    uint64_t available;
    const uint64_t reserve = 2ull << 30;
    if (!ds4_linux_nonmovable_memory(&available)) {
        fprintf(stderr, "ds4: ROCm cannot read usable host memory; allocation refused\n");
        return false;
    }
    if (available >= reserve && bytes <= available - reserve) return true;
    fprintf(stderr, "ds4: ROCm allocation refused: %.2f MiB requested, "
            "%.2f GiB usable after excluding CMA, 2 GiB OS reserve\n",
            (double)bytes / 1048576.0, (double)available / 1073741824.0);
    return false;
}

template <typename T>
static inline hipError_t ds4_rocm_malloc(T **ptr, size_t bytes) {
    if (!ds4_rocm_allocation_fits(bytes)) return hipErrorOutOfMemory;
    return hipMalloc((void **)ptr, bytes);
}

template <typename T>
static inline hipError_t ds4_rocm_malloc_host(T **ptr, size_t bytes) {
    if (!ds4_rocm_allocation_fits(bytes, true)) return hipErrorOutOfMemory;
    return hipHostMalloc((void **)ptr, bytes);
}

template <typename T>
static inline hipError_t ds4_rocm_malloc_managed(T **ptr, size_t bytes) {
    if (!ds4_rocm_allocation_fits(bytes, true)) return hipErrorOutOfMemory;
    return hipMallocManaged((void **)ptr, bytes);
}

static inline hipError_t ds4_rocm_mem_get_info(size_t *free_b, size_t *total_b) {
    hipError_t err = hipMemGetInfo(free_b, total_b);
    if (err != hipSuccess || !ds4_rocm_uses_host_ram()) return err;
    uint64_t available;
    if (!ds4_linux_nonmovable_memory(&available)) return hipErrorUnknown;
    if (available < *free_b) *free_b = (size_t)available;
    return hipSuccess;
}

#endif
