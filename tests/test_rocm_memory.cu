#include <cassert>
#include "ds4_rocm_memory.h"

int main(void) {
    assert(ds4_rocm_model_arena_bytes(64ull << 20) == (1792ull << 20));
    for (uint64_t mib : {256ull, 640ull, 896ull, 1024ull, 1152ull, 2049ull}) {
        assert(ds4_rocm_model_arena_bytes(mib << 20) == (mib << 20));
    }
    uint64_t available;
    assert(ds4_linux_nonmovable_memory(&available));
    assert(ds4_rocm_uses_host_ram());
    size_t free_b, total_b;
    assert(ds4_rocm_mem_get_info(&free_b, &total_b) == hipSuccess);
    printf("ROCm integrated device, usable %.2f GiB, GPU free %.2f GiB\n",
           (double)available / (1ull << 30), (double)free_b / (1ull << 30));
    void *ptr = nullptr;
    assert(ds4_rocm_malloc(&ptr, (size_t)available) == hipErrorOutOfMemory);
    assert(ptr == nullptr);
    assert(ds4_rocm_malloc_host(&ptr, (size_t)available) == hipErrorOutOfMemory);
    assert(ds4_rocm_malloc_managed(&ptr, (size_t)available) == hipErrorOutOfMemory);
    assert(ds4_rocm_malloc(&ptr, 1u << 20) == hipSuccess);
    assert(hipFree(ptr) == hipSuccess);
    puts("ROCm allocation admission: PASS");
    return 0;
}
