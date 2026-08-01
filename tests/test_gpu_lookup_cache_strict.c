/* Unit test for ds4_gpu_lookup_cache_strict (mgpu-graph-session-execution).
 *
 * Strict variant differs from ds4_gpu_lookup_cache:
 *   - returns 1 ONLY for entries whose device_id == expected_device
 *   - NO host-pointer fallback on miss
 *   - NO different-device fallback
 *
 * The test installs distinct cache entries on devices 0 and 1, then
 * exercises:
 *   1. exact-device match on each
 *   2. wrong-device match (e.g. asking for dev 2 against dev-0/1 entries)
 *      returns 0
 *   3. uncached offset returns 0 (no host-pointer fallback)
 *   4. interior-offset arithmetic still works
 *
 * Requires >= 2 CUDA devices; skips with PASS otherwise.
 */

#include "ds4_gpu.h"
#include "ds4_gpu_mgpu.h"

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(cond, msg)                                                \
    do {                                                                \
        if (!(cond)) {                                                  \
            fprintf(stderr, "FAIL: %s (line %d)\n", (msg), __LINE__);   \
            return 1;                                                   \
        }                                                               \
    } while (0)

int main(void) {
    int dev_count = 0;
    (void)cudaGetDeviceCount(&dev_count);
    fprintf(stderr,
            "test_gpu_lookup_cache_strict: %d CUDA devices visible\n",
            dev_count);
    if (dev_count < 2) {
        fprintf(stderr, "  skipping (need >= 2 devices)\n");
        return 0;
    }

    /* Initialize multi-GPU context for 2 devices. */
    ds4_gpu_config cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.n_gpus = 2;
    cfg.device_indices[0] = 0;
    cfg.device_indices[1] = 1;
    cfg.vram_bytes[0] = 256ull * 1024ull * 1024ull;
    cfg.vram_bytes[1] = 256ull * 1024ull * 1024ull;
    cfg.safety_margin_bytes = 0;
    CHECK(ds4_gpu_init_multi(&cfg) != 0, "ds4_gpu_init_multi");

    /* Synthetic 1 MiB model. */
    const size_t total = 1024 * 1024;
    void *host = NULL;
    CHECK(cudaMallocHost(&host, total) == cudaSuccess, "cudaMallocHost");
    unsigned char *bytes = (unsigned char *)host;
    for (size_t i = 0; i < total; i++) bytes[i] = (unsigned char)(i & 0xff);
    CHECK(ds4_gpu_set_model_map(host, total), "set_model_map");

    /* Install one range on each device.
     *
     * dev 0: [0, 256 KiB)
     * dev 1: [512 KiB, 512 KiB + 128 KiB) */
    ds4_tensor_range r0 = { 0, 256ull * 1024ull, 0 };
    CHECK(ds4_gpu_device_cache_tensors(0, &r0, 1) == 0, "cache dev 0");
    ds4_tensor_range r1 = { 512ull * 1024ull, 128ull * 1024ull, 1 };
    CHECK(ds4_gpu_device_cache_tensors(1, &r1, 1) == 0, "cache dev 1");

    /* 1. Strict lookup with the correct physical device id succeeds. */
    void *p0 = NULL;
    CHECK(ds4_gpu_lookup_cache_strict(0, 1024, 0, &p0) == 1,
          "strict lookup dev 0 base");
    CHECK(p0 != NULL, "strict lookup dev 0 ptr non-null");

    void *p0_interior = NULL;
    CHECK(ds4_gpu_lookup_cache_strict(100, 1024, 0, &p0_interior) == 1,
          "strict lookup dev 0 interior");
    CHECK(p0_interior == (char *)p0 + 100,
          "strict lookup dev 0 interior offset arithmetic");

    void *p1 = NULL;
    CHECK(ds4_gpu_lookup_cache_strict(512ull * 1024ull, 1024, 1, &p1) == 1,
          "strict lookup dev 1 base");
    CHECK(p1 != NULL, "strict lookup dev 1 ptr non-null");

    /* 2. Wrong-device lookup must return 0 even though a covering entry
     *    exists on a different device. */
    void *not_used = NULL;
    CHECK(ds4_gpu_lookup_cache_strict(0, 1024, 1, &not_used) == 0,
          "strict lookup with wrong dev (entry on 0, asking for 1) returns 0");
    CHECK(ds4_gpu_lookup_cache_strict(512ull * 1024ull, 1024, 0, &not_used) == 0,
          "strict lookup with wrong dev (entry on 1, asking for 0) returns 0");

    /* Phantom physical device id beyond what we cached. */
    CHECK(ds4_gpu_lookup_cache_strict(0, 1024, 7, &not_used) == 0,
          "strict lookup with unknown dev id returns 0");

    /* 3. Uncached offset (the gap between r0 and r1) must return 0.
     *    Critically: no host-pointer fallback, no FD-cache fallback. */
    CHECK(ds4_gpu_lookup_cache_strict(300ull * 1024ull, 1024, 0, &not_used) == 0,
          "strict lookup at uncached offset (gap) returns 0 — no host fallback");
    CHECK(ds4_gpu_lookup_cache_strict(300ull * 1024ull, 1024, 1, &not_used) == 0,
          "strict lookup at uncached offset (gap, dev 1) returns 0");

    /* 4. Out-of-range query (offset beyond model) returns 0. */
    CHECK(ds4_gpu_lookup_cache_strict(total + 1, 16, 0, &not_used) == 0,
          "strict lookup beyond model returns 0");

    /* 5. Overflow-safe: bytes = UINT64_MAX must not wrap into a hit. */
    CHECK(ds4_gpu_lookup_cache_strict(100, UINT64_MAX, 0, &not_used) == 0,
          "strict lookup with bytes=UINT64_MAX does not wrap into a hit");

    ds4_gpu_cleanup();
    (void)cudaFreeHost(host);
    fprintf(stderr, "test_gpu_lookup_cache_strict PASS (devs=%d)\n", dev_count);
    return 0;
}
