/* Public-API ring oracle shared by Metal, CUDA and ROCm. No model required.
 * Exact half-representable inputs isolate ownership/indexing from rounding. */
#include "ds4_gpu.h"
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

bool ds4_log_is_tty(FILE *fp) { (void)fp; return false; }

enum { GUARD = 32 };
static const float canary = 123456.0f;

static int check_buffer(const ds4_gpu_tensor *tensor, const float *expected,
                        float *actual, uint64_t count, const char *name,
                        uint32_t cap, uint32_t n, uint32_t pos) {
    if (!ds4_gpu_tensor_read(tensor, 0, actual, count * sizeof(float))) return 0;
    for (uint64_t i = 0; i < count; i++) {
        if (actual[i] != expected[i]) {
            fprintf(stderr, "%s cap=%u N=%u pos=%u index=%llu: %g != %g\n",
                    name, cap, n, pos, (unsigned long long)i,
                    actual[i], expected[i]);
            return 0;
        }
    }
    return 1;
}

static int run_case(uint32_t cap, uint32_t n, uint32_t pos, uint32_t dim) {
    const uint64_t raw_count = (uint64_t)cap * dim + 2u * GUARD;
    const uint64_t kv_count = (uint64_t)n * dim + 2u * GUARD;
    const uint64_t max_count = raw_count > kv_count ? raw_count : kv_count;
    float *reference = (float *)malloc(raw_count * sizeof(float));
    float *source = (float *)malloc(kv_count * sizeof(float));
    float *actual = (float *)malloc(max_count * sizeof(float));
    ds4_gpu_tensor *raw_storage = NULL, *kv_storage = NULL;
    ds4_gpu_tensor *raw = NULL, *kv = NULL, *one = NULL;
    int ok = reference && source && actual;
    if (!ok) goto cleanup;
    for (uint64_t i = 0; i < raw_count; i++) reference[i] = canary;
    for (uint64_t i = 0; i < kv_count; i++) source[i] = canary;
    for (uint32_t t = 0; t < n; t++) {
        for (uint32_t d = 0; d < dim; d++) {
            source[GUARD + (uint64_t)t * dim + d] =
                ((int)(((uint64_t)t * 17u + d * 23u) % 4001u) - 2000) / 8.0f;
        }
    }
    raw_storage = ds4_gpu_tensor_alloc(raw_count * sizeof(float));
    kv_storage = ds4_gpu_tensor_alloc(kv_count * sizeof(float));
    ok = raw_storage && kv_storage;
    if (!ok) goto cleanup;
    raw = ds4_gpu_tensor_view(raw_storage, GUARD * sizeof(float),
                              (uint64_t)cap * dim * sizeof(float));
    kv = ds4_gpu_tensor_view(kv_storage, GUARD * sizeof(float),
                             (uint64_t)n * dim * sizeof(float));
    ok = raw && kv &&
         ds4_gpu_tensor_write(raw_storage, 0, reference, raw_count * sizeof(float)) &&
         ds4_gpu_tensor_write(kv_storage, 0, source, kv_count * sizeof(float));
    if (!ok) goto cleanup;
    /* Sequential insertion is the reference even when a chunk wraps twice. */
    for (uint32_t t = 0; t < n; t++) {
        const uint32_t row = ((uint64_t)pos + t) % cap;
        memcpy(reference + GUARD + (uint64_t)row * dim,
               source + GUARD + (uint64_t)t * dim, dim * sizeof(float));
    }
    ok = ds4_gpu_begin_commands();
    if (!ok) goto cleanup;
    ok = ds4_gpu_store_raw_kv_batch_tensor(raw, kv, cap, pos, n, dim);
    if (!ds4_gpu_end_commands()) ok = 0;
    if (!ds4_gpu_synchronize()) ok = 0;
    if (!ok) goto cleanup;
    ok = check_buffer(raw_storage, reference, actual, raw_count, "batch ring", cap, n, pos) &&
         check_buffer(kv_storage, source, actual, kv_count, "batch source", cap, n, pos);
    if (!ok) goto cleanup;

    /* The next decode insertion must preserve every other row and the guards. */
    one = ds4_gpu_tensor_view(kv, 0, dim * sizeof(float));
    const uint32_t row = ((uint64_t)pos + n) % cap;
    memcpy(reference + GUARD + (uint64_t)row * dim, source + GUARD,
           dim * sizeof(float));
    ok = one && ds4_gpu_begin_commands();
    if (!ok) goto cleanup;
    ok = ds4_gpu_store_raw_kv_tensor(raw, one, cap, row, dim);
    if (!ds4_gpu_end_commands()) ok = 0;
    if (!ds4_gpu_synchronize()) ok = 0;
    if (!ok) goto cleanup;
    ok = check_buffer(raw_storage, reference, actual, raw_count, "decode ring", cap, n, pos) &&
         check_buffer(kv_storage, source, actual, kv_count, "decode source", cap, n, pos);
cleanup:
    ds4_gpu_tensor_free(one);
    ds4_gpu_tensor_free(kv);
    ds4_gpu_tensor_free(raw);
    ds4_gpu_tensor_free(kv_storage);
    ds4_gpu_tensor_free(raw_storage);
    free(actual);
    free(source);
    free(reference);
    return ok;
}

int main(void) {
    if (!ds4_gpu_init()) {
        fprintf(stderr, "raw KV ring: GPU initialization failed\n");
        return 1;
    }
    const uint32_t capacities[] = {1, 31, 256, 8192};
    unsigned cases = 0;
    int ok = 1;
    for (unsigned c = 0; c < sizeof(capacities) / sizeof(capacities[0]) && ok; c++) {
        const uint32_t cap = capacities[c];
        const uint32_t counts[] = {1, cap > 1 ? cap - 1 : 1, cap, cap + 1, 2u * cap + 17u};
        const uint32_t positions[] = {0, cap - 1, UINT32_MAX - 5u};
        for (unsigned n = 0; n < sizeof(counts) / sizeof(counts[0]) && ok; n++) {
            for (unsigned p = 0; p < sizeof(positions) / sizeof(positions[0]) && ok; p++) {
                ok = run_case(cap, counts[n], positions[p], c == 2 ? 257 : 17);
                cases++;
            }
        }
    }
    if (ok) { ok = run_case(8192, 16384, 0, 512); cases++; }
    ds4_gpu_cleanup();
    printf("raw KV ring: %s (%u cases, batch + decode, source and buffer guards)\n",
           ok ? "PASS" : "FAIL", cases);
    return ok ? 0 : 1;
}
