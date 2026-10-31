/* Compare the indexer score kernels bit for bit, and report their rate.
 *
 * The score values only ever feed a top-k, so an end-to-end logits gate cannot
 * see a one-ulp change in them: a different rounding only shows up when it
 * flips a selection near the boundary. This test compares the kernels directly
 * on a large cache, so a fast-math fma contraction or a reordered reduction in
 * the row-grouped kernel fails here instead of in a distant benchmark.
 *
 * Build: make tests/test_v41_indexer_score
 * Run:   ./tests/test_v41_indexer_score
 */
#include "../ds4_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define N_HEAD 32u
#define HEAD_DIM 128u
#define ROW_BYTES (HEAD_DIM * sizeof(float))
#define GUARD_ROWS 16u

static uint64_t rng_state = 0x243f6a8885a308d3ull;

static uint32_t rng_u32(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return (uint32_t)(rng_state >> 32);
}

static float rng_float(void) {
    return (float)((int32_t)rng_u32()) / 2147483648.0f;
}

/* The live index cache holds keys that were already quantized to fp4 (E2M1)
 * with an E8M0 group scale, so a second data set made of exactly representable
 * values keeps the parity check honest for the shipped code path. */
static float rng_fp4(void) {
    static const float steps[8] = { 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f };
    const uint32_t r = rng_u32();
    const float mag = steps[(r >> 8) & 7u];
    return ((r >> 3) & 1u) ? mag : -mag;
}

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec / 1e6;
}

static int run_case(const char *name, uint32_t n_rows, int fp4_data, int f16_cache,
                    int iters) {
    const uint32_t n_head = N_HEAD, head_dim = HEAD_DIM;
    float *q = malloc(sizeof(float) * n_head * head_dim);
    float *weights = malloc(sizeof(float) * n_head);
    const uint32_t output_rows = n_rows + GUARD_ROWS;
    const float sentinel = -1234.5f;
    float *scores_a = malloc(sizeof(float) * output_rows);
    float *scores_b = malloc(sizeof(float) * output_rows);
    if (!q || !weights || !scores_a || !scores_b) return 1;
    for (uint32_t i = 0; i < n_head * head_dim; i++)
        q[i] = fp4_data ? rng_fp4() : rng_float();
    for (uint32_t i = 0; i < n_head; i++) weights[i] = fabsf(rng_float()) * 0.5f;

    const uint64_t cache_elems = (uint64_t)n_rows * head_dim;
    const uint64_t cache_bytes = cache_elems * (f16_cache ? sizeof(_Float16) : sizeof(float));
    void *cache = malloc(cache_bytes);
    if (!cache) return 1;
    for (uint64_t i = 0; i < cache_elems; i++) {
        const float v = fp4_data ? rng_fp4() : rng_float();
        if (f16_cache) ((_Float16 *)cache)[i] = (_Float16)v;
        else ((float *)cache)[i] = v;
    }

    ds4_gpu_tensor *tq = ds4_gpu_tensor_alloc(sizeof(float) * n_head * head_dim);
    ds4_gpu_tensor *tw = ds4_gpu_tensor_alloc(sizeof(float) * n_head);
    ds4_gpu_tensor *tcache = ds4_gpu_tensor_alloc(cache_bytes);
    ds4_gpu_tensor *ta = ds4_gpu_tensor_alloc(sizeof(float) * output_rows);
    ds4_gpu_tensor *tb = ds4_gpu_tensor_alloc(sizeof(float) * output_rows);
    int rc = 1;
    if (!tq || !tw || !tcache || !ta || !tb) goto done;
    if (!ds4_gpu_tensor_fill_f32(ta, sentinel, output_rows) ||
        !ds4_gpu_tensor_fill_f32(tb, sentinel, output_rows)) goto done;
    if (!ds4_gpu_tensor_write(tq, 0, q, sizeof(float) * n_head * head_dim) ||
        !ds4_gpu_tensor_write(tw, 0, weights, sizeof(float) * n_head) ||
        !ds4_gpu_tensor_write(tcache, 0, cache, cache_bytes)) goto done;

    setenv("DS4_METAL_DISABLE_V41_SCORE_ROWS4", "1", 1);
    if (!ds4_gpu_glm_indexer_score_one_tensor(ta, tq, tw, tcache, n_rows,
                                              n_head, head_dim, 1.0f / 64.0f, f16_cache)) goto done;
    unsetenv("DS4_METAL_DISABLE_V41_SCORE_ROWS4");
    if (!ds4_gpu_glm_indexer_score_one_tensor(tb, tq, tw, tcache, n_rows,
                                              n_head, head_dim, 1.0f / 64.0f, f16_cache)) goto done;

    if (!ds4_gpu_tensor_read(ta, 0, scores_a, sizeof(float) * output_rows) ||
        !ds4_gpu_tensor_read(tb, 0, scores_b, sizeof(float) * output_rows)) goto done;
    for (uint32_t i = n_rows; i < output_rows; i++) {
        if (scores_a[i] != sentinel || scores_b[i] != sentinel) {
            fprintf(stderr, "%s: output guard overwritten at row %u\n", name, i);
            goto done;
        }
    }
    int diff = 0;
    double max_abs = 0.0;
    uint32_t first = 0;
    for (uint32_t i = 0; i < n_rows; i++) {
        if (memcmp(&scores_a[i], &scores_b[i], sizeof(float)) != 0) {
            if (!diff) first = i;
            diff++;
            const double d = fabs((double)scores_a[i] - (double)scores_b[i]);
            if (d > max_abs) max_abs = d;
        }
    }
    printf("%-24s rows=%7u %s direct: %s", name, n_rows, f16_cache ? "f16" : "f32",
           diff ? "DIFFER" : "IDENTICAL");
    if (diff) printf(" (%d rows, max_abs=%.3g, first=%u)", diff, max_abs, first);
    printf("\n");

    if (iters > 0) {
        double t_old = 0.0, t_new = 0.0;
        for (int i = 0; i < iters + 2; i++) {
            const int timed = i >= 2;
            setenv("DS4_METAL_DISABLE_V41_SCORE_ROWS4", "1", 1);
            double t0 = now_ms();
            if (!ds4_gpu_glm_indexer_score_one_tensor(ta, tq, tw, tcache, n_rows,
                                                      n_head, head_dim, 1.0f / 64.0f, f16_cache))
                goto done;
            if (timed) t_old += now_ms() - t0;
            unsetenv("DS4_METAL_DISABLE_V41_SCORE_ROWS4");
            t0 = now_ms();
            if (!ds4_gpu_glm_indexer_score_one_tensor(tb, tq, tw, tcache, n_rows,
                                                      n_head, head_dim, 1.0f / 64.0f, f16_cache))
                goto done;
            if (timed) t_new += now_ms() - t0;
        }
        const double bytes = (double)n_rows * ROW_BYTES * iters;
        printf("%-24s                scalar %.2f ms (%.0f GB/s)  rows4 %.2f ms (%.0f GB/s)"
               "  speedup %.2fx\n", "", t_old / iters, bytes / (t_old * 1e6),
               t_new / iters, bytes / (t_new * 1e6), t_old / t_new);
    }

    rc = diff ? 1 : 0;
done:
    ds4_gpu_tensor_free(tq); ds4_gpu_tensor_free(tw); ds4_gpu_tensor_free(tcache);
    ds4_gpu_tensor_free(ta); ds4_gpu_tensor_free(tb);
    free(q); free(weights); free(scores_a); free(scores_b); free(cache);
    return rc;
}

int main(void) {
    if (!ds4_gpu_init()) {
        fprintf(stderr, "metal init failed\n");
        return 1;
    }
    int fail = 0;
    const uint32_t tails[] = {1, 2, 3, 4, 15, 16, 17, 31, 33, 1023, 1025};
    for (size_t i = 0; i < sizeof(tails) / sizeof(*tails); i++) {
        fail |= run_case("tail-f32", tails[i], 0, 0, 0);
        fail |= run_case("tail-f16", tails[i], 0, 1, 0);
    }
    fail |= run_case("random-f32", 1048576u, 0, 0, 10);
    fail |= run_case("fp4-keys-f32", 262144u, 1, 0, 0);
    fail |= run_case("fp4-keys-f16", 262144u, 1, 1, 0);
    fail |= run_case("random-f16", 262144u, 0, 1, 0);
    printf("%s\n", fail ? "FAIL" : "PASS");
    return fail;
}
