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

/* V4.1's full-width bitonic sort can permute equal scores differently
 * if the keys are repacked. Exercise ties at the top-512 cutoff as well
 * as score writes at a partial final block and the output guard. */
static int check_block_selection(void) {
    enum { FULL = 32773, BLOCKS = 2048, TOP = 512 };
    const uint32_t neginf = 0xff800000u;
    ds4_gpu_tensor *q = ds4_gpu_tensor_alloc(N_HEAD * ROW_BYTES);
    ds4_gpu_tensor *weights = ds4_gpu_tensor_alloc(N_HEAD * sizeof(float));
    ds4_gpu_tensor *cache = ds4_gpu_tensor_alloc(FULL * ROW_BYTES);
    ds4_gpu_tensor *blocks = ds4_gpu_tensor_alloc(BLOCKS * sizeof(uint32_t));
    ds4_gpu_tensor *full = ds4_gpu_tensor_alloc(FULL * sizeof(float));
    ds4_gpu_tensor *partial = ds4_gpu_tensor_alloc((FULL + 1u) * sizeof(float));
    ds4_gpu_tensor *full_top = ds4_gpu_tensor_alloc(TOP * sizeof(uint32_t));
    ds4_gpu_tensor *partial_top = ds4_gpu_tensor_alloc(TOP * sizeof(uint32_t));
    int failed = 1;
    if (!q || !weights || !cache || !blocks || !full || !partial || !full_top || !partial_top) goto done;
    float *qp = ds4_gpu_tensor_contents(q), *wp = ds4_gpu_tensor_contents(weights);
    float *keys = ds4_gpu_tensor_contents(cache), *ps = ds4_gpu_tensor_contents(partial);
    uint32_t *ids = ds4_gpu_tensor_contents(blocks);
    for (size_t i = 0; i < N_HEAD * HEAD_DIM; i++) qp[i] = rng_float();
    for (size_t i = 0; i < N_HEAD; i++) wp[i] = rng_float();
    for (size_t i = 0; i < 97 * HEAD_DIM; i++) keys[i] = rng_fp4();
    for (uint32_t r = 97; r < FULL; r++)
        memcpy(keys + r * HEAD_DIM, keys + (r % 97u) * HEAD_DIM, ROW_BYTES);
    for (uint32_t i = 0; i < BLOCKS - 1; i++) ids[i] = 2u * i + 1u;
    ids[BLOCKS - 1] = FULL / 8u;
    /* Block top-k may return any order. */
    for (uint32_t i = 0; i < BLOCKS / 2; i++) {
        uint32_t t = ids[i]; ids[i] = ids[BLOCKS - 1u - i]; ids[BLOCKS - 1u - i] = t;
    }
    for (uint32_t r = 0; r < FULL; r++) memcpy(ps + r, &neginf, sizeof(neginf));
    ps[FULL] = 12345;
    if (!ds4_gpu_glm_indexer_score_one_tensor(full, q, weights, cache, FULL,
                                              N_HEAD, HEAD_DIM, 1.0f / 64.0f, false) ||
        !ds4_gpu_dsv41_indexer_score_blocks(partial, q, weights, cache, blocks, FULL, BLOCKS)) goto done;
    float *fs = ds4_gpu_tensor_contents(full);
    for (uint32_t r = 0; r < FULL; r++) {
        const uint32_t block = r / 8u;
        if (block != FULL / 8u && (!(block & 1u) || block == 4095u))
            memcpy(fs + r, &neginf, sizeof(neginf));
    }
    if (ps[FULL] != 12345 || memcmp(fs, ps, FULL * sizeof(float))) goto done;
    setenv("DS4_METAL_DISABLE_V41_TOPK_PREFIX", "1", 1);
    if (!ds4_gpu_indexer_topk_tensor(full_top, full, FULL, 1, TOP)) goto done;
    unsetenv("DS4_METAL_DISABLE_V41_TOPK_PREFIX");
    if (!ds4_gpu_indexer_topk_tensor(partial_top, partial, FULL, 1, TOP)) goto done;
    failed = memcmp(ds4_gpu_tensor_contents(full_top), ds4_gpu_tensor_contents(partial_top),
                    TOP * sizeof(uint32_t)) != 0;
done:
    unsetenv("DS4_METAL_DISABLE_V41_TOPK_PREFIX");
    ds4_gpu_tensor_free(q); ds4_gpu_tensor_free(weights); ds4_gpu_tensor_free(cache);
    ds4_gpu_tensor_free(blocks); ds4_gpu_tensor_free(full); ds4_gpu_tensor_free(partial);
    ds4_gpu_tensor_free(full_top); ds4_gpu_tensor_free(partial_top);
    printf("candidate block scores and tied top-512: %s\n", failed ? "FAIL" : "identical");
    return failed;
}

/* The V4 Flash kernel reduces 64 heads, with one shared query load per four
 * keys. Check every tail length, exact float scores and the output boundary. */
static int check_flash_score_rows4(void) {
    enum { HEADS = 64, ROWS = 8193 };
    const uint32_t widths[] = {2047, 2048, 2049, 2050, 2051, ROWS};
    ds4_gpu_tensor *q = ds4_gpu_tensor_alloc(HEADS * ROW_BYTES);
    ds4_gpu_tensor *weights = ds4_gpu_tensor_alloc(HEADS * sizeof(float));
    ds4_gpu_tensor *keys = ds4_gpu_tensor_alloc(ROWS * ROW_BYTES);
    ds4_gpu_tensor *one = ds4_gpu_tensor_alloc((ROWS + GUARD_ROWS) * sizeof(float));
    ds4_gpu_tensor *four = ds4_gpu_tensor_alloc((ROWS + GUARD_ROWS) * sizeof(float));
    int failed = 1;
    if (!q || !weights || !keys || !one || !four) goto done;
    float *query = ds4_gpu_tensor_contents(q), *weight = ds4_gpu_tensor_contents(weights);
    float *key = ds4_gpu_tensor_contents(keys), *a = ds4_gpu_tensor_contents(one);
    float *b = ds4_gpu_tensor_contents(four);
    const float scale = 1.0f / sqrtf((float)(HEADS * HEAD_DIM));
    for (int fp4 = 0; fp4 <= 1; fp4++) {
        for (size_t i = 0; i < HEADS * HEAD_DIM; i++)
            query[i] = fp4 ? rng_fp4() : ldexpf(rng_float(), (int)(i % 23) - 11);
        for (size_t i = 0; i < HEADS; i++) weight[i] = rng_float();
        for (size_t i = 0; i < ROWS * HEAD_DIM; i++)
            key[i] = fp4 ? rng_fp4() : ldexpf(rng_float(), (int)(i % 19) - 9);
        for (size_t i = 0; i < sizeof(widths) / sizeof(widths[0]); i++) {
            const uint32_t n = widths[i];
            for (uint32_t j = n; j < n + GUARD_ROWS; j++) b[j] = -123.0f;
            ds4_gpu_test_set_flags(DS4_GPU_TEST_INDEX_SCORE_DIRECT);
            if (!ds4_gpu_indexer_score_one_tensor(one, q, weights, keys, n, HEADS, HEAD_DIM, scale)) goto done;
            ds4_gpu_test_set_flags(DS4_GPU_TEST_COUNT_KERNELS);
            const char *name = "kernel_dsv4_indexer_score_one_rows4";
            const uint64_t before = ds4_gpu_test_kernel_runs(name);
            if (!ds4_gpu_indexer_score_one_tensor(four, q, weights, keys, n, HEADS, HEAD_DIM, scale) ||
                memcmp(a, b, n * sizeof(*a))) goto done;
            if (!getenv("DS4_METAL_MATH_SAFE") &&
                ds4_gpu_test_kernel_runs(name) != before + (n >= 2048u)) goto done;
            for (uint32_t j = n; j < n + GUARD_ROWS; j++) if (b[j] != -123.0f) goto done;
        }
    }
    failed = 0;
done:
    ds4_gpu_test_set_flags(0);
    ds4_gpu_tensor_free(q); ds4_gpu_tensor_free(weights); ds4_gpu_tensor_free(keys);
    ds4_gpu_tensor_free(one); ds4_gpu_tensor_free(four);
    fprintf(stderr, "V4 Flash four-key score, exact scores and tail guards: %s\n", failed ? "FAIL" : "PASS");
    return failed;
}

int main(void) {
    if (!ds4_gpu_init()) {
        fprintf(stderr, "metal init failed\n");
        return 1;
    }
    int fail = check_block_selection();
    fail |= check_flash_score_rows4();
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
