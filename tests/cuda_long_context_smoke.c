#include "ds4_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double monotonic_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1000000000.0;
}

static double getenv_seconds(const char *name, double fallback) {
    const char *s = getenv(name);
    if (!s || !s[0]) return fallback;
    char *end = NULL;
    const double v = strtod(s, &end);
    return end != s && v > 0.0 ? v : fallback;
}

static float qat_value(uint64_t i) {
    static const float levels[] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    };
    const float scale = ldexpf(1.0f, (int)((i / 32u) % 7u) - 4);
    const float sign = ((i * 0x9e3779b97f4a7c15ull) >> 63) ? -1.0f : 1.0f;
    return sign * scale * levels[(i * 13u + i / 17u) & 7u];
}

static int check_large_topk(void) {
    const uint32_t n_comp = 32768;
    const uint32_t n_tokens = 32;
    const uint32_t top_k = 512;
    const uint64_t score_count = (uint64_t)n_comp * n_tokens;
    float *scores_host = (float *)malloc((size_t)score_count * sizeof(float));
    uint32_t *selected_host = (uint32_t *)malloc((size_t)n_tokens * top_k * sizeof(uint32_t));
    if (!scores_host || !selected_host) return 1;

    for (uint32_t t = 0; t < n_tokens; t++) {
        for (uint32_t i = 0; i < n_comp; i++) {
            scores_host[(uint64_t)t * n_comp + i] = (float)i;
        }
    }

    ds4_gpu_tensor *scores = ds4_gpu_tensor_alloc(score_count * sizeof(float));
    ds4_gpu_tensor *selected = ds4_gpu_tensor_alloc((uint64_t)n_tokens * top_k * sizeof(uint32_t));
    int rc = 1;
    double elapsed = 0.0;
    if (scores && selected &&
        ds4_gpu_tensor_write(scores, 0, scores_host, score_count * sizeof(float))) {
        /* Exclude one-time CUDA module/kernel setup from the throughput guard. */
        if (!ds4_gpu_indexer_topk_tensor(selected, scores, n_comp, n_tokens, top_k) ||
            !ds4_gpu_synchronize()) {
            rc = 1;
            goto cleanup;
        }
        const double t0 = monotonic_seconds();
        if (ds4_gpu_indexer_topk_tensor(selected, scores, n_comp, n_tokens, top_k) &&
            ds4_gpu_synchronize()) {
            elapsed = monotonic_seconds() - t0;
            rc = ds4_gpu_tensor_read(selected, 0, selected_host,
                                     (uint64_t)n_tokens * top_k * sizeof(uint32_t)) ? 0 : 1;
        }
    }
    if (rc == 0) {
        for (uint32_t t = 0; t < n_tokens && rc == 0; t++) {
            for (uint32_t i = 0; i < top_k; i++) {
                const uint32_t expected = n_comp - 1u - i;
                const uint32_t got = selected_host[(uint64_t)t * top_k + i];
                if (got != expected) {
                    fprintf(stderr, "top-k mismatch token=%u rank=%u got=%u expected=%u\n",
                            t, i, got, expected);
                    rc = 1;
                    break;
                }
            }
        }
    }
    if (rc == 0) {
        const double max_seconds = getenv_seconds("DS4_CUDA_TOPK_REGRESSION_SEC", 2.0);
        fprintf(stderr, "cuda-regression: top-k n_comp=%u n_tokens=%u elapsed=%.3fs\n",
                n_comp, n_tokens, elapsed);
        if (elapsed > max_seconds) {
            fprintf(stderr, "top-k regression: %.3fs exceeds %.3fs\n", elapsed, max_seconds);
            rc = 1;
        }
    }

cleanup:
    ds4_gpu_tensor_free(selected);
    ds4_gpu_tensor_free(scores);
    free(selected_host);
    free(scores_host);
    return rc;
}

static int check_decode_attention_overflow_path(void) {
    const uint32_t n_head = 8;
    const uint32_t head_dim = 512;
    const uint32_t n_raw = 128;
    const uint32_t n_comp = 8192;
    const uint64_t q_count = (uint64_t)n_head * head_dim;
    const uint64_t raw_count = (uint64_t)n_raw * head_dim;
    const uint64_t comp_count = (uint64_t)n_comp * head_dim;

    float *sinks = (float *)calloc(n_head, sizeof(float));
    float *q_host = (float *)calloc((size_t)q_count, sizeof(float));
    float *raw_host = (float *)calloc((size_t)raw_count, sizeof(float));
    float *comp_host = (float *)calloc((size_t)comp_count, sizeof(float));
    float *heads_host = (float *)calloc((size_t)q_count, sizeof(float));
    if (!sinks || !q_host || !raw_host || !comp_host || !heads_host) return 1;

    for (uint32_t c = 0; c < n_comp; c++) {
        comp_host[(uint64_t)c * head_dim] = 1.0f;
    }

    ds4_gpu_tensor *heads = ds4_gpu_tensor_alloc(q_count * sizeof(float));
    ds4_gpu_tensor *q = ds4_gpu_tensor_alloc(q_count * sizeof(float));
    ds4_gpu_tensor *raw = ds4_gpu_tensor_alloc(raw_count * sizeof(float));
    ds4_gpu_tensor *comp = ds4_gpu_tensor_alloc(comp_count * sizeof(float));
    int rc = 1;
    if (heads && q && raw && comp &&
        ds4_gpu_tensor_write(q, 0, q_host, q_count * sizeof(float)) &&
        ds4_gpu_tensor_write(raw, 0, raw_host, raw_count * sizeof(float)) &&
        ds4_gpu_tensor_write(comp, 0, comp_host, comp_count * sizeof(float)) &&
        ds4_gpu_attention_decode_heads_tensor(heads,
                                              sinks,
                                              n_head * sizeof(float),
                                              0,
                                              q,
                                              raw,
                                              n_raw,
                                              n_raw,
                                              0,
                                              comp,
                                              0,
                                              n_comp,
                                              NULL,
                                              0,
                                              n_head,
                                              head_dim) &&
        ds4_gpu_synchronize() &&
        ds4_gpu_tensor_read(heads, 0, heads_host, q_count * sizeof(float))) {
        rc = 0;
        for (uint32_t h = 0; h < n_head; h++) {
            const float v = heads_host[(uint64_t)h * head_dim];
            if (v < 0.90f) {
                fprintf(stderr, "long attention ignored compressed rows for head=%u value=%f\n",
                        h, (double)v);
                rc = 1;
            }
        }
    }

    ds4_gpu_tensor_free(comp);
    ds4_gpu_tensor_free(raw);
    ds4_gpu_tensor_free(q);
    ds4_gpu_tensor_free(heads);
    free(heads_host);
    free(comp_host);
    free(raw_host);
    free(q_host);
    free(sinks);
    return rc;
}

static int check_b1_indexer_wmma(void) {
    const uint32_t n_comp = 8192u;
    const uint32_t n_head = 64u;
    const uint32_t head_dim = 128u;
    const uint32_t top_k = 512u;
    const uint64_t q_count = (uint64_t)n_head * head_dim;
    const uint64_t cache_count = (uint64_t)n_comp * head_dim;
    float *q_host = malloc((size_t)q_count * sizeof(*q_host));
    float *weights_host = malloc((size_t)n_head * sizeof(*weights_host));
    float *cache_host = malloc((size_t)cache_count * sizeof(*cache_host));
    float *direct_host = malloc((size_t)n_comp * sizeof(*direct_host));
    float *wmma_host = malloc((size_t)n_comp * sizeof(*wmma_host));
    uint32_t *direct_topk = malloc((size_t)top_k * sizeof(*direct_topk));
    uint32_t *wmma_topk = malloc((size_t)top_k * sizeof(*wmma_topk));
    if (!q_host || !weights_host || !cache_host || !direct_host ||
        !wmma_host || !direct_topk || !wmma_topk) {
        free(wmma_topk);
        free(direct_topk);
        free(wmma_host);
        free(direct_host);
        free(cache_host);
        free(weights_host);
        free(q_host);
        return 1;
    }

    for (uint64_t i = 0; i < q_count; i++) {
        q_host[i] = qat_value(i + 7u);
    }
    for (uint32_t i = 0; i < n_head; i++) {
        weights_host[i] = 0.25f + (float)((i * 19u) % 31u) / 31.0f;
    }
    for (uint64_t i = 0; i < cache_count; i++) {
        cache_host[i] = qat_value(i + (i / head_dim) * 97u + 23u);
    }

    ds4_gpu_tensor *q = ds4_gpu_tensor_alloc(q_count * sizeof(float));
    ds4_gpu_tensor *weights = ds4_gpu_tensor_alloc(n_head * sizeof(float));
    ds4_gpu_tensor *cache = ds4_gpu_tensor_alloc(cache_count * sizeof(float));
    ds4_gpu_tensor *scores = ds4_gpu_tensor_alloc(n_comp * sizeof(float));
    ds4_gpu_tensor *selected = ds4_gpu_tensor_alloc(top_k * sizeof(uint32_t));
    int rc = 1;
    if (!q || !weights || !cache || !scores || !selected ||
        !ds4_gpu_tensor_write(q, 0, q_host, q_count * sizeof(float)) ||
        !ds4_gpu_tensor_write(weights, 0, weights_host,
                              n_head * sizeof(float)) ||
        !ds4_gpu_tensor_write(cache, 0, cache_host,
                              cache_count * sizeof(float))) {
        goto cleanup;
    }

    setenv("DS4_CUDA_NO_INDEXER_WMMA", "1", 1);
    if (!ds4_gpu_indexer_score_one_tensor(scores, q, weights, cache,
                                          n_comp, n_head, head_dim,
                                          1.0f / sqrtf(8192.0f)) ||
        !ds4_gpu_indexer_topk_tensor(selected, scores, n_comp, 1u, top_k) ||
        !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(scores, 0, direct_host,
                             n_comp * sizeof(float)) ||
        !ds4_gpu_tensor_read(selected, 0, direct_topk,
                             top_k * sizeof(uint32_t))) {
        goto cleanup;
    }
    unsetenv("DS4_CUDA_NO_INDEXER_WMMA");
    if (!ds4_gpu_indexer_score_one_tensor(scores, q, weights, cache,
                                          n_comp, n_head, head_dim,
                                          1.0f / sqrtf(8192.0f)) ||
        !ds4_gpu_indexer_topk_tensor(selected, scores, n_comp, 1u, top_k) ||
        !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(scores, 0, wmma_host,
                             n_comp * sizeof(float)) ||
        !ds4_gpu_tensor_read(selected, 0, wmma_topk,
                             top_k * sizeof(uint32_t))) {
        goto cleanup;
    }

    float max_abs = 0.0f;
    for (uint32_t i = 0; i < n_comp; i++) {
        const float delta = fabsf(direct_host[i] - wmma_host[i]);
        if (delta > max_abs) max_abs = delta;
    }
    uint32_t topk_diff = 0u;
    for (uint32_t i = 0; i < top_k; i++) {
        if (direct_topk[i] != wmma_topk[i]) topk_diff++;
    }
    fprintf(stderr,
            "cuda-regression: B1 indexer n_comp=%u max_abs=%g topk_diff=%u\n",
            n_comp, (double)max_abs, topk_diff);
    rc = max_abs <= 1.0e-3f && topk_diff == 0u ? 0 : 1;

cleanup:
    unsetenv("DS4_CUDA_NO_INDEXER_WMMA");
    ds4_gpu_tensor_free(selected);
    ds4_gpu_tensor_free(scores);
    ds4_gpu_tensor_free(cache);
    ds4_gpu_tensor_free(weights);
    ds4_gpu_tensor_free(q);
    free(wmma_topk);
    free(direct_topk);
    free(wmma_host);
    free(direct_host);
    free(cache_host);
    free(weights_host);
    free(q_host);
    return rc;
}

int main(void) {
    if (!ds4_gpu_init()) return 1;
    int rc = check_large_topk();
    if (check_decode_attention_overflow_path() != 0) rc = 1;
    if (check_b1_indexer_wmma() != 0) rc = 1;
    ds4_gpu_cleanup();
    if (rc == 0) puts("cuda long-context regression: OK");
    return rc;
}
