#include "ds4_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define TEST_DS4_RMS_EPS 1.0e-6f

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
    const uint32_t n_comp = 8100;
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
                fprintf(stderr, "attention fallback ignored compressed rows for head=%u value=%f\n",
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

static int check_partial_direct_model_cache_precedence(void) {
    if (setenv("DS4_CUDA_DIRECT_MODEL", "1", 1) != 0 ||
        setenv("DS4_CUDA_PARTIAL_WEIGHT_CACHE", "1", 1) != 0 ||
        setenv("DS4_CUDA_WEIGHT_CACHE_LIMIT_MB", "16", 1) != 0) {
        return 1;
    }

    const float rms_weight[4] = {2.0f, 3.0f, 4.0f, 5.0f};
    static float host_model[4];
    const uint64_t model_size = sizeof(host_model);
    float out_host[4] = {0};
    for (uint32_t i = 0; i < 4; i++) host_model[i] = rms_weight[i];

    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(sizeof(rms_weight));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(sizeof(rms_weight));
    int rc = 1;
    const float x_host[4] = {1.0f, 1.0f, 1.0f, 1.0f};

    if (x && out &&
        ds4_gpu_set_model_map(host_model, model_size) &&
        ds4_gpu_tensor_write(x, 0, x_host, sizeof(x_host)) &&
        ds4_gpu_cache_model_range(host_model, model_size, 0, model_size, "test_partial_rms")) {
        for (uint32_t i = 0; i < 4; i++) host_model[i] = -100.0f - (float)i;
        if (ds4_gpu_rms_norm_weight_tensor(out, x, host_model, model_size, 0, 4, TEST_DS4_RMS_EPS) &&
            ds4_gpu_synchronize() &&
            ds4_gpu_tensor_read(out, 0, out_host, sizeof(out_host))) {
            rc = 0;
            const float scale = 1.0f / sqrtf(1.0f + TEST_DS4_RMS_EPS);
            for (uint32_t i = 0; i < 4; i++) {
                const float want = rms_weight[i] * scale;
                if (fabsf(out_host[i] - want) > 1.0e-3f) {
                    fprintf(stderr,
                            "partial direct cache rms mismatch index=%u got=%f expected=%f\n",
                            i,
                            (double)out_host[i],
                            (double)want);
                    rc = 1;
                }
            }
        }
    }

    ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(x);
    return rc;
}

static int check_partial_direct_model_cache_budget_miss(void) {
    if (setenv("DS4_CUDA_DIRECT_MODEL", "1", 1) != 0 ||
        setenv("DS4_CUDA_PARTIAL_WEIGHT_CACHE", "1", 1) != 0 ||
        setenv("DS4_CUDA_WEIGHT_CACHE_LIMIT_MB", "1", 1) != 0) {
        return 1;
    }

    const uint64_t bytes = 2ull * 1024ull * 1024ull;
    unsigned char *host_model = (unsigned char *)malloc((size_t)bytes);
    if (!host_model) return 1;
    const int cached = ds4_gpu_cache_model_range(host_model, bytes, 0, bytes, "test_partial_budget");
    free(host_model);
    if (cached != 0) {
        fprintf(stderr, "partial direct cache budget test unexpectedly cached 2 MiB under 1 MiB limit\n");
        return 1;
    }
    return 0;
}

int main(void) {
    if (!ds4_gpu_init()) return 1;
    int rc = check_large_topk();
    if (check_decode_attention_overflow_path() != 0) rc = 1;
    if (check_partial_direct_model_cache_precedence() != 0) rc = 1;
    if (check_partial_direct_model_cache_budget_miss() != 0) rc = 1;
    ds4_gpu_cleanup();
    if (rc == 0) puts("cuda long-context regression: OK");
    return rc;
}
