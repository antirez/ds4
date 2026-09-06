#include "ds4_gpu.h"

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

static float topk_adversarial_value(uint64_t i) {
    uint32_t x = (uint32_t)i ^ (uint32_t)(i >> 32);
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    /* Deliberate collisions exercise the value/index tie break. */
    if ((i % 29u) == 0u) return 0.5f;
    if ((i % 113u) == 0u) return -0.0f;
    return (float)(int32_t)(x & 0x1ffffu) * (1.0f / 65536.0f) - 1.0f;
}

static int check_stream_topk_matches_reference(void) {
    static const uint32_t n_comp_cases[] = {8193u, 8448u, 32768u};
    const uint32_t n_tokens = 32u;
    const uint32_t top_k = 512u;

    for (uint32_t c = 0; c < sizeof(n_comp_cases) / sizeof(n_comp_cases[0]); c++) {
        const uint32_t n_comp = n_comp_cases[c];
        const uint64_t score_count = (uint64_t)n_comp * n_tokens;
        const uint64_t selected_count = (uint64_t)n_tokens * top_k;
        float *scores_host = malloc((size_t)score_count * sizeof(*scores_host));
        uint32_t *stream_host = malloc((size_t)selected_count * sizeof(*stream_host));
        uint32_t *reference_host = malloc((size_t)selected_count * sizeof(*reference_host));
        ds4_gpu_tensor *scores = ds4_gpu_tensor_alloc(score_count * sizeof(float));
        ds4_gpu_tensor *selected = ds4_gpu_tensor_alloc(selected_count * sizeof(uint32_t));
        int rc = 1;
        if (!scores_host || !stream_host || !reference_host || !scores || !selected) {
            goto cleanup;
        }
        for (uint64_t i = 0; i < score_count; i++) {
            scores_host[i] = topk_adversarial_value(i + (uint64_t)c * 0x100000001b3ull);
        }
        if (!ds4_gpu_tensor_write(scores, 0, scores_host,
                                  score_count * sizeof(float))) {
            goto cleanup;
        }

        unsetenv("DS4_CUDA_NO_TOPK_STREAM");
        if (!ds4_gpu_indexer_topk_tensor(selected, scores, n_comp, n_tokens, top_k) ||
            !ds4_gpu_synchronize() ||
            !ds4_gpu_tensor_read(selected, 0, stream_host,
                                  selected_count * sizeof(uint32_t))) {
            goto cleanup;
        }

        setenv("DS4_CUDA_NO_TOPK_STREAM", "1", 1);
        if (!ds4_gpu_indexer_topk_tensor(selected, scores, n_comp, n_tokens, top_k) ||
            !ds4_gpu_synchronize() ||
            !ds4_gpu_tensor_read(selected, 0, reference_host,
                                  selected_count * sizeof(uint32_t))) {
            goto cleanup;
        }
        unsetenv("DS4_CUDA_NO_TOPK_STREAM");

        for (uint64_t i = 0; i < selected_count; i++) {
            if (stream_host[i] != reference_host[i]) {
                fprintf(stderr,
                        "stream top-k mismatch n_comp=%u token=%llu rank=%llu got=%u expected=%u\n",
                        n_comp,
                        (unsigned long long)(i / top_k),
                        (unsigned long long)(i % top_k),
                        stream_host[i], reference_host[i]);
                goto cleanup;
            }
        }
        fprintf(stderr,
                "cuda-regression: stream top-k matches reference n_comp=%u n_tokens=%u\n",
                n_comp, n_tokens);
        rc = 0;

cleanup:
        unsetenv("DS4_CUDA_NO_TOPK_STREAM");
        ds4_gpu_tensor_free(selected);
        ds4_gpu_tensor_free(scores);
        free(reference_host);
        free(stream_host);
        free(scores_host);
        if (rc != 0) return rc;
    }
    return 0;
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

int main(void) {
    if (!ds4_gpu_init()) return 1;
    int rc = check_large_topk();
    if (check_stream_topk_matches_reference() != 0) rc = 1;
    if (check_decode_attention_overflow_path() != 0) rc = 1;
    ds4_gpu_cleanup();
    if (rc == 0) puts("cuda long-context regression: OK");
    return rc;
}
