#include "ds4_gpu.h"

#include <stdint.h>
#include <math.h>
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
        /* Exclude one-time GPU module/kernel setup from the throughput guard. */
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
        const double max_seconds = getenv_seconds("DS4_GPU_TOPK_REGRESSION_SEC",
                getenv_seconds("DS4_CUDA_TOPK_REGRESSION_SEC", 2.0));
        fprintf(stderr, "gpu-regression: top-k n_comp=%u n_tokens=%u elapsed=%.3fs\n",
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

static int check_decode_attention_ring_reference(void) {
    const uint32_t n_head = 8;
    const uint32_t head_dim = 512;
    const uint32_t n_raw = 257;
    const uint32_t raw_cap = 300;
    const uint32_t raw_start = 270;
    const uint32_t n_comp = 511;
    const uint64_t q_count = (uint64_t)n_head * head_dim;
    const uint64_t raw_count = (uint64_t)raw_cap * head_dim;
    const uint64_t comp_count = (uint64_t)n_comp * head_dim;
    float *sinks = (float *)malloc(n_head * sizeof(float));
    float *q_host = (float *)malloc((size_t)q_count * sizeof(float));
    float *raw_host = (float *)malloc((size_t)raw_count * sizeof(float));
    float *comp_host = (float *)malloc((size_t)comp_count * sizeof(float));
    float *mask_host = (float *)malloc(n_comp * sizeof(float));
    float *heads_host = (float *)malloc((size_t)q_count * sizeof(float));
    float *reference = (float *)malloc((size_t)q_count * sizeof(float));
    float *scores = (float *)malloc((n_raw + n_comp) * sizeof(float));
    if (!sinks || !q_host || !raw_host || !comp_host || !mask_host ||
        !heads_host || !reference || !scores) return 1;

    for (uint32_t h = 0; h < n_head; h++) sinks[h] = -0.25f + (float)h * 0.01f;
    for (uint64_t i = 0; i < q_count; i++)
        q_host[i] = (float)((int)((i * 13u + 5u) % 101u) - 50) / 700.0f;
    for (uint64_t i = 0; i < raw_count; i++)
        raw_host[i] = (float)((int)((i * 17u + 3u) % 127u) - 63) / 500.0f;
    for (uint64_t i = 0; i < comp_count; i++)
        comp_host[i] = (float)((int)((i * 19u + 7u) % 131u) - 65) / 550.0f;
    for (uint32_t c = 0; c < n_comp; c++)
        mask_host[c] = (c % 17u == 0u) ? -1.0e30f : -(float)(c % 7u) * 0.015f;

    const float scale = 1.0f / sqrtf((float)head_dim);
    for (uint32_t h = 0; h < n_head; h++) {
        const float *qh = q_host + (uint64_t)h * head_dim;
        float max_score = sinks[h];
        for (uint32_t r = 0; r < n_raw; r++) {
            const uint32_t row = (raw_start + r) % raw_cap;
            const float *kv = raw_host + (uint64_t)row * head_dim;
            float dot = 0.0f;
            for (uint32_t d = 0; d < head_dim; d++) dot += qh[d] * kv[d];
            scores[r] = dot * scale;
            if (scores[r] > max_score) max_score = scores[r];
        }
        for (uint32_t c = 0; c < n_comp; c++) {
            float s = -3.4e38f;
            if (mask_host[c] > -5.0e29f) {
                const float *kv = comp_host + (uint64_t)c * head_dim;
                float dot = 0.0f;
                for (uint32_t d = 0; d < head_dim; d++) dot += qh[d] * kv[d];
                s = dot * scale + mask_host[c];
            }
            scores[n_raw + c] = s;
            if (s > max_score) max_score = s;
        }
        float denom = expf(sinks[h] - max_score);
        for (uint32_t r = 0; r < n_raw + n_comp; r++) {
            scores[r] = expf(scores[r] - max_score);
            denom += scores[r];
        }
        for (uint32_t d = 0; d < head_dim; d++) {
            float acc = 0.0f;
            for (uint32_t r = 0; r < n_raw; r++) {
                const uint32_t row = (raw_start + r) % raw_cap;
                acc += scores[r] * raw_host[(uint64_t)row * head_dim + d];
            }
            for (uint32_t c = 0; c < n_comp; c++)
                acc += scores[n_raw + c] * comp_host[(uint64_t)c * head_dim + d];
            reference[(uint64_t)h * head_dim + d] = acc / denom;
        }
    }

    ds4_gpu_tensor *heads = ds4_gpu_tensor_alloc(q_count * sizeof(float));
    ds4_gpu_tensor *q = ds4_gpu_tensor_alloc(q_count * sizeof(float));
    ds4_gpu_tensor *raw = ds4_gpu_tensor_alloc(raw_count * sizeof(float));
    ds4_gpu_tensor *comp = ds4_gpu_tensor_alloc(comp_count * sizeof(float));
    ds4_gpu_tensor *mask = ds4_gpu_tensor_alloc((uint64_t)n_comp * sizeof(float));
    int rc = 1;
    if (heads && q && raw && comp && mask &&
        ds4_gpu_tensor_write(q, 0, q_host, q_count * sizeof(float)) &&
        ds4_gpu_tensor_write(raw, 0, raw_host, raw_count * sizeof(float)) &&
        ds4_gpu_tensor_write(comp, 0, comp_host, comp_count * sizeof(float)) &&
        ds4_gpu_tensor_write(mask, 0, mask_host, (uint64_t)n_comp * sizeof(float)) &&
        ds4_gpu_attention_decode_heads_tensor(heads, sinks, n_head * sizeof(float), 0,
                                              q, raw, n_raw, raw_cap, raw_start,
                                              comp, 0, n_comp, mask, 1,
                                              n_head, head_dim) &&
        ds4_gpu_synchronize() &&
        ds4_gpu_tensor_read(heads, 0, heads_host, q_count * sizeof(float))) {
        float max_abs = 0.0f;
        float max_rel = 0.0f;
        for (uint64_t i = 0; i < q_count; i++) {
            const float abs_err = fabsf(heads_host[i] - reference[i]);
            const float rel_err = abs_err / fmaxf(1.0e-4f, fabsf(reference[i]));
            if (abs_err > max_abs) max_abs = abs_err;
            if (rel_err > max_rel) max_rel = rel_err;
        }
        fprintf(stderr, "gpu-regression: attention ring reference max_abs=%g max_rel=%g\n",
                (double)max_abs, (double)max_rel);
        rc = (max_abs <= 2.0e-5f && max_rel <= 2.0e-3f) ? 0 : 1;
    }

    ds4_gpu_tensor_free(mask);
    ds4_gpu_tensor_free(comp);
    ds4_gpu_tensor_free(raw);
    ds4_gpu_tensor_free(q);
    ds4_gpu_tensor_free(heads);
    free(scores); free(reference); free(heads_host); free(mask_host);
    free(comp_host); free(raw_host); free(q_host); free(sinks);
    return rc;
}

int main(void) {
    if (!ds4_gpu_init()) return 1;
    int rc = check_large_topk();
    if (check_decode_attention_overflow_path() != 0) rc = 1;
    if (check_decode_attention_ring_reference() != 0) rc = 1;
    ds4_gpu_cleanup();
    if (rc == 0) puts("GPU long-context regression: OK");
    return rc;
}
