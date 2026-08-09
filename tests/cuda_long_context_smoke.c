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

static int check_pro_topk(void) {
    const uint32_t n_comp = 2048;
    const uint32_t n_tokens = 8;
    const uint32_t top_k = 1024;
    const uint64_t score_count = (uint64_t)n_comp * n_tokens;
    const uint64_t selected_count = (uint64_t)n_tokens * top_k;
    float *scores_host = (float *)malloc((size_t)score_count * sizeof(float));
    uint32_t *selected_host =
        (uint32_t *)malloc((size_t)selected_count * sizeof(uint32_t));
    if (!scores_host || !selected_host) return 1;

    for (uint32_t t = 0; t < n_tokens; t++) {
        for (uint32_t i = 0; i < n_comp; i++) {
            scores_host[(uint64_t)t * n_comp + i] = (float)i;
        }
    }

    ds4_gpu_tensor *scores = ds4_gpu_tensor_alloc(score_count * sizeof(float));
    ds4_gpu_tensor *selected =
        ds4_gpu_tensor_alloc(selected_count * sizeof(uint32_t));
    int rc = 1;
    if (scores && selected &&
        ds4_gpu_tensor_write(scores, 0, scores_host,
                             score_count * sizeof(float)) &&
        ds4_gpu_indexer_topk_tensor(selected, scores, n_comp, n_tokens, top_k) &&
        ds4_gpu_synchronize() &&
        ds4_gpu_tensor_read(selected, 0, selected_host,
                            selected_count * sizeof(uint32_t))) {
        rc = 0;
        for (uint32_t t = 0; t < n_tokens && rc == 0; t++) {
            for (uint32_t i = 0; i < top_k; i++) {
                const uint32_t expected = n_comp - 1u - i;
                const uint32_t got = selected_host[(uint64_t)t * top_k + i];
                if (got != expected) {
                    fprintf(stderr,
                            "Pro top-k mismatch token=%u rank=%u got=%u expected=%u\n",
                            t, i, got, expected);
                    rc = 1;
                    break;
                }
            }
        }
    }

    ds4_gpu_tensor_free(selected);
    ds4_gpu_tensor_free(scores);
    free(selected_host);
    free(scores_host);
    return rc;
}

static int check_pro_indexed_attention(void) {
    const uint32_t n_head = 8;
    const uint32_t head_dim = 512;
    const uint32_t n_tokens = 2;
    const uint32_t n_raw = 128;
    const uint32_t n_comp = 1056;
    const uint32_t top_k = 1024;
    const uint64_t q_count = (uint64_t)n_tokens * n_head * head_dim;
    const uint64_t raw_count = (uint64_t)n_raw * head_dim;
    const uint64_t comp_count = (uint64_t)n_comp * head_dim;
    const uint64_t selected_count = (uint64_t)n_tokens * top_k;

    float *sinks = (float *)calloc(n_head, sizeof(float));
    float *q_host = (float *)calloc((size_t)q_count, sizeof(float));
    float *raw_host = (float *)calloc((size_t)raw_count, sizeof(float));
    float *comp_host = (float *)calloc((size_t)comp_count, sizeof(float));
    float *heads_host = (float *)calloc((size_t)q_count, sizeof(float));
    uint32_t *selected_host =
        (uint32_t *)malloc((size_t)selected_count * sizeof(uint32_t));
    if (!sinks || !q_host || !raw_host || !comp_host || !heads_host ||
        !selected_host) {
        return 1;
    }
    for (uint32_t c = 0; c < n_comp; c++) {
        comp_host[(uint64_t)c * head_dim] = 1.0f;
    }
    for (uint32_t t = 0; t < n_tokens; t++) {
        for (uint32_t i = 0; i < top_k; i++) {
            selected_host[(uint64_t)t * top_k + i] = i;
        }
    }

    ds4_gpu_tensor *heads = ds4_gpu_tensor_alloc(q_count * sizeof(float));
    ds4_gpu_tensor *q = ds4_gpu_tensor_alloc(q_count * sizeof(float));
    ds4_gpu_tensor *raw = ds4_gpu_tensor_alloc(raw_count * sizeof(float));
    ds4_gpu_tensor *comp = ds4_gpu_tensor_alloc(comp_count * sizeof(float));
    ds4_gpu_tensor *selected =
        ds4_gpu_tensor_alloc(selected_count * sizeof(uint32_t));
    int rc = 1;
    if (heads && q && raw && comp && selected &&
        ds4_gpu_tensor_write(q, 0, q_host, q_count * sizeof(float)) &&
        ds4_gpu_tensor_write(raw, 0, raw_host, raw_count * sizeof(float)) &&
        ds4_gpu_tensor_write(comp, 0, comp_host, comp_count * sizeof(float)) &&
        ds4_gpu_tensor_write(selected, 0, selected_host,
                             selected_count * sizeof(uint32_t)) &&
        ds4_gpu_attention_indexed_mixed_batch_heads_tensor(
            heads, sinks, n_head * sizeof(float), 0, q, raw, comp, 0,
            selected, n_tokens, 4096, n_raw, n_raw, 0, n_comp, top_k,
            n_raw, 4, n_head, head_dim) &&
        ds4_gpu_synchronize() &&
        ds4_gpu_tensor_read(heads, 0, heads_host,
                            q_count * sizeof(float))) {
        rc = 0;
        for (uint32_t t = 0; t < n_tokens && rc == 0; t++) {
            for (uint32_t h = 0; h < n_head; h++) {
                const float v =
                    heads_host[((uint64_t)t * n_head + h) * head_dim];
                if (!isfinite(v) || v < 0.85f) {
                    fprintf(stderr,
                            "Pro indexed attention dropped selected rows token=%u head=%u value=%f\n",
                            t, h, (double)v);
                    rc = 1;
                    break;
                }
            }
        }
    }

    if (rc == 0) {
        ds4_gpu_attention_decode_row rows[2] = {0};
        const uintptr_t raw_ptr = (uintptr_t)ds4_gpu_tensor_contents(raw);
        const uintptr_t comp_ptr = (uintptr_t)ds4_gpu_tensor_contents(comp);
        const uintptr_t selected_ptr =
            (uintptr_t)ds4_gpu_tensor_contents(selected);
        for (uint32_t t = 0; t < n_tokens; t++) {
            rows[t].raw_kv = raw_ptr;
            rows[t].comp_kv = comp_ptr;
            rows[t].topk = selected_ptr +
                (uint64_t)t * top_k * sizeof(uint32_t);
            rows[t].pos = 4096u + t;
            rows[t].n_raw = n_raw;
            rows[t].raw_cap = n_raw;
            rows[t].raw_start = 0u;
            rows[t].n_comp = n_comp;
            rows[t].top_k = top_k;
            rows[t].window = n_raw;
            rows[t].ratio = 4u;
            rows[t].indexed = 1u;
        }
        if (!raw_ptr || !comp_ptr || !selected_ptr ||
            !ds4_gpu_attention_decode_rows_rope_tensor(
                heads, sinks, n_head * sizeof(float), 0, q, rows, n_tokens,
                n_head, head_dim, 2, 4096, 10000.0f, 1.0f, 0.0f, 1.0f,
                32.0f, 1.0f) ||
            !ds4_gpu_synchronize() ||
            !ds4_gpu_tensor_read(heads, 0, heads_host,
                                 q_count * sizeof(float))) {
            fprintf(stderr, "Pro grouped indexed attention launch failed\n");
            rc = 1;
        }
        for (uint32_t t = 0; t < n_tokens && rc == 0; t++) {
            for (uint32_t h = 0; h < n_head; h++) {
                const uint64_t off =
                    ((uint64_t)t * n_head + h) * head_dim;
                const float norm = sqrtf(heads_host[off] * heads_host[off] +
                                         heads_host[off + 1u] *
                                             heads_host[off + 1u]);
                if (!isfinite(norm) || norm < 0.85f) {
                    fprintf(stderr,
                            "Pro grouped indexed attention dropped selected rows token=%u head=%u norm=%f\n",
                            t, h, (double)norm);
                    rc = 1;
                    break;
                }
            }
        }
    }

    ds4_gpu_tensor_free(selected);
    ds4_gpu_tensor_free(comp);
    ds4_gpu_tensor_free(raw);
    ds4_gpu_tensor_free(q);
    ds4_gpu_tensor_free(heads);
    free(selected_host);
    free(heads_host);
    free(comp_host);
    free(raw_host);
    free(q_host);
    free(sinks);
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
    if (check_pro_topk() != 0) rc = 1;
    if (check_pro_indexed_attention() != 0) rc = 1;
    if (check_decode_attention_overflow_path() != 0) rc = 1;
    ds4_gpu_cleanup();
    if (rc == 0) puts("cuda long-context regression: OK");
    return rc;
}
