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

static int check_streaming_expert_cache(void) {
    enum {
        N_EXPERT = 4,
        GATE_EXPERT_BYTES = 64,
        DOWN_EXPERT_BYTES = 32,
        EXPERT_BYTES = 2 * GATE_EXPERT_BYTES + DOWN_EXPERT_BYTES,
        MODEL_BYTES = N_EXPERT * EXPERT_BYTES,
        GATE_EXPERT_BYTES2 = 96,
        DOWN_EXPERT_BYTES2 = 64,
        EXPERT_BYTES2 = 2 * GATE_EXPERT_BYTES2 + DOWN_EXPERT_BYTES2,
        MODEL_BYTES2 = N_EXPERT * EXPERT_BYTES2
    };
    unsigned char model[MODEL_BYTES];
    unsigned char model2[MODEL_BYTES2];
    for (size_t i = 0; i < sizeof(model); i++) model[i] = (unsigned char)i;
    for (size_t i = 0; i < sizeof(model2); i++) model2[i] = (unsigned char)(i + 1);

    const ds4_gpu_stream_expert_table table = {
        .model_map = model,
        .model_size = sizeof(model),
        .layer = 0,
        .n_total_expert = N_EXPERT,
        .gate_offset = 0,
        .up_offset = N_EXPERT * GATE_EXPERT_BYTES,
        .down_offset = 2 * N_EXPERT * GATE_EXPERT_BYTES,
        .gate_expert_bytes = GATE_EXPERT_BYTES,
        .down_expert_bytes = DOWN_EXPERT_BYTES,
    };
    const ds4_gpu_stream_expert_table table2 = {
        .model_map = model2,
        .model_size = sizeof(model2),
        .layer = 1,
        .n_total_expert = N_EXPERT,
        .gate_offset = 0,
        .up_offset = N_EXPERT * GATE_EXPERT_BYTES2,
        .down_offset = 2 * N_EXPERT * GATE_EXPERT_BYTES2,
        .gate_expert_bytes = GATE_EXPERT_BYTES2,
        .down_expert_bytes = DOWN_EXPERT_BYTES2,
    };
    const int32_t first[] = {0, 1};
    const int32_t hit[] = {0};
    const int32_t miss[] = {2};
    const int32_t selected[] = {0, 1, 0, 1};
    ds4_gpu_stream_expert_table invalid = table;
    invalid.down_offset = sizeof(model);
    int rc = 1;

    ds4_gpu_set_ssd_streaming(true);
    ds4_gpu_set_streaming_expert_cache_budget(2);
    ds4_gpu_set_streaming_expert_cache_expert_bytes(EXPERT_BYTES);
    if (ds4_gpu_stream_expert_cache_configured_count() != 2 ||
        ds4_gpu_stream_expert_cache_budget_for_expert_size(
                GATE_EXPERT_BYTES, DOWN_EXPERT_BYTES) != 2 ||
        !ds4_gpu_stream_expert_cache_seed_selected(&table, first, 2) ||
        ds4_gpu_stream_expert_cache_current_count() != 2 ||
        !ds4_gpu_stream_expert_cache_seed_selected(&table, hit, 1) ||
        ds4_gpu_stream_expert_cache_current_count() != 2 ||
        !ds4_gpu_stream_expert_cache_prepare_selected_batch(
                &table, selected, 2, 2) ||
        !ds4_gpu_stream_expert_cache_seed_selected(&table, miss, 1) ||
        ds4_gpu_stream_expert_cache_current_count() != 2 ||
        ds4_gpu_stream_expert_cache_seed_selected(&invalid, hit, 1)) {
        fprintf(stderr, "CUDA streaming expert cache regression failed\n");
        goto cleanup;
    }

    ds4_gpu_set_streaming_expert_cache_expert_bytes2(EXPERT_BYTES2);
    ds4_gpu_set_streaming_expert_cache_budget2(2);
    if (ds4_gpu_stream_expert_cache_configured_count() != 4 ||
        ds4_gpu_stream_expert_cache_budget_for_expert_size(
                GATE_EXPERT_BYTES2, DOWN_EXPERT_BYTES2) != 2 ||
        !ds4_gpu_stream_expert_cache_seed_selected(&table2, first, 2) ||
        ds4_gpu_stream_expert_cache_current_count() != 4) {
        fprintf(stderr, "CUDA streaming secondary expert cache regression failed\n");
        goto cleanup;
    }

    ds4_gpu_stream_expert_cache_release_resident();
    if (ds4_gpu_stream_expert_cache_current_count() != 0 ||
        ds4_gpu_stream_expert_cache_configured_count() != 4) {
        fprintf(stderr, "CUDA streaming expert cache release regression failed\n");
        goto cleanup;
    }
    rc = 0;

cleanup:
    ds4_gpu_stream_expert_cache_release_resident();
    ds4_gpu_set_streaming_expert_cache_budget(0);
    ds4_gpu_set_streaming_expert_cache_expert_bytes(0);
    ds4_gpu_set_streaming_expert_cache_budget2(0);
    ds4_gpu_set_streaming_expert_cache_expert_bytes2(0);
    ds4_gpu_set_ssd_streaming(false);
    return rc;
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
    int rc = check_streaming_expert_cache();
    if (check_large_topk() != 0) rc = 1;
    if (check_decode_attention_overflow_path() != 0) rc = 1;
    ds4_gpu_cleanup();
    if (rc == 0) puts("cuda long-context regression: OK");
    return rc;
}
