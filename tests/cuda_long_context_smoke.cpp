#include "ds4_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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
                                              0,
                                              0,
                                              n_comp,
                                              NULL,
                                              0,
                                              n_head,
                                              head_dim,
                                              0) &&
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

static int check_dspark_non_causal_attention(void) {
    const uint32_t n_head = 8;
    const uint32_t head_dim = 512;
    const uint32_t n_tokens = 5;
    const uint32_t n_raw = 5;
    const uint32_t raw_cap = 5;
    const uint64_t q_count = (uint64_t)n_head * head_dim;
    const uint64_t raw_count = (uint64_t)n_raw * head_dim;
    const uint64_t heads_count = (uint64_t)n_tokens * n_head * head_dim;

    float *sinks = (float *)calloc(n_head, sizeof(float));
    float *kvrow = (float *)calloc((size_t)head_dim, sizeof(float));
    float *q_row = (float *)calloc((size_t)n_head * head_dim, sizeof(float));
    float *raw_host = (float *)calloc((size_t)raw_count, sizeof(float));
    float *q_host = (float *)calloc((size_t)q_count * n_tokens, sizeof(float));
    float *heads_causal = (float *)calloc((size_t)heads_count, sizeof(float));
    float *heads_non_causal = (float *)calloc((size_t)heads_count, sizeof(float));
    if (!sinks || !kvrow || !q_row || !raw_host || !q_host || !heads_causal || !heads_non_causal) return 1;

    for (uint32_t d = 0; d < head_dim; d++) {
        kvrow[d] = (float)(d % 16) * 0.1f;
    }
    for (uint32_t d = 0; d < n_head * head_dim; d++) {
        q_row[d] = (float)(d % 8) * 0.2f;
    }
    for (uint32_t t = 0; t < n_raw; t++) {
        memcpy(raw_host + (uint64_t)t * head_dim, kvrow, head_dim * sizeof(float));
    }
    for (uint32_t t = 0; t < n_tokens; t++) {
        memcpy(q_host + (uint64_t)(t * n_head) * head_dim, q_row, (uint64_t)n_head * head_dim * sizeof(float));
    }

    ds4_gpu_tensor *heads_c = ds4_gpu_tensor_alloc(heads_count * sizeof(float));
    ds4_gpu_tensor *heads_nc = ds4_gpu_tensor_alloc(heads_count * sizeof(float));
    ds4_gpu_tensor *q = ds4_gpu_tensor_alloc((uint64_t)n_tokens * q_count * sizeof(float));
    ds4_gpu_tensor *raw = ds4_gpu_tensor_alloc(raw_count * sizeof(float));
    int rc = 1;
    if (heads_c && heads_nc && q && raw &&
        ds4_gpu_tensor_write(q, 0, q_host, (uint64_t)n_tokens * q_count * sizeof(float)) &&
        ds4_gpu_tensor_write(raw, 0, raw_host, raw_count * sizeof(float))) {

        int ok_c = ds4_gpu_attention_decode_raw_batch_heads_tensor(heads_c,
                                                                    sinks,
                                                                    n_head * sizeof(float),
                                                                    0,
                                                                    q,
                                                                    raw,
                                                                    n_tokens, 0,
                                                                    n_raw, raw_cap, 0,
                                                                    0, n_head, head_dim, 0, 0,
                                                                    NULL, NULL, 0, 1);
        int ok_nc = ds4_gpu_attention_decode_raw_batch_heads_tensor(heads_nc,
                                                                     sinks,
                                                                     n_head * sizeof(float),
                                                                     0,
                                                                     q,
                                                                     raw,
                                                                     n_tokens, 0,
                                                                     n_raw, raw_cap, 0,
                                                                     0, n_head, head_dim, 1, 0,
                                                                     NULL, NULL, 0, 1);
        if (ok_c && ok_nc && ds4_gpu_synchronize() &&
            ds4_gpu_tensor_read(heads_c, 0, heads_causal, heads_count * sizeof(float)) &&
            ds4_gpu_tensor_read(heads_nc, 0, heads_non_causal, heads_count * sizeof(float))) {

            int causal_all_equal = 1;
            for (uint32_t t = 1; t < n_tokens; t++) {
                if (memcmp(heads_causal, heads_causal + (uint64_t)t * n_head * head_dim,
                           n_head * head_dim * sizeof(float)) != 0) {
                    causal_all_equal = 0;
                    break;
                }
            }

            int non_causal_all_equal = 1;
            for (uint32_t t = 1; t < n_tokens; t++) {
                if (memcmp(heads_non_causal, heads_non_causal + (uint64_t)t * n_head * head_dim,
                           n_head * head_dim * sizeof(float)) != 0) {
                    non_causal_all_equal = 0;
                    break;
                }
            }

            if (causal_all_equal) {
                fprintf(stderr, "dspark_attn: causal outputs unexpectedly equal\n");
                rc = 1;
            } else if (!non_causal_all_equal) {
                fprintf(stderr, "dspark_attn: non-causal outputs not all equal\n");
                rc = 1;
            } else {
                rc = 0;
            }
        }
    }

    ds4_gpu_tensor_free(raw);
    ds4_gpu_tensor_free(q);
    ds4_gpu_tensor_free(heads_nc);
    ds4_gpu_tensor_free(heads_c);
    free(heads_non_causal);
    free(heads_causal);
    free(q_host);
    free(raw_host);
    free(q_row);
    free(kvrow);
    free(sinks);
    return rc;
}

static int check_dspark_markov_head(void) {
    const uint32_t vocab_size = 4096;
    const uint32_t embed_dim = 256;
    const uint32_t n_draft = 5;
    int32_t prev_tokens[5] = {42, 100, 500, 1200, 3000};

    /* The markov head reads its weights from the model map: stage w1|w2 in one
     * PAGE-ALIGNED buffer so the runtime's per-range cudaHostRegister of the
     * two offsets never overlaps a page (w_bytes is a page multiple here).
     * The buffer is intentionally never freed: its pages stay host-registered
     * until ds4_gpu_cleanup, and munmap of registered pages is UB. */
    const uint64_t w_bytes = (uint64_t)vocab_size * embed_dim * sizeof(float);
    float *map_host = NULL;
    if (posix_memalign((void **)&map_host, 4096, (size_t)(2 * w_bytes)) != 0) return 1;
    memset(map_host, 0, (size_t)(2 * w_bytes));
    float *base_host = (float *)calloc((size_t)vocab_size, sizeof(float));
    if (!map_host || !base_host) return 1;
    float *w1_host = map_host;
    float *w2_host = map_host + (uint64_t)vocab_size * embed_dim;

    for (uint32_t v = 0; v < vocab_size; v++) {
        for (uint32_t i = 0; i < embed_dim; i++) {
            w1_host[(uint64_t)v * embed_dim + i] = (float)((v * 7 + i * 13) % 100) * 0.01f;
            w2_host[(uint64_t)v * embed_dim + i] = (float)((v * 3 + i * 11) % 50) * 0.02f;
        }
        base_host[v] = (float)(v % 200) * 0.01f;
    }

    ds4_gpu_tensor *base = ds4_gpu_tensor_alloc((uint64_t)vocab_size * sizeof(float));
    ds4_gpu_tensor *ref_logits = ds4_gpu_tensor_alloc((uint64_t)vocab_size * sizeof(float));
    int rc = 1;
    if (base && ref_logits &&
        ds4_gpu_tensor_write(base, 0, base_host, (uint64_t)vocab_size * sizeof(float))) {

        int32_t id = prev_tokens[0];
        for (uint32_t step = 0; step < n_draft; step++) {
            int32_t gpu_id = 0;
            if (!ds4_gpu_dspark_markov_step_model(ref_logits, &gpu_id, NULL, base,
                                                  map_host, 2 * w_bytes,
                                                  0, w_bytes,
                                                  id, vocab_size, embed_dim)) {
                rc = 1;
                goto cleanup;
            }

            const float *embed = w1_host + (uint64_t)id * embed_dim;
            float cpu_best = -1e30f;
            int32_t cpu_id = 0;
            for (uint32_t v = 0; v < vocab_size; v++) {
                float dot = 0.0f;
                const float *w2r = w2_host + (uint64_t)v * embed_dim;
                for (uint32_t i = 0; i < embed_dim; i++)
                    dot += w2r[i] * embed[i];
                float val = base_host[v] + dot;
                if (val > cpu_best) { cpu_best = val; cpu_id = (int32_t)v; }
            }

            if (gpu_id != cpu_id) {
                fprintf(stderr, "markov step %u: GPU=%d CPU=%d\n", step, gpu_id, cpu_id);
                rc = 1;
                goto cleanup;
            }
            id = gpu_id;
        }
        rc = 0;
    }

cleanup:
    ds4_gpu_tensor_free(ref_logits);
    ds4_gpu_tensor_free(base);
    free(base_host);
    /* map_host intentionally leaked (host-registered; see above). */
    return rc;
}

static int check_dspark_confidence_head(void) {
    const uint32_t n_positions = 3;
    const uint32_t hidden_dim = 32;
    const uint32_t embed_dim = 8;
    const uint32_t total_dim = hidden_dim + embed_dim;
    const uint32_t vocab_size = 16;
    const int32_t token_ids_host[3] = {3, 7, 12};

    /* The confidence head gathers its embedding row from markov_w1 by token id
     * and reads w1/proj from the model map.  Page-align the buffer and place
     * proj on its own page so the runtime's per-range cudaHostRegister calls
     * never overlap; intentionally never freed (registered pages, see the
     * markov check). */
    const uint64_t proj_offset = 4096;   /* w1 (512 B) fits below it */
    float *map_host = NULL;
    if (posix_memalign((void **)&map_host, 4096,
                       (size_t)(proj_offset + total_dim * sizeof(float))) != 0) return 1;
    memset(map_host, 0, (size_t)(proj_offset + total_dim * sizeof(float)));
    float *hidden_host = (float *)calloc((size_t)n_positions * hidden_dim, sizeof(float));
    float scores_host[3];
    if (!map_host || !hidden_host) return 1;
    float *w1_host = map_host;
    float *proj_host = (float *)((char *)map_host + proj_offset);

    for (uint32_t v = 0; v < vocab_size; v++)
        for (uint32_t i = 0; i < embed_dim; i++)
            w1_host[(uint64_t)v * embed_dim + i] = (float)((v + i) % 3) * 0.3f;
    for (uint32_t i = 0; i < total_dim; i++)
        proj_host[i] = (float)((i % 7) + 1) * 0.1f;
    for (uint32_t p = 0; p < n_positions; p++) {
        for (uint32_t i = 0; i < hidden_dim; i++)
            hidden_host[(uint64_t)p * hidden_dim + i] = (float)((p + i) % 5) * 0.5f;
    }

    ds4_gpu_tensor *scores = ds4_gpu_tensor_alloc((uint64_t)n_positions * sizeof(float));
    ds4_gpu_tensor *hidden = ds4_gpu_tensor_alloc((uint64_t)n_positions * hidden_dim * sizeof(float));
    ds4_gpu_tensor *token_ids = ds4_gpu_tensor_alloc((uint64_t)n_positions * sizeof(int32_t));
    int rc = 1;
    if (scores && hidden && token_ids &&
        ds4_gpu_tensor_write(hidden, 0, hidden_host, (uint64_t)n_positions * hidden_dim * sizeof(float)) &&
        ds4_gpu_tensor_write(token_ids, 0, token_ids_host, (uint64_t)n_positions * sizeof(int32_t)) &&
        ds4_gpu_dspark_confidence_score_model(scores, hidden, token_ids,
                                              map_host,
                                              proj_offset + total_dim * sizeof(float),
                                              0, proj_offset,
                                              n_positions, hidden_dim, embed_dim,
                                              vocab_size) &&
        ds4_gpu_synchronize() &&
        ds4_gpu_tensor_read(scores, 0, scores_host, (uint64_t)n_positions * sizeof(float))) {
        rc = 0;
        for (uint32_t p = 0; p < n_positions; p++) {
            const float *emb = w1_host + (uint64_t)token_ids_host[p] * embed_dim;
            float dot = 0.0f;
            for (uint32_t i = 0; i < hidden_dim; i++)
                dot += hidden_host[(uint64_t)p * hidden_dim + i] * proj_host[i];
            for (uint32_t i = 0; i < embed_dim; i++)
                dot += emb[i] * proj_host[hidden_dim + i];
            float expected = 1.0f / (1.0f + expf(-dot));
            if (fabsf(scores_host[p] - expected) > 1e-5f) {
                fprintf(stderr, "confidence pos %u: GPU=%.6f CPU=%.6f\n",
                        p, (double)scores_host[p], (double)expected);
                rc = 1;
            }
        }
    }

    ds4_gpu_tensor_free(token_ids);
    ds4_gpu_tensor_free(hidden);
    ds4_gpu_tensor_free(scores);
    free(hidden_host);
    /* map_host intentionally leaked (host-registered; see the markov check). */
    return rc;
}

static int check_dspark_fp8_kv_pack(void) {
    const uint32_t head_dim = 512;
    const uint32_t nblk = (head_dim + 63) / 64;
    float *src = (float *)calloc(head_dim, sizeof(float));
    if (!src) return 1;
    for (uint32_t i = 0; i < head_dim; i++) src[i] = ((float)i - 256.0f) * 0.5f;

    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(head_dim * sizeof(float));
    ds4_gpu_tensor *packed = ds4_gpu_tensor_alloc(head_dim);
    ds4_gpu_tensor *scales = ds4_gpu_tensor_alloc(nblk * sizeof(float));
    if (!x || !packed || !scales) { free(src); return 1; }
    if (!ds4_gpu_tensor_write(x, 0, src, head_dim * sizeof(float))) { free(src); return 1; }

    int rc = 0;
    if (!ds4_gpu_dsv4_fp8_kv_pack_tensor(x, packed, scales, 1, head_dim)) {
        fprintf(stderr, "ds4: fp8 kv pack failed\n");
        rc = 1;
    }
    if (rc == 0) {
        uint8_t *packed_host = (uint8_t *)malloc(head_dim);
        float *scales_host = (float *)malloc(nblk * sizeof(float));
        if (!packed_host || !scales_host) rc = 1;
        else {
            if (!ds4_gpu_tensor_read(packed, 0, packed_host, head_dim))
                rc = 1;
            if (!ds4_gpu_tensor_read(scales, 0, scales_host, nblk * sizeof(float)))
                rc = 1;
            if (rc == 0) {
                for (uint32_t i = 0; i < head_dim; i++) {
                    float scale = scales_host[i >> 6];
                    int idx = packed_host[i] & 0x7f;
                    float val = 0.0f;
                    int exp = (idx >> 3) & 15;
                    int mant = idx & 7;
                    if (exp == 0) val = (float)mant * 0.001953125f;
                    else val = (1.0f + (float)mant * 0.125f) * exp2f((float)exp - 7.0f);
                    if (packed_host[i] & 0x80) val = -val;
                    float recon = val * scale;
                    float diff = fabsf(recon - src[i]);
                    float tol = fmaxf(fabsf(src[i]) * 0.3f + 1e-5f, 1e-4f);
                    if (diff > tol) {
                        fprintf(stderr, "ds4: fp8 packing error at %u: src=%.6f recon=%.6f\n",
                                i, (double)src[i], (double)recon);
                        rc = 1;
                        break;
                    }
                }
            }
            free(packed_host);
            free(scales_host);
        }
    }

    ds4_gpu_tensor_free(scales);
    ds4_gpu_tensor_free(packed);
    ds4_gpu_tensor_free(x);
    free(src);
    return rc;
}

/* ---- Stage 1: MX (microscaling) compressed-KV pack/dequant round-trip ---- */
#define MXKV_FP8 1u
#define MXKV_FP4 2u
#define MXKV_BLK 32u

static float mxkv_e4m3_value(int i) {
    int exp = (i >> 3) & 15, mant = i & 7;
    if (exp == 0) return (float)mant * 0.001953125f;
    return (1.0f + (float)mant * 0.125f) * exp2f((float)exp - 7.0f);
}
static float mxkv_e4m3_snap(float x) { /* returns dequantized magnitude, mirrors device */
    float ax = fminf(fabsf(x), 448.0f);
    int lo = 0, hi = 126;
    while (lo < hi) { int mid = (lo + hi + 1) >> 1; if (mxkv_e4m3_value(mid) <= ax) lo = mid; else hi = mid - 1; }
    int best = lo;
    if (best < 126) {
        float bd = fabsf(ax - mxkv_e4m3_value(best)), nd = fabsf(ax - mxkv_e4m3_value(best + 1));
        if (nd < bd || (nd == bd && (((best + 1) & 1) == 0) && ((best & 1) != 0))) best++;
    }
    return (x < 0.0f ? -1.0f : 1.0f) * mxkv_e4m3_value(best);
}
static float mxkv_e2m1_value(int i) {
    static const float t[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    return t[i & 7];
}
static float mxkv_e2m1_snap(float x) {
    float ax = fminf(fabsf(x), 6.0f);
    int best = 0; float bd = fabsf(ax - mxkv_e2m1_value(0));
    for (int i = 1; i < 8; i++) { float d = fabsf(ax - mxkv_e2m1_value(i));
        if (d < bd || (d == bd && ((i & 1) == 0) && ((best & 1) != 0))) { best = i; bd = d; } }
    return (x < 0.0f ? -1.0f : 1.0f) * mxkv_e2m1_value(best);
}
static float mxkv_ref_scale(float amax, float max_repr) {
    float ratio = fmaxf(amax, 1.0e-20f) / max_repr;
    int k = (int)ceilf(log2f(ratio)), e = k + 127;
    if (e < 0) e = 0;
    if (e > 254) e = 254;
    return exp2f((float)(e - 127));
}

static int mxkv_run_case(uint32_t fmt, uint32_t n_tok, uint32_t head_dim, const float *src) {
    const uint32_t nblk = head_dim / MXKV_BLK;
    const uint32_t rowbytes = (fmt == MXKV_FP4 ? head_dim / 2 : head_dim) + nblk;
    const float max_repr = (fmt == MXKV_FP4) ? 6.0f : 448.0f;

    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc((uint64_t)n_tok * head_dim * sizeof(float));
    ds4_gpu_tensor *packed = ds4_gpu_tensor_alloc((uint64_t)n_tok * rowbytes);
    ds4_gpu_tensor *deq = ds4_gpu_tensor_alloc((uint64_t)n_tok * head_dim * sizeof(float));
    float *gpu_deq = (float *)malloc((size_t)n_tok * head_dim * sizeof(float));
    int rc = 1;
    if (x && packed && deq && gpu_deq &&
        ds4_gpu_tensor_write(x, 0, src, (uint64_t)n_tok * head_dim * sizeof(float)) &&
        ds4_gpu_mxkv_pack_tensor(x, packed, fmt, n_tok, head_dim) &&
        ds4_gpu_mxkv_dequant_tensor(packed, deq, fmt, n_tok, head_dim) &&
        ds4_gpu_tensor_read(deq, 0, gpu_deq, (uint64_t)n_tok * head_dim * sizeof(float))) {
        rc = 0;
        double sum_abs = 0.0, sum_ref = 0.0; float worst = 0.0f;
        for (uint32_t r = 0; r < n_tok && rc == 0; r++) {
            for (uint32_t b = 0; b < nblk; b++) {
                float amax = 0.0f;
                for (uint32_t j = 0; j < MXKV_BLK; j++) amax = fmaxf(amax, fabsf(src[r * head_dim + b * MXKV_BLK + j]));
                float scale = mxkv_ref_scale(amax, max_repr);
                for (uint32_t j = 0; j < MXKV_BLK; j++) {
                    uint32_t d = b * MXKV_BLK + j;
                    float in = src[r * head_dim + d];
                    float ref = (fmt == MXKV_FP4 ? mxkv_e2m1_snap(in / scale) : mxkv_e4m3_snap(in / scale)) * scale;
                    float got = gpu_deq[r * head_dim + d];
                    if (fabsf(got - ref) > 1e-3f) {
                        fprintf(stderr, "mxkv fmt=%u mismatch @row%u d%u: gpu=%.6f ref=%.6f (in=%.6f scale=%.6g)\n",
                                fmt, r, d, (double)got, (double)ref, (double)in, (double)scale);
                        rc = 1; break;
                    }
                    sum_abs += fabs((double)got - (double)in);
                    sum_ref += fabs((double)in);
                    worst = fmaxf(worst, fabsf(got - in));
                }
            }
        }
        if (rc == 0) {
            double rel = sum_ref > 0 ? sum_abs / sum_ref : 0.0;
            /* FP8 ~<3% mean rel error, FP4 much coarser but bounded. */
            double bound = (fmt == MXKV_FP4) ? 0.20 : 0.05;
            printf("  mxkv fmt=%u rowbytes=%u mean_rel_err=%.4f worst_abs=%.4f -> %s\n",
                   fmt, rowbytes, rel, (double)worst, rel <= bound ? "OK" : "HIGH");
            if (rel > bound) rc = 1;
        }
    }
    ds4_gpu_tensor_free(deq); ds4_gpu_tensor_free(packed); ds4_gpu_tensor_free(x);
    free(gpu_deq);
    return rc;
}

static int check_mxkv_roundtrip(void) {
    const uint32_t n_tok = 48, head_dim = 512;
    float *src = (float *)malloc((size_t)n_tok * head_dim * sizeof(float));
    if (!src) return 1;
    /* Vary magnitude across blocks and rows so E8M0 per-block scaling is exercised. */
    for (uint32_t r = 0; r < n_tok; r++)
        for (uint32_t d = 0; d < head_dim; d++) {
            float blkscale = exp2f((float)((int)(d / MXKV_BLK % 12) - 6));   /* 2^-6 .. 2^5 */
            float v = sinf((float)(d * 3 + r * 7) * 0.031f) * blkscale * (1.0f + 0.1f * (float)r);
            src[r * head_dim + d] = v;
        }
    int rc = 0;
    if (mxkv_run_case(MXKV_FP8, n_tok, head_dim, src) != 0) rc = 1;
    if (mxkv_run_case(MXKV_FP4, n_tok, head_dim, src) != 0) rc = 1;
    free(src);
    return rc;
}

/* MXKV gather primitive: dequant selected rows (contiguous and transposed). */
static int check_mxkv_gather(void) {
    const uint32_t n_cap = 100, n_sel = 32, head_dim = 512, nblk = head_dim / MXKV_BLK;
    const uint32_t rowbytes = head_dim + nblk;   /* MXFP8 */
    float *src = (float *)malloc((size_t)n_cap * head_dim * sizeof(float));
    int32_t *rows = (int32_t *)malloc(n_sel * sizeof(int32_t));
    if (!src || !rows) return 1;
    for (uint32_t r = 0; r < n_cap; r++)
        for (uint32_t d = 0; d < head_dim; d++)
            src[r * head_dim + d] = sinf((float)(r * 5 + d) * 0.02f) * (1.0f + 0.05f * (float)r);
    for (uint32_t i = 0; i < n_sel; i++) rows[i] = (int32_t)((i * 37 + 11) % n_cap);   /* scattered picks */

    ds4_gpu_tensor *xf = ds4_gpu_tensor_alloc((uint64_t)n_cap * head_dim * sizeof(float));
    ds4_gpu_tensor *cache = ds4_gpu_tensor_alloc((uint64_t)n_cap * rowbytes);
    ds4_gpu_tensor *idx = ds4_gpu_tensor_alloc((uint64_t)n_sel * sizeof(int32_t));
    ds4_gpu_tensor *g0 = ds4_gpu_tensor_alloc((uint64_t)n_sel * head_dim * sizeof(float));
    ds4_gpu_tensor *g1 = ds4_gpu_tensor_alloc((uint64_t)n_sel * head_dim * sizeof(float));
    float *out0 = (float *)malloc((size_t)n_sel * head_dim * sizeof(float));
    float *out1 = (float *)malloc((size_t)n_sel * head_dim * sizeof(float));
    int rc = 1;
    if (xf && cache && idx && g0 && g1 && out0 && out1 &&
        ds4_gpu_tensor_write(xf, 0, src, (uint64_t)n_cap * head_dim * sizeof(float)) &&
        ds4_gpu_tensor_write(idx, 0, rows, (uint64_t)n_sel * sizeof(int32_t)) &&
        ds4_gpu_mxkv_pack_tensor(xf, cache, MXKV_FP8, n_cap, head_dim) &&
        ds4_gpu_mxkv_gather_dequant_tensor(cache, g0, idx, n_sel, n_cap, head_dim, MXKV_FP8, 0) &&
        ds4_gpu_mxkv_gather_dequant_tensor(cache, g1, idx, n_sel, n_cap, head_dim, MXKV_FP8, 1) &&
        ds4_gpu_tensor_read(g0, 0, out0, (uint64_t)n_sel * head_dim * sizeof(float)) &&
        ds4_gpu_tensor_read(g1, 0, out1, (uint64_t)n_sel * head_dim * sizeof(float))) {
        rc = 0;
        for (uint32_t i = 0; i < n_sel && rc == 0; i++) {
            const float *sr = src + (size_t)rows[i] * head_dim;
            for (uint32_t b = 0; b < nblk; b++) {
                float amax = 0.0f;
                for (uint32_t j = 0; j < MXKV_BLK; j++) amax = fmaxf(amax, fabsf(sr[b * MXKV_BLK + j]));
                float scale = mxkv_ref_scale(amax, 448.0f);
                for (uint32_t j = 0; j < MXKV_BLK; j++) {
                    uint32_t d = b * MXKV_BLK + j;
                    float ref = mxkv_e4m3_snap(sr[d] / scale) * scale;
                    if (fabsf(out0[(size_t)i * head_dim + d] - ref) > 1e-3f ||    /* contiguous */
                        fabsf(out1[(size_t)d * n_sel + i] - ref) > 1e-3f) {       /* transposed */
                        fprintf(stderr, "mxkv gather mismatch row%u d%u\n", i, d);
                        rc = 1; break;
                    }
                }
            }
        }
        if (rc == 0) printf("  mxkv gather n_sel=%u (contiguous + transposed) -> OK\n", n_sel);
    }
    ds4_gpu_tensor_free(g1); ds4_gpu_tensor_free(g0); ds4_gpu_tensor_free(idx);
    ds4_gpu_tensor_free(cache); ds4_gpu_tensor_free(xf);
    free(src); free(rows); free(out0); free(out1);
    return rc;
}

/* ---------------------------------------------------------------------------
 * Multi-bank descriptor attention smoke (Tier-2 step 2).
 *
 * Two banks of a raw-ring + compressed-cache pool are filled with different
 * pseudorandom sequences, then the descriptor-aware decode attention runs
 * rows from both banks (mixed seq_ids, per-row positions incl. a wrapped
 * ring and an EMIT-boundary row, qpos ≡ ratio-1 mod ratio) in ONE launch.
 * Every row must be BYTE-identical to the same computation run
 * single-session (scalar mode against that bank's slab view) with the
 * ENGINE-TRUE scalar n_comp — the engine emits a step's compressed row
 * BEFORE attention, so at position qpos the live engine passes
 * (qpos+1)/ratio, and the banked per-row derivation must reproduce exactly
 * that (a floor(qpos/ratio) derivation fails the emit-boundary rows below).
 * Rows must also be independent of batch composition (row permutation).
 * Attention here is deterministic per row: one block per (row, head), so
 * per-row reduction order cannot depend on batchmates.
 * ------------------------------------------------------------------------- */

static uint32_t mb_rng_state = 0x12345u;
static float mb_rand(void) {
    mb_rng_state = mb_rng_state * 1664525u + 1013904223u;
    return ((float)(mb_rng_state >> 9) / 8388608.0f) - 1.0f;   /* [-1, 1) */
}

typedef struct {
    uint32_t bank;
    uint32_t qpos;
    uint32_t ref_n_comp;   /* scalar n_comp for the single-session reference:
                            * ENGINE-TRUE frontier at qpos, i.e. (qpos+1)/ratio
                            * (emit-before-attention), capped by the bank's
                            * comp capacity.  Exception: the heads8-online case
                            * passes the cross-bank superset to pin kernel
                            * selection (see the comment there). */
} mb_row;

/* One descriptor launch over `rows`, then per-row single-session reference
 * launches against bank views; byte-compare.  indexed != 0 exercises the
 * indexed (top-k) entry instead of the mixed entry. */
static int mb_run_case(const char *label,
                       const mb_row *rows, uint32_t n_rows,
                       ds4_gpu_tensor *raw_slab, uint32_t raw_cap,
                       ds4_gpu_tensor *comp_slab, uint32_t comp_cap,
                       uint32_t n_comp_superset, uint32_t window,
                       uint32_t ratio, uint32_t n_banks,
                       const float *sinks, uint32_t n_head, uint32_t head_dim,
                       int indexed, const int32_t *topk_host, uint32_t top_k) {
    const uint64_t row_f32 = (uint64_t)n_head * head_dim;
    const uint64_t q_count = (uint64_t)n_rows * row_f32;
    float *q_host = (float *)malloc(q_count * sizeof(float));
    float *out_batch = (float *)malloc(q_count * sizeof(float));
    float *out_ref = (float *)malloc(row_f32 * sizeof(float));
    int32_t *pos_host = (int32_t *)malloc(n_rows * sizeof(int32_t));
    int32_t *sid_host = (int32_t *)malloc(n_rows * sizeof(int32_t));
    if (!q_host || !out_batch || !out_ref || !pos_host || !sid_host) {
        free(sid_host); free(pos_host); free(out_ref); free(out_batch); free(q_host);
        return 1;
    }
    mb_rng_state = 0xbeef1234u;
    for (uint64_t i = 0; i < q_count; i++) q_host[i] = mb_rand() * 0.25f;
    for (uint32_t r = 0; r < n_rows; r++) {
        pos_host[r] = (int32_t)rows[r].qpos;
        sid_host[r] = (int32_t)rows[r].bank;
    }

    ds4_gpu_tensor *q = ds4_gpu_tensor_alloc(q_count * sizeof(float));
    ds4_gpu_tensor *heads = ds4_gpu_tensor_alloc(q_count * sizeof(float));
    ds4_gpu_tensor *positions = ds4_gpu_tensor_alloc(n_rows * sizeof(int32_t));
    ds4_gpu_tensor *seq_id = ds4_gpu_tensor_alloc(n_rows * sizeof(int32_t));
    ds4_gpu_tensor *topk = NULL;
    int rc = 1;
    if (!q || !heads || !positions || !seq_id ||
        !ds4_gpu_tensor_write(q, 0, q_host, q_count * sizeof(float)) ||
        !ds4_gpu_tensor_write(positions, 0, pos_host, n_rows * sizeof(int32_t)) ||
        !ds4_gpu_tensor_write(seq_id, 0, sid_host, n_rows * sizeof(int32_t))) goto done;
    if (indexed) {
        topk = ds4_gpu_tensor_alloc((uint64_t)n_rows * top_k * sizeof(int32_t));
        if (!topk || !ds4_gpu_tensor_write(topk, 0, topk_host,
                                           (uint64_t)n_rows * top_k * sizeof(int32_t))) goto done;
    }

    /* Descriptor launch: all rows, mixed banks, one call. */
    {
        int ok;
        if (indexed) {
            ok = ds4_gpu_attention_indexed_mixed_batch_heads_tensor(
                    heads, sinks, (uint64_t)n_head * sizeof(float), 0,
                    q, raw_slab, comp_slab, 0, 0, 0, topk,
                    n_rows, 0, window, raw_cap, 0,
                    n_comp_superset, top_k, window, ratio, n_head, head_dim, 0,
                    positions, seq_id, NULL, comp_cap, n_banks);
        } else {
            ok = ds4_gpu_attention_decode_mixed_batch_heads_tensor(
                    heads, sinks, (uint64_t)n_head * sizeof(float), 0,
                    q, raw_slab, n_comp_superset ? comp_slab : NULL, 0, 0, 0,
                    NULL, 0, n_rows, 0, window, raw_cap, 0,
                    n_comp_superset, window, ratio, n_head, head_dim, 0, 0,
                    positions, seq_id, NULL, comp_cap, n_banks);
        }
        if (!ok || !ds4_gpu_synchronize() ||
            !ds4_gpu_tensor_read(heads, 0, out_batch, q_count * sizeof(float))) {
            fprintf(stderr, "multibank %s: descriptor launch failed\n", label);
            goto done;
        }
    }

    /* Per-row single-session references against the bank's slab view.
     *
     * The reference is a classic scalar-mode BATCH of the 3 consecutive
     * positions ending at the row's qpos (NULL descriptors — the unchanged
     * single-session verify-batch shape), and we compare the last row.  A
     * 1-row reference would be wrong-by-construction: the generic kernel's
     * score pass legitimately switches to a cheaper loop when the LAUNCH is
     * single-token (n_tokens == 1), and that loop's float accumulation order
     * differs.  Within multi-row launches the per-row math depends only on
     * (bank, qpos) — which is exactly the batch-composition independence this
     * smoke asserts byte-for-byte. */
    for (uint32_t r = 0; r < n_rows; r++) {
        const mb_row *row = &rows[r];
        const uint32_t ref_rows = 3;
        const uint64_t raw_bank_bytes = (uint64_t)raw_cap * head_dim * sizeof(float);
        const uint64_t comp_bank_bytes = (uint64_t)comp_cap * head_dim * sizeof(float);
        ds4_gpu_tensor *raw_view = ds4_gpu_tensor_view(
                raw_slab, (uint64_t)row->bank * raw_bank_bytes, raw_bank_bytes);
        ds4_gpu_tensor *comp_view = comp_slab
            ? ds4_gpu_tensor_view(comp_slab, (uint64_t)row->bank * comp_bank_bytes,
                                  comp_bank_bytes)
            : NULL;
        ds4_gpu_tensor *q_ref = ds4_gpu_tensor_alloc((uint64_t)ref_rows * row_f32 * sizeof(float));
        ds4_gpu_tensor *h_ref = ds4_gpu_tensor_alloc((uint64_t)ref_rows * row_f32 * sizeof(float));
        ds4_gpu_tensor *tk_ref = indexed
            ? ds4_gpu_tensor_alloc((uint64_t)ref_rows * top_k * sizeof(int32_t))
            : NULL;
        float *q_ref_host = (float *)malloc((uint64_t)ref_rows * row_f32 * sizeof(float));
        int32_t *tk_ref_host = indexed
            ? (int32_t *)malloc((uint64_t)ref_rows * top_k * sizeof(int32_t))
            : NULL;
        int ok = raw_view && q_ref && h_ref && q_ref_host &&
                 (!comp_slab || comp_view) &&
                 (!indexed || (tk_ref && tk_ref_host));
        if (ok) {
            /* rows 0..1 are position-fillers; row 2 is the row under test. */
            for (uint64_t i = 0; i < (uint64_t)(ref_rows - 1) * row_f32; i++)
                q_ref_host[i] = 0.01f;
            memcpy(q_ref_host + (uint64_t)(ref_rows - 1) * row_f32,
                   q_host + (uint64_t)r * row_f32, row_f32 * sizeof(float));
            ok = ds4_gpu_tensor_write(q_ref, 0, q_ref_host,
                                      (uint64_t)ref_rows * row_f32 * sizeof(float));
            if (ok && indexed) {
                for (uint32_t rr = 0; rr < ref_rows; rr++)
                    memcpy(tk_ref_host + (uint64_t)rr * top_k,
                           topk_host + (uint64_t)r * top_k, top_k * sizeof(int32_t));
                ok = ds4_gpu_tensor_write(tk_ref, 0, tk_ref_host,
                                          (uint64_t)ref_rows * top_k * sizeof(int32_t));
            }
        }
        const uint32_t n_raw = (row->qpos + 1u < window) ? row->qpos + 1u : window;
        const uint32_t raw_start = (row->qpos + 1u - n_raw) % raw_cap;
        const uint32_t pos0 = row->qpos - (ref_rows - 1u);
        if (ok) {
            if (indexed) {
                ok = ds4_gpu_attention_indexed_mixed_batch_heads_tensor(
                        h_ref, sinks, (uint64_t)n_head * sizeof(float), 0,
                        q_ref, raw_view, comp_view, 0, 0, 0, tk_ref,
                        ref_rows, pos0, n_raw, raw_cap, raw_start,
                        row->ref_n_comp, top_k, window, ratio, n_head, head_dim, 0,
                        NULL, NULL, NULL, 0, 1);
            } else {
                ok = ds4_gpu_attention_decode_mixed_batch_heads_tensor(
                        h_ref, sinks, (uint64_t)n_head * sizeof(float), 0,
                        q_ref, raw_view, row->ref_n_comp ? comp_view : NULL,
                        0, 0, 0, NULL, 0,
                        ref_rows, pos0, n_raw, raw_cap, raw_start,
                        row->ref_n_comp, window, ratio, n_head, head_dim, 0, 0,
                        NULL, NULL, NULL, 0, 1);
            }
        }
        ok = ok && ds4_gpu_synchronize() &&
             ds4_gpu_tensor_read(h_ref, (uint64_t)(ref_rows - 1) * row_f32 * sizeof(float),
                                 out_ref, row_f32 * sizeof(float));
        free(tk_ref_host);
        free(q_ref_host);
        ds4_gpu_tensor_free(tk_ref);
        ds4_gpu_tensor_free(h_ref);
        ds4_gpu_tensor_free(q_ref);
        ds4_gpu_tensor_free(comp_view);
        ds4_gpu_tensor_free(raw_view);
        if (!ok) {
            fprintf(stderr, "multibank %s: reference launch failed (row %u)\n", label, r);
            goto done;
        }
        if (memcmp(out_ref, out_batch + (uint64_t)r * row_f32,
                   row_f32 * sizeof(float)) != 0) {
            fprintf(stderr, "multibank %s: row %u (bank %u pos %u) != single-session\n",
                    label, r, row->bank, row->qpos);
            goto done;
        }
    }
    printf("  multibank %s: %u rows across %u banks -> byte-identical\n",
           label, n_rows, n_banks);
    rc = 0;

done:
    ds4_gpu_tensor_free(topk);
    ds4_gpu_tensor_free(seq_id);
    ds4_gpu_tensor_free(positions);
    ds4_gpu_tensor_free(heads);
    ds4_gpu_tensor_free(q);
    free(sid_host); free(pos_host); free(out_ref); free(out_batch); free(q_host);
    return rc;
}

static int check_multibank_decode_attention(void) {
    const uint32_t n_head = 8, head_dim = 512;
    const uint32_t raw_cap = 64, window = 32, ratio = 4, n_banks = 2;
    const uint32_t comp_cap = 32;               /* per-bank comp-row stride */
    /* static: the runtime host-registers the model-map page (never freed). */
    static float sinks[8];
    for (uint32_t h = 0; h < n_head; h++) sinks[h] = 0.1f * (float)h;

    const uint64_t raw_count = (uint64_t)n_banks * raw_cap * head_dim;
    const uint64_t comp_count = (uint64_t)n_banks * comp_cap * head_dim;
    float *raw_host = (float *)malloc(raw_count * sizeof(float));
    float *comp_host = (float *)malloc(comp_count * sizeof(float));
    if (!raw_host || !comp_host) {
        free(comp_host); free(raw_host);
        return 1;
    }
    mb_rng_state = 0xace1u;
    for (uint64_t i = 0; i < raw_count; i++) raw_host[i] = mb_rand();
    for (uint64_t i = 0; i < comp_count; i++) comp_host[i] = mb_rand();

    ds4_gpu_tensor *raw_slab = ds4_gpu_tensor_alloc(raw_count * sizeof(float));
    ds4_gpu_tensor *comp_slab = ds4_gpu_tensor_alloc(comp_count * sizeof(float));
    int rc = 1;
    if (!raw_slab || !comp_slab ||
        !ds4_gpu_tensor_write(raw_slab, 0, raw_host, raw_count * sizeof(float)) ||
        !ds4_gpu_tensor_write(comp_slab, 0, comp_host, comp_count * sizeof(float))) goto done;

    {
        /* Generic mixed kernel: bank 0 with a WRAPPED ring (qpos 100 > raw_cap
         * 64), bank 1 at qpos 39 — an EMIT boundary (39 ≡ ratio-1 mod 4) where
         * the engine-true frontier is (39+1)/4 = 10 while a floor(39/4)
         * derivation would see only 9; this row FAILS under the old floor rule
         * (regression teeth for the emit-step divergence bug) — and bank 1
         * again at 37 (non-emit, frontier 9).  References use the engine-true
         * (qpos+1)/4 as scalar n_comp; the batch passes the cross-bank
         * superset 25. */
        const mb_row rows[3] = { {0, 100, 25}, {1, 39, 10}, {1, 37, 9} };
        if (mb_run_case("mixed-generic", rows, 3, raw_slab, raw_cap,
                        comp_slab, comp_cap, 25, window, ratio, n_banks,
                        sinks, n_head, head_dim, 0, NULL, 0) != 0) goto done;
        /* Batch-composition independence: permuted rows, same per-row bytes
         * (mb_run_case re-verifies each row against its reference). */
        const mb_row perm[3] = { {1, 37, 9}, {0, 100, 25}, {1, 39, 10} };
        if (mb_run_case("mixed-generic-permuted", perm, 3, raw_slab, raw_cap,
                        comp_slab, comp_cap, 25, window, ratio, n_banks,
                        sinks, n_head, head_dim, 0, NULL, 0) != 0) goto done;
        /* Raw-only path (n_comp = 0): both banks, one wrapped. */
        const mb_row raw_rows[2] = { {0, 100, 0}, {1, 17, 0} };
        if (mb_run_case("raw-only", raw_rows, 2, raw_slab, raw_cap,
                        NULL, comp_cap, 0, window, ratio, n_banks,
                        sinks, n_head, head_dim, 0, NULL, 0) != 0) goto done;
        /* Indexed (top-k) entry: ids >= the row's visible frontier and -1
         * sentinels must be dropped per row (bank 1 at qpos 39 sees the
         * engine-true 10 visible rows, so ids 12/24 are in-superset but
         * beyond ITS frontier).  Row 1's id 9 is the emit-boundary teeth:
         * exactly the row emitted at step 39 — visible under the engine-true
         * (39+1)/4 = 10 rule, dropped by the old floor(39/4) = 9 rule. */
        const uint32_t top_k = 8;
        const int32_t topk_host[3 * 8] = {
             3, 24,  0, 12, -1,  7, 19,  5,
             8,  2, 12,  9, 24,  0,  6,  4,
             1,  8, -1,  5, 24,  3, 12,  2,
        };
        const mb_row idx_rows[3] = { {0, 100, 25}, {1, 39, 10}, {1, 37, 9} };
        /* Banked mode forces the generic indexed kernel (heads8 variants stay
         * single-bank); pin the scalar reference to the same kernel. */
        setenv("DS4_CUDA_NO_INDEXED_HEADS8", "1", 1);
        const int idx_rc = mb_run_case("indexed-generic", idx_rows, 3, raw_slab, raw_cap,
                                       comp_slab, comp_cap, 25, window, ratio, n_banks,
                                       sinks, n_head, head_dim, 1, topk_host, top_k);
        unsetenv("DS4_CUDA_NO_INDEXED_HEADS8");
        if (idx_rc != 0) goto done;
    }

    /* heads8-online variant: a cross-bank superset n_comp too large for the
     * shared-memory score buffer forces the online kernel for the batch AND
     * for the single-row references.  Bank 0 sits at qpos 45999, an EMIT
     * boundary (45999 ≡ 3 mod 4): its engine-true frontier is 46000/4 =
     * 11500 == the scalar, while floor(45999/4) = 11499 — teeth for the
     * online kernel's per-row rule.  Bank 1 (qpos 20000, frontier 5000)
     * must ALSO get the 11500 scalar in its reference: scalar n_comp picks
     * the kernel (a 5000 scalar fits the score buffer and would take the
     * generic kernel, whose reduction order differs), while the rows a
     * reference actually reads come from the kernel's own (qpos+1)/ratio
     * derivation, which caps at 5000 regardless of the larger scalar. */
    {
        const uint32_t big_comp_cap = 11504;
        const uint32_t big_n_comp = 11500;
        const uint64_t big_count = (uint64_t)n_banks * big_comp_cap * head_dim;
        float *big_host = (float *)malloc(big_count * sizeof(float));
        ds4_gpu_tensor *big_slab = ds4_gpu_tensor_alloc(big_count * sizeof(float));
        int big_rc = 1;
        if (big_host && big_slab) {
            mb_rng_state = 0x5eed5u;
            for (uint64_t i = 0; i < big_count; i++) big_host[i] = mb_rand();
            if (ds4_gpu_tensor_write(big_slab, 0, big_host, big_count * sizeof(float))) {
                const mb_row rows[2] = { {0, 45999, big_n_comp}, {1, 20000, big_n_comp} };
                big_rc = mb_run_case("mixed-online", rows, 2, raw_slab, raw_cap,
                                     big_slab, big_comp_cap, big_n_comp, window,
                                     ratio, n_banks, sinks, n_head, head_dim,
                                     0, NULL, 0);
            }
        }
        ds4_gpu_tensor_free(big_slab);
        free(big_host);
        if (big_rc != 0) goto done;
    }

    /* Dead-row guard teeth: a row with an out-of-pool bank id (-1 sentinel)
     * must produce zero head outputs (fail-visible) while its batchmate is
     * still byte-identical to single-session. */
    {
        const uint64_t row_f32 = (uint64_t)n_head * head_dim;
        const int32_t pos_host2[2] = {100, 37};
        const int32_t sid_host2[2] = {-1, 1};
        float *q_host2 = (float *)malloc(2 * row_f32 * sizeof(float));
        float *out2 = (float *)malloc(2 * row_f32 * sizeof(float));
        ds4_gpu_tensor *q2 = ds4_gpu_tensor_alloc(2 * row_f32 * sizeof(float));
        ds4_gpu_tensor *h2 = ds4_gpu_tensor_alloc(2 * row_f32 * sizeof(float));
        ds4_gpu_tensor *p2 = ds4_gpu_tensor_alloc(2 * sizeof(int32_t));
        ds4_gpu_tensor *s2 = ds4_gpu_tensor_alloc(2 * sizeof(int32_t));
        int dead_rc = 1;
        if (q_host2 && out2 && q2 && h2 && p2 && s2) {
            mb_rng_state = 0xbeef1234u;
            for (uint64_t i = 0; i < 2 * row_f32; i++) q_host2[i] = mb_rand() * 0.25f;
            if (ds4_gpu_tensor_write(q2, 0, q_host2, 2 * row_f32 * sizeof(float)) &&
                ds4_gpu_tensor_write(p2, 0, pos_host2, sizeof(pos_host2)) &&
                ds4_gpu_tensor_write(s2, 0, sid_host2, sizeof(sid_host2)) &&
                ds4_gpu_attention_decode_mixed_batch_heads_tensor(
                        h2, sinks, (uint64_t)n_head * sizeof(float), 0,
                        q2, raw_slab, comp_slab, 0, 0, 0, NULL, 0,
                        2, 0, window, raw_cap, 0,
                        25, window, ratio, n_head, head_dim, 0, 0,
                        p2, s2, NULL, comp_cap, n_banks) &&
                ds4_gpu_synchronize() &&
                ds4_gpu_tensor_read(h2, 0, out2, 2 * row_f32 * sizeof(float))) {
                dead_rc = 0;
                for (uint64_t i = 0; i < row_f32; i++) {
                    if (out2[i] != 0.0f) {
                        fprintf(stderr, "multibank dead-row: nonzero output at %llu\n",
                                (unsigned long long)i);
                        dead_rc = 1;
                        break;
                    }
                }
                if (dead_rc == 0)
                    printf("  multibank dead-row: out-of-pool seq_id -> zero heads\n");
            }
        }
        ds4_gpu_tensor_free(s2); ds4_gpu_tensor_free(p2);
        ds4_gpu_tensor_free(h2); ds4_gpu_tensor_free(q2);
        free(out2); free(q_host2);
        if (dead_rc != 0) goto done;
    }
    rc = 0;

done:
    ds4_gpu_tensor_free(comp_slab);
    ds4_gpu_tensor_free(raw_slab);
    free(comp_host);
    free(raw_host);
    return rc;
}

/* ---------------------------------------------------------------------------
 * Multi-bank descriptor indexer smoke (Tier-2 step 3).
 *
 * The banked indexer scan (generic per-(comp,row) kernel and the n_tokens==1
 * direct-one fast tier) must score each row byte-identically to the same
 * (bank, qpos) scored single-session against that bank's slab view, with the
 * per-row ENGINE-TRUE visible frontier (qpos+1)/ratio — emit-boundary rows
 * (qpos ≡ ratio-1 mod ratio) are the regression teeth for the visibility
 * rule, and the -INF tail beyond a shallow bank's frontier must keep top-k
 * selection inside that bank's own rows.  Dead rows (out-of-pool seq_id)
 * must score -INF everywhere (fail-visible).  Covers both the f32 and the
 * MXFP4-packed (68 B/row) cache formats.
 * ------------------------------------------------------------------------- */

static int mb_idx_run_case(const char *label,
                           const mb_row *rows, uint32_t n_rows,
                           ds4_gpu_tensor *index_slab, uint32_t comp_cap,
                           uint32_t n_comp_superset, uint32_t ratio,
                           uint32_t n_banks, uint32_t n_head, uint32_t head_dim,
                           uint64_t index_bank_bytes, uint32_t top_k) {
    const uint64_t q_row = (uint64_t)n_head * head_dim;
    const float scale = 0.125f;
    float *q_host = (float *)malloc((uint64_t)n_rows * q_row * sizeof(float));
    float *w_host = (float *)malloc((uint64_t)n_rows * n_head * sizeof(float));
    float *out_batch = (float *)malloc((uint64_t)n_rows * n_comp_superset * sizeof(float));
    float *out_ref = (float *)malloc((uint64_t)n_comp_superset * sizeof(float));
    uint32_t *sel_batch = (uint32_t *)malloc((uint64_t)n_rows * top_k * sizeof(uint32_t));
    uint32_t *sel_ref = (uint32_t *)malloc((uint64_t)top_k * sizeof(uint32_t));
    int32_t *pos_host = (int32_t *)malloc(n_rows * sizeof(int32_t));
    int32_t *sid_host = (int32_t *)malloc(n_rows * sizeof(int32_t));
    ds4_gpu_tensor *q = ds4_gpu_tensor_alloc((uint64_t)n_rows * q_row * sizeof(float));
    ds4_gpu_tensor *w = ds4_gpu_tensor_alloc((uint64_t)n_rows * n_head * sizeof(float));
    ds4_gpu_tensor *scores = ds4_gpu_tensor_alloc((uint64_t)n_rows * n_comp_superset * sizeof(float));
    ds4_gpu_tensor *scores_ref = ds4_gpu_tensor_alloc((uint64_t)n_comp_superset * sizeof(float));
    ds4_gpu_tensor *sel = ds4_gpu_tensor_alloc((uint64_t)n_rows * top_k * sizeof(uint32_t));
    ds4_gpu_tensor *sel_ref_t = ds4_gpu_tensor_alloc((uint64_t)top_k * sizeof(uint32_t));
    ds4_gpu_tensor *positions = ds4_gpu_tensor_alloc(n_rows * sizeof(int32_t));
    ds4_gpu_tensor *seq_id = ds4_gpu_tensor_alloc(n_rows * sizeof(int32_t));
    int rc = 1;
    if (!q_host || !w_host || !out_batch || !out_ref || !sel_batch || !sel_ref ||
        !pos_host || !sid_host || !q || !w || !scores || !scores_ref || !sel ||
        !sel_ref_t || !positions || !seq_id) goto done;
    mb_rng_state = 0x1d0b5u;
    for (uint64_t i = 0; i < (uint64_t)n_rows * q_row; i++) q_host[i] = mb_rand() * 0.5f;
    for (uint64_t i = 0; i < (uint64_t)n_rows * n_head; i++) w_host[i] = mb_rand() * 0.5f + 0.75f;
    for (uint32_t r = 0; r < n_rows; r++) {
        pos_host[r] = (int32_t)rows[r].qpos;
        sid_host[r] = (int32_t)rows[r].bank;
    }
    if (!ds4_gpu_tensor_write(q, 0, q_host, (uint64_t)n_rows * q_row * sizeof(float)) ||
        !ds4_gpu_tensor_write(w, 0, w_host, (uint64_t)n_rows * n_head * sizeof(float)) ||
        !ds4_gpu_tensor_write(positions, 0, pos_host, n_rows * sizeof(int32_t)) ||
        !ds4_gpu_tensor_write(seq_id, 0, sid_host, n_rows * sizeof(int32_t))) goto done;

    /* Banked launch: all rows, mixed banks, one call. */
    if (!ds4_gpu_indexer_scores_decode_batch_tensor(scores, q, w, index_slab,
                                                    n_comp_superset, n_rows, 0,
                                                    n_head, head_dim, ratio, scale,
                                                    positions, seq_id, NULL, comp_cap, n_banks) ||
        !ds4_gpu_indexer_topk_tensor(sel, scores, n_comp_superset, n_rows, top_k) ||
        !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(scores, 0, out_batch,
                             (uint64_t)n_rows * n_comp_superset * sizeof(float)) ||
        !ds4_gpu_tensor_read(sel, 0, sel_batch,
                             (uint64_t)n_rows * top_k * sizeof(uint32_t))) {
        fprintf(stderr, "multibank indexer %s: banked launch failed\n", label);
        goto done;
    }

    /* Per-row single-session references against the bank's slab view: the
     * classic scalar path (NULL descriptors) with pos0 = qpos and the SAME
     * superset n_comp (the scalar-path causal clamp is already per-row
     * (pos0+t+1)/ratio, so the -INF tail pattern must match exactly).  A
     * 1-row launch with these head counts stays on the same generic kernel
     * (the direct-one tier requires n_head 64). */
    for (uint32_t r = 0; r < n_rows; r++) {
        const mb_row *row = &rows[r];
        ds4_gpu_tensor *bank_view = ds4_gpu_tensor_view(
                index_slab, (uint64_t)row->bank * index_bank_bytes, index_bank_bytes);
        ds4_gpu_tensor *q_view = ds4_gpu_tensor_view(q, (uint64_t)r * q_row * sizeof(float),
                                                     q_row * sizeof(float));
        ds4_gpu_tensor *w_view = ds4_gpu_tensor_view(w, (uint64_t)r * n_head * sizeof(float),
                                                     (uint64_t)n_head * sizeof(float));
        int ok = bank_view && q_view && w_view &&
                 ds4_gpu_indexer_scores_decode_batch_tensor(scores_ref, q_view, w_view,
                                                            bank_view, n_comp_superset, 1,
                                                            row->qpos, n_head, head_dim,
                                                            ratio, scale,
                                                            NULL, NULL, NULL, 0, 1) &&
                 ds4_gpu_indexer_topk_tensor(sel_ref_t, scores_ref, n_comp_superset, 1, top_k) &&
                 ds4_gpu_synchronize() &&
                 ds4_gpu_tensor_read(scores_ref, 0, out_ref,
                                     (uint64_t)n_comp_superset * sizeof(float)) &&
                 ds4_gpu_tensor_read(sel_ref_t, 0, sel_ref,
                                     (uint64_t)top_k * sizeof(uint32_t));
        ds4_gpu_tensor_free(w_view);
        ds4_gpu_tensor_free(q_view);
        ds4_gpu_tensor_free(bank_view);
        if (!ok) {
            fprintf(stderr, "multibank indexer %s: reference launch failed (row %u)\n", label, r);
            goto done;
        }
        if (memcmp(out_ref, out_batch + (uint64_t)r * n_comp_superset,
                   (uint64_t)n_comp_superset * sizeof(float)) != 0) {
            fprintf(stderr, "multibank indexer %s: scores row %u (bank %u pos %u) != single-session\n",
                    label, r, row->bank, row->qpos);
            goto done;
        }
        if (memcmp(sel_ref, sel_batch + (uint64_t)r * top_k,
                   (uint64_t)top_k * sizeof(uint32_t)) != 0) {
            fprintf(stderr, "multibank indexer %s: top-k row %u (bank %u pos %u) != single-session\n",
                    label, r, row->bank, row->qpos);
            goto done;
        }
        /* Top-k must stay inside the row's own visible frontier whenever it
         * has at least top_k visible rows (shorter rows legitimately carry
         * -INF losers, which the consuming attention kernel drops per row). */
        const uint32_t visible = (row->qpos + 1u) / ratio < n_comp_superset
            ? (row->qpos + 1u) / ratio : n_comp_superset;
        if (visible >= top_k) {
            for (uint32_t k = 0; k < top_k; k++) {
                if (sel_batch[(uint64_t)r * top_k + k] >= visible) {
                    fprintf(stderr, "multibank indexer %s: top-k row %u selected id %u >= visible %u\n",
                            label, r, sel_batch[(uint64_t)r * top_k + k], visible);
                    goto done;
                }
            }
        }
    }
    printf("  multibank indexer %s: %u rows across %u banks -> byte-identical (scores + top-k)\n",
           label, n_rows, n_banks);
    rc = 0;

done:
    ds4_gpu_tensor_free(seq_id); ds4_gpu_tensor_free(positions);
    ds4_gpu_tensor_free(sel_ref_t); ds4_gpu_tensor_free(sel);
    ds4_gpu_tensor_free(scores_ref); ds4_gpu_tensor_free(scores);
    ds4_gpu_tensor_free(w); ds4_gpu_tensor_free(q);
    free(sid_host); free(pos_host); free(sel_ref); free(sel_batch);
    free(out_ref); free(out_batch); free(w_host); free(q_host);
    return rc;
}

static int check_multibank_indexer(void) {
    const uint32_t n_banks = 2, comp_cap = 32, ratio = 4;
    const uint32_t n_comp_sup = 25;   /* cross-bank superset */
    const uint32_t head_dim = 128;
    int rc = 1;

    /* Rows: bank 0 deep (qpos 100 -> visible 25 == superset), bank 1 at the
     * EMIT boundary qpos 39 (visible (39+1)/4 = 10; a floor(39/4) = 9 bug
     * loses score row 9 -> teeth), bank 1 at qpos 37 (visible 9), and bank 1
     * at qpos 17 (visible 4 < top_k 8: the -INF losers case). */
    const mb_row rows[4] = { {0, 100, 0}, {1, 39, 0}, {1, 37, 0}, {1, 17, 0} };

    /* MXFP4-packed cache (production format): synthetic packed rows — 64
     * nibble bytes + 4 E8M0 scale bytes constrained to a sane exponent range
     * (the dequant is well-defined for any codes; references and banked
     * launches read the same bytes). */
    {
        const uint64_t row_bytes = 68;   /* DS4_MXKV_FP4_ROWBYTES(128) */
        const uint64_t bank_bytes = comp_cap * row_bytes;
        const uint64_t total = (uint64_t)n_banks * bank_bytes;
        uint8_t *host = (uint8_t *)malloc(total);
        ds4_gpu_tensor *slab = ds4_gpu_tensor_alloc(total);
        int case_rc = 1;
        if (host && slab) {
            mb_rng_state = 0xf4f4f4u;
            for (uint64_t i = 0; i < total; i++) {
                const uint64_t in_row = i % row_bytes;
                const uint32_t r8 = (uint32_t)(mb_rng_state >> 16) & 0xffu;
                mb_rng_state = mb_rng_state * 1664525u + 1013904223u;
                host[i] = in_row < 64 ? (uint8_t)r8 : (uint8_t)(115u + (r8 & 15u));
            }
            ds4_gpu_indexer_set_fp4(1);
            if (ds4_gpu_tensor_write(slab, 0, host, total)) {
                case_rc = mb_idx_run_case("fp4", rows, 4, slab, comp_cap,
                                          n_comp_sup, ratio, n_banks, 4, head_dim,
                                          bank_bytes, 8);
                /* fp4 direct-one fast tier (n_tokens == 1, n_head 64): the
                 * n_head=4 case above dispatches the generic kernel, so the
                 * banked fp4 DIRECT-ONE tier needs its own row (same slab;
                 * emit-boundary qpos 39 as visibility teeth).  Reference is
                 * the same scalar-mode dispatch on the bank's view. */
                if (case_rc == 0) {
                    const mb_row one[1] = { {1, 39, 0} };
                    case_rc = mb_idx_run_case("fp4-direct-one", one, 1, slab,
                                              comp_cap, n_comp_sup, ratio,
                                              n_banks, 64, head_dim,
                                              bank_bytes, 8);
                }
            }
            ds4_gpu_indexer_set_fp4(0);
        }
        ds4_gpu_tensor_free(slab);
        free(host);
        if (case_rc != 0) goto done;
    }

    /* f32 cache format. */
    {
        const uint64_t bank_bytes = (uint64_t)comp_cap * head_dim * sizeof(float);
        const uint64_t count = (uint64_t)n_banks * comp_cap * head_dim;
        float *host = (float *)malloc(count * sizeof(float));
        ds4_gpu_tensor *slab = ds4_gpu_tensor_alloc(count * sizeof(float));
        int case_rc = 1;
        if (host && slab) {
            mb_rng_state = 0x0f320f3u;
            for (uint64_t i = 0; i < count; i++) host[i] = mb_rand();
            if (ds4_gpu_tensor_write(slab, 0, host, count * sizeof(float))) {
                case_rc = mb_idx_run_case("f32", rows, 4, slab, comp_cap,
                                          n_comp_sup, ratio, n_banks, 4, head_dim,
                                          bank_bytes, 8);
            }
        }
        ds4_gpu_tensor_free(slab);
        free(host);
        if (case_rc != 0) goto done;
    }

    /* Direct-one fast tier (n_tokens == 1, n_head 64): the banked entry must
     * keep the classic single-token reduction order — reference is the SAME
     * scalar-mode dispatch on the bank's view.  Emit-boundary row (qpos 39)
     * as visibility teeth, f32 format. */
    {
        const uint32_t n_head = 64;
        const uint64_t bank_bytes = (uint64_t)comp_cap * head_dim * sizeof(float);
        const uint64_t count = (uint64_t)n_banks * comp_cap * head_dim;
        float *host = (float *)malloc(count * sizeof(float));
        ds4_gpu_tensor *slab = ds4_gpu_tensor_alloc(count * sizeof(float));
        int case_rc = 1;
        if (host && slab) {
            mb_rng_state = 0xd12ec7u;
            for (uint64_t i = 0; i < count; i++) host[i] = mb_rand();
            if (ds4_gpu_tensor_write(slab, 0, host, count * sizeof(float))) {
                const mb_row one[1] = { {1, 39, 0} };
                case_rc = mb_idx_run_case("direct-one", one, 1, slab, comp_cap,
                                          n_comp_sup, ratio, n_banks, n_head, head_dim,
                                          bank_bytes, 8);
            }
        }
        ds4_gpu_tensor_free(slab);
        free(host);
        if (case_rc != 0) goto done;
    }

    /* Dead-row guard: an out-of-pool seq_id must produce an all -INF score
     * row (fail-visible) while its batchmate row scores normally. */
    {
        const uint32_t n_head = 4;
        const uint64_t q_row = (uint64_t)n_head * head_dim;
        const uint64_t bank_bytes = (uint64_t)comp_cap * head_dim * sizeof(float);
        const uint64_t count = (uint64_t)n_banks * comp_cap * head_dim;
        float *host = (float *)malloc(count * sizeof(float));
        float *q_host = (float *)malloc(2 * q_row * sizeof(float));
        float *w_host = (float *)malloc(2 * n_head * sizeof(float));
        float *out = (float *)malloc(2 * (uint64_t)n_comp_sup * sizeof(float));
        const int32_t pos_host[2] = {100, 37};
        const int32_t sid_host[2] = {-1, 1};
        ds4_gpu_tensor *slab = ds4_gpu_tensor_alloc(count * sizeof(float));
        ds4_gpu_tensor *q = ds4_gpu_tensor_alloc(2 * q_row * sizeof(float));
        ds4_gpu_tensor *w = ds4_gpu_tensor_alloc(2 * n_head * sizeof(float));
        ds4_gpu_tensor *scores = ds4_gpu_tensor_alloc(2 * (uint64_t)n_comp_sup * sizeof(float));
        ds4_gpu_tensor *p = ds4_gpu_tensor_alloc(2 * sizeof(int32_t));
        ds4_gpu_tensor *s = ds4_gpu_tensor_alloc(2 * sizeof(int32_t));
        int dead_rc = 1;
        (void)bank_bytes;
        if (host && q_host && w_host && out && slab && q && w && scores && p && s) {
            mb_rng_state = 0xdeadf00u;
            for (uint64_t i = 0; i < count; i++) host[i] = mb_rand();
            for (uint64_t i = 0; i < 2 * q_row; i++) q_host[i] = mb_rand() * 0.5f;
            for (uint64_t i = 0; i < 2 * n_head; i++) w_host[i] = 1.0f;
            if (ds4_gpu_tensor_write(slab, 0, host, count * sizeof(float)) &&
                ds4_gpu_tensor_write(q, 0, q_host, 2 * q_row * sizeof(float)) &&
                ds4_gpu_tensor_write(w, 0, w_host, 2 * n_head * sizeof(float)) &&
                ds4_gpu_tensor_write(p, 0, pos_host, sizeof(pos_host)) &&
                ds4_gpu_tensor_write(s, 0, sid_host, sizeof(sid_host)) &&
                ds4_gpu_indexer_scores_decode_batch_tensor(scores, q, w, slab,
                                                           n_comp_sup, 2, 0,
                                                           n_head, head_dim, ratio, 0.125f,
                                                           p, s, NULL, comp_cap, n_banks) &&
                ds4_gpu_synchronize() &&
                ds4_gpu_tensor_read(scores, 0, out, 2 * (uint64_t)n_comp_sup * sizeof(float))) {
                dead_rc = 0;
                for (uint32_t c = 0; c < n_comp_sup; c++) {
                    if (!(out[c] == -INFINITY)) {
                        fprintf(stderr, "multibank indexer dead-row: finite score at %u\n", c);
                        dead_rc = 1;
                        break;
                    }
                }
                if (dead_rc == 0)
                    printf("  multibank indexer dead-row: out-of-pool seq_id -> all -INF scores\n");
            }
        }
        ds4_gpu_tensor_free(s); ds4_gpu_tensor_free(p); ds4_gpu_tensor_free(scores);
        ds4_gpu_tensor_free(w); ds4_gpu_tensor_free(q); ds4_gpu_tensor_free(slab);
        free(out); free(w_host); free(q_host); free(host);
        if (dead_rc != 0) goto done;
    }
    rc = 0;
done:
    return rc;
}

/* ---------------------------------------------------------------------------
 * Multi-bank raw-store scatter smoke: one banked launch storing rows into
 * two banks' rings (incl. a wrapped slot and a dead row that must store
 * NOTHING) must leave the pool byte-identical to per-bank classic stores.
 * Covers the f16 and f32 ring formats.
 * ------------------------------------------------------------------------- */
static int check_multibank_raw_store(void) {
    const uint32_t n_banks = 2, raw_cap = 64, head_dim = 32, n_rows = 4;
    const int32_t pos_host[4] = {100, 5, 6, 7};    /* 100 % 64 = 36 (wrap) */
    const int32_t sid_host[4] = {0, 1, 1, -1};     /* row 3 dead */
    int rc = 1;
    for (uint32_t f16 = 0; f16 <= 1; f16++) {
        const uint64_t elem = f16 ? 2 : 4;
        const uint64_t bank_bytes = (uint64_t)raw_cap * head_dim * elem;
        const uint64_t total = (uint64_t)n_banks * bank_bytes;
        const uint64_t kv_count = (uint64_t)n_rows * head_dim;
        uint8_t *fill = (uint8_t *)malloc(total);
        uint8_t *got = (uint8_t *)malloc(total);
        uint8_t *want = (uint8_t *)malloc(total);
        float *kv_host = (float *)malloc(kv_count * sizeof(float));
        ds4_gpu_tensor *slab = ds4_gpu_tensor_alloc(total);
        ds4_gpu_tensor *slab_ref = ds4_gpu_tensor_alloc(total);
        ds4_gpu_tensor *kv = ds4_gpu_tensor_alloc(kv_count * sizeof(float));
        ds4_gpu_tensor *positions = ds4_gpu_tensor_alloc(n_rows * sizeof(int32_t));
        ds4_gpu_tensor *seq_id = ds4_gpu_tensor_alloc(n_rows * sizeof(int32_t));
        int ok = fill && got && want && kv_host && slab && slab_ref && kv &&
                 positions && seq_id;
        if (ok) {
            memset(fill, 0x55, total);   /* poison: untouched slots must match */
            mb_rng_state = 0x5708eu + f16;
            for (uint64_t i = 0; i < kv_count; i++) kv_host[i] = mb_rand();
            ok = ds4_gpu_tensor_write(slab, 0, fill, total) &&
                 ds4_gpu_tensor_write(slab_ref, 0, fill, total) &&
                 ds4_gpu_tensor_write(kv, 0, kv_host, kv_count * sizeof(float)) &&
                 ds4_gpu_tensor_write(positions, 0, pos_host, sizeof(pos_host)) &&
                 ds4_gpu_tensor_write(seq_id, 0, sid_host, sizeof(sid_host));
        }
        /* Banked scatter: one launch, all rows. */
        if (ok) ok = ds4_gpu_store_raw_kv_batch_tensor(slab, kv, raw_cap, 0, n_rows,
                                                       head_dim, f16,
                                                       positions, seq_id, n_banks);
        /* Reference: classic per-row stores into the bank views (dead row
         * skipped — it must store nothing). */
        for (uint32_t r = 0; ok && r < n_rows; r++) {
            if (sid_host[r] < 0) continue;
            ds4_gpu_tensor *bank_view = ds4_gpu_tensor_view(
                    slab_ref, (uint64_t)sid_host[r] * bank_bytes, bank_bytes);
            ds4_gpu_tensor *kv_view = ds4_gpu_tensor_view(
                    kv, (uint64_t)r * head_dim * sizeof(float),
                    (uint64_t)head_dim * sizeof(float));
            ok = bank_view && kv_view &&
                 ds4_gpu_store_raw_kv_tensor(bank_view, kv_view, raw_cap,
                                             (uint32_t)pos_host[r] % raw_cap,
                                             head_dim, f16);
            ds4_gpu_tensor_free(kv_view);
            ds4_gpu_tensor_free(bank_view);
        }
        ok = ok && ds4_gpu_synchronize() &&
             ds4_gpu_tensor_read(slab, 0, got, total) &&
             ds4_gpu_tensor_read(slab_ref, 0, want, total);
        if (ok && memcmp(got, want, total) != 0) {
            fprintf(stderr, "multibank raw-store (f16=%u): banked scatter != per-bank stores\n", f16);
            ok = 0;
        }
        ds4_gpu_tensor_free(seq_id); ds4_gpu_tensor_free(positions);
        ds4_gpu_tensor_free(kv); ds4_gpu_tensor_free(slab_ref); ds4_gpu_tensor_free(slab);
        free(kv_host); free(want); free(got); free(fill);
        if (!ok) return 1;
        printf("  multibank raw-store (f16=%u): banked scatter == per-bank stores (dead row stored nothing)\n", f16);
    }
    rc = 0;
    return rc;
}

int main(void) {
    if (!ds4_gpu_init()) return 1;
    int rc = check_large_topk();
    if (check_mxkv_roundtrip() != 0) rc = 1;
    if (check_mxkv_gather() != 0) rc = 1;
    if (check_dspark_fp8_kv_pack() != 0) rc = 1;
    if (check_dspark_markov_head() != 0) rc = 1;
    if (check_dspark_confidence_head() != 0) rc = 1;
    if (check_dspark_non_causal_attention() != 0) rc = 1;
    if (check_decode_attention_overflow_path() != 0) rc = 1;
    if (check_multibank_decode_attention() != 0) rc = 1;
    if (check_multibank_indexer() != 0) rc = 1;
    if (check_multibank_raw_store() != 0) rc = 1;
    ds4_gpu_cleanup();
    if (rc == 0) puts("cuda long-context regression: OK");
    return rc;
}
