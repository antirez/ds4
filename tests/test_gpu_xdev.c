/* Cross-device GPU tensor copy test (mgpu-device-aware-cuda).
 *
 * Exercises:
 *   - ds4_gpu_init_multi single-device and multi-device init paths.
 *   - ds4_gpu_tensor_alloc_on / ds4_gpu_tensor_free_in_place.
 *   - ds4_gpu_tensor_copy_xdev same-device fast path.
 *   - ds4_gpu_tensor_copy_xdev peer-auto and DS4_FORCE_HOST_BOUNCE paths
 *     (when 2+ GPUs are visible). */

/* ds4_gpu_mgpu.h is standalone-C-compatible — it now provides the
 * complete ds4_gpu_tensor struct + typedef so callers can use the
 * bare type name. ds4_gpu.h is also included here only for the
 * legacy ds4_gpu_init / _cleanup / _tensor_read / _tensor_write
 * prototypes the test uses; the mgpu header does not duplicate them. */
#include "ds4_gpu.h"
#include "ds4_gpu_mgpu.h"

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(cond, msg)                                                    \
    do {                                                                    \
        if (!(cond)) {                                                      \
            fprintf(stderr, "FAIL: %s (line %d)\n", (msg), __LINE__);       \
            return 1;                                                       \
        }                                                                   \
    } while (0)

static int run_one(int n_gpus_wanted, int force_bounce) {
    int dev_count = 0;
    (void)cudaGetDeviceCount(&dev_count);
    if (dev_count < 1) {
        fprintf(stderr, "no CUDA devices visible\n");
        return 0;
    }
    if (n_gpus_wanted > dev_count) {
        fprintf(stderr, "skip: wanted %d GPUs, have %d\n",
                n_gpus_wanted, dev_count);
        return 0;
    }

    ds4_gpu_config cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.n_gpus = n_gpus_wanted;
    for (int i = 0; i < n_gpus_wanted; i++) cfg.device_indices[i] = i;
    if (force_bounce) setenv("DS4_FORCE_HOST_BOUNCE", "1", 1);
    else              unsetenv("DS4_FORCE_HOST_BOUNCE");
    CHECK(ds4_gpu_init_multi(&cfg), "init_multi");

    const size_t N = 256 * 1024;
    float *host_src = (float *)malloc(N * sizeof(float));
    float *host_dst = (float *)malloc(N * sizeof(float));
    CHECK(host_src && host_dst, "host alloc");
    for (size_t i = 0; i < N; i++) host_src[i] = (float)(i % 997) * 0.5f;

    ds4_gpu_tensor a; memset(&a, 0, sizeof(a));
    ds4_gpu_tensor b; memset(&b, 0, sizeof(b));
    CHECK(ds4_gpu_tensor_alloc_on(&a, 0, N * sizeof(float)) == 0,
          "alloc_on dev 0");
    int dst_dev = (n_gpus_wanted == 2) ? 1 : 0;
    CHECK(ds4_gpu_tensor_alloc_on(&b, dst_dev, N * sizeof(float)) == 0,
          "alloc_on dst dev");

    CHECK(ds4_gpu_tensor_write(&a, 0, host_src, N * sizeof(float)),
          "tensor_write");
    CHECK(ds4_gpu_tensor_copy_xdev(&b, &a, N * sizeof(float)),
          "tensor_copy_xdev");
    CHECK(ds4_gpu_tensor_read(&b, 0, host_dst, N * sizeof(float)),
          "tensor_read");
    CHECK(memcmp(host_src, host_dst, N * sizeof(float)) == 0,
          "data mismatch");

    /* device_id round-trip. */
    CHECK(ds4_gpu_tensor_device(&a) == 0, "device_id a");
    CHECK(ds4_gpu_tensor_device(&b) == dst_dev, "device_id b");

    ds4_gpu_tensor_free_in_place(&a);
    ds4_gpu_tensor_free_in_place(&b);
    free(host_src);
    free(host_dst);
    ds4_gpu_cleanup();
    fprintf(stderr, "  test_gpu_xdev OK (n_gpus=%d, force_bounce=%d)\n",
            n_gpus_wanted, force_bounce);
    return 0;
}

static int run_copy3(int n_gpus_wanted, int force_bounce) {
    int dev_count = 0;
    (void)cudaGetDeviceCount(&dev_count);
    if (dev_count < 1) {
        fprintf(stderr, "no CUDA devices visible\n");
        return 0;
    }
    if (n_gpus_wanted > dev_count) {
        fprintf(stderr, "skip copy3: wanted %d GPUs, have %d\n",
                n_gpus_wanted, dev_count);
        return 0;
    }

    ds4_gpu_config cfg; memset(&cfg, 0, sizeof(cfg));
    cfg.n_gpus = n_gpus_wanted;
    for (int i = 0; i < n_gpus_wanted; i++) cfg.device_indices[i] = i;
    if (force_bounce) setenv("DS4_FORCE_HOST_BOUNCE", "1", 1);
    else              unsetenv("DS4_FORCE_HOST_BOUNCE");
    CHECK(ds4_gpu_init_multi(&cfg), "copy3 init_multi");

    const uint32_t n_embd = 4096u;
    const uint32_t n_expert = 3u;
    const size_t norm_bytes = (size_t)n_embd * sizeof(float);
    const size_t selected_bytes = (size_t)n_expert * sizeof(int32_t);
    const size_t weights_bytes = (size_t)n_expert * sizeof(float);
    int32_t host_selected[3] = {17, 99, 203};
    float host_weights[3] = {0.25f, 0.5f, 0.75f};
    int32_t got_selected[3] = {0, 0, 0};
    float got_weights[3] = {0.0f, 0.0f, 0.0f};
    float *host_norm = (float *)malloc(norm_bytes);
    float *got_norm = (float *)malloc(norm_bytes);
    CHECK(host_norm && got_norm, "copy3 host alloc");
    for (uint32_t i = 0; i < n_embd; i++) host_norm[i] = (float)((int)i - 1024) * 0.015625f;

    const int dst_dev = (n_gpus_wanted == 2) ? 1 : 0;
    ds4_gpu_tensor norm_src; memset(&norm_src, 0, sizeof(norm_src));
    ds4_gpu_tensor sel_src; memset(&sel_src, 0, sizeof(sel_src));
    ds4_gpu_tensor w_src; memset(&w_src, 0, sizeof(w_src));
    ds4_gpu_tensor norm_dst; memset(&norm_dst, 0, sizeof(norm_dst));
    ds4_gpu_tensor sel_dst; memset(&sel_dst, 0, sizeof(sel_dst));
    ds4_gpu_tensor w_dst; memset(&w_dst, 0, sizeof(w_dst));
    CHECK(ds4_gpu_tensor_alloc_on(&norm_src, 0, norm_bytes) == 0, "copy3 alloc norm_src");
    CHECK(ds4_gpu_tensor_alloc_on(&sel_src, 0, selected_bytes) == 0, "copy3 alloc sel_src");
    CHECK(ds4_gpu_tensor_alloc_on(&w_src, 0, weights_bytes) == 0, "copy3 alloc w_src");
    CHECK(ds4_gpu_tensor_alloc_on(&norm_dst, dst_dev, norm_bytes) == 0, "copy3 alloc norm_dst");
    CHECK(ds4_gpu_tensor_alloc_on(&sel_dst, dst_dev, selected_bytes) == 0, "copy3 alloc sel_dst");
    CHECK(ds4_gpu_tensor_alloc_on(&w_dst, dst_dev, weights_bytes) == 0, "copy3 alloc w_dst");

    int ok = ds4_gpu_tensor_write(&norm_src, 0, host_norm, norm_bytes) &&
             ds4_gpu_tensor_write(&sel_src, 0, host_selected, selected_bytes) &&
             ds4_gpu_tensor_write(&w_src, 0, host_weights, weights_bytes) &&
             ds4_gpu_tensor_copy_xdev3(&norm_dst, &norm_src, norm_bytes,
                                       &sel_dst, &sel_src, selected_bytes,
                                       &w_dst, &w_src, weights_bytes) &&
             ds4_gpu_tensor_read(&norm_dst, 0, got_norm, norm_bytes) &&
             ds4_gpu_tensor_read(&sel_dst, 0, got_selected, selected_bytes) &&
             ds4_gpu_tensor_read(&w_dst, 0, got_weights, weights_bytes);
    CHECK(ok, "copy3 IO");
    CHECK(memcmp(host_norm, got_norm, norm_bytes) == 0, "copy3 norm");
    CHECK(memcmp(host_selected, got_selected, selected_bytes) == 0, "copy3 selected");
    CHECK(memcmp(host_weights, got_weights, weights_bytes) == 0, "copy3 weights");

    ds4_gpu_tensor_free_in_place(&norm_src);
    ds4_gpu_tensor_free_in_place(&sel_src);
    ds4_gpu_tensor_free_in_place(&w_src);
    ds4_gpu_tensor_free_in_place(&norm_dst);
    ds4_gpu_tensor_free_in_place(&sel_dst);
    ds4_gpu_tensor_free_in_place(&w_dst);
    free(host_norm);
    free(got_norm);
    ds4_gpu_cleanup();
    fprintf(stderr, "  copy_xdev3 OK (n_gpus=%d, force_bounce=%d)\n",
            n_gpus_wanted, force_bounce);
    return 0;
}

static int run_top1(void) {
    int dev_count = 0;
    (void)cudaGetDeviceCount(&dev_count);
    if (dev_count < 1) {
        fprintf(stderr, "no CUDA devices visible\n");
        return 0;
    }

    ds4_gpu_config cfg; memset(&cfg, 0, sizeof(cfg));
    cfg.n_gpus = 1;
    cfg.device_indices[0] = 0;
    CHECK(ds4_gpu_init_multi(&cfg), "top1 init_multi");

    const uint32_t n_comp = 131073u;
    const uint32_t n_tokens = 3u;
    const size_t n_score = (size_t)n_comp * n_tokens;
    float *host_scores = (float *)malloc(n_score * sizeof(float));
    uint32_t *host_sel = (uint32_t *)calloc(n_tokens, sizeof(uint32_t));
    CHECK(host_scores && host_sel, "top1 host alloc");
    for (size_t i = 0; i < n_score; i++) host_scores[i] = -1000.0f;

    host_scores[17] = 3.5f;
    host_scores[120000] = 3.5f;             /* tie: lower id wins */
    host_scores[1000] = NAN;                /* NaN is ignored like CPU argmax */
    host_scores[(size_t)n_comp + 5] = 8.0f;
    host_scores[(size_t)n_comp + n_comp - 1u] = 8.0f;
    for (uint32_t i = 0; i < n_comp; i++) {
        host_scores[(size_t)2u * n_comp + i] = -INFINITY;
    }
    host_scores[(size_t)2u * n_comp + 1u] = NAN;

    ds4_gpu_tensor scores; memset(&scores, 0, sizeof(scores));
    ds4_gpu_tensor selected; memset(&selected, 0, sizeof(selected));
    CHECK(ds4_gpu_tensor_alloc_on(&scores, 0, n_score * sizeof(float)) == 0,
          "top1 alloc scores");
    CHECK(ds4_gpu_tensor_alloc_on(&selected, 0, n_tokens * sizeof(uint32_t)) == 0,
          "top1 alloc selected");
    int ok = ds4_gpu_tensor_write(&scores, 0, host_scores, n_score * sizeof(float)) &&
             ds4_gpu_indexer_topk_tensor(&selected, &scores, n_comp, n_tokens, 1) &&
             ds4_gpu_tensor_read(&selected, 0, host_sel, n_tokens * sizeof(uint32_t));
    CHECK(ok, "top1 compute");
    CHECK(host_sel[0] == 17u, "top1 row0");
    CHECK(host_sel[1] == 5u, "top1 row1");
    CHECK(host_sel[2] == 0u, "top1 row2");

    ds4_gpu_tensor_free_in_place(&scores);
    ds4_gpu_tensor_free_in_place(&selected);
    free(host_scores);
    free(host_sel);
    ds4_gpu_cleanup();
    fprintf(stderr, "  top1 OK\n");
    return 0;
}

typedef struct topk_ref_entry {
    float score;
    uint32_t index;
} topk_ref_entry;

static int topk_ref_cmp(const void *ap, const void *bp) {
    const topk_ref_entry *a = (const topk_ref_entry *)ap;
    const topk_ref_entry *b = (const topk_ref_entry *)bp;
    if (a->score > b->score) return -1;
    if (a->score < b->score) return 1;
    return a->index < b->index ? -1 : (a->index > b->index ? 1 : 0);
}

static int run_topk2048(void) {
    int dev_count = 0;
    (void)cudaGetDeviceCount(&dev_count);
    if (dev_count < 1) return 0;

    ds4_gpu_config cfg; memset(&cfg, 0, sizeof(cfg));
    cfg.n_gpus = 1;
    cfg.device_indices[0] = 0;
    CHECK(ds4_gpu_init_multi(&cfg), "topk2048 init_multi");

    const uint32_t cases[] = {3355u, 5003u};
    const uint32_t n_tokens = 2u;
    const uint32_t top_k = 2048u;
    for (uint32_t ci = 0; ci < 2u; ci++) {
        const uint32_t n_comp = cases[ci];
        const uint64_t n_scores = (uint64_t)n_tokens * n_comp;
        float *host_scores = (float *)malloc((size_t)n_scores * sizeof(float));
        uint32_t *host_selected =
            (uint32_t *)malloc((size_t)n_tokens * top_k * sizeof(uint32_t));
        topk_ref_entry *ref =
            (topk_ref_entry *)malloc((size_t)n_comp * sizeof(topk_ref_entry));
        CHECK(host_scores && host_selected && ref, "topk2048 host alloc");
        for (uint32_t t = 0; t < n_tokens; t++) {
            for (uint32_t i = 0; i < n_comp; i++) {
                host_scores[(uint64_t)t * n_comp + i] =
                    (float)(((uint64_t)i * 37u + t * 101u) % 1009u);
            }
        }

        ds4_gpu_tensor scores; memset(&scores, 0, sizeof(scores));
        ds4_gpu_tensor selected; memset(&selected, 0, sizeof(selected));
        CHECK(ds4_gpu_tensor_alloc_on(
                  &scores, 0, n_scores * sizeof(float)) == 0,
              "topk2048 alloc scores");
        CHECK(ds4_gpu_tensor_alloc_on(
                  &selected, 0,
                  (uint64_t)n_tokens * top_k * sizeof(uint32_t)) == 0,
              "topk2048 alloc selected");
        int ok = ds4_gpu_tensor_write(
                     &scores, 0, host_scores, n_scores * sizeof(float)) &&
                 ds4_gpu_indexer_topk_tensor(
                     &selected, &scores, n_comp, n_tokens, top_k) &&
                 ds4_gpu_tensor_read(
                     &selected, 0, host_selected,
                     (uint64_t)n_tokens * top_k * sizeof(uint32_t));
        CHECK(ok, "topk2048 compute");

        for (uint32_t t = 0; t < n_tokens; t++) {
            for (uint32_t i = 0; i < n_comp; i++) {
                ref[i].score = host_scores[(uint64_t)t * n_comp + i];
                ref[i].index = i;
            }
            qsort(ref, n_comp, sizeof(ref[0]), topk_ref_cmp);
            for (uint32_t i = 0; i < top_k; i++) {
                if (host_selected[(uint64_t)t * top_k + i] != ref[i].index) {
                    fprintf(stderr,
                            "FAIL: topk2048 n=%u token=%u rank=%u got=%u want=%u\n",
                            n_comp, t, i,
                            host_selected[(uint64_t)t * top_k + i],
                            ref[i].index);
                    return 1;
                }
            }
        }
        ds4_gpu_tensor_free_in_place(&scores);
        ds4_gpu_tensor_free_in_place(&selected);
        free(host_scores);
        free(host_selected);
        free(ref);
    }
    ds4_gpu_cleanup();
    fprintf(stderr, "  topk2048 OK\n");
    return 0;
}

static int run_glm_selected_attention(void) {
    int dev_count = 0;
    (void)cudaGetDeviceCount(&dev_count);
    if (dev_count < 1) return 0;

    ds4_gpu_config cfg; memset(&cfg, 0, sizeof(cfg));
    cfg.n_gpus = 1;
    cfg.device_indices[0] = 0;
    CHECK(ds4_gpu_init_multi(&cfg), "glm selected attention init_multi");

    const uint32_t n_head = 1u;
    const uint32_t kv_dim = 512u;
    const uint32_t qk_nope = 192u;
    const uint32_t qk_rope = 64u;
    const uint32_t q_dim = qk_nope + qk_rope;
    const uint32_t cache_cap = 5u;
    const uint32_t n_selected = 3u;
    float host_q[q_dim];
    float host_low[kv_dim];
    float host_kv[cache_cap * kv_dim];
    float host_rope[cache_cap * qk_rope];
    float host_kv_compact[n_selected * kv_dim];
    float host_rope_compact[n_selected * qk_rope];
    uint32_t host_selected[3] = {4u, 1u, 3u};
    float causal[kv_dim];
    float indexed[kv_dim];
    for (uint32_t i = 0; i < q_dim; i++)
        host_q[i] = (float)((int)(i % 17u) - 8) * 0.01f;
    /* Isolate selected-row KV gathering. RoPE depends on the original cache
     * row, so a physically compacted reference is equivalent only when the
     * query's RoPE tail contributes zero to the score. */
    for (uint32_t i = qk_nope; i < q_dim; i++) host_q[i] = 0.0f;
    for (uint32_t i = 0; i < kv_dim; i++)
        host_low[i] = (float)((int)(i % 13u) - 6) * 0.02f;
    for (uint32_t i = 0; i < cache_cap * kv_dim; i++)
        host_kv[i] = (float)((int)((i * 7u) % 19u) - 9) * 0.03f;
    for (uint32_t i = 0; i < cache_cap * qk_rope; i++)
        host_rope[i] = (float)((int)((i * 5u) % 11u) - 5) * 0.015f;
    for (uint32_t i = 0; i < n_selected; i++) {
        memcpy(host_kv_compact + (size_t)i * kv_dim,
               host_kv + (size_t)host_selected[i] * kv_dim,
               (size_t)kv_dim * sizeof(float));
        memcpy(host_rope_compact + (size_t)i * qk_rope,
               host_rope + (size_t)host_selected[i] * qk_rope,
               (size_t)qk_rope * sizeof(float));
    }

    ds4_gpu_tensor q; memset(&q, 0, sizeof(q));
    ds4_gpu_tensor low; memset(&low, 0, sizeof(low));
    ds4_gpu_tensor kv; memset(&kv, 0, sizeof(kv));
    ds4_gpu_tensor rope; memset(&rope, 0, sizeof(rope));
    ds4_gpu_tensor kv_compact; memset(&kv_compact, 0, sizeof(kv_compact));
    ds4_gpu_tensor rope_compact; memset(&rope_compact, 0, sizeof(rope_compact));
    ds4_gpu_tensor selected; memset(&selected, 0, sizeof(selected));
    ds4_gpu_tensor out_causal; memset(&out_causal, 0, sizeof(out_causal));
    ds4_gpu_tensor out_indexed; memset(&out_indexed, 0, sizeof(out_indexed));
    CHECK(ds4_gpu_tensor_alloc_on(&q, 0, sizeof(host_q)) == 0,
          "glm selected attention alloc q");
    CHECK(ds4_gpu_tensor_alloc_on(&low, 0, sizeof(host_low)) == 0,
          "glm selected attention alloc low");
    CHECK(ds4_gpu_tensor_alloc_on(&kv, 0, sizeof(host_kv)) == 0,
          "glm selected attention alloc kv");
    CHECK(ds4_gpu_tensor_alloc_on(&rope, 0, sizeof(host_rope)) == 0,
          "glm selected attention alloc rope");
    CHECK(ds4_gpu_tensor_alloc_on(&kv_compact, 0, sizeof(host_kv_compact)) == 0,
          "glm selected attention alloc compact kv");
    CHECK(ds4_gpu_tensor_alloc_on(&rope_compact, 0, sizeof(host_rope_compact)) == 0,
          "glm selected attention alloc compact rope");
    CHECK(ds4_gpu_tensor_alloc_on(&selected, 0, sizeof(host_selected)) == 0,
          "glm selected attention alloc selected");
    CHECK(ds4_gpu_tensor_alloc_on(&out_causal, 0, sizeof(causal)) == 0,
          "glm selected attention alloc causal");
    CHECK(ds4_gpu_tensor_alloc_on(&out_indexed, 0, sizeof(indexed)) == 0,
          "glm selected attention alloc indexed");

    int ok = ds4_gpu_tensor_write(&q, 0, host_q, sizeof(host_q)) &&
             ds4_gpu_tensor_write(&low, 0, host_low, sizeof(host_low)) &&
             ds4_gpu_tensor_write(&kv, 0, host_kv, sizeof(host_kv)) &&
             ds4_gpu_tensor_write(&rope, 0, host_rope, sizeof(host_rope)) &&
             ds4_gpu_tensor_write(&kv_compact, 0, host_kv_compact,
                                  sizeof(host_kv_compact)) &&
             ds4_gpu_tensor_write(&rope_compact, 0, host_rope_compact,
                                  sizeof(host_rope_compact)) &&
             ds4_gpu_tensor_write(&selected, 0, host_selected,
                                  sizeof(host_selected)) &&
             ds4_gpu_glm_attention_indexed_batch_lora_causal_tensor(
                 &out_causal, &q, &low, &kv_compact, &rope_compact,
                 1u, 2u, n_selected, n_selected, false,
                 n_head, kv_dim, qk_nope, qk_rope,
                 0u, 8000000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f) &&
             ds4_gpu_glm_attention_indexed_batch_lora_valid_tensor(
                 &out_indexed, &q, &low, &kv, &rope, &selected,
                 1u, n_selected, cache_cap, false,
                 n_head, kv_dim, qk_nope, qk_rope,
                 0u, 8000000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f) &&
             ds4_gpu_tensor_read(&out_causal, 0, causal, sizeof(causal)) &&
             ds4_gpu_tensor_read(&out_indexed, 0, indexed, sizeof(indexed));
    CHECK(ok, "glm selected attention compute");
    CHECK(memcmp(causal, indexed, sizeof(causal)) == 0,
          "glm selected attention mismatch");

    ds4_gpu_tensor_free_in_place(&q);
    ds4_gpu_tensor_free_in_place(&low);
    ds4_gpu_tensor_free_in_place(&kv);
    ds4_gpu_tensor_free_in_place(&rope);
    ds4_gpu_tensor_free_in_place(&kv_compact);
    ds4_gpu_tensor_free_in_place(&rope_compact);
    ds4_gpu_tensor_free_in_place(&selected);
    ds4_gpu_tensor_free_in_place(&out_causal);
    ds4_gpu_tensor_free_in_place(&out_indexed);
    ds4_gpu_cleanup();
    fprintf(stderr, "  glm_selected_attention OK\n");
    return 0;
}

static int run_glm_indexer_scores(void) {
    int dev_count = 0;
    (void)cudaGetDeviceCount(&dev_count);
    if (dev_count < 1) return 0;

    ds4_gpu_config cfg; memset(&cfg, 0, sizeof(cfg));
    cfg.n_gpus = 1;
    cfg.device_indices[0] = 0;
    CHECK(ds4_gpu_init_multi(&cfg), "glm indexer scores init_multi");

    const uint32_t n_rows = 137u;
    const uint32_t n_tokens = 17u;
    const uint32_t pos0 = 5u;
    const uint32_t n_head = 32u;
    const uint32_t head_dim = 128u;
    const float scale = 1.0f / 64.0f;
    const size_t q_count = (size_t)n_tokens * n_head * head_dim;
    const size_t weight_count = (size_t)n_tokens * n_head;
    const size_t cache_count = (size_t)n_rows * head_dim;
    const size_t score_count = (size_t)n_tokens * n_rows;
    float *host_q = (float *)malloc(q_count * sizeof(float));
    float *host_weights = (float *)malloc(weight_count * sizeof(float));
    float *host_cache = (float *)malloc(cache_count * sizeof(float));
    float *host_fast = (float *)malloc(score_count * sizeof(float));
    float *host_quality = (float *)malloc(score_count * sizeof(float));
    float *host_ref = (float *)malloc(score_count * sizeof(float));
    CHECK(host_q && host_weights && host_cache && host_fast && host_quality &&
          host_ref, "glm indexer scores host alloc");

    for (size_t i = 0; i < q_count; i++)
        host_q[i] = (float)((int)(i % 15u) - 7) / 32.0f;
    for (size_t i = 0; i < weight_count; i++)
        host_weights[i] = (float)((int)(i % 7u) - 3) / 8.0f;
    for (size_t i = 0; i < cache_count; i++)
        host_cache[i] = (float)((int)((i * 5u) % 13u) - 6) / 32.0f;
    for (uint32_t t = 0; t < n_tokens; t++) {
        for (uint32_t row = 0; row < n_rows; row++) {
            float total = 0.0f;
            for (uint32_t h = 0; h < n_head; h++) {
                float dot = 0.0f;
                const float *qh = host_q +
                    ((size_t)t * n_head + h) * head_dim;
                const float *kr = host_cache + (size_t)row * head_dim;
                for (uint32_t d = 0; d < head_dim; d++) dot += qh[d] * kr[d];
                if (dot > 0.0f)
                    total += dot * host_weights[(size_t)t * n_head + h];
            }
            host_ref[(size_t)t * n_rows + row] =
                row < pos0 + t + 1u ? total * scale : -INFINITY;
        }
    }

    ds4_gpu_tensor q; memset(&q, 0, sizeof(q));
    ds4_gpu_tensor weights; memset(&weights, 0, sizeof(weights));
    ds4_gpu_tensor cache; memset(&cache, 0, sizeof(cache));
    ds4_gpu_tensor scores; memset(&scores, 0, sizeof(scores));
    CHECK(ds4_gpu_tensor_alloc_on(&q, 0, q_count * sizeof(float)) == 0,
          "glm indexer scores alloc q");
    CHECK(ds4_gpu_tensor_alloc_on(&weights, 0,
                                  weight_count * sizeof(float)) == 0,
          "glm indexer scores alloc weights");
    CHECK(ds4_gpu_tensor_alloc_on(&cache, 0,
                                  cache_count * sizeof(float)) == 0,
          "glm indexer scores alloc cache");
    CHECK(ds4_gpu_tensor_alloc_on(&scores, 0,
                                  score_count * sizeof(float)) == 0,
          "glm indexer scores alloc output");
    int ok = ds4_gpu_tensor_write(&q, 0, host_q, q_count * sizeof(float)) &&
             ds4_gpu_tensor_write(&weights, 0, host_weights,
                                  weight_count * sizeof(float)) &&
             ds4_gpu_tensor_write(&cache, 0, host_cache,
                                  cache_count * sizeof(float));
    CHECK(ok, "glm indexer scores upload");

    ds4_gpu_set_quality(false);
    ok = ds4_gpu_glm_indexer_scores_batch_tensor(
             &scores, &q, &weights, &cache, n_rows, n_tokens, pos0,
             n_head, head_dim, scale, false) &&
         ds4_gpu_tensor_read(&scores, 0, host_fast,
                             score_count * sizeof(float));
    CHECK(ok, "glm indexer scores fast compute");
    ds4_gpu_set_quality(true);
    ok = ds4_gpu_glm_indexer_scores_batch_tensor(
             &scores, &q, &weights, &cache, n_rows, n_tokens, pos0,
             n_head, head_dim, scale, false) &&
         ds4_gpu_tensor_read(&scores, 0, host_quality,
                             score_count * sizeof(float));
    CHECK(ok, "glm indexer scores quality compute");
    ds4_gpu_set_quality(false);
    CHECK(memcmp(host_fast, host_ref, score_count * sizeof(float)) == 0,
          "glm indexer fast scores mismatch");
    CHECK(memcmp(host_quality, host_ref, score_count * sizeof(float)) == 0,
          "glm indexer quality scores mismatch");

    ds4_gpu_tensor_free_in_place(&q);
    ds4_gpu_tensor_free_in_place(&weights);
    ds4_gpu_tensor_free_in_place(&cache);
    ds4_gpu_tensor_free_in_place(&scores);
    free(host_q);
    free(host_weights);
    free(host_cache);
    free(host_fast);
    free(host_quality);
    free(host_ref);
    ds4_gpu_cleanup();
    fprintf(stderr, "  glm_indexer_scores OK\n");
    return 0;
}

static int run_moe_handoff_pack(void) {
    int dev_count = 0;
    (void)cudaGetDeviceCount(&dev_count);
    if (dev_count < 1) {
        fprintf(stderr, "no CUDA devices visible\n");
        return 0;
    }

    ds4_gpu_config cfg; memset(&cfg, 0, sizeof(cfg));
    cfg.n_gpus = 1;
    cfg.device_indices[0] = 0;
    CHECK(ds4_gpu_init_multi(&cfg), "moe_pack init_multi");

    const uint32_t n_embd = 4096u;
    const uint32_t n_expert = 3u;
    const size_t norm_bytes = (size_t)n_embd * sizeof(float);
    const size_t selected_bytes = (size_t)n_expert * sizeof(int32_t);
    const size_t weights_bytes = (size_t)n_expert * sizeof(float);
    const size_t packed_bytes = norm_bytes + selected_bytes + weights_bytes;

    float *host_norm = (float *)malloc(norm_bytes);
    int32_t *host_selected = (int32_t *)malloc(selected_bytes);
    float *host_weights = (float *)malloc(weights_bytes);
    unsigned char *host_packed = (unsigned char *)calloc(1, packed_bytes);
    CHECK(host_norm && host_selected && host_weights && host_packed, "moe_pack host alloc");
    for (uint32_t i = 0; i < n_embd; i++) host_norm[i] = (float)((int)i - 2000) * 0.03125f;
    for (uint32_t i = 0; i < n_expert; i++) {
        host_selected[i] = (int32_t)(200 + i * 7);
        host_weights[i] = 0.125f * (float)(i + 1);
    }

    ds4_gpu_tensor norm; memset(&norm, 0, sizeof(norm));
    ds4_gpu_tensor selected; memset(&selected, 0, sizeof(selected));
    ds4_gpu_tensor weights; memset(&weights, 0, sizeof(weights));
    ds4_gpu_tensor packed; memset(&packed, 0, sizeof(packed));
    CHECK(ds4_gpu_tensor_alloc_on(&norm, 0, norm_bytes) == 0, "moe_pack alloc norm");
    CHECK(ds4_gpu_tensor_alloc_on(&selected, 0, selected_bytes) == 0, "moe_pack alloc selected");
    CHECK(ds4_gpu_tensor_alloc_on(&weights, 0, weights_bytes) == 0, "moe_pack alloc weights");
    CHECK(ds4_gpu_tensor_alloc_on(&packed, 0, packed_bytes) == 0, "moe_pack alloc packed");

    int ok = ds4_gpu_tensor_write(&norm, 0, host_norm, norm_bytes) &&
             ds4_gpu_tensor_write(&selected, 0, host_selected, selected_bytes) &&
             ds4_gpu_tensor_write(&weights, 0, host_weights, weights_bytes) &&
             ds4_gpu_moe_handoff_pack_tensor(&packed, &norm, &selected, &weights,
                                             n_embd, n_expert) &&
             ds4_gpu_tensor_read(&packed, 0, host_packed, packed_bytes);
    CHECK(ok, "moe_pack compute");
    CHECK(memcmp(host_packed, host_norm, norm_bytes) == 0, "moe_pack norm");
    CHECK(memcmp(host_packed + norm_bytes, host_selected, selected_bytes) == 0,
          "moe_pack selected");
    CHECK(memcmp(host_packed + norm_bytes + selected_bytes, host_weights, weights_bytes) == 0,
          "moe_pack weights");

    ds4_gpu_tensor_free_in_place(&norm);
    ds4_gpu_tensor_free_in_place(&selected);
    ds4_gpu_tensor_free_in_place(&weights);
    ds4_gpu_tensor_free_in_place(&packed);
    free(host_norm);
    free(host_selected);
    free(host_weights);
    free(host_packed);
    ds4_gpu_cleanup();
    fprintf(stderr, "  moe_handoff_pack OK\n");
    return 0;
}

/* Stress sub-test: repeated cross-device round-trips at realistic sizes.
 * Designed to catch the RTX 6000 Ada / driver bug where peer copies pass
 * a single small probe but corrupt data at larger sizes or on repeat. */
static int run_stress(int force_bounce) {
    int dev_count = 0;
    (void)cudaGetDeviceCount(&dev_count);
    if (dev_count < 2) {
        fprintf(stderr, "  skipping stress (need >= 2 devices)\n");
        return 0;
    }

    ds4_gpu_config cfg; memset(&cfg, 0, sizeof(cfg));
    cfg.n_gpus = 2;
    cfg.device_indices[0] = 0; cfg.device_indices[1] = 1;
    if (force_bounce) setenv("DS4_FORCE_HOST_BOUNCE", "1", 1);
    else              unsetenv("DS4_FORCE_HOST_BOUNCE");
    CHECK(ds4_gpu_init_multi(&cfg), "stress init_multi");

    const size_t sizes[] = {
        256u * 1024u,
        1u * 1024u * 1024u,
        16u * 1024u * 1024u,
    };
    const int    n_sizes = (int)(sizeof(sizes) / sizeof(sizes[0]));
    const int    iters   = 32;
    const size_t max_n   = sizes[n_sizes - 1];

    unsigned char *host_src = (unsigned char *)malloc(max_n);
    unsigned char *host_dst = (unsigned char *)malloc(max_n);
    CHECK(host_src && host_dst, "stress host alloc");

    ds4_gpu_tensor a; memset(&a, 0, sizeof(a));
    ds4_gpu_tensor b; memset(&b, 0, sizeof(b));
    CHECK(ds4_gpu_tensor_alloc_on(&a, 0, max_n) == 0, "stress alloc_on 0");
    CHECK(ds4_gpu_tensor_alloc_on(&b, 1, max_n) == 0, "stress alloc_on 1");

    int total_ok = 1;
    for (int s = 0; s < n_sizes && total_ok; s++) {
        size_t n = sizes[s];
        for (int it = 0; it < iters && total_ok; it++) {
            for (size_t k = 0; k < n; k++) {
                host_src[k] = (unsigned char)
                    ((k * 31u + (size_t)it * 17u +
                      (size_t)s * 53u + 11u) & 0xffu);
            }
            memset(host_dst, 0, n);
            int io_ok = ds4_gpu_tensor_write(&a, 0, host_src, n) &&
                        ds4_gpu_tensor_copy_xdev(&b, &a, n) &&
                        ds4_gpu_tensor_read(&b, 0, host_dst, n);
            if (!io_ok) {
                fprintf(stderr,
                    "FAIL: stress IO error size=%zu iter=%d force_bounce=%d\n",
                    n, it, force_bounce);
                total_ok = 0;
                break;
            }
            for (size_t k = 0; k < n; k++) {
                if (host_src[k] != host_dst[k]) {
                    fprintf(stderr,
                        "FAIL: stress data mismatch size=%zu iter=%d offset=%zu"
                        " src=0x%02x dst=0x%02x force_bounce=%d\n",
                        n, it, k,
                        host_src[k], host_dst[k], force_bounce);
                    total_ok = 0;
                    break;
                }
            }
        }
    }

    ds4_gpu_tensor_free_in_place(&a);
    ds4_gpu_tensor_free_in_place(&b);
    free(host_src);
    free(host_dst);
    ds4_gpu_cleanup();
    if (total_ok) {
        fprintf(stderr,
            "  stress OK (force_bounce=%d, %d iters x %d sizes, max %zu MiB)\n",
            force_bounce, iters, n_sizes, max_n / (1024u * 1024u));
    }
    return total_ok ? 0 : 1;
}

static int run_add_xdev(int force_bounce) {
    int dev_count = 0;
    (void)cudaGetDeviceCount(&dev_count);
    if (dev_count < 2) {
        fprintf(stderr, "  skipping add_xdev (need >= 2 devices)\n");
        return 0;
    }

    ds4_gpu_config cfg; memset(&cfg, 0, sizeof(cfg));
    cfg.n_gpus = 2;
    cfg.device_indices[0] = 0; cfg.device_indices[1] = 1;
    if (force_bounce) setenv("DS4_FORCE_HOST_BOUNCE", "1", 1);
    else              unsetenv("DS4_FORCE_HOST_BOUNCE");
    CHECK(ds4_gpu_init_multi(&cfg), "add_xdev init_multi");

    const size_t N = 512 * 1024;
    float *host_local = (float *)malloc(N * sizeof(float));
    float *host_remote = (float *)malloc(N * sizeof(float));
    float *host_out = (float *)malloc(N * sizeof(float));
    CHECK(host_local && host_remote && host_out, "add_xdev host alloc");
    for (size_t i = 0; i < N; i++) {
        host_local[i] = (float)((int)(i % 1009) - 500) * 0.125f;
        host_remote[i] = (float)((int)(i % 997) - 400) * 0.25f;
    }

    ds4_gpu_tensor local; memset(&local, 0, sizeof(local));
    ds4_gpu_tensor remote; memset(&remote, 0, sizeof(remote));
    ds4_gpu_tensor staging; memset(&staging, 0, sizeof(staging));
    ds4_gpu_tensor out; memset(&out, 0, sizeof(out));
    CHECK(ds4_gpu_tensor_alloc_on(&local, 0, N * sizeof(float)) == 0,
          "add_xdev alloc local");
    CHECK(ds4_gpu_tensor_alloc_on(&remote, 1, N * sizeof(float)) == 0,
          "add_xdev alloc remote");
    CHECK(ds4_gpu_tensor_alloc_on(&staging, 0, N * sizeof(float)) == 0,
          "add_xdev alloc staging");
    CHECK(ds4_gpu_tensor_alloc_on(&out, 0, N * sizeof(float)) == 0,
          "add_xdev alloc out");

    int ok = ds4_gpu_tensor_write(&local, 0, host_local, N * sizeof(float)) &&
             ds4_gpu_tensor_write(&remote, 0, host_remote, N * sizeof(float)) &&
             ds4_gpu_add_xdev_tensor(&out, &local, &remote, &staging, (uint32_t)N) &&
             ds4_gpu_tensor_read(&out, 0, host_out, N * sizeof(float));
    CHECK(ok, "add_xdev IO");
    for (size_t i = 0; i < N; i++) {
        float want = host_local[i] + host_remote[i];
        if (host_out[i] != want) {
            fprintf(stderr,
                    "FAIL: add_xdev mismatch i=%zu got=%f want=%f force_bounce=%d\n",
                    i, host_out[i], want, force_bounce);
            return 1;
        }
    }

    ds4_gpu_tensor_free_in_place(&local);
    ds4_gpu_tensor_free_in_place(&remote);
    ds4_gpu_tensor_free_in_place(&staging);
    ds4_gpu_tensor_free_in_place(&out);
    free(host_local);
    free(host_remote);
    free(host_out);
    ds4_gpu_cleanup();
    fprintf(stderr, "  add_xdev OK (force_bounce=%d)\n", force_bounce);
    return 0;
}

static void pack_q8_identity_scale(unsigned char *w,
                                   uint64_t in_dim,
                                   uint64_t out_dim) {
    const uint64_t blocks = (in_dim + 31u) / 32u;
    for (uint64_t r = 0; r < out_dim; r++) {
        for (uint64_t b = 0; b < blocks; b++) {
            unsigned char *blk = w + (r * blocks + b) * 34u;
            /* IEEE-754 half representation of 1.0 on little-endian hosts. */
            blk[0] = 0x00u;
            blk[1] = 0x3cu;
            for (uint64_t j = 0; j < 32u; j++) {
                uint64_t i = b * 32u + j;
                int v = i < in_dim ? (int)((r * 17u + i * 5u + 11u) % 23u) - 11 : 0;
                blk[2u + j] = (unsigned char)(int8_t)v;
            }
        }
    }
}

static int run_glm_decode_attention_staged(void) {
    int dev_count = 0;
    (void)cudaGetDeviceCount(&dev_count);
    if (dev_count < 1) return 0;

    ds4_gpu_config cfg; memset(&cfg, 0, sizeof(cfg));
    cfg.n_gpus = 1;
    cfg.device_indices[0] = 0;
    CHECK(ds4_gpu_init_multi(&cfg), "glm staged attention init_multi");

    const uint32_t n_head = 4u;
    const uint32_t kv_dim = 512u;
    const uint32_t qk_nope = 192u;
    const uint32_t qk_rope = 64u;
    const uint32_t q_dim = qk_nope + qk_rope;
    const uint32_t value_dim = 128u;
    const uint32_t cache_cap = 2053u;
    const uint32_t cases[] = {513u, 2048u};
    const uint64_t value_rows = (uint64_t)n_head * value_dim;
    const uint64_t model_size =
        value_rows * (kv_dim / 32u) * 34u;
    const size_t q_count = (size_t)n_head * q_dim;
    const size_t low_count = (size_t)n_head * kv_dim;
    const size_t kv_count = (size_t)cache_cap * kv_dim;
    const size_t rope_count = (size_t)cache_cap * qk_rope;
    const size_t out_count = (size_t)n_head * value_dim;
    unsigned char *model = (unsigned char *)malloc((size_t)model_size);
    float *host_q = (float *)malloc(2u * q_count * sizeof(float));
    float *host_low = (float *)malloc(2u * low_count * sizeof(float));
    float *host_kv = (float *)malloc(kv_count * sizeof(float));
    float *host_rope = (float *)malloc(rope_count * sizeof(float));
    uint32_t *host_selected =
        (uint32_t *)malloc((size_t)cache_cap * sizeof(uint32_t));
    float *host_fused = (float *)malloc(2u * out_count * sizeof(float));
    float *host_staged = (float *)malloc(2u * out_count * sizeof(float));
    CHECK(model && host_q && host_low && host_kv && host_rope &&
          host_selected && host_fused && host_staged,
          "glm staged attention host alloc");
    pack_q8_identity_scale(model, kv_dim, value_rows);
    for (size_t i = 0; i < 2u * q_count; i++)
        host_q[i] = (float)((int)(i % 19u) - 9) * 0.015625f;
    for (size_t i = 0; i < 2u * low_count; i++)
        host_low[i] = (float)((int)((i * 3u) % 23u) - 11) * 0.0078125f;
    for (size_t i = 0; i < kv_count; i++)
        host_kv[i] = (float)((int)((i * 5u) % 29u) - 14) * 0.00390625f;
    for (size_t i = 0; i < rope_count; i++)
        host_rope[i] = (float)((int)((i * 7u) % 17u) - 8) * 0.0078125f;
    for (uint32_t i = 0; i < cases[1]; i++)
        host_selected[i] = (i * 29u + 7u) % cache_cap;
    CHECK(ds4_gpu_set_model_map(model, model_size),
          "glm staged attention set model map");

    ds4_gpu_tensor q; memset(&q, 0, sizeof(q));
    ds4_gpu_tensor low; memset(&low, 0, sizeof(low));
    ds4_gpu_tensor kv; memset(&kv, 0, sizeof(kv));
    ds4_gpu_tensor rope; memset(&rope, 0, sizeof(rope));
    ds4_gpu_tensor selected; memset(&selected, 0, sizeof(selected));
    ds4_gpu_tensor fused; memset(&fused, 0, sizeof(fused));
    ds4_gpu_tensor staged; memset(&staged, 0, sizeof(staged));
    CHECK(ds4_gpu_tensor_alloc_on(&q, 0, 2u * q_count * sizeof(float)) == 0,
          "glm staged attention alloc q");
    CHECK(ds4_gpu_tensor_alloc_on(&low, 0, 2u * low_count * sizeof(float)) == 0,
          "glm staged attention alloc low");
    CHECK(ds4_gpu_tensor_alloc_on(&kv, 0, kv_count * sizeof(float)) == 0,
          "glm staged attention alloc kv");
    CHECK(ds4_gpu_tensor_alloc_on(&rope, 0, rope_count * sizeof(float)) == 0,
          "glm staged attention alloc rope");
    CHECK(ds4_gpu_tensor_alloc_on(&selected, 0,
                                  (uint64_t)cache_cap * sizeof(uint32_t)) == 0,
          "glm staged attention alloc selected");
    CHECK(ds4_gpu_tensor_alloc_on(&fused, 0, 2u * out_count * sizeof(float)) == 0,
          "glm staged attention alloc fused");
    CHECK(ds4_gpu_tensor_alloc_on(&staged, 0, 2u * out_count * sizeof(float)) == 0,
          "glm staged attention alloc staged");
    int ok = ds4_gpu_tensor_write(&q, 0, host_q,
                                  2u * q_count * sizeof(float)) &&
             ds4_gpu_tensor_write(&low, 0, host_low,
                                  2u * low_count * sizeof(float)) &&
             ds4_gpu_tensor_write(&kv, 0, host_kv, kv_count * sizeof(float)) &&
             ds4_gpu_tensor_write(&rope, 0, host_rope,
                                  rope_count * sizeof(float)) &&
             ds4_gpu_tensor_write(&selected, 0, host_selected,
                                  (uint64_t)cases[1] * sizeof(uint32_t));
    CHECK(ok, "glm staged attention upload");

    for (uint32_t ci = 0; ci < 2u; ci++) {
        const uint32_t n_selected = cases[ci];
        setenv("DS4_GLM_ATTN_NO_STAGED_DECODE", "1", 1);
        ok = ds4_gpu_glm_attention_indexed_decode_typed_tensor(
                 &fused, &q, &low, &kv, &rope,
                 model, model_size, 0u, 8u, &selected,
                 n_selected, cache_cap, false, n_head, kv_dim,
                 qk_nope, qk_rope, value_dim, 0u, 8000000.0f, 1.0f,
                 0.0f, 1.0f, 32.0f, 1.0f) &&
             ds4_gpu_tensor_read(&fused, 0, host_fused,
                                 out_count * sizeof(float));
        CHECK(ok, "glm staged attention fused compute");
        unsetenv("DS4_GLM_ATTN_NO_STAGED_DECODE");
        ok = ds4_gpu_glm_attention_indexed_decode_typed_tensor(
                 &staged, &q, &low, &kv, &rope,
                 model, model_size, 0u, 8u, &selected,
                 n_selected, cache_cap, false, n_head, kv_dim,
                 qk_nope, qk_rope, value_dim, 0u, 8000000.0f, 1.0f,
                 0.0f, 1.0f, 32.0f, 1.0f) &&
             ds4_gpu_tensor_read(&staged, 0, host_staged,
                                 out_count * sizeof(float));
        CHECK(ok, "glm staged attention staged compute");
        if (memcmp(host_fused, host_staged,
                   out_count * sizeof(float)) != 0) {
            fprintf(stderr,
                    "FAIL: glm staged attention mismatch selected=%u\n",
                    n_selected);
            return 1;
        }
    }

    const uint32_t tok2_cases[] = {255u, 513u};
    for (uint32_t ci = 0; ci < 2u; ci++) {
        const uint32_t n_selected = tok2_cases[ci];
        for (uint32_t i = 0; i <= n_selected; i++) host_selected[i] = i;
        CHECK(ds4_gpu_tensor_write(&selected, 0, host_selected,
                                   (uint64_t)(n_selected + 1u) * sizeof(uint32_t)),
              "glm tok2 attention selected upload");
        ds4_gpu_tensor *q0 = ds4_gpu_tensor_view(
                &q, 0, q_count * sizeof(float));
        ds4_gpu_tensor *q1 = ds4_gpu_tensor_view(
                &q, q_count * sizeof(float), q_count * sizeof(float));
        ds4_gpu_tensor *low0 = ds4_gpu_tensor_view(
                &low, 0, low_count * sizeof(float));
        ds4_gpu_tensor *low1 = ds4_gpu_tensor_view(
                &low, low_count * sizeof(float), low_count * sizeof(float));
        ds4_gpu_tensor *ref0 = ds4_gpu_tensor_view(
                &fused, 0, out_count * sizeof(float));
        ds4_gpu_tensor *ref1 = ds4_gpu_tensor_view(
                &fused, out_count * sizeof(float), out_count * sizeof(float));
        CHECK(q0 && q1 && low0 && low1 && ref0 && ref1,
              "glm tok2 attention views");
        ok = ds4_gpu_glm_attention_indexed_decode_typed_tensor(
                     ref0, q0, low0, &kv, &rope,
                     model, model_size, 0u, 8u, &selected,
                     n_selected, cache_cap, false, n_head, kv_dim,
                     qk_nope, qk_rope, value_dim, 0u, 8000000.0f, 1.0f,
                     0.0f, 1.0f, 32.0f, 1.0f) &&
             ds4_gpu_glm_attention_indexed_decode_typed_tensor(
                     ref1, q1, low1, &kv, &rope,
                     model, model_size, 0u, 8u, &selected,
                     n_selected + 1u, cache_cap, false, n_head, kv_dim,
                     qk_nope, qk_rope, value_dim, 0u, 8000000.0f, 1.0f,
                     0.0f, 1.0f, 32.0f, 1.0f);
        CHECK(ok, "glm tok2 attention reference compute");
        ds4_gpu_set_glm_mtp_verify_mode(true);
        ok = ds4_gpu_glm_attention_indexed_decode_typed_tensor(
                     &staged, &q, &low, &kv, &rope,
                     model, model_size, 0u, 8u, &selected,
                     n_selected, cache_cap, false, n_head, kv_dim,
                     qk_nope, qk_rope, value_dim, 0u, 8000000.0f, 1.0f,
                     0.0f, 1.0f, 32.0f, 1.0f);
        ds4_gpu_set_glm_mtp_verify_mode(false);
        ok = ok && ds4_gpu_tensor_read(&fused, 0, host_fused,
                                       2u * out_count * sizeof(float)) &&
             ds4_gpu_tensor_read(&staged, 0, host_staged,
                                 2u * out_count * sizeof(float));
        ds4_gpu_tensor_free(q0);
        ds4_gpu_tensor_free(q1);
        ds4_gpu_tensor_free(low0);
        ds4_gpu_tensor_free(low1);
        ds4_gpu_tensor_free(ref0);
        ds4_gpu_tensor_free(ref1);
        CHECK(ok, "glm tok2 attention batched compute");
        CHECK(memcmp(host_fused, host_staged,
                     2u * out_count * sizeof(float)) == 0,
              "glm tok2 attention exact output");
    }

    ds4_gpu_tensor_free_in_place(&q);
    ds4_gpu_tensor_free_in_place(&low);
    ds4_gpu_tensor_free_in_place(&kv);
    ds4_gpu_tensor_free_in_place(&rope);
    ds4_gpu_tensor_free_in_place(&selected);
    ds4_gpu_tensor_free_in_place(&fused);
    ds4_gpu_tensor_free_in_place(&staged);
    free(model);
    free(host_q);
    free(host_low);
    free(host_kv);
    free(host_rope);
    free(host_selected);
    free(host_fused);
    free(host_staged);
    ds4_gpu_cleanup();
    fprintf(stderr, "  glm_decode_attention_staged OK\n");
    return 0;
}

static void pack_f16_small_mask(uint16_t *w,
                                uint64_t in_dim,
                                uint64_t out_dim) {
    for (uint64_t r = 0; r < out_dim; r++) {
        for (uint64_t i = 0; i < in_dim; i++) {
            int on = ((r * 13u + i * 7u + 5u) % 11u) < 3u;
            w[r * in_dim + i] = on ? 0x3c00u : 0x0000u;
        }
    }
}

static int run_q8_kslice(void) {
    int dev_count = 0;
    (void)cudaGetDeviceCount(&dev_count);
    if (dev_count < 1) {
        fprintf(stderr, "  skipping q8_kslice (need CUDA device)\n");
        return 0;
    }

    ds4_gpu_config cfg; memset(&cfg, 0, sizeof(cfg));
    cfg.n_gpus = 1;
    cfg.device_indices[0] = 0;
    CHECK(ds4_gpu_init_multi(&cfg), "q8_kslice init_multi");

    const uint64_t in_dim = 128;
    const uint64_t out_dim = 96;
    const uint64_t split = 64;
    const uint64_t blocks = (in_dim + 31u) / 32u;
    const uint64_t model_size = out_dim * blocks * 34u;
    unsigned char *model = (unsigned char *)malloc((size_t)model_size);
    float *host_x = (float *)malloc((size_t)in_dim * sizeof(float));
    float *host_full = (float *)malloc((size_t)out_dim * sizeof(float));
    float *host_sum = (float *)malloc((size_t)out_dim * sizeof(float));
    CHECK(model && host_x && host_full && host_sum, "q8_kslice host alloc");
    pack_q8_identity_scale(model, in_dim, out_dim);
    for (uint64_t i = 0; i < in_dim; i++) {
        host_x[i] = (float)((int)(i % 37u) - 18) * 0.03125f;
    }
    CHECK(ds4_gpu_set_model_map(model, model_size), "q8_kslice set model map");

    ds4_gpu_tensor x; memset(&x, 0, sizeof(x));
    ds4_gpu_tensor full; memset(&full, 0, sizeof(full));
    ds4_gpu_tensor p0; memset(&p0, 0, sizeof(p0));
    ds4_gpu_tensor p1; memset(&p1, 0, sizeof(p1));
    ds4_gpu_tensor sum; memset(&sum, 0, sizeof(sum));
    CHECK(ds4_gpu_tensor_alloc_on(&x, 0, in_dim * sizeof(float)) == 0,
          "q8_kslice alloc x");
    CHECK(ds4_gpu_tensor_alloc_on(&full, 0, out_dim * sizeof(float)) == 0,
          "q8_kslice alloc full");
    CHECK(ds4_gpu_tensor_alloc_on(&p0, 0, out_dim * sizeof(float)) == 0,
          "q8_kslice alloc p0");
    CHECK(ds4_gpu_tensor_alloc_on(&p1, 0, out_dim * sizeof(float)) == 0,
          "q8_kslice alloc p1");
    CHECK(ds4_gpu_tensor_alloc_on(&sum, 0, out_dim * sizeof(float)) == 0,
          "q8_kslice alloc sum");
    CHECK(ds4_gpu_tensor_write(&x, 0, host_x, in_dim * sizeof(float)),
          "q8_kslice write x");

    ds4_gpu_tensor *x0 = ds4_gpu_tensor_view(&x, 0, split * sizeof(float));
    ds4_gpu_tensor *x1 = ds4_gpu_tensor_view(&x, split * sizeof(float),
                                             (in_dim - split) * sizeof(float));
    CHECK(x0 && x1, "q8_kslice views");
    int ok = ds4_gpu_matmul_q8_0_tensor(&full, model, model_size, 0,
                                        in_dim, out_dim, &x, 1) &&
             ds4_gpu_matmul_q8_0_kslice_rows_tensor(&p0, model, model_size, 0,
                                               in_dim, out_dim, 0, split, x0, 1) &&
             ds4_gpu_matmul_q8_0_kslice_rows_tensor(&p1, model, model_size, 0,
                                               in_dim, out_dim, split,
                                               in_dim - split, x1, 1) &&
             ds4_gpu_add_tensor(&sum, &p0, &p1, (uint32_t)out_dim) &&
             ds4_gpu_tensor_read(&full, 0, host_full, out_dim * sizeof(float)) &&
             ds4_gpu_tensor_read(&sum, 0, host_sum, out_dim * sizeof(float));
    CHECK(ok, "q8_kslice compute");
    for (uint64_t i = 0; i < out_dim; i++) {
        float diff = fabsf(host_full[i] - host_sum[i]);
        if (diff > 1.0e-4f) {
            fprintf(stderr,
                    "FAIL: q8_kslice mismatch row=%llu full=%f sum=%f diff=%g\n",
                    (unsigned long long)i, host_full[i], host_sum[i], diff);
            return 1;
        }
    }

    ds4_gpu_tensor_free(x0);
    ds4_gpu_tensor_free(x1);
    ds4_gpu_tensor_free_in_place(&x);
    ds4_gpu_tensor_free_in_place(&full);
    ds4_gpu_tensor_free_in_place(&p0);
    ds4_gpu_tensor_free_in_place(&p1);
    ds4_gpu_tensor_free_in_place(&sum);
    free(model);
    free(host_x);
    free(host_full);
    free(host_sum);
    ds4_gpu_cleanup();
    fprintf(stderr, "  q8_kslice OK\n");
    return 0;
}

static int run_q8_matmul_top1_fused(void) {
    int dev_count = 0;
    (void)cudaGetDeviceCount(&dev_count);
    if (dev_count < 1) {
        fprintf(stderr, "  skipping q8_matmul_top1_fused (need CUDA device)\n");
        return 0;
    }

    ds4_gpu_config cfg; memset(&cfg, 0, sizeof(cfg));
    cfg.n_gpus = 1;
    cfg.device_indices[0] = 0;
    CHECK(ds4_gpu_init_multi(&cfg), "q8_top1_fused init_multi");

    const uint64_t in_dim = 160;
    const uint64_t out_dim = 257;
    const uint32_t index_offset = 1234u;
    const uint64_t blocks = (in_dim + 31u) / 32u;
    const uint64_t model_size = out_dim * blocks * 34u;
    unsigned char *model = (unsigned char *)malloc((size_t)model_size);
    float *host_x = (float *)malloc((size_t)in_dim * sizeof(float));
    uint32_t ref_id = 0, fused_id = 0;
    float ref_value = 0.0f, fused_value = 0.0f;
    CHECK(model && host_x, "q8_top1_fused host alloc");
    pack_q8_identity_scale(model, in_dim, out_dim);
    for (uint64_t i = 0; i < in_dim; i++) {
        host_x[i] = (float)((int)(i % 43u) - 21) * 0.0234375f;
    }
    CHECK(ds4_gpu_set_model_map(model, model_size), "q8_top1_fused set model map");

    ds4_gpu_tensor x; memset(&x, 0, sizeof(x));
    ds4_gpu_tensor logits; memset(&logits, 0, sizeof(logits));
    ds4_gpu_tensor ref_selected; memset(&ref_selected, 0, sizeof(ref_selected));
    ds4_gpu_tensor ref_values; memset(&ref_values, 0, sizeof(ref_values));
    ds4_gpu_tensor fused_selected; memset(&fused_selected, 0, sizeof(fused_selected));
    ds4_gpu_tensor fused_values; memset(&fused_values, 0, sizeof(fused_values));
    CHECK(ds4_gpu_tensor_alloc_on(&x, 0, in_dim * sizeof(float)) == 0,
          "q8_top1_fused alloc x");
    CHECK(ds4_gpu_tensor_alloc_on(&logits, 0, out_dim * sizeof(float)) == 0,
          "q8_top1_fused alloc logits");
    CHECK(ds4_gpu_tensor_alloc_on(&ref_selected, 0, sizeof(uint32_t)) == 0,
          "q8_top1_fused alloc ref selected");
    CHECK(ds4_gpu_tensor_alloc_on(&ref_values, 0, sizeof(float)) == 0,
          "q8_top1_fused alloc ref values");
    CHECK(ds4_gpu_tensor_alloc_on(&fused_selected, 0, sizeof(uint32_t)) == 0,
          "q8_top1_fused alloc fused selected");
    CHECK(ds4_gpu_tensor_alloc_on(&fused_values, 0, sizeof(float)) == 0,
          "q8_top1_fused alloc fused values");
    CHECK(ds4_gpu_tensor_write(&x, 0, host_x, in_dim * sizeof(float)),
          "q8_top1_fused write x");

    int ok = ds4_gpu_matmul_q8_0_tensor(&logits, model, model_size, 0,
                                        in_dim, out_dim, &x, 1) &&
             ds4_gpu_indexer_top1_value_tensor(&ref_selected, &ref_values,
                                               &logits, (uint32_t)out_dim, 1,
                                               index_offset) &&
             ds4_gpu_matmul_q8_0_top1_tensor(&fused_selected, &fused_values,
                                             model, model_size, 0, in_dim,
                                             out_dim, &x, index_offset) &&
             ds4_gpu_tensor_read(&ref_selected, 0, &ref_id, sizeof(ref_id)) &&
             ds4_gpu_tensor_read(&fused_selected, 0, &fused_id, sizeof(fused_id)) &&
             ds4_gpu_tensor_read(&ref_values, 0, &ref_value, sizeof(ref_value)) &&
             ds4_gpu_tensor_read(&fused_values, 0, &fused_value, sizeof(fused_value));
    CHECK(ok, "q8_top1_fused compute");
    CHECK(ref_id == fused_id, "q8_top1_fused selected");
    if (fabsf(ref_value - fused_value) > 1.0e-6f) {
        fprintf(stderr,
                "FAIL: q8_top1_fused value mismatch id=%u ref=%f fused=%f diff=%g\n",
                ref_id, ref_value, fused_value, fabsf(ref_value - fused_value));
        return 1;
    }

    ds4_gpu_tensor_free_in_place(&x);
    ds4_gpu_tensor_free_in_place(&logits);
    ds4_gpu_tensor_free_in_place(&ref_selected);
    ds4_gpu_tensor_free_in_place(&ref_values);
    ds4_gpu_tensor_free_in_place(&fused_selected);
    ds4_gpu_tensor_free_in_place(&fused_values);
    free(model);
    free(host_x);
    ds4_gpu_cleanup();
    fprintf(stderr, "  q8_matmul_top1_fused OK\n");
    return 0;
}

static int run_f16_small_out(void) {
    int dev_count = 0;
    (void)cudaGetDeviceCount(&dev_count);
    if (dev_count < 1) {
        fprintf(stderr, "  skipping f16_small_out (need CUDA device)\n");
        return 0;
    }

    setenv("DS4_CUDA_F16_SMALL_OUT", "1", 1);
    unsetenv("DS4_CUDA_NO_F16_SMALL_OUT");

    ds4_gpu_config cfg; memset(&cfg, 0, sizeof(cfg));
    cfg.n_gpus = 1;
    cfg.device_indices[0] = 0;
    CHECK(ds4_gpu_init_multi(&cfg), "f16_small_out init_multi");

    const uint64_t in_dim = 32768;
    const uint64_t out_dim = 16;
    const uint64_t model_size = in_dim * out_dim * sizeof(uint16_t);
    uint16_t *model = (uint16_t *)malloc((size_t)model_size);
    float *host_x = (float *)malloc((size_t)in_dim * sizeof(float));
    float *host_out = (float *)malloc((size_t)out_dim * sizeof(float));
    float *host_ref = (float *)calloc((size_t)out_dim, sizeof(float));
    CHECK(model && host_x && host_out && host_ref, "f16_small_out host alloc");
    pack_f16_small_mask(model, in_dim, out_dim);
    for (uint64_t i = 0; i < in_dim; i++) {
        host_x[i] = (float)((int)(i % 31u) - 15) * 0.03125f;
    }
    for (uint64_t r = 0; r < out_dim; r++) {
        float sum = 0.0f;
        for (uint64_t i = 0; i < in_dim; i++) {
            if (model[r * in_dim + i] == 0x3c00u) sum += host_x[i];
        }
        host_ref[r] = sum;
    }
    CHECK(ds4_gpu_set_model_map(model, model_size), "f16_small_out set model map");

    ds4_gpu_tensor x; memset(&x, 0, sizeof(x));
    ds4_gpu_tensor out; memset(&out, 0, sizeof(out));
    CHECK(ds4_gpu_tensor_alloc_on(&x, 0, in_dim * sizeof(float)) == 0,
          "f16_small_out alloc x");
    CHECK(ds4_gpu_tensor_alloc_on(&out, 0, out_dim * sizeof(float)) == 0,
          "f16_small_out alloc out");
    CHECK(ds4_gpu_tensor_write(&x, 0, host_x, in_dim * sizeof(float)),
          "f16_small_out write x");
    int ok = ds4_gpu_matmul_f16_tensor(&out, model, model_size, 0,
                                       in_dim, out_dim, &x, 1) &&
             ds4_gpu_tensor_read(&out, 0, host_out, out_dim * sizeof(float));
    CHECK(ok, "f16_small_out compute");
    for (uint64_t r = 0; r < out_dim; r++) {
        float diff = fabsf(host_out[r] - host_ref[r]);
        if (diff > 1.0e-3f) {
            fprintf(stderr,
                    "FAIL: f16_small_out mismatch row=%llu got=%f want=%f diff=%g\n",
                    (unsigned long long)r, host_out[r], host_ref[r], diff);
            return 1;
        }
    }

    ds4_gpu_tensor_free_in_place(&x);
    ds4_gpu_tensor_free_in_place(&out);
    free(model);
    free(host_x);
    free(host_out);
    free(host_ref);
    ds4_gpu_cleanup();
    fprintf(stderr, "  f16_small_out OK\n");
    return 0;
}

static int run_f16_small_batch(void) {
    int dev_count = 0;
    (void)cudaGetDeviceCount(&dev_count);
    if (dev_count < 1) {
        fprintf(stderr, "  skipping f16_small_batch (need CUDA device)\n");
        return 0;
    }

    setenv("DS4_CUDA_F16_SMALL_BATCH", "1", 1);
    unsetenv("DS4_CUDA_NO_F16_SMALL_BATCH");
    unsetenv("DS4_CUDA_NO_F16_CUBLAS_BATCH");

    ds4_gpu_config cfg; memset(&cfg, 0, sizeof(cfg));
    cfg.n_gpus = 1;
    cfg.device_indices[0] = 0;
    CHECK(ds4_gpu_init_multi(&cfg), "f16_small_batch init_multi");

    const uint64_t in_dim = 8192;
    const uint64_t out_dim = 24;
    const uint64_t n_tok = 9;
    const uint64_t model_size = in_dim * out_dim * sizeof(uint16_t);
    uint16_t *model = (uint16_t *)malloc((size_t)model_size);
    float *host_x = (float *)malloc((size_t)(n_tok * in_dim) * sizeof(float));
    float *host_fast = (float *)malloc((size_t)(n_tok * out_dim) * sizeof(float));
    float *host_ref = (float *)malloc((size_t)(n_tok * out_dim) * sizeof(float));
    CHECK(model && host_x && host_fast && host_ref, "f16_small_batch host alloc");

    const uint16_t vals[5] = {0x0000u, 0x3c00u, 0xbc00u, 0x3800u, 0xb800u};
    for (uint64_t r = 0; r < out_dim; r++) {
        for (uint64_t i = 0; i < in_dim; i++) {
            model[r * in_dim + i] = vals[(r * 17u + i * 5u + 3u) % 5u];
        }
    }
    for (uint64_t t = 0; t < n_tok; t++) {
        for (uint64_t i = 0; i < in_dim; i++) {
            host_x[t * in_dim + i] = (float)((int)((t * 19u + i * 7u) % 53u) - 26) * 0.015625f;
        }
    }
    CHECK(ds4_gpu_set_model_map(model, model_size), "f16_small_batch set model map");

    ds4_gpu_tensor x; memset(&x, 0, sizeof(x));
    ds4_gpu_tensor fast; memset(&fast, 0, sizeof(fast));
    ds4_gpu_tensor ref; memset(&ref, 0, sizeof(ref));
    CHECK(ds4_gpu_tensor_alloc_on(&x, 0, n_tok * in_dim * sizeof(float)) == 0,
          "f16_small_batch alloc x");
    CHECK(ds4_gpu_tensor_alloc_on(&fast, 0, n_tok * out_dim * sizeof(float)) == 0,
          "f16_small_batch alloc fast");
    CHECK(ds4_gpu_tensor_alloc_on(&ref, 0, n_tok * out_dim * sizeof(float)) == 0,
          "f16_small_batch alloc ref");
    CHECK(ds4_gpu_tensor_write(&x, 0, host_x, n_tok * in_dim * sizeof(float)),
          "f16_small_batch write x");

    int ok = ds4_gpu_matmul_f16_tensor(&fast, model, model_size, 0,
                                       in_dim, out_dim, &x, n_tok);
    setenv("DS4_CUDA_NO_F16_SMALL_BATCH", "1", 1);
    setenv("DS4_CUDA_NO_F16_CUBLAS_BATCH", "1", 1);
    ok = ok && ds4_gpu_matmul_f16_tensor(&ref, model, model_size, 0,
                                         in_dim, out_dim, &x, n_tok) &&
         ds4_gpu_tensor_read(&fast, 0, host_fast, n_tok * out_dim * sizeof(float)) &&
         ds4_gpu_tensor_read(&ref, 0, host_ref, n_tok * out_dim * sizeof(float));
    CHECK(ok, "f16_small_batch compute");
    for (uint64_t i = 0; i < n_tok * out_dim; i++) {
        if (memcmp(&host_fast[i], &host_ref[i], sizeof(float)) != 0) {
            fprintf(stderr,
                    "FAIL: f16_small_batch mismatch i=%llu got=%f want=%f diff=%g\n",
                    (unsigned long long)i,
                    host_fast[i],
                    host_ref[i],
                    fabsf(host_fast[i] - host_ref[i]));
            return 1;
        }
    }

    ds4_gpu_tensor_free_in_place(&x);
    ds4_gpu_tensor_free_in_place(&fast);
    ds4_gpu_tensor_free_in_place(&ref);
    free(model);
    free(host_x);
    free(host_fast);
    free(host_ref);
    ds4_gpu_cleanup();
    fprintf(stderr, "  f16_small_batch OK\n");
    return 0;
}

static int run_f16_pair_batch(void) {
    int dev_count = 0;
    (void)cudaGetDeviceCount(&dev_count);
    if (dev_count < 1) {
        fprintf(stderr, "  skipping f16_pair_batch (need CUDA device)\n");
        return 0;
    }

    unsetenv("DS4_CUDA_NO_F16_PAIR_MATMUL");
    unsetenv("DS4_CUDA_NO_F16_CUBLAS_BATCH");
    unsetenv("DS4_CUDA_F16_SMALL_BATCH");
    unsetenv("DS4_CUDA_NO_F16_SMALL_BATCH");

    ds4_gpu_config cfg; memset(&cfg, 0, sizeof(cfg));
    cfg.n_gpus = 1;
    cfg.device_indices[0] = 0;
    CHECK(ds4_gpu_init_multi(&cfg), "f16_pair_batch init_multi");

    const uint64_t in_dim = 512;
    const uint64_t out_dim = 80;
    const uint64_t n_tok = 11;
    const uint64_t weight_bytes = in_dim * out_dim * sizeof(uint16_t);
    const uint64_t model_size = weight_bytes * 2u;
    uint16_t *model = (uint16_t *)malloc((size_t)model_size);
    float *host_x = (float *)malloc((size_t)(n_tok * in_dim) * sizeof(float));
    float *fast0 = (float *)malloc((size_t)(n_tok * out_dim) * sizeof(float));
    float *fast1 = (float *)malloc((size_t)(n_tok * out_dim) * sizeof(float));
    float *ref0 = (float *)malloc((size_t)(n_tok * out_dim) * sizeof(float));
    float *ref1 = (float *)malloc((size_t)(n_tok * out_dim) * sizeof(float));
    CHECK(model && host_x && fast0 && fast1 && ref0 && ref1, "f16_pair_batch host alloc");

    const uint16_t vals[7] = {0x0000u, 0x3c00u, 0xbc00u, 0x3800u, 0xb800u, 0x3400u, 0xb400u};
    for (uint64_t r = 0; r < out_dim; r++) {
        for (uint64_t i = 0; i < in_dim; i++) {
            model[r * in_dim + i] = vals[(r * 11u + i * 3u + 1u) % 7u];
            model[(weight_bytes / sizeof(uint16_t)) + r * in_dim + i] =
                vals[(r * 13u + i * 5u + 4u) % 7u];
        }
    }
    for (uint64_t t = 0; t < n_tok; t++) {
        for (uint64_t i = 0; i < in_dim; i++) {
            host_x[t * in_dim + i] =
                (float)((int)((t * 23u + i * 17u) % 71u) - 35) * 0.0078125f;
        }
    }
    CHECK(ds4_gpu_set_model_map(model, model_size), "f16_pair_batch set model map");

    ds4_gpu_tensor x; memset(&x, 0, sizeof(x));
    ds4_gpu_tensor out_fast0; memset(&out_fast0, 0, sizeof(out_fast0));
    ds4_gpu_tensor out_fast1; memset(&out_fast1, 0, sizeof(out_fast1));
    ds4_gpu_tensor out_ref0; memset(&out_ref0, 0, sizeof(out_ref0));
    ds4_gpu_tensor out_ref1; memset(&out_ref1, 0, sizeof(out_ref1));
    CHECK(ds4_gpu_tensor_alloc_on(&x, 0, n_tok * in_dim * sizeof(float)) == 0,
          "f16_pair_batch alloc x");
    CHECK(ds4_gpu_tensor_alloc_on(&out_fast0, 0, n_tok * out_dim * sizeof(float)) == 0,
          "f16_pair_batch alloc fast0");
    CHECK(ds4_gpu_tensor_alloc_on(&out_fast1, 0, n_tok * out_dim * sizeof(float)) == 0,
          "f16_pair_batch alloc fast1");
    CHECK(ds4_gpu_tensor_alloc_on(&out_ref0, 0, n_tok * out_dim * sizeof(float)) == 0,
          "f16_pair_batch alloc ref0");
    CHECK(ds4_gpu_tensor_alloc_on(&out_ref1, 0, n_tok * out_dim * sizeof(float)) == 0,
          "f16_pair_batch alloc ref1");
    CHECK(ds4_gpu_tensor_write(&x, 0, host_x, n_tok * in_dim * sizeof(float)),
          "f16_pair_batch write x");

    int ok = ds4_gpu_matmul_f16_pair_tensor(&out_fast0,
                                            &out_fast1,
                                            model,
                                            model_size,
                                            0,
                                            weight_bytes,
                                            in_dim,
                                            out_dim,
                                            &x,
                                            n_tok);
    setenv("DS4_CUDA_NO_F16_PAIR_MATMUL", "1", 1);
    ok = ok && ds4_gpu_matmul_f16_pair_tensor(&out_ref0,
                                               &out_ref1,
                                               model,
                                               model_size,
                                               0,
                                               weight_bytes,
                                               in_dim,
                                               out_dim,
                                               &x,
                                               n_tok) &&
         ds4_gpu_tensor_read(&out_fast0, 0, fast0, n_tok * out_dim * sizeof(float)) &&
         ds4_gpu_tensor_read(&out_fast1, 0, fast1, n_tok * out_dim * sizeof(float)) &&
         ds4_gpu_tensor_read(&out_ref0, 0, ref0, n_tok * out_dim * sizeof(float)) &&
         ds4_gpu_tensor_read(&out_ref1, 0, ref1, n_tok * out_dim * sizeof(float));
    CHECK(ok, "f16_pair_batch compute");
    for (uint64_t i = 0; i < n_tok * out_dim; i++) {
        if (memcmp(&fast0[i], &ref0[i], sizeof(float)) != 0 ||
            memcmp(&fast1[i], &ref1[i], sizeof(float)) != 0) {
            fprintf(stderr,
                    "FAIL: f16_pair_batch mismatch i=%llu fast0=%f ref0=%f fast1=%f ref1=%f\n",
                    (unsigned long long)i, fast0[i], ref0[i], fast1[i], ref1[i]);
            return 1;
        }
    }

    ds4_gpu_tensor_free_in_place(&x);
    ds4_gpu_tensor_free_in_place(&out_fast0);
    ds4_gpu_tensor_free_in_place(&out_fast1);
    ds4_gpu_tensor_free_in_place(&out_ref0);
    ds4_gpu_tensor_free_in_place(&out_ref1);
    free(model);
    free(host_x);
    free(fast0);
    free(fast1);
    free(ref0);
    free(ref1);
    ds4_gpu_cleanup();
    fprintf(stderr, "  f16_pair_batch OK\n");
    return 0;
}

static int run_q8_rowsplit(void) {
    int dev_count = 0;
    (void)cudaGetDeviceCount(&dev_count);
    if (dev_count < 1) {
        fprintf(stderr, "  skipping q8_rowsplit (need CUDA device)\n");
        return 0;
    }

    ds4_gpu_config cfg; memset(&cfg, 0, sizeof(cfg));
    cfg.n_gpus = 1;
    cfg.device_indices[0] = 0;
    CHECK(ds4_gpu_init_multi(&cfg), "q8_rowsplit init_multi");

    const uint64_t in_dim = 96;
    const uint64_t out_dim = 97;
    const uint64_t split = 41;
    const uint64_t blocks = (in_dim + 31u) / 32u;
    const uint64_t row_bytes = blocks * 34u;
    const uint64_t model_size = out_dim * row_bytes;
    unsigned char *model = (unsigned char *)malloc((size_t)model_size);
    float *host_x = (float *)malloc((size_t)in_dim * sizeof(float));
    float *host_full = (float *)malloc((size_t)out_dim * sizeof(float));
    float *host_split = (float *)malloc((size_t)out_dim * sizeof(float));
    CHECK(model && host_x && host_full && host_split, "q8_rowsplit host alloc");
    pack_q8_identity_scale(model, in_dim, out_dim);
    for (uint64_t i = 0; i < in_dim; i++) {
        host_x[i] = (float)((int)(i % 43u) - 21) * 0.015625f;
    }
    CHECK(ds4_gpu_set_model_map(model, model_size), "q8_rowsplit set model map");

    ds4_gpu_tensor x; memset(&x, 0, sizeof(x));
    ds4_gpu_tensor full; memset(&full, 0, sizeof(full));
    ds4_gpu_tensor split_out; memset(&split_out, 0, sizeof(split_out));
    CHECK(ds4_gpu_tensor_alloc_on(&x, 0, in_dim * sizeof(float)) == 0,
          "q8_rowsplit alloc x");
    CHECK(ds4_gpu_tensor_alloc_on(&full, 0, out_dim * sizeof(float)) == 0,
          "q8_rowsplit alloc full");
    CHECK(ds4_gpu_tensor_alloc_on(&split_out, 0, out_dim * sizeof(float)) == 0,
          "q8_rowsplit alloc split");
    CHECK(ds4_gpu_tensor_write(&x, 0, host_x, in_dim * sizeof(float)),
          "q8_rowsplit write x");

    ds4_gpu_tensor *split0 = ds4_gpu_tensor_view(&split_out, 0,
                                                 split * sizeof(float));
    ds4_gpu_tensor *split1 = ds4_gpu_tensor_view(&split_out,
                                                 split * sizeof(float),
                                                 (out_dim - split) * sizeof(float));
    CHECK(split0 && split1, "q8_rowsplit views");
    int ok = ds4_gpu_matmul_q8_0_tensor(&full, model, model_size, 0,
                                        in_dim, out_dim, &x, 1) &&
             ds4_gpu_matmul_q8_0_tensor(split0, model, model_size, 0,
                                        in_dim, split, &x, 1) &&
             ds4_gpu_matmul_q8_0_tensor(split1, model, model_size,
                                        split * row_bytes,
                                        in_dim, out_dim - split, &x, 1) &&
             ds4_gpu_tensor_read(&full, 0, host_full, out_dim * sizeof(float)) &&
             ds4_gpu_tensor_read(&split_out, 0, host_split, out_dim * sizeof(float));
    CHECK(ok, "q8_rowsplit compute");
    for (uint64_t i = 0; i < out_dim; i++) {
        float diff = fabsf(host_full[i] - host_split[i]);
        if (diff > 1.0e-6f) {
            fprintf(stderr,
                    "FAIL: q8_rowsplit mismatch row=%llu full=%f split=%f diff=%g\n",
                    (unsigned long long)i, host_full[i], host_split[i], diff);
            return 1;
        }
    }

    ds4_gpu_tensor_free(split0);
    ds4_gpu_tensor_free(split1);
    ds4_gpu_tensor_free_in_place(&x);
    ds4_gpu_tensor_free_in_place(&full);
    ds4_gpu_tensor_free_in_place(&split_out);
    free(model);
    free(host_x);
    free(host_full);
    free(host_split);
    ds4_gpu_cleanup();
    fprintf(stderr, "  q8_rowsplit OK\n");
    return 0;
}

static int run_q8_pair_batch(void) {
    int dev_count = 0;
    (void)cudaGetDeviceCount(&dev_count);
    if (dev_count < 1) {
        fprintf(stderr, "  skipping q8_pair_batch (need CUDA device)\n");
        return 0;
    }

    setenv("DS4_CUDA_Q8_PAIR_BATCH", "1", 1);

    ds4_gpu_config cfg; memset(&cfg, 0, sizeof(cfg));
    cfg.n_gpus = 1;
    cfg.device_indices[0] = 0;
    CHECK(ds4_gpu_init_multi(&cfg), "q8_pair_batch init_multi");

    const uint64_t in_dim = 96;
    const uint64_t out0_dim = 40;
    const uint64_t out1_dim = 33;
    const uint64_t n_tok = 9;
    const uint64_t blocks = (in_dim + 31u) / 32u;
    const uint64_t row_bytes = blocks * 34u;
    const uint64_t off1 = out0_dim * row_bytes;
    const uint64_t model_size = off1 + out1_dim * row_bytes;
    unsigned char *model = (unsigned char *)malloc((size_t)model_size);
    float *host_x = (float *)malloc((size_t)(n_tok * in_dim) * sizeof(float));
    float *host_sep0 = (float *)malloc((size_t)(n_tok * out0_dim) * sizeof(float));
    float *host_sep1 = (float *)malloc((size_t)(n_tok * out1_dim) * sizeof(float));
    float *host_pair0 = (float *)malloc((size_t)(n_tok * out0_dim) * sizeof(float));
    float *host_pair1 = (float *)malloc((size_t)(n_tok * out1_dim) * sizeof(float));
    CHECK(model && host_x && host_sep0 && host_sep1 && host_pair0 && host_pair1,
          "q8_pair_batch host alloc");
    pack_q8_identity_scale(model, in_dim, out0_dim);
    pack_q8_identity_scale(model + off1, in_dim, out1_dim);
    for (uint64_t i = 0; i < n_tok * in_dim; i++) {
        host_x[i] = (float)((int)(i % 47u) - 23) * 0.03125f;
    }
    CHECK(ds4_gpu_set_model_map(model, model_size), "q8_pair_batch set model map");

    ds4_gpu_tensor x; memset(&x, 0, sizeof(x));
    ds4_gpu_tensor sep0; memset(&sep0, 0, sizeof(sep0));
    ds4_gpu_tensor sep1; memset(&sep1, 0, sizeof(sep1));
    ds4_gpu_tensor pair0; memset(&pair0, 0, sizeof(pair0));
    ds4_gpu_tensor pair1; memset(&pair1, 0, sizeof(pair1));
    CHECK(ds4_gpu_tensor_alloc_on(&x, 0, n_tok * in_dim * sizeof(float)) == 0,
          "q8_pair_batch alloc x");
    CHECK(ds4_gpu_tensor_alloc_on(&sep0, 0, n_tok * out0_dim * sizeof(float)) == 0,
          "q8_pair_batch alloc sep0");
    CHECK(ds4_gpu_tensor_alloc_on(&sep1, 0, n_tok * out1_dim * sizeof(float)) == 0,
          "q8_pair_batch alloc sep1");
    CHECK(ds4_gpu_tensor_alloc_on(&pair0, 0, n_tok * out0_dim * sizeof(float)) == 0,
          "q8_pair_batch alloc pair0");
    CHECK(ds4_gpu_tensor_alloc_on(&pair1, 0, n_tok * out1_dim * sizeof(float)) == 0,
          "q8_pair_batch alloc pair1");
    CHECK(ds4_gpu_tensor_write(&x, 0, host_x, n_tok * in_dim * sizeof(float)),
          "q8_pair_batch write x");

    int ok = ds4_gpu_matmul_q8_0_tensor(&sep0, model, model_size, 0,
                                        in_dim, out0_dim, &x, n_tok) &&
             ds4_gpu_matmul_q8_0_tensor(&sep1, model, model_size, off1,
                                        in_dim, out1_dim, &x, n_tok) &&
             ds4_gpu_matmul_q8_0_pair_tensor(&pair0, &pair1, model, model_size,
                                             0, off1, in_dim, out0_dim, out1_dim,
                                             &x, n_tok) &&
             ds4_gpu_tensor_read(&sep0, 0, host_sep0,
                                 n_tok * out0_dim * sizeof(float)) &&
             ds4_gpu_tensor_read(&sep1, 0, host_sep1,
                                 n_tok * out1_dim * sizeof(float)) &&
             ds4_gpu_tensor_read(&pair0, 0, host_pair0,
                                 n_tok * out0_dim * sizeof(float)) &&
             ds4_gpu_tensor_read(&pair1, 0, host_pair1,
                                 n_tok * out1_dim * sizeof(float));
    CHECK(ok, "q8_pair_batch compute");
    CHECK(memcmp(host_sep0, host_pair0,
                 (size_t)(n_tok * out0_dim) * sizeof(float)) == 0,
          "q8_pair_batch out0");
    CHECK(memcmp(host_sep1, host_pair1,
                 (size_t)(n_tok * out1_dim) * sizeof(float)) == 0,
          "q8_pair_batch out1");

    ds4_gpu_tensor_free_in_place(&x);
    ds4_gpu_tensor_free_in_place(&sep0);
    ds4_gpu_tensor_free_in_place(&sep1);
    ds4_gpu_tensor_free_in_place(&pair0);
    ds4_gpu_tensor_free_in_place(&pair1);
    free(model);
    free(host_x);
    free(host_sep0);
    free(host_sep1);
    free(host_pair0);
    free(host_pair1);
    ds4_gpu_cleanup();
    unsetenv("DS4_CUDA_Q8_PAIR_BATCH");
    fprintf(stderr, "  q8_pair_batch OK\n");
    return 0;
}

static int run_attention_output_tp(void) {
    int dev_count = 0;
    (void)cudaGetDeviceCount(&dev_count);
    if (dev_count < 1) {
        fprintf(stderr, "  skipping attention_output_tp (need CUDA device)\n");
        return 0;
    }

    ds4_gpu_config cfg; memset(&cfg, 0, sizeof(cfg));
    cfg.n_gpus = 1;
    cfg.device_indices[0] = 0;
    CHECK(ds4_gpu_init_multi(&cfg), "attention_tp init_multi");

    const uint64_t group_dim = 64;
    const uint64_t rank = 64;
    const uint32_t n_groups = 4;
    const uint32_t group_cnt = 2;
    const uint64_t out_dim = 48;
    const uint64_t low_dim = (uint64_t)n_groups * rank;
    const uint64_t a_blocks = (group_dim + 31u) / 32u;
    const uint64_t b_blocks = (low_dim + 31u) / 32u;
    const uint64_t a_bytes = (uint64_t)n_groups * rank * a_blocks * 34u;
    const uint64_t b_off = a_bytes;
    const uint64_t b_bytes = out_dim * b_blocks * 34u;
    const uint64_t model_size = a_bytes + b_bytes;
    unsigned char *model = (unsigned char *)malloc((size_t)model_size);
    float *host_heads = (float *)malloc((size_t)n_groups * group_dim * sizeof(float));
    float *host_full = (float *)malloc((size_t)out_dim * sizeof(float));
    float *host_sum = (float *)malloc((size_t)out_dim * sizeof(float));
    CHECK(model && host_heads && host_full && host_sum, "attention_tp host alloc");
    pack_q8_identity_scale(model, group_dim, (uint64_t)n_groups * rank);
    pack_q8_identity_scale(model + b_off, low_dim, out_dim);
    for (uint64_t i = 0; i < (uint64_t)n_groups * group_dim; i++) {
        host_heads[i] = (float)((int)(i % 29u) - 14) * 0.0625f;
    }
    CHECK(ds4_gpu_set_model_map(model, model_size), "attention_tp set model map");

    ds4_gpu_tensor heads; memset(&heads, 0, sizeof(heads));
    ds4_gpu_tensor low; memset(&low, 0, sizeof(low));
    ds4_gpu_tensor low0; memset(&low0, 0, sizeof(low0));
    ds4_gpu_tensor low1; memset(&low1, 0, sizeof(low1));
    ds4_gpu_tensor full; memset(&full, 0, sizeof(full));
    ds4_gpu_tensor p0; memset(&p0, 0, sizeof(p0));
    ds4_gpu_tensor p1; memset(&p1, 0, sizeof(p1));
    ds4_gpu_tensor sum; memset(&sum, 0, sizeof(sum));
    CHECK(ds4_gpu_tensor_alloc_on(&heads, 0, (uint64_t)n_groups * group_dim * sizeof(float)) == 0,
          "attention_tp alloc heads");
    CHECK(ds4_gpu_tensor_alloc_on(&low, 0, low_dim * sizeof(float)) == 0,
          "attention_tp alloc low");
    CHECK(ds4_gpu_tensor_alloc_on(&low0, 0, (uint64_t)group_cnt * rank * sizeof(float)) == 0,
          "attention_tp alloc low0");
    CHECK(ds4_gpu_tensor_alloc_on(&low1, 0, (uint64_t)group_cnt * rank * sizeof(float)) == 0,
          "attention_tp alloc low1");
    CHECK(ds4_gpu_tensor_alloc_on(&full, 0, out_dim * sizeof(float)) == 0,
          "attention_tp alloc full");
    CHECK(ds4_gpu_tensor_alloc_on(&p0, 0, out_dim * sizeof(float)) == 0,
          "attention_tp alloc p0");
    CHECK(ds4_gpu_tensor_alloc_on(&p1, 0, out_dim * sizeof(float)) == 0,
          "attention_tp alloc p1");
    CHECK(ds4_gpu_tensor_alloc_on(&sum, 0, out_dim * sizeof(float)) == 0,
          "attention_tp alloc sum");
    CHECK(ds4_gpu_tensor_write(&heads, 0, host_heads,
                               (uint64_t)n_groups * group_dim * sizeof(float)),
          "attention_tp write heads");

    int ok = ds4_gpu_attention_output_q8_batch_tensor(&full, &low, NULL, NULL,
                                                      model, model_size,
                                                      0, b_off,
                                                      group_dim, rank,
                                                      n_groups, out_dim,
                                                      &heads, 1) &&
             ds4_gpu_attention_output_q8_tp_tensor(&p0, &low0,
                                                   model, model_size,
                                                   0, b_off,
                                                   group_dim, rank,
                                                   n_groups, 0, group_cnt,
                                                   out_dim, &heads) &&
             ds4_gpu_attention_output_q8_tp_tensor(&p1, &low1,
                                                   model, model_size,
                                                   0, b_off,
                                                   group_dim, rank,
                                                   n_groups, group_cnt, group_cnt,
                                                   out_dim, &heads) &&
             ds4_gpu_add_tensor(&sum, &p0, &p1, (uint32_t)out_dim) &&
             ds4_gpu_tensor_read(&full, 0, host_full, out_dim * sizeof(float)) &&
             ds4_gpu_tensor_read(&sum, 0, host_sum, out_dim * sizeof(float));
    CHECK(ok, "attention_tp compute");
    for (uint64_t i = 0; i < out_dim; i++) {
        float diff = fabsf(host_full[i] - host_sum[i]);
        if (diff > 1.0e-3f) {
            fprintf(stderr,
                    "FAIL: attention_tp mismatch row=%llu full=%f sum=%f diff=%g\n",
                    (unsigned long long)i, host_full[i], host_sum[i], diff);
            return 1;
        }
    }

    ds4_gpu_tensor_free_in_place(&heads);
    ds4_gpu_tensor_free_in_place(&low);
    ds4_gpu_tensor_free_in_place(&low0);
    ds4_gpu_tensor_free_in_place(&low1);
    ds4_gpu_tensor_free_in_place(&full);
    ds4_gpu_tensor_free_in_place(&p0);
    ds4_gpu_tensor_free_in_place(&p1);
    ds4_gpu_tensor_free_in_place(&sum);
    free(model);
    free(host_heads);
    free(host_full);
    free(host_sum);
    ds4_gpu_cleanup();
    fprintf(stderr, "  attention_output_tp OK\n");
    return 0;
}

static int run_attention_output_tp_peer_read(void) {
    int dev_count = 0;
    (void)cudaGetDeviceCount(&dev_count);
    if (dev_count < 2) {
        fprintf(stderr, "  skipping attention_output_tp_peer_read (need >= 2 devices)\n");
        return 0;
    }

    unsetenv("DS4_FORCE_HOST_BOUNCE");

    ds4_gpu_config cfg; memset(&cfg, 0, sizeof(cfg));
    cfg.n_gpus = 2;
    cfg.device_indices[0] = 0;
    cfg.device_indices[1] = 1;
    CHECK(ds4_gpu_init_multi(&cfg), "attention_peer init_multi");
    if (!g_gpu_peer_ok[1][0]) {
        ds4_gpu_cleanup();
        fprintf(stderr, "  skipping attention_output_tp_peer_read (no peer access 1<-0)\n");
        return 0;
    }

    const uint64_t group_dim = 64;
    const uint64_t rank = 64;
    const uint32_t n_groups = 4;
    const uint32_t group_cnt = 2;
    const uint64_t out_dim = 48;
    const uint64_t low_dim = (uint64_t)n_groups * rank;
    const uint64_t a_blocks = (group_dim + 31u) / 32u;
    const uint64_t b_blocks = (low_dim + 31u) / 32u;
    const uint64_t a_bytes = (uint64_t)n_groups * rank * a_blocks * 34u;
    const uint64_t b_off = a_bytes;
    const uint64_t b_bytes = out_dim * b_blocks * 34u;
    const uint64_t model_size = a_bytes + b_bytes;
    unsigned char *model = (unsigned char *)malloc((size_t)model_size);
    float *host_heads = (float *)malloc((size_t)n_groups * group_dim * sizeof(float));
    float *host_full = (float *)malloc((size_t)out_dim * sizeof(float));
    float *host_sum = (float *)malloc((size_t)out_dim * sizeof(float));
    CHECK(model && host_heads && host_full && host_sum, "attention_peer host alloc");
    pack_q8_identity_scale(model, group_dim, (uint64_t)n_groups * rank);
    pack_q8_identity_scale(model + b_off, low_dim, out_dim);
    for (uint64_t i = 0; i < (uint64_t)n_groups * group_dim; i++) {
        host_heads[i] = (float)((int)(i % 31u) - 15) * 0.046875f;
    }
    CHECK(ds4_gpu_set_model_map(model, model_size), "attention_peer set model map");
    ds4_tensor_range r0 = { 0, model_size, 0 };
    ds4_tensor_range r1 = { 0, model_size, 1 };
    CHECK(ds4_gpu_device_cache_tensors(0, &r0, 1) == 0,
          "attention_peer cache model dev0");
    CHECK(ds4_gpu_device_cache_tensors(1, &r1, 1) == 0,
          "attention_peer cache model dev1");
    CHECK(ds4_gpu_set_current_device(0) == 0, "attention_peer current dev0");

    ds4_gpu_tensor heads; memset(&heads, 0, sizeof(heads));
    ds4_gpu_tensor low; memset(&low, 0, sizeof(low));
    ds4_gpu_tensor low0; memset(&low0, 0, sizeof(low0));
    ds4_gpu_tensor low1; memset(&low1, 0, sizeof(low1));
    ds4_gpu_tensor full; memset(&full, 0, sizeof(full));
    ds4_gpu_tensor p0; memset(&p0, 0, sizeof(p0));
    ds4_gpu_tensor p1; memset(&p1, 0, sizeof(p1));
    ds4_gpu_tensor staging; memset(&staging, 0, sizeof(staging));
    ds4_gpu_tensor sum; memset(&sum, 0, sizeof(sum));
    CHECK(ds4_gpu_tensor_alloc_on(&heads, 0, (uint64_t)n_groups * group_dim * sizeof(float)) == 0,
          "attention_peer alloc heads");
    CHECK(ds4_gpu_tensor_alloc_on(&low, 0, low_dim * sizeof(float)) == 0,
          "attention_peer alloc low");
    CHECK(ds4_gpu_tensor_alloc_on(&low0, 0, (uint64_t)group_cnt * rank * sizeof(float)) == 0,
          "attention_peer alloc low0");
    CHECK(ds4_gpu_tensor_alloc_on(&low1, 1, (uint64_t)group_cnt * rank * sizeof(float)) == 0,
          "attention_peer alloc low1");
    CHECK(ds4_gpu_tensor_alloc_on(&full, 0, out_dim * sizeof(float)) == 0,
          "attention_peer alloc full");
    CHECK(ds4_gpu_tensor_alloc_on(&p0, 0, out_dim * sizeof(float)) == 0,
          "attention_peer alloc p0");
    CHECK(ds4_gpu_tensor_alloc_on(&p1, 1, out_dim * sizeof(float)) == 0,
          "attention_peer alloc p1");
    CHECK(ds4_gpu_tensor_alloc_on(&staging, 0, out_dim * sizeof(float)) == 0,
          "attention_peer alloc staging");
    CHECK(ds4_gpu_tensor_alloc_on(&sum, 0, out_dim * sizeof(float)) == 0,
          "attention_peer alloc sum");
    CHECK(ds4_gpu_tensor_write(&heads, 0, host_heads,
                               (uint64_t)n_groups * group_dim * sizeof(float)),
          "attention_peer write heads");

    int ok = ds4_gpu_attention_output_q8_batch_tensor(&full, &low, NULL, NULL,
                                                      model, model_size,
                                                      0, b_off,
                                                      group_dim, rank,
                                                      n_groups, out_dim,
                                                      &heads, 1) &&
             ds4_gpu_attention_output_q8_tp_tensor(&p0, &low0,
                                                   model, model_size,
                                                   0, b_off,
                                                   group_dim, rank,
                                                   n_groups, 0, group_cnt,
                                                   out_dim, &heads) &&
             ds4_gpu_tensor_wait_xdev(&heads, 1) &&
             ds4_gpu_set_current_device(1) == 0 &&
             ds4_gpu_attention_output_q8_tp_tensor(&p1, &low1,
                                                   model, model_size,
                                                   0, b_off,
                                                   group_dim, rank,
                                                   n_groups, group_cnt, group_cnt,
                                                   out_dim, &heads) &&
             ds4_gpu_set_current_device(0) == 0 &&
             ds4_gpu_add_xdev_tensor(&sum, &p0, &p1, &staging, (uint32_t)out_dim) &&
             ds4_gpu_tensor_read(&full, 0, host_full, out_dim * sizeof(float)) &&
             ds4_gpu_tensor_read(&sum, 0, host_sum, out_dim * sizeof(float));
    CHECK(ok, "attention_peer compute");
    for (uint64_t i = 0; i < out_dim; i++) {
        float diff = fabsf(host_full[i] - host_sum[i]);
        if (diff > 1.0e-3f) {
            fprintf(stderr,
                    "FAIL: attention_peer mismatch row=%llu full=%f sum=%f diff=%g\n",
                    (unsigned long long)i, host_full[i], host_sum[i], diff);
            return 1;
        }
    }

    ds4_gpu_tensor_free_in_place(&heads);
    ds4_gpu_tensor_free_in_place(&low);
    ds4_gpu_tensor_free_in_place(&low0);
    ds4_gpu_tensor_free_in_place(&low1);
    ds4_gpu_tensor_free_in_place(&full);
    ds4_gpu_tensor_free_in_place(&p0);
    ds4_gpu_tensor_free_in_place(&p1);
    ds4_gpu_tensor_free_in_place(&staging);
    ds4_gpu_tensor_free_in_place(&sum);
    free(model);
    free(host_heads);
    free(host_full);
    free(host_sum);
    ds4_gpu_cleanup();
    fprintf(stderr, "  attention_output_tp_peer_read OK\n");
    return 0;
}

int main(void) {
    int dev_count = 0;
    (void)cudaGetDeviceCount(&dev_count);
    fprintf(stderr, "test_gpu_xdev: %d CUDA devices visible\n", dev_count);

    /* N=1 same-device path. */
    if (run_one(1, 0)) return 1;
    if (run_copy3(1, 0)) return 1;
    if (run_top1()) return 1;
    if (run_topk2048()) return 1;
    if (run_glm_selected_attention()) return 1;
    if (run_glm_indexer_scores()) return 1;
    if (run_glm_decode_attention_staged()) return 1;
    if (run_moe_handoff_pack()) return 1;
    if (run_q8_kslice()) return 1;
    if (run_q8_matmul_top1_fused()) return 1;
    if (run_f16_small_out()) return 1;
    if (run_f16_small_batch()) return 1;
    if (run_f16_pair_batch()) return 1;
    if (run_q8_rowsplit()) return 1;
    if (run_q8_pair_batch()) return 1;
    if (run_attention_output_tp()) return 1;
    /* If 2+ GPUs, exercise peer-auto and forced-bounce paths. */
    if (dev_count >= 2) {
        if (run_one(2, 0)) return 1;
        if (run_copy3(2, 0)) return 1;
        if (run_attention_output_tp_peer_read()) return 1;
        if (run_one(2, 1)) return 1;
        if (run_copy3(2, 1)) return 1;
        /* Stress: catches the cudaMemcpyPeer driver-corruption pattern that
         * single-size single-iter tests can miss. Runs both peer-auto and
         * forced-bounce so we get coverage of both code paths under load. */
        if (run_stress(0)) return 1;
        if (run_stress(1)) return 1;
        if (run_add_xdev(0)) return 1;
        if (run_add_xdev(1)) return 1;
    } else {
        fprintf(stderr, "  skipping multi-GPU paths (need >= 2 devices)\n");
    }
    fprintf(stderr, "test_gpu_xdev PASS\n");
    return 0;
}
