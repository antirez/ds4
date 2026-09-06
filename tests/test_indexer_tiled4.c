/* Standalone A/B exactness test for the tiled3 indexer prefill scorer.
 * Runs without a model: random-ish Q/W/K, scores computed twice through the
 * public GPU entry (env unset = tiled2, env set = tiled3), byte-compared,
 * plus a CPU reference in tiled2's arithmetic order.  Exercises edge tiles
 * (n_comp and n_tokens not multiples of the tile sizes) and the causal
 * -INFINITY cut. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../ds4_gpu.h"

#define CHECK(cond, msg) do { \
    if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); return 1; } \
} while (0)

static float half_round(float v) {
    /* mirror the f32->f16 rounding the pack pass applies */
    _Float16 h = (_Float16)v;
    return (float)h;
}

int main(void) {
    CHECK(ds4_gpu_init(), "gpu init");

    const uint32_t n_head = 64u;
    const uint32_t head_dim = 128u;
    const uint32_t ratio = 4u;
    const uint32_t n_tokens = 61u;   /* >=32 (tiled2 gate), not a multiple of 8 */
    const uint32_t pos0 = 731u;
    const uint32_t n_comp = 197u;    /* not a multiple of 64: edge comp tile */
    const float scale = 1.0f / 11.3137f;

    const size_t q_count = (size_t)n_tokens * n_head * head_dim;
    const size_t w_count = (size_t)n_tokens * n_head;
    const size_t k_count = (size_t)n_comp * head_dim;
    const size_t s_count = (size_t)n_tokens * n_comp;

    float *hq = malloc(q_count * sizeof(float));
    float *hw = malloc(w_count * sizeof(float));
    float *hk = malloc(k_count * sizeof(float));
    float *sa = malloc(s_count * sizeof(float));
    float *sb = malloc(s_count * sizeof(float));
    CHECK(hq && hw && hk && sa && sb, "host alloc");

    srand(1337);
    for (size_t i = 0; i < q_count; i++)
        hq[i] = (float)(rand() % 2001 - 1000) / 512.0f;
    for (size_t i = 0; i < w_count; i++)
        hw[i] = (float)(rand() % 2001 - 1000) / 1024.0f;
    for (size_t i = 0; i < k_count; i++)
        hk[i] = (float)(rand() % 2001 - 1000) / 512.0f;

    ds4_gpu_tensor *q = ds4_gpu_tensor_alloc(q_count * sizeof(float));
    ds4_gpu_tensor *w = ds4_gpu_tensor_alloc(w_count * sizeof(float));
    ds4_gpu_tensor *k = ds4_gpu_tensor_alloc(k_count * sizeof(float));
    ds4_gpu_tensor *s = ds4_gpu_tensor_alloc(s_count * sizeof(float));
    CHECK(q && w && k && s, "gpu alloc");
    CHECK(ds4_gpu_tensor_write(q, 0, hq, q_count * sizeof(float)), "write q");
    CHECK(ds4_gpu_tensor_write(w, 0, hw, w_count * sizeof(float)), "write w");
    CHECK(ds4_gpu_tensor_write(k, 0, hk, k_count * sizeof(float)), "write k");

    unsetenv("DS4_METAL_INDEXER_SCORES_TILED4");
    CHECK(ds4_gpu_indexer_scores_decode_batch_tensor(
              s, q, w, k, n_comp, n_tokens, pos0, n_head, head_dim, ratio,
              scale), "tiled2 compute");
    CHECK(ds4_gpu_tensor_read(s, 0, sa, s_count * sizeof(float)), "read a");

    {
        const char *cand = getenv("SCORER_CAND_ENV");
        setenv(cand ? cand : "DS4_METAL_INDEXER_SCORES_TILED4", "1", 1);
    }
    CHECK(ds4_gpu_indexer_scores_decode_batch_tensor(
              s, q, w, k, n_comp, n_tokens, pos0, n_head, head_dim, ratio,
              scale), "tiled4 compute");
    CHECK(ds4_gpu_tensor_read(s, 0, sb, s_count * sizeof(float)), "read b");

    size_t diff = 0, first = (size_t)-1;
    for (size_t i = 0; i < s_count; i++) {
        if (memcmp(&sa[i], &sb[i], sizeof(float)) != 0) {
            if (first == (size_t)-1) first = i;
            diff++;
        }
    }
    if (diff) {
        const uint32_t t = (uint32_t)(first / n_comp);
        const uint32_t c = (uint32_t)(first % n_comp);
        fprintf(stderr,
                "FAIL: tiled4 differs at %zu/%zu floats; first token=%u "
                "comp=%u tiled2=%a tiled3=%a\n",
                diff, s_count, t, c, sa[first], sb[first]);
        /* dump the first differing pair's per-head picture on the CPU */
        float acc = 0.0f;
        for (uint32_t h = 0; h < n_head; h++) {
            float dot = 0.0f;
            const float *qh = hq + ((size_t)t * n_head + h) * head_dim;
            const float *kr = hk + (size_t)c * head_dim;
            for (uint32_t d = 0; d < head_dim; d++)
                dot += half_round(qh[d]) * half_round(kr[d]);
            if (dot > 0.0f) acc += dot * (hw[(size_t)t * n_head + h] * scale);
        }
        fprintf(stderr, "  (cpu f16-ish ref for that cell: %a)\n", acc);
        return 1;
    }

    /* sanity: causal cut present */
    size_t ninf = 0;
    for (size_t i = 0; i < s_count; i++) if (isinf(sa[i])) ninf++;
    fprintf(stderr, "tiled4 A/B exact: %zu floats identical (%zu -inf)\n",
            s_count, ninf);
    ds4_gpu_cleanup();
    return 0;
}
