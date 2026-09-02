/*
 * Metal-only top-k argsort test (ds4_gpu_indexer_topk_tensor).
 *
 * Exercises the Metal merge kernel's pruning path: intermediate rounds keep
 * top_k instead of work_width, including odd run counts (leftover pass-through)
 * and cases where top_k is small relative to n_comp (prune from the first
 * merge). The existing run_topk2048 in test_gpu_xdev.c is CUDA-only and cannot
 * cover this. Each case's comment names the path it actually exercises; the
 * per-case print (see print_path) mirrors the host shape math so the path is
 * confirmed at run time rather than inferred.
 */

#include "ds4_gpu.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

bool ds4_log_is_tty(FILE *fp) {
    (void)fp;
    return false;
}

typedef struct {
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

/* Mirror ds4_gpu_indexer_topk_tensor's shape math and merge-round loop
 * (ds4_metal.m) to print which path each case exercises. The argsort kernel is
 * a plain Metal kernel, so maxTotalThreadsPerThreadgroup is the 1024 default. */
static void print_path(uint32_t n_comp, uint32_t top_k) {
    const uint32_t max_threads = 1024u;
    uint32_t nth = 1u;
    while (nth < n_comp && 2u * nth <= max_threads) nth *= 2u;
    const uint32_t npr = (n_comp + nth - 1u) / nth;
    const uint32_t block_top_k = top_k < nth ? top_k : nth;
    uint32_t work_width = top_k;
    if (npr > 1) {
        const uint32_t last_block = n_comp - (npr - 1u) * nth;
        work_width = (npr - 1u) * block_top_k +
                     (last_block < block_top_k ? last_block : block_top_k);
    }
    uint32_t rounds = 0, len = block_top_k, total = work_width, nruns = npr;
    while (nruns > 1) {
        const uint32_t nm = (nruns + 1u) / 2u;
        const uint32_t new_len = (2u * len < top_k) ? 2u * len : top_k;
        /* full pairs write new_len; last partial pair min(remainder, new_len);
         * q clamped to nm keeps the accounting self-limiting like the host. */
        const uint32_t q = total / (2u * len);
        const uint32_t r = total % (2u * len);
        total = (q < nm ? q : nm) * new_len + (q < nm ? (r < new_len ? r : new_len) : 0u);
        rounds++;
        len = new_len;
        nruns = nm;
    }
    if (npr <= 1) {
        fprintf(stderr, "  -> nth=%u npr=%u: one-pass (merge kernel never runs)\n",
                nth, npr);
    } else {
        fprintf(stderr, "  -> nth=%u npr=%u work_width=%u: %u merge round(s)\n",
                nth, npr, work_width, rounds);
    }
}

int main(void) {
    /* {n_comp, top_k, n_tokens} */
    const uint32_t cases[][3] = {
        { 3355u, 2048u, 2u },   /* odd npr=4, pruning from an intermediate merge */
        { 5003u, 2048u, 2u },   /* odd npr=5, larger odd count */
        { 300u,  8u,    2u },   /* one-pass (n_comp < nth => npr=1), multi-token */
        { 4096u, 2048u, 2u },   /* power-of-two n_comp, even rounds */
        { 8192u, 8u,    1u },   /* npr=8 maximal-pruning tree (total 64->8), 3 rounds */
        { 8192u, 300u,  1u },   /* non-power-of-two top_k: len stays 300, no transition */
        { 8192u, 1u,    1u },   /* degenerate top_k=1 max-reduction (new_len==1) */
    };
    const uint32_t n_cases = (uint32_t)(sizeof(cases) / sizeof(cases[0]));
    uint32_t ci;

    if (!ds4_gpu_init()) {
        fprintf(stderr, "argsort Metal: ds4_gpu_init failed\n");
        return 1;
    }

    for (ci = 0; ci < n_cases; ci++) {
        const uint32_t n_comp = cases[ci][0];
        const uint32_t top_k = cases[ci][1];
        const uint32_t n_tokens = cases[ci][2];
        print_path(n_comp, top_k);
        const uint64_t n_scores = (uint64_t)n_tokens * n_comp;
        float *host_scores = (float *)malloc((size_t)n_scores * sizeof(float));
        uint32_t *host_selected =
            (uint32_t *)malloc((size_t)n_tokens * top_k * sizeof(uint32_t));
        topk_ref_entry *ref =
            (topk_ref_entry *)malloc((size_t)n_comp * sizeof(topk_ref_entry));
        if (!host_scores || !host_selected || !ref) {
            fprintf(stderr, "argsort Metal: host alloc failed\n");
            return 1;
        }
        /* Deterministic permutation of 0..n_comp-1 per token, so scores are
         * unique (no ties) and float-exact. This validates the pruning/merge
         * logic without depending on tie-break order, which the Metal merge
         * resolves differently from the reference (a pre-existing discrepancy
         * unrelated to pruning). */
        uint32_t *perm = (uint32_t *)malloc((size_t)n_comp * sizeof(uint32_t));
        if (!perm) {
            fprintf(stderr, "argsort Metal: perm alloc failed\n");
            return 1;
        }
        for (uint32_t t = 0; t < n_tokens; t++) {
            for (uint32_t i = 0; i < n_comp; i++) perm[i] = i;
            uint32_t seed = 0x1234567u + 0x9e3779b9u * t;
            for (uint32_t i = n_comp; i > 1; i--) {
                seed = seed * 1664525u + 1013904223u;
                uint32_t j = seed % i;
                uint32_t tmp = perm[i - 1];
                perm[i - 1] = perm[j];
                perm[j] = tmp;
            }
            for (uint32_t i = 0; i < n_comp; i++) {
                host_scores[(uint64_t)t * n_comp + i] = (float)perm[i];
            }
        }
        free(perm);

        ds4_gpu_tensor *scores = ds4_gpu_tensor_alloc(n_scores * sizeof(float));
        ds4_gpu_tensor *selected =
            ds4_gpu_tensor_alloc((uint64_t)n_tokens * top_k * sizeof(uint32_t));
        if (scores == NULL || selected == NULL) {
            fprintf(stderr, "argsort Metal: tensor alloc failed\n");
            return 1;
        }

        int ok = ds4_gpu_tensor_write(
                     scores, 0, host_scores, n_scores * sizeof(float)) &&
                 ds4_gpu_indexer_topk_tensor(
                     selected, scores, n_comp, n_tokens, top_k) &&
                 ds4_gpu_tensor_read(
                     selected, 0, host_selected,
                     (uint64_t)n_tokens * top_k * sizeof(uint32_t));
        if (!ok) {
            fprintf(stderr,
                    "argsort Metal: FAIL compute n=%u top_k=%u\n", n_comp, top_k);
            return 1;
        }

        for (uint32_t t = 0; t < n_tokens; t++) {
            for (uint32_t i = 0; i < n_comp; i++) {
                ref[i].score = host_scores[(uint64_t)t * n_comp + i];
                ref[i].index = i;
            }
            qsort(ref, n_comp, sizeof(ref[0]), topk_ref_cmp);
            for (uint32_t i = 0; i < top_k; i++) {
                if (host_selected[(uint64_t)t * top_k + i] != ref[i].index) {
                    fprintf(stderr,
                            "argsort Metal: FAIL n=%u token=%u rank=%u "
                            "got=%u want=%u\n",
                            n_comp, t, i,
                            host_selected[(uint64_t)t * top_k + i],
                            ref[i].index);
                    return 1;
                }
            }
        }

        ds4_gpu_tensor_free(scores);
        ds4_gpu_tensor_free(selected);
        free(host_scores);
        free(host_selected);
        free(ref);
        fprintf(stderr, "argsort Metal: n=%u top_k=%u OK\n", n_comp, top_k);
    }

    ds4_gpu_cleanup();
    fprintf(stderr, "argsort Metal: all cases PASS\n");
    return 0;
}
