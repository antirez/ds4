/* Radix-select top-k against the merge sort it replaces, on random rows and
 * on rows full of ties. Run: make tests/test_metal_topk_select && ./tests/test_metal_topk_select */
#import <Foundation/Foundation.h>
#include "ds4_gpu.h"
#include "ds4_deepseek41_gpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>

bool ds4_log_is_tty(FILE *fp) { (void)fp; return false; }

static double now(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec + ts.tv_nsec * 1e-9; }

typedef struct { float v; int i; } item;
static int cmp_desc(const void *a, const void *b) {
    const item *x = a, *y = b;
    if (x->v != y->v) return x->v > y->v ? -1 : 1;
    return x->i < y->i ? -1 : x->i > y->i;
}

static int run_case(uint32_t n, uint32_t n_tokens, uint32_t top_k, uint32_t causal_start,
                    uint32_t causal_ratio, int ties, int loops) {
    float *scores = malloc((size_t)n * n_tokens * sizeof(float));
    for (uint32_t t = 0; t < n_tokens; t++)
        for (uint32_t i = 0; i < n; i++) {
            float v = (float)(rand() % 100000) / 1000.0f - 20.0f;
            if (ties) v = (float)(rand() % 7);
            const uint32_t width = causal_ratio ? (causal_start + t + 1u) / causal_ratio : n;
            scores[(size_t)t * n + i] = i < width ? v : -__builtin_inff();
        }
    ds4_gpu_tensor *sc = ds4_gpu_tensor_alloc((uint64_t)n * n_tokens * sizeof(float));
    ds4_gpu_tensor *sel = ds4_gpu_tensor_alloc((uint64_t)top_k * n_tokens * sizeof(int32_t));
    ds4_gpu_tensor_write(sc, 0, scores, (uint64_t)n * n_tokens * sizeof(float));
    int ok = 1;
    double best = 1e9;
    for (int l = 0; l < loops; l++) {
        const double t0 = now();
        int rc = causal_ratio ? ds4_gpu_dsv41_indexer_topk_batch(sel, sc, n, n_tokens, causal_start, causal_ratio)
                              : ds4_gpu_indexer_topk_tensor(sel, sc, n, n_tokens, top_k);
        ds4_gpu_synchronize();
        const double dt = now() - t0;
        if (dt < best) best = dt;
        if (!rc) { printf("  call failed\n"); ok = 0; break; }
    }
    int32_t *out = malloc((size_t)top_k * n_tokens * sizeof(int32_t));
    ds4_gpu_tensor_read(sel, 0, out, (uint64_t)top_k * n_tokens * sizeof(int32_t));
    item *items = malloc((size_t)n * sizeof(item));
    int order_mismatch = 0, set_mismatch = 0;
    for (uint32_t t = 0; ok && t < n_tokens; t++) {
        const uint32_t width = causal_ratio ? (causal_start + t + 1u) / causal_ratio : n;
        const uint32_t w = width < n ? width : n;
        for (uint32_t i = 0; i < w; i++) { items[i].v = scores[(size_t)t * n + i]; items[i].i = (int)i; }
        qsort(items, w, sizeof(item), cmp_desc);
        const uint32_t k = top_k < w ? top_k : w;
        /* Set equality on scores (ties may pick different indices only when
         * they straddle the boundary), exact index equality otherwise. */
        for (uint32_t j = 0; j < k; j++) {
            const int32_t got = out[(size_t)t * top_k + j];
            if (got < 0 || (uint32_t)got >= w) { set_mismatch++; continue; }
            const float gv = scores[(size_t)t * n + got];
            if (gv != items[j].v) set_mismatch++;
            else if (got != items[j].i) order_mismatch++;
        }
    }
    printf("n=%u tokens=%u k=%u causal=%u/%u ties=%d: %s (set mismatches %d, tie-order diffs %d) %.3f ms\n",
           n, n_tokens, top_k, causal_start, causal_ratio, ties,
           ok && !set_mismatch ? "OK" : "FAIL", set_mismatch, order_mismatch, best * 1000.0);
    free(items); free(out); free(scores);
    ds4_gpu_tensor_free(sc); ds4_gpu_tensor_free(sel);
    return ok && !set_mismatch;
}

int main(void) {
    if (!ds4_gpu_init()) { printf("gpu init failed\n"); return 1; }
    srand(7);
    int ok = 1;
    ok &= run_case(4096, 1, 512, 0, 0, 0, 3);
    ok &= run_case(65537, 1, 512, 0, 0, 0, 3);
    ok &= run_case(435000, 1, 512, 0, 0, 0, 5);
    ok &= run_case(435000, 3, 512, 0, 0, 0, 5);
    ok &= run_case(869000, 1, 512, 0, 0, 0, 5);
    ok &= run_case(54000, 1, 2048, 0, 0, 0, 5);
    ok &= run_case(129280, 1, 1, 0, 0, 0, 3);
    ok &= run_case(435000, 1, 512, 0, 0, 1, 3);
    ok &= run_case(435000, 32, 512, 869000, 2, 0, 3);
    ok &= run_case(20000, 16, 512, 40000 - 20, 2, 0, 3);
    printf(ok ? "topk select PASS\n" : "topk select FAIL\n");
    return ok ? 0 : 1;
}
