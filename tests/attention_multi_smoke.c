/* attention_multi_smoke.c — equivalence check for the multi-sequence decode
 * attention kernel: two sequences with different raw/compressed caches must
 * produce, in one ds4_gpu_attention_decode_heads_multi_tensor() call, exactly
 * the heads that two independent ds4_gpu_attention_decode_heads_tensor()
 * calls produce.
 *
 * Usage: tests/attention_multi_smoke
 */
#include "ds4_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N_HEAD 8u
#define N_SEQS 2u

static uint32_t lcg_state = 0x12345678u;
static float frand(void) {
    lcg_state = lcg_state * 1664525u + 1013904223u;
    return ((float)(lcg_state >> 8) / (float)(1u << 24)) - 0.5f;
}

static void fill(float *p, uint64_t n) {
    for (uint64_t i = 0; i < n; i++) p[i] = frand();
}

static ds4_gpu_tensor *upload(const float *host, uint64_t n) {
    ds4_gpu_tensor *t = ds4_gpu_tensor_alloc(n * sizeof(float));
    if (!t || !ds4_gpu_tensor_write(t, 0, host, n * sizeof(float))) {
        fprintf(stderr, "attention_multi_smoke: tensor upload failed\n");
        exit(1);
    }
    return t;
}

static int run_case(uint32_t head_dim, const void *map, uint64_t map_size) {
    /* Two sequences with deliberately different shapes and ring offsets. */
    const uint32_t n_raw[N_SEQS] = {96u, 41u};
    const uint32_t raw_cap[N_SEQS] = {128u, 64u};
    const uint32_t raw_start[N_SEQS] = {37u, 5u};
    const uint32_t n_comp[N_SEQS] = {200u, 0u};

    float *q_host = malloc((size_t)N_SEQS * N_HEAD * head_dim * sizeof(float));
    fill(q_host, (uint64_t)N_SEQS * N_HEAD * head_dim);
    ds4_gpu_tensor *q = upload(q_host, (uint64_t)N_SEQS * N_HEAD * head_dim);

    ds4_gpu_tensor *raw[N_SEQS];
    ds4_gpu_tensor *comp[N_SEQS] = {NULL, NULL};
    ds4_gpu_tensor *mask[N_SEQS] = {NULL, NULL};
    for (uint32_t b = 0; b < N_SEQS; b++) {
        float *raw_host = malloc((size_t)raw_cap[b] * head_dim * sizeof(float));
        fill(raw_host, (uint64_t)raw_cap[b] * head_dim);
        raw[b] = upload(raw_host, (uint64_t)raw_cap[b] * head_dim);
        free(raw_host);
        if (n_comp[b]) {
            float *comp_host = malloc((size_t)n_comp[b] * head_dim * sizeof(float));
            fill(comp_host, (uint64_t)n_comp[b] * head_dim);
            comp[b] = upload(comp_host, (uint64_t)n_comp[b] * head_dim);
            free(comp_host);
            float *mask_host = malloc((size_t)n_comp[b] * sizeof(float));
            for (uint32_t i = 0; i < n_comp[b]; i++) {
                mask_host[i] = (i % 7u == 3u) ? -1.0e30f : frand() * 0.1f;
            }
            mask[b] = upload(mask_host, n_comp[b]);
            free(mask_host);
        }
    }

    const uint64_t head_row = (uint64_t)N_HEAD * head_dim;
    ds4_gpu_tensor *heads_ref = ds4_gpu_tensor_alloc((uint64_t)N_SEQS * head_row * sizeof(float));
    ds4_gpu_tensor *heads_multi = ds4_gpu_tensor_alloc((uint64_t)N_SEQS * head_row * sizeof(float));
    if (!heads_ref || !heads_multi) {
        fprintf(stderr, "attention_multi_smoke: heads alloc failed\n");
        return 1;
    }

    /* Reference: one single-sequence call per slot through row views. */
    for (uint32_t b = 0; b < N_SEQS; b++) {
        ds4_gpu_tensor *hv = ds4_gpu_tensor_view(heads_ref, (uint64_t)b * head_row * sizeof(float),
                                                 head_row * sizeof(float));
        ds4_gpu_tensor *qv = ds4_gpu_tensor_view(q, (uint64_t)b * head_row * sizeof(float),
                                                 head_row * sizeof(float));
        if (!hv || !qv ||
            !ds4_gpu_attention_decode_heads_tensor(hv, map, map_size, 0, qv,
                                                   raw[b], n_raw[b], raw_cap[b], raw_start[b],
                                                   comp[b], 0, n_comp[b],
                                                   mask[b], mask[b] != NULL,
                                                   N_HEAD, head_dim)) {
            fprintf(stderr, "attention_multi_smoke: single call failed (seq %u, head_dim %u)\n",
                    b, head_dim);
            return 1;
        }
    }

    /* Multi: one call for both sequences. */
    ds4_gpu_attn_seqview seqs[N_SEQS];
    memset(seqs, 0, sizeof(seqs));
    for (uint32_t b = 0; b < N_SEQS; b++) {
        seqs[b].raw_kv = raw[b];
        seqs[b].comp_kv = comp[b];
        seqs[b].comp_mask = mask[b];
        seqs[b].n_raw = n_raw[b];
        seqs[b].raw_cap = raw_cap[b];
        seqs[b].raw_start = raw_start[b];
        seqs[b].n_comp = n_comp[b];
    }
    if (!ds4_gpu_attention_decode_heads_multi_tensor(heads_multi, map, map_size, 0,
                                                     q, seqs, N_SEQS, 0,
                                                     N_HEAD, head_dim)) {
        fprintf(stderr, "attention_multi_smoke: multi call failed (head_dim %u)\n", head_dim);
        return 1;
    }
    if (!ds4_gpu_synchronize()) {
        fprintf(stderr, "attention_multi_smoke: synchronize failed\n");
        return 1;
    }

    const uint64_t n_out = (uint64_t)N_SEQS * head_row;
    float *ref_host = malloc(n_out * sizeof(float));
    float *multi_host = malloc(n_out * sizeof(float));
    if (!ds4_gpu_tensor_read(heads_ref, 0, ref_host, n_out * sizeof(float)) ||
        !ds4_gpu_tensor_read(heads_multi, 0, multi_host, n_out * sizeof(float))) {
        fprintf(stderr, "attention_multi_smoke: readback failed\n");
        return 1;
    }
    uint64_t mismatches = 0;
    float max_abs = 0.0f;
    for (uint64_t i = 0; i < n_out; i++) {
        const float d = fabsf(ref_host[i] - multi_host[i]);
        if (d > max_abs) max_abs = d;
        if (d > 1e-6f) mismatches++;
    }
    printf("head_dim=%u: %llu/%llu values differ (max abs diff %g) -> %s\n",
           head_dim,
           (unsigned long long)mismatches,
           (unsigned long long)n_out,
           (double)max_abs,
           mismatches == 0 ? "MATCH" : "MISMATCH");
    free(ref_host);
    free(multi_host);
    free(q_host);
    return mismatches == 0 ? 0 : 1;
}

int main(void) {
    /* Fake "model" that only carries the attention sinks at offset 0. */
    float sinks[N_HEAD];
    for (uint32_t h = 0; h < N_HEAD; h++) sinks[h] = frand();
    if (!ds4_gpu_set_model_map(sinks, sizeof(sinks))) {
        fprintf(stderr, "attention_multi_smoke: set_model_map failed\n");
        return 1;
    }

    int rc = run_case(512u, sinks, sizeof(sinks));
    rc |= run_case(128u, sinks, sizeof(sinks));
    printf(rc == 0 ? "OK\n" : "FAILED\n");
    return rc;
}
