/* rope_multi_bench.c — correctness + cost check for the pos-vector RoPE
 * variants used by batched decode.
 *
 * Correctness: for every decode-layer RoPE call site, one
 * ds4_gpu_*_multi_tensor() launch over B token rows must match, bitwise, B
 * single-token launches on per-row views (the arithmetic per element is the
 * same, so exact equality is required, not a tolerance).  Both the plain and
 * the YaRN (ext_factor != 0) paths are checked.
 *
 * Cost: times a full simulated decode step (43 layers x 4 RoPE sites) three
 * ways — today's consecutive-pos batch kernel (lower bound), the pos-vector
 * kernel, and a per-sequence loop of single-token launches (what batched
 * decode would pay without the new kernels).
 *
 * Usage: tests/rope_multi_bench
 */
#include "ds4_gpu.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define N_SEQS 4u
#define N_LAYER 43u
#define WARMUP 20u
#define ITERS 400u

/* DeepSeek V4 Flash decode-layer RoPE call sites (per token). */
typedef struct {
    const char *name;
    uint32_t n_head;
    uint32_t head_dim;
    uint32_t n_rot;
    int inverse;
    int fused_rms; /* q path uses head_rms_norm_rope_tail */
} rope_site;

static const rope_site SITES[] = {
    {"q_fused", 64u, 512u, 64u, 0, 1},
    {"kv", 1u, 512u, 64u, 0, 0},
    {"heads_inv", 64u, 512u, 64u, 1, 0},
    {"indexer_q", 64u, 128u, 64u, 0, 0},
};
#define N_SITES (sizeof(SITES) / sizeof(SITES[0]))

static const uint32_t POS[N_SEQS] = {1000u, 4213u, 87u, 15991u};

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static uint32_t lcg_state = 0x1234567u;
static float frand(void) {
    lcg_state = lcg_state * 1664525u + 1013904223u;
    return ((float)(lcg_state >> 8) / (float)(1u << 24)) - 0.5f;
}

typedef enum { LAUNCH_BATCH, LAUNCH_MULTI, LAUNCH_LOOP } launch_mode;

static int launch_site(const rope_site *s, launch_mode mode, float ext_factor,
                       ds4_gpu_tensor *buf, ds4_gpu_tensor *views[N_SEQS]) {
    const uint32_t n_ctx_orig = ext_factor != 0.0f ? 4096u : 0u;
    const float freq_scale = ext_factor != 0.0f ? 0.25f : 1.0f;
    switch (mode) {
    case LAUNCH_BATCH:
        if (s->fused_rms) {
            return ds4_gpu_head_rms_norm_rope_tail_tensor(buf, N_SEQS, s->n_head, s->head_dim,
                                                          s->n_rot, POS[0], n_ctx_orig, s->inverse != 0,
                                                          10000.0f, freq_scale, ext_factor, 1.0f,
                                                          32.0f, 1.0f, 1e-6f);
        }
        return ds4_gpu_rope_tail_tensor(buf, N_SEQS, s->n_head, s->head_dim,
                                        s->n_rot, POS[0], n_ctx_orig, s->inverse != 0,
                                        10000.0f, freq_scale, ext_factor, 1.0f, 32.0f, 1.0f);
    case LAUNCH_MULTI:
        if (s->fused_rms) {
            return ds4_gpu_head_rms_norm_rope_tail_multi_tensor(buf, N_SEQS, s->n_head, s->head_dim,
                                                                s->n_rot, POS, n_ctx_orig, s->inverse != 0,
                                                                10000.0f, freq_scale, ext_factor, 1.0f,
                                                                32.0f, 1.0f, 1e-6f);
        }
        return ds4_gpu_rope_tail_multi_tensor(buf, N_SEQS, s->n_head, s->head_dim,
                                              s->n_rot, POS, n_ctx_orig, s->inverse != 0,
                                              10000.0f, freq_scale, ext_factor, 1.0f, 32.0f, 1.0f);
    case LAUNCH_LOOP:
        for (uint32_t b = 0; b < N_SEQS; b++) {
            int ok;
            if (s->fused_rms) {
                ok = ds4_gpu_head_rms_norm_rope_tail_tensor(views[b], 1u, s->n_head, s->head_dim,
                                                            s->n_rot, POS[b], n_ctx_orig, s->inverse != 0,
                                                            10000.0f, freq_scale, ext_factor, 1.0f,
                                                            32.0f, 1.0f, 1e-6f);
            } else {
                ok = ds4_gpu_rope_tail_tensor(views[b], 1u, s->n_head, s->head_dim,
                                              s->n_rot, POS[b], n_ctx_orig, s->inverse != 0,
                                              10000.0f, freq_scale, ext_factor, 1.0f, 32.0f, 1.0f);
            }
            if (!ok) return 0;
        }
        return 1;
    }
    return 0;
}

static int check_site(const rope_site *s, float ext_factor,
                      ds4_gpu_tensor *buf, ds4_gpu_tensor *views[N_SEQS]) {
    const uint64_t n = (uint64_t)N_SEQS * s->n_head * s->head_dim;
    float *host = malloc(n * sizeof(float));
    float *ref = malloc(n * sizeof(float));
    float *got = malloc(n * sizeof(float));
    if (!host || !ref || !got) return 0;
    for (uint64_t i = 0; i < n; i++) host[i] = frand();

    int ok = ds4_gpu_tensor_write(buf, 0, host, n * sizeof(float)) &&
             launch_site(s, LAUNCH_LOOP, ext_factor, buf, views) &&
             ds4_gpu_synchronize() &&
             ds4_gpu_tensor_read(buf, 0, ref, n * sizeof(float));
    if (ok) {
        ok = ds4_gpu_tensor_write(buf, 0, host, n * sizeof(float)) &&
             launch_site(s, LAUNCH_MULTI, ext_factor, buf, views) &&
             ds4_gpu_synchronize() &&
             ds4_gpu_tensor_read(buf, 0, got, n * sizeof(float));
    }
    if (!ok) {
        fprintf(stderr, "rope_multi_bench: launch failed (%s)\n", s->name);
        return 0;
    }
    uint64_t bad = 0;
    for (uint64_t i = 0; i < n; i++) {
        if (memcmp(&ref[i], &got[i], sizeof(float)) != 0) bad++;
    }
    printf("%-10s ext=%g: %llu/%llu values differ -> %s\n",
           s->name, (double)ext_factor,
           (unsigned long long)bad, (unsigned long long)n,
           bad == 0 ? "MATCH" : "MISMATCH");
    free(host);
    free(ref);
    free(got);
    return bad == 0;
}

/* One simulated decode step: every layer runs every RoPE site. */
static int step(launch_mode mode, ds4_gpu_tensor *bufs[N_SITES],
                ds4_gpu_tensor *views[N_SITES][N_SEQS]) {
    for (uint32_t il = 0; il < N_LAYER; il++) {
        for (uint32_t si = 0; si < N_SITES; si++) {
            if (!launch_site(&SITES[si], mode, 0.0f, bufs[si], views[si])) return 0;
        }
    }
    return ds4_gpu_synchronize();
}

static int run_variant(launch_mode mode, const char *label,
                       ds4_gpu_tensor *bufs[N_SITES],
                       ds4_gpu_tensor *views[N_SITES][N_SEQS],
                       double *out_ms_per_step) {
    for (uint32_t i = 0; i < WARMUP; i++) {
        if (!step(mode, bufs, views)) return 0;
    }
    const double t0 = now_sec();
    for (uint32_t i = 0; i < ITERS; i++) {
        if (!step(mode, bufs, views)) return 0;
    }
    const double ms = (now_sec() - t0) * 1000.0 / (double)ITERS;
    printf("%-30s %8.3f ms/step (%u layers x %u sites)\n",
           label, ms, N_LAYER, (unsigned)N_SITES);
    *out_ms_per_step = ms;
    return 1;
}

int main(void) {
    ds4_gpu_tensor *bufs[N_SITES];
    ds4_gpu_tensor *views[N_SITES][N_SEQS];
    for (uint32_t si = 0; si < N_SITES; si++) {
        const uint64_t row = (uint64_t)SITES[si].n_head * SITES[si].head_dim * sizeof(float);
        bufs[si] = ds4_gpu_tensor_alloc((uint64_t)N_SEQS * row);
        if (!bufs[si]) {
            fprintf(stderr, "rope_multi_bench: alloc failed\n");
            return 1;
        }
        for (uint32_t b = 0; b < N_SEQS; b++) {
            views[si][b] = ds4_gpu_tensor_view(bufs[si], (uint64_t)b * row, row);
            if (!views[si][b]) {
                fprintf(stderr, "rope_multi_bench: view failed\n");
                return 1;
            }
        }
    }

    int rc = 0;
    for (uint32_t si = 0; si < N_SITES; si++) {
        if (!check_site(&SITES[si], 0.0f, bufs[si], views[si])) rc = 1;
        if (!check_site(&SITES[si], 1.0f, bufs[si], views[si])) rc = 1;
    }
    if (rc != 0) {
        printf("FAILED\n");
        return rc;
    }

    double ms_batch = 0.0, ms_multi = 0.0, ms_loop = 0.0;
    if (!run_variant(LAUNCH_BATCH, "consecutive batch (baseline)", bufs, views, &ms_batch) ||
        !run_variant(LAUNCH_MULTI, "pos-vector multi", bufs, views, &ms_multi) ||
        !run_variant(LAUNCH_LOOP, "per-seq loop (B=4 views)", bufs, views, &ms_loop)) {
        fprintf(stderr, "rope_multi_bench: bench launch failed\n");
        return 1;
    }
    printf("\nper-seq loop overhead vs pos-vector: %.3f ms/step at B=%u\n",
           ms_loop - ms_multi, N_SEQS);
    printf("OK\n");
    return 0;
}
