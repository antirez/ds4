/* Lockstep TP drafter: intermediate split of IQ2_XXS/Q2_K or native MXFP4.
 *
 * Production drafter shapes (in 4096, mid 2048, out 4096, 256 experts, top 6),
 * so the fused vector path the drafter uses is the one under test.
 *
 * Reference: routed MoE with no TP (raw experts, full intermediate).
 * Split: for rank 0 and rank 1, bind the network TP rank, build the split
 * from the model file and run the same call, which returns this rank's
 * partial output over half of every expert.
 *
 * Checks:
 *  - build size is half of the three expert tensors;
 *  - gate/up intermediate rows: rank r's mid rows equal the reference's rows
 *    [r*mid/2, +mid/2) of every assignment, bit for bit;
 *  - y0 + y1 matches the reference within float summation-order error;
 *  - with sharding suspended, or with no TP rank bound, the split is ignored
 *    and the call returns the reference bit for bit (leader-local drafter);
 *  - unsupported IQ2 widths fail instead of reading raw experts; MXFP4 also
 *    covers one row and the larger-batch dispatch;
 *  - malformed descriptors build nothing; a rebuild after them is complete;
 *  - no NaN/Inf anywhere. */
#include "ds4_gpu.h"
#include "ds4_gpu_tp.h"

#include <errno.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #x); exit(2); } } while (0)

enum { IN_DIM = 4096, MID_DIM = 2048, OUT_DIM = 4096, N_EXP = 256, TOPK = 6 };
enum { IQ2_BYTES = 66, Q2K_BYTES = 84, QK = 256 };
static bool mx;

static uint32_t g_seed = 0x5eedu;
static uint32_t rnd(void) { g_seed = g_seed * 1664525u + 1013904223u; return g_seed; }
static float frand(void) { return (float)((int32_t)(rnd() >> 8) - (1 << 23)) / (float)(1 << 23); }

static void fill_mxfp4(unsigned char *p, uint64_t blocks) {
    for (uint64_t b = 0; b < blocks; b++, p += 17) {
        p[0] = 116u + ((rnd() >> 24) % 5u);
        for (int i = 1; i < 17; i++) p[i] = (unsigned char)(rnd() >> 24);
    }
}

/* IQ2_XXS: fp16 d, then 64 bytes of grid indices, signs and scales. */
static void fill_iq2_xxs(unsigned char *p, uint64_t blocks) {
    for (uint64_t b = 0; b < blocks; b++, p += IQ2_BYTES) {
        const uint16_t d = (uint16_t)(0x1c00u + ((rnd() >> 16) & 0x3ffu));   /* 2^-8 .. 2^-7 */
        memcpy(p, &d, 2);
        for (int i = 2; i < IQ2_BYTES; i++) p[i] = (unsigned char)(rnd() >> 24);
    }
}

/* Q2_K: 16 scale bytes, 64 quant bytes, fp16 d, fp16 dmin. */
static void fill_q2_k(unsigned char *p, uint64_t blocks) {
    for (uint64_t b = 0; b < blocks; b++, p += Q2K_BYTES) {
        for (int i = 0; i < 80; i++) p[i] = (unsigned char)(rnd() >> 24);
        const uint16_t d = (uint16_t)(0x1c00u + ((rnd() >> 16) & 0x3ffu));
        const uint16_t dmin = (uint16_t)(0x1800u + ((rnd() >> 16) & 0x3ffu));
        memcpy(p + 80, &d, 2);
        memcpy(p + 82, &dmin, 2);
    }
}

static int noop_exchange(void *ud, uint32_t layer, uint32_t gate, uint64_t seq) {
    (void)ud; (void)layer; (void)gate; (void)seq;
    return 1;
}

typedef struct {
    unsigned char *map;
    uint64_t size, gate_off, up_off, down_off;
    uint64_t gate_row, gate_expert, down_row, down_expert;
    char path[64];
    int fd;
} model_t;

static int run_moe(const model_t *m, uint32_t n_tokens, const int32_t *ids, const float *w,
                   const float *x, float *y, float *mid, uint32_t mid_dim_out) {
    const uint64_t slots = (uint64_t)n_tokens * TOPK;
    ds4_gpu_tensor *t_x = ds4_gpu_tensor_alloc((uint64_t)n_tokens * IN_DIM * 4u);
    ds4_gpu_tensor *t_ids = ds4_gpu_tensor_alloc(slots * 4u);
    ds4_gpu_tensor *t_w = ds4_gpu_tensor_alloc(slots * 4u);
    ds4_gpu_tensor *t_out = ds4_gpu_tensor_alloc((uint64_t)n_tokens * OUT_DIM * 4u);
    ds4_gpu_tensor *t_gate = ds4_gpu_tensor_alloc(slots * MID_DIM * 4u);
    ds4_gpu_tensor *t_up = ds4_gpu_tensor_alloc(slots * MID_DIM * 4u);
    ds4_gpu_tensor *t_mid = ds4_gpu_tensor_alloc(slots * MID_DIM * 4u);
    ds4_gpu_tensor *t_down = ds4_gpu_tensor_alloc(slots * OUT_DIM * 4u);
    CHECK(t_x && t_ids && t_w && t_out && t_gate && t_up && t_mid && t_down);
    CHECK(ds4_gpu_tensor_write(t_x, 0, x, (uint64_t)n_tokens * IN_DIM * 4u));
    CHECK(ds4_gpu_tensor_write(t_ids, 0, ids, slots * 4u));
    CHECK(ds4_gpu_tensor_write(t_w, 0, w, slots * 4u));
    bool mid_is_f16 = false;
    int ok = ds4_gpu_routed_moe_batch_tensor(t_out, t_gate, t_up, t_mid, t_down,
                                             m->map, m->size, m->gate_off, m->up_off, m->down_off,
                                             mx ? 39u : 16u, mx ? 39u : 10u, m->gate_expert, m->gate_row,
                                             m->down_expert, m->down_row,
                                             IN_DIM, MID_DIM, OUT_DIM, t_ids, t_w,
                                             N_EXP, TOPK, 10.0f, t_x, 0u, n_tokens,
                                             &mid_is_f16, false);
    CHECK(!mid_is_f16);
    CHECK(ds4_gpu_synchronize());
    if (ok && y) {
        CHECK(ds4_gpu_tensor_read(t_out, 0, y, (uint64_t)n_tokens * OUT_DIM * 4u));
        CHECK(ds4_gpu_tensor_read(t_mid, 0, mid, slots * mid_dim_out * 4u));
    }
    ds4_gpu_tensor *all[] = {t_x, t_ids, t_w, t_out, t_gate, t_up, t_mid, t_down};
    for (size_t i = 0; i < sizeof(all) / sizeof(all[0]); i++) ds4_gpu_tensor_free(all[i]);
    return ok;
}

static uint64_t build_split(const model_t *m, uint32_t rank, const uint32_t *types,
                            const uint32_t *in_dims, const uint64_t *offsets) {
    const uint32_t is_down[3] = {0, 0, 1};
    const uint32_t out_dims[3] = {MID_DIM, MID_DIM, OUT_DIM};
    const uint32_t n_experts[3] = {N_EXP, N_EXP, N_EXP};
    return ds4_gpu_build_dspark_tp_split(m->map, m->size, m->size, m->path, rank, offsets,
                                         is_down, in_dims, out_dims, n_experts, types, 3);
}

int main(int argc, char **argv) {
    CHECK(argc == 1 || (argc == 2 && !strcmp(argv[1], "--mxfp4")));
    mx = argc == 2;
    model_t m;
    memset(&m, 0, sizeof(m));
    m.gate_row = mx ? (IN_DIM / 32u) * 17u : (IN_DIM / QK) * IQ2_BYTES;
    m.gate_expert = (uint64_t)MID_DIM * m.gate_row;
    m.down_row = mx ? (MID_DIM / 32u) * 17u : (MID_DIM / QK) * Q2K_BYTES;
    m.down_expert = (uint64_t)OUT_DIM * m.down_row;
    m.gate_off = 4096;
    m.up_off = m.gate_off + ((N_EXP * m.gate_expert + 4095u) & ~4095ull);
    m.down_off = m.up_off + ((N_EXP * m.gate_expert + 4095u) & ~4095ull);
    m.size = m.down_off + N_EXP * m.down_expert + 4096u;
    m.map = calloc(1, (size_t)m.size);
    CHECK(m.map);
    if (mx) {
        fill_mxfp4(m.map + m.gate_off, N_EXP * m.gate_expert / 17u);
        fill_mxfp4(m.map + m.up_off, N_EXP * m.gate_expert / 17u);
        fill_mxfp4(m.map + m.down_off, N_EXP * m.down_expert / 17u);
    } else {
        fill_iq2_xxs(m.map + m.gate_off, N_EXP * m.gate_expert / IQ2_BYTES);
        fill_iq2_xxs(m.map + m.up_off, N_EXP * m.gate_expert / IQ2_BYTES);
        fill_q2_k(m.map + m.down_off, N_EXP * m.down_expert / Q2K_BYTES);
    }
    /* Low LCG bits repeat at these expert strides and can hide bad IDs. */
    for (uint32_t e = 1; e < N_EXP; e++) {
        CHECK(memcmp(m.map + m.gate_off, m.map + m.gate_off + e * m.gate_expert, 1024));
        CHECK(memcmp(m.map + m.down_off, m.map + m.down_off + e * m.down_expert, 1024));
    }
    strcpy(m.path, "/tmp/ds4-dspark-tp-split-XXXXXX");
    m.fd = mkstemp(m.path);
    CHECK(m.fd >= 0);
    for (uint64_t n = 0; n < m.size;) {
        const ssize_t wrote = write(m.fd, m.map + n, (size_t)(m.size - n));
        if (wrote < 0 && errno == EINTR) continue;
        CHECK(wrote > 0);
        n += (uint64_t)wrote;
    }
    const uint64_t offsets[3] = {m.gate_off, m.up_off, m.down_off};
    const uint32_t types[3] = {mx ? 39u : 16u, mx ? 39u : 16u, mx ? 39u : 10u};
    const uint32_t in_dims[3] = {IN_DIM, IN_DIM, MID_DIM};
    const uint64_t want_bytes = N_EXP * (2u * m.gate_expert + m.down_expert) / 2u;

    const uint64_t off0 = 0;
    CHECK(ds4_gpu_init());
    if (!ds4_gpu_device_is_spark()) {
        fprintf(stderr, "CUDA DSpark TP expert split requires Spark\n");
        close(m.fd);
        unlink(m.path);
        free(m.map);
        exit(77);
    }
    CHECK(ds4_gpu_set_model_map_spans(m.map, m.size, &off0, &m.size, 1, 0));
    CHECK(ds4_gpu_set_model_fd_for_map(m.fd, m.map));

    /* Malformed descriptors build nothing. */
    {
        ds4_gpu_tensor *slab = ds4_gpu_tensor_alloc(4096);
        CHECK(slab && ds4_gpu_tp_init(0, slab, 0, 64, 64, noop_exchange, NULL));
        const uint32_t bad_type[3] = {16u, 12u, 10u};
        const uint32_t bad_in[3] = {IN_DIM, IN_DIM, MID_DIM + 256u};
        const uint64_t bad_off[3] = {m.gate_off, m.up_off, m.size};
        CHECK(build_split(&m, 0, bad_type, in_dims, offsets) == 0);
        CHECK(build_split(&m, 0, types, bad_in, offsets) == 0);
        CHECK(build_split(&m, 0, types, in_dims, bad_off) == 0);
        CHECK(build_split(&m, 2, types, in_dims, offsets) == 0);
        CHECK(ds4_gpu_build_dspark_tp_split(m.map, m.size, m.size, m.path, 0, offsets,
                                            NULL, in_dims, in_dims, in_dims, types, 3) == 0);
        /* Nothing active: the drafter keeps its raw experts' ownership path. */
        ds4_gpu_tp_shutdown();
        ds4_gpu_tensor_free(slab);
    }

    const uint32_t token_counts[] = {2, 3, 5, 6, 8, 1, 9};
    int fails = 0;
    for (size_t tc = 0; tc < sizeof(token_counts) / sizeof(token_counts[0]); tc++) {
        const uint32_t n = token_counts[tc];
        if (!mx && (n == 1 || n == 9)) continue;
        const uint64_t slots = (uint64_t)n * TOPK;
        /* Inputs also cover the 9-row call below. */
        const uint32_t n_alloc = 9u;
        float *x = malloc((size_t)n_alloc * IN_DIM * 4u), *w = malloc((size_t)n_alloc * TOPK * 4u);
        int32_t *ids = malloc((size_t)n_alloc * TOPK * 4u);
        float *y_ref = malloc((size_t)n * OUT_DIM * 4u), *mid_ref = malloc((size_t)slots * MID_DIM * 4u);
        float *y_raw = malloc((size_t)n * OUT_DIM * 4u), *mid_raw = malloc((size_t)slots * MID_DIM * 4u);
        float *y_r[2], *mid_r[2];
        CHECK(x && w && ids && y_ref && mid_ref && y_raw && mid_raw);
        for (uint64_t i = 0; i < (uint64_t)n_alloc * IN_DIM; i++) x[i] = frand();
        for (uint32_t t = 0; t < n_alloc; t++) {
            /* Distinct experts per token, some repeats across tokens. */
            const uint32_t base = rnd() % N_EXP;
            for (uint32_t k = 0; k < TOPK; k++) {
                ids[t * TOPK + k] = (int32_t)((base + k * 37u) % N_EXP);
                w[t * TOPK + k] = 0.05f + 0.3f * fabsf(frand());
            }
        }
        CHECK(run_moe(&m, n, ids, w, x, y_ref, mid_ref, MID_DIM));
        int raw_diff = 0;
        for (uint32_t r = 0; r < 2; r++) {
            y_r[r] = malloc((size_t)n * OUT_DIM * 4u);
            mid_r[r] = malloc((size_t)slots * (MID_DIM / 2) * 4u);
            CHECK(y_r[r] && mid_r[r]);
            ds4_gpu_tensor *slab = ds4_gpu_tensor_alloc(4096);
            CHECK(slab);
            CHECK(ds4_gpu_tp_init(r, slab, 0, 64, 64, noop_exchange, NULL));
            CHECK(build_split(&m, r, types, in_dims, offsets) == want_bytes);
            /* A rebuild starts clean and is complete. */
            CHECK(build_split(&m, r, types, in_dims, offsets) == want_bytes);
            CHECK(run_moe(&m, n, ids, w, x, y_r[r], mid_r[r], MID_DIM / 2));
            if (!mx) {
                CHECK(!run_moe(&m, 1, ids, w, x, NULL, NULL, MID_DIM / 2));
                CHECK(!run_moe(&m, 9, ids, w, x, NULL, NULL, MID_DIM / 2));
            }
            /* Leader-local drafting suspends sharding: raw experts, reference result. */
            ds4_gpu_tp_suspend_expert_sharding(1);
            CHECK(run_moe(&m, n, ids, w, x, y_raw, mid_raw, MID_DIM));
            ds4_gpu_tp_suspend_expert_sharding(0);
            raw_diff += memcmp(y_raw, y_ref, (size_t)n * OUT_DIM * 4u) != 0;
            raw_diff += memcmp(mid_raw, mid_ref, (size_t)slots * MID_DIM * 4u) != 0;
            ds4_gpu_tp_shutdown();
            /* No bound rank: raw experts again. */
            CHECK(run_moe(&m, n, ids, w, x, y_raw, mid_raw, MID_DIM));
            raw_diff += memcmp(y_raw, y_ref, (size_t)n * OUT_DIM * 4u) != 0;
            ds4_gpu_release_dspark_tp_split();
            ds4_gpu_tensor_free(slab);
        }
        uint64_t mid_diff = 0;
        for (uint32_t r = 0; r < 2; r++) {
            for (uint64_t a = 0; a < slots; a++) {
                if (memcmp(&mid_r[r][a * (MID_DIM / 2)],
                           &mid_ref[a * MID_DIM + (uint64_t)r * (MID_DIM / 2)],
                           (MID_DIM / 2) * 4u) != 0) {
                    mid_diff++;
                }
            }
        }
        double max_abs = 0.0, max_ref = 0.0;
        int nonfinite = 0;
        for (uint64_t i = 0; i < (uint64_t)n * OUT_DIM; i++) {
            const float s = y_r[0][i] + y_r[1][i];
            if (!isfinite(s) || !isfinite(y_ref[i])) nonfinite++;
            const double d = fabs((double)s - (double)y_ref[i]);
            if (d > max_abs) max_abs = d;
            if (fabs((double)y_ref[i]) > max_ref) max_ref = fabs((double)y_ref[i]);
        }
        const double rel = max_ref > 0 ? max_abs / max_ref : max_abs;
        const bool ok = nonfinite == 0 && rel < 1e-4 && mid_diff == 0 && raw_diff == 0 &&
                        max_ref > 0;
        printf("n_tokens=%u: mid rows differing %llu/%llu, y0+y1 vs ref max|d| %.3g (rel %.3g), "
               "raw paths differing %d, nonfinite %d  %s\n",
               n, (unsigned long long)mid_diff, (unsigned long long)(2u * slots),
               max_abs, rel, raw_diff, nonfinite, ok ? "PASS" : "FAIL");
        fails += !ok;
        free(x); free(w); free(ids); free(y_ref); free(mid_ref); free(y_raw); free(mid_raw);
        for (uint32_t r = 0; r < 2; r++) { free(y_r[r]); free(mid_r[r]); }
    }
    ds4_gpu_cleanup();
    close(m.fd);
    unlink(m.path);
    free(m.map);
    printf("DSpark TP intermediate split: %s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
