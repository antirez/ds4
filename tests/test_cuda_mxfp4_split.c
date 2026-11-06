/* Network-TP intermediate split of MXFP4 routed experts.
 *
 * Reference: routed MoE with no TP (all experts, full intermediate).
 * Split: for rank 0 and rank 1, set the network TP rank, build the split
 * artifacts from the model file and run the same routed MoE call, which then
 * returns this rank's partial output over half of every expert.
 *
 * Checks:
 *  - gate/up intermediate rows: rank r's mid rows must equal the reference's
 *    rows [r*mid/2, +mid/2) of every assignment bit for bit (vector path,
 *    n_tokens <= 8; reported but not required for the MMQ prefill path);
 *  - output: y0 + y1 must match the reference within float summation-order
 *    error (the down projection's K halves are summed across ranks);
 *  - no NaN/Inf anywhere.
 * Shapes are scaled down (in 1024, mid 1024, out 1024, 16 experts) but keep
 * the production layout rules (K multiple of 256, mid/2 multiple of 256). */
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

enum { IN_DIM = 1024, MID_DIM = 1024, OUT_DIM = 1024, N_EXP = 16, TOPK = 6 };

static uint32_t g_seed = 0x7e57u;
static uint32_t rnd(void) { g_seed = g_seed * 1664525u + 1013904223u; return g_seed; }
static float frand(void) { return (float)((int32_t)(rnd() >> 8) - (1 << 23)) / (float)(1 << 23); }

static void fill_mxfp4(unsigned char *p, uint64_t blocks) {
    for (uint64_t b = 0; b < blocks; b++, p += 17) {
        p[0] = (unsigned char)(118 + rnd() % 8u);   /* E8M0 scale 2^-9 .. 2^-2 */
        for (int i = 1; i < 17; i++) p[i] = (unsigned char)rnd();
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

static void setup_gpu(model_t *m) {
    const uint64_t off = 0;
    CHECK(ds4_gpu_init());
    CHECK(ds4_gpu_set_model_map_spans(m->map, m->size, &off, &m->size, 1, 0));
    CHECK(ds4_gpu_set_model_fd_for_map(m->fd, m->map));
}

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
                                          39u, 39u, m->gate_expert, m->gate_row,
                                          m->down_expert, m->down_row,
                                          IN_DIM, MID_DIM, OUT_DIM, t_ids, t_w,
                                          N_EXP, TOPK, 10.0f, t_x, 0u, n_tokens,
                                          &mid_is_f16, false);
    CHECK(!mid_is_f16);
    CHECK(ds4_gpu_synchronize());
    if (ok) {
        CHECK(ds4_gpu_tensor_read(t_out, 0, y, (uint64_t)n_tokens * OUT_DIM * 4u));
        CHECK(ds4_gpu_tensor_read(t_mid, 0, mid, slots * mid_dim_out * 4u));
    }
    ds4_gpu_tensor *all[] = {t_x, t_ids, t_w, t_out, t_gate, t_up, t_mid, t_down};
    for (size_t i = 0; i < sizeof(all) / sizeof(all[0]); i++) ds4_gpu_tensor_free(all[i]);
    return ok;
}

int main(void) {
    model_t m;
    memset(&m, 0, sizeof(m));
    m.gate_row = (IN_DIM / 32) * 17u;
    m.gate_expert = (uint64_t)MID_DIM * m.gate_row;
    m.down_row = (MID_DIM / 32) * 17u;
    m.down_expert = (uint64_t)OUT_DIM * m.down_row;
    m.gate_off = 4096;
    m.up_off = m.gate_off + ((N_EXP * m.gate_expert + 4095u) & ~4095ull);
    m.down_off = m.up_off + ((N_EXP * m.gate_expert + 4095u) & ~4095ull);
    m.size = m.down_off + N_EXP * m.down_expert + 4096u;
    m.map = calloc(1, (size_t)m.size);
    CHECK(m.map);
    fill_mxfp4(m.map + m.gate_off, N_EXP * m.gate_expert / 17u);
    fill_mxfp4(m.map + m.up_off, N_EXP * m.gate_expert / 17u);
    fill_mxfp4(m.map + m.down_off, N_EXP * m.down_expert / 17u);
    strcpy(m.path, "/tmp/ds4-mx-tp-split-XXXXXX");
    m.fd = mkstemp(m.path);
    CHECK(m.fd >= 0);
    for (uint64_t n = 0; n < m.size;) {
        const ssize_t wrote = write(m.fd, m.map + n, (size_t)(m.size - n));
        if (wrote < 0 && errno == EINTR) continue;
        CHECK(wrote > 0);
        n += (uint64_t)wrote;
    }
    const uint64_t offsets[3] = {m.gate_off, m.up_off, m.down_off};
    const uint32_t is_down[3] = {0, 0, 1};
    const uint32_t in_dims[3] = {IN_DIM, IN_DIM, MID_DIM};
    const uint32_t out_dims[3] = {MID_DIM, MID_DIM, OUT_DIM};
    const uint32_t n_experts[3] = {N_EXP, N_EXP, N_EXP};

    setup_gpu(&m);
    const uint32_t token_counts[] = {1, 2, 6, 8, 16};
    int fails = 0;
    for (size_t tc = 0; tc < sizeof(token_counts) / sizeof(token_counts[0]); tc++) {
        const uint32_t n = token_counts[tc];
        const uint64_t slots = (uint64_t)n * TOPK;
        float *x = malloc((size_t)n * IN_DIM * 4u), *w = malloc((size_t)slots * 4u);
        int32_t *ids = malloc((size_t)slots * 4u);
        float *y_ref = malloc((size_t)n * OUT_DIM * 4u), *mid_ref = malloc((size_t)slots * MID_DIM * 4u);
        float *y_r[2], *mid_r[2];
        CHECK(x && w && ids && y_ref && mid_ref);
        for (uint64_t i = 0; i < (uint64_t)n * IN_DIM; i++) x[i] = frand();
        for (uint32_t t = 0; t < n; t++) {
            /* Distinct experts per token, repeats across tokens. */
            const uint32_t base = rnd() % N_EXP;
            for (uint32_t k = 0; k < TOPK; k++) {
                ids[t * TOPK + k] = (int32_t)((base + k * 3u) % N_EXP);
                w[t * TOPK + k] = 0.05f + 0.3f * fabsf(frand());
            }
        }
        CHECK(run_moe(&m, n, ids, w, x, y_ref, mid_ref, MID_DIM));
        for (uint32_t r = 0; r < 2; r++) {
            y_r[r] = malloc((size_t)n * OUT_DIM * 4u);
            mid_r[r] = malloc((size_t)slots * (MID_DIM / 2) * 4u);
            CHECK(y_r[r] && mid_r[r]);
            ds4_gpu_tensor *slab = ds4_gpu_tensor_alloc(4096);
            CHECK(slab);
            CHECK(ds4_gpu_tp_init(r, slab, 0, 64, 64, noop_exchange, NULL));
            CHECK(ds4_gpu_build_mxfp4_tp_split(m.map, m.size, m.size, m.path, r, offsets,
                                               is_down, in_dims, out_dims, n_experts, 3) ==
                  N_EXP * (2u * m.gate_expert + m.down_expert) / 2u);
            CHECK(run_moe(&m, n, ids, w, x, y_r[r], mid_r[r], MID_DIM / 2));
            ds4_gpu_tp_shutdown();
            CHECK(!run_moe(&m, n, ids, w, x, y_r[r], mid_r[r], MID_DIM / 2));
            ds4_gpu_release_mxfp4_tp_split();
            ds4_gpu_tensor_free(slab);
        }
        /* Intermediate rows. */
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
        /* Output. */
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
        const bool vec_path = n <= 8;
        const bool ok = nonfinite == 0 && rel < 1e-4 && (!vec_path || mid_diff == 0);
        printf("n_tokens=%2u %s: mid rows differing %llu/%llu, y0+y1 vs ref max|d| %.3g (rel %.3g), nonfinite %d  %s\n",
               n, vec_path ? "vec" : "mmq", (unsigned long long)mid_diff,
               (unsigned long long)(2u * slots), max_abs, rel, nonfinite, ok ? "PASS" : "FAIL");
        fails += !ok;
        free(x); free(w); free(ids); free(y_ref); free(mid_ref);
        for (uint32_t r = 0; r < 2; r++) { free(y_r[r]); free(mid_r[r]); }
    }
    ds4_gpu_cleanup();
    close(m.fd);
    unlink(m.path);
    free(m.map);
    printf("MXFP4 TP intermediate split: %s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
