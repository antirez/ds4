/* The batched verifier compressor rows path
 * must be bit-identical to the per-row path it replaces.
 *
 * Reference: for each row, ds4_gpu_compressor_update_tensor (store, pool,
 * norm, rope, ratio-4 shift), then FP8 KV quantization (attention) or the
 * indexer QAT (indexer) of an emitted row, then a copy of the whole state
 * into prefix slot t -- exactly the per-row loop in
 * metal_graph_encode_layer_attention_batch.
 *
 * Candidate: ds4_gpu_compressor_verify_rows_tensor, then one RMS norm over
 * all emitted rows, rope per emitted row, one FP8/QAT pass -- exactly the
 * sequence of metal_graph_{attn,index}_compressor_rows_batch.
 *
 * Compared byte for byte: final kv/score state, every snapshot slot, and the
 * whole compressed cache (rows outside the emitted range hold a sentinel and
 * must stay untouched).  Covers ratio 4 and 128, head_dim 512 and 128, F32
 * and F16 APE, every pos0 phase around window boundaries, n_rows 1..8, with
 * and without snapshots, and -inf scores in the initial state. */
#include "ds4_gpu.h"

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* Side-effecting checks must survive -DNDEBUG. */
#define CHECK(x) do { if (!(x)) { fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #x); exit(2); } } while (0)

#define N_ROT 64u
#define SLOTS 8u
#define COMP_CAP 16u

static uint32_t g_seed = 0x5eed1234u;
static FILE *raw_dump;
static float frand(void) {
    g_seed = g_seed * 1664525u + 1013904223u;
    return (float)((int32_t)(g_seed >> 8) - (1 << 23)) / (float)(1 << 23);
}

static uint16_t f32_to_f16_bits(float f) {
    /* Small, exact-enough conversion for test weights in [-1, 1]. */
    uint32_t x;
    memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = (int32_t)((x >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = (x >> 13) & 0x3ffu;
    if (exp <= 0) return (uint16_t)sign;
    if (exp >= 31) return (uint16_t)(sign | 0x7c00u);
    return (uint16_t)(sign | ((uint32_t)exp << 10) | mant);
}

typedef struct {
    unsigned char *model;
    uint64_t size;
    uint64_t ape_f32, ape_f16, norm_attn, norm_index;
} model_buf;

static model_buf make_model(void) {
    model_buf m;
    const uint64_t ape_elems = 2ull * 512u * 128u;   /* covers width * ratio for every config */
    m.ape_f32 = 256;
    m.ape_f16 = m.ape_f32 + ape_elems * 4u;
    m.norm_attn = m.ape_f16 + ape_elems * 2u;
    m.norm_index = m.norm_attn + 512u * 4u;
    m.size = m.norm_index + 128u * 4u + 256u;
    m.model = calloc(1, (size_t)m.size);
    CHECK(m.model);
    for (uint64_t i = 0; i < ape_elems; i++) {
        const float v = frand() * 0.5f;
        memcpy(m.model + m.ape_f32 + i * 4u, &v, 4);
        const uint16_t h = f32_to_f16_bits(v);
        memcpy(m.model + m.ape_f16 + i * 2u, &h, 2);
    }
    for (uint32_t i = 0; i < 512u; i++) {
        const float v = 0.5f + 0.5f * frand();
        memcpy(m.model + m.norm_attn + i * 4u, &v, 4);
    }
    for (uint32_t i = 0; i < 128u; i++) {
        const float v = 0.5f + 0.5f * frand();
        memcpy(m.model + m.norm_index + i * 4u, &v, 4);
    }
    return m;
}

static ds4_gpu_tensor *upload(const void *data, uint64_t bytes) {
    ds4_gpu_tensor *t = ds4_gpu_tensor_alloc(bytes);
    CHECK(t);
    CHECK(ds4_gpu_tensor_write(t, 0, data, bytes));
    return t;
}

static int compare(const char *what, const ds4_gpu_tensor *a, const ds4_gpu_tensor *b, uint64_t bytes) {
    unsigned char *ha = malloc((size_t)bytes), *hb = malloc((size_t)bytes);
    CHECK(ha && hb);
    CHECK(ds4_gpu_tensor_read(a, 0, ha, bytes));
    CHECK(ds4_gpu_tensor_read(b, 0, hb, bytes));
    uint64_t diff = 0, first = UINT64_MAX;
    for (uint64_t i = 0; i < bytes; i++) {
        if (ha[i] != hb[i]) {
            if (first == UINT64_MAX) first = i;
            diff++;
        }
    }
    free(ha);
    free(hb);
    if (diff) {
        printf("    %s: %llu bytes differ (first at byte %llu, float %llu)\n", what,
               (unsigned long long)diff, (unsigned long long)first,
               (unsigned long long)(first / 4u));
    }
    return diff == 0;
}

typedef struct {
    uint32_t ratio, head_dim, ape_type, pos0, n_rows, n_snap;
    int index_mode;   /* 1: indexer (QAT), 0: attention (FP8) */
    int fresh_state;
} cfg;

static int run(const model_buf *m, const cfg *c) {
    const uint32_t coff = c->ratio == 4u ? 2u : 1u;
    const uint64_t width = (uint64_t)coff * c->head_dim;
    const uint64_t state_elems = (uint64_t)coff * c->ratio * width;
    const uint64_t state_bytes = state_elems * 4u;
    const uint64_t rows_bytes = (uint64_t)c->n_rows * width * 4u;
    const uint64_t comp_bytes = (uint64_t)COMP_CAP * c->head_dim * 4u;
    const uint32_t comp_row0 = 3u;   /* rows before it must stay sentinel */
    const uint64_t ape_off = c->ape_type ? m->ape_f16 : m->ape_f32;
    const uint64_t norm_off = c->index_mode ? m->norm_index : m->norm_attn;

    float *state_kv = malloc((size_t)state_bytes), *state_sc = malloc((size_t)state_bytes);
    float *kv = malloc((size_t)rows_bytes), *sc = malloc((size_t)rows_bytes);
    unsigned char *sentinel = malloc((size_t)comp_bytes);
    CHECK(state_kv && state_sc && kv && sc && sentinel);
    for (uint64_t i = 0; i < state_elems; i++) {
        state_kv[i] = frand();
        /* Some never-written rows keep the -inf score of a fresh state. */
        state_sc[i] = c->fresh_state || (i % 7u) == 3u ? -INFINITY : 2.0f * frand();
    }
    for (uint64_t i = 0; i < (uint64_t)c->n_rows * width; i++) {
        kv[i] = frand();
        sc[i] = 2.0f * frand();
    }
    memset(sentinel, 0xa5, (size_t)comp_bytes);

    ds4_gpu_tensor *kv_t = upload(kv, rows_bytes), *sc_t = upload(sc, rows_bytes);
    ds4_gpu_tensor *a_kv = upload(state_kv, state_bytes), *a_sc = upload(state_sc, state_bytes);
    ds4_gpu_tensor *b_kv = upload(state_kv, state_bytes), *b_sc = upload(state_sc, state_bytes);
    ds4_gpu_tensor *a_comp = upload(sentinel, comp_bytes), *b_comp = upload(sentinel, comp_bytes);
    ds4_gpu_tensor *a_snk = ds4_gpu_tensor_alloc(SLOTS * state_bytes);
    ds4_gpu_tensor *a_sns = ds4_gpu_tensor_alloc(SLOTS * state_bytes);
    ds4_gpu_tensor *b_snk = ds4_gpu_tensor_alloc(SLOTS * state_bytes);
    ds4_gpu_tensor *b_sns = ds4_gpu_tensor_alloc(SLOTS * state_bytes);
    CHECK(a_snk && a_sns && b_snk && b_sns);
    {
        unsigned char *z = calloc(1, (size_t)(SLOTS * state_bytes));
        CHECK(z);
        CHECK(ds4_gpu_tensor_write(a_snk, 0, z, SLOTS * state_bytes));
        CHECK(ds4_gpu_tensor_write(a_sns, 0, z, SLOTS * state_bytes));
        CHECK(ds4_gpu_tensor_write(b_snk, 0, z, SLOTS * state_bytes));
        CHECK(ds4_gpu_tensor_write(b_sns, 0, z, SLOTS * state_bytes));
        free(z);
    }
    const float freq_base = 10000.0f, freq_scale = 0.0625f, ext_factor = 1.0f;
    const float attn_factor = 1.0f / (1.0f + 0.1f * logf(16.0f));
    const uint32_t n_ctx_orig = 65536u;

    /* Reference: per-row loop. */
    uint32_t comp_row = comp_row0;
    for (uint32_t t = 0; t < c->n_rows; t++) {
        const uint32_t pos = c->pos0 + t;
        const int emit = ((pos + 1u) % c->ratio) == 0u;
        ds4_gpu_tensor *kv_v = ds4_gpu_tensor_view(kv_t, (uint64_t)t * width * 4u, width * 4u);
        ds4_gpu_tensor *sc_v = ds4_gpu_tensor_view(sc_t, (uint64_t)t * width * 4u, width * 4u);
        CHECK(kv_v && sc_v);
        CHECK(ds4_gpu_compressor_update_tensor(kv_v, sc_v, a_kv, a_sc, a_comp,
                                                m->model, m->size, ape_off, c->ape_type,
                                                norm_off, 0u, c->head_dim, c->ratio, pos,
                                                comp_row, N_ROT, n_ctx_orig, freq_base,
                                                freq_scale, ext_factor, attn_factor,
                                                32.0f, 1.0f, 1e-6f, false, false, false));
        if (emit) {
            ds4_gpu_tensor *row = ds4_gpu_tensor_view(a_comp, (uint64_t)comp_row * c->head_dim * 4u,
                                                      (uint64_t)c->head_dim * 4u);
            CHECK(row);
            if (c->index_mode) CHECK(ds4_gpu_dsv4_indexer_qat_tensor(row, 1, c->head_dim));
            else CHECK(ds4_gpu_dsv4_fp8_kv_quantize_tensor(row, 1, c->head_dim, N_ROT));
            ds4_gpu_tensor_free(row);
            comp_row++;
        }
        if (t < c->n_snap) {
            CHECK(ds4_gpu_tensor_copy(a_snk, (uint64_t)t * state_bytes, a_kv, 0, state_bytes));
            CHECK(ds4_gpu_tensor_copy(a_sns, (uint64_t)t * state_bytes, a_sc, 0, state_bytes));
        }
        ds4_gpu_tensor_free(kv_v);
        ds4_gpu_tensor_free(sc_v);
    }
    const uint32_t emits = comp_row - comp_row0;

    /* Candidate: batched path, same call sequence as ds4.c. */
    CHECK(ds4_gpu_compressor_verify_rows_tensor(kv_t, sc_t, b_kv, b_sc, b_comp, comp_row0,
                                                 c->n_snap ? b_snk : NULL, c->n_snap ? b_sns : NULL,
                                                 c->n_snap, m->model, m->size, ape_off,
                                                 c->ape_type, c->head_dim, c->ratio, c->pos0,
                                                 c->n_rows));
    /* Compare dumps from old/new builds before norm or FP8/QAT can hide
     * a difference in the pooled values. Include untouched cache canaries. */
    if (raw_dump) {
        void *raw = malloc((size_t)comp_bytes);
        CHECK(raw);
        CHECK(ds4_gpu_tensor_read(b_comp, 0, raw, comp_bytes));
        CHECK(fwrite(raw, 1, (size_t)comp_bytes, raw_dump) == comp_bytes);
        free(raw);
    }
    if (emits) {
        ds4_gpu_tensor *rows = ds4_gpu_tensor_view(b_comp, (uint64_t)comp_row0 * c->head_dim * 4u,
                                                   (uint64_t)emits * c->head_dim * 4u);
        CHECK(rows);
        CHECK(ds4_gpu_rms_norm_weight_rows_tensor(rows, rows, m->model, m->size, norm_off,
                                                   c->head_dim, emits, 1e-6f));
        uint32_t k = 0;
        for (uint32_t t = 0; t < c->n_rows; t++) {
            const uint32_t pos = c->pos0 + t;
            if (((pos + 1u) % c->ratio) != 0u) continue;
            ds4_gpu_tensor *row = ds4_gpu_tensor_view(b_comp,
                                                      (uint64_t)(comp_row0 + k) * c->head_dim * 4u,
                                                      (uint64_t)c->head_dim * 4u);
            CHECK(row);
            CHECK(ds4_gpu_rope_tail_tensor(row, 1, 1, c->head_dim, N_ROT, pos + 1u - c->ratio,
                                            n_ctx_orig, false, freq_base, freq_scale,
                                            ext_factor, attn_factor, 32.0f, 1.0f));
            ds4_gpu_tensor_free(row);
            k++;
        }
        if (c->index_mode) CHECK(ds4_gpu_dsv4_indexer_qat_tensor(rows, emits, c->head_dim));
        else CHECK(ds4_gpu_dsv4_fp8_kv_quantize_tensor(rows, emits, c->head_dim, N_ROT));
        ds4_gpu_tensor_free(rows);
    }
    CHECK(ds4_gpu_synchronize());

    int ok = 1;
    ok &= compare("state_kv", a_kv, b_kv, state_bytes);
    ok &= compare("state_score", a_sc, b_sc, state_bytes);
    ok &= compare("snapshot_kv", a_snk, b_snk, SLOTS * state_bytes);
    ok &= compare("snapshot_score", a_sns, b_sns, SLOTS * state_bytes);
    ok &= compare("comp_cache", a_comp, b_comp, comp_bytes);
    if (!ok) {
        printf("  FAIL ratio=%u head_dim=%u ape=%s pos0=%u n_rows=%u n_snap=%u %s emits=%u\n",
               c->ratio, c->head_dim, c->ape_type ? "f16" : "f32", c->pos0, c->n_rows,
               c->n_snap, c->index_mode ? "indexer" : "attention", emits);
    }

    ds4_gpu_tensor *all[] = {kv_t, sc_t, a_kv, a_sc, b_kv, b_sc, a_comp, b_comp,
                             a_snk, a_sns, b_snk, b_sns};
    for (size_t i = 0; i < sizeof(all) / sizeof(all[0]); i++) ds4_gpu_tensor_free(all[i]);
    free(state_kv);
    free(state_sc);
    free(kv);
    free(sc);
    free(sentinel);
    return ok;
}

int main(int argc, char **argv) {
    if (argc > 2) {
        fprintf(stderr, "usage: %s [raw-pool-dump]\n", argv[0]);
        return 2;
    }
    if (argc == 2) {
        raw_dump = fopen(argv[1], "wb");
        CHECK(raw_dump);
    }
    model_buf m = make_model();
    char path[] = "/tmp/ds4-compressor-rows-XXXXXX";
    const int fd = mkstemp(path);
    CHECK(fd >= 0);
    CHECK(unlink(path) == 0);
    for (uint64_t n = 0; n < m.size;) {
        const ssize_t wrote = write(fd, m.model + n, (size_t)(m.size - n));
        if (wrote < 0 && errno == EINTR) continue;
        CHECK(wrote > 0);
        n += (uint64_t)wrote;
    }
    const uint64_t offset = 0;
    CHECK(ds4_gpu_init());
    CHECK(ds4_gpu_set_model_map_spans(m.model, m.size, &offset, &m.size, 1, 0));
    CHECK(ds4_gpu_set_model_fd_for_map(fd, m.model));

    /* pos0 phases: every residue of 4 plus both sides of 128 boundaries. */
    const uint32_t pos_list[] = {0, 1, 2, 3, 4, 5, 6, 7, 120, 121, 122, 123, 124, 125, 126, 127,
                                 128, 250, 251, 252, 253, 254, 255, 1021, 1022, 1023};
    int runs = 0, fails = 0;
    for (int mode = 0; mode < 3; mode++) {
        /* 0: attention ratio 4 (head 512), 1: attention ratio 128, 2: indexer ratio 4 (head 128). */
        const uint32_t ratio = mode == 1 ? 128u : 4u;
        const uint32_t head_dim = mode == 2 ? 128u : 512u;
        for (uint32_t ape = 0; ape < 2; ape++) {
            for (size_t p = 0; p < sizeof(pos_list) / sizeof(pos_list[0]); p++) {
                for (uint32_t n = 1; n <= 8; n++) {
                    for (int snaps = 0; snaps < 3; snaps++) {
                        cfg c = {ratio, head_dim, ape, pos_list[p], n, 0, mode == 2, 0};
                        if (snaps == 1) c.n_snap = n - 1u < 5u ? n - 1u : 5u;
                        if (snaps == 2) c.n_snap = n;
                        if (snaps == 1 && c.n_snap == 0) continue;
                        fails += !run(&m, &c);
                        runs++;
                    }
                }
            }
        }
    }
    for (int mode = 0; mode < 3; mode++) {
        for (uint32_t ape = 0; ape < 2; ape++) {
            for (uint32_t n = 1; n <= 8; n++) {
                cfg c = {mode == 1 ? 128u : 4u, mode == 2 ? 128u : 512u,
                         ape, 0, n, n, mode == 2, 1};
                fails += !run(&m, &c);
                runs++;
            }
        }
    }
    printf("compressor verify rows vs per-row path: %s (%d of %d configurations failing)\n",
           fails ? "FAIL" : "PASS", fails, runs);
    ds4_gpu_cleanup();
    close(fd);
    free(m.model);
    if (raw_dump) CHECK(fclose(raw_dump) == 0);
    return fails ? 1 : 0;
}
