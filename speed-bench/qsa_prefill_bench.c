/* Standalone GPU attribution bench for the Qwen prefill stages.  It drives
 * the production dispatch entry points at the exact model geometry
 * (24 query heads / 2 KV heads / head_dim 256, indexer 4x128, 2048-token
 * budget over ratio-4 blocks = 512-block top-k, 512-query microtiles with
 * 1024-block score tiles, plus the routed Q4_0 expert family and a
 * representative dense Q8_0 projection) so uncaptured per-stage kernel
 * times can be compared against the whole-chunk wall time without
 * perturbing the graph.
 */
#include "ds4_gpu.h"

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

/* ds4_metal.o references the shared logging helper. */
bool ds4_log_is_tty(FILE *fp) {
    (void)fp;
    return false;
}

enum {
    Q_HEADS = 24,
    KV_HEADS = 2,
    HEAD_DIM = 256,
    IDX_HEADS = 4,
    IDX_DIM = 128,
    RATIO = 4,
    BLOCK_TOPK = 512,   /* 2048-token indexer budget / ratio 4 */
    MICROTILE = 512,
    BLOCK_TILE = 1024,
    MAX_ROWS = 8192,
    /* Gated DeltaNet geometry (ds4_qwen4.h). */
    GDN_KEY_HEADS = 16,
    GDN_VALUE_HEADS = 48,
    GDN_HEAD_DIM = 128,
    GDN_ROWS_PER_THREAD = 4,
    /* Representative dense projection: the QSA fused query+gate projection. */
    DENSE_OUT = 12288,
    DENSE_IN = 2560
};

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1.0e6;
}

/* GDN recurrence rows-per-thread for the prefill-width sweep:
 * DS4_BENCH_GDN_ROWS overrides the production default of 4. */
static uint32_t gdn_rows(void) {
    const char *raw = getenv("DS4_BENCH_GDN_ROWS");
    if (raw && raw[0]) {
        char *end = NULL;
        const unsigned long parsed = strtoul(raw, &end, 10);
        if (end != raw && *end == '\0' &&
            (parsed == 1u || parsed == 2u || parsed == 4u))
            return (uint32_t)parsed;
    }
    return GDN_ROWS_PER_THREAD;
}

static uint32_t lcg_state = 0x12345678u;
static float lcg_float(float scale) {
    lcg_state = lcg_state * 1664525u + 1013904223u;
    return ((float)(lcg_state >> 8) / (float)(1u << 24) - 0.5f) * scale;
}

static size_t align_up_size(size_t v, size_t a) {
    return (v + a - 1u) / a * a;
}

/* Fill Q4_0 blocks: a small positive F16 scale plus random nibble pairs.
 * The bench times kernels, not numerics; scales only need to stay normal
 * so dequantized tiles are ordinary floats. */
static void fill_q4_0_run(uint8_t *base, size_t bytes) {
    for (size_t i = 0; i + 18u <= bytes; i += 18u) {
        base[i] = 0x00;
        base[i + 1u] = 0x2c; /* F16 0.03125-ish */
        for (int b = 0; b < 4; b++) {
            const uint32_t r = lcg_state * 1664525u + 1013904223u;
            lcg_state = r;
            memcpy(base + i + 2u + (size_t)b * 4u, &r, 4u);
        }
    }
}

static uint16_t lcg_bf16(void) {
    const float v = lcg_float(2.0f);
    uint32_t bits;
    memcpy(&bits, &v, sizeof(bits));
    return (uint16_t)(bits >> 16);
}

static int median_of(double *samples, int count) {
    for (int i = 1; i < count; i++) {
        const double key = samples[i];
        int j = i - 1;
        while (j >= 0 && samples[j] > key) {
            samples[j + 1] = samples[j];
            j--;
        }
        samples[j + 1] = key;
    }
    return count / 2;
}

int main(int argc, char **argv) {
    uint32_t rows = MAX_ROWS;
    if (argc > 1) rows = (uint32_t)strtoul(argv[1], NULL, 0);
    if (rows == 0u || rows > MAX_ROWS) {
        fprintf(stderr, "qsa-prefill-bench: rows must be 1..%u\n", MAX_ROWS);
        return 1;
    }
    const uint32_t blocks = rows / RATIO;
    if (blocks == 0u) {
        fprintf(stderr, "qsa-prefill-bench: rows must cover at least one block\n");
        return 1;
    }
    const uint32_t selected_blocks = blocks < BLOCK_TOPK ? blocks : BLOCK_TOPK;

    /* --- gathered-attention tensors (worst case: every query selects the
     * full 512-block budget and sees the whole chunk). --- */
    const size_t q_elements = (size_t)rows * Q_HEADS * HEAD_DIM;
    const size_t cache_elements = (size_t)MAX_ROWS * KV_HEADS * HEAD_DIM;
    float *q = malloc(q_elements * sizeof(float));
    float *gate = malloc(q_elements * sizeof(float));
    uint16_t *key = malloc(cache_elements * sizeof(uint16_t));
    uint16_t *value = malloc(cache_elements * sizeof(uint16_t));
    uint32_t *selected = malloc((size_t)rows * BLOCK_TOPK * sizeof(uint32_t));
    uint32_t *counts = malloc((size_t)rows * sizeof(uint32_t));
    uint32_t *visible = malloc((size_t)rows * sizeof(uint32_t));
    float *out = malloc(q_elements * sizeof(float));
    if (!q || !gate || !key || !value || !selected || !counts || !visible ||
        !out) {
        fprintf(stderr, "qsa-prefill-bench: host allocation failed\n");
        return 1;
    }
    for (size_t i = 0; i < q_elements; i++) {
        q[i] = lcg_float(1.0f);
        gate[i] = lcg_float(1.0f);
    }
    for (size_t i = 0; i < cache_elements; i++) {
        key[i] = lcg_bf16();
        value[i] = lcg_bf16();
    }
    for (uint32_t query = 0; query < rows; query++) {
        /* Distinct per-query block orders so no accidental sharing can
         * flatter the gather cost; every query still selects the full
         * budget. */
        uint32_t block = query % blocks;
        for (uint32_t rank = 0; rank < selected_blocks; rank++) {
            selected[(size_t)query * BLOCK_TOPK + rank] = block;
            block = block + 1u >= blocks ? 0u : block + 1u;
        }
        counts[query] = selected_blocks;
        visible[query] = rows;
    }

    ds4_gpu_tensor *q_t = ds4_gpu_tensor_alloc(q_elements * sizeof(float));
    ds4_gpu_tensor *gate_t = ds4_gpu_tensor_alloc(q_elements * sizeof(float));
    ds4_gpu_tensor *key_t =
        ds4_gpu_tensor_alloc(cache_elements * sizeof(uint16_t));
    ds4_gpu_tensor *value_t =
        ds4_gpu_tensor_alloc(cache_elements * sizeof(uint16_t));
    ds4_gpu_tensor *selected_t =
        ds4_gpu_tensor_alloc((size_t)rows * BLOCK_TOPK * sizeof(uint32_t));
    ds4_gpu_tensor *counts_t =
        ds4_gpu_tensor_alloc((size_t)rows * sizeof(uint32_t));
    ds4_gpu_tensor *visible_t =
        ds4_gpu_tensor_alloc((size_t)rows * sizeof(uint32_t));
    ds4_gpu_tensor *out_t = ds4_gpu_tensor_alloc(q_elements * sizeof(float));
    if (!q_t || !gate_t || !key_t || !value_t || !selected_t || !counts_t ||
        !visible_t || !out_t) {
        fprintf(stderr, "qsa-prefill-bench: device allocation failed\n");
        return 1;
    }
    if (ds4_gpu_tensor_write(q_t, 0, q, q_elements * sizeof(float)) == 0 ||
        ds4_gpu_tensor_write(gate_t, 0, gate, q_elements * sizeof(float)) ==
            0 ||
        ds4_gpu_tensor_write(key_t, 0, key,
                             cache_elements * sizeof(uint16_t)) == 0 ||
        ds4_gpu_tensor_write(value_t, 0, value,
                             cache_elements * sizeof(uint16_t)) == 0 ||
        ds4_gpu_tensor_write(selected_t, 0, selected,
                             (size_t)rows * BLOCK_TOPK * sizeof(uint32_t)) ==
            0 ||
        ds4_gpu_tensor_write(counts_t, 0, counts,
                             (size_t)rows * sizeof(uint32_t)) == 0 ||
        ds4_gpu_tensor_write(visible_t, 0, visible,
                             (size_t)rows * sizeof(uint32_t)) == 0) {
        fprintf(stderr, "qsa-prefill-bench: upload failed\n");
        return 1;
    }

    /* --- streaming top-k tensors (causal visible-block ramp like the
     * production chunk at cache_pos=0).  DS4_BENCH_QSA_VISIBLE_BLOCKS
     * overrides the pooled-index geometry to a long-context last-chunk
     * shape: every query sees the full block count, matching the
     * item-22h attribution regime. --- */
    uint32_t visible_geometry = blocks;
    const char *visible_env = getenv("DS4_BENCH_QSA_VISIBLE_BLOCKS");
    if (visible_env && *visible_env) {
        const unsigned long parsed = strtoul(visible_env, NULL, 0);
        if (parsed == 0u || parsed > (1u << 20u)) {
            fprintf(stderr,
                    "qsa-prefill-bench: DS4_BENCH_QSA_VISIBLE_BLOCKS must be "
                    "1..%u\n",
                    1u << 20u);
            return 1;
        }
        visible_geometry = (uint32_t)parsed;
    }
    const size_t idx_q_elements = (size_t)rows * IDX_HEADS * IDX_DIM;
    float *idx_q = malloc(idx_q_elements * sizeof(float));
    uint16_t *pooled_k =
        malloc((size_t)visible_geometry * IDX_HEADS * IDX_DIM *
               sizeof(uint16_t));
    uint32_t *visible_blocks = malloc((size_t)rows * sizeof(uint32_t));
    if (!idx_q || !pooled_k || !visible_blocks) {
        fprintf(stderr, "qsa-prefill-bench: topk host allocation failed\n");
        return 1;
    }
    for (size_t i = 0; i < idx_q_elements; i++) idx_q[i] = lcg_float(1.0f);
    for (size_t i = 0;
         i < (size_t)visible_geometry * IDX_HEADS * IDX_DIM;
         i++)
        pooled_k[i] = lcg_bf16();
    for (uint32_t query = 0; query < rows; query++)
        visible_blocks[query] = visible_geometry;
    ds4_gpu_tensor *idx_q_t =
        ds4_gpu_tensor_alloc(idx_q_elements * sizeof(float));
    ds4_gpu_tensor *pooled_k_t = ds4_gpu_tensor_alloc(
        (size_t)visible_geometry * IDX_HEADS * IDX_DIM * sizeof(uint16_t));
    ds4_gpu_tensor *visible_blocks_t =
        ds4_gpu_tensor_alloc((size_t)rows * sizeof(uint32_t));
    ds4_gpu_tensor *topk_scores_t =
        ds4_gpu_tensor_alloc((size_t)rows * BLOCK_TOPK * sizeof(float));
    ds4_gpu_tensor *topk_indices_t =
        ds4_gpu_tensor_alloc((size_t)rows * BLOCK_TOPK * sizeof(uint32_t));
    ds4_gpu_tensor *topk_tile_t = ds4_gpu_tensor_alloc(
        (size_t)MICROTILE * BLOCK_TILE * sizeof(float));
    if (!idx_q_t || !pooled_k_t || !visible_blocks_t || !topk_scores_t ||
        !topk_indices_t || !topk_tile_t) {
        fprintf(stderr, "qsa-prefill-bench: topk device allocation failed\n");
        return 1;
    }
    if (ds4_gpu_tensor_write(idx_q_t, 0, idx_q,
                             idx_q_elements * sizeof(float)) == 0 ||
        ds4_gpu_tensor_write(pooled_k_t, 0, pooled_k,
                             (size_t)visible_geometry * IDX_HEADS *
                                 IDX_DIM * sizeof(uint16_t)) == 0 ||
        ds4_gpu_tensor_write(visible_blocks_t, 0, visible_blocks,
                             (size_t)rows * sizeof(uint32_t)) == 0) {
        fprintf(stderr, "qsa-prefill-bench: topk upload failed\n");
        return 1;
    }

    /* --- Gated DeltaNet prefill recurrence tensors (BF16-state production
     * variant; mask NULL like the ordinary chunk path). --- */
    const size_t gdn_q_elements = (size_t)rows * GDN_KEY_HEADS * GDN_HEAD_DIM;
    const size_t gdn_v_elements =
        (size_t)rows * GDN_VALUE_HEADS * GDN_HEAD_DIM;
    const size_t gdn_state_elements = (size_t)GDN_VALUE_HEADS *
        GDN_HEAD_DIM * GDN_HEAD_DIM;
    float *gdn_q = malloc(gdn_q_elements * sizeof(float));
    float *gdn_k = malloc(gdn_q_elements * sizeof(float));
    float *gdn_v = malloc(gdn_v_elements * sizeof(float));
    float *gdn_decay = malloc((size_t)rows * GDN_VALUE_HEADS * sizeof(float));
    float *gdn_beta = malloc((size_t)rows * GDN_VALUE_HEADS * sizeof(float));
    float *gdn_out = malloc(gdn_v_elements * sizeof(float));
    uint16_t *gdn_state = malloc(gdn_state_elements * sizeof(uint16_t));
    if (!gdn_q || !gdn_k || !gdn_v || !gdn_decay || !gdn_beta || !gdn_out ||
        !gdn_state) {
        fprintf(stderr, "qsa-prefill-bench: gdn host allocation failed\n");
        return 1;
    }
    for (size_t i = 0; i < gdn_q_elements; i++) {
        gdn_q[i] = lcg_float(1.0f);
        gdn_k[i] = lcg_float(1.0f);
    }
    for (size_t i = 0; i < gdn_v_elements; i++) gdn_v[i] = lcg_float(1.0f);
    for (size_t i = 0; i < (size_t)rows * GDN_VALUE_HEADS; i++) {
        gdn_decay[i] = 0.99f + lcg_float(0.005f);
        gdn_beta[i] = 0.9f + lcg_float(0.05f);
    }
    for (size_t i = 0; i < gdn_state_elements; i++) gdn_state[i] = 0;
    ds4_gpu_tensor *gdn_q_t =
        ds4_gpu_tensor_alloc(gdn_q_elements * sizeof(float));
    ds4_gpu_tensor *gdn_k_t =
        ds4_gpu_tensor_alloc(gdn_q_elements * sizeof(float));
    ds4_gpu_tensor *gdn_v_t =
        ds4_gpu_tensor_alloc(gdn_v_elements * sizeof(float));
    ds4_gpu_tensor *gdn_decay_t = ds4_gpu_tensor_alloc(
        (size_t)rows * GDN_VALUE_HEADS * sizeof(float));
    ds4_gpu_tensor *gdn_beta_t = ds4_gpu_tensor_alloc(
        (size_t)rows * GDN_VALUE_HEADS * sizeof(float));
    ds4_gpu_tensor *gdn_out_t =
        ds4_gpu_tensor_alloc(gdn_v_elements * sizeof(float));
    ds4_gpu_tensor *gdn_state_t =
        ds4_gpu_tensor_alloc(gdn_state_elements * sizeof(uint16_t));
    if (!gdn_q_t || !gdn_k_t || !gdn_v_t || !gdn_decay_t || !gdn_beta_t ||
        !gdn_out_t || !gdn_state_t) {
        fprintf(stderr, "qsa-prefill-bench: gdn device allocation failed\n");
        return 1;
    }
    if (ds4_gpu_tensor_write(gdn_q_t, 0, gdn_q,
                             gdn_q_elements * sizeof(float)) == 0 ||
        ds4_gpu_tensor_write(gdn_k_t, 0, gdn_k,
                             gdn_q_elements * sizeof(float)) == 0 ||
        ds4_gpu_tensor_write(gdn_v_t, 0, gdn_v,
                             gdn_v_elements * sizeof(float)) == 0 ||
        ds4_gpu_tensor_write(gdn_decay_t, 0, gdn_decay,
                             (size_t)rows * GDN_VALUE_HEADS *
                                 sizeof(float)) == 0 ||
        ds4_gpu_tensor_write(gdn_beta_t, 0, gdn_beta,
                             (size_t)rows * GDN_VALUE_HEADS *
                                 sizeof(float)) == 0 ||
        ds4_gpu_tensor_write(gdn_state_t, 0, gdn_state,
                             gdn_state_elements * sizeof(uint16_t)) == 0) {
        fprintf(stderr, "qsa-prefill-bench: gdn upload failed\n");
        return 1;
    }

    /* --- representative dense Q8_0 projection (fused QSA query+gate) and
     * the routed Q4_0 expert family, sharing one registered model map. --- */
    const size_t dense_x_elements = (size_t)rows * DENSE_IN;
    const size_t dense_weight_bytes =
        (size_t)DENSE_OUT * (DENSE_IN / 32u) * 34u; /* Q8_0: 34B/32 values */
    /* Q4_0: 18 B per 32 values.  Gate/up are [512, 640, 2560]; down is
     * [512, 2560, 768] (640 logical + 128-value zero tail). */
    enum {
        MOE_EXPERTS = 512,
        MOE_TOP_K = 10,
        MOE_FF = 640,
        MOE_DOWN_PAD = 768
    };
    const size_t moe_gate_bytes =
        (size_t)MOE_EXPERTS * MOE_FF * (DENSE_IN / 32u) * 18u;
    const size_t moe_down_bytes =
        (size_t)MOE_EXPERTS * DENSE_IN * (MOE_DOWN_PAD / 32u) * 18u;
    const size_t dense_end = align_up_size(dense_weight_bytes, 4096u);
    const size_t moe_gate_offset = dense_end;
    const size_t moe_up_offset = moe_gate_offset + moe_gate_bytes;
    const size_t moe_down_offset = moe_up_offset + moe_gate_bytes;
    const size_t model_total = moe_down_offset + moe_down_bytes;
    void *dense_map = NULL;
    if (posix_memalign(&dense_map, 4096u, model_total) != 0) {
        fprintf(stderr, "qsa-prefill-bench: dense host allocation failed\n");
        return 1;
    }
    uint8_t *dense_weights = dense_map;
    uint8_t *moe_gate = (uint8_t *)dense_map + moe_gate_offset;
    uint8_t *moe_up = (uint8_t *)dense_map + moe_up_offset;
    uint8_t *moe_down = (uint8_t *)dense_map + moe_down_offset;
    float *dense_x = malloc(dense_x_elements * sizeof(float));
    float *dense_out = malloc((size_t)rows * DENSE_OUT * sizeof(float));
    if (!dense_x || !dense_out) {
        fprintf(stderr, "qsa-prefill-bench: dense host allocation failed\n");
        return 1;
    }
    for (size_t i = 0; i < dense_weight_bytes; i++)
        dense_weights[i] = (uint8_t)(lcg_state >> 13);
    fill_q4_0_run(moe_gate, moe_gate_bytes);
    fill_q4_0_run(moe_up, moe_gate_bytes);
    fill_q4_0_run(moe_down, moe_down_bytes);
    if (ds4_gpu_set_model_map(dense_map, model_total) == 0) {
        fprintf(stderr, "qsa-prefill-bench: dense model registration failed\n");
        return 1;
    }
    for (size_t i = 0; i < dense_x_elements; i++)
        dense_x[i] = lcg_float(0.1f);
    ds4_gpu_tensor *dense_x_t =
        ds4_gpu_tensor_alloc(dense_x_elements * sizeof(float));
    ds4_gpu_tensor *dense_out_t =
        ds4_gpu_tensor_alloc((size_t)rows * DENSE_OUT * sizeof(float));
    if (!dense_x_t || !dense_out_t) {
        fprintf(stderr, "qsa-prefill-bench: dense device allocation failed\n");
        return 1;
    }
    if (ds4_gpu_tensor_write(dense_x_t, 0, dense_x,
                             dense_x_elements * sizeof(float)) == 0) {
        fprintf(stderr, "qsa-prefill-bench: dense upload failed\n");
        return 1;
    }

    /* --- routed-expert tensors: unique top-k per row (stride 103 is
     * coprime with 512, so the ten selections per row stay distinct, which
     * the route-map hids slices rely on). --- */
    float *moe_x = malloc(dense_x_elements * sizeof(float));
    int32_t *moe_sel = malloc((size_t)rows * MOE_TOP_K * sizeof(int32_t));
    float *moe_route = malloc((size_t)rows * MOE_TOP_K * sizeof(float));
    if (!moe_x || !moe_sel || !moe_route) {
        fprintf(stderr, "qsa-prefill-bench: moe host allocation failed\n");
        return 1;
    }
    for (size_t i = 0; i < dense_x_elements; i++)
        moe_x[i] = lcg_float(0.1f);
    /* Two route patterns: the uniform lattice spreads every expert's rows
     * over the whole chunk (worst-case gather locality, worst-case 32-row
     * padding at small tiles); the block pattern gives each 32-row group a
     * dedicated expert set so each expert's gathered rows are contiguous,
     * isolating the MMA's B-side locality from the route distribution. */
    const int block_route = getenv("DS4_BENCH_MOE_ROUTE") != NULL &&
        strcmp(getenv("DS4_BENCH_MOE_ROUTE"), "block") == 0;
    for (uint32_t row = 0; row < rows; row++) {
        for (uint32_t slot = 0; slot < MOE_TOP_K; slot++) {
            moe_sel[(size_t)row * MOE_TOP_K + slot] = block_route
                ? (int32_t)((((row / 32u) * MOE_TOP_K + slot) % MOE_EXPERTS))
                : (int32_t)((row * 7u + slot * 103u) % MOE_EXPERTS);
            moe_route[(size_t)row * MOE_TOP_K + slot] =
                (float)(slot + 1u) / 55.0f;
        }
    }
    ds4_gpu_tensor *moe_x_t =
        ds4_gpu_tensor_alloc(dense_x_elements * sizeof(float));
    ds4_gpu_tensor *moe_sel_t =
        ds4_gpu_tensor_alloc((size_t)rows * MOE_TOP_K * sizeof(int32_t));
    ds4_gpu_tensor *moe_route_t =
        ds4_gpu_tensor_alloc((size_t)rows * MOE_TOP_K * sizeof(float));
    ds4_gpu_tensor *moe_mid_t = ds4_gpu_tensor_alloc(
        (size_t)rows * MOE_TOP_K * MOE_FF * sizeof(float));
    ds4_gpu_tensor *moe_out_t =
        ds4_gpu_tensor_alloc((size_t)rows * DENSE_IN * sizeof(float));
    if (!moe_x_t || !moe_sel_t || !moe_route_t || !moe_mid_t || !moe_out_t) {
        fprintf(stderr, "qsa-prefill-bench: moe device allocation failed\n");
        return 1;
    }
    if (ds4_gpu_tensor_write(moe_x_t, 0, moe_x,
                             dense_x_elements * sizeof(float)) == 0 ||
        ds4_gpu_tensor_write(moe_sel_t, 0, moe_sel,
                             (size_t)rows * MOE_TOP_K * sizeof(int32_t)) ==
            0 ||
        ds4_gpu_tensor_write(moe_route_t, 0, moe_route,
                             (size_t)rows * MOE_TOP_K * sizeof(float)) == 0) {
        fprintf(stderr, "qsa-prefill-bench: moe upload failed\n");
        return 1;
    }

    enum { WARMUP = 2, SAMPLES = 5 };
    double attn_samples[SAMPLES];
    double topk_samples[SAMPLES];
    double gdn_samples[SAMPLES];
    double dense_samples[SAMPLES];
    double moe_samples[SAMPLES];
    for (int iter = 0; iter < WARMUP + SAMPLES; iter++) {
        const int sample = iter - WARMUP;

        if (ds4_gpu_synchronize() == 0) {
            fprintf(stderr, "qsa-prefill-bench: sync failed\n");
            return 1;
        }
        double t0 = now_ms();
        if (ds4_gpu_qwen4_qsa_attention_bf16(
                out_t, q_t, gate_t, key_t, value_t, selected_t, counts_t,
                visible_t, rows, MAX_ROWS, Q_HEADS, KV_HEADS, HEAD_DIM,
                BLOCK_TOPK, RATIO) == 0) {
            fprintf(stderr, "qsa-prefill-bench: attention dispatch failed\n");
            return 1;
        }
        if (ds4_gpu_synchronize() == 0) {
            fprintf(stderr, "qsa-prefill-bench: sync failed\n");
            return 1;
        }
        if (sample >= 0) attn_samples[sample] = now_ms() - t0;

        /* One full pass of 512-query microtiles, mirroring the production
         * loop (shared score-sheet scratch, per-tile views). */
        t0 = now_ms();
        for (uint32_t start = 0; start < rows; start += MICROTILE) {
            const uint32_t tile =
                rows - start < MICROTILE ? rows - start : MICROTILE;
            ds4_gpu_tensor *q_view = ds4_gpu_tensor_view(
                idx_q_t, (uint64_t)start * IDX_HEADS * IDX_DIM * sizeof(float),
                (uint64_t)tile * IDX_HEADS * IDX_DIM * sizeof(float));
            ds4_gpu_tensor *score_view = ds4_gpu_tensor_view(
                topk_scores_t, (uint64_t)start * BLOCK_TOPK * sizeof(float),
                (uint64_t)tile * BLOCK_TOPK * sizeof(float));
            ds4_gpu_tensor *index_view = ds4_gpu_tensor_view(
                topk_indices_t,
                (uint64_t)start * BLOCK_TOPK * sizeof(uint32_t),
                (uint64_t)tile * BLOCK_TOPK * sizeof(uint32_t));
            ds4_gpu_tensor *count_view = ds4_gpu_tensor_view(
                counts_t, (uint64_t)start * sizeof(uint32_t),
                (uint64_t)tile * sizeof(uint32_t));
            ds4_gpu_tensor *visible_view = ds4_gpu_tensor_view(
                visible_blocks_t, (uint64_t)start * sizeof(uint32_t),
                (uint64_t)tile * sizeof(uint32_t));
            if (!q_view || !score_view || !index_view || !count_view ||
                !visible_view ||
                ds4_gpu_qwen4_qsa_stream_topk_bf16(
                    score_view, index_view, count_view, topk_tile_t, q_view,
                    pooled_k_t, visible_view, tile, visible_geometry,
                    IDX_HEADS, IDX_DIM, BLOCK_TOPK, BLOCK_TILE) == 0) {
                fprintf(stderr,
                        "qsa-prefill-bench: stream topk dispatch failed\n");
                return 1;
            }
            ds4_gpu_tensor_free(visible_view);
            ds4_gpu_tensor_free(count_view);
            ds4_gpu_tensor_free(index_view);
            ds4_gpu_tensor_free(score_view);
            ds4_gpu_tensor_free(q_view);
        }
        if (ds4_gpu_synchronize() == 0) {
            fprintf(stderr, "qsa-prefill-bench: sync failed\n");
            return 1;
        }
        if (sample >= 0) topk_samples[sample] = now_ms() - t0;

        t0 = now_ms();
        if (ds4_gpu_qwen4_gdn_prefill_bf16_state(
                gdn_out_t, gdn_state_t, gdn_q_t, gdn_k_t, gdn_v_t,
                gdn_decay_t, gdn_beta_t, NULL, rows, GDN_KEY_HEADS,
                GDN_VALUE_HEADS, GDN_HEAD_DIM, gdn_rows()) == 0) {
            fprintf(stderr, "qsa-prefill-bench: gdn dispatch failed\n");
            return 1;
        }
        if (ds4_gpu_synchronize() == 0) {
            fprintf(stderr, "qsa-prefill-bench: sync failed\n");
            return 1;
        }
        if (sample >= 0) gdn_samples[sample] = now_ms() - t0;

        t0 = now_ms();
        if (ds4_gpu_matmul_q8_0_tensor(
                dense_out_t, dense_map, model_total, 0u, DENSE_IN,
                DENSE_OUT, dense_x_t, rows) == 0) {
            fprintf(stderr, "qsa-prefill-bench: dense dispatch failed\n");
            return 1;
        }
        if (ds4_gpu_synchronize() == 0) {
            fprintf(stderr, "qsa-prefill-bench: sync failed\n");
            return 1;
        }
        if (sample >= 0) dense_samples[sample] = now_ms() - t0;

        t0 = now_ms();
        if (ds4_gpu_qwen4_moe_q4_0_model(
                moe_out_t, moe_mid_t, moe_x_t, moe_sel_t, moe_route_t,
                dense_map, model_total, moe_gate_offset, moe_up_offset,
                moe_down_offset, DENSE_IN, MOE_FF, DENSE_IN, MOE_EXPERTS,
                MOE_TOP_K, rows) == 0) {
            fprintf(stderr, "qsa-prefill-bench: moe dispatch failed\n");
            return 1;
        }
        if (ds4_gpu_synchronize() == 0) {
            fprintf(stderr, "qsa-prefill-bench: sync failed\n");
            return 1;
        }
        if (sample >= 0) moe_samples[sample] = now_ms() - t0;
    }
    if (ds4_gpu_tensor_read(out_t, 0, out, q_elements * sizeof(float)) == 0) {
        fprintf(stderr, "qsa-prefill-bench: output readback failed\n");
        return 1;
    }

    const double attn_ms = attn_samples[median_of(attn_samples, SAMPLES)];
    const double topk_ms = topk_samples[median_of(topk_samples, SAMPLES)];
    const double gdn_ms = gdn_samples[median_of(gdn_samples, SAMPLES)];
    const double dense_ms = dense_samples[median_of(dense_samples, SAMPLES)];
    const double moe_ms = moe_samples[median_of(moe_samples, SAMPLES)];
    const char *tile_env = getenv("DS4_QWEN4_MOE_MUL_MM_TILE_ROWS");
    /* 12 of the 48 layers are QSA layers; 36 are Gated DeltaNet layers;
     * every layer runs the routed-expert MoE block. */
    printf("qsa-prefill-bench rows=%u blocks=%u visible_blocks=%u "
           "selected_per_query=%u moe_tile_rows=%s\n",
           rows, blocks, visible_geometry, selected_blocks * RATIO,
           tile_env && tile_env[0] ? tile_env : "8192(default)");
    printf("  routed MoE Q4_0     %8.2f ms/layer   x48 = %8.1f ms/chunk\n",
           moe_ms, moe_ms * 48.0);
    printf("  gathered attention  %8.2f ms/layer   x12 = %8.1f ms/chunk\n",
           attn_ms, attn_ms * 12.0);
    printf("  streaming top-k     %8.2f ms/layer   x12 = %8.1f ms/chunk\n",
           topk_ms, topk_ms * 12.0);
    printf("  gdn recurrence R%u   %8.2f ms/layer   x36 = %8.1f ms/chunk\n",
           gdn_rows(), gdn_ms, gdn_ms * 36.0);
    printf("  dense q8 %ux%u %8.2f ms        x48 ~ %8.1f ms/chunk (per-proj)\n",
           DENSE_OUT, DENSE_IN, dense_ms, dense_ms * 48.0);
    printf("  K/V gather traffic/layer ~%.1f GB (ideal shared %.2f GB)\n",
           (double)rows * Q_HEADS * (selected_blocks * RATIO) * HEAD_DIM *
               2u * 2u / 1.0e9,
           (double)KV_HEADS * rows * HEAD_DIM * 2u * 2u / 1.0e9);
    return 0;
}
