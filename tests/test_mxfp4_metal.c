#define _DARWIN_C_SOURCE

#include "ds4_gpu.h"

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define MXFP4_TYPE 39u
#define QK_MXFP4 32u
#define N_TOTAL_EXPERT 8u
#define N_EXPERT 6u
#define DIM 256u
#define BATCH_TOKENS 48u
#define ATTN_GROUPS 8u
#define ATTN_GROUP_DIM 4096u
#define ATTN_RANK 1024u

typedef struct {
    uint8_t e;
    uint8_t qs[QK_MXFP4 / 2u];
} block_mxfp4;

typedef struct {
    _Float16 d;
    int8_t qs[32];
} block_q8_0;
_Static_assert(sizeof(block_q8_0) == 34u,
               "Q8_0 test fixture must match the Metal block stride");

static const float mxfp4_values[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
   -0.0f,-0.5f,-1.0f,-1.5f,-2.0f,-3.0f,-4.0f,-6.0f,
};

bool ds4_log_is_tty(FILE *fp) {
    (void)fp;
    return false;
}

static uint64_t align_up(uint64_t value, uint64_t alignment) {
    return (value + alignment - 1u) / alignment * alignment;
}

typedef struct {
    const char *name;
    char *value;
    bool had_value;
} saved_env;

static int save_env(saved_env *saved, const char *name) {
    const char *value = getenv(name);
    saved->name = name;
    saved->had_value = value != NULL;
    saved->value = value ? strdup(value) : NULL;
    return !value || saved->value != NULL;
}

static int restore_env(saved_env *saved) {
    int rc = 0;
    if (saved->had_value) {
        if (!saved->value) return 0;
        rc = setenv(saved->name, saved->value, 1);
    } else {
        rc = unsetenv(saved->name);
    }
    free(saved->value);
    saved->value = NULL;
    return rc == 0;
}

static float e8m0_to_f32(uint8_t e) {
    uint32_t bits = e == 0 ? 0x00400000u : (uint32_t)e << 23u;
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static float dot_mxfp4(const block_mxfp4 *row, const float *x) {
    float sum = 0.0f;
    for (uint32_t block = 0; block < DIM / QK_MXFP4; block++) {
        const block_mxfp4 *b = row + block;
        const float scale = e8m0_to_f32(b->e);
        for (uint32_t i = 0; i < QK_MXFP4 / 2u; i++) {
            const uint8_t q = b->qs[i];
            sum += scale * mxfp4_values[q & 15u] *
                   x[block * QK_MXFP4 + i];
            sum += scale * mxfp4_values[q >> 4u] *
                   x[block * QK_MXFP4 + i + QK_MXFP4 / 2u];
        }
    }
    return sum;
}

static void fill_matrix(block_mxfp4 *matrix, uint32_t salt) {
    const uint32_t blocks_per_row = DIM / QK_MXFP4;
    for (uint32_t expert = 0; expert < N_TOTAL_EXPERT; expert++) {
        for (uint32_t row = 0; row < DIM; row++) {
            block_mxfp4 *blocks = matrix +
                ((uint64_t)expert * DIM + row) * blocks_per_row;
            for (uint32_t block = 0; block < blocks_per_row; block++) {
                block_mxfp4 *b = blocks + block;
                b->e = (uint8_t)(120u +
                    ((salt + expert * 3u + row + block * 5u) % 6u));
                for (uint32_t i = 0; i < QK_MXFP4 / 2u; i++) {
                    const uint8_t lo = (uint8_t)(
                        (salt + expert * 7u + row * 3u + block + i) & 15u);
                    const uint8_t hi = (uint8_t)(
                        (salt * 3u + expert + row * 5u + block * 7u + i * 3u) & 15u);
                    b->qs[i] = (uint8_t)(lo | (hi << 4u));
                }
            }
        }
    }
}

static int compare_values(const char *name, const float *actual,
                          const float *expected, uint64_t count,
                          float tolerance) {
    float max_abs = 0.0f;
    double sum_sq = 0.0;
    uint64_t max_index = 0;
    for (uint64_t i = 0; i < count; i++) {
        if (!isfinite(actual[i]) || !isfinite(expected[i])) {
            fprintf(stderr, "MXFP4 Metal %s non-finite value at %llu\n",
                    name, (unsigned long long)i);
            return 0;
        }
        const float error = fabsf(actual[i] - expected[i]);
        if (error > max_abs) {
            max_abs = error;
            max_index = i;
        }
        sum_sq += (double)error * error;
    }
    const double rms = sqrt(sum_sq / (double)count);
    fprintf(stderr,
            "MXFP4 Metal %-4s max_abs=%g rms=%g at=%llu\n",
            name, max_abs, rms, (unsigned long long)max_index);
    return max_abs <= tolerance;
}

int main(void) {
    const uint64_t page = (uint64_t)getpagesize();
    const uint64_t row_bytes =
        (DIM / QK_MXFP4) * sizeof(block_mxfp4);
    const uint64_t expert_bytes = DIM * row_bytes;
    const uint64_t tensor_bytes = N_TOTAL_EXPERT * expert_bytes;
    const uint64_t gate_offset = 0;
    const uint64_t up_offset = align_up(tensor_bytes, page);
    const uint64_t down_offset = align_up(up_offset + tensor_bytes, page);
    const uint64_t attn_row_bytes =
        (ATTN_GROUP_DIM / 32u) * sizeof(block_q8_0);
    const uint64_t attn_weight_bytes =
        (uint64_t)ATTN_GROUPS * ATTN_RANK * attn_row_bytes;
    const uint64_t attn_offset =
        align_up(down_offset + tensor_bytes, page);
    const uint64_t model_size =
        align_up(attn_offset + attn_weight_bytes, page);
    void *model = NULL;
    if (posix_memalign(&model, (size_t)page, (size_t)model_size) != 0) {
        fprintf(stderr, "MXFP4 Metal test model allocation failed\n");
        return 1;
    }
    memset(model, 0, (size_t)model_size);
    fill_matrix((block_mxfp4 *)((uint8_t *)model + gate_offset), 1u);
    fill_matrix((block_mxfp4 *)((uint8_t *)model + up_offset), 5u);
    fill_matrix((block_mxfp4 *)((uint8_t *)model + down_offset), 9u);
    block_q8_0 *attn_matrix =
        (block_q8_0 *)((uint8_t *)model + attn_offset);
    const uint64_t attn_blocks =
        (uint64_t)ATTN_GROUPS * ATTN_RANK * (ATTN_GROUP_DIM / 32u);
    for (uint64_t block = 0; block < attn_blocks; block++) {
        attn_matrix[block].d = (_Float16)(1.0f / 128.0f);
        for (uint32_t i = 0; i < 32u; i++) {
            attn_matrix[block].qs[i] =
                (int8_t)((int32_t)((block * 11u + i * 7u) % 31u) - 15);
        }
    }

    float x[DIM];
    int32_t selected[N_EXPERT] = { 0, 2, 3, 5, 6, 7 };
    float weights[N_EXPERT] = { 0.24f, 0.20f, 0.18f, 0.16f, 0.12f, 0.10f };
    for (uint32_t i = 0; i < DIM; i++) {
        x[i] = (float)((int32_t)((i * 13u) % 31u) - 15) / 64.0f;
    }

    const uint64_t pair_count = (uint64_t)N_EXPERT * DIM;
    float *gate_ref = calloc((size_t)pair_count, sizeof(float));
    float *up_ref = calloc((size_t)pair_count, sizeof(float));
    float *mid_ref = calloc((size_t)pair_count, sizeof(float));
    float *out_ref = calloc(DIM, sizeof(float));
    float *gate_gpu = calloc((size_t)pair_count, sizeof(float));
    float *up_gpu = calloc((size_t)pair_count, sizeof(float));
    float *mid_gpu = calloc((size_t)pair_count, sizeof(float));
    float *experts_gpu = calloc((size_t)pair_count, sizeof(float));
    float *out_gpu = calloc(DIM, sizeof(float));
    float *gate_fast = calloc((size_t)pair_count, sizeof(float));
    float *up_fast = calloc((size_t)pair_count, sizeof(float));
    float *mid_fast = calloc((size_t)pair_count, sizeof(float));
    float *experts_fast = calloc((size_t)pair_count, sizeof(float));
    float *out_fast = calloc(DIM, sizeof(float));
    if (!gate_ref || !up_ref || !mid_ref || !out_ref ||
        !gate_gpu || !up_gpu || !mid_gpu || !experts_gpu || !out_gpu ||
        !gate_fast || !up_fast || !mid_fast || !experts_fast || !out_fast) {
        fprintf(stderr, "MXFP4 Metal host tensor allocation failed\n");
        return 1;
    }

    const block_mxfp4 *gate_matrix =
        (const block_mxfp4 *)((const uint8_t *)model + gate_offset);
    const block_mxfp4 *up_matrix =
        (const block_mxfp4 *)((const uint8_t *)model + up_offset);
    const block_mxfp4 *down_matrix =
        (const block_mxfp4 *)((const uint8_t *)model + down_offset);
    const uint64_t blocks_per_expert = expert_bytes / sizeof(block_mxfp4);
    const uint64_t blocks_per_row = row_bytes / sizeof(block_mxfp4);
    for (uint32_t slot = 0; slot < N_EXPERT; slot++) {
        const uint32_t expert = (uint32_t)selected[slot];
        for (uint32_t row = 0; row < DIM; row++) {
            const uint64_t pair = (uint64_t)slot * DIM + row;
            gate_ref[pair] = dot_mxfp4(
                gate_matrix + (uint64_t)expert * blocks_per_expert +
                    (uint64_t)row * blocks_per_row,
                x);
            up_ref[pair] = dot_mxfp4(
                up_matrix + (uint64_t)expert * blocks_per_expert +
                    (uint64_t)row * blocks_per_row,
                x);
            const float g = fminf(gate_ref[pair], 7.0f);
            const float u = fmaxf(-7.0f, fminf(up_ref[pair], 7.0f));
            mid_ref[pair] = (g / (1.0f + expf(-g))) * u * weights[slot];
        }
    }
    for (uint32_t row = 0; row < DIM; row++) {
        for (uint32_t slot = 0; slot < N_EXPERT; slot++) {
            const uint32_t expert = (uint32_t)selected[slot];
            out_ref[row] += dot_mxfp4(
                down_matrix + (uint64_t)expert * blocks_per_expert +
                    (uint64_t)row * blocks_per_row,
                mid_ref + (uint64_t)slot * DIM);
        }
    }

    int ok = ds4_gpu_init() && ds4_gpu_set_model_map(model, model_size);
    ok = ok && ds4_gpu_test_decode_pipeline_fast_lookup();
    if (ok) {
        fprintf(stderr,
                "MXFP4 Metal decode pipeline fast lookup inactive/populate/hit guards PASS\n");
    }
    ok = ok && ds4_gpu_test_decode_pipeline_fast_lookup_ext();
    if (ok) {
        fprintf(stderr,
                "MXFP4 Metal decode pipeline fast lookup ext (nsg+nxpsg) guards PASS\n");
    }
    uint16_t legacy_half_bits[256u * 16u];
    uint16_t lut_half_bits[256u * 16u];
    ok = ok && ds4_gpu_test_mxfp4_down_half_lut(
        legacy_half_bits, lut_half_bits);
    if (ok) {
        uint32_t mismatches = 0;
        for (uint32_t i = 0; i < 256u * 16u; i++) {
            if (legacy_half_bits[i] != lut_half_bits[i]) {
                if (mismatches < 16u) {
                    fprintf(stderr,
                            "MXFP4 Metal half-LUT raw-bit mismatch e=%u q=%u "
                            "legacy=0x%04x lut=0x%04x\n",
                            i >> 4u, i & 15u,
                            legacy_half_bits[i], lut_half_bits[i]);
                }
                mismatches++;
            }
        }
        if (mismatches != 0u) {
            fprintf(stderr,
                    "MXFP4 Metal half-LUT raw-bit mismatches=%u/4096\n",
                    mismatches);
            ok = 0;
        }
    }
    if (ok) {
        fprintf(stderr,
                "MXFP4 Metal half-LUT raw-bit A/B exact for all 4096 e/q pairs\n");
    }
    ds4_gpu_set_quality(false);
    ds4_gpu_set_ssd_streaming(false);
    ds4_gpu_tensor *x_tensor = ds4_gpu_tensor_alloc(sizeof(x));
    ds4_gpu_tensor *selected_tensor = ds4_gpu_tensor_alloc(sizeof(selected));
    ds4_gpu_tensor *weights_tensor = ds4_gpu_tensor_alloc(sizeof(weights));
    ds4_gpu_tensor *gate_tensor = ds4_gpu_tensor_alloc(pair_count * sizeof(float));
    ds4_gpu_tensor *up_tensor = ds4_gpu_tensor_alloc(pair_count * sizeof(float));
    ds4_gpu_tensor *mid_tensor = ds4_gpu_tensor_alloc(pair_count * sizeof(float));
    ds4_gpu_tensor *experts_tensor = ds4_gpu_tensor_alloc(pair_count * sizeof(float));
    ds4_gpu_tensor *out_tensor = ds4_gpu_tensor_alloc(DIM * sizeof(float));
    ok = ok && x_tensor && selected_tensor && weights_tensor && gate_tensor &&
         up_tensor && mid_tensor && experts_tensor && out_tensor;
    ok = ok && ds4_gpu_tensor_write(x_tensor, 0, x, sizeof(x));
    ok = ok && ds4_gpu_tensor_write(
        selected_tensor, 0, selected, sizeof(selected));
    ok = ok && ds4_gpu_tensor_write(
        weights_tensor, 0, weights, sizeof(weights));
    ok = ok && ds4_gpu_tensor_fill_f32(
        gate_tensor, -101.0f, pair_count);
    ok = ok && ds4_gpu_tensor_fill_f32(
        up_tensor, -102.0f, pair_count);
    ok = ok && ds4_gpu_tensor_fill_f32(
        mid_tensor, -103.0f, pair_count);
    ok = ok && ds4_gpu_tensor_fill_f32(
        experts_tensor, -104.0f, pair_count);
    ok = ok && ds4_gpu_tensor_fill_f32(out_tensor, -105.0f, DIM);
    ok = ok && ds4_gpu_routed_moe_one_tensor(
        out_tensor, gate_tensor, up_tensor, mid_tensor, experts_tensor,
        model, model_size, gate_offset, up_offset, down_offset,
        MXFP4_TYPE, MXFP4_TYPE, expert_bytes, row_bytes,
        expert_bytes, row_bytes, DIM, DIM, DIM,
        selected_tensor, weights_tensor, N_TOTAL_EXPERT, N_EXPERT,
        7.0f, x_tensor, NULL, 0u, true);
    ok = ok && ds4_gpu_tensor_read(
        gate_tensor, 0, gate_gpu, pair_count * sizeof(float));
    ok = ok && ds4_gpu_tensor_read(
        up_tensor, 0, up_gpu, pair_count * sizeof(float));
    ok = ok && ds4_gpu_tensor_read(
        mid_tensor, 0, mid_gpu, pair_count * sizeof(float));
    ok = ok && ds4_gpu_tensor_read(
        experts_tensor, 0, experts_gpu, pair_count * sizeof(float));
    ok = ok && ds4_gpu_tensor_read(
        out_tensor, 0, out_gpu, DIM * sizeof(float));

    /* The routed-MXFP4 helper uses its precompiled pipeline globals, so it is
     * deliberately neutral to the fast mirror. Toggle the per-token hint
     * around two active runs and one inactive guard-tail run to prove unrelated
     * decode dispatches stay byte-exact. Poison and compare every writable
     * tensor byte-for-byte on all three runs. */
    bool fast_lookup_ok = ok;
    const int previous_fast_lookup =
        ds4_gpu_set_decode_pipeline_fast_lookup(1);
    for (uint32_t run = 0; run < 3u && fast_lookup_ok; run++) {
        if (run == 2u) {
            (void)ds4_gpu_set_decode_pipeline_fast_lookup(0);
        }
        fast_lookup_ok =
            ds4_gpu_tensor_fill_f32(gate_tensor, -101.0f, pair_count) &&
            ds4_gpu_tensor_fill_f32(up_tensor, -102.0f, pair_count) &&
            ds4_gpu_tensor_fill_f32(mid_tensor, -103.0f, pair_count) &&
            ds4_gpu_tensor_fill_f32(experts_tensor, -104.0f, pair_count) &&
            ds4_gpu_tensor_fill_f32(out_tensor, -105.0f, DIM) &&
            ds4_gpu_routed_moe_one_tensor(
                out_tensor, gate_tensor, up_tensor, mid_tensor, experts_tensor,
                model, model_size, gate_offset, up_offset, down_offset,
                MXFP4_TYPE, MXFP4_TYPE, expert_bytes, row_bytes,
                expert_bytes, row_bytes, DIM, DIM, DIM,
                selected_tensor, weights_tensor, N_TOTAL_EXPERT, N_EXPERT,
                7.0f, x_tensor, NULL, 0u, true) &&
            ds4_gpu_tensor_read(
                gate_tensor, 0, gate_fast, pair_count * sizeof(float)) &&
            ds4_gpu_tensor_read(
                up_tensor, 0, up_fast, pair_count * sizeof(float)) &&
            ds4_gpu_tensor_read(
                mid_tensor, 0, mid_fast, pair_count * sizeof(float)) &&
            ds4_gpu_tensor_read(
                experts_tensor, 0, experts_fast,
                pair_count * sizeof(float)) &&
            ds4_gpu_tensor_read(
                out_tensor, 0, out_fast, DIM * sizeof(float));
        const bool gate_exact =
            memcmp(gate_fast, gate_gpu, pair_count * sizeof(float)) == 0;
        const bool up_exact =
            memcmp(up_fast, up_gpu, pair_count * sizeof(float)) == 0;
        const bool mid_exact =
            memcmp(mid_fast, mid_gpu, pair_count * sizeof(float)) == 0;
        const bool experts_exact =
            memcmp(experts_fast, experts_gpu,
                   pair_count * sizeof(float)) == 0;
        const bool out_exact =
            memcmp(out_fast, out_gpu, DIM * sizeof(float)) == 0;
        if (fast_lookup_ok &&
            (!gate_exact || !up_exact || !mid_exact ||
             !experts_exact || !out_exact)) {
            fprintf(stderr,
                    "MXFP4 Metal decode pipeline fast lookup byte-exact A/B mismatch "
                    "run=%u gate=%d up=%d mid=%d experts=%d out=%d\n",
                    run + 1u, gate_exact, up_exact, mid_exact,
                    experts_exact, out_exact);
            fast_lookup_ok = false;
        } else if (fast_lookup_ok) {
            fprintf(stderr,
                    "MXFP4 Metal decode pipeline fast lookup byte-exact A/B PASS run=%u mode=%s\n",
                    run + 1u, run < 2u ? "active" : "inactive");
        }
    }
    (void)ds4_gpu_set_decode_pipeline_fast_lookup(previous_fast_lookup);
    ok = ok && fast_lookup_ok;

    if (ok) {
        ok = compare_values("gate", gate_gpu, gate_ref, pair_count, 2.0e-5f) &&
             compare_values("up", up_gpu, up_ref, pair_count, 2.0e-5f) &&
             compare_values("mid", mid_gpu, mid_ref, pair_count, 2.0e-5f) &&
             compare_values("out", out_gpu, out_ref, DIM, 2.0e-4f);
    }

    /* Compare the exact fixed-shape attention-output LOW kernel directly
     * against its generic rollback. Force the static PSO through the test
     * flag so this remains coverage on newer Apple GPUs, and poison the full
     * destination before each run to catch partial writes. */
    const uint64_t attn_heads_count =
        (uint64_t)ATTN_GROUPS * ATTN_GROUP_DIM;
    const uint64_t attn_low_count =
        (uint64_t)ATTN_GROUPS * ATTN_RANK;
    const size_t attn_heads_bytes =
        (size_t)attn_heads_count * sizeof(float);
    const size_t attn_low_bytes =
        (size_t)attn_low_count * sizeof(float);
    float *attn_heads = malloc(attn_heads_bytes);
    uint8_t *attn_poison = malloc(attn_low_bytes);
    uint8_t *attn_generic = malloc(attn_low_bytes);
    uint8_t *attn_static = malloc(attn_low_bytes);
    ds4_gpu_tensor *attn_heads_tensor =
        ds4_gpu_tensor_alloc(attn_heads_bytes);
    ds4_gpu_tensor *attn_low_tensor =
        ds4_gpu_tensor_alloc(attn_low_bytes);
    saved_env decode_ports_env = { 0 };
    saved_env attn_static_env = { 0 };
    bool attn_env_ok = save_env(
        &decode_ports_env, "DS4_METAL_DISABLE_PRE_M5_DECODE_PORTS") != 0;
    attn_env_ok = save_env(
        &attn_static_env,
        "DS4_METAL_DISABLE_PRE_M5_ATTN_OUT_LOW_Q8_STATIC") != 0 &&
        attn_env_ok;
    bool attn_exact = ok && attn_env_ok && attn_heads && attn_poison &&
        attn_generic && attn_static && attn_heads_tensor && attn_low_tensor;
    if (attn_heads) {
        for (uint64_t i = 0; i < attn_heads_count; i++) {
            attn_heads[i] =
                (float)((int32_t)((i * 17u) % 257u) - 128) / 256.0f;
        }
    }
    if (attn_poison) memset(attn_poison, 0xa5, attn_low_bytes);
    attn_exact = attn_exact && ds4_gpu_tensor_write(
        attn_heads_tensor, 0, attn_heads, attn_heads_bytes);

    if (attn_env_ok &&
        setenv("DS4_METAL_DISABLE_PRE_M5_ATTN_OUT_LOW_Q8_STATIC", "1", 1) != 0) {
        attn_exact = false;
    }
    ds4_gpu_test_set_flags(0);
    attn_exact = attn_exact && ds4_gpu_tensor_write(
        attn_low_tensor, 0, attn_poison, attn_low_bytes);
    attn_exact = attn_exact && ds4_gpu_attention_output_low_q8_tensor(
        attn_low_tensor, model, model_size, attn_offset,
        ATTN_GROUP_DIM, ATTN_RANK, ATTN_GROUPS,
        attn_heads_tensor);
    attn_exact = attn_exact && ds4_gpu_tensor_read(
        attn_low_tensor, 0, attn_generic, attn_low_bytes);

    if (attn_env_ok &&
        (unsetenv("DS4_METAL_DISABLE_PRE_M5_DECODE_PORTS") != 0 ||
         unsetenv("DS4_METAL_DISABLE_PRE_M5_ATTN_OUT_LOW_Q8_STATIC") != 0)) {
        attn_exact = false;
    }
    ds4_gpu_test_set_flags(DS4_GPU_TEST_ATTN_OUT_LOW_Q8_STATIC);
    attn_exact = attn_exact && ds4_gpu_tensor_write(
        attn_low_tensor, 0, attn_poison, attn_low_bytes);
    attn_exact = attn_exact && ds4_gpu_attention_output_low_q8_tensor(
        attn_low_tensor, model, model_size, attn_offset,
        ATTN_GROUP_DIM, ATTN_RANK, ATTN_GROUPS,
        attn_heads_tensor);
    attn_exact = attn_exact && ds4_gpu_tensor_read(
        attn_low_tensor, 0, attn_static, attn_low_bytes);
    ds4_gpu_test_set_flags(0);

    if (attn_exact && memcmp(attn_generic, attn_static, attn_low_bytes) != 0) {
        fprintf(stderr,
                "MXFP4 Metal attention-output LOW static/generic A/B mismatch\n");
        attn_exact = false;
    } else if (attn_exact) {
        fprintf(stderr,
                "MXFP4 Metal attention-output LOW static/generic A/B exact\n");
    }
    const bool attn_static_env_restored = restore_env(&attn_static_env) != 0;
    const bool decode_ports_env_restored = restore_env(&decode_ports_env) != 0;
    ok = ok && attn_exact &&
        attn_static_env_restored && decode_ports_env_restored;
    ds4_gpu_tensor_free(attn_low_tensor);
    ds4_gpu_tensor_free(attn_heads_tensor);
    free(attn_static);
    free(attn_generic);
    free(attn_poison);
    free(attn_heads);

    /* The production large-prefill path stores its fused SwiGLU result as
     * FP16 before the down projection. Compare that independent grouped-MMA
     * path against the same scalar reference with the documented rounding. */
    const uint64_t batch_pairs = (uint64_t)BATCH_TOKENS * pair_count;
    const uint64_t batch_out_count = (uint64_t)BATCH_TOKENS * DIM;
    float *x_batch = calloc((size_t)BATCH_TOKENS * DIM, sizeof(float));
    int32_t *selected_batch = calloc(
        (size_t)BATCH_TOKENS * N_EXPERT, sizeof(int32_t));
    float *weights_batch = calloc(
        (size_t)BATCH_TOKENS * N_EXPERT, sizeof(float));
    float *mid_batch_expected = calloc((size_t)batch_pairs, sizeof(float));
    float *mid_batch_actual = calloc((size_t)batch_pairs, sizeof(float));
    _Float16 *mid_batch_baseline = calloc(
        (size_t)batch_pairs, sizeof(_Float16));
    _Float16 *mid_batch_storage = calloc(
        (size_t)batch_pairs, sizeof(_Float16));
    _Float16 *mid_batch_half_lut_baseline = calloc(
        (size_t)batch_pairs, sizeof(_Float16));
    float *out_batch_expected = calloc((size_t)batch_out_count, sizeof(float));
    float *out_batch_baseline = calloc(
        (size_t)batch_out_count, sizeof(float));
    float *out_batch_half_lut_baseline = calloc(
        (size_t)batch_out_count, sizeof(float));
    float *out_batch_actual = calloc((size_t)batch_out_count, sizeof(float));
    float *experts_batch_half_lut_baseline = calloc(
        (size_t)batch_out_count * N_EXPERT, sizeof(float));
    float *experts_batch_actual = calloc(
        (size_t)batch_out_count * N_EXPERT, sizeof(float));
    uint8_t *batch_poison = malloc(
        (size_t)batch_out_count * N_EXPERT * sizeof(float));
    float mid_half_ref[N_EXPERT * DIM];
    float out_half_ref[DIM];
    memset(out_half_ref, 0, sizeof(out_half_ref));
    for (uint64_t pair = 0; pair < pair_count; pair++) {
        mid_half_ref[pair] = (float)(_Float16)mid_ref[pair];
    }
    for (uint32_t row = 0; row < DIM; row++) {
        for (uint32_t slot = 0; slot < N_EXPERT; slot++) {
            const uint32_t expert = (uint32_t)selected[slot];
            out_half_ref[row] += dot_mxfp4(
                down_matrix + (uint64_t)expert * blocks_per_expert +
                    (uint64_t)row * blocks_per_row,
                mid_half_ref + (uint64_t)slot * DIM);
        }
    }
    for (uint32_t token = 0; token < BATCH_TOKENS; token++) {
        memcpy(x_batch + (uint64_t)token * DIM, x, sizeof(x));
        memcpy(selected_batch + (uint64_t)token * N_EXPERT,
               selected, sizeof(selected));
        memcpy(weights_batch + (uint64_t)token * N_EXPERT,
               weights, sizeof(weights));
        memcpy(mid_batch_expected + (uint64_t)token * pair_count,
               mid_half_ref, sizeof(mid_half_ref));
        memcpy(out_batch_expected + (uint64_t)token * DIM,
               out_half_ref, sizeof(out_half_ref));
    }

    ds4_gpu_tensor *x_batch_tensor = ds4_gpu_tensor_alloc(
        (uint64_t)BATCH_TOKENS * sizeof(x));
    ds4_gpu_tensor *selected_batch_tensor = ds4_gpu_tensor_alloc(
        (uint64_t)BATCH_TOKENS * sizeof(selected));
    ds4_gpu_tensor *weights_batch_tensor = ds4_gpu_tensor_alloc(
        (uint64_t)BATCH_TOKENS * sizeof(weights));
    ds4_gpu_tensor *gate_batch_tensor = ds4_gpu_tensor_alloc(
        batch_pairs * sizeof(float));
    ds4_gpu_tensor *up_batch_tensor = ds4_gpu_tensor_alloc(
        batch_pairs * sizeof(float));
    ds4_gpu_tensor *mid_batch_tensor = ds4_gpu_tensor_alloc(
        batch_pairs * sizeof(float));
    ds4_gpu_tensor *experts_batch_tensor = ds4_gpu_tensor_alloc(
        batch_out_count * N_EXPERT * sizeof(float));
    ds4_gpu_tensor *out_batch_tensor = ds4_gpu_tensor_alloc(
        batch_out_count * sizeof(float));
    bool mid_is_f16 = false;
    ok = ok && x_batch && selected_batch && weights_batch &&
         mid_batch_expected && mid_batch_actual && mid_batch_baseline &&
         mid_batch_storage && mid_batch_half_lut_baseline &&
         out_batch_expected && out_batch_baseline &&
         out_batch_half_lut_baseline && out_batch_actual &&
         experts_batch_half_lut_baseline && experts_batch_actual &&
         batch_poison &&
         x_batch_tensor && selected_batch_tensor && weights_batch_tensor &&
         gate_batch_tensor && up_batch_tensor && mid_batch_tensor &&
         experts_batch_tensor && out_batch_tensor;
    ok = ok && ds4_gpu_tensor_write(
        x_batch_tensor, 0, x_batch, BATCH_TOKENS * sizeof(x));
    ok = ok && ds4_gpu_tensor_write(
        selected_batch_tensor, 0, selected_batch,
        BATCH_TOKENS * sizeof(selected));
    ok = ok && ds4_gpu_tensor_write(
        weights_batch_tensor, 0, weights_batch,
        BATCH_TOKENS * sizeof(weights));
    if (batch_poison) {
        memset(batch_poison, 0xa5,
               (size_t)batch_out_count * N_EXPERT * sizeof(float));
    }
    /* Keep the established per-feature A/B checks isolated now that the
     * production defaults also cover this 48-token shape. The widened
     * dispatcher gets its own aggregate rollback comparison below. */
    const char *small_prefill_rollback_name =
        "DS4_METAL_DISABLE_PRE_M5_MXFP4_MOE_SMALL_PREFILL";
    const char *small_prefill_rollback_previous =
        getenv(small_prefill_rollback_name);
    char *small_prefill_rollback_saved = small_prefill_rollback_previous ?
        strdup(small_prefill_rollback_previous) : NULL;
    if ((small_prefill_rollback_previous && !small_prefill_rollback_saved) ||
        setenv(small_prefill_rollback_name, "1", 1) != 0) {
        fprintf(stderr,
                "MXFP4 Metal could not isolate short-prefill feature tests\n");
        ok = 0;
    }
    /* With 48 identical routes, every occupied expert has a 32-row tile plus
     * a 16-row tail tile. First capture the generic rollback as the baseline
     * for the independent pair, compact-pair, and down-tail checks. */
    ds4_gpu_test_set_flags(0);
    ok = ok && ds4_gpu_tensor_write(
        mid_batch_tensor, 0, batch_poison,
        batch_pairs * sizeof(_Float16));
    ok = ok && ds4_gpu_tensor_write(
        experts_batch_tensor, 0, batch_poison,
        batch_out_count * N_EXPERT * sizeof(float));
    ok = ok && ds4_gpu_tensor_write(
        out_batch_tensor, 0, batch_poison,
        batch_out_count * sizeof(float));
    ok = ok && ds4_gpu_routed_moe_batch_tensor(
        out_batch_tensor, gate_batch_tensor, up_batch_tensor,
        mid_batch_tensor, experts_batch_tensor,
        model, model_size, gate_offset, up_offset, down_offset,
        MXFP4_TYPE, MXFP4_TYPE, expert_bytes, row_bytes,
        expert_bytes, row_bytes, DIM, DIM, DIM,
        selected_batch_tensor, weights_batch_tensor,
        N_TOTAL_EXPERT, N_EXPERT, 7.0f, x_batch_tensor,
        0u, BATCH_TOKENS, &mid_is_f16, false, true);
    ok = ok && mid_is_f16;
    ok = ok && ds4_gpu_tensor_read(
        mid_batch_tensor, 0, mid_batch_baseline,
        batch_pairs * sizeof(_Float16));
    ok = ok && ds4_gpu_tensor_read(
        out_batch_tensor, 0, out_batch_baseline,
        batch_out_count * sizeof(float));

    /* The half-LUT candidate retains the down-tail kernel, so give it a
     * separate down-tail baseline instead of conflating two features. */
    mid_is_f16 = false;
    ds4_gpu_test_set_flags(DS4_GPU_TEST_MXFP4_DOWN_TAIL_CULL);
    ok = ok && ds4_gpu_tensor_write(
        mid_batch_tensor, 0, batch_poison,
        batch_pairs * sizeof(_Float16));
    ok = ok && ds4_gpu_tensor_write(
        experts_batch_tensor, 0, batch_poison,
        batch_out_count * N_EXPERT * sizeof(float));
    ok = ok && ds4_gpu_tensor_write(
        out_batch_tensor, 0, batch_poison,
        batch_out_count * sizeof(float));
    ok = ok && ds4_gpu_routed_moe_batch_tensor(
        out_batch_tensor, gate_batch_tensor, up_batch_tensor,
        mid_batch_tensor, experts_batch_tensor,
        model, model_size, gate_offset, up_offset, down_offset,
        MXFP4_TYPE, MXFP4_TYPE, expert_bytes, row_bytes,
        expert_bytes, row_bytes, DIM, DIM, DIM,
        selected_batch_tensor, weights_batch_tensor,
        N_TOTAL_EXPERT, N_EXPERT, 7.0f, x_batch_tensor,
        0u, BATCH_TOKENS, &mid_is_f16, false, true);
    ok = ok && mid_is_f16;
    ok = ok && ds4_gpu_tensor_read(
        mid_batch_tensor, 0, mid_batch_half_lut_baseline,
        batch_pairs * sizeof(_Float16));
    ok = ok && ds4_gpu_tensor_read(
        out_batch_tensor, 0, out_batch_half_lut_baseline,
        batch_out_count * sizeof(float));
    ok = ok && ds4_gpu_tensor_read(
        experts_batch_tensor, 0, experts_batch_half_lut_baseline,
        batch_out_count * N_EXPERT * sizeof(float));

    /* Force the exact half-result dequantization table only for the resident
     * MXFP4/F16-mid/single-rank down projection. Poison every writable output
     * and run twice in this process; the FP16 mid, expert-major F32 down
     * scratch, and summed F32 output must all match the established kernel
     * byte-for-byte on each repetition. */
    ds4_gpu_test_set_flags(
        DS4_GPU_TEST_MXFP4_DOWN_TAIL_CULL |
        DS4_GPU_TEST_MXFP4_DOWN_HALF_LUT);
    for (uint32_t run = 0; run < 2u && ok; run++) {
        mid_is_f16 = false;
        ok = ok && ds4_gpu_tensor_write(
            mid_batch_tensor, 0, batch_poison,
            batch_pairs * sizeof(_Float16));
        ok = ok && ds4_gpu_tensor_write(
            experts_batch_tensor, 0, batch_poison,
            batch_out_count * N_EXPERT * sizeof(float));
        ok = ok && ds4_gpu_tensor_write(
            out_batch_tensor, 0, batch_poison,
            batch_out_count * sizeof(float));
        ok = ok && ds4_gpu_routed_moe_batch_tensor(
            out_batch_tensor, gate_batch_tensor, up_batch_tensor,
            mid_batch_tensor, experts_batch_tensor,
            model, model_size, gate_offset, up_offset, down_offset,
            MXFP4_TYPE, MXFP4_TYPE, expert_bytes, row_bytes,
            expert_bytes, row_bytes, DIM, DIM, DIM,
            selected_batch_tensor, weights_batch_tensor,
            N_TOTAL_EXPERT, N_EXPERT, 7.0f, x_batch_tensor,
            0u, BATCH_TOKENS, &mid_is_f16, false, true);
        ok = ok && mid_is_f16;
        ok = ok && ds4_gpu_tensor_read(
            mid_batch_tensor, 0, mid_batch_storage,
            batch_pairs * sizeof(_Float16));
        ok = ok && ds4_gpu_tensor_read(
            experts_batch_tensor, 0, experts_batch_actual,
            batch_out_count * N_EXPERT * sizeof(float));
        ok = ok && ds4_gpu_tensor_read(
            out_batch_tensor, 0, out_batch_actual,
            batch_out_count * sizeof(float));
        if (ok &&
            (memcmp(mid_batch_storage, mid_batch_half_lut_baseline,
                    batch_pairs * sizeof(_Float16)) != 0 ||
             memcmp(experts_batch_actual, experts_batch_half_lut_baseline,
                    batch_out_count * N_EXPERT * sizeof(float)) != 0 ||
             memcmp(out_batch_actual, out_batch_half_lut_baseline,
                    batch_out_count * sizeof(float)) != 0)) {
            fprintf(stderr,
                    "MXFP4 Metal down half-LUT poisoned A/B mismatch on repetition %u\n",
                    run + 1u);
            ok = 0;
        } else if (ok) {
            fprintf(stderr,
                    "MXFP4 Metal down half-LUT poisoned A/B exact on repetition %u\n",
                    run + 1u);
        }
    }
    ds4_gpu_test_set_flags(0);

    mid_is_f16 = false;
    ds4_gpu_test_set_flags(DS4_GPU_TEST_MXFP4_PAIR_TAIL_CULL);
    ok = ok && ds4_gpu_tensor_write(
        mid_batch_tensor, 0, batch_poison,
        batch_pairs * sizeof(_Float16));
    ok = ok && ds4_gpu_tensor_write(
        experts_batch_tensor, 0, batch_poison,
        batch_out_count * N_EXPERT * sizeof(float));
    ok = ok && ds4_gpu_tensor_write(
        out_batch_tensor, 0, batch_poison,
        batch_out_count * sizeof(float));
    ok = ok && ds4_gpu_routed_moe_batch_tensor(
        out_batch_tensor, gate_batch_tensor, up_batch_tensor,
        mid_batch_tensor, experts_batch_tensor,
        model, model_size, gate_offset, up_offset, down_offset,
        MXFP4_TYPE, MXFP4_TYPE, expert_bytes, row_bytes,
        expert_bytes, row_bytes, DIM, DIM, DIM,
        selected_batch_tensor, weights_batch_tensor,
        N_TOTAL_EXPERT, N_EXPERT, 7.0f, x_batch_tensor,
        0u, BATCH_TOKENS, &mid_is_f16, false, true);
    ok = ok && mid_is_f16;
    ok = ok && ds4_gpu_tensor_read(
        mid_batch_tensor, 0, mid_batch_storage,
        batch_pairs * sizeof(_Float16));
    ok = ok && ds4_gpu_tensor_read(
        out_batch_tensor, 0, out_batch_actual,
        batch_out_count * sizeof(float));
    ds4_gpu_test_set_flags(0);
    if (ok && (memcmp(mid_batch_storage, mid_batch_baseline,
                      batch_pairs * sizeof(_Float16)) != 0 ||
               memcmp(out_batch_actual, out_batch_baseline,
                      batch_out_count * sizeof(float)) != 0)) {
        fprintf(stderr,
                "MXFP4 Metal pair tail-cull A/B mismatch for 16-row expert tails\n");
        ok = 0;
    } else if (ok) {
        fprintf(stderr,
                "MXFP4 Metal pair tail-cull A/B exact for 16-row expert tails\n");
    }

    /* Force the diagnostic-only 32x32/64-thread pair kernel. Its second
     * SIMDgroup is inactive on each 16-row expert tail, while both groups
     * remain in staging and at every threadgroup barrier. Require its FP16
     * intermediate and final F32 output to match the established 64x32 path
     * byte-for-byte. */
    mid_is_f16 = false;
    ds4_gpu_test_set_flags(DS4_GPU_TEST_MXFP4_PAIR_COMPACT_TILE);
    ok = ok && ds4_gpu_tensor_write(
        mid_batch_tensor, 0, batch_poison,
        batch_pairs * sizeof(_Float16));
    ok = ok && ds4_gpu_tensor_write(
        experts_batch_tensor, 0, batch_poison,
        batch_out_count * N_EXPERT * sizeof(float));
    ok = ok && ds4_gpu_tensor_write(
        out_batch_tensor, 0, batch_poison,
        batch_out_count * sizeof(float));
    ok = ok && ds4_gpu_routed_moe_batch_tensor(
        out_batch_tensor, gate_batch_tensor, up_batch_tensor,
        mid_batch_tensor, experts_batch_tensor,
        model, model_size, gate_offset, up_offset, down_offset,
        MXFP4_TYPE, MXFP4_TYPE, expert_bytes, row_bytes,
        expert_bytes, row_bytes, DIM, DIM, DIM,
        selected_batch_tensor, weights_batch_tensor,
        N_TOTAL_EXPERT, N_EXPERT, 7.0f, x_batch_tensor,
        0u, BATCH_TOKENS, &mid_is_f16, false, true);
    ok = ok && mid_is_f16;
    ok = ok && ds4_gpu_tensor_read(
        mid_batch_tensor, 0, mid_batch_storage,
        batch_pairs * sizeof(_Float16));
    ok = ok && ds4_gpu_tensor_read(
        out_batch_tensor, 0, out_batch_actual,
        batch_out_count * sizeof(float));
    ds4_gpu_test_set_flags(0);
    if (ok && (memcmp(mid_batch_storage, mid_batch_baseline,
                      batch_pairs * sizeof(_Float16)) != 0 ||
               memcmp(out_batch_actual, out_batch_baseline,
                      batch_out_count * sizeof(float)) != 0)) {
        fprintf(stderr,
                "MXFP4 Metal compact pair A/B mismatch for 16-row expert tails\n");
        ok = 0;
    } else if (ok) {
        fprintf(stderr,
                "MXFP4 Metal compact pair A/B exact for 16-row expert tails\n");
    }

    mid_is_f16 = false;
    ds4_gpu_test_set_flags(DS4_GPU_TEST_MXFP4_DOWN_TAIL_CULL);
    ok = ok && ds4_gpu_tensor_write(
        mid_batch_tensor, 0, batch_poison,
        batch_pairs * sizeof(_Float16));
    ok = ok && ds4_gpu_tensor_write(
        experts_batch_tensor, 0, batch_poison,
        batch_out_count * N_EXPERT * sizeof(float));
    ok = ok && ds4_gpu_tensor_write(
        out_batch_tensor, 0, batch_poison,
        batch_out_count * sizeof(float));
    ok = ok && ds4_gpu_routed_moe_batch_tensor(
        out_batch_tensor, gate_batch_tensor, up_batch_tensor,
        mid_batch_tensor, experts_batch_tensor,
        model, model_size, gate_offset, up_offset, down_offset,
        MXFP4_TYPE, MXFP4_TYPE, expert_bytes, row_bytes,
        expert_bytes, row_bytes, DIM, DIM, DIM,
        selected_batch_tensor, weights_batch_tensor,
        N_TOTAL_EXPERT, N_EXPERT, 7.0f, x_batch_tensor,
        0u, BATCH_TOKENS, &mid_is_f16, false, true);
    ok = ok && mid_is_f16;
    ok = ok && ds4_gpu_tensor_read(
        mid_batch_tensor, 0, mid_batch_storage,
        batch_pairs * sizeof(_Float16));
    ok = ok && ds4_gpu_tensor_read(
        out_batch_tensor, 0, out_batch_actual,
        batch_out_count * sizeof(float));
    ds4_gpu_test_set_flags(0);
    if (ok && (memcmp(mid_batch_storage, mid_batch_baseline,
                      batch_pairs * sizeof(_Float16)) != 0 ||
               memcmp(out_batch_actual, out_batch_baseline,
                      batch_out_count * sizeof(float)) != 0)) {
        fprintf(stderr,
                "MXFP4 Metal down tail-cull A/B mismatch for 16-row expert tails\n");
        ok = 0;
    } else if (ok) {
        fprintf(stderr,
                "MXFP4 Metal down tail-cull A/B exact for 16-row expert tails\n");
    }
    for (uint64_t i = 0; i < batch_pairs; i++) {
        mid_batch_actual[i] = (float)mid_batch_storage[i];
    }
    if (ok) {
        ok = compare_values("bmid", mid_batch_actual, mid_batch_expected,
                            batch_pairs, 2.0e-3f) &&
             compare_values("bout", out_batch_actual, out_batch_expected,
                            batch_out_count, 2.0e-3f);
    }

    /* Exercise token-centric map construction with uneven expert occupancy.
     * Experts 0..4 receive 48 rows, expert 5 receives 33, expert 6 receives
     * 15, and expert 7 is empty. This produces 13 occupied descriptors against
     * a padded direct work capacity of 17, including 16-, 15-, and 1-row
     * tails. Keep the compact pair kernel fixed across the baseline and both
     * scatter repetitions, and require exact FP16 mid and F32 output bytes. */
    for (uint32_t token = 0; token < BATCH_TOKENS; token++) {
        int32_t * token_selected =
            selected_batch + (uint64_t)token * N_EXPERT;
        for (uint32_t slot = 0; slot < N_EXPERT - 1u; slot++) {
            token_selected[slot] = (int32_t)slot;
        }
        token_selected[N_EXPERT - 1u] = token < 33u ? 5 : 6;
    }
    ok = ok && ds4_gpu_tensor_write(
        selected_batch_tensor, 0, selected_batch,
        BATCH_TOKENS * sizeof(selected));
    ds4_gpu_test_set_flags(DS4_GPU_TEST_MXFP4_PAIR_COMPACT_TILE);
    mid_is_f16 = false;
    ok = ok && ds4_gpu_routed_moe_batch_tensor(
        out_batch_tensor, gate_batch_tensor, up_batch_tensor,
        mid_batch_tensor, experts_batch_tensor,
        model, model_size, gate_offset, up_offset, down_offset,
        MXFP4_TYPE, MXFP4_TYPE, expert_bytes, row_bytes,
        expert_bytes, row_bytes, DIM, DIM, DIM,
        selected_batch_tensor, weights_batch_tensor,
        N_TOTAL_EXPERT, N_EXPERT, 7.0f, x_batch_tensor,
        0u, BATCH_TOKENS, &mid_is_f16, false, true);
    ok = ok && mid_is_f16;
    ok = ok && ds4_gpu_tensor_read(
        mid_batch_tensor, 0, mid_batch_baseline,
        batch_pairs * sizeof(_Float16));
    ok = ok && ds4_gpu_tensor_read(
        out_batch_tensor, 0, out_batch_baseline,
        batch_out_count * sizeof(float));

    ds4_gpu_test_set_flags(
        DS4_GPU_TEST_MXFP4_PAIR_COMPACT_TILE |
        DS4_GPU_TEST_MXFP4_MAP_SCATTER);
    for (uint32_t run = 0; run < 2u && ok; run++) {
        mid_is_f16 = false;
        ok = ok && ds4_gpu_tensor_write(
            mid_batch_tensor, 0, batch_poison,
            batch_pairs * sizeof(_Float16));
        ok = ok && ds4_gpu_tensor_write(
            experts_batch_tensor, 0, batch_poison,
            batch_out_count * N_EXPERT * sizeof(float));
        ok = ok && ds4_gpu_tensor_write(
            out_batch_tensor, 0, batch_poison,
            batch_out_count * sizeof(float));
        ok = ok && ds4_gpu_routed_moe_batch_tensor(
            out_batch_tensor, gate_batch_tensor, up_batch_tensor,
            mid_batch_tensor, experts_batch_tensor,
            model, model_size, gate_offset, up_offset, down_offset,
            MXFP4_TYPE, MXFP4_TYPE, expert_bytes, row_bytes,
            expert_bytes, row_bytes, DIM, DIM, DIM,
            selected_batch_tensor, weights_batch_tensor,
            N_TOTAL_EXPERT, N_EXPERT, 7.0f, x_batch_tensor,
            0u, BATCH_TOKENS, &mid_is_f16, false, true);
        ok = ok && mid_is_f16;
        ok = ok && ds4_gpu_tensor_read(
            mid_batch_tensor, 0, mid_batch_storage,
            batch_pairs * sizeof(_Float16));
        ok = ok && ds4_gpu_tensor_read(
            out_batch_tensor, 0, out_batch_actual,
            batch_out_count * sizeof(float));
        if (ok && (memcmp(mid_batch_storage, mid_batch_baseline,
                          batch_pairs * sizeof(_Float16)) != 0 ||
                   memcmp(out_batch_actual, out_batch_baseline,
                          batch_out_count * sizeof(float)) != 0)) {
            fprintf(stderr,
                    "MXFP4 Metal map scatter A/B mismatch on repetition %u\n",
                    run + 1u);
            ok = 0;
        } else if (ok) {
            fprintf(stderr,
                    "MXFP4 Metal map scatter A/B exact on repetition %u\n",
                    run + 1u);
        }
    }
    ds4_gpu_test_set_flags(0);

    /* Finally compare the complete automatic 32..2047-token default bundle
     * with its aggregate rollback. Reuse the uneven routes above so the map
     * contains full tiles plus 16-, 15-, and 1-row tails. */
    mid_is_f16 = false;
    ok = ok && ds4_gpu_tensor_write(
        mid_batch_tensor, 0, batch_poison,
        batch_pairs * sizeof(_Float16));
    ok = ok && ds4_gpu_tensor_write(
        experts_batch_tensor, 0, batch_poison,
        batch_out_count * N_EXPERT * sizeof(float));
    ok = ok && ds4_gpu_tensor_write(
        out_batch_tensor, 0, batch_poison,
        batch_out_count * sizeof(float));
    ok = ok && ds4_gpu_routed_moe_batch_tensor(
        out_batch_tensor, gate_batch_tensor, up_batch_tensor,
        mid_batch_tensor, experts_batch_tensor,
        model, model_size, gate_offset, up_offset, down_offset,
        MXFP4_TYPE, MXFP4_TYPE, expert_bytes, row_bytes,
        expert_bytes, row_bytes, DIM, DIM, DIM,
        selected_batch_tensor, weights_batch_tensor,
        N_TOTAL_EXPERT, N_EXPERT, 7.0f, x_batch_tensor,
        0u, BATCH_TOKENS, &mid_is_f16, false, true);
    ok = ok && mid_is_f16;
    ok = ok && ds4_gpu_tensor_read(
        mid_batch_tensor, 0, mid_batch_baseline,
        batch_pairs * sizeof(_Float16));
    ok = ok && ds4_gpu_tensor_read(
        experts_batch_tensor, 0, experts_batch_half_lut_baseline,
        batch_out_count * N_EXPERT * sizeof(float));
    ok = ok && ds4_gpu_tensor_read(
        out_batch_tensor, 0, out_batch_baseline,
        batch_out_count * sizeof(float));

    if (unsetenv(small_prefill_rollback_name) != 0) {
        fprintf(stderr,
                "MXFP4 Metal could not enable short-prefill defaults\n");
        ok = 0;
    }
    mid_is_f16 = false;
    ok = ok && ds4_gpu_tensor_write(
        mid_batch_tensor, 0, batch_poison,
        batch_pairs * sizeof(_Float16));
    ok = ok && ds4_gpu_tensor_write(
        experts_batch_tensor, 0, batch_poison,
        batch_out_count * N_EXPERT * sizeof(float));
    ok = ok && ds4_gpu_tensor_write(
        out_batch_tensor, 0, batch_poison,
        batch_out_count * sizeof(float));
    ok = ok && ds4_gpu_routed_moe_batch_tensor(
        out_batch_tensor, gate_batch_tensor, up_batch_tensor,
        mid_batch_tensor, experts_batch_tensor,
        model, model_size, gate_offset, up_offset, down_offset,
        MXFP4_TYPE, MXFP4_TYPE, expert_bytes, row_bytes,
        expert_bytes, row_bytes, DIM, DIM, DIM,
        selected_batch_tensor, weights_batch_tensor,
        N_TOTAL_EXPERT, N_EXPERT, 7.0f, x_batch_tensor,
        0u, BATCH_TOKENS, &mid_is_f16, false, true);
    ok = ok && mid_is_f16;
    ok = ok && ds4_gpu_tensor_read(
        mid_batch_tensor, 0, mid_batch_storage,
        batch_pairs * sizeof(_Float16));
    ok = ok && ds4_gpu_tensor_read(
        experts_batch_tensor, 0, experts_batch_actual,
        batch_out_count * N_EXPERT * sizeof(float));
    ok = ok && ds4_gpu_tensor_read(
        out_batch_tensor, 0, out_batch_actual,
        batch_out_count * sizeof(float));
    if (ok && (memcmp(mid_batch_storage, mid_batch_baseline,
                      batch_pairs * sizeof(_Float16)) != 0 ||
               memcmp(experts_batch_actual,
                      experts_batch_half_lut_baseline,
                      batch_out_count * N_EXPERT * sizeof(float)) != 0 ||
               memcmp(out_batch_actual, out_batch_baseline,
                      batch_out_count * sizeof(float)) != 0)) {
        fprintf(stderr,
                "MXFP4 Metal short-prefill default bundle A/B mismatch\n");
        ok = 0;
    } else if (ok) {
        fprintf(stderr,
                "MXFP4 Metal short-prefill default bundle A/B exact\n");
    }
    if (small_prefill_rollback_saved) {
        if (setenv(small_prefill_rollback_name,
                   small_prefill_rollback_saved, 1) != 0) {
            fprintf(stderr,
                    "MXFP4 Metal could not restore short-prefill rollback\n");
            ok = 0;
        }
    } else if (unsetenv(small_prefill_rollback_name) != 0) {
        fprintf(stderr,
                "MXFP4 Metal could not restore short-prefill environment\n");
        ok = 0;
    }
    free(small_prefill_rollback_saved);

    ds4_gpu_tensor_free(out_batch_tensor);
    ds4_gpu_tensor_free(experts_batch_tensor);
    ds4_gpu_tensor_free(mid_batch_tensor);
    ds4_gpu_tensor_free(up_batch_tensor);
    ds4_gpu_tensor_free(gate_batch_tensor);
    ds4_gpu_tensor_free(weights_batch_tensor);
    ds4_gpu_tensor_free(selected_batch_tensor);
    ds4_gpu_tensor_free(x_batch_tensor);
    free(out_batch_actual);
    free(out_batch_half_lut_baseline);
    free(out_batch_baseline);
    free(out_batch_expected);
    free(experts_batch_actual);
    free(experts_batch_half_lut_baseline);
    free(batch_poison);
    free(mid_batch_storage);
    free(mid_batch_half_lut_baseline);
    free(mid_batch_baseline);
    free(mid_batch_actual);
    free(mid_batch_expected);
    free(weights_batch);
    free(selected_batch);
    free(x_batch);

    ds4_gpu_tensor_free(out_tensor);
    ds4_gpu_tensor_free(experts_tensor);
    ds4_gpu_tensor_free(mid_tensor);
    ds4_gpu_tensor_free(up_tensor);
    ds4_gpu_tensor_free(gate_tensor);
    ds4_gpu_tensor_free(weights_tensor);
    ds4_gpu_tensor_free(selected_tensor);
    ds4_gpu_tensor_free(x_tensor);
    ds4_gpu_cleanup();
    free(out_fast);
    free(experts_fast);
    free(mid_fast);
    free(up_fast);
    free(gate_fast);
    free(out_gpu);
    free(experts_gpu);
    free(mid_gpu);
    free(up_gpu);
    free(gate_gpu);
    free(out_ref);
    free(mid_ref);
    free(up_ref);
    free(gate_ref);
    free(model);

    fprintf(stderr, "MXFP4 Metal fused MoE: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
