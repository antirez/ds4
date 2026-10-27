/* Synthetic end-to-end test for the ROCm MXFP4 routed-MoE paths.
 *
 * The ROCm kernels quantize both the input and the fused FP32 SwiGLU mid
 * activation to Q8_K.  The CPU oracle below independently mirrors that
 * quantization before applying the MXFP4 weights.  Four repeated routing
 * patterns keep the 512-token reference inexpensive while still exercising
 * token indexing, expert bucketing, rectangular expert matrices, multiple
 * Q8_K chunks per row, and expert IDs at both ends of a 256-expert table.
 */

#include "ds4_gpu.h"

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#define MXFP4_TYPE 39u
#define QK_MXFP4 32u
#define QK_K 256u
/* Default to the DS4 Flash routed-expert shape: model_dim 4096 means 16
 * Q8_K activation chunks per token, which is what the resident tile and
 * decode kernels see in production.  Smaller synthetic dims miss the
 * no-staging and multi-chunk code paths entirely. */
#ifndef N_TOTAL_EXPERT
#define N_TOTAL_EXPERT 256u
#endif
#ifndef N_EXPERT
#define N_EXPERT 6u
#endif
#ifndef MODEL_DIM
#define MODEL_DIM 4096u
#endif
#ifndef FFN_DIM
#define FFN_DIM 2048u
#endif
#define N_PATTERN 4u
#define CLAMP 7.0f

typedef struct {
    uint8_t e;
    uint8_t qs[QK_MXFP4 / 2u];
} block_mxfp4;

typedef struct {
    float d;
    int8_t qs[QK_K];
} ref_block_q8_K;

typedef struct {
    float x[MODEL_DIM];
    int32_t selected[N_EXPERT];
    float weights[N_EXPERT];
    float mid[N_EXPERT * FFN_DIM];
} reference_pattern;

static const float mxfp4_values[16] = {
     0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

static uint64_t align_up(uint64_t value, uint64_t alignment) {
    return (value + alignment - 1u) / alignment * alignment;
}

static uint32_t mix32(uint32_t x) {
    x ^= x >> 16u;
    x *= 0x7feb352du;
    x ^= x >> 15u;
    x *= 0x846ca68bu;
    x ^= x >> 16u;
    return x;
}

static float e8m0_to_f32(uint8_t e) {
    uint32_t bits = e == 0u ? 0x00400000u : (uint32_t)e << 23u;
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static void quantize_q8_K_block(ref_block_q8_K *out, const float *x) {
    float amax = 0.0f;
    float maxv = 0.0f;
    for (uint32_t i = 0; i < QK_K; i++) {
        const float ax = fabsf(x[i]);
        /* The device reduction also keeps the lower index on an exact tie. */
        if (ax > amax) {
            amax = ax;
            maxv = x[i];
        }
    }
    if (amax == 0.0f) {
        out->d = 0.0f;
        memset(out->qs, 0, sizeof(out->qs));
        return;
    }

    const float iscale = -127.0f / maxv;
    out->d = 1.0f / iscale;
    for (uint32_t i = 0; i < QK_K; i++) {
        long q = lrintf(iscale * x[i]);
        if (q > 127) q = 127;
        if (q < -128) q = -128;
        out->qs[i] = (int8_t)q;
    }
}

static void quantize_q8_K(ref_block_q8_K *out,
                          const float *x,
                          uint32_t n) {
    for (uint32_t block = 0; block < n / QK_K; block++) {
        quantize_q8_K_block(out + block, x + (uint64_t)block * QK_K);
    }
}

static float dot_mxfp4_q8_K(const block_mxfp4 *row,
                            const ref_block_q8_K *x,
                            uint32_t input_dim) {
    float sum = 0.0f;
    const uint32_t mxfp4_per_q8 = QK_K / QK_MXFP4;
    for (uint32_t block = 0; block < input_dim / QK_MXFP4; block++) {
        const block_mxfp4 *b = row + block;
        const ref_block_q8_K *xb = x + block / mxfp4_per_q8;
        const uint32_t q8_offset = (block % mxfp4_per_q8) * QK_MXFP4;
        const float scale = e8m0_to_f32(b->e) * xb->d;
        for (uint32_t i = 0; i < QK_MXFP4 / 2u; i++) {
            const uint8_t q = b->qs[i];
            sum += scale * mxfp4_values[q & 15u] *
                   (float)xb->qs[q8_offset + i];
            sum += scale * mxfp4_values[q >> 4u] *
                   (float)xb->qs[q8_offset + i + QK_MXFP4 / 2u];
        }
    }
    return sum;
}

static void fill_matrix(block_mxfp4 *matrix,
                        uint32_t rows,
                        uint32_t input_dim,
                        uint32_t salt) {
    const uint32_t blocks_per_row = input_dim / QK_MXFP4;
    for (uint32_t expert = 0; expert < N_TOTAL_EXPERT; expert++) {
        for (uint32_t row = 0; row < rows; row++) {
            block_mxfp4 *blocks = matrix +
                ((uint64_t)expert * rows + row) * blocks_per_row;
            for (uint32_t block = 0; block < blocks_per_row; block++) {
                block_mxfp4 *b = blocks + block;
                const uint32_t key = salt ^ (expert * 0x9e3779b9u) ^
                                     (row * 0x85ebca6bu) ^
                                     (block * 0xc2b2ae35u);
                b->e = (uint8_t)(120u + mix32(key) % 5u);
                for (uint32_t i = 0; i < QK_MXFP4 / 2u; i++) {
                    const uint32_t h = mix32(key + i * 0x27d4eb2du);
                    b->qs[i] = (uint8_t)((h & 15u) | (((h >> 9u) & 15u) << 4u));
                }
            }
        }
    }
}

static const block_mxfp4 *matrix_row(const block_mxfp4 *matrix,
                                      uint32_t expert,
                                      uint32_t row,
                                      uint32_t rows,
                                      uint32_t input_dim) {
    return matrix + ((uint64_t)expert * rows + row) *
                    (input_dim / QK_MXFP4);
}

static void init_patterns(reference_pattern patterns[N_PATTERN]) {
    static const int32_t selected[N_PATTERN][N_EXPERT] = {
        {   0,   1,   2,   3,   4, 255 },
        {  17,  63, 127, 128, 200, 254 },
        { 255,   0, 129,  42,  11, 201 },
        {   5,  85, 170, 250,  13, 199 },
    };
    static const float base_weights[N_EXPERT] = {
        0.25f, 0.20f, 0.18f, 0.15f, 0.12f, 0.10f,
    };

    memset(patterns, 0, sizeof(reference_pattern) * N_PATTERN);
    for (uint32_t p = 0; p < N_PATTERN; p++) {
        memcpy(patterns[p].selected, selected[p], sizeof(selected[p]));
        for (uint32_t slot = 0; slot < N_EXPERT; slot++) {
            patterns[p].weights[slot] =
                base_weights[(slot + p) % N_EXPERT];
        }
        for (uint32_t i = 0; i < MODEL_DIM; i++) {
            const uint32_t h = mix32(i + 1u + p * 0x9e3779b9u);
            patterns[p].x[i] =
                (float)((int32_t)(h % 255u) - 127) / 256.0f;
        }
        /* Give each vector a unique signed maximum so Q8_K scale selection
         * is deterministic on both the CPU and GPU. */
        patterns[p].x[19u + p * 47u] = (p & 1u) ? 0.875f : -0.875f;
    }
}

static void build_reference(reference_pattern *pattern,
                            const block_mxfp4 *gate_matrix,
                            const block_mxfp4 *up_matrix) {
    ref_block_q8_K xq[MODEL_DIM / QK_K];
    ref_block_q8_K midq[N_EXPERT][FFN_DIM / QK_K];
    quantize_q8_K(xq, pattern->x, MODEL_DIM);

    for (uint32_t slot = 0; slot < N_EXPERT; slot++) {
        const uint32_t expert = (uint32_t)pattern->selected[slot];
        float *mid = pattern->mid + (uint64_t)slot * FFN_DIM;
        for (uint32_t row = 0; row < FFN_DIM; row++) {
            float gate = dot_mxfp4_q8_K(
                matrix_row(gate_matrix, expert, row, FFN_DIM, MODEL_DIM),
                xq, MODEL_DIM);
            float up = dot_mxfp4_q8_K(
                matrix_row(up_matrix, expert, row, FFN_DIM, MODEL_DIM),
                xq, MODEL_DIM);
            if (gate > CLAMP) gate = CLAMP;
            if (up > CLAMP) up = CLAMP;
            if (up < -CLAMP) up = -CLAMP;
            mid[row] = (gate / (1.0f + expf(-gate))) * up *
                       pattern->weights[slot];
        }
        quantize_q8_K(midq[slot], mid, FFN_DIM);
    }
}

static int compare_repeated(const char *name,
                            const float *actual,
                            uint32_t n_tokens,
                            uint32_t token_elems,
                            const reference_pattern patterns[N_PATTERN],
                            float abs_tolerance,
                            float rel_tolerance) {
    float max_abs = 0.0f;
    float max_ratio = 0.0f;
    uint64_t max_abs_index = 0u;
    uint64_t max_ratio_index = 0u;
    uint64_t failures = 0u;
    const uint64_t count = (uint64_t)n_tokens * token_elems;

    for (uint32_t token = 0; token < n_tokens; token++) {
        const float *expected = patterns[token % N_PATTERN].mid;
        for (uint32_t i = 0; i < token_elems; i++) {
            const uint64_t index = (uint64_t)token * token_elems + i;
            const float got = actual[index];
            const float want = expected[i];
            if (!isfinite(got) || !isfinite(want)) {
                fprintf(stderr,
                        "MXFP4 ROCm tokens=%u %s non-finite at token=%u element=%u "
                        "got=%g expected=%g\n",
                        n_tokens, name, token, i, got, want);
                return 0;
            }
            const float error = fabsf(got - want);
            const float allowed = abs_tolerance + rel_tolerance * fabsf(want);
            const float ratio = allowed > 0.0f ? error / allowed : error;
            if (error > max_abs) {
                max_abs = error;
                max_abs_index = index;
            }
            if (ratio > max_ratio) {
                max_ratio = ratio;
                max_ratio_index = index;
            }
            if (error > allowed) failures++;
        }
    }

    fprintf(stderr,
            "MXFP4 ROCm tokens=%-3u %-3s max_abs=%-10g at=%llu "
            "max_tol_ratio=%g at=%llu failures=%llu/%llu\n",
            n_tokens, name, max_abs, (unsigned long long)max_abs_index,
            max_ratio, (unsigned long long)max_ratio_index,
            (unsigned long long)failures, (unsigned long long)count);
    return failures == 0u;
}

/* The GPU quantizes its own gate/up mid before the down projection, and a
 * one-LSB rounding flip against the analytic reference mid is legitimate:
 * both mids sit within float tolerance of each other, yet one flipped
 * activation moves a down dot by up to d * 2^(e-127) * 6, which real-shape
 * scales push past any fixed tolerance.  Judge the down stage on the GPU's
 * own mid instead: quantize it with the identical CPU algorithm and require
 * the down kernels to match that reference tightly.  Expected rows are
 * cached per pattern; repeated tokens reuse them after a bitwise mid check. */
static int check_out_from_gpu_mid(const float *out_actual,
                                  const float *mid_actual,
                                  uint32_t n_tokens,
                                  const reference_pattern patterns[N_PATTERN],
                                  const block_mxfp4 *down_matrix,
                                  float abs_tolerance,
                                  float rel_tolerance) {
    static ref_block_q8_K midq[N_EXPERT][FFN_DIM / QK_K];
    static float expected[N_PATTERN][MODEL_DIM];
    int32_t cached_token[N_PATTERN] = { -1, -1, -1, -1 };
    float max_abs = 0.0f;
    float max_ratio = 0.0f;
    uint64_t max_abs_index = 0u;
    uint64_t max_ratio_index = 0u;
    uint64_t failures = 0u;
    const uint64_t count = (uint64_t)n_tokens * MODEL_DIM;

    for (uint32_t token = 0; token < n_tokens; token++) {
        const uint32_t p = token % N_PATTERN;
        const reference_pattern *pattern = &patterns[p];
        const float *mid = mid_actual + (uint64_t)token * N_EXPERT * FFN_DIM;
        const uint64_t mid_bytes = (uint64_t)N_EXPERT * FFN_DIM * sizeof(float);
        if (cached_token[p] < 0 ||
            memcmp(mid_actual + (uint64_t)cached_token[p] * N_EXPERT * FFN_DIM,
                   mid, mid_bytes) != 0) {
            for (uint32_t slot = 0; slot < N_EXPERT; slot++) {
                quantize_q8_K(midq[slot], mid + (uint64_t)slot * FFN_DIM, FFN_DIM);
            }
            for (uint32_t row = 0; row < MODEL_DIM; row++) {
                float want = 0.0f;
                for (uint32_t slot = 0; slot < N_EXPERT; slot++) {
                    const uint32_t expert = (uint32_t)pattern->selected[slot];
                    want += dot_mxfp4_q8_K(
                        matrix_row(down_matrix, expert, row, MODEL_DIM, FFN_DIM),
                        midq[slot], FFN_DIM);
                }
                expected[p][row] = want;
            }
            cached_token[p] = (int32_t)token;
        }
        for (uint32_t i = 0; i < MODEL_DIM; i++) {
            const uint64_t index = (uint64_t)token * MODEL_DIM + i;
            const float got = out_actual[index];
            const float want = expected[p][i];
            if (!isfinite(got) || !isfinite(want)) {
                fprintf(stderr,
                        "MXFP4 ROCm tokens=%u out non-finite at token=%u element=%u "
                        "got=%g expected=%g\n",
                        n_tokens, token, i, got, want);
                return 0;
            }
            const float error = fabsf(got - want);
            const float allowed = abs_tolerance + rel_tolerance * fabsf(want);
            const float ratio = allowed > 0.0f ? error / allowed : error;
            if (error > max_abs) {
                max_abs = error;
                max_abs_index = index;
            }
            if (ratio > max_ratio) {
                max_ratio = ratio;
                max_ratio_index = index;
            }
            if (error > allowed) failures++;
        }
    }

    fprintf(stderr,
            "MXFP4 ROCm tokens=%-3u out max_abs=%-10g at=%llu "
            "max_tol_ratio=%g at=%llu failures=%llu/%llu\n",
            n_tokens, max_abs, (unsigned long long)max_abs_index,
            max_ratio, (unsigned long long)max_ratio_index,
            (unsigned long long)failures, (unsigned long long)count);
    return failures == 0u;
}

static int run_case(uint32_t n_tokens,
                    const void *model,
                    uint64_t model_size,
                    uint64_t gate_offset,
                    uint64_t up_offset,
                    uint64_t down_offset,
                    uint64_t gate_expert_bytes,
                    uint64_t gate_row_bytes,
                    uint64_t down_expert_bytes,
                    uint64_t down_row_bytes,
                    const reference_pattern patterns[N_PATTERN],
                    float *snapshot_out,
                    const float *expect_out) {
    const uint64_t token_x_count = (uint64_t)n_tokens * MODEL_DIM;
    const uint64_t route_count = (uint64_t)n_tokens * N_EXPERT;
    const uint64_t mid_count = route_count * FFN_DIM;
    const uint64_t down_count = route_count * MODEL_DIM;
    const uint64_t out_count = token_x_count;
    float *x = (float *)calloc((size_t)token_x_count, sizeof(float));
    int32_t *selected = (int32_t *)calloc((size_t)route_count, sizeof(int32_t));
    float *weights = (float *)calloc((size_t)route_count, sizeof(float));
    float *mid_actual = (float *)calloc((size_t)mid_count, sizeof(float));
    float *out_actual = (float *)calloc((size_t)out_count, sizeof(float));
    ds4_gpu_tensor *x_tensor = NULL;
    ds4_gpu_tensor *selected_tensor = NULL;
    ds4_gpu_tensor *weights_tensor = NULL;
    ds4_gpu_tensor *gate_tensor = NULL;
    ds4_gpu_tensor *up_tensor = NULL;
    ds4_gpu_tensor *mid_tensor = NULL;
    ds4_gpu_tensor *experts_tensor = NULL;
    ds4_gpu_tensor *out_tensor = NULL;
    int ok = x && selected && weights && mid_actual && out_actual;

    for (uint32_t token = 0; ok && token < n_tokens; token++) {
        const reference_pattern *pattern = &patterns[token % N_PATTERN];
        memcpy(x + (uint64_t)token * MODEL_DIM,
               pattern->x, sizeof(pattern->x));
        memcpy(selected + (uint64_t)token * N_EXPERT,
               pattern->selected, sizeof(pattern->selected));
        memcpy(weights + (uint64_t)token * N_EXPERT,
               pattern->weights, sizeof(pattern->weights));
    }

    if (ok) x_tensor = ds4_gpu_tensor_alloc(token_x_count * sizeof(float));
    if (ok) selected_tensor = ds4_gpu_tensor_alloc(route_count * sizeof(int32_t));
    if (ok) weights_tensor = ds4_gpu_tensor_alloc(route_count * sizeof(float));
    if (ok) gate_tensor = ds4_gpu_tensor_alloc(mid_count * sizeof(float));
    if (ok) up_tensor = ds4_gpu_tensor_alloc(mid_count * sizeof(float));
    if (ok) mid_tensor = ds4_gpu_tensor_alloc(mid_count * sizeof(float));
    if (ok) experts_tensor = ds4_gpu_tensor_alloc(down_count * sizeof(float));
    if (ok) out_tensor = ds4_gpu_tensor_alloc(out_count * sizeof(float));
    ok = ok && x_tensor && selected_tensor && weights_tensor && gate_tensor &&
         up_tensor && mid_tensor && experts_tensor && out_tensor;

    ok = ok && ds4_gpu_tensor_write(
        x_tensor, 0u, x, token_x_count * sizeof(float));
    ok = ok && ds4_gpu_tensor_write(
        selected_tensor, 0u, selected, route_count * sizeof(int32_t));
    ok = ok && ds4_gpu_tensor_write(
        weights_tensor, 0u, weights, route_count * sizeof(float));

    if (ok && n_tokens == 1u) {
        ok = ds4_gpu_routed_moe_one_tensor(
            out_tensor, gate_tensor, up_tensor, mid_tensor, experts_tensor,
            model, model_size, gate_offset, up_offset, down_offset,
            MXFP4_TYPE, MXFP4_TYPE, gate_expert_bytes, gate_row_bytes,
            down_expert_bytes, down_row_bytes,
            MODEL_DIM, FFN_DIM, MODEL_DIM,
            selected_tensor, weights_tensor, N_TOTAL_EXPERT, N_EXPERT,
            CLAMP, x_tensor, NULL, 0u, true);
    } else if (ok) {
        bool mid_is_f16 = true;
        ok = ds4_gpu_routed_moe_batch_tensor(
            out_tensor, gate_tensor, up_tensor, mid_tensor, experts_tensor,
            model, model_size, gate_offset, up_offset, down_offset,
            MXFP4_TYPE, MXFP4_TYPE, gate_expert_bytes, gate_row_bytes,
            down_expert_bytes, down_row_bytes,
            MODEL_DIM, FFN_DIM, MODEL_DIM,
            selected_tensor, weights_tensor, N_TOTAL_EXPERT, N_EXPERT,
            CLAMP, x_tensor, 0u, n_tokens, &mid_is_f16, true);
        if (ok && mid_is_f16) {
            fprintf(stderr,
                    "MXFP4 ROCm tokens=%u unexpectedly reported FP16 mid storage\n",
                    n_tokens);
            ok = 0;
        }
    }

    if (!ok) {
        fprintf(stderr, "MXFP4 ROCm tokens=%u launch failed\n", n_tokens);
    }
    ok = ok && ds4_gpu_tensor_read(
        mid_tensor, 0u, mid_actual, mid_count * sizeof(float));
    ok = ok && ds4_gpu_tensor_read(
        out_tensor, 0u, out_actual, out_count * sizeof(float));

    if (ok) {
        /* Gate/up are optional scratch outputs in the optimized ROCm paths;
         * mid is the public, stable result of that fused stage. */
        const int mid_ok = compare_repeated(
            "mid", mid_actual, n_tokens, N_EXPERT * FFN_DIM, patterns,
            1.0e-4f, 1.0e-4f);
        const int out_ok = check_out_from_gpu_mid(
            out_actual, mid_actual, n_tokens, patterns,
            (const block_mxfp4 *)((const char *)model + down_offset),
            2.0e-4f, 1.0e-4f);
        ok = mid_ok && out_ok;
    }
    if (ok && snapshot_out) {
        memcpy(snapshot_out, out_actual, out_count * sizeof(float));
    }
    if (ok && expect_out &&
        memcmp(expect_out, out_actual, out_count * sizeof(float)) != 0) {
        fprintf(stderr,
                "MXFP4 ROCm tokens=%u bitwise mismatch vs default tile path\n",
                n_tokens);
        ok = 0;
    }

    ds4_gpu_tensor_free(out_tensor);
    ds4_gpu_tensor_free(experts_tensor);
    ds4_gpu_tensor_free(mid_tensor);
    ds4_gpu_tensor_free(up_tensor);
    ds4_gpu_tensor_free(gate_tensor);
    ds4_gpu_tensor_free(weights_tensor);
    ds4_gpu_tensor_free(selected_tensor);
    ds4_gpu_tensor_free(x_tensor);
    free(out_actual);
    free(mid_actual);
    free(weights);
    free(selected);
    free(x);
    return ok;
}

int main(void) {
    /* Two through four tokens exercise the direct tiny-batch path; five
     * tokens returns to expert-sorted tiles. Larger cases cover prefill. */
    static const uint32_t token_cases[] = {
        1u, 2u, 3u, 4u, 5u, 32u, 128u, 512u,
    };
    const uint64_t gate_row_bytes =
        (MODEL_DIM / QK_MXFP4) * sizeof(block_mxfp4);
    const uint64_t gate_expert_bytes = FFN_DIM * gate_row_bytes;
    const uint64_t gate_tensor_bytes =
        N_TOTAL_EXPERT * gate_expert_bytes;
    const uint64_t down_row_bytes =
        (FFN_DIM / QK_MXFP4) * sizeof(block_mxfp4);
    const uint64_t down_expert_bytes = MODEL_DIM * down_row_bytes;
    const uint64_t down_tensor_bytes =
        N_TOTAL_EXPERT * down_expert_bytes;
    const uint64_t gate_offset = 0u;
    const uint64_t up_offset = align_up(gate_tensor_bytes, 4096u);
    const uint64_t down_offset =
        align_up(up_offset + gate_tensor_bytes, 4096u);
    const uint64_t model_size =
        align_up(down_offset + down_tensor_bytes, 4096u);
    FILE *model_file = NULL;
    void *model = MAP_FAILED;
    reference_pattern *patterns = NULL;
    int initialized = 0;
    int ok = sizeof(block_mxfp4) == 17u;

    if (!ok) {
        fprintf(stderr, "MXFP4 ROCm unexpected block size %zu (expected 17)\n",
                sizeof(block_mxfp4));
        return 1;
    }

    model_file = tmpfile();
    if (model_file &&
        ftruncate(fileno(model_file), (off_t)model_size) == 0) {
        model = mmap(NULL, (size_t)model_size, PROT_READ | PROT_WRITE,
                     MAP_SHARED, fileno(model_file), 0);
    }
    patterns = (reference_pattern *)calloc(N_PATTERN, sizeof(*patterns));
    if (!model_file || model == MAP_FAILED || !patterns) {
        fprintf(stderr, "MXFP4 ROCm host allocation failed\n");
        ok = 0;
        goto cleanup;
    }
    memset(model, 0, (size_t)model_size);
    fill_matrix((block_mxfp4 *)((uint8_t *)model + gate_offset),
                FFN_DIM, MODEL_DIM, 0x12345678u);
    fill_matrix((block_mxfp4 *)((uint8_t *)model + up_offset),
                FFN_DIM, MODEL_DIM, 0x9abcdef0u);
    fill_matrix((block_mxfp4 *)((uint8_t *)model + down_offset),
                MODEL_DIM, FFN_DIM, 0x0f1e2d3cu);
    init_patterns(patterns);

    const block_mxfp4 *gate_matrix =
        (const block_mxfp4 *)((const uint8_t *)model + gate_offset);
    const block_mxfp4 *up_matrix =
        (const block_mxfp4 *)((const uint8_t *)model + up_offset);
    for (uint32_t p = 0; p < N_PATTERN; p++) {
        build_reference(&patterns[p], gate_matrix, up_matrix);
    }

    fprintf(stderr,
            "MXFP4 ROCm synthetic model: %.2f MiB, experts=%u, "
            "model_dim=%u, ffn_dim=%u, selected=%u\n",
            (double)model_size / 1048576.0, N_TOTAL_EXPERT,
            MODEL_DIM, FFN_DIM, N_EXPERT);
    ok = ds4_gpu_init();
    initialized = ok;
    if (!ok) {
        fprintf(stderr, "MXFP4 ROCm ds4_gpu_init failed\n");
        goto cleanup;
    }
    ds4_gpu_set_quality(false);
    ds4_gpu_set_ssd_streaming(false);
    const uint64_t model_offsets[] = {
        gate_offset, up_offset, down_offset,
    };
    const uint64_t model_sizes[] = {
        gate_tensor_bytes, gate_tensor_bytes, down_tensor_bytes,
    };
    const uint64_t max_tensor_bytes =
        gate_tensor_bytes > down_tensor_bytes ?
        gate_tensor_bytes : down_tensor_bytes;
    ok = ds4_gpu_set_model_map(model, model_size) &&
         ds4_gpu_set_model_fd(fileno(model_file)) &&
         ds4_gpu_set_model_map_spans(
             model, model_size, model_offsets, model_sizes,
             sizeof(model_offsets) / sizeof(model_offsets[0]),
             max_tensor_bytes);
    if (!ok) {
        fprintf(stderr, "MXFP4 ROCm model cache setup failed\n");
        goto cleanup;
    }

    float *snap128 = (float *)malloc(128u * MODEL_DIM * sizeof(float));
    float *snap512 = (float *)malloc(512u * MODEL_DIM * sizeof(float));
    ok = ok && snap128 && snap512;
    for (uint32_t i = 0; ok && i < sizeof(token_cases) / sizeof(token_cases[0]); i++) {
        float *snap = token_cases[i] == 128u ? snap128 :
                      token_cases[i] == 512u ? snap512 : NULL;
        if (!run_case(token_cases[i], model, model_size,
                      gate_offset, up_offset, down_offset,
                      gate_expert_bytes, gate_row_bytes,
                      down_expert_bytes, down_row_bytes, patterns,
                      snap, NULL)) {
            ok = 0;
        }
    }
    /* Occupancy/path variants must reproduce the default tile path bit
     * for bit: same accumulation order by construction, so the out
     * tensors compare with memcmp, not a tolerance. */
    if (ok) {
        const struct {
            const char *name;
            const char *value;
        } variant_envs[] = {
            { "DS4_ROCM_ENABLE_MXFP4_TILE4", "1" },
            { "DS4_ROCM_ENABLE_MXFP4_ROW64", "1" },
            { "DS4_ROCM_MXFP4_DOWN_RGROUP", "4" },
        };
        for (uint32_t v = 0; v < sizeof(variant_envs) / sizeof(variant_envs[0]); v++) {
            setenv(variant_envs[v].name, variant_envs[v].value, 1);
            const int vok =
                run_case(128u, model, model_size,
                         gate_offset, up_offset, down_offset,
                         gate_expert_bytes, gate_row_bytes,
                         down_expert_bytes, down_row_bytes, patterns,
                         NULL, snap128) &&
                run_case(512u, model, model_size,
                         gate_offset, up_offset, down_offset,
                         gate_expert_bytes, gate_row_bytes,
                         down_expert_bytes, down_row_bytes, patterns,
                         NULL, snap512);
            unsetenv(variant_envs[v].name);
            fprintf(stderr, "MXFP4 ROCm variant %s=%s: %s\n",
                    variant_envs[v].name, variant_envs[v].value,
                    vok ? "bitwise OK" : "MISMATCH");
            if (!vok) ok = 0;
        }
    }
    free(snap128);
    free(snap512);

cleanup:
    if (initialized) {
        ds4_gpu_set_model_fd(-1);
        ds4_gpu_cleanup();
    }
    free(patterns);
    if (model != MAP_FAILED) munmap(model, (size_t)model_size);
    if (model_file) fclose(model_file);
    fprintf(stderr, "MXFP4 ROCm routed MoE: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
