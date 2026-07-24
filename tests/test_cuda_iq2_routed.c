/* Numerical oracle for the CUDA all-IQ2_XXS routed-MoE path.
 *
 * Builds a small deterministic two-token, three-expert problem, evaluates it
 * through the production CUDA dispatcher, and compares every intermediate and
 * final row with the scalar CPU IQ2_XXS implementation from ds4.c.
 */

#include "ds4.h"
#include "ds4_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DIM 256u
#define N_EXPERT 3u
#define N_USED 2u
#define N_TOKEN 2u
#define IQ2_BLOCK_BYTES 66u
#define IQ2_TYPE 16u

typedef struct {
    uint16_t d;
    uint16_t qs[DIM / 8u];
} test_iq2_block;

typedef char test_iq2_block_size[(sizeof(test_iq2_block) == IQ2_BLOCK_BYTES) ? 1 : -1];

static void fail(const char *what) {
    fprintf(stderr, "FAIL: %s\n", what);
    exit(1);
}

static void require(int condition, const char *what) {
    if (!condition) fail(what);
}

static uint32_t next_u32(uint32_t *state) {
    *state = *state * 1664525u + 1013904223u;
    return *state;
}

static void fill_matrix(test_iq2_block *matrix, uint32_t salt) {
    uint32_t state = 0x9e3779b9u ^ salt;
    for (uint32_t expert = 0; expert < N_EXPERT; expert++) {
        for (uint32_t row = 0; row < DIM; row++) {
            test_iq2_block *block = &matrix[(uint64_t)expert * DIM + row];
            block->d = 0x2800u; /* IEEE binary16 0.03125. */
            for (uint32_t i = 0; i < DIM / 8u; i++) {
                block->qs[i] = (uint16_t)(next_u32(&state) >> 8);
            }
        }
    }
}

static float dot_row(const test_iq2_block *matrix,
                     uint32_t expert,
                     uint32_t row,
                     const float *values) {
    return ds4_test_iq2_xxs_dot_f32(
            DIM, &matrix[(uint64_t)expert * DIM + row], values);
}

static void compare_array(const char *name,
                          const float *got,
                          const float *want,
                          uint64_t count,
                          float abs_tol,
                          float rel_tol,
                          float *worst_abs_out,
                          float *worst_rel_out) {
    float worst_abs = 0.0f;
    float worst_rel = 0.0f;
    uint64_t worst_i = 0;
    for (uint64_t i = 0; i < count; i++) {
        const float delta = fabsf(got[i] - want[i]);
        const float scale = fmaxf(fabsf(got[i]), fabsf(want[i]));
        const float rel = scale > 1.0e-7f ? delta / scale : delta;
        if (!isfinite(got[i]) || !isfinite(want[i]) ||
            (delta > abs_tol && rel > rel_tol)) {
            fprintf(stderr,
                    "FAIL: %s[%llu] got=%g want=%g abs=%g rel=%g "
                    "tolerance=(%g,%g)\n",
                    name,
                    (unsigned long long)i,
                    got[i],
                    want[i],
                    delta,
                    rel,
                    abs_tol,
                    rel_tol);
            exit(1);
        }
        if (delta > worst_abs) {
            worst_abs = delta;
            worst_i = i;
        }
        if (rel > worst_rel) worst_rel = rel;
    }
    fprintf(stderr,
            "%s: count=%llu worst_abs=%g at=%llu worst_rel=%g\n",
            name,
            (unsigned long long)count,
            worst_abs,
            (unsigned long long)worst_i,
            worst_rel);
    if (worst_abs_out) *worst_abs_out = worst_abs;
    if (worst_rel_out) *worst_rel_out = worst_rel;
}

int main(void) {
    const uint64_t rows_per_matrix = (uint64_t)N_EXPERT * DIM;
    const uint64_t matrix_bytes = rows_per_matrix * sizeof(test_iq2_block);
    const uint64_t model_bytes = 3u * matrix_bytes;
    void *model_raw = NULL;
    require(posix_memalign(&model_raw, 4096u, model_bytes) == 0,
            "allocate aligned model map");
    test_iq2_block *gate_model = model_raw;
    test_iq2_block *up_model = (test_iq2_block *)((char *)model_raw + matrix_bytes);
    test_iq2_block *down_model = (test_iq2_block *)((char *)model_raw + 2u * matrix_bytes);
    fill_matrix(gate_model, 1u);
    fill_matrix(up_model, 2u);
    fill_matrix(down_model, 3u);

    float x[N_TOKEN * DIM];
    for (uint32_t token = 0; token < N_TOKEN; token++) {
        for (uint32_t i = 0; i < DIM; i++) {
            x[(uint64_t)token * DIM + i] =
                    0.075f * sinf((float)(i + 1u) * (0.031f + 0.007f * token)) +
                    0.025f * cosf((float)(i + 3u) * (0.017f + 0.003f * token));
        }
    }
    const int32_t selected[N_TOKEN * N_USED] = {0, 2, 1, 0};
    const float route_weights[N_TOKEN * N_USED] = {0.65f, 0.35f, 0.55f, 0.45f};
    const uint64_t slot_count = (uint64_t)N_TOKEN * N_USED;
    const uint64_t routed_count = slot_count * DIM;
    const uint64_t out_count = (uint64_t)N_TOKEN * DIM;

    float *want_gate = calloc(routed_count, sizeof(float));
    float *want_up = calloc(routed_count, sizeof(float));
    float *want_mid = calloc(routed_count, sizeof(float));
    float *want_down = calloc(routed_count, sizeof(float));
    float *want_out = calloc(out_count, sizeof(float));
    float *got_gate = calloc(routed_count, sizeof(float));
    float *got_up = calloc(routed_count, sizeof(float));
    float *got_mid = calloc(routed_count, sizeof(float));
    float *got_down = calloc(routed_count, sizeof(float));
    float *got_out = calloc(out_count, sizeof(float));
    require(want_gate && want_up && want_mid && want_down && want_out &&
            got_gate && got_up && got_mid && got_down && got_out,
            "allocate oracle buffers");

    for (uint32_t token = 0; token < N_TOKEN; token++) {
        const float *token_x = &x[(uint64_t)token * DIM];
        for (uint32_t slot = 0; slot < N_USED; slot++) {
            const uint64_t pair = (uint64_t)token * N_USED + slot;
            const uint32_t expert = (uint32_t)selected[pair];
            for (uint32_t row = 0; row < DIM; row++) {
                const uint64_t index = pair * DIM + row;
                const float gate = dot_row(gate_model, expert, row, token_x);
                const float up = dot_row(up_model, expert, row, token_x);
                want_gate[index] = gate;
                want_up[index] = up;
                want_mid[index] =
                        (gate / (1.0f + expf(-gate))) * up * route_weights[pair];
            }
            for (uint32_t row = 0; row < DIM; row++) {
                const uint64_t index = pair * DIM + row;
                want_down[index] = dot_row(
                        down_model, expert, row, &want_mid[pair * DIM]);
                want_out[(uint64_t)token * DIM + row] += want_down[index];
            }
        }
    }

    require(ds4_gpu_init() != 0, "initialize CUDA backend");
    require(ds4_gpu_set_model_map(model_raw, model_bytes) != 0,
            "register test model map");

    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(out_count * sizeof(float));
    ds4_gpu_tensor *gate = ds4_gpu_tensor_alloc(routed_count * sizeof(float));
    ds4_gpu_tensor *up = ds4_gpu_tensor_alloc(routed_count * sizeof(float));
    ds4_gpu_tensor *mid = ds4_gpu_tensor_alloc(routed_count * sizeof(float));
    ds4_gpu_tensor *down = ds4_gpu_tensor_alloc(routed_count * sizeof(float));
    ds4_gpu_tensor *selected_gpu = ds4_gpu_tensor_alloc(sizeof(selected));
    ds4_gpu_tensor *weights_gpu = ds4_gpu_tensor_alloc(sizeof(route_weights));
    ds4_gpu_tensor *x_gpu = ds4_gpu_tensor_alloc(sizeof(x));
    require(out && gate && up && mid && down && selected_gpu && weights_gpu && x_gpu,
            "allocate CUDA tensors");
    require(ds4_gpu_tensor_write(selected_gpu, 0, selected, sizeof(selected)) != 0,
            "upload selected experts");
    require(ds4_gpu_tensor_write(weights_gpu, 0, route_weights, sizeof(route_weights)) != 0,
            "upload route weights");
    require(ds4_gpu_tensor_write(x_gpu, 0, x, sizeof(x)) != 0,
            "upload activations");

    bool mid_is_f16 = true;
    require(ds4_gpu_routed_moe_batch_tensor(
                    out,
                    gate,
                    up,
                    mid,
                    down,
                    model_raw,
                    model_bytes,
                    0,
                    matrix_bytes,
                    2u * matrix_bytes,
                    IQ2_TYPE,
                    IQ2_TYPE,
                    (uint64_t)DIM * IQ2_BLOCK_BYTES,
                    IQ2_BLOCK_BYTES,
                    (uint64_t)DIM * IQ2_BLOCK_BYTES,
                    IQ2_BLOCK_BYTES,
                    DIM,
                    DIM,
                    DIM,
                    selected_gpu,
                    weights_gpu,
                    N_EXPERT,
                    N_USED,
                    0.0f,
                    x_gpu,
                    0,
                    N_TOKEN,
                    &mid_is_f16,
                    true) != 0,
            "launch CUDA IQ2 routed MoE");
    require(!mid_is_f16, "IQ2 routed midpoint remains f32");
    require(ds4_gpu_synchronize() != 0, "synchronize CUDA routed MoE");
    require(ds4_gpu_tensor_read(gate, 0, got_gate, routed_count * sizeof(float)) != 0,
            "read gate output");
    require(ds4_gpu_tensor_read(up, 0, got_up, routed_count * sizeof(float)) != 0,
            "read up output");
    require(ds4_gpu_tensor_read(mid, 0, got_mid, routed_count * sizeof(float)) != 0,
            "read midpoint output");
    require(ds4_gpu_tensor_read(down, 0, got_down, routed_count * sizeof(float)) != 0,
            "read down output");
    require(ds4_gpu_tensor_read(out, 0, got_out, out_count * sizeof(float)) != 0,
            "read routed output");

    float worst_abs = 0.0f;
    float worst_rel = 0.0f;
    compare_array("gate", got_gate, want_gate, routed_count,
                  2.0e-5f, 2.0e-5f, NULL, NULL);
    compare_array("up", got_up, want_up, routed_count,
                  2.0e-5f, 2.0e-5f, NULL, NULL);
    compare_array("mid", got_mid, want_mid, routed_count,
                  3.0e-5f, 3.0e-5f, NULL, NULL);
    compare_array("down", got_down, want_down, routed_count,
                  2.0e-4f, 2.0e-4f, NULL, NULL);
    compare_array("out", got_out, want_out, out_count,
                  3.0e-4f, 3.0e-4f, &worst_abs, &worst_rel);

    ds4_gpu_tensor_free(x_gpu);
    ds4_gpu_tensor_free(weights_gpu);
    ds4_gpu_tensor_free(selected_gpu);
    ds4_gpu_tensor_free(down);
    ds4_gpu_tensor_free(mid);
    ds4_gpu_tensor_free(up);
    ds4_gpu_tensor_free(gate);
    ds4_gpu_tensor_free(out);
    ds4_gpu_cleanup();
    free(got_out);
    free(got_down);
    free(got_mid);
    free(got_up);
    free(got_gate);
    free(want_out);
    free(want_down);
    free(want_mid);
    free(want_up);
    free(want_gate);
    free(model_raw);

    fprintf(stderr,
            "test_cuda_iq2_routed PASS tokens=%u experts=%u/%u dim=%u "
            "output_worst_abs=%g output_worst_rel=%g\n",
            N_TOKEN,
            N_USED,
            N_EXPERT,
            DIM,
            worst_abs,
            worst_rel);
    return 0;
}
