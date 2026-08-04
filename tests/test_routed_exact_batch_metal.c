#define _DARWIN_C_SOURCE

#include "ds4_gpu.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define ROWS 5u
#define SLOTS 6u
#define N_TOTAL_EXPERT 8u
#define IN_DIM 4096u
#define MID_DIM 2048u
#define OUT_DIM 4096u

typedef struct {
    const char *name;
    uint32_t gate_type;
    uint32_t down_type;
    uint32_t gate_block_width;
    uint32_t gate_block_bytes;
    uint32_t down_block_width;
    uint32_t down_block_bytes;
    uint32_t gate_scale_bytes;
    uint32_t down_scale_bytes;
} quant_case;

static const quant_case quant_cases[] = {
    { "MXFP4/MXFP4",  39u, 39u, 32u, 17u, 32u, 17u, 1u, 1u },
    { "Q4_K/Q4_K",    12u, 12u, 256u, 144u, 256u, 144u, 4u, 4u },
};

bool ds4_log_is_tty(FILE *fp) {
    (void)fp;
    return false;
}

static uint64_t align_up(uint64_t value, uint64_t alignment) {
    return (value + alignment - 1u) / alignment * alignment;
}

static uint64_t quant_row_bytes(uint32_t dim,
                                uint32_t block_width,
                                uint32_t block_bytes) {
    return (dim / block_width) * (uint64_t)block_bytes;
}

static void fill_quant_tensor(void *base,
                              uint32_t matrix_rows,
                              uint64_t row_bytes,
                              uint32_t block_bytes,
                              uint32_t scale_bytes,
                              bool mxfp4,
                              uint32_t salt) {
    uint8_t *bytes = base;
    for (uint32_t expert = 0; expert < N_TOTAL_EXPERT; expert++) {
        for (uint32_t row = 0; row < matrix_rows; row++) {
            uint8_t *row_ptr = bytes +
                ((uint64_t)expert * matrix_rows + row) * row_bytes;
            for (uint64_t block = 0;
                 block < row_bytes / block_bytes;
                 block++) {
                uint8_t *dst = row_ptr + block * block_bytes;
                const uint32_t seed = salt + expert * 29u + row * 17u +
                    (uint32_t)block * 13u;
                if (mxfp4) {
                    dst[0] = (uint8_t)(120u + (seed & 3u));
                } else {
                    const _Float16 d = (_Float16)(1.0f / 2048.0f);
                    const uint32_t scale_offset =
                        block_bytes == 84u ? block_bytes - 4u : 0u;
                    memcpy(dst + scale_offset, &d, sizeof(d));
                    if (scale_bytes == 4u) {
                        const _Float16 dmin =
                            (_Float16)(1.0f / 4096.0f);
                        memcpy(dst + scale_offset + sizeof(d),
                               &dmin,
                               sizeof(dmin));
                    }
                }
                for (uint32_t i = 0; i < block_bytes; i++) {
                    const uint32_t scale_offset =
                        block_bytes == 84u ? block_bytes - 4u : 0u;
                    if (i >= scale_offset &&
                        i < scale_offset + scale_bytes) {
                        continue;
                    }
                    dst[i] = (uint8_t)(seed + i * 7u);
                }
            }
        }
    }
}

static int clear_tensor(ds4_gpu_tensor *tensor, uint64_t bytes) {
    void *zero = calloc(1, (size_t)bytes);
    if (!zero) return 0;
    const int ok = ds4_gpu_tensor_write(tensor, 0, zero, bytes);
    free(zero);
    return ok;
}

static int compare_bits(const char *case_name,
                        const char *tensor_name,
                        const float *actual,
                        const float *expected,
                        uint64_t count) {
    for (uint64_t i = 0; i < count; i++) {
        if (memcmp(actual + i, expected + i, sizeof(float)) != 0) {
            uint32_t a = 0;
            uint32_t e = 0;
            memcpy(&a, actual + i, sizeof(a));
            memcpy(&e, expected + i, sizeof(e));
            fprintf(stderr,
                    "exact routed batch %s %s mismatch at %llu: "
                    "0x%08x vs 0x%08x\n",
                    case_name,
                    tensor_name,
                    (unsigned long long)i,
                    a,
                    e);
            return 0;
        }
    }
    return 1;
}

static int execute_serial_rows(
        ds4_gpu_tensor *out,
        ds4_gpu_tensor *gate,
        ds4_gpu_tensor *up,
        ds4_gpu_tensor *mid,
        ds4_gpu_tensor *experts,
        ds4_gpu_tensor *selected,
        ds4_gpu_tensor *weights,
        ds4_gpu_tensor *x,
        const void *model,
        uint64_t model_size,
        uint64_t gate_offset,
        uint64_t up_offset,
        uint64_t down_offset,
        const quant_case *quant,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes) {
    const uint64_t pair_row_bytes =
        (uint64_t)SLOTS * MID_DIM * sizeof(float);
    const uint64_t down_row_values = (uint64_t)SLOTS * OUT_DIM;
    const uint64_t down_row_values_bytes =
        down_row_values * sizeof(float);
    const uint64_t out_row_bytes = (uint64_t)OUT_DIM * sizeof(float);
    int ok = ds4_gpu_begin_commands();
    for (uint32_t row = 0; ok && row < ROWS; row++) {
        ds4_gpu_tensor *out_row = ds4_gpu_tensor_view(
                out, (uint64_t)row * out_row_bytes, out_row_bytes);
        ds4_gpu_tensor *gate_row = ds4_gpu_tensor_view(
                gate, (uint64_t)row * pair_row_bytes, pair_row_bytes);
        ds4_gpu_tensor *up_row = ds4_gpu_tensor_view(
                up, (uint64_t)row * pair_row_bytes, pair_row_bytes);
        ds4_gpu_tensor *mid_row = ds4_gpu_tensor_view(
                mid, (uint64_t)row * pair_row_bytes, pair_row_bytes);
        ds4_gpu_tensor *experts_row = ds4_gpu_tensor_view(
                experts,
                (uint64_t)row * down_row_values_bytes,
                down_row_values_bytes);
        ds4_gpu_tensor *selected_row = ds4_gpu_tensor_view(
                selected,
                (uint64_t)row * SLOTS * sizeof(int32_t),
                SLOTS * sizeof(int32_t));
        ds4_gpu_tensor *weights_row = ds4_gpu_tensor_view(
                weights,
                (uint64_t)row * SLOTS * sizeof(float),
                SLOTS * sizeof(float));
        ds4_gpu_tensor *x_row = ds4_gpu_tensor_view(
                x,
                (uint64_t)row * IN_DIM * sizeof(float),
                IN_DIM * sizeof(float));
        ok = out_row && gate_row && up_row && mid_row && experts_row &&
             selected_row && weights_row && x_row &&
             ds4_gpu_routed_moe_one_tensor(
                     out_row,
                     gate_row,
                     up_row,
                     mid_row,
                     experts_row,
                     model,
                     model_size,
                     gate_offset,
                     up_offset,
                     down_offset,
                     quant->gate_type,
                     quant->down_type,
                     gate_expert_bytes,
                     gate_row_bytes,
                     down_expert_bytes,
                     down_row_bytes,
                     IN_DIM,
                     MID_DIM,
                     OUT_DIM,
                     selected_row,
                     weights_row,
                     N_TOTAL_EXPERT,
                     SLOTS,
                     7.0f,
                     x_row,
                     NULL,
                     0u,
                     false);
        ds4_gpu_tensor_free(x_row);
        ds4_gpu_tensor_free(weights_row);
        ds4_gpu_tensor_free(selected_row);
        ds4_gpu_tensor_free(experts_row);
        ds4_gpu_tensor_free(mid_row);
        ds4_gpu_tensor_free(up_row);
        ds4_gpu_tensor_free(gate_row);
        ds4_gpu_tensor_free(out_row);
    }
    return ds4_gpu_end_commands() && ok;
}

static int execute_exact_batch(
        ds4_gpu_tensor *out,
        ds4_gpu_tensor *gate,
        ds4_gpu_tensor *up,
        ds4_gpu_tensor *mid,
        ds4_gpu_tensor *experts,
        ds4_gpu_tensor *selected,
        ds4_gpu_tensor *weights,
        ds4_gpu_tensor *x,
        const void *model,
        uint64_t model_size,
        uint64_t gate_offset,
        uint64_t up_offset,
        uint64_t down_offset,
        const quant_case *quant,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes) {
    bool mid_is_f16 = true;
    int ok = ds4_gpu_begin_commands();
    ok = ok && ds4_gpu_routed_moe_batch_tensor(
            out,
            gate,
            up,
            mid,
            experts,
            model,
            model_size,
            gate_offset,
            up_offset,
            down_offset,
            quant->gate_type,
            quant->down_type,
            gate_expert_bytes,
            gate_row_bytes,
            down_expert_bytes,
            down_row_bytes,
            IN_DIM,
            MID_DIM,
            OUT_DIM,
            selected,
            weights,
            N_TOTAL_EXPERT,
            SLOTS,
            7.0f,
            x,
            0u,
            ROWS,
            true,
            &mid_is_f16,
            false);
    ok = ds4_gpu_end_commands() && ok && !mid_is_f16;
    return ok;
}

static int run_case(const quant_case *quant, void **model_keepalive) {
    const uint64_t page = (uint64_t)getpagesize();
    const uint64_t gate_row_bytes = quant_row_bytes(
            IN_DIM, quant->gate_block_width, quant->gate_block_bytes);
    const uint64_t down_row_bytes = quant_row_bytes(
            MID_DIM, quant->down_block_width, quant->down_block_bytes);
    const uint64_t gate_expert_bytes = MID_DIM * gate_row_bytes;
    const uint64_t down_expert_bytes = OUT_DIM * down_row_bytes;
    const uint64_t gate_tensor_bytes =
        N_TOTAL_EXPERT * gate_expert_bytes;
    const uint64_t down_tensor_bytes =
        N_TOTAL_EXPERT * down_expert_bytes;
    const uint64_t gate_offset = 0;
    const uint64_t up_offset = align_up(gate_tensor_bytes, page);
    const uint64_t down_offset =
        align_up(up_offset + gate_tensor_bytes, page);
    const uint64_t model_size =
        align_up(down_offset + down_tensor_bytes, page);
    void *model = NULL;
    if (posix_memalign(&model, (size_t)page, (size_t)model_size) != 0) {
        return 0;
    }
    *model_keepalive = model;
    memset(model, 0, (size_t)model_size);
    fill_quant_tensor((uint8_t *)model + gate_offset,
                      MID_DIM,
                      gate_row_bytes,
                      quant->gate_block_bytes,
                      quant->gate_scale_bytes,
                      quant->gate_type == 39u,
                      3u);
    fill_quant_tensor((uint8_t *)model + up_offset,
                      MID_DIM,
                      gate_row_bytes,
                      quant->gate_block_bytes,
                      quant->gate_scale_bytes,
                      quant->gate_type == 39u,
                      71u);
    fill_quant_tensor((uint8_t *)model + down_offset,
                      OUT_DIM,
                      down_row_bytes,
                      quant->down_block_bytes,
                      quant->down_scale_bytes,
                      quant->down_type == 39u,
                      139u);
    if (!ds4_gpu_set_model_map(model, model_size)) return 0;

    const uint64_t pair_elems = (uint64_t)ROWS * SLOTS * MID_DIM;
    const uint64_t pair_bytes = pair_elems * sizeof(float);
    const uint64_t down_elems = (uint64_t)ROWS * SLOTS * OUT_DIM;
    const uint64_t down_bytes = down_elems * sizeof(float);
    const uint64_t out_elems = (uint64_t)ROWS * OUT_DIM;
    const uint64_t out_bytes = out_elems * sizeof(float);
    float x[ROWS * IN_DIM];
    int32_t selected[ROWS * SLOTS];
    float weights[ROWS * SLOTS];
    static const int32_t ids[ROWS][SLOTS] = {
        { 0, 1, 2, 3, 4, 5 },
        { 0, 2, 3, 4, 5, 6 },
        { 0, 1, 3, 5, 6, 7 },
        { 1, 2, 3, 4, 6, 7 },
        { 0, 2, 4, 5, 6, 7 },
    };
    for (uint32_t row = 0; row < ROWS; row++) {
        for (uint32_t i = 0; i < IN_DIM; i++) {
            x[row * IN_DIM + i] =
                (float)((int32_t)((i * 13u + row * 19u) % 61u) - 30) /
                (256.0f + 16.0f * row);
        }
        for (uint32_t slot = 0; slot < SLOTS; slot++) {
            selected[row * SLOTS + slot] = ids[row][slot];
            weights[row * SLOTS + slot] =
                (float)(slot + 1u + row) / 43.0f;
        }
    }

    ds4_gpu_tensor *x_tensor = ds4_gpu_tensor_alloc(sizeof(x));
    ds4_gpu_tensor *selected_tensor = ds4_gpu_tensor_alloc(sizeof(selected));
    ds4_gpu_tensor *weights_tensor = ds4_gpu_tensor_alloc(sizeof(weights));
    ds4_gpu_tensor *gate_tensor = ds4_gpu_tensor_alloc(pair_bytes);
    ds4_gpu_tensor *up_tensor = ds4_gpu_tensor_alloc(pair_bytes);
    ds4_gpu_tensor *mid_tensor = ds4_gpu_tensor_alloc(pair_bytes);
    ds4_gpu_tensor *experts_tensor = ds4_gpu_tensor_alloc(down_bytes);
    ds4_gpu_tensor *out_tensor = ds4_gpu_tensor_alloc(out_bytes);
    float *gate_ref = malloc((size_t)pair_bytes);
    float *up_ref = malloc((size_t)pair_bytes);
    float *mid_ref = malloc((size_t)pair_bytes);
    float *out_ref = malloc((size_t)out_bytes);
    float *gate_batch = malloc((size_t)pair_bytes);
    float *up_batch = malloc((size_t)pair_bytes);
    float *mid_batch = malloc((size_t)pair_bytes);
    float *out_batch = malloc((size_t)out_bytes);
    int ok = x_tensor && selected_tensor && weights_tensor && gate_tensor &&
        up_tensor && mid_tensor && experts_tensor && out_tensor &&
        gate_ref && up_ref && mid_ref && out_ref &&
        gate_batch && up_batch && mid_batch && out_batch;
    ok = ok && ds4_gpu_tensor_write(x_tensor, 0, x, sizeof(x)) &&
        ds4_gpu_tensor_write(
                selected_tensor, 0, selected, sizeof(selected)) &&
        ds4_gpu_tensor_write(
                weights_tensor, 0, weights, sizeof(weights));

    ok = ok && clear_tensor(gate_tensor, pair_bytes) &&
        clear_tensor(up_tensor, pair_bytes) &&
        clear_tensor(mid_tensor, pair_bytes) &&
        clear_tensor(experts_tensor, down_bytes) &&
        clear_tensor(out_tensor, out_bytes) &&
        execute_serial_rows(
                out_tensor, gate_tensor, up_tensor, mid_tensor,
                experts_tensor, selected_tensor, weights_tensor, x_tensor,
                model, model_size, gate_offset, up_offset, down_offset,
                quant, gate_expert_bytes, gate_row_bytes,
                down_expert_bytes, down_row_bytes) &&
        ds4_gpu_tensor_read(gate_tensor, 0, gate_ref, pair_bytes) &&
        ds4_gpu_tensor_read(up_tensor, 0, up_ref, pair_bytes) &&
        ds4_gpu_tensor_read(mid_tensor, 0, mid_ref, pair_bytes) &&
        ds4_gpu_tensor_read(out_tensor, 0, out_ref, out_bytes);

    ok = ok && clear_tensor(gate_tensor, pair_bytes) &&
        clear_tensor(up_tensor, pair_bytes) &&
        clear_tensor(mid_tensor, pair_bytes) &&
        clear_tensor(experts_tensor, down_bytes) &&
        clear_tensor(out_tensor, out_bytes) &&
        execute_exact_batch(
                out_tensor, gate_tensor, up_tensor, mid_tensor,
                experts_tensor, selected_tensor, weights_tensor, x_tensor,
                model, model_size, gate_offset, up_offset, down_offset,
                quant, gate_expert_bytes, gate_row_bytes,
                down_expert_bytes, down_row_bytes) &&
        ds4_gpu_tensor_read(gate_tensor, 0, gate_batch, pair_bytes) &&
        ds4_gpu_tensor_read(up_tensor, 0, up_batch, pair_bytes) &&
        ds4_gpu_tensor_read(mid_tensor, 0, mid_batch, pair_bytes) &&
        ds4_gpu_tensor_read(out_tensor, 0, out_batch, out_bytes);

    ok = ok && compare_bits(
            quant->name, "gate", gate_batch, gate_ref, pair_elems) &&
        compare_bits(quant->name, "up", up_batch, up_ref, pair_elems) &&
        compare_bits(quant->name, "mid", mid_batch, mid_ref, pair_elems) &&
        compare_bits(quant->name, "out", out_batch, out_ref, out_elems);
    fprintf(stderr, "exact routed batch %-14s: %s\n",
            quant->name, ok ? "PASS" : "FAIL");

    free(out_batch);
    free(mid_batch);
    free(up_batch);
    free(gate_batch);
    free(out_ref);
    free(mid_ref);
    free(up_ref);
    free(gate_ref);
    ds4_gpu_tensor_free(out_tensor);
    ds4_gpu_tensor_free(experts_tensor);
    ds4_gpu_tensor_free(mid_tensor);
    ds4_gpu_tensor_free(up_tensor);
    ds4_gpu_tensor_free(gate_tensor);
    ds4_gpu_tensor_free(weights_tensor);
    ds4_gpu_tensor_free(selected_tensor);
    ds4_gpu_tensor_free(x_tensor);
    return ok;
}

int main(void) {
    unsetenv("DS4_METAL_FORCE_TINY_MOE_MM_ID");
    unsetenv("DS4_METAL_MOE_WRITE_CLAMPED_ACT");
    int ok = ds4_gpu_init();
    ds4_gpu_set_quality(false);
    ds4_gpu_set_ssd_streaming(false);

    void *models[sizeof(quant_cases) / sizeof(quant_cases[0])] = { NULL };
    for (uint32_t i = 0;
         ok && i < sizeof(quant_cases) / sizeof(quant_cases[0]);
         i++) {
        ok = run_case(&quant_cases[i], &models[i]);
    }

    ds4_gpu_cleanup();
    for (uint32_t i = 0;
         i < sizeof(models) / sizeof(models[0]);
         i++) {
        free(models[i]);
    }
    fprintf(stderr, "Metal exact routed batch: %s\n",
            ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
