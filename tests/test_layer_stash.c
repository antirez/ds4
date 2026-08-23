#define _DARWIN_C_SOURCE

#include "ds4_gpu.h"

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define ROWS 37u
#define SLOTS 5u
#define WIDTH 13u

bool ds4_log_is_tty(FILE *fp) {
    (void)fp;
    return false;
}

float ds4_nvfp4_global_scale(uint64_t offset) {
    (void)offset;
    return 1.0f;
}

static float expected(uint32_t row, uint32_t slot, uint32_t col) {
    return (float)(slot * 100000u + row * 100u + col);
}

int main(void) {
    const uint64_t src_count = (uint64_t)ROWS * WIDTH;
    const uint64_t dst_count = src_count * SLOTS;
    float *src = malloc((size_t)src_count * sizeof(*src));
    float *got = malloc((size_t)dst_count * sizeof(*got));
    ds4_gpu_tensor *src_gpu[SLOTS] = {0};
    ds4_gpu_tensor *dst_gpu = NULL;
    int ok = src != NULL && got != NULL && ds4_gpu_init();

    if (ok) {
        dst_gpu = ds4_gpu_tensor_alloc(dst_count * sizeof(float));
        ok = dst_gpu != NULL;
    }
    for (uint32_t slot = 0; ok && slot < SLOTS; slot++) {
        src_gpu[slot] = ds4_gpu_tensor_alloc(src_count * sizeof(float));
        ok = src_gpu[slot] != NULL;
        for (uint32_t row = 0; row < ROWS; row++) {
            for (uint32_t col = 0; col < WIDTH; col++) {
                src[(uint64_t)row * WIDTH + col] = expected(row, slot, col);
            }
        }
        if (ok) {
            ok = ds4_gpu_tensor_write(
                src_gpu[slot], 0, src, src_count * sizeof(float));
        }
    }

    if (ok) ok = ds4_gpu_begin_commands();
    for (uint32_t slot = 0; ok && slot < SLOTS; slot++) {
        ok = ds4_gpu_interleave_rows_f32_tensor(
            dst_gpu, src_gpu[slot], WIDTH, ROWS, SLOTS, slot);
    }
    if (ok) ok = ds4_gpu_end_commands();
    if (ok) {
        ok = ds4_gpu_tensor_read(dst_gpu, 0, got, dst_count * sizeof(float));
    }

    uint64_t checked = 0;
    for (uint32_t row = 0; ok && row < ROWS; row++) {
        for (uint32_t slot = 0; ok && slot < SLOTS; slot++) {
            for (uint32_t col = 0; col < WIDTH; col++) {
                const uint64_t index =
                    ((uint64_t)row * SLOTS + slot) * WIDTH + col;
                const float want = expected(row, slot, col);
                if (fabsf(got[index] - want) > 0.0f) {
                    fprintf(stderr,
                            "layer stash mismatch row=%u slot=%u col=%u got=%.1f want=%.1f\n",
                            row, slot, col, got[index], want);
                    ok = 0;
                    break;
                }
                checked++;
            }
        }
    }

    if (ok && ds4_gpu_interleave_rows_f32_tensor(
            dst_gpu, src_gpu[0], WIDTH, ROWS, SLOTS, SLOTS)) {
        fprintf(stderr, "layer stash accepted out-of-range slot\n");
        ok = 0;
    }

    for (uint32_t slot = 0; slot < SLOTS; slot++) {
        ds4_gpu_tensor_free(src_gpu[slot]);
    }
    ds4_gpu_tensor_free(dst_gpu);
    ds4_gpu_cleanup();
    free(got);
    free(src);

    if (!ok) return 1;
    printf("layer stash interleave PASS rows=%u slots=%u width=%u values=%llu\n",
           ROWS, SLOTS, WIDTH, (unsigned long long)checked);
    return 0;
}
