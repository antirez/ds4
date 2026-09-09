#define _DARWIN_C_SOURCE
#include "ds4_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

enum { INPUT = 1024, OUTPUT = 128, ROWS = 128 };
typedef struct { uint16_t d; int8_t qs[32]; } q8_block;
typedef struct { uint16_t d; uint8_t qs[16]; } q4_block;
typedef struct { uint16_t d, dmin; uint8_t scales[12], qs[128]; } q4k_block;

static int run_type(uint32_t type) {
    const uint64_t row_bytes = type == 8 ? INPUT/32*sizeof(q8_block) :
        type == 2 ? INPUT/32*sizeof(q4_block) : INPUT/256*sizeof(q4k_block);
    const uint64_t page = getpagesize();
    const uint64_t model_size = (row_bytes*OUTPUT + page - 1)/page*page;
    void *model = NULL;
    float *weights = malloc((size_t)INPUT*OUTPUT*sizeof(float));
    float *x = malloc((size_t)INPUT*ROWS*sizeof(float));
    float *out = malloc((size_t)OUTPUT*ROWS*sizeof(float));
    if (posix_memalign(&model, page, model_size) || !weights || !x || !out) return 0;
    memset(model, 0, model_size);
    for (int r = 0; r < OUTPUT; r++) {
        for (int k = 0; k < INPUT; k++) {
            const unsigned q = (r*7 + k*13 + k/32) & 15;
            if (type == 8) {
                q8_block *b = (q8_block *)((char *)model + r*row_bytes) + k/32;
                b->d = 0x2000;
                b->qs[k%32] = (int)q - 8;
            } else if (type == 2) {
                q4_block *b = (q4_block *)((char *)model + r*row_bytes) + k/32;
                b->d = 0x2000;
                b->qs[k%16] |= q << (k%32 < 16 ? 0 : 4);
            } else {
                q4k_block *b = (q4k_block *)((char *)model + r*row_bytes) + k/256;
                b->d = 0x2000;
                memset(b->scales, 1, sizeof(b->scales));
                b->qs[(k%256)/64*32 + k%32] |= q << (k%64 < 32 ? 0 : 4);
            }
            weights[r*INPUT + k] = ((int)q - (type == 12 ? 0 : 8))/128.0f;
        }
    }
    for (int i = 0; i < INPUT*ROWS; i++) x[i] = (i*17%31 - 15)/64.0f;
    int ok = ds4_gpu_set_model_map(model, model_size);
    ds4_gpu_tensor *xt = ds4_gpu_tensor_alloc((uint64_t)INPUT*ROWS*sizeof(float));
    ds4_gpu_tensor *ot = ds4_gpu_tensor_alloc((uint64_t)OUTPUT*ROWS*sizeof(float));
    ok = ok && xt && ot && ds4_gpu_tensor_write(xt, 0, x, (uint64_t)INPUT*ROWS*sizeof(float));
    const int sizes[] = {1, 2, 3, 4, 5, 32, 64, 128};
    for (unsigned s = 0; s < sizeof(sizes)/sizeof(*sizes) && ok; s++) {
        const int rows = sizes[s];
        for (int repeat = 0; repeat < 3 && ok; repeat++) {
            ok = ds4_gpu_tensor_fill_f32(ot, NAN, (uint64_t)OUTPUT*ROWS);
            if (type == 8) {
                ok = ok && ds4_gpu_matmul_q8_0_decode_mpp_tensor(
                    ot, model, model_size, 0, INPUT, OUTPUT, xt, rows);
            } else {
                ok = ok && ds4_gpu_matmul_quant_tensor(
                    ot, model, model_size, 0, type, INPUT, OUTPUT, xt, rows);
            }
            ok = ok && ds4_gpu_tensor_read(ot, 0, out, (uint64_t)OUTPUT*ROWS*sizeof(float));
            for (int t = 0; t < rows && ok; t++) {
                for (int r = 0; r < OUTPUT; r++) {
                    float expected = 0;
                    for (int k = 0; k < INPUT; k++) expected += weights[r*INPUT + k]*x[t*INPUT + k];
                    if (!isfinite(out[t*OUTPUT + r]) || out[t*OUTPUT + r] != expected) {
                        fprintf(stderr, "dense MPP type=%u rows=%d repeat=%d t=%d r=%d got=%g expected=%g\n",
                            type, rows, repeat, t, r, out[t*OUTPUT + r], expected);
                        ok = 0;
                        break;
                    }
                }
            }
            for (int i = rows*OUTPUT; i < ROWS*OUTPUT && ok; i++)
                if (!isnan(out[i])) ok = 0;
        }
        fprintf(stderr, "dense MPP type=%u rows=%d: %s\n", type, rows, ok ? "PASS" : "FAIL");
    }
    ds4_gpu_tensor_free(ot);
    ds4_gpu_tensor_free(xt);
    ds4_gpu_cleanup();
    free(out);
    free(x);
    free(weights);
    free(model);
    return ok;
}

int main(void) {
    const unsigned types[] = {8, 2, 12};
    for (unsigned i = 0; i < sizeof(types)/sizeof(*types); i++) {
        if (!ds4_gpu_init()) return 1;
        if (!ds4_gpu_device_is_m5_apple_silicon()) {
            fprintf(stderr, "dense MPP: skipped (requires M5 tensor cores)\n");
            ds4_gpu_cleanup();
            return 0;
        }
        ds4_gpu_set_quality(false);
        if (!run_type(types[i])) return 1;
    }
    return 0;
}
