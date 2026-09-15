#define _DARWIN_C_SOURCE
#include "ds4_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
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

static uint32_t rng = 41;
static float random_float(void) {
    rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
    return ((int32_t)(rng & 0xffffffu) - 0x800000) / 8388608.0f;
}

static double seconds(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9 * t.tv_nsec;
}

static int run_router(void) {
    enum { K = 5120, M = 384, CAP = 8192 };
    const size_t bytes = (size_t)CAP * M * sizeof(float);
    const size_t xbytes = (size_t)CAP * K * sizeof(float);
    float *w = NULL, *x = malloc(xbytes), *ref = malloc(bytes), *out = malloc(bytes);
    if (posix_memalign((void **)&w, getpagesize(), K*M*sizeof(float)) || !x || !ref || !out) return 0;
    for (unsigned i = 0; i < K*M; i++) w[i] = random_float() * .03f;
    for (size_t i = 0; i < xbytes / sizeof(float); i++) x[i] = random_float();
    int ok = ds4_gpu_set_model_map(w, K*M*sizeof(float));
    ds4_gpu_tensor *xt = ds4_gpu_tensor_alloc(xbytes);
    ds4_gpu_tensor *yt = ds4_gpu_tensor_alloc(bytes + sizeof(float));
    ok = ok && xt && yt && ds4_gpu_tensor_write(xt, 0, x, xbytes);
    const unsigned sizes[] = {1, 8, 31, 32, 33, 128, 255, 256, 257, 511, 512, 513, 2048, 4096, 8192};
    for (unsigned si = 0; si < sizeof(sizes)/sizeof(*sizes) && ok; si++) {
        const unsigned rows = sizes[si];
        const unsigned modes[] = {0, 1, 2, 2, 1, 0};
        for (unsigned pass = 0; pass < sizeof(modes)/sizeof(*modes) && ok; pass++) {
            const unsigned mode = modes[pass];
            if (!mode) setenv("DS4_METAL_DISABLE_V41_ROUTER_BATCH", "1", 1);
            else unsetenv("DS4_METAL_DISABLE_V41_ROUTER_BATCH");
            if (mode == 1) setenv("DS4_METAL_DISABLE_V41_ROUTER_MM", "1", 1);
            else unsetenv("DS4_METAL_DISABLE_V41_ROUTER_MM");
            ok = ds4_gpu_tensor_fill_f32(yt, 12345, (size_t)CAP*M + 1);
            double elapsed = 0;
            for (unsigned repeat = 0; repeat < 4 && ok; repeat++) {
                const double begin = seconds();
                ok = ds4_gpu_matmul_f32_tensor(yt, w, K*M*sizeof(float), 0, K, M, xt, rows) &&
                    ds4_gpu_synchronize();
                if (repeat) elapsed += (seconds() - begin) / 3;
            }
            ok = ok && ds4_gpu_tensor_read(yt, 0, out, bytes);
            float guard = 0;
            ok = ok && ds4_gpu_tensor_read(yt, (uint64_t)rows*M*sizeof(float), &guard, sizeof(guard)) && guard == 12345;
            if (!pass) memcpy(ref, out, (size_t)rows*M*sizeof(float));
            double worst = 0;
            for (size_t i = 0; i < (size_t)rows*M && ok; i++) {
                ok = isfinite(out[i]) && fabs(out[i] - ref[i]) < 1e-4 * (1 + fabs(ref[i]));
                worst = fmax(worst, fabs(out[i] - ref[i]));
            }
            double oracle = 0;
            for (unsigned sample = 0; sample < 32 && ok; sample++) {
                const unsigned r = (sample * 79u) % rows, m = (sample * 13u) % M;
                double expected = 0;
                for (unsigned k = 0; k < K; k++) expected += (double)w[(size_t)m*K + k] * x[(size_t)r*K + k];
                const double error = fabs(out[(size_t)r*M + m] - expected);
                oracle = fmax(oracle, error);
                ok = error < 4e-6 * (1 + fabs(expected));
                if (!ok) fprintf(stderr, "Router oracle mode=%u row=%u col=%u actual=%.9g expected=%.9g\n",
                    mode, r, m, out[(size_t)r*M + m], expected);
            }
            fprintf(stderr, "F32 router rows=%u mode=%u pass=%u %.3f ms max_delta=%.9g oracle=%.9g: %s\n",
                rows, mode, pass, elapsed*1000, worst, oracle, ok ? "PASS" : "FAIL");
        }
    }
    unsetenv("DS4_METAL_DISABLE_V41_ROUTER_BATCH");
    unsetenv("DS4_METAL_DISABLE_V41_ROUTER_MM");
    ds4_gpu_tensor_free(yt); ds4_gpu_tensor_free(xt); ds4_gpu_cleanup();
    free(w); free(x); free(ref); free(out);
    return ok;
}

int main(int argc, char **argv) {
    if (argc == 2 && !strcmp(argv[1], "--router")) {
        if (!ds4_gpu_init()) return 1;
        ds4_gpu_set_quality(false);
        return run_router() ? 0 : 1;
    }
    if (argc != 1) {
        fprintf(stderr, "usage: %s [--router]\n", argv[0]);
        return 1;
    }
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
