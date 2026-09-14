#define _DARWIN_C_SOURCE
#include "ds4_gpu.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

enum { D = 256, O = 260, E = 8, S = 3, STRIDE = 4 };
typedef struct { uint32_t type, block, values, ff; uint64_t gate, up, down; } format;
static uint32_t rng = 123;
static uint32_t random_u32(void) {
    rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
    return rng;
}
static uint64_t align_up(uint64_t n, uint64_t a) { return (n + a - 1) / a * a; }
static void half_at(uint8_t *p, uint16_t v) { memcpy(p, &v, sizeof(v)); }
static void fill(uint8_t *p, uint64_t bytes, const format *f) {
    for (uint64_t off = 0; off < bytes; off += f->block) {
        uint8_t *b = p + off;
        for (uint32_t j = 0; j < f->block; j++) b[j] = random_u32();
        const uint16_t scale = 0x1800u + (random_u32() % 5u) * 0x100u;
        if (f->type == 39) b[0] = 118u + random_u32() % 5u;
        else if (f->type == 10) { half_at(b + 80, scale); half_at(b + 82, 0x1400); }
        else { half_at(b, scale); if (f->type == 12) half_at(b + 2, 0x1400); }
    }
}

static int check(const void *map, uint64_t bytes, const format *f, uint32_t T) {
    const uint32_t F = f->ff;
    const uint64_t mid_n = (uint64_t)T * STRIDE * F, out_n = (uint64_t)T * STRIDE * O;
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc((uint64_t)T * D * sizeof(float));
    ds4_gpu_tensor *ids = ds4_gpu_tensor_alloc((uint64_t)T * S * sizeof(int32_t));
    ds4_gpu_tensor *lists = ds4_gpu_tensor_alloc((uint64_t)E * T * sizeof(int32_t));
    ds4_gpu_tensor *counts = ds4_gpu_tensor_alloc(E * sizeof(int32_t));
    ds4_gpu_tensor *mid = ds4_gpu_tensor_alloc(mid_n * sizeof(float));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(out_n * sizeof(float));
    float *host_x = malloc((uint64_t)T * D * sizeof(float));
    int32_t *host_ids = malloc((uint64_t)T * S * sizeof(int32_t));
    float *ref_mid = malloc(mid_n * sizeof(float)), *ref_out = malloc(out_n * sizeof(float));
    /* Padded Q2 cases have a wider intermediate than the output projection. */
    float *actual = malloc((mid_n > out_n ? mid_n : out_n) * sizeof(float));
    int ok = x && ids && lists && counts && mid && out && host_x && host_ids && ref_mid && ref_out && actual;
    if (!ok) goto done;
    for (uint64_t i = 0; i < (uint64_t)T * D; i++) host_x[i] = ((int)(random_u32() % 257) - 128) / 1024.0f;
    /* One hot expert, partial tiles, noncontiguous token/slot scatter, and an
     * empty expert. The fourth output slot must retain its poison value. */
    for (uint32_t t = 0; t < T; t++) {
        host_ids[t * S] = 0;
        host_ids[t * S + 1] = 1 + t % 3;
        host_ids[t * S + 2] = 4 + t % 3;
    }
    setenv("DS4_QWEN4_MOE_MM_NAX", "0", 1);   /* the knob matrix exercises the simdgroup tiles */
    ok = ds4_gpu_tensor_write(x, 0, host_x, (uint64_t)T * D * sizeof(float)) &&
         ds4_gpu_tensor_write(ids, 0, host_ids, (uint64_t)T * S * sizeof(int32_t)) &&
         ds4_gpu_qwen4_moe_build_lists_tensor(lists, counts, ids, T, S, E, T);
    for (int run = 0; run < 14 && ok; run++) {
        ok = setenv("DS4_QWEN4_MOE_MM_SPECIALIZE", (run > 0 && run < 6) || run == 7 || run == 8 ? "1" : "0", 1) == 0;
        const char *tiles[] = {"4", "1", "2", "4", "4", "2", "4", "4", "4", "4", "4", "4", "4", "4"};
        ok = ok && setenv("DS4_QWEN4_MOE_MID_NT", tiles[run], 1) == 0 &&
             setenv("DS4_QWEN4_MOE_DOWN_NT", run >= 7 ? "8" : tiles[run], 1) == 0;
        ok = ok && setenv("DS4_QWEN4_MOE_TAILS", run == 4 || run == 5 || run == 8 || run == 10 ? "1" : "0", 1) == 0;
        if (run >= 11) {
            unsetenv("DS4_QWEN4_MOE_DOWN_NT");
            if (run == 13) unsetenv("DS4_QWEN4_PREFILL_REUSE");
            else setenv("DS4_QWEN4_PREFILL_REUSE", run == 12 ? "1" : "0", 1);
        }
        ok = ok && ds4_gpu_tensor_fill_f32(mid, NAN, mid_n) &&
             ds4_gpu_tensor_fill_f32(out, NAN, out_n) &&
             ds4_gpu_begin_commands() &&
             ds4_gpu_qwen4_moe_mm_mid_tensor(mid, x, lists, counts, map, bytes, f->gate, f->up,
                                              f->type, E, T, S, STRIDE, D, F, T) &&
             ds4_gpu_qwen4_moe_mm_down_tensor(out, mid, lists, counts, map, bytes, f->down,
                                               f->type, E, T, S, STRIDE, F, O, T);
        if (!ds4_gpu_end_commands()) ok = 0;
        for (int stage = 0; stage < 2 && ok; stage++) {
            const uint64_t n = stage ? out_n : mid_n;
            const uint32_t width = stage ? O : F;
            float *ref = stage ? ref_out : ref_mid;
            ok = ds4_gpu_tensor_read(stage ? out : mid, 0, actual, n * sizeof(float));
            for (uint64_t i = 0; i < n && ok; i++) {
                const bool live = (i / width) % STRIDE < S;
                if ((live && !isfinite(actual[i])) || (!live && !isnan(actual[i]))) ok = 0;
            }
            if (ok && run == 0) memcpy(ref, actual, n * sizeof(float));
            else if (ok && memcmp(ref, actual, n * sizeof(float)) != 0) ok = 0;
            if (!ok) fprintf(stderr, "Qwen MoE specialization mismatch type=%u T=%u run=%d stage=%d\n", f->type, T, run, stage);
        }
    }
    if (ok) printf("PASS Qwen MoE specialization type=%u T=%u mid/down exact, padding intact\n", f->type, T);
done:
    unsetenv("DS4_QWEN4_PREFILL_REUSE");
    unsetenv("DS4_QWEN4_MOE_MM_SPECIALIZE");
    unsetenv("DS4_QWEN4_MOE_TAILS");
    unsetenv("DS4_QWEN4_MOE_MID_NT");
    unsetenv("DS4_QWEN4_MOE_DOWN_NT");
    ds4_gpu_tensor_free(x); ds4_gpu_tensor_free(ids); ds4_gpu_tensor_free(lists);
    ds4_gpu_tensor_free(counts); ds4_gpu_tensor_free(mid); ds4_gpu_tensor_free(out);
    free(host_x); free(host_ids); free(ref_mid); free(ref_out); free(actual);
    return ok;
}

int main(void) {
    format formats[] = {{12,144,256,256,0,0,0}, {16,66,256,256,0,0,0}, {10,84,256,640,0,0,0},
                        {8,34,32,256,0,0,0}, {39,17,32,256,0,0,0}, {2,18,32,256,0,0,0}};
    const uint32_t sizes[] = {8, 9, 16, 17, 24, 31, 32, 33, 48, 63, 64, 65, 95, 96, 97, 127, 128, 129, 257, 8191, 8192, 8193};
    const uint64_t page = (uint64_t)sysconf(_SC_PAGESIZE);
    uint64_t bytes = 0;
    for (unsigned i = 0; i < sizeof(formats) / sizeof(*formats); i++) {
        format *f = &formats[i];
        const uint64_t gu = (uint64_t)E * f->ff * (D / f->values) * f->block;
        const uint64_t dw = (uint64_t)E * O * ((f->ff + f->values - 1u) / f->values) * f->block;
        f->gate = bytes; f->up = align_up(bytes + gu, page);
        f->down = align_up(f->up + gu, page); bytes = align_up(f->down + dw, page);
    }
    void *map = NULL;
    if (posix_memalign(&map, page, bytes) != 0) return 1;
    memset(map, 0, bytes);
    for (unsigned i = 0; i < sizeof(formats) / sizeof(*formats); i++) {
        const format *f = &formats[i];
        const uint64_t gu = (uint64_t)E * f->ff * (D / f->values) * f->block;
        fill((uint8_t *)map + f->gate, gu, f); fill((uint8_t *)map + f->up, gu, f);
        fill((uint8_t *)map + f->down, (uint64_t)E * O * ((f->ff + f->values - 1u) / f->values) * f->block, f);
    }
    int ok = ds4_gpu_init() && ds4_gpu_set_model_map(map, bytes);
    /* Alternate formats in one engine to catch incorrectly keyed pipelines. */
    for (unsigned n = 0; n < sizeof(sizes) / sizeof(*sizes) && ok; n++)
        for (unsigned f = 0; f < sizeof(formats) / sizeof(*formats) && ok; f++)
            ok = check(map, bytes, &formats[f], sizes[n]);
    ds4_gpu_cleanup(); free(map);
    return ok ? 0 : 1;
}
