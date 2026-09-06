#define _DARWIN_C_SOURCE
#include "ds4_gpu.h"

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

bool ds4_log_is_tty(FILE *fp) { (void)fp; return false; }

enum { HEADS = 64, DIM = 512, TOKENS = 576, COMP = TOKENS / 4 };

int main(void) {
    const char *disable = "DS4_METAL_DISABLE_PRE_M5_STATIC_MIXED_BLOCK_SKIP";
    const char *require = "DS4_METAL_REQUIRE_PRE_M5_STATIC_MIXED_BLOCK_SKIP";
    if (unsetenv(disable) || unsetenv(require) ||
        !ds4_gpu_init()) return 1;
    if (!ds4_gpu_device_is_pre_m5_apple_silicon()) {
        fprintf(stderr, "static-mixed block skip oracle requires pre-M5 Apple Silicon\n");
        ds4_gpu_cleanup();
        return 77;
    }

    const size_t q_bytes = (size_t)TOKENS * HEADS * DIM * sizeof(float);
    const size_t raw_bytes = TOKENS * DIM * sizeof(float);
    const size_t comp_bytes = COMP * DIM * sizeof(_Float16);
    const size_t model_bytes = (size_t)sysconf(_SC_PAGESIZE);
    void *model = NULL;
    float *q = malloc(q_bytes);
    float *raw = malloc(raw_bytes);
    _Float16 *comp = malloc(comp_bytes);
    float *reference = malloc(q_bytes);
    float *actual = malloc(q_bytes);
    ds4_gpu_tensor *qt = ds4_gpu_tensor_alloc(q_bytes);
    ds4_gpu_tensor *rawt = ds4_gpu_tensor_alloc(raw_bytes);
    ds4_gpu_tensor *compt = ds4_gpu_tensor_alloc(comp_bytes);
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(q_bytes);
    int ok = posix_memalign(&model, model_bytes, model_bytes) == 0 &&
        q && raw && comp && reference && actual && qt && rawt && compt && out;
    if (!ok) goto cleanup;
    memset(model, 0, model_bytes);
    for (size_t i = 0; i < q_bytes / sizeof(float); ++i)
        q[i] = (float)((int)((i * 29u + 13u) % 251u) - 125) / 64.0f;
    for (unsigned i = 0; i < TOKENS * DIM; ++i)
        raw[i] = (float)((int)((i * 53u + 7u) % 509u) - 254) / 127.0f;
    for (unsigned i = 0; i < COMP * DIM; ++i)
        comp[i] = (_Float16)((float)((int)((i * 73u + 19u) % 503u) - 251) / 123.0f);
    for (unsigned h = 0; h < HEADS; ++h)
        ((float *)model)[h] = (float)((int)(h % 3u) - 1) * 80.0f;
    ds4_gpu_set_quality(false);
    ok = ds4_gpu_set_model_map(model, model_bytes) &&
         ds4_gpu_tensor_write(qt, 0, q, q_bytes) &&
         ds4_gpu_tensor_write(rawt, 0, raw, raw_bytes) &&
         ds4_gpu_tensor_write(compt, 0, comp, comp_bytes);

    /* Both promoted ratios cover raw-window transitions, raw/compressed
     * gaps, and compressed visibility stops. The 576-token cases pad the
     * terminal block for both ratios, with an incomplete final ratio128
     * group. Smaller and unsupported token/window shapes keep fallback. */
    const struct { uint32_t tokens, ratio, window; } shapes[] = {
        {64, 4, 128}, {64, 128, 128},
        {128, 4, 128}, {128, 128, 128},
        {192, 4, 128}, {256, 4, 128},
        {512, 4, 128}, {512, 128, 128},
        {576, 4, 128}, {576, 128, 128},
        {129, 4, 128}, {256, 4, 63},
    };
    const size_t shape_count = sizeof(shapes) / sizeof(shapes[0]);
    uint64_t compared = 0;
    for (size_t shape = 0; ok && shape < shape_count; ++shape) {
        const uint32_t n = shapes[shape].tokens;
        const uint32_t ratio = shapes[shape].ratio;
        const uint32_t window = shapes[shape].window;
        const size_t count = (size_t)n * HEADS * DIM;
        const size_t bytes = count * sizeof(float);
        ok = (n >= 512 && n <= 2048 && n % 64 == 0 && window == 128
              ? setenv(require, "1", 1) :
              unsetenv(require)) == 0;
        for (unsigned variant = 0; ok && variant < 3; ++variant) {
            /* Baseline and rollback must explicitly disable the default-on
             * specialization; disabled oracle runs also bypass REQUIRE. */
            ok = (variant == 1 ? unsetenv(disable) : setenv(disable, "1", 1)) == 0;
            ok = ok && ds4_gpu_tensor_fill_f32(out, NAN, count) &&
                ds4_gpu_attention_prefill_static_mixed_heads_tensor(
                    out, model, model_bytes, 0, qt, rawt, compt, 1,
                    n, n / ratio, window, ratio, HEADS, DIM) &&
                ds4_gpu_tensor_read(out, 0, actual, bytes);
            for (size_t i = 0; ok && i < count; ++i) {
                /* The project builds with -ffast-math: test exponent bits so
                 * an unwritten NaN poison cannot pass an optimized-out check. */
                uint32_t bits;
                memcpy(&bits, &actual[i], sizeof(bits));
                if ((bits & 0x7f800000u) == 0x7f800000u) ok = 0;
            }
            if (ok && variant == 0) memcpy(reference, actual, bytes);
            if (ok && variant != 0) {
                ok = memcmp(reference, actual, bytes) == 0;
                compared += count;
            }
            if (!ok) fprintf(stderr,
                "static-mixed attention failed tokens=%u ratio=%u window=%u variant=%u\n",
                n, ratio, window, variant);
        }
    }
    /* Valid, already-allocated 256-token buffers prove the lower scope gate
     * rejects REQUIRE, instead of silently selecting the generic pipeline. */
    if (ok) {
        ok = unsetenv(disable) == 0 && setenv(require, "1", 1) == 0 &&
            !ds4_gpu_attention_prefill_static_mixed_heads_tensor(
                out, model, model_bytes, 0, qt, rawt, compt, 1,
                256, 64, 128, 4, HEADS, DIM);
        if (!ok) fprintf(stderr, "static-mixed attention failed excluded 256-token scope\n");
    }
    unsetenv(disable);
    unsetenv(require);
    if (ok) fprintf(stderr,
        "static-mixed attention: %zu shapes, %llu floats bit-identical; 256-token scope rejected\n",
        shape_count, (unsigned long long)compared);

cleanup:
    ds4_gpu_tensor_free(qt);
    ds4_gpu_tensor_free(rawt);
    ds4_gpu_tensor_free(compt);
    ds4_gpu_tensor_free(out);
    ds4_gpu_cleanup();
    free(model);
    free(q);
    free(raw);
    free(comp);
    free(reference);
    free(actual);
    return ok ? 0 : 1;
}
