#define _DARWIN_C_SOURCE
#include "ds4_gpu.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

enum { D = 256, H = 256, O = 260, E = 128, K = 8 };
typedef struct { uint8_t scales[16], qs[64]; uint16_t d, dmin; } q2_block;
typedef struct { uint16_t d, dmin; uint8_t scales[12], qs[128]; } q4_block;
static uint32_t rng = 7;
static uint32_t random_u32(void) {
    rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
    return rng;
}
static uint64_t aligned(uint64_t n) {
    uint64_t p = (uint64_t)getpagesize();
    return (n + p - 1) / p * p;
}
static void fill(void *dst, uint32_t type, uint64_t blocks) {
    for (uint64_t i = 0; i < blocks; i++) {
        if (type == 10) {
            q2_block *b = (q2_block *)dst + i;
            b->d = b->dmin = 0x1800;
            for (unsigned j = 0; j < sizeof(b->scales); j++) b->scales[j] = random_u32();
            for (unsigned j = 0; j < sizeof(b->qs); j++) b->qs[j] = random_u32();
        } else {
            q4_block *b = (q4_block *)dst + i;
            b->d = b->dmin = 0x1000;
            for (unsigned j = 0; j < sizeof(b->scales); j++) b->scales[j] = random_u32();
            for (unsigned j = 0; j < sizeof(b->qs); j++) b->qs[j] = random_u32();
        }
    }
}
static int check(uint32_t gate_type, uint32_t down_type, uint32_t tokens) {
    uint64_t gr = gate_type == 10 ? sizeof(q2_block) : sizeof(q4_block);
    uint64_t dr = down_type == 10 ? sizeof(q2_block) : sizeof(q4_block);
    uint64_t ge = H * gr, de = O * dr;
    uint64_t up = aligned(E * ge), down = aligned(up + E * ge);
    uint64_t size = aligned(down + E * de), stride = K * H + (tokens < 96 ? 32 : 0);
    void *model = NULL;
    FILE *source = NULL;
    float *input = calloc((size_t)tokens * D, sizeof(float));
    int32_t *ids = calloc((size_t)tokens * K, sizeof(int32_t));
    float *weights = calloc((size_t)tokens * K, sizeof(float));
    float *reference = calloc((size_t)tokens * O, sizeof(float));
    float *reference_mixed = calloc((size_t)tokens * O, sizeof(float));
    float *actual = calloc((size_t)tokens * O, sizeof(float));
    ds4_gpu_tensor *x = NULL, *selected = NULL, *w = NULL, *mid = NULL, *out = NULL;
    int ok = !posix_memalign(&model, getpagesize(), size) && input && ids && weights && reference && reference_mixed && actual;
    if (!ok) goto cleanup;
    memset(model, 0, size);
    fill(model, gate_type, E * H);
    fill((char *)model + up, gate_type, E * H);
    fill((char *)model + down, down_type, E * O);
    source = tmpfile();
    ok = source && fwrite(model, 1, size, source) == size && !fflush(source);
    ds4_gpu_set_glm_model(true);
    if (!ok || !ds4_gpu_init() || !ds4_gpu_set_model_map(model, size)) { ok = 0; goto cleanup; }
    ds4_gpu_set_quality(false);
    x = ds4_gpu_tensor_alloc((uint64_t)tokens * D * 4);
    selected = ds4_gpu_tensor_alloc((uint64_t)tokens * K * 4);
    w = ds4_gpu_tensor_alloc((uint64_t)tokens * K * 4);
    mid = ds4_gpu_tensor_alloc((uint64_t)tokens * stride * 4);
    out = ds4_gpu_tensor_alloc((uint64_t)tokens * O * 4);
    ok = x && selected && w && mid && out;
    if (!ok) goto cleanup;
    for (uint32_t i = 0; i < tokens * D; i++) input[i] = ((int)(random_u32() % 101) - 50) / 16.0f;
    for (uint32_t t = 0; t < tokens; t++) for (uint32_t k = 0; k < K; k++) {
        ids[t * K + k] = (t * 7 + k * 11) % E;
        weights[t * K + k] = (k + 1) / 36.0f;
    }
    ok = ds4_gpu_tensor_write(x, 0, input, (uint64_t)tokens * D * 4) &&
         ds4_gpu_tensor_write(selected, 0, ids, (uint64_t)tokens * K * 4) &&
         ds4_gpu_tensor_write(w, 0, weights, (uint64_t)tokens * K * 4);
    for (int run = 0; run < 5 && ok; run++) {
        bool changed_ids = run == 1 || run == 4;
        for (uint32_t t = 0; t < tokens; t++) for (uint32_t k = 0; k < K; k++)
            ids[t * K + k] = (t * 7 + k * 11 + (changed_ids && k < 4 ? 1 : 0)) % E;
        ok = ds4_gpu_tensor_write(selected, 0, ids, (uint64_t)tokens * K * 4);
        if (run == 0 || run == 2) ds4_gpu_set_ssd_streaming(run >= 2);
        if (run == 2) {
            /* Only the file has real weights: a mapped fallback must fail. */
            memset(model, 0, size);
            ds4_gpu_set_model_fd(fileno(source));
            ds4_gpu_set_streaming_expert_cache_expert_bytes(2 * ge + de);
            ds4_gpu_set_streaming_expert_cache_budget(32);
        }
        ok = ok && ds4_gpu_tensor_fill_f32(out, NAN, (uint64_t)tokens * O);
        ok = ok && ds4_gpu_begin_commands();
        if (tokens == 1) {
            ok = ok && ds4_gpu_glm_routed_moe_one_tensor(out, mid, model, size,
                0, up, down, gate_type, gate_type, down_type, ge, gr, ge, gr,
                de, dr, D, H, O, selected, w, E, K, 7.0f, 0, x, false);
        } else {
            ok = ok && ds4_gpu_glm_routed_moe_batch_tensor(out, mid, model, size,
                0, up, down, gate_type, gate_type, down_type, ge, gr, ge, gr,
                de, dr, D, H, O, selected, w, E, K, 7.0f, 0, x, tokens, stride, false);
        }
        if (ds4_gpu_commands_active() && !ds4_gpu_end_commands()) ok = 0;
        ok = ok && ds4_gpu_tensor_read(out, 0, actual, (uint64_t)tokens * O * 4);
        float *expected = changed_ids ? reference_mixed : reference;
        if (run < 2) {
            float peak = 0;
            for (uint32_t i = 0; i < tokens * O; i++) {
                if (!isfinite(actual[i])) ok = 0;
                if (fabsf(actual[i]) > peak) peak = fabsf(actual[i]);
            }
            if (peak < 0.01f) { fprintf(stderr, "reference peak too small: %g\n", peak); ok = 0; }
            memcpy(expected, actual, (size_t)tokens * O * 4);
        }
        else for (uint32_t i = 0; i < tokens * O && ok; i++) {
            if (!isfinite(actual[i]) || fabsf(actual[i] - expected[i]) > 2e-4f * (1 + fabsf(expected[i]))) {
                fprintf(stderr, "mismatch types=%u/%u tokens=%u run=%d i=%u ref=%g actual=%g\n",
                    gate_type, down_type, tokens, run, i, expected[i], actual[i]);
                ok = 0;
            }
        }
        if (tokens == 1 && run >= 2) ds4_gpu_print_memory_report(run == 2 ? "cold" : run == 3 ? "all-hit" : "mixed-hit");
        fprintf(stderr, "GLM cache run=%d ok=%d\n", run, ok);
    }
cleanup:
    ds4_gpu_tensor_free(x); ds4_gpu_tensor_free(selected); ds4_gpu_tensor_free(w);
    ds4_gpu_tensor_free(mid); ds4_gpu_tensor_free(out);
    ds4_gpu_set_model_fd(-1); ds4_gpu_cleanup();
    if (source) fclose(source);
    free(model); free(input); free(ids); free(weights); free(reference); free(reference_mixed); free(actual);
    fprintf(stderr, "GLM cache %u/%u tokens=%u: %s\n", gate_type, down_type, tokens, ok ? "PASS" : "FAIL");
    return ok;
}
int main(void) {
    const uint32_t types[][2] = {{10, 12}, {10, 10}, {12, 12}};
    const uint32_t tokens[] = {1, 28, 96};
    for (unsigned t = 0; t < 3; t++) for (unsigned n = 0; n < 3; n++)
        if (!check(types[t][0], types[t][1], tokens[n])) return 1;
    return 0;
}
