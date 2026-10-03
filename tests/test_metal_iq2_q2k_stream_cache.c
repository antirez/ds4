/* SSD-streaming decode for IQ2_XXS gate/up + Q2_K down routed experts must
 * read the selected experts through the file-backed expert cache for any
 * selected-expert count, not only the six that DeepSeek V4 Flash uses.
 * GLM 5.3 Flash Q2 has this layout with eight experts per token.
 *
 * Each case computes a resident reference, then zeroes the mapped weights and
 * repeats the call in streaming mode with only the file holding real weights:
 * an output that still matches the reference can only have come from the
 * cache. */
#define _DARWIN_C_SOURCE
#include "ds4_gpu.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

enum { D = 256, H = 256, O = 260, E = 128, TYPE_Q2_K = 10, TYPE_IQ2_XXS = 16 };
typedef struct { uint16_t d; uint16_t qs[32]; } iq2_xxs_block;
typedef struct { uint8_t scales[16], qs[64]; uint16_t d, dmin; } q2_k_block;

static uint32_t rng = 11;
static uint32_t random_u32(void) {
    rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
    return rng;
}

static uint64_t page_aligned(uint64_t n) {
    uint64_t p = (uint64_t)getpagesize();
    return (n + p - 1) / p * p;
}

static void fill_iq2_xxs(void *dst, uint64_t blocks) {
    for (uint64_t i = 0; i < blocks; i++) {
        iq2_xxs_block *b = (iq2_xxs_block *)dst + i;
        b->d = 0x1800;
        for (unsigned j = 0; j < 32; j++) b->qs[j] = (uint16_t)random_u32();
    }
}

static void fill_q2_k(void *dst, uint64_t blocks) {
    for (uint64_t i = 0; i < blocks; i++) {
        q2_k_block *b = (q2_k_block *)dst + i;
        b->d = b->dmin = 0x1800;
        for (unsigned j = 0; j < sizeof(b->scales); j++) b->scales[j] = (uint8_t)random_u32();
        for (unsigned j = 0; j < sizeof(b->qs); j++) b->qs[j] = (uint8_t)random_u32();
    }
}

static int check(uint32_t n_expert) {
    const uint64_t gate_row = sizeof(iq2_xxs_block) * (D / 256);
    const uint64_t down_row = sizeof(q2_k_block) * (H / 256);
    const uint64_t gate_expert = H * gate_row, down_expert = O * down_row;
    const uint64_t up = page_aligned(E * gate_expert);
    const uint64_t down = page_aligned(up + E * gate_expert);
    const uint64_t size = page_aligned(down + E * down_expert);
    void *model = NULL;
    FILE *source = NULL;
    float *input = calloc(D, sizeof(float));
    int32_t *ids = calloc(n_expert, sizeof(int32_t));
    float *weights = calloc(n_expert, sizeof(float));
    float *reference = calloc(O, sizeof(float));
    float *reference_changed = calloc(O, sizeof(float));
    float *actual = calloc(O, sizeof(float));
    ds4_gpu_tensor *x = NULL, *selected = NULL, *w = NULL;
    ds4_gpu_tensor *gate = NULL, *upt = NULL, *mid = NULL, *experts = NULL, *out = NULL;
    int ok = !posix_memalign(&model, getpagesize(), size) &&
             input && ids && weights && reference && reference_changed && actual;
    if (!ok) goto cleanup;
    memset(model, 0, size);
    fill_iq2_xxs(model, E * H * (D / 256));
    fill_iq2_xxs((char *)model + up, E * H * (D / 256));
    fill_q2_k((char *)model + down, E * O * (H / 256));
    source = tmpfile();
    ok = source && fwrite(model, 1, size, source) == size && !fflush(source);
    ds4_gpu_set_glm_model(true);
    if (!ok || !ds4_gpu_init() || !ds4_gpu_set_model_map(model, size)) { ok = 0; goto cleanup; }
    ds4_gpu_set_quality(false);
    x = ds4_gpu_tensor_alloc(D * 4);
    selected = ds4_gpu_tensor_alloc(n_expert * 4);
    w = ds4_gpu_tensor_alloc(n_expert * 4);
    gate = ds4_gpu_tensor_alloc((uint64_t)n_expert * H * 4);
    upt = ds4_gpu_tensor_alloc((uint64_t)n_expert * H * 4);
    mid = ds4_gpu_tensor_alloc((uint64_t)n_expert * H * 4);
    experts = ds4_gpu_tensor_alloc((uint64_t)n_expert * O * 4);
    out = ds4_gpu_tensor_alloc(O * 4);
    ok = x && selected && w && gate && upt && mid && experts && out;
    if (!ok) goto cleanup;
    for (uint32_t i = 0; i < D; i++) input[i] = ((int)(random_u32() % 101) - 50) / 16.0f;
    for (uint32_t k = 0; k < n_expert; k++) weights[k] = (k + 1) / 36.0f;
    ok = ds4_gpu_tensor_write(x, 0, input, D * 4) &&
         ds4_gpu_tensor_write(w, 0, weights, n_expert * 4);

    /* Runs 0-1 are resident references; 2 is cold, 3 all-hit, 4 half-changed. */
    for (int run = 0; run < 5 && ok; run++) {
        const bool changed_ids = run == 1 || run == 4;
        for (uint32_t k = 0; k < n_expert; k++)
            ids[k] = (int32_t)((k * 11 + (changed_ids && k < n_expert / 2 ? 1 : 0)) % E);
        ok = ds4_gpu_tensor_write(selected, 0, ids, n_expert * 4);
        if (run == 0 || run == 2) ds4_gpu_set_ssd_streaming(run >= 2);
        if (run == 2) {
            memset(model, 0, size);
            ds4_gpu_set_model_fd(fileno(source));
            ds4_gpu_set_streaming_expert_cache_expert_bytes(2 * gate_expert + down_expert);
            ds4_gpu_set_streaming_expert_cache_budget(32);
        }
        ok = ok && ds4_gpu_tensor_fill_f32(out, NAN, O);
        ok = ok && ds4_gpu_begin_commands();
        ok = ok && ds4_gpu_routed_moe_one_tensor(out, gate, upt, mid, experts, model, size,
                0, up, down, TYPE_IQ2_XXS, TYPE_Q2_K,
                gate_expert, gate_row, down_expert, down_row,
                D, H, O, selected, w, E, n_expert, 7.0f, x, NULL, 0, false);
        if (ds4_gpu_commands_active() && !ds4_gpu_end_commands()) ok = 0;
        ok = ok && ds4_gpu_tensor_read(out, 0, actual, O * 4);
        float *expected = changed_ids ? reference_changed : reference;
        if (run < 2) {
            float peak = 0;
            for (uint32_t i = 0; i < O; i++) {
                if (!isfinite(actual[i])) ok = 0;
                if (fabsf(actual[i]) > peak) peak = fabsf(actual[i]);
            }
            if (peak < 0.01f) { fprintf(stderr, "reference peak too small: %g\n", peak); ok = 0; }
            memcpy(expected, actual, O * 4);
        } else {
            for (uint32_t i = 0; i < O && ok; i++) {
                if (!isfinite(actual[i]) ||
                    fabsf(actual[i] - expected[i]) > 2e-4f * (1 + fabsf(expected[i]))) {
                    fprintf(stderr, "mismatch n_expert=%u run=%d i=%u ref=%g actual=%g\n",
                            n_expert, run, i, expected[i], actual[i]);
                    ok = 0;
                }
            }
        }
    }
cleanup:
    ds4_gpu_tensor_free(x); ds4_gpu_tensor_free(selected); ds4_gpu_tensor_free(w);
    ds4_gpu_tensor_free(gate); ds4_gpu_tensor_free(upt); ds4_gpu_tensor_free(mid);
    ds4_gpu_tensor_free(experts); ds4_gpu_tensor_free(out);
    ds4_gpu_set_ssd_streaming(false);
    ds4_gpu_set_model_fd(-1); ds4_gpu_cleanup();
    if (source) fclose(source);
    free(model); free(input); free(ids); free(weights);
    free(reference); free(reference_changed); free(actual);
    fprintf(stderr, "IQ2_XXS/Q2_K stream cache n_expert=%u: %s\n", n_expert, ok ? "PASS" : "FAIL");
    return ok;
}

int main(void) {
    /* 6 is DeepSeek V4 Flash's existing selected-slot path; 8 is GLM 5.3. */
    const uint32_t counts[] = {6, 8};
    int ok = 1;
    for (unsigned i = 0; i < sizeof(counts) / sizeof(counts[0]); i++) ok &= check(counts[i]);
    return ok ? 0 : 1;
}
