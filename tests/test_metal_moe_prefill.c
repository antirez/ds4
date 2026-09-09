#define _DARWIN_C_SOURCE
#include "ds4_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

enum { INPUT = 256, MID = 512, OUTPUT = 260, EXPERTS = 256, SELECTED = 6 };
typedef struct { uint16_t d; uint8_t qs[64]; } iq2_block;
typedef struct { uint8_t scales[16], qs[64]; uint16_t d, dmin; } q2_block;
typedef struct { uint8_t e, qs[16]; } mxfp4_block;

static uint32_t rng = 1;
static uint32_t random_u32(void) {
    rng ^= rng << 13;
    rng ^= rng >> 17;
    rng ^= rng << 5;
    return rng;
}

static uint64_t aligned(uint64_t n, uint64_t page) {
    return (n + page - 1) / page * page;
}

static int check_case(const void *model, uint64_t model_size,
                      uint64_t up_off, uint64_t down_off, uint32_t tokens,
                      bool mxfp4, int tp_rank) {
    const uint64_t gate_row = mxfp4 ? INPUT / 32 * sizeof(mxfp4_block) : sizeof(iq2_block);
    const uint64_t down_row = mxfp4 ? MID / 32 * sizeof(mxfp4_block) : 2 * sizeof(q2_block);
    const uint64_t pairs = (uint64_t)tokens * SELECTED;
    const uint64_t counts[] = {pairs * MID, pairs * MID, pairs * MID,
                              pairs * OUTPUT, (uint64_t)tokens * OUTPUT};
    ds4_gpu_tensor *result[5] = {0};
    void *reference[5] = {0};
    float *actual = malloc(counts[0] * sizeof(float));
    float *x = malloc((uint64_t)tokens * INPUT * sizeof(float));
    int32_t *ids = malloc(pairs * sizeof(int32_t));
    float *weights = malloc(pairs * sizeof(float));
    ds4_gpu_tensor *xt = ds4_gpu_tensor_alloc((uint64_t)tokens * INPUT * sizeof(float));
    ds4_gpu_tensor *it = ds4_gpu_tensor_alloc(pairs * sizeof(int32_t));
    ds4_gpu_tensor *wt = ds4_gpu_tensor_alloc(pairs * sizeof(float));
    int ok = actual && x && ids && weights && xt && it && wt;
    for (int i = 0; i < 5 && ok; i++) {
        result[i] = ds4_gpu_tensor_alloc(counts[i] * sizeof(float));
        reference[i] = malloc(counts[i] * sizeof(float));
        ok = result[i] && reference[i];
    }
    if (!ok) goto cleanup;
    for (uint64_t i = 0; i < (uint64_t)tokens * INPUT; i++)
        x[i] = ((int)(random_u32() % 101) - 50) / 256.0f;
    /* One hot expert, uneven small groups and many unused experts. */
    for (uint32_t t = 0; t < tokens; t++) {
        ids[(uint64_t)t * SELECTED] = 0;
        weights[(uint64_t)t * SELECTED] = 1.0f / SELECTED;
        for (uint32_t s = 1; s < SELECTED; s++) {
            ids[(uint64_t)t * SELECTED + s] = 1 + (t * 7 + s * 19) % 239;
            weights[(uint64_t)t * SELECTED + s] = 1.0f / SELECTED;
        }
    }
    ok = ds4_gpu_tensor_write(xt, 0, x, (uint64_t)tokens * INPUT * sizeof(float)) &&
         ds4_gpu_tensor_write(it, 0, ids, pairs * sizeof(int32_t)) &&
         ds4_gpu_tensor_write(wt, 0, weights, pairs * sizeof(float));
    for (int run = 0; run < 3 && ok; run++) {
        if (run == 0) setenv("DS4_METAL_DISABLE_ROUTED_MPP_PACKED", "1", 1);
        else unsetenv("DS4_METAL_DISABLE_ROUTED_MPP_PACKED");
        if (run == 0 && tokens < 32)
            setenv("DS4_METAL_DISABLE_TINY_PAIR_SWIGLU_FUSION", "1", 1);
        else unsetenv("DS4_METAL_DISABLE_TINY_PAIR_SWIGLU_FUSION");
        for (int i = 0; i < 5 && ok; i++)
            ok = ds4_gpu_tensor_fill_f32(result[i], NAN, counts[i]);
        bool half_mid = false;
        ok = ok && ds4_gpu_routed_moe_batch_tensor(
            result[4], result[0], result[1], result[2], result[3],
            model, model_size, 0, up_off, down_off, mxfp4 ? 39 : 16, mxfp4 ? 39 : 10,
            MID * gate_row, gate_row, OUTPUT * down_row, down_row,
            INPUT, MID, OUTPUT, it, wt, EXPERTS, SELECTED, 7.0f,
            xt, 0, tokens, &half_mid, true);
        ok = ok && half_mid == (tokens >= 32);
        for (int i = 0; i < 5 && ok; i++) {
            /* Tiny kernels leave unowned intermediates untouched; the fused
             * down reduction also omits the optional expert output tensor. */
            if (tokens <= 4 && i == 3) continue;
            const uint64_t bytes = counts[i] * (i == 2 && half_mid ? sizeof(uint16_t) : sizeof(float));
            ok = ds4_gpu_tensor_read(result[i], 0, actual, bytes);
            for (uint64_t j = 0; j < counts[i] && ok; j++) {
                if (tokens < 32 && i < 3 && tp_rank >= 0 &&
                    ids[j / MID] / (EXPERTS / 2) != tp_rank) continue;
                const float v = i == 2 && half_mid ? (float)((_Float16 *)actual)[j] : actual[j];
                if (!isfinite(v)) {
                    fprintf(stderr, "MoE nonfinite tokens=%u run=%d tensor=%d index=%llu\n",
                            tokens, run, i, (unsigned long long)j);
                    ok = 0;
                }
                if (run && tokens < 32) {
                    const float ref = ((float *)reference[i])[j];
                    if (fabsf(v - ref) > 2e-5f * (1.0f + fabsf(ref))) {
                        fprintf(stderr, "MoE tiny mismatch tokens=%u run=%d tensor=%d index=%llu ref=%g actual=%g\n",
                                tokens, run, i, (unsigned long long)j, ref, v);
                        ok = 0;
                    }
                }
            }
            if (run == 0) memcpy(reference[i], actual, bytes);
            else if (tokens >= 32 && memcmp(reference[i], actual, bytes)) {
                fprintf(stderr, "MoE prefill mismatch tokens=%u run=%d tensor=%d\n", tokens, run, i);
                ok = 0;
            }
        }
    }
cleanup:
    unsetenv("DS4_METAL_DISABLE_ROUTED_MPP_PACKED");
    unsetenv("DS4_METAL_DISABLE_TINY_PAIR_SWIGLU_FUSION");
    for (int i = 0; i < 5; i++) {
        ds4_gpu_tensor_free(result[i]);
        free(reference[i]);
    }
    ds4_gpu_tensor_free(xt);
    ds4_gpu_tensor_free(it);
    ds4_gpu_tensor_free(wt);
    free(actual);
    free(x);
    free(ids);
    free(weights);
    fprintf(stderr, "MoE prefill %s tokens=%u: %s\n", mxfp4 ? "MXFP4" : "IQ2/Q2", tokens, ok ? "PASS" : "FAIL");
    return ok;
}

/* Exercise the production MXFP4 static shapes, comparing batched TP against
 * the existing one-row kernels. Only selected experts need nonzero weights. */
static int check_static_batch(void) {
    enum { D = 4096, H = 2048, N = 6, E = 256 };
    const uint64_t row = D / 32 * sizeof(mxfp4_block);
    const uint64_t down_row = H / 32 * sizeof(mxfp4_block);
    const uint64_t expert = H * row, tensor = E * expert, bytes = 3 * tensor;
    void *model = NULL;
    if (posix_memalign(&model, getpagesize(), bytes)) return 0;
    memset(model, 0, bytes);
    const int32_t active[] = {0, 1, 63, 125, 126, 127, 128, 129, 130, 192, 254, 255};
    for (int w = 0; w < 3; w++) {
        for (unsigned e = 0; e < sizeof(active) / sizeof(*active); e++) {
            mxfp4_block *b = (mxfp4_block *)((char *)model + w * tensor + active[e] * expert);
            for (uint64_t j = 0; j < expert / sizeof(*b); j++) {
                b[j].e = 118 + random_u32() % 5;
                for (int q = 0; q < 16; q++) b[j].qs[q] = random_u32();
            }
        }
    }
    float *x = malloc(N * D * sizeof(float));
    float *reference = malloc(N * D * sizeof(float));
    float *actual = malloc(N * D * sizeof(float));
    int32_t ids[N * SELECTED];
    float weights[N * SELECTED];
    ds4_gpu_tensor *xt = ds4_gpu_tensor_alloc(N * D * sizeof(float));
    ds4_gpu_tensor *it = ds4_gpu_tensor_alloc(sizeof(ids));
    ds4_gpu_tensor *wt = ds4_gpu_tensor_alloc(sizeof(weights));
    ds4_gpu_tensor *gate = ds4_gpu_tensor_alloc(N * SELECTED * H * sizeof(float));
    ds4_gpu_tensor *up = ds4_gpu_tensor_alloc(N * SELECTED * H * sizeof(float));
    ds4_gpu_tensor *mid = ds4_gpu_tensor_alloc(N * SELECTED * H * sizeof(float));
    ds4_gpu_tensor *down = ds4_gpu_tensor_alloc(N * SELECTED * D * sizeof(float));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(N * D * sizeof(float));
    int ok = x && reference && actual && xt && it && wt && gate && up && mid && down && out;
    if (!ok) goto done;
    for (int i = 0; i < N * D; i++) x[i] = ((int)(random_u32() % 101) - 50) / 256.0f;
    for (int r = 0; r < N; r++) for (int s = 0; s < SELECTED; s++) {
        ids[r * SELECTED + s] = active[(r * 3 + s) % 12];
        weights[r * SELECTED + s] = 1.0f / SELECTED;
    }
    ok = ds4_gpu_set_model_map(model, bytes) &&
         ds4_gpu_tensor_write(xt, 0, x, N * D * sizeof(float)) &&
         ds4_gpu_tensor_write(it, 0, ids, sizeof(ids)) &&
         ds4_gpu_tensor_write(wt, 0, weights, sizeof(weights));
    setenv("DS4_TP_NO_KEEPALIVE", "1", 1);
    for (uint32_t rank = 0; rank < 2 && ok; rank++) {
        ok = ds4_gpu_tp_init(rank, NULL, 0, 0, 0, NULL, NULL);
        for (int r = 0; r < N && ok; r++) {
            ds4_gpu_tensor *xr = ds4_gpu_tensor_view(xt, r * D * sizeof(float), D * sizeof(float));
            ds4_gpu_tensor *ir = ds4_gpu_tensor_view(it, r * SELECTED * sizeof(int32_t), SELECTED * sizeof(int32_t));
            ds4_gpu_tensor *wr = ds4_gpu_tensor_view(wt, r * SELECTED * sizeof(float), SELECTED * sizeof(float));
            ok = xr && ir && wr && ds4_gpu_routed_moe_one_tensor(
                out, gate, up, mid, down, model, bytes, 0, tensor, 2 * tensor,
                39, 39, expert, row, expert, down_row, D, H, D,
                ir, wr, E, SELECTED, 7.0f, xr, NULL, 0, true) &&
                ds4_gpu_tensor_read(out, 0, reference + r * D, D * sizeof(float));
            ds4_gpu_tensor_free(xr);
            ds4_gpu_tensor_free(ir);
            ds4_gpu_tensor_free(wr);
        }
        const uint32_t sizes[] = {6, 2, 5, 3, 4, 6};
        for (unsigned i = 0; i < sizeof(sizes) / sizeof(*sizes) && ok; i++) {
            bool half_mid = true;
            const uint32_t n = sizes[i];
            ok = ds4_gpu_tensor_fill_f32(mid, NAN, N * SELECTED * H) &&
                 ds4_gpu_tensor_fill_f32(out, NAN, N * D) &&
                 ds4_gpu_routed_moe_batch_tensor(
                    out, gate, up, mid, down, model, bytes, 0, tensor, 2 * tensor,
                    39, 39, expert, row, expert, down_row, D, H, D,
                    it, wt, E, SELECTED, 7.0f, xt, 0, n, &half_mid, true) &&
                 !half_mid && ds4_gpu_tensor_read(out, 0, actual, n * D * sizeof(float));
            for (uint32_t j = 0; j < n * D && ok; j++)
                ok = isfinite(actual[j]) && isfinite(reference[j]) && actual[j] == reference[j];
            fprintf(stderr, "MXFP4 static batch rank=%u rows=%u exact: %s\n", rank, n, ok ? "PASS" : "FAIL");
        }
        ds4_gpu_tp_shutdown();
    }
    unsetenv("DS4_TP_NO_KEEPALIVE");
done:
    ds4_gpu_tensor_free(xt); ds4_gpu_tensor_free(it); ds4_gpu_tensor_free(wt);
    ds4_gpu_tensor_free(gate); ds4_gpu_tensor_free(up); ds4_gpu_tensor_free(mid);
    ds4_gpu_tensor_free(down); ds4_gpu_tensor_free(out);
    ds4_gpu_cleanup();
    free(x); free(reference); free(actual); free(model);
    return ok;
}

int main(void) {
    const uint64_t page = getpagesize();
    const uint64_t up_off = aligned((uint64_t)EXPERTS * MID * sizeof(iq2_block), page);
    const uint64_t down_off = up_off * 2;
    const uint64_t model_size = aligned(down_off + (uint64_t)EXPERTS * OUTPUT * 2 * sizeof(q2_block), page);
    void *model = NULL;
    if (posix_memalign(&model, page, model_size)) return 1;
    memset(model, 0, model_size);
    for (uint64_t off = 0; off < down_off; off += up_off) {
        iq2_block *w = (iq2_block *)((char *)model + off);
        for (uint32_t b = 0; b < EXPERTS * MID; b++) {
            w[b].d = 0x1400; /* 2^-10, finite nontrivial activations. */
            for (uint32_t j = 0; j < sizeof(w[b].qs); j++) w[b].qs[j] = random_u32();
        }
    }
    q2_block *w = (q2_block *)((char *)model + down_off);
    for (uint32_t b = 0; b < EXPERTS * OUTPUT * 2; b++) {
        w[b].d = w[b].dmin = 0x2000;
        for (uint32_t j = 0; j < sizeof(w[b].scales); j++) w[b].scales[j] = random_u32();
        for (uint32_t j = 0; j < sizeof(w[b].qs); j++) w[b].qs[j] = random_u32();
    }
    int ok = ds4_gpu_init() && ds4_gpu_set_model_map(model, model_size);
    ds4_gpu_set_quality(false);
    ds4_gpu_set_ssd_streaming(false);
    const uint32_t sizes[] = {511, 512, 513, 1024, 2048, 4096, 512};
    for (unsigned i = 0; i < sizeof(sizes) / sizeof(*sizes) && ok; i++)
        ok = check_case(model, model_size, up_off, down_off, sizes[i], false, -1);
    ds4_gpu_cleanup();
    free(model);
    if (!ok) return 1;

    const uint64_t mx_up = aligned((uint64_t)EXPERTS * MID * INPUT / 32 * sizeof(mxfp4_block), page);
    const uint64_t mx_down = mx_up * 2;
    const uint64_t mx_size = aligned(mx_down + (uint64_t)EXPERTS * OUTPUT * MID / 32 * sizeof(mxfp4_block), page);
    if (posix_memalign(&model, page, mx_size)) return 1;
    memset(model, 0, mx_size);
    const uint64_t offsets[] = {0, mx_up, mx_down};
    for (int i = 0; i < 3; i++) {
        mxfp4_block *b = (mxfp4_block *)((char *)model + offsets[i]);
        const uint64_t count = (uint64_t)EXPERTS * (i == 2 ? OUTPUT * MID : MID * INPUT) / 32;
        for (uint64_t j = 0; j < count; j++) {
            b[j].e = 118 + random_u32() % 5;
            for (int q = 0; q < 16; q++) b[j].qs[q] = random_u32();
        }
    }
    ok = ds4_gpu_init() && ds4_gpu_set_model_map(model, mx_size);
    ds4_gpu_set_quality(false);
    ds4_gpu_set_ssd_streaming(false);
    /* Bind ownership only; no gates or network exchange are needed here. */
    setenv("DS4_TP_NO_KEEPALIVE", "1", 1);
    for (uint32_t rank = 0; rank < 2 && ok; rank++) {
        ok = ds4_gpu_tp_init(rank, NULL, 0, 0, 0, NULL, NULL);
        for (unsigned i = 0; i < sizeof(sizes) / sizeof(*sizes) && ok; i++)
            ok = check_case(model, mx_size, mx_up, mx_down, sizes[i], true, (int)rank);
        for (uint32_t n = 2; n <= 6 && ok; n++)
            ok = check_case(model, mx_size, mx_up, mx_down, n, true, (int)rank);
        ds4_gpu_tp_shutdown();
    }
    unsetenv("DS4_TP_NO_KEEPALIVE");
    ds4_gpu_cleanup();
    free(model);
    if (ok) {
        ok = ds4_gpu_init();
        if (ok && ds4_gpu_device_is_m5_apple_silicon()) ok = check_static_batch();
        ds4_gpu_cleanup();
    }
    return ok ? 0 : 1;
}
