#define _DARWIN_C_SOURCE
#include "ds4_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

enum { SELECTED = 6 };
static uint32_t INPUT = 256, MID = 512, OUTPUT = 260, EXPERTS = 256;
typedef struct { uint16_t d; uint8_t qs[64]; } iq2_block;
typedef struct { uint8_t scales[16], qs[64]; uint16_t d, dmin; } q2_block;
typedef struct { uint16_t d, dmin; uint8_t scales[12], qs[128]; } q4_block;
typedef struct { uint8_t e, qs[16]; } mxfp4_block;

static uint32_t rng = 1;
static double now_seconds(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec / 1e9;
}
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
    const uint64_t gate_row = mxfp4 ? INPUT / 32 * sizeof(mxfp4_block) : INPUT / 256 * sizeof(iq2_block);
    const uint64_t down_row = mxfp4 ? MID / 32 * sizeof(mxfp4_block) : MID / 256 * sizeof(q2_block);
    const uint64_t pairs = (uint64_t)tokens * SELECTED;
    const bool v41_decode = !mxfp4 && tokens <= 8 && EXPERTS == 384 &&
        INPUT == 5120 && MID == 2304 && OUTPUT == 5120;
    const uint64_t counts[] = {pairs * MID, pairs * MID, pairs * MID,
                              pairs * OUTPUT, (uint64_t)tokens * OUTPUT};
    ds4_gpu_tensor *result[5] = {0};
    void *reference[5] = {0};
    const uint64_t max_count = counts[0] > counts[3] ? counts[0] : counts[3];
    float *actual = malloc(max_count * sizeof(float));
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
            ids[(uint64_t)t * SELECTED + s] = 1 + (t * 7 + s * 19) % (EXPERTS - 17);
            weights[(uint64_t)t * SELECTED + s] = 1.0f / SELECTED;
        }
    }
    ok = ds4_gpu_tensor_write(xt, 0, x, (uint64_t)tokens * INPUT * sizeof(float)) &&
         ds4_gpu_tensor_write(it, 0, ids, pairs * sizeof(int32_t)) &&
         ds4_gpu_tensor_write(wt, 0, weights, pairs * sizeof(float));
    for (int run = 0; run < 5 && ok; run++) {
        if (run == 0) setenv("DS4_METAL_DISABLE_ROUTED_MPP_PACKED", "1", 1);
        else unsetenv("DS4_METAL_DISABLE_ROUTED_MPP_PACKED");
        if (run == 0) setenv("DS4_METAL_DISABLE_MOE_MM_ID_PAIR_SWIGLU", "1", 1);
        else unsetenv("DS4_METAL_DISABLE_MOE_MM_ID_PAIR_SWIGLU");
        if (run == 1 || run == 4) setenv("DS4_METAL_DISABLE_V41_PACKED_M32N128", "1", 1);
        else unsetenv("DS4_METAL_DISABLE_V41_PACKED_M32N128");
        if (run == 0 && tokens < 32)
            setenv("DS4_METAL_DISABLE_TINY_PAIR_SWIGLU_FUSION", "1", 1);
        else unsetenv("DS4_METAL_DISABLE_TINY_PAIR_SWIGLU_FUSION");
        for (int i = 0; i < 5 && ok; i++)
            ok = ds4_gpu_tensor_fill_f32(result[i], NAN, counts[i]);
        bool half_mid = false;
        const double begin = now_seconds();
        ok = ok && ds4_gpu_routed_moe_batch_tensor(
            result[4], result[0], result[1], result[2], result[3],
            model, model_size, 0, up_off, down_off, mxfp4 ? 39 : 16, mxfp4 ? 39 : 10,
            MID * gate_row, gate_row, OUTPUT * down_row, down_row,
            INPUT, MID, OUTPUT, it, wt, EXPERTS, SELECTED, 7.0f,
            xt, 0, tokens, &half_mid, true);
        fprintf(stderr, "MoE tile tokens=%u run=%d %.3f ms\n", tokens, run, (now_seconds() - begin) * 1000);
        ok = ok && half_mid == (tokens >= 32);
        for (int i = 0; i < 5 && ok; i++) {
            /* Fused kernels need not materialize gate/up. Compare their
             * consumed mid and final outputs with the unfused reference.
             * The tiny fused down reduction also omits expert outputs. */
            if ((tokens <= 4 || v41_decode) && i == 3) continue;
            if (run && i < 2) continue;
            const uint64_t bytes = counts[i] * (i == 2 && half_mid ? sizeof(uint16_t) : sizeof(float));
            ok = ds4_gpu_tensor_read(result[i], 0, actual, bytes);
            for (uint64_t j = 0; j < counts[i] && ok; j++) {
                if (tokens < 32 && i < 3 && tp_rank >= 0 &&
                    (uint32_t)ids[j / MID] / (EXPERTS / 2) != (uint32_t)tp_rank) continue;
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
    unsetenv("DS4_METAL_DISABLE_MOE_MM_ID_PAIR_SWIGLU");
    unsetenv("DS4_METAL_DISABLE_V41_PACKED_M32N128");
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

/* Compare production decode shapes with independent one-row calls. Only
 * selected experts need nonzero weights. Q4 also covers single-host batches. */
static int check_static_batch(bool q4) {
    const uint32_t D = q4 ? 5120 : 4096, H = q4 ? 2304 : 2048;
    const uint32_t N = q4 ? 8 : 6, E = q4 ? 384 : 256;
    const uint32_t type = q4 ? 12 : 39;
    const uint64_t row = q4 ? D / 256 * sizeof(q4_block) : D / 32 * sizeof(mxfp4_block);
    const uint64_t down_row = q4 ? H / 256 * sizeof(q4_block) : H / 32 * sizeof(mxfp4_block);
    const uint64_t expert = H * row, tensor = E * expert, bytes = 3 * tensor;
    void *model = NULL;
    if (posix_memalign(&model, getpagesize(), bytes)) return 0;
    memset(model, 0, bytes);
    const int32_t mx_active[] = {0, 1, 63, 125, 126, 127, 128, 129, 130, 192, 254, 255};
    const int32_t q4_active[] = {0, 1, 63, 127, 190, 191, 192, 193, 255, 381, 382, 383};
    const int32_t *active = q4 ? q4_active : mx_active;
    for (int w = 0; w < 3; w++) {
        for (unsigned e = 0; e < 12; e++) {
            void *data = (char *)model + w * tensor + active[e] * expert;
            if (q4) {
                q4_block *b = data;
                for (uint64_t j = 0; j < expert / sizeof(*b); j++) {
                    b[j].d = b[j].dmin = 0x1800;
                    for (unsigned k = 0; k < sizeof(b[j].scales); k++) b[j].scales[k] = random_u32();
                    for (unsigned k = 0; k < sizeof(b[j].qs); k++) b[j].qs[k] = random_u32();
                }
            } else {
                mxfp4_block *b = data;
                for (uint64_t j = 0; j < expert / sizeof(*b); j++) {
                    b[j].e = 118 + random_u32() % 5;
                    for (int k = 0; k < 16; k++) b[j].qs[k] = random_u32();
                }
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
    for (uint32_t i = 0; i < N * D; i++) x[i] = ((int)(random_u32() % 101) - 50) / 256.0f;
    for (uint32_t r = 0; r < N; r++) for (int s = 0; s < SELECTED; s++) {
        ids[r * SELECTED + s] = active[(r * 3 + s) % 12];
        weights[r * SELECTED + s] = q4 ? (s + 1) / 21.0f : 1.0f / SELECTED;
    }
    ok = ds4_gpu_set_model_map(model, bytes) &&
         ds4_gpu_tensor_write(xt, 0, x, N * D * sizeof(float)) &&
         ds4_gpu_tensor_write(it, 0, ids, sizeof(ids)) &&
         ds4_gpu_tensor_write(wt, 0, weights, sizeof(weights));
    setenv("DS4_TP_NO_KEEPALIVE", "1", 1);
    for (int rank = q4 ? -1 : 0; rank < 2 && ok; rank++) {
        if (rank >= 0) ok = ds4_gpu_tp_init((uint32_t)rank, NULL, 0, 0, 0, NULL, NULL);
        for (uint32_t r = 0; r < N && ok; r++) {
            ds4_gpu_tensor *xr = ds4_gpu_tensor_view(xt, r * D * sizeof(float), D * sizeof(float));
            ds4_gpu_tensor *ir = ds4_gpu_tensor_view(it, r * SELECTED * sizeof(int32_t), SELECTED * sizeof(int32_t));
            ds4_gpu_tensor *wr = ds4_gpu_tensor_view(wt, r * SELECTED * sizeof(float), SELECTED * sizeof(float));
            ok = xr && ir && wr && ds4_gpu_routed_moe_one_tensor(
                out, gate, up, mid, down, model, bytes, 0, tensor, 2 * tensor,
                type, type, expert, row, expert, down_row, D, H, D,
                ir, wr, E, SELECTED, 7.0f, xr, NULL, 0, true) &&
                ds4_gpu_tensor_read(out, 0, reference + r * D, D * sizeof(float));
            ds4_gpu_tensor_free(xr);
            ds4_gpu_tensor_free(ir);
            ds4_gpu_tensor_free(wr);
        }
        const uint32_t sizes[] = {6, 2, 5, 3, 4, 6};
        const uint32_t q4_sizes[] = {8, 1, 2, 3, 4, 5, 6, 7, 8};
        const unsigned n_sizes = q4 ? sizeof(q4_sizes) / sizeof(*q4_sizes) : sizeof(sizes) / sizeof(*sizes);
        for (unsigned i = 0; i < n_sizes && ok; i++) {
            bool half_mid = true;
            const uint32_t n = q4 ? q4_sizes[i] : sizes[i];
            ok = ds4_gpu_tensor_fill_f32(mid, NAN, N * SELECTED * H) &&
                 ds4_gpu_tensor_fill_f32(out, NAN, N * D) &&
                 ds4_gpu_routed_moe_batch_tensor(
                    out, gate, up, mid, down, model, bytes, 0, tensor, 2 * tensor,
                    type, type, expert, row, expert, down_row, D, H, D,
                    it, wt, E, SELECTED, 7.0f, xt, 0, n, &half_mid, true) &&
                 !half_mid && ds4_gpu_tensor_read(out, 0, actual, N * D * sizeof(float));
            for (uint32_t j = 0; j < N * D && ok; j++) {
                ok = j >= n * D ? isnan(actual[j]) :
                    isfinite(actual[j]) && isfinite(reference[j]) && actual[j] == reference[j];
                if (!ok) fprintf(stderr, "decode mismatch index=%u expected=%.9g actual=%.9g\n",
                    j, j < n * D ? reference[j] : NAN, actual[j]);
            }
            fprintf(stderr, "%s static batch rank=%d rows=%u exact: %s\n",
                q4 ? "V4.1 Q4" : "MXFP4", rank, n, ok ? "PASS" : "FAIL");
        }
        if (rank >= 0) ds4_gpu_tp_shutdown();
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

static int check_address_batch(const void *model, uint64_t bytes,
                               uint64_t up_off, uint64_t down_off) {
    enum { T = 512 };
    FILE *source = tmpfile();
    const uint64_t xb = T * INPUT * sizeof(float), ob = T * OUTPUT * sizeof(float);
    const uint64_t mb = T * SELECTED * MID * sizeof(float), ib = T * SELECTED * sizeof(int32_t);
    float *x = malloc(xb), *weights = malloc(ib), *reference = malloc(ob), *actual = malloc(ob);
    int32_t *ids = malloc(ib);
    ds4_gpu_tensor *xt = ds4_gpu_tensor_alloc(xb), *it = ds4_gpu_tensor_alloc(ib);
    ds4_gpu_tensor *wt = ds4_gpu_tensor_alloc(ib), *gate = ds4_gpu_tensor_alloc(mb);
    ds4_gpu_tensor *up = ds4_gpu_tensor_alloc(mb), *mid = ds4_gpu_tensor_alloc(mb);
    ds4_gpu_tensor *experts = ds4_gpu_tensor_alloc(T * SELECTED * OUTPUT * sizeof(float));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(ob);
    int ok = source && fwrite(model, 1, bytes, source) == bytes && fflush(source) == 0 &&
        ds4_gpu_set_model_fd(fileno(source)) &&
        x && weights && reference && actual && ids && xt && it && wt && gate && up && mid && experts && out;
    if (!ok) goto done;
    for (uint32_t i = 0; i < T * INPUT; i++) x[i] = ((int)(random_u32() % 101) - 50) / 256.0f;
    for (int t = 0; t < T; t++) for (int k = 0; k < SELECTED; k++) {
        ids[t * SELECTED + k] = (t * 13 + k * 17) % EXPERTS;
        weights[t * SELECTED + k] = (k + 1) / 21.0f;
    }
    ds4_gpu_set_ssd_streaming(true);
    ds4_gpu_set_streaming_expert_cache_budget(EXPERTS);
    ds4_gpu_set_streaming_expert_cache_expert_bytes(
        2u * MID * sizeof(iq2_block) + OUTPUT * 2u * sizeof(q2_block));
    ok = ds4_gpu_tensor_write(xt, 0, x, xb) && ds4_gpu_tensor_write(it, 0, ids, ib) &&
         ds4_gpu_tensor_write(wt, 0, weights, ib);
    const uint32_t sizes[] = {169, 170, 171, 257, 512};
    for (unsigned c = 0; ok && c < sizeof(sizes) / sizeof(*sizes); c++) {
        for (int split = 1; split >= 0 && ok; split--) {
            for (uint32_t start = 0; start < sizes[c] && ok;) {
                uint32_t n = sizes[c] - start;
                if (split && n > 128) n = 128;
                ds4_gpu_tensor *xr = ds4_gpu_tensor_view(xt, (uint64_t)start * INPUT * 4u, (uint64_t)n * INPUT * 4u);
                ds4_gpu_tensor *ir = ds4_gpu_tensor_view(it, (uint64_t)start * SELECTED * 4u, (uint64_t)n * SELECTED * 4u);
                ds4_gpu_tensor *wr = ds4_gpu_tensor_view(wt, (uint64_t)start * SELECTED * 4u, (uint64_t)n * SELECTED * 4u);
                bool half = true;
                ok = xr && ir && wr && ds4_gpu_begin_commands() &&
                    ds4_gpu_routed_moe_batch_tensor(out, gate, up, mid, experts,
                        model, bytes, 0, up_off, down_off, 16, 10,
                        MID * sizeof(iq2_block), sizeof(iq2_block),
                        OUTPUT * 2u * sizeof(q2_block), 2u * sizeof(q2_block),
                        INPUT, MID, OUTPUT, ir, wr, EXPERTS, SELECTED, 7.0f,
                        xr, 0, n, &half, false) && !half;
                if (ds4_gpu_commands_active() && !ds4_gpu_end_commands()) ok = 0;
                if (ok) ok = ds4_gpu_tensor_read(out, 0,
                    (split ? reference : actual) + (uint64_t)start * OUTPUT, (uint64_t)n * OUTPUT * 4u);
                ds4_gpu_tensor_free(xr); ds4_gpu_tensor_free(ir); ds4_gpu_tensor_free(wr);
                start += n;
            }
        }
        for (uint32_t i = 0; ok && i < sizes[c] * OUTPUT; i++)
            ok = isfinite(actual[i]) && reference[i] == actual[i];
        fprintf(stderr, "SSD IQ2/Q2 batch versus split batches, rows=%u: %s\n", sizes[c], ok ? "PASS" : "FAIL");
    }
done:
    ds4_gpu_tensor_free(xt); ds4_gpu_tensor_free(it); ds4_gpu_tensor_free(wt);
    ds4_gpu_tensor_free(gate); ds4_gpu_tensor_free(up); ds4_gpu_tensor_free(mid);
    ds4_gpu_tensor_free(experts); ds4_gpu_tensor_free(out);
    free(x); free(weights); free(reference); free(actual); free(ids);
    ds4_gpu_set_model_fd(-1);
    if (source) fclose(source);
    return ok;
}

/* Exercise the existing 32-row tile at its occupied-row boundaries, then
 * time the same arithmetic at V4.1 dimensions. No model inference is needed. */
static int check_tail_case(const void *model, uint64_t bytes,
                           uint64_t up_off, uint64_t down_off,
                           uint32_t tokens, int skewed, uint32_t occupancy, int benchmark) {
    enum { GUARD_BYTES = 256 };
    if (tokens < 32 || (occupancy && (occupancy > 32 || occupancy > tokens || EXPERTS < 2 * SELECTED)))
        return 0;
    const uint64_t pairs = (uint64_t)tokens * SELECTED;
    const uint64_t xb = (uint64_t)tokens * INPUT * sizeof(float);
    const uint64_t mb = pairs * MID * sizeof(float);
    const uint64_t eb = pairs * OUTPUT * sizeof(float);
    const uint64_t ob = (uint64_t)tokens * OUTPUT * sizeof(float);
    const uint64_t ib = pairs * sizeof(int32_t);
    const uint64_t sizes[] = {xb, ib, ib, mb, mb, mb, eb, ob};
    const uint64_t consumed[] = {mb / 2, eb, ob};
    ds4_gpu_tensor *t[8] = {0};
    ds4_gpu_tensor *storage[3] = {0};
    uint8_t guard[GUARD_BYTES], observed[GUARD_BYTES];
    memset(guard, 0xa5, sizeof(guard));
    float *x = malloc(xb), *weights = malloc(ib);
    int32_t *ids = malloc(ib);
    void *reference[3] = {malloc(mb / 2), malloc(eb), malloc(ob)};
    void *actual = malloc(mb > eb ? mb : eb);
    int ok = x && weights && ids && actual && reference[0] && reference[1] && reference[2];
    for (unsigned i = 0; i < 5 && ok; i++) ok = (t[i] = ds4_gpu_tensor_alloc(sizes[i])) != NULL;
    for (unsigned i = 0; i < 3 && ok; i++) {
        storage[i] = ds4_gpu_tensor_alloc(sizes[i + 5] + 2 * GUARD_BYTES);
        ok = storage[i] && (t[i + 5] = ds4_gpu_tensor_view(storage[i], GUARD_BYTES, sizes[i + 5])) != NULL;
    }
    if (!ok) goto done;
    for (uint64_t i = 0; i < xb / sizeof(float); i++) {
        x[i] = ((int)(random_u32() % 101) - 50) / 256.0f;
        // Distinguish the copy's half rounding from merely copying inputs
        // already exactly representable in half. Include signed ties and
        // both neighbors, plus a half-subnormal boundary.
        const float tie = 1.0f + 0x1p-11f;
        switch (i % 32u) {
        case 0: x[i] = tie; break;
        case 1: x[i] = nextafterf(tie, INFINITY); break;
        case 2: x[i] = nextafterf(tie, -INFINITY); break;
        case 3: x[i] = -tie; break;
        case 4: x[i] = nextafterf(-tie, -INFINITY); break;
        case 5: x[i] = 0x1p-25f; break;
        case 6: x[i] = nextafterf(0x1p-25f, INFINITY); break;
        }
    }
    for (uint32_t row = 0; row < tokens; row++) for (uint32_t s = 0; s < SELECTED; s++) {
        /* Controlled routes give each of the first six experts exactly
         * occupancy rows; disjoint experts consume the rest. Global batches
         * stay >=32 so sparse experts exercise grouped MMA, not tiny matvec. */
        ids[(uint64_t)row * SELECTED + s] = occupancy ? (int32_t)(s + (row < occupancy ? 0 : SELECTED)) :
            skewed == 2 ? (int32_t)((row * SELECTED + s) % EXPERTS) :
            !skewed ? (int32_t)s :
            s == 0 ? 0 : 1 + (row * 7 + s * 19) % (EXPERTS - 17);
        weights[(uint64_t)row * SELECTED + s] = (s + 1) / 21.0f;
    }
    if (occupancy) {
        uint32_t counts[2 * SELECTED] = {0};
        for (uint64_t i = 0; i < pairs; i++) counts[ids[i]]++;
        for (unsigned i = 0; i < 2 * SELECTED; i++) {
            if (counts[i] != (i < SELECTED ? occupancy : tokens - occupancy)) {
                fprintf(stderr, "V4.1 tail invalid occupancy expert=%u count=%u\n", i, counts[i]);
                ok = 0;
            }
        }
        if (!ok) goto done;
    }
    if (benchmark) {
        uint32_t *counts = calloc(EXPERTS, sizeof(*counts));
        uint32_t buckets[3] = {0};
        if (!counts) { ok = 0; goto done; }
        for (uint64_t i = 0; i < pairs; i++) counts[ids[i]]++;
        for (uint32_t e = 0; e < EXPERTS; e++) {
            buckets[2] += counts[e] / 32u;
            const uint32_t tail = counts[e] % 32u;
            if (tail) buckets[tail <= 8u ? 0 : tail <= 16u ? 1 : 2]++;
        }
        free(counts);
        fprintf(stderr, "V4.1 tail routes rows=%u skewed=%d n8=%u n16=%u n32=%u\n",
            tokens, skewed, buckets[0], buckets[1], buckets[2]);
    }
    ok = ds4_gpu_tensor_write(t[0], 0, x, xb) && ds4_gpu_tensor_write(t[1], 0, ids, ib) &&
         ds4_gpu_tensor_write(t[2], 0, weights, ib);
    const uint32_t cull16 = DS4_GPU_TEST_V41_MOE_REFERENCE |
        DS4_GPU_TEST_V41_PAIR_TAIL_CULL | DS4_GPU_TEST_V41_DOWN_TAIL_CULL;
    const uint32_t small = cull16 | DS4_GPU_TEST_V41_MOE_SMALL_TILES;
    // Q2_K128 now changes only the N8 bucket; N16/N32 remain release kernels.
    // Keep mixed occupancy cases to verify both sides of that dispatch.
    const uint32_t flags[] = {DS4_GPU_TEST_V41_MOE_REFERENCE,
        DS4_GPU_TEST_V41_MOE_REFERENCE | DS4_GPU_TEST_V41_PAIR_TAIL_CULL,
        DS4_GPU_TEST_V41_MOE_REFERENCE | DS4_GPU_TEST_V41_DOWN_TAIL_CULL,
        cull16, 0u, small, small | DS4_GPU_TEST_V41_MOE_RHS_F16,
        small | DS4_GPU_TEST_V41_MOE_Q2_K128,
        small | DS4_GPU_TEST_V41_MOE_RHS_F16 | DS4_GPU_TEST_V41_MOE_Q2_K128};
    enum { VARIANTS = sizeof(flags) / sizeof(flags[0]) };
    /* One full warmup sweep, then five measured sweeps with alternating
     * order. Report their median; every warmup and sample still checks exact
     * outputs and guards. REFERENCE remains a numerical oracle. Compare the release
     * column across builds with the same test and the old/new Metal object
     * to measure the change against production dispatch. All timed variants
     * borrow a caller-owned command buffer, as layer prefill does. */
    enum { SAMPLES = 5 };
    const unsigned explicit_runs = benchmark ? (1u + SAMPLES) * VARIANTS : VARIANTS;
    double samples[VARIANTS][SAMPLES] = {{0}};
    double elapsed[VARIANTS] = {0};
    for (unsigned run = 0; run < explicit_runs + 6u && ok; run++) {
        const bool automatic = run >= explicit_runs;
        const bool force_ssd = run >= explicit_runs + 2u;
        const unsigned variant = force_ssd ? 5u + run - explicit_runs - 2u : automatic ? 0u :
            (run / VARIANTS) % 2u ? VARIANTS - 1u - run % VARIANTS : run % VARIANTS;
        /* Also validate release dispatch against the forced reference, with
         * both resident views and the streamed full-layer binding mode. */
        ds4_gpu_set_ssd_streaming(run >= explicit_runs + 1u);
        ds4_gpu_test_set_flags(force_ssd ? flags[variant] : automatic ? 0u : flags[variant]);
        for (unsigned i = 5; i < 8 && ok; i++)
            ok = ds4_gpu_tensor_fill_f32(t[i], NAN, sizes[i] / sizeof(float));
        /* Mid retains F32 capacity, but its produced half range ends at
         * mb/2. Guard that boundary as well as the F32 output view bounds. */
        for (unsigned i = 0; i < 3 && ok; i++)
            ok = ds4_gpu_tensor_write(storage[i], 0, guard, sizeof(guard)) &&
                 ds4_gpu_tensor_write(storage[i], GUARD_BYTES + consumed[i], guard, sizeof(guard));
        bool half_mid = false;
        const double begin = now_seconds();
        const bool caller_batch = benchmark || force_ssd || (!automatic && variant >= 5u);
        if (caller_batch && ok) ok = ds4_gpu_begin_commands();
        ok = ok && ds4_gpu_routed_moe_batch_tensor(t[7], t[3], t[4], t[5], t[6],
            model, bytes, 0, up_off, down_off, 16, 10,
            (uint64_t)MID * INPUT / 256 * sizeof(iq2_block), INPUT / 256 * sizeof(iq2_block),
            (uint64_t)OUTPUT * MID / 256 * sizeof(q2_block), MID / 256 * sizeof(q2_block),
            INPUT, MID, OUTPUT, t[1], t[2], EXPERTS, SELECTED, 7.0f,
            t[0], 0, tokens, &half_mid, true) && half_mid;
        // The forced arm also covers indirect-argument publication and work
        // buffer retention when MoE borrows an unretained caller-owned CB.
        if (caller_batch && ds4_gpu_commands_active()) ok = ds4_gpu_end_commands() && ok;
        const double ms = (now_seconds() - begin) * 1000;
        if (run >= VARIANTS && !automatic)
            samples[variant][run / VARIANTS - 1u] = ms;
        for (unsigned i = 0; i < 3 && ok; i++) {
            const uint64_t size = consumed[i];
            ok = ds4_gpu_tensor_read(t[i + 5], 0, actual, size);
            for (uint64_t j = 0; j < size / (i ? sizeof(float) : sizeof(uint16_t)) && ok; j++)
                ok = isfinite(i ? ((float *)actual)[j] : (float)((_Float16 *)actual)[j]);
            if (!run) memcpy(reference[i], actual, size);
            else if (ok && memcmp(reference[i], actual, size)) {
                fprintf(stderr, "V4.1 tail mismatch rows=%u skewed=%d occupancy=%u variant=%u tensor=%u\n",
                    tokens, skewed, occupancy, variant, i);
                ok = 0;
            }
            for (unsigned side = 0; side < 2 && ok; side++) {
                ok = ds4_gpu_tensor_read(storage[i], side ? GUARD_BYTES + size : 0,
                    observed, sizeof(observed)) && !memcmp(guard, observed, sizeof(guard));
                if (!ok) fprintf(stderr,
                    "V4.1 tail guard rows=%u occupancy=%u variant=%u tensor=%u side=%u\n",
                    tokens, occupancy, variant, i, side);
            }
        }
        // The transient half input must not modify any caller-owned source.
        const void *inputs[] = {x, ids, weights};
        for (unsigned i = 0; i < 3 && ok; i++) {
            ok = ds4_gpu_tensor_read(t[i], 0, actual, sizes[i]) &&
                !memcmp(inputs[i], actual, sizes[i]);
            if (!ok) fprintf(stderr, "V4.1 tail input modified variant=%u input=%u\n", variant, i);
        }
        if (automatic) fprintf(stderr, "V4.1 tail auto rows=%u ssd=%u force_small=%u: %s\n",
            tokens, run >= explicit_runs + 1u, force_ssd, ok ? "PASS" : "FAIL");
    }
    if (benchmark && ok) {
        for (unsigned variant = 0; variant < VARIANTS; variant++) {
            for (unsigned i = 1; i < SAMPLES; i++) {
                const double value = samples[variant][i];
                unsigned j = i;
                while (j && samples[variant][j - 1u] > value) {
                    samples[variant][j] = samples[variant][j - 1u];
                    j--;
                }
                samples[variant][j] = value;
            }
            elapsed[variant] = samples[variant][SAMPLES / 2u];
        }
        fprintf(stderr,
        "V4.1 tail timing rows=%u skewed=%d stat=median5 warmup=1 sweeps=5 uncull=%.3f pair=%.3f down=%.3f both=%.3f release=%.3f force_small=%.3f rhs_f16=%.3f q2_k128=%.3f rhs_q2=%.3f ms\n",
        tokens, skewed, elapsed[0], elapsed[1], elapsed[2], elapsed[3], elapsed[4], elapsed[5],
        elapsed[6], elapsed[7], elapsed[8]);
    }
done:
    ds4_gpu_test_set_flags(0);
    ds4_gpu_set_ssd_streaming(false);
    for (unsigned i = 0; i < 8; i++) ds4_gpu_tensor_free(t[i]);
    for (unsigned i = 0; i < 3; i++) ds4_gpu_tensor_free(storage[i]);
    for (unsigned i = 0; i < 3; i++) free(reference[i]);
    free(x); free(weights); free(ids); free(actual);
    fprintf(stderr, "V4.1 tail exact rows=%u skewed=%d occupancy=%u: %s\n",
        tokens, skewed, occupancy, ok ? "PASS" : "FAIL");
    return ok;
}

int main(int argc, char **argv) {
    if (argc == 2 && !strcmp(argv[1], "--v41-q4-decode")) {
        if (!ds4_gpu_init()) return 1;
        ds4_gpu_set_quality(false);
        ds4_gpu_set_ssd_streaming(false);
        return check_static_batch(true) ? 0 : 1;
    }
    const bool address = argc == 2 && !strcmp(argv[1], "--ssd-address");
    const bool v41_tp = argc == 2 && !strcmp(argv[1], "--v41-tp");
    const bool v41_decode = argc == 2 && !strcmp(argv[1], "--v41-decode");
    const bool tail = argc == 2 && !strcmp(argv[1], "--v41-tail-cull");
    const bool tail_large = (argc == 2 || argc == 3) && !strcmp(argv[1], "--v41-tail-cull-large");
    uint32_t tail_rows = 0;
    if (tail_large && argc == 3) {
        char *end = NULL;
        const unsigned long rows = strtoul(argv[2], &end, 10);
        if (!end || *end || rows < 1025u || rows > 2048u) {
            fprintf(stderr, "V4.1 large benchmark rows must be in 1025..2048\n");
            return 1;
        }
        tail_rows = (uint32_t)rows;
    }
    const bool tail_small = argc == 2 && !strcmp(argv[1], "--v41-tail-cull-small");
    const bool tail_benchmark = tail || tail_large;
    const bool v41 = v41_tp || v41_decode || tail_benchmark || (argc == 2 && !strcmp(argv[1], "--v41"));
    if (argc != 1 && !address && !v41 && !tail_small) {
        fprintf(stderr, "usage: %s [--ssd-address | --v41 | --v41-tp | --v41-decode | --v41-q4-decode | --v41-tail-cull | --v41-tail-cull-large [rows] | --v41-tail-cull-small]\n", argv[0]);
        return 1;
    }
    if (v41) { INPUT = 5120; MID = 2304; OUTPUT = 5124; EXPERTS = 384; }
    if (v41_decode || tail_benchmark) OUTPUT = 5120;
    const uint64_t page = getpagesize();
    const uint64_t up_off = aligned((uint64_t)EXPERTS * MID * INPUT / 256 * sizeof(iq2_block), page);
    const uint64_t down_off = up_off * 2;
    const uint64_t model_size = aligned(down_off + (uint64_t)EXPERTS * OUTPUT * MID / 256 * sizeof(q2_block), page);
    void *model = NULL;
    if (posix_memalign(&model, page, model_size)) return 1;
    memset(model, 0, model_size);
    for (uint64_t off = 0; off < down_off; off += up_off) {
        iq2_block *w = (iq2_block *)((char *)model + off);
        for (uint64_t b = 0; b < (uint64_t)EXPERTS * MID * INPUT / 256; b++) {
            w[b].d = 0x1400; /* 2^-10, finite nontrivial activations. */
            for (uint32_t j = 0; j < sizeof(w[b].qs); j++) w[b].qs[j] = random_u32();
        }
    }
    q2_block *w = (q2_block *)((char *)model + down_off);
    for (uint64_t b = 0; b < (uint64_t)EXPERTS * OUTPUT * MID / 256; b++) {
        w[b].d = w[b].dmin = 0x2000;
        for (uint32_t j = 0; j < sizeof(w[b].scales); j++) w[b].scales[j] = random_u32();
        for (uint32_t j = 0; j < sizeof(w[b].qs); j++) w[b].qs[j] = random_u32();
    }
    int ok = ds4_gpu_init() && ds4_gpu_set_model_map(model, model_size);
    ds4_gpu_set_quality(false);
    ds4_gpu_set_ssd_streaming(false);
    if (tail_benchmark || tail_small) {
        if (!ds4_gpu_device_is_pre_m5_apple_silicon()) {
            fprintf(stderr, "V4.1 tail oracle requires pre-M5 Apple Silicon\n");
            ok = 0;
        }
        const uint32_t boundary[] = {32, 33, 34, 47, 48, 49, 63, 64, 65, 80, 81,
            1024, 1025, 1241, 2048, 2049};
        const uint32_t timing[] = {128, 437, 1024};
        const uint32_t timing_large[] = {1241, 1280, 2048};
        const uint32_t occupancy[] = {1, 7, 8, 9, 15, 16, 17, 23, 24, 25, 31, 32};
        /* The complementary experts have 64-n rows, covering a full tile
         * followed by a tail, while the first six have only n rows. */
        for (unsigned i = 0; !tail_large && i < sizeof(occupancy) / sizeof(*occupancy) && ok; i++)
            ok = check_tail_case(model, model_size, up_off, down_off, 64, 0, occupancy[i], 0);
        const uint32_t *rows = tail_rows ? &tail_rows : tail_large ? timing_large : tail ? timing : boundary;
        const unsigned n = tail_rows ? 1u : tail_large ? sizeof(timing_large) / sizeof(*timing_large) :
            tail ? sizeof(timing) / sizeof(*timing) : sizeof(boundary) / sizeof(*boundary);
        for (unsigned i = 0; i < n && ok; i++) {
            if (tail_small) ok = check_tail_case(model, model_size, up_off, down_off, rows[i], 0, 0, 0);
            if (ok) ok = check_tail_case(model, model_size, up_off, down_off, rows[i], 1, 0, tail_benchmark);
            /* Balanced 437 rows give every expert exactly 6 or 7 routes;
             * 1024 gives 16, 1241 gives 19/20 and 2048 gives 32. The skewed
             * case also exercises small tails after complete N32 tiles. */
            if (ok && tail_benchmark) ok = check_tail_case(model, model_size, up_off, down_off, rows[i], 2, 0, 1);
        }
        ds4_gpu_cleanup();
        free(model);
        return ok ? 0 : 1;
    }
    if (v41_decode) {
        setenv("DS4_TP_NO_KEEPALIVE", "1", 1);
        for (int rank = -1; rank < 2 && ok; rank++) {
            if (rank >= 0) ok = ds4_gpu_tp_init((uint32_t)rank, NULL, 0, 0, 0, NULL, NULL);
            for (uint32_t rows = 1; rows <= 9 && ok; rows++)
                ok = check_case(model, model_size, up_off, down_off, rows, false, rank);
            if (rank >= 0) ds4_gpu_tp_shutdown();
        }
        unsetenv("DS4_TP_NO_KEEPALIVE");
        ds4_gpu_cleanup();
        free(model);
        return ok ? 0 : 1;
    }
    if (v41_tp) {
        /* Exercise both expert-ownership partitions without network gates. */
        setenv("DS4_TP_NO_KEEPALIVE", "1", 1);
        for (uint32_t rank = 0; rank < 2 && ok; rank++) {
            ok = ds4_gpu_tp_init(rank, NULL, 0, 0, 0, NULL, NULL);
            if (ok) ok = check_case(model, model_size, up_off, down_off, 8192, false, (int)rank);
            ds4_gpu_tp_shutdown();
        }
        unsetenv("DS4_TP_NO_KEEPALIVE");
        ds4_gpu_cleanup();
        free(model);
        return ok ? 0 : 1;
    }
    if (address) {
        if (ok) ok = check_address_batch(model, model_size, up_off, down_off);
        ds4_gpu_cleanup();
        free(model);
        return ok ? 0 : 1;
    }
    const uint32_t sizes[] = {511, 512, 513, 1024, 2048, 4096, 8191, 8192, 8193, 512};
    for (unsigned i = 0; i < sizeof(sizes) / sizeof(*sizes) && ok; i++)
        ok = check_case(model, model_size, up_off, down_off, sizes[i], false, -1);
    if (v41 && ok) {
        /* A streamed full layer must use the same matrix kernels, without
         * copying its experts back into a decode cache. */
        ds4_gpu_set_ssd_streaming(true);
        ds4_gpu_set_streaming_expert_cache_budget(EXPERTS);
        for (unsigned i = 0; i < sizeof(sizes) / sizeof(*sizes) && ok; i++)
            ok = check_case(model, model_size, up_off, down_off, sizes[i], false, -1);
    }
    ds4_gpu_cleanup();
    free(model);
    if (!ok || v41) return ok ? 0 : 1;

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
        if (ok && ds4_gpu_device_is_m5_apple_silicon()) ok = check_static_batch(false);
        ds4_gpu_cleanup();
    }
    return ok ? 0 : 1;
}
