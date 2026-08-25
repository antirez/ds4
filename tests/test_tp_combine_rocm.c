/* Stage-1 tests for the ROCm tensor-parallel combine kernels.
 *
 * The combine (world-of-two partial add) must be bit-exact against a CPU
 * float reference: TP's acceptable divergence lives in the reduction order
 * of the split matvecs, never in the combine.  The spin-combine variants
 * additionally prove the in-band stamp actually gates the combine: the
 * spinning kernel is launched before the payload exists and must wait for
 * the writer (a second-stream kernel, or the host via a pinned stamp).
 *
 * Loopback only: the NHI imported-pool path is exercised by the transport
 * stage on the two-node rig; this suite runs on any single ROCm device.
 */

#include "ds4_gpu.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_checks, g_failures;

static void check(int ok, const char *what) {
    g_checks++;
    if (!ok) {
        g_failures++;
        fprintf(stderr, "FAIL %s\n", what);
    } else {
        printf("ROCm TP %-44s PASS\n", what);
    }
}

static int test_hc_expand_add(void) {
    enum { N_EMBD = 4096, N_HC = 4, HC_VALUES = N_EMBD * N_HC };
    float *a = (float *)malloc(N_EMBD * sizeof(float));
    float *b = (float *)malloc(N_EMBD * sizeof(float));
    float *residual = (float *)malloc(HC_VALUES * sizeof(float));
    float *got = (float *)malloc(HC_VALUES * sizeof(float));
    float *ref = (float *)malloc(HC_VALUES * sizeof(float));
    const float post[N_HC] = {1.0f, 0.5f, -1.0f, 0.25f};
    float comb[N_HC * N_HC] = {0};
    ds4_gpu_tensor *a_t = NULL, *b_t = NULL, *sum_t = NULL;
    ds4_gpu_tensor *residual_t = NULL, *post_t = NULL, *comb_t = NULL;
    ds4_gpu_tensor *got_t = NULL, *ref_t = NULL;
    int ok = a && b && residual && got && ref;
    for (uint32_t h = 0; ok && h < N_HC; h++) comb[h * N_HC + h] = 1.0f;
    for (uint32_t d = 0; ok && d < N_EMBD; d++) {
        a[d] = (float)((int)(d % 7u) - 3);
        b[d] = (float)((int)(d % 5u) - 2);
        for (uint32_t h = 0; h < N_HC; h++)
            residual[(uint64_t)h * N_EMBD + d] =
                (float)((int)((d + h * 3u) % 11u) - 5);
    }
    if (ok) a_t = ds4_gpu_tensor_alloc(N_EMBD * sizeof(float));
    if (ok) b_t = ds4_gpu_tensor_alloc(N_EMBD * sizeof(float));
    if (ok) sum_t = ds4_gpu_tensor_alloc(N_EMBD * sizeof(float));
    if (ok) residual_t = ds4_gpu_tensor_alloc(HC_VALUES * sizeof(float));
    if (ok) post_t = ds4_gpu_tensor_alloc(sizeof(post));
    if (ok) comb_t = ds4_gpu_tensor_alloc(sizeof(comb));
    if (ok) got_t = ds4_gpu_tensor_alloc(HC_VALUES * sizeof(float));
    if (ok) ref_t = ds4_gpu_tensor_alloc(HC_VALUES * sizeof(float));
    ok = ok && a_t && b_t && sum_t && residual_t && post_t && comb_t &&
         got_t && ref_t &&
         ds4_gpu_tensor_write(a_t, 0, a, N_EMBD * sizeof(float)) &&
         ds4_gpu_tensor_write(b_t, 0, b, N_EMBD * sizeof(float)) &&
         ds4_gpu_tensor_write(residual_t, 0, residual,
                              HC_VALUES * sizeof(float)) &&
         ds4_gpu_tensor_write(post_t, 0, post, sizeof(post)) &&
         ds4_gpu_tensor_write(comb_t, 0, comb, sizeof(comb)) &&
         ds4_gpu_hc_expand_add_tensor(got_t, a_t, b_t, residual_t,
                                      post_t, comb_t, N_EMBD, N_HC) &&
         ds4_gpu_add_tensor(sum_t, a_t, b_t, N_EMBD) &&
         ds4_gpu_hc_expand_tensor(ref_t, sum_t, residual_t,
                                  post_t, comb_t, N_EMBD, N_HC) &&
         ds4_gpu_tensor_read(got_t, 0, got, HC_VALUES * sizeof(float)) &&
         ds4_gpu_tensor_read(ref_t, 0, ref, HC_VALUES * sizeof(float));
    if (ok) ok = memcmp(got, ref, HC_VALUES * sizeof(float)) == 0;

    ds4_gpu_tensor_free(ref_t);
    ds4_gpu_tensor_free(got_t);
    ds4_gpu_tensor_free(comb_t);
    ds4_gpu_tensor_free(post_t);
    ds4_gpu_tensor_free(residual_t);
    ds4_gpu_tensor_free(sum_t);
    ds4_gpu_tensor_free(b_t);
    ds4_gpu_tensor_free(a_t);
    free(ref);
    free(got);
    free(residual);
    free(b);
    free(a);
    return ok;
}

static void fill_q8_block(unsigned char *block, uint32_t seed) {
    block[0] = 0x00u;
    block[1] = 0x3cu; /* IEEE fp16 1.0 */
    int8_t *q = (int8_t *)(block + 2u);
    for (uint32_t i = 0; i < 32u; i++)
        q[i] = (int8_t)((int)((seed + i * 3u) % 7u) - 3);
}

/* Exercise the exact Flash shared-expert geometry used between the ATTN and
 * FFN gates: each rank writes a compact 1024-lane gate/up/SwiGLU half, then
 * expands the matching K slice of the 2048x4096 Q8 down projection. */
static int test_shared_slice(void) {
    enum { N_EMBD = 4096, SHARED = 2048, HALF = 1024 };
    const uint64_t gu_row = (N_EMBD / 32u) * 34u;
    const uint64_t down_row = (SHARED / 32u) * 34u;
    const uint64_t gate_bytes = (uint64_t)SHARED * gu_row;
    const uint64_t up_off = (gate_bytes + 4095u) & ~4095ull;
    const uint64_t up_bytes = gate_bytes;
    const uint64_t down_off = (up_off + up_bytes + 4095u) & ~4095ull;
    const uint64_t down_bytes = (uint64_t)N_EMBD * down_row;
    const uint64_t model_bytes = (down_off + down_bytes + 4095u) & ~4095ull;
    unsigned char *model = (unsigned char *)malloc((size_t)model_bytes);
    FILE *model_file = NULL;
    float *x = (float *)malloc(N_EMBD * sizeof(float));
    float *mid_full = (float *)malloc(SHARED * sizeof(float));
    float *mid0 = (float *)malloc(HALF * sizeof(float));
    float *mid1 = (float *)malloc(HALF * sizeof(float));
    float *out_full = (float *)malloc(N_EMBD * sizeof(float));
    float *out_sum = (float *)malloc(N_EMBD * sizeof(float));
    ds4_gpu_tensor *x_t = NULL;
    ds4_gpu_tensor *gate_full_t = NULL, *up_full_t = NULL, *mid_full_t = NULL;
    ds4_gpu_tensor *gate0_t = NULL, *up0_t = NULL, *mid0_t = NULL;
    ds4_gpu_tensor *gate1_t = NULL, *up1_t = NULL, *mid1_t = NULL;
    ds4_gpu_tensor *out_full_t = NULL, *out0_t = NULL, *out1_t = NULL;
    ds4_gpu_tensor *out_sum_t = NULL;
    int ok = model && x && mid_full && mid0 && mid1 && out_full && out_sum;
    if (!ok) goto done;
    memset(model, 0, (size_t)model_bytes);
    const struct { uint64_t off, bytes; uint32_t seed; } ranges[3] = {
        {0, gate_bytes, 17u}, {up_off, up_bytes, 101u},
        {down_off, down_bytes, 211u},
    };
    for (uint32_t r = 0; r < 3; r++) {
        for (uint64_t off = 0, block = 0; off < ranges[r].bytes;
             off += 34u, block++) {
            unsigned char *q = model + ranges[r].off + off;
            fill_q8_block(q, ranges[r].seed + (uint32_t)(block * 7u));
            q[0] = 0x00u;
            q[1] = 0x24u; /* fp16 1/64 */
        }
    }
    for (uint32_t i = 0; i < N_EMBD; i++) {
        x[i] = (float)((int)((i * 37u) % 257u) - 128) * (1.0f / 127.0f) +
               (float)((int)(i % 13u) - 6) * 0.00031f;
    }

    model_file = tmpfile();
    const uint64_t offsets[3] = {0, up_off, down_off};
    const uint64_t sizes[3] = {gate_bytes, up_bytes, down_bytes};
    ok = model_file != NULL &&
         fwrite(model, 1, (size_t)model_bytes, model_file) == model_bytes &&
         fflush(model_file) == 0 &&
         ds4_gpu_set_model_map(model, model_bytes) &&
         ds4_gpu_set_model_fd_for_map(fileno(model_file), model) &&
         ds4_gpu_set_model_map_spans(model, model_bytes, offsets, sizes, 3,
                                     down_bytes);
    if (ok) x_t = ds4_gpu_tensor_alloc(N_EMBD * sizeof(float));
    if (ok) gate_full_t = ds4_gpu_tensor_alloc(SHARED * sizeof(float));
    if (ok) up_full_t = ds4_gpu_tensor_alloc(SHARED * sizeof(float));
    if (ok) mid_full_t = ds4_gpu_tensor_alloc(SHARED * sizeof(float));
    if (ok) gate0_t = ds4_gpu_tensor_alloc(SHARED * sizeof(float));
    if (ok) up0_t = ds4_gpu_tensor_alloc(SHARED * sizeof(float));
    if (ok) mid0_t = ds4_gpu_tensor_alloc(SHARED * sizeof(float));
    if (ok) gate1_t = ds4_gpu_tensor_alloc(SHARED * sizeof(float));
    if (ok) up1_t = ds4_gpu_tensor_alloc(SHARED * sizeof(float));
    if (ok) mid1_t = ds4_gpu_tensor_alloc(SHARED * sizeof(float));
    if (ok) out_full_t = ds4_gpu_tensor_alloc(N_EMBD * sizeof(float));
    if (ok) out0_t = ds4_gpu_tensor_alloc(N_EMBD * sizeof(float));
    if (ok) out1_t = ds4_gpu_tensor_alloc(N_EMBD * sizeof(float));
    if (ok) out_sum_t = ds4_gpu_tensor_alloc(N_EMBD * sizeof(float));
    ok = ok && x_t && gate_full_t && up_full_t && mid_full_t &&
         gate0_t && up0_t && mid0_t && gate1_t && up1_t && mid1_t &&
         out_full_t && out0_t && out1_t && out_sum_t &&
         ds4_gpu_tensor_write(x_t, 0, x, N_EMBD * sizeof(float)) &&
         ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(
             gate_full_t, up_full_t, mid_full_t, model, model_bytes,
             0, up_off, N_EMBD, SHARED, x_t, 10.0f) &&
         ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(
             gate0_t, up0_t, mid0_t, model, model_bytes,
             0, up_off, N_EMBD, HALF, x_t, 10.0f) &&
         ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(
             gate1_t, up1_t, mid1_t, model, model_bytes,
             (uint64_t)HALF * gu_row,
             up_off + (uint64_t)HALF * gu_row,
             N_EMBD, HALF, x_t, 10.0f) &&
         ds4_gpu_matmul_q8_0_tensor(out_full_t, model, model_bytes,
                                    down_off, SHARED, N_EMBD,
                                    mid_full_t, 1) &&
         ds4_gpu_matmul_quant_kslice_tensor(out0_t, model, model_bytes,
                                             down_off, 8u, SHARED, 0, HALF,
                                             N_EMBD, mid0_t, 0) &&
         ds4_gpu_matmul_quant_kslice_tensor(out1_t, model, model_bytes,
                                             down_off, 8u, SHARED, HALF, HALF,
                                             N_EMBD, mid1_t, 0) &&
         ds4_gpu_add_tensor(out_sum_t, out0_t, out1_t, N_EMBD) &&
         ds4_gpu_tensor_read(mid_full_t, 0, mid_full,
                             SHARED * sizeof(float)) &&
         ds4_gpu_tensor_read(mid0_t, 0, mid0, HALF * sizeof(float)) &&
         ds4_gpu_tensor_read(mid1_t, 0, mid1, HALF * sizeof(float)) &&
         ds4_gpu_tensor_read(out_full_t, 0, out_full,
                             N_EMBD * sizeof(float)) &&
         ds4_gpu_tensor_read(out_sum_t, 0, out_sum,
                             N_EMBD * sizeof(float));
    if (ok && ds4_gpu_matmul_quant_kslice_tensor(
            out0_t, model, model_bytes, down_off, 39u,
            SHARED, 0, HALF, N_EMBD, mid0_t, 0)) {
        ok = 0; /* Unsupported typed dispatch must remain fail-closed. */
    }
    if (ok) {
        ok = memcmp(mid_full, mid0, HALF * sizeof(float)) == 0 &&
             memcmp(mid_full + HALF, mid1, HALF * sizeof(float)) == 0;
    }
    if (ok) {
        float max_abs = 0.0f;
        double sum_sq = 0.0;
        uint32_t finite_count = 0;
        int parity_ok = 1;
        for (uint32_t i = 0; i < N_EMBD; i++) {
            if (!isfinite(out_full[i]) || !isfinite(out_sum[i])) {
                parity_ok = 0;
                continue;
            }
            finite_count++;
            const float d = out_full[i] - out_sum[i];
            const float ad = d < 0.0f ? -d : d;
            const float av = out_full[i] < 0.0f ? -out_full[i] : out_full[i];
            if (ad > max_abs) max_abs = ad;
            sum_sq += (double)d * d;
            if (ad > 1.0e-3f * (1.0f + av)) parity_ok = 0;
        }
        const char *prequant = getenv("DS4_TEST_TP_PREQUANT");
        fprintf(stderr,
                "ROCm TP shared-half parity max_abs=%g rmse=%g "
                "finite=%u/%u prequant=%s\n",
                max_abs,
                finite_count ? sqrt(sum_sq / finite_count) : INFINITY,
                finite_count, (uint32_t)N_EMBD,
                prequant && prequant[0] && strcmp(prequant, "0") != 0 ?
                    "on" : "off");
        ok = parity_ok;
    }

done:
    ds4_gpu_tensor_free(out_sum_t);
    ds4_gpu_tensor_free(out1_t);
    ds4_gpu_tensor_free(out0_t);
    ds4_gpu_tensor_free(out_full_t);
    ds4_gpu_tensor_free(mid1_t);
    ds4_gpu_tensor_free(up1_t);
    ds4_gpu_tensor_free(gate1_t);
    ds4_gpu_tensor_free(mid0_t);
    ds4_gpu_tensor_free(up0_t);
    ds4_gpu_tensor_free(gate0_t);
    ds4_gpu_tensor_free(mid_full_t);
    ds4_gpu_tensor_free(up_full_t);
    ds4_gpu_tensor_free(gate_full_t);
    ds4_gpu_tensor_free(x_t);
    if (model_file) fclose(model_file);
    free(out_sum);
    free(out_full);
    free(mid1);
    free(mid0);
    free(mid_full);
    free(x);
    free(model);
    return ok;
}

/* Rank 1's output groups are compact at heads[0], even though their A/B
 * weight window starts at group 1.  Poison the unused upper half so this
 * catches an accidental group0 shift while also exercising Q8 K-slice. */
static int test_attention_slice(void) {
    enum { GD = 32, RANK = 32, GROUPS = 2, OUT = 16 };
    const uint64_t a_bytes = (uint64_t)GROUPS * RANK * 34u;
    const uint64_t b_off = (a_bytes + 255u) & ~255ull;
    const uint64_t b_bytes = (uint64_t)OUT * GROUPS * 34u;
    const uint64_t model_bytes = (b_off + b_bytes + 4095u) & ~4095ull;
    unsigned char *model = (unsigned char *)calloc(1, (size_t)model_bytes);
    FILE *model_file = NULL;
    float heads[GROUPS * GD], expected[OUT], actual[OUT];
    ds4_gpu_tensor *heads_t = NULL, *low_t = NULL, *out_t = NULL;
    int ok = model != NULL;
    if (!ok) return 0;

    for (uint32_t g = 0; g < GROUPS; g++) {
        for (uint32_t r = 0; r < RANK; r++)
            fill_q8_block(model + ((uint64_t)g * RANK + r) * 34u,
                          11u + g * 97u + r * 5u);
    }
    for (uint32_t r = 0; r < OUT; r++) {
        for (uint32_t b = 0; b < GROUPS; b++)
            fill_q8_block(model + b_off + ((uint64_t)r * GROUPS + b) * 34u,
                          23u + r * 13u + b * 71u);
    }
    for (uint32_t i = 0; i < GD; i++) {
        heads[i] = (float)((int)(i % 7u) - 3);
        heads[GD + i] = 1000.0f + (float)i; /* must never be read */
    }
    float low_ref[RANK];
    for (uint32_t r = 0; r < RANK; r++) {
        const int8_t *q = (const int8_t *)(model +
            ((uint64_t)RANK + r) * 34u + 2u);
        float sum = 0.0f;
        for (uint32_t i = 0; i < GD; i++) sum += (float)q[i] * heads[i];
        low_ref[r] = sum;
    }
    for (uint32_t r = 0; r < OUT; r++) {
        const int8_t *q = (const int8_t *)(model + b_off +
            ((uint64_t)r * GROUPS + 1u) * 34u + 2u);
        float sum = 0.0f;
        for (uint32_t i = 0; i < RANK; i++) sum += (float)q[i] * low_ref[i];
        expected[r] = sum;
    }

    setenv("DS4_ROCM_DSV4_PREQUANT_DECODE", "0", 1);
    const uint64_t offsets[2] = {0, b_off};
    const uint64_t sizes[2] = {a_bytes, b_bytes};
    model_file = tmpfile();
    ok = model_file != NULL &&
         fwrite(model, 1, (size_t)model_bytes, model_file) == model_bytes &&
         fflush(model_file) == 0 &&
         ds4_gpu_set_model_map(model, model_bytes) &&
         ds4_gpu_set_model_fd_for_map(fileno(model_file), model) &&
         ds4_gpu_set_model_map_spans(model, model_bytes, offsets, sizes, 2,
                                      a_bytes > b_bytes ? a_bytes : b_bytes);
    if (ok) heads_t = ds4_gpu_tensor_alloc(sizeof(heads));
    if (ok) low_t = ds4_gpu_tensor_alloc((uint64_t)GROUPS * RANK * sizeof(float));
    if (ok) out_t = ds4_gpu_tensor_alloc(sizeof(actual));
    ok = ok && heads_t && low_t && out_t &&
         ds4_gpu_tensor_write(heads_t, 0, heads, sizeof(heads)) &&
         ds4_gpu_attention_output_q8_tp_tensor(
             out_t, low_t, model, model_bytes, 0, b_off,
             GD, RANK, GROUPS, 1, 1, OUT, heads_t) &&
         ds4_gpu_tensor_read(out_t, 0, actual, sizeof(actual));
    if (ok) ok = memcmp(actual, expected, sizeof(actual)) == 0;

    ds4_gpu_tensor_free(out_t);
    ds4_gpu_tensor_free(low_t);
    ds4_gpu_tensor_free(heads_t);
    if (model_file) fclose(model_file);
    free(model);
    return ok;
}

int main(void) {
    srand(20260825);
    const char *prequant_env = getenv("DS4_TEST_TP_PREQUANT");
    const int production_prequant =
        prequant_env && prequant_env[0] && strcmp(prequant_env, "0") != 0;
    if (production_prequant)
        unsetenv("DS4_ROCM_DSV4_PREQUANT_DECODE");
    else
        setenv("DS4_ROCM_DSV4_PREQUANT_DECODE", "0", 1);
    if (!ds4_gpu_init()) {
        fprintf(stderr, "test_tp_combine_rocm: no ROCm device available\n");
        return 1;
    }

    if (production_prequant) {
        check(test_shared_slice(),
              "production-prequant shared halves recombine");
        ds4_gpu_cleanup();
        printf("test_tp_combine_rocm: %d/%d checks passed (%d failed)\n",
               g_checks - g_failures, g_checks, g_failures);
        return g_failures ? 1 : 0;
    }

    /* Bit-exact combine at DS4's hidden size, an odd size, and a small one. */
    check(ds4_gpu_tp_test_combine(7168, 8), "combine 7168 (hidden) bitexact");
    check(ds4_gpu_tp_test_combine(8191, 4), "combine 8191 (odd) bitexact");
    check(ds4_gpu_tp_test_combine(256, 4), "combine 256 bitexact");

    /* Spin-combine, writer-kernel arm: in-band stamp in a plain pool
     * (kernel-to-kernel visibility goes through L2 and needs no UC). */
    check(ds4_gpu_tp_test_spin_exchange(0, 0, 7168, 16),
          "spin-combine kernel-writer plain pool");

    /* Spin-combine on an UNCACHED pool: the production MTYPE for
     * wave-resident polling of NHI-written slots (contract rule 6). */
    check(ds4_gpu_tp_test_spin_exchange(1, 0, 7168, 16),
          "spin-combine kernel-writer uncached pool");

    /* Host-mediated stamp (pinned control word, contract rule 8), payload
     * arriving by copy engine while the wave spins. */
    check(ds4_gpu_tp_test_spin_exchange(1, 1, 7168, 8),
          "spin-combine host-stamp uncached pool");

    check(test_hc_expand_add(),
          "TP folded HC expand-add matches explicit combine");

    check(test_shared_slice(),
          "Flash shared-expert halves recombine");

    check(test_attention_slice(),
          "rank1 compact-head Q8 attention/K-slice");

    printf("test_tp_combine_rocm: %d/%d checks passed (%d failed)\n",
           g_checks - g_failures, g_checks, g_failures);
    return g_failures ? 1 : 0;
}
