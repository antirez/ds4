/* Qwen 3.8 gated-residual (hyper-connection) and PLE GPU kernels vs scalar
 * host references: grouped RMSNorm, silu/sigmoid mixing glue, stream
 * injection, PLE stream gate (signed-sqrt clamp), and the dilated PLE conv
 * with rolling state (including a chunked continuation).
 */

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

#include "ds4.h"
#include "ds4_gpu.h"

bool ds4_log_is_tty(FILE *fp) {
    (void)fp;
    return false;
}

enum {
    STREAMS = 4,
    EMBD = 512,
    WIDE = STREAMS * EMBD,
    ROWS = 7,
    LOWRANK = 20,
    NORM_OFFSET = 0,
    CONV_OFFSET = NORM_OFFSET + WIDE * 4,
    MODEL_BYTES = 65536,
    PLE_TAPS = 4,
    PLE_DILATION = 3,
    PLE_STATE = 9,
};

static const float EPS = 1e-6f;

static void require_ok(int ok, const char *what) {
    if (!ok) {
        fprintf(stderr, "%s failed\n", what);
        exit(1);
    }
}

static void require_close(const char *what, uint32_t index, float actual,
                          double expected, double tolerance) {
    if (!isfinite(actual) || fabs((double)actual - expected) > tolerance) {
        fprintf(stderr, "%s[%u]: got %.9g, expected %.9g (tolerance %.9g)\n",
                what, index, (double)actual, expected, tolerance);
        exit(1);
    }
}

static float pseudo(uint32_t seed, uint32_t i) {
    uint32_t x = seed * 2654435761u + i * 40503u;
    x ^= x >> 13;
    x *= 1274126177u;
    x ^= x >> 16;
    return ((float)(x & 0xffffffu) / (float)0x1000000u) - 0.5f;
}

static double silu(double x) {
    return x / (1.0 + exp(-x));
}

static double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

int main(void) {
    uint8_t *model = mmap(NULL, MODEL_BYTES, PROT_READ | PROT_WRITE,
                          MAP_PRIVATE | MAP_ANON, -1, 0);
    if (model == MAP_FAILED) {
        perror("mmap");
        return 1;
    }
    float *norm_w = (float *)(model + NORM_OFFSET);
    float *conv_w = (float *)(model + CONV_OFFSET);
    for (uint32_t i = 0; i < WIDE; i++)
        norm_w[i] = 0.85f + 0.3f * pseudo(3, i);
    for (uint32_t i = 0; i < WIDE * PLE_TAPS; i++)
        conv_w[i] = 0.4f * pseudo(5, i) + ((i % PLE_TAPS) == 3 ? 0.8f : 0.0f);

    require_ok(ds4_gpu_init(), "GPU initialization");
    require_ok(ds4_gpu_set_model_map(model, MODEL_BYTES),
               "model map registration");

    static float xs[ROWS * WIDE];
    for (uint32_t i = 0; i < ROWS * WIDE; i++) xs[i] = 1.5f * pseudo(7, i);

    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(sizeof(xs));
    ds4_gpu_tensor *normed = ds4_gpu_tensor_alloc(sizeof(xs));
    require_ok(x && normed, "norm tensor allocation");
    require_ok(ds4_gpu_tensor_write(x, 0, xs, sizeof(xs)), "x write");
    require_ok(ds4_gpu_qwen38_hc_group_norm(
        normed, x, model, MODEL_BYTES, NORM_OFFSET,
        STREAMS, EMBD, ROWS, EPS), "group norm");
    static float normed_actual[ROWS * WIDE];
    require_ok(ds4_gpu_tensor_read(normed, 0, normed_actual,
                                   sizeof(normed_actual)), "norm read");
    static double normed_host[ROWS * WIDE];
    for (uint32_t r = 0; r < ROWS; r++) {
        for (uint32_t s = 0; s < STREAMS; s++) {
            double sumsq = 0.0;
            const uint32_t base = (r * STREAMS + s) * EMBD;
            for (uint32_t d = 0; d < EMBD; d++)
                sumsq += (double)xs[base + d] * (double)xs[base + d];
            const double scale = 1.0 / sqrt(sumsq / EMBD + (double)EPS);
            for (uint32_t d = 0; d < EMBD; d++) {
                normed_host[base + d] = (double)xs[base + d] * scale *
                                        (double)norm_w[s * EMBD + d];
                require_close("HC group norm", base + d,
                              normed_actual[base + d],
                              normed_host[base + d], 1e-4);
            }
        }
    }

    /* mix_silu: x = silu(x / STREAMS) over [ROWS, LOWRANK]. */
    static float lows[ROWS * LOWRANK];
    for (uint32_t i = 0; i < ROWS * LOWRANK; i++) lows[i] = 3.0f * pseudo(11, i);
    ds4_gpu_tensor *low = ds4_gpu_tensor_alloc(sizeof(lows));
    require_ok(low && ds4_gpu_tensor_write(low, 0, lows, sizeof(lows)),
               "low write");
    require_ok(ds4_gpu_qwen38_hc_mix_silu(low, STREAMS, LOWRANK, ROWS),
               "mix silu");
    static float low_actual[ROWS * LOWRANK];
    require_ok(ds4_gpu_tensor_read(low, 0, low_actual, sizeof(low_actual)),
               "low read");
    for (uint32_t i = 0; i < ROWS * LOWRANK; i++)
        require_close("HC mix silu", i, low_actual[i],
                      silu((double)lows[i] / STREAMS), 1e-5);

    /* combine: mixed = mean_s sigmoid(mix) * normed. */
    static float mixes[ROWS * WIDE];
    for (uint32_t i = 0; i < ROWS * WIDE; i++) mixes[i] = 2.0f * pseudo(13, i);
    ds4_gpu_tensor *mix = ds4_gpu_tensor_alloc(sizeof(mixes));
    ds4_gpu_tensor *mixed = ds4_gpu_tensor_alloc(
        (uint64_t)ROWS * EMBD * sizeof(float));
    require_ok(mix && mixed, "combine tensor allocation");
    require_ok(ds4_gpu_tensor_write(mix, 0, mixes, sizeof(mixes)), "mix write");
    require_ok(ds4_gpu_qwen38_hc_combine(mixed, mix, normed,
                                         STREAMS, EMBD, ROWS), "combine");
    static float mixed_actual[ROWS * EMBD];
    require_ok(ds4_gpu_tensor_read(mixed, 0, mixed_actual,
                                   sizeof(mixed_actual)), "mixed read");
    for (uint32_t r = 0; r < ROWS; r++) {
        for (uint32_t d = 0; d < EMBD; d++) {
            double acc = 0.0;
            for (uint32_t s = 0; s < STREAMS; s++) {
                const uint32_t i = (r * STREAMS + s) * EMBD + d;
                acc += sigmoid((double)mixes[i]) * normed_host[i];
            }
            require_close("HC combine", r * EMBD + d,
                          mixed_actual[r * EMBD + d], acc / STREAMS, 1e-4);
        }
    }

    /* inject: h += y (x) 2*sigmoid(inject / STREAMS). */
    static float ys[ROWS * EMBD], injs[ROWS * STREAMS], hs[ROWS * WIDE];
    for (uint32_t i = 0; i < ROWS * EMBD; i++) ys[i] = pseudo(17, i);
    for (uint32_t i = 0; i < ROWS * STREAMS; i++) injs[i] = 2.5f * pseudo(19, i);
    memcpy(hs, xs, sizeof(hs));
    ds4_gpu_tensor *y = ds4_gpu_tensor_alloc(sizeof(ys));
    ds4_gpu_tensor *inj = ds4_gpu_tensor_alloc(sizeof(injs));
    require_ok(y && inj, "inject tensor allocation");
    require_ok(ds4_gpu_tensor_write(y, 0, ys, sizeof(ys)), "y write");
    require_ok(ds4_gpu_tensor_write(inj, 0, injs, sizeof(injs)), "inject write");
    require_ok(ds4_gpu_tensor_write(x, 0, hs, sizeof(hs)), "h write");
    require_ok(ds4_gpu_qwen38_hc_inject(x, y, inj, STREAMS, EMBD, ROWS),
               "inject");
    static float h_actual[ROWS * WIDE];
    require_ok(ds4_gpu_tensor_read(x, 0, h_actual, sizeof(h_actual)), "h read");
    for (uint32_t r = 0; r < ROWS; r++) {
        for (uint32_t s = 0; s < STREAMS; s++) {
            const double w = 2.0 * sigmoid((double)injs[r * STREAMS + s] / STREAMS);
            for (uint32_t d = 0; d < EMBD; d++) {
                const uint32_t i = (r * STREAMS + s) * EMBD + d;
                require_close("HC inject", i, h_actual[i],
                              (double)hs[i] + (double)ys[r * EMBD + d] * w,
                              1e-4);
            }
        }
    }

    /* PLE gate: signed-sqrt-scaled dot gate times the shared value. */
    static float keys[ROWS * WIDE], queries[ROWS * WIDE], vals[ROWS * EMBD];
    for (uint32_t i = 0; i < ROWS * WIDE; i++) {
        keys[i] = 0.6f * pseudo(23, i);
        queries[i] = 0.6f * pseudo(29, i);
    }
    for (uint32_t i = 0; i < ROWS * EMBD; i++) vals[i] = pseudo(31, i);
    /* Force a near-zero dot on one (row, stream) to hit the clamp. */
    for (uint32_t d = 0; d < EMBD; d++) queries[(2 * STREAMS + 1) * EMBD + d] = 0.0f;
    ds4_gpu_tensor *kt = ds4_gpu_tensor_alloc(sizeof(keys));
    ds4_gpu_tensor *qt = ds4_gpu_tensor_alloc(sizeof(queries));
    ds4_gpu_tensor *vt = ds4_gpu_tensor_alloc(sizeof(vals));
    ds4_gpu_tensor *gated = ds4_gpu_tensor_alloc(sizeof(keys));
    require_ok(kt && qt && vt && gated, "gate tensor allocation");
    require_ok(ds4_gpu_tensor_write(kt, 0, keys, sizeof(keys)), "keys write");
    require_ok(ds4_gpu_tensor_write(qt, 0, queries, sizeof(queries)),
               "queries write");
    require_ok(ds4_gpu_tensor_write(vt, 0, vals, sizeof(vals)), "values write");
    require_ok(ds4_gpu_qwen38_ple_gate(gated, kt, qt, vt,
                                       STREAMS, EMBD, ROWS), "ple gate");
    static float gated_actual[ROWS * WIDE];
    require_ok(ds4_gpu_tensor_read(gated, 0, gated_actual,
                                   sizeof(gated_actual)), "gated read");
    static double gated_host[ROWS * WIDE];
    for (uint32_t r = 0; r < ROWS; r++) {
        for (uint32_t s = 0; s < STREAMS; s++) {
            const uint32_t base = (r * STREAMS + s) * EMBD;
            double dot = 0.0;
            for (uint32_t d = 0; d < EMBD; d++)
                dot += (double)keys[base + d] * (double)queries[base + d];
            dot /= sqrt((double)EMBD);
            double magnitude = sqrt(fmax(fabs(dot), 1e-6));
            const double gate = sigmoid(dot < 0.0 ? -magnitude : magnitude);
            for (uint32_t d = 0; d < EMBD; d++) {
                gated_host[base + d] = gate * (double)vals[r * EMBD + d];
                require_close("PLE gate", base + d, gated_actual[base + d],
                              gated_host[base + d], 1e-4);
            }
        }
    }

    /* PLE conv: dilated depthwise conv over rows with rolling state, added
     * to the raw gated values; batch run vs host, then chunked (4 + 3). */
    static float gn[ROWS * WIDE];
    for (uint32_t i = 0; i < ROWS * WIDE; i++) gn[i] = 1.2f * pseudo(37, i);
    static double conv_expected[ROWS * WIDE];
    {
        static double state[PLE_STATE][WIDE];
        memset(state, 0, sizeof(state));
        for (uint32_t r = 0; r < ROWS; r++) {
            for (uint32_t c = 0; c < WIDE; c++) {
                double acc = (double)gn[r * WIDE + c] *
                             (double)conv_w[c * PLE_TAPS + PLE_TAPS - 1];
                for (uint32_t w = 0; w + 1 < PLE_TAPS; w++) {
                    acc += state[w * PLE_DILATION][c] *
                           (double)conv_w[c * PLE_TAPS + w];
                }
                conv_expected[r * WIDE + c] =
                    gated_host[r * WIDE + c] + silu(acc);
            }
            for (uint32_t s2 = 0; s2 + 1 < PLE_STATE; s2++) {
                for (uint32_t c = 0; c < WIDE; c++)
                    state[s2][c] = state[s2 + 1][c];
            }
            for (uint32_t c = 0; c < WIDE; c++)
                state[PLE_STATE - 1][c] = (double)gn[r * WIDE + c];
        }
    }
    ds4_gpu_tensor *gnorm = ds4_gpu_tensor_alloc(sizeof(gn));
    ds4_gpu_tensor *conv_state = ds4_gpu_tensor_alloc(
        (uint64_t)PLE_STATE * WIDE * sizeof(float));
    require_ok(gnorm && conv_state, "conv tensor allocation");
    require_ok(ds4_gpu_tensor_write(gnorm, 0, gn, sizeof(gn)), "gnorm write");
    require_ok(ds4_gpu_tensor_write(gated, 0, gated_actual,
                                    sizeof(gated_actual)), "gated reset");
    require_ok(ds4_gpu_tensor_fill_f32(conv_state, 0.0f,
        (uint64_t)PLE_STATE * WIDE), "conv state clear");
    require_ok(ds4_gpu_qwen38_ple_conv(
        gated, gnorm, conv_state, model, MODEL_BYTES, CONV_OFFSET,
        STREAMS, EMBD, ROWS), "ple conv");
    static float conv_actual[ROWS * WIDE];
    require_ok(ds4_gpu_tensor_read(gated, 0, conv_actual, sizeof(conv_actual)),
               "conv read");
    for (uint32_t i = 0; i < ROWS * WIDE; i++)
        require_close("PLE conv", i, conv_actual[i], conv_expected[i], 2e-4);

    /* Chunked continuation. */
    enum { CHUNK = 4 };
    require_ok(ds4_gpu_tensor_write(gated, 0, gated_actual,
                                    sizeof(gated_actual)), "gated reset 2");
    require_ok(ds4_gpu_tensor_fill_f32(conv_state, 0.0f,
        (uint64_t)PLE_STATE * WIDE), "conv state clear 2");
    require_ok(ds4_gpu_qwen38_ple_conv(
        gated, gnorm, conv_state, model, MODEL_BYTES, CONV_OFFSET,
        STREAMS, EMBD, CHUNK), "ple conv chunk 1");
    /* Second chunk works on the tail views of the same buffers. */
    static float gated_tail[(ROWS - CHUNK) * WIDE];
    static float gn_tail[(ROWS - CHUNK) * WIDE];
    memcpy(gated_tail, gated_actual + CHUNK * WIDE, sizeof(gated_tail));
    memcpy(gn_tail, gn + CHUNK * WIDE, sizeof(gn_tail));
    ds4_gpu_tensor *gated2 = ds4_gpu_tensor_alloc(sizeof(gated_tail));
    ds4_gpu_tensor *gnorm2 = ds4_gpu_tensor_alloc(sizeof(gn_tail));
    require_ok(gated2 && gnorm2, "chunk tensor allocation");
    require_ok(ds4_gpu_tensor_write(gated2, 0, gated_tail, sizeof(gated_tail)),
               "chunk gated write");
    require_ok(ds4_gpu_tensor_write(gnorm2, 0, gn_tail, sizeof(gn_tail)),
               "chunk gnorm write");
    require_ok(ds4_gpu_qwen38_ple_conv(
        gated2, gnorm2, conv_state, model, MODEL_BYTES, CONV_OFFSET,
        STREAMS, EMBD, ROWS - CHUNK), "ple conv chunk 2");
    static float chunk_actual[(ROWS - CHUNK) * WIDE];
    require_ok(ds4_gpu_tensor_read(gated2, 0, chunk_actual,
                                   sizeof(chunk_actual)), "chunk read");
    for (uint32_t i = 0; i < (ROWS - CHUNK) * WIDE; i++)
        require_close("PLE conv chunked", i, chunk_actual[i],
                      conv_expected[CHUNK * WIDE + i], 2e-4);

    /* Router: top-10 of 512 by logit, softmax weights renormalized over
     * the selected set. */
    enum { EXPERTS = 512, TOPK = 10, RROWS = 5 };
    static float logits[RROWS * EXPERTS];
    for (uint32_t i = 0; i < RROWS * EXPERTS; i++)
        logits[i] = 4.0f * pseudo(41, i);
    ds4_gpu_tensor *lg = ds4_gpu_tensor_alloc(sizeof(logits));
    ds4_gpu_tensor *sel = ds4_gpu_tensor_alloc(
        (uint64_t)RROWS * TOPK * sizeof(int32_t));
    ds4_gpu_tensor *wts = ds4_gpu_tensor_alloc(
        (uint64_t)RROWS * TOPK * sizeof(float));
    require_ok(lg && sel && wts, "router tensor allocation");
    require_ok(ds4_gpu_tensor_write(lg, 0, logits, sizeof(logits)),
               "logits write");
    require_ok(ds4_gpu_qwen38_router_select_tensor(sel, wts, lg,
                                                   EXPERTS, TOPK, RROWS),
               "router select");
    static int32_t sel_actual[RROWS * TOPK];
    static float wts_actual[RROWS * TOPK];
    require_ok(ds4_gpu_tensor_read(sel, 0, sel_actual, sizeof(sel_actual)),
               "selected read");
    require_ok(ds4_gpu_tensor_read(wts, 0, wts_actual, sizeof(wts_actual)),
               "weights read");
    for (uint32_t r = 0; r < RROWS; r++) {
        /* Host top-k by logit (stable on index for exact ties). */
        int32_t order[EXPERTS];
        for (uint32_t e = 0; e < EXPERTS; e++) order[e] = (int32_t)e;
        for (uint32_t a = 0; a < TOPK; a++) {
            uint32_t best = a;
            for (uint32_t b = a + 1; b < EXPERTS; b++) {
                const float la = logits[r * EXPERTS + order[best]];
                const float lb = logits[r * EXPERTS + order[b]];
                if (lb > la || (lb == la && order[b] < order[best])) best = b;
            }
            const int32_t tmp = order[a];
            order[a] = order[best];
            order[best] = tmp;
        }
        double top = (double)logits[r * EXPERTS + order[0]];
        double sum = 0.0;
        for (uint32_t a = 0; a < TOPK; a++)
            sum += exp((double)logits[r * EXPERTS + order[a]] - top);
        for (uint32_t a = 0; a < TOPK; a++) {
            if (sel_actual[r * TOPK + a] != order[a]) {
                fprintf(stderr, "router selection mismatch row %u slot %u: "
                        "got %d, expected %d\n", r, a,
                        sel_actual[r * TOPK + a], order[a]);
                return 1;
            }
            const double expected_w =
                exp((double)logits[r * EXPERTS + order[a]] - top) / sum;
            require_close("router weight", r * TOPK + a,
                          wts_actual[r * TOPK + a], expected_w, 1e-5);
        }
    }

    ds4_gpu_cleanup();
    munmap(model, MODEL_BYTES);
    puts("Qwen3.8 HC/PLE GPU tests: PASS");
    return 0;
}
