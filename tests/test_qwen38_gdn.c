/* Qwen 3.8 Gated DeltaNet GPU kernels vs a scalar host reference.
 *
 * The host model mirrors the reference implementation: depthwise causal
 * conv-4 + SiLU on q/k/v, L2-normalized q/k (eps 1e-6) with the 1/sqrt(128)
 * query scale, per-value-head decay exp(-exp(A_log) * softplus(a + dt_bias)),
 * sigmoid beta, the delta-rule state update, and a per-head RMSNorm with a
 * sigmoid output gate. QK heads are shared by ratio = n_v_heads / n_qk_heads
 * consecutive value heads.
 *
 * Checks: batch pass vs host, token-by-token passes vs host (which also
 * proves decode/prefill consistency), and chunked-state continuation.
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
    D = 128,
    QK_HEADS = 2,
    V_HEADS = 6,
    RATIO = V_HEADS / QK_HEADS,
    QK_PROJ = QK_HEADS * D,
    V_PROJ = V_HEADS * D,
    TOKENS = 17,
    HISTORY = 3,
    CONV_ELEMENTS = HISTORY * (2 * QK_PROJ + V_PROJ),
    STATE_ELEMENTS = V_PROJ * D,
    Q_CONV_OFFSET = 0,
    K_CONV_OFFSET = Q_CONV_OFFSET + QK_PROJ * 4 * 4,
    V_CONV_OFFSET = K_CONV_OFFSET + QK_PROJ * 4 * 4,
    A_LOG_OFFSET = V_CONV_OFFSET + V_PROJ * 4 * 4,
    DT_BIAS_OFFSET = A_LOG_OFFSET + V_HEADS * 4,
    NORM_OFFSET = DT_BIAS_OFFSET + V_HEADS * 4,
    MODEL_BYTES = 65536,
};

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

static double silu(double x) {
    return x / (1.0 + exp(-x));
}

static double softplus(double x) {
    return x > 20.0 ? x : log(1.0 + exp(x));
}

/* Host reference state. */
typedef struct {
    double conv_q[HISTORY][QK_PROJ];
    double conv_k[HISTORY][QK_PROJ];
    double conv_v[HISTORY][V_PROJ];
    double state[V_HEADS][D][D]; /* [head][value][key] */
} host_gdn_state;

static void host_gdn_step(
        host_gdn_state *hs,
        const float *q_conv, const float *k_conv, const float *v_conv,
        const float *a_log, const float *dt_bias, const float *norm,
        const float *q_raw, const float *k_raw, const float *v_raw,
        const float *alpha_raw, const float *beta_raw, const float *z_raw,
        double *out) {
    double q[QK_PROJ], k[QK_PROJ], v[V_PROJ];

    for (uint32_t c = 0; c < QK_PROJ; c++) {
        double q_acc = 0.0, k_acc = 0.0;
        for (uint32_t w = 0; w < HISTORY; w++) {
            q_acc += hs->conv_q[w][c] * (double)q_conv[c * 4 + w];
            k_acc += hs->conv_k[w][c] * (double)k_conv[c * 4 + w];
        }
        q_acc += (double)q_raw[c] * (double)q_conv[c * 4 + 3];
        k_acc += (double)k_raw[c] * (double)k_conv[c * 4 + 3];
        hs->conv_q[0][c] = hs->conv_q[1][c];
        hs->conv_q[1][c] = hs->conv_q[2][c];
        hs->conv_q[2][c] = (double)q_raw[c];
        hs->conv_k[0][c] = hs->conv_k[1][c];
        hs->conv_k[1][c] = hs->conv_k[2][c];
        hs->conv_k[2][c] = (double)k_raw[c];
        q[c] = silu(q_acc);
        k[c] = silu(k_acc);
    }
    for (uint32_t c = 0; c < V_PROJ; c++) {
        double v_acc = 0.0;
        for (uint32_t w = 0; w < HISTORY; w++) {
            v_acc += hs->conv_v[w][c] * (double)v_conv[c * 4 + w];
        }
        v_acc += (double)v_raw[c] * (double)v_conv[c * 4 + 3];
        hs->conv_v[0][c] = hs->conv_v[1][c];
        hs->conv_v[1][c] = hs->conv_v[2][c];
        hs->conv_v[2][c] = (double)v_raw[c];
        v[c] = silu(v_acc);
    }

    for (uint32_t g = 0; g < QK_HEADS; g++) {
        double q_sumsq = 0.0, k_sumsq = 0.0;
        for (uint32_t d = 0; d < D; d++) {
            q_sumsq += q[g * D + d] * q[g * D + d];
            k_sumsq += k[g * D + d] * k[g * D + d];
        }
        const double q_scale =
            (1.0 / sqrt(q_sumsq + 1.0e-6)) * (1.0 / sqrt((double)D));
        const double k_scale = 1.0 / sqrt(k_sumsq + 1.0e-6);
        for (uint32_t d = 0; d < D; d++) {
            q[g * D + d] *= q_scale;
            k[g * D + d] *= k_scale;
        }
    }

    for (uint32_t h = 0; h < V_HEADS; h++) {
        const uint32_t g = h / RATIO;
        const double decay = exp(-exp((double)a_log[h]) *
                                 softplus((double)alpha_raw[h] +
                                          (double)dt_bias[h]));
        const double beta = 1.0 / (1.0 + exp(-(double)beta_raw[h]));
        double head_out[D];
        for (uint32_t value = 0; value < D; value++) {
            double hk = 0.0;
            for (uint32_t key = 0; key < D; key++) {
                hs->state[h][value][key] *= decay;
                hk += hs->state[h][value][key] * k[g * D + key];
            }
            const double delta = (v[h * D + value] - hk) * beta;
            double hq = 0.0;
            for (uint32_t key = 0; key < D; key++) {
                hs->state[h][value][key] += k[g * D + key] * delta;
                hq += hs->state[h][value][key] * q[g * D + key];
            }
            head_out[value] = hq;
        }
        double sumsq = 0.0;
        for (uint32_t value = 0; value < D; value++)
            sumsq += head_out[value] * head_out[value];
        const double scale = 1.0 / sqrt(sumsq / (double)D + 1.0e-6);
        for (uint32_t value = 0; value < D; value++) {
            const double z = (double)z_raw[h * D + value];
            out[h * D + value] = head_out[value] * scale *
                (double)norm[value] / (1.0 + exp(-z));
        }
    }
}

static float pseudo(uint32_t seed, uint32_t i) {
    /* Cheap deterministic pattern in about [-0.5, 0.5]. */
    uint32_t x = seed * 2654435761u + i * 40503u;
    x ^= x >> 13;
    x *= 1274126177u;
    x ^= x >> 16;
    return ((float)(x & 0xffffffu) / (float)0x1000000u) - 0.5f;
}

int main(void) {
    uint8_t *model = mmap(NULL, MODEL_BYTES, PROT_READ | PROT_WRITE,
                          MAP_PRIVATE | MAP_ANON, -1, 0);
    if (model == MAP_FAILED) {
        perror("mmap");
        return 1;
    }
    float *q_conv = (float *)(model + Q_CONV_OFFSET);
    float *k_conv = (float *)(model + K_CONV_OFFSET);
    float *v_conv = (float *)(model + V_CONV_OFFSET);
    float *a_log = (float *)(model + A_LOG_OFFSET);
    float *dt_bias = (float *)(model + DT_BIAS_OFFSET);
    float *norm = (float *)(model + NORM_OFFSET);
    for (uint32_t c = 0; c < QK_PROJ; c++) {
        for (uint32_t w = 0; w < 4; w++) {
            q_conv[c * 4 + w] = 0.2f * pseudo(11, c * 4 + w) +
                                (w == 3 ? 0.9f : 0.0f);
            k_conv[c * 4 + w] = 0.2f * pseudo(13, c * 4 + w) +
                                (w == 3 ? 0.9f : 0.0f);
        }
    }
    for (uint32_t c = 0; c < V_PROJ; c++) {
        for (uint32_t w = 0; w < 4; w++) {
            v_conv[c * 4 + w] = 0.2f * pseudo(17, c * 4 + w) +
                                (w == 3 ? 0.9f : 0.0f);
        }
    }
    for (uint32_t h = 0; h < V_HEADS; h++) {
        a_log[h] = -0.6f + 0.25f * (float)h;
        dt_bias[h] = -0.3f + 0.15f * (float)h;
    }
    for (uint32_t d = 0; d < D; d++) norm[d] = 0.8f + 0.004f * (float)d;

    require_ok(ds4_gpu_init(), "GPU initialization");
    require_ok(ds4_gpu_set_model_map(model, MODEL_BYTES),
               "model map registration");

    /* Deterministic activations for every token. */
    static float qs[TOKENS * QK_PROJ], ks[TOKENS * QK_PROJ];
    static float vs[TOKENS * V_PROJ], zs[TOKENS * V_PROJ];
    static float alphas[TOKENS * V_HEADS], betas[TOKENS * V_HEADS];
    for (uint32_t i = 0; i < TOKENS * QK_PROJ; i++) {
        qs[i] = 1.6f * pseudo(23, i);
        ks[i] = 1.6f * pseudo(29, i);
    }
    for (uint32_t i = 0; i < TOKENS * V_PROJ; i++) {
        vs[i] = 1.6f * pseudo(31, i);
        zs[i] = 2.0f * pseudo(37, i);
    }
    for (uint32_t i = 0; i < TOKENS * V_HEADS; i++) {
        alphas[i] = 2.0f * pseudo(41, i);
        betas[i] = 2.0f * pseudo(43, i);
    }

    /* Host reference over the full sequence. */
    static host_gdn_state hs;
    memset(&hs, 0, sizeof(hs));
    static double expected[TOKENS * V_PROJ];
    for (uint32_t t = 0; t < TOKENS; t++) {
        host_gdn_step(&hs, q_conv, k_conv, v_conv, a_log, dt_bias, norm,
                      qs + t * QK_PROJ, ks + t * QK_PROJ, vs + t * V_PROJ,
                      alphas + t * V_HEADS, betas + t * V_HEADS,
                      zs + t * V_PROJ, expected + t * V_PROJ);
    }

    ds4_gpu_tensor *q = ds4_gpu_tensor_alloc(sizeof(qs));
    ds4_gpu_tensor *k = ds4_gpu_tensor_alloc(sizeof(ks));
    ds4_gpu_tensor *v = ds4_gpu_tensor_alloc(sizeof(vs));
    ds4_gpu_tensor *z = ds4_gpu_tensor_alloc(sizeof(zs));
    ds4_gpu_tensor *alpha = ds4_gpu_tensor_alloc(sizeof(alphas));
    ds4_gpu_tensor *beta = ds4_gpu_tensor_alloc(sizeof(betas));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(sizeof(vs));
    ds4_gpu_tensor *conv = ds4_gpu_tensor_alloc(CONV_ELEMENTS * sizeof(float));
    ds4_gpu_tensor *state = ds4_gpu_tensor_alloc(STATE_ELEMENTS * sizeof(float));
    require_ok(q && k && v && z && alpha && beta && out && conv && state,
               "tensor allocation");

    /* 1. Whole-sequence batch pass vs the host reference. */
    require_ok(ds4_gpu_tensor_write(q, 0, qs, sizeof(qs)), "Q write");
    require_ok(ds4_gpu_tensor_write(k, 0, ks, sizeof(ks)), "K write");
    require_ok(ds4_gpu_tensor_write(v, 0, vs, sizeof(vs)), "V write");
    require_ok(ds4_gpu_tensor_write(z, 0, zs, sizeof(zs)), "gate write");
    require_ok(ds4_gpu_tensor_write(alpha, 0, alphas, sizeof(alphas)), "alpha write");
    require_ok(ds4_gpu_tensor_write(beta, 0, betas, sizeof(betas)), "beta write");
    require_ok(ds4_gpu_tensor_fill_f32(conv, 0.0f, CONV_ELEMENTS), "conv clear");
    require_ok(ds4_gpu_tensor_fill_f32(state, 0.0f, STATE_ELEMENTS), "state clear");
    require_ok(ds4_gpu_qwen38_gdn(
        out, conv, state, q, k, v, alpha, beta, z,
        model, MODEL_BYTES, Q_CONV_OFFSET, K_CONV_OFFSET, V_CONV_OFFSET,
        A_LOG_OFFSET, DT_BIAS_OFFSET, NORM_OFFSET,
        QK_HEADS, V_HEADS, TOKENS, 1e-6f), "GDN batch pass");
    static float batch_actual[TOKENS * V_PROJ];
    require_ok(ds4_gpu_tensor_read(out, 0, batch_actual, sizeof(batch_actual)),
               "batch output read");
    for (uint32_t i = 0; i < TOKENS * V_PROJ; i++)
        require_close("GDN batch vs host", i, batch_actual[i], expected[i], 3e-4);

    /* 2. Token-by-token passes vs the host reference. */
    require_ok(ds4_gpu_tensor_fill_f32(conv, 0.0f, CONV_ELEMENTS), "conv reset");
    require_ok(ds4_gpu_tensor_fill_f32(state, 0.0f, STATE_ELEMENTS), "state reset");
    for (uint32_t t = 0; t < TOKENS; t++) {
        require_ok(ds4_gpu_tensor_write(q, 0, qs + t * QK_PROJ,
                                        QK_PROJ * sizeof(float)), "step Q write");
        require_ok(ds4_gpu_tensor_write(k, 0, ks + t * QK_PROJ,
                                        QK_PROJ * sizeof(float)), "step K write");
        require_ok(ds4_gpu_tensor_write(v, 0, vs + t * V_PROJ,
                                        V_PROJ * sizeof(float)), "step V write");
        require_ok(ds4_gpu_tensor_write(z, 0, zs + t * V_PROJ,
                                        V_PROJ * sizeof(float)), "step gate write");
        require_ok(ds4_gpu_tensor_write(alpha, 0, alphas + t * V_HEADS,
                                        V_HEADS * sizeof(float)), "step alpha write");
        require_ok(ds4_gpu_tensor_write(beta, 0, betas + t * V_HEADS,
                                        V_HEADS * sizeof(float)), "step beta write");
        require_ok(ds4_gpu_qwen38_gdn(
            out, conv, state, q, k, v, alpha, beta, z,
            model, MODEL_BYTES, Q_CONV_OFFSET, K_CONV_OFFSET, V_CONV_OFFSET,
            A_LOG_OFFSET, DT_BIAS_OFFSET, NORM_OFFSET,
            QK_HEADS, V_HEADS, 1, 1e-6f), "GDN single-token pass");
        static float step_actual[V_PROJ];
        require_ok(ds4_gpu_tensor_read(out, 0, step_actual, sizeof(step_actual)),
                   "step output read");
        for (uint32_t i = 0; i < V_PROJ; i++)
            require_close("GDN decode vs host", t * V_PROJ + i,
                          step_actual[i], expected[t * V_PROJ + i], 3e-4);
    }

    /* 3. Chunked continuation: 9 + 8 tokens must equal the batch pass. */
    enum { CHUNK = 9 };
    require_ok(ds4_gpu_tensor_write(q, 0, qs, sizeof(qs)), "chunk Q write");
    require_ok(ds4_gpu_tensor_write(k, 0, ks, sizeof(ks)), "chunk K write");
    require_ok(ds4_gpu_tensor_write(v, 0, vs, sizeof(vs)), "chunk V write");
    require_ok(ds4_gpu_tensor_write(z, 0, zs, sizeof(zs)), "chunk gate write");
    require_ok(ds4_gpu_tensor_write(alpha, 0, alphas, sizeof(alphas)), "chunk alpha write");
    require_ok(ds4_gpu_tensor_write(beta, 0, betas, sizeof(betas)), "chunk beta write");
    require_ok(ds4_gpu_tensor_fill_f32(conv, 0.0f, CONV_ELEMENTS), "chunk conv clear");
    require_ok(ds4_gpu_tensor_fill_f32(state, 0.0f, STATE_ELEMENTS), "chunk state clear");
    require_ok(ds4_gpu_qwen38_gdn(
        out, conv, state, q, k, v, alpha, beta, z,
        model, MODEL_BYTES, Q_CONV_OFFSET, K_CONV_OFFSET, V_CONV_OFFSET,
        A_LOG_OFFSET, DT_BIAS_OFFSET, NORM_OFFSET,
        QK_HEADS, V_HEADS, CHUNK, 1e-6f), "GDN chunk 1");
    /* Second chunk reads activations at the chunk offset via fresh writes. */
    require_ok(ds4_gpu_tensor_write(q, 0, qs + CHUNK * QK_PROJ,
                                    (TOKENS - CHUNK) * QK_PROJ * sizeof(float)),
               "chunk2 Q write");
    require_ok(ds4_gpu_tensor_write(k, 0, ks + CHUNK * QK_PROJ,
                                    (TOKENS - CHUNK) * QK_PROJ * sizeof(float)),
               "chunk2 K write");
    require_ok(ds4_gpu_tensor_write(v, 0, vs + CHUNK * V_PROJ,
                                    (TOKENS - CHUNK) * V_PROJ * sizeof(float)),
               "chunk2 V write");
    require_ok(ds4_gpu_tensor_write(z, 0, zs + CHUNK * V_PROJ,
                                    (TOKENS - CHUNK) * V_PROJ * sizeof(float)),
               "chunk2 gate write");
    require_ok(ds4_gpu_tensor_write(alpha, 0, alphas + CHUNK * V_HEADS,
                                    (TOKENS - CHUNK) * V_HEADS * sizeof(float)),
               "chunk2 alpha write");
    require_ok(ds4_gpu_tensor_write(beta, 0, betas + CHUNK * V_HEADS,
                                    (TOKENS - CHUNK) * V_HEADS * sizeof(float)),
               "chunk2 beta write");
    require_ok(ds4_gpu_qwen38_gdn(
        out, conv, state, q, k, v, alpha, beta, z,
        model, MODEL_BYTES, Q_CONV_OFFSET, K_CONV_OFFSET, V_CONV_OFFSET,
        A_LOG_OFFSET, DT_BIAS_OFFSET, NORM_OFFSET,
        QK_HEADS, V_HEADS, TOKENS - CHUNK, 1e-6f), "GDN chunk 2");
    static float chunk_actual[(TOKENS - CHUNK) * V_PROJ];
    require_ok(ds4_gpu_tensor_read(out, 0, chunk_actual, sizeof(chunk_actual)),
               "chunk output read");
    for (uint32_t i = 0; i < (TOKENS - CHUNK) * V_PROJ; i++)
        require_close("GDN chunked continuation", i, chunk_actual[i],
                      expected[CHUNK * V_PROJ + i], 3e-4);

    ds4_gpu_tensor_free(state);
    ds4_gpu_tensor_free(conv);
    ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(beta);
    ds4_gpu_tensor_free(alpha);
    ds4_gpu_tensor_free(z);
    ds4_gpu_tensor_free(v);
    ds4_gpu_tensor_free(k);
    ds4_gpu_tensor_free(q);
    ds4_gpu_cleanup();
    munmap(model, MODEL_BYTES);
    puts("Qwen3.8 GDN GPU tests: PASS");
    return 0;
}
