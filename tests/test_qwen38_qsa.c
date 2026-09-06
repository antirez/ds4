/* Qwen 3.8 QSA dense-attention GPU kernels vs a scalar host reference.
 *
 * Host model: per-head-dim RMSNorm on q and k (weights already carry the
 * baked +1), 64-dim neox-style RoPE prefix, f16 KV cache append (the host
 * rounds through _Float16 to match), causal GQA attention at scale
 * 1/sqrt(256), and the fused per-head sigmoid output gate taken from the
 * raw second half of the q projection.
 *
 * Checks: batch pass vs host, token-by-token passes vs host, and a
 * chunked continuation over a shared cache.
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
    D = 256,
    ROPE = 64,
    Q_HEADS = 4,
    KV_HEADS = 2,
    RATIO = Q_HEADS / KV_HEADS,
    TOKENS = 13,
    CACHE_CAP = 32,
    Q_ROW = Q_HEADS * D,
    KV_ROW = KV_HEADS * D,
    OUT_ROW = Q_HEADS * D,
    Q_NORM_OFFSET = 0,
    K_NORM_OFFSET = 4096,
    MODEL_BYTES = 16384,
};

static const float FREQ_BASE = 10000000.0f;
static const float NORM_EPS = 1e-6f;

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

static void host_norm_rope(const float *raw, const float *weight,
                           uint32_t pos, double *out) {
    double sumsq = 0.0;
    for (uint32_t d = 0; d < D; d++) sumsq += (double)raw[d] * (double)raw[d];
    const double scale = 1.0 / sqrt(sumsq / (double)D + (double)NORM_EPS);
    double normed[D];
    for (uint32_t d = 0; d < D; d++)
        normed[d] = (double)raw[d] * scale * (double)weight[d];
    for (uint32_t d = 0; d < D; d++) out[d] = normed[d];
    for (uint32_t i = 0; i < ROPE / 2; i++) {
        const double inv_freq = pow((double)FREQ_BASE,
                                    -(double)(2 * i) / (double)ROPE);
        const double angle = (double)pos * inv_freq;
        const double c = cos(angle);
        const double s = sin(angle);
        const double a = normed[i];
        const double b = normed[i + ROPE / 2];
        out[i] = a * c - b * s;
        out[i + ROPE / 2] = b * c + a * s;
    }
}

static double to_f16(double x) {
    return (double)(_Float16)x;
}

int main(void) {
    uint8_t *model = mmap(NULL, MODEL_BYTES, PROT_READ | PROT_WRITE,
                          MAP_PRIVATE | MAP_ANON, -1, 0);
    if (model == MAP_FAILED) {
        perror("mmap");
        return 1;
    }
    float *q_norm = (float *)(model + Q_NORM_OFFSET);
    float *k_norm = (float *)(model + K_NORM_OFFSET);
    for (uint32_t d = 0; d < D; d++) {
        q_norm[d] = 0.9f + 0.002f * (float)d / 2.56f;
        k_norm[d] = 1.1f - 0.001f * (float)d / 2.56f;
    }

    require_ok(ds4_gpu_init(), "GPU initialization");
    require_ok(ds4_gpu_set_model_map(model, MODEL_BYTES),
               "model map registration");

    static float qs[TOKENS * Q_ROW], gs[TOKENS * Q_ROW];
    static float ks[TOKENS * KV_ROW], vs[TOKENS * KV_ROW];
    for (uint32_t i = 0; i < TOKENS * Q_ROW; i++) {
        qs[i] = 1.4f * pseudo(51, i);
        gs[i] = 1.4f * pseudo(59, i);
    }
    for (uint32_t i = 0; i < TOKENS * KV_ROW; i++) {
        ks[i] = 1.4f * pseudo(53, i);
        vs[i] = 1.4f * pseudo(57, i);
    }

    /* Host reference. */
    static double host_kcache[TOKENS][KV_HEADS][D];
    static double host_vcache[TOKENS][KV_HEADS][D];
    static double expected[TOKENS * OUT_ROW];
    for (uint32_t t = 0; t < TOKENS; t++) {
        for (uint32_t h = 0; h < KV_HEADS; h++) {
            double roped[D];
            host_norm_rope(ks + (t * KV_HEADS + h) * D, k_norm, t, roped);
            for (uint32_t d = 0; d < D; d++) {
                host_kcache[t][h][d] = to_f16(roped[d]);
                host_vcache[t][h][d] =
                    to_f16((double)vs[(t * KV_HEADS + h) * D + d]);
            }
        }
        for (uint32_t h = 0; h < Q_HEADS; h++) {
            const float *hq = qs + (t * Q_HEADS + h) * D;
            const float *hg = gs + (t * Q_HEADS + h) * D;
            double roped_q[D];
            host_norm_rope(hq, q_norm, t, roped_q);
            const uint32_t kvh = h / RATIO;
            double scores[TOKENS];
            double best = -INFINITY;
            for (uint32_t p = 0; p <= t; p++) {
                double dot = 0.0;
                for (uint32_t d = 0; d < D; d++)
                    dot += roped_q[d] * host_kcache[p][kvh][d];
                scores[p] = dot / 16.0;
                if (scores[p] > best) best = scores[p];
            }
            double den = 0.0;
            for (uint32_t p = 0; p <= t; p++) {
                scores[p] = exp(scores[p] - best);
                den += scores[p];
            }
            for (uint32_t d = 0; d < D; d++) {
                double num = 0.0;
                for (uint32_t p = 0; p <= t; p++)
                    num += scores[p] * host_vcache[p][kvh][d];
                const double gate = (double)hg[d];
                expected[(t * Q_HEADS + h) * D + d] =
                    (num / den) / (1.0 + exp(-gate));
            }
        }
    }

    ds4_gpu_tensor *q = ds4_gpu_tensor_alloc(sizeof(qs));
    ds4_gpu_tensor *og = ds4_gpu_tensor_alloc(sizeof(gs));
    ds4_gpu_tensor *k = ds4_gpu_tensor_alloc(sizeof(ks));
    ds4_gpu_tensor *v = ds4_gpu_tensor_alloc(sizeof(vs));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(
        (uint64_t)TOKENS * OUT_ROW * sizeof(float));
    ds4_gpu_tensor *k_cache = ds4_gpu_tensor_alloc(
        (uint64_t)CACHE_CAP * KV_ROW * sizeof(uint16_t));
    ds4_gpu_tensor *v_cache = ds4_gpu_tensor_alloc(
        (uint64_t)CACHE_CAP * KV_ROW * sizeof(uint16_t));
    require_ok(q && og && k && v && out && k_cache && v_cache,
               "tensor allocation");

    /* 1. Whole-sequence batch pass. */
    require_ok(ds4_gpu_tensor_write(q, 0, qs, sizeof(qs)), "Q write");
    require_ok(ds4_gpu_tensor_write(og, 0, gs, sizeof(gs)), "gate write");
    require_ok(ds4_gpu_tensor_write(k, 0, ks, sizeof(ks)), "K write");
    require_ok(ds4_gpu_tensor_write(v, 0, vs, sizeof(vs)), "V write");
    require_ok(ds4_gpu_qwen38_qsa(
        out, k_cache, v_cache, q, og, k, v,
        model, MODEL_BYTES, Q_NORM_OFFSET, K_NORM_OFFSET,
        Q_HEADS, KV_HEADS, D, ROPE, TOKENS, 0, CACHE_CAP,
        FREQ_BASE, NORM_EPS), "QSA batch pass");
    static float batch_actual[TOKENS * OUT_ROW];
    require_ok(ds4_gpu_tensor_read(out, 0, batch_actual, sizeof(batch_actual)),
               "batch output read");
    for (uint32_t i = 0; i < TOKENS * OUT_ROW; i++)
        require_close("QSA batch vs host", i, batch_actual[i], expected[i], 2e-3);

    /* 2. Token-by-token passes on a fresh cache. */
    require_ok(ds4_gpu_tensor_fill_f32(k_cache, 0.0f,
        (uint64_t)CACHE_CAP * KV_ROW / 2u), "k cache clear");
    require_ok(ds4_gpu_tensor_fill_f32(v_cache, 0.0f,
        (uint64_t)CACHE_CAP * KV_ROW / 2u), "v cache clear");
    for (uint32_t t = 0; t < TOKENS; t++) {
        require_ok(ds4_gpu_tensor_write(q, 0, qs + t * Q_ROW,
                                        Q_ROW * sizeof(float)), "step Q write");
        require_ok(ds4_gpu_tensor_write(og, 0, gs + t * Q_ROW,
                                        Q_ROW * sizeof(float)),
                   "step gate write");
        require_ok(ds4_gpu_tensor_write(k, 0, ks + t * KV_ROW,
                                        KV_ROW * sizeof(float)), "step K write");
        require_ok(ds4_gpu_tensor_write(v, 0, vs + t * KV_ROW,
                                        KV_ROW * sizeof(float)), "step V write");
        require_ok(ds4_gpu_qwen38_qsa(
            out, k_cache, v_cache, q, og, k, v,
            model, MODEL_BYTES, Q_NORM_OFFSET, K_NORM_OFFSET,
            Q_HEADS, KV_HEADS, D, ROPE, 1, t, CACHE_CAP,
            FREQ_BASE, NORM_EPS), "QSA single-token pass");
        static float step_actual[OUT_ROW];
        require_ok(ds4_gpu_tensor_read(out, 0, step_actual, sizeof(step_actual)),
                   "step output read");
        for (uint32_t i = 0; i < OUT_ROW; i++)
            require_close("QSA decode vs host", t * OUT_ROW + i,
                          step_actual[i], expected[t * OUT_ROW + i], 2e-3);
    }

    /* 3. Chunked continuation: 8 + 5 tokens over one cache. */
    enum { CHUNK = 8 };
    require_ok(ds4_gpu_tensor_fill_f32(k_cache, 0.0f,
        (uint64_t)CACHE_CAP * KV_ROW / 2u), "chunk k cache clear");
    require_ok(ds4_gpu_tensor_fill_f32(v_cache, 0.0f,
        (uint64_t)CACHE_CAP * KV_ROW / 2u), "chunk v cache clear");
    require_ok(ds4_gpu_tensor_write(q, 0, qs, sizeof(qs)), "chunk Q write");
    require_ok(ds4_gpu_tensor_write(og, 0, gs, sizeof(gs)), "chunk gate write");
    require_ok(ds4_gpu_tensor_write(k, 0, ks, sizeof(ks)), "chunk K write");
    require_ok(ds4_gpu_tensor_write(v, 0, vs, sizeof(vs)), "chunk V write");
    require_ok(ds4_gpu_qwen38_qsa(
        out, k_cache, v_cache, q, og, k, v,
        model, MODEL_BYTES, Q_NORM_OFFSET, K_NORM_OFFSET,
        Q_HEADS, KV_HEADS, D, ROPE, CHUNK, 0, CACHE_CAP,
        FREQ_BASE, NORM_EPS), "QSA chunk 1");
    require_ok(ds4_gpu_tensor_write(q, 0, qs + CHUNK * Q_ROW,
        (TOKENS - CHUNK) * Q_ROW * sizeof(float)), "chunk2 Q write");
    require_ok(ds4_gpu_tensor_write(og, 0, gs + CHUNK * Q_ROW,
        (TOKENS - CHUNK) * Q_ROW * sizeof(float)), "chunk2 gate write");
    require_ok(ds4_gpu_tensor_write(k, 0, ks + CHUNK * KV_ROW,
        (TOKENS - CHUNK) * KV_ROW * sizeof(float)), "chunk2 K write");
    require_ok(ds4_gpu_tensor_write(v, 0, vs + CHUNK * KV_ROW,
        (TOKENS - CHUNK) * KV_ROW * sizeof(float)), "chunk2 V write");
    require_ok(ds4_gpu_qwen38_qsa(
        out, k_cache, v_cache, q, og, k, v,
        model, MODEL_BYTES, Q_NORM_OFFSET, K_NORM_OFFSET,
        Q_HEADS, KV_HEADS, D, ROPE, TOKENS - CHUNK, CHUNK, CACHE_CAP,
        FREQ_BASE, NORM_EPS), "QSA chunk 2");
    static float chunk_actual[(TOKENS - CHUNK) * OUT_ROW];
    require_ok(ds4_gpu_tensor_read(out, 0, chunk_actual, sizeof(chunk_actual)),
               "chunk output read");
    for (uint32_t i = 0; i < (TOKENS - CHUNK) * OUT_ROW; i++)
        require_close("QSA chunked continuation", i, chunk_actual[i],
                      expected[CHUNK * OUT_ROW + i], 2e-3);

    ds4_gpu_tensor_free(v_cache);
    ds4_gpu_tensor_free(k_cache);
    ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(v);
    ds4_gpu_tensor_free(k);
    ds4_gpu_tensor_free(og);
    ds4_gpu_tensor_free(q);
    ds4_gpu_cleanup();
    munmap(model, MODEL_BYTES);
    puts("Qwen3.8 QSA GPU tests: PASS");
    return 0;
}
