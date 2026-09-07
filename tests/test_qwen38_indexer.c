/* Qwen 3.8 QSA indexer GPU path vs a scalar host reference.
 *
 * Host model of Qwen4ExpTextQSAIndexer: raw 128-wide keys cached per token
 * (rounded through f16 like the GPU cache), pooled 4 at a time as an FP32
 * mean, RMSNorm with the key norm, neox RoPE of the first 64 dims at the
 * block's first position; queries RMSNorm + RoPE at their own position;
 * score of a complete block = sum over the 4 heads of ReLU(q.k) / sqrt(128);
 * top-k blocks by score, plus every token of the incomplete tail block.
 *
 * Checks, over two prefill chunks whose boundary splits a block: the pooled
 * cache, the score matrix (including -inf for blocks not yet complete for a
 * row), the expanded selection sets, and the attention kernel's walk over the
 * selection list against host attention masked to the same positions.
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
    ID = 128,          /* indexer head dim */
    IHEADS = 4,
    ROPE = 64,
    POOL = 4,
    BUDGET = 16,       /* token budget under test: 4 blocks */
    BLOCK_TOPK = BUDGET / POOL,
    TOKENS = 41,       /* 10 complete blocks and a 1-token tail */
    CHUNK1 = 26,       /* the boundary falls inside block 6 */
    CACHE_CAP = 48,
    MASK_WORDS = (CACHE_CAP + 31) / 32,
    BLOCKS_CAP = CACHE_CAP / POOL,
    /* attention side, the QSA kernel's own geometry */
    AD = 256,
    Q_HEADS = 4,
    KV_HEADS = 2,
    RATIO = Q_HEADS / KV_HEADS,
    Q_NORM_OFFSET = 0,
    K_NORM_OFFSET = 1024,
    AQ_NORM_OFFSET = 2048,
    AK_NORM_OFFSET = 4096,
    MODEL_BYTES = 8192,
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

static double to_f16(double x) {
    return (double)(_Float16)x;
}

/* RMSNorm over `dim` then neox RoPE of the first ROPE dims at `pos`. */
static void host_norm_rope(const double *raw, const float *weight, uint32_t dim,
                           uint32_t pos, double *out) {
    double sumsq = 0.0;
    for (uint32_t d = 0; d < dim; d++) sumsq += raw[d] * raw[d];
    const double scale = 1.0 / sqrt(sumsq / (double)dim + (double)NORM_EPS);
    for (uint32_t d = 0; d < dim; d++) out[d] = raw[d] * scale * (double)weight[d];
    for (uint32_t i = 0; i < ROPE / 2; i++) {
        const double inv_freq = pow((double)FREQ_BASE, -(double)(2 * i) / (double)ROPE);
        const double angle = (double)pos * inv_freq;
        const double c = cos(angle), s = sin(angle);
        const double a = out[i], b = out[i + ROPE / 2];
        out[i] = a * c - b * s;
        out[i + ROPE / 2] = b * c + a * s;
    }
}

int main(void) {
    uint8_t *model = mmap(NULL, MODEL_BYTES, PROT_READ | PROT_WRITE,
                          MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (model == MAP_FAILED) {
        perror("mmap");
        return 1;
    }
    float *q_norm = (float *)(model + Q_NORM_OFFSET);
    float *k_norm = (float *)(model + K_NORM_OFFSET);
    float *aq_norm = (float *)(model + AQ_NORM_OFFSET);
    float *ak_norm = (float *)(model + AK_NORM_OFFSET);
    for (uint32_t d = 0; d < ID; d++) {
        q_norm[d] = 1.0f + 0.2f * pseudo(11, d);
        k_norm[d] = 1.0f + 0.2f * pseudo(12, d);
    }
    for (uint32_t d = 0; d < AD; d++) {
        aq_norm[d] = 1.0f + 0.2f * pseudo(13, d);
        ak_norm[d] = 1.0f + 0.2f * pseudo(14, d);
    }
    require_ok(ds4_gpu_set_model_map(model, MODEL_BYTES), "model map registration");

    /* Indexer inputs. */
    static float iq[TOKENS][IHEADS][ID];
    static float ik[TOKENS][ID];
    for (uint32_t t = 0; t < TOKENS; t++) {
        for (uint32_t h = 0; h < IHEADS; h++)
            for (uint32_t d = 0; d < ID; d++)
                iq[t][h][d] = 2.0f * pseudo(21 + h, t * ID + d);
        for (uint32_t d = 0; d < ID; d++) ik[t][d] = 2.0f * pseudo(31, t * ID + d);
    }
    /* Attention inputs. */
    static float aq[TOKENS][Q_HEADS][AD], ag[TOKENS][Q_HEADS][AD];
    static float ak[TOKENS][KV_HEADS][AD], av[TOKENS][KV_HEADS][AD];
    for (uint32_t t = 0; t < TOKENS; t++) {
        for (uint32_t h = 0; h < Q_HEADS; h++)
            for (uint32_t d = 0; d < AD; d++) {
                aq[t][h][d] = pseudo(41 + h, t * AD + d);
                ag[t][h][d] = 4.0f * pseudo(51 + h, t * AD + d);
            }
        for (uint32_t h = 0; h < KV_HEADS; h++)
            for (uint32_t d = 0; d < AD; d++) {
                ak[t][h][d] = pseudo(61 + h, t * AD + d);
                av[t][h][d] = pseudo(71 + h, t * AD + d);
            }
    }

    /* Host reference: pooled blocks and normalized queries. */
    static double host_pool[TOKENS / POOL][ID];
    for (uint32_t b = 0; b < TOKENS / POOL; b++) {
        double mean[ID];
        for (uint32_t d = 0; d < ID; d++) {
            double sum = 0.0;
            for (uint32_t r = 0; r < POOL; r++) sum += to_f16(ik[b * POOL + r][d]);
            mean[d] = sum / (double)POOL;
        }
        double roped[ID];
        host_norm_rope(mean, k_norm, ID, b * POOL, roped);
        for (uint32_t d = 0; d < ID; d++) host_pool[b][d] = to_f16(roped[d]);
    }
    static double host_q[TOKENS][IHEADS][ID];
    for (uint32_t t = 0; t < TOKENS; t++)
        for (uint32_t h = 0; h < IHEADS; h++) {
            double raw[ID];
            for (uint32_t d = 0; d < ID; d++) raw[d] = iq[t][h][d];
            host_norm_rope(raw, q_norm, ID, t, host_q[t][h]);
        }
    /* Scores and the reference selection per row. */
    static double host_score[TOKENS][TOKENS / POOL];
    static bool host_sel[TOKENS][TOKENS];
    static bool host_sel_sure[TOKENS];
    for (uint32_t t = 0; t < TOKENS; t++) {
        const uint32_t visible = (t + 1) / POOL;
        for (uint32_t b = 0; b < visible; b++) {
            double score = 0.0;
            for (uint32_t h = 0; h < IHEADS; h++) {
                double dot = 0.0;
                for (uint32_t d = 0; d < ID; d++) dot += host_q[t][h][d] * host_pool[b][d];
                if (dot > 0.0) score += dot;
            }
            host_score[t][b] = score / sqrt((double)ID);
        }
        /* top-k blocks: mark tokens; record whether the k-th/(k+1)-th margin
         * is wide enough for the GPU's f32 ordering to be pinned down. */
        const uint32_t k = visible < BLOCK_TOPK ? visible : BLOCK_TOPK;
        bool taken[TOKENS / POOL] = {false};
        double kth = 1e300;
        for (uint32_t i = 0; i < k; i++) {
            int best = -1;
            for (uint32_t b = 0; b < visible; b++)
                if (!taken[b] && (best < 0 || host_score[t][b] > host_score[t][best])) best = (int)b;
            taken[best] = true;
            if (host_score[t][best] < kth) kth = host_score[t][best];
            for (uint32_t r = 0; r < POOL; r++) host_sel[t][best * POOL + r] = true;
        }
        double next = -1e300;
        for (uint32_t b = 0; b < visible; b++)
            if (!taken[b] && host_score[t][b] > next) next = host_score[t][b];
        host_sel_sure[t] = k == visible || kth - next > 1e-3;
        for (uint32_t p = visible * POOL; p <= t; p++) host_sel[t][p] = true;
    }

    /* GPU buffers. */
    ds4_gpu_tensor *raw_cache = ds4_gpu_tensor_alloc((uint64_t)CACHE_CAP * ID * 2);
    ds4_gpu_tensor *pool_cache = ds4_gpu_tensor_alloc((uint64_t)BLOCKS_CAP * ID * 2);
    ds4_gpu_tensor *gq = ds4_gpu_tensor_alloc(sizeof(iq));
    ds4_gpu_tensor *gk = ds4_gpu_tensor_alloc(sizeof(ik));
    ds4_gpu_tensor *ones = ds4_gpu_tensor_alloc((uint64_t)TOKENS * IHEADS * sizeof(float));
    ds4_gpu_tensor *scores = ds4_gpu_tensor_alloc((uint64_t)TOKENS * BLOCKS_CAP * sizeof(float));
    ds4_gpu_tensor *pool_sel = ds4_gpu_tensor_alloc((uint64_t)TOKENS * BLOCK_TOPK * sizeof(uint32_t));
    ds4_gpu_tensor *sel = ds4_gpu_tensor_alloc((uint64_t)TOKENS * MASK_WORDS * sizeof(uint32_t));
    ds4_gpu_tensor *q = ds4_gpu_tensor_alloc(sizeof(aq));
    ds4_gpu_tensor *og = ds4_gpu_tensor_alloc(sizeof(ag));
    ds4_gpu_tensor *k = ds4_gpu_tensor_alloc(sizeof(ak));
    ds4_gpu_tensor *v = ds4_gpu_tensor_alloc(sizeof(av));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(sizeof(aq));
    ds4_gpu_tensor *k_cache = ds4_gpu_tensor_alloc((uint64_t)CACHE_CAP * KV_HEADS * AD * 2);
    ds4_gpu_tensor *v_cache = ds4_gpu_tensor_alloc((uint64_t)CACHE_CAP * KV_HEADS * AD * 2);
    require_ok(raw_cache && pool_cache && gq && gk && ones && scores && pool_sel &&
               sel && q && og && k && v && out && k_cache && v_cache, "allocation");
    require_ok(ds4_gpu_tensor_fill_f32(ones, 1.0f, (uint64_t)TOKENS * IHEADS), "ones");

    static uint32_t gpu_sel[TOKENS][MASK_WORDS];
    static float gpu_out[TOKENS][Q_HEADS][AD];
    const uint32_t chunk_start[2] = {0, CHUNK1};
    const uint32_t chunk_rows[2] = {CHUNK1, TOKENS - CHUNK1};
    for (uint32_t c = 0; c < 2; c++) {
        const uint32_t pos0 = chunk_start[c], rows = chunk_rows[c];
        require_ok(ds4_gpu_tensor_write(gq, 0, iq[pos0], (uint64_t)rows * IHEADS * ID * sizeof(float)), "iq write");
        require_ok(ds4_gpu_tensor_write(gk, 0, ik[pos0], (uint64_t)rows * ID * sizeof(float)), "ik write");
        require_ok(ds4_gpu_tensor_write(q, 0, aq[pos0], (uint64_t)rows * Q_HEADS * AD * sizeof(float)), "q write");
        require_ok(ds4_gpu_tensor_write(og, 0, ag[pos0], (uint64_t)rows * Q_HEADS * AD * sizeof(float)), "gate write");
        require_ok(ds4_gpu_tensor_write(k, 0, ak[pos0], (uint64_t)rows * KV_HEADS * AD * sizeof(float)), "k write");
        require_ok(ds4_gpu_tensor_write(v, 0, av[pos0], (uint64_t)rows * KV_HEADS * AD * sizeof(float)), "v write");

        require_ok(ds4_gpu_qwen38_indexer_keys(raw_cache, pool_cache, gq, gk, model, MODEL_BYTES,
                       Q_NORM_OFFSET, K_NORM_OFFSET, IHEADS, ID, ROPE, rows, pos0, CACHE_CAP, POOL,
                       FREQ_BASE, NORM_EPS), "indexer keys");
        const uint32_t blocks = (pos0 + rows) / POOL;
        const uint32_t top_k = blocks < BLOCK_TOPK ? blocks : BLOCK_TOPK;
        require_ok(ds4_gpu_glm53_indexer_scores_batch_tensor(scores, gq, ones, pool_cache, blocks, rows, pos0,
                       POOL, IHEADS, ID, 1.0f / sqrtf((float)ID), true), "indexer scores");
        require_ok(ds4_gpu_indexer_topk_tensor(pool_sel, scores, blocks, rows, top_k), "indexer top-k");
        require_ok(ds4_gpu_tensor_fill_f32(sel, 0.0f, (uint64_t)rows * MASK_WORDS), "mask clear");
        require_ok(ds4_gpu_qwen38_indexer_expand(sel, pool_sel, rows, pos0, top_k, BUDGET, POOL,
                       MASK_WORDS), "selection expansion");
        require_ok(ds4_gpu_qwen38_qsa(out, k_cache, v_cache, q, og, k, v, model, MODEL_BYTES,
                       AQ_NORM_OFFSET, AK_NORM_OFFSET, Q_HEADS, KV_HEADS, AD, ROPE, rows, pos0, CACHE_CAP,
                       FREQ_BASE, NORM_EPS, sel, MASK_WORDS), "sparse QSA pass");
        require_ok(ds4_gpu_synchronize(), "synchronize");

        /* Pooled cache: every block complete so far. */
        static uint16_t pool_bits[BLOCKS_CAP][ID];
        require_ok(ds4_gpu_tensor_read(pool_cache, 0, pool_bits, (uint64_t)blocks * ID * 2), "pool read");
        for (uint32_t b = 0; b < blocks; b++)
            for (uint32_t d = 0; d < ID; d++) {
                _Float16 h; memcpy(&h, &pool_bits[b][d], 2);
                require_close("pooled key", b * ID + d, (float)h, host_pool[b][d], 3e-3);
            }
        /* Scores: visible blocks match, blocks not yet complete for the row are -inf. */
        static float gpu_scores[TOKENS][BLOCKS_CAP];
        require_ok(ds4_gpu_tensor_read(scores, 0, gpu_scores, (uint64_t)rows * blocks * sizeof(float)), "scores read");
        for (uint32_t r = 0; r < rows; r++) {
            const uint32_t t = pos0 + r, visible = (t + 1) / POOL;
            const float *row = &gpu_scores[0][0] + (uint64_t)r * blocks;
            for (uint32_t b = 0; b < blocks; b++) {
                if (b < visible) require_close("indexer score", t * blocks + b, row[b], host_score[t][b], 1e-2);
                else require_ok(row[b] < -1e30f, "score of an incomplete block is -inf");
            }
        }
        /* Selection sets. */
        require_ok(ds4_gpu_tensor_read(sel, 0, gpu_sel, (uint64_t)rows * MASK_WORDS * sizeof(uint32_t)), "selection read");
        for (uint32_t r = 0; r < rows; r++) {
            const uint32_t t = pos0 + r;
            bool got[TOKENS] = {false};
            for (uint32_t p = 0; p < TOKENS; p++) {
                got[p] = (gpu_sel[r][p / 32] >> (p % 32)) & 1u;
                require_ok(!got[p] || p <= t, "selected positions are causal");
            }
            if (!host_sel_sure[t]) continue; /* a near-tie: ordering is not pinned down */
            for (uint32_t p = 0; p <= t; p++) {
                if (got[p] != host_sel[t][p]) {
                    fprintf(stderr, "selection mismatch row %u pos %u: gpu %d host %d\n", t, p, got[p], host_sel[t][p]);
                    return 1;
                }
            }
        }
        /* Attention over the selection list vs host attention on the same set. */
        require_ok(ds4_gpu_tensor_read(out, 0, gpu_out, (uint64_t)rows * Q_HEADS * AD * sizeof(float)), "out read");
        static double kc[TOKENS][KV_HEADS][AD], vc[TOKENS][KV_HEADS][AD];
        for (uint32_t r = 0; r < rows; r++) {
            const uint32_t t = pos0 + r;
            for (uint32_t h = 0; h < KV_HEADS; h++) {
                double raw[AD], roped[AD];
                for (uint32_t d = 0; d < AD; d++) raw[d] = ak[t][h][d];
                host_norm_rope(raw, ak_norm, AD, t, roped);
                for (uint32_t d = 0; d < AD; d++) {
                    kc[t][h][d] = to_f16(roped[d]);
                    vc[t][h][d] = to_f16(av[t][h][d]);
                }
            }
        }
        for (uint32_t r = 0; r < rows; r++) {
            const uint32_t t = pos0 + r;
            bool got[TOKENS] = {false};
            for (uint32_t p = 0; p <= t; p++) got[p] = (gpu_sel[r][p / 32] >> (p % 32)) & 1u;
            for (uint32_t h = 0; h < Q_HEADS; h++) {
                const uint32_t kvh = h / RATIO;
                double raw[AD], hq[AD];
                for (uint32_t d = 0; d < AD; d++) raw[d] = aq[t][h][d];
                host_norm_rope(raw, aq_norm, AD, t, hq);
                double w[TOKENS], best = -1e300, den = 0.0;
                for (uint32_t p = 0; p <= t; p++) {
                    if (!got[p]) { w[p] = 0.0; continue; }
                    double dot = 0.0;
                    for (uint32_t d = 0; d < AD; d++) dot += hq[d] * kc[p][kvh][d];
                    w[p] = dot / 16.0;
                    if (w[p] > best) best = w[p];
                }
                for (uint32_t p = 0; p <= t; p++) {
                    w[p] = got[p] ? exp(w[p] - best) : 0.0;
                    den += w[p];
                }
                for (uint32_t d = 0; d < AD; d++) {
                    double num = 0.0;
                    for (uint32_t p = 0; p <= t; p++) num += w[p] * vc[p][kvh][d];
                    const double gate = 1.0 / (1.0 + exp(-(double)ag[t][h][d]));
                    require_close("sparse attention", (t * Q_HEADS + h) * AD + d, gpu_out[r][h][d], (num / den) * gate, 2e-3);
                }
            }
        }
        printf("  chunk %u (pos0=%u rows=%u): pooled cache, scores, selections, attention: ok\n", c + 1, pos0, rows);
    }
    /* Under budget every complete block is selected: with a budget covering
     * the whole sequence the selection must be the full causal prefix of
     * every row, which is what makes the dense walk exact below the budget. */
    {
        enum { BIG_BUDGET = 64 };
        ds4_gpu_tensor *big_pool_sel = ds4_gpu_tensor_alloc((uint64_t)TOKENS * (BIG_BUDGET / POOL) * sizeof(uint32_t));
        ds4_gpu_tensor *big_sel = ds4_gpu_tensor_alloc((uint64_t)TOKENS * MASK_WORDS * sizeof(uint32_t));
        require_ok(big_pool_sel && big_sel, "big selection allocation");
        require_ok(ds4_gpu_tensor_write(gq, 0, iq, sizeof(iq)), "iq write (all)");
        require_ok(ds4_gpu_tensor_write(gk, 0, ik, sizeof(ik)), "ik write (all)");
        require_ok(ds4_gpu_qwen38_indexer_keys(raw_cache, pool_cache, gq, gk, model, MODEL_BYTES,
                       Q_NORM_OFFSET, K_NORM_OFFSET, IHEADS, ID, ROPE, TOKENS, 0, CACHE_CAP, POOL,
                       FREQ_BASE, NORM_EPS), "indexer keys (all)");
        const uint32_t blocks = TOKENS / POOL;
        require_ok(ds4_gpu_glm53_indexer_scores_batch_tensor(scores, gq, ones, pool_cache, blocks, TOKENS, 0,
                       POOL, IHEADS, ID, 1.0f / sqrtf((float)ID), true), "indexer scores (all)");
        require_ok(ds4_gpu_indexer_topk_tensor(big_pool_sel, scores, blocks, TOKENS, blocks), "top-k (all blocks)");
        require_ok(ds4_gpu_tensor_fill_f32(big_sel, 0.0f, (uint64_t)TOKENS * MASK_WORDS), "mask clear (all)");
        require_ok(ds4_gpu_qwen38_indexer_expand(big_sel, big_pool_sel, TOKENS, 0, blocks, BIG_BUDGET, POOL,
                       MASK_WORDS), "expansion (all)");
        require_ok(ds4_gpu_synchronize(), "synchronize (all)");
        static uint32_t all_sel[TOKENS][MASK_WORDS];
        require_ok(ds4_gpu_tensor_read(big_sel, 0, all_sel, sizeof(all_sel)), "selection read (all)");
        for (uint32_t t = 0; t < TOKENS; t++) {
            for (uint32_t p = 0; p < TOKENS; p++) {
                const bool got = (all_sel[t][p / 32] >> (p % 32)) & 1u;
                require_ok(!got || p <= t, "under budget: selected positions are causal");
                if (p <= t && !got) {
                    fprintf(stderr, "under budget: row %u misses position %u\n", t, p);
                    return 1;
                }
            }
        }
        printf("  under budget: every row selects its whole prefix: ok\n");
        ds4_gpu_tensor_free(big_pool_sel);
        ds4_gpu_tensor_free(big_sel);
    }
    printf("Qwen3.8 indexer GPU tests: PASS\n");
    return 0;
}
