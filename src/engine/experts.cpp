#include "pulsar_engine_internal.h"



/* Dispatch: call the right gate/up mid builder based on tensor type. */
void matvec_experts_mid_prequant(
        float            *mid,
        const pulsar_model  *m,
        const pulsar_tensor *gate_w,
        const pulsar_tensor *up_w,
        const block_q8_K *xq,
        const int        *selected,
        const float      *expert_weight,
        int               n_expert,
        float             clamp) {
    if (gate_w->type == PULSAR_TENSOR_IQ2_XXS) {
        matvec_iq2_xxs_experts_mid_prequant(mid, m, gate_w, up_w, xq,
                                            selected, expert_weight, n_expert, clamp);
    } else {
        pulsar_die("unsupported gate/up expert tensor type");
    }
}



/* Dispatch: call the right down-projection accumulator based on tensor type. */
void matvec_experts_down_accum_prequant(
        float            *out,
        const pulsar_model  *m,
        const pulsar_tensor *w,
        const block_q8_K *xq,
        const int        *selected,
        int               n_expert) {
    if (w->type == PULSAR_TENSOR_Q2_K) {
        matvec_q2_k_experts_accum_prequant(out, m, w, xq, selected, n_expert);
    } else {
        pulsar_die("unsupported down expert tensor type");
    }
}



/* Dispatch: single-expert gate/up pair for tracing. */
void matvec_expert_pair_prequant(
        float            *out0,
        float            *out1,
        const pulsar_model  *m,
        const pulsar_tensor *w0,
        const pulsar_tensor *w1,
        const block_q8_K *xq,
        uint32_t          expert) {
    if (w0->type == PULSAR_TENSOR_IQ2_XXS) {
        matvec_iq2_xxs_expert_pair_prequant(out0, out1, m, w0, w1, xq, expert);
    } else {
        pulsar_die("unsupported gate/up expert tensor type");
    }
}



/* Dispatch: single-expert down projection for tracing. */
void matvec_expert_down(
        float            *out,
        const pulsar_model  *m,
        const pulsar_tensor *w,
        const float      *x,
        uint32_t          expert) {
    if (w->type == PULSAR_TENSOR_Q2_K) {
        matvec_q2_k_expert(out, m, w, x, expert);
    } else {
        pulsar_die("unsupported down expert tensor type");
    }
}



static PULSAR_MAYBE_UNUSED void sum_down_pairs_worker(void *vctx, uint64_t row0, uint64_t row1) {
    sum_down_pairs_ctx *ctx = static_cast<sum_down_pairs_ctx *>(vctx);
    for (uint64_t idx = row0; idx < row1; idx++) {
        const uint32_t token = (uint32_t)(idx / ctx->out_dim);
        const uint64_t row = idx - (uint64_t)token * ctx->out_dim;
        float acc = 0.0f;
        for (uint32_t slot = 0; slot < PULSAR_N_EXPERT_USED; slot++) {
            const uint64_t pair_id = (uint64_t)token * PULSAR_N_EXPERT_USED + slot;
            acc += ctx->down_pair[pair_id * ctx->out_dim + row];
        }
        ctx->moe[idx] = acc;
    }
}



/* =========================================================================
 * Hyper-Connection Transforms.
 * =========================================================================
 *
 * DeepSeek V4 Flash keeps four hyper-connection streams per token.  Before
 * attention or FFN, a learned small projection chooses how to reduce the HC
 * state into the 4096-wide sublayer input.  After the sublayer, the post and
 * combine weights expand the result back into the four-stream HC state.
 */

/* Decode the HC control projection.  The output contains pre weights, post
 * gates, and a small doubly-normalized combine matrix. */
void hc_split_sinkhorn_one(
        float       * out,
        const float * mix,
        const float * scale,
        const float * base,
        int           n_hc,
        int           iters,
        float         eps) {
    const float pre_scale  = scale[0];
    const float post_scale = scale[1];
    const float comb_scale = scale[2];

    for (int i = 0; i < n_hc; i++) {
        const float z = mix[i] * pre_scale + base[i];
        out[i] = 1.0f / (1.0f + expf(-z)) + eps;
    }

    for (int i = 0; i < n_hc; i++) {
        const int off = n_hc + i;
        const float z = mix[off] * post_scale + base[off];
        out[off] = 2.0f / (1.0f + expf(-z));
    }

    float c[16 * 16];

    for (int dst = 0; dst < n_hc; dst++) {
        float row_max = PULSAR_NEG_INF;
        for (int src = 0; src < n_hc; src++) {
            const int idx = src + dst * n_hc;
            const int off = 2 * n_hc + idx;
            const float v = mix[off] * comb_scale + base[off];
            c[idx] = v;
            if (v > row_max) row_max = v;
        }

        float row_sum = 0.0f;
        for (int src = 0; src < n_hc; src++) {
            const int idx = src + dst * n_hc;
            const float v = expf(c[idx] - row_max);
            c[idx] = v;
            row_sum += v;
        }

        const float inv = 1.0f / row_sum;
        for (int src = 0; src < n_hc; src++) {
            const int idx = src + dst * n_hc;
            c[idx] = c[idx] * inv + eps;
        }
    }

    for (int src = 0; src < n_hc; src++) {
        float sum = 0.0f;
        for (int dst = 0; dst < n_hc; dst++) sum += c[src + dst * n_hc];

        const float inv = 1.0f / (sum + eps);
        for (int dst = 0; dst < n_hc; dst++) c[src + dst * n_hc] *= inv;
    }

    for (int iter = 1; iter < iters; iter++) {
        for (int dst = 0; dst < n_hc; dst++) {
            float sum = 0.0f;
            for (int src = 0; src < n_hc; src++) sum += c[src + dst * n_hc];

            const float inv = 1.0f / (sum + eps);
            for (int src = 0; src < n_hc; src++) c[src + dst * n_hc] *= inv;
        }

        for (int src = 0; src < n_hc; src++) {
            float sum = 0.0f;
            for (int dst = 0; dst < n_hc; dst++) sum += c[src + dst * n_hc];

            const float inv = 1.0f / (sum + eps);
            for (int dst = 0; dst < n_hc; dst++) c[src + dst * n_hc] *= inv;
        }
    }

    for (int i = 0; i < n_hc * n_hc; i++) out[2 * n_hc + i] = c[i];
}

