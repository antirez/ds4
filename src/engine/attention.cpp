#include "ds4_engine_internal.h"



void layer_q_projection_with_lora_one(
        const ds4_model   * model,
        const ds4_layer_weights * layer,
        const float       * norm,
        float             * q,
        float             * qr_norm) {
    const uint32_t q_rank = DS4_N_LORA_Q;
    float *qr = (float *)xmalloc((size_t)q_rank * sizeof(qr[0]));
    const float *q_a_norm = (const float *)tensor_data(model, layer->attn_q_a_norm);

    matvec_q8_0(qr, model, layer->attn_q_a, norm);
    rms_norm_weight(qr_norm, qr, q_a_norm, q_rank, DS4_RMS_EPS);
    matvec_q8_0(q, model, layer->attn_q_b, qr_norm);
    head_rms_norm_inplace(q, DS4_N_HEAD, DS4_N_HEAD_DIM, DS4_RMS_EPS);

    free(qr);
}



/* KV projection has one KV head of width 512, followed by a learned RMSNorm. */
void layer_kv_projection_normed_one(
        const ds4_model   * model,
        const ds4_layer_weights * layer,
        const float       * normed,
        float             * kv) {
    float *raw = (float *)xmalloc((size_t)DS4_N_HEAD_DIM * sizeof(raw[0]));

    const float *kv_norm = (const float *)tensor_data(model, layer->attn_kv_a_norm);

    matvec_q8_0(raw, model, layer->attn_kv, normed);
    rms_norm_weight(kv, raw, kv_norm, DS4_N_HEAD_DIM, DS4_RMS_EPS);

    free(raw);
}









static float rope_yarn_ramp(float low, float high, int i0) {
    const float y = ((float)(i0 / 2) - low) / fmaxf(0.001f, high - low);
    return 1.0f - fminf(1.0f, fmaxf(0.0f, y));
}



static float rope_yarn_corr_dim(int n_dims, uint64_t n_ctx_orig, float n_rot, float base) {
    return (float)n_dims * logf((float)n_ctx_orig / (n_rot * 2.0f * (float)M_PI)) / (2.0f * logf(base));
}



static void rope_yarn_corr_dims(int n_dims, uint64_t n_ctx_orig, float freq_base, float beta_fast, float beta_slow, float dims[2]) {
    const float start = floorf(rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_fast, freq_base));
    const float end = ceilf(rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_slow, freq_base));
    dims[0] = fmaxf(0.0f, start);
    dims[1] = fminf((float)(n_dims - 1), end);
}



/* Apply DS4 RoPE only to the tail of each head.  Compressed layers use the
 * long-context frequency base and scale; inverse mode rotates attention output
 * back before the grouped output projection. */
static void rope_tail_ext_inplace(
        float    * x,
        uint32_t   n_head,
        uint32_t   head_dim,
        uint32_t   n_rot,
        uint32_t   pos,
        uint64_t   n_ctx_orig,
        float      freq_base,
        float      freq_scale,
        float      ext_factor,
        float      attn_factor,
        float      beta_fast,
        float      beta_slow,
        bool       inverse) {
    const uint32_t n_nope = head_dim - n_rot;
    const float theta_scale = powf(freq_base, -2.0f / (float)n_rot);
    const float sin_sign = inverse ? -1.0f : 1.0f;
    float corr_dims[2] = { 0.0f, 0.0f };
    if (ext_factor != 0.0f) {
        rope_yarn_corr_dims((int)n_rot, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);
    }

    for (uint32_t h = 0; h < n_head; h++) {
        float *tail = x + (uint64_t)h * head_dim + n_nope;
        float theta_extrap = (float)pos;

        for (uint32_t i = 0; i < n_rot; i += 2) {
            const float theta_interp = freq_scale * theta_extrap;
            float theta = theta_interp;
            float mscale = attn_factor;

            if (ext_factor != 0.0f) {
                const float ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], (int)i) * ext_factor;
                theta = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
                mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
            }

            const float c = cosf(theta) * mscale;
            const float s = sin_sign * sinf(theta) * mscale;
            const float x0 = tail[i + 0];
            const float x1 = tail[i + 1];

            tail[i + 0] = x0 * c - x1 * s;
            tail[i + 1] = x0 * s + x1 * c;

            theta_extrap *= theta_scale;
        }
    }
}



/* Dense layers and compressed layers use different RoPE bases. */
float layer_rope_freq_base(uint32_t il) {
    return ds4_layer_compress_ratio(il) != 0 && DS4_COMPRESS_ROPE_FREQ_BASE > 0.0f
        ? DS4_COMPRESS_ROPE_FREQ_BASE
        : DS4_ROPE_FREQ_BASE;
}



float layer_rope_freq_scale(uint32_t il) {
    if (ds4_layer_compress_ratio(il) == 0 || DS4_ROPE_SCALE_FACTOR <= 0.0f) {
        return 1.0f;
    }
    return 1.0f / DS4_ROPE_SCALE_FACTOR;
}



void rope_tail_layer_inplace(
        float            * x,
        uint32_t           n_head,
        uint32_t           head_dim,
        uint32_t           n_rot,
        uint32_t           pos,
        uint32_t           il,
        bool               inverse) {
    const bool compressed = ds4_layer_compress_ratio(il) != 0;
    const float freq_base = layer_rope_freq_base(il);
    const float freq_scale = layer_rope_freq_scale(il);
    const float ext_factor = compressed && DS4_ROPE_SCALE_FACTOR > 1.0f ? 1.0f : 0.0f;
    float attn_factor = 1.0f;
    if (ext_factor != 0.0f && freq_scale > 0.0f) {
        /*
         * This YaRN helper applies magnitude scaling internally. DeepSeek V4
         * reference RoPE uses interpolation without that magnitude change, so
         * pass the inverse factor here and let the helper cancel itself out.
         */
        attn_factor /= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }

    rope_tail_ext_inplace(x, n_head, head_dim, n_rot, pos,
                          compressed ? DS4_ROPE_ORIG_CTX : 0,
                          freq_base,
                          freq_scale,
                          ext_factor,
                          attn_factor,
                          DS4_ROPE_YARN_BETA_FAST,
                          DS4_ROPE_YARN_BETA_SLOW,
                          inverse);
}



static void rope_tail_batch_worker(void *vctx, uint64_t t0, uint64_t t1) {
    rope_tail_batch_ctx *ctx = static_cast<rope_tail_batch_ctx *>(vctx);
    for (uint64_t tt = t0; tt < t1; tt++) {
        rope_tail_layer_inplace(ctx->x + tt * ctx->stride,
                                ctx->n_head,
                                ctx->head_dim,
                                ctx->n_rot,
                                ctx->pos0 + (uint32_t)tt,
                                ctx->il,
                                ctx->inverse);
    }
}



void rope_tail_layer_batch_inplace(
        float            *x,
        uint64_t          stride,
        uint32_t          n_head,
        uint32_t          head_dim,
        uint32_t          n_rot,
        uint32_t          pos0,
        uint32_t          il,
        bool              inverse,
        uint32_t          n_tok) {
    rope_tail_batch_ctx ctx = {
        .x = x,
        .stride = stride,
        .n_head = n_head,
        .head_dim = head_dim,
        .n_rot = n_rot,
        .pos0 = pos0,
        .il = il,
        .inverse = inverse,
    };
    ds4_parallel_for_min_rows(n_tok, rope_tail_batch_worker, &ctx, 1);
}



float sigmoid_stable(float x) {
    if (x >= 0.0f) {
        const float e = expf(-x);
        return 1.0f / (1.0f + e);
    } else {
        const float e = expf(x);
        return e / (1.0f + e);
    }
}



/* Sink-aware attention over a set of KV rows.  The learned sink logit is part
 * of the softmax denominator but contributes no value vector. */
void layer_attention_rows_one(
        float             * out_heads,
        const ds4_model   * model,
        const ds4_layer_weights * layer,
        const float       * q,
        const float       * kv_rows,
        uint32_t            n_kv) {
    const float *sinks = (const float *)tensor_data(model, layer->attn_sinks);
    const float kq_scale = 1.0f / sqrtf((float)DS4_N_HEAD_DIM);
    float score_stack[512];
    float *score = n_kv <= 512 ? score_stack : (float *)xmalloc((size_t)n_kv * sizeof(score[0]));

    for (uint32_t h = 0; h < DS4_N_HEAD; h++) {
        const float *qh = q + (uint64_t)h * DS4_N_HEAD_DIM;

        float max_score = sinks[h];
        for (uint32_t r = 0; r < n_kv; r++) {
            const float *kv = kv_rows + (uint64_t)r * DS4_N_HEAD_DIM;
            score[r] = dot_f32(qh, kv, DS4_N_HEAD_DIM) * kq_scale;
            if (score[r] > max_score) max_score = score[r];
        }

        float *oh = out_heads + (uint64_t)h * DS4_N_HEAD_DIM;
        memset(oh, 0, (size_t)DS4_N_HEAD_DIM * sizeof(oh[0]));

        float denom = expf(sinks[h] - max_score);
        for (uint32_t r = 0; r < n_kv; r++) {
            const float weight = expf(score[r] - max_score);
            const float *kv = kv_rows + (uint64_t)r * DS4_N_HEAD_DIM;
            denom += weight;
            axpy_f32(oh, kv, weight, DS4_N_HEAD_DIM);
        }

        const float inv = 1.0f / denom;
        scale_f32(oh, inv, DS4_N_HEAD_DIM);
    }

    if (score != score_stack) free(score);
}



void layer_attention_one(
        float             * out_heads,
        const ds4_model   * model,
        const ds4_layer_weights * layer,
        const float       * q,
        const float       * kv) {
    layer_attention_rows_one(out_heads, model, layer, q, kv, 1);
}



/* Attention output projection is grouped: each group first maps its heads to
 * a 1024-rank low vector, then all groups are projected back to 4096. */
void layer_grouped_out_one(
        float             * out,
        const ds4_model   * model,
        const ds4_layer_weights * layer,
        const float       * heads) {
    const uint32_t n_groups = 8;
    const uint32_t group_heads = DS4_N_HEAD / n_groups;
    const uint32_t group_dim = DS4_N_HEAD_DIM * group_heads;
    const uint32_t rank = 1024;

    float *low = (float *)xcalloc((size_t)n_groups * rank, sizeof(low[0]));

    matvec_q8_0_grouped_rows(low, model, layer->attn_output_a, heads, n_groups, group_dim, rank);

    matvec_q8_0(out, model, layer->attn_output_b, low);
    free(low);
}






void layer_grouped_out_batch(
        float             * out,
        const ds4_model   * model,
        const ds4_layer_weights * layer,
        const float       * heads,
        uint32_t            n_tok) {
    const uint32_t n_groups = 8;
    const uint32_t group_heads = DS4_N_HEAD / n_groups;
    const uint32_t group_dim = DS4_N_HEAD_DIM * group_heads;
    const uint32_t rank = 1024;

    float *low = (float *)xcalloc((size_t)n_tok * n_groups * rank, sizeof(low[0]));

    matmul_q8_0_grouped_batch(low, model, layer->attn_output_a, heads,
                              n_tok, n_groups, group_dim, rank);
    matmul_q8_0_batch(out, model, layer->attn_output_b, low, n_tok);

    free(low);
}



/* =========================================================================
 * Mixture-of-Experts FFN.
 * =========================================================================
 *
 * This is the FFN half of each layer.  It includes the shared expert, routed
 * expert selection, IQ2_XXS gate/up projections, SwiGLU, Q2_K down projection,
 * and the HC post step that returns the result to four-stream state.
 */

float silu(float x) {
    return x * sigmoid_stable(x);
}

