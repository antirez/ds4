#include "ds4_engine_internal.h"



/* Reduce the four HC streams into the plain embedding vector consumed by a
 * normal attention or FFN sublayer. */
void hc_weighted_sum_one(
        float       * out,
        const float * x,
        const float * weights,
        uint32_t      n_embd,
        uint32_t      n_hc) {
    for (uint32_t d = 0; d < n_embd; d++) {
        float acc = 0.0f;
        for (uint32_t h = 0; h < n_hc; h++) {
            acc += x[(uint64_t)h * n_embd + d] * weights[h];
        }
        out[d] = acc;
    }
}



/* HC pre step for one token.  It normalizes the HC state, projects the control
 * vector, runs the Sinkhorn split, and emits the sublayer input plus post data.
 * The caller supplies scratch buffers (out/post/comb/flat); shared by the GPU
 * decode path via hc_pre_from_state_one. */
void hc_pre_from_state_one_scratch(
        const ds4_model   * model,
        const ds4_tensor  * fn,
        const ds4_tensor  * scale_tensor,
        const ds4_tensor  * base_tensor,
        const float       * residual_hc,
        float             * out,
        float             * post,
        float             * comb,
        float             * flat,
        bool                serial_fn) {
    const uint32_t n_hc = DS4_N_HC;
    const uint64_t hc_dim = (uint64_t)DS4_N_EMBD * n_hc;

    float mix[24];
    float split[24];

    rms_norm_no_weight(flat, residual_hc, hc_dim, DS4_RMS_EPS);
    if (serial_fn) {
        matvec_f16_serial(mix, model, fn, flat);
    } else {
        matvec_f16(mix, model, fn, flat);
    }

    const float *scale = (const float *)tensor_data(model, scale_tensor);
    const float *base = (const float *)tensor_data(model, base_tensor);
    hc_split_sinkhorn_one(split, mix, scale, base, (int)n_hc, DS4_N_HC_SINKHORN_ITER, 1.0e-6f);
    hc_weighted_sum_one(out, residual_hc, split, DS4_N_EMBD, n_hc);

    memcpy(post, split + n_hc, n_hc * sizeof(post[0]));
    memcpy(comb, split + 2 * n_hc, n_hc * n_hc * sizeof(comb[0]));
}



void hc_pre_from_state_one(
        const ds4_model   * model,
        const ds4_tensor  * fn,
        const ds4_tensor  * scale_tensor,
        const ds4_tensor  * base_tensor,
        const float       * residual_hc,
        float             * out,
        float             * post,
        float             * comb) {
    const uint64_t hc_dim = (uint64_t)DS4_N_EMBD * DS4_N_HC;
    float *flat = (float *)xmalloc((size_t)hc_dim * sizeof(flat[0]));

    hc_pre_from_state_one_scratch(model,
                                  fn, scale_tensor, base_tensor,
                                  residual_hc, out, post, comb,
                                  flat, false);
    free(flat);
}



void layer_attn_pre_one(
        const ds4_model   * model,
        const ds4_layer_weights * layer,
        const float       * token_embd,
        float             * out,
        float             * residual_hc,
        float             * post,
        float             * comb) {
    const uint32_t n_hc = DS4_N_HC;

    for (uint32_t h = 0; h < n_hc; h++) {
        memcpy(residual_hc + (uint64_t)h * DS4_N_EMBD, token_embd, (size_t)DS4_N_EMBD * sizeof(token_embd[0]));
    }

    hc_pre_from_state_one(model,
                          layer->hc_attn_fn,
                          layer->hc_attn_scale,
                          layer->hc_attn_base,
                          residual_hc, out, post, comb);
}



/* The input embedding starts all HC streams with the same token vector. */
void hc_from_plain_embedding(float *out_hc, const float *x, uint32_t n_embd, uint32_t n_hc) {
    for (uint32_t h = 0; h < n_hc; h++) {
        memcpy(out_hc + (uint64_t)h * n_embd, x, (size_t)n_embd * sizeof(x[0]));
    }
}



/* HC post step for one sublayer output.  It injects the new block output and
 * mixes the previous HC streams through the learned combine matrix. */
void hc_post_one(
        float       * out_hc,
        const float * block_out,
        const float * residual_hc,
        const float * post,
        const float * comb,
        uint32_t      n_embd,
        uint32_t      n_hc) {
    for (uint32_t dst = 0; dst < n_hc; dst++) {
        for (uint32_t d = 0; d < n_embd; d++) {
            float acc = block_out[d] * post[dst];

            for (uint32_t src = 0; src < n_hc; src++) {
                /* The HC combine matrix is addressed as [dst_hc, src_hc]. */
                acc += comb[dst + src * n_hc] * residual_hc[(uint64_t)src * n_embd + d];
            }

            out_hc[(uint64_t)dst * n_embd + d] = acc;
        }
    }
}



static void hc_post_batch_worker(void *vctx, uint64_t t0, uint64_t t1) {
    hc_post_batch_ctx *ctx = static_cast<hc_post_batch_ctx *>(vctx);
    for (uint64_t t = t0; t < t1; t++) {
        hc_post_one(ctx->out_hc + t * ctx->hc_dim,
                    ctx->block_out + t * ctx->n_embd,
                    ctx->residual_hc + t * ctx->hc_dim,
                    ctx->post + t * ctx->n_hc,
                    ctx->comb + t * ctx->n_hc * ctx->n_hc,
                    ctx->n_embd,
                    ctx->n_hc);
    }
}



void hc_post_batch(
        float       * out_hc,
        const float * block_out,
        const float * residual_hc,
        const float * post,
        const float * comb,
        uint32_t      n_tok,
        uint32_t      n_embd,
        uint32_t      n_hc) {
    hc_post_batch_ctx ctx = {
        .out_hc = out_hc,
        .block_out = block_out,
        .residual_hc = residual_hc,
        .post = post,
        .comb = comb,
        .hc_dim = (uint64_t)n_hc * n_embd,
        .n_embd = n_embd,
        .n_hc = n_hc,
    };
    ds4_parallel_for_min_rows(n_tok, hc_post_batch_worker, &ctx, 1);
}



static void hc_post_sum_batch_worker(void *vctx, uint64_t t0, uint64_t t1) {
    hc_post_sum_batch_ctx *ctx = static_cast<hc_post_sum_batch_ctx *>(vctx);
    for (uint64_t t = t0; t < t1; t++) {
        const float *moe = ctx->moe + t * ctx->n_embd;
        const float *shared = ctx->shared + t * ctx->n_embd;
        const float *residual = ctx->residual_hc + t * ctx->hc_dim;
        const float *post = ctx->post + t * ctx->n_hc;
        const float *comb = ctx->comb + t * ctx->n_hc * ctx->n_hc;
        float *out = ctx->out_hc + t * ctx->hc_dim;

        for (uint32_t dst = 0; dst < ctx->n_hc; dst++) {
            for (uint32_t d = 0; d < ctx->n_embd; d++) {
                float acc = (moe[d] + shared[d]) * post[dst];
                for (uint32_t src = 0; src < ctx->n_hc; src++) {
                    acc += comb[dst + src * ctx->n_hc] *
                        residual[(uint64_t)src * ctx->n_embd + d];
                }
                out[(uint64_t)dst * ctx->n_embd + d] = acc;
            }
        }
    }
}



void hc_post_sum_batch(
        float       * out_hc,
        const float * moe,
        const float * shared,
        const float * residual_hc,
        const float * post,
        const float * comb,
        uint32_t      n_tok,
        uint32_t      n_embd,
        uint32_t      n_hc) {
    hc_post_sum_batch_ctx ctx = {
        .out_hc = out_hc,
        .moe = moe,
        .shared = shared,
        .residual_hc = residual_hc,
        .post = post,
        .comb = comb,
        .hc_dim = (uint64_t)n_hc * n_embd,
        .n_embd = n_embd,
        .n_hc = n_hc,
    };
    ds4_parallel_for_min_rows(n_tok, hc_post_sum_batch_worker, &ctx, 1);
}



static void hc_pre_norm_batch_worker(void *vctx, uint64_t t0, uint64_t t1) {
    hc_pre_norm_batch_ctx *ctx = static_cast<hc_pre_norm_batch_ctx *>(vctx);
    const float *norm_w = (const float *)tensor_data(ctx->model, ctx->norm_w);
    float *flat = (float *)xmalloc((size_t)ctx->hc_dim * sizeof(flat[0]));

    for (uint64_t t = t0; t < t1; t++) {
        const float *residual = ctx->inp_hc + t * ctx->hc_dim;
        if (ctx->residual_hc) {
            float *dst = ctx->residual_hc + t * ctx->hc_dim;
            memcpy(dst, residual, (size_t)ctx->hc_dim * sizeof(dst[0]));
            residual = dst;
        }

        hc_pre_from_state_one_scratch(ctx->model,
                                      ctx->fn,
                                      ctx->scale,
                                      ctx->base,
                                      residual,
                                      ctx->cur + t * DS4_N_EMBD,
                                      ctx->post + t * ctx->n_hc,
                                      ctx->comb + t * ctx->n_hc * ctx->n_hc,
                                      flat,
                                      true);
        rms_norm_weight(ctx->norm + t * DS4_N_EMBD,
                        ctx->cur + t * DS4_N_EMBD,
                        norm_w,
                        DS4_N_EMBD,
                        DS4_RMS_EPS);
    }

    free(flat);
}



/* Batched HC pre plus RMSNorm.  Prefill uses this to keep the layer-major
 * token batch in contiguous arrays. */
void hc_pre_norm_batch(
        const ds4_model  * model,
        const ds4_tensor * fn,
        const ds4_tensor * scale,
        const ds4_tensor * base,
        const ds4_tensor * norm_w,
        const float      * inp_hc,
        float            * residual_hc,
        float            * cur,
        float            * norm,
        float            * post,
        float            * comb,
        uint32_t           n_tok) {
    hc_pre_norm_batch_ctx ctx = {
        .model = model,
        .fn = fn,
        .scale = scale,
        .base = base,
        .norm_w = norm_w,
        .inp_hc = inp_hc,
        .residual_hc = residual_hc,
        .cur = cur,
        .norm = norm,
        .post = post,
        .comb = comb,
        .hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD,
        .n_hc = DS4_N_HC,
    };
    ds4_parallel_for_min_rows(n_tok, hc_pre_norm_batch_worker, &ctx, 1);
}



void layer_attn_norm_one(
        float             * out,
        const ds4_model   * model,
        const ds4_layer_weights * layer,
        const float       * x) {
    const float *attn_norm = (const float *)tensor_data(model, layer->attn_norm);
    rms_norm_weight(out, x, attn_norm, DS4_N_EMBD, DS4_RMS_EPS);
}



/* =========================================================================
 * Attention Projections, RoPE, and Attention Output.
 * =========================================================================
 *
 * This block performs the attention half of a transformer layer: HC pre,
 * attention RMSNorm, Q and KV projections, layer-specific RoPE, sink-aware
 * attention over raw and compressed KV rows, and the grouped LoRA output
 * projection back to embedding width.
 */

/* Q projection is low-rank: Q8_0 into the model-specific LoRA-Q rank,
 * RMSNorm, then Q8_0 back to all attention heads. */
void layer_q_projection_normed_one(
        const ds4_model   * model,
        const ds4_layer_weights * layer,
        const float       * norm,
        float             * q) {
    const uint32_t q_rank = DS4_N_LORA_Q;
    float *qr = (float *)xmalloc((size_t)q_rank * sizeof(qr[0]));
    float *qr_norm = (float *)xmalloc((size_t)q_rank * sizeof(qr_norm[0]));

    const float *q_a_norm = (const float *)tensor_data(model, layer->attn_q_a_norm);

    matvec_q8_0(qr, model, layer->attn_q_a, norm);
    rms_norm_weight(qr_norm, qr, q_a_norm, q_rank, DS4_RMS_EPS);
    matvec_q8_0(q, model, layer->attn_q_b, qr_norm);
    head_rms_norm_inplace(q, DS4_N_HEAD, DS4_N_HEAD_DIM, DS4_RMS_EPS);

    free(qr_norm);
    free(qr);
}

