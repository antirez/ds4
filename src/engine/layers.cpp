#include "pulsar_engine_internal.h"



uint32_t pulsar_default_raw_cap(uint32_t ctx_size) {
    uint32_t raw_cap = PULSAR_N_SWA;
    if (raw_cap > ctx_size) raw_cap = ctx_size;
    if (raw_cap == 0) raw_cap = 1;
    return raw_cap;
}



uint32_t pulsar_prefill_cap_for_prompt(int prompt_len,
                                           uint32_t requested_chunk) {
    if (prompt_len <= 0) return 1;
    uint32_t cap = (uint32_t)prompt_len;

    if (requested_chunk != 0) {
        cap = requested_chunk;
    } else {
        const char *env = getenv("DS4_CUDA_PREFILL_CHUNK");
        if (env && env[0]) {
            char *endp = NULL;
            const long v = strtol(env, &endp, 10);
            if (endp != env) {
                if (v <= 0) return cap;
                cap = (uint32_t)v;
            }
        } else if (prompt_len > 4096) {
            cap = PULSAR_MODEL_VARIANT == PULSAR_VARIANT_PRO ? 8192u : 4096u;
        }
    }

    if (cap == 0) cap = 1;
    if (cap > (uint32_t)prompt_len) cap = (uint32_t)prompt_len;
    return cap;
}



/* Allocate all CPU decode temporaries once.  This keeps generation deterministic
 * from the VM's point of view and makes accidental hot-loop malloc visible. */






/* Append to the raw SWA cache.  Once full, it slides by one row. */






/* After prefill, clear unused compressor state rows so decode starts from the
 * same partial-window state the streaming path would have produced. */






/* Pool the current compression window with a softmax over per-dimension scores.
 * Ratio-4 layers keep two lanes: attention compression and indexer compression. */



/* Streaming compressor update for one token.  It projects kv/score rows,
 * updates the rolling state, and emits a compressed KV row on ratio boundaries. */






/* Attention over raw SWA rows plus optional compressed rows.  Ratio-4 layers
 * pass an indexer mask to hide compressed rows not selected for this token. */












/* Prefix prefill attention for a fresh prompt.  It computes each token's view
 * of the raw window and compressed rows without running the decode loop. */



/* Ratio-4 layers use an auxiliary indexer to select which compressed rows are
 * visible to attention.  This is the CPU allocation-owning helper. */



/* Scratch-backed indexer selection for decode. */



/* Single-token attention sublayer with raw SWA cache and DS4 compression. */



/* Batched prefill attention.  It projects Q/KV for all tokens, streams them
 * through the same raw/compressed cache updates, then runs prefix attention. */



/* Full transformer layer for one decode token: attention sublayer followed by
 * FFN sublayer, both operating on the HC state. */















/* Diagnostic first-token layer without cache history: the token attends only
 * to itself, useful for checking a minimal end-to-end slice. */
void layer_forward_self_one(
        float                   * out_hc,
        const pulsar_model         * model,
        const pulsar_layer_weights * layer,
        const float             * inp_hc,
        uint32_t                  il,
        uint32_t                  pos,
        int                       token) {
    const uint32_t n_hc = PULSAR_N_HC;
    const uint64_t q_dim = (uint64_t)PULSAR_N_HEAD * PULSAR_N_HEAD_DIM;

    float *attn_cur = (float *)xmalloc((size_t)PULSAR_N_EMBD * sizeof(attn_cur[0]));
    float *attn_norm = (float *)xmalloc((size_t)PULSAR_N_EMBD * sizeof(attn_norm[0]));
    float *attn_residual = (float *)xmalloc((size_t)n_hc * PULSAR_N_EMBD * sizeof(attn_residual[0]));
    float *q = (float *)xmalloc((size_t)q_dim * sizeof(q[0]));
    float *kv = (float *)xmalloc((size_t)PULSAR_N_HEAD_DIM * sizeof(kv[0]));
    float *heads = (float *)xmalloc((size_t)q_dim * sizeof(heads[0]));
    float *attn_out = (float *)xmalloc((size_t)PULSAR_N_EMBD * sizeof(attn_out[0]));
    float *after_attn_hc = (float *)xmalloc((size_t)n_hc * PULSAR_N_EMBD * sizeof(after_attn_hc[0]));
    float post[4];
    float comb[16];

    memcpy(attn_residual, inp_hc, (size_t)n_hc * PULSAR_N_EMBD * sizeof(inp_hc[0]));
    hc_pre_from_state_one(model,
                          layer->hc_attn_fn,
                          layer->hc_attn_scale,
                          layer->hc_attn_base,
                          attn_residual, attn_cur, post, comb);

    layer_attn_norm_one(attn_norm, model, layer, attn_cur);
    layer_q_projection_normed_one(model, layer, attn_norm, q);
    layer_kv_projection_normed_one(model, layer, attn_norm, kv);
    rope_tail_layer_inplace(q, PULSAR_N_HEAD, PULSAR_N_HEAD_DIM, PULSAR_N_ROT, pos, il, false);
    rope_tail_layer_inplace(kv, PULSAR_N_HEAD_KV, PULSAR_N_HEAD_DIM, PULSAR_N_ROT, pos, il, false);
    dsv4_fp8_kv_quantize_row_inplace_cpu(kv, PULSAR_N_HEAD_DIM, PULSAR_N_ROT);
    f16_round_inplace_cpu(kv, PULSAR_N_HEAD_DIM);

    layer_attention_one(heads, model, layer, q, kv);
    rope_tail_layer_inplace(heads, PULSAR_N_HEAD, PULSAR_N_HEAD_DIM, PULSAR_N_ROT, pos, il, true);
    layer_grouped_out_one(attn_out, model, layer, heads);
    hc_post_one(after_attn_hc, attn_out, attn_residual, post, comb, PULSAR_N_EMBD, n_hc);

    layer_ffn_one(out_hc, model, layer, after_attn_hc, il, token,
                  NULL, 0.0f, false);

    free(after_attn_hc);
    free(attn_out);
    free(heads);
    free(kv);
    free(q);
    free(attn_residual);
    free(attn_norm);
    free(attn_cur);
}






/* Collapse final HC streams into the ordinary embedding vector before the
 * output norm and vocabulary projection. */
static void output_hc_head_one(
        float             * out,
        const pulsar_model   * model,
        const pulsar_weights * weights,
        const float       * inp_hc) {
    const uint32_t n_hc = PULSAR_N_HC;
    const uint64_t hc_dim = (uint64_t)PULSAR_N_EMBD * n_hc;
    float *flat = (float *)xmalloc((size_t)hc_dim * sizeof(flat[0]));
    float *pre = (float *)xmalloc((size_t)n_hc * sizeof(pre[0]));
    float *w = (float *)xmalloc((size_t)n_hc * sizeof(w[0]));

    rms_norm_no_weight(flat, inp_hc, hc_dim, PULSAR_RMS_EPS);
    matvec_f16(pre, model, weights->output_hc_fn, flat);

    const float *scale = (const float *)tensor_data(model, weights->output_hc_scale);
    const float *base = (const float *)tensor_data(model, weights->output_hc_base);
    for (uint32_t i = 0; i < n_hc; i++) {
        w[i] = sigmoid_stable(pre[i] * scale[0] + base[i]) + PULSAR_HC_EPS;
    }

    hc_weighted_sum_one(out, inp_hc, w, PULSAR_N_EMBD, n_hc);

    free(w);
    free(pre);
    free(flat);
}



/* Final language-model head: HC collapse, RMSNorm, and Q8_0 vocab projection. */
void output_logits_one(
        float             * logits,
        const pulsar_model   * model,
        const pulsar_weights * weights,
        const float       * inp_hc) {
    float *embd = (float *)xmalloc((size_t)PULSAR_N_EMBD * sizeof(embd[0]));
    float *norm = (float *)xmalloc((size_t)PULSAR_N_EMBD * sizeof(norm[0]));

    output_hc_head_one(embd, model, weights, inp_hc);
    rms_norm_weight(norm, embd, (const float *)tensor_data(model, weights->output_norm), PULSAR_N_EMBD, PULSAR_RMS_EPS);

    matvec_q8_0(logits, model, weights->output, norm);

    free(norm);
    free(embd);
}



/* Allocation-free logits head for CPU decode. */



int sample_argmax(const float *logits, uint32_t n_vocab);



/* =========================================================================
 * GPU Reference Comparison Helpers.
 * =========================================================================
 *
 * These small scalar helpers are used only by diagnostics that compare the C
 * reference path with the GPU executor.
 */

float max_abs_diff(const float *a, const float *b, uint64_t n) {
    float max_diff = 0.0f;
    for (uint64_t i = 0; i < n; i++) {
        const float diff = fabsf(a[i] - b[i]);
        if (diff > max_diff) max_diff = diff;
    }
    return max_diff;
}

