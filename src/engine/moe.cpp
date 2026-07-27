#include "pulsar_engine_internal.h"



float softplus_stable(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return expf(x);
    return log1pf(expf(x));
}



void swiglu(float *out, const float *gate, const float *up, uint64_t n, float clamp) {
    for (uint64_t i = 0; i < n; i++) {
        float g = gate[i];
        float u = up[i];
        if (clamp > 1.0e-6f) {
            if (g > clamp) g = clamp;
            if (u > clamp) u = clamp;
            if (u < -clamp) u = -clamp;
        }
        out[i] = silu(g) * u;
    }
}



/* The shared expert is a normal Q8_0 SwiGLU MLP that runs for every token. */
void layer_shared_ffn_one(
        float             * out,
        const pulsar_model   * model,
        const pulsar_layer_weights * layer,
        const float       * x) {
    float *gate = (float *)xmalloc((size_t)PULSAR_N_FF_EXP * sizeof(gate[0]));
    float *up = (float *)xmalloc((size_t)PULSAR_N_FF_EXP * sizeof(up[0]));
    float *mid = (float *)xmalloc((size_t)PULSAR_N_FF_EXP * sizeof(mid[0]));
    const uint64_t in_dim = layer->ffn_gate_shexp->dim[0];
    const uint64_t blocks = (in_dim + 31) / 32;
    int8_t *xq = (int8_t *)xmalloc((size_t)blocks * 32);
    float *xscale = (float *)xmalloc((size_t)blocks * sizeof(xscale[0]));

    if (layer->ffn_up_shexp->type != 8 ||
        layer->ffn_gate_shexp->type != 8 ||
        layer->ffn_up_shexp->dim[0] != in_dim) {
        pulsar_die("shared expert gate/up tensors do not share a Q8_0 input layout");
    }

    quantize_q8_0_activation(x, xq, xscale, in_dim);
    matvec_q8_0_pair_prequant(gate, up, model,
                              layer->ffn_gate_shexp,
                              layer->ffn_up_shexp,
                              xq, xscale);
    swiglu(mid, gate, up, PULSAR_N_FF_EXP, PULSAR_SWIGLU_CLAMP_EXP);
    matvec_q8_0(out, model, layer->ffn_down_shexp, mid);

    free(xscale);
    free(xq);
    free(mid);
    free(up);
    free(gate);
}






static void swiglu_batch_worker(void *vctx, uint64_t t0, uint64_t t1) {
    swiglu_batch_ctx *ctx = static_cast<swiglu_batch_ctx *>(vctx);
    for (uint64_t t = t0; t < t1; t++) {
        swiglu(ctx->mid + t * ctx->n,
               ctx->gate + t * ctx->n,
               ctx->up + t * ctx->n,
               ctx->n,
               ctx->clamp);
    }
}



static void layer_shared_ffn_batch(
        float             * out,
        const pulsar_model   * model,
        const pulsar_layer_weights * layer,
        const float       * x,
        uint32_t            n_tok) {
    const uint64_t in_dim = layer->ffn_gate_shexp->dim[0];
    const uint64_t hidden = layer->ffn_gate_shexp->dim[1];

    if (layer->ffn_up_shexp->type != 8 ||
        layer->ffn_gate_shexp->type != 8 ||
        layer->ffn_down_shexp->type != 8 ||
        layer->ffn_up_shexp->dim[0] != in_dim ||
        layer->ffn_up_shexp->dim[1] != hidden ||
        layer->ffn_down_shexp->dim[0] != hidden) {
        pulsar_die("shared expert tensors do not share the expected Q8_0 layout");
    }

    float *gate = (float *)xmalloc((size_t)n_tok * hidden * sizeof(gate[0]));
    float *up = (float *)xmalloc((size_t)n_tok * hidden * sizeof(up[0]));
    float *mid = (float *)xmalloc((size_t)n_tok * hidden * sizeof(mid[0]));

    matmul_q8_0_pair_batch(gate, up, model,
                           layer->ffn_gate_shexp,
                           layer->ffn_up_shexp,
                           x,
                           n_tok);

    swiglu_batch_ctx swiglu_ctx = {
        .mid = mid,
        .gate = gate,
        .up = up,
        .n = hidden,
        .clamp = PULSAR_SWIGLU_CLAMP_EXP,
    };
    pulsar_parallel_for(n_tok, swiglu_batch_worker, &swiglu_ctx);

    matmul_q8_0_batch(out, model, layer->ffn_down_shexp, mid, n_tok);

    free(mid);
    free(up);
    free(gate);
}



/* Early DS4 layers use token-id hash routing instead of top-k routing. */
void layer_hash_selected_experts(
        int                    selected[PULSAR_MAX_EXPERT_USED],
        const pulsar_model       *model,
        const pulsar_layer_weights *layer,
        int                    token) {
    pulsar_tensor *t = layer->ffn_gate_tid2eid;
    if (!t) pulsar_die("hash routing table is missing for this layer");
    if (t->type != 26 || t->ndim != 2 || t->dim[0] != PULSAR_N_EXPERT_USED) {
        pulsar_die("ffn_gate_tid2eid.weight has an unexpected layout");
    }
    if (token < 0 || (uint64_t)token >= t->dim[1]) {
        pulsar_die("token id is outside the hash routing table");
    }

    const int32_t *table = (const int32_t *)tensor_data(model, t);
    const int32_t *row = table + (uint64_t)token * PULSAR_N_EXPERT_USED;
    for (uint32_t i = 0; i < PULSAR_N_EXPERT_USED; i++) selected[i] = row[i];
}



/* Router scores use sqrt(softplus(logit)); normalization happens only after
 * the six selected experts are known. */
static void layer_router_probs_one(
        float             probs[PULSAR_MAX_EXPERT],
        const pulsar_model   * model,
        const pulsar_layer_weights * layer,
        const float       * x) {
    float logits[PULSAR_MAX_EXPERT];

    matvec_any(logits, model, layer->ffn_gate_inp, x);
    for (uint32_t i = 0; i < PULSAR_N_EXPERT; i++) {
        probs[i] = sqrtf(softplus_stable(logits[i]));
    }
}



void layer_hash_router_weights_from_probs(
        float             weights_out[PULSAR_MAX_EXPERT_USED],
        const float       probs[PULSAR_MAX_EXPERT],
        const int          selected[PULSAR_MAX_EXPERT_USED]) {
    float sum = 0.0f;
    for (uint32_t i = 0; i < PULSAR_N_EXPERT_USED; i++) {
        if (selected[i] < 0 || (uint32_t)selected[i] >= PULSAR_N_EXPERT) pulsar_die("hash-selected expert is outside router range");
        weights_out[i] = probs[selected[i]];
        sum += weights_out[i];
    }

    if (sum < 6.103515625e-5f) sum = 6.103515625e-5f;
    for (uint32_t i = 0; i < PULSAR_N_EXPERT_USED; i++) {
        weights_out[i] = weights_out[i] / sum * PULSAR_EXPERT_WEIGHT_SCALE;
    }
}



void layer_hash_router_weights_one(
        float             weights_out[PULSAR_MAX_EXPERT_USED],
        const pulsar_model   * model,
        const pulsar_layer_weights * layer,
        const float       * x,
        const int          selected[PULSAR_MAX_EXPERT_USED]) {
    float probs[PULSAR_MAX_EXPERT];

    layer_router_probs_one(probs, model, layer, x);
    layer_hash_router_weights_from_probs(weights_out, probs, selected);
}



static void topk_desc(const float *score, int n, int k, int *idx) {
    for (int i = 0; i < k; i++) idx[i] = -1;

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < k; j++) {
            if (idx[j] < 0 || score[i] > score[idx[j]]) {
                for (int m = k - 1; m > j; m--) idx[m] = idx[m - 1];
                idx[j] = i;
                break;
            }
        }
    }
}



/* Later layers choose the six experts by biased top-k, but weight them using
 * the unbiased router probabilities. */
void layer_topk_selected_experts_from_probs(
        int                    selected[PULSAR_MAX_EXPERT_USED],
        float                  expert_weight[PULSAR_MAX_EXPERT_USED],
        const pulsar_model       *model,
        const pulsar_layer_weights *layer,
        const float           probs[PULSAR_MAX_EXPERT]);



void layer_topk_selected_experts(
        int                    selected[PULSAR_MAX_EXPERT_USED],
        float                  expert_weight[PULSAR_MAX_EXPERT_USED],
        const pulsar_model       *model,
        const pulsar_layer_weights *layer,
        const float           *x) {
    float probs[PULSAR_MAX_EXPERT];

    layer_router_probs_one(probs, model, layer, x);
    layer_topk_selected_experts_from_probs(selected, expert_weight, model, layer, probs);
}



void layer_topk_selected_experts_from_probs(
        int                    selected[PULSAR_MAX_EXPERT_USED],
        float                  expert_weight[PULSAR_MAX_EXPERT_USED],
        const pulsar_model       *model,
        const pulsar_layer_weights *layer,
        const float           probs[PULSAR_MAX_EXPERT]) {
    float selection[PULSAR_MAX_EXPERT];

    memcpy(selection, probs, sizeof(selection));

    if (layer->ffn_exp_probs_b) {
        const float *bias = (const float *)tensor_data(model, layer->ffn_exp_probs_b);
        for (uint32_t i = 0; i < PULSAR_N_EXPERT; i++) selection[i] += bias[i];
    }

    topk_desc(selection, (int)PULSAR_N_EXPERT, (int)PULSAR_N_EXPERT_USED, selected);

    float sum = 0.0f;
    for (uint32_t i = 0; i < PULSAR_N_EXPERT_USED; i++) {
        expert_weight[i] = probs[selected[i]];
        sum += expert_weight[i];
    }
    if (sum < 6.103515625e-5f) sum = 6.103515625e-5f;
    for (uint32_t i = 0; i < PULSAR_N_EXPERT_USED; i++) {
        expert_weight[i] = expert_weight[i] / sum * PULSAR_EXPERT_WEIGHT_SCALE;
    }
}



void print_vec_stats(const char *name, const float *x, uint64_t n);



/* Single-token routed MoE.  It selects six experts, runs IQ2_XXS gate/up,
 * applies SwiGLU and router weights, then accumulates Q2_K down projections. */
static void layer_routed_moe_one(
        float             * out,
        const pulsar_model   * model,
        const pulsar_layer_weights * layer,
        const float       * x,
        uint32_t            il,
        int                 token,
        float               clamp,
        bool                trace) {
    int selected[PULSAR_MAX_EXPERT_USED];
    float expert_weight[PULSAR_MAX_EXPERT_USED];
    float *gate = trace ? (float *)xmalloc((size_t)PULSAR_N_FF_EXP * sizeof(gate[0])) : NULL;
    float *up = trace ? (float *)xmalloc((size_t)PULSAR_N_FF_EXP * sizeof(up[0])) : NULL;
    float *mid = trace ? (float *)xmalloc((size_t)PULSAR_N_FF_EXP * sizeof(mid[0])) : NULL;
    float *mid_all = trace ? NULL : (float *)xmalloc((size_t)PULSAR_N_EXPERT_USED * PULSAR_N_FF_EXP * sizeof(mid_all[0]));
    float *down = trace ? (float *)xmalloc((size_t)PULSAR_N_EMBD * sizeof(down[0])) : NULL;
    const uint64_t expert_in_dim = layer->ffn_gate_exps->dim[0];
    const uint64_t down_in_dim = layer->ffn_down_exps->dim[0];
    if (expert_in_dim % QK_K != 0) pulsar_die("routed expert input is not QK_K aligned");
    if (down_in_dim != PULSAR_N_FF_EXP || down_in_dim % QK_K != 0) pulsar_die("routed expert down input has an unexpected layout");
    block_q8_K *xq = (block_q8_K *)xmalloc((size_t)(expert_in_dim / QK_K) * sizeof(xq[0]));
    block_q8_K *midq = trace ? NULL : (block_q8_K *)xmalloc((size_t)PULSAR_N_EXPERT_USED * (down_in_dim / QK_K) * sizeof(midq[0]));

    memset(out, 0, (size_t)PULSAR_N_EMBD * sizeof(out[0]));
    pulsar_quantize_row_q8_K(x, xq, (int64_t)expert_in_dim);

    if (layer->ffn_gate_tid2eid) {
        layer_hash_selected_experts(selected, model, layer, token);
        layer_hash_router_weights_one(expert_weight, model, layer, x, selected);
    } else {
        layer_topk_selected_experts(selected, expert_weight, model, layer, x);
    }

    if (!trace) {
        matvec_experts_mid_prequant(mid_all, model,
                                    layer->ffn_gate_exps,
                                    layer->ffn_up_exps,
                                    xq,
                                    selected,
                                    expert_weight,
                                    PULSAR_N_EXPERT_USED,
                                    clamp);
        for (uint32_t i = 0; i < PULSAR_N_EXPERT_USED; i++) {
            pulsar_quantize_row_q8_K(mid_all + (uint64_t)i * down_in_dim,
                                  midq + (uint64_t)i * (down_in_dim / QK_K),
                                  (int64_t)down_in_dim);
        }
        matvec_experts_down_accum_prequant(out, model, layer->ffn_down_exps, midq, selected, PULSAR_N_EXPERT_USED);
    } else {
        for (uint32_t i = 0; i < PULSAR_N_EXPERT_USED; i++) {
            const uint32_t expert = (uint32_t)selected[i];

            matvec_expert_pair_prequant(gate, up, model,
                                        layer->ffn_gate_exps,
                                        layer->ffn_up_exps,
                                        xq,
                                        expert);
            char name[64];
            snprintf(name, sizeof(name), "blk.%u expert %u gate", il, expert);
            print_vec_stats(name, gate, PULSAR_N_FF_EXP);
            snprintf(name, sizeof(name), "blk.%u expert %u up", il, expert);
            print_vec_stats(name, up, PULSAR_N_FF_EXP);

            /*
             * DeepSeek V4 clamps routed expert gate/up values before SwiGLU and
             * applies the router weight before the down projection.
             */
            const float limit = clamp;
            for (uint32_t j = 0; j < PULSAR_N_FF_EXP; j++) {
                if (limit > 1.0e-6f) {
                    if (gate[j] > limit) gate[j] = limit;
                    if (up[j] > limit) up[j] = limit;
                    if (up[j] < -limit) up[j] = -limit;
                }
                mid[j] = silu(gate[j]) * up[j] * expert_weight[i];
            }

            snprintf(name, sizeof(name), "blk.%u expert %u mid", il, expert);
            print_vec_stats(name, mid, PULSAR_N_FF_EXP);

            matvec_expert_down(down, model, layer->ffn_down_exps, mid, expert);
            snprintf(name, sizeof(name), "blk.%u expert %u down", il, expert);
            print_vec_stats(name, down, PULSAR_N_EMBD);
            for (uint32_t j = 0; j < PULSAR_N_EMBD; j++) out[j] += down[j];
        }
    }

    free(midq);
    free(xq);
    free(down);
    free(mid_all);
    free(mid);
    free(up);
    free(gate);
}



/* Decode version of routed MoE: same math as layer_routed_moe_one(), but all
 * large temporaries come from the persistent scratch arena. */
void layer_routed_moe_one_prealloc(
        float             * out,
        const pulsar_model   * model,
        const pulsar_layer_weights * layer,
        const float       * x,
        uint32_t            il,
        int                 token,
        float               clamp,
        float              * mid_all,
        block_q8_K         * xq,
        block_q8_K         * midq) {
    int selected[PULSAR_MAX_EXPERT_USED];
    float expert_weight[PULSAR_MAX_EXPERT_USED];
    const uint64_t expert_in_dim = layer->ffn_gate_exps->dim[0];
    const uint64_t down_in_dim = layer->ffn_down_exps->dim[0];

    if (expert_in_dim % QK_K != 0) pulsar_die("routed expert input is not QK_K aligned");
    if (down_in_dim != PULSAR_N_FF_EXP || down_in_dim % QK_K != 0) pulsar_die("routed expert down input has an unexpected layout");

    memset(out, 0, (size_t)PULSAR_N_EMBD * sizeof(out[0]));
    pulsar_quantize_row_q8_K(x, xq, (int64_t)expert_in_dim);

    if (layer->ffn_gate_tid2eid) {
        layer_hash_selected_experts(selected, model, layer, token);
        layer_hash_router_weights_one(expert_weight, model, layer, x, selected);
    } else {
        layer_topk_selected_experts(selected, expert_weight, model, layer, x);
    }

    matvec_experts_mid_prequant(mid_all, model,
                                layer->ffn_gate_exps,
                                layer->ffn_up_exps,
                                xq,
                                selected,
                                expert_weight,
                                PULSAR_N_EXPERT_USED,
                                clamp);

    for (uint32_t i = 0; i < PULSAR_N_EXPERT_USED; i++) {
        pulsar_quantize_row_q8_K(mid_all + (uint64_t)i * down_in_dim,
                              midq + (uint64_t)i * (down_in_dim / QK_K),
                              (int64_t)down_in_dim);
    }
    matvec_experts_down_accum_prequant(out, model, layer->ffn_down_exps, midq, selected, PULSAR_N_EXPERT_USED);

    (void)il;
}



/* Prefill MoE groups token/expert pairs by expert so each active expert's
 * rows are scanned once for the whole token batch. */
static void layer_routed_moe_batch(
        float             * moe,
        const pulsar_model   * model,
        const pulsar_layer_weights * layer,
        const float       * norm,
        const int         * token_ids,
        uint32_t            n_tok,
        uint32_t            il,
        float               clamp) {
    const uint64_t expert_in_dim = layer->ffn_gate_exps->dim[0];
    const uint64_t expert_out_dim = layer->ffn_gate_exps->dim[1];
    const uint64_t down_in_dim = layer->ffn_down_exps->dim[0];
    const uint64_t down_out_dim = layer->ffn_down_exps->dim[1];
    if (expert_in_dim % QK_K != 0) pulsar_die("routed expert input is not QK_K aligned");
    if (down_in_dim % QK_K != 0) pulsar_die("routed expert down input is not QK_K aligned");
    if (expert_out_dim != down_in_dim || down_out_dim != PULSAR_N_EMBD) {
        pulsar_die("routed expert tensor layout is unexpected");
    }

    const uint32_t total_pairs = n_tok * PULSAR_N_EXPERT_USED;
    uint32_t counts[PULSAR_MAX_EXPERT + 1] = {0};
    uint32_t cursor[PULSAR_MAX_EXPERT] = {0};
    uint32_t active_expert[PULSAR_MAX_EXPERT];
    uint32_t n_active = 0;

    int *selected = (int *)xmalloc((size_t)total_pairs * sizeof(selected[0]));
    float *pair_weight = (float *)xmalloc((size_t)total_pairs * sizeof(pair_weight[0]));
    pulsar_expert_pair *pairs = (pulsar_expert_pair *)xmalloc((size_t)total_pairs * sizeof(pairs[0]));

    const uint64_t xq_blocks = expert_in_dim / QK_K;
    block_q8_K *xq = (block_q8_K *)xmalloc((size_t)n_tok * xq_blocks * sizeof(xq[0]));
    for (uint32_t t = 0; t < n_tok; t++) {
        pulsar_quantize_row_q8_K(norm + (uint64_t)t * expert_in_dim,
                              xq + (uint64_t)t * xq_blocks,
                              (int64_t)expert_in_dim);

        int sel[PULSAR_MAX_EXPERT_USED];
        float weights[PULSAR_MAX_EXPERT_USED];
        if (layer->ffn_gate_tid2eid) {
            layer_hash_selected_experts(sel, model, layer, token_ids[t]);
            layer_hash_router_weights_one(weights, model, layer, norm + (uint64_t)t * expert_in_dim, sel);
        } else {
            layer_topk_selected_experts(sel, weights, model, layer, norm + (uint64_t)t * expert_in_dim);
        }

        for (uint32_t slot = 0; slot < PULSAR_N_EXPERT_USED; slot++) {
            const uint32_t pair_id = t * PULSAR_N_EXPERT_USED + slot;
            selected[pair_id] = sel[slot];
            pair_weight[pair_id] = weights[slot];
            pairs[pair_id] = (pulsar_expert_pair){ .token = t, .slot = slot };
            if (sel[slot] < 0 || (uint32_t)sel[slot] >= PULSAR_N_EXPERT) pulsar_die("selected expert is outside range");
            counts[(uint32_t)sel[slot] + 1]++;
        }
    }

    for (uint32_t e = 0; e < PULSAR_N_EXPERT; e++) {
        counts[e + 1] += counts[e];
        cursor[e] = counts[e];
        if (counts[e + 1] != counts[e]) active_expert[n_active++] = e;
    }

    uint32_t *pair_ids = (uint32_t *)xmalloc((size_t)total_pairs * sizeof(pair_ids[0]));
    for (uint32_t p = 0; p < total_pairs; p++) {
        const uint32_t e = (uint32_t)selected[p];
        pair_ids[cursor[e]++] = p;
    }

    float *mid = (float *)xmalloc((size_t)total_pairs * expert_out_dim * sizeof(mid[0]));

    const uint32_t gate_type = layer->ffn_gate_exps->type;

    /* Build mid vectors: dispatch based on gate/up tensor type. */
    if (gate_type == PULSAR_TENSOR_IQ2_XXS) {
        matvec_iq2_xxs_batch_mid_ctx mid_ctx = {
            .mid = mid,
            .gate_base = {},   /* filled per active expert below */
            .up_base = {},
            .xq = xq,
            .pairs = pairs,
            .pair_ids = pair_ids,
            .expert_offset = counts,
            .active_expert = active_expert,
            .pair_weight = pair_weight,
            .clamp = clamp,
            .in_dim = expert_in_dim,
            .out_dim = expert_out_dim,
            .gate_row_bytes = {},   /* filled per active expert below */
            .up_row_bytes = {},
            .xq_blocks = xq_blocks,
        };

        for (uint32_t ai = 0; ai < n_active; ai++) {
            const uint32_t e = active_expert[ai];
            uint64_t gate_in_dim, gate_out_dim;
            uint64_t up_in_dim, up_out_dim;
            mid_ctx.gate_base[e] = tensor_expert_bytes(model, layer->ffn_gate_exps, e,
                                                       &gate_in_dim, &gate_out_dim, &mid_ctx.gate_row_bytes[e]);
            mid_ctx.up_base[e] = tensor_expert_bytes(model, layer->ffn_up_exps, e,
                                                     &up_in_dim, &up_out_dim, &mid_ctx.up_row_bytes[e]);
            if (gate_in_dim != expert_in_dim || up_in_dim != expert_in_dim ||
                gate_out_dim != expert_out_dim || up_out_dim != expert_out_dim) {
                pulsar_die("batch expert tensor layout mismatch");
            }
        }

        pulsar_parallel_for((uint64_t)n_active * expert_out_dim, matvec_iq2_xxs_batch_mid_worker, &mid_ctx);
    } else {
        pulsar_die("unsupported gate/up expert tensor type for batch");
    }

    const uint64_t midq_blocks = down_in_dim / QK_K;
    block_q8_K *midq = (block_q8_K *)xmalloc((size_t)total_pairs * midq_blocks * sizeof(midq[0]));
    quantize_mid_pairs_ctx quant_ctx = {
        .mid = mid,
        .midq = midq,
        .down_in_dim = down_in_dim,
        .down_blocks = midq_blocks,
    };
    pulsar_parallel_for(total_pairs, quantize_mid_pairs_worker, &quant_ctx);
    free(mid);

    /* Down projection: dispatch based on down tensor type. */
    const uint32_t down_type = layer->ffn_down_exps->type;

    if (down_type == PULSAR_TENSOR_Q2_K) {
        matvec_q2_k_batch_accum_rows_ctx down_ctx = {
            .moe = moe,
            .base = {},   /* filled per active expert below */
            .midq = midq,
            .pairs = pairs,
            .pair_ids = pair_ids,
            .expert_offset = counts,
            .active_expert = active_expert,
            .n_active = n_active,
            .n_tok = n_tok,
            .in_dim = down_in_dim,
            .out_dim = down_out_dim,
            .row_bytes = {},   /* filled per active expert below */
            .midq_blocks = midq_blocks,
        };

        for (uint32_t ai = 0; ai < n_active; ai++) {
            const uint32_t e = active_expert[ai];
            uint64_t in_dim, out_dim;
            down_ctx.base[e] = tensor_expert_bytes(model, layer->ffn_down_exps, e,
                                                   &in_dim, &out_dim, &down_ctx.row_bytes[e]);
            if (in_dim != down_in_dim || out_dim != down_out_dim) {
                pulsar_die("batch expert tensor layout mismatch");
            }
        }

        pulsar_parallel_for(down_out_dim, matvec_q2_k_batch_accum_rows_worker, &down_ctx);
    } else {
        pulsar_die("unsupported down expert tensor type for batch");
    }

    free(midq);
    free(pair_ids);
    free(xq);
    free(pairs);
    free(pair_weight);
    free(selected);

    (void)il;
}



void print_vec_stats(const char *name, const float *x, uint64_t n);



/* Full FFN sublayer for one token: HC pre, RMSNorm, routed MoE, shared expert,
 * sum, and HC post. */
void layer_ffn_one(
        float             * out_hc,
        const pulsar_model   * model,
        const pulsar_layer_weights * layer,
        const float       * inp_hc,
        uint32_t            il,
        int                 token,
        const float       * steering_dirs,
        float               steering_scale,
        bool                trace) {
    const uint32_t n_hc = PULSAR_N_HC;
    static int profile_env = -1;
    const bool profile = gpu_graph_env_flag("DS4_DECODE_PROFILE_DETAIL", &profile_env);
    const double t_start = profile ? now_sec() : 0.0;
    double t_hc = 0.0;
    double t_norm = 0.0;
    double t_routed = 0.0;
    double t_shared = 0.0;
    double t_post = 0.0;
    float *ffn_cur = (float *)xmalloc((size_t)PULSAR_N_EMBD * sizeof(ffn_cur[0]));
    float *norm = (float *)xmalloc((size_t)PULSAR_N_EMBD * sizeof(norm[0]));
    float *moe = (float *)xmalloc((size_t)PULSAR_N_EMBD * sizeof(moe[0]));
    float *shared = (float *)xmalloc((size_t)PULSAR_N_EMBD * sizeof(shared[0]));
    float *ffn_out = (float *)xmalloc((size_t)PULSAR_N_EMBD * sizeof(ffn_out[0]));
    float post[4];
    float comb[16];

    double t0 = profile ? now_sec() : 0.0;
    hc_pre_from_state_one(model,
                          layer->hc_ffn_fn,
                          layer->hc_ffn_scale,
                          layer->hc_ffn_base,
                          inp_hc, ffn_cur, post, comb);
    if (profile) t_hc = now_sec() - t0;
    if (trace) {
        char name[64];
        snprintf(name, sizeof(name), "blk.%u ffn_cur", il);
        print_vec_stats(name, ffn_cur, PULSAR_N_EMBD);
    }

    t0 = profile ? now_sec() : 0.0;
    const float *ffn_norm = (const float *)tensor_data(model, layer->ffn_norm);
    rms_norm_weight(norm, ffn_cur, ffn_norm, PULSAR_N_EMBD, PULSAR_RMS_EPS);
    if (profile) t_norm = now_sec() - t0;
    if (trace) {
        char name[64];
        snprintf(name, sizeof(name), "blk.%u ffn_norm", il);
        print_vec_stats(name, norm, PULSAR_N_EMBD);
    }

    t0 = profile ? now_sec() : 0.0;
    layer_routed_moe_one(moe, model, layer, norm, il, token, PULSAR_SWIGLU_CLAMP_EXP, trace);
    if (profile) t_routed = now_sec() - t0;
    if (trace) {
        char name[64];
        snprintf(name, sizeof(name), "blk.%u routed_moe", il);
        print_vec_stats(name, moe, PULSAR_N_EMBD);
    }
    t0 = profile ? now_sec() : 0.0;
    layer_shared_ffn_one(shared, model, layer, norm);
    if (profile) t_shared = now_sec() - t0;
    if (trace) {
        char name[64];
        snprintf(name, sizeof(name), "blk.%u shared_ffn", il);
        print_vec_stats(name, shared, PULSAR_N_EMBD);
    }

    t0 = profile ? now_sec() : 0.0;
    for (uint32_t i = 0; i < PULSAR_N_EMBD; i++) {
        ffn_out[i] = moe[i] + shared[i];
    }
    cpu_directional_steering_project_rows(ffn_out, steering_dirs, il, 1, steering_scale);
    if (trace) {
        char name[64];
        snprintf(name, sizeof(name), "blk.%u ffn_out", il);
        print_vec_stats(name, ffn_out, PULSAR_N_EMBD);
    }

    hc_post_one(out_hc, ffn_out, inp_hc, post, comb, PULSAR_N_EMBD, n_hc);
    if (profile) t_post = now_sec() - t0;
    if (trace) {
        char name[64];
        snprintf(name, sizeof(name), "blk.%u ffn_post_hc", il);
        print_vec_stats(name, out_hc, (uint64_t)n_hc * PULSAR_N_EMBD);
    }

    if (profile) {
        fprintf(stderr,
                "pulsar: decode detail layer %u ffn hc=%.3f norm=%.3f routed=%.3f shared=%.3f post=%.3f total=%.3f ms\n",
                il,
                t_hc * 1000.0,
                t_norm * 1000.0,
                t_routed * 1000.0,
                t_shared * 1000.0,
                t_post * 1000.0,
                (now_sec() - t_start) * 1000.0);
    }

    free(ffn_out);
    free(shared);
    free(moe);
    free(norm);
    free(ffn_cur);
}



/* Allocation-free decode FFN using the persistent CPU scratch buffers. */



void layer_ffn_batch(
        float             * out_hc,
        const pulsar_model   * model,
        const pulsar_layer_weights * layer,
        const float       * inp_hc,
        const int         * token_ids,
        uint32_t            n_tok,
        uint32_t            il,
        const float       * steering_dirs,
        float               steering_scale) {
    if (n_tok == 0) return;
    const uint32_t n_hc = PULSAR_N_HC;
    const uint64_t hc_dim = (uint64_t)n_hc * PULSAR_N_EMBD;
    float *ffn_cur = (float *)xmalloc((size_t)n_tok * PULSAR_N_EMBD * sizeof(ffn_cur[0]));
    float *norm = (float *)xmalloc((size_t)n_tok * PULSAR_N_EMBD * sizeof(norm[0]));
    float *moe = (float *)xmalloc((size_t)n_tok * PULSAR_N_EMBD * sizeof(moe[0]));
    float *shared = (float *)xmalloc((size_t)n_tok * PULSAR_N_EMBD * sizeof(shared[0]));
    float *post = (float *)xmalloc((size_t)n_tok * n_hc * sizeof(post[0]));
    float *comb = (float *)xmalloc((size_t)n_tok * n_hc * n_hc * sizeof(comb[0]));
    const float *ffn_norm = (const float *)tensor_data(model, layer->ffn_norm);

    for (uint32_t t = 0; t < n_tok; t++) {
        hc_pre_from_state_one(model,
                              layer->hc_ffn_fn,
                              layer->hc_ffn_scale,
                              layer->hc_ffn_base,
                              inp_hc + (uint64_t)t * hc_dim,
                              ffn_cur + (uint64_t)t * PULSAR_N_EMBD,
                              post + (uint64_t)t * n_hc,
                              comb + (uint64_t)t * n_hc * n_hc);
        rms_norm_weight(norm + (uint64_t)t * PULSAR_N_EMBD,
                        ffn_cur + (uint64_t)t * PULSAR_N_EMBD,
                        ffn_norm,
                        PULSAR_N_EMBD,
                        PULSAR_RMS_EPS);
    }

    layer_routed_moe_batch(moe, model, layer, norm, token_ids, n_tok, il, PULSAR_SWIGLU_CLAMP_EXP);
    layer_shared_ffn_batch(shared, model, layer, norm, n_tok);

    if (cpu_directional_steering_enabled(steering_dirs, steering_scale)) {
        float *ffn_out = (float *)xmalloc((size_t)n_tok * PULSAR_N_EMBD * sizeof(ffn_out[0]));
        for (uint64_t i = 0; i < (uint64_t)n_tok * PULSAR_N_EMBD; i++) {
            ffn_out[i] = moe[i] + shared[i];
        }
        cpu_directional_steering_project_rows(ffn_out, steering_dirs, il, n_tok, steering_scale);
        hc_post_batch(out_hc,
                      ffn_out,
                      inp_hc,
                      post,
                      comb,
                      n_tok,
                      PULSAR_N_EMBD,
                      n_hc);
        free(ffn_out);
    } else {
        hc_post_sum_batch(out_hc,
                          moe,
                          shared,
                          inp_hc,
                          post,
                          comb,
                          n_tok,
                          PULSAR_N_EMBD,
                          n_hc);
    }

    free(comb);
    free(post);
    free(shared);
    free(moe);
    free(norm);
    free(ffn_cur);
}



static void routed_moe_tokens_worker(void *vctx, uint64_t t0, uint64_t t1) {
    routed_moe_tokens_ctx *ctx = static_cast<routed_moe_tokens_ctx *>(vctx);
    float *routed_mid = (float *)xmalloc((size_t)PULSAR_N_EXPERT_USED * PULSAR_N_FF_EXP * sizeof(routed_mid[0]));
    block_q8_K *routed_xq = (block_q8_K *)xmalloc((size_t)(ctx->expert_in_dim / QK_K) * sizeof(routed_xq[0]));
    block_q8_K *routed_midq = (block_q8_K *)xmalloc((size_t)PULSAR_N_EXPERT_USED * (ctx->down_in_dim / QK_K) * sizeof(routed_midq[0]));

    for (uint64_t t = t0; t < t1; t++) {
        layer_routed_moe_one_prealloc(ctx->moe + t * PULSAR_N_EMBD,
                                      ctx->model,
                                      ctx->layer,
                                      ctx->norm + t * PULSAR_N_EMBD,
                                      ctx->il,
                                      ctx->token_ids[t],
                                      PULSAR_SWIGLU_CLAMP_EXP,
                                      routed_mid,
                                      routed_xq,
                                      routed_midq);
    }

    free(routed_midq);
    free(routed_xq);
    free(routed_mid);
}



static void layer_routed_moe_tokens_parallel(
        float             * moe,
        const pulsar_model   * model,
        const pulsar_layer_weights * layer,
        const float       * norm,
        const int         * token_ids,
        uint32_t            n_tok,
        uint32_t            il) {
    routed_moe_tokens_ctx ctx = {
        .moe = moe,
        .model = model,
        .layer = layer,
        .norm = norm,
        .token_ids = token_ids,
        .expert_in_dim = layer->ffn_gate_exps->dim[0],
        .down_in_dim = layer->ffn_down_exps->dim[0],
        .il = il,
    };
    pulsar_parallel_for_min_rows(n_tok, routed_moe_tokens_worker, &ctx, 1);
}



/* Default prefill FFN path.  HC and shared expert are batched, while routed
 * experts can run either token-parallel or expert-grouped depending on size. */
void layer_ffn_shared_batch(
        float             * out_hc,
        const pulsar_model   * model,
        const pulsar_layer_weights * layer,
        const float       * inp_hc,
        const int         * token_ids,
        uint32_t            n_tok,
        uint32_t            il,
        const float       * steering_dirs,
        float               steering_scale) {
    static int profile_env = -1;
    const bool profile = gpu_graph_env_flag("DS4_PREFILL_PROFILE_DETAIL", &profile_env);
    const double t_start = profile ? now_sec() : 0.0;
    double t_hc_norm = 0.0;
    double t_routed = 0.0;
    double t_shared = 0.0;
    double t_post = 0.0;
    const uint32_t n_hc = PULSAR_N_HC;
    float *ffn_cur = (float *)xmalloc((size_t)n_tok * PULSAR_N_EMBD * sizeof(ffn_cur[0]));
    float *norm = (float *)xmalloc((size_t)n_tok * PULSAR_N_EMBD * sizeof(norm[0]));
    float *moe = (float *)xmalloc((size_t)n_tok * PULSAR_N_EMBD * sizeof(moe[0]));
    float *shared = (float *)xmalloc((size_t)n_tok * PULSAR_N_EMBD * sizeof(shared[0]));
    float *post = (float *)xmalloc((size_t)n_tok * n_hc * sizeof(post[0]));
    float *comb = (float *)xmalloc((size_t)n_tok * n_hc * n_hc * sizeof(comb[0]));
    const uint64_t expert_in_dim = layer->ffn_gate_exps->dim[0];
    const uint64_t down_in_dim = layer->ffn_down_exps->dim[0];
    static int no_routed_parallel_env = -1;
    const bool routed_token_parallel =
        !gpu_graph_env_flag("DS4_NO_ROUTED_TOKEN_PARALLEL", &no_routed_parallel_env) &&
        n_tok >= 64;
    float *routed_mid = routed_token_parallel ? NULL : (float *)xmalloc((size_t)PULSAR_N_EXPERT_USED * PULSAR_N_FF_EXP * sizeof(routed_mid[0]));
    block_q8_K *routed_xq = routed_token_parallel ? NULL : (block_q8_K *)xmalloc((size_t)(expert_in_dim / QK_K) * sizeof(routed_xq[0]));
    block_q8_K *routed_midq = routed_token_parallel ? NULL : (block_q8_K *)xmalloc((size_t)PULSAR_N_EXPERT_USED * (down_in_dim / QK_K) * sizeof(routed_midq[0]));

    double t0 = profile ? now_sec() : 0.0;
    hc_pre_norm_batch(model,
                      layer->hc_ffn_fn,
                      layer->hc_ffn_scale,
                      layer->hc_ffn_base,
                      layer->ffn_norm,
                      inp_hc,
                      NULL,
                      ffn_cur,
                      norm,
                      post,
                      comb,
                      n_tok);
    if (profile) t_hc_norm = now_sec() - t0;

    t0 = profile ? now_sec() : 0.0;
    if (routed_token_parallel) {
        layer_routed_moe_tokens_parallel(moe, model, layer, norm, token_ids, n_tok, il);
    } else {
        for (uint32_t t = 0; t < n_tok; t++) {
            layer_routed_moe_one_prealloc(moe + (uint64_t)t * PULSAR_N_EMBD,
                                          model,
                                          layer,
                                          norm + (uint64_t)t * PULSAR_N_EMBD,
                                          il,
                                          token_ids[t],
                                          PULSAR_SWIGLU_CLAMP_EXP,
                                          routed_mid,
                                          routed_xq,
                                          routed_midq);
        }
    }
    if (profile) t_routed = now_sec() - t0;

    t0 = profile ? now_sec() : 0.0;
    layer_shared_ffn_batch(shared, model, layer, norm, n_tok);
    if (profile) t_shared = now_sec() - t0;

    t0 = profile ? now_sec() : 0.0;
    if (cpu_directional_steering_enabled(steering_dirs, steering_scale)) {
        float *ffn_out = (float *)xmalloc((size_t)n_tok * PULSAR_N_EMBD * sizeof(ffn_out[0]));
        for (uint64_t i = 0; i < (uint64_t)n_tok * PULSAR_N_EMBD; i++) {
            ffn_out[i] = moe[i] + shared[i];
        }
        cpu_directional_steering_project_rows(ffn_out, steering_dirs, il, n_tok, steering_scale);
        hc_post_batch(out_hc,
                      ffn_out,
                      inp_hc,
                      post,
                      comb,
                      n_tok,
                      PULSAR_N_EMBD,
                      n_hc);
        free(ffn_out);
    } else {
        hc_post_sum_batch(out_hc,
                          moe,
                          shared,
                          inp_hc,
                          post,
                          comb,
                          n_tok,
                          PULSAR_N_EMBD,
                          n_hc);
    }
    if (profile) t_post = now_sec() - t0;

    if (profile) {
        fprintf(stderr,
                "pulsar: prefill detail layer %u ffn hc_norm=%.3f routed=%.3f shared=%.3f post=%.3f total=%.3f\n",
                il, t_hc_norm, t_routed, t_shared, t_post, now_sec() - t_start);
    }

    free(comb);
    free(post);
    free(routed_midq);
    free(routed_xq);
    free(routed_mid);
    free(shared);
    free(moe);
    free(norm);
    free(ffn_cur);
}



static void layer_ffn_tokens_worker(void *vctx, uint64_t t0, uint64_t t1) {
    layer_ffn_tokens_ctx *ctx = static_cast<layer_ffn_tokens_ctx *>(vctx);
    for (uint64_t t = t0; t < t1; t++) {
        layer_ffn_one(ctx->out_hc + t * ctx->hc_dim,
                      ctx->model,
                      ctx->layer,
                      ctx->inp_hc + t * ctx->hc_dim,
                      ctx->il,
                      ctx->token_ids[t],
                      ctx->steering_dirs,
                      ctx->steering_scale,
                      false);
    }
}



void layer_ffn_tokens_parallel(
        float             * out_hc,
        const pulsar_model   * model,
        const pulsar_layer_weights * layer,
        const float       * inp_hc,
        const int         * token_ids,
        uint32_t            n_tok,
        uint32_t            il,
        const float       * steering_dirs,
        float               steering_scale) {
    layer_ffn_tokens_ctx ctx = {
        .out_hc = out_hc,
        .model = model,
        .layer = layer,
        .inp_hc = inp_hc,
        .token_ids = token_ids,
        .steering_dirs = steering_dirs,
        .steering_scale = steering_scale,
        .hc_dim = (uint64_t)PULSAR_N_HC * PULSAR_N_EMBD,
        .il = il,
    };
    pulsar_parallel_for(n_tok, layer_ffn_tokens_worker, &ctx);
}



void output_logits_one(
        float             * logits,
        const pulsar_model   * model,
        const pulsar_weights * weights,
        const float       * inp_hc);

