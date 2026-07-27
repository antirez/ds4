#include "pulsar_engine_internal.h"



static float required_f32(const pulsar_model *m, const char *key) {
    float v = 0.0f;
    if (!model_get_f32_compat(m, key, &v)) {
        fprintf(stderr, "pulsar: required metadata key is missing: %s\n", key);
        exit(1);
    }
    return v;
}



static bool required_bool(const pulsar_model *m, const char *key) {
    bool v = false;
    if (!model_get_bool(m, key, &v)) {
        fprintf(stderr, "pulsar: required metadata key is missing: %s\n", key);
        exit(1);
    }
    return v;
}



static pulsar_tensor *required_tensor(const pulsar_model *m, const char *name) {
    pulsar_tensor *t = model_find_tensor(m, name);
    if (!t) {
        fprintf(stderr, "pulsar: required tensor is missing: %s\n", name);
        exit(1);
    }
    return t;
}



static pulsar_tensor *tensor_by_namef(const pulsar_model *m, const char *fmt, uint32_t layer) {
    char name[128];
    int n = snprintf(name, sizeof(name), fmt, layer);
    if (n < 0 || (size_t)n >= sizeof(name)) pulsar_die("tensor name is too long");
    return model_find_tensor(m, name);
}



static pulsar_tensor *required_tensorf(const pulsar_model *m, const char *fmt, uint32_t layer) {
    char name[128];
    int n = snprintf(name, sizeof(name), fmt, layer);
    if (n < 0 || (size_t)n >= sizeof(name)) pulsar_die("tensor name is too long");
    return required_tensor(m, name);
}



static void tensor_expect_layout(
        const pulsar_tensor *t,
        uint32_t          type,
        uint32_t          ndim,
        uint64_t          d0,
        uint64_t          d1,
        uint64_t          d2) {
    if (!t) pulsar_die("internal error: missing tensor while validating layout");
    if (t->type != type) {
        fprintf(stderr,
                "pulsar: tensor %.*s has type %s, expected %s\n",
                (int)t->name.len,
                t->name.ptr,
                tensor_type_name(t->type),
                tensor_type_name(type));
        exit(1);
    }
    if (t->ndim != ndim) {
        fprintf(stderr,
                "pulsar: tensor %.*s has %u dimensions, expected %u\n",
                (int)t->name.len,
                t->name.ptr,
                t->ndim,
                ndim);
        exit(1);
    }

    const uint64_t want[3] = { d0, d1, d2 };
    for (uint32_t i = 0; i < ndim; i++) {
        if (t->dim[i] == want[i]) continue;
        fprintf(stderr,
                "pulsar: tensor %.*s has dim[%u]=%" PRIu64 ", expected %" PRIu64 "\n",
                (int)t->name.len,
                t->name.ptr,
                i,
                t->dim[i],
                want[i]);
        exit(1);
    }
}



static void tensor_expect_optional(
        const pulsar_tensor *t,
        uint32_t          type,
        uint32_t          ndim,
        uint64_t          d0,
        uint64_t          d1,
        uint64_t          d2) {
    if (t) tensor_expect_layout(t, type, ndim, d0, d1, d2);
}
/* MXFP8 workhorse weight: either the classic interleaved type (FP8_E4M3, 38) or
 * its pre-stored device layout (MXFP8_LT, 41). Both share dims and byte
 * accounting; the FP8 matmul resolver dispatches on the registered offset. */
static void tensor_expect_mxfp8(
        const pulsar_tensor *t,
        uint32_t          ndim,
        uint64_t          d0,
        uint64_t          d1,
        uint64_t          d2) {
    if (!t) pulsar_die("internal error: missing tensor while validating layout");
    if (t->type == PULSAR_TENSOR_FP8_E4M3)
        tensor_expect_layout(t, PULSAR_TENSOR_FP8_E4M3, ndim, d0, d1, d2);
    else if (t->type == PULSAR_TENSOR_MXFP8_LT)
        tensor_expect_layout(t, PULSAR_TENSOR_MXFP8_LT, ndim, d0, d1, d2);
    else
        pulsar_die("tensor has unsupported weight type; expected FP8_E4M3 or MXFP8_LT");
}
static void tensor_expect_plain_or_mxfp8(
        const pulsar_tensor *t,
        uint32_t          ndim,
        uint64_t          d0,
        uint64_t          d1,
        uint64_t          d2) {
    if (!t) pulsar_die("internal error: missing tensor while validating layout");
    if (t->type == PULSAR_TENSOR_F16)
        tensor_expect_layout(t, PULSAR_TENSOR_F16, ndim, d0, d1, d2);
    else if (t->type == PULSAR_TENSOR_FP8_E4M3)
        tensor_expect_layout(t, PULSAR_TENSOR_FP8_E4M3, ndim, d0, d1, d2);
    else
        /* Deliberately does NOT accept MXFP8_LT: the weights routed here
         * (hc_attn_fn/hc_ffn_fn, ffn_gate_inp, attn/indexer compressors,
         * output_hc_fn) take the PLAIN matmul path (gpu_graph_matmul_plain_tensor),
         * which has no type-41 branch. Pre-storing one as MXFP8_LT would decode to
         * garbage, so a future FP8 variant of these must fail FAST here at load,
         * not silently pass. Only the tensor_expect_mxfp8 workhorse set (routed
         * through cuda_fp8_mx_weight) may be MXFP8_LT. */
        pulsar_die("tensor has unsupported weight type; expected F16 or FP8_E4M3");
}




static void tensor_expect_plain_layout(
        const pulsar_tensor *t,
        uint32_t          ndim,
        uint64_t          d0,
        uint64_t          d1,
        uint64_t          d2) {
    if (!t) pulsar_die("internal error: missing tensor while validating layout");
    if (t->type != PULSAR_TENSOR_F16 && t->type != PULSAR_TENSOR_F32) {
        fprintf(stderr,
                "pulsar: tensor %.*s has type %s, expected F16 or F32\n",
                (int)t->name.len,
                t->name.ptr,
                tensor_type_name(t->type));
        exit(1);
    }
    tensor_expect_layout(t, t->type, ndim, d0, d1, d2);
}



static bool tensor_is_routed_expert_type(uint32_t type) {
    return type == PULSAR_TENSOR_IQ2_XXS ||
           type == PULSAR_TENSOR_Q2_K ||
           type == PULSAR_TENSOR_FP4_E2M1 ||
           type == PULSAR_TENSOR_CUTLASS_MXFP4;
}



static PULSAR_MAYBE_UNUSED uint64_t routed_expert_block_bytes(uint32_t type) {
    switch (type) {
    case PULSAR_TENSOR_IQ2_XXS: return sizeof(block_iq2_xxs);
    case PULSAR_TENSOR_Q2_K:    return sizeof(block_q2_K);
    /* MXFP4: 17 bytes / 32 vals = [1 E8M0 scale][16 bytes = 32x E2M1]. Per-QK_K
     * (256 vals) = 8 sub-blocks * 17 = 136 bytes, matching the other per-QK_K sizes. */
    case PULSAR_TENSOR_FP4_E2M1: return (QK_K / 32) * 17;
    default:                 pulsar_die("unsupported routed expert tensor type");
    }
    return 0;
}



PULSAR_MAYBE_UNUSED uint64_t routed_expert_row_bytes(const pulsar_tensor *t) {
    if ((t->dim[0] % QK_K) != 0) pulsar_die("routed expert row is not QK_K aligned");
    return (t->dim[0] / QK_K) * routed_expert_block_bytes(t->type);
}



/* Computes (gate_expert_bytes, gate_row_bytes, down_expert_bytes, down_row_bytes)
 * for any supported routed-expert quant combo, centralizing the
 * dispatch-site pattern `row_bytes = routed_expert_row_bytes(t); expert_bytes =
 * t->dim[1] * row_bytes` that's repeated across gpu_prefill.c/gpu_decode.c.
 *
 * For CUTLASS_MXFP4 (type 40) "row_bytes" has no ordinary per-row meaning --
 * the tensor is expert-major ColumnMajor+swizzle with no per-row byte stride
 * at all. It instead carries the data/SF split point within each expert's
 * block: the SF blob starts *row_bytes bytes into that expert's slice, and
 * *expert_bytes is the full [data + SF] stride to the next expert. Callers
 * that dispatch on gate->type == PULSAR_TENSOR_CUTLASS_MXFP4 must read it that
 * way; only the CUTLASS MoE path does. */
bool routed_expert_gate_down_layout(
        const pulsar_tensor *gate,
        const pulsar_tensor *down,
        uint64_t         *gate_expert_bytes,
        uint64_t         *gate_row_bytes,
        uint64_t         *down_expert_bytes,
        uint64_t         *down_row_bytes) {
    /* NOTE: gate and down are NOT always the same type -- gate/up and down
     * formats pair freely per layer by design (see
     * tensor_expect_routed_expert_combo). Each side's layout is computed
     * independently so MIXED layers (cutlass_mxfp4 on one side, iq2/q2k on the
     * other) resolve correctly: the CUTLASS_MXFP4 side yields stride/split-point,
     * the dp4a side yields ordinary expert/row byte counts. */
    if (!gate || !down) return false;

    if (gate->type == PULSAR_TENSOR_CUTLASS_MXFP4) {
        uint64_t gate_sf, gate_stride;
        cutlass_mxfp4_expert_layout(gate->dim[0], gate->dim[1],
                                     gate_row_bytes, &gate_sf, &gate_stride);
        *gate_expert_bytes = gate_stride;
    } else {
        *gate_row_bytes = routed_expert_row_bytes(gate);
        if (*gate_row_bytes == 0 || gate->dim[1] > UINT64_MAX / *gate_row_bytes) return false;
        *gate_expert_bytes = gate->dim[1] * *gate_row_bytes;
    }

    if (down->type == PULSAR_TENSOR_CUTLASS_MXFP4) {
        uint64_t down_sf, down_stride;
        cutlass_mxfp4_expert_layout(down->dim[0], down->dim[1],
                                     down_row_bytes, &down_sf, &down_stride);
        *down_expert_bytes = down_stride;
    } else {
        *down_row_bytes = routed_expert_row_bytes(down);
        if (*down_row_bytes == 0 || down->dim[1] > UINT64_MAX / *down_row_bytes) return false;
        *down_expert_bytes = down->dim[1] * *down_row_bytes;
    }
    return true;
}



/* The CUDA routed-MoE dispatcher selects kernels per role and per layer:
 * gate/up (always a matching pair -- the fused gate+up kernels assume one
 * format) in {IQ2_XXS, Q2_K, MXFP4}, down in {IQ2_XXS, Q2_K, MXFP4}, in any
 * pairing, and the combo may differ layer to layer (prisma per-layer
 * allocation). CUTLASS_MXFP4 is the exception: the grouped tensor-core GEMM
 * path runs the whole expert FFN in one dispatch, so cutlass gate/up
 * requires cutlass down (and vice versa). Reject anything else at load with
 * one clear error instead of a silent kernel-dispatch failure at the first
 * MoE layer. */
static void tensor_expect_routed_expert_combo(
        const pulsar_tensor *gate,
        const pulsar_tensor *up,
        const pulsar_tensor *down) {
    /* gate/up must match (the fused gate+up kernels assume one format). Each of
     * gate/up and down may independently be a dp4a quant (iq2_xxs/q2_k/mxfp4) OR
     * CUTLASS_MXFP4 (type 40) -- the GPU MoE path handles all-cutlass (uniform,
     * grouped/gemv), all-dp4a (heterogeneous), AND the two MIXED shapes
     * (cutlass gate/up + iq2/q2k down; iq2/q2k gate/up + cutlass down) via
     * per-projection dispatch. The ONE combo the GPU can't compose is CUTLASS
     * mixed with the legacy 17-byte FP4_E2M1 (type 39) on the other side -- that
     * dp4a format can't be read by CUTLASS and vice-versa; reject it fail-closed
     * (it never occurs after the type-40 unification repack, which converts every
     * type-39 mxfp4 to type-40). */
    const bool gate_up_pair = gate->type == up->type;
    const bool gate_cut = gate->type == PULSAR_TENSOR_CUTLASS_MXFP4;
    const bool down_cut = down->type == PULSAR_TENSOR_CUTLASS_MXFP4;
    const bool gate_dp4a = gate->type == PULSAR_TENSOR_IQ2_XXS ||
                           gate->type == PULSAR_TENSOR_Q2_K ||
                           gate->type == PULSAR_TENSOR_FP4_E2M1;
    const bool down_dp4a = down->type == PULSAR_TENSOR_IQ2_XXS ||
                           down->type == PULSAR_TENSOR_Q2_K ||
                           down->type == PULSAR_TENSOR_FP4_E2M1;
    const bool gate_ok = gate_dp4a || gate_cut;
    const bool down_ok = down_dp4a || down_cut;
    /* the unhandled cross: CUTLASS on one side + legacy type-39 FP4 on the other */
    const bool bad_mix = (gate_cut && down->type == PULSAR_TENSOR_FP4_E2M1) ||
                         (down_cut && gate->type == PULSAR_TENSOR_FP4_E2M1);
    if (gate_up_pair && gate_ok && down_ok && !bad_mix) return;
    fprintf(stderr,
            "pulsar: unsupported routed expert quant combo at tensor %.*s: "
            "gate=%s up=%s down=%s; gate/up must match and be one of "
            "iq2_xxs/q2_k/mxfp4/cutlass_mxfp4, down one of the same; a CUTLASS_MXFP4 "
            "side may NOT pair with a legacy type-39 mxfp4 side; "
            "combos may differ per layer\n",
            (int)gate->name.len,
            gate->name.ptr,
            tensor_type_name(gate->type),
            tensor_type_name(up->type),
            tensor_type_name(down->type));
    exit(1);
}



static void tensor_expect_routed_expert(
        const pulsar_tensor *t,
        uint32_t          ndim,
        uint64_t          d0,
        uint64_t          d1,
        uint64_t          d2) {
    if (!t) pulsar_die("internal error: missing routed expert tensor while validating layout");
    if (!tensor_is_routed_expert_type(t->type)) {
        fprintf(stderr,
                "pulsar: tensor %.*s has type %u (%s), expected a routed expert quant type\n",
                (int)t->name.len,
                t->name.ptr,
                t->type,
                tensor_type_name(t->type));
        exit(1);
    }
    if (t->ndim != ndim) {
        fprintf(stderr,
                "pulsar: tensor %.*s has %u dimensions, expected %u\n",
                (int)t->name.len,
                t->name.ptr,
                t->ndim,
                ndim);
        exit(1);
    }

    const uint64_t want[3] = { d0, d1, d2 };
    for (uint32_t i = 0; i < ndim; i++) {
        if (t->dim[i] == want[i]) continue;
        fprintf(stderr,
                "pulsar: tensor %.*s has dim[%u]=%" PRIu64 ", expected %" PRIu64 "\n",
                (int)t->name.len,
                t->name.ptr,
                i,
                t->dim[i],
                want[i]);
        exit(1);
    }
}



bool weights_have_output_head(const pulsar_weights *w) {
    return w &&
           w->output_hc_base &&
           w->output_hc_fn &&
           w->output_hc_scale &&
           w->output_norm &&
           w->output;
}



static bool weights_have_partial_output_head(const pulsar_weights *w) {
    return w &&
           (w->output_hc_base ||
            w->output_hc_fn ||
            w->output_hc_scale ||
            w->output_norm ||
            w->output);
}



static bool weights_layer_has_required(const pulsar_layer_weights *l, uint32_t il) {
    if (!l) return false;
    if (!l->hc_attn_fn ||
        !l->hc_attn_scale ||
        !l->hc_attn_base ||
        !l->attn_norm ||
        !l->attn_q_a ||
        !l->attn_q_a_norm ||
        !l->attn_q_b ||
        !l->attn_kv ||
        !l->attn_kv_a_norm ||
        !l->attn_sinks ||
        !l->attn_output_a ||
        !l->attn_output_b ||
        !l->hc_ffn_fn ||
        !l->hc_ffn_scale ||
        !l->hc_ffn_base ||
        !l->ffn_norm ||
        !l->ffn_gate_inp ||
        !l->ffn_gate_exps ||
        !l->ffn_up_exps ||
        !l->ffn_down_exps ||
        !l->ffn_gate_shexp ||
        !l->ffn_up_shexp ||
        !l->ffn_down_shexp)
    {
        return false;
    }

    const uint32_t ratio = pulsar_layer_compress_ratio(il);
    if (ratio != 0 &&
        (!l->attn_compressor_ape ||
         !l->attn_compressor_kv ||
         !l->attn_compressor_gate ||
         !l->attn_compressor_norm))
    {
        return false;
    }
    if (ratio == 4 &&
        (!l->indexer_attn_q_b ||
         !l->indexer_proj ||
         !l->indexer_compressor_ape ||
         !l->indexer_compressor_kv ||
         !l->indexer_compressor_gate ||
         !l->indexer_compressor_norm))
    {
        return false;
    }
    if (il < PULSAR_N_HASH_LAYER && !l->ffn_gate_tid2eid) return false;
    return true;
}



const pulsar_layer_weights *weights_first_bound_layer(const pulsar_weights *w) {
    if (!w) return NULL;
    for (uint32_t il = 0; il < PULSAR_N_LAYER; il++) {
        if (weights_layer_has_required(&w->layer[il], il)) return &w->layer[il];
    }
    return NULL;
}



/* Verify every tensor type and dimension used by the specialized pipeline.
 * Token embedding and output head are validated when present. */
static void weights_validate_layout(
        const pulsar_weights *w,
        uint32_t           layer_start,
        uint32_t           layer_end,
        bool               require_token_embd,
        bool               require_output) {
    const uint64_t hc_dim = (uint64_t)PULSAR_N_EMBD * PULSAR_N_HC;
    const uint64_t hc_mix_dim = 2u * PULSAR_N_HC + (uint64_t)PULSAR_N_HC * PULSAR_N_HC;
    const uint64_t q_dim = (uint64_t)PULSAR_N_HEAD * PULSAR_N_HEAD_DIM;
    const uint64_t out_low_dim = (uint64_t)PULSAR_N_OUT_GROUP * PULSAR_N_LORA_O;

    if (!w) pulsar_die("internal error: missing weights while validating layout");
    if (layer_start >= PULSAR_N_LAYER) pulsar_die("invalid first layer in weight layout validation");
    if (layer_end == UINT32_MAX) layer_end = PULSAR_N_LAYER - 1u;
    if (layer_end >= PULSAR_N_LAYER || layer_end < layer_start) {
        pulsar_die("invalid layer range in weight layout validation");
    }

    if (require_token_embd && !w->token_embd) pulsar_die("required token embedding tensor is missing");
    if (w->token_embd) {
        tensor_expect_layout(w->token_embd, PULSAR_TENSOR_F16, 2, PULSAR_N_EMBD, PULSAR_N_VOCAB, 0);
    }

    const bool have_output = weights_have_output_head(w);
    if (require_output && !have_output) pulsar_die("required output head tensors are missing");
    if (weights_have_partial_output_head(w) && !have_output) pulsar_die("partial output head in GGUF");
    if (have_output) {
        tensor_expect_layout(w->output_hc_base,  PULSAR_TENSOR_F32,  1, PULSAR_N_HC, 0, 0);
        tensor_expect_plain_or_mxfp8(w->output_hc_fn, 2, hc_dim, PULSAR_N_HC, 0);
        tensor_expect_layout(w->output_hc_scale, PULSAR_TENSOR_F32,  1, 1, 0, 0);
        tensor_expect_layout(w->output_norm,     PULSAR_TENSOR_F32,  1, PULSAR_N_EMBD, 0, 0);
        /* Output head is BF16 (kept lossless; the engine has a dedicated BF16
         * matmul) or MXFP8 (routed to the FP8 matmul). */
        if (w->output->type == PULSAR_TENSOR_BF16)
            tensor_expect_layout(w->output,      PULSAR_TENSOR_BF16, 2, PULSAR_N_EMBD, PULSAR_N_VOCAB, 0);
        else
            tensor_expect_mxfp8(w->output,       2, PULSAR_N_EMBD, PULSAR_N_VOCAB, 0);
    }

    for (uint32_t il = layer_start; il <= layer_end; il++) {
        const pulsar_layer_weights *l = &w->layer[il];
        const uint32_t ratio = pulsar_layer_compress_ratio(il);
        if (!weights_layer_has_required(l, il)) {
            fprintf(stderr, "pulsar: required tensors for layer %u are missing\n", il);
            exit(1);
        }

        tensor_expect_plain_or_mxfp8(l->hc_attn_fn, 2, hc_dim, hc_mix_dim, 0);
        tensor_expect_layout(l->hc_attn_scale,  PULSAR_TENSOR_F32,  1, 3, 0, 0);
        tensor_expect_layout(l->hc_attn_base,   PULSAR_TENSOR_F32,  1, hc_mix_dim, 0, 0);
        tensor_expect_layout(l->attn_norm,      PULSAR_TENSOR_F32,  1, PULSAR_N_EMBD, 0, 0);
        tensor_expect_mxfp8(l->attn_q_a,        2, PULSAR_N_EMBD, PULSAR_N_LORA_Q, 0);
        tensor_expect_layout(l->attn_q_a_norm,  PULSAR_TENSOR_F32,  1, PULSAR_N_LORA_Q, 0, 0);
        tensor_expect_mxfp8(l->attn_q_b,        2, PULSAR_N_LORA_Q, q_dim, 0);
        tensor_expect_mxfp8(l->attn_kv,         2, PULSAR_N_EMBD, PULSAR_N_HEAD_DIM, 0);
        tensor_expect_layout(l->attn_kv_a_norm, PULSAR_TENSOR_F32,  1, PULSAR_N_HEAD_DIM, 0, 0);
        tensor_expect_layout(l->attn_sinks,     PULSAR_TENSOR_F32,  1, PULSAR_N_HEAD, 0, 0);
        tensor_expect_mxfp8(l->attn_output_a,   2, PULSAR_N_HEAD_DIM * (PULSAR_N_HEAD / PULSAR_N_OUT_GROUP), out_low_dim, 0);
        tensor_expect_mxfp8(l->attn_output_b,   2, out_low_dim, PULSAR_N_EMBD, 0);

        if (ratio != 0) {
            const uint32_t coff = ratio == 4 ? 2u : 1u;
            const uint64_t comp_width = (uint64_t)coff * PULSAR_N_HEAD_DIM;
            tensor_expect_plain_or_mxfp8(l->attn_compressor_ape, 2, comp_width, ratio, 0);
            tensor_expect_plain_or_mxfp8(l->attn_compressor_kv, 2, PULSAR_N_EMBD, comp_width, 0);
            tensor_expect_plain_or_mxfp8(l->attn_compressor_gate, 2, PULSAR_N_EMBD, comp_width, 0);
            tensor_expect_layout(l->attn_compressor_norm, PULSAR_TENSOR_F32, 1, PULSAR_N_HEAD_DIM, 0, 0);
        }
        if (ratio == 4) {
            const uint64_t index_q_dim = (uint64_t)PULSAR_N_INDEXER_HEAD * PULSAR_N_INDEXER_HEAD_DIM;
            const uint64_t index_width = 2u * PULSAR_N_INDEXER_HEAD_DIM;
            tensor_expect_plain_or_mxfp8(l->indexer_attn_q_b, 2, PULSAR_N_LORA_Q, index_q_dim, 0);
            tensor_expect_plain_or_mxfp8(l->indexer_proj, 2, PULSAR_N_EMBD, PULSAR_N_INDEXER_HEAD, 0);
            tensor_expect_plain_or_mxfp8(l->indexer_compressor_ape, 2, index_width, ratio, 0);
            tensor_expect_plain_or_mxfp8(l->indexer_compressor_kv, 2, PULSAR_N_EMBD, index_width, 0);
            tensor_expect_plain_or_mxfp8(l->indexer_compressor_gate, 2, PULSAR_N_EMBD, index_width, 0);
            tensor_expect_layout(l->indexer_compressor_norm,   PULSAR_TENSOR_F32, 1, PULSAR_N_INDEXER_HEAD_DIM, 0, 0);
        }

        tensor_expect_plain_or_mxfp8(l->hc_ffn_fn, 2, hc_dim, hc_mix_dim, 0);
        tensor_expect_layout(l->hc_ffn_scale,   PULSAR_TENSOR_F32,  1, 3, 0, 0);
        tensor_expect_layout(l->hc_ffn_base,    PULSAR_TENSOR_F32,  1, hc_mix_dim, 0, 0);
        tensor_expect_layout(l->ffn_norm,       PULSAR_TENSOR_F32,  1, PULSAR_N_EMBD, 0, 0);
        /* Router + bias stay padded to the full n_expert; only the expert
         * weight tensors are dense-trimmed to the per-layer survivor count
         * (== n_expert for un-pruned models). */
        const uint32_t n_layer_expert = pulsar_layer_n_expert(il);
        tensor_expect_plain_or_mxfp8(l->ffn_gate_inp, 2, PULSAR_N_EMBD, PULSAR_N_EXPERT, 0);
        tensor_expect_optional(l->ffn_exp_probs_b, PULSAR_TENSOR_F32, 1, PULSAR_N_EXPERT, 0, 0);
        tensor_expect_routed_expert(l->ffn_gate_exps, 3, PULSAR_N_EMBD, PULSAR_N_FF_EXP, n_layer_expert);
        tensor_expect_routed_expert(l->ffn_up_exps,   3, PULSAR_N_EMBD, PULSAR_N_FF_EXP, n_layer_expert);
        tensor_expect_routed_expert(l->ffn_down_exps, 3, PULSAR_N_FF_EXP, PULSAR_N_EMBD, n_layer_expert);
        tensor_expect_routed_expert_combo(l->ffn_gate_exps,
                                          l->ffn_up_exps,
                                          l->ffn_down_exps);
        tensor_expect_mxfp8(l->ffn_gate_shexp, 2, PULSAR_N_EMBD, PULSAR_N_FF_EXP, 0);
        tensor_expect_mxfp8(l->ffn_up_shexp,   2, PULSAR_N_EMBD, PULSAR_N_FF_EXP, 0);
        tensor_expect_mxfp8(l->ffn_down_shexp, 2, PULSAR_N_FF_EXP, PULSAR_N_EMBD, 0);
        if (il < PULSAR_N_HASH_LAYER) {
            tensor_expect_layout(l->ffn_gate_tid2eid, PULSAR_TENSOR_I32, 2, PULSAR_N_EXPERT_USED, PULSAR_N_VOCAB, 0);
        }
    }
}



static bool pulsar_shape_matches_metadata(
        const pulsar_shape *s,
        uint32_t n_layer,
        uint32_t n_embd,
        uint32_t n_vocab,
        uint32_t n_head,
        uint32_t n_head_kv,
        uint32_t n_head_dim,
        uint32_t n_value_dim,
        uint32_t n_rot,
        uint32_t n_lora_q,
        uint32_t n_lora_o,
        uint32_t n_out_group,
        uint32_t n_expert,
        uint32_t n_expert_used,
        uint32_t n_ff_exp,
        uint32_t n_expert_shared,
        uint32_t n_hash_layer,
        uint32_t n_swa,
        uint32_t n_indexer_head,
        uint32_t n_indexer_head_dim,
        uint32_t n_indexer_top_k,
        uint32_t n_hc,
        uint32_t n_hc_sinkhorn_iter) {
    return s->n_layer == n_layer &&
           s->n_embd == n_embd &&
           s->n_vocab == n_vocab &&
           s->n_head == n_head &&
           s->n_head_kv == n_head_kv &&
           s->n_head_dim == n_head_dim &&
           s->n_value_dim == n_value_dim &&
           s->n_rot == n_rot &&
           s->n_lora_q == n_lora_q &&
           s->n_lora_o == n_lora_o &&
           s->n_out_group == n_out_group &&
           s->n_expert == n_expert &&
           s->n_expert_used == n_expert_used &&
           s->n_ff_exp == n_ff_exp &&
           s->n_expert_shared == n_expert_shared &&
           s->n_hash_layer == n_hash_layer &&
           s->n_swa == n_swa &&
           s->n_indexer_head == n_indexer_head &&
           s->n_indexer_head_dim == n_indexer_head_dim &&
           s->n_indexer_top_k == n_indexer_top_k &&
           s->n_hc == n_hc &&
           s->n_hc_sinkhorn_iter == n_hc_sinkhorn_iter;
}



static void pulsar_select_shape_from_metadata(
        uint32_t n_layer,
        uint32_t n_embd,
        uint32_t n_vocab,
        uint32_t n_head,
        uint32_t n_head_kv,
        uint32_t n_head_dim,
        uint32_t n_value_dim,
        uint32_t n_rot,
        uint32_t n_lora_q,
        uint32_t n_lora_o,
        uint32_t n_out_group,
        uint32_t n_expert,
        uint32_t n_expert_used,
        uint32_t n_ff_exp,
        uint32_t n_expert_shared,
        uint32_t n_hash_layer,
        uint32_t n_swa,
        uint32_t n_indexer_head,
        uint32_t n_indexer_head_dim,
        uint32_t n_indexer_top_k,
        uint32_t n_hc,
        uint32_t n_hc_sinkhorn_iter) {
    if (pulsar_shape_matches_metadata(&PULSAR_SHAPE_FLASH,
                                   n_layer, n_embd, n_vocab, n_head, n_head_kv,
                                   n_head_dim, n_value_dim, n_rot, n_lora_q,
                                   n_lora_o, n_out_group, n_expert,
                                   n_expert_used, n_ff_exp, n_expert_shared,
                                   n_hash_layer, n_swa, n_indexer_head,
                                   n_indexer_head_dim, n_indexer_top_k, n_hc,
                                   n_hc_sinkhorn_iter)) {
        g_pulsar_shape = PULSAR_SHAPE_FLASH;
        return;
    }
    if (pulsar_shape_matches_metadata(&PULSAR_SHAPE_PRO,
                                   n_layer, n_embd, n_vocab, n_head, n_head_kv,
                                   n_head_dim, n_value_dim, n_rot, n_lora_q,
                                   n_lora_o, n_out_group, n_expert,
                                   n_expert_used, n_ff_exp, n_expert_shared,
                                   n_hash_layer, n_swa, n_indexer_head,
                                   n_indexer_head_dim, n_indexer_top_k, n_hc,
                                   n_hc_sinkhorn_iter)) {
        g_pulsar_shape = PULSAR_SHAPE_PRO;
        return;
    }

    fprintf(stderr,
            "pulsar: unsupported DeepSeek4 shape: layers=%u embd=%u heads=%u "
            "q_lora=%u out_groups=%u experts=%u ff_exp=%u indexer_top_k=%u\n",
            n_layer,
            n_embd,
            n_head,
            n_lora_q,
            n_out_group,
            n_expert,
            n_ff_exp,
            n_indexer_top_k);
    exit(1);
}



static void validate_compress_ratio_metadata(const pulsar_model *m) {
    const char *key = "deepseek4.attention.compress_ratios";
    pulsar_array_ref arr;
    if (!model_get_array(m, key, &arr) ||
        (arr.type != GGUF_VALUE_UINT32 && arr.type != GGUF_VALUE_INT32)) {
        fprintf(stderr, "pulsar: required int32/uint32 array metadata key is missing: %s\n", key);
        exit(1);
    }
    if (arr.len < PULSAR_N_LAYER) {
        pulsar_die("deepseek4.attention.compress_ratios is shorter than the layer count");
    }

    memset(g_pulsar_compress_ratios, 0, sizeof(g_pulsar_compress_ratios));
    pulsar_cursor c = cursor_at(m, arr.data_pos);
    for (uint32_t il = 0; il < PULSAR_N_LAYER; il++) {
        uint32_t got = 0;
        if (arr.type == GGUF_VALUE_UINT32) {
            if (!cursor_u32(&c, &got)) pulsar_die(c.error);
        } else {
            int32_t v = 0;
            if (!cursor_read(&c, &v, sizeof(v))) pulsar_die(c.error);
            if (v < 0) pulsar_die("metadata array contains a negative value");
            got = (uint32_t)v;
        }

        const uint32_t expected = pulsar_expected_layer_compress_ratio(il);
        if (got != expected) {
            fprintf(stderr,
                    "pulsar: unexpected DeepSeek4 compression ratio at layer %u for %s: got %u, expected %u\n",
                    il, PULSAR_MODEL_SHAPE_NAME, got, expected);
            exit(1);
        }
        g_pulsar_compress_ratios[il] = got;
    }
}



static void config_expect_f32(const char *name, float got, float expected);



/* REAP ds4-compact-v1 expert pruning: populate g_pulsar_layer_expert_count from
 * reap.layer.keep_count. Absent/disabled -> array stays zeroed (every layer
 * falls back to the full n_expert). When present, each entry must be in
 * [1, n_expert]; the router/bias tensors stay padded to n_expert and only the
 * expert weight tensors are dense-trimmed to keep_count. */
static void validate_reap_metadata(const pulsar_model *m) {
    memset(g_pulsar_layer_expert_count, 0, sizeof(g_pulsar_layer_expert_count));

    bool enabled = false;
    if (!model_get_bool(m, "reap.enabled", &enabled) || !enabled) return;

    const char *key = "reap.layer.keep_count";
    pulsar_array_ref arr;
    if (!model_get_array(m, key, &arr) ||
        (arr.type != GGUF_VALUE_UINT32 && arr.type != GGUF_VALUE_INT32)) {
        pulsar_die("reap.enabled is set but reap.layer.keep_count is missing or not an int32/uint32 array");
    }
    if (arr.len < PULSAR_N_LAYER) {
        pulsar_die("reap.layer.keep_count is shorter than the layer count");
    }

    pulsar_cursor c = cursor_at(m, arr.data_pos);
    for (uint32_t il = 0; il < PULSAR_N_LAYER; il++) {
        uint32_t got = 0;
        if (arr.type == GGUF_VALUE_UINT32) {
            if (!cursor_u32(&c, &got)) pulsar_die(c.error);
        } else {
            int32_t v = 0;
            if (!cursor_read(&c, &v, sizeof(v))) pulsar_die(c.error);
            if (v < 0) pulsar_die("reap.layer.keep_count contains a negative value");
            got = (uint32_t)v;
        }
        if (got == 0 || got > PULSAR_N_EXPERT) {
            fprintf(stderr,
                    "pulsar: reap.layer.keep_count[%u]=%u out of range [1, %u]\n",
                    il, got, PULSAR_N_EXPERT);
            exit(1);
        }
        g_pulsar_layer_expert_count[il] = got;
    }
}



static void validate_swiglu_clamp_metadata(const pulsar_model *m) {
    const char *key = "deepseek4.swiglu_clamp_exp";
    pulsar_array_ref arr;
    if (!model_get_array(m, key, &arr) ||
        (arr.type != GGUF_VALUE_FLOAT32 && arr.type != GGUF_VALUE_FLOAT64)) {
        fprintf(stderr, "pulsar: required float array metadata key is missing: %s\n", key);
        exit(1);
    }
    if (arr.len < PULSAR_N_LAYER) {
        pulsar_die("deepseek4.swiglu_clamp_exp is shorter than the layer count");
    }

    pulsar_cursor c = cursor_at(m, arr.data_pos);
    for (uint32_t i = 0; i < PULSAR_N_LAYER; i++) {
        float got = 0.0f;
        if (arr.type == GGUF_VALUE_FLOAT32) {
            if (!cursor_read(&c, &got, sizeof(got))) pulsar_die(c.error);
        } else {
            double v = 0.0;
            if (!cursor_read(&c, &v, sizeof(v))) pulsar_die(c.error);
            got = (float)v;
        }
        config_expect_f32("swiglu_clamp_exp", got, PULSAR_SWIGLU_CLAMP_EXP);
    }
}



static void config_expect_u32(const char *name, uint32_t got, uint32_t expected) {
    if (got == expected) return;
    fprintf(stderr, "pulsar: expected %s=%u for %s, got %u\n",
            name, expected, PULSAR_MODEL_SHAPE_NAME, got);
    exit(1);
}



static void config_expect_f32(const char *name, float got, float expected) {
    const float scale = fabsf(expected) > 1.0f ? fabsf(expected) : 1.0f;
    if (fabsf(got - expected) <= scale * 1.0e-6f) return;
    fprintf(stderr, "pulsar: expected %s=%.9g for %s, got %.9g\n",
            name, (double)expected, PULSAR_MODEL_SHAPE_NAME, (double)got);
    exit(1);
}



static void config_expect_bool(const char *name, bool got, bool expected) {
    if (got == expected) return;
    fprintf(stderr, "pulsar: expected %s=%s for %s, got %s\n",
            name, expected ? "true" : "false", PULSAR_MODEL_SHAPE_NAME, got ? "true" : "false");
    exit(1);
}



static void config_validate_fixed_shape(uint32_t n_layer) {
    config_expect_u32("block_count",                  n_layer,                 PULSAR_N_LAYER);
}



/* Validate metadata values that affect semantics: attention shape, HC count,
 * expert routing, RoPE scaling, compression ratios, and SwiGLU clamp. */
void config_validate_model(const pulsar_model *m) {
    const uint32_t n_layer = required_u32(m, "deepseek4.block_count");
    const uint32_t n_embd = required_u32(m, "deepseek4.embedding_length");
    const uint32_t n_vocab = required_u32(m, "deepseek4.vocab_size");
    const uint32_t n_head = required_u32(m, "deepseek4.attention.head_count");
    const uint32_t n_head_kv = required_u32(m, "deepseek4.attention.head_count_kv");
    const uint32_t n_head_dim = required_u32(m, "deepseek4.attention.key_length");
    const uint32_t n_value_dim = required_u32(m, "deepseek4.attention.value_length");
    const uint32_t n_rot = required_u32(m, "deepseek4.rope.dimension_count");
    const uint32_t n_lora_q = required_u32(m, "deepseek4.attention.q_lora_rank");
    const uint32_t n_lora_o = required_u32(m, "deepseek4.attention.output_lora_rank");
    const uint32_t n_out_group = required_u32(m, "deepseek4.attention.output_group_count");
    const uint32_t n_expert = required_u32(m, "deepseek4.expert_count");
    const uint32_t n_expert_used = required_u32(m, "deepseek4.expert_used_count");
    const uint32_t n_ff_exp = required_u32(m, "deepseek4.expert_feed_forward_length");
    const uint32_t n_expert_shared = required_u32(m, "deepseek4.expert_shared_count");
    const uint32_t n_hash_layer = required_u32(m, "deepseek4.hash_layer_count");
    uint32_t n_expert_groups = 0;
    uint32_t n_group_used = 0;
    model_get_u32(m, "deepseek4.expert_group_count", &n_expert_groups);
    model_get_u32(m, "deepseek4.expert_group_used_count", &n_group_used);
    const uint32_t n_swa = required_u32(m, "deepseek4.attention.sliding_window");
    const uint32_t n_indexer_head = required_u32(m, "deepseek4.attention.indexer.head_count");
    const uint32_t n_indexer_head_dim = required_u32(m, "deepseek4.attention.indexer.key_length");
    const uint32_t n_indexer_top_k = required_u32(m, "deepseek4.attention.indexer.top_k");
    const uint32_t n_hc = required_u32(m, "deepseek4.hyper_connection.count");
    const uint32_t n_hc_sinkhorn_iter = required_u32(m, "deepseek4.hyper_connection.sinkhorn_iterations");

    pulsar_select_shape_from_metadata(n_layer,
                                   n_embd,
                                   n_vocab,
                                   n_head,
                                   n_head_kv,
                                   n_head_dim,
                                   n_value_dim,
                                   n_rot,
                                   n_lora_q,
                                   n_lora_o,
                                   n_out_group,
                                   n_expert,
                                   n_expert_used,
                                   n_ff_exp,
                                   n_expert_shared,
                                   n_hash_layer,
                                   n_swa,
                                   n_indexer_head,
                                   n_indexer_head_dim,
                                   n_indexer_top_k,
                                   n_hc,
                                   n_hc_sinkhorn_iter);

    config_expect_u32("embedding_length",            n_embd,         PULSAR_N_EMBD);
    config_expect_u32("vocab_size",                  n_vocab,        PULSAR_N_VOCAB);
    config_expect_u32("attention.head_count",        n_head,         PULSAR_N_HEAD);
    config_expect_u32("attention.key_length",        n_head_dim,     PULSAR_N_HEAD_DIM);
    config_expect_u32("attention.head_count_kv",     n_head_kv,      PULSAR_N_HEAD_KV);
    config_expect_u32("attention.value_length",      n_value_dim,    PULSAR_N_VALUE_DIM);
    config_expect_u32("rope.dimension_count",        n_rot,          PULSAR_N_ROT);
    config_expect_u32("attention.output_group_count", n_out_group,    PULSAR_N_OUT_GROUP);
    config_expect_u32("attention.q_lora_rank",       n_lora_q,        PULSAR_N_LORA_Q);
    config_expect_u32("attention.output_lora_rank",  n_lora_o,        PULSAR_N_LORA_O);
    config_expect_u32("expert_count",               n_expert,        PULSAR_N_EXPERT);
    config_expect_u32("expert_used_count",          n_expert_used,   PULSAR_N_EXPERT_USED);
    config_expect_u32("expert_feed_forward_length", n_ff_exp,        PULSAR_N_FF_EXP);
    config_expect_u32("expert_shared_count",         n_expert_shared, PULSAR_N_EXPERT_SHARED);
    config_expect_u32("hash_layer_count",            n_hash_layer,    PULSAR_N_HASH_LAYER);
    config_expect_u32("expert_group_count",         n_expert_groups, 0);
    config_expect_u32("expert_group_used_count",    n_group_used,    0);

    config_expect_u32("attention.sliding_window",     n_swa,                   PULSAR_N_SWA);
    config_expect_u32("attention.indexer.head_count", n_indexer_head,     PULSAR_N_INDEXER_HEAD);
    config_expect_u32("attention.indexer.key_length", n_indexer_head_dim, PULSAR_N_INDEXER_HEAD_DIM);
    config_expect_u32("attention.indexer.top_k",      n_indexer_top_k,    PULSAR_N_INDEXER_TOP_K);
    config_expect_u32("hyper_connection.count", n_hc, PULSAR_N_HC);
    config_expect_u32("hyper_connection.sinkhorn_iterations", n_hc_sinkhorn_iter, PULSAR_N_HC_SINKHORN_ITER);

    config_validate_fixed_shape(n_layer);
    validate_compress_ratio_metadata(m);

    validate_reap_metadata(m);

    validate_swiglu_clamp_metadata(m);

    uint64_t rope_orig_ctx = PULSAR_ROPE_ORIG_CTX;
    model_get_u64_compat(m, "deepseek4.rope.scaling.original_context_length", &rope_orig_ctx);
    if (rope_orig_ctx != PULSAR_ROPE_ORIG_CTX) {
        fprintf(stderr, "pulsar: expected rope.scaling.original_context_length=%" PRIu64
                " for %s, got %" PRIu64 "\n",
                (uint64_t)PULSAR_ROPE_ORIG_CTX, PULSAR_MODEL_SHAPE_NAME, rope_orig_ctx);
        exit(1);
    }
    const float rope_freq_base = required_f32(m, "deepseek4.rope.freq_base");
    config_expect_f32("rope.freq_base", rope_freq_base, PULSAR_ROPE_FREQ_BASE);
    float rope_scale_factor = PULSAR_ROPE_SCALE_FACTOR;
    model_get_f32_compat(m, "deepseek4.rope.scaling.factor", &rope_scale_factor);
    config_expect_f32("rope.scaling.factor", rope_scale_factor, PULSAR_ROPE_SCALE_FACTOR);
    float rope_yarn_beta_fast = PULSAR_ROPE_YARN_BETA_FAST;
    model_get_f32_compat(m, "deepseek4.rope.scaling.yarn_beta_fast", &rope_yarn_beta_fast);
    config_expect_f32("rope.scaling.yarn_beta_fast", rope_yarn_beta_fast, PULSAR_ROPE_YARN_BETA_FAST);
    float rope_yarn_beta_slow = PULSAR_ROPE_YARN_BETA_SLOW;
    model_get_f32_compat(m, "deepseek4.rope.scaling.yarn_beta_slow", &rope_yarn_beta_slow);
    config_expect_f32("rope.scaling.yarn_beta_slow", rope_yarn_beta_slow, PULSAR_ROPE_YARN_BETA_SLOW);
    const float compress_rope_freq_base = required_f32(m, "deepseek4.attention.compress_rope_freq_base");
    config_expect_f32("attention.compress_rope_freq_base", compress_rope_freq_base, PULSAR_COMPRESS_ROPE_FREQ_BASE);
    const float expert_weight_scale = required_f32(m, "deepseek4.expert_weights_scale");
    config_expect_f32("expert_weights_scale", expert_weight_scale, PULSAR_EXPERT_WEIGHT_SCALE);
    const float rms_eps = required_f32(m, "deepseek4.attention.layer_norm_rms_epsilon");
    config_expect_f32("attention.layer_norm_rms_epsilon", rms_eps, PULSAR_RMS_EPS);
    const float hc_eps = required_f32(m, "deepseek4.hyper_connection.epsilon");
    config_expect_f32("hyper_connection.epsilon", hc_eps, PULSAR_HC_EPS);
    const bool expert_weight_norm = required_bool(m, "deepseek4.expert_weights_norm");
    config_expect_bool("expert_weights_norm", expert_weight_norm, true);
}



/* Weight formats the engine still decodes.  Legacy Q4_K and Q8_0 weight
 * support has been removed; reject such GGUFs up front with one clear error
 * instead of failing on the first per-tensor layout check. */
static bool weights_tensor_type_supported(uint32_t type) {
    switch (type) {
    case PULSAR_TENSOR_F32:
    case PULSAR_TENSOR_F16:
    case PULSAR_TENSOR_Q2_K:
    case PULSAR_TENSOR_IQ2_XXS:
    case PULSAR_TENSOR_I32:
    case PULSAR_TENSOR_BF16:
    case PULSAR_TENSOR_FP8_E4M3:
    case PULSAR_TENSOR_MXFP8_LT:
    case PULSAR_TENSOR_FP4_E2M1:
    case PULSAR_TENSOR_CUTLASS_MXFP4:
        return true;
    default:
        return false;
    }
}



static void weights_reject_unsupported_types(const pulsar_model *m) {
    bool seen[256] = { false };
    bool any = false;

    for (uint64_t i = 0; i < m->n_tensors; i++) {
        const uint32_t type = m->tensors[i].type;
        if (weights_tensor_type_supported(type)) continue;
        if (type < 256 && seen[type]) continue;
        if (type < 256) seen[type] = true;
        fprintf(stderr,
                "pulsar: unsupported weight tensor type %s (first tensor: %.*s)\n",
                tensor_type_name(type),
                (int)m->tensors[i].name.len,
                m->tensors[i].name.ptr);
        any = true;
    }
    if (any) {
        fprintf(stderr,
                "pulsar: supported weight tensor types: f32, f16, bf16, i32, q2_k, "
                "iq2_xxs, fp8_e4m3 (MXFP8), mxfp4\n");
        exit(1);
    }
}



static void weights_bind_output(pulsar_weights *w, const pulsar_model *m, bool required, bool optional) {
    if (required) {
        w->output_hc_base   = required_tensor(m, "output_hc_base.weight");
        w->output_hc_fn     = required_tensor(m, "output_hc_fn.weight");
        w->output_hc_scale  = required_tensor(m, "output_hc_scale.weight");
        w->output_norm      = required_tensor(m, "output_norm.weight");
        w->output           = required_tensor(m, "output.weight");
        return;
    }
    if (!optional) return;

    w->output_hc_base   = model_find_tensor(m, "output_hc_base.weight");
    w->output_hc_fn     = model_find_tensor(m, "output_hc_fn.weight");
    w->output_hc_scale  = model_find_tensor(m, "output_hc_scale.weight");
    w->output_norm      = model_find_tensor(m, "output_norm.weight");
    w->output           = model_find_tensor(m, "output.weight");
    if (weights_have_partial_output_head(w) && !weights_have_output_head(w)) {
        pulsar_die("partial output head in GGUF");
    }
}



static void weights_bind_layer(pulsar_layer_weights *l, const pulsar_model *m, uint32_t il) {
    const uint32_t compress_ratio = pulsar_layer_compress_ratio(il);

    l->hc_attn_fn      = required_tensorf(m, "blk.%u.hc_attn_fn.weight", il);
    l->hc_attn_scale   = required_tensorf(m, "blk.%u.hc_attn_scale.weight", il);
    l->hc_attn_base    = required_tensorf(m, "blk.%u.hc_attn_base.weight", il);
    l->attn_norm       = required_tensorf(m, "blk.%u.attn_norm.weight", il);
    l->attn_q_a        = required_tensorf(m, "blk.%u.attn_q_a.weight", il);
    l->attn_q_a_norm   = required_tensorf(m, "blk.%u.attn_q_a_norm.weight", il);
    l->attn_q_b        = required_tensorf(m, "blk.%u.attn_q_b.weight", il);
    l->attn_kv         = required_tensorf(m, "blk.%u.attn_kv.weight", il);
    l->attn_kv_a_norm  = required_tensorf(m, "blk.%u.attn_kv_a_norm.weight", il);
    l->attn_sinks      = required_tensorf(m, "blk.%u.attn_sinks.weight", il);
    l->attn_output_a   = required_tensorf(m, "blk.%u.attn_output_a.weight", il);
    l->attn_output_b   = required_tensorf(m, "blk.%u.attn_output_b.weight", il);
    if (compress_ratio != 0) {
        l->attn_compressor_ape  = required_tensorf(m, "blk.%u.attn_compressor_ape.weight", il);
        l->attn_compressor_kv   = required_tensorf(m, "blk.%u.attn_compressor_kv.weight", il);
        l->attn_compressor_gate = required_tensorf(m, "blk.%u.attn_compressor_gate.weight", il);
        l->attn_compressor_norm = required_tensorf(m, "blk.%u.attn_compressor_norm.weight", il);
    }
    if (compress_ratio == 4) {
        l->indexer_attn_q_b = required_tensorf(m, "blk.%u.indexer.attn_q_b.weight", il);
        l->indexer_proj     = required_tensorf(m, "blk.%u.indexer.proj.weight", il);
        l->indexer_compressor_ape  = required_tensorf(m, "blk.%u.indexer_compressor_ape.weight", il);
        l->indexer_compressor_kv   = required_tensorf(m, "blk.%u.indexer_compressor_kv.weight", il);
        l->indexer_compressor_gate = required_tensorf(m, "blk.%u.indexer_compressor_gate.weight", il);
        l->indexer_compressor_norm = required_tensorf(m, "blk.%u.indexer_compressor_norm.weight", il);
    }
    l->hc_ffn_fn       = required_tensorf(m, "blk.%u.hc_ffn_fn.weight", il);
    l->hc_ffn_scale    = required_tensorf(m, "blk.%u.hc_ffn_scale.weight", il);
    l->hc_ffn_base     = required_tensorf(m, "blk.%u.hc_ffn_base.weight", il);
    l->ffn_norm        = required_tensorf(m, "blk.%u.ffn_norm.weight", il);
    l->ffn_gate_inp    = required_tensorf(m, "blk.%u.ffn_gate_inp.weight", il);
    l->ffn_exp_probs_b = tensor_by_namef(m, "blk.%u.exp_probs_b.bias", il);
    l->ffn_gate_exps   = required_tensorf(m, "blk.%u.ffn_gate_exps.weight", il);
    l->ffn_up_exps     = required_tensorf(m, "blk.%u.ffn_up_exps.weight", il);
    l->ffn_down_exps   = required_tensorf(m, "blk.%u.ffn_down_exps.weight", il);
    l->ffn_gate_shexp  = required_tensorf(m, "blk.%u.ffn_gate_shexp.weight", il);
    l->ffn_up_shexp    = required_tensorf(m, "blk.%u.ffn_up_shexp.weight", il);
    l->ffn_down_shexp  = required_tensorf(m, "blk.%u.ffn_down_shexp.weight", il);

    if (il < PULSAR_N_HASH_LAYER) {
        l->ffn_gate_tid2eid = required_tensorf(m, "blk.%u.ffn_gate_tid2eid.weight", il);
    }
}



/* Bind tensor names once into the fixed DS4 layer layout.  This is the point
 * where stringly GGUF metadata becomes direct model-specific pointers. */
void weights_bind(pulsar_weights *w, const pulsar_model *m) {
    memset(w, 0, sizeof(*w));
    weights_reject_unsupported_types(m);

    w->token_embd = required_tensor(m, "token_embd.weight");
    weights_bind_output(w, m, true, false);

    for (uint32_t il = 0; il < PULSAR_N_LAYER; il++) {
        weights_bind_layer(&w->layer[il], m, il);
    }

    weights_validate_layout(w, 0, PULSAR_N_LAYER - 1u, true, true);
}





void weights_free(pulsar_weights *w) {
    memset(w, 0, sizeof(*w));
}

static void dspark_weights_validate_layout(const pulsar_dspark_weights *w) {
    const uint32_t E = w->embed_dim;
    const uint32_t V = w->vocab_size;
    /*
     * The tensor-layout checks below are self-referential: they validate every
     * DSpark tensor against E/V read from the support GGUF's own metadata.  But
     * the runtime drives DSpark with the target model's compiled constants
     * (PULSAR_N_EMBD embed buffers, PULSAR_N_VOCAB logits stride), so a support model
     * built for a different base would pass this validation and then produce
     * misaligned reads or garbage drafts.  Pin E/V to the target here.
     */
    if (E != PULSAR_N_EMBD || V != PULSAR_N_VOCAB) {
        char msg[160];
        snprintf(msg, sizeof(msg),
                 "dspark: support model shape (embed=%u vocab=%u) does not match "
                 "target (embed=%u vocab=%u)",
                 E, V, (uint32_t)PULSAR_N_EMBD, (uint32_t)PULSAR_N_VOCAB);
        pulsar_die(msg);
    }
    const uint64_t hc_dim = (uint64_t)E * PULSAR_N_HC;
    const uint64_t hc_mix_dim = 2u * PULSAR_N_HC + (uint64_t)PULSAR_N_HC * PULSAR_N_HC;
    const uint64_t q_dim = (uint64_t)PULSAR_N_HEAD * PULSAR_N_HEAD_DIM;
    const uint64_t out_low_dim = (uint64_t)PULSAR_N_OUT_GROUP * PULSAR_N_LORA_O;

    tensor_expect_mxfp8(w->main_proj, 2, 3ull * E, E, 0);
    tensor_expect_layout(w->main_norm, PULSAR_TENSOR_F32, 1, E, 0, 0);

    for (int li = 0; li < 3; li++) {
        const pulsar_layer_weights *l = &w->layer[li];
        tensor_expect_plain_layout(l->hc_attn_fn, 2, hc_dim, hc_mix_dim, 0);
        tensor_expect_layout(l->hc_attn_scale, PULSAR_TENSOR_F32, 1, 3, 0, 0);
        tensor_expect_layout(l->hc_attn_base, PULSAR_TENSOR_F32, 1, hc_mix_dim, 0, 0);
        tensor_expect_layout(l->attn_norm, PULSAR_TENSOR_F32, 1, E, 0, 0);
        tensor_expect_mxfp8(l->attn_q_a, 2, E, PULSAR_N_LORA_Q, 0);
        tensor_expect_layout(l->attn_q_a_norm, PULSAR_TENSOR_F32, 1, PULSAR_N_LORA_Q, 0, 0);
        tensor_expect_mxfp8(l->attn_q_b, 2, PULSAR_N_LORA_Q, q_dim, 0);
        tensor_expect_mxfp8(l->attn_kv, 2, E, PULSAR_N_HEAD_DIM, 0);
        tensor_expect_layout(l->attn_kv_a_norm, PULSAR_TENSOR_F32, 1, PULSAR_N_HEAD_DIM, 0, 0);
        tensor_expect_layout(l->attn_sinks, PULSAR_TENSOR_F32, 1, PULSAR_N_HEAD, 0, 0);
        tensor_expect_mxfp8(l->attn_output_a, 2,
                             PULSAR_N_HEAD_DIM * (PULSAR_N_HEAD / PULSAR_N_OUT_GROUP), out_low_dim, 0);
        tensor_expect_mxfp8(l->attn_output_b, 2, out_low_dim, E, 0);
        tensor_expect_plain_layout(l->hc_ffn_fn, 2, hc_dim, hc_mix_dim, 0);
        tensor_expect_layout(l->hc_ffn_scale, PULSAR_TENSOR_F32, 1, 3, 0, 0);
        tensor_expect_layout(l->hc_ffn_base, PULSAR_TENSOR_F32, 1, hc_mix_dim, 0, 0);
        tensor_expect_layout(l->ffn_norm, PULSAR_TENSOR_F32, 1, E, 0, 0);
        tensor_expect_plain_layout(l->ffn_gate_inp, 2, E, PULSAR_N_EXPERT, 0);
        tensor_expect_routed_expert(l->ffn_gate_exps, 3, E, PULSAR_N_FF_EXP, PULSAR_N_EXPERT);
        tensor_expect_routed_expert(l->ffn_up_exps,   3, E, PULSAR_N_FF_EXP, PULSAR_N_EXPERT);
        tensor_expect_routed_expert(l->ffn_down_exps, 3, PULSAR_N_FF_EXP, E, PULSAR_N_EXPERT);
        tensor_expect_routed_expert_combo(l->ffn_gate_exps,
                                          l->ffn_up_exps,
                                          l->ffn_down_exps);
        tensor_expect_mxfp8(l->ffn_gate_shexp, 2, E, PULSAR_N_FF_EXP, 0);
        tensor_expect_mxfp8(l->ffn_up_shexp,   2, E, PULSAR_N_FF_EXP, 0);
        tensor_expect_mxfp8(l->ffn_down_shexp, 2, PULSAR_N_FF_EXP, E, 0);
    }

    tensor_expect_layout(w->markov_w1, PULSAR_TENSOR_F32, 2, 256, V, 0);
    tensor_expect_layout(w->markov_w2, PULSAR_TENSOR_F32, 2, 256, V, 0);
    tensor_expect_layout(w->confidence_proj, PULSAR_TENSOR_F32, 1, E + 256, 0, 0);
    tensor_expect_layout(w->hc_head_base, PULSAR_TENSOR_F32, 1, PULSAR_N_HC, 0, 0);
    tensor_expect_layout(w->hc_head_fn, PULSAR_TENSOR_F32, 2, (uint64_t)PULSAR_N_HC * E, PULSAR_N_HC, 0);
    tensor_expect_layout(w->hc_head_scale, PULSAR_TENSOR_F32, 1, 1, 0, 0);
    tensor_expect_layout(w->final_norm, PULSAR_TENSOR_F32, 1, E, 0, 0);
}

void dspark_weights_bind(pulsar_dspark_weights *w, const pulsar_model *m) {
    memset(w, 0, sizeof(*w));
    weights_reject_unsupported_types(m);

    w->embed_dim = required_u32(m, "deepseek_v4_dspark.embedding_length");
    w->main_proj = required_tensor(m, "dspark.main_proj.weight");
    w->main_norm = required_tensor(m, "dspark.main_norm.weight");

    for (int li = 0; li < 3; li++) {
        pulsar_layer_weights *l = &w->layer[li];
        l->hc_attn_fn      = required_tensorf(m, "dspark.%d.hc_attn_fn.weight", li);
        l->hc_attn_scale   = required_tensorf(m, "dspark.%d.hc_attn_scale.weight", li);
        l->hc_attn_base    = required_tensorf(m, "dspark.%d.hc_attn_base.weight", li);
        l->attn_norm       = required_tensorf(m, "dspark.%d.attn_norm.weight", li);
        l->attn_q_a        = required_tensorf(m, "dspark.%d.attn_q_a.weight", li);
        l->attn_q_a_norm   = required_tensorf(m, "dspark.%d.attn_q_a_norm.weight", li);
        l->attn_q_b        = required_tensorf(m, "dspark.%d.attn_q_b.weight", li);
        l->attn_kv         = required_tensorf(m, "dspark.%d.attn_kv.weight", li);
        l->attn_kv_a_norm  = required_tensorf(m, "dspark.%d.attn_kv_a_norm.weight", li);
        l->attn_sinks      = required_tensorf(m, "dspark.%d.attn_sinks.weight", li);
        l->attn_output_a   = required_tensorf(m, "dspark.%d.attn_output_a.weight", li);
        l->attn_output_b   = required_tensorf(m, "dspark.%d.attn_output_b.weight", li);
        l->hc_ffn_fn       = required_tensorf(m, "dspark.%d.hc_ffn_fn.weight", li);
        l->hc_ffn_scale    = required_tensorf(m, "dspark.%d.hc_ffn_scale.weight", li);
        l->hc_ffn_base     = required_tensorf(m, "dspark.%d.hc_ffn_base.weight", li);
        l->ffn_norm        = required_tensorf(m, "dspark.%d.ffn_norm.weight", li);
        l->ffn_gate_inp    = required_tensorf(m, "dspark.%d.ffn_gate_inp.weight", li);
        l->ffn_gate_exps   = required_tensorf(m, "dspark.%d.ffn_gate_exps.weight", li);
        l->ffn_up_exps     = required_tensorf(m, "dspark.%d.ffn_up_exps.weight", li);
        l->ffn_down_exps   = required_tensorf(m, "dspark.%d.ffn_down_exps.weight", li);
        l->ffn_gate_shexp  = required_tensorf(m, "dspark.%d.ffn_gate_shexp.weight", li);
        l->ffn_up_shexp    = required_tensorf(m, "dspark.%d.ffn_up_shexp.weight", li);
        l->ffn_down_shexp  = required_tensorf(m, "dspark.%d.ffn_down_shexp.weight", li);
    }

    w->markov_w1        = required_tensor(m, "dspark.2.markov_head.markov_w1.weight");
    w->markov_w2        = required_tensor(m, "dspark.2.markov_head.markov_w2.weight");
    w->confidence_proj  = required_tensor(m, "dspark.2.confidence_head.proj.weight");
    w->hc_head_base     = required_tensor(m, "dspark.2.hc_head_base.weight");
    w->hc_head_fn       = required_tensor(m, "dspark.2.hc_head_fn.weight");
    w->hc_head_scale    = required_tensor(m, "dspark.2.hc_head_scale.weight");
    w->final_norm       = required_tensor(m, "dspark.2.norm.weight");

    w->vocab_size = (uint32_t)w->markov_w1->dim[1];

    {
        uint32_t target_ids[3];
        if (model_get_u32(m, "dspark.target_layer_ids.0", &target_ids[0]) &&
            model_get_u32(m, "dspark.target_layer_ids.1", &target_ids[1]) &&
            model_get_u32(m, "dspark.target_layer_ids.2", &target_ids[2])) {
            memcpy(w->target_layer_ids, target_ids, sizeof(target_ids));
        } else {
            /*
             * Fall back to the target model's last three layers.  Derive from
             * the compiled target shape (PULSAR_N_LAYER), NOT the support GGUF's
             * block_count — that field describes the 3-layer draft backbone, so
             * using it would capture target layers {0,1,2} instead of the tail.
             */
            w->target_layer_ids[0] = PULSAR_N_LAYER - 3;
            w->target_layer_ids[1] = PULSAR_N_LAYER - 2;
            w->target_layer_ids[2] = PULSAR_N_LAYER - 1;
        }
    }

    dspark_weights_validate_layout(w);
}



/* Load one token embedding row and expand it to float activations. */
void embed_token_f16(const pulsar_model *m, const pulsar_weights *w, int token, float *out) {
    pulsar_tensor *te = w->token_embd;
    if (token < 0 || (uint64_t)token >= te->dim[1]) {
        pulsar_die("token id is outside the embedding table");
    }

    const uint16_t *base = (const uint16_t *)tensor_data(m, te);
    const uint64_t stride = te->dim[0];
    const uint16_t *row = base + (uint64_t)token * stride;

    for (uint64_t i = 0; i < stride; i++) {
        out[i] = f16_to_f32(row[i]);
    }
}



/* RMSNorm without a learned scale, used by hyper-connection control vectors. */
void rms_norm_no_weight(float *out, const float *x, uint64_t n, float eps) {
    double ss = 0.0;
    for (uint64_t i = 0; i < n; i++) ss += (double)x[i] * x[i];

    const float scale = 1.0f / sqrtf((float)(ss / (double)n) + eps);
    for (uint64_t i = 0; i < n; i++) out[i] = x[i] * scale;
}



/* Standard DS4 RMSNorm with learned per-channel scale. */
void rms_norm_weight(float *out, const float *x, const float *weight, uint64_t n, float eps) {
    double ss = 0.0;
    for (uint64_t i = 0; i < n; i++) ss += (double)x[i] * x[i];

    const float scale = 1.0f / sqrtf((float)(ss / (double)n) + eps);
    for (uint64_t i = 0; i < n; i++) out[i] = x[i] * scale * weight[i];
}



/* Normalize each attention head independently after Q projection. */
void head_rms_norm_inplace(float *x, uint32_t n_head, uint32_t head_dim, float eps) {
    for (uint32_t h = 0; h < n_head; h++) {
        float *head = x + (uint64_t)h * head_dim;
        double ss = 0.0;
        for (uint32_t i = 0; i < head_dim; i++) ss += (double)head[i] * head[i];

        const float scale = 1.0f / sqrtf((float)(ss / (double)head_dim) + eps);
        for (uint32_t i = 0; i < head_dim; i++) head[i] *= scale;
    }
}



static inline float dot_f16_row(const uint16_t *row, const float *x, uint64_t n) {
#if defined(__ARM_NEON)
    uint64_t i = 0;
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    for (; i + 8 <= n; i += 8) {
        const float16x8_t hv = vreinterpretq_f16_u16(vld1q_u16(row + i));
        const float32x4_t h0 = vcvt_f32_f16(vget_low_f16(hv));
        const float32x4_t h1 = vcvt_f32_f16(vget_high_f16(hv));
        acc0 = vfmaq_f32(acc0, h0, vld1q_f32(x + i));
        acc1 = vfmaq_f32(acc1, h1, vld1q_f32(x + i + 4));
    }

    float acc = vaddvq_f32(vaddq_f32(acc0, acc1));
    for (; i < n; i++) acc += f16_to_f32(row[i]) * x[i];
    return acc;
#else
    float acc = 0.0f;
    for (uint64_t i = 0; i < n; i++) acc += f16_to_f32(row[i]) * x[i];
    return acc;
#endif
}



static void matvec_f16_worker(void *vctx, uint64_t row0, uint64_t row1) {
    matvec_f16_ctx *ctx = static_cast<matvec_f16_ctx *>(vctx);

    for (uint64_t o = row0; o < row1; o++) {
        const uint16_t *row = ctx->data + o * ctx->in_dim;
        ctx->out[o] = dot_f16_row(row, ctx->x, ctx->in_dim);
    }
}



/* Dense F16 matvec for small control projections such as HC and router heads. */
void matvec_f16(float *out, const pulsar_model *m, const pulsar_tensor *w, const float *x) {
    if (w->type != 1 || w->ndim != 2) pulsar_die("expected a 2D F16 tensor");

    const uint64_t in_dim = w->dim[0];
    const uint64_t out_dim = w->dim[1];
    matvec_f16_ctx ctx = {
        .out = out,
        .data = (const uint16_t *)tensor_data(m, w),
        .x = x,
        .in_dim = in_dim,
    };

    const uint64_t ops = in_dim * out_dim;
    const uint64_t min_rows = ops >= 262144 ? 1 : 512;
    pulsar_parallel_for_min_rows(out_dim, matvec_f16_worker, &ctx, min_rows);
}



void matvec_f16_serial(float *out, const pulsar_model *m, const pulsar_tensor *w, const float *x) {
    if (w->type != 1 || w->ndim != 2) pulsar_die("expected a 2D F16 tensor");

    const uint64_t in_dim = w->dim[0];
    const uint64_t out_dim = w->dim[1];
    const uint16_t *data = (const uint16_t *)tensor_data(m, w);
    for (uint64_t o = 0; o < out_dim; o++) {
        out[o] = dot_f16_row(data + o * in_dim, x, in_dim);
    }
}



static inline int32_t dot_i8_32(const int8_t *a, const int8_t *b, uint64_t n) {
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    if (n == 32) {
        int32x4_t acc = vdupq_n_s32(0);
        acc = vdotq_s32(acc, vld1q_s8(a),      vld1q_s8(b));
        acc = vdotq_s32(acc, vld1q_s8(a + 16), vld1q_s8(b + 16));
        return vaddvq_s32(acc);
    }
#endif
    int32_t sum = 0;
    for (uint64_t i = 0; i < n; i++) sum += (int32_t)a[i] * (int32_t)b[i];
    return sum;
}



static inline float dot_q8_0_row(
        const uint8_t *row,
        const int8_t  *xq,
        const float   *xscale,
        uint64_t       in_dim,
        uint64_t       blocks) {
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    if ((in_dim & 31u) == 0) {
        float32x4_t accv0 = vdupq_n_f32(0.0f);
        float32x4_t accv1 = vdupq_n_f32(0.0f);

        uint64_t b = 0;
        for (; b + 1 < blocks; b += 2) {
            uint16_t scale_bits0;
            uint16_t scale_bits1;
            memcpy(&scale_bits0, row + b * 34, sizeof(scale_bits0));
            memcpy(&scale_bits1, row + (b + 1) * 34, sizeof(scale_bits1));

            const int8_t *qs0 = (const int8_t *)(row + b * 34 + 2);
            const int8_t *qs1 = (const int8_t *)(row + (b + 1) * 34 + 2);
            const int8_t *xq0 = xq + b * 32;
            const int8_t *xq1 = xq + (b + 1) * 32;

            int32x4_t dot0 = vdupq_n_s32(0);
            dot0 = vdotq_s32(dot0, vld1q_s8(qs0),      vld1q_s8(xq0));
            dot0 = vdotq_s32(dot0, vld1q_s8(qs0 + 16), vld1q_s8(xq0 + 16));

            int32x4_t dot1 = vdupq_n_s32(0);
            dot1 = vdotq_s32(dot1, vld1q_s8(qs1),      vld1q_s8(xq1));
            dot1 = vdotq_s32(dot1, vld1q_s8(qs1 + 16), vld1q_s8(xq1 + 16));

            accv0 = vfmaq_n_f32(accv0, vcvtq_f32_s32(dot0), f16_to_f32(scale_bits0) * xscale[b]);
            accv1 = vfmaq_n_f32(accv1, vcvtq_f32_s32(dot1), f16_to_f32(scale_bits1) * xscale[b + 1]);
        }

        if (b < blocks) {
            uint16_t scale_bits;
            memcpy(&scale_bits, row + b * 34, sizeof(scale_bits));
            const int8_t *qs = (const int8_t *)(row + b * 34 + 2);
            const int8_t *xqb = xq + b * 32;
            int32x4_t dot = vdupq_n_s32(0);
            dot = vdotq_s32(dot, vld1q_s8(qs),      vld1q_s8(xqb));
            dot = vdotq_s32(dot, vld1q_s8(qs + 16), vld1q_s8(xqb + 16));
            accv0 = vfmaq_n_f32(accv0, vcvtq_f32_s32(dot), f16_to_f32(scale_bits) * xscale[b]);
        }

        return vaddvq_f32(vaddq_f32(accv0, accv1));
    }
#endif

    float acc = 0.0f;
    for (uint64_t b = 0; b < blocks; b++) {
        uint16_t scale_bits;
        memcpy(&scale_bits, row + b * 34, sizeof(scale_bits));
        const int8_t *qs = (const int8_t *)(row + b * 34 + 2);

        const uint64_t i0 = b * 32;
        const uint64_t n = in_dim - i0 < 32 ? in_dim - i0 : 32;
        acc += f16_to_f32(scale_bits) * xscale[b] * (float)dot_i8_32(qs, xq + i0, n);
    }
    return acc;
}



static inline void dot_q8_0_row_2(
        const uint8_t *row,
        const int8_t  *xq0,
        const float   *xscale0,
        const int8_t  *xq1,
        const float   *xscale1,
        uint64_t       in_dim,
        uint64_t       blocks,
        float         *out0,
        float         *out1) {
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    if ((in_dim & 31u) == 0) {
        float32x4_t acc00 = vdupq_n_f32(0.0f);
        float32x4_t acc01 = vdupq_n_f32(0.0f);
        float32x4_t acc10 = vdupq_n_f32(0.0f);
        float32x4_t acc11 = vdupq_n_f32(0.0f);

        uint64_t b = 0;
        for (; b + 1 < blocks; b += 2) {
            uint16_t scale_bits0;
            uint16_t scale_bits1;
            memcpy(&scale_bits0, row + b * 34, sizeof(scale_bits0));
            memcpy(&scale_bits1, row + (b + 1) * 34, sizeof(scale_bits1));

            const int8_t *qs0 = (const int8_t *)(row + b * 34 + 2);
            const int8_t *qs1 = (const int8_t *)(row + (b + 1) * 34 + 2);

            int32x4_t d00 = vdupq_n_s32(0);
            d00 = vdotq_s32(d00, vld1q_s8(qs0),      vld1q_s8(xq0 + b * 32));
            d00 = vdotq_s32(d00, vld1q_s8(qs0 + 16), vld1q_s8(xq0 + b * 32 + 16));
            int32x4_t d01 = vdupq_n_s32(0);
            d01 = vdotq_s32(d01, vld1q_s8(qs1),      vld1q_s8(xq0 + (b + 1) * 32));
            d01 = vdotq_s32(d01, vld1q_s8(qs1 + 16), vld1q_s8(xq0 + (b + 1) * 32 + 16));

            int32x4_t d10 = vdupq_n_s32(0);
            d10 = vdotq_s32(d10, vld1q_s8(qs0),      vld1q_s8(xq1 + b * 32));
            d10 = vdotq_s32(d10, vld1q_s8(qs0 + 16), vld1q_s8(xq1 + b * 32 + 16));
            int32x4_t d11 = vdupq_n_s32(0);
            d11 = vdotq_s32(d11, vld1q_s8(qs1),      vld1q_s8(xq1 + (b + 1) * 32));
            d11 = vdotq_s32(d11, vld1q_s8(qs1 + 16), vld1q_s8(xq1 + (b + 1) * 32 + 16));

            const float s0 = f16_to_f32(scale_bits0);
            const float s1 = f16_to_f32(scale_bits1);
            acc00 = vfmaq_n_f32(acc00, vcvtq_f32_s32(d00), s0 * xscale0[b]);
            acc01 = vfmaq_n_f32(acc01, vcvtq_f32_s32(d01), s1 * xscale0[b + 1]);
            acc10 = vfmaq_n_f32(acc10, vcvtq_f32_s32(d10), s0 * xscale1[b]);
            acc11 = vfmaq_n_f32(acc11, vcvtq_f32_s32(d11), s1 * xscale1[b + 1]);
        }

        if (b < blocks) {
            uint16_t scale_bits;
            memcpy(&scale_bits, row + b * 34, sizeof(scale_bits));
            const int8_t *qs = (const int8_t *)(row + b * 34 + 2);

            int32x4_t d0 = vdupq_n_s32(0);
            d0 = vdotq_s32(d0, vld1q_s8(qs),      vld1q_s8(xq0 + b * 32));
            d0 = vdotq_s32(d0, vld1q_s8(qs + 16), vld1q_s8(xq0 + b * 32 + 16));
            int32x4_t d1 = vdupq_n_s32(0);
            d1 = vdotq_s32(d1, vld1q_s8(qs),      vld1q_s8(xq1 + b * 32));
            d1 = vdotq_s32(d1, vld1q_s8(qs + 16), vld1q_s8(xq1 + b * 32 + 16));

            const float s0 = f16_to_f32(scale_bits);
            acc00 = vfmaq_n_f32(acc00, vcvtq_f32_s32(d0), s0 * xscale0[b]);
            acc10 = vfmaq_n_f32(acc10, vcvtq_f32_s32(d1), s0 * xscale1[b]);
        }

        *out0 = vaddvq_f32(vaddq_f32(acc00, acc01));
        *out1 = vaddvq_f32(vaddq_f32(acc10, acc11));
        return;
    }
#endif

    *out0 = dot_q8_0_row(row, xq0, xscale0, in_dim, blocks);
    *out1 = dot_q8_0_row(row, xq1, xscale1, in_dim, blocks);
}



static inline PULSAR_MAYBE_UNUSED void dot_q8_0_row_pair(
        const uint8_t *row0,
        const uint8_t *row1,
        const int8_t  *xq,
        const float   *xscale,
        uint64_t       in_dim,
        uint64_t       blocks,
        float         *out0,
        float         *out1) {
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    if ((in_dim & 31u) == 0) {
        float32x4_t acc00 = vdupq_n_f32(0.0f);
        float32x4_t acc01 = vdupq_n_f32(0.0f);
        float32x4_t acc10 = vdupq_n_f32(0.0f);
        float32x4_t acc11 = vdupq_n_f32(0.0f);

        uint64_t b = 0;
        for (; b + 1 < blocks; b += 2) {
            uint16_t s00, s01, s10, s11;
            memcpy(&s00, row0 + b * 34, sizeof(s00));
            memcpy(&s01, row0 + (b + 1) * 34, sizeof(s01));
            memcpy(&s10, row1 + b * 34, sizeof(s10));
            memcpy(&s11, row1 + (b + 1) * 34, sizeof(s11));

            const int8_t *xq0 = xq + b * 32;
            const int8_t *xq1 = xq + (b + 1) * 32;
            const int8x16_t xv00 = vld1q_s8(xq0);
            const int8x16_t xv01 = vld1q_s8(xq0 + 16);
            const int8x16_t xv10 = vld1q_s8(xq1);
            const int8x16_t xv11 = vld1q_s8(xq1 + 16);

            const int8_t *q00 = (const int8_t *)(row0 + b * 34 + 2);
            const int8_t *q01 = (const int8_t *)(row0 + (b + 1) * 34 + 2);
            const int8_t *q10 = (const int8_t *)(row1 + b * 34 + 2);
            const int8_t *q11 = (const int8_t *)(row1 + (b + 1) * 34 + 2);

            int32x4_t d00 = vdupq_n_s32(0);
            d00 = vdotq_s32(d00, vld1q_s8(q00),      xv00);
            d00 = vdotq_s32(d00, vld1q_s8(q00 + 16), xv01);
            int32x4_t d01 = vdupq_n_s32(0);
            d01 = vdotq_s32(d01, vld1q_s8(q01),      xv10);
            d01 = vdotq_s32(d01, vld1q_s8(q01 + 16), xv11);
            int32x4_t d10 = vdupq_n_s32(0);
            d10 = vdotq_s32(d10, vld1q_s8(q10),      xv00);
            d10 = vdotq_s32(d10, vld1q_s8(q10 + 16), xv01);
            int32x4_t d11 = vdupq_n_s32(0);
            d11 = vdotq_s32(d11, vld1q_s8(q11),      xv10);
            d11 = vdotq_s32(d11, vld1q_s8(q11 + 16), xv11);

            acc00 = vfmaq_n_f32(acc00, vcvtq_f32_s32(d00), f16_to_f32(s00) * xscale[b]);
            acc01 = vfmaq_n_f32(acc01, vcvtq_f32_s32(d01), f16_to_f32(s01) * xscale[b + 1]);
            acc10 = vfmaq_n_f32(acc10, vcvtq_f32_s32(d10), f16_to_f32(s10) * xscale[b]);
            acc11 = vfmaq_n_f32(acc11, vcvtq_f32_s32(d11), f16_to_f32(s11) * xscale[b + 1]);
        }

        if (b < blocks) {
            uint16_t s0, s1;
            memcpy(&s0, row0 + b * 34, sizeof(s0));
            memcpy(&s1, row1 + b * 34, sizeof(s1));
            const int8_t *xqb = xq + b * 32;
            const int8x16_t xv0 = vld1q_s8(xqb);
            const int8x16_t xv1 = vld1q_s8(xqb + 16);
            const int8_t *q0 = (const int8_t *)(row0 + b * 34 + 2);
            const int8_t *q1 = (const int8_t *)(row1 + b * 34 + 2);
            int32x4_t d0 = vdupq_n_s32(0);
            d0 = vdotq_s32(d0, vld1q_s8(q0),      xv0);
            d0 = vdotq_s32(d0, vld1q_s8(q0 + 16), xv1);
            int32x4_t d1 = vdupq_n_s32(0);
            d1 = vdotq_s32(d1, vld1q_s8(q1),      xv0);
            d1 = vdotq_s32(d1, vld1q_s8(q1 + 16), xv1);
            acc00 = vfmaq_n_f32(acc00, vcvtq_f32_s32(d0), f16_to_f32(s0) * xscale[b]);
            acc10 = vfmaq_n_f32(acc10, vcvtq_f32_s32(d1), f16_to_f32(s1) * xscale[b]);
        }

        *out0 = vaddvq_f32(vaddq_f32(acc00, acc01));
        *out1 = vaddvq_f32(vaddq_f32(acc10, acc11));
        return;
    }
#endif

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    for (uint64_t b = 0; b < blocks; b++) {
        uint16_t s0_bits;
        uint16_t s1_bits;
        memcpy(&s0_bits, row0 + b * 34, sizeof(s0_bits));
        memcpy(&s1_bits, row1 + b * 34, sizeof(s1_bits));
        const int8_t *q0 = (const int8_t *)(row0 + b * 34 + 2);
        const int8_t *q1 = (const int8_t *)(row1 + b * 34 + 2);
        const uint64_t i0 = b * 32;
        const uint64_t n = in_dim - i0 < 32 ? in_dim - i0 : 32;
        acc0 += f16_to_f32(s0_bits) * xscale[b] * (float)dot_i8_32(q0, xq + i0, n);
        acc1 += f16_to_f32(s1_bits) * xscale[b] * (float)dot_i8_32(q1, xq + i0, n);
    }
    *out0 = acc0;
    *out1 = acc1;
}



void quantize_q8_0_activation(const float *x, int8_t *xq, float *scale, uint64_t n) {
    const uint64_t blocks = (n + 31) / 32;
    for (uint64_t b = 0; b < blocks; b++) {
        const uint64_t i0 = b * 32;
        const uint64_t bn = n - i0 < 32 ? n - i0 : 32;
        float amax = 0.0f;
        for (uint64_t i = 0; i < bn; i++) {
            const float ax = fabsf(x[i0 + i]);
            if (ax > amax) amax = ax;
        }
        const float d = amax / 127.0f;
        const float id = d != 0.0f ? 1.0f / d : 0.0f;
        scale[b] = d;
        for (uint64_t i = 0; i < bn; i++) {
            int v = (int)lrintf(x[i0 + i] * id);
            if (v > 127) v = 127;
            if (v < -128) v = -128;
            xq[i0 + i] = (int8_t)v;
        }
        for (uint64_t i = bn; i < 32 && i0 + i < blocks * 32; i++) {
            xq[i0 + i] = 0;
        }
    }
}



static void quantize_q8_0_batch_worker(void *vctx, uint64_t t0, uint64_t t1) {
    quantize_q8_0_batch_ctx *ctx = static_cast<quantize_q8_0_batch_ctx *>(vctx);
    for (uint64_t t = t0; t < t1; t++) {
        quantize_q8_0_activation(ctx->x + t * ctx->in_dim,
                                 ctx->xq + t * ctx->blocks * 32,
                                 ctx->xscale + t * ctx->blocks,
                                 ctx->in_dim);
    }
}



static void quantize_q8_0_activation_batch(
        const float *x,
        int8_t      *xq,
        float       *xscale,
        uint64_t     n_tok,
        uint64_t     in_dim) {
    quantize_q8_0_batch_ctx ctx = {
        .x = x,
        .xq = xq,
        .xscale = xscale,
        .in_dim = in_dim,
        .blocks = (in_dim + 31) / 32,
    };
    pulsar_parallel_for(n_tok, quantize_q8_0_batch_worker, &ctx);
}



static void matvec_q8_0_worker(void *vctx, uint64_t r0, uint64_t r1) {
    matvec_q8_0_ctx *ctx = static_cast<matvec_q8_0_ctx *>(vctx);

    for (uint64_t r = r0; r < r1; r++) {
        const uint64_t o = ctx->row0 + r;
        const uint8_t *row = ctx->data + o * ctx->blocks * 34;
        ctx->out[r] = dot_q8_0_row(row, ctx->xq, ctx->xscale, ctx->in_dim, ctx->blocks);
    }
}



static void matvec_q8_0_pair_worker(void *vctx, uint64_t r0, uint64_t r1) {
    matvec_q8_0_pair_ctx *ctx = static_cast<matvec_q8_0_pair_ctx *>(vctx);

    for (uint64_t r = r0; r < r1; r++) {
        const uint8_t *row0 = ctx->data0 + r * ctx->blocks * 34;
        const uint8_t *row1 = ctx->data1 + r * ctx->blocks * 34;
        dot_q8_0_row_pair(row0, row1, ctx->xq, ctx->xscale, ctx->in_dim, ctx->blocks,
                          ctx->out0 + r, ctx->out1 + r);
    }
}



static void matvec_q8_0_grouped_worker(void *vctx, uint64_t r0, uint64_t r1) {
    matvec_q8_0_grouped_ctx *ctx = static_cast<matvec_q8_0_grouped_ctx *>(vctx);

    for (uint64_t idx = r0; idx < r1; idx++) {
        const uint64_t group = idx / ctx->rank;
        const uint64_t row_in_group = idx - group * ctx->rank;
        const uint64_t tensor_row = group * ctx->rank + row_in_group;
        const uint8_t *row = ctx->data + tensor_row * ctx->blocks * 34;
        const int8_t *xq = ctx->xq + group * ctx->blocks * 32;
        const float *xscale = ctx->xscale + group * ctx->blocks;
        ctx->out[idx] = dot_q8_0_row(row, xq, xscale, ctx->in_dim, ctx->blocks);
    }
}



static void matmul_q8_0_grouped_batch_worker(void *vctx, uint64_t r0, uint64_t r1) {
    matmul_q8_0_grouped_batch_ctx *ctx = static_cast<matmul_q8_0_grouped_batch_ctx *>(vctx);

    for (uint64_t idx = r0; idx < r1; idx++) {
        const uint64_t group = idx / ctx->rank;
        const uint64_t row_in_group = idx - group * ctx->rank;
        const uint64_t tensor_row = group * ctx->rank + row_in_group;
        const uint8_t *row = ctx->data + tensor_row * ctx->blocks * 34;

        uint64_t t = 0;
        for (; t + 1 < ctx->n_tok; t += 2) {
            const uint64_t xbase0 = (t * ctx->n_groups + group) * ctx->blocks;
            const uint64_t xbase1 = ((t + 1) * ctx->n_groups + group) * ctx->blocks;
            dot_q8_0_row_2(row,
                           ctx->xq + xbase0 * 32,
                           ctx->xscale + xbase0,
                           ctx->xq + xbase1 * 32,
                           ctx->xscale + xbase1,
                           ctx->group_dim,
                           ctx->blocks,
                           ctx->out + t * ctx->n_groups * ctx->rank + group * ctx->rank + row_in_group,
                           ctx->out + (t + 1) * ctx->n_groups * ctx->rank + group * ctx->rank + row_in_group);
        }
        for (; t < ctx->n_tok; t++) {
            const uint64_t xbase = (t * ctx->n_groups + group) * ctx->blocks;
            ctx->out[t * ctx->n_groups * ctx->rank + group * ctx->rank + row_in_group] =
                dot_q8_0_row(row,
                             ctx->xq + xbase * 32,
                             ctx->xscale + xbase,
                             ctx->group_dim,
                             ctx->blocks);
        }
    }
}



static void matmul_q8_0_batch_worker(void *vctx, uint64_t r0, uint64_t r1) {
    matmul_q8_0_batch_ctx *ctx = static_cast<matmul_q8_0_batch_ctx *>(vctx);

    for (uint64_t r = r0; r < r1; r++) {
        const uint8_t *row = ctx->data + r * ctx->blocks * 34;
        uint64_t t = 0;
        for (; t + 1 < ctx->n_tok; t += 2) {
            dot_q8_0_row_2(row,
                           ctx->xq + t * ctx->blocks * 32,
                           ctx->xscale + t * ctx->blocks,
                           ctx->xq + (t + 1) * ctx->blocks * 32,
                           ctx->xscale + (t + 1) * ctx->blocks,
                           ctx->in_dim,
                           ctx->blocks,
                           ctx->out + t * ctx->out_dim + r,
                           ctx->out + (t + 1) * ctx->out_dim + r);
        }
        for (; t < ctx->n_tok; t++) {
            ctx->out[t * ctx->out_dim + r] =
                dot_q8_0_row(row,
                             ctx->xq + t * ctx->blocks * 32,
                             ctx->xscale + t * ctx->blocks,
                             ctx->in_dim,
                             ctx->blocks);
        }
    }
}



static void matmul_q8_0_pair_batch_worker(void *vctx, uint64_t r0, uint64_t r1) {
    matmul_q8_0_pair_batch_ctx *ctx = static_cast<matmul_q8_0_pair_batch_ctx *>(vctx);

    for (uint64_t r = r0; r < r1; r++) {
        const uint8_t *row0 = ctx->data0 + r * ctx->blocks * 34;
        const uint8_t *row1 = ctx->data1 + r * ctx->blocks * 34;
        uint64_t t = 0;
        for (; t + 1 < ctx->n_tok; t += 2) {
            const int8_t *xq0 = ctx->xq + t * ctx->blocks * 32;
            const float *xscale0 = ctx->xscale + t * ctx->blocks;
            const int8_t *xq1 = ctx->xq + (t + 1) * ctx->blocks * 32;
            const float *xscale1 = ctx->xscale + (t + 1) * ctx->blocks;
            dot_q8_0_row_2(row0, xq0, xscale0, xq1, xscale1, ctx->in_dim, ctx->blocks,
                           ctx->out0 + t * ctx->out_dim + r,
                           ctx->out0 + (t + 1) * ctx->out_dim + r);
            dot_q8_0_row_2(row1, xq0, xscale0, xq1, xscale1, ctx->in_dim, ctx->blocks,
                           ctx->out1 + t * ctx->out_dim + r,
                           ctx->out1 + (t + 1) * ctx->out_dim + r);
        }
        for (; t < ctx->n_tok; t++) {
            const int8_t *xq = ctx->xq + t * ctx->blocks * 32;
            const float *xscale = ctx->xscale + t * ctx->blocks;
            dot_q8_0_row_pair(row0, row1, xq, xscale, ctx->in_dim, ctx->blocks,
                              ctx->out0 + t * ctx->out_dim + r,
                              ctx->out1 + t * ctx->out_dim + r);
        }
    }
}



/* Multiply selected Q8_0 rows by an activation that has already been quantized
 * once.  This avoids repeated activation quantization for paired projections. */
static void matvec_q8_0_rows_prequant(
        float           * out,
        const pulsar_model * m,
        const pulsar_tensor * w,
        const int8_t    * xq,
        const float     * xscale,
        uint64_t          row0,
        uint64_t          n_rows) {
    if (w->type != 8 || w->ndim != 2) pulsar_die("expected a 2D Q8_0 tensor");

    const uint64_t in_dim = w->dim[0];
    const uint64_t out_dim = w->dim[1];
    if (row0 > out_dim || n_rows > out_dim - row0) pulsar_die("Q8_0 row range is outside tensor");
    const uint64_t ctx_blocks = (in_dim + 31) / 32;

    matvec_q8_0_ctx ctx = {
        .out = out,
        .data = (const uint8_t *)tensor_data(m, w),
        .xq = xq,
        .xscale = xscale,
        .in_dim = in_dim,
        .row0 = row0,
        .blocks = ctx_blocks,
    };
    pulsar_parallel_for(n_rows, matvec_q8_0_worker, &ctx);
}



static PULSAR_MAYBE_UNUSED void matvec_q8_0_prequant(
        float           * out,
        const pulsar_model * m,
        const pulsar_tensor * w,
        const int8_t    * xq,
        const float     * xscale) {
    matvec_q8_0_rows_prequant(out, m, w, xq, xscale, 0, w->dim[1]);
}



/* Compute two Q8_0 projections from the same input, used by gate/up and
 * compressor kv/score pairs. */
void matvec_q8_0_pair_prequant(
        float           * out0,
        float           * out1,
        const pulsar_model * m,
        const pulsar_tensor * w0,
        const pulsar_tensor * w1,
        const int8_t    * xq,
        const float     * xscale) {
    if (w0->type != 8 || w1->type != 8 || w0->ndim != 2 || w1->ndim != 2) {
        pulsar_die("expected two 2D Q8_0 tensors");
    }
    if (w0->dim[0] != w1->dim[0] || w0->dim[1] != w1->dim[1]) {
        pulsar_die("paired Q8_0 tensors do not have the same shape");
    }

    const uint64_t in_dim = w0->dim[0];
    matvec_q8_0_pair_ctx ctx = {
        .out0 = out0,
        .out1 = out1,
        .data0 = (const uint8_t *)tensor_data(m, w0),
        .data1 = (const uint8_t *)tensor_data(m, w1),
        .xq = xq,
        .xscale = xscale,
        .in_dim = in_dim,
        .blocks = (in_dim + 31) / 32,
    };
    pulsar_parallel_for(w0->dim[1], matvec_q8_0_pair_worker, &ctx);
}



static void matmul_q8_0_batch_prequant(
        float           * out,
        const pulsar_model * m,
        const pulsar_tensor * w,
        const int8_t    * xq,
        const float     * xscale,
        uint64_t          n_tok) {
    if (w->type != 8 || w->ndim != 2) pulsar_die("expected a 2D Q8_0 tensor");

    matmul_q8_0_batch_ctx ctx = {
        .out = out,
        .data = (const uint8_t *)tensor_data(m, w),
        .xq = xq,
        .xscale = xscale,
        .n_tok = n_tok,
        .in_dim = w->dim[0],
        .out_dim = w->dim[1],
        .blocks = (w->dim[0] + 31) / 32,
    };
    pulsar_parallel_for(ctx.out_dim, matmul_q8_0_batch_worker, &ctx);
}



static void matmul_q8_0_pair_batch_prequant(
        float           * out0,
        float           * out1,
        const pulsar_model * m,
        const pulsar_tensor * w0,
        const pulsar_tensor * w1,
        const int8_t    * xq,
        const float     * xscale,
        uint64_t          n_tok) {
    if (w0->type != 8 || w1->type != 8 || w0->ndim != 2 || w1->ndim != 2) {
        pulsar_die("expected two 2D Q8_0 tensors");
    }
    if (w0->dim[0] != w1->dim[0] || w0->dim[1] != w1->dim[1]) {
        pulsar_die("paired Q8_0 tensors do not have the same shape");
    }

    matmul_q8_0_pair_batch_ctx ctx = {
        .out0 = out0,
        .out1 = out1,
        .data0 = (const uint8_t *)tensor_data(m, w0),
        .data1 = (const uint8_t *)tensor_data(m, w1),
        .xq = xq,
        .xscale = xscale,
        .n_tok = n_tok,
        .in_dim = w0->dim[0],
        .out_dim = w0->dim[1],
        .blocks = (w0->dim[0] + 31) / 32,
    };
    pulsar_parallel_for(ctx.out_dim, matmul_q8_0_pair_batch_worker, &ctx);
}



/* Batched Q8_0 matmul for prefill: quantize all token activations, then scan
 * weight rows once per output channel. */
void matmul_q8_0_batch(
        float           * out,
        const pulsar_model * m,
        const pulsar_tensor * w,
        const float     * x,
        uint64_t          n_tok) {
    if (w->type != 8 || w->ndim != 2) pulsar_die("expected a 2D Q8_0 tensor");

    const uint64_t in_dim = w->dim[0];
    const uint64_t blocks = (in_dim + 31) / 32;
    int8_t *xq = (int8_t *)xmalloc((size_t)n_tok * blocks * 32);
    float *xscale = (float *)xmalloc((size_t)n_tok * blocks * sizeof(xscale[0]));

    quantize_q8_0_activation_batch(x, xq, xscale, n_tok, in_dim);
    matmul_q8_0_batch_prequant(out, m, w, xq, xscale, n_tok);

    free(xscale);
    free(xq);
}



void matmul_q8_0_pair_batch(
        float           * out0,
        float           * out1,
        const pulsar_model * m,
        const pulsar_tensor * w0,
        const pulsar_tensor * w1,
        const float     * x,
        uint64_t          n_tok) {
    if (w0->type != 8 || w1->type != 8 || w0->ndim != 2 || w1->ndim != 2) {
        pulsar_die("expected two 2D Q8_0 tensors");
    }
    if (w0->dim[0] != w1->dim[0] || w0->dim[1] != w1->dim[1]) {
        pulsar_die("paired Q8_0 tensors do not have the same shape");
    }

    const uint64_t in_dim = w0->dim[0];
    const uint64_t blocks = (in_dim + 31) / 32;
    int8_t *xq = (int8_t *)xmalloc((size_t)n_tok * blocks * 32);
    float *xscale = (float *)xmalloc((size_t)n_tok * blocks * sizeof(xscale[0]));

    quantize_q8_0_activation_batch(x, xq, xscale, n_tok, in_dim);
    matmul_q8_0_pair_batch_prequant(out0, out1, m, w0, w1, xq, xscale, n_tok);

    free(xscale);
    free(xq);
}



static void matvec_q8_0_rows(
        float           * out,
        const pulsar_model * m,
        const pulsar_tensor * w,
        const float     * x,
        uint64_t          row0,
        uint64_t          n_rows) {
    if (w->type != 8 || w->ndim != 2) pulsar_die("expected a 2D Q8_0 tensor");

    const uint64_t in_dim = w->dim[0];
    const uint64_t ctx_blocks = (in_dim + 31) / 32;
    int8_t *xq = (int8_t *)xmalloc((size_t)ctx_blocks * 32);
    float *xscale = (float *)xmalloc((size_t)ctx_blocks * sizeof(xscale[0]));

    quantize_q8_0_activation(x, xq, xscale, in_dim);
    matvec_q8_0_rows_prequant(out, m, w, xq, xscale, row0, n_rows);

    free(xscale);
    free(xq);
}



/* Single-token Q8_0 matvec, used heavily in decode. */
void matvec_q8_0(float *out, const pulsar_model *m, const pulsar_tensor *w, const float *x) {
    matvec_q8_0_rows(out, m, w, x, 0, w->dim[1]);
}



void matvec_any(float *out, const pulsar_model *m, const pulsar_tensor *w, const float *x);



/* Decode scratch owns this temporary activation quantization so generation
 * can assert that the hot path performs no malloc. */












void matvec_q8_0_grouped_rows(
        float           * out,
        const pulsar_model * m,
        const pulsar_tensor * w,
        const float     * x,
        uint32_t          n_groups,
        uint64_t          group_dim,
        uint64_t          rank) {
    if (w->type != 8 || w->ndim != 2) pulsar_die("expected a 2D Q8_0 tensor");
    if (w->dim[0] != group_dim || w->dim[1] < (uint64_t)n_groups * rank) {
        pulsar_die("grouped Q8_0 tensor has an unexpected layout");
    }

    const uint64_t blocks = (group_dim + 31) / 32;
    int8_t *xq = (int8_t *)xmalloc((size_t)n_groups * blocks * 32);
    float *xscale = (float *)xmalloc((size_t)n_groups * blocks * sizeof(xscale[0]));

    for (uint32_t g = 0; g < n_groups; g++) {
        quantize_q8_0_activation(x + (uint64_t)g * group_dim,
                                 xq + (uint64_t)g * blocks * 32,
                                 xscale + (uint64_t)g * blocks,
                                 group_dim);
    }

    matvec_q8_0_grouped_ctx ctx = {
        .out = out,
        .data = (const uint8_t *)tensor_data(m, w),
        .xq = xq,
        .xscale = xscale,
        .in_dim = group_dim,
        .blocks = blocks,
        .rank = rank,
    };
    pulsar_parallel_for((uint64_t)n_groups * rank, matvec_q8_0_grouped_worker, &ctx);

    free(xscale);
    free(xq);
}






void matmul_q8_0_grouped_batch(
        float           * out,
        const pulsar_model * m,
        const pulsar_tensor * w,
        const float     * x,
        uint64_t          n_tok,
        uint32_t          n_groups,
        uint64_t          group_dim,
        uint64_t          rank) {
    if (w->type != 8 || w->ndim != 2) pulsar_die("expected a 2D Q8_0 tensor");
    if (w->dim[0] != group_dim || w->dim[1] < (uint64_t)n_groups * rank) {
        pulsar_die("grouped Q8_0 tensor has an unexpected layout");
    }

    const uint64_t blocks = (group_dim + 31) / 32;
    int8_t *xq = (int8_t *)xmalloc((size_t)n_tok * n_groups * blocks * 32);
    float *xscale = (float *)xmalloc((size_t)n_tok * n_groups * blocks * sizeof(xscale[0]));

    for (uint64_t t = 0; t < n_tok; t++) {
        for (uint32_t g = 0; g < n_groups; g++) {
            const uint64_t xbase = (t * n_groups + g) * blocks;
            quantize_q8_0_activation(x + t * n_groups * group_dim + (uint64_t)g * group_dim,
                                     xq + xbase * 32,
                                     xscale + xbase,
                                     group_dim);
        }
    }

    matmul_q8_0_grouped_batch_ctx ctx = {
        .out = out,
        .data = (const uint8_t *)tensor_data(m, w),
        .xq = xq,
        .xscale = xscale,
        .n_tok = n_tok,
        .n_groups = n_groups,
        .group_dim = group_dim,
        .blocks = blocks,
        .rank = rank,
    };
    pulsar_parallel_for((uint64_t)n_groups * rank, matmul_q8_0_grouped_batch_worker, &ctx);

    free(xscale);
    free(xq);
}



static void matvec_f32_worker(void *vctx, uint64_t row0, uint64_t row1) {
    matvec_f32_ctx *ctx = static_cast<matvec_f32_ctx *>(vctx);

    for (uint64_t o = row0; o < row1; o++) {
        double acc = 0.0;
        const float *row = ctx->data + o * ctx->in_dim;
        for (uint64_t i = 0; i < ctx->in_dim; i++) {
            acc += (double)row[i] * ctx->x[i];
        }
        ctx->out[o] = (float)acc;
    }
}



static void matvec_f32(float *out, const pulsar_model *m, const pulsar_tensor *w, const float *x) {
    if (w->type != 0 || w->ndim != 2) pulsar_die("expected a 2D F32 tensor");

    matvec_f32_ctx ctx = {
        .out = out,
        .data = (const float *)tensor_data(m, w),
        .x = x,
        .in_dim = w->dim[0],
    };
    pulsar_parallel_for(w->dim[1], matvec_f32_worker, &ctx);
}



/* Dispatch for dense F32/F16/Q8_0 tensors used by auxiliary projections. */
void matvec_any(float *out, const pulsar_model *m, const pulsar_tensor *w, const float *x) {
    switch (w->type) {
    case 0: matvec_f32(out, m, w, x); break;
    case 1: matvec_f16(out, m, w, x); break;
    case 8: matvec_q8_0(out, m, w, x); break;
    default:
        pulsar_die("unsupported tensor type for dense matvec");
    }
}



float tensor_1d_value(const pulsar_model *m, const pulsar_tensor *t, uint64_t i) {
    if (i >= t->elements) pulsar_die("tensor scalar index is out of bounds");
    if (t->type == 0) {
        const float *p = (const float *)tensor_data(m, t);
        return p[i];
    }
    if (t->type == 1) {
        const uint16_t *p = (const uint16_t *)tensor_data(m, t);
        return f16_to_f32(p[i]);
    }
    pulsar_die("unsupported tensor scalar type");
    return 0.0f;
}



float tensor_2d_value(const pulsar_model *m, const pulsar_tensor *t, uint64_t x, uint64_t y) {
    if (t->ndim != 2 || x >= t->dim[0] || y >= t->dim[1]) {
        pulsar_die("tensor 2D index is out of bounds");
    }
    return tensor_1d_value(m, t, y * t->dim[0] + x);
}



/* Locate one expert's 2D matrix inside a 3D GGUF expert tensor. */
const uint8_t *tensor_expert_bytes(
        const pulsar_model  *m,
        const pulsar_tensor *w,
        uint32_t          expert,
        uint64_t         *in_dim,
        uint64_t         *out_dim,
        uint64_t         *row_bytes) {
    if (w->ndim != 3) pulsar_die("expected a 3D expert tensor");
    if (expert >= w->dim[2]) pulsar_die("expert id is outside expert tensor");

    *in_dim = w->dim[0];
    *out_dim = w->dim[1];

    const gguf_type_info *info = tensor_type(w->type);
    if (!info || info->block_elems == 0) pulsar_die("unsupported expert tensor type");
    const uint64_t blocks = (*in_dim + info->block_elems - 1) / info->block_elems;
    *row_bytes = blocks * info->block_bytes;

    const uint64_t expert_bytes = *out_dim * *row_bytes;
    return (const uint8_t *)tensor_data(m, w) + (uint64_t)expert * expert_bytes;
}



static void matvec_iq2_xxs_pair_worker(void *vctx, uint64_t row0, uint64_t row1) {
    matvec_iq2_xxs_pair_ctx *ctx = static_cast<matvec_iq2_xxs_pair_ctx *>(vctx);
    for (uint64_t row = row0; row < row1; row++) {
        const block_iq2_xxs *br0 = (const block_iq2_xxs *)(ctx->base0 + row * ctx->row_bytes0);
        const block_iq2_xxs *br1 = (const block_iq2_xxs *)(ctx->base1 + row * ctx->row_bytes1);
        pulsar_vec_dot_iq2_xxs_pair_q8_K((int)ctx->in_dim, &ctx->out0[row], &ctx->out1[row], br0, br1, ctx->xq);
    }
}



/* Project one routed expert's gate and up matrices.  Both are IQ2_XXS and
 * share the same Q8_K activation. */
void matvec_iq2_xxs_expert_pair_prequant(
        float            *out0,
        float            *out1,
        const pulsar_model  *m,
        const pulsar_tensor *w0,
        const pulsar_tensor *w1,
        const block_q8_K *xq,
        uint32_t          expert) {
    if (w0->type != 16 || w1->type != 16) pulsar_die("expected IQ2_XXS expert tensors");

    uint64_t in_dim0, out_dim0, row_bytes0;
    uint64_t in_dim1, out_dim1, row_bytes1;
    const uint8_t *base0 = tensor_expert_bytes(m, w0, expert, &in_dim0, &out_dim0, &row_bytes0);
    const uint8_t *base1 = tensor_expert_bytes(m, w1, expert, &in_dim1, &out_dim1, &row_bytes1);
    if (in_dim0 != in_dim1 || out_dim0 != out_dim1) pulsar_die("paired IQ2_XXS expert tensors do not match");
    if (in_dim0 % QK_K != 0) pulsar_die("IQ2_XXS expert row is not QK_K aligned");

    matvec_iq2_xxs_pair_ctx ctx = {
        .out0 = out0,
        .out1 = out1,
        .base0 = base0,
        .base1 = base1,
        .xq = xq,
        .in_dim = in_dim0,
        .row_bytes0 = row_bytes0,
        .row_bytes1 = row_bytes1,
    };
    pulsar_parallel_for(out_dim0, matvec_iq2_xxs_pair_worker, &ctx);
}



float silu(float x);



static void matvec_iq2_xxs_mid_worker(void *vctx, uint64_t row0, uint64_t row1) {
    matvec_iq2_xxs_mid_ctx *ctx = static_cast<matvec_iq2_xxs_mid_ctx *>(vctx);

    for (uint64_t idx = row0; idx < row1; idx++) {
        const int slot = (int)(idx / ctx->out_dim);
        const uint64_t row = idx - (uint64_t)slot * ctx->out_dim;
        float gate = 0.0f;
        float up = 0.0f;

        const block_iq2_xxs *gate_row = (const block_iq2_xxs *)(ctx->gate_base[slot] + row * ctx->gate_row_bytes[slot]);
        const block_iq2_xxs *up_row = (const block_iq2_xxs *)(ctx->up_base[slot] + row * ctx->up_row_bytes[slot]);
        pulsar_vec_dot_iq2_xxs_pair_q8_K((int)ctx->in_dim, &gate, &up, gate_row, up_row, ctx->xq);

        if (ctx->clamp > 1.0e-6f) {
            if (gate > ctx->clamp) gate = ctx->clamp;
            if (up > ctx->clamp) up = ctx->clamp;
            if (up < -ctx->clamp) up = -ctx->clamp;
        }
        ctx->mid[idx] = silu(gate) * up * ctx->expert_weight[slot];
    }
}



/* Build all selected expert hidden vectors: IQ2_XXS gate/up, clamp, SwiGLU,
 * and router weight.  The down projection runs later on the quantized mids. */
void matvec_iq2_xxs_experts_mid_prequant(
        float            *mid,
        const pulsar_model  *m,
        const pulsar_tensor *gate_w,
        const pulsar_tensor *up_w,
        const block_q8_K *xq,
        const int        *selected,
        const float      *expert_weight,
        int               n_expert,
        float             clamp) {
    if (gate_w->type != 16 || up_w->type != 16) pulsar_die("expected IQ2_XXS expert tensors");
    if (n_expert < 1 || (uint32_t)n_expert > PULSAR_N_EXPERT_USED) pulsar_die("unexpected routed expert count");

    uint64_t in_dim0 = 0;
    uint64_t out_dim0 = 0;
    matvec_iq2_xxs_mid_ctx ctx = {
        .mid = mid,
        .xq = xq,
        .clamp = clamp,
        .n_expert = n_expert,
    };

    for (int i = 0; i < n_expert; i++) {
        uint64_t gate_in_dim, gate_out_dim;
        uint64_t up_in_dim, up_out_dim;
        ctx.gate_base[i] = tensor_expert_bytes(m, gate_w, (uint32_t)selected[i],
                                               &gate_in_dim, &gate_out_dim, &ctx.gate_row_bytes[i]);
        ctx.up_base[i] = tensor_expert_bytes(m, up_w, (uint32_t)selected[i],
                                             &up_in_dim, &up_out_dim, &ctx.up_row_bytes[i]);
        if (gate_in_dim != up_in_dim || gate_out_dim != up_out_dim) {
            pulsar_die("paired IQ2_XXS expert tensors do not match");
        }
        if (i == 0) {
            in_dim0 = gate_in_dim;
            out_dim0 = gate_out_dim;
        } else if (gate_in_dim != in_dim0 || gate_out_dim != out_dim0) {
            pulsar_die("IQ2_XXS expert tensors do not share a layout");
        }
        ctx.expert_weight[i] = expert_weight[i];
    }
    if (in_dim0 % QK_K != 0) pulsar_die("IQ2_XXS expert row is not QK_K aligned");

    ctx.in_dim = in_dim0;
    ctx.out_dim = out_dim0;
    pulsar_parallel_for((uint64_t)n_expert * out_dim0, matvec_iq2_xxs_mid_worker, &ctx);
}



static void matvec_q2_k_worker(void *vctx, uint64_t row0, uint64_t row1) {
    matvec_q2_k_ctx *ctx = static_cast<matvec_q2_k_ctx *>(vctx);
    for (uint64_t row = row0; row < row1; row++) {
        const block_q2_K *br = (const block_q2_K *)(ctx->base + row * ctx->row_bytes);
        pulsar_vec_dot_q2_K_q8_K((int)ctx->in_dim, &ctx->out[row], br, ctx->xq);
    }
}



/* Single expert Q2_K down projection, kept mostly for tracing and diagnostics. */
void matvec_q2_k_expert(
        float            *out,
        const pulsar_model  *m,
        const pulsar_tensor *w,
        const float      *x,
        uint32_t          expert) {
    if (w->type != 10) pulsar_die("expected a Q2_K expert tensor");

    uint64_t in_dim, out_dim, row_bytes;
    const uint8_t *base = tensor_expert_bytes(m, w, expert, &in_dim, &out_dim, &row_bytes);
    if (in_dim % QK_K != 0) pulsar_die("Q2_K expert row is not QK_K aligned");

    block_q8_K *xq = (block_q8_K *)xmalloc((size_t)(in_dim / QK_K) * sizeof(xq[0]));
    pulsar_quantize_row_q8_K(x, xq, (int64_t)in_dim);

    matvec_q2_k_ctx ctx = {
        .out = out,
        .base = base,
        .xq = xq,
        .in_dim = in_dim,
        .row_bytes = row_bytes,
    };
    pulsar_parallel_for(out_dim, matvec_q2_k_worker, &ctx);

    free(xq);
}



static void matvec_q2_k_accum_worker(void *vctx, uint64_t row0, uint64_t row1) {
    matvec_q2_k_accum_ctx *ctx = static_cast<matvec_q2_k_accum_ctx *>(vctx);

    for (uint64_t row = row0; row < row1; row++) {
        float acc = 0.0f;
        for (int i = 0; i < ctx->n_expert; i++) {
            float v = 0.0f;
            const block_q2_K *br = (const block_q2_K *)(ctx->base[i] + row * ctx->row_bytes[i]);
            pulsar_vec_dot_q2_K_q8_K((int)ctx->in_dim, &v, br, ctx->xq[i]);
            acc += v;
        }
        ctx->out[row] = acc;
    }
}



/* Accumulate all selected experts' Q2_K down projections directly into the
 * 4096-wide MoE output. */
void matvec_q2_k_experts_accum_prequant(
        float            *out,
        const pulsar_model  *m,
        const pulsar_tensor *w,
        const block_q8_K *xq,
        const int        *selected,
        int               n_expert) {
    if (w->type != 10) pulsar_die("expected a Q2_K expert tensor");
    if (n_expert < 1 || (uint32_t)n_expert > PULSAR_N_EXPERT_USED) pulsar_die("unexpected routed expert count");

    uint64_t in_dim0 = 0;
    uint64_t out_dim0 = 0;
    const uint8_t *base[PULSAR_MAX_EXPERT_USED];
    uint64_t row_bytes[PULSAR_MAX_EXPERT_USED];

    for (int i = 0; i < n_expert; i++) {
        uint64_t in_dim, out_dim;
        base[i] = tensor_expert_bytes(m, w, (uint32_t)selected[i], &in_dim, &out_dim, &row_bytes[i]);
        if (i == 0) {
            in_dim0 = in_dim;
            out_dim0 = out_dim;
        } else if (in_dim != in_dim0 || out_dim != out_dim0) {
            pulsar_die("Q2_K expert tensors do not share a layout");
        }
    }
    if (in_dim0 % QK_K != 0) pulsar_die("Q2_K expert row is not QK_K aligned");

    const uint64_t n_blocks = in_dim0 / QK_K;
    matvec_q2_k_accum_ctx ctx = {
        .out = out,
        .in_dim = in_dim0,
        .n_expert = n_expert,
    };
    for (int i = 0; i < n_expert; i++) {
        ctx.base[i] = base[i];
        ctx.row_bytes[i] = row_bytes[i];
        ctx.xq[i] = xq + (uint64_t)i * n_blocks;
    }

    pulsar_parallel_for(out_dim0, matvec_q2_k_accum_worker, &ctx);
}



void matvec_iq2_xxs_batch_mid_worker(void *vctx, uint64_t task0, uint64_t task1) {
    matvec_iq2_xxs_batch_mid_ctx *ctx = static_cast<matvec_iq2_xxs_batch_mid_ctx *>(vctx);

    for (uint64_t task = task0; task < task1; task++) {
        const uint32_t active_idx = (uint32_t)(task / ctx->out_dim);
        const uint64_t row = task - (uint64_t)active_idx * ctx->out_dim;
        const uint32_t expert = ctx->active_expert[active_idx];
        const uint32_t begin = ctx->expert_offset[expert];
        const uint32_t end = ctx->expert_offset[expert + 1];

        const block_iq2_xxs *gate_row = (const block_iq2_xxs *)(ctx->gate_base[expert] + row * ctx->gate_row_bytes[expert]);
        const block_iq2_xxs *up_row = (const block_iq2_xxs *)(ctx->up_base[expert] + row * ctx->up_row_bytes[expert]);

        for (uint32_t i = begin; i < end; i++) {
            const uint32_t pair_id = ctx->pair_ids[i];
            const pulsar_expert_pair pair = ctx->pairs[pair_id];
            const block_q8_K *xq = ctx->xq + (uint64_t)pair.token * ctx->xq_blocks;
            float gate = 0.0f;
            float up = 0.0f;

            pulsar_vec_dot_iq2_xxs_pair_q8_K((int)ctx->in_dim, &gate, &up, gate_row, up_row, xq);

            if (ctx->clamp > 1.0e-6f) {
                if (gate > ctx->clamp) gate = ctx->clamp;
                if (up > ctx->clamp) up = ctx->clamp;
                if (up < -ctx->clamp) up = -ctx->clamp;
            }

            ctx->mid[(uint64_t)pair_id * ctx->out_dim + row] = silu(gate) * up * ctx->pair_weight[pair_id];
        }
    }
}



void quantize_mid_pairs_worker(void *vctx, uint64_t p0, uint64_t p1) {
    quantize_mid_pairs_ctx *ctx = static_cast<quantize_mid_pairs_ctx *>(vctx);
    for (uint64_t p = p0; p < p1; p++) {
        pulsar_quantize_row_q8_K(ctx->mid + p * ctx->down_in_dim,
                              ctx->midq + p * ctx->down_blocks,
                              (int64_t)ctx->down_in_dim);
    }
}



static PULSAR_MAYBE_UNUSED void matvec_q2_k_batch_down_worker(void *vctx, uint64_t task0, uint64_t task1) {
    matvec_q2_k_batch_down_ctx *ctx = static_cast<matvec_q2_k_batch_down_ctx *>(vctx);

    for (uint64_t task = task0; task < task1; task++) {
        const uint32_t active_idx = (uint32_t)(task / ctx->out_dim);
        const uint64_t row = task - (uint64_t)active_idx * ctx->out_dim;
        const uint32_t expert = ctx->active_expert[active_idx];
        const uint32_t begin = ctx->expert_offset[expert];
        const uint32_t end = ctx->expert_offset[expert + 1];
        const block_q2_K *br = (const block_q2_K *)(ctx->base[expert] + row * ctx->row_bytes[expert]);

        for (uint32_t i = begin; i < end; i++) {
            const uint32_t pair_id = ctx->pair_ids[i];
            const block_q8_K *xq = ctx->midq + (uint64_t)pair_id * ctx->midq_blocks;
            pulsar_vec_dot_q2_K_q8_K((int)ctx->in_dim,
                                  ctx->down_pair + (uint64_t)pair_id * ctx->out_dim + row,
                                  br, xq);
        }
    }
}



void matvec_q2_k_batch_accum_rows_worker(void *vctx, uint64_t row0, uint64_t row1) {
    matvec_q2_k_batch_accum_rows_ctx *ctx = static_cast<matvec_q2_k_batch_accum_rows_ctx *>(vctx);

    for (uint64_t row = row0; row < row1; row++) {
        for (uint32_t t = 0; t < ctx->n_tok; t++) {
            ctx->moe[(uint64_t)t * ctx->out_dim + row] = 0.0f;
        }

        for (uint32_t ai = 0; ai < ctx->n_active; ai++) {
            const uint32_t expert = ctx->active_expert[ai];
            const uint32_t begin = ctx->expert_offset[expert];
            const uint32_t end = ctx->expert_offset[expert + 1];
            const block_q2_K *br = (const block_q2_K *)(ctx->base[expert] + row * ctx->row_bytes[expert]);

            for (uint32_t i = begin; i < end; i++) {
                const uint32_t pair_id = ctx->pair_ids[i];
                const pulsar_expert_pair pair = ctx->pairs[pair_id];
                const block_q8_K *xq = ctx->midq + (uint64_t)pair_id * ctx->midq_blocks;
                float v = 0.0f;

                pulsar_vec_dot_q2_K_q8_K((int)ctx->in_dim, &v, br, xq);
                ctx->moe[(uint64_t)pair.token * ctx->out_dim + row] += v;
            }
        }
    }
}

