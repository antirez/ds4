/* Thinking Machines "Inkling 2 Mini" (GGUF architecture inkling2_loop, as written by
 * llama.cpp): loader, tokenizer and a CPU reference forward pass.  Included by ds4.c
 * (the code needs its static GGUF, vocabulary and thread-pool helpers).
 *
 * Scope of this file: config validation, tensor binding, byte-level BPE, and an
 * f32 oracle that matches llama.cpp.  There is no GPU graph, session, server or chat
 * template for this family yet: every other entry point refuses with
 * "inkling2: not implemented yet".
 *
 * Model in one paragraph (Mini): 24 physical layers, E=2304, 24 query heads and 8 KV
 * heads of 128, no RoPE.  Positions enter only through a learned bias table over the
 * query-key distance (16 profiles mixed per token and head) and, on global layers, a log
 * scale tau(pos).  Layers 0-4, 6-10, 12-16 and 18-22 are local (window 1024); 5, 11, 17
 * and 23 are global.  Residuals are "block attention residuals": every sub-layer input is
 * a softmax-weighted mix of the stored block starts and the current partial sum.  Layers
 * 6..17 run three times (the loop): pass 1 plain, passes 2 and 3 restart from the prelude
 * output and receive gated messages from the previous pass.  So one token costs 48 layer
 * visits ("slots"); every slot owns its K/V cache and conv tails. */

#define IK_E          2304   /* model width */
#define IK_H          24     /* query heads */
#define IK_HK         8      /* KV heads, three query heads each */
#define IK_D          128    /* head dimension */
#define IK_R          16     /* relative-bias profiles per head */
#define IK_KC         4      /* short-conv taps */
#define IK_NEXP       128
#define IK_TOPK       8
#include "ds4_inkling2_env.h"
#define IK_FF         1152   /* expert hidden width; gate_up is 2*IK_FF rows */
#define IK_NLAYER     24     /* physical layers */
#define IK_NSLOT      48     /* layer visits per token: 24 + 2 extra loop passes of 12 */
#define IK_LOOP_START 6
#define IK_LOOP_END   18
#define IK_LOOP_N     3
#define IK_BLOCK      6      /* layers per attention-residual block */
#define IK_EXTENT     1024   /* bias table length; also the local window */
#define IK_VOCAB      200058 /* real tokens; token_embd/output have 201024 padded rows */
#define IK_VOCAB_PAD  201024
#define IK_CONVC      6656   /* conv channels per slot: K 1024 | V 1024 | attn out 2304 | ffn out 2304 */
#define IK_MAX_IN     4608   /* widest matmul input (loop controller down projection) */
#define IK_TILE       64     /* tokens per matmul tile: x tile stays in cache */
#define IK_MAX_STATES 8      /* block starts + current, with room for the loop bank */

#define IK_EPS        1e-6f
#define IK_QK_CLIP    22.627417f
#define IK_REL_CLIP   8.0f
#define IK_SOFTCAP    30.0f

static bool ik_layer_is_global(int ip) { return ip == 5 || ip == 11 || ip == 17 || ip == 23; }
static int  ik_slot_layer(int slot) { return slot < IK_NLAYER ? slot : IK_LOOP_START + (slot - IK_NLAYER) % (IK_LOOP_END - IK_LOOP_START); }

/* ------------------------------------------------------------------------- */
/* Config and tensor binding                                                 */
/* ------------------------------------------------------------------------- */

#define IK_T(type) (1ull << (type))
#define IK_T_F32    IK_T(DS4_TENSOR_F32)
#define IK_T_DENSE  (IK_T(DS4_TENSOR_F32) | IK_T(DS4_TENSOR_F16) | IK_T(DS4_TENSOR_BF16) | IK_T(DS4_TENSOR_Q8_0))
#define IK_T_EXPERT (IK_T_DENSE | IK_T(DS4_TENSOR_MXFP4) | IK_T(DS4_TENSOR_NVFP4))

typedef struct {
    const ds4_tensor *q, *k, *v, *o, *gate, *rel, *sink;
    const ds4_tensor *q_clip, *rel_bias, *rel_clip, *k_conv, *v_conv, *post_conv, *ffn_post_conv;
    const ds4_tensor *attn_norm;            /* local layers: one weight for the Q, K and V inputs */
    const ds4_tensor *inp_norm[3];          /* global layers: separate Q, K, V input norms */
    const ds4_tensor *dyn_proj[2];          /* global layers: input-dependent K, V conv taps (4 x 8) */
    const ds4_tensor *attn_res_query;       /* depth-mix queries: 1 local, 3 global; absent at layer 0 */
    const ds4_tensor *ffn_res_query, *ffn_norm, *router, *router_bias;
    const ds4_tensor *gate_up, *down;       /* [expert][row][col]; gate rows 0..1151, up rows 1152.. */
    const ds4_tensor *gate_up_scale, *down_scale;  /* NVFP4 file: per-expert multiplier on the matmul output */
    const ds4_tensor *msg_query;            /* looped layers: 4 queries (Q, K, V, FFN) over the previous pass */
    const ds4_tensor *ctl_down[4], *ctl_gain[4], *ctl_forget[4];
} ik_layer;

typedef struct ik_gpu ik_gpu;   /* GPU state and graph: ds4_inkling2_gpu.inc (Metal builds only) */
#ifdef DS4_HAS_INKLING2_GPU
static int iks_selftest(ds4_engine *e, const ds4_tokens *prompt);   /* ds4_inkling2_session.inc */
static int iks_generate_argmax(ds4_engine *e, const ds4_tokens *prompt, int n_predict, int ctx_size,
                               ds4_token_emit_fn emit, ds4_generation_done_fn done, void *emit_ud,
                               ds4_session_progress_fn progress, void *progress_ud);
#endif

typedef struct ds4_inkling2 {
    const ds4_model *m;
    const ds4_tensor *embd, *embd_norm, *out, *out_norm, *out_res_query;
    ik_layer layer[IK_NLAYER];
} ds4_inkling2;

static void ik_die(const char *fmt, const char *a) {
    fprintf(stderr, "ds4: inkling2: ");
    fprintf(stderr, fmt, a);
    fputc('\n', stderr);
    exit(1);
}

static void ik_expect_u32(const ds4_model *m, const char *key, uint32_t want) {
    char k[96];
    snprintf(k, sizeof(k), "inkling2_loop.%s", key);
    const uint32_t got = required_u32(m, k);
    if (got != want) {
        fprintf(stderr, "ds4: inkling2: %s is %u, this build is compiled for %u\n", k, got, want);
        exit(1);
    }
}

static void ik_expect_f32(const ds4_model *m, const char *key, float want) {
    char k[96];
    snprintf(k, sizeof(k), "inkling2_loop.%s", key);
    const float got = required_f32(m, k);
    if (fabsf(got - want) > 1e-6f * fabsf(want)) {
        fprintf(stderr, "ds4: inkling2: %s is %g, this build is compiled for %g\n", k, (double)got, (double)want);
        exit(1);
    }
}

/* Per-layer arrays (head counts, local/global pattern): every entry must equal `want`
 * (the pattern wants 1 for local, 0 for global per ik_layer_is_global). */
static void ik_expect_layer_array(const ds4_model *m, const char *key, bool pattern, uint32_t want) {
    char k[96];
    snprintf(k, sizeof(k), "inkling2_loop.%s", key);
    ds4_array_ref a;
    if (!model_get_array(m, k, &a) || a.len != IK_NLAYER) ik_die("%s must be an array of 24 values", k);
    ds4_cursor c = cursor_at(m, a.data_pos);
    for (uint32_t il = 0; il < IK_NLAYER; il++) {
        uint32_t v = 0;
        if (pattern) {
            uint8_t b = 0;
            if (a.type != GGUF_VALUE_BOOL || !cursor_read(&c, &b, 1)) ik_die("%s must be a bool array", k);
            v = b ? 1u : 0u;
            if (v != (ik_layer_is_global((int)il) ? 0u : 1u)) ik_die("%s differs from the compiled local/global layout", k);
        } else {
            int32_t i32 = 0;
            if (a.type != GGUF_VALUE_INT32 && a.type != GGUF_VALUE_UINT32) ik_die("%s must be an integer array", k);
            if (!cursor_read(&c, &i32, 4)) ik_die("%s is truncated", k);
            v = (uint32_t)i32;
            if (v != want) ik_die("%s differs from the compiled head counts", k);
        }
    }
}

/* Called from config_validate_model when general.architecture is inkling2_loop. */
static void inkling2_config_validate(const ds4_model *m) {
    g_ds4_inkling2 = true;
    ik_expect_u32(m, "block_count", IK_NLAYER);
    ik_expect_u32(m, "embedding_length", IK_E);
    ik_expect_u32(m, "attention.key_length", IK_D);
    ik_expect_u32(m, "attention.value_length", IK_D);
    ik_expect_u32(m, "attention.key_length_swa", IK_D);
    ik_expect_u32(m, "attention.value_length_swa", IK_D);
    ik_expect_u32(m, "attention.sliding_window", IK_EXTENT);
    ik_expect_u32(m, "attention.relative_length", IK_R);
    ik_expect_u32(m, "attention.relative_extent", IK_EXTENT);
    ik_expect_u32(m, "attention.log_scale_floor", 8192);
    ik_expect_u32(m, "shortconv.l_cache", IK_KC);
    ik_expect_u32(m, "attn_res.block_size", IK_BLOCK);
    ik_expect_u32(m, "expert_count", IK_NEXP);
    ik_expect_u32(m, "expert_used_count", IK_TOPK);
    ik_expect_u32(m, "expert_feed_forward_length", IK_FF);
    ik_expect_u32(m, "vocab_size", IK_VOCAB_PAD);
    ik_expect_u32(m, "vocab_size_unpadded", IK_VOCAB);
    ik_expect_u32(m, "rope.dimension_count", 0);
    ik_expect_u32(m, "loop.start", IK_LOOP_START);
    ik_expect_u32(m, "loop.end", IK_LOOP_END);
    ik_expect_u32(m, "loop.count", IK_LOOP_N);
    ik_expect_u32(m, "loop.gate_rank", 64);
    if (required_bool(m, "inkling2_loop.loop.varying")) ik_die("%s", "loop.varying models are not supported");
    ik_expect_f32(m, "attention.layer_norm_rms_epsilon", IK_EPS);
    ik_expect_f32(m, "attention.qk_clip", IK_QK_CLIP);
    ik_expect_f32(m, "attention.relative_clip", IK_REL_CLIP);
    ik_expect_f32(m, "attention.log_scale_alpha", 0.1f);
    ik_expect_f32(m, "expert_weights_scale", 8.0f);
    ik_expect_f32(m, "activation.situ_beta", 4.0f);
    ik_expect_f32(m, "activation.situ_linear_beta", 25.0f);
    ik_expect_f32(m, "final_logit_softcapping", IK_SOFTCAP);
    ik_expect_layer_array(m, "attention.head_count", false, IK_H);
    ik_expect_layer_array(m, "attention.head_count_kv", false, IK_HK);
    ik_expect_layer_array(m, "attention.sliding_window_pattern", true, 0);

    ds4_str s = {0};
    if (!model_get_string(m, "tokenizer.ggml.model", &s) || !ds4_streq(s, "gpt2") ||
        !model_get_string(m, "tokenizer.ggml.pre", &s) || !ds4_streq(s, "inkling2")) {
        ik_die("%s", "tokenizer must be gpt2 byte-level BPE with pre-tokenizer inkling2");
    }
    ds4_array_ref toks;
    if (!model_get_array(m, "tokenizer.ggml.tokens", &toks) || toks.len != IK_VOCAB) {
        ik_die("%s", "tokenizer.ggml.tokens must hold exactly 200058 entries");
    }
}

typedef struct {
    const ds4_model *m;
    uint8_t *claimed;       /* one flag per GGUF tensor: leftovers are rejected */
} ik_bind;

/* Look up one tensor, check ggml shape (ne0, ne1, ne2; missing dims are 1) and
 * type, and mark it as claimed. */
static const ds4_tensor *ik_tensor(ik_bind *b, uint64_t types, uint64_t ne0, uint64_t ne1, uint64_t ne2,
                                   const char *fmt, int layer) {
    char name[96];
    snprintf(name, sizeof(name), fmt, layer);
    const ds4_tensor *t = required_tensor(b->m, name);
    const uint64_t want[3] = {ne0, ne1, ne2};
    bool ok = t->ndim <= 3 && t->type < 64 && (types & IK_T(t->type));
    for (uint32_t i = 0; ok && i < 3; i++) ok = (i < t->ndim ? t->dim[i] : 1) == want[i];
    if (ok) {
        const uint32_t be = tensor_type(t->type)->block_elems;
        ok = ne0 % be == 0;
    }
    if (!ok) {
        fprintf(stderr, "ds4: inkling2: tensor %s has type %s and shape [%" PRIu64 ",%" PRIu64 ",%" PRIu64
                        "], expected shape [%" PRIu64 ",%" PRIu64 ",%" PRIu64 "] and a supported type\n",
                name, tensor_type_name(t->type), t->dim[0], t->ndim > 1 ? t->dim[1] : 1,
                t->ndim > 2 ? t->dim[2] : 1, ne0, ne1, ne2);
        exit(1);
    }
    b->claimed[t - b->m->tensors] = 1;
    return t;
}

static const ds4_tensor *ik_tensor_opt(ik_bind *b, uint64_t types, uint64_t ne0, uint64_t ne1, uint64_t ne2,
                                       const char *fmt, int layer) {
    char name[96];
    snprintf(name, sizeof(name), fmt, layer);
    return model_find_tensor(b->m, name) ? ik_tensor(b, types, ne0, ne1, ne2, fmt, layer) : NULL;
}

static ds4_inkling2 *inkling2_bind(const ds4_model *m) {
    ds4_inkling2 *w = xcalloc(1, sizeof(*w));
    ik_bind b = { m, xcalloc((size_t)m->n_tensors, 1) };
    w->m = m;
    w->embd = ik_tensor(&b, IK_T_DENSE, IK_E, IK_VOCAB_PAD, 1, "token_embd.weight", 0);
    w->embd_norm = ik_tensor(&b, IK_T_F32, IK_E, 1, 1, "token_embd_norm.weight", 0);
    w->out = ik_tensor(&b, IK_T_DENSE, IK_E, IK_VOCAB_PAD, 1, "output.weight", 0);
    w->out_norm = ik_tensor(&b, IK_T_F32, IK_E, 1, 1, "output_norm.weight", 0);
    w->out_res_query = ik_tensor(&b, IK_T_F32, IK_E, 1, 1, "output_res_query.weight", 0);
    for (int il = 0; il < IK_NLAYER; il++) {
        ik_layer *L = &w->layer[il];
        const bool global = ik_layer_is_global(il);
        const bool looped = il >= IK_LOOP_START && il < IK_LOOP_END;
        L->q = ik_tensor(&b, IK_T_DENSE, IK_E, IK_H * IK_D, 1, "blk.%d.attn_q.weight", il);
        L->k = ik_tensor(&b, IK_T_DENSE, IK_E, IK_HK * IK_D, 1, "blk.%d.attn_k.weight", il);
        L->v = ik_tensor(&b, IK_T_DENSE, IK_E, IK_HK * IK_D, 1, "blk.%d.attn_v.weight", il);
        L->o = ik_tensor(&b, IK_T_DENSE, IK_H * IK_D, IK_E, 1, "blk.%d.attn_output.weight", il);
        L->gate = ik_tensor(&b, IK_T_DENSE, IK_E, IK_H * IK_D, 1, "blk.%d.attn_gate.weight", il);
        L->rel = ik_tensor(&b, IK_T_DENSE, IK_E, IK_R * IK_H, 1, "blk.%d.attn_rel.weight", il);
        L->sink = ik_tensor(&b, IK_T_DENSE, IK_E, IK_H, 1, "blk.%d.attn_sink_proj.weight", il);
        L->q_clip = ik_tensor(&b, IK_T_F32, IK_D, IK_H, 1, "blk.%d.attn_q_clip.weight", il);
        L->rel_bias = ik_tensor(&b, IK_T_F32, IK_R, IK_EXTENT, 1, "blk.%d.attn_rel_bias.weight", il);
        L->rel_clip = ik_tensor(&b, IK_T_F32, IK_R, IK_H, 1, "blk.%d.attn_rel_clip.weight", il);
        L->k_conv = ik_tensor(&b, IK_T_F32, IK_KC, IK_HK * IK_D, 1, "blk.%d.attn_k_conv.weight", il);
        L->v_conv = ik_tensor(&b, IK_T_F32, IK_KC, IK_HK * IK_D, 1, "blk.%d.attn_v_conv.weight", il);
        L->post_conv = ik_tensor(&b, IK_T_F32, IK_KC, IK_E, 1, "blk.%d.attn_post_conv.weight", il);
        L->ffn_post_conv = ik_tensor(&b, IK_T_F32, IK_KC, IK_E, 1, "blk.%d.ffn_post_conv.weight", il);
        if (global) {
            L->inp_norm[0] = ik_tensor(&b, IK_T_F32, IK_E, 1, 1, "blk.%d.attn_q_inp_norm.weight", il);
            L->inp_norm[1] = ik_tensor(&b, IK_T_F32, IK_E, 1, 1, "blk.%d.attn_k_inp_norm.weight", il);
            L->inp_norm[2] = ik_tensor(&b, IK_T_F32, IK_E, 1, 1, "blk.%d.attn_v_inp_norm.weight", il);
            L->dyn_proj[0] = ik_tensor(&b, IK_T_DENSE, IK_E, IK_KC * IK_HK, 1, "blk.%d.attn_k_conv_proj.weight", il);
            L->dyn_proj[1] = ik_tensor(&b, IK_T_DENSE, IK_E, IK_KC * IK_HK, 1, "blk.%d.attn_v_conv_proj.weight", il);
        } else {
            L->attn_norm = ik_tensor(&b, IK_T_F32, IK_E, 1, 1, "blk.%d.attn_norm.weight", il);
        }
        if (il > 0) L->attn_res_query = ik_tensor(&b, IK_T_F32, IK_E, global ? 3 : 1, 1, "blk.%d.attn_res_query.weight", il);
        L->ffn_res_query = ik_tensor(&b, IK_T_F32, IK_E, 1, 1, "blk.%d.ffn_res_query.weight", il);
        L->ffn_norm = ik_tensor(&b, IK_T_F32, IK_E, 1, 1, "blk.%d.ffn_norm.weight", il);
        L->router = ik_tensor(&b, IK_T_F32, IK_E, IK_NEXP, 1, "blk.%d.ffn_gate_inp.weight", il);
        L->router_bias = ik_tensor(&b, IK_T_F32, IK_NEXP, 1, 1, "blk.%d.exp_probs_b.bias", il);
        L->gate_up = ik_tensor(&b, IK_T_EXPERT, IK_E, 2 * IK_FF, IK_NEXP, "blk.%d.ffn_gate_up_exps.weight", il);
        L->down = ik_tensor(&b, IK_T_EXPERT, IK_FF, IK_E, IK_NEXP, "blk.%d.ffn_down_exps.weight", il);
        L->gate_up_scale = ik_tensor_opt(&b, IK_T_F32, IK_NEXP, 1, 1, "blk.%d.ffn_gate_up_exps.scale", il);
        L->down_scale = ik_tensor_opt(&b, IK_T_F32, IK_NEXP, 1, 1, "blk.%d.ffn_down_exps.scale", il);
        if (looped) {
            static const char *const ctl[4] = {"q", "k", "v", "ffn"};
            L->msg_query = ik_tensor(&b, IK_T_F32, IK_E, 4, 1, "blk.%d.loop_msg_query.weight", il);
            for (int j = 0; j < 4; j++) {
                char f[64];
                snprintf(f, sizeof(f), "blk.%%d.loop_%s_ctrl_down.weight", ctl[j]);
                L->ctl_down[j] = ik_tensor(&b, IK_T_F32, 2 * IK_E, 64, 1, f, il);
                snprintf(f, sizeof(f), "blk.%%d.loop_%s_ctrl_gain.weight", ctl[j]);
                L->ctl_gain[j] = ik_tensor(&b, IK_T_F32, 64, IK_E, 1, f, il);
                snprintf(f, sizeof(f), "blk.%%d.loop_%s_ctrl_forget.weight", ctl[j]);
                L->ctl_forget[j] = ik_tensor(&b, IK_T_F32, 64, IK_E, 1, f, il);
            }
        }
    }
    for (uint64_t i = 0; i < m->n_tensors; i++) {
        if (!b.claimed[i]) {
            char name[160];
            snprintf(name, sizeof(name), "%.*s", (int)m->tensors[i].name.len, m->tensors[i].name.ptr);
            ik_die("unexpected tensor %s", name);
        }
    }
    free(b.claimed);
    return w;
}

/* ------------------------------------------------------------------------- */
/* Row dequantization and matrix products                                    */
/* ------------------------------------------------------------------------- */

static float ik_ue4m3_lut[128];

/* Unsigned E4M3 block scale of NVFP4: 4 exponent bits (bias 7), 3 mantissa bits; 0 and 0x7f read as 0. */
static void ik_ue4m3_init(void) {
    for (int x = 0; x < 128; x++) {
        const int e = x >> 3, man = x & 7;
        ik_ue4m3_lut[x] = (x == 0 || x == 0x7f) ? 0.0f :
                          e == 0 ? ldexpf((float)man, -9) : ldexpf(1.0f + (float)man / 8.0f, e - 7);
    }
}

/* One weight row as f32.  F32 rows are used in place; everything else is expanded into buf. */
static const float *ik_row_f32(uint32_t type, const uint8_t *row, uint64_t n, float *buf) {
    switch (type) {
    case DS4_TENSOR_F32:
        return (const float *)row;
    case DS4_TENSOR_F16:
        for (uint64_t i = 0; i < n; i++) { uint16_t h; memcpy(&h, row + 2 * i, 2); buf[i] = f16_to_f32(h); }
        return buf;
    case DS4_TENSOR_BF16:
        for (uint64_t i = 0; i < n; i++) {
            uint16_t h; memcpy(&h, row + 2 * i, 2);
            const uint32_t u = (uint32_t)h << 16;
            memcpy(&buf[i], &u, 4);
        }
        return buf;
    case DS4_TENSOR_Q8_0:   /* 32 int8 with one f16 scale */
        for (uint64_t b = 0; b < n / 32; b++, row += 34) {
            uint16_t h; memcpy(&h, row, 2);
            const float d = f16_to_f32(h);
            for (int j = 0; j < 32; j++) buf[b * 32 + j] = d * (float)(int8_t)row[2 + j];
        }
        return buf;
    case DS4_TENSOR_MXFP4:  /* 32 E2M1 nibbles with one E8M0 scale; byte j holds elements j and j+16 */
        for (uint64_t b = 0; b < n / 32; b++, row += 17) {
            const float d = ds4_e8m0_to_f32(row[0]);
            for (int j = 0; j < 16; j++) {
                buf[b * 32 + j] = d * ds4_mxfp4_values[row[1 + j] & 15];
                buf[b * 32 + 16 + j] = d * ds4_mxfp4_values[row[1 + j] >> 4];
            }
        }
        return buf;
    case DS4_TENSOR_NVFP4:  /* 64 E2M1 nibbles: 4 UE4M3 scales (one per 16), then 8 bytes per 16 (j and j+8) */
        for (uint64_t b = 0; b < n / 64; b++, row += 36) {
            for (int s = 0; s < 4; s++) {
                const float d = ik_ue4m3_lut[row[s] & 0x7f];
                const uint8_t *q = row + 4 + 8 * s;
                for (int j = 0; j < 8; j++) {
                    buf[b * 64 + s * 16 + j] = d * ds4_mxfp4_values[q[j] & 15];
                    buf[b * 64 + s * 16 + 8 + j] = d * ds4_mxfp4_values[q[j] >> 4];
                }
            }
        }
        return buf;
    default:
        ik_die("row of unsupported tensor type %s", tensor_type_name(type));
        return NULL;
    }
}

typedef struct {
    uint32_t type;
    const uint8_t *w;       /* first row */
    uint64_t in, row_bytes;
    const float *x;         /* [n][in] */
    int n;
    float *y;               /* [n][out] */
    uint64_t out;
} ik_mm;

static void ik_mm_worker(void *vc, uint64_t r0, uint64_t r1) {
    const ik_mm *c = vc;
    float buf[IK_MAX_IN];
    for (int t0 = 0; t0 < c->n; t0 += IK_TILE) {
        const int nt = c->n - t0 < IK_TILE ? c->n - t0 : IK_TILE;
        for (uint64_t r = r0; r < r1; r++) {
            /* The row is expanded once per token tile and dotted with the whole tile. */
            const float *w = ik_row_f32(c->type, c->w + r * c->row_bytes, c->in, buf);
            /* Tokens are always dotted in groups of four (the last group repeats its final token): every token
             * then sums its products in the same order, so a logit does not depend on which tokens share its
             * batch.  Batch prefill and one-token decoding give identical results. */
            for (int t = 0; t < nt; t += 4) {
                const float *xs[4];
                for (int j = 0; j < 4; j++) xs[j] = c->x + (uint64_t)(t0 + (t + j < nt ? t + j : nt - 1)) * c->in;
                float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
                for (uint64_t k = 0; k < c->in; k++) {
                    const float wk = w[k];
                    s0 += wk * xs[0][k]; s1 += wk * xs[1][k]; s2 += wk * xs[2][k]; s3 += wk * xs[3][k];
                }
                float *y = c->y + (uint64_t)(t0 + t) * c->out + r;
                y[0] = s0;
                if (t + 1 < nt) y[c->out] = s1;
                if (t + 2 < nt) y[2 * c->out] = s2;
                if (t + 3 < nt) y[3 * c->out] = s3;
            }
        }
    }
}

/* y[t][r] = dot(row r of the matrix at w, x[t]) for r < out_rows, rows of type `type` and width `in`. */
static void ik_matmul_rows(uint32_t type, const uint8_t *w, uint64_t in, uint64_t out_rows,
                           const float *x, int n, float *y) {
    uint64_t row_bytes = 0;
    tensor_nbytes(type, in, &row_bytes);
    ik_mm c = { type, w, in, row_bytes, x, n, y, out_rows };
    ds4_parallel_for_min_rows(out_rows, ik_mm_worker, &c, 16);
}

static void ik_matmul(const ds4_model *m, const ds4_tensor *t, uint64_t out_rows, const float *x, int n, float *y) {
    ik_matmul_rows(t->type, tensor_data(m, t), t->dim[0], out_rows, x, n, y);
}

/* One expert's [out][in] slice of a 3-D expert tensor, read into *buf with pread (NULL if the read fails).  Experts are never mapped in
 * bulk: a long prompt touches every expert, and a read-only mapping of the whole file would charge all of it to
 * the process.  Only slices of selected experts are read. */
static const uint8_t *ik_expert(const ds4_model *m, const ds4_tensor *t, uint64_t e, uint8_t **buf, size_t *cap) {
    uint64_t row_bytes = 0;
    tensor_nbytes(t->type, t->dim[0], &row_bytes);
    const size_t bytes = (size_t)(row_bytes * t->dim[1]);
    if (*cap < bytes) { *buf = xrealloc(*buf, bytes); *cap = bytes; }
    for (size_t done = 0; done < bytes;) {
        const ssize_t r = pread(m->fd, *buf + done, bytes - done, (off_t)(t->abs_offset + e * bytes + done));
        if (r < 0 && errno == EINTR) continue;
        if (r <= 0) return NULL;   /* the caller reports it: a served engine must not exit here */
        done += (size_t)r;
    }
    return *buf;
}

/* ------------------------------------------------------------------------- */
/* State                                                                     */
/* ------------------------------------------------------------------------- */

/* Everything that persists between calls, per slot (48 slots, see the header comment):
 *  - K and V rows after conv, clip (K) and RMS norm (V), as IEEE half like llama.cpp's cache;
 *    attention of the same chunk reads them back, so rounding is part of the model.
 *    Local slots are rings of IK_EXTENT + chunk rows: a chunk is written before it is read,
 *    and its first query still needs the row IK_EXTENT-1 positions back.  Global slots keep
 *    every position (row = position).
 *  - the last three conv inputs (before conv) of all 6656 channels, f32, oldest first.
 * Depth-mix block starts, loop messages and router results are recomputed per token and never stored. */
typedef struct {
    uint32_t cap;                 /* positions the global slots can hold */
    uint32_t chunk;               /* largest chunk this state accepts */
    uint32_t pos;                 /* tokens consumed so far */
    uint32_t rows[IK_NSLOT];
    uint16_t *k[IK_NSLOT], *v[IK_NSLOT];
    float *conv[IK_NSLOT];        /* [IK_CONVC][3] */
} ik_state;

static ik_state *ik_state_new(uint32_t cap, uint32_t chunk) {
    ik_state *s = xcalloc(1, sizeof(*s));
    s->cap = cap;
    s->chunk = chunk;
    for (int slot = 0; slot < IK_NSLOT; slot++) {
        s->rows[slot] = ik_layer_is_global(ik_slot_layer(slot)) ? cap : IK_EXTENT + chunk;
        s->k[slot] = xcalloc((size_t)s->rows[slot] * IK_HK * IK_D, 2);
        s->v[slot] = xcalloc((size_t)s->rows[slot] * IK_HK * IK_D, 2);
        s->conv[slot] = xcalloc((size_t)IK_CONVC * (IK_KC - 1), 4);
    }
    return s;
}

static void ik_state_free(ik_state *s) {
    if (!s) return;
    for (int slot = 0; slot < IK_NSLOT; slot++) { free(s->k[slot]); free(s->v[slot]); free(s->conv[slot]); }
    free(s);
}

/* ------------------------------------------------------------------------- */
/* Small kernels                                                             */
/* ------------------------------------------------------------------------- */

static float ik_sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

static void ik_rms(const float *x, float *y, const float *w, int n) {
    float ss = 0;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    const float r = 1.0f / sqrtf(ss / (float)n + IK_EPS);
    for (int i = 0; i < n; i++) y[i] = x[i] * r * (w ? w[i] : 1.0f);
}

/* x * min(1, T/|x|): llama.cpp's l2_norm(x, eps=T) * T. */
static void ik_clip_l2(float *x, int n, float limit) {
    float ss = 0;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    const float nrm = sqrtf(ss), s = limit / (nrm > limit ? nrm : limit);
    for (int i = 0; i < n; i++) x[i] *= s;
}

static double ik_sum(const float *x, size_t n) { double s = 0; for (size_t i = 0; i < n; i++) s += x[i]; return s; }

static const float *ik_f32(const ds4_model *m, const ds4_tensor *t) { return (const float *)tensor_data(m, t); }

typedef struct { const float *const *states; int ns; const float *queries; int nq, n; float *out; } ik_mix_ctx;

static void ik_mix_worker(void *vc, uint64_t t0, uint64_t t1) {
    const ik_mix_ctx *c = vc;
    float norm[IK_MAX_STATES][IK_E];
    for (uint64_t t = t0; t < t1; t++) {
        for (int s = 0; s < c->ns; s++) ik_rms(c->states[s] + t * IK_E, norm[s], NULL, IK_E);
        for (int q = 0; q < c->nq; q++) {
            float sc[IK_MAX_STATES], mx = -1e30f, den = 0;
            for (int s = 0; s < c->ns; s++) {
                float d = 0;
                for (int i = 0; i < IK_E; i++) d += norm[s][i] * c->queries[(uint64_t)q * IK_E + i];
                sc[s] = d / sqrtf((float)IK_E);
                if (sc[s] > mx) mx = sc[s];
            }
            for (int s = 0; s < c->ns; s++) { sc[s] = expf(sc[s] - mx); den += sc[s]; }
            float *o = c->out + ((uint64_t)q * c->n + t) * IK_E;
            for (int i = 0; i < IK_E; i++) o[i] = 0;
            for (int s = 0; s < c->ns; s++) {
                const float a = sc[s] / den;
                for (int i = 0; i < IK_E; i++) o[i] += a * norm[s][i];
            }
        }
    }
}

/* Depth attention: out[q][t] = softmax over states of (rmsn(S_s[t]) . query_q / sqrt(E)) applied to the
 * normalised states.  States are the stored block starts plus the current partial sum. */
static void ik_mix(const float *const *states, int ns, const float *queries, int nq, int n, float *out) {
    ik_mix_ctx c = { states, ns, queries, nq, n, out };
    ds4_parallel_for_min_rows((uint64_t)n, ik_mix_worker, &c, 8);
}

typedef struct { const float *x, *w; float *y; int n_rows, dim; } ik_rms_ctx;

static void ik_rms_worker(void *vc, uint64_t r0, uint64_t r1) {
    const ik_rms_ctx *c = vc;
    for (uint64_t r = r0; r < r1; r++) ik_rms(c->x + r * c->dim, c->y + r * c->dim, c->w, c->dim);
}

static void ik_rms_rows(const float *x, const float *w, float *y, int n_rows, int dim) {
    ik_rms_ctx c = { x, w, y, n_rows, dim };
    ds4_parallel_for_min_rows((uint64_t)n_rows, ik_rms_worker, &c, 16);
}

typedef struct {
    const float *x;         /* [n][C] conv inputs */
    float *tail;            /* [C][3] this region's conv tail */
    const float *w;         /* [C][4], tap 0 = oldest */
    const float *dyn;       /* [n][nh*4] or NULL; dyn[t][head][lag], lag 0 = current token */
    float *out;
    int n, C, hd;
    float scale;
} ik_conv_ctx;

static void ik_conv_worker(void *vc, uint64_t c0, uint64_t c1) {
    const ik_conv_ctx *c = vc;
    const int T = c->n, C = c->C;
    float *col = xmalloc((size_t)(IK_KC - 1 + T) * sizeof(float));   /* one channel: tail (3) then the chunk */
    for (uint64_t ch = c0; ch < c1; ch++) {
        memcpy(col, c->tail + ch * 3, 3 * sizeof(float));
        for (int t = 0; t < T; t++) col[3 + t] = c->x[(uint64_t)t * C + ch];
        const float *w = c->w + ch * IK_KC;
        for (int t = 0; t < T; t++) {
            /* col[t + 3] is x[t]; the window oldest..newest is col[t .. t + 3].  Gathering the window from one
             * flat array keeps the arithmetic identical for every position, whatever the chunk size. */
            float y = w[0] * col[t] + w[1] * col[t + 1] + w[2] * col[t + 2] + w[3] * col[t + 3];
            if (c->dyn) {
                const float *d = c->dyn + (uint64_t)t * (C / c->hd) * IK_KC + (ch / c->hd) * IK_KC;
                for (int lag = 0; lag < IK_KC; lag++) y += d[lag] * col[t + 3 - lag];
            }
            c->out[(uint64_t)t * C + ch] = (col[t + 3] + y) * c->scale;
        }
        memcpy(c->tail + ch * 3, col + T, 3 * sizeof(float));   /* the next chunk starts from the last three inputs */
    }
    free(col);
}

/* Causal depthwise conv + residual: out = (x + conv(x)) / sqrt(K+1), or with input-dependent taps added to
 * the static ones (/ sqrt(2K+1)).  The tail (last three inputs, pre-conv) is advanced by the chunk. */
static void ik_conv(const float *w, float *tail, const float *x, const float *dyn, int hd, int n, int C, float *out) {
    ik_conv_ctx c = { x, tail, w, dyn, out, n, C, hd, dyn ? 1.0f / 3.0f : 1.0f / sqrtf((float)IK_KC + 1.0f) };
    ds4_parallel_for_min_rows((uint64_t)C, ik_conv_worker, &c, 64);
}

/* Attention over the cached half-precision rows. */
static float ik_dot_f16(const float *q, const uint16_t *k, int n) {
    float s = 0;
#if defined(__ARM_NEON)
    float32x4_t acc = vdupq_n_f32(0.0f);
    for (int i = 0; i < n; i += 8) {
        const uint16x8_t h = vld1q_u16(k + i);
        acc = vfmaq_f32(acc, vld1q_f32(q + i), vcvt_f32_f16(vreinterpret_f16_u16(vget_low_u16(h))));
        acc = vfmaq_f32(acc, vld1q_f32(q + i + 4), vcvt_f32_f16(vreinterpret_f16_u16(vget_high_u16(h))));
    }
    s = vaddvq_f32(acc);
#else
    for (int i = 0; i < n; i++) s += q[i] * f16_to_f32(k[i]);
#endif
    return s;
}

static void ik_axpy_f16(float *o, float a, const uint16_t *v, int n) {
#if defined(__ARM_NEON)
    for (int i = 0; i < n; i += 8) {
        const uint16x8_t h = vld1q_u16(v + i);
        vst1q_f32(o + i, vfmaq_n_f32(vld1q_f32(o + i), vcvt_f32_f16(vreinterpret_f16_u16(vget_low_u16(h))), a));
        vst1q_f32(o + i + 4, vfmaq_n_f32(vld1q_f32(o + i + 4), vcvt_f32_f16(vreinterpret_f16_u16(vget_high_u16(h))), a));
    }
#else
    for (int i = 0; i < n; i++) o[i] += a * f16_to_f32(v[i]);
#endif
}

typedef struct {
    const float *Q;         /* [n][H][D], clipped (global layers also times tau) */
    const float *Rr;        /* [n][H][R] relative states, same treatment */
    const float *sink;      /* [n][H] */
    const float *relb;      /* [EXTENT][R]: bias profile per distance */
    const uint16_t *K, *V;  /* slot cache; position p is row p % rows */
    uint32_t rows, p0;
    bool local;
    float *O;               /* [n][H][D] */
} ik_attn_ctx;

static void ik_attn_worker(void *vc, uint64_t i0, uint64_t i1) {
    const ik_attn_ctx *c = vc;
    const uint64_t max_keys = c->local ? IK_EXTENT : c->p0 + (i1 - 1) / IK_H + 1;
    float *s = xmalloc(max_keys * sizeof(float));
    for (uint64_t i = i0; i < i1; i++) {
        const uint64_t t = i / IK_H, h = i % IK_H, g = h / (IK_H / IK_HK);
        const uint64_t p = c->p0 + t, first = c->local && p >= IK_EXTENT ? p - (IK_EXTENT - 1) : 0;
        const float *q = c->Q + i * IK_D, *r = c->Rr + i * IK_R;
        const float sink = c->sink[i];
        float mx = 0.0f;   /* the virtual key has logit 0 */
        for (uint64_t k = first; k <= p; k++) {
            const uint64_t d = p - k;
            float b = 0;
            if (d < IK_EXTENT) for (int j = 0; j < IK_R; j++) b += r[j] * c->relb[d * IK_R + j];
            const uint16_t *kr = c->K + (uint64_t)(k % c->rows) * IK_HK * IK_D + g * IK_D;
            const float v = ik_dot_f16(q, kr, IK_D) / sqrtf((float)IK_D) + b - sink;
            s[k - first] = v;
            if (v > mx) mx = v;
        }
        float den = expf(-mx);
        for (uint64_t k = first; k <= p; k++) { s[k - first] = expf(s[k - first] - mx); den += s[k - first]; }
        float *o = c->O + i * IK_D;
        for (int j = 0; j < IK_D; j++) o[j] = 0;
        for (uint64_t k = first; k <= p; k++)
            ik_axpy_f16(o, s[k - first] / den, c->V + (uint64_t)(k % c->rows) * IK_HK * IK_D + g * IK_D, IK_D);
    }
    free(s);
}

/* ------------------------------------------------------------------------- */
/* Forward pass                                                              */
/* ------------------------------------------------------------------------- */

static FILE *g_ik_dump;   /* per-visit sums and expert ids (diagnostic) */
/* Diagnostic: (visit, token) pairs where the 8th expert is replaced by the 9th, to test whether a near-tied
 * router decision explains a logit difference against llama.cpp. */
static int g_ik_flip[64][2], g_ik_nflip;
static float g_ik_tie = 1e-4f;   /* margin below which the visits dump lists a router decision as a near-tie */
/* Diagnostic DS4_INKLING2_FT_PICKS=file: raw int32 [48 slots][ntok][8] expert ids (another run's router decisions);
 * every router decision is replaced by these (weights still come from this run's own probabilities).  This is how a
 * GPU run is compared with the CPU oracle: routing is the one thing that amplifies tiny rounding differences.
 * DS4_INKLING2_FT_PICKLOG=file lists each decision where the forced picks differ from this run's own top 8. */
static int32_t *g_ik_picks;
static int g_ik_picks_ntok;
static FILE *g_ik_picklog;
/* DS4_INKLING2_PICKS_OUT=file: write every router decision of this run in the same layout (CPU and GPU). */
static int32_t *g_ik_pick_out;
static uint32_t g_ik_pick_out_ntok;

typedef struct {
    const ds4_inkling2 *w;
    ik_state *st;
    int n;                        /* tokens in this chunk */
    uint32_t p0;                  /* absolute position of the first token */
    float *tau;                   /* [n] log scale for global layers */
    const float *blocks[IK_MAX_STATES];   /* block starts of the current pass */
    int nb;
    void **arena;                 /* every stream buffer lives until the end of the chunk */
    int na, visit;
    const char *err;              /* first failure of this chunk (the chunk is abandoned), NULL if none */
} ik_run;

static float *ik_new(ik_run *r, size_t floats) {
    r->arena = xrealloc(r->arena, (size_t)(r->na + 1) * sizeof(void *));
    return r->arena[r->na++] = xmalloc(floats * sizeof(float));
}

/* cur = cur - forget*cur + gain*msg with gates computed from both (rms-normalised) inputs. */
static void ik_control(ik_run *r, const ik_layer *L, int j, float *cur, const float *msg) {
    const ds4_model *m = r->w->m;
    const int n = r->n;
    float *g = xmalloc((size_t)n * 2 * IK_E * sizeof(float));
    float *d = xmalloc((size_t)n * 64 * sizeof(float));
    float *gain = xmalloc((size_t)n * IK_E * sizeof(float));
    float *forget = xmalloc((size_t)n * IK_E * sizeof(float));
    for (int t = 0; t < n; t++) {
        ik_rms(cur + (uint64_t)t * IK_E, g + (uint64_t)t * 2 * IK_E, NULL, IK_E);
        ik_rms(msg + (uint64_t)t * IK_E, g + (uint64_t)t * 2 * IK_E + IK_E, NULL, IK_E);
    }
    ik_matmul(m, L->ctl_down[j], 64, g, n, d);
    ik_matmul(m, L->ctl_gain[j], IK_E, d, n, gain);
    ik_matmul(m, L->ctl_forget[j], IK_E, d, n, forget);
    for (uint64_t i = 0; i < (uint64_t)n * IK_E; i++) cur[i] += -tanhf(forget[i]) * cur[i] + tanhf(gain[i]) * msg[i];
    free(g); free(d); free(gain); free(forget);
}

/* Router + routed experts for the chunk.  Only the selected experts are read: tokens are
 * grouped per expert so each expert slice is streamed once per chunk. */
static void ik_moe(ik_run *r, const ik_layer *L, const float *h, float *out, int slot) {
    const ds4_model *m = r->w->m;
    const int n = r->n;
    float *logits = xmalloc((size_t)n * IK_NEXP * sizeof(float));
    ik_matmul(m, L->router, IK_NEXP, h, n, logits);
    if (g_ik_dump) fprintf(g_ik_dump, "router visit=%d slot=%d ffn_moe_logits=%.6f\n", r->visit, slot, ik_sum(logits, (size_t)n * IK_NEXP));
    const float *bias = ik_f32(m, L->router_bias);
    int *sel = xmalloc((size_t)n * IK_TOPK * sizeof(int));
    float *wt = xmalloc((size_t)n * IK_TOPK * sizeof(float));
    int cnt[IK_NEXP] = {0};
    for (int t = 0; t < n; t++) {
        float p[IK_NEXP], sc[IK_NEXP], sum = 0;
        for (int e = 0; e < IK_NEXP; e++) { p[e] = ik_sigmoid(logits[(uint64_t)t * IK_NEXP + e]); sc[e] = p[e] + bias[e]; }
        bool taken[IK_NEXP] = {false};
        int ninth = -1;
        for (int k = 0; k <= IK_TOPK; k++) {   /* top-8 of p + bias; ties go to the lower expert id */
            int best = -1;
            for (int e = 0; e < IK_NEXP; e++) if (!taken[e] && (best < 0 || sc[e] > sc[best])) best = e;
            taken[best] = true;
            if (k == IK_TOPK) { ninth = best; break; }
            sel[t * IK_TOPK + k] = best;
        }
        if (g_ik_dump && sc[sel[t * IK_TOPK + IK_TOPK - 1]] - sc[ninth] < g_ik_tie)   /* near-tie: llama.cpp may differ here */
            fprintf(g_ik_dump, "tie visit=%d slot=%d tok=%d pos=%u margin=%.3g e8=%d e9=%d\n", r->visit, slot, t, r->p0 + (uint32_t)t,
                    (double)(sc[sel[t * IK_TOPK + IK_TOPK - 1]] - sc[ninth]), sel[t * IK_TOPK + IK_TOPK - 1], ninth);
        for (int i = 0; i < g_ik_nflip; i++)
            if (g_ik_flip[i][0] == r->visit && g_ik_flip[i][1] == t) sel[t * IK_TOPK + IK_TOPK - 1] = ninth;
        if (g_ik_picks) {
            const uint32_t ap = r->p0 + (uint32_t)t;
            if ((int)ap >= g_ik_picks_ntok) { r->err = "pick file too short"; break; }
            const int32_t *pk = g_ik_picks + ((size_t)slot * (size_t)g_ik_picks_ntok + ap) * IK_TOPK;
            const float my8 = sc[sel[t * IK_TOPK + IK_TOPK - 1]];
            int nd = 0;
            float gap = 0;
            for (int k = 0; k < IK_TOPK; k++) {
                if (pk[k] < 0 || pk[k] >= IK_NEXP) { r->err = "bad expert id in the pick file"; break; }
                bool found = false;
                for (int j = 0; j < IK_TOPK; j++) found |= sel[t * IK_TOPK + j] == pk[k];
                if (!found) { nd++; if (my8 - sc[pk[k]] > gap) gap = my8 - sc[pk[k]]; }
            }
            if (r->err) break;
            if (nd && g_ik_picklog)
                fprintf(g_ik_picklog, "pickdiff slot=%d pos=%u n=%d gap=%.3g margin89=%.3g\n", slot, ap, nd, (double)gap, (double)(my8 - sc[ninth]));
            for (int k = 0; k < IK_TOPK; k++) sel[t * IK_TOPK + k] = pk[k];
        }
        if (g_ik_pick_out && r->p0 + (uint32_t)t < g_ik_pick_out_ntok)
            memcpy(g_ik_pick_out + ((size_t)slot * g_ik_pick_out_ntok + r->p0 + (uint32_t)t) * IK_TOPK, sel + t * IK_TOPK, IK_TOPK * sizeof(int32_t));
        for (int k = 0; k < IK_TOPK; k++) { sum += p[sel[t * IK_TOPK + k]]; cnt[sel[t * IK_TOPK + k]]++; }
        for (int k = 0; k < IK_TOPK; k++) wt[t * IK_TOPK + k] = p[sel[t * IK_TOPK + k]] / (sum + 1e-20f) * 8.0f;   /* bias picks, p weighs */
    }
    for (uint64_t i = 0; i < (uint64_t)n * IK_E; i++) out[i] = 0;
    float *hx = xmalloc((size_t)n * IK_E * sizeof(float)), *gu = xmalloc((size_t)n * 2 * IK_FF * sizeof(float));
    float *ac = xmalloc((size_t)n * IK_FF * sizeof(float)), *yy = xmalloc((size_t)n * IK_E * sizeof(float));
    int *tok = xmalloc((size_t)n * sizeof(int)), *kk = xmalloc((size_t)n * sizeof(int));
    uint8_t *wbuf = NULL;
    size_t wcap = 0;
    const float *gs = L->gate_up_scale ? ik_f32(m, L->gate_up_scale) : NULL;
    const float *ds = L->down_scale ? ik_f32(m, L->down_scale) : NULL;
    for (int e = 0; e < IK_NEXP; e++) {
        if (!cnt[e]) continue;
        int mt = 0;
        for (int t = 0; t < n; t++)
            for (int k = 0; k < IK_TOPK; k++)
                if (sel[t * IK_TOPK + k] == e) { tok[mt] = t; kk[mt] = k; memcpy(hx + (uint64_t)mt * IK_E, h + (uint64_t)t * IK_E, IK_E * sizeof(float)); mt++; }
        const uint8_t *wgu = ik_expert(m, L->gate_up, (uint64_t)e, &wbuf, &wcap);
        if (!wgu) { r->err = "cannot read an expert slice"; break; }
        ik_matmul_rows(L->gate_up->type, wgu, IK_E, 2 * IK_FF, hx, mt, gu);
        for (int i = 0; i < mt; i++) {
            float *g = gu + (uint64_t)i * 2 * IK_FF, *u = g + IK_FF;
            const float s = gs ? gs[e] : 1.0f;
            for (int j = 0; j < IK_FF; j++) {   /* SiTU: softcapped gate * sigmoid(gate) * softcapped up */
                const float a = g[j] * s, b = u[j] * s;
                ac[(uint64_t)i * IK_FF + j] = 4.0f * tanhf(a / 4.0f) * ik_sigmoid(a) * (25.0f * tanhf(b / 25.0f));
            }
        }
        const uint8_t *wdn = ik_expert(m, L->down, (uint64_t)e, &wbuf, &wcap);
        if (!wdn) { r->err = "cannot read an expert slice"; break; }
        ik_matmul_rows(L->down->type, wdn, IK_FF, IK_E, ac, mt, yy);
        const float s = ds ? ds[e] : 1.0f;
        for (int i = 0; i < mt; i++) {
            const float w = wt[tok[i] * IK_TOPK + kk[i]] * s;
            float *o = out + (uint64_t)tok[i] * IK_E;
            for (int j = 0; j < IK_E; j++) o[j] += w * yy[(uint64_t)i * IK_E + j];
        }
    }
    if (g_ik_dump) {
        double sum = 0;
        for (int t = 0; t < n; t++) {
            fprintf(g_ik_dump, "ids visit=%d slot=%d tok=%d pos=%u", r->visit, slot, t, r->p0 + (uint32_t)t);
            for (int k = 0; k < IK_TOPK; k++) { fprintf(g_ik_dump, "%c%d", k ? ',' : ' ', sel[t * IK_TOPK + k]); sum += sel[t * IK_TOPK + k]; }
            fputc('\n', g_ik_dump);
        }
        fprintf(g_ik_dump, "topk_sum visit=%d slot=%d %.0f\n", r->visit, slot, sum);
    }
    free(wbuf); free(logits); free(sel); free(wt); free(hx); free(gu); free(ac); free(yy); free(tok); free(kk);
}

/* One layer visit: slot `slot` runs physical layer `ip` over the chunk.  msg is NULL on the first pass,
 * otherwise [4][n][E] messages for the Q, K, V and FFN inputs.  Returns the new stream (arena-owned). */
static float *ik_visit(ik_run *r, int slot, int ip, float *x, const float *msg) {
    const ds4_model *m = r->w->m;
    const ik_layer *L = &r->w->layer[ip];
    ik_state *st = r->st;
    const int n = r->n;
    const bool global = ik_layer_is_global(ip);
    const size_t NE = (size_t)n * IK_E;
    const int nq = global ? 3 : 1;

    /* 1. attention inputs: depth mix of the block starts and x (layer 0 has no queries) */
    const float *in[3] = { x, x, x };
    float *mixed = NULL;
    if (ip > 0) {
        const float *states[IK_MAX_STATES];
        memcpy(states, r->blocks, (size_t)r->nb * sizeof(float *));
        states[r->nb] = x;
        mixed = xmalloc((size_t)nq * NE * sizeof(float));
        ik_mix(states, r->nb + 1, ik_f32(m, L->attn_res_query), nq, n, mixed);
        for (int j = 0; j < 3; j++) in[j] = mixed + (global ? (size_t)j : 0) * NE;
    }
    /* 2. loop messages: local layers get three different inputs from here on */
    float *ibuf[3];
    for (int j = 0; j < 3; j++) {
        ibuf[j] = xmalloc(NE * sizeof(float));
        memcpy(ibuf[j], in[j], NE * sizeof(float));
        if (msg) ik_control(r, L, j, ibuf[j], msg + (size_t)j * NE);
        /* 3. input norms (one shared weight on local layers, three on global ones) */
        ik_rms_rows(ibuf[j], ik_f32(m, global ? L->inp_norm[j] : L->attn_norm), ibuf[j], n, IK_E);
    }
    /* 4. projections */
    float *Q = xmalloc((size_t)n * IK_H * IK_D * sizeof(float)), *Kp = xmalloc((size_t)n * IK_HK * IK_D * sizeof(float));
    float *Vp = xmalloc((size_t)n * IK_HK * IK_D * sizeof(float)), *Rr = xmalloc((size_t)n * IK_H * IK_R * sizeof(float));
    float *sink = xmalloc((size_t)n * IK_H * sizeof(float)), *gate = xmalloc((size_t)n * IK_H * IK_D * sizeof(float));
    ik_matmul(m, L->q, IK_H * IK_D, ibuf[0], n, Q);
    ik_matmul(m, L->k, IK_HK * IK_D, ibuf[1], n, Kp);
    ik_matmul(m, L->v, IK_HK * IK_D, ibuf[2], n, Vp);
    ik_matmul(m, L->rel, IK_R * IK_H, ibuf[0], n, Rr);       /* layout head*16 + r */
    ik_matmul(m, L->sink, IK_H, ibuf[0], n, sink);
    ik_matmul(m, L->gate, IK_H * IK_D, ibuf[0], n, gate);
    float *dk = NULL, *dv = NULL;
    if (global) {   /* 32 outputs per token, layout kv_head*4 + lag, lag 0 = current token */
        dk = xmalloc((size_t)n * IK_KC * IK_HK * sizeof(float)); dv = xmalloc((size_t)n * IK_KC * IK_HK * sizeof(float));
        ik_matmul(m, L->dyn_proj[0], IK_KC * IK_HK, ibuf[1], n, dk);
        ik_matmul(m, L->dyn_proj[1], IK_KC * IK_HK, ibuf[2], n, dv);
    }
    float *Kc = xmalloc((size_t)n * IK_HK * IK_D * sizeof(float)), *Vc = xmalloc((size_t)n * IK_HK * IK_D * sizeof(float));
    ik_conv(ik_f32(m, L->k_conv), st->conv[slot] + 0 * 3, Kp, dk, IK_D, n, IK_HK * IK_D, Kc);
    ik_conv(ik_f32(m, L->v_conv), st->conv[slot] + 1024 * 3, Vp, dv, IK_D, n, IK_HK * IK_D, Vc);
    /* 5. normalisations; tau scales Q and R on global layers after the clip */
    const float *qclip = ik_f32(m, L->q_clip), *rclip = ik_f32(m, L->rel_clip);
    for (int t = 0; t < n; t++) {
        for (int h = 0; h < IK_H; h++) {
            float *q = Q + ((uint64_t)t * IK_H + h) * IK_D, *rr = Rr + ((uint64_t)t * IK_H + h) * IK_R;
            for (int j = 0; j < IK_D; j++) q[j] *= qclip[h * IK_D + j];
            ik_clip_l2(q, IK_D, IK_QK_CLIP);
            for (int j = 0; j < IK_R; j++) rr[j] *= rclip[h * IK_R + j];
            ik_clip_l2(rr, IK_R, IK_REL_CLIP);
            if (global) {
                for (int j = 0; j < IK_D; j++) q[j] *= r->tau[t];
                for (int j = 0; j < IK_R; j++) rr[j] *= r->tau[t];
            }
        }
        for (int g = 0; g < IK_HK; g++) {
            float *k = Kc + ((uint64_t)t * IK_HK + g) * IK_D, *v = Vc + ((uint64_t)t * IK_HK + g) * IK_D;
            ik_clip_l2(k, IK_D, IK_QK_CLIP);
            ik_rms(v, v, NULL, IK_D);
            /* cache rows are half precision; the attention below reads them back */
            const uint64_t row = (uint64_t)((r->p0 + (uint32_t)t) % st->rows[slot]);
            for (int j = 0; j < IK_D; j++) {
                st->k[slot][(row * IK_HK + g) * IK_D + j] = f32_to_f16(k[j]);
                st->v[slot][(row * IK_HK + g) * IK_D + j] = f32_to_f16(v[j]);
            }
        }
    }
    /* 6. attention: scores = q.k/sqrt(D) + relative bias - sink, plus a virtual key (logit 0, value 0) */
    float *O = xmalloc((size_t)n * IK_H * IK_D * sizeof(float));
    ik_attn_ctx ac = { Q, Rr, sink, ik_f32(m, L->rel_bias), st->k[slot], st->v[slot], st->rows[slot], r->p0, !global, O };
    ds4_parallel_for_min_rows((uint64_t)n * IK_H, ik_attn_worker, &ac, 16);
    const double o_sum = g_ik_dump ? ik_sum(O, (size_t)n * IK_H * IK_D) : 0.0;
    /* 7. output gate, projection, post conv */
    for (uint64_t i = 0; i < (uint64_t)n * IK_H * IK_D; i++) O[i] *= ik_sigmoid(gate[i]);
    float *a = xmalloc(NE * sizeof(float)), *ap = xmalloc(NE * sizeof(float));
    ik_matmul(m, L->o, IK_E, O, n, ap);
    ik_conv(ik_f32(m, L->post_conv), st->conv[slot] + 2048 * 3, ap, NULL, 1, n, IK_E, a);
    /* 8. block bookkeeping: at the first layer of each block the stream restarts from the attention output */
    float *ffn_inp = a;
    if (ip % IK_BLOCK == 0) {
        r->blocks[r->nb++] = x;
    } else {
        for (size_t i = 0; i < NE; i++) a[i] += x[i];
    }
    /* 9. MoE on a depth mix of the block starts and ffn_inp */
    const float *states[IK_MAX_STATES];
    memcpy(states, r->blocks, (size_t)r->nb * sizeof(float *));
    states[r->nb] = ffn_inp;
    float *fm = xmalloc(NE * sizeof(float)), *h = xmalloc(NE * sizeof(float));
    ik_mix(states, r->nb + 1, ik_f32(m, L->ffn_res_query), 1, n, fm);
    if (msg) ik_control(r, L, 3, fm, msg + (size_t)3 * NE);
    ik_rms_rows(fm, ik_f32(m, L->ffn_norm), h, n, IK_E);
    float *f = xmalloc(NE * sizeof(float)), *fc = xmalloc(NE * sizeof(float));
    ik_moe(r, L, h, f, slot);
    ik_conv(ik_f32(m, L->ffn_post_conv), st->conv[slot] + 4352 * 3, f, NULL, 1, n, IK_E, fc);
    float *y = ik_new(r, NE);
    for (size_t i = 0; i < NE; i++) y[i] = ffn_inp[i] + fc[i];
    if (g_ik_dump) {
        fprintf(g_ik_dump, "visit=%d slot=%d layer=%d ffn_inp=%.6f ffn_moe_out=%.6f ffn_out=%.6f l_out=%.6f\n",
                r->visit, slot, ip, ik_sum(ffn_inp, NE), ik_sum(f, NE), ik_sum(fc, NE), ik_sum(y, NE));
        fprintf(g_ik_dump, "attn visit=%d slot=%d Qcur=%.6f Kcur=%.6f Vcur=%.6f Rcur=%.6f kqv_out=%.6f\n", r->visit, slot,
                ik_sum(Q, (size_t)n * IK_H * IK_D), ik_sum(Kc, (size_t)n * IK_HK * IK_D), ik_sum(Vc, (size_t)n * IK_HK * IK_D),
                ik_sum(Rr, (size_t)n * IK_H * IK_R), o_sum);
    }
    if (g_ik_dump)
        for (int t = 0; t < n; t++)
            fprintf(g_ik_dump, "lout visit=%d slot=%d pos=%u %.9g\n", r->visit, slot, r->p0 + (uint32_t)t, ik_sum(y + (size_t)t * IK_E, IK_E));
    r->visit++;
    free(mixed); free(Q); free(Kp); free(Vp); free(Rr); free(sink); free(gate); free(dk); free(dv);
    free(Kc); free(Vc); free(O); free(a); free(ap); free(fm); free(h); free(f); free(fc);
    for (int j = 0; j < 3; j++) free(ibuf[j]);
    return y;
}

/* Run n tokens at positions st->pos.. through the 48 visits.  logits receives IK_VOCAB soft-capped values for
 * each of the chunk rows listed in want[] (row indexes inside the chunk). */
static int ik_forward_chunk(const ds4_inkling2 *w, ik_state *st, const int *tokens, int n,
                            const int *want, int nwant, float *logits, const char **err) {
    const ds4_model *m = w->m;
    *err = NULL;
    if (n <= 0 || (uint32_t)n > st->chunk || (uint64_t)st->pos + (uint64_t)n > st->cap) {
        *err = "chunk exceeds the state capacity";
        return 1;
    }
    for (int t = 0; t < n; t++)
        if (tokens[t] < 0 || tokens[t] >= IK_VOCAB) { *err = "token id out of range"; return 1; }
    const size_t NE = (size_t)n * IK_E;
    ik_run r = { .w = w, .st = st, .n = n, .p0 = st->pos };
    r.tau = xmalloc((size_t)n * sizeof(float));
    for (int t = 0; t < n; t++) r.tau[t] = 1.0f + 0.1f * logf(fmaxf(1.0f, (float)(r.p0 + (uint32_t)t + 1) / 8192.0f));

    /* embedding row, then a weighted RMS norm (text tokens only; there is no positional embedding) */
    float *x = ik_new(&r, NE), buf[IK_E];
    uint64_t row_bytes = 0;
    tensor_nbytes(w->embd->type, IK_E, &row_bytes);
    const uint8_t *emb = tensor_data(m, w->embd);
    for (int t = 0; t < n; t++) {
        ik_rms(ik_row_f32(w->embd->type, emb + (uint64_t)tokens[t] * row_bytes, IK_E, buf), x + (uint64_t)t * IK_E,
               ik_f32(m, w->embd_norm), IK_E);
    }

    /* Schedule (48 visits): prelude 0-5; core 6-17 three times; postlude 18-23.  Passes 2 and 3 restart from the
     * prelude output and block list; their messages mix the previous pass's layer-12 block start ("bank") with its
     * layer-17 output.  Slots: 0-17, then 24-35 and 36-47 for the extra passes, then 18-23. */
    for (int il = 0; il < IK_LOOP_START && !r.err; il++) x = ik_visit(&r, il, il, x, NULL);
    float *prelude_x = x, *bank_out = NULL;
    const int prelude_nb = r.nb;
    const float *bank[IK_MAX_STATES];
    int nbank = 0;
    for (int pass = 0; pass < IK_LOOP_N && !r.err; pass++) {
        if (pass > 0) {
            nbank = r.nb - (prelude_nb + 1);
            memcpy(bank, r.blocks + prelude_nb + 1, (size_t)nbank * sizeof(float *));
            bank_out = x;
            x = prelude_x;
            r.nb = prelude_nb;
        }
        for (int ip = IK_LOOP_START; ip < IK_LOOP_END && !r.err; ip++) {
            const int slot = pass == 0 ? ip : IK_NLAYER + (pass - 1) * (IK_LOOP_END - IK_LOOP_START) + (ip - IK_LOOP_START);
            float *msg = NULL;
            if (pass > 0) {
                const float *states[IK_MAX_STATES];
                memcpy(states, bank, (size_t)nbank * sizeof(float *));
                states[nbank] = bank_out;
                msg = xmalloc(4 * NE * sizeof(float));
                ik_mix(states, nbank + 1, ik_f32(m, w->layer[ip].msg_query), 4, n, msg);
            }
            x = ik_visit(&r, slot, ip, x, msg);
            free(msg);
        }
    }
    for (int il = IK_LOOP_END; il < IK_NLAYER && !r.err; il++) x = ik_visit(&r, il, il, x, NULL);

    /* output: depth mix over every block start and the final stream, norm, LM head, soft cap */
    if (nwant > 0 && !r.err) {
        const float *states[IK_MAX_STATES];
        memcpy(states, r.blocks, (size_t)r.nb * sizeof(float *));
        states[r.nb] = x;
        float *mo = xmalloc(NE * sizeof(float)), *hh = xmalloc((size_t)nwant * IK_E * sizeof(float));
        ik_mix(states, r.nb + 1, ik_f32(m, w->out_res_query), 1, n, mo);
        for (int i = 0; i < nwant; i++)
            ik_rms(mo + (uint64_t)want[i] * IK_E, hh + (uint64_t)i * IK_E, ik_f32(m, w->out_norm), IK_E);
        ik_matmul(m, w->out, IK_VOCAB, hh, nwant, logits);
        for (uint64_t i = 0; i < (uint64_t)nwant * IK_VOCAB; i++) logits[i] = IK_SOFTCAP * tanhf(logits[i] / IK_SOFTCAP);
        if (g_ik_dump)
            for (int i = 0; i < nwant; i++)
                fprintf(g_ik_dump, "result row=%d result_norm=%.6f result_output=%.6f\n", want[i], ik_sum(hh + (uint64_t)i * IK_E, IK_E),
                        ik_sum(logits + (uint64_t)i * IK_VOCAB, IK_VOCAB));
        free(mo); free(hh);
    }
    for (int i = 0; i < r.na; i++) free(r.arena[i]);
    free(r.arena);
    free(r.tau);
    if (r.err) {   /* the K/V rows and conv tails of this chunk are half written: the caller must drop the state */
        *err = r.err;
        return 1;
    }
    st->pos += (uint32_t)n;
    return 0;
}

/* ------------------------------------------------------------------------- */
/* Tokenizer                                                                 */
/* ------------------------------------------------------------------------- */

/* Special tokens (GGUF token types 2, 3, 4), longest text first as in llama.cpp: they are cut out of the text
 * as whole tokens when special parsing is on. */
static struct { int *id; int n; } g_ik_special;

static void ik_vocab_init(ds4_vocab *v, const ds4_model *model) {
    v->bos_id = v->system_id = v->user_id = v->assistant_id = v->observation_id = v->sop_id = -1;
    v->think_start_id = v->think_end_id = v->tool_call_start_id = v->tool_call_end_id = -1;
    v->tool_response_start_id = v->tool_response_end_id = -1;
    v->arg_key_start_id = v->arg_key_end_id = v->arg_value_start_id = v->arg_value_end_id = v->dsml_id = -1;
    v->im_start_id = v->im_end_id = v->endoftext_id = -1;   /* no BOS ever; chat rendering is a later stage */
    if (!model_get_token_id(model, "tokenizer.ggml.eos_token_id", &v->eos_id)) v->eos_id = -1;

    ds4_array_ref ty;
    if (!model_get_array(model, "tokenizer.ggml.token_type", &ty) || ty.len != (uint64_t)v->n_vocab ||
        ty.type != GGUF_VALUE_INT32) {
        ik_die("%s", "tokenizer.ggml.token_type must be an int32 array matching the vocabulary");
    }
    free(g_ik_special.id);
    g_ik_special.id = xmalloc((size_t)v->n_vocab * sizeof(int));
    g_ik_special.n = 0;
    ds4_cursor c = cursor_at(model, ty.data_pos);
    for (int i = 0; i < v->n_vocab; i++) {
        int32_t t = 0;
        if (!cursor_read(&c, &t, 4)) ik_die("%s", "tokenizer.ggml.token_type is truncated");
        if (t >= 2 && t <= 4) g_ik_special.id[g_ik_special.n++] = i;
    }
    for (int i = 1; i < g_ik_special.n; i++) {   /* insertion sort by text length, descending */
        const int id = g_ik_special.id[i];
        int j = i;
        while (j > 0 && v->token[g_ik_special.id[j - 1]].len < v->token[id].len) { g_ik_special.id[j] = g_ik_special.id[j - 1]; j--; }
        g_ik_special.id[j] = id;
    }
}

/* GPT-4o pre-tokenizer (tokenizer.ggml.pre = inkling2), the ordered alternation
 *   [^\r\n\p{L}\p{N}]? U* W+ SUF? | [^\r\n\p{L}\p{N}]? U+ W* SUF? | \p{N}{1,3} | ' '?[^\s\p{L}\p{N}]+[\r\n/]* |
 *   \s*[\r\n]+ | \s+(?!\S) | \s+
 * with U = (\p{L}|\p{M}) and not a-z, W = (\p{L}|\p{M}) and not A-Z, SUF = '(s|t|re|ve|m|ll|d) (any case).
 * Non-ASCII letters are in both U and W.  Each rule below reproduces the regex backtracking by hand. */
#define IK_NONE UINT64_MAX

static bool ik_is_lm(const qwen4_char_info *c) { return c->valid && (c->letter || c->mark); }
static bool ik_is_up(const qwen4_char_info *c) { return ik_is_lm(c) && !(c->cp >= 'a' && c->cp <= 'z'); }
static bool ik_is_lw(const qwen4_char_info *c) { return ik_is_lm(c) && !(c->cp >= 'A' && c->cp <= 'Z'); }

static uint64_t ik_suffix(const char *s, uint64_t len, uint64_t p) {
    if (p + 1 < len && s[p] == '\'') {
        const char a = (char)(s[p + 1] | 0x20), b = p + 2 < len ? (char)(s[p + 2] | 0x20) : 0;
        if (a == 's' || a == 't' || a == 'm' || a == 'd') return p + 2;
        if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l')) return p + 3;
    }
    return p;
}

/* U* W+ (a1) or U+ W* (a2) starting at p; IK_NONE if it does not match. */
static uint64_t ik_word_body(const char *s, uint64_t len, uint64_t p, bool a1) {
    qwen4_char_info c = qwen4_char_at(s, len, p);
    uint64_t q = p;
    if (!a1 && !ik_is_up(&c)) return IK_NONE;
    while (ik_is_up(&c)) { q = c.next; c = qwen4_char_at(s, len, q); }
    if (!a1) {                                  /* U+ is greedy and W* may be empty: no backtracking needed */
        while (ik_is_lw(&c)) { q = c.next; c = qwen4_char_at(s, len, q); }
        return q;
    }
    for (;;) {                                  /* U* gives characters back until W+ can start */
        c = qwen4_char_at(s, len, q);
        if (ik_is_lw(&c)) {
            while (ik_is_lw(&c)) { q = c.next; c = qwen4_char_at(s, len, q); }
            return q;
        }
        if (q == p) return IK_NONE;
        do q--; while (q > p && ((uint8_t)s[q] & 0xc0) == 0x80);
    }
}

static uint64_t ik_word(const char *s, uint64_t len, uint64_t pos, bool a1) {
    const qwen4_char_info c = qwen4_char_at(s, len, pos);
    if (c.valid && c.cp != '\r' && c.cp != '\n' && !c.letter && !c.number) {   /* optional one-character prefix */
        const uint64_t e = ik_word_body(s, len, c.next, a1);
        if (e != IK_NONE) return ik_suffix(s, len, e);
    }
    const uint64_t e = ik_word_body(s, len, pos, a1);
    return e == IK_NONE ? IK_NONE : ik_suffix(s, len, e);
}

static uint64_t ik_next_piece(const char *s, uint64_t len, uint64_t pos) {
    uint64_t e = ik_word(s, len, pos, true);
    if (e == IK_NONE) e = ik_word(s, len, pos, false);
    if (e != IK_NONE) return e;
    qwen4_char_info c = qwen4_char_at(s, len, pos);
    if (c.number) {                                         /* \p{N}{1,3} */
        uint64_t q = pos;
        for (int i = 0; i < 3 && c.valid && c.number; i++) { q = c.next; c = qwen4_char_at(s, len, q); }
        return q;
    }
    c = qwen4_char_at(s, len, pos);
    uint64_t q = pos + (s[pos] == ' ');                     /* ' '?[^\s\p{L}\p{N}]+[\r\n/]* */
    qwen4_char_info p = qwen4_char_at(s, len, q);
    if (p.valid && !p.space && !p.letter && !p.number) {
        while (p.valid && !p.space && !p.letter && !p.number) { q = p.next; p = qwen4_char_at(s, len, q); }
        while (q < len && (s[q] == '\r' || s[q] == '\n' || s[q] == '/')) q++;
        return q;
    }
    if (!c.space) return c.next;                            /* unreachable for valid text; keeps progress */
    uint64_t run_end = pos, last_nl = 0, n_ws = 0, last_ws = pos;   /* whitespace run */
    for (qwen4_char_info w = c; w.valid && w.space; w = qwen4_char_at(s, len, run_end)) {
        last_ws = run_end;
        run_end = w.next;
        n_ws++;
        if (w.cp == '\r' || w.cp == '\n') last_nl = run_end;
    }
    if (last_nl) return last_nl;                            /* \s*[\r\n]+ */
    if (n_ws > 1 && run_end < len) return last_ws;          /* \s+(?!\S) gives the last space back to the next word */
    return run_end;                                         /* \s+(?!\S) at end of text, or a lone space */
}

/* ignore_merges: a pre-token that is a whole vocabulary entry is used as is, otherwise BPE merges it. */
static void ik_emit_piece(const ds4_vocab *v, const char *s, uint64_t len, token_vec *out) {
    uint64_t el = 0;
    char *enc = byte_encode((ds4_str){ s, len }, &el);
    int tok = -1;
    if (table_get(&v->token_to_id, enc, el, &tok)) token_vec_push(out, tok);
    else bpe_emit_piece(v, (ds4_str){ s, len }, out);
    free(enc);
}

static void ik_split_emit(const ds4_vocab *v, const char *s, uint64_t len, token_vec *out) {
    for (uint64_t pos = 0; pos < len;) {
        uint64_t e = ik_next_piece(s, len, pos);
        if (e <= pos) e = next_utf8_char(s, len, pos);
        ik_emit_piece(v, s + pos, e - pos, out);
        pos = e;
    }
}

typedef struct { uint64_t off, len; int token; } ik_frag;   /* token < 0: raw text still to tokenize */
typedef struct { ik_frag *v; int n, cap; } ik_frags;

static void ik_frags_push(ik_frags *a, uint64_t off, uint64_t len, int token) {
    if (a->n == a->cap) {
        a->cap = a->cap ? a->cap * 2 : 16;
        a->v = xrealloc(a->v, (size_t)a->cap * sizeof(a->v[0]));
    }
    a->v[a->n++] = (ik_frag){ off, len, token };
}

/* Cuts the special tokens out of text as whole tokens, longest token first like llama.cpp's partition (each token is applied to the
 * raw pieces left by the longer ones, matches do not overlap, leftmost first).  One pass per special token that appears; the first
 * byte is found with memchr, so text without special tokens costs almost nothing and many matches stay linear. */
static ik_frags ik_special_partition(const ds4_vocab *v, const char *text, uint64_t len) {
    ik_frags f = {0};
    ik_frags_push(&f, 0, len, -1);
    for (int si = 0; si < g_ik_special.n; si++) {
        const ds4_str sp = v->token[g_ik_special.id[si]];
        if (!sp.len) continue;
        ik_frags g = {0};
        bool hit = false;
        for (int i = 0; i < f.n; i++) {
            const ik_frag fr = f.v[i];
            if (fr.token >= 0) { ik_frags_push(&g, fr.off, fr.len, fr.token); continue; }
            const uint64_t end = fr.off + fr.len;
            uint64_t p = fr.off, seg = fr.off;
            while (p + sp.len <= end) {
                const char *q = memchr(text + p, sp.ptr[0], (size_t)(end - sp.len + 1 - p));
                if (!q) break;
                p = (uint64_t)(q - text);
                if (memcmp(text + p, sp.ptr, (size_t)sp.len) != 0) { p++; continue; }
                hit = true;
                if (p > seg) ik_frags_push(&g, seg, p - seg, -1);
                ik_frags_push(&g, p, sp.len, g_ik_special.id[si]);
                p += sp.len;
                seg = p;
            }
            if (end > seg) ik_frags_push(&g, seg, end - seg, -1);
        }
        if (hit) { free(f.v); f = g; }
        else free(g.v);
    }
    return f;
}

static void ik_tokenize_n(const ds4_vocab *v, const char *text, uint64_t len, bool parse_special, token_vec *out) {
    if (!parse_special || g_ik_special.n == 0) { ik_split_emit(v, text, len, out); return; }
    ik_frags f = ik_special_partition(v, text, len);
    for (int i = 0; i < f.n; i++) {
        if (f.v[i].token >= 0) token_vec_push(out, f.v[i].token);
        else if (f.v[i].len) {
            char *tmp = xmalloc((size_t)f.v[i].len + 1);
            memcpy(tmp, text + f.v[i].off, (size_t)f.v[i].len);
            tmp[f.v[i].len] = '\0';
            ik_split_emit(v, tmp, f.v[i].len, out);
            free(tmp);
        }
    }
    free(f.v);
}

static void ik_tokenize(const ds4_vocab *v, const char *text, bool parse_special, token_vec *out) {
    ik_tokenize_n(v, text, strlen(text), parse_special, out);
}

/* ------------------------------------------------------------------------- */
/* Oracle: ./ds4 --cpu --first-token-test with DS4_INKLING2_* variables      */
/* ------------------------------------------------------------------------- */

static double ik_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

/* Integers from a comma/space separated list or from a file holding such a list. */
static int ik_parse_ints(const char *arg, int **out) {
    char *buf = NULL;
    FILE *fp = fopen(arg, "rb");
    if (fp) {
        fseek(fp, 0, SEEK_END);
        const long sz = ftell(fp);
        fseek(fp, 0, SEEK_SET);
        buf = xmalloc((size_t)sz + 1);
        if (fread(buf, 1, (size_t)sz, fp) != (size_t)sz) ik_die("cannot read %s", arg);
        buf[sz] = '\0';
        fclose(fp);
    } else {
        buf = ds4_strdup(arg);
    }
    int n = 0, cap = 64;
    int *v = xmalloc((size_t)cap * sizeof(int));
    for (const char *p = buf; *p;) {
        if (!isdigit((unsigned char)*p)) { p++; continue; }
        char *end;
        const long x = strtol(p, &end, 10);
        if (n == cap) { cap *= 2; v = xrealloc(v, (size_t)cap * sizeof(int)); }
        v[n++] = (int)x;
        p = end;
    }
    free(buf);
    *out = v;
    return n;
}

/* Four hex digits at p; false if fewer than four characters remain or one is not a hex digit. */
static bool ik_hex4(const char *p, uint32_t *out) {
    uint32_t v = 0;
    for (int i = 0; i < 4; i++) {
        const int c = (unsigned char)p[i];
        if (c >= '0' && c <= '9') v = v * 16 + (uint32_t)(c - '0');
        else if (c >= 'a' && c <= 'f') v = v * 16 + (uint32_t)(c - 'a' + 10);
        else if (c >= 'A' && c <= 'F') v = v * 16 + (uint32_t)(c - 'A' + 10);
        else return false;   /* also catches the terminating NUL: nothing is read past it */
    }
    *out = v;
    return true;
}

/* One JSON string literal (the part after the opening quote) to UTF-8 (may contain NUL); false if it is
 * unterminated or has a malformed escape.  The output never exceeds the input length. */
static bool ik_json_string(const char *p, char **out, uint64_t *out_len) {
    char *o = xmalloc(strlen(p) + 1), *w = o;
    bool ok = true;
    for (; *p && *p != '"'; p++) {
        if (*p != '\\') { *w++ = *p; continue; }
        p++;
        if (!*p) { ok = false; break; }   /* a lone backslash at the end of the line */
        switch (*p) {
        case 'n': *w++ = '\n'; break;
        case 't': *w++ = '\t'; break;
        case 'r': *w++ = '\r'; break;
        case 'b': *w++ = '\b'; break;
        case 'f': *w++ = '\f'; break;
        case 'u': {
            uint32_t cp, lo;
            if (!ik_hex4(p + 1, &cp)) { ok = false; break; }
            p += 4;
            if (cp >= 0xd800 && cp < 0xdc00 && p[1] == '\\' && p[2] == 'u') {
                if (!ik_hex4(p + 3, &lo) || lo < 0xdc00 || lo >= 0xe000) { ok = false; break; }
                cp = 0x10000 + ((cp - 0xd800) << 10) + (lo - 0xdc00);
                p += 6;
            }
            utf8_put(&w, cp);
            break;
        }
        default: *w++ = *p; break;   /* \" \\ \/ */
        }
        if (!ok) break;
    }
    *w = '\0';
    *out = o;
    *out_len = (uint64_t)(w - o);
    return ok && *p == '"';
}

/* DS4_INKLING2_TOK_FILE: one line per string, "0" or "1" (parse special tokens) followed by a JSON string
 * literal.  Prints one JSON array of ids per line. */
static int ik_tokenize_file(const ds4_vocab *v, const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) ik_die("cannot open %s", path);
    size_t cap = 1 << 16;
    char *line = xmalloc(cap);
    int rc = 0;
    for (size_t n = 0;;) {
        const int ch = fgetc(fp);
        if (ch != '\n' && ch != EOF) {
            if (n + 2 > cap) { cap *= 2; line = xrealloc(line, cap); }
            line[n++] = (char)ch;
            continue;
        }
        if (n == 0 && ch == EOF) break;
        line[n] = '\0';
        char *text = NULL;
        uint64_t text_len = 0;
        if (n < 3 || line[1] != '"' || !ik_json_string(line + 2, &text, &text_len)) { fprintf(stderr, "ds4: inkling2: bad line in %s\n", path); rc = 1; break; }
        token_vec ids = {0};
        ik_tokenize_n(v, text, text_len, line[0] == '1', &ids);
        putchar('[');
        for (int i = 0; i < ids.len; i++) printf("%s%d", i ? "," : "", ids.v[i]);
        puts("]");
        token_vec_free(&ids);
        free(text);
        n = 0;
        if (ch == EOF) break;
    }
    free(line);
    fclose(fp);
    return rc;
}

#ifdef DS4_HAS_INKLING2_GPU
#include "ds4_inkling2_gpu.inc"
#endif

static int ik_cmp_int(const void *a, const void *b) { return *(const int *)a - *(const int *)b; }

/* Debug driver.  Tokens come from DS4_INKLING2_FT_TOKENS (ids or a file of ids) or from the prompt; logits are
 * produced at DS4_INKLING2_FT_POS (default: last token) and written as top-K log-probabilities (JSON lines,
 * DS4_INKLING2_FT_OUT, K = DS4_INKLING2_FT_TOPK, default 50) and optionally as raw f32 rows
 * (DS4_INKLING2_FT_LOGITS).  DS4_INKLING2_FT_VISITS dumps per-visit sums and expert ids; DS4_INKLING2_CHUNK sets
 * the tokens per forward call (default 256): one chunk covering the prompt is the batch path, 1 is token-by-token
 * decoding from the cached state.  DS4_INKLING2_FT_FLIP=visit:token,... takes the 9th expert instead of the 8th at
 * those router decisions (near-tie diagnostic; the visits dump lists decisions with a margin under
 * DS4_INKLING2_FT_TIE, default 1e-4).
 * DS4_INKLING2_TOK_FILE only tokenizes. */
static int ik_first_token_test(ds4_engine *e, const ds4_tokens *prompt) {
    const char *tok_file = getenv("DS4_INKLING2_TOK_FILE");
    if (tok_file && tok_file[0]) return ik_tokenize_file(&e->vocab, tok_file);

    const char *env = getenv("DS4_INKLING2_FT_TOKENS");
    int *tokens = NULL, ntok;
    if (env && env[0]) {
        ntok = ik_parse_ints(env, &tokens);
    } else {
        ntok = prompt->len;
        tokens = xmalloc((size_t)(ntok > 0 ? ntok : 1) * sizeof(int));
        memcpy(tokens, prompt->v, (size_t)ntok * sizeof(int));
    }
    if (ntok <= 0) { fprintf(stderr, "ds4: inkling2: no tokens\n"); free(tokens); return 1; }
    int *pos = NULL, npos = 1;
    if ((env = getenv("DS4_INKLING2_FT_POS")) && env[0]) {
        npos = ik_parse_ints(env, &pos);
        qsort(pos, (size_t)npos, sizeof(int), ik_cmp_int);
    } else {
        pos = xmalloc(sizeof(int));
        pos[0] = ntok - 1;
    }
    for (int i = 0; i < npos; i++) if (pos[i] < 0 || pos[i] >= ntok) ik_die("%s", "DS4_INKLING2_FT_POS is outside the token list");
    int chunk = 256, topk = 50;
    if ((env = getenv("DS4_INKLING2_CHUNK")) && atoi(env) > 0) chunk = atoi(env);
    if ((env = getenv("DS4_INKLING2_FT_TOPK")) && atoi(env) > 0) topk = atoi(env);
    if ((env = getenv("DS4_INKLING2_FT_TIE")) && atof(env) > 0) g_ik_tie = (float)atof(env);
    if ((env = getenv("DS4_INKLING2_FT_FLIP")) && env[0]) {   /* visit:token,visit:token */
        int *v = NULL, nv = ik_parse_ints(env, &v);
        for (int i = 0; i + 1 < nv && g_ik_nflip < 64; i += 2, g_ik_nflip++) { g_ik_flip[g_ik_nflip][0] = v[i]; g_ik_flip[g_ik_nflip][1] = v[i + 1]; }
        free(v);
    }
    if ((env = getenv("DS4_INKLING2_FT_PICKS")) && env[0]) {
        FILE *pf = fopen(env, "rb");
        if (!pf) ik_die("cannot open %s", env);
        fseek(pf, 0, SEEK_END);
        const long sz = ftell(pf);
        fseek(pf, 0, SEEK_SET);
        if (sz <= 0 || sz % (IK_NSLOT * IK_TOPK * 4) != 0) ik_die("%s is not a pick file (int32 [48][ntok][8])", env);
        g_ik_picks = xmalloc((size_t)sz);
        if (fread(g_ik_picks, 1, (size_t)sz, pf) != (size_t)sz) ik_die("cannot read %s", env);
        fclose(pf);
        g_ik_picks_ntok = (int)(sz / (IK_NSLOT * IK_TOPK * 4));
        if ((env = getenv("DS4_INKLING2_FT_PICKLOG")) && env[0]) g_ik_picklog = fopen(env, "w");
        fprintf(stderr, "ds4: inkling2: forcing router picks for %d tokens\n", g_ik_picks_ntok);
    }
    FILE *fout = (env = getenv("DS4_INKLING2_FT_OUT")) && env[0] ? fopen(env, "w") : stdout;
    FILE *flog = (env = getenv("DS4_INKLING2_FT_LOGITS")) && env[0] ? fopen(env, "wb") : NULL;
    g_ik_dump = (env = getenv("DS4_INKLING2_FT_VISITS")) && env[0] ? fopen(env, "w") : NULL;
    if (!fout || (getenv("DS4_INKLING2_FT_LOGITS") && !flog) || (getenv("DS4_INKLING2_FT_VISITS") && !g_ik_dump))
        ik_die("%s", "cannot open an output file");

    /* one logits row per requested position of a chunk, never more than there are positions */
    const int logit_rows = chunk < npos ? chunk : npos;
    ik_state *st = NULL;
    ik_gpu *gst = NULL;
#ifdef DS4_HAS_INKLING2_GPU
    if (e->backend == DS4_BACKEND_METAL) {
        gst = ikg_new(e->ik, (uint32_t)ntok, (uint32_t)chunk, (uint32_t)logit_rows);
        if (!gst) { fprintf(stderr, "ds4: inkling2: cannot create the GPU state\n"); return 1; }
        ikg_reset(gst);
        gst->dump = g_ik_dump;
        if (g_ik_picks && !ikg_set_forced(gst, g_ik_picks, (uint32_t)g_ik_picks_ntok)) { fprintf(stderr, "ds4: inkling2: cannot upload the pick file\n"); return 1; }
        if ((env = getenv("DS4_INKLING2_PICKS_OUT")) && env[0]) gst->pick_log = xcalloc((size_t)IK_NSLOT * (size_t)ntok * IK_TOPK, sizeof(int32_t));
    }
#endif
    if (!gst && (env = getenv("DS4_INKLING2_PICKS_OUT")) && env[0]) {
        g_ik_pick_out = xcalloc((size_t)IK_NSLOT * (size_t)ntok * IK_TOPK, sizeof(int32_t));
        g_ik_pick_out_ntok = (uint32_t)ntok;
    }
    if (!gst) st = ik_state_new((uint32_t)ntok, (uint32_t)chunk);
    float *logits = xmalloc((size_t)logit_rows * IK_VOCAB * sizeof(float));
    int *want = xmalloc((size_t)chunk * sizeof(int));
    int *top_id = xmalloc((size_t)topk * sizeof(int));
    double *top_lp = xmalloc((size_t)topk * sizeof(double));
    const double t0 = ik_now();
    int next_pos = 0;
    /* DS4_INKLING2_DECODE_TAIL=k: the last k tokens go one at a time (prefill in chunks, then decoding). */
    int tail = 0, step = chunk;
    if ((env = getenv("DS4_INKLING2_DECODE_TAIL")) && atoi(env) > 0) tail = atoi(env);
    for (int c0 = 0; c0 < ntok;) {
        step = (tail && c0 >= ntok - tail) ? 1 : chunk;
        int n = ntok - c0 < step ? ntok - c0 : step;
        if (tail && c0 < ntok - tail && n > ntok - tail - c0) n = ntok - tail - c0;   /* the tail starts on a chunk edge */
        int nw = 0;
        while (next_pos + nw < npos && pos[next_pos + nw] < c0 + n) { want[nw] = pos[next_pos + nw] - c0; nw++; }
        const double tc = ik_now();
        const char *ferr = NULL;
        int frc;
#ifdef DS4_HAS_INKLING2_GPU
        if (gst) frc = ikg_forward_chunk(gst, tokens + c0, n, want, nw, logits, &ferr);
        else
#endif
            frc = ik_forward_chunk(e->ik, st, tokens + c0, n, want, nw, logits, &ferr);
        if (frc) {
            fprintf(stderr, "ds4: inkling2: forward failed: %s\n", ferr);
            return 1;
        }
        fprintf(stderr, "ds4: inkling2: tokens %d-%d of %d in %.2fs\n", c0, c0 + n, ntok, ik_now() - tc);
        for (int i = 0; i < nw; i++) {
            const float *l = logits + (uint64_t)i * IK_VOCAB;
            float mx = l[0];
            for (int j = 1; j < IK_VOCAB; j++) if (l[j] > mx) mx = l[j];
            double z = 0;
            for (int j = 0; j < IK_VOCAB; j++) z += exp((double)(l[j] - mx));
            const double lse = (double)mx + log(z);
            int nt = 0;
            for (int j = 0; j < IK_VOCAB; j++) {   /* top-K by insertion; the threshold keeps this cheap */
                if (nt == topk && (double)l[j] <= top_lp[nt - 1] + lse) continue;
                int k = nt < topk ? nt++ : nt - 1;
                while (k > 0 && top_lp[k - 1] < (double)l[j] - lse) { top_lp[k] = top_lp[k - 1]; top_id[k] = top_id[k - 1]; k--; }
                top_lp[k] = (double)l[j] - lse;
                top_id[k] = j;
            }
            fprintf(fout, "{\"pos\":%d,\"top\":[", pos[next_pos + i]);
            for (int k = 0; k < nt; k++) fprintf(fout, "%s[%d,%.6f]", k ? "," : "", top_id[k], top_lp[k]);
            fputs("]}\n", fout);
            if (flog) fwrite(l, sizeof(float), IK_VOCAB, flog);
        }
        next_pos += nw;
        c0 += n;
    }
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    fprintf(stderr, "ds4: inkling2: %d tokens in %.1fs, peak RSS %.2f GiB\n", ntok, ik_now() - t0,
            (double)ru.ru_maxrss / (1024.0 * 1024.0 * 1024.0));
    if (fout != stdout) fclose(fout);
    if (flog) fclose(flog);
    if (g_ik_dump) fclose(g_ik_dump);
    g_ik_dump = NULL;
    if (g_ik_picklog) fclose(g_ik_picklog);
    g_ik_picklog = NULL;
    free(g_ik_picks);
    g_ik_picks = NULL;
#ifdef DS4_HAS_INKLING2_GPU
    if (gst && gst->pick_log) {
        FILE *pf = fopen(getenv("DS4_INKLING2_PICKS_OUT"), "wb");
        if (pf) { fwrite(gst->pick_log, sizeof(int32_t), (size_t)IK_NSLOT * (size_t)ntok * IK_TOPK, pf); fclose(pf); }
        else fprintf(stderr, "ds4: inkling2: cannot write the pick file\n");
    }
    if (gst) { gst->dump = NULL; ikg_free(gst); }
#endif
    if (g_ik_pick_out) {
        FILE *pf = fopen(getenv("DS4_INKLING2_PICKS_OUT"), "wb");
        if (pf) { fwrite(g_ik_pick_out, sizeof(int32_t), (size_t)IK_NSLOT * (size_t)ntok * IK_TOPK, pf); fclose(pf); }
        else fprintf(stderr, "ds4: inkling2: cannot write the pick file\n");
        free(g_ik_pick_out);
        g_ik_pick_out = NULL;
    }
    if (st) ik_state_free(st);
    free(tokens); free(pos); free(logits); free(want); free(top_id); free(top_lp);
    return 0;
}

static void vocab_load(ds4_vocab *vocab, const ds4_model *model);   /* tokenizer section of ds4.c */

/* Engine open for this family: CPU oracle only. */
#ifdef DS4_HAS_INKLING2_GPU
static int ik_check_memory_budget(uint64_t model_bytes, uint32_t ctx, int nsess);   /* ds4_inkling2_gpu.inc */
#endif
static int inkling2_engine_open(ds4_engine *e, const ds4_engine_options *opt) {
    bool metal_ok = false;
#ifdef DS4_HAS_INKLING2_GPU
    metal_ok = e->backend == DS4_BACKEND_METAL;
#endif
    if (!opt->inspect_only && !(metal_ok || (opt->first_token_test && e->backend == DS4_BACKEND_CPU))) {
        fprintf(stderr, "ds4: inkling2: needs --metal (the CPU path only runs --first-token-test as a parity oracle)\n");
        return 1;
    }
    /* Not in scope for this family yet: say so instead of misbehaving. */
    const char *unsupported = NULL;
    if (opt->mtp_path && opt->mtp_path[0]) unsupported = "an MTP model (speculative decoding)";
    else if (opt->vision_path && opt->vision_path[0]) unsupported = "vision";
    else if (opt->dspark) unsupported = "DSpark";
    else if (opt->ssd_streaming) unsupported = "SSD streaming";
    else if (opt->load_slice || opt->distributed.role != DS4_DISTRIBUTED_NONE || opt->tp.role != DS4_TP_NONE ||
             opt->cuda_tensor_parallel) unsupported = "distributed or tensor-parallel execution";
    else if (opt->directional_steering_file && opt->directional_steering_file[0]) unsupported = "directional steering";
    if (unsupported && !opt->inspect_only) {
        fprintf(stderr, "ds4: inkling2: %s is not supported yet\n", unsupported);
        return 1;
    }
#ifdef DS4_HAS_INKLING2_GPU
    if (metal_ok && !opt->inspect_only && !opt->first_token_test && opt->context_size > 0 &&
        ik_check_memory_budget(e->model.size, (uint32_t)opt->context_size, opt->placement_session_count_hint) != 0) return 1;
#endif
    ik_ue4m3_init();
    e->ik = inkling2_bind(&e->model);
    vocab_load(&e->vocab, &e->model);
#ifdef DS4_HAS_INKLING2_GPU
    if (metal_ok && !opt->inspect_only) {
        if (!ds4_gpu_init() || !ds4_gpu_ik_ready()) {
            fprintf(stderr, "ds4: inkling2: cannot load the Metal kernels (metal/inkling2.metal)\n");
            return 1;
        }
        if (!ds4_gpu_set_model_map_range(e->model.map, e->model.size, e->model.tensor_data_pos,
                                         e->model.size - e->model.tensor_data_pos, e->model.max_tensor_bytes)) {
            fprintf(stderr, "ds4: inkling2: cannot map the model for Metal\n");
            return 1;
        }
    }
#endif
    fprintf(stderr, "ds4: inkling2: %" PRIu64 " tensors bound, vocabulary %d tokens\n", e->model.n_tensors, e->vocab.n_vocab);
    return 0;
}
