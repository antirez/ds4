/* Recursive Flint (fork-owned): weight binding and layout validation.
 * Textually included into ds4.c after the tensor_expect_* helpers and just
 * before weights_have_output_head, whose family dispatch (and that of
 * weights_layer_has_required, weights_validate_layout, weights_bind_output
 * and weights_bind_layer) redirects here.
 *
 * The upstream ds4_layer_weights slots hold each matrix's `.weight` tensor
 * so generic walkers see the right bytes; the MLX scales and biases and the
 * per-expert pack have no upstream slot and live in the fork-side tables
 * below, indexed by layer. */

typedef struct {
    ds4_tensor *scales;
    ds4_tensor *biases;
} flint_quant_aux;

typedef struct {
    flint_quant_aux attn_q, attn_k, attn_v, attn_output;
    flint_quant_aux ffn_gate, ffn_up, ffn_down;                   /* dense layers */
    flint_quant_aux ffn_gate_shexp, ffn_up_shexp, ffn_down_shexp; /* MoE layers */
    ds4_tensor *ffn_exps_pack;                                    /* MoE layers */
} flint_layer_aux;

static flint_layer_aux g_flint_layer_aux[DS4_MAX_LAYER];
static flint_quant_aux g_flint_token_embd_aux;
static flint_quant_aux g_flint_output_aux;

static ds4_tensor *flint_tensor(const ds4_model *m, bool required, const char *fmt, ...) {
    char name[160];
    va_list ap;
    va_start(ap, fmt);
    const int n = vsnprintf(name, sizeof(name), fmt, ap);
    va_end(ap);
    if (n < 0 || (size_t)n >= sizeof(name)) ds4_die("tensor name is too long");
    return required ? required_tensor(m, name) : model_find_tensor(m, name);
}

/* weight + scales + biases of one MLX affine matrix */
static void flint_bind_quant(const ds4_model *m, bool required, const char *base,
                             ds4_tensor **weight, flint_quant_aux *aux) {
    *weight = flint_tensor(m, required, "%s.weight", base);
    aux->scales = flint_tensor(m, required, "%s.scales", base);
    aux->biases = flint_tensor(m, required, "%s.biases", base);
}

static void flint_bind_quant_layer(const ds4_model *m, uint32_t il, const char *base,
                                   ds4_tensor **weight, flint_quant_aux *aux) {
    char full[96];
    snprintf(full, sizeof(full), "blk.%u.%s", il, base);
    flint_bind_quant(m, true, full, weight, aux);
}

static void weights_bind_flint_output(ds4_weights *w, const ds4_model *m, bool required, bool optional) {
    if (w->token_embd) {
        g_flint_token_embd_aux.scales = required_tensor(m, "token_embd.scales");
        g_flint_token_embd_aux.biases = required_tensor(m, "token_embd.biases");
    }
    if (required || optional) {
        w->output_norm = flint_tensor(m, required, "output_norm.weight");
        flint_bind_quant(m, required, "output", &w->output, &g_flint_output_aux);
    }
    if (optional && (w->output_norm || w->output) && !(w->output_norm && w->output)) {
        ds4_die("partial output head in GGUF");
    }
}

static void weights_bind_flint_layer(ds4_layer_weights *l, const ds4_model *m, uint32_t il) {
    flint_layer_aux *a = &g_flint_layer_aux[il];
    l->attn_norm = required_tensorf(m, "blk.%u.attn_norm.weight", il);
    flint_bind_quant_layer(m, il, "attn_q", &l->attn_q, &a->attn_q);
    flint_bind_quant_layer(m, il, "attn_k", &l->attn_k, &a->attn_k);
    flint_bind_quant_layer(m, il, "attn_v", &l->attn_v, &a->attn_v);
    flint_bind_quant_layer(m, il, "attn_output", &l->attn_output, &a->attn_output);
    l->attn_q_norm = required_tensorf(m, "blk.%u.attn_q_norm.weight", il);
    l->attn_k_norm = required_tensorf(m, "blk.%u.attn_k_norm.weight", il);
    l->attn_sinks = required_tensorf(m, "blk.%u.attn_sinks.weight", il);
    l->ffn_norm = required_tensorf(m, "blk.%u.ffn_norm.weight", il);
    if (il < DS4_N_LEADING_DENSE) {
        flint_bind_quant_layer(m, il, "ffn_gate", &l->ffn_gate, &a->ffn_gate);
        flint_bind_quant_layer(m, il, "ffn_up", &l->ffn_up, &a->ffn_up);
        flint_bind_quant_layer(m, il, "ffn_down", &l->ffn_down, &a->ffn_down);
    } else {
        l->ffn_gate_inp = required_tensorf(m, "blk.%u.ffn_gate_inp.weight", il);
        flint_bind_quant_layer(m, il, "ffn_gate_shexp", &l->ffn_gate_shexp, &a->ffn_gate_shexp);
        flint_bind_quant_layer(m, il, "ffn_up_shexp", &l->ffn_up_shexp, &a->ffn_up_shexp);
        flint_bind_quant_layer(m, il, "ffn_down_shexp", &l->ffn_down_shexp, &a->ffn_down_shexp);
        a->ffn_exps_pack = required_tensorf(m, "blk.%u.ffn_exps.pack", il);
    }
}

static bool weights_flint_layer_has_required(const ds4_layer_weights *l, uint32_t il) {
    if (!l || il >= DS4_N_LAYER) return false;
    const flint_layer_aux *a = &g_flint_layer_aux[il];
    if (!l->attn_norm || !l->attn_q || !l->attn_k || !l->attn_v || !l->attn_output ||
        !l->attn_q_norm || !l->attn_k_norm || !l->attn_sinks || !l->ffn_norm) {
        return false;
    }
    if (il < DS4_N_LEADING_DENSE) {
        return l->ffn_gate && l->ffn_up && l->ffn_down;
    }
    return l->ffn_gate_inp && l->ffn_gate_shexp && l->ffn_up_shexp && l->ffn_down_shexp &&
           a->ffn_exps_pack;
}

/* An MLX affine matrix of [out, in]: nibbles or bytes in `weight`, BF16
 * [out, in/64] scales and biases.  GGUF dims list the fastest first. */
static void flint_expect_quant(const ds4_tensor *weight, const flint_quant_aux *aux,
                               uint32_t type, uint64_t in_dim, uint64_t out_dim) {
    tensor_expect_layout(weight, type, 2, in_dim, out_dim, 0);
    tensor_expect_layout(aux->scales, DS4_TENSOR_BF16, 2, in_dim / 64u, out_dim, 0);
    tensor_expect_layout(aux->biases, DS4_TENSOR_BF16, 2, in_dim / 64u, out_dim, 0);
}

static void weights_validate_flint_layout(const ds4_weights *w, uint32_t layer_start, uint32_t layer_end,
                                          bool require_token_embd, bool require_output) {
    const uint64_t q_dim = (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM;
    const uint64_t kv_dim = (uint64_t)DS4_N_HEAD_KV * DS4_N_HEAD_DIM;
    if (!w) ds4_die("internal error: missing weights while validating Flint layout");
    if (layer_start >= DS4_N_LAYER) ds4_die("invalid first layer in Flint weight layout validation");
    if (layer_end == UINT32_MAX) layer_end = DS4_N_LAYER - 1u;
    if (layer_end >= DS4_N_LAYER || layer_end < layer_start) {
        ds4_die("invalid layer range in Flint weight layout validation");
    }
    if (require_token_embd && !w->token_embd) ds4_die("required token embedding tensor is missing");
    if (w->token_embd) {
        flint_expect_quant(w->token_embd, &g_flint_token_embd_aux, DS4_TENSOR_FLINT_U8, DS4_N_EMBD, DS4_N_VOCAB);
    }
    if (require_output && !(w->output_norm && w->output)) ds4_die("required output head tensors are missing");
    if (w->output_norm) tensor_expect_layout(w->output_norm, DS4_TENSOR_BF16, 1, DS4_N_EMBD, 0, 0);
    if (w->output) {
        flint_expect_quant(w->output, &g_flint_output_aux, DS4_TENSOR_FLINT_U4, DS4_N_EMBD, DS4_N_VOCAB);
    }
    for (uint32_t il = layer_start; il <= layer_end; il++) {
        const ds4_layer_weights *l = &w->layer[il];
        const flint_layer_aux *a = &g_flint_layer_aux[il];
        if (!weights_flint_layer_has_required(l, il)) {
            fprintf(stderr, "ds4: Flint layer %u is missing required tensors\n", il);
            exit(1);
        }
        tensor_expect_layout(l->attn_norm, DS4_TENSOR_BF16, 1, DS4_N_EMBD, 0, 0);
        flint_expect_quant(l->attn_q, &a->attn_q, DS4_TENSOR_FLINT_U4, DS4_N_EMBD, q_dim);
        flint_expect_quant(l->attn_k, &a->attn_k, DS4_TENSOR_FLINT_U4, DS4_N_EMBD, kv_dim);
        flint_expect_quant(l->attn_v, &a->attn_v, DS4_TENSOR_FLINT_U4, DS4_N_EMBD, kv_dim);
        flint_expect_quant(l->attn_output, &a->attn_output, DS4_TENSOR_FLINT_U4, q_dim, DS4_N_EMBD);
        tensor_expect_layout(l->attn_q_norm, DS4_TENSOR_BF16, 1, DS4_N_HEAD_DIM, 0, 0);
        tensor_expect_layout(l->attn_k_norm, DS4_TENSOR_BF16, 1, DS4_N_HEAD_DIM, 0, 0);
        tensor_expect_layout(l->attn_sinks, DS4_TENSOR_BF16, 1, DS4_N_HEAD, 0, 0);
        tensor_expect_layout(l->ffn_norm, DS4_TENSOR_BF16, 1, DS4_N_EMBD, 0, 0);
        if (il < DS4_N_LEADING_DENSE) {
            flint_expect_quant(l->ffn_gate, &a->ffn_gate, DS4_TENSOR_FLINT_U4, DS4_N_EMBD, DS4_N_FF_DENSE);
            flint_expect_quant(l->ffn_up, &a->ffn_up, DS4_TENSOR_FLINT_U4, DS4_N_EMBD, DS4_N_FF_DENSE);
            flint_expect_quant(l->ffn_down, &a->ffn_down, DS4_TENSOR_FLINT_U4, DS4_N_FF_DENSE, DS4_N_EMBD);
        } else {
            tensor_expect_layout(l->ffn_gate_inp, DS4_TENSOR_BF16, 2, DS4_N_EMBD, DS4_N_EXPERT, 0);
            flint_expect_quant(l->ffn_gate_shexp, &a->ffn_gate_shexp, DS4_TENSOR_FLINT_U4, DS4_N_EMBD, DS4_FLINT_N_FF_SHARED);
            flint_expect_quant(l->ffn_up_shexp, &a->ffn_up_shexp, DS4_TENSOR_FLINT_U4, DS4_N_EMBD, DS4_FLINT_N_FF_SHARED);
            flint_expect_quant(l->ffn_down_shexp, &a->ffn_down_shexp, DS4_TENSOR_FLINT_U4, DS4_FLINT_N_FF_SHARED, DS4_N_EMBD);
            tensor_expect_layout(a->ffn_exps_pack, DS4_TENSOR_FLINT_PACK, 2, DS4_FLINT_EXPERT_PACK_BYTES, DS4_N_EXPERT, 0);
            if (a->ffn_exps_pack->abs_offset % 256u != 0) {
                fprintf(stderr, "ds4: Flint layer %u expert pack is not 256-byte aligned\n", il);
                exit(1);
            }
        }
    }
}
