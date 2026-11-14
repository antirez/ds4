/* Recursive Flint (fork-owned), textually included into ds4.c right after
 * the Qwen3.8 layer predicates.  docs/RECURSIVE_FLINT.md is the plan.
 *
 * Flint is its own model family: 49 layers of GQA attention (32 query heads,
 * 4 KV heads of 128, per-head QK RMSNorm, learned per-head attention sinks),
 * two leading dense SwiGLU layers and 47 MoE layers of 256 experts with
 * sigmoid top-6 routing plus one shared expert.  35 layers attend over a
 * sliding window of 513 positions with NeoX RoPE; the 14 layers listed in
 * DS4_FLINT_GLOBAL_LAYER_MASK attend over the whole context without RoPE.
 *
 * The shape holds only Flint's numbers.  Fields that belong to other
 * families (MLA, DSA indexer, hyper-connections, GDN, PLE, MTP) stay zero;
 * every shared function that would read them is redirected to Flint code
 * before it does.  Fork families and variants start at 200 so that an
 * upstream addition never collides with them. */
static const ds4_shape DS4_SHAPE_FLINT = {
    .name = "Recursive Flint",
    .family = DS4_MODEL_FAMILY_FLINT,
    .variant = DS4_VARIANT_FLINT,
    .n_layer = 49,
    .n_embd = 2048,
    .n_vocab = 150144,
    .n_head = 32,
    .n_head_kv = 4,
    .n_head_dim = 128,
    .n_value_dim = 128,
    .n_rot = 128,
    .n_expert = 256,
    .n_expert_used = 6,
    .n_expert_shared = 1,
    .n_ff_exp = 512,
    .n_ff_dense = 6144,
    .n_swa = 513,
    .n_leading_dense = 2,
    .rms_eps = 1.0e-6f,
    .rope_freq_base = 10000.0f,
    .rope_orig_ctx = 262144,
};

/* Shared-expert width differs from the routed one; the upstream shape has a
 * single expert width, so Flint carries its own constant. */
#define DS4_FLINT_N_FF_SHARED 1024u
/* One routed expert as stored: gate, up and down with their scales and
 * biases, contiguous (gguf-tools/flint_int4_to_gguf.py, flint.expert_pack.*). */
#define DS4_FLINT_EXPERT_PACK_BYTES 1769472u
/* Full-attention (no RoPE, whole-context KV) layers: 0, 1, 5, 9, ..., 45, 48. */
#define DS4_FLINT_N_GLOBAL_LAYER 14u
#define DS4_FLINT_GLOBAL_LAYER_MASK \
    ((1ull << 0) | (1ull << 1) | (1ull << 5) | (1ull << 9) | (1ull << 13) | \
     (1ull << 17) | (1ull << 21) | (1ull << 25) | (1ull << 29) | (1ull << 33) | \
     (1ull << 37) | (1ull << 41) | (1ull << 45) | (1ull << 48))

static bool ds4_model_is_flint(void) {
    return DS4_MODEL_FAMILY == DS4_MODEL_FAMILY_FLINT;
}

static bool ds4_flint_layer_is_global(uint32_t il) {
    return il < 64u && ((DS4_FLINT_GLOBAL_LAYER_MASK >> il) & 1ull) != 0;
}
