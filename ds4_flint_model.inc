/* Recursive Flint (fork-owned): GGUF metadata validation and the engine
 * gate.  Textually included into ds4.c just before config_validate_model,
 * which dispatches the architecture string "flint" here.  Weight binding,
 * the per-layer descriptor table and the graph land in later steps of
 * docs/RECURSIVE_FLINT.md. */

static void config_validate_flint_model(const ds4_model *m) {
    g_ds4_shape = DS4_SHAPE_FLINT;
    memset(g_ds4_compress_ratios, 0, sizeof(g_ds4_compress_ratios));

    config_expect_u32("block_count", required_u32(m, "flint.block_count"), DS4_N_LAYER);
    config_expect_u32("context_length", required_u32(m, "flint.context_length"),
                      (uint32_t)g_ds4_shape.rope_orig_ctx);
    config_expect_u32("embedding_length", required_u32(m, "flint.embedding_length"), DS4_N_EMBD);
    config_expect_u32("vocab_size", required_u32(m, "flint.vocab_size"), DS4_N_VOCAB);
    config_expect_u32("feed_forward_length", required_u32(m, "flint.feed_forward_length"),
                      DS4_N_FF_DENSE);
    config_expect_u32("expert_feed_forward_length",
                      required_u32(m, "flint.expert_feed_forward_length"), DS4_N_FF_EXP);
    config_expect_u32("expert_shared_feed_forward_length",
                      required_u32(m, "flint.expert_shared_feed_forward_length"),
                      DS4_FLINT_N_FF_SHARED);
    config_expect_u32("expert_count", required_u32(m, "flint.expert_count"), DS4_N_EXPERT);
    config_expect_u32("expert_used_count", required_u32(m, "flint.expert_used_count"),
                      DS4_N_EXPERT_USED);
    config_expect_u32("leading_dense_block_count",
                      required_u32(m, "flint.leading_dense_block_count"), DS4_N_LEADING_DENSE);
    config_expect_u32("attention.head_count",
                      required_u32(m, "flint.attention.head_count"), DS4_N_HEAD);
    config_expect_u32("attention.head_count_kv",
                      required_u32(m, "flint.attention.head_count_kv"), DS4_N_HEAD_KV);
    config_expect_u32("attention.key_length",
                      required_u32(m, "flint.attention.key_length"), DS4_N_HEAD_DIM);
    config_expect_u32("attention.value_length",
                      required_u32(m, "flint.attention.value_length"), DS4_N_VALUE_DIM);
    config_expect_u32("attention.sliding_window",
                      required_u32(m, "flint.attention.sliding_window"), DS4_N_SWA);
    config_expect_epsilon("attention.layer_norm_rms_epsilon",
                          required_f32(m, "flint.attention.layer_norm_rms_epsilon"),
                          DS4_RMS_EPS);
    config_expect_u32("rope.dimension_count",
                      required_u32(m, "flint.rope.dimension_count"), DS4_N_ROT);
    config_expect_f32("rope.freq_base", required_f32(m, "flint.rope.freq_base"),
                      DS4_ROPE_FREQ_BASE);

    uint32_t global_layers[DS4_FLINT_N_GLOBAL_LAYER];
    uint32_t n_global = 0;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        if (ds4_flint_layer_is_global(il)) global_layers[n_global++] = il;
    }
    if (n_global != DS4_FLINT_N_GLOBAL_LAYER) ds4_die("flint: global layer mask is inconsistent");
    config_expect_u32_array(m, "flint.full_attention_layers", global_layers, n_global);

    /* Only the INT4 checkpoint layout rec runs is supported: MLX affine
     * int4, groups of 64, 8-bit embeddings (docs/RECURSIVE_FLINT.md). */
    ds4_str fmt = {0};
    if (!model_get_string(m, "flint.weight_format", &fmt) || !ds4_streq(fmt, "mlx-affine-int4")) {
        fprintf(stderr, "ds4: flint.weight_format must be mlx-affine-int4\n");
        exit(1);
    }
    config_expect_u32("quant.bits", required_u32(m, "flint.quant.bits"), 4);
    config_expect_u32("quant.group_size", required_u32(m, "flint.quant.group_size"), 64);
    config_expect_u32("quant.embedding_bits", required_u32(m, "flint.quant.embedding_bits"), 8);
}

/* Engine gate, consulted right after config_validate_model (inspect-only
 * opens skip it). */
static bool flint_engine_supported(const ds4_engine_options *opt, ds4_backend backend) {
    const char *why = NULL;
    if (backend != DS4_BACKEND_METAL) why = "needs the Metal backend";
    else if (opt->first_token_test || opt->metal_graph_test) why = "has no CPU reference diagnostics";
    else if (opt->glm_mtp || (opt->mtp_path && opt->mtp_path[0])) why = "has no MTP weights";
    if (why) fprintf(stderr, "ds4: Recursive Flint %s\n", why);
    return why == NULL;
}

bool ds4_engine_is_flint(ds4_engine *e) {
    (void)e;
    return ds4_model_is_flint();
}
