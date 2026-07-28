int ds4_gpu_router_select_tensor(
        ds4_gpu_tensor       *selected,
        ds4_gpu_tensor       *weights,
        ds4_gpu_tensor       *probs,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                bias_offset,
        uint64_t                hash_offset,
        uint32_t                hash_rows,
        uint32_t                token,
        uint32_t                n_expert,
        uint32_t                n_expert_used,
        float                   expert_weight_scale,
        uint32_t                n_expert_groups,
        uint32_t                n_group_used,
        bool                    has_bias,
        bool                    hash_mode,
        const ds4_gpu_tensor *logits) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!selected || !weights || !probs || !logits || !model_map ||
        n_expert == 0 || n_expert_used == 0) return 0;
    if (hash_mode && token >= hash_rows) return 0;
    if (n_expert_groups > 1u || n_group_used > 0u) {
        fprintf(stderr, "ds4: Metal router group gating is not part of this DeepSeek V4 path\n");
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> logitsbuf = ds4_gpu_tensor_buffer(logits);
        id<MTLBuffer> selectedbuf = ds4_gpu_tensor_buffer(selected);
        id<MTLBuffer> weightsbuf = ds4_gpu_tensor_buffer(weights);
        id<MTLBuffer> probsbuf = ds4_gpu_tensor_buffer(probs);
        if (!logitsbuf || !selectedbuf || !weightsbuf || !probsbuf ||
            ds4_gpu_tensor_bytes(logits) < (uint64_t)n_expert * sizeof(float) ||
            ds4_gpu_tensor_bytes(selected) < (uint64_t)n_expert_used * sizeof(int) ||
            ds4_gpu_tensor_bytes(weights) < (uint64_t)n_expert_used * sizeof(float) ||
            ds4_gpu_tensor_bytes(probs) < (uint64_t)n_expert * sizeof(float)) {
            fprintf(stderr, "ds4: Metal router select received undersized buffers\n");
            return 0;
        }

        uint64_t bias_inner = 0;
        uint64_t hash_inner = 0;
        id<MTLBuffer> biasbuf = nil;
        id<MTLBuffer> hashbuf = nil;
        NSUInteger bias_set_offset = 0;
        NSUInteger hash_set_offset = 0;
        if (has_bias && !hash_mode) {
            const uint64_t bias_bytes = (uint64_t)n_expert * sizeof(float);
            biasbuf = ds4_gpu_wrap_model_range(model_map, model_size, bias_offset, bias_bytes, &bias_inner);
            if (!biasbuf) return 0;
            bias_set_offset = (NSUInteger)bias_inner;
        }
        if (hash_mode) {
            const uint64_t hash_bytes = (uint64_t)hash_rows * n_expert_used * sizeof(int32_t);
            hashbuf = ds4_gpu_wrap_model_range(model_map, model_size, hash_offset, hash_bytes, &hash_inner);
            if (!hashbuf) return 0;
            hash_set_offset = (NSUInteger)hash_inner;
        }

        const bool had_batch = g_batch_cb != nil;
        if (!had_batch && ds4_gpu_begin_commands() == 0) return 0;
        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        const int32_t token_i32 = (int32_t)token;
        int ok = cb &&
                 ds4_gpu_encode_router_select(cb,
                                                      selected,
                                                      weights,
                                                      probs,
                                                      logitsbuf,
                                                      ds4_gpu_tensor_offset(logits),
                                                      biasbuf,
                                                      bias_set_offset,
                                                      hashbuf,
                                                      hash_set_offset,
                                                      nil,
                                                      0,
                                                      &token_i32,
                                                      hash_rows,
                                                      1,
                                                      n_expert,
                                                      n_expert_used,
                                                      expert_weight_scale,
                                                      has_bias && !hash_mode,
                                                      hash_mode);
        if (!had_batch) {
            ok = ds4_gpu_end_commands() != 0 && ok;
        }
        if (!ok) return 0;
    }

    return 1;
}

int ds4_gpu_router_select_batch_tensor(
        ds4_gpu_tensor       *selected,
        ds4_gpu_tensor       *weights,
        ds4_gpu_tensor       *probs,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                bias_offset,
        uint64_t                hash_offset,
        uint32_t                hash_rows,
        uint32_t                n_expert_groups,
        uint32_t                n_group_used,
        bool                    has_bias,
        bool                    hash_mode,
        const ds4_gpu_tensor *logits,
        const ds4_gpu_tensor *tokens,
        uint32_t                n_expert,
        uint32_t                n_expert_used,
        float                   expert_weight_scale,
        uint32_t                n_tokens) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!selected || !weights || !probs || !logits || !tokens || !model_map ||
        n_expert == 0 || n_expert_used == 0 || n_tokens == 0) return 0;
    if (n_expert_groups > 1u || n_group_used > 0u) {
        fprintf(stderr, "ds4: Metal router group gating is not part of this DeepSeek V4 path\n");
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> logitsbuf = ds4_gpu_tensor_buffer(logits);
        id<MTLBuffer> selectedbuf = ds4_gpu_tensor_buffer(selected);
        id<MTLBuffer> weightsbuf = ds4_gpu_tensor_buffer(weights);
        id<MTLBuffer> probsbuf = ds4_gpu_tensor_buffer(probs);
        id<MTLBuffer> tokensbuf = ds4_gpu_tensor_buffer(tokens);
        if (!logitsbuf || !selectedbuf || !weightsbuf || !probsbuf || !tokensbuf ||
            ds4_gpu_tensor_bytes(logits) < (uint64_t)n_tokens * n_expert * sizeof(float) ||
            ds4_gpu_tensor_bytes(selected) < (uint64_t)n_tokens * n_expert_used * sizeof(int) ||
            ds4_gpu_tensor_bytes(weights) < (uint64_t)n_tokens * n_expert_used * sizeof(float) ||
            ds4_gpu_tensor_bytes(probs) < (uint64_t)n_tokens * n_expert * sizeof(float) ||
            ds4_gpu_tensor_bytes(tokens) < (uint64_t)n_tokens * sizeof(int32_t)) {
            fprintf(stderr, "ds4: Metal router batch select received undersized buffers\n");
            return 0;
        }

        uint64_t bias_inner = 0;
        uint64_t hash_inner = 0;
        id<MTLBuffer> biasbuf = nil;
        id<MTLBuffer> hashbuf = nil;
        NSUInteger bias_set_offset = 0;
        NSUInteger hash_set_offset = 0;
        if (has_bias && !hash_mode) {
            const uint64_t bias_bytes = (uint64_t)n_expert * sizeof(float);
            biasbuf = ds4_gpu_wrap_model_range(model_map, model_size, bias_offset, bias_bytes, &bias_inner);
            if (!biasbuf) return 0;
            bias_set_offset = (NSUInteger)bias_inner;
        }
        if (hash_mode) {
            const uint64_t hash_bytes = (uint64_t)hash_rows * n_expert_used * sizeof(int32_t);
            hashbuf = ds4_gpu_wrap_model_range(model_map, model_size, hash_offset, hash_bytes, &hash_inner);
            if (!hashbuf) return 0;
            hash_set_offset = (NSUInteger)hash_inner;
        }

        const bool had_batch = g_batch_cb != nil;
        if (!had_batch && ds4_gpu_begin_commands() == 0) return 0;
        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        int ok = cb &&
                 ds4_gpu_encode_router_select(cb,
                                                      selected,
                                                      weights,
                                                      probs,
                                                      logitsbuf,
                                                      ds4_gpu_tensor_offset(logits),
                                                      biasbuf,
                                                      bias_set_offset,
                                                      hashbuf,
                                                      hash_set_offset,
                                                      tokensbuf,
                                                      ds4_gpu_tensor_offset(tokens),
                                                      NULL,
                                                      hash_rows,
                                                      n_tokens,
                                                      n_expert,
                                                      n_expert_used,
                                                      expert_weight_scale,
                                                      has_bias && !hash_mode,
                                                      hash_mode);
        if (!had_batch) {
            ok = ds4_gpu_end_commands() != 0 && ok;
        }
        if (!ok) return 0;
    }

    return 1;
}

int ds4_gpu_routed_moe_set_selected_override(const int32_t *selected, uint32_t n_selected) {
    if (n_selected > DS4_METAL_MAX_ROUTED_EXPERT_USED ||
        (!selected && n_selected != 0)) return 0;
    for (uint32_t i = 0; i < n_selected; i++) {
        g_routed_moe_selected_override[i] = selected[i];
    }
    g_routed_moe_selected_override_n = n_selected;
    return 1;
}

int ds4_gpu_routed_moe_one_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        ds4_gpu_tensor       *experts,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                down_offset,
        uint32_t                gate_type,
        uint32_t                down_type,
        uint64_t                gate_expert_bytes,
        uint64_t                gate_row_bytes,
        uint64_t                down_expert_bytes,
        uint64_t                down_row_bytes,
        uint32_t                expert_in_dim,
        uint32_t                expert_mid_dim,
        uint32_t                out_dim,
        const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *weights,
        uint32_t                n_total_expert,
        uint32_t                n_expert,
        float                   clamp,
        const ds4_gpu_tensor *x,
        const ds4_gpu_tensor *add_in,
        uint32_t                layer_index,
        bool                    force_resident) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    /* TP sharding: only the owned contiguous expert range is mapped,
     * so bind from the owned base, validate only its bytes, and tell the
     * kernels the first expert id present at that base. */
    uint32_t first_expert = 0;
    uint32_t n_bind_expert = 0;
    ds4_gpu_tp_expert_range(n_total_expert, &first_expert, &n_bind_expert);
    const int32_t tp_expert_base_host = (int32_t)first_expert;
    gate_offset += (uint64_t)first_expert * gate_expert_bytes;
    up_offset += (uint64_t)first_expert * gate_expert_bytes;
    down_offset += (uint64_t)first_expert * down_expert_bytes;

    if (!out || !gate || !up || !mid || !x || !model_map || !selected || !weights ||
        n_total_expert == 0 || n_expert == 0 ||
        n_expert > DS4_METAL_MAX_ROUTED_EXPERT_USED ||
        gate_expert_bytes == 0 || down_expert_bytes == 0 ||
        gate_row_bytes == 0 || down_row_bytes == 0) {
        if (getenv("DS4_GLM_TP_DEBUG")) fprintf(stderr, "ds4: routed_moe_one silent return at line %d\n", 32044);
        return 0;
    }
    if ((expert_in_dim % 256u) != 0 || (expert_mid_dim % 256u) != 0) { if (getenv("DS4_GLM_TP_DEBUG")) fprintf(stderr, "ds4: routed_moe_one silent return at line %d\n", 32047); return 0; }
    ds4_gpu_stream_expert_cache_note_token(layer_index);

    @autoreleasepool {
        id<MTLBuffer> xbuf = ds4_gpu_tensor_buffer(x);
        id<MTLBuffer> gatebuf = ds4_gpu_tensor_buffer(gate);
        id<MTLBuffer> upbuf = ds4_gpu_tensor_buffer(up);
        id<MTLBuffer> midbuf = ds4_gpu_tensor_buffer(mid);
        id<MTLBuffer> outbuf = ds4_gpu_tensor_buffer(out);
        id<MTLBuffer> expertsbuf = ds4_gpu_tensor_buffer(experts);
        id<MTLBuffer> selectedbuf = ds4_gpu_tensor_buffer(selected);
        id<MTLBuffer> selected_exec_buf = selectedbuf;
        NSUInteger selected_exec_off = ds4_gpu_tensor_offset(selected);
        id<MTLBuffer> weightsbuf = ds4_gpu_tensor_buffer(weights);
        const uint64_t x_bytes = (uint64_t)expert_in_dim * sizeof(float);
        const uint64_t mid_bytes = (uint64_t)n_expert * expert_mid_dim * sizeof(float);
        const uint64_t out_bytes = (uint64_t)out_dim * sizeof(float);
        if (!xbuf || !gatebuf || !upbuf || !midbuf || !outbuf || !selectedbuf || !weightsbuf ||
            ds4_gpu_tensor_bytes(x) < x_bytes ||
            ds4_gpu_tensor_bytes(gate) < mid_bytes ||
            ds4_gpu_tensor_bytes(up) < mid_bytes ||
            ds4_gpu_tensor_bytes(mid) < mid_bytes ||
            ds4_gpu_tensor_bytes(out) < out_bytes ||
            ds4_gpu_tensor_bytes(selected) < (uint64_t)n_expert * sizeof(int) ||
            ds4_gpu_tensor_bytes(weights) < (uint64_t)n_expert * sizeof(float)) {
            fprintf(stderr, "ds4: Metal routed tensor MoE received undersized activation buffers\n");
            return 0;
        }
        if (n_expert > 1 &&
            (!expertsbuf ||
             ds4_gpu_tensor_bytes(experts) < (uint64_t)n_expert * out_dim * sizeof(float))) {
            fprintf(stderr, "ds4: Metal routed tensor MoE received undersized expert output buffer\n");
            return 0;
        }

        if ((uint64_t)n_total_expert > UINT64_MAX / gate_expert_bytes ||
            (uint64_t)n_total_expert > UINT64_MAX / down_expert_bytes) {
            fprintf(stderr, "ds4: Metal routed MoE tensor byte size overflow\n");
            return 0;
        }
        const uint64_t gate_tensor_bytes = (uint64_t)n_bind_expert * gate_expert_bytes;
        const uint64_t down_tensor_bytes = (uint64_t)n_bind_expert * down_expert_bytes;
        uint64_t gate_inner = 0;
        uint64_t up_inner = 0;
        uint64_t down_inner = 0;
        id<MTLBuffer> gate_buf = nil;
        id<MTLBuffer> up_buf = nil;
        id<MTLBuffer> down_buf = nil;
        __unsafe_unretained id<MTLBuffer> gate_slot_bufs[DS4_METAL_MAX_ROUTED_EXPERT_USED] = { nil };
        __unsafe_unretained id<MTLBuffer> up_slot_bufs[DS4_METAL_MAX_ROUTED_EXPERT_USED] = { nil };
        __unsafe_unretained id<MTLBuffer> down_slot_bufs[DS4_METAL_MAX_ROUTED_EXPERT_USED] = { nil };
        ds4_gpu_stream_expert_cache_entry *stream_slot_entries[DS4_METAL_MAX_ROUTED_EXPERT_USED] = { NULL };
        NSUInteger gate_slot_offsets[DS4_METAL_MAX_ROUTED_EXPERT_USED] = { 0 };
        NSUInteger up_slot_offsets[DS4_METAL_MAX_ROUTED_EXPERT_USED] = { 0 };
        NSUInteger down_slot_offsets[DS4_METAL_MAX_ROUTED_EXPERT_USED] = { 0 };
        uint64_t stream_gate_abs_offsets[DS4_METAL_MAX_ROUTED_EXPERT_USED] = { 0 };
        uint64_t stream_up_abs_offsets[DS4_METAL_MAX_ROUTED_EXPERT_USED] = { 0 };
        uint64_t stream_down_abs_offsets[DS4_METAL_MAX_ROUTED_EXPERT_USED] = { 0 };
        id<MTLBuffer> stream_gate_addr_buf = nil;
        id<MTLBuffer> stream_up_addr_buf = nil;
        id<MTLBuffer> stream_down_addr_buf = nil;
        bool use_stream_expert_addr_table = false;
        bool use_stream_expert_masked_addr_table = false;
        bool use_stream_compact_addr_table = false;
        bool use_stream_expert_cache = false;
        bool use_stream_expert_split_candidate = false;
        bool use_stream_expert_split_deferred = false;
        bool stream_expert_split_completed = false;
        uint32_t stream_expert_resident_mask = 0;
        uint32_t stream_expert_missing_mask = 0;
        __unsafe_unretained id<MTLBuffer> gate_group6_bufs[6] = { nil, nil, nil, nil, nil, nil };
        __unsafe_unretained id<MTLBuffer> up_group6_bufs[6] = { nil, nil, nil, nil, nil, nil };
        __unsafe_unretained id<MTLBuffer> down_group6_bufs[6] = { nil, nil, nil, nil, nil, nil };
        NSUInteger gate_group6_offsets[6] = { 0, 0, 0, 0, 0, 0 };
        NSUInteger up_group6_offsets[6] = { 0, 0, 0, 0, 0, 0 };
        NSUInteger down_group6_offsets[6] = { 0, 0, 0, 0, 0, 0 };
        __unsafe_unretained id<MTLBuffer> gate_group8_bufs[8] = { nil, nil, nil, nil, nil, nil, nil, nil };
        __unsafe_unretained id<MTLBuffer> up_group8_bufs[8] = { nil, nil, nil, nil, nil, nil, nil, nil };
        __unsafe_unretained id<MTLBuffer> down_group8_bufs[8] = { nil, nil, nil, nil, nil, nil, nil, nil };
        NSUInteger gate_group8_offsets[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
        NSUInteger up_group8_offsets[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
        NSUInteger down_group8_offsets[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
        __unsafe_unretained id<MTLBuffer> gate_group24_bufs[24] = { nil };
        __unsafe_unretained id<MTLBuffer> up_group24_bufs[24] = { nil };
        __unsafe_unretained id<MTLBuffer> down_group24_bufs[24] = { nil };
        NSUInteger gate_group24_offsets[24] = { 0 };
        NSUInteger up_group24_offsets[24] = { 0 };
        NSUInteger down_group24_offsets[24] = { 0 };
        DS4MetalQ4ExpertTable *gate_table = nil;
        DS4MetalQ4ExpertTable *up_table = nil;
        DS4MetalQ4ExpertTable *down_table = nil;
        id q4_table_layer_residency = nil;
        int32_t selected_ids[DS4_METAL_MAX_ROUTED_EXPERT_USED] = { 0 };

        const uint32_t n_tokens = 1;
        const uint32_t pair_rows = n_tokens * n_expert;
        const uint64_t down_scratch_bytes = (uint64_t)pair_rows * out_dim * sizeof(float);
        if ((n_expert > 1 && !expertsbuf &&
             !ds4_gpu_ensure_scratch_buffer(&g_moe_down_scratch_buffer,
                                              &g_moe_down_scratch_bytes,
                                              (NSUInteger)down_scratch_bytes,
                                              "ds4_moe_down_scratch"))) {
            if (getenv("DS4_GLM_TP_DEBUG")) fprintf(stderr, "ds4: routed_moe_one silent return at line %d\n", 32148);
            return 0;
        }

        const uint32_t gate_nr0 = ds4_gpu_routed_mv_nr0(gate_type);
        const uint32_t down_nr0 = ds4_gpu_routed_mv_nr0(down_type);
        id<MTLComputePipelineState> gate_mv_pipeline = ds4_gpu_routed_mv_pipeline(gate_type);
        id<MTLComputePipelineState> down_mv_pipeline = ds4_gpu_routed_mv_pipeline(down_type);
        if (gate_nr0 == 0 || down_nr0 == 0 || !gate_mv_pipeline || !down_mv_pipeline) {
            fprintf(stderr, "ds4: unsupported Metal routed MoE quant types gate=%u down=%u\n",
                    gate_type, down_type);
            if (getenv("DS4_GLM_TP_DEBUG")) fprintf(stderr, "ds4: routed_moe_one silent return at line %d\n", 32158);
            return 0;
        }

        ds4_gpu_mul_mv_id_args gate_args =
            ds4_gpu_make_mul_mv_id_args(expert_in_dim, expert_mid_dim, n_total_expert,
                                          gate_row_bytes, gate_expert_bytes,
                                          1, n_expert, n_tokens, gate_nr0);
        ds4_gpu_mul_mv_id_args down_args =
            ds4_gpu_make_mul_mv_id_args(expert_mid_dim, out_dim, n_total_expert,
                                          down_row_bytes, down_expert_bytes,
                                          n_expert, n_expert, n_tokens, down_nr0);
        /* Tensor-parallel expert ownership; non-TP calls keep tp_world at 1. */
        gate_args.tp_rank = g_tp_split_rank;
        gate_args.tp_world = g_tp_split_world;
        gate_args.tp_expert_base = tp_expert_base_host;
        down_args.tp_rank = g_tp_split_rank;
        down_args.tp_world = g_tp_split_world;
        down_args.tp_addend = add_in != NULL;
        down_args.tp_expert_base = tp_expert_base_host;

        const NSUInteger gate_smem = ds4_gpu_routed_mv_smem(gate_type);
        const NSUInteger down_smem = ds4_gpu_routed_mv_smem(down_type);
        const NSUInteger gate_nsg = ds4_gpu_routed_mv_nsg(gate_type);
        const NSUInteger down_nsg = ds4_gpu_routed_mv_nsg(down_type);
        const bool gate_rows_per_group_is_nr0 = ds4_gpu_routed_mv_rows_per_group_is_nr0(gate_type);
        const bool down_rows_per_group_is_nr0 = ds4_gpu_routed_mv_rows_per_group_is_nr0(down_type);
        int ok = 1;
        const bool write_clamped_moe =
            getenv("DS4_METAL_MOE_WRITE_CLAMPED_ACT") != NULL;
        id<MTLComputePipelineState> pair_swiglu_pipeline = nil;
        if (gate_type == DS4_METAL_TENSOR_IQ2_XXS) {
            pair_swiglu_pipeline = g_moe_mul_mv_id_iq2_xxs_pair_swiglu_pipeline;
        } else if (gate_type == DS4_METAL_TENSOR_Q4_K) {
            pair_swiglu_pipeline = g_moe_mul_mv_id_q4_k_pair_swiglu_pipeline;
        }
        const bool fuse_pair_swiglu =
            !g_quality_mode &&
            !write_clamped_moe &&
            getenv("DS4_METAL_DISABLE_ROUTED_PAIR_SWIGLU_FUSION") == NULL &&
            pair_swiglu_pipeline != nil;
        id<MTLComputePipelineState> down_sum6_pipeline = nil;
        if (down_type == DS4_METAL_TENSOR_Q2_K) {
            down_sum6_pipeline = g_moe_mul_mv_id_q2_k_sum6_pipeline;
        } else if (down_type == DS4_METAL_TENSOR_Q4_K) {
            down_sum6_pipeline = g_moe_mul_mv_id_q4_k_sum6_pipeline;
        } else if (down_type == DS4_METAL_TENSOR_IQ2_XXS &&
                   g_tp_split_world == 2) {
            /* IQ2 down-sum exists for the GLM TP resident split only; the
             * streaming paths must keep their addr/masked chain (expert
             * bytes are not at their model-map offsets when streamed). */
            down_sum6_pipeline = g_moe_mul_mv_id_iq2_xxs_sum6_pipeline;
        }
        const bool direct_down_sum =
            !g_quality_mode &&
            (n_expert == 6 || (n_expert == 8 && g_tp_split_world == 2)) &&
            n_tokens == 1 &&
            down_sum6_pipeline != nil;
        /* The expert-ownership split lives only in the fused id pair+sum6
         * kernels; every other routed variant would silently compute full
         * sums on both ranks and double the combine.  Fail fast instead. */
        if ((g_tp_split_world > 1 || add_in) && !(fuse_pair_swiglu && direct_down_sum)) {
            fprintf(stderr, "ds4: tensor-parallel routed MoE requires the fused pair+sum6 decode path\n");
            return 0;
        }
        const uint64_t q4_selected_min_tensor_bytes = 2ull * 1024ull * 1024ull * 1024ull;
        /*
         * The grouped Q4 experiment keeps selected IDs on GPU, but it also walks
         * every expert window in the layer.  On PRO Q4 this measured far slower
         * than the active selected-slot path, so keep it opt-in for profiling.
         */
        const bool enable_q4_grouped_experts =
            getenv("DS4_METAL_ENABLE_Q4_GROUPED_EXPERTS") != NULL &&
            getenv("DS4_METAL_DISABLE_Q4_GROUPED_EXPERTS") == NULL;
        const bool use_q4_grouped_experts =
            gate_type == DS4_METAL_TENSOR_Q4_K &&
            down_type == DS4_METAL_TENSOR_Q4_K &&
            n_expert == 6 &&
            n_tokens == 1 &&
            n_total_expert >= 128 &&
            gate_tensor_bytes >= q4_selected_min_tensor_bytes &&
            down_tensor_bytes >= q4_selected_min_tensor_bytes &&
            fuse_pair_swiglu &&
            direct_down_sum &&
            g_moe_mul_mv_group_q4_k_pair_swiglu_pipeline != nil &&
            g_moe_mul_mv_group_q4_k_sum6_pipeline != nil &&
            enable_q4_grouped_experts;
        const uint32_t q4_expert_group_size =
            use_q4_grouped_experts ? ds4_gpu_q4_expert_group_size(n_total_expert) : 0;
        const bool q4_grouped_boundary =
            use_q4_grouped_experts &&
            g_batch_cb != nil &&
            getenv("DS4_METAL_DISABLE_Q4_GROUPED_BOUNDARY") == NULL;
        const bool q4_grouped_cache_views =
            getenv("DS4_METAL_Q4_GROUPED_CACHE_VIEWS") != NULL;
        const uint32_t q4_group6_expert_group_size = 64;
        const bool use_q4_group6_experts =
            !use_q4_grouped_experts &&
            gate_type == DS4_METAL_TENSOR_Q4_K &&
            down_type == DS4_METAL_TENSOR_Q4_K &&
            n_expert == 6 &&
            n_tokens == 1 &&
            n_total_expert == q4_group6_expert_group_size * 6u &&
            gate_tensor_bytes >= q4_selected_min_tensor_bytes &&
            down_tensor_bytes >= q4_selected_min_tensor_bytes &&
            fuse_pair_swiglu &&
            direct_down_sum &&
            g_moe_mul_mv_group6_q4_k_pair_swiglu_pipeline != nil &&
            g_moe_mul_mv_group6_q4_k_sum6_pipeline != nil &&
            getenv("DS4_METAL_ENABLE_Q4_GROUP6_EXPERT_TABLE") != NULL &&
            getenv("DS4_METAL_DISABLE_Q4_GROUP6_EXPERT_TABLE") == NULL;
        const uint32_t q4_group8_expert_group_size = 48;
        const bool use_q4_group8_experts =
            !use_q4_grouped_experts &&
            !use_q4_group6_experts &&
            gate_type == DS4_METAL_TENSOR_Q4_K &&
            down_type == DS4_METAL_TENSOR_Q4_K &&
            n_expert == 6 &&
            n_tokens == 1 &&
            n_total_expert == q4_group8_expert_group_size * 8u &&
            gate_tensor_bytes >= q4_selected_min_tensor_bytes &&
            down_tensor_bytes >= q4_selected_min_tensor_bytes &&
            fuse_pair_swiglu &&
            direct_down_sum &&
            g_moe_mul_mv_group8_q4_k_pair_swiglu_pipeline != nil &&
            g_moe_mul_mv_group8_q4_k_sum6_pipeline != nil &&
            getenv("DS4_METAL_ENABLE_Q4_GROUP8_EXPERT_TABLE") != NULL &&
            getenv("DS4_METAL_DISABLE_Q4_GROUP8_EXPERT_TABLE") == NULL;
        const uint32_t q4_group24_expert_group_size = 16;
        const bool use_q4_group24_experts =
            !use_q4_grouped_experts &&
            !use_q4_group6_experts &&
            !use_q4_group8_experts &&
            gate_type == DS4_METAL_TENSOR_Q4_K &&
            down_type == DS4_METAL_TENSOR_Q4_K &&
            n_expert == 6 &&
            n_tokens == 1 &&
            n_total_expert == q4_group24_expert_group_size * 24u &&
            gate_tensor_bytes >= q4_selected_min_tensor_bytes &&
            down_tensor_bytes >= q4_selected_min_tensor_bytes &&
            direct_down_sum &&
            g_moe_mul_mv_group24_q4_k_id_pipeline != nil &&
            g_moe_mul_mv_group24_q4_k_sum6_pipeline != nil &&
            getenv("DS4_METAL_ENABLE_Q4_GROUP24_EXPERT_TABLE") != NULL &&
            getenv("DS4_METAL_DISABLE_Q4_GROUP24_EXPERT_TABLE") == NULL;
        const bool q4_group24_exact_views =
            use_q4_group24_experts &&
            getenv("DS4_METAL_Q4_GROUP24_EXACT_VIEWS") != NULL &&
            getenv("DS4_METAL_Q4_GROUP24_BASE_VIEWS") == NULL;
        const uint64_t max_buffer_len = g_device ? (uint64_t)[g_device maxBufferLength] : 0;
        const bool can_wrap_q4_exact_tensors =
            max_buffer_len != 0 &&
            gate_tensor_bytes <= max_buffer_len &&
            down_tensor_bytes <= max_buffer_len;
        /*
         * The full-tensor Q4 ID path is the closest arithmetic analogue to IQ2,
         * but PRO Q4 routed tensors are multi-GiB.  Keep it opt-in: even with
         * per-layer command boundaries it is much slower than binding only the
         * six active experts on current M3 Ultra Metal.
         */
        const bool enable_q4_exact_tensor_id =
            getenv("DS4_METAL_ENABLE_Q4_EXACT_TENSOR_ID") != NULL &&
            getenv("DS4_METAL_DISABLE_Q4_EXACT_TENSOR_ID") == NULL;
        const bool use_q4_exact_tensor_id =
            !use_q4_grouped_experts &&
            !use_q4_group6_experts &&
            !use_q4_group8_experts &&
            !use_q4_group24_experts &&
            gate_type == DS4_METAL_TENSOR_Q4_K &&
            down_type == DS4_METAL_TENSOR_Q4_K &&
            n_expert == 6 &&
            n_tokens == 1 &&
            n_total_expert == 384 &&
            gate_tensor_bytes >= q4_selected_min_tensor_bytes &&
            down_tensor_bytes >= q4_selected_min_tensor_bytes &&
            can_wrap_q4_exact_tensors &&
            fuse_pair_swiglu &&
            direct_down_sum &&
            enable_q4_exact_tensor_id;
        const bool q4_exact_boundary =
            use_q4_exact_tensor_id &&
            g_batch_cb != nil &&
            getenv("DS4_METAL_DISABLE_Q4_EXACT_BOUNDARY") == NULL;
        const bool q4_expert_table_auto =
            gate_type == DS4_METAL_TENSOR_Q4_K &&
            down_type == DS4_METAL_TENSOR_Q4_K &&
            ds4_gpu_pro_q4_expert_table_auto_enabled(n_total_expert,
                                                     n_expert,
                                                     gate_tensor_bytes,
                                                     down_tensor_bytes);
        const bool q4_expert_address_auto =
            gate_type == DS4_METAL_TENSOR_Q4_K &&
            down_type == DS4_METAL_TENSOR_Q4_K &&
            ds4_gpu_pro_q4_expert_address_auto_enabled(n_total_expert,
                                                       n_expert,
                                                       gate_tensor_bytes,
                                                       down_tensor_bytes);
        const bool q4_table_queue_residency =
            ds4_gpu_q4_table_queue_residency_enabled(q4_expert_table_auto ||
                                                     q4_expert_address_auto);
        const bool use_q4_expert_address_table =
            !use_q4_grouped_experts &&
            !use_q4_group6_experts &&
            !use_q4_group8_experts &&
            !use_q4_group24_experts &&
            !use_q4_exact_tensor_id &&
            gate_type == DS4_METAL_TENSOR_Q4_K &&
            down_type == DS4_METAL_TENSOR_Q4_K &&
            n_expert == 6 &&
            n_tokens == 1 &&
            n_total_expert == 384 &&
            gate_tensor_bytes >= q4_selected_min_tensor_bytes &&
            down_tensor_bytes >= q4_selected_min_tensor_bytes &&
            fuse_pair_swiglu &&
            direct_down_sum &&
            g_moe_mul_mv_addr_q4_k_pair_swiglu_pipeline != nil &&
            g_moe_mul_mv_addr_q4_k_sum6_pipeline != nil &&
            (getenv("DS4_METAL_ENABLE_Q4_EXPERT_ADDRESS_TABLE") != NULL ||
             q4_expert_address_auto) &&
            (getenv("DS4_METAL_Q4_ADDR_USE_RESOURCES") != NULL ||
             getenv("DS4_METAL_Q4_TABLE_RESIDENCY_SET") != NULL ||
             q4_table_queue_residency ||
             ds4_gpu_q4_table_model_residency_enabled() ||
             getenv("DS4_METAL_USE_QUEUE_RESIDENCY_SET") != NULL) &&
            getenv("DS4_METAL_DISABLE_Q4_EXPERT_ADDRESS_TABLE") == NULL;
        const bool enable_q4_expert_table =
            getenv("DS4_METAL_ENABLE_Q4_EXPERT_TABLE") != NULL ||
            q4_expert_table_auto;
        const bool use_q4_expert_table =
            !use_q4_grouped_experts &&
            !use_q4_group6_experts &&
            !use_q4_group8_experts &&
            !use_q4_group24_experts &&
            !use_q4_exact_tensor_id &&
            !use_q4_expert_address_table &&
            gate_type == DS4_METAL_TENSOR_Q4_K &&
            down_type == DS4_METAL_TENSOR_Q4_K &&
            n_expert == 6 &&
            n_tokens == 1 &&
            n_total_expert == 384 &&
            gate_tensor_bytes >= q4_selected_min_tensor_bytes &&
            down_tensor_bytes >= q4_selected_min_tensor_bytes &&
            fuse_pair_swiglu &&
            direct_down_sum &&
            g_moe_mul_mv_table_q4_k_pair_swiglu_pipeline != nil &&
            g_moe_mul_mv_table_q4_k_sum6_pipeline != nil &&
            g_moe_table_q4_pair_gate_encoder != nil &&
            g_moe_table_q4_pair_up_encoder != nil &&
            g_moe_table_q4_sum_down_encoder != nil &&
            enable_q4_expert_table &&
            (getenv("DS4_METAL_Q4_TABLE_USE_RESOURCES") != NULL ||
             getenv("DS4_METAL_Q4_TABLE_RESIDENCY_SET") != NULL ||
             q4_table_queue_residency ||
             ds4_gpu_q4_table_model_residency_enabled() ||
             getenv("DS4_METAL_USE_QUEUE_RESIDENCY_SET") != NULL) &&
            getenv("DS4_METAL_DISABLE_Q4_EXPERT_TABLE") == NULL;
        const bool q4_table_boundary =
            use_q4_expert_table &&
            g_batch_cb != nil &&
            getenv("DS4_METAL_Q4_TABLE_RESIDENCY_SET") != NULL &&
            !q4_table_queue_residency &&
            getenv("DS4_METAL_DISABLE_Q4_TABLE_BOUNDARY") == NULL;
        const bool enable_q4_gather_slots =
            getenv("DS4_METAL_ENABLE_Q4_GATHER_SLOTS") != NULL &&
            getenv("DS4_METAL_DISABLE_Q4_GATHER_SLOTS") == NULL;
        const bool use_q4_gather_slots =
            !use_q4_grouped_experts &&
            !use_q4_group6_experts &&
            !use_q4_group8_experts &&
            !use_q4_group24_experts &&
            !use_q4_exact_tensor_id &&
            !use_q4_expert_address_table &&
            !use_q4_expert_table &&
            gate_type == DS4_METAL_TENSOR_Q4_K &&
            down_type == DS4_METAL_TENSOR_Q4_K &&
            n_expert == 6 &&
            n_tokens == 1 &&
            n_total_expert == q4_group6_expert_group_size * 6u &&
            gate_tensor_bytes >= q4_selected_min_tensor_bytes &&
            down_tensor_bytes >= q4_selected_min_tensor_bytes &&
            fuse_pair_swiglu &&
            direct_down_sum &&
            g_moe_q4_gather_slots6_pipeline != nil &&
            g_moe_mul_mv_slots6_q4_k_pair_swiglu_pipeline != nil &&
            g_moe_mul_mv_slots6_q4_k_sum6_pipeline != nil &&
            enable_q4_gather_slots;
        const bool use_q4_selected_slots =
            !force_resident &&
            !use_q4_grouped_experts &&
            !use_q4_group6_experts &&
            !use_q4_group8_experts &&
            !use_q4_group24_experts &&
            !use_q4_exact_tensor_id &&
            !use_q4_expert_address_table &&
            !use_q4_expert_table &&
            !use_q4_gather_slots &&
            gate_type == DS4_METAL_TENSOR_Q4_K &&
            down_type == DS4_METAL_TENSOR_Q4_K &&
            n_expert == 6 &&
            n_tokens == 1 &&
            n_total_expert >= 128 &&
            (g_ssd_streaming_mode ||
             (gate_tensor_bytes >= q4_selected_min_tensor_bytes &&
              down_tensor_bytes >= q4_selected_min_tensor_bytes)) &&
            fuse_pair_swiglu &&
            direct_down_sum &&
            ds4_gpu_q4_selected_paths_allowed() &&
            g_moe_mul_mv_slots6_q4_k_pair_swiglu_pipeline != nil &&
            g_moe_mul_mv_slots6_q4_k_sum6_pipeline != nil &&
            getenv("DS4_METAL_DISABLE_Q4_SELECTED_EXPERT_VIEWS") == NULL;
        const bool use_iq2_selected_slots =
            !force_resident &&
            g_ssd_streaming_mode &&
            gate_type == DS4_METAL_TENSOR_IQ2_XXS &&
            down_type == DS4_METAL_TENSOR_Q2_K &&
            n_expert == 6 &&
            n_tokens == 1 &&
            fuse_pair_swiglu &&
            direct_down_sum &&
            g_moe_mul_mv_slots6_iq2_xxs_pair_swiglu_pipeline != nil &&
            g_moe_mul_mv_slots6_q2_k_sum6_pipeline != nil &&
            getenv("DS4_METAL_DISABLE_IQ2_SELECTED_EXPERT_VIEWS") == NULL;
        const bool use_iq2_stream_addr_table =
            !force_resident &&
            g_ssd_streaming_mode &&
            gate_type == DS4_METAL_TENSOR_IQ2_XXS &&
            down_type == DS4_METAL_TENSOR_IQ2_XXS &&
            n_expert <= DS4_METAL_MAX_ROUTED_EXPERT_USED &&
            n_tokens == 1 &&
            fuse_pair_swiglu &&
            g_moe_mul_mv_addr_iq2_xxs_pair_swiglu_pipeline != nil &&
            g_moe_mul_mv_addr_iq2_xxs_pipeline != nil &&
            getenv("DS4_METAL_DISABLE_IQ2_STREAM_ADDR_TABLE") == NULL;
        const bool use_selected_slots =
            use_q4_selected_slots || use_iq2_selected_slots || use_iq2_stream_addr_table;
        id<MTLComputePipelineState> slots_pair_swiglu_pipeline =
            use_iq2_selected_slots ? g_moe_mul_mv_slots6_iq2_xxs_pair_swiglu_pipeline :
            g_moe_mul_mv_slots6_q4_k_pair_swiglu_pipeline;
        id<MTLComputePipelineState> slots_sum6_pipeline =
            use_iq2_selected_slots ? g_moe_mul_mv_slots6_q2_k_sum6_pipeline :
            g_moe_mul_mv_slots6_q4_k_sum6_pipeline;
        const char *selected_profile_env = getenv("DS4_METAL_SELECTED_PROFILE");
        if (!selected_profile_env) {
            selected_profile_env = getenv("DS4_METAL_Q4_SELECTED_PROFILE");
        }
        const char *selected_profile_layer_env = getenv("DS4_METAL_SELECTED_PROFILE_LAYER");
        if (!selected_profile_layer_env) {
            selected_profile_layer_env = getenv("DS4_METAL_Q4_SELECTED_PROFILE_LAYER");
        }
        bool selected_profile_layer_match = true;
        if (selected_profile_layer_env && selected_profile_layer_env[0]) {
            char *end = NULL;
            long layer = strtol(selected_profile_layer_env, &end, 10);
            selected_profile_layer_match =
                end && *end == '\0' && layer >= 0 && (uint32_t)layer == layer_index;
        }
        const bool selected_profile =
            use_selected_slots &&
            selected_profile_env != NULL &&
            selected_profile_layer_match;
        const bool q4_selected_shared_event =
            use_q4_selected_slots &&
            getenv("DS4_METAL_Q4_SELECTED_SHARED_EVENT") != NULL;
        const bool q4_selected_base_views =
            use_q4_selected_slots &&
            getenv("DS4_METAL_Q4_SELECTED_USE_BASE_VIEWS") != NULL &&
            getenv("DS4_METAL_Q4_SELECTED_EXACT_VIEWS") == NULL;
        const bool q4_selected_transient_views =
            use_q4_selected_slots &&
            !q4_selected_base_views &&
            getenv("DS4_METAL_Q4_SELECTED_TRANSIENT_VIEWS") != NULL;
        const char *q4_selected_view_mode =
            q4_selected_base_views ? "base" :
            (q4_selected_transient_views ? "transient" : "cached");
        if (!use_selected_slots) {
            g_routed_moe_selected_override_n = 0;
        }
        if (use_q4_expert_address_table) {
            gate_table = ds4_gpu_q4_expert_address_table(model_map,
                                                         model_size,
                                                         gate_offset,
                                                         gate_expert_bytes,
                                                         n_total_expert);
            up_table = ds4_gpu_q4_expert_address_table(model_map,
                                                       model_size,
                                                       up_offset,
                                                       gate_expert_bytes,
                                                       n_total_expert);
            down_table = ds4_gpu_q4_expert_address_table(model_map,
                                                         model_size,
                                                         down_offset,
                                                         down_expert_bytes,
                                                         n_total_expert);
            if (!gate_table || !up_table || !down_table) {
                if (getenv("DS4_GLM_TP_DEBUG")) fprintf(stderr, "ds4: routed_moe_one silent return at line %d\n", 32546);
                return 0;
            }
            q4_table_layer_residency =
                ds4_gpu_q4_expert_layer_residency_set(gate_table,
                                                       up_table,
                                                       down_table,
                                                       q4_expert_address_auto);
        } else if (use_q4_expert_table) {
            gate_table = ds4_gpu_q4_expert_table(model_map,
                                                 model_size,
                                                 gate_offset,
                                                 gate_expert_bytes,
                                                 n_total_expert,
                                                 g_moe_table_q4_pair_gate_encoder);
            up_table = ds4_gpu_q4_expert_table(model_map,
                                               model_size,
                                               up_offset,
                                               gate_expert_bytes,
                                               n_total_expert,
                                               g_moe_table_q4_pair_up_encoder);
            down_table = ds4_gpu_q4_expert_table(model_map,
                                                 model_size,
                                                 down_offset,
                                                 down_expert_bytes,
                                                 n_total_expert,
                                                 g_moe_table_q4_sum_down_encoder);
            if (!gate_table || !up_table || !down_table) {
                if (getenv("DS4_GLM_TP_DEBUG")) fprintf(stderr, "ds4: routed_moe_one silent return at line %d\n", 32573);
                return 0;
            }
            q4_table_layer_residency =
                ds4_gpu_q4_expert_layer_residency_set(gate_table,
                                                       up_table,
                                                       down_table,
                                                       q4_expert_table_auto);
        } else if (use_q4_exact_tensor_id) {
            gate_buf = ds4_gpu_wrap_model_exact_range(model_map,
                                                      model_size,
                                                      gate_offset,
                                                      gate_tensor_bytes,
                                                      &gate_inner);
            up_buf = ds4_gpu_wrap_model_exact_range(model_map,
                                                    model_size,
                                                    up_offset,
                                                    gate_tensor_bytes,
                                                    &up_inner);
            down_buf = ds4_gpu_wrap_model_exact_range(model_map,
                                                      model_size,
                                                      down_offset,
                                                      down_tensor_bytes,
                                                      &down_inner);
            if (!gate_buf || !up_buf || !down_buf) { if (getenv("DS4_GLM_TP_DEBUG")) fprintf(stderr, "ds4: routed_moe_one silent return at line %d\n", 32601); return 0; }
        } else if (use_q4_group6_experts || use_q4_gather_slots) {
            if ((uint64_t)q4_group6_expert_group_size > UINT64_MAX / gate_expert_bytes ||
                (uint64_t)q4_group6_expert_group_size > UINT64_MAX / down_expert_bytes) {
                fprintf(stderr, "ds4: Metal routed MoE Q4 group6 byte size overflow\n");
                return 0;
            }
            const uint64_t gate_group_bytes =
                (uint64_t)q4_group6_expert_group_size * gate_expert_bytes;
            const uint64_t down_group_bytes =
                (uint64_t)q4_group6_expert_group_size * down_expert_bytes;
            for (uint32_t i = 0; i < 6; i++) {
                const uint64_t gate_rel = (uint64_t)i * gate_group_bytes;
                const uint64_t down_rel = (uint64_t)i * down_group_bytes;
                if (gate_rel > gate_tensor_bytes ||
                    gate_group_bytes > gate_tensor_bytes - gate_rel ||
                    down_rel > down_tensor_bytes ||
                    down_group_bytes > down_tensor_bytes - down_rel ||
                    gate_rel > UINT64_MAX - gate_offset ||
                    gate_rel > UINT64_MAX - up_offset ||
                    down_rel > UINT64_MAX - down_offset) {
                    fprintf(stderr, "ds4: Metal routed MoE Q4 group6 offset overflow\n");
                    return 0;
                }

                uint64_t group_inner = 0;
                gate_group6_bufs[i] = ds4_gpu_wrap_model_range(model_map,
                                                                model_size,
                                                                gate_offset + gate_rel,
                                                                gate_group_bytes,
                                                                &group_inner);
                gate_group6_offsets[i] = (NSUInteger)group_inner;
                group_inner = 0;
                up_group6_bufs[i] = ds4_gpu_wrap_model_range(model_map,
                                                              model_size,
                                                              up_offset + gate_rel,
                                                              gate_group_bytes,
                                                              &group_inner);
                up_group6_offsets[i] = (NSUInteger)group_inner;
                group_inner = 0;
                down_group6_bufs[i] = ds4_gpu_wrap_model_range(model_map,
                                                                model_size,
                                                                down_offset + down_rel,
                                                                down_group_bytes,
                                                                &group_inner);
                down_group6_offsets[i] = (NSUInteger)group_inner;
                if (!gate_group6_bufs[i] || !up_group6_bufs[i] || !down_group6_bufs[i]) {
                    if (getenv("DS4_GLM_TP_DEBUG")) fprintf(stderr, "ds4: routed_moe_one silent return at line %d\n", 32643);
                    return 0;
                }
            }
            if (use_q4_gather_slots) {
                if (gate_expert_bytes > NSUIntegerMax ||
                    down_expert_bytes > NSUIntegerMax ||
                    gate_expert_bytes > UINT64_MAX / 6u ||
                    down_expert_bytes > UINT64_MAX / 6u ||
                    6ull * gate_expert_bytes > NSUIntegerMax ||
                    6ull * down_expert_bytes > NSUIntegerMax) {
                    fprintf(stderr, "ds4: Metal routed MoE Q4 gather scratch byte size overflow\n");
                    return 0;
                }
                const NSUInteger gate_slots_bytes = (NSUInteger)(6ull * gate_expert_bytes);
                const NSUInteger down_slots_bytes = (NSUInteger)(6ull * down_expert_bytes);
                if (!ds4_gpu_ensure_scratch_buffer(&g_moe_q4_gate_slots_buffer,
                                                    &g_moe_q4_gate_slots_bytes,
                                                    gate_slots_bytes,
                                                    "ds4_moe_q4_gate_slots") ||
                    !ds4_gpu_ensure_scratch_buffer(&g_moe_q4_up_slots_buffer,
                                                    &g_moe_q4_up_slots_bytes,
                                                    gate_slots_bytes,
                                                    "ds4_moe_q4_up_slots") ||
                    !ds4_gpu_ensure_scratch_buffer(&g_moe_q4_down_slots_buffer,
                                                    &g_moe_q4_down_slots_bytes,
                                                    down_slots_bytes,
                                                    "ds4_moe_q4_down_slots")) {
                    if (getenv("DS4_GLM_TP_DEBUG")) fprintf(stderr, "ds4: routed_moe_one silent return at line %d\n", 32670);
                    return 0;
                }
                for (uint32_t i = 0; i < 6; i++) {
                    gate_slot_bufs[i] = g_moe_q4_gate_slots_buffer;
                    up_slot_bufs[i] = g_moe_q4_up_slots_buffer;
                    down_slot_bufs[i] = g_moe_q4_down_slots_buffer;
                    gate_slot_offsets[i] = (NSUInteger)((uint64_t)i * gate_expert_bytes);
                    up_slot_offsets[i] = (NSUInteger)((uint64_t)i * gate_expert_bytes);
                    down_slot_offsets[i] = (NSUInteger)((uint64_t)i * down_expert_bytes);
                }
            }
        } else if (use_q4_group8_experts) {
            if ((uint64_t)q4_group8_expert_group_size > UINT64_MAX / gate_expert_bytes ||
                (uint64_t)q4_group8_expert_group_size > UINT64_MAX / down_expert_bytes) {
                fprintf(stderr, "ds4: Metal routed MoE Q4 group8 byte size overflow\n");
                return 0;
            }
            const uint64_t gate_group_bytes =
                (uint64_t)q4_group8_expert_group_size * gate_expert_bytes;
            const uint64_t down_group_bytes =
                (uint64_t)q4_group8_expert_group_size * down_expert_bytes;
            for (uint32_t i = 0; i < 8; i++) {
                const uint64_t gate_rel = (uint64_t)i * gate_group_bytes;
                const uint64_t down_rel = (uint64_t)i * down_group_bytes;
                if (gate_rel > gate_tensor_bytes ||
                    gate_group_bytes > gate_tensor_bytes - gate_rel ||
                    down_rel > down_tensor_bytes ||
                    down_group_bytes > down_tensor_bytes - down_rel ||
                    gate_rel > UINT64_MAX - gate_offset ||
                    gate_rel > UINT64_MAX - up_offset ||
                    down_rel > UINT64_MAX - down_offset) {
                    fprintf(stderr, "ds4: Metal routed MoE Q4 group8 offset overflow\n");
                    return 0;
                }

                uint64_t group_inner = 0;
                gate_group8_bufs[i] = ds4_gpu_wrap_model_range(model_map,
                                                                model_size,
                                                                gate_offset + gate_rel,
                                                                gate_group_bytes,
                                                                &group_inner);
                gate_group8_offsets[i] = (NSUInteger)group_inner;
                group_inner = 0;
                up_group8_bufs[i] = ds4_gpu_wrap_model_range(model_map,
                                                              model_size,
                                                              up_offset + gate_rel,
                                                              gate_group_bytes,
                                                              &group_inner);
                up_group8_offsets[i] = (NSUInteger)group_inner;
                group_inner = 0;
                down_group8_bufs[i] = ds4_gpu_wrap_model_range(model_map,
                                                                model_size,
                                                                down_offset + down_rel,
                                                                down_group_bytes,
                                                                &group_inner);
                down_group8_offsets[i] = (NSUInteger)group_inner;
                if (!gate_group8_bufs[i] || !up_group8_bufs[i] || !down_group8_bufs[i]) {
                    if (getenv("DS4_GLM_TP_DEBUG")) fprintf(stderr, "ds4: routed_moe_one silent return at line %d\n", 32727);
                    return 0;
                }
            }
        } else if (use_q4_group24_experts) {
            if ((uint64_t)q4_group24_expert_group_size > UINT64_MAX / gate_expert_bytes ||
                (uint64_t)q4_group24_expert_group_size > UINT64_MAX / down_expert_bytes) {
                fprintf(stderr, "ds4: Metal routed MoE Q4 group24 byte size overflow\n");
                return 0;
            }
            const uint64_t gate_group_bytes =
                (uint64_t)q4_group24_expert_group_size * gate_expert_bytes;
            const uint64_t down_group_bytes =
                (uint64_t)q4_group24_expert_group_size * down_expert_bytes;
            for (uint32_t i = 0; i < 24; i++) {
                const uint64_t gate_rel = (uint64_t)i * gate_group_bytes;
                const uint64_t down_rel = (uint64_t)i * down_group_bytes;
                if (gate_rel > gate_tensor_bytes ||
                    gate_group_bytes > gate_tensor_bytes - gate_rel ||
                    down_rel > down_tensor_bytes ||
                    down_group_bytes > down_tensor_bytes - down_rel ||
                    gate_rel > UINT64_MAX - gate_offset ||
                    gate_rel > UINT64_MAX - up_offset ||
                    down_rel > UINT64_MAX - down_offset) {
                    fprintf(stderr, "ds4: Metal routed MoE Q4 group24 offset overflow\n");
                    return 0;
                }

                uint64_t group_inner = 0;
                gate_group24_bufs[i] = q4_group24_exact_views ?
                    ds4_gpu_wrap_model_exact_range(model_map,
                                                   model_size,
                                                   gate_offset + gate_rel,
                                                   gate_group_bytes,
                                                   &group_inner) :
                    ds4_gpu_wrap_model_range(model_map,
                                             model_size,
                                             gate_offset + gate_rel,
                                             gate_group_bytes,
                                             &group_inner);
                gate_group24_offsets[i] = (NSUInteger)group_inner;
                group_inner = 0;
                up_group24_bufs[i] = q4_group24_exact_views ?
                    ds4_gpu_wrap_model_exact_range(model_map,
                                                   model_size,
                                                   up_offset + gate_rel,
                                                   gate_group_bytes,
                                                   &group_inner) :
                    ds4_gpu_wrap_model_range(model_map,
                                             model_size,
                                             up_offset + gate_rel,
                                             gate_group_bytes,
                                             &group_inner);
                up_group24_offsets[i] = (NSUInteger)group_inner;
                group_inner = 0;
                down_group24_bufs[i] = q4_group24_exact_views ?
                    ds4_gpu_wrap_model_exact_range(model_map,
                                                   model_size,
                                                   down_offset + down_rel,
                                                   down_group_bytes,
                                                   &group_inner) :
                    ds4_gpu_wrap_model_range(model_map,
                                             model_size,
                                             down_offset + down_rel,
                                             down_group_bytes,
                                             &group_inner);
                down_group24_offsets[i] = (NSUInteger)group_inner;
                if (!gate_group24_bufs[i] || !up_group24_bufs[i] || !down_group24_bufs[i]) {
                    if (getenv("DS4_GLM_TP_DEBUG")) fprintf(stderr, "ds4: routed_moe_one silent return at line %d\n", 32794);
                    return 0;
                }
            }
        } else if (use_selected_slots) {
            const bool selected_timing =
                selected_profile ||
                ds4_gpu_stream_expert_timing_summary_enabled();
            double selected_t0 = selected_timing ? ds4_gpu_now_ms() : 0.0;
            double selected_read_ms = 0.0;
            double selected_sync_ms = 0.0;
            double selected_copy_ms = 0.0;
            double selected_wrap_ms = 0.0;
            uint64_t selected_cache_hits0 = g_stream_expert_cache_hits;
            uint64_t selected_cache_misses0 = g_stream_expert_cache_misses;
            uint64_t selected_cache_wraps0 = g_stream_expert_cache_wraps;
            uint64_t selected_cache_evictions0 = g_stream_expert_cache_evictions;
            const char *selected_id_source = "readback";
            bool selected_ids_available = true;
            bool selected_exec_ids_from_host = false;
            const int stream_expert_cache_size_known =
                ds4_gpu_stream_expert_cache_note_expert_size(gate_expert_bytes,
                                                             down_expert_bytes);
            const bool use_iq2_full_expert_addr_table =
                use_iq2_selected_slots &&
                ds4_gpu_stream_full_expert_addr_table_requested() &&
                g_moe_mul_mv_addr_iq2_xxs_pair_swiglu_pipeline != nil &&
                g_moe_mul_mv_addr_q2_k_sum6_pipeline != nil;
            use_stream_expert_cache =
                !use_iq2_full_expert_addr_table &&
                (use_iq2_selected_slots || use_iq2_stream_addr_table || use_q4_selected_slots) &&
                stream_expert_cache_size_known &&
                ds4_gpu_stream_expert_cache_effective_cap(layer_index,
                                                          n_total_expert,
                                                          n_expert) != 0;
            if (use_iq2_stream_addr_table && !use_stream_expert_cache) {
                fprintf(stderr,
                        "ds4: Metal IQ2/IQ2 streaming decode requires a non-empty expert cache\n");
                if (getenv("DS4_GLM_TP_DEBUG")) fprintf(stderr, "ds4: routed_moe_one silent return at line %d\n", 32831);
                return 0;
            }
            const bool stream_split_ready =
                use_stream_expert_cache &&
                ds4_gpu_stream_expert_split_ready();
            const bool use_stream_compact_addr =
                use_stream_expert_cache &&
                use_iq2_selected_slots &&
                ds4_gpu_stream_compact_addr_requested() &&
                !stream_split_ready &&
                !ds4_gpu_stream_expert_masked_addr_requested() &&
                g_moe_mul_mv_addr_iq2_xxs_pair_swiglu_pipeline != nil &&
                g_moe_mul_mv_addr_q2_k_sum6_pipeline != nil;
            use_stream_expert_split_candidate =
                use_stream_expert_cache &&
                use_iq2_selected_slots &&
                !use_stream_compact_addr &&
                stream_split_ready &&
                g_moe_mul_mv_addr_iq2_xxs_pair_swiglu_masked_pipeline != nil &&
                g_moe_mul_mv_addr_q2_k_sum6_masked_pipeline != nil;
            const bool use_stream_hit_validator =
                use_stream_expert_cache &&
                use_iq2_selected_slots &&
                ds4_gpu_stream_expert_hit_validator_requested() &&
                g_moe_stream_expert_cache_validate_pipeline != nil &&
                g_moe_mul_mv_addr_iq2_xxs_pair_swiglu_pipeline != nil &&
                g_moe_mul_mv_addr_q2_k_sum6_pipeline != nil &&
                ds4_gpu_stream_expert_cache_addr_buffers(layer_index,
                                                         &stream_gate_addr_buf,
                                                         &stream_up_addr_buf,
                                                         &stream_down_addr_buf);
            if (use_iq2_full_expert_addr_table) {
                selected_id_source = "gpu-full-addr";
                selected_ids_available = false;
                g_routed_moe_selected_override_n = 0;
                ds4_gpu_stream_expert_cache_entry *full_entry = NULL;
                if (!ds4_gpu_stream_full_expert_addr_table_prepare(model_map,
                                                                   model_size,
                                                                   layer_index,
                                                                   n_total_expert,
                                                                   gate_offset,
                                                                   up_offset,
                                                                   down_offset,
                                                                   gate_expert_bytes,
                                                                   down_expert_bytes,
                                                                   &stream_gate_addr_buf,
                                                                   &stream_up_addr_buf,
                                                                   &stream_down_addr_buf,
                                                                   &full_entry)) {
                    if (getenv("DS4_GLM_TP_DEBUG")) fprintf(stderr, "ds4: routed_moe_one silent return at line %d\n", 32880);
                    return 0;
                }
                for (uint32_t i = 0; i < n_expert; i++) {
                    stream_slot_entries[i] = full_entry;
                }
                use_stream_expert_addr_table = true;
            } else {
                const int replayed_selected_ids =
                    ds4_gpu_moe_selected_trace_replay(selected_ids, n_expert);
                if (replayed_selected_ids < 0) {
                    if (getenv("DS4_GLM_TP_DEBUG")) fprintf(stderr, "ds4: routed_moe_one silent return at line %d\n", 32890);
                    return 0;
                } else if (replayed_selected_ids > 0) {
                    selected_id_source = "replay";
                    selected_exec_ids_from_host = true;
                    g_routed_moe_selected_override_n = 0;
                } else if (g_routed_moe_selected_override_n == n_expert) {
                    memcpy(selected_ids,
                           g_routed_moe_selected_override,
                           (size_t)n_expert * sizeof(selected_ids[0]));
                    selected_id_source = "override";
                    selected_exec_ids_from_host = true;
                    g_routed_moe_selected_override_n = 0;
                } else if (use_stream_hit_validator) {
                    g_routed_moe_selected_override_n = 0;
                    uint32_t validator_all_cached = 0;
                    uint32_t validator_miss_mask = 0;
                    uint32_t validator_invalid_mask = 0;
                    if (!ds4_gpu_stream_expert_cache_validate_selected(selected,
                                                                       stream_gate_addr_buf,
                                                                       stream_up_addr_buf,
                                                                       stream_down_addr_buf,
                                                                       n_total_expert,
                                                                       n_expert,
                                                                       selected_ids,
                                                                       &validator_all_cached,
                                                                       &validator_miss_mask,
                                                                       &validator_invalid_mask)) {
                        if (getenv("DS4_GLM_TP_DEBUG")) fprintf(stderr, "ds4: routed_moe_one silent return at line %d\n", 32917);
                        return 0;
                    }
                    selected_id_source =
                        validator_invalid_mask != 0 ? "validator-invalid" :
                        (validator_miss_mask != 0 ? "validator-miss" :
                        (validator_all_cached != 0 ? "validator-hit" : "validator-miss"));
                } else {
                    g_routed_moe_selected_override_n = 0;
                    if (g_batch_cb != nil) {
                        double selected_boundary_t0 =
                            selected_timing ? ds4_gpu_now_ms() : 0.0;
                        if (q4_selected_shared_event) {
                            if (ds4_gpu_signal_batch_and_wait_event("selected-id readback") == 0) { if (getenv("DS4_GLM_TP_DEBUG")) fprintf(stderr, "ds4: routed_moe_one silent return at line %d\n", 32942); return 0; }
                        } else if (ds4_gpu_end_commands() == 0) {
                            if (getenv("DS4_GLM_TP_DEBUG")) fprintf(stderr, "ds4: routed_moe_one silent return at line %d\n", 32931);
                            return 0;
                        }
                        if (selected_timing) {
                            selected_sync_ms +=
                                ds4_gpu_now_ms() - selected_boundary_t0;
                        }
                        double selected_copy_t0 =
                            selected_timing ? ds4_gpu_now_ms() : 0.0;
                        if (ds4_gpu_tensor_read(selected,
                                                0,
                                                selected_ids,
                                                (uint64_t)n_expert * sizeof(selected_ids[0])) == 0) {
                            if (getenv("DS4_GLM_TP_DEBUG")) fprintf(stderr, "ds4: routed_moe_one silent return at line %d\n", 32943);
                            return 0;
                        }
                        if (selected_timing) {
                            selected_copy_ms +=
                                ds4_gpu_now_ms() - selected_copy_t0;
                        }
                        if (!q4_selected_shared_event) {
                            selected_boundary_t0 =
                                selected_timing ? ds4_gpu_now_ms() : 0.0;
                            if (ds4_gpu_begin_commands() == 0) { if (getenv("DS4_GLM_TP_DEBUG")) fprintf(stderr, "ds4: routed_moe_one silent return at line %d\n", 32967); return 0; }
                            if (selected_timing) {
                                selected_sync_ms +=
                                    ds4_gpu_now_ms() - selected_boundary_t0;
                            }
                        }
                    } else {
                        double selected_copy_t0 =
                            selected_timing ? ds4_gpu_now_ms() : 0.0;
                        if (ds4_gpu_tensor_read(selected,
                                                0,
                                                selected_ids,
                                                (uint64_t)n_expert * sizeof(selected_ids[0])) == 0) {
                            if (getenv("DS4_GLM_TP_DEBUG")) fprintf(stderr, "ds4: routed_moe_one silent return at line %d\n", 32965);
                            return 0;
                        }
                        if (selected_timing) {
                            selected_copy_ms +=
                                ds4_gpu_now_ms() - selected_copy_t0;
                        }
                    }
                }
            }
            if (selected_timing) {
                selected_read_ms = selected_sync_ms + selected_copy_ms;
                selected_t0 = ds4_gpu_now_ms();
            }

            if (selected_ids_available) {
                for (uint32_t i = 0; i < n_expert; i++) {
                    if (selected_ids[i] < 0 || (uint32_t)selected_ids[i] >= n_total_expert) {
                        fprintf(stderr,
                                "ds4: Metal routed MoE selected expert id %d is outside 0..%u\n",
                                selected_ids[i],
                                n_total_expert);
                        if (getenv("DS4_GLM_TP_DEBUG")) fprintf(stderr, "ds4: routed_moe_one silent return at line %d\n", 32986);
                        return 0;
                    }
                }
                ds4_gpu_stream_expert_cache_note_selected_hotness(layer_index,
                                                                  selected_ids,
                                                                  n_expert);
                if (!ds4_gpu_moe_selected_trace_record(selected_ids, n_expert) ||
                    !ds4_gpu_moe_selected_hotlist_record(layer_index,
                                                         selected_ids,
                                                         n_expert,
                                                         n_total_expert)) {
                    if (getenv("DS4_GLM_TP_DEBUG")) fprintf(stderr, "ds4: routed_moe_one silent return at line %d\n", 32997);
                    return 0;
                }

                for (uint32_t i = 0; i < n_expert; i++) {
                    const uint64_t expert_id = (uint64_t)(uint32_t)selected_ids[i];
                    if (expert_id > UINT64_MAX / gate_expert_bytes ||
                        expert_id > UINT64_MAX / down_expert_bytes) {
                        fprintf(stderr, "ds4: Metal routed MoE selected expert offset overflow\n");
                        return 0;
                    }
                    const uint64_t gate_rel = expert_id * gate_expert_bytes;
                    const uint64_t down_rel = expert_id * down_expert_bytes;
                    if (gate_rel > UINT64_MAX - gate_offset ||
                        gate_rel > UINT64_MAX - up_offset ||
                        down_rel > UINT64_MAX - down_offset) {
                        fprintf(stderr, "ds4: Metal routed MoE selected expert offset overflow\n");
                        return 0;
                    }
                    stream_gate_abs_offsets[i] = gate_offset + gate_rel;
                    stream_up_abs_offsets[i] = up_offset + gate_rel;
                    stream_down_abs_offsets[i] = down_offset + down_rel;

                    if (use_stream_expert_cache) {
                        ds4_gpu_stream_expert_cache_entry *entry = NULL;
                        entry = ds4_gpu_stream_expert_cache_peek(model_map,
                                                                 model_size,
                                                                 layer_index,
                                                                 (uint32_t)selected_ids[i],
                                                                 n_total_expert,
                                                                 n_expert,
                                                                 stream_gate_abs_offsets[i],
                                                                 stream_up_abs_offsets[i],
                                                                 stream_down_abs_offsets[i],
                                                                 gate_expert_bytes,
                                                                 down_expert_bytes);
                        if (!entry) {
                            stream_expert_missing_mask |= 1u << i;
                            continue;
                        }
                        stream_expert_resident_mask |= 1u << i;
                        stream_slot_entries[i] = entry;
                        gate_slot_bufs[i] = entry->gate_buffer;
                        gate_slot_offsets[i] = entry->gate_inner;
                        up_slot_bufs[i] = entry->up_buffer;
                        up_slot_offsets[i] = entry->up_inner;
                        down_slot_bufs[i] = entry->down_buffer;
                        down_slot_offsets[i] = entry->down_inner;
                        continue;
                    }

                    uint64_t slot_inner = 0;
                    gate_slot_bufs[i] = q4_selected_base_views ?
                        ds4_gpu_wrap_model_range(model_map,
                                                 model_size,
                                                 gate_offset + gate_rel,
                                                 gate_expert_bytes,
                                                 &slot_inner) :
                        (q4_selected_transient_views ?
                            ds4_gpu_wrap_model_exact_range_transient(model_map,
                                                                     model_size,
                                                                     gate_offset + gate_rel,
                                                                     gate_expert_bytes,
                                                                     &slot_inner) :
                            ds4_gpu_wrap_model_exact_range(model_map,
                                                           model_size,
                                                           gate_offset + gate_rel,
                                                           gate_expert_bytes,
                                                           &slot_inner));
                    gate_slot_offsets[i] = (NSUInteger)slot_inner;
                    slot_inner = 0;
                    up_slot_bufs[i] = q4_selected_base_views ?
                        ds4_gpu_wrap_model_range(model_map,
                                                 model_size,
                                                 up_offset + gate_rel,
                                                 gate_expert_bytes,
                                                 &slot_inner) :
                        (q4_selected_transient_views ?
                            ds4_gpu_wrap_model_exact_range_transient(model_map,
                                                                     model_size,
                                                                     up_offset + gate_rel,
                                                                     gate_expert_bytes,
                                                                     &slot_inner) :
                            ds4_gpu_wrap_model_exact_range(model_map,
                                                           model_size,
                                                           up_offset + gate_rel,
                                                           gate_expert_bytes,
                                                           &slot_inner));
                    up_slot_offsets[i] = (NSUInteger)slot_inner;
                    slot_inner = 0;
                    down_slot_bufs[i] = q4_selected_base_views ?
                        ds4_gpu_wrap_model_range(model_map,
                                                 model_size,
                                                 down_offset + down_rel,
                                                 down_expert_bytes,
                                                 &slot_inner) :
                        (q4_selected_transient_views ?
                            ds4_gpu_wrap_model_exact_range_transient(model_map,
                                                                     model_size,
                                                                     down_offset + down_rel,
                                                                     down_expert_bytes,
                                                                     &slot_inner) :
                            ds4_gpu_wrap_model_exact_range(model_map,
                                                           model_size,
                                                           down_offset + down_rel,
                                                           down_expert_bytes,
                                                           &slot_inner));
                    down_slot_offsets[i] = (NSUInteger)slot_inner;
                    if (!gate_slot_bufs[i] || !up_slot_bufs[i] || !down_slot_bufs[i]) {
                        if (getenv("DS4_GLM_TP_DEBUG")) fprintf(stderr, "ds4: routed_moe_one silent return at line %d\n", 33105);
                        return 0;
                    }
                }
            }
            if (use_stream_expert_cache) {
                use_stream_expert_addr_table =
                    ((use_iq2_selected_slots &&
                      ds4_gpu_stream_expert_addr_table_kernel_requested() &&
                      g_moe_mul_mv_addr_iq2_xxs_pair_swiglu_pipeline != nil &&
                      g_moe_mul_mv_addr_q2_k_sum6_pipeline != nil) ||
                     use_iq2_stream_addr_table) &&
                    ds4_gpu_stream_expert_cache_addr_buffers(layer_index,
                                                             &stream_gate_addr_buf,
                                                             &stream_up_addr_buf,
                                                             &stream_down_addr_buf);
                if (use_iq2_stream_addr_table && !use_stream_expert_addr_table) {
                    fprintf(stderr,
                            "ds4: Metal IQ2/IQ2 streaming decode could not prepare expert address buffers\n");
                    if (getenv("DS4_GLM_TP_DEBUG")) fprintf(stderr, "ds4: routed_moe_one silent return at line %d\n", 33123);
                    return 0;
                }
                use_stream_expert_masked_addr_table =
                    use_stream_expert_addr_table &&
                    use_iq2_selected_slots &&
                    ds4_gpu_stream_expert_masked_addr_requested() &&
                    g_moe_mul_mv_addr_iq2_xxs_pair_swiglu_masked_pipeline != nil &&
                    g_moe_mul_mv_addr_q2_k_sum6_masked_pipeline != nil;
                use_stream_expert_split_deferred =
                    use_stream_expert_split_candidate &&
                    use_stream_expert_masked_addr_table &&
                    stream_expert_resident_mask != 0 &&
                    stream_expert_missing_mask != 0 &&
                    ds4_gpu_stream_expert_split_worthwhile(stream_expert_resident_mask,
                                                           stream_expert_missing_mask) &&
                    g_batch_cb != nil &&
                    getenv("DS4_METAL_MOE_ONE_STAGE_PROFILE") == NULL;
                if (use_stream_expert_split_deferred) {
                    const ds4_gpu_stream_expert_table table = {
                        .model_map = model_map,
                        .model_size = model_size,
                        .layer = layer_index,
                        .n_total_expert = n_total_expert,
                        .gate_offset = gate_offset,
                        .up_offset = up_offset,
                        .down_offset = down_offset,
                        .gate_expert_bytes = gate_expert_bytes,
                        .down_expert_bytes = down_expert_bytes,
                    };
                    if (!ds4_gpu_stream_expert_cache_begin_selected_load(
                                &table,
                                selected_ids,
                                n_expert)) {
                        if (getenv("DS4_GLM_TP_DEBUG")) fprintf(stderr, "ds4: routed_moe_one silent return at line %d\n", 33156);
                        return 0;
                    }
                }
                if (stream_expert_missing_mask != 0 &&
                    !use_stream_expert_split_deferred) {
                    if (!ds4_gpu_stream_expert_cache_load_selected_missing(
                            model_map,
                            model_size,
                            layer_index,
                            selected_ids,
                            n_total_expert,
                            n_expert,
                            stream_gate_abs_offsets,
                            stream_up_abs_offsets,
                            stream_down_abs_offsets,
                            gate_expert_bytes,
                            down_expert_bytes,
                            stream_expert_missing_mask,
                            stream_slot_entries)) {
                        if (getenv("DS4_GLM_TP_DEBUG")) fprintf(stderr, "ds4: routed_moe_one silent return at line %d\n", 33175);
                        return 0;
                    }
                    for (uint32_t i = 0; i < n_expert; i++) {
                        if ((stream_expert_missing_mask & (1u << i)) == 0) continue;
                        ds4_gpu_stream_expert_cache_entry *entry = stream_slot_entries[i];
                        if (!entry) { if (getenv("DS4_GLM_TP_DEBUG")) fprintf(stderr, "ds4: routed_moe_one silent return at line %d\n", 33202); return 0; }
                        gate_slot_bufs[i] = entry->gate_buffer;
                        gate_slot_offsets[i] = entry->gate_inner;
                        up_slot_bufs[i] = entry->up_buffer;
                        up_slot_offsets[i] = entry->up_inner;
                        down_slot_bufs[i] = entry->down_buffer;
                        down_slot_offsets[i] = entry->down_inner;
                    }
                }
                if (use_iq2_stream_addr_table && use_stream_expert_addr_table) {
                    for (uint32_t i = 0; i < n_expert; i++) {
                        ds4_gpu_stream_expert_cache_entry *entry = stream_slot_entries[i];
                        if (!entry) { if (getenv("DS4_GLM_TP_DEBUG")) fprintf(stderr, "ds4: routed_moe_one silent return at line %d\n", 33214); return 0; }
                        if (!ds4_gpu_stream_expert_cache_set_addr_slot_raw(
                                    layer_index,
                                    (uint32_t)selected_ids[i],
                                    entry->gate_buffer,
                                    entry->gate_inner,
                                    entry->up_buffer,
                                    entry->up_inner,
                                    entry->down_buffer,
                                    entry->down_inner)) {
                            if (getenv("DS4_GLM_TP_DEBUG")) fprintf(stderr, "ds4: routed_moe_one silent return at line %d\n", 33202);
                            return 0;
                        }
                    }
                }
                ds4_gpu_stream_expert_cache_prune_layer(layer_index,
                                                        n_total_expert,
                                                        n_expert,
                                                        selected_ids,
                                                        n_expert);
                ds4_gpu_stream_expert_cache_prune_global(layer_index,
                                                         selected_ids,
                                                         n_expert);
                if (use_stream_compact_addr) {
                    id<MTLBuffer> compact_selected = nil;
                    if (!ds4_gpu_stream_compact_addr_prepare(layer_index,
                                                             stream_slot_entries,
                                                             n_expert,
                                                             &stream_gate_addr_buf,
                                                             &stream_up_addr_buf,
                                                             &stream_down_addr_buf,
                                                             &compact_selected)) {
                        if (getenv("DS4_GLM_TP_DEBUG")) fprintf(stderr, "ds4: routed_moe_one silent return at line %d\n", 33223);
                        return 0;
                    }
                    selected_exec_buf = compact_selected;
                    selected_exec_off = 0;
                    use_stream_expert_addr_table = true;
                    use_stream_expert_masked_addr_table = false;
                    use_stream_compact_addr_table = true;
                }
                if (use_stream_expert_addr_table &&
                    !use_stream_compact_addr_table &&
                    selected_exec_ids_from_host) {
                    if (!ds4_gpu_stream_selected_ids_prepare(layer_index,
                                                             selected_ids,
                                                             n_expert,
                                                             &selected_exec_buf,
                                                             &selected_exec_off)) {
                        if (getenv("DS4_GLM_TP_DEBUG")) fprintf(stderr, "ds4: routed_moe_one silent return at line %d\n", 33239);
                        return 0;
                    }
                }
            }
            if (selected_timing) {
                selected_wrap_ms = ds4_gpu_now_ms() - selected_t0;
                if (use_stream_expert_cache) {
                    ds4_gpu_stream_expert_timing_note_cache_class(
                            stream_expert_resident_mask,
                            stream_expert_missing_mask);
                }
                ds4_gpu_stream_expert_timing_note_selected(selected_sync_ms,
                                                           selected_copy_ms,
                                                           selected_wrap_ms);
            }
            if (selected_profile) {
                const uint64_t selected_cache_hits =
                    g_stream_expert_cache_hits - selected_cache_hits0;
                const uint64_t selected_cache_misses =
                    g_stream_expert_cache_misses - selected_cache_misses0;
                const uint64_t selected_cache_wraps =
                    g_stream_expert_cache_wraps - selected_cache_wraps0;
                const uint64_t selected_cache_evictions =
                    g_stream_expert_cache_evictions - selected_cache_evictions0;
                const char *selected_path =
                    use_iq2_stream_addr_table ? "iq2/iq2" :
                    (use_iq2_selected_slots ? "iq2/q2" : "q4/q4");
                const char *selected_view_mode =
                    use_stream_expert_split_deferred ? "stream-split" :
                    use_stream_expert_masked_addr_table ? "stream-addr-mask" :
                    use_stream_compact_addr_table ? "stream-compact-addr" :
                    use_stream_expert_addr_table ? "stream-addr" :
                    (use_stream_expert_cache ? "stream-cache" :
                    (use_iq2_selected_slots ? "exact-cache" : q4_selected_view_mode));
                fprintf(stderr,
                        "ds4: Metal selected views layer=%u path=%s mode=%s ids=%s "
                        "experts=%d,%d,%d,%d,%d,%d expert_gate=%.2f MiB "
                        "expert_down=%.2f MiB read=%.3f ms bind=%.3f ms "
                        "cache_hits=%llu cache_misses=%llu cache_wraps=%llu cache_evictions=%llu\n",
                        layer_index,
                        selected_path,
                        selected_view_mode,
                        selected_id_source,
                        selected_ids_available ? selected_ids[0] : -1,
                        selected_ids_available ? selected_ids[1] : -1,
                        selected_ids_available ? selected_ids[2] : -1,
                        selected_ids_available ? selected_ids[3] : -1,
                        selected_ids_available ? selected_ids[4] : -1,
                        selected_ids_available ? selected_ids[5] : -1,
                        ds4_gpu_mib(gate_expert_bytes),
                        ds4_gpu_mib(down_expert_bytes),
                        selected_read_ms,
                        selected_wrap_ms,
                        (unsigned long long)selected_cache_hits,
                        (unsigned long long)selected_cache_misses,
                        (unsigned long long)selected_cache_wraps,
                        (unsigned long long)selected_cache_evictions);
            }
        } else if (!use_q4_grouped_experts) {
            gate_buf = ds4_gpu_wrap_model_range(model_map, model_size, gate_offset, gate_tensor_bytes, &gate_inner);
            up_buf = ds4_gpu_wrap_model_range(model_map, model_size, up_offset, gate_tensor_bytes, &up_inner);
            down_buf = ds4_gpu_wrap_model_range(model_map, model_size, down_offset, down_tensor_bytes, &down_inner);
            if (!gate_buf || !up_buf || !down_buf) { if (getenv("DS4_GLM_TP_DEBUG")) fprintf(stderr, "ds4: routed_moe_one silent return at line %d\n", 33326); return 0; }
        }
        if (q4_grouped_boundary || q4_exact_boundary || q4_table_boundary) {
            if (ds4_gpu_end_commands() == 0 || ds4_gpu_begin_commands() == 0) {
                if (getenv("DS4_GLM_TP_DEBUG")) fprintf(stderr, "ds4: routed_moe_one silent return at line %d\n", 33305);
                return 0;
            }
        }

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) { if (getenv("DS4_GLM_TP_DEBUG")) fprintf(stderr, "ds4: routed_moe_one silent return at line %d\n", 33337); return 0; }
        if ((use_q4_expert_address_table || use_q4_expert_table) &&
            !ds4_gpu_use_model_residency_set(cb)) {
            if (getenv("DS4_GLM_TP_DEBUG")) fprintf(stderr, "ds4: routed_moe_one silent return at line %d\n", 33314);
            return 0;
        }
        if (!q4_table_queue_residency &&
            q4_table_layer_residency &&
            [cb respondsToSelector:@selector(useResidencySet:)]) {
            [cb useResidencySet:q4_table_layer_residency];
        }

        const bool moe_one_stage_profile =
            g_batch_cb != nil &&
            ds4_gpu_stage_profile_enabled_for_layer("DS4_METAL_MOE_ONE_STAGE_PROFILE",
                                                    "DS4_METAL_MOE_ONE_STAGE_PROFILE_LAYER",
                                                    layer_index);
        const char *moe_one_stage_filter = getenv("DS4_METAL_MOE_STAGE_PROFILE_FILTER");
        const char *moe_one_path =
            use_q4_grouped_experts ? "q4_grouped_pair_swiglu" :
            use_q4_group6_experts ? "q4_group6_pair_swiglu" :
            use_q4_group8_experts ? "q4_group8_pair_swiglu" :
            use_q4_group24_experts ? "q4_group24_split_gate_up" :
            use_q4_exact_tensor_id ? "q4_exact_pair_swiglu" :
            use_q4_expert_address_table ? "q4_addr_pair_swiglu" :
            use_q4_expert_table ? "q4_table_pair_swiglu" :
            use_q4_gather_slots ? "q4_gather_slots6_pair_swiglu" :
            use_stream_expert_split_deferred ? "iq2_stream_split_pair_swiglu" :
            use_stream_expert_masked_addr_table ? "iq2_stream_addr_mask_pair_swiglu" :
            use_stream_expert_addr_table ? "iq2_stream_addr_pair_swiglu" :
            use_iq2_selected_slots ? "iq2_slots6_pair_swiglu" :
            use_q4_selected_slots ? "q4_slots6_pair_swiglu" :
            (fuse_pair_swiglu ? "pair_swiglu" :
            ((!g_quality_mode &&
              ((gate_type == DS4_METAL_TENSOR_IQ2_XXS && g_moe_mul_mv_id_iq2_xxs_pair_pipeline) ||
               (gate_type == DS4_METAL_TENSOR_Q4_K && g_moe_mul_mv_id_q4_k_pair_pipeline))) ? "pair" : "single"));
        double moe_one_stage_t0 = moe_one_stage_profile ? ds4_gpu_now_ms() : 0.0;
        if (moe_one_stage_profile) {
            if (ds4_gpu_end_commands() == 0 || ds4_gpu_begin_commands() == 0) {
                if (getenv("DS4_GLM_TP_DEBUG")) fprintf(stderr, "ds4: routed_moe_one silent return at line %d\n", 33349);
                return 0;
            }
            cb = ds4_gpu_command_buffer(&owned);
            if (!cb) { if (getenv("DS4_GLM_TP_DEBUG")) fprintf(stderr, "ds4: routed_moe_one silent return at line %d\n", 33380); return 0; }
            moe_one_stage_t0 = ds4_gpu_now_ms();
        }
#define DS4_METAL_PROFILE_MOE_ONE_STAGE(name) do { \
            if (ok && moe_one_stage_profile) { \
                if (ds4_gpu_end_commands() == 0) { \
                    ok = 0; \
                } else { \
                    const char *stage_name = (name); \
                    const double now_ms = ds4_gpu_now_ms(); \
                    const int print_stage = \
                        !moe_one_stage_filter || !moe_one_stage_filter[0] || \
                        strstr(stage_name, moe_one_stage_filter) != NULL; \
                    if (print_stage) { \
                        fprintf(stderr, \
                                "ds4: Metal routed MoE one stage layer=%u pairs=%u experts=%u " \
                                "gate=%s down=%s path=%s %s=%.3f ms\n", \
                                layer_index, pair_rows, n_expert, \
                                ds4_gpu_metal_tensor_type_name(gate_type), \
                                ds4_gpu_metal_tensor_type_name(down_type), \
                                moe_one_path, \
                                stage_name, now_ms - moe_one_stage_t0); \
                    } \
                    moe_one_stage_t0 = now_ms; \
                    if (ds4_gpu_begin_commands() == 0) { \
                        ok = 0; \
                    } else { \
                        cb = ds4_gpu_command_buffer(&owned); \
                        if (!cb) ok = 0; \
                    } \
                } \
            } \
        } while (0)
        if (use_q4_gather_slots) {
            ds4_gpu_q4_gather_slots6_args gate_gather_args = {
                .expert_bytes = gate_expert_bytes,
                .group_size = q4_group6_expert_group_size,
                .n_slots = n_expert,
            };
            ds4_gpu_q4_gather_slots6_args down_gather_args = {
                .expert_bytes = down_expert_bytes,
                .group_size = q4_group6_expert_group_size,
                .n_slots = n_expert,
            };
            ok = ds4_gpu_encode_q4_gather_slots6(cb,
                                                  g_moe_q4_gather_slots6_pipeline,
                                                  &gate_gather_args,
                                                  gate_group6_bufs,
                                                  gate_group6_offsets,
                                                  selectedbuf,
                                                  ds4_gpu_tensor_offset(selected),
                                                  g_moe_q4_gate_slots_buffer,
                                                  0) &&
                 ds4_gpu_encode_q4_gather_slots6(cb,
                                                  g_moe_q4_gather_slots6_pipeline,
                                                  &gate_gather_args,
                                                  up_group6_bufs,
                                                  up_group6_offsets,
                                                  selectedbuf,
                                                  ds4_gpu_tensor_offset(selected),
                                                  g_moe_q4_up_slots_buffer,
                                                  0) &&
                 ds4_gpu_encode_q4_gather_slots6(cb,
                                                  g_moe_q4_gather_slots6_pipeline,
                                                  &down_gather_args,
                                                  down_group6_bufs,
                                                  down_group6_offsets,
                                                  selectedbuf,
                                                  ds4_gpu_tensor_offset(selected),
                                                  g_moe_q4_down_slots_buffer,
                                                  0);
        }
        if (use_q4_gather_slots) {
            DS4_METAL_PROFILE_MOE_ONE_STAGE("q4_gather");
        }
        if (!ok) { if (getenv("DS4_GLM_TP_DEBUG")) fprintf(stderr, "ds4: routed_moe_one silent return at line %d\n", 33455); return 0; }
        if (use_q4_grouped_experts) {
            ds4_gpu_dsv4_moe_swiglu_weight_args act_args = {
                .width = expert_mid_dim,
                .rows = pair_rows,
                .gate_row_stride = (uint64_t)expert_mid_dim * sizeof(float),
                .up_row_stride = (uint64_t)expert_mid_dim * sizeof(float),
                .mid_row_stride = (uint64_t)expert_mid_dim * sizeof(float),
                .weight_stride = sizeof(float),
                .write_clamped = 0,
                .clamp_value = clamp,
            };
            for (uint32_t expert_base = 0; ok && expert_base < n_total_expert; expert_base += q4_expert_group_size) {
                const uint32_t expert_count =
                    q4_expert_group_size < n_total_expert - expert_base ?
                    q4_expert_group_size : n_total_expert - expert_base;
                if ((uint64_t)expert_base > UINT64_MAX / gate_expert_bytes ||
                    (uint64_t)expert_count > UINT64_MAX / gate_expert_bytes) {
                    ok = 0;
                    break;
                }
                const uint64_t group_rel = (uint64_t)expert_base * gate_expert_bytes;
                const uint64_t group_bytes = (uint64_t)expert_count * gate_expert_bytes;
                if (group_rel > UINT64_MAX - gate_offset ||
                    group_rel > UINT64_MAX - up_offset) {
                    ok = 0;
                    break;
                }
                uint64_t gate_group_inner = 0;
                uint64_t up_group_inner = 0;
                id<MTLBuffer> gate_group_buf =
                    q4_grouped_cache_views ?
                    ds4_gpu_wrap_model_exact_range(model_map,
                                                   model_size,
                                                   gate_offset + group_rel,
                                                   group_bytes,
                                                   &gate_group_inner) :
                    ds4_gpu_wrap_model_exact_range_transient(model_map,
                                                             model_size,
                                                             gate_offset + group_rel,
                                                             group_bytes,
                                                             &gate_group_inner);
                id<MTLBuffer> up_group_buf =
                    q4_grouped_cache_views ?
                    ds4_gpu_wrap_model_exact_range(model_map,
                                                   model_size,
                                                   up_offset + group_rel,
                                                   group_bytes,
                                                   &up_group_inner) :
                    ds4_gpu_wrap_model_exact_range_transient(model_map,
                                                             model_size,
                                                             up_offset + group_rel,
                                                             group_bytes,
                                                             &up_group_inner);
                if (!gate_group_buf || !up_group_buf) {
                    ok = 0;
                    break;
                }
                ds4_gpu_moe_expert_group_args group_args = {
                    .expert_base = expert_base,
                    .expert_count = expert_count,
                    .accumulate = 0,
                    .pad0 = 0,
                };
                ok = ds4_gpu_encode_mul_mv_group_q4_pair_swiglu(cb,
                                                                g_moe_mul_mv_group_q4_k_pair_swiglu_pipeline,
                                                                &gate_args,
                                                                &act_args,
                                                                &group_args,
                                                                gate_group_buf,
                                                                (NSUInteger)gate_group_inner,
                                                                up_group_buf,
                                                                (NSUInteger)up_group_inner,
                                                                xbuf,
                                                                ds4_gpu_tensor_offset(x),
                                                                gatebuf,
                                                                ds4_gpu_tensor_offset(gate),
                                                                upbuf,
                                                                ds4_gpu_tensor_offset(up),
                                                                midbuf,
                                                                ds4_gpu_tensor_offset(mid),
                                                                selectedbuf,
                                                                ds4_gpu_tensor_offset(selected),
                                                                weightsbuf,
                                                                ds4_gpu_tensor_offset(weights),
                                                                gate_smem,
                                                                2,
                                                                false);
            }
        } else if (use_q4_expert_address_table) {
            ds4_gpu_dsv4_moe_swiglu_weight_args act_args = {
                .width = expert_mid_dim,
                .rows = pair_rows,
                .gate_row_stride = (uint64_t)expert_mid_dim * sizeof(float),
                .up_row_stride = (uint64_t)expert_mid_dim * sizeof(float),
                .mid_row_stride = (uint64_t)expert_mid_dim * sizeof(float),
                .weight_stride = sizeof(float),
                .write_clamped = 0,
                .clamp_value = clamp,
            };
            ok = ds4_gpu_encode_mul_mv_addr_q4_pair_swiglu(cb,
                                                           g_moe_mul_mv_addr_q4_k_pair_swiglu_pipeline,
                                                           &gate_args,
                                                           &act_args,
                                                           gate_table,
                                                           up_table,
                                                           xbuf,
                                                           ds4_gpu_tensor_offset(x),
                                                           gatebuf,
                                                           ds4_gpu_tensor_offset(gate),
                                                           upbuf,
                                                           ds4_gpu_tensor_offset(up),
                                                           midbuf,
                                                           ds4_gpu_tensor_offset(mid),
                                                           selectedbuf,
                                                           ds4_gpu_tensor_offset(selected),
                                                           weightsbuf,
                                                           ds4_gpu_tensor_offset(weights),
                                                           gate_smem,
                                                           2,
                                                           false);
        } else if (use_q4_expert_table) {
            ds4_gpu_dsv4_moe_swiglu_weight_args act_args = {
                .width = expert_mid_dim,
                .rows = pair_rows,
                .gate_row_stride = (uint64_t)expert_mid_dim * sizeof(float),
                .up_row_stride = (uint64_t)expert_mid_dim * sizeof(float),
                .mid_row_stride = (uint64_t)expert_mid_dim * sizeof(float),
                .weight_stride = sizeof(float),
                .write_clamped = 0,
                .clamp_value = clamp,
            };
            ok = ds4_gpu_encode_mul_mv_table_q4_pair_swiglu(cb,
                                                            g_moe_mul_mv_table_q4_k_pair_swiglu_pipeline,
                                                            &gate_args,
                                                            &act_args,
                                                            gate_table,
                                                            up_table,
                                                            xbuf,
                                                            ds4_gpu_tensor_offset(x),
                                                            gatebuf,
                                                            ds4_gpu_tensor_offset(gate),
                                                            upbuf,
                                                            ds4_gpu_tensor_offset(up),
                                                            midbuf,
                                                            ds4_gpu_tensor_offset(mid),
                                                            selectedbuf,
                                                            ds4_gpu_tensor_offset(selected),
                                                            weightsbuf,
                                                            ds4_gpu_tensor_offset(weights),
                                                            gate_smem,
                                                            2,
                                                            false,
                                                            q4_table_queue_residency);
        } else if (use_q4_group6_experts) {
            ds4_gpu_dsv4_moe_swiglu_weight_args act_args = {
                .width = expert_mid_dim,
                .rows = pair_rows,
                .gate_row_stride = (uint64_t)expert_mid_dim * sizeof(float),
                .up_row_stride = (uint64_t)expert_mid_dim * sizeof(float),
                .mid_row_stride = (uint64_t)expert_mid_dim * sizeof(float),
                .weight_stride = sizeof(float),
                .write_clamped = 0,
                .clamp_value = clamp,
            };
            ok = ds4_gpu_encode_mul_mv_group6_pair_swiglu(cb,
                                                            g_moe_mul_mv_group6_q4_k_pair_swiglu_pipeline,
                                                            &gate_args,
                                                            &act_args,
                                                            gate_group6_bufs,
                                                            gate_group6_offsets,
                                                            up_group6_bufs,
                                                            up_group6_offsets,
                                                            xbuf,
                                                            ds4_gpu_tensor_offset(x),
                                                            gatebuf,
                                                            ds4_gpu_tensor_offset(gate),
                                                            upbuf,
                                                            ds4_gpu_tensor_offset(up),
                                                            midbuf,
                                                            ds4_gpu_tensor_offset(mid),
                                                            selectedbuf,
                                                            ds4_gpu_tensor_offset(selected),
                                                            weightsbuf,
                                                            ds4_gpu_tensor_offset(weights),
                                                            gate_smem,
                                                            2,
                                                            false);
        } else if (use_q4_group8_experts) {
            ds4_gpu_dsv4_moe_swiglu_weight_args act_args = {
                .width = expert_mid_dim,
                .rows = pair_rows,
                .gate_row_stride = (uint64_t)expert_mid_dim * sizeof(float),
                .up_row_stride = (uint64_t)expert_mid_dim * sizeof(float),
                .mid_row_stride = (uint64_t)expert_mid_dim * sizeof(float),
                .weight_stride = sizeof(float),
                .write_clamped = 0,
                .clamp_value = clamp,
            };
            ok = ds4_gpu_encode_mul_mv_group8_pair_swiglu(cb,
                                                            g_moe_mul_mv_group8_q4_k_pair_swiglu_pipeline,
                                                            &gate_args,
                                                            &act_args,
                                                            gate_group8_bufs,
                                                            gate_group8_offsets,
                                                            up_group8_bufs,
                                                            up_group8_offsets,
                                                            xbuf,
                                                            ds4_gpu_tensor_offset(x),
                                                            gatebuf,
                                                            ds4_gpu_tensor_offset(gate),
                                                            upbuf,
                                                            ds4_gpu_tensor_offset(up),
                                                            midbuf,
                                                            ds4_gpu_tensor_offset(mid),
                                                            selectedbuf,
                                                            ds4_gpu_tensor_offset(selected),
                                                            weightsbuf,
                                                            ds4_gpu_tensor_offset(weights),
                                                            gate_smem,
                                                            2,
                                                            false);
        } else if (use_q4_group24_experts) {
            ok = ds4_gpu_encode_mul_mv_group24_id(cb,
                                                  g_moe_mul_mv_group24_q4_k_id_pipeline,
                                                  &gate_args,
                                                  gate_group24_bufs,
                                                  gate_group24_offsets,
                                                  xbuf,
                                                  ds4_gpu_tensor_offset(x),
                                                  gatebuf,
                                                  ds4_gpu_tensor_offset(gate),
                                                  selectedbuf,
                                                  ds4_gpu_tensor_offset(selected),
                                                  gate_smem,
                                                  2,
                                                  false) &&
                 ds4_gpu_encode_mul_mv_group24_id(cb,
                                                  g_moe_mul_mv_group24_q4_k_id_pipeline,
                                                  &gate_args,
                                                  up_group24_bufs,
                                                  up_group24_offsets,
                                                  xbuf,
                                                  ds4_gpu_tensor_offset(x),
                                                  upbuf,
                                                  ds4_gpu_tensor_offset(up),
                                                  selectedbuf,
                                                  ds4_gpu_tensor_offset(selected),
                                                  gate_smem,
                                                  2,
                                                  false);
        } else if (use_q4_gather_slots || use_selected_slots) {
            ds4_gpu_dsv4_moe_swiglu_weight_args act_args = {
                .width = expert_mid_dim,
                .rows = pair_rows,
                .gate_row_stride = (uint64_t)expert_mid_dim * sizeof(float),
                .up_row_stride = (uint64_t)expert_mid_dim * sizeof(float),
                .mid_row_stride = (uint64_t)expert_mid_dim * sizeof(float),
                .weight_stride = sizeof(float),
                .write_clamped = 0,
                .clamp_value = clamp,
            };
            if (use_stream_expert_addr_table) {
                if (use_stream_expert_masked_addr_table) {
                    if (use_stream_expert_split_deferred) {
                        const bool stream_split_profile =
                            getenv("DS4_METAL_STREAMING_EXPERT_SPLIT_PROFILE") != NULL;
                        const bool stream_split_timing =
                            stream_split_profile ||
                            ds4_gpu_stream_expert_timing_summary_enabled();
                        double stream_split_t0 =
                            stream_split_timing ? ds4_gpu_now_ms() : 0.0;
                        ds4_gpu_stream_expert_split_args resident_pair_args = {
                            .active_mask = stream_expert_resident_mask,
                            .accumulate = 0u,
                        };
                        ok = ds4_gpu_encode_mul_mv_addr_iq2_pair_swiglu_masked(cb,
                                                                               g_moe_mul_mv_addr_iq2_xxs_pair_swiglu_masked_pipeline,
                                                                               &gate_args,
                                                                               &act_args,
                                                                               &resident_pair_args,
                                                                               stream_slot_entries,
                                                                               stream_gate_addr_buf,
                                                                               stream_up_addr_buf,
                                                                               xbuf,
                                                                               ds4_gpu_tensor_offset(x),
                                                                               gatebuf,
                                                                               ds4_gpu_tensor_offset(gate),
                                                                               upbuf,
                                                                               ds4_gpu_tensor_offset(up),
                                                                               midbuf,
                                                                               ds4_gpu_tensor_offset(mid),
                                                                               selected_exec_buf,
                                                                               selected_exec_off,
                                                                               weightsbuf,
                                                                               ds4_gpu_tensor_offset(weights),
                                                                               gate_smem,
                                                                               2,
                                                                               false);
                        if (ok) {
                            ok = ds4_gpu_flush_commands();
                            if (ok) {
                                cb = ds4_gpu_command_buffer(&owned);
                                if (!cb) ok = 0;
                            }
                        }
                        const double stream_split_resident_ms =
                            stream_split_timing ? ds4_gpu_now_ms() - stream_split_t0 : 0.0;
                        if (stream_split_timing) stream_split_t0 = ds4_gpu_now_ms();
                        const double stream_split_missing_start_ms = stream_split_t0;
                        double stream_split_missing_load_ms = 0.0;
                        double stream_split_missing_slot_ms = 0.0;
                        double stream_split_missing_prune_ms = 0.0;
                        double stream_split_missing_addr_ms = 0.0;
                        double stream_split_missing_wait_ms = 0.0;
                        if (ok) {
                            ok = ds4_gpu_stream_expert_cache_load_selected_missing(
                                    model_map,
                                    model_size,
                                    layer_index,
                                    selected_ids,
                                    n_total_expert,
                                    n_expert,
                                    stream_gate_abs_offsets,
                                    stream_up_abs_offsets,
                                    stream_down_abs_offsets,
                                    gate_expert_bytes,
                                    down_expert_bytes,
                                    stream_expert_missing_mask,
                                    stream_slot_entries);
                            if (stream_split_timing) {
                                const double now_ms = ds4_gpu_now_ms();
                                stream_split_missing_load_ms =
                                    now_ms - stream_split_t0;
                                stream_split_t0 = now_ms;
                            }
                            if (ok) {
                                for (uint32_t i = 0; i < n_expert; i++) {
                                    if ((stream_expert_missing_mask & (1u << i)) == 0) continue;
                                    ds4_gpu_stream_expert_cache_entry *entry =
                                        stream_slot_entries[i];
                                    if (!entry) {
                                        ok = 0;
                                        break;
                                    }
                                    gate_slot_bufs[i] = entry->gate_buffer;
                                    gate_slot_offsets[i] = entry->gate_inner;
                                    up_slot_bufs[i] = entry->up_buffer;
                                    up_slot_offsets[i] = entry->up_inner;
                                    down_slot_bufs[i] = entry->down_buffer;
                                    down_slot_offsets[i] = entry->down_inner;
                                }
                            }
                        }
                        if (stream_split_timing) {
                            const double now_ms = ds4_gpu_now_ms();
                            stream_split_missing_slot_ms =
                                now_ms - stream_split_t0;
                            stream_split_t0 = now_ms;
                        }
                        if (ok) {
                            ds4_gpu_stream_expert_cache_prune_layer(layer_index,
                                                                    n_total_expert,
                                                                    n_expert,
                                                                    selected_ids,
                                                                    n_expert);
                            ds4_gpu_stream_expert_cache_prune_global(layer_index,
                                                                     selected_ids,
                                                                     n_expert);
                            if (stream_split_timing) {
                                const double now_ms = ds4_gpu_now_ms();
                                stream_split_missing_prune_ms =
                                    now_ms - stream_split_t0;
                                stream_split_t0 = now_ms;
                            }
                            ok = ds4_gpu_stream_expert_cache_addr_buffers(layer_index,
                                                                          &stream_gate_addr_buf,
                                                                          &stream_up_addr_buf,
                                                                          &stream_down_addr_buf);
                            if (stream_split_timing) {
                                const double now_ms = ds4_gpu_now_ms();
                                stream_split_missing_addr_ms =
                                    now_ms - stream_split_t0;
                                stream_split_t0 = now_ms;
                            }
                        }
                        if (ok) {
                            /*
                             * The resident stage was submitted before the
                             * CPU read of missing experts so I/O can overlap
                             * with GPU work. The missing stage reuses the same
                             * gate/up/mid scratch buffers, so it must not
                             * execute until the resident command buffer has
                             * finished. The down/sum pass is issued once after
                             * all six mid slots exist; this keeps the final
                             * accumulation order stable regardless of the
                             * resident/missing split.
                             */
                            ok = ds4_gpu_wait_pending_command_buffers(
                                    "streaming expert split resident");
                            if (stream_split_timing) {
                                const double now_ms = ds4_gpu_now_ms();
                                stream_split_missing_wait_ms =
                                    now_ms - stream_split_t0;
                                stream_split_t0 = now_ms;
                            }
                        }
                        const double stream_split_missing_ms =
                            stream_split_timing ?
                                ds4_gpu_now_ms() - stream_split_missing_start_ms :
                                0.0;
                        ds4_gpu_stream_expert_split_args missing_pair_args = {
                            .active_mask = stream_expert_missing_mask,
                            .accumulate = 0u,
                        };
                        ds4_gpu_stream_expert_split_args all_down_args = {
                            .active_mask = 0x3fu,
                            .accumulate = 0u,
                        };
                        if (ok) {
                            ok = ds4_gpu_encode_mul_mv_addr_iq2_pair_swiglu_masked(cb,
                                                                                   g_moe_mul_mv_addr_iq2_xxs_pair_swiglu_masked_pipeline,
                                                                                   &gate_args,
                                                                                   &act_args,
                                                                                   &missing_pair_args,
                                                                                   stream_slot_entries,
                                                                                   stream_gate_addr_buf,
                                                                                   stream_up_addr_buf,
                                                                                   xbuf,
                                                                                   ds4_gpu_tensor_offset(x),
                                                                                   gatebuf,
                                                                                   ds4_gpu_tensor_offset(gate),
                                                                                   upbuf,
                                                                                   ds4_gpu_tensor_offset(up),
                                                                                   midbuf,
                                                                                   ds4_gpu_tensor_offset(mid),
                                                                                   selected_exec_buf,
                                                                                   selected_exec_off,
                                                                                   weightsbuf,
                                                                                   ds4_gpu_tensor_offset(weights),
                                                                                   gate_smem,
                                                                                   2,
                                                                                   false);
                        }
                        if (ok) {
                            ok = ds4_gpu_encode_mul_mv_addr_q2_sum6_masked(cb,
                                                                           g_moe_mul_mv_addr_q2_k_sum6_masked_pipeline,
                                                                           &down_args,
                                                                           &all_down_args,
                                                                           stream_slot_entries,
                                                                           stream_down_addr_buf,
                                                                           midbuf,
                                                                           ds4_gpu_tensor_offset(mid),
                                                                           outbuf,
                                                                           ds4_gpu_tensor_offset(out),
                                                                           selected_exec_buf,
                                                                           selected_exec_off,
                                                                           down_smem,
                                                                           2);
                        }
                        stream_expert_split_completed = ok;
                        if (stream_split_timing) {
                            ds4_gpu_stream_expert_timing_note_split(
                                    stream_expert_resident_mask,
                                    stream_expert_missing_mask,
                                    stream_split_resident_ms,
                                    stream_split_missing_ms);
                            ds4_gpu_stream_expert_timing_note_split_missing_detail(
                                    stream_split_missing_load_ms,
                                    stream_split_missing_slot_ms,
                                    stream_split_missing_prune_ms,
                                    stream_split_missing_addr_ms,
                                    stream_split_missing_wait_ms);
                        }
                        if (stream_split_profile) {
                            fprintf(stderr,
                                    "ds4: Metal streaming expert split layer=%u "
                                    "resident=0x%02x missing=0x%02x resident_submit=%.3f ms "
                                    "missing_bind=%.3f ms\n",
                                    layer_index,
                                    stream_expert_resident_mask,
                                    stream_expert_missing_mask,
                                    stream_split_resident_ms,
                                    stream_split_missing_ms);
                        }
                    } else {
                        ds4_gpu_stream_expert_split_args split_args = {
                            .active_mask = 0x3fu,
                            .accumulate = 0u,
                        };
                        ok = ds4_gpu_encode_mul_mv_addr_iq2_pair_swiglu_masked(cb,
                                                                               g_moe_mul_mv_addr_iq2_xxs_pair_swiglu_masked_pipeline,
                                                                               &gate_args,
                                                                               &act_args,
                                                                               &split_args,
                                                                               stream_slot_entries,
                                                                               stream_gate_addr_buf,
                                                                               stream_up_addr_buf,
                                                                               xbuf,
                                                                               ds4_gpu_tensor_offset(x),
                                                                               gatebuf,
                                                                               ds4_gpu_tensor_offset(gate),
                                                                               upbuf,
                                                                               ds4_gpu_tensor_offset(up),
                                                                               midbuf,
                                                                               ds4_gpu_tensor_offset(mid),
                                                                               selected_exec_buf,
                                                                               selected_exec_off,
                                                                               weightsbuf,
                                                                               ds4_gpu_tensor_offset(weights),
                                                                               gate_smem,
                                                                               2,
                                                                               false);
                    }
                } else {
                    ok = ds4_gpu_encode_mul_mv_addr_iq2_pair_swiglu(cb,
                                                                    g_moe_mul_mv_addr_iq2_xxs_pair_swiglu_pipeline,
                                                                    &gate_args,
                                                                    &act_args,
                                                                    stream_slot_entries,
                                                                    n_expert,
                                                                    stream_gate_addr_buf,
                                                                    stream_up_addr_buf,
                                                                    xbuf,
                                                                    ds4_gpu_tensor_offset(x),
                                                                    gatebuf,
                                                                    ds4_gpu_tensor_offset(gate),
                                                                    upbuf,
                                                                    ds4_gpu_tensor_offset(up),
                                                                    midbuf,
                                                                    ds4_gpu_tensor_offset(mid),
                                                                    selected_exec_buf,
                                                                    selected_exec_off,
                                                                    weightsbuf,
                                                                    ds4_gpu_tensor_offset(weights),
                                                                    gate_smem,
                                                                    2,
                                                                    false,
                                                                    nil,
                                                                    nil);
                }
            } else {
                ok = (!use_stream_expert_cache ||
                      ds4_gpu_stream_expert_cache_mark_entries_inflight(
                              stream_slot_entries,
                              n_expert,
                              0)) &&
                     ds4_gpu_encode_mul_mv_slots6_pair_swiglu(cb,
                                                              slots_pair_swiglu_pipeline,
                                                              &gate_args,
                                                              &act_args,
                                                              gate_slot_bufs,
                                                              gate_slot_offsets,
                                                              up_slot_bufs,
                                                              up_slot_offsets,
                                                              xbuf,
                                                              ds4_gpu_tensor_offset(x),
                                                              gatebuf,
                                                              ds4_gpu_tensor_offset(gate),
                                                              upbuf,
                                                              ds4_gpu_tensor_offset(up),
                                                              midbuf,
                                                              ds4_gpu_tensor_offset(mid),
                                                              weightsbuf,
                                                              ds4_gpu_tensor_offset(weights),
                                                              gate_smem,
                                                              2,
                                                              false);
            }
        } else if (fuse_pair_swiglu) {
            ds4_gpu_dsv4_moe_swiglu_weight_args act_args = {
                .width = expert_mid_dim,
                .rows = pair_rows,
                .gate_row_stride = (uint64_t)expert_mid_dim * sizeof(float),
                .up_row_stride = (uint64_t)expert_mid_dim * sizeof(float),
                .mid_row_stride = (uint64_t)expert_mid_dim * sizeof(float),
                .weight_stride = sizeof(float),
                .write_clamped = 0,
                .clamp_value = clamp,
            };
            ok = ds4_gpu_encode_mul_mv_id_pair_swiglu(cb,
                                                        pair_swiglu_pipeline,
                                                        &gate_args,
                                                        &act_args,
                                                        gate_buf,
                                                        (NSUInteger)gate_inner,
                                                        up_buf,
                                                        (NSUInteger)up_inner,
                                                        xbuf,
                                                        ds4_gpu_tensor_offset(x),
                                                        gatebuf,
                                                        ds4_gpu_tensor_offset(gate),
                                                        upbuf,
                                                        ds4_gpu_tensor_offset(up),
                                                        midbuf,
                                                        ds4_gpu_tensor_offset(mid),
                                                        selectedbuf,
                                                        ds4_gpu_tensor_offset(selected),
                                                        weightsbuf,
                                                        ds4_gpu_tensor_offset(weights),
                                                        gate_smem,
                                                        2,
                                                        false);
        } else if (!g_quality_mode &&
                   gate_type == DS4_METAL_TENSOR_IQ2_XXS &&
                   g_moe_mul_mv_id_iq2_xxs_pair_pipeline) {
            ok = ds4_gpu_encode_mul_mv_id_pair(cb,
                                                 g_moe_mul_mv_id_iq2_xxs_pair_pipeline,
                                                 &gate_args,
                                                 gate_buf,
                                                 (NSUInteger)gate_inner,
                                                 up_buf,
                                                 (NSUInteger)up_inner,
                                                 xbuf,
                                                 ds4_gpu_tensor_offset(x),
                                                 gatebuf,
                                                 ds4_gpu_tensor_offset(gate),
                                                 upbuf,
                                                 ds4_gpu_tensor_offset(up),
                                                 selectedbuf,
                                                 ds4_gpu_tensor_offset(selected),
                                                 gate_smem,
                                                 2,
                                                 false);
        } else if (!g_quality_mode &&
                   gate_type == DS4_METAL_TENSOR_Q4_K &&
                   g_moe_mul_mv_id_q4_k_pair_pipeline) {
            ok = ds4_gpu_encode_mul_mv_id_pair(cb,
                                                 g_moe_mul_mv_id_q4_k_pair_pipeline,
                                                 &gate_args,
                                                 gate_buf,
                                                 (NSUInteger)gate_inner,
                                                 up_buf,
                                                 (NSUInteger)up_inner,
                                                 xbuf,
                                                 ds4_gpu_tensor_offset(x),
                                                 gatebuf,
                                                 ds4_gpu_tensor_offset(gate),
                                                 upbuf,
                                                 ds4_gpu_tensor_offset(up),
                                                 selectedbuf,
                                                 ds4_gpu_tensor_offset(selected),
                                                 gate_smem,
                                                 2,
                                                 false);
        } else {
            ok = ds4_gpu_encode_mul_mv_id(cb,
                                            gate_mv_pipeline,
                                            &gate_args,
                                            gate_buf,
                                            (NSUInteger)gate_inner,
                                            xbuf,
                                            ds4_gpu_tensor_offset(x),
                                            gatebuf,
                                            ds4_gpu_tensor_offset(gate),
                                            selectedbuf,
                                            ds4_gpu_tensor_offset(selected),
                                            gate_smem,
                                            gate_nsg,
                                            gate_rows_per_group_is_nr0) &&
                 ds4_gpu_encode_mul_mv_id(cb,
                                            gate_mv_pipeline,
                                            &gate_args,
                                            up_buf,
                                            (NSUInteger)up_inner,
                                            xbuf,
                                            ds4_gpu_tensor_offset(x),
                                            upbuf,
                                            ds4_gpu_tensor_offset(up),
                                            selectedbuf,
                                            ds4_gpu_tensor_offset(selected),
                                            gate_smem,
                                            gate_nsg,
                                            gate_rows_per_group_is_nr0);
        }
        DS4_METAL_PROFILE_MOE_ONE_STAGE("gate_up");
        if (ok && (!fuse_pair_swiglu || use_q4_group24_experts)) {
            ok = ds4_gpu_encode_moe_swiglu_weight(cb,
                                                    gatebuf,
                                                    ds4_gpu_tensor_offset(gate),
                                                    upbuf,
                                                    ds4_gpu_tensor_offset(up),
                                                    midbuf,
                                                    ds4_gpu_tensor_offset(mid),
                                                    weightsbuf,
                                                    ds4_gpu_tensor_offset(weights),
                                                    expert_mid_dim,
                                                    pair_rows,
                                                    clamp,
                                                    false);
        }
        DS4_METAL_PROFILE_MOE_ONE_STAGE("activation_weight");

        id<MTLBuffer> down_dst = n_expert == 1 ? outbuf : (expertsbuf ? expertsbuf : g_moe_down_scratch_buffer);
        NSUInteger down_dst_off = n_expert == 1 ? ds4_gpu_tensor_offset(out) :
            (expertsbuf ? ds4_gpu_tensor_offset(experts) : 0);
        if (ok && stream_expert_split_completed) {
            /* The split path already wrote the resident partial output and
             * accumulated the missing experts into out. */
        } else if (ok && use_q4_grouped_experts) {
            bool first_group = true;
            for (uint32_t expert_base = 0; ok && expert_base < n_total_expert; expert_base += q4_expert_group_size) {
                const uint32_t expert_count =
                    q4_expert_group_size < n_total_expert - expert_base ?
                    q4_expert_group_size : n_total_expert - expert_base;
                if ((uint64_t)expert_base > UINT64_MAX / down_expert_bytes ||
                    (uint64_t)expert_count > UINT64_MAX / down_expert_bytes) {
                    ok = 0;
                    break;
                }
                const uint64_t group_rel = (uint64_t)expert_base * down_expert_bytes;
                const uint64_t group_bytes = (uint64_t)expert_count * down_expert_bytes;
                if (group_rel > UINT64_MAX - down_offset) {
                    ok = 0;
                    break;
                }
                uint64_t down_group_inner = 0;
                id<MTLBuffer> down_group_buf =
                    q4_grouped_cache_views ?
                    ds4_gpu_wrap_model_exact_range(model_map,
                                                   model_size,
                                                   down_offset + group_rel,
                                                   group_bytes,
                                                   &down_group_inner) :
                    ds4_gpu_wrap_model_exact_range_transient(model_map,
                                                             model_size,
                                                             down_offset + group_rel,
                                                             group_bytes,
                                                             &down_group_inner);
                if (!down_group_buf) {
                    ok = 0;
                    break;
                }
                ds4_gpu_moe_expert_group_args group_args = {
                    .expert_base = expert_base,
                    .expert_count = expert_count,
                    .accumulate = first_group ? 0u : 1u,
                    .pad0 = 0,
                };
                ok = ds4_gpu_encode_mul_mv_group_q4_sum6(cb,
                                                          g_moe_mul_mv_group_q4_k_sum6_pipeline,
                                                          &down_args,
                                                          &group_args,
                                                          down_group_buf,
                                                          (NSUInteger)down_group_inner,
                                                          midbuf,
                                                          ds4_gpu_tensor_offset(mid),
                                                          outbuf,
                                                          ds4_gpu_tensor_offset(out),
                                                          selectedbuf,
                                                          ds4_gpu_tensor_offset(selected),
                                                          down_smem,
                                                          2);
                first_group = false;
            }
        } else if (ok && use_q4_expert_address_table) {
            ok = ds4_gpu_encode_mul_mv_addr_q4_sum6(cb,
                                                    g_moe_mul_mv_addr_q4_k_sum6_pipeline,
                                                    &down_args,
                                                    down_table,
                                                    midbuf,
                                                    ds4_gpu_tensor_offset(mid),
                                                    outbuf,
                                                    ds4_gpu_tensor_offset(out),
                                                    selectedbuf,
                                                    ds4_gpu_tensor_offset(selected),
                                                    down_smem,
                                                    2);
        } else if (ok && use_q4_expert_table) {
            ok = ds4_gpu_encode_mul_mv_table_q4_sum6(cb,
                                                    g_moe_mul_mv_table_q4_k_sum6_pipeline,
                                                    &down_args,
                                                    down_table,
                                                    midbuf,
                                                    ds4_gpu_tensor_offset(mid),
                                                    outbuf,
                                                    ds4_gpu_tensor_offset(out),
                                                    selectedbuf,
                                                    ds4_gpu_tensor_offset(selected),
                                                    down_smem,
                                                    2,
                                                    q4_table_queue_residency);
        } else if (ok && use_q4_group6_experts) {
            ok = ds4_gpu_encode_mul_mv_group6_sum6(cb,
                                                    g_moe_mul_mv_group6_q4_k_sum6_pipeline,
                                                    &down_args,
                                                    down_group6_bufs,
                                                    down_group6_offsets,
                                                    midbuf,
                                                    ds4_gpu_tensor_offset(mid),
                                                    outbuf,
                                                    ds4_gpu_tensor_offset(out),
                                                    selectedbuf,
                                                    ds4_gpu_tensor_offset(selected),
                                                    down_smem,
                                                    2);
        } else if (ok && use_q4_group8_experts) {
            ok = ds4_gpu_encode_mul_mv_group8_sum6(cb,
                                                    g_moe_mul_mv_group8_q4_k_sum6_pipeline,
                                                    &down_args,
                                                    down_group8_bufs,
                                                    down_group8_offsets,
                                                    midbuf,
                                                    ds4_gpu_tensor_offset(mid),
                                                    outbuf,
                                                    ds4_gpu_tensor_offset(out),
                                                    selectedbuf,
                                                    ds4_gpu_tensor_offset(selected),
                                                    down_smem,
                                                    2);
        } else if (ok && use_q4_group24_experts) {
            ok = ds4_gpu_encode_mul_mv_group24_sum6(cb,
                                                    g_moe_mul_mv_group24_q4_k_sum6_pipeline,
                                                    &down_args,
                                                    down_group24_bufs,
                                                    down_group24_offsets,
                                                    midbuf,
                                                    ds4_gpu_tensor_offset(mid),
                                                    outbuf,
                                                    ds4_gpu_tensor_offset(out),
                                                    selectedbuf,
                                                    ds4_gpu_tensor_offset(selected),
                                                    down_smem,
                                                    2);
        } else if (ok && (use_q4_gather_slots || use_selected_slots)) {
            if (use_stream_expert_addr_table) {
                if (down_type == DS4_METAL_TENSOR_IQ2_XXS) {
                    ok = ds4_gpu_encode_mul_mv_addr_iq2(cb,
                                                        g_moe_mul_mv_addr_iq2_xxs_pipeline,
                                                        &down_args,
                                                        stream_slot_entries,
                                                        n_expert,
                                                        stream_down_addr_buf,
                                                        midbuf,
                                                        ds4_gpu_tensor_offset(mid),
                                                        down_dst,
                                                        down_dst_off,
                                                        selected_exec_buf,
                                                        selected_exec_off,
                                                        down_smem,
                                                        2,
                                                        false);
                } else if (use_stream_expert_masked_addr_table) {
                    ds4_gpu_stream_expert_split_args split_args = {
                        .active_mask = 0x3fu,
                        .accumulate = 0u,
                    };
                    ok = ds4_gpu_encode_mul_mv_addr_q2_sum6_masked(cb,
                                                                   g_moe_mul_mv_addr_q2_k_sum6_masked_pipeline,
                                                                   &down_args,
                                                                   &split_args,
                                                                   stream_slot_entries,
                                                                   stream_down_addr_buf,
                                                                   midbuf,
                                                                   ds4_gpu_tensor_offset(mid),
                                                                   outbuf,
                                                                   ds4_gpu_tensor_offset(out),
                                                                   selected_exec_buf,
                                                                   selected_exec_off,
                                                                   down_smem,
                                                                   2);
                } else {
                    ok = ds4_gpu_encode_mul_mv_addr_q2_sum6(cb,
                                                             g_moe_mul_mv_addr_q2_k_sum6_pipeline,
                                                             &down_args,
                                                             stream_slot_entries,
                                                             n_expert,
                                                             stream_down_addr_buf,
                                                             midbuf,
                                                             ds4_gpu_tensor_offset(mid),
                                                             outbuf,
                                                             ds4_gpu_tensor_offset(out),
                                                             selected_exec_buf,
                                                             selected_exec_off,
                                                             down_smem,
                                                             2,
                                                             nil);
                }
            } else {
                ok = (!use_stream_expert_cache ||
                      ds4_gpu_stream_expert_cache_mark_entries_inflight(
                              stream_slot_entries,
                              n_expert,
                              0)) &&
                     ds4_gpu_encode_mul_mv_slots6_sum6(cb,
                                                       slots_sum6_pipeline,
                                                       &down_args,
                                                       down_slot_bufs,
                                                       down_slot_offsets,
                                                       midbuf,
                                                       ds4_gpu_tensor_offset(mid),
                                                       outbuf,
                                                       ds4_gpu_tensor_offset(out),
                                                       down_smem,
                                                       2);
            }
        } else if (ok && direct_down_sum) {
            ok = ds4_gpu_encode_mul_mv_id_sum6(cb,
                                                 down_sum6_pipeline,
                                                 &down_args,
                                                 down_buf,
                                                 (NSUInteger)down_inner,
                                                 midbuf,
                                                 ds4_gpu_tensor_offset(mid),
                                                 outbuf,
                                                 ds4_gpu_tensor_offset(out),
                                                 selectedbuf,
                                                 ds4_gpu_tensor_offset(selected),
                                                 add_in ? ds4_gpu_tensor_buffer(add_in) : nil,
                                                 add_in ? ds4_gpu_tensor_offset(add_in) : 0,
                                                 down_smem,
                                                 2);
        } else if (ok) {
            ok = ds4_gpu_encode_mul_mv_id(cb,
                                                 down_mv_pipeline,
                                                 &down_args,
                                                 down_buf,
                                                 (NSUInteger)down_inner,
                                                 midbuf,
                                                 ds4_gpu_tensor_offset(mid),
                                                 down_dst,
                                                 down_dst_off,
                                                 selectedbuf,
                                                 ds4_gpu_tensor_offset(selected),
                                                 down_smem,
                                                 down_nsg,
                                                 down_rows_per_group_is_nr0);
        }
        DS4_METAL_PROFILE_MOE_ONE_STAGE("down");
        if (ok && n_expert > 1 && !direct_down_sum && !stream_expert_split_completed) {
            ok = ds4_gpu_encode_moe_sum_experts(cb,
                                                       down_dst,
                                                       down_dst_off,
                                                       outbuf,
                                                       ds4_gpu_tensor_offset(out),
                                                       out_dim,
                                                       n_expert,
                                                       n_tokens);
        }
        DS4_METAL_PROFILE_MOE_ONE_STAGE("sum");
        if (!ok) { if (getenv("DS4_GLM_TP_DEBUG")) fprintf(stderr, "ds4: routed_moe_one silent return at line %d\n", 34395); return 0; }

        if (!ds4_gpu_finish_command_buffer(cb, owned, "routed tensor MoE")) { if (getenv("DS4_GLM_TP_DEBUG")) fprintf(stderr, "ds4: routed_moe_one silent return at line %d\n", 34397); return 0; }
        if (q4_grouped_boundary || q4_exact_boundary || q4_table_boundary) {
            if (ds4_gpu_end_commands() == 0 || ds4_gpu_begin_commands() == 0) {
                if (getenv("DS4_GLM_TP_DEBUG")) fprintf(stderr, "ds4: routed_moe_one silent return at line %d\n", 34372);
                return 0;
            }
        }
#undef DS4_METAL_PROFILE_MOE_ONE_STAGE
    }

    return 1;
}

int ds4_gpu_routed_moe_batch_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        ds4_gpu_tensor       *experts,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                down_offset,
        uint32_t                gate_type,
        uint32_t                down_type,
        uint64_t                gate_expert_bytes,
        uint64_t                gate_row_bytes,
        uint64_t                down_expert_bytes,
        uint64_t                down_row_bytes,
        uint32_t                expert_in_dim,
        uint32_t                expert_mid_dim,
        uint32_t                out_dim,
        const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *weights,
        uint32_t                n_total_expert,
        uint32_t                n_expert,
        float                   clamp,
        const ds4_gpu_tensor *x,
        uint32_t                layer_index,
        uint32_t                n_tokens,
        bool                   *mid_is_f16,
        bool                    force_resident) {
    (void)force_resident;
    if (!g_initialized && !ds4_gpu_init()) return 0;
    /* TP sharding (see ds4_gpu_routed_moe_one_tensor): bind from the owned
     * expert range and rebase ids in the kernels. */
    uint32_t first_expert = 0;
    uint32_t n_bind_expert = 0;
    ds4_gpu_tp_expert_range(n_total_expert, &first_expert, &n_bind_expert);
    const int32_t tp_expert_base_host = (int32_t)first_expert;
    gate_offset += (uint64_t)first_expert * gate_expert_bytes;
    up_offset += (uint64_t)first_expert * gate_expert_bytes;
    down_offset += (uint64_t)first_expert * down_expert_bytes;
    if (!out || !gate || !up || !mid || !x || !model_map || !selected || !weights ||
        n_tokens == 0 || n_total_expert == 0 || n_expert == 0 ||
        n_expert > DS4_METAL_MAX_ROUTED_EXPERT_USED) {
        return 0;
    }
    if (gate_expert_bytes == 0 || down_expert_bytes == 0 ||
        gate_row_bytes == 0 || down_row_bytes == 0) {
        return 0;
    }
    if ((expert_in_dim % 256u) != 0 || (expert_mid_dim % 256u) != 0) return 0;
    if ((uint64_t)n_total_expert > UINT64_MAX / gate_expert_bytes ||
        (uint64_t)n_total_expert > UINT64_MAX / down_expert_bytes) {
        fprintf(stderr, "ds4: Metal routed batch MoE tensor byte size overflow\n");
        return 0;
    }
    const uint64_t gate_tensor_bytes = (uint64_t)n_bind_expert * gate_expert_bytes;
    const uint64_t down_tensor_bytes = (uint64_t)n_bind_expert * down_expert_bytes;

    /*
     * PRO Q4 routed expert tensors are multi-GiB per layer.  A one-token
     * layer-slice prefill should use the one-token path so it can either bind
     * exact tensor views with GPU-selected IDs or fall back to selected-expert
     * views.  Keep this guarded by tensor size so Flash and mixed-Flash Q4 keep
     * their existing fast path.
     */
    const uint64_t q4_selected_min_tensor_bytes = 2ull * 1024ull * 1024ull * 1024ull;
    const bool can_single_token_q4_grouped =
        getenv("DS4_METAL_ENABLE_Q4_GROUPED_EXPERTS") != NULL &&
        getenv("DS4_METAL_DISABLE_Q4_GROUPED_EXPERTS") == NULL &&
        g_moe_mul_mv_group_q4_k_pair_swiglu_pipeline != nil &&
        g_moe_mul_mv_group_q4_k_sum6_pipeline != nil;
    const bool can_single_token_q4_selected_slots =
        ds4_gpu_q4_selected_paths_allowed() &&
        getenv("DS4_METAL_DISABLE_Q4_SELECTED_EXPERT_VIEWS") == NULL &&
        g_moe_mul_mv_slots6_q4_k_pair_swiglu_pipeline != nil &&
        g_moe_mul_mv_slots6_q4_k_sum6_pipeline != nil;
    const bool can_single_token_q4_group6 =
        n_total_expert == 384 &&
        getenv("DS4_METAL_ENABLE_Q4_GROUP6_EXPERT_TABLE") != NULL &&
        getenv("DS4_METAL_DISABLE_Q4_GROUP6_EXPERT_TABLE") == NULL &&
        g_moe_mul_mv_group6_q4_k_pair_swiglu_pipeline != nil &&
        g_moe_mul_mv_group6_q4_k_sum6_pipeline != nil;
    const bool can_single_token_q4_group8 =
        n_total_expert == 384 &&
        getenv("DS4_METAL_ENABLE_Q4_GROUP8_EXPERT_TABLE") != NULL &&
        getenv("DS4_METAL_DISABLE_Q4_GROUP8_EXPERT_TABLE") == NULL &&
        g_moe_mul_mv_group8_q4_k_pair_swiglu_pipeline != nil &&
        g_moe_mul_mv_group8_q4_k_sum6_pipeline != nil;
    const bool can_single_token_q4_group24 =
        n_total_expert == 384 &&
        getenv("DS4_METAL_ENABLE_Q4_GROUP24_EXPERT_TABLE") != NULL &&
        getenv("DS4_METAL_DISABLE_Q4_GROUP24_EXPERT_TABLE") == NULL &&
        g_moe_mul_mv_group24_q4_k_id_pipeline != nil &&
        g_moe_mul_mv_group24_q4_k_sum6_pipeline != nil;
    const uint64_t max_buffer_len = g_device ? (uint64_t)[g_device maxBufferLength] : 0;
    const bool can_single_token_q4_exact_tensor_id =
        n_total_expert == 384 &&
        max_buffer_len != 0 &&
        gate_tensor_bytes <= max_buffer_len &&
        down_tensor_bytes <= max_buffer_len &&
        getenv("DS4_METAL_ENABLE_Q4_EXACT_TENSOR_ID") != NULL &&
        getenv("DS4_METAL_DISABLE_Q4_EXACT_TENSOR_ID") == NULL;
    const bool enable_single_token_q4_expert_table =
        getenv("DS4_METAL_ENABLE_Q4_EXPERT_TABLE") != NULL ||
        ds4_gpu_pro_q4_expert_table_auto_enabled(n_total_expert,
                                                 n_expert,
                                                 gate_tensor_bytes,
                                                 down_tensor_bytes);
    const bool can_single_token_q4_expert_table =
        n_total_expert == 384 &&
        enable_single_token_q4_expert_table &&
        getenv("DS4_METAL_DISABLE_Q4_EXPERT_TABLE") == NULL &&
        g_moe_mul_mv_table_q4_k_pair_swiglu_pipeline != nil &&
        g_moe_mul_mv_table_q4_k_sum6_pipeline != nil &&
        g_moe_table_q4_pair_gate_encoder != nil &&
        g_moe_table_q4_pair_up_encoder != nil &&
        g_moe_table_q4_sum_down_encoder != nil;
    const bool can_single_token_q4_expert_address_table =
        n_total_expert == 384 &&
        getenv("DS4_METAL_ENABLE_Q4_EXPERT_ADDRESS_TABLE") != NULL &&
        getenv("DS4_METAL_DISABLE_Q4_EXPERT_ADDRESS_TABLE") == NULL &&
        g_moe_mul_mv_addr_q4_k_pair_swiglu_pipeline != nil &&
        g_moe_mul_mv_addr_q4_k_sum6_pipeline != nil;
    const bool use_single_token_q4_one_tensor =
        gate_type == DS4_METAL_TENSOR_Q4_K &&
        down_type == DS4_METAL_TENSOR_Q4_K &&
        n_tokens == 1 &&
        n_expert == 6 &&
        n_total_expert >= 128 &&
        (g_ssd_streaming_mode ||
         (gate_tensor_bytes >= q4_selected_min_tensor_bytes &&
          down_tensor_bytes >= q4_selected_min_tensor_bytes)) &&
        !g_quality_mode &&
        getenv("DS4_METAL_MOE_WRITE_CLAMPED_ACT") == NULL &&
        getenv("DS4_METAL_DISABLE_ROUTED_PAIR_SWIGLU_FUSION") == NULL &&
        (can_single_token_q4_grouped ||
         can_single_token_q4_group6 ||
         can_single_token_q4_group8 ||
         can_single_token_q4_group24 ||
         can_single_token_q4_exact_tensor_id ||
         can_single_token_q4_expert_address_table ||
         can_single_token_q4_expert_table ||
         can_single_token_q4_selected_slots);
    if (use_single_token_q4_one_tensor) {
        if (mid_is_f16) *mid_is_f16 = false;
        return ds4_gpu_routed_moe_one_tensor(out,
                                             gate,
                                             up,
                                             mid,
                                             experts,
                                             model_map,
                                             model_size,
                                             gate_offset,
                                             up_offset,
                                             down_offset,
                                             gate_type,
                                             down_type,
                                             gate_expert_bytes,
                                             gate_row_bytes,
                                             down_expert_bytes,
                                             down_row_bytes,
                                             expert_in_dim,
                                             expert_mid_dim,
                                             out_dim,
                                             selected,
                                             weights,
                                             n_total_expert,
                                             n_expert,
                                             clamp,
                                             x,
                                             NULL,
                                             layer_index,
                                             false);
    }

    @autoreleasepool {
        id<MTLBuffer> xbuf = ds4_gpu_tensor_buffer(x);
        id<MTLBuffer> gatebuf = ds4_gpu_tensor_buffer(gate);
        id<MTLBuffer> upbuf = ds4_gpu_tensor_buffer(up);
        id<MTLBuffer> midbuf = ds4_gpu_tensor_buffer(mid);
        id<MTLBuffer> outbuf = ds4_gpu_tensor_buffer(out);
        id<MTLBuffer> expertsbuf = ds4_gpu_tensor_buffer(experts);
        id<MTLBuffer> selectedbuf = ds4_gpu_tensor_buffer(selected);
        id<MTLBuffer> weightsbuf = ds4_gpu_tensor_buffer(weights);
        const uint64_t x_bytes = (uint64_t)n_tokens * expert_in_dim * sizeof(float);
        const uint64_t mid_bytes = (uint64_t)n_tokens * n_expert * expert_mid_dim * sizeof(float);
        const uint64_t out_bytes = (uint64_t)n_tokens * out_dim * sizeof(float);
        const uint64_t selected_bytes = (uint64_t)n_tokens * n_expert * sizeof(int);
        const uint64_t weights_bytes = (uint64_t)n_tokens * n_expert * sizeof(float);
        if (!xbuf || !gatebuf || !upbuf || !midbuf || !outbuf || !selectedbuf || !weightsbuf ||
            ds4_gpu_tensor_bytes(x) < x_bytes ||
            ds4_gpu_tensor_bytes(gate) < mid_bytes ||
            ds4_gpu_tensor_bytes(up) < mid_bytes ||
            ds4_gpu_tensor_bytes(mid) < mid_bytes ||
            ds4_gpu_tensor_bytes(out) < out_bytes ||
            ds4_gpu_tensor_bytes(selected) < selected_bytes ||
            ds4_gpu_tensor_bytes(weights) < weights_bytes) {
            fprintf(stderr, "ds4: Metal routed batch MoE received undersized activation buffers\n");
            return 0;
        }
        uint64_t gate_inner = 0;
        uint64_t up_inner = 0;
        uint64_t down_inner = 0;
        id<MTLBuffer> gate_buf = nil;
        id<MTLBuffer> up_buf = nil;
        id<MTLBuffer> down_buf = nil;
        DS4MetalQ4ExpertTable *gate_table = nil;
        DS4MetalQ4ExpertTable *up_table = nil;
        DS4MetalQ4ExpertTable *down_table = nil;
        id q4_table_layer_residency = nil;
        id<MTLBuffer> stream_gate_addr_buf = nil;
        id<MTLBuffer> stream_up_addr_buf = nil;
        id<MTLBuffer> stream_down_addr_buf = nil;
        id<MTLBuffer> stream_overflow_gate = nil;
        id<MTLBuffer> stream_overflow_up = nil;
        id<MTLBuffer> stream_overflow_down = nil;
        ds4_gpu_stream_expert_cache_entry
            *stream_resources[DS4_METAL_STREAM_EXPERT_CACHE_MAX_EXPERT] = { NULL };
        uint32_t stream_resource_count = 0;
        uint32_t stream_unique = 0;

        const uint32_t pair_rows = n_tokens * n_expert;
        const uint64_t down_scratch_bytes = (uint64_t)pair_rows * out_dim * sizeof(float);

        const uint32_t gate_nr0 = ds4_gpu_routed_mv_nr0(gate_type);
        const uint32_t down_nr0 = ds4_gpu_routed_mv_nr0(down_type);
        id<MTLComputePipelineState> gate_mv_pipeline = ds4_gpu_routed_mv_pipeline(gate_type);
        id<MTLComputePipelineState> down_mv_pipeline = ds4_gpu_routed_mv_pipeline(down_type);
        id<MTLComputePipelineState> gate_mm_pipeline = nil;
        id<MTLComputePipelineState> up_mm_pipeline = nil;
        id<MTLComputePipelineState> down_mm_pipeline = nil;
        id<MTLComputePipelineState> pair_swiglu_mm_pipeline = nil;
        if (gate_nr0 == 0 || down_nr0 == 0 || !gate_mv_pipeline || !down_mv_pipeline) {
            fprintf(stderr, "ds4: unsupported Metal routed batch MoE quant types gate=%u down=%u\n",
                    gate_type, down_type);
            return 0;
        }
        const bool use_iq2_batch_selected_addr =
            ds4_gpu_stream_prefill_batch_selected_addr_enabled(n_tokens,
                                                               n_total_expert,
                                                               n_expert,
                                                               gate_type,
                                                               down_type) &&
            n_tokens > 1 &&
            n_total_expert <= DS4_METAL_STREAM_EXPERT_CACHE_MAX_EXPERT &&
            !g_quality_mode &&
            getenv("DS4_METAL_MOE_WRITE_CLAMPED_ACT") == NULL &&
            getenv("DS4_METAL_DISABLE_ROUTED_PAIR_SWIGLU_FUSION") == NULL &&
            g_moe_mul_mv_addr_iq2_xxs_pair_swiglu_pipeline != nil &&
            g_moe_mul_mv_addr_q2_k_sum6_pipeline != nil;

        ds4_gpu_mul_mv_id_args gate_args =
            ds4_gpu_make_mul_mv_id_args(expert_in_dim, expert_mid_dim, n_total_expert,
                                          gate_row_bytes, gate_expert_bytes,
                                          1, n_expert, n_tokens, gate_nr0);
        gate_args.tp_rank = g_tp_split_rank;
        gate_args.tp_world = g_tp_split_world;
        gate_args.tp_expert_base = tp_expert_base_host;
        ds4_gpu_mul_mv_id_args down_args =
            ds4_gpu_make_mul_mv_id_args(expert_mid_dim, out_dim, n_total_expert,
                                          down_row_bytes, down_expert_bytes,
                                          n_expert, n_expert, n_tokens, down_nr0);
        down_args.tp_rank = g_tp_split_rank;
        down_args.tp_world = g_tp_split_world;
        down_args.tp_expert_base = tp_expert_base_host;
        const bool q4_batch_expert_table_auto =
            gate_type == DS4_METAL_TENSOR_Q4_K &&
            down_type == DS4_METAL_TENSOR_Q4_K &&
            ds4_gpu_pro_q4_expert_table_auto_enabled(n_total_expert,
                                                     n_expert,
                                                     gate_tensor_bytes,
                                                     down_tensor_bytes);
        const bool q4_batch_table_queue_residency =
            ds4_gpu_q4_table_queue_residency_enabled(q4_batch_expert_table_auto);
        const bool enable_q4_batch_expert_table =
            getenv("DS4_METAL_ENABLE_Q4_BATCH_EXPERT_TABLE") != NULL ||
            getenv("DS4_METAL_ENABLE_Q4_EXPERT_TABLE") != NULL ||
            q4_batch_expert_table_auto;
        const bool use_q4_batch_expert_table =
            gate_type == DS4_METAL_TENSOR_Q4_K &&
            down_type == DS4_METAL_TENSOR_Q4_K &&
            n_expert == 6 &&
            n_tokens > 1 &&
            n_total_expert == 384 &&
            gate_tensor_bytes >= q4_selected_min_tensor_bytes &&
            down_tensor_bytes >= q4_selected_min_tensor_bytes &&
            !g_quality_mode &&
            getenv("DS4_METAL_MOE_WRITE_CLAMPED_ACT") == NULL &&
            getenv("DS4_METAL_DISABLE_ROUTED_PAIR_SWIGLU_FUSION") == NULL &&
            getenv("DS4_METAL_DISABLE_Q4_BATCH_EXPERT_TABLE") == NULL &&
            getenv("DS4_METAL_DISABLE_Q4_EXPERT_TABLE") == NULL &&
            g_moe_mul_mv_table_q4_k_pair_swiglu_pipeline != nil &&
            g_moe_mul_mv_table_q4_k_sum6_pipeline != nil &&
            g_moe_table_q4_pair_gate_encoder != nil &&
            g_moe_table_q4_pair_up_encoder != nil &&
            g_moe_table_q4_sum_down_encoder != nil &&
            enable_q4_batch_expert_table &&
            (getenv("DS4_METAL_Q4_TABLE_USE_RESOURCES") != NULL ||
             getenv("DS4_METAL_Q4_TABLE_RESIDENCY_SET") != NULL ||
             q4_batch_table_queue_residency ||
             ds4_gpu_q4_table_model_residency_enabled());
        const bool use_mm_id =
            !use_q4_batch_expert_table &&
            !use_iq2_batch_selected_addr &&
            n_tokens >= 32u &&
            ds4_gpu_mul_mm_id_map0_name(n_expert) != NULL;
        /*
         * MTP verification is neither normal decode nor large prefill: the
         * target model must verify a tiny suffix (up to DSpark's 5-token
         * block) in one layer-major pass.  For that shape the prefill
         * expert-major GEMM path
         * is too large, but the decode pair kernels are exactly the right
         * primitive: they read the same activation once and compute routed
         * gate/up together for every selected expert row.  Keep this limited to
         * tiny batches so ordinary prefill keeps using the higher-throughput
         * grouped matmul path.
         */
        const bool use_tiny_pair_mv =
            !g_quality_mode &&
            n_tokens <= 5u &&
            !use_q4_batch_expert_table &&
            !use_mm_id &&
            ((gate_type == DS4_METAL_TENSOR_IQ2_XXS && g_moe_mul_mv_id_iq2_xxs_pair_pipeline) ||
             (gate_type == DS4_METAL_TENSOR_Q4_K && g_moe_mul_mv_id_q4_k_pair_pipeline));
        id<MTLComputePipelineState> tiny_pair_swiglu_pipeline = nil;
        if (gate_type == DS4_METAL_TENSOR_IQ2_XXS) {
            tiny_pair_swiglu_pipeline = g_moe_mul_mv_id_iq2_xxs_pair_swiglu_pipeline;
        } else if (gate_type == DS4_METAL_TENSOR_Q4_K) {
            tiny_pair_swiglu_pipeline = g_moe_mul_mv_id_q4_k_pair_swiglu_pipeline;
        }
        const bool use_tiny_pair_swiglu =
            use_tiny_pair_mv &&
            tiny_pair_swiglu_pipeline != nil &&
            getenv("DS4_METAL_DISABLE_TINY_PAIR_SWIGLU_FUSION") == NULL &&
            getenv("DS4_METAL_MOE_WRITE_CLAMPED_ACT") == NULL;
        ds4_gpu_mul_mm_id_map_args gate_map_args = { 0 };
        ds4_gpu_mul_mm_id_args gate_mm_args = { 0 };
        ds4_gpu_mul_mm_id_args down_mm_args = { 0 };
        id<MTLComputePipelineState> map_pipeline = nil;
        /*
         * The grouped routed-MoE matmul loads activation tiles as half before
         * using SIMD-group MMA.  Store the SwiGLU/route-weight intermediate in
         * that same precision so the down projection avoids a large F32 mid
         * write/read. --quality keeps the older F32 intermediate.
         */
        const bool request_mid_f16 =
            !g_quality_mode &&
            !use_q4_batch_expert_table &&
            !use_iq2_batch_selected_addr;
        /*
         * Fused gate+up grouped matmul with the SwiGLU epilogue.  The IQ2
         * variant stays opt-in behind its env flag; the Q4_K variant is the
         * default path — same MMA accumulation order and epilogue math as the
         * separate GEMMs + swiglu pass, so the mid tensor is bit-identical.
         */
        const bool use_mm_id_pair_swiglu =
            use_mm_id &&
            g_tp_split_world != 2 &&    /* pair-swiglu mm kernel lacks expert ownership */
            request_mid_f16 &&
            n_expert == 6 &&
            ((gate_type == DS4_METAL_TENSOR_IQ2_XXS &&
              down_type == DS4_METAL_TENSOR_Q2_K &&
              getenv("DS4_METAL_ENABLE_MOE_MM_ID_PAIR_SWIGLU") != NULL) ||
             (gate_type == DS4_METAL_TENSOR_Q4_K &&
              down_type == DS4_METAL_TENSOR_Q4_K)) &&
            getenv("DS4_METAL_DISABLE_MOE_MM_ID_PAIR_SWIGLU") == NULL &&
            getenv("DS4_METAL_MOE_WRITE_CLAMPED_ACT") == NULL &&
            getenv("DS4_METAL_GRAPH_DUMP_PREFIX") == NULL;
        if (use_mm_id) {
            gate_map_args =
                ds4_gpu_make_mul_mm_id_map_args(expert_in_dim, n_total_expert, 1, n_expert, n_tokens);
            gate_mm_args =
                ds4_gpu_make_mul_mm_id_args(expert_in_dim, expert_mid_dim, n_total_expert,
                                              gate_row_bytes, gate_expert_bytes,
                                              1, n_expert, n_tokens);
            down_mm_args =
                ds4_gpu_make_mul_mm_id_args_src1_size(expert_mid_dim, out_dim, n_total_expert,
                                                        down_row_bytes, down_expert_bytes,
                                                        n_expert, n_expert, n_tokens,
                                                        request_mid_f16 ? sizeof(uint16_t) : sizeof(float));
            gate_mm_args.tp_rank = g_tp_split_rank;
            gate_mm_args.tp_world = g_tp_split_world;
            gate_mm_args.tp_expert_base = tp_expert_base_host;
            down_mm_args.tp_rank = g_tp_split_rank;
            down_mm_args.tp_world = g_tp_split_world;
            down_mm_args.tp_expert_base = tp_expert_base_host;

            map_pipeline = ds4_gpu_get_pipeline(ds4_gpu_mul_mm_id_map0_name(n_expert));
            gate_mm_pipeline = ds4_gpu_routed_mm_pipeline(gate_type);
            up_mm_pipeline = ds4_gpu_routed_mm_pipeline(gate_type);
            down_mm_pipeline = request_mid_f16 ?
                ds4_gpu_routed_mm_f16_rhs_pipeline(down_type) :
                ds4_gpu_routed_mm_pipeline(down_type);
            if (use_mm_id_pair_swiglu) {
                pair_swiglu_mm_pipeline =
                    ds4_gpu_get_pipeline(gate_type == DS4_METAL_TENSOR_Q4_K ?
                        "kernel_mul_mm_id_q4_K_pair_swiglu_f16" :
                        "kernel_mul_mm_id_iq2_xxs_pair_swiglu_f16");
            }
            if (!map_pipeline || !gate_mm_pipeline || !up_mm_pipeline || !down_mm_pipeline ||
                (use_mm_id_pair_swiglu && !pair_swiglu_mm_pipeline)) {
                return 0;
            }
        }

        if (use_iq2_batch_selected_addr) {
            const int had_batch = g_batch_cb != nil;
            if (had_batch && ds4_gpu_end_commands() == 0) {
                return 0;
            }
            g_stream_prefill_batch_selected_addr_building++;
            if (!ds4_gpu_stream_expert_cache_prepare_selected_batch(
                        model_map,
                        model_size,
                        layer_index,
                        selected,
                        n_tokens,
                        n_total_expert,
                        n_expert,
                        gate_offset,
                        up_offset,
                        down_offset,
                        gate_expert_bytes,
                        down_expert_bytes,
                        &stream_gate_addr_buf,
                        &stream_up_addr_buf,
                        &stream_down_addr_buf,
                        stream_resources,
                        &stream_resource_count,
                        &stream_unique,
                        &stream_overflow_gate,
                        &stream_overflow_up,
                        &stream_overflow_down)) {
                g_stream_prefill_batch_selected_addr_building--;
                return 0;
            }
            g_stream_prefill_batch_selected_addr_building--;
            if (stream_unique == 0) {
                ds4_gpu_stream_expert_cache_clear_layer(layer_index);
                fprintf(stderr,
                        "ds4: Metal streaming prefill batch selected addr layer=%u "
                        "produced no resident experts\n",
                        layer_index);
                return 0;
            }
            for (uint32_t i = 0; i < stream_resource_count; i++) {
                ds4_gpu_stream_expert_cache_entry *entry = stream_resources[i];
                if (!entry ||
                    !entry->valid ||
                    !entry->gate_buffer ||
                    !entry->up_buffer ||
                    !entry->down_buffer) {
                    fprintf(stderr,
                            "ds4: Metal streaming prefill batch selected addr layer=%u "
                            "lost resident expert resource %u/%u during preparation\n",
                            layer_index,
                            i,
                            stream_resource_count);
                    ds4_gpu_stream_expert_cache_clear_layer(layer_index);
                    return 0;
                }
            }
            if (had_batch && ds4_gpu_begin_commands() == 0) {
                fprintf(stderr,
                        "ds4: Metal streaming prefill batch selected addr layer=%u "
                        "failed to reopen command batch after preparation\n",
                        layer_index);
                ds4_gpu_stream_expert_cache_clear_layer(layer_index);
                return 0;
            }
        }

        if (use_q4_batch_expert_table) {
            gate_table = ds4_gpu_q4_expert_table(model_map,
                                                 model_size,
                                                 gate_offset,
                                                 gate_expert_bytes,
                                                 n_total_expert,
                                                 g_moe_table_q4_pair_gate_encoder);
            up_table = ds4_gpu_q4_expert_table(model_map,
                                               model_size,
                                               up_offset,
                                               gate_expert_bytes,
                                               n_total_expert,
                                               g_moe_table_q4_pair_up_encoder);
            down_table = ds4_gpu_q4_expert_table(model_map,
                                                 model_size,
                                                 down_offset,
                                                 down_expert_bytes,
                                                 n_total_expert,
                                                 g_moe_table_q4_sum_down_encoder);
            if (!gate_table || !up_table || !down_table) {
                return 0;
            }
            q4_table_layer_residency =
                ds4_gpu_q4_expert_layer_residency_set(gate_table,
                                                       up_table,
                                                       down_table,
                                                       q4_batch_expert_table_auto);
            if ((q4_batch_table_queue_residency ||
                 getenv("DS4_METAL_Q4_TABLE_RESIDENCY_SET") != NULL) &&
                !q4_table_layer_residency &&
                !ds4_gpu_q4_table_model_residency_enabled()) {
                fprintf(stderr, "ds4: Metal Q4 batch expert table residency set is not available\n");
                return 0;
            }
        } else if (use_iq2_batch_selected_addr) {
            if (n_expert > 1 && (!expertsbuf ||
                ds4_gpu_tensor_bytes(experts) < down_scratch_bytes)) {
                if (!ds4_gpu_ensure_scratch_buffer(&g_moe_down_scratch_buffer,
                                                   &g_moe_down_scratch_bytes,
                                                   (NSUInteger)down_scratch_bytes,
                                                   "ds4_moe_down_scratch")) {
                    ds4_gpu_stream_expert_cache_clear_layer(layer_index);
                    return 0;
                }
            }
        } else {
            if (n_expert > 1 && (!expertsbuf ||
                ds4_gpu_tensor_bytes(experts) < down_scratch_bytes)) {
                if (!ds4_gpu_ensure_scratch_buffer(&g_moe_down_scratch_buffer,
                                                   &g_moe_down_scratch_bytes,
                                                   (NSUInteger)down_scratch_bytes,
                                                   "ds4_moe_down_scratch")) {
                    return 0;
                }
            }
            gate_buf = ds4_gpu_wrap_model_range(model_map,
                                                model_size,
                                                gate_offset,
                                                gate_tensor_bytes,
                                                &gate_inner);
            up_buf = ds4_gpu_wrap_model_range(model_map,
                                              model_size,
                                              up_offset,
                                              gate_tensor_bytes,
                                              &up_inner);
            down_buf = ds4_gpu_wrap_model_range(model_map,
                                                model_size,
                                                down_offset,
                                                down_tensor_bytes,
                                                &down_inner);
            if (!gate_buf || !up_buf || !down_buf) return 0;
            if (getenv("DS4_GLM_TP_DEBUG") && layer_index == 3) {
                fprintf(stderr,
                        "ds4: batch mv binds l=%u rank=%d base=%d gate=%llu+%llu "
                        "up=%llu down=%llu inner=%llu/%llu/%llu ne02=%d nei0=%d nr0=%d\n",
                        layer_index, g_tp_split_rank, tp_expert_base_host,
                        (unsigned long long)gate_offset,
                        (unsigned long long)gate_tensor_bytes,
                        (unsigned long long)up_offset,
                        (unsigned long long)down_offset,
                        (unsigned long long)gate_inner,
                        (unsigned long long)up_inner,
                        (unsigned long long)down_inner,
                        gate_args.ne02, gate_args.nei0, gate_args.nr0);
            }
        }

        const bool q4_batch_table_boundary =
            use_q4_batch_expert_table &&
            g_batch_cb != nil &&
            getenv("DS4_METAL_Q4_TABLE_RESIDENCY_SET") != NULL &&
            !q4_batch_table_queue_residency &&
            getenv("DS4_METAL_DISABLE_Q4_TABLE_BOUNDARY") == NULL;
        if (q4_batch_table_boundary) {
            if (ds4_gpu_end_commands() == 0 || ds4_gpu_begin_commands() == 0) {
                return 0;
            }
        }

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;
        if (use_q4_batch_expert_table && !ds4_gpu_use_model_residency_set(cb)) {
            return 0;
        }
        if (!q4_batch_table_queue_residency &&
            q4_table_layer_residency &&
            [cb respondsToSelector:@selector(useResidencySet:)]) {
            [cb useResidencySet:q4_table_layer_residency];
        }
        const bool moe_stage_profile =
            g_batch_cb != nil &&
            ds4_gpu_stage_profile_enabled_for_layer("DS4_METAL_MOE_STAGE_PROFILE",
                                                    "DS4_METAL_MOE_STAGE_PROFILE_LAYER",
                                                    layer_index);
        const char *moe_stage_filter = getenv("DS4_METAL_MOE_STAGE_PROFILE_FILTER");
        const char *moe_path =
            use_q4_batch_expert_table ? "q4_table_pair_swiglu" :
            use_iq2_batch_selected_addr ? "iq2_batch_stream_addr" :
            use_mm_id_pair_swiglu ? "mm_id_pair_swiglu" :
            use_mm_id ? "mm_id" :
            use_tiny_pair_swiglu ? "tiny_pair_swiglu" :
            (use_tiny_pair_mv ? "tiny_pair_mv" : "mv");
        double moe_stage_t0 = moe_stage_profile ? ds4_gpu_now_ms() : 0.0;
        if (moe_stage_profile) {
            if (ds4_gpu_end_commands() == 0 || ds4_gpu_begin_commands() == 0) {
                return 0;
            }
            cb = ds4_gpu_command_buffer(&owned);
            if (!cb) return 0;
            moe_stage_t0 = ds4_gpu_now_ms();
        }
#define DS4_METAL_PROFILE_MOE_STAGE(name) do { \
            if (ok && moe_stage_profile) { \
                if (ds4_gpu_end_commands() == 0) { \
                    ok = 0; \
                } else { \
                    const char *stage_name = (name); \
                    const double now_ms = ds4_gpu_now_ms(); \
                    const int print_stage = \
                        !moe_stage_filter || !moe_stage_filter[0] || \
                        strstr(stage_name, moe_stage_filter) != NULL; \
                    if (print_stage) { \
                        fprintf(stderr, \
                                "ds4: Metal routed MoE stage layer=%u tokens=%u pairs=%u experts=%u " \
                                "gate=%s down=%s path=%s mid=%s %s=%.3f ms\n", \
                                layer_index, n_tokens, pair_rows, n_expert, \
                                ds4_gpu_metal_tensor_type_name(gate_type), \
                                ds4_gpu_metal_tensor_type_name(down_type), \
                                moe_path, \
                                request_mid_f16 ? "f16" : "f32", \
                                stage_name, now_ms - moe_stage_t0); \
                    } \
                    moe_stage_t0 = now_ms; \
                    if (ds4_gpu_begin_commands() == 0) { \
                        ok = 0; \
                    } else { \
                        cb = ds4_gpu_command_buffer(&owned); \
                        if (!cb) ok = 0; \
                    } \
                } \
            } \
        } while (0)

        const NSUInteger gate_smem = ds4_gpu_routed_mv_smem(gate_type);
        const NSUInteger down_smem = ds4_gpu_routed_mv_smem(down_type);
        const NSUInteger gate_nsg = ds4_gpu_routed_mv_nsg(gate_type);
        const NSUInteger down_nsg = ds4_gpu_routed_mv_nsg(down_type);
        const bool gate_rows_per_group_is_nr0 = ds4_gpu_routed_mv_rows_per_group_is_nr0(gate_type);
        const bool down_rows_per_group_is_nr0 = ds4_gpu_routed_mv_rows_per_group_is_nr0(down_type);
        id<MTLComputePipelineState> down_sum6_pipeline = nil;
        if (down_type == DS4_METAL_TENSOR_Q2_K) {
            down_sum6_pipeline = g_moe_mul_mv_id_q2_k_sum6_pipeline;
        } else if (down_type == DS4_METAL_TENSOR_Q4_K) {
            down_sum6_pipeline = g_moe_mul_mv_id_q4_k_sum6_pipeline;
        }
        const bool direct_down_sum =
            !g_quality_mode &&
            !use_q4_batch_expert_table &&
            !use_mm_id &&
            n_expert == 6 &&
            n_tokens <= 4u &&
            down_sum6_pipeline != nil;
        int ok = 0;
        if (use_iq2_batch_selected_addr) {
            ds4_gpu_dsv4_moe_swiglu_weight_args act_args = {
                .width = expert_mid_dim,
                .rows = pair_rows,
                .gate_row_stride = (uint64_t)expert_mid_dim * sizeof(float),
                .up_row_stride = (uint64_t)expert_mid_dim * sizeof(float),
                .mid_row_stride = (uint64_t)expert_mid_dim * sizeof(float),
                .weight_stride = sizeof(float),
                .write_clamped = 0,
                .clamp_value = clamp,
            };
            ok = ds4_gpu_encode_mul_mv_addr_iq2_pair_swiglu(
                    cb,
                    g_moe_mul_mv_addr_iq2_xxs_pair_swiglu_pipeline,
                    &gate_args,
                    &act_args,
                    stream_resources,
                    stream_resource_count,
                    stream_gate_addr_buf,
                    stream_up_addr_buf,
                    xbuf,
                    ds4_gpu_tensor_offset(x),
                    gatebuf,
                    ds4_gpu_tensor_offset(gate),
                    upbuf,
                    ds4_gpu_tensor_offset(up),
                    midbuf,
                    ds4_gpu_tensor_offset(mid),
                    selectedbuf,
                    ds4_gpu_tensor_offset(selected),
                    weightsbuf,
                    ds4_gpu_tensor_offset(weights),
                    gate_smem,
                    2,
                    false,
                    stream_overflow_gate,
                    stream_overflow_up);
            if (!ok) {
                fprintf(stderr,
                        "ds4: Metal streaming prefill batch selected addr layer=%u "
                        "failed to encode gate/up path tokens=%u unique=%u\n",
                        layer_index,
                        n_tokens,
                        stream_unique);
            }
        } else if (use_q4_batch_expert_table) {
            ds4_gpu_dsv4_moe_swiglu_weight_args act_args = {
                .width = expert_mid_dim,
                .rows = pair_rows,
                .gate_row_stride = (uint64_t)expert_mid_dim * sizeof(float),
                .up_row_stride = (uint64_t)expert_mid_dim * sizeof(float),
                .mid_row_stride = (uint64_t)expert_mid_dim * sizeof(float),
                .weight_stride = sizeof(float),
                .write_clamped = 0,
                .clamp_value = clamp,
            };
            ok = ds4_gpu_encode_mul_mv_table_q4_pair_swiglu(cb,
                                                            g_moe_mul_mv_table_q4_k_pair_swiglu_pipeline,
                                                            &gate_args,
                                                            &act_args,
                                                            gate_table,
                                                            up_table,
                                                            xbuf,
                                                            ds4_gpu_tensor_offset(x),
                                                            gatebuf,
                                                            ds4_gpu_tensor_offset(gate),
                                                            upbuf,
                                                            ds4_gpu_tensor_offset(up),
                                                            midbuf,
                                                            ds4_gpu_tensor_offset(mid),
                                                            selectedbuf,
                                                            ds4_gpu_tensor_offset(selected),
                                                            weightsbuf,
                                                            ds4_gpu_tensor_offset(weights),
                                                            gate_smem,
                                                            2,
                                                            false,
                                                            q4_batch_table_queue_residency);
        } else if (use_mm_id) {
            /*
             * The routed pair ids are the same for gate, up, and down. Build
             * the expert-major work map once, then reuse it for all three
             * batched expert matmuls.
             */
            ok = ds4_gpu_encode_mul_mm_id_map(cb,
                                                map_pipeline,
                                                &gate_map_args,
                                                &gate_mm_args,
                                                selectedbuf,
                                                ds4_gpu_tensor_offset(selected));
            DS4_METAL_PROFILE_MOE_STAGE("map");
            if (ok && use_mm_id_pair_swiglu) {
                ds4_gpu_dsv4_moe_swiglu_weight_args act_args = {
                    .width = expert_mid_dim,
                    .rows = pair_rows,
                    .gate_row_stride = (uint64_t)expert_mid_dim * sizeof(float),
                    .up_row_stride = (uint64_t)expert_mid_dim * sizeof(float),
                    .mid_row_stride = (uint64_t)expert_mid_dim * sizeof(uint16_t),
                    .weight_stride = sizeof(float),
                    .write_clamped = 0,
                    .clamp_value = clamp,
                };
                ok = ds4_gpu_encode_mul_mm_id_iq2_pair_swiglu_f16(cb,
                                                                   pair_swiglu_mm_pipeline,
                                                                   &gate_mm_args,
                                                                   &act_args,
                                                                   gate_buf,
                                                                   (NSUInteger)gate_inner,
                                                                   up_buf,
                                                                   (NSUInteger)up_inner,
                                                                   xbuf,
                                                                   ds4_gpu_tensor_offset(x),
                                                                   midbuf,
                                                                   ds4_gpu_tensor_offset(mid),
                                                                   weightsbuf,
                                                                   ds4_gpu_tensor_offset(weights));
                DS4_METAL_PROFILE_MOE_STAGE("gate_up_fused");
            } else if (ok) {
                ok = ds4_gpu_encode_mul_mm_id_mapped_tile(cb,
                                                           gate_mm_pipeline,
                                                           &gate_mm_args,
                                                           gate_buf,
                                                           (NSUInteger)gate_inner,
                                                           xbuf,
                                                           ds4_gpu_tensor_offset(x),
                                                           gatebuf,
                                                           ds4_gpu_tensor_offset(gate),
                                                           8192u);
                DS4_METAL_PROFILE_MOE_STAGE("gate");
            }
            if (ok && !use_mm_id_pair_swiglu) {
                ok = ds4_gpu_encode_mul_mm_id_mapped_tile(cb,
                                                   up_mm_pipeline,
                                                   &gate_mm_args,
                                                   up_buf,
                                                   (NSUInteger)up_inner,
                                                   xbuf,
                                                   ds4_gpu_tensor_offset(x),
                                                   upbuf,
                                                   ds4_gpu_tensor_offset(up),
                                                   8192u);
                DS4_METAL_PROFILE_MOE_STAGE("up");
            }
        } else if (use_tiny_pair_swiglu) {
            ds4_gpu_dsv4_moe_swiglu_weight_args act_args = {
                .width = expert_mid_dim,
                .rows = pair_rows,
                .gate_row_stride = (uint64_t)expert_mid_dim * sizeof(float),
                .up_row_stride = (uint64_t)expert_mid_dim * sizeof(float),
                .mid_row_stride = (uint64_t)expert_mid_dim * sizeof(float),
                .weight_stride = sizeof(float),
                .write_clamped = 0,
                .clamp_value = clamp,
            };
            ok = ds4_gpu_encode_mul_mv_id_pair_swiglu(cb,
                                                       tiny_pair_swiglu_pipeline,
                                                       &gate_args,
                                                       &act_args,
                                                       gate_buf,
                                                       (NSUInteger)gate_inner,
                                                       up_buf,
                                                       (NSUInteger)up_inner,
                                                       xbuf,
                                                       ds4_gpu_tensor_offset(x),
                                                       gatebuf,
                                                       ds4_gpu_tensor_offset(gate),
                                                       upbuf,
                                                       ds4_gpu_tensor_offset(up),
                                                       midbuf,
                                                       ds4_gpu_tensor_offset(mid),
                                                       selectedbuf,
                                                       ds4_gpu_tensor_offset(selected),
                                                       weightsbuf,
                                                       ds4_gpu_tensor_offset(weights),
                                                       gate_smem,
                                                       2,
                                                       false);
        } else if (use_tiny_pair_mv) {
            id<MTLComputePipelineState> pair_pipeline =
                gate_type == DS4_METAL_TENSOR_IQ2_XXS ?
                    g_moe_mul_mv_id_iq2_xxs_pair_pipeline :
                    g_moe_mul_mv_id_q4_k_pair_pipeline;
            ok = ds4_gpu_encode_mul_mv_id_pair(cb,
                                                 pair_pipeline,
                                                 &gate_args,
                                                 gate_buf,
                                                 (NSUInteger)gate_inner,
                                                 up_buf,
                                                 (NSUInteger)up_inner,
                                                 xbuf,
                                                 ds4_gpu_tensor_offset(x),
                                                 gatebuf,
                                                 ds4_gpu_tensor_offset(gate),
                                                 upbuf,
                                                 ds4_gpu_tensor_offset(up),
                                                 selectedbuf,
                                                 ds4_gpu_tensor_offset(selected),
                                                 gate_smem,
                                                 2,
                                                 false);
        } else {
            ok = ds4_gpu_encode_mul_mv_id(cb,
                                                  gate_mv_pipeline,
                                                  &gate_args,
                                                  gate_buf,
                                                  (NSUInteger)gate_inner,
                                                  xbuf,
                                                  ds4_gpu_tensor_offset(x),
                                                  gatebuf,
                                                  ds4_gpu_tensor_offset(gate),
                                                  selectedbuf,
                                                  ds4_gpu_tensor_offset(selected),
                                                  gate_smem,
                                                  gate_nsg,
                                                  gate_rows_per_group_is_nr0) &&
                 ds4_gpu_encode_mul_mv_id(cb,
                                                  gate_mv_pipeline,
                                                  &gate_args,
                                                  up_buf,
                                                  (NSUInteger)up_inner,
                                                  xbuf,
                                                  ds4_gpu_tensor_offset(x),
                                                  upbuf,
                                                  ds4_gpu_tensor_offset(up),
                                                  selectedbuf,
                                                  ds4_gpu_tensor_offset(selected),
                                                  gate_smem,
                                                  gate_nsg,
                                                  gate_rows_per_group_is_nr0);
        }
        DS4_METAL_PROFILE_MOE_STAGE("gate_up");
        const bool use_fused_activation = !g_quality_mode && !use_q4_batch_expert_table;
        const bool use_mid_f16 =
            use_mm_id &&
            use_fused_activation &&
            request_mid_f16;
        if (mid_is_f16) *mid_is_f16 = use_mid_f16;
        if (ok && use_iq2_batch_selected_addr) {
            /* The address-table pair kernel already wrote weighted SwiGLU rows into mid. */
        } else if (ok && use_q4_batch_expert_table) {
            /* The table pair kernel already wrote weighted SwiGLU rows into mid. */
        } else if (ok && use_mm_id_pair_swiglu) {
            /* The fused batch mm_id pair kernel already wrote weighted f16 SwiGLU rows into mid. */
        } else if (ok && use_tiny_pair_swiglu) {
            /* The fused tiny pair kernel already wrote weighted F32 SwiGLU rows into mid. */
        } else if (ok && use_fused_activation) {
            ok = ds4_gpu_encode_moe_swiglu_weight(cb,
                                                    gatebuf,
                                                    ds4_gpu_tensor_offset(gate),
                                                    upbuf,
                                                    ds4_gpu_tensor_offset(up),
                                                    midbuf,
                                                    ds4_gpu_tensor_offset(mid),
                                                    weightsbuf,
                                                    ds4_gpu_tensor_offset(weights),
                                                    expert_mid_dim,
                                                    pair_rows,
                                                    clamp,
                                                    use_mid_f16);
        } else if (ok && clamp > 1.0e-6f) {
            ok = ds4_gpu_encode_unary_f32_rows(cb,
                                                 g_unary_clamp_pipeline,
                                                 gatebuf,
                                                 ds4_gpu_tensor_offset(gate),
                                                 gatebuf,
                                                 ds4_gpu_tensor_offset(gate),
                                                 expert_mid_dim,
                                                 pair_rows,
                                                 0,
                                                 -FLT_MAX,
                                                 clamp);
            if (ok) {
                ok = ds4_gpu_encode_unary_f32_rows(cb,
                                                     g_unary_silu_pipeline,
                                                     gatebuf,
                                                     ds4_gpu_tensor_offset(gate),
                                                     midbuf,
                                                     ds4_gpu_tensor_offset(mid),
                                                     expert_mid_dim,
                                                     pair_rows,
                                                     1,
                                                     0.0f,
                                                     0.0f);
            }
            if (ok) {
                ok = ds4_gpu_encode_unary_f32_rows(cb,
                                                 g_unary_clamp_pipeline,
                                                 upbuf,
                                                 ds4_gpu_tensor_offset(up),
                                                 upbuf,
                                                 ds4_gpu_tensor_offset(up),
                                                 expert_mid_dim,
                                                 pair_rows,
                                                 0,
                                                 -clamp,
                                                 clamp);
            }
            if (ok) {
                ds4_gpu_bin_args mul_args =
                    ds4_gpu_make_bin_same_rows_args(expert_mid_dim, pair_rows);
                ok = ds4_gpu_encode_bin_f32_rows(cb,
                                                   g_mul_pipeline,
                                                   &mul_args,
                                                   midbuf,
                                                   ds4_gpu_tensor_offset(mid),
                                                   upbuf,
                                                   ds4_gpu_tensor_offset(up),
                                                   midbuf,
                                                   ds4_gpu_tensor_offset(mid));
            }
        } else if (ok) {
            ok = ds4_gpu_encode_swiglu_flat(cb,
                                              gatebuf,
                                              ds4_gpu_tensor_offset(gate),
                                              upbuf,
                                              ds4_gpu_tensor_offset(up),
                                              midbuf,
                                              ds4_gpu_tensor_offset(mid),
                                              (uint32_t)((uint64_t)pair_rows * expert_mid_dim));
        }
        if (ok && !use_fused_activation && !use_q4_batch_expert_table) {
            ds4_gpu_bin_args weight_args =
                ds4_gpu_make_bin_rowwise_scalar_args(expert_mid_dim, pair_rows);
            ok = ds4_gpu_encode_bin_f32_rows(cb,
                                               g_bin_mul_scalar_pipeline,
                                               &weight_args,
                                               midbuf,
                                               ds4_gpu_tensor_offset(mid),
                                               weightsbuf,
                                               ds4_gpu_tensor_offset(weights),
                                               midbuf,
                                               ds4_gpu_tensor_offset(mid));
        }
        DS4_METAL_PROFILE_MOE_STAGE("activation_weight");

        id<MTLBuffer> down_dst = n_expert == 1 ? outbuf : (expertsbuf ? expertsbuf : g_moe_down_scratch_buffer);
        NSUInteger down_dst_off = n_expert == 1 ? ds4_gpu_tensor_offset(out) :
            (expertsbuf ? ds4_gpu_tensor_offset(experts) : 0);
        if (ok) {
            if (use_iq2_batch_selected_addr) {
                ok = ds4_gpu_encode_mul_mv_addr_q2_sum6(
                        cb,
                        g_moe_mul_mv_addr_q2_k_sum6_pipeline,
                        &down_args,
                        stream_resources,
                        stream_resource_count,
                        stream_down_addr_buf,
                        midbuf,
                        ds4_gpu_tensor_offset(mid),
                        outbuf,
                        ds4_gpu_tensor_offset(out),
                        selectedbuf,
                        ds4_gpu_tensor_offset(selected),
                        down_smem,
                        2,
                        stream_overflow_down);
                if (!ok) {
                    fprintf(stderr,
                            "ds4: Metal streaming prefill batch selected addr layer=%u "
                            "failed to encode down path tokens=%u unique=%u\n",
                            layer_index,
                            n_tokens,
                            stream_unique);
                }
            } else if (use_q4_batch_expert_table) {
                ok = ds4_gpu_encode_mul_mv_table_q4_sum6(cb,
                                                         g_moe_mul_mv_table_q4_k_sum6_pipeline,
                                                         &down_args,
                                                         down_table,
                                                         midbuf,
                                                         ds4_gpu_tensor_offset(mid),
                                                         outbuf,
                                                         ds4_gpu_tensor_offset(out),
                                                         selectedbuf,
                                                         ds4_gpu_tensor_offset(selected),
                                                         down_smem,
                                                         2,
                                                         q4_batch_table_queue_residency);
            } else if (direct_down_sum) {
                ok = ds4_gpu_encode_mul_mv_id_sum6(cb,
                                                     down_sum6_pipeline,
                                                     &down_args,
                                                     down_buf,
                                                     (NSUInteger)down_inner,
                                                     midbuf,
                                                     ds4_gpu_tensor_offset(mid),
                                                     outbuf,
                                                     ds4_gpu_tensor_offset(out),
                                                     selectedbuf,
                                                     ds4_gpu_tensor_offset(selected),
                                                     nil,
                                                     0,
                                                     down_smem,
                                                     2);
            } else if (use_mm_id) {
                ok = ds4_gpu_encode_mul_mm_id_mapped_tile(cb,
                                                       down_mm_pipeline,
                                                       &down_mm_args,
                                                       down_buf,
                                                       (NSUInteger)down_inner,
                                                       midbuf,
                                                       ds4_gpu_tensor_offset(mid),
                                                       down_dst,
                                                       down_dst_off,
                                                       8192u);
            } else {
                ok = ds4_gpu_encode_mul_mv_id(cb,
                                                     down_mv_pipeline,
                                                     &down_args,
                                                     down_buf,
                                                     (NSUInteger)down_inner,
                                                     midbuf,
                                                     ds4_gpu_tensor_offset(mid),
                                                     down_dst,
                                                     down_dst_off,
                                                     selectedbuf,
                                                     ds4_gpu_tensor_offset(selected),
                                                     down_smem,
                                                     down_nsg,
                                                     down_rows_per_group_is_nr0);
            }
        }
        DS4_METAL_PROFILE_MOE_STAGE("down");
        if (ok &&
            n_expert > 1 &&
            !direct_down_sum &&
            !use_q4_batch_expert_table &&
            !use_iq2_batch_selected_addr) {
            ok = ds4_gpu_encode_moe_sum_experts(cb,
                                                       down_dst,
                                                       down_dst_off,
                                                       outbuf,
                                                       ds4_gpu_tensor_offset(out),
                                                       out_dim,
                                                       n_expert,
                                                       n_tokens);
        }
        DS4_METAL_PROFILE_MOE_STAGE("sum");
        if (!ok) {
            fprintf(stderr,
                    "ds4: Metal routed batch MoE failed before submit layer=%u tokens=%u "
                    "gate=%s down=%s path=%s\n",
                    layer_index,
                    n_tokens,
                    ds4_gpu_metal_tensor_type_name(gate_type),
                    ds4_gpu_metal_tensor_type_name(down_type),
                    moe_path);
            return 0;
        }

        if (!ds4_gpu_finish_command_buffer(cb, owned, "routed batch MoE")) {
            if (use_iq2_batch_selected_addr) {
                ds4_gpu_stream_expert_cache_clear_layer(layer_index);
            }
            return 0;
        }
        if (use_iq2_batch_selected_addr) {
            if (!owned) {
                if (ds4_gpu_end_commands() == 0) {
                    ds4_gpu_stream_expert_cache_clear_layer(layer_index);
                    return 0;
                }
                if (ds4_gpu_begin_commands() == 0) {
                    fprintf(stderr,
                            "ds4: Metal streaming prefill batch selected addr layer=%u "
                            "failed to reopen command batch after execution\n",
                            layer_index);
                    return 0;
                }
            }
        }
        if (q4_batch_table_boundary) {
            if (ds4_gpu_end_commands() == 0 || ds4_gpu_begin_commands() == 0) {
                return 0;
            }
        }
#undef DS4_METAL_PROFILE_MOE_STAGE
    }

    return 1;
}
