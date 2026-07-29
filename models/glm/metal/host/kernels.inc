int ds4_gpu_glm_kv_lora_rms_norm_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *kv_raw,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint32_t              n_tokens,
        uint32_t              kv_raw_dim,
        uint32_t              kv_lora_dim,
        float                 eps) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!out || !kv_raw || !model_map ||
        n_tokens == 0 || kv_raw_dim == 0 || kv_lora_dim == 0 ||
        kv_lora_dim > kv_raw_dim || (kv_lora_dim & 3u) != 0 ||
        !isfinite(eps) || eps < 0.0f) {
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> rawbuf = ds4_gpu_tensor_buffer(kv_raw);
        id<MTLBuffer> outbuf = ds4_gpu_tensor_buffer(out);
        const uint64_t raw_bytes = (uint64_t)n_tokens * kv_raw_dim * sizeof(float);
        const uint64_t out_bytes = (uint64_t)n_tokens * kv_lora_dim * sizeof(float);
        const uint64_t weight_bytes = (uint64_t)kv_lora_dim * sizeof(float);
        if (!rawbuf || !outbuf ||
            ds4_gpu_tensor_bytes(kv_raw) < raw_bytes ||
            ds4_gpu_tensor_bytes(out) < out_bytes) {
            fprintf(stderr, "ds4: Metal GLM KV RMS norm received undersized buffers\n");
            return 0;
        }
        if (weight_offset > model_size || weight_bytes > model_size - weight_offset) {
            fprintf(stderr, "ds4: Metal GLM KV RMS norm weight range is outside the mapped model\n");
            return 0;
        }

        const bool exact_decode_weight_view =
            n_tokens == 1u &&
            weight_bytes <= (1ull << 20) &&
            getenv("DS4_METAL_DISABLE_DECODE_NORM_EXACT_VIEWS") == NULL;
        uint64_t weight_inner = 0;
        id<MTLBuffer> weightbuf = exact_decode_weight_view ?
            ds4_gpu_wrap_model_exact_range(model_map,
                                           model_size,
                                           weight_offset,
                                           weight_bytes,
                                           &weight_inner) :
            ds4_gpu_wrap_model_range(model_map,
                                     model_size,
                                     weight_offset,
                                     weight_bytes,
                                     &weight_inner);
        if (!weightbuf) return 0;

        id<MTLComputePipelineState> pipeline =
            ds4_gpu_hot_pipeline(g_glm_kv_lora_rms_norm_pipeline,
                                 "kernel_glm_kv_lora_rms_norm");
        if (!pipeline) return 0;

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        ds4_gpu_glm_kv_lora_rms_norm_args args = {
            .n_tokens = n_tokens,
            .kv_raw_dim = kv_raw_dim,
            .kv_lora_dim = kv_lora_dim,
            .eps = eps,
        };
        const NSUInteger nth = ds4_gpu_rms_norm_threads(kv_lora_dim);

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:rawbuf offset:ds4_gpu_tensor_offset(kv_raw) atIndex:1];
        [enc setBuffer:weightbuf offset:(NSUInteger)weight_inner atIndex:2];
        [enc setBuffer:outbuf offset:ds4_gpu_tensor_offset(out) atIndex:3];
        [enc setThreadgroupMemoryLength:nth * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)n_tokens, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "GLM KV RMS norm")) return 0;
    }

    return 1;
}

int ds4_gpu_glm_k_b_project_typed_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *kv_norm,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint32_t              weight_type,
        uint32_t              n_tokens,
        uint32_t              kv_lora_dim,
        uint32_t              qk_nope,
        uint32_t              n_head) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!out || !kv_norm || !model_map ||
        n_tokens == 0 || kv_lora_dim == 0 ||
        qk_nope == 0 || n_head == 0) {
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> outbuf = ds4_gpu_tensor_buffer(out);
        id<MTLBuffer> kvbuf = ds4_gpu_tensor_buffer(kv_norm);
        uint64_t row_bytes = 0;
        if (!ds4_gpu_quant_row_bytes(weight_type, qk_nope, &row_bytes)) {
            fprintf(stderr, "ds4: Metal GLM k_b projection received unsupported weight type\n");
            return 0;
        }
        const uint64_t weight_rows = (uint64_t)n_head * kv_lora_dim;
        const uint64_t weight_bytes = weight_rows * row_bytes;
        const uint64_t kv_bytes = (uint64_t)n_tokens * kv_lora_dim * sizeof(float);
        const uint64_t out_bytes = (uint64_t)n_tokens * n_head * qk_nope * sizeof(float);
        if (!outbuf || !kvbuf ||
            ds4_gpu_tensor_bytes(kv_norm) < kv_bytes ||
            ds4_gpu_tensor_bytes(out) < out_bytes) {
            fprintf(stderr, "ds4: Metal GLM k_b projection received undersized buffers\n");
            return 0;
        }
        if (weight_offset > model_size || weight_bytes > model_size - weight_offset) {
            fprintf(stderr, "ds4: Metal GLM k_b projection range is outside the mapped model\n");
            return 0;
        }
        const NSUInteger q_blocks = ((NSUInteger)qk_nope + 31u) / 32u;
        if (q_blocks > 8u) {
            fprintf(stderr, "ds4: Metal GLM k_b projection q width is too large for the tiled kernel\n");
            return 0;
        }

        uint64_t weight_inner = 0;
        id<MTLBuffer> weightbuf =
            ds4_gpu_wrap_model_range(model_map,
                                      model_size,
                                      weight_offset,
                                      weight_bytes,
                                      &weight_inner);
        if (!weightbuf) return 0;

        id<MTLComputePipelineState> pipeline =
            ds4_gpu_hot_pipeline(g_glm_k_b_project_pipeline,
                                 "kernel_glm_k_b_project_q8_0");
        if (!pipeline) return 0;

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        ds4_gpu_glm_k_b_project_args args = {
            .n_tokens = n_tokens,
            .kv_lora_dim = kv_lora_dim,
            .qk_nope = qk_nope,
            .n_head = n_head,
            .row_bytes = (uint32_t)row_bytes,
            .weight_type = weight_type,
            .pad1 = 0,
            .pad2 = 0,
        };

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:weightbuf offset:(NSUInteger)weight_inner atIndex:1];
        [enc setBuffer:kvbuf offset:ds4_gpu_tensor_offset(kv_norm) atIndex:2];
        [enc setBuffer:outbuf offset:ds4_gpu_tensor_offset(out) atIndex:3];
        [enc setThreadgroupMemoryLength:(NSUInteger)kv_lora_dim * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)n_tokens, (NSUInteger)n_head, 1)
             threadsPerThreadgroup:MTLSizeMake(32, q_blocks, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "GLM k_b projection")) return 0;
    }

    return 1;
}

int ds4_gpu_glm_k_b_project_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *kv_norm,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint32_t              n_tokens,
        uint32_t              kv_lora_dim,
        uint32_t              qk_nope,
        uint32_t              n_head) {
    return ds4_gpu_glm_k_b_project_typed_tensor(out,
                                                kv_norm,
                                                model_map,
                                                model_size,
                                                weight_offset,
                                                DS4_METAL_TENSOR_Q8_0,
                                                n_tokens,
                                                kv_lora_dim,
                                                qk_nope,
                                                n_head);
}

int ds4_gpu_glm_store_compact_kv_tensor(
        ds4_gpu_tensor       *kv_lora_cache,
        ds4_gpu_tensor       *k_rope_cache,
        const ds4_gpu_tensor *kv_norm,
        const ds4_gpu_tensor *kv_raw,
        uint32_t              pos0,
        uint32_t              n_tokens,
        uint32_t              cache_cap,
        uint32_t              kv_raw_dim,
        uint32_t              kv_lora_dim,
        uint32_t              qk_rope,
        bool                  cache_f16) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!kv_lora_cache || !k_rope_cache || !kv_norm || !kv_raw ||
        n_tokens == 0 || cache_cap == 0 ||
        kv_raw_dim == 0 || kv_lora_dim == 0 || qk_rope == 0 ||
        kv_lora_dim > kv_raw_dim ||
        qk_rope > kv_raw_dim - kv_lora_dim ||
        pos0 > cache_cap || n_tokens > cache_cap - pos0) {
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> kvcachebuf = ds4_gpu_tensor_buffer(kv_lora_cache);
        id<MTLBuffer> ropecachebuf = ds4_gpu_tensor_buffer(k_rope_cache);
        id<MTLBuffer> kvnormbuf = ds4_gpu_tensor_buffer(kv_norm);
        id<MTLBuffer> kvrawbuf = ds4_gpu_tensor_buffer(kv_raw);
        const uint64_t cache_elem_bytes = cache_f16 ? sizeof(uint16_t) : sizeof(float);
        const uint64_t kv_cache_bytes =
            (uint64_t)cache_cap * kv_lora_dim * cache_elem_bytes;
        const uint64_t rope_cache_bytes =
            (uint64_t)cache_cap * qk_rope * cache_elem_bytes;
        const uint64_t kv_norm_bytes =
            (uint64_t)n_tokens * kv_lora_dim * sizeof(float);
        const uint64_t kv_raw_bytes =
            (uint64_t)n_tokens * kv_raw_dim * sizeof(float);
        if (!kvcachebuf || !ropecachebuf || !kvnormbuf || !kvrawbuf ||
            ds4_gpu_tensor_bytes(kv_lora_cache) < kv_cache_bytes ||
            ds4_gpu_tensor_bytes(k_rope_cache) < rope_cache_bytes ||
            ds4_gpu_tensor_bytes(kv_norm) < kv_norm_bytes ||
            ds4_gpu_tensor_bytes(kv_raw) < kv_raw_bytes) {
            fprintf(stderr, "ds4: Metal GLM compact KV store received undersized buffers\n");
            return 0;
        }

        id<MTLComputePipelineState> pipeline =
            ds4_gpu_hot_pipeline(g_glm_store_compact_kv_pipeline,
                                 "kernel_glm_store_compact_kv");
        if (!pipeline) return 0;

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        ds4_gpu_glm_store_compact_kv_args args = {
            .pos0 = pos0,
            .n_tokens = n_tokens,
            .cache_cap = cache_cap,
            .kv_raw_dim = kv_raw_dim,
            .kv_lora_dim = kv_lora_dim,
            .qk_rope = qk_rope,
            .cache_f16 = cache_f16 ? 1u : 0u,
            .pad1 = 0,
        };

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:kvnormbuf offset:ds4_gpu_tensor_offset(kv_norm) atIndex:1];
        [enc setBuffer:kvrawbuf offset:ds4_gpu_tensor_offset(kv_raw) atIndex:2];
        [enc setBuffer:kvcachebuf offset:ds4_gpu_tensor_offset(kv_lora_cache) atIndex:3];
        [enc setBuffer:ropecachebuf offset:ds4_gpu_tensor_offset(k_rope_cache) atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)n_tokens, 2, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "GLM compact KV store")) return 0;
    }

    return 1;
}

int ds4_gpu_glm_qkv_norm_store_compact_kv_tensor(
        ds4_gpu_tensor       *q_out,
        const ds4_gpu_tensor *q,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              q_weight_offset,
        uint32_t              q_n,
        ds4_gpu_tensor       *kv_lora_cache,
        ds4_gpu_tensor       *k_rope_cache,
        const ds4_gpu_tensor *kv_raw,
        uint64_t              kv_weight_offset,
        uint32_t              pos0,
        uint32_t              n_tokens,
        uint32_t              cache_cap,
        uint32_t              kv_raw_dim,
        uint32_t              kv_lora_dim,
        uint32_t              qk_rope,
        bool                  cache_f16,
        float                 eps) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!q_out || !q || !kv_lora_cache || !k_rope_cache || !kv_raw ||
        !model_map || n_tokens == 0 || cache_cap == 0 ||
        q_n == 0 || kv_raw_dim == 0 || kv_lora_dim == 0 || qk_rope == 0 ||
        (q_n & 3u) != 0 || (kv_lora_dim & 3u) != 0 ||
        kv_lora_dim > kv_raw_dim ||
        qk_rope > kv_raw_dim - kv_lora_dim ||
        pos0 > cache_cap || n_tokens > cache_cap - pos0) {
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> qbuf = ds4_gpu_tensor_buffer(q);
        id<MTLBuffer> qoutbuf = ds4_gpu_tensor_buffer(q_out);
        id<MTLBuffer> kvcachebuf = ds4_gpu_tensor_buffer(kv_lora_cache);
        id<MTLBuffer> ropecachebuf = ds4_gpu_tensor_buffer(k_rope_cache);
        id<MTLBuffer> kvrawbuf = ds4_gpu_tensor_buffer(kv_raw);
        const uint64_t cache_elem_bytes = cache_f16 ? sizeof(uint16_t) : sizeof(float);
        const uint64_t q_row_bytes = (uint64_t)q_n * sizeof(float);
        const uint64_t kv_weight_bytes = (uint64_t)kv_lora_dim * sizeof(float);
        const uint64_t kv_cache_bytes =
            (uint64_t)cache_cap * kv_lora_dim * cache_elem_bytes;
        const uint64_t rope_cache_bytes =
            (uint64_t)cache_cap * qk_rope * cache_elem_bytes;
        const uint64_t kv_raw_bytes =
            (uint64_t)n_tokens * kv_raw_dim * sizeof(float);
        if (!qbuf || !qoutbuf || !kvcachebuf || !ropecachebuf || !kvrawbuf ||
            ds4_gpu_tensor_bytes(q) < q_row_bytes * n_tokens ||
            ds4_gpu_tensor_bytes(q_out) < q_row_bytes * n_tokens ||
            ds4_gpu_tensor_bytes(kv_lora_cache) < kv_cache_bytes ||
            ds4_gpu_tensor_bytes(k_rope_cache) < rope_cache_bytes ||
            ds4_gpu_tensor_bytes(kv_raw) < kv_raw_bytes) {
            fprintf(stderr, "ds4: Metal GLM fused q/kv norm compact store received undersized buffers\n");
            return 0;
        }
        if (q_weight_offset > model_size || q_row_bytes > model_size - q_weight_offset ||
            kv_weight_offset > model_size || kv_weight_bytes > model_size - kv_weight_offset) {
            fprintf(stderr, "ds4: Metal GLM fused q/kv norm compact store weight range is outside the mapped model\n");
            return 0;
        }

        uint64_t q_weight_inner = 0;
        id<MTLBuffer> q_weightbuf = ds4_gpu_wrap_model_range(model_map, model_size,
                                                              q_weight_offset, q_row_bytes,
                                                              &q_weight_inner);
        if (!q_weightbuf) return 0;
        uint64_t kv_weight_inner = 0;
        id<MTLBuffer> kv_weightbuf = ds4_gpu_wrap_model_range(model_map, model_size,
                                                               kv_weight_offset, kv_weight_bytes,
                                                               &kv_weight_inner);
        if (!kv_weightbuf) return 0;

        id<MTLComputePipelineState> pipeline =
            ds4_gpu_hot_pipeline(g_glm_qkv_norm_store_compact_kv_pipeline,
                                 "kernel_glm_qkv_norm_store_compact_kv");
        if (!pipeline) return 0;

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        ds4_gpu_glm_qkv_norm_store_compact_kv_args args = {
            .pos0 = pos0,
            .n_tokens = n_tokens,
            .cache_cap = cache_cap,
            .q_n = q_n,
            .q_n4 = q_n / 4u,
            .kv_raw_dim = kv_raw_dim,
            .kv_lora_dim = kv_lora_dim,
            .kv_lora_n4 = kv_lora_dim / 4u,
            .qk_rope = qk_rope,
            .cache_f16 = cache_f16 ? 1u : 0u,
            .eps = eps,
            .pad0 = 0,
        };

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:qbuf offset:ds4_gpu_tensor_offset(q) atIndex:1];
        [enc setBuffer:q_weightbuf offset:(NSUInteger)q_weight_inner atIndex:2];
        [enc setBuffer:qoutbuf offset:ds4_gpu_tensor_offset(q_out) atIndex:3];
        [enc setBuffer:kvrawbuf offset:ds4_gpu_tensor_offset(kv_raw) atIndex:4];
        [enc setBuffer:kv_weightbuf offset:(NSUInteger)kv_weight_inner atIndex:5];
        [enc setBuffer:kvcachebuf offset:ds4_gpu_tensor_offset(kv_lora_cache) atIndex:6];
        [enc setBuffer:ropecachebuf offset:ds4_gpu_tensor_offset(k_rope_cache) atIndex:7];
        [enc setThreadgroupMemoryLength:32u * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)n_tokens, 3, 1)
             threadsPerThreadgroup:MTLSizeMake(ds4_gpu_rms_norm_threads(q_n), 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "GLM fused q/kv norm compact store")) return 0;
    }

    return 1;
}

int ds4_gpu_glm_store_indexer_k_tensor(
        ds4_gpu_tensor       *indexer_key_cache,
        const ds4_gpu_tensor *raw_k,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint64_t              bias_offset,
        uint32_t              pos0,
        uint32_t              n_tokens,
        uint32_t              cache_cap,
        uint32_t              head_dim,
        uint32_t              rot_dim,
        uint32_t              n_ctx_orig,
        float                 eps,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow,
        bool                  cache_f16) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!indexer_key_cache || !raw_k || !model_map ||
        n_tokens == 0 || cache_cap == 0 ||
        head_dim == 0 || rot_dim == 0 ||
        rot_dim > head_dim || (rot_dim & 1u) != 0 ||
        pos0 > cache_cap || n_tokens > cache_cap - pos0) {
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> cachebuf = ds4_gpu_tensor_buffer(indexer_key_cache);
        id<MTLBuffer> rawbuf = ds4_gpu_tensor_buffer(raw_k);
        const uint64_t cache_elem_bytes = cache_f16 ? sizeof(uint16_t) : sizeof(float);
        const uint64_t cache_bytes =
            (uint64_t)cache_cap * head_dim * cache_elem_bytes;
        const uint64_t raw_bytes =
            (uint64_t)n_tokens * head_dim * sizeof(float);
        const uint64_t norm_bytes = (uint64_t)head_dim * sizeof(float);
        if (!cachebuf || !rawbuf ||
            ds4_gpu_tensor_bytes(indexer_key_cache) < cache_bytes ||
            ds4_gpu_tensor_bytes(raw_k) < raw_bytes) {
            fprintf(stderr, "ds4: Metal GLM indexer K store received undersized buffers\n");
            return 0;
        }
        if (weight_offset > model_size || norm_bytes > model_size - weight_offset ||
            bias_offset > model_size || norm_bytes > model_size - bias_offset) {
            fprintf(stderr, "ds4: Metal GLM indexer K norm range is outside the mapped model\n");
            return 0;
        }

        uint64_t weight_inner = 0;
        uint64_t bias_inner = 0;
        id<MTLBuffer> weightbuf =
            ds4_gpu_wrap_model_range(model_map, model_size,
                                     weight_offset, norm_bytes,
                                     &weight_inner);
        id<MTLBuffer> biasbuf =
            ds4_gpu_wrap_model_range(model_map, model_size,
                                     bias_offset, norm_bytes,
                                     &bias_inner);
        if (!weightbuf || !biasbuf) return 0;

        id<MTLComputePipelineState> pipeline =
            ds4_gpu_hot_pipeline(g_glm_store_indexer_k_pipeline,
                                 "kernel_glm_store_indexer_k");
        if (!pipeline) return 0;

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        ds4_gpu_glm_store_indexer_k_args args = {
            .pos0 = pos0,
            .n_tokens = n_tokens,
            .cache_cap = cache_cap,
            .head_dim = head_dim,
            .rot_dim = rot_dim,
            .n_ctx_orig = n_ctx_orig,
            .cache_f16 = cache_f16 ? 1u : 0u,
            .pad0 = 0,
            .eps = eps,
            .freq_base = freq_base,
            .freq_scale = freq_scale,
            .ext_factor = ext_factor,
            .attn_factor = attn_factor,
            .beta_fast = beta_fast,
            .beta_slow = beta_slow,
            .pad1 = 0.0f,
        };
        const NSUInteger nth = ds4_gpu_rms_norm_threads(head_dim);

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:rawbuf offset:ds4_gpu_tensor_offset(raw_k) atIndex:1];
        [enc setBuffer:weightbuf offset:(NSUInteger)weight_inner atIndex:2];
        [enc setBuffer:biasbuf offset:(NSUInteger)bias_inner atIndex:3];
        [enc setBuffer:cachebuf offset:ds4_gpu_tensor_offset(indexer_key_cache) atIndex:4];
        [enc setThreadgroupMemoryLength:nth * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)n_tokens, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "GLM indexer K store")) return 0;
    }

    return 1;
}

int ds4_gpu_glm_build_kv_cache_tensor(
        ds4_gpu_tensor       *key_cache,
        ds4_gpu_tensor       *value_cache,
        const ds4_gpu_tensor *kv_raw,
        const ds4_gpu_tensor *k_nope,
        const ds4_gpu_tensor *value,
        uint32_t              pos0,
        uint32_t              n_tokens,
        uint32_t              cache_cap,
        uint32_t              n_head,
        uint32_t              kv_raw_dim,
        uint32_t              kv_lora_dim,
        uint32_t              qk_nope,
        uint32_t              qk_rope,
        uint32_t              value_dim,
        uint32_t              n_ctx_orig,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow,
        bool                  cache_f16) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    const uint32_t qk_dim = qk_nope + qk_rope;
    if (!key_cache || !value_cache || !kv_raw || !k_nope || !value ||
        n_tokens == 0 || cache_cap == 0 || n_head == 0 ||
        kv_raw_dim == 0 || kv_lora_dim == 0 ||
        qk_nope == 0 || qk_rope == 0 || value_dim == 0 ||
        kv_lora_dim + qk_rope > kv_raw_dim ||
        qk_dim < qk_nope || (qk_rope & 1u) != 0 ||
        pos0 > cache_cap || n_tokens > cache_cap - pos0 ||
        !isfinite(freq_base) || freq_base <= 0.0f ||
        !isfinite(freq_scale) || freq_scale <= 0.0f ||
        !isfinite(ext_factor) || !isfinite(attn_factor) ||
        !isfinite(beta_fast) || !isfinite(beta_slow)) {
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> keybuf = ds4_gpu_tensor_buffer(key_cache);
        id<MTLBuffer> valbuf = ds4_gpu_tensor_buffer(value_cache);
        id<MTLBuffer> rawbuf = ds4_gpu_tensor_buffer(kv_raw);
        id<MTLBuffer> knbuf = ds4_gpu_tensor_buffer(k_nope);
        id<MTLBuffer> vbuf = ds4_gpu_tensor_buffer(value);
        const uint64_t cache_elem_bytes = cache_f16 ? sizeof(uint16_t) : sizeof(float);
        const uint64_t key_bytes = (uint64_t)cache_cap * n_head * qk_dim * cache_elem_bytes;
        const uint64_t cache_value_bytes = (uint64_t)cache_cap * n_head * value_dim * cache_elem_bytes;
        const uint64_t raw_bytes = (uint64_t)n_tokens * kv_raw_dim * sizeof(float);
        const uint64_t kn_bytes = (uint64_t)n_tokens * n_head * qk_nope * sizeof(float);
        const uint64_t value_bytes = (uint64_t)n_tokens * n_head * value_dim * sizeof(float);
        if (!keybuf || !valbuf || !rawbuf || !knbuf || !vbuf ||
            ds4_gpu_tensor_bytes(key_cache) < key_bytes ||
            ds4_gpu_tensor_bytes(value_cache) < cache_value_bytes ||
            ds4_gpu_tensor_bytes(kv_raw) < raw_bytes ||
            ds4_gpu_tensor_bytes(k_nope) < kn_bytes ||
            ds4_gpu_tensor_bytes(value) < value_bytes) {
            fprintf(stderr, "ds4: Metal GLM KV cache builder received undersized buffers\n");
            return 0;
        }

        const bool decode_group4 =
            n_tokens == 1u &&
            n_head >= 4u &&
            ds4_gpu_env_bool("DS4_METAL_DISABLE_GLM_DECODE_KV_GROUP4") <= 0;
        id<MTLComputePipelineState> pipeline =
            decode_group4 ?
            ds4_gpu_hot_pipeline(g_glm_build_kv_cache_decode_group4_pipeline,
                                 "kernel_glm_build_kv_cache_decode_group4") :
            ds4_gpu_hot_pipeline(g_glm_build_kv_cache_pipeline,
                                 "kernel_glm_build_kv_cache");
        if (!pipeline) return 0;

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        ds4_gpu_glm_build_kv_cache_args args = {
            .pos0 = pos0,
            .n_tokens = n_tokens,
            .cache_cap = cache_cap,
            .n_head = n_head,
            .kv_raw_dim = kv_raw_dim,
            .kv_lora_dim = kv_lora_dim,
            .qk_nope = qk_nope,
            .qk_rope = qk_rope,
            .value_dim = value_dim,
            .n_ctx_orig = n_ctx_orig,
            .cache_f16 = cache_f16 ? 1u : 0u,
            .pad0 = 0u,
            .freq_base = freq_base,
            .freq_scale = freq_scale,
            .ext_factor = ext_factor,
            .attn_factor = attn_factor,
            .beta_fast = beta_fast,
            .beta_slow = beta_slow,
        };

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:rawbuf offset:ds4_gpu_tensor_offset(kv_raw) atIndex:1];
        [enc setBuffer:knbuf offset:ds4_gpu_tensor_offset(k_nope) atIndex:2];
        [enc setBuffer:vbuf offset:ds4_gpu_tensor_offset(value) atIndex:3];
        [enc setBuffer:keybuf offset:ds4_gpu_tensor_offset(key_cache) atIndex:4];
        [enc setBuffer:valbuf offset:ds4_gpu_tensor_offset(value_cache) atIndex:5];
        const NSUInteger group_y = decode_group4 ?
            ((NSUInteger)n_head + 3u) / 4u :
            (NSUInteger)n_head;
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)n_tokens, group_y, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "GLM KV cache build")) return 0;
    }

    return 1;
}

int ds4_gpu_glm_build_kv_cache_flash_tensor(
        ds4_gpu_tensor       *key_cache,
        ds4_gpu_tensor       *value_cache,
        const ds4_gpu_tensor *kv_raw,
        const ds4_gpu_tensor *k_nope,
        const ds4_gpu_tensor *value,
        uint32_t              pos0,
        uint32_t              n_tokens,
        uint32_t              cache_cap,
        uint32_t              n_head,
        uint32_t              kv_raw_dim,
        uint32_t              kv_lora_dim,
        uint32_t              qk_nope,
        uint32_t              qk_rope,
        uint32_t              value_dim,
        uint32_t              n_ctx_orig,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow,
        bool                  cache_f16) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    const uint32_t qk_dim = qk_nope + qk_rope;
    if (!key_cache || !value_cache || !kv_raw || !k_nope || !value ||
        pos0 != 0 || n_tokens == 0 || cache_cap == 0 || n_head == 0 ||
        kv_raw_dim == 0 || kv_lora_dim == 0 ||
        qk_nope == 0 || qk_rope == 0 || value_dim == 0 ||
        kv_lora_dim + qk_rope > kv_raw_dim ||
        qk_dim < qk_nope || (qk_rope & 1u) != 0 ||
        n_tokens > cache_cap ||
        !isfinite(freq_base) || freq_base <= 0.0f ||
        !isfinite(freq_scale) || freq_scale <= 0.0f ||
        !isfinite(ext_factor) || !isfinite(attn_factor) ||
        !isfinite(beta_fast) || !isfinite(beta_slow)) {
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> keybuf = ds4_gpu_tensor_buffer(key_cache);
        id<MTLBuffer> valbuf = ds4_gpu_tensor_buffer(value_cache);
        id<MTLBuffer> rawbuf = ds4_gpu_tensor_buffer(kv_raw);
        id<MTLBuffer> knbuf = ds4_gpu_tensor_buffer(k_nope);
        id<MTLBuffer> vbuf = ds4_gpu_tensor_buffer(value);
        const uint64_t cache_elem_bytes = cache_f16 ? sizeof(uint16_t) : sizeof(float);
        const uint64_t key_bytes = (uint64_t)cache_cap * n_head * qk_dim * cache_elem_bytes;
        const uint64_t cache_value_bytes = (uint64_t)cache_cap * n_head * value_dim * cache_elem_bytes;
        const uint64_t raw_bytes = (uint64_t)n_tokens * kv_raw_dim * sizeof(float);
        const uint64_t kn_bytes = (uint64_t)n_tokens * n_head * qk_nope * sizeof(float);
        const uint64_t value_bytes = (uint64_t)n_tokens * n_head * value_dim * sizeof(float);
        if (!keybuf || !valbuf || !rawbuf || !knbuf || !vbuf ||
            ds4_gpu_tensor_bytes(key_cache) < key_bytes ||
            ds4_gpu_tensor_bytes(value_cache) < cache_value_bytes ||
            ds4_gpu_tensor_bytes(kv_raw) < raw_bytes ||
            ds4_gpu_tensor_bytes(k_nope) < kn_bytes ||
            ds4_gpu_tensor_bytes(value) < value_bytes) {
            fprintf(stderr, "ds4: Metal GLM staged KV cache builder received undersized buffers\n");
            return 0;
        }

        const NSUInteger q_row_bytes_f16 = (NSUInteger)qk_dim * sizeof(uint16_t);
        const NSUInteger v_row_bytes_f16 = (NSUInteger)value_dim * sizeof(uint16_t);
        const NSUInteger key_f16_offset = 0;
        const NSUInteger key_f16_bytes =
            (NSUInteger)n_tokens * (NSUInteger)n_head * q_row_bytes_f16;
        const NSUInteger value_f16_offset = key_f16_bytes;
        const NSUInteger value_f16_bytes =
            (NSUInteger)n_tokens * (NSUInteger)n_head * v_row_bytes_f16;
        const NSUInteger kv_f16_bytes = key_f16_bytes + value_f16_bytes;
        if (!ds4_gpu_ensure_scratch_buffer(&g_flash_attn_kv_buffer,
                                           &g_flash_attn_kv_bytes,
                                           kv_f16_bytes,
                                           "ds4_glm_flash_attn_kv_f16")) {
            return 0;
        }

        id<MTLComputePipelineState> pipeline =
            ds4_gpu_hot_pipeline(g_glm_build_kv_cache_flash_pipeline,
                                 "kernel_glm_build_kv_cache_flash");
        if (!pipeline) return 0;

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        ds4_gpu_glm_build_kv_cache_args args = {
            .pos0 = pos0,
            .n_tokens = n_tokens,
            .cache_cap = cache_cap,
            .n_head = n_head,
            .kv_raw_dim = kv_raw_dim,
            .kv_lora_dim = kv_lora_dim,
            .qk_nope = qk_nope,
            .qk_rope = qk_rope,
            .value_dim = value_dim,
            .n_ctx_orig = n_ctx_orig,
            .cache_f16 = cache_f16 ? 1u : 0u,
            .pad0 = 0u,
            .freq_base = freq_base,
            .freq_scale = freq_scale,
            .ext_factor = ext_factor,
            .attn_factor = attn_factor,
            .beta_fast = beta_fast,
            .beta_slow = beta_slow,
        };

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:rawbuf offset:ds4_gpu_tensor_offset(kv_raw) atIndex:1];
        [enc setBuffer:knbuf offset:ds4_gpu_tensor_offset(k_nope) atIndex:2];
        [enc setBuffer:vbuf offset:ds4_gpu_tensor_offset(value) atIndex:3];
        [enc setBuffer:keybuf offset:ds4_gpu_tensor_offset(key_cache) atIndex:4];
        [enc setBuffer:valbuf offset:ds4_gpu_tensor_offset(value_cache) atIndex:5];
        [enc setBuffer:g_flash_attn_kv_buffer offset:key_f16_offset atIndex:6];
        [enc setBuffer:g_flash_attn_kv_buffer offset:value_f16_offset atIndex:7];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)n_tokens, (NSUInteger)n_head, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "GLM staged KV cache build")) return 0;
    }

    return 1;
}

static int ds4_gpu_glm_attention_flash_tensor_impl(
        ds4_gpu_tensor       *heads,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *key_cache,
        const ds4_gpu_tensor *value_cache,
        uint32_t              pos0,
        uint32_t              n_tokens,
        uint32_t              cache_len,
        uint32_t              cache_cap,
        uint32_t              n_head,
        uint32_t              qk_dim,
        uint32_t              value_dim,
        bool                  cache_f16,
        int                   kv_pre_staged) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!heads || !q || !key_cache || !value_cache ||
        n_tokens == 0 || cache_len == 0 || cache_cap == 0 ||
        n_head == 0 || qk_dim != 256u || value_dim != 256u ||
        cache_len > cache_cap ||
        pos0 > cache_len || n_tokens > cache_len - pos0 ||
        cache_len > ds4_gpu_glm_flash_attention_max_cache_len()) {
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> headsbuf = ds4_gpu_tensor_buffer(heads);
        id<MTLBuffer> qbuf = ds4_gpu_tensor_buffer(q);
        id<MTLBuffer> keybuf = ds4_gpu_tensor_buffer(key_cache);
        id<MTLBuffer> valbuf = ds4_gpu_tensor_buffer(value_cache);
        const uint64_t heads_bytes = (uint64_t)n_tokens * n_head * value_dim * sizeof(float);
        const uint64_t q_bytes = (uint64_t)n_tokens * n_head * qk_dim * sizeof(float);
        const uint64_t cache_elem_bytes = cache_f16 ? sizeof(uint16_t) : sizeof(float);
        const uint64_t key_bytes = (uint64_t)cache_cap * n_head * qk_dim * cache_elem_bytes;
        const uint64_t value_bytes = (uint64_t)cache_cap * n_head * value_dim * cache_elem_bytes;
        if (!headsbuf || !qbuf || !keybuf || !valbuf ||
            ds4_gpu_tensor_bytes(heads) < heads_bytes ||
            ds4_gpu_tensor_bytes(q) < q_bytes ||
            ds4_gpu_tensor_bytes(key_cache) < key_bytes ||
            ds4_gpu_tensor_bytes(value_cache) < value_bytes) {
            fprintf(stderr, "ds4: Metal GLM FlashAttention received undersized buffers\n");
            return 0;
        }
        const uint64_t key_elems = (uint64_t)cache_len * n_head * qk_dim;
        const uint64_t value_elems = (uint64_t)cache_len * n_head * value_dim;
        if (key_elems > UINT32_MAX || value_elems > UINT32_MAX) {
            return 0;
        }

        const uint32_t nqptg = 8;
        const uint32_t ncpsg = 64;
        const uint32_t nsg = 4;
        const bool has_kvpad = (cache_len % ncpsg) != 0;
        const bool bc_mask = (n_tokens % nqptg) != 0;
        const NSUInteger q_row_bytes = (NSUInteger)qk_dim * sizeof(float);
        const NSUInteger q_row_bytes_f16 = (NSUInteger)qk_dim * sizeof(uint16_t);
        const NSUInteger v_row_bytes = (NSUInteger)value_dim * sizeof(float);
        const NSUInteger v_row_bytes_f16 = (NSUInteger)value_dim * sizeof(uint16_t);
        const NSUInteger mask_bytes = (NSUInteger)n_tokens * (NSUInteger)cache_len * sizeof(uint16_t);
        const NSUInteger key_f16_offset = 0;
        const NSUInteger key_f16_bytes =
            (NSUInteger)cache_len * (NSUInteger)n_head * q_row_bytes_f16;
        const NSUInteger value_f16_offset = key_f16_bytes;
        const NSUInteger value_f16_bytes =
            (NSUInteger)cache_len * (NSUInteger)n_head * v_row_bytes_f16;
        const NSUInteger kv_f16_bytes = key_f16_bytes + value_f16_bytes;
        const NSUInteger pad_bytes = has_kvpad
            ? (NSUInteger)ncpsg * ((NSUInteger)n_head * (q_row_bytes_f16 + v_row_bytes_f16) +
                                   (NSUInteger)n_tokens * sizeof(uint16_t))
            : 1u;
        const NSUInteger nblk0 = ((NSUInteger)cache_len + ncpsg - 1u) / ncpsg;
        const NSUInteger nblk1 = ((NSUInteger)n_tokens + nqptg - 1u) / nqptg;
        const NSUInteger blk_bytes = ds4_gpu_align_up_ns(nblk0 * nblk1, 32u);

        id<MTLBuffer> mask_buffer =
            ds4_gpu_glm_prefill_mask_buffer(pos0, n_tokens, cache_len, mask_bytes);
        if (!mask_buffer) return 0;
        if (kv_pre_staged) {
            if (!g_flash_attn_kv_buffer || g_flash_attn_kv_bytes < kv_f16_bytes) {
                fprintf(stderr, "ds4: GLM staged FlashAttention KV scratch is missing\n");
                return 0;
            }
        } else if (!ds4_gpu_ensure_scratch_buffer(&g_flash_attn_kv_buffer,
                                                  &g_flash_attn_kv_bytes,
                                                  kv_f16_bytes,
                                                  "ds4_glm_flash_attn_kv_f16")) {
            return 0;
        }
        if (!ds4_gpu_ensure_scratch_buffer(&g_flash_attn_pad_buffer,
                                             &g_flash_attn_pad_bytes,
                                             pad_bytes,
                                             "ds4_glm_flash_attn_pad") ||
            !ds4_gpu_ensure_scratch_buffer(&g_flash_attn_blk_buffer,
                                             &g_flash_attn_blk_bytes,
                                             blk_bytes,
                                             "ds4_glm_flash_attn_blk")) {
            return 0;
        }

        id<MTLComputePipelineState> pad_pipeline = nil;
        if (has_kvpad) {
            pad_pipeline = ds4_gpu_get_flash_attn_pad_pipeline(true, (int32_t)ncpsg);
            if (!pad_pipeline) return 0;
        }
        id<MTLComputePipelineState> blk_pipeline =
            ds4_gpu_get_flash_attn_blk_pipeline((int32_t)nqptg, (int32_t)ncpsg);
        id<MTLComputePipelineState> attn_pipeline =
            ds4_gpu_get_flash_attn_pipeline("kernel_flash_attn_ext_f16_dk256_dv256",
                                              true, false, false, false, has_kvpad, bc_mask,
                                              (int32_t)qk_dim,
                                              (int32_t)value_dim,
                                              (int32_t)nsg);
        if (!blk_pipeline || !attn_pipeline) return 0;

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        if (!kv_pre_staged) {
            const bool copied = cache_f16 ?
                (ds4_gpu_encode_cpy_f16_f16_3d(cb,
                                               keybuf,
                                               ds4_gpu_tensor_offset(key_cache),
                                               g_flash_attn_kv_buffer,
                                               key_f16_offset,
                                               qk_dim,
                                               cache_len,
                                               n_head,
                                               (uint64_t)n_head * q_row_bytes_f16,
                                               q_row_bytes_f16,
                                               q_row_bytes_f16,
                                               (uint64_t)cache_len * q_row_bytes_f16) &&
                 ds4_gpu_encode_cpy_f16_f16_3d(cb,
                                               valbuf,
                                               ds4_gpu_tensor_offset(value_cache),
                                               g_flash_attn_kv_buffer,
                                               value_f16_offset,
                                               value_dim,
                                               cache_len,
                                               n_head,
                                               (uint64_t)n_head * v_row_bytes_f16,
                                               v_row_bytes_f16,
                                               v_row_bytes_f16,
                                               (uint64_t)cache_len * v_row_bytes_f16)) :
                (ds4_gpu_encode_cpy_f32_f16_3d(cb,
                                               keybuf,
                                               ds4_gpu_tensor_offset(key_cache),
                                               g_flash_attn_kv_buffer,
                                               key_f16_offset,
                                               qk_dim,
                                               cache_len,
                                               n_head,
                                               (uint64_t)n_head * q_row_bytes,
                                               q_row_bytes,
                                               q_row_bytes_f16,
                                               (uint64_t)cache_len * q_row_bytes_f16) &&
                 ds4_gpu_encode_cpy_f32_f16_3d(cb,
                                               valbuf,
                                               ds4_gpu_tensor_offset(value_cache),
                                               g_flash_attn_kv_buffer,
                                               value_f16_offset,
                                               value_dim,
                                               cache_len,
                                               n_head,
                                               (uint64_t)n_head * v_row_bytes,
                                               v_row_bytes,
                                               v_row_bytes_f16,
                                               (uint64_t)cache_len * v_row_bytes_f16));
            if (!copied) {
                return 0;
            }
        }

        if (has_kvpad) {
            ds4_gpu_flash_attn_pad_args pad_args = {
                .ne11 = (int32_t)cache_len,
                .ne_12_2 = (int32_t)n_head,
                .ne_12_3 = 1,
                .nb11 = q_row_bytes_f16,
                .nb12 = (uint64_t)cache_len * q_row_bytes_f16,
                .nb13 = (uint64_t)cache_len * (uint64_t)n_head * q_row_bytes_f16,
                .nb21 = v_row_bytes_f16,
                .nb22 = (uint64_t)cache_len * v_row_bytes_f16,
                .nb23 = (uint64_t)cache_len * (uint64_t)n_head * v_row_bytes_f16,
                .ne31 = (int32_t)n_tokens,
                .ne32 = 1,
                .ne33 = 1,
                .nb31 = (uint64_t)cache_len * sizeof(uint16_t),
                .nb32 = mask_bytes,
                .nb33 = mask_bytes,
            };

            id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
            [enc setComputePipelineState:pad_pipeline];
            [enc setBytes:&pad_args length:sizeof(pad_args) atIndex:0];
            [enc setBuffer:g_flash_attn_kv_buffer offset:key_f16_offset atIndex:1];
            [enc setBuffer:g_flash_attn_kv_buffer offset:value_f16_offset atIndex:2];
            [enc setBuffer:mask_buffer offset:0 atIndex:3];
            [enc setBuffer:g_flash_attn_pad_buffer offset:0 atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake(ncpsg, n_head, 1)
                 threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            ds4_gpu_end_compute_encoder(cb, enc);
        }

        ds4_gpu_flash_attn_blk_args blk_args = {
            .ne01 = (int32_t)n_tokens,
            .ne30 = (int32_t)cache_len,
            .ne31 = (int32_t)n_tokens,
            .ne32 = 1,
            .ne33 = 1,
            .nb31 = (uint64_t)cache_len * sizeof(uint16_t),
            .nb32 = mask_bytes,
            .nb33 = mask_bytes,
        };

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:blk_pipeline];
        [enc setBytes:&blk_args length:sizeof(blk_args) atIndex:0];
        [enc setBuffer:mask_buffer offset:0 atIndex:1];
        [enc setBuffer:g_flash_attn_blk_buffer offset:0 atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake(nblk0, nblk1, 1)
             threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        ds4_gpu_flash_attn_vec_args args = {
            .ne01 = (int32_t)n_tokens,
            .ne02 = (int32_t)n_head,
            .ne03 = 1,
            .nb01 = (uint64_t)n_head * q_row_bytes,
            .nb02 = q_row_bytes,
            .nb03 = (uint64_t)n_tokens * n_head * q_row_bytes,
            .ne11 = (int32_t)cache_len,
            .ne_12_2 = (int32_t)n_head,
            .ne_12_3 = 1,
            .ns10 = (int32_t)qk_dim,
            .nb11 = q_row_bytes_f16,
            .nb12 = (uint64_t)cache_len * q_row_bytes_f16,
            .nb13 = (uint64_t)cache_len * (uint64_t)n_head * q_row_bytes_f16,
            .ns20 = (int32_t)value_dim,
            .nb21 = v_row_bytes_f16,
            .nb22 = (uint64_t)cache_len * v_row_bytes_f16,
            .nb23 = (uint64_t)cache_len * (uint64_t)n_head * v_row_bytes_f16,
            .ne31 = (int32_t)n_tokens,
            .ne32 = 1,
            .ne33 = 1,
            .nb31 = (uint64_t)cache_len * sizeof(uint16_t),
            .nb32 = mask_bytes,
            .nb33 = mask_bytes,
            .ne1 = (int32_t)n_head,
            .ne2 = (int32_t)n_tokens,
            .ne3 = 1,
            .scale = 1.0f / sqrtf((float)qk_dim),
            .max_bias = 0.0f,
            .m0 = 0.0f,
            .m1 = 0.0f,
            .n_head_log2 = 0,
            .logit_softcap = 0.0f,
        };

        const NSUInteger padded_v = ds4_gpu_align_up_ns(value_dim, 64u);
        const NSUInteger shared_elems = (NSUInteger)nqptg *
            ((NSUInteger)qk_dim + 2u * padded_v + 2u * (2u * (NSUInteger)ncpsg));
        const NSUInteger shared_bytes = ds4_gpu_align_up_ns(shared_elems * (sizeof(float) / 2u), 16u);

        enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:attn_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:qbuf offset:ds4_gpu_tensor_offset(q) atIndex:1];
        [enc setBuffer:g_flash_attn_kv_buffer offset:key_f16_offset atIndex:2];
        [enc setBuffer:g_flash_attn_kv_buffer offset:value_f16_offset atIndex:3];
        [enc setBuffer:mask_buffer offset:0 atIndex:4];
        [enc setBuffer:qbuf offset:ds4_gpu_tensor_offset(q) atIndex:5];
        [enc setBuffer:g_flash_attn_pad_buffer offset:0 atIndex:6];
        [enc setBuffer:g_flash_attn_blk_buffer offset:0 atIndex:7];
        [enc setBuffer:headsbuf offset:ds4_gpu_tensor_offset(heads) atIndex:8];
        [enc setThreadgroupMemoryLength:shared_bytes atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(nblk1, n_head, 1)
             threadsPerThreadgroup:MTLSizeMake(32, nsg, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "GLM FlashAttention")) return 0;
    }

    return 1;
}

int ds4_gpu_glm_attention_flash_tensor(
        ds4_gpu_tensor       *heads,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *key_cache,
        const ds4_gpu_tensor *value_cache,
        uint32_t              pos0,
        uint32_t              n_tokens,
        uint32_t              cache_len,
        uint32_t              cache_cap,
        uint32_t              n_head,
        uint32_t              qk_dim,
        uint32_t              value_dim,
        bool                  cache_f16) {
    return ds4_gpu_glm_attention_flash_tensor_impl(heads,
                                                   q,
                                                   key_cache,
                                                   value_cache,
                                                   pos0,
                                                   n_tokens,
                                                   cache_len,
                                                   cache_cap,
                                                   n_head,
                                                   qk_dim,
                                                   value_dim,
                                                   cache_f16,
                                                   0);
}

int ds4_gpu_glm_attention_flash_staged_tensor(
        ds4_gpu_tensor       *heads,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *key_cache,
        const ds4_gpu_tensor *value_cache,
        uint32_t              pos0,
        uint32_t              n_tokens,
        uint32_t              cache_len,
        uint32_t              cache_cap,
        uint32_t              n_head,
        uint32_t              qk_dim,
        uint32_t              value_dim,
        bool                  cache_f16) {
    if (pos0 != 0 || n_tokens != cache_len) return 0;
    return ds4_gpu_glm_attention_flash_tensor_impl(heads,
                                                   q,
                                                   key_cache,
                                                   value_cache,
                                                   pos0,
                                                   n_tokens,
                                                   cache_len,
                                                   cache_cap,
                                                   n_head,
                                                   qk_dim,
                                                   value_dim,
                                                   cache_f16,
                                                   1);
}

int ds4_gpu_glm_attention_full_tensor(
        ds4_gpu_tensor       *heads,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *key_cache,
        const ds4_gpu_tensor *value_cache,
        uint32_t              pos0,
        uint32_t              n_tokens,
        uint32_t              cache_len,
        uint32_t              cache_cap,
        uint32_t              n_head,
        uint32_t              qk_dim,
        uint32_t              value_dim,
        bool                  cache_f16) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!heads || !q || !key_cache || !value_cache ||
        n_tokens == 0 || cache_len == 0 || cache_cap == 0 ||
        n_head == 0 || qk_dim == 0 || value_dim == 0 ||
        (qk_dim & 3u) != 0 ||
        cache_len > cache_cap ||
        pos0 > cache_len || n_tokens > cache_len - pos0 ||
        cache_len > ds4_gpu_glm_full_attention_max_cache_len()) {
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> headsbuf = ds4_gpu_tensor_buffer(heads);
        id<MTLBuffer> qbuf = ds4_gpu_tensor_buffer(q);
        id<MTLBuffer> keybuf = ds4_gpu_tensor_buffer(key_cache);
        id<MTLBuffer> valbuf = ds4_gpu_tensor_buffer(value_cache);
        const uint64_t heads_bytes = (uint64_t)n_tokens * n_head * value_dim * sizeof(float);
        const uint64_t q_bytes = (uint64_t)n_tokens * n_head * qk_dim * sizeof(float);
        const uint64_t cache_elem_bytes = cache_f16 ? sizeof(uint16_t) : sizeof(float);
        const uint64_t key_bytes = (uint64_t)cache_cap * n_head * qk_dim * cache_elem_bytes;
        const uint64_t value_bytes = (uint64_t)cache_cap * n_head * value_dim * cache_elem_bytes;
        if (!headsbuf || !qbuf || !keybuf || !valbuf ||
            ds4_gpu_tensor_bytes(heads) < heads_bytes ||
            ds4_gpu_tensor_bytes(q) < q_bytes ||
            ds4_gpu_tensor_bytes(key_cache) < key_bytes ||
            ds4_gpu_tensor_bytes(value_cache) < value_bytes) {
            fprintf(stderr, "ds4: Metal GLM attention received undersized buffers\n");
            return 0;
        }

        id<MTLComputePipelineState> pipeline =
            ds4_gpu_hot_pipeline(g_glm_attention_full_pipeline,
                                 "kernel_glm_attention_full");
        if (!pipeline) return 0;

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        const uint32_t full_attn_mode = 2u;
        ds4_gpu_glm_attention_full_args args = {
            .pos0 = pos0,
            .n_tokens = n_tokens,
            .cache_len = cache_len,
            .cache_cap = cache_cap,
            .n_head = n_head,
            .qk_dim = qk_dim,
            .value_dim = value_dim,
            .pad0 = full_attn_mode,
            .cache_f16 = cache_f16 ? 1u : 0u,
            .pad1 = 0u,
            .pad2 = 0u,
            .scale = 1.0f / sqrtf((float)qk_dim),
        };

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:qbuf offset:ds4_gpu_tensor_offset(q) atIndex:1];
        [enc setBuffer:keybuf offset:ds4_gpu_tensor_offset(key_cache) atIndex:2];
        [enc setBuffer:valbuf offset:ds4_gpu_tensor_offset(value_cache) atIndex:3];
        [enc setBuffer:headsbuf offset:ds4_gpu_tensor_offset(heads) atIndex:4];
        [enc setThreadgroupMemoryLength:(256u + (NSUInteger)cache_len) * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)n_tokens, (NSUInteger)n_head, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "GLM full attention")) return 0;
    }

    return 1;
}

int ds4_gpu_glm_fill_selected_range_tensor(
        ds4_gpu_tensor *selected,
        uint32_t        n_selected) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!selected || n_selected == 0) return 0;

    @autoreleasepool {
        id<MTLBuffer> selectedbuf = ds4_gpu_tensor_buffer(selected);
        const uint64_t selected_bytes = (uint64_t)n_selected * sizeof(uint32_t);
        if (!selectedbuf || ds4_gpu_tensor_bytes(selected) < selected_bytes) {
            fprintf(stderr, "ds4: Metal GLM selected range received undersized buffer\n");
            return 0;
        }

        id<MTLComputePipelineState> pipeline =
            ds4_gpu_hot_pipeline(g_glm_fill_selected_range_pipeline,
                                 "kernel_glm_fill_selected_range");
        if (!pipeline) return 0;

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        ds4_gpu_glm_fill_selected_range_args args = {
            .n_selected = n_selected,
        };
        const NSUInteger nth = 256u;
        const NSUInteger n_groups = ((NSUInteger)n_selected + nth - 1u) / nth;

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:selectedbuf offset:ds4_gpu_tensor_offset(selected) atIndex:1];
        [enc dispatchThreadgroups:MTLSizeMake(n_groups, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "GLM selected range")) return 0;
    }

    return 1;
}

int ds4_gpu_glm_fill_selected_range_batch_tensor(
        ds4_gpu_tensor *selected,
        uint32_t        n_tokens,
        uint32_t        pos0,
        uint32_t        n_selected,
        uint32_t        pad_row) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!selected || n_tokens == 0 || n_selected == 0) return 0;

    @autoreleasepool {
        id<MTLBuffer> selectedbuf = ds4_gpu_tensor_buffer(selected);
        const uint64_t total = (uint64_t)n_tokens * n_selected;
        if (n_tokens != 0 && total / n_tokens != n_selected) return 0;
        if (total > UINT64_MAX / sizeof(uint32_t)) return 0;
        const uint64_t selected_bytes = total * sizeof(uint32_t);
        if (!selectedbuf || ds4_gpu_tensor_bytes(selected) < selected_bytes) {
            fprintf(stderr, "ds4: Metal GLM selected range batch received undersized buffer\n");
            return 0;
        }

        id<MTLComputePipelineState> pipeline =
            ds4_gpu_hot_pipeline(g_glm_fill_selected_range_batch_pipeline,
                                 "kernel_glm_fill_selected_range_batch");
        if (!pipeline) return 0;

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        ds4_gpu_glm_fill_selected_range_batch_args args = {
            .n_tokens = n_tokens,
            .pos0 = pos0,
            .n_selected = n_selected,
            .pad_row = pad_row,
        };
        const NSUInteger nth = 256u;
        const NSUInteger n_groups = ((NSUInteger)total + nth - 1u) / nth;

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:selectedbuf offset:ds4_gpu_tensor_offset(selected) atIndex:1];
        [enc dispatchThreadgroups:MTLSizeMake(n_groups, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "GLM selected range batch")) return 0;
    }

    return 1;
}

static int ds4_gpu_glm_rope_tail_offset_tensor(
        ds4_gpu_tensor *x,
        uint32_t        n_tokens,
        uint32_t        n_head,
        uint32_t        head_dim,
        uint32_t        rot_dim,
        uint32_t        rot_offset,
        uint32_t        pos0,
        uint32_t        n_ctx_orig,
        float           freq_base,
        float           freq_scale,
        float           ext_factor,
        float           attn_factor,
        float           beta_fast,
        float           beta_slow,
        const char     *label) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!x || n_tokens == 0 || n_head == 0 || head_dim == 0 ||
        rot_dim == 0 || rot_offset > head_dim || rot_dim > head_dim - rot_offset ||
        (rot_dim & 1u) != 0 ||
        pos0 > UINT32_MAX - n_tokens ||
        !isfinite(freq_base) || freq_base <= 0.0f ||
        !isfinite(freq_scale) || freq_scale <= 0.0f ||
        !isfinite(ext_factor) || !isfinite(attn_factor) ||
        !isfinite(beta_fast) || !isfinite(beta_slow)) {
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> xbuf = ds4_gpu_tensor_buffer(x);
        const uint64_t bytes = (uint64_t)n_tokens * n_head * head_dim * sizeof(float);
        if (!xbuf || ds4_gpu_tensor_bytes(x) < bytes) {
            fprintf(stderr, "ds4: Metal %s received undersized buffer\n",
                    label ? label : "GLM RoPE");
            return 0;
        }

        id<MTLComputePipelineState> pipeline =
            ds4_gpu_hot_pipeline(g_glm_indexer_rope_tail_pipeline,
                                 "kernel_glm_indexer_rope_tail_f32");
        if (!pipeline) return 0;

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        ds4_gpu_glm_rope_tail_args args = {
            .n_tokens = n_tokens,
            .n_head = n_head,
            .head_dim = head_dim,
            .rot_dim = rot_dim,
            .rot_offset = rot_offset,
            .pos0 = pos0,
            .n_ctx_orig = n_ctx_orig,
            .freq_base = freq_base,
            .freq_scale = freq_scale,
            .ext_factor = ext_factor,
            .attn_factor = attn_factor,
            .beta_fast = beta_fast,
            .beta_slow = beta_slow,
        };
        const NSUInteger nth = ds4_gpu_rms_norm_threads(rot_dim);

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:xbuf offset:ds4_gpu_tensor_offset(x) atIndex:1];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)n_head, (NSUInteger)n_tokens, 1)
             threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, label ? label : "GLM RoPE")) return 0;
    }

    return 1;
}

int ds4_gpu_glm_rope_tail_tensor(
        ds4_gpu_tensor *x,
        uint32_t        n_tokens,
        uint32_t        n_head,
        uint32_t        head_dim,
        uint32_t        rot_dim,
        uint32_t        pos0,
        uint32_t        n_ctx_orig,
        float           freq_base,
        float           freq_scale,
        float           ext_factor,
        float           attn_factor,
        float           beta_fast,
        float           beta_slow) {
    if (rot_dim > head_dim) return 0;
    return ds4_gpu_glm_rope_tail_offset_tensor(x,
                                               n_tokens,
                                               n_head,
                                               head_dim,
                                               rot_dim,
                                               head_dim - rot_dim,
                                               pos0,
                                               n_ctx_orig,
                                               freq_base,
                                               freq_scale,
                                               ext_factor,
                                               attn_factor,
                                               beta_fast,
                                               beta_slow,
                                               "GLM RoPE");
}

int ds4_gpu_glm_indexer_rope_tail_tensor(
        ds4_gpu_tensor *x,
        uint32_t        n_tokens,
        uint32_t        n_head,
        uint32_t        head_dim,
        uint32_t        rot_dim,
        uint32_t        pos0,
        uint32_t        n_ctx_orig,
        float           freq_base,
        float           freq_scale,
        float           ext_factor,
        float           attn_factor,
        float           beta_fast,
        float           beta_slow) {
    return ds4_gpu_glm_rope_tail_offset_tensor(x,
                                               n_tokens,
                                               n_head,
                                               head_dim,
                                               rot_dim,
                                               0,
                                               pos0,
                                               n_ctx_orig,
                                               freq_base,
                                               freq_scale,
                                               ext_factor,
                                               attn_factor,
                                               beta_fast,
                                               beta_slow,
                                               "GLM indexer RoPE");
}

int ds4_gpu_glm_indexer_score_one_tensor(
        ds4_gpu_tensor       *scores,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *indexer_key_cache,
        uint32_t              n_rows,
        uint32_t              n_head,
        uint32_t              head_dim,
        float                 scale,
        bool                  cache_f16) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!scores || !q || !weights || !indexer_key_cache ||
        n_rows == 0 || n_head == 0 || head_dim == 0 ||
        !isfinite(scale) || scale <= 0.0f) {
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> scoresbuf = ds4_gpu_tensor_buffer(scores);
        id<MTLBuffer> qbuf = ds4_gpu_tensor_buffer(q);
        id<MTLBuffer> weightsbuf = ds4_gpu_tensor_buffer(weights);
        id<MTLBuffer> cachebuf = ds4_gpu_tensor_buffer(indexer_key_cache);
        const uint64_t cache_elem_bytes = cache_f16 ? sizeof(uint16_t) : sizeof(float);
        const uint64_t score_bytes = (uint64_t)n_rows * sizeof(float);
        const uint64_t q_bytes = (uint64_t)n_head * head_dim * sizeof(float);
        const uint64_t weights_bytes = (uint64_t)n_head * sizeof(float);
        const uint64_t cache_bytes = (uint64_t)n_rows * head_dim * cache_elem_bytes;
        if (!scoresbuf || !qbuf || !weightsbuf || !cachebuf ||
            ds4_gpu_tensor_bytes(scores) < score_bytes ||
            ds4_gpu_tensor_bytes(q) < q_bytes ||
            ds4_gpu_tensor_bytes(weights) < weights_bytes ||
            ds4_gpu_tensor_bytes(indexer_key_cache) < cache_bytes) {
            fprintf(stderr, "ds4: Metal GLM indexer score received undersized buffers\n");
            return 0;
        }

        ds4_gpu_glm_indexer_score_one_args args = {
            .n_rows = n_rows,
            .n_head = n_head,
            .head_dim = head_dim,
            .cache_f16 = cache_f16 ? 1u : 0u,
            .scale = scale,
        };

        if (n_head == 32u && head_dim == 128u) {
            id<MTLComputePipelineState> direct_pipeline =
                ds4_gpu_hot_pipeline(g_glm_indexer_score_one_direct_pipeline,
                                     "kernel_glm_indexer_score_one_direct");
            if (!direct_pipeline) return 0;

            int owned = 0;
            id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
            if (!cb) return 0;

            id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
            [enc setComputePipelineState:direct_pipeline];
            [enc setBytes:&args length:sizeof(args) atIndex:0];
            [enc setBuffer:qbuf offset:ds4_gpu_tensor_offset(q) atIndex:1];
            [enc setBuffer:weightsbuf offset:ds4_gpu_tensor_offset(weights) atIndex:2];
            [enc setBuffer:cachebuf offset:ds4_gpu_tensor_offset(indexer_key_cache) atIndex:3];
            [enc setBuffer:scoresbuf offset:ds4_gpu_tensor_offset(scores) atIndex:4];
            [enc setThreadgroupMemoryLength:(128u + 4u) * sizeof(float) atIndex:0];
            [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)n_rows, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
            ds4_gpu_end_compute_encoder(cb, enc);

            if (!ds4_gpu_finish_command_buffer(cb, owned, "GLM indexer direct score")) return 0;
            return 1;
        }

        id<MTLComputePipelineState> pipeline =
            ds4_gpu_hot_pipeline(g_glm_indexer_score_one_pipeline,
                                 "kernel_glm_indexer_score_one");
        if (!pipeline) return 0;

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        const NSUInteger nth = ds4_gpu_rms_norm_threads(head_dim);

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:qbuf offset:ds4_gpu_tensor_offset(q) atIndex:1];
        [enc setBuffer:weightsbuf offset:ds4_gpu_tensor_offset(weights) atIndex:2];
        [enc setBuffer:cachebuf offset:ds4_gpu_tensor_offset(indexer_key_cache) atIndex:3];
        [enc setBuffer:scoresbuf offset:ds4_gpu_tensor_offset(scores) atIndex:4];
        [enc setThreadgroupMemoryLength:nth * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)n_rows, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "GLM indexer score")) return 0;
    }

    return 1;
}

int ds4_gpu_glm_indexer_scores_batch_tensor(
        ds4_gpu_tensor       *scores,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *indexer_key_cache,
        uint32_t              n_rows,
        uint32_t              n_tokens,
        uint32_t              pos0,
        uint32_t              n_head,
        uint32_t              head_dim,
        float                 scale,
        bool                  cache_f16) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!scores || !q || !weights || !indexer_key_cache ||
        n_rows == 0 || n_tokens == 0 || n_head == 0 || head_dim != 128 ||
        pos0 >= n_rows || n_tokens > n_rows - pos0 ||
        !isfinite(scale) || scale <= 0.0f) {
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> scoresbuf = ds4_gpu_tensor_buffer(scores);
        id<MTLBuffer> qbuf = ds4_gpu_tensor_buffer(q);
        id<MTLBuffer> weightsbuf = ds4_gpu_tensor_buffer(weights);
        id<MTLBuffer> cachebuf = ds4_gpu_tensor_buffer(indexer_key_cache);
        const uint64_t cache_elem_bytes = cache_f16 ? sizeof(uint16_t) : sizeof(float);
        const uint64_t score_bytes = (uint64_t)n_rows * n_tokens * sizeof(float);
        const uint64_t q_bytes = (uint64_t)n_tokens * n_head * head_dim * sizeof(float);
        const uint64_t weights_bytes = (uint64_t)n_tokens * n_head * sizeof(float);
        const uint64_t cache_bytes = (uint64_t)n_rows * head_dim * cache_elem_bytes;
        if (!scoresbuf || !qbuf || !weightsbuf || !cachebuf ||
            ds4_gpu_tensor_bytes(scores) < score_bytes ||
            ds4_gpu_tensor_bytes(q) < q_bytes ||
            ds4_gpu_tensor_bytes(weights) < weights_bytes ||
            ds4_gpu_tensor_bytes(indexer_key_cache) < cache_bytes) {
            fprintf(stderr, "ds4: Metal GLM indexer batch scores received undersized buffers\n");
            return 0;
        }

        const bool force_scalar = g_quality_mode;
        const bool use_tiled_f32 = false;
        const bool use_tiled = !force_scalar && n_tokens >= 8u &&
                               n_head == 32u && head_dim == 128u;
        id<MTLComputePipelineState> pipeline =
            use_tiled
                ? ds4_gpu_hot_pipeline(use_tiled_f32 ? g_glm_indexer_scores_tiled_f32_pipeline
                                                     : g_glm_indexer_scores_tiled_pipeline,
                                       use_tiled_f32 ? "kernel_glm_indexer_scores_tiled_f32"
                                                     : "kernel_glm_indexer_scores_tiled")
                : ds4_gpu_hot_pipeline(g_glm_indexer_scores_batch_pipeline,
                                       "kernel_glm_indexer_scores_batch");
        if (!pipeline) return 0;

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        ds4_gpu_glm_indexer_scores_batch_args args = {
            .n_rows = n_rows,
            .n_tokens = n_tokens,
            .n_head = n_head,
            .head_dim = head_dim,
            .pos0 = pos0,
            .cache_f16 = cache_f16 ? 1u : 0u,
            .q_token_stride = (uint64_t)n_head * head_dim * sizeof(float),
            .q_head_stride = (uint64_t)head_dim * sizeof(float),
            .weights_token_stride = (uint64_t)n_head * sizeof(float),
            .score_token_stride = (uint64_t)n_rows * sizeof(float),
            .scale = scale,
        };
        const NSUInteger nth = ds4_gpu_rms_norm_threads(head_dim);

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:qbuf offset:ds4_gpu_tensor_offset(q) atIndex:1];
        [enc setBuffer:weightsbuf offset:ds4_gpu_tensor_offset(weights) atIndex:2];
        [enc setBuffer:cachebuf offset:ds4_gpu_tensor_offset(indexer_key_cache) atIndex:3];
        [enc setBuffer:scoresbuf offset:ds4_gpu_tensor_offset(scores) atIndex:4];
        if (use_tiled) {
            const NSUInteger q_shared = 8u * 128u;
            const NSUInteger k_shared = 32u * 128u;
            const NSUInteger dot_shared = 8u * 32u;
            if (use_tiled_f32) {
                [enc setThreadgroupMemoryLength:(q_shared + k_shared + dot_shared) *
                                                sizeof(float) atIndex:0];
            } else {
                [enc setThreadgroupMemoryLength:(q_shared + k_shared) * sizeof(uint16_t) +
                                                dot_shared * sizeof(float) atIndex:0];
            }
            [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)n_rows + 31u) / 32u,
                                                  ((NSUInteger)n_tokens + 7u) / 8u,
                                                  1)
                 threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
        } else {
            [enc setThreadgroupMemoryLength:nth * sizeof(float) atIndex:0];
            [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)n_rows,
                                                  (NSUInteger)n_tokens,
                                                  1)
                 threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
        }
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "GLM indexer batch scores")) return 0;
    }

    return 1;
}

int ds4_gpu_glm_qk_lowrank_typed_tensor(
        ds4_gpu_tensor       *qk_low,
        const ds4_gpu_tensor *q,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint32_t              weight_type,
        uint32_t              n_head,
        uint32_t              kv_lora_dim,
        uint32_t              qk_nope,
        uint32_t              qk_dim) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!qk_low || !q || !model_map ||
        n_head == 0 || kv_lora_dim == 0 ||
        qk_nope == 0 || qk_nope > qk_dim) {
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> outbuf = ds4_gpu_tensor_buffer(qk_low);
        id<MTLBuffer> qbuf = ds4_gpu_tensor_buffer(q);
        uint64_t row_bytes = 0;
        if (!ds4_gpu_quant_row_bytes(weight_type, qk_nope, &row_bytes)) {
            fprintf(stderr, "ds4: Metal GLM qk lowrank received unsupported weight type\n");
            return 0;
        }
        const uint64_t weight_rows = (uint64_t)n_head * kv_lora_dim;
        const uint64_t weight_bytes = weight_rows * row_bytes;
        const uint64_t out_bytes = (uint64_t)n_head * kv_lora_dim * sizeof(float);
        const uint64_t q_bytes = (uint64_t)n_head * qk_dim * sizeof(float);
        if (!outbuf || !qbuf ||
            ds4_gpu_tensor_bytes(qk_low) < out_bytes ||
            ds4_gpu_tensor_bytes(q) < q_bytes) {
            fprintf(stderr, "ds4: Metal GLM qk lowrank received undersized buffers\n");
            return 0;
        }
        if (weight_offset > model_size || weight_bytes > model_size - weight_offset) {
            fprintf(stderr, "ds4: Metal GLM qk lowrank range is outside the mapped model\n");
            return 0;
        }

        uint64_t weight_inner = 0;
        id<MTLBuffer> weightbuf =
            ds4_gpu_wrap_model_range(model_map,
                                      model_size,
                                      weight_offset,
                                      weight_bytes,
                                      &weight_inner);
        if (!weightbuf) return 0;

        const int use_glm52 =
            n_head == 64u &&
            kv_lora_dim == 512u &&
            qk_nope == 192u &&
            qk_dim == 256u &&
            row_bytes == 204u &&
            weight_type == DS4_METAL_TENSOR_Q8_0;
        /* Coalesced simdgroup variant: lanes split the 192-dot so weight
         * reads coalesce, and 2048 threadgroups replace 64. The thread-
         * per-row kernels measured ~7.5x off the weight-bandwidth floor
         * (7.2ms of the decode token by skip-ablation). Covers the GLM 5.2
         * shape for both Q8_0 k_b and the DenseQ4 GGUF's Q4_0 k_b (the Q8
         * fast path above never engaged there). */
        const int use_glm52_sg =
            n_head == 64u &&
            kv_lora_dim == 512u &&
            qk_nope == 192u &&
            qk_dim == 256u &&
            ((weight_type == DS4_METAL_TENSOR_Q8_0 && row_bytes == 204u) ||
             (weight_type == DS4_METAL_TENSOR_Q4_0 && row_bytes == 108u)) &&
            getenv("DS4_METAL_DISABLE_GLM_QKLOW_SG") == NULL &&
            ds4_gpu_hot_pipeline(g_glm_qk_lowrank_glm52_sg_pipeline,
                                 "kernel_glm_qk_lowrank_q8_0_glm52_sg") != nil;
        if (getenv("DS4_METAL_GLM_QKLOW_DEBUG")) {
            static int printed = 0;
            if (!printed) {
                printed = 1;
                fprintf(stderr, "ds4: qk_lowrank decode path: use_glm52=%d sg=%d n_head=%u kv=%u nope=%u dim=%u rb=%llu type=%u\n",
                        use_glm52, use_glm52_sg, n_head, kv_lora_dim, qk_nope, qk_dim,
                        (unsigned long long)row_bytes, weight_type);
            }
        }
        id<MTLComputePipelineState> pipeline =
            use_glm52_sg ?
                ds4_gpu_hot_pipeline(g_glm_qk_lowrank_glm52_sg_pipeline,
                                     "kernel_glm_qk_lowrank_q8_0_glm52_sg") :
            use_glm52 ?
                ds4_gpu_hot_pipeline(g_glm_qk_lowrank_glm52_pipeline,
                                     "kernel_glm_qk_lowrank_q8_0_glm52") :
                ds4_gpu_hot_pipeline(g_glm_qk_lowrank_pipeline,
                                     "kernel_glm_qk_lowrank_q8_0");
        if (!pipeline) return 0;

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        ds4_gpu_glm_qk_lowrank_args args = {
            .n_head = n_head,
            .kv_lora_dim = kv_lora_dim,
            .qk_nope = qk_nope,
            .qk_dim = qk_dim,
            .row_bytes = (uint32_t)row_bytes,
            .weight_type = weight_type,
            .pad1 = 0,
            .pad2 = 0,
        };

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:weightbuf offset:(NSUInteger)weight_inner atIndex:1];
        [enc setBuffer:qbuf offset:ds4_gpu_tensor_offset(q) atIndex:2];
        [enc setBuffer:outbuf offset:ds4_gpu_tensor_offset(qk_low) atIndex:3];
        if (use_glm52_sg) {
            /* 8 simdgroups x 2 rows per threadgroup: (64, 512/16) grid. */
            [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)n_head,
                                                  (NSUInteger)(kv_lora_dim / 16u),
                                                  1)
                 threadsPerThreadgroup:MTLSizeMake(32, 8, 1)];
        } else {
            if (use_glm52) {
                [enc setThreadgroupMemoryLength:192u * sizeof(float) atIndex:0];
            }
            [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)n_head, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "GLM qk lowrank")) return 0;
    }

    return 1;
}

int ds4_gpu_glm_qk_lowrank_q8_0_tensor(
        ds4_gpu_tensor       *qk_low,
        const ds4_gpu_tensor *q,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint32_t              n_head,
        uint32_t              kv_lora_dim,
        uint32_t              qk_nope,
        uint32_t              qk_dim) {
    return ds4_gpu_glm_qk_lowrank_typed_tensor(qk_low,
                                               q,
                                               model_map,
                                               model_size,
                                               weight_offset,
                                               DS4_METAL_TENSOR_Q8_0,
                                               n_head,
                                               kv_lora_dim,
                                               qk_nope,
                                               qk_dim);
}

int ds4_gpu_glm_qk_lowrank_typed_batch_tensor(
        ds4_gpu_tensor       *qk_low,
        const ds4_gpu_tensor *q,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint32_t              weight_type,
        uint32_t              n_tokens,
        uint32_t              n_head,
        uint32_t              kv_lora_dim,
        uint32_t              qk_nope,
        uint32_t              qk_dim) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!qk_low || !q || !model_map ||
        n_tokens == 0 || n_head == 0 || kv_lora_dim == 0 ||
        qk_nope == 0 || qk_nope > qk_dim) {
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> outbuf = ds4_gpu_tensor_buffer(qk_low);
        id<MTLBuffer> qbuf = ds4_gpu_tensor_buffer(q);
        uint64_t row_bytes = 0;
        if (!ds4_gpu_quant_row_bytes(weight_type, qk_nope, &row_bytes)) {
            fprintf(stderr, "ds4: Metal GLM batch qk lowrank received unsupported weight type\n");
            return 0;
        }
        const uint64_t weight_rows = (uint64_t)n_head * kv_lora_dim;
        const uint64_t weight_bytes = weight_rows * row_bytes;
        const uint64_t out_bytes =
            (uint64_t)n_tokens * n_head * kv_lora_dim * sizeof(float);
        const uint64_t q_bytes =
            (uint64_t)n_tokens * n_head * qk_dim * sizeof(float);
        if (!outbuf || !qbuf ||
            ds4_gpu_tensor_bytes(qk_low) < out_bytes ||
            ds4_gpu_tensor_bytes(q) < q_bytes) {
            fprintf(stderr, "ds4: Metal GLM batch qk lowrank received undersized buffers\n");
            return 0;
        }
        if (weight_offset > model_size || weight_bytes > model_size - weight_offset) {
            fprintf(stderr, "ds4: Metal GLM batch qk lowrank range is outside the mapped model\n");
            return 0;
        }

        uint64_t weight_inner = 0;
        id<MTLBuffer> weightbuf =
            ds4_gpu_wrap_model_range(model_map,
                                      model_size,
                                      weight_offset,
                                      weight_bytes,
                                      &weight_inner);
        if (!weightbuf) return 0;

        const int use_glm52_t4 =
            n_tokens >= 4u &&
            n_head == 64u &&
            kv_lora_dim == 512u &&
            qk_nope == 192u &&
            qk_dim == 256u &&
            row_bytes == 204u &&
            weight_type == DS4_METAL_TENSOR_Q8_0;
        id<MTLComputePipelineState> pipeline =
            use_glm52_t4 ?
                ds4_gpu_hot_pipeline(g_glm_qk_lowrank_batch_glm52_t4_pipeline,
                                     "kernel_glm_qk_lowrank_q8_0_batch_glm52_t4") :
                ds4_gpu_hot_pipeline(g_glm_qk_lowrank_batch_pipeline,
                                     "kernel_glm_qk_lowrank_q8_0_batch");
        if (!pipeline) return 0;

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        uint32_t head_base = 0;
        uint32_t head_count = n_head;
        ds4_gpu_tp_attn_head_range(n_head, 8u, &head_base, &head_count);
        ds4_gpu_glm_qk_lowrank_batch_args args = {
            .n_tokens = n_tokens,
            .n_head = n_head,
            .kv_lora_dim = kv_lora_dim,
            .qk_nope = qk_nope,
            .qk_dim = qk_dim,
            .row_bytes = (uint32_t)row_bytes,
            .weight_type = weight_type,
            .head_base = head_base,
        };

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:weightbuf offset:(NSUInteger)weight_inner atIndex:1];
        [enc setBuffer:qbuf offset:ds4_gpu_tensor_offset(q) atIndex:2];
        [enc setBuffer:outbuf offset:ds4_gpu_tensor_offset(qk_low) atIndex:3];
        if (use_glm52_t4) {
            [enc setThreadgroupMemoryLength:4u * 192u * sizeof(float) atIndex:0];
            [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)head_count,
                                                  ((NSUInteger)n_tokens + 3u) / 4u,
                                                  1)
                 threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        } else {
            [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)head_count,
                                                  (NSUInteger)n_tokens,
                                                  1)
                 threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "GLM batch qk lowrank")) return 0;
    }

    return 1;
}

int ds4_gpu_glm_qk_lowrank_q8_0_batch_tensor(
        ds4_gpu_tensor       *qk_low,
        const ds4_gpu_tensor *q,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint32_t              n_tokens,
        uint32_t              n_head,
        uint32_t              kv_lora_dim,
        uint32_t              qk_nope,
        uint32_t              qk_dim) {
    return ds4_gpu_glm_qk_lowrank_typed_batch_tensor(qk_low,
                                                     q,
                                                     model_map,
                                                     model_size,
                                                     weight_offset,
                                                     DS4_METAL_TENSOR_Q8_0,
                                                     n_tokens,
                                                     n_head,
                                                     kv_lora_dim,
                                                     qk_nope,
                                                     qk_dim);
}

int ds4_gpu_glm_value_project_typed_batch_heads_tensor(
        ds4_gpu_tensor       *heads,
        const ds4_gpu_tensor *lora,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint32_t              weight_type,
        uint32_t              n_tokens,
        uint32_t              n_head,
        uint32_t              kv_lora_dim,
        uint32_t              value_dim) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!heads || !lora || !model_map ||
        n_tokens == 0 || n_head == 0 ||
        kv_lora_dim == 0 || value_dim == 0) {
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> headsbuf = ds4_gpu_tensor_buffer(heads);
        id<MTLBuffer> lorabuf = ds4_gpu_tensor_buffer(lora);
        uint64_t row_bytes = 0;
        if (!ds4_gpu_quant_row_bytes(weight_type, kv_lora_dim, &row_bytes)) {
            fprintf(stderr, "ds4: Metal GLM batch value project received unsupported weight type\n");
            return 0;
        }
        const uint64_t weight_rows = (uint64_t)n_head * value_dim;
        const uint64_t weight_bytes = weight_rows * row_bytes;
        const uint64_t heads_bytes =
            (uint64_t)n_tokens * n_head * value_dim * sizeof(float);
        const uint64_t lora_bytes =
            (uint64_t)n_tokens * n_head * kv_lora_dim * sizeof(float);
        if (!headsbuf || !lorabuf ||
            ds4_gpu_tensor_bytes(heads) < heads_bytes ||
            ds4_gpu_tensor_bytes(lora) < lora_bytes) {
            fprintf(stderr, "ds4: Metal GLM batch value project received undersized buffers\n");
            return 0;
        }
        if (weight_offset > model_size || weight_bytes > model_size - weight_offset) {
            fprintf(stderr, "ds4: Metal GLM batch value project range is outside the mapped model\n");
            return 0;
        }

        uint64_t weight_inner = 0;
        id<MTLBuffer> weightbuf =
            ds4_gpu_wrap_model_range(model_map,
                                      model_size,
                                      weight_offset,
                                      weight_bytes,
                                      &weight_inner);
        if (!weightbuf) return 0;

        const int use_mma =
            n_head == 64u &&
            kv_lora_dim == 512u &&
            value_dim == 256u &&
            row_bytes == 544u &&
            weight_type == DS4_METAL_TENSOR_Q8_0 &&
            n_tokens >= 32u;
        id<MTLComputePipelineState> pipeline =
            use_mma ?
                ds4_gpu_hot_pipeline(g_glm_value_project_q8_0_batch_heads_mma_pipeline,
                                     "kernel_glm_value_project_q8_0_batch_heads_mma") :
                ds4_gpu_hot_pipeline(g_glm_value_project_q8_0_batch_heads_pipeline,
                                     "kernel_glm_value_project_q8_0_batch_heads");
        if (!pipeline) return 0;

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        uint32_t head_base = 0;
        uint32_t head_count = n_head;
        ds4_gpu_tp_attn_head_range(n_head, 8u, &head_base, &head_count);
        ds4_gpu_glm_qk_lowrank_batch_args args = {
            .n_tokens = n_tokens,
            .n_head = n_head,
            .kv_lora_dim = kv_lora_dim,
            .qk_nope = 0,
            .qk_dim = value_dim,
            .row_bytes = (uint32_t)row_bytes,
            .weight_type = weight_type,
            .head_base = head_base,
        };

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:weightbuf offset:(NSUInteger)weight_inner atIndex:1];
        [enc setBuffer:lorabuf offset:ds4_gpu_tensor_offset(lora) atIndex:2];
        [enc setBuffer:headsbuf offset:ds4_gpu_tensor_offset(heads) atIndex:3];
        if (use_mma) {
            [enc setThreadgroupMemoryLength:16u * 1024u atIndex:0];
            [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)n_tokens + 31u) / 32u,
                                                  ((NSUInteger)value_dim + 63u) / 64u,
                                                  (NSUInteger)head_count)
                 threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        } else {
            [enc setThreadgroupMemoryLength:(NSUInteger)kv_lora_dim * sizeof(float)
                                    atIndex:0];
            [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)head_count,
                                                  (NSUInteger)n_tokens,
                                                  1)
                 threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "GLM batch value project")) return 0;
    }

    return 1;
}

int ds4_gpu_glm_value_project_q8_0_batch_heads_tensor(
        ds4_gpu_tensor       *heads,
        const ds4_gpu_tensor *lora,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint32_t              n_tokens,
        uint32_t              n_head,
        uint32_t              kv_lora_dim,
        uint32_t              value_dim) {
    return ds4_gpu_glm_value_project_typed_batch_heads_tensor(heads,
                                                              lora,
                                                              model_map,
                                                              model_size,
                                                              weight_offset,
                                                              DS4_METAL_TENSOR_Q8_0,
                                                              n_tokens,
                                                              n_head,
                                                              kv_lora_dim,
                                                              value_dim);
}

int ds4_gpu_glm_attention_indexed_decode_typed_tensor(
        ds4_gpu_tensor       *heads,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *qk_low,
        const ds4_gpu_tensor *kv_lora_cache,
        const ds4_gpu_tensor *k_rope_cache,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              value_weight_offset,
        uint32_t              value_weight_type,
        const ds4_gpu_tensor *selected,
        uint32_t              n_selected,
        uint32_t              cache_cap,
        bool                  cache_f16,
        uint32_t              n_head,
        uint32_t              kv_lora_dim,
        uint32_t              qk_nope,
        uint32_t              qk_rope,
        uint32_t              value_dim,
        uint32_t              n_ctx_orig,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    const uint32_t qk_dim = qk_nope + qk_rope;
    if (!heads || !q || !qk_low || !kv_lora_cache || !k_rope_cache ||
        !model_map || !selected ||
        n_selected == 0 || cache_cap == 0 || n_selected > cache_cap ||
        n_head == 0 || kv_lora_dim == 0 ||
        qk_nope == 0 || qk_rope == 0 || (qk_rope & 1u) != 0 ||
        value_dim == 0 || qk_dim < qk_nope ||
        !isfinite(freq_base) || freq_base <= 0.0f ||
        !isfinite(freq_scale) || freq_scale <= 0.0f ||
        !isfinite(ext_factor) || !isfinite(attn_factor) ||
        !isfinite(beta_fast) || !isfinite(beta_slow)) {
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> headsbuf = ds4_gpu_tensor_buffer(heads);
        id<MTLBuffer> qbuf = ds4_gpu_tensor_buffer(q);
        id<MTLBuffer> lowbuf = ds4_gpu_tensor_buffer(qk_low);
        id<MTLBuffer> kvcachebuf = ds4_gpu_tensor_buffer(kv_lora_cache);
        id<MTLBuffer> ropecachebuf = ds4_gpu_tensor_buffer(k_rope_cache);
        id<MTLBuffer> selectedbuf = ds4_gpu_tensor_buffer(selected);
        const uint64_t cache_elem_bytes = cache_f16 ? sizeof(uint16_t) : sizeof(float);
        uint64_t value_row_bytes = 0;
        if (!ds4_gpu_quant_row_bytes(value_weight_type, kv_lora_dim, &value_row_bytes)) {
            fprintf(stderr, "ds4: Metal GLM indexed attention received unsupported value type\n");
            return 0;
        }
        const uint64_t value_weight_rows = (uint64_t)n_head * value_dim;
        const uint64_t value_weight_bytes = value_weight_rows * value_row_bytes;
        const uint64_t heads_bytes = (uint64_t)n_head * value_dim * sizeof(float);
        const uint64_t q_bytes = (uint64_t)n_head * qk_dim * sizeof(float);
        const uint64_t low_bytes = (uint64_t)n_head * kv_lora_dim * sizeof(float);
        const uint64_t kv_cache_bytes = (uint64_t)cache_cap * kv_lora_dim * cache_elem_bytes;
        const uint64_t rope_cache_bytes = (uint64_t)cache_cap * qk_rope * cache_elem_bytes;
        const uint64_t selected_bytes = (uint64_t)n_selected * sizeof(uint32_t);
        if (!headsbuf || !qbuf || !lowbuf || !kvcachebuf || !ropecachebuf || !selectedbuf ||
            ds4_gpu_tensor_bytes(heads) < heads_bytes ||
            ds4_gpu_tensor_bytes(q) < q_bytes ||
            ds4_gpu_tensor_bytes(qk_low) < low_bytes ||
            ds4_gpu_tensor_bytes(kv_lora_cache) < kv_cache_bytes ||
            ds4_gpu_tensor_bytes(k_rope_cache) < rope_cache_bytes ||
            ds4_gpu_tensor_bytes(selected) < selected_bytes) {
            fprintf(stderr, "ds4: Metal GLM indexed attention received undersized buffers\n");
            return 0;
        }
        if (value_weight_offset > model_size ||
            value_weight_bytes > model_size - value_weight_offset) {
            fprintf(stderr, "ds4: Metal GLM indexed attention value range is outside the mapped model\n");
            return 0;
        }

        uint64_t value_inner = 0;
        id<MTLBuffer> valuebuf =
            ds4_gpu_wrap_model_range(model_map,
                                      model_size,
                                      value_weight_offset,
                                      value_weight_bytes,
                                      &value_inner);
        if (!valuebuf) return 0;

        id<MTLComputePipelineState> pipeline =
            ds4_gpu_hot_pipeline(g_glm_attention_indexed_decode_pipeline,
                                 "kernel_glm_attention_indexed_decode");
        if (!pipeline) return 0;

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        ds4_gpu_glm_attention_indexed_decode_args args = {
            .n_selected = n_selected,
            .cache_cap = cache_cap,
            .cache_f16 = cache_f16 ? 1u : 0u,
            .n_head = n_head,
            .kv_lora_dim = kv_lora_dim,
            .qk_nope = qk_nope,
            .qk_rope = qk_rope,
            .value_dim = value_dim,
            .n_ctx_orig = n_ctx_orig,
            .value_row_bytes = (uint32_t)value_row_bytes,
            .scale = 1.0f / sqrtf((float)qk_dim),
            .freq_base = freq_base,
            .freq_scale = freq_scale,
            .ext_factor = ext_factor,
            .attn_factor = attn_factor,
            .beta_fast = beta_fast,
            .beta_slow = beta_slow,
            .value_type = value_weight_type,
        };
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:qbuf offset:ds4_gpu_tensor_offset(q) atIndex:1];
        [enc setBuffer:lowbuf offset:ds4_gpu_tensor_offset(qk_low) atIndex:2];
        [enc setBuffer:kvcachebuf offset:ds4_gpu_tensor_offset(kv_lora_cache) atIndex:3];
        [enc setBuffer:ropecachebuf offset:ds4_gpu_tensor_offset(k_rope_cache) atIndex:4];
        [enc setBuffer:valuebuf offset:(NSUInteger)value_inner atIndex:5];
        [enc setBuffer:selectedbuf offset:ds4_gpu_tensor_offset(selected) atIndex:6];
        [enc setBuffer:headsbuf offset:ds4_gpu_tensor_offset(heads) atIndex:7];
        const NSUInteger scratch_floats =
            256u + (NSUInteger)n_selected + (NSUInteger)kv_lora_dim;
        [enc setThreadgroupMemoryLength:scratch_floats * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)n_head, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "GLM indexed attention decode")) return 0;
    }

    return 1;
}

int ds4_gpu_glm_attention_indexed_decode_tensor(
        ds4_gpu_tensor       *heads,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *qk_low,
        const ds4_gpu_tensor *kv_lora_cache,
        const ds4_gpu_tensor *k_rope_cache,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              value_weight_offset,
        const ds4_gpu_tensor *selected,
        uint32_t              n_selected,
        uint32_t              cache_cap,
        bool                  cache_f16,
        uint32_t              n_head,
        uint32_t              kv_lora_dim,
        uint32_t              qk_nope,
        uint32_t              qk_rope,
        uint32_t              value_dim,
        uint32_t              n_ctx_orig,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow) {
    return ds4_gpu_glm_attention_indexed_decode_typed_tensor(heads,
                                                             q,
                                                             qk_low,
                                                             kv_lora_cache,
                                                             k_rope_cache,
                                                             model_map,
                                                             model_size,
                                                             value_weight_offset,
                                                             DS4_METAL_TENSOR_Q8_0,
                                                             selected,
                                                             n_selected,
                                                             cache_cap,
                                                             cache_f16,
                                                             n_head,
                                                             kv_lora_dim,
                                                             qk_nope,
                                                             qk_rope,
                                                             value_dim,
                                                             n_ctx_orig,
                                                             freq_base,
                                                             freq_scale,
                                                             ext_factor,
                                                             attn_factor,
                                                             beta_fast,
                                                             beta_slow);
}

int ds4_gpu_glm_attention_indexed_decode_split_group8_typed_tensor(
        ds4_gpu_tensor       *heads,
        ds4_gpu_tensor       *partial_lora,
        ds4_gpu_tensor       *partial_ms,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *qk_low,
        const ds4_gpu_tensor *kv_lora_cache,
        const ds4_gpu_tensor *k_rope_cache,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              value_weight_offset,
        uint32_t              value_weight_type,
        const ds4_gpu_tensor *selected,
        uint32_t              n_selected,
        bool                  selected_rows_valid,
        uint32_t              cache_cap,
        bool                  cache_f16,
        uint32_t              n_head,
        uint32_t              kv_lora_dim,
        uint32_t              qk_nope,
        uint32_t              qk_rope,
        uint32_t              value_dim,
        uint32_t              n_ctx_orig,
        uint32_t              block_rows,
        uint32_t              n_blocks,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    const uint32_t qk_dim = qk_nope + qk_rope;
    const uint32_t needed_blocks =
        block_rows != 0u ? (n_selected + block_rows - 1u) / block_rows : 0u;
    if (!heads || !partial_lora || !partial_ms || !q || !qk_low ||
        !kv_lora_cache || !k_rope_cache || !model_map || !selected ||
        n_selected == 0 || cache_cap == 0 || n_selected > cache_cap ||
        n_head == 0 || (n_head % 8u) != 0 ||
        kv_lora_dim != 512u ||
        qk_nope == 0 || qk_rope != 64u ||
        value_dim == 0 || qk_dim < qk_nope ||
        block_rows == 0u || needed_blocks == 0u ||
        n_blocks < needed_blocks || n_blocks > 64u ||
        !cache_f16 ||
        !isfinite(freq_base) || freq_base <= 0.0f ||
        !isfinite(freq_scale) || freq_scale <= 0.0f ||
        !isfinite(ext_factor) || !isfinite(attn_factor) ||
        !isfinite(beta_fast) || !isfinite(beta_slow)) {
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> headsbuf = ds4_gpu_tensor_buffer(heads);
        id<MTLBuffer> partial_lorabuf = ds4_gpu_tensor_buffer(partial_lora);
        id<MTLBuffer> partial_msbuf = ds4_gpu_tensor_buffer(partial_ms);
        id<MTLBuffer> qbuf = ds4_gpu_tensor_buffer(q);
        id<MTLBuffer> lowbuf = ds4_gpu_tensor_buffer(qk_low);
        id<MTLBuffer> kvcachebuf = ds4_gpu_tensor_buffer(kv_lora_cache);
        id<MTLBuffer> ropecachebuf = ds4_gpu_tensor_buffer(k_rope_cache);
        id<MTLBuffer> selectedbuf = ds4_gpu_tensor_buffer(selected);
        const uint64_t cache_elem_bytes = sizeof(uint16_t);
        uint64_t value_row_bytes = 0;
        if (!ds4_gpu_quant_row_bytes(value_weight_type, kv_lora_dim, &value_row_bytes)) {
            fprintf(stderr, "ds4: Metal GLM split grouped indexed attention received unsupported value type\n");
            return 0;
        }
        const uint64_t value_weight_rows = (uint64_t)n_head * value_dim;
        const uint64_t value_weight_bytes = value_weight_rows * value_row_bytes;
        const uint64_t heads_bytes = (uint64_t)n_head * value_dim * sizeof(float);
        const uint64_t partial_lora_bytes =
            (uint64_t)n_blocks * n_head * kv_lora_dim * sizeof(float);
        const uint64_t partial_ms_bytes =
            (uint64_t)n_blocks * n_head * 2u * sizeof(float);
        const uint64_t q_bytes = (uint64_t)n_head * qk_dim * sizeof(float);
        const uint64_t low_bytes = (uint64_t)n_head * kv_lora_dim * sizeof(float);
        const uint64_t kv_cache_bytes = (uint64_t)cache_cap * kv_lora_dim * cache_elem_bytes;
        const uint64_t rope_cache_bytes = (uint64_t)cache_cap * qk_rope * cache_elem_bytes;
        const uint64_t selected_bytes = (uint64_t)n_selected * sizeof(uint32_t);
        if (!headsbuf || !partial_lorabuf || !partial_msbuf ||
            !qbuf || !lowbuf || !kvcachebuf || !ropecachebuf || !selectedbuf ||
            ds4_gpu_tensor_bytes(heads) < heads_bytes ||
            ds4_gpu_tensor_bytes(partial_lora) < partial_lora_bytes ||
            ds4_gpu_tensor_bytes(partial_ms) < partial_ms_bytes ||
            ds4_gpu_tensor_bytes(q) < q_bytes ||
            ds4_gpu_tensor_bytes(qk_low) < low_bytes ||
            ds4_gpu_tensor_bytes(kv_lora_cache) < kv_cache_bytes ||
            ds4_gpu_tensor_bytes(k_rope_cache) < rope_cache_bytes ||
            ds4_gpu_tensor_bytes(selected) < selected_bytes) {
            fprintf(stderr, "ds4: Metal GLM split grouped indexed attention received undersized buffers\n");
            return 0;
        }
        if (value_weight_offset > model_size ||
            value_weight_bytes > model_size - value_weight_offset) {
            fprintf(stderr, "ds4: Metal GLM split grouped indexed attention value range is outside the mapped model\n");
            return 0;
        }

        uint64_t value_inner = 0;
        id<MTLBuffer> valuebuf =
            ds4_gpu_wrap_model_range(model_map,
                                      model_size,
                                      value_weight_offset,
                                      value_weight_bytes,
                                      &value_inner);
        if (!valuebuf) return 0;

        const bool use_valid_fullheads =
            selected_rows_valid && (n_head % 8u) == 0u;
        id<MTLComputePipelineState> partial_pipeline =
            ds4_gpu_hot_pipeline(use_valid_fullheads ?
                                     g_glm_attention_indexed_decode_split_group8_partial_valid_fullheads_pipeline :
                                     g_glm_attention_indexed_decode_split_group8_partial_pipeline,
                                 use_valid_fullheads ?
                                     "kernel_glm_attention_indexed_decode_split_group8_partial_valid_fullheads" :
                                     "kernel_glm_attention_indexed_decode_split_group8_partial");
        const bool use_reduce16 =
            n_blocks == 16u && block_rows == 128u &&
            n_selected == 2048u && value_dim == 256u;
        id<MTLComputePipelineState> reduce_pipeline =
            ds4_gpu_hot_pipeline(use_reduce16 ?
                                     g_glm_attention_indexed_decode_split_group8_reduce16_pipeline :
                                     g_glm_attention_indexed_decode_split_group8_reduce_pipeline,
                                 use_reduce16 ?
                                     "kernel_glm_attention_indexed_decode_split_group8_reduce16" :
                                     "kernel_glm_attention_indexed_decode_split_group8_reduce");
        if (!partial_pipeline || !reduce_pipeline) return 0;

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        ds4_gpu_glm_attention_indexed_decode_split_args args = {
            .n_selected = n_selected,
            .cache_cap = cache_cap,
            .cache_f16 = 1u,
            .n_head = n_head,
            .kv_lora_dim = kv_lora_dim,
            .qk_nope = qk_nope,
            .qk_rope = qk_rope,
            .value_dim = value_dim,
            .n_ctx_orig = n_ctx_orig,
            .value_row_bytes = (uint32_t)value_row_bytes,
            .block_rows = block_rows,
            .n_blocks = n_blocks,
            .scale = 1.0f / sqrtf((float)qk_dim),
            .freq_base = freq_base,
            .freq_scale = freq_scale,
            .ext_factor = ext_factor,
            .attn_factor = attn_factor,
            .beta_fast = beta_fast,
            .beta_slow = beta_slow,
            .value_type = value_weight_type,
        };
        const NSUInteger stage_rows = 16u;
        const NSUInteger kv_vecs = (NSUInteger)kv_lora_dim / 4u;
        const NSUInteger rope_vecs = (NSUInteger)qk_rope / 4u;
        const NSUInteger partial_scratch_bytes =
            stage_rows * kv_vecs * sizeof(uint16_t) * 4u +
            stage_rows * rope_vecs * sizeof(float) * 4u;

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:partial_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:qbuf offset:ds4_gpu_tensor_offset(q) atIndex:1];
        [enc setBuffer:lowbuf offset:ds4_gpu_tensor_offset(qk_low) atIndex:2];
        [enc setBuffer:kvcachebuf offset:ds4_gpu_tensor_offset(kv_lora_cache) atIndex:3];
        [enc setBuffer:ropecachebuf offset:ds4_gpu_tensor_offset(k_rope_cache) atIndex:4];
        [enc setBuffer:selectedbuf offset:ds4_gpu_tensor_offset(selected) atIndex:5];
        [enc setBuffer:partial_lorabuf offset:ds4_gpu_tensor_offset(partial_lora) atIndex:6];
        [enc setBuffer:partial_msbuf offset:ds4_gpu_tensor_offset(partial_ms) atIndex:7];
        [enc setThreadgroupMemoryLength:partial_scratch_bytes atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)n_head / 8u, (NSUInteger)n_blocks, 1)
             threadsPerThreadgroup:MTLSizeMake(32, 8, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        const NSUInteger reduce_scratch_floats = 256u + 64u + (NSUInteger)kv_lora_dim;
        enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:reduce_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:partial_lorabuf offset:ds4_gpu_tensor_offset(partial_lora) atIndex:1];
        [enc setBuffer:partial_msbuf offset:ds4_gpu_tensor_offset(partial_ms) atIndex:2];
        [enc setBuffer:valuebuf offset:(NSUInteger)value_inner atIndex:3];
        [enc setBuffer:headsbuf offset:ds4_gpu_tensor_offset(heads) atIndex:4];
        [enc setThreadgroupMemoryLength:reduce_scratch_floats * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)n_head, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "GLM split grouped indexed attention decode")) return 0;
    }

    return 1;
}

int ds4_gpu_glm_attention_indexed_decode_split_group8_tensor(
        ds4_gpu_tensor       *heads,
        ds4_gpu_tensor       *partial_lora,
        ds4_gpu_tensor       *partial_ms,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *qk_low,
        const ds4_gpu_tensor *kv_lora_cache,
        const ds4_gpu_tensor *k_rope_cache,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              value_weight_offset,
        const ds4_gpu_tensor *selected,
        uint32_t              n_selected,
        bool                  selected_rows_valid,
        uint32_t              cache_cap,
        bool                  cache_f16,
        uint32_t              n_head,
        uint32_t              kv_lora_dim,
        uint32_t              qk_nope,
        uint32_t              qk_rope,
        uint32_t              value_dim,
        uint32_t              n_ctx_orig,
        uint32_t              block_rows,
        uint32_t              n_blocks,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow) {
    return ds4_gpu_glm_attention_indexed_decode_split_group8_typed_tensor(heads,
                                                                          partial_lora,
                                                                          partial_ms,
                                                                          q,
                                                                          qk_low,
                                                                          kv_lora_cache,
                                                                          k_rope_cache,
                                                                          model_map,
                                                                          model_size,
                                                                          value_weight_offset,
                                                                          DS4_METAL_TENSOR_Q8_0,
                                                                          selected,
                                                                          n_selected,
                                                                          selected_rows_valid,
                                                                          cache_cap,
                                                                          cache_f16,
                                                                          n_head,
                                                                          kv_lora_dim,
                                                                          qk_nope,
                                                                          qk_rope,
                                                                          value_dim,
                                                                          n_ctx_orig,
                                                                          block_rows,
                                                                          n_blocks,
                                                                          freq_base,
                                                                          freq_scale,
                                                                          ext_factor,
                                                                          attn_factor,
                                                                          beta_fast,
                                                                          beta_slow);
}

int ds4_gpu_glm_attention_indexed_batch_typed_tensor(
        ds4_gpu_tensor       *heads,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *qk_low,
        const ds4_gpu_tensor *kv_lora_cache,
        const ds4_gpu_tensor *k_rope_cache,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              value_weight_offset,
        uint32_t              value_weight_type,
        const ds4_gpu_tensor *selected,
        uint32_t              n_tokens,
        uint32_t              n_selected,
        uint32_t              cache_cap,
        bool                  cache_f16,
        uint32_t              n_head,
        uint32_t              kv_lora_dim,
        uint32_t              qk_nope,
        uint32_t              qk_rope,
        uint32_t              value_dim,
        uint32_t              n_ctx_orig,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    const uint32_t qk_dim = qk_nope + qk_rope;
    if (!heads || !q || !qk_low || !kv_lora_cache || !k_rope_cache ||
        !model_map || !selected ||
        n_tokens == 0 || n_selected == 0 || cache_cap == 0 || n_selected > cache_cap ||
        n_head == 0 || kv_lora_dim == 0 ||
        qk_nope == 0 || qk_rope == 0 || (qk_rope & 1u) != 0 ||
        value_dim == 0 || qk_dim < qk_nope ||
        !isfinite(freq_base) || freq_base <= 0.0f ||
        !isfinite(freq_scale) || freq_scale <= 0.0f ||
        !isfinite(ext_factor) || !isfinite(attn_factor) ||
        !isfinite(beta_fast) || !isfinite(beta_slow)) {
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> headsbuf = ds4_gpu_tensor_buffer(heads);
        id<MTLBuffer> qbuf = ds4_gpu_tensor_buffer(q);
        id<MTLBuffer> lowbuf = ds4_gpu_tensor_buffer(qk_low);
        id<MTLBuffer> kvcachebuf = ds4_gpu_tensor_buffer(kv_lora_cache);
        id<MTLBuffer> ropecachebuf = ds4_gpu_tensor_buffer(k_rope_cache);
        id<MTLBuffer> selectedbuf = ds4_gpu_tensor_buffer(selected);
        const uint64_t cache_elem_bytes = cache_f16 ? sizeof(uint16_t) : sizeof(float);
        uint64_t value_row_bytes = 0;
        if (!ds4_gpu_quant_row_bytes(value_weight_type, kv_lora_dim, &value_row_bytes)) {
            fprintf(stderr, "ds4: Metal GLM indexed batch attention received unsupported value type\n");
            return 0;
        }
        const uint64_t value_weight_rows = (uint64_t)n_head * value_dim;
        const uint64_t value_weight_bytes = value_weight_rows * value_row_bytes;
        const uint64_t heads_bytes =
            (uint64_t)n_tokens * n_head * value_dim * sizeof(float);
        const uint64_t q_bytes =
            (uint64_t)n_tokens * n_head * qk_dim * sizeof(float);
        const uint64_t low_bytes =
            (uint64_t)n_tokens * n_head * kv_lora_dim * sizeof(float);
        const uint64_t kv_cache_bytes = (uint64_t)cache_cap * kv_lora_dim * cache_elem_bytes;
        const uint64_t rope_cache_bytes = (uint64_t)cache_cap * qk_rope * cache_elem_bytes;
        const uint64_t selected_bytes =
            (uint64_t)n_tokens * n_selected * sizeof(uint32_t);
        if (!headsbuf || !qbuf || !lowbuf || !kvcachebuf || !ropecachebuf || !selectedbuf ||
            ds4_gpu_tensor_bytes(heads) < heads_bytes ||
            ds4_gpu_tensor_bytes(q) < q_bytes ||
            ds4_gpu_tensor_bytes(qk_low) < low_bytes ||
            ds4_gpu_tensor_bytes(kv_lora_cache) < kv_cache_bytes ||
            ds4_gpu_tensor_bytes(k_rope_cache) < rope_cache_bytes ||
            ds4_gpu_tensor_bytes(selected) < selected_bytes) {
            fprintf(stderr, "ds4: Metal GLM indexed batch attention received undersized buffers\n");
            return 0;
        }
        if (value_weight_offset > model_size ||
            value_weight_bytes > model_size - value_weight_offset) {
            fprintf(stderr, "ds4: Metal GLM indexed batch attention value range is outside the mapped model\n");
            return 0;
        }

        uint64_t value_inner = 0;
        id<MTLBuffer> valuebuf =
            ds4_gpu_wrap_model_range(model_map, model_size,
                                      value_weight_offset,
                                      value_weight_bytes,
                                      &value_inner);
        if (!valuebuf) return 0;

        const uint64_t grouped_attn = (n_tokens >= 128u) ? 8u : 2u;
        const NSUInteger q2_bit_words = ((NSUInteger)cache_cap + 31u) / 32u;
        const NSUInteger q2_scratch_bytes =
            q2_bit_words * sizeof(uint32_t) +
            4u * ((NSUInteger)kv_lora_dim + (NSUInteger)qk_rope) * sizeof(uint16_t) +
            8u * (NSUInteger)kv_lora_dim * sizeof(float);
        const NSUInteger max_tg_mem = [g_device maxThreadgroupMemoryLength];
        const bool q2_fits = max_tg_mem == 0 || q2_scratch_bytes <= max_tg_mem;
        const bool use_q2_group4 = grouped_attn == 4u && n_tokens >= 2u && q2_fits;
        const bool use_group8 =
            grouped_attn == 8u ||
            (grouped_attn == 4u && !use_q2_group4);
        const bool use_group2 = grouped_attn == 1u || grouped_attn == 2u;
        id<MTLComputePipelineState> pipeline = use_q2_group4 ?
            ds4_gpu_hot_pipeline(g_glm_attention_indexed_batch_q2_group4_pipeline,
                                 "kernel_glm_attention_indexed_batch_q2_group4") :
            (use_group8 ?
            ds4_gpu_hot_pipeline(g_glm_attention_indexed_batch_group8_pipeline,
                                 "kernel_glm_attention_indexed_batch_group8") :
            (use_group2 ?
            ds4_gpu_hot_pipeline(g_glm_attention_indexed_batch_group2_pipeline,
                                 "kernel_glm_attention_indexed_batch_group2") :
            ds4_gpu_hot_pipeline(g_glm_attention_indexed_batch_pipeline,
                                 "kernel_glm_attention_indexed_batch")));
        if (!pipeline) return 0;

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        ds4_gpu_glm_attention_indexed_batch_args args = {
            .n_tokens = n_tokens,
            .n_selected = n_selected,
            .cache_cap = cache_cap,
            .cache_f16 = cache_f16 ? 1u : 0u,
            .n_head = n_head,
            .kv_lora_dim = kv_lora_dim,
            .qk_nope = qk_nope,
            .qk_rope = qk_rope,
            .value_dim = value_dim,
            .n_ctx_orig = n_ctx_orig,
            .value_row_bytes = (uint32_t)value_row_bytes,
            .value_type = value_weight_type,
            .scale = 1.0f / sqrtf((float)qk_dim),
            .freq_base = freq_base,
            .freq_scale = freq_scale,
            .ext_factor = ext_factor,
            .attn_factor = attn_factor,
            .beta_fast = beta_fast,
            .beta_slow = beta_slow,
            .head_base = 0,
        };
        const NSUInteger scratch_bytes = use_q2_group4 ?
            q2_scratch_bytes :
            (use_group8 ?
             (8u * ((NSUInteger)kv_lora_dim + (NSUInteger)qk_rope) * sizeof(uint16_t) +
              8u * (NSUInteger)kv_lora_dim * sizeof(float)) :
            ((use_group2 ?
             (512u + 2u * (NSUInteger)n_selected + 2u * (NSUInteger)kv_lora_dim) :
             (256u + (NSUInteger)n_selected + (NSUInteger)kv_lora_dim)) * sizeof(float)));

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:qbuf offset:ds4_gpu_tensor_offset(q) atIndex:1];
        [enc setBuffer:lowbuf offset:ds4_gpu_tensor_offset(qk_low) atIndex:2];
        [enc setBuffer:kvcachebuf offset:ds4_gpu_tensor_offset(kv_lora_cache) atIndex:3];
        [enc setBuffer:ropecachebuf offset:ds4_gpu_tensor_offset(k_rope_cache) atIndex:4];
        [enc setBuffer:valuebuf offset:(NSUInteger)value_inner atIndex:5];
        [enc setBuffer:selectedbuf offset:ds4_gpu_tensor_offset(selected) atIndex:6];
        [enc setBuffer:headsbuf offset:ds4_gpu_tensor_offset(heads) atIndex:7];
        [enc setThreadgroupMemoryLength:scratch_bytes atIndex:0];
        if (use_q2_group4) {
            [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)n_head + 3u) / 4u,
                                                  ((NSUInteger)n_tokens + 1u) / 2u,
                                                  1)
                 threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
        } else if (use_group8) {
            [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)n_head + 7u) / 8u,
                                                  (NSUInteger)n_tokens,
                                                  1)
                 threadsPerThreadgroup:MTLSizeMake(32, 8, 1)];
        } else if (use_group2) {
            [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)n_head + 1u) / 2u,
                                                  (NSUInteger)n_tokens,
                                                  1)
                 threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        } else {
            [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)n_head, (NSUInteger)n_tokens, 1)
                 threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "GLM indexed batch attention")) return 0;
    }

    return 1;
}

int ds4_gpu_glm_attention_indexed_batch_tensor(
        ds4_gpu_tensor       *heads,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *qk_low,
        const ds4_gpu_tensor *kv_lora_cache,
        const ds4_gpu_tensor *k_rope_cache,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              value_weight_offset,
        const ds4_gpu_tensor *selected,
        uint32_t              n_tokens,
        uint32_t              n_selected,
        uint32_t              cache_cap,
        bool                  cache_f16,
        uint32_t              n_head,
        uint32_t              kv_lora_dim,
        uint32_t              qk_nope,
        uint32_t              qk_rope,
        uint32_t              value_dim,
        uint32_t              n_ctx_orig,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow) {
    return ds4_gpu_glm_attention_indexed_batch_typed_tensor(heads,
                                                            q,
                                                            qk_low,
                                                            kv_lora_cache,
                                                            k_rope_cache,
                                                            model_map,
                                                            model_size,
                                                            value_weight_offset,
                                                            DS4_METAL_TENSOR_Q8_0,
                                                            selected,
                                                            n_tokens,
                                                            n_selected,
                                                            cache_cap,
                                                            cache_f16,
                                                            n_head,
                                                            kv_lora_dim,
                                                            qk_nope,
                                                            qk_rope,
                                                            value_dim,
                                                            n_ctx_orig,
                                                            freq_base,
                                                            freq_scale,
                                                            ext_factor,
                                                            attn_factor,
                                                            beta_fast,
                                                            beta_slow);
}

int ds4_gpu_sort_i32_rows_asc_tensor(
        ds4_gpu_tensor       *dst,
        const ds4_gpu_tensor *src,
        uint32_t              row_width,
        uint32_t              n_rows) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!dst || !src || row_width == 0 || n_rows == 0 ||
        (row_width & (row_width - 1u)) != 0) {
        return 0;
    }

    @autoreleasepool {
        id<MTLComputePipelineState> pipeline =
            ds4_gpu_hot_pipeline(g_dsv4_sort_i32_rows_asc_pipeline,
                                 "kernel_dsv4_sort_i32_rows_asc");
        if (!pipeline) return 0;

        const uint64_t bytes = (uint64_t)row_width * n_rows * sizeof(int32_t);
        id<MTLBuffer> srcbuf = ds4_gpu_tensor_buffer(src);
        id<MTLBuffer> dstbuf = ds4_gpu_tensor_buffer(dst);
        if (!srcbuf || !dstbuf ||
            ds4_gpu_tensor_bytes(src) < bytes ||
            ds4_gpu_tensor_bytes(dst) < bytes) {
            fprintf(stderr, "ds4: Metal row sort received undersized buffers\n");
            return 0;
        }

        NSUInteger threads = (NSUInteger)row_width;
        const NSUInteger max_threads = pipeline.maxTotalThreadsPerThreadgroup;
        if (max_threads != 0 && threads > max_threads) threads = max_threads;
        if (threads == 0) return 0;

        const NSUInteger scratch_bytes = (NSUInteger)row_width * sizeof(int32_t);
        const NSUInteger max_tg_mem = [g_device maxThreadgroupMemoryLength];
        if (max_tg_mem != 0 && scratch_bytes > max_tg_mem) {
            fprintf(stderr, "ds4: Metal row sort scratch exceeds threadgroup memory limit\n");
            return 0;
        }

        ds4_gpu_dsv4_topk_mask_args args = {
            .ne00 = (int64_t)row_width,
            .ne01 = (int64_t)n_rows,
            .nb00 = sizeof(int32_t),
            .nb01 = (uint64_t)row_width * sizeof(int32_t),
            .ne0 = (int64_t)row_width,
            .ne1 = (int64_t)n_rows,
            .nb0 = sizeof(int32_t),
            .nb1 = (uint64_t)row_width * sizeof(int32_t),
        };

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:srcbuf offset:ds4_gpu_tensor_offset(src) atIndex:1];
        [enc setBuffer:dstbuf offset:ds4_gpu_tensor_offset(dst) atIndex:2];
        [enc setThreadgroupMemoryLength:scratch_bytes atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)n_rows, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "sort i32 rows asc")) return 0;
    }

    return 1;
}

static int ds4_gpu_glm_attention_indexed_batch_lora_layout_tensor(
        ds4_gpu_tensor       *lora_out,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *qk_low,
        const ds4_gpu_tensor *kv_lora_cache,
        const ds4_gpu_tensor *k_rope_cache,
        const ds4_gpu_tensor *selected,
        uint32_t              n_tokens,
        uint32_t              n_selected,
        uint32_t              cache_cap,
        bool                  cache_f16,
        uint32_t              n_head,
        uint32_t              kv_lora_dim,
        uint32_t              qk_nope,
        uint32_t              qk_rope,
        uint32_t              n_ctx_orig,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow,
        bool                  selected_rows_valid) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    const uint32_t qk_dim = qk_nope + qk_rope;
    if (!lora_out || !q || !qk_low || !kv_lora_cache || !k_rope_cache || !selected ||
        n_tokens == 0 || n_selected == 0 || cache_cap == 0 || n_selected > cache_cap ||
        n_head == 0 || kv_lora_dim == 0 ||
        qk_nope == 0 || qk_rope == 0 || (qk_rope & 1u) != 0 ||
        qk_dim < qk_nope ||
        !isfinite(freq_base) || freq_base <= 0.0f ||
        !isfinite(freq_scale) || freq_scale <= 0.0f ||
        !isfinite(ext_factor) || !isfinite(attn_factor) ||
        !isfinite(beta_fast) || !isfinite(beta_slow)) {
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> lorabuf = ds4_gpu_tensor_buffer(lora_out);
        id<MTLBuffer> qbuf = ds4_gpu_tensor_buffer(q);
        id<MTLBuffer> lowbuf = ds4_gpu_tensor_buffer(qk_low);
        id<MTLBuffer> kvcachebuf = ds4_gpu_tensor_buffer(kv_lora_cache);
        id<MTLBuffer> ropecachebuf = ds4_gpu_tensor_buffer(k_rope_cache);
        id<MTLBuffer> selectedbuf = ds4_gpu_tensor_buffer(selected);
        const uint64_t cache_elem_bytes = cache_f16 ? sizeof(uint16_t) : sizeof(float);
        const uint64_t lora_bytes =
            (uint64_t)n_tokens * n_head * kv_lora_dim * sizeof(float);
        const uint64_t q_bytes =
            (uint64_t)n_tokens * n_head * qk_dim * sizeof(float);
        const uint64_t low_bytes =
            (uint64_t)n_tokens * n_head * kv_lora_dim * sizeof(float);
        const uint64_t kv_cache_bytes =
            (uint64_t)cache_cap * kv_lora_dim * cache_elem_bytes;
        const uint64_t rope_cache_bytes =
            (uint64_t)cache_cap * qk_rope * cache_elem_bytes;
        const uint64_t selected_bytes =
            (uint64_t)n_tokens * n_selected * sizeof(uint32_t);
        if (!lorabuf || !qbuf || !lowbuf || !kvcachebuf || !ropecachebuf || !selectedbuf ||
            ds4_gpu_tensor_bytes(lora_out) < lora_bytes ||
            ds4_gpu_tensor_bytes(q) < q_bytes ||
            ds4_gpu_tensor_bytes(qk_low) < low_bytes ||
            ds4_gpu_tensor_bytes(kv_lora_cache) < kv_cache_bytes ||
            ds4_gpu_tensor_bytes(k_rope_cache) < rope_cache_bytes ||
            ds4_gpu_tensor_bytes(selected) < selected_bytes) {
            fprintf(stderr, "ds4: Metal GLM indexed batch attention-lora received undersized buffers\n");
            return 0;
        }

        const bool use_vec_lora =
            cache_f16 && kv_lora_dim == 512u && qk_rope == 64u;
        const bool full_head_groups = (n_head % 8u) == 0u;
        id<MTLComputePipelineState> pipeline = nil;
        if (use_vec_lora && selected_rows_valid && full_head_groups) {
            pipeline = ds4_gpu_hot_pipeline(
                    g_glm_attention_indexed_batch_lora_group8_vec_valid_fullheads_pipeline,
                    "kernel_glm_attention_indexed_batch_lora_group8_vec_valid_fullheads");
        } else if (use_vec_lora && selected_rows_valid) {
            pipeline = ds4_gpu_hot_pipeline(
                    g_glm_attention_indexed_batch_lora_group8_vec_valid_pipeline,
                    "kernel_glm_attention_indexed_batch_lora_group8_vec_valid");
        } else if (use_vec_lora) {
            pipeline = ds4_gpu_hot_pipeline(
                    g_glm_attention_indexed_batch_lora_group8_vec_pipeline,
                    "kernel_glm_attention_indexed_batch_lora_group8_vec");
        } else {
            pipeline = ds4_gpu_hot_pipeline(g_glm_attention_indexed_batch_group8_pipeline,
                                            "kernel_glm_attention_indexed_batch_group8");
        }
        if (!pipeline) return 0;

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        ds4_gpu_glm_attention_indexed_batch_args args = {
            .n_tokens = n_tokens,
            .n_selected = n_selected,
            .cache_cap = cache_cap,
            .cache_f16 = cache_f16 ? 1u : 0u,
            .n_head = n_head,
            .kv_lora_dim = kv_lora_dim,
            .qk_nope = qk_nope,
            .qk_rope = qk_rope,
            .value_dim = kv_lora_dim,
            .n_ctx_orig = n_ctx_orig,
            .value_row_bytes = 0,
            .value_type = 1u,
            .scale = 1.0f / sqrtf((float)qk_dim),
            .freq_base = freq_base,
            .freq_scale = freq_scale,
            .ext_factor = ext_factor,
            .attn_factor = attn_factor,
            .beta_fast = beta_fast,
            .beta_slow = beta_slow,
            .head_base = 0,
        };
        uint32_t head_count = n_head;
        ds4_gpu_tp_attn_head_range(n_head, 8u, &args.head_base, &head_count);
        const NSUInteger scratch_bytes = use_vec_lora ?
            (16u * ((NSUInteger)kv_lora_dim / 4u) * sizeof(uint16_t) * 4u +
             16u * ((NSUInteger)qk_rope / 4u) * sizeof(float) * 4u) :
            (8u * ((NSUInteger)kv_lora_dim + (NSUInteger)qk_rope) * sizeof(uint16_t) +
             8u * (NSUInteger)kv_lora_dim * sizeof(float));

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:qbuf offset:ds4_gpu_tensor_offset(q) atIndex:1];
        [enc setBuffer:lowbuf offset:ds4_gpu_tensor_offset(qk_low) atIndex:2];
        [enc setBuffer:kvcachebuf offset:ds4_gpu_tensor_offset(kv_lora_cache) atIndex:3];
        [enc setBuffer:ropecachebuf offset:ds4_gpu_tensor_offset(k_rope_cache) atIndex:4];
        if (use_vec_lora) {
            [enc setBuffer:selectedbuf offset:ds4_gpu_tensor_offset(selected) atIndex:5];
            [enc setBuffer:lorabuf offset:ds4_gpu_tensor_offset(lora_out) atIndex:6];
        } else {
            [enc setBuffer:lorabuf offset:ds4_gpu_tensor_offset(lora_out) atIndex:5];
            [enc setBuffer:selectedbuf offset:ds4_gpu_tensor_offset(selected) atIndex:6];
            [enc setBuffer:lorabuf offset:ds4_gpu_tensor_offset(lora_out) atIndex:7];
        }
        [enc setThreadgroupMemoryLength:scratch_bytes atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)head_count + 7u) / 8u,
                                              (NSUInteger)n_tokens,
                                              1)
             threadsPerThreadgroup:MTLSizeMake(32, 8, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "GLM indexed batch attention-lora")) return 0;
    }

    return 1;
}

int ds4_gpu_glm_attention_indexed_batch_lora_causal_tensor(
        ds4_gpu_tensor       *lora_out,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *qk_low,
        const ds4_gpu_tensor *kv_lora_cache,
        const ds4_gpu_tensor *k_rope_cache,
        uint32_t              n_tokens,
        uint32_t              pos0,
        uint32_t              n_selected,
        uint32_t              cache_cap,
        bool                  cache_f16,
        uint32_t              n_head,
        uint32_t              kv_lora_dim,
        uint32_t              qk_nope,
        uint32_t              qk_rope,
        uint32_t              n_ctx_orig,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    const uint32_t qk_dim = qk_nope + qk_rope;
    if (!lora_out || !q || !qk_low || !kv_lora_cache || !k_rope_cache ||
        n_tokens == 0 || n_selected == 0 || cache_cap == 0 || n_selected > cache_cap ||
        pos0 > n_selected || n_tokens > n_selected - pos0 ||
        n_head == 0 || kv_lora_dim != 512u ||
        qk_nope == 0 || qk_rope != 64u ||
        qk_dim < qk_nope || !cache_f16 ||
        !isfinite(freq_base) || freq_base <= 0.0f ||
        !isfinite(freq_scale) || freq_scale <= 0.0f ||
        !isfinite(ext_factor) || !isfinite(attn_factor) ||
        !isfinite(beta_fast) || !isfinite(beta_slow)) {
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> lorabuf = ds4_gpu_tensor_buffer(lora_out);
        id<MTLBuffer> qbuf = ds4_gpu_tensor_buffer(q);
        id<MTLBuffer> lowbuf = ds4_gpu_tensor_buffer(qk_low);
        id<MTLBuffer> kvcachebuf = ds4_gpu_tensor_buffer(kv_lora_cache);
        id<MTLBuffer> ropecachebuf = ds4_gpu_tensor_buffer(k_rope_cache);
        const uint64_t cache_elem_bytes = sizeof(uint16_t);
        const uint64_t lora_bytes =
            (uint64_t)n_tokens * n_head * kv_lora_dim * sizeof(float);
        const uint64_t q_bytes =
            (uint64_t)n_tokens * n_head * qk_dim * sizeof(float);
        const uint64_t low_bytes =
            (uint64_t)n_tokens * n_head * kv_lora_dim * sizeof(float);
        const uint64_t kv_cache_bytes =
            (uint64_t)cache_cap * kv_lora_dim * cache_elem_bytes;
        const uint64_t rope_cache_bytes =
            (uint64_t)cache_cap * qk_rope * cache_elem_bytes;
        if (!lorabuf || !qbuf || !lowbuf || !kvcachebuf || !ropecachebuf ||
            ds4_gpu_tensor_bytes(lora_out) < lora_bytes ||
            ds4_gpu_tensor_bytes(q) < q_bytes ||
            ds4_gpu_tensor_bytes(qk_low) < low_bytes ||
            ds4_gpu_tensor_bytes(kv_lora_cache) < kv_cache_bytes ||
            ds4_gpu_tensor_bytes(k_rope_cache) < rope_cache_bytes) {
            fprintf(stderr, "ds4: Metal GLM causal batch attention-lora received undersized buffers\n");
            return 0;
        }

        const bool full_head_groups = (n_head % 8u) == 0u;
        id<MTLComputePipelineState> pipeline = full_head_groups ?
            ds4_gpu_hot_pipeline(
                    g_glm_attention_indexed_batch_lora_group8_vec_causal_fullheads_pipeline,
                    "kernel_glm_attention_indexed_batch_lora_group8_vec_causal_fullheads") :
            ds4_gpu_hot_pipeline(
                    g_glm_attention_indexed_batch_lora_group8_vec_causal_pipeline,
                    "kernel_glm_attention_indexed_batch_lora_group8_vec_causal");
        if (!pipeline) return 0;

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        ds4_gpu_glm_attention_indexed_batch_args args = {
            .n_tokens = n_tokens,
            .n_selected = n_selected,
            .cache_cap = cache_cap,
            .cache_f16 = 1u,
            .n_head = n_head,
            .kv_lora_dim = kv_lora_dim,
            .qk_nope = qk_nope,
            .qk_rope = qk_rope,
            .value_dim = kv_lora_dim,
            .n_ctx_orig = n_ctx_orig,
            .value_row_bytes = 0,
            .value_type = 1u,
            .pos0 = pos0,
            .scale = 1.0f / sqrtf((float)qk_dim),
            .freq_base = freq_base,
            .freq_scale = freq_scale,
            .ext_factor = ext_factor,
            .attn_factor = attn_factor,
            .beta_fast = beta_fast,
            .beta_slow = beta_slow,
            .head_base = 0,
        };
        uint32_t head_count = n_head;
        ds4_gpu_tp_attn_head_range(n_head, 8u, &args.head_base, &head_count);
        const NSUInteger scratch_bytes =
            16u * ((NSUInteger)kv_lora_dim / 4u) * sizeof(uint16_t) * 4u +
            16u * ((NSUInteger)qk_rope / 4u) * sizeof(float) * 4u;

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:qbuf offset:ds4_gpu_tensor_offset(q) atIndex:1];
        [enc setBuffer:lowbuf offset:ds4_gpu_tensor_offset(qk_low) atIndex:2];
        [enc setBuffer:kvcachebuf offset:ds4_gpu_tensor_offset(kv_lora_cache) atIndex:3];
        [enc setBuffer:ropecachebuf offset:ds4_gpu_tensor_offset(k_rope_cache) atIndex:4];
        [enc setBuffer:lorabuf offset:ds4_gpu_tensor_offset(lora_out) atIndex:5];
        [enc setThreadgroupMemoryLength:scratch_bytes atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)head_count + 7u) / 8u,
                                              (NSUInteger)n_tokens,
                                              1)
             threadsPerThreadgroup:MTLSizeMake(32, 8, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "GLM causal indexed batch attention-lora")) return 0;
    }

    return 1;
}

int ds4_gpu_glm_attention_indexed_batch_lora_tensor(
        ds4_gpu_tensor       *lora_out,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *qk_low,
        const ds4_gpu_tensor *kv_lora_cache,
        const ds4_gpu_tensor *k_rope_cache,
        const ds4_gpu_tensor *selected,
        uint32_t              n_tokens,
        uint32_t              n_selected,
        uint32_t              cache_cap,
        bool                  cache_f16,
        uint32_t              n_head,
        uint32_t              kv_lora_dim,
        uint32_t              qk_nope,
        uint32_t              qk_rope,
        uint32_t              n_ctx_orig,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow) {
    return ds4_gpu_glm_attention_indexed_batch_lora_layout_tensor(lora_out,
                                                                 q,
                                                                 qk_low,
                                                                 kv_lora_cache,
                                                                 k_rope_cache,
                                                                 selected,
                                                                 n_tokens,
                                                                 n_selected,
                                                                 cache_cap,
                                                                 cache_f16,
                                                                 n_head,
                                                                 kv_lora_dim,
                                                                 qk_nope,
                                                                 qk_rope,
                                                                 n_ctx_orig,
                                                                 freq_base,
                                                                 freq_scale,
                                                                 ext_factor,
                                                                 attn_factor,
                                                                 beta_fast,
                                                                 beta_slow,
                                                                 false);
}

int ds4_gpu_glm_attention_indexed_batch_lora_valid_tensor(
        ds4_gpu_tensor       *lora_out,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *qk_low,
        const ds4_gpu_tensor *kv_lora_cache,
        const ds4_gpu_tensor *k_rope_cache,
        const ds4_gpu_tensor *selected,
        uint32_t              n_tokens,
        uint32_t              n_selected,
        uint32_t              cache_cap,
        bool                  cache_f16,
        uint32_t              n_head,
        uint32_t              kv_lora_dim,
        uint32_t              qk_nope,
        uint32_t              qk_rope,
        uint32_t              n_ctx_orig,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow) {
    return ds4_gpu_glm_attention_indexed_batch_lora_layout_tensor(lora_out,
                                                                 q,
                                                                 qk_low,
                                                                 kv_lora_cache,
                                                                 k_rope_cache,
                                                                 selected,
                                                                 n_tokens,
                                                                 n_selected,
                                                                 cache_cap,
                                                                 cache_f16,
                                                                 n_head,
                                                                 kv_lora_dim,
                                                                 qk_nope,
                                                                 qk_rope,
                                                                 n_ctx_orig,
                                                                 freq_base,
                                                                 freq_scale,
                                                                 ext_factor,
                                                                 attn_factor,
                                                                 beta_fast,
                                                                 beta_slow,
                                                                 true);
}

int ds4_gpu_glm_router_select_tensor(
        ds4_gpu_tensor       *selected,
        ds4_gpu_tensor       *weights,
        ds4_gpu_tensor       *probs,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                bias_offset,
        const ds4_gpu_tensor *logits,
        uint32_t                n_expert,
        uint32_t                n_expert_used,
        float                   expert_weight_scale) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!selected || !weights || !probs || !logits || !model_map ||
        n_expert == 0 || n_expert > 256u ||
        n_expert_used == 0 || n_expert_used > n_expert) {
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> logitsbuf = ds4_gpu_tensor_buffer(logits);
        id<MTLBuffer> selectedbuf = ds4_gpu_tensor_buffer(selected);
        id<MTLBuffer> weightsbuf = ds4_gpu_tensor_buffer(weights);
        id<MTLBuffer> probsbuf = ds4_gpu_tensor_buffer(probs);
        if (!logitsbuf || !selectedbuf || !weightsbuf || !probsbuf ||
            ds4_gpu_tensor_bytes(logits) < (uint64_t)n_expert * sizeof(float) ||
            ds4_gpu_tensor_bytes(selected) < (uint64_t)n_expert_used * sizeof(int32_t) ||
            ds4_gpu_tensor_bytes(weights) < (uint64_t)n_expert_used * sizeof(float) ||
            ds4_gpu_tensor_bytes(probs) < (uint64_t)n_expert * sizeof(float)) {
            fprintf(stderr, "ds4: Metal GLM router received undersized buffers\n");
            return 0;
        }

        const uint64_t bias_bytes = (uint64_t)n_expert * sizeof(float);
        if (bias_offset > model_size || bias_bytes > model_size - bias_offset) {
            fprintf(stderr, "ds4: Metal GLM router bias range is outside the mapped model\n");
            return 0;
        }
        const bool exact_bias_view =
            bias_bytes <= (1ull << 20) &&
            getenv("DS4_METAL_DISABLE_DECODE_ROUTER_BIAS_EXACT_VIEWS") == NULL;
        uint64_t bias_inner = 0;
        id<MTLBuffer> biasbuf = exact_bias_view ?
            ds4_gpu_wrap_model_exact_range(model_map,
                                           model_size,
                                           bias_offset,
                                           bias_bytes,
                                           &bias_inner) :
            ds4_gpu_wrap_model_range(model_map,
                                     model_size,
                                     bias_offset,
                                     bias_bytes,
                                     &bias_inner);
        if (!biasbuf) return 0;

        id<MTLComputePipelineState> pipeline =
            ds4_gpu_hot_pipeline(g_glm_router_select_one_pipeline,
                                 "kernel_glm_router_select_one");
        if (!pipeline) return 0;

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        ds4_gpu_glm_router_select_one_args args = {
            .n_expert = n_expert,
            .n_expert_used = n_expert_used,
            .expert_weight_scale = expert_weight_scale,
            .pad0 = 0,
        };

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:logitsbuf offset:ds4_gpu_tensor_offset(logits) atIndex:1];
        [enc setBuffer:biasbuf offset:(NSUInteger)bias_inner atIndex:2];
        [enc setBuffer:selectedbuf offset:ds4_gpu_tensor_offset(selected) atIndex:3];
        [enc setBuffer:weightsbuf offset:ds4_gpu_tensor_offset(weights) atIndex:4];
        [enc setBuffer:probsbuf offset:ds4_gpu_tensor_offset(probs) atIndex:5];
        [enc setThreadgroupMemoryLength:256u * sizeof(float) + 256u * sizeof(int32_t) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "GLM router select")) return 0;
    }

    return 1;
}

int ds4_gpu_glm_router_select_batch_tensor(
        ds4_gpu_tensor       *selected,
        ds4_gpu_tensor       *weights,
        ds4_gpu_tensor       *probs,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                bias_offset,
        const ds4_gpu_tensor *logits,
        uint32_t                n_expert,
        uint32_t                n_expert_used,
        float                   expert_weight_scale,
        uint32_t                n_tokens) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!selected || !weights || !probs || !logits || !model_map ||
        n_tokens == 0 ||
        n_expert == 0 || n_expert > 256u ||
        n_expert_used == 0 || n_expert_used > n_expert) {
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> logitsbuf = ds4_gpu_tensor_buffer(logits);
        id<MTLBuffer> selectedbuf = ds4_gpu_tensor_buffer(selected);
        id<MTLBuffer> weightsbuf = ds4_gpu_tensor_buffer(weights);
        id<MTLBuffer> probsbuf = ds4_gpu_tensor_buffer(probs);
        const uint64_t logits_bytes = (uint64_t)n_tokens * n_expert * sizeof(float);
        const uint64_t selected_bytes = (uint64_t)n_tokens * n_expert_used * sizeof(int32_t);
        const uint64_t weights_bytes = (uint64_t)n_tokens * n_expert_used * sizeof(float);
        const uint64_t probs_bytes = (uint64_t)n_tokens * n_expert * sizeof(float);
        if (!logitsbuf || !selectedbuf || !weightsbuf || !probsbuf ||
            ds4_gpu_tensor_bytes(logits) < logits_bytes ||
            ds4_gpu_tensor_bytes(selected) < selected_bytes ||
            ds4_gpu_tensor_bytes(weights) < weights_bytes ||
            ds4_gpu_tensor_bytes(probs) < probs_bytes) {
            fprintf(stderr, "ds4: Metal GLM batch router received undersized buffers\n");
            return 0;
        }

        const uint64_t bias_bytes = (uint64_t)n_expert * sizeof(float);
        if (bias_offset > model_size || bias_bytes > model_size - bias_offset) {
            fprintf(stderr, "ds4: Metal GLM batch router bias range is outside the mapped model\n");
            return 0;
        }
        uint64_t bias_inner = 0;
        id<MTLBuffer> biasbuf = ds4_gpu_wrap_model_range(model_map, model_size,
                                                         bias_offset, bias_bytes,
                                                         &bias_inner);
        if (!biasbuf) return 0;

        id<MTLComputePipelineState> pipeline =
            ds4_gpu_hot_pipeline(g_glm_router_select_one_pipeline,
                                 "kernel_glm_router_select_one");
        if (!pipeline) return 0;

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        ds4_gpu_glm_router_select_one_args args = {
            .n_expert = n_expert,
            .n_expert_used = n_expert_used,
            .expert_weight_scale = expert_weight_scale,
            .pad0 = 0,
        };

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:logitsbuf offset:ds4_gpu_tensor_offset(logits) atIndex:1];
        [enc setBuffer:biasbuf offset:(NSUInteger)bias_inner atIndex:2];
        [enc setBuffer:selectedbuf offset:ds4_gpu_tensor_offset(selected) atIndex:3];
        [enc setBuffer:weightsbuf offset:ds4_gpu_tensor_offset(weights) atIndex:4];
        [enc setBuffer:probsbuf offset:ds4_gpu_tensor_offset(probs) atIndex:5];
        [enc setThreadgroupMemoryLength:256u * sizeof(float) + 256u * sizeof(int32_t) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)n_tokens, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "GLM batch router select")) return 0;
    }

    return 1;
}

static bool ds4_gpu_glm_gate_pair_type_supported(
        uint32_t gate_type,
        uint32_t up_type) {
    return gate_type == up_type &&
           (gate_type == DS4_METAL_TENSOR_Q2_K ||
            gate_type == DS4_METAL_TENSOR_Q4_K ||
            gate_type == DS4_METAL_TENSOR_Q5_K);
}

static bool ds4_gpu_glm_down_type_supported(uint32_t down_type) {
    return down_type == DS4_METAL_TENSOR_Q2_K ||
           down_type == DS4_METAL_TENSOR_Q4_K ||
           down_type == DS4_METAL_TENSOR_Q5_K ||
           down_type == DS4_METAL_TENSOR_Q6_K;
}

int ds4_gpu_glm_routed_moe_one_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                down_offset,
        uint32_t                gate_type,
        uint32_t                up_type,
        uint32_t                down_type,
        uint64_t                gate_expert_bytes,
        uint64_t                gate_row_bytes,
        uint64_t                up_expert_bytes,
        uint64_t                up_row_bytes,
        uint64_t                down_expert_bytes,
        uint64_t                down_row_bytes,
        uint32_t                expert_in_dim,
        uint32_t                expert_mid_dim,
        uint32_t                out_dim,
        const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *weights,
        uint32_t                n_total_expert,
        uint32_t                n_expert,
        uint32_t                layer_index,
        const ds4_gpu_tensor *x,
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
    up_offset += (uint64_t)first_expert * up_expert_bytes;
    down_offset += (uint64_t)first_expert * down_expert_bytes;

    if (!out || !mid || !model_map || !selected || !weights || !x ||
        n_total_expert == 0 || n_expert == 0 || n_expert > 256u ||
        n_expert > n_total_expert ||
        expert_in_dim == 0 || expert_mid_dim == 0 || out_dim == 0 ||
        gate_expert_bytes == 0 || gate_row_bytes == 0 ||
        up_expert_bytes == 0 || up_row_bytes == 0 ||
        down_expert_bytes == 0 || down_row_bytes == 0 ||
        (expert_in_dim % 256u) != 0 ||
        (expert_mid_dim % 256u) != 0 ||
        !ds4_gpu_glm_gate_pair_type_supported(gate_type, up_type) ||
        !ds4_gpu_glm_down_type_supported(down_type)) {
        return 0;
    }

    if ((uint64_t)n_total_expert > UINT64_MAX / gate_expert_bytes ||
        (uint64_t)n_total_expert > UINT64_MAX / up_expert_bytes ||
        (uint64_t)n_total_expert > UINT64_MAX / down_expert_bytes ||
        (uint64_t)expert_mid_dim > UINT64_MAX / gate_row_bytes ||
        (uint64_t)expert_mid_dim > UINT64_MAX / up_row_bytes ||
        (uint64_t)out_dim > UINT64_MAX / down_row_bytes) {
        fprintf(stderr, "ds4: Metal GLM routed MoE tensor byte size overflow\n");
        return 0;
    }

    const uint64_t gate_tensor_bytes = (uint64_t)n_bind_expert * gate_expert_bytes;
    const uint64_t up_tensor_bytes = (uint64_t)n_bind_expert * up_expert_bytes;
    const uint64_t down_tensor_bytes = (uint64_t)n_bind_expert * down_expert_bytes;
    if (gate_expert_bytes != (uint64_t)expert_mid_dim * gate_row_bytes ||
        up_expert_bytes != (uint64_t)expert_mid_dim * up_row_bytes ||
        down_expert_bytes != (uint64_t)out_dim * down_row_bytes) {
        fprintf(stderr, "ds4: Metal GLM routed MoE received inconsistent expert strides\n");
        return 0;
    }
    if (layer_index == 3u) {
        ds4_gpu_stream_expert_cache_note_decode_token();
    }
    if (gate_offset > model_size || gate_tensor_bytes > model_size - gate_offset ||
        up_offset > model_size || up_tensor_bytes > model_size - up_offset ||
        down_offset > model_size || down_tensor_bytes > model_size - down_offset) {
        fprintf(stderr, "ds4: Metal GLM routed MoE tensor range is outside the mapped model\n");
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> xbuf = ds4_gpu_tensor_buffer(x);
        id<MTLBuffer> midbuf = ds4_gpu_tensor_buffer(mid);
        id<MTLBuffer> outbuf = ds4_gpu_tensor_buffer(out);
        id<MTLBuffer> selectedbuf = ds4_gpu_tensor_buffer(selected);
        id<MTLBuffer> weightsbuf = ds4_gpu_tensor_buffer(weights);
        const uint64_t x_bytes = (uint64_t)expert_in_dim * sizeof(float);
        const uint64_t mid_bytes = (uint64_t)n_expert * expert_mid_dim * sizeof(float);
        const uint64_t out_bytes = (uint64_t)out_dim * sizeof(float);
        if (!xbuf || !midbuf || !outbuf || !selectedbuf || !weightsbuf ||
            ds4_gpu_tensor_bytes(x) < x_bytes ||
            ds4_gpu_tensor_bytes(mid) < mid_bytes ||
            ds4_gpu_tensor_bytes(out) < out_bytes ||
            ds4_gpu_tensor_bytes(selected) < (uint64_t)n_expert * sizeof(int32_t) ||
            ds4_gpu_tensor_bytes(weights) < (uint64_t)n_expert * sizeof(float)) {
            fprintf(stderr, "ds4: Metal GLM routed MoE received undersized activation buffers\n");
            return 0;
        }

        const BOOL gate_pair_q2 = gate_type == DS4_METAL_TENSOR_Q2_K;
        const BOOL gate_pair_q5 = gate_type == DS4_METAL_TENSOR_Q5_K;
        const BOOL down_scalar_q2 = down_type == DS4_METAL_TENSOR_Q2_K;
        const BOOL down_scalar_q4 = down_type == DS4_METAL_TENSOR_Q4_K;
        const BOOL down_simd_q4 = down_scalar_q4;
        const BOOL down_simd_q5 = down_type == DS4_METAL_TENSOR_Q5_K;
        const BOOL down_simd_q6 = down_type == DS4_METAL_TENSOR_Q6_K;
        const BOOL down_simd = down_simd_q4 || down_simd_q5 || down_simd_q6;
        const BOOL stream_addr_q2 =
            gate_pair_q2 && down_scalar_q2 &&
            g_glm_q2_k_addr_pair_swiglu2_f32_pipeline != nil &&
            g_glm_q2_k_addr_down_f32_pipeline != nil;
        const BOOL stream_addr_q4 =
            !gate_pair_q2 && !gate_pair_q5 && down_scalar_q4 &&
            g_glm_q4_k_addr_pair_swiglu_f32_pipeline != nil &&
            g_glm_q4_k_addr_down_f32_pipeline != nil;
        BOOL use_stream_expert_addr_table =
            g_ssd_streaming_mode &&
            !force_resident &&
            getenv("DS4_METAL_GLM_DISABLE_STREAMING_EXPERT_CACHE") == NULL &&
            (stream_addr_q2 || stream_addr_q4) &&
            layer_index < DS4_METAL_STREAM_EXPERT_CACHE_MAX_LAYER &&
            n_total_expert <= DS4_METAL_STREAM_EXPERT_CACHE_MAX_EXPERT &&
            n_expert <= 8u &&
            ds4_gpu_stream_expert_cache_configured_budget() >= n_expert &&
            ds4_gpu_stream_expert_cache_note_expert_size(gate_expert_bytes,
                                                         down_expert_bytes) &&
            ds4_gpu_stream_expert_cache_effective_cap(layer_index,
                                                      n_total_expert,
                                                      n_expert) != 0;
        int32_t stream_selected_ids[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        ds4_gpu_stream_expert_cache_entry *stream_entries[8] = {
            NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL
        };
        uint64_t stream_gate_abs_offsets[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        uint64_t stream_up_abs_offsets[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        uint64_t stream_down_abs_offsets[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        uint32_t stream_missing_mask = 0;
        uint32_t stream_entry_count = 0;
        uint32_t stream_resident_mask = 0;
        BOOL use_stream_split_deferred = false;
        id<MTLBuffer> stream_gate_addr_buf = nil;
        id<MTLBuffer> stream_up_addr_buf = nil;
        id<MTLBuffer> stream_down_addr_buf = nil;
        const int glm_stream_timing =
            ds4_gpu_stream_expert_timing_summary_enabled();

        if (use_stream_expert_addr_table) {
            const int had_batch = g_batch_cb != nil;
            int stream_ok = 1;
            const int have_prefetched_selected =
                ds4_gpu_glm_stream_selected_prefetch_take(model_map,
                                                          model_size,
                                                          layer_index,
                                                          n_total_expert,
                                                          n_expert,
                                                          gate_offset,
                                                          up_offset,
                                                          down_offset,
                                                          gate_expert_bytes,
                                                          down_expert_bytes,
                                                          stream_selected_ids);
            if (have_prefetched_selected) {
                if (had_batch && g_batch_has_work &&
                    g_stream_expert_pending_load.active &&
                    ds4_gpu_flush_commands() == 0) {
                    stream_ok = 0;
                }
            } else {
                if (had_batch && ds4_gpu_end_commands() == 0) {
                    stream_ok = 0;
                }
                if (stream_ok &&
                    ds4_gpu_tensor_read(selected,
                                        0,
                                        stream_selected_ids,
                                        (uint64_t)n_expert * sizeof(stream_selected_ids[0])) == 0) {
                    stream_ok = 0;
                }
            }
            for (uint32_t i = 0; stream_ok && i < n_expert; i++) {
                if (stream_selected_ids[i] < 0 ||
                    (uint32_t)stream_selected_ids[i] >= n_total_expert) {
                    fprintf(stderr,
                            "ds4: Metal GLM routed MoE selected expert id %d is outside 0..%u\n",
                            stream_selected_ids[i],
                            n_total_expert);
                    stream_ok = 0;
                }
            }
            if (stream_ok) {
                ds4_gpu_stream_expert_cache_note_selected_hotness(layer_index,
                                                                  stream_selected_ids,
                                                                  n_expert);
                if (!ds4_gpu_moe_selected_hotlist_record(layer_index,
                                                         stream_selected_ids,
                                                         n_expert,
                                                         n_total_expert)) {
                    stream_ok = 0;
                }
            }
            if (stream_ok) {
                g_glm_stream_expert_addr_table_building++;
                for (uint32_t i = 0; stream_ok && i < n_expert; i++) {
                    const uint64_t expert_id = (uint64_t)(uint32_t)stream_selected_ids[i];
                    if (expert_id > UINT64_MAX / gate_expert_bytes ||
                        expert_id > UINT64_MAX / down_expert_bytes) {
                        fprintf(stderr, "ds4: Metal GLM routed MoE selected expert offset overflow\n");
                        stream_ok = 0;
                        break;
                    }
                    const uint64_t gate_rel = expert_id * gate_expert_bytes;
                    const uint64_t down_rel = expert_id * down_expert_bytes;
                    if (gate_rel > UINT64_MAX - gate_offset ||
                        gate_rel > UINT64_MAX - up_offset ||
                        down_rel > UINT64_MAX - down_offset) {
                        fprintf(stderr, "ds4: Metal GLM routed MoE selected expert offset overflow\n");
                        stream_ok = 0;
                        break;
                    }
                    stream_gate_abs_offsets[i] = gate_offset + gate_rel;
                    stream_up_abs_offsets[i] = up_offset + gate_rel;
                    stream_down_abs_offsets[i] = down_offset + down_rel;
                    stream_entries[i] =
                        ds4_gpu_stream_expert_cache_peek(model_map,
                                                         model_size,
                                                         layer_index,
                                                         (uint32_t)stream_selected_ids[i],
                                                         n_total_expert,
                                                         n_expert,
                                                         stream_gate_abs_offsets[i],
                                                         stream_up_abs_offsets[i],
                                                         stream_down_abs_offsets[i],
                                                         gate_expert_bytes,
                                                         down_expert_bytes);
                    if (!stream_entries[i]) {
                        stream_missing_mask |= 1u << i;
                    } else {
                        stream_resident_mask |= 1u << i;
                    }
                }
                if (stream_ok && glm_stream_timing) {
                    ds4_gpu_stream_expert_timing_note_cache_class(
                            stream_resident_mask,
                            stream_missing_mask);
                }
                use_stream_split_deferred =
                    stream_ok &&
                    getenv("DS4_METAL_DISABLE_GLM_STREAMING_EXPERT_SPLIT") == NULL &&
                    stream_missing_mask != 0 &&
                    stream_resident_mask != 0 &&
                    ds4_gpu_stream_expert_split_worthwhile(stream_resident_mask,
                                                           stream_missing_mask) &&
                    ds4_gpu_stream_expert_split_ready() &&
                    ((gate_pair_q2 &&
                      g_glm_q2_k_addr_pair_swiglu2_masked_f32_pipeline != nil) ||
                     (!gate_pair_q2 && !gate_pair_q5 &&
                      g_glm_q4_k_addr_pair_swiglu_masked_f32_pipeline != nil));
                if (use_stream_split_deferred) {
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
                                stream_selected_ids,
                                n_expert)) {
                        stream_ok = 0;
                    }
                }
                if (stream_ok && stream_missing_mask != 0 &&
                    !use_stream_split_deferred &&
                    !ds4_gpu_stream_expert_cache_load_selected_missing(
                            model_map,
                            model_size,
                            layer_index,
                            stream_selected_ids,
                            n_total_expert,
                            n_expert,
                            stream_gate_abs_offsets,
                            stream_up_abs_offsets,
                            stream_down_abs_offsets,
                            gate_expert_bytes,
                            down_expert_bytes,
                            stream_missing_mask,
                            stream_entries)) {
                    fprintf(stderr,
                            "ds4: Metal GLM streaming expert cache failed to load "
                            "layer=%u missing=0x%x budget=%u\n",
                            layer_index,
                            stream_missing_mask,
                            ds4_gpu_stream_expert_cache_configured_budget());
                    stream_ok = 0;
                }
                for (uint32_t i = 0; stream_ok && i < n_expert; i++) {
                    ds4_gpu_stream_expert_cache_entry *entry = stream_entries[i];
                    if (!entry) {
                        if (use_stream_split_deferred &&
                            (stream_missing_mask & (1u << i)) != 0) {
                            continue;
                        }
                        stream_ok = 0;
                        break;
                    }
                    if (!ds4_gpu_stream_expert_cache_set_addr_slot(
                                layer_index,
                                (uint32_t)stream_selected_ids[i],
                                entry->gate_buffer,
                                entry->gate_inner,
                                entry->up_buffer,
                                entry->up_inner,
                                entry->down_buffer,
                                entry->down_inner)) {
                        stream_ok = 0;
                        break;
                    }
                    stream_entry_count++;
                }
                if (stream_ok &&
                    !ds4_gpu_stream_expert_cache_addr_buffers(layer_index,
                                                              &stream_gate_addr_buf,
                                                              &stream_up_addr_buf,
                                                              &stream_down_addr_buf)) {
                    stream_ok = 0;
                }
                g_glm_stream_expert_addr_table_building--;
            }
            if (!have_prefetched_selected && had_batch &&
                ds4_gpu_begin_commands() == 0) {
                stream_ok = 0;
            }
            if (!stream_ok) return 0;
            ds4_gpu_stream_expert_cache_prune_layer(layer_index,
                                                    n_total_expert,
                                                    n_expert,
                                                    stream_selected_ids,
                                                    n_expert);
            ds4_gpu_stream_expert_cache_prune_global(layer_index,
                                                     stream_selected_ids,
                                                     n_expert);
        }

        uint64_t gate_inner = 0;
        uint64_t up_inner = 0;
        uint64_t down_inner = 0;
        id<MTLBuffer> gatebuf = nil;
        id<MTLBuffer> upbuf = nil;
        id<MTLBuffer> downbuf = nil;
        if (!use_stream_expert_addr_table) {
            gatebuf = ds4_gpu_wrap_model_range(model_map, model_size,
                                               gate_offset, gate_tensor_bytes,
                                               &gate_inner);
            upbuf = ds4_gpu_wrap_model_range(model_map, model_size,
                                             up_offset, up_tensor_bytes,
                                             &up_inner);
            downbuf = ds4_gpu_wrap_model_range(model_map, model_size,
                                               down_offset, down_tensor_bytes,
                                               &down_inner);
            if (!gatebuf || !upbuf || !downbuf) return 0;
        }

        id<MTLComputePipelineState> pair_pipeline =
            (use_stream_split_deferred ?
             (gate_pair_q2 ?
              ds4_gpu_hot_pipeline(g_glm_q2_k_addr_pair_swiglu2_masked_f32_pipeline,
                                   "kernel_glm_q2_K_addr_pair_swiglu2_f32_masked") :
              ds4_gpu_hot_pipeline(g_glm_q4_k_addr_pair_swiglu_masked_f32_pipeline,
                                   "kernel_glm_q4_K_addr_pair_swiglu_f32_masked")) :
             use_stream_expert_addr_table ?
             (gate_pair_q2 ?
              ds4_gpu_hot_pipeline(g_glm_q2_k_addr_pair_swiglu2_f32_pipeline,
                                   "kernel_glm_q2_K_addr_pair_swiglu2_f32") :
              ds4_gpu_hot_pipeline(g_glm_q4_k_addr_pair_swiglu_f32_pipeline,
                                   "kernel_glm_q4_K_addr_pair_swiglu_f32")) :
             gate_pair_q2 ?
             ds4_gpu_hot_pipeline(g_glm_q2_k_pair_swiglu_f32_pipeline,
                                  "kernel_glm_q2_K_pair_swiglu_f32") :
             gate_pair_q5 ?
             ds4_gpu_hot_pipeline(g_glm_q5_k_pair_swiglu_f32_pipeline,
                                  "kernel_glm_q5_K_pair_swiglu_f32") :
             ds4_gpu_hot_pipeline(g_glm_q4_k_pair_swiglu2_f32_pipeline,
                                  "kernel_glm_q4_K_pair_swiglu2_f32"));
        id<MTLComputePipelineState> down_pipeline =
            (use_stream_expert_addr_table ?
             (down_scalar_q2 ?
              ds4_gpu_hot_pipeline(g_glm_q2_k_addr_down_f32_pipeline,
                                   "kernel_glm_q2_K_addr_down_f32") :
              ds4_gpu_hot_pipeline(g_glm_q4_k_addr_down_f32_pipeline,
                                   "kernel_glm_q4_K_addr_down_f32")) :
             down_scalar_q2 ?
             ds4_gpu_hot_pipeline(g_glm_q2_k_down_f32_pipeline,
                                  "kernel_glm_q2_K_down_f32") :
             down_scalar_q4 ?
             ds4_gpu_hot_pipeline(g_glm_q4_k_down_f32_pipeline,
                                  "kernel_glm_q4_K_down_f32") :
             down_simd_q5 ?
             ds4_gpu_hot_pipeline(g_glm_q5_k_down_f32_pipeline,
                                  "kernel_glm_q5_K_down_f32") :
             ds4_gpu_hot_pipeline(g_glm_q6_k_down_f32_pipeline,
                                  "kernel_glm_q6_K_down_f32"));
        if (!pair_pipeline || !down_pipeline) return 0;

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        const bool glm_moe_stage_profile =
            g_batch_cb != nil &&
            ds4_gpu_stage_profile_enabled_for_layer(
                    "DS4_METAL_GLM_MOE_ONE_STAGE_PROFILE",
                    "DS4_METAL_GLM_MOE_ONE_STAGE_PROFILE_LAYER",
                    layer_index);
        const char *glm_moe_stage_filter =
            getenv("DS4_METAL_GLM_MOE_STAGE_PROFILE_FILTER");
        const char *glm_pair_path =
            use_stream_expert_addr_table ?
            (gate_pair_q2 ? "q2_stream_addr_swiglu" :
                            "q4_stream_addr_swiglu") :
            gate_pair_q2 ? "q2_scalar_swiglu" :
            gate_pair_q5 ? "q5_pair_simd_swiglu" : "q4_pair2_simd_swiglu";
        const char *glm_down_path =
            use_stream_expert_addr_table ?
            (down_scalar_q2 ? "q2_stream_addr_down" :
                              "q4_stream_addr_down_simd") :
            down_scalar_q2 ? "q2_down_simd" :
            down_scalar_q4 ? "q4_down_simd" :
            down_simd_q5 ? "q5_down_simd" : "q6_down_simd";
        double glm_moe_stage_t0 = 0.0;
        if (glm_moe_stage_profile) {
            if (ds4_gpu_end_commands() == 0 || ds4_gpu_begin_commands() == 0) {
                return 0;
            }
            cb = ds4_gpu_command_buffer(&owned);
            if (!cb) return 0;
            glm_moe_stage_t0 = ds4_gpu_now_ms();
        }
        int ok = 1;
#define DS4_METAL_PROFILE_GLM_MOE_ONE_STAGE(name) do { \
            if (ok && glm_moe_stage_profile) { \
                if (ds4_gpu_end_commands() == 0) { \
                    ok = 0; \
                } else { \
                    const char *stage_name = (name); \
                    const double now_ms = ds4_gpu_now_ms(); \
                    const int print_stage = \
                        !glm_moe_stage_filter || !glm_moe_stage_filter[0] || \
                        strstr(stage_name, glm_moe_stage_filter) != NULL; \
                    if (print_stage) { \
                        fprintf(stderr, \
                                "ds4: Metal GLM routed MoE one stage layer=%u tokens=1 experts=%u " \
                                "gate=%s down=%s pair=%s down_path=%s %s=%.3f ms\n", \
                                layer_index, n_expert, \
                                ds4_gpu_metal_tensor_type_name(gate_type), \
                                ds4_gpu_metal_tensor_type_name(down_type), \
                                glm_pair_path, glm_down_path, \
                                stage_name, now_ms - glm_moe_stage_t0); \
                    } \
                    glm_moe_stage_t0 = now_ms; \
                    if (ds4_gpu_begin_commands() == 0) { \
                        ok = 0; \
                    } else { \
                        cb = ds4_gpu_command_buffer(&owned); \
                        if (!cb) ok = 0; \
                    } \
                } \
            } \
        } while (0)

        ds4_gpu_glm_routed_moe_args args = {
            .tp_rank = g_tp_split_rank,
            .tp_world = g_tp_split_world,
            .tp_expert_base = tp_expert_base_host,
            .in_dim = expert_in_dim,
            .mid_dim = expert_mid_dim,
            .out_dim = out_dim,
            .n_total_expert = n_total_expert,
            .n_expert_used = n_expert,
            .n_tokens = 1,
            .mid_token_stride = n_expert * expert_mid_dim,
            .down_type = down_type,
            .gate_expert_bytes = gate_expert_bytes,
            .gate_row_bytes = gate_row_bytes,
            .up_expert_bytes = up_expert_bytes,
            .up_row_bytes = up_row_bytes,
            .down_expert_bytes = down_expert_bytes,
            .down_row_bytes = down_row_bytes,
        };
        const NSUInteger pair_x_groups =
            gate_pair_q2 ? (use_stream_expert_addr_table ?
                            (NSUInteger)((expert_mid_dim + 1u) / 2u) :
                            (NSUInteger)((expert_mid_dim + 7u) / 8u)) :
            gate_pair_q5 ? (NSUInteger)((expert_mid_dim + 7u) / 8u) :
            use_stream_expert_addr_table ? (NSUInteger)((expert_mid_dim + 3u) / 4u) :
            (NSUInteger)((expert_mid_dim + 1u) / 2u);
        const NSUInteger pair_threadgroup_bytes = 0u;
        const NSUInteger pair_threads = 64u;
        const NSUInteger down_x_groups =
            down_scalar_q2 ? (NSUInteger)((out_dim + 7u) / 8u) :
            down_simd_q4 ? (NSUInteger)((out_dim + 3u) / 4u) :
            down_simd_q5 ? (NSUInteger)((out_dim + 3u) / 4u) :
            down_simd_q6 ? (NSUInteger)((out_dim + 3u) / 4u) :
            (NSUInteger)out_dim;
        const NSUInteger down_threadgroup_bytes =
            (down_scalar_q2 || down_simd) ? 0u : 256u * sizeof(float);
        const NSUInteger down_threads =
            (down_scalar_q2 || down_simd) ? 64u : 256u;
        if (use_stream_expert_addr_table &&
            !ds4_gpu_stream_expert_cache_mark_entries_inflight(
                    stream_entries,
                    use_stream_split_deferred ? n_expert : stream_entry_count,
                    use_stream_split_deferred ? stream_resident_mask : 0)) {
            return 0;
        }

        const int glm_stream_split_timing =
            use_stream_split_deferred && glm_stream_timing;
        double glm_stream_split_t0 =
            glm_stream_split_timing ? ds4_gpu_now_ms() : 0.0;
        double glm_stream_split_resident_ms = 0.0;
        double glm_stream_split_missing_load_ms = 0.0;
        double glm_stream_split_missing_wait_ms = 0.0;

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pair_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        if (use_stream_split_deferred) {
            [enc setBytes:&stream_resident_mask length:sizeof(stream_resident_mask) atIndex:1];
            [enc setBuffer:stream_gate_addr_buf offset:0u atIndex:2];
            [enc setBuffer:stream_up_addr_buf offset:0u atIndex:3];
            [enc setBuffer:xbuf offset:ds4_gpu_tensor_offset(x) atIndex:4];
            [enc setBuffer:selectedbuf offset:ds4_gpu_tensor_offset(selected) atIndex:5];
            [enc setBuffer:weightsbuf offset:ds4_gpu_tensor_offset(weights) atIndex:6];
            [enc setBuffer:midbuf offset:ds4_gpu_tensor_offset(mid) atIndex:7];
        } else {
            [enc setBuffer:use_stream_expert_addr_table ? stream_gate_addr_buf : gatebuf
                    offset:use_stream_expert_addr_table ? 0u : (NSUInteger)gate_inner
                   atIndex:1];
            [enc setBuffer:use_stream_expert_addr_table ? stream_up_addr_buf : upbuf
                    offset:use_stream_expert_addr_table ? 0u : (NSUInteger)up_inner
                   atIndex:2];
            [enc setBuffer:xbuf offset:ds4_gpu_tensor_offset(x) atIndex:3];
            [enc setBuffer:selectedbuf offset:ds4_gpu_tensor_offset(selected) atIndex:4];
            [enc setBuffer:weightsbuf offset:ds4_gpu_tensor_offset(weights) atIndex:5];
            [enc setBuffer:midbuf offset:ds4_gpu_tensor_offset(mid) atIndex:6];
        }
        if (use_stream_expert_addr_table) {
            const uint32_t use_count =
                use_stream_split_deferred ? n_expert : stream_entry_count;
            const uint32_t use_mask =
                use_stream_split_deferred ? stream_resident_mask : 0;
            for (uint32_t i = 0; i < use_count; i++) {
                if (use_mask != 0 && (use_mask & (1u << i)) == 0) continue;
                [enc useResource:stream_entries[i]->gate_buffer usage:MTLResourceUsageRead];
                [enc useResource:stream_entries[i]->up_buffer usage:MTLResourceUsageRead];
            }
        }
        if (pair_threadgroup_bytes != 0u) {
            [enc setThreadgroupMemoryLength:pair_threadgroup_bytes atIndex:0];
        }
        [enc dispatchThreadgroups:MTLSizeMake(pair_x_groups,
                                              (NSUInteger)n_expert,
                                              1)
             threadsPerThreadgroup:MTLSizeMake(pair_threads, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);
        DS4_METAL_PROFILE_GLM_MOE_ONE_STAGE("pair");

        if (ok && use_stream_split_deferred) {
            id<MTLCommandBuffer> resident_cb = nil;
            const int resident_owned = owned;
            if (owned) {
                resident_cb = cb;
                [resident_cb commit];
                cb = nil;
            } else {
                ok = ds4_gpu_flush_commands();
            }
            if (glm_stream_split_timing) {
                const double now_ms = ds4_gpu_now_ms();
                glm_stream_split_resident_ms = now_ms - glm_stream_split_t0;
                glm_stream_split_t0 = now_ms;
            }
            if (ok) {
                ok = ds4_gpu_stream_expert_pending_load_finish(stream_entries);
            }
            if (glm_stream_split_timing) {
                const double now_ms = ds4_gpu_now_ms();
                glm_stream_split_missing_load_ms = now_ms - glm_stream_split_t0;
                glm_stream_split_t0 = now_ms;
            }
            if (ok) {
                ds4_gpu_stream_expert_cache_prune_layer(layer_index,
                                                        n_total_expert,
                                                        n_expert,
                                                        stream_selected_ids,
                                                        n_expert);
                ds4_gpu_stream_expert_cache_prune_global(layer_index,
                                                         stream_selected_ids,
                                                         n_expert);
                ok = ds4_gpu_stream_expert_cache_addr_buffers(layer_index,
                                                              &stream_gate_addr_buf,
                                                              &stream_up_addr_buf,
                                                              &stream_down_addr_buf);
            }
            if (ok) {
                if (resident_owned) {
                    ok = ds4_gpu_wait_command_buffer(
                            resident_cb,
                            "GLM streaming split resident pair");
                    ds4_gpu_stream_expert_cache_note_owned_completed();
                } else {
                    ok = ds4_gpu_wait_pending_command_buffers(
                            "GLM streaming split resident pair");
                }
            }
            if (glm_stream_split_timing) {
                const double now_ms = ds4_gpu_now_ms();
                glm_stream_split_missing_wait_ms = now_ms - glm_stream_split_t0;
                glm_stream_split_t0 = now_ms;
                ds4_gpu_stream_expert_timing_note_split(
                        stream_resident_mask,
                        stream_missing_mask,
                        glm_stream_split_resident_ms,
                        glm_stream_split_missing_load_ms +
                        glm_stream_split_missing_wait_ms);
                ds4_gpu_stream_expert_timing_note_split_missing_detail(
                        glm_stream_split_missing_load_ms,
                        0.0,
                        0.0,
                        0.0,
                        glm_stream_split_missing_wait_ms);
            }
            if (ok &&
                !ds4_gpu_stream_expert_cache_mark_entries_inflight(stream_entries,
                                                                   n_expert,
                                                                   stream_missing_mask)) {
                ok = 0;
            }
            if (ok) {
                cb = ds4_gpu_command_buffer(&owned);
                if (!cb) ok = 0;
            }
            if (ok) {
                enc = ds4_gpu_compute_encoder(cb);
                [enc setComputePipelineState:pair_pipeline];
                [enc setBytes:&args length:sizeof(args) atIndex:0];
                [enc setBytes:&stream_missing_mask length:sizeof(stream_missing_mask) atIndex:1];
                [enc setBuffer:stream_gate_addr_buf offset:0u atIndex:2];
                [enc setBuffer:stream_up_addr_buf offset:0u atIndex:3];
                [enc setBuffer:xbuf offset:ds4_gpu_tensor_offset(x) atIndex:4];
                [enc setBuffer:selectedbuf offset:ds4_gpu_tensor_offset(selected) atIndex:5];
                [enc setBuffer:weightsbuf offset:ds4_gpu_tensor_offset(weights) atIndex:6];
                [enc setBuffer:midbuf offset:ds4_gpu_tensor_offset(mid) atIndex:7];
                for (uint32_t i = 0; i < n_expert; i++) {
                    if ((stream_missing_mask & (1u << i)) == 0) continue;
                    [enc useResource:stream_entries[i]->gate_buffer usage:MTLResourceUsageRead];
                    [enc useResource:stream_entries[i]->up_buffer usage:MTLResourceUsageRead];
                }
                if (pair_threadgroup_bytes != 0u) {
                    [enc setThreadgroupMemoryLength:pair_threadgroup_bytes atIndex:0];
                }
                [enc dispatchThreadgroups:MTLSizeMake(pair_x_groups,
                                                      (NSUInteger)n_expert,
                                                      1)
                     threadsPerThreadgroup:MTLSizeMake(pair_threads, 1, 1)];
                ds4_gpu_end_compute_encoder(cb, enc);
            }
        }

        if (!ok) return 0;

        enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:down_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:use_stream_expert_addr_table ? stream_down_addr_buf : downbuf
                offset:use_stream_expert_addr_table ? 0u : (NSUInteger)down_inner
               atIndex:1];
        [enc setBuffer:selectedbuf offset:ds4_gpu_tensor_offset(selected) atIndex:2];
        [enc setBuffer:midbuf offset:ds4_gpu_tensor_offset(mid) atIndex:3];
        [enc setBuffer:outbuf offset:ds4_gpu_tensor_offset(out) atIndex:4];
        if (use_stream_expert_addr_table) {
            const uint32_t use_count =
                use_stream_split_deferred ? n_expert : stream_entry_count;
            for (uint32_t i = 0; i < use_count; i++) {
                [enc useResource:stream_entries[i]->down_buffer usage:MTLResourceUsageRead];
            }
        }
        if (down_threadgroup_bytes != 0u) {
            [enc setThreadgroupMemoryLength:down_threadgroup_bytes atIndex:0];
        }
        [enc dispatchThreadgroups:MTLSizeMake(down_x_groups, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(down_threads, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);
        DS4_METAL_PROFILE_GLM_MOE_ONE_STAGE("down");

        if (!ok) return 0;
        if (!ds4_gpu_finish_command_buffer(cb, owned, "GLM routed MoE")) return 0;
#undef DS4_METAL_PROFILE_GLM_MOE_ONE_STAGE
    }

    return 1;
}

static bool ds4_gpu_glm_grouped_moe_fast_default(void) {
    return !g_quality_mode;
}

static bool ds4_gpu_glm_routed_moe_batch_grouped_available(
        uint32_t gate_type,
        uint32_t up_type,
        uint32_t down_type,
        uint32_t n_expert,
        uint32_t n_tokens) {
    if (n_tokens < 32u ||
        !ds4_gpu_glm_gate_pair_type_supported(gate_type, up_type) ||
        !ds4_gpu_glm_down_type_supported(down_type) ||
        ds4_gpu_mul_mm_id_map0_name(n_expert) == NULL) {
        return false;
    }
    const bool fast_default = ds4_gpu_glm_grouped_moe_fast_default();
    if (!fast_default) {
        return false;
    }
    if (n_tokens < 96u) return false;

    return ds4_gpu_get_pipeline(ds4_gpu_mul_mm_id_map0_name(n_expert)) != nil &&
           ds4_gpu_routed_mm_pipeline(gate_type) != nil &&
           ds4_gpu_routed_mm_pipeline(up_type) != nil &&
           ds4_gpu_routed_mm_f16_rhs_pipeline(down_type) != nil;
}

static bool ds4_gpu_glm_grouped_moe_layer_enabled(uint32_t layer_index) {
    (void)layer_index;
    return true;
}

static int ds4_gpu_glm_routed_moe_batch_grouped_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                down_offset,
        uint32_t                gate_type,
        uint32_t                up_type,
        uint32_t                down_type,
        uint64_t                gate_expert_bytes,
        uint64_t                gate_row_bytes,
        uint64_t                up_expert_bytes,
        uint64_t                up_row_bytes,
        uint64_t                down_expert_bytes,
        uint64_t                down_row_bytes,
        uint32_t                expert_in_dim,
        uint32_t                expert_mid_dim,
        uint32_t                out_dim,
        const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *weights,
        uint32_t                n_total_expert,
        uint32_t                n_expert,
        uint32_t                layer_index,
        const ds4_gpu_tensor *x,
        uint32_t                n_tokens) {
    if (!ds4_gpu_glm_routed_moe_batch_grouped_available(gate_type,
                                                         up_type,
                                                         down_type,
                                                         n_expert,
                                                         n_tokens)) {
        return 0;
    }
    if (n_expert > UINT32_MAX / n_tokens ||
        (uint64_t)n_total_expert > UINT64_MAX / gate_expert_bytes ||
        (uint64_t)n_total_expert > UINT64_MAX / up_expert_bytes ||
        (uint64_t)n_total_expert > UINT64_MAX / down_expert_bytes) {
        return 0;
    }

    const uint32_t pair_rows = n_tokens * n_expert;
    if ((uint64_t)pair_rows > UINT64_MAX / expert_mid_dim ||
        (uint64_t)pair_rows > UINT64_MAX / out_dim ||
        (uint64_t)n_tokens > UINT64_MAX / expert_in_dim ||
        (uint64_t)n_tokens > UINT64_MAX / out_dim) {
        return 0;
    }

    const bool mid_f16 = true;
    const NSUInteger mm_id_threadgroup_bytes = 8192u;
    const uint64_t compact_mid_values = (uint64_t)pair_rows * expert_mid_dim;
    const uint64_t down_values = (uint64_t)pair_rows * out_dim;
    const uint64_t x_values = (uint64_t)n_tokens * expert_in_dim;
    const uint64_t out_values = (uint64_t)n_tokens * out_dim;
    if (compact_mid_values > UINT64_MAX / sizeof(float) ||
        compact_mid_values > UINT64_MAX / (mid_f16 ? sizeof(uint16_t) : sizeof(float)) ||
        down_values > UINT64_MAX / sizeof(float) ||
        x_values > UINT64_MAX / sizeof(float) ||
        out_values > UINT64_MAX / sizeof(float)) {
        return 0;
    }

    const uint64_t gate_scratch_bytes = compact_mid_values * sizeof(float);
    const uint64_t mid_bytes = compact_mid_values * (mid_f16 ? sizeof(uint16_t) : sizeof(float));
    const uint64_t down_scratch_bytes = down_values * sizeof(float);
    const uint64_t x_bytes = x_values * sizeof(float);
    const uint64_t out_bytes = out_values * sizeof(float);
    const uint64_t selected_values = (uint64_t)n_tokens * n_expert;
    const uint64_t selected_bytes = selected_values * sizeof(int32_t);
    const uint64_t weights_bytes = selected_values * sizeof(float);
    if (gate_scratch_bytes > UINT64_MAX - gate_scratch_bytes ||
        gate_scratch_bytes > NSUIntegerMax ||
        gate_scratch_bytes * 2ull > NSUIntegerMax ||
        down_scratch_bytes > NSUIntegerMax) {
        return 0;
    }

    uint32_t first_expert = 0;
    uint32_t n_bind_expert = 0;
    ds4_gpu_tp_expert_range(n_total_expert, &first_expert, &n_bind_expert);
    gate_offset += (uint64_t)first_expert * gate_expert_bytes;
    up_offset += (uint64_t)first_expert * up_expert_bytes;
    down_offset += (uint64_t)first_expert * down_expert_bytes;
    const uint64_t gate_tensor_bytes = (uint64_t)n_bind_expert * gate_expert_bytes;
    const uint64_t up_tensor_bytes = (uint64_t)n_bind_expert * up_expert_bytes;
    const uint64_t down_tensor_bytes = (uint64_t)n_bind_expert * down_expert_bytes;

    @autoreleasepool {
        id<MTLBuffer> xbuf = ds4_gpu_tensor_buffer(x);
        id<MTLBuffer> midbuf = ds4_gpu_tensor_buffer(mid);
        id<MTLBuffer> outbuf = ds4_gpu_tensor_buffer(out);
        id<MTLBuffer> selectedbuf = ds4_gpu_tensor_buffer(selected);
        id<MTLBuffer> weightsbuf = ds4_gpu_tensor_buffer(weights);
        if (!xbuf || !midbuf || !outbuf || !selectedbuf || !weightsbuf ||
            ds4_gpu_tensor_bytes(x) < x_bytes ||
            ds4_gpu_tensor_bytes(mid) < mid_bytes ||
            ds4_gpu_tensor_bytes(out) < out_bytes ||
            ds4_gpu_tensor_bytes(selected) < selected_bytes ||
            ds4_gpu_tensor_bytes(weights) < weights_bytes) {
            fprintf(stderr, "ds4: Metal GLM grouped routed MoE received undersized activation buffers\n");
            return 0;
        }
        if (!ds4_gpu_ensure_scratch_buffer(&g_moe_gate_scratch_buffer,
                                           &g_moe_gate_scratch_bytes,
                                           (NSUInteger)(gate_scratch_bytes * 2ull),
                                           "ds4_glm_moe_gate_up_scratch")) {
            return 0;
        }
        if (n_expert > 1 &&
            !ds4_gpu_ensure_scratch_buffer(&g_moe_down_scratch_buffer,
                                           &g_moe_down_scratch_bytes,
                                           (NSUInteger)down_scratch_bytes,
                                           "ds4_glm_moe_down_scratch")) {
            return 0;
        }

        uint64_t gate_inner = 0;
        uint64_t up_inner = 0;
        uint64_t down_inner = 0;
        id<MTLBuffer> gatebuf = ds4_gpu_wrap_model_range(model_map, model_size,
                                                         gate_offset, gate_tensor_bytes,
                                                         &gate_inner);
        id<MTLBuffer> upbuf = ds4_gpu_wrap_model_range(model_map, model_size,
                                                       up_offset, up_tensor_bytes,
                                                       &up_inner);
        id<MTLBuffer> downbuf = ds4_gpu_wrap_model_range(model_map, model_size,
                                                         down_offset, down_tensor_bytes,
                                                         &down_inner);
        if (!gatebuf || !upbuf || !downbuf) return 0;

        id<MTLComputePipelineState> map_pipeline =
            ds4_gpu_get_pipeline(ds4_gpu_mul_mm_id_map0_name(n_expert));
        id<MTLComputePipelineState> gate_pipeline = ds4_gpu_routed_mm_pipeline(gate_type);
        id<MTLComputePipelineState> up_pipeline = ds4_gpu_routed_mm_pipeline(up_type);
        id<MTLComputePipelineState> down_pipeline =
            ds4_gpu_routed_mm_f16_rhs_pipeline(down_type);
        if (!map_pipeline || !gate_pipeline || !up_pipeline || !down_pipeline) {
            return 0;
        }

        ds4_gpu_mul_mm_id_map_args map_args =
            ds4_gpu_make_mul_mm_id_map_args(expert_in_dim, n_total_expert, 1, n_expert, n_tokens);
        ds4_gpu_mul_mm_id_args gate_args =
            ds4_gpu_make_mul_mm_id_args(expert_in_dim, expert_mid_dim, n_total_expert,
                                          gate_row_bytes, gate_expert_bytes,
                                          1, n_expert, n_tokens);
        ds4_gpu_mul_mm_id_args up_args =
            ds4_gpu_make_mul_mm_id_args(expert_in_dim, expert_mid_dim, n_total_expert,
                                          up_row_bytes, up_expert_bytes,
                                          1, n_expert, n_tokens);
        ds4_gpu_mul_mm_id_args down_args =
            ds4_gpu_make_mul_mm_id_args_src1_size(expert_mid_dim, out_dim, n_total_expert,
                                                    down_row_bytes, down_expert_bytes,
                                                    n_expert, n_expert, n_tokens,
                                                    mid_f16 ? sizeof(uint16_t) : sizeof(float));
        gate_args.tp_rank = g_tp_split_rank;
        gate_args.tp_world = g_tp_split_world;
        gate_args.tp_expert_base = (int32_t)first_expert;
        up_args.tp_rank = g_tp_split_rank;
        up_args.tp_world = g_tp_split_world;
        up_args.tp_expert_base = (int32_t)first_expert;
        down_args.tp_rank = g_tp_split_rank;
        down_args.tp_world = g_tp_split_world;
        down_args.tp_expert_base = (int32_t)first_expert;
        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        const bool glm_moe_stage_profile = false;
        const char *glm_moe_stage_filter = NULL;
        double glm_moe_stage_t0 = 0.0;
        if (glm_moe_stage_profile) {
            if (ds4_gpu_end_commands() == 0 || ds4_gpu_begin_commands() == 0) {
                return 0;
            }
            cb = ds4_gpu_command_buffer(&owned);
            if (!cb) return 0;
            glm_moe_stage_t0 = ds4_gpu_now_ms();
        }

        int ok = 1;
#define DS4_METAL_PROFILE_GLM_GROUPED_MOE_STAGE(name) do { \
            if (ok && glm_moe_stage_profile) { \
                if (ds4_gpu_end_commands() == 0) { \
                    ok = 0; \
                } else { \
                    const char *stage_name = (name); \
                    const double now_ms = ds4_gpu_now_ms(); \
                    const int print_stage = \
                        !glm_moe_stage_filter || !glm_moe_stage_filter[0] || \
                        strstr(stage_name, glm_moe_stage_filter) != NULL; \
                    if (print_stage) { \
                        fprintf(stderr, \
                                "ds4: Metal GLM grouped routed MoE stage layer=%u tokens=%u pairs=%u experts=%u " \
                                "gate=%s down=%s mid=%s %s=%.3f ms\n", \
                                layer_index, n_tokens, pair_rows, n_expert, \
                                ds4_gpu_metal_tensor_type_name(gate_type), \
                                ds4_gpu_metal_tensor_type_name(down_type), \
                                mid_f16 ? "f16" : "f32", \
                                stage_name, now_ms - glm_moe_stage_t0); \
                    } \
                    glm_moe_stage_t0 = now_ms; \
                    if (ds4_gpu_begin_commands() == 0) { \
                        ok = 0; \
                    } else { \
                        cb = ds4_gpu_command_buffer(&owned); \
                        if (!cb) ok = 0; \
                    } \
                } \
            } \
        } while (0)

        ok = ds4_gpu_encode_mul_mm_id_map(cb,
                                           map_pipeline,
                                           &map_args,
                                           &gate_args,
                                           selectedbuf,
                                           ds4_gpu_tensor_offset(selected));
        DS4_METAL_PROFILE_GLM_GROUPED_MOE_STAGE("map");
        if (ok) {
            ok = ds4_gpu_encode_mul_mm_id_mapped_tile(cb,
                                                       gate_pipeline,
                                                       &gate_args,
                                                       gatebuf,
                                                       (NSUInteger)gate_inner,
                                                       xbuf,
                                                       ds4_gpu_tensor_offset(x),
                                                       g_moe_gate_scratch_buffer,
                                                       0,
                                                       mm_id_threadgroup_bytes);
        }
        DS4_METAL_PROFILE_GLM_GROUPED_MOE_STAGE("gate");
        if (ok) {
            ok = ds4_gpu_encode_mul_mm_id_mapped_tile(cb,
                                                       up_pipeline,
                                                       &up_args,
                                                       upbuf,
                                                       (NSUInteger)up_inner,
                                                       xbuf,
                                                       ds4_gpu_tensor_offset(x),
                                                       g_moe_gate_scratch_buffer,
                                                       (NSUInteger)gate_scratch_bytes,
                                                       mm_id_threadgroup_bytes);
        }
        DS4_METAL_PROFILE_GLM_GROUPED_MOE_STAGE("up");
        if (ok) {
            ok = ds4_gpu_encode_moe_swiglu_weight(cb,
                                                   g_moe_gate_scratch_buffer,
                                                   0,
                                                   g_moe_gate_scratch_buffer,
                                                   (NSUInteger)gate_scratch_bytes,
                                                   midbuf,
                                                   ds4_gpu_tensor_offset(mid),
                                                   weightsbuf,
                                                   ds4_gpu_tensor_offset(weights),
                                                   expert_mid_dim,
                                                   pair_rows,
                                                   0.0f,
                                                   mid_f16);
        }
        DS4_METAL_PROFILE_GLM_GROUPED_MOE_STAGE("activation_weight");

        id<MTLBuffer> down_dst = n_expert == 1 ? outbuf : g_moe_down_scratch_buffer;
        NSUInteger down_dst_off = n_expert == 1 ? ds4_gpu_tensor_offset(out) : 0;
        if (ok) {
            ok = ds4_gpu_encode_mul_mm_id_mapped_tile(cb,
                                                       down_pipeline,
                                                       &down_args,
                                                       downbuf,
                                                       (NSUInteger)down_inner,
                                                       midbuf,
                                                       ds4_gpu_tensor_offset(mid),
                                                       down_dst,
                                                       down_dst_off,
                                                       mm_id_threadgroup_bytes);
        }
        DS4_METAL_PROFILE_GLM_GROUPED_MOE_STAGE("down");
        if (ok && n_expert > 1) {
            ok = ds4_gpu_encode_moe_sum_experts(cb,
                                                 down_dst,
                                                 down_dst_off,
                                                 outbuf,
                                                 ds4_gpu_tensor_offset(out),
                                                 out_dim,
                                                 n_expert,
                                                 n_tokens);
        }
        DS4_METAL_PROFILE_GLM_GROUPED_MOE_STAGE("sum");
        if (!ok) return 0;

        if (!ds4_gpu_finish_command_buffer(cb, owned, "GLM grouped routed batch MoE")) {
            return 0;
        }
#undef DS4_METAL_PROFILE_GLM_GROUPED_MOE_STAGE
    }

    return 1;
}

static int ds4_gpu_glm_routed_moe_batch_grouped_addr_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *mid,
        id<MTLBuffer>          gate_addrs,
        id<MTLBuffer>          up_addrs,
        id<MTLBuffer>          down_addrs,
        id<MTLBuffer>          overflow_gate,
        id<MTLBuffer>          overflow_up,
        id<MTLBuffer>          overflow_down,
        uint32_t               gate_type,
        uint32_t               up_type,
        uint32_t               down_type,
        uint64_t               gate_expert_bytes,
        uint64_t               gate_row_bytes,
        uint64_t               up_expert_bytes,
        uint64_t               up_row_bytes,
        uint64_t               down_expert_bytes,
        uint64_t               down_row_bytes,
        uint32_t               expert_in_dim,
        uint32_t               expert_mid_dim,
        uint32_t               out_dim,
        const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *weights,
        uint32_t               n_total_expert,
        uint32_t               n_expert,
        uint32_t               layer_index,
        const ds4_gpu_tensor *x,
        uint32_t               n_tokens,
        ds4_gpu_stream_expert_cache_entry * const *resources,
        uint32_t               resource_count) {
    if (n_expert > UINT32_MAX / n_tokens ||
        !gate_addrs || !up_addrs || !down_addrs ||
        !ds4_gpu_glm_routed_moe_batch_grouped_available(gate_type,
                                                        up_type,
                                                        down_type,
                                                        n_expert,
                                                        n_tokens)) {
        return 0;
    }

    const uint32_t pair_rows = n_tokens * n_expert;
    if ((uint64_t)pair_rows > UINT64_MAX / expert_mid_dim ||
        (uint64_t)pair_rows > UINT64_MAX / out_dim ||
        (uint64_t)n_tokens > UINT64_MAX / expert_in_dim ||
        (uint64_t)n_tokens > UINT64_MAX / out_dim) {
        return 0;
    }

    const bool mid_f16 = true;
    const NSUInteger mm_id_threadgroup_bytes = 8192u;
    const uint64_t compact_mid_values = (uint64_t)pair_rows * expert_mid_dim;
    const uint64_t down_values = (uint64_t)pair_rows * out_dim;
    const uint64_t x_values = (uint64_t)n_tokens * expert_in_dim;
    const uint64_t out_values = (uint64_t)n_tokens * out_dim;
    if (compact_mid_values > UINT64_MAX / sizeof(float) ||
        compact_mid_values > UINT64_MAX / sizeof(uint16_t) ||
        down_values > UINT64_MAX / sizeof(float) ||
        x_values > UINT64_MAX / sizeof(float) ||
        out_values > UINT64_MAX / sizeof(float)) {
        return 0;
    }

    const uint64_t gate_scratch_bytes = compact_mid_values * sizeof(float);
    const uint64_t mid_bytes = compact_mid_values * sizeof(uint16_t);
    const uint64_t down_scratch_bytes = down_values * sizeof(float);
    const uint64_t x_bytes = x_values * sizeof(float);
    const uint64_t out_bytes = out_values * sizeof(float);
    const uint64_t selected_values = (uint64_t)n_tokens * n_expert;
    const uint64_t selected_bytes = selected_values * sizeof(int32_t);
    const uint64_t weights_bytes = selected_values * sizeof(float);
    if (gate_scratch_bytes > UINT64_MAX - gate_scratch_bytes ||
        gate_scratch_bytes > NSUIntegerMax ||
        gate_scratch_bytes * 2ull > NSUIntegerMax ||
        down_scratch_bytes > NSUIntegerMax) {
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> xbuf = ds4_gpu_tensor_buffer(x);
        id<MTLBuffer> midbuf = ds4_gpu_tensor_buffer(mid);
        id<MTLBuffer> outbuf = ds4_gpu_tensor_buffer(out);
        id<MTLBuffer> selectedbuf = ds4_gpu_tensor_buffer(selected);
        id<MTLBuffer> weightsbuf = ds4_gpu_tensor_buffer(weights);
        if (!xbuf || !midbuf || !outbuf || !selectedbuf || !weightsbuf ||
            ds4_gpu_tensor_bytes(x) < x_bytes ||
            ds4_gpu_tensor_bytes(mid) < mid_bytes ||
            ds4_gpu_tensor_bytes(out) < out_bytes ||
            ds4_gpu_tensor_bytes(selected) < selected_bytes ||
            ds4_gpu_tensor_bytes(weights) < weights_bytes) {
            fprintf(stderr, "ds4: Metal GLM grouped-address routed MoE received undersized activation buffers\n");
            return 0;
        }
        if (!ds4_gpu_ensure_scratch_buffer(&g_moe_gate_scratch_buffer,
                                           &g_moe_gate_scratch_bytes,
                                           (NSUInteger)(gate_scratch_bytes * 2ull),
                                           "ds4_glm_moe_gate_up_scratch")) {
            return 0;
        }
        if (n_expert > 1 &&
            !ds4_gpu_ensure_scratch_buffer(&g_moe_down_scratch_buffer,
                                           &g_moe_down_scratch_bytes,
                                           (NSUInteger)down_scratch_bytes,
                                           "ds4_glm_moe_down_scratch")) {
            return 0;
        }

        id<MTLComputePipelineState> map_pipeline =
            ds4_gpu_get_pipeline(ds4_gpu_mul_mm_id_map0_name(n_expert));
        id<MTLComputePipelineState> gate_pipeline =
            ds4_gpu_routed_mm_addr_pipeline(gate_type);
        id<MTLComputePipelineState> up_pipeline =
            ds4_gpu_routed_mm_addr_pipeline(up_type);
        id<MTLComputePipelineState> down_pipeline =
            ds4_gpu_routed_mm_addr_f16_rhs_pipeline(down_type);
        if (!map_pipeline || !gate_pipeline || !up_pipeline || !down_pipeline) {
            return 0;
        }

        ds4_gpu_mul_mm_id_map_args map_args =
            ds4_gpu_make_mul_mm_id_map_args(expert_in_dim, n_total_expert, 1, n_expert, n_tokens);
        ds4_gpu_mul_mm_id_args gate_args =
            ds4_gpu_make_mul_mm_id_args(expert_in_dim, expert_mid_dim, n_total_expert,
                                          gate_row_bytes, gate_expert_bytes,
                                          1, n_expert, n_tokens);
        ds4_gpu_mul_mm_id_args up_args =
            ds4_gpu_make_mul_mm_id_args(expert_in_dim, expert_mid_dim, n_total_expert,
                                          up_row_bytes, up_expert_bytes,
                                          1, n_expert, n_tokens);
        ds4_gpu_mul_mm_id_args down_args =
            ds4_gpu_make_mul_mm_id_args_src1_size(expert_mid_dim, out_dim, n_total_expert,
                                                    down_row_bytes, down_expert_bytes,
                                                    n_expert, n_expert, n_tokens,
                                                    sizeof(uint16_t));
        if (!ds4_gpu_stream_expert_cache_mark_entries_inflight(resources,
                                                               resource_count,
                                                               0)) {
            return 0;
        }

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        const bool glm_moe_stage_profile = false;
        const char *glm_moe_stage_filter = NULL;
        double glm_moe_stage_t0 = 0.0;
        if (glm_moe_stage_profile) {
            if (ds4_gpu_end_commands() == 0 || ds4_gpu_begin_commands() == 0) {
                return 0;
            }
            cb = ds4_gpu_command_buffer(&owned);
            if (!cb) return 0;
            glm_moe_stage_t0 = ds4_gpu_now_ms();
        }

        int ok = 1;
#define DS4_METAL_PROFILE_GLM_GROUPED_ADDR_MOE_STAGE(name) do { \
            if (ok && glm_moe_stage_profile) { \
                if (ds4_gpu_end_commands() == 0) { \
                    ok = 0; \
                } else { \
                    const char *stage_name = (name); \
                    const double now_ms = ds4_gpu_now_ms(); \
                    const int print_stage = \
                        !glm_moe_stage_filter || !glm_moe_stage_filter[0] || \
                        strstr(stage_name, glm_moe_stage_filter) != NULL; \
                    if (print_stage) { \
                        fprintf(stderr, \
                                "ds4: Metal GLM grouped-address routed MoE stage layer=%u tokens=%u pairs=%u experts=%u " \
                                "gate=%s down=%s mid=f16 %s=%.3f ms\n", \
                                layer_index, n_tokens, pair_rows, n_expert, \
                                ds4_gpu_metal_tensor_type_name(gate_type), \
                                ds4_gpu_metal_tensor_type_name(down_type), \
                                stage_name, now_ms - glm_moe_stage_t0); \
                    } \
                    glm_moe_stage_t0 = now_ms; \
                    if (ds4_gpu_begin_commands() == 0) { \
                        ok = 0; \
                    } else { \
                        cb = ds4_gpu_command_buffer(&owned); \
                        if (!cb) ok = 0; \
                    } \
                } \
            } \
        } while (0)

        ok = ds4_gpu_encode_mul_mm_id_map(cb,
                                           map_pipeline,
                                           &map_args,
                                           &gate_args,
                                           selectedbuf,
                                           ds4_gpu_tensor_offset(selected));
        DS4_METAL_PROFILE_GLM_GROUPED_ADDR_MOE_STAGE("map");
        if (ok) {
            ok = ds4_gpu_encode_mul_mm_id_addr_mapped_tile(cb,
                                                           gate_pipeline,
                                                           &gate_args,
                                                           gate_addrs,
                                                           xbuf,
                                                           ds4_gpu_tensor_offset(x),
                                                           g_moe_gate_scratch_buffer,
                                                           0,
                                                           mm_id_threadgroup_bytes,
                                                           resources,
                                                           resource_count,
                                                           0,
                                                           overflow_gate);
        }
        DS4_METAL_PROFILE_GLM_GROUPED_ADDR_MOE_STAGE("gate");
        if (ok) {
            ok = ds4_gpu_encode_mul_mm_id_addr_mapped_tile(cb,
                                                           up_pipeline,
                                                           &up_args,
                                                           up_addrs,
                                                           xbuf,
                                                           ds4_gpu_tensor_offset(x),
                                                           g_moe_gate_scratch_buffer,
                                                           (NSUInteger)gate_scratch_bytes,
                                                           mm_id_threadgroup_bytes,
                                                           resources,
                                                           resource_count,
                                                           1,
                                                           overflow_up);
        }
        DS4_METAL_PROFILE_GLM_GROUPED_ADDR_MOE_STAGE("up");
        if (ok) {
            ok = ds4_gpu_encode_moe_swiglu_weight(cb,
                                                   g_moe_gate_scratch_buffer,
                                                   0,
                                                   g_moe_gate_scratch_buffer,
                                                   (NSUInteger)gate_scratch_bytes,
                                                   midbuf,
                                                   ds4_gpu_tensor_offset(mid),
                                                   weightsbuf,
                                                   ds4_gpu_tensor_offset(weights),
                                                   expert_mid_dim,
                                                   pair_rows,
                                                   0.0f,
                                                   mid_f16);
        }
        DS4_METAL_PROFILE_GLM_GROUPED_ADDR_MOE_STAGE("activation_weight");

        id<MTLBuffer> down_dst = n_expert == 1 ? outbuf : g_moe_down_scratch_buffer;
        NSUInteger down_dst_off = n_expert == 1 ? ds4_gpu_tensor_offset(out) : 0;
        if (ok) {
            ok = ds4_gpu_encode_mul_mm_id_addr_mapped_tile(cb,
                                                           down_pipeline,
                                                           &down_args,
                                                           down_addrs,
                                                           midbuf,
                                                           ds4_gpu_tensor_offset(mid),
                                                           down_dst,
                                                           down_dst_off,
                                                           mm_id_threadgroup_bytes,
                                                           resources,
                                                           resource_count,
                                                           2,
                                                           overflow_down);
        }
        DS4_METAL_PROFILE_GLM_GROUPED_ADDR_MOE_STAGE("down");
        if (ok && n_expert > 1) {
            ok = ds4_gpu_encode_moe_sum_experts(cb,
                                                 down_dst,
                                                 down_dst_off,
                                                 outbuf,
                                                 ds4_gpu_tensor_offset(out),
                                                 out_dim,
                                                 n_expert,
                                                 n_tokens);
        }
        DS4_METAL_PROFILE_GLM_GROUPED_ADDR_MOE_STAGE("sum");
        if (!ok) return 0;

        if (!ds4_gpu_finish_command_buffer(cb, owned,
                                           "GLM grouped-address routed batch MoE")) {
            return 0;
        }
#undef DS4_METAL_PROFILE_GLM_GROUPED_ADDR_MOE_STAGE
    }

    return 1;
}

static int ds4_gpu_glm_routed_moe_batch_tensor_impl(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                down_offset,
        uint32_t                gate_type,
        uint32_t                up_type,
        uint32_t                down_type,
        uint64_t                gate_expert_bytes,
        uint64_t                gate_row_bytes,
        uint64_t                up_expert_bytes,
        uint64_t                up_row_bytes,
        uint64_t                down_expert_bytes,
        uint64_t                down_row_bytes,
        uint32_t                expert_in_dim,
        uint32_t                expert_mid_dim,
        uint32_t                out_dim,
        const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *weights,
        uint32_t                n_total_expert,
        uint32_t                n_expert,
        uint32_t                layer_index,
        const ds4_gpu_tensor *x,
        uint32_t                n_tokens,
        uint32_t                mid_token_stride,
        bool                    allow_grouped,
        bool                    force_scalar_q4_pair) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!out || !mid || !model_map || !selected || !weights || !x ||
        n_tokens == 0 ||
        n_total_expert == 0 || n_expert == 0 || n_expert > 256u ||
        n_expert > n_total_expert ||
        expert_in_dim == 0 || expert_mid_dim == 0 || out_dim == 0 ||
        gate_expert_bytes == 0 || gate_row_bytes == 0 ||
        up_expert_bytes == 0 || up_row_bytes == 0 ||
        down_expert_bytes == 0 || down_row_bytes == 0 ||
        (expert_in_dim % 256u) != 0 ||
        (expert_mid_dim % 256u) != 0 ||
        !ds4_gpu_glm_gate_pair_type_supported(gate_type, up_type) ||
        !ds4_gpu_glm_down_type_supported(down_type)) {
        return 0;
    }

    const uint64_t per_token_mid = (uint64_t)n_expert * expert_mid_dim;
    if ((uint64_t)mid_token_stride < per_token_mid ||
        (uint64_t)n_total_expert > UINT64_MAX / gate_expert_bytes ||
        (uint64_t)n_total_expert > UINT64_MAX / up_expert_bytes ||
        (uint64_t)n_total_expert > UINT64_MAX / down_expert_bytes ||
        (uint64_t)expert_mid_dim > UINT64_MAX / gate_row_bytes ||
        (uint64_t)expert_mid_dim > UINT64_MAX / up_row_bytes ||
        (uint64_t)out_dim > UINT64_MAX / down_row_bytes ||
        (uint64_t)n_tokens > UINT64_MAX / expert_in_dim ||
        (uint64_t)n_tokens > UINT64_MAX / out_dim) {
        fprintf(stderr, "ds4: Metal GLM routed batch MoE tensor byte size overflow\n");
        return 0;
    }

    const uint64_t full_gate_tensor_bytes =
        (uint64_t)n_total_expert * gate_expert_bytes;
    const uint64_t full_up_tensor_bytes =
        (uint64_t)n_total_expert * up_expert_bytes;
    const uint64_t full_down_tensor_bytes =
        (uint64_t)n_total_expert * down_expert_bytes;
    if (gate_expert_bytes != (uint64_t)expert_mid_dim * gate_row_bytes ||
        up_expert_bytes != (uint64_t)expert_mid_dim * up_row_bytes ||
        down_expert_bytes != (uint64_t)out_dim * down_row_bytes) {
        fprintf(stderr, "ds4: Metal GLM routed batch MoE received inconsistent expert strides\n");
        return 0;
    }
    if (gate_offset > model_size || full_gate_tensor_bytes > model_size - gate_offset ||
        up_offset > model_size || full_up_tensor_bytes > model_size - up_offset ||
        down_offset > model_size || full_down_tensor_bytes > model_size - down_offset) {
        fprintf(stderr, "ds4: Metal GLM routed batch MoE tensor range is outside the mapped model\n");
        return 0;
    }

    if (allow_grouped &&
        (!g_ssd_streaming_mode ||
         ds4_gpu_glm_streaming_prefill_full_layer_active()) &&
        ds4_gpu_glm_grouped_moe_layer_enabled(layer_index) &&
        ds4_gpu_glm_routed_moe_batch_grouped_available(gate_type,
                                                       up_type,
                                                       down_type,
                                                       n_expert,
                                                       n_tokens)) {
        return ds4_gpu_glm_routed_moe_batch_grouped_tensor(out,
                                                           mid,
                                                           model_map,
                                                           model_size,
                                                           gate_offset,
                                                           up_offset,
                                                           down_offset,
                                                           gate_type,
                                                           up_type,
                                                           down_type,
                                                           gate_expert_bytes,
                                                           gate_row_bytes,
                                                           up_expert_bytes,
                                                           up_row_bytes,
                                                           down_expert_bytes,
                                                           down_row_bytes,
                                                           expert_in_dim,
                                                           expert_mid_dim,
                                                           out_dim,
                                                           selected,
                                                           weights,
                                                           n_total_expert,
                                                           n_expert,
                                                           layer_index,
                                                           x,
                                                           n_tokens);
    }

    uint32_t first_expert = 0;
    uint32_t n_bind_expert = 0;
    ds4_gpu_tp_expert_range(n_total_expert, &first_expert, &n_bind_expert);
    gate_offset += (uint64_t)first_expert * gate_expert_bytes;
    up_offset += (uint64_t)first_expert * up_expert_bytes;
    down_offset += (uint64_t)first_expert * down_expert_bytes;
    const uint64_t gate_tensor_bytes = (uint64_t)n_bind_expert * gate_expert_bytes;
    const uint64_t up_tensor_bytes = (uint64_t)n_bind_expert * up_expert_bytes;
    const uint64_t down_tensor_bytes = (uint64_t)n_bind_expert * down_expert_bytes;

    @autoreleasepool {
        id<MTLBuffer> xbuf = ds4_gpu_tensor_buffer(x);
        id<MTLBuffer> midbuf = ds4_gpu_tensor_buffer(mid);
        id<MTLBuffer> outbuf = ds4_gpu_tensor_buffer(out);
        id<MTLBuffer> selectedbuf = ds4_gpu_tensor_buffer(selected);
        id<MTLBuffer> weightsbuf = ds4_gpu_tensor_buffer(weights);
        const uint64_t x_values = (uint64_t)n_tokens * expert_in_dim;
        const uint64_t out_values = (uint64_t)n_tokens * out_dim;
        const uint64_t mid_values =
            (uint64_t)(n_tokens - 1u) * mid_token_stride + per_token_mid;
        const uint64_t selected_values = (uint64_t)n_tokens * n_expert;
        if (x_values > UINT64_MAX / sizeof(float) ||
            out_values > UINT64_MAX / sizeof(float) ||
            mid_values > UINT64_MAX / sizeof(float) ||
            selected_values > UINT64_MAX / sizeof(int32_t)) {
            fprintf(stderr, "ds4: Metal GLM routed batch MoE activation byte size overflow\n");
            return 0;
        }
        const uint64_t x_bytes = x_values * sizeof(float);
        const uint64_t mid_bytes = mid_values * sizeof(float);
        const uint64_t out_bytes = out_values * sizeof(float);
        const uint64_t selected_bytes = selected_values * sizeof(int32_t);
        const uint64_t weights_bytes = selected_values * sizeof(float);
        if (!xbuf || !midbuf || !outbuf || !selectedbuf || !weightsbuf ||
            ds4_gpu_tensor_bytes(x) < x_bytes ||
            ds4_gpu_tensor_bytes(mid) < mid_bytes ||
            ds4_gpu_tensor_bytes(out) < out_bytes ||
            ds4_gpu_tensor_bytes(selected) < selected_bytes ||
            ds4_gpu_tensor_bytes(weights) < weights_bytes) {
            fprintf(stderr, "ds4: Metal GLM routed batch MoE received undersized activation buffers\n");
            return 0;
        }

        uint64_t gate_inner = 0;
        uint64_t up_inner = 0;
        uint64_t down_inner = 0;
        id<MTLBuffer> gatebuf = nil;
        id<MTLBuffer> upbuf = nil;
        id<MTLBuffer> downbuf = nil;
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

        const BOOL gate_pair_q2 = gate_type == DS4_METAL_TENSOR_Q2_K;
        const BOOL gate_pair_q5 = gate_type == DS4_METAL_TENSOR_Q5_K;
        const BOOL down_scalar_q2 = down_type == DS4_METAL_TENSOR_Q2_K;
        const BOOL down_scalar_q4 = down_type == DS4_METAL_TENSOR_Q4_K;
        const BOOL down_simd_q4 = down_scalar_q4;
        const BOOL down_simd_q5 = down_type == DS4_METAL_TENSOR_Q5_K;
        const BOOL down_simd_q6 = down_type == DS4_METAL_TENSOR_Q6_K;
        const BOOL down_simd = down_simd_q4 || down_simd_q5 || down_simd_q6;
        const BOOL stream_addr_q2 =
            gate_pair_q2 && down_scalar_q2 &&
            g_glm_q2_k_addr_pair_swiglu2_f32_pipeline != nil &&
            g_glm_q2_k_addr_down_f32_pipeline != nil;
        const BOOL stream_addr_q4 =
            !gate_pair_q2 && !gate_pair_q5 && down_scalar_q4 &&
            g_glm_q4_k_addr_pair_swiglu_f32_pipeline != nil &&
            g_glm_q4_k_addr_down_f32_pipeline != nil;
        BOOL use_stream_expert_addr_table =
            g_ssd_streaming_mode &&
            !ds4_gpu_glm_streaming_prefill_full_layer_active() &&
            n_tokens > 1 &&
            (stream_addr_q2 || stream_addr_q4) &&
            layer_index < DS4_METAL_STREAM_EXPERT_CACHE_MAX_LAYER &&
            n_total_expert <= DS4_METAL_STREAM_EXPERT_CACHE_MAX_EXPERT &&
            n_expert <= DS4_METAL_STREAM_EXPERT_CACHE_MAX_SELECTED &&
            ds4_gpu_stream_expert_cache_note_expert_size(gate_expert_bytes,
                                                         down_expert_bytes) &&
            ds4_gpu_stream_expert_cache_effective_cap(layer_index,
                                                      n_total_expert,
                                                      n_expert) != 0;
        const BOOL use_stream_grouped_addr_table =
            use_stream_expert_addr_table &&
            allow_grouped &&
            getenv("DS4_METAL_GLM_DISABLE_STREAMING_GROUPED_ADDR_PREFILL") == NULL &&
            ds4_gpu_glm_routed_moe_batch_grouped_available(gate_type,
                                                           up_type,
                                                           down_type,
                                                           n_expert,
                                                           n_tokens);
        const BOOL enable_q4_pair4 = true;
        const BOOL q4_scalar_pair = false;
        const BOOL q4_pair2 =
            !use_stream_expert_addr_table &&
            !gate_pair_q5 && !q4_scalar_pair &&
            (force_scalar_q4_pair || !enable_q4_pair4);
        id<MTLComputePipelineState> pair_pipeline =
            use_stream_expert_addr_table ?
             (gate_pair_q2 ?
              ds4_gpu_hot_pipeline(g_glm_q2_k_addr_pair_swiglu2_f32_pipeline,
                                   "kernel_glm_q2_K_addr_pair_swiglu2_f32") :
              ds4_gpu_hot_pipeline(g_glm_q4_k_addr_pair_swiglu_f32_pipeline,
                                   "kernel_glm_q4_K_addr_pair_swiglu_f32")) :
            gate_pair_q2 ?
             ds4_gpu_hot_pipeline(g_glm_q2_k_pair_swiglu_f32_pipeline,
                                  "kernel_glm_q2_K_pair_swiglu_f32") :
            gate_pair_q5 ?
             ds4_gpu_hot_pipeline(g_glm_q5_k_pair_swiglu_f32_pipeline,
                                  "kernel_glm_q5_K_pair_swiglu_f32") :
            (q4_scalar_pair ?
             ds4_gpu_hot_pipeline(g_glm_q4_k_pair_swiglu_f32_pipeline,
                                  "kernel_glm_q4_K_pair_swiglu_f32") :
             q4_pair2 ?
              ds4_gpu_hot_pipeline(g_glm_q4_k_pair_swiglu2_f32_pipeline,
                                   "kernel_glm_q4_K_pair_swiglu2_f32") :
              ds4_gpu_hot_pipeline(g_glm_q4_k_pair_swiglu4_f32_pipeline,
                                   "kernel_glm_q4_K_pair_swiglu4_f32"));
        id<MTLComputePipelineState> down_pipeline =
            use_stream_expert_addr_table ?
             (down_scalar_q2 ?
              ds4_gpu_hot_pipeline(g_glm_q2_k_addr_down_f32_pipeline,
                                   "kernel_glm_q2_K_addr_down_f32") :
              ds4_gpu_hot_pipeline(g_glm_q4_k_addr_down_f32_pipeline,
                                   "kernel_glm_q4_K_addr_down_f32")) :
            down_scalar_q2 ?
            ds4_gpu_hot_pipeline(g_glm_q2_k_down_f32_pipeline,
                                 "kernel_glm_q2_K_down_f32") :
            down_scalar_q4 ?
            ds4_gpu_hot_pipeline(g_glm_q4_k_down_f32_pipeline,
                                 "kernel_glm_q4_K_down_f32") :
            down_simd_q5 ?
            ds4_gpu_hot_pipeline(g_glm_q5_k_down_f32_pipeline,
                                 "kernel_glm_q5_K_down_f32") :
            ds4_gpu_hot_pipeline(g_glm_q6_k_down_f32_pipeline,
                                 "kernel_glm_q6_K_down_f32");
        if (!pair_pipeline || !down_pipeline) return 0;

        if (use_stream_expert_addr_table) {
            const int had_batch = g_batch_cb != nil;
            if (had_batch && ds4_gpu_end_commands() == 0) {
                return 0;
            }
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
                return 0;
            }
            if (stream_unique == 0) {
                ds4_gpu_stream_expert_cache_clear_layer(layer_index);
                return 0;
            }
            if (had_batch && ds4_gpu_begin_commands() == 0) {
                ds4_gpu_stream_expert_cache_clear_layer(layer_index);
                return 0;
            }
            if (use_stream_grouped_addr_table) {
                return ds4_gpu_glm_routed_moe_batch_grouped_addr_tensor(
                        out,
                        mid,
                        stream_gate_addr_buf,
                        stream_up_addr_buf,
                        stream_down_addr_buf,
                        stream_overflow_gate,
                        stream_overflow_up,
                        stream_overflow_down,
                        gate_type,
                        up_type,
                        down_type,
                        gate_expert_bytes,
                        gate_row_bytes,
                        up_expert_bytes,
                        up_row_bytes,
                        down_expert_bytes,
                        down_row_bytes,
                        expert_in_dim,
                        expert_mid_dim,
                        out_dim,
                        selected,
                        weights,
                        n_total_expert,
                        n_expert,
                        layer_index,
                        x,
                        n_tokens,
                        stream_resources,
                        stream_resource_count);
            }
        } else {
            gatebuf = ds4_gpu_wrap_model_range(model_map, model_size,
                                               gate_offset, gate_tensor_bytes,
                                               &gate_inner);
            upbuf = ds4_gpu_wrap_model_range(model_map, model_size,
                                             up_offset, up_tensor_bytes,
                                             &up_inner);
            downbuf = ds4_gpu_wrap_model_range(model_map, model_size,
                                               down_offset, down_tensor_bytes,
                                               &down_inner);
            if (!gatebuf || !upbuf || !downbuf) return 0;
        }

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        const bool glm_moe_stage_profile = false;
        const char *glm_moe_stage_filter = NULL;
        const char *glm_pair_path = use_stream_expert_addr_table ?
                                    (gate_pair_q2 ? "q2_stream_addr_swiglu" :
                                                    "q4_stream_addr_swiglu") :
                                    gate_pair_q2 ? "q2_scalar_swiglu" :
                                    gate_pair_q5 ? "q5_pair_simd_swiglu" :
                                    (q4_scalar_pair ? "q4_scalar_swiglu" :
                                    (q4_pair2 ? "q4_pair2_simd_swiglu" :
                                                "q4_pair4_simd_swiglu"));
        const char *glm_down_path =
            use_stream_expert_addr_table ?
            (down_scalar_q2 ? "q2_stream_addr_down" : "q4_stream_addr_down_simd") :
            down_scalar_q2 ? "q2_down_scalar" :
            down_scalar_q4 ? "q4_down_simd" :
            down_simd_q5 ? "q5_down_simd" : "q6_down_simd";
        double glm_moe_stage_t0 = 0.0;
        if (glm_moe_stage_profile) {
            if (ds4_gpu_end_commands() == 0 || ds4_gpu_begin_commands() == 0) {
                return 0;
            }
            cb = ds4_gpu_command_buffer(&owned);
            if (!cb) return 0;
            glm_moe_stage_t0 = ds4_gpu_now_ms();
        }
        int ok = 1;
#define DS4_METAL_PROFILE_GLM_MOE_BATCH_STAGE(name) do { \
            if (ok && glm_moe_stage_profile) { \
                if (ds4_gpu_end_commands() == 0) { \
                    ok = 0; \
                } else { \
                    const char *stage_name = (name); \
                    const double now_ms = ds4_gpu_now_ms(); \
                    const int print_stage = \
                        !glm_moe_stage_filter || !glm_moe_stage_filter[0] || \
                        strstr(stage_name, glm_moe_stage_filter) != NULL; \
                    if (print_stage) { \
                        fprintf(stderr, \
                                "ds4: Metal GLM routed MoE batch stage layer=%u tokens=%u experts=%u " \
                                "gate=%s down=%s pair=%s down_path=%s %s=%.3f ms\n", \
                                layer_index, n_tokens, n_expert, \
                                ds4_gpu_metal_tensor_type_name(gate_type), \
                                ds4_gpu_metal_tensor_type_name(down_type), \
                                glm_pair_path, glm_down_path, \
                                stage_name, now_ms - glm_moe_stage_t0); \
                    } \
                    glm_moe_stage_t0 = now_ms; \
                    if (ds4_gpu_begin_commands() == 0) { \
                        ok = 0; \
                    } else { \
                        cb = ds4_gpu_command_buffer(&owned); \
                        if (!cb) ok = 0; \
                    } \
                } \
            } \
        } while (0)

        ds4_gpu_glm_routed_moe_args args = {
            .tp_rank = g_tp_split_rank,
            .tp_world = g_tp_split_world,
            .tp_expert_base = (int32_t)first_expert,
            .in_dim = expert_in_dim,
            .mid_dim = expert_mid_dim,
            .out_dim = out_dim,
            .n_total_expert = n_total_expert,
            .n_expert_used = n_expert,
            .n_tokens = n_tokens,
            .mid_token_stride = mid_token_stride,
            .down_type = down_type,
            .gate_expert_bytes = gate_expert_bytes,
            .gate_row_bytes = gate_row_bytes,
            .up_expert_bytes = up_expert_bytes,
            .up_row_bytes = up_row_bytes,
            .down_expert_bytes = down_expert_bytes,
            .down_row_bytes = down_row_bytes,
        };
        const NSUInteger pair_x_groups =
            gate_pair_q2 ? (use_stream_expert_addr_table ?
                            (NSUInteger)((expert_mid_dim + 1u) / 2u) :
                            (NSUInteger)((expert_mid_dim + 7u) / 8u)) :
            gate_pair_q5 ? (NSUInteger)((expert_mid_dim + 7u) / 8u) :
            use_stream_expert_addr_table ? (NSUInteger)((expert_mid_dim + 3u) / 4u) :
            q4_scalar_pair ? (NSUInteger)expert_mid_dim :
            q4_pair2 ? (NSUInteger)((expert_mid_dim + 1u) / 2u) :
            (NSUInteger)((expert_mid_dim + 7u) / 8u);
        const NSUInteger pair_threadgroup_bytes =
            q4_scalar_pair ? 512u * sizeof(float) : 0u;
        const NSUInteger pair_threads =
            q4_scalar_pair ? 256u : 64u;
        const NSUInteger down_x_groups =
            down_scalar_q2 ? (NSUInteger)((out_dim + 7u) / 8u) :
            down_simd_q4 ? (NSUInteger)((out_dim + 3u) / 4u) :
            down_simd_q5 ? (NSUInteger)((out_dim + 3u) / 4u) :
            down_simd_q6 ? (NSUInteger)((out_dim + 3u) / 4u) :
            (NSUInteger)out_dim;
        const NSUInteger down_threadgroup_bytes =
            down_simd ? 0u : 256u * sizeof(float);
        const NSUInteger down_threads =
            down_simd ? 64u : 256u;
        if (use_stream_expert_addr_table &&
            !ds4_gpu_stream_expert_cache_mark_entries_inflight(stream_resources,
                                                               stream_resource_count,
                                                               0)) {
            return 0;
        }

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pair_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:use_stream_expert_addr_table ? stream_gate_addr_buf : gatebuf
                offset:use_stream_expert_addr_table ? 0u : (NSUInteger)gate_inner
               atIndex:1];
        [enc setBuffer:use_stream_expert_addr_table ? stream_up_addr_buf : upbuf
                offset:use_stream_expert_addr_table ? 0u : (NSUInteger)up_inner
               atIndex:2];
        [enc setBuffer:xbuf offset:ds4_gpu_tensor_offset(x) atIndex:3];
        [enc setBuffer:selectedbuf offset:ds4_gpu_tensor_offset(selected) atIndex:4];
        [enc setBuffer:weightsbuf offset:ds4_gpu_tensor_offset(weights) atIndex:5];
        [enc setBuffer:midbuf offset:ds4_gpu_tensor_offset(mid) atIndex:6];
        if (use_stream_expert_addr_table) {
            for (uint32_t i = 0; i < stream_resource_count; i++) {
                [enc useResource:stream_resources[i]->gate_buffer usage:MTLResourceUsageRead];
                [enc useResource:stream_resources[i]->up_buffer usage:MTLResourceUsageRead];
            }
            if (stream_overflow_gate) [enc useResource:stream_overflow_gate usage:MTLResourceUsageRead];
            if (stream_overflow_up) [enc useResource:stream_overflow_up usage:MTLResourceUsageRead];
        }
        if (pair_threadgroup_bytes != 0u) {
            [enc setThreadgroupMemoryLength:pair_threadgroup_bytes atIndex:0];
        }
        [enc dispatchThreadgroups:MTLSizeMake(pair_x_groups,
                                              (NSUInteger)n_expert,
                                              (NSUInteger)n_tokens)
             threadsPerThreadgroup:MTLSizeMake(pair_threads, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);
        DS4_METAL_PROFILE_GLM_MOE_BATCH_STAGE("pair");

        enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:down_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:use_stream_expert_addr_table ? stream_down_addr_buf : downbuf
                offset:use_stream_expert_addr_table ? 0u : (NSUInteger)down_inner
               atIndex:1];
        [enc setBuffer:selectedbuf offset:ds4_gpu_tensor_offset(selected) atIndex:2];
        [enc setBuffer:midbuf offset:ds4_gpu_tensor_offset(mid) atIndex:3];
        [enc setBuffer:outbuf offset:ds4_gpu_tensor_offset(out) atIndex:4];
        if (use_stream_expert_addr_table) {
            for (uint32_t i = 0; i < stream_resource_count; i++) {
                [enc useResource:stream_resources[i]->down_buffer usage:MTLResourceUsageRead];
            }
            if (stream_overflow_down) [enc useResource:stream_overflow_down usage:MTLResourceUsageRead];
        }
        if (down_threadgroup_bytes != 0u) {
            [enc setThreadgroupMemoryLength:down_threadgroup_bytes atIndex:0];
        }
        [enc dispatchThreadgroups:MTLSizeMake(down_x_groups,
                                              (NSUInteger)n_tokens,
                                              1)
             threadsPerThreadgroup:MTLSizeMake(down_threads, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);
        DS4_METAL_PROFILE_GLM_MOE_BATCH_STAGE("down");

        if (!ok) return 0;
        if (!ds4_gpu_finish_command_buffer(cb, owned, "GLM routed batch MoE")) return 0;
#undef DS4_METAL_PROFILE_GLM_MOE_BATCH_STAGE
    }

    return 1;
}

int ds4_gpu_glm_routed_moe_batch_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                down_offset,
        uint32_t                gate_type,
        uint32_t                up_type,
        uint32_t                down_type,
        uint64_t                gate_expert_bytes,
        uint64_t                gate_row_bytes,
        uint64_t                up_expert_bytes,
        uint64_t                up_row_bytes,
        uint64_t                down_expert_bytes,
        uint64_t                down_row_bytes,
        uint32_t                expert_in_dim,
        uint32_t                expert_mid_dim,
        uint32_t                out_dim,
        const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *weights,
        uint32_t                n_total_expert,
        uint32_t                n_expert,
        uint32_t                layer_index,
        const ds4_gpu_tensor *x,
        uint32_t                n_tokens,
        uint32_t                mid_token_stride,
        bool                    force_resident) {
    (void)force_resident;
    return ds4_gpu_glm_routed_moe_batch_tensor_impl(out,
                                                    mid,
                                                    model_map,
                                                    model_size,
                                                    gate_offset,
                                                    up_offset,
                                                    down_offset,
                                                    gate_type,
                                                    up_type,
                                                    down_type,
                                                    gate_expert_bytes,
                                                    gate_row_bytes,
                                                    up_expert_bytes,
                                                    up_row_bytes,
                                                    down_expert_bytes,
                                                    down_row_bytes,
                                                    expert_in_dim,
                                                    expert_mid_dim,
                                                    out_dim,
                                                    selected,
                                                    weights,
                                                    n_total_expert,
                                                    n_expert,
                                                    layer_index,
                                                    x,
                                                    n_tokens,
                                                    mid_token_stride,
                                                    true,
                                                    false);
}

int ds4_gpu_glm_routed_moe_batch_direct_scalar_q4_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                down_offset,
        uint32_t                gate_type,
        uint32_t                up_type,
        uint32_t                down_type,
        uint64_t                gate_expert_bytes,
        uint64_t                gate_row_bytes,
        uint64_t                up_expert_bytes,
        uint64_t                up_row_bytes,
        uint64_t                down_expert_bytes,
        uint64_t                down_row_bytes,
        uint32_t                expert_in_dim,
        uint32_t                expert_mid_dim,
        uint32_t                out_dim,
        const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *weights,
        uint32_t                n_total_expert,
        uint32_t                n_expert,
        uint32_t                layer_index,
        const ds4_gpu_tensor *x,
        uint32_t                n_tokens,
        uint32_t                mid_token_stride) {
    return ds4_gpu_glm_routed_moe_batch_tensor_impl(out,
                                                    mid,
                                                    model_map,
                                                    model_size,
                                                    gate_offset,
                                                    up_offset,
                                                    down_offset,
                                                    gate_type,
                                                    up_type,
                                                    down_type,
                                                    gate_expert_bytes,
                                                    gate_row_bytes,
                                                    up_expert_bytes,
                                                    up_row_bytes,
                                                    down_expert_bytes,
                                                    down_row_bytes,
                                                    expert_in_dim,
                                                    expert_mid_dim,
                                                    out_dim,
                                                    selected,
                                                    weights,
                                                    n_total_expert,
                                                    n_expert,
                                                    layer_index,
                                                    x,
                                                    n_tokens,
                                                    mid_token_stride,
                                                    false,
                                                    false);
}
