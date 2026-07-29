static int ds4_gpu_encode_get_rows_f16(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        weight,
        NSUInteger           weight_offset,
        id<MTLBuffer>        tokens,
        NSUInteger           tokens_offset,
        id<MTLBuffer>        out,
        NSUInteger           out_offset,
        uint32_t             n_vocab,
        uint32_t             n_tokens,
        uint32_t             n_embd) {
    if (!cb || !weight || !tokens || !out || n_vocab == 0 || n_tokens == 0 || n_embd == 0) {
        return 0;
    }

    const uint64_t src_row_bytes = (uint64_t)n_embd * sizeof(uint16_t);
    const uint64_t dst_row_bytes = (uint64_t)n_embd * sizeof(float);
    const uint64_t token_bytes = (uint64_t)n_tokens * sizeof(int32_t);
    ds4_gpu_get_rows_args args = {
        .ne00t = (int32_t)n_embd,
        .ne00 = (int32_t)n_embd,
        .nb01 = src_row_bytes,
        .nb02 = (uint64_t)n_vocab * src_row_bytes,
        .nb03 = (uint64_t)n_vocab * src_row_bytes,
        .ne10 = (int32_t)n_tokens,
        .nb10 = sizeof(int32_t),
        .nb11 = token_bytes,
        .nb12 = token_bytes,
        .nb1 = dst_row_bytes,
        .nb2 = (uint64_t)n_tokens * dst_row_bytes,
        .nb3 = (uint64_t)n_tokens * dst_row_bytes,
    };

    NSUInteger nth = (NSUInteger)n_embd;
    const NSUInteger max_threads = g_get_rows_f16_pipeline.maxTotalThreadsPerThreadgroup;
    if (nth > max_threads) nth = max_threads;
    if (nth == 0) nth = 1;
    const NSUInteger nw0 = ((NSUInteger)n_embd + nth - 1u) / nth;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:g_get_rows_f16_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:weight offset:weight_offset atIndex:1];
    [enc setBuffer:tokens offset:tokens_offset atIndex:2];
    [enc setBuffer:out offset:out_offset atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake(nw0 * n_tokens, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

static uint64_t ds4_gpu_q8_0_row_bytes(uint32_t n_embd) {
    return (((uint64_t)n_embd + 31u) / 32u) * 34u;
}

static int ds4_gpu_quant_row_bytes(
        uint32_t  type,
        uint32_t  n_embd,
        uint64_t *row_bytes_out) {
    if (!row_bytes_out || n_embd == 0) return 0;
    switch (type) {
    case DS4_METAL_TENSOR_Q8_0:
        *row_bytes_out = (((uint64_t)n_embd + 31u) / 32u) * 34u;
        return 1;
    case DS4_METAL_TENSOR_Q4_0:
        *row_bytes_out = (((uint64_t)n_embd + 31u) / 32u) * 18u;
        return 1;
    case DS4_METAL_TENSOR_Q4_K:
        if ((n_embd % 256u) != 0) return 0;
        *row_bytes_out = ((uint64_t)n_embd / 256u) * 144u;
        return 1;
    default:
        return 0;
    }
}

static int ds4_gpu_q8_0_table_bytes(
        uint32_t  n_vocab,
        uint32_t  n_embd,
        uint64_t *bytes_out) {
    if (!bytes_out || n_vocab == 0 || n_embd == 0) return 0;
    const uint64_t row_bytes = ds4_gpu_q8_0_row_bytes(n_embd);
    if (row_bytes != 0 && (uint64_t)n_vocab > UINT64_MAX / row_bytes) return 0;
    *bytes_out = (uint64_t)n_vocab * row_bytes;
    return 1;
}

static int ds4_gpu_quant_table_bytes(
        uint32_t  type,
        uint32_t  n_vocab,
        uint32_t  n_embd,
        uint64_t *bytes_out) {
    if (!bytes_out || n_vocab == 0 || n_embd == 0) return 0;
    uint64_t row_bytes = 0;
    if (!ds4_gpu_quant_row_bytes(type, n_embd, &row_bytes)) return 0;
    if (row_bytes != 0 && (uint64_t)n_vocab > UINT64_MAX / row_bytes) return 0;
    *bytes_out = (uint64_t)n_vocab * row_bytes;
    return 1;
}

static int ds4_gpu_encode_get_rows_q8_0(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        weight,
        NSUInteger           weight_offset,
        id<MTLBuffer>        tokens,
        NSUInteger           tokens_offset,
        const int32_t       *single_token,
        id<MTLBuffer>        out,
        NSUInteger           out_offset,
        uint32_t             n_vocab,
        uint32_t             n_tokens,
        uint32_t             n_embd) {
    if (!cb || !weight || !out || n_vocab == 0 || n_tokens == 0 || n_embd == 0) {
        return 0;
    }
    if (!tokens && (!single_token || n_tokens != 1)) {
        return 0;
    }

    ds4_gpu_get_rows_q8_0_args args = {
        .n_embd = (int32_t)n_embd,
        .n_vocab = (int32_t)n_vocab,
        .n_tokens = (int32_t)n_tokens,
        .src_row_bytes = ds4_gpu_q8_0_row_bytes(n_embd),
        .dst_row_bytes = (uint64_t)n_embd * sizeof(float),
        .token_stride = sizeof(int32_t),
    };

    NSUInteger nth = 32u;
    const NSUInteger max_threads = g_get_rows_q8_0_pipeline.maxTotalThreadsPerThreadgroup;
    if (nth > max_threads) nth = max_threads;
    if (nth == 0) nth = 1;
    const NSUInteger nblocks = ((NSUInteger)n_embd + 31u) / 32u;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:g_get_rows_q8_0_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:weight offset:weight_offset atIndex:1];
    if (tokens) {
        [enc setBuffer:tokens offset:tokens_offset atIndex:2];
    } else {
        [enc setBytes:single_token length:sizeof(*single_token) atIndex:2];
    }
    [enc setBuffer:out offset:out_offset atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake(nblocks, n_tokens, 1)
         threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

static int ds4_gpu_encode_get_rows_quant(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        weight,
        NSUInteger           weight_offset,
        uint32_t             weight_type,
        id<MTLBuffer>        tokens,
        NSUInteger           tokens_offset,
        const int32_t       *single_token,
        id<MTLBuffer>        out,
        NSUInteger           out_offset,
        uint32_t             n_vocab,
        uint32_t             n_tokens,
        uint32_t             n_embd) {
    if (weight_type == DS4_METAL_TENSOR_Q8_0) {
        return ds4_gpu_encode_get_rows_q8_0(cb,
                                            weight,
                                            weight_offset,
                                            tokens,
                                            tokens_offset,
                                            single_token,
                                            out,
                                            out_offset,
                                            n_vocab,
                                            n_tokens,
                                            n_embd);
    }
    if (!cb || !weight || !out || n_vocab == 0 || n_tokens == 0 || n_embd == 0) {
        return 0;
    }
    if (!tokens && (!single_token || n_tokens != 1)) {
        return 0;
    }

    uint64_t src_row_bytes = 0;
    if (!ds4_gpu_quant_row_bytes(weight_type, n_embd, &src_row_bytes)) return 0;
    ds4_gpu_get_rows_q8_0_args args = {
        .n_embd = (int32_t)n_embd,
        .n_vocab = (int32_t)n_vocab,
        .n_tokens = (int32_t)n_tokens,
        .src_row_bytes = src_row_bytes,
        .dst_row_bytes = (uint64_t)n_embd * sizeof(float),
        .token_stride = sizeof(int32_t),
    };

    id<MTLComputePipelineState> pipeline = nil;
    NSUInteger block_width = 0;
    if (weight_type == DS4_METAL_TENSOR_Q4_0) {
        pipeline = g_get_rows_q4_0_pipeline;
        block_width = 32u;
    } else if (weight_type == DS4_METAL_TENSOR_Q4_K) {
        pipeline = g_get_rows_q4_K_pipeline;
        block_width = 256u;
    }
    if (!pipeline || block_width == 0) return 0;

    NSUInteger nth = block_width;
    const NSUInteger max_threads = pipeline.maxTotalThreadsPerThreadgroup;
    if (nth > max_threads) nth = max_threads;
    if (nth == 0) nth = 1;
    const NSUInteger nblocks = ((NSUInteger)n_embd + block_width - 1u) / block_width;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:weight offset:weight_offset atIndex:1];
    if (tokens) {
        [enc setBuffer:tokens offset:tokens_offset atIndex:2];
    } else {
        [enc setBytes:single_token length:sizeof(*single_token) atIndex:2];
    }
    [enc setBuffer:out offset:out_offset atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake(nblocks, n_tokens, 1)
         threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

static int ds4_gpu_encode_repeat_hc_embedding(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        rows,
        NSUInteger           rows_offset,
        id<MTLBuffer>        out,
        NSUInteger           out_offset,
        uint32_t             n_tokens,
        uint32_t             n_embd,
        uint32_t             n_hc) {
    if (!cb || !rows || !out || n_tokens == 0 || n_embd == 0 || n_hc == 0) return 0;

    const uint64_t embd_bytes = (uint64_t)n_embd * sizeof(float);
    ds4_gpu_repeat_args args = {
        .ne00 = (int32_t)n_embd,
        .ne01 = 1,
        .ne02 = (int32_t)n_tokens,
        .ne03 = 1,
        .nb00 = sizeof(float),
        .nb01 = embd_bytes,
        .nb02 = embd_bytes,
        .nb03 = (uint64_t)n_tokens * embd_bytes,
        .ne0 = (int32_t)n_embd,
        .ne1 = (int32_t)n_hc,
        .ne2 = (int32_t)n_tokens,
        .ne3 = 1,
        .nb0 = sizeof(float),
        .nb1 = embd_bytes,
        .nb2 = (uint64_t)n_hc * embd_bytes,
        .nb3 = (uint64_t)n_tokens * n_hc * embd_bytes,
    };

    NSUInteger nth = (NSUInteger)n_embd;
    const NSUInteger max_threads = g_repeat_f32_pipeline.maxTotalThreadsPerThreadgroup;
    if (nth > max_threads) nth = max_threads;
    if (nth == 0) nth = 1;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:g_repeat_f32_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:rows offset:rows_offset atIndex:1];
    [enc setBuffer:out offset:out_offset atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake(n_hc, n_tokens, 1)
         threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

int ds4_gpu_embed_token_q8_0_tensor(
        ds4_gpu_tensor *out,
        const void       *model_map,
        uint64_t          model_size,
        uint64_t          weight_offset,
        uint32_t          n_vocab,
        uint32_t          token,
        uint32_t          n_embd) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!out || !model_map || n_vocab == 0 || token >= n_vocab || n_embd == 0) {
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> outbuf = ds4_gpu_tensor_buffer(out);
        const uint64_t out_bytes = (uint64_t)n_embd * sizeof(float);
        if (!outbuf || ds4_gpu_tensor_bytes(out) < out_bytes) {
            fprintf(stderr, "ds4: Metal Q8_0 embedding received undersized output buffer\n");
            return 0;
        }

        uint64_t weight_bytes = 0;
        if (!ds4_gpu_q8_0_table_bytes(n_vocab, n_embd, &weight_bytes) ||
            weight_offset > model_size ||
            weight_bytes > model_size - weight_offset) {
            fprintf(stderr, "ds4: Metal Q8_0 embedding range is outside the mapped model\n");
            return 0;
        }

        uint64_t inner_offset = 0;
        uint32_t token_for_kernel = token;
        id<MTLBuffer> wbuf = nil;
        const bool exact_token_row =
            getenv("DS4_METAL_DISABLE_TOKEN_EMBED_EXACT_VIEW") == NULL;
        if (exact_token_row) {
            const uint64_t row_bytes = ds4_gpu_q8_0_row_bytes(n_embd);
            const uint64_t token_rel = (uint64_t)token * row_bytes;
            if (token_rel > weight_bytes || row_bytes > weight_bytes - token_rel) {
                fprintf(stderr, "ds4: Metal Q8_0 embedding token row is outside the mapped table\n");
                return 0;
            }
            wbuf = ds4_gpu_wrap_model_exact_range(model_map,
                                                  model_size,
                                                  weight_offset + token_rel,
                                                  row_bytes,
                                                  &inner_offset);
            token_for_kernel = 0;
        } else {
            wbuf = ds4_gpu_wrap_model_range(model_map,
                                            model_size,
                                            weight_offset,
                                            weight_bytes,
                                            &inner_offset);
        }
        if (!wbuf) return 0;

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        const int32_t token_i32 = (int32_t)token_for_kernel;
        if (!ds4_gpu_encode_get_rows_q8_0(cb,
                                           wbuf,
                                           (NSUInteger)inner_offset,
                                           nil,
                                           0,
                                           &token_i32,
                                           outbuf,
                                           ds4_gpu_tensor_offset(out),
                                           n_vocab,
                                           1,
                                           n_embd)) {
            return 0;
        }

        if (!ds4_gpu_finish_command_buffer(cb, owned, "q8_0 embed token")) return 0;
    }

    return 1;
}

int ds4_gpu_embed_tokens_q8_0_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *tokens,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                n_vocab,
        uint32_t                n_tokens,
        uint32_t                n_embd) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!out || !tokens || !model_map || n_vocab == 0 || n_tokens == 0 || n_embd == 0) {
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> outbuf = ds4_gpu_tensor_buffer(out);
        id<MTLBuffer> tokbuf = ds4_gpu_tensor_buffer(tokens);
        const uint64_t out_bytes = (uint64_t)n_tokens * n_embd * sizeof(float);
        const uint64_t token_bytes = (uint64_t)n_tokens * sizeof(int32_t);
        if (!outbuf || !tokbuf ||
            ds4_gpu_tensor_bytes(out) < out_bytes ||
            ds4_gpu_tensor_bytes(tokens) < token_bytes) {
            fprintf(stderr, "ds4: Metal Q8_0 batched embedding received undersized buffers\n");
            return 0;
        }

        uint64_t weight_bytes = 0;
        if (!ds4_gpu_q8_0_table_bytes(n_vocab, n_embd, &weight_bytes) ||
            weight_offset > model_size ||
            weight_bytes > model_size - weight_offset) {
            fprintf(stderr, "ds4: Metal Q8_0 batched embedding range is outside the mapped model\n");
            return 0;
        }

        uint64_t inner_offset = 0;
        id<MTLBuffer> wbuf =
            ds4_gpu_wrap_model_range(model_map,
                                     model_size,
                                     weight_offset,
                                     weight_bytes,
                                     &inner_offset);
        if (!wbuf) return 0;

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        if (!ds4_gpu_encode_get_rows_q8_0(cb,
                                           wbuf,
                                           (NSUInteger)inner_offset,
                                           tokbuf,
                                           ds4_gpu_tensor_offset(tokens),
                                           NULL,
                                           outbuf,
                                           ds4_gpu_tensor_offset(out),
                                           n_vocab,
                                           n_tokens,
                                           n_embd)) {
            return 0;
        }

        if (!ds4_gpu_finish_command_buffer(cb, owned, "q8_0 embed tokens")) return 0;
    }

    return 1;
}

int ds4_gpu_embed_token_quant_tensor(
        ds4_gpu_tensor *out,
        const void       *model_map,
        uint64_t          model_size,
        uint64_t          weight_offset,
        uint32_t          weight_type,
        uint32_t          n_vocab,
        uint32_t          token,
        uint32_t          n_embd) {
    if (weight_type == DS4_METAL_TENSOR_Q8_0) {
        return ds4_gpu_embed_token_q8_0_tensor(out,
                                               model_map,
                                               model_size,
                                               weight_offset,
                                               n_vocab,
                                               token,
                                               n_embd);
    }
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!out || !model_map || n_vocab == 0 || token >= n_vocab || n_embd == 0) {
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> outbuf = ds4_gpu_tensor_buffer(out);
        const uint64_t out_bytes = (uint64_t)n_embd * sizeof(float);
        if (!outbuf || ds4_gpu_tensor_bytes(out) < out_bytes) {
            fprintf(stderr, "ds4: Metal quant embedding received undersized output buffer\n");
            return 0;
        }

        uint64_t weight_bytes = 0;
        uint64_t row_bytes = 0;
        if (!ds4_gpu_quant_table_bytes(weight_type, n_vocab, n_embd, &weight_bytes) ||
            !ds4_gpu_quant_row_bytes(weight_type, n_embd, &row_bytes) ||
            weight_offset > model_size ||
            weight_bytes > model_size - weight_offset) {
            fprintf(stderr, "ds4: Metal quant embedding range is outside the mapped model\n");
            return 0;
        }

        const uint64_t token_rel = (uint64_t)token * row_bytes;
        if (token_rel > weight_bytes || row_bytes > weight_bytes - token_rel) {
            fprintf(stderr, "ds4: Metal quant embedding token row is outside the mapped table\n");
            return 0;
        }

        uint64_t inner_offset = 0;
        id<MTLBuffer> wbuf =
            ds4_gpu_wrap_model_exact_range(model_map,
                                           model_size,
                                           weight_offset + token_rel,
                                           row_bytes,
                                           &inner_offset);
        if (!wbuf) return 0;

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        const int32_t token_i32 = 0;
        if (!ds4_gpu_encode_get_rows_quant(cb,
                                           wbuf,
                                           (NSUInteger)inner_offset,
                                           weight_type,
                                           nil,
                                           0,
                                           &token_i32,
                                           outbuf,
                                           ds4_gpu_tensor_offset(out),
                                           n_vocab,
                                           1,
                                           n_embd)) {
            return 0;
        }

        if (!ds4_gpu_finish_command_buffer(cb, owned, "quant embed token")) return 0;
    }

    return 1;
}

int ds4_gpu_embed_tokens_quant_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *tokens,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                weight_type,
        uint32_t                n_vocab,
        uint32_t                n_tokens,
        uint32_t                n_embd) {
    if (weight_type == DS4_METAL_TENSOR_Q8_0) {
        return ds4_gpu_embed_tokens_q8_0_tensor(out,
                                                tokens,
                                                model_map,
                                                model_size,
                                                weight_offset,
                                                n_vocab,
                                                n_tokens,
                                                n_embd);
    }
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!out || !tokens || !model_map || n_vocab == 0 || n_tokens == 0 || n_embd == 0) {
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> outbuf = ds4_gpu_tensor_buffer(out);
        id<MTLBuffer> tokbuf = ds4_gpu_tensor_buffer(tokens);
        const uint64_t out_bytes = (uint64_t)n_tokens * n_embd * sizeof(float);
        const uint64_t token_bytes = (uint64_t)n_tokens * sizeof(int32_t);
        if (!outbuf || !tokbuf ||
            ds4_gpu_tensor_bytes(out) < out_bytes ||
            ds4_gpu_tensor_bytes(tokens) < token_bytes) {
            fprintf(stderr, "ds4: Metal quant batched embedding received undersized buffers\n");
            return 0;
        }

        uint64_t weight_bytes = 0;
        if (!ds4_gpu_quant_table_bytes(weight_type, n_vocab, n_embd, &weight_bytes) ||
            weight_offset > model_size ||
            weight_bytes > model_size - weight_offset) {
            fprintf(stderr, "ds4: Metal quant batched embedding range is outside the mapped model\n");
            return 0;
        }

        uint64_t inner_offset = 0;
        id<MTLBuffer> wbuf =
            ds4_gpu_wrap_model_range(model_map,
                                     model_size,
                                     weight_offset,
                                     weight_bytes,
                                     &inner_offset);
        if (!wbuf) return 0;

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        if (!ds4_gpu_encode_get_rows_quant(cb,
                                           wbuf,
                                           (NSUInteger)inner_offset,
                                           weight_type,
                                           tokbuf,
                                           ds4_gpu_tensor_offset(tokens),
                                           NULL,
                                           outbuf,
                                           ds4_gpu_tensor_offset(out),
                                           n_vocab,
                                           n_tokens,
                                           n_embd)) {
            return 0;
        }

        if (!ds4_gpu_finish_command_buffer(cb, owned, "quant embed tokens")) return 0;
    }

    return 1;
}

int ds4_gpu_embed_token_hc_tensor(
        ds4_gpu_tensor *out_hc,
        const void       *model_map,
        uint64_t          model_size,
        uint64_t          weight_offset,
        uint32_t          n_vocab,
        uint32_t          token,
        uint32_t          n_embd,
        uint32_t          n_hc) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!out_hc || !model_map || n_vocab == 0 || token >= n_vocab || n_embd == 0 || n_hc == 0) {
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> outbuf = ds4_gpu_tensor_buffer(out_hc);
        const uint64_t out_bytes = (uint64_t)n_embd * n_hc * sizeof(float);
        if (!outbuf || ds4_gpu_tensor_bytes(out_hc) < out_bytes) {
            fprintf(stderr, "ds4: Metal graph embedding received undersized HC output buffer\n");
            return 0;
        }

        const uint64_t weight_bytes = (uint64_t)n_vocab * n_embd * sizeof(uint16_t);
        if (weight_offset > model_size || weight_bytes > model_size - weight_offset) {
            fprintf(stderr, "ds4: Metal graph embedding range is outside the mapped model\n");
            return 0;
        }

        uint64_t inner_offset = 0;
        id<MTLBuffer> wbuf =
            ds4_gpu_wrap_model_range(model_map,
                                     model_size,
                                     weight_offset,
                                     weight_bytes,
                                     &inner_offset);
        if (!wbuf) return 0;

        const NSUInteger row_bytes = (NSUInteger)n_embd * sizeof(float);
        if (!ds4_gpu_ensure_scratch_buffer(&g_embed_rows_buffer,
                                             &g_embed_rows_bytes,
                                             row_bytes,
                                             "ds4_embed_rows")) {
            return 0;
        }

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        const int32_t token_i32 = (int32_t)token;
        const uint64_t src_row_bytes = (uint64_t)n_embd * sizeof(uint16_t);
        const uint64_t dst_row_bytes = (uint64_t)n_embd * sizeof(float);
        ds4_gpu_get_rows_args args = {
            .ne00t = (int32_t)n_embd,
            .ne00 = (int32_t)n_embd,
            .nb01 = src_row_bytes,
            .nb02 = (uint64_t)n_vocab * src_row_bytes,
            .nb03 = (uint64_t)n_vocab * src_row_bytes,
            .ne10 = 1,
            .nb10 = sizeof(int32_t),
            .nb11 = sizeof(int32_t),
            .nb12 = sizeof(int32_t),
            .nb1 = dst_row_bytes,
            .nb2 = dst_row_bytes,
            .nb3 = dst_row_bytes,
        };
        NSUInteger nth = (NSUInteger)n_embd;
        const NSUInteger max_threads = g_get_rows_f16_pipeline.maxTotalThreadsPerThreadgroup;
        if (nth > max_threads) nth = max_threads;
        if (nth == 0) nth = 1;
        const NSUInteger nw0 = ((NSUInteger)n_embd + nth - 1u) / nth;
        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:g_get_rows_f16_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:wbuf offset:(NSUInteger)inner_offset atIndex:1];
        [enc setBytes:&token_i32 length:sizeof(token_i32) atIndex:2];
        [enc setBuffer:g_embed_rows_buffer offset:0 atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(nw0, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_encode_repeat_hc_embedding(cb,
                                                  g_embed_rows_buffer,
                                                  0,
                                                  outbuf,
                                                  ds4_gpu_tensor_offset(out_hc),
                                                  1,
                                                  n_embd,
                                                  n_hc)) {
            return 0;
        }

        if (!ds4_gpu_finish_command_buffer(cb, owned, "graph embed token")) return 0;
    }

    return 1;
}

int ds4_gpu_embed_tokens_hc_tensor(
        ds4_gpu_tensor       *out_hc,
        const ds4_gpu_tensor *tokens,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                n_vocab,
        uint32_t                n_tokens,
        uint32_t                n_embd,
        uint32_t                n_hc) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!out_hc || !tokens || !model_map || n_vocab == 0 || n_tokens == 0 || n_embd == 0 || n_hc == 0) {
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> outbuf = ds4_gpu_tensor_buffer(out_hc);
        id<MTLBuffer> tokbuf = ds4_gpu_tensor_buffer(tokens);
        const uint64_t out_bytes = (uint64_t)n_tokens * n_embd * n_hc * sizeof(float);
        const uint64_t token_bytes = (uint64_t)n_tokens * sizeof(int32_t);
        if (!outbuf || !tokbuf ||
            ds4_gpu_tensor_bytes(out_hc) < out_bytes ||
            ds4_gpu_tensor_bytes(tokens) < token_bytes) {
            fprintf(stderr, "ds4: Metal graph batched embedding received undersized buffers\n");
            return 0;
        }

        const uint64_t weight_bytes = (uint64_t)n_vocab * n_embd * sizeof(uint16_t);
        if (weight_offset > model_size || weight_bytes > model_size - weight_offset) {
            fprintf(stderr, "ds4: Metal graph batched embedding range is outside the mapped model\n");
            return 0;
        }

        uint64_t inner_offset = 0;
        id<MTLBuffer> wbuf =
            ds4_gpu_wrap_model_range(model_map,
                                     model_size,
                                     weight_offset,
                                     weight_bytes,
                                     &inner_offset);
        if (!wbuf) return 0;

        const NSUInteger rows_bytes = (NSUInteger)n_tokens * n_embd * sizeof(float);
        if (!ds4_gpu_ensure_scratch_buffer(&g_embed_rows_buffer,
                                             &g_embed_rows_bytes,
                                             rows_bytes,
                                             "ds4_embed_rows")) {
            return 0;
        }

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        if (!ds4_gpu_encode_get_rows_f16(cb,
                                           wbuf,
                                           (NSUInteger)inner_offset,
                                           tokbuf,
                                           ds4_gpu_tensor_offset(tokens),
                                           g_embed_rows_buffer,
                                           0,
                                           n_vocab,
                                           n_tokens,
                                           n_embd) ||
            !ds4_gpu_encode_repeat_hc_embedding(cb,
                                                  g_embed_rows_buffer,
                                                  0,
                                                  outbuf,
                                                  ds4_gpu_tensor_offset(out_hc),
                                                  n_tokens,
                                                  n_embd,
                                                  n_hc)) {
            return 0;
        }

        if (!ds4_gpu_finish_command_buffer(cb, owned, "graph embed tokens")) return 0;
    }

    return 1;
}
