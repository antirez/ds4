int ds4_gpu_indexer_score_one_tensor(
        ds4_gpu_tensor       *scores,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *index_comp,
        uint32_t                n_comp,
        uint32_t                n_head,
        uint32_t                head_dim,
        float                   scale) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!scores || !q || !weights || !index_comp ||
        n_comp == 0 || n_head == 0 || head_dim == 0) {
        return 0;
    }

    @autoreleasepool {
        const uint64_t q_bytes = (uint64_t)n_head * head_dim * sizeof(float);
        const uint64_t weight_bytes = (uint64_t)n_head * sizeof(float);
        const uint64_t comp_bytes = (uint64_t)n_comp * head_dim * sizeof(float);
        const uint64_t score_bytes = (uint64_t)n_comp * sizeof(float);
        id<MTLBuffer> qbuf = ds4_gpu_tensor_buffer(q);
        id<MTLBuffer> wbuf = ds4_gpu_tensor_buffer(weights);
        id<MTLBuffer> compbuf = ds4_gpu_tensor_buffer(index_comp);
        id<MTLBuffer> scorebuf = ds4_gpu_tensor_buffer(scores);
        if (!qbuf || !wbuf || !compbuf || !scorebuf ||
            ds4_gpu_tensor_bytes(q) < q_bytes ||
            ds4_gpu_tensor_bytes(weights) < weight_bytes ||
            ds4_gpu_tensor_bytes(index_comp) < comp_bytes ||
            ds4_gpu_tensor_bytes(scores) < score_bytes) {
            fprintf(stderr, "ds4: Metal graph indexer score received undersized buffers\n");
            return 0;
        }

        if (n_head == 64 && head_dim == 128) {
            id<MTLComputePipelineState> direct_pipeline =
                ds4_gpu_hot_pipeline(g_dsv4_indexer_score_one_direct_pipeline,
                                        "kernel_dsv4_indexer_score_one_direct");
            if (!direct_pipeline) return 0;

            ds4_gpu_dsv4_indexer_scores_fused_args args = {
                .n_comp = n_comp,
                .n_tokens = 1,
                .n_head = n_head,
                .head_dim = head_dim,
                .pos0 = 0,
                .ratio = 4,
                .q_token_stride = (uint64_t)n_head * head_dim * sizeof(float),
                .q_head_stride = (uint64_t)head_dim * sizeof(float),
                .weights_token_stride = (uint64_t)n_head * sizeof(float),
                .index_row_stride = (uint64_t)head_dim * sizeof(float),
                .score_token_stride = (uint64_t)n_comp * sizeof(float),
                .scale = scale,
            };

            int owned = 0;
            id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
            if (!cb) return 0;
            id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
            [enc setComputePipelineState:direct_pipeline];
            [enc setBytes:&args length:sizeof(args) atIndex:0];
            [enc setBuffer:qbuf offset:ds4_gpu_tensor_offset(q) atIndex:1];
            [enc setBuffer:wbuf offset:ds4_gpu_tensor_offset(weights) atIndex:2];
            [enc setBuffer:compbuf offset:ds4_gpu_tensor_offset(index_comp) atIndex:3];
            [enc setBuffer:scorebuf offset:ds4_gpu_tensor_offset(scores) atIndex:4];
            [enc setThreadgroupMemoryLength:(128u + 4u) * sizeof(float) atIndex:0];
            [enc dispatchThreadgroups:MTLSizeMake(n_comp, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
            ds4_gpu_end_compute_encoder(cb, enc);

            if (!ds4_gpu_finish_command_buffer(cb, owned, "indexer direct score")) return 0;
            return 1;
        }

        const uint64_t head_score_bytes = (uint64_t)n_comp * n_head * sizeof(float);
        if (!ds4_gpu_ensure_scratch_buffer(&g_indexer_head_scores_buffer,
                                             &g_indexer_head_scores_bytes,
                                             (NSUInteger)head_score_bytes,
                                             "ds4_indexer_head_scores")) {
            return 0;
        }

        ds4_gpu_q8_0_matvec_args dot_args =
            ds4_gpu_make_f32_mv_args(head_dim, n_comp, n_head);
        ds4_gpu_mv_dispatch dot_dispatch =
            ds4_gpu_make_plain_mv_dispatch(head_dim, 1);
        dot_args.nr0 = dot_dispatch.nr0;
        id<MTLComputePipelineState> dot_pipeline =
            ds4_gpu_get_mul_mv_pipeline(dot_dispatch.function_name, dot_dispatch.nsg);
        if (!dot_pipeline) return 0;
        ds4_gpu_dsv4_indexer_weighted_sum_args sum_args = {
            .ne00 = (int64_t)n_comp,
            .ne01 = 1,
            .ne02 = (int64_t)n_head,
            .nb00 = sizeof(float),
            .nb01 = (uint64_t)n_comp * sizeof(float),
            .nb02 = (uint64_t)n_comp * sizeof(float),
            .ne10 = (int64_t)n_head,
            .ne11 = 1,
            .nb10 = sizeof(float),
            .nb11 = (uint64_t)n_head * sizeof(float),
            .ne0 = (int64_t)n_comp,
            .ne1 = 1,
            .nb0 = sizeof(float),
            .nb1 = (uint64_t)n_comp * sizeof(float),
            .scale = scale,
        };

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:dot_pipeline];
        [enc setBytes:&dot_args length:sizeof(dot_args) atIndex:0];
        [enc setBuffer:compbuf offset:ds4_gpu_tensor_offset(index_comp) atIndex:1];
        [enc setBuffer:qbuf offset:ds4_gpu_tensor_offset(q) atIndex:2];
        [enc setBuffer:g_indexer_head_scores_buffer offset:0 atIndex:3];
        if (dot_dispatch.smem) {
            [enc setThreadgroupMemoryLength:dot_dispatch.smem atIndex:0];
        }
        [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)n_comp + (NSUInteger)dot_dispatch.nr0 - 1u) / (NSUInteger)dot_dispatch.nr0,
                                              n_head,
                                              1)
             threadsPerThreadgroup:MTLSizeMake(32, (NSUInteger)dot_dispatch.nsg, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:g_dsv4_indexer_weighted_sum_pipeline];
        [enc setBytes:&sum_args length:sizeof(sum_args) atIndex:0];
        [enc setBuffer:g_indexer_head_scores_buffer offset:0 atIndex:1];
        [enc setBuffer:wbuf offset:ds4_gpu_tensor_offset(weights) atIndex:2];
        [enc setBuffer:scorebuf offset:ds4_gpu_tensor_offset(scores) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)n_comp + 255u) / 256u, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "indexer score")) return 0;
    }

    return 1;
}

static int ds4_gpu_indexer_scores_batch_tensor(
        ds4_gpu_tensor       *scores,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *index_comp,
        uint32_t                n_comp,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_head,
        uint32_t                head_dim,
        uint32_t                ratio,
        float                   scale) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!scores || !q || !weights || !index_comp ||
        n_comp == 0 || n_tokens == 0 || n_head == 0 || head_dim == 0 || ratio == 0) {
        return 0;
    }

    @autoreleasepool {
        const uint64_t q_bytes = (uint64_t)n_tokens * n_head * head_dim * sizeof(float);
        const uint64_t weight_bytes = (uint64_t)n_tokens * n_head * sizeof(float);
        const uint64_t comp_bytes = (uint64_t)n_comp * head_dim * sizeof(float);
        const uint64_t score_bytes = (uint64_t)n_comp * n_tokens * sizeof(float);
        id<MTLBuffer> qbuf = ds4_gpu_tensor_buffer(q);
        id<MTLBuffer> wbuf = ds4_gpu_tensor_buffer(weights);
        id<MTLBuffer> compbuf = ds4_gpu_tensor_buffer(index_comp);
        id<MTLBuffer> scorebuf = ds4_gpu_tensor_buffer(scores);
        if (!qbuf || !wbuf || !compbuf || !scorebuf ||
            ds4_gpu_tensor_bytes(q) < q_bytes ||
            ds4_gpu_tensor_bytes(weights) < weight_bytes ||
            ds4_gpu_tensor_bytes(index_comp) < comp_bytes ||
            ds4_gpu_tensor_bytes(scores) < score_bytes) {
            fprintf(stderr, "ds4: Metal graph indexer prefill scores received undersized buffers\n");
            return 0;
        }
        if (head_dim != 128) {
            fprintf(stderr, "ds4: Metal fused DS4 indexer scores expect 128-wide rows\n");
            return 0;
        }
        /*
         * The NAX/TensorOps score builder is a prefill-only win.  At small
         * batches and in one-token decode the setup cost is not amortized, so
         * those paths keep the older direct/tiled score kernels.
         */
        const bool use_nax = ds4_gpu_mpp_available() && n_tokens >= 16u;
        id<MTLComputePipelineState> pipeline = ds4_gpu_get_pipeline(
            use_nax ? "kernel_dsv4_indexer_scores_nax" :
            (g_quality_mode ? "kernel_dsv4_indexer_scores_tiled_f32"
                            : "kernel_dsv4_indexer_scores_tiled"));
        if (!pipeline) return 0;

        ds4_gpu_dsv4_indexer_scores_fused_args args = {
            .n_comp = n_comp,
            .n_tokens = n_tokens,
            .n_head = n_head,
            .head_dim = head_dim,
            .pos0 = pos0,
            .ratio = ratio,
            .q_token_stride = (uint64_t)n_head * head_dim * sizeof(float),
            .q_head_stride = (uint64_t)head_dim * sizeof(float),
            .weights_token_stride = (uint64_t)n_head * sizeof(float),
            .index_row_stride = (uint64_t)head_dim * sizeof(float),
            .score_token_stride = (uint64_t)n_comp * sizeof(float),
            .scale = scale,
        };

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:qbuf offset:ds4_gpu_tensor_offset(q) atIndex:1];
        [enc setBuffer:wbuf offset:ds4_gpu_tensor_offset(weights) atIndex:2];
        [enc setBuffer:compbuf offset:ds4_gpu_tensor_offset(index_comp) atIndex:3];
        [enc setBuffer:scorebuf offset:ds4_gpu_tensor_offset(scores) atIndex:4];
        if (use_nax) {
            const NSUInteger q_shared = 2u * 32u * 32u;
            const NSUInteger k_shared = 32u * 128u;
            const NSUInteger dot_shared = 32u * 32u;
            [enc setThreadgroupMemoryLength:(q_shared + k_shared) * sizeof(uint16_t) +
                                            dot_shared * sizeof(float) atIndex:0];
            [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)n_comp + 31u) / 32u,
                                                  ((NSUInteger)n_tokens + 15u) / 16u,
                                                  1)
                 threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        } else if (g_quality_mode) {
            const NSUInteger q_shared = 8u * 128u;
            const NSUInteger k_shared = 32u * 128u;
            const NSUInteger dot_shared = 8u * 32u;
            [enc setThreadgroupMemoryLength:(q_shared + k_shared + dot_shared) * sizeof(float) atIndex:0];
            [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)n_comp + 31u) / 32u,
                                                  ((NSUInteger)n_tokens + 7u) / 8u,
                                                  1)
                 threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
        } else {
            const NSUInteger q_shared = 8u * 128u;
            const NSUInteger k_shared = 32u * 128u;
            const NSUInteger dot_shared = 8u * 32u;
            [enc setThreadgroupMemoryLength:(q_shared + k_shared) * sizeof(uint16_t) +
                                            dot_shared * sizeof(float) atIndex:0];
            [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)n_comp + 31u) / 32u,
                                                  ((NSUInteger)n_tokens + 7u) / 8u,
                                                  1)
                 threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
        }
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "indexer prefill scores")) return 0;
    }

    return 1;
}

int ds4_gpu_indexer_scores_prefill_tensor(
        ds4_gpu_tensor       *scores,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *index_comp,
        uint32_t                n_comp,
        uint32_t                n_tokens,
        uint32_t                n_head,
        uint32_t                head_dim,
        uint32_t                ratio,
        float                   scale) {
    return ds4_gpu_indexer_scores_batch_tensor(scores,
                                                 q,
                                                 weights,
                                                 index_comp,
                                                 n_comp,
                                                 n_tokens,
                                                 0,
                                                 n_head,
                                                 head_dim,
                                                 ratio,
                                                 scale);
}

int ds4_gpu_indexer_scores_decode_batch_tensor(
        ds4_gpu_tensor       *scores,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *index_comp,
        uint32_t                n_comp,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_head,
        uint32_t                head_dim,
        uint32_t                ratio,
        float                   scale) {
    return ds4_gpu_indexer_scores_batch_tensor(scores,
                                                 q,
                                                 weights,
                                                 index_comp,
                                                 n_comp,
                                                 n_tokens,
                                                 pos0,
                                                 n_head,
                                                 head_dim,
                                                 ratio,
                                                 scale);
}

int ds4_gpu_indexer_topk_tensor(
        ds4_gpu_tensor       *selected,
        const ds4_gpu_tensor *scores,
        uint32_t                n_comp,
        uint32_t                n_tokens,
        uint32_t                top_k) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!selected || !scores || n_comp == 0 || n_tokens == 0 || top_k == 0 || top_k > n_comp) return 0;

    @autoreleasepool {
        const uint64_t score_bytes = (uint64_t)n_comp * n_tokens * sizeof(float);
        const uint64_t selected_bytes = (uint64_t)top_k * n_tokens * sizeof(uint32_t);
        id<MTLBuffer> scorebuf = ds4_gpu_tensor_buffer(scores);
        id<MTLBuffer> selbuf = ds4_gpu_tensor_buffer(selected);
        if (!scorebuf || !selbuf ||
            ds4_gpu_tensor_bytes(scores) < score_bytes ||
            ds4_gpu_tensor_bytes(selected) < selected_bytes) {
            fprintf(stderr, "ds4: Metal graph indexer top-k received undersized buffers\n");
            return 0;
        }
        NSUInteger max_threads = g_argsort_f32_i32_desc_pipeline.maxTotalThreadsPerThreadgroup;
        if (max_threads == 0) max_threads = 256;
        int32_t nth = 1;
        while ((uint32_t)nth < n_comp && (uint64_t)2u * (uint64_t)nth <= (uint64_t)max_threads) {
            nth *= 2;
        }
        const int32_t npr = (int32_t)((n_comp + (uint32_t)nth - 1u) / (uint32_t)nth);
        const int32_t block_top_k = (int32_t)(top_k < (uint32_t)nth ? top_k : (uint32_t)nth);
        int32_t work_width = (int32_t)top_k;
        if (npr > 1) {
            const int32_t last_block = (int32_t)n_comp - (npr - 1) * nth;
            work_width = (npr - 1) * block_top_k + (last_block < block_top_k ? last_block : block_top_k);
        }
        const uint64_t scratch_row_bytes = (uint64_t)work_width * sizeof(uint32_t);
        const bool one_pass = npr <= 1;
        const uint64_t scratch_bytes = one_pass ? scratch_row_bytes * n_tokens :
            2u * scratch_row_bytes * n_tokens;
        if (!ds4_gpu_ensure_scratch_buffer(&g_indexer_topk_buffer,
                                             &g_indexer_topk_bytes,
                                             (NSUInteger)scratch_bytes,
                                             "ds4_indexer_topk")) {
            return 0;
        }

        ds4_gpu_kargs_argsort args = {
            .ne00 = (int32_t)n_comp,
            .ne01 = (int32_t)n_tokens,
            .ne02 = 1,
            .ne03 = 1,
            .nb00 = sizeof(float),
            .nb01 = (uint64_t)n_comp * sizeof(float),
            .nb02 = (uint64_t)n_comp * n_tokens * sizeof(float),
            .nb03 = (uint64_t)n_comp * n_tokens * sizeof(float),
            .ne0 = work_width,
            .ne1 = (int32_t)n_tokens,
            .ne2 = 1,
            .ne3 = 1,
            .top_k = block_top_k,
        };
        // kernel_argsort_f32_i32_desc stages the block's scores behind the
        // index array: nth int32 indices + nth float scores.
        const NSUInteger smem = (((NSUInteger)nth * (sizeof(int32_t) + sizeof(float))) + 15u) & ~(NSUInteger)15u;

        NSUInteger cur_off = 0;
        NSUInteger next_off = (NSUInteger)scratch_row_bytes * n_tokens;
        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:g_argsort_f32_i32_desc_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:scorebuf offset:ds4_gpu_tensor_offset(scores) atIndex:1];
        [enc setBuffer:one_pass ? selbuf : g_indexer_topk_buffer
              offset:one_pass ? ds4_gpu_tensor_offset(selected) : cur_off
             atIndex:2];
        [enc setThreadgroupMemoryLength:smem atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)npr * n_tokens, 1, 1)
             threadsPerThreadgroup:MTLSizeMake((NSUInteger)nth, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        int32_t len = block_top_k;
        while (len < work_width) {
            const int32_t nm = (work_width + 2 * len - 1) / (2 * len);
            const bool final_merge = nm == 1;
            NSUInteger merge_threads = g_argsort_merge_f32_i32_desc_pipeline.maxTotalThreadsPerThreadgroup;
            if (merge_threads == 0 || merge_threads > 512u) merge_threads = 512u;
            if (merge_threads > (NSUInteger)len) merge_threads = (NSUInteger)len;
            if (merge_threads == 0) merge_threads = 1;

            ds4_gpu_kargs_argsort_merge merge_args = {
                .ne00 = (int64_t)n_comp,
                .ne01 = (int64_t)n_tokens,
                .ne02 = 1,
                .ne03 = 1,
                .nb00 = sizeof(float),
                .nb01 = (uint64_t)n_comp * sizeof(float),
                .nb02 = (uint64_t)n_comp * n_tokens * sizeof(float),
                .nb03 = (uint64_t)n_comp * n_tokens * sizeof(float),
                .ne0 = work_width,
                .ne1 = (int32_t)n_tokens,
                .ne2 = 1,
                .ne3 = 1,
                .top_k = nm == 1 ? (int32_t)top_k : work_width,
                .len = len,
            };

            enc = ds4_gpu_compute_encoder(cb);
            [enc setComputePipelineState:g_argsort_merge_f32_i32_desc_pipeline];
            [enc setBytes:&merge_args length:sizeof(merge_args) atIndex:0];
            [enc setBuffer:scorebuf offset:ds4_gpu_tensor_offset(scores) atIndex:1];
            [enc setBuffer:g_indexer_topk_buffer offset:cur_off atIndex:2];
            [enc setBuffer:final_merge ? selbuf : g_indexer_topk_buffer
                  offset:final_merge ? ds4_gpu_tensor_offset(selected) : next_off
                 atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)nm * n_tokens, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(merge_threads, 1, 1)];
            ds4_gpu_end_compute_encoder(cb, enc);

            const NSUInteger tmp = cur_off;
            cur_off = next_off;
            next_off = tmp;
            len <<= 1;
        }

        if (!ds4_gpu_finish_command_buffer(cb, owned, "indexer top-k")) return 0;
    }

    return 1;
}

int ds4_gpu_argmax_tensor(
        ds4_gpu_tensor       *out_idx,
        const ds4_gpu_tensor *logits,
        uint32_t                n_vocab) {
    if (!out_idx || !logits || n_vocab == 0) return 0;
    if (ds4_gpu_tensor_bytes(out_idx) < sizeof(int32_t) ||
        ds4_gpu_tensor_bytes(logits) < (uint64_t)n_vocab * sizeof(float)) {
        fprintf(stderr, "ds4: Metal graph argmax received undersized buffers\n");
        return 0;
    }

    return ds4_gpu_indexer_topk_tensor(out_idx, logits, n_vocab, 1, 1);
}

int ds4_gpu_dsv4_topk_mask_tensor(
        ds4_gpu_tensor       *mask,
        const ds4_gpu_tensor *topk,
        uint32_t                n_comp,
        uint32_t                n_tokens,
        uint32_t                top_k) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!mask || !topk || n_comp == 0 || n_tokens == 0 || top_k == 0) return 0;

    @autoreleasepool {
        const uint64_t topk_bytes = (uint64_t)top_k * n_tokens * sizeof(int32_t);
        const uint64_t mask_bytes = (uint64_t)n_comp * n_tokens * sizeof(float);
        id<MTLBuffer> topkbuf = ds4_gpu_tensor_buffer(topk);
        id<MTLBuffer> maskbuf = ds4_gpu_tensor_buffer(mask);
        if (!topkbuf || !maskbuf ||
            ds4_gpu_tensor_bytes(topk) < topk_bytes ||
            ds4_gpu_tensor_bytes(mask) < mask_bytes) {
            fprintf(stderr, "ds4: Metal dsv4 top-k mask received undersized buffers\n");
            return 0;
        }

        ds4_gpu_dsv4_topk_mask_args args = {
            .ne00 = (int64_t)top_k,
            .ne01 = (int64_t)n_tokens,
            .nb00 = sizeof(int32_t),
            .nb01 = (uint64_t)top_k * sizeof(int32_t),
            .ne0 = (int64_t)n_comp,
            .ne1 = (int64_t)n_tokens,
            .nb0 = sizeof(float),
            .nb1 = (uint64_t)n_comp * sizeof(float),
        };

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:g_dsv4_topk_mask_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:topkbuf offset:ds4_gpu_tensor_offset(topk) atIndex:1];
        [enc setBuffer:maskbuf offset:ds4_gpu_tensor_offset(mask) atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake((((NSUInteger)n_comp * n_tokens) + 255u) / 256u, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:g_dsv4_topk_mask_scatter_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:topkbuf offset:ds4_gpu_tensor_offset(topk) atIndex:1];
        [enc setBuffer:maskbuf offset:ds4_gpu_tensor_offset(mask) atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake((((NSUInteger)top_k * n_tokens) + 255u) / 256u, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "dsv4 top-k mask")) return 0;
    }

    return 1;
}

static int ds4_gpu_matmul_q8_0_legacy_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok,
        bool                    prefer_decode_mpp,
        bool                    force_model_view) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if ((in_dim & 31u) != 0 ||
        in_dim > UINT32_MAX || out_dim > UINT32_MAX || n_tok > UINT32_MAX) {
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> xbuf = ds4_gpu_tensor_buffer(x);
        id<MTLBuffer> outbuf = ds4_gpu_tensor_buffer(out);
        const uint64_t x_bytes = n_tok * in_dim * sizeof(float);
        const uint64_t out_bytes = n_tok * out_dim * sizeof(float);
        if (!xbuf || !outbuf ||
            ds4_gpu_tensor_bytes(x) < x_bytes ||
            ds4_gpu_tensor_bytes(out) < out_bytes) {
            fprintf(stderr, "ds4: Metal Q8_0 tensor matmul received undersized activation buffers\n");
            return 0;
        }

        const uint64_t blocks = in_dim / 32;
        const uint64_t row_bytes = blocks * 34;
        const uint64_t weight_bytes = out_dim * row_bytes;
        if (weight_offset > model_size || weight_bytes > model_size - weight_offset) {
            fprintf(stderr, "ds4: Metal Q8_0 tensor matmul range is outside the mapped model\n");
            return 0;
        }

        uint64_t inner_offset = 0;
        id<MTLBuffer> wbuf = force_model_view ?
            ds4_gpu_wrap_model_range(model_map,
                                     model_size,
                                     weight_offset,
                                     weight_bytes,
                                     &inner_offset) :
            ds4_gpu_wrap_q8_decode_model_range(model_map,
                                               model_size,
                                               weight_offset,
                                               weight_bytes,
                                               n_tok,
                                               &inner_offset);
        if (!wbuf) {
            return 0;
        }

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        if (n_tok == 1) {
            if (ds4_gpu_mpp_available() &&
                (prefer_decode_mpp || getenv("DS4_METAL_Q8_DECODE_MPP") != NULL) &&
                getenv("DS4_METAL_DISABLE_Q8_DECODE_MPP") == NULL &&
                (in_dim % 64u) == 0) {
                const char *nax_fn = "kernel_mul_mm_q8_0_f32_nax_direct_rhs";
                id<MTLComputePipelineState> mpp_pipeline =
                    ds4_gpu_get_mul_mm_pipeline(nax_fn, false, false);
                if (mpp_pipeline) {
                    ds4_gpu_mul_mm_args args =
                        ds4_gpu_make_mm_args(in_dim, out_dim, n_tok, row_bytes);

                    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
                    [enc setComputePipelineState:mpp_pipeline];
                    [enc setBytes:&args length:sizeof(args) atIndex:0];
                    [enc setBuffer:wbuf offset:(NSUInteger)inner_offset atIndex:1];
                    [enc setBuffer:xbuf offset:ds4_gpu_tensor_offset(x) atIndex:2];
                    [enc setBuffer:outbuf offset:ds4_gpu_tensor_offset(out) atIndex:3];
                    [enc setThreadgroupMemoryLength:64u * 32u * sizeof(uint16_t) atIndex:0];
                    [enc dispatchThreadgroups:MTLSizeMake(1u,
                                                          ((NSUInteger)out_dim + 63u) / 64u,
                                                          1u)
                         threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                    ds4_gpu_end_compute_encoder(cb, enc);

                    if (!ds4_gpu_finish_command_buffer(cb, owned, "Q8_0 decode MPP matmul")) {
                        return 0;
                    }
                    return 1;
                }
                ds4_gpu_warn_mpp_fallback();
            }

            ds4_gpu_q8_0_matvec_args mv_args = ds4_gpu_make_q8_0_mv_args(in_dim, out_dim);
            ds4_gpu_mv_dispatch mv_dispatch = ds4_gpu_make_q8_0_mv_dispatch();
            if (out_dim > 65536u) mv_dispatch.nsg = 8;
            const bool force_output_nr4 =
                getenv("DS4_METAL_ENABLE_OUTPUT_Q8_NR4") != NULL;
            const bool output_shape =
                in_dim == 4096u && out_dim == 129280u;
            const bool use_output_nr4 =
                !g_quality_mode && (out_dim & 3u) == 0u &&
                (force_output_nr4 ||
                 (output_shape && ds4_gpu_device_name_contains("M3"))) &&
                getenv("DS4_METAL_DISABLE_M3_OUTPUT_Q8_NR4") == NULL;
            if (use_output_nr4) {
                mv_dispatch.function_name =
                    "kernel_mul_mv_q8_0_f32_nr4";
                mv_dispatch.nr0 = 4;
                mv_dispatch.smem = 32u * 4u * sizeof(float);
            }
            mv_args.nr0 = mv_dispatch.nr0;
            id<MTLComputePipelineState> pipeline =
                ds4_gpu_get_mul_mv_pipeline(mv_dispatch.function_name, mv_dispatch.nsg);
            if (!pipeline) return 0;

            id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
            [enc setComputePipelineState:pipeline];
            [enc setBytes:&mv_args length:sizeof(mv_args) atIndex:0];
            [enc setBuffer:wbuf offset:(NSUInteger)inner_offset atIndex:1];
            [enc setBuffer:xbuf offset:ds4_gpu_tensor_offset(x) atIndex:2];
            [enc setBuffer:outbuf offset:ds4_gpu_tensor_offset(out) atIndex:3];
            [enc setThreadgroupMemoryLength:mv_dispatch.smem atIndex:0];
            [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)out_dim + (NSUInteger)mv_dispatch.nr0 - 1u) / (NSUInteger)mv_dispatch.nr0,
                                                  1,
                                                  1)
                 threadsPerThreadgroup:MTLSizeMake(32, (NSUInteger)mv_dispatch.nsg, 1)];
            ds4_gpu_end_compute_encoder(cb, enc);

            if (!ds4_gpu_finish_command_buffer(cb, owned, "Q8_0 tensor matvec")) {
                return 0;
            }
            return 1;
        }

        const uint64_t mv_ext_max_tokens =
            ds4_gpu_env_u64("DS4_METAL_Q8_MV_EXT_MAX_TOKENS", 16u, 2u, 128u);
        if (n_tok <= mv_ext_max_tokens && (in_dim % 128u) == 0) {
            const int16_t nsg = 2;
            const int16_t nxpsg = ds4_gpu_mv_ext_nxpsg(in_dim, n_tok);
            const int16_t r1ptg = ds4_gpu_mv_ext_r1ptg(n_tok);
            const char *fn_name = ds4_gpu_mv_ext_name(1, r1ptg);
            id<MTLComputePipelineState> pipeline =
                fn_name ? ds4_gpu_get_mul_mv_ext_pipeline(fn_name, nsg, nxpsg) : nil;
            if (!pipeline) return 0;

            const int16_t nypsg = 32 / nxpsg;
            const uint64_t r0ptg = (uint64_t)nypsg * (uint64_t)nsg;
            ds4_gpu_mul_mv_ext_args args =
                ds4_gpu_make_mv_ext_args(in_dim, out_dim, n_tok, 34, row_bytes);

            id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
            [enc setComputePipelineState:pipeline];
            [enc setBytes:&args length:sizeof(args) atIndex:0];
            [enc setBuffer:wbuf offset:(NSUInteger)inner_offset atIndex:1];
            [enc setBuffer:xbuf offset:ds4_gpu_tensor_offset(x) atIndex:2];
            [enc setBuffer:outbuf offset:ds4_gpu_tensor_offset(out) atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)out_dim + (NSUInteger)r0ptg - 1u) / (NSUInteger)r0ptg,
                                                  ((NSUInteger)n_tok + (NSUInteger)r1ptg - 1u) / (NSUInteger)r1ptg,
                                                  1)
                 threadsPerThreadgroup:MTLSizeMake(32, (NSUInteger)nsg, 1)];
            ds4_gpu_end_compute_encoder(cb, enc);

            if (!ds4_gpu_finish_command_buffer(cb, owned, "Q8_0 tensor mul_mv_ext")) {
                return 0;
            }
            return 1;
        }

        /*
         * Dense Q8_0 prefill is the cleanest DS4 TensorOps shape: M/N/K are
         * aligned and the RHS activation matrix is already dense.  The retained
         * kernel dequantizes each 64x32 weight tile to half in threadgroup
         * memory, then uses direct-RHS MPP for the activation tile.  This avoids
         * staging RHS into threadgroup memory and was the direct replacement for
         * the slower generic MPP prototype.
         */
        if (ds4_gpu_mpp_available() &&
            n_tok >= 32u &&
            (in_dim % 64u) == 0 &&
            (out_dim % 64u) == 0 &&
            (n_tok % 32u) == 0) {
            uint64_t nax_tile_n = 32u;
            if ((n_tok % 128u) == 0) {
                nax_tile_n = 128u;
            } else if ((n_tok % 64u) == 0) {
                nax_tile_n = 64u;
            }
            const char *nax_fn = nax_tile_n == 128u
                ? "kernel_mul_mm_q8_0_f32_nax_direct_rhs_n128"
                : (nax_tile_n == 64u
                    ? "kernel_mul_mm_q8_0_f32_nax_direct_rhs_n64"
                    : "kernel_mul_mm_q8_0_f32_nax_direct_rhs");
            id<MTLComputePipelineState> pipeline =
                ds4_gpu_get_mul_mm_pipeline(nax_fn, false, false);
            if (pipeline) {
                ds4_gpu_mul_mm_args args = ds4_gpu_make_mm_args(in_dim, out_dim, n_tok, row_bytes);

                id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
                [enc setComputePipelineState:pipeline];
                [enc setBytes:&args length:sizeof(args) atIndex:0];
                [enc setBuffer:wbuf offset:(NSUInteger)inner_offset atIndex:1];
                [enc setBuffer:xbuf offset:ds4_gpu_tensor_offset(x) atIndex:2];
                [enc setBuffer:outbuf offset:ds4_gpu_tensor_offset(out) atIndex:3];
                [enc setThreadgroupMemoryLength:2u * 64u * 32u * sizeof(uint16_t) atIndex:0];
                [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)(n_tok / nax_tile_n),
                                                      (NSUInteger)out_dim / 64u,
                                                      1)
                     threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                ds4_gpu_end_compute_encoder(cb, enc);

                if (!ds4_gpu_finish_command_buffer(cb, owned, "Q8_0 NAX tensor matmul")) {
                    return 0;
                }
                return 1;
            }
            ds4_gpu_warn_mpp_fallback();
        }

        const bool bc_inp = (in_dim % 32u) != 0;
        const bool bc_out = (out_dim % 64u) != 0 || (n_tok % 32u) != 0;
        id<MTLComputePipelineState> pipeline =
            ds4_gpu_get_mul_mm_pipeline("kernel_mul_mm_q8_0_f32", bc_inp, bc_out);
        if (!pipeline) return 0;

        ds4_gpu_mul_mm_args args = ds4_gpu_make_mm_args(in_dim, out_dim, n_tok, row_bytes);

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:wbuf offset:(NSUInteger)inner_offset atIndex:1];
        [enc setBuffer:xbuf offset:ds4_gpu_tensor_offset(x) atIndex:2];
        [enc setBuffer:outbuf offset:ds4_gpu_tensor_offset(out) atIndex:3];
        [enc setThreadgroupMemoryLength:(bc_out ? 8192u : 6144u) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)n_tok + 31u) / 32u,
                                              ((NSUInteger)out_dim + 63u) / 64u,
                                              1)
             threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "Q8_0 tensor matmul")) {
            return 0;
        }
    }

    return 1;
}
