int ds4_gpu_store_raw_kv_tensor(
        ds4_gpu_tensor       *raw_cache,
        const ds4_gpu_tensor *kv,
        uint32_t                raw_cap,
        uint32_t                row,
        uint32_t                head_dim) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!raw_cache || !kv || raw_cap == 0 || row >= raw_cap || head_dim == 0 || raw_cap > INT32_MAX) return 0;

    @autoreleasepool {
        const uint64_t raw_bytes = (uint64_t)raw_cap * head_dim * sizeof(float);
        if (ds4_gpu_tensor_bytes(raw_cache) < raw_bytes) {
            fprintf(stderr, "ds4: Metal raw KV store received undersized destination buffer\n");
            return 0;
        }

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        const int32_t row_i32 = (int32_t)row;
        if (!ds4_gpu_encode_f16_round_copy_for_raw_store(cb, kv, head_dim) ||
            !ds4_gpu_encode_set_rows_f32_i32(cb, raw_cache,
                                               g_raw_store_round_buffer,
                                               0,
                                               &row_i32,
                                               1,
                                               raw_cap,
                                               head_dim)) {
            return 0;
        }

        if (!ds4_gpu_finish_command_buffer(cb, owned, "raw KV DS4 set_rows store")) return 0;
    }

    return 1;
}

/* Release decode fused KV finalizer.  Reference paths are selected by the C
 * graph driver; this Objective-C entry point always means "use the fused
 * Metal kernel." */
int ds4_gpu_kv_fp8_store_raw_tensor(
        ds4_gpu_tensor *kv,
        ds4_gpu_tensor *raw_cache,
        uint32_t          raw_cap,
        uint32_t          row,
        uint32_t          head_dim,
        uint32_t          n_rot) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!kv || !raw_cache || raw_cap == 0 || row >= raw_cap || head_dim == 0 ||
        n_rot > head_dim || raw_cap > INT32_MAX) {
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> kvbuf = ds4_gpu_tensor_buffer(kv);
        id<MTLBuffer> rawbuf = ds4_gpu_tensor_buffer(raw_cache);
        const uint64_t kv_bytes = (uint64_t)head_dim * sizeof(float);
        const uint64_t raw_bytes = (uint64_t)raw_cap * head_dim * sizeof(float);
        if (!kvbuf || !rawbuf ||
            ds4_gpu_tensor_bytes(kv) < kv_bytes ||
            ds4_gpu_tensor_bytes(raw_cache) < raw_bytes) {
            fprintf(stderr, "ds4: Metal fused KV FP8/raw-store received undersized buffers\n");
            return 0;
        }

        ds4_gpu_dsv4_kv_fp8_store_args args = {
            .head_dim = (int32_t)head_dim,
            .n_rot = (int32_t)n_rot,
            .raw_row = (int32_t)row,
        };

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:g_dsv4_kv_fp8_store_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:kvbuf offset:ds4_gpu_tensor_offset(kv) atIndex:1];
        [enc setBuffer:rawbuf offset:ds4_gpu_tensor_offset(raw_cache) atIndex:2];
        [enc setThreadgroupMemoryLength:64u * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "KV FP8/raw-store fused")) return 0;
    }

    return 1;
}

int ds4_gpu_store_raw_kv_batch_tensor(
        ds4_gpu_tensor       *raw_cache,
        const ds4_gpu_tensor *kv,
        uint32_t                raw_cap,
        uint32_t                pos0,
        uint32_t                n_tokens,
        uint32_t                head_dim) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!raw_cache || !kv || raw_cap == 0 || n_tokens == 0 || head_dim == 0 || raw_cap > INT32_MAX) return 0;

    @autoreleasepool {
        const uint64_t raw_bytes = (uint64_t)raw_cap * head_dim * sizeof(float);
        if (ds4_gpu_tensor_bytes(raw_cache) < raw_bytes) {
            fprintf(stderr, "ds4: Metal raw KV batch store received undersized destination buffer\n");
            return 0;
        }

        int32_t rows_stack[512];
        int32_t *rows = rows_stack;
        if (n_tokens > (uint32_t)(sizeof(rows_stack) / sizeof(rows_stack[0]))) {
            rows = malloc((size_t)n_tokens * sizeof(*rows));
            if (!rows) {
                fprintf(stderr, "ds4: failed to allocate raw KV set_rows index list\n");
                return 0;
            }
        }
        for (uint32_t t = 0; t < n_tokens; t++) {
            rows[t] = (int32_t)((pos0 + t) % raw_cap);
        }

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) {
            if (rows != rows_stack) free(rows);
            return 0;
        }

        const uint64_t n = (uint64_t)n_tokens * head_dim;
        const int ok = n <= UINT32_MAX &&
            ds4_gpu_encode_f16_round_copy_for_raw_store(cb, kv, (uint32_t)n) &&
            ds4_gpu_encode_set_rows_f32_i32(cb, raw_cache,
                                               g_raw_store_round_buffer,
                                               0,
                                               rows,
                                               n_tokens,
                                               raw_cap,
                                               head_dim);
        if (rows != rows_stack) free(rows);
        if (!ok) return 0;

        if (!ds4_gpu_finish_command_buffer(cb, owned, "raw KV batch DS4 set_rows store")) return 0;
    }

    return 1;
}

static int ds4_gpu_encode_compressor_score_with_ape(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        score_src,
        NSUInteger           score_src_offset,
        id<MTLBuffer>        score_dst,
        NSUInteger           score_dst_offset,
        id<MTLBuffer>        apebuf,
        NSUInteger           ape_offset,
        uint32_t             ape_type,
        uint32_t             width,
        uint32_t             ratio,
        uint32_t             pos0,
        uint32_t             n_tokens) {
    if (!cb || !score_src || !score_dst || !apebuf ||
        width == 0 || ratio == 0 || n_tokens == 0 ||
        (ape_type != 0u && ape_type != 1u)) {
        return 0;
    }

    const uint64_t total_elems64 = (uint64_t)n_tokens * width;
    if (total_elems64 > UINT32_MAX) {
        fprintf(stderr, "ds4: Metal compressor APE add received too many elements\n");
        return 0;
    }
    const uint32_t total_elems = (uint32_t)total_elems64;

    const bool force_fused =
        getenv("DS4_METAL_ENABLE_COMPRESSOR_APE_ADD") != NULL;
    const bool use_fused =
        (ds4_gpu_device_name_contains("M3") || force_fused) &&
        getenv("DS4_METAL_DISABLE_M3_COMPRESSOR_APE_ADD") == NULL;
    if (use_fused) {
        id<MTLComputePipelineState> pipeline = ds4_gpu_get_pipeline(
            ape_type == 1u ? "kernel_dsv4_compressor_score_ape_f16"
                           : "kernel_dsv4_compressor_score_ape_f32");
        if (pipeline) {
            ds4_gpu_dsv4_compressor_score_ape_args args = {
                .width = width,
                .ratio = ratio,
                .pos0 = pos0,
                .n_tokens = n_tokens,
            };
            NSUInteger nth = pipeline.maxTotalThreadsPerThreadgroup;
            if (nth > 256u) nth = 256u;
            if (nth == 0) return 0;

            id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
            [enc setComputePipelineState:pipeline];
            [enc setBytes:&args length:sizeof(args) atIndex:0];
            [enc setBuffer:score_src offset:score_src_offset atIndex:1];
            [enc setBuffer:apebuf offset:ape_offset atIndex:2];
            [enc setBuffer:score_dst offset:score_dst_offset atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake(
                    ((NSUInteger)total_elems + nth - 1u) / nth, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
            ds4_gpu_end_compute_encoder(cb, enc);
            return 1;
        }
        if (force_fused) return 0;
    }

    const NSUInteger scratch_bytes = (NSUInteger)total_elems * sizeof(float);
    if (!ds4_gpu_ensure_scratch_buffer(&g_compressor_store_ape_buffer,
                                         &g_compressor_store_ape_bytes,
                                         scratch_bytes,
                                         "ds4_compressor_store_ape")) {
        return 0;
    }

    const uint64_t elem_ape = ape_type == 1u ? 2u : 4u;
    uint32_t copied_rows = 0;
    uint32_t pos_mod = pos0 % ratio;
    while (copied_rows < n_tokens) {
        uint32_t seg_rows = ratio - pos_mod;
        if (seg_rows > n_tokens - copied_rows) seg_rows = n_tokens - copied_rows;
        const uint32_t seg_elems = seg_rows * width;
        const NSUInteger src_off = ape_offset + (NSUInteger)pos_mod * width * elem_ape;
        const NSUInteger dst_off = (NSUInteger)copied_rows * width * sizeof(float);
        int ok;
        if (ape_type == 1u) {
            ok = ds4_gpu_encode_cpy_f16_f32_1d(cb,
                                                 apebuf,
                                                 src_off,
                                                 g_compressor_store_ape_buffer,
                                                 dst_off,
                                                 seg_elems);
        } else {
            ok = ds4_gpu_encode_cpy_f32_f32_1d(cb,
                                                 apebuf,
                                                 src_off,
                                                 g_compressor_store_ape_buffer,
                                                 dst_off,
                                                 seg_elems);
        }
        if (!ok) return 0;
        copied_rows += seg_rows;
        pos_mod = 0;
    }

    return ds4_gpu_encode_add_f32_1d(cb,
                                       score_src,
                                       score_src_offset,
                                       g_compressor_store_ape_buffer,
                                       0,
                                       score_dst,
                                       score_dst_offset,
                                       total_elems);
}

static int ds4_gpu_encode_compressor_set_rows_projected(
        id<MTLCommandBuffer> cb,
        ds4_gpu_tensor    *state_kv,
        ds4_gpu_tensor    *state_score,
        id<MTLBuffer>        kvbuf,
        NSUInteger           kv_offset,
        id<MTLBuffer>        scorebuf,
        NSUInteger           score_offset,
        id<MTLBuffer>        apebuf,
        NSUInteger           ape_offset,
        uint32_t             ape_type,
        uint32_t             width,
        uint32_t             ratio,
        uint32_t             pos0,
        const int32_t       *rows,
        uint32_t             n_rows,
        uint32_t             state_rows) {
    if (!cb || !state_kv || !state_score || !kvbuf || !scorebuf ||
        !apebuf || !rows || width == 0 || n_rows == 0 || state_rows == 0) {
        return 0;
    }

    const NSUInteger score_scratch_bytes = (NSUInteger)n_rows * width * sizeof(float);
    if (!ds4_gpu_ensure_scratch_buffer(&g_compressor_store_score_buffer,
                                         &g_compressor_store_score_bytes,
                                         score_scratch_bytes,
                                         "ds4_compressor_store_score")) {
        return 0;
    }

    return ds4_gpu_encode_compressor_score_with_ape(cb,
                                                      scorebuf,
                                                      score_offset,
                                                      g_compressor_store_score_buffer,
                                                      0,
                                                      apebuf,
                                                      ape_offset,
                                                      ape_type,
                                                      width,
                                                      ratio,
                                                      pos0,
                                                      n_rows) &&
           ds4_gpu_encode_set_rows_f32_i32(cb,
                                             state_kv,
                                             kvbuf,
                                             kv_offset,
                                             rows,
                                             n_rows,
                                             state_rows,
                                             width) &&
           ds4_gpu_encode_set_rows_f32_i32(cb,
                                             state_score,
                                             g_compressor_store_score_buffer,
                                             0,
                                             rows,
                                             n_rows,
                                             state_rows,
                                             width);
}

static int ds4_gpu_compressor_store_one_tensor(
        const ds4_gpu_tensor *kv,
        const ds4_gpu_tensor *sc,
        ds4_gpu_tensor       *state_kv,
        ds4_gpu_tensor       *state_score,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint32_t                width,
        uint32_t                ratio,
        uint32_t                pos) {
    if (!kv || !sc || !state_kv || !state_score || !model_map ||
        width == 0 || ratio == 0 || (ape_type != 0u && ape_type != 1u)) {
        return 0;
    }

    id<MTLComputePipelineState> pipeline =
        ds4_gpu_hot_pipeline(g_dsv4_compressor_store_one_pipeline,
                                "kernel_dsv4_compressor_store_one");
    if (!pipeline) return 0;

    const uint32_t state_rows = ratio == 4u ? 2u * ratio : ratio;
    const uint64_t elem_ape = ape_type == 1u ? 2u : 4u;
    const uint64_t row_bytes = (uint64_t)width * sizeof(float);
    const uint64_t state_bytes = (uint64_t)state_rows * row_bytes;
    const uint64_t ape_bytes = (uint64_t)width * ratio * elem_ape;
    if (ape_offset > model_size || ape_bytes > model_size - ape_offset ||
        ds4_gpu_tensor_bytes(kv) < row_bytes ||
        ds4_gpu_tensor_bytes(sc) < row_bytes ||
        ds4_gpu_tensor_bytes(state_kv) < state_bytes ||
        ds4_gpu_tensor_bytes(state_score) < state_bytes) {
        return 0;
    }

    uint64_t ape_inner = 0;
    id<MTLBuffer> apebuf = ds4_gpu_wrap_model_range(model_map, model_size,
                                                       ape_offset, ape_bytes,
                                                       &ape_inner);
    id<MTLBuffer> kvbuf = ds4_gpu_tensor_buffer(kv);
    id<MTLBuffer> scbuf = ds4_gpu_tensor_buffer(sc);
    id<MTLBuffer> statekvbuf = ds4_gpu_tensor_buffer(state_kv);
    id<MTLBuffer> statescbuf = ds4_gpu_tensor_buffer(state_score);
    if (!apebuf || !kvbuf || !scbuf || !statekvbuf || !statescbuf) return 0;

    ds4_gpu_dsv4_compressor_store_one_args args = {
        .width = width,
        .ratio = ratio,
        .pos = pos,
        .ape_type = ape_type,
    };

    int owned = 0;
    id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
    if (!cb) return 0;

    const NSUInteger nth = 256u;
    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:kvbuf offset:ds4_gpu_tensor_offset(kv) atIndex:1];
    [enc setBuffer:scbuf offset:ds4_gpu_tensor_offset(sc) atIndex:2];
    [enc setBuffer:apebuf offset:(NSUInteger)ape_inner atIndex:3];
    [enc setBuffer:statekvbuf offset:ds4_gpu_tensor_offset(state_kv) atIndex:4];
    [enc setBuffer:statescbuf offset:ds4_gpu_tensor_offset(state_score) atIndex:5];
    [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)width + nth - 1u) / nth, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);

    return ds4_gpu_finish_command_buffer(cb, owned, "compressor one-row store");
}

int ds4_gpu_compressor_store_batch_tensor(
        const ds4_gpu_tensor *kv,
        const ds4_gpu_tensor *sc,
        ds4_gpu_tensor       *state_kv,
        ds4_gpu_tensor       *state_score,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint32_t                head_dim,
        uint32_t                ratio,
        uint32_t                pos0,
        uint32_t                n_tokens) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!kv || !sc || !state_kv || !state_score || !model_map ||
        head_dim == 0 || ratio == 0 || n_tokens == 0 ||
        (ape_type != 0u && ape_type != 1u)) {
        return 0;
    }

    @autoreleasepool {
        const uint32_t coff = ratio == 4u ? 2u : 1u;
        const uint32_t width = coff * head_dim;
        const uint32_t state_rows = coff * ratio;
        const uint64_t elem_ape = ape_type == 1u ? 2u : 4u;
        const uint64_t kv_bytes = (uint64_t)n_tokens * width * sizeof(float);
        const uint64_t state_bytes = (uint64_t)state_rows * width * sizeof(float);
        const uint64_t ape_bytes = (uint64_t)width * ratio * elem_ape;

        if (ape_offset > model_size || ape_bytes > model_size - ape_offset) {
            fprintf(stderr, "ds4: Metal compressor batch APE range is outside the mapped model\n");
            return 0;
        }

        id<MTLBuffer> kvbuf = ds4_gpu_tensor_buffer(kv);
        id<MTLBuffer> scbuf = ds4_gpu_tensor_buffer(sc);
        if (!kvbuf || !scbuf ||
            ds4_gpu_tensor_bytes(kv) < kv_bytes ||
            ds4_gpu_tensor_bytes(sc) < kv_bytes ||
            ds4_gpu_tensor_bytes(state_kv) < state_bytes ||
            ds4_gpu_tensor_bytes(state_score) < state_bytes) {
            fprintf(stderr, "ds4: Metal compressor batch store received undersized buffers\n");
            return 0;
        }

        uint64_t ape_inner = 0;
        id<MTLBuffer> apebuf = ds4_gpu_wrap_model_range(model_map, model_size, ape_offset, ape_bytes, &ape_inner);
        if (!apebuf) return 0;

        const uint64_t total_elems64 = (uint64_t)n_tokens * width;
        if (total_elems64 > UINT32_MAX || state_rows > INT32_MAX) {
            fprintf(stderr, "ds4: Metal compressor batch store received too many elements\n");
            return 0;
        }
        const uint32_t total_elems = (uint32_t)total_elems64;
        const NSUInteger scratch_bytes = (NSUInteger)total_elems * sizeof(float);
        if (!ds4_gpu_ensure_scratch_buffer(&g_compressor_store_ape_buffer,
                                             &g_compressor_store_ape_bytes,
                                             scratch_bytes,
                                             "ds4_compressor_store_ape") ||
            !ds4_gpu_ensure_scratch_buffer(&g_compressor_store_score_buffer,
                                             &g_compressor_store_score_bytes,
                                             scratch_bytes,
                                             "ds4_compressor_store_score")) {
            return 0;
        }

        int32_t rows_stack[16];
        int32_t *rows = rows_stack;
        if (n_tokens > (uint32_t)(sizeof(rows_stack) / sizeof(rows_stack[0]))) {
            rows = malloc((size_t)n_tokens * sizeof(*rows));
            if (!rows) {
                fprintf(stderr, "ds4: failed to allocate compressor set_rows index list\n");
                return 0;
            }
        }
        for (uint32_t t = 0; t < n_tokens; t++) {
            const uint32_t pos_mod = (pos0 + t) % ratio;
            rows[t] = (int32_t)(ratio == 4u ? ratio + pos_mod : pos_mod);
        }

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) {
            if (rows != rows_stack) free(rows);
            return 0;
        }

        int ok = 1;
        uint32_t copied_rows = 0;
        uint32_t pos_mod = pos0 % ratio;
        while (ok && copied_rows < n_tokens) {
            uint32_t seg_rows = ratio - pos_mod;
            if (seg_rows > n_tokens - copied_rows) seg_rows = n_tokens - copied_rows;
            const uint32_t seg_elems = seg_rows * width;
            const NSUInteger src_off = (NSUInteger)ape_inner +
                                       (NSUInteger)pos_mod * width * elem_ape;
            const NSUInteger dst_off = (NSUInteger)copied_rows * width * sizeof(float);
            if (ape_type == 1u) {
                ok = ds4_gpu_encode_cpy_f16_f32_1d(cb,
                                                     apebuf,
                                                     src_off,
                                                     g_compressor_store_ape_buffer,
                                                     dst_off,
                                                     seg_elems);
            } else {
                ok = ds4_gpu_encode_cpy_f32_f32_1d(cb,
                                                     apebuf,
                                                     src_off,
                                                     g_compressor_store_ape_buffer,
                                                     dst_off,
                                                     seg_elems);
            }
            copied_rows += seg_rows;
            pos_mod = 0;
        }

        if (ok) {
            ok = ds4_gpu_encode_add_f32_1d(cb,
                                             scbuf,
                                             ds4_gpu_tensor_offset(sc),
                                             g_compressor_store_ape_buffer,
                                             0,
                                             g_compressor_store_score_buffer,
                                             0,
                                             total_elems);
        }
        if (ok) {
            ok = ds4_gpu_encode_set_rows_f32_i32(cb,
                                                   state_kv,
                                                   kvbuf,
                                                   ds4_gpu_tensor_offset(kv),
                                                   rows,
                                                   n_tokens,
                                                   state_rows,
                                                   width);
        }
        if (ok) {
            ok = ds4_gpu_encode_set_rows_f32_i32(cb,
                                                   state_score,
                                                   g_compressor_store_score_buffer,
                                                   0,
                                                   rows,
                                                   n_tokens,
                                                   state_rows,
                                                   width);
        }
        if (rows != rows_stack) free(rows);
        if (!ok) return 0;

        if (!ds4_gpu_finish_command_buffer(cb, owned, "compressor batch DS4 store")) return 0;
    }

    return 1;
}

static ds4_gpu_bin_args ds4_gpu_make_bin_contiguous_3d_args(
        uint32_t cols,
        uint32_t rows,
        uint32_t planes) {
    const uint64_t row_bytes = (uint64_t)cols * sizeof(float);
    const uint64_t plane_bytes = (uint64_t)rows * row_bytes;
    return (ds4_gpu_bin_args) {
        .ne00 = (int32_t)cols,
        .ne01 = (int32_t)rows,
        .ne02 = (int32_t)planes,
        .ne03 = 1,
        .nb00 = sizeof(float),
        .nb01 = row_bytes,
        .nb02 = plane_bytes,
        .nb03 = (uint64_t)planes * plane_bytes,
        .ne10 = (int32_t)cols,
        .ne11 = (int32_t)rows,
        .ne12 = (int32_t)planes,
        .ne13 = 1,
        .nb10 = sizeof(float),
        .nb11 = row_bytes,
        .nb12 = plane_bytes,
        .nb13 = (uint64_t)planes * plane_bytes,
        .ne0 = (int32_t)cols,
        .ne1 = (int32_t)rows,
        .ne2 = (int32_t)planes,
        .ne3 = 1,
        .nb0 = sizeof(float),
        .nb1 = row_bytes,
        .nb2 = plane_bytes,
        .nb3 = (uint64_t)planes * plane_bytes,
        .offs = 0,
        .o1 = { 0 },
    };
}

static int ds4_gpu_encode_softmax_f32_contiguous(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        src,
        NSUInteger           src_off,
        id<MTLBuffer>        dst,
        NSUInteger           dst_off,
        uint32_t             width,
        uint32_t             rows,
        uint32_t             planes) {
    if (!cb || !src || !dst || width == 0 || rows == 0 || planes == 0) return 0;

    const uint64_t row_bytes = (uint64_t)width * sizeof(float);
    const uint64_t plane_bytes = (uint64_t)rows * row_bytes;
    ds4_gpu_softmax_args args = {
        .ne00 = (int32_t)width,
        .ne01 = (int32_t)rows,
        .ne02 = (int32_t)planes,
        .nb01 = row_bytes,
        .nb02 = plane_bytes,
        .nb03 = (uint64_t)planes * plane_bytes,
        .ne11 = (int32_t)width,
        .ne12 = (int32_t)rows,
        .ne13 = (int32_t)planes,
        .nb11 = row_bytes,
        .nb12 = plane_bytes,
        .nb13 = (uint64_t)planes * plane_bytes,
        .nb1 = row_bytes,
        .nb2 = plane_bytes,
        .nb3 = (uint64_t)planes * plane_bytes,
        .scale = 1.0f,
        .max_bias = 0.0f,
        .m0 = 0.0f,
        .m1 = 0.0f,
        .n_head_log2 = 1,
    };

    id<MTLComputePipelineState> pipeline =
        (width % 4u) == 0 ? g_soft_max_f32_4_pipeline : g_soft_max_f32_pipeline;
    if (!pipeline) return 0;

    NSUInteger nth = 32u;
    if ((width % 4u) == 0) {
        while (nth < (NSUInteger)(width / 4u) &&
               nth * (NSUInteger)rows * (NSUInteger)planes < 256u) {
            nth *= 2u;
        }
    } else {
        while (nth < (NSUInteger)width &&
               nth * (NSUInteger)rows * (NSUInteger)planes < 256u) {
            nth *= 2u;
        }
    }
    const NSUInteger max_threads = pipeline.maxTotalThreadsPerThreadgroup;
    if (nth > max_threads) nth = max_threads;
    if (nth == 0) nth = 1u;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:src offset:src_off atIndex:1];
    [enc setBuffer:src offset:src_off atIndex:2];
    [enc setBuffer:src offset:src_off atIndex:3];
    [enc setBuffer:dst offset:dst_off atIndex:4];
    [enc setThreadgroupMemoryLength:32u * sizeof(float) atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake(rows, planes, 1)
         threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

static int ds4_gpu_encode_dsv4_softmax_pool_one_comp_ggml(
        id<MTLCommandBuffer> cb,
        ds4_gpu_tensor    *out,
        id<MTLBuffer>        kvbuf,
        NSUInteger           kv_offset,
        uint64_t             kv_nb0,
        uint64_t             kv_nb1,
        uint64_t             kv_nb2,
        id<MTLBuffer>        scorebuf,
        NSUInteger           score_offset,
        uint64_t             score_nb0,
        uint64_t             score_nb1,
        uint64_t             score_nb2,
        uint32_t             n_rows,
        uint32_t             head_dim) {
    id<MTLBuffer> outbuf = ds4_gpu_tensor_buffer(out);
    if (!cb || !outbuf || !kvbuf || !scorebuf || n_rows == 0 || head_dim == 0 ||
        ds4_gpu_tensor_bytes(out) < (uint64_t)head_dim * sizeof(float)) {
        return 0;
    }

    const NSUInteger pack_bytes = (NSUInteger)n_rows * head_dim * sizeof(float);
    if (!ds4_gpu_ensure_scratch_buffer(&g_compressor_pool_product_buffer,
                                         &g_compressor_pool_product_bytes,
                                         pack_bytes,
                                         "ds4_compressor_pool_product") ||
        !ds4_gpu_ensure_scratch_buffer(&g_compressor_pool_score_cont_buffer,
                                         &g_compressor_pool_score_cont_bytes,
                                         pack_bytes,
                                         "ds4_compressor_pool_score_cont") ||
        !ds4_gpu_ensure_scratch_buffer(&g_compressor_pool_softmax_buffer,
                                         &g_compressor_pool_softmax_bytes,
                                         pack_bytes,
                                         "ds4_compressor_pool_softmax")) {
        return 0;
    }

    const uint64_t cont_row_stride = (uint64_t)n_rows * sizeof(float);
    const uint64_t cont_plane_stride = (uint64_t)head_dim * cont_row_stride;

    /*
     * Keep the n_comp == 1 compressor path as the unfused graph sequence:
     *
     *   score = soft_max(contiguous(score))
     *   pooled = sum_rows(contiguous(kv) * score)
     *
     * The fused DS4 pool kernel is mathematically equivalent, but it reduces in
     * a different order. That is enough to create ~1e-6 compressor differences
     * and later FP8/routing flips, so this path intentionally keeps the same
     * operation boundary and memory layout as the graph.
     */
    ds4_gpu_bin_args mul_args =
        ds4_gpu_make_bin_contiguous_3d_args(n_rows, head_dim, 1);

    return
        ds4_gpu_encode_cpy_f32_f32_3d_src_strided(cb,
                                                    kvbuf,
                                                    kv_offset,
                                                    g_compressor_pool_product_buffer,
                                                    0,
                                                    n_rows,
                                                    head_dim,
                                                    1,
                                                    kv_nb0,
                                                    kv_nb1,
                                                    kv_nb2,
                                                    cont_row_stride,
                                                    cont_plane_stride) &&
        ds4_gpu_encode_cpy_f32_f32_3d_src_strided(cb,
                                                    scorebuf,
                                                    score_offset,
                                                    g_compressor_pool_score_cont_buffer,
                                                    0,
                                                    n_rows,
                                                    head_dim,
                                                    1,
                                                    score_nb0,
                                                    score_nb1,
                                                    score_nb2,
                                                    cont_row_stride,
                                                    cont_plane_stride) &&
        ds4_gpu_encode_softmax_f32_contiguous(cb,
                                                g_compressor_pool_score_cont_buffer,
                                                0,
                                                g_compressor_pool_softmax_buffer,
                                                0,
                                                n_rows,
                                                head_dim,
                                                1) &&
        ds4_gpu_encode_bin_f32_rows(cb,
                                      g_mul_pipeline,
                                      &mul_args,
                                      g_compressor_pool_product_buffer,
                                      0,
                                      g_compressor_pool_softmax_buffer,
                                      0,
                                      g_compressor_pool_product_buffer,
                                      0) &&
        ds4_gpu_encode_sum_rows_f32(cb,
                                      g_compressor_pool_product_buffer,
                                      0,
                                      outbuf,
                                      ds4_gpu_tensor_offset(out),
                                      n_rows,
                                      head_dim);
}

static int ds4_gpu_encode_dsv4_softmax_pool(
        id<MTLCommandBuffer> cb,
        ds4_gpu_tensor    *out,
        id<MTLBuffer>        kvbuf,
        NSUInteger           kv_offset,
        uint64_t             kv_nb0,
        uint64_t             kv_nb1,
        uint64_t             kv_nb2,
        id<MTLBuffer>        scorebuf,
        NSUInteger           score_offset,
        uint64_t             score_nb0,
        uint64_t             score_nb1,
        uint64_t             score_nb2,
        uint32_t             n_rows,
        uint32_t             head_dim,
        uint32_t             n_comp) {
    id<MTLBuffer> outbuf = ds4_gpu_tensor_buffer(out);
    if (!cb || !outbuf || !kvbuf || !scorebuf ||
        n_rows == 0 || head_dim == 0 || n_comp == 0 ||
        ds4_gpu_tensor_bytes(out) < (uint64_t)head_dim * n_comp * sizeof(float)) {
        return 0;
    }

    if (n_comp == 1) {
        return ds4_gpu_encode_dsv4_softmax_pool_one_comp_ggml(cb,
                                                                out,
                                                                kvbuf,
                                                                kv_offset,
                                                                kv_nb0,
                                                                kv_nb1,
                                                                kv_nb2,
                                                                scorebuf,
                                                                score_offset,
                                                                score_nb0,
                                                                score_nb1,
                                                                score_nb2,
                                                                n_rows,
                                                                head_dim);
    }

    ds4_gpu_dsv4_softmax_pool_args args = {
        .ne00 = (int64_t)n_rows,
        .ne01 = (int64_t)head_dim,
        .ne02 = (int64_t)n_comp,
        .nb00 = kv_nb0,
        .nb01 = kv_nb1,
        .nb02 = kv_nb2,
        .nb10 = score_nb0,
        .nb11 = score_nb1,
        .nb12 = score_nb2,
        .ne0 = (int64_t)head_dim,
        .ne1 = (int64_t)n_comp,
        .nb0 = sizeof(float),
        .nb1 = (uint64_t)head_dim * sizeof(float),
    };
    const uint64_t n = (uint64_t)head_dim * n_comp;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:g_dsv4_softmax_pool_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:kvbuf offset:kv_offset atIndex:1];
    [enc setBuffer:scorebuf offset:score_offset atIndex:2];
    [enc setBuffer:outbuf offset:ds4_gpu_tensor_offset(out) atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)n + 255u) / 256u, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

static int ds4_gpu_encode_concat_f32_dim1(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        src0,
        NSUInteger           src0_offset,
        uint32_t             src0_rows,
        uint64_t             src0_row_stride,
        id<MTLBuffer>        src1,
        NSUInteger           src1_offset,
        uint32_t             src1_rows,
        uint64_t             src1_row_stride,
        id<MTLBuffer>        dst,
        NSUInteger           dst_offset,
        uint32_t             cols,
        uint64_t             dst_row_stride) {
    if (!cb || !src0 || !src1 || !dst || cols == 0 || src0_rows == 0 || src1_rows == 0) {
        return 0;
    }

    const uint32_t rows = src0_rows + src1_rows;
    const uint64_t src0_plane = (uint64_t)src0_rows * src0_row_stride;
    const uint64_t src1_plane = (uint64_t)src1_rows * src1_row_stride;
    const uint64_t dst_plane = (uint64_t)rows * dst_row_stride;
    ds4_gpu_concat_args args = {
        .ne00 = (int32_t)cols,
        .ne01 = (int32_t)src0_rows,
        .ne02 = 1,
        .ne03 = 1,
        .nb00 = sizeof(float),
        .nb01 = src0_row_stride,
        .nb02 = src0_plane,
        .nb03 = src0_plane,
        .ne10 = (int32_t)cols,
        .ne11 = (int32_t)src1_rows,
        .ne12 = 1,
        .ne13 = 1,
        .nb10 = sizeof(float),
        .nb11 = src1_row_stride,
        .nb12 = src1_plane,
        .nb13 = src1_plane,
        .ne0 = (int32_t)cols,
        .ne1 = (int32_t)rows,
        .ne2 = 1,
        .ne3 = 1,
        .nb0 = sizeof(float),
        .nb1 = dst_row_stride,
        .nb2 = dst_plane,
        .nb3 = dst_plane,
        .dim = 1,
    };

    NSUInteger nth = cols < 1024u ? (NSUInteger)cols : 1024u;
    const NSUInteger max_threads = g_concat_pipeline.maxTotalThreadsPerThreadgroup;
    if (nth > max_threads) nth = max_threads;
    if (nth == 0) nth = 1;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:g_concat_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:src0 offset:src0_offset atIndex:1];
    [enc setBuffer:src1 offset:src1_offset atIndex:2];
    [enc setBuffer:dst offset:dst_offset atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

static int ds4_gpu_compressor_pack_ratio4_fusion_mode(uint32_t head_dim) {
    const bool force =
        getenv("DS4_METAL_ENABLE_COMPRESSOR_RATIO4_PACK_FUSION") != NULL;
    const bool default_shape = head_dim == 128u || head_dim == 512u;
    if (getenv("DS4_METAL_DISABLE_M3_COMPRESSOR_RATIO4_PACK_FUSION") != NULL ||
        (!(ds4_gpu_device_name_contains("M3") && default_shape) && !force)) {
        return 0;
    }
    if (g_dsv4_compressor_pack_ratio4_pipeline == nil) {
        return force ? -1 : 0;
    }
    return 1;
}

static bool ds4_gpu_buffer_ranges_overlap(
        id<MTLBuffer> a,
        NSUInteger    a_offset,
        uint64_t      a_bytes,
        id<MTLBuffer> b,
        NSUInteger    b_offset,
        uint64_t      b_bytes) {
    if (a != b || a_bytes == 0u || b_bytes == 0u) return false;
    if (a_offset <= b_offset) {
        return a_bytes > (uint64_t)(b_offset - a_offset);
    }
    return b_bytes > (uint64_t)(a_offset - b_offset);
}

static int ds4_gpu_compressor_ratio4_direct_pool_mode(
        uint32_t head_dim,
        uint32_t n_comp) {
    // One compressed row intentionally uses the legacy GGML reduction graph.
    if (n_comp <= 1u) return 0;

    const bool force =
        getenv("DS4_METAL_ENABLE_COMPRESSOR_RATIO4_DIRECT_POOL") != NULL;
    const bool default_shape = head_dim == 128u || head_dim == 512u;
    if (getenv("DS4_METAL_DISABLE_M3_COMPRESSOR_RATIO4_DIRECT_POOL") != NULL ||
        (!(ds4_gpu_device_name_contains("M3") && default_shape) && !force)) {
        return 0;
    }
    if (g_dsv4_softmax_pool_ratio4_direct_pipeline == nil) {
        return force ? -1 : 0;
    }
    return 1;
}

static int ds4_gpu_encode_compressor_ratio4_direct_pool(
        id<MTLCommandBuffer> cb,
        ds4_gpu_tensor    *out,
        id<MTLBuffer>        kvbuf,
        NSUInteger           kv_offset,
        id<MTLBuffer>        scorebuf,
        NSUInteger           score_offset,
        id<MTLBuffer>        statekvbuf,
        NSUInteger           state_kv_offset,
        id<MTLBuffer>        statescbuf,
        NSUInteger           state_score_offset,
        uint32_t             head_dim,
        uint32_t             n_comp,
        bool                 replay) {
    id<MTLBuffer> outbuf = ds4_gpu_tensor_buffer(out);
    const uint64_t n = (uint64_t)head_dim * n_comp;
    if (!cb || !outbuf || !kvbuf || !scorebuf || !statekvbuf || !statescbuf ||
        head_dim == 0u || n_comp <= 1u || n > UINT32_MAX ||
        ds4_gpu_tensor_bytes(out) < n * sizeof(float)) {
        return 0;
    }

    id<MTLComputePipelineState> pipeline = ds4_gpu_hot_pipeline(
        g_dsv4_softmax_pool_ratio4_direct_pipeline,
        "kernel_dsv4_softmax_pool_ratio4_direct");
    if (!pipeline) return 0;

    ds4_gpu_dsv4_softmax_pool_ratio4_direct_args args = {
        .n_rows = 8,
        .head_dim = head_dim,
        .n_comp = n_comp,
        .replay = replay ? 1u : 0u,
        .pad = 0u,
    };

    NSUInteger nth = 256u;
    if (nth > pipeline.maxTotalThreadsPerThreadgroup) {
        nth = pipeline.maxTotalThreadsPerThreadgroup;
    }
    if (nth == 0u) return 0;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:kvbuf offset:kv_offset atIndex:1];
    [enc setBuffer:scorebuf offset:score_offset atIndex:2];
    [enc setBuffer:statekvbuf offset:state_kv_offset atIndex:3];
    [enc setBuffer:statescbuf offset:state_score_offset atIndex:4];
    [enc setBuffer:outbuf offset:ds4_gpu_tensor_offset(out) atIndex:5];
    [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)n + nth - 1u) / nth, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

static int ds4_gpu_encode_compressor_pack_ratio4(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        kvbuf,
        NSUInteger           kv_offset,
        id<MTLBuffer>        scorebuf,
        NSUInteger           score_offset,
        id<MTLBuffer>        statekvbuf,
        NSUInteger           state_kv_offset,
        id<MTLBuffer>        statescbuf,
        NSUInteger           state_score_offset,
        uint32_t             head_dim,
        uint32_t             n_comp,
        bool                 replay) {
    if (!cb || !kvbuf || !scorebuf || !statekvbuf || !statescbuf ||
        !g_compressor_pool_kv_buffer || !g_compressor_pool_score_buffer ||
        head_dim == 0 || n_comp == 0 || head_dim > UINT32_MAX / 2u) {
        return 0;
    }

    const uint64_t total_elems64 = (uint64_t)n_comp * 8u * head_dim;
    if (total_elems64 > UINT32_MAX || n_comp > UINT32_MAX / 8u) return 0;
    id<MTLComputePipelineState> pipeline = ds4_gpu_hot_pipeline(
        g_dsv4_compressor_pack_ratio4_pipeline,
        "kernel_dsv4_compressor_pack_ratio4");
    if (!pipeline) return 0;

    NSUInteger nth = head_dim;
    if (nth > 256u) nth = 256u;
    if (nth > pipeline.maxTotalThreadsPerThreadgroup) {
        nth = pipeline.maxTotalThreadsPerThreadgroup;
    }
    if (nth == 0) return 0;

    ds4_gpu_dsv4_compressor_pack_ratio4_args args = {
        .head_dim = head_dim,
        .n_comp = n_comp,
        .replay = replay ? 1u : 0u,
        .n_threads = (uint32_t)nth,
    };

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:kvbuf offset:kv_offset atIndex:1];
    [enc setBuffer:scorebuf offset:score_offset atIndex:2];
    [enc setBuffer:statekvbuf offset:state_kv_offset atIndex:3];
    [enc setBuffer:statescbuf offset:state_score_offset atIndex:4];
    [enc setBuffer:g_compressor_pool_kv_buffer offset:0 atIndex:5];
    [enc setBuffer:g_compressor_pool_score_buffer offset:0 atIndex:6];
    [enc dispatchThreadgroups:MTLSizeMake(n_comp, 8u, 1)
         threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

static int ds4_gpu_encode_compressor_pool(
        id<MTLCommandBuffer>   cb,
        ds4_gpu_tensor      *out,
        const ds4_gpu_tensor *state_kv,
        const ds4_gpu_tensor *state_score,
        uint32_t               head_dim,
        uint32_t               ratio) {
    id<MTLBuffer> statekvbuf = ds4_gpu_tensor_buffer(state_kv);
    id<MTLBuffer> statescbuf = ds4_gpu_tensor_buffer(state_score);
    if (!cb || !out || !statekvbuf || !statescbuf || head_dim == 0 || ratio == 0) return 0;

    const uint32_t coff = ratio == 4u ? 2u : 1u;
    const uint32_t width = coff * head_dim;
    const uint32_t rows = coff * ratio;
    const uint64_t state_bytes = (uint64_t)width * rows * sizeof(float);
    if (ds4_gpu_tensor_bytes(state_kv) < state_bytes ||
        ds4_gpu_tensor_bytes(state_score) < state_bytes) {
        return 0;
    }

    if (ratio != 4u) {
        const uint64_t row_stride = (uint64_t)width * sizeof(float);
        return ds4_gpu_encode_dsv4_softmax_pool(cb,
                                                  out,
                                                  statekvbuf,
                                                  ds4_gpu_tensor_offset(state_kv),
                                                  row_stride,
                                                  sizeof(float),
                                                  (uint64_t)rows * row_stride,
                                                  statescbuf,
                                                  ds4_gpu_tensor_offset(state_score),
                                                  row_stride,
                                                  sizeof(float),
                                                  (uint64_t)rows * row_stride,
                                                  ratio,
                                                  head_dim,
                                                  1);
    }

    const NSUInteger packed_bytes = (NSUInteger)8u * head_dim * sizeof(float);
    if (!ds4_gpu_ensure_scratch_buffer(&g_compressor_pool_kv_buffer,
                                         &g_compressor_pool_kv_bytes,
                                         packed_bytes,
                                         "ds4_compressor_pool_kv") ||
        !ds4_gpu_ensure_scratch_buffer(&g_compressor_pool_score_buffer,
                                         &g_compressor_pool_score_bytes,
                                         packed_bytes,
                                         "ds4_compressor_pool_score")) {
        return 0;
    }

    const uint64_t state_row_stride = (uint64_t)width * sizeof(float);
    const uint64_t pool_row_stride = (uint64_t)head_dim * sizeof(float);
    const NSUInteger curr_offset = (NSUInteger)4u * state_row_stride +
                                   (NSUInteger)head_dim * sizeof(float);
    if (!ds4_gpu_encode_concat_f32_dim1(cb,
                                          statekvbuf,
                                          ds4_gpu_tensor_offset(state_kv),
                                          4,
                                          state_row_stride,
                                          statekvbuf,
                                          ds4_gpu_tensor_offset(state_kv) + curr_offset,
                                          4,
                                          state_row_stride,
                                          g_compressor_pool_kv_buffer,
                                          0,
                                          head_dim,
                                          pool_row_stride) ||
        !ds4_gpu_encode_concat_f32_dim1(cb,
                                          statescbuf,
                                          ds4_gpu_tensor_offset(state_score),
                                          4,
                                          state_row_stride,
                                          statescbuf,
                                          ds4_gpu_tensor_offset(state_score) + curr_offset,
                                          4,
                                          state_row_stride,
                                          g_compressor_pool_score_buffer,
                                          0,
                                          head_dim,
                                          pool_row_stride)) {
        return 0;
    }

    return ds4_gpu_encode_dsv4_softmax_pool(cb,
                                              out,
                                              g_compressor_pool_kv_buffer,
                                              0,
                                              pool_row_stride,
                                              sizeof(float),
                                              packed_bytes,
                                              g_compressor_pool_score_buffer,
                                              0,
                                              pool_row_stride,
                                              sizeof(float),
                                              packed_bytes,
                                              8,
                                              head_dim,
                                              1);
}

static int ds4_gpu_encode_compressor_shift_ratio4(
        id<MTLCommandBuffer> cb,
        ds4_gpu_tensor    *state_kv,
        ds4_gpu_tensor    *state_score,
        uint32_t             width) {
    id<MTLBuffer> statekvbuf = ds4_gpu_tensor_buffer(state_kv);
    id<MTLBuffer> statescbuf = ds4_gpu_tensor_buffer(state_score);
    if (!cb || !statekvbuf || !statescbuf || !g_dsv4_ratio4_shift_pipeline || width == 0) return 0;

    ds4_gpu_dsv4_ratio4_shift_args args = { .width = width };
    const uint32_t n = 4u * width;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:g_dsv4_ratio4_shift_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:statekvbuf offset:ds4_gpu_tensor_offset(state_kv) atIndex:1];
    [enc setBuffer:statescbuf offset:ds4_gpu_tensor_offset(state_score) atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)n + 255u) / 256u, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

int ds4_gpu_compressor_prefill_tensor(
        ds4_gpu_tensor       *comp_cache,
        ds4_gpu_tensor       *state_kv,
        ds4_gpu_tensor       *state_score,
        const ds4_gpu_tensor *kv,
        const ds4_gpu_tensor *sc,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint64_t                norm_offset,
        uint32_t                norm_type,
        uint32_t                head_dim,
        uint32_t                ratio,
        uint32_t                pos0,
        uint32_t                n_tokens,
        uint32_t                n_rot,
        uint32_t                n_ctx_orig,
        bool                    quantize_fp8,
        float                   freq_base,
        float                   freq_scale,
        float                   ext_factor,
        float                   attn_factor,
        float                   beta_fast,
        float                   beta_slow,
        float                   rms_eps) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!comp_cache || !state_kv || !state_score || !kv || !sc || !model_map ||
        head_dim == 0 || ratio == 0 || n_tokens == 0 ||
        n_rot > head_dim || (n_rot & 1u) != 0 ||
        (ape_type != 0u && ape_type != 1u) ||
        norm_type != 0u) {
        return 0;
    }

    @autoreleasepool {
        const uint32_t coff = ratio == 4u ? 2u : 1u;
        const uint32_t width = coff * head_dim;
        const uint32_t state_rows = coff * ratio;
        const uint32_t n_comp = n_tokens / ratio;
        const uint32_t cutoff = n_comp * ratio;
        const uint32_t rem = n_tokens - cutoff;
        const uint64_t elem_ape = ape_type == 1u ? 2u : 4u;
        const uint64_t kv_bytes = (uint64_t)n_tokens * width * sizeof(float);
        const uint64_t state_bytes = (uint64_t)state_rows * width * sizeof(float);
        const uint64_t comp_bytes = (uint64_t)n_comp * head_dim * sizeof(float);
        const uint64_t ape_bytes = (uint64_t)width * ratio * elem_ape;
        const uint64_t norm_bytes = (uint64_t)head_dim * sizeof(float);

        if (ape_offset > model_size || ape_bytes > model_size - ape_offset ||
            norm_offset > model_size || norm_bytes > model_size - norm_offset) {
            fprintf(stderr, "ds4: Metal compressor prefill tensor range is outside the mapped model\n");
            return 0;
        }

        id<MTLBuffer> kvbuf = ds4_gpu_tensor_buffer(kv);
        id<MTLBuffer> scbuf = ds4_gpu_tensor_buffer(sc);
        id<MTLBuffer> compbuf = ds4_gpu_tensor_buffer(comp_cache);
        id<MTLBuffer> statekvbuf = ds4_gpu_tensor_buffer(state_kv);
        id<MTLBuffer> statescbuf = ds4_gpu_tensor_buffer(state_score);
        if (!kvbuf || !scbuf || !compbuf || !statekvbuf || !statescbuf ||
            ds4_gpu_tensor_bytes(kv) < kv_bytes ||
            ds4_gpu_tensor_bytes(sc) < kv_bytes ||
            ds4_gpu_tensor_bytes(state_kv) < state_bytes ||
            ds4_gpu_tensor_bytes(state_score) < state_bytes ||
            (n_comp && ds4_gpu_tensor_bytes(comp_cache) < comp_bytes)) {
            fprintf(stderr, "ds4: Metal compressor prefill received undersized buffers\n");
            return 0;
        }

        uint64_t ape_inner = 0;
        id<MTLBuffer> apebuf = ds4_gpu_wrap_model_range(model_map, model_size, ape_offset, ape_bytes, &ape_inner);
        if (!apebuf) return 0;

        const bool had_batch = g_batch_cb != nil;
        if (!had_batch && ds4_gpu_begin_commands() == 0) return 0;

        int ok = 1;
        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb || owned) ok = 0;

        if (ok) {
            ok = ds4_gpu_encode_fill_f32_rows(cb,
                                                statekvbuf,
                                                ds4_gpu_tensor_offset(state_kv),
                                                width,
                                                state_rows,
                                                0.0f) &&
                 ds4_gpu_encode_fill_f32_rows(cb,
                                                statescbuf,
                                                ds4_gpu_tensor_offset(state_score),
                                                width,
                                                state_rows,
                                                ds4_gpu_negative_infinity());
        }

        if (ok && ratio == 4u) {
            int32_t rows_prev[4] = { 0, 1, 2, 3 };
            const int have_prev = cutoff >= ratio ? 1 : 0;
            const uint32_t prev_start = rem == 0 ? cutoff - ratio : cutoff - ratio;
            if (have_prev) {
                ok = ds4_gpu_encode_compressor_set_rows_projected(cb,
                                                                     state_kv,
                                                                     state_score,
                                                                     kvbuf,
                                                                     ds4_gpu_tensor_offset(kv) +
                                                                             (NSUInteger)prev_start * width * sizeof(float),
                                                                     scbuf,
                                                                     ds4_gpu_tensor_offset(sc) +
                                                                             (NSUInteger)prev_start * width * sizeof(float),
                                                                     apebuf,
                                                                     (NSUInteger)ape_inner,
                                                                     ape_type,
                                                                     width,
                                                                     ratio,
                                                                     pos0 + prev_start,
                                                                     rows_prev,
                                                                     4,
                                                                     state_rows);
            }
            if (ok && rem != 0) {
                int32_t rows_cur[4];
                for (uint32_t i = 0; i < rem; i++) rows_cur[i] = (int32_t)(ratio + i);
                ok = ds4_gpu_encode_compressor_set_rows_projected(cb,
                                                                     state_kv,
                                                                     state_score,
                                                                     kvbuf,
                                                                     ds4_gpu_tensor_offset(kv) +
                                                                             (NSUInteger)cutoff * width * sizeof(float),
                                                                     scbuf,
                                                                     ds4_gpu_tensor_offset(sc) +
                                                                             (NSUInteger)cutoff * width * sizeof(float),
                                                                     apebuf,
                                                                     (NSUInteger)ape_inner,
                                                                     ape_type,
                                                                     width,
                                                                     ratio,
                                                                     pos0 + cutoff,
                                                                     rows_cur,
                                                                     rem,
                                                                     state_rows);
            }
        } else if (ok && rem != 0) {
            int32_t rows[128];
            if (rem > (uint32_t)(sizeof(rows) / sizeof(rows[0]))) {
                fprintf(stderr, "ds4: Metal compressor prefill remainder exceeds local row list\n");
                ok = 0;
            } else {
                for (uint32_t i = 0; i < rem; i++) rows[i] = (int32_t)i;
                ok = ds4_gpu_encode_compressor_set_rows_projected(cb,
                                                                     state_kv,
                                                                     state_score,
                                                                     kvbuf,
                                                                     ds4_gpu_tensor_offset(kv) +
                                                                             (NSUInteger)cutoff * width * sizeof(float),
                                                                     scbuf,
                                                                     ds4_gpu_tensor_offset(sc) +
                                                                             (NSUInteger)cutoff * width * sizeof(float),
                                                                     apebuf,
                                                                     (NSUInteger)ape_inner,
                                                                     ape_type,
                                                                     width,
                                                                     ratio,
                                                                     pos0 + cutoff,
                                                                     rows,
                                                                     rem,
                                                                     state_rows);
            }
        }

        if (ok && n_comp != 0) {
            const NSUInteger score_bytes = (NSUInteger)cutoff * width * sizeof(float);
            if (!ds4_gpu_ensure_scratch_buffer(&g_compressor_store_score_buffer,
                                                 &g_compressor_store_score_bytes,
                                                 score_bytes,
                                                 "ds4_compressor_store_score")) {
                ok = 0;
            }
            if (ok) {
                ok = ds4_gpu_encode_compressor_score_with_ape(cb,
                                                                 scbuf,
                                                                 ds4_gpu_tensor_offset(sc),
                                                                 g_compressor_store_score_buffer,
                                                                 0,
                                                                 apebuf,
                                                                 (NSUInteger)ape_inner,
                                                                 ape_type,
                                                                 width,
                                                                 ratio,
                                                                 pos0,
                                                                 cutoff);
            }

            if (ok && ratio == 4u) {
                const int direct_pool_mode =
                    ds4_gpu_compressor_ratio4_direct_pool_mode(head_dim, n_comp);
                if (ok && direct_pool_mode < 0) ok = 0;
                const uint64_t direct_output_bytes =
                    (uint64_t)n_comp * head_dim * sizeof(float);
                const uint64_t direct_input_bytes =
                    (uint64_t)n_comp * 4u * width * sizeof(float);
                const bool direct_pool_overlap = direct_pool_mode > 0 &&
                    ds4_gpu_buffer_ranges_overlap(
                        compbuf,
                        ds4_gpu_tensor_offset(comp_cache),
                        direct_output_bytes,
                        kvbuf,
                        ds4_gpu_tensor_offset(kv),
                        direct_input_bytes);
                const bool use_direct_pool =
                    direct_pool_mode > 0 && !direct_pool_overlap;
                const NSUInteger pack_bytes = (NSUInteger)n_comp * 8u * head_dim * sizeof(float);
                if (ok && !use_direct_pool &&
                    (!ds4_gpu_ensure_scratch_buffer(&g_compressor_pool_kv_buffer,
                                                      &g_compressor_pool_kv_bytes,
                                                      pack_bytes,
                                                      "ds4_compressor_pool_kv") ||
                     !ds4_gpu_ensure_scratch_buffer(&g_compressor_pool_score_buffer,
                                                      &g_compressor_pool_score_bytes,
                                                      pack_bytes,
                                                      "ds4_compressor_pool_score"))) {
                    ok = 0;
                }
                const int pack_fusion_mode = use_direct_pool ? 0 :
                    ds4_gpu_compressor_pack_ratio4_fusion_mode(head_dim);
                if (ok && pack_fusion_mode < 0) ok = 0;
                const bool use_pack_fusion = pack_fusion_mode > 0;
                if (ok && use_direct_pool) {
                    ok = ds4_gpu_encode_compressor_ratio4_direct_pool(
                        cb,
                        comp_cache,
                        kvbuf,
                        ds4_gpu_tensor_offset(kv),
                        g_compressor_store_score_buffer,
                        0,
                        statekvbuf,
                        ds4_gpu_tensor_offset(state_kv),
                        statescbuf,
                        ds4_gpu_tensor_offset(state_score),
                        head_dim,
                        n_comp,
                        false);
                }
                if (ok && use_pack_fusion) {
                    ok = ds4_gpu_encode_compressor_pack_ratio4(
                        cb,
                        kvbuf,
                        ds4_gpu_tensor_offset(kv),
                        g_compressor_store_score_buffer,
                        0,
                        statekvbuf,
                        ds4_gpu_tensor_offset(state_kv),
                        statescbuf,
                        ds4_gpu_tensor_offset(state_score),
                        head_dim,
                        n_comp,
                        false);
                }
                if (ok && !use_direct_pool && !use_pack_fusion) {
                    ok = ds4_gpu_encode_fill_f32_rows(cb,
                                                        g_compressor_pool_kv_buffer,
                                                        0,
                                                        head_dim,
                                                        8u * n_comp,
                                                        0.0f) &&
                         ds4_gpu_encode_fill_f32_rows(cb,
                                                        g_compressor_pool_score_buffer,
                                                        0,
                                                        head_dim,
                                                        8u * n_comp,
                                                        ds4_gpu_negative_infinity());
                }
                if (ok && !use_direct_pool && !use_pack_fusion) {
                    const uint64_t src_row_stride = (uint64_t)width * sizeof(float);
                    const uint64_t src_plane_stride = (uint64_t)ratio * src_row_stride;
                    const uint64_t dst_row_stride = (uint64_t)head_dim * sizeof(float);
                    const uint64_t dst_plane_stride = 8ull * dst_row_stride;
                    ok = ds4_gpu_encode_cpy_f32_f32_3d(cb,
                                                         kvbuf,
                                                         ds4_gpu_tensor_offset(kv) +
                                                                 (NSUInteger)head_dim * sizeof(float),
                                                         g_compressor_pool_kv_buffer,
                                                         (NSUInteger)4u * head_dim * sizeof(float),
                                                         head_dim,
                                                         ratio,
                                                         n_comp,
                                                         src_row_stride,
                                                         src_plane_stride,
                                                         dst_row_stride,
                                                         dst_plane_stride) &&
                         ds4_gpu_encode_cpy_f32_f32_3d(cb,
                                                         g_compressor_store_score_buffer,
                                                         (NSUInteger)head_dim * sizeof(float),
                                                         g_compressor_pool_score_buffer,
                                                         (NSUInteger)4u * head_dim * sizeof(float),
                                                         head_dim,
                                                         ratio,
                                                         n_comp,
                                                         src_row_stride,
                                                         src_plane_stride,
                                                         dst_row_stride,
                                                         dst_plane_stride);
                }
                if (ok && !use_direct_pool && !use_pack_fusion && n_comp > 1u) {
                    const uint64_t src_row_stride = (uint64_t)width * sizeof(float);
                    const uint64_t src_plane_stride = (uint64_t)ratio * src_row_stride;
                    const uint64_t dst_row_stride = (uint64_t)head_dim * sizeof(float);
                    const uint64_t dst_plane_stride = 8ull * dst_row_stride;
                    ok = ds4_gpu_encode_cpy_f32_f32_3d(cb,
                                                         kvbuf,
                                                         ds4_gpu_tensor_offset(kv),
                                                         g_compressor_pool_kv_buffer,
                                                         dst_plane_stride,
                                                         head_dim,
                                                         ratio,
                                                         n_comp - 1u,
                                                         src_row_stride,
                                                         src_plane_stride,
                                                         dst_row_stride,
                                                         dst_plane_stride) &&
                         ds4_gpu_encode_cpy_f32_f32_3d(cb,
                                                         g_compressor_store_score_buffer,
                                                         0,
                                                         g_compressor_pool_score_buffer,
                                                         dst_plane_stride,
                                                         head_dim,
                                                         ratio,
                                                         n_comp - 1u,
                                                         src_row_stride,
                                                         src_plane_stride,
                                                         dst_row_stride,
                                                         dst_plane_stride);
                }
                if (ok && !use_direct_pool) {
                    ok = ds4_gpu_encode_dsv4_softmax_pool(cb,
                                                            comp_cache,
                                                            g_compressor_pool_kv_buffer,
                                                            0,
                                                            (uint64_t)head_dim * sizeof(float),
                                                            sizeof(float),
                                                            8ull * head_dim * sizeof(float),
                                                            g_compressor_pool_score_buffer,
                                                            0,
                                                            (uint64_t)head_dim * sizeof(float),
                                                            sizeof(float),
                                                            8ull * head_dim * sizeof(float),
                                                            8,
                                                            head_dim,
                                                            n_comp);
                }
            } else if (ok) {
                const uint64_t row_stride = (uint64_t)width * sizeof(float);
                ok = ds4_gpu_encode_dsv4_softmax_pool(cb,
                                                        comp_cache,
                                                        kvbuf,
                                                        ds4_gpu_tensor_offset(kv),
                                                        row_stride,
                                                        sizeof(float),
                                                        (uint64_t)ratio * row_stride,
                                                        g_compressor_store_score_buffer,
                                                        0,
                                                        row_stride,
                                                        sizeof(float),
                                                        (uint64_t)ratio * row_stride,
                                                        ratio,
                                                        head_dim,
                                                        n_comp);
            }
        }

        if (ok && n_comp != 0) {
            ok = ds4_gpu_rms_norm_weight_rows_tensor(comp_cache,
                                                       comp_cache,
                                                       model_map,
                                                       model_size,
                                                       norm_offset,
                                                       head_dim,
                                                       n_comp,
                                                       rms_eps) != 0;
        }
        if (ok && n_comp != 0 && n_rot != 0) {
            ds4_gpu_rope_tail_batch_args rope_args = ds4_gpu_make_rope_tail_args(
                n_comp, 1, head_dim, n_rot, n_ctx_orig, false,
                freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
            cb = ds4_gpu_command_buffer(&owned);
            ok = cb && !owned &&
                 ds4_gpu_encode_rope_tail_inplace(cb,
                                                    compbuf,
                                                    ds4_gpu_tensor_offset(comp_cache),
                                                    &rope_args,
                                                    n_comp,
                                                    1,
                                                    head_dim,
                                                    pos0,
                                                    ratio);
        }
        if (ok && n_comp != 0 && quantize_fp8) {
            ok = ds4_gpu_dsv4_fp8_kv_quantize_tensor(comp_cache, n_comp, head_dim, n_rot) != 0;
        }

        if (!had_batch) {
            const int end_ok = ds4_gpu_end_commands();
            ok = end_ok && ok;
        }
        return ok ? 1 : 0;
    }
}

int ds4_gpu_compressor_prefill_ratio4_replay_tensor(
        ds4_gpu_tensor       *comp_cache,
        ds4_gpu_tensor       *state_kv,
        ds4_gpu_tensor       *state_score,
        const ds4_gpu_tensor *kv,
        const ds4_gpu_tensor *sc,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint64_t                norm_offset,
        uint32_t                norm_type,
        uint32_t                head_dim,
        uint32_t                pos0,
        uint32_t                n_tokens,
        uint32_t                n_rot,
        uint32_t                n_ctx_orig,
        bool                    quantize_fp8,
        float                   freq_base,
        float                   freq_scale,
        float                   ext_factor,
        float                   attn_factor,
        float                   beta_fast,
        float                   beta_slow,
        float                   rms_eps) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!comp_cache || !state_kv || !state_score || !kv || !sc || !model_map ||
        head_dim == 0 || n_tokens == 0 || (n_tokens & 3u) != 0 || (pos0 & 3u) != 0 ||
        n_rot > head_dim || (n_rot & 1u) != 0 ||
        (ape_type != 0u && ape_type != 1u) ||
        norm_type != 0u) {
        return 0;
    }

    @autoreleasepool {
        const uint32_t ratio = 4u;
        const uint32_t width = 2u * head_dim;
        const uint32_t state_rows = 8u;
        const uint32_t n_comp = n_tokens / ratio;
        const uint64_t elem_ape = ape_type == 1u ? 2u : 4u;
        const uint64_t kv_bytes = (uint64_t)n_tokens * width * sizeof(float);
        const uint64_t state_bytes = (uint64_t)state_rows * width * sizeof(float);
        const uint64_t comp_bytes = (uint64_t)n_comp * head_dim * sizeof(float);
        const uint64_t ape_bytes = (uint64_t)width * ratio * elem_ape;
        const uint64_t norm_bytes = (uint64_t)head_dim * sizeof(float);

        if (ape_offset > model_size || ape_bytes > model_size - ape_offset ||
            norm_offset > model_size || norm_bytes > model_size - norm_offset) {
            fprintf(stderr, "ds4: Metal compressor replay tensor range is outside the mapped model\n");
            return 0;
        }

        id<MTLBuffer> kvbuf = ds4_gpu_tensor_buffer(kv);
        id<MTLBuffer> scbuf = ds4_gpu_tensor_buffer(sc);
        id<MTLBuffer> compbuf = ds4_gpu_tensor_buffer(comp_cache);
        id<MTLBuffer> statekvbuf = ds4_gpu_tensor_buffer(state_kv);
        id<MTLBuffer> statescbuf = ds4_gpu_tensor_buffer(state_score);
        if (!kvbuf || !scbuf || !compbuf || !statekvbuf || !statescbuf ||
            ds4_gpu_tensor_bytes(kv) < kv_bytes ||
            ds4_gpu_tensor_bytes(sc) < kv_bytes ||
            ds4_gpu_tensor_bytes(state_kv) < state_bytes ||
            ds4_gpu_tensor_bytes(state_score) < state_bytes ||
            ds4_gpu_tensor_bytes(comp_cache) < comp_bytes) {
            fprintf(stderr, "ds4: Metal compressor replay received undersized buffers\n");
            return 0;
        }

        uint64_t ape_inner = 0;
        id<MTLBuffer> apebuf = ds4_gpu_wrap_model_range(model_map, model_size, ape_offset, ape_bytes, &ape_inner);
        if (!apebuf) return 0;

        const bool had_batch = g_batch_cb != nil;
        if (!had_batch && ds4_gpu_begin_commands() == 0) return 0;

        int ok = 1;
        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb || owned) ok = 0;

        const NSUInteger score_bytes = (NSUInteger)n_tokens * width * sizeof(float);
        const NSUInteger pack_bytes = (NSUInteger)n_comp * 8u * head_dim * sizeof(float);
        const int direct_pool_mode =
            ds4_gpu_compressor_ratio4_direct_pool_mode(head_dim, n_comp);
        if (ok && direct_pool_mode < 0) ok = 0;
        const uint64_t direct_output_bytes =
            (uint64_t)n_comp * head_dim * sizeof(float);
        const uint64_t direct_input_bytes =
            (uint64_t)n_tokens * width * sizeof(float);
        const uint64_t direct_state_bytes =
            (uint64_t)4u * width * sizeof(float);
        const bool direct_pool_overlap = direct_pool_mode > 0 &&
            (ds4_gpu_buffer_ranges_overlap(
                 compbuf,
                 ds4_gpu_tensor_offset(comp_cache),
                 direct_output_bytes,
                 kvbuf,
                 ds4_gpu_tensor_offset(kv),
                 direct_input_bytes) ||
             ds4_gpu_buffer_ranges_overlap(
                 compbuf,
                 ds4_gpu_tensor_offset(comp_cache),
                 direct_output_bytes,
                 statekvbuf,
                 ds4_gpu_tensor_offset(state_kv),
                 direct_state_bytes) ||
             ds4_gpu_buffer_ranges_overlap(
                 compbuf,
                 ds4_gpu_tensor_offset(comp_cache),
                 direct_output_bytes,
                 statescbuf,
                 ds4_gpu_tensor_offset(state_score),
                 direct_state_bytes));
        const bool use_direct_pool =
            direct_pool_mode > 0 && !direct_pool_overlap;
        if (ok && (!ds4_gpu_ensure_scratch_buffer(&g_compressor_store_score_buffer,
                                                    &g_compressor_store_score_bytes,
                                                    score_bytes,
                                                    "ds4_compressor_store_score") ||
                   (!use_direct_pool &&
                    (!ds4_gpu_ensure_scratch_buffer(&g_compressor_pool_kv_buffer,
                                                     &g_compressor_pool_kv_bytes,
                                                     pack_bytes,
                                                     "ds4_compressor_pool_kv") ||
                     !ds4_gpu_ensure_scratch_buffer(&g_compressor_pool_score_buffer,
                                                     &g_compressor_pool_score_bytes,
                                                     pack_bytes,
                                                     "ds4_compressor_pool_score"))))) {
            ok = 0;
        }

        if (ok) {
            ok = ds4_gpu_encode_compressor_score_with_ape(cb,
                                                            scbuf,
                                                            ds4_gpu_tensor_offset(sc),
                                                            g_compressor_store_score_buffer,
                                                            0,
                                                            apebuf,
                                                            (NSUInteger)ape_inner,
                                                            ape_type,
                                                            width,
                                                            ratio,
                                                            pos0,
                                                            n_tokens);
        }

        const uint64_t src_row_stride = (uint64_t)width * sizeof(float);
        const uint64_t src_plane_stride = (uint64_t)ratio * src_row_stride;
        const uint64_t dst_row_stride = (uint64_t)head_dim * sizeof(float);
        const uint64_t dst_plane_stride = 8ull * dst_row_stride;
        const NSUInteger state_off = ds4_gpu_tensor_offset(state_kv);
        const NSUInteger state_score_off = ds4_gpu_tensor_offset(state_score);
        const int pack_fusion_mode = use_direct_pool ? 0 :
            ds4_gpu_compressor_pack_ratio4_fusion_mode(head_dim);
        if (ok && pack_fusion_mode < 0) ok = 0;
        const bool use_pack_fusion = pack_fusion_mode > 0;

        if (ok && use_direct_pool) {
            ok = ds4_gpu_encode_compressor_ratio4_direct_pool(
                cb,
                comp_cache,
                kvbuf,
                ds4_gpu_tensor_offset(kv),
                g_compressor_store_score_buffer,
                0,
                statekvbuf,
                state_off,
                statescbuf,
                state_score_off,
                head_dim,
                n_comp,
                true);
        }

        if (ok && use_pack_fusion) {
            ok = ds4_gpu_encode_compressor_pack_ratio4(
                cb,
                kvbuf,
                ds4_gpu_tensor_offset(kv),
                g_compressor_store_score_buffer,
                0,
                statekvbuf,
                state_off,
                statescbuf,
                state_score_off,
                head_dim,
                n_comp,
                true);
        }

        if (ok && !use_direct_pool && !use_pack_fusion) {
            ok = ds4_gpu_encode_fill_f32_rows(cb,
                                                g_compressor_pool_kv_buffer,
                                                0,
                                                head_dim,
                                                8u * n_comp,
                                                0.0f) &&
                 ds4_gpu_encode_fill_f32_rows(cb,
                                                g_compressor_pool_score_buffer,
                                                0,
                                                head_dim,
                                                8u * n_comp,
                                                ds4_gpu_negative_infinity());
        }

        if (ok && !use_direct_pool && !use_pack_fusion) {
            /*
             * The aligned nonzero ratio-4 path replays the current ubatch
             * compressor, but seeds the first compressed row with the previous
             * compressor state. Rows 0..3 are the previous half, rows 4..7 are
             * the current half.
             */
            ok = ds4_gpu_encode_cpy_f32_f32_3d(cb,
                                                 statekvbuf,
                                                 state_off,
                                                 g_compressor_pool_kv_buffer,
                                                 0,
                                                 head_dim,
                                                 ratio,
                                                 1,
                                                 src_row_stride,
                                                 (uint64_t)ratio * src_row_stride,
                                                 dst_row_stride,
                                                 dst_plane_stride) &&
                 ds4_gpu_encode_cpy_f32_f32_3d(cb,
                                                 statescbuf,
                                                 state_score_off,
                                                 g_compressor_pool_score_buffer,
                                                 0,
                                                 head_dim,
                                                 ratio,
                                                 1,
                                                 src_row_stride,
                                                 (uint64_t)ratio * src_row_stride,
                                                 dst_row_stride,
                                                 dst_plane_stride);
        }
        if (ok && !use_direct_pool && !use_pack_fusion) {
            ok = ds4_gpu_encode_cpy_f32_f32_3d(cb,
                                                 kvbuf,
                                                 ds4_gpu_tensor_offset(kv) +
                                                         (NSUInteger)head_dim * sizeof(float),
                                                 g_compressor_pool_kv_buffer,
                                                 (NSUInteger)4u * head_dim * sizeof(float),
                                                 head_dim,
                                                 ratio,
                                                 n_comp,
                                                 src_row_stride,
                                                 src_plane_stride,
                                                 dst_row_stride,
                                                 dst_plane_stride) &&
                 ds4_gpu_encode_cpy_f32_f32_3d(cb,
                                                 g_compressor_store_score_buffer,
                                                 (NSUInteger)head_dim * sizeof(float),
                                                 g_compressor_pool_score_buffer,
                                                 (NSUInteger)4u * head_dim * sizeof(float),
                                                 head_dim,
                                                 ratio,
                                                 n_comp,
                                                 src_row_stride,
                                                 src_plane_stride,
                                                 dst_row_stride,
                                                 dst_plane_stride);
        }
        if (ok && !use_direct_pool && !use_pack_fusion && n_comp > 1u) {
            ok = ds4_gpu_encode_cpy_f32_f32_3d(cb,
                                                 kvbuf,
                                                 ds4_gpu_tensor_offset(kv),
                                                 g_compressor_pool_kv_buffer,
                                                 dst_plane_stride,
                                                 head_dim,
                                                 ratio,
                                                 n_comp - 1u,
                                                 src_row_stride,
                                                 src_plane_stride,
                                                 dst_row_stride,
                                                 dst_plane_stride) &&
                 ds4_gpu_encode_cpy_f32_f32_3d(cb,
                                                 g_compressor_store_score_buffer,
                                                 0,
                                                 g_compressor_pool_score_buffer,
                                                 dst_plane_stride,
                                                 head_dim,
                                                 ratio,
                                                 n_comp - 1u,
                                                 src_row_stride,
                                                 src_plane_stride,
                                                 dst_row_stride,
                                                 dst_plane_stride);
        }
        if (ok && !use_direct_pool) {
            ok = ds4_gpu_encode_dsv4_softmax_pool(cb,
                                                    comp_cache,
                                                    g_compressor_pool_kv_buffer,
                                                    0,
                                                    dst_row_stride,
                                                    sizeof(float),
                                                    dst_plane_stride,
                                                    g_compressor_pool_score_buffer,
                                                    0,
                                                    dst_row_stride,
                                                    sizeof(float),
                                                    dst_plane_stride,
                                                    8,
                                                    head_dim,
                                                    n_comp);
        }
        if (ok) {
            ok = ds4_gpu_rms_norm_weight_rows_tensor(comp_cache,
                                                       comp_cache,
                                                       model_map,
                                                       model_size,
                                                       norm_offset,
                                                       head_dim,
                                                       n_comp,
                                                       rms_eps) != 0;
        }
        if (ok && n_rot != 0) {
            ds4_gpu_rope_tail_batch_args rope_args = ds4_gpu_make_rope_tail_args(
                n_comp, 1, head_dim, n_rot, n_ctx_orig, false,
                freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
            cb = ds4_gpu_command_buffer(&owned);
            ok = cb && !owned &&
                 ds4_gpu_encode_rope_tail_inplace(cb,
                                                    compbuf,
                                                    ds4_gpu_tensor_offset(comp_cache),
                                                    &rope_args,
                                                    n_comp,
                                                    1,
                                                    head_dim,
                                                    pos0,
                                                    ratio);
        }
        if (ok && quantize_fp8) {
            ok = ds4_gpu_dsv4_fp8_kv_quantize_tensor(comp_cache, n_comp, head_dim, n_rot) != 0;
        }

        if (ok) {
            ok = ds4_gpu_encode_fill_f32_rows(cb,
                                                statekvbuf,
                                                state_off,
                                                width,
                                                state_rows,
                                                0.0f) &&
                 ds4_gpu_encode_fill_f32_rows(cb,
                                                statescbuf,
                                                state_score_off,
                                                width,
                                                state_rows,
                                                ds4_gpu_negative_infinity());
        }
        if (ok) {
            int32_t rows_prev[4] = { 0, 1, 2, 3 };
            const uint32_t prev_start = n_tokens - ratio;
            ok = ds4_gpu_encode_compressor_set_rows_projected(cb,
                                                                 state_kv,
                                                                 state_score,
                                                                 kvbuf,
                                                                 ds4_gpu_tensor_offset(kv) +
                                                                         (NSUInteger)prev_start * width * sizeof(float),
                                                                 scbuf,
                                                                 ds4_gpu_tensor_offset(sc) +
                                                                         (NSUInteger)prev_start * width * sizeof(float),
                                                                 apebuf,
                                                                 (NSUInteger)ape_inner,
                                                                 ape_type,
                                                                 width,
                                                                 ratio,
                                                                 pos0 + prev_start,
                                                                 rows_prev,
                                                                 ratio,
                                                                 state_rows);
        }

        if (!had_batch) {
            const int end_ok = ds4_gpu_end_commands();
            ok = end_ok && ok;
        }
        return ok ? 1 : 0;
    }
}

int ds4_gpu_compressor_prefill_state_ratio4_tensor(
        ds4_gpu_tensor       *state_kv,
        ds4_gpu_tensor       *state_score,
        const ds4_gpu_tensor *kv_tail,
        const ds4_gpu_tensor *sc_tail,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint32_t                head_dim,
        uint32_t                pos0) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!state_kv || !state_score || !kv_tail || !sc_tail || !model_map ||
        head_dim == 0 || (ape_type != 0u && ape_type != 1u)) {
        return 0;
    }

    @autoreleasepool {
        const uint32_t ratio = 4u;
        const uint32_t width = 2u * head_dim;
        const uint32_t state_rows = 8u;
        const uint64_t elem_ape = ape_type == 1u ? 2u : 4u;
        const uint64_t tail_bytes = (uint64_t)ratio * width * sizeof(float);
        const uint64_t state_bytes = (uint64_t)state_rows * width * sizeof(float);
        const uint64_t ape_bytes = (uint64_t)ratio * width * elem_ape;

        if (ape_offset > model_size || ape_bytes > model_size - ape_offset) {
            fprintf(stderr, "ds4: Metal compressor prefill-state APE range is outside the mapped model\n");
            return 0;
        }

        id<MTLBuffer> kvbuf = ds4_gpu_tensor_buffer(kv_tail);
        id<MTLBuffer> scbuf = ds4_gpu_tensor_buffer(sc_tail);
        id<MTLBuffer> statekvbuf = ds4_gpu_tensor_buffer(state_kv);
        id<MTLBuffer> statescbuf = ds4_gpu_tensor_buffer(state_score);
        if (!kvbuf || !scbuf || !statekvbuf || !statescbuf ||
            ds4_gpu_tensor_bytes(kv_tail) < tail_bytes ||
            ds4_gpu_tensor_bytes(sc_tail) < tail_bytes ||
            ds4_gpu_tensor_bytes(state_kv) < state_bytes ||
            ds4_gpu_tensor_bytes(state_score) < state_bytes) {
            fprintf(stderr, "ds4: Metal compressor prefill-state received undersized buffers\n");
            return 0;
        }

        uint64_t ape_inner = 0;
        id<MTLBuffer> apebuf = ds4_gpu_wrap_model_range(model_map, model_size, ape_offset, ape_bytes, &ape_inner);
        if (!apebuf) return 0;

        const bool had_batch = g_batch_cb != nil;
        if (!had_batch && ds4_gpu_begin_commands() == 0) return 0;

        int ok = 1;
        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb || owned) ok = 0;

        if (ok) {
            ok = ds4_gpu_encode_fill_f32_rows(cb,
                                                statekvbuf,
                                                ds4_gpu_tensor_offset(state_kv),
                                                width,
                                                state_rows,
                                                0.0f) &&
                 ds4_gpu_encode_fill_f32_rows(cb,
                                                statescbuf,
                                                ds4_gpu_tensor_offset(state_score),
                                                width,
                                                state_rows,
                                                ds4_gpu_negative_infinity());
        }
        if (ok) {
            int32_t rows[4] = { 0, 1, 2, 3 };
            ok = ds4_gpu_encode_compressor_set_rows_projected(cb,
                                                                 state_kv,
                                                                 state_score,
                                                                 kvbuf,
                                                                 ds4_gpu_tensor_offset(kv_tail),
                                                                 scbuf,
                                                                 ds4_gpu_tensor_offset(sc_tail),
                                                                 apebuf,
                                                                 (NSUInteger)ape_inner,
                                                                 ape_type,
                                                                 width,
                                                                 ratio,
                                                                 pos0,
                                                                 rows,
                                                                 ratio,
                                                                 state_rows);
        }

        if (!had_batch) {
            const int end_ok = ds4_gpu_end_commands();
            ok = end_ok && ok;
        }
        return ok ? 1 : 0;
    }
}

int ds4_gpu_compressor_update_tensor(
        const ds4_gpu_tensor *kv_cur,
        const ds4_gpu_tensor *sc_cur,
        ds4_gpu_tensor       *state_kv,
        ds4_gpu_tensor       *state_score,
        ds4_gpu_tensor       *comp_cache,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint64_t                norm_offset,
        uint32_t                norm_type,
        uint32_t                head_dim,
        uint32_t                ratio,
        uint32_t                pos,
        uint32_t                comp_row,
        uint32_t                n_rot,
        uint32_t                n_ctx_orig,
        float                   freq_base,
        float                   freq_scale,
        float                   ext_factor,
        float                   attn_factor,
        float                   beta_fast,
        float                   beta_slow,
        float                   rms_eps,
        bool                    state_already_stored) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!kv_cur || !sc_cur || !state_kv || !state_score || !comp_cache ||
        !model_map || head_dim == 0 || ratio == 0 ||
        n_rot > head_dim || (n_rot & 1u) != 0 ||
        (ape_type != 0u && ape_type != 1u) ||
        norm_type != 0u) {
        return 0;
    }

    @autoreleasepool {
        const uint32_t coff = ratio == 4u ? 2u : 1u;
        const uint32_t width = coff * head_dim;
        const uint32_t state_rows = coff * ratio;
        const uint32_t emit = ((pos + 1u) % ratio) == 0u ? 1u : 0u;
        const uint64_t elem_ape = ape_type == 1u ? 2u : 4u;
        const uint64_t kv_bytes = (uint64_t)width * sizeof(float);
        const uint64_t state_bytes = (uint64_t)state_rows * width * sizeof(float);
        const uint64_t comp_bytes = (uint64_t)(comp_row + (emit ? 1u : 0u)) * head_dim * sizeof(float);
        const uint64_t ape_bytes = (uint64_t)width * ratio * elem_ape;
        const uint64_t norm_bytes = (uint64_t)head_dim * sizeof(float);

        if (ape_offset > model_size || ape_bytes > model_size - ape_offset ||
            norm_offset > model_size || norm_bytes > model_size - norm_offset) {
            fprintf(stderr, "ds4: Metal compressor tensor range is outside the mapped model\n");
            return 0;
        }

        id<MTLBuffer> kvbuf = ds4_gpu_tensor_buffer(kv_cur);
        id<MTLBuffer> scbuf = ds4_gpu_tensor_buffer(sc_cur);
        id<MTLBuffer> compbuf = ds4_gpu_tensor_buffer(comp_cache);
        if (!kvbuf || !scbuf || !compbuf ||
            ds4_gpu_tensor_bytes(kv_cur) < kv_bytes ||
            ds4_gpu_tensor_bytes(sc_cur) < kv_bytes ||
            ds4_gpu_tensor_bytes(state_kv) < state_bytes ||
            ds4_gpu_tensor_bytes(state_score) < state_bytes ||
            (emit && ds4_gpu_tensor_bytes(comp_cache) < comp_bytes)) {
            fprintf(stderr, "ds4: Metal compressor update received undersized buffers\n");
            return 0;
        }

        if (!state_already_stored) {
            const bool use_store_one =
                getenv("DS4_METAL_DISABLE_COMPRESSOR_STORE_ONE") == NULL;
            const int store_ok = use_store_one
                ? ds4_gpu_compressor_store_one_tensor(kv_cur,
                                                        sc_cur,
                                                        state_kv,
                                                        state_score,
                                                        model_map,
                                                        model_size,
                                                        ape_offset,
                                                        ape_type,
                                                        width,
                                                        ratio,
                                                        pos)
                : ds4_gpu_compressor_store_batch_tensor(kv_cur,
                                                          sc_cur,
                                                          state_kv,
                                                          state_score,
                                                          model_map,
                                                          model_size,
                                                          ape_offset,
                                                          ape_type,
                                                          head_dim,
                                                          ratio,
                                                          pos,
                                                          1);
            if (!store_ok) {
                return 0;
            }
        }
        if (!emit) return 1;

        ds4_gpu_tensor *comp_row_view = ds4_gpu_tensor_view(
                comp_cache,
                (uint64_t)comp_row * head_dim * sizeof(float),
                (uint64_t)head_dim * sizeof(float));
        if (!comp_row_view) return 0;

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        int ok = cb &&
                 ds4_gpu_encode_compressor_pool(cb,
                                                  comp_row_view,
                                                  state_kv,
                                                  state_score,
                                                  head_dim,
                                                  ratio);
        if (ok) ok = ds4_gpu_finish_command_buffer(cb, owned, "compressor DS4 softmax pool");
        if (ok) {
            ok = ds4_gpu_rms_norm_weight_rows_tensor(comp_row_view,
                                                       comp_row_view,
                                                       model_map,
                                                       model_size,
                                                       norm_offset,
                                                       head_dim,
                                                       1,
                                                       rms_eps) != 0;
        }
        if (ok) {
            const uint32_t comp_pos = pos + 1u - ratio;
            ok = ds4_gpu_rope_tail_tensor(comp_row_view,
                                            1,
                                            1,
                                            head_dim,
                                            n_rot,
                                            comp_pos,
                                            n_ctx_orig,
                                            false,
                                            freq_base,
                                            freq_scale,
                                            ext_factor,
                                            attn_factor,
                                            beta_fast,
                                            beta_slow) != 0;
        }
        if (ok && ratio == 4u) {
            cb = ds4_gpu_command_buffer(&owned);
            ok = cb &&
                 ds4_gpu_encode_compressor_shift_ratio4(cb,
                                                          state_kv,
                                                          state_score,
                                                          width);
            if (ok) ok = ds4_gpu_finish_command_buffer(cb, owned, "compressor ratio4 state shift");
        }
        ds4_gpu_tensor_free(comp_row_view);
        if (!ok) return 0;
    }

    return 1;
}

static int ds4_gpu_encode_fill_f32_rows(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        buf,
        NSUInteger           offset,
        uint32_t             width,
        uint32_t             rows,
        float                value) {
    if (!cb || !buf || width == 0 || rows == 0 || (width & 3u) != 0) return 0;

    ds4_gpu_unary_args args = ds4_gpu_make_unary_rows_args(width, rows, 1, 0.0f, 0.0f);
    args.val = value;

    NSUInteger nth_max = g_unary_fill_pipeline.maxTotalThreadsPerThreadgroup;
    if (nth_max > 256u) nth_max = 256u;
    NSUInteger nth = (NSUInteger)args.ne00;
    if (nth > nth_max) nth = nth_max;
    if (nth == 0) nth = 1u;
    const NSUInteger nk0 = ((NSUInteger)args.ne00 + nth - 1u) / nth;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:g_unary_fill_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:buf offset:offset atIndex:1];
    [enc setBuffer:buf offset:offset atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake(nk0 * (NSUInteger)args.ne01,
                                          (NSUInteger)args.ne02,
                                          (NSUInteger)args.ne03)
         threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

int ds4_gpu_attention_output_q8_batch_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *low,
        ds4_gpu_tensor       *group_tmp,
        ds4_gpu_tensor       *low_tmp,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                out_a_offset,
        uint64_t                out_b_offset,
        uint64_t                group_dim,
        uint64_t                rank,
        uint32_t                n_groups,
        uint64_t                out_dim,
        const ds4_gpu_tensor *heads,
        uint32_t                n_tokens) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!out || !low || !group_tmp || !low_tmp || !heads || !model_map ||
        group_dim == 0 || rank == 0 || n_groups == 0 || out_dim == 0 || n_tokens == 0 ||
        group_dim > UINT32_MAX || rank > UINT32_MAX || out_dim > UINT32_MAX) {
        return 0;
    }

    @autoreleasepool {
        const uint64_t low_dim = (uint64_t)n_groups * rank;
        if ((group_dim % 32u) != 0 || (low_dim % 32u) != 0 || low_dim > UINT32_MAX) {
            fprintf(stderr, "ds4: Metal attention output batch received invalid q8 dimensions\n");
            return 0;
        }
        const uint64_t row_a_bytes = (group_dim / 32u) * 34u;
        const uint64_t row_b_bytes = (low_dim / 32u) * 34u;
        const uint64_t out_a_bytes = (uint64_t)n_groups * rank * row_a_bytes;
        const uint64_t out_b_bytes = out_dim * row_b_bytes;
        if (out_a_offset > model_size || out_a_bytes > model_size - out_a_offset ||
            out_b_offset > model_size || out_b_bytes > model_size - out_b_offset) {
            fprintf(stderr, "ds4: Metal attention output batch weights are outside the mapped model\n");
            return 0;
        }

        const uint64_t heads_bytes = (uint64_t)n_tokens * n_groups * group_dim * sizeof(float);
        const uint64_t low_bytes = (uint64_t)n_tokens * low_dim * sizeof(float);
        const uint64_t out_bytes = (uint64_t)n_tokens * out_dim * sizeof(float);
        if (ds4_gpu_tensor_bytes(heads) < heads_bytes ||
            ds4_gpu_tensor_bytes(low) < low_bytes ||
            ds4_gpu_tensor_bytes(out) < out_bytes) {
            fprintf(stderr, "ds4: Metal attention output batch received undersized buffers\n");
            return 0;
        }
        (void)group_tmp;
        (void)low_tmp;

        const bool use_direct_low =
            n_tokens < 32u && getenv("DS4_METAL_DISABLE_ATTN_OUT_LOW_DIRECT") == NULL;
        /* The exported TensorOps attention-output kernel is a 64-token tile.
         * Keep this on full tiles only; smaller multiples of 32 use the legacy
         * path instead of relying on cooperative tensor partial RHS bounds. */
        const bool use_mpp_low =
            n_tokens >= 32u &&
            (n_tokens % DS4_METAL_ATTN_OUT_MPP_TILE_N) == 0 &&
            ds4_gpu_use_mpp_attn_out_low_matmul();
        const NSUInteger ids_bytes = (NSUInteger)n_tokens * (NSUInteger)n_groups * sizeof(int32_t);
        id<MTLBuffer> group_ids_buffer = nil;
        if (!use_direct_low && !use_mpp_low) {
            if (getenv("DS4_METAL_DISABLE_ATTN_OUT_IDS_CACHE") != NULL) {
                group_ids_buffer =
                    ds4_gpu_new_transient_buffer(ids_bytes, "attention output group ids");
                if (!group_ids_buffer) {
                    return 0;
                }
            } else {
                if (!ds4_gpu_ensure_scratch_buffer(&g_attn_out_group_ids_buffer,
                                                     &g_attn_out_group_ids_bytes,
                                                     ids_bytes,
                                                     "ds4_attention_output_group_ids")) {
                    return 0;
                }
                group_ids_buffer = g_attn_out_group_ids_buffer;
            }
            int32_t *ids = (int32_t *)[group_ids_buffer contents];
            for (uint32_t t = 0; t < n_tokens; t++) {
                for (uint32_t group = 0; group < n_groups; group++) {
                    ids[(uint64_t)t * n_groups + group] = (int32_t)group;
                }
            }
        }

        uint64_t out_a_inner = 0;
        id<MTLBuffer> out_a_buf =
            ds4_gpu_wrap_model_range(model_map, model_size,
                                       out_a_offset, out_a_bytes,
                                       &out_a_inner);
        if (!out_a_buf) return 0;

        const bool had_batch = g_batch_cb != nil;
        if (!had_batch && ds4_gpu_begin_commands() == 0) return 0;

        bool ok = true;
        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb || owned) {
            ok = false;
        }
        const bool attn_out_profile =
            getenv("DS4_METAL_ATTN_OUT_STAGE_PROFILE") != NULL && g_batch_cb != nil;
        if (ok && attn_out_profile) {
            if (ds4_gpu_end_commands() == 0 || ds4_gpu_begin_commands() == 0) {
                ok = false;
            } else {
                cb = ds4_gpu_command_buffer(&owned);
                if (!cb || owned) ok = false;
            }
        }
        double attn_out_t0 = attn_out_profile ? ds4_gpu_now_ms() : 0.0;
#define DS4_METAL_PROFILE_ATTN_OUT_STAGE(name) do { \
            if (ok && attn_out_profile) { \
                if (ds4_gpu_end_commands() == 0) { \
                    ok = false; \
                } else { \
                    const double now_ms = ds4_gpu_now_ms(); \
                    fprintf(stderr, \
                            "ds4: Metal attention output stage tokens=%u %s=%.3f ms\n", \
                            n_tokens, (name), now_ms - attn_out_t0); \
                    attn_out_t0 = now_ms; \
                    if (ds4_gpu_begin_commands() == 0) { \
                        ok = false; \
                    } else { \
                        cb = ds4_gpu_command_buffer(&owned); \
                        if (!cb || owned) ok = false; \
                    } \
                } \
            } \
        } while (0)

        if (ok) {
            /*
             * Batched attention-output projections switch from the vector
             * kernel to the SIMD matrix kernel once the batch has at least 32
             * tokens.  This preserves the single-token generation path while
             * keeping prefill accumulation stable.
             */
            if (use_mpp_low) {
                ds4_gpu_mul_mm_id_args mm_args =
                    ds4_gpu_make_mul_mm_id_args((uint32_t)group_dim,
                                                  (uint32_t)rank,
                                                  n_groups,
                                                  row_a_bytes,
                                                  (uint64_t)rank * row_a_bytes,
                                                  n_groups,
                                                  n_groups,
                                                  n_tokens);
                /*
                 * Direct RHS lets MPP read the dense low-rank activation tile
                 * directly from device memory instead of staging a second
                 * threadgroup tile.  The retained attention-output path is the
                 * 64-token direct-RHS kernel; the older staged-RHS and 32-token
                 * variants were not kept as alternate runtime modes.
                 */
                const char *attn_out_pipeline_name =
                    "kernel_attn_out_low_q8_0_mpp_direct_rhs_n64";
                id<MTLComputePipelineState> mm_pipeline =
                    ds4_gpu_get_mul_mm_id_pipeline(attn_out_pipeline_name, false);
                ok = ds4_gpu_encode_attn_out_low_q8_mpp(cb,
                                                          mm_pipeline,
                                                          &mm_args,
                                                          out_a_buf,
                                                          (NSUInteger)out_a_inner,
                                                          ds4_gpu_tensor_buffer(heads),
                                                          ds4_gpu_tensor_offset(heads),
                                                          ds4_gpu_tensor_buffer(low),
                                                          ds4_gpu_tensor_offset(low)) != 0;
                if (!ok) {
                    ds4_gpu_warn_mpp_fallback();
                    if (ds4_gpu_mul_mm_id_map0_name(n_groups) != NULL) {
                        if (getenv("DS4_METAL_DISABLE_ATTN_OUT_IDS_CACHE") != NULL) {
                            group_ids_buffer =
                                ds4_gpu_new_transient_buffer(ids_bytes, "attention output group ids");
                        } else if (ds4_gpu_ensure_scratch_buffer(&g_attn_out_group_ids_buffer,
                                                                   &g_attn_out_group_ids_bytes,
                                                                   ids_bytes,
                                                                   "ds4_attention_output_group_ids")) {
                            group_ids_buffer = g_attn_out_group_ids_buffer;
                        }
                        if (group_ids_buffer) {
                            int32_t *ids = (int32_t *)[group_ids_buffer contents];
                            for (uint32_t t = 0; t < n_tokens; t++) {
                                for (uint32_t group = 0; group < n_groups; group++) {
                                    ids[(uint64_t)t * n_groups + group] = (int32_t)group;
                                }
                            }
                            ds4_gpu_mul_mm_id_map_args map_args =
                                ds4_gpu_make_mul_mm_id_map_args((uint32_t)group_dim,
                                                                  n_groups,
                                                                  n_groups,
                                                                  n_groups,
                                                                  n_tokens);
                            id<MTLComputePipelineState> map_pipeline =
                                ds4_gpu_get_pipeline(ds4_gpu_mul_mm_id_map0_name(n_groups));
                            id<MTLComputePipelineState> fallback_pipeline =
                                ds4_gpu_get_mul_mm_id_pipeline("kernel_mul_mm_id_q8_0_f32", false);
                            ok = ds4_gpu_encode_mul_mm_id(cb,
                                                            map_pipeline,
                                                            fallback_pipeline,
                                                            &map_args,
                                                            &mm_args,
                                                            out_a_buf,
                                                            (NSUInteger)out_a_inner,
                                                            ds4_gpu_tensor_buffer(heads),
                                                            ds4_gpu_tensor_offset(heads),
                                                            ds4_gpu_tensor_buffer(low),
                                                            ds4_gpu_tensor_offset(low),
                                                            group_ids_buffer,
                                                            0) != 0;
                        }
                    }
                }
            } else if (n_tokens >= 32u && ds4_gpu_mul_mm_id_map0_name(n_groups) != NULL) {
                ds4_gpu_mul_mm_id_map_args map_args =
                    ds4_gpu_make_mul_mm_id_map_args((uint32_t)group_dim,
                                                      n_groups,
                                                      n_groups,
                                                      n_groups,
                                                      n_tokens);
                ds4_gpu_mul_mm_id_args mm_args =
                    ds4_gpu_make_mul_mm_id_args((uint32_t)group_dim,
                                                  (uint32_t)rank,
                                                  n_groups,
                                                  row_a_bytes,
                                                  (uint64_t)rank * row_a_bytes,
                                                  n_groups,
                                                  n_groups,
                                                  n_tokens);
                id<MTLComputePipelineState> map_pipeline =
                    ds4_gpu_get_pipeline(ds4_gpu_mul_mm_id_map0_name(n_groups));
                id<MTLComputePipelineState> mm_pipeline =
                    ds4_gpu_get_mul_mm_id_pipeline("kernel_mul_mm_id_q8_0_f32", false);
                ok = ds4_gpu_encode_mul_mm_id(cb,
                                                map_pipeline,
                                                mm_pipeline,
                                                &map_args,
                                                &mm_args,
                                                out_a_buf,
                                                (NSUInteger)out_a_inner,
                                                ds4_gpu_tensor_buffer(heads),
                                                ds4_gpu_tensor_offset(heads),
                                                ds4_gpu_tensor_buffer(low),
                                                ds4_gpu_tensor_offset(low),
                                                group_ids_buffer,
                                                0) != 0;
            } else if (use_direct_low) {
                ds4_gpu_mul_mv_id_args args = {
                    .nei0 = (int32_t)n_groups,
                    .nei1 = (int32_t)n_tokens,
                    .nbi1 = 0,
                    .ne00 = (int32_t)group_dim,
                    .ne01 = (int32_t)rank,
                    .ne02 = (int32_t)n_groups,
                    .nb00 = 34,
                    .nb01 = row_a_bytes,
                    .nb02 = (uint64_t)rank * row_a_bytes,
                    .ne10 = (int32_t)group_dim,
                    .ne11 = (int32_t)n_groups,
                    .ne12 = (int32_t)n_tokens,
                    .ne13 = 1,
                    .nb10 = sizeof(float),
                    .nb11 = (uint64_t)group_dim * sizeof(float),
                    .nb12 = (uint64_t)n_groups * group_dim * sizeof(float),
                    .ne0 = (int32_t)rank,
                    .ne1 = (int32_t)n_groups,
                    .nb1 = (uint64_t)rank * sizeof(float),
                    .nr0 = 2,
                };
                id<MTLComputePipelineState> pipeline =
                    ds4_gpu_get_mul_mv_pipeline("kernel_dsv4_attn_out_low_q8_0_f32", 4);
                ok = ds4_gpu_encode_attn_out_low_q8_direct(cb,
                                                             pipeline,
                                                             &args,
                                                             out_a_buf,
                                                             (NSUInteger)out_a_inner,
                                                             ds4_gpu_tensor_buffer(heads),
                                                             ds4_gpu_tensor_offset(heads),
                                                             ds4_gpu_tensor_buffer(low),
                                                             ds4_gpu_tensor_offset(low),
                                                             32u * 2u * sizeof(float),
                                                             4,
                                                             true) != 0;
            } else {
                ds4_gpu_mul_mv_id_args args = {
                    .nei0 = (int32_t)n_groups,
                    .nei1 = (int32_t)n_tokens,
                    .nbi1 = (uint64_t)n_groups * sizeof(int32_t),
                    .ne00 = (int32_t)group_dim,
                    .ne01 = (int32_t)rank,
                    .ne02 = (int32_t)n_groups,
                    .nb00 = 34,
                    .nb01 = row_a_bytes,
                    .nb02 = (uint64_t)rank * row_a_bytes,
                    .ne10 = (int32_t)group_dim,
                    .ne11 = (int32_t)n_groups,
                    .ne12 = (int32_t)n_tokens,
                    .ne13 = 1,
                    .nb10 = sizeof(float),
                    .nb11 = (uint64_t)group_dim * sizeof(float),
                    .nb12 = (uint64_t)n_groups * group_dim * sizeof(float),
                    .ne0 = (int32_t)rank,
                    .ne1 = (int32_t)n_groups,
                    .nb1 = (uint64_t)rank * sizeof(float),
                    .nr0 = 2,
                };
                id<MTLComputePipelineState> pipeline =
                    ds4_gpu_get_mul_mv_pipeline("kernel_mul_mv_id_q8_0_f32", 4);
                ok = ds4_gpu_encode_mul_mv_id(cb,
                                                pipeline,
                                                &args,
                                                out_a_buf,
                                                (NSUInteger)out_a_inner,
                                                ds4_gpu_tensor_buffer(heads),
                                                ds4_gpu_tensor_offset(heads),
                                                ds4_gpu_tensor_buffer(low),
                                                ds4_gpu_tensor_offset(low),
                                                group_ids_buffer,
                                                0,
                                                32u * 2u * sizeof(float),
                                                4,
                                                true) != 0;
            }
        }
        DS4_METAL_PROFILE_ATTN_OUT_STAGE("low_proj");

        if (ok) {
            ok = ds4_gpu_matmul_q8_0_tensor(out, model_map, model_size,
                                              out_b_offset,
                                              low_dim, out_dim, low, n_tokens) != 0;
        }
        DS4_METAL_PROFILE_ATTN_OUT_STAGE("out_proj");

        if (!had_batch) {
            ok = ds4_gpu_end_commands() != 0 && ok;
        }
#undef DS4_METAL_PROFILE_ATTN_OUT_STAGE
        return ok ? 1 : 0;
    }
}

int ds4_gpu_attention_output_q4_K_batch_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *low,
        ds4_gpu_tensor       *group_tmp,
        ds4_gpu_tensor       *low_tmp,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                out_a_offset,
        uint64_t                out_b_offset,
        uint32_t                out_b_type,
        uint64_t                group_dim,
        uint64_t                rank,
        uint32_t                n_groups,
        uint64_t                out_dim,
        const ds4_gpu_tensor *heads,
        uint32_t                n_tokens) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!out || !low || !heads || !model_map ||
        group_dim == 0 || rank == 0 || n_groups == 0 || out_dim == 0 || n_tokens == 0 ||
        group_dim > UINT32_MAX || rank > UINT32_MAX || out_dim > UINT32_MAX) {
        return 0;
    }
    if (n_tokens < 32u) return 0;

    @autoreleasepool {
        const uint64_t low_dim = (uint64_t)n_groups * rank;
        if ((group_dim % 256u) != 0 || (low_dim % 256u) != 0 || low_dim > UINT32_MAX) {
            return 0;
        }

        uint64_t row_a_bytes = 0;
        uint64_t row_b_bytes = 0;
        if (!ds4_gpu_quant_row_bytes(DS4_METAL_TENSOR_Q4_K, (uint32_t)group_dim, &row_a_bytes) ||
            !ds4_gpu_quant_row_bytes(out_b_type, (uint32_t)low_dim, &row_b_bytes)) {
            return 0;
        }

        const uint64_t out_a_bytes = (uint64_t)n_groups * rank * row_a_bytes;
        const uint64_t out_b_bytes = out_dim * row_b_bytes;
        if (out_a_offset > model_size || out_a_bytes > model_size - out_a_offset ||
            out_b_offset > model_size || out_b_bytes > model_size - out_b_offset) {
            fprintf(stderr, "ds4: Metal Q4 attention output batch weights are outside the mapped model\n");
            return 0;
        }

        const uint64_t heads_bytes = (uint64_t)n_tokens * n_groups * group_dim * sizeof(float);
        const uint64_t low_bytes = (uint64_t)n_tokens * low_dim * sizeof(float);
        const uint64_t out_bytes = (uint64_t)n_tokens * out_dim * sizeof(float);
        if (ds4_gpu_tensor_bytes(heads) < heads_bytes ||
            ds4_gpu_tensor_bytes(low) < low_bytes ||
            ds4_gpu_tensor_bytes(out) < out_bytes) {
            fprintf(stderr, "ds4: Metal Q4 attention output batch received undersized buffers\n");
            return 0;
        }
        (void)group_tmp;
        (void)low_tmp;

        const NSUInteger ids_bytes = (NSUInteger)n_tokens * (NSUInteger)n_groups * sizeof(int32_t);
        id<MTLBuffer> group_ids_buffer = nil;
        if (getenv("DS4_METAL_DISABLE_ATTN_OUT_IDS_CACHE") != NULL) {
            group_ids_buffer =
                ds4_gpu_new_transient_buffer(ids_bytes, "attention output Q4 group ids");
        } else if (ds4_gpu_ensure_scratch_buffer(&g_attn_out_group_ids_buffer,
                                                 &g_attn_out_group_ids_bytes,
                                                 ids_bytes,
                                                 "ds4_attention_output_group_ids")) {
            group_ids_buffer = g_attn_out_group_ids_buffer;
        }
        if (!group_ids_buffer) return 0;

        int32_t *ids = (int32_t *)[group_ids_buffer contents];
        for (uint32_t t = 0; t < n_tokens; t++) {
            for (uint32_t group = 0; group < n_groups; group++) {
                ids[(uint64_t)t * n_groups + group] = (int32_t)group;
            }
        }

        uint64_t out_a_inner = 0;
        id<MTLBuffer> out_a_buf =
            ds4_gpu_wrap_model_range(model_map, model_size,
                                     out_a_offset, out_a_bytes,
                                     &out_a_inner);
        if (!out_a_buf) return 0;

        const bool had_batch = g_batch_cb != nil;
        if (!had_batch && ds4_gpu_begin_commands() == 0) return 0;

        bool ok = true;
        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb || owned) ok = false;

        if (ok) {
            ds4_gpu_mul_mm_id_map_args map_args =
                ds4_gpu_make_mul_mm_id_map_args((uint32_t)group_dim,
                                                n_groups,
                                                n_groups,
                                                n_groups,
                                                n_tokens);
            ds4_gpu_mul_mm_id_args mm_args =
                ds4_gpu_make_mul_mm_id_args((uint32_t)group_dim,
                                            (uint32_t)rank,
                                            n_groups,
                                            row_a_bytes,
                                            (uint64_t)rank * row_a_bytes,
                                            n_groups,
                                            n_groups,
                                            n_tokens);
            id<MTLComputePipelineState> map_pipeline =
                ds4_gpu_get_pipeline(ds4_gpu_mul_mm_id_map0_name(n_groups));
            id<MTLComputePipelineState> mm_pipeline =
                ds4_gpu_get_mul_mm_id_pipeline("kernel_mul_mm_id_q4_K_f32", false);
            ok = ds4_gpu_encode_mul_mm_id(cb,
                                          map_pipeline,
                                          mm_pipeline,
                                          &map_args,
                                          &mm_args,
                                          out_a_buf,
                                          (NSUInteger)out_a_inner,
                                          ds4_gpu_tensor_buffer(heads),
                                          ds4_gpu_tensor_offset(heads),
                                          ds4_gpu_tensor_buffer(low),
                                          ds4_gpu_tensor_offset(low),
                                          group_ids_buffer,
                                          0) != 0;
        }

        if (ok) {
            ok = ds4_gpu_matmul_quant_tensor(out,
                                             model_map,
                                             model_size,
                                             out_b_offset,
                                             out_b_type,
                                             low_dim,
                                             out_dim,
                                             low,
                                             n_tokens) != 0;
        }

        if (!had_batch) {
            ok = ds4_gpu_end_commands() != 0 && ok;
        }
        return ok ? 1 : 0;
    }
}

int ds4_gpu_attention_output_q8_batch_f16_tensor(
        ds4_gpu_tensor       *out_h,
        ds4_gpu_tensor       *low,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                out_a_offset,
        uint64_t                out_b_offset,
        uint64_t                group_dim,
        uint64_t                rank,
        uint32_t                n_groups,
        uint64_t                out_dim,
        const ds4_gpu_tensor *heads,
        uint32_t                n_tokens) {
    (void)out_h; (void)low; (void)model_map; (void)model_size;
    (void)out_a_offset; (void)out_b_offset; (void)group_dim; (void)rank;
    (void)n_groups; (void)out_dim; (void)heads; (void)n_tokens;
    return 0;
}

int ds4_gpu_matmul_q8_0_kslice_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                full_in_dim,
        uint64_t                k_off,
        uint64_t                k_cnt,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                x_elem_off) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if ((full_in_dim & 31u) != 0 || (k_off & 31u) != 0 || (k_cnt & 31u) != 0 ||
        k_cnt == 0 || k_off + k_cnt > full_in_dim ||
        full_in_dim > UINT32_MAX || out_dim > UINT32_MAX) {
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> xbuf = ds4_gpu_tensor_buffer(x);
        id<MTLBuffer> outbuf = ds4_gpu_tensor_buffer(out);
        if (!xbuf || !outbuf ||
            ds4_gpu_tensor_bytes(x) < (x_elem_off + k_cnt) * sizeof(float) ||
            ds4_gpu_tensor_bytes(out) < out_dim * sizeof(float)) {
            fprintf(stderr, "ds4: Metal Q8_0 kslice matmul received undersized buffers\n");
            return 0;
        }
        const uint64_t row_bytes = (full_in_dim / 32u) * 34u;
        const uint64_t weight_bytes = out_dim * row_bytes;
        if (weight_offset > model_size || weight_bytes > model_size - weight_offset) {
            fprintf(stderr, "ds4: Metal Q8_0 kslice weights are outside the mapped model\n");
            return 0;
        }
        uint64_t inner = 0;
        id<MTLBuffer> wbuf = ds4_gpu_wrap_model_range(model_map, model_size,
                                                        weight_offset, weight_bytes, &inner);
        if (!wbuf) return 0;

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        /* Same matvec kernel as the full projection: ne00 bounds the k loop
         * while nb01/nb02 keep the full-row stride, so each row reads only
         * the owned k window. */
        ds4_gpu_q8_0_matvec_args mv_args = ds4_gpu_make_q8_0_mv_args(full_in_dim, out_dim);
        mv_args.ne00 = (int32_t)k_cnt;
        mv_args.ne10 = (int32_t)k_cnt;
        ds4_gpu_mv_dispatch mv_dispatch = ds4_gpu_make_q8_0_mv_dispatch();
        if (out_dim > 65536u) mv_dispatch.nsg = 8;
        mv_args.nr0 = mv_dispatch.nr0;
        id<MTLComputePipelineState> pipeline =
            ds4_gpu_get_mul_mv_pipeline(mv_dispatch.function_name, mv_dispatch.nsg);
        if (!pipeline) return 0;

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&mv_args length:sizeof(mv_args) atIndex:0];
        [enc setBuffer:wbuf offset:(NSUInteger)(inner + (k_off / 32u) * 34u) atIndex:1];
        [enc setBuffer:xbuf
                offset:(NSUInteger)(ds4_gpu_tensor_offset(x) + x_elem_off * sizeof(float))
               atIndex:2];
        [enc setBuffer:outbuf offset:ds4_gpu_tensor_offset(out) atIndex:3];
        [enc setThreadgroupMemoryLength:mv_dispatch.smem atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)out_dim + (NSUInteger)mv_dispatch.nr0 - 1u) / (NSUInteger)mv_dispatch.nr0,
                                              1,
                                              1)
             threadsPerThreadgroup:MTLSizeMake(32, (NSUInteger)mv_dispatch.nsg, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "Q8_0 kslice matvec")) {
            return 0;
        }
        return 1;
    }
}

int ds4_gpu_matmul_quant_kslice_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                weight_type,
        uint64_t                full_in_dim,
        uint64_t                k_off,
        uint64_t                k_cnt,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                x_elem_off) {
    if (weight_type == DS4_METAL_TENSOR_Q8_0) {
        return ds4_gpu_matmul_q8_0_kslice_tensor(out,
                                                 model_map,
                                                 model_size,
                                                 weight_offset,
                                                 full_in_dim,
                                                 k_off,
                                                 k_cnt,
                                                 out_dim,
                                                 x,
                                                 x_elem_off);
    }
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!out || !x || !model_map ||
        full_in_dim == 0 || k_cnt == 0 || out_dim == 0 ||
        k_off + k_cnt > full_in_dim ||
        full_in_dim > UINT32_MAX || k_cnt > UINT32_MAX ||
        out_dim > UINT32_MAX) {
        return 0;
    }

    uint64_t block_elems = 0;
    uint64_t block_bytes = 0;
    if (weight_type == DS4_METAL_TENSOR_Q4_K) {
        block_elems = 256u;
        block_bytes = 144u;
    } else if (weight_type == DS4_METAL_TENSOR_Q4_0) {
        block_elems = 32u;
        block_bytes = 18u;
    } else {
        fprintf(stderr, "ds4: Metal quant kslice received unsupported type %u\n", weight_type);
        return 0;
    }
    if ((full_in_dim % block_elems) != 0 ||
        (k_off % block_elems) != 0 ||
        (k_cnt % block_elems) != 0) {
        fprintf(stderr, "ds4: Metal quant kslice dimensions are not block aligned\n");
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> xbuf = ds4_gpu_tensor_buffer(x);
        id<MTLBuffer> outbuf = ds4_gpu_tensor_buffer(out);
        if (!xbuf || !outbuf ||
            ds4_gpu_tensor_bytes(x) < (x_elem_off + k_cnt) * sizeof(float) ||
            ds4_gpu_tensor_bytes(out) < out_dim * sizeof(float)) {
            fprintf(stderr, "ds4: Metal quant kslice matmul received undersized buffers\n");
            return 0;
        }

        const uint64_t row_bytes = (full_in_dim / block_elems) * block_bytes;
        const uint64_t weight_bytes = out_dim * row_bytes;
        if (weight_offset > model_size || weight_bytes > model_size - weight_offset) {
            fprintf(stderr, "ds4: Metal quant kslice weights are outside the mapped model\n");
            return 0;
        }
        uint64_t inner = 0;
        id<MTLBuffer> wbuf = ds4_gpu_wrap_model_range(model_map, model_size,
                                                        weight_offset, weight_bytes, &inner);
        if (!wbuf) return 0;

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        if (weight_type == DS4_METAL_TENSOR_Q4_K &&
            (k_cnt % 256u) == 0 &&
            getenv("DS4_METAL_DISABLE_Q4_MV_CLASSIC") == NULL) {
            const int16_t nsg = 2;
            id<MTLComputePipelineState> pipeline =
                ds4_gpu_get_mul_mv_ext_pipeline("kernel_mul_mv_q4_K_dense_f32", nsg, 8);
            if (pipeline) {
                ds4_gpu_q8_0_matvec_args args = {
                    .ne00 = (int32_t)k_cnt,
                    .ne01 = (int32_t)out_dim,
                    .ne02 = 1,
                    .nb00 = 1,
                    .nb01 = row_bytes,
                    .nb02 = row_bytes * out_dim,
                    .nb03 = row_bytes * out_dim,
                    .ne10 = (int32_t)k_cnt,
                    .ne11 = 1,
                    .ne12 = 1,
                    .nb10 = sizeof(float),
                    .nb11 = k_cnt * sizeof(float),
                    .nb12 = k_cnt * sizeof(float),
                    .nb13 = k_cnt * sizeof(float),
                    .ne0 = (int32_t)out_dim,
                    .ne1 = 1,
                    .nr0 = 2,
                    .r2 = 1,
                    .r3 = 1,
                };
                const uint64_t rows_ptg = (uint64_t)nsg * 2u;
                const uint64_t w_skip = (k_off / block_elems) * block_bytes;

                id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
                [enc setComputePipelineState:pipeline];
                [enc setBytes:&args length:sizeof(args) atIndex:0];
                [enc setBuffer:wbuf offset:(NSUInteger)(inner + w_skip) atIndex:1];
                [enc setBuffer:xbuf
                        offset:(NSUInteger)(ds4_gpu_tensor_offset(x) + x_elem_off * sizeof(float))
                       atIndex:2];
                [enc setBuffer:outbuf offset:ds4_gpu_tensor_offset(out) atIndex:3];
                [enc setThreadgroupMemoryLength:32 atIndex:0];
                [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)out_dim + rows_ptg - 1u) / rows_ptg,
                                                      1,
                                                      1)
                     threadsPerThreadgroup:MTLSizeMake(32, (NSUInteger)nsg, 1)];
                ds4_gpu_end_compute_encoder(cb, enc);

                if (!ds4_gpu_finish_command_buffer(cb, owned, "Q4_K kslice matvec")) return 0;
                return 1;
            }
        }

        fprintf(stderr, "ds4: Metal quant kslice has no kernel for type %u\n",
                weight_type);
        return 0;
    }
}

int ds4_gpu_attention_output_q8_tp_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *low,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                out_a_offset,
        uint64_t                out_b_offset,
        uint64_t                group_dim,
        uint64_t                rank,
        uint32_t                n_groups_total,
        uint32_t                group0,
        uint32_t                group_cnt,
        uint64_t                out_dim,
        const ds4_gpu_tensor *heads) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!out || !low || !heads || !model_map ||
        group_dim == 0 || rank == 0 || group_cnt == 0 ||
        group0 + group_cnt > n_groups_total ||
        (group_dim % 32u) != 0 || ((rank * group_cnt) % 32u) != 0 ||
        group_dim > UINT32_MAX || rank > UINT32_MAX || out_dim > UINT32_MAX) {
        return 0;
    }

    @autoreleasepool {
        const uint64_t low_dim_total = (uint64_t)n_groups_total * rank;
        const uint64_t row_a_bytes = (group_dim / 32u) * 34u;
        const uint64_t a_group_bytes = rank * row_a_bytes;
        const uint64_t out_a_bytes = (uint64_t)n_groups_total * a_group_bytes;
        if (out_a_offset > model_size || out_a_bytes > model_size - out_a_offset) {
            fprintf(stderr, "ds4: Metal TP attention output weights are outside the mapped model\n");
            return 0;
        }
        /* The heads buffer holds only the owned groups, compact at its
         * base (the head slice keeps q/attention output halves packed). */
        if (ds4_gpu_tensor_bytes(heads) < (uint64_t)group_cnt * group_dim * sizeof(float) ||
            ds4_gpu_tensor_bytes(low) < (uint64_t)group_cnt * rank * sizeof(float) ||
            ds4_gpu_tensor_bytes(out) < out_dim * sizeof(float)) {
            fprintf(stderr, "ds4: Metal TP attention output received undersized buffers\n");
            return 0;
        }

        uint64_t out_a_inner = 0;
        id<MTLBuffer> out_a_buf =
            ds4_gpu_wrap_model_range(model_map, model_size,
                                       out_a_offset, out_a_bytes, &out_a_inner);
        if (!out_a_buf) return 0;

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb || owned) return 0;

        /* Low projection for the owned groups only: identical dispatch to
         * the single-node direct path with the weight base, heads input and
         * group count shifted to the slice.  The owned low half lands
         * compactly at low[0 .. group_cnt*rank). */
        ds4_gpu_mul_mv_id_args args = {
            .nei0 = (int32_t)group_cnt,
            .nei1 = 1,
            .nbi1 = 0,
            .ne00 = (int32_t)group_dim,
            .ne01 = (int32_t)rank,
            .ne02 = (int32_t)group_cnt,
            .nb00 = 34,
            .nb01 = row_a_bytes,
            .nb02 = a_group_bytes,
            .ne10 = (int32_t)group_dim,
            .ne11 = (int32_t)group_cnt,
            .ne12 = 1,
            .ne13 = 1,
            .nb10 = sizeof(float),
            .nb11 = (uint64_t)group_dim * sizeof(float),
            .nb12 = (uint64_t)group_cnt * group_dim * sizeof(float),
            .ne0 = (int32_t)rank,
            .ne1 = (int32_t)group_cnt,
            .nb1 = (uint64_t)rank * sizeof(float),
            .nr0 = 2,
        };
        id<MTLComputePipelineState> pipeline =
            ds4_gpu_get_mul_mv_pipeline("kernel_dsv4_attn_out_low_q8_0_f32", 4);
        int ok = ds4_gpu_encode_attn_out_low_q8_direct(cb,
                pipeline,
                &args,
                out_a_buf,
                (NSUInteger)(out_a_inner + (uint64_t)group0 * a_group_bytes),
                ds4_gpu_tensor_buffer(heads),
                ds4_gpu_tensor_offset(heads),
                ds4_gpu_tensor_buffer(low),
                ds4_gpu_tensor_offset(low),
                32u * 2u * sizeof(float),
                4,
                true);
        if (!ok) return 0;

        /* Expand projection over the owned k window only; the result is this
         * rank's partial attention block output. */
        return ds4_gpu_matmul_q8_0_kslice_tensor(out, model_map, model_size,
                                                   out_b_offset,
                                                   low_dim_total,
                                                   (uint64_t)group0 * rank,
                                                   (uint64_t)group_cnt * rank,
                                                   out_dim, low, 0);
    }
}

int ds4_gpu_attention_output_low_q8_tensor(
        ds4_gpu_tensor       *low,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                out_a_offset,
        uint64_t                group_dim,
        uint64_t                rank,
        uint32_t                n_groups,
        const ds4_gpu_tensor *heads) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!low || !heads || !model_map || group_dim == 0 || rank == 0 ||
        n_groups == 0 || group_dim > UINT32_MAX || rank > UINT32_MAX) {
        return 0;
    }

    @autoreleasepool {
        const uint64_t low_dim = (uint64_t)n_groups * rank;
        if ((group_dim % 32u) != 0 || low_dim > UINT32_MAX) {
            fprintf(stderr, "ds4: Metal attention output low received invalid q8 dimensions\n");
            return 0;
        }

        const uint64_t row_a_bytes = (group_dim / 32u) * 34u;
        const uint64_t out_a_bytes = (uint64_t)n_groups * rank * row_a_bytes;
        if (out_a_offset > model_size || out_a_bytes > model_size - out_a_offset) {
            fprintf(stderr, "ds4: Metal attention output low weights are outside the mapped model\n");
            return 0;
        }

        const uint64_t heads_bytes = (uint64_t)n_groups * group_dim * sizeof(float);
        const uint64_t low_bytes = low_dim * sizeof(float);
        if (ds4_gpu_tensor_bytes(heads) < heads_bytes ||
            ds4_gpu_tensor_bytes(low) < low_bytes) {
            fprintf(stderr, "ds4: Metal attention output low received undersized buffers\n");
            return 0;
        }

        uint64_t out_a_inner = 0;
        id<MTLBuffer> out_a_buf =
            ds4_gpu_wrap_model_range(model_map, model_size,
                                       out_a_offset, out_a_bytes,
                                       &out_a_inner);
        if (!out_a_buf) return 0;

        const bool had_batch = g_batch_cb != nil;
        if (!had_batch && ds4_gpu_begin_commands() == 0) return 0;

        bool ok = true;
        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb || owned) {
            ok = false;
        }

        if (ok) {
            ds4_gpu_mul_mv_id_args args = {
                .nei0 = (int32_t)n_groups,
                .nei1 = 1,
                .nbi1 = 0,
                .ne00 = (int32_t)group_dim,
                .ne01 = (int32_t)rank,
                .ne02 = (int32_t)n_groups,
                .nb00 = 34,
                .nb01 = row_a_bytes,
                .nb02 = (uint64_t)rank * row_a_bytes,
                .ne10 = (int32_t)group_dim,
                .ne11 = (int32_t)n_groups,
                .ne12 = 1,
                .ne13 = 1,
                .nb10 = sizeof(float),
                .nb11 = (uint64_t)group_dim * sizeof(float),
                .nb12 = (uint64_t)n_groups * group_dim * sizeof(float),
                .ne0 = (int32_t)rank,
                .ne1 = (int32_t)n_groups,
                .nb1 = (uint64_t)rank * sizeof(float),
                .nr0 = 2,
            };
            id<MTLComputePipelineState> pipeline =
                ds4_gpu_get_mul_mv_pipeline("kernel_dsv4_attn_out_low_q8_0_f32", 4);
            ok = ds4_gpu_encode_attn_out_low_q8_direct(cb,
                                                         pipeline,
                                                         &args,
                                                         out_a_buf,
                                                         (NSUInteger)out_a_inner,
                                                         ds4_gpu_tensor_buffer(heads),
                                                         ds4_gpu_tensor_offset(heads),
                                                         ds4_gpu_tensor_buffer(low),
                                                         ds4_gpu_tensor_offset(low),
                                                         32u * 2u * sizeof(float),
                                                         4,
                                                         true) != 0;
        }

        if (!had_batch) {
            ok = ds4_gpu_end_commands() != 0 && ok;
        }
        return ok ? 1 : 0;
    }
}

int ds4_gpu_attention_output_low_q4_K_slice_tensor(
        ds4_gpu_tensor       *low,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                out_a_offset,
        uint64_t                group_dim,
        uint64_t                rank,
        uint32_t                group0,
        uint32_t                group_cnt,
        const ds4_gpu_tensor *heads) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!low || !heads || !model_map || group_dim == 0 || rank == 0 ||
        group_cnt == 0 || group_dim > UINT32_MAX || rank > UINT32_MAX) {
        return 0;
    }

    @autoreleasepool {
        if ((group_dim % 256u) != 0) {
            fprintf(stderr, "ds4: Metal attention output low received invalid Q4_K dimensions\n");
            return 0;
        }

        uint64_t row_a_bytes = 0;
        if (!ds4_gpu_quant_row_bytes(DS4_METAL_TENSOR_Q4_K,
                                     (uint32_t)group_dim,
                                     &row_a_bytes)) {
            return 0;
        }
        if (rank > UINT64_MAX / row_a_bytes) return 0;
        const uint64_t group_weight_bytes = rank * row_a_bytes;
        if (group0 > UINT64_MAX / group_weight_bytes ||
            group_cnt > UINT64_MAX / group_weight_bytes) {
            return 0;
        }
        const uint64_t group_skip = (uint64_t)group0 * group_weight_bytes;
        const uint64_t out_a_bytes = (uint64_t)group_cnt * group_weight_bytes;
        if (out_a_offset > UINT64_MAX - group_skip ||
            out_a_offset + group_skip > model_size ||
            out_a_bytes > model_size - (out_a_offset + group_skip)) {
            fprintf(stderr, "ds4: Metal Q4 attention output low weights are outside the mapped model\n");
            return 0;
        }

        const uint64_t heads_bytes = (uint64_t)group_cnt * group_dim * sizeof(float);
        const uint64_t low_bytes = (uint64_t)group_cnt * rank * sizeof(float);
        if (ds4_gpu_tensor_bytes(heads) < heads_bytes ||
            ds4_gpu_tensor_bytes(low) < low_bytes) {
            fprintf(stderr, "ds4: Metal Q4 attention output low received undersized buffers\n");
            return 0;
        }

        uint64_t out_a_inner = 0;
        id<MTLBuffer> out_a_buf =
            ds4_gpu_wrap_model_range(model_map, model_size,
                                      out_a_offset + group_skip,
                                      out_a_bytes,
                                      &out_a_inner);
        if (!out_a_buf) return 0;

        const bool had_batch = g_batch_cb != nil;
        if (!had_batch && ds4_gpu_begin_commands() == 0) return 0;

        bool ok = true;
        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb || owned) {
            ok = false;
        }

        if (ok) {
            ds4_gpu_mul_mv_id_args args = {
                .nei0 = (int32_t)group_cnt,
                .nei1 = 1,
                .nbi1 = 0,
                .ne00 = (int32_t)group_dim,
                .ne01 = (int32_t)rank,
                .ne02 = (int32_t)group_cnt,
                .nb00 = 1,
                .nb01 = row_a_bytes,
                .nb02 = group_weight_bytes,
                .ne10 = (int32_t)group_dim,
                .ne11 = (int32_t)group_cnt,
                .ne12 = 1,
                .ne13 = 1,
                .nb10 = sizeof(float),
                .nb11 = (uint64_t)group_dim * sizeof(float),
                .nb12 = (uint64_t)group_cnt * group_dim * sizeof(float),
                .ne0 = (int32_t)rank,
                .ne1 = (int32_t)group_cnt,
                .nb1 = (uint64_t)rank * sizeof(float),
                .nr0 = 2,
            };
            const NSUInteger nsg = 2;
            id<MTLComputePipelineState> pipeline =
                ds4_gpu_get_mul_mv_pipeline("kernel_dsv4_attn_out_low_q4_K_f32", (int16_t)nsg);
            ok = ds4_gpu_encode_attn_out_low_q8_direct(cb,
                                                         pipeline,
                                                         &args,
                                                         out_a_buf,
                                                         (NSUInteger)out_a_inner,
                                                         ds4_gpu_tensor_buffer(heads),
                                                         ds4_gpu_tensor_offset(heads),
                                                         ds4_gpu_tensor_buffer(low),
                                                         ds4_gpu_tensor_offset(low),
                                                         32u,
                                                         nsg,
                                                         false) != 0;
        }

        if (!had_batch) {
            ok = ds4_gpu_end_commands() != 0 && ok;
        }
        return ok ? 1 : 0;
    }
}

static NSUInteger ds4_gpu_align_up_ns(NSUInteger value, NSUInteger align) {
    return (value + align - 1u) & ~(align - 1u);
}

static int ds4_gpu_encode_cpy_f32_f32_1d(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        src,
        NSUInteger           src_off,
        id<MTLBuffer>        dst,
        NSUInteger           dst_off,
        uint32_t             n) {
    if (!cb || !src || !dst || n == 0) return 0;

    ds4_gpu_cpy_args args =
        ds4_gpu_make_cpy_1d_args(n, sizeof(float), sizeof(float));
    const NSUInteger nth = ds4_gpu_cpy_threads(n, g_cpy_f32_f32_pipeline);
    const NSUInteger groups = ((NSUInteger)n + nth - 1u) / nth;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:g_cpy_f32_f32_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:src offset:src_off atIndex:1];
    [enc setBuffer:dst offset:dst_off atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);

    return 1;
}

static int ds4_gpu_encode_cpy_f32_f32_3d(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        src,
        NSUInteger           src_off,
        id<MTLBuffer>        dst,
        NSUInteger           dst_off,
        uint32_t             cols,
        uint32_t             rows,
        uint32_t             planes,
        uint64_t             src_row_stride,
        uint64_t             src_plane_stride,
        uint64_t             dst_row_stride,
        uint64_t             dst_plane_stride) {
    if (!cb || !src || !dst || cols == 0 || rows == 0 || planes == 0) return 0;

    ds4_gpu_cpy_args args = {
        .nk0 = (int64_t)cols,
        .ne00 = (int64_t)cols,
        .ne01 = (int64_t)rows,
        .ne02 = (int64_t)planes,
        .ne03 = 1,
        .nb00 = sizeof(float),
        .nb01 = src_row_stride,
        .nb02 = src_plane_stride,
        .nb03 = (uint64_t)planes * src_plane_stride,
        .ne0 = (int64_t)cols,
        .ne1 = (int64_t)rows,
        .ne2 = (int64_t)planes,
        .ne3 = 1,
        .nb0 = sizeof(float),
        .nb1 = dst_row_stride,
        .nb2 = dst_plane_stride,
        .nb3 = (uint64_t)planes * dst_plane_stride,
    };
    const NSUInteger nth = ds4_gpu_cpy_threads(cols, g_cpy_f32_f32_pipeline);
    const NSUInteger col_groups = ((NSUInteger)cols + nth - 1u) / nth;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:g_cpy_f32_f32_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:src offset:src_off atIndex:1];
    [enc setBuffer:dst offset:dst_off atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake(col_groups * rows, planes, 1)
         threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);

    return 1;
}

static int ds4_gpu_encode_cpy_f32_f32_3d_src_strided(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        src,
        NSUInteger           src_off,
        id<MTLBuffer>        dst,
        NSUInteger           dst_off,
        uint32_t             cols,
        uint32_t             rows,
        uint32_t             planes,
        uint64_t             src_col_stride,
        uint64_t             src_row_stride,
        uint64_t             src_plane_stride,
        uint64_t             dst_row_stride,
        uint64_t             dst_plane_stride) {
    if (!cb || !src || !dst || cols == 0 || rows == 0 || planes == 0) return 0;

    ds4_gpu_cpy_args args = {
        .nk0 = (int64_t)cols,
        .ne00 = (int64_t)cols,
        .ne01 = (int64_t)rows,
        .ne02 = (int64_t)planes,
        .ne03 = 1,
        .nb00 = src_col_stride,
        .nb01 = src_row_stride,
        .nb02 = src_plane_stride,
        .nb03 = (uint64_t)planes * src_plane_stride,
        .ne0 = (int64_t)cols,
        .ne1 = (int64_t)rows,
        .ne2 = (int64_t)planes,
        .ne3 = 1,
        .nb0 = sizeof(float),
        .nb1 = dst_row_stride,
        .nb2 = dst_plane_stride,
        .nb3 = (uint64_t)planes * dst_plane_stride,
    };
    const NSUInteger nth = ds4_gpu_cpy_threads(cols, g_cpy_f32_f32_pipeline);
    const NSUInteger col_groups = ((NSUInteger)cols + nth - 1u) / nth;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:g_cpy_f32_f32_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:src offset:src_off atIndex:1];
    [enc setBuffer:dst offset:dst_off atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake(col_groups * rows, planes, 1)
         threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);

    return 1;
}

static int ds4_gpu_encode_cpy_f32_f16_1d(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        src,
        NSUInteger           src_off,
        id<MTLBuffer>        dst,
        NSUInteger           dst_off,
        uint32_t             n) {
    if (!cb || !src || !dst || n == 0) return 0;

    const int use_contiguous =
        ds4_gpu_env_bool("DS4_METAL_DISABLE_CONTIG_F32_F16_COPY") <= 0;
    id<MTLComputePipelineState> pipeline = use_contiguous
        ? g_cpy_contig_f32_f16_pipeline
        : g_cpy_f32_f16_pipeline;
    const NSUInteger work_items = use_contiguous
        ? ((NSUInteger)n + 3u) / 4u
        : (NSUInteger)n;
    const NSUInteger nth = ds4_gpu_cpy_threads((uint32_t)work_items, pipeline);
    const NSUInteger groups = (work_items + nth - 1u) / nth;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:pipeline];
    if (use_contiguous) {
        [enc setBytes:&n length:sizeof(n) atIndex:0];
    } else {
        ds4_gpu_cpy_args args =
            ds4_gpu_make_cpy_1d_args(n, sizeof(float), sizeof(uint16_t));
        [enc setBytes:&args length:sizeof(args) atIndex:0];
    }
    [enc setBuffer:src offset:src_off atIndex:1];
    [enc setBuffer:dst offset:dst_off atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);

    return 1;
}

static int ds4_gpu_encode_cpy_f32_f16_2d(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        src,
        NSUInteger           src_off,
        id<MTLBuffer>        dst,
        NSUInteger           dst_off,
        uint32_t             cols,
        uint32_t             rows,
        uint64_t             src_row_stride,
        uint64_t             dst_row_stride) {
    if (!cb || !src || !dst || cols == 0 || rows == 0) return 0;

    ds4_gpu_cpy_args args = {
        .nk0 = (int64_t)cols,
        .ne00 = (int64_t)cols,
        .ne01 = (int64_t)rows,
        .ne02 = 1,
        .ne03 = 1,
        .nb00 = sizeof(float),
        .nb01 = src_row_stride,
        .nb02 = (uint64_t)rows * src_row_stride,
        .nb03 = (uint64_t)rows * src_row_stride,
        .ne0 = (int64_t)cols,
        .ne1 = (int64_t)rows,
        .ne2 = 1,
        .ne3 = 1,
        .nb0 = sizeof(uint16_t),
        .nb1 = dst_row_stride,
        .nb2 = (uint64_t)rows * dst_row_stride,
        .nb3 = (uint64_t)rows * dst_row_stride,
    };
    const NSUInteger nth = ds4_gpu_cpy_threads(cols, g_cpy_f32_f16_pipeline);
    const NSUInteger col_groups = ((NSUInteger)cols + nth - 1u) / nth;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:g_cpy_f32_f16_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:src offset:src_off atIndex:1];
    [enc setBuffer:dst offset:dst_off atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake(col_groups * rows, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);

    return 1;
}

static int ds4_gpu_encode_cpy_f32_f16_3d(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        src,
        NSUInteger           src_off,
        id<MTLBuffer>        dst,
        NSUInteger           dst_off,
        uint32_t             cols,
        uint32_t             rows,
        uint32_t             planes,
        uint64_t             src_row_stride,
        uint64_t             src_plane_stride,
        uint64_t             dst_row_stride,
        uint64_t             dst_plane_stride) {
    if (!cb || !src || !dst || cols == 0 || rows == 0 || planes == 0) return 0;

    ds4_gpu_cpy_args args = {
        .nk0 = (int64_t)cols,
        .ne00 = (int64_t)cols,
        .ne01 = (int64_t)rows,
        .ne02 = (int64_t)planes,
        .ne03 = 1,
        .nb00 = sizeof(float),
        .nb01 = src_row_stride,
        .nb02 = src_plane_stride,
        .nb03 = (uint64_t)planes * src_plane_stride,
        .ne0 = (int64_t)cols,
        .ne1 = (int64_t)rows,
        .ne2 = (int64_t)planes,
        .ne3 = 1,
        .nb0 = sizeof(uint16_t),
        .nb1 = dst_row_stride,
        .nb2 = dst_plane_stride,
        .nb3 = (uint64_t)planes * dst_plane_stride,
    };
    const NSUInteger nth = ds4_gpu_cpy_threads(cols, g_cpy_f32_f16_pipeline);
    const NSUInteger col_groups = ((NSUInteger)cols + nth - 1u) / nth;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:g_cpy_f32_f16_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:src offset:src_off atIndex:1];
    [enc setBuffer:dst offset:dst_off atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake(col_groups * rows, planes, 1)
         threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);

    return 1;
}

static int ds4_gpu_encode_cpy_f16_f16_3d(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        src,
        NSUInteger           src_off,
        id<MTLBuffer>        dst,
        NSUInteger           dst_off,
        uint32_t             cols,
        uint32_t             rows,
        uint32_t             planes,
        uint64_t             src_row_stride,
        uint64_t             src_plane_stride,
        uint64_t             dst_row_stride,
        uint64_t             dst_plane_stride) {
    if (!cb || !src || !dst || cols == 0 || rows == 0 || planes == 0) return 0;

    ds4_gpu_cpy_args args = {
        .nk0 = (int64_t)cols,
        .ne00 = (int64_t)cols,
        .ne01 = (int64_t)rows,
        .ne02 = (int64_t)planes,
        .ne03 = 1,
        .nb00 = sizeof(uint16_t),
        .nb01 = src_row_stride,
        .nb02 = src_plane_stride,
        .nb03 = (uint64_t)planes * src_plane_stride,
        .ne0 = (int64_t)cols,
        .ne1 = (int64_t)rows,
        .ne2 = (int64_t)planes,
        .ne3 = 1,
        .nb0 = sizeof(uint16_t),
        .nb1 = dst_row_stride,
        .nb2 = dst_plane_stride,
        .nb3 = (uint64_t)planes * dst_plane_stride,
    };
    const NSUInteger nth = ds4_gpu_cpy_threads(cols, g_cpy_f16_f16_pipeline);
    const NSUInteger col_groups = ((NSUInteger)cols + nth - 1u) / nth;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:g_cpy_f16_f16_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:src offset:src_off atIndex:1];
    [enc setBuffer:dst offset:dst_off atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake(col_groups * rows, planes, 1)
         threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);

    return 1;
}

static int ds4_gpu_encode_cpy_f16_f32_1d(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        src,
        NSUInteger           src_off,
        id<MTLBuffer>        dst,
        NSUInteger           dst_off,
        uint32_t             n) {
    if (!cb || !src || !dst || n == 0) return 0;

    const int use_contiguous =
        ds4_gpu_env_bool("DS4_METAL_DISABLE_CONTIG_F32_F16_COPY") <= 0;
    id<MTLComputePipelineState> pipeline = use_contiguous
        ? g_cpy_contig_f16_f32_pipeline
        : g_cpy_f16_f32_pipeline;
    const NSUInteger work_items = use_contiguous
        ? ((NSUInteger)n + 3u) / 4u
        : (NSUInteger)n;
    const NSUInteger nth = ds4_gpu_cpy_threads((uint32_t)work_items, pipeline);
    const NSUInteger groups = (work_items + nth - 1u) / nth;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:pipeline];
    if (use_contiguous) {
        [enc setBytes:&n length:sizeof(n) atIndex:0];
    } else {
        ds4_gpu_cpy_args args =
            ds4_gpu_make_cpy_1d_args(n, sizeof(uint16_t), sizeof(float));
        [enc setBytes:&args length:sizeof(args) atIndex:0];
    }
    [enc setBuffer:src offset:src_off atIndex:1];
    [enc setBuffer:dst offset:dst_off atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);

    return 1;
}

static int ds4_gpu_encode_copy_to_f16_1d(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        src,
        NSUInteger           src_off,
        bool                 src_is_f16,
        id<MTLBuffer>        dst,
        NSUInteger           dst_off,
        uint32_t             n) {
    if (!cb || !src || !dst) return 0;
    if (n == 0) return 1;
    if (!src_is_f16) {
        return ds4_gpu_encode_cpy_f32_f16_1d(cb, src, src_off, dst, dst_off, n);
    }

    if (ds4_gpu_env_bool("DS4_METAL_DISABLE_CONTIG_F16_F16_COPY") <= 0) {
        const NSUInteger work_items = ((NSUInteger)n + 3u) / 4u;
        const NSUInteger nth = ds4_gpu_cpy_threads(
                (uint32_t)work_items,
                g_cpy_contig_f16_f16_pipeline);
        const NSUInteger groups = (work_items + nth - 1u) / nth;

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:g_cpy_contig_f16_f16_pipeline];
        [enc setBytes:&n length:sizeof(n) atIndex:0];
        [enc setBuffer:src offset:src_off atIndex:1];
        [enc setBuffer:dst offset:dst_off atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);
        return 1;
    }

    if (g_batch_cb && cb == g_batch_cb) ds4_gpu_close_batch_encoder();
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    if (!blit) return 0;
    [blit copyFromBuffer:src
            sourceOffset:src_off
                toBuffer:dst
       destinationOffset:dst_off
                    size:(NSUInteger)n * sizeof(uint16_t)];
    [blit endEncoding];
    return 1;
}

static int ds4_gpu_encode_copy_raw_ring_to_f16(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        raw,
        NSUInteger           raw_offset,
        uint32_t             raw_cap,
        uint32_t             raw_start,
        uint32_t             n_raw,
        uint32_t             head_dim,
        id<MTLBuffer>        dst,
        NSUInteger           dst_offset) {
    if (!cb || !raw || !dst || raw_cap == 0 || raw_start >= raw_cap ||
        n_raw == 0 || n_raw > raw_cap || head_dim == 0) {
        return 0;
    }

    const NSUInteger raw_row_bytes = (NSUInteger)head_dim * sizeof(float);
    const NSUInteger dst_row_bytes = (NSUInteger)head_dim * sizeof(uint16_t);
    const uint32_t tail_rows = raw_cap - raw_start < n_raw
        ? raw_cap - raw_start
        : n_raw;
    const uint32_t head_rows = n_raw - tail_rows;
    const uint64_t tail_count = (uint64_t)tail_rows * head_dim;
    const uint64_t head_count = (uint64_t)head_rows * head_dim;
    const uint64_t raw_inner = (uint64_t)raw_start * raw_row_bytes;
    const uint64_t dst_inner = (uint64_t)tail_rows * dst_row_bytes;
    if (tail_count > UINT32_MAX || head_count > UINT32_MAX ||
        raw_inner > NSUIntegerMax - raw_offset ||
        dst_inner > NSUIntegerMax - dst_offset) {
        return 0;
    }

    if (tail_rows &&
        !ds4_gpu_encode_cpy_f32_f16_1d(
                cb,
                raw,
                raw_offset + (NSUInteger)raw_inner,
                dst,
                dst_offset,
                (uint32_t)tail_count)) {
        return 0;
    }
    if (head_rows &&
        !ds4_gpu_encode_cpy_f32_f16_1d(
                cb,
                raw,
                raw_offset,
                dst,
                dst_offset + (NSUInteger)dst_inner,
                (uint32_t)head_count)) {
        return 0;
    }
    return 1;
}

static int ds4_gpu_encode_flash_kv_stage_f16(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        raw,
        NSUInteger           raw_offset,
        uint32_t             raw_cap,
        uint32_t             raw_start,
        uint32_t             n_raw,
        id<MTLBuffer>        comp,
        NSUInteger           comp_offset,
        bool                 comp_is_f16,
        uint32_t             n_comp,
        uint32_t             head_dim,
        id<MTLBuffer>        dst,
        NSUInteger           dst_offset,
        id<MTLBuffer>        mask,
        NSUInteger           mask_offset,
        id<MTLBuffer>        pad,
        NSUInteger           pad_offset,
        bool                 fuse_pad,
        bool                 shared_pad,
        bool                *did_fuse_pad) {
    if (did_fuse_pad) *did_fuse_pad = false;
    if (!cb || !raw || !comp || !dst || raw_cap == 0 ||
        raw_start >= raw_cap || n_raw == 0 || n_raw > raw_cap ||
        n_comp == 0 || head_dim == 0) {
        return 0;
    }

    const bool force =
        getenv("DS4_METAL_ENABLE_GATHERED_KV_STAGE") != NULL;
    const bool disabled =
        getenv("DS4_METAL_DISABLE_M3_GATHERED_KV_STAGE") != NULL;
    const bool require =
        getenv("DS4_METAL_REQUIRE_GATHERED_KV_STAGE") != NULL;
    const bool supported_shape =
        comp_is_f16 && head_dim == 512u && raw_cap <= UINT32_MAX / 128u;
    const uint64_t row_vecs64 = 128u;
    const uint64_t comp_count64 = (uint64_t)n_comp * head_dim;
    const uint64_t dst_comp_inner64 =
        (uint64_t)n_raw * head_dim * sizeof(uint16_t);
    if (comp_count64 > UINT32_MAX ||
        dst_comp_inner64 > NSUIntegerMax - dst_offset) {
        return 0;
    }
    const uint64_t total_vecs64 =
        ((uint64_t)n_raw + n_comp) * row_vecs64;
    const bool valid_grid =
        total_vecs64 != 0 && total_vecs64 <= UINT32_MAX;
    const bool eligible =
        supported_shape && valid_grid && !g_quality_mode && !disabled &&
        g_flash_kv_stage_f16_pipeline != nil &&
        (ds4_gpu_device_name_contains("M3") ||
         ds4_gpu_device_name_contains("M5") || force);
    const bool component_disabled = eligible &&
        (ds4_gpu_env_bool("DS4_METAL_DISABLE_CONTIG_F32_F16_COPY") > 0 ||
         ds4_gpu_env_bool("DS4_METAL_DISABLE_CONTIG_F16_F16_COPY") > 0);
    const bool use_fusion = eligible && !component_disabled;
    const bool use_pad_fusion =
        use_fusion && fuse_pad && mask != nil && pad != nil &&
        getenv("DS4_METAL_DISABLE_M3_GATHERED_KV_PAD_FUSION") == NULL;
    if (require && supported_shape && !use_fusion) {
        fprintf(stderr,
                "ds4: required Metal gathered KV staging kernel was not selected\n");
        return 0;
    }

    if (use_fusion) {
        ds4_gpu_flash_kv_stage_f16_args args = {
            .raw_cap = raw_cap,
            .raw_start = raw_start,
            .n_raw = n_raw,
            .n_comp = n_comp,
            .pad_rows = use_pad_fusion ? 32u : 0u,
            .shared_pad = use_pad_fusion && shared_pad ? 1u : 0u,
        };
        const NSUInteger total_vecs = (NSUInteger)total_vecs64 +
            (use_pad_fusion ? 32u * 128u + 32u : 0u);
        NSUInteger nth =
            g_flash_kv_stage_f16_pipeline.maxTotalThreadsPerThreadgroup;
        if (nth > 256u) nth = 256u;
        if (nth > total_vecs) nth = total_vecs;
        if (nth == 0) return 0;
        const NSUInteger groups = (total_vecs + nth - 1u) / nth;

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:g_flash_kv_stage_f16_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:raw offset:raw_offset atIndex:1];
        [enc setBuffer:comp offset:comp_offset atIndex:2];
        [enc setBuffer:dst offset:dst_offset atIndex:3];
        [enc setBuffer:(use_pad_fusion ? mask : dst)
                 offset:(use_pad_fusion ? mask_offset : dst_offset)
                atIndex:4];
        [enc setBuffer:(use_pad_fusion ? pad : dst)
                 offset:(use_pad_fusion ? pad_offset : dst_offset)
                atIndex:5];
        [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);
        if (did_fuse_pad) *did_fuse_pad = use_pad_fusion;
        return 1;
    }

    if (!ds4_gpu_encode_copy_raw_ring_to_f16(cb,
                                              raw,
                                              raw_offset,
                                              raw_cap,
                                              raw_start,
                                              n_raw,
                                              head_dim,
                                              dst,
                                              dst_offset)) {
        return 0;
    }
    return ds4_gpu_encode_copy_to_f16_1d(
        cb,
        comp,
        comp_offset,
        comp_is_f16,
        dst,
        dst_offset + (NSUInteger)dst_comp_inner64,
        (uint32_t)comp_count64);
}

int ds4_gpu_flash_kv_stage_f16_tensor(
        ds4_gpu_tensor       *dst,
        const ds4_gpu_tensor *raw,
        uint32_t                raw_cap,
        uint32_t                raw_start,
        uint32_t                n_raw,
        const ds4_gpu_tensor *comp,
        uint32_t                comp_is_f16,
        uint32_t                n_comp,
        uint32_t                head_dim) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!dst || !raw || !comp || raw_cap == 0 || raw_start >= raw_cap ||
        n_raw == 0 || n_raw > raw_cap || n_comp == 0 ||
        comp_is_f16 == 0 || head_dim != 512u) {
        return 0;
    }

    @autoreleasepool {
        const uint64_t raw_bytes =
            (uint64_t)raw_cap * head_dim * sizeof(float);
        const uint64_t comp_bytes =
            (uint64_t)n_comp * head_dim *
            (comp_is_f16 ? sizeof(uint16_t) : sizeof(float));
        const uint64_t dst_bytes =
            ((uint64_t)n_raw + n_comp) * head_dim * sizeof(uint16_t);
        id<MTLBuffer> rawbuf = ds4_gpu_tensor_buffer(raw);
        id<MTLBuffer> compbuf = ds4_gpu_tensor_buffer(comp);
        id<MTLBuffer> dstbuf = ds4_gpu_tensor_buffer(dst);
        if (!rawbuf || !compbuf || !dstbuf ||
            ds4_gpu_tensor_bytes(raw) < raw_bytes ||
            ds4_gpu_tensor_bytes(comp) < comp_bytes ||
            ds4_gpu_tensor_bytes(dst) < dst_bytes) {
            fprintf(stderr,
                    "ds4: Metal gathered KV staging received undersized buffers\n");
            return 0;
        }

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;
        if (!ds4_gpu_encode_flash_kv_stage_f16(
                cb,
                rawbuf,
                ds4_gpu_tensor_offset(raw),
                raw_cap,
                raw_start,
                n_raw,
                compbuf,
                ds4_gpu_tensor_offset(comp),
                comp_is_f16 != 0,
                n_comp,
                head_dim,
                dstbuf,
                ds4_gpu_tensor_offset(dst),
                nil,
                0,
                nil,
                0,
                false,
                false,
                NULL)) {
            return 0;
        }
        if (!ds4_gpu_finish_command_buffer(
                cb, owned, "gathered KV staging")) {
            return 0;
        }
    }
    return 1;
}

static int ds4_gpu_encode_fill_f16_1d(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        buf,
        NSUInteger           offset,
        uint32_t             n,
        float                value) {
    if (!cb || !buf || n == 0) return 0;

    ds4_gpu_unary_args args = ds4_gpu_make_unary_rows_args(n, 1, 0, 0.0f, 0.0f);
    args.val = value;

    NSUInteger nth = (NSUInteger)n;
    const NSUInteger max_threads = g_unary_fill_f16_pipeline.maxTotalThreadsPerThreadgroup;
    if (nth > max_threads) nth = max_threads;
    if (nth > 256u) nth = 256u;
    if (nth == 0) nth = 1u;
    const NSUInteger groups = ((NSUInteger)n + nth - 1u) / nth;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:g_unary_fill_f16_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:buf offset:offset atIndex:1];
    [enc setBuffer:buf offset:offset atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);

    return 1;
}

static int ds4_gpu_encode_flash_attention_raw_heads(
        id<MTLCommandBuffer>  cb,
        ds4_gpu_tensor     *heads,
        id<MTLBuffer>         sinks_buf,
        NSUInteger            sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        uint32_t              n_raw,
        uint32_t              raw_cap,
        uint32_t              raw_start,
        uint32_t              n_head,
        uint32_t              head_dim) {
    if (head_dim != 512 || n_head == 0 || n_raw == 0 || raw_cap < n_raw) {
        return 0;
    }

    id<MTLBuffer> qbuf = ds4_gpu_tensor_buffer(q);
    id<MTLBuffer> rawbuf = ds4_gpu_tensor_buffer(raw_kv);
    id<MTLBuffer> headsbuf = ds4_gpu_tensor_buffer(heads);
    const uint64_t q_bytes = (uint64_t)n_head * head_dim * sizeof(float);
    const uint64_t raw_bytes = (uint64_t)raw_cap * head_dim * sizeof(float);
    const uint64_t heads_bytes = q_bytes;
    if (!qbuf || !rawbuf || !headsbuf || !sinks_buf ||
        ds4_gpu_tensor_bytes(q) < q_bytes ||
        ds4_gpu_tensor_bytes(raw_kv) < raw_bytes ||
        ds4_gpu_tensor_bytes(heads) < heads_bytes) {
        fprintf(stderr, "ds4: Metal DS4 FlashAttention received undersized buffers\n");
        return 0;
    }

    const uint32_t ncpsg = 32;
    const uint32_t nwg = 32;
    const uint32_t nsg = ds4_gpu_flash_attn_vec_nsg(n_raw, nwg, ncpsg);
    const NSUInteger row_bytes = (NSUInteger)head_dim * sizeof(float);
    const NSUInteger row_bytes_f16 = (NSUInteger)head_dim * sizeof(uint16_t);
    const NSUInteger mask_bytes = (NSUInteger)n_raw * sizeof(uint16_t);
    const NSUInteger kv_bytes = (NSUInteger)n_raw * row_bytes_f16;
    const NSUInteger pad_bytes = 2u * (NSUInteger)ncpsg * row_bytes_f16 +
                                 (NSUInteger)ncpsg * sizeof(uint16_t);
    const NSUInteger nrows = (NSUInteger)n_head;
    const NSUInteger tmp_bytes = nrows * (NSUInteger)head_dim * (NSUInteger)nwg * sizeof(float) +
                                 nrows * (2u * (NSUInteger)nwg) * sizeof(float);

    id<MTLBuffer> mask_buffer =
        ds4_gpu_new_transient_buffer(mask_bytes, "ds4_flash_attn_mask");
    if (!mask_buffer ||
        !ds4_gpu_ensure_scratch_buffer(&g_flash_attn_kv_buffer,
                                         &g_flash_attn_kv_bytes,
                                         kv_bytes,
                                         "ds4_flash_attn_kv_f16") ||
        !ds4_gpu_ensure_scratch_buffer(&g_flash_attn_pad_buffer,
                                         &g_flash_attn_pad_bytes,
                                         pad_bytes,
                                         "ds4_flash_attn_pad") ||
        !ds4_gpu_ensure_scratch_buffer(&g_flash_attn_tmp_buffer,
                                         &g_flash_attn_tmp_bytes,
                                         tmp_bytes,
                                         "ds4_flash_attn_tmp")) {
        return 0;
    }
    memset([mask_buffer contents], 0, mask_bytes);

    id<MTLComputePipelineState> pad_pipeline = nil;
    if ((n_raw % ncpsg) != 0) {
        pad_pipeline = ds4_gpu_get_flash_attn_pad_pipeline(true, (int32_t)ncpsg);
        if (!pad_pipeline) return 0;
    }
    id<MTLComputePipelineState> vec_pipeline =
        ds4_gpu_get_flash_attn_vec_pipeline("kernel_flash_attn_ext_vec_f16_dk512_dv512",
                                              true, true, false, false, (n_raw % ncpsg) != 0,
                                              false,
                                              (int32_t)head_dim,
                                              (int32_t)head_dim,
                                              (int32_t)nsg,
                                              (int32_t)nwg);
    id<MTLComputePipelineState> reduce_pipeline =
        ds4_gpu_get_flash_attn_reduce_pipeline((int32_t)head_dim, (int32_t)nwg);
    if (!vec_pipeline || !reduce_pipeline) return 0;

    if (!ds4_gpu_encode_copy_raw_ring_to_f16(cb,
                                              rawbuf,
                                              ds4_gpu_tensor_offset(raw_kv),
                                              raw_cap,
                                              raw_start,
                                              n_raw,
                                              head_dim,
                                              g_flash_attn_kv_buffer,
                                              0)) {
        return 0;
    }

    if ((n_raw % ncpsg) != 0) {
        ds4_gpu_flash_attn_pad_args pad_args = {
            .ne11 = (int32_t)n_raw,
            .ne_12_2 = 1,
            .ne_12_3 = 1,
            .nb11 = row_bytes_f16,
            .nb12 = (uint64_t)n_raw * row_bytes_f16,
            .nb13 = (uint64_t)n_raw * row_bytes_f16,
            .nb21 = row_bytes_f16,
            .nb22 = (uint64_t)n_raw * row_bytes_f16,
            .nb23 = (uint64_t)n_raw * row_bytes_f16,
            .ne31 = 1,
            .ne32 = 1,
            .ne33 = 1,
            .nb31 = mask_bytes,
            .nb32 = mask_bytes,
            .nb33 = mask_bytes,
        };

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pad_pipeline];
        [enc setBytes:&pad_args length:sizeof(pad_args) atIndex:0];
        [enc setBuffer:g_flash_attn_kv_buffer offset:0 atIndex:1];
        [enc setBuffer:g_flash_attn_kv_buffer offset:0 atIndex:2];
        [enc setBuffer:mask_buffer offset:0 atIndex:3];
        [enc setBuffer:g_flash_attn_pad_buffer offset:0 atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(ncpsg, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);
    }

    ds4_gpu_flash_attn_vec_args vec_args = {
        .ne01 = 1,
        .ne02 = (int32_t)n_head,
        .ne03 = 1,
        .nb01 = (uint64_t)n_head * row_bytes,
        .nb02 = row_bytes,
        .nb03 = (uint64_t)n_head * row_bytes,
        .ne11 = (int32_t)n_raw,
        .ne_12_2 = 1,
        .ne_12_3 = 1,
        .ns10 = (int32_t)head_dim,
        .nb11 = row_bytes_f16,
        .nb12 = (uint64_t)n_raw * row_bytes_f16,
        .nb13 = (uint64_t)n_raw * row_bytes_f16,
        .ns20 = (int32_t)head_dim,
        .nb21 = row_bytes_f16,
        .nb22 = (uint64_t)n_raw * row_bytes_f16,
        .nb23 = (uint64_t)n_raw * row_bytes_f16,
        .ne31 = 1,
        .ne32 = 1,
        .ne33 = 1,
        .nb31 = mask_bytes,
        .nb32 = mask_bytes,
        .nb33 = mask_bytes,
        .ne1 = (int32_t)n_head,
        .ne2 = 1,
        .ne3 = 1,
        .scale = 1.0f / sqrtf((float)head_dim),
        .max_bias = 0.0f,
        .m0 = 0.0f,
        .m1 = 0.0f,
        .n_head_log2 = 0,
        .logit_softcap = 0.0f,
    };

    const NSUInteger shared_elems = (ds4_gpu_align_up_ns(head_dim, 128u) +
                                     4u * ncpsg +
                                     2u * ds4_gpu_align_up_ns(head_dim, 128u)) * nsg;
    const NSUInteger shared_bytes = ds4_gpu_align_up_ns(shared_elems * (sizeof(float) / 2u), 16u);

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:vec_pipeline];
    [enc setBytes:&vec_args length:sizeof(vec_args) atIndex:0];
    [enc setBuffer:qbuf offset:ds4_gpu_tensor_offset(q) atIndex:1];
    [enc setBuffer:g_flash_attn_kv_buffer offset:0 atIndex:2];
    [enc setBuffer:g_flash_attn_kv_buffer offset:0 atIndex:3];
    [enc setBuffer:mask_buffer offset:0 atIndex:4];
    [enc setBuffer:sinks_buf offset:sinks_offset atIndex:5];
    [enc setBuffer:g_flash_attn_pad_buffer offset:0 atIndex:6];
    [enc setBuffer:g_flash_attn_tmp_buffer offset:0 atIndex:7];
    [enc setThreadgroupMemoryLength:shared_bytes atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake(1, n_head, nwg)
         threadsPerThreadgroup:MTLSizeMake(32, nsg, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);

    ds4_gpu_flash_attn_reduce_args reduce_args = {
        .nrows = (int32_t)nrows,
    };
    enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:reduce_pipeline];
    [enc setBytes:&reduce_args length:sizeof(reduce_args) atIndex:0];
    [enc setBuffer:g_flash_attn_tmp_buffer offset:0 atIndex:1];
    [enc setBuffer:headsbuf offset:ds4_gpu_tensor_offset(heads) atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake(nrows, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(32u * nwg, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);

    return 1;
}

/* Rectangular causal/window mask for raw prefill: q covers rows
 * [q_row0, q_row0 + n_q) of an n_kv-token block whose keys all live in
 * rows [0, n_kv).  The square prefill case is q_row0 == 0, n_q == n_kv. */
static void ds4_gpu_fill_raw_prefill_mask(
        uint16_t *mask,
        uint32_t  q_row0,
        uint32_t  n_q,
        uint32_t  n_kv,
        uint32_t  window) {
    const uint16_t neg_inf_half = 0xfc00u;
    for (uint32_t q = 0; q < n_q; q++) {
        const uint32_t qpos = q_row0 + q;
        uint16_t *row = mask + (uint64_t)q * n_kv;
        for (uint32_t k = 0; k < n_kv; k++) {
            const bool causal = k <= qpos;
            const bool in_window = window == 0 || qpos - k < window;
            row[k] = causal && in_window ? 0u : neg_inf_half;
        }
    }
}

static void ds4_gpu_fill_glm_prefill_mask(
        uint16_t *mask,
        uint32_t  pos0,
        uint32_t  n_tokens,
        uint32_t  cache_len) {
    const uint16_t neg_inf_half = 0xfc00u;
    for (uint32_t q = 0; q < n_tokens; q++) {
        const uint32_t qpos = pos0 + q;
        uint16_t *row = mask + (uint64_t)q * cache_len;
        for (uint32_t k = 0; k < cache_len; k++) {
            row[k] = k <= qpos ? 0u : neg_inf_half;
        }
    }
}

static id<MTLBuffer> ds4_gpu_glm_prefill_mask_buffer(
        uint32_t pos0,
        uint32_t n_tokens,
        uint32_t cache_len,
        NSUInteger mask_bytes) {
    const int same_shape =
        g_glm_flash_attn_mask_valid &&
        g_glm_flash_attn_mask_buffer &&
        g_glm_flash_attn_mask_bytes >= mask_bytes &&
        g_glm_flash_attn_mask_pos0 == pos0 &&
        g_glm_flash_attn_mask_tokens == n_tokens &&
        g_glm_flash_attn_mask_cache_len == cache_len;
    if (same_shape) return g_glm_flash_attn_mask_buffer;

    if (g_glm_flash_attn_mask_buffer) {
        [g_transient_buffers addObject:g_glm_flash_attn_mask_buffer];
        g_glm_flash_attn_mask_buffer = nil;
    }
    g_glm_flash_attn_mask_bytes = 0;
    g_glm_flash_attn_mask_valid = 0;
    if (!ds4_gpu_ensure_scratch_buffer(&g_glm_flash_attn_mask_buffer,
                                        &g_glm_flash_attn_mask_bytes,
                                        mask_bytes,
                                        "ds4_glm_flash_attn_mask")) {
        return nil;
    }

    ds4_gpu_fill_glm_prefill_mask((uint16_t *)[g_glm_flash_attn_mask_buffer contents],
                                  pos0,
                                  n_tokens,
                                  cache_len);
    g_glm_flash_attn_mask_pos0 = pos0;
    g_glm_flash_attn_mask_tokens = n_tokens;
    g_glm_flash_attn_mask_cache_len = cache_len;
    g_glm_flash_attn_mask_valid = 1;
    return g_glm_flash_attn_mask_buffer;
}

static void ds4_gpu_fill_raw_decode_batch_mask(
        uint16_t *mask,
        uint32_t  n_tokens,
        uint32_t  n_raw,
        uint32_t  pos0,
        uint32_t  window) {
    const uint16_t neg_inf_half = 0xfc00u;
    const uint32_t last_pos = pos0 + n_tokens - 1u;
    /* The caller has already copied the SWA ring into logical order when it
     * wraps, so key row k represents first_raw_pos + k. */
    const uint32_t first_raw_pos = last_pos + 1u - n_raw;
    for (uint32_t q = 0; q < n_tokens; q++) {
        const uint32_t qpos = pos0 + q;
        uint16_t *row = mask + (uint64_t)q * n_raw;
        for (uint32_t k = 0; k < n_raw; k++) {
            const uint32_t kpos = first_raw_pos + k;
            const bool causal = kpos <= qpos;
            const bool in_window = causal && (window == 0 || qpos - kpos < window);
            row[k] = causal && in_window ? 0u : neg_inf_half;
        }
    }
}

static void ds4_gpu_fill_mixed_decode_batch_mask(
        uint16_t *mask,
        uint32_t  n_tokens,
        uint32_t  n_raw,
        uint32_t  n_comp,
        uint32_t  pos0,
        uint32_t  window,
        uint32_t  ratio) {
    const uint16_t neg_inf_half = 0xfc00u;
    const uint32_t n_keys = n_raw + n_comp;
    const uint32_t last_pos = pos0 + n_tokens - 1u;
    /* Raw keys are laid out by logical position; compressed keys follow them. */
    const uint32_t first_raw_pos = last_pos + 1u - n_raw;
    for (uint32_t q = 0; q < n_tokens; q++) {
        const uint32_t qpos = pos0 + q;
        uint16_t *row = mask + (uint64_t)q * n_keys;
        for (uint32_t k = 0; k < n_raw; k++) {
            const uint32_t kpos = first_raw_pos + k;
            const bool causal = kpos <= qpos;
            const bool in_window = causal && (window == 0 || qpos - kpos < window);
            row[k] = causal && in_window ? 0u : neg_inf_half;
        }
        const uint32_t n_visible = (qpos + 1u) / ratio;
        for (uint32_t c = 0; c < n_comp; c++) {
            row[n_raw + c] = c < n_visible ? 0u : neg_inf_half;
        }
    }
}

/* Rectangular causal/window + compressed-key visibility mask: q covers rows
 * [q_row0, q_row0 + n_q) of an n_tokens-token chunk whose raw keys all stay
 * resident, followed by n_comp compressed keys.  The square prefill case is
 * q_row0 == 0, n_q == n_tokens. */
static void ds4_gpu_fill_static_mixed_prefill_mask(
        uint16_t *mask,
        uint32_t  q_row0,
        uint32_t  n_q,
        uint32_t  n_tokens,
        uint32_t  n_comp,
        uint32_t  window,
        uint32_t  ratio) {
    const uint16_t neg_inf_half = 0xfc00u;
    const uint32_t n_keys = n_tokens + n_comp;
    for (uint32_t q = 0; q < n_q; q++) {
        const uint32_t qpos = q_row0 + q;
        uint16_t *row = mask + (uint64_t)q * n_keys;
        for (uint32_t k = 0; k < n_tokens; k++) {
            const bool causal = k <= qpos;
            const bool in_window = window == 0 || qpos - k < window;
            row[k] = causal && in_window ? 0u : neg_inf_half;
        }

        const uint32_t n_visible = (qpos + 1u) / ratio;
        for (uint32_t c = 0; c < n_comp; c++) {
            row[n_tokens + c] = c < n_visible ? 0u : neg_inf_half;
        }
    }
}

/* Static-mixed prefill FlashAttention over a rectangular problem: q holds
 * n_q query rows for token positions [q_row0, q_row0 + n_q) of the chunk,
 * while the keys stay full (all n_tokens raw rows plus n_comp compressed
 * rows).  The classic square prefill is q_row0 == 0, n_q == n_tokens. */
static int ds4_gpu_encode_flash_attention_prefill_static_mixed_heads_nonvec_long(
        id<MTLCommandBuffer> __strong *cbp,
        ds4_gpu_tensor      *heads,
        id<MTLBuffer>          sinks_buf,
        NSUInteger             sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        const ds4_gpu_tensor *comp_kv,
        uint32_t               comp_kv_f16,
        const ds4_gpu_tensor *comp_mask,
        uint32_t               use_comp_mask,
        uint32_t               q_row0,
        uint32_t               n_q,
        uint32_t               n_tokens,
        uint32_t               n_comp,
        uint32_t               window,
        uint32_t               ratio,
        uint32_t               n_head,
        uint32_t               head_dim) {
    if (!cbp || !*cbp) return 0;
    id<MTLCommandBuffer> cb = *cbp;
    if (head_dim != 512 || n_head == 0 || n_q == 0 || n_tokens == 0 || ratio == 0) {
        return 0;
    }

    const uint32_t n_keys = n_tokens + n_comp;
    id<MTLBuffer> qbuf = ds4_gpu_tensor_buffer(q);
    id<MTLBuffer> rawbuf = ds4_gpu_tensor_buffer(raw_kv);
    id<MTLBuffer> compbuf = n_comp ? ds4_gpu_tensor_buffer(comp_kv) : rawbuf;
    id<MTLBuffer> maskbuf = use_comp_mask ? ds4_gpu_tensor_buffer(comp_mask) : rawbuf;
    id<MTLBuffer> headsbuf = ds4_gpu_tensor_buffer(heads);
    const uint64_t q_bytes = (uint64_t)n_q * n_head * head_dim * sizeof(float);
    const uint64_t raw_bytes = (uint64_t)n_tokens * head_dim * sizeof(float);
    const uint64_t comp_bytes = (uint64_t)n_comp * head_dim *
                                (comp_kv_f16 ? sizeof(uint16_t) : sizeof(float));
    const uint64_t comp_mask_bytes = use_comp_mask
        ? (uint64_t)n_comp * (q_row0 + n_q) * sizeof(float) : 0u;
    if (!qbuf || !rawbuf || !compbuf || !maskbuf || !headsbuf || !sinks_buf ||
        ds4_gpu_tensor_bytes(q) < q_bytes ||
        ds4_gpu_tensor_bytes(raw_kv) < raw_bytes ||
        (n_comp && ds4_gpu_tensor_bytes(comp_kv) < comp_bytes) ||
        (use_comp_mask && ds4_gpu_tensor_bytes(comp_mask) < comp_mask_bytes) ||
        ds4_gpu_tensor_bytes(heads) < q_bytes) {
        fprintf(stderr, "ds4: Metal prefill static mixed DS4 non-vector FlashAttention received undersized buffers\n");
        return 0;
    }

    const uint32_t nqptg = 8;
    const uint32_t ncpsg = 64;
    const uint32_t nsg = head_dim >= 512 ? 8u : 4u;
    const bool has_kvpad = (n_keys % ncpsg) != 0;
    const bool bc_mask = (n_q % nqptg) != 0;
    const NSUInteger row_bytes = (NSUInteger)head_dim * sizeof(float);
    const NSUInteger row_bytes_f16 = (NSUInteger)head_dim * sizeof(uint16_t);
    const NSUInteger mask_bytes = (NSUInteger)n_keys * (NSUInteger)n_q * sizeof(uint16_t);
    const NSUInteger kv_bytes = (NSUInteger)n_keys * row_bytes_f16;
    const NSUInteger pad_bytes = has_kvpad
        ? (NSUInteger)ncpsg * (2u * row_bytes_f16 + (NSUInteger)n_q * sizeof(uint16_t))
        : 1u;
    const NSUInteger nblk0 = ((NSUInteger)n_keys + ncpsg - 1u) / ncpsg;
    const NSUInteger nblk1 = ((NSUInteger)n_q + nqptg - 1u) / nqptg;
    const NSUInteger blk_bytes = ds4_gpu_align_up_ns(nblk0 * nblk1, 32u);

    const uint32_t mask_cache_kind = ratio == 4u
        ? DS4_GPU_PREFILL_MASK_CACHE_RATIO4
        : (ratio == 128u ? DS4_GPU_PREFILL_MASK_CACHE_RATIO128 : 0u);
    bool mask_cache_created = false;
    ds4_gpu_zero_prefix_prefill_mask_cache_entry *mask_cache =
        use_comp_mask == 0u && mask_cache_kind != 0u &&
        q_row0 == 0u && n_q == n_tokens
            ? ds4_gpu_get_zero_prefix_prefill_mask_cache(mask_cache_kind,
                                                          n_tokens,
                                                          n_comp,
                                                          n_keys,
                                                          window,
                                                          ratio,
                                                          nqptg,
                                                          ncpsg,
                                                          has_kvpad,
                                                          bc_mask,
                                                          mask_bytes,
                                                          blk_bytes,
                                                          &mask_cache_created)
            : NULL;
    id<MTLBuffer> mask_buffer = mask_cache
        ? mask_cache->mask
        : ds4_gpu_new_transient_buffer(mask_bytes, "ds4_flash_attn_mask");
    if (mask_cache && mask_cache_created) {
        ds4_gpu_fill_static_mixed_prefill_mask((uint16_t *)[mask_buffer contents],
                                                 0u,
                                                 n_tokens,
                                                 n_tokens,
                                                 n_comp,
                                                 window,
                                                 ratio);
        mask_cache->valid = true;
    }
    if (!mask_buffer ||
        !ds4_gpu_ensure_scratch_buffer(&g_flash_attn_kv_buffer,
                                         &g_flash_attn_kv_bytes,
                                         kv_bytes,
                                         "ds4_flash_attn_kv_f16") ||
        !ds4_gpu_ensure_scratch_buffer(&g_flash_attn_pad_buffer,
                                         &g_flash_attn_pad_bytes,
                                         pad_bytes,
                                         "ds4_flash_attn_pad") ||
        (!mask_cache &&
         !ds4_gpu_ensure_scratch_buffer(&g_flash_attn_blk_buffer,
                                          &g_flash_attn_blk_bytes,
                                          blk_bytes,
                                          "ds4_flash_attn_blk"))) {
        return 0;
    }
    id<MTLBuffer> blk_buffer = mask_cache
        ? mask_cache->blk
        : g_flash_attn_blk_buffer;

    const bool flash_stage_profile =
        getenv("DS4_METAL_FLASH_ATTN_STAGE_PROFILE") != NULL && g_batch_cb != nil;
    double flash_stage_t0 = 0.0;
    if (flash_stage_profile) {
        if (ds4_gpu_end_commands() == 0 || ds4_gpu_begin_commands() == 0) {
            return 0;
        }
        int profile_owned = 0;
        cb = ds4_gpu_command_buffer(&profile_owned);
        if (!cb || profile_owned) return 0;
        *cbp = cb;
        flash_stage_t0 = ds4_gpu_now_ms();
    }
#define DS4_METAL_PROFILE_FLASH_ATTN_STAGE(name) do { \
        if (flash_stage_profile) { \
            if (!ds4_gpu_flash_attn_stage_profile_boundary(cbp, \
                    "static_mixed_nonvec", (name), n_q, n_comp, n_keys, \
                    n_head, head_dim, window, ratio, &flash_stage_t0)) { \
                return 0; \
            } \
            cb = *cbp; \
        } \
    } while (0)

    if (!ds4_gpu_encode_cpy_f32_f16_1d(cb,
                                         rawbuf,
                                         ds4_gpu_tensor_offset(raw_kv),
                                         g_flash_attn_kv_buffer,
                                         0,
                                         n_tokens * head_dim)) {
        return 0;
    }
    DS4_METAL_PROFILE_FLASH_ATTN_STAGE("copy_raw");
    if (n_comp &&
        !ds4_gpu_encode_copy_to_f16_1d(cb,
                                       compbuf,
                                       ds4_gpu_tensor_offset(comp_kv),
                                       comp_kv_f16 != 0,
                                       g_flash_attn_kv_buffer,
                                       (NSUInteger)n_tokens * row_bytes_f16,
                                       n_comp * head_dim)) {
        return 0;
    }
    if (n_comp) {
        DS4_METAL_PROFILE_FLASH_ATTN_STAGE("copy_comp");
    }

    if (!mask_cache) {
        ds4_gpu_fill_static_mixed_prefill_mask((uint16_t *)[mask_buffer contents],
                                                 q_row0,
                                                 n_q,
                                                 n_tokens,
                                                 n_comp,
                                                 window,
                                                 ratio);
    }
    DS4_METAL_PROFILE_FLASH_ATTN_STAGE("mask_fill");
    if (use_comp_mask && n_comp != 0) {
        if (!ds4_gpu_encode_cpy_f32_f16_2d(cb,
                                             maskbuf,
                                             ds4_gpu_tensor_offset(comp_mask) +
                                                 (NSUInteger)((uint64_t)q_row0 * n_comp * sizeof(float)),
                                             mask_buffer,
                                             (NSUInteger)n_tokens * sizeof(uint16_t),
                                             n_comp,
                                             n_q,
                                             (uint64_t)n_comp * sizeof(float),
                                             (uint64_t)n_keys * sizeof(uint16_t))) {
            return 0;
        }
        DS4_METAL_PROFILE_FLASH_ATTN_STAGE("mask_comp_copy");
    }

    id<MTLComputePipelineState> pad_pipeline = nil;
    if (has_kvpad) {
        pad_pipeline = ds4_gpu_get_flash_attn_pad_pipeline(true, (int32_t)ncpsg);
        if (!pad_pipeline) return 0;
    }
    id<MTLComputePipelineState> blk_pipeline =
        ds4_gpu_get_flash_attn_blk_pipeline((int32_t)nqptg, (int32_t)ncpsg);
    id<MTLComputePipelineState> attn_pipeline =
        ds4_gpu_get_flash_attn_pipeline("kernel_flash_attn_ext_f16_dk512_dv512",
                                          true, true, false, false, has_kvpad, bc_mask,
                                          (int32_t)head_dim,
                                          (int32_t)head_dim,
                                          (int32_t)nsg);
    if (!blk_pipeline || !attn_pipeline) return 0;

    if (has_kvpad) {
        ds4_gpu_flash_attn_pad_args pad_args = {
            .ne11 = (int32_t)n_keys,
            .ne_12_2 = 1,
            .ne_12_3 = 1,
            .nb11 = row_bytes_f16,
            .nb12 = (uint64_t)n_keys * row_bytes_f16,
            .nb13 = (uint64_t)n_keys * row_bytes_f16,
            .nb21 = row_bytes_f16,
            .nb22 = (uint64_t)n_keys * row_bytes_f16,
            .nb23 = (uint64_t)n_keys * row_bytes_f16,
            .ne31 = (int32_t)n_q,
            .ne32 = 1,
            .ne33 = 1,
            .nb31 = (uint64_t)n_keys * sizeof(uint16_t),
            .nb32 = mask_bytes,
            .nb33 = mask_bytes,
        };

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pad_pipeline];
        [enc setBytes:&pad_args length:sizeof(pad_args) atIndex:0];
        [enc setBuffer:g_flash_attn_kv_buffer offset:0 atIndex:1];
        [enc setBuffer:g_flash_attn_kv_buffer offset:0 atIndex:2];
        [enc setBuffer:mask_buffer offset:0 atIndex:3];
        [enc setBuffer:g_flash_attn_pad_buffer offset:0 atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(ncpsg, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);
        DS4_METAL_PROFILE_FLASH_ATTN_STAGE("pad");
    }

    ds4_gpu_flash_attn_blk_args blk_args = {
        .ne01 = (int32_t)n_q,
        .ne30 = (int32_t)n_keys,
        .ne31 = (int32_t)n_q,
        .ne32 = 1,
        .ne33 = 1,
        .nb31 = (uint64_t)n_keys * sizeof(uint16_t),
        .nb32 = mask_bytes,
        .nb33 = mask_bytes,
    };

    id<MTLComputeCommandEncoder> enc = nil;
    if (!mask_cache || !mask_cache->blk_ready) {
        enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:blk_pipeline];
        [enc setBytes:&blk_args length:sizeof(blk_args) atIndex:0];
        [enc setBuffer:mask_buffer offset:0 atIndex:1];
        [enc setBuffer:blk_buffer offset:0 atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake(nblk0, nblk1, 1)
             threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);
        if (mask_cache) mask_cache->blk_ready = true;
    }
    DS4_METAL_PROFILE_FLASH_ATTN_STAGE("block_map");

    ds4_gpu_flash_attn_vec_args args = {
        .ne01 = (int32_t)n_q,
        .ne02 = (int32_t)n_head,
        .ne03 = 1,
        .nb01 = (uint64_t)n_head * row_bytes,
        .nb02 = row_bytes,
        .nb03 = (uint64_t)n_q * n_head * row_bytes,
        .ne11 = (int32_t)n_keys,
        .ne_12_2 = 1,
        .ne_12_3 = 1,
        .ns10 = (int32_t)head_dim,
        .nb11 = row_bytes_f16,
        .nb12 = (uint64_t)n_keys * row_bytes_f16,
        .nb13 = (uint64_t)n_keys * row_bytes_f16,
        .ns20 = (int32_t)head_dim,
        .nb21 = row_bytes_f16,
        .nb22 = (uint64_t)n_keys * row_bytes_f16,
        .nb23 = (uint64_t)n_keys * row_bytes_f16,
        .ne31 = (int32_t)n_q,
        .ne32 = 1,
        .ne33 = 1,
        .nb31 = (uint64_t)n_keys * sizeof(uint16_t),
        .nb32 = mask_bytes,
        .nb33 = mask_bytes,
        .ne1 = (int32_t)n_head,
        .ne2 = (int32_t)n_q,
        .ne3 = 1,
        .scale = 1.0f / sqrtf((float)head_dim),
        .max_bias = 0.0f,
        .m0 = 0.0f,
        .m1 = 0.0f,
        .n_head_log2 = 0,
        .logit_softcap = 0.0f,
    };

    const NSUInteger padded_v = ds4_gpu_align_up_ns(head_dim, 64u);
    const NSUInteger shared_elems = (NSUInteger)nqptg *
        ((NSUInteger)head_dim + 2u * padded_v + 2u * (2u * (NSUInteger)ncpsg));
    const NSUInteger shared_bytes = ds4_gpu_align_up_ns(shared_elems * (sizeof(float) / 2u), 16u);

    enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:attn_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:qbuf offset:ds4_gpu_tensor_offset(q) atIndex:1];
    [enc setBuffer:g_flash_attn_kv_buffer offset:0 atIndex:2];
    [enc setBuffer:g_flash_attn_kv_buffer offset:0 atIndex:3];
    [enc setBuffer:mask_buffer offset:0 atIndex:4];
    [enc setBuffer:sinks_buf offset:sinks_offset atIndex:5];
    [enc setBuffer:g_flash_attn_pad_buffer offset:0 atIndex:6];
    [enc setBuffer:blk_buffer offset:0 atIndex:7];
    [enc setBuffer:headsbuf offset:ds4_gpu_tensor_offset(heads) atIndex:8];
    [enc setThreadgroupMemoryLength:shared_bytes atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake(nblk1, n_head, 1)
         threadsPerThreadgroup:MTLSizeMake(32, nsg, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    DS4_METAL_PROFILE_FLASH_ATTN_STAGE("attention");

#undef DS4_METAL_PROFILE_FLASH_ATTN_STAGE
    return 1;
}

static int ds4_gpu_encode_flash_attention_prefill_static_mixed_heads_vec(
        id<MTLCommandBuffer> __strong *cbp,
        ds4_gpu_tensor      *heads,
        id<MTLBuffer>          sinks_buf,
        NSUInteger             sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        const ds4_gpu_tensor *comp_kv,
        uint32_t               comp_kv_f16,
        const ds4_gpu_tensor *comp_mask,
        uint32_t               use_comp_mask,
        uint32_t               n_tokens,
        uint32_t               n_comp,
        uint32_t               window,
        uint32_t               ratio,
        uint32_t               n_head,
        uint32_t               head_dim) {
    if (!cbp || !*cbp) return 0;
    id<MTLCommandBuffer> cb = *cbp;
    if (head_dim != 512 || n_head == 0 || n_tokens == 0 || ratio == 0) {
        return 0;
    }

    const uint32_t n_keys = n_tokens + n_comp;
    id<MTLBuffer> qbuf = ds4_gpu_tensor_buffer(q);
    id<MTLBuffer> rawbuf = ds4_gpu_tensor_buffer(raw_kv);
    id<MTLBuffer> compbuf = n_comp ? ds4_gpu_tensor_buffer(comp_kv) : rawbuf;
    id<MTLBuffer> maskbuf = use_comp_mask ? ds4_gpu_tensor_buffer(comp_mask) : rawbuf;
    id<MTLBuffer> headsbuf = ds4_gpu_tensor_buffer(heads);
    const uint64_t q_bytes = (uint64_t)n_tokens * n_head * head_dim * sizeof(float);
    const uint64_t raw_bytes = (uint64_t)n_tokens * head_dim * sizeof(float);
    const uint64_t comp_bytes = (uint64_t)n_comp * head_dim *
                                (comp_kv_f16 ? sizeof(uint16_t) : sizeof(float));
    const uint64_t comp_mask_bytes = use_comp_mask ? (uint64_t)n_comp * n_tokens * sizeof(float) : 0u;
    if (!qbuf || !rawbuf || !compbuf || !maskbuf || !headsbuf || !sinks_buf ||
        ds4_gpu_tensor_bytes(q) < q_bytes ||
        ds4_gpu_tensor_bytes(raw_kv) < raw_bytes ||
        (n_comp && ds4_gpu_tensor_bytes(comp_kv) < comp_bytes) ||
        (use_comp_mask && ds4_gpu_tensor_bytes(comp_mask) < comp_mask_bytes) ||
        ds4_gpu_tensor_bytes(heads) < q_bytes) {
        fprintf(stderr, "ds4: Metal prefill static mixed DS4 FlashAttention received undersized buffers\n");
        return 0;
    }

    const uint32_t ncpsg = 32;
    const uint32_t nwg = 32;
    const uint32_t nsg = ds4_gpu_flash_attn_vec_nsg(n_keys, nwg, ncpsg);
    const NSUInteger row_bytes = (NSUInteger)head_dim * sizeof(float);
    const NSUInteger row_bytes_f16 = (NSUInteger)head_dim * sizeof(uint16_t);
    const NSUInteger mask_bytes = (NSUInteger)n_keys * (NSUInteger)n_tokens * sizeof(uint16_t);
    const NSUInteger kv_bytes = (NSUInteger)n_keys * row_bytes_f16;
    const bool has_kvpad = (n_keys % ncpsg) != 0;
    const NSUInteger pad_bytes = has_kvpad
        ? (NSUInteger)ncpsg * (2u * row_bytes_f16 + (NSUInteger)n_tokens * sizeof(uint16_t))
        : 1u;
    const NSUInteger nrows = (NSUInteger)n_tokens * n_head;
    const NSUInteger tmp_bytes = nrows * (NSUInteger)head_dim * (NSUInteger)nwg * sizeof(float) +
                                 nrows * (2u * (NSUInteger)nwg) * sizeof(float);

    id<MTLBuffer> mask_buffer =
        ds4_gpu_new_transient_buffer(mask_bytes, "ds4_flash_attn_mask");
    if (!mask_buffer ||
        !ds4_gpu_ensure_scratch_buffer(&g_flash_attn_kv_buffer,
                                         &g_flash_attn_kv_bytes,
                                         kv_bytes,
                                         "ds4_flash_attn_kv") ||
        !ds4_gpu_ensure_scratch_buffer(&g_flash_attn_pad_buffer,
                                         &g_flash_attn_pad_bytes,
                                         pad_bytes,
                                         "ds4_flash_attn_pad") ||
        !ds4_gpu_ensure_scratch_buffer(&g_flash_attn_tmp_buffer,
                                         &g_flash_attn_tmp_bytes,
                                         tmp_bytes,
                                         "ds4_flash_attn_tmp")) {
        return 0;
    }

    const bool flash_stage_profile =
        getenv("DS4_METAL_FLASH_ATTN_STAGE_PROFILE") != NULL && g_batch_cb != nil;
    double flash_stage_t0 = 0.0;
    if (flash_stage_profile) {
        if (ds4_gpu_end_commands() == 0 || ds4_gpu_begin_commands() == 0) {
            return 0;
        }
        int profile_owned = 0;
        cb = ds4_gpu_command_buffer(&profile_owned);
        if (!cb || profile_owned) return 0;
        *cbp = cb;
        flash_stage_t0 = ds4_gpu_now_ms();
    }
#define DS4_METAL_PROFILE_FLASH_ATTN_STAGE(name) do { \
        if (flash_stage_profile) { \
            if (!ds4_gpu_flash_attn_stage_profile_boundary(cbp, \
                    "static_mixed_vec", (name), n_tokens, n_comp, n_keys, \
                    n_head, head_dim, window, ratio, &flash_stage_t0)) { \
                return 0; \
            } \
            cb = *cbp; \
        } \
    } while (0)

    if (!ds4_gpu_encode_cpy_f32_f16_1d(cb,
                                         rawbuf,
                                         ds4_gpu_tensor_offset(raw_kv),
                                         g_flash_attn_kv_buffer,
                                         0,
                                         n_tokens * head_dim)) {
        return 0;
    }
    DS4_METAL_PROFILE_FLASH_ATTN_STAGE("copy_raw");
    if (n_comp) {
        if (!ds4_gpu_encode_copy_to_f16_1d(cb,
                                           compbuf,
                                           ds4_gpu_tensor_offset(comp_kv),
                                           comp_kv_f16 != 0,
                                           g_flash_attn_kv_buffer,
                                           (NSUInteger)n_tokens * row_bytes_f16,
                                           n_comp * head_dim)) {
            return 0;
        }
        DS4_METAL_PROFILE_FLASH_ATTN_STAGE("copy_comp");
    }

    ds4_gpu_fill_static_mixed_prefill_mask((uint16_t *)[mask_buffer contents],
                                             0,
                                             n_tokens,
                                             n_tokens,
                                             n_comp,
                                             window,
                                             ratio);
    DS4_METAL_PROFILE_FLASH_ATTN_STAGE("mask_fill");
    if (use_comp_mask && n_comp != 0) {
        if (!ds4_gpu_encode_cpy_f32_f16_2d(cb,
                                             maskbuf,
                                             ds4_gpu_tensor_offset(comp_mask),
                                             mask_buffer,
                                             (NSUInteger)n_tokens * sizeof(uint16_t),
                                             n_comp,
                                             n_tokens,
                                             (uint64_t)n_comp * sizeof(float),
                                             (uint64_t)n_keys * sizeof(uint16_t))) {
            return 0;
        }
        DS4_METAL_PROFILE_FLASH_ATTN_STAGE("mask_comp_copy");
    }

    id<MTLComputePipelineState> pad_pipeline = nil;
    id<MTLComputeCommandEncoder> enc = nil;
    if (has_kvpad) {
        pad_pipeline = ds4_gpu_get_flash_attn_pad_pipeline(true, (int32_t)ncpsg);
        if (!pad_pipeline) return 0;
    }
    id<MTLComputePipelineState> vec_pipeline =
        ds4_gpu_get_flash_attn_vec_pipeline("kernel_flash_attn_ext_vec_f16_dk512_dv512",
                                              true, true, false, false, has_kvpad,
                                              false,
                                              (int32_t)head_dim,
                                              (int32_t)head_dim,
                                              (int32_t)nsg,
                                              (int32_t)nwg);
    id<MTLComputePipelineState> reduce_pipeline =
        ds4_gpu_get_flash_attn_reduce_pipeline((int32_t)head_dim, (int32_t)nwg);
    if (!vec_pipeline || !reduce_pipeline) return 0;

    if (has_kvpad) {
        ds4_gpu_flash_attn_pad_args pad_args = {
            .ne11 = (int32_t)n_keys,
            .ne_12_2 = 1,
            .ne_12_3 = 1,
            .nb11 = row_bytes_f16,
            .nb12 = (uint64_t)n_keys * row_bytes_f16,
            .nb13 = (uint64_t)n_keys * row_bytes_f16,
            .nb21 = row_bytes_f16,
            .nb22 = (uint64_t)n_keys * row_bytes_f16,
            .nb23 = (uint64_t)n_keys * row_bytes_f16,
            .ne31 = (int32_t)n_tokens,
            .ne32 = 1,
            .ne33 = 1,
            .nb31 = (uint64_t)n_keys * sizeof(uint16_t),
            .nb32 = mask_bytes,
            .nb33 = mask_bytes,
        };

        enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pad_pipeline];
        [enc setBytes:&pad_args length:sizeof(pad_args) atIndex:0];
        [enc setBuffer:g_flash_attn_kv_buffer offset:0 atIndex:1];
        [enc setBuffer:g_flash_attn_kv_buffer offset:0 atIndex:2];
        [enc setBuffer:mask_buffer offset:0 atIndex:3];
        [enc setBuffer:g_flash_attn_pad_buffer offset:0 atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(ncpsg, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);
        DS4_METAL_PROFILE_FLASH_ATTN_STAGE("pad");
    }

    ds4_gpu_flash_attn_vec_args vec_args = {
        .ne01 = (int32_t)n_tokens,
        .ne02 = (int32_t)n_head,
        .ne03 = 1,
        .nb01 = (uint64_t)n_head * row_bytes,
        .nb02 = row_bytes,
        .nb03 = (uint64_t)n_tokens * n_head * row_bytes,
        .ne11 = (int32_t)n_keys,
        .ne_12_2 = 1,
        .ne_12_3 = 1,
        .ns10 = (int32_t)head_dim,
        .nb11 = row_bytes_f16,
        .nb12 = (uint64_t)n_keys * row_bytes_f16,
        .nb13 = (uint64_t)n_keys * row_bytes_f16,
        .ns20 = (int32_t)head_dim,
        .nb21 = row_bytes_f16,
        .nb22 = (uint64_t)n_keys * row_bytes_f16,
        .nb23 = (uint64_t)n_keys * row_bytes_f16,
        .ne31 = (int32_t)n_tokens,
        .ne32 = 1,
        .ne33 = 1,
        .nb31 = (uint64_t)n_keys * sizeof(uint16_t),
        .nb32 = mask_bytes,
        .nb33 = mask_bytes,
        .ne1 = (int32_t)n_head,
        .ne2 = (int32_t)n_tokens,
        .ne3 = 1,
        .scale = 1.0f / sqrtf((float)head_dim),
        .max_bias = 0.0f,
        .m0 = 0.0f,
        .m1 = 0.0f,
        .n_head_log2 = 0,
        .logit_softcap = 0.0f,
    };

    const NSUInteger shared_elems = (ds4_gpu_align_up_ns(head_dim, 128u) +
                                     4u * ncpsg +
                                     2u * ds4_gpu_align_up_ns(head_dim, 128u)) * nsg;
    const NSUInteger shared_bytes = ds4_gpu_align_up_ns(shared_elems * (sizeof(float) / 2u), 16u);

    enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:vec_pipeline];
    [enc setBytes:&vec_args length:sizeof(vec_args) atIndex:0];
    [enc setBuffer:qbuf offset:ds4_gpu_tensor_offset(q) atIndex:1];
    [enc setBuffer:g_flash_attn_kv_buffer offset:0 atIndex:2];
    [enc setBuffer:g_flash_attn_kv_buffer offset:0 atIndex:3];
    [enc setBuffer:mask_buffer offset:0 atIndex:4];
    [enc setBuffer:sinks_buf offset:sinks_offset atIndex:5];
    [enc setBuffer:g_flash_attn_pad_buffer offset:0 atIndex:6];
    [enc setBuffer:g_flash_attn_tmp_buffer offset:0 atIndex:7];
    [enc setThreadgroupMemoryLength:shared_bytes atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake(n_tokens, n_head, nwg)
         threadsPerThreadgroup:MTLSizeMake(32, nsg, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    DS4_METAL_PROFILE_FLASH_ATTN_STAGE("attention_vec");

    ds4_gpu_flash_attn_reduce_args reduce_args = {
        .nrows = (int32_t)nrows,
    };
    enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:reduce_pipeline];
    [enc setBytes:&reduce_args length:sizeof(reduce_args) atIndex:0];
    [enc setBuffer:g_flash_attn_tmp_buffer offset:0 atIndex:1];
    [enc setBuffer:headsbuf offset:ds4_gpu_tensor_offset(heads) atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake(nrows, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(32u * nwg, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    DS4_METAL_PROFILE_FLASH_ATTN_STAGE("attention_reduce");

#undef DS4_METAL_PROFILE_FLASH_ATTN_STAGE
    return 1;
}

static int ds4_gpu_encode_flash_attention_prefill_static_mixed_heads_nonvec(
        id<MTLCommandBuffer> __strong *cbp,
        ds4_gpu_tensor      *heads,
        id<MTLBuffer>          sinks_buf,
        NSUInteger             sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        const ds4_gpu_tensor *comp_kv,
        uint32_t               comp_kv_f16,
        const ds4_gpu_tensor *comp_mask,
        uint32_t               use_comp_mask,
        uint32_t               q_row0,
        uint32_t               n_q,
        uint32_t               n_tokens,
        uint32_t               n_comp,
        uint32_t               window,
        uint32_t               ratio,
        uint32_t               n_head,
        uint32_t               head_dim) {
    /* The vector sibling below only handles the small square case; every
     * rectangular (TP row-split) problem goes through the long path. */
    if (n_tokens >= 20 || q_row0 != 0 || n_q != n_tokens) {
        return ds4_gpu_encode_flash_attention_prefill_static_mixed_heads_nonvec_long(cbp,
                                                                                       heads,
                                                                                       sinks_buf,
                                                                                       sinks_offset,
                                                                                       q,
                                                                                       raw_kv,
                                                                                       comp_kv,
                                                                                       comp_kv_f16,
                                                                                       comp_mask,
                                                                                       use_comp_mask,
                                                                                       q_row0,
                                                                                       n_q,
                                                                                       n_tokens,
                                                                                       n_comp,
                                                                                       window,
                                                                                       ratio,
                                                                                       n_head,
                                                                                       head_dim);
    }
    return ds4_gpu_encode_flash_attention_prefill_static_mixed_heads_vec(cbp,
                                                                           heads,
                                                                           sinks_buf,
                                                                           sinks_offset,
                                                                           q,
                                                                           raw_kv,
                                                                           comp_kv,
                                                                           comp_kv_f16,
                                                                           comp_mask,
                                                                           use_comp_mask,
                                                                           n_tokens,
                                                                           n_comp,
                                                                           window,
                                                                           ratio,
                                                                           n_head,
                                                                           head_dim);
}

/* Raw prefill FlashAttention over a rectangular problem: q holds n_q query
 * rows that correspond to token positions [q_row0, q_row0 + n_q) of the
 * chunk, raw_kv holds all n_kv key rows, and heads receives one output row
 * per query row.  The classic square prefill is q_row0 == 0, n_q == n_kv. */
static int ds4_gpu_encode_flash_attention_prefill_raw_heads_nonvec(
        id<MTLCommandBuffer> __strong *cbp,
        ds4_gpu_tensor      *heads,
        id<MTLBuffer>          sinks_buf,
        NSUInteger             sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        uint32_t               q_row0,
        uint32_t               n_q,
        uint32_t               n_kv,
        uint32_t               window,
        uint32_t               n_head,
        uint32_t               head_dim) {
    if (!cbp || !*cbp) return 0;
    id<MTLCommandBuffer> cb = *cbp;
    if (head_dim != 512 || n_head == 0 || n_q == 0 || n_kv == 0) {
        return 0;
    }

    id<MTLBuffer> qbuf = ds4_gpu_tensor_buffer(q);
    id<MTLBuffer> rawbuf = ds4_gpu_tensor_buffer(raw_kv);
    id<MTLBuffer> headsbuf = ds4_gpu_tensor_buffer(heads);
    const uint64_t q_bytes = (uint64_t)n_q * n_head * head_dim * sizeof(float);
    const uint64_t raw_bytes = (uint64_t)n_kv * head_dim * sizeof(float);
    if (!qbuf || !rawbuf || !headsbuf || !sinks_buf ||
        ds4_gpu_tensor_bytes(q) < q_bytes ||
        ds4_gpu_tensor_bytes(raw_kv) < raw_bytes ||
        ds4_gpu_tensor_bytes(heads) < q_bytes) {
        fprintf(stderr, "ds4: Metal prefill raw DS4 non-vector FlashAttention received undersized buffers\n");
        return 0;
    }

    const uint32_t nqptg = 8;
    const uint32_t ncpsg = 64;
    const uint32_t nsg = head_dim >= 512 ? 8u : 4u;
    const bool has_kvpad = (n_kv % ncpsg) != 0;
    const bool bc_mask = (n_q % nqptg) != 0;
    const NSUInteger row_bytes = (NSUInteger)head_dim * sizeof(float);
    const NSUInteger row_bytes_f16 = (NSUInteger)head_dim * sizeof(uint16_t);
    const NSUInteger mask_bytes = (NSUInteger)n_q * (NSUInteger)n_kv * sizeof(uint16_t);
    const NSUInteger kv_bytes = (NSUInteger)n_kv * row_bytes_f16;
    const NSUInteger pad_bytes = has_kvpad
        ? (NSUInteger)ncpsg * (2u * row_bytes_f16 + (NSUInteger)n_q * sizeof(uint16_t))
        : 1u;
    const NSUInteger nblk0 = ((NSUInteger)n_kv + ncpsg - 1u) / ncpsg;
    const NSUInteger nblk1 = ((NSUInteger)n_q + nqptg - 1u) / nqptg;
    const NSUInteger blk_bytes = ds4_gpu_align_up_ns(nblk0 * nblk1, 32u);

    bool mask_cache_created = false;
    ds4_gpu_zero_prefix_prefill_mask_cache_entry *mask_cache =
        q_row0 == 0u && n_q == n_kv
            ? ds4_gpu_get_zero_prefix_prefill_mask_cache(
                  DS4_GPU_PREFILL_MASK_CACHE_RAW,
                  n_kv,
                  0u,
                  n_kv,
                  window,
                  0u,
                  nqptg,
                  ncpsg,
                  has_kvpad,
                  bc_mask,
                  mask_bytes,
                  blk_bytes,
                  &mask_cache_created)
            : NULL;
    id<MTLBuffer> mask_buffer = mask_cache
        ? mask_cache->mask
        : ds4_gpu_new_transient_buffer(mask_bytes, "ds4_flash_attn_mask");
    if (mask_cache && mask_cache_created) {
        ds4_gpu_fill_raw_prefill_mask((uint16_t *)[mask_buffer contents],
                                      0u, n_kv, n_kv, window);
        mask_cache->valid = true;
    }
    if (!mask_buffer ||
        !ds4_gpu_ensure_scratch_buffer(&g_flash_attn_kv_buffer,
                                         &g_flash_attn_kv_bytes,
                                         kv_bytes,
                                         "ds4_flash_attn_kv_f16") ||
        !ds4_gpu_ensure_scratch_buffer(&g_flash_attn_pad_buffer,
                                         &g_flash_attn_pad_bytes,
                                         pad_bytes,
                                         "ds4_flash_attn_pad") ||
        (!mask_cache &&
         !ds4_gpu_ensure_scratch_buffer(&g_flash_attn_blk_buffer,
                                          &g_flash_attn_blk_bytes,
                                          blk_bytes,
                                          "ds4_flash_attn_blk"))) {
        return 0;
    }
    id<MTLBuffer> blk_buffer = mask_cache
        ? mask_cache->blk
        : g_flash_attn_blk_buffer;

    const bool flash_stage_profile =
        getenv("DS4_METAL_FLASH_ATTN_STAGE_PROFILE") != NULL && g_batch_cb != nil;
    double flash_stage_t0 = 0.0;
    if (flash_stage_profile) {
        if (ds4_gpu_end_commands() == 0 || ds4_gpu_begin_commands() == 0) {
            return 0;
        }
        int profile_owned = 0;
        cb = ds4_gpu_command_buffer(&profile_owned);
        if (!cb || profile_owned) return 0;
        *cbp = cb;
        flash_stage_t0 = ds4_gpu_now_ms();
    }
#define DS4_METAL_PROFILE_FLASH_ATTN_STAGE(name) do { \
        if (flash_stage_profile) { \
            if (!ds4_gpu_flash_attn_stage_profile_boundary(cbp, \
                    "raw_nonvec", (name), n_q, 0, n_kv, \
                    n_head, head_dim, window, 0, &flash_stage_t0)) { \
                return 0; \
            } \
            cb = *cbp; \
        } \
    } while (0)

    if (!mask_cache) {
        ds4_gpu_fill_raw_prefill_mask((uint16_t *)[mask_buffer contents],
                                      q_row0, n_q, n_kv, window);
    }
    DS4_METAL_PROFILE_FLASH_ATTN_STAGE("mask_fill");

    id<MTLComputePipelineState> pad_pipeline = nil;
    if (has_kvpad) {
        pad_pipeline = ds4_gpu_get_flash_attn_pad_pipeline(true, (int32_t)ncpsg);
        if (!pad_pipeline) return 0;
    }
    id<MTLComputePipelineState> blk_pipeline =
        ds4_gpu_get_flash_attn_blk_pipeline((int32_t)nqptg, (int32_t)ncpsg);
    id<MTLComputePipelineState> attn_pipeline =
        ds4_gpu_get_flash_attn_pipeline("kernel_flash_attn_ext_f16_dk512_dv512",
                                          true, true, false, false, has_kvpad, bc_mask,
                                          (int32_t)head_dim,
                                          (int32_t)head_dim,
                                          (int32_t)nsg);
    if (!blk_pipeline || !attn_pipeline) return 0;

    if (!ds4_gpu_encode_cpy_f32_f16_1d(cb,
                                         rawbuf,
                                         ds4_gpu_tensor_offset(raw_kv),
                                         g_flash_attn_kv_buffer,
                                         0,
                                         n_kv * head_dim)) {
        return 0;
    }
    DS4_METAL_PROFILE_FLASH_ATTN_STAGE("copy_raw");

    if (has_kvpad) {
        ds4_gpu_flash_attn_pad_args pad_args = {
            .ne11 = (int32_t)n_kv,
            .ne_12_2 = 1,
            .ne_12_3 = 1,
            .nb11 = row_bytes_f16,
            .nb12 = (uint64_t)n_kv * row_bytes_f16,
            .nb13 = (uint64_t)n_kv * row_bytes_f16,
            .nb21 = row_bytes_f16,
            .nb22 = (uint64_t)n_kv * row_bytes_f16,
            .nb23 = (uint64_t)n_kv * row_bytes_f16,
            .ne31 = (int32_t)n_q,
            .ne32 = 1,
            .ne33 = 1,
            .nb31 = (uint64_t)n_kv * sizeof(uint16_t),
            .nb32 = mask_bytes,
            .nb33 = mask_bytes,
        };

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pad_pipeline];
        [enc setBytes:&pad_args length:sizeof(pad_args) atIndex:0];
        [enc setBuffer:g_flash_attn_kv_buffer offset:0 atIndex:1];
        [enc setBuffer:g_flash_attn_kv_buffer offset:0 atIndex:2];
        [enc setBuffer:mask_buffer offset:0 atIndex:3];
        [enc setBuffer:g_flash_attn_pad_buffer offset:0 atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(ncpsg, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);
        DS4_METAL_PROFILE_FLASH_ATTN_STAGE("pad");
    }

    ds4_gpu_flash_attn_blk_args blk_args = {
        .ne01 = (int32_t)n_q,
        .ne30 = (int32_t)n_kv,
        .ne31 = (int32_t)n_q,
        .ne32 = 1,
        .ne33 = 1,
        .nb31 = (uint64_t)n_kv * sizeof(uint16_t),
        .nb32 = mask_bytes,
        .nb33 = mask_bytes,
    };

    id<MTLComputeCommandEncoder> enc = nil;
    if (!mask_cache || !mask_cache->blk_ready) {
        enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:blk_pipeline];
        [enc setBytes:&blk_args length:sizeof(blk_args) atIndex:0];
        [enc setBuffer:mask_buffer offset:0 atIndex:1];
        [enc setBuffer:blk_buffer offset:0 atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake(nblk0, nblk1, 1)
             threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);
        if (mask_cache) mask_cache->blk_ready = true;
    }
    DS4_METAL_PROFILE_FLASH_ATTN_STAGE("block_map");

    ds4_gpu_flash_attn_vec_args args = {
        .ne01 = (int32_t)n_q,
        .ne02 = (int32_t)n_head,
        .ne03 = 1,
        .nb01 = (uint64_t)n_head * row_bytes,
        .nb02 = row_bytes,
        .nb03 = (uint64_t)n_q * n_head * row_bytes,
        .ne11 = (int32_t)n_kv,
        .ne_12_2 = 1,
        .ne_12_3 = 1,
        .ns10 = (int32_t)head_dim,
        .nb11 = row_bytes_f16,
        .nb12 = (uint64_t)n_kv * row_bytes_f16,
        .nb13 = (uint64_t)n_kv * row_bytes_f16,
        .ns20 = (int32_t)head_dim,
        .nb21 = row_bytes_f16,
        .nb22 = (uint64_t)n_kv * row_bytes_f16,
        .nb23 = (uint64_t)n_kv * row_bytes_f16,
        .ne31 = (int32_t)n_q,
        .ne32 = 1,
        .ne33 = 1,
        .nb31 = (uint64_t)n_kv * sizeof(uint16_t),
        .nb32 = mask_bytes,
        .nb33 = mask_bytes,
        .ne1 = (int32_t)n_head,
        .ne2 = (int32_t)n_q,
        .ne3 = 1,
        .scale = 1.0f / sqrtf((float)head_dim),
        .max_bias = 0.0f,
        .m0 = 0.0f,
        .m1 = 0.0f,
        .n_head_log2 = 0,
        .logit_softcap = 0.0f,
    };

    const NSUInteger padded_v = ds4_gpu_align_up_ns(head_dim, 64u);
    const NSUInteger shared_elems = (NSUInteger)nqptg *
        ((NSUInteger)head_dim + 2u * padded_v + 2u * (2u * (NSUInteger)ncpsg));
    const NSUInteger shared_bytes = ds4_gpu_align_up_ns(shared_elems * (sizeof(float) / 2u), 16u);

    enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:attn_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:qbuf offset:ds4_gpu_tensor_offset(q) atIndex:1];
    [enc setBuffer:g_flash_attn_kv_buffer offset:0 atIndex:2];
    [enc setBuffer:g_flash_attn_kv_buffer offset:0 atIndex:3];
    [enc setBuffer:mask_buffer offset:0 atIndex:4];
    [enc setBuffer:sinks_buf offset:sinks_offset atIndex:5];
    [enc setBuffer:g_flash_attn_pad_buffer offset:0 atIndex:6];
    [enc setBuffer:blk_buffer offset:0 atIndex:7];
    [enc setBuffer:headsbuf offset:ds4_gpu_tensor_offset(heads) atIndex:8];
    [enc setThreadgroupMemoryLength:shared_bytes atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake(nblk1, n_head, 1)
         threadsPerThreadgroup:MTLSizeMake(32, nsg, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    DS4_METAL_PROFILE_FLASH_ATTN_STAGE("attention");

#undef DS4_METAL_PROFILE_FLASH_ATTN_STAGE
    return 1;
}

static int ds4_gpu_encode_flash_attention_prefill_raw_heads(
        id<MTLCommandBuffer> __strong *cbp,
        ds4_gpu_tensor      *heads,
        id<MTLBuffer>          sinks_buf,
        NSUInteger             sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        uint32_t               q_row0,
        uint32_t               n_q,
        uint32_t               n_kv,
        uint32_t               window,
        uint32_t               n_head,
        uint32_t               head_dim) {
    if (!cbp || !*cbp) return 0;
    id<MTLCommandBuffer> cb = *cbp;
    if (head_dim != 512 || n_head == 0 || n_q == 0 || n_kv == 0) {
        return 0;
    }
    /* The vector sibling below only handles the small square case; every
     * rectangular (TP row-split) problem goes through the non-vector path. */
    if (n_kv >= 20 || q_row0 != 0 || n_q != n_kv) {
        return ds4_gpu_encode_flash_attention_prefill_raw_heads_nonvec(cbp,
                                                                         heads,
                                                                         sinks_buf,
                                                                         sinks_offset,
                                                                         q,
                                                                         raw_kv,
                                                                         q_row0,
                                                                         n_q,
                                                                         n_kv,
                                                                         window,
                                                                         n_head,
                                                                         head_dim);
    }
    const uint32_t n_tokens = n_q;

    id<MTLBuffer> qbuf = ds4_gpu_tensor_buffer(q);
    id<MTLBuffer> rawbuf = ds4_gpu_tensor_buffer(raw_kv);
    id<MTLBuffer> headsbuf = ds4_gpu_tensor_buffer(heads);
    const uint64_t q_bytes = (uint64_t)n_tokens * n_head * head_dim * sizeof(float);
    const uint64_t raw_bytes = (uint64_t)n_tokens * head_dim * sizeof(float);
    if (!qbuf || !rawbuf || !headsbuf || !sinks_buf ||
        ds4_gpu_tensor_bytes(q) < q_bytes ||
        ds4_gpu_tensor_bytes(raw_kv) < raw_bytes ||
        ds4_gpu_tensor_bytes(heads) < q_bytes) {
        fprintf(stderr, "ds4: Metal prefill raw DS4 FlashAttention received undersized buffers\n");
        return 0;
    }

    const uint32_t ncpsg = 32;
    const uint32_t nwg = 32;
    const uint32_t nsg = ds4_gpu_flash_attn_vec_nsg(n_tokens, nwg, ncpsg);
    const NSUInteger row_bytes = (NSUInteger)head_dim * sizeof(float);
    const NSUInteger row_bytes_f16 = (NSUInteger)head_dim * sizeof(uint16_t);
    const NSUInteger mask_bytes = (NSUInteger)n_tokens * (NSUInteger)n_tokens * sizeof(uint16_t);
    const NSUInteger kv_f16_offset = 0;
    const NSUInteger kv_f16_bytes = (NSUInteger)n_tokens * row_bytes_f16;
    const NSUInteger pad_bytes = 2u * (NSUInteger)ncpsg * row_bytes_f16 +
                                 (NSUInteger)ncpsg * (NSUInteger)n_tokens * sizeof(uint16_t);
    const NSUInteger nrows = (NSUInteger)n_tokens * n_head;
    const NSUInteger tmp_bytes = nrows * (NSUInteger)head_dim * (NSUInteger)nwg * sizeof(float) +
                                 nrows * (2u * (NSUInteger)nwg) * sizeof(float);

    id<MTLBuffer> mask_buffer =
        ds4_gpu_new_transient_buffer(mask_bytes, "ds4_flash_attn_mask");
    if (!mask_buffer ||
        !ds4_gpu_ensure_scratch_buffer(&g_flash_attn_pad_buffer,
                                         &g_flash_attn_pad_bytes,
                                         pad_bytes,
                                         "ds4_flash_attn_pad") ||
        !ds4_gpu_ensure_scratch_buffer(&g_flash_attn_kv_buffer,
                                         &g_flash_attn_kv_bytes,
                                         kv_f16_bytes,
                                         "ds4_flash_attn_kv_f16") ||
        !ds4_gpu_ensure_scratch_buffer(&g_flash_attn_tmp_buffer,
                                         &g_flash_attn_tmp_bytes,
                                         tmp_bytes,
                                         "ds4_flash_attn_tmp")) {
        return 0;
    }

    const bool flash_stage_profile =
        getenv("DS4_METAL_FLASH_ATTN_STAGE_PROFILE") != NULL && g_batch_cb != nil;
    double flash_stage_t0 = 0.0;
    if (flash_stage_profile) {
        if (ds4_gpu_end_commands() == 0 || ds4_gpu_begin_commands() == 0) {
            return 0;
        }
        int profile_owned = 0;
        cb = ds4_gpu_command_buffer(&profile_owned);
        if (!cb || profile_owned) return 0;
        *cbp = cb;
        flash_stage_t0 = ds4_gpu_now_ms();
    }
#define DS4_METAL_PROFILE_FLASH_ATTN_STAGE(name) do { \
        if (flash_stage_profile) { \
            if (!ds4_gpu_flash_attn_stage_profile_boundary(cbp, \
                    "raw_vec", (name), n_tokens, 0, n_tokens, \
                    n_head, head_dim, window, 0, &flash_stage_t0)) { \
                return 0; \
            } \
            cb = *cbp; \
        } \
    } while (0)

    ds4_gpu_fill_raw_prefill_mask((uint16_t *)[mask_buffer contents], 0, n_tokens, n_tokens, window);
    DS4_METAL_PROFILE_FLASH_ATTN_STAGE("mask_fill");

    id<MTLComputePipelineState> pad_pipeline = nil;
    if ((n_tokens % ncpsg) != 0) {
        pad_pipeline = ds4_gpu_get_flash_attn_pad_pipeline(true, (int32_t)ncpsg);
        if (!pad_pipeline) return 0;
    }
    id<MTLComputePipelineState> vec_pipeline =
        ds4_gpu_get_flash_attn_vec_pipeline("kernel_flash_attn_ext_vec_f16_dk512_dv512",
                                              true, true, false, false, true,
                                              false,
                                              (int32_t)head_dim,
                                              (int32_t)head_dim,
                                              (int32_t)nsg,
                                              (int32_t)nwg);
    id<MTLComputePipelineState> reduce_pipeline =
        ds4_gpu_get_flash_attn_reduce_pipeline((int32_t)head_dim, (int32_t)nwg);
    if (!vec_pipeline || !reduce_pipeline) return 0;

    if (!ds4_gpu_encode_cpy_f32_f16_1d(cb,
                                         rawbuf,
                                         ds4_gpu_tensor_offset(raw_kv),
                                         g_flash_attn_kv_buffer,
                                         kv_f16_offset,
                                         n_tokens * head_dim)) {
        return 0;
    }
    DS4_METAL_PROFILE_FLASH_ATTN_STAGE("copy_raw");

    if ((n_tokens % ncpsg) != 0) {
        ds4_gpu_flash_attn_pad_args pad_args = {
            .ne11 = (int32_t)n_tokens,
            .ne_12_2 = 1,
            .ne_12_3 = 1,
            .nb11 = row_bytes_f16,
            .nb12 = (uint64_t)n_tokens * row_bytes_f16,
            .nb13 = (uint64_t)n_tokens * row_bytes_f16,
            .nb21 = row_bytes_f16,
            .nb22 = (uint64_t)n_tokens * row_bytes_f16,
            .nb23 = (uint64_t)n_tokens * row_bytes_f16,
            .ne31 = (int32_t)n_tokens,
            .ne32 = 1,
            .ne33 = 1,
            .nb31 = (uint64_t)n_tokens * sizeof(uint16_t),
            .nb32 = mask_bytes,
            .nb33 = mask_bytes,
        };

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pad_pipeline];
        [enc setBytes:&pad_args length:sizeof(pad_args) atIndex:0];
        [enc setBuffer:g_flash_attn_kv_buffer offset:kv_f16_offset atIndex:1];
        [enc setBuffer:g_flash_attn_kv_buffer offset:kv_f16_offset atIndex:2];
        [enc setBuffer:mask_buffer offset:0 atIndex:3];
        [enc setBuffer:g_flash_attn_pad_buffer offset:0 atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(ncpsg, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);
        DS4_METAL_PROFILE_FLASH_ATTN_STAGE("pad");
    }

    ds4_gpu_flash_attn_vec_args vec_args = {
        .ne01 = (int32_t)n_tokens,
        .ne02 = (int32_t)n_head,
        .ne03 = 1,
        .nb01 = (uint64_t)n_head * row_bytes,
        .nb02 = row_bytes,
        .nb03 = (uint64_t)n_tokens * n_head * row_bytes,
        .ne11 = (int32_t)n_tokens,
        .ne_12_2 = 1,
        .ne_12_3 = 1,
        .ns10 = (int32_t)head_dim,
        .nb11 = row_bytes_f16,
        .nb12 = (uint64_t)n_tokens * row_bytes_f16,
        .nb13 = (uint64_t)n_tokens * row_bytes_f16,
        .ns20 = (int32_t)head_dim,
        .nb21 = row_bytes_f16,
        .nb22 = (uint64_t)n_tokens * row_bytes_f16,
        .nb23 = (uint64_t)n_tokens * row_bytes_f16,
        .ne31 = (int32_t)n_tokens,
        .ne32 = 1,
        .ne33 = 1,
        .nb31 = (uint64_t)n_tokens * sizeof(uint16_t),
        .nb32 = mask_bytes,
        .nb33 = mask_bytes,
        .ne1 = (int32_t)n_head,
        .ne2 = (int32_t)n_tokens,
        .ne3 = 1,
        .scale = 1.0f / sqrtf((float)head_dim),
        .max_bias = 0.0f,
        .m0 = 0.0f,
        .m1 = 0.0f,
        .n_head_log2 = 0,
        .logit_softcap = 0.0f,
    };

    const NSUInteger shared_elems = (ds4_gpu_align_up_ns(head_dim, 128u) +
                                     4u * ncpsg +
                                     2u * ds4_gpu_align_up_ns(head_dim, 128u)) * nsg;
    const NSUInteger shared_bytes = ds4_gpu_align_up_ns(shared_elems * (sizeof(float) / 2u), 16u);

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:vec_pipeline];
    [enc setBytes:&vec_args length:sizeof(vec_args) atIndex:0];
    [enc setBuffer:qbuf offset:ds4_gpu_tensor_offset(q) atIndex:1];
    [enc setBuffer:g_flash_attn_kv_buffer offset:kv_f16_offset atIndex:2];
    [enc setBuffer:g_flash_attn_kv_buffer offset:kv_f16_offset atIndex:3];
    [enc setBuffer:mask_buffer offset:0 atIndex:4];
    [enc setBuffer:sinks_buf offset:sinks_offset atIndex:5];
    [enc setBuffer:g_flash_attn_pad_buffer offset:0 atIndex:6];
    [enc setBuffer:g_flash_attn_tmp_buffer offset:0 atIndex:7];
    [enc setThreadgroupMemoryLength:shared_bytes atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake(n_tokens, n_head, nwg)
         threadsPerThreadgroup:MTLSizeMake(32, nsg, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    DS4_METAL_PROFILE_FLASH_ATTN_STAGE("attention_vec");

    ds4_gpu_flash_attn_reduce_args reduce_args = {
        .nrows = (int32_t)nrows,
    };
    enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:reduce_pipeline];
    [enc setBytes:&reduce_args length:sizeof(reduce_args) atIndex:0];
    [enc setBuffer:g_flash_attn_tmp_buffer offset:0 atIndex:1];
    [enc setBuffer:headsbuf offset:ds4_gpu_tensor_offset(heads) atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake(nrows, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(32u * nwg, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    DS4_METAL_PROFILE_FLASH_ATTN_STAGE("attention_reduce");

#undef DS4_METAL_PROFILE_FLASH_ATTN_STAGE
    return 1;
}

static int ds4_gpu_encode_flash_attention_gathered_heads(
        id<MTLCommandBuffer>   cb,
        ds4_gpu_tensor      *heads,
        id<MTLBuffer>          sinks_buf,
        NSUInteger             sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        uint32_t               n_raw,
        uint32_t               raw_cap,
        uint32_t               raw_start,
        const ds4_gpu_tensor *comp_kv,
        uint32_t               comp_kv_f16,
        uint32_t               n_comp,
        const ds4_gpu_tensor *comp_mask,
        uint32_t               use_mask,
        uint32_t               n_head,
        uint32_t               head_dim) {
    const uint32_t n_keys = n_raw + n_comp;
    if (head_dim != 512 || n_head == 0 || n_raw == 0 || n_keys == 0 ||
        raw_cap < n_raw || n_keys < n_raw) {
        return 0;
    }

    id<MTLBuffer> qbuf = ds4_gpu_tensor_buffer(q);
    id<MTLBuffer> rawbuf = ds4_gpu_tensor_buffer(raw_kv);
    id<MTLBuffer> compbuf = n_comp ? ds4_gpu_tensor_buffer(comp_kv) : nil;
    id<MTLBuffer> headsbuf = ds4_gpu_tensor_buffer(heads);
    id<MTLBuffer> maskbuf = use_mask ? ds4_gpu_tensor_buffer(comp_mask) : nil;
    const uint64_t q_bytes = (uint64_t)n_head * head_dim * sizeof(float);
    const uint64_t raw_bytes = (uint64_t)raw_cap * head_dim * sizeof(float);
    const uint64_t comp_bytes = (uint64_t)n_comp * head_dim *
                                (comp_kv_f16 ? sizeof(uint16_t) : sizeof(float));
    const uint64_t comp_mask_bytes = use_mask ? (uint64_t)n_comp * sizeof(float) : 0u;
    if (!qbuf || !rawbuf || !headsbuf || !sinks_buf ||
        (n_comp && !compbuf) ||
        (use_mask && !maskbuf) ||
        ds4_gpu_tensor_bytes(q) < q_bytes ||
        ds4_gpu_tensor_bytes(raw_kv) < raw_bytes ||
        (n_comp && ds4_gpu_tensor_bytes(comp_kv) < comp_bytes) ||
        ds4_gpu_tensor_bytes(heads) < q_bytes ||
        (use_mask && ds4_gpu_tensor_bytes(comp_mask) < comp_mask_bytes)) {
        fprintf(stderr, "ds4: Metal gathered DS4 FlashAttention received undersized buffers\n");
        return 0;
    }

    const uint32_t ncpsg = 32;
    const uint32_t nwg = 32;
    const uint32_t nsg = ds4_gpu_flash_attn_vec_nsg(n_keys, nwg, ncpsg);
    const NSUInteger row_bytes = (NSUInteger)head_dim * sizeof(float);
    const NSUInteger row_bytes_f16 = (NSUInteger)head_dim * sizeof(uint16_t);
    const NSUInteger mask_bytes = (NSUInteger)n_keys * sizeof(uint16_t);
    const NSUInteger kv_bytes = (NSUInteger)n_keys * row_bytes_f16;
    const NSUInteger pad_bytes = 2u * (NSUInteger)ncpsg * row_bytes_f16 +
                                 (NSUInteger)ncpsg * sizeof(uint16_t);
    const NSUInteger nrows = (NSUInteger)n_head;
    const NSUInteger tmp_bytes = nrows * (NSUInteger)head_dim * (NSUInteger)nwg * sizeof(float) +
                                 nrows * (2u * (NSUInteger)nwg) * sizeof(float);

    const bool use_persistent_zero_mask =
        use_mask == 0u &&
        (ds4_gpu_device_name_contains("M3") ||
         getenv("DS4_METAL_ENABLE_PERSISTENT_ZERO_ATTN_MASK") != NULL) &&
        getenv("DS4_METAL_DISABLE_M3_PERSISTENT_ZERO_ATTN_MASK") == NULL;

    if (!(use_persistent_zero_mask
              ? ds4_gpu_ensure_zero_attention_mask(mask_bytes)
              : ds4_gpu_ensure_scratch_buffer(&g_flash_attn_mask_buffer,
                                                &g_flash_attn_mask_bytes,
                                                mask_bytes,
                                                "ds4_flash_attn_mask")) ||
        !ds4_gpu_ensure_scratch_buffer(&g_flash_attn_kv_buffer,
                                         &g_flash_attn_kv_bytes,
                                         kv_bytes,
                                         "ds4_flash_attn_kv") ||
        !ds4_gpu_ensure_scratch_buffer(&g_flash_attn_pad_buffer,
                                         &g_flash_attn_pad_bytes,
                                         pad_bytes,
                                         "ds4_flash_attn_pad") ||
        !ds4_gpu_ensure_scratch_buffer(&g_flash_attn_tmp_buffer,
                                         &g_flash_attn_tmp_bytes,
                                         tmp_bytes,
                                         "ds4_flash_attn_tmp")) {
        return 0;
    }
    id<MTLBuffer> flash_mask_buffer = use_persistent_zero_mask
        ? g_flash_attn_zero_mask_buffer
        : g_flash_attn_mask_buffer;

    const bool has_kvpad = (n_keys % ncpsg) != 0;
    const bool use_shared_kvpad =
        has_kvpad &&
        (ds4_gpu_device_name_contains("M3") ||
         getenv("DS4_METAL_ENABLE_SHARED_KV_PAD") != NULL) &&
        getenv("DS4_METAL_DISABLE_M3_SHARED_KV_PAD") == NULL;
    id<MTLComputePipelineState> vec_pipeline =
        ds4_gpu_get_flash_attn_vec_pipeline("kernel_flash_attn_ext_vec_f16_dk512_dv512",
                                              true, true, false, false, has_kvpad,
                                              use_shared_kvpad,
                                              (int32_t)head_dim,
                                              (int32_t)head_dim,
                                              (int32_t)nsg,
                                              (int32_t)nwg);
    id<MTLComputePipelineState> reduce_pipeline =
        ds4_gpu_get_flash_attn_reduce_pipeline((int32_t)head_dim, (int32_t)nwg);
    if (!vec_pipeline || !reduce_pipeline) return 0;

    if (!use_persistent_zero_mask &&
        !ds4_gpu_encode_fill_f16_1d(cb, flash_mask_buffer, 0, n_keys, 0.0f)) {
        return 0;
    }
    if (use_mask && n_comp &&
        !ds4_gpu_encode_cpy_f32_f16_1d(cb,
                                         maskbuf,
                                         ds4_gpu_tensor_offset(comp_mask),
                                         flash_mask_buffer,
                                         (NSUInteger)n_raw * sizeof(uint16_t),
                                         n_comp)) {
        return 0;
    }

    bool pad_fused = false;
    if (!ds4_gpu_encode_flash_kv_stage_f16(
            cb,
            rawbuf,
            ds4_gpu_tensor_offset(raw_kv),
            raw_cap,
            raw_start,
            n_raw,
            compbuf,
            ds4_gpu_tensor_offset(comp_kv),
            comp_kv_f16 != 0,
            n_comp,
            head_dim,
            g_flash_attn_kv_buffer,
            0,
            flash_mask_buffer,
            0,
            g_flash_attn_pad_buffer,
            0,
            has_kvpad,
            use_shared_kvpad,
            &pad_fused)) {
        return 0;
    }

    id<MTLComputePipelineState> pad_pipeline = nil;
    if (has_kvpad && !pad_fused) {
        pad_pipeline = ds4_gpu_get_flash_attn_pad_pipeline(true, (int32_t)ncpsg);
        if (!pad_pipeline) return 0;
    }

    if (has_kvpad && !pad_fused) {
        ds4_gpu_flash_attn_pad_args pad_args = {
            .ne11 = (int32_t)n_keys,
            .ne_12_2 = 1,
            .ne_12_3 = 1,
            .nb11 = row_bytes_f16,
            .nb12 = (uint64_t)n_keys * row_bytes_f16,
            .nb13 = (uint64_t)n_keys * row_bytes_f16,
            .nb21 = row_bytes_f16,
            .nb22 = (uint64_t)n_keys * row_bytes_f16,
            .nb23 = (uint64_t)n_keys * row_bytes_f16,
            .ne31 = 1,
            .ne32 = 1,
            .ne33 = 1,
            .nb31 = mask_bytes,
            .nb32 = mask_bytes,
            .nb33 = mask_bytes,
        };

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pad_pipeline];
        [enc setBytes:&pad_args length:sizeof(pad_args) atIndex:0];
        [enc setBuffer:g_flash_attn_kv_buffer offset:0 atIndex:1];
        [enc setBuffer:g_flash_attn_kv_buffer offset:0 atIndex:2];
        [enc setBuffer:flash_mask_buffer offset:0 atIndex:3];
        [enc setBuffer:g_flash_attn_pad_buffer offset:0 atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(ncpsg, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);
    }

    ds4_gpu_flash_attn_vec_args vec_args = {
        .ne01 = 1,
        .ne02 = (int32_t)n_head,
        .ne03 = 1,
        .nb01 = (uint64_t)n_head * row_bytes,
        .nb02 = row_bytes,
        .nb03 = (uint64_t)n_head * row_bytes,
        .ne11 = (int32_t)n_keys,
        .ne_12_2 = 1,
        .ne_12_3 = 1,
        .ns10 = (int32_t)head_dim,
        .nb11 = row_bytes_f16,
        .nb12 = (uint64_t)n_keys * row_bytes_f16,
        .nb13 = (uint64_t)n_keys * row_bytes_f16,
        .ns20 = (int32_t)head_dim,
        .nb21 = row_bytes_f16,
        .nb22 = (uint64_t)n_keys * row_bytes_f16,
        .nb23 = (uint64_t)n_keys * row_bytes_f16,
        .ne31 = 1,
        .ne32 = 1,
        .ne33 = 1,
        .nb31 = mask_bytes,
        .nb32 = mask_bytes,
        .nb33 = mask_bytes,
        .ne1 = (int32_t)n_head,
        .ne2 = 1,
        .ne3 = 1,
        .scale = 1.0f / sqrtf((float)head_dim),
        .max_bias = 0.0f,
        .m0 = 0.0f,
        .m1 = 0.0f,
        .n_head_log2 = 0,
        .logit_softcap = 0.0f,
    };

    const NSUInteger shared_elems = (ds4_gpu_align_up_ns(head_dim, 128u) +
                                     4u * ncpsg +
                                     2u * ds4_gpu_align_up_ns(head_dim, 128u)) * nsg;
    const NSUInteger shared_bytes = ds4_gpu_align_up_ns(shared_elems * (sizeof(float) / 2u), 16u);

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:vec_pipeline];
    [enc setBytes:&vec_args length:sizeof(vec_args) atIndex:0];
    [enc setBuffer:qbuf offset:ds4_gpu_tensor_offset(q) atIndex:1];
    [enc setBuffer:g_flash_attn_kv_buffer offset:0 atIndex:2];
    [enc setBuffer:g_flash_attn_kv_buffer offset:0 atIndex:3];
    [enc setBuffer:flash_mask_buffer offset:0 atIndex:4];
    [enc setBuffer:sinks_buf offset:sinks_offset atIndex:5];
    [enc setBuffer:g_flash_attn_pad_buffer offset:0 atIndex:6];
    [enc setBuffer:g_flash_attn_tmp_buffer offset:0 atIndex:7];
    [enc setThreadgroupMemoryLength:shared_bytes atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake(1, n_head, nwg)
         threadsPerThreadgroup:MTLSizeMake(32, nsg, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);

    ds4_gpu_flash_attn_reduce_args reduce_args = {
        .nrows = (int32_t)nrows,
    };
    enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:reduce_pipeline];
    [enc setBytes:&reduce_args length:sizeof(reduce_args) atIndex:0];
    [enc setBuffer:g_flash_attn_tmp_buffer offset:0 atIndex:1];
    [enc setBuffer:headsbuf offset:ds4_gpu_tensor_offset(heads) atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake(nrows, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(32u * nwg, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);

    return 1;
}

static int ds4_gpu_encode_flash_attention_decode_raw_batch_heads(
        id<MTLCommandBuffer>   cb,
        ds4_gpu_tensor      *heads,
        id<MTLBuffer>          sinks_buf,
        NSUInteger             sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        uint32_t               n_tokens,
        uint32_t               pos0,
        uint32_t               n_raw,
        uint32_t               raw_cap,
        uint32_t               raw_start,
        uint32_t               window,
        uint32_t               n_head,
        uint32_t               head_dim,
        bool                   noncausal) {
    if (head_dim != 512 || n_head == 0 || n_tokens == 0 ||
        n_raw == 0 || raw_cap < n_raw || raw_start >= raw_cap) {
        return 0;
    }

    id<MTLBuffer> qbuf = ds4_gpu_tensor_buffer(q);
    id<MTLBuffer> rawbuf = ds4_gpu_tensor_buffer(raw_kv);
    id<MTLBuffer> headsbuf = ds4_gpu_tensor_buffer(heads);
    const uint64_t q_bytes = (uint64_t)n_tokens * n_head * head_dim * sizeof(float);
    const uint64_t raw_bytes = (uint64_t)raw_cap * head_dim * sizeof(float);
    if (!qbuf || !rawbuf || !headsbuf || !sinks_buf ||
        ds4_gpu_tensor_bytes(q) < q_bytes ||
        ds4_gpu_tensor_bytes(raw_kv) < raw_bytes ||
        ds4_gpu_tensor_bytes(heads) < q_bytes) {
        fprintf(stderr, "ds4: Metal decode raw batch FlashAttention received undersized buffers\n");
        return 0;
    }

    const uint32_t nqptg = 8;
    const uint32_t ncpsg = 64;
    const uint32_t nsg = head_dim >= 512 ? 8u : 4u;
    const bool has_kvpad = (n_raw % ncpsg) != 0;
    const bool bc_mask = (n_tokens % nqptg) != 0;
    const NSUInteger row_bytes = (NSUInteger)head_dim * sizeof(float);
    const NSUInteger row_bytes_f16 = (NSUInteger)head_dim * sizeof(uint16_t);
    const NSUInteger mask_bytes = (NSUInteger)n_raw * (NSUInteger)n_tokens * sizeof(uint16_t);
    const NSUInteger kv_bytes = (NSUInteger)n_raw * row_bytes_f16;
    const NSUInteger pad_bytes = has_kvpad
        ? (NSUInteger)ncpsg * (2u * row_bytes_f16 + (NSUInteger)n_tokens * sizeof(uint16_t))
        : 1u;
    const NSUInteger nblk0 = ((NSUInteger)n_raw + ncpsg - 1u) / ncpsg;
    const NSUInteger nblk1 = ((NSUInteger)n_tokens + nqptg - 1u) / nqptg;
    const NSUInteger blk_bytes = ds4_gpu_align_up_ns(nblk0 * nblk1, 32u);

    id<MTLBuffer> mask_buffer =
        ds4_gpu_new_transient_buffer(mask_bytes, "ds4_flash_attn_mask");
    if (!mask_buffer ||
        !ds4_gpu_ensure_scratch_buffer(&g_flash_attn_kv_buffer,
                                         &g_flash_attn_kv_bytes,
                                         kv_bytes,
                                         "ds4_flash_attn_kv_f16") ||
        !ds4_gpu_ensure_scratch_buffer(&g_flash_attn_pad_buffer,
                                         &g_flash_attn_pad_bytes,
                                         pad_bytes,
                                         "ds4_flash_attn_pad") ||
        !ds4_gpu_ensure_scratch_buffer(&g_flash_attn_blk_buffer,
                                         &g_flash_attn_blk_bytes,
                                         blk_bytes,
                                         "ds4_flash_attn_blk")) {
        return 0;
    }

    if (!ds4_gpu_encode_copy_raw_ring_to_f16(cb,
                                              rawbuf,
                                              ds4_gpu_tensor_offset(raw_kv),
                                              raw_cap,
                                              raw_start,
                                              n_raw,
                                              head_dim,
                                              g_flash_attn_kv_buffer,
                                              0)) {
        return 0;
    }

    if (noncausal) {
        memset([mask_buffer contents], 0, mask_bytes);
    } else {
        ds4_gpu_fill_raw_decode_batch_mask((uint16_t *)[mask_buffer contents],
                                             n_tokens,
                                             n_raw,
                                             pos0,
                                             window);
    }

    id<MTLComputePipelineState> pad_pipeline = nil;
    if (has_kvpad) {
        pad_pipeline = ds4_gpu_get_flash_attn_pad_pipeline(true, (int32_t)ncpsg);
        if (!pad_pipeline) return 0;
    }
    id<MTLComputePipelineState> blk_pipeline =
        ds4_gpu_get_flash_attn_blk_pipeline((int32_t)nqptg, (int32_t)ncpsg);
    id<MTLComputePipelineState> attn_pipeline =
        ds4_gpu_get_flash_attn_pipeline("kernel_flash_attn_ext_f16_dk512_dv512",
                                          true, true, false, false, has_kvpad, bc_mask,
                                          (int32_t)head_dim,
                                          (int32_t)head_dim,
                                          (int32_t)nsg);
    if (!blk_pipeline || !attn_pipeline) return 0;

    if (has_kvpad) {
        ds4_gpu_flash_attn_pad_args pad_args = {
            .ne11 = (int32_t)n_raw,
            .ne_12_2 = 1,
            .ne_12_3 = 1,
            .nb11 = row_bytes_f16,
            .nb12 = (uint64_t)n_raw * row_bytes_f16,
            .nb13 = (uint64_t)n_raw * row_bytes_f16,
            .nb21 = row_bytes_f16,
            .nb22 = (uint64_t)n_raw * row_bytes_f16,
            .nb23 = (uint64_t)n_raw * row_bytes_f16,
            .ne31 = (int32_t)n_tokens,
            .ne32 = 1,
            .ne33 = 1,
            .nb31 = (uint64_t)n_raw * sizeof(uint16_t),
            .nb32 = mask_bytes,
            .nb33 = mask_bytes,
        };

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pad_pipeline];
        [enc setBytes:&pad_args length:sizeof(pad_args) atIndex:0];
        [enc setBuffer:g_flash_attn_kv_buffer offset:0 atIndex:1];
        [enc setBuffer:g_flash_attn_kv_buffer offset:0 atIndex:2];
        [enc setBuffer:mask_buffer offset:0 atIndex:3];
        [enc setBuffer:g_flash_attn_pad_buffer offset:0 atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(ncpsg, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);
    }

    ds4_gpu_flash_attn_blk_args blk_args = {
        .ne01 = (int32_t)n_tokens,
        .ne30 = (int32_t)n_raw,
        .ne31 = (int32_t)n_tokens,
        .ne32 = 1,
        .ne33 = 1,
        .nb31 = (uint64_t)n_raw * sizeof(uint16_t),
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
        .nb01 = (uint64_t)n_head * row_bytes,
        .nb02 = row_bytes,
        .nb03 = (uint64_t)n_tokens * n_head * row_bytes,
        .ne11 = (int32_t)n_raw,
        .ne_12_2 = 1,
        .ne_12_3 = 1,
        .ns10 = (int32_t)head_dim,
        .nb11 = row_bytes_f16,
        .nb12 = (uint64_t)n_raw * row_bytes_f16,
        .nb13 = (uint64_t)n_raw * row_bytes_f16,
        .ns20 = (int32_t)head_dim,
        .nb21 = row_bytes_f16,
        .nb22 = (uint64_t)n_raw * row_bytes_f16,
        .nb23 = (uint64_t)n_raw * row_bytes_f16,
        .ne31 = (int32_t)n_tokens,
        .ne32 = 1,
        .ne33 = 1,
        .nb31 = (uint64_t)n_raw * sizeof(uint16_t),
        .nb32 = mask_bytes,
        .nb33 = mask_bytes,
        .ne1 = (int32_t)n_head,
        .ne2 = (int32_t)n_tokens,
        .ne3 = 1,
        .scale = 1.0f / sqrtf((float)head_dim),
        .max_bias = 0.0f,
        .m0 = 0.0f,
        .m1 = 0.0f,
        .n_head_log2 = 0,
        .logit_softcap = 0.0f,
    };

    const NSUInteger padded_v = ds4_gpu_align_up_ns(head_dim, 64u);
    const NSUInteger shared_elems = (NSUInteger)nqptg *
        ((NSUInteger)head_dim + 2u * padded_v + 2u * (2u * (NSUInteger)ncpsg));
    const NSUInteger shared_bytes = ds4_gpu_align_up_ns(shared_elems * (sizeof(float) / 2u), 16u);

    enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:attn_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:qbuf offset:ds4_gpu_tensor_offset(q) atIndex:1];
    [enc setBuffer:g_flash_attn_kv_buffer offset:0 atIndex:2];
    [enc setBuffer:g_flash_attn_kv_buffer offset:0 atIndex:3];
    [enc setBuffer:mask_buffer offset:0 atIndex:4];
    [enc setBuffer:sinks_buf offset:sinks_offset atIndex:5];
    [enc setBuffer:g_flash_attn_pad_buffer offset:0 atIndex:6];
    [enc setBuffer:g_flash_attn_blk_buffer offset:0 atIndex:7];
    [enc setBuffer:headsbuf offset:ds4_gpu_tensor_offset(heads) atIndex:8];
    [enc setThreadgroupMemoryLength:shared_bytes atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake(nblk1, n_head, 1)
         threadsPerThreadgroup:MTLSizeMake(32, nsg, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);

    return 1;
}

static int ds4_gpu_encode_flash_attention_decode_mixed_batch_heads(
        id<MTLCommandBuffer>   cb,
        ds4_gpu_tensor      *heads,
        id<MTLBuffer>          sinks_buf,
        NSUInteger             sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        const ds4_gpu_tensor *comp_kv,
        uint32_t               comp_kv_f16,
        const ds4_gpu_tensor *comp_mask,
        uint32_t               use_comp_mask,
        uint32_t               n_tokens,
        uint32_t               pos0,
        uint32_t               n_raw,
        uint32_t               raw_cap,
        uint32_t               raw_start,
        uint32_t               n_comp,
        uint32_t               window,
        uint32_t               ratio,
        uint32_t               n_head,
        uint32_t               head_dim) {
    if (n_comp == 0) {
        return ds4_gpu_encode_flash_attention_decode_raw_batch_heads(cb,
                                                                       heads,
                                                                       sinks_buf,
                                                                       sinks_offset,
                                                                       q,
                                                                       raw_kv,
                                                                       n_tokens,
                                                                       pos0,
                                                                       n_raw,
                                                                       raw_cap,
                                                                       raw_start,
                                                                       window,
                                                                       n_head,
                                                                       head_dim,
                                                                       false);
    }
    if (head_dim != 512 || n_head == 0 || n_tokens == 0 ||
        n_raw == 0 || raw_cap < n_raw || raw_start >= raw_cap ||
        ratio == 0 || !comp_kv || (use_comp_mask && !comp_mask)) {
        return 0;
    }

    const uint32_t n_keys = n_raw + n_comp;
    id<MTLBuffer> qbuf = ds4_gpu_tensor_buffer(q);
    id<MTLBuffer> rawbuf = ds4_gpu_tensor_buffer(raw_kv);
    id<MTLBuffer> compbuf = ds4_gpu_tensor_buffer(comp_kv);
    id<MTLBuffer> maskbuf = use_comp_mask ? ds4_gpu_tensor_buffer(comp_mask) : rawbuf;
    id<MTLBuffer> headsbuf = ds4_gpu_tensor_buffer(heads);
    const uint64_t q_bytes = (uint64_t)n_tokens * n_head * head_dim * sizeof(float);
    const uint64_t raw_bytes = (uint64_t)raw_cap * head_dim * sizeof(float);
    const uint64_t comp_bytes = (uint64_t)n_comp * head_dim *
                                (comp_kv_f16 ? sizeof(uint16_t) : sizeof(float));
    const uint64_t comp_mask_bytes = use_comp_mask ? (uint64_t)n_comp * n_tokens * sizeof(float) : 0u;
    if (!qbuf || !rawbuf || !compbuf || !maskbuf || !headsbuf || !sinks_buf ||
        ds4_gpu_tensor_bytes(q) < q_bytes ||
        ds4_gpu_tensor_bytes(raw_kv) < raw_bytes ||
        ds4_gpu_tensor_bytes(comp_kv) < comp_bytes ||
        (use_comp_mask && ds4_gpu_tensor_bytes(comp_mask) < comp_mask_bytes) ||
        ds4_gpu_tensor_bytes(heads) < q_bytes) {
        fprintf(stderr, "ds4: Metal decode mixed batch FlashAttention received undersized buffers\n");
        return 0;
    }

    const uint32_t nqptg = 8;
    const uint32_t ncpsg = 64;
    const uint32_t nsg = head_dim >= 512 ? 8u : 4u;
    const bool has_kvpad = (n_keys % ncpsg) != 0;
    const bool bc_mask = (n_tokens % nqptg) != 0;
    const NSUInteger row_bytes = (NSUInteger)head_dim * sizeof(float);
    const NSUInteger row_bytes_f16 = (NSUInteger)head_dim * sizeof(uint16_t);
    const NSUInteger mask_bytes = (NSUInteger)n_keys * (NSUInteger)n_tokens * sizeof(uint16_t);
    const NSUInteger kv_bytes = (NSUInteger)n_keys * row_bytes_f16;
    const NSUInteger pad_bytes = has_kvpad
        ? (NSUInteger)ncpsg * (2u * row_bytes_f16 + (NSUInteger)n_tokens * sizeof(uint16_t))
        : 1u;
    const NSUInteger nblk0 = ((NSUInteger)n_keys + ncpsg - 1u) / ncpsg;
    const NSUInteger nblk1 = ((NSUInteger)n_tokens + nqptg - 1u) / nqptg;
    const NSUInteger blk_bytes = ds4_gpu_align_up_ns(nblk0 * nblk1, 32u);

    id<MTLBuffer> mask_buffer =
        ds4_gpu_new_transient_buffer(mask_bytes, "ds4_flash_attn_mask");
    if (!mask_buffer ||
        !ds4_gpu_ensure_scratch_buffer(&g_flash_attn_kv_buffer,
                                         &g_flash_attn_kv_bytes,
                                         kv_bytes,
                                         "ds4_flash_attn_kv_f16") ||
        !ds4_gpu_ensure_scratch_buffer(&g_flash_attn_pad_buffer,
                                         &g_flash_attn_pad_bytes,
                                         pad_bytes,
                                         "ds4_flash_attn_pad") ||
        !ds4_gpu_ensure_scratch_buffer(&g_flash_attn_blk_buffer,
                                         &g_flash_attn_blk_bytes,
                                         blk_bytes,
                                         "ds4_flash_attn_blk")) {
        return 0;
    }

    if (!ds4_gpu_encode_copy_raw_ring_to_f16(cb,
                                              rawbuf,
                                              ds4_gpu_tensor_offset(raw_kv),
                                              raw_cap,
                                              raw_start,
                                              n_raw,
                                              head_dim,
                                              g_flash_attn_kv_buffer,
                                              0) ||
        !ds4_gpu_encode_copy_to_f16_1d(cb,
                                       compbuf,
                                       ds4_gpu_tensor_offset(comp_kv),
                                       comp_kv_f16 != 0,
                                       g_flash_attn_kv_buffer,
                                       (NSUInteger)n_raw * row_bytes_f16,
                                       n_comp * head_dim)) {
        return 0;
    }

    ds4_gpu_fill_mixed_decode_batch_mask((uint16_t *)[mask_buffer contents],
                                           n_tokens,
                                           n_raw,
                                           n_comp,
                                           pos0,
                                           window,
                                           ratio);
    if (use_comp_mask) {
        if (!ds4_gpu_encode_cpy_f32_f16_2d(cb,
                                             maskbuf,
                                             ds4_gpu_tensor_offset(comp_mask),
                                             mask_buffer,
                                             (NSUInteger)n_raw * sizeof(uint16_t),
                                             n_comp,
                                             n_tokens,
                                             (uint64_t)n_comp * sizeof(float),
                                             (uint64_t)n_keys * sizeof(uint16_t))) {
            return 0;
        }
    }

    id<MTLComputePipelineState> pad_pipeline = nil;
    if (has_kvpad) {
        pad_pipeline = ds4_gpu_get_flash_attn_pad_pipeline(true, (int32_t)ncpsg);
        if (!pad_pipeline) return 0;
    }
    id<MTLComputePipelineState> blk_pipeline =
        ds4_gpu_get_flash_attn_blk_pipeline((int32_t)nqptg, (int32_t)ncpsg);
    id<MTLComputePipelineState> attn_pipeline =
        ds4_gpu_get_flash_attn_pipeline("kernel_flash_attn_ext_f16_dk512_dv512",
                                          true, true, false, false, has_kvpad, bc_mask,
                                          (int32_t)head_dim,
                                          (int32_t)head_dim,
                                          (int32_t)nsg);
    if (!blk_pipeline || !attn_pipeline) return 0;

    if (has_kvpad) {
        ds4_gpu_flash_attn_pad_args pad_args = {
            .ne11 = (int32_t)n_keys,
            .ne_12_2 = 1,
            .ne_12_3 = 1,
            .nb11 = row_bytes_f16,
            .nb12 = (uint64_t)n_keys * row_bytes_f16,
            .nb13 = (uint64_t)n_keys * row_bytes_f16,
            .nb21 = row_bytes_f16,
            .nb22 = (uint64_t)n_keys * row_bytes_f16,
            .nb23 = (uint64_t)n_keys * row_bytes_f16,
            .ne31 = (int32_t)n_tokens,
            .ne32 = 1,
            .ne33 = 1,
            .nb31 = (uint64_t)n_keys * sizeof(uint16_t),
            .nb32 = mask_bytes,
            .nb33 = mask_bytes,
        };

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pad_pipeline];
        [enc setBytes:&pad_args length:sizeof(pad_args) atIndex:0];
        [enc setBuffer:g_flash_attn_kv_buffer offset:0 atIndex:1];
        [enc setBuffer:g_flash_attn_kv_buffer offset:0 atIndex:2];
        [enc setBuffer:mask_buffer offset:0 atIndex:3];
        [enc setBuffer:g_flash_attn_pad_buffer offset:0 atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(ncpsg, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);
    }

    ds4_gpu_flash_attn_blk_args blk_args = {
        .ne01 = (int32_t)n_tokens,
        .ne30 = (int32_t)n_keys,
        .ne31 = (int32_t)n_tokens,
        .ne32 = 1,
        .ne33 = 1,
        .nb31 = (uint64_t)n_keys * sizeof(uint16_t),
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
        .nb01 = (uint64_t)n_head * row_bytes,
        .nb02 = row_bytes,
        .nb03 = (uint64_t)n_tokens * n_head * row_bytes,
        .ne11 = (int32_t)n_keys,
        .ne_12_2 = 1,
        .ne_12_3 = 1,
        .ns10 = (int32_t)head_dim,
        .nb11 = row_bytes_f16,
        .nb12 = (uint64_t)n_keys * row_bytes_f16,
        .nb13 = (uint64_t)n_keys * row_bytes_f16,
        .ns20 = (int32_t)head_dim,
        .nb21 = row_bytes_f16,
        .nb22 = (uint64_t)n_keys * row_bytes_f16,
        .nb23 = (uint64_t)n_keys * row_bytes_f16,
        .ne31 = (int32_t)n_tokens,
        .ne32 = 1,
        .ne33 = 1,
        .nb31 = (uint64_t)n_keys * sizeof(uint16_t),
        .nb32 = mask_bytes,
        .nb33 = mask_bytes,
        .ne1 = (int32_t)n_head,
        .ne2 = (int32_t)n_tokens,
        .ne3 = 1,
        .scale = 1.0f / sqrtf((float)head_dim),
        .max_bias = 0.0f,
        .m0 = 0.0f,
        .m1 = 0.0f,
        .n_head_log2 = 0,
        .logit_softcap = 0.0f,
    };

    const NSUInteger padded_v = ds4_gpu_align_up_ns(head_dim, 64u);
    const NSUInteger shared_elems = (NSUInteger)nqptg *
        ((NSUInteger)head_dim + 2u * padded_v + 2u * (2u * (NSUInteger)ncpsg));
    const NSUInteger shared_bytes = ds4_gpu_align_up_ns(shared_elems * (sizeof(float) / 2u), 16u);

    enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:attn_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:qbuf offset:ds4_gpu_tensor_offset(q) atIndex:1];
    [enc setBuffer:g_flash_attn_kv_buffer offset:0 atIndex:2];
    [enc setBuffer:g_flash_attn_kv_buffer offset:0 atIndex:3];
    [enc setBuffer:mask_buffer offset:0 atIndex:4];
    [enc setBuffer:sinks_buf offset:sinks_offset atIndex:5];
    [enc setBuffer:g_flash_attn_pad_buffer offset:0 atIndex:6];
    [enc setBuffer:g_flash_attn_blk_buffer offset:0 atIndex:7];
    [enc setBuffer:headsbuf offset:ds4_gpu_tensor_offset(heads) atIndex:8];
    [enc setThreadgroupMemoryLength:shared_bytes atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake(nblk1, n_head, 1)
         threadsPerThreadgroup:MTLSizeMake(32, nsg, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);

    return 1;
}

int ds4_gpu_attention_prefill_raw_heads_range_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        uint32_t                q_row0,
        uint32_t                n_q,
        uint32_t                n_kv,
        uint32_t                window,
        uint32_t                n_head,
        uint32_t                head_dim) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!heads || !q || !raw_kv || !model_map || n_q == 0 || n_kv == 0) return 0;

    @autoreleasepool {
        if (sinks_offset > model_size || (uint64_t)n_head * sizeof(float) > model_size - sinks_offset) {
            fprintf(stderr, "ds4: Metal attention sinks range is outside the mapped model\n");
            return 0;
        }

        uint64_t sinks_inner = 0;
        id<MTLBuffer> sinks_buf = ds4_gpu_wrap_model_range(model_map, model_size,
                                                             sinks_offset,
                                                             (uint64_t)n_head * sizeof(float),
                                                             &sinks_inner);
        if (!sinks_buf) return 0;

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        if (!ds4_gpu_encode_flash_attention_prefill_raw_heads(&cb,
                                                                heads,
                                                                sinks_buf,
                                                                (NSUInteger)sinks_inner,
                                                                q,
                                                                raw_kv,
                                                                q_row0,
                                                                n_q,
                                                                n_kv,
                                                                window,
                                                                n_head,
                                                                head_dim)) {
            return 0;
        }

        if (!ds4_gpu_finish_command_buffer(cb, owned, "graph prefill raw attention heads")) return 0;
    }

    return 1;
}

int ds4_gpu_attention_prefill_raw_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        uint32_t                n_tokens,
        uint32_t                window,
        uint32_t                n_head,
        uint32_t                head_dim) {
    return ds4_gpu_attention_prefill_raw_heads_range_tensor(heads,
                                                              model_map,
                                                              model_size,
                                                              sinks_offset,
                                                              q,
                                                              raw_kv,
                                                              0,
                                                              n_tokens,
                                                              n_tokens,
                                                              window,
                                                              n_head,
                                                              head_dim);
}

int ds4_gpu_attention_decode_raw_batch_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_raw,
        uint32_t                raw_cap,
        uint32_t                raw_start,
        uint32_t                window,
        uint32_t                n_head,
        uint32_t                head_dim) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!heads || !q || !raw_kv || !model_map || n_tokens == 0 ||
        n_raw == 0 || raw_cap < n_raw || raw_start >= raw_cap) {
        return 0;
    }

    @autoreleasepool {
        if (sinks_offset > model_size || (uint64_t)n_head * sizeof(float) > model_size - sinks_offset) {
            fprintf(stderr, "ds4: Metal attention sinks range is outside the mapped model\n");
            return 0;
        }

        uint64_t sinks_inner = 0;
        id<MTLBuffer> sinks_buf = ds4_gpu_wrap_model_range(model_map, model_size,
                                                             sinks_offset,
                                                             (uint64_t)n_head * sizeof(float),
                                                             &sinks_inner);
        if (!sinks_buf) return 0;

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        if (!ds4_gpu_encode_flash_attention_decode_raw_batch_heads(cb,
                                                                     heads,
                                                                     sinks_buf,
                                                                     (NSUInteger)sinks_inner,
                                                                     q,
                                                                     raw_kv,
                                                                     n_tokens,
                                                                     pos0,
                                                                     n_raw,
                                                                     raw_cap,
                                                                     raw_start,
                                                                     window,
                                                                     n_head,
                                                                     head_dim,
                                                                     false)) {
            return 0;
        }

        if (!ds4_gpu_finish_command_buffer(cb, owned, "graph decode raw batch attention heads")) return 0;
    }

    return 1;
}

int ds4_gpu_attention_noncausal_raw_batch_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        uint32_t                n_tokens,
        uint32_t                n_raw,
        uint32_t                raw_cap,
        uint32_t                raw_start,
        uint32_t                n_head,
        uint32_t                head_dim) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!heads || !q || !raw_kv || !model_map ||
        n_tokens == 0 || n_raw == 0 || raw_cap < n_raw ||
        raw_start >= raw_cap || n_head == 0 || head_dim == 0) {
        return 0;
    }

    @autoreleasepool {
        const uint64_t sink_bytes = (uint64_t)n_head * sizeof(float);
        if (sinks_offset > model_size || sink_bytes > model_size - sinks_offset) {
            fprintf(stderr, "ds4: Metal noncausal attention sinks range is outside the mapped model\n");
            return 0;
        }

        uint64_t sinks_inner = 0;
        id<MTLBuffer> sinks_buf = ds4_gpu_wrap_model_range(model_map, model_size,
                                                             sinks_offset,
                                                             sink_bytes,
                                                             &sinks_inner);
        if (!sinks_buf) return 0;

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        if (!ds4_gpu_encode_flash_attention_decode_raw_batch_heads(cb,
                                                                     heads,
                                                                     sinks_buf,
                                                                     (NSUInteger)sinks_inner,
                                                                     q,
                                                                     raw_kv,
                                                                     n_tokens,
                                                                     0,
                                                                     n_raw,
                                                                     raw_cap,
                                                                     raw_start,
                                                                     0,
                                                                     n_head,
                                                                     head_dim,
                                                                     true)) {
            return 0;
        }

        if (!ds4_gpu_finish_command_buffer(cb, owned, "graph noncausal raw batch attention heads")) return 0;
    }

    return 1;
}

int ds4_gpu_attention_decode_mixed_batch_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        const ds4_gpu_tensor *comp_kv,
        uint32_t                comp_kv_f16,
        const ds4_gpu_tensor *comp_mask,
        uint32_t                use_comp_mask,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_raw,
        uint32_t                raw_cap,
        uint32_t                raw_start,
        uint32_t                n_comp,
        uint32_t                window,
        uint32_t                ratio,
        uint32_t                n_head,
        uint32_t                head_dim) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!heads || !q || !raw_kv || !model_map || n_tokens == 0 ||
        n_raw == 0 || raw_cap < n_raw || raw_start >= raw_cap ||
        ratio == 0 || (n_comp != 0 && !comp_kv) ||
        (use_comp_mask != 0 && !comp_mask)) {
        return 0;
    }

    @autoreleasepool {
        if (sinks_offset > model_size || (uint64_t)n_head * sizeof(float) > model_size - sinks_offset) {
            fprintf(stderr, "ds4: Metal attention sinks range is outside the mapped model\n");
            return 0;
        }

        uint64_t sinks_inner = 0;
        id<MTLBuffer> sinks_buf = ds4_gpu_wrap_model_range(model_map, model_size,
                                                             sinks_offset,
                                                             (uint64_t)n_head * sizeof(float),
                                                             &sinks_inner);
        if (!sinks_buf) return 0;

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        if (!ds4_gpu_encode_flash_attention_decode_mixed_batch_heads(cb,
                                                                       heads,
                                                                       sinks_buf,
                                                                       (NSUInteger)sinks_inner,
                                                                       q,
                                                                       raw_kv,
                                                                       comp_kv,
                                                                       comp_kv_f16,
                                                                       comp_mask,
                                                                       use_comp_mask,
                                                                       n_tokens,
                                                                       pos0,
                                                                       n_raw,
                                                                       raw_cap,
                                                                       raw_start,
                                                                       n_comp,
                                                                       window,
                                                                       ratio,
                                                                       n_head,
                                                                       head_dim)) {
            return 0;
        }

        if (!ds4_gpu_finish_command_buffer(cb, owned, "graph decode mixed batch attention heads")) return 0;
    }

    return 1;
}

int ds4_gpu_attention_indexed_mixed_batch_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        const ds4_gpu_tensor *comp_kv,
        uint32_t                comp_kv_f16,
        const ds4_gpu_tensor *topk,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_raw,
        uint32_t                raw_cap,
        uint32_t                raw_start,
        uint32_t                n_comp,
        uint32_t                top_k,
        uint32_t                window,
        uint32_t                ratio,
        uint32_t                n_head,
        uint32_t                head_dim) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!heads || !model_map || !q || !raw_kv || !comp_kv || !topk ||
        n_tokens == 0 || n_raw == 0 || raw_cap < n_raw || raw_start >= raw_cap ||
        n_comp == 0 || top_k == 0 || top_k > n_comp || (top_k & (top_k - 1u)) != 0 ||
        ratio == 0 || n_head == 0 || head_dim != 512) {
        return 0;
    }

    @autoreleasepool {
        if (sinks_offset > model_size || (uint64_t)n_head * sizeof(float) > model_size - sinks_offset) {
            fprintf(stderr, "ds4: Metal indexed attention sinks range is outside the mapped model\n");
            return 0;
        }

        const uint64_t row_bytes = (uint64_t)head_dim * sizeof(float);
        const uint64_t row_bytes_f16 = (uint64_t)head_dim * sizeof(uint16_t);
        const uint64_t q_bytes = (uint64_t)n_tokens * n_head * row_bytes;
        const uint64_t raw_bytes = (uint64_t)raw_cap * row_bytes;
        const uint64_t comp_bytes = (uint64_t)n_comp * (comp_kv_f16 ? row_bytes_f16 : row_bytes);
        const uint64_t topk_bytes = (uint64_t)top_k * n_tokens * sizeof(int32_t);
        id<MTLBuffer> qbuf = ds4_gpu_tensor_buffer(q);
        id<MTLBuffer> rawbuf = ds4_gpu_tensor_buffer(raw_kv);
        id<MTLBuffer> compbuf = ds4_gpu_tensor_buffer(comp_kv);
        id<MTLBuffer> topkbuf = ds4_gpu_tensor_buffer(topk);
        id<MTLBuffer> headsbuf = ds4_gpu_tensor_buffer(heads);
        if (!qbuf || !rawbuf || !compbuf || !topkbuf || !headsbuf ||
            ds4_gpu_tensor_bytes(q) < q_bytes ||
            ds4_gpu_tensor_bytes(raw_kv) < raw_bytes ||
            ds4_gpu_tensor_bytes(comp_kv) < comp_bytes ||
            ds4_gpu_tensor_bytes(topk) < topk_bytes ||
            ds4_gpu_tensor_bytes(heads) < q_bytes) {
            fprintf(stderr, "ds4: Metal indexed mixed attention received undersized buffers\n");
            return 0;
        }

        uint64_t sinks_inner = 0;
        id<MTLBuffer> sinks_buf = ds4_gpu_wrap_model_range(model_map, model_size,
                                                             sinks_offset,
                                                             (uint64_t)n_head * sizeof(float),
                                                             &sinks_inner);
        if (!sinks_buf) return 0;

        id<MTLComputePipelineState> sort_pipeline =
            ds4_gpu_hot_pipeline(g_dsv4_sort_i32_rows_asc_pipeline,
                                    "kernel_dsv4_sort_i32_rows_asc");
        const bool decode_one_token = n_tokens == 1u;
        id<MTLComputePipelineState> attn_pipeline =
            decode_one_token ?
            ds4_gpu_hot_pipeline(g_dsv4_indexed_attention_heads8_rb16_pipeline,
                                   "kernel_dsv4_indexed_mixed_attention_heads8_rb16") :
            ds4_gpu_hot_pipeline(g_dsv4_indexed_attention_heads8_pipeline,
                                   "kernel_dsv4_indexed_mixed_attention_heads8");
        if (!sort_pipeline || !attn_pipeline) return 0;
        if ((NSUInteger)top_k > sort_pipeline.maxTotalThreadsPerThreadgroup) {
            fprintf(stderr, "ds4: Metal indexed attention top-k exceeds sort threadgroup limit\n");
            return 0;
        }
        /*
         * Fast decode attends to the same full top-k compressed rows but keeps
         * them in score order, avoiding a chronological sort dispatch.
         * --quality restores the sorted order for stricter reproducibility.
         */
        const bool skip_decode_sort = !g_quality_mode && decode_one_token;
        if (!skip_decode_sort &&
            !ds4_gpu_ensure_scratch_buffer(&g_indexed_topk_buffer,
                                             &g_indexed_topk_bytes,
                                             (NSUInteger)topk_bytes,
                                             "ds4_indexed_topk_sorted")) {
            return 0;
        }

        ds4_gpu_dsv4_topk_mask_args sort_args = {
            .ne00 = (int64_t)top_k,
            .ne01 = (int64_t)n_tokens,
            .nb00 = sizeof(int32_t),
            .nb01 = (uint64_t)top_k * sizeof(int32_t),
            .ne0 = (int64_t)top_k,
            .ne1 = (int64_t)n_tokens,
            .nb0 = sizeof(int32_t),
            .nb1 = (uint64_t)top_k * sizeof(int32_t),
        };
        ds4_gpu_dsv4_indexed_attention_args attn_args = {
            .n_tokens = n_tokens,
            .n_head = n_head,
            .n_raw = n_raw,
            .raw_cap = raw_cap,
            .raw_start = raw_start,
            .n_comp = n_comp,
            .top_k = top_k,
            .pos0 = pos0,
            .window = window,
            .ratio = ratio,
            .comp_kv_f16 = comp_kv_f16 ? 1u : 0u,
            .pad0 = 0,
            .q_token_stride = (uint64_t)n_head * row_bytes,
            .q_head_stride = row_bytes,
            .raw_row_stride = row_bytes,
            .comp_row_stride = comp_kv_f16 ? row_bytes_f16 : row_bytes,
            .topk_token_stride = (uint64_t)top_k * sizeof(int32_t),
            .dst_token_stride = (uint64_t)n_head * row_bytes,
            .dst_head_stride = row_bytes,
            .scale = 1.0f / sqrtf((float)head_dim),
        };

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        id<MTLComputeCommandEncoder> enc = nil;
        if (!skip_decode_sort) {
            enc = ds4_gpu_compute_encoder(cb);
            [enc setComputePipelineState:sort_pipeline];
            [enc setBytes:&sort_args length:sizeof(sort_args) atIndex:0];
            [enc setBuffer:topkbuf offset:ds4_gpu_tensor_offset(topk) atIndex:1];
            [enc setBuffer:g_indexed_topk_buffer offset:0 atIndex:2];
            [enc setThreadgroupMemoryLength:(NSUInteger)top_k * sizeof(int32_t) atIndex:0];
            [enc dispatchThreadgroups:MTLSizeMake(n_tokens, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(top_k, 1, 1)];
            ds4_gpu_end_compute_encoder(cb, enc);
        }

        enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:attn_pipeline];
        [enc setBytes:&attn_args length:sizeof(attn_args) atIndex:0];
        [enc setBuffer:qbuf offset:ds4_gpu_tensor_offset(q) atIndex:1];
        [enc setBuffer:rawbuf offset:ds4_gpu_tensor_offset(raw_kv) atIndex:2];
        [enc setBuffer:compbuf offset:ds4_gpu_tensor_offset(comp_kv) atIndex:3];
        [enc setBuffer:skip_decode_sort ? topkbuf : g_indexed_topk_buffer
              offset:skip_decode_sort ? ds4_gpu_tensor_offset(topk) : 0
             atIndex:4];
        [enc setBuffer:sinks_buf offset:(NSUInteger)sinks_inner atIndex:5];
        [enc setBuffer:headsbuf offset:ds4_gpu_tensor_offset(heads) atIndex:6];
        [enc setThreadgroupMemoryLength:(decode_one_token ? 16u : 1u) *
                                        128u * 4u * sizeof(uint16_t)
                                atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)n_tokens, ((NSUInteger)n_head + 7u) / 8u, 1)
             threadsPerThreadgroup:MTLSizeMake(32, 8, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "graph indexed mixed attention heads")) return 0;
    }

    return 1;
}

int ds4_gpu_attention_prefill_static_mixed_heads_range_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        const ds4_gpu_tensor *comp_kv,
        uint32_t                comp_kv_f16,
        uint32_t                q_row0,
        uint32_t                n_q,
        uint32_t                n_tokens,
        uint32_t                n_comp,
        uint32_t                window,
        uint32_t                ratio,
        uint32_t                n_head,
        uint32_t                head_dim) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!heads || !q || !raw_kv || !model_map || n_q == 0 || n_tokens == 0 ||
        ratio == 0 || (n_comp != 0 && !comp_kv)) {
        return 0;
    }

    @autoreleasepool {
        if (sinks_offset > model_size || (uint64_t)n_head * sizeof(float) > model_size - sinks_offset) {
            fprintf(stderr, "ds4: Metal attention sinks range is outside the mapped model\n");
            return 0;
        }

        uint64_t sinks_inner = 0;
        id<MTLBuffer> sinks_buf = ds4_gpu_wrap_model_range(model_map, model_size,
                                                             sinks_offset,
                                                             (uint64_t)n_head * sizeof(float),
                                                             &sinks_inner);
        if (!sinks_buf) return 0;

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        if (!ds4_gpu_encode_flash_attention_prefill_static_mixed_heads_nonvec(&cb,
                                                                                heads,
                                                                                sinks_buf,
                                                                                (NSUInteger)sinks_inner,
                                                                                q,
                                                                                raw_kv,
                                                                                comp_kv,
                                                                                comp_kv_f16,
                                                                                NULL,
                                                                                0,
                                                                                q_row0,
                                                                                n_q,
                                                                                n_tokens,
                                                                                n_comp,
                                                                                window,
                                                                                ratio,
                                                                                n_head,
                                                                                head_dim)) {
            return 0;
        }

        if (!ds4_gpu_finish_command_buffer(cb, owned, "graph prefill static mixed attention heads")) return 0;
    }

    return 1;
}

int ds4_gpu_attention_prefill_static_mixed_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        const ds4_gpu_tensor *comp_kv,
        uint32_t                comp_kv_f16,
        uint32_t                n_tokens,
        uint32_t                n_comp,
        uint32_t                window,
        uint32_t                ratio,
        uint32_t                n_head,
        uint32_t                head_dim) {
    return ds4_gpu_attention_prefill_static_mixed_heads_range_tensor(heads,
                                                                       model_map,
                                                                       model_size,
                                                                       sinks_offset,
                                                                       q,
                                                                       raw_kv,
                                                                       comp_kv,
                                                                       comp_kv_f16,
                                                                       0,
                                                                       n_tokens,
                                                                       n_tokens,
                                                                       n_comp,
                                                                       window,
                                                                       ratio,
                                                                       n_head,
                                                                       head_dim);
}

int ds4_gpu_attention_prefill_masked_mixed_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        const ds4_gpu_tensor *comp_kv,
        uint32_t                comp_kv_f16,
        const ds4_gpu_tensor *comp_mask,
        uint32_t                n_tokens,
        uint32_t                n_comp,
        uint32_t                window,
        uint32_t                ratio,
        uint32_t                n_head,
        uint32_t                head_dim) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!heads || !q || !raw_kv || !comp_kv || !comp_mask || !model_map ||
        n_tokens == 0 || n_comp == 0 || ratio == 0) {
        return 0;
    }

    @autoreleasepool {
        if (sinks_offset > model_size || (uint64_t)n_head * sizeof(float) > model_size - sinks_offset) {
            fprintf(stderr, "ds4: Metal attention sinks range is outside the mapped model\n");
            return 0;
        }

        uint64_t sinks_inner = 0;
        id<MTLBuffer> sinks_buf = ds4_gpu_wrap_model_range(model_map, model_size,
                                                             sinks_offset,
                                                             (uint64_t)n_head * sizeof(float),
                                                             &sinks_inner);
        if (!sinks_buf) return 0;

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        if (!ds4_gpu_encode_flash_attention_prefill_static_mixed_heads_nonvec(&cb,
                                                                                heads,
                                                                                sinks_buf,
                                                                                (NSUInteger)sinks_inner,
                                                                                q,
                                                                                raw_kv,
                                                                                comp_kv,
                                                                                comp_kv_f16,
                                                                                comp_mask,
                                                                                1,
                                                                                0,
                                                                                n_tokens,
                                                                                n_tokens,
                                                                                n_comp,
                                                                                window,
                                                                                ratio,
                                                                                n_head,
                                                                                head_dim)) {
            return 0;
        }

        if (!ds4_gpu_finish_command_buffer(cb, owned, "graph prefill masked mixed attention heads")) return 0;
    }

    return 1;
}

int ds4_gpu_attention_decode_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        uint32_t                n_raw,
        uint32_t                raw_cap,
        uint32_t                raw_start,
        const ds4_gpu_tensor *comp_kv,
        uint32_t                comp_kv_f16,
        uint32_t                n_comp,
        const ds4_gpu_tensor *comp_mask,
        uint32_t                use_mask,
        uint32_t                n_head,
        uint32_t                head_dim) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!heads || !model_map || !q || !raw_kv ||
        n_raw == 0 || n_head == 0 || head_dim == 0 ||
        raw_cap < n_raw || raw_start >= raw_cap ||
        n_raw > UINT32_MAX - n_comp || n_raw + n_comp > 8192u ||
        (n_comp != 0 && !comp_kv) ||
        (use_mask != 0 && !comp_mask)) {
        return 0;
    }

    @autoreleasepool {
        const uint64_t q_bytes = (uint64_t)n_head * head_dim * sizeof(float);
        const uint64_t raw_bytes = (uint64_t)raw_cap * head_dim * sizeof(float);
        const uint64_t comp_bytes = (uint64_t)n_comp * head_dim *
                                    (comp_kv_f16 ? sizeof(uint16_t) : sizeof(float));
        const uint64_t sink_bytes = (uint64_t)n_head * sizeof(float);
        if (sinks_offset > model_size || sink_bytes > model_size - sinks_offset) {
            fprintf(stderr, "ds4: Metal graph attention heads sink range is outside the mapped model\n");
            return 0;
        }

        id<MTLBuffer> qbuf = ds4_gpu_tensor_buffer(q);
        id<MTLBuffer> rawbuf = ds4_gpu_tensor_buffer(raw_kv);
        id<MTLBuffer> compbuf = n_comp ? ds4_gpu_tensor_buffer(comp_kv) : rawbuf;
        id<MTLBuffer> maskbuf = use_mask ? ds4_gpu_tensor_buffer(comp_mask) : rawbuf;
        id<MTLBuffer> headsbuf = ds4_gpu_tensor_buffer(heads);
        const uint64_t comp_mask_bytes = use_mask ? (uint64_t)n_comp * sizeof(float) : 0u;
        if (!qbuf || !rawbuf || !compbuf || !maskbuf || !headsbuf ||
            ds4_gpu_tensor_bytes(q) < q_bytes ||
            ds4_gpu_tensor_bytes(raw_kv) < raw_bytes ||
            (n_comp && ds4_gpu_tensor_bytes(comp_kv) < comp_bytes) ||
            (use_mask && ds4_gpu_tensor_bytes(comp_mask) < comp_mask_bytes) ||
            ds4_gpu_tensor_bytes(heads) < q_bytes) {
            fprintf(stderr, "ds4: Metal graph attention heads received undersized buffers\n");
            return 0;
        }

        uint64_t sinks_inner = 0;
        id<MTLBuffer> sinks_buf = ds4_gpu_wrap_model_range(model_map, model_size, sinks_offset, sink_bytes, &sinks_inner);
        if (!sinks_buf) return 0;

        if (n_comp == 0) {
            int owned = 0;
            id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
            if (!cb) return 0;

            if (!ds4_gpu_encode_flash_attention_raw_heads(cb,
                                                            heads,
                                                            sinks_buf,
                                                            (NSUInteger)sinks_inner,
                                                            q,
                                                            raw_kv,
                                                            n_raw,
                                                            raw_cap,
                                                            raw_start,
                                                            n_head,
                                                            head_dim)) {
                return 0;
            }

            if (!ds4_gpu_finish_command_buffer(cb, owned, "graph raw attention heads")) return 0;
            return 1;
        }

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        if (!ds4_gpu_encode_flash_attention_gathered_heads(cb,
                                                             heads,
                                                             sinks_buf,
                                                             (NSUInteger)sinks_inner,
                                                             q,
                                                             raw_kv,
                                                             n_raw,
                                                             raw_cap,
                                                             raw_start,
                                                             comp_kv,
                                                             comp_kv_f16,
                                                             n_comp,
                                                             comp_mask,
                                                             use_mask,
                                                             n_head,
                                                             head_dim)) {
            return 0;
        }

        if (!ds4_gpu_finish_command_buffer(cb, owned, "graph attention heads")) return 0;
    }

    return 1;
}
