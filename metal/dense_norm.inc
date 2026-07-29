int ds4_gpu_matmul_q8_0_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if ((in_dim & 31u) != 0 ||
        in_dim > UINT32_MAX || out_dim > UINT32_MAX || n_tok > UINT32_MAX) {
        return 0;
    }

    const int profile_requested =
        n_tok > 8u && ds4_gpu_env_bool("DS4_METAL_Q8_PREFILL_PROFILE") > 0;
    int profile_prefill = 0;
    int split_batch_for_profile = 0;
    const char *profile_label = NULL;
    char profile_label_buf[128];
    char profile_fallback[128];
    if (profile_requested) {
        snprintf(profile_fallback, sizeof(profile_fallback),
                 "q8 weight_off=%llu in=%llu out=%llu tok=%llu",
                 (unsigned long long)weight_offset,
                 (unsigned long long)in_dim,
                 (unsigned long long)out_dim,
                 (unsigned long long)n_tok);
        snprintf(profile_label_buf, sizeof(profile_label_buf), "%s", profile_fallback);
        profile_label = profile_label_buf;
        const char *profile_filter = getenv("DS4_METAL_Q8_PREFILL_PROFILE_FILTER");
        profile_prefill =
            profile_requested &&
            (!profile_filter || !profile_filter[0] ||
             strstr(profile_label, profile_filter) != NULL);
    }
    if (profile_prefill) {
        if (g_batch_cb) {
            if (ds4_gpu_end_commands() == 0 || ds4_gpu_begin_commands() == 0) {
                return 0;
            }
            split_batch_for_profile = 1;
        }
    }

    const double profile_t0 = profile_prefill ? ds4_gpu_now_ms() : 0.0;
    int ok = ds4_gpu_matmul_q8_0_legacy_tensor(out, model_map, model_size,
                                               weight_offset, in_dim, out_dim,
                                               x, n_tok, false, false);
    if (profile_prefill) {
        if (split_batch_for_profile && ds4_gpu_end_commands() == 0) {
            ok = 0;
        }
        const double elapsed_ms = ds4_gpu_now_ms() - profile_t0;
        fprintf(stderr,
                "ds4: Metal Q8_0 prefill profile %s in=%llu out=%llu tok=%llu %.3f ms\n",
                profile_label ? profile_label : profile_fallback,
                (unsigned long long)in_dim,
                (unsigned long long)out_dim,
                (unsigned long long)n_tok,
                elapsed_ms);
        if (split_batch_for_profile && ds4_gpu_begin_commands() == 0) {
            ok = 0;
        }
    }
    return ok;
}

int ds4_gpu_matmul_q8_0_decode_rows_exact_tensor(
        ds4_gpu_tensor       *out,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint64_t              in_dim,
        uint64_t              out_dim,
        const ds4_gpu_tensor *x,
        uint32_t              n_rows) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!out || !x || !model_map || n_rows == 0 ||
        n_rows > INT32_MAX || in_dim == 0 || out_dim == 0 ||
        (in_dim & 31u) != 0 || in_dim > UINT32_MAX ||
        out_dim > UINT32_MAX ||
        in_dim > UINT64_MAX / n_rows / sizeof(float) ||
        out_dim > UINT64_MAX / n_rows / sizeof(float) ||
        ds4_gpu_tensor_bytes(x) <
            (uint64_t)n_rows * in_dim * sizeof(float) ||
        ds4_gpu_tensor_bytes(out) <
            (uint64_t)n_rows * out_dim * sizeof(float)) {
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> xbuf = ds4_gpu_tensor_buffer(x);
        id<MTLBuffer> outbuf = ds4_gpu_tensor_buffer(out);
        const uint64_t blocks = in_dim / 32u;
        const uint64_t row_bytes = blocks * 34u;
        if (!xbuf || !outbuf ||
            out_dim > UINT64_MAX / row_bytes) {
            return 0;
        }
        const uint64_t weight_bytes = out_dim * row_bytes;
        if (weight_offset > model_size ||
            weight_bytes > model_size - weight_offset) {
            return 0;
        }

        uint64_t inner_offset = 0;
        id<MTLBuffer> wbuf = ds4_gpu_wrap_q8_decode_model_range(
                model_map, model_size, weight_offset, weight_bytes, 1u,
                &inner_offset);
        if (!wbuf) return 0;

        ds4_gpu_mv_dispatch dispatch = ds4_gpu_make_q8_0_mv_dispatch();
        if (out_dim > 65536u) dispatch.nsg = 8;
        ds4_gpu_q8_0_matvec_args args =
            ds4_gpu_make_q8_0_mv_args(in_dim, out_dim);
        args.ne11 = (int32_t)n_rows;
        args.nb12 = (uint64_t)n_rows * in_dim * sizeof(float);
        args.nb13 = args.nb12;
        args.ne1 = (int32_t)n_rows;
        args.nr0 = dispatch.nr0;

        id<MTLComputePipelineState> pipeline =
            ds4_gpu_get_mul_mv_pipeline(dispatch.function_name, dispatch.nsg);
        if (!pipeline) return 0;

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;
        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:wbuf offset:(NSUInteger)inner_offset atIndex:1];
        [enc setBuffer:xbuf offset:ds4_gpu_tensor_offset(x) atIndex:2];
        [enc setBuffer:outbuf offset:ds4_gpu_tensor_offset(out) atIndex:3];
        [enc setThreadgroupMemoryLength:dispatch.smem atIndex:0];
        [enc dispatchThreadgroups:
                MTLSizeMake(((NSUInteger)out_dim +
                             (NSUInteger)dispatch.nr0 - 1u) /
                                (NSUInteger)dispatch.nr0,
                            (NSUInteger)n_rows,
                            1)
             threadsPerThreadgroup:
                MTLSizeMake(32, (NSUInteger)dispatch.nsg, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        return ds4_gpu_finish_command_buffer(
                cb, owned, "Q8_0 exact decode-row matvec");
    }
}

int ds4_gpu_matmul_q8_0_decode_mpp_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok) {
    return ds4_gpu_matmul_q8_0_legacy_tensor(out, model_map, model_size,
                                             weight_offset, in_dim, out_dim,
                                             x, n_tok, true, false);
}

int ds4_gpu_matmul_q8_0_decode_mpp_model_view_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok) {
    return ds4_gpu_matmul_q8_0_legacy_tensor(out, model_map, model_size,
                                             weight_offset, in_dim, out_dim,
                                             x, n_tok, true, true);
}

static const char *ds4_gpu_q4_mv_ext_name(uint32_t weight_type, int16_t r1ptg) {
    const char *prefix = NULL;
    if (weight_type == DS4_METAL_TENSOR_Q4_K) {
        prefix = "kernel_mul_mv_ext_q4_K_f32_r1_";
    } else if (weight_type == DS4_METAL_TENSOR_Q4_0) {
        prefix = "kernel_mul_mv_ext_q4_0_f32_r1_";
    } else {
        return NULL;
    }
    switch (r1ptg) {
    case 1: return weight_type == DS4_METAL_TENSOR_Q4_K ?
        "kernel_mul_mv_ext_q4_K_f32_r1_1" : "kernel_mul_mv_ext_q4_0_f32_r1_1";
    case 2: return weight_type == DS4_METAL_TENSOR_Q4_K ?
        "kernel_mul_mv_ext_q4_K_f32_r1_2" : "kernel_mul_mv_ext_q4_0_f32_r1_2";
    case 3: return weight_type == DS4_METAL_TENSOR_Q4_K ?
        "kernel_mul_mv_ext_q4_K_f32_r1_3" : "kernel_mul_mv_ext_q4_0_f32_r1_3";
    case 4: return weight_type == DS4_METAL_TENSOR_Q4_K ?
        "kernel_mul_mv_ext_q4_K_f32_r1_4" : "kernel_mul_mv_ext_q4_0_f32_r1_4";
    case 5: return weight_type == DS4_METAL_TENSOR_Q4_K ?
        "kernel_mul_mv_ext_q4_K_f32_r1_5" : "kernel_mul_mv_ext_q4_0_f32_r1_5";
    default:
        (void)prefix;
        return NULL;
    }
}

static const char *ds4_gpu_q4_mm_name(uint32_t weight_type) {
    switch (weight_type) {
    case DS4_METAL_TENSOR_Q4_0: return "kernel_mul_mm_q4_0_f32";
    case DS4_METAL_TENSOR_Q4_K: return "kernel_mul_mm_q4_K_f32";
    default: return NULL;
    }
}

static const char *ds4_gpu_q4_nax_name(uint32_t weight_type, uint64_t tile_n) {
    if (weight_type == DS4_METAL_TENSOR_Q4_0) {
        return tile_n == 128u ? "kernel_mul_mm_q4_0_f32_nax_direct_rhs_n128" :
               tile_n == 64u  ? "kernel_mul_mm_q4_0_f32_nax_direct_rhs_n64" :
                                 "kernel_mul_mm_q4_0_f32_nax_direct_rhs";
    }
    if (weight_type == DS4_METAL_TENSOR_Q4_K) {
        return tile_n == 128u ? "kernel_mul_mm_q4_K_f32_nax_direct_rhs_n128" :
               tile_n == 64u  ? "kernel_mul_mm_q4_K_f32_nax_direct_rhs_n64" :
                                 "kernel_mul_mm_q4_K_f32_nax_direct_rhs";
    }
    return NULL;
}

static int ds4_gpu_matmul_quant_impl_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                weight_type,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok,
        bool                    prefer_decode_mpp,
        bool                    force_model_view) {
    if (weight_type == DS4_METAL_TENSOR_Q8_0) {
        return ds4_gpu_matmul_q8_0_legacy_tensor(out,
                                                 model_map,
                                                 model_size,
                                                 weight_offset,
                                                 in_dim,
                                                 out_dim,
                                                 x,
                                                 n_tok,
                                                 prefer_decode_mpp,
                                                 force_model_view);
    }
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!out || !x || !model_map ||
        in_dim == 0 || out_dim == 0 || n_tok == 0 ||
        in_dim > UINT32_MAX || out_dim > UINT32_MAX || n_tok > UINT32_MAX) {
        return 0;
    }

    uint64_t row_bytes = 0;
    if (!ds4_gpu_quant_row_bytes(weight_type, (uint32_t)in_dim, &row_bytes)) {
        fprintf(stderr, "ds4: Metal quant matmul received unsupported type/dim (%u, in=%llu)\n",
                weight_type,
                (unsigned long long)in_dim);
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
            fprintf(stderr, "ds4: Metal quant tensor matmul received undersized activation buffers\n");
            return 0;
        }

        if (out_dim > UINT64_MAX / row_bytes) return 0;
        const uint64_t weight_bytes = out_dim * row_bytes;
        if (weight_offset > model_size || weight_bytes > model_size - weight_offset) {
            fprintf(stderr, "ds4: Metal quant tensor matmul range is outside the mapped model\n");
            return 0;
        }

        uint64_t inner_offset = 0;
        id<MTLBuffer> wbuf = force_model_view ?
            ds4_gpu_wrap_model_range(model_map,
                                     model_size,
                                     weight_offset,
                                     weight_bytes,
                                     &inner_offset) :
            ds4_gpu_wrap_model_range(model_map,
                                     model_size,
                                     weight_offset,
                                     weight_bytes,
                                     &inner_offset);
        if (!wbuf) return 0;

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        /*
         * Small-batch Q4_K goes to the classic (llama.cpp-style) matvec:
         * the mul_mv_ext family tops out around 220 GB/s on M5 for the GLM
         * DenseQ4 decode shapes while this impl streams 530-650 GB/s
         * (misc/q4mv_bench.m). Falls through to ext when unavailable.
         */
        if (weight_type == DS4_METAL_TENSOR_Q4_K &&
            n_tok <= 8u &&
            (in_dim % 256u) == 0 &&
            getenv("DS4_METAL_DISABLE_Q4_MV_CLASSIC") == NULL) {
            const int16_t nsg = 2;
            id<MTLComputePipelineState> pipeline =
                ds4_gpu_get_mul_mv_ext_pipeline("kernel_mul_mv_q4_K_dense_f32", nsg, 8);
            if (pipeline) {
                ds4_gpu_q8_0_matvec_args args = {
                    .ne00 = (int32_t)in_dim,
                    .ne01 = (int32_t)out_dim,
                    .ne02 = 1,
                    .nb00 = 1,
                    .nb01 = row_bytes,
                    .nb02 = row_bytes * out_dim,
                    .nb03 = row_bytes * out_dim,
                    .ne10 = (int32_t)in_dim,
                    .ne11 = (int32_t)n_tok,
                    .ne12 = 1,
                    .nb10 = sizeof(float),
                    .nb11 = in_dim * sizeof(float),
                    .nb12 = in_dim * n_tok * sizeof(float),
                    .nb13 = in_dim * n_tok * sizeof(float),
                    .ne0 = (int32_t)out_dim,
                    .ne1 = (int32_t)n_tok,
                    .nr0 = 2,
                    .r2 = 1,
                    .r3 = 1,
                };
                const uint64_t rows_ptg = (uint64_t)nsg * 2u;

                id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
                [enc setComputePipelineState:pipeline];
                [enc setBytes:&args length:sizeof(args) atIndex:0];
                [enc setBuffer:wbuf offset:(NSUInteger)inner_offset atIndex:1];
                [enc setBuffer:xbuf offset:ds4_gpu_tensor_offset(x) atIndex:2];
                [enc setBuffer:outbuf offset:ds4_gpu_tensor_offset(out) atIndex:3];
                [enc setThreadgroupMemoryLength:32 atIndex:0];
                [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)out_dim + rows_ptg - 1u) / rows_ptg,
                                                      (NSUInteger)n_tok,
                                                      1)
                     threadsPerThreadgroup:MTLSizeMake(32, (NSUInteger)nsg, 1)];
                ds4_gpu_end_compute_encoder(cb, enc);

                if (!ds4_gpu_finish_command_buffer(cb, owned, "Q4_K classic mul_mv")) return 0;
                return 1;
            }
        }

        if (n_tok <= 8u && (in_dim % 128u) == 0) {
            const int16_t nsg = 2;
            const int16_t nxpsg = ds4_gpu_mv_ext_nxpsg(in_dim, n_tok);
            const int16_t r1ptg = (n_tok == 1u) ? 1 : ds4_gpu_mv_ext_r1ptg(n_tok);
            const char *fn_name = ds4_gpu_q4_mv_ext_name(weight_type, r1ptg);
            id<MTLComputePipelineState> pipeline =
                fn_name ? ds4_gpu_get_mul_mv_ext_pipeline(fn_name, nsg, nxpsg) : nil;
            if (pipeline) {
                const int16_t nypsg = 32 / nxpsg;
                const uint64_t r0ptg = (uint64_t)nypsg * (uint64_t)nsg;
                ds4_gpu_mul_mv_ext_args args =
                    ds4_gpu_make_mv_ext_args(in_dim, out_dim, n_tok, row_bytes, row_bytes);

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

                if (!ds4_gpu_finish_command_buffer(cb, owned, "Q4 tensor mul_mv_ext")) return 0;
                return 1;
            }
        }

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
            const char *nax_fn = ds4_gpu_q4_nax_name(weight_type, nax_tile_n);
            id<MTLComputePipelineState> pipeline =
                nax_fn ? ds4_gpu_get_mul_mm_pipeline(nax_fn, false, false) : nil;
            if (pipeline) {
                ds4_gpu_mul_mm_args args = ds4_gpu_make_mm_args(in_dim, out_dim, n_tok, row_bytes);

                id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
                [enc setComputePipelineState:pipeline];
                [enc setBytes:&args length:sizeof(args) atIndex:0];
                [enc setBuffer:wbuf offset:(NSUInteger)inner_offset atIndex:1];
                [enc setBuffer:xbuf offset:ds4_gpu_tensor_offset(x) atIndex:2];
                [enc setBuffer:outbuf offset:ds4_gpu_tensor_offset(out) atIndex:3];
                [enc setThreadgroupMemoryLength:64u * 32u * sizeof(uint16_t) atIndex:0];
                [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)(n_tok / nax_tile_n),
                                                      (NSUInteger)out_dim / 64u,
                                                      1)
                     threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                ds4_gpu_end_compute_encoder(cb, enc);

                if (!ds4_gpu_finish_command_buffer(cb, owned, "Q4 NAX tensor matmul")) return 0;
                return 1;
            }
            ds4_gpu_warn_mpp_fallback();
        }

        const char *mm_fn = ds4_gpu_q4_mm_name(weight_type);
        const bool bc_inp = (in_dim % 32u) != 0;
        const bool bc_out = (out_dim % 64u) != 0 || (n_tok % 32u) != 0;
        id<MTLComputePipelineState> pipeline =
            mm_fn ? ds4_gpu_get_mul_mm_pipeline(mm_fn, bc_inp, bc_out) : nil;
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

        if (!ds4_gpu_finish_command_buffer(cb, owned, "Q4 tensor matmul")) return 0;
    }

    return 1;
}

int ds4_gpu_matmul_quant_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                weight_type,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok) {
    return ds4_gpu_matmul_quant_impl_tensor(out,
                                            model_map,
                                            model_size,
                                            weight_offset,
                                            weight_type,
                                            in_dim,
                                            out_dim,
                                            x,
                                            n_tok,
                                            false,
                                            false);
}

int ds4_gpu_matmul_quant_decode_mpp_model_view_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                weight_type,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok) {
    /* GLM decode dense matvecs: the classic Q8_0 kernels stream 570-613
     * GB/s on the GLM shapes while the MPP/nax matrix path measures ~150
     * GB/s at n_tok=1 (TP decode 8.75 -> 16.64 t/s on the IQ2+Q8 gguf).
     * MPP stays available as an opt-in via DS4_METAL_Q8_DECODE_MPP. */
    const bool prefer_mpp = getenv("DS4_METAL_Q8_DECODE_MPP") != NULL;
    return ds4_gpu_matmul_quant_impl_tensor(out,
                                            model_map,
                                            model_size,
                                            weight_offset,
                                            weight_type,
                                            in_dim,
                                            out_dim,
                                            x,
                                            n_tok,
                                            prefer_mpp,
                                            true);
}

int ds4_gpu_matmul_quant_rows_scalar_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                weight_type,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok) {
    return ds4_gpu_matmul_quant_tensor(out,
                                       model_map,
                                       model_size,
                                       weight_offset,
                                       weight_type,
                                       in_dim,
                                       out_dim,
                                       x,
                                       n_tok);
}

int ds4_gpu_matmul_q8_0_rows_scalar_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok) {
    if (n_tok == 1) {
        return ds4_gpu_matmul_q8_0_tensor(out,
                                          model_map,
                                          model_size,
                                          weight_offset,
                                          in_dim,
                                          out_dim,
                                          x,
                                          1);
    }
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!out || !x || !model_map ||
        n_tok == 0 || n_tok > INT32_MAX ||
        (in_dim & 31u) != 0 ||
        in_dim > UINT32_MAX || out_dim > UINT32_MAX ||
        in_dim > UINT64_MAX / sizeof(float) ||
        out_dim > UINT64_MAX / sizeof(float)) {
        return 0;
    }

    const uint64_t x_row_bytes = in_dim * sizeof(float);
    const uint64_t out_row_bytes = out_dim * sizeof(float);
    if ((x_row_bytes != 0 && n_tok > UINT64_MAX / x_row_bytes) ||
        (out_row_bytes != 0 && n_tok > UINT64_MAX / out_row_bytes)) {
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> xbuf = ds4_gpu_tensor_buffer(x);
        id<MTLBuffer> outbuf = ds4_gpu_tensor_buffer(out);
        const uint64_t x_bytes = n_tok * x_row_bytes;
        const uint64_t out_bytes = n_tok * out_row_bytes;
        if (!xbuf || !outbuf ||
            ds4_gpu_tensor_bytes(x) < x_bytes ||
            ds4_gpu_tensor_bytes(out) < out_bytes) {
            fprintf(stderr, "ds4: Metal Q8_0 scalar-row matmul received undersized activation buffers\n");
            return 0;
        }

        const uint64_t blocks = in_dim / 32;
        if (blocks > UINT64_MAX / 34u) return 0;
        const uint64_t row_bytes = blocks * 34u;
        if (out_dim > UINT64_MAX / row_bytes) return 0;
        const uint64_t weight_bytes = out_dim * row_bytes;
        if (weight_offset > model_size || weight_bytes > model_size - weight_offset) {
            fprintf(stderr, "ds4: Metal Q8_0 scalar-row matmul range is outside the mapped model\n");
            return 0;
        }

        uint64_t inner_offset = 0;
        id<MTLBuffer> wbuf =
            ds4_gpu_wrap_model_range(model_map, model_size, weight_offset, weight_bytes, &inner_offset);
        if (!wbuf) return 0;

        ds4_gpu_q8_0_matvec_args mv_args = ds4_gpu_make_q8_0_mv_args(in_dim, out_dim);
        ds4_gpu_mv_dispatch mv_dispatch = ds4_gpu_make_q8_0_mv_dispatch();
        if (out_dim > 65536u) mv_dispatch.nsg = 8;
        mv_args.nr0 = mv_dispatch.nr0;
        id<MTLComputePipelineState> pipeline =
            ds4_gpu_get_mul_mv_pipeline(mv_dispatch.function_name, mv_dispatch.nsg);
        if (!pipeline) return 0;

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&mv_args length:sizeof(mv_args) atIndex:0];
        [enc setBuffer:wbuf offset:(NSUInteger)inner_offset atIndex:1];
        [enc setThreadgroupMemoryLength:mv_dispatch.smem atIndex:0];

        const NSUInteger x_base = ds4_gpu_tensor_offset(x);
        const NSUInteger out_base = ds4_gpu_tensor_offset(out);
        for (uint64_t t = 0; t < n_tok; t++) {
            [enc setBuffer:xbuf offset:x_base + (NSUInteger)(t * x_row_bytes) atIndex:2];
            [enc setBuffer:outbuf offset:out_base + (NSUInteger)(t * out_row_bytes) atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)out_dim + (NSUInteger)mv_dispatch.nr0 - 1u) /
                                                      (NSUInteger)mv_dispatch.nr0,
                                                  1,
                                                  1)
                 threadsPerThreadgroup:MTLSizeMake(32, (NSUInteger)mv_dispatch.nsg, 1)];
        }
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "Q8_0 scalar-row matvecs")) {
            return 0;
        }
    }

    return 1;
}

int ds4_gpu_matmul_q8_0_pair_tensor(
        ds4_gpu_tensor       *out0,
        ds4_gpu_tensor       *out1,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight0_offset,
        uint64_t                weight1_offset,
        uint64_t                in_dim,
        uint64_t                out0_dim,
        uint64_t                out1_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!out0 || !out1 || !model_map || !x || n_tok != 1 ||
        out0_dim == 0 || out1_dim == 0 || (in_dim & 31u) != 0 ||
        in_dim > UINT32_MAX || out0_dim > UINT32_MAX || out1_dim > UINT32_MAX) {
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> xbuf = ds4_gpu_tensor_buffer(x);
        id<MTLBuffer> out0buf = ds4_gpu_tensor_buffer(out0);
        id<MTLBuffer> out1buf = ds4_gpu_tensor_buffer(out1);
        const uint64_t x_bytes = in_dim * sizeof(float);
        const uint64_t out0_bytes = out0_dim * sizeof(float);
        const uint64_t out1_bytes = out1_dim * sizeof(float);
        if (!xbuf || !out0buf || !out1buf ||
            ds4_gpu_tensor_bytes(x) < x_bytes ||
            ds4_gpu_tensor_bytes(out0) < out0_bytes ||
            ds4_gpu_tensor_bytes(out1) < out1_bytes) {
            fprintf(stderr, "ds4: Metal paired Q8_0 matvec received undersized activation buffers\n");
            return 0;
        }

        const uint64_t row_bytes = (in_dim / 32u) * 34u;
        const uint64_t weight0_bytes = out0_dim * row_bytes;
        const uint64_t weight1_bytes = out1_dim * row_bytes;
        if (weight0_offset > model_size || weight0_bytes > model_size - weight0_offset ||
            weight1_offset > model_size || weight1_bytes > model_size - weight1_offset) {
            fprintf(stderr, "ds4: Metal paired Q8_0 matvec range is outside the mapped model\n");
            return 0;
        }

        uint64_t inner0 = 0;
        uint64_t inner1 = 0;
        id<MTLBuffer> weight0buf =
            ds4_gpu_wrap_model_range(model_map, model_size,
                                     weight0_offset, weight0_bytes, &inner0);
        id<MTLBuffer> weight1buf =
            ds4_gpu_wrap_model_range(model_map, model_size,
                                     weight1_offset, weight1_bytes, &inner1);
        if (!weight0buf || !weight1buf) return 0;

        ds4_gpu_mv_dispatch dispatch0 = ds4_gpu_make_q8_0_mv_dispatch();
        ds4_gpu_mv_dispatch dispatch1 = ds4_gpu_make_q8_0_mv_dispatch();
        if (out0_dim > 65536u) dispatch0.nsg = 8;
        if (out1_dim > 65536u) dispatch1.nsg = 8;
        /* A common threadgroup shape is required to retain each standalone
         * reduction tree. Mixed 4/8-simdgroup extents use the existing fallback. */
        if (dispatch0.nsg != dispatch1.nsg) return 0;

        ds4_gpu_q8_0_matvec_args args0 = ds4_gpu_make_q8_0_mv_args(in_dim, out0_dim);
        ds4_gpu_q8_0_matvec_args args1 = ds4_gpu_make_q8_0_mv_args(in_dim, out1_dim);
        args0.nr0 = dispatch0.nr0;
        args1.nr0 = dispatch1.nr0;
        id<MTLComputePipelineState> pipeline =
            ds4_gpu_get_mul_mv_pipeline("kernel_mul_mv_q8_0_f32_pair", dispatch0.nsg);
        if (!pipeline) return 0;

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&args0 length:sizeof(args0) atIndex:0];
        [enc setBytes:&args1 length:sizeof(args1) atIndex:1];
        [enc setBuffer:weight0buf offset:(NSUInteger)inner0 atIndex:2];
        [enc setBuffer:weight1buf offset:(NSUInteger)inner1 atIndex:3];
        [enc setBuffer:xbuf offset:ds4_gpu_tensor_offset(x) atIndex:4];
        [enc setBuffer:out0buf offset:ds4_gpu_tensor_offset(out0) atIndex:5];
        [enc setBuffer:out1buf offset:ds4_gpu_tensor_offset(out1) atIndex:6];
        [enc setThreadgroupMemoryLength:2u * dispatch0.smem atIndex:0];
        const uint64_t max_out_dim = out0_dim > out1_dim ? out0_dim : out1_dim;
        [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)max_out_dim +
                                               (NSUInteger)dispatch0.nr0 - 1u) /
                                              (NSUInteger)dispatch0.nr0,
                                              1,
                                              1)
             threadsPerThreadgroup:MTLSizeMake(32, (NSUInteger)dispatch0.nsg, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "paired Q8_0 matvec")) return 0;
    }

    return 1;
}

int ds4_gpu_matmul_q8_0_f16_out_tensor(
        ds4_gpu_tensor       *out_h,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok) {
    (void)out_h; (void)model_map; (void)model_size; (void)weight_offset;
    (void)in_dim; (void)out_dim; (void)x; (void)n_tok;
    return 0;
}

static int ds4_gpu_shared_gate_up_swiglu_q8_0_impl(
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        float                   clamp,
        int                     store_gate_up,
        bool                    force_model_view) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!mid || !x || !model_map ||
        (store_gate_up && (!gate || !up)) ||
        (in_dim & 31u) != 0 ||
        in_dim > UINT32_MAX || out_dim > UINT32_MAX ||
        !isfinite(clamp) || clamp < 0.0f) {
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> xbuf = ds4_gpu_tensor_buffer(x);
        id<MTLBuffer> midbuf = ds4_gpu_tensor_buffer(mid);
        id<MTLBuffer> gatebuf = store_gate_up ?
            ds4_gpu_tensor_buffer(gate) : midbuf;
        id<MTLBuffer> upbuf = store_gate_up ?
            ds4_gpu_tensor_buffer(up) : midbuf;
        const uint64_t x_bytes = in_dim * sizeof(float);
        const uint64_t out_bytes = out_dim * sizeof(float);
        if (!xbuf || !gatebuf || !upbuf || !midbuf ||
            ds4_gpu_tensor_bytes(x) < x_bytes ||
            (store_gate_up && ds4_gpu_tensor_bytes(gate) < out_bytes) ||
            (store_gate_up && ds4_gpu_tensor_bytes(up) < out_bytes) ||
            ds4_gpu_tensor_bytes(mid) < out_bytes) {
            fprintf(stderr, "ds4: Metal shared expert fused gate/up received undersized activation buffers\n");
            return 0;
        }

        const uint64_t blocks = in_dim / 32;
        const uint64_t row_bytes = blocks * 34;
        const uint64_t weight_bytes = out_dim * row_bytes;
        if (gate_offset > model_size || weight_bytes > model_size - gate_offset ||
            up_offset > model_size || weight_bytes > model_size - up_offset) {
            fprintf(stderr, "ds4: Metal shared expert fused gate/up range is outside the mapped model\n");
            return 0;
        }

        const bool exact_decode_views =
            !force_model_view &&
            getenv("DS4_METAL_ENABLE_Q8_DECODE_EXACT_VIEWS") != NULL &&
            getenv("DS4_METAL_DISABLE_Q8_DECODE_EXACT_VIEWS") == NULL;
        uint64_t gate_inner = 0;
        uint64_t up_inner = 0;
        id<MTLBuffer> gate_wbuf = exact_decode_views ?
            ds4_gpu_wrap_model_exact_range(model_map,
                                           model_size,
                                           gate_offset,
                                           weight_bytes,
                                           &gate_inner) :
            ds4_gpu_wrap_model_range(model_map,
                                     model_size,
                                     gate_offset,
                                     weight_bytes,
                                     &gate_inner);
        id<MTLBuffer> up_wbuf = exact_decode_views ?
            ds4_gpu_wrap_model_exact_range(model_map,
                                           model_size,
                                           up_offset,
                                           weight_bytes,
                                           &up_inner) :
            ds4_gpu_wrap_model_range(model_map,
                                     model_size,
                                     up_offset,
                                     weight_bytes,
                                     &up_inner);
        if (!gate_wbuf || !up_wbuf) return 0;

        ds4_gpu_q8_0_matvec_args args = ds4_gpu_make_q8_0_mv_args(in_dim, out_dim);
        ds4_gpu_mv_dispatch mv_dispatch = ds4_gpu_make_q8_0_mv_dispatch();
        args.nr0 = mv_dispatch.nr0;
        const char *fn_name = store_gate_up ?
            (mv_dispatch.nr0 >= 4 ?
             "kernel_dsv4_shared_gate_up_swiglu_q8_0_r4" :
             "kernel_dsv4_shared_gate_up_swiglu_q8_0") :
            (mv_dispatch.nr0 >= 4 ?
             "kernel_dsv4_shared_mid_swiglu_q8_0_r4" :
             "kernel_dsv4_shared_mid_swiglu_q8_0");
        id<MTLComputePipelineState> pipeline =
            ds4_gpu_get_mul_mv_pipeline(fn_name, mv_dispatch.nsg);
        if (!pipeline) return 0;

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:gate_wbuf offset:(NSUInteger)gate_inner atIndex:1];
        [enc setBuffer:up_wbuf offset:(NSUInteger)up_inner atIndex:2];
        [enc setBuffer:xbuf offset:ds4_gpu_tensor_offset(x) atIndex:3];
        [enc setBuffer:gatebuf offset:(store_gate_up ?
                                       ds4_gpu_tensor_offset(gate) :
                                       ds4_gpu_tensor_offset(mid)) atIndex:4];
        [enc setBuffer:upbuf offset:(store_gate_up ?
                                     ds4_gpu_tensor_offset(up) :
                                     ds4_gpu_tensor_offset(mid)) atIndex:5];
        [enc setBuffer:midbuf offset:ds4_gpu_tensor_offset(mid) atIndex:6];
        [enc setBytes:&clamp length:sizeof(clamp) atIndex:7];
        [enc setThreadgroupMemoryLength:2u * mv_dispatch.smem atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)out_dim + (NSUInteger)mv_dispatch.nr0 - 1u) /
                                                  (NSUInteger)mv_dispatch.nr0,
                                              1,
                                              1)
             threadsPerThreadgroup:MTLSizeMake(32, (NSUInteger)mv_dispatch.nsg, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb,
                                           owned,
                                           store_gate_up ?
                                           "shared expert fused gate/up" :
                                           "shared expert fused mid")) {
            return 0;
        }
    }

    return 1;
}

int ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        float                   clamp) {
    return ds4_gpu_shared_gate_up_swiglu_q8_0_impl(gate,
                                                   up,
                                                   mid,
                                                   model_map,
                                                   model_size,
                                                   gate_offset,
                                                   up_offset,
                                                   in_dim,
                                                   out_dim,
                                                   x,
                                                   clamp,
                                                   1,
                                                   false);
}

int ds4_gpu_shared_mid_swiglu_q8_0_tensor(
        ds4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        float                   clamp) {
    return ds4_gpu_shared_gate_up_swiglu_q8_0_impl(NULL,
                                                   NULL,
                                                   mid,
                                                   model_map,
                                                   model_size,
                                                   gate_offset,
                                                   up_offset,
                                                   in_dim,
                                                   out_dim,
                                                   x,
                                                   clamp,
                                                   0,
                                                   true);
}

int ds4_gpu_shared_gate_up_swiglu_q8_0_model_view_tensor(
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        float                   clamp) {
    return ds4_gpu_shared_gate_up_swiglu_q8_0_impl(gate,
                                                   up,
                                                   mid,
                                                   model_map,
                                                   model_size,
                                                   gate_offset,
                                                   up_offset,
                                                   in_dim,
                                                   out_dim,
                                                   x,
                                                   clamp,
                                                   1,
                                                   true);
}

int ds4_gpu_shared_gate_up_swiglu_q8_0_rows_tensor(
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok,
        float                   clamp) {
    if (n_tok == 1) {
        return ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(gate,
                                                         up,
                                                         mid,
                                                         model_map,
                                                         model_size,
                                                         gate_offset,
                                                         up_offset,
                                                         in_dim,
                                                         out_dim,
                                                         x,
                                                         clamp);
    }
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!gate || !up || !mid || !x || !model_map ||
        n_tok == 0 ||
        (in_dim & 31u) != 0 || (in_dim % 128u) != 0 ||
        in_dim > UINT32_MAX || out_dim > UINT32_MAX || n_tok > UINT32_MAX ||
        !isfinite(clamp) || clamp < 0.0f) {
        return 0;
    }

    const uint64_t mv_ext_max_tokens =
        ds4_gpu_env_u64("DS4_METAL_Q8_MV_EXT_MAX_TOKENS", 16u, 2u, 128u);
    if (n_tok > mv_ext_max_tokens) return 0;

    @autoreleasepool {
        id<MTLBuffer> xbuf = ds4_gpu_tensor_buffer(x);
        id<MTLBuffer> gatebuf = ds4_gpu_tensor_buffer(gate);
        id<MTLBuffer> upbuf = ds4_gpu_tensor_buffer(up);
        id<MTLBuffer> midbuf = ds4_gpu_tensor_buffer(mid);
        const uint64_t x_bytes = n_tok * in_dim * sizeof(float);
        const uint64_t out_bytes = n_tok * out_dim * sizeof(float);
        if (!xbuf || !gatebuf || !upbuf || !midbuf ||
            ds4_gpu_tensor_bytes(x) < x_bytes ||
            ds4_gpu_tensor_bytes(gate) < out_bytes ||
            ds4_gpu_tensor_bytes(up) < out_bytes ||
            ds4_gpu_tensor_bytes(mid) < out_bytes) {
            fprintf(stderr, "ds4: Metal fused Q8_0 gate/up rows received undersized activation buffers\n");
            return 0;
        }

        const uint64_t blocks = in_dim / 32;
        const uint64_t row_bytes = blocks * 34;
        const uint64_t weight_bytes = out_dim * row_bytes;
        if (gate_offset > model_size || weight_bytes > model_size - gate_offset ||
            up_offset > model_size || weight_bytes > model_size - up_offset) {
            fprintf(stderr, "ds4: Metal fused Q8_0 gate/up rows range is outside the mapped model\n");
            return 0;
        }

        uint64_t gate_inner = 0;
        uint64_t up_inner = 0;
        id<MTLBuffer> gate_wbuf =
            ds4_gpu_wrap_model_range(model_map, model_size, gate_offset, weight_bytes, &gate_inner);
        id<MTLBuffer> up_wbuf =
            ds4_gpu_wrap_model_range(model_map, model_size, up_offset, weight_bytes, &up_inner);
        if (!gate_wbuf || !up_wbuf) return 0;

        const int16_t nsg = 2;
        const int16_t nxpsg = ds4_gpu_mv_ext_nxpsg(in_dim, n_tok);
        const int16_t r1ptg = ds4_gpu_mv_ext_r1ptg(n_tok);
        const char *fn_name = ds4_gpu_mv_ext_q8_pair_swiglu_name(r1ptg);
        id<MTLComputePipelineState> pipeline =
            fn_name ? ds4_gpu_get_mul_mv_ext_pipeline(fn_name, nsg, nxpsg) : nil;
        if (!pipeline) return 0;

        const int16_t nypsg = 32 / nxpsg;
        const uint64_t r0ptg = (uint64_t)nypsg * (uint64_t)nsg;
        ds4_gpu_mul_mv_ext_args args =
            ds4_gpu_make_mv_ext_args(in_dim, out_dim, n_tok, 34, row_bytes);

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:gate_wbuf offset:(NSUInteger)gate_inner atIndex:1];
        [enc setBuffer:up_wbuf offset:(NSUInteger)up_inner atIndex:2];
        [enc setBuffer:xbuf offset:ds4_gpu_tensor_offset(x) atIndex:3];
        [enc setBuffer:gatebuf offset:ds4_gpu_tensor_offset(gate) atIndex:4];
        [enc setBuffer:upbuf offset:ds4_gpu_tensor_offset(up) atIndex:5];
        [enc setBuffer:midbuf offset:ds4_gpu_tensor_offset(mid) atIndex:6];
        [enc setBytes:&clamp length:sizeof(clamp) atIndex:7];
        [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)out_dim + (NSUInteger)r0ptg - 1u) /
                                                  (NSUInteger)r0ptg,
                                              ((NSUInteger)n_tok + (NSUInteger)r1ptg - 1u) /
                                                  (NSUInteger)r1ptg,
                                              1)
             threadsPerThreadgroup:MTLSizeMake(32, (NSUInteger)nsg, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "fused Q8_0 gate/up rows")) {
            return 0;
        }
    }

    return 1;
}

int ds4_gpu_shared_gate_up_swiglu_q8_0_rows_scalar_tensor(
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok,
        float                   clamp) {
    if (n_tok == 1) {
        return ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(gate,
                                                         up,
                                                         mid,
                                                         model_map,
                                                         model_size,
                                                         gate_offset,
                                                         up_offset,
                                                         in_dim,
                                                         out_dim,
                                                         x,
                                                         clamp);
    }
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!gate || !up || !mid || !x || !model_map ||
        n_tok == 0 ||
        (in_dim & 31u) != 0 ||
        in_dim > UINT32_MAX || out_dim > UINT32_MAX || n_tok > UINT32_MAX ||
        in_dim > UINT64_MAX / sizeof(float) ||
        out_dim > UINT64_MAX / sizeof(float) ||
        !isfinite(clamp) || clamp < 0.0f) {
        return 0;
    }

    const uint64_t x_row_bytes = in_dim * sizeof(float);
    const uint64_t out_row_bytes = out_dim * sizeof(float);
    if ((x_row_bytes != 0 && n_tok > UINT64_MAX / x_row_bytes) ||
        (out_row_bytes != 0 && n_tok > UINT64_MAX / out_row_bytes)) {
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> xbuf = ds4_gpu_tensor_buffer(x);
        id<MTLBuffer> gatebuf = ds4_gpu_tensor_buffer(gate);
        id<MTLBuffer> upbuf = ds4_gpu_tensor_buffer(up);
        id<MTLBuffer> midbuf = ds4_gpu_tensor_buffer(mid);
        const uint64_t x_bytes = n_tok * x_row_bytes;
        const uint64_t out_bytes = n_tok * out_row_bytes;
        if (!xbuf || !gatebuf || !upbuf || !midbuf ||
            ds4_gpu_tensor_bytes(x) < x_bytes ||
            ds4_gpu_tensor_bytes(gate) < out_bytes ||
            ds4_gpu_tensor_bytes(up) < out_bytes ||
            ds4_gpu_tensor_bytes(mid) < out_bytes) {
            fprintf(stderr, "ds4: Metal shared expert scalar-row fused gate/up received undersized activation buffers\n");
            return 0;
        }

        const uint64_t blocks = in_dim / 32;
        if (blocks > UINT64_MAX / 34u) return 0;
        const uint64_t row_bytes = blocks * 34u;
        if (out_dim > UINT64_MAX / row_bytes) return 0;
        const uint64_t weight_bytes = out_dim * row_bytes;
        if (gate_offset > model_size || weight_bytes > model_size - gate_offset ||
            up_offset > model_size || weight_bytes > model_size - up_offset) {
            fprintf(stderr, "ds4: Metal shared expert scalar-row fused gate/up range is outside the mapped model\n");
            return 0;
        }

        uint64_t gate_inner = 0;
        uint64_t up_inner = 0;
        id<MTLBuffer> gate_wbuf =
            ds4_gpu_wrap_model_range(model_map, model_size, gate_offset, weight_bytes, &gate_inner);
        id<MTLBuffer> up_wbuf =
            ds4_gpu_wrap_model_range(model_map, model_size, up_offset, weight_bytes, &up_inner);
        if (!gate_wbuf || !up_wbuf) return 0;

        ds4_gpu_q8_0_matvec_args args = ds4_gpu_make_q8_0_mv_args(in_dim, out_dim);
        ds4_gpu_mv_dispatch mv_dispatch = ds4_gpu_make_q8_0_mv_dispatch();
        args.nr0 = mv_dispatch.nr0;
        args.ne11 = (int32_t)n_tok;
        args.nb12 = n_tok * x_row_bytes;
        args.nb13 = args.nb12;
        args.ne1 = (int32_t)n_tok;
        const char *fn_name = mv_dispatch.nr0 >= 4 ?
            "kernel_dsv4_shared_gate_up_swiglu_q8_0_r4" :
            "kernel_dsv4_shared_gate_up_swiglu_q8_0";
        id<MTLComputePipelineState> pipeline =
            ds4_gpu_get_mul_mv_pipeline(fn_name, mv_dispatch.nsg);
        if (!pipeline) return 0;

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:gate_wbuf offset:(NSUInteger)gate_inner atIndex:1];
        [enc setBuffer:up_wbuf offset:(NSUInteger)up_inner atIndex:2];
        [enc setBuffer:xbuf offset:ds4_gpu_tensor_offset(x) atIndex:3];
        [enc setBuffer:gatebuf offset:ds4_gpu_tensor_offset(gate) atIndex:4];
        [enc setBuffer:upbuf offset:ds4_gpu_tensor_offset(up) atIndex:5];
        [enc setBuffer:midbuf offset:ds4_gpu_tensor_offset(mid) atIndex:6];
        [enc setBytes:&clamp length:sizeof(clamp) atIndex:7];
        [enc setThreadgroupMemoryLength:2u * mv_dispatch.smem atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(
                ((NSUInteger)out_dim + (NSUInteger)mv_dispatch.nr0 - 1u) /
                    (NSUInteger)mv_dispatch.nr0,
                (NSUInteger)n_tok,
                1)
             threadsPerThreadgroup:
                MTLSizeMake(32, (NSUInteger)mv_dispatch.nsg, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "shared expert scalar-row fused gate/up")) {
            return 0;
        }
    }

    return 1;
}

int ds4_gpu_matmul_f16_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (in_dim > UINT32_MAX || out_dim > UINT32_MAX || n_tok > UINT32_MAX) return 0;

    @autoreleasepool {
        id<MTLBuffer> xbuf = ds4_gpu_tensor_buffer(x);
        id<MTLBuffer> outbuf = ds4_gpu_tensor_buffer(out);
        const uint64_t x_bytes = n_tok * in_dim * sizeof(float);
        const uint64_t out_bytes = n_tok * out_dim * sizeof(float);
        if (!xbuf || !outbuf ||
            ds4_gpu_tensor_bytes(x) < x_bytes ||
            ds4_gpu_tensor_bytes(out) < out_bytes) {
            fprintf(stderr, "ds4: Metal F16 tensor matmul received undersized activation buffers\n");
            return 0;
        }

        const uint64_t row_bytes = in_dim * sizeof(uint16_t);
        const uint64_t weight_bytes = row_bytes * out_dim;
        if (weight_offset > model_size || weight_bytes > model_size - weight_offset) {
            fprintf(stderr, "ds4: Metal F16 tensor matmul range is outside the mapped model\n");
            return 0;
        }

        uint64_t inner_offset = 0;
        id<MTLBuffer> wbuf =
            ds4_gpu_wrap_f32_decode_model_range(model_map,
                                                model_size,
                                                weight_offset,
                                                weight_bytes,
                                                n_tok,
                                                &inner_offset);
        if (!wbuf) return 0;

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        if (n_tok == 1) {
            ds4_gpu_f16_matvec_args mv_args = ds4_gpu_make_f16_mv_args(in_dim, out_dim);
            ds4_gpu_mv_dispatch mv_dispatch =
                ds4_gpu_make_plain_mv_dispatch(in_dim, 0);
            if (!g_quality_mode && (out_dim == 512u || out_dim == 1024u) && in_dim >= 4096u) {
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
            if (mv_dispatch.smem) {
                [enc setThreadgroupMemoryLength:mv_dispatch.smem atIndex:0];
            }
            [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)out_dim + (NSUInteger)mv_dispatch.nr0 - 1u) / (NSUInteger)mv_dispatch.nr0,
                                                  1,
                                                  1)
                 threadsPerThreadgroup:MTLSizeMake(32, (NSUInteger)mv_dispatch.nsg, 1)];
            ds4_gpu_end_compute_encoder(cb, enc);

            if (!ds4_gpu_finish_command_buffer(cb, owned, "F16 tensor matvec")) return 0;
            return 1;
        }

        if (n_tok <= 8 && (in_dim % 128u) == 0) {
            const int16_t nsg = 2;
            const int16_t nxpsg = ds4_gpu_mv_ext_nxpsg(in_dim, n_tok);
            const int16_t r1ptg = ds4_gpu_mv_ext_r1ptg(n_tok);
            const char *fn_name = ds4_gpu_mv_ext_name(0, r1ptg);
            id<MTLComputePipelineState> pipeline =
                fn_name ? ds4_gpu_get_mul_mv_ext_pipeline(fn_name, nsg, nxpsg) : nil;
            if (!pipeline) return 0;

            const int16_t nypsg = 32 / nxpsg;
            const uint64_t r0ptg = (uint64_t)nypsg * (uint64_t)nsg;
            ds4_gpu_mul_mv_ext_args args =
                ds4_gpu_make_mv_ext_args(in_dim, out_dim, n_tok, sizeof(uint16_t), row_bytes);

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

            if (!ds4_gpu_finish_command_buffer(cb, owned, "F16 tensor mul_mv_ext")) return 0;
            return 1;
        }

        /*
         * Same direct-RHS TensorOps structure as Q8_0, but for F16 model
         * matrices.  The 128-token RHS tile is kept when the batch alignment
         * allows it because the later tile_n=64 retest was neutral/slower.
         */
        if (ds4_gpu_mpp_available() &&
            n_tok >= 32u &&
            (in_dim % 32u) == 0 &&
            (out_dim % 64u) == 0 &&
            (n_tok % 32u) == 0) {
            uint64_t nax_tile_n = 32u;
            if ((n_tok % 128u) == 0) {
                nax_tile_n = 128u;
            } else if ((n_tok % 64u) == 0) {
                nax_tile_n = 64u;
            }
            const char *nax_fn = nax_tile_n == 128u
                ? "kernel_mul_mm_f16_f32_mpp_direct_rhs_n128"
                : (nax_tile_n == 64u
                    ? "kernel_mul_mm_f16_f32_mpp_direct_rhs_n64"
                    : "kernel_mul_mm_f16_f32_mpp_direct_rhs");
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

                if (!ds4_gpu_finish_command_buffer(cb, owned, "F16 NAX tensor matmul")) {
                    return 0;
                }
                return 1;
            }
            ds4_gpu_warn_mpp_fallback();
        }

        const bool bc_inp = (in_dim % 32u) != 0;
        const bool bc_out = (out_dim % 64u) != 0 || (n_tok % 32u) != 0;
        id<MTLComputePipelineState> pipeline =
            ds4_gpu_get_mul_mm_pipeline("kernel_mul_mm_f16_f32", bc_inp, bc_out);
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

        if (!ds4_gpu_finish_command_buffer(cb, owned, "F16 tensor matmul")) return 0;
    }

    return 1;
}

int ds4_gpu_matmul_f16_pair_tensor(
        ds4_gpu_tensor       *out_a,
        ds4_gpu_tensor       *out_b,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_a_offset,
        uint64_t                weight_b_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (in_dim > UINT32_MAX || out_dim > UINT32_MAX || n_tok != 1 || (in_dim & 3u) != 0) return 0;

    @autoreleasepool {
        id<MTLBuffer> xbuf = ds4_gpu_tensor_buffer(x);
        id<MTLBuffer> outabuf = ds4_gpu_tensor_buffer(out_a);
        id<MTLBuffer> outbbuf = ds4_gpu_tensor_buffer(out_b);
        const uint64_t x_bytes = in_dim * sizeof(float);
        const uint64_t out_bytes = out_dim * sizeof(float);
        if (!xbuf || !outabuf || !outbbuf ||
            ds4_gpu_tensor_bytes(x) < x_bytes ||
            ds4_gpu_tensor_bytes(out_a) < out_bytes ||
            ds4_gpu_tensor_bytes(out_b) < out_bytes) {
            fprintf(stderr, "ds4: Metal F16 paired matvec received undersized activation buffers\n");
            return 0;
        }

        const uint64_t row_bytes = in_dim * sizeof(uint16_t);
        const uint64_t weight_bytes = row_bytes * out_dim;
        if (weight_a_offset > model_size || weight_bytes > model_size - weight_a_offset ||
            weight_b_offset > model_size || weight_bytes > model_size - weight_b_offset) {
            fprintf(stderr, "ds4: Metal F16 paired matvec range is outside the mapped model\n");
            return 0;
        }

        uint64_t inner_a = 0;
        uint64_t inner_b = 0;
        id<MTLBuffer> wabuf = ds4_gpu_wrap_model_range(model_map, model_size,
                                                         weight_a_offset, weight_bytes,
                                                         &inner_a);
        id<MTLBuffer> wbbuf = ds4_gpu_wrap_model_range(model_map, model_size,
                                                         weight_b_offset, weight_bytes,
                                                         &inner_b);
        if (!wabuf || !wbbuf) return 0;

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        ds4_gpu_f16_matvec_args mv_args = ds4_gpu_make_f16_mv_args(in_dim, out_dim);
        ds4_gpu_mv_dispatch mv_dispatch = ds4_gpu_make_plain_mv_dispatch(in_dim, 0);
        if (ds4_gpu_use_compressor_pair_nr4() &&
            (out_dim == 512u || out_dim == 1024u) && in_dim >= 4096u) {
            mv_dispatch.nr0 = 4;
            mv_dispatch.smem = 32u * 4u * sizeof(float);
        }
        mv_args.nr0 = mv_dispatch.nr0;
        id<MTLComputePipelineState> pipeline =
            ds4_gpu_get_mul_mv_pipeline("kernel_mul_mv_f16_f32_pair_4", mv_dispatch.nsg);
        if (!pipeline) return 0;

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&mv_args length:sizeof(mv_args) atIndex:0];
        [enc setBuffer:wabuf offset:(NSUInteger)inner_a atIndex:1];
        [enc setBuffer:wbbuf offset:(NSUInteger)inner_b atIndex:2];
        [enc setBuffer:xbuf offset:ds4_gpu_tensor_offset(x) atIndex:3];
        [enc setBuffer:outabuf offset:ds4_gpu_tensor_offset(out_a) atIndex:4];
        [enc setBuffer:outbbuf offset:ds4_gpu_tensor_offset(out_b) atIndex:5];
        if (mv_dispatch.smem) {
            [enc setThreadgroupMemoryLength:mv_dispatch.smem atIndex:0];
        }
        [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)out_dim + (NSUInteger)mv_dispatch.nr0 - 1u) / (NSUInteger)mv_dispatch.nr0,
                                              1,
                                              1)
             threadsPerThreadgroup:MTLSizeMake(32, (NSUInteger)mv_dispatch.nsg, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "F16 paired matvec")) return 0;
    }

    return 1;
}

int ds4_gpu_matmul_f16_pair_compressor_store_tensor(
        ds4_gpu_tensor       *out_kv,
        ds4_gpu_tensor       *out_score,
        ds4_gpu_tensor       *state_kv,
        ds4_gpu_tensor       *state_score,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_kv_offset,
        uint64_t                weight_score_offset,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint64_t                in_dim,
        uint32_t                width,
        const ds4_gpu_tensor *x,
        uint32_t                ratio,
        uint32_t                pos) {
    if (!g_initialized && !ds4_gpu_init()) return -1;
    const bool force =
        getenv("DS4_METAL_ENABLE_COMPRESSOR_PAIR_STATE_STORE") != NULL;
    if ((g_quality_mode ||
         (!ds4_gpu_device_name_contains("M3") &&
          !ds4_gpu_device_name_contains("M5") && !force)) ||
        getenv("DS4_METAL_DISABLE_M3_COMPRESSOR_PAIR_STATE_STORE") != NULL ||
        getenv("DS4_METAL_DISABLE_COMPRESSOR_PAIR_PROJ") != NULL ||
        getenv("DS4_METAL_DISABLE_COMPRESSOR_STORE_ONE") != NULL) {
        return 0;
    }
    if (!out_kv || !out_score || !state_kv || !state_score || !model_map || !x ||
        in_dim == 0 || width == 0 || ratio == 0 ||
        (ape_type != 0u && ape_type != 1u)) {
        return -1;
    }
    if (in_dim != 4096u ||
        (width != 256u && width != 512u && width != 1024u) ||
        (ratio != 4u && ratio != 128u)) {
        return 0;
    }

    @autoreleasepool {
        const uint32_t state_rows = ratio == 4u ? 2u * ratio : ratio;
        const uint64_t row_bytes = in_dim * sizeof(uint16_t);
        const uint64_t weight_bytes = row_bytes * width;
        const uint64_t out_bytes = (uint64_t)width * sizeof(float);
        const uint64_t state_bytes =
            (uint64_t)state_rows * width * sizeof(float);
        const uint64_t ape_elem = ape_type == 1u ? sizeof(uint16_t) : sizeof(float);
        const uint64_t ape_bytes = (uint64_t)ratio * width * ape_elem;
        if (weight_kv_offset > model_size ||
            weight_bytes > model_size - weight_kv_offset ||
            weight_score_offset > model_size ||
            weight_bytes > model_size - weight_score_offset ||
            ape_offset > model_size || ape_bytes > model_size - ape_offset) {
            return -1;
        }

        id<MTLBuffer> xbuf = ds4_gpu_tensor_buffer(x);
        id<MTLBuffer> outkvbuf = ds4_gpu_tensor_buffer(out_kv);
        id<MTLBuffer> outscorebuf = ds4_gpu_tensor_buffer(out_score);
        id<MTLBuffer> statekvbuf = ds4_gpu_tensor_buffer(state_kv);
        id<MTLBuffer> statescbuf = ds4_gpu_tensor_buffer(state_score);
        if (!xbuf || !outkvbuf || !outscorebuf || !statekvbuf || !statescbuf ||
            ds4_gpu_tensor_bytes(x) < in_dim * sizeof(float) ||
            ds4_gpu_tensor_bytes(out_kv) < out_bytes ||
            ds4_gpu_tensor_bytes(out_score) < out_bytes ||
            ds4_gpu_tensor_bytes(state_kv) < state_bytes ||
            ds4_gpu_tensor_bytes(state_score) < state_bytes) {
            return -1;
        }

        uint64_t weight_kv_inner = 0;
        uint64_t weight_score_inner = 0;
        uint64_t ape_inner = 0;
        id<MTLBuffer> weightkvbuf = ds4_gpu_wrap_model_range(
            model_map, model_size, weight_kv_offset, weight_bytes,
            &weight_kv_inner);
        id<MTLBuffer> weightscorebuf = ds4_gpu_wrap_model_range(
            model_map, model_size, weight_score_offset, weight_bytes,
            &weight_score_inner);
        id<MTLBuffer> apebuf = ds4_gpu_wrap_model_range(
            model_map, model_size, ape_offset, ape_bytes, &ape_inner);
        if (!weightkvbuf || !weightscorebuf || !apebuf) return -1;

        ds4_gpu_f16_matvec_args mv_args =
            ds4_gpu_make_f16_mv_args(in_dim, width);
        ds4_gpu_mv_dispatch mv_dispatch =
            ds4_gpu_make_plain_mv_dispatch(in_dim, 0);
        if (ds4_gpu_use_compressor_pair_nr4() &&
            (width == 512u || width == 1024u) && in_dim >= 4096u) {
            mv_dispatch.nr0 = 4;
            mv_dispatch.smem = 32u * 4u * sizeof(float);
        }
        mv_args.nr0 = mv_dispatch.nr0;
        ds4_gpu_dsv4_compressor_store_one_args store_args = {
            .width = width,
            .ratio = ratio,
            .pos = pos,
            .ape_type = ape_type,
        };
        id<MTLComputePipelineState> pipeline = ds4_gpu_get_mul_mv_pipeline(
            "kernel_mul_mv_f16_f32_pair_compressor_store_4",
            mv_dispatch.nsg);
        if (!pipeline) return -1;

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return -1;
        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&mv_args length:sizeof(mv_args) atIndex:0];
        [enc setBytes:&store_args length:sizeof(store_args) atIndex:1];
        [enc setBuffer:weightkvbuf offset:(NSUInteger)weight_kv_inner atIndex:2];
        [enc setBuffer:weightscorebuf offset:(NSUInteger)weight_score_inner atIndex:3];
        [enc setBuffer:xbuf offset:ds4_gpu_tensor_offset(x) atIndex:4];
        [enc setBuffer:outkvbuf offset:ds4_gpu_tensor_offset(out_kv) atIndex:5];
        [enc setBuffer:outscorebuf offset:ds4_gpu_tensor_offset(out_score) atIndex:6];
        [enc setBuffer:apebuf offset:(NSUInteger)ape_inner atIndex:7];
        [enc setBuffer:statekvbuf offset:ds4_gpu_tensor_offset(state_kv) atIndex:8];
        [enc setBuffer:statescbuf offset:ds4_gpu_tensor_offset(state_score) atIndex:9];
        if (mv_dispatch.smem) {
            [enc setThreadgroupMemoryLength:mv_dispatch.smem atIndex:0];
        }
        [enc dispatchThreadgroups:MTLSizeMake(
                ((NSUInteger)width + (NSUInteger)mv_dispatch.nr0 - 1u) /
                    (NSUInteger)mv_dispatch.nr0,
                1,
                1)
             threadsPerThreadgroup:MTLSizeMake(
                32, (NSUInteger)mv_dispatch.nsg, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(
                cb, owned, "F16 paired matvec compressor state store")) {
            return -1;
        }
    }

    return 1;
}

int ds4_gpu_matmul_f32_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (in_dim > UINT32_MAX || out_dim > UINT32_MAX || n_tok == 0 || n_tok > UINT32_MAX) return 0;
    if (in_dim > UINT64_MAX / n_tok / sizeof(float) ||
        out_dim > UINT64_MAX / n_tok / sizeof(float)) {
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
            fprintf(stderr, "ds4: Metal F32 tensor matmul received undersized activation buffers\n");
            return 0;
        }

        const uint64_t row_bytes = in_dim * sizeof(float);
        const uint64_t weight_bytes = row_bytes * out_dim;
        if (weight_offset > model_size || weight_bytes > model_size - weight_offset) {
            fprintf(stderr, "ds4: Metal F32 tensor matmul range is outside the mapped model\n");
            return 0;
        }

        uint64_t inner_offset = 0;
        id<MTLBuffer> wbuf =
            ds4_gpu_wrap_f32_decode_model_range(model_map,
                                                model_size,
                                                weight_offset,
                                                weight_bytes,
                                                n_tok,
                                                &inner_offset);
        if (!wbuf) return 0;

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        if (n_tok == 1) {
            ds4_gpu_q8_0_matvec_args mv_args = ds4_gpu_make_f32_mv_args(in_dim, out_dim, 1);
            ds4_gpu_mv_dispatch mv_dispatch = ds4_gpu_make_plain_mv_dispatch(in_dim, 1);
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
            if (mv_dispatch.smem) {
                [enc setThreadgroupMemoryLength:mv_dispatch.smem atIndex:0];
            }
            [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)out_dim + (NSUInteger)mv_dispatch.nr0 - 1u) / (NSUInteger)mv_dispatch.nr0,
                                                  1,
                                                  1)
                 threadsPerThreadgroup:MTLSizeMake(32, (NSUInteger)mv_dispatch.nsg, 1)];
            ds4_gpu_end_compute_encoder(cb, enc);

            if (!ds4_gpu_finish_command_buffer(cb, owned, "F32 tensor matvec")) return 0;
            return 1;
        }

        if (n_tok <= 8 && (in_dim % 128u) == 0) {
            const int16_t nsg = 2;
            const int16_t nxpsg = ds4_gpu_mv_ext_nxpsg(in_dim, n_tok);
            const int16_t r1ptg = ds4_gpu_mv_ext_r1ptg(n_tok);
            const char *fn_name = ds4_gpu_mv_ext_f32_name(r1ptg);
            id<MTLComputePipelineState> pipeline =
                fn_name ? ds4_gpu_get_mul_mv_ext_pipeline(fn_name, nsg, nxpsg) : nil;
            if (!pipeline) return 0;

            const int16_t nypsg = 32 / nxpsg;
            const uint64_t r0ptg = (uint64_t)nypsg * (uint64_t)nsg;
            ds4_gpu_mul_mv_ext_args args =
                ds4_gpu_make_mv_ext_args(in_dim, out_dim, n_tok, sizeof(float), row_bytes);

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

            if (!ds4_gpu_finish_command_buffer(cb, owned, "F32 tensor mul_mv_ext")) return 0;
            return 1;
        }

        /* Generic multi-row path (GLM prefill shapes: one grid row per
         * token through the plain matvec pipeline). */
        {
            ds4_gpu_q8_0_matvec_args mv_args = ds4_gpu_make_f32_mv_args(in_dim, out_dim, n_tok);
            ds4_gpu_mv_dispatch mv_dispatch = ds4_gpu_make_plain_mv_dispatch(in_dim, 1);
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
            if (mv_dispatch.smem) {
                [enc setThreadgroupMemoryLength:mv_dispatch.smem atIndex:0];
            }
            [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)out_dim + (NSUInteger)mv_dispatch.nr0 - 1u) / (NSUInteger)mv_dispatch.nr0,
                                                  (NSUInteger)n_tok,
                                                  1)
                 threadsPerThreadgroup:MTLSizeMake(32, (NSUInteger)mv_dispatch.nsg, 1)];
            ds4_gpu_end_compute_encoder(cb, enc);

            if (!ds4_gpu_finish_command_buffer(cb, owned, "F32 tensor matmul")) return 0;
        }
    }

    return 1;
}

int ds4_gpu_repeat_hc_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *row,
        uint32_t                n_embd,
        uint32_t                n_hc) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!out || !row || n_embd == 0 || n_hc == 0) return 0;

    @autoreleasepool {
        id<MTLBuffer> rowbuf = ds4_gpu_tensor_buffer(row);
        id<MTLBuffer> outbuf = ds4_gpu_tensor_buffer(out);
        const uint64_t row_bytes = (uint64_t)n_embd * sizeof(float);
        const uint64_t out_bytes = row_bytes * n_hc;
        if (!rowbuf || !outbuf ||
            ds4_gpu_tensor_bytes(row) < row_bytes ||
            ds4_gpu_tensor_bytes(out) < out_bytes) {
            fprintf(stderr, "ds4: Metal HC repeat received undersized buffers\n");
            return 0;
        }

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;
        if (!ds4_gpu_encode_repeat_hc_embedding(cb,
                                                  rowbuf,
                                                  ds4_gpu_tensor_offset(row),
                                                  outbuf,
                                                  ds4_gpu_tensor_offset(out),
                                                  1,
                                                  n_embd,
                                                  n_hc)) {
            return 0;
        }
        if (!ds4_gpu_finish_command_buffer(cb, owned, "HC repeat")) return 0;
    }

    return 1;
}

int ds4_gpu_repeat_hc_rows_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *rows,
        uint32_t                n_tokens,
        uint32_t                n_embd,
        uint32_t                n_hc) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!out || !rows || n_tokens == 0 || n_embd == 0 || n_hc == 0) return 0;

    @autoreleasepool {
        id<MTLBuffer> rowsbuf = ds4_gpu_tensor_buffer(rows);
        id<MTLBuffer> outbuf = ds4_gpu_tensor_buffer(out);
        const uint64_t row_bytes = (uint64_t)n_embd * sizeof(float);
        const uint64_t rows_bytes = (uint64_t)n_tokens * row_bytes;
        const uint64_t out_bytes = rows_bytes * n_hc;
        if (!rowsbuf || !outbuf ||
            ds4_gpu_tensor_bytes(rows) < rows_bytes ||
            ds4_gpu_tensor_bytes(out) < out_bytes) {
            fprintf(stderr, "ds4: Metal HC row repeat received undersized buffers\n");
            return 0;
        }

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;
        if (!ds4_gpu_encode_repeat_hc_embedding(cb,
                                                  rowsbuf,
                                                  ds4_gpu_tensor_offset(rows),
                                                  outbuf,
                                                  ds4_gpu_tensor_offset(out),
                                                  n_tokens,
                                                  n_embd,
                                                  n_hc)) {
            return 0;
        }
        if (!ds4_gpu_finish_command_buffer(cb, owned, "HC row repeat")) return 0;
    }

    return 1;
}

int ds4_gpu_rms_norm_plain_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *x,
        uint32_t                n,
        float                   eps) {
    return ds4_gpu_rms_norm_plain_rows_tensor(out, x, n, 1, eps);
}

int ds4_gpu_rms_norm_plain_rows_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *x,
        uint32_t                n,
        uint32_t                rows,
        float                   eps) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (n == 0 || rows == 0 || (n & 3u) != 0) return 0;

    @autoreleasepool {
        id<MTLBuffer> xbuf = ds4_gpu_tensor_buffer(x);
        id<MTLBuffer> outbuf = ds4_gpu_tensor_buffer(out);
        const uint64_t bytes = (uint64_t)n * rows * sizeof(float);
        if (!xbuf || !outbuf ||
            ds4_gpu_tensor_bytes(x) < bytes ||
            ds4_gpu_tensor_bytes(out) < bytes) {
            fprintf(stderr, "ds4: Metal plain RMS norm received undersized activation buffers\n");
            return 0;
        }

        ds4_gpu_rms_norm_args args = ds4_gpu_make_rms_norm_args(n, rows, eps);
        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:g_rms_norm_plain_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:xbuf offset:ds4_gpu_tensor_offset(x) atIndex:1];
        [enc setBuffer:xbuf offset:ds4_gpu_tensor_offset(x) atIndex:2];
        [enc setBuffer:xbuf offset:ds4_gpu_tensor_offset(x) atIndex:3];
        [enc setBuffer:outbuf offset:ds4_gpu_tensor_offset(out) atIndex:4];
        [enc setThreadgroupMemoryLength:32u * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(ds4_gpu_rms_norm_threads(n), 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "plain RMS norm")) return 0;
    }

    return 1;
}

static int ds4_gpu_hc_rms_scale_project_mode(
        uint32_t in_dim,
        uint32_t out_dim,
        uint32_t n_rows) {
    const bool hard_shape = n_rows > 8u &&
        (in_dim == 16384u || in_dim == 28672u) && out_dim == 24u;
    if (!hard_shape) return 0;

    const bool force =
        getenv("DS4_METAL_ENABLE_HC_RMS_SCALE_PROJ") != NULL;
    if (getenv("DS4_METAL_DISABLE_M3_HC_RMS_SCALE_PROJ") != NULL ||
        (!ds4_gpu_device_name_contains("M3") && !force)) {
        return 0;
    }
    if (g_rms_norm_scale_pipeline == nil) {
        return force ? -1 : 0;
    }
    return 1;
}

int ds4_gpu_hc_rms_scale_project_f16_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *scale_scratch,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                in_dim,
        uint32_t                out_dim,
        const ds4_gpu_tensor *x,
        uint32_t                n_rows,
        float                   eps) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!out || !scale_scratch || !model_map || !x ||
        in_dim == 0u || out_dim == 0u || n_rows == 0u ||
        n_rows > INT32_MAX) {
        return 0;
    }

    const int mode =
        ds4_gpu_hc_rms_scale_project_mode(in_dim, out_dim, n_rows);
    if (mode < 0) return 0;
    if (mode == 0) {
        return ds4_gpu_rms_norm_plain_rows_tensor(
                   scale_scratch, x, in_dim, n_rows, eps) != 0 &&
               ds4_gpu_matmul_f16_tensor(
                   out, model_map, model_size, weight_offset,
                   in_dim, out_dim, scale_scratch, n_rows) != 0;
    }

    @autoreleasepool {
        const uint64_t x_elems = (uint64_t)in_dim * n_rows;
        const uint64_t out_elems = (uint64_t)out_dim * n_rows;
        if (x_elems > UINT64_MAX / sizeof(float) ||
            out_elems > UINT64_MAX / sizeof(float)) {
            return 0;
        }
        const uint64_t x_bytes = x_elems * sizeof(float);
        const uint64_t out_bytes = out_elems * sizeof(float);
        const uint64_t scale_bytes = (uint64_t)n_rows * sizeof(float);

        id<MTLBuffer> xbuf = ds4_gpu_tensor_buffer(x);
        id<MTLBuffer> outbuf = ds4_gpu_tensor_buffer(out);
        id<MTLBuffer> scalebuf = ds4_gpu_tensor_buffer(scale_scratch);
        if (!xbuf || !outbuf || !scalebuf ||
            ds4_gpu_tensor_bytes(x) < x_bytes ||
            ds4_gpu_tensor_bytes(out) < out_bytes ||
            ds4_gpu_tensor_bytes(scale_scratch) < scale_bytes) {
            fprintf(stderr, "ds4: Metal HC RMS scale projection received undersized activation buffers\n");
            return 0;
        }

        const uint64_t weight_row_bytes = (uint64_t)in_dim * sizeof(uint16_t);
        if ((uint64_t)out_dim > UINT64_MAX / weight_row_bytes) return 0;
        const uint64_t weight_bytes = (uint64_t)out_dim * weight_row_bytes;
        if (weight_offset > model_size ||
            weight_bytes > model_size - weight_offset) {
            fprintf(stderr, "ds4: Metal HC RMS scale projection weight range is outside the mapped model\n");
            return 0;
        }

        const bool bc_inp = (in_dim % 32u) != 0u;
        const bool bc_out = (out_dim % 64u) != 0u ||
                            (n_rows % 32u) != 0u;
        id<MTLComputePipelineState> mm_pipeline =
            ds4_gpu_get_mul_mm_pipeline(
                "kernel_mul_mm_f16_f32_scaled", bc_inp, bc_out);
        if (!mm_pipeline) {
            if (getenv("DS4_METAL_ENABLE_HC_RMS_SCALE_PROJ") != NULL) {
                return 0;
            }
            return ds4_gpu_rms_norm_plain_rows_tensor(
                       scale_scratch, x, in_dim, n_rows, eps) != 0 &&
                   ds4_gpu_matmul_f16_tensor(
                       out, model_map, model_size, weight_offset,
                       in_dim, out_dim, scale_scratch, n_rows) != 0;
        }

        id<MTLComputePipelineState> scale_pipeline = ds4_gpu_hot_pipeline(
            g_rms_norm_scale_pipeline, "kernel_rms_norm_scale_f32_4");
        if (!scale_pipeline) return 0;

        uint64_t weight_inner = 0;
        id<MTLBuffer> weightbuf = ds4_gpu_wrap_model_range(
            model_map, model_size, weight_offset, weight_bytes, &weight_inner);
        if (!weightbuf) return 0;

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        ds4_gpu_rms_norm_args norm_args =
            ds4_gpu_make_rms_norm_args(in_dim, n_rows, eps);
        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:scale_pipeline];
        [enc setBytes:&norm_args length:sizeof(norm_args) atIndex:0];
        [enc setBuffer:xbuf offset:ds4_gpu_tensor_offset(x) atIndex:1];
        [enc setBuffer:scalebuf
                offset:ds4_gpu_tensor_offset(scale_scratch)
               atIndex:2];
        [enc setThreadgroupMemoryLength:32u * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(n_rows, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(
                 ds4_gpu_rms_norm_threads(in_dim), 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        ds4_gpu_mul_mm_args mm_args = ds4_gpu_make_mm_args(
            in_dim, out_dim, n_rows, weight_row_bytes);
        enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:mm_pipeline];
        [enc setBytes:&mm_args length:sizeof(mm_args) atIndex:0];
        [enc setBuffer:weightbuf offset:(NSUInteger)weight_inner atIndex:1];
        [enc setBuffer:xbuf offset:ds4_gpu_tensor_offset(x) atIndex:2];
        [enc setBuffer:outbuf offset:ds4_gpu_tensor_offset(out) atIndex:3];
        [enc setBuffer:scalebuf
                offset:ds4_gpu_tensor_offset(scale_scratch)
               atIndex:4];
        [enc setThreadgroupMemoryLength:(bc_out ? 8192u : 6144u) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(
                                          ((NSUInteger)n_rows + 31u) / 32u,
                                          ((NSUInteger)out_dim + 63u) / 64u,
                                          1)
             threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(
                cb, owned, "HC RMS scale F16 projection")) {
            return 0;
        }
    }

    return 1;
}

int ds4_gpu_rms_norm_weight_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *x,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                n,
        float                   eps) {
    return ds4_gpu_rms_norm_weight_rows_tensor(out, x, model_map, model_size, weight_offset, n, 1, eps);
}

int ds4_gpu_rms_norm_weight_rows_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *x,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                n,
        uint32_t                rows,
        float                   eps) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (n == 0 || rows == 0 || (n & 3u) != 0) return 0;

    @autoreleasepool {
        id<MTLBuffer> xbuf = ds4_gpu_tensor_buffer(x);
        id<MTLBuffer> outbuf = ds4_gpu_tensor_buffer(out);
        const uint64_t row_bytes = (uint64_t)n * sizeof(float);
        const uint64_t bytes = row_bytes * rows;
        if (!xbuf || !outbuf ||
            ds4_gpu_tensor_bytes(x) < bytes ||
            ds4_gpu_tensor_bytes(out) < bytes) {
            fprintf(stderr, "ds4: Metal weighted RMS norm received undersized activation buffers\n");
            return 0;
        }
        if (weight_offset > model_size || row_bytes > model_size - weight_offset) {
            fprintf(stderr, "ds4: Metal weighted RMS norm range is outside the mapped model\n");
            return 0;
        }

        const bool exact_decode_weight_view =
            rows == 1u &&
            row_bytes <= (1ull << 20) &&
            getenv("DS4_METAL_ENABLE_DECODE_NORM_EXACT_VIEWS") != NULL &&
            getenv("DS4_METAL_DISABLE_DECODE_NORM_EXACT_VIEWS") == NULL;
        uint64_t inner_offset = 0;
        id<MTLBuffer> wbuf = exact_decode_weight_view ?
            ds4_gpu_wrap_model_exact_range(model_map,
                                           model_size,
                                           weight_offset,
                                           row_bytes,
                                           &inner_offset) :
            ds4_gpu_wrap_model_range(model_map,
                                     model_size,
                                     weight_offset,
                                     row_bytes,
                                     &inner_offset);
        if (!wbuf) return 0;

        ds4_gpu_rms_norm_args args = ds4_gpu_make_rms_norm_args(n, rows, eps);
        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:g_rms_norm_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:xbuf offset:ds4_gpu_tensor_offset(x) atIndex:1];
        [enc setBuffer:wbuf offset:(NSUInteger)inner_offset atIndex:2];
        [enc setBuffer:xbuf offset:ds4_gpu_tensor_offset(x) atIndex:3];
        [enc setBuffer:outbuf offset:ds4_gpu_tensor_offset(out) atIndex:4];
        [enc setThreadgroupMemoryLength:32u * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(ds4_gpu_rms_norm_threads(n), 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "weighted RMS norm")) return 0;
    }

    return 1;
}

int ds4_gpu_add_rms_norm_weight_tensor(
        ds4_gpu_tensor       *norm_out,
        ds4_gpu_tensor       *sum_out,
        const ds4_gpu_tensor *a,
        const ds4_gpu_tensor *b,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                n,
        float                   eps) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!norm_out || !sum_out || !a || !b || n == 0 || (n & 3u) != 0) return 0;

    @autoreleasepool {
        id<MTLBuffer> abuf = ds4_gpu_tensor_buffer(a);
        id<MTLBuffer> bbuf = ds4_gpu_tensor_buffer(b);
        id<MTLBuffer> sumbuf = ds4_gpu_tensor_buffer(sum_out);
        id<MTLBuffer> normbuf = ds4_gpu_tensor_buffer(norm_out);
        const uint64_t row_bytes = (uint64_t)n * sizeof(float);
        if (!abuf || !bbuf || !sumbuf || !normbuf ||
            ds4_gpu_tensor_bytes(a) < row_bytes ||
            ds4_gpu_tensor_bytes(b) < row_bytes ||
            ds4_gpu_tensor_bytes(sum_out) < row_bytes ||
            ds4_gpu_tensor_bytes(norm_out) < row_bytes) {
            fprintf(stderr, "ds4: Metal add+RMS norm received undersized activation buffers\n");
            return 0;
        }
        if (weight_offset > model_size || row_bytes > model_size - weight_offset) {
            fprintf(stderr, "ds4: Metal add+RMS norm range is outside the mapped model\n");
            return 0;
        }

        const bool exact_decode_weight_view =
            row_bytes <= (1ull << 20) &&
            getenv("DS4_METAL_ENABLE_DECODE_NORM_EXACT_VIEWS") != NULL &&
            getenv("DS4_METAL_DISABLE_DECODE_NORM_EXACT_VIEWS") == NULL;
        uint64_t inner_offset = 0;
        id<MTLBuffer> wbuf = exact_decode_weight_view ?
            ds4_gpu_wrap_model_exact_range(model_map,
                                           model_size,
                                           weight_offset,
                                           row_bytes,
                                           &inner_offset) :
            ds4_gpu_wrap_model_range(model_map,
                                     model_size,
                                     weight_offset,
                                     row_bytes,
                                     &inner_offset);
        if (!wbuf) return 0;

        ds4_gpu_rms_norm_args args = ds4_gpu_make_rms_norm_args(n, 1, eps);
        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:g_add_rms_norm_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:abuf offset:ds4_gpu_tensor_offset(a) atIndex:1];
        [enc setBuffer:bbuf offset:ds4_gpu_tensor_offset(b) atIndex:2];
        [enc setBuffer:wbuf offset:(NSUInteger)inner_offset atIndex:3];
        [enc setBuffer:sumbuf offset:ds4_gpu_tensor_offset(sum_out) atIndex:4];
        [enc setBuffer:normbuf offset:ds4_gpu_tensor_offset(norm_out) atIndex:5];
        [enc setThreadgroupMemoryLength:32u * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(ds4_gpu_rms_norm_threads(n), 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "add+RMS norm")) return 0;
    }

    return 1;
}

int ds4_gpu_dsv4_qkv_rms_norm_rows_tensor(
        ds4_gpu_tensor       *q_out,
        const ds4_gpu_tensor *q,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                q_weight_offset,
        uint32_t                q_n,
        ds4_gpu_tensor       *kv_out,
        const ds4_gpu_tensor *kv,
        uint64_t                kv_weight_offset,
        uint32_t                kv_n,
        uint32_t                rows,
        float                   eps) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!q_out || !q || !kv_out || !kv || q_n == 0 || kv_n == 0 || rows == 0 ||
        (q_n & 3u) != 0 || (kv_n & 3u) != 0) {
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> qbuf = ds4_gpu_tensor_buffer(q);
        id<MTLBuffer> qoutbuf = ds4_gpu_tensor_buffer(q_out);
        id<MTLBuffer> kvbuf = ds4_gpu_tensor_buffer(kv);
        id<MTLBuffer> kvoutbuf = ds4_gpu_tensor_buffer(kv_out);

        const uint64_t q_row_bytes = (uint64_t)q_n * sizeof(float);
        const uint64_t kv_row_bytes = (uint64_t)kv_n * sizeof(float);
        if (!qbuf || !qoutbuf || !kvbuf || !kvoutbuf ||
            ds4_gpu_tensor_bytes(q) < q_row_bytes * rows ||
            ds4_gpu_tensor_bytes(q_out) < q_row_bytes * rows ||
            ds4_gpu_tensor_bytes(kv) < kv_row_bytes * rows ||
            ds4_gpu_tensor_bytes(kv_out) < kv_row_bytes * rows) {
            fprintf(stderr, "ds4: Metal fused q/kv RMS norm received undersized activation buffers\n");
            return 0;
        }
        if (q_weight_offset > model_size || q_row_bytes > model_size - q_weight_offset ||
            kv_weight_offset > model_size || kv_row_bytes > model_size - kv_weight_offset) {
            fprintf(stderr, "ds4: Metal fused q/kv RMS norm weight range is outside the mapped model\n");
            return 0;
        }

        uint64_t q_inner_offset = 0;
        uint64_t kv_inner_offset = 0;
        id<MTLBuffer> q_wbuf = ds4_gpu_wrap_model_range(model_map, model_size,
                                                          q_weight_offset, q_row_bytes,
                                                          &q_inner_offset);
        if (!q_wbuf) return 0;
        id<MTLBuffer> kv_wbuf = ds4_gpu_wrap_model_range(model_map, model_size,
                                                           kv_weight_offset, kv_row_bytes,
                                                           &kv_inner_offset);
        if (!kv_wbuf) return 0;

        ds4_gpu_qkv_rms_norm_args args = {
            .q_n = (int32_t)q_n,
            .q_n4 = (int32_t)(q_n / 4u),
            .kv_n = (int32_t)kv_n,
            .kv_n4 = (int32_t)(kv_n / 4u),
            .q_row_stride = q_row_bytes,
            .kv_row_stride = kv_row_bytes,
            .eps = eps,
        };

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:g_dsv4_qkv_rms_norm_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:qbuf offset:ds4_gpu_tensor_offset(q) atIndex:1];
        [enc setBuffer:q_wbuf offset:(NSUInteger)q_inner_offset atIndex:2];
        [enc setBuffer:qoutbuf offset:ds4_gpu_tensor_offset(q_out) atIndex:3];
        [enc setBuffer:kvbuf offset:ds4_gpu_tensor_offset(kv) atIndex:4];
        [enc setBuffer:kv_wbuf offset:(NSUInteger)kv_inner_offset atIndex:5];
        [enc setBuffer:kvoutbuf offset:ds4_gpu_tensor_offset(kv_out) atIndex:6];
        [enc setThreadgroupMemoryLength:32u * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(rows, 2, 1)
             threadsPerThreadgroup:MTLSizeMake(ds4_gpu_rms_norm_threads(q_n), 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "fused q/kv RMS norm")) return 0;
    }

    return 1;
}

int ds4_gpu_head_rms_norm_tensor(
        ds4_gpu_tensor *x,
        uint32_t          n_tok,
        uint32_t          n_head,
        uint32_t          head_dim,
        float             eps) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!x || n_tok == 0 || n_head == 0 || head_dim == 0 || (head_dim & 3u) != 0) return 0;

    @autoreleasepool {
        id<MTLBuffer> xbuf = ds4_gpu_tensor_buffer(x);
        const uint64_t bytes = (uint64_t)n_tok * n_head * head_dim * sizeof(float);
        if (!xbuf || ds4_gpu_tensor_bytes(x) < bytes) {
            fprintf(stderr, "ds4: Metal head RMS norm received undersized activation buffer\n");
            return 0;
        }

        ds4_gpu_rms_norm_args args = ds4_gpu_make_rms_norm_3d_args(head_dim, n_head, n_tok, eps);

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:g_rms_norm_plain_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:xbuf offset:ds4_gpu_tensor_offset(x) atIndex:1];
        [enc setBuffer:xbuf offset:ds4_gpu_tensor_offset(x) atIndex:2];
        [enc setBuffer:xbuf offset:ds4_gpu_tensor_offset(x) atIndex:3];
        [enc setBuffer:xbuf offset:ds4_gpu_tensor_offset(x) atIndex:4];
        [enc setThreadgroupMemoryLength:32u * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(n_head, n_tok, 1)
             threadsPerThreadgroup:MTLSizeMake(ds4_gpu_rms_norm_pipeline_threads(head_dim, g_rms_norm_plain_pipeline), 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "head RMS norm")) return 0;
    }

    return 1;
}

int ds4_gpu_rope_tail_tensor(
        ds4_gpu_tensor *x,
        uint32_t          n_tok,
        uint32_t          n_head,
        uint32_t          head_dim,
        uint32_t          n_rot,
        uint32_t          pos0,
        uint32_t          n_ctx_orig,
        bool              inverse,
        float             freq_base,
        float             freq_scale,
        float             ext_factor,
        float             attn_factor,
        float             beta_fast,
        float             beta_slow) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!x || n_tok == 0 || n_head == 0 || head_dim == 0 || n_rot > head_dim || (n_rot & 1u) != 0) {
        return 0;
    }
    if (n_rot == 0) return 1;

    @autoreleasepool {
        id<MTLBuffer> xbuf = ds4_gpu_tensor_buffer(x);
        const uint64_t bytes = (uint64_t)n_tok * n_head * head_dim * sizeof(float);
        if (!xbuf || ds4_gpu_tensor_bytes(x) < bytes) {
            fprintf(stderr, "ds4: Metal RoPE received undersized activation buffer\n");
            return 0;
        }

        ds4_gpu_rope_tail_batch_args args = ds4_gpu_make_rope_tail_args(
            n_tok, n_head, head_dim, n_rot, n_ctx_orig, inverse,
            freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        if (!ds4_gpu_encode_rope_tail_inplace(cb,
                                                xbuf,
                                                ds4_gpu_tensor_offset(x),
                                                &args,
                                                n_tok,
                                                n_head,
                                                head_dim,
                                                pos0,
                                                1)) {
            return 0;
        }

        if (!ds4_gpu_finish_command_buffer(cb, owned, "RoPE tail")) return 0;
    }

    return 1;
}

int ds4_gpu_head_rms_norm_rope_tail_tensor(
        ds4_gpu_tensor *x,
        uint32_t          n_tok,
        uint32_t          n_head,
        uint32_t          head_dim,
        uint32_t          n_rot,
        uint32_t          pos0,
        uint32_t          n_ctx_orig,
        bool              inverse,
        float             freq_base,
        float             freq_scale,
        float             ext_factor,
        float             attn_factor,
        float             beta_fast,
        float             beta_slow,
        float             eps) {
    (void)x; (void)n_tok; (void)n_head; (void)head_dim; (void)n_rot;
    (void)pos0; (void)n_ctx_orig; (void)inverse; (void)freq_base;
    (void)freq_scale; (void)ext_factor; (void)attn_factor;
    (void)beta_fast; (void)beta_slow; (void)eps;
    return 0;
}

int ds4_gpu_attn_q_b_f16_head_rms_rope_tail_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *q_half,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint64_t              in_dim,
        uint64_t              out_dim,
        const ds4_gpu_tensor *x,
        uint32_t              n_tok,
        uint32_t              n_head,
        uint32_t              head_dim,
        uint32_t              n_rot,
        uint32_t              pos0,
        uint32_t              n_ctx_orig,
        bool                  inverse,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow,
        float                 eps) {
    (void)out; (void)q_half; (void)model_map; (void)model_size;
    (void)weight_offset; (void)in_dim; (void)out_dim; (void)x;
    (void)n_tok; (void)n_head; (void)head_dim; (void)n_rot; (void)pos0;
    (void)n_ctx_orig; (void)inverse; (void)freq_base; (void)freq_scale;
    (void)ext_factor; (void)attn_factor; (void)beta_fast; (void)beta_slow;
    (void)eps;
    return 0;
}

int ds4_gpu_dsv4_fp8_kv_quantize_tensor(
        ds4_gpu_tensor *x,
        uint32_t          n_tok,
        uint32_t          head_dim,
        uint32_t          n_rot) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!x || n_tok == 0 || head_dim == 0 || n_rot > head_dim) return 0;
    if (n_rot == head_dim) return 1;

    @autoreleasepool {
        id<MTLBuffer> xbuf = ds4_gpu_tensor_buffer(x);
        const uint64_t bytes = (uint64_t)n_tok * head_dim * sizeof(float);
        if (!xbuf || ds4_gpu_tensor_bytes(x) < bytes) {
            fprintf(stderr, "ds4: Metal DSV4 FP8 KV quantize received undersized activation buffer\n");
            return 0;
        }

        ds4_gpu_dsv4_fp8_kv_quantize_args args = {
            .ne00 = head_dim,
            .ne01 = n_tok,
            .ne02 = 1,
            .ne03 = 1,
            .nb00 = sizeof(float),
            .nb01 = (uint64_t)head_dim * sizeof(float),
            .nb02 = (uint64_t)n_tok * head_dim * sizeof(float),
            .nb03 = (uint64_t)n_tok * head_dim * sizeof(float),
            .nb0 = sizeof(float),
            .nb1 = (uint64_t)head_dim * sizeof(float),
            .nb2 = (uint64_t)n_tok * head_dim * sizeof(float),
            .nb3 = (uint64_t)n_tok * head_dim * sizeof(float),
            .n_rot = (int32_t)n_rot,
        };

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:g_dsv4_fp8_kv_quantize_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:xbuf offset:ds4_gpu_tensor_offset(x) atIndex:1];
        [enc setBuffer:xbuf offset:ds4_gpu_tensor_offset(x) atIndex:2];
        [enc setThreadgroupMemoryLength:64u * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(n_tok, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "DSV4 FP8 KV quantize")) return 0;
    }

    return 1;
}

int ds4_gpu_dsv4_indexer_qat_tensor(
        ds4_gpu_tensor *x,
        uint32_t          n_rows,
        uint32_t          head_dim) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!x || n_rows == 0 || head_dim != 128u) return 0;

    @autoreleasepool {
        id<MTLBuffer> xbuf = ds4_gpu_tensor_buffer(x);
        const uint64_t bytes = (uint64_t)n_rows * head_dim * sizeof(float);
        if (!xbuf || ds4_gpu_tensor_bytes(x) < bytes) {
            fprintf(stderr, "ds4: Metal DSV4 indexer QAT received undersized activation buffer\n");
            return 0;
        }

        ds4_gpu_dsv4_indexer_qat_args args = {
            .n_rows = n_rows,
            .head_dim = head_dim,
            .row_stride = (uint64_t)head_dim * sizeof(float),
        };

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:g_dsv4_indexer_qat_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:xbuf offset:ds4_gpu_tensor_offset(x) atIndex:1];
        [enc setThreadgroupMemoryLength:256u * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(n_rows, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "DSV4 indexer Hadamard+FP4")) return 0;
    }

    return 1;
}

static void ds4_gpu_set_rows_thread_shape(
        uint32_t    width,
        NSUInteger *nth_out,
        NSUInteger *nrptg_out) {
    const NSUInteger nk0 = width ? (NSUInteger)width : 1u;
    const NSUInteger max_threads = g_set_rows_f32_i32_pipeline
        ? (NSUInteger)g_set_rows_f32_i32_pipeline.maxTotalThreadsPerThreadgroup
        : 1024u;

    NSUInteger nth = 32u;
    while (nth < nk0 && nth < max_threads) {
        nth *= 2u;
    }

    NSUInteger nrptg = 1u;
    if (nth > nk0) {
        nrptg = (nth + nk0 - 1u) / nk0;
        nth = nk0;
        if (nrptg * nth > max_threads) {
            nrptg--;
        }
    }

    if (nth > nk0) nth = nk0;
    if (nth == 0u) nth = 1u;
    if (nrptg == 0u) nrptg = 1u;

    *nth_out = nth;
    *nrptg_out = nrptg;
}

static int ds4_gpu_encode_f16_round_copy_for_raw_store(
        id<MTLCommandBuffer>   cb,
        const ds4_gpu_tensor *src,
        uint32_t               n) {
    id<MTLBuffer> srcbuf = ds4_gpu_tensor_buffer(src);
    const uint64_t src_bytes = (uint64_t)n * sizeof(float);
    if (!srcbuf || ds4_gpu_tensor_bytes(src) < src_bytes) {
        fprintf(stderr, "ds4: Metal raw KV store received undersized source buffer\n");
        return 0;
    }
    if (!ds4_gpu_ensure_scratch_buffer(&g_f16_round_scratch_buffer,
                                         &g_f16_round_scratch_bytes,
                                         (NSUInteger)n * sizeof(uint16_t),
                                         "ds4_f16_round_scratch") ||
        !ds4_gpu_ensure_scratch_buffer(&g_raw_store_round_buffer,
                                         &g_raw_store_round_bytes,
                                         (NSUInteger)n * sizeof(float),
                                         "ds4_raw_store_round")) {
        return 0;
    }

    if (!ds4_gpu_encode_cpy_f32_f16_1d(cb,
                                         srcbuf,
                                         ds4_gpu_tensor_offset(src),
                                         g_f16_round_scratch_buffer,
                                         0,
                                         n)) {
        return 0;
    }
    return ds4_gpu_encode_cpy_f16_f32_1d(cb,
                                           g_f16_round_scratch_buffer,
                                           0,
                                           g_raw_store_round_buffer,
                                           0,
                                           n);
}

static int ds4_gpu_encode_set_rows_f32_i32(
        id<MTLCommandBuffer> cb,
        ds4_gpu_tensor    *dst,
        id<MTLBuffer>        srcbuf,
        NSUInteger           src_off,
        const int32_t       *rows,
        uint32_t             n_rows,
        uint32_t             dst_rows,
        uint32_t             width) {
    id<MTLBuffer> dstbuf = ds4_gpu_tensor_buffer(dst);
    const uint64_t dst_bytes = (uint64_t)dst_rows * width * sizeof(float);
    const uint64_t src_bytes = (uint64_t)n_rows * width * sizeof(float);
    if (!dstbuf || !srcbuf || !rows || n_rows == 0 || width == 0 ||
        ds4_gpu_tensor_bytes(dst) < dst_bytes ||
        src_bytes > NSUIntegerMax - src_off) {
        fprintf(stderr, "ds4: Metal DS4 set_rows received invalid buffers\n");
        return 0;
    }

    const uint64_t row_bytes = (uint64_t)width * sizeof(float);
    const uint64_t rows_bytes = (uint64_t)n_rows * sizeof(int32_t);
    ds4_gpu_set_rows_args args = {
        .nk0 = (int32_t)width,
        .ne01 = (int32_t)n_rows,
        .nb01 = row_bytes,
        .nb02 = (uint64_t)n_rows * row_bytes,
        .nb03 = (uint64_t)n_rows * row_bytes,
        .ne11 = 1,
        .ne12 = 1,
        .nb10 = sizeof(int32_t),
        .nb11 = rows_bytes,
        .nb12 = rows_bytes,
        .nb1 = row_bytes,
        .nb2 = (uint64_t)dst_rows * row_bytes,
        .nb3 = (uint64_t)dst_rows * row_bytes,
    };

    NSUInteger nth;
    NSUInteger nrptg;
    ds4_gpu_set_rows_thread_shape(width, &nth, &nrptg);

    id<MTLBuffer> rowsbuf = nil;
    if (rows_bytes > 4096u) {
        rowsbuf = ds4_gpu_new_transient_buffer((NSUInteger)rows_bytes, "ds4_set_rows_indices");
        if (!rowsbuf) return 0;
        memcpy([rowsbuf contents], rows, (NSUInteger)rows_bytes);
    }

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:g_set_rows_f32_i32_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:srcbuf offset:src_off atIndex:1];
    if (rowsbuf) {
        [enc setBuffer:rowsbuf offset:0 atIndex:2];
    } else {
        [enc setBytes:rows length:(NSUInteger)rows_bytes atIndex:2];
    }
    [enc setBuffer:dstbuf offset:ds4_gpu_tensor_offset(dst) atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)n_rows + nrptg - 1u) / nrptg, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(nth, nrptg, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);

    return 1;
}

static int ds4_gpu_encode_add_f32_1d(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        a,
        NSUInteger           a_off,
        id<MTLBuffer>        b,
        NSUInteger           b_off,
        id<MTLBuffer>        out,
        NSUInteger           out_off,
        uint32_t             n) {
    if (!cb || !a || !b || !out || n == 0) return 0;

    ds4_gpu_add_flat_args args = { .n = n };
    NSUInteger nth = g_add2_pipeline.maxTotalThreadsPerThreadgroup;
    if (nth > 256u) nth = 256u;
    if (nth > (NSUInteger)n) nth = (NSUInteger)n;
    if (nth == 0u) nth = 1u;
    const NSUInteger groups = ((NSUInteger)n + nth - 1u) / nth;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:g_add2_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:a offset:a_off atIndex:1];
    [enc setBuffer:b offset:b_off atIndex:2];
    [enc setBuffer:out offset:out_off atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}
