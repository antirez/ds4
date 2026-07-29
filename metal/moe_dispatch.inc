static NSUInteger ds4_gpu_bin_threads(uint32_t width, id<MTLComputePipelineState> pipeline) {
    NSUInteger nth_max = pipeline.maxTotalThreadsPerThreadgroup;
    if (nth_max > 256u) nth_max = 256u;
    NSUInteger nth = 1u;
    while (2u * nth < (NSUInteger)width && nth < nth_max) nth *= 2u;
    return nth ? nth : 1u;
}

static int ds4_gpu_encode_unary_f32_rows(
        id<MTLCommandBuffer>        cb,
        id<MTLComputePipelineState> pipeline,
        id<MTLBuffer>               src,
        NSUInteger                  src_off,
        id<MTLBuffer>               dst,
        NSUInteger                  dst_off,
        uint32_t                    width,
        uint32_t                    rows,
        int                         c4,
        float                       min,
        float                       max) {
    if (!cb || !pipeline || !src || !dst || width == 0 || rows == 0) return 0;
    if (c4 && (width & 3u) != 0) return 0;

    ds4_gpu_unary_args args = ds4_gpu_make_unary_rows_args(width, rows, c4, 0.0f, 0.0f);
    args.min = min;
    args.max = max;

    NSUInteger nth_max = pipeline.maxTotalThreadsPerThreadgroup;
    if (nth_max > 256u) nth_max = 256u;
    NSUInteger nth = (NSUInteger)args.ne00;
    if (nth > nth_max) nth = nth_max;
    if (nth == 0) nth = 1u;
    const NSUInteger nk0 = ((NSUInteger)args.ne00 + nth - 1u) / nth;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:src offset:src_off atIndex:1];
    [enc setBuffer:dst offset:dst_off atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake(nk0 * (NSUInteger)args.ne01,
                                          (NSUInteger)args.ne02,
                                          (NSUInteger)args.ne03)
         threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

static int ds4_gpu_encode_bin_f32_rows(
        id<MTLCommandBuffer>        cb,
        id<MTLComputePipelineState> pipeline,
        const ds4_gpu_bin_args   *args,
        id<MTLBuffer>               a,
        NSUInteger                  a_off,
        id<MTLBuffer>               b,
        NSUInteger                  b_off,
        id<MTLBuffer>               out,
        NSUInteger                  out_off) {
    if (!cb || !pipeline || !args || !a || !b || !out || args->ne0 <= 0 || args->ne1 <= 0) {
        return 0;
    }

    const NSUInteger nth = ds4_gpu_bin_threads((uint32_t)args->ne0, pipeline);
    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:pipeline];
    [enc setBytes:args length:sizeof(*args) atIndex:0];
    [enc setBuffer:a offset:a_off atIndex:1];
    [enc setBuffer:b offset:b_off atIndex:2];
    [enc setBuffer:out offset:out_off atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)args->ne1,
                                          (NSUInteger)args->ne2,
                                          (NSUInteger)args->ne3)
         threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

static ds4_gpu_bin_args ds4_gpu_make_bin_rowwise_scalar_args(uint32_t width, uint32_t rows) {
    const uint64_t lhs_row_bytes = (uint64_t)width * sizeof(float);
    const uint64_t rhs_row_bytes = sizeof(float);
    return (ds4_gpu_bin_args) {
        .ne00 = (int32_t)width,
        .ne01 = (int32_t)rows,
        .ne02 = 1,
        .ne03 = 1,
        .nb00 = sizeof(float),
        .nb01 = lhs_row_bytes,
        .nb02 = (uint64_t)rows * lhs_row_bytes,
        .nb03 = (uint64_t)rows * lhs_row_bytes,
        .ne10 = 1,
        .ne11 = (int32_t)rows,
        .ne12 = 1,
        .ne13 = 1,
        .nb10 = sizeof(float),
        .nb11 = rhs_row_bytes,
        .nb12 = (uint64_t)rows * rhs_row_bytes,
        .nb13 = (uint64_t)rows * rhs_row_bytes,
        .ne0 = (int32_t)width,
        .ne1 = (int32_t)rows,
        .ne2 = 1,
        .ne3 = 1,
        .nb0 = sizeof(float),
        .nb1 = lhs_row_bytes,
        .nb2 = (uint64_t)rows * lhs_row_bytes,
        .nb3 = (uint64_t)rows * lhs_row_bytes,
        .offs = 0,
        .o1 = { 0 },
    };
}

static ds4_gpu_mul_mv_id_args ds4_gpu_make_mul_mv_id_args(
        uint32_t src0_cols,
        uint32_t src0_rows,
        uint32_t src0_experts,
        uint64_t src0_row_bytes,
        uint64_t src0_expert_bytes,
        uint32_t src1_expert_rows,
        uint32_t selected_experts,
        uint32_t n_tokens,
        uint32_t nr0) {
    const uint64_t src1_row_bytes = (uint64_t)src0_cols * sizeof(float);
    const uint64_t src0_blocks = src0_cols / 256u;
    const uint64_t src0_block_bytes = src0_blocks ? src0_row_bytes / src0_blocks : 1u;
    return (ds4_gpu_mul_mv_id_args) {
        .nei0 = (int32_t)selected_experts,
        .nei1 = (int32_t)n_tokens,
        .nbi1 = (uint64_t)selected_experts * sizeof(int32_t),
        .ne00 = (int32_t)src0_cols,
        .ne01 = (int32_t)src0_rows,
        .ne02 = (int32_t)src0_experts,
        .nb00 = src0_block_bytes,
        .nb01 = src0_row_bytes,
        .nb02 = src0_expert_bytes,
        .ne10 = (int32_t)src0_cols,
        .ne11 = (int32_t)src1_expert_rows,
        .ne12 = (int32_t)n_tokens,
        .ne13 = 1,
        .nb10 = sizeof(float),
        .nb11 = src1_row_bytes,
        .nb12 = (uint64_t)src1_expert_rows * src1_row_bytes,
        .ne0 = (int32_t)src0_rows,
        .ne1 = (int32_t)selected_experts,
        .nb1 = (uint64_t)src0_rows * sizeof(float),
        .nr0 = (int32_t)nr0,
    };
}

static ds4_gpu_mul_mm_id_map_args ds4_gpu_make_mul_mm_id_map_args(
        uint32_t src0_cols,
        uint32_t src0_experts,
        uint32_t src1_expert_rows,
        uint32_t selected_experts,
        uint32_t n_tokens) {
    const uint64_t src1_row_bytes = (uint64_t)src0_cols * sizeof(float);
    return (ds4_gpu_mul_mm_id_map_args) {
        .ne02 = (int32_t)src0_experts,
        .ne10 = (int32_t)src0_cols,
        .ne11 = (int32_t)src1_expert_rows,
        .nb11 = src1_row_bytes,
        .nb12 = (uint64_t)src1_expert_rows * src1_row_bytes,
        .ne21 = (int32_t)n_tokens,
        .ne20 = (int32_t)selected_experts,
        .nb21 = (uint64_t)selected_experts * sizeof(int32_t),
    };
}

static ds4_gpu_mul_mm_id_args ds4_gpu_make_mul_mm_id_args(
        uint32_t src0_cols,
        uint32_t src0_rows,
        uint32_t src0_experts,
        uint64_t src0_row_bytes,
        uint64_t src0_expert_bytes,
        uint32_t src1_expert_rows,
        uint32_t selected_experts,
        uint32_t n_tokens) {
    return ds4_gpu_make_mul_mm_id_args_src1_size(src0_cols,
                                                   src0_rows,
                                                   src0_experts,
                                                   src0_row_bytes,
                                                   src0_expert_bytes,
                                                   src1_expert_rows,
                                                   selected_experts,
                                                   n_tokens,
                                                   sizeof(float));
}

static ds4_gpu_mul_mm_id_args ds4_gpu_make_mul_mm_id_args_src1_size(
        uint32_t src0_cols,
        uint32_t src0_rows,
        uint32_t src0_experts,
        uint64_t src0_row_bytes,
        uint64_t src0_expert_bytes,
        uint32_t src1_expert_rows,
        uint32_t selected_experts,
        uint32_t n_tokens,
        uint32_t src1_elem_size) {
    const uint64_t src1_row_bytes = (uint64_t)src0_cols * src1_elem_size;
    return (ds4_gpu_mul_mm_id_args) {
        .ne00 = (int32_t)src0_cols,
        .ne02 = (int32_t)src0_experts,
        .nb01 = src0_row_bytes,
        .nb02 = src0_expert_bytes,
        .nb03 = (uint64_t)src0_experts * src0_expert_bytes,
        .ne11 = (int32_t)src1_expert_rows,
        .nb10 = src1_elem_size,
        .nb11 = src1_row_bytes,
        .nb12 = (uint64_t)src1_expert_rows * src1_row_bytes,
        .nb13 = (uint64_t)n_tokens * (uint64_t)src1_expert_rows * src1_row_bytes,
        .ne20 = (int32_t)selected_experts,
        .ne21 = (int32_t)n_tokens,
        .ne0 = (int32_t)src0_rows,
        .ne1 = (int32_t)selected_experts,
        .r2 = 1,
        .r3 = 1,
    };
}

static uint32_t ds4_gpu_routed_mv_nr0(uint32_t type) {
    switch (type) {
    case DS4_METAL_TENSOR_Q8_0:    return 2;
    case DS4_METAL_TENSOR_Q8_K:    return 2;
    case DS4_METAL_TENSOR_Q4_K:    return 2;
    case DS4_METAL_TENSOR_Q2_K:
    case DS4_METAL_TENSOR_IQ2_XXS: return 4;
    default:                       return 0;
    }
}

static const char *ds4_gpu_metal_tensor_type_name(uint32_t type) {
    switch (type) {
    case DS4_METAL_TENSOR_IQ2_XXS: return "iq2_xxs";
    case DS4_METAL_TENSOR_Q2_K:    return "q2_k";
    case DS4_METAL_TENSOR_Q4_K:    return "q4_k";
    case DS4_METAL_TENSOR_Q5_K:    return "q5_k";
    case DS4_METAL_TENSOR_Q6_K:    return "q6_k";
    default:                       return "unknown";
    }
}

static const char *ds4_gpu_trim_env_value(const char *env, size_t *len_out) {
    if (len_out) *len_out = 0;
    if (!env) return NULL;

    while (isspace((unsigned char)*env)) env++;
    size_t n = strlen(env);
    while (n > 0 && isspace((unsigned char)env[n - 1])) n--;
    if (len_out) *len_out = n;
    return env;
}

static bool ds4_gpu_profile_layer_value_match(const char *env, uint32_t layer_index) {
    size_t env_len = 0;
    env = ds4_gpu_trim_env_value(env, &env_len);
    if (!env || env_len == 0) return true;
    if (ds4_gpu_env_value_eq(env, env_len, "all")) return true;

    const char *p = env;
    const char *end_env = env + env_len;
    while (p < end_env) {
        while (p < end_env && (*p == ' ' || *p == '\t' || *p == ',')) p++;
        if (p >= end_env) break;

        char *end = NULL;
        const unsigned long first = strtoul(p, &end, 10);
        if (end == p || end > end_env || first > UINT32_MAX) return false;

        unsigned long last = first;
        p = end;
        if (p < end_env && *p == '-') {
            p++;
            last = strtoul(p, &end, 10);
            if (end == p || end > end_env || last > UINT32_MAX) return false;
            p = end;
        }

        if (first <= layer_index && layer_index <= last) return true;
        while (p < end_env && (*p == ' ' || *p == '\t')) p++;
        if (p < end_env && *p != ',') return false;
    }
    return false;
}

static bool ds4_gpu_stage_profile_enabled_for_layer(const char *flag_env_name,
                                                    const char *layer_env_name,
                                                    uint32_t    layer_index) {
    size_t flag_len = 0;
    const char *flag = ds4_gpu_trim_env_value(getenv(flag_env_name), &flag_len);
    if (!flag) return false;

    size_t layer_len = 0;
    const char *layer = ds4_gpu_trim_env_value(getenv(layer_env_name), &layer_len);
    const bool has_layer_filter = layer && layer_len != 0;

    if (flag_len != 0) {
        if (ds4_gpu_env_value_eq(flag, flag_len, "0") ||
            ds4_gpu_env_value_eq(flag, flag_len, "false") ||
            ds4_gpu_env_value_eq(flag, flag_len, "no") ||
            ds4_gpu_env_value_eq(flag, flag_len, "off")) {
            return false;
        }
        if (!has_layer_filter &&
            !ds4_gpu_env_value_eq(flag, flag_len, "1") &&
            !ds4_gpu_env_value_eq(flag, flag_len, "true") &&
            !ds4_gpu_env_value_eq(flag, flag_len, "yes") &&
            !ds4_gpu_env_value_eq(flag, flag_len, "on") &&
            !ds4_gpu_env_value_eq(flag, flag_len, "all")) {
            return ds4_gpu_profile_layer_value_match(flag, layer_index);
        }
    }

    return ds4_gpu_profile_layer_value_match(layer, layer_index);
}

static NSUInteger ds4_gpu_routed_mv_smem(uint32_t type) {
    if (type == DS4_METAL_TENSOR_Q8_0) {
        return 32u * 2u * sizeof(float);
    }
    if (type == DS4_METAL_TENSOR_IQ2_XXS) {
        return 256u * sizeof(uint64_t) + 128u * sizeof(uint8_t);
    }
    return 0;
}

static NSUInteger ds4_gpu_routed_mv_nsg(uint32_t type) {
    return type == DS4_METAL_TENSOR_Q8_0 ? 4u : 2u;
}

static bool ds4_gpu_routed_mv_rows_per_group_is_nr0(uint32_t type) {
    return type == DS4_METAL_TENSOR_Q8_0;
}

static id<MTLComputePipelineState> ds4_gpu_routed_mv_pipeline(uint32_t type) {
    switch (type) {
    case DS4_METAL_TENSOR_Q8_0:
        return ds4_gpu_get_mul_mv_pipeline("kernel_mul_mv_id_q8_0_f32", 4);
    case DS4_METAL_TENSOR_Q8_K:
        return ds4_gpu_get_mul_mv_pipeline("kernel_mul_mv_id_q8_K_f32", 2);
    case DS4_METAL_TENSOR_IQ2_XXS: return g_moe_mul_mv_id_iq2_xxs_pipeline;
    case DS4_METAL_TENSOR_Q2_K:    return g_moe_mul_mv_id_q2_k_pipeline;
    case DS4_METAL_TENSOR_Q4_K:    return g_moe_mul_mv_id_q4_k_pipeline;
    default:                       return nil;
    }
}

static id<MTLComputePipelineState> ds4_gpu_routed_mm_pipeline(uint32_t type) {
    switch (type) {
    case DS4_METAL_TENSOR_Q8_0:
        return ds4_gpu_get_mul_mm_id_pipeline("kernel_mul_mm_id_q8_0_f32", false);
    case DS4_METAL_TENSOR_Q8_K:
        return ds4_gpu_get_mul_mm_id_pipeline("kernel_mul_mm_id_q8_K_f32", false);
    case DS4_METAL_TENSOR_IQ2_XXS:
        return ds4_gpu_get_mul_mm_id_pipeline("kernel_mul_mm_id_iq2_xxs_f32", false);
    case DS4_METAL_TENSOR_Q2_K:
        return ds4_gpu_get_mul_mm_id_pipeline("kernel_mul_mm_id_q2_K_f32", false);
    case DS4_METAL_TENSOR_Q4_K:
        return ds4_gpu_get_mul_mm_id_pipeline("kernel_mul_mm_id_q4_K_f32", false);
    case DS4_METAL_TENSOR_Q5_K:
        return ds4_gpu_get_mul_mm_id_pipeline("kernel_mul_mm_id_q5_K_f32", false);
    case DS4_METAL_TENSOR_Q6_K:
        return ds4_gpu_get_mul_mm_id_pipeline("kernel_mul_mm_id_q6_K_f32", false);
    default:
        return nil;
    }
}

static id<MTLComputePipelineState> ds4_gpu_routed_mm_addr_pipeline(uint32_t type) {
    switch (type) {
    case DS4_METAL_TENSOR_Q2_K:
        return ds4_gpu_get_mul_mm_id_pipeline("kernel_mul_mm_id_addr_q2_K_f32", false);
    case DS4_METAL_TENSOR_Q4_K:
        return ds4_gpu_get_mul_mm_id_pipeline("kernel_mul_mm_id_addr_q4_K_f32", false);
    default:
        return nil;
    }
}

static id<MTLComputePipelineState> ds4_gpu_routed_mm_f16_rhs_pipeline(uint32_t type) {
    switch (type) {
    case DS4_METAL_TENSOR_Q8_0:
        return ds4_gpu_get_mul_mm_id_pipeline("kernel_mul_mm_id_q8_0_f16", false);
    case DS4_METAL_TENSOR_Q8_K:
        return ds4_gpu_get_mul_mm_id_pipeline("kernel_mul_mm_id_q8_K_f16", false);
    case DS4_METAL_TENSOR_IQ2_XXS:
        return ds4_gpu_get_mul_mm_id_pipeline("kernel_mul_mm_id_iq2_xxs_f16", false);
    case DS4_METAL_TENSOR_Q2_K:
        return ds4_gpu_get_mul_mm_id_pipeline("kernel_mul_mm_id_q2_K_f16", false);
    case DS4_METAL_TENSOR_Q4_K:
        return ds4_gpu_get_mul_mm_id_pipeline("kernel_mul_mm_id_q4_K_f16", false);
    case DS4_METAL_TENSOR_Q5_K:
        return ds4_gpu_get_mul_mm_id_pipeline("kernel_mul_mm_id_q5_K_f16", false);
    case DS4_METAL_TENSOR_Q6_K:
        return ds4_gpu_get_mul_mm_id_pipeline("kernel_mul_mm_id_q6_K_f16", false);
    default:
        return nil;
    }
}

static id<MTLComputePipelineState> ds4_gpu_routed_mm_addr_f16_rhs_pipeline(uint32_t type) {
    switch (type) {
    case DS4_METAL_TENSOR_Q2_K:
        return ds4_gpu_get_mul_mm_id_pipeline("kernel_mul_mm_id_addr_q2_K_f16", false);
    case DS4_METAL_TENSOR_Q4_K:
        return ds4_gpu_get_mul_mm_id_pipeline("kernel_mul_mm_id_addr_q4_K_f16", false);
    default:
        return nil;
    }
}

static int ds4_gpu_encode_mul_mv_id(
        id<MTLCommandBuffer>        cb,
        id<MTLComputePipelineState> pipeline,
        const ds4_gpu_mul_mv_id_args *args,
        id<MTLBuffer>               src0,
        NSUInteger                  src0_off,
        id<MTLBuffer>               src1,
        NSUInteger                  src1_off,
        id<MTLBuffer>               dst,
        NSUInteger                  dst_off,
        id<MTLBuffer>               ids,
        NSUInteger                  ids_off,
        NSUInteger                  threadgroup_bytes,
        NSUInteger                  nsg,
        bool                        rows_per_group_is_nr0) {
    if (!cb || !pipeline || !args || !src0 || !src1 || !dst || !ids ||
        args->ne00 <= 0 || args->ne01 <= 0 || args->nei0 <= 0 || args->nei1 <= 0) {
        return 0;
    }

    const NSUInteger nr0 = (NSUInteger)args->nr0;
    const NSUInteger rows_per_group = rows_per_group_is_nr0 ? nr0 : nr0 * nsg;
    const NSUInteger row_groups = ((NSUInteger)args->ne01 + rows_per_group - 1u) / rows_per_group;
    const NSUInteger pairs = (NSUInteger)args->nei0 * (NSUInteger)args->nei1;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:pipeline];
    [enc setBytes:args length:sizeof(*args) atIndex:0];
    [enc setBuffer:src0 offset:src0_off atIndex:1];
    [enc setBuffer:src1 offset:src1_off atIndex:2];
    [enc setBuffer:dst  offset:dst_off  atIndex:3];
    [enc setBuffer:ids  offset:ids_off  atIndex:4];
    if (threadgroup_bytes != 0) {
        [enc setThreadgroupMemoryLength:threadgroup_bytes atIndex:0];
    }
    [enc dispatchThreadgroups:MTLSizeMake(row_groups, 1, pairs)
         threadsPerThreadgroup:MTLSizeMake(32, nsg, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

static int ds4_gpu_encode_attn_out_low_q8_direct(
        id<MTLCommandBuffer>        cb,
        id<MTLComputePipelineState> pipeline,
        const ds4_gpu_mul_mv_id_args *args,
        id<MTLBuffer>               src0,
        NSUInteger                  src0_off,
        id<MTLBuffer>               src1,
        NSUInteger                  src1_off,
        id<MTLBuffer>               dst,
        NSUInteger                  dst_off,
        NSUInteger                  threadgroup_bytes,
        NSUInteger                  nsg,
        bool                        rows_per_group_is_nr0) {
    if (!cb || !pipeline || !args || !src0 || !src1 || !dst ||
        args->ne00 <= 0 || args->ne01 <= 0 || args->nei0 <= 0 || args->nei1 <= 0) {
        return 0;
    }

    /* Two row conventions in the classic matvec family: Q8_0 k-splits one
     * nr0-row group across all nsg simdgroups (cross-simdgroup reduce), so a
     * threadgroup covers nr0 rows; Q4_K gives each simdgroup its own nr0
     * rows, covering nr0*nsg.  Dispatching Q8 with the Q4 stride leaves
     * (nsg-1)/nsg of the output rows unwritten. */
    const NSUInteger rows_per_group = rows_per_group_is_nr0 ?
        (NSUInteger)args->nr0 : (NSUInteger)args->nr0 * nsg;
    const NSUInteger row_groups = ((NSUInteger)args->ne01 + rows_per_group - 1u) / rows_per_group;
    const NSUInteger pairs = (NSUInteger)args->nei0 * (NSUInteger)args->nei1;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:pipeline];
    [enc setBytes:args length:sizeof(*args) atIndex:0];
    [enc setBuffer:src0 offset:src0_off atIndex:1];
    [enc setBuffer:src1 offset:src1_off atIndex:2];
    [enc setBuffer:dst  offset:dst_off  atIndex:3];
    if (threadgroup_bytes != 0) {
        [enc setThreadgroupMemoryLength:threadgroup_bytes atIndex:0];
    }
    [enc dispatchThreadgroups:MTLSizeMake(row_groups, 1, pairs)
         threadsPerThreadgroup:MTLSizeMake(32, nsg, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

static int ds4_gpu_encode_mul_mv_id_pair(
        id<MTLCommandBuffer>        cb,
        id<MTLComputePipelineState> pipeline,
        const ds4_gpu_mul_mv_id_args *args,
        id<MTLBuffer>               src0_a,
        NSUInteger                  src0_a_off,
        id<MTLBuffer>               src0_b,
        NSUInteger                  src0_b_off,
        id<MTLBuffer>               src1,
        NSUInteger                  src1_off,
        id<MTLBuffer>               dst_a,
        NSUInteger                  dst_a_off,
        id<MTLBuffer>               dst_b,
        NSUInteger                  dst_b_off,
        id<MTLBuffer>               ids,
        NSUInteger                  ids_off,
        NSUInteger                  threadgroup_bytes,
        NSUInteger                  nsg,
        bool                        rows_per_group_is_nr0) {
    if (!cb || !pipeline || !args || !src0_a || !src0_b || !src1 || !dst_a || !dst_b || !ids ||
        args->ne00 <= 0 || args->ne01 <= 0 || args->nei0 <= 0 || args->nei1 <= 0) {
        return 0;
    }

    const NSUInteger nr0 = (NSUInteger)args->nr0;
    const NSUInteger rows_per_group = rows_per_group_is_nr0 ? nr0 : nr0 * nsg;
    const NSUInteger row_groups = ((NSUInteger)args->ne01 + rows_per_group - 1u) / rows_per_group;
    const NSUInteger pairs = (NSUInteger)args->nei0 * (NSUInteger)args->nei1;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:pipeline];
    [enc setBytes:args length:sizeof(*args) atIndex:0];
    [enc setBuffer:src0_a offset:src0_a_off atIndex:1];
    [enc setBuffer:src0_b offset:src0_b_off atIndex:2];
    [enc setBuffer:src1   offset:src1_off   atIndex:3];
    [enc setBuffer:dst_a  offset:dst_a_off  atIndex:4];
    [enc setBuffer:dst_b  offset:dst_b_off  atIndex:5];
    [enc setBuffer:ids    offset:ids_off    atIndex:6];
    if (threadgroup_bytes != 0) {
        [enc setThreadgroupMemoryLength:threadgroup_bytes atIndex:0];
    }
    [enc dispatchThreadgroups:MTLSizeMake(row_groups, 1, pairs)
         threadsPerThreadgroup:MTLSizeMake(32, nsg, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

static int ds4_gpu_encode_mul_mv_id_pair_swiglu(
        id<MTLCommandBuffer>        cb,
        id<MTLComputePipelineState> pipeline,
        const ds4_gpu_mul_mv_id_args *args,
        const ds4_gpu_dsv4_moe_swiglu_weight_args *act,
        id<MTLBuffer>               src0_a,
        NSUInteger                  src0_a_off,
        id<MTLBuffer>               src0_b,
        NSUInteger                  src0_b_off,
        id<MTLBuffer>               src1,
        NSUInteger                  src1_off,
        id<MTLBuffer>               dst_a,
        NSUInteger                  dst_a_off,
        id<MTLBuffer>               dst_b,
        NSUInteger                  dst_b_off,
        id<MTLBuffer>               dst_mid,
        NSUInteger                  dst_mid_off,
        id<MTLBuffer>               ids,
        NSUInteger                  ids_off,
        id<MTLBuffer>               weights,
        NSUInteger                  weights_off,
        NSUInteger                  threadgroup_bytes,
        NSUInteger                  nsg,
        bool                        rows_per_group_is_nr0) {
    if (!cb || !pipeline || !args || !act ||
        !src0_a || !src0_b || !src1 || !dst_a || !dst_b || !dst_mid || !ids || !weights ||
        args->ne00 <= 0 || args->ne01 <= 0 || args->nei0 <= 0 || args->nei1 <= 0) {
        return 0;
    }

    const NSUInteger nr0 = (NSUInteger)args->nr0;
    const NSUInteger rows_per_group = rows_per_group_is_nr0 ? nr0 : nr0 * nsg;
    const NSUInteger row_groups = ((NSUInteger)args->ne01 + rows_per_group - 1u) / rows_per_group;
    const NSUInteger pairs = (NSUInteger)args->nei0 * (NSUInteger)args->nei1;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:pipeline];
    [enc setBytes:args length:sizeof(*args) atIndex:0];
    [enc setBytes:act  length:sizeof(*act)  atIndex:1];
    [enc setBuffer:src0_a  offset:src0_a_off  atIndex:2];
    [enc setBuffer:src0_b  offset:src0_b_off  atIndex:3];
    [enc setBuffer:src1    offset:src1_off    atIndex:4];
    [enc setBuffer:dst_a   offset:dst_a_off   atIndex:5];
    [enc setBuffer:dst_b   offset:dst_b_off   atIndex:6];
    [enc setBuffer:dst_mid offset:dst_mid_off atIndex:7];
    [enc setBuffer:ids     offset:ids_off     atIndex:8];
    [enc setBuffer:weights offset:weights_off atIndex:9];
    if (threadgroup_bytes != 0) {
        [enc setThreadgroupMemoryLength:threadgroup_bytes atIndex:0];
    }
    [enc dispatchThreadgroups:MTLSizeMake(row_groups, 1, pairs)
         threadsPerThreadgroup:MTLSizeMake(32, nsg, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

static int ds4_gpu_encode_mul_mv_table_q4_pair_swiglu(
        id<MTLCommandBuffer>        cb,
        id<MTLComputePipelineState> pipeline,
        const ds4_gpu_mul_mv_id_args *args,
        const ds4_gpu_dsv4_moe_swiglu_weight_args *act,
        DS4MetalQ4ExpertTable      *gate_table,
        DS4MetalQ4ExpertTable      *up_table,
        id<MTLBuffer>               src1,
        NSUInteger                  src1_off,
        id<MTLBuffer>               dst_a,
        NSUInteger                  dst_a_off,
        id<MTLBuffer>               dst_b,
        NSUInteger                  dst_b_off,
        id<MTLBuffer>               dst_mid,
        NSUInteger                  dst_mid_off,
        id<MTLBuffer>               ids,
        NSUInteger                  ids_off,
        id<MTLBuffer>               weights,
        NSUInteger                  weights_off,
        NSUInteger                  threadgroup_bytes,
        NSUInteger                  nsg,
        bool                        rows_per_group_is_nr0,
        bool                        queue_residency) {
    if (!cb || !pipeline || !args || !act || !gate_table || !up_table ||
        !gate_table.argumentBuffer || !up_table.argumentBuffer ||
        !src1 || !dst_a || !dst_b || !dst_mid || !ids || !weights ||
        args->ne00 <= 0 || args->ne01 <= 0 || args->nei0 != 6 || args->nei1 <= 0 ||
        args->ne02 <= 0 || args->ne02 > 384) {
        return 0;
    }

    const NSUInteger nr0 = (NSUInteger)args->nr0;
    const NSUInteger rows_per_group = rows_per_group_is_nr0 ? nr0 : nr0 * nsg;
    const NSUInteger row_groups = ((NSUInteger)args->ne01 + rows_per_group - 1u) / rows_per_group;
    const NSUInteger pairs = (NSUInteger)args->nei0 * (NSUInteger)args->nei1;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:pipeline];
    [enc setBytes:args length:sizeof(*args) atIndex:0];
    [enc setBytes:act  length:sizeof(*act)  atIndex:1];
    [enc setBuffer:gate_table.argumentBuffer offset:0 atIndex:2];
    [enc setBuffer:up_table.argumentBuffer   offset:0 atIndex:3];
    [enc setBuffer:src1    offset:src1_off    atIndex:4];
    [enc setBuffer:dst_a   offset:dst_a_off   atIndex:5];
    [enc setBuffer:dst_b   offset:dst_b_off   atIndex:6];
    [enc setBuffer:dst_mid offset:dst_mid_off atIndex:7];
    [enc setBuffer:ids     offset:ids_off     atIndex:8];
    [enc setBuffer:weights offset:weights_off atIndex:9];
    if (!ds4_gpu_bind_q4_expert_table_anchors(enc, gate_table, 10, 6) ||
        !ds4_gpu_bind_q4_expert_table_anchors(enc, up_table, 16, 6)) {
        ds4_gpu_end_compute_encoder(cb, enc);
        return 0;
    }
    ds4_gpu_use_q4_expert_table_resources(cb, enc, gate_table, queue_residency);
    ds4_gpu_use_q4_expert_table_resources(cb, enc, up_table, queue_residency);
    if (threadgroup_bytes != 0) {
        [enc setThreadgroupMemoryLength:threadgroup_bytes atIndex:0];
    }
    [enc dispatchThreadgroups:MTLSizeMake(row_groups, 1, pairs)
         threadsPerThreadgroup:MTLSizeMake(32, nsg, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

static int ds4_gpu_encode_mul_mv_addr_q4_pair_swiglu(
        id<MTLCommandBuffer>        cb,
        id<MTLComputePipelineState> pipeline,
        const ds4_gpu_mul_mv_id_args *args,
        const ds4_gpu_dsv4_moe_swiglu_weight_args *act,
        DS4MetalQ4ExpertTable      *gate_table,
        DS4MetalQ4ExpertTable      *up_table,
        id<MTLBuffer>               src1,
        NSUInteger                  src1_off,
        id<MTLBuffer>               dst_a,
        NSUInteger                  dst_a_off,
        id<MTLBuffer>               dst_b,
        NSUInteger                  dst_b_off,
        id<MTLBuffer>               dst_mid,
        NSUInteger                  dst_mid_off,
        id<MTLBuffer>               ids,
        NSUInteger                  ids_off,
        id<MTLBuffer>               weights,
        NSUInteger                  weights_off,
        NSUInteger                  threadgroup_bytes,
        NSUInteger                  nsg,
        bool                        rows_per_group_is_nr0) {
    if (!cb || !pipeline || !args || !act || !gate_table || !up_table ||
        !gate_table.addressBuffer || !up_table.addressBuffer ||
        !src1 || !dst_a || !dst_b || !dst_mid || !ids || !weights ||
        args->ne00 <= 0 || args->ne01 <= 0 || args->nei0 != 6 || args->nei1 <= 0 ||
        args->ne02 <= 0 || args->ne02 > 384) {
        return 0;
    }

    const NSUInteger nr0 = (NSUInteger)args->nr0;
    const NSUInteger rows_per_group = rows_per_group_is_nr0 ? nr0 : nr0 * nsg;
    const NSUInteger row_groups = ((NSUInteger)args->ne01 + rows_per_group - 1u) / rows_per_group;
    const NSUInteger pairs = (NSUInteger)args->nei0 * (NSUInteger)args->nei1;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:pipeline];
    [enc setBytes:args length:sizeof(*args) atIndex:0];
    [enc setBytes:act  length:sizeof(*act)  atIndex:1];
    [enc setBuffer:gate_table.addressBuffer offset:0 atIndex:2];
    [enc setBuffer:up_table.addressBuffer   offset:0 atIndex:3];
    [enc setBuffer:src1    offset:src1_off    atIndex:4];
    [enc setBuffer:dst_a   offset:dst_a_off   atIndex:5];
    [enc setBuffer:dst_b   offset:dst_b_off   atIndex:6];
    [enc setBuffer:dst_mid offset:dst_mid_off atIndex:7];
    [enc setBuffer:ids     offset:ids_off     atIndex:8];
    [enc setBuffer:weights offset:weights_off atIndex:9];
    if (!ds4_gpu_bind_q4_expert_table_anchors(enc, gate_table, 10, 6) ||
        !ds4_gpu_bind_q4_expert_table_anchors(enc, up_table, 16, 6)) {
        ds4_gpu_end_compute_encoder(cb, enc);
        return 0;
    }
    if (getenv("DS4_METAL_Q4_ADDR_USE_RESOURCES") != NULL) {
        ds4_gpu_use_q4_expert_table_resources(cb, enc, gate_table, false);
        ds4_gpu_use_q4_expert_table_resources(cb, enc, up_table, false);
    }
    if (threadgroup_bytes != 0) {
        [enc setThreadgroupMemoryLength:threadgroup_bytes atIndex:0];
    }
    [enc dispatchThreadgroups:MTLSizeMake(row_groups, 1, pairs)
         threadsPerThreadgroup:MTLSizeMake(32, nsg, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

static int ds4_gpu_encode_mul_mv_table_q4_sum6(
        id<MTLCommandBuffer>        cb,
        id<MTLComputePipelineState> pipeline,
        const ds4_gpu_mul_mv_id_args *args,
        DS4MetalQ4ExpertTable      *table,
        id<MTLBuffer>               src1,
        NSUInteger                  src1_off,
        id<MTLBuffer>               dst,
        NSUInteger                  dst_off,
        id<MTLBuffer>               ids,
        NSUInteger                  ids_off,
        NSUInteger                  threadgroup_bytes,
        NSUInteger                  nsg,
        bool                        queue_residency) {
    if (!cb || !pipeline || !args || !table || !table.argumentBuffer ||
        !src1 || !dst || !ids ||
        args->ne00 <= 0 || args->ne01 <= 0 || args->nei0 != 6 || args->nei1 <= 0 ||
        args->ne02 <= 0 || args->ne02 > 384) {
        return 0;
    }

    const NSUInteger rows_per_group = (NSUInteger)args->nr0 * nsg;
    const NSUInteger row_groups = ((NSUInteger)args->ne01 + rows_per_group - 1u) / rows_per_group;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:pipeline];
    [enc setBytes:args length:sizeof(*args) atIndex:0];
    [enc setBuffer:table.argumentBuffer offset:0 atIndex:1];
    [enc setBuffer:src1 offset:src1_off atIndex:2];
    [enc setBuffer:dst  offset:dst_off  atIndex:3];
    [enc setBuffer:ids  offset:ids_off  atIndex:4];
    if (!ds4_gpu_bind_q4_expert_table_anchors(enc, table, 5, 6)) {
        ds4_gpu_end_compute_encoder(cb, enc);
        return 0;
    }
    ds4_gpu_use_q4_expert_table_resources(cb, enc, table, queue_residency);
    if (threadgroup_bytes != 0) {
        [enc setThreadgroupMemoryLength:threadgroup_bytes atIndex:0];
    }
    [enc dispatchThreadgroups:MTLSizeMake(row_groups, (NSUInteger)args->nei1, 1)
         threadsPerThreadgroup:MTLSizeMake(32, nsg, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

static int ds4_gpu_encode_mul_mv_addr_q4_sum6(
        id<MTLCommandBuffer>        cb,
        id<MTLComputePipelineState> pipeline,
        const ds4_gpu_mul_mv_id_args *args,
        DS4MetalQ4ExpertTable      *table,
        id<MTLBuffer>               src1,
        NSUInteger                  src1_off,
        id<MTLBuffer>               dst,
        NSUInteger                  dst_off,
        id<MTLBuffer>               ids,
        NSUInteger                  ids_off,
        NSUInteger                  threadgroup_bytes,
        NSUInteger                  nsg) {
    if (!cb || !pipeline || !args || !table || !table.addressBuffer ||
        !src1 || !dst || !ids ||
        args->ne00 <= 0 || args->ne01 <= 0 || args->nei0 != 6 || args->nei1 <= 0 ||
        args->ne02 <= 0 || args->ne02 > 384) {
        return 0;
    }

    const NSUInteger rows_per_group = (NSUInteger)args->nr0 * nsg;
    const NSUInteger row_groups = ((NSUInteger)args->ne01 + rows_per_group - 1u) / rows_per_group;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:pipeline];
    [enc setBytes:args length:sizeof(*args) atIndex:0];
    [enc setBuffer:table.addressBuffer offset:0 atIndex:1];
    [enc setBuffer:src1 offset:src1_off atIndex:2];
    [enc setBuffer:dst  offset:dst_off  atIndex:3];
    [enc setBuffer:ids  offset:ids_off  atIndex:4];
    if (!ds4_gpu_bind_q4_expert_table_anchors(enc, table, 5, 6)) {
        ds4_gpu_end_compute_encoder(cb, enc);
        return 0;
    }
    if (getenv("DS4_METAL_Q4_ADDR_USE_RESOURCES") != NULL) {
        ds4_gpu_use_q4_expert_table_resources(cb, enc, table, false);
    }
    if (threadgroup_bytes != 0) {
        [enc setThreadgroupMemoryLength:threadgroup_bytes atIndex:0];
    }
    [enc dispatchThreadgroups:MTLSizeMake(row_groups, (NSUInteger)args->nei1, 1)
         threadsPerThreadgroup:MTLSizeMake(32, nsg, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

static uint32_t ds4_gpu_q4_expert_group_size(uint32_t n_total_expert) {
    uint32_t group_size = 32;
    const char *env = getenv("DS4_METAL_Q4_EXPERT_GROUP_SIZE");
    if (env && env[0]) {
        char *end = NULL;
        unsigned long v = strtoul(env, &end, 10);
        if (end != env && *end == '\0' && v > 0 && v <= UINT32_MAX) {
            group_size = (uint32_t)v;
        }
    }
    if (group_size == 0) group_size = 1;
    if (group_size > n_total_expert) group_size = n_total_expert;
    return group_size;
}

static int ds4_gpu_encode_mul_mv_group_q4_pair_swiglu(
        id<MTLCommandBuffer>        cb,
        id<MTLComputePipelineState> pipeline,
        const ds4_gpu_mul_mv_id_args *args,
        const ds4_gpu_dsv4_moe_swiglu_weight_args *act,
        const ds4_gpu_moe_expert_group_args *group,
        id<MTLBuffer>               src0_a,
        NSUInteger                  src0_a_off,
        id<MTLBuffer>               src0_b,
        NSUInteger                  src0_b_off,
        id<MTLBuffer>               src1,
        NSUInteger                  src1_off,
        id<MTLBuffer>               dst_a,
        NSUInteger                  dst_a_off,
        id<MTLBuffer>               dst_b,
        NSUInteger                  dst_b_off,
        id<MTLBuffer>               dst_mid,
        NSUInteger                  dst_mid_off,
        id<MTLBuffer>               ids,
        NSUInteger                  ids_off,
        id<MTLBuffer>               weights,
        NSUInteger                  weights_off,
        NSUInteger                  threadgroup_bytes,
        NSUInteger                  nsg,
        bool                        rows_per_group_is_nr0) {
    if (!cb || !pipeline || !args || !act || !group ||
        !src0_a || !src0_b || !src1 || !dst_a || !dst_b || !dst_mid || !ids || !weights ||
        args->ne00 <= 0 || args->ne01 <= 0 || args->nei0 <= 0 || args->nei1 <= 0 ||
        group->expert_count == 0) {
        return 0;
    }

    const NSUInteger nr0 = (NSUInteger)args->nr0;
    const NSUInteger rows_per_group = rows_per_group_is_nr0 ? nr0 : nr0 * nsg;
    const NSUInteger row_groups = ((NSUInteger)args->ne01 + rows_per_group - 1u) / rows_per_group;
    const NSUInteger pairs = (NSUInteger)args->nei0 * (NSUInteger)args->nei1;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:pipeline];
    [enc setBytes:args  length:sizeof(*args)  atIndex:0];
    [enc setBytes:act   length:sizeof(*act)   atIndex:1];
    [enc setBytes:group length:sizeof(*group) atIndex:2];
    [enc setBuffer:src0_a  offset:src0_a_off  atIndex:3];
    [enc setBuffer:src0_b  offset:src0_b_off  atIndex:4];
    [enc setBuffer:src1    offset:src1_off    atIndex:5];
    [enc setBuffer:dst_a   offset:dst_a_off   atIndex:6];
    [enc setBuffer:dst_b   offset:dst_b_off   atIndex:7];
    [enc setBuffer:dst_mid offset:dst_mid_off atIndex:8];
    [enc setBuffer:ids     offset:ids_off     atIndex:9];
    [enc setBuffer:weights offset:weights_off atIndex:10];
    if (threadgroup_bytes != 0) {
        [enc setThreadgroupMemoryLength:threadgroup_bytes atIndex:0];
    }
    [enc dispatchThreadgroups:MTLSizeMake(row_groups, 1, pairs)
         threadsPerThreadgroup:MTLSizeMake(32, nsg, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

static int ds4_gpu_encode_mul_mv_group_q4_sum6(
        id<MTLCommandBuffer>        cb,
        id<MTLComputePipelineState> pipeline,
        const ds4_gpu_mul_mv_id_args *args,
        const ds4_gpu_moe_expert_group_args *group,
        id<MTLBuffer>               src0,
        NSUInteger                  src0_off,
        id<MTLBuffer>               src1,
        NSUInteger                  src1_off,
        id<MTLBuffer>               dst,
        NSUInteger                  dst_off,
        id<MTLBuffer>               ids,
        NSUInteger                  ids_off,
        NSUInteger                  threadgroup_bytes,
        NSUInteger                  nsg) {
    if (!cb || !pipeline || !args || !group || !src0 || !src1 || !dst || !ids ||
        args->ne00 <= 0 || args->ne01 <= 0 || args->nei0 != 6 || args->nei1 <= 0 ||
        group->expert_count == 0) {
        return 0;
    }

    const NSUInteger rows_per_group = (NSUInteger)args->nr0 * nsg;
    const NSUInteger row_groups = ((NSUInteger)args->ne01 + rows_per_group - 1u) / rows_per_group;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:pipeline];
    [enc setBytes:args  length:sizeof(*args)  atIndex:0];
    [enc setBytes:group length:sizeof(*group) atIndex:1];
    [enc setBuffer:src0 offset:src0_off atIndex:2];
    [enc setBuffer:src1 offset:src1_off atIndex:3];
    [enc setBuffer:dst  offset:dst_off  atIndex:4];
    [enc setBuffer:ids  offset:ids_off  atIndex:5];
    if (threadgroup_bytes != 0) {
        [enc setThreadgroupMemoryLength:threadgroup_bytes atIndex:0];
    }
    [enc dispatchThreadgroups:MTLSizeMake(row_groups, (NSUInteger)args->nei1, 1)
         threadsPerThreadgroup:MTLSizeMake(32, nsg, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

static int ds4_gpu_encode_mul_mv_id_sum6(
        id<MTLCommandBuffer>        cb,
        id<MTLComputePipelineState> pipeline,
        const ds4_gpu_mul_mv_id_args *args,
        id<MTLBuffer>               src0,
        NSUInteger                  src0_off,
        id<MTLBuffer>               src1,
        NSUInteger                  src1_off,
        id<MTLBuffer>               dst,
        NSUInteger                  dst_off,
        id<MTLBuffer>               ids,
        NSUInteger                  ids_off,
        id<MTLBuffer>               add_in,
        NSUInteger                  add_in_off,
        NSUInteger                  threadgroup_bytes,
        NSUInteger                  nsg) {
    if (!cb || !pipeline || !args || !src0 || !src1 || !dst || !ids ||
        args->ne00 <= 0 || args->ne01 <= 0 ||
        args->nei0 <= 0 || args->nei0 > 8 || args->nei1 <= 0) {
        return 0;
    }

    const NSUInteger rows_per_group = (NSUInteger)args->nr0 * nsg;
    const NSUInteger row_groups = ((NSUInteger)args->ne01 + rows_per_group - 1u) / rows_per_group;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:pipeline];
    [enc setBytes:args length:sizeof(*args) atIndex:0];
    [enc setBuffer:src0 offset:src0_off atIndex:1];
    [enc setBuffer:src1 offset:src1_off atIndex:2];
    [enc setBuffer:dst  offset:dst_off  atIndex:3];
    [enc setBuffer:ids  offset:ids_off  atIndex:4];
    [enc setBuffer:(add_in ? add_in : dst) offset:(add_in ? add_in_off : dst_off) atIndex:5];
    if (threadgroup_bytes != 0) {
        [enc setThreadgroupMemoryLength:threadgroup_bytes atIndex:0];
    }
    [enc dispatchThreadgroups:MTLSizeMake(row_groups, (NSUInteger)args->nei1, 1)
         threadsPerThreadgroup:MTLSizeMake(32, nsg, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

static int ds4_gpu_encode_q4_gather_slots6(
        id<MTLCommandBuffer>        cb,
        id<MTLComputePipelineState> pipeline,
        const ds4_gpu_q4_gather_slots6_args *args,
        __unsafe_unretained id<MTLBuffer> src_groups[6],
        const NSUInteger            src_group_offsets[6],
        id<MTLBuffer>               ids,
        NSUInteger                  ids_off,
        id<MTLBuffer>               dst,
        NSUInteger                  dst_off) {
    if (!cb || !pipeline || !args || !src_groups || !src_group_offsets || !ids || !dst ||
        args->expert_bytes == 0 || (args->expert_bytes & 15u) != 0 ||
        args->group_size == 0 || args->n_slots == 0 || args->n_slots > 6) {
        return 0;
    }
    for (uint32_t i = 0; i < 6; i++) {
        if (!src_groups[i]) return 0;
    }

    const uint64_t chunks = args->expert_bytes >> 4;
    if (chunks == 0 || chunks > NSUIntegerMax) return 0;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:pipeline];
    [enc setBytes:args length:sizeof(*args) atIndex:0];
    for (uint32_t i = 0; i < 6; i++) {
        [enc setBuffer:src_groups[i] offset:src_group_offsets[i] atIndex:1 + i];
    }
    [enc setBuffer:ids offset:ids_off atIndex:7];
    [enc setBuffer:dst offset:dst_off atIndex:8];
    const NSUInteger threads = 256u;
    [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)chunks + threads - 1u) / threads,
                                          (NSUInteger)args->n_slots,
                                          1)
         threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

static int ds4_gpu_encode_mul_mv_slots6_pair_swiglu(
        id<MTLCommandBuffer>        cb,
        id<MTLComputePipelineState> pipeline,
        const ds4_gpu_mul_mv_id_args *args,
        const ds4_gpu_dsv4_moe_swiglu_weight_args *act,
        __unsafe_unretained id<MTLBuffer> src0_a[6],
        const NSUInteger            src0_a_off[6],
        __unsafe_unretained id<MTLBuffer> src0_b[6],
        const NSUInteger            src0_b_off[6],
        id<MTLBuffer>               src1,
        NSUInteger                  src1_off,
        id<MTLBuffer>               dst_a,
        NSUInteger                  dst_a_off,
        id<MTLBuffer>               dst_b,
        NSUInteger                  dst_b_off,
        id<MTLBuffer>               dst_mid,
        NSUInteger                  dst_mid_off,
        id<MTLBuffer>               weights,
        NSUInteger                  weights_off,
        NSUInteger                  threadgroup_bytes,
        NSUInteger                  nsg,
        bool                        rows_per_group_is_nr0) {
    if (!cb || !pipeline || !args || !act || !src0_a || !src0_a_off || !src0_b || !src0_b_off ||
        !src1 || !dst_a || !dst_b || !dst_mid || !weights ||
        args->ne00 <= 0 || args->ne01 <= 0 || args->nei0 != 6 || args->nei1 <= 0) {
        return 0;
    }
    for (uint32_t i = 0; i < 6; i++) {
        if (!src0_a[i] || !src0_b[i]) return 0;
    }

    const NSUInteger nr0 = (NSUInteger)args->nr0;
    const NSUInteger rows_per_group = rows_per_group_is_nr0 ? nr0 : nr0 * nsg;
    const NSUInteger row_groups = ((NSUInteger)args->ne01 + rows_per_group - 1u) / rows_per_group;
    const NSUInteger pairs = (NSUInteger)args->nei0 * (NSUInteger)args->nei1;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:pipeline];
    [enc setBytes:args length:sizeof(*args) atIndex:0];
    [enc setBytes:act  length:sizeof(*act)  atIndex:1];
    for (uint32_t i = 0; i < 6; i++) {
        [enc setBuffer:src0_a[i] offset:src0_a_off[i] atIndex:2 + i];
    }
    for (uint32_t i = 0; i < 6; i++) {
        [enc setBuffer:src0_b[i] offset:src0_b_off[i] atIndex:8 + i];
    }
    [enc setBuffer:src1    offset:src1_off    atIndex:14];
    [enc setBuffer:dst_a   offset:dst_a_off   atIndex:15];
    [enc setBuffer:dst_b   offset:dst_b_off   atIndex:16];
    [enc setBuffer:dst_mid offset:dst_mid_off atIndex:17];
    [enc setBuffer:weights offset:weights_off atIndex:18];
    if (threadgroup_bytes != 0) {
        [enc setThreadgroupMemoryLength:threadgroup_bytes atIndex:0];
    }
    [enc dispatchThreadgroups:MTLSizeMake(row_groups, 1, pairs)
         threadsPerThreadgroup:MTLSizeMake(32, nsg, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

static int ds4_gpu_encode_mul_mv_slots6_sum6(
        id<MTLCommandBuffer>        cb,
        id<MTLComputePipelineState> pipeline,
        const ds4_gpu_mul_mv_id_args *args,
        __unsafe_unretained id<MTLBuffer> src0[6],
        const NSUInteger            src0_off[6],
        id<MTLBuffer>               src1,
        NSUInteger                  src1_off,
        id<MTLBuffer>               dst,
        NSUInteger                  dst_off,
        NSUInteger                  threadgroup_bytes,
        NSUInteger                  nsg) {
    if (!cb || !pipeline || !args || !src0 || !src0_off || !src1 || !dst ||
        args->ne00 <= 0 || args->ne01 <= 0 || args->nei0 != 6 || args->nei1 <= 0) {
        return 0;
    }
    for (uint32_t i = 0; i < 6; i++) {
        if (!src0[i]) return 0;
    }

    const NSUInteger rows_per_group = (NSUInteger)args->nr0 * nsg;
    const NSUInteger row_groups = ((NSUInteger)args->ne01 + rows_per_group - 1u) / rows_per_group;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:pipeline];
    [enc setBytes:args length:sizeof(*args) atIndex:0];
    for (uint32_t i = 0; i < 6; i++) {
        [enc setBuffer:src0[i] offset:src0_off[i] atIndex:1 + i];
    }
    [enc setBuffer:src1 offset:src1_off atIndex:7];
    [enc setBuffer:dst  offset:dst_off  atIndex:8];
    if (threadgroup_bytes != 0) {
        [enc setThreadgroupMemoryLength:threadgroup_bytes atIndex:0];
    }
    [enc dispatchThreadgroups:MTLSizeMake(row_groups, (NSUInteger)args->nei1, 1)
         threadsPerThreadgroup:MTLSizeMake(32, nsg, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

static int ds4_gpu_encode_mul_mv_group6_pair_swiglu(
        id<MTLCommandBuffer>        cb,
        id<MTLComputePipelineState> pipeline,
        const ds4_gpu_mul_mv_id_args *args,
        const ds4_gpu_dsv4_moe_swiglu_weight_args *act,
        __unsafe_unretained id<MTLBuffer> src0_a[6],
        const NSUInteger            src0_a_off[6],
        __unsafe_unretained id<MTLBuffer> src0_b[6],
        const NSUInteger            src0_b_off[6],
        id<MTLBuffer>               src1,
        NSUInteger                  src1_off,
        id<MTLBuffer>               dst_a,
        NSUInteger                  dst_a_off,
        id<MTLBuffer>               dst_b,
        NSUInteger                  dst_b_off,
        id<MTLBuffer>               dst_mid,
        NSUInteger                  dst_mid_off,
        id<MTLBuffer>               ids,
        NSUInteger                  ids_off,
        id<MTLBuffer>               weights,
        NSUInteger                  weights_off,
        NSUInteger                  threadgroup_bytes,
        NSUInteger                  nsg,
        bool                        rows_per_group_is_nr0) {
    if (!cb || !pipeline || !args || !act || !src0_a || !src0_a_off || !src0_b || !src0_b_off ||
        !src1 || !dst_a || !dst_b || !dst_mid || !ids || !weights ||
        args->ne00 <= 0 || args->ne01 <= 0 || args->nei0 != 6 || args->nei1 <= 0 ||
        args->ne02 != 384) {
        return 0;
    }
    for (uint32_t i = 0; i < 6; i++) {
        if (!src0_a[i] || !src0_b[i]) return 0;
    }

    const NSUInteger nr0 = (NSUInteger)args->nr0;
    const NSUInteger rows_per_group = rows_per_group_is_nr0 ? nr0 : nr0 * nsg;
    const NSUInteger row_groups = ((NSUInteger)args->ne01 + rows_per_group - 1u) / rows_per_group;
    const NSUInteger pairs = (NSUInteger)args->nei0 * (NSUInteger)args->nei1;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:pipeline];
    [enc setBytes:args length:sizeof(*args) atIndex:0];
    [enc setBytes:act  length:sizeof(*act)  atIndex:1];
    for (uint32_t i = 0; i < 6; i++) {
        [enc setBuffer:src0_a[i] offset:src0_a_off[i] atIndex:2 + i];
    }
    for (uint32_t i = 0; i < 6; i++) {
        [enc setBuffer:src0_b[i] offset:src0_b_off[i] atIndex:8 + i];
    }
    [enc setBuffer:src1    offset:src1_off    atIndex:14];
    [enc setBuffer:dst_a   offset:dst_a_off   atIndex:15];
    [enc setBuffer:dst_b   offset:dst_b_off   atIndex:16];
    [enc setBuffer:dst_mid offset:dst_mid_off atIndex:17];
    [enc setBuffer:ids     offset:ids_off     atIndex:18];
    [enc setBuffer:weights offset:weights_off atIndex:19];
    if (threadgroup_bytes != 0) {
        [enc setThreadgroupMemoryLength:threadgroup_bytes atIndex:0];
    }
    [enc dispatchThreadgroups:MTLSizeMake(row_groups, 1, pairs)
         threadsPerThreadgroup:MTLSizeMake(32, nsg, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

static int ds4_gpu_encode_mul_mv_group6_sum6(
        id<MTLCommandBuffer>        cb,
        id<MTLComputePipelineState> pipeline,
        const ds4_gpu_mul_mv_id_args *args,
        __unsafe_unretained id<MTLBuffer> src0[6],
        const NSUInteger            src0_off[6],
        id<MTLBuffer>               src1,
        NSUInteger                  src1_off,
        id<MTLBuffer>               dst,
        NSUInteger                  dst_off,
        id<MTLBuffer>               ids,
        NSUInteger                  ids_off,
        NSUInteger                  threadgroup_bytes,
        NSUInteger                  nsg) {
    if (!cb || !pipeline || !args || !src0 || !src0_off || !src1 || !dst || !ids ||
        args->ne00 <= 0 || args->ne01 <= 0 || args->nei0 != 6 || args->nei1 <= 0 ||
        args->ne02 != 384) {
        return 0;
    }
    for (uint32_t i = 0; i < 6; i++) {
        if (!src0[i]) return 0;
    }

    const NSUInteger rows_per_group = (NSUInteger)args->nr0 * nsg;
    const NSUInteger row_groups = ((NSUInteger)args->ne01 + rows_per_group - 1u) / rows_per_group;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:pipeline];
    [enc setBytes:args length:sizeof(*args) atIndex:0];
    for (uint32_t i = 0; i < 6; i++) {
        [enc setBuffer:src0[i] offset:src0_off[i] atIndex:1 + i];
    }
    [enc setBuffer:src1 offset:src1_off atIndex:7];
    [enc setBuffer:dst  offset:dst_off  atIndex:8];
    [enc setBuffer:ids  offset:ids_off  atIndex:9];
    if (threadgroup_bytes != 0) {
        [enc setThreadgroupMemoryLength:threadgroup_bytes atIndex:0];
    }
    [enc dispatchThreadgroups:MTLSizeMake(row_groups, (NSUInteger)args->nei1, 1)
         threadsPerThreadgroup:MTLSizeMake(32, nsg, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

static int ds4_gpu_encode_mul_mv_addr_iq2_pair_swiglu(
        id<MTLCommandBuffer>        cb,
        id<MTLComputePipelineState> pipeline,
        const ds4_gpu_mul_mv_id_args *args,
        const ds4_gpu_dsv4_moe_swiglu_weight_args *act,
        ds4_gpu_stream_expert_cache_entry * const *entries,
        uint32_t                    n_entries,
        id<MTLBuffer>               gate_addrs,
        id<MTLBuffer>               up_addrs,
        id<MTLBuffer>               src1,
        NSUInteger                  src1_off,
        id<MTLBuffer>               dst_a,
        NSUInteger                  dst_a_off,
        id<MTLBuffer>               dst_b,
        NSUInteger                  dst_b_off,
        id<MTLBuffer>               dst_mid,
        NSUInteger                  dst_mid_off,
        id<MTLBuffer>               ids,
        NSUInteger                  ids_off,
        id<MTLBuffer>               weights,
        NSUInteger                  weights_off,
        NSUInteger                  threadgroup_bytes,
        NSUInteger                  nsg,
        bool                        rows_per_group_is_nr0,
        id<MTLBuffer>               overflow_gate,
        id<MTLBuffer>               overflow_up) {
    if (!cb || !pipeline || !args || !act || !entries ||
        (n_entries == 0 && !overflow_gate) ||
        !gate_addrs || !up_addrs ||
        !src1 || !dst_a || !dst_b || !dst_mid || !ids || !weights ||
        args->ne00 <= 0 || args->ne01 <= 0 ||
        args->nei0 <= 0 || args->nei0 > DS4_METAL_MAX_ROUTED_EXPERT_USED ||
        args->nei1 <= 0 ||
        args->ne02 <= 0 || args->ne02 > 384) {
        return 0;
    }
    for (uint32_t i = 0; i < n_entries; i++) {
        if (!entries[i] || !entries[i]->gate_buffer || !entries[i]->up_buffer) return 0;
    }
    if (!ds4_gpu_stream_expert_cache_mark_entries_inflight(entries,
                                                           n_entries,
                                                           0)) {
        return 0;
    }

    const NSUInteger nr0 = (NSUInteger)args->nr0;
    const NSUInteger rows_per_group = rows_per_group_is_nr0 ? nr0 : nr0 * nsg;
    const NSUInteger row_groups = ((NSUInteger)args->ne01 + rows_per_group - 1u) / rows_per_group;
    const NSUInteger pairs = (NSUInteger)args->nei0 * (NSUInteger)args->nei1;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:pipeline];
    [enc setBytes:args length:sizeof(*args) atIndex:0];
    [enc setBytes:act  length:sizeof(*act)  atIndex:1];
    [enc setBuffer:gate_addrs offset:0 atIndex:2];
    [enc setBuffer:up_addrs   offset:0 atIndex:3];
    [enc setBuffer:src1       offset:src1_off    atIndex:4];
    [enc setBuffer:dst_a      offset:dst_a_off   atIndex:5];
    [enc setBuffer:dst_b      offset:dst_b_off   atIndex:6];
    [enc setBuffer:dst_mid    offset:dst_mid_off atIndex:7];
    [enc setBuffer:ids        offset:ids_off     atIndex:8];
    [enc setBuffer:weights    offset:weights_off atIndex:9];
    for (uint32_t i = 0; i < n_entries; i++) {
        [enc useResource:entries[i]->gate_buffer usage:MTLResourceUsageRead];
        [enc useResource:entries[i]->up_buffer usage:MTLResourceUsageRead];
    }
    /* Overflow experts are addressed straight into the mapped model views
     * when a layer's unique selected set exceeds the cache budget. */
    if (overflow_gate) [enc useResource:overflow_gate usage:MTLResourceUsageRead];
    if (overflow_up) [enc useResource:overflow_up usage:MTLResourceUsageRead];
    if (threadgroup_bytes != 0) {
        [enc setThreadgroupMemoryLength:threadgroup_bytes atIndex:0];
    }
    [enc dispatchThreadgroups:MTLSizeMake(row_groups, 1, pairs)
         threadsPerThreadgroup:MTLSizeMake(32, nsg, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

static int ds4_gpu_encode_mul_mv_addr_iq2(
        id<MTLCommandBuffer>        cb,
        id<MTLComputePipelineState> pipeline,
        const ds4_gpu_mul_mv_id_args *args,
        ds4_gpu_stream_expert_cache_entry * const *entries,
        uint32_t                    n_entries,
        id<MTLBuffer>               addrs,
        id<MTLBuffer>               src1,
        NSUInteger                  src1_off,
        id<MTLBuffer>               dst,
        NSUInteger                  dst_off,
        id<MTLBuffer>               ids,
        NSUInteger                  ids_off,
        NSUInteger                  threadgroup_bytes,
        NSUInteger                  nsg,
        bool                        rows_per_group_is_nr0) {
    if (!cb || !pipeline || !args || !entries || n_entries == 0 ||
        n_entries > DS4_METAL_MAX_ROUTED_EXPERT_USED ||
        !addrs || !src1 || !dst || !ids ||
        args->ne00 <= 0 || args->ne01 <= 0 ||
        args->nei0 <= 0 || args->nei0 > DS4_METAL_MAX_ROUTED_EXPERT_USED ||
        args->nei1 <= 0 ||
        args->ne02 <= 0 || args->ne02 > 384) {
        return 0;
    }
    for (uint32_t i = 0; i < n_entries; i++) {
        if (!entries[i] || !entries[i]->down_buffer) return 0;
    }
    if (!ds4_gpu_stream_expert_cache_mark_entries_inflight(entries,
                                                           n_entries,
                                                           0)) {
        return 0;
    }

    const NSUInteger nr0 = (NSUInteger)args->nr0;
    const NSUInteger rows_per_group = rows_per_group_is_nr0 ? nr0 : nr0 * nsg;
    const NSUInteger row_groups = ((NSUInteger)args->ne01 + rows_per_group - 1u) / rows_per_group;
    const NSUInteger pairs = (NSUInteger)args->nei0 * (NSUInteger)args->nei1;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:pipeline];
    [enc setBytes:args length:sizeof(*args) atIndex:0];
    [enc setBuffer:addrs offset:0 atIndex:1];
    [enc setBuffer:src1  offset:src1_off atIndex:2];
    [enc setBuffer:dst   offset:dst_off  atIndex:3];
    [enc setBuffer:ids   offset:ids_off  atIndex:4];
    for (uint32_t i = 0; i < n_entries; i++) {
        [enc useResource:entries[i]->down_buffer usage:MTLResourceUsageRead];
    }
    if (threadgroup_bytes != 0) {
        [enc setThreadgroupMemoryLength:threadgroup_bytes atIndex:0];
    }
    [enc dispatchThreadgroups:MTLSizeMake(row_groups, 1, pairs)
         threadsPerThreadgroup:MTLSizeMake(32, nsg, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

static int ds4_gpu_encode_mul_mv_addr_q2_sum6(
        id<MTLCommandBuffer>        cb,
        id<MTLComputePipelineState> pipeline,
        const ds4_gpu_mul_mv_id_args *args,
        ds4_gpu_stream_expert_cache_entry * const *entries,
        uint32_t                    n_entries,
        id<MTLBuffer>               addrs,
        id<MTLBuffer>               src1,
        NSUInteger                  src1_off,
        id<MTLBuffer>               dst,
        NSUInteger                  dst_off,
        id<MTLBuffer>               ids,
        NSUInteger                  ids_off,
        NSUInteger                  threadgroup_bytes,
        NSUInteger                  nsg,
        id<MTLBuffer>               overflow_down) {
    if (!cb || !pipeline || !args || !entries ||
        (n_entries == 0 && !overflow_down) ||
        !addrs || !src1 || !dst || !ids ||
        args->ne00 <= 0 || args->ne01 <= 0 || args->nei0 != 6 || args->nei1 <= 0 ||
        args->ne02 <= 0 || args->ne02 > 384) {
        return 0;
    }
    for (uint32_t i = 0; i < n_entries; i++) {
        if (!entries[i] || !entries[i]->down_buffer) return 0;
    }
    if (!ds4_gpu_stream_expert_cache_mark_entries_inflight(entries,
                                                           n_entries,
                                                           0)) {
        return 0;
    }

    const NSUInteger rows_per_group = (NSUInteger)args->nr0 * nsg;
    const NSUInteger row_groups = ((NSUInteger)args->ne01 + rows_per_group - 1u) / rows_per_group;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:pipeline];
    [enc setBytes:args length:sizeof(*args) atIndex:0];
    [enc setBuffer:addrs offset:0 atIndex:1];
    [enc setBuffer:src1  offset:src1_off atIndex:2];
    [enc setBuffer:dst   offset:dst_off  atIndex:3];
    [enc setBuffer:ids   offset:ids_off  atIndex:4];
    for (uint32_t i = 0; i < n_entries; i++) {
        [enc useResource:entries[i]->down_buffer usage:MTLResourceUsageRead];
    }
    if (overflow_down) [enc useResource:overflow_down usage:MTLResourceUsageRead];
    if (threadgroup_bytes != 0) {
        [enc setThreadgroupMemoryLength:threadgroup_bytes atIndex:0];
    }
    [enc dispatchThreadgroups:MTLSizeMake(row_groups, (NSUInteger)args->nei1, 1)
         threadsPerThreadgroup:MTLSizeMake(32, nsg, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

static int ds4_gpu_encode_mul_mv_addr_iq2_pair_swiglu_masked(
        id<MTLCommandBuffer>        cb,
        id<MTLComputePipelineState> pipeline,
        const ds4_gpu_mul_mv_id_args *args,
        const ds4_gpu_dsv4_moe_swiglu_weight_args *act,
        const ds4_gpu_stream_expert_split_args *split,
        ds4_gpu_stream_expert_cache_entry * const entries[6],
        id<MTLBuffer>               gate_addrs,
        id<MTLBuffer>               up_addrs,
        id<MTLBuffer>               src1,
        NSUInteger                  src1_off,
        id<MTLBuffer>               dst_a,
        NSUInteger                  dst_a_off,
        id<MTLBuffer>               dst_b,
        NSUInteger                  dst_b_off,
        id<MTLBuffer>               dst_mid,
        NSUInteger                  dst_mid_off,
        id<MTLBuffer>               ids,
        NSUInteger                  ids_off,
        id<MTLBuffer>               weights,
        NSUInteger                  weights_off,
        NSUInteger                  threadgroup_bytes,
        NSUInteger                  nsg,
        bool                        rows_per_group_is_nr0) {
    if (!cb || !pipeline || !args || !act || !split || !entries ||
        !gate_addrs || !up_addrs || !src1 || !dst_a || !dst_b || !dst_mid ||
        !ids || !weights ||
        args->ne00 <= 0 || args->ne01 <= 0 || args->nei0 != 6 || args->nei1 <= 0 ||
        args->ne02 <= 0 || args->ne02 > 384) {
        return 0;
    }
    for (uint32_t i = 0; i < 6; i++) {
        if ((split->active_mask & (1u << i)) == 0) continue;
        if (!entries[i] || !entries[i]->gate_buffer || !entries[i]->up_buffer) return 0;
    }
    if (!ds4_gpu_stream_expert_cache_mark_entries_inflight(entries,
                                                           6,
                                                           split->active_mask)) {
        return 0;
    }

    const NSUInteger nr0 = (NSUInteger)args->nr0;
    const NSUInteger rows_per_group = rows_per_group_is_nr0 ? nr0 : nr0 * nsg;
    const NSUInteger row_groups = ((NSUInteger)args->ne01 + rows_per_group - 1u) / rows_per_group;
    const NSUInteger pairs = (NSUInteger)args->nei0 * (NSUInteger)args->nei1;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:pipeline];
    [enc setBytes:args  length:sizeof(*args)  atIndex:0];
    [enc setBytes:act   length:sizeof(*act)   atIndex:1];
    [enc setBytes:split length:sizeof(*split) atIndex:2];
    [enc setBuffer:gate_addrs offset:0 atIndex:3];
    [enc setBuffer:up_addrs   offset:0 atIndex:4];
    [enc setBuffer:src1       offset:src1_off    atIndex:5];
    [enc setBuffer:dst_a      offset:dst_a_off   atIndex:6];
    [enc setBuffer:dst_b      offset:dst_b_off   atIndex:7];
    [enc setBuffer:dst_mid    offset:dst_mid_off atIndex:8];
    [enc setBuffer:ids        offset:ids_off     atIndex:9];
    [enc setBuffer:weights    offset:weights_off atIndex:10];
    for (uint32_t i = 0; i < 6; i++) {
        if ((split->active_mask & (1u << i)) == 0) continue;
        [enc useResource:entries[i]->gate_buffer usage:MTLResourceUsageRead];
        [enc useResource:entries[i]->up_buffer usage:MTLResourceUsageRead];
    }
    if (threadgroup_bytes != 0) {
        [enc setThreadgroupMemoryLength:threadgroup_bytes atIndex:0];
    }
    [enc dispatchThreadgroups:MTLSizeMake(row_groups, 1, pairs)
         threadsPerThreadgroup:MTLSizeMake(32, nsg, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

static int ds4_gpu_encode_mul_mv_addr_q2_sum6_masked(
        id<MTLCommandBuffer>        cb,
        id<MTLComputePipelineState> pipeline,
        const ds4_gpu_mul_mv_id_args *args,
        const ds4_gpu_stream_expert_split_args *split,
        ds4_gpu_stream_expert_cache_entry * const entries[6],
        id<MTLBuffer>               addrs,
        id<MTLBuffer>               src1,
        NSUInteger                  src1_off,
        id<MTLBuffer>               dst,
        NSUInteger                  dst_off,
        id<MTLBuffer>               ids,
        NSUInteger                  ids_off,
        NSUInteger                  threadgroup_bytes,
        NSUInteger                  nsg) {
    if (!cb || !pipeline || !args || !split || !entries || !addrs ||
        !src1 || !dst || !ids ||
        args->ne00 <= 0 || args->ne01 <= 0 || args->nei0 != 6 || args->nei1 <= 0 ||
        args->ne02 <= 0 || args->ne02 > 384) {
        return 0;
    }
    for (uint32_t i = 0; i < 6; i++) {
        if ((split->active_mask & (1u << i)) == 0) continue;
        if (!entries[i] || !entries[i]->down_buffer) return 0;
    }
    if (!ds4_gpu_stream_expert_cache_mark_entries_inflight(entries,
                                                           6,
                                                           split->active_mask)) {
        return 0;
    }

    const NSUInteger rows_per_group = (NSUInteger)args->nr0 * nsg;
    const NSUInteger row_groups = ((NSUInteger)args->ne01 + rows_per_group - 1u) / rows_per_group;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:pipeline];
    [enc setBytes:args  length:sizeof(*args)  atIndex:0];
    [enc setBytes:split length:sizeof(*split) atIndex:1];
    [enc setBuffer:addrs offset:0 atIndex:2];
    [enc setBuffer:src1  offset:src1_off atIndex:3];
    [enc setBuffer:dst   offset:dst_off  atIndex:4];
    [enc setBuffer:ids   offset:ids_off  atIndex:5];
    for (uint32_t i = 0; i < 6; i++) {
        if ((split->active_mask & (1u << i)) == 0) continue;
        [enc useResource:entries[i]->down_buffer usage:MTLResourceUsageRead];
    }
    if (threadgroup_bytes != 0) {
        [enc setThreadgroupMemoryLength:threadgroup_bytes atIndex:0];
    }
    [enc dispatchThreadgroups:MTLSizeMake(row_groups, (NSUInteger)args->nei1, 1)
         threadsPerThreadgroup:MTLSizeMake(32, nsg, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

static int ds4_gpu_encode_mul_mv_group8_pair_swiglu(
        id<MTLCommandBuffer>        cb,
        id<MTLComputePipelineState> pipeline,
        const ds4_gpu_mul_mv_id_args *args,
        const ds4_gpu_dsv4_moe_swiglu_weight_args *act,
        __unsafe_unretained id<MTLBuffer> src0_a[8],
        const NSUInteger            src0_a_off[8],
        __unsafe_unretained id<MTLBuffer> src0_b[8],
        const NSUInteger            src0_b_off[8],
        id<MTLBuffer>               src1,
        NSUInteger                  src1_off,
        id<MTLBuffer>               dst_a,
        NSUInteger                  dst_a_off,
        id<MTLBuffer>               dst_b,
        NSUInteger                  dst_b_off,
        id<MTLBuffer>               dst_mid,
        NSUInteger                  dst_mid_off,
        id<MTLBuffer>               ids,
        NSUInteger                  ids_off,
        id<MTLBuffer>               weights,
        NSUInteger                  weights_off,
        NSUInteger                  threadgroup_bytes,
        NSUInteger                  nsg,
        bool                        rows_per_group_is_nr0) {
    if (!cb || !pipeline || !args || !act || !src0_a || !src0_a_off || !src0_b || !src0_b_off ||
        !src1 || !dst_a || !dst_b || !dst_mid || !ids || !weights ||
        args->ne00 <= 0 || args->ne01 <= 0 || args->nei0 != 6 || args->nei1 <= 0 ||
        args->ne02 != 384) {
        return 0;
    }
    for (uint32_t i = 0; i < 8; i++) {
        if (!src0_a[i] || !src0_b[i]) return 0;
    }

    const NSUInteger nr0 = (NSUInteger)args->nr0;
    const NSUInteger rows_per_group = rows_per_group_is_nr0 ? nr0 : nr0 * nsg;
    const NSUInteger row_groups = ((NSUInteger)args->ne01 + rows_per_group - 1u) / rows_per_group;
    const NSUInteger pairs = (NSUInteger)args->nei0 * (NSUInteger)args->nei1;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:pipeline];
    [enc setBytes:args length:sizeof(*args) atIndex:0];
    [enc setBytes:act  length:sizeof(*act)  atIndex:1];
    for (uint32_t i = 0; i < 8; i++) {
        [enc setBuffer:src0_a[i] offset:src0_a_off[i] atIndex:2 + i];
    }
    for (uint32_t i = 0; i < 8; i++) {
        [enc setBuffer:src0_b[i] offset:src0_b_off[i] atIndex:10 + i];
    }
    [enc setBuffer:src1    offset:src1_off    atIndex:18];
    [enc setBuffer:dst_a   offset:dst_a_off   atIndex:19];
    [enc setBuffer:dst_b   offset:dst_b_off   atIndex:20];
    [enc setBuffer:dst_mid offset:dst_mid_off atIndex:21];
    [enc setBuffer:ids     offset:ids_off     atIndex:22];
    [enc setBuffer:weights offset:weights_off atIndex:23];
    if (threadgroup_bytes != 0) {
        [enc setThreadgroupMemoryLength:threadgroup_bytes atIndex:0];
    }
    [enc dispatchThreadgroups:MTLSizeMake(row_groups, 1, pairs)
         threadsPerThreadgroup:MTLSizeMake(32, nsg, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

static int ds4_gpu_encode_mul_mv_group8_sum6(
        id<MTLCommandBuffer>        cb,
        id<MTLComputePipelineState> pipeline,
        const ds4_gpu_mul_mv_id_args *args,
        __unsafe_unretained id<MTLBuffer> src0[8],
        const NSUInteger            src0_off[8],
        id<MTLBuffer>               src1,
        NSUInteger                  src1_off,
        id<MTLBuffer>               dst,
        NSUInteger                  dst_off,
        id<MTLBuffer>               ids,
        NSUInteger                  ids_off,
        NSUInteger                  threadgroup_bytes,
        NSUInteger                  nsg) {
    if (!cb || !pipeline || !args || !src0 || !src0_off || !src1 || !dst || !ids ||
        args->ne00 <= 0 || args->ne01 <= 0 || args->nei0 != 6 || args->nei1 <= 0 ||
        args->ne02 != 384) {
        return 0;
    }
    for (uint32_t i = 0; i < 8; i++) {
        if (!src0[i]) return 0;
    }

    const NSUInteger rows_per_group = (NSUInteger)args->nr0 * nsg;
    const NSUInteger row_groups = ((NSUInteger)args->ne01 + rows_per_group - 1u) / rows_per_group;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:pipeline];
    [enc setBytes:args length:sizeof(*args) atIndex:0];
    for (uint32_t i = 0; i < 8; i++) {
        [enc setBuffer:src0[i] offset:src0_off[i] atIndex:1 + i];
    }
    [enc setBuffer:src1 offset:src1_off atIndex:9];
    [enc setBuffer:dst  offset:dst_off  atIndex:10];
    [enc setBuffer:ids  offset:ids_off  atIndex:11];
    if (threadgroup_bytes != 0) {
        [enc setThreadgroupMemoryLength:threadgroup_bytes atIndex:0];
    }
    [enc dispatchThreadgroups:MTLSizeMake(row_groups, (NSUInteger)args->nei1, 1)
         threadsPerThreadgroup:MTLSizeMake(32, nsg, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

static int ds4_gpu_encode_mul_mv_group24_id(
        id<MTLCommandBuffer>        cb,
        id<MTLComputePipelineState> pipeline,
        const ds4_gpu_mul_mv_id_args *args,
        __unsafe_unretained id<MTLBuffer> src0[24],
        const NSUInteger            src0_off[24],
        id<MTLBuffer>               src1,
        NSUInteger                  src1_off,
        id<MTLBuffer>               dst,
        NSUInteger                  dst_off,
        id<MTLBuffer>               ids,
        NSUInteger                  ids_off,
        NSUInteger                  threadgroup_bytes,
        NSUInteger                  nsg,
        bool                        rows_per_group_is_nr0) {
    if (!cb || !pipeline || !args || !src0 || !src0_off || !src1 || !dst || !ids ||
        args->ne00 <= 0 || args->ne01 <= 0 || args->nei0 != 6 || args->nei1 <= 0 ||
        args->ne02 != 384) {
        return 0;
    }
    for (uint32_t i = 0; i < 24; i++) {
        if (!src0[i]) return 0;
    }

    const NSUInteger nr0 = (NSUInteger)args->nr0;
    const NSUInteger rows_per_group = rows_per_group_is_nr0 ? nr0 : nr0 * nsg;
    const NSUInteger row_groups = ((NSUInteger)args->ne01 + rows_per_group - 1u) / rows_per_group;
    const NSUInteger pairs = (NSUInteger)args->nei0 * (NSUInteger)args->nei1;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:pipeline];
    [enc setBytes:args length:sizeof(*args) atIndex:0];
    for (uint32_t i = 0; i < 24; i++) {
        [enc setBuffer:src0[i] offset:src0_off[i] atIndex:1 + i];
    }
    [enc setBuffer:src1 offset:src1_off atIndex:25];
    [enc setBuffer:dst  offset:dst_off  atIndex:26];
    [enc setBuffer:ids  offset:ids_off  atIndex:27];
    if (threadgroup_bytes != 0) {
        [enc setThreadgroupMemoryLength:threadgroup_bytes atIndex:0];
    }
    [enc dispatchThreadgroups:MTLSizeMake(row_groups, 1, pairs)
         threadsPerThreadgroup:MTLSizeMake(32, nsg, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

static int ds4_gpu_encode_mul_mv_group24_sum6(
        id<MTLCommandBuffer>        cb,
        id<MTLComputePipelineState> pipeline,
        const ds4_gpu_mul_mv_id_args *args,
        __unsafe_unretained id<MTLBuffer> src0[24],
        const NSUInteger            src0_off[24],
        id<MTLBuffer>               src1,
        NSUInteger                  src1_off,
        id<MTLBuffer>               dst,
        NSUInteger                  dst_off,
        id<MTLBuffer>               ids,
        NSUInteger                  ids_off,
        NSUInteger                  threadgroup_bytes,
        NSUInteger                  nsg) {
    if (!cb || !pipeline || !args || !src0 || !src0_off || !src1 || !dst || !ids ||
        args->ne00 <= 0 || args->ne01 <= 0 || args->nei0 != 6 || args->nei1 <= 0 ||
        args->ne02 != 384) {
        return 0;
    }
    for (uint32_t i = 0; i < 24; i++) {
        if (!src0[i]) return 0;
    }

    const NSUInteger rows_per_group = (NSUInteger)args->nr0 * nsg;
    const NSUInteger row_groups = ((NSUInteger)args->ne01 + rows_per_group - 1u) / rows_per_group;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:pipeline];
    [enc setBytes:args length:sizeof(*args) atIndex:0];
    for (uint32_t i = 0; i < 24; i++) {
        [enc setBuffer:src0[i] offset:src0_off[i] atIndex:1 + i];
    }
    [enc setBuffer:src1 offset:src1_off atIndex:25];
    [enc setBuffer:dst  offset:dst_off  atIndex:26];
    [enc setBuffer:ids  offset:ids_off  atIndex:27];
    if (threadgroup_bytes != 0) {
        [enc setThreadgroupMemoryLength:threadgroup_bytes atIndex:0];
    }
    [enc dispatchThreadgroups:MTLSizeMake(row_groups, (NSUInteger)args->nei1, 1)
         threadsPerThreadgroup:MTLSizeMake(32, nsg, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

static int ds4_gpu_encode_mul_mm_id(
        id<MTLCommandBuffer>        cb,
        id<MTLComputePipelineState> map_pipeline,
        id<MTLComputePipelineState> mm_pipeline,
        const ds4_gpu_mul_mm_id_map_args *map_args,
        const ds4_gpu_mul_mm_id_args *mm_args,
        id<MTLBuffer>               src0,
        NSUInteger                  src0_off,
        id<MTLBuffer>               src1,
        NSUInteger                  src1_off,
        id<MTLBuffer>               dst,
        NSUInteger                  dst_off,
        id<MTLBuffer>               ids,
        NSUInteger                  ids_off) {
    if (!cb || !map_pipeline || !mm_pipeline || !map_args || !mm_args ||
        !src0 || !src1 || !dst || !ids ||
        mm_args->ne00 <= 0 || mm_args->ne0 <= 0 ||
        mm_args->ne20 <= 0 || mm_args->ne21 <= 0 || mm_args->ne02 <= 0) {
        return 0;
    }

    return ds4_gpu_encode_mul_mm_id_map(cb,
                                          map_pipeline,
                                          map_args,
                                          mm_args,
                                          ids,
                                          ids_off) &&
           ds4_gpu_encode_mul_mm_id_mapped(cb,
                                             mm_pipeline,
                                             mm_args,
                                             src0,
                                             src0_off,
                                             src1,
                                             src1_off,
                                             dst,
                                             dst_off);
}

static int ds4_gpu_encode_mul_mm_id_map(
        id<MTLCommandBuffer>        cb,
        id<MTLComputePipelineState> map_pipeline,
        const ds4_gpu_mul_mm_id_map_args *map_args,
        const ds4_gpu_mul_mm_id_args *mm_args,
        id<MTLBuffer>               ids,
        NSUInteger                  ids_off) {
    if (!cb || !map_pipeline || !map_args || !mm_args || !ids ||
        mm_args->ne20 <= 0 || mm_args->ne21 <= 0 || mm_args->ne02 <= 0) {
        return 0;
    }

    const NSUInteger tpe_bytes = (NSUInteger)mm_args->ne02 * sizeof(int32_t);
    const NSUInteger hids_bytes = (NSUInteger)mm_args->ne02 * (NSUInteger)mm_args->ne21 * sizeof(int32_t);
    if (tpe_bytes > NSUIntegerMax - hids_bytes) return 0;
    if (!ds4_gpu_ensure_scratch_buffer(&g_moe_id_map_buffer,
                                         &g_moe_id_map_bytes,
                                         tpe_bytes + hids_bytes,
                                         "ds4_moe_id_map")) {
        return 0;
    }

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:map_pipeline];
    [enc setBytes:map_args length:sizeof(*map_args) atIndex:0];
    [enc setBuffer:ids offset:ids_off atIndex:1];
    [enc setBuffer:g_moe_id_map_buffer offset:0 atIndex:2];
    [enc setBuffer:g_moe_id_map_buffer offset:tpe_bytes atIndex:3];
    [enc setThreadgroupMemoryLength:(NSUInteger)mm_args->ne02 * (NSUInteger)mm_args->ne20 * sizeof(uint16_t) atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
         threadsPerThreadgroup:MTLSizeMake((NSUInteger)mm_args->ne02, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

static int ds4_gpu_encode_mul_mm_id_mapped_tile(
        id<MTLCommandBuffer>        cb,
        id<MTLComputePipelineState> mm_pipeline,
        const ds4_gpu_mul_mm_id_args *mm_args,
        id<MTLBuffer>               src0,
        NSUInteger                  src0_off,
        id<MTLBuffer>               src1,
        NSUInteger                  src1_off,
        id<MTLBuffer>               dst,
        NSUInteger                  dst_off,
        NSUInteger                  threadgroup_bytes) {
    if (!cb || !mm_pipeline || !mm_args || !src0 || !src1 || !dst ||
        !g_moe_id_map_buffer ||
        mm_args->ne00 <= 0 || mm_args->ne0 <= 0 ||
        mm_args->ne20 <= 0 || mm_args->ne21 <= 0 || mm_args->ne02 <= 0) {
        return 0;
    }
    /*
     * The routed MoE grouped matmul uses the legacy 32-token expert-major tile.
     * The removed TensorOps variant was not semantically stable on evals, so keep
     * this encoder tied to the tested simdgroup kernel shape.
     */
    const NSUInteger tile_n = 32u;
    const bool use_resource_hints =
        getenv("DS4_METAL_MOE_MM_ID_USE_RESOURCES") != NULL &&
        getenv("DS4_METAL_DISABLE_MOE_MM_ID_USE_RESOURCES") == NULL;

    const NSUInteger tpe_bytes = (NSUInteger)mm_args->ne02 * sizeof(int32_t);
    const NSUInteger hids_bytes = (NSUInteger)mm_args->ne02 * (NSUInteger)mm_args->ne21 * sizeof(int32_t);
    if (tpe_bytes > NSUIntegerMax - hids_bytes ||
        g_moe_id_map_bytes < tpe_bytes + hids_bytes) {
        return 0;
    }

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:mm_pipeline];
    [enc setBytes:mm_args length:sizeof(*mm_args) atIndex:0];
    [enc setBuffer:src0 offset:src0_off atIndex:1];
    [enc setBuffer:src1 offset:src1_off atIndex:2];
    [enc setBuffer:g_moe_id_map_buffer offset:0 atIndex:3];
    [enc setBuffer:g_moe_id_map_buffer offset:tpe_bytes atIndex:4];
    [enc setBuffer:dst offset:dst_off atIndex:5];
    if (use_resource_hints) {
        [enc useResource:src0 usage:MTLResourceUsageRead];
    }
    [enc setThreadgroupMemoryLength:threadgroup_bytes atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)mm_args->ne21 + tile_n - 1u) / tile_n,
                                          ((NSUInteger)mm_args->ne0 + 63u) / 64u,
                                          (NSUInteger)mm_args->ne02)
         threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

static int ds4_gpu_encode_mul_mm_id_addr_mapped_tile(
        id<MTLCommandBuffer>        cb,
        id<MTLComputePipelineState> mm_pipeline,
        const ds4_gpu_mul_mm_id_args *mm_args,
        id<MTLBuffer>               src0_addrs,
        id<MTLBuffer>               src1,
        NSUInteger                  src1_off,
        id<MTLBuffer>               dst,
        NSUInteger                  dst_off,
        NSUInteger                  threadgroup_bytes,
        ds4_gpu_stream_expert_cache_entry * const *resources,
        uint32_t                    resource_count,
        uint32_t                    resource_kind,
        id<MTLBuffer>               overflow_resource) {
    if (!cb || !mm_pipeline || !mm_args || !src0_addrs || !src1 || !dst ||
        !g_moe_id_map_buffer ||
        mm_args->ne00 <= 0 || mm_args->ne0 <= 0 ||
        mm_args->ne20 <= 0 || mm_args->ne21 <= 0 || mm_args->ne02 <= 0) {
        return 0;
    }

    const NSUInteger tile_n = 32u;
    const NSUInteger tpe_bytes = (NSUInteger)mm_args->ne02 * sizeof(int32_t);
    const NSUInteger hids_bytes =
        (NSUInteger)mm_args->ne02 * (NSUInteger)mm_args->ne21 * sizeof(int32_t);
    if (tpe_bytes > NSUIntegerMax - hids_bytes ||
        g_moe_id_map_bytes < tpe_bytes + hids_bytes) {
        return 0;
    }

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:mm_pipeline];
    [enc setBytes:mm_args length:sizeof(*mm_args) atIndex:0];
    [enc setBuffer:src0_addrs offset:0 atIndex:1];
    [enc setBuffer:src1 offset:src1_off atIndex:2];
    [enc setBuffer:g_moe_id_map_buffer offset:0 atIndex:3];
    [enc setBuffer:g_moe_id_map_buffer offset:tpe_bytes atIndex:4];
    [enc setBuffer:dst offset:dst_off atIndex:5];
    [enc useResource:src0_addrs usage:MTLResourceUsageRead];
    for (uint32_t i = 0; resources && i < resource_count; i++) {
        ds4_gpu_stream_expert_cache_entry *entry = resources[i];
        if (!entry) continue;
        id<MTLBuffer> b =
            resource_kind == 0 ? entry->gate_buffer :
            resource_kind == 1 ? entry->up_buffer :
                                 entry->down_buffer;
        if (b) [enc useResource:b usage:MTLResourceUsageRead];
    }
    if (overflow_resource) {
        [enc useResource:overflow_resource usage:MTLResourceUsageRead];
    }
    [enc setThreadgroupMemoryLength:threadgroup_bytes atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)mm_args->ne21 + tile_n - 1u) / tile_n,
                                          ((NSUInteger)mm_args->ne0 + 63u) / 64u,
                                          (NSUInteger)mm_args->ne02)
         threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

static int ds4_gpu_encode_mul_mm_id_iq2_pair_swiglu_f16(
        id<MTLCommandBuffer>        cb,
        id<MTLComputePipelineState> pipeline,
        const ds4_gpu_mul_mm_id_args *mm_args,
        const ds4_gpu_dsv4_moe_swiglu_weight_args *act_args,
        id<MTLBuffer>               gate_src0,
        NSUInteger                  gate_src0_off,
        id<MTLBuffer>               up_src0,
        NSUInteger                  up_src0_off,
        id<MTLBuffer>               src1,
        NSUInteger                  src1_off,
        id<MTLBuffer>               mid,
        NSUInteger                  mid_off,
        id<MTLBuffer>               weights,
        NSUInteger                  weights_off) {
    if (!cb || !pipeline || !mm_args || !act_args ||
        !gate_src0 || !up_src0 || !src1 || !mid || !weights ||
        !g_moe_id_map_buffer ||
        mm_args->ne00 <= 0 || mm_args->ne0 <= 0 ||
        mm_args->ne20 <= 0 || mm_args->ne21 <= 0 || mm_args->ne02 <= 0) {
        return 0;
    }

    const NSUInteger tpe_bytes = (NSUInteger)mm_args->ne02 * sizeof(int32_t);
    const NSUInteger hids_bytes = (NSUInteger)mm_args->ne02 * (NSUInteger)mm_args->ne21 * sizeof(int32_t);
    if (tpe_bytes > NSUIntegerMax - hids_bytes ||
        g_moe_id_map_bytes < tpe_bytes + hids_bytes) {
        return 0;
    }

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:pipeline];
    [enc setBytes:mm_args length:sizeof(*mm_args) atIndex:0];
    [enc setBytes:act_args length:sizeof(*act_args) atIndex:1];
    [enc setBuffer:gate_src0 offset:gate_src0_off atIndex:2];
    [enc setBuffer:up_src0 offset:up_src0_off atIndex:3];
    [enc setBuffer:src1 offset:src1_off atIndex:4];
    [enc setBuffer:g_moe_id_map_buffer offset:0 atIndex:5];
    [enc setBuffer:g_moe_id_map_buffer offset:tpe_bytes atIndex:6];
    [enc setBuffer:mid offset:mid_off atIndex:7];
    [enc setBuffer:weights offset:weights_off atIndex:8];
    [enc setThreadgroupMemoryLength:16384u atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)mm_args->ne21 + 31u) / 32u,
                                          ((NSUInteger)mm_args->ne0 + 63u) / 64u,
                                          (NSUInteger)mm_args->ne02)
         threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

static int ds4_gpu_encode_mul_mm_id_mapped(
        id<MTLCommandBuffer>        cb,
        id<MTLComputePipelineState> mm_pipeline,
        const ds4_gpu_mul_mm_id_args *mm_args,
        id<MTLBuffer>               src0,
        NSUInteger                  src0_off,
        id<MTLBuffer>               src1,
        NSUInteger                  src1_off,
        id<MTLBuffer>               dst,
        NSUInteger                  dst_off) {
    return ds4_gpu_encode_mul_mm_id_mapped_tile(cb,
                                                  mm_pipeline,
                                                  mm_args,
                                                  src0,
                                                  src0_off,
                                                  src1,
                                                  src1_off,
                                                  dst,
                                                  dst_off,
                                                  8192u);
}

static int ds4_gpu_encode_attn_out_low_q8_mpp(
        id<MTLCommandBuffer>           cb,
        id<MTLComputePipelineState>    pipeline,
        const ds4_gpu_mul_mm_id_args *mm_args,
        id<MTLBuffer>                  src0,
        NSUInteger                     src0_off,
        id<MTLBuffer>                  src1,
        NSUInteger                     src1_off,
        id<MTLBuffer>                  dst,
        NSUInteger                     dst_off) {
    if (!cb || !pipeline || !mm_args || !src0 || !src1 || !dst ||
        mm_args->ne00 <= 0 || mm_args->ne0 <= 0 ||
        mm_args->ne02 <= 0 || mm_args->ne1 <= 0 || mm_args->ne21 <= 0) {
        return 0;
    }

    const uint32_t tile_n = DS4_METAL_ATTN_OUT_MPP_TILE_N;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:pipeline];
    [enc setBytes:mm_args length:sizeof(*mm_args) atIndex:0];
    [enc setBuffer:src0 offset:src0_off atIndex:1];
    [enc setBuffer:src1 offset:src1_off atIndex:2];
    [enc setBuffer:dst offset:dst_off atIndex:3];
    [enc setThreadgroupMemoryLength:8192u atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)mm_args->ne21 + (NSUInteger)tile_n - 1u) / (NSUInteger)tile_n,
                                          ((NSUInteger)mm_args->ne0 + 63u) / 64u,
                                          (NSUInteger)mm_args->ne02)
         threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

static int ds4_gpu_encode_swiglu_flat(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        gate,
        NSUInteger           gate_off,
        id<MTLBuffer>        up,
        NSUInteger           up_off,
        id<MTLBuffer>        out,
        NSUInteger           out_off,
        uint32_t             n) {
    if (!cb || !gate || !up || !out || n == 0) return 0;

    ds4_gpu_glu_args args = {
        .ne00 = (int32_t)n,
        .nb01 = (uint64_t)n * sizeof(float),
        .ne10 = (int32_t)n,
        .nb11 = (uint64_t)n * sizeof(float),
        .ne0 = (int32_t)n,
        .nb1 = (uint64_t)n * sizeof(float),
        .i00 = 0,
        .i10 = 0,
        .alpha = 1.0f,
        .limit = 0.0f,
    };
    NSUInteger nth = g_swiglu_flat_pipeline.maxTotalThreadsPerThreadgroup;
    if (nth > 256u) nth = 256u;
    if (nth > (NSUInteger)n) nth = (NSUInteger)n;
    if (nth == 0u) nth = 1u;
    const NSUInteger groups = ((NSUInteger)n + nth - 1u) / nth;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:g_swiglu_flat_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:gate offset:gate_off atIndex:1];
    [enc setBuffer:up   offset:up_off   atIndex:2];
    [enc setBuffer:out  offset:out_off  atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

static int ds4_gpu_encode_moe_swiglu_weight(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        gate,
        NSUInteger           gate_off,
        id<MTLBuffer>        up,
        NSUInteger           up_off,
        id<MTLBuffer>        mid,
        NSUInteger           mid_off,
        id<MTLBuffer>        weights,
        NSUInteger           weights_off,
        uint32_t             width,
        uint32_t             rows,
        float                clamp_value,
        bool                 mid_f16) {
    if (!cb || !gate || !up || !mid || !weights || width == 0 || rows == 0) return 0;

    id<MTLComputePipelineState> pipeline =
        ds4_gpu_get_pipeline(mid_f16 ? "kernel_dsv4_moe_swiglu_weight_f16" :
                                         "kernel_dsv4_moe_swiglu_weight");
    if (!pipeline) return 0;

    ds4_gpu_dsv4_moe_swiglu_weight_args args = {
        .width = width,
        .rows = rows,
        .gate_row_stride = (uint64_t)width * sizeof(float),
        .up_row_stride = (uint64_t)width * sizeof(float),
        .mid_row_stride = (uint64_t)width * (mid_f16 ? sizeof(uint16_t) : sizeof(float)),
        .weight_stride = sizeof(float),
        .write_clamped = getenv("DS4_METAL_MOE_WRITE_CLAMPED_ACT") != NULL ? 1u : 0u,
        .clamp_value = clamp_value,
    };

    NSUInteger nth = pipeline.maxTotalThreadsPerThreadgroup;
    if (nth > 256u) nth = 256u;
    if (nth > width) nth = width;
    if (nth == 0) nth = 1u;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:gate    offset:gate_off    atIndex:1];
    [enc setBuffer:up      offset:up_off      atIndex:2];
    [enc setBuffer:mid     offset:mid_off     atIndex:3];
    [enc setBuffer:weights offset:weights_off atIndex:4];
    [enc dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

static int ds4_gpu_encode_moe_sum6(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        experts,
        NSUInteger           experts_off,
        id<MTLBuffer>        out,
        NSUInteger           out_off,
        uint32_t             out_dim,
        uint32_t             n_tokens) {
    if (!cb || !experts || !out || out_dim == 0 || n_tokens == 0) return 0;

    if (!g_moe_sum6_pipeline) return 0;

    const uint64_t out_row_bytes = (uint64_t)out_dim * sizeof(float);
    ds4_gpu_dsv4_moe_sum6_args args = {
        .width = out_dim,
        .tokens = n_tokens,
        .src_token_stride = 6u * out_row_bytes,
        .dst_token_stride = out_row_bytes,
    };

    NSUInteger nth = g_moe_sum6_pipeline.maxTotalThreadsPerThreadgroup;
    if (nth > 256u) nth = 256u;
    if (nth > out_dim) nth = out_dim;
    if (nth == 0) nth = 1u;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:g_moe_sum6_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:experts offset:experts_off atIndex:1];
    [enc setBuffer:out     offset:out_off     atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)n_tokens, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

static int ds4_gpu_encode_moe_sum8(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        experts,
        NSUInteger           experts_off,
        id<MTLBuffer>        out,
        NSUInteger           out_off,
        uint32_t             out_dim,
        uint32_t             n_tokens) {
    if (!cb || !experts || !out || out_dim == 0 || n_tokens == 0) return 0;

    if (!g_moe_sum8_pipeline) return 0;

    const uint64_t out_row_bytes = (uint64_t)out_dim * sizeof(float);
    ds4_gpu_dsv4_moe_sum6_args args = {
        .width = out_dim,
        .tokens = n_tokens,
        .src_token_stride = 8u * out_row_bytes,
        .dst_token_stride = out_row_bytes,
    };

    NSUInteger nth = g_moe_sum8_pipeline.maxTotalThreadsPerThreadgroup;
    if (nth > 256u) nth = 256u;
    if (nth > out_dim) nth = out_dim;
    if (nth == 0) nth = 1u;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:g_moe_sum8_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:experts offset:experts_off atIndex:1];
    [enc setBuffer:out     offset:out_off     atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)n_tokens, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

static ds4_gpu_bin_args ds4_gpu_make_moe_add_args(
        uint32_t out_dim,
        uint32_t n_tokens,
        uint64_t src0_token_stride,
        uint64_t src1_token_stride,
        uint64_t dst_token_stride) {
    return (ds4_gpu_bin_args) {
        .ne00 = (int32_t)out_dim,
        .ne01 = (int32_t)n_tokens,
        .ne02 = 1,
        .ne03 = 1,
        .nb00 = sizeof(float),
        .nb01 = src0_token_stride,
        .nb02 = (uint64_t)n_tokens * src0_token_stride,
        .nb03 = (uint64_t)n_tokens * src0_token_stride,
        .ne10 = (int32_t)out_dim,
        .ne11 = (int32_t)n_tokens,
        .ne12 = 1,
        .ne13 = 1,
        .nb10 = sizeof(float),
        .nb11 = src1_token_stride,
        .nb12 = (uint64_t)n_tokens * src1_token_stride,
        .nb13 = (uint64_t)n_tokens * src1_token_stride,
        .ne0 = (int32_t)out_dim,
        .ne1 = (int32_t)n_tokens,
        .ne2 = 1,
        .ne3 = 1,
        .nb0 = sizeof(float),
        .nb1 = dst_token_stride,
        .nb2 = (uint64_t)n_tokens * dst_token_stride,
        .nb3 = (uint64_t)n_tokens * dst_token_stride,
        .offs = 0,
        .o1 = { 0 },
    };
}

static int ds4_gpu_encode_moe_sum_experts(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        experts,
        NSUInteger           experts_off,
        id<MTLBuffer>        out,
        NSUInteger           out_off,
        uint32_t             out_dim,
        uint32_t             n_expert,
        uint32_t             n_tokens) {
    if (!cb || !experts || !out || out_dim == 0 || n_expert < 2 || n_tokens == 0) return 0;

    const uint64_t out_row_bytes = (uint64_t)out_dim * sizeof(float);
    const uint64_t expert_token_stride = (uint64_t)n_expert * out_row_bytes;

    if (n_expert == 6 &&
        ds4_gpu_encode_moe_sum6(cb,
                                  experts,
                                  experts_off,
                                  out,
                                  out_off,
                                  out_dim,
                                  n_tokens)) {
        return 1;
    }

    if (n_expert == 8 &&
        ds4_gpu_encode_moe_sum8(cb,
                                  experts,
                                  experts_off,
                                  out,
                                  out_off,
                                  out_dim,
                                  n_tokens)) {
        return 1;
    }

    ds4_gpu_bin_args first =
        ds4_gpu_make_moe_add_args(out_dim, n_tokens, expert_token_stride, expert_token_stride, out_row_bytes);
    if (!ds4_gpu_encode_bin_f32_rows(cb,
                                       g_add_pipeline,
                                       &first,
                                       experts,
                                       experts_off,
                                       experts,
                                       experts_off + (NSUInteger)out_row_bytes,
                                       out,
                                       out_off)) {
        return 0;
    }

    ds4_gpu_bin_args accum =
        ds4_gpu_make_moe_add_args(out_dim, n_tokens, out_row_bytes, expert_token_stride, out_row_bytes);
    for (uint32_t slot = 2; slot < n_expert; slot++) {
        if (!ds4_gpu_encode_bin_f32_rows(cb,
                                           g_add_pipeline,
                                           &accum,
                                           out,
                                           out_off,
                                           experts,
                                           experts_off + (NSUInteger)((uint64_t)slot * out_row_bytes),
                                           out,
                                           out_off)) {
            return 0;
        }
    }
    return 1;
}

static int ds4_gpu_encode_get_rows_i32_token_rows(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        table,
        NSUInteger           table_off,
        id<MTLBuffer>        tokens,
        NSUInteger           tokens_off,
        const int32_t       *token_inline,
        id<MTLBuffer>        selected,
        NSUInteger           selected_off,
        uint32_t             hash_rows,
        uint32_t             n_cols,
        uint32_t             n_tokens) {
    if (!cb || !table || !selected || hash_rows == 0 || n_cols == 0 || n_tokens == 0) return 0;
    if (!tokens && !token_inline) return 0;

    const uint64_t table_row_bytes = (uint64_t)n_cols * sizeof(int32_t);
    const uint64_t token_bytes = (uint64_t)n_tokens * sizeof(int32_t);
    ds4_gpu_get_rows_args args = {
        .ne00t = (int64_t)n_cols,
        .ne00 = (int64_t)n_cols,
        .nb01 = table_row_bytes,
        .nb02 = (uint64_t)hash_rows * table_row_bytes,
        .nb03 = (uint64_t)hash_rows * table_row_bytes,
        .ne10 = (int32_t)n_tokens,
        .nb10 = sizeof(int32_t),
        .nb11 = token_bytes,
        .nb12 = token_bytes,
        .nb1 = table_row_bytes,
        .nb2 = (uint64_t)n_tokens * table_row_bytes,
        .nb3 = (uint64_t)n_tokens * table_row_bytes,
    };

    NSUInteger nth = (NSUInteger)n_cols;
    const NSUInteger max_threads = g_get_rows_i32_pipeline.maxTotalThreadsPerThreadgroup;
    if (nth > max_threads) nth = max_threads;
    if (nth == 0) nth = 1u;
    const NSUInteger nw0 = ((NSUInteger)n_cols + nth - 1u) / nth;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:g_get_rows_i32_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:table offset:table_off atIndex:1];
    if (tokens) {
        [enc setBuffer:tokens offset:tokens_off atIndex:2];
    } else {
        [enc setBytes:token_inline length:sizeof(*token_inline) atIndex:2];
    }
    [enc setBuffer:selected offset:selected_off atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake(nw0 * n_tokens, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

static int ds4_gpu_encode_get_rows_f32_router_weights(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        probs,
        NSUInteger           probs_off,
        id<MTLBuffer>        selected,
        NSUInteger           selected_off,
        id<MTLBuffer>        weights,
        NSUInteger           weights_off,
        uint32_t             n_expert,
        uint32_t             n_expert_used,
        uint32_t             n_tokens) {
    if (!cb || !probs || !selected || !weights || n_expert == 0 || n_expert_used == 0 || n_tokens == 0) return 0;

    const uint64_t probs_token_bytes = (uint64_t)n_expert * sizeof(float);
    const uint64_t selected_row_bytes = (uint64_t)n_expert_used * sizeof(int32_t);
    const uint64_t weights_row_bytes = (uint64_t)n_expert_used * sizeof(float);
    ds4_gpu_get_rows_args args = {
        .ne00t = 1,
        .ne00 = 1,
        .nb01 = sizeof(float),
        .nb02 = probs_token_bytes,
        .nb03 = (uint64_t)n_tokens * probs_token_bytes,
        .ne10 = (int64_t)n_expert_used,
        .nb10 = sizeof(int32_t),
        .nb11 = selected_row_bytes,
        .nb12 = (uint64_t)n_tokens * selected_row_bytes,
        .nb1 = sizeof(float),
        .nb2 = weights_row_bytes,
        .nb3 = (uint64_t)n_tokens * weights_row_bytes,
    };

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:g_get_rows_f32_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:probs offset:probs_off atIndex:1];
    [enc setBuffer:selected offset:selected_off atIndex:2];
    [enc setBuffer:weights offset:weights_off atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)n_expert_used, n_tokens, 1)
         threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

static int ds4_gpu_encode_sum_rows_f32(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        src,
        NSUInteger           src_off,
        id<MTLBuffer>        dst,
        NSUInteger           dst_off,
        uint32_t             width,
        uint32_t             rows) {
    if (!cb || !src || !dst || width == 0 || rows == 0) return 0;

    const uint64_t src_row_bytes = (uint64_t)width * sizeof(float);
    ds4_gpu_kargs_sum_rows args = {
        .ne00 = (int64_t)width,
        .ne01 = (int64_t)rows,
        .ne02 = 1,
        .ne03 = 1,
        .nb00 = sizeof(float),
        .nb01 = src_row_bytes,
        .nb02 = (uint64_t)rows * src_row_bytes,
        .nb03 = (uint64_t)rows * src_row_bytes,
        .ne0 = 1,
        .ne1 = (int64_t)rows,
        .ne2 = 1,
        .ne3 = 1,
        .nb0 = sizeof(float),
        .nb1 = sizeof(float),
        .nb2 = (uint64_t)rows * sizeof(float),
        .nb3 = (uint64_t)rows * sizeof(float),
    };

    NSUInteger nth = 32u;
    const NSUInteger max_threads = g_sum_rows_f32_f32_pipeline.maxTotalThreadsPerThreadgroup;
    while (nth < (NSUInteger)args.ne00 && nth < max_threads) nth *= 2u;
    if (nth > max_threads) nth = max_threads;
    if (nth > (NSUInteger)args.ne00) nth = (NSUInteger)args.ne00;
    if (nth == 0) nth = 1u;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:g_sum_rows_f32_f32_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:src offset:src_off atIndex:1];
    [enc setBuffer:dst offset:dst_off atIndex:2];
    [enc setThreadgroupMemoryLength:32u * sizeof(float) atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

static int ds4_gpu_encode_router_select(
        id<MTLCommandBuffer>  cb,
        ds4_gpu_tensor     *selected,
        ds4_gpu_tensor     *weights,
        ds4_gpu_tensor     *probs,
        id<MTLBuffer>         logitsbuf,
        NSUInteger            logits_off,
        id<MTLBuffer>         biasbuf,
        NSUInteger            bias_off,
        id<MTLBuffer>         hashbuf,
        NSUInteger            hash_off,
        id<MTLBuffer>         tokensbuf,
        NSUInteger            tokens_off,
        const int32_t        *single_token,
        uint32_t              hash_rows,
        uint32_t              n_tokens,
        uint32_t              n_expert,
        uint32_t              n_expert_used,
        float                 expert_weight_scale,
        bool                  has_bias,
        bool                  hash_mode) {
    id<MTLBuffer> selectedbuf = ds4_gpu_tensor_buffer(selected);
    id<MTLBuffer> weightsbuf = ds4_gpu_tensor_buffer(weights);
    id<MTLBuffer> probsbuf = ds4_gpu_tensor_buffer(probs);
    const NSUInteger selected_off = ds4_gpu_tensor_offset(selected);
    const NSUInteger weights_off = ds4_gpu_tensor_offset(weights);
    const NSUInteger probs_off = ds4_gpu_tensor_offset(probs);

    if (!cb || !selectedbuf || !weightsbuf || !probsbuf || !logitsbuf ||
        n_tokens == 0 || n_expert == 0 || n_expert_used == 0) return 0;

    const NSUInteger probs_bytes = (NSUInteger)n_tokens * (NSUInteger)n_expert * sizeof(float);
    const bool flash_router_fast_path =
        n_expert == 256u &&
        n_expert_used == 6u &&
        fabsf(expert_weight_scale - 1.5f) <= 1.0e-6f;

    int ok = 0;
    if (flash_router_fast_path &&
        !g_quality_mode && n_tokens == 1 &&
        getenv("DS4_METAL_DISABLE_ROUTER_SELECT_FUSION") == NULL) {
        const bool force_simd_weights_fusion =
            getenv("DS4_METAL_ENABLE_ROUTER_SIMD_WEIGHTS_FUSION") != NULL;
        const bool use_simd_finalize =
            !hash_mode &&
            g_dsv4_router_finalize_one_simd_pipeline != nil &&
            g_dsv4_router_finalize_one_simd_pipeline.threadExecutionWidth == 32u &&
            g_dsv4_router_finalize_one_simd_pipeline.maxTotalThreadsPerThreadgroup >= 256u &&
            (ds4_gpu_device_name_contains("M3") ||
             ds4_gpu_device_name_contains("M5") ||
             getenv("DS4_METAL_ENABLE_ROUTER_SIMD_FINALIZE") != NULL ||
             force_simd_weights_fusion) &&
            getenv("DS4_METAL_DISABLE_M3_ROUTER_SIMD_FINALIZE") == NULL;
        const bool use_simd_weights_fusion =
            use_simd_finalize &&
            g_dsv4_router_finalize_weights_one_simd_pipeline != nil &&
            g_dsv4_router_finalize_weights_one_simd_pipeline.threadExecutionWidth == 32u &&
            g_dsv4_router_finalize_weights_one_simd_pipeline.maxTotalThreadsPerThreadgroup >= 256u &&
            (ds4_gpu_device_name_contains("M3") ||
             ds4_gpu_device_name_contains("M5") ||
             force_simd_weights_fusion) &&
            getenv("DS4_METAL_DISABLE_M3_ROUTER_SIMD_WEIGHTS_FUSION") == NULL;
        id<MTLComputePipelineState> softplus_sqrt_pipeline =
            ds4_gpu_hot_pipeline(g_dsv4_softplus_sqrt_pipeline,
                                    "kernel_dsv4_softplus_sqrt_f32_4");
        id<MTLComputePipelineState> router_finalize_pipeline =
            ds4_gpu_hot_pipeline(
                use_simd_weights_fusion
                    ? g_dsv4_router_finalize_weights_one_simd_pipeline
                    : use_simd_finalize
                    ? g_dsv4_router_finalize_one_simd_pipeline
                    : g_dsv4_router_finalize_one_pipeline,
                use_simd_weights_fusion
                    ? "kernel_dsv4_router_finalize_weights_one_simd"
                    : use_simd_finalize
                    ? "kernel_dsv4_router_finalize_one_simd"
                    : "kernel_dsv4_router_finalize_one");
        id<MTLComputePipelineState> router_weights_pipeline = use_simd_weights_fusion
            ? nil
            : ds4_gpu_hot_pipeline(g_dsv4_router_weights_one_pipeline,
                                      "kernel_dsv4_router_weights_one");
        if (!softplus_sqrt_pipeline || !router_finalize_pipeline ||
            (!use_simd_weights_fusion && !router_weights_pipeline)) return 0;

        ok = ds4_gpu_encode_unary_f32_rows(cb,
                                             softplus_sqrt_pipeline,
                                             logitsbuf,
                                             logits_off,
                                             probsbuf,
                                             probs_off,
                                             n_expert,
                                             1,
                                             1,
                                             0.0f,
                                             0.0f);
        if (!ok) return 0;

        const bool use_token_buffer = single_token == NULL;
        ds4_gpu_dsv4_router_select_one_args args = {
            .has_bias = has_bias ? 1u : 0u,
            .hash_mode = hash_mode ? 1u : 0u,
            .use_token_buffer = use_token_buffer ? 1u : 0u,
            .token = single_token ? (uint32_t)*single_token : 0u,
            .hash_rows = hash_rows,
        };

        const float zero_f32 = 0.0f;
        const int32_t zero_i32 = 0;
        if ((has_bias && !biasbuf) ||
            (hash_mode && !hashbuf) ||
            (use_token_buffer && !tokensbuf)) {
            return 0;
        }

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:router_finalize_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:probsbuf offset:probs_off atIndex:1];
        if (has_bias) {
            [enc setBuffer:biasbuf offset:bias_off atIndex:2];
        } else {
            [enc setBytes:&zero_f32 length:sizeof(zero_f32) atIndex:2];
        }
        if (hash_mode) {
            [enc setBuffer:hashbuf offset:hash_off atIndex:3];
        } else {
            [enc setBytes:&zero_i32 length:sizeof(zero_i32) atIndex:3];
        }
        if (use_token_buffer) {
            [enc setBuffer:tokensbuf offset:tokens_off atIndex:4];
        } else {
            [enc setBytes:&zero_i32 length:sizeof(zero_i32) atIndex:4];
        }
        [enc setBuffer:selectedbuf offset:selected_off atIndex:5];
        if (use_simd_weights_fusion) {
            [enc setBuffer:weightsbuf offset:weights_off atIndex:6];
        }
        const NSUInteger router_finalize_scratch_bytes = use_simd_finalize
            ? 2u * (256u * sizeof(float) + 256u * sizeof(int32_t))
            : 256u * sizeof(float) + 256u * sizeof(int32_t);
        [enc setThreadgroupMemoryLength:router_finalize_scratch_bytes atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (use_simd_weights_fusion) return 1;

        enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:router_weights_pipeline];
        [enc setBuffer:probsbuf offset:probs_off atIndex:0];
        [enc setBuffer:selectedbuf offset:selected_off atIndex:1];
        [enc setBuffer:weightsbuf offset:weights_off atIndex:2];
        [enc dispatchThreads:MTLSizeMake(6, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(6, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);
        return 1;
    }

    if (flash_router_fast_path && !g_quality_mode && n_tokens == 1) {
        id<MTLComputePipelineState> softplus_sqrt_pipeline =
            ds4_gpu_hot_pipeline(g_dsv4_softplus_sqrt_pipeline,
                                    "kernel_dsv4_softplus_sqrt_f32_4");
        ok = softplus_sqrt_pipeline &&
             ds4_gpu_encode_unary_f32_rows(cb,
                                             softplus_sqrt_pipeline,
                                             logitsbuf,
                                             logits_off,
                                             probsbuf,
                                             probs_off,
                                             n_expert,
                                             1,
                                             1,
                                             0.0f,
                                             0.0f);
    } else {
        ok = ds4_gpu_encode_unary_f32_rows(cb,
                                             g_unary_softplus_pipeline,
                                             logitsbuf,
                                             logits_off,
                                             probsbuf,
                                             probs_off,
                                             n_expert,
                                             n_tokens,
                                             1,
                                             0.0f,
                                             0.0f) &&
             ds4_gpu_encode_unary_f32_rows(cb,
                                             g_unary_sqrt_pipeline,
                                             probsbuf,
                                             probs_off,
                                             probsbuf,
                                             probs_off,
                                             n_expert,
                                             n_tokens,
                                             1,
                                             0.0f,
                                             0.0f);
    }
    if (!ok) return 0;

    if (hash_mode) {
        ok = ds4_gpu_encode_get_rows_i32_token_rows(cb,
                                                      hashbuf,
                                                      hash_off,
                                                      tokensbuf,
                                                      tokens_off,
                                                      single_token,
                                                      selectedbuf,
                                                      selected_off,
                                                      hash_rows,
                                                      n_expert_used,
                                                      n_tokens);
    } else {
        ds4_gpu_tensor *score_tensor = probs;
        DS4MetalTensor *selection_view = nil;

        if (has_bias) {
            if (!biasbuf ||
                !ds4_gpu_ensure_scratch_buffer(&g_router_selection_buffer,
                                                 &g_router_selection_bytes,
                                                 probs_bytes,
                                                 "ds4_router_selection")) {
                return 0;
            }

            ds4_gpu_bin_args add_args = ds4_gpu_make_bin_rows_args(n_expert, n_tokens, n_expert);
            ok = ds4_gpu_encode_bin_f32_rows(cb,
                                               g_add_pipeline,
                                               &add_args,
                                               probsbuf,
                                               probs_off,
                                               biasbuf,
                                               bias_off,
                                               g_router_selection_buffer,
                                               0);
            if (!ok) return 0;

            selection_view = [DS4MetalTensor new];
            selection_view.buffer = g_router_selection_buffer;
            selection_view.offset = 0;
            selection_view.bytes = probs_bytes;
            selection_view.owner = 0;
            score_tensor = (__bridge ds4_gpu_tensor *)selection_view;
        }

        ok = ds4_gpu_indexer_topk_tensor(selected, score_tensor, n_expert, n_tokens, n_expert_used) != 0;
    }
    if (!ok) return 0;

    const bool use_batch_weights_fusion =
        flash_router_fast_path && !g_quality_mode && n_tokens > 1u &&
        g_dsv4_router_weights_batch_pipeline != nil &&
        (ds4_gpu_device_name_contains("M3") ||
         getenv("DS4_METAL_ENABLE_ROUTER_WEIGHTS_BATCH_FUSION") != NULL) &&
        getenv("DS4_METAL_DISABLE_M3_ROUTER_WEIGHTS_BATCH_FUSION") == NULL &&
        getenv("DS4_METAL_DISABLE_ROUTER_SELECT_FUSION") == NULL;
    if (use_batch_weights_fusion) {
        const float scale = expert_weight_scale;
        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:g_dsv4_router_weights_batch_pipeline];
        [enc setBytes:&scale length:sizeof(scale) atIndex:0];
        [enc setBuffer:probsbuf offset:probs_off atIndex:1];
        [enc setBuffer:selectedbuf offset:selected_off atIndex:2];
        [enc setBuffer:weightsbuf offset:weights_off atIndex:3];
        [enc setThreadgroupMemoryLength:40u * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(n_tokens, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(6, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);
        return 1;
    }

    if (flash_router_fast_path && !g_quality_mode && n_tokens == 1) {
        id<MTLComputePipelineState> router_weights_pipeline =
            ds4_gpu_hot_pipeline(g_dsv4_router_weights_one_pipeline,
                                    "kernel_dsv4_router_weights_one");
        if (!router_weights_pipeline) return 0;
        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:router_weights_pipeline];
        [enc setBuffer:probsbuf offset:probs_off atIndex:0];
        [enc setBuffer:selectedbuf offset:selected_off atIndex:1];
        [enc setBuffer:weightsbuf offset:weights_off atIndex:2];
        [enc dispatchThreads:MTLSizeMake(6, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(6, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);
        return 1;
    }

    const NSUInteger sum_bytes = (NSUInteger)n_tokens * sizeof(float);
    if (!ds4_gpu_ensure_scratch_buffer(&g_router_weight_sum_buffer,
                                         &g_router_weight_sum_bytes,
                                         sum_bytes,
                                         "ds4_router_weight_sum")) {
        return 0;
    }

    ok = ds4_gpu_encode_get_rows_f32_router_weights(cb,
                                                      probsbuf,
                                                      probs_off,
                                                      selectedbuf,
                                                      selected_off,
                                                      weightsbuf,
                                                      weights_off,
                                                      n_expert,
                                                      n_expert_used,
                                                      n_tokens) &&
         ds4_gpu_encode_sum_rows_f32(cb,
                                       weightsbuf,
                                       weights_off,
                                       g_router_weight_sum_buffer,
                                       0,
                                       n_expert_used,
                                       n_tokens) &&
         ds4_gpu_encode_unary_f32_rows(cb,
                                         g_unary_clamp_pipeline,
                                         g_router_weight_sum_buffer,
                                         0,
                                         g_router_weight_sum_buffer,
                                         0,
                                         1,
                                         n_tokens,
                                         0,
                                         6.103515625e-5f,
                                         ds4_gpu_positive_infinity());
    if (!ok) return 0;

    ds4_gpu_bin_args div_args = ds4_gpu_make_bin_rowwise_scalar_args(n_expert_used, n_tokens);
    const float scale = expert_weight_scale;
    ds4_gpu_bin_args scale_args = ds4_gpu_make_bin_rows_args(n_expert_used, n_tokens, 1);

    ok = ds4_gpu_encode_bin_f32_rows(cb,
                                       g_bin_div_row_pipeline,
                                       &div_args,
                                       weightsbuf,
                                       weights_off,
                                       g_router_weight_sum_buffer,
                                       0,
                                       weightsbuf,
                                       weights_off);
    if (!ok) return 0;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:g_bin_mul_scalar_pipeline];
    [enc setBytes:&scale_args length:sizeof(scale_args) atIndex:0];
    [enc setBuffer:weightsbuf offset:weights_off atIndex:1];
    [enc setBytes:&scale length:sizeof(scale) atIndex:2];
    [enc setBuffer:weightsbuf offset:weights_off atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)scale_args.ne1,
                                          (NSUInteger)scale_args.ne2,
                                          (NSUInteger)scale_args.ne3)
         threadsPerThreadgroup:MTLSizeMake(ds4_gpu_bin_threads(n_expert_used, g_bin_mul_scalar_pipeline), 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);

    return 1;
}
