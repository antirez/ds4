extern "C" int ds4_gpu_swiglu_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *gate, const ds4_gpu_tensor *up, uint32_t n, float clamp, float weight) {
    if (!cuda_tensor_has_f32(out, n) || !cuda_tensor_has_f32(gate, n) || !cuda_tensor_has_f32(up, n)) return 0;
    if (n == 0u) return 1;
    swiglu_kernel<<<(n + 255) / 256, 256>>>((float *)out->ptr, (const float *)gate->ptr, (const float *)up->ptr, n, clamp, weight);
    return cuda_ok(cudaGetLastError(), "swiglu launch");
}
extern "C" int ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(
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
    if (!gate || !up || !mid || !model_map || !x ||
        in_dim == 0u || out_dim == 0u || in_dim > UINT32_MAX || out_dim > UINT32_MAX) {
        return 0;
    }
    const uint64_t blocks = (in_dim + 31u) / 32u;
    uint64_t row_bytes = 0;
    uint64_t weight_bytes = 0;
    uint64_t x_bytes = 0;
    uint64_t out_bytes = 0;
    if (!cuda_u64_mul_checked(blocks, 34u, &row_bytes) ||
        !cuda_u64_mul_checked(out_dim, row_bytes, &weight_bytes) ||
        !cuda_u64_mul3_checked(in_dim, 1u, sizeof(float), &x_bytes) ||
        !cuda_u64_mul3_checked(out_dim, 1u, sizeof(float), &out_bytes) ||
        !cuda_tensor_has_bytes(x, x_bytes) || !cuda_tensor_has_bytes(gate, out_bytes) ||
        !cuda_tensor_has_bytes(up, out_bytes) || !cuda_tensor_has_bytes(mid, out_bytes)) {
        return 0;
    }
    if (in_dim == 4096u && (in_dim & 31u) == 0u &&
        cuda_model_range_fits(model_size, gate_offset, weight_bytes) &&
        cuda_model_range_fits(model_size, up_offset, weight_bytes) &&
        !cuda_runtime_config()->disable_shared_gate_up_fused_w32) {
        const char *wg = cuda_model_range_ptr(model_map, gate_offset, weight_bytes, "shared_gate_q8");
        const char *wu = cuda_model_range_ptr(model_map, up_offset, weight_bytes, "shared_up_q8");
        if (!wg || !wu) return 0;
        const int store_gate_up = (g_quality_mode || cuda_runtime_config()->graph_dump) ? 1 : 0;
        const unsigned rows_per_block = 32u;
        shared_gate_up_swiglu_q8_0_rows_w32_kernel<<<
                (unsigned)((out_dim + rows_per_block - 1u) / rows_per_block),
                rows_per_block * 32u>>>(
                (float *)gate->ptr,
                (float *)up->ptr,
                (float *)mid->ptr,
                reinterpret_cast<const unsigned char *>(wg),
                reinterpret_cast<const unsigned char *>(wu),
                (const float *)x->ptr,
                (uint32_t)blocks,
                out_dim,
                row_bytes,
                store_gate_up,
                clamp);
        return cuda_ok(cudaGetLastError(), "shared gate/up fused q8 launch");
    }
    return ds4_gpu_matmul_q8_0_pair_tensor(gate, up,
                                             model_map, model_size,
                                             gate_offset, up_offset,
                                             in_dim, out_dim, out_dim,
                                             x, 1) &&
           ds4_gpu_swiglu_tensor(mid, gate, up, (uint32_t)out_dim, clamp, 1.0f);
}

extern "C" int ds4_gpu_shared_gate_up_swiglu_q8_0_rows_scalar_tensor(
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
    (void)gate; (void)up; (void)mid; (void)model_map; (void)model_size;
    (void)gate_offset; (void)up_offset; (void)in_dim; (void)out_dim;
    (void)x; (void)n_tok; (void)clamp;
    return 0;
}

extern "C" int ds4_gpu_shared_gate_up_swiglu_q8_0_batch_tensor(
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
        uint64_t                n_tok);

extern "C" int ds4_gpu_shared_mid_swiglu_q8_0_tensor(
        ds4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        float                   clamp) {
    if (!mid || out_dim == 0u || out_dim > UINT32_MAX) return 0;
    const uint64_t blocks = (in_dim + 31u) / 32u;
    uint64_t xq_bytes = 0;
    uint64_t scale_bytes = 0;
    uint64_t outputs_bytes = 0;
    if (!cuda_u64_mul_checked(blocks, 32u, &xq_bytes) ||
        !cuda_u64_mul_checked(blocks, sizeof(float), &scale_bytes) ||
        !cuda_u64_mul3_checked(2u, out_dim, sizeof(float), &outputs_bytes)) {
        return 0;
    }
    if (xq_bytes > UINT64_MAX - 15u) return 0;
    const uint64_t scale_offset = (xq_bytes + 15u) & ~15ull;
    if (scale_offset > UINT64_MAX - scale_bytes) return 0;
    const uint64_t nested_scratch_bytes =
        (scale_offset + scale_bytes + 15u) & ~15ull;
    if (nested_scratch_bytes > UINT64_MAX - outputs_bytes) return 0;
    const uint64_t tmp_bytes = nested_scratch_bytes + outputs_bytes;
    void *tmp = cuda_tmp_alloc(tmp_bytes, "shared gate/up mid wrapper");
    if (!tmp) return 0;
    ds4_gpu_tensor gate_tmp = {
        (char *)tmp + nested_scratch_bytes, out_dim * sizeof(float), 0
    };
    ds4_gpu_tensor up_tmp = { (char *)gate_tmp.ptr + out_dim * sizeof(float),
                              out_dim * sizeof(float),
                              0 };
    return ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(&gate_tmp,
                                                     &up_tmp,
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

extern "C" int ds4_gpu_shared_gate_up_swiglu_q8_0_model_view_tensor(
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

extern "C" int ds4_gpu_shared_gate_up_swiglu_q8_0_rows_tensor(
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
    if (n_tok == 1u) {
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
    if (clamp > 1.0e-6f) return 0;
    return ds4_gpu_shared_gate_up_swiglu_q8_0_batch_tensor(gate,
                                                           up,
                                                           mid,
                                                           model_map,
                                                           model_size,
                                                           gate_offset,
                                                           up_offset,
                                                           in_dim,
                                                           out_dim,
                                                           x,
                                                           n_tok);
}

extern "C" int ds4_gpu_shared_gate_up_swiglu_q8_0_batch_tensor(
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
        uint64_t                n_tok) {
    uint64_t x_bytes = 0, out_bytes = 0;
    if (!gate || !up || !mid || !model_map || !x || n_tok == 0 ||
        (in_dim & 31u) != 0u || in_dim == 0u || out_dim == 0u ||
        in_dim > UINT32_MAX || out_dim > UINT32_MAX || n_tok > UINT32_MAX ||
        !cuda_u64_mul3_checked(n_tok, in_dim, sizeof(float), &x_bytes) ||
        !cuda_u64_mul3_checked(n_tok, out_dim, sizeof(float), &out_bytes) ||
        x->bytes < x_bytes || gate->bytes < out_bytes || up->bytes < out_bytes || mid->bytes < out_bytes) {
        return 0;
    }
    const uint64_t blocks = (in_dim + 31u) / 32u;
    uint64_t row_bytes = 0, weight_bytes = 0;
    if (!cuda_u64_mul_checked(blocks, 34u, &row_bytes) ||
        !cuda_u64_mul_checked(out_dim, row_bytes, &weight_bytes)) return 0;
    if (gate_offset > model_size || up_offset > model_size ||
        weight_bytes > model_size - gate_offset || weight_bytes > model_size - up_offset) {
        return 0;
    }
    const char *wg = cuda_model_range_ptr(model_map, gate_offset, weight_bytes, "shared_gate_q8_batch");
    const char *wu = cuda_model_range_ptr(model_map, up_offset, weight_bytes, "shared_up_q8_batch");
    if (!wg || !wu) return 0;

    const uint32_t rows_per_block = 32u;
    const uint32_t tile = 16u;
    const uint32_t block_tile = 16u;
    const dim3 grid((uint32_t)((out_dim + rows_per_block - 1u) / rows_per_block),
                    (uint32_t)((n_tok + tile - 1u) / tile),
                    1u);
    const size_t shmem = (size_t)tile * block_tile * 32u * sizeof(float);
    const int store_gate_up = (g_quality_mode || cuda_runtime_config()->graph_dump) ? 1 : 0;
#define DS4_LAUNCH_SHARED_GU_BATCH(TT, BT) \
    shared_gate_up_swiglu_q8_0_batch_sharedx_w32_kernel<TT, BT><<<grid, rows_per_block * 32u, shmem>>>( \
            (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr, \
            reinterpret_cast<const unsigned char *>(wg), reinterpret_cast<const unsigned char *>(wu), \
            (const float *)x->ptr, (uint32_t)blocks, (uint32_t)out_dim, (uint32_t)n_tok, row_bytes, store_gate_up)
    DS4_LAUNCH_SHARED_GU_BATCH(16u, 16u);
#undef DS4_LAUNCH_SHARED_GU_BATCH
    return cuda_ok(cudaGetLastError(), "shared gate/up fused q8 batch launch");
}
