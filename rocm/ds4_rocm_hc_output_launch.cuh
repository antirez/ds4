extern "C" int ds4_gpu_repeat_hc_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *row, uint32_t n_embd, uint32_t n_hc) {
    uint64_t n = 0;
    if (n_embd == 0 || n_hc == 0 ||
        !cuda_u64_mul_checked(n_embd, n_hc, &n) ||
        !cuda_tensor_has_f32(row, n_embd) || !cuda_tensor_has_f32(out, n)) {
        return 0;
    }
    repeat_hc_kernel<<<(n + 255) / 256, 256>>>((float *)out->ptr, (const float *)row->ptr, n_embd, n_hc);
    return cuda_ok(cudaGetLastError(), "repeat_hc launch");
}

extern "C" int ds4_gpu_hc_split_sinkhorn_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *mix, const void *model_map, uint64_t model_size, uint64_t scale_offset, uint64_t base_offset, uint32_t n_hc, uint32_t sinkhorn_iters, float eps) {
    if (!out || !mix || !model_map || n_hc != 4) return 0;
    const uint64_t mix_bytes = 24ull * sizeof(float);
    if (!cuda_model_range_fits(model_size, scale_offset, 3ull * sizeof(float)) ||
        !cuda_model_range_fits(model_size, base_offset, mix_bytes) ||
        !cuda_tensor_has_bytes(mix, mix_bytes) || !cuda_tensor_has_bytes(out, mix_bytes)) return 0;
    const float *scale = (const float *)cuda_model_range_ptr(model_map, scale_offset, 3ull * sizeof(float), "hc_scale");
    const float *base = (const float *)cuda_model_range_ptr(model_map, base_offset, mix_bytes, "hc_base");
    if (!scale || !base) return 0;
    uint32_t n_rows = (uint32_t)(mix->bytes / mix_bytes);
    if (out->bytes / mix_bytes < n_rows) n_rows = (uint32_t)(out->bytes / mix_bytes);
    hc_split_sinkhorn_kernel<<<(n_rows + 255) / 256, 256>>>(
        (float *)out->ptr, (const float *)mix->ptr,
        scale,
        base,
        n_rows, sinkhorn_iters, eps);
    return cuda_ok(cudaGetLastError(), "hc_split_sinkhorn launch");
}
static int cuda_hc_flat_token_count(const ds4_gpu_tensor *out, uint32_t n_embd, uint64_t *n_tokens) {
    if (!out || n_embd == 0u || !n_tokens) return 0;
    uint64_t row_bytes = 0;
    if (!cuda_u64_mul3_checked(n_embd, 1u, sizeof(float), &row_bytes) ||
        row_bytes == 0u || out->bytes < row_bytes || (out->bytes % row_bytes) != 0u) return 0;
    *n_tokens = out->bytes / row_bytes;
    return *n_tokens != 0u && *n_tokens <= UINT32_MAX;
}

static int cuda_hc_hc_token_count(const ds4_gpu_tensor *out_hc, uint32_t n_embd, uint32_t n_hc, uint64_t *n_tokens) {
    if (!out_hc || n_embd == 0u || n_hc == 0u || !n_tokens) return 0;
    uint64_t row_elems = 0, row_bytes = 0;
    if (!cuda_u64_mul_checked(n_hc, n_embd, &row_elems) ||
        !cuda_u64_mul_checked(row_elems, sizeof(float), &row_bytes) ||
        row_bytes == 0u || out_hc->bytes < row_bytes || (out_hc->bytes % row_bytes) != 0u) return 0;
    *n_tokens = out_hc->bytes / row_bytes;
    return *n_tokens != 0u && *n_tokens <= UINT32_MAX;
}

static int cuda_hc_mix_width(uint32_t n_hc, uint64_t *mix_hc) {
    if (n_hc == 0u || !mix_hc) return 0;
    const uint64_t h = (uint64_t)n_hc;
    const uint64_t mix = 2ull * h + h * h;
    if (mix > UINT32_MAX) return 0;
    *mix_hc = mix;
    return 1;
}

extern "C" int ds4_gpu_hc_weighted_sum_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *residual_hc, const ds4_gpu_tensor *weights, uint32_t n_embd, uint32_t n_hc) {
    uint64_t n_tokens64 = 0, residual_bytes = 0, weights_bytes = 0;
    if (!out || !residual_hc || !weights || n_hc == 0u ||
        !cuda_hc_flat_token_count(out, n_embd, &n_tokens64) ||
        !cuda_u64_mul3_checked(n_tokens64, (uint64_t)n_hc * n_embd, sizeof(float), &residual_bytes) ||
        !cuda_u64_mul3_checked(n_tokens64, n_hc, sizeof(float), &weights_bytes) ||
        residual_hc->bytes < residual_bytes || weights->bytes < weights_bytes) return 0;
    uint32_t n_tokens = (uint32_t)n_tokens64;
    hc_weighted_sum_kernel<<<((uint64_t)n_embd * n_tokens + 255) / 256, 256>>>(
        (float *)out->ptr, (const float *)residual_hc->ptr, (const float *)weights->ptr,
        n_embd, n_hc, n_tokens, n_hc);
    return cuda_ok(cudaGetLastError(), "hc_weighted_sum launch");
}
extern "C" int ds4_gpu_hc_weighted_sum_split_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *residual_hc, const ds4_gpu_tensor *split, uint32_t n_embd, uint32_t n_hc) {
    uint64_t n_tokens64 = 0, residual_bytes = 0, split_bytes = 0, mix_hc = 0;
    if (!out || !residual_hc || !split ||
        !cuda_hc_flat_token_count(out, n_embd, &n_tokens64) ||
        !cuda_hc_mix_width(n_hc, &mix_hc) ||
        !cuda_u64_mul3_checked(n_tokens64, (uint64_t)n_hc * n_embd, sizeof(float), &residual_bytes) ||
        !cuda_u64_mul3_checked(n_tokens64, mix_hc, sizeof(float), &split_bytes) ||
        residual_hc->bytes < residual_bytes || split->bytes < split_bytes) return 0;
    uint32_t n_tokens = (uint32_t)n_tokens64;
    uint32_t stride = (uint32_t)mix_hc;
    hc_weighted_sum_kernel<<<((uint64_t)n_embd * n_tokens + 255) / 256, 256>>>(
        (float *)out->ptr, (const float *)residual_hc->ptr, (const float *)split->ptr,
        n_embd, n_hc, n_tokens, stride);
    return cuda_ok(cudaGetLastError(), "hc_weighted_sum_split launch");
}
extern "C" int ds4_gpu_hc_split_weighted_sum_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *split,
        const ds4_gpu_tensor *mix,
        const ds4_gpu_tensor *residual_hc,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                scale_offset,
        uint64_t                base_offset,
        uint32_t                n_embd,
        uint32_t                n_hc,
        uint32_t                sinkhorn_iters,
        float                   eps) {
    if (!out || !split || !mix || !residual_hc || !model_map ||
        n_embd == 0 || n_hc != 4) {
        return 0;
    }
    const uint64_t mix_hc = 2ull * n_hc + (uint64_t)n_hc * n_hc;
    const uint64_t mix_bytes = mix_hc * sizeof(float);
    const uint64_t out_row_bytes = (uint64_t)n_embd * sizeof(float);
    const uint64_t residual_row_bytes = (uint64_t)n_hc * n_embd * sizeof(float);
    if (out->bytes < out_row_bytes || out->bytes % out_row_bytes != 0 ||
        scale_offset > model_size || 3ull * sizeof(float) > model_size - scale_offset ||
        base_offset > model_size || mix_bytes > model_size - base_offset) {
        return 0;
    }
    uint64_t n_rows = out->bytes / out_row_bytes;
    if (mix->bytes < n_rows * mix_bytes ||
        split->bytes < n_rows * mix_bytes ||
        residual_hc->bytes < n_rows * residual_row_bytes) {
        return 0;
    }
    const float *scale = (const float *)cuda_model_range_ptr(model_map, scale_offset, 3ull * sizeof(float), "hc_scale");
    const float *base = (const float *)cuda_model_range_ptr(model_map, base_offset, mix_bytes, "hc_base");
    if (!scale || !base) return 0;
    hc_split_weighted_sum_fused_kernel<<<(uint32_t)n_rows, 256>>>(
            (float *)out->ptr,
            (float *)split->ptr,
            (const float *)mix->ptr,
            (const float *)residual_hc->ptr,
            scale,
            base,
            n_embd, n_hc, (uint32_t)n_rows, sinkhorn_iters, eps);
    return cuda_ok(cudaGetLastError(), "hc split weighted sum launch");
}
extern "C" int ds4_gpu_hc_split_weighted_sum_norm_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *norm_out,
        ds4_gpu_tensor       *split,
        const ds4_gpu_tensor *mix,
        const ds4_gpu_tensor *residual_hc,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                scale_offset,
        uint64_t                base_offset,
        uint64_t                norm_weight_offset,
        uint32_t                n_embd,
        uint32_t                n_hc,
        uint32_t                sinkhorn_iters,
        float                   eps,
        float                   norm_eps) {
    if (!out || !norm_out || !split || !mix || !residual_hc || !model_map ||
        n_embd == 0 || n_hc != 4) {
        return 0;
    }
    const uint64_t mix_hc = 2ull * n_hc + (uint64_t)n_hc * n_hc;
    const uint64_t mix_bytes = mix_hc * sizeof(float);
    const uint64_t out_row_bytes = (uint64_t)n_embd * sizeof(float);
    const uint64_t residual_row_bytes = (uint64_t)n_hc * n_embd * sizeof(float);
    if (out->bytes < out_row_bytes || out->bytes % out_row_bytes != 0 ||
        norm_out->bytes < out->bytes ||
        scale_offset > model_size || 3ull * sizeof(float) > model_size - scale_offset ||
        base_offset > model_size || mix_bytes > model_size - base_offset ||
        norm_weight_offset > model_size ||
        (uint64_t)n_embd * sizeof(float) > model_size - norm_weight_offset) {
        return 0;
    }
    uint64_t n_rows = out->bytes / out_row_bytes;
    if (n_rows == 1) {
        if (mix->bytes < n_rows * mix_bytes ||
            split->bytes < n_rows * mix_bytes ||
            residual_hc->bytes < n_rows * residual_row_bytes) {
            return 0;
        }
        const float *scale = (const float *)cuda_model_range_ptr(model_map, scale_offset,
                3ull * sizeof(float), "hc_scale");
        const float *base = (const float *)cuda_model_range_ptr(model_map, base_offset,
                mix_bytes, "hc_base");
        const float *norm_w = (const float *)cuda_model_range_ptr(model_map, norm_weight_offset,
                (uint64_t)n_embd * sizeof(float), "hc_norm_weight");
        if (!scale || !base || !norm_w) return 0;
        hc_split_weighted_sum_norm_fused_kernel<<<(uint32_t)n_rows, 256>>>(
                (float *)out->ptr,
                (float *)norm_out->ptr,
                (float *)split->ptr,
                (const float *)mix->ptr,
                (const float *)residual_hc->ptr,
                scale,
                base,
                norm_w,
                n_embd, n_hc, (uint32_t)n_rows, sinkhorn_iters, eps, norm_eps);
        return cuda_ok(cudaGetLastError(), "hc split weighted sum norm launch");
    }
    return ds4_gpu_hc_split_weighted_sum_tensor(out, split, mix, residual_hc,
                                                  model_map, model_size,
                                                  scale_offset, base_offset,
                                                  n_embd, n_hc,
                                                  sinkhorn_iters, eps) &&
           ds4_gpu_rms_norm_weight_tensor(norm_out, out, model_map, model_size,
                                            norm_weight_offset, n_embd, norm_eps);
}
extern "C" int ds4_gpu_output_hc_weights_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *pre,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                scale_offset,
        uint64_t                base_offset,
        uint32_t                n_hc,
        float                   eps) {
    if (!out || !pre || !model_map || n_hc == 0) return 0;
    const uint64_t row_bytes = (uint64_t)n_hc * sizeof(float);
    if (row_bytes == 0 || out->bytes < row_bytes || out->bytes % row_bytes != 0 ||
        pre->bytes < out->bytes ||
        scale_offset > model_size || sizeof(float) > model_size - scale_offset ||
        base_offset > model_size || row_bytes > model_size - base_offset) {
        return 0;
    }
    const uint64_t n_tokens = out->bytes / row_bytes;
    const float *scale = (const float *)cuda_model_range_ptr(model_map, scale_offset, sizeof(float), "output_hc_scale");
    const float *base = (const float *)cuda_model_range_ptr(model_map, base_offset, row_bytes, "output_hc_base");
    if (!scale || !base) return 0;
    uint64_t n = n_tokens * n_hc;
    output_hc_weights_kernel<<<(n + 255) / 256, 256>>>(
            (float *)out->ptr,
            (const float *)pre->ptr,
            scale,
            base,
            n_hc,
            (uint32_t)n_tokens,
            eps);
    return cuda_ok(cudaGetLastError(), "output hc weights launch");
}
extern "C" int ds4_gpu_hc_expand_tensor(ds4_gpu_tensor *out_hc, const ds4_gpu_tensor *block_out, const ds4_gpu_tensor *residual_hc, const ds4_gpu_tensor *post, const ds4_gpu_tensor *comb, uint32_t n_embd, uint32_t n_hc) {
    uint64_t n_tokens64 = 0, flat_bytes = 0, hc_bytes = 0, post_bytes = 0, comb_bytes = 0, comb_stride = 0;
    if (!out_hc || !block_out || !residual_hc || !post || !comb ||
        !cuda_hc_hc_token_count(out_hc, n_embd, n_hc, &n_tokens64) ||
        !cuda_u64_mul3_checked(n_tokens64, n_embd, sizeof(float), &flat_bytes) ||
        !cuda_u64_mul3_checked(n_tokens64, (uint64_t)n_hc * n_embd, sizeof(float), &hc_bytes) ||
        !cuda_u64_mul3_checked(n_tokens64, n_hc, sizeof(float), &post_bytes) ||
        !cuda_u64_mul_checked(n_hc, n_hc, &comb_stride) || comb_stride > UINT32_MAX ||
        !cuda_u64_mul3_checked(n_tokens64, comb_stride, sizeof(float), &comb_bytes) ||
        block_out->bytes < flat_bytes || residual_hc->bytes < hc_bytes ||
        post->bytes < post_bytes || comb->bytes < comb_bytes) return 0;
    uint32_t n_tokens = (uint32_t)n_tokens64;
    uint64_t n_elem = (uint64_t)n_tokens * n_hc * n_embd;
    hc_expand_kernel<<<(n_elem + 255) / 256, 256>>>((float *)out_hc->ptr,
                                                    (const float *)block_out->ptr,
                                                    (const float *)block_out->ptr,
                                                    (const float *)residual_hc->ptr,
                                                    (const float *)post->ptr,
                                                    (const float *)comb->ptr,
                                                    n_embd, n_hc, n_tokens,
                                                    n_hc, (uint32_t)comb_stride, 0);
    return cuda_ok(cudaGetLastError(), "hc_expand launch");
}
extern "C" int ds4_gpu_hc_expand_split_tensor(ds4_gpu_tensor *out_hc, const ds4_gpu_tensor *block_out, const ds4_gpu_tensor *residual_hc, const ds4_gpu_tensor *split, uint32_t n_embd, uint32_t n_hc) {
    uint64_t n_tokens64 = 0, flat_bytes = 0, hc_bytes = 0, split_bytes = 0, mix_hc64 = 0;
    if (!out_hc || !block_out || !residual_hc || !split ||
        !cuda_hc_hc_token_count(out_hc, n_embd, n_hc, &n_tokens64) ||
        !cuda_hc_mix_width(n_hc, &mix_hc64) ||
        !cuda_u64_mul3_checked(n_tokens64, n_embd, sizeof(float), &flat_bytes) ||
        !cuda_u64_mul3_checked(n_tokens64, (uint64_t)n_hc * n_embd, sizeof(float), &hc_bytes) ||
        !cuda_u64_mul3_checked(n_tokens64, mix_hc64, sizeof(float), &split_bytes) ||
        block_out->bytes < flat_bytes || residual_hc->bytes < hc_bytes || split->bytes < split_bytes) return 0;
    uint32_t n_tokens = (uint32_t)n_tokens64;
    if (n_hc == 4u) {
        const uint64_t n = (uint64_t)n_tokens * n_embd;
        hc_expand4_kernel<<<(n + 255) / 256, 256>>>((float *)out_hc->ptr,
                                                    (const float *)block_out->ptr,
                                                    (const float *)residual_hc->ptr,
                                                    (const float *)split->ptr,
                                                    n_embd,
                                                    n_tokens);
        return cuda_ok(cudaGetLastError(), "hc_expand_split4 launch");
    }
    uint32_t mix_hc = (uint32_t)mix_hc64;
    uint64_t n_elem = (uint64_t)n_tokens * n_hc * n_embd;
    const float *base = (const float *)split->ptr;
    hc_expand_kernel<<<(n_elem + 255) / 256, 256>>>((float *)out_hc->ptr,
                                                    (const float *)block_out->ptr,
                                                    (const float *)block_out->ptr,
                                                    (const float *)residual_hc->ptr,
                                                    base + n_hc,
                                                    base + 2u * n_hc,
                                                    n_embd, n_hc, n_tokens,
                                                    mix_hc, mix_hc, 0);
    return cuda_ok(cudaGetLastError(), "hc_expand_split launch");
}
extern "C" int ds4_gpu_hc_expand_split_half_tensor(ds4_gpu_tensor *out_hc, const ds4_gpu_tensor *block_out_h, const ds4_gpu_tensor *residual_hc, const ds4_gpu_tensor *split, uint32_t n_embd, uint32_t n_hc) {
    uint64_t n_tokens64 = 0, flat_half_bytes = 0, hc_bytes = 0, split_bytes = 0, mix_hc64 = 0;
    if (!out_hc || !block_out_h || !residual_hc || !split ||
        !cuda_hc_hc_token_count(out_hc, n_embd, n_hc, &n_tokens64) ||
        !cuda_hc_mix_width(n_hc, &mix_hc64) ||
        !cuda_u64_mul3_checked(n_tokens64, n_embd, sizeof(__half), &flat_half_bytes) ||
        !cuda_u64_mul3_checked(n_tokens64, (uint64_t)n_hc * n_embd, sizeof(float), &hc_bytes) ||
        !cuda_u64_mul3_checked(n_tokens64, mix_hc64, sizeof(float), &split_bytes) ||
        block_out_h->bytes < flat_half_bytes || residual_hc->bytes < hc_bytes || split->bytes < split_bytes) return 0;
    uint32_t n_tokens = (uint32_t)n_tokens64;
    if (n_hc == 4u) {
        const uint64_t n = (uint64_t)n_tokens * n_embd;
        hc_expand4_half_kernel<<<(n + 255) / 256, 256>>>((float *)out_hc->ptr,
                                                         (const __half *)block_out_h->ptr,
                                                         (const float *)residual_hc->ptr,
                                                         (const float *)split->ptr,
                                                         n_embd,
                                                         n_tokens);
        return cuda_ok(cudaGetLastError(), "hc_expand_split_half4 launch");
    }
    uint32_t mix_hc = (uint32_t)mix_hc64;
    uint64_t n_elem = (uint64_t)n_tokens * n_hc * n_embd;
    const float *base = (const float *)split->ptr;
    hc_expand_half_kernel<<<(n_elem + 255) / 256, 256>>>((float *)out_hc->ptr,
                                                         (const __half *)block_out_h->ptr,
                                                         (const float *)residual_hc->ptr,
                                                         base + n_hc,
                                                         base + 2u * n_hc,
                                                         n_embd, n_hc, n_tokens,
                                                         mix_hc, mix_hc);
    return cuda_ok(cudaGetLastError(), "hc_expand_split_half launch");
}
extern "C" int ds4_gpu_hc_expand_add_split_tensor(ds4_gpu_tensor *out_hc, const ds4_gpu_tensor *block_out, const ds4_gpu_tensor *block_add, const ds4_gpu_tensor *residual_hc, const ds4_gpu_tensor *split, uint32_t n_embd, uint32_t n_hc) {
    uint64_t n_tokens64 = 0, flat_bytes = 0, hc_bytes = 0, split_bytes = 0, mix_hc64 = 0;
    if (!out_hc || !block_out || !block_add || !residual_hc || !split ||
        !cuda_hc_hc_token_count(out_hc, n_embd, n_hc, &n_tokens64) ||
        !cuda_hc_mix_width(n_hc, &mix_hc64) ||
        !cuda_u64_mul3_checked(n_tokens64, n_embd, sizeof(float), &flat_bytes) ||
        !cuda_u64_mul3_checked(n_tokens64, (uint64_t)n_hc * n_embd, sizeof(float), &hc_bytes) ||
        !cuda_u64_mul3_checked(n_tokens64, mix_hc64, sizeof(float), &split_bytes) ||
        block_out->bytes < flat_bytes || block_add->bytes < flat_bytes ||
        residual_hc->bytes < hc_bytes || split->bytes < split_bytes) return 0;
    uint32_t n_tokens = (uint32_t)n_tokens64;
    if (n_hc == 4u) {
        const uint64_t n = (uint64_t)n_tokens * n_embd;
        hc_expand4_add_kernel<<<(n + 255) / 256, 256>>>((float *)out_hc->ptr,
                                                        (const float *)block_out->ptr,
                                                        (const float *)block_add->ptr,
                                                        (const float *)residual_hc->ptr,
                                                        (const float *)split->ptr,
                                                        n_embd,
                                                        n_tokens);
        return cuda_ok(cudaGetLastError(), "hc_expand_add_split4 launch");
    }
    uint32_t mix_hc = (uint32_t)mix_hc64;
    uint64_t n_elem = (uint64_t)n_tokens * n_hc * n_embd;
    const float *base = (const float *)split->ptr;
    hc_expand_kernel<<<(n_elem + 255) / 256, 256>>>((float *)out_hc->ptr,
                                                    (const float *)block_out->ptr,
                                                    (const float *)block_add->ptr,
                                                    (const float *)residual_hc->ptr,
                                                    base + n_hc,
                                                    base + 2u * n_hc,
                                                    n_embd, n_hc, n_tokens,
                                                    mix_hc, mix_hc, 1);
    return cuda_ok(cudaGetLastError(), "hc_expand_add_split launch");
}
extern "C" int ds4_gpu_hc_expand_add_split_half_add_tensor(ds4_gpu_tensor *out_hc, const ds4_gpu_tensor *block_out, const ds4_gpu_tensor *block_add_h, const ds4_gpu_tensor *residual_hc, const ds4_gpu_tensor *split, uint32_t n_embd, uint32_t n_hc) {
    uint64_t n_tokens64 = 0, flat_bytes = 0, flat_half_bytes = 0, hc_bytes = 0, split_bytes = 0, mix_hc64 = 0;
    if (!out_hc || !block_out || !block_add_h || !residual_hc || !split ||
        !cuda_hc_hc_token_count(out_hc, n_embd, n_hc, &n_tokens64) ||
        !cuda_hc_mix_width(n_hc, &mix_hc64) ||
        !cuda_u64_mul3_checked(n_tokens64, n_embd, sizeof(float), &flat_bytes) ||
        !cuda_u64_mul3_checked(n_tokens64, n_embd, sizeof(__half), &flat_half_bytes) ||
        !cuda_u64_mul3_checked(n_tokens64, (uint64_t)n_hc * n_embd, sizeof(float), &hc_bytes) ||
        !cuda_u64_mul3_checked(n_tokens64, mix_hc64, sizeof(float), &split_bytes) ||
        block_out->bytes < flat_bytes || block_add_h->bytes < flat_half_bytes ||
        residual_hc->bytes < hc_bytes || split->bytes < split_bytes) return 0;
    uint32_t n_tokens = (uint32_t)n_tokens64;
    if (n_hc == 4u) {
        const uint64_t n = (uint64_t)n_tokens * n_embd;
        hc_expand4_add_half_kernel<<<(n + 255) / 256, 256>>>((float *)out_hc->ptr,
                                                             (const float *)block_out->ptr,
                                                             (const __half *)block_add_h->ptr,
                                                             (const float *)residual_hc->ptr,
                                                             (const float *)split->ptr,
                                                             n_embd,
                                                             n_tokens);
        return cuda_ok(cudaGetLastError(), "hc_expand_add_split_half4 launch");
    }
    uint32_t mix_hc = (uint32_t)mix_hc64;
    uint64_t n_elem = (uint64_t)n_tokens * n_hc * n_embd;
    const float *base = (const float *)split->ptr;
    hc_expand_add_half_kernel<<<(n_elem + 255) / 256, 256>>>((float *)out_hc->ptr,
                                                             (const float *)block_out->ptr,
                                                             (const __half *)block_add_h->ptr,
                                                             (const float *)residual_hc->ptr,
                                                             base + n_hc,
                                                             base + 2u * n_hc,
                                                             n_embd, n_hc, n_tokens,
                                                             mix_hc, mix_hc);
    return cuda_ok(cudaGetLastError(), "hc_expand_add_split_half_add launch");
}
extern "C" int ds4_gpu_shared_down_hc_expand_q8_0_tensor(
        ds4_gpu_tensor       *out_hc,
        ds4_gpu_tensor       *shared_out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *shared_mid,
        const ds4_gpu_tensor *routed_out,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc) {
    return cuda_matmul_q8_0_hc_expand_tensor_labeled(out_hc, shared_out,
                                                    model_map, model_size,
                                                    weight_offset,
                                                    in_dim, out_dim,
                                                    shared_mid,
                                                    routed_out,
                                                    residual_hc,
                                                    split,
                                                    n_embd, n_hc,
                                                    "shared_down_hc_expand");
}

extern "C" int ds4_gpu_matmul_q8_0_hc_expand_tensor(
        ds4_gpu_tensor       *out_hc,
        ds4_gpu_tensor       *block_out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc) {
    return cuda_matmul_q8_0_hc_expand_tensor_labeled(out_hc, block_out,
                                                    model_map, model_size,
                                                    weight_offset,
                                                    in_dim, out_dim,
                                                    x,
                                                    NULL,
                                                    residual_hc,
                                                    split,
                                                    n_embd, n_hc,
                                                    "q8_hc_expand");
}
