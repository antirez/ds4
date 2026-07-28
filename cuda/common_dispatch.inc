/* GLM opt-in: batched q8_0 matmuls with blocks > 32 may run as a
 * streaming dequant-to-f16 GEMM (exact-q8 native kernels only cover
 * blocks <= 32). Never enabled on DS4 paths, keeping them byte-stable. */
static int g_q8_dequant_gemm_enabled = 0;
extern "C" void ds4_gpu_enable_q8_dequant_gemm(void) {
    g_q8_dequant_gemm_enabled = 1;
}

__global__ static void q8_0_dequant_f16_kernel(
        __half *out,
        const unsigned char *w,
        uint64_t total_blocks,
        uint32_t blocks_per_row,
        uint32_t in_dim) {
    /* Two threads per q8_0 block; each converts 16 values with half2
     * stores so a warp writes 512B contiguously per block pair. */
    const uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t b = tid >> 1;
    if (b >= total_blocks) return;
    const uint32_t half_idx = (uint32_t)tid & 1u;
    const unsigned char *blk = w + b * 34u;
    const float d = __half2float(*(const __half *)blk);
    const int8_t *q = (const int8_t *)(blk + 2) + half_idx * 16u;
    const uint64_t row = b / blocks_per_row;
    const uint32_t col = (uint32_t)(b - row * blocks_per_row) * 32u +
                         half_idx * 16u;
    __half2 *dst = (__half2 *)(out + row * in_dim + col);
    #pragma unroll
    for (int k = 0; k < 8; k++) {
        dst[k] = __floats2half2_rn(d * (float)q[2 * k],
                                   d * (float)q[2 * k + 1]);
    }
}

static int cuda_matmul_q8_0_tensor_labeled(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim, const ds4_gpu_tensor *x, uint64_t n_tok, const char *label) {
    if (!out || !x || !model_map) return 0;
    uint64_t blocks = (in_dim + 31) / 32;
    if (weight_offset > model_size || out_dim > UINT64_MAX / (blocks * 34)) return 0;
    uint64_t weight_bytes = out_dim * blocks * 34;
    if (weight_bytes > model_size - weight_offset) return 0;
    if (x->bytes < n_tok * in_dim * sizeof(float) ||
        out->bytes < n_tok * out_dim * sizeof(float)) return 0;
    const int logical_tier = ds4_tensor_device_idx(out);
    const int physical_device =
        (g_n_gpus > 1 && logical_tier >= 0 && logical_tier < g_n_gpus)
            ? g_gpu[logical_tier].device_id : 0;
    const char *wptr = cuda_resolve_weight_ptr(model_map, weight_offset, weight_bytes, logical_tier, "q8_0");
    if (!wptr) return 0;
    if (g_cublas_ready && n_tok > 1) {
        const float *w_f32 = cuda_q8_f32_ptr(model_map, weight_offset, weight_bytes, in_dim, out_dim, physical_device, label);
        if (w_f32) {
            const float alpha = 1.0f;
            const float beta = 0.0f;
            cublasStatus_t st = cublasSgemm(cuda_cublas_for_tier(logical_tier),
                                            CUBLAS_OP_T,
                                            CUBLAS_OP_N,
                                            (int)out_dim,
                                            (int)n_tok,
                                            (int)in_dim,
                                            &alpha,
                                            w_f32,
                                            (int)in_dim,
                                            (const float *)x->ptr,
                                            (int)in_dim,
                                            &beta,
                                            (float *)out->ptr,
                                            (int)out_dim);
            return cublas_ok(st, "q8 fp32 matmul");
        }
        const __half *w_f16 = cuda_q8_f16_ptr(model_map, weight_offset, weight_bytes, in_dim, out_dim, physical_device, label);
        if (w_f16) {
            const uint64_t xh_count = n_tok * in_dim;
            __half *xh = (__half *)cuda_tmp_alloc_on(logical_tier, xh_count * sizeof(__half), "q8 f16 gemm activations");
            if (!xh) return 0;
            f32_to_f16_kernel<<<(xh_count + 255) / 256, 256>>>(xh, (const float *)x->ptr, xh_count);
            if (!cuda_ok(cudaGetLastError(), "q8 f16 activation convert launch")) return 0;
            const float alpha = 1.0f;
            const float beta = 0.0f;
            cublasStatus_t st = cublasGemmEx(cuda_cublas_for_tier(logical_tier),
                                             CUBLAS_OP_T,
                                             CUBLAS_OP_N,
                                             (int)out_dim,
                                             (int)n_tok,
                                             (int)in_dim,
                                             &alpha,
                                             w_f16,
                                             CUDA_R_16F,
                                             (int)in_dim,
                                             xh,
                                             CUDA_R_16F,
                                             (int)in_dim,
                                             &beta,
                                             out->ptr,
                                             CUDA_R_32F,
                                             (int)out_dim,
                                             CUDA_R_32F,
                                             CUBLAS_GEMM_DEFAULT);
            if (st == CUBLAS_STATUS_SUCCESS) return 1;
            fprintf(stderr, "ds4: cuBLAS q8 f16 matmul failed: status %d\n", (int)st);
            cuda_q8_f16_cache_disable_after_failure("cuBLAS f16 matmul failure",
                                                    in_dim * out_dim * sizeof(__half));
            /* The F16 expansion cache is only an optimization.  If cuBLAS
             * rejects the cached path under memory pressure, retry the same
             * operation through the native Q8 kernels below. */
        }
    }
    if (g_q8_dequant_gemm_enabled && g_cublas_ready &&
        n_tok >= 128u && blocks > 32u && (in_dim & 31u) == 0u) {
        /* Streaming dequant + f16 GEMM: the exact-q8 batched kernels only
         * cover blocks <= 32 (DS4 TP shard widths); the per-token fallback
         * re-reads the full weight per token (~30x the bytes at GLM dims).
         * Scratch layout: [w_f16][x_f16] in one arena grab. */
        const uint64_t wh_bytes = in_dim * out_dim * sizeof(__half);
        const uint64_t xh_off = (wh_bytes + 255u) & ~255ull;
        const uint64_t oo_off =
            (xh_off + n_tok * in_dim * sizeof(__half) + 255u) & ~255ull;
        const uint64_t gemm_tmp = oo_off + n_tok * out_dim * sizeof(float);
        /* Scratch must live on the EXECUTING device: logical_tier is the
         * out tensor's tier (0 for GLM graph buffers), and a GEMM reading
         * its staged weights across PCIe costs ~20ms instead of ~0.1ms. */
        int exec_tier = logical_tier;
        {
            int cur_dev = -1;
            if (cudaGetDevice(&cur_dev) == cudaSuccess) {
                for (int t = 0; t < g_n_gpus; t++) {
                    if (g_gpu[t].device_id == cur_dev) { exec_tier = t; break; }
                }
            }
        }
        void *tmp16 = cuda_tmp_alloc_on(exec_tier, gemm_tmp, "q8 dequant gemm");
        if (tmp16) {
            __half *wh = (__half *)tmp16;
            __half *xh = (__half *)((char *)tmp16 + xh_off);
            /* GEMM into device-local scratch, then one bulk D2D to the
             * (possibly peer-mapped) out tensor: scattered peer stores
             * from GEMM kernels run at <1GB/s over PCIe. */
            float *olocal = (float *)((char *)tmp16 + oo_off);
            const uint64_t total_blocks = out_dim * blocks;
            q8_0_dequant_f16_kernel<<<(unsigned)((total_blocks * 2u + 255u) / 256u), 256>>>(
                    wh, reinterpret_cast<const unsigned char *>(wptr),
                    total_blocks, (uint32_t)blocks, (uint32_t)in_dim);
            const uint64_t xh_count = n_tok * in_dim;
            f32_to_f16_kernel<<<(xh_count + 255) / 256, 256>>>(
                    xh, (const float *)x->ptr, xh_count);
            if (cuda_ok(cudaGetLastError(), "q8 dequant gemm staging")) {
                const int gemm_trace = getenv("DS4_GLM_GEMM_TRACE") != NULL;
                cudaEvent_t ev0, ev1, ev2;
                if (gemm_trace) {
                    cudaEventCreate(&ev0); cudaEventCreate(&ev1); cudaEventCreate(&ev2);
                    cudaEventRecord(ev0);
                }
                const float alpha = 1.0f;
                const float beta = 0.0f;
                if (gemm_trace) cudaEventRecord(ev1);
                cublasStatus_t st = cublasGemmEx(
                        cuda_cublas_for_tier(logical_tier),
                        CUBLAS_OP_T, CUBLAS_OP_N,
                        (int)out_dim, (int)n_tok, (int)in_dim,
                        &alpha,
                        wh, CUDA_R_16F, (int)in_dim,
                        xh, CUDA_R_16F, (int)in_dim,
                        &beta,
                        olocal, CUDA_R_32F, (int)out_dim,
                        CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
                if (st == CUBLAS_STATUS_SUCCESS) {
                    if (!cuda_ok(cudaMemcpyAsync(out->ptr, olocal,
                                 n_tok * out_dim * sizeof(float),
                                 cudaMemcpyDeviceToDevice, 0),
                                 "q8 dequant gemm out copy")) {
                        st = CUBLAS_STATUS_INTERNAL_ERROR;
                    }
                }
                if (gemm_trace) {
                    cudaEventRecord(ev2);
                    cudaEventSynchronize(ev2);
                    float stage_ms = 0, gemm_ms = 0;
                    cudaEventElapsedTime(&stage_ms, ev0, ev1);
                    cudaEventElapsedTime(&gemm_ms, ev1, ev2);
                    fprintf(stderr,
                        "ds4: gemm trace in=%llu out=%llu n=%llu stage(before)=%.2f gemm=%.2f ms\n",
                        (unsigned long long)in_dim, (unsigned long long)out_dim,
                        (unsigned long long)n_tok, stage_ms, gemm_ms);
                    cudaEventDestroy(ev0); cudaEventDestroy(ev1); cudaEventDestroy(ev2);
                }
                if (st == CUBLAS_STATUS_SUCCESS) return 1;
                fprintf(stderr,
                        "ds4: q8 dequant gemm failed: status %d; using native path\n",
                        (int)st);
            }
        }
    }
    const uint64_t xq_bytes = n_tok * blocks * 32u;
    const uint64_t scale_offset = (xq_bytes + 15u) & ~15ull;
    const uint64_t tmp_bytes = scale_offset + n_tok * blocks * sizeof(float);
    void *tmp = cuda_tmp_alloc_on(logical_tier, tmp_bytes, "q8_0 prequant");
    if (!tmp) return 0;
    int8_t *xq = (int8_t *)tmp;
    float *xscale = (float *)((char *)tmp + scale_offset);
    const int use_dp4a = cuda_q8_use_dp4a();
    dim3 qgrid((unsigned)blocks, (unsigned)n_tok, 1);
    quantize_q8_0_f32_kernel<<<qgrid, 32>>>(xq, xscale, (const float *)x->ptr, in_dim, blocks);
    if (!cuda_ok(cudaGetLastError(), "matmul_q8_0 quantize launch")) return 0;
    if (n_tok == 1) {
        matmul_q8_0_preq_warp8_kernel<<<((unsigned)out_dim + 7u) / 8u, 256>>>(
                (float *)out->ptr,
                reinterpret_cast<const unsigned char *>(wptr),
                xq,
                xscale,
                in_dim,
                out_dim,
                blocks,
                use_dp4a);
        return cuda_ok(cudaGetLastError(), "matmul_q8_0 warp launch");
    }
    const bool force_decode_warp =
        n_tok == 2u && g_glm_mtp_verify_mode;
    if (n_tok > 1u && !force_decode_warp) {
        /* T matches the reduction width of whichever reference kernel would
         * have run: warp tree (32) for blocks <= 32, exact-thread tree
         * otherwise. */
        const uint32_t mma_T = blocks <= 32u ? 32u : cuda_q8_exact_threads(blocks);
        const int mma_rc = cuda_q8_mma_try_launch(
                (float *)out->ptr, reinterpret_cast<const unsigned char *>(wptr),
                xq, xscale, in_dim, out_dim, n_tok, blocks, blocks, out_dim, mma_T);
        if (mma_rc) return mma_rc > 0;
    }
    if (getenv("DS4_CUDA_NO_Q8_BATCH_WARP") == NULL &&
        getenv("DS4_CUDA_NO_Q8_BATCH_TOK8") == NULL &&
        blocks <= 32u &&
        n_tok >= 8u) {
        dim3 bgrid(((unsigned)out_dim + 7u) / 8u, ((unsigned)n_tok + 7u) / 8u, 1);
        matmul_q8_0_preq_batch_warp8_tok8_kernel<<<bgrid, 256>>>(
                (float *)out->ptr,
                reinterpret_cast<const unsigned char *>(wptr),
                xq,
                xscale,
                in_dim,
                out_dim,
                n_tok,
                blocks,
                use_dp4a);
        return cuda_ok(cudaGetLastError(), "matmul_q8_0 batch tok8 warp launch");
    }
    if (getenv("DS4_CUDA_NO_Q8_BATCH_WARP") == NULL &&
        getenv("DS4_CUDA_NO_Q8_BATCH_TOK4") == NULL &&
        blocks <= 32u &&
        n_tok >= 4u) {
        dim3 bgrid(((unsigned)out_dim + 7u) / 8u, ((unsigned)n_tok + 3u) / 4u, 1);
        matmul_q8_0_preq_batch_warp8_tok4_kernel<<<bgrid, 256>>>(
                (float *)out->ptr,
                reinterpret_cast<const unsigned char *>(wptr),
                xq,
                xscale,
                in_dim,
                out_dim,
                n_tok,
                blocks,
                use_dp4a);
        return cuda_ok(cudaGetLastError(), "matmul_q8_0 batch tok4 warp launch");
    }
    if (getenv("DS4_CUDA_NO_Q8_BATCH_WARP") == NULL &&
        (blocks <= 32u || force_decode_warp)) {
        if (force_decode_warp &&
            getenv("DS4_CUDA_GLM_VERIFY_NO_Q8_TOK2") == NULL) {
            matmul_q8_0_preq_batch_warp8_tok2_kernel
                    <<<((unsigned)out_dim + 7u) / 8u, 256>>>(
                    (float *)out->ptr,
                    reinterpret_cast<const unsigned char *>(wptr),
                    xq,
                    xscale,
                    in_dim,
                    out_dim,
                    blocks,
                    use_dp4a);
            return cuda_ok(cudaGetLastError(),
                           "matmul_q8_0 batch tok2 warp launch");
        }
        dim3 bgrid(((unsigned)out_dim + 7u) / 8u, (unsigned)n_tok, 1);
        matmul_q8_0_preq_batch_warp8_kernel<<<bgrid, 256>>>(
                (float *)out->ptr,
                reinterpret_cast<const unsigned char *>(wptr),
                xq,
                xscale,
                in_dim,
                out_dim,
                n_tok,
                blocks,
                use_dp4a);
        return cuda_ok(cudaGetLastError(), "matmul_q8_0 batch warp launch");
    }
    const unsigned exact_threads = cuda_q8_exact_threads(blocks);
    if (getenv("DS4_CUDA_NO_Q8_BATCH_EXACT_TOK2") == NULL &&
        n_tok >= 2u) {
        dim3 bgrid((unsigned)out_dim, ((unsigned)n_tok + 1u) / 2u, 1);
        matmul_q8_0_preq_batch_tok2_exact_kernel<<<bgrid, exact_threads>>>(
                (float *)out->ptr,
                reinterpret_cast<const unsigned char *>(wptr),
                xq,
                xscale,
                in_dim,
                out_dim,
                n_tok,
                blocks,
                use_dp4a);
        return cuda_ok(cudaGetLastError(), "matmul_q8_0 exact tok2 launch");
    }
    dim3 grid((unsigned)out_dim, (unsigned)n_tok, 1);
    matmul_q8_0_preq_kernel<<<grid, exact_threads>>>((float *)out->ptr,
                                                     reinterpret_cast<const unsigned char *>(wptr),
                                                     xq,
                                                     xscale,
                                                     in_dim, out_dim, n_tok, blocks,
                                                     use_dp4a);
    return cuda_ok(cudaGetLastError(), "matmul_q8_0 launch");
}

extern "C" int ds4_gpu_matmul_q8_0_tensor(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim, const ds4_gpu_tensor *x, uint64_t n_tok) {
    return cuda_matmul_q8_0_tensor_labeled(out, model_map, model_size, weight_offset,
                                           in_dim, out_dim, x, n_tok, "q8_0");
}

extern "C" int ds4_gpu_matmul_q8_0_top1_tensor(
        ds4_gpu_tensor       *selected,
        ds4_gpu_tensor       *values,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint32_t                index_offset) {
    if (!selected || !values || !x || !model_map ||
        in_dim == 0 || out_dim == 0 || out_dim > UINT32_MAX) {
        return 0;
    }
    const uint64_t blocks = (in_dim + 31u) / 32u;
    if (weight_offset > model_size || out_dim > UINT64_MAX / (blocks * 34u)) {
        return 0;
    }
    const uint64_t weight_bytes = out_dim * blocks * 34u;
    if (weight_bytes > model_size - weight_offset ||
        x->bytes < in_dim * sizeof(float) ||
        selected->bytes < sizeof(uint32_t) ||
        values->bytes < sizeof(float)) {
        return 0;
    }
    const int logical_tier = ds4_tensor_device_idx(selected);
    const char *wptr = cuda_resolve_weight_ptr(model_map,
                                               weight_offset,
                                               weight_bytes,
                                               logical_tier,
                                               "q8_0_top1");
    if (!wptr) return 0;

    const uint64_t xq_bytes = blocks * 32u;
    const uint64_t scale_offset = (xq_bytes + 15u) & ~15ull;
    const uint64_t key_offset =
        (scale_offset + blocks * sizeof(float) + 7u) & ~7ull;
    const uint64_t tmp_bytes = key_offset + sizeof(unsigned long long);
    void *tmp = cuda_tmp_alloc_on(logical_tier, tmp_bytes, "q8_0 top1 prequant");
    if (!tmp) return 0;
    int8_t *xq = (int8_t *)tmp;
    float *xscale = (float *)((char *)tmp + scale_offset);
    unsigned long long *best_key =
        (unsigned long long *)((char *)tmp + key_offset);
    const int use_dp4a = cuda_q8_use_dp4a();

    if (!cuda_ok(cudaMemsetAsync(best_key, 0, sizeof(*best_key)),
                 "matmul_q8_0_top1 clear")) {
        return 0;
    }
    quantize_q8_0_f32_kernel<<<(unsigned)blocks, 32>>>(
            xq,
            xscale,
            (const float *)x->ptr,
            in_dim,
            blocks);
    if (!cuda_ok(cudaGetLastError(), "matmul_q8_0_top1 quantize launch")) return 0;
    matmul_q8_0_top1_preq_warp8_kernel<<<((unsigned)out_dim + 7u) / 8u, 256>>>(
            best_key,
            reinterpret_cast<const unsigned char *>(wptr),
            xq,
            xscale,
            in_dim,
            out_dim,
            blocks,
            index_offset,
            use_dp4a);
    if (!cuda_ok(cudaGetLastError(), "matmul_q8_0_top1 launch")) return 0;
    matmul_q8_0_top1_unpack_kernel<<<1, 1>>>(
            (uint32_t *)selected->ptr,
            (float *)values->ptr,
            best_key);
    return cuda_ok(cudaGetLastError(), "matmul_q8_0_top1 unpack launch");
}

extern "C" int ds4_gpu_matmul_q8_0_kslice_rows_tensor(
        ds4_gpu_tensor *out,
        const void *model_map,
        uint64_t model_size,
        uint64_t weight_offset,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t in_start,
        uint64_t in_count,
        const ds4_gpu_tensor *x,
        uint64_t n_tok) {
    if (!out || !x || !model_map || in_dim == 0 || out_dim == 0 ||
        in_count == 0 || n_tok == 0 || n_tok > 65535u) return 0;
    if ((in_start % 32u) != 0 || (in_count % 32u) != 0 ||
        in_start > in_dim || in_count > in_dim - in_start) return 0;
    const uint64_t full_blocks = (in_dim + 31u) / 32u;
    const uint64_t block_start = in_start / 32u;
    const uint64_t slice_blocks = in_count / 32u;
    if (weight_offset > model_size || out_dim > UINT64_MAX / (full_blocks * 34u)) return 0;
    const uint64_t weight_bytes = out_dim * full_blocks * 34u;
    if (in_count > UINT64_MAX / n_tok || out_dim > UINT64_MAX / n_tok) {
        return 0;
    }
    if (weight_bytes > model_size - weight_offset ||
        x->bytes < n_tok * in_count * sizeof(float) ||
        out->bytes < n_tok * out_dim * sizeof(float)) return 0;
    const int logical_tier = ds4_tensor_device_idx(out);
    const unsigned char *wptr = reinterpret_cast<const unsigned char *>(
            cuda_resolve_weight_ptr(model_map, weight_offset, weight_bytes,
                                    logical_tier, "q8_0_kslice"));
    if (!wptr) return 0;

    const uint64_t xq_bytes = n_tok * slice_blocks * 32u;
    const uint64_t scale_offset = (xq_bytes + 15u) & ~15ull;
    const uint64_t tmp_bytes =
        scale_offset + n_tok * slice_blocks * sizeof(float);
    void *tmp = cuda_tmp_alloc_on(logical_tier, tmp_bytes, "q8_0 kslice prequant");
    if (!tmp) return 0;
    int8_t *xq = (int8_t *)tmp;
    float *xscale = (float *)((char *)tmp + scale_offset);
    const int use_dp4a = cuda_q8_use_dp4a();
    const dim3 qgrid((unsigned)slice_blocks, (unsigned)n_tok, 1u);
    quantize_q8_0_f32_kernel<<<qgrid, 32>>>(
            xq,
            xscale,
            (const float *)x->ptr,
            in_count,
            slice_blocks);
    if (!cuda_ok(cudaGetLastError(), "matmul_q8_0_kslice quantize launch")) return 0;
    const dim3 grid(((unsigned)out_dim + 7u) / 8u,
                    (unsigned)n_tok, 1u);
    matmul_q8_0_kslice_preq_warp8_kernel<<<grid, 256>>>(
            (float *)out->ptr,
            wptr,
            xq,
            xscale,
            in_count,
            out_dim,
            full_blocks,
            block_start,
            slice_blocks,
            use_dp4a);
    return cuda_ok(cudaGetLastError(), "matmul_q8_0_kslice launch");
}

extern "C" int ds4_gpu_matmul_q8_0_kslice_hc_expand_add_tensor(
        ds4_gpu_tensor       *out_hc,
        ds4_gpu_tensor       *block_out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        uint64_t                in_start,
        uint64_t                in_count,
        const ds4_gpu_tensor *x,
        const ds4_gpu_tensor *block_add,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc) {
    if (!out_hc || !block_out || !x || !block_add || !residual_hc || !split ||
        !model_map || in_dim == 0 || out_dim == 0 || in_count == 0 ||
        n_embd == 0 || n_hc == 0 || out_dim != (uint64_t)n_embd) {
        return 0;
    }
    if ((in_start % 32u) != 0 || (in_count % 32u) != 0 ||
        in_start > in_dim || in_count > in_dim - in_start) return 0;
    const uint64_t full_blocks = (in_dim + 31u) / 32u;
    const uint64_t block_start = in_start / 32u;
    const uint64_t slice_blocks = in_count / 32u;
    if (weight_offset > model_size || out_dim > UINT64_MAX / (full_blocks * 34u)) return 0;
    const uint64_t weight_bytes = out_dim * full_blocks * 34u;
    const uint64_t hc_bytes = (uint64_t)n_hc * n_embd * sizeof(float);
    const uint64_t split_bytes =
        (uint64_t)(2u * n_hc + n_hc * n_hc) * sizeof(float);
    if (weight_bytes > model_size - weight_offset ||
        x->bytes < in_count * sizeof(float) ||
        block_out->bytes < out_dim * sizeof(float) ||
        block_add->bytes < out_dim * sizeof(float) ||
        residual_hc->bytes < hc_bytes ||
        split->bytes < split_bytes ||
        out_hc->bytes < hc_bytes) {
        return 0;
    }
    const int logical_tier = ds4_tensor_device_idx(out_hc);
    const unsigned char *wptr = reinterpret_cast<const unsigned char *>(
            cuda_resolve_weight_ptr(model_map, weight_offset, weight_bytes,
                                    logical_tier, "q8_0_kslice_hc_expand_add"));
    if (!wptr) return 0;

    const uint64_t xq_bytes = slice_blocks * 32u;
    const uint64_t scale_offset = (xq_bytes + 15u) & ~15ull;
    const uint64_t tmp_bytes = scale_offset + slice_blocks * sizeof(float);
    void *tmp = cuda_tmp_alloc_on(logical_tier, tmp_bytes, "q8_0 kslice hc expand prequant");
    if (!tmp) return 0;
    int8_t *xq = (int8_t *)tmp;
    float *xscale = (float *)((char *)tmp + scale_offset);
    const int use_dp4a = cuda_q8_use_dp4a();
    quantize_q8_0_f32_kernel<<<(unsigned)slice_blocks, 32>>>(
            xq,
            xscale,
            (const float *)x->ptr,
            in_count,
            slice_blocks);
    if (!cuda_ok(cudaGetLastError(), "matmul_q8_0_kslice_hc_expand_add quantize launch")) return 0;
    matmul_q8_0_kslice_hc_expand_add_preq_warp8_kernel<<<((unsigned)out_dim + 7u) / 8u, 256>>>(
            (float *)out_hc->ptr,
            (float *)block_out->ptr,
            (const float *)block_add->ptr,
            (const float *)residual_hc->ptr,
            (const float *)split->ptr,
            wptr,
            xq,
            xscale,
            in_count,
            out_dim,
            full_blocks,
            block_start,
            slice_blocks,
            n_embd,
            n_hc,
            use_dp4a);
    return cuda_ok(cudaGetLastError(), "matmul_q8_0_kslice_hc_expand_add launch");
}

extern "C" int ds4_gpu_matmul_q8_0_pair_tensor(
        ds4_gpu_tensor *out0,
        ds4_gpu_tensor *out1,
        const void *model_map,
        uint64_t model_size,
        uint64_t weight0_offset,
        uint64_t weight1_offset,
        uint64_t in_dim,
        uint64_t out0_dim,
        uint64_t out1_dim,
        const ds4_gpu_tensor *x,
        uint64_t n_tok) {
    if (!out0 || !out1 || !x || !model_map || in_dim == 0 || out0_dim == 0 || out1_dim == 0 || n_tok == 0) {
        return 0;
    }
    const uint64_t blocks = (in_dim + 31) / 32;
    if (weight0_offset > model_size || weight1_offset > model_size ||
        out0_dim > UINT64_MAX / (blocks * 34) ||
        out1_dim > UINT64_MAX / (blocks * 34)) {
        return 0;
    }
    const uint64_t weight0_bytes = out0_dim * blocks * 34;
    const uint64_t weight1_bytes = out1_dim * blocks * 34;
    if (weight0_bytes > model_size - weight0_offset ||
        weight1_bytes > model_size - weight1_offset ||
        x->bytes < in_dim * sizeof(float) ||
        out0->bytes < out0_dim * sizeof(float) ||
        out1->bytes < out1_dim * sizeof(float)) {
        return 0;
    }
    const int logical_tier = ds4_tensor_device_idx(out0);
    const char *w0 = cuda_resolve_weight_ptr(model_map, weight0_offset, weight0_bytes, logical_tier, "q8_0_pair0");
    const char *w1 = cuda_resolve_weight_ptr(model_map, weight1_offset, weight1_bytes, logical_tier, "q8_0_pair1");
    if (!w0 || !w1) return 0;

    const bool force_decode_warp =
        n_tok == 2u && g_glm_mtp_verify_mode;
    if (n_tok != 1 && !force_decode_warp && !g_q8_cache_suppressed &&
        getenv("DS4_CUDA_Q8_PAIR_BATCH") == NULL) {
        return cuda_matmul_q8_0_tensor_labeled(out0, model_map, model_size, weight0_offset,
                                               in_dim, out0_dim, x, n_tok, "q8_0_pair0") &&
               cuda_matmul_q8_0_tensor_labeled(out1, model_map, model_size, weight1_offset,
                                               in_dim, out1_dim, x, n_tok, "q8_0_pair1");
    }

    const uint64_t xq_bytes = n_tok * blocks * 32u;
    const uint64_t scale_offset = (xq_bytes + 15u) & ~15ull;
    const uint64_t tmp_bytes = scale_offset + n_tok * blocks * sizeof(float);
    void *tmp = cuda_tmp_alloc_on(logical_tier, tmp_bytes, "q8_0 pair prequant");
    if (!tmp) return 0;
    int8_t *xq = (int8_t *)tmp;
    float *xscale = (float *)((char *)tmp + scale_offset);
    const int use_dp4a = cuda_q8_use_dp4a();
    dim3 qgrid((unsigned)blocks, (unsigned)n_tok, 1);
    quantize_q8_0_f32_kernel<<<qgrid, 32>>>(xq, xscale, (const float *)x->ptr, in_dim, blocks);
    if (!cuda_ok(cudaGetLastError(), "matmul_q8_0 pair quantize launch")) return 0;
    if (n_tok != 1) {
        if (force_decode_warp &&
            getenv("DS4_CUDA_GLM_VERIFY_NO_Q8_TOK2") == NULL) {
            matmul_q8_0_preq_batch_warp8_tok2_kernel
                    <<<((unsigned)out0_dim + 7u) / 8u, 256>>>(
                    (float *)out0->ptr,
                    reinterpret_cast<const unsigned char *>(w0),
                    xq, xscale, in_dim, out0_dim, blocks, use_dp4a);
            if (!cuda_ok(cudaGetLastError(),
                         "matmul_q8_0 pair0 tok2 warp launch")) {
                return 0;
            }
            matmul_q8_0_preq_batch_warp8_tok2_kernel
                    <<<((unsigned)out1_dim + 7u) / 8u, 256>>>(
                    (float *)out1->ptr,
                    reinterpret_cast<const unsigned char *>(w1),
                    xq, xscale, in_dim, out1_dim, blocks, use_dp4a);
            return cuda_ok(cudaGetLastError(),
                           "matmul_q8_0 pair1 tok2 warp launch");
        }
        const uint32_t mma_T = blocks <= 32u ? 32u : cuda_q8_exact_threads(blocks);
        int mma_rc = cuda_q8_mma_try_launch(
                (float *)out0->ptr, reinterpret_cast<const unsigned char *>(w0),
                xq, xscale, in_dim, out0_dim, n_tok, blocks, blocks, out0_dim, mma_T);
        if (mma_rc < 0) return 0;
        if (mma_rc > 0) {
            mma_rc = cuda_q8_mma_try_launch(
                    (float *)out1->ptr, reinterpret_cast<const unsigned char *>(w1),
                    xq, xscale, in_dim, out1_dim, n_tok, blocks, blocks, out1_dim, mma_T);
            if (mma_rc > 0) return 1;
            return 0;
        }
        if (getenv("DS4_CUDA_NO_Q8_BATCH_WARP") == NULL &&
            getenv("DS4_CUDA_NO_Q8_BATCH_TOK8") == NULL &&
            blocks <= 32u &&
            n_tok >= 8u) {
            dim3 grid0(((unsigned)out0_dim + 7u) / 8u, ((unsigned)n_tok + 7u) / 8u, 1);
            matmul_q8_0_preq_batch_warp8_tok8_kernel<<<grid0, 256>>>(
                    (float *)out0->ptr,
                    reinterpret_cast<const unsigned char *>(w0),
                    xq,
                    xscale,
                    in_dim,
                    out0_dim,
                    n_tok,
                    blocks,
                    use_dp4a);
            if (!cuda_ok(cudaGetLastError(), "matmul_q8_0 pair0 batch tok8 launch")) return 0;
            dim3 grid1(((unsigned)out1_dim + 7u) / 8u, ((unsigned)n_tok + 7u) / 8u, 1);
            matmul_q8_0_preq_batch_warp8_tok8_kernel<<<grid1, 256>>>(
                    (float *)out1->ptr,
                    reinterpret_cast<const unsigned char *>(w1),
                    xq,
                    xscale,
                    in_dim,
                    out1_dim,
                    n_tok,
                    blocks,
                    use_dp4a);
            return cuda_ok(cudaGetLastError(), "matmul_q8_0 pair1 batch tok8 launch");
        }
        if (getenv("DS4_CUDA_NO_Q8_BATCH_WARP") == NULL &&
            getenv("DS4_CUDA_NO_Q8_BATCH_TOK4") == NULL &&
            blocks <= 32u &&
            n_tok >= 4u) {
            dim3 grid0(((unsigned)out0_dim + 7u) / 8u, ((unsigned)n_tok + 3u) / 4u, 1);
            matmul_q8_0_preq_batch_warp8_tok4_kernel<<<grid0, 256>>>(
                    (float *)out0->ptr,
                    reinterpret_cast<const unsigned char *>(w0),
                    xq,
                    xscale,
                    in_dim,
                    out0_dim,
                    n_tok,
                    blocks,
                    use_dp4a);
            if (!cuda_ok(cudaGetLastError(), "matmul_q8_0 pair0 batch tok4 launch")) return 0;
            dim3 grid1(((unsigned)out1_dim + 7u) / 8u, ((unsigned)n_tok + 3u) / 4u, 1);
            matmul_q8_0_preq_batch_warp8_tok4_kernel<<<grid1, 256>>>(
                    (float *)out1->ptr,
                    reinterpret_cast<const unsigned char *>(w1),
                    xq,
                    xscale,
                    in_dim,
                    out1_dim,
                    n_tok,
                    blocks,
                    use_dp4a);
            return cuda_ok(cudaGetLastError(), "matmul_q8_0 pair1 batch tok4 launch");
        }
        if (getenv("DS4_CUDA_NO_Q8_BATCH_WARP") == NULL &&
            blocks <= 32u) {
            dim3 grid0(((unsigned)out0_dim + 7u) / 8u, (unsigned)n_tok, 1);
            matmul_q8_0_preq_batch_warp8_kernel<<<grid0, 256>>>(
                    (float *)out0->ptr,
                    reinterpret_cast<const unsigned char *>(w0),
                    xq,
                    xscale,
                    in_dim,
                    out0_dim,
                    n_tok,
                    blocks,
                    use_dp4a);
            if (!cuda_ok(cudaGetLastError(), "matmul_q8_0 pair0 batch warp launch")) return 0;
            dim3 grid1(((unsigned)out1_dim + 7u) / 8u, (unsigned)n_tok, 1);
            matmul_q8_0_preq_batch_warp8_kernel<<<grid1, 256>>>(
                    (float *)out1->ptr,
                    reinterpret_cast<const unsigned char *>(w1),
                    xq,
                    xscale,
                    in_dim,
                    out1_dim,
                    n_tok,
                    blocks,
                    use_dp4a);
            return cuda_ok(cudaGetLastError(), "matmul_q8_0 pair1 batch warp launch");
        }
        if (getenv("DS4_CUDA_NO_Q8_PAIR_BATCH_EXACT") == NULL) {
            const uint64_t max_out_dim = out0_dim > out1_dim ? out0_dim : out1_dim;
            const unsigned exact_threads = cuda_q8_exact_threads(blocks);
            if (getenv("DS4_CUDA_NO_Q8_PAIR_BATCH_EXACT_TOK2") == NULL &&
                n_tok >= 2u) {
                dim3 grid((unsigned)max_out_dim, ((unsigned)n_tok + 1u) / 2u, 1);
                matmul_q8_0_pair_preq_batch_tok2_exact_kernel<<<grid, exact_threads>>>(
                        (float *)out0->ptr,
                        (float *)out1->ptr,
                        reinterpret_cast<const unsigned char *>(w0),
                        reinterpret_cast<const unsigned char *>(w1),
                        xq,
                        xscale,
                        in_dim,
                        out0_dim,
                        out1_dim,
                        n_tok,
                        blocks,
                        use_dp4a);
                return cuda_ok(cudaGetLastError(), "matmul_q8_0 pair exact tok2 launch");
            }
            dim3 grid((unsigned)max_out_dim, (unsigned)n_tok, 1);
            matmul_q8_0_pair_preq_batch_kernel<<<grid, exact_threads>>>(
                    (float *)out0->ptr,
                    (float *)out1->ptr,
                    reinterpret_cast<const unsigned char *>(w0),
                    reinterpret_cast<const unsigned char *>(w1),
                    xq,
                    xscale,
                    in_dim,
                    out0_dim,
                    out1_dim,
                    n_tok,
                    blocks,
                    use_dp4a);
            return cuda_ok(cudaGetLastError(), "matmul_q8_0 pair batch exact launch");
        }
        const unsigned exact_threads = cuda_q8_exact_threads(blocks);
        dim3 grid0((unsigned)out0_dim, (unsigned)n_tok, 1);
        matmul_q8_0_preq_kernel<<<grid0, exact_threads>>>((float *)out0->ptr,
                                                          reinterpret_cast<const unsigned char *>(w0),
                                                          xq,
                                                          xscale,
                                                          in_dim, out0_dim, n_tok, blocks,
                                                          use_dp4a);
        if (!cuda_ok(cudaGetLastError(), "matmul_q8_0 pair0 batch launch")) return 0;
        dim3 grid1((unsigned)out1_dim, (unsigned)n_tok, 1);
        matmul_q8_0_preq_kernel<<<grid1, exact_threads>>>((float *)out1->ptr,
                                                          reinterpret_cast<const unsigned char *>(w1),
                                                          xq,
                                                          xscale,
                                                          in_dim, out1_dim, n_tok, blocks,
                                                          use_dp4a);
        return cuda_ok(cudaGetLastError(), "matmul_q8_0 pair1 batch launch");
    }
    const uint64_t max_out = out0_dim > out1_dim ? out0_dim : out1_dim;
    matmul_q8_0_pair_preq_warp8_kernel<<<((unsigned)max_out + 7u) / 8u, 256>>>(
            (float *)out0->ptr,
            (float *)out1->ptr,
            reinterpret_cast<const unsigned char *>(w0),
            reinterpret_cast<const unsigned char *>(w1),
            xq,
            xscale,
            in_dim,
            out0_dim,
            out1_dim,
            blocks,
            use_dp4a);
    return cuda_ok(cudaGetLastError(), "matmul_q8_0 pair warp launch");
}

extern "C" int ds4_gpu_matmul_q8_0_decode_rows_exact_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint32_t                n_rows) {
    if (!out || !x || !model_map || in_dim == 0u || out_dim == 0u ||
        n_rows == 0u ||
        x->bytes < (uint64_t)n_rows * in_dim * sizeof(float) ||
        out->bytes < (uint64_t)n_rows * out_dim * sizeof(float)) {
        return 0;
    }
    const uint64_t blocks = (in_dim + 31u) / 32u;
    if (weight_offset > model_size ||
        out_dim > UINT64_MAX / (blocks * 34u)) {
        return 0;
    }
    const uint64_t weight_bytes = out_dim * blocks * 34u;
    if (weight_bytes > model_size - weight_offset) return 0;
    const int logical_tier = ds4_tensor_device_idx(out);
    if (logical_tier < 0 || logical_tier >= g_n_gpus ||
        ds4_tensor_device_idx(x) != logical_tier) {
        return 0;
    }
    const char *wptr = cuda_resolve_weight_ptr(
            model_map, weight_offset, weight_bytes, logical_tier,
            "q8_0 decode rows exact");
    if (!wptr) return 0;

    const uint64_t xq_bytes = (uint64_t)n_rows * blocks * 32u;
    const uint64_t scale_offset = (xq_bytes + 15u) & ~15ull;
    const uint64_t tmp_bytes =
        scale_offset + (uint64_t)n_rows * blocks * sizeof(float);
    void *tmp = cuda_tmp_alloc_on(
            logical_tier, tmp_bytes, "q8_0 decode rows exact prequant");
    if (!tmp) return 0;
    int8_t *xq = (int8_t *)tmp;
    float *xscale = (float *)((char *)tmp + scale_offset);
    dim3 qgrid((unsigned)blocks, n_rows, 1u);
    quantize_q8_0_f32_kernel<<<qgrid, 32>>>(
            xq, xscale, (const float *)x->ptr, in_dim, blocks);
    if (!cuda_ok(cudaGetLastError(),
                 "q8_0 decode rows exact quantize launch")) {
        return 0;
    }
    dim3 grid(((unsigned)out_dim + 7u) / 8u, n_rows, 1u);
    matmul_q8_0_preq_warp8_kernel<<<grid, 256>>>(
            (float *)out->ptr,
            reinterpret_cast<const unsigned char *>(wptr),
            xq, xscale, in_dim, out_dim, blocks, cuda_q8_use_dp4a());
    return cuda_ok(cudaGetLastError(),
                   "q8_0 decode rows exact warp launch");
}

extern "C" int ds4_gpu_matmul_q8_0_pair_decode_rows_exact_tensor(
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
        uint32_t                n_rows) {
    if (!out0 || !out1 || !x || !model_map || in_dim == 0u ||
        out0_dim == 0u || out1_dim == 0u || n_rows == 0u ||
        x->bytes < (uint64_t)n_rows * in_dim * sizeof(float) ||
        out0->bytes < (uint64_t)n_rows * out0_dim * sizeof(float) ||
        out1->bytes < (uint64_t)n_rows * out1_dim * sizeof(float)) {
        return 0;
    }
    const uint64_t blocks = (in_dim + 31u) / 32u;
    if (weight0_offset > model_size || weight1_offset > model_size ||
        out0_dim > UINT64_MAX / (blocks * 34u) ||
        out1_dim > UINT64_MAX / (blocks * 34u)) {
        return 0;
    }
    const uint64_t weight0_bytes = out0_dim * blocks * 34u;
    const uint64_t weight1_bytes = out1_dim * blocks * 34u;
    if (weight0_bytes > model_size - weight0_offset ||
        weight1_bytes > model_size - weight1_offset) {
        return 0;
    }
    const int logical_tier = ds4_tensor_device_idx(out0);
    if (logical_tier < 0 || logical_tier >= g_n_gpus ||
        ds4_tensor_device_idx(out1) != logical_tier ||
        ds4_tensor_device_idx(x) != logical_tier) {
        return 0;
    }
    const char *w0 = cuda_resolve_weight_ptr(
            model_map, weight0_offset, weight0_bytes, logical_tier,
            "q8_0 pair decode rows exact gate");
    const char *w1 = cuda_resolve_weight_ptr(
            model_map, weight1_offset, weight1_bytes, logical_tier,
            "q8_0 pair decode rows exact up");
    if (!w0 || !w1) return 0;

    const uint64_t xq_bytes = (uint64_t)n_rows * blocks * 32u;
    const uint64_t scale_offset = (xq_bytes + 15u) & ~15ull;
    const uint64_t tmp_bytes =
        scale_offset + (uint64_t)n_rows * blocks * sizeof(float);
    void *tmp = cuda_tmp_alloc_on(
            logical_tier, tmp_bytes, "q8_0 pair decode rows exact prequant");
    if (!tmp) return 0;
    int8_t *xq = (int8_t *)tmp;
    float *xscale = (float *)((char *)tmp + scale_offset);
    dim3 qgrid((unsigned)blocks, n_rows, 1u);
    quantize_q8_0_f32_kernel<<<qgrid, 32>>>(
            xq, xscale, (const float *)x->ptr, in_dim, blocks);
    if (!cuda_ok(cudaGetLastError(),
                 "q8_0 pair decode rows exact quantize launch")) {
        return 0;
    }
    const uint64_t max_out = out0_dim > out1_dim ? out0_dim : out1_dim;
    dim3 grid(((unsigned)max_out + 7u) / 8u, n_rows, 1u);
    matmul_q8_0_pair_preq_warp8_kernel<<<grid, 256>>>(
            (float *)out0->ptr,
            (float *)out1->ptr,
            reinterpret_cast<const unsigned char *>(w0),
            reinterpret_cast<const unsigned char *>(w1),
            xq, xscale, in_dim, out0_dim, out1_dim, blocks,
            cuda_q8_use_dp4a());
    return cuda_ok(cudaGetLastError(),
                   "q8_0 pair decode rows exact warp launch");
}

static int cuda_matmul_q8_0_hc_expand_tensor_labeled(
        ds4_gpu_tensor       *out_hc,
        ds4_gpu_tensor       *block_out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        const ds4_gpu_tensor *block_add,
        const ds4_gpu_tensor *block_add2,
        const ds4_gpu_tensor *owned_home_slots,
        const ds4_gpu_tensor *owned_peer_packed,
        const ds4_gpu_tensor *owned_selected,
        uint32_t                owned_expert_split,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc,
        const char             *label) {
    if (!out_hc || !block_out || !x || !residual_hc || !split || !model_map ||
        in_dim == 0 || out_dim == 0 || n_embd == 0 || n_hc == 0 ||
        out_dim != (uint64_t)n_embd) {
        return 0;
    }
    const uint64_t blocks = (in_dim + 31) / 32;
    if (weight_offset > model_size || out_dim > UINT64_MAX / (blocks * 34)) return 0;
    const uint64_t weight_bytes = out_dim * blocks * 34;
    const uint64_t hc_bytes = (uint64_t)n_hc * n_embd * sizeof(float);
    const uint64_t split_bytes = (uint64_t)(2u * n_hc + n_hc * n_hc) * sizeof(float);
    if (weight_bytes > model_size - weight_offset ||
        x->bytes < in_dim * sizeof(float) ||
        block_out->bytes < out_dim * sizeof(float) ||
        residual_hc->bytes < hc_bytes ||
        split->bytes < split_bytes ||
        out_hc->bytes < hc_bytes ||
        (block_add && block_add->bytes < out_dim * sizeof(float)) ||
        (block_add2 && block_add2->bytes < out_dim * sizeof(float)) ||
        ((owned_home_slots || owned_peer_packed || owned_selected) &&
         (!owned_home_slots || !owned_peer_packed || !owned_selected ||
          owned_expert_split == 0u ||
          owned_home_slots->bytes < 6u * out_dim * sizeof(float) ||
          owned_peer_packed->bytes < 4u * out_dim * sizeof(float) ||
          owned_selected->bytes < 6u * sizeof(int32_t)))) {
        return 0;
    }
    const int logical_tier = ds4_tensor_device_idx(out_hc);
    const char *wptr = cuda_resolve_weight_ptr(model_map, weight_offset, weight_bytes, logical_tier, label ? label : "q8_0_hc_expand");
    if (!wptr) return 0;

    const uint64_t xq_bytes = blocks * 32u;
    const uint64_t scale_offset = (xq_bytes + 15u) & ~15ull;
    const uint64_t tmp_bytes = scale_offset + blocks * sizeof(float);
    void *tmp = cuda_tmp_alloc_on(logical_tier, tmp_bytes, "q8_0 hc expand prequant");
    if (!tmp) return 0;
    int8_t *xq = (int8_t *)tmp;
    float *xscale = (float *)((char *)tmp + scale_offset);
    const int use_dp4a = cuda_q8_use_dp4a();
    quantize_q8_0_f32_kernel<<<(unsigned)blocks, 32>>>(xq, xscale, (const float *)x->ptr, in_dim, blocks);
    if (!cuda_ok(cudaGetLastError(), "matmul_q8_0_hc_expand quantize launch")) return 0;
    matmul_q8_0_hc_expand_preq_warp8_kernel<<<((unsigned)out_dim + 7u) / 8u, 256>>>(
            (float *)out_hc->ptr,
            (float *)block_out->ptr,
            block_add ? (const float *)block_add->ptr : (const float *)block_out->ptr,
            block_add2 ? (const float *)block_add2->ptr : (const float *)block_out->ptr,
            owned_home_slots ? (const float *)owned_home_slots->ptr : NULL,
            owned_peer_packed ? (const float *)owned_peer_packed->ptr : NULL,
            owned_selected ? (const int32_t *)owned_selected->ptr : NULL,
            (const float *)residual_hc->ptr,
            (const float *)split->ptr,
            reinterpret_cast<const unsigned char *>(wptr),
            xq,
            xscale,
            in_dim,
            out_dim,
            n_embd,
            n_hc,
            blocks,
            block_add ? 1 : 0,
            block_add2 ? 1 : 0,
            owned_home_slots ? 1 : 0,
            owned_expert_split,
            use_dp4a);
    return cuda_ok(cudaGetLastError(), "matmul_q8_0_hc_expand launch");
}

extern "C" int ds4_gpu_matmul_f16_tensor(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim, const ds4_gpu_tensor *x, uint64_t n_tok) {
    if (!out || !x || !model_map) return 0;
    if (weight_offset > model_size || out_dim > UINT64_MAX / in_dim) return 0;
    uint64_t weight_bytes = out_dim * in_dim * sizeof(uint16_t);
    if (weight_bytes > model_size - weight_offset) return 0;
    if (x->bytes < n_tok * in_dim * sizeof(float) ||
        out->bytes < n_tok * out_dim * sizeof(float)) return 0;
    const int logical_tier = ds4_tensor_device_idx(out);
    const char *wptr = cuda_resolve_weight_ptr(model_map, weight_offset, weight_bytes, logical_tier, "f16");
    if (!wptr) return 0;
    const __half *w = (const __half *)wptr;
    const int serial_f16 = getenv("DS4_CUDA_SERIAL_F16_MATMUL") != NULL;
    const int router_shape = in_dim == 4096u && out_dim == 256u && n_tok == 1u;
    const int serial_router =
        !serial_f16 &&
        router_shape &&
        getenv("DS4_CUDA_SERIAL_ROUTER") != NULL;
    const int ordered_router =
        !serial_f16 &&
        !serial_router &&
        n_tok == 1u &&
        getenv("DS4_CUDA_NO_ORDERED_F16_MATMUL") == NULL;
    const int small_out_one_token =
        !serial_f16 &&
        !serial_router &&
        !g_quality_mode &&
        n_tok == 1u &&
        out_dim <= 32u &&
        in_dim >= 8192u &&
        getenv("DS4_CUDA_F16_SMALL_OUT") != NULL &&
        getenv("DS4_CUDA_NO_ORDERED_F16_MATMUL") == NULL &&
        getenv("DS4_CUDA_NO_F16_SMALL_OUT") == NULL;
    if (small_out_one_token) {
        matmul_f16_small_out_hx_ordered_chunks_kernel<<<(unsigned)out_dim, 32>>>(
                (float *)out->ptr,
                w,
                (const float *)x->ptr,
                in_dim,
                out_dim);
        return cuda_ok(cudaGetLastError(), "matmul_f16_small_out_hx_ordered_chunks launch");
    }
    const int small_out_batch =
        !serial_f16 &&
        !serial_router &&
        n_tok > 1u &&
        out_dim <= 32u &&
        in_dim >= 4096u &&
        (g_quality_mode || getenv("DS4_CUDA_F16_SMALL_BATCH") != NULL) &&
        getenv("DS4_CUDA_NO_F16_SMALL_BATCH") == NULL;
    if (small_out_batch) {
        matmul_f16_small_out_batch_kernel<<<(unsigned)n_tok, 256>>>(
                (float *)out->ptr,
                w,
                (const float *)x->ptr,
                in_dim,
                out_dim,
                n_tok);
        return cuda_ok(cudaGetLastError(), "matmul_f16_small_out_batch launch");
    }
    const int cublas_one_token =
        n_tok == 1u &&
        getenv("DS4_CUDA_NO_F16_CUBLAS_ONE") == NULL &&
        (!g_quality_mode || getenv("DS4_CUDA_F16_CUBLAS_ONE") != NULL);
    const int cublas_batch =
        n_tok > 1u && getenv("DS4_CUDA_NO_F16_CUBLAS_BATCH") == NULL;
    if (!serial_f16 && g_cublas_ready && (cublas_batch || cublas_one_token)) {
        const uint64_t xh_count = n_tok * in_dim;
        __half *xh = (__half *)cuda_tmp_alloc_on(logical_tier, xh_count * sizeof(__half), "f16 gemm activations");
        if (!xh) return 0;
        f32_to_f16_kernel<<<(xh_count + 255) / 256, 256>>>(xh, (const float *)x->ptr, xh_count);
        if (!cuda_ok(cudaGetLastError(), "f16 activation convert launch")) return 0;
        const float alpha = 1.0f;
        const float beta = 0.0f;
        cublasStatus_t st = cublasGemmEx(cuda_cublas_for_tier(logical_tier),
                                         CUBLAS_OP_T,
                                         CUBLAS_OP_N,
                                         (int)out_dim,
                                         (int)n_tok,
                                         (int)in_dim,
                                         &alpha,
                                         w,
                                         CUDA_R_16F,
                                         (int)in_dim,
                                         xh,
                                         CUDA_R_16F,
                                         (int)in_dim,
                                         &beta,
                                         out->ptr,
                                         CUDA_R_32F,
                                         (int)out_dim,
                                         CUDA_R_32F,
                                         CUBLAS_GEMM_DEFAULT);
        return cublas_ok(st, "f16 matmul");
    }
    dim3 grid((unsigned)out_dim, (unsigned)n_tok, 1);
    if (serial_f16 || serial_router) {
        matmul_f16_serial_kernel<<<grid, 1>>>((float *)out->ptr, w, (const float *)x->ptr, in_dim, out_dim, n_tok);
        return cuda_ok(cudaGetLastError(), serial_router ? "matmul_f16_router_serial launch" : "matmul_f16_serial launch");
    }
    if (ordered_router) {
        matmul_f16_ordered_chunks_kernel<<<grid, 32>>>((float *)out->ptr, w, (const float *)x->ptr, in_dim, out_dim, n_tok);
        return cuda_ok(cudaGetLastError(), "matmul_f16_ordered_chunks launch");
    }
    matmul_f16_kernel<<<grid, 256>>>((float *)out->ptr, w, (const float *)x->ptr, in_dim, out_dim, n_tok);
    return cuda_ok(cudaGetLastError(), "matmul_f16 launch");
}

extern "C" int ds4_gpu_matmul_f16_router_rows_exact_tensor(
        ds4_gpu_tensor *out,
        const void *model_map,
        uint64_t model_size,
        uint64_t weight_offset,
        const ds4_gpu_tensor *x,
        uint32_t n_rows) {
    const uint64_t in_dim = 4096u;
    const uint64_t out_dim = 256u;
    if (!out || !x || !model_map || n_rows == 0u ||
        weight_offset > model_size) {
        return 0;
    }
    const uint64_t weight_bytes = in_dim * out_dim * sizeof(uint16_t);
    if (weight_bytes > model_size - weight_offset ||
        x->bytes < (uint64_t)n_rows * in_dim * sizeof(float) ||
        out->bytes < (uint64_t)n_rows * out_dim * sizeof(float)) {
        return 0;
    }
    if (n_rows == 1u) {
        return ds4_gpu_matmul_f16_tensor(
                out, model_map, model_size, weight_offset,
                in_dim, out_dim, x, 1);
    }
    const int logical_tier = ds4_tensor_device_idx(out);
    if (ds4_tensor_device_idx(x) != logical_tier || !g_cublas_ready) return 0;
    const __half *w = (const __half *)cuda_resolve_weight_ptr(
            model_map, weight_offset, weight_bytes, logical_tier,
            "f16_router_rows_exact");
    if (!w) return 0;

    const uint64_t xh_count = (uint64_t)n_rows * in_dim;
    __half *xh = (__half *)cuda_tmp_alloc_on(
            logical_tier, xh_count * sizeof(__half),
            "f16 exact router batch activations");
    if (!xh) return 0;
    f32_to_f16_kernel<<<(xh_count + 255u) / 256u, 256>>>(
            xh, (const float *)x->ptr, xh_count);
    if (!cuda_ok(cudaGetLastError(),
                 "f16 exact router activation convert launch")) {
        return 0;
    }
    const float alpha = 1.0f;
    const float beta = 0.0f;
    /* Larger batchCount values let cuBLAS select a different reduction and
     * change logits. Four-row calls match the one-row decode bit for bit on
     * this projection, while still replacing most per-session launches. */
    uint32_t row = 0;
    for (; row + 4u <= n_rows; row += 4u) {
        cublasStatus_t st = cublasGemmStridedBatchedEx(
                cuda_cublas_for_tier(logical_tier),
                CUBLAS_OP_T,
                CUBLAS_OP_N,
                (int)out_dim,
                1,
                (int)in_dim,
                &alpha,
                w,
                CUDA_R_16F,
                (int)in_dim,
                0,
                xh + (uint64_t)row * in_dim,
                CUDA_R_16F,
                (int)in_dim,
                (long long int)in_dim,
                &beta,
                (float *)out->ptr + (uint64_t)row * out_dim,
                CUDA_R_32F,
                (int)out_dim,
                (long long int)out_dim,
                4,
                CUDA_R_32F,
                CUBLAS_GEMM_DEFAULT);
        if (!cublas_ok(st, "f16 exact router row batch")) return 0;
    }
    for (; row < n_rows; row++) {
        ds4_gpu_tensor out_row = *out;
        ds4_gpu_tensor x_row = *x;
        out_row.ptr = (float *)out->ptr + (uint64_t)row * out_dim;
        out_row.bytes = out_dim * sizeof(float);
        x_row.ptr = (float *)x->ptr + (uint64_t)row * in_dim;
        x_row.bytes = in_dim * sizeof(float);
        if (!ds4_gpu_matmul_f16_tensor(
                    &out_row, model_map, model_size, weight_offset,
                    in_dim, out_dim, &x_row, 1)) {
            return 0;
        }
    }
    return 1;
}

extern "C" int ds4_gpu_matmul_f16_pair_tensor(
        ds4_gpu_tensor *out0,
        ds4_gpu_tensor *out1,
        const void *model_map,
        uint64_t model_size,
        uint64_t weight0_offset,
        uint64_t weight1_offset,
        uint64_t in_dim,
        uint64_t out_dim,
        const ds4_gpu_tensor *x,
        uint64_t n_tok) {
    if (!out0 || !out1 || !x || !model_map || in_dim == 0 || out_dim == 0 || n_tok == 0) {
        return 0;
    }
    if (getenv("DS4_CUDA_NO_F16_PAIR_MATMUL") != NULL ||
        getenv("DS4_CUDA_SERIAL_F16_MATMUL") != NULL ||
        getenv("DS4_CUDA_SERIAL_ROUTER") != NULL ||
        getenv("DS4_CUDA_NO_ORDERED_F16_MATMUL") != NULL) {
        return ds4_gpu_matmul_f16_tensor(out0, model_map, model_size, weight0_offset,
                                           in_dim, out_dim, x, n_tok) &&
               ds4_gpu_matmul_f16_tensor(out1, model_map, model_size, weight1_offset,
                                           in_dim, out_dim, x, n_tok);
    }
    if (weight0_offset > model_size || weight1_offset > model_size ||
        out_dim > UINT64_MAX / in_dim ||
        n_tok > UINT64_MAX / in_dim ||
        n_tok > UINT64_MAX / out_dim) {
        return 0;
    }
    const uint64_t weight_bytes = out_dim * in_dim * sizeof(uint16_t);
    const uint64_t x_bytes = n_tok * in_dim * sizeof(float);
    const uint64_t out_bytes = n_tok * out_dim * sizeof(float);
    if (weight_bytes > model_size - weight0_offset ||
        weight_bytes > model_size - weight1_offset ||
        x->bytes < x_bytes ||
        out0->bytes < out_bytes ||
        out1->bytes < out_bytes) {
        return 0;
    }
    const int logical_tier = ds4_tensor_device_idx(out0);
    if (ds4_tensor_device_idx(out1) != logical_tier) {
        return ds4_gpu_matmul_f16_tensor(out0, model_map, model_size, weight0_offset,
                                           in_dim, out_dim, x, n_tok) &&
               ds4_gpu_matmul_f16_tensor(out1, model_map, model_size, weight1_offset,
                                           in_dim, out_dim, x, n_tok);
    }
    const __half *w0 = (const __half *)cuda_resolve_weight_ptr(model_map, weight0_offset, weight_bytes, logical_tier, "f16_pair0");
    const __half *w1 = (const __half *)cuda_resolve_weight_ptr(model_map, weight1_offset, weight_bytes, logical_tier, "f16_pair1");
    if (!w0 || !w1) return 0;
    if (n_tok > 1) {
        const bool small_out_batch_requested =
            out_dim <= 32u &&
            in_dim >= 4096u &&
            (g_quality_mode || getenv("DS4_CUDA_F16_SMALL_BATCH") != NULL) &&
            getenv("DS4_CUDA_NO_F16_SMALL_BATCH") == NULL;
        if (!small_out_batch_requested &&
            g_cublas_ready &&
            getenv("DS4_CUDA_NO_F16_CUBLAS_BATCH") == NULL) {
            const uint64_t xh_count = n_tok * in_dim;
            __half *xh = (__half *)cuda_tmp_alloc_on(logical_tier,
                                                     xh_count * sizeof(__half),
                                                     "f16 pair gemm activations");
            if (!xh) return 0;
            f32_to_f16_kernel<<<(xh_count + 255) / 256, 256>>>(
                    xh, (const float *)x->ptr, xh_count);
            if (!cuda_ok(cudaGetLastError(), "f16 pair activation convert launch")) return 0;
            const float alpha = 1.0f;
            const float beta = 0.0f;
            cublasStatus_t st = cublasGemmEx(cuda_cublas_for_tier(logical_tier),
                                             CUBLAS_OP_T,
                                             CUBLAS_OP_N,
                                             (int)out_dim,
                                             (int)n_tok,
                                             (int)in_dim,
                                             &alpha,
                                             w0,
                                             CUDA_R_16F,
                                             (int)in_dim,
                                             xh,
                                             CUDA_R_16F,
                                             (int)in_dim,
                                             &beta,
                                             out0->ptr,
                                             CUDA_R_32F,
                                             (int)out_dim,
                                             CUDA_R_32F,
                                             CUBLAS_GEMM_DEFAULT);
            if (!cublas_ok(st, "f16 pair matmul0")) return 0;
            st = cublasGemmEx(cuda_cublas_for_tier(logical_tier),
                              CUBLAS_OP_T,
                              CUBLAS_OP_N,
                              (int)out_dim,
                              (int)n_tok,
                              (int)in_dim,
                              &alpha,
                              w1,
                              CUDA_R_16F,
                              (int)in_dim,
                              xh,
                              CUDA_R_16F,
                              (int)in_dim,
                              &beta,
                              out1->ptr,
                              CUDA_R_32F,
                              (int)out_dim,
                              CUDA_R_32F,
                              CUBLAS_GEMM_DEFAULT);
            return cublas_ok(st, "f16 pair matmul1");
        }
        return ds4_gpu_matmul_f16_tensor(out0, model_map, model_size, weight0_offset,
                                           in_dim, out_dim, x, n_tok) &&
               ds4_gpu_matmul_f16_tensor(out1, model_map, model_size, weight1_offset,
                                           in_dim, out_dim, x, n_tok);
    }
    matmul_f16_pair_ordered_chunks_kernel<<<(unsigned)out_dim, 32>>>(
        (float *)out0->ptr,
        (float *)out1->ptr,
        w0,
        w1,
        (const float *)x->ptr,
        in_dim,
        out_dim,
        out_dim);
    return cuda_ok(cudaGetLastError(), "matmul_f16_pair_ordered_chunks launch");
}

extern "C" int ds4_gpu_matmul_f16_pair_compressor_store_tensor(
        ds4_gpu_tensor *out_kv,
        ds4_gpu_tensor *out_score,
        ds4_gpu_tensor *state_kv,
        ds4_gpu_tensor *state_score,
        const void *model_map,
        uint64_t model_size,
        uint64_t weight_kv_offset,
        uint64_t weight_score_offset,
        uint64_t ape_offset,
        uint32_t ape_type,
        uint64_t in_dim,
        uint32_t width,
        const ds4_gpu_tensor *x,
        uint32_t ratio,
        uint32_t pos) {
    (void)out_kv;
    (void)out_score;
    (void)state_kv;
    (void)state_score;
    (void)model_map;
    (void)model_size;
    (void)weight_kv_offset;
    (void)weight_score_offset;
    (void)ape_offset;
    (void)ape_type;
    (void)in_dim;
    (void)width;
    (void)x;
    (void)ratio;
    (void)pos;
    return 0;
}

extern "C" int ds4_gpu_matmul_f32_tensor(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim, const ds4_gpu_tensor *x, uint64_t n_tok) {
    if (!out || !x || !model_map || in_dim == 0 || out_dim == 0 || n_tok == 0) return 0;
    if (weight_offset > model_size || out_dim > UINT64_MAX / in_dim) return 0;
    uint64_t weight_elems = out_dim * in_dim;
    if (weight_elems > UINT64_MAX / sizeof(float)) return 0;
    uint64_t weight_bytes = weight_elems * sizeof(float);
    if (weight_bytes > model_size - weight_offset) return 0;
    if (x->bytes < n_tok * in_dim * sizeof(float) ||
        out->bytes < n_tok * out_dim * sizeof(float)) return 0;
    const int logical_tier = ds4_tensor_device_idx(out);
    const char *wptr = cuda_resolve_weight_ptr(model_map, weight_offset, weight_bytes, logical_tier, "f32");
    if (!wptr) return 0;
    const float *w = (const float *)wptr;
    if (g_cublas_ready && n_tok > 1) {
        const float alpha = 1.0f;
        const float beta = 0.0f;
        cublasStatus_t st = cublasSgemm(cuda_cublas_for_tier(logical_tier),
                                        CUBLAS_OP_T,
                                        CUBLAS_OP_N,
                                        (int)out_dim,
                                        (int)n_tok,
                                        (int)in_dim,
                                        &alpha,
                                        w,
                                        (int)in_dim,
                                        (const float *)x->ptr,
                                        (int)in_dim,
                                        &beta,
                                        (float *)out->ptr,
                                        (int)out_dim);
        return cublas_ok(st, "f32 matmul");
    }
    dim3 grid((unsigned)out_dim, (unsigned)n_tok, 1);
    matmul_f32_kernel<<<grid, 256>>>((float *)out->ptr, w, (const float *)x->ptr, in_dim, out_dim, n_tok);
    return cuda_ok(cudaGetLastError(), "matmul_f32 launch");
}

extern "C" int ds4_gpu_repeat_hc_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *row, uint32_t n_embd, uint32_t n_hc) {
    if (!out || !row || n_embd == 0 || n_hc == 0 ||
        row->bytes < (uint64_t)n_embd * sizeof(float) ||
        out->bytes < (uint64_t)n_embd * n_hc * sizeof(float)) {
        return 0;
    }
    uint64_t n = (uint64_t)n_embd * n_hc;
    repeat_hc_kernel<<<(n + 255) / 256, 256>>>((float *)out->ptr, (const float *)row->ptr, n_embd, n_hc);
    return cuda_ok(cudaGetLastError(), "repeat_hc launch");
}


/* Non-causal batch attention over a raw KV ring for the DSpark draft block.
 * Every query row attends over all n_raw visible rows plus the per-head sink,
 * with the same exact one-block max/denominator/value accumulation order as
 * the reference decode attention (scores in shared, sequential value pass). */
__global__ static void attention_noncausal_raw_batch_heads_kernel(
        float *heads,
        const float *sinks,
        const float *q,
        const float *raw_kv,
        uint32_t n_tokens,
        uint32_t n_raw,
        uint32_t raw_cap,
        uint32_t raw_start,
        uint32_t n_head,
        uint32_t head_dim) {
    const uint32_t tok = blockIdx.x;
    const uint32_t h = blockIdx.y;
    if (tok >= n_tokens || h >= n_head) return;
    extern __shared__ float sh_scores[]; /* n_raw floats */
    const float *qh = q + ((uint64_t)tok * n_head + h) * head_dim;
    const float scale = rsqrtf((float)head_dim);
    for (uint32_t r = threadIdx.x; r < n_raw; r += blockDim.x) {
        const uint32_t row = (raw_start + r) % raw_cap;
        const float *kv = raw_kv + (uint64_t)row * head_dim;
        float dot = 0.0f;
        for (uint32_t d = 0; d < head_dim; d++) dot += qh[d] * kv[d];
        sh_scores[r] = dot * scale;
    }
    __syncthreads();
    __shared__ float partial[256];
    __shared__ float max_s;
    __shared__ float denom;
    float local_max = sinks[h];
    for (uint32_t r = threadIdx.x; r < n_raw; r += blockDim.x) {
        local_max = fmaxf(local_max, sh_scores[r]);
    }
    partial[threadIdx.x] = local_max;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1u; stride > 0u; stride >>= 1u) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] = fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) max_s = partial[0];
    __syncthreads();
    float den_local = 0.0f;
    for (uint32_t r = threadIdx.x; r < n_raw; r += blockDim.x) {
        sh_scores[r] = expf(sh_scores[r] - max_s);
        den_local += sh_scores[r];
    }
    partial[threadIdx.x] = den_local;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1u; stride > 0u; stride >>= 1u) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) denom = partial[0] + expf(sinks[h] - max_s);
    __syncthreads();
    float *oh = heads + ((uint64_t)tok * n_head + h) * head_dim;
    for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (uint32_t r = 0; r < n_raw; r++) {
            const uint32_t row = (raw_start + r) % raw_cap;
            acc += raw_kv[(uint64_t)row * head_dim + d] * sh_scores[r];
        }
        oh[d] = acc / denom;
    }
}

extern "C" int ds4_gpu_attention_noncausal_raw_batch_heads_tensor(
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
    if (!heads || !q || !raw_kv || !model_map ||
        n_tokens == 0 || n_raw == 0 || raw_cap < n_raw ||
        raw_start >= raw_cap || n_head == 0 || head_dim == 0 ||
        sinks_offset > model_size ||
        (uint64_t)n_head * sizeof(float) > model_size - sinks_offset ||
        heads->bytes < (uint64_t)n_tokens * n_head * head_dim * sizeof(float) ||
        q->bytes < (uint64_t)n_tokens * n_head * head_dim * sizeof(float) ||
        raw_kv->bytes < (uint64_t)raw_cap * head_dim * sizeof(float)) {
        return 0;
    }
    const int logical_tier = ds4_tensor_device_idx(heads);
    const float *sinks = (const float *)cuda_resolve_weight_ptr(
            model_map, sinks_offset, (uint64_t)n_head * sizeof(float), logical_tier,
            "dspark_attn_sinks");
    if (!sinks) return 0;
    const size_t shmem = (size_t)n_raw * sizeof(float);
    if (shmem > 32768) return 0; /* draft blocks are tiny; guard anyway */
    dim3 grid(n_tokens, n_head, 1);
    attention_noncausal_raw_batch_heads_kernel<<<grid, 256, shmem>>>(
            (float *)heads->ptr,
            sinks,
            (const float *)q->ptr,
            (const float *)raw_kv->ptr,
            n_tokens, n_raw, raw_cap, raw_start, n_head, head_dim);
    if (!cuda_ok(cudaGetLastError(), "attention noncausal raw batch heads launch")) return 0;
    static int verify_left = -1;
    if (verify_left < 0) {
        verify_left = getenv("DS4_DSPARK_VERIFY_NONCAUSAL") != NULL ? 3 : 0;
    }
    if (verify_left > 0) {
        verify_left--;
        (void)cudaDeviceSynchronize();
        const uint64_t qn = (uint64_t)n_tokens * n_head * head_dim;
        const uint64_t kn = (uint64_t)raw_cap * head_dim;
        std::vector<float> hq(qn), hkv(kn), hout(qn), hsink(n_head);
        (void)cudaMemcpy(hq.data(), q->ptr, qn * 4, cudaMemcpyDeviceToHost);
        (void)cudaMemcpy(hkv.data(), raw_kv->ptr, kn * 4, cudaMemcpyDeviceToHost);
        (void)cudaMemcpy(hout.data(), heads->ptr, qn * 4, cudaMemcpyDeviceToHost);
        (void)cudaMemcpy(hsink.data(), sinks, (uint64_t)n_head * 4, cudaMemcpyDeviceToHost);
        double max_abs = 0.0, max_rel = 0.0;
        const double scale = 1.0 / sqrt((double)head_dim);
        for (uint32_t t = 0; t < n_tokens; t++) {
            for (uint32_t h = 0; h < n_head; h++) {
                std::vector<double> sc(n_raw);
                double mx = (double)hsink[h];
                for (uint32_t r = 0; r < n_raw; r++) {
                    const uint32_t row = (raw_start + r) % raw_cap;
                    double dot = 0.0;
                    for (uint32_t d = 0; d < head_dim; d++) {
                        dot += (double)hq[((uint64_t)t * n_head + h) * head_dim + d] *
                               (double)hkv[(uint64_t)row * head_dim + d];
                    }
                    sc[r] = dot * scale;
                    if (sc[r] > mx) mx = sc[r];
                }
                double den = exp((double)hsink[h] - mx);
                for (uint32_t r = 0; r < n_raw; r++) den += exp(sc[r] - mx);
                for (uint32_t d = 0; d < head_dim; d++) {
                    double acc = 0.0;
                    for (uint32_t r = 0; r < n_raw; r++) {
                        const uint32_t row = (raw_start + r) % raw_cap;
                        acc += exp(sc[r] - mx) * (double)hkv[(uint64_t)row * head_dim + d];
                    }
                    const double ref = acc / den;
                    const double got = (double)hout[((uint64_t)t * n_head + h) * head_dim + d];
                    const double ad = fabs(ref - got);
                    if (ad > max_abs) max_abs = ad;
                    if (fabs(ref) > 1e-3 && ad / fabs(ref) > max_rel) max_rel = ad / fabs(ref);
                }
            }
        }
        fprintf(stderr,
                "ds4: DSpark noncausal verify n_tok=%u n_raw=%u start=%u cap=%u "
                "max_abs=%.3e max_rel=%.3e\n",
                n_tokens, n_raw, raw_start, raw_cap, max_abs, max_rel);
    }
    return 1;
}

extern "C" int ds4_gpu_repeat_hc_rows_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *rows, uint32_t n_tokens, uint32_t n_embd, uint32_t n_hc) {
    uint64_t rows_elems = 0;
    uint64_t out_elems = 0;
    if (!out || !rows || n_tokens == 0 || n_embd == 0 || n_hc == 0 ||
        (uint64_t)n_tokens > UINT64_MAX / n_embd ||
        (rows_elems = (uint64_t)n_tokens * n_embd) > UINT64_MAX / n_hc ||
        (out_elems = rows_elems * n_hc) > UINT64_MAX / sizeof(float) ||
        rows_elems > UINT64_MAX / sizeof(float) ||
        rows->bytes < rows_elems * sizeof(float) ||
        out->bytes < out_elems * sizeof(float)) {
        return 0;
    }
    const uint64_t blocks = (out_elems + 255u) / 256u;
    if (blocks > UINT32_MAX) return 0;
    repeat_hc_rows_kernel<<<(unsigned)blocks, 256>>>((float *)out->ptr, (const float *)rows->ptr, n_tokens, n_embd, n_hc);
    return cuda_ok(cudaGetLastError(), "repeat_hc_rows launch");
}

extern "C" int ds4_gpu_rms_norm_plain_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *x, uint32_t n, float eps) {
    if (!out || !x || out->bytes < (uint64_t)n * sizeof(float) ||
        x->bytes < (uint64_t)n * sizeof(float)) return 0;
    if (n == 4096u) {
        rms_norm_plain_fast4096_kernel<<<1, 256>>>((float *)out->ptr, (const float *)x->ptr, n, 1, eps);
    } else if ((n & 2047u) == 0u) {
        rms_norm_plain_batch8_kernel<<<1, 256>>>((float *)out->ptr, (const float *)x->ptr, n, 1, eps);
    } else {
        rms_norm_plain_kernel<<<1, 256>>>((float *)out->ptr, (const float *)x->ptr, n, 1, eps);
    }
    return cuda_ok(cudaGetLastError(), "rms_norm_plain launch");
}
extern "C" int ds4_gpu_rms_norm_plain_rows_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *x, uint32_t n, uint32_t rows, float eps) {
    if (!out || !x || out->bytes < (uint64_t)n * rows * sizeof(float) ||
        x->bytes < (uint64_t)n * rows * sizeof(float)) return 0;
    if (n == 4096u) {
        rms_norm_plain_fast4096_kernel<<<rows, 256>>>((float *)out->ptr, (const float *)x->ptr, n, rows, eps);
    } else if ((n & 2047u) == 0u) {
        rms_norm_plain_batch8_kernel<<<rows, 256>>>((float *)out->ptr, (const float *)x->ptr, n, rows, eps);
    } else {
        rms_norm_plain_kernel<<<rows, 256>>>((float *)out->ptr, (const float *)x->ptr, n, rows, eps);
    }
    return cuda_ok(cudaGetLastError(), "rms_norm_plain launch");
}
extern "C" int ds4_gpu_rms_norm_weight_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *x, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint32_t n, float eps) {
    if (!out || !x || !model_map || weight_offset > model_size ||
        model_size - weight_offset < (uint64_t)n * sizeof(float) ||
        out->bytes < (uint64_t)n * sizeof(float) ||
        x->bytes < (uint64_t)n * sizeof(float)) return 0;
    const int logical_tier = ds4_tensor_device_idx(out);
    const char *wptr = cuda_resolve_weight_ptr(model_map, weight_offset, (uint64_t)n * sizeof(float), logical_tier, "rms_weight");
    if (!wptr) return 0;
    const float *w = (const float *)wptr;
    rms_norm_weight_kernel<<<1, 256>>>((float *)out->ptr, (const float *)x->ptr, w, n, 1, eps);
    return cuda_ok(cudaGetLastError(), "rms_norm_weight launch");
}
extern "C" int ds4_gpu_rms_norm_weight_rows_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *x, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint32_t n, uint32_t rows, float eps) {
    if (!out || !x || !model_map || weight_offset > model_size ||
        model_size - weight_offset < (uint64_t)n * sizeof(float) ||
        out->bytes < (uint64_t)n * rows * sizeof(float) ||
        x->bytes < (uint64_t)n * rows * sizeof(float)) return 0;
    const int logical_tier = ds4_tensor_device_idx(out);
    const char *wptr = cuda_resolve_weight_ptr(model_map, weight_offset, (uint64_t)n * sizeof(float), logical_tier, "rms_weight");
    if (!wptr) return 0;
    const float *w = (const float *)wptr;
    rms_norm_weight_kernel<<<rows, 256>>>((float *)out->ptr, (const float *)x->ptr, w, n, rows, eps);
    return cuda_ok(cudaGetLastError(), "rms_norm_weight launch");
}
extern "C" int ds4_gpu_dsv4_qkv_rms_norm_rows_tensor(
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
    if (!g_cuda_disable_qkv_rms_fused) {
        if (!q_out || !q || !kv_out || !kv || !model_map ||
            q_weight_offset > model_size ||
            kv_weight_offset > model_size ||
            model_size - q_weight_offset < (uint64_t)q_n * sizeof(float) ||
            model_size - kv_weight_offset < (uint64_t)kv_n * sizeof(float) ||
            q_out->bytes < (uint64_t)q_n * rows * sizeof(float) ||
            q->bytes < (uint64_t)q_n * rows * sizeof(float) ||
            kv_out->bytes < (uint64_t)kv_n * rows * sizeof(float) ||
            kv->bytes < (uint64_t)kv_n * rows * sizeof(float)) {
            return 0;
        }
        const int logical_tier = ds4_tensor_device_idx(q_out);
        const float *q_w = (const float *)cuda_resolve_weight_ptr(model_map,
                q_weight_offset, (uint64_t)q_n * sizeof(float), logical_tier, "q_rms_weight");
        const float *kv_w = (const float *)cuda_resolve_weight_ptr(model_map,
                kv_weight_offset, (uint64_t)kv_n * sizeof(float), logical_tier, "kv_rms_weight");
        if (!q_w || !kv_w) return 0;
        dim3 grid(rows, 2u, 1u);
        dsv4_qkv_rms_norm_rows_kernel<<<grid, 256>>>(
                (float *)q_out->ptr,
                (const float *)q->ptr,
                q_w,
                q_n,
                (float *)kv_out->ptr,
                (const float *)kv->ptr,
                kv_w,
                kv_n,
                rows,
                eps);
        return cuda_ok(cudaGetLastError(), "dsv4 qkv rms norm rows launch");
    }
    return ds4_gpu_rms_norm_weight_rows_tensor(q_out, q, model_map, model_size,
                                                 q_weight_offset, q_n, rows, eps) &&
           ds4_gpu_rms_norm_weight_rows_tensor(kv_out, kv, model_map, model_size,
                                                 kv_weight_offset, kv_n, rows, eps);
}

extern "C" int ds4_gpu_dsv4_qkv_rms_norm_rows_kv_rope_tensor(
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
        uint32_t                kv_n_head,
        uint32_t                kv_head_dim,
        uint32_t                n_rot,
        uint32_t                pos0,
        uint32_t                n_ctx_orig,
        bool                    inverse,
        float                   freq_base,
        float                   freq_scale,
        float                   ext_factor,
        float                   attn_factor,
        float                   beta_fast,
        float                   beta_slow,
        float                   eps) {
    if (g_cuda_disable_qkv_rms_fused) return 0;
    if (!q_out || !q || !kv_out || !kv || !model_map ||
        q_weight_offset > model_size ||
        kv_weight_offset > model_size ||
        kv_n_head == 0 || kv_head_dim == 0 ||
        n_rot > kv_head_dim || (n_rot & 1u) ||
        kv_n != kv_n_head * kv_head_dim ||
        model_size - q_weight_offset < (uint64_t)q_n * sizeof(float) ||
        model_size - kv_weight_offset < (uint64_t)kv_n * sizeof(float) ||
        q_out->bytes < (uint64_t)q_n * rows * sizeof(float) ||
        q->bytes < (uint64_t)q_n * rows * sizeof(float) ||
        kv_out->bytes < (uint64_t)kv_n * rows * sizeof(float) ||
        kv->bytes < (uint64_t)kv_n * rows * sizeof(float)) {
        return 0;
    }
    const int logical_tier = ds4_tensor_device_idx(q_out);
    const float *q_w = (const float *)cuda_resolve_weight_ptr(model_map,
            q_weight_offset, (uint64_t)q_n * sizeof(float), logical_tier, "q_rms_weight");
    const float *kv_w = (const float *)cuda_resolve_weight_ptr(model_map,
            kv_weight_offset, (uint64_t)kv_n * sizeof(float), logical_tier, "kv_rms_weight");
    if (!q_w || !kv_w) return 0;
    dim3 grid(rows, 2u, 1u);
    dsv4_qkv_rms_norm_rows_kv_rope_kernel<<<grid, 256>>>(
            (float *)q_out->ptr,
            (const float *)q->ptr,
            q_w,
            q_n,
            (float *)kv_out->ptr,
            (const float *)kv->ptr,
            kv_w,
            kv_n,
            rows,
            kv_n_head,
            kv_head_dim,
            n_rot,
            pos0,
            n_ctx_orig,
            inverse ? 1 : 0,
            freq_base,
            freq_scale,
            ext_factor,
            attn_factor,
            beta_fast,
            beta_slow,
            eps);
    return cuda_ok(cudaGetLastError(), "dsv4 qkv rms norm kv rope launch");
}

extern "C" int ds4_gpu_head_rms_norm_tensor(ds4_gpu_tensor *x, uint32_t n_tok, uint32_t n_head, uint32_t head_dim, float eps) {
    if (!x || x->bytes < (uint64_t)n_tok * n_head * head_dim * sizeof(float)) return 0;
    head_rms_norm_kernel<<<n_tok * n_head, 256>>>((float *)x->ptr, n_tok, n_head, head_dim, eps);
    return cuda_ok(cudaGetLastError(), "head_rms_norm launch");
}
extern "C" int ds4_gpu_head_rms_norm_rope_tail_tensor(ds4_gpu_tensor *x, uint32_t n_tok, uint32_t n_head, uint32_t head_dim, uint32_t n_rot, uint32_t pos0, uint32_t n_ctx_orig, bool inverse, float freq_base, float freq_scale, float ext_factor, float attn_factor, float beta_fast, float beta_slow, float eps) {
    if (!x || n_rot > head_dim || (n_rot & 1u) ||
        x->bytes < (uint64_t)n_tok * n_head * head_dim * sizeof(float)) return 0;
    head_rms_norm_rope_tail_kernel<<<n_tok * n_head, 256>>>((float *)x->ptr, n_tok, n_head, head_dim, n_rot, pos0, n_ctx_orig, inverse ? 1 : 0, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow, eps);
    return cuda_ok(cudaGetLastError(), "head_rms_norm_rope_tail launch");
}
extern "C" int ds4_gpu_dsv4_fp8_kv_quantize_tensor(ds4_gpu_tensor *x, uint32_t n_tok, uint32_t head_dim, uint32_t n_rot) {
    if (!x || n_rot > head_dim || x->bytes < (uint64_t)n_tok * head_dim * sizeof(float)) return 0;
    fp8_kv_quantize_kernel<<<n_tok, 64>>>((float *)x->ptr, n_tok, head_dim, n_rot);
    return cuda_ok(cudaGetLastError(), "fp8_kv_quantize launch");
}
extern "C" int ds4_gpu_dsv4_indexer_qat_tensor(ds4_gpu_tensor *x, uint32_t n_rows, uint32_t head_dim) {
    if (!x || n_rows == 0 || head_dim != 128u ||
        x->bytes < (uint64_t)n_rows * head_dim * sizeof(float)) {
        return 0;
    }
    indexer_hadamard_fp4_kernel<<<n_rows, 128>>>((float *)x->ptr, n_rows, head_dim);
    return cuda_ok(cudaGetLastError(), "indexer_hadamard_fp4 launch");
}
extern "C" int ds4_gpu_rope_tail_tensor(ds4_gpu_tensor *x, uint32_t n_tok, uint32_t n_head, uint32_t head_dim, uint32_t n_rot, uint32_t pos0, uint32_t n_ctx_orig, bool inverse, float freq_base, float freq_scale, float ext_factor, float attn_factor, float beta_fast, float beta_slow) {
    if (!x || n_rot > head_dim || (n_rot & 1) || x->bytes < (uint64_t)n_tok * n_head * head_dim * sizeof(float)) return 0;
    uint32_t pairs = n_tok * n_head * (n_rot / 2);
    rope_tail_kernel<<<(pairs + 255) / 256, 256>>>((float *)x->ptr, n_tok, n_head, head_dim, n_rot, pos0, 1, n_ctx_orig, inverse ? 1 : 0, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
    return cuda_ok(cudaGetLastError(), "rope_tail launch");
}
extern "C" int ds4_gpu_rope_tail_decode_rows_tensor(
        ds4_gpu_tensor                       *x,
        const ds4_gpu_attention_decode_row   *rows,
        uint32_t                              n_rows,
        uint32_t                              n_head,
        uint32_t                              head_dim,
        uint32_t                              n_rot,
        uint32_t                              n_ctx_orig,
        bool                                  inverse,
        float                                 freq_base,
        float                                 freq_scale,
        float                                 ext_factor,
        float                                 attn_factor,
        float                                 beta_fast,
        float                                 beta_slow) {
    if (!x || !rows || n_rows == 0u ||
        n_rows > DS4_GPU_ATTENTION_DECODE_BATCH_MAX || n_head == 0u ||
        n_rot == 0u || n_rot > head_dim || (n_rot & 1u) != 0u ||
        x->bytes < (uint64_t)n_rows * n_head * head_dim * sizeof(float)) {
        return 0;
    }
    cuda_attention_decode_row_table table;
    memset(&table, 0, sizeof(table));
    for (uint32_t i = 0; i < n_rows; i++) table.row[i].pos = rows[i].pos;
    const uint32_t pairs = n_rows * n_head * (n_rot / 2u);
    rope_tail_decode_rows_kernel<<<(pairs + 255u) / 256u, 256>>>(
            (float *)x->ptr, table, n_rows, n_head, head_dim, n_rot,
            n_ctx_orig, inverse ? 1 : 0, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
    return cuda_ok(cudaGetLastError(), "rope tail decode rows launch");
}
extern "C" int ds4_gpu_store_raw_kv_tensor(ds4_gpu_tensor *raw_cache, const ds4_gpu_tensor *kv, uint32_t raw_cap, uint32_t row, uint32_t head_dim);
extern "C" int ds4_gpu_kv_fp8_store_raw_tensor(
        ds4_gpu_tensor *kv,
        ds4_gpu_tensor *raw_cache,
        uint32_t          raw_cap,
        uint32_t          raw_row,
        uint32_t          head_dim,
        uint32_t          n_rot) {
    if (!kv || !raw_cache || raw_cap == 0u || n_rot > head_dim ||
        kv->device_id != raw_cache->device_id ||
        kv->bytes < (uint64_t)head_dim * sizeof(float) ||
        raw_cache->bytes < (uint64_t)raw_cap * head_dim * sizeof(float)) {
        return 0;
    }
    cuda_attention_decode_row_table table;
    memset(&table, 0, sizeof(table));
    table.row[0].raw_kv = (uint64_t)(uintptr_t)raw_cache->ptr;
    table.row[0].raw_cap = raw_cap;
    table.row[0].raw_start = raw_row % raw_cap;
    fp8_kv_quantize_store_rows_kernel<<<1, 64>>>(
            (float *)kv->ptr, table, 1u, head_dim, n_rot);
    return cuda_ok(cudaGetLastError(), "fp8 KV quantize/store launch");
}

extern "C" int ds4_gpu_kv_fp8_store_raw_decode_rows_tensor(
        ds4_gpu_tensor        *kv,
        ds4_gpu_tensor *const *raw_caches,
        const uint32_t        *raw_caps,
        const uint32_t        *raw_rows,
        uint32_t               n_rows,
        uint32_t               head_dim,
        uint32_t               n_rot) {
    if (!kv || !raw_caches || !raw_caps || !raw_rows || n_rows == 0u ||
        n_rows > DS4_GPU_ATTENTION_DECODE_BATCH_MAX || n_rot > head_dim ||
        kv->bytes < (uint64_t)n_rows * head_dim * sizeof(float)) {
        return 0;
    }
    cuda_attention_decode_row_table table;
    memset(&table, 0, sizeof(table));
    for (uint32_t i = 0; i < n_rows; i++) {
        const ds4_gpu_tensor *raw = raw_caches[i];
        if (!raw || raw_caps[i] == 0u || raw_rows[i] >= raw_caps[i] ||
            raw->device_id != kv->device_id ||
            raw->bytes < (uint64_t)raw_caps[i] * head_dim * sizeof(float)) {
            return 0;
        }
        table.row[i].raw_kv = (uint64_t)(uintptr_t)raw->ptr;
        table.row[i].raw_cap = raw_caps[i];
        table.row[i].raw_start = raw_rows[i];
    }
    fp8_kv_quantize_store_rows_kernel<<<n_rows, 64>>>(
            (float *)kv->ptr, table, n_rows, head_dim, n_rot);
    return cuda_ok(cudaGetLastError(), "fp8 KV quantize/store rows launch");
}
extern "C" int ds4_gpu_store_raw_kv_tensor(ds4_gpu_tensor *raw_cache, const ds4_gpu_tensor *kv, uint32_t raw_cap, uint32_t row, uint32_t head_dim) {
    if (!raw_cache || !kv || raw_cap == 0 ||
        raw_cache->bytes < (uint64_t)raw_cap * head_dim * sizeof(float) ||
        kv->bytes < (uint64_t)head_dim * sizeof(float)) return 0;
    store_raw_kv_batch_kernel<<<(head_dim + 255) / 256, 256>>>((float *)raw_cache->ptr, (const float *)kv->ptr, raw_cap, row, 1, head_dim);
    return cuda_ok(cudaGetLastError(), "store_raw_kv launch");
}
extern "C" int ds4_gpu_store_raw_kv_batch_tensor(ds4_gpu_tensor *raw_cache, const ds4_gpu_tensor *kv, uint32_t raw_cap, uint32_t pos0, uint32_t n_tokens, uint32_t head_dim) {
    if (!raw_cache || !kv || raw_cap == 0 ||
        raw_cache->bytes < (uint64_t)raw_cap * head_dim * sizeof(float) ||
        kv->bytes < (uint64_t)n_tokens * head_dim * sizeof(float)) return 0;
    uint64_t n = (uint64_t)n_tokens * head_dim;
    store_raw_kv_batch_kernel<<<(n + 255) / 256, 256>>>((float *)raw_cache->ptr, (const float *)kv->ptr, raw_cap, pos0, n_tokens, head_dim);
    return cuda_ok(cudaGetLastError(), "store_raw_kv_batch launch");
}
extern "C" int ds4_gpu_compressor_store_batch_tensor(
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
    if (!kv || !sc || !state_kv || !state_score || !model_map ||
        head_dim == 0 || ratio == 0 || n_tokens == 0 ||
        (ape_type != 0u && ape_type != 1u)) {
        return 0;
    }
    const uint32_t coff = ratio == 4u ? 2u : 1u;
    const uint32_t width = coff * head_dim;
    const uint32_t state_rows = coff * ratio;
    const uint64_t elem_ape = ape_type == 1u ? 2u : 4u;
    const uint64_t kv_bytes = (uint64_t)n_tokens * width * sizeof(float);
    const uint64_t state_bytes = (uint64_t)state_rows * width * sizeof(float);
    const uint64_t ape_bytes = (uint64_t)width * ratio * elem_ape;
    if (ape_offset > model_size || ape_bytes > model_size - ape_offset ||
        kv->bytes < kv_bytes || sc->bytes < kv_bytes ||
        state_kv->bytes < state_bytes || state_score->bytes < state_bytes) {
        return 0;
    }
    const int logical_tier = ds4_tensor_device_idx(state_kv);
    const char *ape = cuda_resolve_weight_ptr(model_map, ape_offset, ape_bytes, logical_tier, "compressor_ape");
    if (!ape) return 0;
    uint64_t n = (uint64_t)n_tokens * width;
    compressor_store_kernel<<<(n + 255) / 256, 256>>>(
            (const float *)kv->ptr,
            (const float *)sc->ptr,
            (float *)state_kv->ptr,
            (float *)state_score->ptr,
            ape,
            0,
            ape_type,
            head_dim,
            ratio,
            pos0,
            n_tokens);
    return cuda_ok(cudaGetLastError(), "compressor store launch");
}

extern "C" int ds4_gpu_compressor_update_tensor(
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
    if (!kv_cur || !sc_cur || !state_kv || !state_score || !comp_cache ||
        !model_map || head_dim == 0 || ratio == 0 ||
        n_rot > head_dim || (n_rot & 1u) != 0 ||
        (ape_type != 0u && ape_type != 1u) || norm_type != 0u) {
        return 0;
    }
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
        norm_offset > model_size || norm_bytes > model_size - norm_offset ||
        kv_cur->bytes < kv_bytes || sc_cur->bytes < kv_bytes ||
        state_kv->bytes < state_bytes || state_score->bytes < state_bytes ||
        (emit && comp_cache->bytes < comp_bytes)) {
        return 0;
    }
    if (!state_already_stored) {
        if (!ds4_gpu_compressor_store_batch_tensor(kv_cur, sc_cur, state_kv, state_score,
                                                     model_map, model_size, ape_offset, ape_type,
                                                     head_dim, ratio, pos, 1)) {
            return 0;
        }
    }
    if (!emit) return 1;
    ds4_gpu_tensor *comp_row_view = ds4_gpu_tensor_view(
            comp_cache,
            (uint64_t)comp_row * head_dim * sizeof(float),
            (uint64_t)head_dim * sizeof(float));
    if (!comp_row_view) return 0;
    compressor_update_pool_kernel<<<(head_dim + 255) / 256, 256>>>(
            (float *)comp_row_view->ptr,
            (const float *)state_kv->ptr,
            (const float *)state_score->ptr,
            head_dim,
            ratio);
    int ok = cuda_ok(cudaGetLastError(), "compressor update pool launch");
    if (ok) ok = ds4_gpu_rms_norm_weight_rows_tensor(comp_row_view, comp_row_view,
                                                       model_map, model_size, norm_offset,
                                                       head_dim, 1, rms_eps);
    if (ok) ok = ds4_gpu_rope_tail_tensor(comp_row_view, 1, 1, head_dim, n_rot,
                                            pos + 1u - ratio, n_ctx_orig, false,
                                            freq_base, freq_scale, ext_factor, attn_factor,
                                            beta_fast, beta_slow);
    ds4_gpu_tensor_free(comp_row_view);
    if (ok && ratio == 4u) {
        uint64_t half = 4ull * width;
        compressor_shift_ratio4_kernel<<<(half + 255) / 256, 256>>>(
                (float *)state_kv->ptr, (float *)state_score->ptr, width);
        ok = cuda_ok(cudaGetLastError(), "compressor ratio4 shift launch");
    }
    return ok;
}
extern "C" int ds4_gpu_compressor_prefill_tensor(
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
    if (!comp_cache || !state_kv || !state_score || !kv || !sc || !model_map ||
        head_dim == 0 || ratio == 0 || n_tokens == 0 ||
        n_rot > head_dim || (n_rot & 1u) != 0 ||
        (ape_type != 0u && ape_type != 1u) || norm_type != 0u) {
        return 0;
    }

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
        norm_offset > model_size || norm_bytes > model_size - norm_offset ||
        kv->bytes < kv_bytes || sc->bytes < kv_bytes ||
        state_kv->bytes < state_bytes || state_score->bytes < state_bytes ||
        (n_comp && comp_cache->bytes < comp_bytes)) {
        return 0;
    }
    const int logical_tier = ds4_tensor_device_idx(state_kv);
    const char *ape = cuda_resolve_weight_ptr(model_map, ape_offset, ape_bytes, logical_tier, "compressor_ape");
    if (!ape) return 0;

    uint64_t state_n = (uint64_t)state_rows * width;
    if (!cuda_ok(cudaMemsetAsync(state_kv->ptr, 0, (size_t)(state_n * sizeof(float))),
                 "compressor state kv zero")) return 0;
    fill_f32_kernel<<<(state_n + 255) / 256, 256>>>((float *)state_score->ptr, state_n, -INFINITY);
    if (!cuda_ok(cudaGetLastError(), "compressor state score fill launch")) return 0;

    if (ratio == 4u) {
        if (cutoff >= ratio) {
            uint32_t prev_start = cutoff - ratio;
            uint64_t n = (uint64_t)ratio * width;
            compressor_set_rows_kernel<<<(n + 255) / 256, 256>>>(
                    (float *)state_kv->ptr, (float *)state_score->ptr,
                    (const float *)kv->ptr, (const float *)sc->ptr,
                    ape, 0, ape_type, width, ratio, pos0,
                    prev_start, 0, ratio);
            if (!cuda_ok(cudaGetLastError(), "compressor prefill prev state launch")) return 0;
        }
        if (rem != 0) {
            uint64_t n = (uint64_t)rem * width;
            compressor_set_rows_kernel<<<(n + 255) / 256, 256>>>(
                    (float *)state_kv->ptr, (float *)state_score->ptr,
                    (const float *)kv->ptr, (const float *)sc->ptr,
                    ape, 0, ape_type, width, ratio, pos0,
                    cutoff, ratio, rem);
            if (!cuda_ok(cudaGetLastError(), "compressor prefill rem state launch")) return 0;
        }
    } else if (rem != 0) {
        uint64_t n = (uint64_t)rem * width;
        compressor_set_rows_kernel<<<(n + 255) / 256, 256>>>(
                (float *)state_kv->ptr, (float *)state_score->ptr,
                (const float *)kv->ptr, (const float *)sc->ptr,
                ape, 0, ape_type, width, ratio, pos0,
                cutoff, 0, rem);
        if (!cuda_ok(cudaGetLastError(), "compressor prefill rem state launch")) return 0;
    }
    if (n_comp != 0) {
        dim3 grid((head_dim + 255) / 256, n_comp, 1);
        compressor_prefill_pool_kernel<<<grid, 256>>>(
                (float *)comp_cache->ptr,
                (const float *)kv->ptr,
                (const float *)sc->ptr,
                (const float *)state_kv->ptr,
                (const float *)state_score->ptr,
                ape, 0, ape_type, head_dim, ratio, pos0, n_comp, 0);
        if (!cuda_ok(cudaGetLastError(), "compressor prefill pool launch")) return 0;
        if (!ds4_gpu_rms_norm_weight_rows_tensor(comp_cache, comp_cache,
                                                   model_map, model_size, norm_offset,
                                                   head_dim, n_comp, rms_eps)) return 0;
        if (n_rot != 0) {
            const uint32_t pairs = n_comp * (n_rot / 2u);
            rope_tail_kernel<<<(pairs + 255) / 256, 256>>>(
                    (float *)comp_cache->ptr, n_comp, 1, head_dim, n_rot,
                    pos0, ratio, n_ctx_orig, 0, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow);
            if (!cuda_ok(cudaGetLastError(), "compressor prefill rope launch")) return 0;
        }
        if (quantize_fp8 && !ds4_gpu_dsv4_fp8_kv_quantize_tensor(comp_cache, n_comp, head_dim, n_rot)) return 0;
    }
    return 1;
}
extern "C" int ds4_gpu_compressor_prefill_ratio4_replay_tensor(
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
    if (!comp_cache || !state_kv || !state_score || !kv || !sc || !model_map ||
        head_dim == 0 || n_tokens == 0 || (n_tokens & 3u) != 0 || (pos0 & 3u) != 0 ||
        n_rot > head_dim || (n_rot & 1u) != 0 ||
        (ape_type != 0u && ape_type != 1u) || norm_type != 0u) {
        return 0;
    }

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
        norm_offset > model_size || norm_bytes > model_size - norm_offset ||
        kv->bytes < kv_bytes || sc->bytes < kv_bytes ||
        state_kv->bytes < state_bytes || state_score->bytes < state_bytes ||
        comp_cache->bytes < comp_bytes) {
        return 0;
    }
    const int logical_tier = ds4_tensor_device_idx(comp_cache);
    const char *ape = cuda_resolve_weight_ptr(model_map, ape_offset, ape_bytes, logical_tier, "compressor_ape");
    if (!ape) return 0;
    dim3 grid((head_dim + 255) / 256, n_comp, 1);
    compressor_prefill_pool_kernel<<<grid, 256>>>(
            (float *)comp_cache->ptr,
            (const float *)kv->ptr,
            (const float *)sc->ptr,
            (const float *)state_kv->ptr,
            (const float *)state_score->ptr,
            ape, 0, ape_type, head_dim, ratio, pos0, n_comp, 1);
    if (!cuda_ok(cudaGetLastError(), "compressor replay pool launch")) return 0;
    if (!ds4_gpu_rms_norm_weight_rows_tensor(comp_cache, comp_cache,
                                               model_map, model_size, norm_offset,
                                               head_dim, n_comp, rms_eps)) return 0;
    if (n_rot != 0) {
        const uint32_t pairs = n_comp * (n_rot / 2u);
        rope_tail_kernel<<<(pairs + 255) / 256, 256>>>(
                (float *)comp_cache->ptr, n_comp, 1, head_dim, n_rot,
                pos0, ratio, n_ctx_orig, 0, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow);
        if (!cuda_ok(cudaGetLastError(), "compressor replay rope launch")) return 0;
    }
    if (quantize_fp8 && !ds4_gpu_dsv4_fp8_kv_quantize_tensor(comp_cache, n_comp, head_dim, n_rot)) return 0;

    uint64_t state_n = (uint64_t)state_rows * width;
    if (!cuda_ok(cudaMemsetAsync(state_kv->ptr, 0, (size_t)(state_n * sizeof(float))),
                 "compressor replay state kv zero")) return 0;
    fill_f32_kernel<<<(state_n + 255) / 256, 256>>>((float *)state_score->ptr, state_n, -INFINITY);
    if (!cuda_ok(cudaGetLastError(), "compressor replay state score fill launch")) return 0;
    uint32_t prev_start = n_tokens - ratio;
    uint64_t n = (uint64_t)ratio * width;
    compressor_set_rows_kernel<<<(n + 255) / 256, 256>>>(
            (float *)state_kv->ptr, (float *)state_score->ptr,
            (const float *)kv->ptr, (const float *)sc->ptr,
            ape, 0, ape_type, width, ratio, pos0,
            prev_start, 0, ratio);
    return cuda_ok(cudaGetLastError(), "compressor replay state launch");
}
extern "C" int ds4_gpu_compressor_prefill_state_ratio4_tensor(
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
    if (!state_kv || !state_score || !kv_tail || !sc_tail || !model_map ||
        head_dim == 0 || (ape_type != 0u && ape_type != 1u)) {
        return 0;
    }
    const uint32_t ratio = 4u;
    const uint32_t width = 2u * head_dim;
    const uint32_t state_rows = 8u;
    const uint64_t elem_ape = ape_type == 1u ? 2u : 4u;
    const uint64_t tail_bytes = (uint64_t)ratio * width * sizeof(float);
    const uint64_t state_bytes = (uint64_t)state_rows * width * sizeof(float);
    const uint64_t ape_bytes = (uint64_t)ratio * width * elem_ape;
    if (ape_offset > model_size || ape_bytes > model_size - ape_offset ||
        kv_tail->bytes < tail_bytes || sc_tail->bytes < tail_bytes ||
        state_kv->bytes < state_bytes || state_score->bytes < state_bytes) {
        return 0;
    }
    const int logical_tier = ds4_tensor_device_idx(state_kv);
    const char *ape = cuda_resolve_weight_ptr(model_map, ape_offset, ape_bytes, logical_tier, "compressor_ape");
    if (!ape) return 0;
    uint64_t state_n = (uint64_t)state_rows * width;
    if (!cuda_ok(cudaMemsetAsync(state_kv->ptr, 0, (size_t)(state_n * sizeof(float))),
                 "compressor state kv zero")) return 0;
    fill_f32_kernel<<<(state_n + 255) / 256, 256>>>((float *)state_score->ptr, state_n, -INFINITY);
    if (!cuda_ok(cudaGetLastError(), "compressor state score fill launch")) return 0;
    uint64_t n = (uint64_t)ratio * width;
    compressor_set_rows_kernel<<<(n + 255) / 256, 256>>>(
            (float *)state_kv->ptr, (float *)state_score->ptr,
            (const float *)kv_tail->ptr, (const float *)sc_tail->ptr,
            ape, 0, ape_type, width, ratio, pos0,
            0, 0, ratio);
    return cuda_ok(cudaGetLastError(), "compressor state set launch");
}

/* perf-02 split-KV / flash-decode launch helper (opt-in, default OFF).
 * Returns 1 if the split path handled the launch, 0 if the caller should fall
 * through to the existing attention_decode_mixed_kernel path.
 *
 * Engages only for the single-token decode shape (n_tokens==1) and only when
 * DS4_CUDA_SPLITKV_DECODE is set. S==1 is NOT handled here: the caller dispatches
 * the old kernel as the bit-exact anchor when S would be 1.
 */
static int attention_decode_splitkv_launch(
        int logical_tier,
        float *heads,
        const float *sinks,
        const float *q,
        const float *raw_kv,
        const float *comp_kv,
        const float *comp_mask,
        uint32_t use_comp_mask,
        uint32_t pos0,
        uint32_t n_raw,
        uint32_t raw_cap,
        uint32_t raw_start,
        uint32_t n_comp,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_head,
        uint32_t head_dim) {
    /* n_tokens is fixed at 1 for the split path; compute the EXACT logical row
     * count the kernel will use (raw_count + visible_comp) so S is sized to the
     * real work. raw_count MUST apply the same window logic as the kernel /
     * reference, otherwise a true-S==1 case (e.g. ratio=1, window=1, n_raw>=2,
     * n_comp=0) could be over-estimated to S>1 and engage split-KV instead of
     * the bit-exact old-kernel anchor. The count is head-independent. */
    const bool single_all = (ratio == 0u);
    uint32_t qpos = pos0;            /* t==0, n_tokens==1 */
    uint32_t first_raw_pos = pos0 + 1u - n_raw;
    uint32_t visible_comp = single_all ? n_comp : (n_comp ? (qpos + 1u) / ratio : 0u);
    if (visible_comp > n_comp) visible_comp = n_comp;
    uint32_t raw_count = 0;
    uint32_t raw_first_idx = 0;
    if (n_raw != 0) {
        const uint32_t raw_last_pos = first_raw_pos + n_raw - 1u;
        if (single_all) {
            raw_count = n_raw > 256u ? 256u : n_raw;
        } else if (qpos >= first_raw_pos) {
            uint32_t lo = first_raw_pos;
            if (window != 0 && qpos + 1u > window) {
                const uint32_t wlo = qpos + 1u - window;
                if (wlo > lo) lo = wlo;
            }
            const uint32_t hi = qpos < raw_last_pos ? qpos : raw_last_pos;
            if (hi >= lo) {
                raw_first_idx = lo - first_raw_pos;
                raw_count = hi - lo + 1u;
                if (raw_count > 256u) raw_count = 256u;
            }
        }
    }
    uint32_t n_score = raw_count + visible_comp;
    if (n_score == 0u) return 0;     /* nothing to do; let old path handle it */
    const int manual_splitkv = cuda_env_flag_enabled("DS4_CUDA_SPLITKV_DECODE", 0);
    const uint32_t scoped_min_score =
        (g_decode_fast_attention && !manual_splitkv) ? 512u : 0u;
    uint32_t min_score = cuda_parse_u32_env_clamped("DS4_CUDA_SPLITKV_MIN_SCORE",
                                                    scoped_min_score, 0u,
                                                    DS4_CUDA_ATTENTION_SCORE_CAP,
                                                    NULL);
    if (n_score < min_score) return 0;
    /* S = clamp(ceil(n_score / CHUNK), 1, S_MAX); raise to S_FLOOR for short
     * context to fill more SMs, but never exceed n_score (no empty chunks).
     * Optional tuning knobs are guarded by min_needed so every block's chunk
     * still fits the fixed shared score buffer. */
    const uint32_t split_cap = DS4_CUDA_SPLITKV_SCORE_CAP;
    const uint32_t min_needed = (n_score + split_cap - 1u) / split_cap;
    uint32_t chunk = cuda_parse_u32_env_clamped("DS4_CUDA_SPLITKV_CHUNK",
                                                DS4_CUDA_SPLITKV_CHUNK,
                                                1u, split_cap, NULL);
    uint32_t s_floor = cuda_parse_u32_env_clamped("DS4_CUDA_SPLITKV_S_FLOOR",
                                                  DS4_CUDA_SPLITKV_S_FLOOR,
                                                  1u, DS4_CUDA_SPLITKV_S_MAX, NULL);
    uint32_t s_max = cuda_parse_u32_env_clamped("DS4_CUDA_SPLITKV_S_MAX",
                                                DS4_CUDA_SPLITKV_S_MAX,
                                                1u, DS4_CUDA_SPLITKV_S_MAX, NULL);
    int exact_present = 0;
    uint32_t S = cuda_parse_u32_env_clamped("DS4_CUDA_SPLITKV_S",
                                            0u, 1u, DS4_CUDA_SPLITKV_S_MAX,
                                            &exact_present);
    if (!exact_present) {
        S = (n_score + chunk - 1u) / chunk;
        if (S < s_floor) S = s_floor < n_score ? s_floor : n_score;
        if (S > s_max) S = s_max;
    }
    if (S < min_needed) S = min_needed;
    if (S > n_score) S = n_score;
    if (S <= 1u) return 0;           /* S==1: caller uses the old kernel anchor */
    if (cuda_env_flag_enabled("DS4_CUDA_SPLITKV_GLOBAL_SOFTMAX", 0)) {
        const uint64_t score_count = (uint64_t)n_head * n_score;
        const uint64_t score_bytes = score_count * sizeof(float);
        const uint64_t denom_offset = (score_bytes + 255u) & ~255ull;
        const uint64_t denom_bytes = (uint64_t)n_head * sizeof(float);
        const uint64_t partial_offset = (denom_offset + denom_bytes + 255u) & ~255ull;
        const uint64_t partial_bytes = (uint64_t)n_head * S * head_dim * sizeof(float);
        void *tmp = cuda_tmp_alloc_on(logical_tier,
                                      partial_offset + partial_bytes,
                                      "attention splitkv global softmax");
        if (!tmp) return 0;
        float *scores = (float *)tmp;
        float *denom = (float *)((char *)tmp + denom_offset);
        float *partials = (float *)((char *)tmp + partial_offset);
        dim3 score_grid(1, n_head, S);
        attention_decode_score_split_scores_kernel<<<score_grid, 256>>>(
                scores, q, raw_kv, comp_kv, comp_mask, use_comp_mask,
                pos0, n_raw, raw_cap, raw_start, n_comp, window, ratio,
                n_head, head_dim, S);
        if (!cuda_ok(cudaGetLastError(), "attention splitkv global score launch")) return -1;
        attention_decode_global_softmax_kernel<<<n_head, 256>>>(
                scores, denom, sinks, n_score, n_head);
        if (!cuda_ok(cudaGetLastError(), "attention splitkv global softmax launch")) return -1;
        dim3 value_grid(1, n_head, S);
        attention_decode_split_value_kernel<<<value_grid, 256>>>(
                partials, scores, raw_kv, comp_kv, raw_count, raw_first_idx,
                raw_cap, raw_start, n_score, n_head, head_dim, S);
        if (!cuda_ok(cudaGetLastError(), "attention splitkv global value launch")) return -1;
        dim3 combine_grid(1, n_head, 1);
        attention_decode_split_value_combine_kernel<<<combine_grid, 256>>>(
                heads, partials, denom, n_head, head_dim, S);
        if (!cuda_ok(cudaGetLastError(), "attention splitkv global combine launch")) return -1;
        return 1;
    }
    /* Partials scratch: n_head * S * (head_dim + 2) floats (n_tokens==1). */
    uint64_t stride = (uint64_t)head_dim + 2u;
    uint64_t count = (uint64_t)n_head * S * stride;
    float *partials = (float *)cuda_tmp_alloc_on(logical_tier, count * sizeof(float),
                                                 "attention splitkv partials");
    if (!partials) return 0;
    dim3 split_grid(1, n_head, S);
    attention_decode_splitkv_kernel<<<split_grid, 256>>>(partials,
                                                         q,
                                                         raw_kv,
                                                         comp_kv,
                                                         comp_mask,
                                                         use_comp_mask,
                                                         1, pos0, n_raw, raw_cap, raw_start,
                                                         n_comp, window, ratio, n_head, head_dim, S);
    if (!cuda_ok(cudaGetLastError(), "attention splitkv partial launch")) return -1;
    dim3 combine_grid(1, n_head, 1);
    attention_decode_splitkv_combine_kernel<<<combine_grid, 256>>>(heads,
                                                                   sinks,
                                                                   partials,
                                                                   1, n_head, head_dim, S);
    if (!cuda_ok(cudaGetLastError(), "attention splitkv combine launch")) return -1;
    return 1;
}

extern "C" int ds4_gpu_attention_decode_heads_tensor(
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
    if (comp_kv_f16 ||
        !heads || !q || !raw_kv || !model_map || n_raw == 0 || raw_cap < n_raw ||
        raw_start >= raw_cap || (n_comp != 0 && !comp_kv) || (use_mask && !comp_mask) ||
        sinks_offset > model_size ||
        (uint64_t)n_head * sizeof(float) > model_size - sinks_offset ||
        heads->bytes < (uint64_t)n_head * head_dim * sizeof(float) ||
        q->bytes < (uint64_t)n_head * head_dim * sizeof(float) ||
        raw_kv->bytes < (uint64_t)raw_cap * head_dim * sizeof(float) ||
        (n_comp && comp_kv->bytes < (uint64_t)n_comp * head_dim * sizeof(float)) ||
        (use_mask && comp_mask->bytes < (uint64_t)n_comp * sizeof(float))) {
        return 0;
    }
    const int logical_tier = ds4_tensor_device_idx(heads);
    const float *sinks = (const float *)cuda_resolve_weight_ptr(
            model_map, sinks_offset, (uint64_t)n_head * sizeof(float), logical_tier, "attn_sinks");
    if (!sinks) return 0;
    if (!cuda_attention_score_buffer_fits(n_comp)) {
        if (!use_mask && head_dim == 512u &&
            !g_cuda_no_window_attention) {
            const uint32_t synthetic_pos0 = n_raw - 1u;
            dim3 online_grid(1, (n_head + 7u) / 8u, 1);
            attention_decode_mixed_heads8_online_kernel<<<online_grid, 256>>>((float *)heads->ptr,
                                                                              sinks,
                                                                              (const float *)q->ptr,
                                                                              (const float *)raw_kv->ptr,
                                                                              n_comp ? (const float *)comp_kv->ptr : (const float *)raw_kv->ptr,
                                                                              1,
                                                                              synthetic_pos0,
                                                                              n_raw,
                                                                              raw_cap,
                                                                              raw_start,
                                                                              n_comp,
                                                                              0,
                                                                              0,
                                                                              n_head,
                                                                              head_dim);
            return cuda_ok(cudaGetLastError(), "attention decode online launch");
        }
        fprintf(stderr, "ds4: CUDA attention score buffer too small for %u compressed rows\n", n_comp);
        return 0;
    }
    if (!use_mask && head_dim == 512u &&
        g_cuda_decode_heads8_online &&
        !g_cuda_no_window_attention) {
        const uint32_t synthetic_pos0 = n_raw - 1u;
        dim3 online_grid(1, (n_head + 7u) / 8u, 1);
        attention_decode_mixed_heads8_online_kernel<<<online_grid, 256>>>((float *)heads->ptr,
                                                                          sinks,
                                                                          (const float *)q->ptr,
                                                                          (const float *)raw_kv->ptr,
                                                                          n_comp ? (const float *)comp_kv->ptr : (const float *)raw_kv->ptr,
                                                                          1,
                                                                          synthetic_pos0,
                                                                          n_raw,
                                                                          raw_cap,
                                                                          raw_start,
                                                                          n_comp,
                                                                          0,
                                                                          0,
                                                                          n_head,
                                                                          head_dim);
        return cuda_ok(cudaGetLastError(), "attention decode heads8 online launch");
    }
    const uint32_t score_lanes =
        g_cuda_decode_score4 ? 4u : (g_cuda_decode_score8 ? 8u : 0u);
    const uint32_t threads =
        head_dim == 512u && score_lanes == 0u &&
        !g_cuda_no_decode_value512 ? 512u : 256u;
    int score_split_rc = attention_decode_score_split_launch(
            logical_tier, (float *)heads->ptr, sinks, (const float *)q->ptr,
            (const float *)raw_kv->ptr,
            n_comp ? (const float *)comp_kv->ptr : (const float *)raw_kv->ptr,
            use_mask ? (const float *)comp_mask->ptr : NULL, use_mask,
            0, n_raw, raw_cap, raw_start, n_comp, 0, 0,
            n_head, head_dim, threads, NULL);
    if (score_split_rc == 1) {
        return cuda_ok(cudaGetLastError(), "attention exact score split launch");
    }
    if (score_split_rc < 0) return 0;
    /* perf-02 split-KV opt-in (default OFF). n_tokens==1 here by construction.
     * S==1 / disabled / unhandled -> rc 0, fall through to the old kernel. */
    if (cuda_splitkv_decode_requested()) {
        int rc = attention_decode_splitkv_launch(
                logical_tier, (float *)heads->ptr, sinks, (const float *)q->ptr,
                (const float *)raw_kv->ptr,
                n_comp ? (const float *)comp_kv->ptr : (const float *)raw_kv->ptr,
                use_mask ? (const float *)comp_mask->ptr : NULL, use_mask,
                0, n_raw, raw_cap, raw_start, n_comp, 0, 0, n_head, head_dim);
        if (rc == 1) return cuda_ok(cudaGetLastError(), "attention decode splitkv launch");
        if (rc < 0) return 0;
    }
    dim3 grid(1, n_head, 1);
    attention_decode_mixed_kernel<<<grid, threads>>>((float *)heads->ptr,
                                                     sinks,
                                                     (const float *)q->ptr,
                                                     (const float *)raw_kv->ptr,
                                                     n_comp ? (const float *)comp_kv->ptr : (const float *)raw_kv->ptr,
                                                     use_mask ? (const float *)comp_mask->ptr : NULL,
                                                     use_mask,
                                                     1, 0, n_raw, raw_cap, raw_start, n_comp,
                                                     0, 0, n_head, head_dim,
                                                     score_lanes);
    return cuda_ok(cudaGetLastError(), "attention decode launch");
}

extern "C" int ds4_gpu_attention_decode_heads_rope_tensor(
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
        uint32_t                head_dim,
        uint32_t                n_rot,
        uint32_t                pos0,
        uint32_t                n_ctx_orig,
        float                   freq_base,
        float                   freq_scale,
        float                   ext_factor,
        float                   attn_factor,
        float                   beta_fast,
        float                   beta_slow,
        int                    *fused_inv_rope) {
    if (fused_inv_rope) *fused_inv_rope = 0;
    if (!g_cuda_exact_score_split_fuse_inv_rope ||
        n_rot == 0u || n_rot > head_dim || (n_rot & 1u) ||
        head_dim != 512u) {
        return ds4_gpu_attention_decode_heads_tensor(
                heads, model_map, model_size, sinks_offset, q, raw_kv,
                n_raw, raw_cap, raw_start, comp_kv, comp_kv_f16, n_comp,
                comp_mask, use_mask, n_head, head_dim);
    }
    if (!use_mask && g_cuda_decode_heads8_online && !g_cuda_no_window_attention) {
        return ds4_gpu_attention_decode_heads_tensor(
                heads, model_map, model_size, sinks_offset, q, raw_kv,
                n_raw, raw_cap, raw_start, comp_kv, comp_kv_f16, n_comp,
                comp_mask, use_mask, n_head, head_dim);
    }
    if (comp_kv_f16 ||
        !heads || !q || !raw_kv || !model_map || n_raw == 0 || raw_cap < n_raw ||
        raw_start >= raw_cap || (n_comp != 0 && !comp_kv) || (use_mask && !comp_mask) ||
        sinks_offset > model_size ||
        (uint64_t)n_head * sizeof(float) > model_size - sinks_offset ||
        heads->bytes < (uint64_t)n_head * head_dim * sizeof(float) ||
        q->bytes < (uint64_t)n_head * head_dim * sizeof(float) ||
        raw_kv->bytes < (uint64_t)raw_cap * head_dim * sizeof(float) ||
        (n_comp && comp_kv->bytes < (uint64_t)n_comp * head_dim * sizeof(float)) ||
        (use_mask && comp_mask->bytes < (uint64_t)n_comp * sizeof(float)) ||
        !cuda_attention_score_buffer_fits(n_comp)) {
        return ds4_gpu_attention_decode_heads_tensor(
                heads, model_map, model_size, sinks_offset, q, raw_kv,
                n_raw, raw_cap, raw_start, comp_kv, comp_kv_f16, n_comp,
                comp_mask, use_mask, n_head, head_dim);
    }
    const uint32_t score_lanes =
        g_cuda_decode_score4 ? 4u : (g_cuda_decode_score8 ? 8u : 0u);
    const uint32_t threads =
        score_lanes == 0u && !g_cuda_no_decode_value512 ? 512u : 256u;
    if (threads < 512u) {
        return ds4_gpu_attention_decode_heads_tensor(
                heads, model_map, model_size, sinks_offset, q, raw_kv,
                n_raw, raw_cap, raw_start, comp_kv, comp_kv_f16, n_comp,
                comp_mask, use_mask, n_head, head_dim);
    }
    const int logical_tier = ds4_tensor_device_idx(heads);
    const float *sinks = (const float *)cuda_resolve_weight_ptr(
            model_map, sinks_offset, (uint64_t)n_head * sizeof(float),
            logical_tier, "attn_sinks");
    if (!sinks) return 0;
    cuda_attention_inv_rope_params rope;
    rope.n_rot = n_rot;
    rope.pos0 = pos0;
    rope.n_ctx_orig = n_ctx_orig;
    rope.freq_base = freq_base;
    rope.freq_scale = freq_scale;
    rope.ext_factor = ext_factor;
    rope.attn_factor = attn_factor;
    rope.beta_fast = beta_fast;
    rope.beta_slow = beta_slow;
    int score_split_rc = attention_decode_score_split_launch(
            logical_tier, (float *)heads->ptr, sinks, (const float *)q->ptr,
            (const float *)raw_kv->ptr,
            n_comp ? (const float *)comp_kv->ptr : (const float *)raw_kv->ptr,
            use_mask ? (const float *)comp_mask->ptr : NULL, use_mask,
            0, n_raw, raw_cap, raw_start, n_comp, 0, 0,
            n_head, head_dim, threads, &rope);
    if (score_split_rc == 1) {
        if (fused_inv_rope) *fused_inv_rope = 1;
        return cuda_ok(cudaGetLastError(),
                       "attention exact score split fused inv rope launch");
    }
    if (score_split_rc < 0) return 0;
    return ds4_gpu_attention_decode_heads_tensor(
            heads, model_map, model_size, sinks_offset, q, raw_kv,
            n_raw, raw_cap, raw_start, comp_kv, comp_kv_f16, n_comp,
            comp_mask, use_mask, n_head, head_dim);
}

extern "C" int ds4_gpu_attention_decode_rows_rope_tensor(
        ds4_gpu_tensor                       *heads,
        const void                           *model_map,
        uint64_t                              model_size,
        uint64_t                              sinks_offset,
        const ds4_gpu_tensor                 *q,
        const ds4_gpu_attention_decode_row   *rows,
        uint32_t                              n_rows,
        uint32_t                              n_head,
        uint32_t                              head_dim,
        uint32_t                              n_rot,
        uint32_t                              n_ctx_orig,
        float                                 freq_base,
        float                                 freq_scale,
        float                                 ext_factor,
        float                                 attn_factor,
        float                                 beta_fast,
        float                                 beta_slow) {
    if (!heads || !q || !rows || !model_map || n_rows < 2u ||
        n_rows > DS4_GPU_ATTENTION_DECODE_BATCH_MAX || n_head == 0u ||
        head_dim != 512u || n_rot == 0u || n_rot > head_dim ||
        (n_rot & 1u) != 0u || sinks_offset > model_size ||
        (uint64_t)n_head * sizeof(float) > model_size - sinks_offset ||
        heads->bytes < (uint64_t)n_rows * n_head * head_dim * sizeof(float) ||
        q->bytes < (uint64_t)n_rows * n_head * head_dim * sizeof(float)) {
        return 0;
    }
    const int logical_tier = ds4_tensor_device_idx(heads);
    if (logical_tier < 0 || logical_tier >= g_n_gpus ||
        ds4_tensor_device_idx(q) != logical_tier) {
        return 0;
    }

    /* This first grouped path mirrors the promoted default decode exactly.
     * Alternative score kernels and graph/split-KV experiments retain the
     * one-session dispatcher until they gain equivalent row-table variants. */
    if (cuda_env_flag_enabled("DS4_CUDA_NO_EXACT_SCORE_SPLIT_DECODE", 0) ||
        !cuda_env_flag_enabled("DS4_CUDA_EXACT_SCORE_SPLIT_DECODE", 1) ||
        cuda_splitkv_decode_requested() ||
        g_cuda_decode_heads8_online || g_cuda_decode_score4 ||
        g_cuda_decode_score8 || g_cuda_no_decode_value512 ||
        g_cuda_exact_score_split_graph || g_cuda_exact_score_split_ldg ||
        g_cuda_exact_score_split_vec4 ||
        g_cuda_exact_score_split_vec4_plain ||
        g_cuda_exact_score_split_dim2 ||
        g_cuda_exact_score_split_fuse_inv_rope ||
        getenv("DS4_CUDA_NO_SCORE_TILE") != NULL ||
        getenv("DS4_CUDA_EXACT_SCORE_SPLIT_MIN_SCORE") != NULL ||
        getenv("DS4_CUDA_EXACT_SCORE_SPLIT_CHUNK") != NULL ||
        getenv("DS4_CUDA_EXACT_SCORE_SPLIT_S_FLOOR") != NULL ||
        getenv("DS4_CUDA_EXACT_SCORE_SPLIT_S_MAX") != NULL ||
        getenv("DS4_CUDA_EXACT_SCORE_SPLIT_S") != NULL) {
        return 0;
    }

    cuda_attention_decode_row_table table;
    memset(&table, 0, sizeof(table));
    uint32_t max_dense_score = 0u;
    bool have_dense = false;
    bool have_indexed = false;
    for (uint32_t i = 0; i < n_rows; i++) {
        const ds4_gpu_attention_decode_row r = rows[i];
        if (r.raw_kv == 0u || r.n_raw == 0u || r.raw_cap < r.n_raw ||
            r.raw_start >= r.raw_cap || (r.n_comp != 0u && r.comp_kv == 0u)) {
            return 0;
        }
        if (r.indexed) {
            if (r.comp_kv == 0u || r.topk == 0u || r.n_comp == 0u ||
                r.top_k == 0u || r.top_k > 512u || r.ratio == 0u) {
                return 0;
            }
            have_indexed = true;
        } else {
            const uint32_t raw_count = r.n_raw > 256u ? 256u : r.n_raw;
            const uint32_t n_score = raw_count + r.n_comp;
            /* n_score==1 takes the legacy one-block kernel and is not a
             * score-split shape. Decode after any nonempty prompt is >1. */
            if (n_score <= 1u || n_score > DS4_CUDA_ATTENTION_SCORE_CAP) {
                return 0;
            }
            if (n_score > max_dense_score) max_dense_score = n_score;
            have_dense = true;
        }
        table.row[i] = r;
    }

    const float *sinks = (const float *)cuda_resolve_weight_ptr(
        model_map, sinks_offset, (uint64_t)n_head * sizeof(float),
        logical_tier, "attn_sinks_rows");
    if (!sinks) return 0;

    if (have_dense) {
        if ((uint64_t)n_rows > UINT64_MAX / n_head ||
            (uint64_t)n_rows * n_head > UINT64_MAX / max_dense_score) {
            return 0;
        }
        const uint64_t score_count =
            (uint64_t)n_rows * n_head * max_dense_score;
        float *scores = (float *)cuda_tmp_alloc_on(
            logical_tier, score_count * sizeof(float),
            "attention exact decode rows");
        if (!scores) return 0;

        const size_t tile_shmem =
            (size_t)(DS4_SCORE_TILE_HEADS + DS4_SCORE_TILE_ROWS) *
            DS4_SCORE_TILE_STRIDE * sizeof(float);
        static int tile_shmem_ready[DS4_MAX_GPUS] = {0};
        int physical_device = 0;
        if (cudaGetDevice(&physical_device) != cudaSuccess ||
            physical_device < 0 || physical_device >= DS4_MAX_GPUS) {
            return 0;
        }
        if (!tile_shmem_ready[physical_device]) {
            if (!cuda_ok(cudaFuncSetAttribute(
                    attention_decode_score_split_scores_tile512_rows_kernel,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    (int)tile_shmem),
                    "attention score rows shared-memory opt-in")) {
                return 0;
            }
            tile_shmem_ready[physical_device] = 1;
        }
        dim3 score_grid(
            (max_dense_score + DS4_SCORE_TILE_ROWS - 1u) /
                DS4_SCORE_TILE_ROWS,
            (n_head + DS4_SCORE_TILE_HEADS - 1u) /
                DS4_SCORE_TILE_HEADS,
            n_rows);
        attention_decode_score_split_scores_tile512_rows_kernel
            <<<score_grid, 256, tile_shmem>>>(
                scores, (const float *)q->ptr, table, n_rows,
                max_dense_score, n_head, head_dim);
        if (!cuda_ok(cudaGetLastError(),
                     "attention exact score rows launch")) {
            return 0;
        }
        dim3 final_grid(n_rows, n_head, 1u);
        attention_decode_score_split_finalize_rows_kernel
            <<<final_grid, 512>>>(
                (float *)heads->ptr, sinks, scores, table, n_rows,
                max_dense_score, n_head, head_dim);
        if (!cuda_ok(cudaGetLastError(),
                     "attention exact finalize rows launch")) {
            return 0;
        }
    }
    if (have_indexed) {
        dim3 indexed_grid(n_rows, n_head, 1u);
        attention_indexed_mixed_decode_rows_kernel<<<indexed_grid, 256>>>(
            (float *)heads->ptr, sinks, (const float *)q->ptr, table,
            n_rows, n_head, head_dim);
        if (!cuda_ok(cudaGetLastError(),
                     "attention indexed decode rows launch")) {
            return 0;
        }
    }

    const uint32_t pairs = n_rows * n_head * (n_rot / 2u);
    rope_tail_decode_rows_kernel<<<(pairs + 255u) / 256u, 256>>>(
        (float *)heads->ptr, table, n_rows, n_head, head_dim, n_rot,
        n_ctx_orig, 1, freq_base, freq_scale, ext_factor, attn_factor,
        beta_fast, beta_slow);
    return cuda_ok(cudaGetLastError(),
                   "attention decode rows inverse rope launch");
}

extern "C" int ds4_gpu_attention_prefill_raw_heads_tensor(ds4_gpu_tensor *heads, const void *model_map, uint64_t model_size, uint64_t sinks_offset, const ds4_gpu_tensor *q, const ds4_gpu_tensor *raw_kv, uint32_t n_tokens, uint32_t window, uint32_t n_head, uint32_t head_dim) {
    if (!heads || !q || !raw_kv || !model_map || sinks_offset > model_size ||
        model_size - sinks_offset < (uint64_t)n_head * sizeof(float) ||
        heads->bytes < (uint64_t)n_tokens * n_head * head_dim * sizeof(float) ||
        q->bytes < (uint64_t)n_tokens * n_head * head_dim * sizeof(float) ||
        raw_kv->bytes < (uint64_t)n_tokens * head_dim * sizeof(float) ||
        window > 256) return 0;
    const int logical_tier = ds4_tensor_device_idx(heads);
    const float *sinks = (const float *)cuda_resolve_weight_ptr(
            model_map, sinks_offset, (uint64_t)n_head * sizeof(float), logical_tier, "attn_sinks");
    if (!sinks) return 0;
    if (n_tokens > 1 && head_dim == 512 &&
        getenv("DS4_CUDA_NO_WINDOW_ATTENTION") == NULL &&
        (getenv("DS4_CUDA_WINDOW_ATTENTION") != NULL || (!g_quality_mode && n_tokens >= 128u))) {
        dim3 grid(n_tokens, (n_head + 7u) / 8u, 1);
        attention_static_mixed_heads8_online_kernel<<<grid, 256>>>((float *)heads->ptr,
                                                                   sinks,
                                                                   (const float *)q->ptr,
                                                                   (const float *)raw_kv->ptr,
                                                                   (const float *)raw_kv->ptr,
                                                                   n_tokens,
                                                                   0,
                                                                   window,
                                                                   1,
                                                                   n_head,
                                                                   head_dim);
        return cuda_ok(cudaGetLastError(), "attention raw window launch");
    }
    if (g_cublas_ready && n_tokens > 1 && head_dim == 512 &&
        getenv("DS4_CUDA_NO_CUBLAS_ATTENTION") == NULL) {
        const uint32_t n_keys = n_tokens;
        const uint64_t score_count = (uint64_t)n_head * n_tokens * n_keys;
        const uint64_t out_count = (uint64_t)n_head * n_tokens * head_dim;
        const uint64_t score_bytes = score_count * sizeof(float);
        const uint64_t out_offset = (score_bytes + 255u) & ~255ull;
        const uint64_t tmp_bytes = out_offset + out_count * sizeof(float);
        float *tmp = (float *)cuda_tmp_alloc_on(logical_tier, tmp_bytes, "attention raw cublas");
        if (!tmp) return 0;
        float *scores = tmp;
        float *out_tmp = (float *)((char *)tmp + out_offset);
        const float alpha = rsqrtf((float)head_dim);
        const float beta = 0.0f;
        cublasStatus_t st = cublasSgemmStridedBatched(cuda_cublas_for_tier(logical_tier),
                                                      CUBLAS_OP_T,
                                                      CUBLAS_OP_N,
                                                      (int)n_keys,
                                                      (int)n_tokens,
                                                      (int)head_dim,
                                                      &alpha,
                                                      (const float *)raw_kv->ptr,
                                                      (int)head_dim,
                                                      0,
                                                      (const float *)q->ptr,
                                                      (int)(n_head * head_dim),
                                                      (long long)head_dim,
                                                      &beta,
                                                      scores,
                                                      (int)n_keys,
                                                      (long long)n_keys * n_tokens,
                                                      (int)n_head);
        if (!cublas_ok(st, "attention raw score gemm")) return 0;
        dim3 sgrid(n_tokens, n_head, 1);
        attention_prefill_raw_softmax_kernel<<<sgrid, 256>>>(scores, sinks, n_tokens, window, n_keys);
        if (!cuda_ok(cudaGetLastError(), "attention raw softmax launch")) return 0;
        const float one = 1.0f;
        st = cublasSgemmStridedBatched(cuda_cublas_for_tier(logical_tier),
                                       CUBLAS_OP_N,
                                       CUBLAS_OP_N,
                                       (int)head_dim,
                                       (int)n_tokens,
                                       (int)n_keys,
                                       &one,
                                       (const float *)raw_kv->ptr,
                                       (int)head_dim,
                                       0,
                                       scores,
                                       (int)n_keys,
                                       (long long)n_keys * n_tokens,
                                       &beta,
                                       out_tmp,
                                       (int)head_dim,
                                       (long long)head_dim * n_tokens,
                                       (int)n_head);
        if (!cublas_ok(st, "attention raw value gemm")) return 0;
        uint64_t n = (uint64_t)n_tokens * n_head * head_dim;
        attention_prefill_unpack_heads_kernel<<<(n + 255) / 256, 256>>>((float *)heads->ptr,
                                                                        out_tmp,
                                                                        n_tokens,
                                                                        n_head,
                                                                        head_dim);
        return cuda_ok(cudaGetLastError(), "attention raw unpack launch");
    }
    dim3 grid(n_tokens, n_head, 1);
    attention_prefill_raw_kernel<<<grid, 128>>>((float *)heads->ptr,
                                                sinks,
                                                (const float *)q->ptr,
                                                (const float *)raw_kv->ptr,
                                                n_tokens, window, n_head, head_dim);
    return cuda_ok(cudaGetLastError(), "attention_prefill_raw launch");
}
static int attention_decode_batch_launch(
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
    if (comp_kv_f16 ||
        !heads || !q || !raw_kv || !model_map || n_tokens == 0 ||
        n_raw == 0 || raw_cap < n_raw || raw_start >= raw_cap ||
        (n_comp != 0 && !comp_kv) || (use_comp_mask && !comp_mask) ||
        sinks_offset > model_size ||
        (uint64_t)n_head * sizeof(float) > model_size - sinks_offset ||
        heads->bytes < (uint64_t)n_tokens * n_head * head_dim * sizeof(float) ||
        q->bytes < (uint64_t)n_tokens * n_head * head_dim * sizeof(float) ||
        raw_kv->bytes < (uint64_t)raw_cap * head_dim * sizeof(float) ||
        (n_comp && comp_kv->bytes < (uint64_t)n_comp * head_dim * sizeof(float)) ||
        (use_comp_mask && comp_mask->bytes < (uint64_t)n_tokens * n_comp * sizeof(float))) {
        return 0;
    }
    if (n_comp != 0 && ratio == 0) return 0;
    const int logical_tier = ds4_tensor_device_idx(heads);
    const float *sinks = (const float *)cuda_resolve_weight_ptr(
            model_map, sinks_offset, (uint64_t)n_head * sizeof(float), logical_tier, "attn_sinks");
    if (!sinks) return 0;
    if (!cuda_attention_score_buffer_fits(n_comp)) {
        if (!use_comp_mask && head_dim == 512u &&
            !g_cuda_no_window_attention) {
            dim3 online_grid(n_tokens, (n_head + 7u) / 8u, 1);
            attention_decode_mixed_heads8_online_kernel<<<online_grid, 256>>>((float *)heads->ptr,
                                                                              sinks,
                                                                              (const float *)q->ptr,
                                                                              (const float *)raw_kv->ptr,
                                                                              n_comp ? (const float *)comp_kv->ptr : (const float *)raw_kv->ptr,
                                                                              n_tokens,
                                                                              pos0,
                                                                              n_raw,
                                                                              raw_cap,
                                                                              raw_start,
                                                                              n_comp,
                                                                              window,
                                                                              ratio,
                                                                              n_head,
                                                                              head_dim);
            return cuda_ok(cudaGetLastError(), "attention decode online launch");
        }
        fprintf(stderr, "ds4: CUDA attention score buffer too small for %u compressed rows\n", n_comp);
        return 0;
    }
    if (!use_comp_mask && n_tokens > 1 && head_dim == 512 &&
        !g_cuda_no_window_attention &&
        (getenv("DS4_CUDA_WINDOW_ATTENTION") != NULL || (!g_quality_mode && n_tokens >= 128u))) {
        dim3 grid(n_tokens, (n_head + 7u) / 8u, 1);
        attention_decode_mixed_heads8_online_kernel<<<grid, 256>>>((float *)heads->ptr,
                                                                   sinks,
                                                                   (const float *)q->ptr,
                                                                   (const float *)raw_kv->ptr,
                                                                   n_comp ? (const float *)comp_kv->ptr : (const float *)raw_kv->ptr,
                                                                   n_tokens,
                                                                   pos0,
                                                                   n_raw,
                                                                   raw_cap,
                                                                   raw_start,
                                                                   n_comp,
                                                                   window,
                                                                   ratio,
                                                                   n_head,
                                                                   head_dim);
        return cuda_ok(cudaGetLastError(), "attention decode window launch");
    }
    if (!use_comp_mask && n_tokens == 1u && head_dim == 512 &&
        g_cuda_decode_heads8_online &&
        !g_cuda_no_window_attention) {
        dim3 grid(1, (n_head + 7u) / 8u, 1);
        attention_decode_mixed_heads8_online_kernel<<<grid, 256>>>((float *)heads->ptr,
                                                                   sinks,
                                                                   (const float *)q->ptr,
                                                                   (const float *)raw_kv->ptr,
                                                                   n_comp ? (const float *)comp_kv->ptr : (const float *)raw_kv->ptr,
                                                                   n_tokens,
                                                                   pos0,
                                                                   n_raw,
                                                                   raw_cap,
                                                                   raw_start,
                                                                   n_comp,
                                                                   window,
                                                                   ratio,
                                                                   n_head,
                                                                   head_dim);
        return cuda_ok(cudaGetLastError(), "attention decode heads8 online batch launch");
    }
    const uint32_t score_lanes =
        g_cuda_decode_score4 ? 4u : (g_cuda_decode_score8 ? 8u : 0u);
    const uint32_t threads =
        n_tokens == 1u && head_dim == 512u && score_lanes == 0u &&
        !g_cuda_no_decode_value512 ? 512u : 256u;
    if (n_tokens == 1u) {
        int score_split_rc = attention_decode_score_split_launch(
                logical_tier, (float *)heads->ptr, sinks, (const float *)q->ptr,
                (const float *)raw_kv->ptr,
                n_comp ? (const float *)comp_kv->ptr : (const float *)raw_kv->ptr,
                use_comp_mask ? (const float *)comp_mask->ptr : NULL, use_comp_mask,
                pos0, n_raw, raw_cap, raw_start, n_comp, window, ratio,
                n_head, head_dim, threads, NULL);
        if (score_split_rc == 1) {
            return cuda_ok(cudaGetLastError(), "attention exact score split batch launch");
        }
        if (score_split_rc < 0) return 0;
    }
    /* perf-02 split-KV opt-in (default OFF). Single-token decode only; multi-
     * token batch shapes already fill the grid and fall through unchanged.
     * S==1 / disabled / unhandled -> rc 0, fall through to the old kernel. */
    if (n_tokens == 1u && cuda_splitkv_decode_requested()) {
        int rc = attention_decode_splitkv_launch(
                logical_tier, (float *)heads->ptr, sinks, (const float *)q->ptr,
                (const float *)raw_kv->ptr,
                n_comp ? (const float *)comp_kv->ptr : (const float *)raw_kv->ptr,
                use_comp_mask ? (const float *)comp_mask->ptr : NULL, use_comp_mask,
                pos0, n_raw, raw_cap, raw_start, n_comp, window, ratio, n_head, head_dim);
        if (rc == 1) return cuda_ok(cudaGetLastError(), "attention decode splitkv batch launch");
        if (rc < 0) return 0;
    }
    dim3 grid(n_tokens, n_head, 1);
    attention_decode_mixed_kernel<<<grid, threads>>>((float *)heads->ptr,
                                                     sinks,
                                                     (const float *)q->ptr,
                                                     (const float *)raw_kv->ptr,
                                                     n_comp ? (const float *)comp_kv->ptr : (const float *)raw_kv->ptr,
                                                     use_comp_mask ? (const float *)comp_mask->ptr : NULL,
                                                     use_comp_mask, n_tokens, pos0, n_raw, raw_cap,
                                                     raw_start, n_comp, window, ratio, n_head, head_dim,
                                                     score_lanes);
    return cuda_ok(cudaGetLastError(), "attention decode batch launch");
}

extern "C" int ds4_gpu_attention_decode_raw_batch_heads_tensor(
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
    return attention_decode_batch_launch(heads, model_map, model_size, sinks_offset,
                                      q, raw_kv, NULL, 0, NULL, 0, n_tokens, pos0,
                                      n_raw, raw_cap, raw_start, 0, window, 1,
                                      n_head, head_dim);
}

extern "C" int ds4_gpu_attention_decode_mixed_batch_heads_tensor(
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
    if (comp_kv_f16) return 0;
    return attention_decode_batch_launch(heads, model_map, model_size, sinks_offset,
                                      q, raw_kv, comp_kv, comp_kv_f16, comp_mask, use_comp_mask,
                                      n_tokens, pos0, n_raw, raw_cap, raw_start,
                                      n_comp, window, ratio, n_head, head_dim);
}

extern "C" int ds4_gpu_attention_indexed_mixed_batch_heads_tensor(
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
    if (comp_kv_f16 ||
        !heads || !q || !raw_kv || !comp_kv || !topk || !model_map ||
        n_tokens == 0 || n_raw == 0 || raw_cap < n_raw || raw_start >= raw_cap ||
        n_comp == 0 || top_k == 0 ||
        sinks_offset > model_size ||
        (uint64_t)n_head * sizeof(float) > model_size - sinks_offset ||
        heads->bytes < (uint64_t)n_tokens * n_head * head_dim * sizeof(float) ||
        q->bytes < (uint64_t)n_tokens * n_head * head_dim * sizeof(float) ||
        raw_kv->bytes < (uint64_t)raw_cap * head_dim * sizeof(float) ||
        comp_kv->bytes < (uint64_t)n_comp * head_dim * sizeof(float) ||
        topk->bytes < (uint64_t)n_tokens * top_k * sizeof(int32_t)) {
        return 0;
    }
    if (top_k > 512u) return 0;
    const int logical_tier = ds4_tensor_device_idx(heads);
    const float *sinks = (const float *)cuda_resolve_weight_ptr(
            model_map, sinks_offset, (uint64_t)n_head * sizeof(float), logical_tier, "attn_sinks");
    if (!sinks) return 0;
    const int32_t *topk_ptr = (const int32_t *)topk->ptr;
    if (n_tokens > 1u && top_k == 512u &&
        getenv("DS4_CUDA_NO_INDEXED_TOPK_SORT") == NULL) {
        const uint64_t sort_bytes = (uint64_t)n_tokens * top_k * sizeof(int32_t);
        int32_t *sorted = (int32_t *)cuda_tmp_alloc_on(logical_tier, sort_bytes, "indexed attention topk sort");
        if (!sorted) return 0;
        indexed_topk_sort_512_asc_kernel<<<n_tokens, 512>>>(sorted, topk_ptr, n_tokens);
        if (!cuda_ok(cudaGetLastError(), "indexed attention topk sort launch")) return 0;
        topk_ptr = sorted;
    }
    if (n_tokens > 1 && head_dim == 512 && top_k <= 512u &&
        getenv("DS4_CUDA_NO_INDEXED_HEADS8") == NULL) {
        if (getenv("DS4_CUDA_INDEXED_TWOPASS") == NULL) {
            dim3 grid(n_tokens, (n_head + 15u) / 16u, 1);
            attention_indexed_mixed_heads8_online_kernel<8, 16><<<grid, 512>>>((float *)heads->ptr,
                                                                               sinks,
                                                                               (const float *)q->ptr,
                                                                               (const float *)raw_kv->ptr,
                                                                               (const float *)comp_kv->ptr,
                                                                               topk_ptr,
                                                                               n_tokens,
                                                                               pos0,
                                                                               n_raw,
                                                                               raw_cap,
                                                                               raw_start,
                                                                               n_comp,
                                                                               top_k,
                                                                               window,
                                                                               ratio,
                                                                               n_head,
                                                                               head_dim);
            return cuda_ok(cudaGetLastError(), "attention indexed online launch");
        }
        dim3 grid(n_tokens, (n_head + 7u) / 8u, 1);
        attention_indexed_mixed_heads8_rb4_kernel<<<grid, 256>>>((float *)heads->ptr,
                                                                 sinks,
                                                                 (const float *)q->ptr,
                                                                 (const float *)raw_kv->ptr,
                                                                 (const float *)comp_kv->ptr,
                                                                 topk_ptr,
                                                                 n_tokens,
                                                                 pos0,
                                                                 n_raw,
                                                                 raw_cap,
                                                                 raw_start,
                                                                 n_comp,
                                                                 top_k,
                                                                 window,
                                                                 ratio,
                                                                 n_head,
                                                                 head_dim);
        return cuda_ok(cudaGetLastError(), "attention indexed heads8 launch");
    }
    dim3 grid(n_tokens, n_head, 1);
    attention_indexed_mixed_kernel<<<grid, 256>>>((float *)heads->ptr,
                                                  sinks,
                                                  (const float *)q->ptr,
                                                  (const float *)raw_kv->ptr,
                                                  (const float *)comp_kv->ptr,
                                                  topk_ptr,
                                                  n_tokens,
                                                  pos0,
                                                  n_raw,
                                                  raw_cap,
                                                  raw_start,
                                                  n_comp,
                                                  top_k,
                                                  window,
                                                  ratio,
                                                  n_head,
                                                  head_dim);
    return cuda_ok(cudaGetLastError(), "attention indexed mixed launch");
}

static int attention_prefill_mixed_launch(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        const ds4_gpu_tensor *comp_kv,
        const ds4_gpu_tensor *comp_mask,
        uint32_t                use_comp_mask,
        uint32_t                n_tokens,
        uint32_t                n_comp,
        uint32_t                window,
        uint32_t                ratio,
        uint32_t                n_head,
        uint32_t                head_dim) {
    if (!heads || !q || !raw_kv || !model_map || n_tokens == 0 || ratio == 0 ||
        (n_comp != 0 && !comp_kv) || (use_comp_mask && !comp_mask) ||
        sinks_offset > model_size ||
        (uint64_t)n_head * sizeof(float) > model_size - sinks_offset ||
        heads->bytes < (uint64_t)n_tokens * n_head * head_dim * sizeof(float) ||
        q->bytes < (uint64_t)n_tokens * n_head * head_dim * sizeof(float) ||
        raw_kv->bytes < (uint64_t)n_tokens * head_dim * sizeof(float) ||
        (n_comp && comp_kv->bytes < (uint64_t)n_comp * head_dim * sizeof(float)) ||
        (use_comp_mask && comp_mask->bytes < (uint64_t)n_tokens * n_comp * sizeof(float))) {
        return 0;
    }
    const int logical_tier = ds4_tensor_device_idx(heads);
    const float *sinks = (const float *)cuda_resolve_weight_ptr(
            model_map, sinks_offset, (uint64_t)n_head * sizeof(float), logical_tier, "attn_sinks");
    if (!sinks) return 0;
    if (!use_comp_mask && n_tokens > 1 && head_dim == 512 &&
        getenv("DS4_CUDA_NO_WINDOW_ATTENTION") == NULL &&
        (getenv("DS4_CUDA_WINDOW_ATTENTION") != NULL || (!g_quality_mode && n_tokens >= 128u))) {
        dim3 grid(n_tokens, (n_head + 7u) / 8u, 1);
        attention_static_mixed_heads8_online_kernel<<<grid, 256>>>((float *)heads->ptr,
                                                                   sinks,
                                                                   (const float *)q->ptr,
                                                                   (const float *)raw_kv->ptr,
                                                                   n_comp ? (const float *)comp_kv->ptr : (const float *)raw_kv->ptr,
                                                                   n_tokens,
                                                                   n_comp,
                                                                   window,
                                                                   ratio,
                                                                   n_head,
                                                                   head_dim);
        return cuda_ok(cudaGetLastError(), "attention mixed window launch");
    }
    if (g_cublas_ready && n_tokens > 1 && head_dim == 512 &&
        getenv("DS4_CUDA_NO_CUBLAS_ATTENTION") == NULL) {
        const uint32_t n_keys = n_tokens + n_comp;
        const uint64_t kv_count = (uint64_t)n_keys * head_dim;
        const uint64_t score_count = (uint64_t)n_head * n_tokens * n_keys;
        const uint64_t out_count = (uint64_t)n_head * n_tokens * head_dim;
        const uint64_t kv_bytes = kv_count * sizeof(float);
        const uint64_t score_offset = (kv_bytes + 255u) & ~255ull;
        const uint64_t score_bytes = score_count * sizeof(float);
        const uint64_t out_offset = score_offset + ((score_bytes + 255u) & ~255ull);
        const uint64_t tmp_bytes = out_offset + out_count * sizeof(float);
        float *tmp = (float *)cuda_tmp_alloc_on(logical_tier, tmp_bytes, "attention mixed cublas");
        if (!tmp) return 0;
        float *kv = tmp;
        float *scores = (float *)((char *)tmp + score_offset);
        float *out_tmp = (float *)((char *)tmp + out_offset);
        attention_prefill_pack_mixed_kv_kernel<<<(kv_count + 255) / 256, 256>>>(
                kv,
                (const float *)raw_kv->ptr,
                n_comp ? (const float *)comp_kv->ptr : (const float *)raw_kv->ptr,
                n_tokens,
                n_comp,
                head_dim);
        if (!cuda_ok(cudaGetLastError(), "attention mixed kv pack launch")) return 0;
        const float alpha = rsqrtf((float)head_dim);
        const float beta = 0.0f;
        cublasStatus_t st = cublasSgemmStridedBatched(cuda_cublas_for_tier(logical_tier),
                                                      CUBLAS_OP_T,
                                                      CUBLAS_OP_N,
                                                      (int)n_keys,
                                                      (int)n_tokens,
                                                      (int)head_dim,
                                                      &alpha,
                                                      kv,
                                                      (int)head_dim,
                                                      0,
                                                      (const float *)q->ptr,
                                                      (int)(n_head * head_dim),
                                                      (long long)head_dim,
                                                      &beta,
                                                      scores,
                                                      (int)n_keys,
                                                      (long long)n_keys * n_tokens,
                                                      (int)n_head);
        if (!cublas_ok(st, "attention mixed score gemm")) return 0;
        dim3 sgrid(n_tokens, n_head, 1);
        attention_prefill_mixed_softmax_kernel<<<sgrid, 256>>>(
                scores,
                sinks,
                use_comp_mask ? (const float *)comp_mask->ptr : NULL,
                use_comp_mask,
                n_tokens,
                n_comp,
                window,
                ratio,
                n_keys);
        if (!cuda_ok(cudaGetLastError(), "attention mixed softmax launch")) return 0;
        const float one = 1.0f;
        st = cublasSgemmStridedBatched(cuda_cublas_for_tier(logical_tier),
                                       CUBLAS_OP_N,
                                       CUBLAS_OP_N,
                                       (int)head_dim,
                                       (int)n_tokens,
                                       (int)n_keys,
                                       &one,
                                       kv,
                                       (int)head_dim,
                                       0,
                                       scores,
                                       (int)n_keys,
                                       (long long)n_keys * n_tokens,
                                       &beta,
                                       out_tmp,
                                       (int)head_dim,
                                       (long long)head_dim * n_tokens,
                                       (int)n_head);
        if (!cublas_ok(st, "attention mixed value gemm")) return 0;
        uint64_t n = (uint64_t)n_tokens * n_head * head_dim;
        attention_prefill_unpack_heads_kernel<<<(n + 255) / 256, 256>>>((float *)heads->ptr,
                                                                        out_tmp,
                                                                        n_tokens,
                                                                        n_head,
                                                                        head_dim);
        return cuda_ok(cudaGetLastError(), "attention mixed unpack launch");
    }
    dim3 grid(n_tokens, n_head, 1);
    attention_prefill_mixed_kernel<<<grid, 256>>>((float *)heads->ptr,
                                                  sinks,
                                                  (const float *)q->ptr,
                                                  (const float *)raw_kv->ptr,
                                                  n_comp ? (const float *)comp_kv->ptr : (const float *)raw_kv->ptr,
                                                  use_comp_mask ? (const float *)comp_mask->ptr : NULL,
                                                  use_comp_mask, n_tokens, n_comp, window, ratio,
                                                  n_head, head_dim);
    return cuda_ok(cudaGetLastError(), "attention prefill mixed launch");
}

extern "C" int ds4_gpu_attention_prefill_static_mixed_heads_tensor(
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
    if (comp_kv_f16) return 0;
    return attention_prefill_mixed_launch(heads, model_map, model_size, sinks_offset,
                                       q, raw_kv, comp_kv, NULL, 0, n_tokens,
                                       n_comp, window, ratio, n_head, head_dim);
}

extern "C" int ds4_gpu_attention_prefill_masked_mixed_heads_tensor(
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
    if (comp_kv_f16) return 0;
    return attention_prefill_mixed_launch(heads, model_map, model_size, sinks_offset,
                                       q, raw_kv, comp_kv, comp_mask, 1, n_tokens,
                                       n_comp, window, ratio, n_head, head_dim);
}
extern "C" int ds4_gpu_attention_output_q8_batch_tensor(
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
    (void)group_tmp;
    (void)low_tmp;
    if (!out || !low || !heads || !model_map ||
        group_dim == 0 || rank == 0 || n_groups == 0 || out_dim == 0 || n_tokens == 0) {
        return 0;
    }
    const uint64_t low_dim = (uint64_t)n_groups * rank;
    const uint64_t blocks_a = (group_dim + 31) / 32;
    const uint64_t blocks_b = (low_dim + 31) / 32;
    const uint64_t out_a_bytes = (uint64_t)n_groups * rank * blocks_a * 34;
    const uint64_t out_b_bytes = out_dim * blocks_b * 34;
    if (out_a_offset > model_size || out_b_offset > model_size ||
        out_a_bytes > model_size - out_a_offset ||
        out_b_bytes > model_size - out_b_offset ||
        heads->bytes < (uint64_t)n_tokens * n_groups * group_dim * sizeof(float) ||
        low->bytes < (uint64_t)n_tokens * low_dim * sizeof(float) ||
        out->bytes < (uint64_t)n_tokens * out_dim * sizeof(float)) {
        return 0;
    }
    const int logical_tier = ds4_tensor_device_idx(out);
    const int physical_device =
        (g_n_gpus > 1 && logical_tier >= 0 && logical_tier < g_n_gpus)
            ? g_gpu[logical_tier].device_id : 0;
    const unsigned char *out_a = reinterpret_cast<const unsigned char *>(
            cuda_resolve_weight_ptr(model_map, out_a_offset, out_a_bytes, logical_tier, "attn_out_a"));
    const unsigned char *out_b = reinterpret_cast<const unsigned char *>(
            cuda_resolve_weight_ptr(model_map, out_b_offset, out_b_bytes, logical_tier, "attn_out_b"));
    if (!out_a || !out_b) return 0;

    const uint32_t profile = getenv("DS4_CUDA_ATTN_OUTPUT_PROFILE") != NULL;
    cudaEvent_t prof_ev[3] = {NULL, NULL, NULL};
    if (profile) {
        for (uint32_t i = 0; i < 3u; i++) {
            if (cudaEventCreate(&prof_ev[i]) != cudaSuccess) {
                for (uint32_t j = 0; j < i; j++) (void)cudaEventDestroy(prof_ev[j]);
                memset(prof_ev, 0, sizeof(prof_ev));
                break;
            }
        }
        if (prof_ev[0]) (void)cudaEventRecord(prof_ev[0], 0);
    }

    const __half *out_a_f16 = NULL;
    uint32_t out_a_cublas_min_tokens = 2u;
    const char *out_a_min_env = getenv("DS4_CUDA_ATTENTION_OUTPUT_A_CUBLAS_MIN");
    if (out_a_min_env && out_a_min_env[0]) {
        char *endp = NULL;
        long v = strtol(out_a_min_env, &endp, 10);
        if (endp != out_a_min_env && v > 1 && v < 4096) out_a_cublas_min_tokens = (uint32_t)v;
    }
    if (!g_quality_mode &&
        g_cublas_ready &&
        n_tokens >= out_a_cublas_min_tokens &&
        getenv("DS4_CUDA_NO_CUBLAS_ATTENTION_OUTPUT_A") == NULL) {
        out_a_f16 = cuda_q8_f16_ptr(model_map, out_a_offset, out_a_bytes, group_dim, low_dim, physical_device, "attn_output_a");
    }
    if (out_a_f16) {
        const uint64_t heads_h_count = (uint64_t)n_groups * n_tokens * group_dim;
        const uint64_t low_tmp_count = (uint64_t)n_groups * n_tokens * rank;
        const uint64_t heads_h_bytes = heads_h_count * sizeof(__half);
        const uint64_t low_tmp_offset = (heads_h_bytes + 255u) & ~255ull;
        const uint64_t tmp_bytes = low_tmp_offset + low_tmp_count * sizeof(float);
        void *tmp = cuda_tmp_alloc_on(logical_tier, tmp_bytes, "attention output a cublas");
        if (!tmp) return 0;
        __half *heads_h = (__half *)tmp;
        float *low_packed = (float *)((char *)tmp + low_tmp_offset);
        attention_pack_group_heads_f16_kernel<<<(heads_h_count + 255) / 256, 256>>>(
                heads_h,
                (const float *)heads->ptr,
                n_tokens,
                n_groups,
                group_dim);
        if (!cuda_ok(cudaGetLastError(), "attention_output_q8_a pack launch")) return 0;
        const float alpha = 1.0f;
        const float beta = 0.0f;
        cublasStatus_t st = cublasGemmStridedBatchedEx(cuda_cublas_for_tier(logical_tier),
                                                       CUBLAS_OP_T,
                                                       CUBLAS_OP_N,
                                                       (int)rank,
                                                       (int)n_tokens,
                                                       (int)group_dim,
                                                       &alpha,
                                                       out_a_f16,
                                                       CUDA_R_16F,
                                                       (int)group_dim,
                                                       (long long)rank * group_dim,
                                                       heads_h,
                                                       CUDA_R_16F,
                                                       (int)group_dim,
                                                       (long long)n_tokens * group_dim,
                                                       &beta,
                                                       low_packed,
                                                       CUDA_R_32F,
                                                       (int)rank,
                                                       (long long)rank * n_tokens,
                                                       (int)n_groups,
                                                       CUDA_R_32F,
                                                       CUBLAS_GEMM_DEFAULT);
        if (!cublas_ok(st, "attention output a gemm")) return 0;
        attention_unpack_group_low_kernel<<<(low_tmp_count + 255) / 256, 256>>>(
                (float *)low->ptr,
                low_packed,
                n_tokens,
                n_groups,
                rank);
        if (!cuda_ok(cudaGetLastError(), "attention_output_q8_a unpack launch")) return 0;
    } else {
        const uint64_t x_rows = (uint64_t)n_tokens * n_groups;
        const uint64_t xq_bytes = x_rows * blocks_a * 32u;
        const uint64_t scale_offset = (xq_bytes + 15u) & ~15ull;
        const uint64_t tmp_bytes = scale_offset + x_rows * blocks_a * sizeof(float);
        void *tmp = cuda_tmp_alloc_on(logical_tier, tmp_bytes, "attention output a q8 prequant");
        if (!tmp) return 0;
        int8_t *xq = (int8_t *)tmp;
        float *xscale = (float *)((char *)tmp + scale_offset);
        const int use_dp4a = cuda_q8_use_dp4a();
        dim3 qgrid((unsigned)blocks_a, (unsigned)x_rows, 1);
        quantize_q8_0_f32_kernel<<<qgrid, 32>>>(xq,
                                                xscale,
                                                (const float *)heads->ptr,
                                                group_dim,
                                                blocks_a);
        if (!cuda_ok(cudaGetLastError(), "attention_output_q8_a prequant launch")) return 0;
        int grouped_mma_done = 0;
        if (n_tokens >= 8u) {
            /* One mma launch per group: T=32 matches the warp tree of the
             * grouped reference kernels (multi-term slots for blocks > 32). */
            grouped_mma_done = 1;
            for (uint32_t g = 0; g < n_groups && grouped_mma_done == 1; g++) {
                const int rc = cuda_q8_mma_try_launch(
                        (float *)low->ptr + (uint64_t)g * rank,
                        reinterpret_cast<const unsigned char *>(out_a) +
                            (uint64_t)g * rank * blocks_a * 34u,
                        xq + (uint64_t)g * blocks_a * 32u,
                        xscale + (uint64_t)g * blocks_a,
                        group_dim, rank, n_tokens, blocks_a,
                        (uint64_t)n_groups * blocks_a, low_dim, 32u);
                if (rc < 0) return 0;
                if (rc == 0) grouped_mma_done = 0;
            }
        }
        if (grouped_mma_done) {
            /* handled */
        } else if (getenv("DS4_CUDA_NO_ATTN_A_TOK2") == NULL && n_tokens >= 2u) {
            dim3 grid_a(((unsigned)low_dim + 7u) / 8u, ((unsigned)n_tokens + 1u) / 2u, 1);
            grouped_q8_0_a_preq_warp8_tok2_kernel<<<grid_a, 512>>>((float *)low->ptr,
                                                                   out_a,
                                                                   xq,
                                                                   xscale,
                                                                   group_dim,
                                                                   rank,
                                                                   n_groups,
                                                                   n_tokens,
                                                                   blocks_a,
                                                                   use_dp4a);
        } else {
            dim3 grid_a(((unsigned)low_dim + 7u) / 8u, (unsigned)n_tokens, 1);
            grouped_q8_0_a_preq_warp8_kernel<<<grid_a, 256>>>((float *)low->ptr,
                                                              out_a,
                                                              xq,
                                                              xscale,
                                                              group_dim,
                                                              rank,
                                                              n_groups,
                                                              n_tokens,
                                                              blocks_a,
                                                              use_dp4a);
        }
        if (!cuda_ok(cudaGetLastError(), "attention_output_q8_a preq launch")) return 0;
    }

    if (prof_ev[1]) (void)cudaEventRecord(prof_ev[1], 0);
    (void)out_b;
    int ok = cuda_matmul_q8_0_tensor_labeled(out,
                                             model_map,
                                             model_size,
                                             out_b_offset,
                                             low_dim,
                                             out_dim,
                                             low,
                                             n_tokens,
                                             "attn_output_b");
    if (prof_ev[2]) {
        (void)cudaEventRecord(prof_ev[2], 0);
        if (cudaEventSynchronize(prof_ev[2]) == cudaSuccess) {
            float ms_a = 0.0f, ms_b = 0.0f, ms_total = 0.0f;
            (void)cudaEventElapsedTime(&ms_a, prof_ev[0], prof_ev[1]);
            (void)cudaEventElapsedTime(&ms_b, prof_ev[1], prof_ev[2]);
            (void)cudaEventElapsedTime(&ms_total, prof_ev[0], prof_ev[2]);
            fprintf(stderr,
                    "ds4: CUDA attention output profile tokens=%u groups=%u group_dim=%llu rank=%llu low=%llu out=%llu A=%.3f B=%.3f total=%.3f ms\n",
                    n_tokens,
                    n_groups,
                    (unsigned long long)group_dim,
                    (unsigned long long)rank,
                    (unsigned long long)low_dim,
                    (unsigned long long)out_dim,
                    ms_a,
                    ms_b,
                    ms_total);
        }
        for (uint32_t i = 0; i < 3u; i++) (void)cudaEventDestroy(prof_ev[i]);
    }
    return ok;
}
extern "C" int ds4_gpu_attention_output_low_q8_rows_exact_tensor(
        ds4_gpu_tensor       *low,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                out_a_offset,
        uint64_t                group_dim,
        uint64_t                rank,
        uint32_t                n_groups_total,
        uint32_t                group0,
        uint32_t                group_cnt,
        const ds4_gpu_tensor *heads,
        uint32_t                n_rows) {
    if (!low || !heads || !model_map || group_dim == 0 || rank == 0 ||
        n_groups_total == 0 || group_cnt == 0 ||
        group0 > n_groups_total || group_cnt > n_groups_total - group0 ||
        n_rows == 0 || (uint64_t)n_rows * group_cnt > 65535u) {
        return 0;
    }
    const uint64_t low_dim = (uint64_t)group_cnt * rank;
    const uint64_t blocks_a = (group_dim + 31) / 32;
    const uint64_t row_a_bytes = blocks_a * 34u;
    const uint64_t a_offset =
        out_a_offset + (uint64_t)group0 * rank * row_a_bytes;
    const uint64_t out_a_bytes = low_dim * row_a_bytes;
    if (a_offset < out_a_offset || a_offset > model_size ||
        out_a_bytes > model_size - a_offset ||
        heads->bytes < (uint64_t)n_rows * n_groups_total * group_dim *
            sizeof(float) ||
        low->bytes < (uint64_t)n_rows * low_dim * sizeof(float)) {
        return 0;
    }
    const int logical_tier = ds4_tensor_device_idx(low);
    const unsigned char *out_a = reinterpret_cast<const unsigned char *>(
            cuda_resolve_weight_ptr(model_map, a_offset, out_a_bytes,
                                    logical_tier, "attn_out_a_rows"));
    if (!out_a) return 0;

    const uint64_t x_rows = (uint64_t)n_rows * group_cnt;
    const uint64_t xq_bytes = x_rows * blocks_a * 32u;
    const uint64_t scale_offset = (xq_bytes + 15u) & ~15ull;
    const uint64_t tmp_bytes = scale_offset + x_rows * blocks_a * sizeof(float);
    void *tmp = cuda_tmp_alloc_on(logical_tier, tmp_bytes, "attention output low q8 prequant");
    if (!tmp) return 0;
    int8_t *xq = (int8_t *)tmp;
    float *xscale = (float *)((char *)tmp + scale_offset);
    const int use_dp4a = cuda_q8_use_dp4a();
    dim3 qgrid((unsigned)blocks_a, (unsigned)x_rows, 1);
    quantize_q8_0_group_slice_rows_kernel<<<qgrid, 32>>>(
            xq,
            xscale,
            (const float *)heads->ptr,
            group_dim,
            blocks_a,
            n_groups_total,
            group0,
            group_cnt);
    if (!cuda_ok(cudaGetLastError(),
                 "attention_output_low_q8 rows prequant launch")) return 0;
    dim3 grid_a(((unsigned)low_dim + 7u) / 8u, n_rows, 1u);
    grouped_q8_0_a_preq_warp8_kernel<<<grid_a, 256>>>((float *)low->ptr,
                                                      out_a,
                                                      xq,
                                                      xscale,
                                                      group_dim,
                                                      rank,
                                                      group_cnt,
                                                      n_rows,
                                                      blocks_a,
                                                      use_dp4a);
    return cuda_ok(cudaGetLastError(),
                   "attention_output_low_q8 rows launch");
}

extern "C" int ds4_gpu_attention_output_low_q8_tensor(
        ds4_gpu_tensor       *low,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                out_a_offset,
        uint64_t                group_dim,
        uint64_t                rank,
        uint32_t                n_groups,
        const ds4_gpu_tensor *heads) {
    return ds4_gpu_attention_output_low_q8_rows_exact_tensor(
            low, model_map, model_size, out_a_offset, group_dim, rank,
            n_groups, 0u, n_groups, heads, 1u);
}

extern "C" int ds4_gpu_attention_output_q8_tp_tensor(
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
    if (!out || !low || !heads || !model_map ||
        group_dim == 0 || rank == 0 || n_groups_total == 0 ||
        group_cnt == 0 || group0 > n_groups_total ||
        group_cnt > n_groups_total - group0 || out_dim == 0) {
        return 0;
    }
    const uint64_t blocks_a = (group_dim + 31u) / 32u;
    const uint64_t row_a_bytes = blocks_a * 34u;
    const uint64_t low_dim_total = (uint64_t)n_groups_total * rank;
    const uint64_t k_off = (uint64_t)group0 * rank;
    const uint64_t k_cnt = (uint64_t)group_cnt * rank;
    if ((k_off % 32u) != 0 || (k_cnt % 32u) != 0) return 0;
    if (heads->bytes < (uint64_t)(group0 + group_cnt) * group_dim * sizeof(float) ||
        low->bytes < k_cnt * sizeof(float) ||
        out->bytes < out_dim * sizeof(float)) {
        return 0;
    }

    ds4_gpu_tensor heads_slice = *heads;
    heads_slice.ptr = (char *)heads->ptr + (uint64_t)group0 * group_dim * sizeof(float);
    heads_slice.bytes = (uint64_t)group_cnt * group_dim * sizeof(float);
    heads_slice.owner = 0;

    const uint64_t a_off = out_a_offset + (uint64_t)group0 * rank * row_a_bytes;
    return ds4_gpu_attention_output_low_q8_tensor(low,
                                                  model_map,
                                                  model_size,
                                                  a_off,
                                                  group_dim,
                                                  rank,
                                                  group_cnt,
                                                  &heads_slice) &&
           ds4_gpu_matmul_q8_0_kslice_rows_tensor(out,
                                             model_map,
                                             model_size,
                                             out_b_offset,
                                             low_dim_total,
                                             out_dim,
                                             k_off,
                                             k_cnt,
                                             low,
                                             1);
}
extern "C" int ds4_gpu_swiglu_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *gate, const ds4_gpu_tensor *up, uint32_t n, float clamp, float weight) {
    if (!out || !gate || !up ||
        out->bytes < (uint64_t)n * sizeof(float) ||
        gate->bytes < (uint64_t)n * sizeof(float) ||
        up->bytes < (uint64_t)n * sizeof(float)) return 0;
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
    if (getenv("DS4_CUDA_DISABLE_SHARED_GATE_UP_PAIR") == NULL) {
        return ds4_gpu_matmul_q8_0_pair_tensor(gate, up,
                                                 model_map, model_size,
                                                 gate_offset, up_offset,
                                                 in_dim, out_dim, out_dim,
                                                 x, 1) &&
               ds4_gpu_swiglu_tensor(mid, gate, up, (uint32_t)out_dim, clamp, 1.0f);
    }
    return ds4_gpu_matmul_q8_0_tensor(gate, model_map, model_size,
                                        gate_offset, in_dim, out_dim, x, 1) &&
           ds4_gpu_matmul_q8_0_tensor(up, model_map, model_size,
                                        up_offset, in_dim, out_dim, x, 1) &&
           ds4_gpu_swiglu_tensor(mid, gate, up, (uint32_t)out_dim, clamp, 1.0f);
}

extern "C" int ds4_gpu_shared_mid_swiglu_q8_0_decode_exact_tensor(
        ds4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        float                   clamp,
        const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *prequant,
        uint32_t                expert_split,
        bool                    home_rank) {
    if (!mid || !x || !model_map || in_dim == 0u || out_dim == 0u ||
        x->bytes < in_dim * sizeof(float) ||
        mid->bytes < out_dim * sizeof(float) ||
        (selected && (selected->bytes < 6u * sizeof(int32_t) ||
                      expert_split == 0u))) {
        return 0;
    }
    const uint64_t blocks = (in_dim + 31u) / 32u;
    if (gate_offset > model_size || up_offset > model_size ||
        out_dim > UINT64_MAX / (blocks * 34u)) {
        return 0;
    }
    const uint64_t weight_bytes = out_dim * blocks * 34u;
    if (weight_bytes > model_size - gate_offset ||
        weight_bytes > model_size - up_offset) {
        return 0;
    }
    const int logical_tier = ds4_tensor_device_idx(x);
    if (logical_tier < 0 || logical_tier >= g_n_gpus) return 0;
    if (selected && ds4_tensor_device_idx(selected) != logical_tier) return 0;
    if (prequant && ds4_tensor_device_idx(prequant) != logical_tier) return 0;
    const int mid_tier = ds4_tensor_device_idx(mid);
    if (mid_tier != logical_tier && !g_gpu_peer_ok[logical_tier][mid_tier]) {
        return 0;
    }
    const char *gate_w = cuda_resolve_weight_ptr(
            model_map, gate_offset, weight_bytes, logical_tier,
            "shared_mid_gate_exact");
    const char *up_w = cuda_resolve_weight_ptr(
            model_map, up_offset, weight_bytes, logical_tier,
            "shared_mid_up_exact");
    if (!gate_w || !up_w) return 0;

    const uint64_t xq_bytes = blocks * 32u;
    const uint64_t scale_offset = (xq_bytes + 15u) & ~15ull;
    const uint64_t tmp_bytes = scale_offset + blocks * sizeof(float);
    int8_t *xq;
    float *xscale;
    if (prequant) {
        if (prequant->bytes < tmp_bytes) return 0;
        xq = (int8_t *)prequant->ptr;
        xscale = (float *)((char *)prequant->ptr + scale_offset);
    } else {
        void *tmp = cuda_tmp_alloc_on(logical_tier, tmp_bytes,
                                      "shared mid q8 exact prequant");
        if (!tmp) return 0;
        xq = (int8_t *)tmp;
        xscale = (float *)((char *)tmp + scale_offset);
        quantize_q8_0_f32_kernel<<<(unsigned)blocks, 32>>>(
                xq, xscale, (const float *)x->ptr, in_dim, blocks);
        if (!cuda_ok(cudaGetLastError(),
                     "shared mid q8 exact quantize launch")) {
            return 0;
        }
    }
    shared_mid_q8_0_preq_warp8_exact_kernel<<<
            ((unsigned)out_dim + 7u) / 8u, 256>>>(
            (float *)mid->ptr,
            (const unsigned char *)gate_w,
            (const unsigned char *)up_w,
            xq,
            xscale,
            in_dim,
            out_dim,
            blocks,
            clamp,
            selected ? (const int32_t *)selected->ptr : NULL,
            expert_split,
            home_rank,
            cuda_q8_use_dp4a());
    return cuda_ok(cudaGetLastError(), "shared mid q8 exact launch");
}
extern "C" int ds4_gpu_add_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *a, const ds4_gpu_tensor *b, uint32_t n) {
    if (!out || !a || !b ||
        out->bytes < (uint64_t)n * sizeof(float) ||
        a->bytes < (uint64_t)n * sizeof(float) ||
        b->bytes < (uint64_t)n * sizeof(float)) return 0;
    add_kernel<<<(n + 255) / 256, 256>>>((float *)out->ptr, (const float *)a->ptr, (const float *)b->ptr, n);
    return cuda_ok(cudaGetLastError(), "add launch");
}

extern "C" int ds4_gpu_add_xdev_tensor(ds4_gpu_tensor *out,
                                        const ds4_gpu_tensor *local,
                                        const ds4_gpu_tensor *remote,
                                        ds4_gpu_tensor *remote_tmp,
                                        uint32_t n) {
    if (!out || !local || !remote ||
        out->bytes < (uint64_t)n * sizeof(float) ||
        local->bytes < (uint64_t)n * sizeof(float) ||
        remote->bytes < (uint64_t)n * sizeof(float)) return 0;
    if (n == 0) return 1;

    const int od = ds4_tensor_device_idx(out);
    const int ld = ds4_tensor_device_idx(local);
    const int rd = ds4_tensor_device_idx(remote);
    if (od != ld) return 0;

    const ds4_gpu_tensor *rhs = remote;
    if (rd != od) {
        if (!remote_tmp ||
            remote_tmp->bytes < (uint64_t)n * sizeof(float) ||
            ds4_tensor_device_idx(remote_tmp) != od) return 0;
        if (!ds4_gpu_tensor_copy_xdev(remote_tmp, remote,
                                      (uint64_t)n * sizeof(float))) return 0;
        rhs = remote_tmp;
    }

    int ok = 0;
    WITH_DEVICE(g_gpu[od].device_id) {
        cudaStream_t s = (cudaStream_t)g_gpu[od].stream;
        add_kernel<<<(n + 255u) / 256u, 256, 0, s>>>(
                (float *)out->ptr,
                (const float *)local->ptr,
                (const float *)rhs->ptr,
                n);
        ok = cuda_ok(cudaGetLastError(), "xdev add launch");
        cudaEvent_t e = (cudaEvent_t)g_gpu[od].boundary_event;
        if (ok) ok = cuda_ok(cudaEventRecord(e, s), "xdev add event record");
        if (ok) ok = cuda_ok(cudaStreamWaitEvent(0, e, 0), "xdev add default wait");
        if (ok && g_xdev_sync_debug) {
            ok = cuda_ok(cudaStreamSynchronize(s), "xdev add sync");
        }
    }
    return ok;
}
extern "C" int ds4_gpu_directional_steering_project_tensor(
        ds4_gpu_tensor       *x,
        const ds4_gpu_tensor *directions,
        uint32_t                layer,
        uint32_t                width,
        uint32_t                rows,
        float                   scale) {
    if (!x || !directions || width == 0 || rows == 0 || scale == 0.0f) return 0;
    const uint64_t x_bytes = (uint64_t)width * rows * sizeof(float);
    const uint64_t dir_bytes = (uint64_t)(layer + 1u) * width * sizeof(float);
    if (x->bytes < x_bytes || directions->bytes < dir_bytes) return 0;

    uint32_t nth = 256u;
    while (nth > width && nth > 1u) nth >>= 1;
    directional_steering_project_kernel<<<rows, nth>>>(
            (float *)x->ptr,
            (const float *)directions->ptr,
            layer,
            width,
            rows,
            scale);
    return cuda_ok(cudaGetLastError(), "directional steering launch");
}
extern "C" int ds4_gpu_router_select_tensor(ds4_gpu_tensor *selected, ds4_gpu_tensor *weights, ds4_gpu_tensor *probs, const void *model_map, uint64_t model_size, uint64_t bias_offset, uint64_t hash_offset, uint32_t hash_rows, uint32_t token, uint32_t n_expert, uint32_t n_expert_used, float expert_weight_scale, uint32_t n_expert_groups, uint32_t n_group_used, bool has_bias, bool hash_mode, const ds4_gpu_tensor *logits) {
    if (!selected || !weights || !probs || !logits || !model_map || n_expert_groups > 1u || n_group_used > 0u) return 0;
    if (n_expert != 256u || n_expert_used != 6u || fabsf(expert_weight_scale - 1.5f) > 1.0e-6f) return 0;
    int32_t tok = (int32_t)token;
    int ok = 1;
    const float *bias = NULL;
    const int32_t *hash = NULL;
    const int logical_tier = ds4_tensor_device_idx(selected);
    if (ok && has_bias && !hash_mode) {
        if (bias_offset > model_size || model_size - bias_offset < 256u * sizeof(float)) ok = 0;
        else bias = (const float *)cuda_resolve_weight_ptr(model_map, bias_offset, 256u * sizeof(float), logical_tier, "router_bias");
        if (!bias) ok = 0;
    }
    if (ok && hash_mode) {
        const uint64_t hash_bytes = (uint64_t)hash_rows * 6u * sizeof(int32_t);
        if (hash_offset > model_size || hash_bytes > model_size - hash_offset) ok = 0;
        else hash = (const int32_t *)cuda_resolve_weight_ptr(model_map, hash_offset, hash_bytes, logical_tier, "router_hash");
        if (!hash) ok = 0;
    }
    if (ok) {
        if (getenv("DS4_CUDA_NO_WARP_ROUTER_SELECT") == NULL &&
            getenv("DS4_CUDA_NO_PARALLEL_ROUTER_SELECT") == NULL) {
            dim3 block(32, 4, 1);
            router_select_warp_topk_kernel<<<1, block>>>((int32_t *)selected->ptr, (float *)weights->ptr, (float *)probs->ptr,
                                                         bias, hash, (const float *)logits->ptr, NULL, tok, hash_rows, 1,
                                                         has_bias && !hash_mode, hash_mode);
        } else if (getenv("DS4_CUDA_NO_PARALLEL_ROUTER_SELECT") == NULL) {
            router_select_parallel_kernel<<<1, 256>>>((int32_t *)selected->ptr, (float *)weights->ptr, (float *)probs->ptr,
                                                      bias, hash, (const float *)logits->ptr, NULL, tok, hash_rows, 1,
                                                      has_bias && !hash_mode, hash_mode);
        } else {
            router_select_kernel<<<1, 1>>>((int32_t *)selected->ptr, (float *)weights->ptr, (float *)probs->ptr,
                                          bias, hash, (const float *)logits->ptr, NULL, tok, hash_rows, 1,
                                          has_bias && !hash_mode, hash_mode);
        }
        ok = cuda_ok(cudaGetLastError(), "router_select launch");
    }
    return ok;
}
extern "C" int ds4_gpu_router_select_batch_tensor(ds4_gpu_tensor *selected, ds4_gpu_tensor *weights, ds4_gpu_tensor *probs, const void *model_map, uint64_t model_size, uint64_t bias_offset, uint64_t hash_offset, uint32_t hash_rows, uint32_t n_expert_groups, uint32_t n_group_used, bool has_bias, bool hash_mode, const ds4_gpu_tensor *logits, const ds4_gpu_tensor *tokens, uint32_t n_expert, uint32_t n_expert_used, float expert_weight_scale, uint32_t n_tokens) {
    if (n_expert != 256u || n_expert_used != 6u || fabsf(expert_weight_scale - 1.5f) > 1.0e-6f) return 0;
    if (!selected || !weights || !probs || !logits || !tokens || !model_map || n_tokens == 0 ||
        n_expert_groups > 1u || n_group_used > 0u ||
        logits->bytes < (uint64_t)n_tokens * 256u * sizeof(float) ||
        probs->bytes < (uint64_t)n_tokens * 256u * sizeof(float) ||
        selected->bytes < (uint64_t)n_tokens * 6u * sizeof(int32_t) ||
        weights->bytes < (uint64_t)n_tokens * 6u * sizeof(float)) {
        return 0;
    }
    const float *bias = NULL;
    const int32_t *hash = NULL;
    const int logical_tier = ds4_tensor_device_idx(selected);
    if (has_bias && !hash_mode) {
        if (bias_offset > model_size || model_size - bias_offset < 256u * sizeof(float)) return 0;
        bias = (const float *)cuda_resolve_weight_ptr(model_map, bias_offset, 256u * sizeof(float), logical_tier, "router_bias");
        if (!bias) return 0;
    }
    if (hash_mode) {
        const uint64_t hash_bytes = (uint64_t)hash_rows * 6u * sizeof(int32_t);
        if (hash_offset > model_size || hash_bytes > model_size - hash_offset) return 0;
        hash = (const int32_t *)cuda_resolve_weight_ptr(model_map, hash_offset, hash_bytes, logical_tier, "router_hash");
        if (!hash) return 0;
    }
    if (getenv("DS4_CUDA_NO_WARP_ROUTER_SELECT") == NULL &&
        getenv("DS4_CUDA_NO_PARALLEL_ROUTER_SELECT") == NULL) {
        dim3 block(32, 4, 1);
        router_select_warp_topk_kernel<<<(n_tokens + 3u) / 4u, block>>>((int32_t *)selected->ptr,
                                                                        (float *)weights->ptr,
                                                                        (float *)probs->ptr,
                                                                        bias,
                                                                        hash,
                                                                        (const float *)logits->ptr,
                                                                        (const int32_t *)tokens->ptr,
                                                                        0,
                                                                        hash_rows,
                                                                        n_tokens,
                                                                        has_bias && !hash_mode,
                                                                        hash_mode);
    } else if (getenv("DS4_CUDA_NO_PARALLEL_ROUTER_SELECT") == NULL) {
        router_select_parallel_kernel<<<n_tokens, 256>>>((int32_t *)selected->ptr,
                                                         (float *)weights->ptr,
                                                         (float *)probs->ptr,
                                                         bias,
                                                         hash,
                                                         (const float *)logits->ptr,
                                                         (const int32_t *)tokens->ptr,
                                                         0,
                                                         hash_rows,
                                                         n_tokens,
                                                         has_bias && !hash_mode,
                                                         hash_mode);
    } else {
        router_select_kernel<<<n_tokens, 1>>>((int32_t *)selected->ptr,
                                              (float *)weights->ptr,
                                              (float *)probs->ptr,
                                              bias,
                                              hash,
                                              (const float *)logits->ptr,
                                              (const int32_t *)tokens->ptr,
                                              0,
                                              hash_rows,
                                              n_tokens,
                                              has_bias && !hash_mode,
                                              hash_mode);
    }
    return cuda_ok(cudaGetLastError(), "router_select launch");
}
