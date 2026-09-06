__global__ static void matmul_f16_tiny_batch_wave_kernel(
        float *out,
        const half *w,
        const float *x,
        uint32_t in_dim,
        uint32_t out_dim,
        uint32_t n_tok) {
    const uint32_t row = blockIdx.x;
    const uint32_t tok = blockIdx.y;
    const uint32_t lane = threadIdx.x;
    if (row >= out_dim || tok >= n_tok) return;

    const half *wr = w + (uint64_t)row * in_dim;
    const float *xr = x + (uint64_t)tok * in_dim;
    float sum = 0.0f;
    for (uint32_t i = lane; i < in_dim; i += 32u) {
        const float xv = __half2float(__float2half(xr[i]));
        sum += __half2float(wr[i]) * xv;
    }
    sum = warp_sum_f32(sum);
    if (lane == 0u) out[(uint64_t)tok * out_dim + row] = sum;
}

template <uint32_t BM, uint32_t BN>
__global__ static void matmul_f16_smallm_wmma_kernel(
        float *out, const half *w, const half *x,
        uint32_t m, uint32_t n, uint32_t k) {
    static_assert(BM % 16u == 0u && BN % 16u == 0u, "16x16 WMMA tiles");
    constexpr uint32_t WM = BM / 16u;
    constexpr uint32_t WN = BN / 16u;
    constexpr uint32_t WAVES = WM * WN;
    static_assert(WAVES <= 32u, "at most 1024 wave32 threads");

    __shared__ half sa[BM * 16u];
    __shared__ half sb[16u * BN];
    const uint32_t tid = threadIdx.x;
    const uint32_t wave = tid >> 5u;
    const uint32_t wm = wave % WM;
    const uint32_t wn = wave / WM;
    const uint32_t m0 = blockIdx.x * BM;
    const uint32_t n0 = blockIdx.y * BN;

    using fa_t = rocwmma::fragment<rocwmma::matrix_a, 16, 16, 16, half,
                                   rocwmma::row_major>;
    using fb_t = rocwmma::fragment<rocwmma::matrix_b, 16, 16, 16, half,
                                   rocwmma::col_major>;
    using fc_t = rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, float>;
    fa_t a;
    fb_t b;
    fc_t c;
    rocwmma::fill_fragment(c, 0.0f);

    for (uint32_t k0 = 0; k0 < k; k0 += 16u) {
        for (uint32_t p = tid; p < BM * 8u; p += WAVES * 32u) {
            const uint32_t j = p * 2u;
            const uint32_t kk = j & 15u;
            const uint32_t row = j >> 4u;
            *reinterpret_cast<uint32_t *>(sa + j) =
                    *reinterpret_cast<const uint32_t *>(
                            w + (k0 + kk) + (uint64_t)(m0 + row) * k);
        }
        for (uint32_t p = tid; p < 8u * BN; p += WAVES * 32u) {
            const uint32_t j = p * 2u;
            const uint32_t kk = j & 15u;
            const uint32_t col = j >> 4u;
            *reinterpret_cast<uint32_t *>(sb + j) =
                    *reinterpret_cast<const uint32_t *>(
                            x + (k0 + kk) + (uint64_t)(n0 + col) * k);
        }
        __syncthreads();
        rocwmma::load_matrix_sync(a, sa + wm * 16u * 16u, 16u);
        rocwmma::load_matrix_sync(b, sb + wn * 16u * 16u, 16u);
        rocwmma::mma_sync(c, a, b, c);
        __syncthreads();
    }
    rocwmma::store_matrix_sync(
            out + (m0 + wm * 16u) + (uint64_t)(n0 + wn * 16u) * m,
            c, m, rocwmma::mem_col_major);
}

__global__ static void matmul_f16_tinym24_wmma_kernel(
        float *out, const half *w, const half *x) {
    constexpr uint32_t M=24u, N=2048u, K=16384u, BM=32u, BN=64u, WAVES=8u;
    __shared__ half sa[BM*16u]; __shared__ half sb[16u*BN]; __shared__ float sc[BM*BN];
    const uint32_t tid=threadIdx.x, wave=tid>>5u, wm=wave&1u, wn=wave>>1u, n0=blockIdx.x*BN;
    using fa_t=rocwmma::fragment<rocwmma::matrix_a,16,16,16,half,rocwmma::col_major>;
    using fb_t=rocwmma::fragment<rocwmma::matrix_b,16,16,16,half,rocwmma::col_major>;
    using fc_t=rocwmma::fragment<rocwmma::accumulator,16,16,16,float>;
    fa_t a; fb_t b; fc_t c; rocwmma::fill_fragment(c,0.0f);
    for (uint32_t k0=0;k0<K;k0+=16u) {
        for (uint32_t j=tid;j<BM*16u;j+=WAVES*32u) { const uint32_t r=j&31u, k=j>>5u; sa[j]=r<M?w[(uint64_t)r*K+k0+k]:__float2half(0.0f); }
        for (uint32_t p=tid;p<8u*BN;p+=WAVES*32u) { const uint32_t j=p*2u,k=j&15u,n=j>>4u; *reinterpret_cast<uint32_t*>(sb+j)=*reinterpret_cast<const uint32_t*>(x+k0+k+(uint64_t)(n0+n)*K); }
        __syncthreads(); rocwmma::load_matrix_sync(a,sa+wm*16u,BM); rocwmma::load_matrix_sync(b,sb+wn*256u,16u); rocwmma::mma_sync(c,a,b,c); __syncthreads();
    }
    rocwmma::store_matrix_sync(sc+wm*16u+wn*16u*BM,c,BM,rocwmma::mem_col_major); __syncthreads();
    for (uint32_t i=tid;i<M*BN;i+=WAVES*32u) { const uint32_t n=i/M,r=i-n*M; out[r+(uint64_t)(n0+n)*M]=sc[r+n*BM]; }
}

template <uint32_t BT>
static void cuda_launch_q8_batch_sharedx_bt(
        float *out,
        const unsigned char *w,
        const float *x,
        uint32_t n_blocks,
        uint32_t out_dim,
        uint32_t n_tok,
        uint64_t row_bytes,
        dim3 grid,
        uint32_t rows_per_block,
        uint32_t tile) {
    const size_t shmem = (size_t)tile * BT * 32u * sizeof(float);
    if (tile == 2u) {
        matmul_q8_0_f32_batch_sharedx_warp_rows_w32_toktile_kernel<2u, BT><<<grid, rows_per_block * 32u, shmem>>>(out, w, x, n_blocks, out_dim, n_tok, row_bytes);
    } else if (tile == 3u) {
        matmul_q8_0_f32_batch_sharedx_warp_rows_w32_toktile_kernel<3u, BT><<<grid, rows_per_block * 32u, shmem>>>(out, w, x, n_blocks, out_dim, n_tok, row_bytes);
    } else if (tile == 4u) {
        matmul_q8_0_f32_batch_sharedx_warp_rows_w32_toktile_kernel<4u, BT><<<grid, rows_per_block * 32u, shmem>>>(out, w, x, n_blocks, out_dim, n_tok, row_bytes);
    } else if (tile == 5u) {
        matmul_q8_0_f32_batch_sharedx_warp_rows_w32_toktile_kernel<5u, BT><<<grid, rows_per_block * 32u, shmem>>>(out, w, x, n_blocks, out_dim, n_tok, row_bytes);
    } else if (tile == 8u) {
        matmul_q8_0_f32_batch_sharedx_warp_rows_w32_toktile_kernel<8u, BT><<<grid, rows_per_block * 32u, shmem>>>(out, w, x, n_blocks, out_dim, n_tok, row_bytes);
    } else if (tile == 16u) {
        matmul_q8_0_f32_batch_sharedx_warp_rows_w32_toktile_kernel<16u, BT><<<grid, rows_per_block * 32u, shmem>>>(out, w, x, n_blocks, out_dim, n_tok, row_bytes);
    } else {
        matmul_q8_0_f32_batch_sharedx_warp_rows_w32_toktile_kernel<32u, BT><<<grid, rows_per_block * 32u, shmem>>>(out, w, x, n_blocks, out_dim, n_tok, row_bytes);
    }
}

static void cuda_launch_q8_batch_sharedx(
        float *out,
        const unsigned char *w,
        const float *x,
        uint32_t n_blocks,
        uint32_t out_dim,
        uint32_t n_tok,
        uint64_t row_bytes,
        uint32_t rows_per_block,
        uint32_t tile,
        uint32_t block_tile) {
    const dim3 grid((out_dim + rows_per_block - 1u) / rows_per_block,
                    (n_tok + tile - 1u) / tile,
                    1u);
    if (block_tile == 8u) {
        cuda_launch_q8_batch_sharedx_bt<8u>(out, w, x, n_blocks, out_dim, n_tok, row_bytes, grid, rows_per_block, tile);
    } else if (block_tile == 32u) {
        cuda_launch_q8_batch_sharedx_bt<32u>(out, w, x, n_blocks, out_dim, n_tok, row_bytes, grid, rows_per_block, tile);
    } else {
        cuda_launch_q8_batch_sharedx_bt<16u>(out, w, x, n_blocks, out_dim, n_tok, row_bytes, grid, rows_per_block, tile);
    }
}

template <uint32_t BT>
static void cuda_launch_grouped_q8_a_sharedx_bt(
        float *low,
        const unsigned char *w,
        const float *heads,
        uint32_t n_tokens,
        uint32_t n_groups,
        uint32_t n_blocks,
        uint32_t rank,
        uint64_t row_bytes,
        dim3 grid,
        uint32_t rows_per_block,
        uint32_t tile) {
    const size_t shmem = (size_t)tile * BT * 32u * sizeof(float);
    if (tile == 2u) {
        grouped_q8_0_a_f32_batch_sharedx_chunked_w32_kernel<2u, BT><<<grid, rows_per_block * 32u, shmem>>>(low, w, heads, n_tokens, n_groups, n_blocks, rank, row_bytes);
    } else if (tile == 4u) {
        grouped_q8_0_a_f32_batch_sharedx_chunked_w32_kernel<4u, BT><<<grid, rows_per_block * 32u, shmem>>>(low, w, heads, n_tokens, n_groups, n_blocks, rank, row_bytes);
    } else if (tile == 8u) {
        grouped_q8_0_a_f32_batch_sharedx_chunked_w32_kernel<8u, BT><<<grid, rows_per_block * 32u, shmem>>>(low, w, heads, n_tokens, n_groups, n_blocks, rank, row_bytes);
    } else if (tile == 16u) {
        grouped_q8_0_a_f32_batch_sharedx_chunked_w32_kernel<16u, BT><<<grid, rows_per_block * 32u, shmem>>>(low, w, heads, n_tokens, n_groups, n_blocks, rank, row_bytes);
    } else {
        grouped_q8_0_a_f32_batch_sharedx_chunked_w32_kernel<32u, BT><<<grid, rows_per_block * 32u, shmem>>>(low, w, heads, n_tokens, n_groups, n_blocks, rank, row_bytes);
    }
}

static void cuda_launch_grouped_q8_a_sharedx(
        float *low,
        const unsigned char *w,
        const float *heads,
        uint32_t n_tokens,
        uint32_t n_groups,
        uint32_t n_blocks,
        uint32_t rank,
        uint64_t row_bytes,
        uint32_t rows_per_block,
        uint32_t tile,
        uint32_t block_tile) {
    const uint32_t row_blocks = (rank + rows_per_block - 1u) / rows_per_block;
    const dim3 grid(n_groups * row_blocks,
                    (n_tokens + tile - 1u) / tile,
                    1u);
    if (block_tile == 8u) {
        cuda_launch_grouped_q8_a_sharedx_bt<8u>(low, w, heads, n_tokens, n_groups, n_blocks, rank, row_bytes, grid, rows_per_block, tile);
    } else if (block_tile == 32u) {
        cuda_launch_grouped_q8_a_sharedx_bt<32u>(low, w, heads, n_tokens, n_groups, n_blocks, rank, row_bytes, grid, rows_per_block, tile);
    } else {
        cuda_launch_grouped_q8_a_sharedx_bt<16u>(low, w, heads, n_tokens, n_groups, n_blocks, rank, row_bytes, grid, rows_per_block, tile);
    }
}

template <uint32_t BT>
static void cuda_launch_grouped_q8_a_sharedx_strided_bt(
        float *low,
        const unsigned char *w,
        const float *heads,
        uint32_t n_tokens,
        uint32_t n_groups,
        uint32_t n_blocks,
        uint32_t rank,
        uint32_t x_token_stride,
        uint32_t x_group_stride,
        uint64_t row_bytes,
        dim3 grid,
        uint32_t rows_per_block,
        uint32_t tile) {
    const size_t shmem = (size_t)tile * BT * 32u * sizeof(float);
    if (tile == 2u) {
        grouped_q8_0_a_f32_batch_sharedx_chunked_strided_w32_kernel<2u, BT>
            <<<grid, rows_per_block * 32u, shmem>>>(
                low, w, heads, n_tokens, n_groups, n_blocks, rank,
                x_token_stride, x_group_stride, row_bytes);
    } else if (tile == 4u) {
        grouped_q8_0_a_f32_batch_sharedx_chunked_strided_w32_kernel<4u, BT>
            <<<grid, rows_per_block * 32u, shmem>>>(
                low, w, heads, n_tokens, n_groups, n_blocks, rank,
                x_token_stride, x_group_stride, row_bytes);
    } else if (tile == 8u) {
        grouped_q8_0_a_f32_batch_sharedx_chunked_strided_w32_kernel<8u, BT>
            <<<grid, rows_per_block * 32u, shmem>>>(
                low, w, heads, n_tokens, n_groups, n_blocks, rank,
                x_token_stride, x_group_stride, row_bytes);
    } else if (tile == 16u) {
        grouped_q8_0_a_f32_batch_sharedx_chunked_strided_w32_kernel<16u, BT>
            <<<grid, rows_per_block * 32u, shmem>>>(
                low, w, heads, n_tokens, n_groups, n_blocks, rank,
                x_token_stride, x_group_stride, row_bytes);
    } else {
        grouped_q8_0_a_f32_batch_sharedx_chunked_strided_w32_kernel<32u, BT>
            <<<grid, rows_per_block * 32u, shmem>>>(
                low, w, heads, n_tokens, n_groups, n_blocks, rank,
                x_token_stride, x_group_stride, row_bytes);
    }
}

static void cuda_launch_grouped_q8_a_sharedx_strided(
        float *low,
        const unsigned char *w,
        const float *heads,
        uint32_t n_tokens,
        uint32_t n_groups,
        uint32_t n_blocks,
        uint32_t rank,
        uint32_t x_token_stride,
        uint32_t x_group_stride,
        uint64_t row_bytes,
        uint32_t rows_per_block,
        uint32_t tile,
        uint32_t block_tile) {
    const uint32_t row_blocks =
        (rank + rows_per_block - 1u) / rows_per_block;
    const dim3 grid(n_groups * row_blocks,
                    (n_tokens + tile - 1u) / tile,
                    1u);
    if (block_tile == 8u) {
        cuda_launch_grouped_q8_a_sharedx_strided_bt<8u>(
            low, w, heads, n_tokens, n_groups, n_blocks, rank,
            x_token_stride, x_group_stride, row_bytes, grid,
            rows_per_block, tile);
    } else if (block_tile == 32u) {
        cuda_launch_grouped_q8_a_sharedx_strided_bt<32u>(
            low, w, heads, n_tokens, n_groups, n_blocks, rank,
            x_token_stride, x_group_stride, row_bytes, grid,
            rows_per_block, tile);
    } else {
        cuda_launch_grouped_q8_a_sharedx_strided_bt<16u>(
            low, w, heads, n_tokens, n_groups, n_blocks, rank,
            x_token_stride, x_group_stride, row_bytes, grid,
            rows_per_block, tile);
    }
}

static int cuda_matmul_q8_0_tensor_f16_gemm(
        ds4_gpu_tensor *out,
        const void *model_map,
        uint64_t model_size,
        uint64_t weight_offset,
        uint64_t in_dim,
        uint64_t out_dim,
        const ds4_gpu_tensor *x,
        uint64_t n_tok,
        const char *label) {
    if (!g_cublas_ready || !out || !x || !model_map ||
        in_dim == 0u || out_dim == 0u || n_tok == 0u ||
        in_dim > UINT32_MAX || out_dim > UINT32_MAX || n_tok > UINT32_MAX) return 0;
    const uint64_t blocks = (in_dim + 31u) / 32u;
    uint64_t row_bytes = 0, weight_bytes = 0, x_bytes = 0, out_bytes = 0;
    if (weight_offset > model_size ||
        !cuda_u64_mul_checked(blocks, 34u, &row_bytes) ||
        !cuda_u64_mul_checked(out_dim, row_bytes, &weight_bytes) ||
        weight_bytes > model_size - weight_offset ||
        !cuda_u64_mul3_checked(n_tok, in_dim, sizeof(float), &x_bytes) ||
        !cuda_u64_mul3_checked(n_tok, out_dim, sizeof(float), &out_bytes) ||
        x->bytes < x_bytes || out->bytes < out_bytes) return 0;
    const __half *w_f16 = cuda_q8_f16_ptr(model_map, weight_offset, weight_bytes, in_dim, out_dim, label);
    if (!w_f16) return 0;
    const uint64_t xh_count = n_tok * in_dim;
    __half *xh = (__half *)cuda_tmp_alloc(xh_count * sizeof(__half), "q8 f16 gemm activations");
    if (!xh) return 0;
    f32_to_f16_kernel<<<(xh_count + 255u) / 256u, 256>>>(xh, (const float *)x->ptr, xh_count);
    if (!cuda_ok(cudaGetLastError(), "q8 f16 activation convert launch")) return 0;
    if (in_dim == 16384u && out_dim == 24u && n_tok == 2048u &&
        ds4_rocm_gfx1151_flag("DS4_ROCM_F16_TINYM_WMMA")) {
        matmul_f16_tinym24_wmma_kernel<<<32u, 256u>>>((float *)out->ptr, w_f16, xh);
        return cuda_ok(cudaGetLastError(), "q8 f16 tinym24 wmma launch");
    }
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasStatus_t st = cublasGemmEx(g_cublas,
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
                                     CUBLAS_COMPUTE_32F,
                                     CUBLAS_GEMM_DEFAULT);
    if (st == CUBLAS_STATUS_SUCCESS) return 1;
    fprintf(stderr, "ds4: " DS4_GPU_BLAS_NAME " q8 f16 matmul failed: status %d\n", (int)st);
    cuda_q8_f16_cache_disable_after_failure(DS4_GPU_BLAS_NAME " f16 matmul failure",
                                            in_dim * out_dim * sizeof(__half));
    return 0;
}

static int cuda_matmul_q8_0_tensor_f16_gemm_out_half(
        ds4_gpu_tensor *out_h,
        const void *model_map,
        uint64_t model_size,
        uint64_t weight_offset,
        uint64_t in_dim,
        uint64_t out_dim,
        const ds4_gpu_tensor *x,
        uint64_t n_tok,
        const char *label) {
    if (!g_cublas_ready || !out_h || !x || !model_map ||
        in_dim == 0u || out_dim == 0u || n_tok == 0u ||
        in_dim > UINT32_MAX || out_dim > UINT32_MAX || n_tok > UINT32_MAX) return 0;
    const uint64_t blocks = (in_dim + 31u) / 32u;
    uint64_t row_bytes = 0, weight_bytes = 0, x_bytes = 0, out_bytes = 0;
    if (weight_offset > model_size ||
        !cuda_u64_mul_checked(blocks, 34u, &row_bytes) ||
        !cuda_u64_mul_checked(out_dim, row_bytes, &weight_bytes) ||
        weight_bytes > model_size - weight_offset ||
        !cuda_u64_mul3_checked(n_tok, in_dim, sizeof(float), &x_bytes) ||
        !cuda_u64_mul3_checked(n_tok, out_dim, sizeof(__half), &out_bytes) ||
        x->bytes < x_bytes || out_h->bytes < out_bytes) return 0;
    const __half *w_f16 = cuda_q8_f16_ptr(model_map, weight_offset, weight_bytes, in_dim, out_dim, label);
    if (!w_f16) return 0;
    const uint64_t xh_count = n_tok * in_dim;
    __half *xh = (__half *)cuda_tmp_alloc(xh_count * sizeof(__half), "q8 f16-out gemm activations");
    if (!xh) return 0;
    f32_to_f16_kernel<<<(xh_count + 255u) / 256u, 256>>>(xh, (const float *)x->ptr, xh_count);
    if (!cuda_ok(cudaGetLastError(), "q8 f16-out activation convert launch")) return 0;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasStatus_t st = cublasGemmEx(g_cublas,
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
                                     out_h->ptr,
                                     CUDA_R_16F,
                                     (int)out_dim,
                                     CUBLAS_COMPUTE_32F,
                                     CUBLAS_GEMM_DEFAULT);
    if (st == CUBLAS_STATUS_SUCCESS) return 1;
    fprintf(stderr, "ds4: " DS4_GPU_BLAS_NAME " q8 f16-out matmul failed: status %d\n", (int)st);
    cuda_q8_f16_cache_disable_after_failure(DS4_GPU_BLAS_NAME " f16-out matmul failure",
                                            in_dim * out_dim * sizeof(__half));
    return 0;
}

extern "C" int ds4_gpu_matmul_q8_0_f16_out_tensor(
        ds4_gpu_tensor       *out_h,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok) {
    return cuda_matmul_q8_0_tensor_f16_gemm_out_half(out_h, model_map, model_size,
                                                     weight_offset, in_dim, out_dim,
                                                     x, n_tok, "q8_f16_out");
}

static int cuda_matmul_q8_0_tensor_labeled(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim, const ds4_gpu_tensor *x, uint64_t n_tok, const char *label) {
    if (!out || !x || !model_map ||
        in_dim == 0u || out_dim == 0u || n_tok == 0u ||
        in_dim > UINT32_MAX || out_dim > UINT32_MAX || n_tok > UINT32_MAX) return 0;
    uint64_t blocks = (in_dim + 31u) / 32u;
    uint64_t row_bytes = 0, weight_bytes = 0, x_bytes = 0, out_bytes = 0;
    if (weight_offset > model_size ||
        !cuda_u64_mul_checked(blocks, 34u, &row_bytes) ||
        !cuda_u64_mul_checked(out_dim, row_bytes, &weight_bytes) ||
        weight_bytes > model_size - weight_offset ||
        !cuda_u64_mul3_checked(n_tok, in_dim, sizeof(float), &x_bytes) ||
        !cuda_u64_mul3_checked(n_tok, out_dim, sizeof(float), &out_bytes) ||
        x->bytes < x_bytes || out->bytes < out_bytes) return 0;
    if (n_tok > 1 && !g_quality_mode &&
        cuda_runtime_config()->shared_down_cublas && in_dim == 2048u && out_dim == 4096u &&
        cuda_matmul_q8_0_tensor_f16_gemm(out, model_map, model_size, weight_offset,
                                         in_dim, out_dim, x, n_tok, label ? label : "shared_expert")) {
        return 1;
    }
    const char *wptr = cuda_model_range_ptr(model_map, weight_offset, weight_bytes, "q8_0");
    if (!wptr) return 0;
    if (n_tok == 1 && !cuda_q8_prequant_decode_enabled()) {
        const bool extended_sharedx =
            in_dim > 8192u &&
            in_dim <= 16384u &&
            cuda_runtime_config()->q8_decode_sharedx_64k;
        if ((in_dim & 31u) == 0u &&
            (in_dim <= 8192u || extended_sharedx)) {
            const unsigned rows_per_block = 32u;
            const unsigned threads = rows_per_block * 32u;
            matmul_q8_0_f32_sharedx_warp_rows_w32_kernel<<<
                    (unsigned)((out_dim + rows_per_block - 1u) / rows_per_block),
                    threads,
                    (size_t)in_dim * sizeof(float)>>>(
                    (float *)out->ptr,
                    reinterpret_cast<const unsigned char *>(wptr),
                    (const float *)x->ptr,
                    (uint32_t)blocks,
                    out_dim,
                    blocks * 34u);
            const cudaError_t launch_err = cudaGetLastError();
            if (launch_err == cudaSuccess) {
                if (extended_sharedx) {
                    static int notice_printed = 0;
                    if (!notice_printed) {
                        fprintf(stderr,
                                DS4_GPU_LOG_PREFIX
                                "Q8 one-token shared-input kernel enabled "
                                "through 64 KiB LDS (in_dim=%llu)\n",
                                (unsigned long long)in_dim);
                        notice_printed = 1;
                    }
                }
                return 1;
            }
            if (!extended_sharedx) {
                return cuda_ok(launch_err,
                               "matmul_q8_0 f32 sharedx launch");
            }
            static int fallback_notice_printed = 0;
            if (!fallback_notice_printed) {
                fprintf(stderr,
                        DS4_GPU_LOG_PREFIX
                        "Q8 64 KiB shared-input launch unavailable "
                        "(%s); falling back to the warp-row kernel\n",
                        cudaGetErrorString(launch_err));
                fallback_notice_printed = 1;
            }
        }
        matmul_q8_0_f32_warp8_kernel<<<((unsigned)out_dim + 7u) / 8u, 256>>>(
                (float *)out->ptr,
                reinterpret_cast<const unsigned char *>(wptr),
                (const float *)x->ptr,
                in_dim,
                out_dim,
                blocks);
        return cuda_ok(cudaGetLastError(), "matmul_q8_0 f32 warp launch");
    }
    if (n_tok > 1) {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
        if (!g_quality_mode &&
            g_dspark_verify_mode && n_tok <= 6u &&
            in_dim <= INT_MAX && out_dim <= INT_MAX &&
            ds4_rocm_gfx1151_flag("DS4_ROCM_DSPARK_Q8_MMVQ") &&
            ds4_mmq_init(0) == 0 &&
            ds4_mmq_q8_0_dense_vec(
                wptr, (const float *)x->ptr, (float *)out->ptr,
                (int)out_dim, (int)n_tok, (int)in_dim,
                (cudaStream_t)0) == 0) {
            return 1;
        }
        /* The 2048-row projection hits a 72-VGPR occupancy cliff in exact8;
         * its retained 24-VGPR kernel is 2.4x faster on gfx1151. */
        if (!g_quality_mode && (in_dim % 256u) == 0u &&
            n_tok <= 5u && out_dim != 2048u && out_dim <= UINT32_MAX) {
            constexpr uint32_t rows_per_block = 32u;
            const dim3 grid(((uint32_t)out_dim + rows_per_block - 1u) /
                                rows_per_block,
                            1u, 1u);
            const size_t shmem = (size_t)n_tok * 8u * 32u * sizeof(float);
            if (n_tok == 2u) {
                matmul_q8_0_f32_batch_sharedx_exact8_kernel<2u>
                    <<<grid, rows_per_block * 32u, shmem>>>(
                        (float *)out->ptr,
                        reinterpret_cast<const unsigned char *>(wptr),
                        (const float *)x->ptr, (uint32_t)blocks,
                        (uint32_t)out_dim, blocks * 34u);
            } else if (n_tok == 3u) {
                matmul_q8_0_f32_batch_sharedx_exact8_kernel<3u>
                    <<<grid, rows_per_block * 32u, shmem>>>(
                        (float *)out->ptr,
                        reinterpret_cast<const unsigned char *>(wptr),
                        (const float *)x->ptr, (uint32_t)blocks,
                        (uint32_t)out_dim, blocks * 34u);
            } else if (n_tok == 4u) {
                matmul_q8_0_f32_batch_sharedx_exact8_kernel<4u>
                    <<<grid, rows_per_block * 32u, shmem>>>(
                        (float *)out->ptr,
                        reinterpret_cast<const unsigned char *>(wptr),
                        (const float *)x->ptr, (uint32_t)blocks,
                        (uint32_t)out_dim, blocks * 34u);
            } else {
                matmul_q8_0_f32_batch_sharedx_exact8_kernel<5u>
                    <<<grid, rows_per_block * 32u, shmem>>>(
                        (float *)out->ptr,
                        reinterpret_cast<const unsigned char *>(wptr),
                        (const float *)x->ptr, (uint32_t)blocks,
                        (uint32_t)out_dim, blocks * 34u);
            }
            return cuda_ok(cudaGetLastError(),
                           "matmul_q8_0 f32 tiny exact8 launch");
        }
        if (!g_quality_mode && (in_dim % 32u) == 0u &&
            out_dim >= 1024u &&
            n_tok >= 256u &&
            in_dim <= UINT32_MAX && out_dim <= UINT32_MAX && n_tok <= UINT32_MAX) {
            if (out_dim >= 8192u) {
                const dim3 grid((uint32_t)((out_dim + 255u) / 256u),
                                (uint32_t)((n_tok + 63u) / 64u),
                                1u);
                matmul_q8_0_f32_batch_wmma_rowtile_kernel<256u, 16u><<<grid, 512u>>>(
                        (float *)out->ptr,
                        reinterpret_cast<const unsigned char *>(wptr),
                        (const float *)x->ptr,
                        (uint32_t)n_tok,
                        (uint32_t)in_dim,
                        (uint32_t)out_dim,
                        blocks * 34u);
                return cuda_ok(cudaGetLastError(), "matmul_q8_0 f32 batch wmma 16w launch");
            }
            const dim3 grid((uint32_t)((out_dim + 127u) / 128u),
                            (uint32_t)((n_tok + 63u) / 64u),
                            1u);
            matmul_q8_0_f32_batch_wmma_rowtile_kernel<128u, 8u><<<grid, 256u>>>(
                    (float *)out->ptr,
                    reinterpret_cast<const unsigned char *>(wptr),
                    (const float *)x->ptr,
                    (uint32_t)n_tok,
                    (uint32_t)in_dim,
                    (uint32_t)out_dim,
                    blocks * 34u);
            return cuda_ok(cudaGetLastError(), "matmul_q8_0 f32 batch wmma 8w launch");
        }
#endif
        if ((in_dim & 31u) == 0u && out_dim <= UINT32_MAX && n_tok <= UINT32_MAX) {
            const uint32_t rows_per_block = 32u;
            const bool exact_tiny = n_tok <= 5u;
            const uint32_t tile = exact_tiny ? (uint32_t)n_tok :
                                  n_tok <= 2u ? 2u :
                                  n_tok <= 4u ? 4u :
                                  n_tok <= 8u ? 8u :
                                  n_tok <= 16u ? 16u : 32u;
            const uint32_t block_tile = n_tok <= 8u ? 8u : 16u;
            cuda_launch_q8_batch_sharedx((float *)out->ptr,
                                         reinterpret_cast<const unsigned char *>(wptr),
                                         (const float *)x->ptr,
                                         (uint32_t)blocks,
                                         (uint32_t)out_dim,
                                         (uint32_t)n_tok,
                                         blocks * 34u,
                                         rows_per_block,
                                         tile,
                                         block_tile);
            return cuda_ok(cudaGetLastError(), "matmul_q8_0 f32 batch sharedx launch");
        }
        dim3 bgrid(((unsigned)out_dim + 7u) / 8u, (unsigned)n_tok, 1);
        matmul_q8_0_f32_batch_warp8_kernel<<<bgrid, 256>>>(
                (float *)out->ptr,
                reinterpret_cast<const unsigned char *>(wptr),
                (const float *)x->ptr,
                in_dim,
                out_dim,
                n_tok,
                blocks);
        return cuda_ok(cudaGetLastError(), "matmul_q8_0 f32 batch warp launch");
    }
    if (g_cublas_ready && n_tok > 1) {
        const __half *w_f16 = cuda_q8_f16_ptr(model_map, weight_offset, weight_bytes, in_dim, out_dim, label);
        if (w_f16) {
            const uint64_t xh_count = n_tok * in_dim;
            __half *xh = (__half *)cuda_tmp_alloc(xh_count * sizeof(__half), "q8 f16 gemm activations");
            if (!xh) return 0;
            f32_to_f16_kernel<<<(xh_count + 255) / 256, 256>>>(xh, (const float *)x->ptr, xh_count);
            if (!cuda_ok(cudaGetLastError(), "q8 f16 activation convert launch")) return 0;
            const float alpha = 1.0f;
            const float beta = 0.0f;
            cublasStatus_t st = cublasGemmEx(g_cublas,
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
                                             CUBLAS_COMPUTE_32F,
                                             CUBLAS_GEMM_DEFAULT);
            if (st == CUBLAS_STATUS_SUCCESS) return 1;
            fprintf(stderr, "ds4: " DS4_GPU_BLAS_NAME " q8 f16 matmul failed: status %d\n", (int)st);
            cuda_q8_f16_cache_disable_after_failure(DS4_GPU_BLAS_NAME " f16 matmul failure",
                                                    in_dim * out_dim * sizeof(__half));
            /* The F16 expansion cache is only an optimization.  If cuBLAS
             * rejects the cached path under memory pressure, retry the same
             * operation through the native Q8 kernels below. */
        }
    }
    const uint64_t xq_bytes = n_tok * blocks * 32u;
    const uint64_t scale_offset = (xq_bytes + 15u) & ~15ull;
    const uint64_t tmp_bytes = scale_offset + n_tok * blocks * sizeof(float);
    void *tmp = cuda_tmp_alloc(tmp_bytes, "q8_0 prequant");
    if (!tmp) return 0;
    int8_t *xq = (int8_t *)tmp;
    float *xscale = (float *)((char *)tmp + scale_offset);
    const ds4_rocm_runtime_config *cfg = cuda_runtime_config();
    const int use_dp4a = 1;
    dim3 qgrid((unsigned)blocks, (unsigned)n_tok, 1);
    quantize_q8_0_f32_kernel<<<qgrid, 32>>>(xq, xscale, (const float *)x->ptr, in_dim, blocks);
    if (!cuda_ok(cudaGetLastError(), "matmul_q8_0 quantize launch")) return 0;
    if (n_tok == 1) {
        const uint32_t rows_per_block = cfg->q8_decode_rpb;
        matmul_q8_0_preq_rows_w32_kernel<<<
                ((unsigned)out_dim + rows_per_block - 1u) / rows_per_block,
                rows_per_block * 32u>>>(
                (float *)out->ptr,
                reinterpret_cast<const unsigned char *>(wptr),
                xq,
                xscale,
                in_dim,
                out_dim,
                blocks,
                rows_per_block,
                use_dp4a);
        return cuda_ok(cudaGetLastError(), "matmul_q8_0 rows launch");
    }
    if (blocks <= 32u) {
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
    dim3 grid((unsigned)out_dim, (unsigned)n_tok, 1);
    matmul_q8_0_preq_kernel<<<grid, 256>>>((float *)out->ptr,
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

extern "C" int ds4_gpu_matmul_q8_0_decode_mpp_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok) {
    return cuda_matmul_q8_0_tensor_labeled(out,
                                           model_map,
                                           model_size,
                                           weight_offset,
                                           in_dim,
                                           out_dim,
                                           x,
                                           n_tok,
                                           "q8_0_decode");
}

extern "C" int ds4_gpu_matmul_q8_0_decode_mpp_model_view_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok) {
    return cuda_matmul_q8_0_tensor_labeled(out,
                                           model_map,
                                           model_size,
                                           weight_offset,
                                           in_dim,
                                           out_dim,
                                           x,
                                           n_tok,
                                           "q8_0_decode_model_view");
}

extern "C" int ds4_gpu_matmul_q8_0_rows_scalar_tensor(
        ds4_gpu_tensor *out,
        const void *model_map,
        uint64_t model_size,
        uint64_t weight_offset,
        uint64_t in_dim,
        uint64_t out_dim,
        const ds4_gpu_tensor *x,
        uint64_t n_tok) {
    (void)out; (void)model_map; (void)model_size; (void)weight_offset;
    (void)in_dim; (void)out_dim; (void)x; (void)n_tok;
    return 0;
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
    if (!out0 || !out1 || !x || !model_map ||
        in_dim == 0 || out0_dim == 0 || out1_dim == 0 || n_tok == 0 ||
        in_dim > UINT32_MAX || out0_dim > UINT32_MAX || out1_dim > UINT32_MAX || n_tok > UINT32_MAX) {
        return 0;
    }
    if (n_tok != 1) {
        return cuda_matmul_q8_0_tensor_labeled(out0, model_map, model_size, weight0_offset,
                                               in_dim, out0_dim, x, n_tok, "q8_0_pair0") &&
               cuda_matmul_q8_0_tensor_labeled(out1, model_map, model_size, weight1_offset,
                                               in_dim, out1_dim, x, n_tok, "q8_0_pair1");
    }
    const uint64_t blocks = (in_dim + 31u) / 32u;
    uint64_t row_bytes = 0, weight0_bytes = 0, weight1_bytes = 0;
    if (weight0_offset > model_size || weight1_offset > model_size ||
        !cuda_u64_mul_checked(blocks, 34u, &row_bytes) ||
        !cuda_u64_mul_checked(out0_dim, row_bytes, &weight0_bytes) ||
        !cuda_u64_mul_checked(out1_dim, row_bytes, &weight1_bytes)) {
        return 0;
    }
    if (weight0_bytes > model_size - weight0_offset ||
        weight1_bytes > model_size - weight1_offset ||
        x->bytes < in_dim * sizeof(float) ||
        out0->bytes < out0_dim * sizeof(float) ||
        out1->bytes < out1_dim * sizeof(float)) {
        return 0;
    }
    const char *w0 = cuda_model_range_ptr(model_map, weight0_offset, weight0_bytes, "q8_0_pair0");
    const char *w1 = cuda_model_range_ptr(model_map, weight1_offset, weight1_bytes, "q8_0_pair1");
    if (!w0 || !w1) return 0;
    if (!cuda_q8_prequant_decode_enabled()) {
        const uint64_t max_out = out0_dim > out1_dim ? out0_dim : out1_dim;
        if ((in_dim & 31u) == 0u && in_dim <= 8192u) {
            const unsigned rows_per_block = 32u;
            const unsigned threads = rows_per_block * 32u;
            matmul_q8_0_pair_f32_sharedx_warp_rows_w32_kernel<<<
                    (unsigned)((max_out + rows_per_block - 1u) / rows_per_block),
                    threads,
                    (size_t)in_dim * sizeof(float)>>>(
                    (float *)out0->ptr,
                    (float *)out1->ptr,
                    reinterpret_cast<const unsigned char *>(w0),
                    reinterpret_cast<const unsigned char *>(w1),
                    (const float *)x->ptr,
                    (uint32_t)blocks,
                    out0_dim,
                    out1_dim,
                    blocks * 34u);
            return cuda_ok(cudaGetLastError(), "matmul_q8_0 pair f32 sharedx launch");
        }
        matmul_q8_0_pair_f32_warp8_kernel<<<((unsigned)max_out + 7u) / 8u, 256>>>(
                (float *)out0->ptr,
                (float *)out1->ptr,
                reinterpret_cast<const unsigned char *>(w0),
                reinterpret_cast<const unsigned char *>(w1),
                (const float *)x->ptr,
                in_dim,
                out0_dim,
                out1_dim,
                blocks);
        return cuda_ok(cudaGetLastError(), "matmul_q8_0 pair f32 warp launch");
    }

    const uint64_t xq_bytes = blocks * 32u;
    const uint64_t scale_offset = (xq_bytes + 15u) & ~15ull;
    const uint64_t tmp_bytes = scale_offset + blocks * sizeof(float);
    void *tmp = cuda_tmp_alloc(tmp_bytes, "q8_0 pair prequant");
    if (!tmp) return 0;
    int8_t *xq = (int8_t *)tmp;
    float *xscale = (float *)((char *)tmp + scale_offset);
    const int use_dp4a = 1;
    dim3 qgrid((unsigned)blocks, 1, 1);
    quantize_q8_0_f32_kernel<<<qgrid, 32>>>(
            xq, xscale, (const float *)x->ptr, in_dim, blocks);
    if (!cuda_ok(cudaGetLastError(), "matmul_q8_0 pair quantize launch")) {
        return 0;
    }
    const uint64_t max_out = out0_dim > out1_dim ? out0_dim : out1_dim;
    matmul_q8_0_pair_preq_warp8_kernel<<<
            ((unsigned)max_out + 7u) / 8u, 256>>>(
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
        (block_add && block_add->bytes < out_dim * sizeof(float))) {
        return 0;
    }
    const char *wptr = cuda_model_range_ptr(model_map, weight_offset, weight_bytes, label ? label : "q8_0_hc_expand");
    if (!wptr) return 0;
    if (!cuda_q8_prequant_decode_enabled()) {
        if ((in_dim & 31u) == 0u && in_dim <= 8192u) {
            const unsigned rows_per_block = 32u;
            const unsigned threads = rows_per_block * 32u;
            matmul_q8_0_hc_expand_f32_sharedx_warp_rows_w32_kernel<<<
                    (unsigned)((out_dim + rows_per_block - 1u) / rows_per_block),
                    threads,
                    (size_t)in_dim * sizeof(float)>>>(
                    (float *)out_hc->ptr,
                    (float *)block_out->ptr,
                    block_add ? (const float *)block_add->ptr : (const float *)block_out->ptr,
                    (const float *)residual_hc->ptr,
                    (const float *)split->ptr,
                    reinterpret_cast<const unsigned char *>(wptr),
                    (const float *)x->ptr,
                    (uint32_t)blocks,
                    out_dim,
                    blocks * 34u,
                    n_embd,
                    n_hc,
                    block_add ? 1 : 0);
            return cuda_ok(cudaGetLastError(), "matmul_q8_0_hc_expand f32 sharedx launch");
        }
        matmul_q8_0_hc_expand_f32_warp8_kernel<<<
                ((unsigned)out_dim + 7u) / 8u, 256>>>(
                (float *)out_hc->ptr,
                (float *)block_out->ptr,
                block_add ? (const float *)block_add->ptr : (const float *)block_out->ptr,
                (const float *)residual_hc->ptr,
                (const float *)split->ptr,
                reinterpret_cast<const unsigned char *>(wptr),
                (const float *)x->ptr,
                in_dim,
                out_dim,
                n_embd,
                n_hc,
                blocks,
                block_add ? 1 : 0);
        return cuda_ok(cudaGetLastError(), "matmul_q8_0_hc_expand f32 launch");
    }

    const uint64_t xq_bytes = blocks * 32u;
    const uint64_t scale_offset = (xq_bytes + 15u) & ~15ull;
    const uint64_t tmp_bytes = scale_offset + blocks * sizeof(float);
    void *tmp = cuda_tmp_alloc(tmp_bytes, "q8_0 hc expand prequant");
    if (!tmp) return 0;
    int8_t *xq = (int8_t *)tmp;
    float *xscale = (float *)((char *)tmp + scale_offset);
    const ds4_rocm_runtime_config *cfg = cuda_runtime_config();
    const int use_dp4a = 1;
    quantize_q8_0_f32_kernel<<<(unsigned)blocks, 32>>>(
            xq, xscale, (const float *)x->ptr, in_dim, blocks);
    if (!cuda_ok(cudaGetLastError(), "matmul_q8_0_hc_expand quantize launch")) {
        return 0;
    }
    const uint32_t rows_per_block = cfg->q8_hc_decode_rpb;
    matmul_q8_0_hc_expand_preq_rows_w32_kernel<<<
            ((unsigned)out_dim + rows_per_block - 1u) / rows_per_block,
            rows_per_block * 32u>>>(
            (float *)out_hc->ptr,
            (float *)block_out->ptr,
            block_add ? (const float *)block_add->ptr : (const float *)block_out->ptr,
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
            rows_per_block,
            block_add ? 1 : 0,
            use_dp4a);
    return cuda_ok(cudaGetLastError(), "matmul_q8_0_hc_expand rows launch");
}

extern "C" int ds4_gpu_matmul_f16_tensor(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim, const ds4_gpu_tensor *x, uint64_t n_tok) {
    if (!out || !x || !model_map ||
        in_dim == 0u || out_dim == 0u || n_tok == 0u ||
        in_dim > UINT32_MAX || out_dim > UINT32_MAX || n_tok > UINT32_MAX) return 0;
    uint64_t weight_bytes = 0, x_bytes = 0, out_bytes = 0;
    if (weight_offset > model_size ||
        !cuda_u64_mul3_checked(out_dim, in_dim, sizeof(uint16_t), &weight_bytes) ||
        weight_bytes > model_size - weight_offset ||
        !cuda_u64_mul3_checked(n_tok, in_dim, sizeof(float), &x_bytes) ||
        !cuda_u64_mul3_checked(n_tok, out_dim, sizeof(float), &out_bytes) ||
        x->bytes < x_bytes || out->bytes < out_bytes) return 0;
    const char *wptr = cuda_model_range_ptr(model_map, weight_offset, weight_bytes, "f16");
    if (!wptr) return 0;
    const __half *w = (const __half *)wptr;
    const int ordered_decode = n_tok == 1u;
    if (n_tok > 1u && n_tok <= 8u && in_dim == 16384u && out_dim == 24u) {
        const dim3 grid((uint32_t)out_dim, (uint32_t)n_tok, 1u);
        matmul_f16_tiny_batch_wave_kernel<<<grid, 32u>>>(
                (float *)out->ptr, w, (const float *)x->ptr,
                (uint32_t)in_dim, (uint32_t)out_dim, (uint32_t)n_tok);
        return cuda_ok(cudaGetLastError(), "f16 tiny-batch wave launch");
    }
    if (g_cublas_ready && n_tok > 1) {
        const uint64_t xh_count = n_tok * in_dim;
        __half *xh = (__half *)cuda_tmp_alloc(xh_count * sizeof(__half), "f16 gemm activations");
        if (!xh) return 0;
        f32_to_f16_kernel<<<(xh_count + 255) / 256, 256>>>(xh, (const float *)x->ptr, xh_count);
        if (!cuda_ok(cudaGetLastError(), "f16 activation convert launch")) return 0;
#ifdef __HIP_PLATFORM_AMD__
        if ((n_tok == 2048u || n_tok == 4096u) &&
            hipblaslt_prefill_solution_index((uint32_t)out_dim, (uint32_t)n_tok,
                                            (uint32_t)in_dim) >= 0 &&
            !g_glm_model && !g_quality_mode &&
            g_rocblas_f16_solution_set == DS4_ROCBLAS_F16_SOLUTIONS_5_6_8D1AE90E &&
            ds4_rocm_is_gfx1151()) {
            if (hipblaslt_gemm_tn_f16_out_f32_prefill((float *)out->ptr, w, xh,
                        (uint32_t)out_dim, (uint32_t)n_tok, (uint32_t)in_dim)) return 1;
        }
#endif
        if (in_dim == 16384u && out_dim == 24u && n_tok == 2048u &&
            ds4_rocm_gfx1151_flag("DS4_ROCM_F16_TINYM_WMMA")) {
            matmul_f16_tinym24_wmma_kernel<<<32u, 256u>>>((float *)out->ptr, w, xh);
            return cuda_ok(cudaGetLastError(), "f16 tinym24 wmma launch");
        }
        if (in_dim == 4096u && n_tok == 2048u &&
            ds4_rocm_gfx1151_flag("DS4_ROCM_F16_SMALLM_WMMA")) {
            if (out_dim == 64u || out_dim == 512u || out_dim == 1024u) {
                const dim3 grid((uint32_t)out_dim / 64u, (uint32_t)n_tok / 64u, 1u);
                matmul_f16_smallm_wmma_kernel<64u, 64u><<<grid, 512u>>>(
                        (float *)out->ptr, w, xh,
                        (uint32_t)out_dim, (uint32_t)n_tok, (uint32_t)in_dim);
                return cuda_ok(cudaGetLastError(), "f16 smallm wmma launch");
            }
            if (out_dim == 256u) {
                const dim3 grid((uint32_t)out_dim / 128u, (uint32_t)n_tok / 64u, 1u);
                matmul_f16_smallm_wmma_kernel<128u, 64u><<<grid, 1024u>>>(
                        (float *)out->ptr, w, xh,
                        (uint32_t)out_dim, (uint32_t)n_tok, (uint32_t)in_dim);
                return cuda_ok(cudaGetLastError(), "f16 smallm wmma launch");
            }
        }
        if (n_tok == 2048u && in_dim == 1024u && out_dim == 8192u &&
            ds4_rocm_gfx1151_flag("DS4_ROCM_F16_LARGEM_WMMA")) {
            const dim3 grid((uint32_t)out_dim / 64u, (uint32_t)n_tok / 64u, 1u);
            matmul_f16_smallm_wmma_kernel<64u, 64u><<<grid, 512u>>>(
                    (float *)out->ptr, w, xh, (uint32_t)out_dim,
                    (uint32_t)n_tok, (uint32_t)in_dim);
            return cuda_ok(cudaGetLastError(), "f16 largem wmma launch");
        }
        const float alpha = 1.0f;
        const float beta = 0.0f;
#ifdef __HIP_PLATFORM_AMD__
        if ((n_tok == 4u ||
             (g_dspark_verify_mode && n_tok <= 6u)) &&
            g_rocblas_ready &&
            g_rocblas_f16_solution_set != DS4_ROCBLAS_F16_SOLUTIONS_NONE &&
            !__atomic_load_n(&g_rocblas_f16_solutions_disabled, __ATOMIC_RELAXED) &&
            ds4_rocm_gfx1151_flag("DS4_ROCM_F16_Q4_SOLUTIONS")) {
            int32_t solution = 0;
            if (in_dim == 4096u &&
                (out_dim == 64u || out_dim == 256u ||
                 out_dim == 512u || out_dim == 1024u)) {
                if (g_rocblas_f16_solution_set ==
                    DS4_ROCBLAS_F16_SOLUTIONS_5_5_CD957402) solution = -217;
                if (g_rocblas_f16_solution_set ==
                    DS4_ROCBLAS_F16_SOLUTIONS_5_6_8D1AE90E) solution = -50;
            } else if (in_dim == 1024u && out_dim == 8192u) {
                if (g_rocblas_f16_solution_set ==
                    DS4_ROCBLAS_F16_SOLUTIONS_5_5_CD957402) solution = -216;
                if (g_rocblas_f16_solution_set ==
                    DS4_ROCBLAS_F16_SOLUTIONS_5_6_8D1AE90E) solution = -49;
            }
            if (solution != 0) {
                const rocblas_status rst = rocblas_gemm_ex(
                        g_rocblas,
                        rocblas_operation_transpose,
                        rocblas_operation_none,
                        (rocblas_int)out_dim,
                        (rocblas_int)n_tok,
                        (rocblas_int)in_dim,
                        &alpha,
                        w,
                        rocblas_datatype_f16_r,
                        (rocblas_int)in_dim,
                        xh,
                        rocblas_datatype_f16_r,
                        (rocblas_int)in_dim,
                        &beta,
                        out->ptr,
                        rocblas_datatype_f32_r,
                        (rocblas_int)out_dim,
                        out->ptr,
                        rocblas_datatype_f32_r,
                        (rocblas_int)out_dim,
                        rocblas_datatype_f32_r,
                        rocblas_gemm_algo_solution_index,
                        solution,
                        0u);
                if (rst == rocblas_status_success) return 1;
                __atomic_store_n(&g_rocblas_f16_solutions_disabled, 1, __ATOMIC_RELAXED);
            }
        }
#endif
        cublasStatus_t st = cublasGemmEx(g_cublas,
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
                                         CUBLAS_COMPUTE_32F,
                                         CUBLAS_GEMM_DEFAULT);
        return cublas_ok(st, "f16 matmul");
    }
    /* The 4096x256 F16 router projection is latency-bound and the ordered
     * 32-thread row kernel is at least as fast on gfx1151; keep shared-X for
     * compressor/indexer F16 decode where reusing x across rows is the win. */
    const bool f16_decode_router_shape = (in_dim == 4096u && out_dim == 256u);
    if (n_tok == 1u && !g_quality_mode && !cuda_runtime_config()->graph_dump &&
        !f16_decode_router_shape) {
        if (in_dim <= 8192u && in_dim * sizeof(float) <= 65536u) {
            const uint32_t rows_per_block = 32u;
            matmul_f16_f32_sharedx_warp_rows_w32_kernel<<<
                    ((unsigned)out_dim + rows_per_block - 1u) / rows_per_block,
                    rows_per_block * 32u,
                    (size_t)in_dim * sizeof(float)>>>(
                    (float *)out->ptr, w, (const float *)x->ptr, (uint32_t)in_dim, out_dim);
            return cuda_ok(cudaGetLastError(), "matmul_f16 sharedx launch");
        }
    }
    dim3 grid((unsigned)out_dim, (unsigned)n_tok, 1);
    if (ordered_decode) {
        matmul_f16_ordered_chunks_kernel<<<grid, 32>>>((float *)out->ptr, w, (const float *)x->ptr, in_dim, out_dim, n_tok);
        return cuda_ok(cudaGetLastError(), "matmul_f16_ordered_chunks launch");
    }
    matmul_f16_kernel<<<grid, 256>>>((float *)out->ptr, w, (const float *)x->ptr, in_dim, out_dim, n_tok);
    return cuda_ok(cudaGetLastError(), "matmul_f16 launch");
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
    if (!out0 || !out1 || !x || !model_map || in_dim == 0 || out_dim == 0 || n_tok == 0 ||
        in_dim > UINT32_MAX || out_dim > UINT32_MAX || n_tok > UINT32_MAX) {
        return 0;
    }
    if (n_tok != 1) {
        return ds4_gpu_matmul_f16_tensor(out0, model_map, model_size, weight0_offset,
                                           in_dim, out_dim, x, n_tok) &&
               ds4_gpu_matmul_f16_tensor(out1, model_map, model_size, weight1_offset,
                                           in_dim, out_dim, x, n_tok);
    }
    uint64_t weight_bytes = 0;
    if (weight0_offset > model_size || weight1_offset > model_size ||
        !cuda_u64_mul3_checked(out_dim, in_dim, sizeof(uint16_t), &weight_bytes)) {
        return 0;
    }
    if (weight_bytes > model_size - weight0_offset ||
        weight_bytes > model_size - weight1_offset ||
        x->bytes < in_dim * sizeof(float) ||
        out0->bytes < out_dim * sizeof(float) ||
        out1->bytes < out_dim * sizeof(float)) {
        return 0;
    }
    const __half *w0 = (const __half *)cuda_model_range_ptr(model_map, weight0_offset, weight_bytes, "f16_pair0");
    const __half *w1 = (const __half *)cuda_model_range_ptr(model_map, weight1_offset, weight_bytes, "f16_pair1");
    if (!w0 || !w1) return 0;
    if (!g_quality_mode && !cuda_runtime_config()->graph_dump) {
        if (in_dim <= 8192u && in_dim * sizeof(float) <= 65536u) {
            const uint32_t rows_per_block = 32u;
            matmul_f16_pair_f32_sharedx_warp_rows_w32_kernel<<<
                    ((unsigned)out_dim + rows_per_block - 1u) / rows_per_block,
                    rows_per_block * 32u,
                    (size_t)in_dim * sizeof(float)>>>(
                    (float *)out0->ptr, (float *)out1->ptr, w0, w1,
                    (const float *)x->ptr, (uint32_t)in_dim, out_dim);
            return cuda_ok(cudaGetLastError(), "matmul_f16_pair sharedx launch");
        }
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
    if (!out || !x || !model_map || in_dim == 0 || out_dim == 0 || n_tok == 0 ||
        in_dim > UINT32_MAX || out_dim > UINT32_MAX || n_tok > UINT32_MAX) return 0;
    uint64_t weight_bytes = 0, x_bytes = 0, out_bytes = 0;
    if (weight_offset > model_size ||
        !cuda_u64_mul3_checked(out_dim, in_dim, sizeof(float), &weight_bytes) ||
        weight_bytes > model_size - weight_offset ||
        !cuda_u64_mul3_checked(n_tok, in_dim, sizeof(float), &x_bytes) ||
        !cuda_u64_mul3_checked(n_tok, out_dim, sizeof(float), &out_bytes) ||
        x->bytes < x_bytes || out->bytes < out_bytes) return 0;
    const char *wptr = cuda_model_range_ptr(model_map, weight_offset, weight_bytes, "f32");
    if (!wptr) return 0;
    const float *w = (const float *)wptr;
    if (g_cublas_ready && n_tok > 1) {
        const float alpha = 1.0f;
        const float beta = 0.0f;
        cublasStatus_t st = cublasSgemm(g_cublas,
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
