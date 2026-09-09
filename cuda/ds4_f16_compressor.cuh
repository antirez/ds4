#ifndef DS4_CUDA_F16_COMPRESSOR_CUH
#define DS4_CUDA_F16_COMPRESSOR_CUH

/* Restore the automatic GB10 compressor kernels from ff749b84. Both fused
 * paths retain the canonical contiguous per-lane K chunks and ascending lane
 * fold; the transposed half2 layout coalesces weights without reassociating
 * arithmetic. Only the output-owning lane updates the selected state row. */

__global__ static void matmul_f16_pair_compressor_store_ordered_chunks_kernel(
        float *out_kv,
        float *out_score,
        float *state_kv,
        float *state_score,
        const __half *w_kv,
        const __half *w_score,
        const float *x,
        const void *ape,
        uint32_t ape_type,
        uint64_t in_dim,
        uint32_t width,
        uint32_t ratio,
        uint32_t pos) {
    const uint32_t row = blockIdx.x;
    if (row >= width) return;

    __shared__ float partial_kv[32];
    __shared__ float partial_score[32];
    const uint32_t tid = threadIdx.x;
    float sum_kv = 0.0f;
    float sum_score = 0.0f;
    const uint64_t chunk = (in_dim + 31u) / 32u;
    const uint64_t k0 = (uint64_t)tid * chunk;
    uint64_t k1 = k0 + chunk;
    if (k1 > in_dim) k1 = in_dim;
    const __half *wr_kv = w_kv + (uint64_t)row * in_dim;
    const __half *wr_score = w_score + (uint64_t)row * in_dim;
    for (uint64_t i = k0; i < k1; i++) {
        const float xv = x[i];
        sum_kv += __half2float(wr_kv[i]) * xv;
        sum_score += __half2float(wr_score[i]) * xv;
    }
    partial_kv[tid] = sum_kv;
    partial_score[tid] = sum_score;
    __syncthreads();
    if (tid == 0u) {
        float total_kv = 0.0f;
        float total_score = 0.0f;
        for (uint32_t i = 0; i < 32u; i++) {
            total_kv += partial_kv[i];
            total_score += partial_score[i];
        }
        out_kv[row] = total_kv;
        out_score[row] = total_score;
        const uint32_t pos_mod = pos % ratio;
        const uint32_t dst_row = ratio == 4u ? ratio + pos_mod : pos_mod;
        const uint64_t ape_index = (uint64_t)pos_mod * width + row;
        const float ape_value = ape_type == 1u
            ? __half2float(((const __half *)ape)[ape_index])
            : ((const float *)ape)[ape_index];
        state_kv[(uint64_t)dst_row * width + row] = total_kv;
        state_score[(uint64_t)dst_row * width + row] =
            total_score + ape_value;
    }
}

__global__ static void f16_pair_chunk32_repack_kernel(
        __half2 *dst,
        const __half *w0,
        const __half *w1,
        uint64_t in_dim,
        uint32_t width) {
    const uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t count = (uint64_t)width * in_dim;
    if (gid >= count) return;
    const uint64_t row = gid / in_dim;
    const uint64_t k = gid - row * in_dim;
    const uint64_t chunk = in_dim / 32u;
    const uint64_t lane = k / chunk;
    const uint64_t iter = k - lane * chunk;
    const uint64_t dst_idx = (row * chunk + iter) * 32u + lane;
    dst[dst_idx] = __halves2half2(w0[gid], w1[gid]);
}

__global__ static void matmul_f16_pair_compressor_store_chunk32_prefetch8_kernel(
        float *out_kv,
        float *out_score,
        float *state_kv,
        float *state_score,
        const __half2 *w_pair,
        const float *x,
        const void *ape,
        uint32_t ape_type,
        uint64_t in_dim,
        uint32_t width,
        uint32_t ratio,
        uint32_t pos) {
    const uint32_t row = blockIdx.x;
    if (row >= width) return;

    __shared__ float partial_kv[32];
    __shared__ float partial_score[32];
    const uint32_t tid = threadIdx.x;
    float sum_kv = 0.0f;
    float sum_score = 0.0f;
    const uint64_t chunk = in_dim / 32u;
    const uint64_t k0 = (uint64_t)tid * chunk;
    const __half2 *wr = w_pair + (uint64_t)row * in_dim;
    for (uint64_t j = 0; j < chunk; j += 8u) {
        const float xv0 = x[k0 + j + 0u];
        const float xv1 = x[k0 + j + 1u];
        const float xv2 = x[k0 + j + 2u];
        const float xv3 = x[k0 + j + 3u];
        const float xv4 = x[k0 + j + 4u];
        const float xv5 = x[k0 + j + 5u];
        const float xv6 = x[k0 + j + 6u];
        const float xv7 = x[k0 + j + 7u];
        const __half2 wp0 = wr[(j + 0u) * 32u + tid];
        const __half2 wp1 = wr[(j + 1u) * 32u + tid];
        const __half2 wp2 = wr[(j + 2u) * 32u + tid];
        const __half2 wp3 = wr[(j + 3u) * 32u + tid];
        const __half2 wp4 = wr[(j + 4u) * 32u + tid];
        const __half2 wp5 = wr[(j + 5u) * 32u + tid];
        const __half2 wp6 = wr[(j + 6u) * 32u + tid];
        const __half2 wp7 = wr[(j + 7u) * 32u + tid];
        sum_kv += __half2float(__low2half(wp0)) * xv0;
        sum_score += __half2float(__high2half(wp0)) * xv0;
        sum_kv += __half2float(__low2half(wp1)) * xv1;
        sum_score += __half2float(__high2half(wp1)) * xv1;
        sum_kv += __half2float(__low2half(wp2)) * xv2;
        sum_score += __half2float(__high2half(wp2)) * xv2;
        sum_kv += __half2float(__low2half(wp3)) * xv3;
        sum_score += __half2float(__high2half(wp3)) * xv3;
        sum_kv += __half2float(__low2half(wp4)) * xv4;
        sum_score += __half2float(__high2half(wp4)) * xv4;
        sum_kv += __half2float(__low2half(wp5)) * xv5;
        sum_score += __half2float(__high2half(wp5)) * xv5;
        sum_kv += __half2float(__low2half(wp6)) * xv6;
        sum_score += __half2float(__high2half(wp6)) * xv6;
        sum_kv += __half2float(__low2half(wp7)) * xv7;
        sum_score += __half2float(__high2half(wp7)) * xv7;
    }
    partial_kv[tid] = sum_kv;
    partial_score[tid] = sum_score;
    __syncthreads();
    if (tid == 0u) {
        float total_kv = 0.0f;
        float total_score = 0.0f;
        for (uint32_t i = 0; i < 32u; i++) {
            total_kv += partial_kv[i];
            total_score += partial_score[i];
        }
        out_kv[row] = total_kv;
        out_score[row] = total_score;
        const uint32_t pos_mod = pos % ratio;
        const uint32_t dst_row = ratio == 4u ? ratio + pos_mod : pos_mod;
        const uint64_t ape_index = (uint64_t)pos_mod * width + row;
        const float ape_value = ape_type == 1u
            ? __half2float(((const __half *)ape)[ape_index])
            : ((const float *)ape)[ape_index];
        state_kv[(uint64_t)dst_row * width + row] = total_kv;
        state_score[(uint64_t)dst_row * width + row] = total_score + ape_value;
    }
}

#endif
