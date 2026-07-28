__global__ static void embed_token_hc_kernel(float *out, const unsigned short *w, uint32_t token, uint32_t n_embd, uint32_t n_hc) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t n = n_embd * n_hc;
    if (i >= n) return;
    uint32_t e = i % n_embd;
    out[i] = __half2float(reinterpret_cast<const __half *>(w)[(uint64_t)token * n_embd + e]);
}

__global__ static void embed_tokens_hc_kernel(
        float *out,
        const int32_t *tokens,
        const __half *w,
        uint32_t n_vocab,
        uint32_t n_tokens,
        uint32_t n_embd,
        uint32_t n_hc) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_tokens * n_hc * n_embd;
    if (gid >= n) return;
    uint32_t d = gid % n_embd;
    uint64_t tmp = gid / n_embd;
    uint32_t t = tmp / n_hc;
    int32_t tok_i = tokens[t];
    uint32_t tok = tok_i < 0 ? 0u : (uint32_t)tok_i;
    if (tok >= n_vocab) tok = 0;
    out[gid] = __half2float(w[(uint64_t)tok * n_embd + d]);
}

__global__ static void matmul_f16_kernel(
        float *out,
        const __half *w,
        const float *x,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t n_tok) {
    uint64_t row = (uint64_t)blockIdx.x;
    uint64_t tok = (uint64_t)blockIdx.y;
    if (row >= out_dim || tok >= n_tok) return;

    float sum = 0.0f;
    const __half *wr = w + row * in_dim;
    const float *xr = x + tok * in_dim;
    for (uint64_t i = threadIdx.x; i < in_dim; i += blockDim.x) {
        sum += __half2float(wr[i]) * xr[i];
    }

    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[tok * out_dim + row] = partial[0];
}

__global__ static void matmul_f16_serial_kernel(
        float *out,
        const __half *w,
        const float *x,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t n_tok) {
    uint64_t row = (uint64_t)blockIdx.x;
    uint64_t tok = (uint64_t)blockIdx.y;
    if (row >= out_dim || tok >= n_tok || threadIdx.x != 0) return;

    float sum = 0.0f;
    const __half *wr = w + row * in_dim;
    const float *xr = x + tok * in_dim;
    for (uint64_t i = 0; i < in_dim; i++) {
        sum += __half2float(wr[i]) * xr[i];
    }
    out[tok * out_dim + row] = sum;
}

__global__ static void matmul_f16_ordered_chunks_kernel(
        float *out,
        const __half *w,
        const float *x,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t n_tok) {
    uint64_t row = (uint64_t)blockIdx.x;
    uint64_t tok = (uint64_t)blockIdx.y;
    if (row >= out_dim || tok >= n_tok) return;

    __shared__ float partial[32];
    const uint32_t tid = threadIdx.x;
    float sum = 0.0f;
    const uint64_t chunk = (in_dim + 31u) / 32u;
    const uint64_t k0 = (uint64_t)tid * chunk;
    uint64_t k1 = k0 + chunk;
    if (k1 > in_dim) k1 = in_dim;
    const __half *wr = w + row * in_dim;
    const float *xr = x + tok * in_dim;
    for (uint64_t i = k0; i < k1; i++) {
        sum += __half2float(wr[i]) * xr[i];
    }
    partial[tid] = sum;
    __syncthreads();
    if (tid == 0) {
        float total = 0.0f;
        for (uint32_t i = 0; i < 32u; i++) total += partial[i];
        out[tok * out_dim + row] = total;
    }
}

__global__ static void matmul_f16_small_out_hx_ordered_chunks_kernel(
        float *out,
        const __half *w,
        const float *x,
        uint64_t in_dim,
        uint64_t out_dim) {
    uint64_t row = (uint64_t)blockIdx.x;
    if (row >= out_dim) return;

    __shared__ float partial[32];
    const uint32_t tid = threadIdx.x;
    float sum = 0.0f;
    const uint64_t chunk = (in_dim + 31u) / 32u;
    const uint64_t k0 = (uint64_t)tid * chunk;
    uint64_t k1 = k0 + chunk;
    if (k1 > in_dim) k1 = in_dim;
    const __half *wr = w + row * in_dim;
    for (uint64_t i = k0; i < k1; i++) {
        const float xv = __half2float(__float2half(x[i]));
        sum += __half2float(wr[i]) * xv;
    }
    partial[tid] = sum;
    __syncthreads();
    if (tid == 0) {
        float total = 0.0f;
        for (uint32_t i = 0; i < 32u; i++) total += partial[i];
        out[row] = total;
    }
}

__global__ static void matmul_f16_small_out_batch_kernel(
        float *out,
        const __half *w,
        const float *x,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t n_tok) {
    const uint64_t tok = (uint64_t)blockIdx.x;
    const uint32_t tid = threadIdx.x;
    if (tok >= n_tok || out_dim > 32u || blockDim.x != 256u) return;

    float acc[32];
    #pragma unroll
    for (uint32_t r = 0; r < 32u; r++) acc[r] = 0.0f;

    const float *xr = x + tok * in_dim;
    for (uint64_t i = tid; i < in_dim; i += 256u) {
        const float xv = xr[i];
        #pragma unroll
        for (uint32_t r = 0; r < 32u; r++) {
            if (r < out_dim) {
                acc[r] += __half2float(w[(uint64_t)r * in_dim + i]) * xv;
            }
        }
    }

    __shared__ float partial[32 * 256];
    #pragma unroll
    for (uint32_t r = 0; r < 32u; r++) {
        if (r < out_dim) partial[r * 256u + tid] = acc[r];
    }
    __syncthreads();

    for (uint32_t stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            #pragma unroll
            for (uint32_t r = 0; r < 32u; r++) {
                if (r < out_dim) {
                    partial[r * 256u + tid] += partial[r * 256u + tid + stride];
                }
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        #pragma unroll
        for (uint32_t r = 0; r < 32u; r++) {
            if (r < out_dim) out[tok * out_dim + r] = partial[r * 256u];
        }
    }
}

__global__ static void matmul_f16_pair_ordered_chunks_kernel(
        float *out0,
        float *out1,
        const __half *w0,
        const __half *w1,
        const float *x,
        uint64_t in_dim,
        uint64_t out0_dim,
        uint64_t out1_dim) {
    uint64_t row = (uint64_t)blockIdx.x;
    if (row >= out0_dim && row >= out1_dim) return;

    __shared__ float partial0[32];
    __shared__ float partial1[32];
    const uint32_t tid = threadIdx.x;
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    const uint64_t chunk = (in_dim + 31u) / 32u;
    const uint64_t k0 = (uint64_t)tid * chunk;
    uint64_t k1 = k0 + chunk;
    if (k1 > in_dim) k1 = in_dim;
    const __half *wr0 = row < out0_dim ? w0 + row * in_dim : w0;
    const __half *wr1 = row < out1_dim ? w1 + row * in_dim : w1;
    for (uint64_t i = k0; i < k1; i++) {
        const float xv = x[i];
        if (row < out0_dim) sum0 += __half2float(wr0[i]) * xv;
        if (row < out1_dim) sum1 += __half2float(wr1[i]) * xv;
    }
    partial0[tid] = sum0;
    partial1[tid] = sum1;
    __syncthreads();
    if (tid == 0) {
        float total0 = 0.0f;
        float total1 = 0.0f;
        for (uint32_t i = 0; i < 32u; i++) {
            total0 += partial0[i];
            total1 += partial1[i];
        }
        if (row < out0_dim) out0[row] = total0;
        if (row < out1_dim) out1[row] = total1;
    }
}

__global__ static void matmul_f32_kernel(
        float *out,
        const float *w,
        const float *x,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t n_tok) {
    uint64_t row = (uint64_t)blockIdx.x;
    uint64_t tok = (uint64_t)blockIdx.y;
    if (row >= out_dim || tok >= n_tok) return;

    float sum = 0.0f;
    const float *wr = w + row * in_dim;
    const float *xr = x + tok * in_dim;
    for (uint64_t i = threadIdx.x; i < in_dim; i += blockDim.x) {
        sum += wr[i] * xr[i];
    }

    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[tok * out_dim + row] = partial[0];
}

__global__ static void repeat_hc_kernel(float *out, const float *row, uint32_t n_embd, uint32_t n_hc) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_embd * n_hc;
    if (i >= n) return;
    out[i] = row[i % n_embd];
}

__global__ static void repeat_hc_rows_kernel(float *out, const float *rows, uint32_t n_tokens, uint32_t n_embd, uint32_t n_hc) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_tokens * n_hc * n_embd;
    if (i >= n) return;

    uint64_t hc_row = (uint64_t)n_hc * n_embd;
    uint64_t tok = i / hc_row;
    uint64_t embd = i % n_embd;
    out[i] = rows[tok * n_embd + embd];
}

__global__ static void pack_slot_rows_f32_kernel(float *out, const float *slots, uint32_t n_rows, uint32_t width, uint32_t n_slots, uint32_t slot_cap) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_rows * n_slots * width;
    if (i >= n) return;

    uint64_t col = i % width;
    uint64_t slot = (i / width) % n_slots;
    uint64_t row = i / ((uint64_t)n_slots * width);
    out[i] = slots[((slot * slot_cap) + row) * width + col];
}

__global__ static void f32_to_f16_kernel(__half *out, const float *x, uint64_t n) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half(x[i]);
}

__device__ static float warp_sum_f32(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, offset);
    }
    return v;
}

__device__ static float warp_max_f32(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, offset));
    }
    return v;
}

__device__ static float dot4_f32(float4 a, float4 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

__device__ __forceinline__ static int32_t load_i8x4_i32_aligned(const int8_t *p) {
    return *(const int32_t *)p;
}

__device__ __forceinline__ static int32_t load_i8x4_i32_unaligned(const int8_t *p) {
    const uint8_t *u = (const uint8_t *)p;
    return (int32_t)((uint32_t)u[0] |
                     ((uint32_t)u[1] << 8) |
                     ((uint32_t)u[2] << 16) |
                     ((uint32_t)u[3] << 24));
}

__device__ __forceinline__ static int32_t dot_i8x32_dp4a(const int8_t *a, const int8_t *b) {
    int32_t dot = 0;
#pragma unroll
    for (uint32_t i = 0; i < 32u; i += 4u) {
        dot = __dp4a(load_i8x4_i32_unaligned(a + i), load_i8x4_i32_aligned(b + i), dot);
    }
    return dot;
}

__device__ __forceinline__ static int32_t dot_i8_block(const int8_t *a, const int8_t *b, uint64_t n, int use_dp4a) {
    if (use_dp4a && n == 32u) return dot_i8x32_dp4a(a, b);
    int32_t dot = 0;
    for (uint64_t i = 0; i < n; i++) dot += (int32_t)a[i] * (int32_t)b[i];
    return dot;
}

__global__ static DS4_CUDA_UNUSED void matmul_q8_0_kernel(
        float *out,
        const unsigned char *w,
        const float *x,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t n_tok) {
    uint64_t row = (uint64_t)blockIdx.x;
    uint64_t tok = (uint64_t)blockIdx.y;
    if (row >= out_dim || tok >= n_tok) return;
    const uint64_t blocks = (in_dim + 31) / 32;
    const unsigned char *wr = w + row * blocks * 34;
    const float *xr = x + tok * in_dim;
    float acc = 0.0f;

    for (uint64_t b = threadIdx.x; b < blocks; b += blockDim.x) {
        uint64_t i0 = b * 32;
        uint64_t bn = in_dim - i0 < 32 ? in_dim - i0 : 32;
        float amax = 0.0f;
        for (uint64_t i = 0; i < bn; i++) amax = fmaxf(amax, fabsf(xr[i0 + i]));
        float d = amax / 127.0f;
        float id = d != 0.0f ? 1.0f / d : 0.0f;
        const __half *scale_h = (const __half *)(wr + b * 34);
        const int8_t *qs = (const int8_t *)(wr + b * 34 + 2);
        int dot = 0;
        for (uint64_t i = 0; i < bn; i++) {
            int q = (int)lrintf(xr[i0 + i] * id);
            q = q > 127 ? 127 : (q < -128 ? -128 : q);
            dot += (int)qs[i] * q;
        }
        acc += __half2float(*scale_h) * d * (float)dot;
    }

    __shared__ float partial[256];
    partial[threadIdx.x] = acc;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[tok * out_dim + row] = partial[0];
}

__global__ static void quantize_q8_0_f32_kernel(
        int8_t *xq,
        float *xscale,
        const float *x,
        uint64_t in_dim,
        uint64_t blocks) {
    uint64_t b = blockIdx.x;
    uint64_t tok = blockIdx.y;
    if (b >= blocks) return;
    uint64_t i0 = b * 32;
    uint64_t bn = in_dim - i0 < 32 ? in_dim - i0 : 32;
    const float *xr = x + tok * in_dim + i0;

    float a = 0.0f;
    if (threadIdx.x < bn) a = fabsf(xr[threadIdx.x]);
    __shared__ float vals[32];
    vals[threadIdx.x] = a;
    __syncthreads();
    for (uint32_t stride = 16; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) vals[threadIdx.x] = fmaxf(vals[threadIdx.x], vals[threadIdx.x + stride]);
        __syncthreads();
    }
    const float d = vals[0] / 127.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    if (threadIdx.x == 0) xscale[tok * blocks + b] = d;
    int8_t *dst = xq + (tok * blocks + b) * 32;
    if (threadIdx.x < bn) {
        int v = (int)lrintf(xr[threadIdx.x] * id);
        v = v > 127 ? 127 : (v < -128 ? -128 : v);
        dst[threadIdx.x] = (int8_t)v;
    } else {
        dst[threadIdx.x] = 0;
    }
}

__global__ static void quantize_q8_0_group_slice_rows_kernel(
        int8_t *xq,
        float *xscale,
        const float *x,
        uint64_t group_dim,
        uint64_t blocks,
        uint32_t n_groups_total,
        uint32_t group0,
        uint32_t group_cnt) {
    const uint64_t b = blockIdx.x;
    const uint64_t packed_row = blockIdx.y;
    if (b >= blocks) return;
    const uint64_t token = packed_row / group_cnt;
    const uint64_t group = group0 + packed_row - token * group_cnt;
    const uint64_t i0 = b * 32u;
    const uint64_t bn = group_dim - i0 < 32u ? group_dim - i0 : 32u;
    const float *xr = x +
        (token * n_groups_total + group) * group_dim + i0;

    float a = 0.0f;
    if (threadIdx.x < bn) a = fabsf(xr[threadIdx.x]);
    __shared__ float vals[32];
    vals[threadIdx.x] = a;
    __syncthreads();
    for (uint32_t stride = 16u; stride > 0u; stride >>= 1u) {
        if (threadIdx.x < stride) {
            vals[threadIdx.x] =
                fmaxf(vals[threadIdx.x], vals[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    const float d = vals[0] / 127.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    if (threadIdx.x == 0u) xscale[packed_row * blocks + b] = d;
    int8_t *dst = xq + (packed_row * blocks + b) * 32u;
    if (threadIdx.x < bn) {
        int v = (int)lrintf(xr[threadIdx.x] * id);
        v = v > 127 ? 127 : (v < -128 ? -128 : v);
        dst[threadIdx.x] = (int8_t)v;
    } else {
        dst[threadIdx.x] = 0;
    }
}

__global__ static void matmul_q8_0_preq_kernel(
        float *out,
        const unsigned char *w,
        const int8_t *xq,
        const float *xscale,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t n_tok,
        uint64_t blocks,
        int use_dp4a) {
    uint64_t row = (uint64_t)blockIdx.x;
    uint64_t tok = (uint64_t)blockIdx.y;
    if (row >= out_dim || tok >= n_tok) return;
    const unsigned char *wr = w + row * blocks * 34;
    const int8_t *xqr = xq + tok * blocks * 32;
    const float *xsr = xscale + tok * blocks;
    float acc = 0.0f;
    for (uint64_t b = threadIdx.x; b < blocks; b += blockDim.x) {
        uint64_t i0 = b * 32;
        uint64_t bn = in_dim - i0 < 32 ? in_dim - i0 : 32;
        const __half *scale_h = (const __half *)(wr + b * 34);
        const int8_t *qs = (const int8_t *)(wr + b * 34 + 2);
        const int8_t *xqb = xqr + b * 32;
        int dot = dot_i8_block(qs, xqb, bn, use_dp4a);
        acc += __half2float(*scale_h) * xsr[b] * (float)dot;
    }
    __shared__ float partial[256];
    partial[threadIdx.x] = acc;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[tok * out_dim + row] = partial[0];
}

__global__ static void matmul_q8_0_preq_warp8_kernel(
        float *out,
        const unsigned char *w,
        const int8_t *xq,
        const float *xscale,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t blocks,
        int use_dp4a) {
    uint64_t row = (uint64_t)blockIdx.x * 8u + (threadIdx.x >> 5u);
    const uint64_t tok = (uint64_t)blockIdx.y;
    uint32_t lane = threadIdx.x & 31u;
    if (row >= out_dim) return;
    const unsigned char *wr = w + row * blocks * 34;
    const int8_t *xqr = xq + tok * blocks * 32u;
    const float *xsr = xscale + tok * blocks;
    float acc = 0.0f;
    for (uint64_t b = lane; b < blocks; b += 32u) {
        uint64_t i0 = b * 32;
        uint64_t bn = in_dim - i0 < 32 ? in_dim - i0 : 32;
        const __half *scale_h = (const __half *)(wr + b * 34);
        const int8_t *qs = (const int8_t *)(wr + b * 34 + 2);
        const int8_t *xqb = xqr + b * 32;
        int dot = dot_i8_block(qs, xqb, bn, use_dp4a);
        acc += __half2float(*scale_h) * xsr[b] * (float)dot;
    }
    acc = warp_sum_f32(acc);
    if (lane == 0) out[tok * out_dim + row] = acc;
}

__device__ __forceinline__ static uint32_t q8_top1_float_ordered_key(float v) {
    const uint32_t u = __float_as_uint(v);
    return (u & 0x80000000u) ? ~u : (u ^ 0x80000000u);
}

__device__ __forceinline__ static uint64_t q8_top1_pack_key(float v, uint32_t idx) {
    return ((uint64_t)q8_top1_float_ordered_key(v) << 32u) |
           (uint64_t)(0xffffffffu - idx);
}

__device__ __forceinline__ static float q8_top1_unpack_value(uint32_t ordered) {
    const uint32_t u = (ordered & 0x80000000u)
        ? (ordered ^ 0x80000000u)
        : ~ordered;
    return __uint_as_float(u);
}

__global__ static void matmul_q8_0_top1_preq_warp8_kernel(
        unsigned long long *best_key,
        const unsigned char *w,
        const int8_t *xq,
        const float *xscale,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t blocks,
        uint32_t index_offset,
        int use_dp4a) {
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    const uint64_t row = (uint64_t)blockIdx.x * 8u + warp;
    const bool valid = row < out_dim;
    float acc = 0.0f;

    if (valid) {
        const unsigned char *wr = w + row * blocks * 34;
        for (uint64_t b = lane; b < blocks; b += 32u) {
            uint64_t i0 = b * 32;
            uint64_t bn = in_dim - i0 < 32 ? in_dim - i0 : 32;
            const __half *scale_h = (const __half *)(wr + b * 34);
            const int8_t *qs = (const int8_t *)(wr + b * 34 + 2);
            const int8_t *xqb = xq + b * 32;
            int dot = dot_i8_block(qs, xqb, bn, use_dp4a);
            acc += __half2float(*scale_h) * xscale[b] * (float)dot;
        }
    }
    acc = warp_sum_f32(acc);

    __shared__ unsigned long long keys[8];
    if (lane == 0u) {
        keys[warp] = valid
            ? (unsigned long long)q8_top1_pack_key(acc, index_offset + (uint32_t)row)
            : 0ull;
    }
    __syncthreads();

    if (threadIdx.x == 0u) {
        unsigned long long block_best = keys[0];
        #pragma unroll
        for (uint32_t i = 1u; i < 8u; i++) {
            if (keys[i] > block_best) block_best = keys[i];
        }
        (void)atomicMax(best_key, block_best);
    }
}

__global__ static void matmul_q8_0_top1_unpack_kernel(
        uint32_t *selected,
        float *values,
        const unsigned long long *best_key) {
    if (threadIdx.x != 0u || blockIdx.x != 0u) return;
    const uint64_t key = (uint64_t)best_key[0];
    const uint32_t ordered = (uint32_t)(key >> 32u);
    const uint32_t idx = 0xffffffffu - (uint32_t)key;
    selected[0] = idx;
    values[0] = q8_top1_unpack_value(ordered);
}

__global__ static void matmul_q8_0_kslice_preq_warp8_kernel(
        float *out,
        const unsigned char *w,
        const int8_t *xq,
        const float *xscale,
        uint64_t slice_dim,
        uint64_t out_dim,
        uint64_t full_blocks,
        uint64_t block_start,
        uint64_t slice_blocks,
        int use_dp4a) {
    const uint64_t row =
        (uint64_t)blockIdx.x * 8u + (threadIdx.x >> 5u);
    const uint64_t tok = blockIdx.y;
    const uint32_t lane = threadIdx.x & 31u;
    if (row >= out_dim) return;
    out += tok * out_dim;
    xq += tok * slice_blocks * 32u;
    xscale += tok * slice_blocks;
    const unsigned char *wr =
        w + row * full_blocks * 34u + block_start * 34u;
    float acc = 0.0f;
    for (uint64_t b = lane; b < slice_blocks; b += 32u) {
        uint64_t i0 = b * 32u;
        uint64_t bn = slice_dim - i0 < 32u ? slice_dim - i0 : 32u;
        const __half *scale_h = (const __half *)(wr + b * 34u);
        const int8_t *qs = (const int8_t *)(wr + b * 34u + 2u);
        const int8_t *xqb = xq + b * 32u;
        int dot = dot_i8_block(qs, xqb, bn, use_dp4a);
        acc += __half2float(*scale_h) * xscale[b] * (float)dot;
    }
    acc = warp_sum_f32(acc);
    if (lane == 0) out[row] = acc;
}

__global__ static void matmul_q8_0_pair_preq_warp8_kernel(
        float *out0,
        float *out1,
        const unsigned char *w0,
        const unsigned char *w1,
        const int8_t *xq,
        const float *xscale,
        uint64_t in_dim,
        uint64_t out0_dim,
        uint64_t out1_dim,
        uint64_t blocks,
        int use_dp4a) {
    uint64_t row = (uint64_t)blockIdx.x * 8u + (threadIdx.x >> 5u);
    const uint64_t tok = (uint64_t)blockIdx.y;
    uint32_t lane = threadIdx.x & 31u;
    if (row >= out0_dim && row >= out1_dim) return;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    const unsigned char *wr0 = row < out0_dim ? w0 + row * blocks * 34 : NULL;
    const unsigned char *wr1 = row < out1_dim ? w1 + row * blocks * 34 : NULL;
    const int8_t *xqr = xq + tok * blocks * 32u;
    const float *xsr = xscale + tok * blocks;
    for (uint64_t b = lane; b < blocks; b += 32u) {
        uint64_t i0 = b * 32;
        uint64_t bn = in_dim - i0 < 32 ? in_dim - i0 : 32;
        const int8_t *xqb = xqr + b * 32;
        const float xs = xsr[b];
        if (wr0) {
            const __half *scale_h = (const __half *)(wr0 + b * 34);
            const int8_t *qs = (const int8_t *)(wr0 + b * 34 + 2);
            int dot = dot_i8_block(qs, xqb, bn, use_dp4a);
            acc0 += __half2float(*scale_h) * xs * (float)dot;
        }
        if (wr1) {
            const __half *scale_h = (const __half *)(wr1 + b * 34);
            const int8_t *qs = (const int8_t *)(wr1 + b * 34 + 2);
            int dot = dot_i8_block(qs, xqb, bn, use_dp4a);
            acc1 += __half2float(*scale_h) * xs * (float)dot;
        }
    }
    acc0 = warp_sum_f32(acc0);
    acc1 = warp_sum_f32(acc1);
    if (lane == 0) {
        if (row < out0_dim) out0[tok * out0_dim + row] = acc0;
        if (row < out1_dim) out1[tok * out1_dim + row] = acc1;
    }
}

__global__ static void shared_mid_q8_0_preq_warp8_exact_kernel(
        float *mid,
        const unsigned char *gate_w,
        const unsigned char *up_w,
        const int8_t *xq,
        const float *xscale,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t blocks,
        float clamp,
        const int32_t *selected,
        uint32_t expert_split,
        bool home_rank,
        int use_dp4a) {
    const uint64_t row = (uint64_t)blockIdx.x * 8u + (threadIdx.x >> 5u);
    const uint32_t lane = threadIdx.x & 31u;
    if (row >= out_dim) return;
    if (selected) {
        /* Complementary predicates select exactly one writer; ties stay on
         * the home rank to avoid an unnecessary peer store. */
        uint32_t home_count = 0u;
        uint32_t peer_count = 0u;
        #pragma unroll
        for (uint32_t i = 0; i < 6u; i++) {
            const int32_t expert = selected[i];
            if (expert >= 0 && (uint32_t)expert < expert_split) {
                home_count++;
            } else if (expert >= 0 &&
                       (uint32_t)expert < 2u * expert_split) {
                peer_count++;
            }
        }
        const bool assigned = home_rank
            ? home_count <= peer_count : peer_count < home_count;
        if (!assigned) return;
    }
    const unsigned char *gate_row = gate_w + row * blocks * 34u;
    const unsigned char *up_row = up_w + row * blocks * 34u;
    float gate = 0.0f;
    float up = 0.0f;
    for (uint64_t b = lane; b < blocks; b += 32u) {
        const uint64_t i0 = b * 32u;
        const uint64_t bn = in_dim - i0 < 32u ? in_dim - i0 : 32u;
        const int8_t *xqb = xq + b * 32u;
        const float xs = xscale[b];
        const unsigned char *gb = gate_row + b * 34u;
        const unsigned char *ub = up_row + b * 34u;
        gate += __half2float(*(const __half *)gb) * xs *
                (float)dot_i8_block((const int8_t *)(gb + 2u), xqb, bn,
                                    use_dp4a);
        up += __half2float(*(const __half *)ub) * xs *
              (float)dot_i8_block((const int8_t *)(ub + 2u), xqb, bn,
                                  use_dp4a);
    }
    gate = warp_sum_f32(gate);
    up = warp_sum_f32(up);
    if (lane == 0u) {
        if (clamp > 1.0e-6f) {
            gate = fminf(gate, clamp);
            up = fminf(fmaxf(up, -clamp), clamp);
        }
        const float silu = gate / (1.0f + expf(-gate));
        mid[row] = silu * up * 1.0f;
    }
}

__global__ static void matmul_q8_0_pair_preq_batch_kernel(
        float *out0,
        float *out1,
        const unsigned char *w0,
        const unsigned char *w1,
        const int8_t *xq,
        const float *xscale,
        uint64_t in_dim,
        uint64_t out0_dim,
        uint64_t out1_dim,
        uint64_t n_tok,
        uint64_t blocks,
        int use_dp4a) {
    const uint64_t row = (uint64_t)blockIdx.x;
    const uint64_t tok = (uint64_t)blockIdx.y;
    if (tok >= n_tok) return;
    const int has0 = row < out0_dim;
    const int has1 = row < out1_dim;
    if (!has0 && !has1) return;

    const unsigned char *wr0 = has0 ? w0 + row * blocks * 34u : NULL;
    const unsigned char *wr1 = has1 ? w1 + row * blocks * 34u : NULL;
    const int8_t *xqr = xq + tok * blocks * 32u;
    const float *xsr = xscale + tok * blocks;
    float acc0 = 0.0f;
    float acc1 = 0.0f;

    for (uint64_t b = threadIdx.x; b < blocks; b += blockDim.x) {
        const uint64_t i0 = b * 32u;
        const uint64_t bn = in_dim - i0 < 32u ? in_dim - i0 : 32u;
        const int8_t *xqb = xqr + b * 32u;
        const float xs = xsr[b];
        if (has0) {
            const __half *scale_h = (const __half *)(wr0 + b * 34u);
            const int8_t *qs = (const int8_t *)(wr0 + b * 34u + 2u);
            const int dot = dot_i8_block(qs, xqb, bn, use_dp4a);
            acc0 += __half2float(*scale_h) * xs * (float)dot;
        }
        if (has1) {
            const __half *scale_h = (const __half *)(wr1 + b * 34u);
            const int8_t *qs = (const int8_t *)(wr1 + b * 34u + 2u);
            const int dot = dot_i8_block(qs, xqb, bn, use_dp4a);
            acc1 += __half2float(*scale_h) * xs * (float)dot;
        }
    }

    __shared__ float partial0[256];
    __shared__ float partial1[256];
    partial0[threadIdx.x] = acc0;
    partial1[threadIdx.x] = acc1;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1u; stride > 0u; stride >>= 1u) {
        if (threadIdx.x < stride) {
            partial0[threadIdx.x] += partial0[threadIdx.x + stride];
            partial1[threadIdx.x] += partial1[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        if (has0) out0[tok * out0_dim + row] = partial0[0];
        if (has1) out1[tok * out1_dim + row] = partial1[0];
    }
}

__global__ static void matmul_q8_0_pair_preq_batch_tok2_exact_kernel(
        float *out0,
        float *out1,
        const unsigned char *w0,
        const unsigned char *w1,
        const int8_t *xq,
        const float *xscale,
        uint64_t in_dim,
        uint64_t out0_dim,
        uint64_t out1_dim,
        uint64_t n_tok,
        uint64_t blocks,
        int use_dp4a) {
    const uint64_t row = (uint64_t)blockIdx.x;
    const uint64_t tok0 = (uint64_t)blockIdx.y * 2u;
    if (tok0 >= n_tok) return;
    const int has0 = row < out0_dim;
    const int has1 = row < out1_dim;
    if (!has0 && !has1) return;
    const int valid1 = tok0 + 1u < n_tok;

    const unsigned char *wr0 = has0 ? w0 + row * blocks * 34u : NULL;
    const unsigned char *wr1 = has1 ? w1 + row * blocks * 34u : NULL;
    const int8_t *xqr0 = xq + tok0 * blocks * 32u;
    const int8_t *xqr1 = valid1 ? xqr0 + blocks * 32u : xqr0;
    const float *xsr0 = xscale + tok0 * blocks;
    const float *xsr1 = valid1 ? xsr0 + blocks : xsr0;
    float acc00 = 0.0f;
    float acc01 = 0.0f;
    float acc10 = 0.0f;
    float acc11 = 0.0f;

    for (uint64_t b = threadIdx.x; b < blocks; b += blockDim.x) {
        const uint64_t i0 = b * 32u;
        const uint64_t bn = in_dim - i0 < 32u ? in_dim - i0 : 32u;
        const int8_t *xqb0 = xqr0 + b * 32u;
        const int8_t *xqb1 = xqr1 + b * 32u;
        const float xs0 = xsr0[b];
        const float xs1 = valid1 ? xsr1[b] : 0.0f;
        if (has0) {
            const __half *scale_h = (const __half *)(wr0 + b * 34u);
            const int8_t *qs = (const int8_t *)(wr0 + b * 34u + 2u);
            const int dot0 = dot_i8_block(qs, xqb0, bn, use_dp4a);
            int dot1 = 0;
            if (valid1) dot1 = dot_i8_block(qs, xqb1, bn, use_dp4a);
            const float ws = __half2float(*scale_h);
            acc00 += ws * xs0 * (float)dot0;
            if (valid1) acc01 += ws * xs1 * (float)dot1;
        }
        if (has1) {
            const __half *scale_h = (const __half *)(wr1 + b * 34u);
            const int8_t *qs = (const int8_t *)(wr1 + b * 34u + 2u);
            const int dot0 = dot_i8_block(qs, xqb0, bn, use_dp4a);
            int dot1 = 0;
            if (valid1) dot1 = dot_i8_block(qs, xqb1, bn, use_dp4a);
            const float ws = __half2float(*scale_h);
            acc10 += ws * xs0 * (float)dot0;
            if (valid1) acc11 += ws * xs1 * (float)dot1;
        }
    }

    __shared__ float partial00[256];
    __shared__ float partial01[256];
    __shared__ float partial10[256];
    __shared__ float partial11[256];
    partial00[threadIdx.x] = acc00;
    partial01[threadIdx.x] = acc01;
    partial10[threadIdx.x] = acc10;
    partial11[threadIdx.x] = acc11;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1u; stride > 0u; stride >>= 1u) {
        if (threadIdx.x < stride) {
            partial00[threadIdx.x] += partial00[threadIdx.x + stride];
            partial01[threadIdx.x] += partial01[threadIdx.x + stride];
            partial10[threadIdx.x] += partial10[threadIdx.x + stride];
            partial11[threadIdx.x] += partial11[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        if (has0) {
            out0[tok0 * out0_dim + row] = partial00[0];
            if (valid1) out0[(tok0 + 1u) * out0_dim + row] = partial01[0];
        }
        if (has1) {
            out1[tok0 * out1_dim + row] = partial10[0];
            if (valid1) out1[(tok0 + 1u) * out1_dim + row] = partial11[0];
        }
    }
}

__device__ static float moe_owned_packed_combine_row(
        const float *home_slots,
        const float *peer_packed,
        const int32_t *selected,
        uint32_t row,
        uint32_t out_dim,
        uint32_t expert_split);

__global__ static void matmul_q8_0_hc_expand_preq_warp8_kernel(
        float *out_hc,
        float *block_out,
        const float *block_add,
        const float *block_add2,
        const float *owned_home_slots,
        const float *owned_peer_packed,
        const int32_t *owned_selected,
        const float *residual_hc,
        const float *split,
        const unsigned char *w,
        const int8_t *xq,
        const float *xscale,
        uint64_t in_dim,
        uint64_t out_dim,
        uint32_t n_embd,
        uint32_t n_hc,
        uint64_t blocks,
        int has_add,
        int has_add2,
        int has_owned_slots,
        uint32_t owned_expert_split,
        int use_dp4a) {
    const uint64_t row = (uint64_t)blockIdx.x * 8u + (threadIdx.x >> 5u);
    const uint32_t lane = threadIdx.x & 31u;
    if (row >= out_dim) return;
    const unsigned char *wr = w + row * blocks * 34;
    float acc = 0.0f;
    for (uint64_t b = lane; b < blocks; b += 32u) {
        const uint64_t i0 = b * 32;
        const uint64_t bn = in_dim - i0 < 32 ? in_dim - i0 : 32;
        const __half *scale_h = (const __half *)(wr + b * 34);
        const int8_t *qs = (const int8_t *)(wr + b * 34 + 2);
        const int8_t *xqb = xq + b * 32;
        int dot = dot_i8_block(qs, xqb, bn, use_dp4a);
        acc += __half2float(*scale_h) * xscale[b] * (float)dot;
    }
    acc = warp_sum_f32(acc);
    if (lane == 0) {
        const uint32_t d = (uint32_t)row;
        block_out[d] = acc;
        float block_v = acc;
        if (has_owned_slots) {
            const float routed = moe_owned_packed_combine_row(
                    owned_home_slots,
                    owned_peer_packed,
                    owned_selected,
                    d,
                    (uint32_t)out_dim,
                    owned_expert_split);
            block_v = __fadd_rn(block_v, routed);
        } else if (has_add) {
            float add_v = block_add[d];
            if (has_add2) add_v += block_add2[d];
            block_v += add_v;
        }
        const float *post = split + n_hc;
        const float *comb = split + 2u * n_hc;
        for (uint32_t dst_hc = 0; dst_hc < n_hc; dst_hc++) {
            float hc_acc = block_v * post[dst_hc];
            for (uint32_t src_hc = 0; src_hc < n_hc; src_hc++) {
                const float comb_v = comb[dst_hc + (uint64_t)src_hc * n_hc];
                const float res_v = residual_hc[(uint64_t)src_hc * n_embd + d];
                hc_acc += comb_v * res_v;
            }
            out_hc[(uint64_t)dst_hc * n_embd + d] = hc_acc;
        }
    }
}

__global__ static void matmul_q8_0_kslice_hc_expand_add_preq_warp8_kernel(
        float *out_hc,
        float *block_out,
        const float *block_add,
        const float *residual_hc,
        const float *split,
        const unsigned char *w,
        const int8_t *xq,
        const float *xscale,
        uint64_t slice_dim,
        uint64_t out_dim,
        uint64_t full_blocks,
        uint64_t block_start,
        uint64_t slice_blocks,
        uint32_t n_embd,
        uint32_t n_hc,
        int use_dp4a) {
    const uint64_t row = (uint64_t)blockIdx.x * 8u + (threadIdx.x >> 5u);
    const uint32_t lane = threadIdx.x & 31u;
    if (row >= out_dim) return;
    const unsigned char *wr = w + row * full_blocks * 34u + block_start * 34u;
    float acc = 0.0f;
    for (uint64_t b = lane; b < slice_blocks; b += 32u) {
        const uint64_t i0 = b * 32u;
        const uint64_t bn = slice_dim - i0 < 32u ? slice_dim - i0 : 32u;
        const __half *scale_h = (const __half *)(wr + b * 34u);
        const int8_t *qs = (const int8_t *)(wr + b * 34u + 2u);
        const int8_t *xqb = xq + b * 32u;
        const int dot = dot_i8_block(qs, xqb, bn, use_dp4a);
        acc += __half2float(*scale_h) * xscale[b] * (float)dot;
    }
    acc = warp_sum_f32(acc);
    if (lane == 0) {
        const uint32_t d = (uint32_t)row;
        block_out[d] = acc;
        const float block_v = acc + block_add[d];
        const float *post = split + n_hc;
        const float *comb = split + 2u * n_hc;
        for (uint32_t dst_hc = 0; dst_hc < n_hc; dst_hc++) {
            float hc_acc = block_v * post[dst_hc];
            for (uint32_t src_hc = 0; src_hc < n_hc; src_hc++) {
                const float comb_v = comb[dst_hc + (uint64_t)src_hc * n_hc];
                const float res_v = residual_hc[(uint64_t)src_hc * n_embd + d];
                hc_acc += comb_v * res_v;
            }
            out_hc[(uint64_t)dst_hc * n_embd + d] = hc_acc;
        }
    }
}

__global__ static void matmul_q8_0_preq_batch_warp8_kernel(
        float *out,
        const unsigned char *w,
        const int8_t *xq,
        const float *xscale,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t n_tok,
        uint64_t blocks,
        int use_dp4a) {
    const uint64_t row = (uint64_t)blockIdx.x * 8u + (threadIdx.x >> 5u);
    const uint64_t tok = (uint64_t)blockIdx.y;
    const uint32_t lane = threadIdx.x & 31u;
    if (row >= out_dim || tok >= n_tok) return;

    const unsigned char *wr = w + row * blocks * 34;
    const int8_t *xqr = xq + tok * blocks * 32;
    const float *xsr = xscale + tok * blocks;
    float acc = 0.0f;
    for (uint64_t b = lane; b < blocks; b += 32u) {
        const uint64_t i0 = b * 32;
        const uint64_t bn = in_dim - i0 < 32 ? in_dim - i0 : 32;
        const __half *scale_h = (const __half *)(wr + b * 34);
        const int8_t *qs = (const int8_t *)(wr + b * 34 + 2);
        const int8_t *xqb = xqr + b * 32;
        int dot = dot_i8_block(qs, xqb, bn, use_dp4a);
        acc += __half2float(*scale_h) * xsr[b] * (float)dot;
    }
    acc = warp_sum_f32(acc);
    if (lane == 0) out[tok * out_dim + row] = acc;
}

__global__ static void matmul_q8_0_preq_batch_warp8_tok2_kernel(
        float *out,
        const unsigned char *w,
        const int8_t *xq,
        const float *xscale,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t blocks,
        int use_dp4a) {
    const uint64_t row = (uint64_t)blockIdx.x * 8u + (threadIdx.x >> 5u);
    const uint32_t lane = threadIdx.x & 31u;
    if (row >= out_dim) return;

    const unsigned char *wr = w + row * blocks * 34u;
    const int8_t *xqr0 = xq;
    const int8_t *xqr1 = xq + blocks * 32u;
    const float *xsr0 = xscale;
    const float *xsr1 = xscale + blocks;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    for (uint64_t b = lane; b < blocks; b += 32u) {
        const uint64_t i0 = b * 32u;
        const uint64_t bn = in_dim - i0 < 32u ? in_dim - i0 : 32u;
        const __half *scale_h = (const __half *)(wr + b * 34u);
        const int8_t *qs = (const int8_t *)(wr + b * 34u + 2u);
        const int8_t *xqb0 = xqr0 + b * 32u;
        const int8_t *xqb1 = xqr1 + b * 32u;
        int dot0 = 0;
        int dot1 = 0;
        if (use_dp4a && bn == 32u) {
#pragma unroll
            for (uint32_t i = 0; i < 32u; i += 4u) {
                const int32_t w4 = load_i8x4_i32_unaligned(qs + i);
                dot0 = __dp4a(w4, load_i8x4_i32_aligned(xqb0 + i), dot0);
                dot1 = __dp4a(w4, load_i8x4_i32_aligned(xqb1 + i), dot1);
            }
        } else {
            dot0 = dot_i8_block(qs, xqb0, bn, use_dp4a);
            dot1 = dot_i8_block(qs, xqb1, bn, use_dp4a);
        }
        const float ws = __half2float(*scale_h);
        acc0 += ws * xsr0[b] * (float)dot0;
        acc1 += ws * xsr1[b] * (float)dot1;
    }
    acc0 = warp_sum_f32(acc0);
    acc1 = warp_sum_f32(acc1);
    if (lane == 0) {
        out[row] = acc0;
        out[out_dim + row] = acc1;
    }
}

__global__ static void matmul_q8_0_preq_batch_warp8_tok4_kernel(
        float *out,
        const unsigned char *w,
        const int8_t *xq,
        const float *xscale,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t n_tok,
        uint64_t blocks,
        int use_dp4a) {
    const uint64_t row = (uint64_t)blockIdx.x * 8u + (threadIdx.x >> 5u);
    const uint64_t tok0 = (uint64_t)blockIdx.y * 4u;
    const uint32_t lane = threadIdx.x & 31u;
    if (row >= out_dim || tok0 >= n_tok) return;

    const unsigned char *wr = w + row * blocks * 34;
    const int8_t *xqr0 = xq + tok0 * blocks * 32;
    const int8_t *xqr1 = xqr0 + blocks * 32;
    const int8_t *xqr2 = xqr1 + blocks * 32;
    const int8_t *xqr3 = xqr2 + blocks * 32;
    const float *xsr0 = xscale + tok0 * blocks;
    const float *xsr1 = xsr0 + blocks;
    const float *xsr2 = xsr1 + blocks;
    const float *xsr3 = xsr2 + blocks;
    const int valid1 = tok0 + 1u < n_tok;
    const int valid2 = tok0 + 2u < n_tok;
    const int valid3 = tok0 + 3u < n_tok;

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    for (uint64_t b = lane; b < blocks; b += 32u) {
        const uint64_t i0 = b * 32;
        const uint64_t bn = in_dim - i0 < 32 ? in_dim - i0 : 32;
        const __half *scale_h = (const __half *)(wr + b * 34);
        const int8_t *qs = (const int8_t *)(wr + b * 34 + 2);
        const int8_t *xqb0 = xqr0 + b * 32;
        const int8_t *xqb1 = xqr1 + b * 32;
        const int8_t *xqb2 = xqr2 + b * 32;
        const int8_t *xqb3 = xqr3 + b * 32;
        int dot0 = 0;
        int dot1 = 0;
        int dot2 = 0;
        int dot3 = 0;
        if (use_dp4a && bn == 32u) {
#pragma unroll
            for (uint32_t i = 0; i < 32u; i += 4u) {
                const int32_t w4 = load_i8x4_i32_unaligned(qs + i);
                dot0 = __dp4a(w4, load_i8x4_i32_aligned(xqb0 + i), dot0);
                if (valid1) dot1 = __dp4a(w4, load_i8x4_i32_aligned(xqb1 + i), dot1);
                if (valid2) dot2 = __dp4a(w4, load_i8x4_i32_aligned(xqb2 + i), dot2);
                if (valid3) dot3 = __dp4a(w4, load_i8x4_i32_aligned(xqb3 + i), dot3);
            }
        } else {
            dot0 = dot_i8_block(qs, xqb0, bn, use_dp4a);
            if (valid1) dot1 = dot_i8_block(qs, xqb1, bn, use_dp4a);
            if (valid2) dot2 = dot_i8_block(qs, xqb2, bn, use_dp4a);
            if (valid3) dot3 = dot_i8_block(qs, xqb3, bn, use_dp4a);
        }
        const float ws = __half2float(*scale_h);
        acc0 += ws * xsr0[b] * (float)dot0;
        if (valid1) acc1 += ws * xsr1[b] * (float)dot1;
        if (valid2) acc2 += ws * xsr2[b] * (float)dot2;
        if (valid3) acc3 += ws * xsr3[b] * (float)dot3;
    }
    acc0 = warp_sum_f32(acc0);
    acc1 = warp_sum_f32(acc1);
    acc2 = warp_sum_f32(acc2);
    acc3 = warp_sum_f32(acc3);
    if (lane == 0) {
        out[tok0 * out_dim + row] = acc0;
        if (valid1) out[(tok0 + 1u) * out_dim + row] = acc1;
        if (valid2) out[(tok0 + 2u) * out_dim + row] = acc2;
        if (valid3) out[(tok0 + 3u) * out_dim + row] = acc3;
    }
}

__global__ static void matmul_q8_0_preq_batch_warp8_tok8_kernel(
        float *out,
        const unsigned char *w,
        const int8_t *xq,
        const float *xscale,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t n_tok,
        uint64_t blocks,
        int use_dp4a) {
    const uint64_t row = (uint64_t)blockIdx.x * 8u + (threadIdx.x >> 5u);
    const uint64_t tok0 = (uint64_t)blockIdx.y * 8u;
    const uint32_t lane = threadIdx.x & 31u;
    if (row >= out_dim || tok0 >= n_tok) return;

    const unsigned char *wr = w + row * blocks * 34;
    const uint64_t xq_stride = blocks * 32u;
    const int8_t *xqr0 = xq + tok0 * xq_stride;
    const int valid1 = tok0 + 1u < n_tok;
    const int valid2 = tok0 + 2u < n_tok;
    const int valid3 = tok0 + 3u < n_tok;
    const int valid4 = tok0 + 4u < n_tok;
    const int valid5 = tok0 + 5u < n_tok;
    const int valid6 = tok0 + 6u < n_tok;
    const int valid7 = tok0 + 7u < n_tok;
    const int8_t *xqr1 = valid1 ? xqr0 + xq_stride : xqr0;
    const int8_t *xqr2 = valid2 ? xqr1 + xq_stride : xqr0;
    const int8_t *xqr3 = valid3 ? xqr2 + xq_stride : xqr0;
    const int8_t *xqr4 = valid4 ? xqr3 + xq_stride : xqr0;
    const int8_t *xqr5 = valid5 ? xqr4 + xq_stride : xqr0;
    const int8_t *xqr6 = valid6 ? xqr5 + xq_stride : xqr0;
    const int8_t *xqr7 = valid7 ? xqr6 + xq_stride : xqr0;
    const float *xsr0 = xscale + tok0 * blocks;
    const float *xsr1 = valid1 ? xsr0 + blocks : xsr0;
    const float *xsr2 = valid2 ? xsr1 + blocks : xsr0;
    const float *xsr3 = valid3 ? xsr2 + blocks : xsr0;
    const float *xsr4 = valid4 ? xsr3 + blocks : xsr0;
    const float *xsr5 = valid5 ? xsr4 + blocks : xsr0;
    const float *xsr6 = valid6 ? xsr5 + blocks : xsr0;
    const float *xsr7 = valid7 ? xsr6 + blocks : xsr0;

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    float acc4 = 0.0f;
    float acc5 = 0.0f;
    float acc6 = 0.0f;
    float acc7 = 0.0f;
    for (uint64_t b = lane; b < blocks; b += 32u) {
        const uint64_t i0 = b * 32;
        const uint64_t bn = in_dim - i0 < 32 ? in_dim - i0 : 32;
        const __half *scale_h = (const __half *)(wr + b * 34);
        const int8_t *qs = (const int8_t *)(wr + b * 34 + 2);
        const int8_t *xqb0 = xqr0 + b * 32;
        const int8_t *xqb1 = xqr1 + b * 32;
        const int8_t *xqb2 = xqr2 + b * 32;
        const int8_t *xqb3 = xqr3 + b * 32;
        const int8_t *xqb4 = xqr4 + b * 32;
        const int8_t *xqb5 = xqr5 + b * 32;
        const int8_t *xqb6 = xqr6 + b * 32;
        const int8_t *xqb7 = xqr7 + b * 32;
        int dot0 = 0;
        int dot1 = 0;
        int dot2 = 0;
        int dot3 = 0;
        int dot4 = 0;
        int dot5 = 0;
        int dot6 = 0;
        int dot7 = 0;
        if (use_dp4a && bn == 32u) {
#pragma unroll
            for (uint32_t i = 0; i < 32u; i += 4u) {
                const int32_t w4 = load_i8x4_i32_unaligned(qs + i);
                dot0 = __dp4a(w4, load_i8x4_i32_aligned(xqb0 + i), dot0);
                if (valid1) dot1 = __dp4a(w4, load_i8x4_i32_aligned(xqb1 + i), dot1);
                if (valid2) dot2 = __dp4a(w4, load_i8x4_i32_aligned(xqb2 + i), dot2);
                if (valid3) dot3 = __dp4a(w4, load_i8x4_i32_aligned(xqb3 + i), dot3);
                if (valid4) dot4 = __dp4a(w4, load_i8x4_i32_aligned(xqb4 + i), dot4);
                if (valid5) dot5 = __dp4a(w4, load_i8x4_i32_aligned(xqb5 + i), dot5);
                if (valid6) dot6 = __dp4a(w4, load_i8x4_i32_aligned(xqb6 + i), dot6);
                if (valid7) dot7 = __dp4a(w4, load_i8x4_i32_aligned(xqb7 + i), dot7);
            }
        } else {
            dot0 = dot_i8_block(qs, xqb0, bn, use_dp4a);
            if (valid1) dot1 = dot_i8_block(qs, xqb1, bn, use_dp4a);
            if (valid2) dot2 = dot_i8_block(qs, xqb2, bn, use_dp4a);
            if (valid3) dot3 = dot_i8_block(qs, xqb3, bn, use_dp4a);
            if (valid4) dot4 = dot_i8_block(qs, xqb4, bn, use_dp4a);
            if (valid5) dot5 = dot_i8_block(qs, xqb5, bn, use_dp4a);
            if (valid6) dot6 = dot_i8_block(qs, xqb6, bn, use_dp4a);
            if (valid7) dot7 = dot_i8_block(qs, xqb7, bn, use_dp4a);
        }
        const float ws = __half2float(*scale_h);
        acc0 += ws * xsr0[b] * (float)dot0;
        if (valid1) acc1 += ws * xsr1[b] * (float)dot1;
        if (valid2) acc2 += ws * xsr2[b] * (float)dot2;
        if (valid3) acc3 += ws * xsr3[b] * (float)dot3;
        if (valid4) acc4 += ws * xsr4[b] * (float)dot4;
        if (valid5) acc5 += ws * xsr5[b] * (float)dot5;
        if (valid6) acc6 += ws * xsr6[b] * (float)dot6;
        if (valid7) acc7 += ws * xsr7[b] * (float)dot7;
    }
    acc0 = warp_sum_f32(acc0);
    acc1 = warp_sum_f32(acc1);
    acc2 = warp_sum_f32(acc2);
    acc3 = warp_sum_f32(acc3);
    acc4 = warp_sum_f32(acc4);
    acc5 = warp_sum_f32(acc5);
    acc6 = warp_sum_f32(acc6);
    acc7 = warp_sum_f32(acc7);
    if (lane == 0) {
        out[tok0 * out_dim + row] = acc0;
        if (valid1) out[(tok0 + 1u) * out_dim + row] = acc1;
        if (valid2) out[(tok0 + 2u) * out_dim + row] = acc2;
        if (valid3) out[(tok0 + 3u) * out_dim + row] = acc3;
        if (valid4) out[(tok0 + 4u) * out_dim + row] = acc4;
        if (valid5) out[(tok0 + 5u) * out_dim + row] = acc5;
        if (valid6) out[(tok0 + 6u) * out_dim + row] = acc6;
        if (valid7) out[(tok0 + 7u) * out_dim + row] = acc7;
    }
}

__global__ static void matmul_q8_0_preq_batch_tok2_exact_kernel(
        float *out,
        const unsigned char *w,
        const int8_t *xq,
        const float *xscale,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t n_tok,
        uint64_t blocks,
        int use_dp4a) {
    const uint64_t row = (uint64_t)blockIdx.x;
    const uint64_t tok0 = (uint64_t)blockIdx.y * 2u;
    if (row >= out_dim || tok0 >= n_tok) return;
    const int valid1 = tok0 + 1u < n_tok;
    const unsigned char *wr = w + row * blocks * 34u;
    const int8_t *xqr0 = xq + tok0 * blocks * 32u;
    const int8_t *xqr1 = valid1 ? xqr0 + blocks * 32u : xqr0;
    const float *xsr0 = xscale + tok0 * blocks;
    const float *xsr1 = valid1 ? xsr0 + blocks : xsr0;
    float acc0 = 0.0f;
    float acc1 = 0.0f;

    for (uint64_t b = threadIdx.x; b < blocks; b += blockDim.x) {
        const uint64_t i0 = b * 32u;
        const uint64_t bn = in_dim - i0 < 32u ? in_dim - i0 : 32u;
        const __half *scale_h = (const __half *)(wr + b * 34u);
        const int8_t *qs = (const int8_t *)(wr + b * 34u + 2u);
        const int8_t *xqb0 = xqr0 + b * 32u;
        const int8_t *xqb1 = xqr1 + b * 32u;
        const int dot0 = dot_i8_block(qs, xqb0, bn, use_dp4a);
        int dot1 = 0;
        if (valid1) dot1 = dot_i8_block(qs, xqb1, bn, use_dp4a);
        const float ws = __half2float(*scale_h);
        acc0 += ws * xsr0[b] * (float)dot0;
        if (valid1) acc1 += ws * xsr1[b] * (float)dot1;
    }

    __shared__ float partial0[256];
    __shared__ float partial1[256];
    partial0[threadIdx.x] = acc0;
    partial1[threadIdx.x] = acc1;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1u; stride > 0u; stride >>= 1u) {
        if (threadIdx.x < stride) {
            partial0[threadIdx.x] += partial0[threadIdx.x + stride];
            partial1[threadIdx.x] += partial1[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        out[tok0 * out_dim + row] = partial0[0];
        if (valid1) out[(tok0 + 1u) * out_dim + row] = partial1[0];
    }
}


/* ---- INT8 tensor-core exact Q8_0 batch matmul --------------------------
 * Bit-identical replacement for the exact tok2/warp8-family batched Q8_0
 * kernels. Each output element's reduction is the reference's strided
 * halving tree over T slots (T = reduction width: 32 for the warp kernels,
 * cuda_q8_exact_threads(blocks) for the exact kernels; slots >= blocks hold
 * +0.0f). The kernel decomposes that tree as: 32 streams at stride T/32
 * whose 32 terms per outer step j combine via an adjacent-pairwise static
 * register stack taken in bit-reversed stream order (== the top five strided
 * tree levels), plus per-(j&3) sequential accumulators and a fixed tail for
 * the remaining levels. Fuzz-verified bitwise against both reference
 * kernels across shapes, including blocks < T and ragged out_dim/n_tok.
 * Rollback: DS4_CUDA_NO_Q8_MMA=1. */
__device__ __forceinline__ static uint32_t ldu32_unaligned(const uint8_t *p) {
    const uintptr_t addr = (uintptr_t)p;
    const uint32_t *base = (const uint32_t *)(addr & ~(uintptr_t)3);
    const uint32_t lo = base[0];
    const uint32_t hi = base[1];
    return __funnelshift_r(lo, hi, (uint32_t)(addr & 3u) * 8u);
}

__device__ __forceinline__ static void mma_m16n8k32_s8(
        int32_t &c0, int32_t &c1, int32_t &c2, int32_t &c3,
        uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
        uint32_t b0, uint32_t b1) {
#if __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+r"(c0),"+r"(c1),"+r"(c2),"+r"(c3)
                 : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
#else
    (void)a0;(void)a1;(void)a2;(void)a3;(void)b0;(void)b1;(void)c0;(void)c1;(void)c2;(void)c3;
#endif
}

__device__ __forceinline__ static uint32_t bitrev5(uint32_t i) {
    return ((i & 1u) << 4) | ((i & 2u) << 2) | (i & 4u) | ((i & 8u) >> 2) | ((i & 16u) >> 4);
}

template <uint32_t T>
__global__ static void matmul_q8_0_mma_exact_kernel(
        float *out,
        const unsigned char *w,
        const int8_t *xq,
        const float *xscale,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t n_tok,
        uint64_t blocks,
        uint64_t a_stride_blocks, /* activation row stride in blocks (>= blocks) */
        uint64_t out_stride) {    /* output token stride in floats (>= out_dim) */
    extern __shared__ unsigned char q8mma_sh[];
    __half *sh_ws = (__half *)q8mma_sh;                    /* 64 rows x blocks */
    float *sh_xs = (float *)(q8mma_sh + 64u * blocks * 2u); /* 16 toks x blocks */
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint64_t row_base = (uint64_t)blockIdx.x * 64u;
    const uint64_t tok_base = (uint64_t)blockIdx.y * 16u;

    /* stage weight scales (64 rows) and activation scales (16 tokens) */
    for (uint32_t idx = threadIdx.x; idx < 64u * (uint32_t)blocks; idx += blockDim.x) {
        const uint32_t rl = idx / (uint32_t)blocks;
        const uint32_t b = idx - rl * (uint32_t)blocks;
        uint64_t row = row_base + rl;
        if (row >= out_dim) row = out_dim - 1u;
        sh_ws[idx] = *(const __half *)(w + row * blocks * 34u + (uint64_t)b * 34u);
    }
    for (uint32_t idx = threadIdx.x; idx < 16u * (uint32_t)blocks; idx += blockDim.x) {
        const uint32_t tl = idx / (uint32_t)blocks;
        const uint32_t b = idx - tl * (uint32_t)blocks;
        const uint64_t tok = tok_base + tl;
        sh_xs[idx] = tok < n_tok ? xscale[tok * a_stride_blocks + b] : 0.0f;
    }
    __syncthreads();

    const uint64_t row0 = row_base + (uint64_t)warp * 8u;
    /* thread's C elements: rows n0,n0+1; tokens mt0, mt0+8 */
    const uint32_t n0 = (lane & 3u) * 2u;
    const uint32_t mt0 = lane >> 2u;
    const uint64_t tokA = tok_base + mt0;
    const uint64_t tokB = tok_base + mt0 + 8u;
    /* A source rows for loads (fragment layout): rows lane>>2 and (lane>>2)+8 */
    const uint64_t a_tok_lo = tok_base + (lane >> 2u);
    const uint64_t a_tok_hi = a_tok_lo + 8u;
    const int8_t *aq_lo = xq + (a_tok_lo < n_tok ? a_tok_lo : 0u) * a_stride_blocks * 32u;
    const int8_t *aq_hi = xq + (a_tok_hi < n_tok ? a_tok_hi : 0u) * a_stride_blocks * 32u;
    const bool a_lo_ok = a_tok_lo < n_tok;
    const bool a_hi_ok = a_tok_hi < n_tok;
    /* B source row for loads: row lane>>2 within the warp tile */
    uint64_t b_row = row0 + (lane >> 2u);
    if (b_row >= out_dim) b_row = out_dim - 1u;
    const unsigned char *b_wr = w + b_row * blocks * 34u;

    /* per-element (4) x per-(j&3) accumulators */
    float acc00 = 0.0f, acc01 = 0.0f, acc02 = 0.0f, acc03 = 0.0f;
    float acc10 = 0.0f, acc11 = 0.0f, acc12 = 0.0f, acc13 = 0.0f;
    float acc20 = 0.0f, acc21 = 0.0f, acc22 = 0.0f, acc23 = 0.0f;
    float acc30 = 0.0f, acc31 = 0.0f, acc32 = 0.0f, acc33 = 0.0f;

    const uint32_t stride = T / 32u;
    const uint32_t rl_ws0 = warp * 8u + n0;      /* local row for ws of element cols */
    const uint32_t tl_xsA = mt0;                 /* local token rows for xs */
    const uint32_t tl_xsB = mt0 + 8u;

    for (uint32_t j = 0; j < stride; j++) {
        /* adjacent-pairwise static stack over 32 terms in bitrev5 m order */
        float s0e0 = 0, s1e0 = 0, s2e0 = 0, s3e0 = 0, s4e0 = 0;
        float s0e1 = 0, s1e1 = 0, s2e1 = 0, s3e1 = 0, s4e1 = 0;
        float s0e2 = 0, s1e2 = 0, s2e2 = 0, s3e2 = 0, s4e2 = 0;
        float s0e3 = 0, s1e3 = 0, s2e3 = 0, s3e3 = 0, s4e3 = 0;
#pragma unroll
        for (uint32_t i = 0; i < 32u; i++) {
            const uint32_t m = bitrev5(i);
            const uint32_t s = j + m * stride; /* slot index */
            float t0 = 0.0f, t1 = 0.0f, t2 = 0.0f, t3 = 0.0f;
            /* slot s sums blocks {s + k*T} sequentially (multi-term when
             * blocks > T, exactly like the per-lane strided walk). */
            for (uint32_t b = s; b < blocks; b += T) {
                const uint32_t koff = (lane & 3u) * 4u;
                const int8_t *ablk_lo = aq_lo + b * 32u;
                const int8_t *ablk_hi = aq_hi + b * 32u;
                const uint32_t a0 = a_lo_ok ? *(const uint32_t *)(ablk_lo + koff) : 0u;
                const uint32_t a1 = a_hi_ok ? *(const uint32_t *)(ablk_hi + koff) : 0u;
                const uint32_t a2 = a_lo_ok ? *(const uint32_t *)(ablk_lo + 16u + koff) : 0u;
                const uint32_t a3 = a_hi_ok ? *(const uint32_t *)(ablk_hi + 16u + koff) : 0u;
                const uint8_t *bq = (const uint8_t *)(b_wr + (uint64_t)b * 34u + 2u);
                const uint32_t b0 = ldu32_unaligned(bq + koff);
                const uint32_t b1 = ldu32_unaligned(bq + 16u + koff);
                int32_t c0 = 0, c1 = 0, c2 = 0, c3 = 0;
                mma_m16n8k32_s8(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1);
                /* term = ws * xs * dot, same expression as reference */
                const float ws0 = __half2float(sh_ws[rl_ws0 * (uint32_t)blocks + b]);
                const float ws1 = __half2float(sh_ws[(rl_ws0 + 1u) * (uint32_t)blocks + b]);
                const float xsA = sh_xs[tl_xsA * (uint32_t)blocks + b];
                const float xsB = sh_xs[tl_xsB * (uint32_t)blocks + b];
                t0 += ws0 * xsA * (float)c0;
                t1 += ws1 * xsA * (float)c1;
                t2 += ws0 * xsB * (float)c2;
                t3 += ws1 * xsB * (float)c3;
            }
            /* static adjacent stack push (compile-time resolved) */
            if ((i & 1u) == 0u) { s0e0 = t0; s0e1 = t1; s0e2 = t2; s0e3 = t3; }
            else {
                t0 = s0e0 + t0; t1 = s0e1 + t1; t2 = s0e2 + t2; t3 = s0e3 + t3;
                if ((i & 2u) == 0u) { s1e0 = t0; s1e1 = t1; s1e2 = t2; s1e3 = t3; }
                else {
                    t0 = s1e0 + t0; t1 = s1e1 + t1; t2 = s1e2 + t2; t3 = s1e3 + t3;
                    if ((i & 4u) == 0u) { s2e0 = t0; s2e1 = t1; s2e2 = t2; s2e3 = t3; }
                    else {
                        t0 = s2e0 + t0; t1 = s2e1 + t1; t2 = s2e2 + t2; t3 = s2e3 + t3;
                        if ((i & 8u) == 0u) { s3e0 = t0; s3e1 = t1; s3e2 = t2; s3e3 = t3; }
                        else {
                            t0 = s3e0 + t0; t1 = s3e1 + t1; t2 = s3e2 + t2; t3 = s3e3 + t3;
                            if ((i & 16u) == 0u) { s4e0 = t0; s4e1 = t1; s4e2 = t2; s4e3 = t3; }
                            else {
                                t0 = s4e0 + t0; t1 = s4e1 + t1; t2 = s4e2 + t2; t3 = s4e3 + t3;
                                /* i == 31: t is the finished x_j */
                                switch (j & 3u) {
                                case 0u: acc00 += t0; acc10 += t1; acc20 += t2; acc30 += t3; break;
                                case 1u: acc01 += t0; acc11 += t1; acc21 += t2; acc31 += t3; break;
                                case 2u: acc02 += t0; acc12 += t1; acc22 += t2; acc32 += t3; break;
                                default: acc03 += t0; acc13 += t1; acc23 += t2; acc33 += t3; break;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    /* tail combine per T */
    float r0, r1, r2, r3;
    if (T == 32u) {
        r0 = acc00; r1 = acc10; r2 = acc20; r3 = acc30;
    } else if (T == 64u) {
        r0 = acc00 + acc01; r1 = acc10 + acc11; r2 = acc20 + acc21; r3 = acc30 + acc31;
    } else {
        r0 = (acc00 + acc02) + (acc01 + acc03);
        r1 = (acc10 + acc12) + (acc11 + acc13);
        r2 = (acc20 + acc22) + (acc21 + acc23);
        r3 = (acc30 + acc32) + (acc31 + acc33);
    }
    /* writes */
    const uint64_t rowa = row0 + n0;
    const uint64_t rowb = rowa + 1u;
    if (tokA < n_tok) {
        if (rowa < out_dim) out[tokA * out_stride + rowa] = r0;
        if (rowb < out_dim) out[tokA * out_stride + rowb] = r1;
    }
    if (tokB < n_tok) {
        if (rowa < out_dim) out[tokB * out_stride + rowa] = r2;
        if (rowb < out_dim) out[tokB * out_stride + rowb] = r3;
    }
}

static int cuda_q4_mma_ok(void);
static int cuda_q8_mma_attr_ready[DS4_MAX_GPUS][4];
static int cuda_q8_mma_try_launch(
        float *out,
        const unsigned char *w,
        const int8_t *xq,
        const float *xscale,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t n_tok,
        uint64_t blocks,
        uint64_t a_stride_blocks,
        uint64_t out_stride,
        uint32_t T) {
    static int disabled = -1;
    if (disabled < 0) disabled = getenv("DS4_CUDA_NO_Q8_MMA") != NULL ? 1 : 0;
    if (disabled || !cuda_q4_mma_ok()) return 0;
    if ((in_dim & 31u) != 0u || blocks > 256u || n_tok < 8u) return 0;
    if (((uintptr_t)w & 1u) || ((uintptr_t)xq & 3u) || ((uintptr_t)xscale & 3u)) return 0;
    const size_t shmem = (size_t)(64u * blocks * 2u + 16u * blocks * 4u);
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev < 0 || dev >= DS4_MAX_GPUS) return 0;
    const int ti = T == 32u ? 0 : (T == 64u ? 1 : (T == 128u ? 2 : 3));
    dim3 grid(((unsigned)out_dim + 63u) / 64u, ((unsigned)n_tok + 15u) / 16u, 1);
#define DS4_Q8_MMA_LAUNCH(TT) \
    do { \
        if (!cuda_q8_mma_attr_ready[dev][ti]) { \
            cudaFuncAttributes fn_attr; \
            if (cudaFuncGetAttributes(&fn_attr, matmul_q8_0_mma_exact_kernel<TT>) != cudaSuccess || \
                fn_attr.binaryVersion < 80) { \
                disabled = 1; \
                return 0; \
            } \
            if (cudaFuncSetAttribute(matmul_q8_0_mma_exact_kernel<TT>, \
                                     cudaFuncAttributeMaxDynamicSharedMemorySize, \
                                     (int)(64u * 256u * 2u + 16u * 256u * 4u)) != cudaSuccess) { \
                disabled = 1; \
                return 0; \
            } \
            cuda_q8_mma_attr_ready[dev][ti] = 1; \
        } \
        matmul_q8_0_mma_exact_kernel<TT><<<grid, 256, shmem>>>( \
                out, w, xq, xscale, in_dim, out_dim, n_tok, blocks, \
                a_stride_blocks, out_stride); \
    } while (0)
    if (T == 32u) DS4_Q8_MMA_LAUNCH(32u);
    else if (T == 64u) DS4_Q8_MMA_LAUNCH(64u);
    else if (T == 128u) DS4_Q8_MMA_LAUNCH(128u);
    else DS4_Q8_MMA_LAUNCH(256u);
#undef DS4_Q8_MMA_LAUNCH
    return cuda_ok(cudaGetLastError(), "matmul_q8_0 mma launch") ? 1 : -1;
}


__global__ static void dequant_q8_0_to_f16_kernel(
        __half *out,
        const unsigned char *w,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t blocks) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = in_dim * out_dim;
    if (gid >= n) return;
    uint64_t row = gid / in_dim;
    uint64_t i = gid - row * in_dim;
    uint64_t b = i / 32;
    uint64_t j = i - b * 32;
    const unsigned char *blk = w + (row * blocks + b) * 34;
    const __half scale = *(const __half *)blk;
    const int8_t q = *(const int8_t *)(blk + 2 + j);
    out[gid] = __hmul(scale, __float2half((float)q));
}

__global__ static void dequant_q8_0_to_f32_kernel(
        float *out,
        const unsigned char *w,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t blocks) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = in_dim * out_dim;
    if (gid >= n) return;
    uint64_t row = gid / in_dim;
    uint64_t i = gid - row * in_dim;
    uint64_t b = i / 32;
    uint64_t j = i - b * 32;
    const unsigned char *blk = w + (row * blocks + b) * 34;
    const float scale = __half2float(*(const __half *)blk);
    const int8_t q = *(const int8_t *)(blk + 2 + j);
    out[gid] = scale * (float)q;
}

__global__ static void grouped_q8_0_a_preq_warp8_kernel(
        float *low,
        const unsigned char *w,
        const int8_t *xq,
        const float *xscale,
        uint64_t group_dim,
        uint64_t rank,
        uint32_t n_groups,
        uint32_t n_tokens,
        uint64_t blocks,
        int use_dp4a) {
    const uint64_t row = (uint64_t)blockIdx.x * 8u + (threadIdx.x >> 5u);
    const uint64_t tok = (uint64_t)blockIdx.y;
    const uint32_t lane = threadIdx.x & 31u;
    const uint64_t low_dim = (uint64_t)n_groups * rank;
    if (row >= low_dim || tok >= n_tokens) return;

    const uint64_t group = row / rank;
    const uint64_t row_in_group = row - group * rank;
    const unsigned char *wr = w + (group * rank + row_in_group) * blocks * 34;
    const uint64_t xrow = tok * (uint64_t)n_groups + group;
    const int8_t *xqr = xq + xrow * blocks * 32;
    const float *xsr = xscale + xrow * blocks;
    float acc = 0.0f;

    for (uint64_t b = lane; b < blocks; b += 32u) {
        const uint64_t i0 = b * 32;
        const uint64_t bn = group_dim - i0 < 32 ? group_dim - i0 : 32;
        const __half *scale_h = (const __half *)(wr + b * 34);
        const int8_t *qs = (const int8_t *)(wr + b * 34 + 2);
        const int8_t *xqb = xqr + b * 32;
        int dot = dot_i8_block(qs, xqb, bn, use_dp4a);
        acc += __half2float(*scale_h) * xsr[b] * (float)dot;
    }
    acc = warp_sum_f32(acc);
    if (lane == 0) low[tok * low_dim + row] = acc;
}

__global__ static void grouped_q8_0_a_preq_warp8_tok2_kernel(
        float *low,
        const unsigned char *w,
        const int8_t *xq,
        const float *xscale,
        uint64_t group_dim,
        uint64_t rank,
        uint32_t n_groups,
        uint32_t n_tokens,
        uint64_t blocks,
        int use_dp4a) {
    const uint32_t tid_in_tok = threadIdx.x & 255u;
    const uint64_t row = (uint64_t)blockIdx.x * 8u + (tid_in_tok >> 5u);
    const uint64_t tok = (uint64_t)blockIdx.y * 2u + (threadIdx.x >> 8u);
    const uint32_t lane = threadIdx.x & 31u;
    const uint64_t low_dim = (uint64_t)n_groups * rank;

    float acc = 0.0f;
    if (row < low_dim && tok < n_tokens) {
        const uint64_t group = row / rank;
        const uint64_t row_in_group = row - group * rank;
        const unsigned char *wr = w + (group * rank + row_in_group) * blocks * 34u;
        const uint64_t xrow = tok * (uint64_t)n_groups + group;
        const int8_t *xqr = xq + xrow * blocks * 32u;
        const float *xsr = xscale + xrow * blocks;

        for (uint64_t b = lane; b < blocks; b += 32u) {
            const uint64_t i0 = b * 32u;
            const uint64_t bn = group_dim - i0 < 32u ? group_dim - i0 : 32u;
            const __half *scale_h = (const __half *)(wr + b * 34u);
            const int8_t *qs = (const int8_t *)(wr + b * 34u + 2u);
            const int8_t *xqb = xqr + b * 32u;
            const int dot = dot_i8_block(qs, xqb, bn, use_dp4a);
            acc += __half2float(*scale_h) * xsr[b] * (float)dot;
        }
    }
    acc = warp_sum_f32(acc);
    if (lane == 0 && row < low_dim && tok < n_tokens) {
        low[tok * low_dim + row] = acc;
    }
}

__global__ static void rms_norm_plain_kernel(float *out, const float *x, uint32_t n, uint32_t rows, float eps) {
    uint32_t row = blockIdx.x;
    if (row >= rows) return;
    const float *xr = x + (uint64_t)row * n;
    float *orow = out + (uint64_t)row * n;
    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        float v = xr[i];
        sum += v * v;
    }
    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    float scale = rsqrtf(partial[0] / (float)n + eps);
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        orow[i] = xr[i] * scale;
    }
}

/* Latency-optimized RMS norm for the common n==4096 decode shape: one global
 * read pass with register-batched loads, same per-thread accumulation order
 * and shared-memory tree as rms_norm_plain_kernel (bit-identical, fuzz
 * checked). */
__global__ static void rms_norm_plain_fast4096_kernel(float *out, const float *x, uint32_t n, uint32_t rows, float eps) {
    uint32_t row = blockIdx.x;
    if (row >= rows) return;
    const float *xr = x + (uint64_t)row * n;
    float *orow = out + (uint64_t)row * n;
    float v[16];
#pragma unroll
    for (uint32_t j = 0; j < 16u; j++) v[j] = xr[threadIdx.x + j * 256u];
    float sum = 0.0f;
#pragma unroll
    for (uint32_t j = 0; j < 16u; j++) sum += v[j] * v[j];
    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    float scale = rsqrtf(partial[0] / (float)n + eps);
#pragma unroll
    for (uint32_t j = 0; j < 16u; j++) orow[threadIdx.x + j * 256u] = v[j] * scale;
}

/* Batched-load RMS norm for larger rows (n multiple of 2048, e.g. the 16384
 * HC-concatenated decode rows). Two passes like the reference kernel, but
 * eight independent loads are issued per accumulation group; the per-thread
 * accumulation order (ascending i with stride 256) is unchanged, so results
 * are bit-identical. */
__global__ static void rms_norm_plain_batch8_kernel(float *out, const float *x, uint32_t n, uint32_t rows, float eps) {
    uint32_t row = blockIdx.x;
    if (row >= rows) return;
    const float *xr = x + (uint64_t)row * n;
    float *orow = out + (uint64_t)row * n;
    float sum = 0.0f;
#pragma unroll 1
    for (uint32_t i = threadIdx.x; i < n; i += 2048u) {
        const float v0 = xr[i];
        const float v1 = xr[i + 256u];
        const float v2 = xr[i + 512u];
        const float v3 = xr[i + 768u];
        const float v4 = xr[i + 1024u];
        const float v5 = xr[i + 1280u];
        const float v6 = xr[i + 1536u];
        const float v7 = xr[i + 1792u];
        sum += v0 * v0;
        sum += v1 * v1;
        sum += v2 * v2;
        sum += v3 * v3;
        sum += v4 * v4;
        sum += v5 * v5;
        sum += v6 * v6;
        sum += v7 * v7;
    }
    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    float scale = rsqrtf(partial[0] / (float)n + eps);
#pragma unroll 1
    for (uint32_t i = threadIdx.x; i < n; i += 2048u) {
        const float v0 = xr[i];
        const float v1 = xr[i + 256u];
        const float v2 = xr[i + 512u];
        const float v3 = xr[i + 768u];
        const float v4 = xr[i + 1024u];
        const float v5 = xr[i + 1280u];
        const float v6 = xr[i + 1536u];
        const float v7 = xr[i + 1792u];
        orow[i] = v0 * scale;
        orow[i + 256u] = v1 * scale;
        orow[i + 512u] = v2 * scale;
        orow[i + 768u] = v3 * scale;
        orow[i + 1024u] = v4 * scale;
        orow[i + 1280u] = v5 * scale;
        orow[i + 1536u] = v6 * scale;
        orow[i + 1792u] = v7 * scale;
    }
}

__global__ static void rms_norm_weight_kernel(float *out, const float *x, const float *w, uint32_t n, uint32_t rows, float eps) {
    uint32_t row = blockIdx.x;
    if (row >= rows) return;
    const float *xr = x + (uint64_t)row * n;
    float *orow = out + (uint64_t)row * n;
    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        float v = xr[i];
        sum += v * v;
    }
    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    float scale = rsqrtf(partial[0] / (float)n + eps);
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        orow[i] = xr[i] * scale * w[i];
    }
}

__global__ static void dsv4_qkv_rms_norm_rows_kernel(
        float *q_out,
        const float *q,
        const float *q_w,
        uint32_t q_n,
        float *kv_out,
        const float *kv,
        const float *kv_w,
        uint32_t kv_n,
        uint32_t rows,
        float eps) {
    const uint32_t row = blockIdx.x;
    const uint32_t which = blockIdx.y;
    if (row >= rows || which > 1u) return;
    const uint32_t n = which == 0u ? q_n : kv_n;
    const float *xr = (which == 0u ? q : kv) + (uint64_t)row * n;
    float *orow = (which == 0u ? q_out : kv_out) + (uint64_t)row * n;
    const float *w = which == 0u ? q_w : kv_w;
    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = xr[i];
        sum += v * v;
    }
    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    const float scale = rsqrtf(partial[0] / (float)n + eps);
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        orow[i] = xr[i] * scale * w[i];
    }
}

__global__ static void head_rms_norm_kernel(float *x, uint32_t n_tok, uint32_t n_head, uint32_t head_dim, float eps) {
    uint32_t row = blockIdx.x;
    if (row >= n_tok * n_head) return;
    float *xr = x + (uint64_t)row * head_dim;
    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x) {
        float v = xr[i];
        sum += v * v;
    }
    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    float scale = rsqrtf(partial[0] / (float)head_dim + eps);
    for (uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x) xr[i] *= scale;
}

__device__ static float rope_yarn_ramp_dev(float low, float high, int i0);

__global__ static void dsv4_qkv_rms_norm_rows_kv_rope_kernel(
        float *q_out,
        const float *q,
        const float *q_w,
        uint32_t q_n,
        float *kv_out,
        const float *kv,
        const float *kv_w,
        uint32_t kv_n,
        uint32_t rows,
        uint32_t kv_n_head,
        uint32_t kv_head_dim,
        uint32_t n_rot,
        uint32_t pos0,
        uint32_t n_ctx_orig,
        int inverse,
        float freq_base,
        float freq_scale,
        float ext_factor,
        float attn_factor,
        float beta_fast,
        float beta_slow,
        float eps) {
    const uint32_t row = blockIdx.x;
    const uint32_t which = blockIdx.y;
    if (row >= rows || which > 1u) return;
    const uint32_t n = which == 0u ? q_n : kv_n;
    const float *xr = (which == 0u ? q : kv) + (uint64_t)row * n;
    float *orow = (which == 0u ? q_out : kv_out) + (uint64_t)row * n;
    const float *w = which == 0u ? q_w : kv_w;
    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = xr[i];
        sum += v * v;
    }
    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    const float scale = rsqrtf(partial[0] / (float)n + eps);
    if (which == 0u) {
        for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
            orow[i] = xr[i] * scale * w[i];
        }
        return;
    }

    const uint32_t n_nope = kv_head_dim - n_rot;
    for (uint32_t h = 0; h < kv_n_head; h++) {
        const uint32_t head_base = h * kv_head_dim;
        for (uint32_t d = threadIdx.x; d < n_nope; d += blockDim.x) {
            const uint32_t i = head_base + d;
            orow[i] = xr[i] * scale * w[i];
        }
    }

    float corr0 = 0.0f, corr1 = 0.0f;
    if (ext_factor != 0.0f) {
        float denom = 2.0f * logf(freq_base);
        corr0 = floorf((float)n_rot * logf((float)n_ctx_orig / (beta_fast * 2.0f * (float)M_PI)) / denom);
        corr1 = ceilf((float)n_rot * logf((float)n_ctx_orig / (beta_slow * 2.0f * (float)M_PI)) / denom);
        corr0 = fmaxf(0.0f, corr0);
        corr1 = fminf((float)(n_rot - 1), corr1);
    }
    const uint32_t pairs_per_head = n_rot / 2u;
    const uint32_t total_pairs = kv_n_head * pairs_per_head;
    for (uint32_t p = threadIdx.x; p < total_pairs; p += blockDim.x) {
        const uint32_t h = p / pairs_per_head;
        const uint32_t pair = p - h * pairs_per_head;
        const uint32_t d = n_nope + pair * 2u;
        const uint32_t i0 = h * kv_head_dim + d;
        const uint32_t i = pair * 2u;
        float theta_extrap = (float)(pos0 + row) * powf(freq_base, -((float)i) / (float)n_rot);
        float theta_interp = freq_scale * theta_extrap;
        float theta = theta_interp;
        float mscale = attn_factor;
        if (ext_factor != 0.0f) {
            float ramp_mix = rope_yarn_ramp_dev(corr0, corr1, (int)i) * ext_factor;
            theta = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
            mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
        }
        float c = cosf(theta) * mscale;
        float s = sinf(theta) * mscale;
        if (inverse) s = -s;
        const float x0 = xr[i0] * scale * w[i0];
        const float x1 = xr[i0 + 1u] * scale * w[i0 + 1u];
        orow[i0] = x0 * c - x1 * s;
        orow[i0 + 1u] = x0 * s + x1 * c;
    }
}

__global__ static void head_rms_norm_rope_tail_kernel(
        float *x,
        uint32_t n_tok,
        uint32_t n_head,
        uint32_t head_dim,
        uint32_t n_rot,
        uint32_t pos0,
        uint32_t n_ctx_orig,
        int inverse,
        float freq_base,
        float freq_scale,
        float ext_factor,
        float attn_factor,
        float beta_fast,
        float beta_slow,
        float eps) {
    uint32_t row = blockIdx.x;
    if (row >= n_tok * n_head) return;
    uint32_t t = row / n_head;
    float *xr = x + (uint64_t)row * head_dim;
    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x) {
        float v = xr[i];
        sum += v * v;
    }
    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    const float scale = rsqrtf(partial[0] / (float)head_dim + eps);
    const uint32_t n_nope = head_dim - n_rot;
    for (uint32_t i = threadIdx.x; i < n_nope; i += blockDim.x) {
        xr[i] *= scale;
    }

    float corr0 = 0.0f, corr1 = 0.0f;
    if (ext_factor != 0.0f) {
        float denom = 2.0f * logf(freq_base);
        corr0 = floorf((float)n_rot * logf((float)n_ctx_orig / (beta_fast * 2.0f * (float)M_PI)) / denom);
        corr1 = ceilf((float)n_rot * logf((float)n_ctx_orig / (beta_slow * 2.0f * (float)M_PI)) / denom);
        corr0 = fmaxf(0.0f, corr0);
        corr1 = fminf((float)(n_rot - 1), corr1);
    }
    for (uint32_t pair = threadIdx.x; pair < n_rot / 2; pair += blockDim.x) {
        uint32_t i = pair * 2u;
        float theta_extrap = (float)(pos0 + t) * powf(freq_base, -((float)i) / (float)n_rot);
        float theta_interp = freq_scale * theta_extrap;
        float theta = theta_interp;
        float mscale = attn_factor;
        if (ext_factor != 0.0f) {
            float ramp_mix = rope_yarn_ramp_dev(corr0, corr1, (int)i) * ext_factor;
            theta = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
            mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
        }
        float c = cosf(theta) * mscale;
        float s = sinf(theta) * mscale;
        if (inverse) s = -s;
        float *tail = xr + n_nope;
        float x0 = tail[i] * scale;
        float x1 = tail[i + 1] * scale;
        tail[i] = x0 * c - x1 * s;
        tail[i + 1] = x0 * s + x1 * c;
    }
}

__device__ static float rope_yarn_ramp_dev(float low, float high, int i0) {
    float y = ((float)(i0 / 2) - low) / fmaxf(0.001f, high - low);
    return 1.0f - fminf(1.0f, fmaxf(0.0f, y));
}

__global__ static void rope_tail_kernel(
        float *x,
        uint32_t n_tok,
        uint32_t n_head,
        uint32_t head_dim,
        uint32_t n_rot,
        uint32_t pos0,
        uint32_t pos_stride,
        uint32_t n_ctx_orig,
        int inverse,
        float freq_base,
        float freq_scale,
        float ext_factor,
        float attn_factor,
        float beta_fast,
        float beta_slow) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t pairs = n_tok * n_head * (n_rot / 2);
    if (gid >= pairs) return;
    uint32_t pair = gid % (n_rot / 2);
    uint32_t tmp = gid / (n_rot / 2);
    uint32_t h = tmp % n_head;
    uint32_t t = tmp / n_head;
    uint32_t n_nope = head_dim - n_rot;
    uint32_t i = pair * 2;

    float corr0 = 0.0f, corr1 = 0.0f;
    if (ext_factor != 0.0f) {
        float denom = 2.0f * logf(freq_base);
        corr0 = floorf((float)n_rot * logf((float)n_ctx_orig / (beta_fast * 2.0f * (float)M_PI)) / denom);
        corr1 = ceilf((float)n_rot * logf((float)n_ctx_orig / (beta_slow * 2.0f * (float)M_PI)) / denom);
        corr0 = fmaxf(0.0f, corr0);
        corr1 = fminf((float)(n_rot - 1), corr1);
    }

    float theta_extrap = (float)(pos0 + t * pos_stride) * powf(freq_base, -((float)i) / (float)n_rot);
    float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    float mscale = attn_factor;
    if (ext_factor != 0.0f) {
        float ramp_mix = rope_yarn_ramp_dev(corr0, corr1, (int)i) * ext_factor;
        theta = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
        mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    float c = cosf(theta) * mscale;
    float s = sinf(theta) * mscale;
    if (inverse) s = -s;

    float *tail = x + ((uint64_t)t * n_head + h) * head_dim + n_nope;
    float x0 = tail[i];
    float x1 = tail[i + 1];
    tail[i] = x0 * c - x1 * s;
    tail[i + 1] = x0 * s + x1 * c;
}

__global__ static void rope_tail_decode_rows_kernel(
        float *x,
        cuda_attention_decode_row_table rows,
        uint32_t n_rows,
        uint32_t n_head,
        uint32_t head_dim,
        uint32_t n_rot,
        uint32_t n_ctx_orig,
        int inverse,
        float freq_base,
        float freq_scale,
        float ext_factor,
        float attn_factor,
        float beta_fast,
        float beta_slow) {
    const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t pairs = n_rows * n_head * (n_rot / 2u);
    if (gid >= pairs) return;
    const uint32_t pair = gid % (n_rot / 2u);
    const uint32_t tmp = gid / (n_rot / 2u);
    const uint32_t h = tmp % n_head;
    const uint32_t row = tmp / n_head;
    const uint32_t n_nope = head_dim - n_rot;
    const uint32_t i = pair * 2u;

    float corr0 = 0.0f, corr1 = 0.0f;
    if (ext_factor != 0.0f) {
        const float denom = 2.0f * logf(freq_base);
        corr0 = floorf((float)n_rot *
                       logf((float)n_ctx_orig /
                            (beta_fast * 2.0f * (float)M_PI)) / denom);
        corr1 = ceilf((float)n_rot *
                      logf((float)n_ctx_orig /
                           (beta_slow * 2.0f * (float)M_PI)) / denom);
        corr0 = fmaxf(0.0f, corr0);
        corr1 = fminf((float)(n_rot - 1u), corr1);
    }

    const float theta_extrap = (float)rows.row[row].pos *
        powf(freq_base, -((float)i) / (float)n_rot);
    const float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    float mscale = attn_factor;
    if (ext_factor != 0.0f) {
        const float ramp_mix =
            rope_yarn_ramp_dev(corr0, corr1, (int)i) * ext_factor;
        theta = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
        mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    const float c = cosf(theta) * mscale;
    float s = sinf(theta) * mscale;
    if (inverse) s = -s;

    float *tail = x + ((uint64_t)row * n_head + h) * head_dim + n_nope;
    const float x0 = tail[i];
    const float x1 = tail[i + 1u];
    tail[i] = x0 * c - x1 * s;
    tail[i + 1u] = x0 * s + x1 * c;
}

__device__ static float dsv4_e4m3fn_value_dev(int i) {
    int exp = (i >> 3) & 15;
    int mant = i & 7;
    if (exp == 0) return (float)mant * 0.001953125f;
    return (1.0f + (float)mant * 0.125f) * exp2f((float)exp - 7.0f);
}

__device__ static float dsv4_e4m3fn_dequant_dev(float x) {
    float sign = x < 0.0f ? -1.0f : 1.0f;
    float ax = fminf(fabsf(x), 448.0f);
    int lo = 0, hi = 126;
    while (lo < hi) {
        int mid = (lo + hi + 1) >> 1;
        if (dsv4_e4m3fn_value_dev(mid) <= ax) lo = mid;
        else hi = mid - 1;
    }
    int best = lo;
    if (best < 126) {
        float bd = fabsf(ax - dsv4_e4m3fn_value_dev(best));
        float nd = fabsf(ax - dsv4_e4m3fn_value_dev(best + 1));
        if (nd < bd || (nd == bd && (((best + 1) & 1) == 0) && ((best & 1) != 0))) best++;
    }
    return sign * dsv4_e4m3fn_value_dev(best);
}

__device__ static float dsv4_e2m1fn_value_dev(int i) {
    switch (i & 7) {
    case 0: return 0.0f;
    case 1: return 0.5f;
    case 2: return 1.0f;
    case 3: return 1.5f;
    case 4: return 2.0f;
    case 5: return 3.0f;
    case 6: return 4.0f;
    default: return 6.0f;
    }
}

__device__ static float dsv4_e2m1fn_dequant_dev(float x) {
    float sign = x < 0.0f ? -1.0f : 1.0f;
    float ax = fminf(fabsf(x), 6.0f);
    int best = 0;
    float best_diff = fabsf(ax - dsv4_e2m1fn_value_dev(0));
    for (int i = 1; i < 8; i++) {
        float diff = fabsf(ax - dsv4_e2m1fn_value_dev(i));
        if (diff < best_diff || (diff == best_diff && ((i & 1) == 0) && ((best & 1) != 0))) {
            best = i;
            best_diff = diff;
        }
    }
    return sign * dsv4_e2m1fn_value_dev(best);
}

__device__ static float model_scalar_dev(const void *base, uint64_t offset, uint32_t type, uint64_t idx) {
    const char *p = (const char *)base + offset;
    if (type == 1u) return __half2float(((const __half *)p)[idx]);
    return ((const float *)p)[idx];
}

__device__ static float rope_yarn_ramp_cpu_equiv_dev(float low, float high, int i0) {
    float y = ((float)(i0 / 2) - low) / fmaxf(0.001f, high - low);
    return 1.0f - fminf(1.0f, fmaxf(0.0f, y));
}

__device__ static DS4_CUDA_UNUSED void rope_tail_one_dev(float *x, uint32_t head_dim, uint32_t n_rot, uint32_t pos, uint32_t n_ctx_orig, float freq_base, float freq_scale, float ext_factor, float attn_factor, float beta_fast, float beta_slow) {
    uint32_t n_nope = head_dim - n_rot;
    float corr0 = 0.0f, corr1 = 0.0f;
    if (ext_factor != 0.0f) {
        float denom = 2.0f * logf(freq_base);
        corr0 = fmaxf(0.0f, floorf((float)n_rot * logf((float)n_ctx_orig / (beta_fast * 2.0f * (float)M_PI)) / denom));
        corr1 = fminf((float)(n_rot - 1), ceilf((float)n_rot * logf((float)n_ctx_orig / (beta_slow * 2.0f * (float)M_PI)) / denom));
    }
    for (uint32_t i = 0; i < n_rot; i += 2) {
        float theta_extrap = (float)pos * powf(freq_base, -((float)i) / (float)n_rot);
        float theta_interp = freq_scale * theta_extrap;
        float theta = theta_interp;
        float mscale = attn_factor;
        if (ext_factor != 0.0f) {
            float mix = rope_yarn_ramp_cpu_equiv_dev(corr0, corr1, (int)i) * ext_factor;
            theta = theta_interp * (1.0f - mix) + theta_extrap * mix;
            mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
        }
        float c = cosf(theta) * mscale;
        float s = sinf(theta) * mscale;
        float x0 = x[n_nope + i];
        float x1 = x[n_nope + i + 1];
        x[n_nope + i] = x0 * c - x1 * s;
        x[n_nope + i + 1] = x0 * s + x1 * c;
    }
}

__device__ static void fp8_kv_quantize_row(
        float    *xr,
        uint32_t  head_dim,
        uint32_t  n_rot,
        float    *scratch) {
    uint32_t tid = threadIdx.x;
    uint32_t n_nope = head_dim - n_rot;
    for (uint32_t off = 0; off < n_nope; off += 64) {
        float v = 0.0f;
        if (off + tid < n_nope) v = xr[off + tid];
        scratch[tid] = off + tid < n_nope ? fabsf(v) : 0.0f;
        __syncthreads();
        for (uint32_t stride = 32; stride > 0; stride >>= 1) {
            if (tid < stride) scratch[tid] = fmaxf(scratch[tid], scratch[tid + stride]);
            __syncthreads();
        }
        float scale = exp2f(ceilf(log2f(fmaxf(scratch[0], 1.0e-4f) / 448.0f)));
        if (off + tid < n_nope) {
            float q = dsv4_e4m3fn_dequant_dev(fminf(448.0f, fmaxf(-448.0f, v / scale))) * scale;
            xr[off + tid] = q;
        }
        __syncthreads();
    }
}

__global__ static void fp8_kv_quantize_kernel(
        float    *x,
        uint32_t  n_tok,
        uint32_t  head_dim,
        uint32_t  n_rot) {
    uint32_t row = blockIdx.x;
    if (row >= n_tok) return;
    __shared__ float scratch[64];
    fp8_kv_quantize_row(
            x + (uint64_t)row * head_dim, head_dim, n_rot, scratch);
}

__global__ static void fp8_kv_quantize_store_rows_kernel(
        float                           *x,
        cuda_attention_decode_row_table  rows,
        uint32_t                         n_rows,
        uint32_t                         head_dim,
        uint32_t                         n_rot) {
    const uint32_t row = blockIdx.x;
    if (row >= n_rows) return;
    __shared__ float scratch[64];
    float *xr = x + (uint64_t)row * head_dim;
    fp8_kv_quantize_row(xr, head_dim, n_rot, scratch);

    const ds4_gpu_attention_decode_row dsc = rows.row[row];
    float *raw = (float *)(uintptr_t)dsc.raw_kv;
    for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        raw[(uint64_t)dsc.raw_start * head_dim + d] =
            __half2float(__float2half(xr[d]));
    }
}

__global__ static void indexer_hadamard_fp4_kernel(float *x, uint32_t n_rows, uint32_t head_dim) {
    uint32_t row = blockIdx.x;
    uint32_t tid = threadIdx.x;
    if (row >= n_rows || head_dim != 128u || tid >= 128u) return;

    __shared__ float vals[128];
    __shared__ float absbuf[128];
    float *xr = x + (uint64_t)row * head_dim;
    vals[tid] = xr[tid];
    __syncthreads();

    for (uint32_t stride = 1u; stride < 128u; stride <<= 1u) {
        if ((tid & stride) == 0u) {
            uint32_t base = (tid & ~(2u * stride - 1u)) + (tid & (stride - 1u));
            float a = vals[base];
            float b = vals[base + stride];
            vals[base] = a + b;
            vals[base + stride] = a - b;
        }
        __syncthreads();
    }

    float v = vals[tid] * 0.08838834764831845f;
    uint32_t fp4_block = tid >> 5u;
    uint32_t lane = tid & 31u;
    uint32_t block_base = fp4_block * 32u;
    absbuf[tid] = fabsf(v);
    __syncthreads();

    for (uint32_t stride = 16u; stride > 0u; stride >>= 1u) {
        if (lane < stride) {
            absbuf[block_base + lane] = fmaxf(absbuf[block_base + lane],
                                              absbuf[block_base + lane + stride]);
        }
        __syncthreads();
    }

    float amax = fmaxf(absbuf[block_base], 7.052966104933725e-38f);
    float scale = exp2f(ceilf(log2f(amax / 6.0f)));
    xr[tid] = dsv4_e2m1fn_dequant_dev(fminf(6.0f, fmaxf(-6.0f, v / scale))) * scale;
}

__global__ static void store_raw_kv_batch_kernel(float *raw, const float *kv, uint32_t raw_cap, uint32_t pos0, uint32_t n_tokens, uint32_t head_dim) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_tokens * head_dim;
    if (gid >= n) return;
    uint32_t d = gid % head_dim;
    uint32_t t = gid / head_dim;
    uint32_t row = (pos0 + t) % raw_cap;
    raw[(uint64_t)row * head_dim + d] = __half2float(__float2half(kv[(uint64_t)t * head_dim + d]));
}

__global__ static void attention_prefill_raw_kernel(
        float *heads,
        const float *sinks,
        const float *q,
        const float *raw_kv,
        uint32_t n_tokens,
        uint32_t window,
        uint32_t n_head,
        uint32_t head_dim) {
    uint32_t t = blockIdx.x;
    uint32_t h = blockIdx.y;
    if (t >= n_tokens || h >= n_head) return;
    uint32_t raw_count = t + 1 < window ? t + 1 : window;
    uint32_t raw_start = t + 1 - raw_count;
    const float *qh = q + ((uint64_t)t * n_head + h) * head_dim;
    __shared__ float scores[256];
    __shared__ float partial[128];
    __shared__ float max_s;
    __shared__ float denom;
    float scale = rsqrtf((float)head_dim);
    float local_max = sinks[h];
    __syncthreads();
    for (uint32_t r = threadIdx.x; r < raw_count; r += blockDim.x) {
        const float *kv = raw_kv + (uint64_t)(raw_start + r) * head_dim;
        float dot = 0.0f;
        for (uint32_t d = 0; d < head_dim; d++) dot += qh[d] * kv[d];
        scores[r] = dot * scale;
        local_max = fmaxf(local_max, scores[r]);
    }
    partial[threadIdx.x] = local_max;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] = fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
        __syncthreads();
    }
    if (threadIdx.x == 0) max_s = partial[0];
    __syncthreads();
    if (threadIdx.x == 0) {
        float den = expf(sinks[h] - max_s);
        for (uint32_t r = 0; r < raw_count; r++) {
            scores[r] = expf(scores[r] - max_s);
            den += scores[r];
        }
        denom = den;
    }
    __syncthreads();
    float *oh = heads + ((uint64_t)t * n_head + h) * head_dim;
    for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (uint32_t r = 0; r < raw_count; r++) {
            acc += raw_kv[(uint64_t)(raw_start + r) * head_dim + d] * scores[r];
        }
        oh[d] = acc / denom;
    }
}

__global__ static void attention_prefill_mixed_kernel(
        float *heads,
        const float *sinks,
        const float *q,
        const float *raw_kv,
        const float *comp_kv,
        const float *comp_mask,
        uint32_t use_comp_mask,
        uint32_t n_tokens,
        uint32_t n_comp,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_head,
        uint32_t head_dim) {
    uint32_t t = blockIdx.x;
    uint32_t h = blockIdx.y;
    if (t >= n_tokens || h >= n_head) return;
    const float *qh = q + ((uint64_t)t * n_head + h) * head_dim;
    uint32_t raw_start = (window != 0 && t + 1u > window) ? t + 1u - window : 0u;
    uint32_t raw_count = t + 1u - raw_start;
    uint32_t visible_comp = (t + 1u) / ratio;
    if (visible_comp > n_comp) visible_comp = n_comp;
    __shared__ float scores[512];
    __shared__ float partial[256];
    __shared__ float max_s;
    __shared__ float denom;
    float scale = rsqrtf((float)head_dim);
    float local_max = sinks[h];
    uint32_t n_score = raw_count + visible_comp;

    for (uint32_t r = threadIdx.x; r < raw_count; r += blockDim.x) {
        const float *kvrow = raw_kv + (uint64_t)(raw_start + r) * head_dim;
        float dot = 0.0f;
        for (uint32_t d = 0; d < head_dim; d++) dot += qh[d] * kvrow[d];
        scores[r] = dot * scale;
        local_max = fmaxf(local_max, scores[r]);
    }
    for (uint32_t c = threadIdx.x; c < visible_comp; c += blockDim.x) {
        float add = use_comp_mask ? comp_mask[(uint64_t)t * n_comp + c] : 0.0f;
        float s = -INFINITY;
        if (add > -1.0e20f) {
            const float *kvrow = comp_kv + (uint64_t)c * head_dim;
            float dot = 0.0f;
            for (uint32_t d = 0; d < head_dim; d++) dot += qh[d] * kvrow[d];
            s = dot * scale + add;
        }
        scores[raw_count + c] = s;
        local_max = fmaxf(local_max, s);
    }
    partial[threadIdx.x] = local_max;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] = fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
        __syncthreads();
    }
    if (threadIdx.x == 0) max_s = partial[0];
    __syncthreads();
    float den_local = 0.0f;
    for (uint32_t i = threadIdx.x; i < n_score; i += blockDim.x) {
        scores[i] = expf(scores[i] - max_s);
        den_local += scores[i];
    }
    partial[threadIdx.x] = den_local;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) denom = partial[0] + expf(sinks[h] - max_s);
    __syncthreads();
    float *oh = heads + ((uint64_t)t * n_head + h) * head_dim;
    for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (uint32_t r = 0; r < raw_count; r++) acc += raw_kv[(uint64_t)(raw_start + r) * head_dim + d] * scores[r];
        for (uint32_t c = 0; c < visible_comp; c++) acc += comp_kv[(uint64_t)c * head_dim + d] * scores[raw_count + c];
        oh[d] = acc / denom;
    }
}

__global__ static void attention_prefill_raw_softmax_kernel(
        float *scores,
        const float *sinks,
        uint32_t n_tokens,
        uint32_t window,
        uint32_t n_keys) {
    uint32_t t = blockIdx.x;
    uint32_t h = blockIdx.y;
    if (t >= n_tokens) return;
    float *row = scores + ((uint64_t)h * n_tokens + t) * n_keys;
    __shared__ float partial[256];
    __shared__ float max_s;
    __shared__ float denom;
    float local_max = sinks[h];
    for (uint32_t k = threadIdx.x; k < n_keys; k += blockDim.x) {
        bool valid = k <= t && (window == 0 || t - k < window);
        float s = valid ? row[k] : -INFINITY;
        row[k] = s;
        local_max = fmaxf(local_max, s);
    }
    partial[threadIdx.x] = local_max;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] = fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
        __syncthreads();
    }
    if (threadIdx.x == 0) max_s = partial[0];
    __syncthreads();
    float den_local = 0.0f;
    for (uint32_t k = threadIdx.x; k < n_keys; k += blockDim.x) {
        float p = isfinite(row[k]) ? expf(row[k] - max_s) : 0.0f;
        row[k] = p;
        den_local += p;
    }
    partial[threadIdx.x] = den_local;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) denom = partial[0] + expf(sinks[h] - max_s);
    __syncthreads();
    for (uint32_t k = threadIdx.x; k < n_keys; k += blockDim.x) row[k] /= denom;
}

__global__ static void attention_prefill_mixed_softmax_kernel(
        float *scores,
        const float *sinks,
        const float *comp_mask,
        uint32_t use_comp_mask,
        uint32_t n_tokens,
        uint32_t n_comp,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_keys) {
    uint32_t t = blockIdx.x;
    uint32_t h = blockIdx.y;
    if (t >= n_tokens || ratio == 0) return;
    float *row = scores + ((uint64_t)h * n_tokens + t) * n_keys;
    __shared__ float partial[256];
    __shared__ float max_s;
    __shared__ float denom;
    float local_max = sinks[h];
    const uint32_t visible_comp = (t + 1u) / ratio;
    for (uint32_t k = threadIdx.x; k < n_keys; k += blockDim.x) {
        float s = -INFINITY;
        if (k < n_tokens) {
            if (k <= t && (window == 0 || t - k < window)) s = row[k];
        } else {
            uint32_t c = k - n_tokens;
            if (c < n_comp && c < visible_comp) {
                float add = use_comp_mask ? comp_mask[(uint64_t)t * n_comp + c] : 0.0f;
                if (add > -1.0e20f) s = row[k] + add;
            }
        }
        row[k] = s;
        local_max = fmaxf(local_max, s);
    }
    partial[threadIdx.x] = local_max;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] = fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
        __syncthreads();
    }
    if (threadIdx.x == 0) max_s = partial[0];
    __syncthreads();
    float den_local = 0.0f;
    for (uint32_t k = threadIdx.x; k < n_keys; k += blockDim.x) {
        float p = isfinite(row[k]) ? expf(row[k] - max_s) : 0.0f;
        row[k] = p;
        den_local += p;
    }
    partial[threadIdx.x] = den_local;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) denom = partial[0] + expf(sinks[h] - max_s);
    __syncthreads();
    for (uint32_t k = threadIdx.x; k < n_keys; k += blockDim.x) row[k] /= denom;
}

__global__ static void attention_prefill_pack_mixed_kv_kernel(
        float *dst,
        const float *raw_kv,
        const float *comp_kv,
        uint32_t n_tokens,
        uint32_t n_comp,
        uint32_t head_dim) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)(n_tokens + n_comp) * head_dim;
    if (gid >= n) return;
    uint32_t d = gid % head_dim;
    uint32_t r = gid / head_dim;
    dst[gid] = r < n_tokens ? raw_kv[(uint64_t)r * head_dim + d]
                             : comp_kv[(uint64_t)(r - n_tokens) * head_dim + d];
}

__global__ static void attention_prefill_unpack_heads_kernel(
        float *heads,
        const float *tmp,
        uint32_t n_tokens,
        uint32_t n_head,
        uint32_t head_dim) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_tokens * n_head * head_dim;
    if (gid >= n) return;
    uint32_t d = gid % head_dim;
    uint64_t q = gid / head_dim;
    uint32_t h = q % n_head;
    uint32_t t = q / n_head;
    heads[gid] = tmp[((uint64_t)h * n_tokens + t) * head_dim + d];
}

__global__ static void attention_pack_group_heads_f16_kernel(
        __half *dst,
        const float *heads,
        uint32_t n_tokens,
        uint32_t n_groups,
        uint32_t group_dim) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_groups * n_tokens * group_dim;
    if (gid >= n) return;
    uint32_t d = gid % group_dim;
    uint64_t q = gid / group_dim;
    uint32_t t = q % n_tokens;
    uint32_t g = q / n_tokens;
    dst[gid] = __float2half(heads[((uint64_t)t * n_groups + g) * group_dim + d]);
}

__global__ static void attention_unpack_group_low_kernel(
        float *low,
        const float *tmp,
        uint32_t n_tokens,
        uint32_t n_groups,
        uint32_t rank) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_groups * n_tokens * rank;
    if (gid >= n) return;
    uint32_t r = gid % rank;
    uint64_t q = gid / rank;
    uint32_t t = q % n_tokens;
    uint32_t g = q / n_tokens;
    uint32_t low_dim = n_groups * rank;
    low[(uint64_t)t * low_dim + (uint64_t)g * rank + r] = tmp[gid];
}

__global__ static void attention_decode_mixed_kernel(
        float *heads,
        const float *sinks,
        const float *q,
        const float *raw_kv,
        const float *comp_kv,
        const float *comp_mask,
        uint32_t use_comp_mask,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t n_raw,
        uint32_t raw_cap,
        uint32_t raw_start,
        uint32_t n_comp,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_head,
        uint32_t head_dim,
        uint32_t score_lanes_single) {
    uint32_t t = blockIdx.x;
    uint32_t h = blockIdx.y;
    if (t >= n_tokens || h >= n_head) return;
    const bool single_all = (n_tokens == 1u && ratio == 0u);
    uint32_t qpos = pos0 + t;
    uint32_t first_raw_pos = pos0 + n_tokens - n_raw;
    uint32_t visible_comp = single_all ? n_comp : (n_comp ? (qpos + 1u) / ratio : 0u);
    if (visible_comp > n_comp) visible_comp = n_comp;
    const float *qh = q + ((uint64_t)t * n_head + h) * head_dim;
    __shared__ float scores[DS4_CUDA_ATTENTION_SCORE_CAP];
    __shared__ uint32_t raw_rows[256];
    __shared__ float partial[256];
    __shared__ float max_s;
    __shared__ float denom;
    __shared__ uint32_t raw_count;
    __shared__ uint32_t raw_first_idx;
    const uint32_t score_threads = blockDim.x > 256u ? 256u : blockDim.x;
    const bool score_thread = threadIdx.x < score_threads;
    float scale = rsqrtf((float)head_dim);
    if (threadIdx.x == 0) {
        raw_count = 0;
        raw_first_idx = 0;
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
    }
    __syncthreads();
    if (score_thread) {
        for (uint32_t r = threadIdx.x; r < raw_count; r += score_threads) {
            raw_rows[r] = (raw_start + raw_first_idx + r) % raw_cap;
        }
    }
    __syncthreads();
    uint32_t n_score = raw_count + visible_comp;
    float local_max = sinks[h];
    if (score_thread) {
        if (visible_comp == 0 || (n_tokens == 1u && score_lanes_single == 0u)) {
            for (uint32_t r = threadIdx.x; r < raw_count; r += score_threads) {
                const float *kvrow = raw_kv + (uint64_t)raw_rows[r] * head_dim;
                float dot = 0.0f;
                for (uint32_t d = 0; d < head_dim; d++) dot += qh[d] * kvrow[d];
                scores[r] = dot * scale;
                local_max = fmaxf(local_max, scores[r]);
            }
            for (uint32_t c = threadIdx.x; c < visible_comp; c += score_threads) {
                float add = use_comp_mask ? comp_mask[(uint64_t)t * n_comp + c] : 0.0f;
                float s = -INFINITY;
                if (add > -1.0e20f) {
                    const float *kvrow = comp_kv + (uint64_t)c * head_dim;
                    float dot = 0.0f;
                    for (uint32_t d = 0; d < head_dim; d++) dot += qh[d] * kvrow[d];
                    s = dot * scale + add;
                }
                scores[raw_count + c] = s;
                local_max = fmaxf(local_max, s);
            }
        } else if (n_tokens == 1u && score_lanes_single == 4u) {
            uint32_t qlane = threadIdx.x & 3u;
            uint32_t qgroup = threadIdx.x >> 2u;
            for (uint32_t row0 = 0; row0 < n_score; row0 += 64u) {
                uint32_t row = row0 + qgroup;
                if (row < n_score) {
                    float add = 0.0f;
                    const float *kvrow = NULL;
                    if (row < raw_count) {
                        kvrow = raw_kv + (uint64_t)raw_rows[row] * head_dim;
                    } else {
                        uint32_t c = row - raw_count;
                        add = use_comp_mask ? comp_mask[(uint64_t)t * n_comp + c] : 0.0f;
                        if (add > -1.0e20f) kvrow = comp_kv + (uint64_t)c * head_dim;
                    }
                    float s = -INFINITY;
                    if (kvrow) {
                        float dot = 0.0f;
                        for (uint32_t d = qlane; d < head_dim; d += 4u) dot += qh[d] * kvrow[d];
                        const uint32_t mask = 0xfu << (threadIdx.x & 28u);
                        dot += __shfl_down_sync(mask, dot, 2, 4);
                        dot += __shfl_down_sync(mask, dot, 1, 4);
                        s = dot * scale + add;
                    }
                    if (qlane == 0) scores[row] = s;
                }
            }
            __syncthreads();
            for (uint32_t i = threadIdx.x; i < n_score; i += score_threads) {
                local_max = fmaxf(local_max, scores[i]);
            }
        } else {
            uint32_t qlane = threadIdx.x & 7u;
            uint32_t qgroup = threadIdx.x >> 3u;
            for (uint32_t row0 = 0; row0 < n_score; row0 += 32u) {
                uint32_t row = row0 + qgroup;
                if (row < n_score) {
                    float add = 0.0f;
                    const float *kvrow = NULL;
                    if (row < raw_count) {
                        kvrow = raw_kv + (uint64_t)raw_rows[row] * head_dim;
                    } else {
                        uint32_t c = row - raw_count;
                        add = use_comp_mask ? comp_mask[(uint64_t)t * n_comp + c] : 0.0f;
                        if (add > -1.0e20f) kvrow = comp_kv + (uint64_t)c * head_dim;
                    }
                    float s = -INFINITY;
                    if (kvrow) {
                        float dot = 0.0f;
                        for (uint32_t d = qlane; d < head_dim; d += 8u) dot += qh[d] * kvrow[d];
                        const uint32_t mask = 0xffu << (threadIdx.x & 24u);
                        for (uint32_t off = 4u; off > 0u; off >>= 1u) {
                            dot += __shfl_down_sync(mask, dot, off, 8);
                        }
                        s = dot * scale + add;
                    }
                    if (qlane == 0) scores[row] = s;
                }
            }
            __syncthreads();
            for (uint32_t i = threadIdx.x; i < n_score; i += score_threads) {
                local_max = fmaxf(local_max, scores[i]);
            }
        }
    }
    if (score_thread) partial[threadIdx.x] = local_max;
    __syncthreads();
    for (uint32_t stride = score_threads >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] = fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
        __syncthreads();
    }
    if (threadIdx.x == 0) max_s = partial[0];
    __syncthreads();
    float den_local = 0.0f;
    if (score_thread) {
        for (uint32_t i = threadIdx.x; i < n_score; i += score_threads) {
            scores[i] = expf(scores[i] - max_s);
            den_local += scores[i];
        }
    }
    if (score_thread) partial[threadIdx.x] = den_local;
    __syncthreads();
    for (uint32_t stride = score_threads >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) denom = partial[0] + expf(sinks[h] - max_s);
    __syncthreads();
    float *oh = heads + ((uint64_t)t * n_head + h) * head_dim;
    if (head_dim == 512u && blockDim.x >= 512u) {
        uint32_t d = threadIdx.x;
        float acc = 0.0f;
        for (uint32_t r = 0; r < raw_count; r++) {
            float s = scores[r];
            const float *kv = raw_kv + (uint64_t)raw_rows[r] * head_dim;
            acc += kv[d] * s;
        }
        for (uint32_t c = 0; c < visible_comp; c++) {
            float s = scores[raw_count + c];
            const float *kv = comp_kv + (uint64_t)c * head_dim;
            acc += kv[d] * s;
        }
        oh[d] = acc / denom;
    } else if (head_dim == 512u && blockDim.x == 256u) {
        uint32_t d0 = threadIdx.x;
        uint32_t d1 = d0 + 256u;
        float acc0 = 0.0f;
        float acc1 = 0.0f;
        for (uint32_t r = 0; r < raw_count; r++) {
            float s = scores[r];
            const float *kv = raw_kv + (uint64_t)raw_rows[r] * head_dim;
            acc0 += kv[d0] * s;
            acc1 += kv[d1] * s;
        }
        for (uint32_t c = 0; c < visible_comp; c++) {
            float s = scores[raw_count + c];
            const float *kv = comp_kv + (uint64_t)c * head_dim;
            acc0 += kv[d0] * s;
            acc1 += kv[d1] * s;
        }
        oh[d0] = acc0 / denom;
        oh[d1] = acc1 / denom;
    } else {
        for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
            float acc = 0.0f;
            for (uint32_t r = 0; r < raw_count; r++) acc += raw_kv[(uint64_t)raw_rows[r] * head_dim + d] * scores[r];
            for (uint32_t c = 0; c < visible_comp; c++) acc += comp_kv[(uint64_t)c * head_dim + d] * scores[raw_count + c];
            oh[d] = acc / denom;
        }
    }
}

__global__ static void attention_decode_score_split_scores_kernel(
        float *score_out,
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
        uint32_t head_dim,
        uint32_t S) {
    const uint32_t h = blockIdx.y;
    const uint32_t j = blockIdx.z;
    if (h >= n_head || j >= S) return;
    const bool single_all = (ratio == 0u);
    const uint32_t qpos = pos0;
    const uint32_t first_raw_pos = pos0 + 1u - n_raw;
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
    const uint32_t n_score = raw_count + visible_comp;
    if (n_score == 0u) return;

    const uint32_t qbase = n_score / S;
    const uint32_t rem = n_score % S;
    const uint32_t g0 = j * qbase + (j < rem ? j : rem);
    const uint32_t cnt = qbase + (j < rem ? 1u : 0u);
    const uint32_t g1 = g0 + cnt;
    const float *qh = q + (uint64_t)h * head_dim;
    float *row_scores = score_out + (uint64_t)h * n_score;
    const float scale = rsqrtf((float)head_dim);

    for (uint32_t g = g0 + threadIdx.x; g < g1; g += blockDim.x) {
        float s = -INFINITY;
        if (g < raw_count) {
            const uint32_t raw_row =
                (raw_start + raw_first_idx + g) % raw_cap;
            const float *kvrow = raw_kv + (uint64_t)raw_row * head_dim;
            float dot = 0.0f;
            for (uint32_t d = 0; d < head_dim; d++) dot += qh[d] * kvrow[d];
            s = dot * scale;
        } else {
            const uint32_t cidx = g - raw_count;
            const float add = use_comp_mask ? comp_mask[(uint64_t)cidx] : 0.0f;
            if (add > -1.0e20f) {
                const float *kvrow = comp_kv + (uint64_t)cidx * head_dim;
                float dot = 0.0f;
                for (uint32_t d = 0; d < head_dim; d++) dot += qh[d] * kvrow[d];
                s = dot * scale + add;
            }
        }
        row_scores[g] = s;
    }
}

__device__ __forceinline__ float ds4_dot_scalar_ldg(
        const float *a,
        const float *b,
        uint32_t n) {
    float dot = 0.0f;
    for (uint32_t d = 0; d < n; d++) dot += __ldg(a + d) * __ldg(b + d);
    return dot;
}

__global__ static void attention_decode_score_split_scores_ldg_kernel(
        float *score_out,
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
        uint32_t head_dim,
        uint32_t S) {
    const uint32_t h = blockIdx.y;
    const uint32_t j = blockIdx.z;
    if (h >= n_head || j >= S) return;
    const bool single_all = (ratio == 0u);
    const uint32_t qpos = pos0;
    const uint32_t first_raw_pos = pos0 + 1u - n_raw;
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
    const uint32_t n_score = raw_count + visible_comp;
    if (n_score == 0u) return;

    const uint32_t qbase = n_score / S;
    const uint32_t rem = n_score % S;
    const uint32_t g0 = j * qbase + (j < rem ? j : rem);
    const uint32_t cnt = qbase + (j < rem ? 1u : 0u);
    const uint32_t g1 = g0 + cnt;
    const float *qh = q + (uint64_t)h * head_dim;
    float *row_scores = score_out + (uint64_t)h * n_score;
    const float scale = rsqrtf((float)head_dim);

    for (uint32_t g = g0 + threadIdx.x; g < g1; g += blockDim.x) {
        float s = -INFINITY;
        if (g < raw_count) {
            const uint32_t raw_row =
                (raw_start + raw_first_idx + g) % raw_cap;
            const float *kvrow = raw_kv + (uint64_t)raw_row * head_dim;
            const float dot = ds4_dot_scalar_ldg(qh, kvrow, head_dim);
            s = dot * scale;
        } else {
            const uint32_t cidx = g - raw_count;
            const float add = use_comp_mask ? comp_mask[(uint64_t)cidx] : 0.0f;
            if (add > -1.0e20f) {
                const float *kvrow = comp_kv + (uint64_t)cidx * head_dim;
                const float dot = ds4_dot_scalar_ldg(qh, kvrow, head_dim);
                s = dot * scale + add;
            }
        }
        row_scores[g] = s;
    }
}

/* Head-tiled exact score kernel for head_dim==512.
 *
 * The reference score kernel assigns one (head, row-chunk) per block and lets
 * every thread walk one KV row with a scalar sequential dot. Because MQA
 * shares the same KV rows across all 64 heads, that reference layout re-reads
 * every KV row once per head, and the per-thread row walk is fully
 * uncoalesced (threads stride 2KB apart), which multiplies L2 traffic again.
 *
 * This kernel keeps the per-score arithmetic bit-identical (same ascending-d
 * scalar accumulation `dot += q[d] * kv[d]`, same `dot * scale [+ add]`
 * epilogue, same masked-row/raw-window classification) but stages a 16-row KV
 * tile and a 16-head Q tile in shared memory with coalesced global loads, so
 * each KV row is read from L2 once per 16 heads instead of once per head.
 * Scores are independent outputs, so retiling the (head, row) space cannot
 * change any output bit as long as each individual dot keeps its order. */
#define DS4_SCORE_TILE_HEADS 16u
#define DS4_SCORE_TILE_ROWS 16u
#define DS4_SCORE_TILE_STRIDE 516u /* 512 + 4 floats: 16B-aligned rows, banks shifted by 4 */

__global__ static void attention_decode_score_split_scores_tile512_kernel(
        float *score_out,
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
    const bool single_all = (ratio == 0u);
    const uint32_t qpos = pos0;
    const uint32_t first_raw_pos = pos0 + 1u - n_raw;
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
    const uint32_t n_score = raw_count + visible_comp;
    if (n_score == 0u) return;

    extern __shared__ float score_tile_shared[];
    float *sh_q = score_tile_shared; /* 16 x 516 */
    float *sh_kv = sh_q + DS4_SCORE_TILE_HEADS * DS4_SCORE_TILE_STRIDE; /* 16 x 516 */
    __shared__ float sh_add[DS4_SCORE_TILE_ROWS];

    const uint32_t g_base = blockIdx.x * DS4_SCORE_TILE_ROWS;
    const uint32_t h_base = blockIdx.y * DS4_SCORE_TILE_HEADS;
    if (g_base >= n_score || h_base >= n_head) return;

    /* Cooperative Q tile load: 16 heads x 512 floats, float4 coalesced. */
    {
        const float4 *q4 = (const float4 *)(q + (uint64_t)h_base * 512u);
        const uint32_t tile_heads =
            n_head - h_base < DS4_SCORE_TILE_HEADS ? n_head - h_base : DS4_SCORE_TILE_HEADS;
        for (uint32_t idx = threadIdx.x; idx < tile_heads * 128u; idx += blockDim.x) {
            const uint32_t hh = idx >> 7u;      /* head within tile */
            const uint32_t dd = idx & 127u;     /* float4 within row */
            const float4 v = q4[hh * 128u + dd];
            float *dst = sh_q + hh * DS4_SCORE_TILE_STRIDE + dd * 4u;
            dst[0] = v.x; dst[1] = v.y; dst[2] = v.z; dst[3] = v.w;
        }
    }
    /* Row classification + mask staging (thread per row). */
    if (threadIdx.x < DS4_SCORE_TILE_ROWS) {
        const uint32_t g = g_base + threadIdx.x;
        float add = -INFINITY;
        if (g < n_score) {
            if (g < raw_count) {
                add = 0.0f; /* raw rows are always visible */
            } else {
                const uint32_t cidx = g - raw_count;
                add = use_comp_mask ? comp_mask[(uint64_t)cidx] : 0.0f;
            }
        }
        sh_add[threadIdx.x] = add;
    }
    __syncthreads();
    /* Cooperative KV tile load: two rows at a time, float4 coalesced.
     * Masked rows (add <= -1e20) are skipped; their scores never read KV. */
    {
        const uint32_t rows_per_pass = blockDim.x >> 7u; /* 128 threads per row */
        const uint32_t rr0 = threadIdx.x >> 7u;
        const uint32_t dd = threadIdx.x & 127u;
        for (uint32_t r = rr0; r < DS4_SCORE_TILE_ROWS; r += rows_per_pass) {
            const uint32_t g = g_base + r;
            if (g >= n_score) continue;
            const bool visible = g < raw_count || sh_add[r] > -1.0e20f;
            if (!visible) continue;
            const float4 *src;
            if (g < raw_count) {
                const uint32_t raw_row = (raw_start + raw_first_idx + g) % raw_cap;
                src = (const float4 *)(raw_kv + (uint64_t)raw_row * 512u);
            } else {
                const uint32_t cidx = g - raw_count;
                src = (const float4 *)(comp_kv + (uint64_t)cidx * 512u);
            }
            const float4 v = src[dd];
            float *dst = sh_kv + r * DS4_SCORE_TILE_STRIDE + dd * 4u;
            dst[0] = v.x; dst[1] = v.y; dst[2] = v.z; dst[3] = v.w;
        }
    }
    __syncthreads();

    /* One score per thread: r = tid&15 (consecutive threads, coalesced score
     * writes), h = tid>>4. The dot keeps the reference kernel's exact scalar
     * ascending-d accumulation. */
    const uint32_t r = threadIdx.x & (DS4_SCORE_TILE_ROWS - 1u);
    const uint32_t h = h_base + (threadIdx.x >> 4u);
    const uint32_t g = g_base + r;
    if (h >= n_head || g >= n_score) return;
    const float scale = rsqrtf((float)head_dim);
    float *row_scores = score_out + (uint64_t)h * n_score;
    const float *qh = sh_q + (uint64_t)(threadIdx.x >> 4u) * DS4_SCORE_TILE_STRIDE;
    const float *kvrow = sh_kv + (uint64_t)r * DS4_SCORE_TILE_STRIDE;
    float s = -INFINITY;
    const bool need_dot = g < raw_count || sh_add[r] > -1.0e20f;
    if (need_dot) {
        /* The reference kernel's runtime-trip loop compiles to one sequential
         * FFMA chain. Keep exactly that accumulation order here: batched loads
         * for latency hiding, but a single explicit ascending fma chain. */
        float dot = 0.0f;
#pragma unroll 1
        for (uint32_t d = 0; d < 512u; d += 8u) {
            const float a0 = qh[d + 0u], a1 = qh[d + 1u];
            const float a2 = qh[d + 2u], a3 = qh[d + 3u];
            const float a4 = qh[d + 4u], a5 = qh[d + 5u];
            const float a6 = qh[d + 6u], a7 = qh[d + 7u];
            const float b0 = kvrow[d + 0u], b1 = kvrow[d + 1u];
            const float b2 = kvrow[d + 2u], b3 = kvrow[d + 3u];
            const float b4 = kvrow[d + 4u], b5 = kvrow[d + 5u];
            const float b6 = kvrow[d + 6u], b7 = kvrow[d + 7u];
            dot = __fmaf_rn(a0, b0, dot);
            dot = __fmaf_rn(a1, b1, dot);
            dot = __fmaf_rn(a2, b2, dot);
            dot = __fmaf_rn(a3, b3, dot);
            dot = __fmaf_rn(a4, b4, dot);
            dot = __fmaf_rn(a5, b5, dot);
            dot = __fmaf_rn(a6, b6, dot);
            dot = __fmaf_rn(a7, b7, dot);
        }
        if (g < raw_count) {
            s = dot * scale;
        } else {
            /* The reference expression `dot * scale + add` contracts to one
             * FFMA; keep that exact contraction explicit. */
            s = __fmaf_rn(dot, scale, sh_add[r]);
        }
    }
    row_scores[g] = s;
}

/* Multi-session form of the exact tiled score kernel. Each z-slice selects a
 * private KV table entry, while every individual score keeps the same scalar
 * ascending-d FMA chain as the one-session kernel. */
__global__ static void attention_decode_score_split_scores_tile512_rows_kernel(
        float *score_out,
        const float *q,
        cuda_attention_decode_row_table rows,
        uint32_t n_rows,
        uint32_t score_stride,
        uint32_t n_head,
        uint32_t head_dim) {
    const uint32_t row = blockIdx.z;
    if (row >= n_rows) return;
    const ds4_gpu_attention_decode_row dsc = rows.row[row];
    if (dsc.indexed) return;

    const float *raw_kv = (const float *)(uintptr_t)dsc.raw_kv;
    const float *comp_kv = (const float *)(uintptr_t)dsc.comp_kv;
    const bool single_all = dsc.ratio == 0u;
    const uint32_t qpos = dsc.pos;
    const uint32_t first_raw_pos = dsc.pos + 1u - dsc.n_raw;
    uint32_t visible_comp = single_all
        ? dsc.n_comp
        : (dsc.n_comp ? (qpos + 1u) / dsc.ratio : 0u);
    if (visible_comp > dsc.n_comp) visible_comp = dsc.n_comp;

    uint32_t raw_count = 0u;
    uint32_t raw_first_idx = 0u;
    if (dsc.n_raw != 0u) {
        const uint32_t raw_last_pos = first_raw_pos + dsc.n_raw - 1u;
        if (single_all) {
            raw_count = dsc.n_raw > 256u ? 256u : dsc.n_raw;
        } else if (qpos >= first_raw_pos) {
            uint32_t lo = first_raw_pos;
            if (dsc.window != 0u && qpos + 1u > dsc.window) {
                const uint32_t wlo = qpos + 1u - dsc.window;
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
    const uint32_t n_score = raw_count + visible_comp;

    extern __shared__ float score_tile_shared[];
    float *sh_q = score_tile_shared;
    float *sh_kv = sh_q + DS4_SCORE_TILE_HEADS * DS4_SCORE_TILE_STRIDE;

    const uint32_t g_base = blockIdx.x * DS4_SCORE_TILE_ROWS;
    const uint32_t h_base = blockIdx.y * DS4_SCORE_TILE_HEADS;
    if (g_base >= n_score || h_base >= n_head) return;

    {
        const float4 *q4 = (const float4 *)(
            q + ((uint64_t)row * n_head + h_base) * head_dim);
        const uint32_t tile_heads =
            n_head - h_base < DS4_SCORE_TILE_HEADS
                ? n_head - h_base : DS4_SCORE_TILE_HEADS;
        for (uint32_t idx = threadIdx.x;
             idx < tile_heads * 128u;
             idx += blockDim.x) {
            const uint32_t hh = idx >> 7u;
            const uint32_t dd = idx & 127u;
            const float4 v = q4[hh * 128u + dd];
            float *dst = sh_q + hh * DS4_SCORE_TILE_STRIDE + dd * 4u;
            dst[0] = v.x; dst[1] = v.y; dst[2] = v.z; dst[3] = v.w;
        }
    }
    __syncthreads();
    {
        const uint32_t rows_per_pass = blockDim.x >> 7u;
        const uint32_t rr0 = threadIdx.x >> 7u;
        const uint32_t dd = threadIdx.x & 127u;
        for (uint32_t r = rr0; r < DS4_SCORE_TILE_ROWS; r += rows_per_pass) {
            const uint32_t g = g_base + r;
            if (g >= n_score) continue;
            const float4 *src;
            if (g < raw_count) {
                const uint32_t raw_row =
                    (dsc.raw_start + raw_first_idx + g) % dsc.raw_cap;
                src = (const float4 *)(raw_kv + (uint64_t)raw_row * head_dim);
            } else {
                src = (const float4 *)(comp_kv +
                    (uint64_t)(g - raw_count) * head_dim);
            }
            const float4 v = src[dd];
            float *dst = sh_kv + r * DS4_SCORE_TILE_STRIDE + dd * 4u;
            dst[0] = v.x; dst[1] = v.y; dst[2] = v.z; dst[3] = v.w;
        }
    }
    __syncthreads();

    const uint32_t r = threadIdx.x & (DS4_SCORE_TILE_ROWS - 1u);
    const uint32_t h = h_base + (threadIdx.x >> 4u);
    const uint32_t g = g_base + r;
    if (h >= n_head || g >= n_score) return;
    const float scale = rsqrtf((float)head_dim);
    float *row_scores = score_out +
        ((uint64_t)row * n_head + h) * score_stride;
    const float *qh = sh_q +
        (uint64_t)(threadIdx.x >> 4u) * DS4_SCORE_TILE_STRIDE;
    const float *kvrow = sh_kv + (uint64_t)r * DS4_SCORE_TILE_STRIDE;
    float dot = 0.0f;
#pragma unroll 1
    for (uint32_t dd = 0; dd < 512u; dd += 8u) {
        const float a0 = qh[dd + 0u], a1 = qh[dd + 1u];
        const float a2 = qh[dd + 2u], a3 = qh[dd + 3u];
        const float a4 = qh[dd + 4u], a5 = qh[dd + 5u];
        const float a6 = qh[dd + 6u], a7 = qh[dd + 7u];
        const float b0 = kvrow[dd + 0u], b1 = kvrow[dd + 1u];
        const float b2 = kvrow[dd + 2u], b3 = kvrow[dd + 3u];
        const float b4 = kvrow[dd + 4u], b5 = kvrow[dd + 5u];
        const float b6 = kvrow[dd + 6u], b7 = kvrow[dd + 7u];
        dot = __fmaf_rn(a0, b0, dot);
        dot = __fmaf_rn(a1, b1, dot);
        dot = __fmaf_rn(a2, b2, dot);
        dot = __fmaf_rn(a3, b3, dot);
        dot = __fmaf_rn(a4, b4, dot);
        dot = __fmaf_rn(a5, b5, dot);
        dot = __fmaf_rn(a6, b6, dot);
        dot = __fmaf_rn(a7, b7, dot);
    }
    row_scores[g] = g < raw_count
        ? dot * scale
        : __fmaf_rn(dot, scale, 0.0f);
}

__device__ __forceinline__ float ds4_dot512_float4_ordered(
        const float *a,
        const float *b) {
    const float4 *a4 = (const float4 *)a;
    const float4 *b4 = (const float4 *)b;
    float dot = 0.0f;
#pragma unroll 1
    for (uint32_t i = 0; i < 128u; i++) {
        const float4 av = a4[i];
        const float4 bv = b4[i];
        dot = __fadd_rn(dot, __fmul_rn(av.x, bv.x));
        dot = __fadd_rn(dot, __fmul_rn(av.y, bv.y));
        dot = __fadd_rn(dot, __fmul_rn(av.z, bv.z));
        dot = __fadd_rn(dot, __fmul_rn(av.w, bv.w));
    }
    return dot;
}

__device__ __forceinline__ float ds4_dot512_float4_plain(
        const float *a,
        const float *b) {
    const float4 *a4 = (const float4 *)a;
    const float4 *b4 = (const float4 *)b;
    float dot = 0.0f;
#pragma unroll 1
    for (uint32_t i = 0; i < 128u; i++) {
        const float4 av = a4[i];
        const float4 bv = b4[i];
        dot += av.x * bv.x;
        dot += av.y * bv.y;
        dot += av.z * bv.z;
        dot += av.w * bv.w;
    }
    return dot;
}

__global__ static void attention_decode_score_split_scores_vec4_kernel(
        float *score_out,
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
        uint32_t S) {
    const uint32_t h = blockIdx.y;
    const uint32_t j = blockIdx.z;
    if (h >= n_head || j >= S) return;
    const uint32_t head_dim = 512u;
    const bool single_all = (ratio == 0u);
    const uint32_t qpos = pos0;
    const uint32_t first_raw_pos = pos0 + 1u - n_raw;
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
    const uint32_t n_score = raw_count + visible_comp;
    if (n_score == 0u) return;

    const uint32_t qbase = n_score / S;
    const uint32_t rem = n_score % S;
    const uint32_t g0 = j * qbase + (j < rem ? j : rem);
    const uint32_t cnt = qbase + (j < rem ? 1u : 0u);
    const uint32_t g1 = g0 + cnt;
    const float *qh = q + (uint64_t)h * head_dim;
    float *row_scores = score_out + (uint64_t)h * n_score;
    const float scale = rsqrtf((float)head_dim);

    for (uint32_t g = g0 + threadIdx.x; g < g1; g += blockDim.x) {
        float s = -INFINITY;
        if (g < raw_count) {
            const uint32_t raw_row =
                (raw_start + raw_first_idx + g) % raw_cap;
            const float *kvrow = raw_kv + (uint64_t)raw_row * head_dim;
            const float dot = ds4_dot512_float4_ordered(qh, kvrow);
            s = dot * scale;
        } else {
            const uint32_t cidx = g - raw_count;
            const float add = use_comp_mask ? comp_mask[(uint64_t)cidx] : 0.0f;
            if (add > -1.0e20f) {
                const float *kvrow = comp_kv + (uint64_t)cidx * head_dim;
                const float dot = ds4_dot512_float4_ordered(qh, kvrow);
                s = dot * scale + add;
            }
        }
        row_scores[g] = s;
    }
}

__global__ static void attention_decode_score_split_scores_vec4_plain_kernel(
        float *score_out,
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
        uint32_t S) {
    const uint32_t h = blockIdx.y;
    const uint32_t j = blockIdx.z;
    if (h >= n_head || j >= S) return;
    const uint32_t head_dim = 512u;
    const bool single_all = (ratio == 0u);
    const uint32_t qpos = pos0;
    const uint32_t first_raw_pos = pos0 + 1u - n_raw;
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
    const uint32_t n_score = raw_count + visible_comp;
    if (n_score == 0u) return;

    const uint32_t qbase = n_score / S;
    const uint32_t rem = n_score % S;
    const uint32_t g0 = j * qbase + (j < rem ? j : rem);
    const uint32_t cnt = qbase + (j < rem ? 1u : 0u);
    const uint32_t g1 = g0 + cnt;
    const float *qh = q + (uint64_t)h * head_dim;
    float *row_scores = score_out + (uint64_t)h * n_score;
    const float scale = rsqrtf((float)head_dim);

    for (uint32_t g = g0 + threadIdx.x; g < g1; g += blockDim.x) {
        float s = -INFINITY;
        if (g < raw_count) {
            const uint32_t raw_row =
                (raw_start + raw_first_idx + g) % raw_cap;
            const float *kvrow = raw_kv + (uint64_t)raw_row * head_dim;
            const float dot = ds4_dot512_float4_plain(qh, kvrow);
            s = dot * scale;
        } else {
            const uint32_t cidx = g - raw_count;
            const float add = use_comp_mask ? comp_mask[(uint64_t)cidx] : 0.0f;
            if (add > -1.0e20f) {
                const float *kvrow = comp_kv + (uint64_t)cidx * head_dim;
                const float dot = ds4_dot512_float4_plain(qh, kvrow);
                s = dot * scale + add;
            }
        }
        row_scores[g] = s;
    }
}

__global__ static void attention_decode_score_split_finalize_kernel(
        float *heads,
        const float *sinks,
        const float *score_in,
        const float *raw_kv,
        const float *comp_kv,
        uint32_t pos0,
        uint32_t n_raw,
        uint32_t raw_cap,
        uint32_t raw_start,
        uint32_t n_comp,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_head,
        uint32_t head_dim) {
    const uint32_t h = blockIdx.y;
    if (h >= n_head) return;
    const bool single_all = (ratio == 0u);
    const uint32_t qpos = pos0;
    const uint32_t first_raw_pos = pos0 + 1u - n_raw;
    uint32_t visible_comp = single_all ? n_comp : (n_comp ? (qpos + 1u) / ratio : 0u);
    if (visible_comp > n_comp) visible_comp = n_comp;

    __shared__ float scores[DS4_CUDA_ATTENTION_SCORE_CAP];
    __shared__ uint32_t raw_rows[256];
    __shared__ float partial[256];
    __shared__ float max_s;
    __shared__ float denom;
    __shared__ uint32_t raw_count_s;
    __shared__ uint32_t raw_first_idx_s;

    const uint32_t score_threads = blockDim.x > 256u ? 256u : blockDim.x;
    const bool score_thread = threadIdx.x < score_threads;
    if (threadIdx.x == 0) {
        raw_count_s = 0;
        raw_first_idx_s = 0;
        if (n_raw != 0) {
            const uint32_t raw_last_pos = first_raw_pos + n_raw - 1u;
            if (single_all) {
                raw_count_s = n_raw > 256u ? 256u : n_raw;
            } else if (qpos >= first_raw_pos) {
                uint32_t lo = first_raw_pos;
                if (window != 0 && qpos + 1u > window) {
                    const uint32_t wlo = qpos + 1u - window;
                    if (wlo > lo) lo = wlo;
                }
                const uint32_t hi = qpos < raw_last_pos ? qpos : raw_last_pos;
                if (hi >= lo) {
                    raw_first_idx_s = lo - first_raw_pos;
                    raw_count_s = hi - lo + 1u;
                    if (raw_count_s > 256u) raw_count_s = 256u;
                }
            }
        }
    }
    __syncthreads();
    const uint32_t raw_count = raw_count_s;
    const uint32_t raw_first_idx = raw_first_idx_s;
    if (score_thread) {
        for (uint32_t r = threadIdx.x; r < raw_count; r += score_threads) {
            raw_rows[r] = (raw_start + raw_first_idx + r) % raw_cap;
        }
    }
    __syncthreads();
    const uint32_t n_score = raw_count + visible_comp;
    const float *row_scores = score_in + (uint64_t)h * n_score;
    float local_max = sinks[h];
    if (score_thread) {
        for (uint32_t i = threadIdx.x; i < n_score; i += score_threads) {
            const float s = row_scores[i];
            scores[i] = s;
            local_max = fmaxf(local_max, s);
        }
    }
    if (score_thread) partial[threadIdx.x] = local_max;
    __syncthreads();
    for (uint32_t stride = score_threads >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] =
                fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) max_s = partial[0];
    __syncthreads();
    float den_local = 0.0f;
    if (score_thread) {
        for (uint32_t i = threadIdx.x; i < n_score; i += score_threads) {
            scores[i] = expf(scores[i] - max_s);
            den_local += scores[i];
        }
    }
    if (score_thread) partial[threadIdx.x] = den_local;
    __syncthreads();
    for (uint32_t stride = score_threads >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) denom = partial[0] + expf(sinks[h] - max_s);
    __syncthreads();
    float *oh = heads + (uint64_t)h * head_dim;
    if (head_dim == 512u && blockDim.x >= 512u) {
        const uint32_t d = threadIdx.x;
        float acc = 0.0f;
        for (uint32_t r = 0; r < raw_count; r++) {
            const float s = scores[r];
            const float *kv = raw_kv + (uint64_t)raw_rows[r] * head_dim;
            acc += kv[d] * s;
        }
        for (uint32_t c = 0; c < visible_comp; c++) {
            const float s = scores[raw_count + c];
            const float *kv = comp_kv + (uint64_t)c * head_dim;
            acc += kv[d] * s;
        }
        oh[d] = acc / denom;
    } else if (head_dim == 512u && blockDim.x == 256u) {
        const uint32_t d0 = threadIdx.x;
        const uint32_t d1 = d0 + 256u;
        float acc0 = 0.0f;
        float acc1 = 0.0f;
        for (uint32_t r = 0; r < raw_count; r++) {
            const float s = scores[r];
            const float *kv = raw_kv + (uint64_t)raw_rows[r] * head_dim;
            acc0 += kv[d0] * s;
            acc1 += kv[d1] * s;
        }
        for (uint32_t c = 0; c < visible_comp; c++) {
            const float s = scores[raw_count + c];
            const float *kv = comp_kv + (uint64_t)c * head_dim;
            acc0 += kv[d0] * s;
            acc1 += kv[d1] * s;
        }
        oh[d0] = acc0 / denom;
        oh[d1] = acc1 / denom;
    } else {
        for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
            float acc = 0.0f;
            for (uint32_t r = 0; r < raw_count; r++) {
                acc += raw_kv[(uint64_t)raw_rows[r] * head_dim + d] * scores[r];
            }
            for (uint32_t c = 0; c < visible_comp; c++) {
                acc += comp_kv[(uint64_t)c * head_dim + d] * scores[raw_count + c];
            }
            oh[d] = acc / denom;
        }
    }
}

__global__ static void attention_decode_score_split_finalize_rows_kernel(
        float *heads,
        const float *sinks,
        const float *score_in,
        cuda_attention_decode_row_table rows,
        uint32_t n_rows,
        uint32_t score_stride,
        uint32_t n_head,
        uint32_t head_dim) {
    const uint32_t row = blockIdx.x;
    const uint32_t h = blockIdx.y;
    if (row >= n_rows || h >= n_head) return;
    const ds4_gpu_attention_decode_row dsc = rows.row[row];
    if (dsc.indexed) return;
    const float *raw_kv = (const float *)(uintptr_t)dsc.raw_kv;
    const float *comp_kv = (const float *)(uintptr_t)dsc.comp_kv;
    const bool single_all = dsc.ratio == 0u;
    const uint32_t qpos = dsc.pos;
    const uint32_t first_raw_pos = dsc.pos + 1u - dsc.n_raw;
    uint32_t visible_comp = single_all
        ? dsc.n_comp
        : (dsc.n_comp ? (qpos + 1u) / dsc.ratio : 0u);
    if (visible_comp > dsc.n_comp) visible_comp = dsc.n_comp;

    __shared__ float scores[DS4_CUDA_ATTENTION_SCORE_CAP];
    __shared__ uint32_t raw_rows[256];
    __shared__ float partial[256];
    __shared__ float max_s;
    __shared__ float denom;
    __shared__ uint32_t raw_count_s;
    __shared__ uint32_t raw_first_idx_s;

    const uint32_t score_threads = blockDim.x > 256u ? 256u : blockDim.x;
    const bool score_thread = threadIdx.x < score_threads;
    if (threadIdx.x == 0u) {
        raw_count_s = 0u;
        raw_first_idx_s = 0u;
        if (dsc.n_raw != 0u) {
            const uint32_t raw_last_pos = first_raw_pos + dsc.n_raw - 1u;
            if (single_all) {
                raw_count_s = dsc.n_raw > 256u ? 256u : dsc.n_raw;
            } else if (qpos >= first_raw_pos) {
                uint32_t lo = first_raw_pos;
                if (dsc.window != 0u && qpos + 1u > dsc.window) {
                    const uint32_t wlo = qpos + 1u - dsc.window;
                    if (wlo > lo) lo = wlo;
                }
                const uint32_t hi = qpos < raw_last_pos ? qpos : raw_last_pos;
                if (hi >= lo) {
                    raw_first_idx_s = lo - first_raw_pos;
                    raw_count_s = hi - lo + 1u;
                    if (raw_count_s > 256u) raw_count_s = 256u;
                }
            }
        }
    }
    __syncthreads();
    const uint32_t raw_count = raw_count_s;
    const uint32_t raw_first_idx = raw_first_idx_s;
    if (score_thread) {
        for (uint32_t r = threadIdx.x; r < raw_count; r += score_threads) {
            raw_rows[r] =
                (dsc.raw_start + raw_first_idx + r) % dsc.raw_cap;
        }
    }
    __syncthreads();
    const uint32_t n_score = raw_count + visible_comp;
    const float *row_scores = score_in +
        ((uint64_t)row * n_head + h) * score_stride;
    float local_max = sinks[h];
    if (score_thread) {
        for (uint32_t i = threadIdx.x; i < n_score; i += score_threads) {
            const float s = row_scores[i];
            scores[i] = s;
            local_max = fmaxf(local_max, s);
        }
        partial[threadIdx.x] = local_max;
    }
    __syncthreads();
    for (uint32_t stride = score_threads >> 1u;
         stride > 0u;
         stride >>= 1u) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] =
                fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0u) max_s = partial[0];
    __syncthreads();
    float den_local = 0.0f;
    if (score_thread) {
        for (uint32_t i = threadIdx.x; i < n_score; i += score_threads) {
            scores[i] = expf(scores[i] - max_s);
            den_local += scores[i];
        }
        partial[threadIdx.x] = den_local;
    }
    __syncthreads();
    for (uint32_t stride = score_threads >> 1u;
         stride > 0u;
         stride >>= 1u) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0u) {
        denom = partial[0] + expf(sinks[h] - max_s);
    }
    __syncthreads();

    float *oh = heads + ((uint64_t)row * n_head + h) * head_dim;
    if (head_dim == 512u && blockDim.x >= 512u) {
        const uint32_t dim = threadIdx.x;
        float acc = 0.0f;
        for (uint32_t r = 0; r < raw_count; r++) {
            const float *kv = raw_kv + (uint64_t)raw_rows[r] * head_dim;
            acc += kv[dim] * scores[r];
        }
        for (uint32_t c = 0; c < visible_comp; c++) {
            const float *kv = comp_kv + (uint64_t)c * head_dim;
            acc += kv[dim] * scores[raw_count + c];
        }
        oh[dim] = acc / denom;
    } else {
        for (uint32_t dim = threadIdx.x;
             dim < head_dim;
             dim += blockDim.x) {
            float acc = 0.0f;
            for (uint32_t r = 0; r < raw_count; r++) {
                acc += raw_kv[(uint64_t)raw_rows[r] * head_dim + dim] *
                       scores[r];
            }
            for (uint32_t c = 0; c < visible_comp; c++) {
                acc += comp_kv[(uint64_t)c * head_dim + dim] *
                       scores[raw_count + c];
            }
            oh[dim] = acc / denom;
        }
    }
}

__global__ static void attention_decode_score_split_finalize_dim2_kernel(
        float *heads,
        const float *sinks,
        const float *score_in,
        const float *raw_kv,
        const float *comp_kv,
        uint32_t pos0,
        uint32_t n_raw,
        uint32_t raw_cap,
        uint32_t raw_start,
        uint32_t n_comp,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_head,
        uint32_t head_dim) {
    const uint32_t dim_half = blockIdx.x;
    const uint32_t h = blockIdx.y;
    if (h >= n_head || head_dim != 512u || dim_half >= 2u) return;
    const bool single_all = (ratio == 0u);
    const uint32_t qpos = pos0;
    const uint32_t first_raw_pos = pos0 + 1u - n_raw;
    uint32_t visible_comp = single_all ? n_comp : (n_comp ? (qpos + 1u) / ratio : 0u);
    if (visible_comp > n_comp) visible_comp = n_comp;

    __shared__ float scores[DS4_CUDA_ATTENTION_SCORE_CAP];
    __shared__ uint32_t raw_rows[256];
    __shared__ float partial[256];
    __shared__ float max_s;
    __shared__ float denom;
    __shared__ uint32_t raw_count_s;
    __shared__ uint32_t raw_first_idx_s;

    const uint32_t score_threads = 256u;
    if (threadIdx.x == 0) {
        raw_count_s = 0;
        raw_first_idx_s = 0;
        if (n_raw != 0) {
            const uint32_t raw_last_pos = first_raw_pos + n_raw - 1u;
            if (single_all) {
                raw_count_s = n_raw > 256u ? 256u : n_raw;
            } else if (qpos >= first_raw_pos) {
                uint32_t lo = first_raw_pos;
                if (window != 0 && qpos + 1u > window) {
                    const uint32_t wlo = qpos + 1u - window;
                    if (wlo > lo) lo = wlo;
                }
                const uint32_t hi = qpos < raw_last_pos ? qpos : raw_last_pos;
                if (hi >= lo) {
                    raw_first_idx_s = lo - first_raw_pos;
                    raw_count_s = hi - lo + 1u;
                    if (raw_count_s > 256u) raw_count_s = 256u;
                }
            }
        }
    }
    __syncthreads();
    const uint32_t raw_count = raw_count_s;
    const uint32_t raw_first_idx = raw_first_idx_s;
    for (uint32_t r = threadIdx.x; r < raw_count; r += score_threads) {
        raw_rows[r] = (raw_start + raw_first_idx + r) % raw_cap;
    }
    __syncthreads();

    const uint32_t n_score = raw_count + visible_comp;
    const float *row_scores = score_in + (uint64_t)h * n_score;
    float local_max = sinks[h];
    for (uint32_t i = threadIdx.x; i < n_score; i += score_threads) {
        const float s = row_scores[i];
        scores[i] = s;
        local_max = fmaxf(local_max, s);
    }
    partial[threadIdx.x] = local_max;
    __syncthreads();
    for (uint32_t stride = score_threads >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] =
                fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) max_s = partial[0];
    __syncthreads();

    float den_local = 0.0f;
    for (uint32_t i = threadIdx.x; i < n_score; i += score_threads) {
        scores[i] = expf(scores[i] - max_s);
        den_local += scores[i];
    }
    partial[threadIdx.x] = den_local;
    __syncthreads();
    for (uint32_t stride = score_threads >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) denom = partial[0] + expf(sinks[h] - max_s);
    __syncthreads();

    const uint32_t d = dim_half * 256u + threadIdx.x;
    float acc = 0.0f;
    for (uint32_t r = 0; r < raw_count; r++) {
        const float s = scores[r];
        const float *kv = raw_kv + (uint64_t)raw_rows[r] * head_dim;
        acc += kv[d] * s;
    }
    for (uint32_t c = 0; c < visible_comp; c++) {
        const float s = scores[raw_count + c];
        const float *kv = comp_kv + (uint64_t)c * head_dim;
        acc += kv[d] * s;
    }
    heads[(uint64_t)h * head_dim + d] = acc / denom;
}

__global__ static void attention_decode_global_softmax_kernel(
        float *score_inout,
        float *denom_out,
        const float *sinks,
        uint32_t n_score,
        uint32_t n_head) {
    const uint32_t h = blockIdx.x;
    if (h >= n_head || n_score == 0u || n_score > DS4_CUDA_ATTENTION_SCORE_CAP) return;
    __shared__ float scores[DS4_CUDA_ATTENTION_SCORE_CAP];
    __shared__ float partial[256];
    __shared__ float max_s;
    __shared__ float denom_s;
    const uint32_t score_threads = blockDim.x > 256u ? 256u : blockDim.x;
    const bool score_thread = threadIdx.x < score_threads;
    float *row_scores = score_inout + (uint64_t)h * n_score;

    float local_max = sinks[h];
    if (score_thread) {
        for (uint32_t i = threadIdx.x; i < n_score; i += score_threads) {
            const float s = row_scores[i];
            scores[i] = s;
            local_max = fmaxf(local_max, s);
        }
        partial[threadIdx.x] = local_max;
    }
    __syncthreads();
    for (uint32_t stride = score_threads >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] =
                fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) max_s = partial[0];
    __syncthreads();

    float den_local = 0.0f;
    if (score_thread) {
        for (uint32_t i = threadIdx.x; i < n_score; i += score_threads) {
            const float e = expf(scores[i] - max_s);
            scores[i] = e;
            den_local += e;
        }
        partial[threadIdx.x] = den_local;
    }
    __syncthreads();
    for (uint32_t stride = score_threads >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) denom_s = partial[0] + expf(sinks[h] - max_s);
    __syncthreads();

    if (score_thread) {
        for (uint32_t i = threadIdx.x; i < n_score; i += score_threads) {
            row_scores[i] = scores[i];
        }
    }
    if (threadIdx.x == 0) denom_out[h] = denom_s;
}

__global__ static void attention_decode_split_value_kernel(
        float *partials,
        const float *score_exp,
        const float *raw_kv,
        const float *comp_kv,
        uint32_t raw_count,
        uint32_t raw_first_idx,
        uint32_t raw_cap,
        uint32_t raw_start,
        uint32_t n_score,
        uint32_t n_head,
        uint32_t head_dim,
        uint32_t S) {
    const uint32_t h = blockIdx.y;
    const uint32_t j = blockIdx.z;
    if (h >= n_head || j >= S || n_score == 0u) return;
    const uint32_t qbase = n_score / S;
    const uint32_t rem = n_score % S;
    const uint32_t g0 = j * qbase + (j < rem ? j : rem);
    const uint32_t cnt = qbase + (j < rem ? 1u : 0u);
    const uint32_t g1 = g0 + cnt;
    const float *row_scores = score_exp + (uint64_t)h * n_score;
    float *pout = partials + ((uint64_t)h * S + j) * head_dim;
    for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (uint32_t g = g0; g < g1; g++) {
            const float s = row_scores[g];
            if (g < raw_count) {
                const uint32_t raw_row =
                    (raw_start + raw_first_idx + g) % raw_cap;
                acc += raw_kv[(uint64_t)raw_row * head_dim + d] * s;
            } else {
                const uint32_t c = g - raw_count;
                acc += comp_kv[(uint64_t)c * head_dim + d] * s;
            }
        }
        pout[d] = acc;
    }
}

__global__ static void attention_decode_split_value_combine_kernel(
        float *heads,
        const float *partials,
        const float *denom,
        uint32_t n_head,
        uint32_t head_dim,
        uint32_t S) {
    const uint32_t h = blockIdx.y;
    if (h >= n_head) return;
    const float *base = partials + (uint64_t)h * S * head_dim;
    const float den = denom[h];
    float *oh = heads + (uint64_t)h * head_dim;
    for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (uint32_t j = 0; j < S; j++) {
            acc += base[(uint64_t)j * head_dim + d];
        }
        oh[d] = acc / den;
    }
}

typedef struct {
    uint32_t n_rot;
    uint32_t pos0;
    uint32_t n_ctx_orig;
    float    freq_base;
    float    freq_scale;
    float    ext_factor;
    float    attn_factor;
    float    beta_fast;
    float    beta_slow;
} cuda_attention_inv_rope_params;

static void attention_decode_score_split_graph_destroy_one(int logical_tier) {
    if (logical_tier < 0 || logical_tier >= DS4_MAX_GPUS) return;
    cuda_score_split_graph_cache *c = &g_score_split_graph[logical_tier];
    if (c->exec) (void)cudaGraphExecDestroy(c->exec);
    if (c->graph) (void)cudaGraphDestroy(c->graph);
    memset(c, 0, sizeof(*c));
}

static int attention_decode_score_split_graph_launch(
        int logical_tier,
        float *heads,
        const float *sinks,
        float *scores,
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
        uint32_t head_dim,
        uint32_t final_threads,
        uint32_t S,
        const cuda_attention_inv_rope_params *inv_rope) {
    if (logical_tier < 0 || logical_tier >= DS4_MAX_GPUS) return 0;
    cuda_score_split_graph_cache *c = &g_score_split_graph[logical_tier];
    const bool graph_inv_rope =
        inv_rope &&
        head_dim == 512u &&
        inv_rope->n_rot != 0u &&
        inv_rope->n_rot <= head_dim &&
        (inv_rope->n_rot & 1u) == 0u;
    const bool shape_match =
        c->valid &&
        c->n_head == n_head &&
        c->head_dim == head_dim &&
        c->S == S &&
        c->final_threads == final_threads &&
        c->fuses_inv_rope == (graph_inv_rope ? 1 : 0) &&
        (!graph_inv_rope || c->n_rot == inv_rope->n_rot);
    if (c->valid && !shape_match) {
        attention_decode_score_split_graph_destroy_one(logical_tier);
        c = &g_score_split_graph[logical_tier];
    }

    dim3 score_grid(1, n_head, S);
    dim3 final_grid(1, n_head, 1);
    dim3 score_block(256, 1, 1);
    dim3 final_block(final_threads, 1, 1);

    void *score_args[] = {
        &scores, &q, &raw_kv, &comp_kv, &comp_mask, &use_comp_mask,
        &pos0, &n_raw, &raw_cap, &raw_start, &n_comp, &window, &ratio,
        &n_head, &head_dim, &S
    };
    cudaKernelNodeParams score_params;
    memset(&score_params, 0, sizeof(score_params));
    score_params.func = (void *)attention_decode_score_split_scores_kernel;
    score_params.gridDim = score_grid;
    score_params.blockDim = score_block;
    score_params.sharedMemBytes = 0;
    score_params.kernelParams = score_args;
    score_params.extra = NULL;

    void *final_args[] = {
        &heads, &sinks, &scores, &raw_kv, &comp_kv, &pos0, &n_raw,
        &raw_cap, &raw_start, &n_comp, &window, &ratio, &n_head, &head_dim
    };
    cudaKernelNodeParams final_params;
    memset(&final_params, 0, sizeof(final_params));
    final_params.func = (void *)attention_decode_score_split_finalize_kernel;
    final_params.gridDim = final_grid;
    final_params.blockDim = final_block;
    final_params.sharedMemBytes = 0;
    final_params.kernelParams = final_args;
    final_params.extra = NULL;

    uint32_t rope_n_tok = 1u;
    uint32_t rope_pos_stride = 1u;
    int rope_inverse = 1;
    uint32_t rope_n_rot = graph_inv_rope ? inv_rope->n_rot : 0u;
    uint32_t rope_pos0 = graph_inv_rope ? inv_rope->pos0 : 0u;
    uint32_t rope_n_ctx_orig = graph_inv_rope ? inv_rope->n_ctx_orig : 0u;
    float rope_freq_base = graph_inv_rope ? inv_rope->freq_base : 0.0f;
    float rope_freq_scale = graph_inv_rope ? inv_rope->freq_scale : 0.0f;
    float rope_ext_factor = graph_inv_rope ? inv_rope->ext_factor : 0.0f;
    float rope_attn_factor = graph_inv_rope ? inv_rope->attn_factor : 0.0f;
    float rope_beta_fast = graph_inv_rope ? inv_rope->beta_fast : 0.0f;
    float rope_beta_slow = graph_inv_rope ? inv_rope->beta_slow : 0.0f;
    void *rope_args[] = {
        &heads, &rope_n_tok, &n_head, &head_dim, &rope_n_rot,
        &rope_pos0, &rope_pos_stride, &rope_n_ctx_orig, &rope_inverse,
        &rope_freq_base, &rope_freq_scale, &rope_ext_factor,
        &rope_attn_factor, &rope_beta_fast, &rope_beta_slow
    };
    cudaKernelNodeParams rope_params;
    memset(&rope_params, 0, sizeof(rope_params));
    if (graph_inv_rope) {
        const uint32_t pairs = n_head * (rope_n_rot / 2u);
        rope_params.func = (void *)rope_tail_kernel;
        rope_params.gridDim = dim3((pairs + 255u) / 256u, 1, 1);
        rope_params.blockDim = dim3(256, 1, 1);
        rope_params.sharedMemBytes = 0;
        rope_params.kernelParams = rope_args;
        rope_params.extra = NULL;
    }

    if (!c->valid) {
        cudaError_t err = cudaGraphCreate(&c->graph, 0);
        if (err != cudaSuccess) {
            fprintf(stderr, "ds4: attention score-split graph create failed: %s\n",
                    cudaGetErrorString(err));
            attention_decode_score_split_graph_destroy_one(logical_tier);
            return -1;
        }
        err = cudaGraphAddKernelNode(&c->score_node, c->graph, NULL, 0,
                                     &score_params);
        if (err == cudaSuccess) {
            err = cudaGraphAddKernelNode(&c->final_node, c->graph,
                                         &c->score_node, 1, &final_params);
        }
        if (err == cudaSuccess && graph_inv_rope) {
            err = cudaGraphAddKernelNode(&c->rope_node, c->graph,
                                         &c->final_node, 1, &rope_params);
        }
        if (err == cudaSuccess) {
            err = cudaGraphInstantiate(&c->exec, c->graph, NULL, NULL, 0);
        }
        if (err != cudaSuccess) {
            fprintf(stderr,
                    "ds4: attention score-split graph instantiate failed: %s\n",
                    cudaGetErrorString(err));
            attention_decode_score_split_graph_destroy_one(logical_tier);
            return -1;
        }
        c->n_head = n_head;
        c->head_dim = head_dim;
        c->S = S;
        c->final_threads = final_threads;
        c->n_rot = graph_inv_rope ? inv_rope->n_rot : 0u;
        c->fuses_inv_rope = graph_inv_rope ? 1 : 0;
        c->valid = 1;
    } else {
        cudaError_t err =
            cudaGraphExecKernelNodeSetParams(c->exec, c->score_node,
                                             &score_params);
        if (err == cudaSuccess) {
            err = cudaGraphExecKernelNodeSetParams(c->exec, c->final_node,
                                                   &final_params);
        }
        if (err == cudaSuccess && graph_inv_rope) {
            err = cudaGraphExecKernelNodeSetParams(c->exec, c->rope_node,
                                                   &rope_params);
        }
        if (err != cudaSuccess) {
            fprintf(stderr,
                    "ds4: attention score-split graph update failed: %s\n",
                    cudaGetErrorString(err));
            attention_decode_score_split_graph_destroy_one(logical_tier);
            return -1;
        }
    }

    cudaError_t err = cudaGraphLaunch(c->exec, 0);
    if (err != cudaSuccess) {
        fprintf(stderr, "ds4: attention score-split graph launch failed: %s\n",
                cudaGetErrorString(err));
        attention_decode_score_split_graph_destroy_one(logical_tier);
        return -1;
    }
    return 1;
}

static int attention_decode_score_split_launch(
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
        uint32_t head_dim,
        uint32_t final_threads,
        const cuda_attention_inv_rope_params *inv_rope) {
    if (cuda_env_flag_enabled("DS4_CUDA_NO_EXACT_SCORE_SPLIT_DECODE", 0)) return 0;
    const int explicit_exact =
        cuda_env_flag_enabled("DS4_CUDA_EXACT_SCORE_SPLIT_DECODE", 0);
    if (!cuda_env_flag_enabled("DS4_CUDA_EXACT_SCORE_SPLIT_DECODE", 1)) return 0;
    if (!explicit_exact && cuda_splitkv_decode_requested()) return 0;
    if (g_cuda_decode_score4 || g_cuda_decode_score8) return 0;
    if (head_dim == 0u || n_head == 0u) return 0;
    const bool single_all = (ratio == 0u);
    const uint32_t qpos = pos0;
    const uint32_t first_raw_pos = pos0 + 1u - n_raw;
    uint32_t visible_comp = single_all ? n_comp : (n_comp ? (qpos + 1u) / ratio : 0u);
    if (visible_comp > n_comp) visible_comp = n_comp;
    uint32_t raw_count = 0;
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
                raw_count = hi - lo + 1u;
                if (raw_count > 256u) raw_count = 256u;
            }
        }
    }
    const uint32_t n_score = raw_count + visible_comp;
    if (n_score == 0u || n_score > DS4_CUDA_ATTENTION_SCORE_CAP) return 0;
    /* With the head-tiled score kernel the exact score-split path beats the
     * one-block mixed kernel even for short score counts, so the gate that
     * used to protect short contexts (512) now defaults to 1. */
    const uint32_t min_score = cuda_parse_u32_env_clamped(
            "DS4_CUDA_EXACT_SCORE_SPLIT_MIN_SCORE", 1u, 0u,
            DS4_CUDA_ATTENTION_SCORE_CAP, NULL);
    if (n_score < min_score) return 0;
    uint32_t chunk = cuda_parse_u32_env_clamped(
            "DS4_CUDA_EXACT_SCORE_SPLIT_CHUNK", DS4_CUDA_SPLITKV_CHUNK,
            1u, DS4_CUDA_ATTENTION_SCORE_CAP, NULL);
    uint32_t s_floor = cuda_parse_u32_env_clamped(
            "DS4_CUDA_EXACT_SCORE_SPLIT_S_FLOOR", 6u,
            1u, DS4_CUDA_SPLITKV_S_MAX, NULL);
    uint32_t s_max = cuda_parse_u32_env_clamped(
            "DS4_CUDA_EXACT_SCORE_SPLIT_S_MAX", DS4_CUDA_SPLITKV_S_MAX,
            1u, DS4_CUDA_SPLITKV_S_MAX, NULL);
    int exact_present = 0;
    uint32_t S = cuda_parse_u32_env_clamped(
            "DS4_CUDA_EXACT_SCORE_SPLIT_S", 0u, 1u,
            DS4_CUDA_SPLITKV_S_MAX, &exact_present);
    if (!exact_present) {
        S = (n_score + chunk - 1u) / chunk;
        if (S < s_floor) S = s_floor < n_score ? s_floor : n_score;
        if (S > s_max) S = s_max;
    }
    if (S > n_score) S = n_score;
    if (S <= 1u) return 0;
    const bool graph_inv_rope =
        g_cuda_exact_score_split_fuse_inv_rope &&
        inv_rope &&
        head_dim == 512u &&
        final_threads >= 512u &&
        inv_rope->n_rot != 0u &&
        inv_rope->n_rot <= 512u &&
        (inv_rope->n_rot & 1u) == 0u;

    const uint64_t score_count = (uint64_t)n_head * n_score;
    float *scores = (float *)cuda_tmp_alloc_on(logical_tier,
                                               score_count * sizeof(float),
                                               "attention exact score split");
    if (!scores) return 0;
    const bool use_ldg_scores = g_cuda_exact_score_split_ldg;
    const bool use_vec4_plain_scores =
        !use_ldg_scores &&
        g_cuda_exact_score_split_vec4_plain &&
        head_dim == 512u;
    const bool use_vec4_scores =
        !use_ldg_scores &&
        !use_vec4_plain_scores &&
        (g_cuda_exact_score_split_vec4 || g_decode_score_vec4) &&
        head_dim == 512u;
    const bool use_dim2_finalize =
        g_cuda_exact_score_split_dim2 &&
        head_dim == 512u &&
        final_threads >= 512u &&
        !graph_inv_rope;
    if ((g_cuda_exact_score_split_graph || graph_inv_rope) &&
        !use_dim2_finalize &&
        !use_ldg_scores &&
        !use_vec4_plain_scores &&
        !use_vec4_scores)
    {
        int rc = attention_decode_score_split_graph_launch(
            logical_tier, heads, sinks, scores, q, raw_kv, comp_kv, comp_mask,
            use_comp_mask, pos0, n_raw, raw_cap, raw_start, n_comp, window,
            ratio, n_head, head_dim, final_threads, S,
            graph_inv_rope ? inv_rope : NULL);
        if (rc == 1) return 1;
        if (rc < 0) return -1;
    }
    if (graph_inv_rope) return 0;
    static int score_tile_disabled = -1;
    if (score_tile_disabled < 0) {
        score_tile_disabled = getenv("DS4_CUDA_NO_SCORE_TILE") != NULL ? 1 : 0;
    }
    if (!score_tile_disabled &&
        head_dim == 512u &&
        !use_ldg_scores &&
        !use_vec4_plain_scores &&
        !use_vec4_scores) {
        /* cudaFuncSetAttribute() applies to the current device only, so opt in
         * to >48KB dynamic shared memory once per device. */
        static int tile_shmem_ready[DS4_MAX_GPUS] = {0};
        const size_t tile_shmem =
            (size_t)(DS4_SCORE_TILE_HEADS + DS4_SCORE_TILE_ROWS) *
            DS4_SCORE_TILE_STRIDE * sizeof(float);
        int tile_dev = 0;
        cudaGetDevice(&tile_dev);
        if (tile_dev >= 0 && tile_dev < DS4_MAX_GPUS &&
            !tile_shmem_ready[tile_dev]) {
            if (!cuda_ok(cudaFuncSetAttribute(
                             attention_decode_score_split_scores_tile512_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int)tile_shmem),
                         "attention score tile shared-memory opt-in")) {
                score_tile_disabled = 1;
            }
            tile_shmem_ready[tile_dev] = 1;
        }
        if (score_tile_disabled) {
            return 0; /* retry via the generic path on the next call */
        }
        dim3 tile_grid((n_score + DS4_SCORE_TILE_ROWS - 1u) / DS4_SCORE_TILE_ROWS,
                       (n_head + DS4_SCORE_TILE_HEADS - 1u) / DS4_SCORE_TILE_HEADS,
                       1);
        attention_decode_score_split_scores_tile512_kernel<<<tile_grid, 256, tile_shmem>>>(
                scores, q, raw_kv, comp_kv, comp_mask, use_comp_mask,
                pos0, n_raw, raw_cap, raw_start, n_comp, window, ratio,
                n_head, head_dim);
        if (!cuda_ok(cudaGetLastError(), "attention exact score split tile launch")) return -1;
    } else {
    dim3 score_grid(1, n_head, S);
    if (use_ldg_scores) {
        attention_decode_score_split_scores_ldg_kernel<<<score_grid, 256>>>(
                scores, q, raw_kv, comp_kv, comp_mask, use_comp_mask,
                pos0, n_raw, raw_cap, raw_start, n_comp, window, ratio,
                n_head, head_dim, S);
    } else if (use_vec4_plain_scores) {
        attention_decode_score_split_scores_vec4_plain_kernel<<<score_grid, 256>>>(
                scores, q, raw_kv, comp_kv, comp_mask, use_comp_mask,
                pos0, n_raw, raw_cap, raw_start, n_comp, window, ratio,
                n_head, S);
    } else if (use_vec4_scores) {
        attention_decode_score_split_scores_vec4_kernel<<<score_grid, 256>>>(
                scores, q, raw_kv, comp_kv, comp_mask, use_comp_mask,
                pos0, n_raw, raw_cap, raw_start, n_comp, window, ratio,
                n_head, S);
    } else {
        attention_decode_score_split_scores_kernel<<<score_grid, 256>>>(
                scores, q, raw_kv, comp_kv, comp_mask, use_comp_mask,
                pos0, n_raw, raw_cap, raw_start, n_comp, window, ratio,
                n_head, head_dim, S);
    }
    if (!cuda_ok(cudaGetLastError(), "attention exact score split scores launch")) return -1;
    }
    if (use_dim2_finalize) {
        dim3 final_grid(2, n_head, 1);
        attention_decode_score_split_finalize_dim2_kernel<<<final_grid, 256>>>(
                heads, sinks, scores, raw_kv, comp_kv,
                pos0, n_raw, raw_cap, raw_start, n_comp, window, ratio,
                n_head, head_dim);
        if (!cuda_ok(cudaGetLastError(), "attention exact score split dim2 finalize launch")) return -1;
    } else {
        dim3 final_grid(1, n_head, 1);
        attention_decode_score_split_finalize_kernel<<<final_grid, final_threads>>>(
                heads, sinks, scores, raw_kv, comp_kv,
                pos0, n_raw, raw_cap, raw_start, n_comp, window, ratio,
                n_head, head_dim);
        if (!cuda_ok(cudaGetLastError(), "attention exact score split finalize launch")) return -1;
    }
    return 1;
}

/* ---- perf-02 split-KV / flash-decode (opt-in, default OFF) ----------------
 *
 * attention_decode_splitkv_kernel computes a partial online-softmax over a
 * contiguous chunk of the flattened logical row set [0, n_score) used by
 * attention_decode_mixed_kernel (raw rows first, then compressed rows, same
 * ascending ordering). Each block handles (t = blockIdx.x, h = blockIdx.y,
 * chunk = blockIdx.z) and writes a partial (m_j, l_j, acc_j[head_dim]) WITHOUT
 * the sink term. attention_decode_splitkv_combine_kernel merges the S partials
 * per (t,h), folds the sink once, and writes the final normalized head output.
 *
 * The math is the standard flash-attention online-softmax rescale and is
 * algebraically identical to attention_decode_mixed_kernel; it is NOT
 * guaranteed bit-identical in FP32 (different expf inputs + add/mul grouping),
 * hence default-OFF behind DS4_CUDA_SPLITKV_DECODE and the S==1 dispatch to the
 * old kernel as the bit-exact anchor (handled in the launch helper).
 *
 * Partials scratch layout (per logical tier), contiguous floats:
 *   stride = head_dim + 2
 *   base(t,h,j) = ((t*n_head + h)*S + j) * stride
 *     [0]               = m_j   (chunk running max; -INF if empty/all-masked)
 *     [1]               = l_j   (chunk denominator sum exp(s - m_j))
 *     [2 .. 2+head_dim) = acc_j[head_dim] (chunk weighted value sum)
 */
__global__ static void attention_decode_splitkv_kernel(
        float *partials,
        const float *q,
        const float *raw_kv,
        const float *comp_kv,
        const float *comp_mask,
        uint32_t use_comp_mask,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t n_raw,
        uint32_t raw_cap,
        uint32_t raw_start,
        uint32_t n_comp,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_head,
        uint32_t head_dim,
        uint32_t S) {
    uint32_t t = blockIdx.x;
    uint32_t h = blockIdx.y;
    uint32_t j = blockIdx.z;
    if (t >= n_tokens || h >= n_head || j >= S) return;
    const bool single_all = (n_tokens == 1u && ratio == 0u);
    uint32_t qpos = pos0 + t;
    uint32_t first_raw_pos = pos0 + n_tokens - n_raw;
    uint32_t visible_comp = single_all ? n_comp : (n_comp ? (qpos + 1u) / ratio : 0u);
    if (visible_comp > n_comp) visible_comp = n_comp;
    const float *qh = q + ((uint64_t)t * n_head + h) * head_dim;
    /* scores buffer holds only this chunk's rows. The launch helper guarantees
     * cnt <= DS4_CUDA_SPLITKV_SCORE_CAP, including env-tuned split counts. */
    __shared__ float scores[DS4_CUDA_SPLITKV_SCORE_CAP];
    __shared__ uint32_t raw_rows[256];
    __shared__ float partial[256];
    __shared__ float m_s;
    __shared__ float l_s;
    __shared__ uint32_t raw_count;
    __shared__ uint32_t raw_first_idx;
    float scale = rsqrtf((float)head_dim);
    if (threadIdx.x == 0) {
        raw_count = 0;
        raw_first_idx = 0;
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
    }
    __syncthreads();
    uint32_t n_score = raw_count + visible_comp;
    /* even split of [0, n_score) across S chunks: first (n_score % S) chunks
     * get base+1, identical deterministic partition for every block. */
    uint32_t qbase = n_score / S;
    uint32_t rem = n_score % S;
    uint32_t g0 = j * qbase + (j < rem ? j : rem);
    uint32_t cnt = qbase + (j < rem ? 1u : 0u);
    uint32_t g1 = g0 + cnt;          /* exclusive end of this chunk */
    /* Map raw rows that fall in this chunk into shared raw_rows[]. The chunk's
     * raw portion is [raw_lo, raw_hi). cnt <= CHUNK and raw rows <= 256, so
     * the slice fits raw_rows[256]. */
    uint32_t raw_lo = g0 < raw_count ? g0 : raw_count;
    uint32_t raw_hi = g1 < raw_count ? g1 : raw_count;
    for (uint32_t r = raw_lo + threadIdx.x; r < raw_hi; r += blockDim.x) {
        raw_rows[r - raw_lo] = (raw_start + raw_first_idx + r) % raw_cap;
    }
    __syncthreads();
    float *pout = partials + (((uint64_t)t * n_head + h) * S + j) * (head_dim + 2u);
    /* Pass 1: scores for this chunk's rows into shared scores[0..cnt). */
    float local_max = -INFINITY;
    for (uint32_t i = threadIdx.x; i < cnt; i += blockDim.x) {
        uint32_t g = g0 + i;
        float s;
        if (g < raw_count) {
            const float *kvrow = raw_kv + (uint64_t)raw_rows[g - raw_lo] * head_dim;
            float dot = 0.0f;
            for (uint32_t d = 0; d < head_dim; d++) dot += qh[d] * kvrow[d];
            s = dot * scale;
        } else {
            uint32_t c = g - raw_count;
            float add = use_comp_mask ? comp_mask[(uint64_t)t * n_comp + c] : 0.0f;
            s = -INFINITY;
            if (add > -1.0e20f) {
                const float *kvrow = comp_kv + (uint64_t)c * head_dim;
                float dot = 0.0f;
                for (uint32_t d = 0; d < head_dim; d++) dot += qh[d] * kvrow[d];
                s = dot * scale + add;
            }
        }
        scores[i] = s;
        local_max = fmaxf(local_max, s);
    }
    partial[threadIdx.x] = local_max;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] = fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
        __syncthreads();
    }
    if (threadIdx.x == 0) m_s = partial[0];
    __syncthreads();
    float chunk_max = m_s;
    /* All-masked / empty-chunk guard: never evaluate exp(-INF - -INF) -> NaN.
     * Write zero partial (m=-INF, l=0, acc=0) and return. */
    if (!isfinite(chunk_max)) {
        for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) pout[2u + d] = 0.0f;
        if (threadIdx.x == 0) { pout[0] = -INFINITY; pout[1] = 0.0f; }
        return;
    }
    /* Pass 2: exponentiate in place and reduce denominator. */
    float den_local = 0.0f;
    for (uint32_t i = threadIdx.x; i < cnt; i += blockDim.x) {
        float e = expf(scores[i] - chunk_max);
        scores[i] = e;
        den_local += e;
    }
    partial[threadIdx.x] = den_local;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) l_s = partial[0];
    __syncthreads();
    /* Pass 3: weighted value accumulation over this chunk's rows (ascending g),
     * preserving raw-then-comp ordering to match the reference accumulation. */
    if (head_dim == 512u && blockDim.x == 256u) {
        uint32_t d0 = threadIdx.x;
        uint32_t d1 = d0 + 256u;
        float acc0 = 0.0f;
        float acc1 = 0.0f;
        for (uint32_t i = 0; i < cnt; i++) {
            uint32_t g = g0 + i;
            float s = scores[i];
            const float *kv = (g < raw_count)
                    ? raw_kv + (uint64_t)raw_rows[g - raw_lo] * head_dim
                    : comp_kv + (uint64_t)(g - raw_count) * head_dim;
            acc0 += kv[d0] * s;
            acc1 += kv[d1] * s;
        }
        pout[2u + d0] = acc0;
        pout[2u + d1] = acc1;
    } else {
        for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
            float acc = 0.0f;
            for (uint32_t i = 0; i < cnt; i++) {
                uint32_t g = g0 + i;
                float s = scores[i];
                const float *kv = (g < raw_count)
                        ? raw_kv + (uint64_t)raw_rows[g - raw_lo] * head_dim
                        : comp_kv + (uint64_t)(g - raw_count) * head_dim;
                acc += kv[d] * s;
            }
            pout[2u + d] = acc;
        }
    }
    if (threadIdx.x == 0) {
        pout[0] = chunk_max;
        pout[1] = l_s;
    }
}

__global__ static void attention_decode_splitkv_combine_kernel(
        float *heads,
        const float *sinks,
        const float *partials,
        uint32_t n_tokens,
        uint32_t n_head,
        uint32_t head_dim,
        uint32_t S) {
    uint32_t t = blockIdx.x;
    uint32_t h = blockIdx.y;
    if (t >= n_tokens || h >= n_head) return;
    const float *base = partials + (((uint64_t)t * n_head + h) * S) * (head_dim + 2u);
    uint32_t stride = head_dim + 2u;
    __shared__ float M_s;
    __shared__ float L_s;
    if (threadIdx.x == 0) {
        /* Global max M = max(sink, max_j m_j); sink placed first to match the
         * reference (sink seeds local_max). */
        float M = sinks[h];
        for (uint32_t jj = 0; jj < S; jj++) {
            float m_j = base[(uint64_t)jj * stride];
            M = fmaxf(M, m_j);   /* -INF partials never raise M */
        }
        M_s = M;
        /* L = Σ_j exp(m_j - M) * l_j + exp(sink - M); sink term added last to
         * mirror the reference's denom = Σ scores + expf(sink - max). Chunks
         * with l_j == 0 / m_j == -INF contribute exactly 0 (guarded to avoid
         * exp(-INF - finite) * 0 edge cases). */
        float L = 0.0f;
        for (uint32_t jj = 0; jj < S; jj++) {
            float m_j = base[(uint64_t)jj * stride];
            float l_j = base[(uint64_t)jj * stride + 1u];
            if (l_j != 0.0f && isfinite(m_j)) L += expf(m_j - M) * l_j;
        }
        L += expf(sinks[h] - M);
        L_s = L;
    }
    __syncthreads();
    float M = M_s;
    float L = L_s;
    float *oh = heads + ((uint64_t)t * n_head + h) * head_dim;
    for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float A = 0.0f;
        for (uint32_t jj = 0; jj < S; jj++) {
            float m_j = base[(uint64_t)jj * stride];
            float l_j = base[(uint64_t)jj * stride + 1u];
            if (l_j != 0.0f && isfinite(m_j)) {
                A += expf(m_j - M) * base[(uint64_t)jj * stride + 2u + d];
            }
        }
        oh[d] = A / L;
    }
}

__device__ __forceinline__ void attention_compact_topk_stable(
        uint32_t *comp_rows,
        uint32_t *comp_count,
        uint32_t *warp_offsets,
        const int32_t *topk,
        uint32_t top_k,
        uint32_t visible_comp) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t n_warp = blockDim.x >> 5u;
    if (threadIdx.x == 0u) *comp_count = 0u;
    __syncthreads();

    for (uint32_t base = 0u; base < 512u; base += blockDim.x) {
        const uint32_t i = base + threadIdx.x;
        const int32_t c = i < top_k ? topk[i] : -1;
        const bool valid = c >= 0 && (uint32_t)c < visible_comp;
        const uint32_t mask = __ballot_sync(0xffffffffu, valid);
        if (lane == 0u) warp_offsets[warp] = __popc(mask);
        __syncthreads();
        if (threadIdx.x == 0u) {
            uint32_t out = *comp_count;
            for (uint32_t w = 0u; w < n_warp; w++) {
                const uint32_t count = warp_offsets[w];
                warp_offsets[w] = out;
                out += count;
            }
            *comp_count = out;
        }
        __syncthreads();
        if (valid) {
            const uint32_t lanes_before = lane == 0u
                ? 0u : ((1u << lane) - 1u);
            const uint32_t slot = warp_offsets[warp] +
                                  __popc(mask & lanes_before);
            if (slot < 512u) comp_rows[slot] = (uint32_t)c;
        }
        __syncthreads();
    }
}

__global__ static void attention_indexed_mixed_kernel(
        float *heads,
        const float *sinks,
        const float *q,
        const float *raw_kv,
        const float *comp_kv,
        const int32_t *topk,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t n_raw,
        uint32_t raw_cap,
        uint32_t raw_start,
        uint32_t n_comp,
        uint32_t top_k,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_head,
        uint32_t head_dim) {
    uint32_t t = blockIdx.x;
    uint32_t h = blockIdx.y;
    if (t >= n_tokens || h >= n_head) return;
    uint32_t qpos = pos0 + t;
    uint32_t first_raw_pos = pos0 + n_tokens - n_raw;
    uint32_t visible_comp = n_comp;
    if (ratio != 0) {
        visible_comp = (qpos + 1u) / ratio;
        if (visible_comp > n_comp) visible_comp = n_comp;
    }
    const float *qh = q + ((uint64_t)t * n_head + h) * head_dim;
    __shared__ float scores[768];
    __shared__ uint32_t raw_rows[256];
    __shared__ uint32_t comp_rows[512];
    __shared__ float partial[256];
    __shared__ float max_s;
    __shared__ float denom;
    __shared__ uint32_t raw_count;
    __shared__ uint32_t raw_first_idx;
    __shared__ uint32_t comp_count;
    __shared__ uint32_t comp_warp_offsets[8];
    float scale = rsqrtf((float)head_dim);
    if (threadIdx.x == 0) {
        raw_count = 0;
        raw_first_idx = 0;
        if (n_raw != 0) {
            const uint32_t raw_last_pos = first_raw_pos + n_raw - 1u;
            if (qpos >= first_raw_pos) {
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
    }
    __syncthreads();
    for (uint32_t r = threadIdx.x; r < raw_count; r += blockDim.x) {
        raw_rows[r] = (raw_start + raw_first_idx + r) % raw_cap;
    }
    attention_compact_topk_stable(
        comp_rows, &comp_count, comp_warp_offsets,
        topk + (uint64_t)t * top_k, top_k, visible_comp);
    uint32_t n_score = raw_count + comp_count;
    float local_max = sinks[h];
    if (comp_count == 0) {
        for (uint32_t r = threadIdx.x; r < raw_count; r += blockDim.x) {
            const float *kvrow = raw_kv + (uint64_t)raw_rows[r] * head_dim;
            float dot = 0.0f;
            for (uint32_t d = 0; d < head_dim; d++) dot += qh[d] * kvrow[d];
            scores[r] = dot * scale;
            local_max = fmaxf(local_max, scores[r]);
        }
    } else {
        uint32_t qlane = threadIdx.x & 7u;
        uint32_t qgroup = threadIdx.x >> 3u;
        for (uint32_t row0 = 0; row0 < n_score; row0 += 32u) {
            uint32_t row = row0 + qgroup;
            if (row < n_score) {
                const float *kvrow = row < raw_count
                    ? raw_kv + (uint64_t)raw_rows[row] * head_dim
                    : comp_kv + (uint64_t)comp_rows[row - raw_count] * head_dim;
                float dot = 0.0f;
                for (uint32_t d = qlane; d < head_dim; d += 8u) dot += qh[d] * kvrow[d];
                const uint32_t mask = 0xffu << (threadIdx.x & 24u);
                for (uint32_t off = 4u; off > 0u; off >>= 1u) {
                    dot += __shfl_down_sync(mask, dot, off, 8);
                }
                if (qlane == 0) scores[row] = dot * scale;
            }
        }
        __syncthreads();
        for (uint32_t i = threadIdx.x; i < n_score; i += blockDim.x) {
            local_max = fmaxf(local_max, scores[i]);
        }
    }
    partial[threadIdx.x] = local_max;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] = fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
        __syncthreads();
    }
    if (threadIdx.x == 0) max_s = partial[0];
    __syncthreads();
    float den_local = 0.0f;
    for (uint32_t i = threadIdx.x; i < n_score; i += blockDim.x) {
        scores[i] = expf(scores[i] - max_s);
        den_local += scores[i];
    }
    partial[threadIdx.x] = den_local;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) denom = partial[0] + expf(sinks[h] - max_s);
    __syncthreads();
    float *oh = heads + ((uint64_t)t * n_head + h) * head_dim;
    if (head_dim == 512u && blockDim.x == 256u) {
        uint32_t d0 = threadIdx.x;
        uint32_t d1 = d0 + 256u;
        float acc0 = 0.0f;
        float acc1 = 0.0f;
        for (uint32_t r = 0; r < raw_count; r++) {
            float s = scores[r];
            const float *kv = raw_kv + (uint64_t)raw_rows[r] * head_dim;
            acc0 += kv[d0] * s;
            acc1 += kv[d1] * s;
        }
        for (uint32_t c = 0; c < comp_count; c++) {
            float s = scores[raw_count + c];
            const float *kv = comp_kv + (uint64_t)comp_rows[c] * head_dim;
            acc0 += kv[d0] * s;
            acc1 += kv[d1] * s;
        }
        oh[d0] = acc0 / denom;
        oh[d1] = acc1 / denom;
    } else {
        for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
            float acc = 0.0f;
            for (uint32_t r = 0; r < raw_count; r++) acc += raw_kv[(uint64_t)raw_rows[r] * head_dim + d] * scores[r];
            for (uint32_t s = 0; s < comp_count; s++) acc += comp_kv[(uint64_t)comp_rows[s] * head_dim + d] * scores[raw_count + s];
            oh[d] = acc / denom;
        }
    }
}

__global__ static void attention_indexed_mixed_decode_rows_kernel(
        float *heads,
        const float *sinks,
        const float *q,
        cuda_attention_decode_row_table rows,
        uint32_t n_rows,
        uint32_t n_head,
        uint32_t head_dim) {
    const uint32_t row = blockIdx.x;
    const uint32_t h = blockIdx.y;
    if (row >= n_rows || h >= n_head) return;
    const ds4_gpu_attention_decode_row dsc = rows.row[row];
    if (!dsc.indexed) return;
    const float *raw_kv = (const float *)(uintptr_t)dsc.raw_kv;
    const float *comp_kv = (const float *)(uintptr_t)dsc.comp_kv;
    const int32_t *topk = (const int32_t *)(uintptr_t)dsc.topk;
    const uint32_t qpos = dsc.pos;
    const uint32_t first_raw_pos = dsc.pos + 1u - dsc.n_raw;
    uint32_t visible_comp = dsc.n_comp;
    if (dsc.ratio != 0u) {
        visible_comp = (qpos + 1u) / dsc.ratio;
        if (visible_comp > dsc.n_comp) visible_comp = dsc.n_comp;
    }
    const float *qh = q + ((uint64_t)row * n_head + h) * head_dim;
    __shared__ float scores[768];
    __shared__ uint32_t raw_rows[256];
    __shared__ uint32_t comp_rows[512];
    __shared__ float partial[256];
    __shared__ float max_s;
    __shared__ float denom;
    __shared__ uint32_t raw_count;
    __shared__ uint32_t raw_first_idx;
    __shared__ uint32_t comp_count;
    __shared__ uint32_t comp_warp_offsets[8];
    const float scale = rsqrtf((float)head_dim);
    if (threadIdx.x == 0u) {
        raw_count = 0u;
        raw_first_idx = 0u;
        if (dsc.n_raw != 0u) {
            const uint32_t raw_last_pos = first_raw_pos + dsc.n_raw - 1u;
            if (qpos >= first_raw_pos) {
                uint32_t lo = first_raw_pos;
                if (dsc.window != 0u && qpos + 1u > dsc.window) {
                    const uint32_t wlo = qpos + 1u - dsc.window;
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
    }
    __syncthreads();
    for (uint32_t r = threadIdx.x; r < raw_count; r += blockDim.x) {
        raw_rows[r] =
            (dsc.raw_start + raw_first_idx + r) % dsc.raw_cap;
    }
    attention_compact_topk_stable(
        comp_rows, &comp_count, comp_warp_offsets,
        topk, dsc.top_k, visible_comp);
    const uint32_t n_score = raw_count + comp_count;
    float local_max = sinks[h];
    if (comp_count == 0u) {
        for (uint32_t r = threadIdx.x; r < raw_count; r += blockDim.x) {
            const float *kvrow = raw_kv + (uint64_t)raw_rows[r] * head_dim;
            float dot = 0.0f;
            for (uint32_t dim = 0; dim < head_dim; dim++) {
                dot += qh[dim] * kvrow[dim];
            }
            scores[r] = dot * scale;
            local_max = fmaxf(local_max, scores[r]);
        }
    } else {
        const uint32_t qlane = threadIdx.x & 7u;
        const uint32_t qgroup = threadIdx.x >> 3u;
        for (uint32_t row0 = 0; row0 < n_score; row0 += 32u) {
            const uint32_t score_row = row0 + qgroup;
            if (score_row < n_score) {
                const float *kvrow = score_row < raw_count
                    ? raw_kv + (uint64_t)raw_rows[score_row] * head_dim
                    : comp_kv +
                        (uint64_t)comp_rows[score_row - raw_count] * head_dim;
                float dot = 0.0f;
                for (uint32_t dim = qlane; dim < head_dim; dim += 8u) {
                    dot += qh[dim] * kvrow[dim];
                }
                const uint32_t mask = 0xffu << (threadIdx.x & 24u);
                for (uint32_t off = 4u; off > 0u; off >>= 1u) {
                    dot += __shfl_down_sync(mask, dot, off, 8);
                }
                if (qlane == 0u) scores[score_row] = dot * scale;
            }
        }
        __syncthreads();
        for (uint32_t i = threadIdx.x; i < n_score; i += blockDim.x) {
            local_max = fmaxf(local_max, scores[i]);
        }
    }
    partial[threadIdx.x] = local_max;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1u;
         stride > 0u;
         stride >>= 1u) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] =
                fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0u) max_s = partial[0];
    __syncthreads();
    float den_local = 0.0f;
    for (uint32_t i = threadIdx.x; i < n_score; i += blockDim.x) {
        scores[i] = expf(scores[i] - max_s);
        den_local += scores[i];
    }
    partial[threadIdx.x] = den_local;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1u;
         stride > 0u;
         stride >>= 1u) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0u) {
        denom = partial[0] + expf(sinks[h] - max_s);
    }
    __syncthreads();
    float *oh = heads + ((uint64_t)row * n_head + h) * head_dim;
    if (head_dim == 512u && blockDim.x == 256u) {
        const uint32_t d0 = threadIdx.x;
        const uint32_t d1 = d0 + 256u;
        float acc0 = 0.0f;
        float acc1 = 0.0f;
        for (uint32_t r = 0; r < raw_count; r++) {
            const float s = scores[r];
            const float *kv = raw_kv + (uint64_t)raw_rows[r] * head_dim;
            acc0 += kv[d0] * s;
            acc1 += kv[d1] * s;
        }
        for (uint32_t c = 0; c < comp_count; c++) {
            const float s = scores[raw_count + c];
            const float *kv = comp_kv + (uint64_t)comp_rows[c] * head_dim;
            acc0 += kv[d0] * s;
            acc1 += kv[d1] * s;
        }
        oh[d0] = acc0 / denom;
        oh[d1] = acc1 / denom;
    } else {
        for (uint32_t dim = threadIdx.x;
             dim < head_dim;
             dim += blockDim.x) {
            float acc = 0.0f;
            for (uint32_t r = 0; r < raw_count; r++) {
                acc += raw_kv[(uint64_t)raw_rows[r] * head_dim + dim] *
                       scores[r];
            }
            for (uint32_t c = 0; c < comp_count; c++) {
                acc += comp_kv[(uint64_t)comp_rows[c] * head_dim + dim] *
                       scores[raw_count + c];
            }
            oh[dim] = acc / denom;
        }
    }
}

__global__ static void attention_indexed_mixed_heads8_rb4_kernel(
        float *heads,
        const float *sinks,
        const float *q,
        const float *raw_kv,
        const float *comp_kv,
        const int32_t *topk,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t n_raw,
        uint32_t raw_cap,
        uint32_t raw_start,
        uint32_t n_comp,
        uint32_t top_k,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_head,
        uint32_t head_dim) {
    uint32_t t = blockIdx.x;
    uint32_t head_group = blockIdx.y;
    if (t >= n_tokens || head_dim != 512u) return;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t head = head_group * 8u + warp;
    const bool valid_head = head < n_head;

    __shared__ uint32_t raw_rows[256];
    __shared__ uint32_t comp_rows[512];
    __shared__ uint32_t raw_count;
    __shared__ uint32_t raw_first_idx;
    __shared__ uint32_t comp_count;
    __shared__ float4 kv_shared[4 * 128];
    __shared__ float scores[8 * 768];

    uint32_t qpos = pos0 + t;
    uint32_t first_raw_pos = pos0 + n_tokens - n_raw;
    uint32_t visible_comp = n_comp;
    if (ratio != 0) {
        visible_comp = (qpos + 1u) / ratio;
        if (visible_comp > n_comp) visible_comp = n_comp;
    }

    if (threadIdx.x == 0) {
        raw_count = 0;
        raw_first_idx = 0;
        comp_count = 0;
        if (n_raw != 0) {
            const uint32_t raw_last_pos = first_raw_pos + n_raw - 1u;
            if (qpos >= first_raw_pos) {
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
    }
    __syncthreads();
    for (uint32_t r = threadIdx.x; r < raw_count; r += blockDim.x) {
        raw_rows[r] = (raw_start + raw_first_idx + r) % raw_cap;
    }
    if (threadIdx.x == 0) {
        for (uint32_t i = 0; i < top_k && comp_count < 512u; i++) {
            int32_t c = topk[(uint64_t)t * top_k + i];
            if (c >= 0 && (uint32_t)c < visible_comp) comp_rows[comp_count++] = (uint32_t)c;
        }
    }
    __syncthreads();

    const uint32_t n_score = raw_count + comp_count;
    const float scale = rsqrtf((float)head_dim);
    const float4 *q4 = valid_head
        ? (const float4 *)(q + ((uint64_t)t * n_head + head) * head_dim)
        : NULL;
    float4 q0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 q1 = q0, q2 = q0, q3 = q0;
    if (valid_head) {
        q0 = q4[lane +  0u];
        q1 = q4[lane + 32u];
        q2 = q4[lane + 64u];
        q3 = q4[lane + 96u];
    }

    for (uint32_t row0 = 0; row0 < n_score; row0 += 4u) {
        const uint32_t nr = n_score - row0 < 4u ? n_score - row0 : 4u;
        for (uint32_t off = threadIdx.x; off < nr * 128u; off += blockDim.x) {
            const uint32_t rr = off >> 7u;
            const uint32_t c4 = off & 127u;
            const uint32_t sr = row0 + rr;
            const float4 *src = sr < raw_count
                ? (const float4 *)(raw_kv + (uint64_t)raw_rows[sr] * head_dim)
                : (const float4 *)(comp_kv + (uint64_t)comp_rows[sr - raw_count] * head_dim);
            kv_shared[off] = src[c4];
        }
        __syncthreads();
        if (valid_head) {
            for (uint32_t rr = 0; rr < nr; rr++) {
                const float4 *kv4 = kv_shared + rr * 128u;
                float dot = dot4_f32(q0, kv4[lane +  0u]) +
                            dot4_f32(q1, kv4[lane + 32u]) +
                            dot4_f32(q2, kv4[lane + 64u]) +
                            dot4_f32(q3, kv4[lane + 96u]);
                dot = warp_sum_f32(dot);
                if (lane == 0) scores[warp * 768u + row0 + rr] = dot * scale;
            }
        }
        __syncthreads();
    }

    float max_s = valid_head ? sinks[head] : -INFINITY;
    if (valid_head) {
        const float *score_row = scores + warp * 768u;
        for (uint32_t i = lane; i < n_score; i += 32u) max_s = fmaxf(max_s, score_row[i]);
        max_s = warp_max_f32(max_s);
        max_s = __shfl_sync(0xffffffffu, max_s, 0);
    }
    float den = 0.0f;
    if (valid_head) {
        float *score_row = scores + warp * 768u;
        for (uint32_t i = lane; i < n_score; i += 32u) {
            float p = expf(score_row[i] - max_s);
            score_row[i] = p;
            den += p;
        }
        den = warp_sum_f32(den);
        den += expf(sinks[head] - max_s);
        den = __shfl_sync(0xffffffffu, den, 0);
    }

    float4 o0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 o1 = o0, o2 = o0, o3 = o0;
    for (uint32_t row0 = 0; row0 < n_score; row0 += 4u) {
        const uint32_t nr = n_score - row0 < 4u ? n_score - row0 : 4u;
        for (uint32_t off = threadIdx.x; off < nr * 128u; off += blockDim.x) {
            const uint32_t rr = off >> 7u;
            const uint32_t c4 = off & 127u;
            const uint32_t sr = row0 + rr;
            const float4 *src = sr < raw_count
                ? (const float4 *)(raw_kv + (uint64_t)raw_rows[sr] * head_dim)
                : (const float4 *)(comp_kv + (uint64_t)comp_rows[sr - raw_count] * head_dim);
            kv_shared[off] = src[c4];
        }
        __syncthreads();
        if (valid_head) {
            const float *score_row = scores + warp * 768u;
            for (uint32_t rr = 0; rr < nr; rr++) {
                const float p = den == 0.0f ? 0.0f : score_row[row0 + rr] / den;
                const float4 *kv4 = kv_shared + rr * 128u;
                float4 k0 = kv4[lane +  0u];
                float4 k1 = kv4[lane + 32u];
                float4 k2 = kv4[lane + 64u];
                float4 k3 = kv4[lane + 96u];
                o0.x += k0.x * p; o0.y += k0.y * p; o0.z += k0.z * p; o0.w += k0.w * p;
                o1.x += k1.x * p; o1.y += k1.y * p; o1.z += k1.z * p; o1.w += k1.w * p;
                o2.x += k2.x * p; o2.y += k2.y * p; o2.z += k2.z * p; o2.w += k2.w * p;
                o3.x += k3.x * p; o3.y += k3.y * p; o3.z += k3.z * p; o3.w += k3.w * p;
            }
        }
        __syncthreads();
    }
    if (valid_head) {
        float4 *out4 = (float4 *)(heads + ((uint64_t)t * n_head + head) * head_dim);
        out4[lane +  0u] = o0;
        out4[lane + 32u] = o1;
        out4[lane + 64u] = o2;
        out4[lane + 96u] = o3;
    }
}

template <uint32_t ROWS_PER_STAGE, uint32_t HEADS_PER_GROUP>
__global__ static void attention_indexed_mixed_heads8_online_kernel(
        float *heads,
        const float *sinks,
        const float *q,
        const float *raw_kv,
        const float *comp_kv,
        const int32_t *topk,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t n_raw,
        uint32_t raw_cap,
        uint32_t raw_start,
        uint32_t n_comp,
        uint32_t top_k,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_head,
        uint32_t head_dim) {
    uint32_t t = blockIdx.x;
    uint32_t head_group = blockIdx.y;
    if (t >= n_tokens || head_dim != 512u) return;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t head = head_group * HEADS_PER_GROUP + warp;
    const bool valid_head = head < n_head;

    __shared__ uint32_t raw_rows[256];
    __shared__ uint32_t raw_count;
    __shared__ uint32_t raw_first_idx;
    __shared__ float4 kv_shared[ROWS_PER_STAGE * 128];

    uint32_t qpos = pos0 + t;
    uint32_t first_raw_pos = pos0 + n_tokens - n_raw;
    uint32_t visible_comp = n_comp;
    if (ratio != 0) {
        visible_comp = (qpos + 1u) / ratio;
        if (visible_comp > n_comp) visible_comp = n_comp;
    }

    if (threadIdx.x == 0) {
        raw_count = 0;
        raw_first_idx = 0;
        if (n_raw != 0) {
            const uint32_t raw_last_pos = first_raw_pos + n_raw - 1u;
            if (qpos >= first_raw_pos) {
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
    }
    __syncthreads();
    for (uint32_t r = threadIdx.x; r < raw_count; r += blockDim.x) {
        raw_rows[r] = (raw_start + raw_first_idx + r) % raw_cap;
    }
    __syncthreads();

    uint32_t comp_count = top_k < visible_comp ? top_k : visible_comp;
    if (comp_count > 512u) comp_count = 512u;
    const uint32_t n_score = raw_count + comp_count;
    const float scale = rsqrtf((float)head_dim);
    const float4 *q4 = valid_head
        ? (const float4 *)(q + ((uint64_t)t * n_head + head) * head_dim)
        : NULL;
    float4 q0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 q1 = q0, q2 = q0, q3 = q0;
    if (valid_head) {
        q0 = q4[lane +  0u];
        q1 = q4[lane + 32u];
        q2 = q4[lane + 64u];
        q3 = q4[lane + 96u];
    }

    float max_s = -INFINITY;
    float sum_s = 0.0f;
    float4 o0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 o1 = o0, o2 = o0, o3 = o0;

    for (uint32_t row0 = 0; row0 < n_score; row0 += ROWS_PER_STAGE) {
        const uint32_t nr = n_score - row0 < ROWS_PER_STAGE ? n_score - row0 : ROWS_PER_STAGE;
        for (uint32_t off = threadIdx.x; off < nr * 128u; off += blockDim.x) {
            const uint32_t rr = off >> 7u;
            const uint32_t c4 = off & 127u;
            const uint32_t sr = row0 + rr;
            const uint32_t comp_idx = sr < raw_count
                ? 0u
                : (uint32_t)topk[(uint64_t)t * top_k + (sr - raw_count)];
            const float4 *src = sr < raw_count
                ? (const float4 *)(raw_kv + (uint64_t)raw_rows[sr] * head_dim)
                : (const float4 *)(comp_kv + (uint64_t)comp_idx * head_dim);
            kv_shared[off] = src[c4];
        }
        __syncthreads();
        if (valid_head) {
            for (uint32_t rr = 0; rr < nr; rr++) {
                const float4 *kv4 = kv_shared + rr * 128u;
                float4 k0 = kv4[lane +  0u];
                float4 k1 = kv4[lane + 32u];
                float4 k2 = kv4[lane + 64u];
                float4 k3 = kv4[lane + 96u];
                float score = dot4_f32(q0, k0) +
                              dot4_f32(q1, k1) +
                              dot4_f32(q2, k2) +
                              dot4_f32(q3, k3);
                score = warp_sum_f32(score) * scale;
                score = __shfl_sync(0xffffffffu, score, 0);

                const float new_m = fmaxf(max_s, score);
                const float old_scale = expf(max_s - new_m);
                const float row_scale = expf(score - new_m);
                sum_s = sum_s * old_scale + row_scale;
                o0.x = o0.x * old_scale + k0.x * row_scale;
                o0.y = o0.y * old_scale + k0.y * row_scale;
                o0.z = o0.z * old_scale + k0.z * row_scale;
                o0.w = o0.w * old_scale + k0.w * row_scale;
                o1.x = o1.x * old_scale + k1.x * row_scale;
                o1.y = o1.y * old_scale + k1.y * row_scale;
                o1.z = o1.z * old_scale + k1.z * row_scale;
                o1.w = o1.w * old_scale + k1.w * row_scale;
                o2.x = o2.x * old_scale + k2.x * row_scale;
                o2.y = o2.y * old_scale + k2.y * row_scale;
                o2.z = o2.z * old_scale + k2.z * row_scale;
                o2.w = o2.w * old_scale + k2.w * row_scale;
                o3.x = o3.x * old_scale + k3.x * row_scale;
                o3.y = o3.y * old_scale + k3.y * row_scale;
                o3.z = o3.z * old_scale + k3.z * row_scale;
                o3.w = o3.w * old_scale + k3.w * row_scale;
                max_s = new_m;
            }
        }
        __syncthreads();
    }

    if (valid_head) {
        const float sink = sinks[head];
        const float new_m = fmaxf(max_s, sink);
        const float old_scale = expf(max_s - new_m);
        const float sink_scale = expf(sink - new_m);
        sum_s = sum_s * old_scale + sink_scale;
        o0.x *= old_scale; o0.y *= old_scale; o0.z *= old_scale; o0.w *= old_scale;
        o1.x *= old_scale; o1.y *= old_scale; o1.z *= old_scale; o1.w *= old_scale;
        o2.x *= old_scale; o2.y *= old_scale; o2.z *= old_scale; o2.w *= old_scale;
        o3.x *= old_scale; o3.y *= old_scale; o3.z *= old_scale; o3.w *= old_scale;

        const float inv_s = sum_s == 0.0f ? 0.0f : 1.0f / sum_s;
        o0.x *= inv_s; o0.y *= inv_s; o0.z *= inv_s; o0.w *= inv_s;
        o1.x *= inv_s; o1.y *= inv_s; o1.z *= inv_s; o1.w *= inv_s;
        o2.x *= inv_s; o2.y *= inv_s; o2.z *= inv_s; o2.w *= inv_s;
        o3.x *= inv_s; o3.y *= inv_s; o3.z *= inv_s; o3.w *= inv_s;
        float4 *out4 = (float4 *)(heads + ((uint64_t)t * n_head + head) * head_dim);
        out4[lane +  0u] = o0;
        out4[lane + 32u] = o1;
        out4[lane + 64u] = o2;
        out4[lane + 96u] = o3;
    }
}

__global__ static void attention_static_mixed_heads8_online_kernel(
        float *heads,
        const float *sinks,
        const float *q,
        const float *raw_kv,
        const float *comp_kv,
        uint32_t n_tokens,
        uint32_t n_comp,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_head,
        uint32_t head_dim) {
    uint32_t t = blockIdx.x;
    uint32_t head_group = blockIdx.y;
    if (t >= n_tokens || head_dim != 512u) return;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t head = head_group * 8u + warp;
    const bool valid_head = head < n_head;

    __shared__ float4 kv_shared[4 * 128];

    const uint32_t raw_count = window != 0u && t + 1u > window ? window : t + 1u;
    const uint32_t raw_start = t + 1u - raw_count;
    uint32_t comp_count = 0;
    if (n_comp != 0u && ratio != 0u) {
        comp_count = (t + 1u) / ratio;
        if (comp_count > n_comp) comp_count = n_comp;
    }
    const uint32_t n_score = raw_count + comp_count;
    const float scale = rsqrtf((float)head_dim);
    const float4 *q4 = valid_head
        ? (const float4 *)(q + ((uint64_t)t * n_head + head) * head_dim)
        : NULL;
    float4 q0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 q1 = q0, q2 = q0, q3 = q0;
    if (valid_head) {
        q0 = q4[lane +  0u];
        q1 = q4[lane + 32u];
        q2 = q4[lane + 64u];
        q3 = q4[lane + 96u];
    }

    float max_s = -INFINITY;
    float sum_s = 0.0f;
    float4 o0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 o1 = o0, o2 = o0, o3 = o0;

    for (uint32_t row0 = 0; row0 < n_score; row0 += 4u) {
        const uint32_t nr = n_score - row0 < 4u ? n_score - row0 : 4u;
        for (uint32_t off = threadIdx.x; off < nr * 128u; off += blockDim.x) {
            const uint32_t rr = off >> 7u;
            const uint32_t c4 = off & 127u;
            const uint32_t sr = row0 + rr;
            const float4 *src = sr < raw_count
                ? (const float4 *)(raw_kv + (uint64_t)(raw_start + sr) * head_dim)
                : (const float4 *)(comp_kv + (uint64_t)(sr - raw_count) * head_dim);
            kv_shared[off] = src[c4];
        }
        __syncthreads();
        if (valid_head) {
            for (uint32_t rr = 0; rr < nr; rr++) {
                const float4 *kv4 = kv_shared + rr * 128u;
                float4 k0 = kv4[lane +  0u];
                float4 k1 = kv4[lane + 32u];
                float4 k2 = kv4[lane + 64u];
                float4 k3 = kv4[lane + 96u];
                float score = dot4_f32(q0, k0) +
                              dot4_f32(q1, k1) +
                              dot4_f32(q2, k2) +
                              dot4_f32(q3, k3);
                score = warp_sum_f32(score) * scale;
                score = __shfl_sync(0xffffffffu, score, 0);

                const float new_m = fmaxf(max_s, score);
                const float old_scale = expf(max_s - new_m);
                const float row_scale = expf(score - new_m);
                sum_s = sum_s * old_scale + row_scale;
                o0.x = o0.x * old_scale + k0.x * row_scale;
                o0.y = o0.y * old_scale + k0.y * row_scale;
                o0.z = o0.z * old_scale + k0.z * row_scale;
                o0.w = o0.w * old_scale + k0.w * row_scale;
                o1.x = o1.x * old_scale + k1.x * row_scale;
                o1.y = o1.y * old_scale + k1.y * row_scale;
                o1.z = o1.z * old_scale + k1.z * row_scale;
                o1.w = o1.w * old_scale + k1.w * row_scale;
                o2.x = o2.x * old_scale + k2.x * row_scale;
                o2.y = o2.y * old_scale + k2.y * row_scale;
                o2.z = o2.z * old_scale + k2.z * row_scale;
                o2.w = o2.w * old_scale + k2.w * row_scale;
                o3.x = o3.x * old_scale + k3.x * row_scale;
                o3.y = o3.y * old_scale + k3.y * row_scale;
                o3.z = o3.z * old_scale + k3.z * row_scale;
                o3.w = o3.w * old_scale + k3.w * row_scale;
                max_s = new_m;
            }
        }
        __syncthreads();
    }

    if (valid_head) {
        const float sink = sinks[head];
        const float new_m = fmaxf(max_s, sink);
        const float old_scale = expf(max_s - new_m);
        const float sink_scale = expf(sink - new_m);
        sum_s = sum_s * old_scale + sink_scale;
        o0.x *= old_scale; o0.y *= old_scale; o0.z *= old_scale; o0.w *= old_scale;
        o1.x *= old_scale; o1.y *= old_scale; o1.z *= old_scale; o1.w *= old_scale;
        o2.x *= old_scale; o2.y *= old_scale; o2.z *= old_scale; o2.w *= old_scale;
        o3.x *= old_scale; o3.y *= old_scale; o3.z *= old_scale; o3.w *= old_scale;

        const float inv_s = sum_s == 0.0f ? 0.0f : 1.0f / sum_s;
        o0.x *= inv_s; o0.y *= inv_s; o0.z *= inv_s; o0.w *= inv_s;
        o1.x *= inv_s; o1.y *= inv_s; o1.z *= inv_s; o1.w *= inv_s;
        o2.x *= inv_s; o2.y *= inv_s; o2.z *= inv_s; o2.w *= inv_s;
        o3.x *= inv_s; o3.y *= inv_s; o3.z *= inv_s; o3.w *= inv_s;
        float4 *out4 = (float4 *)(heads + ((uint64_t)t * n_head + head) * head_dim);
        out4[lane +  0u] = o0;
        out4[lane + 32u] = o1;
        out4[lane + 64u] = o2;
        out4[lane + 96u] = o3;
    }
}

__global__ static void attention_decode_mixed_heads8_online_kernel(
        float *heads,
        const float *sinks,
        const float *q,
        const float *raw_kv,
        const float *comp_kv,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t n_raw,
        uint32_t raw_cap,
        uint32_t raw_start,
        uint32_t n_comp,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_head,
        uint32_t head_dim) {
    uint32_t t = blockIdx.x;
    uint32_t head_group = blockIdx.y;
    if (t >= n_tokens || head_dim != 512u) return;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t head = head_group * 8u + warp;
    const bool valid_head = head < n_head;

    __shared__ uint32_t raw_rows[256];
    __shared__ uint32_t raw_count_s;
    __shared__ uint32_t raw_first_idx_s;
    __shared__ float4 kv_shared[4 * 128];

    const uint32_t qpos = pos0 + t;
    const uint32_t first_raw_pos = pos0 + n_tokens - n_raw;
    uint32_t comp_count = 0;
    if (n_comp != 0u) {
        if (n_tokens == 1u && ratio == 0u) {
            comp_count = n_comp;
        } else if (ratio != 0u) {
            comp_count = (qpos + 1u) / ratio;
            if (comp_count > n_comp) comp_count = n_comp;
        }
    }
    if (threadIdx.x == 0) {
        uint32_t raw_count = 0;
        uint32_t raw_first_idx = 0;
        if (n_raw != 0u) {
            const uint32_t raw_last_pos = first_raw_pos + n_raw - 1u;
            if (qpos >= first_raw_pos) {
                uint32_t lo = first_raw_pos;
                if (window != 0u && qpos + 1u > window) {
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
        raw_count_s = raw_count;
        raw_first_idx_s = raw_first_idx;
    }
    __syncthreads();
    const uint32_t raw_count = raw_count_s;
    const uint32_t raw_first_idx = raw_first_idx_s;
    for (uint32_t r = threadIdx.x; r < raw_count; r += blockDim.x) {
        raw_rows[r] = (raw_start + raw_first_idx + r) % raw_cap;
    }
    __syncthreads();

    const uint32_t n_score = raw_count + comp_count;
    const float scale = rsqrtf((float)head_dim);
    const float4 *q4 = valid_head
        ? (const float4 *)(q + ((uint64_t)t * n_head + head) * head_dim)
        : NULL;
    float4 q0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 q1 = q0, q2 = q0, q3 = q0;
    if (valid_head) {
        q0 = q4[lane +  0u];
        q1 = q4[lane + 32u];
        q2 = q4[lane + 64u];
        q3 = q4[lane + 96u];
    }

    float max_s = -INFINITY;
    float sum_s = 0.0f;
    float4 o0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 o1 = o0, o2 = o0, o3 = o0;

    for (uint32_t row0 = 0; row0 < n_score; row0 += 4u) {
        const uint32_t nr = n_score - row0 < 4u ? n_score - row0 : 4u;
        for (uint32_t off = threadIdx.x; off < nr * 128u; off += blockDim.x) {
            const uint32_t rr = off >> 7u;
            const uint32_t c4 = off & 127u;
            const uint32_t sr = row0 + rr;
            const float4 *src = sr < raw_count
                ? (const float4 *)(raw_kv + (uint64_t)raw_rows[sr] * head_dim)
                : (const float4 *)(comp_kv + (uint64_t)(sr - raw_count) * head_dim);
            kv_shared[off] = src[c4];
        }
        __syncthreads();
        if (valid_head) {
            for (uint32_t rr = 0; rr < nr; rr++) {
                const float4 *kv4 = kv_shared + rr * 128u;
                float4 k0 = kv4[lane +  0u];
                float4 k1 = kv4[lane + 32u];
                float4 k2 = kv4[lane + 64u];
                float4 k3 = kv4[lane + 96u];
                float score = dot4_f32(q0, k0) +
                              dot4_f32(q1, k1) +
                              dot4_f32(q2, k2) +
                              dot4_f32(q3, k3);
                score = warp_sum_f32(score) * scale;
                score = __shfl_sync(0xffffffffu, score, 0);

                const float new_m = fmaxf(max_s, score);
                const float old_scale = expf(max_s - new_m);
                const float row_scale = expf(score - new_m);
                sum_s = sum_s * old_scale + row_scale;
                o0.x = o0.x * old_scale + k0.x * row_scale;
                o0.y = o0.y * old_scale + k0.y * row_scale;
                o0.z = o0.z * old_scale + k0.z * row_scale;
                o0.w = o0.w * old_scale + k0.w * row_scale;
                o1.x = o1.x * old_scale + k1.x * row_scale;
                o1.y = o1.y * old_scale + k1.y * row_scale;
                o1.z = o1.z * old_scale + k1.z * row_scale;
                o1.w = o1.w * old_scale + k1.w * row_scale;
                o2.x = o2.x * old_scale + k2.x * row_scale;
                o2.y = o2.y * old_scale + k2.y * row_scale;
                o2.z = o2.z * old_scale + k2.z * row_scale;
                o2.w = o2.w * old_scale + k2.w * row_scale;
                o3.x = o3.x * old_scale + k3.x * row_scale;
                o3.y = o3.y * old_scale + k3.y * row_scale;
                o3.z = o3.z * old_scale + k3.z * row_scale;
                o3.w = o3.w * old_scale + k3.w * row_scale;
                max_s = new_m;
            }
        }
        __syncthreads();
    }

    if (valid_head) {
        const float sink = sinks[head];
        const float new_m = fmaxf(max_s, sink);
        const float old_scale = expf(max_s - new_m);
        const float sink_scale = expf(sink - new_m);
        sum_s = sum_s * old_scale + sink_scale;
        o0.x *= old_scale; o0.y *= old_scale; o0.z *= old_scale; o0.w *= old_scale;
        o1.x *= old_scale; o1.y *= old_scale; o1.z *= old_scale; o1.w *= old_scale;
        o2.x *= old_scale; o2.y *= old_scale; o2.z *= old_scale; o2.w *= old_scale;
        o3.x *= old_scale; o3.y *= old_scale; o3.z *= old_scale; o3.w *= old_scale;

        const float inv_s = sum_s == 0.0f ? 0.0f : 1.0f / sum_s;
        o0.x *= inv_s; o0.y *= inv_s; o0.z *= inv_s; o0.w *= inv_s;
        o1.x *= inv_s; o1.y *= inv_s; o1.z *= inv_s; o1.w *= inv_s;
        o2.x *= inv_s; o2.y *= inv_s; o2.z *= inv_s; o2.w *= inv_s;
        o3.x *= inv_s; o3.y *= inv_s; o3.z *= inv_s; o3.w *= inv_s;
        float4 *out4 = (float4 *)(heads + ((uint64_t)t * n_head + head) * head_dim);
        out4[lane +  0u] = o0;
        out4[lane + 32u] = o1;
        out4[lane + 64u] = o2;
        out4[lane + 96u] = o3;
    }
}
