#include "ds4_rocm_q4_qb_epilogue_layout.cuh"

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
    const float theta_scale = powf(freq_base, -2.0f / (float)n_rot);
    for (uint32_t pair = threadIdx.x; pair < n_rot / 2; pair += blockDim.x) {
        uint32_t i = pair * 2u;
        float theta_extrap = (float)(pos0 + t) * powf(theta_scale, (float)pair);
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

// Q4 F32-output prefill only: eight independent heads per workgroup, one
// wave32 per head. Retain all 512 inputs in registers and reproduce the
// original reduction tree without LDS or block barriers. Q8 and the generic
// API below deliberately retain their previous dispatch.
__launch_bounds__(256)
__global__ static void rocm_q4_qb_f32_epilogue_wave32_kernel(
        float *x, uint32_t n_tok, uint32_t n_head, uint32_t head_dim,
        uint32_t n_rot, uint32_t pos0, uint32_t n_ctx_orig, int inverse,
        float freq_base, float freq_scale, float ext_factor, float attn_factor,
        float beta_fast, float beta_slow, float eps) {
#if defined(__HIP_DEVICE_COMPILE__) && __HIP_DEVICE_COMPILE__ && \
    defined(__gfx1151__) && \
    (!defined(__AMDGCN_WAVEFRONT_SIZE__) || __AMDGCN_WAVEFRONT_SIZE__ == 32)
    namespace ep = ds4_rocm_q4_qb_epilogue;
    const uint32_t row = blockIdx.x * ep::heads_per_block + threadIdx.x / 32u;
    if (row >= n_tok * n_head) return; // Uniform over the complete wave.
    const uint32_t lane = threadIdx.x & 31u;
    const MASK_T mask = static_cast<MASK_T>(0xffffffffu);
    const uint32_t t = row / n_head;
    float *xr = x + (uint64_t)row * head_dim;
    float v[ep::values];
#pragma unroll
    for (uint32_t j = 0; j < ep::values; ++j)
        v[j] = xr[ep::column(lane, j)];

    float sum;
    {
#pragma clang fp reassociate(off) contract(on)
        sum = ep::fold_columns(v);
#pragma unroll
        for (uint32_t stride = 16u; stride; stride >>= 1u) {
            const float other = __shfl_down_sync(mask, sum, stride, 32);
            if (lane < stride) sum += other;
        }
        sum = __shfl_sync(mask, sum, 0, 32);
    }
    const float scale = rsqrtf(sum / (float)head_dim + eps);
#pragma unroll
    for (uint32_t j = 0; j < 14u; ++j)
        xr[ep::column(lane, j)] = v[j] * scale;

    // Each lane now owns one rotary pair. Both 32-value source halves must
    // be shuffled before selecting: a lane-dependent source register would
    // mix the halves when reading a lane on the other side of the wave.
    const uint32_t source = (lane & 15u) * 2u;
    const float lo0 = __shfl_sync(mask, v[14], source, 32);
    const float lo1 = __shfl_sync(mask, v[14], source + 1u, 32);
    const float hi0 = __shfl_sync(mask, v[15], source, 32);
    const float hi1 = __shfl_sync(mask, v[15], source + 1u, 32);
    const float x0 = (lane < 16u ? lo0 : hi0) * scale;
    const float x1 = (lane < 16u ? lo1 : hi1) * scale;

    // Retain the generic kernel's YaRN expressions and F32 boundary.
    float corr0 = 0.0f, corr1 = 0.0f;
    if (ext_factor != 0.0f) {
        float denom = 2.0f * logf(freq_base);
        corr0 = floorf((float)n_rot * logf((float)n_ctx_orig / (beta_fast * 2.0f * (float)M_PI)) / denom);
        corr1 = ceilf((float)n_rot * logf((float)n_ctx_orig / (beta_slow * 2.0f * (float)M_PI)) / denom);
        corr0 = fmaxf(0.0f, corr0);
        corr1 = fminf((float)(n_rot - 1), corr1);
    }
    const float theta_scale = powf(freq_base, -2.0f / (float)n_rot);
    const uint32_t i = lane * 2u;
    const float theta_extrap = (float)(pos0 + t) * powf(theta_scale, (float)lane);
    const float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    float mscale = attn_factor;
    if (ext_factor != 0.0f) {
        const float ramp_mix = rope_yarn_ramp_dev(corr0, corr1, (int)i) * ext_factor;
        theta = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
        mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    const float c = cosf(theta) * mscale;
    float s = sinf(theta) * mscale;
    if (inverse) s = -s;
    float *tail = xr + head_dim - n_rot;
    tail[i] = x0 * c - x1 * s;
    tail[i + 1u] = x0 * s + x1 * c;
#else
    (void)x; (void)n_tok; (void)n_head; (void)head_dim; (void)n_rot;
    (void)pos0; (void)n_ctx_orig; (void)inverse; (void)freq_base;
    (void)freq_scale; (void)ext_factor; (void)attn_factor;
    (void)beta_fast; (void)beta_slow; (void)eps;
#endif
}

__global__ static void head_rms_norm_rope_tail_from_half_kernel(
        float *out,
        const __half *x,
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
    const __half *xr = x + (uint64_t)row * head_dim;
    float *orow = out + (uint64_t)row * head_dim;
    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x) {
        float v = __half2float(xr[i]);
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
        orow[i] = __half2float(xr[i]) * scale;
    }

    float corr0 = 0.0f, corr1 = 0.0f;
    if (ext_factor != 0.0f) {
        float denom = 2.0f * logf(freq_base);
        corr0 = floorf((float)n_rot * logf((float)n_ctx_orig / (beta_fast * 2.0f * (float)M_PI)) / denom);
        corr1 = ceilf((float)n_rot * logf((float)n_ctx_orig / (beta_slow * 2.0f * (float)M_PI)) / denom);
        corr0 = fmaxf(0.0f, corr0);
        corr1 = fminf((float)(n_rot - 1), corr1);
    }
    const float theta_scale = powf(freq_base, -2.0f / (float)n_rot);
    const __half *tail = xr + n_nope;
    float *otail = orow + n_nope;
    for (uint32_t pair = threadIdx.x; pair < n_rot / 2; pair += blockDim.x) {
        uint32_t i = pair * 2u;
        float theta_extrap = (float)(pos0 + t) * powf(theta_scale, (float)pair);
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
        float x0 = __half2float(tail[i]) * scale;
        float x1 = __half2float(tail[i + 1]) * scale;
        otail[i] = x0 * c - x1 * s;
        otail[i + 1] = x0 * s + x1 * c;
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

    const float theta_scale = powf(freq_base, -2.0f / (float)n_rot);
    float theta_extrap = (float)(pos0 + t * pos_stride) * powf(theta_scale, (float)pair);
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

__device__ static float model_ape_value_dev(const void *base, uint64_t offset, uint32_t type,
                                            uint32_t width, uint32_t row, uint32_t col) {
    const char *p = (const char *)base + offset;
    if (type == 1u) return __half2float(((const __half *)p)[(uint64_t)row * width + col]);
    if (type == 8u) {
        const uint64_t row_bytes = ((uint64_t)width + 31u) / 32u * 34u;
        const unsigned char *blk = (const unsigned char *)p + (uint64_t)row * row_bytes + (uint64_t)(col >> 5) * 34u;
        const float d = q8_0_scale_scalar(blk);
        const int8_t q = ((const int8_t *)(blk + 2u))[col & 31u];
        return d * (float)q;
    }
    return ((const float *)p)[(uint64_t)row * width + col];
}

extern "C" int ds4_gpu_rms_norm_plain_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *x, uint32_t n, float eps) {
    if (!cuda_tensor_has_f32(out, n) || !cuda_tensor_has_f32(x, n)) return 0;
    if (n == 0u) return 1;
    rms_norm_plain_kernel<<<1, 256>>>((float *)out->ptr, (const float *)x->ptr, n, 1, eps);
    return cuda_ok(cudaGetLastError(), "rms_norm_plain launch");
}
extern "C" int ds4_gpu_rms_norm_plain_rows_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *x, uint32_t n, uint32_t rows, float eps) {
    if (!cuda_tensor_has_elems2(out, n, rows, sizeof(float)) ||
        !cuda_tensor_has_elems2(x, n, rows, sizeof(float))) return 0;
    if (n == 0u || rows == 0u) return 1;
    rms_norm_plain_kernel<<<rows, 256>>>((float *)out->ptr, (const float *)x->ptr, n, rows, eps);
    return cuda_ok(cudaGetLastError(), "rms_norm_plain launch");
}
extern "C" int ds4_gpu_rms_norm_weight_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *x, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint32_t n, float eps) {
    uint64_t weight_bytes = 0;
    if (!model_map || !cuda_u64_mul_checked(n, sizeof(float), &weight_bytes) ||
        !cuda_model_range_fits(model_size, weight_offset, weight_bytes) ||
        !cuda_tensor_has_f32(out, n) || !cuda_tensor_has_f32(x, n)) return 0;
    if (n == 0u) return 1;
    const char *wptr = cuda_model_range_ptr(model_map, weight_offset, weight_bytes, "rms_weight");
    if (!wptr) return 0;
    const float *w = (const float *)wptr;
    rms_norm_weight_kernel<<<1, 256>>>((float *)out->ptr, (const float *)x->ptr, w, n, 1, eps);
    return cuda_ok(cudaGetLastError(), "rms_norm_weight launch");
}
extern "C" int ds4_gpu_rms_norm_weight_rows_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *x, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint32_t n, uint32_t rows, float eps) {
    uint64_t weight_bytes = 0;
    if (!model_map || !cuda_u64_mul_checked(n, sizeof(float), &weight_bytes) ||
        !cuda_model_range_fits(model_size, weight_offset, weight_bytes) ||
        !cuda_tensor_has_elems2(out, n, rows, sizeof(float)) ||
        !cuda_tensor_has_elems2(x, n, rows, sizeof(float))) return 0;
    if (n == 0u || rows == 0u) return 1;
    const char *wptr = cuda_model_range_ptr(model_map, weight_offset, weight_bytes, "rms_weight");
    if (!wptr) return 0;
    const float *w = (const float *)wptr;
    rms_norm_weight_kernel<<<rows, 256>>>((float *)out->ptr, (const float *)x->ptr, w, n, rows, eps);
    return cuda_ok(cudaGetLastError(), "rms_norm_weight launch");
}

extern "C" int ds4_gpu_add_rms_norm_weight_tensor(
        ds4_gpu_tensor       *norm_out,
        ds4_gpu_tensor       *sum_out,
        const ds4_gpu_tensor *a,
        const ds4_gpu_tensor *b,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                n,
        float                   eps) {
    return ds4_gpu_add_tensor(sum_out, a, b, n) &&
           ds4_gpu_rms_norm_weight_tensor(norm_out,
                                          sum_out,
                                          model_map,
                                          model_size,
                                          weight_offset,
                                          n,
                                          eps);
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
    uint64_t q_weight_bytes = 0, kv_weight_bytes = 0;
    if (!model_map || !cuda_u64_mul_checked(q_n, sizeof(float), &q_weight_bytes) ||
        !cuda_u64_mul_checked(kv_n, sizeof(float), &kv_weight_bytes) ||
        !cuda_model_range_fits(model_size, q_weight_offset, q_weight_bytes) ||
        !cuda_model_range_fits(model_size, kv_weight_offset, kv_weight_bytes) ||
        !cuda_tensor_has_elems2(q_out, q_n, rows, sizeof(float)) ||
        !cuda_tensor_has_elems2(q, q_n, rows, sizeof(float)) ||
        !cuda_tensor_has_elems2(kv_out, kv_n, rows, sizeof(float)) ||
        !cuda_tensor_has_elems2(kv, kv_n, rows, sizeof(float))) {
        return 0;
    }
    if ((q_n == 0u && kv_n == 0u) || rows == 0u) return 1;
    const float *q_w = (const float *)cuda_model_range_ptr(model_map,
            q_weight_offset, q_weight_bytes, "q_rms_weight");
    const float *kv_w = (const float *)cuda_model_range_ptr(model_map,
            kv_weight_offset, kv_weight_bytes, "kv_rms_weight");
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
extern "C" int ds4_gpu_head_rms_norm_tensor(ds4_gpu_tensor *x, uint32_t n_tok, uint32_t n_head, uint32_t head_dim, float eps) {
    uint64_t rows64 = 0;
    if (!cuda_u64_mul_checked(n_tok, n_head, &rows64) || rows64 > UINT32_MAX ||
        !cuda_tensor_has_elems3(x, n_tok, n_head, head_dim, sizeof(float))) return 0;
    if (rows64 == 0u || head_dim == 0u) return 1;
    head_rms_norm_kernel<<<(uint32_t)rows64, 256>>>((float *)x->ptr, n_tok, n_head, head_dim, eps);
    return cuda_ok(cudaGetLastError(), "head_rms_norm launch");
}
extern "C" int ds4_gpu_head_rms_norm_rope_tail_tensor(ds4_gpu_tensor *x, uint32_t n_tok, uint32_t n_head, uint32_t head_dim, uint32_t n_rot, uint32_t pos0, uint32_t n_ctx_orig, bool inverse, float freq_base, float freq_scale, float ext_factor, float attn_factor, float beta_fast, float beta_slow, float eps) {
    uint64_t rows64 = 0;
    if (n_rot > head_dim || (n_rot & 1u) ||
        !cuda_u64_mul_checked(n_tok, n_head, &rows64) || rows64 > UINT32_MAX ||
        !cuda_tensor_has_elems3(x, n_tok, n_head, head_dim, sizeof(float))) return 0;
    if (rows64 == 0u || head_dim == 0u) return 1;
    head_rms_norm_rope_tail_kernel<<<(uint32_t)rows64, 256>>>((float *)x->ptr, n_tok, n_head, head_dim, n_rot, pos0, n_ctx_orig, inverse ? 1 : 0, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow, eps);
    return cuda_ok(cudaGetLastError(), "head_rms_norm_rope_tail launch");
}

static uint64_t g_rocm_q4_qb_f32_epilogue_calls;
static pthread_once_t g_rocm_q4_qb_f32_epilogue_stats_once = PTHREAD_ONCE_INIT;

static void rocm_q4_qb_f32_epilogue_report(void) {
    fprintf(stderr, DS4_GPU_LOG_PREFIX "Q4 q_b F32 epilogue: wave32_calls=%llu\n",
            (unsigned long long)__atomic_load_n(
                &g_rocm_q4_qb_f32_epilogue_calls, __ATOMIC_RELAXED));
}
static void rocm_q4_qb_f32_epilogue_register_report(void) {
    (void)atexit(rocm_q4_qb_f32_epilogue_report);
}
extern "C" void ds4_rocm_test_q4_qb_f32_epilogue_reset(void) {
    __atomic_store_n(&g_rocm_q4_qb_f32_epilogue_calls, 0u, __ATOMIC_RELAXED);
}
extern "C" uint64_t ds4_rocm_test_q4_qb_f32_epilogue_get_calls(void) {
    return __atomic_load_n(&g_rocm_q4_qb_f32_epilogue_calls, __ATOMIC_RELAXED);
}

static bool rocm_q4_qb_gfx1151_wave32_device(void) {
#if defined(__HIP_PLATFORM_AMD__)
    int device = -1;
    if (cudaGetDevice(&device) != cudaSuccess) return false;
    static thread_local int cached_device = -1;
    static thread_local bool compatible = false;
    if (device != cached_device) {
        cudaDeviceProp prop = {};
        if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) return false;
        compatible = prop.warpSize == 32 &&
            strncmp(prop.gcnArchName, "gfx1151", 7) == 0 &&
            (prop.gcnArchName[7] == '\0' || prop.gcnArchName[7] == ':');
        cached_device = device;
    }
    return compatible;
#else
    return false;
#endif
}

// Invoked only by the Q4 cached/transient F16 GEMMs with F32 output. Do not
// install in the untyped public norm API: that would also change Q8/decode.
static int rocm_q4_qb_f32_epilogue_tensor(
        ds4_gpu_tensor *x, uint32_t n_tok, uint32_t n_head, uint32_t head_dim,
        uint32_t n_rot, uint32_t pos0, uint32_t n_ctx_orig, bool inverse,
        float freq_base, float freq_scale, float ext_factor, float attn_factor,
        float beta_fast, float beta_slow, float eps) {
    namespace ep = ds4_rocm_q4_qb_epilogue;
    if (getenv("DS4_ROCM_Q4_QB_F32_EPILOGUE_STATS") != NULL)
        pthread_once(&g_rocm_q4_qb_f32_epilogue_stats_once,
                     rocm_q4_qb_f32_epilogue_register_report);
    const bool disabled = rocm_q4_attn_q_b_env_bool(
        "DS4_ROCM_DISABLE_Q4_QB_F32_EPILOGUE") == 1;
    if (!ep::select(n_tok, n_head, head_dim, n_rot,
                    !disabled && !g_quality_mode && !g_ssd_streaming_mode &&
                        ep::shape(n_tok, n_head, head_dim, n_rot) &&
                        rocm_q4_qb_gfx1151_wave32_device(),
                    g_quality_mode, g_ssd_streaming_mode, disabled)) {
        return ds4_gpu_head_rms_norm_rope_tail_tensor(
            x, n_tok, n_head, head_dim, n_rot, pos0, n_ctx_orig, inverse,
            freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow, eps);
    }
    if (!cuda_tensor_has_elems3(x, n_tok, n_head, head_dim, sizeof(float))) return 0;
    const uint32_t rows = n_tok * n_head; // Bounded by the strict shape gate.
    rocm_q4_qb_f32_epilogue_wave32_kernel<<<
        (rows + ep::heads_per_block - 1u) / ep::heads_per_block, ep::threads>>>(
            (float *)x->ptr, n_tok, n_head, head_dim, n_rot, pos0, n_ctx_orig,
            inverse ? 1 : 0, freq_base, freq_scale, ext_factor, attn_factor,
            beta_fast, beta_slow, eps);
    const int ok = cuda_ok(cudaGetLastError(), "Q4 q_b F32 wave32 epilogue launch");
    if (ok) __atomic_fetch_add(&g_rocm_q4_qb_f32_epilogue_calls, 1u, __ATOMIC_RELAXED);
    return ok; // Never replay over an accepted or failed in-flight writer.
}

// Narrow native oracle hook: same validation/dispatch as both production sites.
extern "C" int ds4_rocm_test_q4_qb_f32_epilogue(
        ds4_gpu_tensor *x, uint32_t n_tok, uint32_t n_head, uint32_t head_dim,
        uint32_t n_rot, uint32_t pos0, uint32_t n_ctx_orig, bool inverse,
        float freq_base, float freq_scale, float ext_factor, float attn_factor,
        float beta_fast, float beta_slow, float eps) {
    return rocm_q4_qb_f32_epilogue_tensor(
        x, n_tok, n_head, head_dim, n_rot, pos0, n_ctx_orig, inverse,
        freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow, eps);
}

static int rocm_q4_attn_q_b_prefixes_overlap(
        const void *a, uint64_t a_bytes,
        const void *b, uint64_t b_bytes) {
    const uintptr_t ap = reinterpret_cast<uintptr_t>(a);
    const uintptr_t bp = reinterpret_cast<uintptr_t>(b);
    return ap <= bp ? (uint64_t)(bp - ap) < a_bytes
                    : (uint64_t)(ap - bp) < b_bytes;
}

static int rocm_q4_attn_q_b_f16_head_rms_rope_tail_tensor(
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
    /* Decode and tiny/final chunks retain the native Q4_K path without even
     * taking the cache mutex. REQUIRE applies only to configured candidates. */
    if (n_tok < 32u) return 0;
    const int required = rocm_q4_attn_q_b_f16_required();
    if (!rocm_q4_attn_q_b_f16_enabled() && !required) return 0;
    if ((uint64_t)n_tok < rocm_q4_attn_q_b_f16_min_tokens()) return 0;
    rocm_q4_attn_q_b_f16_note_candidate();
    const int f16_output = rocm_q4_attn_q_b_f16_output_enabled();

    uint64_t x_elems = 0;
    uint64_t out_elems = 0;
    uint64_t x_bytes = 0;
    uint64_t out_bytes = 0;
    uint64_t out_f16_bytes = 0;
    uint64_t head_rows = 0;
    if (!rocm_q4_attn_q_b_f16_policy_allowed() ||
        rocm_q4_attn_q_b_f16_circuit_open() ||
        !g_cublas_ready || !out || !out->ptr ||
        !x || !x->ptr || !model_map ||
        model_map != g_model_host_base ||
        model_size != g_model_registered_size ||
        in_dim != DS4_ROCM_Q4_ATTN_Q_B_IN_DIM ||
        out_dim != DS4_ROCM_Q4_ATTN_Q_B_OUT_DIM ||
        n_head == 0u || head_dim == 0u ||
        out_dim != (uint64_t)n_head * head_dim ||
        n_rot > head_dim || (n_rot & 1u) != 0u ||
        n_tok > (uint32_t)INT_MAX ||
        pos0 > (uint32_t)INT_MAX - n_tok ||
        !cuda_u64_mul_checked(n_tok, in_dim, &x_elems) ||
        !cuda_u64_mul_checked(n_tok, out_dim, &out_elems) ||
        !cuda_u64_mul_checked(n_tok, n_head, &head_rows) ||
        head_rows > UINT32_MAX ||
        (x_elems + 255u) / 256u > UINT32_MAX ||
        !cuda_u64_mul_checked(x_elems, sizeof(float), &x_bytes) ||
        !cuda_u64_mul_checked(out_elems, sizeof(float), &out_bytes) ||
        !cuda_u64_mul_checked(out_elems, sizeof(__half), &out_f16_bytes) ||
        x->bytes < x_bytes || out->bytes < out_bytes) {
        return rocm_q4_attn_q_b_f16_fallback(required, 1, 0);
    }

    const uint64_t blocks_per_row = in_dim / CUDA_QK_K;
    uint64_t row_bytes = 0;
    uint64_t weight_bytes = 0;
    if (!cuda_u64_mul_checked(blocks_per_row,
                              sizeof(cuda_block_q4_K), &row_bytes) ||
        !cuda_u64_mul_checked(out_dim, row_bytes, &weight_bytes) ||
        !cuda_model_range_fits(model_size, weight_offset, weight_bytes)) {
        return rocm_q4_attn_q_b_f16_fallback(required, 1, 0);
    }

    __half *unused_weight_f16 = NULL;
    __half *xh = NULL;
    __half *q_scratch = NULL;
    if (!rocm_q4_attn_q_b_transient_f16_acquire(
            n_tok, f16_output,
            &unused_weight_f16, &xh, &q_scratch)) {
        return rocm_q4_attn_q_b_f16_fallback(required, 0, 0);
    }
    (void)unused_weight_f16;
    const int use_graph_q_half =
        f16_output && q_half && q_half->ptr &&
        q_half->bytes >= out_f16_bytes &&
        !rocm_q4_attn_q_b_prefixes_overlap(
            q_half->ptr, out_f16_bytes, out->ptr, out_bytes) &&
        !rocm_q4_attn_q_b_prefixes_overlap(
            q_half->ptr, out_f16_bytes, x->ptr, x_bytes);
    __half *const qh = use_graph_q_half
                     ? (__half *)q_half->ptr : q_scratch;

    const __half *w_f16 = rocm_q4_attn_q_b_f16_acquire(
        model_map, model_size, weight_offset, weight_bytes, in_dim, out_dim,
        DS4_ROCM_Q4_K_TYPE);
    if (!w_f16) {
        rocm_q4_attn_q_b_transient_f16_release_acquired();
        return rocm_q4_attn_q_b_f16_fallback(required, 0, 0);
    }

    /* Keep both the persistent weight pin and the dedicated X/Q staging lock
     * through the complete enqueue sequence. Lifecycle release takes the same
     * locks before synchronizing, so it cannot miss the final epilogue. */
    f32_to_f16_kernel<<<(x_elems + 255u) / 256u, 256>>>(
        xh, (const float *)x->ptr, x_elems);
    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        fprintf(stderr,
                DS4_GPU_LOG_PREFIX
                "Q4 attn q_b F16 activation conversion failed: %s\n",
                cudaGetErrorString(launch_err));
        (void)cudaGetLastError();
        rocm_q4_attn_q_b_f16_release_acquired();
        rocm_q4_attn_q_b_transient_f16_release_acquired();
        return rocm_q4_attn_q_b_f16_fallback(required, 0, 1);
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;
    /* The release path remains F32 by default. The explicit F16-output arm
     * matches Q8's boundary and consumes either caller staging or ROCm-owned
     * Q_F16 without materializing the large F32 Q. */
    const cublasStatus_t st = cublasGemmEx(
        g_cublas,
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
        f16_output ? (void *)qh : out->ptr,
        f16_output ? CUDA_R_16F : CUDA_R_32F,
        (int)out_dim,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT);
    if (st != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr,
                DS4_GPU_LOG_PREFIX
                DS4_GPU_BLAS_NAME
                " cached Q4 attn q_b F16/F16-to-%s matmul failed: "
                "status %d\n",
                f16_output ? "F16" : "F32",
                (int)st);
        rocm_q4_attn_q_b_f16_release_acquired();
        rocm_q4_attn_q_b_transient_f16_release_acquired();
        return rocm_q4_attn_q_b_f16_fallback(required, 0, 1);
    }

    int tail_ok = 0;
    if (f16_output) {
        head_rms_norm_rope_tail_from_half_kernel<<<(uint32_t)head_rows, 256>>>(
                (float *)out->ptr, qh,
                n_tok, n_head, head_dim, n_rot, pos0, n_ctx_orig,
                inverse ? 1 : 0, freq_base, freq_scale, ext_factor,
                attn_factor, beta_fast, beta_slow, eps);
        launch_err = cudaGetLastError();
        tail_ok = launch_err == cudaSuccess;
    } else {
        tail_ok = rocm_q4_qb_f32_epilogue_tensor(
            out, n_tok, n_head, head_dim, n_rot, pos0, n_ctx_orig,
            inverse, freq_base, freq_scale, ext_factor, attn_factor,
            beta_fast, beta_slow, eps);
    }
    rocm_q4_attn_q_b_f16_release_acquired();
    rocm_q4_attn_q_b_transient_f16_release_acquired();
    if (!tail_ok) {
        if (f16_output) {
            fprintf(stderr,
                    DS4_GPU_LOG_PREFIX
                    "cached Q4 attn q_b F16-out epilogue launch failed: %s\n",
                    cudaGetErrorString(launch_err));
            (void)cudaGetLastError();
        }
        /* GEMM has already accepted the output writer.  Never authorize the
         * caller to replay native Q4 over an asynchronous/partial result. */
        return rocm_q4_attn_q_b_f16_fallback(1, 0, 1);
    }
    return 1;
}

/* Resident default: expand only the current Q4_K q_b matrix into the shared
 * 64 MiB W_F16 region, stage X_F16 beside it, and consume both inputs
 * immediately with hipBLAS. The opt-in F16-output arm also stages Q_F16 before
 * its fused epilogue. No caller may reuse the allocation between these steps. */
static int rocm_q4_attn_q_b_transient_f16_head_rms_rope_tail_tensor(
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
    if ((uint64_t)n_tok <
            rocm_q4_attn_q_b_transient_f16_min_tokens()) {
        return 0;
    }
    rocm_q4_attn_q_b_f16_note_candidate();
    const int f16_output = rocm_q4_attn_q_b_f16_output_enabled();

    uint64_t x_elems = 0;
    uint64_t out_elems = 0;
    uint64_t head_rows = 0;
    uint64_t x_bytes = 0;
    uint64_t out_bytes = 0;
    uint64_t out_f16_bytes = 0;
    if (!rocm_q4_attn_q_b_transient_f16_policy_allowed() ||
        rocm_q4_attn_q_b_f16_circuit_open() ||
        !g_cublas_ready || !out || !out->ptr ||
        !x || !x->ptr || !model_map ||
        model_map != g_model_host_base ||
        model_size != g_model_registered_size ||
        in_dim != DS4_ROCM_Q4_ATTN_Q_B_IN_DIM ||
        out_dim != DS4_ROCM_Q4_ATTN_Q_B_OUT_DIM ||
        n_head == 0u || head_dim == 0u ||
        out_dim != (uint64_t)n_head * head_dim ||
        n_rot > head_dim || (n_rot & 1u) != 0u ||
        n_tok > (uint32_t)INT_MAX ||
        pos0 > (uint32_t)INT_MAX - n_tok ||
        !cuda_u64_mul_checked(n_tok, in_dim, &x_elems) ||
        !cuda_u64_mul_checked(n_tok, out_dim, &out_elems) ||
        !cuda_u64_mul_checked(n_tok, n_head, &head_rows) ||
        head_rows > UINT32_MAX ||
        (x_elems + 255u) / 256u > UINT32_MAX ||
        !cuda_u64_mul_checked(x_elems, sizeof(float), &x_bytes) ||
        !cuda_u64_mul_checked(out_elems, sizeof(float), &out_bytes) ||
        !cuda_u64_mul_checked(out_elems, sizeof(__half), &out_f16_bytes) ||
        x->bytes < x_bytes || out->bytes < out_bytes) {
        return rocm_q4_attn_q_b_f16_fallback(0, 1, 0);
    }

    uint64_t row_bytes = 0;
    uint64_t weight_bytes = 0;
    const uint64_t blocks_per_row = in_dim / CUDA_QK_K;
    if (!cuda_u64_mul_checked(blocks_per_row,
                              sizeof(cuda_block_q4_K), &row_bytes) ||
        !cuda_u64_mul_checked(out_dim, row_bytes, &weight_bytes) ||
        !cuda_model_range_fits(model_size, weight_offset, weight_bytes)) {
        return rocm_q4_attn_q_b_f16_fallback(0, 1, 0);
    }

    __half *w_f16 = NULL;
    __half *x_f16 = NULL;
    __half *q_scratch = NULL;
    if (!rocm_q4_attn_q_b_transient_f16_acquire(
            n_tok, f16_output, &w_f16, &x_f16, &q_scratch)) {
        return rocm_q4_attn_q_b_f16_fallback(0, 0, 0);
    }
    const int use_graph_q_half =
        f16_output && q_half && q_half->ptr &&
        q_half->bytes >= out_f16_bytes &&
        !rocm_q4_attn_q_b_prefixes_overlap(
            q_half->ptr, out_f16_bytes, out->ptr, out_bytes) &&
        !rocm_q4_attn_q_b_prefixes_overlap(
            q_half->ptr, out_f16_bytes, x->ptr, x_bytes);
    __half *const qh = use_graph_q_half
                     ? (__half *)q_half->ptr : q_scratch;

    const char *w_q4 = rocm_q4_attn_q_b_device_resident_source(
        model_map, weight_offset, weight_bytes);
    if (!w_q4) {
        rocm_q4_attn_q_b_transient_f16_release_acquired();
        return rocm_q4_attn_q_b_f16_fallback(0, 0, 1);
    }

    const uint64_t total_chunks = out_dim * (in_dim / 16u);
    if (ds4_q4_dequant::select(
            in_dim, out_dim, n_tok, (uintptr_t)w_q4, (uintptr_t)w_f16,
            rocm_q4_qb_gfx1151_wave32_device(), g_quality_mode, g_ssd_streaming_mode,
            getenv("DS4_ROCM_DISABLE_Q4_PREFILL_DEQUANT_VEC") != NULL)) {
        ds4_q4_dequant_f16_vec16_kernel<<<
            (uint32_t)((total_chunks + 255u) / 256u), 256>>>(
                w_f16, (const cuda_block_q4_K *)w_q4, out_dim * blocks_per_row);
    } else {
        rocm_dequant_q4_K_attn_q_b_f16_kernel<<<
            (uint32_t)((total_chunks + 255u) / 256u), 256>>>(
                w_f16, (const cuda_block_q4_K *)w_q4,
                in_dim, out_dim, blocks_per_row);
    }
    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        fprintf(stderr,
                DS4_GPU_LOG_PREFIX
                "Q4 attn_q_b transient dequant launch failed: %s\n",
                cudaGetErrorString(launch_err));
        (void)cudaGetLastError();
        rocm_q4_attn_q_b_transient_f16_release_acquired();
        return rocm_q4_attn_q_b_f16_fallback(0, 0, 1);
    }

    f32_to_f16_kernel<<<(x_elems + 255u) / 256u, 256>>>(
        x_f16, (const float *)x->ptr, x_elems);
    launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        fprintf(stderr,
                DS4_GPU_LOG_PREFIX
                "Q4 attn_q_b transient activation conversion failed: %s\n",
                cudaGetErrorString(launch_err));
        (void)cudaGetLastError();
        rocm_q4_attn_q_b_transient_f16_release_acquired();
        return rocm_q4_attn_q_b_f16_fallback(0, 0, 1);
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;
    /* Keep the release path's F32 output by default. The explicit F16-output
     * experiment uses the same boundary for cached and transient weights. */
    const cublasStatus_t st = cublasGemmEx(
        g_cublas,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        (int)out_dim,
        (int)n_tok,
        (int)in_dim,
        &alpha,
        w_f16,
        CUDA_R_16F,
        (int)in_dim,
        x_f16,
        CUDA_R_16F,
        (int)in_dim,
        &beta,
        f16_output ? (void *)qh : out->ptr,
        f16_output ? CUDA_R_16F : CUDA_R_32F,
        (int)out_dim,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT);
    if (st != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr,
                DS4_GPU_LOG_PREFIX
                DS4_GPU_BLAS_NAME
                " transient Q4 attn q_b F16/F16-to-%s matmul failed: "
                "status %d\n",
                f16_output ? "F16" : "F32",
                (int)st);
        rocm_q4_attn_q_b_transient_f16_release_acquired();
        return rocm_q4_attn_q_b_f16_fallback(0, 0, 1);
    }

    int tail_ok = 0;
    if (f16_output) {
        head_rms_norm_rope_tail_from_half_kernel<<<(uint32_t)head_rows, 256>>>(
                (float *)out->ptr, qh,
                n_tok, n_head, head_dim, n_rot, pos0, n_ctx_orig,
                inverse ? 1 : 0, freq_base, freq_scale, ext_factor,
                attn_factor, beta_fast, beta_slow, eps);
        launch_err = cudaGetLastError();
        tail_ok = launch_err == cudaSuccess;
    } else {
        tail_ok = rocm_q4_qb_f32_epilogue_tensor(
            out, n_tok, n_head, head_dim, n_rot, pos0, n_ctx_orig,
            inverse, freq_base, freq_scale, ext_factor, attn_factor,
            beta_fast, beta_slow, eps);
    }
    rocm_q4_attn_q_b_transient_f16_release_acquired();
    if (!tail_ok) {
        if (f16_output) {
            fprintf(stderr,
                    DS4_GPU_LOG_PREFIX
                    "transient Q4 attn q_b F16-out epilogue launch failed: %s\n",
                    cudaGetErrorString(launch_err));
            (void)cudaGetLastError();
        }
        /* GEMM was accepted and may already be executing.  Returning zero
         * would make the graph replay native Q4 over an in-flight writer. */
        return rocm_q4_attn_q_b_f16_fallback(1, 0, 1);
    }
    return 1;
}

/* A strict Q8_K-wave32 request belongs to the exact Q4 path.  The F16
 * sidecars must yield before validation, allocation, dequantization or GEMM;
 * the caller can then enter the canonical Q4 fallback, whose selector either
 * launches the required quantizer or fails closed on an unsupported device. */
enum {
    ROCM_Q4_ATTN_Q_B_Q8_WAVE32_CONFLICT = -1,
    ROCM_Q4_ATTN_Q_B_Q8_WAVE32_KEEP_F16 = 0,
    ROCM_Q4_ATTN_Q_B_Q8_WAVE32_YIELD = 1,
};

static int rocm_q4_attn_q_b_required_q8_wave32_policy(
        uint32_t weight_type,
        uint32_t n_tok,
        int q8_wave32_required,
        int f16_cache_required) {
    if (weight_type != DS4_ROCM_Q4_K_TYPE || n_tok <= 8u ||
        !q8_wave32_required) {
        return ROCM_Q4_ATTN_Q_B_Q8_WAVE32_KEEP_F16;
    }
    return f16_cache_required
        ? ROCM_Q4_ATTN_Q_B_Q8_WAVE32_CONFLICT
        : ROCM_Q4_ATTN_Q_B_Q8_WAVE32_YIELD;
}

extern "C" int ds4_rocm_test_q4_attn_q_b_yield_to_q8_wave32_policy(
        uint32_t weight_type,
        uint32_t n_tok,
        int q8_wave32_required,
        int f16_cache_required) {
    return rocm_q4_attn_q_b_required_q8_wave32_policy(
        weight_type, n_tok, q8_wave32_required != 0,
        f16_cache_required != 0);
}

extern "C" int ds4_gpu_attn_q_b_f16_head_rms_rope_tail_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *q_half,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint32_t              weight_type,
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
    const int q8_wave32_required = rocm_q4_attn_q_b_env_bool(
        "DS4_ROCM_REQUIRE_Q4_PREFILL_Q8_K_WAVE32") == 1;
    const int f16_cache_required = rocm_q4_attn_q_b_f16_required();
    const int q8_wave32_policy =
        rocm_q4_attn_q_b_required_q8_wave32_policy(
            weight_type, n_tok, q8_wave32_required, f16_cache_required);
    if (q8_wave32_policy == ROCM_Q4_ATTN_Q_B_Q8_WAVE32_CONFLICT) {
        fprintf(stderr,
                DS4_GPU_LOG_PREFIX
                "Q4 attn_q_b prefill cannot require both the F16 cache "
                "and the Q8_K wave32 quantizer\n");
        return -1;
    }
    if (q8_wave32_policy == ROCM_Q4_ATTN_Q_B_Q8_WAVE32_YIELD) {
        return 0;
    }
    if (weight_type == DS4_ROCM_Q4_K_TYPE) {
        const int persistent_requested =
            f16_cache_required ||
            (rocm_q4_attn_q_b_f16_enabled() &&
             !rocm_q4_attn_q_b_f16_disabled());
        if (persistent_requested) {
            return rocm_q4_attn_q_b_f16_head_rms_rope_tail_tensor(
                out, q_half, model_map, model_size, weight_offset,
                in_dim, out_dim, x, n_tok, n_head, head_dim, n_rot, pos0,
                n_ctx_orig, inverse, freq_base, freq_scale, ext_factor,
                attn_factor, beta_fast, beta_slow, eps);
        }
        return rocm_q4_attn_q_b_transient_f16_head_rms_rope_tail_tensor(
            out, q_half, model_map, model_size, weight_offset,
            in_dim, out_dim, x, n_tok, n_head, head_dim, n_rot, pos0,
            n_ctx_orig, inverse, freq_base, freq_scale, ext_factor,
            attn_factor, beta_fast, beta_slow, eps);
    }
    if (weight_type != 8u || !g_cublas_ready || !out || !q_half || !x || !model_map || n_tok == 0 ||
        n_rot > head_dim || (n_rot & 1u) || out_dim != (uint64_t)n_head * head_dim ||
        x->bytes < (uint64_t)n_tok * in_dim * sizeof(float) ||
        out->bytes < (uint64_t)n_tok * out_dim * sizeof(float) ||
        q_half->bytes < (uint64_t)n_tok * out_dim * sizeof(__half)) return 0;
    const uint64_t blocks = (in_dim + 31u) / 32u;
    if (weight_offset > model_size || out_dim > UINT64_MAX / (blocks * 34u)) return 0;
    const uint64_t weight_bytes = out_dim * blocks * 34u;
    if (weight_bytes > model_size - weight_offset) return 0;
    const __half *w_f16 = cuda_q8_f16_ptr(model_map, weight_offset, weight_bytes, in_dim, out_dim, "attn_q_b");
    if (!w_f16) return 0;
    const uint64_t xh_count = (uint64_t)n_tok * in_dim;
    __half *xh = (__half *)cuda_tmp_alloc(xh_count * sizeof(__half), "attn q_b f16 activations");
    if (!xh) return 0;
    f32_to_f16_kernel<<<(xh_count + 255u) / 256u, 256>>>(xh, (const float *)x->ptr, xh_count);
    if (!cuda_ok(cudaGetLastError(), "attn q_b f16 activation convert launch")) return 0;
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
                                     q_half->ptr,
                                     CUDA_R_16F,
                                     (int)out_dim,
                                     CUBLAS_COMPUTE_32F,
                                     CUBLAS_GEMM_DEFAULT);
    if (st != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "ds4: " DS4_GPU_BLAS_NAME " attn q_b f16-out matmul failed: status %d\n", (int)st);
        return 0;
    }
    head_rms_norm_rope_tail_from_half_kernel<<<n_tok * n_head, 256>>>(
            (float *)out->ptr, (const __half *)q_half->ptr, n_tok, n_head, head_dim, n_rot,
            pos0, n_ctx_orig, inverse ? 1 : 0, freq_base, freq_scale, ext_factor, attn_factor,
            beta_fast, beta_slow, eps);
    return cuda_ok(cudaGetLastError(), "attn q_b f16-out head_rms_norm_rope launch");
}

static int cuda_rope_tail_stride_tensor(ds4_gpu_tensor *x, uint32_t n_tok, uint32_t n_head, uint32_t head_dim, uint32_t n_rot, uint32_t pos0, uint32_t pos_stride, uint32_t n_ctx_orig, bool inverse, float freq_base, float freq_scale, float ext_factor, float attn_factor, float beta_fast, float beta_slow) {
    if (!x || n_rot > head_dim || (n_rot & 1) || x->bytes < (uint64_t)n_tok * n_head * head_dim * sizeof(float)) return 0;
    uint32_t pairs = n_tok * n_head * (n_rot / 2);
    rope_tail_kernel<<<(pairs + 255) / 256, 256>>>((float *)x->ptr, n_tok, n_head, head_dim, n_rot, pos0, pos_stride, n_ctx_orig, inverse ? 1 : 0, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
    return cuda_ok(cudaGetLastError(), "rope_tail launch");
}

extern "C" int ds4_gpu_rope_tail_tensor(ds4_gpu_tensor *x, uint32_t n_tok, uint32_t n_head, uint32_t head_dim, uint32_t n_rot, uint32_t pos0, uint32_t n_ctx_orig, bool inverse, float freq_base, float freq_scale, float ext_factor, float attn_factor, float beta_fast, float beta_slow) {
    return cuda_rope_tail_stride_tensor(x, n_tok, n_head, head_dim, n_rot, pos0, 1u, n_ctx_orig, inverse, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
}
