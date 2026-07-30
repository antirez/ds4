#include "pulsar_cuda_internal.h"
#include <cuda_fp8.h>



/* Plain (no-weight) RMSNorm. Input x is ALWAYS an HC residual carrier (the
 * hc_dim-wide flatten before each sublayer / the output head) — every caller
 * feeds cur_hc/after_attn_hc/batch_cur_hc etc. — so x loads through pulsar_hc_load
 * (BF16 storage promoted to f32). Output and the sum-of-squares stay f32. */
/* BIT-EXACT ILP rewrite (2026-07-21).  The decode call sites launch this with
 * grid==1 (one block, 256 threads, n == hc_dim == 16384) -- 1 of 48 SMs, 8
 * warps, and a scalar strided load loop the compiler will not unroll because
 * blockDim.x is a runtime value.  That leaves ~2 loads in flight per warp and
 * makes the kernel pure memory LATENCY.  Templating the block width makes the
 * stride a compile-time constant so the batch below issues UNROLL independent
 * loads before the first FMA.
 *
 * REDUCTION ORDER IS UNCHANGED, WHICH IS THE WHOLE POINT.  Thread t still owns
 * exactly {t, t+BLK, t+2*BLK, ...} and still accumulates them in increasing
 * order; batching UNROLL of those into registers first only reorders the LOADS,
 * never the adds.  The shared-memory pairwise tree is byte-for-byte the old one
 * (no warp shuffles, no split-K): every float op happens in the same sequence,
 * so this is bit-exact by construction, not by measurement. */
template <uint32_t BLK, uint32_t UNROLL>
__global__ static void rms_norm_plain_kernel(float *out, const pulsar_hc_t *x, uint32_t n, uint32_t rows, float eps) {
    uint32_t row = blockIdx.x;
    if (row >= rows) return;
    const pulsar_hc_t *xr = x + (uint64_t)row * n;
    float *orow = out + (uint64_t)row * n;
    const uint32_t tid = threadIdx.x;
    float sum = 0.0f;
    uint32_t i = tid;
    for (; i + (UNROLL - 1u) * BLK < n; i += BLK * UNROLL) {
        float v[UNROLL];
        #pragma unroll
        for (uint32_t u = 0; u < UNROLL; u++) v[u] = pulsar_hc_load(xr, i + u * BLK);
        #pragma unroll
        for (uint32_t u = 0; u < UNROLL; u++) sum += v[u] * v[u];
    }
    for (; i < n; i += BLK) {
        float v = pulsar_hc_load(xr, i);
        sum += v * v;
    }
    __shared__ float partial[BLK];
    partial[tid] = sum;
    __syncthreads();
    for (uint32_t stride = BLK >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        __syncthreads();
    }
    float scale = rsqrtf(partial[0] / (float)n + eps);
    i = tid;
    for (; i + (UNROLL - 1u) * BLK < n; i += BLK * UNROLL) {
        float v[UNROLL];
        #pragma unroll
        for (uint32_t u = 0; u < UNROLL; u++) v[u] = pulsar_hc_load(xr, i + u * BLK);
        #pragma unroll
        for (uint32_t u = 0; u < UNROLL; u++) orow[i + u * BLK] = v[u] * scale;
    }
    for (; i < n; i += BLK) {
        orow[i] = pulsar_hc_load(xr, i) * scale;
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



/* positions (both RoPE kernels): per-row absolute query positions for banked
 * multi-session batches; NULL degenerates to the classic consecutive pos0+t
 * rule bit-exactly (same arithmetic on the same value). */
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
        float eps,
        const int32_t * __restrict__ positions) {
    uint32_t row = blockIdx.x;
    if (row >= n_tok * n_head) return;
    uint32_t t = row / n_head;
    const uint32_t rope_pos = positions ? (uint32_t)positions[t] : pos0 + t;
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
        float theta_extrap = (float)rope_pos * powf(freq_base, -((float)i) / (float)n_rot);
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
        float beta_slow,
        const int32_t * __restrict__ positions) {
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

    const uint32_t rope_pos = positions ? (uint32_t)positions[t] : pos0 + t * pos_stride;
    float theta_extrap = (float)rope_pos * powf(freq_base, -((float)i) / (float)n_rot);
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
    /* Native e4m3 round-trip (cvt.rn.satfinite). PROVEN bit-identical to the
     * former 7-iteration binary search (each step an exp2f) by an exhaustive
     * sweep of all 2^32 finite float bit patterns: 4278190080 checked, zero
     * mismatches, including every RNE tie and the subnormal range
     * (temp/fp8test.cu). NaN inputs differ (search clamped to 448, native
     * propagates NaN) -- activations are finite, never hit. */
    return (float)__nv_fp8_e4m3(x);
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

/* Encode to an OCP E2M1 (float_e2m1_t) 4-bit nibble: [sign:1][magnitude:3].
 * The magnitude table matches dsv4_e2m1fn_value_dev, i.e. CUTLASS float_e2m1_t. */
__device__ static uint8_t dsv4_e2m1fn_encode_dev(float x) {
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
    return (uint8_t)((best & 7) | ((x < 0.0f) ? 0x8u : 0u));
}

__device__ static float dsv4_e2m1fn_decode_dev(uint8_t nib, float scale) {
    float val = dsv4_e2m1fn_value_dev(nib & 7);
    return (nib & 8u) ? (-val * scale) : (val * scale);
}

/* E8M0 microscale: a power-of-two scale stored as a single byte = exponent+127
 * (byte 255 is reserved NaN). Encoding uses dsv4_e8m0_encode_scale_exact_dev
 * below (bit-pattern ceil-log2, fast-math-safe). */
__device__ static float dsv4_e8m0_decode_scale_dev(uint8_t byte) {
    return exp2f((float)((int)byte - 127));
}

/* ===== INVARIANT: FRESH data -> fast path; RE-ENCODED data -> exact path =====
 * The `exp2f(ceilf(log2f(amax/K)))` form used by the quantize kernels below is
 * NOT safe on already-quantized input, and the split between the two forms in
 * this file is DELIBERATE. Do not "simplify" it to one path.
 *
 * VERIFIED on GB10 (2026-07-21, SASS + full-binade sweeps): under
 * --use_fast_math, `log2f` lowers to MUFU.LG2, whose error is one-sided HIGH.
 * `lg2.approx(2^k)` is wrong for EVERY k in [-126,-1] — i.e. the entire range
 * real block scales occupy — so at `amax == K*2^k` exactly, ceilf() rounds up
 * one extra step and the scale comes out 2x TOO LARGE. (`ex2.approx` and
 * `div.approx` were both verified EXACT here; MUFU.LG2 is the sole culprit.
 * Host `-ffast-math` is unaffected: glibc log2f stays exact at pow2.)
 *
 * WHY THE FAST FORM IS FINE HERE: these kernels see FRESH f32 activations, so
 * hitting `amax == K*2^k` is a 1-in-2^23 mantissa coincidence (~2.4e-7/block for
 * K=448, ~1.2e-7 for K=6) — well under the noise floor of the quantization
 * itself, and deterministic, so no bit-exactness gate destabilizes.
 *
 * WHY RE-ENCODING IS CATASTROPHIC: quantized values live on a dyadic LATTICE.
 * After E2M1, block values are {0,.5,1,1.5,2,3,4,6}*scale, so a block whose max
 * sits on the top code has `amax == 6*scale` EXACTLY — a guaranteed hit, and
 * `v/scale` rounds to 6.0 for anything above 5.0, so ~1/3 of re-encoded FP4
 * blocks land on it (~5% for E4M3's 448). A competing GB10 fork shipped exactly
 * this bug and lost 10.5% of their fp4 indexer lanes to zero.
 *
 * SO: every re-encode path already routes through the EXACT helper below —
 * `mxkv_pack_kernel`, `attn_pack_repack_kernel`/`attn_pack_exact_e8_dev` — and
 * `gpu_decode.c:498` passes `quantize_fp8=false` under pack to avoid double-
 * rounding. **If you ever feed already-quantized data into
 * `fp8_kv_quantize*`, `attn_pack_store`, or `indexer_hadamard_fp4*`, the misround
 * rate jumps from 1e-7 to ~5% (E4M3) or ~33% (FP4) instantly.** The plan-32
 * evict/restore and plan-33 fork/restore work is precisely the kind of change
 * that could do this: restore paths MUST keep using the exact repack entry
 * points, never these fresh quantizers.
 *
 * KNOWN (benign, unfixed): `indexer_hadamard_fp4*_kernel`'s amax floor
 * 7.052966104933725e-38f is bit-exactly `6.0f * 2^-126` — i.e. it sits ON a
 * misround point, so an all-zero/all-subnormal indexer block takes e8=2 here
 * where the exact path gives e8=1, 100% of the time. Value impact NIL (every
 * lane encodes to nibble 0 either way), but the stored E8M0 BYTE differs from
 * what mxkv_pack writes for the same row — a latent flake for any gate that
 * byte-compares those two paths on an empty row. Left as-is because fixing it
 * is bit-changing against current goldens for zero value gain; fold it in next
 * time goldens are re-baselined anyway. (The E4M3 floor 1.0e-4f was checked and
 * is NOT on a misround point.)
 * ============================================================================ */

/* Exact ceil-log2 E8M0 bucket computed from float bit patterns — no log2f/ceilf,
 * so no --use_fast_math misrounding at bucket boundaries. For a max_repr equal to
 * M_m*2^(E_m-127), the smallest 2^k with max_repr*2^k >= amax has exponent field
 * (E_m+k) and mantissa M_m; comparing positive floats is comparing their bit
 * patterns, so k = exp(amax) - E_m + (mant(amax) > M_m). Re-encoding rows that are
 * already on the (grid * 2^m) lattice is then value-idempotent. */
__device__ static inline uint8_t dsv4_e8m0_encode_scale_exact_dev(float amax, float max_repr, float floor_val) {
    const float a = fmaxf(amax, floor_val);
    const uint32_t u  = __float_as_uint(a);
    const uint32_t um = __float_as_uint(max_repr);
    int e = (int)(u >> 23) - (int)(um >> 23)
          + (((u & 0x7fffffu) > (um & 0x7fffffu)) ? 1 : 0) + 127;
    if (e < 0) e = 0;
    if (e > 254) e = 254;
    return (uint8_t)e;
}



__device__ static float model_scalar_dev(const void *base, uint64_t offset, uint32_t type, uint64_t idx) {
    const char *p = (const char *)base + offset;
    if (type == 1u) return __half2float(((const __half *)p)[idx]);
    return ((const float *)p)[idx];
}









__device__ static uint8_t dsv4_e4m3fn_encode_dev(float x) {
    /* Native e4m3 encode: the former (exp<<3)|mant index IS the e4m3fn bit
     * pattern (sign in 0x80), so the hardware cvt byte is the same encoding.
     * Bit-identity proven by the exhaustive round-trip sweep (see
     * dsv4_e4m3fn_dequant_dev). */
    __nv_fp8_e4m3 f(x);
    return *(const uint8_t *)&f;
}

__device__ static float dsv4_e4m3fn_decode_dev(uint8_t byte, float scale) {
    int idx = byte & 0x7f;
    float val = dsv4_e4m3fn_value_dev(idx);
    return (byte & 0x80) ? (-val * scale) : (val * scale);
}

#define PULSAR_FP8_KV_BLOCK 64u
#define PULSAR_FP8_KV_NBLK(HD) (((HD) + PULSAR_FP8_KV_BLOCK - 1u) / PULSAR_FP8_KV_BLOCK)
#define PULSAR_FP8_KV_ROWBYTES(HD) ((HD) + PULSAR_FP8_KV_NBLK(HD) * sizeof(float))

__global__ static void pack_fp8_kv_kernel(const float *x, uint8_t *packed, float *scales, uint32_t n_tok, uint32_t head_dim) {
    uint32_t row = blockIdx.x;
    uint32_t tid = threadIdx.x;
    if (row >= n_tok) return;
    const float *xr = x + (uint64_t)row * head_dim;
    uint8_t *pr = packed + (uint64_t)row * head_dim;
    float *sr = scales + (uint64_t)row * PULSAR_FP8_KV_NBLK(head_dim);
    __shared__ float scratch[64];
    for (uint32_t off = 0; off < head_dim; off += PULSAR_FP8_KV_BLOCK) {
        float v = 0.0f;
        if (off + tid < head_dim) v = xr[off + tid];
        scratch[tid] = off + tid < head_dim ? fabsf(v) : 0.0f;
        __syncthreads();
        for (uint32_t stride = 32; stride > 0; stride >>= 1) {
            if (tid < stride) scratch[tid] = fmaxf(scratch[tid], scratch[tid + stride]);
            __syncthreads();
        }
        float scale = exp2f(ceilf(log2f(fmaxf(scratch[0], 1.0e-4f) / 448.0f)));
        uint32_t blk = off / PULSAR_FP8_KV_BLOCK;
        if (tid == 0) sr[blk] = scale;
        if (off + tid < head_dim) {
            pr[off + tid] = dsv4_e4m3fn_encode_dev(fminf(448.0f, fmaxf(-448.0f, v / scale)));
        }
        __syncthreads();
    }
}

/* Fused decode KV store: fake-quant the nope dims of the single kv row IN
 * PLACE (same 64-thread block, same reduction and scale math as
 * fp8_kv_quantize_kernel -> bit-identical values land in x), and write the
 * whole row into the raw cache with store_raw_kv_batch_kernel's exact
 * __float2half roundtrip semantics. One launch instead of two per layer per
 * decode token. */
__global__ static void fp8_kv_quantize_store_kernel(
        float *x, float *raw, uint32_t raw_cap, uint32_t pos,
        uint32_t head_dim, uint32_t n_rot, int raw_f16) {
    const uint32_t tid = threadIdx.x;
    const uint32_t n_nope = head_dim - n_rot;
    const uint32_t row = pos % raw_cap;
    __shared__ float scratch[64];
    for (uint32_t off = 0; off < n_nope; off += 64) {
        float v = 0.0f;
        if (off + tid < n_nope) v = x[off + tid];
        scratch[tid] = off + tid < n_nope ? fabsf(v) : 0.0f;
        __syncthreads();
        for (uint32_t stride = 32; stride > 0; stride >>= 1) {
            if (tid < stride) scratch[tid] = fmaxf(scratch[tid], scratch[tid + stride]);
            __syncthreads();
        }
        float scale = exp2f(ceilf(log2f(fmaxf(scratch[0], 1.0e-4f) / 448.0f)));
        if (off + tid < n_nope) {
            float q = dsv4_e4m3fn_dequant_dev(fminf(448.0f, fmaxf(-448.0f, v / scale))) * scale;
            x[off + tid] = q;
            const __half h = __float2half(q);
            if (raw_f16) ((__half *)raw)[(uint64_t)row * head_dim + off + tid] = h;
            else         raw[(uint64_t)row * head_dim + off + tid] = __half2float(h);
        }
        __syncthreads();
    }
    for (uint32_t d = n_nope + tid; d < head_dim; d += 64) {
        const __half h = __float2half(x[d]);
        if (raw_f16) ((__half *)raw)[(uint64_t)row * head_dim + d] = h;
        else         raw[(uint64_t)row * head_dim + d] = __half2float(h);
    }
}

__global__ static void fp8_kv_quantize_kernel(float *x, uint32_t n_tok, uint32_t head_dim, uint32_t n_rot) {
    uint32_t row = blockIdx.x;
    uint32_t tid = threadIdx.x;
    uint32_t n_nope = head_dim - n_rot;
    float *xr = x + (uint64_t)row * head_dim;
    __shared__ float scratch[64];
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

/* PULSAR_ATTN_PACK store: quantize the nope dims of n_rows f32 rows of x with
 * EXACTLY the fp8_kv_quantize_kernel recipe (same reduction, same scale
 * formula, same clamp/roundtrip), write the roundtripped f32 back into x (so
 * the stage/dumps show the same values the f32 pipeline produces), and store
 * the packed rows (see PULSAR_ATTN_PACK_* in pulsar_cuda_internal.h; 712 B at
 * head_dim 512) into `out` at rows [out_row0, out_row0+n_rows).  The rope tail
 * is copied f32 untouched.  Read-back is bit-identical to the f32 path. */
__global__ static void attn_pack_store_kernel(float *x, uint8_t *out, uint32_t out_row0, uint32_t n_rows, uint32_t head_dim, uint32_t n_rot) {
    uint32_t row = blockIdx.x;
    uint32_t tid = threadIdx.x;
    if (row >= n_rows) return;
    const uint32_t n_nope = head_dim - n_rot;
    const uint32_t nblk = n_nope / PULSAR_FP8_KV_BLOCK;
    const uint32_t nblk_pad = (nblk + 3u) & ~3u;
    const uint64_t rowbytes = PULSAR_ATTN_PACK_ROWBYTES(head_dim);
    float *xr = x + (uint64_t)row * head_dim;
    uint8_t *outr = out + (uint64_t)(out_row0 + row) * rowbytes;
    uint8_t *sc = outr + n_nope;
    float *rope = (float *)(outr + n_nope + nblk_pad);
    __shared__ float scratch[64];
    for (uint32_t off = 0; off < n_nope; off += PULSAR_FP8_KV_BLOCK) {
        float v = 0.0f;
        if (off + tid < n_nope) v = xr[off + tid];
        scratch[tid] = off + tid < n_nope ? fabsf(v) : 0.0f;
        __syncthreads();
        for (uint32_t stride = 32; stride > 0; stride >>= 1) {
            if (tid < stride) scratch[tid] = fmaxf(scratch[tid], scratch[tid + stride]);
            __syncthreads();
        }
        const float lg = ceilf(log2f(fmaxf(scratch[0], 1.0e-4f) / 448.0f));
        const float scale = exp2f(lg);
        if (tid == 0) {
            int e = (int)lg + 127;
            if (e < 0) e = 0;
            if (e > 254) e = 254;
            sc[off / PULSAR_FP8_KV_BLOCK] = (uint8_t)e;
        }
        if (off + tid < n_nope) {
            const float c = fminf(448.0f, fmaxf(-448.0f, v / scale));
            xr[off + tid] = dsv4_e4m3fn_dequant_dev(c) * scale;
            outr[off + tid] = dsv4_e4m3fn_encode_dev(c);
        }
        __syncthreads();
    }
    if (tid == 0) {
        for (uint32_t p = nblk; p < nblk_pad; p++) sc[p] = 0;
    }
    for (uint32_t d = tid; d < n_rot; d += blockDim.x) rope[d] = xr[n_nope + d];
}

/* PULSAR_ATTN_PACK dequant: packed rows -> f32 rows (nope = e4m3 value *
 * 2^(e8-127), rope = f32 direct).  Bit-identical to the f32 cache values. */
__global__ static void attn_pack_dequant_kernel(const uint8_t *in, float *out, uint32_t n_rows, uint32_t head_dim, uint32_t n_rot) {
    uint32_t row = blockIdx.x;
    if (row >= n_rows) return;
    const uint32_t n_nope = head_dim - n_rot;
    const uint32_t nblk_pad = (n_nope / PULSAR_FP8_KV_BLOCK + 3u) & ~3u;
    const uint64_t rowbytes = PULSAR_ATTN_PACK_ROWBYTES(head_dim);
    const uint8_t *inr = in + (uint64_t)row * rowbytes;
    const uint8_t *sc = inr + n_nope;
    const float *rope = (const float *)(inr + n_nope + nblk_pad);
    float *outr = out + (uint64_t)row * head_dim;
    for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        if (d < n_nope) {
            outr[d] = dsv4_e4m3fn_decode_dev(inr[d],
                                             dsv4_e8m0_decode_scale_dev(sc[d / PULSAR_FP8_KV_BLOCK]));
        } else {
            outr[d] = rope[d - n_nope];
        }
    }
}

/* Exact e4m3 (max_repr 448) scale bucket for repacking already-roundtripped
 * attn KV rows; see dsv4_e8m0_encode_scale_exact_dev for the bit-pattern trick. */
__device__ static inline uint8_t attn_pack_exact_e8_dev(float amax) {
    return dsv4_e8m0_encode_scale_exact_dev(amax, 448.0f, 1.0e-4f);
}

__global__ static void attn_pack_repack_kernel(const float *x, uint8_t *out, uint32_t out_row0, uint32_t n_rows, uint32_t head_dim, uint32_t n_rot) {
    uint32_t row = blockIdx.x;
    uint32_t tid = threadIdx.x;
    if (row >= n_rows) return;
    const uint32_t n_nope = head_dim - n_rot;
    const uint32_t nblk = n_nope / PULSAR_FP8_KV_BLOCK;
    const uint32_t nblk_pad = (nblk + 3u) & ~3u;
    const uint64_t rowbytes = PULSAR_ATTN_PACK_ROWBYTES(head_dim);
    const float *xr = x + (uint64_t)row * head_dim;
    uint8_t *outr = out + (uint64_t)(out_row0 + row) * rowbytes;
    uint8_t *sc = outr + n_nope;
    float *rope = (float *)(outr + n_nope + nblk_pad);
    __shared__ float scratch[64];
    for (uint32_t off = 0; off < n_nope; off += PULSAR_FP8_KV_BLOCK) {
        float v = 0.0f;
        if (off + tid < n_nope) v = xr[off + tid];
        scratch[tid] = off + tid < n_nope ? fabsf(v) : 0.0f;
        __syncthreads();
        for (uint32_t stride = 32; stride > 0; stride >>= 1) {
            if (tid < stride) scratch[tid] = fmaxf(scratch[tid], scratch[tid + stride]);
            __syncthreads();
        }
        const uint8_t e8 = attn_pack_exact_e8_dev(scratch[0]);
        const float scale = dsv4_e8m0_decode_scale_dev(e8);
        if (tid == 0) sc[off / PULSAR_FP8_KV_BLOCK] = e8;
        if (off + tid < n_nope) {
            outr[off + tid] = dsv4_e4m3fn_encode_dev(fminf(448.0f, fmaxf(-448.0f, v / scale)));
        }
        __syncthreads();
    }
    if (tid == 0) {
        for (uint32_t p = nblk; p < nblk_pad; p++) sc[p] = 0;
    }
    for (uint32_t d = tid; d < n_rot; d += blockDim.x) rope[d] = xr[n_nope + d];
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

/* Same QAT transform as indexer_hadamard_fp4_kernel (bit-identical f32 result
 * written back to x), additionally emitting the row in MXKV FP4 layout — E2M1
 * nibble pairs low-nibble-first followed by one E8M0 byte per 32-block — so a
 * packed indexer cache stores exactly the values the f32 path would.  The E8M0
 * exponent clamp only differs from the unpacked path outside [2^-127, 2^127]
 * scales, which the 7e-38 amax floor already makes unreachable. */
__global__ static void indexer_hadamard_fp4_pack_kernel(float *x, uint8_t *out,
                                                        uint32_t n_rows, uint32_t head_dim) {
    uint32_t row = blockIdx.x;
    uint32_t tid = threadIdx.x;
    if (row >= n_rows || head_dim != 128u || tid >= 128u) return;

    __shared__ float vals[128];
    __shared__ float absbuf[128];
    __shared__ uint8_t nib_sh[128];
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
    int e8 = (int)ceilf(log2f(amax / 6.0f)) + 127;
    e8 = e8 < 0 ? 0 : (e8 > 254 ? 254 : e8);
    float scale = exp2f((float)(e8 - 127));
    uint8_t nib = dsv4_e2m1fn_encode_dev(fminf(6.0f, fmaxf(-6.0f, v / scale)));
    xr[tid] = dsv4_e2m1fn_decode_dev(nib, scale);
    nib_sh[tid] = nib;
    __syncthreads();

    uint8_t *outr = out + (uint64_t)row * PULSAR_MXKV_FP4_ROWBYTES(128u);
    if (tid < 64u) outr[tid] = (uint8_t)(nib_sh[2u * tid] | (nib_sh[2u * tid + 1u] << 4));
    if (lane == 0u) outr[64u + fp4_block] = (uint8_t)e8;
}



/* raw_f16: per-call flag describing the storage format of the PASSED raw
 * cache (__half when set, f32 otherwise).  The value stored is the same
 * __float2half rounding the f32 path roundtrips through, so read-back (as
 * __half2float) is bit-identical in both modes.
 *
 * positions/seq_id/n_banks: per-row multi-session banking (same descriptor
 * contract as the decode attention kernels, pulsar_cuda_attention.cu).  Row t
 * stores to bank seq_id[t]'s ring at slot positions[t] % raw_cap — the ring
 * is position-indexed, so one scatter launch stores all rows across all
 * banks.  Dead rows (out-of-pool seq_id) store NOTHING: a wild id must not
 * scribble another bank's ring (the read-side dead-row guard zeroes the
 * row's outputs).  NULL/NULL/1 degenerates to the classic single-cache
 * consecutive store bit-exactly. */
__global__ static void store_raw_kv_batch_kernel(float *raw, const float *kv, uint32_t raw_cap, uint32_t pos0, uint32_t n_tokens, uint32_t head_dim, int raw_f16,
                                                 const int32_t * __restrict__ positions,
                                                 const int32_t * __restrict__ seq_id,
                                                 uint32_t n_banks) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_tokens * head_dim;
    if (gid >= n) return;
    uint32_t d = gid % head_dim;
    uint32_t t = gid / head_dim;
    if (seq_id && (uint32_t)seq_id[t] >= n_banks) return;
    const uint32_t p = positions ? (uint32_t)positions[t] : pos0 + t;
    uint32_t row = (seq_id ? (uint32_t)seq_id[t] * raw_cap : 0u) + p % raw_cap;
    const __half h = __float2half(kv[(uint64_t)t * head_dim + d]);
    if (raw_f16) ((__half *)raw)[(uint64_t)row * head_dim + d] = h;
    else         raw[(uint64_t)row * head_dim + d] = __half2float(h);
}



__global__ static void fill_f32_kernel(float *x, uint64_t n, float v) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = v;
}



__global__ static void compressor_store_kernel(
        const float *kv,
        const float *sc,
        float *state_kv,
        float *state_score,
        const void *model_map,
        uint64_t ape_offset,
        uint32_t ape_type,
        uint32_t head_dim,
        uint32_t ratio,
        uint32_t pos0,
        uint32_t n_tokens) {
    uint32_t coff = ratio == 4u ? 2u : 1u;
    uint32_t width = coff * head_dim;
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_tokens * width;
    if (gid >= n) return;
    uint32_t t = gid / width;
    uint32_t j = gid - (uint64_t)t * width;
    uint32_t pos_mod = (pos0 + t) % ratio;
    uint32_t dst_row = ratio == 4u ? ratio + pos_mod : pos_mod;
    state_kv[(uint64_t)dst_row * width + j] = kv[(uint64_t)t * width + j];
    state_score[(uint64_t)dst_row * width + j] =
        sc[(uint64_t)t * width + j] + model_scalar_dev(model_map, ape_offset, ape_type, (uint64_t)pos_mod * width + j);
}



__global__ static void compressor_set_rows_kernel(
        float *state_kv,
        float *state_score,
        const float *kv,
        const float *sc,
        const void *model_map,
        uint64_t ape_offset,
        uint32_t ape_type,
        uint32_t width,
        uint32_t ratio,
        uint32_t pos0,
        uint32_t src0,
        uint32_t dst0,
        uint32_t rows) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)rows * width;
    if (gid >= n) return;
    uint32_t r = gid / width;
    uint32_t j = gid - (uint64_t)r * width;
    uint32_t src = src0 + r;
    uint32_t dst = dst0 + r;
    uint32_t phase = (pos0 + src) % ratio;
    state_kv[(uint64_t)dst * width + j] = kv[(uint64_t)src * width + j];
    state_score[(uint64_t)dst * width + j] =
        sc[(uint64_t)src * width + j] + model_scalar_dev(model_map, ape_offset, ape_type, (uint64_t)phase * width + j);
}



__global__ static void compressor_prefill_pool_kernel(
        float *comp,
        const float *kv,
        const float *sc,
        const float *state_kv,
        const float *state_score,
        const void *model_map,
        uint64_t ape_offset,
        uint32_t ape_type,
        uint32_t head_dim,
        uint32_t ratio,
        uint32_t pos0,
        uint32_t n_comp,
        uint32_t replay) {
    uint32_t d = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t c = blockIdx.y;
    if (d >= head_dim || c >= n_comp) return;
    uint32_t coff = ratio == 4u ? 2u : 1u;
    uint32_t width = coff * head_dim;
    float vals[128];
    float scores[128];
    float max_s = -INFINITY;
    uint32_t n_cand = 0;
    if (ratio == 4u) {
        if (replay && c == 0) {
            for (uint32_t r = 0; r < 4; r++) {
                vals[n_cand] = state_kv[(uint64_t)r * width + d];
                scores[n_cand] = state_score[(uint64_t)r * width + d];
                max_s = fmaxf(max_s, scores[n_cand++]);
            }
        } else if (c > 0) {
            uint32_t base = (c - 1u) * ratio;
            for (uint32_t r = 0; r < 4; r++) {
                uint32_t t = base + r;
                float ape = model_scalar_dev(model_map, ape_offset, ape_type, (uint64_t)((pos0 + t) % ratio) * width + d);
                vals[n_cand] = kv[(uint64_t)t * width + d];
                scores[n_cand] = sc[(uint64_t)t * width + d] + ape;
                max_s = fmaxf(max_s, scores[n_cand++]);
            }
        }
        uint32_t base = c * ratio;
        for (uint32_t r = 0; r < 4; r++) {
            uint32_t t = base + r;
            float ape = model_scalar_dev(model_map, ape_offset, ape_type, (uint64_t)((pos0 + t) % ratio) * width + head_dim + d);
            vals[n_cand] = kv[(uint64_t)t * width + head_dim + d];
            scores[n_cand] = sc[(uint64_t)t * width + head_dim + d] + ape;
            max_s = fmaxf(max_s, scores[n_cand++]);
        }
    } else {
        uint32_t base = c * ratio;
        for (uint32_t r = 0; r < ratio; r++) {
            uint32_t t = base + r;
            float ape = model_scalar_dev(model_map, ape_offset, ape_type, (uint64_t)((pos0 + t) % ratio) * width + d);
            vals[n_cand] = kv[(uint64_t)t * width + d];
            scores[n_cand] = sc[(uint64_t)t * width + d] + ape;
            max_s = fmaxf(max_s, scores[n_cand++]);
        }
    }
    float den = 0.0f, acc = 0.0f;
    for (uint32_t i = 0; i < n_cand; i++) {
        float w = expf(scores[i] - max_s);
        den += w;
        acc += vals[i] * w;
    }
    comp[(uint64_t)c * head_dim + d] = den != 0.0f ? acc / den : 0.0f;
}



__global__ static void compressor_update_pool_kernel(
        float *row,
        const float *state_kv,
        const float *state_score,
        uint32_t head_dim,
        uint32_t ratio) {
    uint32_t d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= head_dim) return;
    uint32_t coff = ratio == 4u ? 2u : 1u;
    uint32_t width = coff * head_dim;
    float vals[128];
    float scores[128];
    float max_s = -INFINITY;
    uint32_t n_cand = 0;
    if (ratio == 4u) {
        for (uint32_t r = 0; r < 4; r++) {
            vals[n_cand] = state_kv[(uint64_t)r * width + d];
            scores[n_cand] = state_score[(uint64_t)r * width + d];
            max_s = fmaxf(max_s, scores[n_cand++]);
        }
        for (uint32_t r = 0; r < 4; r++) {
            vals[n_cand] = state_kv[(uint64_t)(ratio + r) * width + head_dim + d];
            scores[n_cand] = state_score[(uint64_t)(ratio + r) * width + head_dim + d];
            max_s = fmaxf(max_s, scores[n_cand++]);
        }
    } else {
        for (uint32_t r = 0; r < ratio; r++) {
            vals[n_cand] = state_kv[(uint64_t)r * width + d];
            scores[n_cand] = state_score[(uint64_t)r * width + d];
            max_s = fmaxf(max_s, scores[n_cand++]);
        }
    }
    float den = 0.0f, acc = 0.0f;
    for (uint32_t i = 0; i < n_cand; i++) {
        float w = expf(scores[i] - max_s);
        den += w;
        acc += vals[i] * w;
    }
    row[d] = den != 0.0f ? acc / den : 0.0f;
}



__global__ static void compressor_shift_ratio4_kernel(float *state_kv, float *state_score, uint32_t width) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t half = 4ull * width;
    if (i >= half) return;
    float v = state_kv[half + i];
    float s = state_score[half + i];
    state_kv[i] = v;
    state_score[i] = s;
    state_kv[half + i] = v;
    state_score[half + i] = s;
}



int pulsar_gpu_rms_norm_plain_tensor(pulsar_gpu_tensor *out, const pulsar_gpu_tensor *x, uint32_t n, float eps) {
    if (!out || !x || out->bytes < (uint64_t)n * sizeof(float) ||
        x->bytes < (uint64_t)n * PULSAR_HC_ELT_SIZE) return 0;   /* x is an HC residual carrier */
    rms_norm_plain_kernel<256, 8><<<1, 256>>>((float *)out->ptr, (const pulsar_hc_t *)x->ptr, n, 1, eps);
    return cuda_ok(cudaGetLastError(), "rms_norm_plain launch");
}


int pulsar_gpu_rms_norm_plain_rows_tensor(pulsar_gpu_tensor *out, const pulsar_gpu_tensor *x, uint32_t n, uint32_t rows, float eps) {
    if (!out || !x || out->bytes < (uint64_t)n * rows * sizeof(float) ||
        x->bytes < (uint64_t)n * rows * PULSAR_HC_ELT_SIZE) return 0;   /* x is an HC residual carrier */
    rms_norm_plain_kernel<256, 8><<<rows, 256>>>((float *)out->ptr, (const pulsar_hc_t *)x->ptr, n, rows, eps);
    return cuda_ok(cudaGetLastError(), "rms_norm_plain launch");
}


int pulsar_gpu_rms_norm_weight_tensor(pulsar_gpu_tensor *out, const pulsar_gpu_tensor *x, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint32_t n, float eps) {
    if (!out || !x || !model_map || weight_offset > model_size ||
        model_size - weight_offset < (uint64_t)n * sizeof(float) ||
        out->bytes < (uint64_t)n * sizeof(float) ||
        x->bytes < (uint64_t)n * sizeof(float)) return 0;
    const char *wptr = cuda_model_range_ptr(model_map, weight_offset, (uint64_t)n * sizeof(float), "rms_weight");
    if (!wptr) return 0;
    const float *w = (const float *)wptr;
    rms_norm_weight_kernel<<<1, 256>>>((float *)out->ptr, (const float *)x->ptr, w, n, 1, eps);
    return cuda_ok(cudaGetLastError(), "rms_norm_weight launch");
}


int pulsar_gpu_rms_norm_weight_rows_tensor(pulsar_gpu_tensor *out, const pulsar_gpu_tensor *x, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint32_t n, uint32_t rows, float eps) {
    if (!out || !x || !model_map || weight_offset > model_size ||
        model_size - weight_offset < (uint64_t)n * sizeof(float) ||
        out->bytes < (uint64_t)n * rows * sizeof(float) ||
        x->bytes < (uint64_t)n * rows * sizeof(float)) return 0;
    const char *wptr = cuda_model_range_ptr(model_map, weight_offset, (uint64_t)n * sizeof(float), "rms_weight");
    if (!wptr) return 0;
    const float *w = (const float *)wptr;
    rms_norm_weight_kernel<<<rows, 256>>>((float *)out->ptr, (const float *)x->ptr, w, n, rows, eps);
    return cuda_ok(cudaGetLastError(), "rms_norm_weight launch");
}


int pulsar_gpu_dsv4_qkv_rms_norm_rows_tensor(
        pulsar_gpu_tensor       *q_out,
        const pulsar_gpu_tensor *q,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                q_weight_offset,
        uint32_t                q_n,
        pulsar_gpu_tensor       *kv_out,
        const pulsar_gpu_tensor *kv,
        uint64_t                kv_weight_offset,
        uint32_t                kv_n,
        uint32_t                rows,
        float                   eps) {
    static const int disable_qkv_rms_fused = getenv("PULSAR_CUDA_DISABLE_QKV_RMS_FUSED") != NULL;
    if (!disable_qkv_rms_fused) {
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
        const float *q_w = (const float *)cuda_model_range_ptr(model_map,
                q_weight_offset, (uint64_t)q_n * sizeof(float), "q_rms_weight");
        const float *kv_w = (const float *)cuda_model_range_ptr(model_map,
                kv_weight_offset, (uint64_t)kv_n * sizeof(float), "kv_rms_weight");
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
    return pulsar_gpu_rms_norm_weight_rows_tensor(q_out, q, model_map, model_size,
                                                 q_weight_offset, q_n, rows, eps) &&
           pulsar_gpu_rms_norm_weight_rows_tensor(kv_out, kv, model_map, model_size,
                                                 kv_weight_offset, kv_n, rows, eps);
}


int pulsar_gpu_head_rms_norm_tensor(pulsar_gpu_tensor *x, uint32_t n_tok, uint32_t n_head, uint32_t head_dim, float eps) {
    if (!x || x->bytes < (uint64_t)n_tok * n_head * head_dim * sizeof(float)) return 0;
    head_rms_norm_kernel<<<n_tok * n_head, 256>>>((float *)x->ptr, n_tok, n_head, head_dim, eps);
    return cuda_ok(cudaGetLastError(), "head_rms_norm launch");
}


int pulsar_gpu_head_rms_norm_rope_tail_tensor(pulsar_gpu_tensor *x, uint32_t n_tok, uint32_t n_head, uint32_t head_dim, uint32_t n_rot, uint32_t pos0, uint32_t n_ctx_orig, bool inverse, float freq_base, float freq_scale, float ext_factor, float attn_factor, float beta_fast, float beta_slow, float eps, const pulsar_gpu_tensor *positions) {
    if (positions && positions->bytes < (uint64_t)n_tok * sizeof(int32_t)) return 0;
    if (!x || n_rot > head_dim || (n_rot & 1u) ||
        x->bytes < (uint64_t)n_tok * n_head * head_dim * sizeof(float)) return 0;
    head_rms_norm_rope_tail_kernel<<<n_tok * n_head, 256>>>((float *)x->ptr, n_tok, n_head, head_dim, n_rot, pos0, n_ctx_orig, inverse ? 1 : 0, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow, eps, positions ? (const int32_t *)positions->ptr : NULL);
    return cuda_ok(cudaGetLastError(), "head_rms_norm_rope_tail launch");
}



int pulsar_gpu_dsv4_fp8_kv_quantize_tensor(pulsar_gpu_tensor *x, uint32_t n_tok, uint32_t head_dim, uint32_t n_rot) {
    if (!x || n_rot > head_dim || x->bytes < (uint64_t)n_tok * head_dim * sizeof(float)) return 0;
    fp8_kv_quantize_kernel<<<n_tok, 64>>>((float *)x->ptr, n_tok, head_dim, n_rot);
    return cuda_ok(cudaGetLastError(), "fp8_kv_quantize launch");
}

int pulsar_gpu_dsv4_fp8_kv_pack_tensor(
        const pulsar_gpu_tensor *x,
        pulsar_gpu_tensor       *packed,
        pulsar_gpu_tensor       *scales,
        uint32_t               n_tok,
        uint32_t               head_dim) {
    uint32_t nblk = PULSAR_FP8_KV_NBLK(head_dim);
    if (!x || !packed || !scales || n_tok == 0 ||
        x->bytes < (uint64_t)n_tok * head_dim * sizeof(float) ||
        packed->bytes < (uint64_t)n_tok * head_dim ||
        scales->bytes < (uint64_t)n_tok * nblk * sizeof(float)) return 0;
    pack_fp8_kv_kernel<<<n_tok, 64>>>((const float *)x->ptr, (uint8_t *)packed->ptr, (float *)scales->ptr, n_tok, head_dim);
    return cuda_ok(cudaGetLastError(), "fp8_kv_pack launch");
}


/*
 * Microscaling (MX) compressed-KV pack: one warp per row, PULSAR_MXKV_BLOCK=32
 * elements per E8M0 scale.  Row layout is [data ...][E8M0 scale bytes ...]:
 * MXFP8 = E4M3 data (1 B/elem); MXFP4 = E2M1 data (2 elems/byte).  head_dim
 * must be a multiple of 32.  See PULSAR_MXKV_* in pulsar_cuda_internal.h.
 */
__global__ static void mxkv_pack_kernel(const float *x, uint8_t *out,
                                        uint32_t n_tok, uint32_t head_dim, uint32_t fmt) {
    uint32_t row = blockIdx.x;
    uint32_t lane = threadIdx.x;                 /* 0..31, one warp */
    if (row >= n_tok) return;
    const float *xr = x + (uint64_t)row * head_dim;
    const uint32_t nblk = head_dim / PULSAR_MXKV_BLOCK;
    const uint32_t data_bytes = (fmt == PULSAR_MXKV_FMT_FP4) ? (head_dim + 1u) / 2u : head_dim;
    const uint32_t rowbytes = data_bytes + nblk;
    uint8_t *outr = out + (uint64_t)row * rowbytes;
    uint8_t *scales = outr + data_bytes;
    const float max_repr = (fmt == PULSAR_MXKV_FMT_FP4) ? 6.0f : 448.0f;

    for (uint32_t b = 0; b < nblk; b++) {
        const uint32_t idx = b * PULSAR_MXKV_BLOCK + lane;
        const float v = (idx < head_dim) ? xr[idx] : 0.0f;
        float a = fabsf(v);
        for (uint32_t s = 16u; s > 0u; s >>= 1) a = fmaxf(a, __shfl_down_sync(0xffffffffu, a, s));
        a = __shfl_sync(0xffffffffu, a, 0);      /* block amax on every lane */
        /* Exact bucket (not the fast-math ceil-log2 path) so save/load of the
         * indexer/attn MX cache is value-idempotent — no boundary misround. */
        const uint8_t e8 = dsv4_e8m0_encode_scale_exact_dev(a, max_repr, 1.0e-20f);
        const float scale = dsv4_e8m0_decode_scale_dev(e8);
        if (lane == 0) scales[b] = e8;
        if (fmt == PULSAR_MXKV_FMT_FP4) {
            const uint8_t nib = dsv4_e2m1fn_encode_dev(v / scale);
            const uint32_t hi = __shfl_down_sync(0xffffffffu, (uint32_t)nib, 1);
            if ((lane & 1u) == 0u && idx < head_dim)
                outr[b * (PULSAR_MXKV_BLOCK / 2u) + (lane >> 1)] = (uint8_t)(nib | (hi << 4));
        } else {
            if (idx < head_dim)
                outr[b * PULSAR_MXKV_BLOCK + lane] = dsv4_e4m3fn_encode_dev(v / scale);
        }
    }
}

__global__ static void mxkv_dequant_kernel(const uint8_t *in, float *out,
                                           uint32_t n_tok, uint32_t head_dim, uint32_t fmt) {
    uint32_t row = blockIdx.x;
    if (row >= n_tok) return;
    const uint32_t nblk = head_dim / PULSAR_MXKV_BLOCK;
    const uint32_t data_bytes = (fmt == PULSAR_MXKV_FMT_FP4) ? (head_dim + 1u) / 2u : head_dim;
    const uint32_t rowbytes = data_bytes + nblk;
    const uint8_t *inr = in + (uint64_t)row * rowbytes;
    const uint8_t *scales = inr + data_bytes;
    float *outr = out + (uint64_t)row * head_dim;
    for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        const float scale = dsv4_e8m0_decode_scale_dev(scales[d / PULSAR_MXKV_BLOCK]);
        if (fmt == PULSAR_MXKV_FMT_FP4) {
            const uint8_t byte = inr[d >> 1];
            const uint8_t nib = (d & 1u) ? (uint8_t)(byte >> 4) : (uint8_t)(byte & 0xfu);
            outr[d] = dsv4_e2m1fn_decode_dev(nib, scale);
        } else {
            outr[d] = dsv4_e4m3fn_decode_dev(inr[d], scale);
        }
    }
}

int pulsar_gpu_mxkv_pack_tensor(const pulsar_gpu_tensor *x, pulsar_gpu_tensor *out,
                                        uint32_t fmt, uint32_t n_tok, uint32_t head_dim) {
    if (!x || !out || n_tok == 0 || (head_dim % PULSAR_MXKV_BLOCK) != 0 ||
        (fmt != PULSAR_MXKV_FMT_FP8 && fmt != PULSAR_MXKV_FMT_FP4) ||
        x->bytes < (uint64_t)n_tok * head_dim * sizeof(float) ||
        out->bytes < (uint64_t)n_tok * PULSAR_MXKV_ROWBYTES(fmt, head_dim)) return 0;
    mxkv_pack_kernel<<<n_tok, 32>>>((const float *)x->ptr, (uint8_t *)out->ptr, n_tok, head_dim, fmt);
    return cuda_ok(cudaGetLastError(), "mxkv_pack launch");
}

int pulsar_gpu_mxkv_dequant_tensor(const pulsar_gpu_tensor *in, pulsar_gpu_tensor *out,
                                           uint32_t fmt, uint32_t n_tok, uint32_t head_dim) {
    if (!in || !out || n_tok == 0 || (head_dim % PULSAR_MXKV_BLOCK) != 0 ||
        (fmt != PULSAR_MXKV_FMT_FP8 && fmt != PULSAR_MXKV_FMT_FP4) ||
        in->bytes < (uint64_t)n_tok * PULSAR_MXKV_ROWBYTES(fmt, head_dim) ||
        out->bytes < (uint64_t)n_tok * head_dim * sizeof(float)) return 0;
    mxkv_dequant_kernel<<<n_tok, 256>>>((const uint8_t *)in->ptr, (float *)out->ptr, n_tok, head_dim, fmt);
    return cuda_ok(cudaGetLastError(), "mxkv_dequant launch");
}

/* PULSAR_ATTN_PACK quantize+store: fp8-roundtrip the nope dims of n_rows f32 rows
 * of x IN PLACE (identical to pulsar_gpu_dsv4_fp8_kv_quantize_tensor) and store
 * the packed rows into `packed` at rows [out_row0, out_row0+n_rows). */
int pulsar_gpu_attn_pack_quantize_store_tensor(pulsar_gpu_tensor *x,
                                                       pulsar_gpu_tensor *packed,
                                                       uint32_t out_row0,
                                                       uint32_t n_rows,
                                                       uint32_t head_dim,
                                                       uint32_t n_rot) {
    if (!x || !packed || n_rows == 0 ||
        n_rot != PULSAR_ATTN_PACK_NROT || head_dim <= n_rot ||
        ((head_dim - n_rot) % PULSAR_FP8_KV_BLOCK) != 0 ||
        x->bytes < (uint64_t)n_rows * head_dim * sizeof(float) ||
        packed->bytes < ((uint64_t)out_row0 + n_rows) * PULSAR_ATTN_PACK_ROWBYTES(head_dim)) {
        return 0;
    }
    attn_pack_store_kernel<<<n_rows, 64>>>((float *)x->ptr, (uint8_t *)packed->ptr,
                                           out_row0, n_rows, head_dim, n_rot);
    return cuda_ok(cudaGetLastError(), "attn_pack_store launch");
}

/* PULSAR_ATTN_PACK dequant: the first n_rows packed rows -> f32 rows in `out`. */
int pulsar_gpu_attn_pack_dequant_tensor(const pulsar_gpu_tensor *in,
                                                pulsar_gpu_tensor *out,
                                                uint32_t n_rows,
                                                uint32_t head_dim,
                                                uint32_t n_rot) {
    if (!in || !out || n_rows == 0 ||
        n_rot != PULSAR_ATTN_PACK_NROT || head_dim <= n_rot ||
        ((head_dim - n_rot) % PULSAR_FP8_KV_BLOCK) != 0 ||
        in->bytes < (uint64_t)n_rows * PULSAR_ATTN_PACK_ROWBYTES(head_dim) ||
        out->bytes < (uint64_t)n_rows * head_dim * sizeof(float)) {
        return 0;
    }
    attn_pack_dequant_kernel<<<n_rows, 256>>>((const uint8_t *)in->ptr, (float *)out->ptr,
                                              n_rows, head_dim, n_rot);
    return cuda_ok(cudaGetLastError(), "attn_pack_dequant launch");
}

/* PULSAR_ATTN_PACK REPACK-ONLY entry (session load): pack n_rows f32 rows that
 * are ALREADY fp8-roundtripped (session files always contain such rows) into
 * `packed` at [out_row0, out_row0+n_rows), using the exact integer-math scale
 * bucket — value-idempotent, unlike the fast-math quantize path which can
 * misround the bucket at scale boundaries.  x is not modified. */
int pulsar_gpu_attn_pack_repack_tensor(const pulsar_gpu_tensor *x,
                                               pulsar_gpu_tensor *packed,
                                               uint32_t out_row0,
                                               uint32_t n_rows,
                                               uint32_t head_dim,
                                               uint32_t n_rot) {
    if (!x || !packed || n_rows == 0 ||
        n_rot != PULSAR_ATTN_PACK_NROT || head_dim <= n_rot ||
        ((head_dim - n_rot) % PULSAR_FP8_KV_BLOCK) != 0 ||
        x->bytes < (uint64_t)n_rows * head_dim * sizeof(float) ||
        packed->bytes < ((uint64_t)out_row0 + n_rows) * PULSAR_ATTN_PACK_ROWBYTES(head_dim)) {
        return 0;
    }
    attn_pack_repack_kernel<<<n_rows, 64>>>((const float *)x->ptr, (uint8_t *)packed->ptr,
                                            out_row0, n_rows, head_dim, n_rot);
    return cuda_ok(cudaGetLastError(), "attn_pack_repack launch");
}

/*
 * Gathered dequant: dequant n_sel rows selected by rows[i] from an MX KV cache
 * (row stride cap_rows) into a contiguous f32 [n_sel][head_dim] buffer.  This is
 * the attention gather primitive — it materializes the top-K compressed rows
 * (and, with an identity index, a raw window) that a query attends to, ready to
 * pack into a GEMM operand.  Optional out_stride lets V be written transposed
 * ([head_dim][n_sel]) by the caller assembling the PV operand.
 */
__global__ static void mxkv_gather_dequant_kernel(const uint8_t *cache, float *out,
                                                  const int32_t *rows, uint32_t n_sel,
                                                  uint32_t cap_rows,
                                                  uint32_t head_dim, uint32_t fmt,
                                                  uint32_t out_row_stride, uint32_t out_col_stride) {
    uint32_t i = blockIdx.x;
    if (i >= n_sel) return;
    int32_t r = rows[i];
    if (r < 0 || (uint32_t)r >= cap_rows) return;
    const uint32_t nblk = head_dim / PULSAR_MXKV_BLOCK;
    const uint32_t data_bytes = (fmt == PULSAR_MXKV_FMT_FP4) ? (head_dim + 1u) / 2u : head_dim;
    const uint32_t rowbytes = data_bytes + nblk;
    const uint8_t *inr = cache + (uint64_t)r * rowbytes;
    const uint8_t *scales = inr + data_bytes;
    for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        const float scale = dsv4_e8m0_decode_scale_dev(scales[d / PULSAR_MXKV_BLOCK]);
        float v;
        if (fmt == PULSAR_MXKV_FMT_FP4) {
            const uint8_t byte = inr[d >> 1];
            const uint8_t nib = (d & 1u) ? (uint8_t)(byte >> 4) : (uint8_t)(byte & 0xfu);
            v = dsv4_e2m1fn_decode_dev(nib, scale);
        } else {
            v = dsv4_e4m3fn_decode_dev(inr[d], scale);
        }
        out[(uint64_t)i * out_row_stride + (uint64_t)d * out_col_stride] = v;
    }
}

/* out is [n_sel][head_dim] contiguous when transpose==0, or [head_dim][n_sel]
 * (column i strided) when transpose!=0 — the latter builds a PV V^T operand. */
int pulsar_gpu_mxkv_gather_dequant_tensor(const pulsar_gpu_tensor *cache, pulsar_gpu_tensor *out,
                                                  const pulsar_gpu_tensor *rows, uint32_t n_sel,
                                                  uint32_t cap_rows, uint32_t head_dim, uint32_t fmt,
                                                  uint32_t transpose) {
    if (!cache || !out || !rows || n_sel == 0 || (head_dim % PULSAR_MXKV_BLOCK) != 0 ||
        (fmt != PULSAR_MXKV_FMT_FP8 && fmt != PULSAR_MXKV_FMT_FP4) ||
        cache->bytes < (uint64_t)cap_rows * PULSAR_MXKV_ROWBYTES(fmt, head_dim) ||
        rows->bytes < (uint64_t)n_sel * sizeof(int32_t) ||
        out->bytes < (uint64_t)n_sel * head_dim * sizeof(float)) return 0;
    const uint32_t out_row_stride = transpose ? 1u : head_dim;      /* stride between rows i */
    const uint32_t out_col_stride = transpose ? n_sel : 1u;         /* stride between dims d */
    mxkv_gather_dequant_kernel<<<n_sel, 256>>>((const uint8_t *)cache->ptr, (float *)out->ptr,
                                               (const int32_t *)rows->ptr, n_sel, cap_rows,
                                               head_dim, fmt,
                                               out_row_stride, out_col_stride);
    return cuda_ok(cudaGetLastError(), "mxkv_gather_dequant launch");
}


int pulsar_gpu_dsv4_indexer_qat_tensor(pulsar_gpu_tensor *x, uint32_t n_rows, uint32_t head_dim) {
    if (!x || n_rows == 0 || head_dim != 128u ||
        x->bytes < (uint64_t)n_rows * head_dim * sizeof(float)) {
        return 0;
    }
    indexer_hadamard_fp4_kernel<<<n_rows, 128>>>((float *)x->ptr, n_rows, head_dim);
    return cuda_ok(cudaGetLastError(), "indexer_hadamard_fp4 launch");
}

/* QAT + pack: roundtrip n_rows f32 rows of x in place (identical to
 * pulsar_gpu_dsv4_indexer_qat_tensor) and store the MXKV FP4 packed rows into
 * `packed` at rows [out_row0, out_row0 + n_rows). */
int pulsar_gpu_dsv4_indexer_qat_pack_tensor(pulsar_gpu_tensor *x,
                                                    pulsar_gpu_tensor *packed,
                                                    uint32_t out_row0,
                                                    uint32_t n_rows,
                                                    uint32_t head_dim) {
    const uint64_t rowbytes = PULSAR_MXKV_FP4_ROWBYTES(128u);
    if (!x || !packed || n_rows == 0 || head_dim != 128u ||
        x->bytes < (uint64_t)n_rows * head_dim * sizeof(float) ||
        packed->bytes < ((uint64_t)out_row0 + n_rows) * rowbytes) {
        return 0;
    }
    indexer_hadamard_fp4_pack_kernel<<<n_rows, 128>>>(
            (float *)x->ptr,
            (uint8_t *)packed->ptr + (uint64_t)out_row0 * rowbytes,
            n_rows, head_dim);
    return cuda_ok(cudaGetLastError(), "indexer_hadamard_fp4_pack launch");
}


int pulsar_gpu_rope_tail_tensor(pulsar_gpu_tensor *x, uint32_t n_tok, uint32_t n_head, uint32_t head_dim, uint32_t n_rot, uint32_t pos0, uint32_t n_ctx_orig, bool inverse, float freq_base, float freq_scale, float ext_factor, float attn_factor, float beta_fast, float beta_slow, const pulsar_gpu_tensor *positions) {
    if (!x || n_rot > head_dim || (n_rot & 1) || x->bytes < (uint64_t)n_tok * n_head * head_dim * sizeof(float)) return 0;
    if (positions && positions->bytes < (uint64_t)n_tok * sizeof(int32_t)) return 0;
    uint32_t pairs = n_tok * n_head * (n_rot / 2);
    rope_tail_kernel<<<(pairs + 255) / 256, 256>>>((float *)x->ptr, n_tok, n_head, head_dim, n_rot, pos0, 1, n_ctx_orig, inverse ? 1 : 0, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow, positions ? (const int32_t *)positions->ptr : NULL);
    return cuda_ok(cudaGetLastError(), "rope_tail launch");
}


int pulsar_gpu_store_raw_kv_tensor(pulsar_gpu_tensor *raw_cache, const pulsar_gpu_tensor *kv, uint32_t raw_cap, uint32_t row, uint32_t head_dim, uint32_t raw_f16);


int pulsar_gpu_kv_fp8_store_raw_tensor(
        pulsar_gpu_tensor *kv,
        pulsar_gpu_tensor *raw_cache,
        uint32_t          raw_cap,
        uint32_t          raw_row,
        uint32_t          head_dim,
        uint32_t          n_rot,
        uint32_t          raw_f16) {
    if (!kv || !raw_cache || raw_cap == 0 || head_dim == 0 || n_rot > head_dim ||
        raw_cache->bytes < (uint64_t)raw_cap * head_dim * (raw_f16 ? sizeof(__half) : sizeof(float)) ||
        kv->bytes < (uint64_t)head_dim * sizeof(float)) return 0;
    fp8_kv_quantize_store_kernel<<<1, 64>>>((float *)kv->ptr, (float *)raw_cache->ptr,
                                            raw_cap, raw_row, head_dim, n_rot, (int)raw_f16);
    return cuda_ok(cudaGetLastError(), "fp8_kv_quantize_store launch");
}


int pulsar_gpu_store_raw_kv_tensor(pulsar_gpu_tensor *raw_cache, const pulsar_gpu_tensor *kv, uint32_t raw_cap, uint32_t row, uint32_t head_dim, uint32_t raw_f16) {
    if (!raw_cache || !kv || raw_cap == 0 ||
        raw_cache->bytes < (uint64_t)raw_cap * head_dim * (raw_f16 ? sizeof(__half) : sizeof(float)) ||
        kv->bytes < (uint64_t)head_dim * sizeof(float)) return 0;
    store_raw_kv_batch_kernel<<<(head_dim + 255) / 256, 256>>>((float *)raw_cache->ptr, (const float *)kv->ptr, raw_cap, row, 1, head_dim, (int)raw_f16,
                                                               NULL, NULL, 1u);
    return cuda_ok(cudaGetLastError(), "store_raw_kv launch");
}


int pulsar_gpu_store_raw_kv_batch_tensor(pulsar_gpu_tensor *raw_cache, const pulsar_gpu_tensor *kv, uint32_t raw_cap, uint32_t pos0, uint32_t n_tokens, uint32_t head_dim, uint32_t raw_f16,
                                                 const pulsar_gpu_tensor *positions, const pulsar_gpu_tensor *seq_id, uint32_t n_banks) {
    /* Descriptor (banked) mode: both arrays or neither; the raw cache operand
     * is the whole bank pool (byte bound scales by n_banks) and the uint32
     * row ABI (seq*raw_cap + slot) must not overflow.  pos0 is ignored when
     * positions != NULL.  Fail-loud, like the banked attention launchers. */
    const int descr = positions != NULL || seq_id != NULL;
    if (descr &&
        (!positions || !seq_id || n_banks == 0 ||
         positions->bytes < (uint64_t)n_tokens * sizeof(int32_t) ||
         seq_id->bytes < (uint64_t)n_tokens * sizeof(int32_t) ||
         (uint64_t)n_banks * raw_cap > 4294967296ull)) {
        fprintf(stderr,
                "pulsar: banked raw store rejected: bad descriptor args "
                "(n_tokens=%u n_banks=%u raw_cap=%u)\n",
                n_tokens, n_banks, raw_cap);
        return 0;
    }
    const uint64_t kv_banks = descr ? n_banks : 1u;
    if (!raw_cache || !kv || raw_cap == 0 ||
        raw_cache->bytes < kv_banks * raw_cap * head_dim * (raw_f16 ? sizeof(__half) : sizeof(float)) ||
        kv->bytes < (uint64_t)n_tokens * head_dim * sizeof(float)) return 0;
    uint64_t n = (uint64_t)n_tokens * head_dim;
    store_raw_kv_batch_kernel<<<(n + 255) / 256, 256>>>((float *)raw_cache->ptr, (const float *)kv->ptr, raw_cap, pos0, n_tokens, head_dim, (int)raw_f16,
                                                        descr ? (const int32_t *)positions->ptr : NULL,
                                                        descr ? (const int32_t *)seq_id->ptr : NULL,
                                                        descr ? n_banks : 1u);
    return cuda_ok(cudaGetLastError(), "store_raw_kv_batch launch");
}


int pulsar_gpu_compressor_store_batch_tensor(
        const pulsar_gpu_tensor *kv,
        const pulsar_gpu_tensor *sc,
        pulsar_gpu_tensor       *state_kv,
        pulsar_gpu_tensor       *state_score,
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
    const char *ape = cuda_model_range_ptr(model_map, ape_offset, ape_bytes, "compressor_ape");
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



int pulsar_gpu_compressor_update_tensor(
        const pulsar_gpu_tensor *kv_cur,
        const pulsar_gpu_tensor *sc_cur,
        pulsar_gpu_tensor       *state_kv,
        pulsar_gpu_tensor       *state_score,
        pulsar_gpu_tensor       *comp_cache,
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
        float                   rms_eps) {
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
    if (!pulsar_gpu_compressor_store_batch_tensor(kv_cur, sc_cur, state_kv, state_score,
                                                 model_map, model_size, ape_offset, ape_type,
                                                 head_dim, ratio, pos, 1)) {
        return 0;
    }
    if (!emit) return 1;
    pulsar_gpu_tensor *comp_row_view = pulsar_gpu_tensor_view(
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
    if (ok) ok = pulsar_gpu_rms_norm_weight_rows_tensor(comp_row_view, comp_row_view,
                                                       model_map, model_size, norm_offset,
                                                       head_dim, 1, rms_eps);
    if (ok) ok = pulsar_gpu_rope_tail_tensor(comp_row_view, 1, 1, head_dim, n_rot,
                                            pos + 1u - ratio, n_ctx_orig, false,
                                            freq_base, freq_scale, ext_factor, attn_factor,
                                            beta_fast, beta_slow, NULL);
    pulsar_gpu_tensor_free(comp_row_view);
    if (ok && ratio == 4u) {
        uint64_t half = 4ull * width;
        compressor_shift_ratio4_kernel<<<(half + 255) / 256, 256>>>(
                (float *)state_kv->ptr, (float *)state_score->ptr, width);
        ok = cuda_ok(cudaGetLastError(), "compressor ratio4 shift launch");
    }
    return ok;
}


int pulsar_gpu_compressor_prefill_tensor(
        pulsar_gpu_tensor       *comp_cache,
        pulsar_gpu_tensor       *state_kv,
        pulsar_gpu_tensor       *state_score,
        const pulsar_gpu_tensor *kv,
        const pulsar_gpu_tensor *sc,
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
    const char *ape = cuda_model_range_ptr(model_map, ape_offset, ape_bytes, "compressor_ape");
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
        if (!pulsar_gpu_rms_norm_weight_rows_tensor(comp_cache, comp_cache,
                                                   model_map, model_size, norm_offset,
                                                   head_dim, n_comp, rms_eps)) return 0;
        if (n_rot != 0) {
            const uint32_t pairs = n_comp * (n_rot / 2u);
            rope_tail_kernel<<<(pairs + 255) / 256, 256>>>(
                    (float *)comp_cache->ptr, n_comp, 1, head_dim, n_rot,
                    pos0, ratio, n_ctx_orig, 0, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow, NULL);
            if (!cuda_ok(cudaGetLastError(), "compressor prefill rope launch")) return 0;
        }
        if (quantize_fp8 && !pulsar_gpu_dsv4_fp8_kv_quantize_tensor(comp_cache, n_comp, head_dim, n_rot)) return 0;
    }
    return 1;
}


int pulsar_gpu_compressor_prefill_ratio4_replay_tensor(
        pulsar_gpu_tensor       *comp_cache,
        pulsar_gpu_tensor       *state_kv,
        pulsar_gpu_tensor       *state_score,
        const pulsar_gpu_tensor *kv,
        const pulsar_gpu_tensor *sc,
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
    const char *ape = cuda_model_range_ptr(model_map, ape_offset, ape_bytes, "compressor_ape");
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
    if (!pulsar_gpu_rms_norm_weight_rows_tensor(comp_cache, comp_cache,
                                               model_map, model_size, norm_offset,
                                               head_dim, n_comp, rms_eps)) return 0;
    if (n_rot != 0) {
        const uint32_t pairs = n_comp * (n_rot / 2u);
        rope_tail_kernel<<<(pairs + 255) / 256, 256>>>(
                (float *)comp_cache->ptr, n_comp, 1, head_dim, n_rot,
                pos0, ratio, n_ctx_orig, 0, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow, NULL);
        if (!cuda_ok(cudaGetLastError(), "compressor replay rope launch")) return 0;
    }
    if (quantize_fp8 && !pulsar_gpu_dsv4_fp8_kv_quantize_tensor(comp_cache, n_comp, head_dim, n_rot)) return 0;

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


int pulsar_gpu_compressor_prefill_state_ratio4_tensor(
        pulsar_gpu_tensor       *state_kv,
        pulsar_gpu_tensor       *state_score,
        const pulsar_gpu_tensor *kv_tail,
        const pulsar_gpu_tensor *sc_tail,
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
    const char *ape = cuda_model_range_ptr(model_map, ape_offset, ape_bytes, "compressor_ape");
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

