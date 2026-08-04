// SPDX-License-Identifier: MIT
// proto_kv_reencode_idem.cu - packed-KV restore re-encode idempotence probe.
//
// v0.2.4 ships REFUSED packed disk-KV restores: the restore path reads the
// F32 committed rows from the file and regenerates the fp8/fp4 mirrors by
// re-encoding, and the kv_crossmode gate observed the resulting packed-server
// continuations DRIFTING (fp8-only and fp4-only independently) despite the
// pow2-scale idempotence argument saying re-encode of committed values is
// value-exact.  This proto isolates the kernel math from the pipeline:
//   fp8 : quantize(x) -> C1,codes1,scales1 ; dequant==C1 ;
//         quantize(C1) -> C2,codes2,scales2 ; C2 ?= C1 (bitwise) ; dequant==C1
//   fp4 : hadamard_qat(x) -> C1 ; fp4_dequant==C1 ;
//         noHad_encode(C1) -> C2 ; C2 ?= C1 ; dequant==C1
// plus a third-iteration fixed-point check and a fast-math boundary scan of
// scale = exp2f(ceilf(log2f(amax/448 or /6))) at exact power-of-two inputs
// (the one corner where fast-math approximation could flip ceilf and shift
// the whole block one exponent).
// Code/scale mismatches with VALUE identity are tallied separately: a block
// whose committed max sits exactly on a half-boundary legitimately re-picks
// scale/2 with all codes shifted one exponent, values bit-identical.
//
// The first battery runs verbatim copies of the v0.2.4 (PRE-fix) ds4_cuda.cu
// bodies and is EXPECTED to fail -- kept as the regression repro.  The _fx
// battery is the fix as shipped in v0.3 (dsv4_pow2_ceil_scale + signbit) and
// is the gate (exit status).  Build MUST mirror the engine's fast-math flags:
//   /usr/local/cuda/bin/nvcc -O3 -std=c++17 --use_fast_math -arch=sm_121 \
//       cuda/mmq/test/proto_kv_reencode_idem.cu -o proto_kv_reencode_idem
// argv: [n_rows_fp8=131072] [n_rows_fp4=131072]

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>

#define CHECK(call) do { cudaError_t e_=(call); if(e_!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s @%d: %s\n",#call,__LINE__,cudaGetErrorString(e_)); exit(1);} } while(0)

// ---------------------------------------------------------------------------
// Verbatim from ds4_cuda.cu (v0.2.4 tag content).
// ---------------------------------------------------------------------------

#define DS4_OPP_C_FP8_NOPE_DEV       448u    /* 512 - 64 */
#define DS4_OPP_C_FP8_BLOCKS_DEV     7u      /* 448 / 64  */
#define DS4_OPP_C_FP8_ROW_BYTES_DEV  704u    /* 448 codes + 64 FP32 tail */

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

__device__ static unsigned char dsv4_e4m3fn_encode_dev(float x) {
    const unsigned int sign_bit = (x < 0.0f) ? 0x80u : 0x00u;
    const float ax = fminf(fabsf(x), 448.0f);
    int lo = 0, hi = 126;
    while (lo < hi) {
        const int mid = (lo + hi + 1) >> 1;
        if (dsv4_e4m3fn_value_dev(mid) <= ax) lo = mid;
        else hi = mid - 1;
    }
    int best = lo;
    if (best < 126) {
        const float bd = fabsf(ax - dsv4_e4m3fn_value_dev(best));
        const float nd = fabsf(ax - dsv4_e4m3fn_value_dev(best + 1));
        if (nd < bd || (nd == bd && (((best + 1) & 1) == 0) && ((best & 1) != 0))) best++;
    }
    return (unsigned char)(sign_bit | (unsigned int)best);
}

__device__ float dsv4_e4m3fn_decode_table[128];

static int fp8_decode_table_init(void) {
    float table[128];
    for (int i = 0; i < 128; i++) {
        const int exp  = (i >> 3) & 15;
        const int mant = i & 7;
        table[i] = (exp == 0)
            ? (float)mant * 0.001953125f
            : (1.0f + (float)mant * 0.125f) * ldexpf(1.0f, exp - 7);
    }
    return cudaMemcpyToSymbol(dsv4_e4m3fn_decode_table, table,
                              sizeof(table), 0, cudaMemcpyHostToDevice) == cudaSuccess;
}

__device__ static float fp8_kv_read(
        const unsigned char * __restrict__ codes_base,
        const float         * __restrict__ scale_base,
        uint32_t                          c,
        uint32_t                          d) {
    const uint64_t row_off = (uint64_t)c * DS4_OPP_C_FP8_ROW_BYTES_DEV;
    if (d < DS4_OPP_C_FP8_NOPE_DEV) {
        const unsigned char code = codes_base[row_off + d];
        const float scale = scale_base[(uint64_t)c * DS4_OPP_C_FP8_BLOCKS_DEV + (d >> 6)];
        const float mag   = dsv4_e4m3fn_decode_table[code & 0x7fu];
        return ((code & 0x80u) ? -mag : mag) * scale;
    }
    const float *tail = (const float *)(codes_base + row_off + DS4_OPP_C_FP8_NOPE_DEV);
    return tail[d - DS4_OPP_C_FP8_NOPE_DEV];
}

__global__ static void fp8_kv_dequant_rows_kernel(
        float * __restrict__ dst,
        const unsigned char * __restrict__ codes_base,
        const float         * __restrict__ scale_base,
        uint32_t n_rows,
        uint32_t head_dim) {
    const uint32_t r = blockIdx.x;
    if (r >= n_rows) return;
    float *out = dst + (uint64_t)r * head_dim;
    for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        out[d] = fp8_kv_read(codes_base, scale_base, r, d);
    }
}

__global__ static void fp8_kv_quantize_kernel(
        float *x, uint32_t n_tok, uint32_t head_dim, uint32_t n_rot,
        unsigned char * __restrict__ codes_base,
        float * __restrict__ scale_base) {
    uint32_t row = blockIdx.x;
    uint32_t tid = threadIdx.x;
    uint32_t n_nope = head_dim - n_rot;
    const uint64_t row_stride = (uint64_t)n_nope + (uint64_t)n_rot * sizeof(float);
    const uint64_t scale_stride = (uint64_t)(n_nope / 64u);
    float *xr = x + (uint64_t)row * head_dim;
    unsigned char *codes_row = codes_base
        ? codes_base + (uint64_t)row * row_stride
        : (unsigned char *)0;
    float *scale_row = scale_base
        ? scale_base + (uint64_t)row * scale_stride
        : (float *)0;
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
            float clamp = fminf(448.0f, fmaxf(-448.0f, v / scale));
            xr[off + tid] = dsv4_e4m3fn_dequant_dev(clamp) * scale;
            if (codes_row) codes_row[off + tid] = dsv4_e4m3fn_encode_dev(clamp);
        }
        if (scale_row && tid == 0) scale_row[off / 64u] = scale;
        __syncthreads();
    }
    if (codes_row && tid < n_rot) {
        float *tail = (float *)(codes_row + (uint64_t)n_nope);
        tail[tid] = xr[(uint64_t)n_nope + tid];
    }
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

__device__ static int dsv4_e2m1fn_level_dev(float ax) {
    ax = fminf(ax, 6.0f);
    int best = 0;
    float best_diff = fabsf(ax - dsv4_e2m1fn_value_dev(0));
    for (int i = 1; i < 8; i++) {
        float diff = fabsf(ax - dsv4_e2m1fn_value_dev(i));
        if (diff < best_diff || (diff == best_diff && ((i & 1) == 0) && ((best & 1) != 0))) {
            best = i;
            best_diff = diff;
        }
    }
    return best;
}

__device__ static float indexer_fp4_read(
        const unsigned char * __restrict__ codes_base,
        const float         * __restrict__ scale_base,
        uint32_t row, uint32_t d) {
    const unsigned char byte = codes_base[(uint64_t)row * 64u + (d >> 1u)];
    const uint32_t nib = ((uint32_t)byte >> (4u * (d & 1u))) & 0xFu;
    const float sign = (nib & 8u) ? -1.0f : 1.0f;
    const float scale = scale_base[(uint64_t)row * 4u + (d >> 5u)];
    return sign * dsv4_e2m1fn_value_dev((int)(nib & 7u)) * scale;
}

__global__ static void indexer_fp4_dequant_rows_kernel(
        float * __restrict__ dst,
        const unsigned char * __restrict__ codes_base,
        const float         * __restrict__ scale_base,
        uint32_t n_rows, uint32_t head_dim) {
    const uint32_t r = blockIdx.x;
    if (r >= n_rows) return;
    float *out = dst + (uint64_t)r * head_dim;
    for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        out[d] = indexer_fp4_read(codes_base, scale_base, r, d);
    }
}

__global__ static void indexer_fp4_encode_rows_kernel(
        float * __restrict__ x, uint32_t n_rows, uint32_t head_dim,
        unsigned char * __restrict__ codes_base, float * __restrict__ scale_base) {
    uint32_t row = blockIdx.x;
    uint32_t tid = threadIdx.x;
    if (row >= n_rows || head_dim != 128u || tid >= 128u) return;
    __shared__ float absbuf[128];
    float *xr = x + (uint64_t)row * head_dim;
    float v = xr[tid];                       /* committed value -- NO Hadamard */
    uint32_t fp4_block = tid >> 5u;
    uint32_t lane = tid & 31u;
    uint32_t block_base = fp4_block * 32u;
    absbuf[tid] = fabsf(v);
    __syncthreads();
    for (uint32_t stride = 16u; stride > 0u; stride >>= 1u) {
        if (lane < stride)
            absbuf[block_base + lane] = fmaxf(absbuf[block_base + lane],
                                              absbuf[block_base + lane + stride]);
        __syncthreads();
    }
    float amax = fmaxf(absbuf[block_base], 7.052966104933725e-38f);
    float scale = exp2f(ceilf(log2f(amax / 6.0f)));
    float clamp = fminf(6.0f, fmaxf(-6.0f, v / scale));
    int level = dsv4_e2m1fn_level_dev(fabsf(clamp));
    xr[tid] = ((clamp < 0.0f ? -1.0f : 1.0f) * dsv4_e2m1fn_value_dev(level)) * scale;
    uint32_t nib = (clamp < 0.0f ? 8u : 0u) | (uint32_t)level;
    uint32_t hi  = __shfl_down_sync(0xffffffffu, nib, 1u);
    if ((tid & 1u) == 0u)  codes_base[(uint64_t)row * 64u + (tid >> 1u)] = (unsigned char)(nib | (hi << 4u));
    if ((tid & 31u) == 0u) scale_base[(uint64_t)row * 4u + (tid >> 5u)] = scale;
}

__global__ static void indexer_hadamard_fp4_kernel(float *x, uint32_t n_rows, uint32_t head_dim,
        unsigned char * __restrict__ codes_base,
        float         * __restrict__ scale_base) {
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
    float clamp = fminf(6.0f, fmaxf(-6.0f, v / scale));
    int level = dsv4_e2m1fn_level_dev(fabsf(clamp));
    xr[tid] = ((clamp < 0.0f ? -1.0f : 1.0f) * dsv4_e2m1fn_value_dev(level)) * scale;
    if (codes_base) {
        uint32_t nib = (clamp < 0.0f ? 8u : 0u) | (uint32_t)level;
        uint32_t hi  = __shfl_down_sync(0xffffffffu, nib, 1u);
        if ((tid & 1u) == 0u)  codes_base[(uint64_t)row * 64u + (tid >> 1u)] = (unsigned char)(nib | (hi << 4u));
        if ((tid & 31u) == 0u) scale_base[(uint64_t)row * 4u + (tid >> 5u)] = scale;
    }
}

// ---------------------------------------------------------------------------
// FIXED variants (candidate engine change): exact pow2-ceil scale via frexpf
// (bit ops, immune to fast-math lg2.approx one-sided error at 2^-n) and
// signbit-based sign handling (preserves -0.0 across the round trip).
// ---------------------------------------------------------------------------

__device__ static float dsv4_pow2_ceil_scale(float r) {
    int e; float m = frexpf(r, &e);          /* r = m * 2^e, m in [0.5, 1) */
    return ldexpf(1.0f, m == 0.5f ? e - 1 : e);
}

__device__ static float dsv4_e4m3fn_dequant_fx(float x) {
    float sign = signbit(x) ? -1.0f : 1.0f;
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

__device__ static unsigned char dsv4_e4m3fn_encode_fx(float x) {
    const unsigned int sign_bit = signbit(x) ? 0x80u : 0x00u;
    const float ax = fminf(fabsf(x), 448.0f);
    int lo = 0, hi = 126;
    while (lo < hi) {
        const int mid = (lo + hi + 1) >> 1;
        if (dsv4_e4m3fn_value_dev(mid) <= ax) lo = mid;
        else hi = mid - 1;
    }
    int best = lo;
    if (best < 126) {
        const float bd = fabsf(ax - dsv4_e4m3fn_value_dev(best));
        const float nd = fabsf(ax - dsv4_e4m3fn_value_dev(best + 1));
        if (nd < bd || (nd == bd && (((best + 1) & 1) == 0) && ((best & 1) != 0))) best++;
    }
    return (unsigned char)(sign_bit | (unsigned int)best);
}

__global__ static void fp8_kv_quantize_kernel_fx(
        float *x, uint32_t n_tok, uint32_t head_dim, uint32_t n_rot,
        unsigned char * __restrict__ codes_base,
        float * __restrict__ scale_base) {
    uint32_t row = blockIdx.x;
    uint32_t tid = threadIdx.x;
    uint32_t n_nope = head_dim - n_rot;
    const uint64_t row_stride = (uint64_t)n_nope + (uint64_t)n_rot * sizeof(float);
    const uint64_t scale_stride = (uint64_t)(n_nope / 64u);
    float *xr = x + (uint64_t)row * head_dim;
    unsigned char *codes_row = codes_base
        ? codes_base + (uint64_t)row * row_stride
        : (unsigned char *)0;
    float *scale_row = scale_base
        ? scale_base + (uint64_t)row * scale_stride
        : (float *)0;
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
        float scale = dsv4_pow2_ceil_scale(fmaxf(scratch[0], 1.0e-4f) / 448.0f);
        if (off + tid < n_nope) {
            float clamp = fminf(448.0f, fmaxf(-448.0f, v / scale));
            xr[off + tid] = dsv4_e4m3fn_dequant_fx(clamp) * scale;
            if (codes_row) codes_row[off + tid] = dsv4_e4m3fn_encode_fx(clamp);
        }
        if (scale_row && tid == 0) scale_row[off / 64u] = scale;
        __syncthreads();
    }
    if (codes_row && tid < n_rot) {
        float *tail = (float *)(codes_row + (uint64_t)n_nope);
        tail[tid] = xr[(uint64_t)n_nope + tid];
    }
}

__global__ static void indexer_fp4_encode_rows_kernel_fx(
        float * __restrict__ x, uint32_t n_rows, uint32_t head_dim,
        unsigned char * __restrict__ codes_base, float * __restrict__ scale_base) {
    uint32_t row = blockIdx.x;
    uint32_t tid = threadIdx.x;
    if (row >= n_rows || head_dim != 128u || tid >= 128u) return;
    __shared__ float absbuf[128];
    float *xr = x + (uint64_t)row * head_dim;
    float v = xr[tid];
    uint32_t fp4_block = tid >> 5u;
    uint32_t lane = tid & 31u;
    uint32_t block_base = fp4_block * 32u;
    absbuf[tid] = fabsf(v);
    __syncthreads();
    for (uint32_t stride = 16u; stride > 0u; stride >>= 1u) {
        if (lane < stride)
            absbuf[block_base + lane] = fmaxf(absbuf[block_base + lane],
                                              absbuf[block_base + lane + stride]);
        __syncthreads();
    }
    float amax = fmaxf(absbuf[block_base], 7.052966104933725e-38f);
    float scale = dsv4_pow2_ceil_scale(amax / 6.0f);
    float clamp = fminf(6.0f, fmaxf(-6.0f, v / scale));
    int level = dsv4_e2m1fn_level_dev(fabsf(clamp));
    xr[tid] = ((signbit(clamp) ? -1.0f : 1.0f) * dsv4_e2m1fn_value_dev(level)) * scale;
    uint32_t nib = (signbit(clamp) ? 8u : 0u) | (uint32_t)level;
    uint32_t hi  = __shfl_down_sync(0xffffffffu, nib, 1u);
    if ((tid & 1u) == 0u)  codes_base[(uint64_t)row * 64u + (tid >> 1u)] = (unsigned char)(nib | (hi << 4u));
    if ((tid & 31u) == 0u) scale_base[(uint64_t)row * 4u + (tid >> 5u)] = scale;
}

__global__ static void indexer_hadamard_fp4_kernel_fx(float *x, uint32_t n_rows, uint32_t head_dim,
        unsigned char * __restrict__ codes_base,
        float         * __restrict__ scale_base) {
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
    float scale = dsv4_pow2_ceil_scale(amax / 6.0f);
    float clamp = fminf(6.0f, fmaxf(-6.0f, v / scale));
    int level = dsv4_e2m1fn_level_dev(fabsf(clamp));
    xr[tid] = ((signbit(clamp) ? -1.0f : 1.0f) * dsv4_e2m1fn_value_dev(level)) * scale;
    if (codes_base) {
        uint32_t nib = (signbit(clamp) ? 8u : 0u) | (uint32_t)level;
        uint32_t hi  = __shfl_down_sync(0xffffffffu, nib, 1u);
        if ((tid & 1u) == 0u)  codes_base[(uint64_t)row * 64u + (tid >> 1u)] = (unsigned char)(nib | (hi << 4u));
        if ((tid & 31u) == 0u) scale_base[(uint64_t)row * 4u + (tid >> 5u)] = scale;
    }
}

// ---------------------------------------------------------------------------
// Proto scaffolding (not engine code).
// ---------------------------------------------------------------------------

// Deterministic per-element fill: xorshift hash of (row,lane) -> [-1,1),
// scaled per row-band regime.  Bands (of 8) for the fp8 input:
//   0 uniform | 1 x1e-5 | 2 x1e4 | 3 pow2 magnitude ladder 2^((row%40)-20)
//   4 block amax forced EXACTLY 448*2^k   (k=(row%40)-20; lane0 of block)
//   5 block amax forced EXACTLY 224*2^k   (post-encode halving boundary)
//   6 block amax 433*2^k (e4m3 rounds it UP to 448*2^k -> boundary amax')
//   7 tiny/zero: x1e-6, every 3rd row all-zero (sub-1e-4-floor blocks)
// Exact 2^k as a float, immune to any exp2f approximation question.
__device__ static float exact_pow2(int k) {
    return __uint_as_float((uint32_t)(127 + k) << 23);
}

__device__ static float hash_unit(uint32_t row, uint32_t lane, uint32_t salt) {
    uint32_t h = row * 2654435761u ^ (lane + 1u) * 2246822519u ^ salt * 3266489917u;
    h ^= h >> 15; h *= 2246822519u; h ^= h >> 13; h *= 3266489917u; h ^= h >> 16;
    return ((float)(h & 0xFFFFFFu) / 8388608.0f) - 1.0f;   // [-1,1)
}

__global__ static void fill_fp8_input_kernel(float *x, uint32_t n_rows, uint32_t salt) {
    uint32_t row = blockIdx.x;
    uint32_t tid = threadIdx.x;          // 512 threads
    if (row >= n_rows || tid >= 512u) return;
    uint32_t band = row / (n_rows / 8u + 1u);
    float u = hash_unit(row, tid, salt);
    float v = u;
    int k = (int)(row % 40u) - 20;
    switch (band) {
    case 0: break;
    case 1: v = u * 1e-5f; break;
    case 2: v = u * 1e4f; break;
    case 3: v = u * exact_pow2(k); break;
    case 4: v = (tid < 448u && (tid & 63u) == 0u) ? 448.0f * exact_pow2(k)
                                                  : u * 100.0f * exact_pow2(k); break;
    case 5: v = (tid < 448u && (tid & 63u) == 0u) ? 224.0f * exact_pow2(k)
                                                  : u * 100.0f * exact_pow2(k); break;
    case 6: v = (tid < 448u && (tid & 63u) == 0u) ? 433.0f * exact_pow2(k)
                                                  : u * 100.0f * exact_pow2(k); break;
    default:
        v = (row % 3u == 0u) ? 0.0f : u * 1e-6f;
        break;
    }
    x[(uint64_t)row * 512u + tid] = v;
}

__global__ static void fill_fp4_input_kernel(float *x, uint32_t n_rows, uint32_t salt) {
    uint32_t row = blockIdx.x;
    uint32_t tid = threadIdx.x;          // 128 threads
    if (row >= n_rows || tid >= 128u) return;
    uint32_t band = row / (n_rows / 6u + 1u);
    float u = hash_unit(row, tid, salt ^ 0x9e3779b9u);
    float v = u;
    int k = (int)(row % 60u) - 30;
    switch (band) {
    case 0: break;
    case 1: v = u * 1e-6f; break;
    case 2: v = u * 1e5f; break;
    case 3: v = u * exact_pow2(k); break;
    case 4: v = (u > 0.0f ? 1.0f : -1.0f) * exact_pow2(k); break;  // pure pow2 lanes
    default:
        v = (row % 3u == 0u) ? 0.0f : u * 1e-30f;
        break;
    }
    x[(uint64_t)row * 128u + tid] = v;
}

struct mismatch_rec { uint32_t idx_hi, idx_lo, a_bits, b_bits; };
#define MAX_RECS 8

__global__ static void compare_f32_kernel(const float *a, const float *b, uint64_t n,
                                          unsigned long long *count,
                                          mismatch_rec *recs, int *nrec) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint32_t ab = __float_as_uint(a[i]);
    uint32_t bb = __float_as_uint(b[i]);
    if (ab != bb) {
        atomicAdd(count, 1ull);
        int slot = atomicAdd(nrec, 1);
        if (slot < MAX_RECS) {
            recs[slot].idx_hi = (uint32_t)(i >> 32);
            recs[slot].idx_lo = (uint32_t)(i & 0xFFFFFFFFu);
            recs[slot].a_bits = ab;
            recs[slot].b_bits = bb;
        }
    }
}

__global__ static void compare_u8_kernel(const unsigned char *a, const unsigned char *b,
                                         uint64_t n, unsigned long long *count) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (a[i] != b[i]) atomicAdd(count, 1ull);
}

// Fast-math boundary scan: the exact-pow2 inputs the re-encode presents when
// a block's committed max lands on 448*2^k (fp8) or {6,3,1.5}*2^k (fp4).
// out[j] = scale computed by the shipped formula; host prints deviations.
__global__ static void boundary_scan_kernel(float *out) {
    int idx = threadIdx.x;               // 61 k values: -30..30
    if (idx >= 61) return;
    int k = idx - 30;
    float p2k = exact_pow2(k);
    // fp8 formula at amax = 448*2^k, 224*2^k, and one-ulp neighbours of 448*2^k
    float a448  = 448.0f * p2k;
    float a224  = 224.0f * p2k;
    float a448u = __uint_as_float(__float_as_uint(a448) + 1u);
    float a448d = __uint_as_float(__float_as_uint(a448) - 1u);
    out[idx*8 + 0] = exp2f(ceilf(log2f(fmaxf(a448,  1.0e-4f) / 448.0f)));
    out[idx*8 + 1] = exp2f(ceilf(log2f(fmaxf(a224,  1.0e-4f) / 448.0f)));
    out[idx*8 + 2] = exp2f(ceilf(log2f(fmaxf(a448u, 1.0e-4f) / 448.0f)));
    out[idx*8 + 3] = exp2f(ceilf(log2f(fmaxf(a448d, 1.0e-4f) / 448.0f)));
    // fp4 formula at amax = 6*2^k, 3*2^k, 1.5*2^k, one-ulp above 6*2^k
    float a6  = 6.0f * p2k;
    float a3  = 3.0f * p2k;
    float a15 = 1.5f * p2k;
    float a6u = __uint_as_float(__float_as_uint(a6) + 1u);
    out[idx*8 + 4] = exp2f(ceilf(log2f(fmaxf(a6,  7.052966104933725e-38f) / 6.0f)));
    out[idx*8 + 5] = exp2f(ceilf(log2f(fmaxf(a3,  7.052966104933725e-38f) / 6.0f)));
    out[idx*8 + 6] = exp2f(ceilf(log2f(fmaxf(a15, 7.052966104933725e-38f) / 6.0f)));
    out[idx*8 + 7] = exp2f(ceilf(log2f(fmaxf(a6u, 7.052966104933725e-38f) / 6.0f)));
}

static unsigned long long report_f32(const char *tag, const float *a, const float *b,
                                     uint64_t n, uint32_t row_dim) {
    unsigned long long *d_count; mismatch_rec *d_recs; int *d_nrec;
    CHECK(cudaMalloc(&d_count, sizeof(*d_count)));
    CHECK(cudaMalloc(&d_recs, MAX_RECS * sizeof(mismatch_rec)));
    CHECK(cudaMalloc(&d_nrec, sizeof(int)));
    CHECK(cudaMemset(d_count, 0, sizeof(*d_count)));
    CHECK(cudaMemset(d_nrec, 0, sizeof(int)));
    uint64_t blocks = (n + 255) / 256;
    compare_f32_kernel<<<(uint32_t)blocks, 256>>>(a, b, n, d_count, d_recs, d_nrec);
    CHECK(cudaDeviceSynchronize());
    unsigned long long count; mismatch_rec recs[MAX_RECS]; int nrec;
    CHECK(cudaMemcpy(&count, d_count, sizeof(count), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(recs, d_recs, sizeof(recs), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(&nrec, d_nrec, sizeof(nrec), cudaMemcpyDeviceToHost));
    printf("  %-28s mismatches=%llu / %llu\n", tag, count, (unsigned long long)n);
    for (int i = 0; i < nrec && i < MAX_RECS; i++) {
        uint64_t idx = ((uint64_t)recs[i].idx_hi << 32) | recs[i].idx_lo;
        float fa, fb;
        memcpy(&fa, &recs[i].a_bits, 4); memcpy(&fb, &recs[i].b_bits, 4);
        printf("    [%d] row=%llu lane=%u  a=%08x(%.9g)  b=%08x(%.9g)\n",
               i, (unsigned long long)(idx / row_dim), (uint32_t)(idx % row_dim),
               recs[i].a_bits, fa, recs[i].b_bits, fb);
    }
    cudaFree(d_count); cudaFree(d_recs); cudaFree(d_nrec);
    return count;
}

static unsigned long long report_u8(const char *tag, const unsigned char *a,
                                    const unsigned char *b, uint64_t n) {
    unsigned long long *d_count;
    CHECK(cudaMalloc(&d_count, sizeof(*d_count)));
    CHECK(cudaMemset(d_count, 0, sizeof(*d_count)));
    uint64_t blocks = (n + 255) / 256;
    compare_u8_kernel<<<(uint32_t)blocks, 256>>>(a, b, n, d_count);
    CHECK(cudaDeviceSynchronize());
    unsigned long long count;
    CHECK(cudaMemcpy(&count, d_count, sizeof(count), cudaMemcpyDeviceToHost));
    printf("  %-28s byte-mismatches=%llu / %llu (level-shift benign iff values match)\n",
           tag, count, (unsigned long long)n);
    cudaFree(d_count);
    return count;
}

int main(int argc, char **argv) {
    uint32_t n_fp8 = argc > 1 ? (uint32_t)atoi(argv[1]) : 131072u;
    uint32_t n_fp4 = argc > 2 ? (uint32_t)atoi(argv[2]) : 131072u;
    if (!fp8_decode_table_init()) { fprintf(stderr, "decode table init failed\n"); return 1; }

    int fails = 0;      /* FIXED-variant failures: the actual gate */
    int orig_fail = 0;  /* shipped-kernel failures: expected repro of the bug */

    // ---------------- fp8 comp path ----------------
    {
        const uint32_t HD = 512u, NROT = 64u;
        uint64_t xn = (uint64_t)n_fp8 * HD;
        uint64_t cb = (uint64_t)n_fp8 * DS4_OPP_C_FP8_ROW_BYTES_DEV;
        uint64_t sn = (uint64_t)n_fp8 * DS4_OPP_C_FP8_BLOCKS_DEV;
        float *x, *d1, *x2, *d2, *s1, *s2;
        unsigned char *c1, *c2;
        CHECK(cudaMalloc(&x,  xn * 4)); CHECK(cudaMalloc(&d1, xn * 4));
        CHECK(cudaMalloc(&x2, xn * 4)); CHECK(cudaMalloc(&d2, xn * 4));
        CHECK(cudaMalloc(&c1, cb)); CHECK(cudaMalloc(&c2, cb));
        CHECK(cudaMalloc(&s1, sn * 4)); CHECK(cudaMalloc(&s2, sn * 4));

        fill_fp8_input_kernel<<<n_fp8, 512>>>(x, n_fp8, 7u);
        fp8_kv_quantize_kernel<<<n_fp8, 64>>>(x, n_fp8, HD, NROT, c1, s1);      // -> C1
        fp8_kv_dequant_rows_kernel<<<n_fp8, 128>>>(d1, c1, s1, n_fp8, HD);      // D1
        CHECK(cudaMemcpy(x2, x, xn * 4, cudaMemcpyDeviceToDevice));
        fp8_kv_quantize_kernel<<<n_fp8, 64>>>(x2, n_fp8, HD, NROT, c2, s2);     // -> C2
        fp8_kv_dequant_rows_kernel<<<n_fp8, 128>>>(d2, c2, s2, n_fp8, HD);      // D2
        CHECK(cudaDeviceSynchronize());

        printf("FP8 comp path (%u rows, head_dim %u):\n", n_fp8, HD);
        unsigned long long m_read  = report_f32("read-identity D1 vs C1", d1, x,  xn, HD);
        unsigned long long m_idem  = report_f32("VALUE idempotence C2 vs C1", x2, x, xn, HD);
        unsigned long long m_read2 = report_f32("read-identity D2 vs C1", d2, x,  xn, HD);
        report_u8("codes2 vs codes1", c1, c2, cb);
        report_f32("scales2 vs scales1", s1, s2, sn, DS4_OPP_C_FP8_BLOCKS_DEV);
        // third iteration: fixed point
        fp8_kv_quantize_kernel<<<n_fp8, 64>>>(x2, n_fp8, HD, NROT, c2, s2);
        CHECK(cudaDeviceSynchronize());
        unsigned long long m_fix = report_f32("fixed-point C3 vs C2(=x2)", x2, x, xn, HD);
        int ok = (m_read == 0 && m_idem == 0 && m_read2 == 0 && m_fix == 0);
        printf("REENC-IDEM FP8: %s\n\n", ok ? "PASS" : "FAIL (repro)");
        if (!ok) orig_fail++;
        cudaFree(x); cudaFree(d1); cudaFree(x2); cudaFree(d2);
        cudaFree(c1); cudaFree(c2); cudaFree(s1); cudaFree(s2);
    }

    // ---------------- fp4 indexer path ----------------
    {
        const uint32_t HD = 128u;
        uint64_t xn = (uint64_t)n_fp4 * HD;
        uint64_t cb = (uint64_t)n_fp4 * 64u;
        uint64_t sn = (uint64_t)n_fp4 * 4u;
        float *x, *d1, *x2, *d2, *s1, *s2;
        unsigned char *c1, *c2;
        CHECK(cudaMalloc(&x,  xn * 4)); CHECK(cudaMalloc(&d1, xn * 4));
        CHECK(cudaMalloc(&x2, xn * 4)); CHECK(cudaMalloc(&d2, xn * 4));
        CHECK(cudaMalloc(&c1, cb)); CHECK(cudaMalloc(&c2, cb));
        CHECK(cudaMalloc(&s1, sn * 4)); CHECK(cudaMalloc(&s2, sn * 4));

        fill_fp4_input_kernel<<<n_fp4, 128>>>(x, n_fp4, 7u);
        indexer_hadamard_fp4_kernel<<<n_fp4, 128>>>(x, n_fp4, HD, c1, s1);       // QAT -> C1
        indexer_fp4_dequant_rows_kernel<<<n_fp4, 128>>>(d1, c1, s1, n_fp4, HD);  // D1
        CHECK(cudaMemcpy(x2, x, xn * 4, cudaMemcpyDeviceToDevice));
        indexer_fp4_encode_rows_kernel<<<n_fp4, 128>>>(x2, n_fp4, HD, c2, s2);   // load-path re-encode -> C2
        indexer_fp4_dequant_rows_kernel<<<n_fp4, 128>>>(d2, c2, s2, n_fp4, HD);  // D2
        CHECK(cudaDeviceSynchronize());

        printf("FP4 indexer path (%u rows, head_dim %u):\n", n_fp4, HD);
        unsigned long long m_read  = report_f32("read-identity D1 vs C1", d1, x,  xn, HD);
        unsigned long long m_idem  = report_f32("VALUE idempotence C2 vs C1", x2, x, xn, HD);
        unsigned long long m_read2 = report_f32("read-identity D2 vs C1", d2, x,  xn, HD);
        report_u8("codes2 vs codes1", c1, c2, cb);
        report_f32("scales2 vs scales1", s1, s2, sn, 4u);
        indexer_fp4_encode_rows_kernel<<<n_fp4, 128>>>(x2, n_fp4, HD, c2, s2);
        CHECK(cudaDeviceSynchronize());
        unsigned long long m_fix = report_f32("fixed-point C3 vs C2(=x2)", x2, x, xn, HD);
        int ok = (m_read == 0 && m_idem == 0 && m_read2 == 0 && m_fix == 0);
        printf("REENC-IDEM FP4: %s\n\n", ok ? "PASS" : "FAIL (repro)");
        if (!ok) orig_fail++;
        cudaFree(x); cudaFree(d1); cudaFree(x2); cudaFree(d2);
        cudaFree(c1); cudaFree(c2); cudaFree(s1); cudaFree(s2);
    }

    // ---------------- FIXED fp8 comp path ----------------
    {
        const uint32_t HD = 512u, NROT = 64u;
        uint64_t xn = (uint64_t)n_fp8 * HD;
        uint64_t cb = (uint64_t)n_fp8 * DS4_OPP_C_FP8_ROW_BYTES_DEV;
        uint64_t sn = (uint64_t)n_fp8 * DS4_OPP_C_FP8_BLOCKS_DEV;
        float *x, *d1, *x2, *d2, *s1, *s2;
        unsigned char *c1, *c2;
        CHECK(cudaMalloc(&x,  xn * 4)); CHECK(cudaMalloc(&d1, xn * 4));
        CHECK(cudaMalloc(&x2, xn * 4)); CHECK(cudaMalloc(&d2, xn * 4));
        CHECK(cudaMalloc(&c1, cb)); CHECK(cudaMalloc(&c2, cb));
        CHECK(cudaMalloc(&s1, sn * 4)); CHECK(cudaMalloc(&s2, sn * 4));

        fill_fp8_input_kernel<<<n_fp8, 512>>>(x, n_fp8, 7u);
        fp8_kv_quantize_kernel_fx<<<n_fp8, 64>>>(x, n_fp8, HD, NROT, c1, s1);
        fp8_kv_dequant_rows_kernel<<<n_fp8, 128>>>(d1, c1, s1, n_fp8, HD);
        CHECK(cudaMemcpy(x2, x, xn * 4, cudaMemcpyDeviceToDevice));
        fp8_kv_quantize_kernel_fx<<<n_fp8, 64>>>(x2, n_fp8, HD, NROT, c2, s2);
        fp8_kv_dequant_rows_kernel<<<n_fp8, 128>>>(d2, c2, s2, n_fp8, HD);
        CHECK(cudaDeviceSynchronize());

        printf("FIXED FP8 comp path (frexpf scale + signbit; %u rows):\n", n_fp8);
        unsigned long long m_read  = report_f32("read-identity D1 vs C1", d1, x,  xn, HD);
        unsigned long long m_idem  = report_f32("VALUE idempotence C2 vs C1", x2, x, xn, HD);
        unsigned long long m_read2 = report_f32("read-identity D2 vs C1", d2, x,  xn, HD);
        unsigned long long m_code  = report_u8("codes2 vs codes1", c1, c2, cb);
        unsigned long long m_scale = report_f32("scales2 vs scales1", s1, s2, sn, DS4_OPP_C_FP8_BLOCKS_DEV);
        fp8_kv_quantize_kernel_fx<<<n_fp8, 64>>>(x2, n_fp8, HD, NROT, c2, s2);
        CHECK(cudaDeviceSynchronize());
        unsigned long long m_fix = report_f32("fixed-point C3 vs C2(=x2)", x2, x, xn, HD);
        int ok = (m_read == 0 && m_idem == 0 && m_read2 == 0 && m_fix == 0);
        printf("REENC-IDEM FP8-FIXED: %s%s\n\n", ok ? "PASS" : "FAIL",
               (ok && (m_code || m_scale)) ? " (codes level-shift, values exact)" : "");
        if (!ok) fails++;
        cudaFree(x); cudaFree(d1); cudaFree(x2); cudaFree(d2);
        cudaFree(c1); cudaFree(c2); cudaFree(s1); cudaFree(s2);
    }

    // ---------------- FIXED fp4 indexer path ----------------
    {
        const uint32_t HD = 128u;
        uint64_t xn = (uint64_t)n_fp4 * HD;
        uint64_t cb = (uint64_t)n_fp4 * 64u;
        uint64_t sn = (uint64_t)n_fp4 * 4u;
        float *x, *d1, *x2, *d2, *s1, *s2;
        unsigned char *c1, *c2;
        CHECK(cudaMalloc(&x,  xn * 4)); CHECK(cudaMalloc(&d1, xn * 4));
        CHECK(cudaMalloc(&x2, xn * 4)); CHECK(cudaMalloc(&d2, xn * 4));
        CHECK(cudaMalloc(&c1, cb)); CHECK(cudaMalloc(&c2, cb));
        CHECK(cudaMalloc(&s1, sn * 4)); CHECK(cudaMalloc(&s2, sn * 4));

        fill_fp4_input_kernel<<<n_fp4, 128>>>(x, n_fp4, 7u);
        indexer_hadamard_fp4_kernel_fx<<<n_fp4, 128>>>(x, n_fp4, HD, c1, s1);
        indexer_fp4_dequant_rows_kernel<<<n_fp4, 128>>>(d1, c1, s1, n_fp4, HD);
        CHECK(cudaMemcpy(x2, x, xn * 4, cudaMemcpyDeviceToDevice));
        indexer_fp4_encode_rows_kernel_fx<<<n_fp4, 128>>>(x2, n_fp4, HD, c2, s2);
        indexer_fp4_dequant_rows_kernel<<<n_fp4, 128>>>(d2, c2, s2, n_fp4, HD);
        CHECK(cudaDeviceSynchronize());

        printf("FIXED FP4 indexer path (frexpf scale + signbit; %u rows):\n", n_fp4);
        unsigned long long m_read  = report_f32("read-identity D1 vs C1", d1, x,  xn, HD);
        unsigned long long m_idem  = report_f32("VALUE idempotence C2 vs C1", x2, x, xn, HD);
        unsigned long long m_read2 = report_f32("read-identity D2 vs C1", d2, x,  xn, HD);
        unsigned long long m_code  = report_u8("codes2 vs codes1", c1, c2, cb);
        unsigned long long m_scale = report_f32("scales2 vs scales1", s1, s2, sn, 4u);
        indexer_fp4_encode_rows_kernel_fx<<<n_fp4, 128>>>(x2, n_fp4, HD, c2, s2);
        CHECK(cudaDeviceSynchronize());
        unsigned long long m_fix = report_f32("fixed-point C3 vs C2(=x2)", x2, x, xn, HD);
        int ok = (m_read == 0 && m_idem == 0 && m_read2 == 0 && m_fix == 0);
        printf("REENC-IDEM FP4-FIXED: %s%s\n\n", ok ? "PASS" : "FAIL",
               (ok && (m_code || m_scale)) ? " (codes level-shift, values exact)" : "");
        if (!ok) fails++;
        cudaFree(x); cudaFree(d1); cudaFree(x2); cudaFree(d2);
        cudaFree(c1); cudaFree(c2); cudaFree(s1); cudaFree(s2);
    }

    // ---------------- fast-math boundary scan ----------------
    {
        float *d_out;
        CHECK(cudaMalloc(&d_out, 61 * 8 * sizeof(float)));
        boundary_scan_kernel<<<1, 64>>>(d_out);
        CHECK(cudaDeviceSynchronize());
        float out[61 * 8];
        CHECK(cudaMemcpy(out, d_out, sizeof(out), cudaMemcpyDeviceToHost));
        cudaFree(d_out);
        int devs = 0, ulps = 0;
        printf("Fast-math boundary scan (exact-pow2 amax inputs, k=-30..30):\n");
        for (int idx = 0; idx < 61; idx++) {
            int k = idx - 30;
            float p2k  = ldexpf(1.0f, k);
            float p2k1 = ldexpf(1.0f, k - 1);
            // expectations from exact real arithmetic; skip cases where the
            // 1e-4 amax floor binds instead.  `ulp` cases (one-ulp-off amax)
            // probe approximation sensitivity, not a re-encode reachable
            // state -- tallied separately as INFO.
            struct { const char *name; float got, want; int skip, ulp; } cases[8] = {
                {"fp8 448*2^k ", out[idx*8+0], p2k,  448.0f*p2k < 1.0e-4f, 0},
                {"fp8 224*2^k ", out[idx*8+1], p2k1, 224.0f*p2k < 1.0e-4f, 0},
                {"fp8 448u    ", out[idx*8+2], 2.0f*p2k, 448.0f*p2k < 1.0e-4f, 1},
                {"fp8 448d    ", out[idx*8+3], p2k,  448.0f*p2k < 1.0e-4f, 1},
                {"fp4 6*2^k   ", out[idx*8+4], p2k,  0, 0},
                {"fp4 3*2^k   ", out[idx*8+5], p2k1, 0, 0},
                {"fp4 1.5*2^k ", out[idx*8+6], ldexpf(1.0f, k-2), 0, 0},
                {"fp4 6u      ", out[idx*8+7], 2.0f*p2k, 0, 1},
            };
            for (int c = 0; c < 8; c++) {
                if (cases[c].skip) continue;
                if (cases[c].got != cases[c].want) {
                    printf("  %s k=%3d %s got=%.9g want=%.9g\n",
                           cases[c].ulp ? "ulp-info " : "DEVIATION",
                           k, cases[c].name, cases[c].got, cases[c].want);
                    if (cases[c].ulp) ulps++; else devs++;
                }
            }
        }
        printf("boundary scan: exact-case deviations=%d%s, ulp-info=%d\n\n", devs,
               devs ? " (MECHANISM CANDIDATE -- fast-math log2f/ceilf at pow2)" : "", ulps);
    }

    printf("PROTO-KV-REENC-IDEM: shipped kernels %s, fixed variants %s\n",
           orig_fail ? "REPRODUCE the bug" : "PASS (bug not reproduced?!)",
           fails == 0 ? "PASS" : "FAIL");
    return fails != 0;
}
