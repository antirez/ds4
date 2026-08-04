// SPDX-License-Identifier: MIT
// proto_indexer_score.cu - indexer score rewrite probe.
//
// Isolates indexer_scores_multiseq_kernel and ten variants: V1/V3 retain a
// bit-identical F32 oracle, while V2/V2b/V4/V5/V5B/V5C/V5C32/V5D use a <=2-ULP
// accumulation-order oracle.  Build with the engine's fast-math mode for the GB10 target
// (48 SMs, 100 KB smem/SM, 64K regs/SM):
// /usr/local/cuda/bin/nvcc -O3 -std=c++17 --use_fast_math -arch=sm_121 cuda/mmq/test/proto_indexer_score.cu -o proto_indexer_score
// argv: [n_comp_deep=59082]

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>

namespace wmma = nvcuda::wmma;

#define CHECK(call) do { cudaError_t e_=(call); if(e_!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s @%d: %s\n",#call,__LINE__,cudaGetErrorString(e_)); exit(1);} } while(0)

#define TOPK_K 2048u

// ---------------------------------------------------------------------------
// BASELINE -- verbatim engine math from the current ds4_cuda.cu.
//
// The engine-only DS4_CUDA_UNUSED attribute on indexer_fp4_read and the
// g_fp4_index_read_path_blocks stats atomic in the score kernel are omitted:
// both are instrumentation/lint artifacts with no effect on score math.
// ---------------------------------------------------------------------------

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

__device__ static float dsv4_pow2_ceil_scale(float r) {
    int e; float m = frexpf(r, &e);          /* r = m * 2^e, m in [0.5, 1) */
    return ldexpf(1.0f, m == 0.5f ? e - 1 : e);
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

__device__ static float indexer_fp4_level_read(
        const unsigned char * __restrict__ codes_base,
        uint32_t row, uint32_t d) {
    const unsigned char byte = codes_base[(uint64_t)row * 64u + (d >> 1u)];
    const uint32_t nib = ((uint32_t)byte >> (4u * (d & 1u))) & 0xFu;
    const float sign = (nib & 8u) ? -1.0f : 1.0f;
    return sign * dsv4_e2m1fn_value_dev((int)(nib & 7u));
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
    float scale = dsv4_pow2_ceil_scale(amax / 6.0f);
    float clamp = fminf(6.0f, fmaxf(-6.0f, v / scale));
    int level = dsv4_e2m1fn_level_dev(fabsf(clamp));
    xr[tid] = ((signbit(clamp) ? -1.0f : 1.0f) * dsv4_e2m1fn_value_dev(level)) * scale;
    uint32_t nib = (clamp < 0.0f ? 8u : 0u) | (uint32_t)level;
    uint32_t hi  = __shfl_down_sync(0xffffffffu, nib, 1u);
    if ((tid & 1u) == 0u)  codes_base[(uint64_t)row * 64u + (tid >> 1u)] = (unsigned char)(nib | (hi << 4u));
    if ((tid & 31u) == 0u) scale_base[(uint64_t)row * 4u + (tid >> 5u)] = scale;
}

__global__ static void indexer_block_scale_rows_kernel(
        const float * __restrict__ x, uint32_t n_rows,
        float * __restrict__ scale_out) {
    const uint32_t row = blockIdx.x;
    const uint32_t lane = threadIdx.x;
    if (row >= n_rows || lane >= 32u) return;

    #pragma unroll
    for (uint32_t block = 0; block < 4u; block++) {
        const float v = x[(uint64_t)row * 128u + block * 32u + lane];
        const float block_max_abs = warp_max_f32(fabsf(v));
        if (lane == 0u) {
            const float amax = fmaxf(
                block_max_abs, 7.052966104933725e-38f);
            scale_out[(uint64_t)row * 4u + block] =
                dsv4_pow2_ceil_scale(amax / 6.0f);
        }
    }
}

__global__ static void indexer_scores_multiseq_kernel(
        float *scores,            /* [n_tokens, n_comp]                       */
        const float *q,           /* [n_tokens, 64, 128]                      */
        const float *weights,     /* [n_tokens, 64]                           */
        const float *index_comp,  /* [N_banks * comp_cap, 128]                */
        uint32_t n_comp,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t ratio,
        uint32_t comp_cap,
        float scale,
        int causal,
        const int32_t * __restrict__ positions,
        const int32_t * __restrict__ seq_id,
        const unsigned char * __restrict__ index_fp4,
        const float         * __restrict__ index_scale) {
    const uint32_t c   = blockIdx.x;
    const uint32_t t   = blockIdx.y;
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t warp = tid >> 5u;
    if (c >= n_comp || t >= n_tokens || tid >= 128u) return;
    const uint32_t qpos = positions ? (uint32_t)positions[t] : pos0;
    if (causal) {
        uint32_t visible = ratio ? (qpos + 1u) / ratio : n_comp;
        if (positions && ratio && visible > qpos / ratio) visible = qpos / ratio;
        if (c >= visible) {
            if (tid == 0) scores[(uint64_t)t * n_comp + c] = -INFINITY;
            return;
        }
    }
    const uint32_t seq_base = seq_id ? (uint32_t)seq_id[t] * comp_cap : 0u;
    const float *qrow = q + (uint64_t)t * 64u * 128u;
    const float *wrow = weights + (uint64_t)t * 64u;
    const bool use_fp4 = (index_fp4 != NULL) && (index_scale != NULL);

    __shared__ float krow[128];
    __shared__ float partial[4];
    if (tid < 128u)
        krow[tid] = use_fp4 ? indexer_fp4_read(index_fp4, index_scale, seq_base + c, tid)
                            : index_comp[(uint64_t)(seq_base + c) * 128u + tid];
    __syncthreads();

    float total = 0.0f;
    for (uint32_t h0 = 0; h0 < 64u; h0 += 4u) {
        const uint32_t h = h0 + warp;
        const float4 qv = ((const float4 *)(qrow + (uint64_t)h * 128u))[lane];
        const float4 kv = ((const float4 *)krow)[lane];
        float dot = qv.x * kv.x + qv.y * kv.y + qv.z * kv.z + qv.w * kv.w;
        dot = warp_sum_f32(dot);
        if (lane == 0) partial[warp] = fmaxf(dot, 0.0f) * wrow[h] * scale;
        __syncthreads();
        if (tid == 0) total += partial[0] + partial[1] + partial[2] + partial[3];
        __syncthreads();
    }
    if (tid == 0) scores[(uint64_t)t * n_comp + c] = total;
}

// ---------------------------------------------------------------------------
// V1 -- four-row CTA tile.  Each warp reuses one Q float4 across four rows.
// Ping-pong partial buffers make the next producer phase independent of the
// preceding consumer, leaving one CTA barrier per head group after row load.
// ---------------------------------------------------------------------------

__global__ static void indexer_scores_multiseq_v1_kernel(
        float *scores, const float *q, const float *weights,
        const float *index_comp, uint32_t n_comp, uint32_t n_tokens,
        uint32_t pos0, uint32_t ratio, uint32_t comp_cap, float scale,
        int causal, const int32_t * __restrict__ positions,
        const int32_t * __restrict__ seq_id,
        const unsigned char * __restrict__ index_fp4,
        const float * __restrict__ index_scale) {
    const uint32_t tile = blockIdx.x;
    const uint32_t t = blockIdx.y;
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t warp = tid >> 5u;
    if (t >= n_tokens || tid >= 128u) return;

    const uint32_t qpos = positions ? (uint32_t)positions[t] : pos0;
    uint32_t visible = n_comp;
    if (causal) {
        visible = ratio ? (qpos + 1u) / ratio : n_comp;
        if (positions && ratio && visible > qpos / ratio) visible = qpos / ratio;
    }
    const uint32_t seq_base = seq_id ? (uint32_t)seq_id[t] * comp_cap : 0u;
    const float *qrow = q + (uint64_t)t * 64u * 128u;
    const float *wrow = weights + (uint64_t)t * 64u;
    const bool use_fp4 = (index_fp4 != NULL) && (index_scale != NULL);

    bool active[4];
    #pragma unroll
    for (uint32_t r = 0; r < 4u; r++) {
        const uint32_t c = tile * 4u + r;
        active[r] = c < n_comp && (!causal || c < visible);
        if (c < n_comp && causal && c >= visible && tid == 0)
            scores[(uint64_t)t * n_comp + c] = -INFINITY;
    }

    __shared__ float krow[4][128];
    __shared__ float partial[2][4][4];
    #pragma unroll
    for (uint32_t r = 0; r < 4u; r++) {
        const uint32_t c = tile * 4u + r;
        if (active[r])
            krow[r][tid] = use_fp4 ? indexer_fp4_read(index_fp4, index_scale, seq_base + c, tid)
                                    : index_comp[(uint64_t)(seq_base + c) * 128u + tid];
    }
    __syncthreads();

    float total[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint32_t h0 = 0; h0 < 64u; h0 += 4u) {
        const uint32_t h = h0 + warp;
        const uint32_t buf = (h0 >> 2u) & 1u;
        const float4 qv = ((const float4 *)(qrow + (uint64_t)h * 128u))[lane];
        #pragma unroll
        for (uint32_t r = 0; r < 4u; r++) {
            if (active[r]) {
                const float4 kv = ((const float4 *)krow[r])[lane];
                float dot = qv.x * kv.x + qv.y * kv.y + qv.z * kv.z + qv.w * kv.w;
                dot = warp_sum_f32(dot);
                if (lane == 0) partial[buf][r][warp] = fmaxf(dot, 0.0f) * wrow[h] * scale;
            }
        }
        __syncthreads();
        if (tid == 0) {
            #pragma unroll
            for (uint32_t r = 0; r < 4u; r++) {
                if (active[r])
                    total[r] += partial[buf][r][0] + partial[buf][r][1]
                              + partial[buf][r][2] + partial[buf][r][3];
            }
        }
        // The next group writes the other buffer.  Its barrier also ensures
        // thread 0 finished this read before this buffer is reused in group+2.
    }

    if (tid == 0) {
        #pragma unroll
        for (uint32_t r = 0; r < 4u; r++) {
            const uint32_t c = tile * 4u + r;
            if (active[r]) scores[(uint64_t)t * n_comp + c] = total[r];
        }
    }
}

// ---------------------------------------------------------------------------
// V2 -- 256-thread, eight-warp autonomous row streaming.
//
// A physical warp does four heads sequentially, but each head's four-product
// expression and fixed five-shuffle reduction depend only on that head's Q/K
// values.  Lane 0 then evaluates partial[0]+partial[1]+partial[2]+partial[3]
// in the same source-level left-to-right group expression and adds groups in
// ascending h0 order.  Thus no cross-warp arithmetic is being reordered.
// There is no shared krow, partial array, or block-wide barrier in the loop.
// ---------------------------------------------------------------------------

#define V2_ROWS_PER_BLOCK 64u

__global__ static void indexer_scores_multiseq_v2_kernel(
        float *scores, const float *q, const float *weights,
        const float *index_comp, uint32_t n_comp, uint32_t n_tokens,
        uint32_t pos0, uint32_t ratio, uint32_t comp_cap, float scale,
        int causal, const int32_t * __restrict__ positions,
        const int32_t * __restrict__ seq_id,
        const unsigned char * __restrict__ index_fp4,
        const float * __restrict__ index_scale) {
    const uint32_t t = blockIdx.y;
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t warp = tid >> 5u;
    if (t >= n_tokens || tid >= 256u) return;

    const uint32_t qpos = positions ? (uint32_t)positions[t] : pos0;
    uint32_t visible = n_comp;
    if (causal) {
        visible = ratio ? (qpos + 1u) / ratio : n_comp;
        if (positions && ratio && visible > qpos / ratio) visible = qpos / ratio;
    }
    const uint32_t seq_base = seq_id ? (uint32_t)seq_id[t] * comp_cap : 0u;
    const float *qrow = q + (uint64_t)t * 64u * 128u;
    const float *wrow = weights + (uint64_t)t * 64u;
    const bool use_fp4 = (index_fp4 != NULL) && (index_scale != NULL);
    const uint32_t chunk_base = blockIdx.x * V2_ROWS_PER_BLOCK;
    const uint32_t unbounded_end = chunk_base + V2_ROWS_PER_BLOCK;
    const uint32_t chunk_end = unbounded_end < n_comp ? unbounded_end : n_comp;

    for (uint32_t c = chunk_base + warp; c < chunk_end; c += 8u) {
        if (causal && c >= visible) {
            if (lane == 0) scores[(uint64_t)t * n_comp + c] = -INFINITY;
            continue;
        }

        float4 kv;
        if (use_fp4) {
            const uint32_t d = lane * 4u;
            kv.x = indexer_fp4_read(index_fp4, index_scale, seq_base + c, d + 0u);
            kv.y = indexer_fp4_read(index_fp4, index_scale, seq_base + c, d + 1u);
            kv.z = indexer_fp4_read(index_fp4, index_scale, seq_base + c, d + 2u);
            kv.w = indexer_fp4_read(index_fp4, index_scale, seq_base + c, d + 3u);
        } else {
            const float *kbase = index_comp + (uint64_t)(seq_base + c) * 128u;
            kv = ((const float4 *)kbase)[lane];
        }

        float total = 0.0f;
        for (uint32_t h0 = 0; h0 < 64u; h0 += 4u) {
            float part[4];
            #pragma unroll
            for (uint32_t j = 0; j < 4u; j++) {
                const uint32_t h = h0 + j;
                const float4 qv = ((const float4 *)(qrow + (uint64_t)h * 128u))[lane];
                float dot = qv.x * kv.x + qv.y * kv.y + qv.z * kv.z + qv.w * kv.w;
                dot = warp_sum_f32(dot);
                part[j] = fmaxf(dot, 0.0f) * wrow[h] * scale;
            }
            if (lane == 0) total += part[0] + part[1] + part[2] + part[3];
        }
        if (lane == 0) scores[(uint64_t)t * n_comp + c] = total;
    }
}

// ---------------------------------------------------------------------------
// V2b -- V2 scheduling with a word-vectorized packed-FP4 read path.
//
// Eight lanes load the row's eight naturally aligned u64 code words.  Each
// word is broadcast to the four lanes that consume its sixteen nibbles; the
// four block scales are likewise loaded once and broadcast to eight lanes.
// The F32 path and all score arithmetic remain identical to V2.
// ---------------------------------------------------------------------------

__global__ static void indexer_scores_multiseq_v2b_kernel(
        float *scores, const float *q, const float *weights,
        const float *index_comp, uint32_t n_comp, uint32_t n_tokens,
        uint32_t pos0, uint32_t ratio, uint32_t comp_cap, float scale,
        int causal, const int32_t * __restrict__ positions,
        const int32_t * __restrict__ seq_id,
        const unsigned char * __restrict__ index_fp4,
        const float * __restrict__ index_scale) {
    const uint32_t t = blockIdx.y;
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t warp = tid >> 5u;
    if (t >= n_tokens || tid >= 256u) return;

    const uint32_t qpos = positions ? (uint32_t)positions[t] : pos0;
    uint32_t visible = n_comp;
    if (causal) {
        visible = ratio ? (qpos + 1u) / ratio : n_comp;
        if (positions && ratio && visible > qpos / ratio) visible = qpos / ratio;
    }
    const uint32_t seq_base = seq_id ? (uint32_t)seq_id[t] * comp_cap : 0u;
    const float *qrow = q + (uint64_t)t * 64u * 128u;
    const float *wrow = weights + (uint64_t)t * 64u;
    const bool use_fp4 = (index_fp4 != NULL) && (index_scale != NULL);
    const uint32_t chunk_base = blockIdx.x * V2_ROWS_PER_BLOCK;
    const uint32_t unbounded_end = chunk_base + V2_ROWS_PER_BLOCK;
    const uint32_t chunk_end = unbounded_end < n_comp ? unbounded_end : n_comp;

    for (uint32_t c = chunk_base + warp; c < chunk_end; c += 8u) {
        if (causal && c >= visible) {
            if (lane == 0) scores[(uint64_t)t * n_comp + c] = -INFINITY;
            continue;
        }

        float4 kv;
        if (use_fp4) {
            const uint32_t row = seq_base + c;
            const uint64_t *code_words = reinterpret_cast<const uint64_t *>(
                index_fp4 + (uint64_t)row * 64u);
            uint64_t code_word = lane < 8u ? code_words[lane] : 0ull;
            code_word = __shfl_sync(0xffffffffu, code_word, lane >> 2u);
            const uint32_t packed = (uint32_t)(code_word >> ((lane & 3u) * 16u));

            const float *row_scales = index_scale + (uint64_t)row * 4u;
            float block_scale = lane < 4u ? row_scales[lane] : 0.0f;
            block_scale = __shfl_sync(0xffffffffu, block_scale, lane >> 3u);

            const uint32_t n0 = (packed >> 0u) & 0xFu;
            const uint32_t n1 = (packed >> 4u) & 0xFu;
            const uint32_t n2 = (packed >> 8u) & 0xFu;
            const uint32_t n3 = (packed >> 12u) & 0xFu;
            kv.x = ((n0 & 8u) ? -1.0f : 1.0f)
                 * dsv4_e2m1fn_value_dev((int)(n0 & 7u)) * block_scale;
            kv.y = ((n1 & 8u) ? -1.0f : 1.0f)
                 * dsv4_e2m1fn_value_dev((int)(n1 & 7u)) * block_scale;
            kv.z = ((n2 & 8u) ? -1.0f : 1.0f)
                 * dsv4_e2m1fn_value_dev((int)(n2 & 7u)) * block_scale;
            kv.w = ((n3 & 8u) ? -1.0f : 1.0f)
                 * dsv4_e2m1fn_value_dev((int)(n3 & 7u)) * block_scale;
        } else {
            const float *kbase = index_comp + (uint64_t)(seq_base + c) * 128u;
            kv = ((const float4 *)kbase)[lane];
        }

        float total = 0.0f;
        for (uint32_t h0 = 0; h0 < 64u; h0 += 4u) {
            float part[4];
            #pragma unroll
            for (uint32_t j = 0; j < 4u; j++) {
                const uint32_t h = h0 + j;
                const float4 qv = ((const float4 *)(qrow + (uint64_t)h * 128u))[lane];
                float dot = qv.x * kv.x + qv.y * kv.y + qv.z * kv.z + qv.w * kv.w;
                dot = warp_sum_f32(dot);
                part[j] = fmaxf(dot, 0.0f) * wrow[h] * scale;
            }
            if (lane == 0) total += part[0] + part[1] + part[2] + part[3];
        }
        if (lane == 0) scores[(uint64_t)t * n_comp + c] = total;
    }
}

// ---------------------------------------------------------------------------
// V3 -- one-row stopgap: next-group Q prefetch plus ping-pong partials.
// The dot, ReLU/weight/scale placement, and thread-0 fold are unchanged.
// ---------------------------------------------------------------------------

__global__ static void indexer_scores_multiseq_v3_kernel(
        float *scores, const float *q, const float *weights,
        const float *index_comp, uint32_t n_comp, uint32_t n_tokens,
        uint32_t pos0, uint32_t ratio, uint32_t comp_cap, float scale,
        int causal, const int32_t * __restrict__ positions,
        const int32_t * __restrict__ seq_id,
        const unsigned char * __restrict__ index_fp4,
        const float * __restrict__ index_scale) {
    const uint32_t c = blockIdx.x;
    const uint32_t t = blockIdx.y;
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t warp = tid >> 5u;
    if (c >= n_comp || t >= n_tokens || tid >= 128u) return;
    const uint32_t qpos = positions ? (uint32_t)positions[t] : pos0;
    if (causal) {
        uint32_t visible = ratio ? (qpos + 1u) / ratio : n_comp;
        if (positions && ratio && visible > qpos / ratio) visible = qpos / ratio;
        if (c >= visible) {
            if (tid == 0) scores[(uint64_t)t * n_comp + c] = -INFINITY;
            return;
        }
    }
    const uint32_t seq_base = seq_id ? (uint32_t)seq_id[t] * comp_cap : 0u;
    const float *qrow = q + (uint64_t)t * 64u * 128u;
    const float *wrow = weights + (uint64_t)t * 64u;
    const bool use_fp4 = (index_fp4 != NULL) && (index_scale != NULL);

    __shared__ float krow[128];
    __shared__ float partial[2][4];
    krow[tid] = use_fp4 ? indexer_fp4_read(index_fp4, index_scale, seq_base + c, tid)
                        : index_comp[(uint64_t)(seq_base + c) * 128u + tid];
    __syncthreads();

    const float4 kv = ((const float4 *)krow)[lane];
    float4 qv = ((const float4 *)(qrow + (uint64_t)warp * 128u))[lane];
    float total = 0.0f;
    for (uint32_t h0 = 0; h0 < 64u; h0 += 4u) {
        const uint32_t h = h0 + warp;
        const uint32_t buf = (h0 >> 2u) & 1u;
        float dot = qv.x * kv.x + qv.y * kv.y + qv.z * kv.z + qv.w * kv.w;
        dot = warp_sum_f32(dot);

        // Issue the next head-group load before this group's producer barrier.
        float4 qnext = qv;
        if (h0 + 4u < 64u)
            qnext = ((const float4 *)(qrow + (uint64_t)(h + 4u) * 128u))[lane];

        if (lane == 0) partial[buf][warp] = fmaxf(dot, 0.0f) * wrow[h] * scale;
        __syncthreads();
        if (tid == 0)
            total += partial[buf][0] + partial[buf][1] + partial[buf][2] + partial[buf][3];
        qv = qnext;
        // The next group uses the other buffer; its barrier closes the reuse
        // hazard before group+2 writes this buffer again.
    }
    if (tid == 0) scores[(uint64_t)t * n_comp + c] = total;
}

// ---------------------------------------------------------------------------
// V4 -- production-shaped 128-row x 16-token WMMA tile.
//
// The decode harness has one sequence/token, so the K tile's bank is selected
// from the first token in this token tile.  Token bounds and score stores keep
// the production 16-token tile shape valid for a partial final token tile.
// ---------------------------------------------------------------------------

__global__ static void indexer_scores_multiseq_v4_kernel(
        float *scores, const float *q, const float *weights,
        const float *index_comp, uint32_t n_comp, uint32_t n_tokens,
        uint32_t pos0, uint32_t ratio, uint32_t comp_cap, float scale,
        int causal, const int32_t * __restrict__ positions,
        const int32_t * __restrict__ seq_id,
        const unsigned char * __restrict__ index_fp4,
        const float * __restrict__ index_scale) {
    const uint32_t tile_c = blockIdx.x * 128u;
    const uint32_t tile_t = blockIdx.y * 16u;
    const uint32_t t = tile_t;
    const uint32_t tid = threadIdx.x;
    const uint32_t warp = tid >> 5u;
    if (tid >= 256u) return;
    const bool use_fp4 = (index_fp4 != NULL) && (index_scale != NULL);
    const uint32_t seq_base = seq_id ? (uint32_t)seq_id[t] * comp_cap : 0u;

    __shared__ __half a_sh[16 * 128];
    __shared__ __half b_sh[128 * 128];
    __shared__ float c_sh[8 * 16 * 16];

    float acc[8];
    #pragma unroll
    for (uint32_t i = 0; i < 8u; i++) acc[i] = 0.0f;

    for (uint32_t i = tid; i < 128u * 128u; i += 256u) {
        const uint32_t c = i >> 7u;
        const uint32_t d = i & 127u;
        const uint32_t comp = tile_c + c;
        float v = 0.0f;
        if (comp < n_comp) {
            v = use_fp4 ? indexer_fp4_read(index_fp4, index_scale, seq_base + comp, d)
                        : index_comp[(uint64_t)(seq_base + comp) * 128u + d];
        }
        b_sh[d + c * 128u] = __float2half(v);
    }
    __syncthreads();

    for (uint32_t h = 0; h < 64u; h++) {
        for (uint32_t i = tid; i < 16u * 128u; i += 256u) {
            const uint32_t r = i >> 7u;
            const uint32_t d = i & 127u;
            const uint32_t token = tile_t + r;
            float v = 0.0f;
            if (token < n_tokens)
                v = q[((uint64_t)token * 64u + h) * 128u + d];
            a_sh[i] = __float2half(v);
        }
        __syncthreads();

        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_frag;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
        wmma::fill_fragment(c_frag, 0.0f);
        const uint32_t col0 = warp * 16u;
        for (uint32_t k0 = 0; k0 < 128u; k0 += 16u) {
            wmma::load_matrix_sync(a_frag, a_sh + k0, 128);
            wmma::load_matrix_sync(b_frag, b_sh + col0 * 128u + k0, 128);
            // This is V4's deliberate divergence from BASELINE: MMA uses
            // half inputs with an internal 128-wide k-reduction into F32,
            // instead of float-in four-products/lane plus the five-shuffle
            // tree; below, heads also fold one-at-a-time rather than through
            // BASELINE's four-head expression before the running total.
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        wmma::store_matrix_sync(c_sh + warp * 16u * 16u,
                                c_frag, 16, wmma::mem_row_major);
        __syncthreads();

        const uint32_t local0 = tid & 255u;
        const uint32_t token0 = tile_t + (local0 >> 4u);
        const float w0 = token0 < n_tokens
            ? weights[(uint64_t)token0 * 64u + h]
            : 0.0f;
        uint32_t slot = 0;
        for (uint32_t i = tid; i < 8u * 16u * 16u; i += 256u, slot++) {
            const uint32_t wtile = i >> 8u;
            const uint32_t local = i & 255u;
            const uint32_t r = local >> 4u;
            const uint32_t c = local & 15u;
            const uint32_t token = tile_t + r;
            const uint32_t comp = tile_c + wtile * 16u + c;
            if (token < n_tokens && comp < n_comp)
                acc[slot] += fmaxf(c_sh[i], 0.0f) * w0 * scale;
        }
        __syncthreads();
    }

    uint32_t slot = 0;
    for (uint32_t i = tid; i < 8u * 16u * 16u; i += 256u, slot++) {
        const uint32_t wtile = i >> 8u;
        const uint32_t local = i & 255u;
        const uint32_t r = local >> 4u;
        const uint32_t c = local & 15u;
        const uint32_t token = tile_t + r;
        const uint32_t comp = tile_c + wtile * 16u + c;
        if (token < n_tokens && comp < n_comp) {
            float out = acc[slot];
            if (causal) {
                const uint32_t qpos = positions ? (uint32_t)positions[token] : pos0;
                uint32_t visible = ratio ? (qpos + 1u) / ratio : n_comp;
                if (positions && ratio && visible > qpos / ratio) visible = qpos / ratio;
                if (comp >= visible) out = -INFINITY;
            }
            scores[(uint64_t)token * n_comp + comp] = out;
        }
    }
}

// ---------------------------------------------------------------------------
// V5 -- 16-comp-row x 16-head WMMA tile for one token per CTA.
//
// The four 16-head tiles remain inside the CTA so every row is reduced over
// all 64 heads before its single score store.  Each 32-wide scale block is a
// separate pair of 16-wide MMA steps, allowing both committed-K legs to use
// unscaled half operands and apply K/Q block scales to the F32 chunk result.
// ---------------------------------------------------------------------------

__global__ static void indexer_scores_multiseq_v5_kernel(
        float *scores, const float *q, const float *weights,
        const float *index_comp, uint32_t n_comp, uint32_t n_tokens,
        uint32_t pos0, uint32_t ratio, uint32_t comp_cap, float scale,
        int causal, const int32_t * __restrict__ positions,
        const int32_t * __restrict__ seq_id,
        const unsigned char * __restrict__ index_fp4,
        const float * __restrict__ index_scale,
        const float * __restrict__ q_block_scale) {
    const uint32_t tile_c = blockIdx.x * 16u;
    const uint32_t t = blockIdx.y;
    const uint32_t tid = threadIdx.x;
    if (t >= n_tokens || tid >= 32u) return;

    const bool use_fp4 = (index_fp4 != NULL) && (index_scale != NULL);
    const uint32_t seq_base = seq_id ? (uint32_t)seq_id[t] * comp_cap : 0u;
    const float *wrow = weights + (uint64_t)t * 64u;

    __shared__ __half a_sh[16 * 128];
    __shared__ __half b_sh[16 * 128];
    __shared__ float c_sh[16 * 16];
    __shared__ float k_scale_block[16];

    float row_total = 0.0f;
    for (uint32_t head0 = 0; head0 < 64u; head0 += 16u) {
        for (uint32_t i = tid; i < 16u * 128u; i += 32u) {
            const uint32_t h = i >> 7u;
            const uint32_t d = i & 127u;
            const uint32_t head = head0 + h;
            float v = q[((uint64_t)t * 64u + head) * 128u + d];
            const float qs = q_block_scale[
                ((uint64_t)t * 64u + head) * 4u + (d >> 5u)];
            v /= qs;
            b_sh[i] = __float2half(v);
        }
        __syncthreads();

        float head_total[8];
        #pragma unroll
        for (uint32_t slot = 0; slot < 8u; slot++) head_total[slot] = 0.0f;

        #pragma unroll
        for (uint32_t block = 0; block < 4u; block++) {
            #pragma unroll
            for (uint32_t r = 0; r < 16u; r++) {
                const uint32_t comp = tile_c + r;
                const uint32_t d = block * 32u + tid;
                float v = 0.0f;
                if (comp < n_comp) {
                    v = use_fp4
                        ? indexer_fp4_level_read(index_fp4, seq_base + comp, d)
                        : index_comp[(uint64_t)(seq_base + comp) * 128u + d];
                }
                if (use_fp4) {
                    if (tid == 0u) {
                        k_scale_block[r] = comp < n_comp
                            ? index_scale[(uint64_t)(seq_base + comp) * 4u + block]
                            : 1.0f;
                    }
                } else {
                    const float block_max_abs = warp_max_f32(fabsf(v));
                    if (tid == 0u) {
                        const float amax = fmaxf(
                            block_max_abs, 7.052966104933725e-38f);
                        k_scale_block[r] = dsv4_pow2_ceil_scale(amax / 6.0f);
                    }
                }
                __syncwarp();
                const float level = use_fp4 ? v : v / k_scale_block[r];
                a_sh[r * 128u + d] = __float2half(level);
            }
            __syncwarp();

            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_frag;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
            wmma::fill_fragment(c_frag, 0.0f);
            #pragma unroll
            for (uint32_t k = 0; k < 32u; k += 16u) {
                const uint32_t k0 = block * 32u + k;
                wmma::load_matrix_sync(a_frag, a_sh + k0, 128);
                wmma::load_matrix_sync(b_frag, b_sh + k0, 128);
                wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            }
            wmma::store_matrix_sync(c_sh, c_frag, 16, wmma::mem_row_major);
            __syncthreads();

            #pragma unroll
            for (uint32_t slot = 0; slot < 8u; slot++) {
                const uint32_t i = tid + slot * 32u;
                const uint32_t r = i >> 4u;
                const uint32_t h = i & 15u;
                const uint32_t comp = tile_c + r;
                if (comp < n_comp) {
                    float chunk = c_sh[i];
                    const float ks = k_scale_block[r];
                    const float qs = q_block_scale[
                        ((uint64_t)t * 64u + head0 + h) * 4u + block];
                    chunk *= ks * qs;
                    // For committed on-grid K/Q in both storage legs, the
                    // remaining BASELINE deviation is F32 accumulation order:
                    // the MMA's internal 32-wide reduction, then ascending
                    // blocks and ascending heads.  Normally each exact half
                    // operand is in {0,.5,1,1.5,2,3,4,6}, so products are <=36
                    // and a 32-term chunk is <=1152.  If committed K's own
                    // amax level is exactly 3, scale re-derivation intentionally
                    // returns half the commit-time scale; that expected edge
                    // can double the K quotient and is reported by the distinct
                    // K-F32-unscaled preflight rather than special-cased here.
                    head_total[slot] += chunk;
                }
            }
            __syncthreads();
        }

        #pragma unroll
        for (uint32_t slot = 0; slot < 8u; slot++)
            c_sh[tid + slot * 32u] = head_total[slot];
        __syncthreads();

        if (tid < 16u) {
            float head_tile_total = 0.0f;
            #pragma unroll
            for (uint32_t h = 0; h < 16u; h++) {
                const uint32_t head = head0 + h;
                head_tile_total += fmaxf(c_sh[tid * 16u + h], 0.0f)
                                 * wrow[head] * scale;
            }
            row_total += head_tile_total;
        }
        __syncthreads();
    }

    if (tid < 16u) {
        const uint32_t comp = tile_c + tid;
        if (comp < n_comp) {
            float out = row_total;
            if (causal) {
                const uint32_t qpos = positions ? (uint32_t)positions[t] : pos0;
                uint32_t visible = ratio ? (qpos + 1u) / ratio : n_comp;
                if (positions && ratio && visible > qpos / ratio) visible = qpos / ratio;
                if (comp >= visible) out = -INFINITY;
            }
            scores[(uint64_t)t * n_comp + comp] = out;
        }
    }
}

// ---------------------------------------------------------------------------
// V5B -- four-warp V5, retaining the 16-row CTA tile.
//
// Warp w owns heads [16*w, 16*w+15].  All four warps cooperatively refresh the
// shared 16x32 K slice for each scale block, while each warp keeps a private
// 16x128 Q slice and WMMA result.  Per-row warp partials are folded by warp 0
// in ascending warp/head-tile order before the single causally-masked store.
// ---------------------------------------------------------------------------

__global__ static void indexer_scores_multiseq_v5b_kernel(
        float *scores, const float *q, const float *weights,
        const float *index_comp, uint32_t n_comp, uint32_t n_tokens,
        uint32_t pos0, uint32_t ratio, uint32_t comp_cap, float scale,
        int causal, const int32_t * __restrict__ positions,
        const int32_t * __restrict__ seq_id,
        const unsigned char * __restrict__ index_fp4,
        const float * __restrict__ index_scale,
        const float * __restrict__ q_block_scale) {
    const uint32_t tile_c = blockIdx.x * 16u;
    const uint32_t t = blockIdx.y;
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t warp = tid >> 5u;
    if (t >= n_tokens || tid >= 128u) return;

    const bool use_fp4 = (index_fp4 != NULL) && (index_scale != NULL);
    const uint32_t seq_base = seq_id ? (uint32_t)seq_id[t] * comp_cap : 0u;
    const uint32_t head0 = warp * 16u;
    const float *wrow = weights + (uint64_t)t * 64u;

    __shared__ __half a_sh[16 * 128];
    __shared__ __half b_sh[4 * 16 * 128];
    __shared__ float c_sh[4 * 16 * 16];
    __shared__ float k_scale_block[16];
    __shared__ float warp_row_partial[4 * 16];

    __half *warp_b_sh = b_sh + warp * 16u * 128u;
    float *warp_c_sh = c_sh + warp * 16u * 16u;
    for (uint32_t i = lane; i < 16u * 128u; i += 32u) {
        const uint32_t h = i >> 7u;
        const uint32_t d = i & 127u;
        const uint32_t head = head0 + h;
        float v = q[((uint64_t)t * 64u + head) * 128u + d];
        const float qs = q_block_scale[
            ((uint64_t)t * 64u + head) * 4u + (d >> 5u)];
        warp_b_sh[i] = __float2half(v / qs);
    }
    __syncthreads();

    float head_total[8];
    #pragma unroll
    for (uint32_t slot = 0; slot < 8u; slot++) head_total[slot] = 0.0f;

    #pragma unroll
    for (uint32_t block = 0; block < 4u; block++) {
        #pragma unroll
        for (uint32_t r = warp; r < 16u; r += 4u) {
            const uint32_t comp = tile_c + r;
            const uint32_t d = block * 32u + lane;
            float v = 0.0f;
            if (comp < n_comp) {
                v = use_fp4
                    ? indexer_fp4_level_read(index_fp4, seq_base + comp, d)
                    : index_comp[(uint64_t)(seq_base + comp) * 128u + d];
            }
            if (use_fp4) {
                if (lane == 0u) {
                    k_scale_block[r] = comp < n_comp
                        ? index_scale[(uint64_t)(seq_base + comp) * 4u + block]
                        : 1.0f;
                }
            } else {
                const float block_max_abs = warp_max_f32(fabsf(v));
                if (lane == 0u) {
                    const float amax = fmaxf(
                        block_max_abs, 7.052966104933725e-38f);
                    k_scale_block[r] = dsv4_pow2_ceil_scale(amax / 6.0f);
                }
            }
            __syncwarp();
            const float level = use_fp4 ? v : v / k_scale_block[r];
            a_sh[r * 128u + d] = __float2half(level);
        }
        __syncthreads();

        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_frag;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
        wmma::fill_fragment(c_frag, 0.0f);
        #pragma unroll
        for (uint32_t k = 0; k < 32u; k += 16u) {
            const uint32_t k0 = block * 32u + k;
            wmma::load_matrix_sync(a_frag, a_sh + k0, 128);
            wmma::load_matrix_sync(b_frag, warp_b_sh + k0, 128);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        wmma::store_matrix_sync(warp_c_sh, c_frag, 16, wmma::mem_row_major);
        __syncwarp();

        #pragma unroll
        for (uint32_t slot = 0; slot < 8u; slot++) {
            const uint32_t i = lane + slot * 32u;
            const uint32_t r = i >> 4u;
            const uint32_t h = i & 15u;
            const uint32_t comp = tile_c + r;
            if (comp < n_comp) {
                float chunk = warp_c_sh[i];
                const float qs = q_block_scale[
                    ((uint64_t)t * 64u + head0 + h) * 4u + block];
                chunk *= k_scale_block[r] * qs;
                head_total[slot] += chunk;
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (uint32_t slot = 0; slot < 8u; slot++)
        warp_c_sh[lane + slot * 32u] = head_total[slot];
    __syncthreads();

    if (lane < 16u) {
        float warp_total = 0.0f;
        #pragma unroll
        for (uint32_t h = 0; h < 16u; h++) {
            const uint32_t head = head0 + h;
            warp_total += fmaxf(warp_c_sh[lane * 16u + h], 0.0f)
                        * wrow[head] * scale;
        }
        warp_row_partial[warp * 16u + lane] = warp_total;
    }
    __syncthreads();

    if (warp == 0u && lane < 16u) {
        const uint32_t comp = tile_c + lane;
        if (comp < n_comp) {
            float out = 0.0f;
            #pragma unroll
            for (uint32_t w = 0; w < 4u; w++)
                out += warp_row_partial[w * 16u + lane];
            if (causal) {
                const uint32_t qpos = positions ? (uint32_t)positions[t] : pos0;
                uint32_t visible = ratio ? (qpos + 1u) / ratio : n_comp;
                if (positions && ratio && visible > qpos / ratio) visible = qpos / ratio;
                if (comp >= visible) out = -INFINITY;
            }
            scores[(uint64_t)t * n_comp + comp] = out;
        }
    }
}

// ---------------------------------------------------------------------------
// V5C -- V5B with both operand block scales supplied by the caller.
//
// The body is identical to V5B except that the dense-F32 K leg reads its
// already-persisted block scale instead of deriving it inside every launch.
// ---------------------------------------------------------------------------

__global__ static void indexer_scores_multiseq_v5c_kernel(
        float *scores, const float *q, const float *weights,
        const float *index_comp, uint32_t n_comp, uint32_t n_tokens,
        uint32_t pos0, uint32_t ratio, uint32_t comp_cap, float scale,
        int causal, const int32_t * __restrict__ positions,
        const int32_t * __restrict__ seq_id,
        const unsigned char * __restrict__ index_fp4,
        const float * __restrict__ index_scale,
        const float * __restrict__ q_block_scale,
        const float * __restrict__ k_block_scale) {
    const uint32_t tile_c = blockIdx.x * 16u;
    const uint32_t t = blockIdx.y;
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t warp = tid >> 5u;
    if (t >= n_tokens || tid >= 128u) return;

    const bool use_fp4 = (index_fp4 != NULL) && (index_scale != NULL);
    const uint32_t seq_base = seq_id ? (uint32_t)seq_id[t] * comp_cap : 0u;
    const uint32_t head0 = warp * 16u;
    const float *wrow = weights + (uint64_t)t * 64u;

    __shared__ __half a_sh[16 * 128];
    __shared__ __half b_sh[4 * 16 * 128];
    __shared__ float c_sh[4 * 16 * 16];
    __shared__ float k_scale_block[16];
    __shared__ float warp_row_partial[4 * 16];

    __half *warp_b_sh = b_sh + warp * 16u * 128u;
    float *warp_c_sh = c_sh + warp * 16u * 16u;
    for (uint32_t i = lane; i < 16u * 128u; i += 32u) {
        const uint32_t h = i >> 7u;
        const uint32_t d = i & 127u;
        const uint32_t head = head0 + h;
        float v = q[((uint64_t)t * 64u + head) * 128u + d];
        const float qs = q_block_scale[
            ((uint64_t)t * 64u + head) * 4u + (d >> 5u)];
        warp_b_sh[i] = __float2half(v / qs);
    }
    __syncthreads();

    float head_total[8];
    #pragma unroll
    for (uint32_t slot = 0; slot < 8u; slot++) head_total[slot] = 0.0f;

    #pragma unroll
    for (uint32_t block = 0; block < 4u; block++) {
        #pragma unroll
        for (uint32_t r = warp; r < 16u; r += 4u) {
            const uint32_t comp = tile_c + r;
            const uint32_t d = block * 32u + lane;
            float v = 0.0f;
            if (comp < n_comp) {
                v = use_fp4
                    ? indexer_fp4_level_read(index_fp4, seq_base + comp, d)
                    : index_comp[(uint64_t)(seq_base + comp) * 128u + d];
            }
            if (use_fp4) {
                if (lane == 0u) {
                    k_scale_block[r] = comp < n_comp
                        ? index_scale[(uint64_t)(seq_base + comp) * 4u + block]
                        : 1.0f;
                }
            } else {
                if (lane == 0u) {
                    k_scale_block[r] = comp < n_comp
                        ? k_block_scale[(uint64_t)(seq_base + comp) * 4u + block]
                        : 1.0f;
                }
            }
            __syncwarp();
            const float level = use_fp4 ? v : v / k_scale_block[r];
            a_sh[r * 128u + d] = __float2half(level);
        }
        __syncthreads();

        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_frag;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
        wmma::fill_fragment(c_frag, 0.0f);
        #pragma unroll
        for (uint32_t k = 0; k < 32u; k += 16u) {
            const uint32_t k0 = block * 32u + k;
            wmma::load_matrix_sync(a_frag, a_sh + k0, 128);
            wmma::load_matrix_sync(b_frag, warp_b_sh + k0, 128);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        wmma::store_matrix_sync(warp_c_sh, c_frag, 16, wmma::mem_row_major);
        __syncwarp();

        #pragma unroll
        for (uint32_t slot = 0; slot < 8u; slot++) {
            const uint32_t i = lane + slot * 32u;
            const uint32_t r = i >> 4u;
            const uint32_t h = i & 15u;
            const uint32_t comp = tile_c + r;
            if (comp < n_comp) {
                float chunk = warp_c_sh[i];
                const float qs = q_block_scale[
                    ((uint64_t)t * 64u + head0 + h) * 4u + block];
                chunk *= k_scale_block[r] * qs;
                head_total[slot] += chunk;
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (uint32_t slot = 0; slot < 8u; slot++)
        warp_c_sh[lane + slot * 32u] = head_total[slot];
    __syncthreads();

    if (lane < 16u) {
        float warp_total = 0.0f;
        #pragma unroll
        for (uint32_t h = 0; h < 16u; h++) {
            const uint32_t head = head0 + h;
            warp_total += fmaxf(warp_c_sh[lane * 16u + h], 0.0f)
                        * wrow[head] * scale;
        }
        warp_row_partial[warp * 16u + lane] = warp_total;
    }
    __syncthreads();

    if (warp == 0u && lane < 16u) {
        const uint32_t comp = tile_c + lane;
        if (comp < n_comp) {
            float out = 0.0f;
            #pragma unroll
            for (uint32_t w = 0; w < 4u; w++)
                out += warp_row_partial[w * 16u + lane];
            if (causal) {
                const uint32_t qpos = positions ? (uint32_t)positions[t] : pos0;
                uint32_t visible = ratio ? (qpos + 1u) / ratio : n_comp;
                if (positions && ratio && visible > qpos / ratio) visible = qpos / ratio;
                if (comp >= visible) out = -INFINITY;
            }
            scores[(uint64_t)t * n_comp + comp] = out;
        }
    }
}

// ---------------------------------------------------------------------------
// V5C32 -- four-warp V5C with a 32-row CTA tile.
//
// Each warp retains its 16-head tile and its single Q/scratch slice, then
// processes the two 16-row A subtiles sequentially.  This keeps every row's
// block and head fold order identical to V5C without duplicating Q shared memory.
// ---------------------------------------------------------------------------

__global__ static void indexer_scores_multiseq_v5c32_kernel(
        float *scores, const float *q, const float *weights,
        const float *index_comp, uint32_t n_comp, uint32_t n_tokens,
        uint32_t pos0, uint32_t ratio, uint32_t comp_cap, float scale,
        int causal, const int32_t * __restrict__ positions,
        const int32_t * __restrict__ seq_id,
        const unsigned char * __restrict__ index_fp4,
        const float * __restrict__ index_scale,
        const float * __restrict__ q_block_scale,
        const float * __restrict__ k_block_scale) {
    const uint32_t tile_c = blockIdx.x * 32u;
    const uint32_t t = blockIdx.y;
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t warp = tid >> 5u;
    if (t >= n_tokens || tid >= 128u) return;

    const bool use_fp4 = (index_fp4 != NULL) && (index_scale != NULL);
    const uint32_t seq_base = seq_id ? (uint32_t)seq_id[t] * comp_cap : 0u;
    const uint32_t head0 = warp * 16u;
    const float *wrow = weights + (uint64_t)t * 64u;

    __shared__ __half a_sh[32 * 128];
    __shared__ __half b_sh[4 * 16 * 128];
    __shared__ float c_sh[4 * 16 * 16];
    __shared__ float k_scale_block[32];
    __shared__ float warp_row_partial[4 * 32];

    __half *warp_b_sh = b_sh + warp * 16u * 128u;
    float *warp_c_sh = c_sh + warp * 16u * 16u;
    for (uint32_t i = lane; i < 16u * 128u; i += 32u) {
        const uint32_t h = i >> 7u;
        const uint32_t d = i & 127u;
        const uint32_t head = head0 + h;
        float v = q[((uint64_t)t * 64u + head) * 128u + d];
        const float qs = q_block_scale[
            ((uint64_t)t * 64u + head) * 4u + (d >> 5u)];
        warp_b_sh[i] = __float2half(v / qs);
    }
    __syncthreads();

    float head_total[16];
    #pragma unroll
    for (uint32_t slot = 0; slot < 16u; slot++) head_total[slot] = 0.0f;

    #pragma unroll
    for (uint32_t block = 0; block < 4u; block++) {
        #pragma unroll
        for (uint32_t r = warp; r < 32u; r += 4u) {
            const uint32_t comp = tile_c + r;
            const uint32_t d = block * 32u + lane;
            float v = 0.0f;
            if (comp < n_comp) {
                v = use_fp4
                    ? indexer_fp4_level_read(index_fp4, seq_base + comp, d)
                    : index_comp[(uint64_t)(seq_base + comp) * 128u + d];
            }
            if (use_fp4) {
                if (lane == 0u) {
                    k_scale_block[r] = comp < n_comp
                        ? index_scale[(uint64_t)(seq_base + comp) * 4u + block]
                        : 1.0f;
                }
            } else {
                if (lane == 0u) {
                    k_scale_block[r] = comp < n_comp
                        ? k_block_scale[(uint64_t)(seq_base + comp) * 4u + block]
                        : 1.0f;
                }
            }
            __syncwarp();
            const float level = use_fp4 ? v : v / k_scale_block[r];
            a_sh[r * 128u + d] = __float2half(level);
        }
        __syncthreads();

        #pragma unroll
        for (uint32_t row_tile = 0; row_tile < 2u; row_tile++) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_frag;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
            wmma::fill_fragment(c_frag, 0.0f);
            #pragma unroll
            for (uint32_t k = 0; k < 32u; k += 16u) {
                const uint32_t k0 = block * 32u + k;
                wmma::load_matrix_sync(
                    a_frag, a_sh + row_tile * 16u * 128u + k0, 128);
                wmma::load_matrix_sync(b_frag, warp_b_sh + k0, 128);
                wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            }
            wmma::store_matrix_sync(warp_c_sh, c_frag, 16, wmma::mem_row_major);
            __syncwarp();

            #pragma unroll
            for (uint32_t slot = 0; slot < 8u; slot++) {
                const uint32_t i = lane + slot * 32u;
                const uint32_t r = row_tile * 16u + (i >> 4u);
                const uint32_t h = i & 15u;
                const uint32_t comp = tile_c + r;
                if (comp < n_comp) {
                    float chunk = warp_c_sh[i];
                    const float qs = q_block_scale[
                        ((uint64_t)t * 64u + head0 + h) * 4u + block];
                    chunk *= k_scale_block[r] * qs;
                    head_total[row_tile * 8u + slot] += chunk;
                }
            }
            __syncthreads();
        }
    }

    #pragma unroll
    for (uint32_t row_tile = 0; row_tile < 2u; row_tile++) {
        #pragma unroll
        for (uint32_t slot = 0; slot < 8u; slot++)
            warp_c_sh[lane + slot * 32u] =
                head_total[row_tile * 8u + slot];
        __syncthreads();

        if (lane < 16u) {
            float warp_total = 0.0f;
            #pragma unroll
            for (uint32_t h = 0; h < 16u; h++) {
                const uint32_t head = head0 + h;
                warp_total += fmaxf(warp_c_sh[lane * 16u + h], 0.0f)
                            * wrow[head] * scale;
            }
            warp_row_partial[
                warp * 32u + row_tile * 16u + lane] = warp_total;
        }
        __syncthreads();
    }

    if (warp == 0u && lane < 32u) {
        const uint32_t comp = tile_c + lane;
        if (comp < n_comp) {
            float out = 0.0f;
            #pragma unroll
            for (uint32_t w = 0; w < 4u; w++)
                out += warp_row_partial[w * 32u + lane];
            if (causal) {
                const uint32_t qpos = positions ? (uint32_t)positions[t] : pos0;
                uint32_t visible = ratio ? (qpos + 1u) / ratio : n_comp;
                if (positions && ratio && visible > qpos / ratio) visible = qpos / ratio;
                if (comp >= visible) out = -INFINITY;
            }
            scores[(uint64_t)t * n_comp + comp] = out;
        }
    }
}

// ---------------------------------------------------------------------------
// V5D -- scale-block-restaged V5C32 with an occupancy-oriented footprint.
//
// The single CTA-wide A tile remains shared and reused by all four warps, but
// A and each warp-private B tile retain only the current 32-dimension scale
// block.  Restaging before each block shrinks A from 32x128 to 32x32 halves and
// B from 4x16x128 to 4x16x32 halves; with unchanged C/scale/partial storage the
// static footprint is 10880 bytes.  The same 16x16x16 fragments receive the
// same half values in block 0..3 and row-tile 0..1 order, and both folds remain
// unchanged.  Six 128-thread CTAs require <=85 registers/thread on a 64K-reg
// SM, so launch_bounds asks the compiler for that ceiling; the attribute ledger
// in main reports the real result because this hint is not a guarantee.
// ---------------------------------------------------------------------------

/* bounds sweep on .15 (2026-07-18): (128,6) 200.1 µs / (128,8) 187.8 µs
 * regs 64, idempotence holds / (128,12) 200.3 µs regs 40, spill cliff.
 * Interior optimum = 8 blocks; occupancy stopped paying linearly past
 * ~50% (pipes ~35%) — the per-CTA stage->MMA->scale-fold dependency
 * chain is the residual floor. */
__global__ static void __launch_bounds__(128, 8) indexer_scores_multiseq_v5d_kernel(
        float *scores, const float *q, const float *weights,
        const float *index_comp, uint32_t n_comp, uint32_t n_tokens,
        uint32_t pos0, uint32_t ratio, uint32_t comp_cap, float scale,
        int causal, const int32_t * __restrict__ positions,
        const int32_t * __restrict__ seq_id,
        const unsigned char * __restrict__ index_fp4,
        const float * __restrict__ index_scale,
        const float * __restrict__ q_block_scale,
        const float * __restrict__ k_block_scale) {
    const uint32_t tile_c = blockIdx.x * 32u;
    const uint32_t t = blockIdx.y;
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t warp = tid >> 5u;
    if (t >= n_tokens || tid >= 128u) return;

    const bool use_fp4 = (index_fp4 != NULL) && (index_scale != NULL);
    const uint32_t seq_base = seq_id ? (uint32_t)seq_id[t] * comp_cap : 0u;
    const uint32_t head0 = warp * 16u;
    const float *wrow = weights + (uint64_t)t * 64u;

    __shared__ __half a_sh[32 * 32];
    __shared__ __half b_sh[4 * 16 * 32];
    __shared__ float c_sh[4 * 16 * 16];
    __shared__ float k_scale_block[32];
    __shared__ float warp_row_partial[4 * 32];

    __half *warp_b_sh = b_sh + warp * 16u * 32u;
    float *warp_c_sh = c_sh + warp * 16u * 16u;

    float head_total[16];
    #pragma unroll
    for (uint32_t slot = 0; slot < 16u; slot++) head_total[slot] = 0.0f;

    #pragma unroll
    for (uint32_t block = 0; block < 4u; block++) {
        for (uint32_t i = lane; i < 16u * 32u; i += 32u) {
            const uint32_t h = i >> 5u;
            const uint32_t block_d = i & 31u;
            const uint32_t d = block * 32u + block_d;
            const uint32_t head = head0 + h;
            float v = q[((uint64_t)t * 64u + head) * 128u + d];
            const float qs = q_block_scale[
                ((uint64_t)t * 64u + head) * 4u + (d >> 5u)];
            warp_b_sh[i] = __float2half(v / qs);
        }

        #pragma unroll
        for (uint32_t r = warp; r < 32u; r += 4u) {
            const uint32_t comp = tile_c + r;
            const uint32_t d = block * 32u + lane;
            float v = 0.0f;
            if (comp < n_comp) {
                v = use_fp4
                    ? indexer_fp4_level_read(index_fp4, seq_base + comp, d)
                    : index_comp[(uint64_t)(seq_base + comp) * 128u + d];
            }
            if (use_fp4) {
                if (lane == 0u) {
                    k_scale_block[r] = comp < n_comp
                        ? index_scale[(uint64_t)(seq_base + comp) * 4u + block]
                        : 1.0f;
                }
            } else {
                if (lane == 0u) {
                    k_scale_block[r] = comp < n_comp
                        ? k_block_scale[(uint64_t)(seq_base + comp) * 4u + block]
                        : 1.0f;
                }
            }
            __syncwarp();
            const float level = use_fp4 ? v : v / k_scale_block[r];
            a_sh[r * 32u + lane] = __float2half(level);
        }
        __syncthreads();

        #pragma unroll
        for (uint32_t row_tile = 0; row_tile < 2u; row_tile++) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_frag;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
            wmma::fill_fragment(c_frag, 0.0f);
            #pragma unroll
            for (uint32_t k = 0; k < 32u; k += 16u) {
                wmma::load_matrix_sync(
                    a_frag, a_sh + row_tile * 16u * 32u + k, 32);
                wmma::load_matrix_sync(b_frag, warp_b_sh + k, 32);
                wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            }
            wmma::store_matrix_sync(warp_c_sh, c_frag, 16, wmma::mem_row_major);
            __syncwarp();

            #pragma unroll
            for (uint32_t slot = 0; slot < 8u; slot++) {
                const uint32_t i = lane + slot * 32u;
                const uint32_t r = row_tile * 16u + (i >> 4u);
                const uint32_t h = i & 15u;
                const uint32_t comp = tile_c + r;
                if (comp < n_comp) {
                    float chunk = warp_c_sh[i];
                    const float qs = q_block_scale[
                        ((uint64_t)t * 64u + head0 + h) * 4u + block];
                    chunk *= k_scale_block[r] * qs;
                    head_total[row_tile * 8u + slot] += chunk;
                }
            }
            __syncthreads();
        }
    }

    #pragma unroll
    for (uint32_t row_tile = 0; row_tile < 2u; row_tile++) {
        #pragma unroll
        for (uint32_t slot = 0; slot < 8u; slot++)
            warp_c_sh[lane + slot * 32u] =
                head_total[row_tile * 8u + slot];
        __syncthreads();

        if (lane < 16u) {
            float warp_total = 0.0f;
            #pragma unroll
            for (uint32_t h = 0; h < 16u; h++) {
                const uint32_t head = head0 + h;
                warp_total += fmaxf(warp_c_sh[lane * 16u + h], 0.0f)
                            * wrow[head] * scale;
            }
            warp_row_partial[
                warp * 32u + row_tile * 16u + lane] = warp_total;
        }
        __syncthreads();
    }

    if (warp == 0u && lane < 32u) {
        const uint32_t comp = tile_c + lane;
        if (comp < n_comp) {
            float out = 0.0f;
            #pragma unroll
            for (uint32_t w = 0; w < 4u; w++)
                out += warp_row_partial[w * 32u + lane];
            if (causal) {
                const uint32_t qpos = positions ? (uint32_t)positions[t] : pos0;
                uint32_t visible = ratio ? (qpos + 1u) / ratio : n_comp;
                if (positions && ratio && visible > qpos / ratio) visible = qpos / ratio;
                if (comp >= visible) out = -INFINITY;
            }
            scores[(uint64_t)t * n_comp + comp] = out;
        }
    }
}

// ---------------------------------------------------------------------------
// Proto scaffolding.
// ---------------------------------------------------------------------------

// Copied verbatim from proto_kv_reencode_idem.cu: deterministic xorshift hash
// of (row,lane,salt), so input magnitudes are reproducible across runs.
__device__ static float hash_unit(uint32_t row, uint32_t lane, uint32_t salt) {
    uint32_t h = row * 2654435761u ^ (lane + 1u) * 2246822519u ^ salt * 3266489917u;
    h ^= h >> 15; h *= 2246822519u; h ^= h >> 13; h *= 3266489917u; h ^= h >> 16;
    return ((float)(h & 0xFFFFFFu) / 8388608.0f) - 1.0f;   // [-1,1)
}

__global__ static void fill_index_kernel(float *x, uint32_t n_rows, uint32_t salt) {
    const uint32_t row = blockIdx.x;
    const uint32_t lane = threadIdx.x;
    if (row >= n_rows || lane >= 128u) return;
    const float u = hash_unit(row, lane, salt);
    float v;
    switch (row & 7u) {
    case 0: v = u; break;
    case 1: v = u * 1.0e-5f; break;
    case 2: v = u * 64.0f; break;
    case 3: v = u * ldexpf(1.0f, (int)(row % 17u) - 8); break;
    case 4: v = fabsf(u) + 0.25f; break;       // all-positive row
    case 5: v = -(fabsf(u) + 0.25f); break;    // all-negative row
    case 6: v = (lane & 1u) ? u * 8.0f : -u * 8.0f; break;
    default: v = (lane % 5u == 0u) ? 0.0f : u * 1.0e-7f; break;
    }
    x[(uint64_t)row * 128u + lane] = v;
}

__global__ static void fill_q_weights_kernel(float *q, float *weights,
                                              uint32_t n_tokens, uint32_t salt) {
    const uint32_t t = blockIdx.x;
    const uint32_t lane = threadIdx.x;
    if (t >= n_tokens || lane >= 128u) return;
    for (uint32_t h = 0; h < 64u; h++) {
        const uint32_t key = t * 64u + h;
        const float u = hash_unit(key, lane, salt);
        float v;
        switch (h & 7u) {
        case 0: v = u; break;
        case 1: v = u * 1.0e-5f; break;
        case 2: v = u * 32.0f; break;
        case 3: v = u * ldexpf(1.0f, (int)(h % 13u) - 6); break;
        case 4: v = fabsf(u) + 0.125f; break;
        case 5: v = -(fabsf(u) + 0.125f); break;
        case 6: v = (lane & 1u) ? u * 4.0f : -u * 4.0f; break;
        default: v = (lane % 7u == 0u) ? 0.0f : u * 1.0e-7f; break;
        }
        q[(uint64_t)t * 64u * 128u + (uint64_t)h * 128u + lane] = v;
        if (lane == 0) {
            const float wu = hash_unit(t, h, salt ^ 0xa511e9b3u);
            float w;
            switch (h % 6u) {
            case 0: w = fabsf(wu) + 0.05f; break;
            case 1: w = -(fabsf(wu) + 0.05f); break;
            case 2: w = wu * 1.0e-4f; break;
            case 3: w = wu * 8.0f; break;
            case 4: w = wu; break;
            default: w = (h & 1u) ? -0.5f : 0.5f; break;
            }
            weights[(uint64_t)t * 64u + h] = w;
        }
    }
}

#define HALF_SCALE_EXP_MIN (-64)
#define HALF_SCALE_EXP_MAX 64
#define HALF_SCALE_EXP_BUCKETS (HALF_SCALE_EXP_MAX - HALF_SCALE_EXP_MIN + 1)

__device__ static uint32_t half_scale_exp_bucket(float scale) {
    int exponent = ilogbf(scale);
    if (exponent < HALF_SCALE_EXP_MIN) exponent = HALF_SCALE_EXP_MIN;
    if (exponent > HALF_SCALE_EXP_MAX) exponent = HALF_SCALE_EXP_MAX;
    return (uint32_t)(exponent - HALF_SCALE_EXP_MIN);
}

__global__ static void check_fp4_half_roundtrip_kernel(
        const unsigned char * __restrict__ codes,
        const float * __restrict__ scales,
        uint32_t n_rows,
        unsigned long long *violations,
        unsigned long long *scale_exp_hist) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t n = (uint64_t)n_rows * 128u;
    if (i >= n) return;
    const uint32_t row = (uint32_t)(i >> 7u);
    const uint32_t d = (uint32_t)(i & 127u);
    const float v = indexer_fp4_read(codes, scales, row, d);
    const float roundtrip = __half2float(__float2half(v));
    if (__float_as_uint(roundtrip) != __float_as_uint(v)) {
        atomicAdd(violations, 1ull);
        const float block_scale = scales[(uint64_t)row * 4u + (d >> 5u)];
        atomicAdd(scale_exp_hist + half_scale_exp_bucket(block_scale), 1ull);
    }
}

__global__ static void check_grid_half_roundtrip_kernel(
        const float * __restrict__ values,
        const float * __restrict__ scales,
        uint32_t n_rows,
        unsigned long long *violations,
        unsigned long long *scale_exp_hist) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t n = (uint64_t)n_rows * 128u;
    if (i >= n) return;
    const uint32_t row = (uint32_t)(i >> 7u);
    const uint32_t d = (uint32_t)(i & 127u);
    const float v = values[i];
    const float roundtrip = __half2float(__float2half(v));
    if (__float_as_uint(roundtrip) != __float_as_uint(v)) {
        atomicAdd(violations, 1ull);
        const float block_scale = scales[(uint64_t)row * 4u + (d >> 5u)];
        atomicAdd(scale_exp_hist + half_scale_exp_bucket(block_scale), 1ull);
    }
}

__global__ static void check_grid_unscaled_half_roundtrip_kernel(
        const float * __restrict__ values,
        const float * __restrict__ scales,
        uint32_t n_rows,
        unsigned long long *violations,
        unsigned long long *scale_exp_hist) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t n = (uint64_t)n_rows * 128u;
    if (i >= n) return;
    const uint32_t row = (uint32_t)(i >> 7u);
    const uint32_t d = (uint32_t)(i & 127u);
    const float block_scale = scales[(uint64_t)row * 4u + (d >> 5u)];
    const float quotient = values[i] / block_scale;
    const float roundtrip = __half2float(__float2half(quotient));
    if (__float_as_uint(roundtrip) != __float_as_uint(quotient)) {
        atomicAdd(violations, 1ull);
        atomicAdd(scale_exp_hist + half_scale_exp_bucket(block_scale), 1ull);
    }
}

__global__ static void check_grid_rederived_unscaled_half_roundtrip_kernel(
        const float * __restrict__ values,
        uint32_t n_rows,
        unsigned long long *violations,
        unsigned long long *scale_exp_hist) {
    const uint32_t row_block = blockIdx.x;
    const uint32_t lane = threadIdx.x;
    const uint32_t row = row_block >> 2u;
    const uint32_t fp4_block = row_block & 3u;
    if (row >= n_rows || lane >= 32u) return;

    const uint32_t d = fp4_block * 32u + lane;
    const float v = values[(uint64_t)row * 128u + d];
    const float block_max_abs = warp_max_f32(fabsf(v));
    float block_scale = 0.0f;
    if (lane == 0u) {
        const float amax = fmaxf(block_max_abs, 7.052966104933725e-38f);
        block_scale = dsv4_pow2_ceil_scale(amax / 6.0f);
    }
    block_scale = __shfl_sync(0xffffffffu, block_scale, 0);
    const float quotient = v / block_scale;
    const float roundtrip = __half2float(__float2half(quotient));
    if (__float_as_uint(roundtrip) != __float_as_uint(quotient)) {
        atomicAdd(violations, 1ull);
        atomicAdd(scale_exp_hist + half_scale_exp_bucket(block_scale), 1ull);
    }
}

static void print_half_preflight(const char *label, unsigned long long violations,
                                 uint64_t n,
                                 const unsigned long long scale_exp_hist[HALF_SCALE_EXP_BUCKETS]) {
    printf("  fp16 preflight %-2s: violations=%llu / %llu  result=%s\n",
           label, violations, (unsigned long long)n,
           violations == 0 ? "PASS" : "FINDING");
    if (violations == 0) return;
    printf("    offending block-scale exponent histogram:");
    for (int bucket = 0; bucket < HALF_SCALE_EXP_BUCKETS; bucket++) {
        if (scale_exp_hist[bucket] == 0) continue;
        const int exponent = bucket + HALF_SCALE_EXP_MIN;
        if (bucket == 0)
            printf(" e<=%d:%llu", exponent, scale_exp_hist[bucket]);
        else if (bucket == HALF_SCALE_EXP_BUCKETS - 1)
            printf(" e>=%d:%llu", exponent, scale_exp_hist[bucket]);
        else
            printf(" e=%d:%llu", exponent, scale_exp_hist[bucket]);
    }
    printf("\n");
}

static unsigned long long report_fp4_half_preflight(
        const char *label, const unsigned char *codes, const float *scales,
        uint32_t n_rows) {
    unsigned long long *d_violations, *d_hist;
    CHECK(cudaMalloc(&d_violations, sizeof(*d_violations)));
    CHECK(cudaMalloc(&d_hist, HALF_SCALE_EXP_BUCKETS * sizeof(*d_hist)));
    CHECK(cudaMemset(d_violations, 0, sizeof(*d_violations)));
    CHECK(cudaMemset(d_hist, 0, HALF_SCALE_EXP_BUCKETS * sizeof(*d_hist)));
    const uint64_t n = (uint64_t)n_rows * 128u;
    const uint64_t blocks = (n + 255u) / 256u;
    check_fp4_half_roundtrip_kernel<<<(uint32_t)blocks, 256>>>(
        codes, scales, n_rows, d_violations, d_hist);
    CHECK(cudaDeviceSynchronize());
    unsigned long long violations;
    unsigned long long hist[HALF_SCALE_EXP_BUCKETS];
    CHECK(cudaMemcpy(&violations, d_violations, sizeof(violations), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(hist, d_hist, sizeof(hist), cudaMemcpyDeviceToHost));
    print_half_preflight(label, violations, n, hist);
    cudaFree(d_violations); cudaFree(d_hist);
    return violations;
}

static unsigned long long report_grid_half_preflight(
        const char *label, const float *values, const float *scales,
        uint32_t n_rows) {
    unsigned long long *d_violations, *d_hist;
    CHECK(cudaMalloc(&d_violations, sizeof(*d_violations)));
    CHECK(cudaMalloc(&d_hist, HALF_SCALE_EXP_BUCKETS * sizeof(*d_hist)));
    CHECK(cudaMemset(d_violations, 0, sizeof(*d_violations)));
    CHECK(cudaMemset(d_hist, 0, HALF_SCALE_EXP_BUCKETS * sizeof(*d_hist)));
    const uint64_t n = (uint64_t)n_rows * 128u;
    const uint64_t blocks = (n + 255u) / 256u;
    check_grid_half_roundtrip_kernel<<<(uint32_t)blocks, 256>>>(
        values, scales, n_rows, d_violations, d_hist);
    CHECK(cudaDeviceSynchronize());
    unsigned long long violations;
    unsigned long long hist[HALF_SCALE_EXP_BUCKETS];
    CHECK(cudaMemcpy(&violations, d_violations, sizeof(violations), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(hist, d_hist, sizeof(hist), cudaMemcpyDeviceToHost));
    print_half_preflight(label, violations, n, hist);
    cudaFree(d_violations); cudaFree(d_hist);
    return violations;
}

static unsigned long long report_grid_unscaled_half_preflight(
        const char *label, const float *values, const float *scales,
        uint32_t n_rows) {
    unsigned long long *d_violations, *d_hist;
    CHECK(cudaMalloc(&d_violations, sizeof(*d_violations)));
    CHECK(cudaMalloc(&d_hist, HALF_SCALE_EXP_BUCKETS * sizeof(*d_hist)));
    CHECK(cudaMemset(d_violations, 0, sizeof(*d_violations)));
    CHECK(cudaMemset(d_hist, 0, HALF_SCALE_EXP_BUCKETS * sizeof(*d_hist)));
    const uint64_t n = (uint64_t)n_rows * 128u;
    const uint64_t blocks = (n + 255u) / 256u;
    check_grid_unscaled_half_roundtrip_kernel<<<(uint32_t)blocks, 256>>>(
        values, scales, n_rows, d_violations, d_hist);
    CHECK(cudaDeviceSynchronize());
    unsigned long long violations;
    unsigned long long hist[HALF_SCALE_EXP_BUCKETS];
    CHECK(cudaMemcpy(&violations, d_violations, sizeof(violations), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(hist, d_hist, sizeof(hist), cudaMemcpyDeviceToHost));
    print_half_preflight(label, violations, n, hist);
    cudaFree(d_violations); cudaFree(d_hist);
    return violations;
}

static unsigned long long report_grid_rederived_unscaled_half_preflight(
        const char *label, const float *values, uint32_t n_rows) {
    unsigned long long *d_violations, *d_hist;
    CHECK(cudaMalloc(&d_violations, sizeof(*d_violations)));
    CHECK(cudaMalloc(&d_hist, HALF_SCALE_EXP_BUCKETS * sizeof(*d_hist)));
    CHECK(cudaMemset(d_violations, 0, sizeof(*d_violations)));
    CHECK(cudaMemset(d_hist, 0, HALF_SCALE_EXP_BUCKETS * sizeof(*d_hist)));
    const uint64_t n = (uint64_t)n_rows * 128u;
    check_grid_rederived_unscaled_half_roundtrip_kernel<<<n_rows * 4u, 32>>>(
        values, n_rows, d_violations, d_hist);
    CHECK(cudaDeviceSynchronize());
    unsigned long long violations;
    unsigned long long hist[HALF_SCALE_EXP_BUCKETS];
    CHECK(cudaMemcpy(&violations, d_violations, sizeof(violations), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(hist, d_hist, sizeof(hist), cudaMemcpyDeviceToHost));
    print_half_preflight(label, violations, n, hist);
    cudaFree(d_violations); cudaFree(d_hist);
    return violations;
}

__device__ __forceinline__ static bool topk_better(
        float av, uint32_t ai, float bv, uint32_t bi) {
    return av > bv || (av == bv && ai < bi);
}

__global__ static void indexer_topk_kernel(
        const float * __restrict__ scores,
        uint32_t * __restrict__ topk_indices,
        uint32_t n_comp,
        uint32_t n_tokens,
        uint32_t top_k) {
    const uint32_t t = blockIdx.x;
    if (t >= n_tokens || threadIdx.x != 0u) return;
    const float *row_scores = scores + (uint64_t)t * n_comp;
    uint32_t *row_topk = topk_indices + (uint64_t)t * top_k;
    for (uint32_t c = 0; c < n_comp; c++) {
        const float candidate = row_scores[c];
        for (uint32_t k = 0; k < top_k; k++) {
            const uint32_t incumbent = k < c ? row_topk[k] : 0u;
            if (k >= c || topk_better(
                    candidate, c, row_scores[incumbent], incumbent)) {
                for (uint32_t shift = top_k - 1u; shift > k; shift--)
                    row_topk[shift] = row_topk[shift - 1u];
                row_topk[k] = c;
                break;
            }
        }
    }
}

__global__ static void mark_topk_mask_kernel(
        const uint32_t * __restrict__ topk_indices,
        uint32_t top_k,
        unsigned char * __restrict__ mask) {
    const uint32_t k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < top_k) mask[topk_indices[k]] = 1u;
}

__global__ static void count_topk_flips_kernel(
        const uint32_t * __restrict__ baseline_topk,
        uint32_t top_k,
        const unsigned char * __restrict__ variant_mask,
        uint32_t *flip_count) {
    const uint32_t k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < top_k && variant_mask[baseline_topk[k]] == 0u)
        atomicAdd(flip_count, 1u);
}

static void launch_topk_selection(
        const float *scores, uint32_t *topk_indices,
        uint32_t n_comp, uint32_t n_tokens, uint32_t top_k) {
    indexer_topk_kernel<<<n_tokens, 1>>>(
        scores, topk_indices, n_comp, n_tokens, top_k);
    CHECK(cudaGetLastError());
}

static uint32_t report_topk_flips(
        const uint32_t *baseline_topk, const uint32_t *variant_topk,
        uint32_t top_k, unsigned char *d_mask, uint32_t n_comp,
        uint32_t *d_flip_count) {
    CHECK(cudaMemset(d_mask, 0, (uint64_t)n_comp * sizeof(*d_mask)));
    CHECK(cudaMemset(d_flip_count, 0, sizeof(*d_flip_count)));
    const uint32_t blocks = (top_k + 255u) / 256u;
    mark_topk_mask_kernel<<<blocks, 256>>>(variant_topk, top_k, d_mask);
    count_topk_flips_kernel<<<blocks, 256>>>(
        baseline_topk, top_k, d_mask, d_flip_count);
    CHECK(cudaDeviceSynchronize());
    uint32_t flips;
    CHECK(cudaMemcpy(&flips, d_flip_count, sizeof(flips), cudaMemcpyDeviceToHost));
    const double percent = top_k ? 100.0 * (double)flips / (double)top_k : 0.0;
    printf("    topk flips: %u / %u (%.1f%%)\n", flips, top_k, percent);
    return flips;
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

struct ulp_report {
    unsigned long long mismatches;
    unsigned long long one_ulp;
    unsigned long long two_ulp;
    unsigned long long over_two_ulp;
    unsigned long long variant_greater;
    unsigned long long variant_less;
    uint32_t max_ulp;
};

__device__ static uint32_t ordered_f32_bits(uint32_t bits) {
    return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

__global__ static void compare_f32_ulp_kernel(const float *baseline, const float *variant,
                                              uint64_t n, ulp_report *report) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const uint32_t baseline_bits = __float_as_uint(baseline[i]);
    const uint32_t variant_bits = __float_as_uint(variant[i]);
    if (baseline_bits == variant_bits) return;

    const uint32_t baseline_ordered = ordered_f32_bits(baseline_bits);
    const uint32_t variant_ordered = ordered_f32_bits(variant_bits);
    const uint32_t ulp = baseline_ordered > variant_ordered
        ? baseline_ordered - variant_ordered
        : variant_ordered - baseline_ordered;
    atomicAdd(&report->mismatches, 1ull);
    if (ulp == 1u)
        atomicAdd(&report->one_ulp, 1ull);
    else if (ulp == 2u)
        atomicAdd(&report->two_ulp, 1ull);
    else
        atomicAdd(&report->over_two_ulp, 1ull);
    atomicMax(&report->max_ulp, ulp);
    if (variant[i] > baseline[i])
        atomicAdd(&report->variant_greater, 1ull);
    else if (variant[i] < baseline[i])
        atomicAdd(&report->variant_less, 1ull);
}

static ulp_report report_f32_ulp(const char *tag, const float *baseline,
                                 const float *variant, uint64_t n) {
    ulp_report *d_report;
    CHECK(cudaMalloc(&d_report, sizeof(*d_report)));
    CHECK(cudaMemset(d_report, 0, sizeof(*d_report)));
    const uint64_t blocks = (n + 255u) / 256u;
    compare_f32_ulp_kernel<<<(uint32_t)blocks, 256>>>(baseline, variant, n, d_report);
    CHECK(cudaDeviceSynchronize());
    ulp_report report;
    CHECK(cudaMemcpy(&report, d_report, sizeof(report), cudaMemcpyDeviceToHost));
    cudaFree(d_report);

    printf("  %-28s mismatches=%llu / %llu  1ulp=%llu  2ulp=%llu  >2ulp=%llu  max=%u\n",
           tag, report.mismatches, (unsigned long long)n,
           report.one_ulp, report.two_ulp, report.over_two_ulp, report.max_ulp);
    const double greater_fraction = report.mismatches
        ? (double)report.variant_greater / (double)report.mismatches : 0.0;
    const double less_fraction = report.mismatches
        ? (double)report.variant_less / (double)report.mismatches : 0.0;
    printf("    bias: variant>baseline=%.1f%%  variant<baseline=%.1f%%",
           greater_fraction * 100.0, less_fraction * 100.0);
    if (report.mismatches != 0 &&
        (greater_fraction >= 0.9 || less_fraction >= 0.9))
        printf("  possible systematic bias");
    printf("\n");
    return report;
}

enum variant_id {
    BASELINE = 0, V1 = 1, V2 = 2, V2B = 3, V3 = 4, V4 = 5, V5 = 6,
    V5B = 7, V5C = 8, V5C32 = 9, V5D = 10, N_VARIANTS = 11
};
static const char *variant_name[N_VARIANTS] = {
    "BASELINE", "V1", "V2", "V2B", "V3", "V4", "V5", "V5B", "V5C", "V5C32",
    "V5D"
};

// All launch geometry is a pure function of n_comp/n_tokens and the variant's
// fixed tile size.  There is no data-dependent host decision, preserving the
// capture-safe launch contract required by a future engine integration.
static void launch_score(
        variant_id variant, float *scores, const float *q, const float *weights,
        const float *index_comp, uint32_t n_comp, uint32_t n_tokens,
        uint32_t pos0, uint32_t ratio, uint32_t comp_cap, float scale,
        int causal, const int32_t *positions, const int32_t *seq_id,
        const unsigned char *index_fp4, const float *index_scale,
        const float *q_block_scale, const float *k_block_scale) {
    if (variant == BASELINE) {
        indexer_scores_multiseq_kernel<<<dim3(n_comp, n_tokens, 1), 128>>>(
            scores, q, weights, index_comp, n_comp, n_tokens, pos0, ratio,
            comp_cap, scale, causal, positions, seq_id, index_fp4, index_scale);
    } else if (variant == V1) {
        indexer_scores_multiseq_v1_kernel<<<dim3((n_comp + 3u) / 4u, n_tokens, 1), 128>>>(
            scores, q, weights, index_comp, n_comp, n_tokens, pos0, ratio,
            comp_cap, scale, causal, positions, seq_id, index_fp4, index_scale);
    } else if (variant == V2) {
        indexer_scores_multiseq_v2_kernel<<<dim3((n_comp + V2_ROWS_PER_BLOCK - 1u) /
                                                  V2_ROWS_PER_BLOCK, n_tokens, 1), 256>>>(
            scores, q, weights, index_comp, n_comp, n_tokens, pos0, ratio,
            comp_cap, scale, causal, positions, seq_id, index_fp4, index_scale);
    } else if (variant == V2B) {
        indexer_scores_multiseq_v2b_kernel<<<dim3((n_comp + V2_ROWS_PER_BLOCK - 1u) /
                                                   V2_ROWS_PER_BLOCK, n_tokens, 1), 256>>>(
            scores, q, weights, index_comp, n_comp, n_tokens, pos0, ratio,
            comp_cap, scale, causal, positions, seq_id, index_fp4, index_scale);
    } else if (variant == V3) {
        indexer_scores_multiseq_v3_kernel<<<dim3(n_comp, n_tokens, 1), 128>>>(
            scores, q, weights, index_comp, n_comp, n_tokens, pos0, ratio,
            comp_cap, scale, causal, positions, seq_id, index_fp4, index_scale);
    } else if (variant == V4) {
        indexer_scores_multiseq_v4_kernel<<<dim3((n_comp + 127u) / 128u,
                                                  (n_tokens + 15u) / 16u, 1), 256>>>(
            scores, q, weights, index_comp, n_comp, n_tokens, pos0, ratio,
            comp_cap, scale, causal, positions, seq_id, index_fp4, index_scale);
    } else if (variant == V5) {
        indexer_scores_multiseq_v5_kernel<<<dim3((n_comp + 15u) / 16u,
                                                  n_tokens, 1), 32>>>(
            scores, q, weights, index_comp, n_comp, n_tokens, pos0, ratio,
            comp_cap, scale, causal, positions, seq_id, index_fp4, index_scale,
            q_block_scale);
    } else if (variant == V5B) {
        indexer_scores_multiseq_v5b_kernel<<<dim3((n_comp + 15u) / 16u,
                                                   n_tokens, 1), 128>>>(
            scores, q, weights, index_comp, n_comp, n_tokens, pos0, ratio,
            comp_cap, scale, causal, positions, seq_id, index_fp4, index_scale,
            q_block_scale);
    } else if (variant == V5C) {
        indexer_scores_multiseq_v5c_kernel<<<dim3((n_comp + 15u) / 16u,
                                                   n_tokens, 1), 128>>>(
            scores, q, weights, index_comp, n_comp, n_tokens, pos0, ratio,
            comp_cap, scale, causal, positions, seq_id, index_fp4, index_scale,
            q_block_scale, k_block_scale);
    } else if (variant == V5C32) {
        indexer_scores_multiseq_v5c32_kernel<<<dim3((n_comp + 31u) / 32u,
                                                     n_tokens, 1), 128>>>(
            scores, q, weights, index_comp, n_comp, n_tokens, pos0, ratio,
            comp_cap, scale, causal, positions, seq_id, index_fp4, index_scale,
            q_block_scale, k_block_scale);
    } else if (variant == V5D) {
        indexer_scores_multiseq_v5d_kernel<<<dim3((n_comp + 31u) / 32u,
                                                   n_tokens, 1), 128>>>(
            scores, q, weights, index_comp, n_comp, n_tokens, pos0, ratio,
            comp_cap, scale, causal, positions, seq_id, index_fp4, index_scale,
            q_block_scale, k_block_scale);
    }
    CHECK(cudaGetLastError());
}

static float time_score(
        variant_id variant, float *scores, const float *q, const float *weights,
        const float *index_comp, uint32_t n_comp, uint32_t n_tokens,
        uint32_t pos0, uint32_t ratio, uint32_t comp_cap, float scale,
        int causal, const int32_t *positions, const int32_t *seq_id,
        const unsigned char *index_fp4, const float *index_scale,
        const float *q_block_scale, const float *k_block_scale) {
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    launch_score(variant, scores, q, weights, index_comp, n_comp, n_tokens,
                 pos0, ratio, comp_cap, scale, causal, positions, seq_id,
                 index_fp4, index_scale, q_block_scale, k_block_scale);
    CHECK(cudaDeviceSynchronize());

    float best_ms = 1.0e30f;
    for (int rep = 0; rep < 5; rep++) {
        CHECK(cudaEventRecord(start));
        launch_score(variant, scores, q, weights, index_comp, n_comp, n_tokens,
                     pos0, ratio, comp_cap, scale, causal, positions, seq_id,
                     index_fp4, index_scale, q_block_scale, k_block_scale);
        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        float ms;
        CHECK(cudaEventElapsedTime(&ms, start, stop));
        if (ms < best_ms) best_ms = ms;
    }
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    return best_ms * 1000.0f;
}

struct case_def {
    const char *name;
    int causal;
    uint32_t seq;
    int32_t qpos;
};

static void run_shape(const char *shape_name, uint32_t n_comp, int do_timing,
                      bool exact[N_VARIANTS], int *fail_tally, int *leg_tally,
                      float deep_us[N_VARIANTS][2]) {
    const uint32_t HEAD_DIM = 128u;
    const uint32_t N_HEAD = 64u;
    const uint32_t N_TOKENS = 1u;
    const uint32_t N_BANKS = 3u;
    const uint32_t RATIO = 4u;
    const uint32_t COMP_CAP = n_comp;
    const float SCALE = 0.08838834764831845f;
    const uint32_t total_rows = N_BANKS * COMP_CAP;
    const uint64_t index_n = (uint64_t)total_rows * HEAD_DIM;
    const uint64_t score_n = (uint64_t)N_TOKENS * n_comp;
    const uint32_t top_k_effective = TOPK_K < n_comp ? TOPK_K : n_comp;

    float *d_index, *d_encode, *d_scale, *d_index_f32_scale, *d_q, *d_weights;
    unsigned char *d_codes;
    float *d_scores[N_VARIANTS];
    int32_t *d_positions, *d_seq_id;
    uint32_t *d_topk_baseline, *d_topk_variant, *d_topk_flip_count;
    unsigned char *d_topk_mask;
    CHECK(cudaMalloc(&d_index, index_n * sizeof(float)));
    CHECK(cudaMalloc(&d_encode, index_n * sizeof(float)));
    CHECK(cudaMalloc(&d_codes, (uint64_t)total_rows * 64u));
    CHECK(cudaMalloc(&d_scale, (uint64_t)total_rows * 4u * sizeof(float)));
    CHECK(cudaMalloc(&d_q, (uint64_t)N_TOKENS * N_HEAD * HEAD_DIM * sizeof(float)));
    CHECK(cudaMalloc(&d_weights, (uint64_t)N_TOKENS * N_HEAD * sizeof(float)));
    CHECK(cudaMalloc(&d_positions, N_TOKENS * sizeof(int32_t)));
    CHECK(cudaMalloc(&d_seq_id, N_TOKENS * sizeof(int32_t)));
    CHECK(cudaMalloc(&d_topk_baseline,
                     (uint64_t)N_TOKENS * top_k_effective * sizeof(uint32_t)));
    CHECK(cudaMalloc(&d_topk_variant,
                     (uint64_t)N_TOKENS * top_k_effective * sizeof(uint32_t)));
    CHECK(cudaMalloc(&d_topk_mask, (uint64_t)n_comp * sizeof(*d_topk_mask)));
    CHECK(cudaMalloc(&d_topk_flip_count, sizeof(*d_topk_flip_count)));
    for (int v = 0; v < N_VARIANTS; v++)
        CHECK(cudaMalloc(&d_scores[v], score_n * sizeof(float)));

    fill_index_kernel<<<total_rows, 128>>>(d_index, total_rows, 7u);
    fill_q_weights_kernel<<<N_TOKENS, 128>>>(d_q, d_weights, N_TOKENS, 19u);
    CHECK(cudaGetLastError());

    // Mirror the production F32 tier's QAT-committed values.  This must happen
    // before d_encode snapshots d_index so the dense and packed K legs contain
    // the same on-grid values through their two different representations.
    unsigned char *d_index_scratch_codes;
    float *d_index_scratch_scale;
    CHECK(cudaMalloc(&d_index_scratch_codes, (uint64_t)total_rows * 64u));
    CHECK(cudaMalloc(&d_index_scratch_scale,
                     (uint64_t)total_rows * 4u * sizeof(float)));
    indexer_fp4_encode_rows_kernel<<<total_rows, 128>>>(
        d_index, total_rows, HEAD_DIM,
        d_index_scratch_codes, d_index_scratch_scale);
    CHECK(cudaGetLastError());
    cudaFree(d_index_scratch_codes);
    cudaFree(d_index_scratch_scale);

    // Hoist the dense-F32 K scale derivation out of every score launch.  This
    // reads the final committed d_index values and remains valid for the rest
    // of run_shape because d_index is not modified again.
    CHECK(cudaMalloc(&d_index_f32_scale,
                     (uint64_t)total_rows * 4u * sizeof(float)));
    indexer_block_scale_rows_kernel<<<total_rows, 32>>>(
        d_index, total_rows, d_index_f32_scale);
    CHECK(cudaGetLastError());

    // Mirror decode-time indexer-Q QAT by committing the proto encoder's
    // no-Hadamard FP4 round trip directly into d_q.  Codes/scales are scratch;
    // the committed grid-valued d_q is the input consumed by every variant.
    const uint32_t q_rows = N_TOKENS * N_HEAD;
    unsigned char *d_q_codes;
    float *d_q_scale;
    CHECK(cudaMalloc(&d_q_codes, (uint64_t)q_rows * 64u));
    CHECK(cudaMalloc(&d_q_scale, (uint64_t)q_rows * 4u * sizeof(float)));
    indexer_fp4_encode_rows_kernel<<<q_rows, 128>>>(
        d_q, q_rows, HEAD_DIM, d_q_codes, d_q_scale);
    CHECK(cudaGetLastError());

    // The encoder rewrites x in place.  Encode a device snapshot so d_index
    // remains the committed dense-F32 leg while codes/scales encode precisely
    // those same on-grid source rows for the packed-FP4 leg.
    CHECK(cudaMemcpy(d_encode, d_index, index_n * sizeof(float), cudaMemcpyDeviceToDevice));
    indexer_fp4_encode_rows_kernel<<<total_rows, 128>>>(
        d_encode, total_rows, HEAD_DIM, d_codes, d_scale);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    cudaFree(d_encode);

    printf("\n%s shape: n_comp=%u, n_tokens=%u, head_dim=%u, n_head=%u\n",
           shape_name, n_comp, N_TOKENS, HEAD_DIM, N_HEAD);
    report_fp4_half_preflight("K", d_codes, d_scale, total_rows);
    report_grid_half_preflight("Q", d_q, d_q_scale, q_rows);
    report_grid_unscaled_half_preflight("Q-unscaled", d_q, d_q_scale, q_rows);
    report_grid_rederived_unscaled_half_preflight(
        "K-F32-unscaled", d_index, total_rows);
    cudaFree(d_q_codes);

    case_def cases[4] = {
        {"noncausal-seq0", 0, 0u, 0},
        {"causal-full-seq0", 1, 0u, (int32_t)(RATIO * n_comp)},
        {"causal-tail-seq1", 1, 1u, (int32_t)(RATIO * (n_comp - 1u))},
        {"causal-half-seq2", 1, 2u, (int32_t)(RATIO * (n_comp / 2u))},
    };

    for (int ci = 0; ci < 4; ci++) {
        const int32_t hpos = cases[ci].qpos;
        const int32_t hseq = (int32_t)cases[ci].seq;
        CHECK(cudaMemcpy(d_positions, &hpos, sizeof(hpos), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_seq_id, &hseq, sizeof(hseq), cudaMemcpyHostToDevice));
        for (int storage = 0; storage < 2; storage++) {
            const unsigned char *codes = storage ? d_codes : NULL;
            const float *scales = storage ? d_scale : NULL;
            printf(" leg %s / %s\n", cases[ci].name, storage ? "FP4" : "F32");
            launch_score(BASELINE, d_scores[BASELINE], d_q, d_weights, d_index,
                         n_comp, N_TOKENS, 0u, RATIO, COMP_CAP, SCALE,
                         cases[ci].causal, d_positions, d_seq_id, codes, scales,
                         d_q_scale, d_index_f32_scale);
            launch_topk_selection(d_scores[BASELINE], d_topk_baseline,
                                  n_comp, N_TOKENS, top_k_effective);
            CHECK(cudaDeviceSynchronize());
            for (int v = V1; v < N_VARIANTS; v++) {
                launch_score((variant_id)v, d_scores[v], d_q, d_weights, d_index,
                             n_comp, N_TOKENS, 0u, RATIO, COMP_CAP, SCALE,
                             cases[ci].causal, d_positions, d_seq_id, codes, scales,
                             d_q_scale, d_index_f32_scale);
                launch_topk_selection(d_scores[v], d_topk_variant,
                                      n_comp, N_TOKENS, top_k_effective);
                CHECK(cudaDeviceSynchronize());
                char tag[64];
                snprintf(tag, sizeof(tag), "%s vs BASELINE", variant_name[v]);
                bool leg_pass;
                if (v == V1 || v == V3) {
                    const unsigned long long mismatches = report_f32(
                        tag, d_scores[BASELINE], d_scores[v], score_n, n_comp);
                    leg_pass = mismatches == 0;
                } else {
                    // V2/V2B/V4/V5/V5B/V5C/V5C32/V5D share the >2-ULP histogram gate.
                    const ulp_report report = report_f32_ulp(
                        tag, d_scores[BASELINE], d_scores[v], score_n);
                    leg_pass = report.over_two_ulp == 0;
                }
                if (v == V5C) {
                    const unsigned long long mismatches = report_f32(
                        "V5C vs V5B (idempotence)", d_scores[V5B],
                        d_scores[V5C], score_n, n_comp);
                    leg_pass = leg_pass && mismatches == 0;
                } else if (v == V5C32) {
                    const unsigned long long mismatches = report_f32(
                        "V5C32 vs V5C (idempotence)", d_scores[V5C],
                        d_scores[V5C32], score_n, n_comp);
                    leg_pass = leg_pass && mismatches == 0;
                } else if (v == V5D) {
                    const unsigned long long mismatches = report_f32(
                        "V5D vs V5C32 (idempotence)", d_scores[V5C32],
                        d_scores[V5D], score_n, n_comp);
                    leg_pass = leg_pass && mismatches == 0;
                }
                report_topk_flips(
                    d_topk_baseline, d_topk_variant, top_k_effective,
                    d_topk_mask, n_comp, d_topk_flip_count);
                if (!leg_pass) {
                    exact[v] = false;
                    (*fail_tally)++;
                }
                printf("    %s leg=%s  running-fails=%d\n", variant_name[v],
                       leg_pass ? "PASS" : "FAIL", *fail_tally);
            }
            (*leg_tally)++;
        }
    }

    if (do_timing) {
        // Full visibility retains the production causal/positions/seq_id path
        // while ensuring all n_comp rows perform score math.
        const int32_t hpos = (int32_t)(RATIO * n_comp);
        const int32_t hseq = 0;
        CHECK(cudaMemcpy(d_positions, &hpos, sizeof(hpos), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_seq_id, &hseq, sizeof(hseq), cudaMemcpyHostToDevice));
        printf("\nDeep timing: best of 5, kernel only, causal full visibility\n");
        for (int v = 0; v < N_VARIANTS; v++) {
            for (int storage = 0; storage < 2; storage++) {
                const unsigned char *codes = storage ? d_codes : NULL;
                const float *scales = storage ? d_scale : NULL;
                deep_us[v][storage] = time_score(
                    (variant_id)v, d_scores[v], d_q, d_weights, d_index,
                    n_comp, N_TOKENS, 0u, RATIO, COMP_CAP, SCALE, 1,
                    d_positions, d_seq_id, codes, scales, d_q_scale,
                    d_index_f32_scale);
                printf("  %-8s %-3s  %.3f us/launch  %.3f projected ms/step\n",
                       variant_name[v], storage ? "FP4" : "F32",
                       deep_us[v][storage], deep_us[v][storage] * 21.0f / 1000.0f);
            }
        }
        for (int storage = 0; storage < 2; storage++) {
            const float measured = deep_us[BASELINE][storage];
            const float deviation = fabsf(measured - 760.0f) / 760.0f;
            printf("  baseline-vs-production %-3s: proto %.3f us vs ~760.000 us",
                   storage ? "FP4" : "F32", measured);
            if (deviation > 0.15f)
                printf("  WARNING: deviation %.1f%% exceeds 15%%\n", deviation * 100.0f);
            else
                printf("  sanity PASS (deviation %.1f%%)\n", deviation * 100.0f);
        }
    }

    for (int v = 0; v < N_VARIANTS; v++) cudaFree(d_scores[v]);
    cudaFree(d_index); cudaFree(d_codes); cudaFree(d_scale);
    cudaFree(d_index_f32_scale);
    cudaFree(d_q); cudaFree(d_q_scale); cudaFree(d_weights);
    cudaFree(d_positions); cudaFree(d_seq_id);
    cudaFree(d_topk_baseline); cudaFree(d_topk_variant);
    cudaFree(d_topk_mask); cudaFree(d_topk_flip_count);
}

// ---------------------------------------------------------------------------
// NT4 batch-invariance leg (engine-integration proof, 2026-07-19).  The
// engine's multiseq decode batches ride grid.y with a distinct (seq, pos)
// per token; every oracle leg above runs n_tokens=1, so this leg proves the
// V5D kernel's per-token addressing directly: a 4-token batch must
// reproduce, bitwise, each token's own n_tokens=1 launch (per-token CTAs
// and independent MMA output rows make batch results token-separable by
// construction -- this checks the ADDRESSING, i.e. the t-indexed q /
// q_block_scale / positions / seq_id reads and the t-strided score write).
// Tokens cover full/tail/half visibility on three different banks, both
// storage legs.  Any bitwise mismatch fails the aggregate.
// ---------------------------------------------------------------------------
static void run_nt4_batch_invariance(int *fail_tally) {
    const uint32_t HEAD_DIM = 128u;
    const uint32_t N_HEAD = 64u;
    const uint32_t RATIO = 4u;
    const uint32_t n_comp = 3062u;
    const uint32_t COMP_CAP = n_comp;
    const uint32_t N_BANKS = 3u;
    const uint32_t NT = 4u;
    const float SCALE = 0.08838834764831845f;
    const uint32_t total_rows = N_BANKS * COMP_CAP;
    const uint64_t index_n = (uint64_t)total_rows * HEAD_DIM;

    float *d_index, *d_encode, *d_scale, *d_kscale, *d_q, *d_weights, *d_qscale;
    unsigned char *d_codes;
    float *d_batch, *d_single;
    int32_t *d_positions, *d_seq_id;
    CHECK(cudaMalloc(&d_index, index_n * sizeof(float)));
    CHECK(cudaMalloc(&d_encode, index_n * sizeof(float)));
    CHECK(cudaMalloc(&d_codes, (uint64_t)total_rows * 64u));
    CHECK(cudaMalloc(&d_scale, (uint64_t)total_rows * 4u * sizeof(float)));
    CHECK(cudaMalloc(&d_kscale, (uint64_t)total_rows * 4u * sizeof(float)));
    CHECK(cudaMalloc(&d_q, (uint64_t)NT * N_HEAD * HEAD_DIM * sizeof(float)));
    CHECK(cudaMalloc(&d_weights, (uint64_t)NT * N_HEAD * sizeof(float)));
    CHECK(cudaMalloc(&d_qscale, (uint64_t)NT * N_HEAD * 4u * sizeof(float)));
    CHECK(cudaMalloc(&d_positions, NT * sizeof(int32_t)));
    CHECK(cudaMalloc(&d_seq_id, NT * sizeof(int32_t)));
    CHECK(cudaMalloc(&d_batch, (uint64_t)NT * n_comp * sizeof(float)));
    CHECK(cudaMalloc(&d_single, (uint64_t)n_comp * sizeof(float)));

    // Same commit pipeline as run_shape: QAT-commit K, snapshot for the
    // packed leg, hoist the dense-F32 K scales, QAT-commit q with the
    // scale emit the engine's scale-only QAT provides.
    unsigned char *d_scratch_codes;
    float *d_scratch_scale;
    CHECK(cudaMalloc(&d_scratch_codes, (uint64_t)total_rows * 64u));
    CHECK(cudaMalloc(&d_scratch_scale, (uint64_t)total_rows * 4u * sizeof(float)));
    fill_index_kernel<<<total_rows, 128>>>(d_index, total_rows, 7u);
    fill_q_weights_kernel<<<NT, 128>>>(d_q, d_weights, NT, 19u);
    indexer_fp4_encode_rows_kernel<<<total_rows, 128>>>(
        d_index, total_rows, HEAD_DIM, d_scratch_codes, d_scratch_scale);
    CHECK(cudaGetLastError());
    cudaFree(d_scratch_codes);
    cudaFree(d_scratch_scale);
    indexer_block_scale_rows_kernel<<<total_rows, 32>>>(
        d_index, total_rows, d_kscale);
    CHECK(cudaMemcpy(d_encode, d_index, index_n * sizeof(float),
                     cudaMemcpyDeviceToDevice));
    indexer_fp4_encode_rows_kernel<<<total_rows, 128>>>(
        d_encode, total_rows, HEAD_DIM, d_codes, d_scale);
    unsigned char *d_q_codes;
    CHECK(cudaMalloc(&d_q_codes, (uint64_t)NT * N_HEAD * 64u));
    indexer_fp4_encode_rows_kernel<<<NT * N_HEAD, 128>>>(
        d_q, NT * N_HEAD, HEAD_DIM, d_q_codes, d_qscale);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    cudaFree(d_encode);
    cudaFree(d_q_codes);

    // full / tail / half visibility on banks 0/1/2, plus a full-visibility
    // repeat of bank 0 (two rows sharing a bank, different q).
    const int32_t hpos[4] = {
        (int32_t)(RATIO * n_comp), (int32_t)(RATIO * (n_comp - 1u)),
        (int32_t)(RATIO * (n_comp / 2u)), (int32_t)(RATIO * n_comp)
    };
    const int32_t hseq[4] = {0, 1, 2, 0};
    CHECK(cudaMemcpy(d_positions, hpos, sizeof(hpos), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_seq_id, hseq, sizeof(hseq), cudaMemcpyHostToDevice));

    printf("\nNT4 batch-invariance: n_comp=%u, n_tokens=%u, "
           "banks {0,1,2,0}, visibility {full,tail,half,full}\n", n_comp, NT);
    for (int storage = 0; storage < 2; storage++) {
        const unsigned char *codes = storage ? d_codes : NULL;
        const float *scales = storage ? d_scale : NULL;
        indexer_scores_multiseq_v5d_kernel<<<dim3((n_comp + 31u) / 32u, NT, 1),
                                             128>>>(
            d_batch, d_q, d_weights, d_index, n_comp, NT, 0u, RATIO,
            COMP_CAP, SCALE, 1, d_positions, d_seq_id, codes, scales,
            d_qscale, d_kscale);
        CHECK(cudaGetLastError());
        for (uint32_t t = 0; t < NT; t++) {
            indexer_scores_multiseq_v5d_kernel<<<dim3((n_comp + 31u) / 32u,
                                                      1, 1), 128>>>(
                d_single, d_q + (uint64_t)t * N_HEAD * HEAD_DIM,
                d_weights + (uint64_t)t * N_HEAD, d_index, n_comp, 1u, 0u,
                RATIO, COMP_CAP, SCALE, 1, d_positions + t, d_seq_id + t,
                codes, scales, d_qscale + (uint64_t)t * N_HEAD * 4u,
                d_kscale);
            CHECK(cudaGetLastError());
            CHECK(cudaDeviceSynchronize());
            char tag[64];
            snprintf(tag, sizeof(tag), "NT4 t=%u %s batch-vs-single",
                     t, storage ? "FP4" : "F32");
            const unsigned long long mism = report_f32(
                tag, d_batch + (uint64_t)t * n_comp, d_single,
                (uint64_t)n_comp, n_comp);
            if (mism != 0) (*fail_tally)++;
        }
    }

    cudaFree(d_index); cudaFree(d_codes); cudaFree(d_scale);
    cudaFree(d_kscale); cudaFree(d_q); cudaFree(d_weights);
    cudaFree(d_qscale); cudaFree(d_positions); cudaFree(d_seq_id);
    cudaFree(d_batch); cudaFree(d_single);
}

static void print_kernel_attributes(const char *name, const void *kernel) {
    cudaFuncAttributes attributes = {};
    CHECK(cudaFuncGetAttributes(&attributes, kernel));
    printf("  %-8s sharedSizeBytes=%zu  numRegs=%d  maxThreadsPerBlock=%d\n",
           name, attributes.sharedSizeBytes, attributes.numRegs,
           attributes.maxThreadsPerBlock);
}

int main(int argc, char **argv) {
    const uint32_t deep_n_comp = argc > 1 ? (uint32_t)strtoul(argv[1], NULL, 10) : 59082u;
    const uint32_t shallow_n_comp = 3062u;
    if (deep_n_comp == 0u || deep_n_comp > 536870911u) {
        fprintf(stderr, "n_comp_deep must be in [1, 536870911]\n");
        return 1;
    }

    bool exact[N_VARIANTS] = {
        true, true, true, true, true, true, true, true, true, true, true
    };
    int fail_tally = 0;
    int leg_tally = 0;
    float deep_us[N_VARIANTS][2] = {{0.0f}};

    // Request the full shared-memory carveout for the V5 family to target the
    // ncu-observed "Block Limit Shared Mem" cap rather than reserving L1.
    CHECK(cudaFuncSetAttribute((const void *)indexer_scores_multiseq_v5_kernel,
                               cudaFuncAttributePreferredSharedMemoryCarveout, 100));
    CHECK(cudaFuncSetAttribute((const void *)indexer_scores_multiseq_v5b_kernel,
                               cudaFuncAttributePreferredSharedMemoryCarveout, 100));
    CHECK(cudaFuncSetAttribute((const void *)indexer_scores_multiseq_v5c_kernel,
                               cudaFuncAttributePreferredSharedMemoryCarveout, 100));
    CHECK(cudaFuncSetAttribute((const void *)indexer_scores_multiseq_v5c32_kernel,
                               cudaFuncAttributePreferredSharedMemoryCarveout, 100));
    CHECK(cudaFuncSetAttribute((const void *)indexer_scores_multiseq_v5d_kernel,
                               cudaFuncAttributePreferredSharedMemoryCarveout, 100));

    printf("proto_indexer_score: strict V1/V3 bits; "
           "V2/V2B/V4/V5/V5B/V5C/V5C32/V5D gate at >2 ULP\n");
    printf("target GB10: 48 SMs, 100 KB smem/SM, 64K regs/SM, sm_121\n");
    printf("\nKernel attributes (static, compile-time):\n");
    print_kernel_attributes("BASELINE", (const void *)indexer_scores_multiseq_kernel);
    print_kernel_attributes("V5C32", (const void *)indexer_scores_multiseq_v5c32_kernel);
    print_kernel_attributes("V5D", (const void *)indexer_scores_multiseq_v5d_kernel);
    run_shape("DEEP", deep_n_comp, 1, exact, &fail_tally, &leg_tally, deep_us);
    run_shape("SHALLOW", shallow_n_comp, 0, exact, &fail_tally, &leg_tally, deep_us);
    // NT4 failures land in fail_tally (and therefore the aggregate) but not
    // in any per-variant `exact` slot: the leg proves engine-shaped batch
    // addressing, which is orthogonal to the per-variant oracle table.
    int nt4_fails_before = fail_tally;
    run_nt4_batch_invariance(&fail_tally);
    printf("\nNT4 batch-invariance: %s\n",
           fail_tally == nt4_fails_before ? "PASS (8/8 bitwise)" : "FAIL");

    const float baseline_us = 0.5f * (deep_us[BASELINE][0] + deep_us[BASELINE][1]);
    printf("\nRanking latency is the mean of the F32 and FP4 best-of-5 timings.\n");
    printf("\nvariant | oracle pass | µs/launch deep | projected ms/step | speedup\n");
    for (int v = 0; v < N_VARIANTS; v++) {
        const float us = 0.5f * (deep_us[v][0] + deep_us[v][1]);
        const float speedup = v == BASELINE ? 1.0f : baseline_us / us;
        printf("%-8s | %-9s | %17.3f | %17.3f | %.2fx\n",
               variant_name[v], exact[v] ? "PASS" : "FAIL",
               us, us * 21.0f / 1000.0f, speedup);
    }
    const variant_id v5_family[] = {V5, V5B, V5C, V5C32, V5D};
    const float fp4_floor_ms = 0.500f;
    printf("\nV5-family per-leg deep timings (21 layers):\n");
    printf("variant  | F32 us/launch | F32 ms/step | FP4 us/launch | FP4 ms/step | "
           "FP4 delta ms/step | %% above floor\n");
    for (uint32_t i = 0; i < sizeof(v5_family) / sizeof(v5_family[0]); i++) {
        const variant_id v = v5_family[i];
        const float f32_ms = deep_us[v][0] * 21.0f / 1000.0f;
        const float fp4_ms = deep_us[v][1] * 21.0f / 1000.0f;
        const float fp4_delta_ms = fp4_ms - fp4_floor_ms;
        const float fp4_above_pct = fp4_delta_ms * 100.0f / fp4_floor_ms;
        printf("%-8s | %13.3f | %11.3f | %13.3f | %11.3f | %+12.3f | %+12.1f%%\n",
               variant_name[v], deep_us[v][0], f32_ms, deep_us[v][1], fp4_ms,
               fp4_delta_ms, fp4_above_pct);
    }
    printf("F32 compulsory stream floor: 3.200 ms/step (21 layers, 240K shape)\n");
    printf("fp4 compulsory stream floor: ~0.500 ms/step (21 layers, 240K shape)\n");
    printf("\nPROTO-INDEXER-SCORE: oracle legs=%d, variant-leg failures=%d, aggregate=%s\n",
           leg_tally, fail_tally, fail_tally == 0 ? "PASS" : "FAIL");
    return fail_tally != 0;
}
