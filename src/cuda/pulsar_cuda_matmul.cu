#include "pulsar_cuda_internal.h"



__global__ static void embed_token_hc_kernel(pulsar_hc_t *out, const unsigned short *w, uint32_t token, uint32_t n_vocab, uint32_t n_embd, uint32_t n_hc) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t n = n_embd * n_hc;
    if (i >= n) return;
    uint32_t e = i % n_embd;
    uint32_t tok = token < n_vocab ? token : n_vocab - 1; /* clamp: an OOB token id is a wild global read */
    pulsar_hc_store(out, i, __half2float(reinterpret_cast<const __half *>(w)[(uint64_t)tok * n_embd + e]));
}



__global__ static void embed_tokens_hc_kernel(
        pulsar_hc_t *out,
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
    pulsar_hc_store(out, gid, __half2float(w[(uint64_t)tok * n_embd + d]));
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


/* Small-batch (2..4 token) f16 GEMV: one weight-row read serves all NT tokens,
 * replacing the cuBLAS GemmEx path which is latency-bound at these shapes and
 * needs an f32->f16 activation convert + tmp alloc per call. Per-token loop
 * structure and shared-memory reduction match matmul_f16_kernel exactly, so each
 * token's output is bit-identical to the n=1 kernel run on that token alone. */
template <int NT>
__global__ static void matmul_f16_nt_kernel(
        float *out,
        const __half *w,
        const float *x,
        uint64_t in_dim,
        uint64_t out_dim) {
    uint64_t row = (uint64_t)blockIdx.x;
    if (row >= out_dim) return;

    float sum[NT];
    #pragma unroll
    for (int t = 0; t < NT; t++) sum[t] = 0.0f;
    const __half *wr = w + row * in_dim;
    for (uint64_t i = threadIdx.x; i < in_dim; i += blockDim.x) {
        const float wv = __half2float(wr[i]);
        #pragma unroll
        for (int t = 0; t < NT; t++) sum[t] += wv * x[t * in_dim + i];
    }

    __shared__ float partial[256];
    #pragma unroll
    for (int t = 0; t < NT; t++) {
        partial[threadIdx.x] = sum[t];
        __syncthreads();
        for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
            __syncthreads();
        }
        if (threadIdx.x == 0) out[t * out_dim + row] = partial[0];
        __syncthreads();
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



__global__ static void f32_to_f16_kernel(__half *out, const float *x, uint64_t n) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half(x[i]);
}



/* BF16 is the high 16 bits of f32, so conversions are pure bit ops (no header). */
__device__ __forceinline__ static float bf16_to_f32(uint16_t b) {
    return __uint_as_float((uint32_t)b << 16);
}


__global__ static void f32_to_bf16_kernel(uint16_t *out, const float *x, uint64_t n) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const uint32_t u = __float_as_uint(x[i]);
        out[i] = (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);  /* round-to-nearest-even */
    }
}


__global__ static void matmul_bf16_kernel(
        float *out,
        const uint16_t *w,
        const float *x,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t n_tok) {
    uint64_t row = (uint64_t)blockIdx.x;
    uint64_t tok = (uint64_t)blockIdx.y;
    if (row >= out_dim || tok >= n_tok) return;
    float sum = 0.0f;
    const uint16_t *wr = w + row * in_dim;
    const float *xr = x + tok * in_dim;
    for (uint64_t i = threadIdx.x; i < in_dim; i += blockDim.x) {
        sum += bf16_to_f32(wr[i]) * xr[i];
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



/* MXFP8 weight block (33B: [E8M0][32 e4m3]) . raw f32 activation block.
 * Decode-side attn-output dot: the e4m3 weight is expanded to float and MAC'd
 * against the unquantized activation, so the only quantization error left on
 * this path is the weight's own (e4m3-packing the activations pushed the
 * perplexity gate; int8 requant is what this change removes). The contiguous
 * 32-element inner loop is what nvcc vectorizes — a lane-per-element
 * (mmvq-style) mapping profiled 2.2x slower. Returns wscale*sum(e4m3*x). */
__device__ __forceinline__ int mx_sfoff(int row, int kb, int KBp);   /* defined below */


/* De-interleaved MXFP8 block dot: contiguous 32-E4M3 block (aligned) + separate
 * E8M0 scale byte. Reads 8 aligned uint32 (4 fp8 each) -> vectorized, vs the raw
 * 33B interleaved kernels' misaligned byte-wise weight reads. */
__device__ __forceinline__ static float dev_dot_fp8mx_deint_block(
        const __nv_fp8_e4m3 *blk, unsigned char scale_byte, const float *xb) {
    const float wscale = __int_as_float((uint32_t)scale_byte << 23);   /* E8M0 */
    float s = 0.0f;
#pragma unroll
    for (int g = 0; g < 8; g++) {
        const uint32_t p = ((const uint32_t *)blk)[g];
        const __nv_fp8_e4m3 *q = (const __nv_fp8_e4m3 *)&p;
#pragma unroll
        for (int j = 0; j < 4; j++) s += __half2float((__half)q[j]) * xb[g * 4 + j];
    }
    return wscale * s;
}


__device__ __forceinline__ static float dev_dot_fp8mx_f32_block(
        const unsigned char *wblk, const float *x, uint64_t bn) {
    const float wscale = __int_as_float((uint32_t)wblk[0] << 23);   /* E8M0 */
    float s = 0.0f;
    for (uint64_t i = 0; i < bn; i++) {
        const __half wh = (__half)(*(const __nv_fp8_e4m3 *)&wblk[1 + i]);
        s += __half2float(wh) * x[i];
    }
    return wscale * s;
}



/* Same dot against an activation block already staged in registers (full
 * 32-element block only). */
__device__ __forceinline__ static float dev_dot_fp8mx_xreg_block(
        const unsigned char *wblk, const float *xb) {
    const float wscale = __int_as_float((uint32_t)wblk[0] << 23);   /* E8M0 */
    float s = 0.0f;
#pragma unroll
    for (int i = 0; i < 32; i++) {
        const __half wh = (__half)(*(const __nv_fp8_e4m3 *)&wblk[1 + i]);
        s += __half2float(wh) * xb[i];
    }
    return wscale * s;
}



/* Fused attn-output kernels: MXFP8 weight rows dotted against RAW f32
 * activations. Each warp covers PULSAR_FP8MX_ROWS consecutive output rows and
 * loads every 32-float activation block into registers ONCE for all of them,
 * so per-row activation traffic drops to int8-path parity (per-row f32 reads
 * straight from global measured ~13% slower end-to-end at decode; a shared-
 * memory staging variant measured worse still). */
enum { PULSAR_FP8MX_ROWS = 4 };



template<bool DEINT>
__global__ static void matmul_fp8mx_hc_expand_warp8_kernel(
        pulsar_hc_t *out_hc,
        float *block_out,
        const float *block_add,
        const pulsar_hc_t *residual_hc,
        const float *split,
        const unsigned char *w,
        const __nv_fp8_e4m3 *wdata,
        const unsigned char *wscale,
        int KBp,
        const float *x,
        uint64_t in_dim,
        uint64_t out_dim,
        uint32_t n_embd,
        uint32_t n_hc,
        uint64_t blocks,
        int has_add) {
    const uint64_t row0 = ((uint64_t)blockIdx.x * 8u + (threadIdx.x >> 5u)) * PULSAR_FP8MX_ROWS;
    const uint32_t lane = threadIdx.x & 31u;
    if (row0 >= out_dim) return;
    const uint32_t nr = out_dim - row0 < PULSAR_FP8MX_ROWS ? (uint32_t)(out_dim - row0)
                                                        : (uint32_t)PULSAR_FP8MX_ROWS;
    const unsigned char *wr = w + row0 * blocks * 33u;
    float acc[PULSAR_FP8MX_ROWS] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint64_t b = lane; b < blocks; b += 32u) {
        const uint64_t i0 = b * 32;
        const uint64_t bn = in_dim - i0 < 32 ? in_dim - i0 : 32;
        if (bn == 32u) {
            float xb[32];
#pragma unroll
            for (int k = 0; k < 8; k++) {
                *(float4 *)&xb[k * 4] = *(const float4 *)&x[i0 + (uint32_t)k * 4u];
            }
#pragma unroll
            for (uint32_t r = 0; r < PULSAR_FP8MX_ROWS; r++) {
                if (r >= nr) continue;
                if (DEINT) {
                    const uint32_t rw = (uint32_t)(row0 + r);
                    acc[r] += dev_dot_fp8mx_deint_block(wdata + (uint64_t)rw * in_dim + i0,
                                                        wscale[mx_sfoff((int)rw, (int)b, KBp)], xb);
                } else {
                    acc[r] += dev_dot_fp8mx_xreg_block(wr + (r * blocks + b) * 33u, xb);
                }
            }
        } else {
            for (uint32_t r = 0; r < nr; r++) {
                acc[r] += dev_dot_fp8mx_f32_block(wr + (r * blocks + b) * 33u, x + i0, bn);
            }
        }
    }
    for (uint32_t r = 0; r < nr; r++) {
        const float red = warp_sum_f32(acc[r]);
        if (lane != 0) continue;
        const uint32_t d = (uint32_t)(row0 + r);
        block_out[d] = red;
        float block_v = red;
        if (has_add) block_v += block_add[d];
        const float *post = split + n_hc;
        const float *comb = split + 2u * n_hc;
        for (uint32_t dst_hc = 0; dst_hc < n_hc; dst_hc++) {
            float hc_acc = block_v * post[dst_hc];
            for (uint32_t src_hc = 0; src_hc < n_hc; src_hc++) {
                const float comb_v = comb[dst_hc + (uint64_t)src_hc * n_hc];
                const float res_v = pulsar_hc_load(residual_hc, (uint64_t)src_hc * n_embd + d);
                hc_acc += comb_v * res_v;
            }
            pulsar_hc_store(out_hc, (uint64_t)dst_hc * n_embd + d, hc_acc);
        }
    }
}



template<bool DEINT>
__global__ static void grouped_fp8mx_a_warp8_kernel(
        float *low,
        const unsigned char *w,
        const __nv_fp8_e4m3 *wdata,
        const unsigned char *wscale,
        int KBp,
        const float *x,
        uint64_t group_dim,
        uint64_t rank,
        uint32_t n_groups,
        uint32_t n_tokens,
        uint64_t blocks) {
    const uint64_t row0 = ((uint64_t)blockIdx.x * 8u + (threadIdx.x >> 5u)) * PULSAR_FP8MX_ROWS;
    const uint64_t tok = (uint64_t)blockIdx.y;
    const uint32_t lane = threadIdx.x & 31u;
    const uint64_t low_dim = (uint64_t)n_groups * rank;
    if (row0 >= low_dim || tok >= n_tokens) return;
    const uint32_t nr = low_dim - row0 < PULSAR_FP8MX_ROWS ? (uint32_t)(low_dim - row0)
                                                        : (uint32_t)PULSAR_FP8MX_ROWS;
    const unsigned char *wr = w + row0 * blocks * 33u;
    const uint64_t group = row0 / rank;
    float acc[PULSAR_FP8MX_ROWS] = {0.0f, 0.0f, 0.0f, 0.0f};

    if ((row0 + nr - 1) / rank == group) {
        /* Common case (rank % PULSAR_FP8MX_ROWS == 0): all rows share the
         * group's activation row, so its blocks are loaded once per warp. */
        const float *xr = x + (tok * (uint64_t)n_groups + group) * group_dim;
        for (uint64_t b = lane; b < blocks; b += 32u) {
            const uint64_t i0 = b * 32;
            const uint64_t bn = group_dim - i0 < 32 ? group_dim - i0 : 32;
            if (bn == 32u) {
                float xb[32];
#pragma unroll
                for (int k = 0; k < 8; k++) {
                    *(float4 *)&xb[k * 4] = *(const float4 *)&xr[i0 + (uint32_t)k * 4u];
                }
#pragma unroll
                for (uint32_t r = 0; r < PULSAR_FP8MX_ROWS; r++) {
                    if (r >= nr) continue;
                    if (DEINT) {
                        const uint32_t rw = (uint32_t)(row0 + r);
                        acc[r] += dev_dot_fp8mx_deint_block(wdata + (uint64_t)rw * group_dim + i0,
                                                            wscale[mx_sfoff((int)rw, (int)b, KBp)], xb);
                    } else {
                        acc[r] += dev_dot_fp8mx_xreg_block(wr + (r * blocks + b) * 33u, xb);
                    }
                }
            } else {
                for (uint32_t r = 0; r < nr; r++) {
                    acc[r] += dev_dot_fp8mx_f32_block(wr + (r * blocks + b) * 33u, xr + i0, bn);
                }
            }
        }
    } else {
        for (uint32_t r = 0; r < nr; r++) {
            const float *xr = x + (tok * (uint64_t)n_groups + (row0 + r) / rank) * group_dim;
            for (uint64_t b = lane; b < blocks; b += 32u) {
                const uint64_t i0 = b * 32;
                const uint64_t bn = group_dim - i0 < 32 ? group_dim - i0 : 32;
                acc[r] += dev_dot_fp8mx_f32_block(wr + (r * blocks + b) * 33u, xr + i0, bn);
            }
        }
    }
    for (uint32_t r = 0; r < nr; r++) {
        const float red = warp_sum_f32(acc[r]);
        if (lane == 0) low[tok * low_dim + row0 + r] = red;
    }
}


/* Small-batch (2..4 token) variant of the grouped o_a GEMV: one launch whose
 * weight blocks are read once and served from L1 across the NT tokens' dots,
 * replacing the per-group block-scaled tensor-core GEMMs that dominated the
 * spec-verify launch storm (8 GEMMs/layer at 2-4 rows each). Per-(row,token)
 * block order and dot helper match grouped_fp8mx_a_warp8_kernel's DEINT fast
 * path exactly, so each token's output is bit-identical to the n=1 kernel.
 * Caller guarantees: deint weight available, rank % PULSAR_FP8MX_ROWS == 0 (row
 * quads never straddle a group), group_dim % 32 == 0 (whole blocks). */
template <int NT>
__global__ static void grouped_fp8mx_a_nt_kernel(
        float *low,
        const __nv_fp8_e4m3 *wdata,
        const unsigned char *wscale,
        int KBp,
        const float *x,
        uint64_t group_dim,
        uint64_t rank,
        uint32_t n_groups,
        uint64_t blocks) {
    const uint64_t row0 = ((uint64_t)blockIdx.x * 8u + (threadIdx.x >> 5u)) * PULSAR_FP8MX_ROWS;
    const uint32_t lane = threadIdx.x & 31u;
    const uint64_t low_dim = (uint64_t)n_groups * rank;
    if (row0 >= low_dim) return;
    const uint64_t group = row0 / rank;
    float acc[PULSAR_FP8MX_ROWS][NT];
#pragma unroll
    for (uint32_t r = 0; r < PULSAR_FP8MX_ROWS; r++)
#pragma unroll
        for (int t = 0; t < NT; t++) acc[r][t] = 0.0f;
    const float *xg = x + group * group_dim;
    const uint64_t tok_stride = (uint64_t)n_groups * group_dim;
    for (uint64_t b = lane; b < blocks; b += 32u) {
        const uint64_t i0 = b * 32;
#pragma unroll
        for (int t = 0; t < NT; t++) {
            float xb[32];
            const float *xr = xg + (uint64_t)t * tok_stride + i0;
#pragma unroll
            for (int k = 0; k < 8; k++) {
                *(float4 *)&xb[k * 4] = *(const float4 *)&xr[(uint32_t)k * 4u];
            }
#pragma unroll
            for (uint32_t r = 0; r < PULSAR_FP8MX_ROWS; r++) {
                const uint32_t rw = (uint32_t)(row0 + r);
                acc[r][t] += dev_dot_fp8mx_deint_block(wdata + (uint64_t)rw * group_dim + i0,
                                                       wscale[mx_sfoff((int)rw, (int)b, KBp)], xb);
            }
        }
    }
#pragma unroll
    for (uint32_t r = 0; r < PULSAR_FP8MX_ROWS; r++) {
#pragma unroll
        for (int t = 0; t < NT; t++) {
            const float red = warp_sum_f32(acc[r][t]);
            if (lane == 0) low[(uint64_t)t * low_dim + row0 + r] = red;
        }
    }
}



int pulsar_gpu_embed_token_hc_tensor(pulsar_gpu_tensor *out_hc, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint32_t n_vocab, uint32_t token, uint32_t n_embd, uint32_t n_hc) {
    if (!out_hc || !model_map || weight_offset >= model_size || n_vocab == 0) return 0;
    /* The kernel writes n_embd*n_hc carrier samples; validate like the batched
     * sibling does. Before the BF16 narrowing an undersized out_hc still had 2x
     * slack — it does not any more, so this check is load-bearing (task #62). */
    if (out_hc->bytes < (uint64_t)n_embd * n_hc * PULSAR_HC_ELT_SIZE) return 0;
    uint64_t weight_bytes = (uint64_t)n_vocab * n_embd * sizeof(uint16_t);
    if (weight_offset > model_size || weight_bytes > model_size - weight_offset) return 0;
    const char *wptr = cuda_model_range_ptr(model_map, weight_offset, weight_bytes, "token_embd");
    if (!wptr) return 0;
    uint32_t n = n_embd * n_hc;
    embed_token_hc_kernel<<<(n + 255) / 256, 256>>>((pulsar_hc_t *)out_hc->ptr, (const unsigned short *)wptr, token, n_vocab, n_embd, n_hc);
    return cuda_ok(cudaGetLastError(), "embed token launch");
}



int pulsar_gpu_embed_tokens_hc_tensor(
        pulsar_gpu_tensor       *out_hc,
        const pulsar_gpu_tensor *tokens_t,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                n_vocab,
        uint32_t                n_tokens,
        uint32_t                n_embd,
        uint32_t                n_hc) {
    if (!out_hc || !tokens_t || !model_map ||
        weight_offset > model_size ||
        (uint64_t)n_vocab * n_embd * sizeof(uint16_t) > model_size - weight_offset ||
        tokens_t->bytes < (uint64_t)n_tokens * sizeof(int32_t) ||
        out_hc->bytes < (uint64_t)n_tokens * n_hc * n_embd * PULSAR_HC_ELT_SIZE) {
        return 0;
    }
    const char *wptr = cuda_model_range_ptr(model_map, weight_offset,
                                            (uint64_t)n_vocab * n_embd * sizeof(uint16_t),
                                            "token_embd");
    if (!wptr) return 0;
    uint64_t n = (uint64_t)n_tokens * n_hc * n_embd;
    embed_tokens_hc_kernel<<<(n + 255) / 256, 256>>>(
        (pulsar_hc_t *)out_hc->ptr,
        (const int32_t *)tokens_t->ptr,
        (const __half *)wptr,
        n_vocab, n_tokens, n_embd, n_hc);
    return cuda_ok(cudaGetLastError(), "embed tokens launch");
}


static int g_cublaslt_ready = 0;


static int cublaslt_ensure(void) {
    if (g_cublaslt_ready) return 1;
    if (cublasLtCreate(&g_cublaslt) != CUBLAS_STATUS_SUCCESS) return 0;
    g_cublaslt_ready = 1; return 1;
}


static inline int mx_rup(int x, int n) { return (x + n - 1) / n * n; }


__device__ __forceinline__ int mx_sfoff(int row, int kb, int KBp) {
    return ((row / 128) * (KBp / 4) + (kb / 4)) * 512
           + (row % 32) * 16 + ((row % 128) / 32) * 4 + (kb % 4);
}


/* X[rows,K] f32 -> E4M3 data[K,rows]col + swizzled E8M0 scale. one warp per (row,kb). */
__global__ static void mxfp8_quant_act_kernel(const float *X, int rows, int K, int KBp,
                                              __nv_fp8_e4m3 *data, unsigned char *scale) {
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x & 31;
    int KB = K / 32; if (warp >= rows * KB) return;
    int row = warp / KB, kb = warp % KB;
    float v = X[(size_t)row * K + kb * 32 + lane], a = fabsf(v);
    for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
    int se = -127; if (a > 0.f) { int e = (int)floorf(log2f(a)); se = e - 7; }
    if (se < -127) se = -127; if (se > 127) se = 127;
    data[(size_t)(kb * 32 + lane) + (size_t)row * K] = (__nv_fp8_e4m3)(v * exp2f((float)-se));
    if (lane == 0) scale[mx_sfoff(row, kb, KBp)] = (unsigned char)(se + 127);
}


/* Grouped variant for the attn-output "a" projection. The activation tensor is
 * heads[tok][group][K] but each group is an independent GEMM, so the E4M3 data
 * is regrouped as n_groups slabs of [K, n_tokens] col-major and the E8M0 scale
 * as n_groups independent swizzles of scale_slab bytes (token = scale row). */
__global__ static void mxfp8_quant_act_grouped_kernel(const float *X, int n_tokens, int n_groups,
                                                      int K, int KBp, __nv_fp8_e4m3 *data,
                                                      unsigned char *scale, size_t scale_slab) {
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x & 31;
    int KB = K / 32; if (warp >= n_tokens * n_groups * KB) return;
    int row = warp / KB, kb = warp % KB;        /* row = tok * n_groups + g */
    int g = row % n_groups, tok = row / n_groups;
    float v = X[(size_t)row * K + kb * 32 + lane], a = fabsf(v);
    for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
    int se = -127; if (a > 0.f) { int e = (int)floorf(log2f(a)); se = e - 7; }
    if (se < -127) se = -127; if (se > 127) se = 127;
    data[((size_t)g * n_tokens + tok) * K + kb * 32 + lane] = (__nv_fp8_e4m3)(v * exp2f((float)-se));
    if (lane == 0) scale[(size_t)g * scale_slab + mx_sfoff(tok, kb, KBp)] = (unsigned char)(se + 127);
}


/* GGUF MXFP8 weight (row-major [out,in], 33B blocks: [E8M0][32xE4M3]) ->
 * E4M3 data[in,out]col + swizzled E8M0 scale. one warp per (out,kb). */
__global__ static void mxfp8_weight_convert_kernel(const unsigned char *src, int out_dim, int in_dim,
                                                   int KBp, __nv_fp8_e4m3 *data, unsigned char *scale) {
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x & 31;
    int KB = in_dim / 32; if (warp >= out_dim * KB) return;
    int o = warp / KB, kb = warp % KB;
    const unsigned char *blk = src + ((size_t)o * KB + kb) * 33;
    data[(size_t)(kb * 32 + lane) + (size_t)o * in_dim] = *(const __nv_fp8_e4m3 *)&blk[1 + lane];
    if (lane == 0) scale[mx_sfoff(o, kb, KBp)] = blk[0];
}


static std::unordered_map<uint64_t, fp8_mx_weight> g_fp8_mx_by_offset;

/* Offsets whose MXFP8 weight is PRE-STORED in the mmap in the exact device
 * layout (de-interleaved E4M3 data + mx_sfoff-swizzled E8M0 scale, contiguous:
 * [data (in*out B)][scale]). For these the resolver skips cudaMalloc+convert and
 * points cuBLASLt straight at g_model_device_base+offset. Populated once at load
 * (cold path); the resolved fp8_mx_weight is then cached in g_fp8_mx_by_offset
 * exactly like a converted weight, so the per-token hot path never probes this
 * set. */
static std::unordered_set<uint64_t> g_mxfp8_lt_offsets;

/* Direct-mapped front cache for cuda_fp8_mx_weight (file-scope so backend
 * cleanup can invalidate it together with g_fp8_mx_by_offset). */
constexpr uint32_t FP8_FC = 2048u;
static uint64_t g_fp8_fc_off[FP8_FC];        /* zero-init; real offsets are never 0 */
static const fp8_mx_weight *g_fp8_fc_ptr[FP8_FC];


/* lazily de-interleave + swizzle an MXFP8 weight into device buffers, cached by offset. */
static const fp8_mx_weight *cuda_fp8_mx_weight(const void *model_map, uint64_t offset, uint64_t weight_bytes,
                                               uint64_t in_dim, uint64_t out_dim, const char *label) {
    /* The same ~300 weight offsets are resolved once per layer every token on
     * the launch-serializing host thread. A tiny direct-mapped cache in front
     * of the unordered_map skips the probe on the hot repeat; a miss or hash
     * collision just falls through (benign), and the cached pointer is
     * re-validated (map references are stable across inserts). */
    constexpr uint32_t FC = FP8_FC;
    uint64_t *fc_off = g_fp8_fc_off;
    const fp8_mx_weight **fc_ptr = g_fp8_fc_ptr;
    const uint32_t slot = (uint32_t)(((offset >> 5) ^ (offset >> 17)) & (FC - 1u));
    if (offset != 0 && fc_off[slot] == offset) {
        const fp8_mx_weight *p = fc_ptr[slot];
        if (p && p->host_base == model_map && p->in_dim == in_dim && p->out_dim == out_dim) return p;
    }
    auto it = g_fp8_mx_by_offset.find(offset);
    if (it != g_fp8_mx_by_offset.end() && it->second.host_base == model_map &&
        it->second.in_dim == in_dim && it->second.out_dim == out_dim) {
        fc_off[slot] = offset; fc_ptr[slot] = &it->second;
        return &it->second;
    }
    int KB = (int)(in_dim / 32), KBp = mx_rup(KB, 4);
    size_t data_bytes = in_dim * out_dim;
    size_t scale_bytes = (size_t)mx_rup((int)out_dim, 128) * KBp;

    /* Pre-stored MXFP8_LT: the de-interleaved E4M3 data and swizzled E8M0 scale
     * are already laid out contiguously in the mmap at [offset .. offset+data ..
     * +scale]. Skip the cudaMalloc+convert entirely and hand cuBLASLt the
     * device-accessible mmap pointers (g_model_device_base+offset). Byte-for-byte
     * identical to what mxfp8_weight_convert_kernel would have produced. */
    if (g_mxfp8_lt_offsets.count(offset)) {
        __nv_fp8_e4m3 *ltdata =
            (__nv_fp8_e4m3 *)cuda_model_range_ptr(model_map, offset, data_bytes, "fp8_mx_lt data");
        unsigned char *ltscale =
            (unsigned char *)cuda_model_range_ptr(model_map, offset + data_bytes, scale_bytes, "fp8_mx_lt scale");
        if (ltdata && ltscale) {
            fp8_mx_weight w = { model_map, offset, in_dim, out_dim, ltdata, ltscale };
            g_fp8_mx_by_offset[offset] = w;
            const fp8_mx_weight *wp = &g_fp8_mx_by_offset[offset];
            fc_off[slot] = offset; fc_ptr[slot] = wp;
            (void)label;
            return wp;
        }
        /* An LT offset whose mmap pointers didn't resolve must NOT fall through
         * to the generic convert path: that path reads 33B-interleaved blocks,
         * but LT bytes are already de-interleaved, so a convert would be
         * garbage. Fail cleanly instead (dispatch will report the error). */
        fprintf(stderr, "pulsar: MXFP8_LT weight at offset %llu did not resolve to "
                "device-accessible mmap pointers\n", (unsigned long long)offset);
        return NULL;
    }

    const unsigned char *src = (const unsigned char *)cuda_model_range_ptr(model_map, offset, weight_bytes, "fp8_mx");
    if (!src) return NULL;
    __nv_fp8_e4m3 *data = NULL; unsigned char *scale = NULL;
    if (cudaMalloc(&data, data_bytes) != cudaSuccess) return NULL;
    if (cudaMalloc(&scale, scale_bytes) != cudaSuccess) { cudaFree(data); return NULL; }
    cudaMemset(scale, 0, scale_bytes);
    int warps = (int)out_dim * KB;
    mxfp8_weight_convert_kernel<<<(warps * 32 + 255) / 256, 256>>>(src, (int)out_dim, (int)in_dim, KBp, data, scale);
    if (!cuda_ok(cudaGetLastError(), "fp8_mx weight convert")) { cudaFree(data); cudaFree(scale); return NULL; }
    fp8_mx_weight w = { model_map, offset, in_dim, out_dim, data, scale };
    g_fp8_mx_by_offset[offset] = w;
    (void)label;
    const fp8_mx_weight *wp = &g_fp8_mx_by_offset[offset];
    fc_off[slot] = offset; fc_ptr[slot] = wp;
    return wp;
}


/* ---- "quantize once, N GEMMs" activation cache --------------------------
 *
 * In the batched prefill path a single normalized activation (batch_attn_norm,
 * [n_tok x n_embd] f32) feeds up to SEVEN block-scaled MXFP8 projections per
 * ratio-4 layer (q_a, kv, attn compressor kv+gate, indexer compressor kv+gate,
 * indexer_proj).  mxfp8_quant_act_kernel is a PURE function of
 * (x, n_tok, in_dim) -- see the kernel: per-warp, no atomics, no state -- so
 * running it once per consumer re-derives byte-identical E4M3 data and E8M0
 * scales while re-reading the whole f32 activation each time.  Cache the first
 * result and hand it to the remaining GEMMs: bit-identical by construction,
 * and it drops ~6/7 of the activation quant traffic on those layers.
 *
 * ALIASING: the cached buffers must NOT come from cuda_tmp_alloc.  That is one
 * process-global scratch region (pulsar_cuda_runtime.cu) that later callers --
 * attention, MoE, the indexer top-k -- freely overwrite or realloc, and such
 * calls DO run between the consumers above.  Dedicated cudaMalloc'd storage is
 * the only safe home for a value that must survive across unrelated kernels.
 *
 * COHERENCE: the cache is keyed on (x->ptr, n_tok, in_dim), which is NOT enough
 * on its own -- batch_attn_norm keeps its pointer and shape while its CONTENTS
 * change every layer.  So a hit additionally requires that the engine has armed
 * the cache for that exact buffer since the last write to it
 * (pulsar_gpu_mxfp8_act_cache_arm, called immediately after the norm that produces
 * it).  Arming invalidates, so a stale hit would require a write with no arm.
 *
 * THREADING: per-thread state, matching the per-thread CUDA streams -- two
 * sessions prefilling concurrently must not share a quantized activation.
 *
 * TWO ENCODINGS, ONE ARMING.  The same activation also feeds F16 cuBLAS GEMMs
 * (this model stores the compressor/indexer projections as F16 while q_a/kv are
 * MXFP8), and pulsar_gpu_matmul_f16_tensor re-runs f32_to_f16_kernel over the whole
 * activation on every call for exactly the same reason.  That conversion is
 * likewise pure, so the cache carries an f16 copy alongside the MXFP8 one and
 * both are filled lazily -- a layer pays for only the encodings it actually
 * uses.  On a ratio-4 layer that is 1 quantization + 1 conversion instead of
 * 2 + 5. */
struct mxfp8_act_cache_t {
    const void    *key_ptr;      /* armed activation buffer (NULL = disarmed) */
    uint64_t       key_ntok;
    uint64_t       key_in_dim;
    int            valid;        /* xq/sx hold the MXFP8 quant of that (ptr,shape) */
    int            valid_h;      /* xh holds the f16 conversion of that (ptr,shape) */
    __nv_fp8_e4m3 *xq;
    size_t         xq_cap;
    unsigned char *sx;
    size_t         sx_cap;
    __half        *xh;
    size_t         xh_cap;
};
static thread_local mxfp8_act_cache_t g_act_cache;

void pulsar_gpu_mxfp8_act_cache_arm(const pulsar_gpu_tensor *x, uint64_t n_tok, uint64_t in_dim) {
    /* Operational kill switch / A-B measurement handle. Read ONCE (this runs
     * per layer, not per token), and disarming restores the exact pre-cache
     * code path in the GEMMs below. */
    static const int off = getenv("DS4_NO_MXFP8_ACT_CACHE") != NULL;
    g_act_cache.valid = 0;
    g_act_cache.valid_h = 0;
    if (off || !x || !x->ptr || n_tok == 0 || in_dim == 0 || (in_dim % 32) != 0) {
        g_act_cache.key_ptr = NULL;
        return;
    }
    g_act_cache.key_ptr    = x->ptr;
    g_act_cache.key_ntok   = n_tok;
    g_act_cache.key_in_dim = in_dim;
}

void pulsar_gpu_mxfp8_act_cache_disarm(void) {
    g_act_cache.key_ptr = NULL;
    g_act_cache.valid   = 0;
    g_act_cache.valid_h = 0;
}

/* Grow-only device buffer for the cache. cudaFree implicitly synchronizes, so
 * a growth cannot pull the old pointer out from under an in-flight kernel. */
static int mxfp8_act_cache_reserve(void **buf, size_t *cap, size_t need, const char *what) {
    if (*cap >= need) return 1;
    void *p = NULL;
    if (*buf) { (void)cudaFree(*buf); *buf = NULL; *cap = 0; }
    if (cudaMalloc(&p, need) != cudaSuccess) {
        (void)cudaGetLastError();
        fprintf(stderr, "pulsar: MXFP8 activation cache alloc failed for %s (%.2f MiB)\n",
                what ? what : "act", (double)need / 1048576.0);
        return 0;
    }
    *buf = p; *cap = need;
    return 1;
}


static int cuda_matmul_fp8_mx_tensor_labeled(pulsar_gpu_tensor *out, const void *model_map, uint64_t model_size,
        uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim, const pulsar_gpu_tensor *x,
        uint64_t n_tok, const char *label) {
    if (!out || !x || !model_map || in_dim % 32 != 0 || !cublaslt_ensure()) return 0;
    uint64_t KB = in_dim / 32, weight_bytes = out_dim * KB * 33;
    if (weight_offset > model_size || weight_bytes > model_size - weight_offset) return 0;
    if (x->bytes < n_tok * in_dim * sizeof(float) || out->bytes < n_tok * out_dim * sizeof(float)) return 0;
    const fp8_mx_weight *w = cuda_fp8_mx_weight(model_map, weight_offset, weight_bytes, in_dim, out_dim, label);
    if (!w) return 0;
    int ntok = (int)n_tok, KBp = mx_rup((int)KB, 4);
    size_t sx_bytes = (size_t)mx_rup(ntok, 128) * KBp;
    size_t wz = 32u << 20;
    /* xq, sx and the cuBLASLt workspace must be DISTINCT buffers, but
     * cuda_tmp_alloc hands out one shared scratch region (later calls alias or
     * realloc/free earlier ones). Carve three non-overlapping, 256-aligned
     * regions from a single allocation instead. */
    /* Armed activation cache (see mxfp8_act_cache_t above): reuse the E4M3 data
     * and E8M0 scales this activation was already quantized into, and take only
     * the cuBLASLt workspace from the shared tmp region. */
    mxfp8_act_cache_t *ac = NULL;
    if (g_act_cache.key_ptr == x->ptr && g_act_cache.key_ntok == n_tok &&
        g_act_cache.key_in_dim == in_dim) {
        if (mxfp8_act_cache_reserve((void **)&g_act_cache.xq, &g_act_cache.xq_cap,
                                    in_dim * (size_t)ntok, "act data") &&
            mxfp8_act_cache_reserve((void **)&g_act_cache.sx, &g_act_cache.sx_cap,
                                    sx_bytes, "act scale")) {
            ac = &g_act_cache;
        } else {
            g_act_cache.valid = 0;   /* fall back to the per-GEMM quantization */
        }
    }
    __nv_fp8_e4m3 *xq;
    unsigned char *sx;
    void *ws;
    if (ac) {
        xq = ac->xq;
        sx = ac->sx;
        ws = cuda_tmp_alloc(wz, "fp8_mx scratch");
        if (!ws) return 0;
    } else {
        size_t off_xq = 0;
        size_t off_sx = (in_dim * (size_t)ntok + 255) & ~(size_t)255;
        size_t off_ws = (off_sx + sx_bytes + 255) & ~(size_t)255;
        char *scratch = (char *)cuda_tmp_alloc(off_ws + wz, "fp8_mx scratch");
        if (!scratch) return 0;
        xq = (__nv_fp8_e4m3 *)(scratch + off_xq);
        sx = (unsigned char *)(scratch + off_sx);
        ws = scratch + off_ws;
    }
    if (!ac || !ac->valid) {
        cudaMemsetAsync(sx, 0, sx_bytes, 0);
        int warps = ntok * (int)KB;
        mxfp8_quant_act_kernel<<<(warps * 32 + 255) / 256, 256>>>((const float *)x->ptr, ntok, (int)in_dim, KBp, xq, sx);
        if (!cuda_ok(cudaGetLastError(), "fp8_mx act quant")) return 0;
        if (ac) ac->valid = 1;
    }
    /* Shape-keyed handle cache: the desc/layout/preference build plus the
     * heuristic query cost ~100us of host time PER GEMM and only the two
     * scale pointers change between calls of the same (in,out,ntok) shape.
     * The workhorse prefill runs a handful of shapes x 43 layers per chunk,
     * so cache the handles and the chosen algo, and update just the scale
     * pointers per call. Round-robin eviction; entries live for the process
     * (bounded by the slot count). */
    struct lt_shape_cache {
        uint64_t in_dim, out_dim; int ntok; int valid;
        cublasLtMatmulDesc_t op;
        cublasLtMatrixLayout_t la, lb, ld;
        cublasLtMatmulHeuristicResult_t h;
    };
    static lt_shape_cache cache[16];
    static int cache_next;
    lt_shape_cache *e = NULL;
    for (int i = 0; i < 16; i++) {
        if (cache[i].valid && cache[i].in_dim == in_dim &&
            cache[i].out_dim == out_dim && cache[i].ntok == ntok) { e = &cache[i]; break; }
    }
    if (!e) {
        lt_shape_cache ne = {};
        ne.in_dim = in_dim; ne.out_dim = out_dim; ne.ntok = ntok;
        if (cublasLtMatmulDescCreate(&ne.op, CUBLAS_COMPUTE_32F, CUDA_R_32F)) return 0;
        cublasOperation_t tA = CUBLAS_OP_T, tB = CUBLAS_OP_N;
        cublasLtMatmulMatrixScale_t mo = CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
        cublasLtMatmulDescSetAttribute(ne.op, CUBLASLT_MATMUL_DESC_TRANSA, &tA, sizeof(tA));
        cublasLtMatmulDescSetAttribute(ne.op, CUBLASLT_MATMUL_DESC_TRANSB, &tB, sizeof(tB));
        cublasLtMatmulDescSetAttribute(ne.op, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &mo, sizeof(mo));
        cublasLtMatmulDescSetAttribute(ne.op, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &mo, sizeof(mo));
        /* The heuristic must see a fully-populated desc (including scale
         * pointers) or it selects a non-MX algo an order of magnitude
         * slower; the pointers are re-set per call below. */
        cublasLtMatmulDescSetAttribute(ne.op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &w->scale, sizeof(w->scale));
        cublasLtMatmulDescSetAttribute(ne.op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &sx, sizeof(sx));
        cublasLtMatrixLayoutCreate(&ne.la, CUDA_R_8F_E4M3, in_dim, out_dim, in_dim);
        cublasLtMatrixLayoutCreate(&ne.lb, CUDA_R_8F_E4M3, in_dim, ntok, in_dim);
        cublasLtMatrixLayoutCreate(&ne.ld, CUDA_R_32F, out_dim, ntok, out_dim);
        cublasLtMatmulPreference_t pf; cublasLtMatmulPreferenceCreate(&pf);
        cublasLtMatmulPreferenceSetAttribute(pf, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wz, sizeof(wz));
        /* determinism: forbid split-K reduction algos (atomic/parallel
         * reduction order varies run-to-run and vs the decode GEMV path);
         * NONE-scheme algos accumulate in a fixed order. */
        {
            uint32_t red = CUBLASLT_REDUCTION_SCHEME_NONE;
            cublasLtMatmulPreferenceSetAttribute(pf, CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK, &red, sizeof(red));
        }
        int got = 0;
        cublasStatus_t hs = cublasLtMatmulAlgoGetHeuristic(g_cublaslt, ne.op, ne.la, ne.lb, ne.ld, ne.ld, pf, 1, &ne.h, &got);
        cublasLtMatmulPreferenceDestroy(pf);
        if (hs != CUBLAS_STATUS_SUCCESS || !got) {
            cublasLtMatrixLayoutDestroy(ne.la); cublasLtMatrixLayoutDestroy(ne.lb);
            cublasLtMatrixLayoutDestroy(ne.ld); cublasLtMatmulDescDestroy(ne.op);
            return 0;
        }
        ne.valid = 1;
        e = &cache[cache_next];
        cache_next = (cache_next + 1) & 15;
        if (e->valid) {
            cublasLtMatrixLayoutDestroy(e->la); cublasLtMatrixLayoutDestroy(e->lb);
            cublasLtMatrixLayoutDestroy(e->ld); cublasLtMatmulDescDestroy(e->op);
        }
        *e = ne;
    }
    cublasLtMatmulDescSetAttribute(e->op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &w->scale, sizeof(w->scale));
    cublasLtMatmulDescSetAttribute(e->op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &sx, sizeof(sx));
    int ok = 0;
    if (ws) {
        float al = 1.f, be = 0.f;
        cublasStatus_t st = cublasLtMatmul(g_cublaslt, e->op, &al, w->data, e->la, xq, e->lb, &be,
                                           out->ptr, e->ld, out->ptr, e->ld, &e->h.algo, ws, wz,
                                           cudaStreamPerThread);
        ok = (st == CUBLAS_STATUS_SUCCESS);
        if (!ok) fprintf(stderr, "pulsar: cuBLASLt MXFP8 matmul failed: status %d\n", (int)st);
    }
    return ok;
}


int pulsar_gpu_matmul_fp8_mx_tensor(pulsar_gpu_tensor *out, const void *model_map, uint64_t model_size,
        uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim, const pulsar_gpu_tensor *x, uint64_t n_tok) {
    return cuda_matmul_fp8_mx_tensor_labeled(out, model_map, model_size, weight_offset,
                                             in_dim, out_dim, x, n_tok, "fp8_mx");
}



/* Prefill attn-output "a" projection as n_groups block-scaled MXFP8xMXFP8 GEMMs.
 *
 * The projection is block-diagonal: group g's [rank, group_dim] weight slice only
 * sees activation rows (tok, g). cuBLASLt's VEC32_UE8M0 scale layout tiles rows in
 * 128-row blocks, so a batched/strided formulation can't share one swizzled scale
 * buffer across groups; instead the weight cache is sliced per group (exact when
 * rank % 128 == 0: both data columns and scale tiles are contiguous per group)
 * and the activations are quantized into per-group data + scale slabs. Each of
 * the n_groups (8/16) GEMMs writes straight into low[tok][g*rank + r] via ldd,
 * so no epilogue pass is needed. */
static int cuda_attention_output_a_mx_gemm(
        pulsar_gpu_tensor *low,
        const void *model_map,
        uint64_t model_size,
        uint64_t out_a_offset,
        uint64_t group_dim,
        uint64_t rank,
        uint32_t n_groups,
        const pulsar_gpu_tensor *heads,
        uint32_t n_tokens) {
    if (group_dim % 32 != 0 || rank % 128 != 0 || !cublaslt_ensure()) return 0;
    const uint64_t low_dim = (uint64_t)n_groups * rank;
    const uint64_t KB = group_dim / 32;
    const uint64_t weight_bytes = low_dim * KB * 33;
    if (out_a_offset > model_size || weight_bytes > model_size - out_a_offset) return 0;
    const fp8_mx_weight *w = cuda_fp8_mx_weight(model_map, out_a_offset, weight_bytes,
                                                group_dim, low_dim, "attn_out_a");
    if (!w) return 0;
    const int KBp = mx_rup((int)KB, 4);
    const size_t x_scale_slab = (size_t)mx_rup((int)n_tokens, 128) * KBp;
    const size_t w_scale_slab = ((size_t)rank / 128) * (size_t)KBp * 128;
    const size_t data_bytes = (size_t)n_tokens * n_groups * group_dim;
    const size_t scale_bytes = (size_t)n_groups * x_scale_slab;
    size_t wz = 32u << 20;
    size_t off_sx = (data_bytes + 255) & ~(size_t)255;
    size_t off_ws = (off_sx + scale_bytes + 255) & ~(size_t)255;
    char *scratch = (char *)cuda_tmp_alloc(off_ws + wz, "attn_out_a mx scratch");
    if (!scratch) return 0;
    __nv_fp8_e4m3 *xq = (__nv_fp8_e4m3 *)scratch;
    unsigned char *sx = (unsigned char *)(scratch + off_sx);
    void *ws = scratch + off_ws;
    cudaMemsetAsync(sx, 0, scale_bytes, 0);
    int warps = (int)n_tokens * (int)n_groups * (int)KB;
    mxfp8_quant_act_grouped_kernel<<<(warps * 32 + 255) / 256, 256>>>(
            (const float *)heads->ptr, (int)n_tokens, (int)n_groups,
            (int)group_dim, KBp, xq, sx, x_scale_slab);
    if (!cuda_ok(cudaGetLastError(), "attn_out_a act quant")) return 0;
    /* Same shape-keyed handle/algo cache as the main MXFP8 GEMM above (and
     * the same gotcha: the heuristic must see scale pointers on the desc or
     * it picks a non-MX algo). The per-group loop swaps only scale pointers,
     * which is already cache-shaped. */
    struct lt_group_cache {
        uint64_t group_dim, rank; uint32_t n_groups; int ntok; int valid;
        cublasLtMatmulDesc_t op;
        cublasLtMatrixLayout_t la, lb, ld;
        cublasLtMatmulHeuristicResult_t h;
    };
    static lt_group_cache cache[8];
    static int cache_next;
    lt_group_cache *e = NULL;
    for (int i = 0; i < 8; i++) {
        if (cache[i].valid && cache[i].group_dim == group_dim && cache[i].rank == rank &&
            cache[i].n_groups == n_groups && cache[i].ntok == (int)n_tokens) { e = &cache[i]; break; }
    }
    if (!e) {
        lt_group_cache ne = {};
        ne.group_dim = group_dim; ne.rank = rank; ne.n_groups = n_groups; ne.ntok = (int)n_tokens;
        if (cublasLtMatmulDescCreate(&ne.op, CUBLAS_COMPUTE_32F, CUDA_R_32F)) return 0;
        cublasOperation_t tA = CUBLAS_OP_T, tB = CUBLAS_OP_N;
        cublasLtMatmulMatrixScale_t mo = CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
        cublasLtMatmulDescSetAttribute(ne.op, CUBLASLT_MATMUL_DESC_TRANSA, &tA, sizeof(tA));
        cublasLtMatmulDescSetAttribute(ne.op, CUBLASLT_MATMUL_DESC_TRANSB, &tB, sizeof(tB));
        cublasLtMatmulDescSetAttribute(ne.op, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &mo, sizeof(mo));
        cublasLtMatmulDescSetAttribute(ne.op, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &mo, sizeof(mo));
        cublasLtMatmulDescSetAttribute(ne.op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &w->scale, sizeof(w->scale));
        cublasLtMatmulDescSetAttribute(ne.op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &sx, sizeof(sx));
        cublasLtMatrixLayoutCreate(&ne.la, CUDA_R_8F_E4M3, group_dim, rank, group_dim);
        cublasLtMatrixLayoutCreate(&ne.lb, CUDA_R_8F_E4M3, group_dim, n_tokens, group_dim);
        cublasLtMatrixLayoutCreate(&ne.ld, CUDA_R_32F, rank, n_tokens, low_dim);
        cublasLtMatmulPreference_t pf; cublasLtMatmulPreferenceCreate(&pf);
        cublasLtMatmulPreferenceSetAttribute(pf, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wz, sizeof(wz));
        /* determinism: forbid split-K reduction algos (atomic/parallel
         * reduction order varies run-to-run and vs the decode GEMV path);
         * NONE-scheme algos accumulate in a fixed order. */
        {
            uint32_t red = CUBLASLT_REDUCTION_SCHEME_NONE;
            cublasLtMatmulPreferenceSetAttribute(pf, CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK, &red, sizeof(red));
        }
        int got = 0;
        cublasStatus_t hs = cublasLtMatmulAlgoGetHeuristic(g_cublaslt, ne.op, ne.la, ne.lb, ne.ld, ne.ld, pf, 1, &ne.h, &got);
        cublasLtMatmulPreferenceDestroy(pf);
        if (hs != CUBLAS_STATUS_SUCCESS || !got) {
            cublasLtMatrixLayoutDestroy(ne.la); cublasLtMatrixLayoutDestroy(ne.lb);
            cublasLtMatrixLayoutDestroy(ne.ld); cublasLtMatmulDescDestroy(ne.op);
            return 0;
        }
        ne.valid = 1;
        e = &cache[cache_next];
        cache_next = (cache_next + 1) & 7;
        if (e->valid) {
            cublasLtMatrixLayoutDestroy(e->la); cublasLtMatrixLayoutDestroy(e->lb);
            cublasLtMatrixLayoutDestroy(e->ld); cublasLtMatmulDescDestroy(e->op);
        }
        *e = ne;
    }
    cublasLtMatmulDesc_t op = e->op;
    cublasLtMatrixLayout_t la = e->la, lb = e->lb, ld = e->ld;
    cublasLtMatmulHeuristicResult_t h = e->h;
    int ok = 0;
    if (ws) {
        ok = 1;
        for (uint32_t g = 0; g < n_groups && ok; g++) {
            const __nv_fp8_e4m3 *ag = w->data + (size_t)g * rank * group_dim;
            const unsigned char *as = w->scale + (size_t)g * w_scale_slab;
            const __nv_fp8_e4m3 *bg = xq + (size_t)g * n_tokens * group_dim;
            const unsigned char *bs = sx + (size_t)g * x_scale_slab;
            float *dg = (float *)low->ptr + (size_t)g * rank;
            cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &as, sizeof(as));
            cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &bs, sizeof(bs));
            float al = 1.f, be = 0.f;
            cublasStatus_t st = cublasLtMatmul(g_cublaslt, op, &al, ag, la, bg, lb, &be,
                                               dg, ld, dg, ld, &h.algo, ws, wz,
                                               cudaStreamPerThread);
            ok = (st == CUBLAS_STATUS_SUCCESS);
            if (!ok) fprintf(stderr, "pulsar: cuBLASLt attn_out_a MXFP8 matmul failed: status %d\n", (int)st);
        }
    }
    return ok;
}



/* Native FP8 decode (mmvq): read the MXFP8 weight 8-bit and dot with the f32
 * activation (single token). One warp per output row; lane L covers in-dim
 * positions {L, 32+L, ...}. No f16 expansion -> no decode bandwidth regression
 * vs the removed Q8_0 path. Used when MXFP8 tensor-cores don't apply (n_tok==1). */
__global__ static void mxfp8_mmvq_kernel(float *out, const unsigned char *w, const float *x,
                                         int in_dim, int out_dim) {
    int o = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    int lane = threadIdx.x & 31;
    if (o >= out_dim) return;
    int KB = in_dim / 32;
    const unsigned char *row = w + (size_t)o * KB * 33;
    float acc = 0.f;
    for (int kb = 0; kb < KB; kb++) {
        const unsigned char *blk = row + (size_t)kb * 33;
        float scale = exp2f((float)blk[0] - 127.f);
        __half wh = (__half)(*(const __nv_fp8_e4m3 *)&blk[1 + lane]);
        acc += __half2float(wh) * scale * x[kb * 32 + lane];
    }
    for (int s = 16; s > 0; s >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, s);
    if (lane == 0) out[o] = acc;
}


/* Decode mmvq over the DE-INTERLEAVED cached weight (contiguous E4M3 data[out,in]
 * + swizzled E8M0 scale), the same buffers the prefill tensor-core path builds via
 * cuda_fp8_mx_weight(). Reading contiguous fp8 lets each lane pull a uint32 (4 E4M3)
 * per step -> 128-wide coalesced loads vs the raw 33B-interleaved kernel's misaligned
 * 1-byte/thread reads. Numerically identical (same fp8 bytes, same raw E8M0 scale byte).
 * Requires in_dim % 128 == 0 (all MLA/shared/head dims qualify); else fall back to raw. */
__global__ static void mxfp8_mmvq_deint_kernel(float *out, const __nv_fp8_e4m3 *data,
                                               const unsigned char *scale, const float *x,
                                               int in_dim, int out_dim, int KBp) {
    int o = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    int lane = threadIdx.x & 31;
    if (o >= out_dim) return;
    const __nv_fp8_e4m3 *row = data + (size_t)o * in_dim;
    float acc = 0.f;
    for (int base = 0; base < in_dim; base += 128) {
        int k = base + lane * 4;                       /* this lane's 4 in-positions */
        uint32_t packed = *(const uint32_t *)(row + k);
        int kb = k >> 5;                               /* 32-elem block for these 4 */
        float sc = __int_as_float((uint32_t)scale[mx_sfoff(o, kb, KBp)] << 23);  /* 2^(e-127), no SFU */
        const __nv_fp8_e4m3 *q = (const __nv_fp8_e4m3 *)&packed;
        const float *xk = x + k;
        #pragma unroll
        for (int j = 0; j < 4; j++) acc += __half2float((__half)q[j]) * sc * xk[j];
    }
    for (int s = 16; s > 0; s >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, s);
    if (lane == 0) out[o] = acc;
}


/* Fused pair of the de-interleaved mmvq: two weights (out0,out1) sharing one
 * activation x and in_dim, computed in a single launch. Each warp owns one
 * global output row -- rows [0,out0_dim) go to weight0/out0, the rest to
 * weight1/out1. Bit-identical to launching mxfp8_mmvq_deint_kernel twice (same
 * per-row math); the win is one launch instead of two on the overhead-bound
 * decode path (q_a+kv from attn_norm, shared gate+up from ffn_norm). */
__global__ static void mxfp8_mmvq_deint_pair_kernel(
        float *out0, float *out1,
        const __nv_fp8_e4m3 *data0, const unsigned char *scale0, int out0_dim,
        const __nv_fp8_e4m3 *data1, const unsigned char *scale1, int out1_dim,
        const float *x, int in_dim, int KBp) {
    int warp = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    int lane = threadIdx.x & 31;
    const __nv_fp8_e4m3 *data;
    const unsigned char *scale;
    float *out;
    int o;
    if (warp < out0_dim) { o = warp;            data = data0; scale = scale0; out = out0; }
    else                 { o = warp - out0_dim; if (o >= out1_dim) return;
                           data = data1; scale = scale1; out = out1; }
    const __nv_fp8_e4m3 *row = data + (size_t)o * in_dim;
    float acc = 0.f;
    for (int base = 0; base < in_dim; base += 128) {
        int k = base + lane * 4;
        uint32_t packed = *(const uint32_t *)(row + k);
        int kb = k >> 5;
        float sc = __int_as_float((uint32_t)scale[mx_sfoff(o, kb, KBp)] << 23);
        const __nv_fp8_e4m3 *q = (const __nv_fp8_e4m3 *)&packed;
        const float *xk = x + k;
        #pragma unroll
        for (int j = 0; j < 4; j++) acc += __half2float((__half)q[j]) * sc * xk[j];
    }
    for (int s = 16; s > 0; s >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, s);
    if (lane == 0) out[o] = acc;
}


/* Small-batch (2..4 token) variant of the de-interleaved mmvq for the spec-decode
 * verify forward. One weight-row read serves all NT tokens (per-token accumulators),
 * vs the tensor-core tile path (latency-bound at 2-4 rows) or NT GEMV relaunches
 * (NT x weight traffic). Per-token multiply/accumulate order matches
 * mxfp8_mmvq_deint_kernel exactly, so each token's output is bit-identical to the
 * n=1 kernel run on that token alone. */
template <int NT>
__global__ static void mxfp8_mmvq_deint_nt_kernel(float *out, const __nv_fp8_e4m3 *data,
                                                  const unsigned char *scale, const float *x,
                                                  int in_dim, int out_dim, int KBp) {
    int o = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    int lane = threadIdx.x & 31;
    if (o >= out_dim) return;
    const __nv_fp8_e4m3 *row = data + (size_t)o * in_dim;
    float acc[NT];
    #pragma unroll
    for (int t = 0; t < NT; t++) acc[t] = 0.f;
    for (int base = 0; base < in_dim; base += 128) {
        int k = base + lane * 4;
        uint32_t packed = *(const uint32_t *)(row + k);
        int kb = k >> 5;
        float sc = __int_as_float((uint32_t)scale[mx_sfoff(o, kb, KBp)] << 23);
        const __nv_fp8_e4m3 *q = (const __nv_fp8_e4m3 *)&packed;
        const float *xk = x + k;
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            const float wj = __half2float((__half)q[j]) * sc;
            #pragma unroll
            for (int t = 0; t < NT; t++) acc[t] += wj * xk[(size_t)t * in_dim + j];
        }
    }
    #pragma unroll
    for (int t = 0; t < NT; t++) {
        float a = acc[t];
        for (int s = 16; s > 0; s >>= 1) a += __shfl_xor_sync(0xffffffffu, a, s);
        if (lane == 0) out[(size_t)t * out_dim + o] = a;
    }
}


/* PER-TENSOR routing: every offset registered at load (the MXFP8 workhorse
 * weights: attn_kv/q_a/q_b, attn_output_a/b, shared experts, output head)
 * takes the FP8 path; anything unregistered is rejected below. */
std::unordered_set<uint64_t> g_fp8_offsets;


void pulsar_gpu_register_fp8_weight(uint64_t weight_offset) { g_fp8_offsets.insert(weight_offset); }


void pulsar_gpu_register_fp8_lt_weight(uint64_t weight_offset) { g_mxfp8_lt_offsets.insert(weight_offset); }


/* Drop every process-global fp8 weight-cache entry. MUST run at backend
 * cleanup (pulsar_gpu_cleanup): pre-stored MXFP8_LT entries point straight into
 * the per-engine model arena that cleanup frees, and a subsequent engine open
 * in the same process typically mmaps the model at the SAME base address --
 * the cache's (host_base, offset, dims) guard then false-positives and serves
 * dangling pointers into freed memory (garbage/NaN activations, and an
 * illegal TMA access inside the cuBLASLt MXFP8 GEMM once a page is gone).
 * Converted (non-LT) entries own their device buffers; free them here so a
 * multi-engine process does not leak one conversion set per engine cycle. */
void cuda_fp8_weight_cache_clear(void) {
    for (auto &kv : g_fp8_mx_by_offset) {
        if (g_mxfp8_lt_offsets.count(kv.first)) continue;   /* arena-owned, freed with the arena */
        if (kv.second.data) (void)cudaFree((void *)kv.second.data);
        if (kv.second.scale) (void)cudaFree((void *)kv.second.scale);
    }
    g_fp8_mx_by_offset.clear();
    memset(g_fp8_fc_off, 0, sizeof(g_fp8_fc_off));
    memset(g_fp8_fc_ptr, 0, sizeof(g_fp8_fc_ptr));
    /* per-load registrations; the next engine open re-registers its own set */
    g_fp8_offsets.clear();
    g_mxfp8_lt_offsets.clear();
}



/* plan-34 phase-2 inc 2 — cuBLASLt ALGO-STABILITY. When armed (a batched
 * multiseq/mixed step), force the M-INDEPENDENT custom per-token GEMV kernels for
 * the whole batched row range [2..PULSAR_MSEQ_MAX=8] instead of switching to a
 * cuBLAS(Lt) tensor-core GEMM at n_tok>=5. cuBLAS(Lt) resolves an M-dependent
 * (ntok-keyed heuristic) algo, so a co-scheduled decode bank's logits would shift
 * with the batch width (measured: M=5/8 differ from M=2). The custom deint_nt /
 * f16_nt kernels compute each output row purely from THAT row's activations, so
 * they are byte-identical across M by construction. Armed once per step
 * (multiseq_step_begin) / cleared (step_end) — NEVER per token. Classic prefill
 * never arms it, so its large-M cuBLAS tensor-core path is unchanged, and the
 * decode-only lane (n_tok<=4, already custom) is bit-for-bit unchanged.
 *
 * plan-34 phase-2 inc 4 — GENERALIZED to a PREFIX ROW COUNT. In a fused mixed
 * step the row layout is [decode rows 0..n_dec) then one K-row prefill run
 * [n_dec..M). The decode prefix must stay M-independent (byte-identical to a
 * decode-only step of width n_dec — gate-4 neutrality); the prefill suffix takes
 * the fast tensor-core path (correctness, not byte-identity). g_mneutral_rows now
 * holds n_dec (0 = pure prefill / not armed; == n_tok = decode-only). Each dense
 * GEMM splits by recursing with the flag set to each range's PURE regime, so the
 * exact inc-2 (custom) and inc-3 (tensor-core) code paths are reused verbatim —
 * the decode prefix launch is LITERALLY the same nt<n_dec> a decode-only step
 * emits, and the prefill suffix is the same tensor-core GEMM a pure-prefill step
 * emits at that width. No kernel logic is duplicated. */
static int g_mneutral_rows = 0;
void pulsar_gpu_matmul_set_batch_mneutral(int n) { g_mneutral_rows = (n > 0) ? n : 0; }
/* Queried cross-TU by the MoE dispatch (pulsar_cuda_moe.cu): the number of leading
 * decode rows that must take the M-independent per-token expert path (the trailing
 * prefill rows take the grouped GEMM). 0 = not armed. Nonzero = armed (inc-2/3
 * read it as a boolean; inc-4 MoE two-pass reads the count to place the split). */
int pulsar_gpu_matmul_batch_mneutral(void) { return g_mneutral_rows; }

static int cuda_matmul_mxfp8_tensor_labeled(pulsar_gpu_tensor *out, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim, const pulsar_gpu_tensor *x, uint64_t n_tok, const char *label) {
    if (!out || !x || !model_map) return 0;
    /* inc 4 prefix-split: 0<n_dec<n_tok => mixed decode+prefill batch. Run the
     * decode prefix [0,n_dec) in the M-independent (decode) regime and the prefill
     * suffix [n_dec,n_tok) in the tensor-core (prefill) regime, by recursing with
     * the flag set to each range's pure value. Offsets are row-major (float rows). */
    {
        const uint64_t n_dec = (uint64_t)g_mneutral_rows;
        if (n_dec > 0 && n_dec < n_tok) {
            const uint64_t inb = in_dim * sizeof(float), outb = out_dim * sizeof(float);
            pulsar_gpu_tensor out_pre = { out->ptr, out->bytes, 0 };
            pulsar_gpu_tensor x_pre   = { x->ptr,   x->bytes,   0 };
            pulsar_gpu_tensor out_suf = { (char *)out->ptr + n_dec * outb,
                                       out->bytes - n_dec * outb, 0 };
            pulsar_gpu_tensor x_suf   = { (char *)x->ptr + n_dec * inb,
                                       x->bytes - n_dec * inb, 0 };
            const int saved = g_mneutral_rows;
            g_mneutral_rows = (int)n_dec;   /* decode prefix: n_dec == n_tok' => all custom */
            int r1 = cuda_matmul_mxfp8_tensor_labeled(&out_pre, model_map, model_size,
                    weight_offset, in_dim, out_dim, &x_pre, n_dec, label);
            g_mneutral_rows = 0;            /* prefill suffix: tensor-core */
            int r2 = cuda_matmul_mxfp8_tensor_labeled(&out_suf, model_map, model_size,
                    weight_offset, in_dim, out_dim, &x_suf, n_tok - n_dec, label);
            g_mneutral_rows = saved;
            return r1 && r2;
        }
    }
    if (g_fp8_offsets.count(weight_offset)) {
        const uint64_t fblocks = (in_dim + 31) / 32;
        const uint64_t fbytes = out_dim * fblocks * 33;
        if (weight_offset > model_size || fbytes > model_size - weight_offset ||
            x->bytes < n_tok * in_dim * sizeof(float) ||
            out->bytes < n_tok * out_dim * sizeof(float)) return 0;
        /* Small batches (spec-decode verify, n_tok 2..4): batched GEMV over the
         * de-interleaved weight. One weight-row read serves all tokens, vs the
         * tensor-core tile path which is latency-bound at these shapes (the
         * measured "CUTLASS launch storm" that made verify(3) cost ~2x a decode
         * token). Bit-identical per token to the n=1 deint mmvq, so verify logits
         * match the decode path's numerics. PULSAR_FP8_GEMV_MAX_N=1 restores the
         * tensor-core dispatch for all n_tok>1. */
        /* 2026-07-21: raising this default 4 -> 8 was TRIED and REVERTED. The
         * "bit-identical" claim above is GEMV-n vs GEMV-1 -- it does NOT extend to
         * the cuBLASLt dispatch this cap hands off to, which is what widening the
         * window actually swaps out. Measured at verify width 6: all 129280 logits
         * differ, max |d| 1.674, RMS 0.312 vs sigma 6.02, and 115/128 generated
         * tokens change. It also reaches real prefill -- the final chunk keeps the
         * exact remainder, so a 4102-token prompt at chunk 4096 ends on n_tok=6 and
         * its logits move (4100 -> n_tok=4 is byte-identical, control). The
         * prefill byte-exact gate pins chunk 4096 with depths 512/2048/4096/6144
         * and never lands a 5..8 remainder, so it does NOT cover this. Keep 4.
         * The width-6 verify win is real but belongs entirely to moe_gemv_cap
         * (see pulsar_cuda_moe.cu) -- these three caps cost ~28 ms/verify on top. */
        static const int gemv_max_n = []{ const char *e = getenv("DS4_FP8_GEMV_MAX_N");
                return e ? atoi(e) : 4; }();
        static const int nt_fp8_raw = getenv("DS4_FP8_MMVQ_RAW") != NULL;
        /* inc 2: raise the custom-nt cap to PULSAR_MSEQ_MAX (8) for a batched step so
         * n_tok 5..8 keep the M-independent kernel instead of cuBLASLt. Default cap
         * (gemv_max_n=4) is unchanged for classic prefill (never armed) and for the
         * decode-only lane (n_tok<=4), which take the identical cases 2/3/4 below. */
        const uint64_t nt_cap = (g_mneutral_rows > 0) ? 8u : (uint64_t)gemv_max_n;
        if (n_tok >= 2 && n_tok <= nt_cap && n_tok <= 8 &&
            in_dim % 128 == 0 && !nt_fp8_raw) {
            const fp8_mx_weight *bw = cuda_fp8_mx_weight(model_map, weight_offset, fbytes,
                                                         in_dim, out_dim, label);
            if (bw) {
                const int KBp = mx_rup((int)(in_dim / 32), 4);
                const unsigned wpb = 8;
                dim3 grid(((unsigned)out_dim + wpb - 1) / wpb);
                #define PULSAR_FP8_NT(N) mxfp8_mmvq_deint_nt_kernel<N><<<grid, wpb * 32>>>( \
                        (float *)out->ptr, bw->data, bw->scale, (const float *)x->ptr, \
                        (int)in_dim, (int)out_dim, KBp)
                switch (n_tok) {
                case 2: PULSAR_FP8_NT(2); break;
                case 3: PULSAR_FP8_NT(3); break;
                case 4: PULSAR_FP8_NT(4); break;
                case 5: PULSAR_FP8_NT(5); break;
                case 6: PULSAR_FP8_NT(6); break;
                case 7: PULSAR_FP8_NT(7); break;
                default: PULSAR_FP8_NT(8); break;   /* n_tok == 8 */
                }
                #undef PULSAR_FP8_NT
                return cuda_ok(cudaGetLastError(), "fp8_mx mmvq deint nt");
            }
        }
        /* Prefill (n_tok>1) uses the cuBLASLt MX tensor-core GEMM; decode falls
         * through to the per-token mmvq kernel below. PULSAR_FP8_NO_MXCORE forces
         * the mmvq path everywhere as an operational fallback. */
        if (n_tok > 1 && getenv("DS4_FP8_NO_MXCORE") == NULL &&
                cuda_matmul_fp8_mx_tensor_labeled(out, model_map, model_size,
                weight_offset, in_dim, out_dim, x, n_tok, label)) return 1;
        const unsigned wpb = 8;  /* output rows per block */
        dim3 grid(((unsigned)out_dim + wpb - 1) / wpb);
        /* Prefer the de-interleaved cached weight (contiguous E4M3 -> coalesced 128-wide
         * loads, vs the raw 33B-interleaved kernel's misaligned 1-byte/thread reads).
         * Bit-exact vs the raw kernel (verified: rel 0 across all workhorse shapes);
         * ~+8% decode. PULSAR_FP8_MMVQ_RAW forces the raw path as an operational fallback. */
        static const int fp8_mmvq_raw = getenv("DS4_FP8_MMVQ_RAW") != NULL;
        const fp8_mx_weight *w = (in_dim % 128 == 0 && !fp8_mmvq_raw)
                ? cuda_fp8_mx_weight(model_map, weight_offset, fbytes, in_dim, out_dim, label)
                : NULL;
        if (w) {
            const int KBp = mx_rup((int)(in_dim / 32), 4);
            for (uint64_t t = 0; t < n_tok; t++)
                mxfp8_mmvq_deint_kernel<<<grid, wpb * 32>>>((float *)out->ptr + t * out_dim,
                        w->data, w->scale, (const float *)x->ptr + t * in_dim,
                        (int)in_dim, (int)out_dim, KBp);
            return cuda_ok(cudaGetLastError(), "fp8_mx mmvq deint");
        }
        /* The raw kernel below reads 33B-INTERLEAVED blocks. A pre-stored
         * MXFP8_LT weight is already de-interleaved, so it must NEVER reach this
         * path — fail closed (rather than run the deint path's `w == NULL`
         * fallthrough on LT bytes, which would be garbage). This also means
         * PULSAR_FP8_MMVQ_RAW (which forces w == NULL above) is incompatible with an
         * MXFP8_LT gguf. */
        if (g_mxfp8_lt_offsets.count(weight_offset)) {
            fprintf(stderr, "pulsar: MXFP8_LT weight at offset %llu cannot use the raw "
                    "interleaved mmvq path (DS4_FP8_MMVQ_RAW must not be set with a "
                    "pre-stored MXFP8_LT gguf)\n", (unsigned long long)weight_offset);
            return 0;
        }
        const unsigned char *wfp8 = (const unsigned char *)cuda_model_range_ptr(
                model_map, weight_offset, fbytes, "fp8_mx_mmvq");
        if (!wfp8) return 0;
        for (uint64_t t = 0; t < n_tok; t++)
            mxfp8_mmvq_kernel<<<grid, wpb * 32>>>((float *)out->ptr + t * out_dim, wfp8,
                                                  (const float *)x->ptr + t * in_dim,
                                                  (int)in_dim, (int)out_dim);
        return cuda_ok(cudaGetLastError(), "fp8_mx mmvq");
    }
    fprintf(stderr,
            "pulsar: matmul %s at offset %llu is not a registered MXFP8 weight "
            "(legacy q8_0 weights are no longer supported)\n",
            label ? label : "mxfp8",
            (unsigned long long)weight_offset);
    return 0;
}



int pulsar_gpu_matmul_mxfp8_tensor(pulsar_gpu_tensor *out, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim, const pulsar_gpu_tensor *x, uint64_t n_tok) {
    return cuda_matmul_mxfp8_tensor_labeled(out, model_map, model_size, weight_offset,
                                           in_dim, out_dim, x, n_tok, "mxfp8");
}



int pulsar_gpu_matmul_mxfp8_pair_tensor(
        pulsar_gpu_tensor *out0,
        pulsar_gpu_tensor *out1,
        const void *model_map,
        uint64_t model_size,
        uint64_t weight0_offset,
        uint64_t weight1_offset,
        uint64_t in_dim,
        uint64_t out0_dim,
        uint64_t out1_dim,
        const pulsar_gpu_tensor *x,
        uint64_t n_tok) {
    if (!out0 || !out1 || !x || !model_map || in_dim == 0 || out0_dim == 0 || out1_dim == 0 || n_tok == 0) {
        return 0;
    }
    /* Decode fast path: both weights share x and in_dim, so fuse the two
     * de-interleaved mmvq launches into one. Only when both are registered
     * de-interleavable MXFP8 weights and the raw mmvq fallback is not forced;
     * otherwise defer to the per-weight path below. */
    static const int fp8_mmvq_raw = getenv("DS4_FP8_MMVQ_RAW") != NULL;
    static const int no_pair_fuse = getenv("DS4_CUDA_NO_MXFP8_PAIR_FUSE") != NULL;
    if (n_tok == 1 && in_dim % 128 == 0 && !fp8_mmvq_raw && !no_pair_fuse &&
        g_fp8_offsets.count(weight0_offset) && g_fp8_offsets.count(weight1_offset)) {
        const uint64_t fblocks = (in_dim + 31) / 32;
        const uint64_t fbytes0 = out0_dim * fblocks * 33;
        const uint64_t fbytes1 = out1_dim * fblocks * 33;
        const bool bounds_ok =
            weight0_offset <= model_size && fbytes0 <= model_size - weight0_offset &&
            weight1_offset <= model_size && fbytes1 <= model_size - weight1_offset &&
            x->bytes >= in_dim * sizeof(float) &&
            out0->bytes >= out0_dim * sizeof(float) &&
            out1->bytes >= out1_dim * sizeof(float);
        if (bounds_ok) {
            const fp8_mx_weight *w0 = cuda_fp8_mx_weight(model_map, weight0_offset, fbytes0,
                                                         in_dim, out0_dim, "mxfp8_pair0");
            const fp8_mx_weight *w1 = cuda_fp8_mx_weight(model_map, weight1_offset, fbytes1,
                                                         in_dim, out1_dim, "mxfp8_pair1");
            if (w0 && w1) {
                const int KBp = mx_rup((int)(in_dim / 32), 4);
                const unsigned wpb = 8;  /* output rows (warps) per 256-thread block */
                dim3 grid(((unsigned)(out0_dim + out1_dim) + wpb - 1) / wpb);
                mxfp8_mmvq_deint_pair_kernel<<<grid, wpb * 32>>>(
                        (float *)out0->ptr, (float *)out1->ptr,
                        w0->data, w0->scale, (int)out0_dim,
                        w1->data, w1->scale, (int)out1_dim,
                        (const float *)x->ptr, (int)in_dim, KBp);
                return cuda_ok(cudaGetLastError(), "fp8_mx mmvq deint pair");
            }
        }
    }
    return cuda_matmul_mxfp8_tensor_labeled(out0, model_map, model_size, weight0_offset,
                                           in_dim, out0_dim, x, n_tok, "mxfp8_pair0") &&
           cuda_matmul_mxfp8_tensor_labeled(out1, model_map, model_size, weight1_offset,
                                           in_dim, out1_dim, x, n_tok, "mxfp8_pair1");
}



int cuda_matmul_fp8_hc_expand_tensor_labeled(
        pulsar_gpu_tensor       *out_hc,
        pulsar_gpu_tensor       *block_out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const pulsar_gpu_tensor *x,
        const pulsar_gpu_tensor *block_add,
        const pulsar_gpu_tensor *residual_hc,
        const pulsar_gpu_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc,
        const char             *label) {
    if (!out_hc || !block_out || !x || !residual_hc || !split || !model_map ||
        in_dim == 0 || out_dim == 0 || n_embd == 0 || n_hc == 0 ||
        out_dim != (uint64_t)n_embd) {
        return 0;
    }
    if (!g_fp8_offsets.count(weight_offset)) return 0;
    const uint64_t wstride = 33u;
    const uint64_t blocks = (in_dim + 31) / 32;
    if (weight_offset > model_size || out_dim > UINT64_MAX / (blocks * wstride)) return 0;
    const uint64_t weight_bytes = out_dim * blocks * wstride;
    const uint64_t hc_bytes = (uint64_t)n_hc * n_embd * PULSAR_HC_ELT_SIZE;   /* residual_hc + out_hc are carriers */
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
    const char *wptr = cuda_model_range_ptr(model_map, weight_offset, weight_bytes, label ? label : "fp8_hc_expand");
    if (!wptr) return 0;

    /* Decode uses the de-interleaved cached weight (coalesced vectorized loads); raw
     * 33B path retained as PULSAR_FP8_MMVQ_RAW fallback. Same weight buffers as mmvq. */
    static const int fp8_raw = getenv("DS4_FP8_MMVQ_RAW") != NULL;
    const fp8_mx_weight *dw = (in_dim % 32 == 0 && !fp8_raw)
            ? cuda_fp8_mx_weight(model_map, weight_offset, weight_bytes, in_dim, out_dim,
                                 label ? label : "fp8_hc_expand")
            : NULL;
    const int KBp = mx_rup((int)(in_dim / 32), 4);
    const dim3 hg(((unsigned)out_dim + 31u) / 32u);
    const float *ba = block_add ? (const float *)block_add->ptr : (const float *)block_out->ptr;
    if (dw) {
        matmul_fp8mx_hc_expand_warp8_kernel<true><<<hg, 256>>>(
                (pulsar_hc_t *)out_hc->ptr, (float *)block_out->ptr, ba,
                (const pulsar_hc_t *)residual_hc->ptr, (const float *)split->ptr,
                (const unsigned char *)wptr, dw->data, dw->scale, KBp, (const float *)x->ptr,
                in_dim, out_dim, n_embd, n_hc, blocks, block_add ? 1 : 0);
    } else {
        matmul_fp8mx_hc_expand_warp8_kernel<false><<<hg, 256>>>(
                (pulsar_hc_t *)out_hc->ptr, (float *)block_out->ptr, ba,
                (const pulsar_hc_t *)residual_hc->ptr, (const float *)split->ptr,
                (const unsigned char *)wptr, (const __nv_fp8_e4m3 *)NULL, (const unsigned char *)NULL, 0,
                (const float *)x->ptr, in_dim, out_dim, n_embd, n_hc, blocks, block_add ? 1 : 0);
    }
    return cuda_ok(cudaGetLastError(), "matmul_fp8_hc_expand launch");
}



int pulsar_gpu_matmul_f16_tensor(pulsar_gpu_tensor *out, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim, const pulsar_gpu_tensor *x, uint64_t n_tok) {
    if (!out || !x || !model_map) return 0;
    /* inc 4 prefix-split (see the mxfp8 twin): decode prefix [0,n_dec) custom-nt,
     * prefill suffix [n_dec,n_tok) cuBLAS tensor-core, via pure-regime recursion. */
    {
        const uint64_t n_dec = (uint64_t)g_mneutral_rows;
        if (n_dec > 0 && n_dec < n_tok) {
            const uint64_t inb = in_dim * sizeof(float), outb = out_dim * sizeof(float);
            pulsar_gpu_tensor out_pre = { out->ptr, out->bytes, 0 };
            pulsar_gpu_tensor x_pre   = { x->ptr,   x->bytes,   0 };
            pulsar_gpu_tensor out_suf = { (char *)out->ptr + n_dec * outb,
                                       out->bytes - n_dec * outb, 0 };
            pulsar_gpu_tensor x_suf   = { (char *)x->ptr + n_dec * inb,
                                       x->bytes - n_dec * inb, 0 };
            const int saved = g_mneutral_rows;
            g_mneutral_rows = (int)n_dec;
            int r1 = pulsar_gpu_matmul_f16_tensor(&out_pre, model_map, model_size,
                    weight_offset, in_dim, out_dim, &x_pre, n_dec);
            g_mneutral_rows = 0;
            int r2 = pulsar_gpu_matmul_f16_tensor(&out_suf, model_map, model_size,
                    weight_offset, in_dim, out_dim, &x_suf, n_tok - n_dec);
            g_mneutral_rows = saved;
            return r1 && r2;
        }
    }
    if (weight_offset > model_size || out_dim > UINT64_MAX / in_dim) return 0;
    uint64_t weight_bytes = out_dim * in_dim * sizeof(uint16_t);
    if (weight_bytes > model_size - weight_offset) return 0;
    if (x->bytes < n_tok * in_dim * sizeof(float) ||
        out->bytes < n_tok * out_dim * sizeof(float)) return 0;
    const char *wptr = cuda_model_range_ptr(model_map, weight_offset, weight_bytes, "f16");
    if (!wptr) return 0;
    const __half *w = (const __half *)wptr;
    /* Small batches (spec-decode verify, n_tok 2..4): batched f16 GEMV. One
     * weight-row read serves all tokens and there is no f32->f16 activation
     * convert or tmp alloc; per-token output is bit-identical to the n=1
     * matmul_f16_kernel. PULSAR_F16_GEMV_MAX_N=1 restores the cuBLAS dispatch. */
    /* 2026-07-21: raising to 8 was TRIED and REVERTED, same as the mxfp8 twin
     * above. This one is the clearest case: the cuBLAS side converts activations
     * to __half for cublasGemmEx while the GEMV stays f32, so the two dispatches
     * were never going to agree. Measured non-bit-exact at widths 6 and 8. */
    static const int f16_gemv_max_n = []{ const char *e = getenv("DS4_F16_GEMV_MAX_N");
            return e ? atoi(e) : 4; }();
    /* inc 2: batched step forces the M-independent nt kernel across [2..8] (see the
     * mxfp8 twin). Default cap 4 for prefill / decode-only (identical cases 2/3/4). */
    const uint64_t f16_nt_cap = (g_mneutral_rows > 0) ? 8u : (uint64_t)f16_gemv_max_n;
    if (n_tok >= 2 && n_tok <= f16_nt_cap && n_tok <= 8) {
        dim3 g((unsigned)out_dim);
        #define PULSAR_F16_NT(N) matmul_f16_nt_kernel<N><<<g, 256>>>( \
                (float *)out->ptr, w, (const float *)x->ptr, in_dim, out_dim)
        switch (n_tok) {
        case 2: PULSAR_F16_NT(2); break;
        case 3: PULSAR_F16_NT(3); break;
        case 4: PULSAR_F16_NT(4); break;
        case 5: PULSAR_F16_NT(5); break;
        case 6: PULSAR_F16_NT(6); break;
        case 7: PULSAR_F16_NT(7); break;
        default: PULSAR_F16_NT(8); break;   /* n_tok == 8 */
        }
        #undef PULSAR_F16_NT
        return cuda_ok(cudaGetLastError(), "matmul_f16 nt launch");
    }
    if (g_cublas_ready && n_tok > 1) {
        const uint64_t xh_count = n_tok * in_dim;
        /* Armed activation cache: this f32 tensor feeds several F16 projections
         * per layer and the conversion is pure, so run it once (see
         * mxfp8_act_cache_t -- and note the cached copy must live outside the
         * shared cuda_tmp scratch, which unrelated kernels overwrite between
         * those calls). */
        mxfp8_act_cache_t *hc = NULL;
        if (g_act_cache.key_ptr == x->ptr && g_act_cache.key_ntok == n_tok &&
            g_act_cache.key_in_dim == in_dim &&
            mxfp8_act_cache_reserve((void **)&g_act_cache.xh, &g_act_cache.xh_cap,
                                    xh_count * sizeof(__half), "act f16")) {
            hc = &g_act_cache;
        }
        __half *xh = hc ? hc->xh
                        : (__half *)cuda_tmp_alloc(xh_count * sizeof(__half), "f16 gemm activations");
        if (!xh) return 0;
        if (!hc || !hc->valid_h) {
            f32_to_f16_kernel<<<(xh_count + 255) / 256, 256>>>(xh, (const float *)x->ptr, xh_count);
            if (!cuda_ok(cudaGetLastError(), "f16 activation convert launch")) return 0;
            if (hc) hc->valid_h = 1;
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
    matmul_f16_kernel<<<grid, 256>>>((float *)out->ptr, w, (const float *)x->ptr, in_dim, out_dim, n_tok);
    return cuda_ok(cudaGetLastError(), "matmul_f16 launch");
}



int pulsar_gpu_matmul_bf16_tensor(pulsar_gpu_tensor *out, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim, const pulsar_gpu_tensor *x, uint64_t n_tok) {
    if (!out || !x || !model_map) return 0;
    if (weight_offset > model_size || out_dim > UINT64_MAX / in_dim) return 0;
    const uint64_t weight_bytes = out_dim * in_dim * sizeof(uint16_t);
    if (weight_bytes > model_size - weight_offset) return 0;
    if (x->bytes < n_tok * in_dim * sizeof(float) ||
        out->bytes < n_tok * out_dim * sizeof(float)) return 0;
    const char *wptr = cuda_model_range_ptr(model_map, weight_offset, weight_bytes, "bf16");
    if (!wptr) return 0;
    const uint16_t *w = (const uint16_t *)wptr;
    if (g_cublas_ready && n_tok > 1) {
        const uint64_t xb_count = n_tok * in_dim;
        uint16_t *xb = (uint16_t *)cuda_tmp_alloc(xb_count * sizeof(uint16_t), "bf16 gemm activations");
        if (!xb) return 0;
        f32_to_bf16_kernel<<<(xb_count + 255) / 256, 256>>>(xb, (const float *)x->ptr, xb_count);
        if (!cuda_ok(cudaGetLastError(), "bf16 activation convert launch")) return 0;
        const float alpha = 1.0f;
        const float beta = 0.0f;
        cublasStatus_t st = cublasGemmEx(g_cublas,
                                         CUBLAS_OP_T, CUBLAS_OP_N,
                                         (int)out_dim, (int)n_tok, (int)in_dim,
                                         &alpha,
                                         w, CUDA_R_16BF, (int)in_dim,
                                         xb, CUDA_R_16BF, (int)in_dim,
                                         &beta,
                                         out->ptr, CUDA_R_32F, (int)out_dim,
                                         CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
        return cublas_ok(st, "bf16 matmul");
    }
    dim3 grid((unsigned)out_dim, (unsigned)n_tok, 1);
    matmul_bf16_kernel<<<grid, 256>>>((float *)out->ptr, w, (const float *)x->ptr, in_dim, out_dim, n_tok);
    return cuda_ok(cudaGetLastError(), "matmul_bf16 launch");
}



int pulsar_gpu_matmul_f16_pair_tensor(
        pulsar_gpu_tensor *out0,
        pulsar_gpu_tensor *out1,
        const void *model_map,
        uint64_t model_size,
        uint64_t weight0_offset,
        uint64_t weight1_offset,
        uint64_t in_dim,
        uint64_t out_dim,
        const pulsar_gpu_tensor *x,
        uint64_t n_tok) {
    if (!out0 || !out1 || !x || !model_map || in_dim == 0 || out_dim == 0 || n_tok == 0) {
        return 0;
    }
    /* Two separate coalesced matmuls (each fast via matmul_f16_kernel). */
    return pulsar_gpu_matmul_f16_tensor(out0, model_map, model_size, weight0_offset,
                                       in_dim, out_dim, x, n_tok) &&
           pulsar_gpu_matmul_f16_tensor(out1, model_map, model_size, weight1_offset,
                                       in_dim, out_dim, x, n_tok);
}



int pulsar_gpu_matmul_f32_tensor(pulsar_gpu_tensor *out, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim, const pulsar_gpu_tensor *x, uint64_t n_tok) {
    if (!out || !x || !model_map || in_dim == 0 || out_dim == 0 || n_tok == 0) return 0;
    if (weight_offset > model_size || out_dim > UINT64_MAX / in_dim) return 0;
    uint64_t weight_elems = out_dim * in_dim;
    if (weight_elems > UINT64_MAX / sizeof(float)) return 0;
    uint64_t weight_bytes = weight_elems * sizeof(float);
    if (weight_bytes > model_size - weight_offset) return 0;
    if (x->bytes < n_tok * in_dim * sizeof(float) ||
        out->bytes < n_tok * out_dim * sizeof(float)) return 0;
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



/* Decode grouped "a" projection: prefer the de-interleaved cached weight (vectorized
 * coalesced loads) over the raw 33B kernel. Bit-exact; PULSAR_FP8_MMVQ_RAW forces raw. */
static int launch_grouped_fp8mx_a(float *low, const void *model_map, uint64_t out_a_offset,
        uint64_t out_a_bytes, const unsigned char *out_a, uint64_t group_dim, uint64_t rank,
        uint32_t n_groups, uint32_t n_tokens, uint64_t blocks_a, uint64_t low_dim,
        const float *heads, const char *label) {
    const dim3 grid_a(((unsigned)low_dim + 31u) / 32u, (unsigned)n_tokens, 1);
    static const int fp8_raw = getenv("DS4_FP8_MMVQ_RAW") != NULL;
    const fp8_mx_weight *dw = (group_dim % 32 == 0 && !fp8_raw)
            ? cuda_fp8_mx_weight(model_map, out_a_offset, out_a_bytes, group_dim, low_dim, label) : NULL;
    const int KBp = mx_rup((int)(group_dim / 32), 4);
    /* Small verify batches: one nt launch, weight blocks L1-shared across tokens;
     * per-token bit-identical to the n=1 DEINT kernel below. */
    if (dw && n_tokens >= 2u && n_tokens <= 4u &&
        rank % PULSAR_FP8MX_ROWS == 0 && low_dim % PULSAR_FP8MX_ROWS == 0) {
        const dim3 g(((unsigned)low_dim + 31u) / 32u);
        switch (n_tokens) {
        case 2: grouped_fp8mx_a_nt_kernel<2><<<g, 256>>>(low, dw->data, dw->scale, KBp,
                heads, group_dim, rank, n_groups, blocks_a); break;
        case 3: grouped_fp8mx_a_nt_kernel<3><<<g, 256>>>(low, dw->data, dw->scale, KBp,
                heads, group_dim, rank, n_groups, blocks_a); break;
        default: grouped_fp8mx_a_nt_kernel<4><<<g, 256>>>(low, dw->data, dw->scale, KBp,
                heads, group_dim, rank, n_groups, blocks_a); break;
        }
        return cuda_ok(cudaGetLastError(), "attention_output_a nt launch");
    }
    if (dw) {
        grouped_fp8mx_a_warp8_kernel<true><<<grid_a, 256>>>(low, out_a, dw->data, dw->scale, KBp,
                heads, group_dim, rank, n_groups, n_tokens, blocks_a);
    } else {
        grouped_fp8mx_a_warp8_kernel<false><<<grid_a, 256>>>(low, out_a,
                (const __nv_fp8_e4m3 *)NULL, (const unsigned char *)NULL, 0,
                heads, group_dim, rank, n_groups, n_tokens, blocks_a);
    }
    return cuda_ok(cudaGetLastError(), "attention_output_a launch");
}


int pulsar_gpu_attention_output_batch_tensor(
        pulsar_gpu_tensor       *out,
        pulsar_gpu_tensor       *low,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                out_a_offset,
        uint64_t                out_b_offset,
        uint64_t                group_dim,
        uint64_t                rank,
        uint32_t                n_groups,
        uint64_t                out_dim,
        const pulsar_gpu_tensor *heads,
        uint32_t                n_tokens) {
    if (!out || !low || !heads || !model_map ||
        group_dim == 0 || rank == 0 || n_groups == 0 || out_dim == 0 || n_tokens == 0) {
        return 0;
    }
    /* inc 4 prefix-split: recurse the whole attn-output stage (a-proj + b-proj)
     * over the decode prefix [0,n_dec) in the M-independent regime and the prefill
     * suffix [n_dec,n_tokens) in the tensor-core regime. Both the warp8/nt 'a' path
     * and the mxfp8 'b' path then run their PURE code for each range (the 'b' proj
     * recurses no further: its sub-range already has n_dec in {n_tok',0}). */
    {
        const uint64_t n_dec = (uint64_t)g_mneutral_rows;
        if (n_dec > 0 && n_dec < (uint64_t)n_tokens) {
            const uint64_t low_dim = (uint64_t)n_groups * rank;
            const uint64_t headb = (uint64_t)n_groups * group_dim * sizeof(float);
            const uint64_t lowb  = low_dim * sizeof(float);
            const uint64_t outb  = out_dim * sizeof(float);
            pulsar_gpu_tensor out_pre = { out->ptr, out->bytes, 0 };
            pulsar_gpu_tensor low_pre = { low->ptr, low->bytes, 0 };
            pulsar_gpu_tensor hd_pre  = { heads->ptr, heads->bytes, 0 };
            pulsar_gpu_tensor out_suf = { (char *)out->ptr + n_dec * outb, out->bytes - n_dec * outb, 0 };
            pulsar_gpu_tensor low_suf = { (char *)low->ptr + n_dec * lowb, low->bytes - n_dec * lowb, 0 };
            pulsar_gpu_tensor hd_suf  = { (char *)heads->ptr + n_dec * headb, heads->bytes - n_dec * headb, 0 };
            const int saved = g_mneutral_rows;
            g_mneutral_rows = (int)n_dec;
            int r1 = pulsar_gpu_attention_output_batch_tensor(&out_pre, &low_pre, model_map,
                    model_size, out_a_offset, out_b_offset, group_dim, rank, n_groups,
                    out_dim, &hd_pre, (uint32_t)n_dec);
            g_mneutral_rows = 0;
            int r2 = pulsar_gpu_attention_output_batch_tensor(&out_suf, &low_suf, model_map,
                    model_size, out_a_offset, out_b_offset, group_dim, rank, n_groups,
                    out_dim, &hd_suf, n_tokens - (uint32_t)n_dec);
            g_mneutral_rows = saved;
            return r1 && r2;
        }
    }
    if (!g_fp8_offsets.count(out_a_offset) || !g_fp8_offsets.count(out_b_offset)) return 0;
    const uint64_t low_dim = (uint64_t)n_groups * rank;
    const uint64_t blocks_a = (group_dim + 31) / 32;
    const uint64_t blocks_b = (low_dim + 31) / 32;
    const uint64_t out_a_bytes = (uint64_t)n_groups * rank * blocks_a * 33u;
    const uint64_t out_b_bytes = out_dim * blocks_b * 33u;
    if (out_a_offset > model_size || out_b_offset > model_size ||
        out_a_bytes > model_size - out_a_offset ||
        out_b_bytes > model_size - out_b_offset ||
        heads->bytes < (uint64_t)n_tokens * n_groups * group_dim * sizeof(float) ||
        low->bytes < (uint64_t)n_tokens * low_dim * sizeof(float) ||
        out->bytes < (uint64_t)n_tokens * out_dim * sizeof(float)) {
        return 0;
    }

    /* "a" projection: prefill takes the block-scaled MXFP8xMXFP8 tensor-core
     * GEMMs; decode and small verify batches (n_tokens<=4) take the
     * register-blocked GEMV path (launch_grouped_fp8mx_a dispatches the nt
     * variant at 2..4 -- one launch vs 8 per-group GEMMs, bit-identical per
     * token to decode's kernel). PULSAR_FP8_GEMV_MAX_N=1 restores the tensor-core
     * dispatch for all n_tokens>1, same as the dense-matmul gate. */
    /* 2026-07-21: raising to 8 was TRIED and REVERTED (see the dense-matmul gate
     * for the measurement). Shares PULSAR_FP8_GEMV_MAX_N with that gate, so the two
     * defaults must move together. */
    static const int a_gemv_max_n = []{ const char *e = getenv("DS4_FP8_GEMV_MAX_N");
            return e ? atoi(e) : 4; }();
    int a_done = 0;
    /* plan-34 inc 2: a batched multiseq/mixed step must NOT take the M-dependent
     * tensor-core (cuBLASLt) 'a' GEMM -- fall to launch_grouped_fp8mx_a, whose
     * warp8/nt kernels are per-token M-independent (bit-identical to the n=1
     * DEINT kernel), so a co-scheduled decode bank's attn-output row is invariant
     * to the batch width. Classic prefill (not armed) keeps the tensor-core path. */
    if (n_tokens > 1 && (int)n_tokens > a_gemv_max_n && g_mneutral_rows == 0 &&
        getenv("DS4_FP8_NO_MXCORE") == NULL) {
        a_done = cuda_attention_output_a_mx_gemm(low, model_map, model_size, out_a_offset,
                                                 group_dim, rank, n_groups, heads, n_tokens);
    }
    if (!a_done) {
        const unsigned char *out_a = reinterpret_cast<const unsigned char *>(
                cuda_model_range_ptr(model_map, out_a_offset, out_a_bytes, "attn_out_a"));
        if (!out_a) return 0;
        if (!launch_grouped_fp8mx_a((float *)low->ptr, model_map, out_a_offset, out_a_bytes, out_a,
                                    group_dim, rank, n_groups, n_tokens, blocks_a, low_dim,
                                    (const float *)heads->ptr, "attn_out_a")) return 0;
    }

    return cuda_matmul_mxfp8_tensor_labeled(out,
                                           model_map,
                                           model_size,
                                           out_b_offset,
                                           low_dim,
                                           out_dim,
                                           low,
                                           n_tokens,
                                           "attn_output_b");
}



int pulsar_gpu_attention_output_low_tensor(
        pulsar_gpu_tensor       *low,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                out_a_offset,
        uint64_t                group_dim,
        uint64_t                rank,
        uint32_t                n_groups,
        const pulsar_gpu_tensor *heads) {
    if (!low || !heads || !model_map || group_dim == 0 || rank == 0 || n_groups == 0) {
        return 0;
    }
    if (!g_fp8_offsets.count(out_a_offset)) return 0;
    const uint64_t low_dim = (uint64_t)n_groups * rank;
    const uint64_t blocks_a = (group_dim + 31) / 32;
    const uint64_t out_a_bytes = (uint64_t)n_groups * rank * blocks_a * 33u;
    if (out_a_offset > model_size ||
        out_a_bytes > model_size - out_a_offset ||
        heads->bytes < (uint64_t)n_groups * group_dim * sizeof(float) ||
        low->bytes < low_dim * sizeof(float)) {
        return 0;
    }
    const unsigned char *out_a = reinterpret_cast<const unsigned char *>(
            cuda_model_range_ptr(model_map, out_a_offset, out_a_bytes, "attn_out_a"));
    if (!out_a) return 0;

    return launch_grouped_fp8mx_a((float *)low->ptr, model_map, out_a_offset, out_a_bytes, out_a,
                                  group_dim, rank, n_groups, 1, blocks_a, low_dim,
                                  (const float *)heads->ptr, "attn_out_a");
}

