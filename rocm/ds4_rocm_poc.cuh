/* Decode-matvec bandwidth POC (DS4_ROCM poc bench).
 *
 * Question: the production decode matvec (matmul_q8_0_preq_rows_w32, one
 * warp per row) streams weights at ~50 GB/s effective on gfx1151 against a
 * ~220 GB/s ceiling.  Is that a kernel-structure artifact or the hardware
 * wall for this access pattern?  Times variants at the production shape
 * (in=2048, out=2048, Q8_0 34-byte blocks), cycling 24 weight copies (102 MiB) so
 * no cache level can hold the working set between iterations:
 *
 *   A) baseline preq kernel (1 warp/row, unaligned scalar block loads)
 *   B) pure-streaming probe (aligned uint4 row loads, no matvec math):
 *      the access pattern's upper bound
 *   C) LDS-staged matvec (cooperative aligned row load into shared, then
 *      the identical per-lane dp4a chain; bit-identical to A)
 *   D) split-row (2 warps per row, LDS combine; accumulation order differs
 *      from A, reported but not gated)
 *   E) fat CTA (rows_per_block=8; same warp count as A)
 *
 * Entry: ds4_gpu_rocm_matvec_poc() prints a GB/s table.  Bench-only; no
 * production code path references it.
 */

#include <stdio.h>
#include <time.h>
static double now_sec_host(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

#define POC_IN_DIM 2048u
#define POC_OUT_DIM 2048u
#define POC_BLOCKS (POC_IN_DIM / 32u)
#define POC_ROW_BYTES (POC_BLOCKS * 34u)
#define POC_WEIGHT_BYTES ((uint64_t)POC_OUT_DIM * POC_ROW_BYTES)
#define POC_WCOPIES 24u
#define POC_ITERS 500u

__global__ static void poc_stream_rows_kernel(
        int32_t *out,
        const unsigned char *w) {
    const uint64_t row = blockIdx.x;
    const uint32_t lane = threadIdx.x & 31u;
    const uint4 *wr = (const uint4 *)(w + row * POC_ROW_BYTES);
    int32_t acc = 0;
    /* 2176 B per row = 68 B per lane: 4 x uint4 + 1 x uint, all aligned. */
    #pragma unroll
    for (uint32_t i = 0; i < 4u; i++) {
        const uint4 v = wr[lane + i * 32u];
        acc += (int32_t)(v.x ^ v.y ^ v.z ^ v.w);
    }
    const uint *tail = (const uint *)(w + row * POC_ROW_BYTES + 4u * 32u * 16u);
    acc += (int32_t)tail[lane];
    acc = warp_sum_f32((float)acc) != 0.0f ? 1 : 0;
    if (lane == 0u) out[row] = acc;
}

__global__ static void poc_lds_matvec_kernel(
        float *out,
        const unsigned char *w,
        const int8_t *xq,
        const float *xscale) {
    __shared__ unsigned char srow[POC_ROW_BYTES];
    const uint64_t row = blockIdx.x;
    const uint32_t lane = threadIdx.x & 31u;
    const uint4 *wr = (const uint4 *)(w + row * POC_ROW_BYTES);
    uint4 *sr = (uint4 *)srow;
    #pragma unroll
    for (uint32_t i = 0; i < 4u; i++) sr[lane + i * 32u] = wr[lane + i * 32u];
    ((uint *)srow)[4u * 32u + lane] =
        ((const uint *)(w + row * POC_ROW_BYTES))[4u * 32u + lane];
    __syncwarp();
    float acc = 0.0f;
    #pragma unroll
    for (uint32_t it = 0; it < POC_BLOCKS / 32u; it++) {
        const uint32_t b = lane + it * 32u;
        const __half *scale_h = (const __half *)(srow + b * 34u);
        const int8_t *qs = (const int8_t *)(srow + b * 34u + 2u);
        const int dot = dot_i8x32_dp4a(qs, xq + b * 32u);
        acc += __half2float(*scale_h) * xscale[b] * (float)dot;
    }
    acc = warp_sum_f32(acc);
    if (lane == 0u) out[row] = acc;
}

__global__ static void poc_splitrow_matvec_kernel(
        float *out,
        const unsigned char *w,
        const int8_t *xq,
        const float *xscale) {
    __shared__ float spart[2];
    const uint64_t row = blockIdx.x;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t half = threadIdx.x >> 5u;
    const unsigned char *wr = w + row * POC_ROW_BYTES;
    float acc = 0.0f;
    /* warp 0: blocks 0..31; warp 1: blocks 32..63 (one block per lane). */
    const uint32_t b = half * 32u + lane;
    const __half *scale_h = (const __half *)(wr + b * 34u);
    const int8_t *qs = (const int8_t *)(wr + b * 34u + 2u);
    const int dot = dot_i8x32_dp4a(qs, xq + b * 32u);
    acc += __half2float(*scale_h) * xscale[b] * (float)dot;
    acc = warp_sum_f32(acc);
    if (lane == 0u) spart[half] = acc;
    __syncthreads();
    if (threadIdx.x == 0u) out[row] = spart[0] + spart[1];
}

extern "C" int ds4_gpu_rocm_matvec_poc(void);
extern "C" int ds4_gpu_rocm_matvec_poc(void) {
    unsigned char *w_dev = NULL;
    int8_t *xq_dev = NULL;
    float *xscale_dev = NULL;
    float *out_dev = NULL;
    int32_t *outi_dev = NULL;
    unsigned char *w_host = NULL;
    const uint64_t wtotal = POC_WEIGHT_BYTES * POC_WCOPIES;
    if (cudaMalloc(&w_dev, wtotal) != cudaSuccess ||
        cudaMalloc(&xq_dev, POC_IN_DIM) != cudaSuccess ||
        cudaMalloc(&xscale_dev, POC_BLOCKS * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&out_dev, POC_OUT_DIM * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&outi_dev, POC_OUT_DIM * sizeof(int32_t)) != cudaSuccess) {
        fprintf(stderr, "poc: alloc failed\n");
        return 1;
    }
    w_host = (unsigned char *)malloc((size_t)POC_WEIGHT_BYTES);
    if (!w_host) return 1;
    uint32_t rng = 12345u;
    for (uint64_t i = 0; i < POC_WEIGHT_BYTES; i += 34u) {
        rng = rng * 1664525u + 1013904223u;
        const uint16_t sc = (uint16_t)(0x2c00u + (rng & 0x3ffu));
        w_host[i] = (unsigned char)(sc & 0xffu);
        w_host[i + 1u] = (unsigned char)(sc >> 8);
        for (uint32_t j = 2u; j < 34u; j++) {
            rng = rng * 1664525u + 1013904223u;
            w_host[i + j] = (unsigned char)(rng >> 16);
        }
    }
    int8_t xq_host[POC_IN_DIM];
    float xs_host[POC_BLOCKS];
    for (uint32_t i = 0; i < POC_IN_DIM; i++) {
        rng = rng * 1664525u + 1013904223u;
        xq_host[i] = (int8_t)(rng >> 24);
    }
    for (uint32_t b = 0; b < POC_BLOCKS; b++) xs_host[b] = 0.001f + 0.0001f * (float)b;
    for (uint32_t c = 0; c < POC_WCOPIES; c++) {
        if (cudaMemcpy(w_dev + c * POC_WEIGHT_BYTES, w_host, POC_WEIGHT_BYTES,
                       cudaMemcpyHostToDevice) != cudaSuccess) return 1;
    }
    free(w_host);
    if (cudaMemcpy(xq_dev, xq_host, POC_IN_DIM, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(xscale_dev, xs_host, POC_BLOCKS * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess) return 1;

    struct variant { const char *name; uint32_t id; };
    const variant variants[] = {
        {"A baseline preq (1 warp/row)", 0u},
        {"B pure-stream probe (upper bound)", 1u},
        {"C LDS-staged matvec", 2u},
        {"D split-row 2 warps/row", 3u},
        {"E fat CTA rpb=8", 4u},
    };
    printf("poc: q8_0 decode matvec in=%u out=%u (%.2f MiB weights x%u copies, %u iters)\n",
           POC_IN_DIM, POC_OUT_DIM, (double)POC_WEIGHT_BYTES / 1048576.0,
           POC_WCOPIES, POC_ITERS);
    float ref[POC_OUT_DIM];
    int have_ref = 0;
    for (uint32_t v = 0; v < sizeof(variants) / sizeof(variants[0]); v++) {
        hipEvent_t e0, e1;
        (void)hipEventCreate(&e0);
        (void)hipEventCreate(&e1);
        /* warmup */
        for (uint32_t i = 0; i < 20u; i++) {
            const unsigned char *wc = w_dev + (i % POC_WCOPIES) * POC_WEIGHT_BYTES;
            switch (variants[v].id) {
            case 0u: matmul_q8_0_preq_rows_w32_kernel<<<POC_OUT_DIM, 32>>>(
                    out_dev, wc, xq_dev, xscale_dev, POC_IN_DIM, POC_OUT_DIM,
                    POC_BLOCKS, 1u, 1); break;
            case 1u: poc_stream_rows_kernel<<<POC_OUT_DIM, 32>>>(outi_dev, wc); break;
            case 2u: poc_lds_matvec_kernel<<<POC_OUT_DIM, 32>>>(
                    out_dev, wc, xq_dev, xscale_dev); break;
            case 3u: poc_splitrow_matvec_kernel<<<POC_OUT_DIM, 64>>>(
                    out_dev, wc, xq_dev, xscale_dev); break;
            default: matmul_q8_0_preq_rows_w32_kernel<<<POC_OUT_DIM / 8u, 256>>>(
                    out_dev, wc, xq_dev, xscale_dev, POC_IN_DIM, POC_OUT_DIM,
                    POC_BLOCKS, 8u, 1); break;
            }
        }
        (void)hipDeviceSynchronize();
        (void)hipEventRecord(e0, 0);
        for (uint32_t i = 0; i < POC_ITERS; i++) {
            const unsigned char *wc = w_dev + (i % POC_WCOPIES) * POC_WEIGHT_BYTES;
            switch (variants[v].id) {
            case 0u: matmul_q8_0_preq_rows_w32_kernel<<<POC_OUT_DIM, 32>>>(
                    out_dev, wc, xq_dev, xscale_dev, POC_IN_DIM, POC_OUT_DIM,
                    POC_BLOCKS, 1u, 1); break;
            case 1u: poc_stream_rows_kernel<<<POC_OUT_DIM, 32>>>(outi_dev, wc); break;
            case 2u: poc_lds_matvec_kernel<<<POC_OUT_DIM, 32>>>(
                    out_dev, wc, xq_dev, xscale_dev); break;
            case 3u: poc_splitrow_matvec_kernel<<<POC_OUT_DIM, 64>>>(
                    out_dev, wc, xq_dev, xscale_dev); break;
            default: matmul_q8_0_preq_rows_w32_kernel<<<POC_OUT_DIM / 8u, 256>>>(
                    out_dev, wc, xq_dev, xscale_dev, POC_IN_DIM, POC_OUT_DIM,
                    POC_BLOCKS, 8u, 1); break;
            }
        }
        (void)hipEventRecord(e1, 0);
        (void)hipEventSynchronize(e1);
        float ms = 0.0f;
        (void)hipEventElapsedTime(&ms, e0, e1);
        const double ms_iter = (double)ms / POC_ITERS;
        const double gbs = (double)POC_WEIGHT_BYTES / (ms_iter * 1.0e6);
        printf("  %-34s %8.1f us/iter  %7.1f GB/s\n", variants[v].name,
               ms_iter * 1000.0, gbs);
        if (variants[v].id != 1u) {
            float got[POC_OUT_DIM];
            if (cudaMemcpy(got, out_dev, sizeof(got), cudaMemcpyDeviceToHost) != cudaSuccess)
                return 1;
            if (!have_ref) {
                memcpy(ref, got, sizeof(ref));
                have_ref = 1;
            } else {
                double maxdiff = 0.0;
                for (uint32_t r = 0; r < POC_OUT_DIM; r++) {
                    const double d = fabs((double)got[r] - (double)ref[r]);
                    if (d > maxdiff) maxdiff = d;
                }
                printf("      vs A: max|diff| = %.3g %s\n", maxdiff,
                       maxdiff == 0.0 ? "(bit-identical)" : "(order differs)");
            }
        }
        (void)hipEventDestroy(e0);
        (void)hipEventDestroy(e1);
    }
    (void)cudaFree(w_dev);
    (void)cudaFree(xq_dev);
    (void)cudaFree(xscale_dev);
    (void)cudaFree(out_dev);
    (void)cudaFree(outi_dev);
    return 0;
}

/* ---- MXFP4 MoE decode gate/up POC -------------------------------------
 * Production shape: 1 token x 6 expert slots, expert_mid=2048, in=2048
 * (grid 64x8 blocks of 256 threads).  F is the production kernel launched
 * directly; G carries two rows per warp (identical per-row accumulation
 * order, so bit-identical, with two independent dot chains for ILP); H
 * swaps the four dword activation loads per half-block for one aligned
 * b128 load from a pre-staged interleaved layout (same dp4a operand
 * order, bit-identical). */

#define POC_MOE_EXPERTS 6u
#define POC_MOE_MID 2048u
#define POC_MOE_XQ_BLOCKS 8u
#define POC_MOE_ROW_BYTES (POC_MOE_XQ_BLOCKS * 8u * 17u)
#define POC_MOE_EXPERT_BYTES ((uint64_t)POC_MOE_MID * POC_MOE_ROW_BYTES)
#define POC_MOE_WCOPIES 24u

template <uint32_t RW>
__global__ static void poc_moe_gateup_rw_kernel(
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const int32_t *selected,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        const float *weights,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        float clamp) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t wave = threadIdx.x >> 5u;
    const uint32_t block_lane = lane >> 1u;
    const uint32_t half = lane & 1u;
    const uint32_t first_row = blockIdx.x * (RW * 8u) + wave * RW;
    const uint32_t pair = blockIdx.y;
    const uint32_t tok = pair / n_expert;
    const uint32_t slot = pair - tok * n_expert;
    const int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) return;
    const uint32_t expert = (uint32_t)expert_i;
    const cuda_block_q8_K *xqb = xq + (uint64_t)tok * xq_blocks;
    float gate[RW], up[RW];
    #pragma unroll
    for (uint32_t rr = 0; rr < RW; rr++) { gate[rr] = 0.0f; up[rr] = 0.0f; }
    const uint32_t mxfp4_blocks = xq_blocks * 8u;
    for (uint32_t mb = block_lane; mb < mxfp4_blocks; mb += 16u) {
        const cuda_block_q8_K *yb = xqb + (mb >> 3u);
        #pragma unroll
        for (uint32_t rr = 0; rr < RW; rr++) {
            const uint32_t row = first_row + rr;
            if (row >= expert_mid_dim) continue;
            const cuda_block_mxfp4 *gate_blocks =
                (const cuda_block_mxfp4 *)(gate_base +
                    (uint64_t)expert * gate_expert_bytes +
                    (uint64_t)row * gate_row_bytes);
            const cuda_block_mxfp4 *up_blocks =
                (const cuda_block_mxfp4 *)(up_base +
                    (uint64_t)expert * gate_expert_bytes +
                    (uint64_t)row * gate_row_bytes);
            const uint32_t subblock = mb & 7u;
            gate[rr] += dev_dot_mxfp4_q8_K_half_block(
                gate_blocks + mb, yb, subblock, half);
            up[rr] += dev_dot_mxfp4_q8_K_half_block(
                up_blocks + mb, yb, subblock, half);
        }
    }
    #pragma unroll
    for (uint32_t rr = 0; rr < RW; rr++) {
        const uint32_t row = first_row + rr;
        if (row >= expert_mid_dim) continue;
        gate[rr] = warp_sum_f32(gate[rr]);
        up[rr] = warp_sum_f32(up[rr]);
        if (lane == 0u) {
            float g = gate[rr];
            float u = up[rr];
            if (clamp > 1.0e-6f) {
                if (g > clamp) g = clamp;
                if (u > clamp) u = clamp;
                if (u < -clamp) u = -clamp;
            }
            mid_out[(uint64_t)pair * expert_mid_dim + row] =
                (g / (1.0f + expf(-g))) * u *
                weights[(uint64_t)tok * n_expert + slot];
        }
    }
}

__global__ static void poc_moe_gateup_b128_kernel(
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const int4 *staged_q8,
        const float *staged_d,
        const int32_t *selected,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        const float *weights,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        float clamp) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t wave = threadIdx.x >> 5u;
    const uint32_t block_lane = lane >> 1u;
    const uint32_t half = lane & 1u;
    const uint32_t row = blockIdx.x * 8u + wave;
    const uint32_t pair = blockIdx.y;
    const uint32_t tok = pair / n_expert;
    const uint32_t slot = pair - tok * n_expert;
    const int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) return;
    const uint32_t expert = (uint32_t)expert_i;
    const uint32_t mxfp4_blocks = xq_blocks * 8u;
    float gate = 0.0f, up = 0.0f;
    const int4 *sq = staged_q8 + (uint64_t)tok * mxfp4_blocks * 2u;
    for (uint32_t mb = block_lane; mb < mxfp4_blocks; mb += 16u) {
        const float yd = staged_d[tok * xq_blocks + (mb >> 3u)];
        const int4 q8v = sq[(uint64_t)mb * 2u + half];
        const uint32_t woff = half * 8u;
        const cuda_block_mxfp4 *gb =
            (const cuda_block_mxfp4 *)(gate_base +
                (uint64_t)expert * gate_expert_bytes +
                (uint64_t)row * gate_row_bytes) + mb;
        const cuda_block_mxfp4 *ub =
            (const cuda_block_mxfp4 *)(up_base +
                (uint64_t)expert * gate_expert_bytes +
                (uint64_t)row * gate_row_bytes) + mb;
        int32_t gsum = 0, usum = 0;
        #pragma unroll
        for (uint32_t j = 0; j < 8u; j += 4u) {
            int32_t gwlo, gwhi, uwlo, uwhi;
            dev_mxfp4_unpack2x4(gb->qs + woff + j, &gwlo, &gwhi);
            dev_mxfp4_unpack2x4(ub->qs + woff + j, &uwlo, &uwhi);
            const int32_t qlo = j == 0u ? q8v.x : q8v.y;
            const int32_t qhi = j == 0u ? q8v.z : q8v.w;
            gsum = __dp4a(gwlo, qlo, gsum);
            gsum = __dp4a(gwhi, qhi, gsum);
            usum = __dp4a(uwlo, qlo, usum);
            usum = __dp4a(uwhi, qhi, usum);
        }
        gate += 0.5f * yd * dev_e8m0_to_f32(gb->e) * (float)gsum;
        up   += 0.5f * yd * dev_e8m0_to_f32(ub->e) * (float)usum;
    }
    gate = warp_sum_f32(gate);
    up = warp_sum_f32(up);
    if (lane == 0u) {
        float g = gate, u = up;
        if (clamp > 1.0e-6f) {
            if (g > clamp) g = clamp;
            if (u > clamp) u = clamp;
            if (u < -clamp) u = -clamp;
        }
        mid_out[(uint64_t)pair * expert_mid_dim + row] =
            (g / (1.0f + expf(-g))) * u *
            weights[(uint64_t)tok * n_expert + slot];
    }
}

extern "C" int ds4_gpu_rocm_moe_poc(void);
extern "C" int ds4_gpu_rocm_moe_poc(void) {
    const uint64_t wcopy = POC_MOE_EXPERT_BYTES * POC_MOE_EXPERTS;
    const uint64_t wtotal = wcopy * 2u * POC_MOE_WCOPIES;
    char *w_dev = NULL;
    cuda_block_q8_K *xq_dev = NULL;
    int4 *staged_dev = NULL;
    float *staged_d_dev = NULL;
    int32_t *sel_dev = NULL;
    float *weights_dev = NULL;
    float *mid_dev = NULL;
    if (cudaMalloc(&w_dev, wtotal) != cudaSuccess ||
        cudaMalloc(&xq_dev, POC_MOE_XQ_BLOCKS * sizeof(cuda_block_q8_K)) != cudaSuccess ||
        cudaMalloc(&staged_dev, POC_MOE_XQ_BLOCKS * 8u * 2u * sizeof(int4)) != cudaSuccess ||
        cudaMalloc(&staged_d_dev, POC_MOE_XQ_BLOCKS * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&sel_dev, POC_MOE_EXPERTS * sizeof(int32_t)) != cudaSuccess ||
        cudaMalloc(&weights_dev, POC_MOE_EXPERTS * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&mid_dev, POC_MOE_EXPERTS * POC_MOE_MID * sizeof(float)) != cudaSuccess) {
        fprintf(stderr, "moe poc: alloc failed\n");
        return 1;
    }
    unsigned char *wh = (unsigned char *)malloc((size_t)(wcopy * 2u));
    if (!wh) return 1;
    uint32_t rng = 999u;
    for (uint64_t i = 0; i < wcopy * 2u; i += 17u) {
        wh[i] = 127u; /* e8m0 scale 1.0 */
        for (uint32_t j = 1u; j < 17u; j++) {
            rng = rng * 1664525u + 1013904223u;
            wh[i + j] = (unsigned char)(rng >> 16);
        }
    }
    cuda_block_q8_K xqh[POC_MOE_XQ_BLOCKS];
    int4 stagedh[POC_MOE_XQ_BLOCKS * 8u * 2u];
    float staged_dh[POC_MOE_XQ_BLOCKS];
    for (uint32_t b = 0; b < POC_MOE_XQ_BLOCKS; b++) {
        xqh[b].d = 0.02f;
        for (uint32_t i = 0; i < 256u; i++) {
            rng = rng * 1664525u + 1013904223u;
            xqh[b].qs[i] = (int8_t)(rng >> 24);
        }
        for (uint32_t i = 0; i < 16u; i++) xqh[b].bsums[i] = 0;
        staged_dh[b] = xqh[b].d;
        for (uint32_t sb = 0; sb < 8u; sb++) {
            for (uint32_t h = 0; h < 2u; h++) {
                const int8_t *lo = xqh[b].qs + sb * 32u + h * 8u;
                const int8_t *hi = lo + 16u;
                int4 v;
                memcpy(&v.x, lo, 4);
                memcpy(&v.y, lo + 4, 4);
                memcpy(&v.z, hi, 4);
                memcpy(&v.w, hi + 4, 4);
                stagedh[(b * 8u + sb) * 2u + h] = v;
            }
        }
    }
    int32_t selh[POC_MOE_EXPERTS];
    float wgh[POC_MOE_EXPERTS];
    for (uint32_t e = 0; e < POC_MOE_EXPERTS; e++) { selh[e] = (int32_t)e; wgh[e] = 1.0f; }
    for (uint32_t c = 0; c < POC_MOE_WCOPIES; c++) {
        if (cudaMemcpy(w_dev + c * wcopy * 2u, wh, wcopy * 2u,
                       cudaMemcpyHostToDevice) != cudaSuccess) return 1;
    }
    free(wh);
    if (cudaMemcpy(xq_dev, xqh, sizeof(xqh), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(staged_dev, stagedh, sizeof(stagedh), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(staged_d_dev, staged_dh, sizeof(staged_dh), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(sel_dev, selh, sizeof(selh), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(weights_dev, wgh, sizeof(wgh), cudaMemcpyHostToDevice) != cudaSuccess) return 1;

    const uint64_t bytes_iter = wcopy * 2u;
    printf("moe poc: gate/up decode 1 tok x %u experts, mid=%u (%.2f MiB weights x%u copies)\n",
           POC_MOE_EXPERTS, POC_MOE_MID, (double)bytes_iter / 1048576.0, POC_MOE_WCOPIES);
    const char *names[] = {"F production gate/up", "G 2 rows/warp ILP", "H b128 activations"};
    float refm[POC_MOE_EXPERTS * POC_MOE_MID];
    int have_ref = 0;
    for (uint32_t v = 0; v < 3u; v++) {
        hipEvent_t e0, e1;
        (void)hipEventCreate(&e0);
        (void)hipEventCreate(&e1);
        for (uint32_t i = 0; i < 20u + POC_ITERS; i++) {
            const char *gc = w_dev + (i % POC_MOE_WCOPIES) * wcopy * 2u;
            const char *uc = gc + wcopy;
            dim3 grid((POC_MOE_MID + (v == 1u ? 15u : 7u)) / (v == 1u ? 16u : 8u),
                      POC_MOE_EXPERTS, 1u);
            if (i == 20u) (void)hipEventRecord(e0, 0);
            switch (v) {
            case 0u:
                moe_gate_up_mid_decode_mxfp4_qwarp32_kernel<<<grid, 256>>>(
                    NULL, NULL, mid_dev, gc, uc, xq_dev, sel_dev, weights_dev,
                    POC_MOE_EXPERT_BYTES, POC_MOE_ROW_BYTES, POC_MOE_XQ_BLOCKS,
                    POC_MOE_MID, POC_MOE_EXPERTS, 0u, 0u, POC_MOE_EXPERTS, 0u, 88.0f);
                break;
            case 1u:
                poc_moe_gateup_rw_kernel<2u><<<grid, 256>>>(
                    mid_dev, gc, uc, xq_dev, sel_dev,
                    POC_MOE_EXPERT_BYTES, POC_MOE_ROW_BYTES, weights_dev, POC_MOE_XQ_BLOCKS,
                    POC_MOE_MID, POC_MOE_EXPERTS, 88.0f);
                break;
            default:
                poc_moe_gateup_b128_kernel<<<grid, 256>>>(
                    mid_dev, gc, uc, staged_dev, staged_d_dev, sel_dev,
                    POC_MOE_EXPERT_BYTES, POC_MOE_ROW_BYTES, weights_dev, POC_MOE_XQ_BLOCKS,
                    POC_MOE_MID, POC_MOE_EXPERTS, 88.0f);
                break;
            }
        }
        (void)hipEventRecord(e1, 0);
        (void)hipEventSynchronize(e1);
        float ms = 0.0f;
        (void)hipEventElapsedTime(&ms, e0, e1);
        const double ms_iter = (double)ms / POC_ITERS;
        printf("  %-26s %8.1f us/iter  %7.1f GB/s\n", names[v],
               ms_iter * 1000.0, (double)bytes_iter / (ms_iter * 1.0e6));
        float got[POC_MOE_EXPERTS * POC_MOE_MID];
        if (cudaMemcpy(got, mid_dev, sizeof(got), cudaMemcpyDeviceToHost) != cudaSuccess) return 1;
        if (!have_ref) { memcpy(refm, got, sizeof(refm)); have_ref = 1; }
        else {
            double maxdiff = 0.0;
            for (uint32_t r = 0; r < POC_MOE_EXPERTS * POC_MOE_MID; r++) {
                const double d = fabs((double)got[r] - (double)refm[r]);
                if (d > maxdiff) maxdiff = d;
            }
            printf("      vs F: max|diff| = %.3g %s\n", maxdiff,
                   maxdiff == 0.0 ? "(bit-identical)" : "(DIFFERS - investigate)");
        }
        (void)hipEventDestroy(e0);
        (void)hipEventDestroy(e1);
    }
    (void)cudaFree(w_dev);
    (void)cudaFree(xq_dev);
    (void)cudaFree(staged_dev);
    (void)cudaFree(staged_d_dev);
    (void)cudaFree(sel_dev);
    (void)cudaFree(weights_dev);
    (void)cudaFree(mid_dev);
    return 0;
}

extern "C" int ds4_gpu_rocm_moe_gap_poc(void);
extern "C" int ds4_gpu_rocm_moe_gap_poc(void) {
    /* Clock-ramp probe: launch the production gate/up kernel once per
     * iteration, with a device sync + host idle gap before each launch,
     * mimicking the production decode stream's sync points.  If the APU
     * downclocks in the gaps, the single-launch latency should be several
     * times the back-to-back bench figure (~44 us). */
    const uint64_t wcopy = POC_MOE_EXPERT_BYTES * POC_MOE_EXPERTS;
    char *w_dev = NULL;
    cuda_block_q8_K *xq_dev = NULL;
    int32_t *sel_dev = NULL;
    float *weights_dev = NULL;
    float *mid_dev = NULL;
    if (cudaMalloc(&w_dev, wcopy * 2u * POC_MOE_WCOPIES) != cudaSuccess ||
        cudaMalloc(&xq_dev, POC_MOE_XQ_BLOCKS * sizeof(cuda_block_q8_K)) != cudaSuccess ||
        cudaMalloc(&sel_dev, POC_MOE_EXPERTS * sizeof(int32_t)) != cudaSuccess ||
        cudaMalloc(&weights_dev, POC_MOE_EXPERTS * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&mid_dev, POC_MOE_EXPERTS * POC_MOE_MID * sizeof(float)) != cudaSuccess) {
        fprintf(stderr, "gap poc: alloc failed\n");
        return 1;
    }
    unsigned char *wh = (unsigned char *)malloc((size_t)(wcopy * 2u));
    cuda_block_q8_K xqh[POC_MOE_XQ_BLOCKS];
    int32_t selh[POC_MOE_EXPERTS];
    float wgh[POC_MOE_EXPERTS];
    uint32_t rng = 4242u;
    for (uint64_t i = 0; i < wcopy * 2u; i += 17u) {
        wh[i] = 127u;
        for (uint32_t j = 1u; j < 17u; j++) { rng = rng * 1664525u + 1013904223u; wh[i + j] = (unsigned char)(rng >> 16); }
    }
    for (uint32_t b = 0; b < POC_MOE_XQ_BLOCKS; b++) {
        xqh[b].d = 0.02f;
        for (uint32_t i = 0; i < 256u; i++) { rng = rng * 1664525u + 1013904223u; xqh[b].qs[i] = (int8_t)(rng >> 24); }
        for (uint32_t i = 0; i < 16u; i++) xqh[b].bsums[i] = 0;
    }
    for (uint32_t e = 0; e < POC_MOE_EXPERTS; e++) { selh[e] = (int32_t)e; wgh[e] = 1.0f; }
    for (uint32_t c = 0; c < POC_MOE_WCOPIES; c++)
        if (cudaMemcpy(w_dev + c * wcopy * 2u, wh, wcopy * 2u, cudaMemcpyHostToDevice) != cudaSuccess) return 1;
    free(wh);
    if (cudaMemcpy(xq_dev, xqh, sizeof(xqh), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(sel_dev, selh, sizeof(selh), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(weights_dev, wgh, sizeof(wgh), cudaMemcpyHostToDevice) != cudaSuccess) return 1;
    const dim3 grid(POC_MOE_MID / 8u, POC_MOE_EXPERTS, 1u);
    /* warm */
    for (uint32_t i = 0; i < 50u; i++)
        moe_gate_up_mid_decode_mxfp4_qwarp32_kernel<<<grid, 256>>>(
            NULL, NULL, mid_dev, w_dev, w_dev + wcopy, xq_dev, sel_dev, weights_dev,
            POC_MOE_EXPERT_BYTES, POC_MOE_ROW_BYTES, POC_MOE_XQ_BLOCKS,
            POC_MOE_MID, POC_MOE_EXPERTS, 0u, 0u, POC_MOE_EXPERTS, 0u, 88.0f);
    (void)hipDeviceSynchronize();
    const uint32_t gaps[] = {0u, 100u, 500u, 2000u};
    for (uint32_t gi = 0; gi < 4u; gi++) {
        const uint32_t gap_us = gaps[gi];
        double total_us = 0.0;
        const uint32_t reps = 60u;
        for (uint32_t i = 0; i < reps; i++) {
            (void)hipDeviceSynchronize();
            if (gap_us) {
                const double t0 = now_sec_host();
                while ((now_sec_host() - t0) * 1.0e6 < (double)gap_us) { }
            }
            hipEvent_t e0, e1;
            (void)hipEventCreate(&e0);
            (void)hipEventCreate(&e1);
            (void)hipEventRecord(e0, 0);
            const char *gc = w_dev + (i % POC_MOE_WCOPIES) * wcopy * 2u;
            moe_gate_up_mid_decode_mxfp4_qwarp32_kernel<<<grid, 256>>>(
                NULL, NULL, mid_dev, gc, gc + wcopy, xq_dev, sel_dev, weights_dev,
                POC_MOE_EXPERT_BYTES, POC_MOE_ROW_BYTES, POC_MOE_XQ_BLOCKS,
                POC_MOE_MID, POC_MOE_EXPERTS, 0u, 0u, POC_MOE_EXPERTS, 0u, 88.0f);
            (void)hipEventRecord(e1, 0);
            (void)hipEventSynchronize(e1);
            float ms = 0.0f;
            (void)hipEventElapsedTime(&ms, e0, e1);
            total_us += (double)ms * 1000.0;
            (void)hipEventDestroy(e0);
            (void)hipEventDestroy(e1);
        }
        printf("  gap %5u us:  %8.1f us/launch  %7.1f GB/s\n", gap_us,
               total_us / reps, (double)(wcopy * 2u) / ((total_us / reps) * 1.0e6));
    }
    (void)cudaFree(w_dev); (void)cudaFree(xq_dev); (void)cudaFree(sel_dev);
    (void)cudaFree(weights_dev); (void)cudaFree(mid_dev);
    return 0;
}
