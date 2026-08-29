// proto_m2_q2k.cu — M2 megakernel program, moe-down Q2K aligned-SoA proto.
//
// Target: mul_mat_vec_q_moe<GGML_TYPE_Q2_K, 2> (cuda/mmq/mmvq.cu:608), the
// production decode down-proj kernel: 43 launches/step at 87.3 us = 3.755
// ms/step reading raw 84-byte block_q2_K stacks at ~190 GB/s vs the 216-233
// GB/s device substrate (nsys33d).  This proto vendors the production kernel
// as the bit-exact reference and races two aligned-SoA layout twins that keep
// the SAME lane->(block,iqs) mapping and float accumulation order (layout-only
// change => bit-identical outputs, M1-Inc4 gate recipe):
//
//   V1: plain SoA sections  [half2 dm[nblk]] [pad64] [u8 scales[nblk*16]]
//       [pad64] [u32 qs[nblk*16]], block order = raw tensor byte order.
//   V2: row-pair SoA keyed to rows_per_block=2 (row pair = 2*blockIdx.x):
//       [uint2 dm2[npair*nb]] [pad64] [int4 sc4[npair*nb*2]] [pad64]
//       [uint2 qs2[npair*nb*16]]; per lane-iter one 8B qs load (both rows),
//       one 16B scales load (both rows' 8B window), one 8B dm load.
//
// Parity: full-buffer bitwise compare vs the raw reference across seeds x id
// patterns (random experts, duplicates, -1 router ids, all -1) x ncols_dst
// {1, 6, 8}, plus graph capture/replay == eager for both variants.
//
// Timing: rotating L-layer weight pool (defeats L2 the way the 43-layer model
// does), production shape M=4096 K=2048 top-6, 6 distinct experts per launch.
//
// build (GB10): nvcc -O3 --use_fast_math -std=c++17 -arch=sm_121 \
//                 -o proto_m2_q2k proto_m2_q2k.cu

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <random>

#define CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

// ---------------------------------------------------------------------------
// Vendored ggml pieces (verbatim math).
// ---------------------------------------------------------------------------
#define QK_K   256
#define QI2_K  16      // QK_K / (4*QR2_K)
#define QR2_K  4
#define QK8_1  32
#define QI8_1  8       // QK8_1 / (4*QR8_1)

typedef struct {
    __half2 ds;        // d (delta), s (d * sum of quants)
    int8_t  qs[QK8_1];
} block_q8_1;
static_assert(sizeof(block_q8_1) == 36, "block_q8_1 size");

typedef struct {
    uint8_t scales[QK_K/16];
    uint8_t qs[QK_K/4];
    __half2 dm;        // d, dmin
} block_q2_K;
static_assert(sizeof(block_q2_K) == 84, "block_q2_K size");

static __device__ __forceinline__ int get_int_b4(const void *x, const int &i32) {
    return ((const int *)x)[i32];   // 4-byte aligned
}

static __device__ __forceinline__ int dp4a_(const int a, const int b, int c) {
    return __dp4a(a, b, c);
}

template<int width = 32>
static __device__ __forceinline__ float warp_reduce_sum(float x) {
#pragma unroll
    for (int offset = width/2; offset > 0; offset >>= 1) {
        x += __shfl_xor_sync(0xffffffff, x, offset, width);
    }
    return x;
}

// vec_dot_q2_K_q8_1_impl_mmvq, verbatim (vecdotq.cuh:364).
static __device__ __forceinline__ float vec_dot_q2_K_q8_1_impl_mmvq(
    const int & v, const int * __restrict__ u, const uint8_t * __restrict__ scales,
    const __half2 & dm2, const float * __restrict__ d8) {

    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

#pragma unroll
    for (int i = 0; i < QR2_K; ++i) {
        const int sc = scales[2*i];

        const int vi = (v >> (2*i)) & 0x03030303;

        sumf_d += d8[i] * (dp4a_(vi, u[i], 0) * (sc & 0xF));

        int m = sc >> 4;
        m |= m <<  8;
        m |= m << 16;
        sumf_m += d8[i] * dp4a_(m, u[i], 0);
    }

    const float2 dm2f = __half22float2(dm2);

    return dm2f.x*sumf_d - dm2f.y*sumf_m;
}

// Same impl but the four scale bytes come from two pre-loaded 32-bit window
// words (V2's 16B scales load).  sc value chain identical: byte (lo + 2*i) of
// the 8-byte window == scales[scale_offset + 2*i] of the raw block.
static __device__ __forceinline__ float vec_dot_q2_K_q8_1_impl_mmvq_w(
    const int & v, const int * __restrict__ u, const uint32_t w0, const uint32_t w1,
    const int lo, const __half2 & dm2, const float * __restrict__ d8) {

    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

#pragma unroll
    for (int i = 0; i < QR2_K; ++i) {
        const int bidx = lo + 2*i;                       // 0..7 within the window
        const uint32_t w = (bidx < 4) ? w0 : w1;
        const int sc = (int)((w >> ((bidx & 3) * 8)) & 0xFFu);

        const int vi = (v >> (2*i)) & 0x03030303;

        sumf_d += d8[i] * (dp4a_(vi, u[i], 0) * (sc & 0xF));

        int m = sc >> 4;
        m |= m <<  8;
        m |= m << 16;
        sumf_m += d8[i] * dp4a_(m, u[i], 0);
    }

    const float2 dm2f = __half22float2(dm2);

    return dm2f.x*sumf_d - dm2f.y*sumf_m;
}

// vec_dot_q2_K_q8_1, verbatim (vecdotq.cuh:814).
static __device__ __forceinline__ float vec_dot_q2_K_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_q2_K * bq2_K = (const block_q2_K *) vbq + kbx;

    const int bq8_offset = QR2_K * (iqs / QI8_1);
    const int scale_offset = iqs - iqs % QI8_1 + (iqs % QI8_1) / (QI8_1/2);

    const uint8_t * scales = bq2_K->scales + scale_offset;

    const int v = get_int_b4(bq2_K->qs, iqs);
    int    u[QR2_K];
    float d8[QR2_K];

#pragma unroll
    for (int i = 0; i < QR2_K; ++ i) {
        u[i]  = get_int_b4(bq8_1[bq8_offset + i].qs, iqs % QI8_1);
        d8[i] = __low2float(bq8_1[bq8_offset + i].ds);
    }

    return vec_dot_q2_K_q8_1_impl_mmvq(v, u, scales, bq2_K->dm, d8);
}

// ---------------------------------------------------------------------------
// Reference: mul_mat_vec_q_moe<GGML_TYPE_Q2_K, 2> (mmvq.cu:608), verbatim up
// to Q2_K constant folding (qk 256, qi 16, vdr 1) and fastmodulo -> % (index
// math only).  GB10 (sm_121 >= Ada): launch bounds MMVQ_MAX_BATCH_SIZE(8)*32.
// ---------------------------------------------------------------------------
template <int c_rows_per_block>
__launch_bounds__(8*32, 1)
static __global__ void ref_mul_mat_vec_q_moe(
        const void * __restrict__ vx, const void * __restrict__ vy, const int32_t * __restrict__ ids,
        float * __restrict__ dst,
        const uint32_t ncols_x, const uint32_t nchannels_y, const uint32_t nrows_x,
        const uint32_t stride_row_x, const uint32_t stride_col_y, const uint32_t stride_col_dst,
        const uint32_t stride_channel_x, const uint32_t stride_channel_y, const uint32_t stride_channel_dst,
        const uint32_t ncols_dst, const uint32_t ids_stride) {

    constexpr int qk  = QK_K;
    constexpr int qi  = QI2_K;
    constexpr int vdr = 1;                 // VDR_Q2_K_Q8_1_MMVQ
    constexpr int warp_size = 32;

    const uint32_t token_idx   = threadIdx.y;
    const int      row0        = c_rows_per_block*blockIdx.x;
    const int      blocks_per_row_x = ncols_x / qk;
    constexpr int  blocks_per_iter  = vdr * warp_size / qi;   // 2

    const uint32_t channel_dst = blockIdx.y;

    if (token_idx >= ncols_dst) {
        return;
    }

    const int32_t  id_raw     = ids[channel_dst + token_idx * ids_stride];
    const bool     invalid_id = id_raw < 0;
    const uint32_t channel_x  = invalid_id ? 0u : (uint32_t)id_raw;
    const uint32_t channel_y  = channel_dst % nchannels_y;

    const block_q8_1 * y = ((const block_q8_1 *) vy) + channel_y*stride_channel_y + token_idx*stride_col_y;
    const int kbx_offset  = channel_x*stride_channel_x + row0*stride_row_x;

    float tmp[c_rows_per_block] = {0.0f};

    for (int kbx = threadIdx.x / (qi/vdr); !invalid_id && kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1);
        const int kqs = vdr * (threadIdx.x % (qi/vdr));

#pragma unroll
        for (int i = 0; i < c_rows_per_block; ++i) {
            tmp[i] += vec_dot_q2_K_q8_1(vx, &y[kby], kbx_offset + i*stride_row_x + kbx, kqs);
        }
    }

#pragma unroll
    for (int i = 0; i < c_rows_per_block; ++i) {
        tmp[i] = warp_reduce_sum<warp_size>(tmp[i]);
    }

    if (threadIdx.x < c_rows_per_block && (c_rows_per_block == 1 || uint32_t(row0 + threadIdx.x) < nrows_x)) {
        dst[channel_dst*stride_channel_dst + token_idx*stride_col_dst + row0 + threadIdx.x] = tmp[threadIdx.x];
    }
}

// ---------------------------------------------------------------------------
// V1: plain SoA twin.  Same skeleton, same lane mapping, same accumulation
// order; only the three weight streams are re-sourced from aligned sections.
// ---------------------------------------------------------------------------
template <int c_rows_per_block>
__launch_bounds__(8*32, 1)
static __global__ void v1_soa_moe_kernel(
        const __half2 * __restrict__ dm_soa,       // [nblk]
        const uint8_t * __restrict__ scales_soa,   // [nblk*16]
        const int     * __restrict__ qs_soa,       // [nblk*16]
        const void * __restrict__ vy, const int32_t * __restrict__ ids,
        float * __restrict__ dst,
        const uint32_t ncols_x, const uint32_t nchannels_y, const uint32_t nrows_x,
        const uint32_t stride_row_x, const uint32_t stride_col_y, const uint32_t stride_col_dst,
        const uint32_t stride_channel_x, const uint32_t stride_channel_y, const uint32_t stride_channel_dst,
        const uint32_t ncols_dst, const uint32_t ids_stride) {

    constexpr int qk  = QK_K;
    constexpr int qi  = QI2_K;
    constexpr int vdr = 1;
    constexpr int warp_size = 32;

    const uint32_t token_idx   = threadIdx.y;
    const int      row0        = c_rows_per_block*blockIdx.x;
    const int      blocks_per_row_x = ncols_x / qk;
    constexpr int  blocks_per_iter  = vdr * warp_size / qi;

    const uint32_t channel_dst = blockIdx.y;

    if (token_idx >= ncols_dst) {
        return;
    }

    const int32_t  id_raw     = ids[channel_dst + token_idx * ids_stride];
    const bool     invalid_id = id_raw < 0;
    const uint32_t channel_x  = invalid_id ? 0u : (uint32_t)id_raw;
    const uint32_t channel_y  = channel_dst % nchannels_y;

    const block_q8_1 * y = ((const block_q8_1 *) vy) + channel_y*stride_channel_y + token_idx*stride_col_y;
    const int kbx_offset  = channel_x*stride_channel_x + row0*stride_row_x;

    float tmp[c_rows_per_block] = {0.0f};

    for (int kbx = threadIdx.x / (qi/vdr); !invalid_id && kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1);
        const int kqs = vdr * (threadIdx.x % (qi/vdr));

        const int iqs = kqs;
        const int bq8_offset = QR2_K * (iqs / QI8_1);
        const int scale_offset = iqs - iqs % QI8_1 + (iqs % QI8_1) / (QI8_1/2);
        const block_q8_1 * bq8_1 = &y[kby];

        int    u[QR2_K];
        float d8[QR2_K];
#pragma unroll
        for (int i = 0; i < QR2_K; ++i) {
            u[i]  = get_int_b4(bq8_1[bq8_offset + i].qs, iqs % QI8_1);
            d8[i] = __low2float(bq8_1[bq8_offset + i].ds);
        }

#pragma unroll
        for (int i = 0; i < c_rows_per_block; ++i) {
            const size_t blk = (size_t)(kbx_offset + i*stride_row_x + kbx);
            const int v = qs_soa[blk*16u + (unsigned)iqs];
            tmp[i] += vec_dot_q2_K_q8_1_impl_mmvq(v, u, scales_soa + blk*16u + (unsigned)scale_offset,
                                                  dm_soa[blk], d8);
        }
    }

#pragma unroll
    for (int i = 0; i < c_rows_per_block; ++i) {
        tmp[i] = warp_reduce_sum<warp_size>(tmp[i]);
    }

    if (threadIdx.x < c_rows_per_block && (c_rows_per_block == 1 || uint32_t(row0 + threadIdx.x) < nrows_x)) {
        dst[channel_dst*stride_channel_dst + token_idx*stride_col_dst + row0 + threadIdx.x] = tmp[threadIdx.x];
    }
}

// ---------------------------------------------------------------------------
// V2: row-pair SoA twin (rows_per_block == 2 baked into the layout).
// Per lane-iter: one uint2 qs load (rows 0/1), one int4 scales-window load
// (both rows' 8B window for this lane's iqs half), one uint2 dm load.  Scale
// BYTES, qs ints, dm halves and the float chain are value-identical to the
// reference; only load shapes change.
// ---------------------------------------------------------------------------
__launch_bounds__(8*32, 1)
static __global__ void v2_pair_soa_moe_kernel(
        const uint2 * __restrict__ dm2_soa,   // [npair*nb]      {row0 half2, row1 half2}
        const int4  * __restrict__ sc4_soa,   // [npair*nb*2]    {row0 w0,w1, row1 w0,w1} per half
        const uint2 * __restrict__ qs2_soa,   // [npair*nb*16]   {row0 int, row1 int}
        const void * __restrict__ vy, const int32_t * __restrict__ ids,
        float * __restrict__ dst,
        const uint32_t ncols_x, const uint32_t nchannels_y, const uint32_t nrows_x,
        const uint32_t stride_col_y, const uint32_t stride_col_dst,
        const uint32_t stride_channel_y, const uint32_t stride_channel_dst,
        const uint32_t ncols_dst, const uint32_t ids_stride) {

    constexpr int qk  = QK_K;
    constexpr int qi  = QI2_K;
    constexpr int vdr = 1;
    constexpr int warp_size = 32;

    const uint32_t token_idx   = threadIdx.y;
    const int      row0        = 2*blockIdx.x;
    const int      blocks_per_row_x = ncols_x / qk;
    constexpr int  blocks_per_iter  = vdr * warp_size / qi;

    const uint32_t channel_dst = blockIdx.y;

    if (token_idx >= ncols_dst) {
        return;
    }

    const int32_t  id_raw     = ids[channel_dst + token_idx * ids_stride];
    const bool     invalid_id = id_raw < 0;
    const uint32_t channel_x  = invalid_id ? 0u : (uint32_t)id_raw;
    const uint32_t channel_y  = channel_dst % nchannels_y;

    const block_q8_1 * y = ((const block_q8_1 *) vy) + channel_y*stride_channel_y + token_idx*stride_col_y;
    // pair-block base for (expert, row pair): pairs are laid out expert-major,
    // pair-major, block-minor -- same linear order as raw blocks per pair.
    const size_t pair_base = ((size_t)channel_x * (nrows_x/2u) + (size_t)blockIdx.x) * (size_t)blocks_per_row_x;

    float tmp[2] = {0.0f, 0.0f};

    for (int kbx = threadIdx.x / (qi/vdr); !invalid_id && kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1);
        const int kqs = vdr * (threadIdx.x % (qi/vdr));

        const int iqs = kqs;
        const int bq8_offset = QR2_K * (iqs / QI8_1);
        const int scale_offset = iqs - iqs % QI8_1 + (iqs % QI8_1) / (QI8_1/2);
        const int whalf = iqs / QI8_1;            // 0: window at scales[0..7], 1: at [8..15]
        const int lo    = scale_offset - 8*whalf; // 0 or 1 within the window
        const block_q8_1 * bq8_1 = &y[kby];

        int    u[QR2_K];
        float d8[QR2_K];
#pragma unroll
        for (int i = 0; i < QR2_K; ++i) {
            u[i]  = get_int_b4(bq8_1[bq8_offset + i].qs, iqs % QI8_1);
            d8[i] = __low2float(bq8_1[bq8_offset + i].ds);
        }

        const size_t pblk = pair_base + (size_t)kbx;
        const uint2 v2  = qs2_soa[pblk*16u + (unsigned)iqs];
        const uint2 dmw = dm2_soa[pblk];
        const int4  scw = sc4_soa[pblk*2u + (unsigned)whalf];
        const __half2 dm0 = *(const __half2 *)&dmw.x;
        const __half2 dm1 = *(const __half2 *)&dmw.y;

        tmp[0] += vec_dot_q2_K_q8_1_impl_mmvq_w((int)v2.x, u, (uint32_t)scw.x, (uint32_t)scw.y, lo, dm0, d8);
        tmp[1] += vec_dot_q2_K_q8_1_impl_mmvq_w((int)v2.y, u, (uint32_t)scw.z, (uint32_t)scw.w, lo, dm1, d8);
    }

#pragma unroll
    for (int i = 0; i < 2; ++i) {
        tmp[i] = warp_reduce_sum<warp_size>(tmp[i]);
    }

    if (threadIdx.x < 2 && uint32_t(row0 + threadIdx.x) < nrows_x) {
        dst[channel_dst*stride_channel_dst + token_idx*stride_col_dst + row0 + threadIdx.x] = tmp[threadIdx.x];
    }
}

// ---------------------------------------------------------------------------
// Host helpers.
// ---------------------------------------------------------------------------
static const int M   = 4096;   // out rows (down out_dim)
static const int K   = 2048;   // in cols (expert_mid_dim)
static const int NB  = K / QK_K;    // 8 superblocks per row
static const int NYB = K / QK8_1;   // 64 q8_1 blocks per activation row
static const int NEXP = 256;
static const int NUSED = 6;

static size_t nblk_layer() { return (size_t)NEXP * M * NB; }

static void fill_raw_layer(std::vector<block_q2_K> &w, std::mt19937 &rng) {
    std::uniform_int_distribution<int> byte(0, 255);
    std::uniform_real_distribution<float> mag(0.001f, 0.05f);
    for (block_q2_K &b : w) {
        for (int i = 0; i < 16; i++) b.scales[i] = (uint8_t)byte(rng);
        for (int i = 0; i < 64; i++) b.qs[i] = (uint8_t)byte(rng);
        b.dm = __floats2half2_rn(mag(rng), mag(rng));
    }
}

static void quantize_q8_1(const float *x, block_q8_1 *y, int n) {
    for (int ib = 0; ib < n / QK8_1; ib++) {
        const float *xb = x + ib*QK8_1;
        float amax = 0.0f;
        for (int j = 0; j < QK8_1; j++) { float a = fabsf(xb[j]); if (a > amax) amax = a; }
        const float d = amax / 127.0f;
        const float idf = d != 0.0f ? 1.0f/d : 0.0f;
        int sum = 0;
        for (int j = 0; j < QK8_1; j++) {
            int q = (int)roundf(xb[j] * idf);
            if (q > 127) q = 127;
            if (q < -127) q = -127;
            y[ib].qs[j] = (int8_t)q;
            sum += q;
        }
        y[ib].ds = __floats2half2_rn(d, d * (float)sum);
    }
}

// V1 repack: [dm nblk*4][pad64][scales nblk*16][pad64][qs nblk*64]
static void repack_v1(const block_q2_K *raw, size_t nblk,
                      __half2 *dm, uint8_t *scales, int *qs) {
    for (size_t b = 0; b < nblk; b++) {
        dm[b] = raw[b].dm;
        memcpy(scales + b*16u, raw[b].scales, 16);
        memcpy(qs + b*16u, raw[b].qs, 64);
    }
}

// V2 repack: pair p covers raw blocks of rows (2p, 2p+1) within an expert.
// Raw linear block index of (expert e, row r, block b) = (e*M + r)*NB + b.
static void repack_v2(const block_q2_K *raw,
                      uint2 *dm2, int4 *sc4, uint2 *qs2) {
    const size_t npair_blk = (size_t)NEXP * (M/2) * NB;
    for (size_t pb = 0; pb < npair_blk; pb++) {
        const size_t b   = pb % NB;
        const size_t pr  = (pb / NB) % (M/2);
        const size_t e   = pb / ((size_t)NB * (M/2));
        const block_q2_K *r0 = raw + ((e*M + 2*pr + 0)*NB + b);
        const block_q2_K *r1 = raw + ((e*M + 2*pr + 1)*NB + b);
        uint2 dmw;
        memcpy(&dmw.x, &r0->dm, 4);
        memcpy(&dmw.y, &r1->dm, 4);
        dm2[pb] = dmw;
        // sc4[pb*2 + h] = {row0 w(h)[0:4], row0 w(h)[4:8], row1 w(h)[0:4], row1 w(h)[4:8]}
        for (int h = 0; h < 2; h++) {
            int4 s;
            memcpy(&s.x, r0->scales + 8*h + 0, 4);
            memcpy(&s.y, r0->scales + 8*h + 4, 4);
            memcpy(&s.z, r1->scales + 8*h + 0, 4);
            memcpy(&s.w, r1->scales + 8*h + 4, 4);
            sc4[pb*2 + h] = s;
        }
        for (int i = 0; i < 16; i++) {
            uint2 q;
            memcpy(&q.x, r0->qs + 4*i, 4);
            memcpy(&q.y, r1->qs + 4*i, 4);
            qs2[pb*16 + i] = q;
        }
    }
}

struct dev_layer {
    block_q2_K *raw = nullptr;
    __half2 *v1_dm = nullptr; uint8_t *v1_sc = nullptr; int *v1_qs = nullptr;
    uint2 *v2_dm = nullptr; int4 *v2_sc = nullptr; uint2 *v2_qs = nullptr;
};

// Production launch config for the down leg (routed_moe_launch vec tier):
// ncols_dst = n_assignments, nchannels_dst = 1, ids_stride = 1,
// stride_col_dst = M, stride_channel_dst = M, stride_col_y = NYB.
static void launch_ref(const dev_layer &L, const block_q8_1 *y, const int32_t *ids,
                       float *dst, int ncols, cudaStream_t s) {
    dim3 grid(M/2, 1);
    dim3 block(32, ncols);
    ref_mul_mat_vec_q_moe<2><<<grid, block, 0, s>>>(
        L.raw, y, ids, dst,
        (uint32_t)K, 1u, (uint32_t)M,
        (uint32_t)NB, (uint32_t)NYB, (uint32_t)M,
        (uint32_t)(M*NB), (uint32_t)NYB, (uint32_t)M,
        (uint32_t)ncols, 1u);
}

static void launch_v1(const dev_layer &L, const block_q8_1 *y, const int32_t *ids,
                      float *dst, int ncols, cudaStream_t s) {
    dim3 grid(M/2, 1);
    dim3 block(32, ncols);
    v1_soa_moe_kernel<2><<<grid, block, 0, s>>>(
        L.v1_dm, L.v1_sc, L.v1_qs, y, ids, dst,
        (uint32_t)K, 1u, (uint32_t)M,
        (uint32_t)NB, (uint32_t)NYB, (uint32_t)M,
        (uint32_t)(M*NB), (uint32_t)NYB, (uint32_t)M,
        (uint32_t)ncols, 1u);
}

static void launch_v2(const dev_layer &L, const block_q8_1 *y, const int32_t *ids,
                      float *dst, int ncols, cudaStream_t s) {
    dim3 grid(M/2, 1);
    dim3 block(32, ncols);
    v2_pair_soa_moe_kernel<<<grid, block, 0, s>>>(
        L.v2_dm, L.v2_sc, L.v2_qs, y, ids, dst,
        (uint32_t)K, 1u, (uint32_t)M,
        (uint32_t)NYB, (uint32_t)M,
        (uint32_t)NYB, (uint32_t)M,
        (uint32_t)ncols, 1u);
}

int main(int argc, char **argv) {
    int n_layers = 6;
    int timing_iters = 300;
    if (argc > 1) n_layers = atoi(argv[1]);
    if (argc > 2) timing_iters = atoi(argv[2]);

    const size_t nblk = nblk_layer();
    const size_t raw_bytes = nblk * sizeof(block_q2_K);
    printf("proto_m2_q2k: M=%d K=%d E=%d nb=%d layers=%d raw=%.1f MiB/layer\n",
           M, K, NEXP, NB, n_layers, raw_bytes / 1048576.0);

    std::mt19937 rng(0xC0FFEE);

    // Host-side raw layer template + repacks (one host copy reused per layer
    // with a per-layer byte perturbation so layers differ without n_layers
    // full random generations).
    std::vector<block_q2_K> hraw(nblk);
    fill_raw_layer(hraw, rng);

    std::vector<__half2> h_v1dm(nblk);
    std::vector<uint8_t> h_v1sc(nblk * 16u);
    std::vector<int>     h_v1qs(nblk * 16u);
    std::vector<uint2>   h_v2dm(nblk / 2u);
    std::vector<int4>    h_v2sc(nblk);            // nblk/2 pairs * 2
    std::vector<uint2>   h_v2qs((nblk / 2u) * 16u);

    std::vector<dev_layer> layers(n_layers);
    for (int l = 0; l < n_layers; l++) {
        if (l > 0) {
            // cheap per-layer variation: rotate scales/qs bytes by l
            for (size_t b = 0; b < nblk; b += 977u) {
                hraw[b].qs[(l*7) & 63] ^= (uint8_t)(0x5A + l);
                hraw[b].scales[(l*3) & 15] ^= (uint8_t)(0xA5 - l);
            }
        }
        repack_v1(hraw.data(), nblk, h_v1dm.data(), h_v1sc.data(), h_v1qs.data());
        repack_v2(hraw.data(), h_v2dm.data(), h_v2sc.data(), h_v2qs.data());

        dev_layer &L = layers[l];
        CHECK(cudaMalloc(&L.raw, raw_bytes));
        CHECK(cudaMalloc(&L.v1_dm, nblk * 4u));
        CHECK(cudaMalloc(&L.v1_sc, nblk * 16u));
        CHECK(cudaMalloc(&L.v1_qs, nblk * 64u));
        CHECK(cudaMalloc(&L.v2_dm, (nblk/2u) * 8u));
        CHECK(cudaMalloc(&L.v2_sc, nblk * 16u));
        CHECK(cudaMalloc(&L.v2_qs, (nblk/2u) * 16u * 8u));
        CHECK(cudaMemcpy(L.raw, hraw.data(), raw_bytes, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(L.v1_dm, h_v1dm.data(), nblk * 4u, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(L.v1_sc, h_v1sc.data(), nblk * 16u, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(L.v1_qs, h_v1qs.data(), nblk * 64u, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(L.v2_dm, h_v2dm.data(), (nblk/2u) * 8u, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(L.v2_sc, h_v2sc.data(), nblk * 16u, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(L.v2_qs, h_v2qs.data(), (nblk/2u) * 16u * 8u, cudaMemcpyHostToDevice));
    }

    // Activations: MAX_COLS q8_1 rows.
    const int MAX_COLS = 8;
    std::vector<float> hx(MAX_COLS * K);
    std::vector<block_q8_1> hy(MAX_COLS * NYB);
    std::uniform_real_distribution<float> xf(-1.0f, 1.0f);
    for (float &v : hx) v = xf(rng);
    for (int c = 0; c < MAX_COLS; c++) quantize_q8_1(hx.data() + c*K, hy.data() + c*NYB, K);
    block_q8_1 *dy = nullptr;
    CHECK(cudaMalloc(&dy, hy.size() * sizeof(block_q8_1)));
    CHECK(cudaMemcpy(dy, hy.data(), hy.size() * sizeof(block_q8_1), cudaMemcpyHostToDevice));

    int32_t *dids = nullptr;
    CHECK(cudaMalloc(&dids, MAX_COLS * sizeof(int32_t)));
    float *dref = nullptr, *dv1 = nullptr, *dv2 = nullptr;
    const size_t out_bytes = (size_t)MAX_COLS * M * sizeof(float);
    CHECK(cudaMalloc(&dref, out_bytes));
    CHECK(cudaMalloc(&dv1, out_bytes));
    CHECK(cudaMalloc(&dv2, out_bytes));

    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // ------------------------------------------------------------------
    // Parity: seeds x id patterns x ncols on rotating layers.
    // ------------------------------------------------------------------
    std::vector<float> href(MAX_COLS * M), hv1(MAX_COLS * M), hv2(MAX_COLS * M);
    const int ncols_cases[3] = {1, 6, 8};
    int checks = 0, fails = 0;
    std::uniform_int_distribution<int> expd(0, NEXP - 1);
    for (int seed = 0; seed < 10; seed++) {
        for (int pat = 0; pat < 4; pat++) {
            for (int nc = 0; nc < 3; nc++) {
                const int ncols = ncols_cases[nc];
                const dev_layer &L = layers[(seed + pat + nc) % n_layers];
                int32_t ids[MAX_COLS];
                for (int i = 0; i < ncols; i++) {
                    switch (pat) {
                        case 0: ids[i] = expd(rng); break;                    // random
                        case 1: ids[i] = (i < 2) ? expd(rng) : ids[i % 2];    // duplicates
                                break;
                        case 2: ids[i] = (i % 3 == 1) ? -1 : expd(rng); break; // sparse -1
                        case 3: ids[i] = -1; break;                            // all -1
                    }
                }
                CHECK(cudaMemcpy(dids, ids, ncols * sizeof(int32_t), cudaMemcpyHostToDevice));
                CHECK(cudaMemset(dref, 0xCB, out_bytes));
                CHECK(cudaMemset(dv1, 0xCB, out_bytes));
                CHECK(cudaMemset(dv2, 0xCB, out_bytes));
                launch_ref(L, dy, dids, dref, ncols, stream);
                launch_v1(L, dy, dids, dv1, ncols, stream);
                launch_v2(L, dy, dids, dv2, ncols, stream);
                CHECK(cudaStreamSynchronize(stream));
                CHECK(cudaGetLastError());
                const size_t cmp = (size_t)ncols * M * sizeof(float);
                CHECK(cudaMemcpy(href.data(), dref, cmp, cudaMemcpyDeviceToHost));
                CHECK(cudaMemcpy(hv1.data(), dv1, cmp, cudaMemcpyDeviceToHost));
                CHECK(cudaMemcpy(hv2.data(), dv2, cmp, cudaMemcpyDeviceToHost));
                checks += 2;
                if (memcmp(href.data(), hv1.data(), cmp) != 0) {
                    fails++;
                    printf("PARITY FAIL V1 seed=%d pat=%d ncols=%d\n", seed, pat, ncols);
                }
                if (memcmp(href.data(), hv2.data(), cmp) != 0) {
                    fails++;
                    printf("PARITY FAIL V2 seed=%d pat=%d ncols=%d\n", seed, pat, ncols);
                }
            }
        }
    }
    printf("parity: %d/%d bit-identical%s\n", checks - fails, checks, fails ? "  <-- FAIL" : "");

    // ------------------------------------------------------------------
    // Capture/replay parity: capture each variant once, replay, compare to
    // the eager output just validated.
    // ------------------------------------------------------------------
    {
        int32_t ids[MAX_COLS];
        for (int i = 0; i < NUSED; i++) ids[i] = expd(rng);
        ids[3] = -1;
        CHECK(cudaMemcpy(dids, ids, NUSED * sizeof(int32_t), cudaMemcpyHostToDevice));
        const dev_layer &L = layers[0];
        const size_t cmp = (size_t)NUSED * M * sizeof(float);

        launch_ref(L, dy, dids, dref, NUSED, stream);
        CHECK(cudaStreamSynchronize(stream));
        CHECK(cudaMemcpy(href.data(), dref, cmp, cudaMemcpyDeviceToHost));

        int replay_fails = 0;
        for (int variant = 0; variant < 2; variant++) {
            float *dout = variant == 0 ? dv1 : dv2;
            CHECK(cudaMemset(dout, 0xCB, out_bytes));
            cudaGraph_t graph;
            cudaGraphExec_t exec;
            CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
            if (variant == 0) launch_v1(L, dy, dids, dout, NUSED, stream);
            else              launch_v2(L, dy, dids, dout, NUSED, stream);
            CHECK(cudaStreamEndCapture(stream, &graph));
            CHECK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
            CHECK(cudaGraphLaunch(exec, stream));
            CHECK(cudaStreamSynchronize(stream));
            CHECK(cudaMemcpy(hv1.data(), dout, cmp, cudaMemcpyDeviceToHost));
            if (memcmp(href.data(), hv1.data(), cmp) != 0) {
                replay_fails++;
                printf("REPLAY FAIL V%d\n", variant + 1);
            }
            CHECK(cudaGraphExecDestroy(exec));
            CHECK(cudaGraphDestroy(graph));
        }
        printf("capture/replay: %d/2 bit-identical%s\n", 2 - replay_fails, replay_fails ? "  <-- FAIL" : "");
        fails += replay_fails;
    }

    // ------------------------------------------------------------------
    // Timing: rotating layers, 6 distinct experts per launch (production
    // decode shape), events around the whole loop.
    // ------------------------------------------------------------------
    {
        // Pre-generate rotating id sets on device (one buffer per iteration
        // slot would perturb the graph; production re-writes ids in place, so
        // do the same: ids uploaded once per timing pass, kernels rotate
        // layers only -- weight reads dominate).
        int32_t ids[MAX_COLS];
        bool used[NEXP] = {false};
        int n = 0;
        while (n < NUSED) {
            int e = expd(rng);
            if (used[e]) continue;
            used[e] = true;
            ids[n++] = e;
        }
        CHECK(cudaMemcpy(dids, ids, NUSED * sizeof(int32_t), cudaMemcpyHostToDevice));

        const double gib = (double)NUSED * M * NB * 84.0;  // distinct-expert weight bytes/launch
        cudaEvent_t e0, e1;
        CHECK(cudaEventCreate(&e0));
        CHECK(cudaEventCreate(&e1));

        for (int variant = 0; variant < 3; variant++) {
            const char *name = variant == 0 ? "ref-raw" : (variant == 1 ? "v1-soa " : "v2-pair");
            float *dout = variant == 0 ? dref : (variant == 1 ? dv1 : dv2);
            // warm
            for (int it = 0; it < 30; it++) {
                const dev_layer &L = layers[it % n_layers];
                if (variant == 0) launch_ref(L, dy, dids, dout, NUSED, stream);
                else if (variant == 1) launch_v1(L, dy, dids, dout, NUSED, stream);
                else launch_v2(L, dy, dids, dout, NUSED, stream);
            }
            CHECK(cudaStreamSynchronize(stream));
            CHECK(cudaEventRecord(e0, stream));
            for (int it = 0; it < timing_iters; it++) {
                const dev_layer &L = layers[it % n_layers];
                if (variant == 0) launch_ref(L, dy, dids, dout, NUSED, stream);
                else if (variant == 1) launch_v1(L, dy, dids, dout, NUSED, stream);
                else launch_v2(L, dy, dids, dout, NUSED, stream);
            }
            CHECK(cudaEventRecord(e1, stream));
            CHECK(cudaStreamSynchronize(stream));
            float ms = 0.0f;
            CHECK(cudaEventElapsedTime(&ms, e0, e1));
            const double us = ms * 1000.0 / timing_iters;
            printf("%s: %8.2f us/launch  %7.1f GB/s  (x43 = %.3f ms/step)\n",
                   name, us, gib / (us * 1e-6) / 1e9, us * 43.0 / 1000.0);
        }
        CHECK(cudaEventDestroy(e0));
        CHECK(cudaEventDestroy(e1));
    }

    printf(fails ? "RESULT: FAIL\n" : "RESULT: PASS\n");
    return fails ? 1 : 0;
}
