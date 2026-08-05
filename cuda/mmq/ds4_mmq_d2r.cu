// SPDX-License-Identifier: MIT
// ds4_mmq_d2r.cu - gated D2R Q2_K MoE down-GEMM production path.
//
// The fused gate/up+SwiGLU+Q8 prefill pipeline (FusedGateUpSmem,
// iq2_gateup_fused_mainloop, gateup_iq2_swiglu_q8_d2r_kernel and its
// launcher) is Portions Copyright (c) 2026 Marco Palaferri (MIT), adapted
// from xangel82/DS4-GB10-GX10-DSpark-CUDA commit 910501e (v0.5 inc-9; the
// launch decomposition was re-derived for this fork's serving profile).

#include "ds4_mmq_d2r.cuh"

#include <type_traits>

#include "common.cuh"
#include "mmq.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstdio>

namespace {

// Debug-only fill telemetry for the partial-tile lever (DS4_MMQ_D2R_STATS=1):
// per-launch column/tile fill summary from expert_bounds.  Synchronizes the
// stream - never enable on perf legs.
static bool d2r_stats_enabled() {
    static int cached = -1;
    if (cached < 0) {
        const char *env = getenv("DS4_MMQ_D2R_STATS");
        cached = (env && env[0] == '1') ? 1 : 0;
    }
    return cached != 0;
}

constexpr int kMTile      = 128;
constexpr int kNTile      = 64;
constexpr int kWarps      = 8;
constexpr int kThreads    = 32 * kWarps;
constexpr int kStages     = 2;
constexpr int kNFrag      = kNTile / 8;
constexpr int kRawStages  = 2;  // k256 raw slots; NT=64 stays under 48 KiB.
constexpr int kRawPairsPerWarp = 8;
constexpr int kRawHalves = 2;
constexpr int kRawQSWordsPerHalf = 8;
constexpr int kRawQSPairStride = 9;  // 8 uint2 payload + 1 uint2 pad to rotate banks.
constexpr int kRawQSChunks = kRawHalves * kRawPairsPerWarp * kRawQSWordsPerHalf;
constexpr int kRawSCChunks = kRawHalves * kRawPairsPerWarp;
constexpr int kRawDMChunks = kRawPairsPerWarp;
constexpr int kRawCopyChunks = kRawQSChunks + kRawSCChunks + kRawDMChunks;
constexpr int kIQ2RawRowsPerWarp = 16;
constexpr int kIQ2RawPairsPerRow = 8;
constexpr int kIQ2RawQCodeChunks = kIQ2RawRowsPerWarp * kIQ2RawPairsPerRow;
constexpr int kIQ2RawQCodeTrips = (kIQ2RawQCodeChunks + 31) / 32;
constexpr int kFusedNTile = 32;
constexpr int kQ8PrefetchItems = kNFrag * 8 * 9;
constexpr int kQ8PrefetchTrips = (kQ8PrefetchItems + kThreads - 1) / kThreads;
constexpr int kRawCopyTrips = (kRawCopyChunks + 31) / 32;

static_assert(kNTile == 64, "D2R production path is CFG1 NT64 only");

static void d2r_print_fill_stats(const char *tag, const int32_t *expert_bounds_dev,
                                 int n_experts, int64_t ne_get_rows,
                                 cudaStream_t stream) {
    enum { kMaxExperts = 1024 };
    static int32_t bounds[kMaxExperts + 1];
    if (n_experts > kMaxExperts) return;
    if (cudaStreamSynchronize(stream) != cudaSuccess) return;
    if (cudaMemcpy(bounds, expert_bounds_dev, (size_t)(n_experts + 1) * sizeof(int32_t),
                   cudaMemcpyDeviceToHost) != cudaSuccess) return;
    int live = 0, tiles = 0, full = 0;
    long long cols = 0;
    int tail_hist[8] = {0, 0, 0, 0, 0, 0, 0, 0};  // tail fill 1-8, 9-16, ..., 57-64
    for (int e = 0; e < n_experts; e++) {
        const int count = bounds[e + 1] - bounds[e];
        if (count <= 0) continue;
        live++;
        cols += count;
        tiles += (count + kNTile - 1) / kNTile;
        full += count / kNTile;
        const int tail = count % kNTile;
        if (tail > 0) tail_hist[(tail - 1) / 8]++;
    }
    fprintf(stderr,
            "ds4: D2R fill %s: rows=%lld cols=%lld live=%d/%d tiles=%d full=%d "
            "tail_hist[1-8..57-64]=[%d %d %d %d %d %d %d %d]\n",
            tag, (long long)ne_get_rows, cols, live, n_experts, tiles, full,
            tail_hist[0], tail_hist[1], tail_hist[2], tail_hist[3],
            tail_hist[4], tail_hist[5], tail_hist[6], tail_hist[7]);
}
static_assert(kStages == 2, "D2R raw-ring schedule expects exactly two q8 stages");
static_assert(kThreads == 256, "D2R CTA is fixed at 256 threads");
static_assert(kQ8PrefetchTrips == 3, "unexpected q8 issue trip count");
static_assert(kRawCopyTrips == 5, "unexpected raw-ring issue trip count");
static_assert(kIQ2RawQCodeTrips == 4, "unexpected IQ2 raw-ring issue trip count");

struct alignas(16) SmemInvariants {
    const char *w_base;
    const half *iq2_dq_base;
    const uint2 *iq2_qs_base;
    const char *q8_tile_base;
    float *out;
    uint32_t sc_off_bytes;
    uint32_t qs_off_bytes;
    uint32_t q8_k128_stride_bytes;
    int nb;
    int k128_iters;
    int M;
    int cta_row0;
    int col_lo;
    int col_count;
    union {
        uint32_t warp_pair0_blk[kWarps];
        uint32_t warp_row0_blk[kWarps];
    };
};

static_assert(sizeof(SmemInvariants) <= 128, "shared invariant table must stay small");

[[maybe_unused]] __device__ __forceinline__ uint32_t q2k_dm_bits(
        const uint2 * __restrict__ dm2, uint64_t pblk, int parity) {
    const uint2 d = dm2[pblk];
    return parity ? d.y : d.x;
}

__device__ __forceinline__ float2 half2_bits_to_float2(uint32_t bits) {
    half2 h;
    *reinterpret_cast<uint32_t *>(&h) = bits;
    return __half22float2(h);
}

[[maybe_unused]] __device__ __forceinline__ uint8_t q2k_scale_byte(
        const int4 * __restrict__ sc4, uint64_t pblk, int parity, int sub16) {
    const int w = sub16 >> 2;
    const int4 s = sc4[pblk * 2ull + (uint64_t)(w >> 1)];
    const uint32_t sw = parity ? ((w & 1) ? (uint32_t)s.w : (uint32_t)s.z)
                               : ((w & 1) ? (uint32_t)s.y : (uint32_t)s.x);
    return (uint8_t)((sw >> (8 * (sub16 & 3))) & 0xFFu);
}

[[maybe_unused]] __device__ __forceinline__ void q2k_scale_packs_for_half(
        const int4 * __restrict__ sc4, uint64_t pblk, int parity, int half,
        uint32_t &pack_lo4, uint32_t &pack_hi4) {
    const int4 s = sc4[pblk * 2ull + (uint64_t)half];
    if (parity) {
        pack_lo4 = (uint32_t)s.z;
        pack_hi4 = (uint32_t)s.w;
    } else {
        pack_lo4 = (uint32_t)s.x;
        pack_hi4 = (uint32_t)s.y;
    }
}

__device__ __forceinline__ void q2k_scale_packs_from_int4(
        int4 s, int parity, uint32_t &pack_lo4, uint32_t &pack_hi4) {
    if (parity) {
        pack_lo4 = (uint32_t)s.z;
        pack_hi4 = (uint32_t)s.w;
    } else {
        pack_lo4 = (uint32_t)s.x;
        pack_hi4 = (uint32_t)s.y;
    }
}

[[maybe_unused]] __device__ __forceinline__ uint8_t q2k_scale_from_packs(
        uint32_t pack_lo4, uint32_t pack_hi4, int t, int sub_in_k32) {
    const int sub = 2 * t + sub_in_k32;
    const uint32_t pack = (sub < 4) ? pack_lo4 : pack_hi4;
    return (uint8_t)((pack >> (8 * (sub & 3))) & 0xFFu);
}

template <int T, int SubInK32>
__device__ __forceinline__ uint8_t q2k_scale_from_packs_t(uint32_t pack_lo4, uint32_t pack_hi4) {
    constexpr int sub = 2 * T + SubInK32;
    const uint32_t pack = (sub < 4) ? pack_lo4 : pack_hi4;
    return (uint8_t)((pack >> (8 * (sub & 3))) & 0xFFu);
}

[[maybe_unused]] __device__ __forceinline__ uint32_t q2k_decode_scaled_reg(
        uint32_t word, int t, uint8_t scale_byte) {
    const uint32_t q = (word >> (2 * t)) & 0x03030303u;
    return q * (uint32_t)(scale_byte & 0x0Fu);
}

template <int T>
__device__ __forceinline__ uint32_t q2k_decode_scaled_reg_t(uint32_t word, uint8_t scale_byte) {
    const uint32_t q = (word >> (2 * T)) & 0x03030303u;
    return q * (uint32_t)(scale_byte & 0x0Fu);
}

__device__ __forceinline__ int sum_i8x4(uint32_t v) {
    return (int)(int8_t)(v >>  0) +
           (int)(int8_t)(v >>  8) +
           (int)(int8_t)(v >> 16) +
           (int)(int8_t)(v >> 24);
}

__device__ __forceinline__ int q8_sum16_words(const block_q8_1_mmq &b, int k0) {
    const uint32_t *p = reinterpret_cast<const uint32_t *>(b.qs + k0);
    return sum_i8x4(p[0]) + sum_i8x4(p[1]) + sum_i8x4(p[2]) + sum_i8x4(p[3]);
}

__device__ __forceinline__ void zero_16B(void *dst) {
    int4 z = make_int4(0, 0, 0, 0);
    *reinterpret_cast<int4 *>(dst) = z;
}

__device__ __forceinline__ void zero_8B(void *dst) {
    uint2 z = make_uint2(0, 0);
    *reinterpret_cast<uint2 *>(dst) = z;
}

__device__ __forceinline__ void cp_async_16B(void *dst, const void *src, bool pred) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    if (pred) {
        const unsigned smem = static_cast<unsigned>(__cvta_generic_to_shared(dst));
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
                     :: "r"(smem), "l"(src));
    } else {
        zero_16B(dst);
    }
#else
    if (pred) {
        *reinterpret_cast<int4 *>(dst) = *reinterpret_cast<const int4 *>(src);
    } else {
        zero_16B(dst);
    }
#endif
}

__device__ __forceinline__ void cp_async_8B(void *dst, const void *src, bool pred) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    if (pred) {
        const unsigned smem = static_cast<unsigned>(__cvta_generic_to_shared(dst));
        asm volatile("cp.async.ca.shared.global [%0], [%1], 8;"
                     :: "r"(smem), "l"(src));
    } else {
        zero_8B(dst);
    }
#else
    if (pred) {
        *reinterpret_cast<uint2 *>(dst) = *reinterpret_cast<const uint2 *>(src);
    } else {
        zero_8B(dst);
    }
#endif
}

__device__ __forceinline__ void cp_async_commit() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;");
#endif
}

template <int KeepGroups>
__device__ __forceinline__ void cp_async_wait_group() {
    static_assert(KeepGroups >= 0 && KeepGroups <= 7, "bad cp.async wait_group depth");
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group %0;" :: "n"(KeepGroups));
#endif
}

__device__ __forceinline__ void cp_async_wait_keep(int keep_groups) {
    switch (keep_groups) {
        case 0: cp_async_wait_group<0>(); break;
        case 1: cp_async_wait_group<1>(); break;
        case 2: cp_async_wait_group<2>(); break;
        case 3: cp_async_wait_group<3>(); break;
        default: cp_async_wait_group<4>(); break;
    }
}

[[maybe_unused]] __device__ __forceinline__ int cp_async_keep_for_tile(
        int tile_iter, int k_iters) {
    int keep = kStages - 1;
    const int newer = k_iters - tile_iter - 1;
    if (keep > newer) {
        keep = newer;
    }
    return keep > 0 ? keep : 0;
}

__device__ __forceinline__ int d2r_lane() {
#if defined(__CUDA_ARCH__)
    uint32_t lane;
    asm volatile("mov.u32 %0, %%tid.x;" : "=r"(lane));
    return (int)lane;
#else
    return (int)threadIdx.x;
#endif
}

__device__ __forceinline__ int d2r_warp() {
#if defined(__CUDA_ARCH__)
    uint32_t warp;
    asm volatile("mov.u32 %0, %%tid.y;" : "=r"(warp));
    return (int)warp;
#else
    return (int)threadIdx.y;
#endif
}

__device__ __forceinline__ int d2r_tid() {
    return (d2r_warp() << 5) | d2r_lane();
}

__device__ __forceinline__ int d2r_group() {
    return d2r_lane() >> 2;
}

__device__ __forceinline__ int d2r_tig() {
    return d2r_lane() & 3;
}

__device__ __forceinline__ int d2r_q8_stage(int k128_iter) {
    return k128_iter & (kStages - 1);
}

__device__ __forceinline__ int d2r_raw_stage(int k256_iter) {
    return k256_iter & (kRawStages - 1);
}

template <bool FullTile>
__device__ __forceinline__ void issue_q8_prefetch_one(
        block_q8_1_mmq (&s_q8)[kStages][kNFrag][8],
        const char * __restrict__ q8_iter_base,
        int col_count, int stage, int t) {
    constexpr int cols = kNFrag * 8;
    static_assert(cols == 64, "NT64 q8 prefetch mapping expects 64 columns");
    const int col_local = t & (cols - 1);
    const int chunk = t >> 6;
    const int nf = col_local >> 3;
    const int c = col_local & 7;
    const bool valid = FullTile ? true : (col_local < col_count);
    void *dst = (char *)&s_q8[stage][nf][c] + chunk * 16;
    const void *src = q8_iter_base + (uint64_t)col_local * sizeof(block_q8_1_mmq) + chunk * 16;
    cp_async_16B(dst, src, valid);
}

template <bool FullTile, int Iter>
__device__ __forceinline__ void issue_q8_prefetch_unrolled(
        block_q8_1_mmq (&s_q8)[kStages][kNFrag][8],
        const char * __restrict__ q8_iter_base,
        int col_count, int stage, int tid) {
    if constexpr (Iter < kQ8PrefetchTrips) {
        const int t = tid + Iter * kThreads;
        if constexpr ((Iter + 1) * kThreads <= kQ8PrefetchItems) {
            issue_q8_prefetch_one<FullTile>(s_q8, q8_iter_base, col_count, stage, t);
        } else {
            if (t < kQ8PrefetchItems) {
                issue_q8_prefetch_one<FullTile>(s_q8, q8_iter_base, col_count, stage, t);
            }
        }
        issue_q8_prefetch_unrolled<FullTile, Iter + 1>(
            s_q8, q8_iter_base, col_count, stage, tid);
    }
}

template <bool FullTile>
__device__ __forceinline__ void issue_q8_prefetch(
        block_q8_1_mmq (&s_q8)[kStages][kNFrag][8],
        const volatile SmemInvariants &s_inv,
        int stage, int k128_iter, int tid) {
    const char *q8_tile_base = s_inv.q8_tile_base;
    const uint32_t k128_stride = s_inv.q8_k128_stride_bytes;
    int col_count = kNTile;
    if constexpr (!FullTile) {
        col_count = s_inv.col_count;
    }
    const char *q8_iter_base = q8_tile_base + (uint64_t)k128_iter * (uint64_t)k128_stride;
    issue_q8_prefetch_unrolled<FullTile, 0>(s_q8, q8_iter_base, col_count, stage, tid);
    cp_async_commit();
}

__device__ __forceinline__ void issue_q8_prefetch_one_fast(
        block_q8_1_mmq (&s_q8)[kStages][kNFrag][8],
        const char * __restrict__ q8_iter_base,
        int stage, int t) {
    constexpr int cols = kNFrag * 8;
    static_assert(cols == 64, "NT64 q8 prefetch mapping expects 64 columns");
    const int col_local = t & (cols - 1);
    const int c = col_local & 7;
    const int nf = col_local >> 3;
    const int chunk = t >> 6;
    void *dst = (char *)&s_q8[stage][nf][c] + chunk * 16;
    const void *src = q8_iter_base + (uint64_t)col_local * sizeof(block_q8_1_mmq) + chunk * 16;
    cp_async_16B(dst, src, true);
}

template <int Iter>
__device__ __forceinline__ void issue_q8_prefetch_fast_unrolled(
        block_q8_1_mmq (&s_q8)[kStages][kNFrag][8],
        const char * __restrict__ q8_iter_base,
        int stage, int tid) {
    if constexpr (Iter < kQ8PrefetchTrips) {
        const int t = tid + Iter * kThreads;
        if constexpr ((Iter + 1) * kThreads <= kQ8PrefetchItems) {
            issue_q8_prefetch_one_fast(s_q8, q8_iter_base, stage, t);
        } else {
            if (t < kQ8PrefetchItems) {
                issue_q8_prefetch_one_fast(s_q8, q8_iter_base, stage, t);
            }
        }
        issue_q8_prefetch_fast_unrolled<Iter + 1>(s_q8, q8_iter_base, stage, tid);
    }
}

__device__ __forceinline__ void issue_q8_prefetch_fast(
        block_q8_1_mmq (&s_q8)[kStages][kNFrag][8],
        const volatile SmemInvariants &s_inv,
        int stage, int k128_iter) {
    const char *q8_iter_base =
        s_inv.q8_tile_base + (uint64_t)k128_iter * (uint64_t)s_inv.q8_k128_stride_bytes;
    issue_q8_prefetch_fast_unrolled<0>(s_q8, q8_iter_base, stage, d2r_tid());
    cp_async_commit();
}

/* flat-pool p5b: the column's Y row byte offset arrives as a per-thread
 * scalar (y_off) instead of being derived from the column slot.  col_local
 * is trip-invariant across the unrolled cp.async issue trips (kThreads is a
 * multiple of every fused TileN), so one register carries the offset for
 * the whole mainloop.  Identity staging (Y row == assignment slot) passes
 * col_local * sizeof(block_q8_1_mmq) and compiles to the pre-p5b address
 * math; indirect staging passes ids_src[col] * sizeof(block_q8_1_mmq)
 * against a token-compact buffer. */
template <bool FullTile, int NFrag>
__device__ __forceinline__ void issue_fused_q8_prefetch_one(
        block_q8_1_mmq (&s_q8)[kStages][NFrag][8],
        const char * __restrict__ q8_iter_base,
        uint32_t y_off, int col_count, int stage, int t) {
    constexpr int TileN = NFrag * 8;
    static_assert(TileN == 8 || TileN == 16 || TileN == 32,
                  "unsupported fused D2R tile width");
    static_assert(kThreads % TileN == 0,
                  "fused q8 y_off expects trip-invariant col_local");
    const int col_local = t & (TileN - 1);
    const int chunk = t / TileN;
    const int nf = col_local >> 3;
    const int c = col_local & 7;
    const bool valid = FullTile ? true : (col_local < col_count);
    void *dst = (char *)&s_q8[stage][nf][c] + chunk * 16;
    const void *src = q8_iter_base + (uint64_t)y_off + chunk * 16;
    cp_async_16B(dst, src, valid);
}

template <bool FullTile, int NFrag, int Iter>
__device__ __forceinline__ void issue_fused_q8_prefetch_unrolled(
        block_q8_1_mmq (&s_q8)[kStages][NFrag][8],
        const char * __restrict__ q8_iter_base,
        uint32_t y_off, int col_count, int stage, int tid) {
    constexpr int Items = NFrag * 8 * 9;
    constexpr int Trips = (Items + kThreads - 1) / kThreads;
    if constexpr (Iter < Trips) {
        const int t = tid + Iter * kThreads;
        if constexpr ((Iter + 1) * kThreads <= Items) {
            issue_fused_q8_prefetch_one<FullTile, NFrag>(
                s_q8, q8_iter_base, y_off, col_count, stage, t);
        } else if (t < Items) {
            issue_fused_q8_prefetch_one<FullTile, NFrag>(
                s_q8, q8_iter_base, y_off, col_count, stage, t);
        }
        issue_fused_q8_prefetch_unrolled<FullTile, NFrag, Iter + 1>(
            s_q8, q8_iter_base, y_off, col_count, stage, tid);
    }
}

template <bool FullTile, int NFrag>
__device__ __forceinline__ void issue_fused_q8_prefetch(
        block_q8_1_mmq (&s_q8)[kStages][NFrag][8],
        const volatile SmemInvariants &s_inv,
        uint32_t y_off, int stage, int k128_iter) {
    constexpr int TileN = NFrag * 8;
    const char *q8_iter_base =
        s_inv.q8_tile_base +
        (uint64_t)k128_iter * (uint64_t)s_inv.q8_k128_stride_bytes;
    const int col_count = FullTile ? TileN : s_inv.col_count;
    issue_fused_q8_prefetch_unrolled<FullTile, NFrag, 0>(
        s_q8, q8_iter_base, y_off, col_count, stage, d2r_tid());
    cp_async_commit();
}

/* Identity-staged entry (Q2_K down tail tiles keep slot == Y row). */
template <bool FullTile, int NFrag>
__device__ __forceinline__ void issue_fused_q8_prefetch(
        block_q8_1_mmq (&s_q8)[kStages][NFrag][8],
        const volatile SmemInvariants &s_inv,
        int stage, int k128_iter) {
    constexpr int TileN = NFrag * 8;
    const uint32_t y_off = (uint32_t)(d2r_tid() & (TileN - 1)) *
                           (uint32_t)sizeof(block_q8_1_mmq);
    issue_fused_q8_prefetch<FullTile, NFrag>(s_q8, s_inv, y_off, stage, k128_iter);
}

struct Q8ColFixF32 {
    float d8[2];
    float sum[8];
};

template <int NFrag>
__device__ __forceinline__ void publish_q8_fix_f32(
        Q8ColFixF32 (&s_q8_fix)[kStages][NFrag][8],
        const block_q8_1_mmq (&s_q8)[kStages][NFrag][8],
        int stage, int tid) {
    if (tid < NFrag * 8) {
        const int col_local = tid;
        const int nf = col_local >> 3;
        const int c = col_local & 7;
        const block_q8_1_mmq &qb = s_q8[stage][nf][c];
        const uint4 d2s6 = *reinterpret_cast<const uint4 *>(qb.d2s6);
        const float2 d01 = half2_bits_to_float2(d2s6.x);
        const float2 s01 = half2_bits_to_float2(d2s6.y);
        const float2 s23 = half2_bits_to_float2(d2s6.z);
        const float2 s45 = half2_bits_to_float2(d2s6.w);
        Q8ColFixF32 &f = s_q8_fix[stage][nf][c];
        f.d8[0] = d01.x;
        f.d8[1] = d01.y;
        f.sum[0] = s01.x;
        f.sum[1] = s01.y;
        f.sum[2] = s23.x;
        f.sum[3] = s23.y;
        f.sum[4] = s45.x;
        f.sum[5] = s45.y;
        f.sum[6] = d01.y * (float)q8_sum16_words(qb, 96);
        f.sum[7] = d01.y * (float)q8_sum16_words(qb, 112);
    }
}

template <int NFrag>
__device__ __forceinline__ void publish_q8_fix_f32_guarded(
        Q8ColFixF32 (&s_q8_fix)[kStages][NFrag][8],
        const block_q8_1_mmq (&s_q8)[kStages][NFrag][8],
        int stage, int tid, int col_count) {
    if (tid < col_count) {
        const int col_local = tid;
        const int nf = col_local >> 3;
        const int c = col_local & 7;
        const block_q8_1_mmq &qb = s_q8[stage][nf][c];
        const uint4 d2s6 = *reinterpret_cast<const uint4 *>(qb.d2s6);
        const float2 d01 = half2_bits_to_float2(d2s6.x);
        const float2 s01 = half2_bits_to_float2(d2s6.y);
        const float2 s23 = half2_bits_to_float2(d2s6.z);
        const float2 s45 = half2_bits_to_float2(d2s6.w);
        Q8ColFixF32 &f = s_q8_fix[stage][nf][c];
        f.d8[0] = d01.x;
        f.d8[1] = d01.y;
        f.sum[0] = s01.x;
        f.sum[1] = s01.y;
        f.sum[2] = s23.x;
        f.sum[3] = s23.y;
        f.sum[4] = s45.x;
        f.sum[5] = s45.y;
        f.sum[6] = d01.y * (float)q8_sum16_words(qb, 96);
        f.sum[7] = d01.y * (float)q8_sum16_words(qb, 112);
    }
}

struct Q8Fix2 {
    float d8;
    float sum0;
    float sum1;
};

struct Q8Fix4 {
    float d8;
    float sum0;
    float sum1;
    float sum2;
    float sum3;
};

template <int NFrag>
__device__ __forceinline__ Q8Fix2 load_q8_fix2(
        const Q8ColFixF32 (&s_q8_fix)[kStages][NFrag][8],
        int stage, int nf, int c, int k_in_q8) {
    const Q8ColFixF32 &sf = s_q8_fix[stage][nf][c];
    const int d8_slot = (k_in_q8 >= 64) ? 1 : 0;
    const int sub128 = k_in_q8 >> 4;
    Q8Fix2 f;
    f.d8 = sf.d8[d8_slot];
    f.sum0 = sf.sum[sub128 + 0];
    f.sum1 = sf.sum[sub128 + 1];
    return f;
}

template <int NFrag>
__device__ __forceinline__ Q8Fix4 load_q8_fix4(
        const Q8ColFixF32 (&s_q8_fix)[kStages][NFrag][8],
        int stage, int nf, int c, int k_in_q8_pair) {
    const Q8ColFixF32 &sf = s_q8_fix[stage][nf][c];
    const int d8_slot = (k_in_q8_pair >= 64) ? 1 : 0;
    const int sub128 = k_in_q8_pair >> 4;
    Q8Fix4 f;
    f.d8 = sf.d8[d8_slot];
    f.sum0 = sf.sum[sub128 + 0];
    f.sum1 = sf.sum[sub128 + 1];
    f.sum2 = sf.sum[sub128 + 2];
    f.sum3 = sf.sum[sub128 + 3];
    return f;
}

struct Q2KWeightHalf {
    uint32_t q_r0_c0;
    uint32_t q_r1_c0;
    uint32_t q_r0_c1;
    uint32_t q_r1_c1;
    uint32_t sc_r0_lo4;
    uint32_t sc_r0_hi4;
    uint32_t sc_r1_lo4;
    uint32_t sc_r1_hi4;
    uint32_t dm_r0;
    uint32_t dm_r1;
};

struct alignas(16) Q2KRawWarpStage {
    uint2 qs[kRawHalves][kRawPairsPerWarp][kRawQSPairStride];
    int4 sc[kRawHalves][kRawPairsPerWarp];
    uint2 dm[kRawPairsPerWarp];
};

static_assert(sizeof(Q2KRawWarpStage) ==
              kRawHalves * kRawPairsPerWarp * kRawQSPairStride * sizeof(uint2) +
              kRawHalves * kRawPairsPerWarp * sizeof(int4) +
              kRawPairsPerWarp * sizeof(uint2),
              "unexpected Q2K raw ring stage size");

struct alignas(16) IQ2RawWarpStage {
    uint2 qs[kIQ2RawRowsPerWarp][kIQ2RawPairsPerRow];
    half dq[kIQ2RawRowsPerWarp];
};

static_assert(sizeof(IQ2RawWarpStage) ==
              kIQ2RawRowsPerWarp * kIQ2RawPairsPerRow * sizeof(uint2) +
              kIQ2RawRowsPerWarp * sizeof(half),
              "unexpected IQ2 raw ring stage size");

template <int TileN>
struct alignas(16) FusedGateUpComputeSmem {
    static_assert(TileN == 8 || TileN == 16 || TileN == 32,
                  "unsupported fused D2R tile width");
    block_q8_1_mmq q8[kStages][TileN / 8][8];
    IQ2RawWarpStage raw[2][kWarps][kRawStages];
    uint2 grid[256];
    volatile SmemInvariants inv[2];
};

template <int TileN>
union alignas(16) FusedGateUpSmem {
    FusedGateUpComputeSmem<TileN> compute;
    float mid[TileN][kMTile];
};

static_assert(sizeof(FusedGateUpComputeSmem<kFusedNTile>) <= 48ull * 1024ull,
              "fused IQ2 gate/up shared memory exceeds 48 KiB");
static_assert(sizeof(FusedGateUpSmem<kFusedNTile>) ==
                  sizeof(FusedGateUpComputeSmem<kFusedNTile>),
              "fused post-MMA staging unexpectedly grows shared memory");
static_assert(2ull * (sizeof(FusedGateUpSmem<kFusedNTile>) +
                      kFusedNTile * sizeof(float)) <= 90ull * 1024ull,
              "fused IQ2 gate/up no longer permits two CTAs per GB10 SM");

constexpr size_t kSmemQ8StageBytes = (size_t)kNFrag * 8 * sizeof(block_q8_1_mmq);
constexpr size_t kSmemTailStageBytes = (size_t)kNFrag * 8 * sizeof(Q8ColFixF32);
constexpr size_t kSmemRawBytes = (size_t)kWarps * kRawStages * sizeof(Q2KRawWarpStage);
constexpr size_t kSmemInvBytes = sizeof(SmemInvariants);
constexpr size_t kSmemStaticBytes = (size_t)kStages * (kSmemQ8StageBytes + kSmemTailStageBytes) +
                                    kSmemRawBytes + kSmemInvBytes;
static_assert(kSmemStaticBytes <= 48ull * 1024ull, "D2R static shared memory exceeds 48 KiB");
constexpr size_t kSmemIQ2RawBytes = (size_t)kWarps * kRawStages * sizeof(IQ2RawWarpStage);
constexpr size_t kSmemIQ2GridBytes = 256u * sizeof(uint2);
constexpr size_t kSmemIQ2StaticBytes = (size_t)kStages * kSmemQ8StageBytes +
                                       kSmemIQ2RawBytes + kSmemIQ2GridBytes + kSmemInvBytes;
static_assert(kSmemIQ2StaticBytes <= 48ull * 1024ull,
              "IQ2 D2R static shared memory exceeds 48 KiB");


__device__ __forceinline__ uint32_t q2k_select_parity(uint2 v, int parity) {
    return parity ? v.y : v.x;
}

__device__ __forceinline__ bool q2k_raw_pair_valid(int warp_row0, int pair, int M) {
    return warp_row0 + 2 * pair < M;
}

template <bool FullTile, int Iter>
__device__ __forceinline__ void issue_q2k_raw_prefetch_iter(
        Q2KRawWarpStage (&s_raw)[kWarps][kRawStages],
        const char * __restrict__ w_base,
        uint32_t sc_off_bytes, uint32_t qs_off_bytes,
        uint32_t warp_pair0_blk, int nb, int raw_stage, int k256_iter,
        int warp_row0, int M, int warp, int lane) {
    static_assert(kRawCopyTrips == 5, "raw prefetch iter specialization expects five trips");
    if constexpr (Iter < 4) {
        constexpr int chunks_per_half = kRawPairsPerWarp * kRawQSWordsPerHalf;
        constexpr int half = (Iter * 32) / chunks_per_half;
        const int t = lane + Iter * 32;
        const int rem = t - half * chunks_per_half;
        const int pair = rem / kRawQSWordsPerHalf;
        const int word = rem & (kRawQSWordsPerHalf - 1);
        const bool row_valid = FullTile ? true : q2k_raw_pair_valid(warp_row0, pair, M);
        const bool valid = k256_iter < nb && row_valid;
        const uint64_t pblk = valid ? ((uint64_t)warp_pair0_blk + (uint64_t)pair * (uint64_t)nb +
                                       (uint64_t)k256_iter) : 0ull;
        void *dst = &s_raw[warp][raw_stage].qs[half][pair][word];
        const void *src = w_base + (uint64_t)qs_off_bytes +
                          pblk * 16ull * sizeof(uint2) +
                          (uint64_t)half * 8ull * sizeof(uint2) +
                          (uint64_t)word * sizeof(uint2);
        cp_async_8B(dst, src, valid);
    } else {
        if (lane < kRawSCChunks) {
            const int half = lane / kRawPairsPerWarp;
            const int pair = lane - half * kRawPairsPerWarp;
            const bool row_valid = FullTile ? true : q2k_raw_pair_valid(warp_row0, pair, M);
            const bool valid = k256_iter < nb && row_valid;
            const uint64_t pblk = valid ? ((uint64_t)warp_pair0_blk + (uint64_t)pair * (uint64_t)nb +
                                           (uint64_t)k256_iter) : 0ull;
            void *dst = &s_raw[warp][raw_stage].sc[half][pair];
            const void *src = w_base + (uint64_t)sc_off_bytes +
                              pblk * 2ull * sizeof(int4) +
                              (uint64_t)half * sizeof(int4);
            cp_async_16B(dst, src, valid);
        } else if (lane < kRawSCChunks + kRawDMChunks) {
            const int pair = lane - kRawSCChunks;
            const bool row_valid = FullTile ? true : q2k_raw_pair_valid(warp_row0, pair, M);
            const bool valid = k256_iter < nb && row_valid;
            const uint64_t pblk = valid ? ((uint64_t)warp_pair0_blk + (uint64_t)pair * (uint64_t)nb +
                                           (uint64_t)k256_iter) : 0ull;
            void *dst = &s_raw[warp][raw_stage].dm[pair];
            const void *src = w_base + pblk * sizeof(uint2);
            cp_async_8B(dst, src, valid);
        }
    }
}

template <bool FullTile, int Iter>
__device__ __forceinline__ void issue_q2k_raw_prefetch_unrolled(
        Q2KRawWarpStage (&s_raw)[kWarps][kRawStages],
        const char * __restrict__ w_base,
        uint32_t sc_off_bytes, uint32_t qs_off_bytes,
        uint32_t warp_pair0_blk, int nb, int raw_stage, int k256_iter,
        int warp_row0, int M, int warp, int lane) {
    if constexpr (Iter < kRawCopyTrips) {
        issue_q2k_raw_prefetch_iter<FullTile, Iter>(
            s_raw, w_base, sc_off_bytes, qs_off_bytes, warp_pair0_blk, nb, raw_stage, k256_iter,
            warp_row0, M, warp, lane);
        issue_q2k_raw_prefetch_unrolled<FullTile, Iter + 1>(
            s_raw, w_base, sc_off_bytes, qs_off_bytes, warp_pair0_blk, nb, raw_stage, k256_iter,
            warp_row0, M, warp, lane);
    }
}

template <bool FullTile>
__device__ __forceinline__ void issue_q2k_raw_prefetch(
        Q2KRawWarpStage (&s_raw)[kWarps][kRawStages],
        const volatile SmemInvariants &s_inv,
        int raw_stage, int k256_iter, int warp, int lane) {
    const char *w_base = s_inv.w_base;
    const uint32_t sc_off_bytes = s_inv.sc_off_bytes;
    const uint32_t qs_off_bytes = s_inv.qs_off_bytes;
    const uint32_t warp_pair0_blk = s_inv.warp_pair0_blk[warp];
    const int nb = s_inv.nb;
    int warp_row0 = 0;
    int M = 0;
    if constexpr (!FullTile) {
        warp_row0 = s_inv.cta_row0 + (warp << 4);
        M = s_inv.M;
    }
    issue_q2k_raw_prefetch_unrolled<FullTile, 0>(
        s_raw, w_base, sc_off_bytes, qs_off_bytes, warp_pair0_blk, nb, raw_stage, k256_iter,
        warp_row0, M, warp, lane);
    cp_async_commit();
}

template <int Iter>
__device__ __forceinline__ void issue_q2k_raw_prefetch_iter_fast(
        Q2KRawWarpStage (&s_raw)[kWarps][kRawStages],
        const char * __restrict__ w_base,
        uint32_t sc_off_bytes, uint32_t qs_off_bytes,
        uint32_t warp_pair0_blk, int nb, int raw_stage, int k256_iter) {
    static_assert(kRawCopyTrips == 5, "raw prefetch iter specialization expects five trips");
    const int warp = d2r_warp();
    const int lane = d2r_lane();
    if constexpr (Iter < 4) {
        constexpr int chunks_per_half = kRawPairsPerWarp * kRawQSWordsPerHalf;
        constexpr int half = (Iter * 32) / chunks_per_half;
        const int t = lane + Iter * 32;
        const int rem = t - half * chunks_per_half;
        const int pair = rem / kRawQSWordsPerHalf;
        const int word = rem & (kRawQSWordsPerHalf - 1);
        const bool valid = k256_iter < nb;
        const uint64_t pblk = valid ? ((uint64_t)warp_pair0_blk + (uint64_t)pair * (uint64_t)nb +
                                       (uint64_t)k256_iter) : 0ull;
        void *dst = &s_raw[warp][raw_stage].qs[half][pair][word];
        const void *src = w_base + (uint64_t)qs_off_bytes +
                          pblk * 16ull * sizeof(uint2) +
                          (uint64_t)half * 8ull * sizeof(uint2) +
                          (uint64_t)word * sizeof(uint2);
        cp_async_8B(dst, src, valid);
    } else {
        if (lane < kRawSCChunks) {
            const int half = lane / kRawPairsPerWarp;
            const int pair = lane - half * kRawPairsPerWarp;
            const bool valid = k256_iter < nb;
            const uint64_t pblk = valid ? ((uint64_t)warp_pair0_blk + (uint64_t)pair * (uint64_t)nb +
                                           (uint64_t)k256_iter) : 0ull;
            void *dst = &s_raw[warp][raw_stage].sc[half][pair];
            const void *src = w_base + (uint64_t)sc_off_bytes +
                              pblk * 2ull * sizeof(int4) +
                              (uint64_t)half * sizeof(int4);
            cp_async_16B(dst, src, valid);
        } else if (lane < kRawSCChunks + kRawDMChunks) {
            const int pair = lane - kRawSCChunks;
            const bool valid = k256_iter < nb;
            const uint64_t pblk = valid ? ((uint64_t)warp_pair0_blk + (uint64_t)pair * (uint64_t)nb +
                                           (uint64_t)k256_iter) : 0ull;
            void *dst = &s_raw[warp][raw_stage].dm[pair];
            const void *src = w_base + pblk * sizeof(uint2);
            cp_async_8B(dst, src, valid);
        }
    }
}

template <int Iter>
__device__ __forceinline__ void issue_q2k_raw_prefetch_fast_unrolled(
        Q2KRawWarpStage (&s_raw)[kWarps][kRawStages],
        const char * __restrict__ w_base,
        uint32_t sc_off_bytes, uint32_t qs_off_bytes,
        uint32_t warp_pair0_blk, int nb, int raw_stage, int k256_iter) {
    if constexpr (Iter < kRawCopyTrips) {
        issue_q2k_raw_prefetch_iter_fast<Iter>(
            s_raw, w_base, sc_off_bytes, qs_off_bytes, warp_pair0_blk, nb, raw_stage, k256_iter);
        issue_q2k_raw_prefetch_fast_unrolled<Iter + 1>(
            s_raw, w_base, sc_off_bytes, qs_off_bytes, warp_pair0_blk, nb, raw_stage, k256_iter);
    }
}

__device__ __forceinline__ void issue_q2k_raw_prefetch_fast(
        Q2KRawWarpStage (&s_raw)[kWarps][kRawStages],
        const volatile SmemInvariants &s_inv,
        int raw_stage, int k256_iter) {
    const int warp = d2r_warp();
    issue_q2k_raw_prefetch_fast_unrolled<0>(
        s_raw, s_inv.w_base, s_inv.sc_off_bytes, s_inv.qs_off_bytes,
        s_inv.warp_pair0_blk[warp], s_inv.nb, raw_stage, k256_iter);
    cp_async_commit();
}

__device__ __forceinline__ void load_q2k_weight_half_raw(
        Q2KWeightHalf &w,
        const Q2KRawWarpStage (&s_raw)[kWarps][kRawStages],
        int warp, int raw_stage, bool row0_ok, bool row1_ok,
        int parity, int half, int group, int tig) {
    const Q2KRawWarpStage &raw = s_raw[warp][raw_stage];
    const int pair0 = group >> 1;
    const int pair1 = pair0 + 4;
    const int p_base = half * 8;
    const int p0 = p_base + tig;
    const int p1 = p_base + tig + 4;

    if (row0_ok) {
        const uint2 q0 = raw.qs[half][pair0][p0 - p_base];
        const uint2 q1 = raw.qs[half][pair0][p1 - p_base];
        w.q_r0_c0 = q2k_select_parity(q0, parity);
        w.q_r0_c1 = q2k_select_parity(q1, parity);
        q2k_scale_packs_from_int4(raw.sc[half][pair0], parity, w.sc_r0_lo4, w.sc_r0_hi4);
        w.dm_r0 = q2k_select_parity(raw.dm[pair0], parity);
    } else {
        w.q_r0_c0 = 0;
        w.q_r0_c1 = 0;
        w.sc_r0_lo4 = 0;
        w.sc_r0_hi4 = 0;
        w.dm_r0 = 0;
    }
    if (row1_ok) {
        const uint2 q0 = raw.qs[half][pair1][p0 - p_base];
        const uint2 q1 = raw.qs[half][pair1][p1 - p_base];
        w.q_r1_c0 = q2k_select_parity(q0, parity);
        w.q_r1_c1 = q2k_select_parity(q1, parity);
        q2k_scale_packs_from_int4(raw.sc[half][pair1], parity, w.sc_r1_lo4, w.sc_r1_hi4);
        w.dm_r1 = q2k_select_parity(raw.dm[pair1], parity);
    } else {
        w.q_r1_c0 = 0;
        w.q_r1_c1 = 0;
        w.sc_r1_lo4 = 0;
        w.sc_r1_hi4 = 0;
        w.dm_r1 = 0;
    }
}

__device__ __forceinline__ void load_q2k_weight_half_raw_fast(
        Q2KWeightHalf &w,
        const Q2KRawWarpStage (&s_raw)[kWarps][kRawStages],
        int k128_iter) {
    const int lane = d2r_lane();
    const int group = lane >> 2;
    load_q2k_weight_half_raw(
        w, s_raw, d2r_warp(), d2r_raw_stage(k128_iter >> 1),
        true, true, group & 1, k128_iter & 1, group, lane & 3);
}

template <int T, typename TileA>
__device__ __forceinline__ void make_A_tile(TileA &A, const Q2KWeightHalf &w) {
    const uint8_t sc00 = q2k_scale_from_packs_t<T, 0>(w.sc_r0_lo4, w.sc_r0_hi4);
    const uint8_t sc01 = q2k_scale_from_packs_t<T, 1>(w.sc_r0_lo4, w.sc_r0_hi4);
    const uint8_t sc10 = q2k_scale_from_packs_t<T, 0>(w.sc_r1_lo4, w.sc_r1_hi4);
    const uint8_t sc11 = q2k_scale_from_packs_t<T, 1>(w.sc_r1_lo4, w.sc_r1_hi4);
    A.x[0] = (int)q2k_decode_scaled_reg_t<T>(w.q_r0_c0, sc00);
    A.x[1] = (int)q2k_decode_scaled_reg_t<T>(w.q_r1_c0, sc10);
    A.x[2] = (int)q2k_decode_scaled_reg_t<T>(w.q_r0_c1, sc01);
    A.x[3] = (int)q2k_decode_scaled_reg_t<T>(w.q_r1_c1, sc11);
}

template <int T>
__device__ __forceinline__ uint16_t min_pack_for_t(uint32_t pack_lo4, uint32_t pack_hi4) {
    const uint8_t sc0 = q2k_scale_from_packs_t<T, 0>(pack_lo4, pack_hi4);
    const uint8_t sc1 = q2k_scale_from_packs_t<T, 1>(pack_lo4, pack_hi4);
    return (uint16_t)((uint16_t)(sc0 >> 4) | ((uint16_t)(sc1 >> 4) << 8));
}

template <int T0, int T1>
__device__ __forceinline__ uint32_t min_pack4_for_pair(uint32_t pack_lo4, uint32_t pack_hi4) {
    const uint8_t sc0 = q2k_scale_from_packs_t<T0, 0>(pack_lo4, pack_hi4);
    const uint8_t sc1 = q2k_scale_from_packs_t<T0, 1>(pack_lo4, pack_hi4);
    const uint8_t sc2 = q2k_scale_from_packs_t<T1, 0>(pack_lo4, pack_hi4);
    const uint8_t sc3 = q2k_scale_from_packs_t<T1, 1>(pack_lo4, pack_hi4);
    return (uint32_t)(sc0 >> 4) |
           ((uint32_t)(sc1 >> 4) << 8) |
           ((uint32_t)(sc2 >> 4) << 16) |
           ((uint32_t)(sc3 >> 4) << 24);
}

template <int NFrag, typename TileB>
__device__ __forceinline__ void load_B_tile(
        TileB &B,
        const block_q8_1_mmq (&s_q8)[kStages][NFrag][8],
        int stage, int nf, int k_in_q8) {
    const int *base = reinterpret_cast<const int *>(&s_q8[stage][nf][0].qs[k_in_q8]);
    ggml_cuda_mma::load_ldmatrix(B, base, sizeof(block_q8_1_mmq) / sizeof(int));
}

struct Q8D4K64PairF32 {
    float2 c0;
    float2 c1;
};

template <int T0, int NFrag>
__device__ __forceinline__ Q8D4K64PairF32 load_q8_d4_k64_pair(
        const block_q8_1_mmq (&s_q8)[kStages][NFrag][8],
        int stage, int nf, int c0, int c1) {
    static_assert((T0 & 1) == 0, "expected even k32 base for k64 d4 load");
    constexpr int d4_base = T0 & 2;
    Q8D4K64PairF32 d;
    d.c0 = *reinterpret_cast<const float2 *>(&s_q8[stage][nf][c0].d4[d4_base]);
    d.c1 = *reinterpret_cast<const float2 *>(&s_q8[stage][nf][c1].d4[d4_base]);
    return d;
}

template <int T>
__device__ __forceinline__ float q8_d4_k64_slot(const float2 &d) {
    if constexpr ((T & 1) == 0) {
        return d.x;
    } else {
        return d.y;
    }
}

__device__ __forceinline__ bool iq2_raw_row_valid(int warp_row0, int row, int M) {
    return warp_row0 + row < M;
}

template <bool FullTile, int Iter>
__device__ __forceinline__ void issue_iq2_raw_codes_iter(
        IQ2RawWarpStage (&s_raw)[kWarps][kRawStages],
        const uint2 * __restrict__ qs_base,
        uint32_t warp_row0_blk, int nb, int raw_stage, int k256_iter,
        int warp_row0, int M, int warp, int lane) {
    if constexpr (Iter < kIQ2RawQCodeTrips) {
        const int t = lane + Iter * 32;
        if (t < kIQ2RawQCodeChunks) {
            const int row = t >> 3;
            const int pair = t & 7;
            const bool row_valid = FullTile ? true : iq2_raw_row_valid(warp_row0, row, M);
            const bool valid = k256_iter < nb && row_valid;
            const uint64_t blk = valid ? ((uint64_t)warp_row0_blk + (uint64_t)row * (uint64_t)nb +
                                           (uint64_t)k256_iter) : 0ull;
            void *dst = &s_raw[warp][raw_stage].qs[row][pair];
            const void *src = qs_base + blk * 8ull + (uint64_t)pair;
            cp_async_8B(dst, src, valid);
        }
        issue_iq2_raw_codes_iter<FullTile, Iter + 1>(
            s_raw, qs_base, warp_row0_blk, nb, raw_stage, k256_iter,
            warp_row0, M, warp, lane);
    }
}

template <bool FullTile>
__device__ __forceinline__ void issue_iq2_raw_prefetch(
        IQ2RawWarpStage (&s_raw)[kWarps][kRawStages],
        const volatile SmemInvariants &s_inv,
        int raw_stage, int k256_iter, int warp, int lane, half dq_val) {
    const uint2 *qs_base = s_inv.iq2_qs_base;
    const uint32_t warp_row0_blk = s_inv.warp_row0_blk[warp];
    const int nb = s_inv.nb;
    int warp_row0 = 0;
    int M = 0;
    if constexpr (!FullTile) {
        warp_row0 = s_inv.cta_row0 + (warp << 4);
        M = s_inv.M;
    }
    issue_iq2_raw_codes_iter<FullTile, 0>(
        s_raw, qs_base, warp_row0_blk, nb, raw_stage, k256_iter,
        warp_row0, M, warp, lane);
    if (lane < kIQ2RawRowsPerWarp) {
        s_raw[warp][raw_stage].dq[lane] = dq_val;
    }
    cp_async_commit();
}

/* The per-block dq halves cannot ride the cp.async ring (2-byte elements,
 * nb-strided rows), so they go global->register->smem.  Issuing the LDG here
 * and passing the value into issue_iq2_raw_prefetch* one k128 iteration later
 * hides the load latency behind a fold; fused LDG.U16->STS.U16 was 52% of the
 * kernel's long-scoreboard stalls (cmd2rncu15b PC sampling). */
template <bool FullTile>
__device__ __forceinline__ half iq2_raw_dq_preload(
        const volatile SmemInvariants &s_inv, int k256_iter) {
    const int lane = d2r_lane();
    half dq = __float2half(0.0f);
    if (lane < kIQ2RawRowsPerWarp) {
        bool valid = k256_iter < s_inv.nb;
        if constexpr (!FullTile) {
            const int warp_row0 = s_inv.cta_row0 + (d2r_warp() << 4);
            valid = valid && iq2_raw_row_valid(warp_row0, lane, s_inv.M);
        }
        if (valid) {
            const uint64_t blk = (uint64_t)s_inv.warp_row0_blk[d2r_warp()] +
                                 (uint64_t)lane * (uint64_t)s_inv.nb + (uint64_t)k256_iter;
            dq = s_inv.iq2_dq_base[blk];
        }
    }
    return dq;
}

template <int Iter>
__device__ __forceinline__ void issue_iq2_raw_codes_iter_fast(
        IQ2RawWarpStage (&s_raw)[kWarps][kRawStages],
        const uint2 * __restrict__ qs_base,
        uint32_t warp_row0_blk, int nb, int raw_stage, int k256_iter) {
    if constexpr (Iter < kIQ2RawQCodeTrips) {
        const int lane = d2r_lane();
        const int t = lane + Iter * 32;
        if (t < kIQ2RawQCodeChunks) {
            const int row = t >> 3;
            const int pair = t & 7;
            const bool valid = k256_iter < nb;
            const uint64_t blk = valid ? ((uint64_t)warp_row0_blk + (uint64_t)row * (uint64_t)nb +
                                           (uint64_t)k256_iter) : 0ull;
            void *dst = &s_raw[d2r_warp()][raw_stage].qs[row][pair];
            const void *src = qs_base + blk * 8ull + (uint64_t)pair;
            cp_async_8B(dst, src, valid);
        }
        issue_iq2_raw_codes_iter_fast<Iter + 1>(
            s_raw, qs_base, warp_row0_blk, nb, raw_stage, k256_iter);
    }
}

__device__ __forceinline__ void issue_iq2_raw_prefetch_fast(
        IQ2RawWarpStage (&s_raw)[kWarps][kRawStages],
        const volatile SmemInvariants &s_inv,
        int raw_stage, int k256_iter, half dq_val) {
    const int warp = d2r_warp();
    const int lane = d2r_lane();
    const uint32_t warp_row0_blk = s_inv.warp_row0_blk[warp];
    issue_iq2_raw_codes_iter_fast<0>(
        s_raw, s_inv.iq2_qs_base, warp_row0_blk, s_inv.nb, raw_stage, k256_iter);
    if (lane < kIQ2RawRowsPerWarp) {
        s_raw[warp][raw_stage].dq[lane] = dq_val;
    }
    cp_async_commit();
}

__device__ __forceinline__ uint32_t iq2_decode_signed_half(
        uint2 code, const uint2 * __restrict__ s_grid, int chunk) {
    const int group = chunk >> 1;
    const int hi = chunk & 1;
    const uint8_t aux = (uint8_t)(code.x >> (8 * group));
    const uint2 grid_pos = s_grid[aux];
    const uint32_t signs8 = unpack_ksigns((uint8_t)(code.y >> (7 * group)));
    const uint32_t sel = hi ? 0x80402010u : 0x08040201u;
    const uint32_t s = __vcmpne4(signs8 & sel, 0);
    const uint32_t grid_half = hi ? grid_pos.y : grid_pos.x;
    return __vsub4(grid_half ^ s, s);
}

template <int T, typename TileA>
__device__ __forceinline__ void make_iq2_A_tile(
        TileA &A, float &dA0, float &dA1,
        const IQ2RawWarpStage (&s_raw)[kWarps][kRawStages],
        const uint2 * __restrict__ s_grid,
        int warp, int raw_stage, bool row0_ok, bool row1_ok,
        int group, int tig) {
    const IQ2RawWarpStage &raw = s_raw[warp][raw_stage];
    constexpr int pair = T;
    const int row0 = group;
    const int row1 = group + 8;
    const uint2 code0 = row0_ok ? raw.qs[row0][pair] : make_uint2(0, 0);
    const uint2 code1 = row1_ok ? raw.qs[row1][pair] : make_uint2(0, 0);

    A.x[0] = row0_ok ? (int)iq2_decode_signed_half(code0, s_grid, tig) : 0;
    A.x[1] = row1_ok ? (int)iq2_decode_signed_half(code1, s_grid, tig) : 0;
    A.x[2] = row0_ok ? (int)iq2_decode_signed_half(code0, s_grid, tig + 4) : 0;
    A.x[3] = row1_ok ? (int)iq2_decode_signed_half(code1, s_grid, tig + 4) : 0;

    const float d0 = row0_ok ? __half2float(raw.dq[row0]) : 0.0f;
    const float d1 = row1_ok ? __half2float(raw.dq[row1]) : 0.0f;
    const int ls0 = (int)(code0.y >> 27) | 1;
    const int ls1 = (int)(code1.y >> 27) | 1;
    dA0 = d0 * (float)ls0 * 0.125f;
    dA1 = d1 * (float)ls1 * 0.125f;
}

template <int NFrag>
__device__ __forceinline__ void fold_iq2_fragment_fast(
        float (&acc)[NFrag][4],
        const ggml_cuda_mma::tile<16, 8, int> &C,
        int nf, float dA0, float dA1, float dB0, float dB1) {
    const float s00 = dA0 * dB0;
    const float s01 = dA0 * dB1;
    const float s10 = dA1 * dB0;
    const float s11 = dA1 * dB1;
    acc[nf][0] = fmaf((float)C.x[0], s00, acc[nf][0]);
    acc[nf][1] = fmaf((float)C.x[1], s01, acc[nf][1]);
    acc[nf][2] = fmaf((float)C.x[2], s10, acc[nf][2]);
    acc[nf][3] = fmaf((float)C.x[3], s11, acc[nf][3]);
}

template <int NFrag>
__device__ __forceinline__ void fold_iq2_fragment_guarded(
        float (&acc)[NFrag][4],
        const ggml_cuda_mma::tile<16, 8, int> &C,
        int nf, float dA0, float dA1, float dB0, float dB1,
        bool row0_ok, bool row1_ok, bool col0_ok, bool col1_ok) {
    const float s00 = dA0 * dB0;
    const float s01 = dA0 * dB1;
    const float s10 = dA1 * dB0;
    const float s11 = dA1 * dB1;
    if (row0_ok && col0_ok) acc[nf][0] = fmaf((float)C.x[0], s00, acc[nf][0]);
    if (row0_ok && col1_ok) acc[nf][1] = fmaf((float)C.x[1], s01, acc[nf][1]);
    if (row1_ok && col0_ok) acc[nf][2] = fmaf((float)C.x[2], s10, acc[nf][2]);
    if (row1_ok && col1_ok) acc[nf][3] = fmaf((float)C.x[3], s11, acc[nf][3]);
}

template <bool FullTile, typename TileA, typename TileB, typename TileC,
          int T0, int T1, int NFrag>
__device__ __forceinline__ void mma_fold_iq2_k32_pair_t(
        float (&acc)[NFrag][TileC::ne],
        const IQ2RawWarpStage (&s_raw)[kWarps][kRawStages],
        const uint2 * __restrict__ s_grid,
        const block_q8_1_mmq (&s_q8)[kStages][NFrag][8],
        int raw_stage, int q8_stage, bool raw_row0_ok, bool raw_row1_ok,
        int warp, int group, int tig, const volatile SmemInvariants &s_inv) {
    static_assert(T1 == T0 + 1, "expected adjacent k32 pair");
    static_assert(TileC::ne == 4, "expected m16n8 s32 accumulator fragment");
    TileA A0;
    TileA A1;
    float dA00;
    float dA01;
    float dA10;
    float dA11;
    make_iq2_A_tile<T0>(A0, dA00, dA01, s_raw, s_grid, warp, raw_stage,
                        raw_row0_ok, raw_row1_ok, group, tig);
    make_iq2_A_tile<T1>(A1, dA10, dA11, s_raw, s_grid, warp, raw_stage,
                        raw_row0_ok, raw_row1_ok, group, tig);

    constexpr int k_in_q8_0 = (T0 & 3) * 32;
    constexpr int k_in_q8_1 = (T1 & 3) * 32;
    const int c0 = TileC::get_j(0);
    const int c1 = TileC::get_j(1);
    int col_count = 0;
    int nf_live = NFrag;
    if constexpr (!FullTile) {
        col_count = s_inv.col_count;
        nf_live = (col_count + 7) >> 3;
    }

#pragma unroll
    for (int nf = 0; nf < NFrag; ++nf) {
        if constexpr (!FullTile) {
            if (nf >= nf_live) {
                break;
            }
        }
        const int col_local = nf * 8;
        bool col0_ok = true;
        bool col1_ok = true;
        if constexpr (!FullTile) {
            col0_ok = (col_local + c0) < col_count;
            col1_ok = (col_local + c1) < col_count;
        }
        const Q8D4K64PairF32 dB = load_q8_d4_k64_pair<T0>(s_q8, q8_stage, nf, c0, c1);

        TileB B0;
        TileC C0;
        load_B_tile(B0, s_q8, q8_stage, nf, k_in_q8_0);
        ggml_cuda_mma::mma(C0, A0, B0);
        if constexpr (FullTile) {
            fold_iq2_fragment_fast(acc, C0, nf, dA00, dA01,
                                   q8_d4_k64_slot<T0>(dB.c0), q8_d4_k64_slot<T0>(dB.c1));
        } else {
            fold_iq2_fragment_guarded(acc, C0, nf, dA00, dA01,
                                      q8_d4_k64_slot<T0>(dB.c0), q8_d4_k64_slot<T0>(dB.c1),
                                      raw_row0_ok, raw_row1_ok, col0_ok, col1_ok);
        }

        TileB B1;
        TileC C1;
        load_B_tile(B1, s_q8, q8_stage, nf, k_in_q8_1);
        ggml_cuda_mma::mma(C1, A1, B1);
        if constexpr (FullTile) {
            fold_iq2_fragment_fast(acc, C1, nf, dA10, dA11,
                                   q8_d4_k64_slot<T1>(dB.c0), q8_d4_k64_slot<T1>(dB.c1));
        } else {
            fold_iq2_fragment_guarded(acc, C1, nf, dA10, dA11,
                                      q8_d4_k64_slot<T1>(dB.c0), q8_d4_k64_slot<T1>(dB.c1),
                                      raw_row0_ok, raw_row1_ok, col0_ok, col1_ok);
        }
    }
}

template <bool FullTile, typename TileA, typename TileB, typename TileC,
          int NFrag>
__device__ __forceinline__ void mma_fold_iq2_k128(
        float (&acc)[NFrag][TileC::ne],
        const IQ2RawWarpStage (&s_raw)[kWarps][kRawStages],
        const uint2 * __restrict__ s_grid,
        const block_q8_1_mmq (&s_q8)[kStages][NFrag][8],
        int k128_iter, const volatile SmemInvariants &s_inv) {
    const int warp = d2r_warp();
    const int group = d2r_group();
    int warp_row0 = 0;
    if constexpr (!FullTile) {
        warp_row0 = s_inv.cta_row0 + (warp << 4);
    }
    bool row0_ok = true;
    bool row1_ok = true;
    if constexpr (!FullTile) {
        row0_ok = (warp_row0 + group) < s_inv.M;
        row1_ok = (warp_row0 + group + 8) < s_inv.M;
    }
    const int tig = d2r_tig();
    const int raw_stage = d2r_raw_stage(k128_iter >> 1);
    const int q8_stage = d2r_q8_stage(k128_iter);
    const int half_pair_base = (k128_iter & 1) ? 4 : 0;

    if (half_pair_base == 0) {
        mma_fold_iq2_k32_pair_t<FullTile, TileA, TileB, TileC, 0, 1>(
            acc, s_raw, s_grid, s_q8, raw_stage, q8_stage,
            row0_ok, row1_ok, warp, group, tig, s_inv);
        mma_fold_iq2_k32_pair_t<FullTile, TileA, TileB, TileC, 2, 3>(
            acc, s_raw, s_grid, s_q8, raw_stage, q8_stage,
            row0_ok, row1_ok, warp, group, tig, s_inv);
    } else {
        mma_fold_iq2_k32_pair_t<FullTile, TileA, TileB, TileC, 4, 5>(
            acc, s_raw, s_grid, s_q8, raw_stage, q8_stage,
            row0_ok, row1_ok, warp, group, tig, s_inv);
        mma_fold_iq2_k32_pair_t<FullTile, TileA, TileB, TileC, 6, 7>(
            acc, s_raw, s_grid, s_q8, raw_stage, q8_stage,
            row0_ok, row1_ok, warp, group, tig, s_inv);
    }
}

template <bool FullTile, typename TileA, typename TileB, typename TileC>
__device__ __forceinline__ void iq2_d2r_mainloop(
        float (&acc)[kNFrag][TileC::ne],
        block_q8_1_mmq (&s_q8)[kStages][kNFrag][8],
        IQ2RawWarpStage (&s_raw)[kWarps][kRawStages],
        const uint2 * __restrict__ s_grid,
        const volatile SmemInvariants &s_inv) {
#pragma unroll
    for (int pf = 0; pf < kStages; ++pf) {
        if (pf < s_inv.k128_iters) {
            if constexpr (FullTile) {
                issue_q8_prefetch_fast(s_q8, s_inv, d2r_q8_stage(pf), pf);
            } else {
                issue_q8_prefetch<false>(s_q8, s_inv, d2r_q8_stage(pf), pf, d2r_tid());
            }
        }
    }

#pragma unroll
    for (int pf = 0; pf < kRawStages; ++pf) {
        if (pf < s_inv.nb) {
            const half dq0 = iq2_raw_dq_preload<FullTile>(s_inv, pf);
            if constexpr (FullTile) {
                issue_iq2_raw_prefetch_fast(s_raw, s_inv, d2r_raw_stage(pf), pf, dq0);
            } else {
                issue_iq2_raw_prefetch<false>(
                    s_raw, s_inv, d2r_raw_stage(pf), pf, d2r_warp(), d2r_lane(), dq0);
            }
        }
    }

    half dq_pend = __float2half(0.0f);
    for (int k128_iter = 0;; ++k128_iter) {
        if (k128_iter >= s_inv.k128_iters) {
            break;
        }
        if ((k128_iter & 1) == 0) {
            int keep_raw = 0;
            if constexpr (kRawStages > 1) {
                keep_raw = ((k128_iter >> 1) + 1 < s_inv.nb) ? 1 : 0;
            }
            cp_async_wait_keep(keep_raw);
        }
        __syncthreads();
        if ((k128_iter & 1) == 0) {
            /* dq LDG for the raw prefetch issued at the NEXT (odd) iteration:
             * a full fold of distance between the load and its smem store. */
            const int raw_pf = (k128_iter >> 1) + kRawStages;
            if (raw_pf < s_inv.nb) {
                dq_pend = iq2_raw_dq_preload<FullTile>(s_inv, raw_pf);
            }
        }

        if constexpr (FullTile) {
            mma_fold_iq2_k128<true, TileA, TileB, TileC>(
                acc, s_raw, s_grid, s_q8, k128_iter, s_inv);
        } else {
            mma_fold_iq2_k128<false, TileA, TileB, TileC>(
                acc, s_raw, s_grid, s_q8, k128_iter, s_inv);
        }

        __syncthreads();
        const int pf_iter = k128_iter + kStages;
        if (pf_iter < s_inv.k128_iters) {
            if constexpr (FullTile) {
                issue_q8_prefetch_fast(s_q8, s_inv, d2r_q8_stage(pf_iter), pf_iter);
            } else {
                issue_q8_prefetch<false>(s_q8, s_inv, d2r_q8_stage(pf_iter), pf_iter, d2r_tid());
            }
        }
        if ((k128_iter & 1) != 0) {
            const int raw_pf = (k128_iter >> 1) + kRawStages;
            if (raw_pf < s_inv.nb) {
                if constexpr (FullTile) {
                    issue_iq2_raw_prefetch_fast(s_raw, s_inv, d2r_raw_stage(raw_pf), raw_pf, dq_pend);
                } else {
                    issue_iq2_raw_prefetch<false>(
                        s_raw, s_inv, d2r_raw_stage(raw_pf), raw_pf, d2r_warp(), d2r_lane(), dq_pend);
                }
            }
        }
    }
}

template <bool FullTile>
__device__ __forceinline__ void issue_fused_iq2_raw_prefetch(
        IQ2RawWarpStage (&s_raw)[kWarps][kRawStages],
        const volatile SmemInvariants &s_inv,
        int raw_stage, int k256_iter) {
    const half dq = iq2_raw_dq_preload<FullTile>(s_inv, k256_iter);
    if constexpr (FullTile) {
        issue_iq2_raw_prefetch_fast(s_raw, s_inv, raw_stage, k256_iter, dq);
    } else {
        issue_iq2_raw_prefetch<false>(
            s_raw, s_inv, raw_stage, k256_iter,
            d2r_warp(), d2r_lane(), dq);
    }
}

template <bool FullTile, int TileN,
          typename TileA, typename TileB, typename TileC>
__device__ __forceinline__ void iq2_gateup_fused_mainloop(
        float (&gate_acc)[TileN / 8][TileC::ne],
        float (&up_acc)[TileN / 8][TileC::ne],
        FusedGateUpComputeSmem<TileN> &s,
        uint32_t y_off) {
    const int k128_iters = s.inv[0].k128_iters;
    const int k256_iters = s.inv[0].nb;

    issue_fused_q8_prefetch<FullTile>(s.q8, s.inv[0], y_off, 0, 0);
    issue_fused_q8_prefetch<FullTile>(s.q8, s.inv[0], y_off, 1, 1);
    issue_fused_iq2_raw_prefetch<FullTile>(s.raw[0], s.inv[0], 0, 0);
    issue_fused_iq2_raw_prefetch<FullTile>(s.raw[1], s.inv[1], 0, 0);
    cp_async_wait_keep(0);
    __syncthreads();

    for (int k256_iter = 0; k256_iter < k256_iters; ++k256_iter) {
        const int next_k256 = k256_iter + 1;
        if (next_k256 < k256_iters) {
            const int next_stage = d2r_raw_stage(next_k256);
            issue_fused_iq2_raw_prefetch<FullTile>(
                s.raw[0], s.inv[0], next_stage, next_k256);
            issue_fused_iq2_raw_prefetch<FullTile>(
                s.raw[1], s.inv[1], next_stage, next_k256);
        }

        const int even_k128 = 2 * k256_iter;
        mma_fold_iq2_k128<FullTile, TileA, TileB, TileC>(
            gate_acc, s.raw[0], s.grid, s.q8, even_k128, s.inv[0]);
        mma_fold_iq2_k128<FullTile, TileA, TileB, TileC>(
            up_acc, s.raw[1], s.grid, s.q8, even_k128, s.inv[1]);

        __syncthreads();
        const int next_even_k128 = even_k128 + 2;
        if (next_even_k128 < k128_iters) {
            issue_fused_q8_prefetch<FullTile>(
                s.q8, s.inv[0], y_off, d2r_q8_stage(next_even_k128), next_even_k128);
        }

        const int odd_k128 = even_k128 + 1;
        mma_fold_iq2_k128<FullTile, TileA, TileB, TileC>(
            gate_acc, s.raw[0], s.grid, s.q8, odd_k128, s.inv[0]);
        mma_fold_iq2_k128<FullTile, TileA, TileB, TileC>(
            up_acc, s.raw[1], s.grid, s.q8, odd_k128, s.inv[1]);

        __syncthreads();
        const int next_odd_k128 = odd_k128 + 2;
        if (next_odd_k128 < k128_iters) {
            issue_fused_q8_prefetch<FullTile>(
                s.q8, s.inv[0], y_off, d2r_q8_stage(next_odd_k128), next_odd_k128);
        }

        if (next_k256 < k256_iters) {
            cp_async_wait_keep(0);
            __syncthreads();
        }
    }
}

[[maybe_unused]] __device__ __forceinline__ void fold_element_k32(
        float &acc, int c, float2 dm, uint16_t min_pack, const Q8Fix2 &fix) {
    acc += (float)c * dm.x * fix.d8;
    acc -= dm.y * (float)(min_pack & 0xFFu) * fix.sum0;
    acc -= dm.y * (float)((min_pack >> 8) & 0xFFu) * fix.sum1;
}

[[maybe_unused]] __device__ __forceinline__ void fold_element_k64(
        float &acc, int c, float2 dm, uint32_t min_pack4, const Q8Fix4 &fix) {
    acc += (float)c * dm.x * fix.d8;
    acc -= dm.y * (float)(min_pack4 & 0xFFu) * fix.sum0;
    acc -= dm.y * (float)((min_pack4 >> 8) & 0xFFu) * fix.sum1;
    acc -= dm.y * (float)((min_pack4 >> 16) & 0xFFu) * fix.sum2;
    acc -= dm.y * (float)((min_pack4 >> 24) & 0xFFu) * fix.sum3;
}

[[maybe_unused]] __device__ __forceinline__ float q2k_bias_k32(
        float dmin, uint16_t min_pack, const Q8Fix2 &fix) {
    const float min0 = (float)(min_pack & 0xFFu);
    const float min1 = (float)((min_pack >> 8) & 0xFFu);
    return dmin * (min0 * fix.sum0 + min1 * fix.sum1);
}

__device__ __forceinline__ float q2k_minsum4_f32(
        uint32_t min_pack4, float sum0, float sum1, float sum2, float sum3) {
    float s = fmaf((float)(min_pack4 & 0xFFu), sum0, 0.0f);
    s = fmaf((float)((min_pack4 >> 8) & 0xFFu), sum1, s);
    s = fmaf((float)((min_pack4 >> 16) & 0xFFu), sum2, s);
    return fmaf((float)((min_pack4 >> 24) & 0xFFu), sum3, s);
}

template <int T0, int T1, int NFrag>
__device__ __forceinline__ void fold_k64_col_fast(
        float &acc0, float &acc1, int c0, int c1,
        const Q8ColFixF32 (&s_q8_fix)[kStages][NFrag][8],
        int stage, int nf, int c, float2 dm0, float2 dm1,
        uint32_t min0, uint32_t min1) {
    static_assert((T0 == 0 && T1 == 1) || (T0 == 2 && T1 == 3), "expected adjacent k64 q8 window");
    constexpr int d8_slot = (T0 >= 2) ? 1 : 0;
    constexpr int sub128 = 2 * T0;
    const Q8ColFixF32 &sf = s_q8_fix[stage][nf][c];
    const float d8 = sf.d8[d8_slot];
    const float sum0 = sf.sum[sub128 + 0];
    const float sum1 = sf.sum[sub128 + 1];
    const float sum2 = sf.sum[sub128 + 2];
    const float sum3 = sf.sum[sub128 + 3];

    const float minsum0 = q2k_minsum4_f32(min0, sum0, sum1, sum2, sum3);
    acc0 = fmaf(-dm0.y, minsum0, fmaf((float)c0, dm0.x * d8, acc0));

    const float minsum1 = q2k_minsum4_f32(min1, sum0, sum1, sum2, sum3);
    acc1 = fmaf(-dm1.y, minsum1, fmaf((float)c1, dm1.x * d8, acc1));
}

template <int T0, int T1, int NFrag>
__device__ __forceinline__ void fold_k64_col_guarded(
        float &acc0, float &acc1, int c0, int c1,
        const Q8ColFixF32 (&s_q8_fix)[kStages][NFrag][8],
        int stage, int nf, int c, float2 dm0, float2 dm1,
        uint32_t min0, uint32_t min1, bool row0_ok, bool row1_ok, bool col_ok) {
    static_assert((T0 == 0 && T1 == 1) || (T0 == 2 && T1 == 3), "expected adjacent k64 q8 window");
    if (!col_ok || (!row0_ok && !row1_ok)) {
        return;
    }
    constexpr int d8_slot = (T0 >= 2) ? 1 : 0;
    constexpr int sub128 = 2 * T0;
    const Q8ColFixF32 &sf = s_q8_fix[stage][nf][c];
    const float d8 = sf.d8[d8_slot];
    const float sum0 = sf.sum[sub128 + 0];
    const float sum1 = sf.sum[sub128 + 1];
    const float sum2 = sf.sum[sub128 + 2];
    const float sum3 = sf.sum[sub128 + 3];

    if (row0_ok) {
        const float minsum0 = q2k_minsum4_f32(min0, sum0, sum1, sum2, sum3);
        acc0 = fmaf(-dm0.y, minsum0, fmaf((float)c0, dm0.x * d8, acc0));
    }
    if (row1_ok) {
        const float minsum1 = q2k_minsum4_f32(min1, sum0, sum1, sum2, sum3);
        acc1 = fmaf(-dm1.y, minsum1, fmaf((float)c1, dm1.x * d8, acc1));
    }
}

template <int NFrag>
__device__ __forceinline__ void fold_fragment_k32_fast(
        float (&acc)[NFrag][4],
        const ggml_cuda_mma::tile<16, 8, int> &C,
        const Q8Fix2 &fix0, const Q8Fix2 &fix1, int nf,
        float2 dm0, float2 dm1, uint16_t min0, uint16_t min1) {
    const float d8c0 = fix0.d8;
    const float d8c1 = fix1.d8;
    const float scale00 = dm0.x * d8c0;
    const float scale01 = dm0.x * d8c1;
    const float scale10 = dm1.x * d8c0;
    const float scale11 = dm1.x * d8c1;
    const float bias00 = q2k_bias_k32(dm0.y, min0, fix0);
    const float bias01 = q2k_bias_k32(dm0.y, min0, fix1);
    const float bias10 = q2k_bias_k32(dm1.y, min1, fix0);
    const float bias11 = q2k_bias_k32(dm1.y, min1, fix1);

    acc[nf][0] = fmaf((float)C.x[0], scale00, acc[nf][0]) - bias00;
    acc[nf][1] = fmaf((float)C.x[1], scale01, acc[nf][1]) - bias01;
    acc[nf][2] = fmaf((float)C.x[2], scale10, acc[nf][2]) - bias10;
    acc[nf][3] = fmaf((float)C.x[3], scale11, acc[nf][3]) - bias11;
}

template <int NFrag>
__device__ __forceinline__ void fold_fragment_k32_guarded(
        float (&acc)[NFrag][4],
        const ggml_cuda_mma::tile<16, 8, int> &C,
        const Q8Fix2 &fix0, const Q8Fix2 &fix1, int nf,
        float2 dm0, float2 dm1, uint16_t min0, uint16_t min1,
        bool row0_ok, bool row1_ok, bool col0_ok, bool col1_ok) {
    const float d8c0 = fix0.d8;
    const float d8c1 = fix1.d8;
    const float scale00 = dm0.x * d8c0;
    const float scale01 = dm0.x * d8c1;
    const float scale10 = dm1.x * d8c0;
    const float scale11 = dm1.x * d8c1;
    const float bias00 = q2k_bias_k32(dm0.y, min0, fix0);
    const float bias01 = q2k_bias_k32(dm0.y, min0, fix1);
    const float bias10 = q2k_bias_k32(dm1.y, min1, fix0);
    const float bias11 = q2k_bias_k32(dm1.y, min1, fix1);

    if (row0_ok && col0_ok) acc[nf][0] = fmaf((float)C.x[0], scale00, acc[nf][0]) - bias00;
    if (row0_ok && col1_ok) acc[nf][1] = fmaf((float)C.x[1], scale01, acc[nf][1]) - bias01;
    if (row1_ok && col0_ok) acc[nf][2] = fmaf((float)C.x[2], scale10, acc[nf][2]) - bias10;
    if (row1_ok && col1_ok) acc[nf][3] = fmaf((float)C.x[3], scale11, acc[nf][3]) - bias11;
}

template <int T0, int T1, int NFrag>
__device__ __forceinline__ void fold_fragment_k64_fast(
        float (&acc)[NFrag][4],
        const ggml_cuda_mma::tile<16, 8, int> &C,
        const Q8ColFixF32 (&s_q8_fix)[kStages][NFrag][8],
        int stage, int nf, int c0, int c1, float2 dm0, float2 dm1,
        uint32_t min0, uint32_t min1) {
    fold_k64_col_fast<T0, T1>(
        acc[nf][0], acc[nf][2], C.x[0], C.x[2],
        s_q8_fix, stage, nf, c0, dm0, dm1, min0, min1);
    fold_k64_col_fast<T0, T1>(
        acc[nf][1], acc[nf][3], C.x[1], C.x[3],
        s_q8_fix, stage, nf, c1, dm0, dm1, min0, min1);
}

template <int T0, int T1, int NFrag>
__device__ __forceinline__ void fold_fragment_k64_guarded(
        float (&acc)[NFrag][4],
        const ggml_cuda_mma::tile<16, 8, int> &C,
        const Q8ColFixF32 (&s_q8_fix)[kStages][NFrag][8],
        int stage, int nf, int c0, int c1, float2 dm0, float2 dm1,
        uint32_t min0, uint32_t min1,
        bool row0_ok, bool row1_ok, bool col0_ok, bool col1_ok) {
    if (col0_ok && (row0_ok || row1_ok)) {
        fold_k64_col_guarded<T0, T1>(
            acc[nf][0], acc[nf][2], C.x[0], C.x[2],
            s_q8_fix, stage, nf, c0, dm0, dm1, min0, min1,
            row0_ok, row1_ok, col0_ok);
    }
    if (col1_ok && (row0_ok || row1_ok)) {
        fold_k64_col_guarded<T0, T1>(
            acc[nf][1], acc[nf][3], C.x[1], C.x[3],
            s_q8_fix, stage, nf, c1, dm0, dm1, min0, min1,
            row0_ok, row1_ok, col1_ok);
    }
}

template <typename TileC, int NFrag>
__device__ __forceinline__ void fold_fragment_k32(
        float (&acc)[NFrag][TileC::ne],
        const TileC &C,
        const Q8ColFixF32 (&s_q8_fix)[kStages][NFrag][8],
        int stage, int nf, int k_in_q8, int warp_row0, int M,
        int col_lo, int col_hi, uint32_t dm0_bits, uint32_t dm1_bits,
        uint16_t min0, uint16_t min1) {
    static_assert(TileC::ne == 4, "expected m16n8 s32 accumulator fragment");
    const int c0 = TileC::get_j(0);
    const int c1 = TileC::get_j(1);
    const int col0 = col_lo + nf * 8 + c0;
    const int col1 = col_lo + nf * 8 + c1;
    const bool col0_ok = col0 < col_hi;
    const bool col1_ok = col1 < col_hi;
    const Q8Fix2 fix0 = load_q8_fix2(s_q8_fix, stage, nf, c0, k_in_q8);
    const Q8Fix2 fix1 = load_q8_fix2(s_q8_fix, stage, nf, c1, k_in_q8);

    const float2 dm0 = half2_bits_to_float2(dm0_bits);
    const float2 dm1 = half2_bits_to_float2(dm1_bits);
    const int row0 = warp_row0 + TileC::get_i(0);
    const int row1 = warp_row0 + TileC::get_i(2);
    const bool row0_ok = row0 < M;
    const bool row1_ok = row1 < M;

    if (row0_ok && col0_ok) {
        fold_element_k32(acc[nf][0], C.x[0], dm0, min0, fix0);
    }
    if (row0_ok && col1_ok) {
        fold_element_k32(acc[nf][1], C.x[1], dm0, min0, fix1);
    }
    if (row1_ok && col0_ok) {
        fold_element_k32(acc[nf][2], C.x[2], dm1, min1, fix0);
    }
    if (row1_ok && col1_ok) {
        fold_element_k32(acc[nf][3], C.x[3], dm1, min1, fix1);
    }
}

template <typename TileC, int NFrag>
__device__ __forceinline__ void fold_fragment_k64(
        float (&acc)[NFrag][TileC::ne],
        const TileC &C,
        const Q8ColFixF32 (&s_q8_fix)[kStages][NFrag][8],
        int stage, int nf, int k_in_q8_pair, int warp_row0, int M,
        int col_lo, int col_hi, uint32_t dm0_bits, uint32_t dm1_bits,
        uint32_t min0, uint32_t min1) {
    static_assert(TileC::ne == 4, "expected m16n8 s32 accumulator fragment");
    const int c0 = TileC::get_j(0);
    const int c1 = TileC::get_j(1);
    const int col0 = col_lo + nf * 8 + c0;
    const int col1 = col_lo + nf * 8 + c1;
    const bool col0_ok = col0 < col_hi;
    const bool col1_ok = col1 < col_hi;
    const Q8Fix4 fix0 = load_q8_fix4(s_q8_fix, stage, nf, c0, k_in_q8_pair);
    const Q8Fix4 fix1 = load_q8_fix4(s_q8_fix, stage, nf, c1, k_in_q8_pair);

    const float2 dm0 = half2_bits_to_float2(dm0_bits);
    const float2 dm1 = half2_bits_to_float2(dm1_bits);
    const int row0 = warp_row0 + TileC::get_i(0);
    const int row1 = warp_row0 + TileC::get_i(2);
    const bool row0_ok = row0 < M;
    const bool row1_ok = row1 < M;

    if (row0_ok && col0_ok) {
        fold_element_k64(acc[nf][0], C.x[0], dm0, min0, fix0);
    }
    if (row0_ok && col1_ok) {
        fold_element_k64(acc[nf][1], C.x[1], dm0, min0, fix1);
    }
    if (row1_ok && col0_ok) {
        fold_element_k64(acc[nf][2], C.x[2], dm1, min1, fix0);
    }
    if (row1_ok && col1_ok) {
        fold_element_k64(acc[nf][3], C.x[3], dm1, min1, fix1);
    }
}

template <typename TileA, typename TileB, typename TileC, bool FullTile, int NFrag>
__device__ __forceinline__ void mma_fold_k128_k32(
        float (&acc)[NFrag][TileC::ne],
        const Q2KWeightHalf &w,
        const block_q8_1_mmq (&s_q8)[kStages][NFrag][8],
        const Q8ColFixF32 (&s_q8_fix)[kStages][NFrag][8],
        int stage, int warp_row0, int M, int col_lo, int col_hi, int nf_live) {
    TileA A0;
    TileA A1;
    TileA A2;
    TileA A3;
    make_A_tile<0>(A0, w);
    make_A_tile<1>(A1, w);
    make_A_tile<2>(A2, w);
    make_A_tile<3>(A3, w);

    const uint32_t min0_p01 = min_pack4_for_pair<0, 1>(w.sc_r0_lo4, w.sc_r0_hi4);
    const uint32_t min1_p01 = min_pack4_for_pair<0, 1>(w.sc_r1_lo4, w.sc_r1_hi4);
    const uint32_t min0_p23 = min_pack4_for_pair<2, 3>(w.sc_r0_lo4, w.sc_r0_hi4);
    const uint32_t min1_p23 = min_pack4_for_pair<2, 3>(w.sc_r1_lo4, w.sc_r1_hi4);
    const float2 dm0 = half2_bits_to_float2(w.dm_r0);
    const float2 dm1 = half2_bits_to_float2(w.dm_r1);
    const int c0 = TileC::get_j(0);
    const int c1 = TileC::get_j(1);
    const bool row0_ok = FullTile ? true : ((warp_row0 + TileC::get_i(0)) < M);
    const bool row1_ok = FullTile ? true : ((warp_row0 + TileC::get_i(2)) < M);

#pragma unroll
    for (int nf = 0; nf < NFrag; ++nf) {
        if constexpr (!FullTile) {
            if (nf >= nf_live) {
                break;
            }
        }
        const int col_frag = col_lo + nf * 8;
        const bool col0_ok = FullTile ? true : ((col_frag + c0) < col_hi);
        const bool col1_ok = FullTile ? true : ((col_frag + c1) < col_hi);

        TileB B0;
        TileC C0;
        const Q8Fix2 fix00 = load_q8_fix2(s_q8_fix, stage, nf, c0, 0);
        const Q8Fix2 fix01 = load_q8_fix2(s_q8_fix, stage, nf, c1, 0);
        load_B_tile(B0, s_q8, stage, nf, 0);
        ggml_cuda_mma::mma(C0, A0, B0);
        if constexpr (FullTile) {
            fold_fragment_k32_fast(acc, C0, fix00, fix01, nf, dm0, dm1,
                                   (uint16_t)min0_p01, (uint16_t)min1_p01);
        } else {
            fold_fragment_k32_guarded(acc, C0, fix00, fix01, nf, dm0, dm1,
                                      (uint16_t)min0_p01, (uint16_t)min1_p01,
                                      row0_ok, row1_ok, col0_ok, col1_ok);
        }

        TileB B1;
        TileC C1;
        const Q8Fix2 fix10 = load_q8_fix2(s_q8_fix, stage, nf, c0, 32);
        const Q8Fix2 fix11 = load_q8_fix2(s_q8_fix, stage, nf, c1, 32);
        load_B_tile(B1, s_q8, stage, nf, 32);
        ggml_cuda_mma::mma(C1, A1, B1);
        if constexpr (FullTile) {
            fold_fragment_k32_fast(acc, C1, fix10, fix11, nf, dm0, dm1,
                                   (uint16_t)(min0_p01 >> 16), (uint16_t)(min1_p01 >> 16));
        } else {
            fold_fragment_k32_guarded(acc, C1, fix10, fix11, nf, dm0, dm1,
                                      (uint16_t)(min0_p01 >> 16), (uint16_t)(min1_p01 >> 16),
                                      row0_ok, row1_ok, col0_ok, col1_ok);
        }

        TileB B2;
        TileC C2;
        const Q8Fix2 fix20 = load_q8_fix2(s_q8_fix, stage, nf, c0, 64);
        const Q8Fix2 fix21 = load_q8_fix2(s_q8_fix, stage, nf, c1, 64);
        load_B_tile(B2, s_q8, stage, nf, 64);
        ggml_cuda_mma::mma(C2, A2, B2);
        if constexpr (FullTile) {
            fold_fragment_k32_fast(acc, C2, fix20, fix21, nf, dm0, dm1,
                                   (uint16_t)min0_p23, (uint16_t)min1_p23);
        } else {
            fold_fragment_k32_guarded(acc, C2, fix20, fix21, nf, dm0, dm1,
                                      (uint16_t)min0_p23, (uint16_t)min1_p23,
                                      row0_ok, row1_ok, col0_ok, col1_ok);
        }

        TileB B3;
        TileC C3;
        const Q8Fix2 fix30 = load_q8_fix2(s_q8_fix, stage, nf, c0, 96);
        const Q8Fix2 fix31 = load_q8_fix2(s_q8_fix, stage, nf, c1, 96);
        load_B_tile(B3, s_q8, stage, nf, 96);
        ggml_cuda_mma::mma(C3, A3, B3);
        if constexpr (FullTile) {
            fold_fragment_k32_fast(acc, C3, fix30, fix31, nf, dm0, dm1,
                                   (uint16_t)(min0_p23 >> 16), (uint16_t)(min1_p23 >> 16));
        } else {
            fold_fragment_k32_guarded(acc, C3, fix30, fix31, nf, dm0, dm1,
                                      (uint16_t)(min0_p23 >> 16), (uint16_t)(min1_p23 >> 16),
                                      row0_ok, row1_ok, col0_ok, col1_ok);
        }
    }
}

template <typename TileA, typename TileB, typename TileC, bool FullTile, int NFrag>
__device__ __forceinline__ void mma_fold_k128_k64(
        float (&acc)[NFrag][TileC::ne],
        const Q2KRawWarpStage (&s_raw)[kWarps][kRawStages],
        const block_q8_1_mmq (&s_q8)[kStages][NFrag][8],
        const Q8ColFixF32 (&s_q8_fix)[kStages][NFrag][8],
        int raw_stage, bool raw_row0_ok, bool raw_row1_ok, int parity, int raw_half,
        int warp, int group, int tig, int stage, int warp_row0, int M, int col_lo, int col_hi,
        int nf_live) {
    const int c0 = TileC::get_j(0);
    const int c1 = TileC::get_j(1);
    const bool row0_ok = FullTile ? true : ((warp_row0 + TileC::get_i(0)) < M);
    const bool row1_ok = FullTile ? true : ((warp_row0 + TileC::get_i(2)) < M);

    {
        TileA A0;
        TileA A1;
        uint32_t min0;
        uint32_t min1;
        float2 dm0;
        float2 dm1;
        {
            Q2KWeightHalf w;
            load_q2k_weight_half_raw(
                w, s_raw, warp, raw_stage, raw_row0_ok, raw_row1_ok,
                parity, raw_half, group, tig);
            make_A_tile<0>(A0, w);
            make_A_tile<1>(A1, w);
            min0 = min_pack4_for_pair<0, 1>(w.sc_r0_lo4, w.sc_r0_hi4);
            min1 = min_pack4_for_pair<0, 1>(w.sc_r1_lo4, w.sc_r1_hi4);
            dm0 = half2_bits_to_float2(w.dm_r0);
            dm1 = half2_bits_to_float2(w.dm_r1);
        }
#pragma unroll
        for (int nf = 0; nf < NFrag; ++nf) {
            if constexpr (!FullTile) {
                if (nf >= nf_live) {
                    break;
                }
            }
            const int col_frag = col_lo + nf * 8;
            const bool col0_ok = FullTile ? true : ((col_frag + c0) < col_hi);
            const bool col1_ok = FullTile ? true : ((col_frag + c1) < col_hi);
            TileC C01;
            TileB B;
            load_B_tile(B, s_q8, stage, nf, 0);
            ggml_cuda_mma::mma(C01, A0, B);
            load_B_tile(B, s_q8, stage, nf, 32);
            ggml_cuda_mma::mma(C01, A1, B);
            if constexpr (FullTile) {
                fold_fragment_k64_fast<0, 1>(
                    acc, C01, s_q8_fix, stage, nf, c0, c1, dm0, dm1, min0, min1);
            } else {
                fold_fragment_k64_guarded<0, 1>(
                    acc, C01, s_q8_fix, stage, nf, c0, c1, dm0, dm1, min0, min1,
                    row0_ok, row1_ok, col0_ok, col1_ok);
            }
        }
    }

    {
        TileA A2;
        TileA A3;
        uint32_t min0;
        uint32_t min1;
        float2 dm0;
        float2 dm1;
        {
            Q2KWeightHalf w;
            load_q2k_weight_half_raw(
                w, s_raw, warp, raw_stage, raw_row0_ok, raw_row1_ok,
                parity, raw_half, group, tig);
            make_A_tile<2>(A2, w);
            make_A_tile<3>(A3, w);
            min0 = min_pack4_for_pair<2, 3>(w.sc_r0_lo4, w.sc_r0_hi4);
            min1 = min_pack4_for_pair<2, 3>(w.sc_r1_lo4, w.sc_r1_hi4);
            dm0 = half2_bits_to_float2(w.dm_r0);
            dm1 = half2_bits_to_float2(w.dm_r1);
        }
#pragma unroll
        for (int nf = 0; nf < NFrag; ++nf) {
            if constexpr (!FullTile) {
                if (nf >= nf_live) {
                    break;
                }
            }
            const int col_frag = col_lo + nf * 8;
            const bool col0_ok = FullTile ? true : ((col_frag + c0) < col_hi);
            const bool col1_ok = FullTile ? true : ((col_frag + c1) < col_hi);
            TileC C23;
            TileB B;
            load_B_tile(B, s_q8, stage, nf, 64);
            ggml_cuda_mma::mma(C23, A2, B);
            load_B_tile(B, s_q8, stage, nf, 96);
            ggml_cuda_mma::mma(C23, A3, B);
            if constexpr (FullTile) {
                fold_fragment_k64_fast<2, 3>(
                    acc, C23, s_q8_fix, stage, nf, c0, c1, dm0, dm1, min0, min1);
            } else {
                fold_fragment_k64_guarded<2, 3>(
                    acc, C23, s_q8_fix, stage, nf, c0, c1, dm0, dm1, min0, min1,
                    row0_ok, row1_ok, col0_ok, col1_ok);
            }
        }
    }
}

template <typename TileA, typename TileB, typename TileC, int NFrag>
__device__ __forceinline__ void mma_fold_k128_k32_fast(
        float (&acc)[NFrag][TileC::ne],
        const Q2KRawWarpStage (&s_raw)[kWarps][kRawStages],
        const block_q8_1_mmq (&s_q8)[kStages][NFrag][8],
        const Q8ColFixF32 (&s_q8_fix)[kStages][NFrag][8],
        int k128_iter) {
    static_assert(TileC::ne == 4, "expected m16n8 s32 accumulator fragment");
    Q2KWeightHalf w;
    load_q2k_weight_half_raw_fast(w, s_raw, k128_iter);

    TileA A0;
    TileA A1;
    TileA A2;
    TileA A3;
    make_A_tile<0>(A0, w);
    make_A_tile<1>(A1, w);
    make_A_tile<2>(A2, w);
    make_A_tile<3>(A3, w);

    const uint32_t min0_p01 = min_pack4_for_pair<0, 1>(w.sc_r0_lo4, w.sc_r0_hi4);
    const uint32_t min1_p01 = min_pack4_for_pair<0, 1>(w.sc_r1_lo4, w.sc_r1_hi4);
    const uint32_t min0_p23 = min_pack4_for_pair<2, 3>(w.sc_r0_lo4, w.sc_r0_hi4);
    const uint32_t min1_p23 = min_pack4_for_pair<2, 3>(w.sc_r1_lo4, w.sc_r1_hi4);
    const float2 dm0 = half2_bits_to_float2(w.dm_r0);
    const float2 dm1 = half2_bits_to_float2(w.dm_r1);
    const int c0 = TileC::get_j(0);
    const int c1 = TileC::get_j(1);

#pragma unroll
    for (int nf = 0; nf < NFrag; ++nf) {
        TileB B0;
        TileC C0;
        const Q8Fix2 fix00 = load_q8_fix2(s_q8_fix, d2r_q8_stage(k128_iter), nf, c0, 0);
        const Q8Fix2 fix01 = load_q8_fix2(s_q8_fix, d2r_q8_stage(k128_iter), nf, c1, 0);
        load_B_tile(B0, s_q8, d2r_q8_stage(k128_iter), nf, 0);
        ggml_cuda_mma::mma(C0, A0, B0);
        fold_fragment_k32_fast(acc, C0, fix00, fix01, nf, dm0, dm1,
                               (uint16_t)min0_p01, (uint16_t)min1_p01);

        TileB B1;
        TileC C1;
        const Q8Fix2 fix10 = load_q8_fix2(s_q8_fix, d2r_q8_stage(k128_iter), nf, c0, 32);
        const Q8Fix2 fix11 = load_q8_fix2(s_q8_fix, d2r_q8_stage(k128_iter), nf, c1, 32);
        load_B_tile(B1, s_q8, d2r_q8_stage(k128_iter), nf, 32);
        ggml_cuda_mma::mma(C1, A1, B1);
        fold_fragment_k32_fast(acc, C1, fix10, fix11, nf, dm0, dm1,
                               (uint16_t)(min0_p01 >> 16), (uint16_t)(min1_p01 >> 16));

        TileB B2;
        TileC C2;
        const Q8Fix2 fix20 = load_q8_fix2(s_q8_fix, d2r_q8_stage(k128_iter), nf, c0, 64);
        const Q8Fix2 fix21 = load_q8_fix2(s_q8_fix, d2r_q8_stage(k128_iter), nf, c1, 64);
        load_B_tile(B2, s_q8, d2r_q8_stage(k128_iter), nf, 64);
        ggml_cuda_mma::mma(C2, A2, B2);
        fold_fragment_k32_fast(acc, C2, fix20, fix21, nf, dm0, dm1,
                               (uint16_t)min0_p23, (uint16_t)min1_p23);

        TileB B3;
        TileC C3;
        const Q8Fix2 fix30 = load_q8_fix2(s_q8_fix, d2r_q8_stage(k128_iter), nf, c0, 96);
        const Q8Fix2 fix31 = load_q8_fix2(s_q8_fix, d2r_q8_stage(k128_iter), nf, c1, 96);
        load_B_tile(B3, s_q8, d2r_q8_stage(k128_iter), nf, 96);
        ggml_cuda_mma::mma(C3, A3, B3);
        fold_fragment_k32_fast(acc, C3, fix30, fix31, nf, dm0, dm1,
                               (uint16_t)(min0_p23 >> 16), (uint16_t)(min1_p23 >> 16));
    }
}

template <typename TileA, typename TileB, typename TileC, int NFrag>
__device__ __forceinline__ void mma_fold_k128_k64_fast(
        float (&acc)[NFrag][TileC::ne],
        const Q2KRawWarpStage (&s_raw)[kWarps][kRawStages],
        const block_q8_1_mmq (&s_q8)[kStages][NFrag][8],
        const Q8ColFixF32 (&s_q8_fix)[kStages][NFrag][8],
        int k128_iter) {
    static_assert(TileC::ne == 4, "expected m16n8 s32 accumulator fragment");
    const int c0 = TileC::get_j(0);
    const int c1 = TileC::get_j(1);

    {
        TileA A0;
        TileA A1;
        uint32_t min0;
        uint32_t min1;
        float2 dm0;
        float2 dm1;
        {
            Q2KWeightHalf w;
            load_q2k_weight_half_raw_fast(w, s_raw, k128_iter);
            make_A_tile<0>(A0, w);
            make_A_tile<1>(A1, w);
            min0 = min_pack4_for_pair<0, 1>(w.sc_r0_lo4, w.sc_r0_hi4);
            min1 = min_pack4_for_pair<0, 1>(w.sc_r1_lo4, w.sc_r1_hi4);
            dm0 = half2_bits_to_float2(w.dm_r0);
            dm1 = half2_bits_to_float2(w.dm_r1);
        }
#pragma unroll
        for (int nf = 0; nf < NFrag; ++nf) {
            TileC C01;
            TileB B;
            load_B_tile(B, s_q8, d2r_q8_stage(k128_iter), nf, 0);
            ggml_cuda_mma::mma(C01, A0, B);
            load_B_tile(B, s_q8, d2r_q8_stage(k128_iter), nf, 32);
            ggml_cuda_mma::mma(C01, A1, B);
            fold_fragment_k64_fast<0, 1>(
                acc, C01, s_q8_fix, d2r_q8_stage(k128_iter), nf, c0, c1, dm0, dm1, min0, min1);
        }
    }

    {
        TileA A2;
        TileA A3;
        uint32_t min0;
        uint32_t min1;
        float2 dm0;
        float2 dm1;
        {
            Q2KWeightHalf w;
            load_q2k_weight_half_raw_fast(w, s_raw, k128_iter);
            make_A_tile<2>(A2, w);
            make_A_tile<3>(A3, w);
            min0 = min_pack4_for_pair<2, 3>(w.sc_r0_lo4, w.sc_r0_hi4);
            min1 = min_pack4_for_pair<2, 3>(w.sc_r1_lo4, w.sc_r1_hi4);
            dm0 = half2_bits_to_float2(w.dm_r0);
            dm1 = half2_bits_to_float2(w.dm_r1);
        }
#pragma unroll
        for (int nf = 0; nf < NFrag; ++nf) {
            TileC C23;
            TileB B;
            load_B_tile(B, s_q8, d2r_q8_stage(k128_iter), nf, 64);
            ggml_cuda_mma::mma(C23, A2, B);
            load_B_tile(B, s_q8, d2r_q8_stage(k128_iter), nf, 96);
            ggml_cuda_mma::mma(C23, A3, B);
            fold_fragment_k64_fast<2, 3>(
                acc, C23, s_q8_fix, d2r_q8_stage(k128_iter), nf, c0, c1, dm0, dm1, min0, min1);
        }
    }
}

template <typename TileA, typename TileB, typename TileC, int T, int NFrag>
__device__ __forceinline__ void mma_fold_k32_t(
        float (&acc)[NFrag][TileC::ne],
        const Q2KWeightHalf &w,
        const block_q8_1_mmq (&s_q8)[kStages][NFrag][8],
        const Q8ColFixF32 (&s_q8_fix)[kStages][NFrag][8],
        int stage, int k_in_q8, int warp_row0, int M, int col_lo, int col_hi) {
    TileA A;
    make_A_tile<T>(A, w);
    const uint16_t min0 = min_pack_for_t<T>(w.sc_r0_lo4, w.sc_r0_hi4);
    const uint16_t min1 = min_pack_for_t<T>(w.sc_r1_lo4, w.sc_r1_hi4);

#pragma unroll
    for (int nf = 0; nf < NFrag; ++nf) {
        TileB B;
        TileC C;
        load_B_tile(B, s_q8, stage, nf, k_in_q8);
        ggml_cuda_mma::mma(C, A, B);
        fold_fragment_k32(acc, C, s_q8_fix, stage, nf, k_in_q8,
                          warp_row0, M, col_lo, col_hi, w.dm_r0, w.dm_r1,
                          min0, min1);
    }
}

template <typename TileA, typename TileB, typename TileC, int T0, int T1, int NFrag>
__device__ __forceinline__ void mma_fold_k64_pair_t(
        float (&acc)[NFrag][TileC::ne],
        const Q2KWeightHalf &w,
        const block_q8_1_mmq (&s_q8)[kStages][NFrag][8],
        const Q8ColFixF32 (&s_q8_fix)[kStages][NFrag][8],
        int stage, int k_in_q8_pair, int warp_row0, int M, int col_lo, int col_hi) {
    TileA A0;
    TileA A1;
    make_A_tile<T0>(A0, w);
    make_A_tile<T1>(A1, w);
    const uint32_t min0 = min_pack4_for_pair<T0, T1>(w.sc_r0_lo4, w.sc_r0_hi4);
    const uint32_t min1 = min_pack4_for_pair<T0, T1>(w.sc_r1_lo4, w.sc_r1_hi4);

#pragma unroll
    for (int nf = 0; nf < NFrag; ++nf) {
        TileC C;
        TileB B0;
        TileB B1;
        load_B_tile(B0, s_q8, stage, nf, k_in_q8_pair);
        ggml_cuda_mma::mma(C, A0, B0);
        load_B_tile(B1, s_q8, stage, nf, k_in_q8_pair + 32);
        ggml_cuda_mma::mma(C, A1, B1);
        fold_fragment_k64(acc, C, s_q8_fix, stage, nf, k_in_q8_pair,
                          warp_row0, M, col_lo, col_hi, w.dm_r0, w.dm_r1,
                          min0, min1);
    }
}

template <bool FullTile, int NFrag,
          typename TileA, typename TileB, typename TileC>
__device__ __forceinline__ void down_q2k_d2r_mainloop(
        float (&acc)[NFrag][TileC::ne],
        block_q8_1_mmq (&s_q8)[kStages][NFrag][8],
        Q8ColFixF32 (&s_q8_fix)[kStages][NFrag][8],
        Q2KRawWarpStage (&s_raw)[kWarps][kRawStages],
        const volatile SmemInvariants &s_inv) {
    static_assert(NFrag == 1 || NFrag == 2 || NFrag == 4 || NFrag == 8,
                  "unsupported Q2 D2R tile width");
    int guarded_nf_live = 0;
    if constexpr (!FullTile) {
        const int guarded_col_count = s_inv.col_count;
        guarded_nf_live = (guarded_col_count + 7) >> 3;
    }

#pragma unroll
    for (int pf = 0; pf < kStages; ++pf) {
        if (pf < s_inv.k128_iters) {
            if constexpr (FullTile) {
                static_assert(NFrag == kNFrag,
                              "full Q2 D2R path is the tuned 64-column kernel");
                issue_q8_prefetch_fast(s_q8, s_inv, d2r_q8_stage(pf), pf);
            } else if constexpr (NFrag == kNFrag) {
                issue_q8_prefetch<false>(s_q8, s_inv, d2r_q8_stage(pf), pf, d2r_tid());
            } else {
                issue_fused_q8_prefetch<false>(s_q8, s_inv, d2r_q8_stage(pf), pf);
            }
        }
    }

#pragma unroll
    for (int pf = 0; pf < kRawStages; ++pf) {
        if (pf < s_inv.nb) {
            if constexpr (FullTile) {
                issue_q2k_raw_prefetch_fast(s_raw, s_inv, d2r_raw_stage(pf), pf);
            } else {
                issue_q2k_raw_prefetch<false>(
                    s_raw, s_inv, d2r_raw_stage(pf), pf, d2r_warp(), d2r_lane());
            }
        }
    }

    for (int k128_iter = 0;; ++k128_iter) {
        if (k128_iter >= s_inv.k128_iters) {
            break;
        }
        if ((k128_iter & 1) == 0) {
            int keep_raw = 0;
            if constexpr (kRawStages > 1) {
                keep_raw = ((k128_iter >> 1) + 1 < s_inv.nb) ? 1 : 0;
            }
            cp_async_wait_keep(keep_raw);
        }
        __syncthreads();
        if constexpr (FullTile) {
            publish_q8_fix_f32(s_q8_fix, s_q8, d2r_q8_stage(k128_iter), d2r_tid());
        } else {
            publish_q8_fix_f32_guarded(
                s_q8_fix, s_q8, d2r_q8_stage(k128_iter), d2r_tid(), guarded_nf_live << 3);
        }
        __syncthreads();

        if constexpr (FullTile) {
            mma_fold_k128_k64_fast<TileA, TileB, TileC>(
                acc, s_raw, s_q8, s_q8_fix, k128_iter);
        } else {
            const int fold_warp = d2r_warp();
            const int fold_group = d2r_group();
            const int fold_warp_row0 = s_inv.cta_row0 + (fold_warp << 4);
            const int fold_M = s_inv.M;
            const int fold_col_lo = s_inv.col_lo;
            const int fold_col_hi = fold_col_lo + s_inv.col_count;
            const bool fold_row0_ok = (fold_warp_row0 + fold_group) < fold_M;
            const bool fold_row1_ok = (fold_warp_row0 + fold_group + 8) < fold_M;
            const int raw_stage = d2r_raw_stage(k128_iter >> 1);
            const int raw_half = k128_iter & 1;
            const int parity = fold_group & 1;
            mma_fold_k128_k64<TileA, TileB, TileC, false>(
                acc, s_raw, s_q8, s_q8_fix, raw_stage,
                fold_row0_ok, fold_row1_ok,
                parity, raw_half, fold_warp, fold_group, d2r_tig(),
                d2r_q8_stage(k128_iter), fold_warp_row0, fold_M, fold_col_lo, fold_col_hi,
                guarded_nf_live);
        }

        __syncthreads();
        const int pf_iter = k128_iter + kStages;
        if (pf_iter < s_inv.k128_iters) {
            if constexpr (FullTile) {
                issue_q8_prefetch_fast(s_q8, s_inv, d2r_q8_stage(pf_iter), pf_iter);
            } else if constexpr (NFrag == kNFrag) {
                issue_q8_prefetch<false>(
                    s_q8, s_inv, d2r_q8_stage(pf_iter), pf_iter, d2r_tid());
            } else {
                issue_fused_q8_prefetch<false>(
                    s_q8, s_inv, d2r_q8_stage(pf_iter), pf_iter);
            }
        }
        if ((k128_iter & 1) != 0) {
            const int raw_pf = (k128_iter >> 1) + kRawStages;
            if (raw_pf < s_inv.nb) {
                if constexpr (FullTile) {
                    issue_q2k_raw_prefetch_fast(s_raw, s_inv, d2r_raw_stage(raw_pf), raw_pf);
                } else {
                    issue_q2k_raw_prefetch<false>(
                        s_raw, s_inv, d2r_raw_stage(raw_pf), raw_pf, d2r_warp(), d2r_lane());
                }
            }
        }
    }
}


template <int TileN>
__global__ __launch_bounds__(kThreads, 2)
void d2r_build_worklist_kernel(const int32_t * __restrict__ expert_bounds,
                               int * __restrict__ work,
                               int * __restrict__ n_items_out,
                               int n_experts) {
    static_assert(TileN > 0, "worklist tile width must be positive");
    __shared__ int scan[kThreads];
    __shared__ int running;
    __shared__ int chunk_base;

    const int tid = (int)threadIdx.x;
    if (tid == 0) {
        running = 0;
    }
    __syncthreads();

    for (int base = 0; base < n_experts; base += kThreads) {
        const int expert = base + tid;
        int tiles = 0;
        if (expert < n_experts) {
            const int count = expert_bounds[expert + 1] - expert_bounds[expert];
            tiles = count > 0 ? ((count + TileN - 1) / TileN) : 0;
        }
        scan[tid] = tiles;
        __syncthreads();

#pragma unroll
        for (int offset = 1; offset < kThreads; offset <<= 1) {
            const int add = tid >= offset ? scan[tid - offset] : 0;
            __syncthreads();
            scan[tid] += add;
            __syncthreads();
        }

        if (tid == 0) {
            chunk_base = running;
        }
        __syncthreads();

        if (expert < n_experts && tiles > 0) {
            const int exclusive = tid == 0 ? 0 : scan[tid - 1];
            const int out_base = chunk_base + exclusive;
            for (int jt = 0; jt < tiles; ++jt) {
                work[out_base + jt] = (expert << 16) | jt;
            }
        }
        __syncthreads();

        if (tid == 0) {
            running += scan[kThreads - 1];
        }
        __syncthreads();
    }

    if (tid == 0) {
        *n_items_out = running;
    }
}

template <int TileN, int BaseTileN>
__global__ __launch_bounds__(kThreads, 2)
void down_q2k_d2r_kernel(const void * __restrict__ W_soa,
                         const block_q8_1_mmq * __restrict__ q8,
                         const int32_t * __restrict__ ids_dst,
                         const int32_t * __restrict__ expert_bounds,
                         const int * __restrict__ work,
                         const int * __restrict__ n_items_ptr,
                         float * __restrict__ out,
                         int M, int K, int n_assign, int E) {
#if defined(TURING_MMA_AVAILABLE)
    const int n_items = *n_items_ptr;
    if ((int)blockIdx.y >= n_items) {
        return;
    }

    const int packed = work[blockIdx.y];
    const int expert = packed >> 16;
    const int jt = packed & 0xFFFF;
    if (expert >= E) {
        return;
    }
    static_assert(TileN == 8 || TileN == 16 || TileN == 32 || TileN == 64,
                  "unsupported Q2 D2R tile width");
    static_assert(BaseTileN == kNTile,
                  "Q2 D2R tail must follow the 64-column full tiles");
    constexpr int NFrag = TileN / 8;

    using tile_A = ggml_cuda_mma::tile<16, 8, int>;
    using tile_B = ggml_cuda_mma::tile<8, 8, int>;
    using tile_C = ggml_cuda_mma::tile<16, 8, int>;

    __shared__ __align__(16) block_q8_1_mmq s_q8[kStages][NFrag][8];
    __shared__ __align__(16) Q8ColFixF32 s_q8_fix[kStages][NFrag][8];
    // CFG1/NT64 static shared: q8 = 2*(8*8*144 + 8*8*40) = 23,552 B;
    // raw = 8 warps*2*(2*8*9*8 qs + 2*8*16 scales + 8*8 dm) = 23,552 B;
    // invariant table <= 128 B; total stays below the 48 KiB launch target. The
    // qs stride pads one uint2 per row-pair half so each lane's four raw uint2
    // reads rotate banks.
    __shared__ __align__(16) Q2KRawWarpStage s_raw[kWarps][kRawStages];
    __shared__ __align__(16) volatile SmemInvariants s_inv;
    /* Scatter indices staged up front so the epilogue's dependent STG chain
     * never waits on an ids_dst LDG (68% of this kernel's long-scoreboard
     * stalls sat on that load, cmd2rncu15b PC sampling). */
    __shared__ int s_out_cols[TileN];

    const int col_lo = expert_bounds[expert] + jt * BaseTileN;
    const int col_hi_full = expert_bounds[expert + 1];
    const int col_tile_hi =
        (col_hi_full < col_lo + TileN) ? col_hi_full : (col_lo + TileN);
    if (col_lo >= col_tile_hi) {
        return;
    }

    const int cta_row0 = (int)blockIdx.x * kMTile;
    const int warp_row0 = cta_row0 + d2r_warp() * 16;
    const int nb = K >> 8;

    if (d2r_tid() == 0) {
        const uint64_t npair = (uint64_t)E * (uint64_t)(M >> 1) * (uint64_t)nb;
        const uint64_t dm_bytes = (npair * 8ull + 63ull) & ~63ull;
        const uint64_t sc_bytes = (npair * 32ull + 63ull) & ~63ull;
        s_inv.w_base = (const char *)W_soa;
        s_inv.q8_tile_base = (const char *)q8 + (uint64_t)col_lo * sizeof(block_q8_1_mmq);
        s_inv.out = out;
        s_inv.sc_off_bytes = (uint32_t)dm_bytes;
        s_inv.qs_off_bytes = (uint32_t)(dm_bytes + sc_bytes);
        s_inv.q8_k128_stride_bytes = (uint32_t)((uint64_t)n_assign * sizeof(block_q8_1_mmq));
        s_inv.nb = nb;
        s_inv.k128_iters = K >> 7;
        s_inv.M = M;
        s_inv.cta_row0 = cta_row0;
        s_inv.col_lo = col_lo;
        s_inv.col_count = col_tile_hi - col_lo;
    }
    if (d2r_lane() == 0) {
        const uint64_t expert_row_pair = (uint64_t)expert * (uint64_t)(M >> 1);
        const uint64_t warp_pair0_base =
            (expert_row_pair + (uint64_t)(warp_row0 >> 1)) * (uint64_t)nb;
        s_inv.warp_pair0_blk[d2r_warp()] = (uint32_t)warp_pair0_base;
    }
    if (d2r_tid() < col_tile_hi - col_lo) {
        s_out_cols[d2r_tid()] = ids_dst[col_lo + d2r_tid()];
    }
    __syncthreads();

    float acc[NFrag][tile_C::ne] = {};

    if constexpr (TileN == kNTile) {
        const bool full_warp_tile =
            (warp_row0 + 15 < M) && (col_lo + TileN <= col_hi_full);
        if (full_warp_tile) {
            down_q2k_d2r_mainloop<true, NFrag, tile_A, tile_B, tile_C>(
                acc, s_q8, s_q8_fix, s_raw, s_inv);
        } else {
            down_q2k_d2r_mainloop<false, NFrag, tile_A, tile_B, tile_C>(
                acc, s_q8, s_q8_fix, s_raw, s_inv);
        }
    } else {
        down_q2k_d2r_mainloop<false, NFrag, tile_A, tile_B, tile_C>(
            acc, s_q8, s_q8_fix, s_raw, s_inv);
    }

    const int out_col_lo = s_inv.col_lo;
    const int out_col_hi = out_col_lo + s_inv.col_count;
    const int out_warp_row0 = s_inv.cta_row0 + (d2r_warp() << 4);
    const int out_M = s_inv.M;
    float *out_base = s_inv.out;
#pragma unroll
    for (int nf = 0; nf < NFrag; ++nf) {
        const int col_frag0 = out_col_lo + nf * 8;
#pragma unroll
        for (int l = 0; l < tile_C::ne; ++l) {
            const int row = out_warp_row0 + tile_C::get_i(l);
            const int col = col_frag0 + tile_C::get_j(l);
            if (row < out_M && col < out_col_hi) {
                const int out_col = s_out_cols[col - out_col_lo];
                out_base[(uint64_t)out_col * (uint64_t)out_M + (uint64_t)row] = acc[nf][l];
            }
        }
    }
#else
    (void)W_soa;
    (void)q8;
    (void)ids_dst;
    (void)expert_bounds;
    (void)work;
    (void)n_items_ptr;
    (void)out;
    (void)M;
    (void)K;
    (void)n_assign;
    (void)E;
#endif
}

__global__ __launch_bounds__(kThreads, 2)
void gateup_iq2_d2r_pair_kernel(const void * __restrict__ gate_soa,
                                const void * __restrict__ up_soa,
                                const block_q8_1_mmq * __restrict__ q8,
                                const int32_t * __restrict__ ids_dst,
                                const int32_t * __restrict__ expert_bounds,
                                const int * __restrict__ work,
                                const int * __restrict__ n_items_ptr,
                                float * __restrict__ out_gate,
                                float * __restrict__ out_up,
                                int M, int K, int n_assign, int E) {
#if defined(TURING_MMA_AVAILABLE)
    const int n_items = *n_items_ptr;
    if ((int)blockIdx.y >= n_items) {
        return;
    }

    const int packed = work[blockIdx.y];
    const int expert = packed >> 16;
    const int jt = packed & 0xFFFF;
    const int leg = (int)blockIdx.z;
    if (expert >= E || leg >= 2) {
        return;
    }

    const int col_lo = expert_bounds[expert] + jt * kNTile;
    const int col_hi_full = expert_bounds[expert + 1];
    const int col_tile_hi = (col_hi_full < col_lo + kNTile) ? col_hi_full : (col_lo + kNTile);
    if (col_lo >= col_tile_hi) {
        return;
    }

    using tile_A = ggml_cuda_mma::tile<16, 8, int>;
    using tile_B = ggml_cuda_mma::tile<8, 8, int>;
    using tile_C = ggml_cuda_mma::tile<16, 8, int>;

    __shared__ __align__(16) block_q8_1_mmq s_q8[kStages][kNFrag][8];
    __shared__ __align__(16) IQ2RawWarpStage s_raw[kWarps][kRawStages];
    __shared__ __align__(16) uint2 s_grid[256];
    __shared__ __align__(16) volatile SmemInvariants s_inv;
    /* Same scatter-index staging as down_q2k_d2r_kernel (see comment there). */
    __shared__ int s_out_cols[kNTile];

    const void *W_soa = leg == 0 ? gate_soa : up_soa;
    float *out = leg == 0 ? out_gate : out_up;
    const int cta_row0 = (int)blockIdx.x * kMTile;
    const int warp_row0 = cta_row0 + d2r_warp() * 16;
    const int nb = K >> 8;

    const bool full_warp_tile = (warp_row0 + 15 < M) && (col_lo + kNTile <= col_hi_full);

    for (int i = d2r_tid(); i < 256; i += kThreads) {
        s_grid[i] = reinterpret_cast<const uint2 *>(iq2xxs_grid)[i];
    }
    if (d2r_tid() == 0) {
        const uint64_t nblk = (uint64_t)E * (uint64_t)M * (uint64_t)nb;
        const uint64_t dq_bytes = (nblk * 2ull + 63ull) & ~63ull;
        s_inv.w_base = (const char *)W_soa;
        s_inv.iq2_dq_base = reinterpret_cast<const half *>(W_soa);
        s_inv.iq2_qs_base =
            reinterpret_cast<const uint2 *>(reinterpret_cast<const char *>(W_soa) + dq_bytes);
        s_inv.q8_tile_base = reinterpret_cast<const char *>(q8) + (uint64_t)col_lo * sizeof(block_q8_1_mmq);
        s_inv.out = out;
        s_inv.sc_off_bytes = 0;
        s_inv.qs_off_bytes = 0;
        s_inv.q8_k128_stride_bytes = (uint32_t)((uint64_t)n_assign * sizeof(block_q8_1_mmq));
        s_inv.nb = nb;
        s_inv.k128_iters = K >> 7;
        s_inv.M = M;
        s_inv.cta_row0 = cta_row0;
        s_inv.col_lo = col_lo;
        s_inv.col_count = col_tile_hi - col_lo;
    }
    if (d2r_lane() == 0) {
        const uint64_t expert_row = (uint64_t)expert * (uint64_t)M;
        const uint64_t warp_row0_base = (expert_row + (uint64_t)warp_row0) * (uint64_t)nb;
        s_inv.warp_row0_blk[d2r_warp()] = (uint32_t)warp_row0_base;
    }
    if (d2r_tid() < col_tile_hi - col_lo) {
        s_out_cols[d2r_tid()] = ids_dst[col_lo + d2r_tid()];
    }
    __syncthreads();

    float acc[kNFrag][tile_C::ne] = {};

    if (full_warp_tile) {
        iq2_d2r_mainloop<true, tile_A, tile_B, tile_C>(
            acc, s_q8, s_raw, s_grid, s_inv);
    } else {
        iq2_d2r_mainloop<false, tile_A, tile_B, tile_C>(
            acc, s_q8, s_raw, s_grid, s_inv);
    }

    const int out_col_lo = s_inv.col_lo;
    const int out_col_hi = out_col_lo + s_inv.col_count;
    const int out_warp_row0 = s_inv.cta_row0 + (d2r_warp() << 4);
    const int out_M = s_inv.M;
    float *out_base = s_inv.out;
#pragma unroll
    for (int nf = 0; nf < kNFrag; ++nf) {
        const int col_frag0 = out_col_lo + nf * 8;
#pragma unroll
        for (int l = 0; l < tile_C::ne; ++l) {
            const int row = out_warp_row0 + tile_C::get_i(l);
            const int col = col_frag0 + tile_C::get_j(l);
            if (row < out_M && col < out_col_hi) {
                const int out_col = s_out_cols[col - out_col_lo];
                out_base[(uint64_t)out_col * (uint64_t)out_M + (uint64_t)row] = acc[nf][l];
            }
        }
    }
#else
    (void)gate_soa;
    (void)up_soa;
    (void)q8;
    (void)ids_dst;
    (void)expert_bounds;
    (void)work;
    (void)n_items_ptr;
    (void)out_gate;
    (void)out_up;
    (void)M;
    (void)K;
    (void)n_assign;
    (void)E;
#endif
}

template <int TileN, int BaseTileN>
__global__ __launch_bounds__(kThreads, 2)
void gateup_iq2_swiglu_q8_d2r_kernel(
        const void * __restrict__ gate_soa,
        const void * __restrict__ up_soa,
        const block_q8_1_mmq * __restrict__ input_q8,
        const int32_t * __restrict__ ids_src,
        const int32_t * __restrict__ ids_dst,
        const int32_t * __restrict__ expert_bounds,
        const float * __restrict__ router_weights,
        const int * __restrict__ work,
        const int * __restrict__ n_items_ptr,
        block_q8_1_mmq * __restrict__ down_q8,
        int M, int K, int n_assign, int n_tokens, int E, float clamp) {
#if defined(TURING_MMA_AVAILABLE)
    const int n_items = *n_items_ptr;
    if ((int)blockIdx.y >= n_items) {
        return;
    }

    const int packed = work[blockIdx.y];
    const int expert = packed >> 16;
    const int jt = packed & 0xFFFF;
    if (expert >= E) {
        return;
    }

    static_assert(TileN == 8 || TileN == 16 || TileN == 32,
                  "unsupported fused D2R tile width");
    static_assert(BaseTileN == kFusedNTile,
                  "fused D2R tail must follow the 32-column full tiles");
    constexpr int NFrag = TileN / 8;
    const int col_lo = expert_bounds[expert] + jt * BaseTileN;
    const int col_hi_full = expert_bounds[expert + 1];
    const int col_tile_hi =
        (col_hi_full < col_lo + TileN) ? col_hi_full : (col_lo + TileN);
    if (col_lo >= col_tile_hi) {
        return;
    }

    using tile_A = ggml_cuda_mma::tile<16, 8, int>;
    using tile_B = ggml_cuda_mma::tile<8, 8, int>;
    using tile_C = ggml_cuda_mma::tile<16, 8, int>;

    __shared__ FusedGateUpSmem<TileN> s_smem;
    __shared__ float s_route_weight[TileN];

    FusedGateUpComputeSmem<TileN> &s = s_smem.compute;
    const int tid = d2r_tid();
    const int cta_row0 = (int)blockIdx.x * kMTile;
    const int warp_row0 = cta_row0 + d2r_warp() * 16;
    const int nb = K >> 8;
    const int col_count = col_tile_hi - col_lo;
    const bool full_tile =
        cta_row0 + kMTile <= M && col_lo + TileN <= col_hi_full;

    for (int i = tid; i < 256; i += kThreads) {
        s.grid[i] = reinterpret_cast<const uint2 *>(iq2xxs_grid)[i];
    }
    if (tid == 0) {
        const void *weights[2] = {gate_soa, up_soa};
        const uint64_t nblk = (uint64_t)E * (uint64_t)M * (uint64_t)nb;
        const uint64_t dq_bytes = (nblk * 2ull + 63ull) & ~63ull;
#pragma unroll
        for (int leg = 0; leg < 2; ++leg) {
            s.inv[leg].w_base = (const char *)weights[leg];
            s.inv[leg].iq2_dq_base = reinterpret_cast<const half *>(weights[leg]);
            s.inv[leg].iq2_qs_base = reinterpret_cast<const uint2 *>(
                reinterpret_cast<const char *>(weights[leg]) + dq_bytes);
            /* p5b: with ids_src the input Q8 buffer is token-compact
             * (quantized once per token, n_tokens row stride) and the
             * column -> row map rides the per-thread y_off below; the
             * slot-gathered form keeps the column offset in the base. */
            s.inv[leg].q8_tile_base = reinterpret_cast<const char *>(input_q8) +
                                      (ids_src ? 0ull
                                               : (uint64_t)col_lo * sizeof(block_q8_1_mmq));
            s.inv[leg].out = nullptr;
            s.inv[leg].sc_off_bytes = 0;
            s.inv[leg].qs_off_bytes = 0;
            s.inv[leg].q8_k128_stride_bytes =
                (uint32_t)((uint64_t)(ids_src ? n_tokens : n_assign) *
                           sizeof(block_q8_1_mmq));
            s.inv[leg].nb = nb;
            s.inv[leg].k128_iters = K >> 7;
            s.inv[leg].M = M;
            s.inv[leg].cta_row0 = cta_row0;
            s.inv[leg].col_lo = col_lo;
            s.inv[leg].col_count = col_count;
        }
    }
    if (d2r_lane() == 0) {
        const uint64_t expert_row = (uint64_t)expert * (uint64_t)M;
        const uint64_t warp_row0_base =
            (expert_row + (uint64_t)warp_row0) * (uint64_t)nb;
        s.inv[0].warp_row0_blk[d2r_warp()] = (uint32_t)warp_row0_base;
        s.inv[1].warp_row0_blk[d2r_warp()] = (uint32_t)warp_row0_base;
    }
    if (tid < col_count) {
        const int pair = ids_dst[col_lo + tid];
        s_route_weight[tid] = router_weights[pair];
    }
    /* p5b: per-thread Y row byte offset (see issue_fused_q8_prefetch_one).
     * Out-of-segment columns are cp.async-predicated off, so their map
     * entry is never read; clamp to 0 to keep the ids_src load in range. */
    const int y_col_local = tid & (TileN - 1);
    uint32_t y_off;
    if (ids_src) {
        y_off = (y_col_local < col_count)
            ? (uint32_t)ids_src[col_lo + y_col_local] *
              (uint32_t)sizeof(block_q8_1_mmq)
            : 0u;
    } else {
        y_off = (uint32_t)y_col_local * (uint32_t)sizeof(block_q8_1_mmq);
    }
    __syncthreads();

    float gate_acc[NFrag][tile_C::ne] = {};
    float up_acc[NFrag][tile_C::ne] = {};
    if (full_tile) {
        iq2_gateup_fused_mainloop<true, TileN, tile_A, tile_B, tile_C>(
            gate_acc, up_acc, s, y_off);
    } else {
        iq2_gateup_fused_mainloop<false, TileN, tile_A, tile_B, tile_C>(
            gate_acc, up_acc, s, y_off);
    }

    /* The MMA rings are dead now. Reuse the same shared allocation as a
     * column-major float tile so the quantization reduction matches the
     * established Q8_1 D2S6 kernel: one warp owns one 128-row column. */
    __syncthreads();
#pragma unroll
    for (int nf = 0; nf < NFrag; ++nf) {
#pragma unroll
        for (int l = 0; l < tile_C::ne; ++l) {
            const int row = d2r_warp() * 16 + tile_C::get_i(l);
            const int col = nf * 8 + tile_C::get_j(l);
            if (cta_row0 + row < M && col < col_count) {
                float gate = isfinite(gate_acc[nf][l]) ? gate_acc[nf][l] : 0.0f;
                float up = isfinite(up_acc[nf][l]) ? up_acc[nf][l] : 0.0f;
                if (clamp > 1.0e-6f) {
                    gate = fminf(gate, clamp);
                    up = fminf(fmaxf(up, -clamp), clamp);
                }
                const float swiglu = (gate / (1.0f + expf(-gate))) * up;
                s_smem.mid[col][row] = swiglu * s_route_weight[col];
            }
        }
    }
    __syncthreads();

    const int lane = d2r_lane();
    for (int col = d2r_warp(); col < col_count; col += kWarps) {
        const float4 xi = *reinterpret_cast<const float4 *>(
            &s_smem.mid[col][lane * 4]);
        float amax = fabsf(xi.x);
        amax = fmaxf(amax, fabsf(xi.y));
        amax = fmaxf(amax, fabsf(xi.z));
        amax = fmaxf(amax, fabsf(xi.w));
#pragma unroll
        for (int offset = 8; offset > 0; offset >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, offset, 32));
        }

        float sum = xi.x + xi.y + xi.z + xi.w;
#pragma unroll
        for (int offset = 2; offset > 0; offset >>= 1) {
            sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset, 32);
        }

        const float d_inv = 127.0f / amax;
        char4 q;
        q.x = roundf(xi.x * d_inv);
        q.y = roundf(xi.y * d_inv);
        q.z = roundf(xi.z * d_inv);
        q.w = roundf(xi.w * d_inv);

        block_q8_1_mmq &out_block =
            down_q8[(uint64_t)(cta_row0 / kMTile) * (uint64_t)n_assign +
                    (uint64_t)(col_lo + col)];
        reinterpret_cast<char4 *>(out_block.qs)[lane] = q;
        if ((lane & 3) == 0 && lane < 24) {
            out_block.d2s6[2 + lane / 4] = __float2half(sum);
        }
        if ((lane & 15) == 0) {
            out_block.d2s6[lane / 16] = __float2half(1.0f / d_inv);
        }
    }
#else
    (void)gate_soa;
    (void)up_soa;
    (void)input_q8;
    (void)ids_src;
    (void)ids_dst;
    (void)expert_bounds;
    (void)router_weights;
    (void)work;
    (void)n_items_ptr;
    (void)down_q8;
    (void)M;
    (void)K;
    (void)n_assign;
    (void)n_tokens;
    (void)E;
    (void)clamp;
#endif
}

static int64_t d2r_work_capacity_for_tile(
        int64_t ncols_max, int n_experts, int tile_n) {
    if (ncols_max <= 0 || n_experts <= 0 || tile_n <= 0) {
        return 0;
    }
    return (ncols_max + tile_n - 1) / tile_n + (int64_t)n_experts;
}

static int64_t d2r_work_capacity(int64_t ncols_max, int n_experts) {
    return d2r_work_capacity_for_tile(ncols_max, n_experts, kNTile);
}

} // namespace

bool ds4_mmq_q2_K_moe_d2r_available(int cc) {
    static int cached_cc = -1;
    static int cached = 0;
    if (cached_cc != cc) {
        cached_cc = cc;
        cached = (GGML_CUDA_CC_IS_NVIDIA(cc) &&
                  ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_AMPERE) ? 1 : 0;
    }
    return cached != 0;
}

bool ds4_mmq_iq2_xxs_moe_d2r_available(int cc) {
    static int cached_cc = -1;
    static int cached = 0;
    if (cached_cc != cc) {
        cached_cc = cc;
        cached = (GGML_CUDA_CC_IS_NVIDIA(cc) &&
                  ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_AMPERE) ? 1 : 0;
    }
    return cached != 0;
}

size_t ds4_mmq_q2_K_moe_d2r_scratch_bytes(int64_t ncols_max, int n_experts) {
    const int64_t capacity = d2r_work_capacity(ncols_max, n_experts);
    if (capacity <= 0 || capacity > (int64_t)(INT_MAX - 1)) {
        return 0;
    }
    return (size_t)capacity * sizeof(int) + sizeof(int);
}

size_t ds4_mmq_iq2_xxs_moe_d2r_pair_scratch_bytes(int64_t ncols_max, int n_experts) {
    const int64_t capacity = d2r_work_capacity(ncols_max, n_experts);
    if (capacity <= 0 || capacity > (int64_t)(INT_MAX - 1)) {
        return 0;
    }
    return (size_t)capacity * sizeof(int) + sizeof(int);
}

size_t ds4_mmq_iq2_xxs_moe_d2r_fused_scratch_bytes(
        int64_t ncols_max, int n_experts) {
    const int64_t capacity =
        d2r_work_capacity_for_tile(ncols_max, n_experts, kFusedNTile);
    if (capacity <= 0 || capacity > (int64_t)(INT_MAX - 1)) {
        return 0;
    }
    return (size_t)capacity * sizeof(int) + sizeof(int);
}

int ds4_mmq_q2_K_moe_d2r_launch(const void *W_soa,
                                int64_t soa_blocks,
                                const void *q8,
                                const int32_t *ids_dst,
                                const int32_t *expert_bounds,
                                float *out,
                                int M,
                                int K,
                                int64_t ne_get_rows,
                                int n_experts,
                                void *worklist_scratch,
                                size_t worklist_scratch_bytes,
                                cudaStream_t stream) {
    const char *tag = "ds4_mmq_q2_K_moe_d2r_launch";
    const int dev = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[dev].cc;
    if (!ds4_mmq_q2_K_moe_d2r_available(cc)) {
        return 1;
    }
    if (!W_soa || !q8 || !ids_dst || !expert_bounds || !out || !worklist_scratch ||
        M <= 0 || (M & 1) != 0 || K <= 0 || K % 256 != 0 || ne_get_rows <= 0 ||
        ne_get_rows > INT_MAX || n_experts <= 0) {
        return -1;
    }

    const int64_t expected_soa_blocks = (int64_t)n_experts * (int64_t)(M >> 1) * (int64_t)(K >> 8);
    if (soa_blocks < expected_soa_blocks) {
        return -1;
    }

    const int64_t capacity64 = d2r_work_capacity(ne_get_rows, n_experts);
    if (capacity64 <= 0 || capacity64 > (int64_t)(INT_MAX - 1)) {
        return -1;
    }
    const size_t needed = (size_t)capacity64 * sizeof(int) + sizeof(int);
    if (worklist_scratch_bytes < needed) {
        return -1;
    }

    int *work = (int *)worklist_scratch;
    int *n_items = work + capacity64;

    /* Single guarded launch: at the serving chunk profile (2048-token
     * chunks -> ~48 assignments per expert bucket) nearly EVERY expert is a
     * ragged tail, so the tail-specialized 8/16/32/64 decomposition turns
     * one dense launch into five mostly-sparse ones and the relaunch +
     * occupancy-ramp tax dominates (reslice6: 64% of down time in tail
     * launches, +1.2 s/180k vs this shape).  The guarded 64-wide path
     * inside the kernel handles ragged tiles exactly as it always did. */
    d2r_build_worklist_kernel<kNTile><<<1, kThreads, 0, stream>>>(
        expert_bounds, work, n_items, n_experts);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: worklist builder launch failed: %s\n", tag, cudaGetErrorString(err));
        return -2;
    }

    if (d2r_stats_enabled()) {
        d2r_print_fill_stats(tag, expert_bounds, n_experts, ne_get_rows, stream);
    }

    /* Row tiles ride blockIdx.x (fastest) and the expert-ordered worklist
     * rides blockIdx.y, so consecutive CTAs share one col-tile's q8 window
     * and consecutive worklist items share one expert's weight slab -- the
     * proto's expert-major L2 schedule.  The flat col-tile-fastest order
     * re-read each q8 window through ~94 MB of intervening traffic (all
     * misses, lts hit 55% vs proto 84%). */
    const dim3 block(32, kWarps, 1);
    const unsigned row_tiles = (unsigned)((M + kMTile - 1) / kMTile);
    const dim3 grid(row_tiles, (unsigned)capacity64, 1);
    down_q2k_d2r_kernel<64, kNTile><<<grid, block, 0, stream>>>(
        W_soa, (const block_q8_1_mmq *)q8, ids_dst, expert_bounds,
        work, n_items, out,
        M, K, (int)ne_get_rows, n_experts);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: main kernel launch failed: %s\n", tag, cudaGetErrorString(err));
        return -3;
    }
    return 0;
}

int ds4_mmq_iq2_xxs_moe_d2r_pair_launch(const void *gate_soa,
                                         const void *up_soa,
                                         int64_t soa_blocks,
                                         const void *q8,
                                         const int32_t *ids_dst,
                                         const int32_t *expert_bounds,
                                         float *out_gate,
                                         float *out_up,
                                         int M,
                                         int K,
                                         int64_t ne_get_rows,
                                         int n_experts,
                                         void *worklist_scratch,
                                         size_t worklist_scratch_bytes,
                                         cudaStream_t stream) {
    const char *tag = "ds4_mmq_iq2_xxs_moe_d2r_pair_launch";
    const int dev = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[dev].cc;
    if (!ds4_mmq_iq2_xxs_moe_d2r_available(cc)) {
        return 1;
    }
    if (!gate_soa || !up_soa || !q8 || !ids_dst || !expert_bounds || !out_gate || !out_up ||
        !worklist_scratch || M <= 0 || K <= 0 || K % 256 != 0 || ne_get_rows <= 0 ||
        ne_get_rows > INT_MAX || n_experts <= 0) {
        return -1;
    }

    const int64_t expected_soa_blocks =
        (int64_t)n_experts * (int64_t)M * (int64_t)(K >> 8);
    if (soa_blocks < expected_soa_blocks) {
        return -1;
    }

    const int64_t capacity64 = d2r_work_capacity(ne_get_rows, n_experts);
    if (capacity64 <= 0 || capacity64 > (int64_t)(INT_MAX - 1)) {
        return -1;
    }
    const size_t needed = (size_t)capacity64 * sizeof(int) + sizeof(int);
    if (worklist_scratch_bytes < needed) {
        return -1;
    }

    int *work = (int *)worklist_scratch;
    int *n_items = work + capacity64;

    d2r_build_worklist_kernel<kNTile><<<1, kThreads, 0, stream>>>(
        expert_bounds, work, n_items, n_experts);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: worklist builder launch failed: %s\n", tag, cudaGetErrorString(err));
        return -2;
    }

    /* Same expert-major schedule as the down launch (see comment there). */
    const dim3 grid((unsigned)((M + kMTile - 1) / kMTile), (unsigned)capacity64, 2);
    const dim3 block(32, kWarps, 1);
    gateup_iq2_d2r_pair_kernel<<<grid, block, 0, stream>>>(
        gate_soa, up_soa, (const block_q8_1_mmq *)q8, ids_dst, expert_bounds, work, n_items,
        out_gate, out_up, M, K, (int)ne_get_rows, n_experts);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: main kernel launch failed: %s\n", tag, cudaGetErrorString(err));
        return -3;
    }
    return 0;
}

int ds4_mmq_iq2_xxs_moe_d2r_fused_launch(
        const void *gate_soa,
        const void *up_soa,
        int64_t soa_blocks,
        const void *input_q8,
        const int32_t *ids_src,
        int n_tokens,
        const int32_t *ids_dst,
        const int32_t *expert_bounds,
        const float *router_weights,
        void *down_q8,
        int M,
        int K,
        int64_t ne_get_rows,
        int n_experts,
        float clamp,
        void *worklist_scratch,
        size_t worklist_scratch_bytes,
        cudaStream_t stream) {
    const char *tag = "ds4_mmq_iq2_xxs_moe_d2r_fused_launch";
    const int dev = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[dev].cc;
    if (!ds4_mmq_iq2_xxs_moe_d2r_available(cc)) {
        return 1;
    }
    if (!gate_soa || !up_soa || !input_q8 || !ids_dst || !expert_bounds ||
        !router_weights || !down_q8 || !worklist_scratch ||
        M <= 0 || M % kMTile != 0 || K <= 0 || K % 256 != 0 ||
        ne_get_rows <= 0 || ne_get_rows > INT_MAX || n_experts <= 0) {
        return -1;
    }
    /* p5b: ids_src selects the token-compact input layout; the compact row
     * count must cover every map entry (entries are token indices). */
    if (ids_src && (n_tokens <= 0 || (int64_t)n_tokens > ne_get_rows)) {
        return -1;
    }

    const int64_t expected_soa_blocks =
        (int64_t)n_experts * (int64_t)M * (int64_t)(K >> 8);
    if (soa_blocks < expected_soa_blocks) {
        return -1;
    }

    const int64_t capacity32 =
        d2r_work_capacity_for_tile(ne_get_rows, n_experts, kFusedNTile);
    if (capacity32 <= 0 || capacity32 > (int64_t)(INT_MAX - 1)) {
        return -1;
    }
    const size_t needed = (size_t)capacity32 * sizeof(int) + sizeof(int);
    if (worklist_scratch_bytes < needed) {
        return -1;
    }

    int *work = (int *)worklist_scratch;
    int *n_items = work + capacity32;

    /* Single guarded launch (see the down launch comment): the upstream
     * fork's tail-specialized 8/16/32 decomposition assumes big expert
     * buckets (its 8k-token chunks -> ~192 cols/expert, full tiles dominate).
     * At OUR 2048-token serving chunks the mean bucket is ~48 cols, every
     * expert carries a ragged tail, and the three extra launches cost more
     * than the narrow-tile waste they avoid (reslice6: 22.6 of 47.5 s in
     * tail launches).  The kernel's per-CTA full_tile guard handles ragged
     * columns in the one launch. */
    d2r_build_worklist_kernel<kFusedNTile><<<1, kThreads, 0, stream>>>(
        expert_bounds, work, n_items, n_experts);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: worklist builder launch failed: %s\n",
                tag, cudaGetErrorString(err));
        return -2;
    }

    if (d2r_stats_enabled()) {
        d2r_print_fill_stats(tag, expert_bounds, n_experts, ne_get_rows, stream);
    }

    const dim3 block(32, kWarps, 1);
    const dim3 grid((unsigned)(M / kMTile), (unsigned)capacity32, 1);
    gateup_iq2_swiglu_q8_d2r_kernel<32, kFusedNTile><<<grid, block, 0, stream>>>(
        gate_soa, up_soa,
        (const block_q8_1_mmq *)input_q8,
        ids_src, ids_dst, expert_bounds, router_weights, work, n_items,
        (block_q8_1_mmq *)down_q8,
        M, K, (int)ne_get_rows, ids_src ? n_tokens : 0, n_experts, clamp);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: main kernel launch failed: %s\n",
                tag, cudaGetErrorString(err));
        return -3;
    }
    return 0;
}

// ============================================================================
// Dense Q8_0 D2R (2026-07-09, proto_gemm_dense_q8_d2r.cu passes 0-5; arc
// record in local/docs/ds4_d2r_decomposition_analysis_2026-07-08.md "DENSE-Q8
// D2R PROTO ARC").
//
// Reads the kind-5 aligned artifact (CUDA_DERIVED_Q8_0_ALIGNED_DENSE, weight
// server --repack-q8-aligned, ADDITIVE) in place: [half dq[nblk]][pad to 64B]
// [int8 qs[nblk*32]], nblk = M*K/32, raw (row-major) block order.  m128n128
// CTA, 16 warps (8 row x 2 col, one CTA/SM), 3-stage rings at k64 cadence
// with a single barrier per iter (wait -> sync -> issue), split q8 staging
// (d4 headers @k128 + 64 B qs halves @k64, both 80 B padded rows), t-phased
// mma.m16n8k32.s8 with per-k32 float fold.
//
// Proto laws baked in (measured, do not "simplify" back):
//  - row-major weight reads, NOT contiguous-stage tiling (L2 slice camping,
//    -27% at identical instruction count);
//  - t-phased A loads, NOT both-up-front (-27%: hard A dependency at the k64
//    head);
//  - 16-warp CTA: the 2x8-warp NT128 shape spills (acc = 64 regs) and the
//    spill reloads become the top stall;
//  - invariants in plain registers (uniform-RF promotable), no smem table.
//
// vs production mul_mat_q<8,128,0> per-launch (cmttmixnsys15.sqlite): down
// [4096x2048] 3.2x, gate/up [2048x4096] 1.37x, q_b [32768x1024] 1.17x f32.
// o_proj [4096x8192] measured SLOWER (0.81x) - mmq is strong at deep K; the
// dispatch in ds4_cuda.cu keeps K=8192 on mmq.
// ============================================================================

namespace {
namespace dq8 {

constexpr int kDqMTile   = 128;
constexpr int kDqNTile   = 128;
constexpr int kDqRowWarps = 8;
constexpr int kDqColWarps = 2;
constexpr int kDqWarps   = kDqRowWarps * kDqColWarps;
constexpr int kDqThreads = 32 * kDqWarps;
constexpr int kDqStages  = 3;
constexpr int kDqRowPad  = 80;  // 64 B payload + 16 B: 20-int ldmatrix stride rotates banks
constexpr int kDqNFrag   = kDqNTile / 8;
constexpr int kDqNFragPerWarp = kDqNFrag / kDqColWarps;

constexpr size_t kDqSmemWQBytes  = (size_t)kDqStages * kDqMTile * kDqRowPad;
constexpr size_t kDqSmemQ8QBytes = (size_t)kDqStages * kDqNTile * kDqRowPad;
constexpr size_t kDqSmemQ8HBytes = (size_t)kDqStages * kDqNTile * 16;
constexpr size_t kDqSmemWDBytes  = (size_t)kDqStages * kDqMTile * sizeof(uint32_t);
constexpr size_t kDqSmemWQOff  = 0;
constexpr size_t kDqSmemQ8QOff = kDqSmemWQOff + kDqSmemWQBytes;
constexpr size_t kDqSmemQ8HOff = kDqSmemQ8QOff + kDqSmemQ8QBytes;
constexpr size_t kDqSmemWDOff  = kDqSmemQ8HOff + kDqSmemQ8HBytes;
constexpr size_t kDqSmemTotalBytes = kDqSmemWDOff + kDqSmemWDBytes;
static_assert(kDqSmemTotalBytes <= 99ull * 1024ull, "dense D2R dynamic smem exceeds sm limit");

struct DenseQ8Params {
    const char *wd_row0;       // dq plane at cta_row0 (row stride nb*2 = K/16)
    const char *wq_row0;       // qs plane at cta_row0 (row stride K)
    const char *q8_tile;       // q8 blocks at col_lo
    uint32_t q8_k128_stride_bytes;
    uint32_t wq_row_stride;    // K bytes
    int k64_iters;
    int k128_iters;
};

__device__ __forceinline__ void dq8_cp_async_4B(void *dst, const void *src, bool pred) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    if (pred) {
        const unsigned smem = static_cast<unsigned>(__cvta_generic_to_shared(dst));
        asm volatile("cp.async.ca.shared.global [%0], [%1], 4;"
                     :: "r"(smem), "l"(src));
    } else {
        *reinterpret_cast<uint32_t *>(dst) = 0u;
    }
#else
    if (pred) {
        *reinterpret_cast<uint32_t *>(dst) = *reinterpret_cast<const uint32_t *>(src);
    } else {
        *reinterpret_cast<uint32_t *>(dst) = 0u;
    }
#endif
}

__device__ __forceinline__ void dq8_issue_q8_qs(
        int8_t (&s_q8q)[kDqStages][kDqNTile][kDqRowPad],
        const DenseQ8Params &p, int k64_iter) {
    const bool pred = k64_iter < p.k64_iters;
    const char *base = p.q8_tile +
                       (uint64_t)(k64_iter >> 1) * p.q8_k128_stride_bytes +
                       16u + (uint64_t)(k64_iter & 1) * 64u;
    const int buf = k64_iter % kDqStages;
    const int chunk = d2r_tid();
    const int col = chunk >> 2;
    const int h = chunk & 3;
    void *dst = &s_q8q[buf][col][h * 16];
    const void *src = base + (uint64_t)col * sizeof(block_q8_1_mmq) + (uint64_t)h * 16u;
    cp_async_16B(dst, src, pred);
}

__device__ __forceinline__ void dq8_issue_q8_hdr(
        float (&s_q8h)[kDqStages][kDqNTile][4],
        const DenseQ8Params &p, int k128_iter) {
    if (d2r_tid() >= kDqNTile) {
        return;
    }
    const bool pred = k128_iter < p.k128_iters;
    const char *base = p.q8_tile + (uint64_t)k128_iter * p.q8_k128_stride_bytes;
    const int col = d2r_tid();
    void *dst = &s_q8h[k128_iter % kDqStages][col][0];
    const void *src = base + (uint64_t)col * sizeof(block_q8_1_mmq);
    cp_async_16B(dst, src, pred);
}

__device__ __forceinline__ void dq8_issue_w(
        int8_t (&s_wq)[kDqStages][kDqMTile][kDqRowPad],
        uint32_t (&s_wd)[kDqStages][kDqMTile],
        const DenseQ8Params &p, int k64_iter) {
    const bool pred = k64_iter < p.k64_iters;
    const int wst = k64_iter % kDqStages;
    const char *wq = p.wq_row0 + (uint64_t)k64_iter * 64u;
    const char *wd = p.wd_row0 + (uint64_t)k64_iter * 4u;
    const uint32_t wq_row_stride = p.wq_row_stride;
    const uint32_t wd_row_stride = p.wq_row_stride >> 4;  // nb*2 = K/16
    const int chunk = d2r_tid();
    const int row = chunk >> 2;
    const int h = chunk & 3;
    void *dst = &s_wq[wst][row][h * 16];
    const void *src = wq + (uint64_t)row * wq_row_stride + (uint64_t)h * 16u;
    cp_async_16B(dst, src, pred);
    if (d2r_tid() < kDqMTile) {
        void *ddst = &s_wd[wst][d2r_tid()];
        const void *dsrc = wd + (uint64_t)d2r_tid() * wd_row_stride;
        dq8_cp_async_4B(ddst, dsrc, pred);
    }
}

template <typename TileA, typename TileB, typename TileC>
__device__ __forceinline__ void dq8_mainloop(
        float (&acc)[kDqNFragPerWarp][4],
        float (&s_q8h)[kDqStages][kDqNTile][4],
        int8_t (&s_q8q)[kDqStages][kDqNTile][kDqRowPad],
        int8_t (&s_wq)[kDqStages][kDqMTile][kDqRowPad],
        uint32_t (&s_wd)[kDqStages][kDqMTile],
        const DenseQ8Params &p) {
    static_assert(TileC::ne == 4, "expected m16n8 s32 accumulator fragment");
    const int k64_iters = p.k64_iters;
    const int lane = d2r_lane();
    const int group = lane >> 2;
    const int wrow = (d2r_warp() >> 1) * 16;
    const int nf0 = (d2r_warp() & 1) * kDqNFragPerWarp;
    const int c0 = TileC::get_j(0);
    const int c1 = TileC::get_j(1);

    dq8_issue_w(s_wq, s_wd, p, 0);
    dq8_issue_q8_qs(s_q8q, p, 0);
    dq8_issue_q8_hdr(s_q8h, p, 0);
    cp_async_commit();
    dq8_issue_w(s_wq, s_wd, p, 1);
    dq8_issue_q8_qs(s_q8q, p, 1);
    dq8_issue_q8_hdr(s_q8h, p, 1);
    cp_async_commit();

    for (int i = 0; i < k64_iters; ++i) {
        // wait -> barrier -> issue: the barrier both publishes stage i CTA-wide
        // and proves all warps finished stage i-1, whose buffer ((i-1)%3 ==
        // (i+2)%3) the issues below overwrite.
        cp_async_wait_group<1>();
        __syncthreads();
        dq8_issue_w(s_wq, s_wd, p, i + 2);
        dq8_issue_q8_qs(s_q8q, p, i + 2);
        if ((i & 1) == 0) {
            dq8_issue_q8_hdr(s_q8h, p, (i >> 1) + 2);
        }
        cp_async_commit();

        const int wst = i % kDqStages;
        const int qbuf = i % kDqStages;
        const int hbuf = (i >> 1) % kDqStages;
        const uint32_t dw0_bits = s_wd[wst][wrow + group];
        const uint32_t dw1_bits = s_wd[wst][wrow + group + 8];
        const float2 dw0 = __half22float2(*reinterpret_cast<const half2 *>(&dw0_bits));
        const float2 dw1 = __half22float2(*reinterpret_cast<const half2 *>(&dw1_bits));

#pragma unroll
        for (int t = 0; t < 2; ++t) {
            TileA A;
            ggml_cuda_mma::load_ldmatrix(
                A, reinterpret_cast<const int *>(&s_wq[wst][wrow][t * 32]),
                kDqRowPad / (int)sizeof(int));
            const int tq8 = (i & 1) * 2 + t;
            const float dwr0 = t ? dw0.y : dw0.x;
            const float dwr1 = t ? dw1.y : dw1.x;
#pragma unroll
            for (int nf = 0; nf < kDqNFragPerWarp; ++nf) {
                TileB B;
                TileC C;
                ggml_cuda_mma::load_ldmatrix(
                    B, reinterpret_cast<const int *>(&s_q8q[qbuf][(nf0 + nf) * 8][t * 32]),
                    kDqRowPad / (int)sizeof(int));
                ggml_cuda_mma::mma(C, A, B);
                const float da0 = s_q8h[hbuf][(nf0 + nf) * 8 + c0][tq8];
                const float da1 = s_q8h[hbuf][(nf0 + nf) * 8 + c1][tq8];
                acc[nf][0] = fmaf((float)C.x[0], dwr0 * da0, acc[nf][0]);
                acc[nf][1] = fmaf((float)C.x[1], dwr0 * da1, acc[nf][1]);
                acc[nf][2] = fmaf((float)C.x[2], dwr1 * da0, acc[nf][2]);
                acc[nf][3] = fmaf((float)C.x[3], dwr1 * da1, acc[nf][3]);
            }
        }
    }
}

__global__ __launch_bounds__(kDqThreads, 1)
void dense_q8_d2r_kernel(const char * __restrict__ wd_plane,
                         const char * __restrict__ wq_plane,
                         const block_q8_1_mmq * __restrict__ q8,
                         float * __restrict__ out,
                         int M, int N, int K, int group_m) {
#if defined(TURING_MMA_AVAILABLE)
    using tile_A = ggml_cuda_mma::tile<16, 8, int>;
    using tile_B = ggml_cuda_mma::tile<8, 8, int>;
    using tile_C = ggml_cuda_mma::tile<16, 8, int>;

    extern __shared__ __align__(16) char dq8_smem[];
    auto &s_wq  = *reinterpret_cast<int8_t (*)[kDqStages][kDqMTile][kDqRowPad]>(dq8_smem + kDqSmemWQOff);
    auto &s_q8q = *reinterpret_cast<int8_t (*)[kDqStages][kDqNTile][kDqRowPad]>(dq8_smem + kDqSmemQ8QOff);
    auto &s_q8h = *reinterpret_cast<float (*)[kDqStages][kDqNTile][4]>(dq8_smem + kDqSmemQ8HOff);
    auto &s_wd  = *reinterpret_cast<uint32_t (*)[kDqStages][kDqMTile]>(dq8_smem + kDqSmemWDOff);

    // Grouped supertile order: group_m row-tiles stay L2-hot while col tiles
    // stream past them.
    const int num_m = M / kDqMTile;
    const int num_n = (N + kDqNTile - 1) / kDqNTile;
    const int width = group_m * num_n;
    const int g = (int)blockIdx.x / width;
    const int rem = (int)blockIdx.x - g * width;
    int gsize = num_m - g * group_m;
    if (gsize > group_m) {
        gsize = group_m;
    }
    const int pid_m = g * group_m + rem % gsize;
    const int pid_n = rem / gsize;

    const int cta_row0 = pid_m * kDqMTile;
    const int col_lo = pid_n * kDqNTile;

    DenseQ8Params p;
    p.wd_row0 = wd_plane + (uint64_t)cta_row0 * (uint64_t)(K >> 4);
    p.wq_row0 = wq_plane + (uint64_t)cta_row0 * (uint64_t)K;
    p.q8_tile = reinterpret_cast<const char *>(q8) +
                (uint64_t)col_lo * sizeof(block_q8_1_mmq);
    p.q8_k128_stride_bytes = (uint32_t)((uint64_t)N * sizeof(block_q8_1_mmq));
    p.wq_row_stride = (uint32_t)K;
    p.k64_iters = K >> 6;
    p.k128_iters = K >> 7;

    float acc[kDqNFragPerWarp][tile_C::ne] = {};

    dq8_mainloop<tile_A, tile_B, tile_C>(acc, s_q8h, s_q8q, s_wq, s_wd, p);

    // Column-major out [N][M]; rows always in range (M % 128 == 0 validated).
    // The isfinite guard preserves the sanitize contract of the mmq path.
    const int out_col_lo = col_lo + (d2r_warp() & 1) * (kDqNFragPerWarp * 8);
    const int out_row0 = cta_row0 + ((d2r_warp() >> 1) << 4);
#pragma unroll
    for (int nf = 0; nf < kDqNFragPerWarp; ++nf) {
        const int col_frag0 = out_col_lo + nf * 8;
#pragma unroll
        for (int l = 0; l < tile_C::ne; ++l) {
            const int row = out_row0 + tile_C::get_i(l);
            const int col = col_frag0 + tile_C::get_j(l);
            if (col < N) {
                const float v = isfinite(acc[nf][l]) ? acc[nf][l] : 0.0f;
                out[(uint64_t)col * (uint64_t)M + (uint64_t)row] = v;
            }
        }
    }
#else
    GGML_UNUSED_VARS(wd_plane, wq_plane, q8, out, M, N, K, group_m);
    NO_DEVICE_CODE;
#endif
}

} // namespace dq8
} // anonymous namespace

bool ds4_mmq_q8_0_dense_d2r_available(int cc) {
    static int cached_cc = -1;
    static int cached = 0;
    if (cached_cc != cc) {
        cached_cc = cc;
        cached = (GGML_CUDA_CC_IS_NVIDIA(cc) &&
                  ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_AMPERE) ? 1 : 0;
    }
    return cached != 0;
}

// W_aligned = kind-5 artifact base; q8 = block_q8_1_mmq activation buffer
// (D4 layout, [k128][col] with stride N, over-allocated by >= 128 blocks for
// the guarded last col tile - the mmq Y buffer's own mmq_x_max slack covers
// this).  out is column-major [N][M] f32, every element written.
int ds4_mmq_q8_0_dense_d2r_launch(
        const void *W_aligned, const void *q8, float *out,
        int M, int N, int K, cudaStream_t stream) {
    using namespace dq8;
    if (!W_aligned || !q8 || !out ||
        M <= 0 || (M % kDqMTile) != 0 || N <= 0 || K <= 0 ||
        (K % 1024) != 0 || (K >> 6) < kDqStages) {
        fprintf(stderr, "ds4_mmq_q8_0_dense_d2r_launch: bad args M=%d N=%d K=%d\n", M, N, K);
        return -1;
    }
    static int smem_opted = 0;
    if (!smem_opted) {
        const cudaError_t aerr = cudaFuncSetAttribute(
            dense_q8_d2r_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
            (int)kDqSmemTotalBytes);
        if (aerr != cudaSuccess) {
            fprintf(stderr, "ds4_mmq_q8_0_dense_d2r_launch: smem opt-in failed: %s\n",
                    cudaGetErrorString(aerr));
            return -2;
        }
        smem_opted = 1;
    }
    const uint64_t nblk = (uint64_t)M * (uint64_t)(K / 32);
    const uint64_t dq_bytes = (nblk * 2u + 63u) & ~63ull;
    const char *wd_plane = (const char *)W_aligned;
    const char *wq_plane = (const char *)W_aligned + dq_bytes;
    // Grouped-order sweep verdicts from the proto: shallow K prefers
    // ungrouped (down/q_b group_m=1), K=4096 prefers 4.
    const int group_m = (K <= 2048) ? 1 : 4;
    const int num_m = M / kDqMTile;
    const int num_n = (N + kDqNTile - 1) / kDqNTile;
    const dim3 grid((unsigned)(num_m * num_n), 1, 1);
    const dim3 block(32, kDqWarps, 1);
    dense_q8_d2r_kernel<<<grid, block, kDqSmemTotalBytes, stream>>>(
        wd_plane, wq_plane, (const block_q8_1_mmq *)q8, out, M, N, K, group_m);
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "ds4_mmq_q8_0_dense_d2r_launch: launch failed: %s\n",
                cudaGetErrorString(err));
        return -3;
    }
    return 0;
}
