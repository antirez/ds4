// SPDX-License-Identifier: MIT
// proto_gemm_dense_q8_d2r.cu - dense Q8_0 decode-to-registers prototype.
//
// Targets the dense-q8 census shapes (cmdq8shape15 2026-07-09, N = chunk
// tokens 4096) where production mul_mat_q<8,128,0> runs far under the
// cuBLASLt-int8 ceiling at the same shapes:
//   sexp gate/up  M= 2048 K=4096   mmq 65 TF  vs Lt-int8 156
//   sexp down     M= 4096 K=2048   mmq 24 TF  vs Lt-int8 111
//   attn q_b      M=32768 K=1024   mmq 56 TF  vs Lt-int8  79-85
//   o_proj        M= 4096 K=8192   mmq 96 TF  vs Lt-int8 145
//
// Weight layout: byte-neutral SoA repack of raw Q8_0 (34 B/block of 32):
//   [half d[M*nb]] [int8 qs[M*K]]   (d row-major [M][nb], qs row-major [M][K])
// Total bytes == raw bytes.  NOTE a "nicer" fully-tiled contiguous-stage
// layout ([row_tile][k64][h][row][16B]) measured 27% SLOWER at identical
// instruction counts: every col-tile CTA of a row group reads the same 8 KiB
// block concurrently and the contiguous span lives on a few L2 slices (slice
// camping).  Row-major keeps each stage's 128 rows K bytes apart -> spread
// across slices; its 2-way ldgsts store conflicts are the cheaper poison.
// This is the REPLACING-artifact candidate (IQ2/Q2K precedent); raw consumers
// fall back to the host mmap.
//
// Kernel: m128n64 CTA (8 warps), q8 activations = production block_q8_1_mmq
// D4 layout (f32 scale per 32) double-buffered at k128; weights double-
// buffered at k64 (128 rows x 64 B + 16 B row pad for ldmatrix bank rotation);
// mma.m16n8k32.s8 with per-k32 float fold acc += (s32)C * d_w(row) * d_a(col).
// Q8_0 has no mins: no bias terms, no fix staging (d4 read straight from the
// q8 block header).  Grid is 1-D with triton-style grouped supertiling
// (group_m row-tiles hot in L2 while col tiles stream) - group_m is a runtime
// arg, swept by the harness.
//
// Build on the GB10/Blackwell box after `make -j6 cuda-spark`:
//   ar rcs libds4mmq.a cuda/mmq/ds4_ggml_stubs.o cuda/mmq/ds4_mmq.o \
//          cuda/mmq/quantize.o cuda/mmq/mmid.o cuda/mmq/mmvq.o
//   /usr/local/cuda/bin/nvcc -O3 --use_fast_math -std=c++17 -arch=sm_121 \
//        -Icuda/mmq cuda/mmq/test/proto_gemm_dense_q8_d2r.cu libds4mmq.a \
//        -lcudart -lcublas -lcuda -o proto_gemm_dense_q8_d2r
// argv: [reps=20] [warmup=5] [N=4096]

#include <type_traits>

#include "ds4_mmq.h"
#include "quantize.cuh"

extern "C" int ds4_cuda_q8_fold_take_q81(const void *src, uint64_t in_dim, void *out) {
    (void)src;
    (void)in_dim;
    (void)out;
    return 0;
}

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

#define CHECK(call) do { \
    cudaError_t err__ = (call); \
    if (err__ != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__, __LINE__, \
                cudaGetErrorString(err__)); \
        exit(1); \
    } \
} while (0)

namespace {

constexpr int kMTile   = 128;
#ifndef PROTO_Q8_NTILE
#define PROTO_Q8_NTILE 128
#endif
constexpr int kNTile   = PROTO_Q8_NTILE;
// 16-warp CTA, one per SM: 8 row-warps x 2 col-warps.  Same 16 warps/SM
// residency as 2x8, but per-thread acc halves to 8 fragments (32 regs) --
// the 2x8-warp NT128 shape spilled ~76 B and the spill reloads were the top
// stall (long_sb 3.02, local_ld 2.65M).
constexpr int kRowWarps = 8;
constexpr int kColWarps = 2;
constexpr int kWarps   = kRowWarps * kColWarps;
constexpr int kThreads = 32 * kWarps;
// 3-stage rings: issues happen at the TOP of each iter for stage i+2 (a
// buffer no in-flight reader touches), so the post-compute __syncthreads of
// the 2-stage schedule disappears -- one barrier per k64.  69 KiB total needs
// the dynamic-smem opt-in (1 CTA/SM is already forced by the 512-thread
// register budget, so no occupancy cost).
constexpr int kStages  = 3;   // q8 qs ring (k64 halves) + header ring (k128)
constexpr int kWStages = 3;   // weight ring, k64 granularity
constexpr int kNFrag   = kNTile / 8;
constexpr int kWRowPad = 80;  // 64 B payload + 16 B pad: 20-int ldmatrix stride rotates banks

// q8 staging is SPLIT so NT=128 fits under 48 KiB static smem: the 16 B
// header (d4 scales) double-buffered at k128 cadence, the 128 B qs payload
// staged as k64 halves (64 B rows, same padded geometry as the weight ring).
constexpr int kQ8QsChunks  = kNTile * 4;  // cols x 4 16B-chunks per k64 half
constexpr int kQ8QsTrips   = (kQ8QsChunks + kThreads - 1) / kThreads;
constexpr int kWQChunks    = kMTile * 4;  // 128 rows x 4 16B-chunks (64 B payload)
constexpr int kWQTrips     = kWQChunks / kThreads;

static_assert(kQ8QsTrips == 1, "unexpected q8 qs issue trip count");
static_assert(kWQTrips == 1, "unexpected weight issue trip count");
static_assert(kNTile == 128, "16-warp decomposition assumes NT=128");
static_assert(kNTile <= kThreads, "q8 header prefetch assumes one trip");
constexpr int kNFragPerWarp = kNFrag / kColWarps;

// Loop invariants live in plain (warp-uniform) registers: everything here
// derives arithmetically from blockIdx + kernel args, so ptxas can promote
// the math to the uniform register file.  (The Q2K proto's shared invariant
// table exists for its memory-sourced expert-worklist lookups - dense has
// none, and routing these through volatile smem forced them into regular
// registers, feeding the 128-reg wall.)
struct DenseQ8Params {
    const char *wd_row0;       // d plane at cta_row0 (row stride nb*2 = K/16)
    const char *wq_row0;       // qs plane at cta_row0 (row stride K)
    const char *q8_tile;       // q8 blocks at col_lo
    uint32_t q8_k128_stride_bytes;
    uint32_t wq_row_stride;    // K bytes
    int k64_iters;
    int k128_iters;
};

uint16_t sane_half(std::mt19937 &rng) {
    const uint16_t mant = (uint16_t)(rng() & 0x03FFu);
    const uint16_t sign = (uint16_t)((rng() & 1u) << 15);
    return (uint16_t)(sign | (14u << 10) | mant);
}

float *device_random_floats(std::mt19937 &rng, size_t n) {
    std::vector<float> h(n);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < n; ++i) {
        h[i] = dist(rng);
    }
    float *d = nullptr;
    CHECK(cudaMalloc(&d, n * sizeof(float)));
    CHECK(cudaMemcpy(d, h.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    return d;
}

template <typename F>
float best_time_ms(F &&fn, int warmup, int reps) {
    cudaEvent_t a, b;
    CHECK(cudaEventCreate(&a));
    CHECK(cudaEventCreate(&b));
    for (int i = 0; i < warmup; ++i) {
        fn();
    }
    CHECK(cudaDeviceSynchronize());
    float best = std::numeric_limits<float>::infinity();
    for (int r = 0; r < reps; ++r) {
        CHECK(cudaEventRecord(a));
        fn();
        CHECK(cudaEventRecord(b));
        CHECK(cudaEventSynchronize(b));
        float ms = 0.0f;
        CHECK(cudaEventElapsedTime(&ms, a, b));
        best = std::min(best, ms);
    }
    CHECK(cudaEventDestroy(a));
    CHECK(cudaEventDestroy(b));
    return best;
}

__device__ __forceinline__ void zero_16B(void *dst) {
    *reinterpret_cast<int4 *>(dst) = make_int4(0, 0, 0, 0);
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

__device__ __forceinline__ void cp_async_4B(void *dst, const void *src, bool pred) {
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

// CTA is launched as dim3(32, kWarps): the ggml mma primitives
// (load_ldmatrix, tile get_i/get_j) use threadIdx.x as the LANE.
__device__ __forceinline__ int d2r_lane() {
#if defined(__CUDA_ARCH__)
    return (int)threadIdx.x;
#else
    return 0;
#endif
}

__device__ __forceinline__ int d2r_warp() {
#if defined(__CUDA_ARCH__)
    return (int)threadIdx.y;
#else
    return 0;
#endif
}

__device__ __forceinline__ int d2r_tid() {
#if defined(__CUDA_ARCH__)
    return (int)(threadIdx.y * 32 + threadIdx.x);
#else
    return 0;
#endif
}

constexpr size_t kSmemQ8HBytes = (size_t)kStages * kNTile * 16;        // headers
constexpr size_t kSmemQ8QBytes = (size_t)kStages * kNTile * kWRowPad;  // qs k64 halves
constexpr size_t kSmemWQBytes  = (size_t)kWStages * kMTile * kWRowPad;
constexpr size_t kSmemWDBytes  = (size_t)kWStages * kMTile * sizeof(uint32_t);
constexpr size_t kSmemWQOff  = 0;
constexpr size_t kSmemQ8QOff = kSmemWQOff + kSmemWQBytes;
constexpr size_t kSmemQ8HOff = kSmemQ8QOff + kSmemQ8QBytes;
constexpr size_t kSmemWDOff  = kSmemQ8HOff + kSmemQ8HBytes;
constexpr size_t kSmemTotalBytes = kSmemWDOff + kSmemWDBytes;
static_assert(kSmemTotalBytes <= 99ull * 1024ull, "dynamic shared memory exceeds sm_121 limit");

// ---- q8 activation prefetch, split staging ----
// qs k64 halves: 128 cols x 64 B (bytes 16 + (k64&1)*64 .. of each col's
// 144 B block), padded 80 B rows — same geometry as the weight ring.
// headers: 128 cols x 16 B (the d4 scale header), k128 cadence.

__device__ __forceinline__ void issue_q8_qs_prefetch(
        int8_t (&s_q8q)[kStages][kNTile][kWRowPad],
        const DenseQ8Params &p,
        int k64_iter) {
    const bool pred = k64_iter < p.k64_iters;
    const char *base = p.q8_tile +
                       (uint64_t)(k64_iter >> 1) * p.q8_k128_stride_bytes +
                       16u + (uint64_t)(k64_iter & 1) * 64u;
    const int buf = k64_iter % kStages;
#pragma unroll
    for (int trip = 0; trip < kQ8QsTrips; ++trip) {
        const int chunk = d2r_tid() + trip * kThreads;
        if (kQ8QsChunks % kThreads != 0 && chunk >= kQ8QsChunks) {
            break;
        }
        const int col = chunk >> 2;
        const int h = chunk & 3;
        void *dst = &s_q8q[buf][col][h * 16];
        const void *src = base + (uint64_t)col * sizeof(block_q8_1_mmq) + (uint64_t)h * 16u;
        cp_async_16B(dst, src, pred);
    }
}

__device__ __forceinline__ void issue_q8_hdr_prefetch(
        float (&s_q8h)[kStages][kNTile][4],
        const DenseQ8Params &p,
        int k128_iter) {
    if (d2r_tid() >= kNTile) {
        return;
    }
    const bool pred = k128_iter < p.k128_iters;
    const char *base = p.q8_tile +
                       (uint64_t)k128_iter * p.q8_k128_stride_bytes;
    const int col = d2r_tid();
    void *dst = &s_q8h[k128_iter % kStages][col][0];
    const void *src = base + (uint64_t)col * sizeof(block_q8_1_mmq);
    cp_async_16B(dst, src, pred);
}

// ---- weight prefetch (one k64 stage = 128 rows x 64 B qs + 128 x 4 B d) ----

__device__ __forceinline__ void issue_w_prefetch(
        int8_t (&s_wq)[kWStages][kMTile][kWRowPad],
        uint32_t (&s_wd)[kWStages][kMTile],
        const DenseQ8Params &p,
        int k64_iter) {
    const bool pred = k64_iter < p.k64_iters;
    const int wst = k64_iter % kWStages;
    // Row-major planes (see layout note at top: contiguous-stage tiling was
    // 27% slower from L2 slice camping).
    const char *wq = p.wq_row0 + (uint64_t)k64_iter * 64u;
    const char *wd = p.wd_row0 + (uint64_t)k64_iter * 4u;
    const uint32_t wq_row_stride = p.wq_row_stride;
    const uint32_t wd_row_stride = p.wq_row_stride >> 4;  // nb*2 = K/16
#pragma unroll
    for (int trip = 0; trip < kWQTrips; ++trip) {
        const int chunk = d2r_tid() + trip * kThreads;
        const int row = chunk >> 2;
        const int h = chunk & 3;
        void *dst = &s_wq[wst][row][h * 16];
        const void *src = wq + (uint64_t)row * wq_row_stride + (uint64_t)h * 16u;
        cp_async_16B(dst, src, pred);
    }
    if (d2r_tid() < kMTile) {
        const int row = d2r_tid();
        void *dst = &s_wd[wst][row];
        const void *src = wd + (uint64_t)row * wd_row_stride;
        cp_async_4B(dst, src, pred);
    }
}

// ---- mainloop ----

template <typename TileA, typename TileB, typename TileC>
__device__ __forceinline__ void dense_q8_d2r_mainloop(
        float (&acc)[kNFragPerWarp][4],
        float (&s_q8h)[kStages][kNTile][4],
        int8_t (&s_q8q)[kStages][kNTile][kWRowPad],
        int8_t (&s_wq)[kWStages][kMTile][kWRowPad],
        uint32_t (&s_wd)[kWStages][kMTile],
        const DenseQ8Params &p) {
    static_assert(TileC::ne == 4, "expected m16n8 s32 accumulator fragment");
    const int k64_iters = p.k64_iters;
    const int lane = d2r_lane();
    const int group = lane >> 2;
    const int wrow = (d2r_warp() >> 1) * 16;          // row-warp
    const int nf0 = (d2r_warp() & 1) * kNFragPerWarp;  // col-warp half
    const int c0 = TileC::get_j(0);
    const int c1 = TileC::get_j(1);

    // Prologue: stages 0 and 1 in flight as two groups.  Loop iters run
    // wait -> barrier -> issue(i+2) -> compute: with 3 buffers the issue
    // never aliases the two live stages, so the 2-stage schedule's
    // post-compute barrier disappears (one barrier per k64).
    issue_w_prefetch(s_wq, s_wd, p, 0);
    issue_q8_qs_prefetch(s_q8q, p, 0);
    issue_q8_hdr_prefetch(s_q8h, p, 0);
    cp_async_commit();
    issue_w_prefetch(s_wq, s_wd, p, 1);
    issue_q8_qs_prefetch(s_q8q, p, 1);
    issue_q8_hdr_prefetch(s_q8h, p, 1);
    cp_async_commit();

    for (int i = 0; i < k64_iters; ++i) {
        // wait -> barrier -> issue: the barrier both publishes stage i's
        // copies CTA-wide and proves every warp finished reading stage i-1,
        // whose buffer ((i-1)%3 == (i+2)%3) the issues below overwrite.
        cp_async_wait_group<1>();
        __syncthreads();
        issue_w_prefetch(s_wq, s_wd, p, i + 2);
        issue_q8_qs_prefetch(s_q8q, p, i + 2);
        if ((i & 1) == 0) {
            issue_q8_hdr_prefetch(s_q8h, p, (i >> 1) + 2);
        }
        cp_async_commit();

        const int wst = i % kWStages;
        const int qbuf = i % kStages;          // qs halves ride the k64 cadence
        const int hbuf = (i >> 1) % kStages;   // headers ride the k128 cadence
        const uint32_t dw0_bits = s_wd[wst][wrow + group];
        const uint32_t dw1_bits = s_wd[wst][wrow + group + 8];
        const float2 dw0 = __half22float2(*reinterpret_cast<const half2 *>(&dw0_bits));
        const float2 dw1 = __half22float2(*reinterpret_cast<const half2 *>(&dw1_bits));

        // t-phased: each A load overlaps the previous phase's B/MMA/fold tail
        // (fusing both A loads up front measured -27%: hard A dependency at the
        // k64 head + burstier MIO).
#pragma unroll
        for (int t = 0; t < 2; ++t) {
            TileA A;
            ggml_cuda_mma::load_ldmatrix(
                A, reinterpret_cast<const int *>(&s_wq[wst][wrow][t * 32]),
                kWRowPad / (int)sizeof(int));
            const int tq8 = (i & 1) * 2 + t;  // k32 index within the q8 k128 block
            const float dwr0 = t ? dw0.y : dw0.x;
            const float dwr1 = t ? dw1.y : dw1.x;
#pragma unroll
            for (int nf = 0; nf < kNFragPerWarp; ++nf) {
                TileB B;
                TileC C;
                ggml_cuda_mma::load_ldmatrix(
                    B, reinterpret_cast<const int *>(&s_q8q[qbuf][(nf0 + nf) * 8][t * 32]),
                    kWRowPad / (int)sizeof(int));
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

__global__ __launch_bounds__(kThreads, 1)
void dense_q8_d2r_kernel(const char * __restrict__ wd_plane,
                         const char * __restrict__ wq_plane,
                         const block_q8_1_mmq * __restrict__ q8,
                         float * __restrict__ outf,
                         __half * __restrict__ outh,
                         int M, int N, int K, int group_m) {
#if defined(TURING_MMA_AVAILABLE)
    using tile_A = ggml_cuda_mma::tile<16, 8, int>;
    using tile_B = ggml_cuda_mma::tile<8, 8, int>;
    using tile_C = ggml_cuda_mma::tile<16, 8, int>;

    extern __shared__ __align__(16) char smem_dyn[];
    auto &s_wq  = *reinterpret_cast<int8_t (*)[kWStages][kMTile][kWRowPad]>(smem_dyn + kSmemWQOff);
    auto &s_q8q = *reinterpret_cast<int8_t (*)[kStages][kNTile][kWRowPad]>(smem_dyn + kSmemQ8QOff);
    auto &s_q8h = *reinterpret_cast<float (*)[kStages][kNTile][4]>(smem_dyn + kSmemQ8HOff);
    auto &s_wd  = *reinterpret_cast<uint32_t (*)[kWStages][kMTile]>(smem_dyn + kSmemWDOff);

    // Triton-style grouped supertile order: group_m row-tiles stay L2-hot
    // while the col tiles stream past them.
    const int num_m = M / kMTile;
    const int num_n = (N + kNTile - 1) / kNTile;
    const int width = group_m * num_n;
    const int g = (int)blockIdx.x / width;
    const int rem = (int)blockIdx.x - g * width;
    int gsize = num_m - g * group_m;
    if (gsize > group_m) {
        gsize = group_m;
    }
    const int pid_m = g * group_m + rem % gsize;
    const int pid_n = rem / gsize;

    const int cta_row0 = pid_m * kMTile;
    const int col_lo = pid_n * kNTile;

    DenseQ8Params p;
    p.wd_row0 = wd_plane + (uint64_t)cta_row0 * (uint64_t)(K >> 4);
    p.wq_row0 = wq_plane + (uint64_t)cta_row0 * (uint64_t)K;
    p.q8_tile = reinterpret_cast<const char *>(q8) +
                (uint64_t)col_lo * sizeof(block_q8_1_mmq);
    p.q8_k128_stride_bytes = (uint32_t)((uint64_t)N * sizeof(block_q8_1_mmq));
    p.wq_row_stride = (uint32_t)K;
    p.k64_iters = K >> 6;
    p.k128_iters = K >> 7;

    float acc[kNFragPerWarp][tile_C::ne] = {};

    dense_q8_d2r_mainloop<tile_A, tile_B, tile_C>(acc, s_q8h, s_q8q, s_wq, s_wd, p);

    const int out_col_lo = col_lo + (d2r_warp() & 1) * (kNFragPerWarp * 8);
    const int out_N = N;
    const int out_M = M;
    const int out_row0 = cta_row0 + ((d2r_warp() >> 1) << 4);
    float *outf_base = outf;
    __half *outh_base = outh;
#pragma unroll
    for (int nf = 0; nf < kNFragPerWarp; ++nf) {
        const int col_frag0 = out_col_lo + nf * 8;
#pragma unroll
        for (int l = 0; l < tile_C::ne; ++l) {
            const int row = out_row0 + tile_C::get_i(l);
            const int col = col_frag0 + tile_C::get_j(l);
            if (col < out_N) {
                const float v = isfinite(acc[nf][l]) ? acc[nf][l] : 0.0f;
                if (outh_base) {
                    outh_base[(uint64_t)col * (uint64_t)out_M + (uint64_t)row] = __float2half(v);
                } else {
                    outf_base[(uint64_t)col * (uint64_t)out_M + (uint64_t)row] = v;
                }
            }
        }
    }
#else
    (void)wd_plane;
    (void)wq_plane;
    (void)q8;
    (void)outf;
    (void)outh;
    (void)M;
    (void)N;
    (void)K;
    (void)group_m;
#endif
}

int launch_dense_q8_d2r(const void *wd_plane, const void *wq_plane,
                        const block_q8_1_mmq *q8, float *outf, __half *outh,
                        int M, int N, int K, int group_m, cudaStream_t stream) {
    if (!wd_plane || !wq_plane || !q8 || (!outf && !outh) ||
        M <= 0 || (M & (kMTile - 1)) != 0 || K % 256 != 0 || N <= 0 || group_m <= 0) {
        fprintf(stderr, "launch_dense_q8_d2r: bad args M=%d N=%d K=%d group_m=%d\n",
                M, N, K, group_m);
        return -1;
    }
    const int num_m = M / kMTile;
    const int num_n = (N + kNTile - 1) / kNTile;
    static int smem_opted = 0;
    if (!smem_opted) {
        cudaFuncSetAttribute(dense_q8_d2r_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int)kSmemTotalBytes);
        smem_opted = 1;
    }
    const dim3 grid((unsigned)(num_m * num_n), 1, 1);
    const dim3 block(32, kWarps, 1);
    dense_q8_d2r_kernel<<<grid, block, kSmemTotalBytes, stream>>>(
        (const char *)wd_plane, (const char *)wq_plane, q8, outf, outh, M, N, K, group_m);
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "dense_q8_d2r_kernel launch failed: %s\n", cudaGetErrorString(err));
        return -2;
    }
    return 0;
}

void quantize_q8_mmq_dense(const float *x, block_q8_1_mmq *q8, int K, int N,
                           cudaStream_t stream) {
    quantize_mmq_q8_1_cuda(
        x, /*ids=*/nullptr, (void *)q8,
        GGML_TYPE_Q8_0, /*ne00=*/K, /*s01=*/(int64_t)K, /*s02=*/0, /*s03=*/0,
        /*ne0=*/K, /*ne1=*/N, /*ne2=*/1, /*ne3=*/1, stream);
    CHECK(cudaGetLastError());
}

struct ParityStats {
    double max_abs = 0.0;
    double max_rel_rms = 0.0;
    int max_row = -1;
    int max_col = -1;
};

ParityStats check_parity(const float *ref, const float *got, int M, int N) {
    std::vector<double> ss((size_t)M, 0.0);
    for (int c = 0; c < N; ++c) {
        const float *rp = ref + (size_t)c * M;
        for (int r = 0; r < M; ++r) {
            ss[(size_t)r] += (double)rp[r] * (double)rp[r];
        }
    }
    ParityStats st;
    for (int c = 0; c < N; ++c) {
        const float *rp = ref + (size_t)c * M;
        const float *gp = got + (size_t)c * M;
        for (int r = 0; r < M; ++r) {
            const double abs_err = std::abs((double)gp[r] - (double)rp[r]);
            const double rms = std::sqrt(ss[(size_t)r] / std::max(1, N));
            const double rel = abs_err / std::max(1.0e-6, rms);
            if (abs_err > st.max_abs) {
                st.max_abs = abs_err;
                st.max_row = r;
                st.max_col = c;
            }
            st.max_rel_rms = std::max(st.max_rel_rms, rel);
        }
    }
    return st;
}

void run_shape(std::mt19937 &rng, const char *label, int M, int K, int N,
               int warmup, int reps, bool do_parity) {
    const int nb = K / 32;
    const uint64_t raw_bytes = (uint64_t)M * nb * 34u;
    const uint64_t d_bytes = (uint64_t)M * nb * 2u;

    // Raw Q8_0 blocks (34 B: half d + 32 int8) for the production reference,
    // and the byte-neutral SoA planes for the D2R kernel, from the same data.
    std::vector<uint8_t> h_raw(raw_bytes);
    std::vector<uint8_t> h_soa(raw_bytes);
    {
        // Row-major SoA planes: d [M][nb] halves, qs [M][K] int8.
        uint8_t *raw = h_raw.data();
        uint16_t *dp = reinterpret_cast<uint16_t *>(h_soa.data());
        int8_t *qp = reinterpret_cast<int8_t *>(h_soa.data() + d_bytes);
        for (int r = 0; r < M; ++r) {
            for (int b = 0; b < nb; ++b) {
                uint8_t *blk = raw + ((uint64_t)r * nb + b) * 34u;
                const uint16_t d = sane_half(rng);
                memcpy(blk, &d, 2);
                for (int j = 0; j < 32; ++j) {
                    blk[2 + j] = (uint8_t)(rng() & 0xFFu);
                }
                dp[(uint64_t)r * nb + b] = d;
                memcpy(qp + (uint64_t)r * K + (uint64_t)b * 32u, blk + 2, 32);
            }
        }
    }

    void *d_raw = nullptr;
    void *d_soa = nullptr;
    CHECK(cudaMalloc(&d_raw, raw_bytes));
    CHECK(cudaMalloc(&d_soa, raw_bytes));
    CHECK(cudaMemcpy(d_raw, h_raw.data(), raw_bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_soa, h_soa.data(), raw_bytes, cudaMemcpyHostToDevice));
    const char *d_wd = (const char *)d_soa;
    const char *d_wq = (const char *)d_soa + d_bytes;

    float *d_x = device_random_floats(rng, (size_t)N * (size_t)K);

    // +kNTile block slack so a guarded last col tile can read past col N-1
    // without faulting (values discarded by the epilogue guard).
    const size_t q8_blocks = (size_t)N * (size_t)(K / 128) + kNTile;
    block_q8_1_mmq *d_q8 = nullptr;
    CHECK(cudaMalloc(&d_q8, q8_blocks * sizeof(block_q8_1_mmq)));
    CHECK(cudaMemset(d_q8, 0, q8_blocks * sizeof(block_q8_1_mmq)));
    quantize_q8_mmq_dense(d_x, d_q8, K, N, 0);

    float *d_ref = nullptr;
    float *d_new = nullptr;
    __half *d_newh = nullptr;
    CHECK(cudaMalloc(&d_ref, (size_t)M * N * sizeof(float)));
    CHECK(cudaMalloc(&d_new, (size_t)M * N * sizeof(float)));
    CHECK(cudaMalloc(&d_newh, (size_t)M * N * sizeof(__half)));

    if (ds4_mmq_q8_0_dense(d_raw, d_x, d_ref, M, N, K, 0) != 0) {
        fprintf(stderr, "reference ds4_mmq_q8_0_dense failed M=%d K=%d\n", M, K);
        exit(1);
    }
    if (launch_dense_q8_d2r(d_wd, d_wq, d_q8, d_new, nullptr, M, N, K, 8, 0) != 0) {
        exit(1);
    }
    CHECK(cudaDeviceSynchronize());

    if (do_parity) {
        std::vector<float> ref((size_t)M * N), got((size_t)M * N);
        CHECK(cudaMemcpy(ref.data(), d_ref, ref.size() * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(got.data(), d_new, got.size() * sizeof(float), cudaMemcpyDeviceToHost));
        const ParityStats st = check_parity(ref.data(), got.data(), M, N);
        const bool pass = st.max_rel_rms <= 2.0e-3;
        printf("  parity %-8s M=%5d K=%4d N=%d: max_abs=%.6g at col=%d row=%d  max_abs/rms(row)=%.6g  %s\n",
               label, M, K, N, st.max_abs, st.max_col, st.max_row, st.max_rel_rms,
               pass ? "PASS" : "FAIL");
        if (!pass && getenv("PROTO_Q8_DEBUG")) {
            // Error structure: per (row-tile, col-tile) fraction of bad elements,
            // plus per row%16 / col%8 / warp-row histograms of bad positions.
            const int ntm = M / 128, ntn = (N + 63) / 64;
            size_t bad_total = 0;
            std::vector<size_t> bad_rowmod(16, 0), bad_colmod(8, 0), bad_warp(8, 0);
            std::vector<size_t> tile_bad((size_t)ntm * ntn, 0);
            for (int c = 0; c < N; ++c) {
                const float *rp = ref.data() + (size_t)c * M;
                const float *gp = got.data() + (size_t)c * M;
                for (int r = 0; r < M; ++r) {
                    const double e = std::abs((double)gp[r] - (double)rp[r]);
                    const double mag = std::max(1.0, std::abs((double)rp[r]));
                    if (e / mag > 1.0e-2) {
                        ++bad_total;
                        ++bad_rowmod[r & 15];
                        ++bad_colmod[c & 7];
                        ++bad_warp[(r >> 4) & 7];
                        ++tile_bad[(size_t)(r / 128) * ntn + (c / 64)];
                    }
                }
            }
            printf("    bad(rel>1e-2) %zu / %zu\n", bad_total, (size_t)M * N);
            printf("    by row%%16:"); for (int i = 0; i < 16; ++i) printf(" %zu", bad_rowmod[i]); printf("\n");
            printf("    by col%%8: "); for (int i = 0; i < 8; ++i) printf(" %zu", bad_colmod[i]); printf("\n");
            printf("    by warp(row/16%%8):"); for (int i = 0; i < 8; ++i) printf(" %zu", bad_warp[i]); printf("\n");
            printf("    tile map (rows=row-tile, cols=col-tile, %% bad, first 8x16):\n");
            for (int tm = 0; tm < std::min(ntm, 8); ++tm) {
                printf("      ");
                for (int tn = 0; tn < std::min(ntn, 16); ++tn) {
                    printf("%3zu", tile_bad[(size_t)tm * ntn + tn] * 100 / (128 * 64));
                }
                printf("\n");
            }
            // A few sample mismatches with ratios (scale bugs show as clean ratios).
            int shown = 0;
            for (int c = 0; c < N && shown < 6; ++c) {
                for (int r = 0; r < M && shown < 6; ++r) {
                    const float rv = ref[(size_t)c * M + r], gv = got[(size_t)c * M + r];
                    if (std::abs(gv - rv) / std::max(1.0f, std::abs(rv)) > 1.0e-2) {
                        printf("    sample r=%d c=%d ref=%.6g got=%.6g got/ref=%.4f\n",
                               r, c, rv, gv, rv != 0.0f ? gv / rv : 0.0f);
                        ++shown;
                    }
                }
            }
        }
        if (!pass) {
            exit(2);
        }
    }

    const double flop = 2.0 * (double)M * (double)K * (double)N;
    const float ref_ms = best_time_ms([&] {
        ds4_mmq_q8_0_dense(d_raw, d_x, d_ref, M, N, K, 0);
    }, warmup, reps);

    printf("  %-8s M=%5d K=%4d N=%d  mmq-ref %7.3f ms %5.1f TF\n",
           label, M, K, N, ref_ms, flop / (ref_ms * 1.0e-3) / 1.0e12);

    float best_kernel_ms = std::numeric_limits<float>::infinity();
    int best_g = 0;
    const int groups[] = {1, 4, 8, 16};
    for (int gi = 0; gi < 4; ++gi) {
        const int g = groups[gi];
        const float ms = best_time_ms([&] {
            launch_dense_q8_d2r(d_wd, d_wq, d_q8, d_new, nullptr, M, N, K, g, 0);
        }, warmup, reps);
        printf("    d2r f32-out group_m=%2d  %7.3f ms %5.1f TF\n",
               g, ms, flop / (ms * 1.0e-3) / 1.0e12);
        if (ms < best_kernel_ms) {
            best_kernel_ms = ms;
            best_g = g;
        }
    }

    const float f16_ms = best_time_ms([&] {
        launch_dense_q8_d2r(d_wd, d_wq, d_q8, nullptr, d_newh, M, N, K, best_g, 0);
    }, warmup, reps);
    const float full_ms = best_time_ms([&] {
        quantize_q8_mmq_dense(d_x, d_q8, K, N, 0);
        launch_dense_q8_d2r(d_wd, d_wq, d_q8, d_new, nullptr, M, N, K, best_g, 0);
    }, warmup, reps);

    printf("    d2r BEST group_m=%d: f32-out %7.3f ms %5.1f TF | f16-out %7.3f ms %5.1f TF | +q8 %7.3f ms %5.1f TF | vs mmq %.2fx\n",
           best_g,
           best_kernel_ms, flop / (best_kernel_ms * 1.0e-3) / 1.0e12,
           f16_ms, flop / (f16_ms * 1.0e-3) / 1.0e12,
           full_ms, flop / (full_ms * 1.0e-3) / 1.0e12,
           ref_ms / best_kernel_ms);

    CHECK(cudaFree(d_raw));
    CHECK(cudaFree(d_soa));
    CHECK(cudaFree(d_x));
    CHECK(cudaFree(d_q8));
    CHECK(cudaFree(d_ref));
    CHECK(cudaFree(d_new));
    CHECK(cudaFree(d_newh));
}

} // namespace

int main(int argc, char **argv) {
    const int reps = (argc > 1) ? atoi(argv[1]) : 20;
    const int warmup = (argc > 2) ? atoi(argv[2]) : 5;
    const int N = (argc > 3) ? atoi(argv[3]) : 4096;

    printf("### proto_gemm_dense_q8_d2r reps=%d warmup=%d N=%d ###\n", reps, warmup, N);
    printf("### CTA m%dn%d, warps=%d, q8 x%d stages, weights x%d stages, dyn smem %zu B ###\n",
           kMTile, kNTile, kWarps, kStages, kWStages, kSmemTotalBytes);
    printf("### Lt-int8 ceilings at these shapes: gate/up 156, down 111, q_b 79-85, o_proj 145 TF ###\n");

    std::mt19937 rng(0xD84D8021u + 13u * (unsigned)N);

    run_shape(rng, "gate/up", 2048, 4096, N, warmup, reps, true);
    run_shape(rng, "down", 4096, 2048, N, warmup, reps, true);
    run_shape(rng, "q_b", 32768, 1024, N, warmup, reps, false);
    run_shape(rng, "o_proj", 4096, 8192, N, warmup, reps, false);
    return 0;
}
