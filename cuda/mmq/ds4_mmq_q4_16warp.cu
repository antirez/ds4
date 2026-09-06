// SPDX-License-Identifier: MIT
// Dense Q4_K x canonical-MMQ-Q8_1, m128n128, 16-warp experiment.
//
// The canonical Turing/Ampere MMQ kernel assigns two 16-row MMA minitiles to
// each of eight warps at N=128.  That leaves 64 F32 accumulators per thread
// and can spill on shallow-K, wide-M prefill projections.  This kernel keeps
// the canonical shared representation and arithmetic but splits each 128-row
// tile over four N-warps for each 32-row band.  Each warp therefore owns
// 32 rows x 32 columns and carries 32 accumulators.  Keeping both 16-row A
// fragments in one warp also preserves the canonical reuse of each B load.
//
// Numerical contract:
//   * canonical Q4_K nibble/scales/min unpack;
//   * canonical Q8_1 DS4 (half scale + half sum) activation blocks;
//   * identical ascending sequence of eight K32 folds per Q4_K block;
//   * the two canonical F32 accumulation statements are kept verbatim;
//   * direct tiling and canonical Stream-K/fixup reduction trees are both
//     available; the caller chooses explicitly at the enqueue boundary.

#include "ds4_mmq_q4_16warp.cuh"

#include "common.cuh"
#include "mmq.cuh"

#include <cstddef>
#include <cstdint>

namespace {
namespace q4w16 {

constexpr int kMTile       = 128;
constexpr int kNTile       = 128;
constexpr int kRowGroups   = 4;
constexpr int kColWarps    = 4;
constexpr int kWarps       = kRowGroups * kColWarps;
constexpr int kThreads     = 32 * kWarps;
constexpr int kRowFrag     = 2;
constexpr int kNFrag       = kNTile / 8;
constexpr int kNFragPerWarp = kNFrag / kColWarps;
constexpr int kMetadataWarps = kMTile / 16;
constexpr int kWeightStride = MMQ_MMA_TILE_X_K_Q8_1;
constexpr int kYStrideInts = sizeof(block_q8_1_mmq) / sizeof(int);
constexpr int kYChunks16   = sizeof(block_q8_1_mmq) / 16;
constexpr size_t kTileElements = (size_t)kMTile * (size_t)kNTile;

constexpr size_t kWeightTileBytes =
    (size_t)kMTile * (size_t)kWeightStride * sizeof(int);
constexpr size_t kYTileBytes =
    (size_t)kNTile * sizeof(block_q8_1_mmq);
constexpr size_t kSharedBytes = kWeightTileBytes + kYTileBytes;
constexpr int kYTileVectors = (int)(kYTileBytes / sizeof(int4));

static_assert(kWarps == 16, "Q4 16-warp decomposition changed");
static_assert(kThreads == 512, "Q4 16-warp CTA must have 512 threads");
static_assert(kRowGroups * kRowFrag * 16 == kMTile,
              "Q4 split-N row coverage changed");
static_assert(kNFragPerWarp == 4, "Q4 split-N fragment count changed");
static_assert(kWeightStride == 76, "canonical Q4_K MMA row stride changed");
static_assert(kYStrideInts == 36, "canonical Q8_1 DS4 stride changed");
static_assert(kYChunks16 == 9, "canonical Q8_1 DS4 block size changed");
static_assert(kYTileBytes % sizeof(int4) == 0,
              "Q8_1 tile must support vectorized copies");
static_assert(MMQ_ITER_K % QK_K == 0 && MMQ_ITER_K / QK_K == 1,
              "Stream-K scheduler must restore canonical K alignment");
static_assert(kSharedBytes == 57344, "Q4 16-warp shared-memory model changed");
static_assert(kSharedBytes <= 99ull * 1024ull,
              "Q4 16-warp kernel exceeds the intended opt-in shared limit");

__device__ __forceinline__ int lane_id() {
    return (int)threadIdx.x;
}

__device__ __forceinline__ int warp_id() {
    return (int)threadIdx.y;
}

__device__ __forceinline__ int linear_tid() {
    return (warp_id() << 5) | lane_id();
}

template <bool need_check>
__device__ __forceinline__ void load_weight_tile(
        const block_q4_K * __restrict__ W,
        int * __restrict__ tile,
        int cta_row0,
        int M,
        int blocks_per_row,
        int kb) {
    int *x_qs = tile;
    half2 *x_dm = reinterpret_cast<half2 *>(x_qs + 2 * MMQ_TILE_NE_K);
    const int lane = lane_id();
    const int warp = warp_id();

    // Canonical load_tiles_q4_K nibble expansion.  With 16 warps each warp
    // visits eight rows; all 128 rows are covered exactly once.
#pragma unroll
    for (int row = warp; row < kMTile; row += kWarps) {
        const int global_row = need_check && cta_row0 + row >= M
            ? M - 1 : cta_row0 + row;
        const block_q4_K *b =
            W + (uint64_t)global_row * (uint64_t)blocks_per_row + kb;
        const int qs0 = get_int_b4(b->qs, lane);
        x_qs[row * kWeightStride + 16 * (lane / 8) + lane % 8 + 0] =
            (qs0 >> 0) & 0x0F0F0F0F;
        x_qs[row * kWeightStride + 16 * (lane / 8) + lane % 8 + 8] =
            (qs0 >> 4) & 0x0F0F0F0F;
    }

    // The canonical loader uses 16 rows/warp and two lanes/row for metadata.
    // Only eight warps participate, so the extra split-N warps do not
    // duplicate any metadata row.
    if (warp < kMetadataWarps) {
        const int row = warp * 16 + lane / 2;
        const int ksc = lane & 1;
        const int global_row = need_check && cta_row0 + row >= M
            ? M - 1 : cta_row0 + row;
        const block_q4_K *b =
            W + (uint64_t)global_row * (uint64_t)blocks_per_row + kb;
        const int *scales = reinterpret_cast<const int *>(b->scales);
        const int sc32 = unpack_scales_q45_K(scales, ksc + 0);
        const int m32  = unpack_scales_q45_K(scales, ksc + 2);
        const uint8_t *sc8 = reinterpret_cast<const uint8_t *>(&sc32);
        const uint8_t *m8  = reinterpret_cast<const uint8_t *>(&m32);
        const half2 dm = b->dm * make_half2(1.0f, -1.0f);
#pragma unroll
        for (int l = 0; l < (int)sizeof(int); ++l) {
            x_dm[row * kWeightStride + (int)sizeof(int) * ksc + l] =
                dm * make_half2(sc8[l], m8[l]);
        }
    }
}

__device__ __forceinline__ void load_y_tile(
        const block_q8_1_mmq * __restrict__ q8,
        block_q8_1_mmq * __restrict__ tile,
        int N,
        int col0,
        int k128) {
    const int tid = linear_tid();

    // The production selector admits complete N128 tiles.  Copy those as one
    // contiguous vector range, matching canonical MMQ's flat cooperative
    // load.  The former column-major mapping made every warp issue 144-byte-
    // strided global loads; flattening turns each warp's accesses into adjacent
    // 16-byte vectors while preserving the shared representation byte-for-byte.
    if (col0 <= N - kNTile) {
        const int4 * __restrict__ src = reinterpret_cast<const int4 *>(
            q8 + (uint64_t)k128 * (uint64_t)N + (uint64_t)col0);
        int4 * __restrict__ dst = reinterpret_cast<int4 *>(tile);
#pragma unroll
        for (int vector = tid; vector < kYTileVectors;
             vector += kThreads) {
            dst[vector] = src[vector];
        }
        return;
    }

    // Keep the guarded per-column copy for the N tail accepted by the direct
    // oracle hook.  Production never takes this path.
    constexpr int threads_per_col = kThreads / kNTile;
    static_assert(threads_per_col == 4,
                  "Q8_1 DS4 copy mapping changed");
    const int col = tid & (kNTile - 1);
#pragma unroll
    for (int chunk = tid >> 7; chunk < kYChunks16;
         chunk += threads_per_col) {
        int4 value = make_int4(0, 0, 0, 0);
        if (col0 + col < N) {
            const char *src = reinterpret_cast<const char *>(
                q8 + (uint64_t)k128 * (uint64_t)N + (uint64_t)(col0 + col));
            value = *reinterpret_cast<const int4 *>(src + chunk * 16);
        }
        char *dst = reinterpret_cast<char *>(tile + col);
        *reinterpret_cast<int4 *>(dst + chunk * 16) = value;
    }
}

template <typename TileA, typename TileB, typename TileC>
__device__ __forceinline__ void fold_y_half(
        float (&acc)[kNFragPerWarp][kRowFrag][TileC::ne],
        const int * __restrict__ x_tile,
        const block_q8_1_mmq * __restrict__ y_tile,
        int x_group0) {
    static_assert(TileC::ne == 4,
                  "expected m16n8 s32 accumulator fragment");
    const half2 *x_dm = reinterpret_cast<const half2 *>(
        x_tile + 2 * MMQ_TILE_NE_K);
    const int warp = warp_id();
    const int row0 = (warp / kColWarps) * (kRowFrag * 16);
    const int nf0 = (warp % kColWarps) * kNFragPerWarp;
    const int c0 = TileC::get_j(0);
    const int c1 = TileC::get_j(1);
    const int r0 = TileC::get_i(0);
    const int r1 = TileC::get_i(2);

    // K32-phased A loads keep only the two fragments needed for this 32-row
    // band live, instead of canonical MMQ's eight K32 phases at once.  Each
    // B fragment is reused by both A fragments exactly as in canonical MMQ.
    // For every output element folds remain in canonical group order 0..7.
#pragma unroll
    for (int local_group = 0; local_group < 4; ++local_group) {
        const int x_group = x_group0 + local_group;
        TileA A[kRowFrag];
        float2 dmA[kRowFrag][2];
#pragma unroll
        for (int nr = 0; nr < kRowFrag; ++nr) {
            const int frag_row0 = row0 + nr * 16;
            ggml_cuda_mma::load_ldmatrix(
                A[nr],
                x_tile + frag_row0 * kWeightStride + x_group * QI8_1,
                kWeightStride);
            dmA[nr][0] = __half22float2(
                x_dm[(frag_row0 + r0) * kWeightStride + x_group]);
            dmA[nr][1] = __half22float2(
                x_dm[(frag_row0 + r1) * kWeightStride + x_group]);
        }

#pragma unroll
        for (int nf = 0; nf < kNFragPerWarp; ++nf) {
            const int col_base = (nf0 + nf) * 8;
            TileB B;
            const int *b_qs = reinterpret_cast<const int *>(
                &y_tile[col_base].qs[local_group * QK8_1]);
            // Canonical NVIDIA MMQ deliberately uses load_generic for B.
            ggml_cuda_mma::load_generic(B, b_qs, kYStrideInts);

            const float2 dsB[2] = {
                __half22float2(y_tile[col_base + c0].ds4[local_group]),
                __half22float2(y_tile[col_base + c1].ds4[local_group]),
            };

            // These are the canonical vec_dot_q8_1_q8_1_mma accumulation
            // statements.  Do not fuse the min correction into the dot fold
            // or change their order: parity depends on this reduction tree.
#pragma unroll
            for (int nr = 0; nr < kRowFrag; ++nr) {
                TileC C;
                ggml_cuda_mma::mma(C, A[nr], B);
#pragma unroll
                for (int l = 0; l < TileC::ne; ++l) {
                    acc[nf][nr][l] +=
                        dmA[nr][l / 2].x * dsB[l % 2].x * C.x[l];
                    acc[nf][nr][l] +=
                        dmA[nr][l / 2].y * dsB[l % 2].y;
                }
            }
        }
    }
}

template <bool to_fixup, bool need_check>
__device__ __forceinline__ void process_tile_range(
        const block_q4_K * __restrict__ W,
        const block_q8_1_mmq * __restrict__ q8,
        float * __restrict__ out,
        float * __restrict__ tmp_fixup,
        int M,
        int N,
        int K,
        int it,
        int jt,
        int kb_start,
        int kb_stop,
        int * __restrict__ x_tile,
        block_q8_1_mmq * __restrict__ y_tile) {
#if defined(TURING_MMA_AVAILABLE)
    using tile_A = ggml_cuda_mma::tile<16, 8, int>;
    using tile_B = ggml_cuda_mma::tile<8, 8, int>;
    using tile_C = ggml_cuda_mma::tile<16, 8, int>;

    const int cta_row0 = it * kMTile;
    const int col0 = jt * kNTile;
    const int blocks_per_row = K / QK_K;
    float acc[kNFragPerWarp][kRowFrag][tile_C::ne] = {};

    for (int kb = kb_start; kb < kb_stop; ++kb) {
        load_weight_tile<need_check>(
            W, x_tile, cta_row0, M, blocks_per_row, kb);
        load_y_tile(q8, y_tile, N, col0, 2 * kb + 0);
        __syncthreads();

        fold_y_half<tile_A, tile_B, tile_C>(acc, x_tile, y_tile, 0);
        __syncthreads();

        load_y_tile(q8, y_tile, N, col0, 2 * kb + 1);
        __syncthreads();

        fold_y_half<tile_A, tile_B, tile_C>(acc, x_tile, y_tile, 4);
        // Protect both shared tiles before the following K256 iteration.
        __syncthreads();
    }

    const int warp = warp_id();
    const int out_row0 =
        cta_row0 + (warp / kColWarps) * (kRowFrag * 16);
    const int out_col0 =
        col0 + (warp % kColWarps) * (kNFragPerWarp * 8);
#pragma unroll
    for (int nf = 0; nf < kNFragPerWarp; ++nf) {
#pragma unroll
        for (int nr = 0; nr < kRowFrag; ++nr) {
#pragma unroll
            for (int l = 0; l < tile_C::ne; ++l) {
                const int row = out_row0 + nr * 16 + tile_C::get_i(l);
                const int col = out_col0 + nf * 8 + tile_C::get_j(l);
                if constexpr (to_fixup) {
                    // Canonical MMQ always materializes a complete 128x128
                    // final-partial tile in block-private storage.  Tail rows
                    // and columns are masked only when the fixup publishes it.
                    tmp_fixup[(size_t)blockIdx.x * kTileElements +
                              (size_t)(col - col0) * kMTile +
                              (size_t)(row - cta_row0)] = acc[nf][nr][l];
                } else if (row < M && col < N) {
                    float value = acc[nf][nr][l];
                    // A leading Stream-K partial is deliberately left
                    // unsanitized.  Canonical fixup sanitizes only after the
                    // complete reduction tree has been reconstructed.
                    if (kb_start == 0 && kb_stop == blocks_per_row &&
                        !isfinite(value)) {
                        value = 0.0f;
                    }
                    out[(uint64_t)col * (uint64_t)M + (uint64_t)row] =
                        value;
                }
            }
        }
    }
#else
    GGML_UNUSED_VARS(W, q8, out, tmp_fixup, M, N, K, it, jt,
                     kb_start, kb_stop);
    GGML_UNUSED_VARS(x_tile, y_tile);
    NO_DEVICE_CODE;
#endif
}

__global__ __launch_bounds__(kThreads, 1)
void dense_q4_16warp_kernel(
        const block_q4_K * __restrict__ W,
        const block_q8_1_mmq * __restrict__ q8,
        float * __restrict__ out,
        int M,
        int N,
        int K) {
#if defined(TURING_MMA_AVAILABLE)
    extern __shared__ __align__(16) unsigned char dynamic_smem[];
    int *x_tile = reinterpret_cast<int *>(dynamic_smem);
    block_q8_1_mmq *y_tile = reinterpret_cast<block_q8_1_mmq *>(
        dynamic_smem + kWeightTileBytes);
    const int it = (int)blockIdx.x;
    const int jt = (int)blockIdx.y;
    process_tile_range<false, false>(
        W, q8, out, nullptr, M, N, K, it, jt,
        0, K / QK_K, x_tile, y_tile);
#else
    GGML_UNUSED_VARS(W, q8, out, M, N, K);
    NO_DEVICE_CODE;
#endif
}

// The integer partition and flattened tile order intentionally mirror
// mul_mat_q<GGML_TYPE_Q4_K, 128, ...>.  One CTA may finish a leading split
// tile, own zero or more complete tiles, and publish one trailing prefix for
// canonical mul_mat_q_stream_k_fixup.
template <bool need_check>
__global__ __launch_bounds__(kThreads, 1)
void dense_q4_16warp_streamk_kernel(
        const block_q4_K * __restrict__ W,
        const block_q8_1_mmq * __restrict__ q8,
        float * __restrict__ out,
        float * __restrict__ tmp_fixup,
        int M,
        int N,
        int K) {
#if defined(TURING_MMA_AVAILABLE)
    extern __shared__ __align__(16) unsigned char dynamic_smem[];
    int *x_tile = reinterpret_cast<int *>(dynamic_smem);
    block_q8_1_mmq *y_tile = reinterpret_cast<block_q8_1_mmq *>(
        dynamic_smem + kWeightTileBytes);

    // Convert before adding the tile bias so syntactically valid INT_MAX
    // dimensions cannot overflow signed arithmetic in device code.
    const int nty = (int)(((unsigned)M + (unsigned)kMTile - 1u) /
                          (unsigned)kMTile);
    const int ntx = (int)(((unsigned)N + (unsigned)kNTile - 1u) /
                          (unsigned)kNTile);
    const int blocks_per_row = K / QK_K;
    const int64_t total = (int64_t)nty * ntx * blocks_per_row;

    int kbc = (int)((int64_t)blockIdx.x * total / gridDim.x);
    const int kbc_stop =
        (int)((int64_t)(blockIdx.x + 1) * total / gridDim.x);

    int kb_start = kbc % blocks_per_row;
    int kb_stop = min(blocks_per_row, kb_start + kbc_stop - kbc);
    while (kbc < kbc_stop && kb_stop == blocks_per_row) {
        const int tile = kbc / blocks_per_row;
        const int jt = tile % ntx;
        const int it = tile / ntx;
        process_tile_range<false, need_check>(
            W, q8, out, tmp_fixup, M, N, K, it, jt,
            kb_start, kb_stop, x_tile, y_tile);

        kbc += blocks_per_row;
        kbc -= kbc % blocks_per_row;
        kb_start = 0;
        kb_stop = min(blocks_per_row, kbc_stop - kbc);
    }

    if (kbc >= kbc_stop) {
        return;
    }

    const int tile = kbc / blocks_per_row;
    const int jt = tile % ntx;
    const int it = tile / ntx;
    process_tile_range<true, need_check>(
        W, q8, out, tmp_fixup, M, N, K, it, jt,
        kb_start, kb_stop, x_tile, y_tile);
#else
    GGML_UNUSED_VARS(W, q8, out, tmp_fixup, M, N, K);
    NO_DEVICE_CODE;
#endif
}

struct streamk_schedule {
    unsigned nty;
    unsigned ntx;
    unsigned ntiles;
    unsigned grid_x;
    bool fixup_needed;
    size_t scratch_bytes;
};

static bool make_streamk_schedule(
        int M, int N, int nsm, streamk_schedule *schedule) {
    if (M <= 0 || N <= 0 || nsm <= 0 || schedule == nullptr) {
        return false;
    }

    const uint64_t nty64 =
        ((uint64_t)(unsigned)M + (uint64_t)kMTile - 1u) / kMTile;
    const uint64_t ntx64 =
        ((uint64_t)(unsigned)N + (uint64_t)kNTile - 1u) / kNTile;
    if (nty64 == 0 || ntx64 == 0 || nty64 > UINT32_MAX ||
        ntx64 > UINT32_MAX || nty64 > UINT32_MAX / ntx64) {
        return false;
    }

    const uint64_t ntiles64 = nty64 * ntx64;
    const uint64_t nsm64 = (uint64_t)(unsigned)nsm;
    const uint64_t nwaves = (ntiles64 + nsm64 - 1u) / nsm64;
    if (nwaves == 0 || nsm64 > UINT64_MAX / nwaves) {
        return false;
    }
    const uint64_t wave_slots = nsm64 * nwaves;
    const uint64_t efficiency = 100u * ntiles64 / wave_slots;
    const uint64_t grid64 = efficiency >= 90u ? ntiles64 : nsm64;
    if (grid64 == 0 || grid64 > UINT32_MAX) {
        return false;
    }

    const bool fixup_needed = ntiles64 % grid64 != 0;
    size_t bytes = 0;
    if (fixup_needed) {
        if (grid64 > SIZE_MAX / kTileElements / sizeof(float)) {
            return false;
        }
        bytes = (size_t)grid64 * kTileElements * sizeof(float);
    }

    schedule->nty = (unsigned)nty64;
    schedule->ntx = (unsigned)ntx64;
    schedule->ntiles = (unsigned)ntiles64;
    schedule->grid_x = (unsigned)grid64;
    schedule->fixup_needed = fixup_needed;
    schedule->scratch_bytes = bytes;
    return true;
}

} // namespace q4w16
} // anonymous namespace

extern "C" int ds4_mmq_q4_K_dense_16warp_available(int cc) {
    return GGML_CUDA_CC_IS_NVIDIA(cc) &&
           ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_AMPERE;
}

extern "C" int ds4_mmq_q4_K_dense_16warp_supported(
        int cc, int M, int N, int K) {
    if (!ds4_mmq_q4_K_dense_16warp_available(cc)) {
        return 0;
    }
    return M >= 1024 && (M % q4w16::kMTile) == 0 &&
           N >= 512 && (N % q4w16::kNTile) == 0 &&
           K >= 1024 && K <= 8192 && (K % QK_K) == 0;
}

extern "C" int ds4_mmq_q4_K_dense_16warp_prepare(void) {
    using namespace q4w16;
    int device = -1;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) {
        return -1;
    }
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess || prop.major < 8 ||
        prop.maxThreadsPerBlock < kThreads) {
        return -2;
    }
#if CUDART_VERSION >= 9000
    if ((size_t)prop.sharedMemPerBlockOptin < kSharedBytes) {
        return -2;
    }
#else
    if ((size_t)prop.sharedMemPerBlock < kSharedBytes) {
        return -2;
    }
#endif
    err = cudaFuncSetAttribute(
        dense_q4_16warp_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        (int)kSharedBytes);
    if (err != cudaSuccess) {
        return -3;
    }
    err = cudaFuncSetAttribute(
        dense_q4_16warp_streamk_kernel<false>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        (int)kSharedBytes);
    if (err != cudaSuccess) {
        return -3;
    }
    err = cudaFuncSetAttribute(
        dense_q4_16warp_streamk_kernel<true>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        (int)kSharedBytes);
    return err == cudaSuccess ? 0 : -3;
}

extern "C" int ds4_mmq_q4_K_dense_16warp_enqueue(
        const void *W,
        const void *q8_ds4,
        float *out,
        int M,
        int N,
        int K,
        cudaStream_t stream) {
    using namespace q4w16;
    if (!W || !q8_ds4 || !out || M <= 0 || N <= 0 || K <= 0 ||
        (M % kMTile) != 0 || (K % QK_K) != 0) {
        return -1;
    }

    // Convert before adding the tile bias: N is a positive signed int, but
    // N + 127 would otherwise overflow for a (syntactically valid) INT_MAX
    // direct-enqueue request.
    const unsigned grid_y =
        ((unsigned)N + (unsigned)kNTile - 1u) / (unsigned)kNTile;
    // CUDA guarantees only 65535 blocks on y/z. Production shapes are far
    // below this, but reject oversized raw-enqueue requests before launch.
    if (grid_y > 65535u) {
        return -1;
    }
    const dim3 grid((unsigned)M / (unsigned)kMTile, grid_y, 1);
    const dim3 block(32, kWarps, 1);
    dense_q4_16warp_kernel<<<grid, block, kSharedBytes, stream>>>(
        static_cast<const block_q4_K *>(W),
        static_cast<const block_q8_1_mmq *>(q8_ds4),
        out, M, N, K);
    const cudaError_t err = cudaGetLastError();
    return err == cudaSuccess ? 0 : -4;
}

extern "C" size_t ds4_mmq_q4_K_dense_16warp_streamk_scratch_bytes(
        int M, int N, int nsm) {
    q4w16::streamk_schedule schedule;
    return q4w16::make_streamk_schedule(M, N, nsm, &schedule)
        ? schedule.scratch_bytes : 0;
}

extern "C" int ds4_mmq_q4_K_dense_16warp_streamk_enqueue(
        const void *W,
        const void *q8_ds4,
        float *out,
        void *scratch,
        size_t scratch_bytes,
        int M,
        int N,
        int K,
        int nsm,
        cudaStream_t stream) {
    using namespace q4w16;
    if (!W || !q8_ds4 || !out || M <= 0 || N <= 0 || K <= 0 ||
        nsm <= 0 || (K % QK_K) != 0) {
        return -1;
    }

    streamk_schedule schedule;
    if (!make_streamk_schedule(M, N, nsm, &schedule)) {
        return -1;
    }

    const uint64_t blocks_per_row = (uint64_t)(unsigned)K / QK_K;
    const uint64_t total = (uint64_t)schedule.ntiles * blocks_per_row;
    // Match the canonical launcher's invariant: the device scheduler stores
    // flattened K-block coordinates in signed int variables.
    if (blocks_per_row == 0 || total >= (1ull << 30)) {
        return -1;
    }

    // Canonical Stream-K degenerates to one complete-K CTA per output tile
    // at high whole-tile efficiency.  Preserve that policy while using the
    // cheaper 2-D direct launch for the aligned production shapes: it avoids
    // per-CTA flattened-index divisions and needs neither scratch nor fixup.
    if (!schedule.fixup_needed && (M % kMTile) == 0) {
        return ds4_mmq_q4_K_dense_16warp_enqueue(
            W, q8_ds4, out, M, N, K, stream);
    }

    if (schedule.fixup_needed) {
        if (!scratch || scratch_bytes < schedule.scratch_bytes ||
            ((uintptr_t)scratch % alignof(float)) != 0) {
            return -1;
        }
        const cudaError_t memset_err = cudaMemsetAsync(
            scratch, 0, schedule.scratch_bytes, stream);
        if (memset_err != cudaSuccess) {
            return -2;
        }
    }

    const dim3 grid(schedule.grid_x, 1, 1);
    const dim3 block(32, kWarps, 1);
    float *tmp_fixup = schedule.fixup_needed
        ? static_cast<float *>(scratch) : nullptr;
    if ((M % kMTile) == 0) {
        dense_q4_16warp_streamk_kernel<false>
            <<<grid, block, kSharedBytes, stream>>>(
                static_cast<const block_q4_K *>(W),
                static_cast<const block_q8_1_mmq *>(q8_ds4),
                out, tmp_fixup, M, N, K);
    } else {
        dense_q4_16warp_streamk_kernel<true>
            <<<grid, block, kSharedBytes, stream>>>(
                static_cast<const block_q4_K *>(W),
                static_cast<const block_q8_1_mmq *>(q8_ds4),
                out, tmp_fixup, M, N, K);
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return -4;
    }

    if (!schedule.fixup_needed) {
        return 0;
    }

    const uint3 blocks_per_ne00_fd = init_fastdiv_values(blocks_per_row);
    const uint3 one_fd = init_fastdiv_values(1);
    const uint3 ntx_fd = init_fastdiv_values(schedule.ntx);
    const dim3 fixup_grid(schedule.grid_x, kMTile / 32, 1);
    const dim3 fixup_block(32, 4, 1);
    if ((M % kMTile) == 0) {
        constexpr bool need_check = false;
        mul_mat_q_stream_k_fixup<
            GGML_TYPE_Q4_K, kNTile, need_check, false>
            <<<fixup_grid, fixup_block, 0, stream>>>(
                /*ids_dst=*/nullptr, /*expert_bounds=*/nullptr, out,
                tmp_fixup, blocks_per_ne00_fd, M, N, M,
                one_fd, /*stride_channel_dst=*/0,
                one_fd, /*stride_sample_dst=*/0, ntx_fd);
    } else {
        constexpr bool need_check = true;
        mul_mat_q_stream_k_fixup<
            GGML_TYPE_Q4_K, kNTile, need_check, false>
            <<<fixup_grid, fixup_block, 0, stream>>>(
                /*ids_dst=*/nullptr, /*expert_bounds=*/nullptr, out,
                tmp_fixup, blocks_per_ne00_fd, M, N, M,
                one_fd, /*stride_channel_dst=*/0,
                one_fd, /*stride_sample_dst=*/0, ntx_fd);
    }
    err = cudaGetLastError();
    return err == cudaSuccess ? 0 : -5;
}
