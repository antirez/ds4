// Device-side compatibility implementation for the rocWMMA subset used by ds4
// on AMD gfx906 GPUs (Vega 20: Radeon VII / Pro VII / MI50). These devices do
// not have the MFMA matrix instructions introduced with gfx908, and the
// official rocWMMA headers reject the architecture.
//
// The ROCm backend uses software-wave32 kernel layouts. This header follows the
// same convention on wave64 hardware: lane = tid & 31 and shuffles use an
// explicit width of 32.
//
// Implemented API subset:
//   - fragment<matrix_a|matrix_b|accumulator, M, N, K, half|float, row_major|col_major>
//   - load_matrix_sync(frag, ptr, ldm)            // default row_major
//   - load_matrix_sync(frag, ptr, ldm, layout)    // explicit layout
//   - fill_fragment(frag, val)
//   - mma_sync(d, a, b, c)                        // d = a*b + c  (in-place ok)
//   - store_matrix_sync(ptr, frag, ldm, layout)
//
// The only supported tile is M=N=K=16, half x half -> float. The internal
// fragment layout does not need to match rocWMMA: 256 row-major elements are
// distributed over 32 logical lanes, eight elements per lane.
//   lane l owns tile[l/2][(l%2)*8 .. (l%2)*8+7]
//
// mma_sync su 32 lane via __shfl width=32:
//   - A[lane/2][k] k=0..15: 8 owned + 8 shfl da lane^1  (8 shfl half)
//   - B[k][(l%2)*8+c] k=0..15,c=0..7: shfl da lane 2*k+(l%2) (128 shfl half)
//   - 8 FMA per (c,k) → 128 FMA/lane per mma_sync
// This prioritizes compatibility over performance. Packed shuffles or a native
// wave64 kernel can replace it later without changing call sites.
#pragma once

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

namespace rocwmma {

// Matrix-kind tags.
struct matrix_a {};
struct matrix_b {};
struct accumulator {};

// Fragment-layout tags.
struct row_major {};
struct col_major {};

// Store layout, matching the names used by rocWMMA call sites.
enum layout_t { mem_row_major, mem_col_major, mem_stride };

namespace detail {

__device__ __forceinline__ int wmma_lane() { return static_cast<int>(threadIdx.x) & 31; }

} // namespace detail

// Eight elements per lane (256 tile elements / 32 logical lanes). Layout
// defaults to row_major, as it does for ds4's rocWMMA accumulator fragments.
template <typename MatKind, int M, int N, int K, typename T, typename Layout = row_major>
struct fragment {
    static_assert(M == 16 && N == 16 && K == 16,
                  "ds4 gfx906 shim supports only 16x16x16 tiles");
    T x[8];
};

// ===================== load_matrix_sync =====================
// matrix_a, row_major: A[i][j] is ptr + i*ldm + j.
template <int M, int N, int K>
__device__ __forceinline__ void
load_matrix_sync(fragment<matrix_a, M, N, K, half, row_major> &frag,
                 const half *ptr, uint32_t ldm) {
    const int l = detail::wmma_lane();
    const int i = l >> 1;
    const int j0 = (l & 1) * 8;
    const half *p = ptr + (uint64_t)i * ldm + j0;
    #pragma unroll
    for (int c = 0; c < 8; ++c) frag.x[c] = p[c];
}

// matrix_a, col_major: A[i][j] is ptr + j*ldm + i.
template <int M, int N, int K>
__device__ __forceinline__ void
load_matrix_sync(fragment<matrix_a, M, N, K, half, col_major> &frag,
                 const half *ptr, uint32_t ldm) {
    const int l = detail::wmma_lane();
    const int i = l >> 1;
    const int j0 = (l & 1) * 8;
    #pragma unroll
    for (int c = 0; c < 8; ++c) frag.x[c] = ptr[(uint64_t)(j0 + c) * ldm + i];
}

// matrix_b, row_major: B[k][j] is ptr + k*ldm + j.
template <int M, int N, int K>
__device__ __forceinline__ void
load_matrix_sync(fragment<matrix_b, M, N, K, half, row_major> &frag,
                 const half *ptr, uint32_t ldm) {
    const int l = detail::wmma_lane();
    const int k = l >> 1;
    const int j0 = (l & 1) * 8;
    const half *p = ptr + (uint64_t)k * ldm + j0;
    #pragma unroll
    for (int c = 0; c < 8; ++c) frag.x[c] = p[c];
}

// matrix_b, col_major: B[k][j] is ptr + j*ldm + k.
template <int M, int N, int K>
__device__ __forceinline__ void
load_matrix_sync(fragment<matrix_b, M, N, K, half, col_major> &frag,
                 const half *ptr, uint32_t ldm) {
    const int l = detail::wmma_lane();
    const int k = l >> 1;
    const int j0 = (l & 1) * 8;
    #pragma unroll
    for (int c = 0; c < 8; ++c) frag.x[c] = ptr[(uint64_t)(j0 + c) * ldm + k];
}

// ===================== fill_fragment =====================
template <int M, int N, int K>
__device__ __forceinline__ void
fill_fragment(fragment<accumulator, M, N, K, float, row_major> &frag, float v) {
    #pragma unroll
    for (int c = 0; c < 8; ++c) frag.x[c] = v;
}
template <int M, int N, int K>
__device__ __forceinline__ void
fill_fragment(fragment<accumulator, M, N, K, float, col_major> &frag, float v) {
    #pragma unroll
    for (int c = 0; c < 8; ++c) frag.x[c] = v;
}

// ===================== mma_sync =====================
// d = a * b + c   (tile 16x16: d[i][j] = sum_k a[i][k]*b[k][j])
// d/c are float accumulators; a/b are half fragments. In-place d == c is safe
// because c is copied to lane-local accumulators before d is written.
template <int M, int N, int K, typename LA, typename LB>
__device__ __forceinline__ void
mma_sync(fragment<accumulator, M, N, K, float, row_major> &d,
         const fragment<matrix_a, M, N, K, half, LA> &a,
         const fragment<matrix_b, M, N, K, half, LB> &b,
         const fragment<accumulator, M, N, K, float, row_major> &cfrag) {
    const int l = detail::wmma_lane();
    const int j0 = (l & 1) * 8;

    // Lane-local accumulator, allowing d to alias cfrag.
    float acc[8];
    #pragma unroll
    for (int c = 0; c < 8; ++c) acc[c] = cfrag.x[c];

    // Gather a full A row: eight owned values plus eight from the peer lane.
    half a_row[16];
    #pragma unroll
    for (int c = 0; c < 8; ++c) a_row[j0 + c] = a.x[c];
    const int peer = l ^ 1;
    #pragma unroll
    for (int c = 0; c < 8; ++c)
        a_row[(1 - (l & 1)) * 8 + c] = __shfl(a.x[c], peer, 32);

    // Gather B[k][j0..j0+7] from lane 2*k+(l&1), then accumulate.
    #pragma unroll
    for (int k = 0; k < 16; ++k) {
        const int src = (k << 1) + (l & 1);
        const float af = __half2float(a_row[k]);
        #pragma unroll
        for (int c = 0; c < 8; ++c) {
            const half bh = __shfl(b.x[c], src, 32);
            acc[c] = fmaf(af, __half2float(bh), acc[c]);
        }
    }

    // Write the destination fragment.
    #pragma unroll
    for (int c = 0; c < 8; ++c) d.x[c] = acc[c];
}

// ===================== store_matrix_sync =====================
template <int M, int N, int K>
__device__ __forceinline__ void
store_matrix_sync(float *ptr,
                  const fragment<accumulator, M, N, K, float, row_major> &frag,
                  uint32_t ldm, layout_t layout) {
    const int l = detail::wmma_lane();
    const int i = l >> 1;
    const int j0 = (l & 1) * 8;
    if (layout == mem_col_major) {
        // C[i][j] is ptr + j*ldm + i.
        #pragma unroll
        for (int c = 0; c < 8; ++c)
            ptr[(uint64_t)(j0 + c) * ldm + i] = frag.x[c];
    } else { // Treat mem_stride as row-major for this restricted API.
        float *p = ptr + (uint64_t)i * ldm + j0;
        #pragma unroll
        for (int c = 0; c < 8; ++c) p[c] = frag.x[c];
    }
}

} // namespace rocwmma

#endif // __HIP_PLATFORM_AMD__ || __HIPCC__
