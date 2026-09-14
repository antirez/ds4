#ifndef DS4_ROCM_INDEXER_REGISTERS_CUH
#define DS4_ROCM_INDEXER_REGISTERS_CUH

#include <stdint.h>
#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <rocwmma/rocwmma.hpp>
#endif

// A fragment contains eight coordinates in the admitted wave32 / 16x16 case.
// Packing preserves each 0..255 label exactly without retaining eight floats.
#if defined(__HIPCC__)
#define DS4_INDEXER_COORD_HD __host__ __device__
#else
#define DS4_INDEXER_COORD_HD
#endif
DS4_INDEXER_COORD_HD static inline uint32_t ds4_indexer_pack_coordinates(
        uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
    return a | (b << 8u) | (c << 16u) | (d << 24u);
}
DS4_INDEXER_COORD_HD static inline uint32_t ds4_indexer_unpack_coordinate(
        uint32_t lo, uint32_t hi, uint32_t slot) {
    return ((slot < 4u ? lo : hi) >> ((slot & 3u) * 8u)) & 255u;
}
#undef DS4_INDEXER_COORD_HD

#if defined(__HIPCC__)
template<bool INPUT_F16> struct ds4_rocm_indexer_registers_input { using type = float; };
template<> struct ds4_rocm_indexer_registers_input<true> { using type = __half; };

// Experimental, deliberately not selected by the production dispatcher.
// Caller admission: gfx1151, wave32, H=64, D=128, grid ceil(C/128) x ceil(T/16),
// block(256,1,1), positive shapes, checked buffer spans / non-aliasing output,
// and non-overflowing pos0+T with ratio>0 when causal. LDS: 37 KiB.
//
// rocWMMA does NOT guarantee a public lane/element ordering for fragment.x.
// Sources inspected: ROCm/rocWMMA tag rocm-6.4.3, rocwmma.hpp (fragment note),
// rocwmma_impl.hpp (load_matrix_sync, mma_sync, store_matrix_sync), and
// internal/io_config.hpp (PostLoadXForm / PostMmaXForm / PreStoreXForm):
// https://github.com/ROCm/rocWMMA/blob/rocm-6.4.3/library/include/rocwmma/rocwmma_impl.hpp
// https://rocm.docs.amd.com/projects/rocWMMA/en/docs-6.4.1/api-reference/api-reference-guide.html
// Instead of assuming an ISA or private library mapping, load unique coordinate
// labels through the SAME accumulator type/API. Corresponding x[i] then names
// the same matrix element after MMA. The native diagnostic must pass on the
// installed rocWMMA/toolchain before this candidate may be enabled.
template<bool INPUT_F16>
__global__ static void ds4_rocm_indexer_scores_registers_kernel(
        float *scores,
        const typename ds4_rocm_indexer_registers_input<INPUT_F16>::type *q,
        const float *weights,
        const typename ds4_rocm_indexer_registers_input<INPUT_F16>::type *index_comp,
        uint32_t n_comp, uint32_t n_tokens, uint32_t pos0,
        uint32_t n_head, uint32_t head_dim, uint32_t ratio,
        float scale, int causal) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx1151__)
    const uint32_t tid = threadIdx.x;
    if (blockDim.x != 256u || blockDim.y != 1u || blockDim.z != 1u ||
        warpSize != 32 || n_head != 64u || head_dim != 128u) return;
    const uint32_t wave = tid >> 5u;
    const uint32_t tile_c = blockIdx.x * 128u;
    const uint32_t tile_t = blockIdx.y * 16u;
    if (causal) {
        const uint32_t last_token = min(tile_t + 16u, n_tokens);
        const uint32_t max_visible = last_token > tile_t
            ? min((pos0 + last_token) / ratio, n_comp) : 0u;
        if (tile_c >= max_visible) {
            for (uint32_t i = tid; i < 16u * 128u; i += 256u) {
                const uint32_t token = tile_t + (i >> 7u);
                const uint32_t comp = tile_c + (i & 127u);
                if (token < n_tokens && comp < n_comp)
                    scores[(uint64_t)token * n_comp + comp] = -INFINITY;
            }
            return;
        }
    }

    using frag_a = rocwmma::fragment<rocwmma::matrix_a, 16, 16, 16,
                                    __half, rocwmma::row_major>;
    using frag_b = rocwmma::fragment<rocwmma::matrix_b, 16, 16, 16,
                                    __half, rocwmma::col_major>;
    using frag_c = rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, float>;
    static_assert(frag_c::num_elements == 8u, "Indexer register candidate needs wave32");
    __shared__ __half a_sh[16 * 128];
    __shared__ __half b_sh[128 * 128];
    __shared__ float coordinate_sh[16 * 16];

    coordinate_sh[tid] = (float)tid;
    __syncthreads();
    frag_c coordinates;
    rocwmma::load_matrix_sync(coordinates, coordinate_sh, 16, rocwmma::mem_row_major);
    const uint32_t coord_lo = ds4_indexer_pack_coordinates(
        (uint32_t)coordinates.x[0], (uint32_t)coordinates.x[1],
        (uint32_t)coordinates.x[2], (uint32_t)coordinates.x[3]);
    const uint32_t coord_hi = ds4_indexer_pack_coordinates(
        (uint32_t)coordinates.x[4], (uint32_t)coordinates.x[5],
        (uint32_t)coordinates.x[6], (uint32_t)coordinates.x[7]);
    float acc[8] = {};

    // Keep input staging and every MMA identical to the established WMMA path.
    // Prepared input is the identical float->half cast, performed once upstream.
    for (uint32_t i = tid; i < 128u * 128u; i += 256u) {
        const uint32_t comp = tile_c + (i >> 7u);
        const uint32_t d = i & 127u;
        __half v = __float2half(0.0f);
        if (comp < n_comp) {
            if constexpr (INPUT_F16) v = index_comp[(uint64_t)comp * head_dim + d];
            else v = __float2half(index_comp[(uint64_t)comp * head_dim + d]);
        }
        b_sh[i] = v;
    }
    __syncthreads();
    for (uint32_t h = 0; h < n_head; h++) {
        for (uint32_t i = tid; i < 16u * 128u; i += 256u) {
            const uint32_t token = tile_t + (i >> 7u);
            const uint32_t d = i & 127u;
            __half v = __float2half(0.0f);
            if (token < n_tokens) {
                if constexpr (INPUT_F16) v = q[((uint64_t)token * n_head + h) * head_dim + d];
                else v = __float2half(q[((uint64_t)token * n_head + h) * head_dim + d]);
            }
            a_sh[i] = v;
        }
        __syncthreads();
        frag_a a;
        frag_b b;
        frag_c c;
        rocwmma::fill_fragment(c, 0.0f);
        for (uint32_t k0 = 0; k0 < 128u; k0 += 16u) {
            rocwmma::load_matrix_sync(a, a_sh + k0, 128);
            rocwmma::load_matrix_sync(b, b_sh + wave * 16u * 128u + k0, 128);
            rocwmma::mma_sync(c, a, b, c);
        }
#pragma unroll
        for (uint32_t i = 0; i < 8u; i++) {
            const uint32_t rc = ds4_indexer_unpack_coordinate(coord_lo, coord_hi, i);
            const uint32_t token = tile_t + (rc >> 4u);
            const uint32_t comp = tile_c + wave * 16u + (rc & 15u);
            if (token < n_tokens && comp < n_comp) {
                const float w = weights[(uint64_t)token * n_head + h];
                acc[i] += fmaxf(c.x[i], 0.0f) * w;
            }
        }
        // Protect Q staging from the next head and retain the headwise compiler
        // boundary: do not reassociate the weighted sum across heads.
        __syncthreads();
    }
#pragma unroll
    for (uint32_t i = 0; i < 8u; i++) {
        const uint32_t rc = ds4_indexer_unpack_coordinate(coord_lo, coord_hi, i);
        const uint32_t token = tile_t + (rc >> 4u);
        const uint32_t comp = tile_c + wave * 16u + (rc & 15u);
        if (token < n_tokens && comp < n_comp) {
            float out = acc[i] * scale;
            if (causal && comp >= (pos0 + token + 1u) / ratio) out = -INFINITY;
            scores[(uint64_t)token * n_comp + comp] = out;
        }
    }
#endif
}
#endif // __HIPCC__
#endif // DS4_ROCM_INDEXER_REGISTERS_CUH
