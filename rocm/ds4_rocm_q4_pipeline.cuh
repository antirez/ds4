// SPDX-License-Identifier: MIT
// Benchmark-only K128 pipeline: one 16-byte vector of lookahead per thread.
#ifndef DS4_ROCM_Q4_PIPELINE_CUH
#define DS4_ROCM_Q4_PIPELINE_CUH
#include <stdint.h>
#include <type_traits>

#if defined(__HIPCC__) || defined(__CUDACC__)
#define DS4_Q4_PIPE_INLINE __host__ __device__ __forceinline__
#else
#define DS4_Q4_PIPE_INLINE inline
#endif

namespace ds4_rocm_q4_pipeline {
enum { tokens = 64u, columns = 128u, pitch = 144u, threads = 512u };

inline bool overlap(const void *a, uint64_t as, const void *b, uint64_t bs) {
    const uintptr_t ap=reinterpret_cast<uintptr_t>(a),bp=reinterpret_cast<uintptr_t>(b);
    return ap<=bp ? uint64_t(bp-ap)<as : uint64_t(ap-bp)<bs;
}

// Raw pointers require caller-owned capacity checks. Reject unsupported device,
// overlapping/overflowing spans and grids even in this benchmark-only API.
// One group is sufficient for Q-B and avoids ambiguous broadcast layouts.
inline bool admit(uint32_t n,uint32_t groups,uint32_t k,uint32_t m,
        uint64_t rb,uint64_t xs,uint64_t xgs,uint64_t os,
        const void *out,const void *w,const void *x,bool rhs_half,bool gfx1151) {
    const uint32_t lanes=rhs_half?8u:4u,element_bytes=rhs_half?2u:4u;
    if(!gfx1151||!out||!w||!x||n==0u||groups!=1u||k==0u||k%256u!=0u||m<8192u||
       (reinterpret_cast<uintptr_t>(out)&3u)||(reinterpret_cast<uintptr_t>(w)&1u)||
       (reinterpret_cast<uintptr_t>(x)&15u)||rb%2u||xs<k||xs%lanes||xgs!=0u||os<m||
       rb<uint64_t(k/256u)*144u||(uint64_t(n)+63u)/64u>UINT16_MAX) return false;
    if(rb>UINT64_MAX/m||
       (n>1u&&(xs>(UINT64_MAX-k)/(n-1u)||os>(UINT64_MAX-m)/(n-1u)))) return false;
    const uint64_t xe=uint64_t(n-1u)*xs+k,oe=uint64_t(n-1u)*os+m;
    if(xe>UINT64_MAX/element_bytes||oe>UINT64_MAX/4u)return false;
    const uint64_t xb=xe*element_bytes,ob=oe*4u,wb=uint64_t(m)*rb;
    if(xb>UINTPTR_MAX-reinterpret_cast<uintptr_t>(x)||
       ob>UINTPTR_MAX-reinterpret_cast<uintptr_t>(out)||
       wb>UINTPTR_MAX-reinterpret_cast<uintptr_t>(w))return false;
    return !overlap(out,ob,x,xb)&&!overlap(out,ob,w,wb);
}

template<typename Activation> struct alignas(16) packet {
    static_assert(std::is_same<Activation, float>::value ||
                  std::is_same<Activation, __half>::value, "F32/F16 RHS only");
    static constexpr uint32_t lanes = 16u / sizeof(Activation);
    Activation values[lanes];
};
static_assert(sizeof(packet<float>) == 16u && sizeof(packet<__half>) == 16u,
              "lookahead stays at one 16-byte vector per thread");

// Retain raw F32 until the subsequent staging call: eagerly converting here
// would wait for the load before issuing the current tile's independent MMA.
// Native ISA inspection must confirm that the compiler keeps this load ahead
// of MMA and delays its wait until consumption; source order alone is no proof.
template<typename Activation>
DS4_Q4_PIPE_INLINE packet<Activation> load_vector(
        const Activation *x, uint32_t n_tok, uint32_t tok0,
        uint32_t group, uint32_t k0, uint32_t j,
        uint64_t token_stride, uint64_t group_stride) {
    packet<Activation> result{};
    const uint32_t token = tok0 + j / columns;
    if (token < n_tok) {
        const Activation *src = x + uint64_t(token) * token_stride +
            uint64_t(group) * group_stride + k0 + j % columns;
        // The strict benchmark admission checks base/row/group alignment.
        __builtin_memcpy(&result, __builtin_assume_aligned(src, 16), sizeof(result));
    }
    return result;
}

template<typename Activation>
DS4_Q4_PIPE_INLINE void store_vector(
        _Float16 *lds, const packet<Activation> &value, uint32_t j) {
    _Float16 *dst = lds + (j / columns) * pitch + j % columns;
    if constexpr (std::is_same<Activation, __half>::value) {
        __builtin_memcpy(dst, &value, sizeof(value));
    } else {
        const half2 h01 = __floats2half2_rn(value.values[0], value.values[1]);
        const half2 h23 = __floats2half2_rn(value.values[2], value.values[3]);
        __builtin_memcpy(dst, &h01, sizeof(h01));
        __builtin_memcpy(dst + 2u, &h23, sizeof(h23));
    }
}

template<typename Activation>
DS4_Q4_PIPE_INLINE void stage_thread(
        _Float16 *lds, const packet<Activation> &first,
        const Activation *x, uint32_t n_tok, uint32_t tok0,
        uint32_t group, uint32_t k0, uint32_t tid,
        uint64_t token_stride, uint64_t group_stride) {
    constexpr uint32_t lanes = packet<Activation>::lanes;
    store_vector(lds, first, tid * lanes);
    for (uint32_t j = (tid + threads) * lanes;
         j < tokens * columns; j += threads * lanes) {
        const auto value = load_vector(x, n_tok, tok0, group, k0, j,
                                       token_stride, group_stride);
        store_vector(lds, value, j);
    }
}
} // namespace ds4_rocm_q4_pipeline
#undef DS4_Q4_PIPE_INLINE

#if defined(__HIPCC__) || defined(__CUDACC__)
/* Benchmark-only alternate entry. The qpair unpack/MMA/store code is kept
 * identical to K128/P144. Lookahead adds four raw VGPR words per thread and
 * no additional LDS; compiler scheduling and occupancy require native checks.
 * No production dispatcher references this entry. */
template <uint32_t ROW_TILE, uint32_t WAVES, uint32_t MIN_BLOCKS,
          typename Activation = float>
__launch_bounds__(WAVES * 32u, MIN_BLOCKS)
__global__ static void
rocm_matmul_q4_K_prefill_wmma_k128_pipeline_rowtile_strided_kernel(
        float *out,
        const char *w_base,
        const Activation *x,
        uint32_t n_tok,
        uint32_t n_groups,
        uint32_t in_dim,
        uint32_t out_dim,
        uint64_t row_bytes,
        uint64_t x_token_stride,
        uint64_t x_group_stride,
        uint64_t out_token_stride) {
#if DS4_ROCM_Q4_GFX1151_WMMA_ROWTILE_DEVICE
    if (warpSize != 32) return;

    const uint32_t tid = threadIdx.x;
    const uint32_t wave = tid >> 5u;
    const uint32_t lane = tid & 31u;
    const uint32_t lane16 = lane & 15u;
    const uint32_t group = blockIdx.z;
    static_assert(ROW_TILE == WAVES * ROCM_Q4_WMMA_FRAGMENT,
                  "one Q4 WMMA wave must own exactly 16 output rows");
    const uint32_t row0 = blockIdx.x * ROW_TILE;
    const uint32_t tok0 = blockIdx.y * ROCM_Q4_WMMA_TOKEN_TILE;
    if (group >= n_groups) return;

    const uint32_t wave_row0 = row0 + wave * ROCM_Q4_WMMA_FRAGMENT;
    const uint32_t my_row = wave_row0 + lane16;
    const uint32_t safe_row = my_row < out_dim ? my_row : out_dim - 1u;
    const cuda_block_q4_K *row_blocks =
        reinterpret_cast<const cuda_block_q4_K *>(
            w_base + ((uint64_t)group * out_dim + safe_row) * row_bytes);
    const uint32_t q4_blocks = in_dim / CUDA_QK_K;

    ds4_q4_float8_t acc0 = {0.0f, 0.0f, 0.0f, 0.0f,
                            0.0f, 0.0f, 0.0f, 0.0f};
    ds4_q4_float8_t acc1 = acc0;
    ds4_q4_float8_t acc2 = acc0;
    ds4_q4_float8_t acc3 = acc0;
    __shared__ __align__(32) _Float16
        lds_x[ROCM_Q4_WMMA_TOKEN_TILE * ROCM_Q4_WMMA_K128_LDS_PITCH];
    static_assert(ROCM_Q4_WMMA_TOKEN_TILE == ds4_rocm_q4_activation::tokens &&
                  ROCM_Q4_WMMA_K128_TILE == ds4_rocm_q4_activation::columns &&
                  ROCM_Q4_WMMA_K128_LDS_PITCH == ds4_rocm_q4_activation::pitch,
                  "F16 activation reuse must retain the K128 LDS layout");

    namespace pipe = ds4_rocm_q4_pipeline;
    static_assert(ROW_TILE == 256u && WAVES == 16u,
                  "benchmark pipeline retains the production Q-B geometry");
    const auto initial = pipe::load_vector(x, n_tok, tok0, group, 0u,
        tid * pipe::packet<Activation>::lanes, x_token_stride, x_group_stride);
    pipe::stage_thread(lds_x, initial, x, n_tok, tok0, group, 0u, tid,
                       x_token_stride, x_group_stride);
    __syncthreads();

    for (uint32_t block_index = 0u; block_index < q4_blocks;
         block_index++) {
        const cuda_block_q4_K *block = row_blocks + block_index;
        const float block_d = dev_f16_to_f32(block->d);
        const float block_dm = dev_f16_to_f32(block->dmin);

        /* Each Q4_K block contains eight qgroups.  Stage groups 0..3 and
         * 4..7 in two passes; the nested loops still consume qgroups in the
         * exact 0,1,...,7 order used by K32 and K64. */
#pragma unroll
        for (uint32_t qpair_base = 0u; qpair_base < 4u;
             qpair_base += 2u) {
            const uint32_t next_k = block_index * CUDA_QK_K +
                qpair_base * 2u * ROCM_Q4_WMMA_K_TILE + ROCM_Q4_WMMA_K128_TILE;
            const bool has_next = next_k < in_dim;
            pipe::packet<Activation> next;
            if (has_next) {
                next = pipe::load_vector(x, n_tok, tok0, group, next_k,
                    tid * pipe::packet<Activation>::lanes,
                    x_token_stride, x_group_stride);
            }


            /* Do not retain both packed qpair payloads at once.  Their
             * lifetime would increase VGPR pressure in the 16-wave q_b
             * workgroup and erase the synchronization win. */
#pragma unroll 1
            for (uint32_t qpair_offset = 0u; qpair_offset < 2u;
                 qpair_offset++) {
                const uint32_t qpair = qpair_base + qpair_offset;
                ds4_q4_uchar16_t packed0;
                ds4_q4_uchar16_t packed1;
                __builtin_memcpy(
                    &packed0, block->qs + qpair * 32u, sizeof(packed0));
                __builtin_memcpy(
                    &packed1,
                    block->qs + qpair * 32u + ROCM_Q4_WMMA_FRAGMENT,
                    sizeof(packed1));

                const ds4_rocm_q4_scales::pair metadata =
                    ds4_rocm_q4_scales::load_pair(block->scales, qpair);

#pragma unroll 1
                for (uint32_t nibble = 0u; nibble < 2u; nibble++) {
                    const uint8_t scale = static_cast<uint8_t>(
                        metadata.scales >> (nibble * 8u));
                    const uint8_t minimum = static_cast<uint8_t>(
                        metadata.minima >> (nibble * 8u));
                    const float d = block_d * (float)scale;
                    const float dm = block_dm * (float)minimum;
                    const uint32_t shift = nibble * 4u;
                    ds4_q4_half16_t weights0;
                    ds4_q4_half16_t weights1;
#pragma unroll
                    for (uint32_t i = 0u; i < ROCM_Q4_WMMA_FRAGMENT; i++) {
                        const uint8_t q0 = (packed0[i] >> shift) & 0x0fu;
                        const uint8_t q1 = (packed1[i] >> shift) & 0x0fu;
                        weights0[i] = (_Float16)(d * (float)q0 - dm);
                        weights1[i] = (_Float16)(d * (float)q1 - dm);
                    }

#pragma unroll
                    for (uint32_t token_tile = 0u; token_tile < 4u;
                         token_tile++) {
                        const uint32_t token_local =
                            token_tile * ROCM_Q4_WMMA_FRAGMENT + lane16;
                        const uint32_t activation_offset =
                            (qpair_offset * 2u + nibble) *
                            ROCM_Q4_WMMA_K_TILE;
                        const _Float16 *activation =
                            lds_x +
                            token_local * ROCM_Q4_WMMA_K128_LDS_PITCH +
                            activation_offset;
                        const ds4_q4_half16_t activation0 =
                            *reinterpret_cast<const ds4_q4_half16_t *>(
                                activation);
                        const ds4_q4_half16_t activation1 =
                            *reinterpret_cast<const ds4_q4_half16_t *>(
                                activation + ROCM_Q4_WMMA_FRAGMENT);
                        if (token_tile == 0u) {
                            acc0 =
                                __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
                                    weights0, activation0, acc0);
                            acc0 =
                                __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
                                    weights1, activation1, acc0);
                        } else if (token_tile == 1u) {
                            acc1 =
                                __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
                                    weights0, activation0, acc1);
                            acc1 =
                                __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
                                    weights1, activation1, acc1);
                        } else if (token_tile == 2u) {
                            acc2 =
                                __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
                                    weights0, activation0, acc2);
                            acc2 =
                                __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
                                    weights1, activation1, acc2);
                        } else {
                            acc3 =
                                __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
                                    weights0, activation0, acc3);
                            acc3 =
                                __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
                                    weights1, activation1, acc3);
                        }
                    }
                }
            }
            if (has_next) {
                // Consumers must finish before reusing the single LDS tile.
                __syncthreads();
                pipe::stage_thread(lds_x, next, x, n_tok, tok0, group,
                                   next_k, tid, x_token_stride, x_group_stride);
                __syncthreads();
            }
        }
    }

#pragma unroll
    for (uint32_t token_tile = 0u; token_tile < 4u; token_tile++) {
        const uint32_t tok =
            tok0 + token_tile * ROCM_Q4_WMMA_FRAGMENT + lane16;
        if (tok >= n_tok) continue;
        const ds4_q4_float8_t acc = token_tile == 0u
            ? acc0
            : (token_tile == 1u ? acc1
                                : (token_tile == 2u ? acc2 : acc3));
#pragma unroll
        for (uint32_t j = 0u; j < 8u; j++) {
            const uint32_t row = wave_row0 + 2u * j + (lane >> 4u);
            if (row < out_dim) {
                out[(uint64_t)tok * out_token_stride +
                    (uint64_t)group * out_dim + row] = acc[j];
            }
        }
    }
#else
    (void)out;
    (void)w_base;
    (void)x;
    (void)n_tok;
    (void)n_groups;
    (void)in_dim;
    (void)out_dim;
    (void)row_bytes;
    (void)x_token_stride;
    (void)x_group_stride;
    (void)out_token_stride;
#endif
}
#endif // GPU kernel; helpers above are also exercised by the host oracle.

#endif
