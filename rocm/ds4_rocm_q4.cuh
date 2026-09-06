// DS4 ROCm dense Q4_K kernels and launch wrappers.
//
// This module is included after ds4_rocm_moe.cuh so it can reuse the
// canonical Q8_K activation quantizer, Q4_K/Q8_K block dot product, and
// quarter-wave reduction already used by the routed-MoE implementation.
// Launches intentionally stay on ROCm's default stream: cuda_tmp_alloc() is a
// single reusable scratch arena whose lifetime is protected by that ordering.

#include "ds4_rocm_q4_lds.cuh"
#include "ds4_rocm_q4_decode.cuh"
#include "ds4_rocm_q4_wmma_load.cuh"
#include <type_traits>

static_assert(sizeof(cuda_block_q4_K) == 144u,
              "ROCm Q4_K block layout must match GGUF");
static_assert(sizeof(cuda_block_q8_K) == 292u,
              "ROCm Q8_K activation block layout must match the dot kernel");

__global__ static void rocm_matmul_q4_K_dense_kernel(
        float *out,
        const char *w_base,
        const cuda_block_q8_K *xq,
        uint64_t row_bytes,
        uint32_t xq_blocks,
        uint32_t out_dim,
        uint32_t n_tok) {
    const uint32_t lane = threadIdx.x & 7u;
    const uint32_t row_lane = threadIdx.x >> 3u;
    const uint32_t tok = blockIdx.y;
    const uint32_t row = blockIdx.x * 32u + row_lane;
    if (tok >= n_tok || row >= out_dim) return;

    const cuda_block_q8_K *xqb = xq + (uint64_t)tok * xq_blocks;
    const cuda_block_q4_K *wr = reinterpret_cast<const cuda_block_q4_K *>(
            w_base + (uint64_t)row * row_bytes);
    float acc = 0.0f;
    for (uint32_t b = lane; b < xq_blocks; b += 8u) {
        acc += dev_dot_q4_K_q8_K_block(wr + b, xqb + b);
    }
    acc = quarter_warp_sum_f32(acc, lane);
    if (lane == 0u) out[(uint64_t)tok * out_dim + row] = acc;
}

/* Experimental standalone Q-b decode only. Keep the generic loop, block
 * dot and Q8_K inputs; dispatch guarantees exactly four K blocks. Removing
 * the four empty workers doubles rows/workgroup, not arithmetic throughput
 * by assertion. GPU parity and timing are required before any promotion. */
__global__ static void rocm_matmul_q4_K_dense_decode_lane4_kernel(
        float *out, const char *w_base, const cuda_block_q8_K *xq,
        uint64_t row_bytes, uint32_t xq_blocks, uint32_t out_dim,
        uint32_t n_tok) {
    const uint32_t lane = ds4_rocm_q4_decode::lane(threadIdx.x);
    const uint32_t row = ds4_rocm_q4_decode::row(blockIdx.x, threadIdx.x);
    const uint32_t tok = blockIdx.y;
    if (tok >= n_tok || row >= out_dim) return;
    const cuda_block_q8_K *xqb = xq + (uint64_t)tok * xq_blocks;
    const cuda_block_q4_K *wr = reinterpret_cast<const cuda_block_q4_K *>(
        w_base + (uint64_t)row * row_bytes);
    float acc = 0.0f;
    for (uint32_t b = lane; b < xq_blocks; b += 8u) {
        acc += dev_dot_q4_K_q8_K_block(wr + b, xqb + b);
    }
    acc = ds4_rocm_q4_decode::canonical_offset4(acc);
    const MASK_T mask = static_cast<MASK_T>(
        ds4_rocm_q4_decode::mask(threadIdx.x, warpSize));
    acc += __shfl_down_sync(mask, acc, 2, 4);
    acc += __shfl_down_sync(mask, acc, 1, 4);
    if (lane == 0u) out[(uint64_t)tok * out_dim + row] = acc;
}

// Same-host-thread launch oracle, not a device-completion/performance counter.
// No atomics or counter updates on the default/rollback path.
static thread_local uint64_t g_rocm_q4_decode_lane4_launches;
extern "C" void ds4_rocm_test_q4_decode_lane4_reset(void) {
    g_rocm_q4_decode_lane4_launches = 0u;
}
extern "C" uint64_t ds4_rocm_test_q4_decode_lane4_get_calls(void) {
    return g_rocm_q4_decode_lane4_launches;
}

static uint32_t rocm_q4_decode_lane4_runtime_wave_size(void) {
#if defined(__HIP_PLATFORM_AMD__)
    int device = -1;
    if (cudaGetDevice(&device) != cudaSuccess || device < 0) return 0u;
    static thread_local int cached_device = -1;
    static thread_local uint32_t cached_wave = 0u;
    if (cached_device == device) return cached_wave;
    cudaDeviceProp prop = {};
    if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) return 0u;
    cached_device = device;
    cached_wave = (uint32_t)prop.warpSize;
    return cached_wave;
#else
    return 0u;
#endif
}

static int rocm_q4_decode_lane4_select(
        uint64_t in_dim, uint64_t out_dim, uint64_t n_tok) {
    // Presence semantics, like dense-pair: even DISABLE=0 means rollback.
    const bool required = getenv("DS4_ROCM_REQUIRE_Q4_DECODE_LANE4") != NULL;
    // Only REQUIRE needs checking outside the exact scope. Do not cache the
    // environment: native same-process A/B deliberately changes these flags.
    if (!ds4_rocm_q4_decode::shape(in_dim, out_dim, n_tok)) {
        if (!required) return ds4_rocm_q4_decode::fallback;
        fprintf(stderr, DS4_GPU_LOG_PREFIX
                "required Q4 decode lane4 outside K1024/M32768/N1..8 "
                "(K=%llu M=%llu N=%llu)\n",
                (unsigned long long)in_dim, (unsigned long long)out_dim,
                (unsigned long long)n_tok);
        return ds4_rocm_q4_decode::required_failure;
    }
    const bool enabled = required ||
        getenv("DS4_ROCM_ENABLE_Q4_DECODE_LANE4") != NULL;
    if (!enabled) return ds4_rocm_q4_decode::fallback;
    const bool disabled = getenv("DS4_ROCM_DISABLE_Q4_DECODE_LANE4") != NULL;
    // Intended arithmetic is exact, but GPU parity/codegen are not validated.
    // Keep quality-mode on the canonical path until that evidence exists.
    const uint32_t wave = !disabled && !g_quality_mode
        ? rocm_q4_decode_lane4_runtime_wave_size() : 0u;
    const int decision = ds4_rocm_q4_decode::select(
        in_dim, out_dim, n_tok, wave, g_quality_mode, enabled, disabled, required);
    if (decision == ds4_rocm_q4_decode::required_failure) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX
                "required Q4 decode lane4 unavailable "
                "(K=%llu M=%llu N=%llu wave=%u disabled=%d quality=%d)\n",
                (unsigned long long)in_dim, (unsigned long long)out_dim,
                (unsigned long long)n_tok, wave, (int)disabled,
                g_quality_mode ? 1 : 0);
    }
    return decision;
}

/* Latency-oriented pair variant of the canonical dense kernel.  Concatenating
 * the two row-tile domains keeps exactly the same per-row dot and reduction
 * order while sharing one launch between Q-A and KV. */
__global__ static void rocm_matmul_q4_K_dense_pair_kernel(
        float *out0,
        float *out1,
        const char *w0,
        const char *w1,
        const cuda_block_q8_K *xq,
        uint64_t row_bytes,
        uint32_t xq_blocks,
        uint32_t out0_dim,
        uint32_t out1_dim,
        uint32_t n_tok) {
    const uint32_t lane = threadIdx.x & 7u;
    const uint32_t row_lane = threadIdx.x >> 3u;
    const uint32_t out0_tiles = (out0_dim - 1u) / 32u + 1u;
    const bool second = blockIdx.x >= out0_tiles;
    const uint32_t row_tile = second ? blockIdx.x - out0_tiles : blockIdx.x;
    const uint32_t row = row_tile * 32u + row_lane;
    const uint32_t tok = blockIdx.y;
    const uint32_t out_dim = second ? out1_dim : out0_dim;
    if (tok >= n_tok || row >= out_dim) return;

    float *const out = second ? out1 : out0;
    const char *const w_base = second ? w1 : w0;
    const cuda_block_q8_K *xqb = xq + (uint64_t)tok * xq_blocks;
    const cuda_block_q4_K *wr = reinterpret_cast<const cuda_block_q4_K *>(
            w_base + (uint64_t)row * row_bytes);
    float acc = 0.0f;
    for (uint32_t b = lane; b < xq_blocks; b += 8u) {
        acc += dev_dot_q4_K_q8_K_block(wr + b, xqb + b);
    }
    acc = quarter_warp_sum_f32(acc, lane);
    if (lane == 0u) out[(uint64_t)tok * out_dim + row] = acc;
}

/* One independent activation row and Q4_K matrix per output group.  This is
 * the canonical dense decode walk with only a group grid dimension added. */
__global__ static void rocm_matmul_q4_K_dense_grouped_decode_kernel(
        float *out, const char *w_base, const cuda_block_q8_K *xq,
        uint64_t row_bytes, uint32_t xq_blocks, uint32_t out_dim,
        uint32_t n_groups) {
    const uint32_t lane = threadIdx.x & 7u;
    const uint32_t row_lane = threadIdx.x >> 3u;
    const uint32_t row = blockIdx.x * 32u + row_lane;
    const uint32_t group = blockIdx.z;
    if (group >= n_groups || row >= out_dim) return;
    const cuda_block_q8_K *xqb = xq + (uint64_t)group * xq_blocks;
    const cuda_block_q4_K *wr = reinterpret_cast<const cuda_block_q4_K *>(
        w_base + ((uint64_t)group * out_dim + row) * row_bytes);
    float acc = 0.0f;
    for (uint32_t b = lane; b < xq_blocks; b += 8u) {
        acc += dev_dot_q4_K_q8_K_block(wr + b, xqb + b);
    }
    acc = quarter_warp_sum_f32(acc, lane);
    if (lane == 0u) out[(uint64_t)group * out_dim + row] = acc;
}

/* Dense Q4_K prefill tile for RDNA/ROCm.
 *
 * The legacy kernel gives one eight-lane group a single (token,row) dot.  As
 * a result every token walks the complete Q4_K row independently and every
 * one of the 32 row groups in a workgroup also fetches the same Q8_K input
 * blocks.  Prefill is therefore dominated by redundant global reads.
 *
 * This kernel keeps the legacy eight-lane block assignment and reduction
 * order, but computes eight token columns at once.  A Q4_K block is decoded
 * once into eight integer dot products, and an 8x8 token/K-block tile of the
 * canonical Q8_K activations is staged in LDS for reuse by all 32 rows.  The
 * K loop advances in groups of eight so lane L still accumulates blocks
 * L,L+8,... in exactly the order used by rocm_matmul_q4_K_dense_kernel.
 *
 * Packed LDS footprint: 8 tokens * 8 K blocks * 292 bytes = 18,688 bytes.
 * The output-B aligned variant uses 304 bytes/block (19,456 bytes total).
 * The final, partial token and K tiles are handled without an early return;
 * every thread reaches the load barrier and the reuse barrier before each
 * subsequent tile. No reuse barrier is needed after the final tile.
 */
enum {
    ROCM_Q4_PREFILL_TOKEN_TILE = 8u,
    ROCM_Q4_PREFILL_KBLOCK_TILE = 8u,
    ROCM_Q4_PREFILL_K1024_KBLOCK_TILE = 4u,
    ROCM_Q4_PREFILL_K1024_ROWS = 64u,
    ROCM_Q4_Q8K_WORDS = sizeof(cuda_block_q8_K) / sizeof(uint32_t),
};
static_assert((sizeof(cuda_block_q8_K) % sizeof(uint32_t)) == 0u,
              "ROCm Q8_K LDS copies require a whole number of words");
static_assert(ROCM_Q4_Q8K_WORDS == ds4_rocm_q4_lds::block_words,
              "Q4 prefill LDS helper must match the Q8_K ABI");

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
/* Direct-Q4 WMMA for resident gfx1151 prefill.
 *
 * This deliberately mirrors the live Q8 WMMA kernel: each wave32 owns 16
 * output rows, the workgroup computes 64 tokens, K advances by 32, only the
 * 64x32 activation tile is staged in 4 KiB of LDS, and accumulator fragments
 * are written straight to the output.  The row tile is shape-selected to 64,
 * 128, or 256 so every activation tile is reused as broadly as the output
 * shape permits; the retained 64-row instantiation is also an A/B control.
 * K64 can stage two adjacent F32 activations per loader iteration, matching
 * the established Q8 WMMA load/conversion pattern; aligned 256-row K128
 * launches stage four.  Each lane
 * dequantizes one Q4_K row/group directly into two F16 register vectors, so
 * there is no Q8_K activation scratch and no persistent F16 weight sidecar.
 *
 * Arithmetic is not bit-identical to Q4_K x Q8_K: activations and transient
 * weights round to F16 before F32 WMMA accumulation.  Host policy therefore
 * enables this path by default only for physically device-resident, gfx1151
 * wave32 prefill outside quality mode.  SSD streaming remains separately
 * gated, and DISABLE provides an authoritative rollback.
 */
enum {
    ROCM_Q4_WMMA_TOKEN_TILE = 64u,
    ROCM_Q4_WMMA_K_TILE = 32u,
    ROCM_Q4_WMMA_K64_TILE = 64u,
    ROCM_Q4_WMMA_K64_LDS_PITCH = 80u,
    ROCM_Q4_WMMA_K128_TILE = 128u,
    ROCM_Q4_WMMA_K128_LDS_PITCH = 144u,
    ROCM_Q4_WMMA_FRAGMENT = 16u,
};
static_assert(ROCM_Q4_WMMA_K64_TILE == 2u * ROCM_Q4_WMMA_K_TILE,
              "K64 must combine exactly two adjacent Q4_K qgroups");
static_assert(ROCM_Q4_WMMA_K64_LDS_PITCH >= ROCM_Q4_WMMA_K64_TILE &&
              (ROCM_Q4_WMMA_K64_LDS_PITCH % ROCM_Q4_WMMA_FRAGMENT) == 0u,
              "K64 LDS rows must preserve aligned half16 WMMA loads");
static_assert(ROCM_Q4_WMMA_K128_TILE == 2u * ROCM_Q4_WMMA_K64_TILE,
              "K128 must combine exactly four adjacent Q4_K qgroups");
static_assert(ROCM_Q4_WMMA_K128_LDS_PITCH >= ROCM_Q4_WMMA_K128_TILE &&
              (ROCM_Q4_WMMA_K128_LDS_PITCH % ROCM_Q4_WMMA_FRAGMENT) == 0u,
              "K128 LDS rows must preserve aligned half16 WMMA loads");

#if defined(__HIP_DEVICE_COMPILE__) && __HIP_DEVICE_COMPILE__ && \
    defined(__gfx1151__) && \
    (!defined(__AMDGCN_WAVEFRONT_SIZE__) || \
     __AMDGCN_WAVEFRONT_SIZE__ == 32)
#define DS4_ROCM_Q4_GFX1151_WMMA_ROWTILE_DEVICE 1
typedef _Float16 __attribute__((ext_vector_type(16))) ds4_q4_half16_t;
typedef float __attribute__((ext_vector_type(8))) ds4_q4_float8_t;
typedef uint8_t __attribute__((ext_vector_type(16))) ds4_q4_uchar16_t;
#else
#define DS4_ROCM_Q4_GFX1151_WMMA_ROWTILE_DEVICE 0
#endif

template <uint32_t ROW_TILE, uint32_t WAVES, uint32_t MIN_BLOCKS,
          bool LOAD2>
__launch_bounds__(WAVES * 32u, MIN_BLOCKS)
__global__ static void rocm_matmul_q4_K_prefill_wmma_rowtile_strided_kernel(
        float *out,
        const char *w_base,
        const float *x,
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
    /* The 16-lane _Float16 vector loads below have 32-byte alignment on the
     * AMDGPU target; half2 itself would require only four bytes. */
    __shared__ __align__(32) _Float16
        lds_x[ROCM_Q4_WMMA_TOKEN_TILE * ROCM_Q4_WMMA_K_TILE];

    for (uint32_t block_index = 0u; block_index < q4_blocks;
         block_index++) {
        const cuda_block_q4_K *block = row_blocks + block_index;
        const float block_d = dev_f16_to_f32(block->d);
        const float block_dm = dev_f16_to_f32(block->dmin);
#pragma unroll
        for (uint32_t qpair = 0u; qpair < 4u; qpair++) {
            /* Adjacent 32-value groups are the low/high nibbles of the same
             * 32 payload bytes.  Keep them in registers across both K tiles
             * instead of issuing the same global loads twice. */
            ds4_q4_uchar16_t packed0;
            ds4_q4_uchar16_t packed1;
            __builtin_memcpy(
                &packed0, block->qs + qpair * 32u, sizeof(packed0));
            __builtin_memcpy(
                &packed1,
                block->qs + qpair * 32u + ROCM_Q4_WMMA_FRAGMENT,
                sizeof(packed1));

            #pragma unroll
            for (uint32_t nibble = 0u; nibble < 2u; nibble++) {
                const uint32_t qgroup = qpair * 2u + nibble;
                const uint32_t group32 = block_index * 8u + qgroup;
                if constexpr (LOAD2) {
                    for (uint32_t j = tid * 2u;
                         j < ROCM_Q4_WMMA_TOKEN_TILE * ROCM_Q4_WMMA_K_TILE;
                         j += blockDim.x * 2u) {
                        const uint32_t tok_local = j >> 5u;
                        const uint32_t kk = j & 31u;
                        const uint32_t tok = tok0 + tok_local;
                        half2 value = __floats2half2_rn(0.0f, 0.0f);
                        if (tok < n_tok) {
                            const float2 pair =
                                *reinterpret_cast<const float2 *>(
                                    x + (uint64_t)tok * x_token_stride +
                                    (uint64_t)group * x_group_stride +
                                    (uint64_t)group32 *
                                        ROCM_Q4_WMMA_K_TILE +
                                    kk);
                            value = __floats2half2_rn(pair.x, pair.y);
                        }
                        *reinterpret_cast<half2 *>(lds_x + j) = value;
                    }
                } else {
                    for (uint32_t j = tid;
                         j < ROCM_Q4_WMMA_TOKEN_TILE * ROCM_Q4_WMMA_K_TILE;
                         j += blockDim.x) {
                        const uint32_t tok_local = j >> 5u;
                        const uint32_t kk = j & 31u;
                        const uint32_t tok = tok0 + tok_local;
                        float value = 0.0f;
                        if (tok < n_tok) {
                            value = x[(uint64_t)tok * x_token_stride +
                                      (uint64_t)group * x_group_stride +
                                      (uint64_t)group32 *
                                          ROCM_Q4_WMMA_K_TILE +
                                      kk];
                        }
                        lds_x[j] = (_Float16)value;
                    }
                }
                __syncthreads();

                uint8_t scale = 0u;
                uint8_t minimum = 0u;
                dev_q4_K_get_scale_min(
                    qgroup, block->scales, &scale, &minimum);
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
                    const _Float16 *activation =
                        lds_x + token_local * ROCM_Q4_WMMA_K_TILE;
                    const ds4_q4_half16_t activation0 =
                        *reinterpret_cast<const ds4_q4_half16_t *>(activation);
                    const ds4_q4_half16_t activation1 =
                        *reinterpret_cast<const ds4_q4_half16_t *>(
                            activation + ROCM_Q4_WMMA_FRAGMENT);
                    if (token_tile == 0u) {
                        acc0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
                            weights0, activation0, acc0);
                        acc0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
                            weights1, activation1, acc0);
                    } else if (token_tile == 1u) {
                        acc1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
                            weights0, activation0, acc1);
                        acc1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
                            weights1, activation1, acc1);
                    } else if (token_tile == 2u) {
                        acc2 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
                            weights0, activation0, acc2);
                        acc2 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
                            weights1, activation1, acc2);
                    } else {
                        acc3 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
                            weights0, activation0, acc3);
                        acc3 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
                            weights1, activation1, acc3);
                    }
                }
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

/* Default K64 variant.  The K32 kernel above intentionally remains
 * byte-for-byte unchanged so its code generation stays a stable rollback and
 * A/B baseline.  K64 stages two adjacent 32-value activation groups
 * at once and consumes the low/high Q4 nibbles in their original order,
 * reducing the LDS barrier pairs from sixteen to eight per Q4_K block.
 *
 * A padded 80-half LDS pitch keeps every 16-half WMMA load 32-byte aligned
 * while rotating successive token rows across LDS banks.  Natural pitch 64
 * would map every 128-byte row to the same bank pattern on RDNA wave32.
 * LOAD4 changes only the aligned activation loader. The default template
 * argument keeps all existing enqueue-only benchmark hooks on their original
 * LOAD2/scalar instantiations, independent of the production opt-out. */
template <uint32_t ROW_TILE, uint32_t WAVES, uint32_t MIN_BLOCKS,
          bool LOAD2, bool LOAD4 = false>
__launch_bounds__(WAVES * 32u, MIN_BLOCKS)
__global__ static void
rocm_matmul_q4_K_prefill_wmma_k64_p80_rowtile_strided_kernel(
        float *out,
        const char *w_base,
        const float *x,
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
        lds_x[ROCM_Q4_WMMA_TOKEN_TILE * ROCM_Q4_WMMA_K64_LDS_PITCH];
    static_assert(ROCM_Q4_WMMA_K64_LDS_PITCH ==
                      ds4_rocm_q4_wmma_load::pitch &&
                  ROCM_Q4_WMMA_K64_TILE == ds4_rocm_q4_wmma_load::columns &&
                  ROCM_Q4_WMMA_TOKEN_TILE == ds4_rocm_q4_wmma_load::tokens,
                  "K64 float4 loader must match the WMMA LDS layout");
    static_assert(!LOAD4 || LOAD2, "float4 extends the aligned float2 loader");

    for (uint32_t block_index = 0u; block_index < q4_blocks;
         block_index++) {
        const cuda_block_q4_K *block = row_blocks + block_index;
        const float block_d = dev_f16_to_f32(block->d);
        const float block_dm = dev_f16_to_f32(block->dmin);
#pragma unroll
        for (uint32_t qpair = 0u; qpair < 4u; qpair++) {
            ds4_q4_uchar16_t packed0;
            ds4_q4_uchar16_t packed1;
            __builtin_memcpy(
                &packed0, block->qs + qpair * 32u, sizeof(packed0));
            __builtin_memcpy(
                &packed1,
                block->qs + qpair * 32u + ROCM_Q4_WMMA_FRAGMENT,
                sizeof(packed1));

            const uint32_t group32_base = block_index * 8u + qpair * 2u;
            if constexpr (LOAD4) {
                // Four adjacent F32 values, with the same two pairwise RN
                // conversions/stores as LOAD2. No wider LDS footprint, new
                // precision boundary, or change to WMMA accumulation order.
                for (uint32_t j = tid * 4u;
                     j < ROCM_Q4_WMMA_TOKEN_TILE * ROCM_Q4_WMMA_K64_TILE;
                     j += blockDim.x * 4u) {
                    half2 value01 = __floats2half2_rn(0.0f, 0.0f);
                    half2 value23 = value01;
                    if (tok0 + ds4_rocm_q4_wmma_load::token(j) < n_tok) {
                        const float4 values =
                            *reinterpret_cast<const float4 *>(
                                x + ds4_rocm_q4_wmma_load::source(
                                    j, tok0, group,
                                    group32_base * ROCM_Q4_WMMA_K_TILE,
                                    x_token_stride, x_group_stride));
                        value01 = __floats2half2_rn(values.x, values.y);
                        value23 = __floats2half2_rn(values.z, values.w);
                    }
                    _Float16 *const dst =
                        lds_x + ds4_rocm_q4_wmma_load::destination(j);
                    *reinterpret_cast<half2 *>(dst) = value01;
                    *reinterpret_cast<half2 *>(dst + 2u) = value23;
                }
            } else if constexpr (LOAD2) {
                for (uint32_t j = tid * 2u;
                     j < ROCM_Q4_WMMA_TOKEN_TILE * ROCM_Q4_WMMA_K64_TILE;
                     j += blockDim.x * 2u) {
                    const uint32_t tok_local = j >> 6u;
                    const uint32_t kk = j & 63u;
                    const uint32_t tok = tok0 + tok_local;
                    half2 value = __floats2half2_rn(0.0f, 0.0f);
                    if (tok < n_tok) {
                        const float2 pair =
                            *reinterpret_cast<const float2 *>(
                                x + (uint64_t)tok * x_token_stride +
                                (uint64_t)group * x_group_stride +
                                (uint64_t)group32_base *
                                    ROCM_Q4_WMMA_K_TILE +
                                kk);
                        value = __floats2half2_rn(pair.x, pair.y);
                    }
                    *reinterpret_cast<half2 *>(
                        lds_x + tok_local * ROCM_Q4_WMMA_K64_LDS_PITCH +
                        kk) = value;
                }
            } else {
                for (uint32_t j = tid;
                     j < ROCM_Q4_WMMA_TOKEN_TILE * ROCM_Q4_WMMA_K64_TILE;
                     j += blockDim.x) {
                    const uint32_t tok_local = j >> 6u;
                    const uint32_t kk = j & 63u;
                    const uint32_t tok = tok0 + tok_local;
                    float value = 0.0f;
                    if (tok < n_tok) {
                        value = x[(uint64_t)tok * x_token_stride +
                                  (uint64_t)group * x_group_stride +
                                  (uint64_t)group32_base *
                                      ROCM_Q4_WMMA_K_TILE +
                                  kk];
                    }
                    lds_x[tok_local * ROCM_Q4_WMMA_K64_LDS_PITCH + kk] =
                        (_Float16)value;
                }
            }
            __syncthreads();

            /* Do not unroll the two nibbles: one pair of half16 weight
             * vectors must die before the next one is materialized.  This
             * caps VGPR pressure while preserving qgroup accumulation order. */
#pragma unroll 1
            for (uint32_t nibble = 0u; nibble < 2u; nibble++) {
                const uint32_t qgroup = qpair * 2u + nibble;
                uint8_t scale = 0u;
                uint8_t minimum = 0u;
                dev_q4_K_get_scale_min(
                    qgroup, block->scales, &scale, &minimum);
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
                    const _Float16 *activation =
                        lds_x +
                        token_local * ROCM_Q4_WMMA_K64_LDS_PITCH +
                        nibble * ROCM_Q4_WMMA_K_TILE;
                    const ds4_q4_half16_t activation0 =
                        *reinterpret_cast<const ds4_q4_half16_t *>(activation);
                    const ds4_q4_half16_t activation1 =
                        *reinterpret_cast<const ds4_q4_half16_t *>(
                            activation + ROCM_Q4_WMMA_FRAGMENT);
                    if (token_tile == 0u) {
                        acc0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
                            weights0, activation0, acc0);
                        acc0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
                            weights1, activation1, acc0);
                    } else if (token_tile == 1u) {
                        acc1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
                            weights0, activation0, acc1);
                        acc1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
                            weights1, activation1, acc1);
                    } else if (token_tile == 2u) {
                        acc2 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
                            weights0, activation0, acc2);
                        acc2 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
                            weights1, activation1, acc2);
                    } else {
                        acc3 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
                            weights0, activation0, acc3);
                        acc3 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
                            weights1, activation1, acc3);
                    }
                }
            }
            if (ds4_rocm_q4_wmma_load::needs_reuse_barrier(
                    block_index, q4_blocks, qpair, 4u)) {
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

/* Long-prefill q_b K128 stage.  Four adjacent 32-value activation groups are
 * staged behind one barrier pair, reducing the K64/P80 synchronization count
 * by another 2x.  P144 retains a 32-byte-aligned half16 base while rotating
 * consecutive token rows across LDS banks.  float4 loads also halve the
 * activation-load instruction count relative to K64's float2 loader without
 * changing any F32->F16 rounding or the qgroup accumulation order.
 *
 * Keep this as a separate kernel and instantiate it only for the 256-row
 * geometry: its 18 KiB LDS tile is appropriate for q_b's 32768 output rows,
 * but could reduce occupancy on the smaller projections. */
template <uint32_t ROW_TILE, uint32_t WAVES, uint32_t MIN_BLOCKS>
__launch_bounds__(WAVES * 32u, MIN_BLOCKS)
__global__ static void
rocm_matmul_q4_K_prefill_wmma_k128_p144_rowtile_strided_kernel(
        float *out,
        const char *w_base,
        const float *x,
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
            const uint32_t group32_base =
                block_index * 8u + qpair_base * 2u;
            for (uint32_t j = tid * 4u;
                 j < ROCM_Q4_WMMA_TOKEN_TILE * ROCM_Q4_WMMA_K128_TILE;
                 j += blockDim.x * 4u) {
                const uint32_t tok_local = j >> 7u;
                const uint32_t kk = j & 127u;
                const uint32_t tok = tok0 + tok_local;
                half2 value01 = __floats2half2_rn(0.0f, 0.0f);
                half2 value23 = value01;
                if (tok < n_tok) {
                    const float4 values =
                        *reinterpret_cast<const float4 *>(
                            x + (uint64_t)tok * x_token_stride +
                            (uint64_t)group * x_group_stride +
                            (uint64_t)group32_base *
                                ROCM_Q4_WMMA_K_TILE +
                            kk);
                    value01 = __floats2half2_rn(values.x, values.y);
                    value23 = __floats2half2_rn(values.z, values.w);
                }
                _Float16 *const dst =
                    lds_x +
                    tok_local * ROCM_Q4_WMMA_K128_LDS_PITCH + kk;
                *reinterpret_cast<half2 *>(dst) = value01;
                *reinterpret_cast<half2 *>(dst + 2u) = value23;
            }
            __syncthreads();

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

#pragma unroll 1
                for (uint32_t nibble = 0u; nibble < 2u; nibble++) {
                    const uint32_t qgroup = qpair * 2u + nibble;
                    uint8_t scale = 0u;
                    uint8_t minimum = 0u;
                    dev_q4_K_get_scale_min(
                        qgroup, block->scales, &scale, &minimum);
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
            if (ds4_rocm_q4_wmma_load::needs_reuse_barrier(
                    block_index, q4_blocks, qpair_base / 2u, 2u)) {
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
#undef DS4_ROCM_Q4_GFX1151_WMMA_ROWTILE_DEVICE
#endif

#include "ds4_rocm_q4_dot.cuh"

/* K=1024 has exactly four Q8_K blocks.  The generic TILE8 kernel leaves half
 * of each eight-lane row group idle and still reserves LDS for eight blocks.
 * Four-lane groups preserve the legacy block/reduction order while doubling
 * the rows produced by a 256-thread workgroup and halving the LDS footprint
 * to 8 tokens * 4 K blocks * 292 bytes = 9,344 bytes. */
__device__ __forceinline__ static float
rocm_q4_K_lane4_sum_f32(float v) {
    /* Build the active-lane mask relative to the physical wave.  A 32-bit
     * mask repeats lanes 0..31 for the upper half of an AMD wave64 and
     * violates HIP's __shfl_down_sync contract even though width=4 keeps the
     * data exchange inside the intended subgroup. */
    const uint32_t wave_lane = threadIdx.x & (warpSize - 1u);
    const MASK_T mask = static_cast<MASK_T>(0x0fu) << (wave_lane & ~3u);
    v += __shfl_down_sync(mask, v, 2, 4);
    v += __shfl_down_sync(mask, v, 1, 4);
    return v;
}

template<bool STREAM_LDS, bool VECTOR_LDS = false>
__global__ static void rocm_matmul_q4_K_prefill_k1024_tile4_kernel(
        float *out,
        const char *w_base,
        const cuda_block_q8_K *xq,
        uint64_t row_bytes,
        uint32_t out_dim,
        uint32_t n_tok) {
    __shared__ __align__(VECTOR_LDS ? 16 : alignof(cuda_block_q8_K))
        cuda_block_q8_K sxq[ROCM_Q4_PREFILL_TOKEN_TILE]
                                         [ROCM_Q4_PREFILL_K1024_KBLOCK_TILE];

    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 3u;
    const uint32_t row_lane = tid >> 2u;
    const uint32_t row = blockIdx.x * ROCM_Q4_PREFILL_K1024_ROWS + row_lane;
    const uint32_t tok0 = blockIdx.y * ROCM_Q4_PREFILL_TOKEN_TILE;
    const uint32_t nt = n_tok - tok0 < ROCM_Q4_PREFILL_TOKEN_TILE
                      ? n_tok - tok0 : ROCM_Q4_PREFILL_TOKEN_TILE;
    const bool row_valid = row < out_dim;
    const cuda_block_q4_K *wr = row_valid
        ? reinterpret_cast<const cuda_block_q4_K *>(
              w_base + (uint64_t)row * row_bytes)
        : NULL;
    float acc[ROCM_Q4_PREFILL_TOKEN_TILE] = {
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    };

    /* The complete K dimension fits one LDS tile.  Flatten the copy so
     * neighboring threads read consecutive words across token/block rows. */
    if (STREAM_LDS) {
        ds4_rocm_q4_lds::copy_thread_selected<ROCM_Q4_PREFILL_K1024_KBLOCK_TILE, VECTOR_LDS>(
            reinterpret_cast<uint32_t *>(sxq),
            reinterpret_cast<const uint32_t *>(xq + (uint64_t)tok0 * 4u),
            tid, blockDim.x, nt, 4u, 4u * ROCM_Q4_Q8K_WORDS);
    } else {
        const uint32_t tile_words = nt * ROCM_Q4_PREFILL_K1024_KBLOCK_TILE *
                                    ROCM_Q4_Q8K_WORDS;
        uint32_t *const sxq_words = reinterpret_cast<uint32_t *>(sxq);
        for (uint32_t i = tid; i < tile_words; i += blockDim.x) {
            const uint32_t block_slot = i / ROCM_Q4_Q8K_WORDS;
            const uint32_t word = i - block_slot * ROCM_Q4_Q8K_WORDS;
            const uint32_t p = block_slot >> 2u;
            const uint32_t bb = block_slot & 3u;
            const uint32_t *const src_words =
                reinterpret_cast<const uint32_t *>(
                    xq + (uint64_t)(tok0 + p) *
                              ROCM_Q4_PREFILL_K1024_KBLOCK_TILE + bb);
            sxq_words[i] = src_words[word];
        }
    }
    __syncthreads();

    if (row_valid) {
        rocm_dot_q4_K_q8_K_block8_reuse_weights(
            wr + lane,
            sxq[0] + lane, sxq[1] + lane,
            sxq[2] + lane, sxq[3] + lane,
            sxq[4] + lane, sxq[5] + lane,
            sxq[6] + lane, sxq[7] + lane,
            nt, acc);

        #pragma unroll
        for (uint32_t p = 0u; p < ROCM_Q4_PREFILL_TOKEN_TILE; p++) {
            if (p < nt) {
                const float v = rocm_q4_K_lane4_sum_f32(acc[p]);
                if (lane == 0u) {
                    out[(uint64_t)(tok0 + p) * out_dim + row] = v;
                }
            }
        }
    }
}

template<bool STREAM_LDS, bool VECTOR_LDS = false, bool ALIGNED_LDS = false>
__global__ static void rocm_matmul_q4_K_prefill_tile8_strided_kernel(
        float *out,
        const char *w_base,
        const cuda_block_q8_K *xq,
        uint64_t row_bytes,
        uint32_t xq_blocks,
        uint32_t out_dim,
        uint32_t n_tok,
        uint64_t xq_token_stride,
        uint64_t out_token_stride) {
    using Q8Block = typename std::conditional<ALIGNED_LDS,
        ds4_rocm_q4_lds::aligned_q8_K, cuda_block_q8_K>::type;
    __shared__ __align__((VECTOR_LDS || ALIGNED_LDS) ? 16 : alignof(cuda_block_q8_K))
        Q8Block sxq[ROCM_Q4_PREFILL_TOKEN_TILE]
                                         [ROCM_Q4_PREFILL_KBLOCK_TILE];

    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 7u;
    const uint32_t row_lane = tid >> 3u;
    const uint32_t row = blockIdx.x * 32u + row_lane;
    const uint32_t tok0 = blockIdx.y * ROCM_Q4_PREFILL_TOKEN_TILE;
    const uint32_t group = blockIdx.z;
    const uint32_t nt = n_tok - tok0 < ROCM_Q4_PREFILL_TOKEN_TILE
                      ? n_tok - tok0 : ROCM_Q4_PREFILL_TOKEN_TILE;
    const bool row_valid = row < out_dim;
    const cuda_block_q4_K *wr = row_valid
        ? reinterpret_cast<const cuda_block_q4_K *>(
              w_base + ((uint64_t)group * out_dim + row) * row_bytes)
        : NULL;
    float acc[ROCM_Q4_PREFILL_TOKEN_TILE] = {
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    };

    for (uint32_t b0 = 0u; b0 < xq_blocks;
         b0 += ROCM_Q4_PREFILL_KBLOCK_TILE) {
        const uint32_t nb = xq_blocks - b0 < ROCM_Q4_PREFILL_KBLOCK_TILE
                          ? xq_blocks - b0
                          : ROCM_Q4_PREFILL_KBLOCK_TILE;

        /* Copy consecutive 32-bit words cooperatively.  Assigning one 292-B
         * struct per lane makes adjacent lanes issue 292-B-strided global
         * loads; flattening the packed blocks gives the memory coalescer long
         * contiguous runs while preserving the fixed eight-block LDS layout. */
        if constexpr (ALIGNED_LDS) {
            const uint64_t src_block = (uint64_t)tok0 * xq_token_stride +
                (uint64_t)group * xq_blocks + b0;
            ds4_rocm_q4_lds::copy_thread_aligned<ROCM_Q4_PREFILL_KBLOCK_TILE>(
                reinterpret_cast<uint32_t *>(sxq),
                reinterpret_cast<const uint32_t *>(xq + src_block),
                tid, blockDim.x, nt, nb,
                (uint64_t)xq_token_stride * ROCM_Q4_Q8K_WORDS);
        } else if (STREAM_LDS) {
            const uint64_t src_block = (uint64_t)tok0 * xq_token_stride +
                (uint64_t)group * xq_blocks + b0;
            ds4_rocm_q4_lds::copy_thread_selected<ROCM_Q4_PREFILL_KBLOCK_TILE, VECTOR_LDS>(
                reinterpret_cast<uint32_t *>(sxq),
                reinterpret_cast<const uint32_t *>(xq + src_block),
                tid, blockDim.x, nt, nb,
                (uint64_t)xq_token_stride * ROCM_Q4_Q8K_WORDS);
        } else {
            const uint32_t tile_words = nt * ROCM_Q4_PREFILL_KBLOCK_TILE *
                                        ROCM_Q4_Q8K_WORDS;
            uint32_t *const sxq_words = reinterpret_cast<uint32_t *>(sxq);
            for (uint32_t i = tid; i < tile_words; i += blockDim.x) {
                const uint32_t block_slot = i / ROCM_Q4_Q8K_WORDS;
                const uint32_t word = i - block_slot * ROCM_Q4_Q8K_WORDS;
                const uint32_t p = block_slot >> 3u;
                const uint32_t bb = block_slot & 7u;
                if (bb < nb) {
                    const uint64_t src_block =
                        (uint64_t)(tok0 + p) * xq_token_stride +
                        (uint64_t)group * xq_blocks + b0 + bb;
                    const uint32_t *const src_words =
                        reinterpret_cast<const uint32_t *>(xq + src_block);
                    sxq_words[i] = src_words[word];
                }
            }
        }
        __syncthreads();

        if (row_valid && lane < nb) {
            rocm_dot_q4_K_q8_K_block8_reuse_weights(
                wr + b0 + lane,
                sxq[0] + lane, sxq[1] + lane,
                sxq[2] + lane, sxq[3] + lane,
                sxq[4] + lane, sxq[5] + lane,
                sxq[6] + lane, sxq[7] + lane,
                nt, acc);
        }
        if (!STREAM_LDS || ds4_rocm_q4_lds::needs_reuse_barrier(b0, xq_blocks))
            __syncthreads();
    }

    if (row_valid) {
        #pragma unroll
        for (uint32_t p = 0u; p < ROCM_Q4_PREFILL_TOKEN_TILE; p++) {
            if (p < nt) {
                const float v = quarter_warp_sum_f32(acc[p], lane);
                if (lane == 0u) {
                    out[(uint64_t)(tok0 + p) * out_token_stride +
                        (uint64_t)group * out_dim + row] = v;
                }
            }
        }
    }
}

/* Two independent dense projections over the same activation tile.  The
 * row-tile ranges are concatenated in grid.x, so Q/KV prefill shares both
 * the Q8_K quantization and a single launch without padding the smaller
 * projection up to the larger one's row count.  Each workgroup still handles
 * only one weight matrix: this preserves the standalone TILE8 block walk and
 * its accumulation order while removing the second host launch. */
template<bool STREAM_LDS, bool VECTOR_LDS = false>
__global__ static void rocm_matmul_q4_K_prefill_tile8_pair_kernel(
        float *out0,
        float *out1,
        const char *w0,
        const char *w1,
        const cuda_block_q8_K *xq,
        uint64_t row_bytes,
        uint32_t xq_blocks,
        uint32_t out0_dim,
        uint32_t out1_dim,
        uint32_t n_tok) {
    __shared__ __align__(VECTOR_LDS ? 16 : alignof(cuda_block_q8_K))
        cuda_block_q8_K sxq[ROCM_Q4_PREFILL_TOKEN_TILE]
                                         [ROCM_Q4_PREFILL_KBLOCK_TILE];

    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 7u;
    const uint32_t row_lane = tid >> 3u;
    const uint32_t out0_tiles = (out0_dim - 1u) / 32u + 1u;
    const bool second = blockIdx.x >= out0_tiles;
    const uint32_t row_tile = second ? blockIdx.x - out0_tiles : blockIdx.x;
    const uint32_t row = row_tile * 32u + row_lane;
    const uint32_t tok0 = blockIdx.y * ROCM_Q4_PREFILL_TOKEN_TILE;
    const uint32_t out_dim = second ? out1_dim : out0_dim;
    float *const out = second ? out1 : out0;
    const char *const w_base = second ? w1 : w0;
    const uint32_t nt = n_tok - tok0 < ROCM_Q4_PREFILL_TOKEN_TILE
                      ? n_tok - tok0 : ROCM_Q4_PREFILL_TOKEN_TILE;
    const bool row_valid = row < out_dim;
    const cuda_block_q4_K *wr = row_valid
        ? reinterpret_cast<const cuda_block_q4_K *>(
              w_base + (uint64_t)row * row_bytes)
        : NULL;
    float acc[ROCM_Q4_PREFILL_TOKEN_TILE] = {
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    };

    for (uint32_t b0 = 0u; b0 < xq_blocks;
         b0 += ROCM_Q4_PREFILL_KBLOCK_TILE) {
        const uint32_t nb = xq_blocks - b0 < ROCM_Q4_PREFILL_KBLOCK_TILE
                          ? xq_blocks - b0
                          : ROCM_Q4_PREFILL_KBLOCK_TILE;
        if (STREAM_LDS) {
            const uint64_t src_block = (uint64_t)tok0 * xq_blocks +
                b0;
            ds4_rocm_q4_lds::copy_thread_selected<ROCM_Q4_PREFILL_KBLOCK_TILE, VECTOR_LDS>(
                reinterpret_cast<uint32_t *>(sxq),
                reinterpret_cast<const uint32_t *>(xq + src_block),
                tid, blockDim.x, nt, nb,
                (uint64_t)xq_blocks * ROCM_Q4_Q8K_WORDS);
        } else {
            const uint32_t tile_words = nt * ROCM_Q4_PREFILL_KBLOCK_TILE *
                                        ROCM_Q4_Q8K_WORDS;
            uint32_t *const sxq_words = reinterpret_cast<uint32_t *>(sxq);
            for (uint32_t i = tid; i < tile_words; i += blockDim.x) {
                const uint32_t block_slot = i / ROCM_Q4_Q8K_WORDS;
                const uint32_t word = i - block_slot * ROCM_Q4_Q8K_WORDS;
                const uint32_t p = block_slot >> 3u;
                const uint32_t bb = block_slot & 7u;
                if (bb < nb) {
                    const uint64_t src_block =
                        (uint64_t)(tok0 + p) * xq_blocks + b0 + bb;
                    const uint32_t *const src_words =
                        reinterpret_cast<const uint32_t *>(xq + src_block);
                    sxq_words[i] = src_words[word];
                }
            }
        }
        __syncthreads();

        if (row_valid && lane < nb) {
            rocm_dot_q4_K_q8_K_block8_reuse_weights(
                wr + b0 + lane,
                sxq[0] + lane, sxq[1] + lane,
                sxq[2] + lane, sxq[3] + lane,
                sxq[4] + lane, sxq[5] + lane,
                sxq[6] + lane, sxq[7] + lane,
                nt, acc);
        }
        if (!STREAM_LDS || ds4_rocm_q4_lds::needs_reuse_barrier(b0, xq_blocks))
            __syncthreads();
    }

    if (row_valid) {
        #pragma unroll
        for (uint32_t p = 0u; p < ROCM_Q4_PREFILL_TOKEN_TILE; p++) {
            if (p < nt) {
                const float v = quarter_warp_sum_f32(acc[p], lane);
                if (lane == 0u) {
                    out[(uint64_t)(tok0 + p) * out_dim + row] = v;
                }
            }
        }
    }
}

// Thread-local launch evidence for the model-free public-runtime oracle.
static thread_local uint64_t g_rocm_q4_prefill_lds_stream_launches;
extern "C" void ds4_rocm_test_q4_prefill_lds_stream_reset(void) {
    g_rocm_q4_prefill_lds_stream_launches = 0;
}
extern "C" uint64_t ds4_rocm_test_q4_prefill_lds_stream_get_calls(void) {
    return g_rocm_q4_prefill_lds_stream_launches;
}
static int rocm_q4_K_prefill_lds_stream_enabled(void) {
    return getenv("DS4_ROCM_DISABLE_Q4_PREFILL_LDS_STREAM") == NULL;
}

// Counts vector-capable launches, not actual vector instructions: a strided
// or misaligned activation tile can still use the scalar device fallback.
static thread_local uint64_t g_rocm_q4_prefill_lds_vector_launches;
extern "C" void ds4_rocm_test_q4_prefill_lds_vector_reset(void) {
    g_rocm_q4_prefill_lds_vector_launches = 0;
}
extern "C" uint64_t ds4_rocm_test_q4_prefill_lds_vector_get_calls(void) {
    return g_rocm_q4_prefill_lds_vector_launches;
}
static int rocm_q4_K_prefill_lds_vector_enabled(void) {
    return getenv("DS4_ROCM_DISABLE_Q4_PREFILL_LDS_VECTOR") == NULL;
}

// Successful-enqueue evidence, not GPU completion or timing. Atomic because
// the existing exit-time profile can run on a different host thread.
static uint64_t g_rocm_q4_prefill_lds_aligned_launches;
extern "C" void ds4_rocm_test_q4_prefill_lds_aligned_reset(void) {
    __atomic_store_n(&g_rocm_q4_prefill_lds_aligned_launches, 0u, __ATOMIC_RELAXED);
}
extern "C" uint64_t ds4_rocm_test_q4_prefill_lds_aligned_get_calls(void) {
    return __atomic_load_n(&g_rocm_q4_prefill_lds_aligned_launches, __ATOMIC_RELAXED);
}

// Independent rollbacks retain scalar streaming and the original schedule.
template<typename... Args>
static void rocm_q4_K_prefill_k1024_tile4_launch(dim3 grid, Args... args) {
    if (rocm_q4_K_prefill_lds_stream_enabled()) {
        if (rocm_q4_K_prefill_lds_vector_enabled()) {
            rocm_matmul_q4_K_prefill_k1024_tile4_kernel<true, true><<<grid, 256>>>(args...);
            ++g_rocm_q4_prefill_lds_vector_launches;
        } else {
            rocm_matmul_q4_K_prefill_k1024_tile4_kernel<true><<<grid, 256>>>(args...);
        }
        ++g_rocm_q4_prefill_lds_stream_launches;
    } else {
        rocm_matmul_q4_K_prefill_k1024_tile4_kernel<false><<<grid, 256>>>(args...);
    }
}

static void rocm_q4_K_prefill_tile8_strided_launch(
        dim3 grid, float *out, const char *w, const cuda_block_q8_K *xq,
        uint64_t row_bytes, uint32_t blocks, uint32_t out_dim, uint32_t n_tok,
        uint64_t xq_token_stride, uint64_t out_token_stride) {
    // Output B stays on exact Q8_K/TILE8 even when output A uses WMMA.
    // Keep this change inside its resident gfx1151 shape; explicit older
    // LDS rollbacks retain their original kernels for independent A/B tests.
    if (ds4_rocm_q4_lds::aligned_scope(
            blocks, out_dim, n_tok, grid.z, g_ssd_streaming_mode, g_quality_mode,
            true, getenv("DS4_ROCM_DISABLE_Q4_PREFILL_LDS_ALIGNED") != NULL) &&
        xq_token_stride == blocks && out_token_stride == out_dim &&
        rocm_q4_K_prefill_lds_stream_enabled() &&
        rocm_q4_K_prefill_lds_vector_enabled() &&
        rocm_attention_runtime_is_gfx1151_wave32(0)) {
        rocm_matmul_q4_K_prefill_tile8_strided_kernel<true, false, true><<<grid, 256>>>(
            out, w, xq, row_bytes, blocks, out_dim, n_tok,
            xq_token_stride, out_token_stride);
        if (hipPeekAtLastError() == hipSuccess)
            __atomic_fetch_add(&g_rocm_q4_prefill_lds_aligned_launches, 1u, __ATOMIC_RELAXED);
        return;
    }
    if (rocm_q4_K_prefill_lds_stream_enabled()) {
        if (rocm_q4_K_prefill_lds_vector_enabled()) {
            rocm_matmul_q4_K_prefill_tile8_strided_kernel<true, true><<<grid, 256>>>(
                out, w, xq, row_bytes, blocks, out_dim, n_tok,
                xq_token_stride, out_token_stride);
            ++g_rocm_q4_prefill_lds_vector_launches;
        } else {
            rocm_matmul_q4_K_prefill_tile8_strided_kernel<true><<<grid, 256>>>(
                out, w, xq, row_bytes, blocks, out_dim, n_tok,
                xq_token_stride, out_token_stride);
        }
        ++g_rocm_q4_prefill_lds_stream_launches;
    } else {
        rocm_matmul_q4_K_prefill_tile8_strided_kernel<false><<<grid, 256>>>(
            out, w, xq, row_bytes, blocks, out_dim, n_tok,
            xq_token_stride, out_token_stride);
    }
}

template<typename... Args>
static void rocm_q4_K_prefill_tile8_pair_launch(dim3 grid, Args... args) {
    if (rocm_q4_K_prefill_lds_stream_enabled()) {
        if (rocm_q4_K_prefill_lds_vector_enabled()) {
            rocm_matmul_q4_K_prefill_tile8_pair_kernel<true, true><<<grid, 256>>>(args...);
            ++g_rocm_q4_prefill_lds_vector_launches;
        } else {
            rocm_matmul_q4_K_prefill_tile8_pair_kernel<true><<<grid, 256>>>(args...);
        }
        ++g_rocm_q4_prefill_lds_stream_launches;
    } else {
        rocm_matmul_q4_K_prefill_tile8_pair_kernel<false><<<grid, 256>>>(args...);
    }
}

static int rocm_q4_K_dense_validate(
        const ds4_gpu_tensor *out,
        const void *model_map,
        uint64_t model_size,
        uint64_t weight_offset,
        uint64_t in_dim,
        uint64_t out_dim,
        const ds4_gpu_tensor *x,
        uint64_t n_tok,
        uint64_t *blocks_out,
        uint64_t *row_bytes_out,
        uint64_t *weight_bytes_out) {
    if (!out || !x || !model_map || !blocks_out || !row_bytes_out ||
        !weight_bytes_out || in_dim == 0u || out_dim == 0u || n_tok == 0u ||
        in_dim > UINT32_MAX || out_dim > UINT32_MAX || n_tok > UINT32_MAX ||
        (in_dim % CUDA_QK_K) != 0u) {
        return 0;
    }

    const uint64_t blocks = in_dim / CUDA_QK_K;
    uint64_t row_bytes = 0;
    uint64_t weight_bytes = 0;
    if (blocks == 0u ||
        !cuda_u64_mul_checked(blocks, sizeof(cuda_block_q4_K), &row_bytes) ||
        !cuda_u64_mul_checked(out_dim, row_bytes, &weight_bytes) ||
        !cuda_model_range_fits(model_size, weight_offset, weight_bytes) ||
        !cuda_tensor_has_elems2(x, n_tok, in_dim, sizeof(float)) ||
        !cuda_tensor_has_elems2(out, n_tok, out_dim, sizeof(float))) {
        return 0;
    }

    *blocks_out = blocks;
    *row_bytes_out = row_bytes;
    *weight_bytes_out = weight_bytes;
    return 1;
}

static cuda_block_q8_K *rocm_q4_K_prequant_alloc(
        uint64_t n_tok,
        uint64_t blocks,
        const char *what) {
    uint64_t bytes = 0;
    if (!cuda_u64_mul3_checked(n_tok, blocks,
                               sizeof(cuda_block_q8_K), &bytes)) {
        return NULL;
    }
    return reinterpret_cast<cuda_block_q8_K *>(cuda_tmp_alloc(bytes, what));
}

static int rocm_q4_K_byte_ranges_overlap(
        const void *ptr0, uint64_t bytes0,
        const void *ptr1, uint64_t bytes1) {
    const uintptr_t p0 = reinterpret_cast<uintptr_t>(ptr0);
    const uintptr_t p1 = reinterpret_cast<uintptr_t>(ptr1);
    return p0 <= p1 ? (uint64_t)(p1 - p0) < bytes0
                    : (uint64_t)(p0 - p1) < bytes1;
}

static int rocm_q4_K_dense_pair_requested(void) {
    return getenv("DS4_ROCM_ENABLE_Q4_DENSE_PAIR") != NULL &&
           getenv("DS4_ROCM_DISABLE_Q4_DENSE_PAIR") == NULL;
}

enum {
    ROCM_Q4_GROUPED_ATTN_A_DEFAULT_K = 4096u,
    ROCM_Q4_GROUPED_ATTN_A_DEFAULT_M = 1024u,
    ROCM_Q4_GROUPED_ATTN_A_DEFAULT_GROUPS = 8u,
};

static int rocm_q4_K_grouped_attn_a_resident_default_scope(
        uint64_t group_dim,
        uint64_t rank,
        uint32_t group0,
        uint32_t group_cnt,
        int resident_decode) {
    /* A batch fallback can pass one row at a time through this same API, so
     * the caller explicitly identifies true decode.  Default only the
     * production decode shape after the complete model has been made
     * resident; explicit ENABLE keeps the existing experimental surface. */
    return resident_decode &&
           !g_ssd_streaming_mode &&
           group_dim == ROCM_Q4_GROUPED_ATTN_A_DEFAULT_K &&
           rank == ROCM_Q4_GROUPED_ATTN_A_DEFAULT_M &&
           group0 == 0u &&
           group_cnt == ROCM_Q4_GROUPED_ATTN_A_DEFAULT_GROUPS;
}

static int rocm_q4_K_prefill_tile8_scope(uint64_t n_tok) {
    /* Keep decode/speculative micro-batches on the latency-oriented legacy
     * kernel.  4096 is DS4's largest supported prefill chunk and bounds the
     * validated tiled-prefill surface. */
    return n_tok > 8u && n_tok <= 4096u;
}

static int rocm_q4_K_prefill_tile8_requested(void) {
    /* TILE8 is the ROCm Q4 prefill default.  Keep the old ENABLE variable
     * harmlessly compatible and retain one authoritative rollback switch. */
    return getenv("DS4_ROCM_DISABLE_Q4_PREFILL_TILE8") == NULL;
}

static int rocm_q4_K_prefill_tile8_required(void) {
    return getenv("DS4_ROCM_REQUIRE_Q4_PREFILL_TILE8") != NULL;
}

enum {
    ROCM_Q4_PREFILL_Q8_WAVE32_REQUIRED_FAILURE = -1,
    ROCM_Q4_PREFILL_Q8_WAVE32_FALLBACK = 0,
    ROCM_Q4_PREFILL_Q8_WAVE32_USE = 1,
};

static uint64_t g_rocm_q4_prefill_q8_wave32_launches;

extern "C" void ds4_rocm_test_q4_prefill_q8_wave32_reset(void) {
    __atomic_store_n(&g_rocm_q4_prefill_q8_wave32_launches, 0u,
                     __ATOMIC_RELAXED);
}

extern "C" uint64_t ds4_rocm_test_q4_prefill_q8_wave32_get_calls(void) {
    return __atomic_load_n(&g_rocm_q4_prefill_q8_wave32_launches,
                           __ATOMIC_RELAXED);
}

/* Pure policy kept separate from device discovery so the precedence and
 * fail-closed contract remain testable on hosts without a visible GPU.
 * REQUIRE is also an opt-in; DISABLE is authoritative. */
static int rocm_q4_K_prefill_q8_wave32_policy(
        int prefill_scope,
        int runtime_compatible,
        int enabled,
        int disabled,
        int required) {
    if (!enabled && !required) {
        return ROCM_Q4_PREFILL_Q8_WAVE32_FALLBACK;
    }
    if (disabled || !prefill_scope || !runtime_compatible) {
        return required ? ROCM_Q4_PREFILL_Q8_WAVE32_REQUIRED_FAILURE
                        : ROCM_Q4_PREFILL_Q8_WAVE32_FALLBACK;
    }
    return ROCM_Q4_PREFILL_Q8_WAVE32_USE;
}

extern "C" int ds4_rocm_test_q4_prefill_q8_wave32_policy(
        int prefill_scope,
        int runtime_compatible,
        int enabled,
        int disabled,
        int required) {
    return rocm_q4_K_prefill_q8_wave32_policy(
        prefill_scope != 0, runtime_compatible != 0, enabled != 0,
        disabled != 0, required != 0);
}

static int rocm_q4_K_prefill_q8_wave32_required(void) {
    return rocm_q4_attn_q_b_env_bool(
        "DS4_ROCM_REQUIRE_Q4_PREFILL_Q8_K_WAVE32") == 1;
}

static int rocm_q4_K_prefill_q8_wave32_select(uint64_t n_tok) {
    const int enabled = rocm_q4_attn_q_b_env_bool(
        "DS4_ROCM_ENABLE_Q4_PREFILL_Q8_K_WAVE32") == 1;
    const int disabled = rocm_q4_attn_q_b_env_bool(
        "DS4_ROCM_DISABLE_Q4_PREFILL_Q8_K_WAVE32") == 1;
    const int required = rocm_q4_K_prefill_q8_wave32_required();
    const int prefill_scope = rocm_q4_K_prefill_tile8_scope(n_tok);
    const int requested = enabled || required;
    const int runtime_compatible = requested && !disabled && prefill_scope
        ? rocm_attention_runtime_is_gfx1151_wave32(0)
        : 0;
    const int decision = rocm_q4_K_prefill_q8_wave32_policy(
        prefill_scope, runtime_compatible, enabled, disabled, required);
    if (decision == ROCM_Q4_PREFILL_Q8_WAVE32_REQUIRED_FAILURE) {
        fprintf(stderr,
                DS4_GPU_LOG_PREFIX
                "required Q4 prefill Q8_K wave32 quantizer is unavailable "
                "(N=%llu scope=%d compatible=%d disabled=%d)\n",
                (unsigned long long)n_tok, prefill_scope,
                runtime_compatible, disabled);
    }
    return decision;
}

static int rocm_q4_K_q8_quantize_launch(
        cuda_block_q8_K *out,
        const float *x,
        uint32_t in_dim,
        uint32_t n_rows,
        int q8_wave32,
        int q8_wave32_required,
        const char *label) {
    const uint32_t blocks = in_dim / CUDA_QK_K;
    if (q8_wave32 == ROCM_Q4_PREFILL_Q8_WAVE32_USE) {
        /* Re-read the active device immediately before enqueue.  Selection
         * may have happened before range resolution/allocation, and HIP's
         * active device is thread-local.  Optional use falls back safely;
         * REQUIRE cannot silently enqueue either the wrong kernel or the
         * canonical one. */
        if (!rocm_attention_runtime_is_gfx1151_wave32(0)) {
            if (q8_wave32_required) {
                fprintf(stderr,
                        DS4_GPU_LOG_PREFIX
                        "required Q8_K wave32 quantizer lost its gfx1151 "
                        "wave32 device before enqueue\n");
                return 0;
            }
            q8_wave32 = ROCM_Q4_PREFILL_Q8_WAVE32_FALLBACK;
        }
    }
    if (q8_wave32 == ROCM_Q4_PREFILL_Q8_WAVE32_USE) {
        const dim3 grid(
            (unsigned)((blocks + ROCM_Q8_K_WAVE32_WAVES_PER_BLOCK - 1u) /
                       ROCM_Q8_K_WAVE32_WAVES_PER_BLOCK),
            n_rows, 1u);
        q8_K_quantize_wave32_kernel<<<
            grid, ROCM_Q8_K_WAVE32_BLOCK_THREADS>>>(
                out, x, in_dim, n_rows);
        const int ok = cuda_ok(cudaGetLastError(), label);
        if (ok) {
            __atomic_fetch_add(&g_rocm_q4_prefill_q8_wave32_launches, 1u,
                               __ATOMIC_RELAXED);
        }
        return ok;
    }

    const dim3 grid((unsigned)blocks, n_rows, 1u);
    q8_K_quantize_kernel<<<grid, 256u>>>(out, x, in_dim, n_rows);
    return cuda_ok(cudaGetLastError(), label);
}

/* Raw-layout oracle used only by the ROCm Q4 parity test.  It lets the test
 * compare every Q8_K byte instead of relying solely on downstream dot
 * products to expose a quantizer mismatch. */
extern "C" int ds4_rocm_test_q8_K_quantize_tensor(
        ds4_gpu_tensor *out,
        const ds4_gpu_tensor *x,
        uint32_t in_dim,
        uint32_t n_rows,
        int use_wave32) {
    if (!out || !x || !out->ptr || !x->ptr || in_dim == 0u || n_rows == 0u ||
        (in_dim % CUDA_QK_K) != 0u) {
        return 0;
    }
    uint64_t x_bytes = 0;
    uint64_t out_bytes = 0;
    if (!cuda_u64_mul3_checked(n_rows, in_dim, sizeof(float), &x_bytes) ||
        !cuda_u64_mul3_checked(n_rows, in_dim / CUDA_QK_K,
                               sizeof(cuda_block_q8_K), &out_bytes) ||
        x->bytes < x_bytes || out->bytes < out_bytes) {
        return 0;
    }
    const int decision = use_wave32
        ? (rocm_attention_runtime_is_gfx1151_wave32(0)
               ? ROCM_Q4_PREFILL_Q8_WAVE32_USE
               : ROCM_Q4_PREFILL_Q8_WAVE32_REQUIRED_FAILURE)
        : ROCM_Q4_PREFILL_Q8_WAVE32_FALLBACK;
    if (decision == ROCM_Q4_PREFILL_Q8_WAVE32_REQUIRED_FAILURE) return 0;
    return rocm_q4_K_q8_quantize_launch(
        reinterpret_cast<cuda_block_q8_K *>(out->ptr),
        reinterpret_cast<const float *>(x->ptr), in_dim, n_rows, decision,
        use_wave32 != 0,
        use_wave32 ? "Q8_K wave32 raw oracle launch"
                   : "Q8_K canonical raw oracle launch");
}

/* Benchmark-only enqueue hook.  The resident ROCm harness validates the
 * active gfx1151/wave32 device, dimensions, allocations and output guards
 * before entering its HIP-event interval.  Keep this entry deliberately free
 * of device queries, allocation, environment parsing and cudaGetLastError so
 * the measured interval contains only the selected quantizer dispatch.  The
 * following end event/synchronization reports any asynchronous failure. */
extern "C" void ds4_rocm_bench_q8_K_quantize_enqueue(
        void *out,
        const void *x,
        uint32_t in_dim,
        uint32_t n_rows,
        int use_wave32) {
    const uint32_t blocks = in_dim / CUDA_QK_K;
    if (use_wave32) {
        const dim3 grid(
            (unsigned)((blocks + ROCM_Q8_K_WAVE32_WAVES_PER_BLOCK - 1u) /
                       ROCM_Q8_K_WAVE32_WAVES_PER_BLOCK),
            n_rows, 1u);
        q8_K_quantize_wave32_kernel<<<
            grid, ROCM_Q8_K_WAVE32_BLOCK_THREADS>>>(
                reinterpret_cast<cuda_block_q8_K *>(out),
                reinterpret_cast<const float *>(x), in_dim, n_rows);
        return;
    }

    const dim3 grid((unsigned)blocks, n_rows, 1u);
    q8_K_quantize_kernel<<<grid, 256u>>>(
        reinterpret_cast<cuda_block_q8_K *>(out),
        reinterpret_cast<const float *>(x), in_dim, n_rows);
}

enum {
    ROCM_Q4_PREFILL_WMMA_REQUIRED_FAILURE = -1,
    ROCM_Q4_PREFILL_WMMA_FALLBACK = 0,
    ROCM_Q4_PREFILL_WMMA_USE = 1,
};

/* Test oracle for strict dispatch: REQUIRE must attest this launch wrapper,
 * not merely produce output through a canonical fallback. */
static uint64_t g_rocm_q4_prefill_wmma_launches;
static uint64_t g_rocm_q4_prefill_wmma_k64_launches;
static uint64_t g_rocm_q4_prefill_wmma_k64_load4_launches;
static uint64_t g_rocm_q4_prefill_wmma_k128_launches;
static int g_rocm_q4_prefill_tile8_report_registered;

static void rocm_q4_K_prefill_tile8_report(void);

static void rocm_q4_K_prefill_stats_register(void) {
    if (getenv("DS4_ROCM_Q4_PREFILL_TILE8_STATS") == NULL) return;
    if (__atomic_exchange_n(&g_rocm_q4_prefill_tile8_report_registered, 1,
                            __ATOMIC_ACQ_REL) == 0) {
        (void)atexit(rocm_q4_K_prefill_tile8_report);
    }
}

extern "C" void ds4_rocm_test_q4_prefill_wmma_reset(void) {
    __atomic_store_n(&g_rocm_q4_prefill_wmma_launches, 0u,
                     __ATOMIC_RELAXED);
    __atomic_store_n(&g_rocm_q4_prefill_wmma_k64_launches, 0u,
                     __ATOMIC_RELAXED);
    __atomic_store_n(&g_rocm_q4_prefill_wmma_k64_load4_launches, 0u,
                     __ATOMIC_RELAXED);
    __atomic_store_n(&g_rocm_q4_prefill_wmma_k128_launches, 0u,
                     __ATOMIC_RELAXED);
}

extern "C" uint64_t ds4_rocm_test_q4_prefill_wmma_get_calls(void) {
    return __atomic_load_n(&g_rocm_q4_prefill_wmma_launches,
                           __ATOMIC_RELAXED);
}

extern "C" uint64_t ds4_rocm_test_q4_prefill_wmma_k64_get_calls(void) {
    return __atomic_load_n(&g_rocm_q4_prefill_wmma_k64_launches,
                           __ATOMIC_RELAXED);
}

extern "C" uint64_t ds4_rocm_test_q4_prefill_wmma_k128_get_calls(void) {
    return __atomic_load_n(&g_rocm_q4_prefill_wmma_k128_launches,
                           __ATOMIC_RELAXED);
}

extern "C" uint64_t ds4_rocm_test_q4_prefill_wmma_k64_load4_get_calls(void) {
    return __atomic_load_n(&g_rocm_q4_prefill_wmma_k64_load4_launches,
                           __ATOMIC_RELAXED);
}

static int rocm_q4_K_prefill_wmma_k64_control_policy(int control) {
    return control != 0;
}

extern "C" int ds4_rocm_test_q4_prefill_wmma_k64_control_policy(
        int control) {
    return rocm_q4_K_prefill_wmma_k64_control_policy(control);
}

/* K128 is the production default for its aligned q_b-style 256-row scope.
 * Keep a value-aware opt-out for tester rollback, layer it on top of the K64
 * control so K64=0 remains a reliable K32 rollback, and retain K64 for every
 * incompatible launch. */
static int rocm_q4_K_prefill_wmma_k128_policy(
        int disabled,
        int k64_enabled,
        uint32_t row_tile,
        int load4_compatible) {
    return disabled != 1 && k64_enabled && row_tile == 256u &&
           load4_compatible;
}

extern "C" int ds4_rocm_test_q4_prefill_wmma_k128_policy(
        int disabled,
        int k64_enabled,
        uint32_t row_tile,
        int load4_compatible) {
    return rocm_q4_K_prefill_wmma_k128_policy(
        disabled, k64_enabled, row_tile, load4_compatible);
}

/* Resident execution is automatic after validation.  Preserve an explicit
 * ENABLE=0 as a compatibility opt-out, while REQUIRE remains a strict
 * assertion rather than the switch that turns the candidate on.  SSD
 * streaming deliberately keeps its separate opt-in and residency contract. */
static int rocm_q4_K_prefill_wmma_requested_policy(
        int ssd_streaming,
        int enabled,
        int ssd_enabled,
        int disabled,
        int required) {
    if (required) return 1;
    if (disabled) return 0;
    return ssd_streaming ? ssd_enabled == 1 : enabled != 0;
}

/* Attention output is a two-stage A -> B projection.  A retains the validated
 * standalone policy: resident execution is automatic, while SSD keeps its
 * explicit gate and physical-device-residency contract. */
static int rocm_q4_K_prefill_wmma_attention_a_requested_policy(
        int ssd_streaming,
        int enabled,
        int ssd_enabled,
        int disabled,
        int required) {
    return rocm_q4_K_prefill_wmma_requested_policy(
        ssd_streaming, enabled, ssd_enabled, disabled, required);
}

/* Applying direct WMMA to both stages compounds the two F16 approximations.
 * Keep attention-output B behind REQUIRE so ordinary ENABLE A/B measurements
 * retain the exact Q8_K+TILE8 second stage.  Test REQUIRE before DISABLE so
 * DISABLE+REQUIRE reaches the normal selector and fails closed. */
static int rocm_q4_K_prefill_wmma_attention_b_requested_policy(
        int ssd_streaming,
        int enabled,
        int ssd_enabled,
        int disabled,
        int required) {
    (void)ssd_streaming;
    (void)enabled;
    (void)ssd_enabled;
    (void)disabled;
    return required != 0;
}

/* An explicitly selected exact Q8_K quantizer owns an optional WMMA default.
 * REQUIRE_WMMA is the only control that may override an optional Q8 request;
 * dual REQUIRE is rejected by each public dispatch before enqueue. */
static int rocm_q4_K_prefill_wmma_yields_to_q8_wave32(
        int q8_wave32,
        int wmma_required) {
    return q8_wave32 == ROCM_Q4_PREFILL_Q8_WAVE32_USE && !wmma_required;
}

extern "C" int ds4_rocm_test_q4_prefill_wmma_yields_to_q8_wave32(
        int q8_selected,
        int wmma_required) {
    const int q8_wave32 = q8_selected
        ? ROCM_Q4_PREFILL_Q8_WAVE32_USE
        : ROCM_Q4_PREFILL_Q8_WAVE32_FALLBACK;
    return rocm_q4_K_prefill_wmma_yields_to_q8_wave32(
        q8_wave32, wmma_required != 0);
}

extern "C" int ds4_rocm_test_q4_prefill_wmma_requested_policy(
        int ssd_streaming,
        int enabled,
        int ssd_enabled,
        int disabled,
        int required) {
    return rocm_q4_K_prefill_wmma_requested_policy(
        ssd_streaming != 0, enabled, ssd_enabled, disabled != 0,
        required != 0);
}

extern "C" int
ds4_rocm_test_q4_prefill_wmma_attention_a_requested_policy(
        int ssd_streaming,
        int enabled,
        int ssd_enabled,
        int disabled,
        int required) {
    return rocm_q4_K_prefill_wmma_attention_a_requested_policy(
        ssd_streaming != 0, enabled, ssd_enabled, disabled != 0,
        required != 0);
}

extern "C" int
ds4_rocm_test_q4_prefill_wmma_attention_b_requested_policy(
        int ssd_streaming,
        int enabled,
        int ssd_enabled,
        int disabled,
        int required) {
    return rocm_q4_K_prefill_wmma_attention_b_requested_policy(
        ssd_streaming != 0, enabled, ssd_enabled, disabled != 0,
        required != 0);
}

static int rocm_q4_K_prefill_wmma_select(
        uint64_t n_tok,
        uint64_t in_dim,
        uint64_t out_dim,
        int weight_device_resident) {
    const int enabled = rocm_q4_attn_q_b_env_bool(
        "DS4_ROCM_ENABLE_Q4_PREFILL_WMMA");
    const int ssd_enabled = rocm_q4_attn_q_b_env_bool(
        "DS4_ROCM_ENABLE_Q4_PREFILL_WMMA_SSD") == 1;
    const int disabled = rocm_q4_attn_q_b_env_bool(
        "DS4_ROCM_DISABLE_Q4_PREFILL_WMMA") == 1;
    const int required = rocm_q4_attn_q_b_env_bool(
        "DS4_ROCM_REQUIRE_Q4_PREFILL_WMMA") == 1;
    const int requested = rocm_q4_K_prefill_wmma_requested_policy(
        g_ssd_streaming_mode, enabled, ssd_enabled, disabled, required);
    if (!requested) return ROCM_Q4_PREFILL_WMMA_FALLBACK;

    const int shape_ok =
        n_tok >= 256u && n_tok <= 4096u &&
        in_dim != 0u && (in_dim % CUDA_QK_K) == 0u &&
        out_dim != 0u && in_dim <= UINT32_MAX &&
        out_dim <= UINT32_MAX && n_tok <= UINT32_MAX;
    const int storage_ok = !g_ssd_streaming_mode ||
                           ((ssd_enabled || required) &&
                            weight_device_resident);
    const int explicit_request = enabled == 1 || ssd_enabled || required;
    const int eligible = !disabled && shape_ok && storage_ok &&
        !g_quality_mode &&
        rocm_attention_runtime_is_gfx1151_wave32(explicit_request);
    if (eligible) return ROCM_Q4_PREFILL_WMMA_USE;
    if (!required) return ROCM_Q4_PREFILL_WMMA_FALLBACK;

    fprintf(stderr,
            DS4_GPU_LOG_PREFIX
            "required Q4_K prefill WMMA is unavailable "
            "(N=%llu K=%llu M=%llu disabled=%d ssd=%d ssd_opt=%d "
            "resident=%d quality=%d)\n",
            (unsigned long long)n_tok,
            (unsigned long long)in_dim,
            (unsigned long long)out_dim,
            disabled,
            g_ssd_streaming_mode ? 1 : 0,
            ssd_enabled,
            weight_device_resident,
            g_quality_mode ? 1 : 0);
    return ROCM_Q4_PREFILL_WMMA_REQUIRED_FAILURE;
}

static int rocm_q4_K_prefill_wmma_load2_compatible(
        const float *x,
        uint64_t x_token_stride,
        uint64_t x_group_stride) {
    return x && (((uintptr_t)x & 7u) == 0u) &&
           ((x_token_stride & 1u) == 0u) &&
           ((x_group_stride & 1u) == 0u);
}

static int rocm_q4_K_prefill_wmma_load4_compatible(
        const float *x,
        uint64_t x_token_stride,
        uint64_t x_group_stride) {
    return x && (((uintptr_t)x & 15u) == 0u) &&
           ((x_token_stride & 3u) == 0u) &&
           ((x_group_stride & 3u) == 0u);
}

static void rocm_q4_K_prefill_wmma_enqueue(
        float *out,
        const char *w,
        const float *x,
        uint32_t n_tok,
        uint32_t n_groups,
        uint32_t in_dim,
        uint32_t out_dim,
        uint64_t row_bytes,
        uint64_t x_token_stride,
        uint64_t x_group_stride,
        uint64_t out_token_stride,
        uint32_t row_tile,
        int load2) {
    const dim3 grid(
        (unsigned)(((uint64_t)out_dim + row_tile - 1u) / row_tile),
        (unsigned)(((uint64_t)n_tok + ROCM_Q4_WMMA_TOKEN_TILE - 1u) /
                   ROCM_Q4_WMMA_TOKEN_TILE),
        n_groups);
    if (row_tile == 256u) {
        if (load2) {
            rocm_matmul_q4_K_prefill_wmma_rowtile_strided_kernel<
                256u, 16u, 1u, true><<<grid, 512u>>>(
                    out, w, x, n_tok, n_groups, in_dim, out_dim, row_bytes,
                    x_token_stride, x_group_stride, out_token_stride);
        } else {
            rocm_matmul_q4_K_prefill_wmma_rowtile_strided_kernel<
                256u, 16u, 1u, false><<<grid, 512u>>>(
                    out, w, x, n_tok, n_groups, in_dim, out_dim, row_bytes,
                    x_token_stride, x_group_stride, out_token_stride);
        }
    } else if (row_tile == 128u) {
        if (load2) {
            rocm_matmul_q4_K_prefill_wmma_rowtile_strided_kernel<
                128u, 8u, 1u, true><<<grid, 256u>>>(
                    out, w, x, n_tok, n_groups, in_dim, out_dim, row_bytes,
                    x_token_stride, x_group_stride, out_token_stride);
        } else {
            rocm_matmul_q4_K_prefill_wmma_rowtile_strided_kernel<
                128u, 8u, 1u, false><<<grid, 256u>>>(
                    out, w, x, n_tok, n_groups, in_dim, out_dim, row_bytes,
                    x_token_stride, x_group_stride, out_token_stride);
        }
    } else {
        /* Preserve the original kernel's occupancy contract so the retained
         * 64-row benchmark arm differs only in row geometry. */
        rocm_matmul_q4_K_prefill_wmma_rowtile_strided_kernel<
            64u, 4u, 2u, false>
            <<<grid, 128u>>>(
                out, w, x, n_tok, n_groups, in_dim, out_dim, row_bytes,
                x_token_stride, x_group_stride, out_token_stride);
    }
}

/* Keep the K64 launch family separate from the established K32 entry points.
 * Besides making rollback a single environment change, this prevents compiler
 * changes caused by a template K_TILE branch from moving the A/B baseline. */
static void rocm_q4_K_prefill_wmma_k64_enqueue(
        float *out,
        const char *w,
        const float *x,
        uint32_t n_tok,
        uint32_t n_groups,
        uint32_t in_dim,
        uint32_t out_dim,
        uint64_t row_bytes,
        uint64_t x_token_stride,
        uint64_t x_group_stride,
        uint64_t out_token_stride,
        uint32_t row_tile,
        int load2,
        int load4 = 0) {
    const dim3 grid(
        (unsigned)(((uint64_t)out_dim + row_tile - 1u) / row_tile),
        (unsigned)(((uint64_t)n_tok + ROCM_Q4_WMMA_TOKEN_TILE - 1u) /
                   ROCM_Q4_WMMA_TOKEN_TILE),
        n_groups);
    if (row_tile == 256u) {
        if (load4) {
            rocm_matmul_q4_K_prefill_wmma_k64_p80_rowtile_strided_kernel<
                256u, 16u, 1u, true, true><<<grid, 512u>>>(
                    out, w, x, n_tok, n_groups, in_dim, out_dim, row_bytes,
                    x_token_stride, x_group_stride, out_token_stride);
        } else if (load2) {
            rocm_matmul_q4_K_prefill_wmma_k64_p80_rowtile_strided_kernel<
                256u, 16u, 1u, true><<<grid, 512u>>>(
                    out, w, x, n_tok, n_groups, in_dim, out_dim, row_bytes,
                    x_token_stride, x_group_stride, out_token_stride);
        } else {
            rocm_matmul_q4_K_prefill_wmma_k64_p80_rowtile_strided_kernel<
                256u, 16u, 1u, false><<<grid, 512u>>>(
                    out, w, x, n_tok, n_groups, in_dim, out_dim, row_bytes,
                    x_token_stride, x_group_stride, out_token_stride);
        }
    } else if (row_tile == 128u) {
        if (load4) {
            rocm_matmul_q4_K_prefill_wmma_k64_p80_rowtile_strided_kernel<
                128u, 8u, 1u, true, true><<<grid, 256u>>>(
                    out, w, x, n_tok, n_groups, in_dim, out_dim, row_bytes,
                    x_token_stride, x_group_stride, out_token_stride);
        } else if (load2) {
            rocm_matmul_q4_K_prefill_wmma_k64_p80_rowtile_strided_kernel<
                128u, 8u, 1u, true><<<grid, 256u>>>(
                    out, w, x, n_tok, n_groups, in_dim, out_dim, row_bytes,
                    x_token_stride, x_group_stride, out_token_stride);
        } else {
            rocm_matmul_q4_K_prefill_wmma_k64_p80_rowtile_strided_kernel<
                128u, 8u, 1u, false><<<grid, 256u>>>(
                    out, w, x, n_tok, n_groups, in_dim, out_dim, row_bytes,
                    x_token_stride, x_group_stride, out_token_stride);
        }
    } else {
        rocm_matmul_q4_K_prefill_wmma_k64_p80_rowtile_strided_kernel<
            64u, 4u, 2u, false><<<grid, 128u>>>(
                out, w, x, n_tok, n_groups, in_dim, out_dim, row_bytes,
                x_token_stride, x_group_stride, out_token_stride);
    }
}

/* The K128 path is deliberately a single q_b-oriented instantiation.
 * Keeping it out of the generic K32/K64 template family preserves the codegen
 * of both established benchmark arms. */
static void rocm_q4_K_prefill_wmma_k128_enqueue(
        float *out,
        const char *w,
        const float *x,
        uint32_t n_tok,
        uint32_t n_groups,
        uint32_t in_dim,
        uint32_t out_dim,
        uint64_t row_bytes,
        uint64_t x_token_stride,
        uint64_t x_group_stride,
        uint64_t out_token_stride) {
    const dim3 grid(
        (unsigned)(((uint64_t)out_dim + 255u) / 256u),
        (unsigned)(((uint64_t)n_tok + ROCM_Q4_WMMA_TOKEN_TILE - 1u) /
                   ROCM_Q4_WMMA_TOKEN_TILE),
        n_groups);
    rocm_matmul_q4_K_prefill_wmma_k128_p144_rowtile_strided_kernel<
        256u, 16u, 1u><<<grid, 512u>>>(
            out, w, x, n_tok, n_groups, in_dim, out_dim, row_bytes,
            x_token_stride, x_group_stride, out_token_stride);
}

static uint32_t rocm_q4_K_prefill_wmma_row_tile(uint32_t out_dim) {
    const uint32_t shape_tile = out_dim >= 8192u
        ? 256u
        : (out_dim >= 1024u ? 128u : 64u);
    const char *raw = getenv("DS4_ROCM_Q4_PREFILL_WMMA_ROW_TILE");
    if (raw) {
        while (isspace((unsigned char)*raw)) raw++;
        /* strtoull accepts a leading minus and wraps some values into the
         * whitelist.  Treat every negative spelling as an invalid unsigned
         * override so the benchmark cannot silently select the wrong arm. */
        if (*raw == '-') return shape_tile;
    }
    const uint64_t requested = rocm_q4_attn_q_b_env_u64(
        "DS4_ROCM_Q4_PREFILL_WMMA_ROW_TILE", 0u, 0u, UINT64_MAX);
    if (requested == 64u || requested == 128u || requested == 256u) {
        return (uint32_t)requested;
    }
    return shape_tile;
}

extern "C" uint32_t ds4_rocm_test_q4_prefill_wmma_row_tile(
        uint32_t out_dim) {
    return rocm_q4_K_prefill_wmma_row_tile(out_dim);
}

static int rocm_q4_K_prefill_wmma_launch(
        float *out,
        const char *w,
        const float *x,
        uint32_t n_tok,
        uint32_t n_groups,
        uint32_t in_dim,
        uint32_t out_dim,
        uint64_t row_bytes,
        uint64_t x_token_stride,
        uint64_t x_group_stride,
        uint64_t out_token_stride,
        const char *label) {
    if (!out || !w || !x || n_groups == 0u) return 0;

    /* Match the proven Q8 row geometry so a 64x32 activation tile is loaded
     * once for 128 rows, or once for 256 rows on the large q_b/output shapes.
     * Small projections retain 64 rows instead of launching mostly idle waves;
     * that instantiation is also the deterministic previous-kernel A/B control.
     * Invalid overrides deliberately fall back to shape selection. */
    const uint32_t row_tile = rocm_q4_K_prefill_wmma_row_tile(out_dim);
    const int load2 = row_tile != 64u &&
        rocm_q4_K_prefill_wmma_load2_compatible(
            x, x_token_stride, x_group_stride);
    /* K64 is the base staging mode once the normal direct-Q4 WMMA selector
     * has accepted this launch; eligible 256-row launches layer K128 on it.
     * Preserve a value-aware K32 rollback: unset/true permits the wider
     * stages and 0/false/no/off selects K32. */
    const int k64_control = rocm_q4_attn_q_b_env_bool(
        "DS4_ROCM_ENABLE_Q4_PREFILL_WMMA_K64");
    const int use_k64 =
        rocm_q4_K_prefill_wmma_k64_control_policy(k64_control);
    const int k128_disabled = rocm_q4_attn_q_b_env_bool(
        "DS4_ROCM_DISABLE_Q4_PREFILL_WMMA_K128");
    const int load4 = rocm_q4_K_prefill_wmma_load4_compatible(
        x, x_token_stride, x_group_stride);
    const int use_k128 = rocm_q4_K_prefill_wmma_k128_policy(
        k128_disabled, use_k64, row_tile, load4);
    const int use_k64_load4 = ds4_rocm_q4_wmma_load::select(
        getenv("DS4_ROCM_DISABLE_Q4_PREFILL_WMMA_K64_LOAD4") != NULL,
        use_k64, use_k128, row_tile, x, x_token_stride, x_group_stride);
    if (use_k128) {
        rocm_q4_K_prefill_wmma_k128_enqueue(
            out, w, x, n_tok, n_groups, in_dim, out_dim, row_bytes,
            x_token_stride, x_group_stride, out_token_stride);
    } else if (use_k64) {
        rocm_q4_K_prefill_wmma_k64_enqueue(
            out, w, x, n_tok, n_groups, in_dim, out_dim, row_bytes,
            x_token_stride, x_group_stride, out_token_stride, row_tile, load2,
            use_k64_load4);
    } else {
        rocm_q4_K_prefill_wmma_enqueue(
            out, w, x, n_tok, n_groups, in_dim, out_dim, row_bytes,
            x_token_stride, x_group_stride, out_token_stride, row_tile, load2);
    }
    const char *launch_label = use_k128
        ? "q4_K prefill WMMA K128/P144 rowtile launch"
        : (use_k64
               ? "q4_K prefill WMMA K64/P80 rowtile launch"
               : (label ? label : "q4_K prefill WMMA rowtile launch"));
    const int ok = cuda_ok(cudaGetLastError(), launch_label);
    if (ok) {
        rocm_q4_K_prefill_stats_register();
        __atomic_fetch_add(&g_rocm_q4_prefill_wmma_launches, 1u,
                           __ATOMIC_RELAXED);
        if (use_k128) {
            __atomic_fetch_add(&g_rocm_q4_prefill_wmma_k128_launches, 1u,
                               __ATOMIC_RELAXED);
        } else if (use_k64) {
            __atomic_fetch_add(&g_rocm_q4_prefill_wmma_k64_launches, 1u,
                               __ATOMIC_RELAXED);
            if (use_k64_load4) {
                __atomic_fetch_add(&g_rocm_q4_prefill_wmma_k64_load4_launches,
                                   1u, __ATOMIC_RELAXED);
            }
        }
    }
    return ok;
}

/* Resident, enqueue-only hooks for the ROCm microbenchmark.  Pointer lookup,
 * policy, environment parsing, and launch-error reporting stay outside HIP
 * event intervals; the explicit tile and loader arguments make each arm
 * self-attesting instead of relying on a mutable process environment. */
extern "C" const void *ds4_rocm_bench_q4_K_resident_weight_ptr(
        const void *model_map,
        uint64_t model_size,
        uint64_t weight_offset,
        uint64_t weight_bytes) {
    if (!cuda_model_range_fits(model_size, weight_offset, weight_bytes)) {
        return NULL;
    }
    return rocm_q4_attn_q_b_device_resident_source(
        model_map, weight_offset, weight_bytes);
}

extern "C" int ds4_rocm_bench_q4_K_wmma_enqueue(
        void *out,
        const void *w,
        const void *x,
        uint32_t n_tok,
        uint32_t n_groups,
        uint32_t in_dim,
        uint32_t out_dim,
        uint64_t row_bytes,
        uint64_t x_token_stride,
        uint64_t x_group_stride,
        uint64_t out_token_stride,
        uint32_t row_tile,
        int load2) {
    if (!out || !w || !x || n_tok == 0u || n_groups == 0u ||
        in_dim == 0u || out_dim == 0u ||
        (in_dim % CUDA_QK_K) != 0u ||
        (load2 != 0 && load2 != 1) ||
        (load2 && row_tile == 64u) ||
        (row_tile != 64u && row_tile != 128u && row_tile != 256u)) {
        return 0;
    }
    const uint64_t minimum_row_bytes =
        ((uint64_t)in_dim / CUDA_QK_K) * sizeof(cuda_block_q4_K);
    if (row_bytes < minimum_row_bytes ||
        (load2 && !rocm_q4_K_prefill_wmma_load2_compatible(
            reinterpret_cast<const float *>(x),
            x_token_stride, x_group_stride))) {
        return 0;
    }
    rocm_q4_K_prefill_wmma_enqueue(
        reinterpret_cast<float *>(out),
        reinterpret_cast<const char *>(w),
        reinterpret_cast<const float *>(x),
        n_tok, n_groups, in_dim, out_dim, row_bytes,
        x_token_stride, x_group_stride, out_token_stride, row_tile, load2);
    return 1;
}

/* Variant hook for a resident, same-process K32/K64 comparison.  The legacy
 * benchmark hook above remains a strict K32 wrapper so existing binaries and
 * source call sites retain their meaning. */
extern "C" int ds4_rocm_bench_q4_K_wmma_variant_enqueue(
        void *out,
        const void *w,
        const void *x,
        uint32_t n_tok,
        uint32_t n_groups,
        uint32_t in_dim,
        uint32_t out_dim,
        uint64_t row_bytes,
        uint64_t x_token_stride,
        uint64_t x_group_stride,
        uint64_t out_token_stride,
        uint32_t row_tile,
        uint32_t k_tile,
        int load2) {
    if (k_tile != ROCM_Q4_WMMA_K_TILE &&
        k_tile != ROCM_Q4_WMMA_K64_TILE) {
        return 0;
    }
    if (!out || !w || !x || n_tok == 0u || n_groups == 0u ||
        in_dim == 0u || out_dim == 0u ||
        (in_dim % CUDA_QK_K) != 0u ||
        (load2 != 0 && load2 != 1) ||
        (load2 && row_tile == 64u) ||
        (row_tile != 64u && row_tile != 128u && row_tile != 256u)) {
        return 0;
    }
    const uint64_t minimum_row_bytes =
        ((uint64_t)in_dim / CUDA_QK_K) * sizeof(cuda_block_q4_K);
    if (row_bytes < minimum_row_bytes ||
        (load2 && !rocm_q4_K_prefill_wmma_load2_compatible(
            reinterpret_cast<const float *>(x),
            x_token_stride, x_group_stride))) {
        return 0;
    }
    if (k_tile == ROCM_Q4_WMMA_K64_TILE) {
        rocm_q4_K_prefill_wmma_k64_enqueue(
            reinterpret_cast<float *>(out),
            reinterpret_cast<const char *>(w),
            reinterpret_cast<const float *>(x),
            n_tok, n_groups, in_dim, out_dim, row_bytes,
            x_token_stride, x_group_stride, out_token_stride, row_tile, load2);
    } else {
        rocm_q4_K_prefill_wmma_enqueue(
            reinterpret_cast<float *>(out),
            reinterpret_cast<const char *>(w),
            reinterpret_cast<const float *>(x),
            n_tok, n_groups, in_dim, out_dim, row_bytes,
            x_token_stride, x_group_stride, out_token_stride, row_tile, load2);
    }
    return 1;
}

/* Strict LOAD4 hook: existing K64 hooks deliberately remain LOAD2. Caller
 * owns architecture/residency checks, just as for the other raw WMMA hooks.
 * Incompatible geometry/alignment fails instead of benchmarking a fallback. */
extern "C" int ds4_rocm_bench_q4_K_wmma_k64_load4_enqueue(
        void *out, const void *w, const void *x, uint32_t n_tok,
        uint32_t n_groups, uint32_t in_dim, uint32_t out_dim,
        uint64_t row_bytes, uint64_t x_token_stride,
        uint64_t x_group_stride, uint64_t out_token_stride,
        uint32_t row_tile) {
    if (!out || !w || n_tok == 0u || n_groups == 0u ||
        in_dim == 0u || out_dim == 0u ||
        (in_dim % CUDA_QK_K) != 0u ||
        !ds4_rocm_q4_wmma_load::select(
            false, true, false, row_tile, x, x_token_stride, x_group_stride) ||
        row_bytes < ((uint64_t)in_dim / CUDA_QK_K) * sizeof(cuda_block_q4_K)) {
        return 0;
    }
    rocm_q4_K_prefill_wmma_k64_enqueue(
        reinterpret_cast<float *>(out), reinterpret_cast<const char *>(w),
        reinterpret_cast<const float *>(x), n_tok, n_groups, in_dim, out_dim,
        row_bytes, x_token_stride, x_group_stride, out_token_stride,
        row_tile, 1, 1);
    return 1;
}

/* Strict direct hook for the q_b K128/P144 candidate.  Unlike production
 * dispatch, incompatibility fails instead of silently falling back to K64 so
 * benchmark output cannot be mislabeled. */
extern "C" int ds4_rocm_bench_q4_K_wmma_k128_enqueue(
        void *out,
        const void *w,
        const void *x,
        uint32_t n_tok,
        uint32_t n_groups,
        uint32_t in_dim,
        uint32_t out_dim,
        uint64_t row_bytes,
        uint64_t x_token_stride,
        uint64_t x_group_stride,
        uint64_t out_token_stride) {
    if (!out || !w || !x || n_tok == 0u || n_groups == 0u ||
        in_dim == 0u || out_dim == 0u ||
        (in_dim % CUDA_QK_K) != 0u || out_dim < 8192u ||
        !rocm_q4_K_prefill_wmma_load4_compatible(
            reinterpret_cast<const float *>(x),
            x_token_stride, x_group_stride)) {
        return 0;
    }
    const uint64_t minimum_row_bytes =
        ((uint64_t)in_dim / CUDA_QK_K) * sizeof(cuda_block_q4_K);
    if (row_bytes < minimum_row_bytes) return 0;
    rocm_q4_K_prefill_wmma_k128_enqueue(
        reinterpret_cast<float *>(out),
        reinterpret_cast<const char *>(w),
        reinterpret_cast<const float *>(x),
        n_tok, n_groups, in_dim, out_dim, row_bytes,
        x_token_stride, x_group_stride, out_token_stride);
    return 1;
}

enum {
    ROCM_Q4_PREFILL_K1024_TILE4_REQUIRED_FAILURE = -1,
    ROCM_Q4_PREFILL_K1024_TILE4_FALLBACK = 0,
    ROCM_Q4_PREFILL_K1024_TILE4_USE = 1,
};

/* Keep this decision independent from the device lookup so the complete
 * policy matrix has a hardware-free oracle.  Resident execution preserves
 * the established automatic default.  SSD execution stays opt-in and may
 * only select TILE4 after the exact weight range has been found in actual
 * device storage; mapped/registered host memory is deliberately insufficient.
 * REQUIRE requests the candidate as well as asserting it, while DISABLE is
 * authoritative in both modes. */
static int rocm_q4_K_prefill_k1024_tile4_policy(
        int ssd_streaming,
        int weight_device_resident,
        int ssd_enabled,
        int disabled,
        int required) {
    if (disabled) {
        return required ? ROCM_Q4_PREFILL_K1024_TILE4_REQUIRED_FAILURE
                        : ROCM_Q4_PREFILL_K1024_TILE4_FALLBACK;
    }
    if (!ssd_streaming) return ROCM_Q4_PREFILL_K1024_TILE4_USE;
    if (!ssd_enabled && !required) {
        return ROCM_Q4_PREFILL_K1024_TILE4_FALLBACK;
    }
    if (!weight_device_resident) {
        return required ? ROCM_Q4_PREFILL_K1024_TILE4_REQUIRED_FAILURE
                        : ROCM_Q4_PREFILL_K1024_TILE4_FALLBACK;
    }
    return ROCM_Q4_PREFILL_K1024_TILE4_USE;
}

/* Test-only pure-policy entry point.  It intentionally performs no HIP call,
 * so hosts with a ROCm toolchain but no visible device can still validate the
 * SSD default, residency gate, and DISABLE/REQUIRE precedence. */
extern "C" int ds4_rocm_test_q4_prefill_k1024_tile4_policy(
        int ssd_streaming,
        int weight_device_resident,
        int ssd_enabled,
        int disabled,
        int required) {
    return rocm_q4_K_prefill_k1024_tile4_policy(
        ssd_streaming != 0, weight_device_resident != 0,
        ssd_enabled != 0, disabled != 0, required != 0);
}

static int rocm_q4_K_prefill_k1024_tile4_resolve(
        uint64_t blocks,
        uint64_t out_dim,
        const void *model_map,
        uint64_t weight_offset,
        uint64_t weight_bytes,
        const char *weight_ptr) {
    if (blocks != ROCM_Q4_PREFILL_K1024_KBLOCK_TILE ||
        out_dim != DS4_ROCM_Q4_ATTN_Q_B_OUT_DIM) {
        return ROCM_Q4_PREFILL_K1024_TILE4_FALLBACK;
    }

    const int enabled = rocm_q4_attn_q_b_env_bool(
        "DS4_ROCM_ENABLE_Q4_PREFILL_K1024_TILE4_SSD") == 1;
    const int disabled = rocm_q4_attn_q_b_env_bool(
        "DS4_ROCM_DISABLE_Q4_PREFILL_K1024_TILE4") == 1;
    const int required = rocm_q4_attn_q_b_env_bool(
        "DS4_ROCM_REQUIRE_Q4_PREFILL_K1024_TILE4") == 1;
    const char *resident_ptr = g_ssd_streaming_mode
        ? rocm_q4_attn_q_b_device_resident_source(
              model_map, weight_offset, weight_bytes)
        : weight_ptr;
    const int weight_device_resident =
        resident_ptr != NULL && resident_ptr == weight_ptr;
    const int decision = rocm_q4_K_prefill_k1024_tile4_policy(
        g_ssd_streaming_mode, weight_device_resident, enabled, disabled,
        required);
    if (decision == ROCM_Q4_PREFILL_K1024_TILE4_REQUIRED_FAILURE) {
        if (disabled) {
            fprintf(stderr,
                    "ds4: required ROCm Q4_K prefill K1024 tile4 is "
                    "disabled\n");
        } else {
            fprintf(stderr,
                    "ds4: required ROCm Q4_K prefill K1024 tile4 has no "
                    "device-resident SSD weight range "
                    "(offset=%llu bytes=%llu)\n",
                    (unsigned long long)weight_offset,
                    (unsigned long long)weight_bytes);
        }
    }
    return decision;
}

static uint64_t g_rocm_q4_prefill_tile8_dense_calls;
static uint64_t g_rocm_q4_prefill_tile8_pair_calls;
static uint64_t g_rocm_q4_prefill_tile8_attention_batch_calls;
static uint64_t g_rocm_q4_prefill_k1024_tile4_calls;
static uint64_t g_rocm_q4_prefill_k1024_tile4_ssd_calls;
static uint64_t g_rocm_q4_prefill_tile8_tokens;
static pthread_mutex_t g_rocm_q4_prefill_tile8_stats_mutex =
    PTHREAD_MUTEX_INITIALIZER;

static uint64_t g_rocm_q4_grouped_attn_a_calls;
static uint64_t g_rocm_q4_grouped_attn_a_dispatches;
static uint64_t g_rocm_q4_grouped_attn_a_groups;
static uint64_t g_rocm_q4_grouped_attn_a_fallbacks;
static uint64_t g_rocm_q4_grouped_attn_a_failures;
static int g_rocm_q4_grouped_attn_a_report_registered;
static pthread_mutex_t g_rocm_q4_grouped_attn_a_stats_mutex =
    PTHREAD_MUTEX_INITIALIZER;

static void rocm_q4_K_grouped_attn_a_report(void) {
    pthread_mutex_lock(&g_rocm_q4_grouped_attn_a_stats_mutex);
    const uint64_t calls = g_rocm_q4_grouped_attn_a_calls;
    const uint64_t dispatches = g_rocm_q4_grouped_attn_a_dispatches;
    const uint64_t groups = g_rocm_q4_grouped_attn_a_groups;
    const uint64_t fallbacks = g_rocm_q4_grouped_attn_a_fallbacks;
    const uint64_t failures = g_rocm_q4_grouped_attn_a_failures;
    pthread_mutex_unlock(&g_rocm_q4_grouped_attn_a_stats_mutex);
    fprintf(stderr,
            "ds4: ROCm Q4_K grouped attention-A decode stats: "
            "calls=%llu dispatches=%llu groups=%llu fallbacks=%llu failures=%llu\n",
            (unsigned long long)calls,
            (unsigned long long)dispatches,
            (unsigned long long)groups,
            (unsigned long long)fallbacks,
            (unsigned long long)failures);
}

static int rocm_q4_K_grouped_attn_a_result(int rc, uint32_t n_groups) {
    if (getenv("DS4_ROCM_Q4_GROUPED_ATTN_A_STATS") != NULL) {
        pthread_mutex_lock(&g_rocm_q4_grouped_attn_a_stats_mutex);
        if (!g_rocm_q4_grouped_attn_a_report_registered) {
            g_rocm_q4_grouped_attn_a_report_registered = 1;
            (void)atexit(rocm_q4_K_grouped_attn_a_report);
        }
        g_rocm_q4_grouped_attn_a_calls++;
        if (rc > 0) {
            g_rocm_q4_grouped_attn_a_dispatches++;
            g_rocm_q4_grouped_attn_a_groups += n_groups;
        } else if (rc < 0) {
            g_rocm_q4_grouped_attn_a_failures++;
        } else {
            g_rocm_q4_grouped_attn_a_fallbacks++;
        }
        pthread_mutex_unlock(&g_rocm_q4_grouped_attn_a_stats_mutex);
    }
    return rc;
}

static void rocm_q4_K_prefill_tile8_report(void) {
    const uint64_t wmma_calls =
        __atomic_load_n(&g_rocm_q4_prefill_wmma_launches,
                        __ATOMIC_RELAXED);
    const uint64_t wmma_k64_calls =
        __atomic_load_n(&g_rocm_q4_prefill_wmma_k64_launches,
                        __ATOMIC_RELAXED);
    const uint64_t wmma_k128_calls =
        __atomic_load_n(&g_rocm_q4_prefill_wmma_k128_launches,
                           __ATOMIC_RELAXED);
    const uint64_t wmma_k64_load4_calls =
        ds4_rocm_test_q4_prefill_wmma_k64_load4_get_calls();
    const uint64_t staged_calls = wmma_k64_calls + wmma_k128_calls;
    const uint64_t wmma_k32_calls =
        wmma_calls >= staged_calls ? wmma_calls - staged_calls : 0u;
    pthread_mutex_lock(&g_rocm_q4_prefill_tile8_stats_mutex);
    const uint64_t dense_calls = g_rocm_q4_prefill_tile8_dense_calls;
    const uint64_t pair_calls = g_rocm_q4_prefill_tile8_pair_calls;
    const uint64_t attention_batch_calls =
        g_rocm_q4_prefill_tile8_attention_batch_calls;
    const uint64_t k1024_tile4_calls =
        g_rocm_q4_prefill_k1024_tile4_calls;
    const uint64_t k1024_tile4_ssd_calls =
        g_rocm_q4_prefill_k1024_tile4_ssd_calls;
    const uint64_t tokens = g_rocm_q4_prefill_tile8_tokens;
    pthread_mutex_unlock(&g_rocm_q4_prefill_tile8_stats_mutex);
    fprintf(stderr,
            "ds4: ROCm Q4_K tiled-prefill stats: "
            "dense_calls=%llu pair_calls=%llu attention_batch_calls=%llu "
            "k1024_tile4_calls=%llu k1024_tile4_ssd_calls=%llu "
            "wmma_calls=%llu wmma_k32_calls=%llu wmma_k64_calls=%llu "
            "wmma_k128_calls=%llu wmma_k64_load4_calls=%llu "
            "lds_aligned_calls=%llu tokens=%llu\n",
            (unsigned long long)dense_calls,
            (unsigned long long)pair_calls,
            (unsigned long long)attention_batch_calls,
            (unsigned long long)k1024_tile4_calls,
            (unsigned long long)k1024_tile4_ssd_calls,
            (unsigned long long)wmma_calls,
            (unsigned long long)wmma_k32_calls,
            (unsigned long long)wmma_k64_calls,
            (unsigned long long)wmma_k128_calls,
            (unsigned long long)wmma_k64_load4_calls,
            (unsigned long long)ds4_rocm_test_q4_prefill_lds_aligned_get_calls(),
            (unsigned long long)tokens);
}

static void rocm_q4_K_prefill_tile8_note(
        uint32_t dense_calls,
        uint32_t pair_calls,
        uint32_t attention_batch_calls,
        uint32_t k1024_tile4_calls,
        uint64_t tokens) {
    if (getenv("DS4_ROCM_Q4_PREFILL_TILE8_STATS") == NULL) return;
    rocm_q4_K_prefill_stats_register();
    pthread_mutex_lock(&g_rocm_q4_prefill_tile8_stats_mutex);
    g_rocm_q4_prefill_tile8_dense_calls += dense_calls;
    g_rocm_q4_prefill_tile8_pair_calls += pair_calls;
    g_rocm_q4_prefill_tile8_attention_batch_calls += attention_batch_calls;
    g_rocm_q4_prefill_k1024_tile4_calls += k1024_tile4_calls;
    if (k1024_tile4_calls && g_ssd_streaming_mode) {
        g_rocm_q4_prefill_k1024_tile4_ssd_calls += k1024_tile4_calls;
    }
    g_rocm_q4_prefill_tile8_tokens += tokens;
    pthread_mutex_unlock(&g_rocm_q4_prefill_tile8_stats_mutex);
}

extern "C" int ds4_rocm_matmul_q4_K_tensor(
        ds4_gpu_tensor *out,
        const void *model_map,
        uint64_t model_size,
        uint64_t weight_offset,
        uint64_t in_dim,
        uint64_t out_dim,
        const ds4_gpu_tensor *x,
        uint64_t n_tok) {
    // REQUIRE is a standalone-API assertion, including shape exclusions.
    // Resolve before range lookup, allocation or any quantizer enqueue.
    const int decode_lane4 = rocm_q4_decode_lane4_select(in_dim, out_dim, n_tok);
    if (decode_lane4 == ds4_rocm_q4_decode::required_failure) return 0;
    uint64_t blocks = 0;
    uint64_t row_bytes = 0;
    uint64_t weight_bytes = 0;
    if (!rocm_q4_K_dense_validate(out, model_map, model_size, weight_offset,
                                  in_dim, out_dim, x, n_tok, &blocks,
                                  &row_bytes, &weight_bytes)) {
        return 0;
    }

    const int prefill_scope = rocm_q4_K_prefill_tile8_scope(n_tok);
    const int prefill_tile8 = rocm_q4_K_prefill_tile8_requested();
    const int prefill_tile8_required = rocm_q4_K_prefill_tile8_required();
    const int q8_wave32 = rocm_q4_K_prefill_q8_wave32_select(n_tok);
    const int q8_wave32_required =
        rocm_q4_K_prefill_q8_wave32_required();
    if (q8_wave32 == ROCM_Q4_PREFILL_Q8_WAVE32_REQUIRED_FAILURE) {
        return 0;
    }
    const int k1024_tile4_shape =
        blocks == ROCM_Q4_PREFILL_K1024_KBLOCK_TILE &&
        out_dim == DS4_ROCM_Q4_ATTN_Q_B_OUT_DIM;
    const int k1024_tile4_required = rocm_q4_attn_q_b_env_bool(
        "DS4_ROCM_REQUIRE_Q4_PREFILL_K1024_TILE4") == 1;
    if (prefill_scope && prefill_tile8_required && !prefill_tile8) {
        fprintf(stderr,
                "ds4: required ROCm Q4_K prefill tile8 is disabled "
                "(n_tok=%llu)\n",
                (unsigned long long)n_tok);
        return 0;
    }
    if (prefill_scope && k1024_tile4_shape && k1024_tile4_required &&
        !prefill_tile8) {
        fprintf(stderr,
                "ds4: required ROCm Q4_K prefill K1024 tile4 cannot run "
                "because tiled prefill is disabled (n_tok=%llu)\n",
                (unsigned long long)n_tok);
        return 0;
    }

    const char *wptr = cuda_model_range_ptr(model_map, weight_offset,
                                            weight_bytes, "q4_K dense");
    if (!wptr) return 0;
    const char *resident_wptr = g_ssd_streaming_mode
        ? rocm_q4_attn_q_b_device_resident_source(
              model_map, weight_offset, weight_bytes)
        : wptr;
    const int weight_device_resident =
        resident_wptr != NULL && resident_wptr == wptr;
    int prefill_wmma = rocm_q4_K_prefill_wmma_select(
        n_tok, in_dim, out_dim, weight_device_resident);
    if (prefill_wmma == ROCM_Q4_PREFILL_WMMA_REQUIRED_FAILURE) return 0;
    if (prefill_wmma == ROCM_Q4_PREFILL_WMMA_USE &&
        q8_wave32 == ROCM_Q4_PREFILL_Q8_WAVE32_USE) {
        const int wmma_required = rocm_q4_attn_q_b_env_bool(
            "DS4_ROCM_REQUIRE_Q4_PREFILL_WMMA") == 1;
        if (wmma_required && q8_wave32_required) {
            fprintf(stderr,
                    DS4_GPU_LOG_PREFIX
                    "Q4_K prefill cannot require both direct WMMA and "
                    "the Q8_K wave32 quantizer\n");
            return 0;
        }
        /* The direct F16 WMMA path has no Q8_K RHS.  An explicitly selected
         * exact quantizer therefore owns an optional automatic WMMA path. */
        if (rocm_q4_K_prefill_wmma_yields_to_q8_wave32(
                q8_wave32, wmma_required)) {
            prefill_wmma = ROCM_Q4_PREFILL_WMMA_FALLBACK;
        }
    }
    if (prefill_wmma == ROCM_Q4_PREFILL_WMMA_USE &&
        prefill_scope && prefill_tile8_required) {
        if (rocm_q4_attn_q_b_env_bool(
                "DS4_ROCM_REQUIRE_Q4_PREFILL_WMMA") == 1) {
            fprintf(stderr,
                    DS4_GPU_LOG_PREFIX
                    "Q4_K prefill cannot require both WMMA and TILE8\n");
            return 0;
        }
        /* REQUIRE_TILE8 owns the dispatch when WMMA is only an optional
         * experiment. */
        prefill_wmma = ROCM_Q4_PREFILL_WMMA_FALLBACK;
    }
    if (prefill_wmma == ROCM_Q4_PREFILL_WMMA_USE &&
        k1024_tile4_shape && k1024_tile4_required) {
        if (rocm_q4_attn_q_b_env_bool(
                "DS4_ROCM_REQUIRE_Q4_PREFILL_WMMA") == 1) {
            fprintf(stderr,
                    DS4_GPU_LOG_PREFIX
                    "Q4_K prefill cannot require both WMMA and K1024 TILE4\n");
            return 0;
        }
        prefill_wmma = ROCM_Q4_PREFILL_WMMA_FALLBACK;
    }
    if (prefill_wmma == ROCM_Q4_PREFILL_WMMA_USE) {
        return rocm_q4_K_prefill_wmma_launch(
            reinterpret_cast<float *>(out->ptr), wptr,
            reinterpret_cast<const float *>(x->ptr),
            (uint32_t)n_tok, 1u, (uint32_t)in_dim, (uint32_t)out_dim,
            row_bytes, in_dim, 0u, out_dim,
            "q4_K dense prefill WMMA rowtile launch");
    }
    int k1024_tile4 = ROCM_Q4_PREFILL_K1024_TILE4_FALLBACK;
    if (prefill_scope && prefill_tile8) {
        k1024_tile4 = rocm_q4_K_prefill_k1024_tile4_resolve(
            blocks, out_dim, model_map, weight_offset, weight_bytes, wptr);
        if (k1024_tile4 ==
            ROCM_Q4_PREFILL_K1024_TILE4_REQUIRED_FAILURE) {
            return 0;
        }
    }
    cuda_block_q8_K *xq = rocm_q4_K_prequant_alloc(
            n_tok, blocks, "q4_K dense prequant");
    if (!xq) return 0;

    if (!rocm_q4_K_q8_quantize_launch(
            xq, reinterpret_cast<const float *>(x->ptr),
            (uint32_t)in_dim, (uint32_t)n_tok, q8_wave32,
            q8_wave32_required,
            "q4_K dense quantize launch")) {
        return 0;
    }

    if (prefill_scope && prefill_tile8) {
        if (k1024_tile4 == ROCM_Q4_PREFILL_K1024_TILE4_USE) {
            const dim3 tiled_grid(
                (unsigned)((out_dim - 1u) /
                           ROCM_Q4_PREFILL_K1024_ROWS + 1u),
                (unsigned)((n_tok - 1u) /
                           ROCM_Q4_PREFILL_TOKEN_TILE + 1u),
                1u);
            rocm_q4_K_prefill_k1024_tile4_launch(tiled_grid,
                    reinterpret_cast<float *>(out->ptr), wptr, xq,
                    row_bytes, (uint32_t)out_dim, (uint32_t)n_tok);
            const int ok = cuda_ok(
                    cudaGetLastError(),
                    "q4_K dense prefill K1024 tile4 launch");
            if (ok) {
                rocm_q4_K_prefill_tile8_note(1u, 0u, 0u, 1u, n_tok);
            }
            return ok;
        }
        const dim3 tiled_grid((unsigned)((out_dim - 1u) / 32u + 1u),
                              (unsigned)((n_tok - 1u) /
                                         ROCM_Q4_PREFILL_TOKEN_TILE + 1u),
                              1u);
        rocm_q4_K_prefill_tile8_strided_launch(tiled_grid,
                reinterpret_cast<float *>(out->ptr), wptr, xq, row_bytes,
                (uint32_t)blocks, (uint32_t)out_dim, (uint32_t)n_tok,
                blocks, out_dim);
        const int ok = cuda_ok(cudaGetLastError(),
                               "q4_K dense prefill tile8 launch");
        if (ok) rocm_q4_K_prefill_tile8_note(1u, 0u, 0u, 0u, n_tok);
        return ok;
    }

    if (decode_lane4 == ds4_rocm_q4_decode::use) {
        const dim3 grid(ds4_rocm_q4_decode::m / ds4_rocm_q4_decode::rows,
                        (unsigned)n_tok, 1u);
        rocm_matmul_q4_K_dense_decode_lane4_kernel<<<grid, 256>>>(
            reinterpret_cast<float *>(out->ptr), wptr, xq, row_bytes,
            (uint32_t)blocks, (uint32_t)out_dim, (uint32_t)n_tok);
        const int ok = cuda_ok(cudaGetLastError(), "q4_K decode lane4 launch");
        if (ok) ++g_rocm_q4_decode_lane4_launches;
        return ok;
    }

    const dim3 grid((unsigned)((out_dim - 1u) / 32u + 1u),
                    (unsigned)n_tok, 1u);
    rocm_matmul_q4_K_dense_kernel<<<grid, 256>>>(
            reinterpret_cast<float *>(out->ptr), wptr, xq, row_bytes,
            (uint32_t)blocks, (uint32_t)out_dim, (uint32_t)n_tok);
    return cuda_ok(cudaGetLastError(), "q4_K dense matmul launch");
}

/* Only REQUIRE_TILE8 makes the fused pair itself a strict contract.  A
 * Q8-wave32-only REQUIRE belongs to the quantizer and can still be satisfied
 * by the graph's two dense fallbacks.  Keep this pre-enqueue policy in one
 * place so validation, alias/range rejection and scratch allocation cannot
 * silently weaken the pair contract.  Decode is outside prefill scope and
 * therefore keeps its legacy optional status. */
static int rocm_q4_K_pair_pre_enqueue_failure_policy(
        int prefill_scope,
        int tile8_required,
        int q8_wave32_required) {
    (void)q8_wave32_required;
    return prefill_scope && tile8_required ? -1 : 0;
}

extern "C" int ds4_rocm_test_q4_pair_pre_enqueue_failure_policy(
        int prefill_scope,
        int tile8_required,
        int q8_wave32_required) {
    return rocm_q4_K_pair_pre_enqueue_failure_policy(
        prefill_scope != 0, tile8_required != 0,
        q8_wave32_required != 0);
}

extern "C" int ds4_gpu_matmul_q4_K_pair_tensor(
        ds4_gpu_tensor *out0,
        ds4_gpu_tensor *out1,
        const void *model_map,
        uint64_t model_size,
        uint64_t weight0_offset,
        uint64_t weight1_offset,
        uint64_t in_dim,
        uint64_t out0_dim,
        uint64_t out1_dim,
        const ds4_gpu_tensor *x,
        uint64_t n_tok) {
    const int prefill_scope = rocm_q4_K_prefill_tile8_scope(n_tok);
    const int tile8_required = rocm_q4_K_prefill_tile8_required();
    const int prefill_required = prefill_scope && tile8_required;
    const int wmma_required = rocm_q4_attn_q_b_env_bool(
        "DS4_ROCM_REQUIRE_Q4_PREFILL_WMMA") == 1;
    const int q8_wave32 = rocm_q4_K_prefill_q8_wave32_select(n_tok);
    const int q8_wave32_required =
        rocm_q4_K_prefill_q8_wave32_required();
    if (q8_wave32 == ROCM_Q4_PREFILL_Q8_WAVE32_REQUIRED_FAILURE) {
        return -1;
    }
    const int pre_enqueue_failure =
        rocm_q4_K_pair_pre_enqueue_failure_policy(
            prefill_scope, tile8_required, q8_wave32_required);
    if (q8_wave32_required && wmma_required) {
        fprintf(stderr,
                DS4_GPU_LOG_PREFIX
                "Q4_K prefill pair cannot require both direct WMMA and "
                "the Q8_K wave32 quantizer\n");
        return -1;
    }

    /* The fused pair consumes a shared Q8_K activation tile.  When the direct
     * F16 WMMA path is selected, return before validation/enqueue so the
     * graph's established fallback issues two dense calls and both can take
     * the strict WMMA path.  This also prevents REQUIRE from falsely passing
     * after silently measuring the TILE8 pair.  SSD preflight is deliberately
     * optimistic about residency: it only decides whether to yield; each dense
     * fallback then proves its exact physical device range before enqueue. */
    if (!prefill_required &&
        !rocm_q4_K_prefill_wmma_yields_to_q8_wave32(
            q8_wave32, wmma_required)) {
        const int wmma0 = rocm_q4_K_prefill_wmma_select(
            n_tok, in_dim, out0_dim, 1);
        const int wmma1 = rocm_q4_K_prefill_wmma_select(
            n_tok, in_dim, out1_dim, 1);
        if (wmma0 == ROCM_Q4_PREFILL_WMMA_REQUIRED_FAILURE ||
            wmma1 == ROCM_Q4_PREFILL_WMMA_REQUIRED_FAILURE) {
            return -1;
        }
        if (wmma0 == ROCM_Q4_PREFILL_WMMA_USE ||
            wmma1 == ROCM_Q4_PREFILL_WMMA_USE) {
            return 0;
        }
    } else if (wmma_required) {
        fprintf(stderr,
                DS4_GPU_LOG_PREFIX
                "Q4_K prefill pair cannot require both WMMA and TILE8\n");
        return -1;
    }

    const int prefill_pair = prefill_scope &&
                             rocm_q4_K_prefill_tile8_requested();
    if (prefill_required && !prefill_pair) {
        fprintf(stderr,
                "ds4: required ROCm Q4_K prefill tile8 pair is disabled "
                "(n_tok=%llu)\n",
                (unsigned long long)n_tok);
        return -1;
    }

    /* Decode keeps its original, separately gated pair path.  Prefill uses
     * the common tile8 gate and shares one canonical Q8_K quantization and
     * one tiled launch across the two projections. */
    const int decode_pair = n_tok <= 8u &&
                            rocm_q4_K_dense_pair_requested();
    if (!prefill_pair && !decode_pair) {
        return pre_enqueue_failure;
    }
    if (!out0 || !out1 || out0 == out1 ||
        (out0 && out1 && out0->ptr == out1->ptr)) {
        return pre_enqueue_failure;
    }

    uint64_t blocks0 = 0, blocks1 = 0;
    uint64_t row_bytes0 = 0, row_bytes1 = 0;
    uint64_t weight0_bytes = 0, weight1_bytes = 0;
    if (!rocm_q4_K_dense_validate(out0, model_map, model_size, weight0_offset,
                                  in_dim, out0_dim, x, n_tok, &blocks0,
                                  &row_bytes0, &weight0_bytes) ||
        !rocm_q4_K_dense_validate(out1, model_map, model_size, weight1_offset,
                                  in_dim, out1_dim, x, n_tok, &blocks1,
                                  &row_bytes1, &weight1_bytes) ||
        blocks0 != blocks1 || row_bytes0 != row_bytes1) {
        return pre_enqueue_failure;
    }
    uint64_t out0_bytes = 0;
    uint64_t out1_bytes = 0;
    if (!cuda_u64_mul3_checked(n_tok, out0_dim, sizeof(float), &out0_bytes) ||
        !cuda_u64_mul3_checked(n_tok, out1_dim, sizeof(float), &out1_bytes) ||
        rocm_q4_K_byte_ranges_overlap(out0->ptr, out0_bytes,
                                      out1->ptr, out1_bytes)) {
        return pre_enqueue_failure;
    }

    const char *w0 = cuda_model_range_ptr(model_map, weight0_offset,
                                          weight0_bytes, "q4_K dense pair0");
    const char *w1 = cuda_model_range_ptr(model_map, weight1_offset,
                                          weight1_bytes, "q4_K dense pair1");
    if (!w0 || !w1) return pre_enqueue_failure;
    cuda_block_q8_K *xq = rocm_q4_K_prequant_alloc(
            n_tok, blocks0, "q4_K dense pair prequant");
    if (!xq) return pre_enqueue_failure;

    if (!rocm_q4_K_q8_quantize_launch(
            xq, reinterpret_cast<const float *>(x->ptr),
            (uint32_t)in_dim, (uint32_t)n_tok, q8_wave32,
            q8_wave32_required,
            "q4_K dense pair quantize launch")) {
        return -1;
    }

    if (prefill_pair) {
        const uint64_t out0_tiles = (out0_dim - 1u) / 32u + 1u;
        const uint64_t out1_tiles = (out1_dim - 1u) / 32u + 1u;
        const dim3 grid((unsigned)(out0_tiles + out1_tiles),
                        (unsigned)((n_tok - 1u) /
                                   ROCM_Q4_PREFILL_TOKEN_TILE + 1u),
                        1u);
        rocm_q4_K_prefill_tile8_pair_launch(grid,
                reinterpret_cast<float *>(out0->ptr),
                reinterpret_cast<float *>(out1->ptr), w0, w1, xq,
                row_bytes0, (uint32_t)blocks0, (uint32_t)out0_dim,
                (uint32_t)out1_dim, (uint32_t)n_tok);
        const int ok = cuda_ok(cudaGetLastError(),
                               "q4_K dense prefill pair tile8 launch");
        if (ok) rocm_q4_K_prefill_tile8_note(0u, 1u, 0u, 0u, n_tok);
        return ok ? 1 : -1;
    }

    const uint64_t out0_tiles = (out0_dim - 1u) / 32u + 1u;
    const uint64_t out1_tiles = (out1_dim - 1u) / 32u + 1u;
    const dim3 grid((unsigned)(out0_tiles + out1_tiles),
                    (unsigned)n_tok, 1u);
    rocm_matmul_q4_K_dense_pair_kernel<<<grid, 256>>>(
            reinterpret_cast<float *>(out0->ptr),
            reinterpret_cast<float *>(out1->ptr), w0, w1, xq,
            row_bytes0, (uint32_t)blocks0, (uint32_t)out0_dim,
            (uint32_t)out1_dim, (uint32_t)n_tok);
    return cuda_ok(cudaGetLastError(), "q4_K dense pair matmul launch")
        ? 1 : -1;
}

extern "C" int ds4_gpu_attention_output_low_q4_K_slice_tensor(
        ds4_gpu_tensor *low, const void *model_map, uint64_t model_size,
        uint64_t out_a_offset, uint64_t group_dim, uint64_t rank,
        uint32_t group0, uint32_t group_cnt,
        const ds4_gpu_tensor *heads, int resident_decode) {
    const int disabled =
        getenv("DS4_ROCM_DISABLE_Q4_GROUPED_ATTN_A") != NULL;
    const int required =
        getenv("DS4_ROCM_REQUIRE_Q4_GROUPED_ATTN_A") != NULL;
    const int enabled =
        getenv("DS4_ROCM_ENABLE_Q4_GROUPED_ATTN_A") != NULL;
    const int resident_default =
        rocm_q4_K_grouped_attn_a_resident_default_scope(
            group_dim, rank, group0, group_cnt, resident_decode);
    /* DISABLE is authoritative; REQUIRE reports that rollback as a failure
     * instead of allowing the graph to false-green through its fallback. */
    if (disabled) {
        if (required) {
            fprintf(stderr,
                    "ds4: required ROCm Q4_K grouped attention-A decode "
                    "is disabled\n");
        }
        return rocm_q4_K_grouped_attn_a_result(required ? -1 : 0, 0u);
    }
    if (!resident_default && !enabled && !required) {
        return rocm_q4_K_grouped_attn_a_result(0, 0u);
    }
    const int pre_enqueue_failure = required ? -1 : 0;
    if (!low || !heads || !model_map || group_dim == 0u || rank == 0u ||
        group_cnt == 0u || group_dim > UINT32_MAX || rank > UINT32_MAX ||
        group_cnt > UINT16_MAX || (group_dim % CUDA_QK_K) != 0u ||
        group0 > UINT32_MAX - group_cnt) {
        return rocm_q4_K_grouped_attn_a_result(pre_enqueue_failure, 0u);
    }

    const uint64_t blocks = group_dim / CUDA_QK_K;
    uint64_t row_bytes = 0, group_weight_bytes = 0, group_skip = 0;
    uint64_t selected_weight_bytes = 0, selected_offset = 0;
    if (blocks == 0u ||
        !cuda_u64_mul_checked(blocks, sizeof(cuda_block_q4_K), &row_bytes) ||
        !cuda_u64_mul_checked(rank, row_bytes, &group_weight_bytes) ||
        !cuda_u64_mul_checked(group0, group_weight_bytes, &group_skip) ||
        !cuda_u64_mul_checked(group_cnt, group_weight_bytes,
                              &selected_weight_bytes) ||
        !cuda_u64_add_checked(out_a_offset, group_skip, &selected_offset) ||
        !cuda_model_range_fits(model_size, selected_offset,
                               selected_weight_bytes) ||
        !cuda_tensor_has_elems2(heads, group_cnt, group_dim, sizeof(float)) ||
        !cuda_tensor_has_elems2(low, group_cnt, rank, sizeof(float))) {
        return rocm_q4_K_grouped_attn_a_result(pre_enqueue_failure, 0u);
    }

    const char *w = cuda_model_range_ptr(
        model_map, selected_offset, selected_weight_bytes,
        "q4_K grouped attention output A decode");
    cuda_block_q8_K *xq = rocm_q4_K_prequant_alloc(
        group_cnt, blocks, "q4_K grouped attention output A decode prequant");
    if (!w || !xq) {
        return rocm_q4_K_grouped_attn_a_result(pre_enqueue_failure, 0u);
    }

    const dim3 qgrid((unsigned)blocks, group_cnt, 1u);
    q8_K_quantize_kernel<<<qgrid, 256>>>(
        xq, reinterpret_cast<const float *>(heads->ptr),
        (uint32_t)group_dim, group_cnt);
    if (!cuda_ok(cudaGetLastError(),
                 "q4_K grouped attention output A decode quantize launch")) {
        return rocm_q4_K_grouped_attn_a_result(-1, 0u);
    }
    const dim3 grid((unsigned)((rank - 1u) / 32u + 1u), 1u, group_cnt);
    rocm_matmul_q4_K_dense_grouped_decode_kernel<<<grid, 256>>>(
        reinterpret_cast<float *>(low->ptr), w, xq, row_bytes,
        (uint32_t)blocks, (uint32_t)rank, group_cnt);
    if (!cuda_ok(cudaGetLastError(),
                 "q4_K grouped attention output A decode matmul launch")) {
        return rocm_q4_K_grouped_attn_a_result(-1, 0u);
    }
    return rocm_q4_K_grouped_attn_a_result(1, group_cnt);
}

/* Quantize token-major [token][group][K] rows once, then apply group-major
 * [group][out_row][K] Q4_K weights directly into token-major output.  A
 * return of -1 means the quantize launch was accepted and callers must not
 * replay a row fallback over potentially submitted work. */
static int rocm_q4_K_prefill_tile8_quant_launch(
        float *out,
        const char *w,
        const float *x,
        uint32_t n_tok,
        uint32_t n_groups,
        uint32_t in_dim,
        uint32_t out_dim,
        uint64_t row_bytes,
        int prefill_wmma,
        int q8_wave32,
        int q8_wave32_required,
        const char *label) {
    uint64_t n_rows = 0;
    uint64_t xq_token_stride = 0;
    uint64_t x_token_stride = 0;
    uint64_t out_token_stride = 0;
    const uint64_t blocks = in_dim / CUDA_QK_K;
    if (!out || !w || !x || n_tok == 0u || n_groups == 0u ||
        in_dim == 0u || out_dim == 0u || blocks == 0u ||
        (in_dim % CUDA_QK_K) != 0u ||
        !cuda_u64_mul_checked(n_tok, n_groups, &n_rows) ||
        /* HIP keeps the portable grid-y limit at 65535.  Real AProjQ4 uses
         * eight groups, so even the 4096-token ceiling remains in range. */
        n_rows > UINT16_MAX ||
        !cuda_u64_mul_checked(n_groups, blocks, &xq_token_stride) ||
        !cuda_u64_mul_checked(n_groups, in_dim, &x_token_stride) ||
        !cuda_u64_mul_checked(n_groups, out_dim, &out_token_stride)) {
        return 0;
    }

    if (prefill_wmma == ROCM_Q4_PREFILL_WMMA_USE) {
        return rocm_q4_K_prefill_wmma_launch(
                   out, w, x, n_tok, n_groups, in_dim, out_dim, row_bytes,
                   x_token_stride, in_dim, out_token_stride,
                   label ? label : "q4_K attention-output WMMA rowtile launch")
            ? 1 : -1;
    }

    cuda_block_q8_K *xq = rocm_q4_K_prequant_alloc(
            n_rows, blocks, label ? label : "q4_K prefill tile8 prequant");
    if (!xq) return 0;

    if (!rocm_q4_K_q8_quantize_launch(
            xq, x, in_dim, (uint32_t)n_rows, q8_wave32,
            q8_wave32_required,
            "q4_K prefill tile8 quantize launch")) {
        return 0;
    }

    const dim3 grid((unsigned)((out_dim - 1u) / 32u + 1u),
                    (unsigned)((n_tok - 1u) /
                               ROCM_Q4_PREFILL_TOKEN_TILE + 1u),
                    n_groups);
    rocm_q4_K_prefill_tile8_strided_launch(grid,
            out, w, xq, row_bytes, (uint32_t)blocks, out_dim, n_tok,
            xq_token_stride, out_token_stride);
    if (!cuda_ok(cudaGetLastError(),
                 "q4_K prefill tile8 matmul launch")) {
        return -1;
    }
    return 1;
}

extern "C" int ds4_gpu_attention_output_q4_K_batch_tensor(
        ds4_gpu_tensor *out,
        ds4_gpu_tensor *low,
        ds4_gpu_tensor *group_tmp,
        ds4_gpu_tensor *low_tmp,
        const void *model_map,
        uint64_t model_size,
        uint64_t out_a_offset,
        uint64_t out_b_offset,
        uint32_t out_b_type,
        uint64_t group_dim,
        uint64_t rank,
        uint32_t n_groups,
        uint64_t out_dim,
        const ds4_gpu_tensor *heads,
        uint32_t n_tokens) {
    (void)group_tmp;
    (void)low_tmp;

    const int tile8_scope = rocm_q4_K_prefill_tile8_scope(n_tokens);
    const int tile8_requested = rocm_q4_K_prefill_tile8_requested();
    const int tile8_required = tile8_scope &&
                               rocm_q4_K_prefill_tile8_required();
    const int q8_wave32 = rocm_q4_K_prefill_q8_wave32_select(n_tokens);
    const int q8_wave32_required =
        rocm_q4_K_prefill_q8_wave32_required();
    if (q8_wave32 == ROCM_Q4_PREFILL_Q8_WAVE32_REQUIRED_FAILURE) {
        return -1;
    }
    const int wmma_enabled = rocm_q4_attn_q_b_env_bool(
        "DS4_ROCM_ENABLE_Q4_PREFILL_WMMA");
    const int wmma_ssd_enabled = rocm_q4_attn_q_b_env_bool(
        "DS4_ROCM_ENABLE_Q4_PREFILL_WMMA_SSD") == 1;
    const int wmma_disabled = rocm_q4_attn_q_b_env_bool(
        "DS4_ROCM_DISABLE_Q4_PREFILL_WMMA") == 1;
    const int wmma_required = rocm_q4_attn_q_b_env_bool(
        "DS4_ROCM_REQUIRE_Q4_PREFILL_WMMA") == 1;
    const int a_wmma_requested =
        rocm_q4_K_prefill_wmma_attention_a_requested_policy(
            g_ssd_streaming_mode, wmma_enabled, wmma_ssd_enabled,
            wmma_disabled, wmma_required);
    const int b_wmma_requested =
        rocm_q4_K_prefill_wmma_attention_b_requested_policy(
            g_ssd_streaming_mode, wmma_enabled, wmma_ssd_enabled,
            wmma_disabled, wmma_required);
    const int attention_wmma_requested =
        a_wmma_requested || b_wmma_requested;
    if (!tile8_scope) {
        /* REQUIRE is a strict diagnostic assertion.  Do not let an
         * unsupported attention-output shape escape into a grouped/per-token
         * fallback that never attests the requested WMMA kernel. */
        return wmma_required ? -1 : 0;
    }
    if (!tile8_requested && !attention_wmma_requested) {
        if (tile8_required || q8_wave32_required) {
            fprintf(stderr,
                    "ds4: required ROCm Q4_K attention-output prefill "
                    "exact path is disabled (n_tok=%u)\n",
                    n_tokens);
            return -1;
        }
        return 0;
    }
    const int pre_enqueue_failure =
        (tile8_required || wmma_required || q8_wave32_required) ? -1 : 0;

    if (!out || !low || !heads || !model_map || group_dim == 0u ||
        rank == 0u || n_groups == 0u || out_dim == 0u ||
        n_groups > UINT16_MAX || group_dim > UINT32_MAX ||
        rank > UINT32_MAX || out_dim > UINT32_MAX ||
        (group_dim % CUDA_QK_K) != 0u ||
        (out_b_type != 12u && out_b_type != 8u)) {
        return pre_enqueue_failure;
    }

    uint64_t low_dim = 0;
    uint64_t heads_rows = 0;
    uint64_t heads_bytes = 0;
    uint64_t low_bytes = 0;
    uint64_t out_bytes = 0;
    if (!cuda_u64_mul_checked(n_groups, rank, &low_dim) ||
        low_dim == 0u || low_dim > UINT32_MAX ||
        !cuda_u64_mul_checked(n_tokens, n_groups, &heads_rows) ||
        !cuda_u64_mul3_checked(heads_rows, group_dim,
                               sizeof(float), &heads_bytes) ||
        !cuda_u64_mul3_checked(n_tokens, low_dim,
                               sizeof(float), &low_bytes) ||
        !cuda_u64_mul3_checked(n_tokens, out_dim,
                               sizeof(float), &out_bytes) ||
        heads->bytes < heads_bytes || low->bytes < low_bytes ||
        out->bytes < out_bytes) {
        return pre_enqueue_failure;
    }

    const uint64_t a_blocks = group_dim / CUDA_QK_K;
    uint64_t row_a_bytes = 0;
    uint64_t out_a_bytes = 0;
    if (!cuda_u64_mul_checked(a_blocks, sizeof(cuda_block_q4_K),
                              &row_a_bytes) ||
        !cuda_u64_mul_checked(low_dim, row_a_bytes, &out_a_bytes) ||
        !cuda_model_range_fits(model_size, out_a_offset, out_a_bytes)) {
        return pre_enqueue_failure;
    }

    uint64_t row_b_bytes = 0;
    uint64_t out_b_bytes = 0;
    if (out_b_type == 12u) {
        if ((low_dim % CUDA_QK_K) != 0u ||
            !cuda_u64_mul_checked(low_dim / CUDA_QK_K,
                                  sizeof(cuda_block_q4_K), &row_b_bytes)) {
            return pre_enqueue_failure;
        }
    } else {
        const uint64_t b_blocks = (low_dim + 31u) / 32u;
        if (!cuda_u64_mul_checked(b_blocks, 34u, &row_b_bytes)) {
            return pre_enqueue_failure;
        }
    }
    if (!cuda_u64_mul_checked(out_dim, row_b_bytes, &out_b_bytes) ||
        !cuda_model_range_fits(model_size, out_b_offset, out_b_bytes)) {
        return pre_enqueue_failure;
    }

    const char *out_a = cuda_model_range_ptr(
            model_map, out_a_offset, out_a_bytes, "q4_K attention output A");
    const char *out_b = cuda_model_range_ptr(
            model_map, out_b_offset, out_b_bytes, "q4_K attention output B");
    if (!out_a || !out_b) return pre_enqueue_failure;

    const char *resident_out_a = g_ssd_streaming_mode
        ? rocm_q4_attn_q_b_device_resident_source(
              model_map, out_a_offset, out_a_bytes)
        : out_a;
    const char *resident_out_b = g_ssd_streaming_mode && out_b_type == 12u
        ? rocm_q4_attn_q_b_device_resident_source(
              model_map, out_b_offset, out_b_bytes)
        : out_b;
    const int out_a_device_resident =
        resident_out_a != NULL && resident_out_a == out_a;
    const int out_b_device_resident =
        resident_out_b != NULL && resident_out_b == out_b;

    /* Resolve both independently requested stages before A can enqueue work.
     * This keeps REQUIRE fail-closed: an ineligible B projection can never
     * make the graph replay a fallback over an already submitted A projection.
     * A retains the validated resident automatic default; B does not inherit
     * it because applying direct WMMA to both stages compounds their F16
     * approximations. */
    int a_wmma = a_wmma_requested
        ? rocm_q4_K_prefill_wmma_select(
              n_tokens, group_dim, rank, out_a_device_resident)
        : ROCM_Q4_PREFILL_WMMA_FALLBACK;
    int b_wmma = out_b_type == 12u && b_wmma_requested
        ? rocm_q4_K_prefill_wmma_select(
              n_tokens, low_dim, out_dim, out_b_device_resident)
        : ROCM_Q4_PREFILL_WMMA_FALLBACK;
    if (a_wmma == ROCM_Q4_PREFILL_WMMA_REQUIRED_FAILURE ||
        b_wmma == ROCM_Q4_PREFILL_WMMA_REQUIRED_FAILURE) {
        return -1;
    }
    if (q8_wave32 == ROCM_Q4_PREFILL_Q8_WAVE32_USE &&
        (a_wmma == ROCM_Q4_PREFILL_WMMA_USE ||
         b_wmma == ROCM_Q4_PREFILL_WMMA_USE)) {
        if (wmma_required && q8_wave32_required) {
            fprintf(stderr,
                    DS4_GPU_LOG_PREFIX
                    "Q4_K attention-output prefill cannot require both "
                    "direct WMMA and the Q8_K wave32 quantizer\n");
            return -1;
        }
        if (rocm_q4_K_prefill_wmma_yields_to_q8_wave32(
                q8_wave32, wmma_required)) {
            a_wmma = ROCM_Q4_PREFILL_WMMA_FALLBACK;
            b_wmma = ROCM_Q4_PREFILL_WMMA_FALLBACK;
        }
    }
    if (tile8_required &&
        (a_wmma == ROCM_Q4_PREFILL_WMMA_USE ||
         b_wmma == ROCM_Q4_PREFILL_WMMA_USE)) {
        if (wmma_required) {
            fprintf(stderr,
                    DS4_GPU_LOG_PREFIX
                    "Q4_K attention-output prefill cannot require both "
                    "WMMA and TILE8\n");
            return -1;
        }
        a_wmma = ROCM_Q4_PREFILL_WMMA_FALLBACK;
        b_wmma = ROCM_Q4_PREFILL_WMMA_FALLBACK;
    }
    if (!tile8_requested &&
        (a_wmma != ROCM_Q4_PREFILL_WMMA_USE ||
         (out_b_type == 12u &&
          b_wmma != ROCM_Q4_PREFILL_WMMA_USE))) {
        return pre_enqueue_failure;
    }

    /* A: the WMMA candidate consumes F32 heads directly; TILE8 retains one
     * quantization over [token,group].  Neither path needs group pack/unpack
     * buffers or an n_tokens*n_groups dispatch loop. */
    const int a_rc = rocm_q4_K_prefill_tile8_quant_launch(
            reinterpret_cast<float *>(low->ptr), out_a,
            reinterpret_cast<const float *>(heads->ptr), n_tokens, n_groups,
            (uint32_t)group_dim, (uint32_t)rank, row_a_bytes,
            a_wmma, q8_wave32, q8_wave32_required,
            "q4_K attention output A WMMA rowtile/tile8");
    if (a_rc <= 0) {
        return a_rc < 0 ? -1 : pre_enqueue_failure;
    }

    int b_rc = 0;
    if (out_b_type == 12u) {
        b_rc = rocm_q4_K_prefill_tile8_quant_launch(
                reinterpret_cast<float *>(out->ptr), out_b,
                reinterpret_cast<const float *>(low->ptr), n_tokens, 1u,
                (uint32_t)low_dim, (uint32_t)out_dim, row_b_bytes,
                b_wmma, q8_wave32, q8_wave32_required,
                "q4_K attention output B WMMA rowtile/tile8");
    } else {
        b_rc = ds4_gpu_matmul_q8_0_tensor(
                out, model_map, model_size, out_b_offset, low_dim, out_dim,
                low, n_tokens);
    }
    if (b_rc <= 0) return -1;

    const int used_q4_tile8 =
        a_wmma != ROCM_Q4_PREFILL_WMMA_USE ||
        (out_b_type == 12u && b_wmma != ROCM_Q4_PREFILL_WMMA_USE);
    if (used_q4_tile8) {
        rocm_q4_K_prefill_tile8_note(0u, 0u, 1u, 0u, n_tokens);
    }
    return 1;
}
