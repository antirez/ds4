// SPDX-License-Identifier: MIT
// ds4_mmq.cu - host wrapper around llama.cpp's vendored mul_mat_q kernels.
//
// Implements the public ds4_mmq_* entry points and explicitly instantiates
// the mul_mat_q_case<T> template for each quant type the caller needs.
//
// Status:
//   Q8_0 dense ............ implemented, parity-tested against CPU reference
//   Q2_K dense ............ pending (Phase 3)
//   IQ2_XXS dense ......... pending (Phase 3)
//   Q8_0 MoE _id .......... pending (Phase 4)
//   Q2_K MoE _id .......... pending (Phase 4)
//   IQ2_XXS MoE _id ....... pending (Phase 4)

#include "ds4_mmq.h"

#include "common.cuh"
#include "mmq.cuh"
#include "quantize.cuh"
#include "mmid.cuh"

#include <cstdio>

// ----------------------------------------------------------------------------
// Init
// ----------------------------------------------------------------------------

extern "C" int ds4_mmq_init(int device) {
    if (device < 0) {
        fprintf(stderr, "ds4_mmq_init: invalid device %d\n", device);
        return -1;
    }
    ggml_cuda_set_device(device);
    // Trigger lazy population of the device-info singleton.
    const auto & info = ggml_cuda_info();
    if (info.device_count == 0) {
        fprintf(stderr, "ds4_mmq_init: no CUDA devices found\n");
        return -1;
    }
    if (device >= info.device_count) {
        fprintf(stderr, "ds4_mmq_init: device %d out of range (have %d)\n",
                device, info.device_count);
        return -1;
    }
    return 0;
}

// ----------------------------------------------------------------------------
// Gating: when should the caller choose mmq over dequant+cublas?
//
// Body lifted verbatim from llama.cpp's ggml/src/ggml-cuda/mmq.cu:267-372
// (we do not vendor mmq.cu itself, since its other half talks to ggml_tensor
// and ggml_backend internals we don't carry over).
// ----------------------------------------------------------------------------

static bool ds4_should_use_mmq_impl(enum ggml_type type, int cc, int64_t ne11, int64_t n_experts) {
#ifdef GGML_CUDA_FORCE_CUBLAS
    GGML_UNUSED(type); GGML_UNUSED(cc); GGML_UNUSED(ne11); GGML_UNUSED(n_experts);
    return false;
#endif

    bool mmq_supported;
    switch (type) {
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_NVFP4:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ4_NL:
            mmq_supported = true;
            break;
        default:
            mmq_supported = false;
            break;
    }
    if (!mmq_supported) return false;

    if (turing_mma_available(cc)) {
        return true;
    }
    if (ggml_cuda_highest_compiled_arch(cc) < GGML_CUDA_CC_DP4A) {
        return false;
    }
#ifdef GGML_CUDA_FORCE_MMQ
    GGML_UNUSED(ne11); GGML_UNUSED(n_experts);
    return true;
#endif

    if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
        return !fp16_mma_hardware_available(cc) || ne11 < MMQ_DP4A_MAX_BATCH_SIZE;
    }
    if (amd_mfma_available(cc)) {
        if (GGML_CUDA_CC_IS_CDNA3(cc)) return true;
        if (n_experts > 64 || ne11 <= 128) return true;
        if (type == GGML_TYPE_Q4_0 || type == GGML_TYPE_Q4_1 ||
            type == GGML_TYPE_Q5_0 || type == GGML_TYPE_Q5_1) return true;
        if (ne11 <= 256 && (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K)) return true;
        return false;
    }
    if (amd_wmma_available(cc)) {
        if (GGML_CUDA_CC_IS_RDNA3(cc)) {
            if (n_experts >= 64) return true;
            switch (type) {
                case GGML_TYPE_Q2_K: return ne11 <= 128;
                case GGML_TYPE_Q6_K: return ne11 <= (GGML_CUDA_CC_IS_RDNA3_0(cc) ? 128 : 256);
                case GGML_TYPE_IQ2_XS:
                case GGML_TYPE_IQ2_S:
                    return GGML_CUDA_CC_IS_RDNA3_5(cc) || ne11 <= 128;
                default: return true;
            }
        }
        return true;
    }
    return (!GGML_CUDA_CC_IS_CDNA(cc)) || ne11 < MMQ_DP4A_MAX_BATCH_SIZE;
}

extern "C" int ds4_mmq_should_use(int type_x, int64_t ne11, int64_t n_experts) {
    const int dev = ggml_cuda_get_device();
    const int cc  = ggml_cuda_info().devices[dev].cc;
    const enum ggml_type t = (enum ggml_type) type_x;
    return ds4_should_use_mmq_impl(t, cc, ne11, n_experts) ? 1 : 0;
}

// ----------------------------------------------------------------------------
// Dense matmul implementation, shared across all three quant types.
//
// Computes  out[col, row] = sum_k W[row, k] * X[k, col]   with W in the
// type-specific block layout and X / out in F32 (X K-innermost row-major,
// out column-major out[col*M + row]).
//
// Mirrors upstream mmq.cu:154-159 (the no-ids branch) but builds mmq_args
// from plain pointers + shape ints instead of ggml_tensor introspection.
// ----------------------------------------------------------------------------

// Per-device singleton context. Owns the pool for stream-K fixup scratch.
// Phase 4 will make this per-stream as well; for now a single context per
// device is sufficient for the dense path.
namespace {

ggml_backend_cuda_context * get_ctx_for_device(int device) {
    static ggml_backend_cuda_context * cached[GGML_CUDA_MAX_DEVICES] = {};
    if (device < 0 || device >= GGML_CUDA_MAX_DEVICES) return nullptr;
    if (!cached[device]) {
        cached[device] = new ggml_backend_cuda_context(device);
    }
    return cached[device];
}

template <ggml_type type>
int ds4_mmq_dense_impl(
        const char  * tag,
        const void  * W,
        const float * X_f32,
        float       * out_f32,
        int           M,
        int           N,
        int           K,
        cudaStream_t  stream) {

    if (!W || !X_f32 || !out_f32) {
        fprintf(stderr, "%s: null pointer\n", tag);
        return -1;
    }
    if (K <= 0 || M <= 0 || N <= 0) {
        fprintf(stderr, "%s: bad shape M=%d N=%d K=%d\n", tag, M, N, K);
        return -1;
    }
    if (K % 256 != 0) {
        // mmq requires K to be a multiple of the largest super-block size
        // it sees during the inner tile loop, which is QK_K=256.
        fprintf(stderr, "%s: K=%d must be a multiple of 256\n", tag, K);
        return -1;
    }

    const int dev = ggml_cuda_get_device();
    const int cc  = ggml_cuda_info().devices[dev].cc;

    ggml_backend_cuda_context * ctx = get_ctx_for_device(dev);
    if (!ctx) {
        fprintf(stderr, "%s: failed to get cuda context for device %d\n", tag, dev);
        return -1;
    }

    // 1. Quantize the F32 activation into the mmq Q8_1 format. The
    //    target_type parameter only affects the activation scale strategy
    //    that the quantizer picks (matched to the weight type's K-block
    //    layout); the output buffer is always Q8_1.
    const int64_t ne00         = K;
    const int64_t ne10_padded  = GGML_PAD((int64_t)K, MATRIX_ROW_PADDING);
    const int64_t ne11         = N;
    const int64_t ne12         = 1;
    const int64_t ne13         = 1;

    const size_t nbytes_src1_q8_1 =
        ne13 * ne12 * ne11 * ne10_padded * sizeof(block_q8_1) / QK8_1 +
        get_mmq_x_max_host(cc) * sizeof(block_q8_1_mmq);

    ggml_cuda_pool_alloc<char> src1_q8_1(ctx->pool(), nbytes_src1_q8_1);

    quantize_mmq_q8_1_cuda(
        X_f32, /*ids=*/nullptr, (void *)src1_q8_1.get(),
        type, /*ne00=*/K, /*s11=*/(int64_t)K, /*s12=*/0, /*s13=*/0,
        /*ne0=*/ne10_padded, /*ne1=*/ne11, /*ne2=*/ne12, /*ne3=*/ne13,
        stream);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: quantize failed: %s\n", tag, cudaGetErrorString(err));
        return -2;
    }

    // 2. Build mmq_args. stride_row_x is in WEIGHT BLOCKS per row, which
    //    is K / blck_size(type). Q8_0 has block size 32; Q2_K and IQ2_XXS
    //    are K-quants with block size 256.
    const int64_t blck   = ggml_blck_size(type);
    const int64_t s01    = (int64_t)K / blck;
    const int64_t s1     = (int64_t)M;
    const int64_t s12    = ne11 * ne10_padded * sizeof(block_q8_1) / (QK8_1 * sizeof(int));
    const int64_t s13    = ne12 * s12;

    const bool use_stream_k =
        (GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_VOLTA) ||
        GGML_CUDA_CC_IS_CDNA(cc);

    const mmq_args args = {
        /*x=*/(const char *)W,
        /*type_x=*/type,
        /*y=*/(const int *)src1_q8_1.get(),
        /*ids_dst=*/nullptr,
        /*expert_bounds=*/nullptr,
        /*dst=*/out_f32,
        /*ncols_x=*/ne00,    /*nrows_x=*/(int64_t)M,    /*ncols_dst=*/ne11,
        /*stride_row_x=*/s01,/*ncols_y=*/ne11,          /*nrows_dst=*/s1,
        /*nchannels_x=*/1,   /*nchannels_y=*/1,
        /*stride_channel_x=*/0, /*stride_channel_y=*/s12, /*stride_channel_dst=*/0,
        /*nsamples_x=*/1,    /*nsamples_y=*/1,
        /*stride_sample_x=*/0, /*stride_sample_y=*/s13, /*stride_sample_dst=*/0,
        /*use_stream_k=*/use_stream_k,
        /*ncols_max=*/ne11,
    };

    mul_mat_q_case<type>(*ctx, args, stream);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: mul_mat_q_case launch failed: %s\n", tag, cudaGetErrorString(err));
        return -3;
    }
    return 0;
}

} // anonymous namespace

extern "C" int ds4_mmq_q8_0_dense(
        const void * W, const float * X, float * out,
        int M, int N, int K, cudaStream_t stream) {
    return ds4_mmq_dense_impl<GGML_TYPE_Q8_0>("ds4_mmq_q8_0_dense", W, X, out, M, N, K, stream);
}

extern "C" int ds4_mmq_q2_K_dense(
        const void * W, const float * X, float * out,
        int M, int N, int K, cudaStream_t stream) {
    return ds4_mmq_dense_impl<GGML_TYPE_Q2_K>("ds4_mmq_q2_K_dense", W, X, out, M, N, K, stream);
}

extern "C" int ds4_mmq_iq2_xxs_dense(
        const void * W, const float * X, float * out,
        int M, int N, int K, cudaStream_t stream) {
    return ds4_mmq_dense_impl<GGML_TYPE_IQ2_XXS>("ds4_mmq_iq2_xxs_dense", W, X, out, M, N, K, stream);
}

extern "C" int ds4_mmq_q4_K_dense(
        const void * W, const float * X, float * out,
        int M, int N, int K, cudaStream_t stream) {
    return ds4_mmq_dense_impl<GGML_TYPE_Q4_K>("ds4_mmq_q4_K_dense", W, X, out, M, N, K, stream);
}

// ----------------------------------------------------------------------------
// MoE matmul implementation, shared across all three quant types.
//
// Mirrors upstream mmq.cu:163-222 (the ids != nullptr branch).  Caller
// provides:
//   - per-expert weights stacked contiguously
//   - per-token activations [n_tokens, K]
//   - routing table ids[t, s] = expert id
// The wrapper invokes:
//   1. ggml_cuda_launch_mm_ids_helper to build (ids_src1, ids_dst,
//      expert_bounds) - permutations that sort assignments by expert.
//   2. quantize_mmq_q8_1_cuda with ids_src1 - gathers and quantizes the
//      activation into the expert-major flat layout.
//   3. mul_mat_q_case<type> with ids_dst + expert_bounds - the matmul.
// ----------------------------------------------------------------------------

namespace {

template <ggml_type type>
int ds4_mmq_moe_impl(
        const char    * tag,
        const void    * W,
        const float   * X_f32,
        const int32_t * ids,
        float         * out_f32,
        int             M,
        int             K,
        int             n_tokens,
        int             n_experts,
        int             n_expert_used,
        cudaStream_t    stream) {

    if (!W || !X_f32 || !ids || !out_f32) {
        fprintf(stderr, "%s: null pointer\n", tag);
        return -1;
    }
    if (M <= 0 || K <= 0 || n_tokens <= 0 || n_experts <= 0 || n_expert_used <= 0) {
        fprintf(stderr, "%s: bad shape M=%d K=%d ntok=%d nexp=%d nused=%d\n",
                tag, M, K, n_tokens, n_experts, n_expert_used);
        return -1;
    }
    if (K % 256 != 0) {
        fprintf(stderr, "%s: K=%d must be a multiple of 256\n", tag, K);
        return -1;
    }
    if (n_expert_used > n_experts) {
        fprintf(stderr, "%s: n_expert_used=%d > n_experts=%d\n", tag, n_expert_used, n_experts);
        return -1;
    }

    const int dev = ggml_cuda_get_device();
    const int cc  = ggml_cuda_info().devices[dev].cc;

    ggml_backend_cuda_context * ctx = get_ctx_for_device(dev);
    if (!ctx) {
        fprintf(stderr, "%s: failed to get cuda context for device %d\n", tag, dev);
        return -1;
    }

    const int64_t ne_get_rows  = (int64_t)n_tokens * n_expert_used;
    const int64_t ne00         = K;
    const int64_t ne10_padded  = GGML_PAD((int64_t)K, MATRIX_ROW_PADDING);
    const int64_t ne11         = 1;             // src1 rows per channel (one per token)
    const int64_t ne12         = n_tokens;      // src1 channels (= tokens)
    const int64_t blck         = ggml_blck_size(type);
    const int64_t s01          = (int64_t)K / blck;
    const int64_t s02          = (int64_t)M * s01;   // per-expert weight stride in blocks

    // 1. Build the expert-major work map.
    ggml_cuda_pool_alloc<int32_t> ids_src1(ctx->pool(), ne_get_rows);
    ggml_cuda_pool_alloc<int32_t> ids_dst(ctx->pool(), ne_get_rows);
    ggml_cuda_pool_alloc<int32_t> expert_bounds(ctx->pool(), n_experts + 1);

    // si1 = stride between tokens in the ids tensor, in elements. Our ids is
    // contiguous [n_tokens, n_expert_used] so si1 = n_expert_used.
    // sis1 = stride between src1 channels in row-units. With ne11=1, sis1=1
    //        means each "channel" of src1 is one row of K floats.
    const int si1  = n_expert_used;
    const int sis1 = 1;

    ggml_cuda_launch_mm_ids_helper(
        ids, ids_src1.get(), ids_dst.get(), expert_bounds.get(),
        n_experts, n_tokens, n_expert_used, /*nchannels_y=*/(int)ne11, si1, sis1, stream);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: mm_ids_helper failed: %s\n", tag, cudaGetErrorString(err));
        return -2;
    }

    // 2. Gather + quantize the activation into Q8_1.
    const size_t nbytes_src1_q8_1 =
        ne_get_rows * ne10_padded * sizeof(block_q8_1) / QK8_1 +
        get_mmq_x_max_host(cc) * sizeof(block_q8_1_mmq);
    ggml_cuda_pool_alloc<char> src1_q8_1(ctx->pool(), nbytes_src1_q8_1);

    // src1 logical [K, ne11=1, ne12=n_tokens, ne13=1] - K innermost, then
    // one row per channel, channels = tokens.
    const int64_t s11_src = (int64_t)K;                                 // stride between rows of a channel
    const int64_t s12_src = (int64_t)K * ne11;                          // stride between channels = K*1
    const int64_t s13_src = (int64_t)K * ne11 * ne12;                   // stride between samples

    quantize_mmq_q8_1_cuda(
        X_f32, ids_src1.get(), (void *)src1_q8_1.get(),
        type, /*ne00=*/K, s11_src, s12_src, s13_src,
        /*ne0=*/ne10_padded, /*ne1=*/ne_get_rows, /*ne2=*/1, /*ne3=*/1,
        stream);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: quantize_mmq_q8_1_cuda failed: %s\n", tag, cudaGetErrorString(err));
        return -3;
    }

    // 3. Build mmq_args for the MoE path.
    //
    // dst layout convention matches upstream's MoE branch
    // (mmq.cu:215-220): dst is interpreted as [M, n_expert_used, n_tokens]
    // with M innermost and n_expert_used as the second dim that mmq writes
    // through ids_dst.  s1 = M (the column stride in the flat dst buffer
    // mmq writes into).  The output is column-major: out[col*M + row].
    const int64_t s1            = (int64_t)M;
    // stride_channel_y per upstream: ne11 * ne10_padded * sizeof(block_q8_1)
    //                                     / (QK8_1 * sizeof(int))
    // In MoE mode the kernel zeroes out the channel-stride contribution to
    // offset_y after reading expert_bounds, so the value is permissive -
    // but we set it consistently with upstream.
    const int64_t s12_mmq = ne11 * ne10_padded * sizeof(block_q8_1) / (QK8_1 * sizeof(int));
    const int64_t s13_mmq = ne12 * s12_mmq;

    const bool use_stream_k =
        (GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_VOLTA) ||
        GGML_CUDA_CC_IS_CDNA(cc);

    const mmq_args args = {
        /*x=*/(const char *)W,
        /*type_x=*/type,
        /*y=*/(const int *)src1_q8_1.get(),
        /*ids_dst=*/ids_dst.get(),
        /*expert_bounds=*/expert_bounds.get(),
        /*dst=*/out_f32,
        /*ncols_x=*/ne00,
        /*nrows_x=*/(int64_t)M,
        /*ncols_dst=*/ne_get_rows,
        /*stride_row_x=*/s01,
        /*ncols_y=*/ne_get_rows,
        /*nrows_dst=*/s1,
        /*nchannels_x=*/(int64_t)n_experts,
        /*nchannels_y=*/(int64_t)n_experts,
        /*stride_channel_x=*/s02,
        /*stride_channel_y=*/s12_mmq,
        /*stride_channel_dst=*/(int64_t)0,
        /*nsamples_x=*/1,
        /*nsamples_y=*/1,
        /*stride_sample_x=*/0,
        /*stride_sample_y=*/s13_mmq,
        /*stride_sample_dst=*/0,
        /*use_stream_k=*/use_stream_k,
        /*ncols_max=*/(int64_t)n_tokens,
    };

    mul_mat_q_case<type>(*ctx, args, stream);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: mul_mat_q_case (moe) launch failed: %s\n", tag, cudaGetErrorString(err));
        return -4;
    }
    return 0;
}

// Paired MoE: one helper + one quantize covers both weights.  See the
// header comment on ds4_mmq_iq2_xxs_moe_pair for motivation.  Internal
// structure mirrors ds4_mmq_moe_impl above; the only differences are the
// two W pointers, the two output pointers, and the second mul_mat_q_case
// launch with a fresh (x, dst) pair.
template <ggml_type type>
int ds4_mmq_moe_pair_impl(
        const char    * tag,
        const void    * W_a,
        const void    * W_b,
        const float   * X_f32,
        const int32_t * ids,
        float         * out_a,
        float         * out_b,
        int             M,
        int             K,
        int             n_tokens,
        int             n_experts,
        int             n_expert_used,
        cudaStream_t    stream) {

    if (!W_a || !W_b || !X_f32 || !ids || !out_a || !out_b) {
        fprintf(stderr, "%s: null pointer\n", tag);
        return -1;
    }
    if (M <= 0 || K <= 0 || n_tokens <= 0 || n_experts <= 0 || n_expert_used <= 0) {
        fprintf(stderr, "%s: bad shape M=%d K=%d ntok=%d nexp=%d nused=%d\n",
                tag, M, K, n_tokens, n_experts, n_expert_used);
        return -1;
    }
    if (K % 256 != 0) {
        fprintf(stderr, "%s: K=%d must be a multiple of 256\n", tag, K);
        return -1;
    }
    if (n_expert_used > n_experts) {
        fprintf(stderr, "%s: n_expert_used=%d > n_experts=%d\n", tag, n_expert_used, n_experts);
        return -1;
    }

    const int dev = ggml_cuda_get_device();
    const int cc  = ggml_cuda_info().devices[dev].cc;

    ggml_backend_cuda_context * ctx = get_ctx_for_device(dev);
    if (!ctx) {
        fprintf(stderr, "%s: failed to get cuda context for device %d\n", tag, dev);
        return -1;
    }

    const int64_t ne_get_rows  = (int64_t)n_tokens * n_expert_used;
    const int64_t ne00         = K;
    const int64_t ne10_padded  = GGML_PAD((int64_t)K, MATRIX_ROW_PADDING);
    const int64_t ne11         = 1;
    const int64_t ne12         = n_tokens;
    const int64_t blck         = ggml_blck_size(type);
    const int64_t s01          = (int64_t)K / blck;
    const int64_t s02          = (int64_t)M * s01;

    ggml_cuda_pool_alloc<int32_t> ids_src1(ctx->pool(), ne_get_rows);
    ggml_cuda_pool_alloc<int32_t> ids_dst(ctx->pool(), ne_get_rows);
    ggml_cuda_pool_alloc<int32_t> expert_bounds(ctx->pool(), n_experts + 1);

    const int si1  = n_expert_used;
    const int sis1 = 1;

    ggml_cuda_launch_mm_ids_helper(
        ids, ids_src1.get(), ids_dst.get(), expert_bounds.get(),
        n_experts, n_tokens, n_expert_used, /*nchannels_y=*/(int)ne11, si1, sis1, stream);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: mm_ids_helper failed: %s\n", tag, cudaGetErrorString(err));
        return -2;
    }

    const size_t nbytes_src1_q8_1 =
        ne_get_rows * ne10_padded * sizeof(block_q8_1) / QK8_1 +
        get_mmq_x_max_host(cc) * sizeof(block_q8_1_mmq);
    ggml_cuda_pool_alloc<char> src1_q8_1(ctx->pool(), nbytes_src1_q8_1);

    const int64_t s11_src = (int64_t)K;
    const int64_t s12_src = (int64_t)K * ne11;
    const int64_t s13_src = (int64_t)K * ne11 * ne12;

    quantize_mmq_q8_1_cuda(
        X_f32, ids_src1.get(), (void *)src1_q8_1.get(),
        type, /*ne00=*/K, s11_src, s12_src, s13_src,
        /*ne0=*/ne10_padded, /*ne1=*/ne_get_rows, /*ne2=*/1, /*ne3=*/1,
        stream);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: quantize_mmq_q8_1_cuda failed: %s\n", tag, cudaGetErrorString(err));
        return -3;
    }

    const int64_t s1      = (int64_t)M;
    const int64_t s12_mmq = ne11 * ne10_padded * sizeof(block_q8_1) / (QK8_1 * sizeof(int));
    const int64_t s13_mmq = ne12 * s12_mmq;

    const bool use_stream_k =
        (GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_VOLTA) ||
        GGML_CUDA_CC_IS_CDNA(cc);

    mmq_args args = {
        /*x=*/(const char *)W_a,
        /*type_x=*/type,
        /*y=*/(const int *)src1_q8_1.get(),
        /*ids_dst=*/ids_dst.get(),
        /*expert_bounds=*/expert_bounds.get(),
        /*dst=*/out_a,
        /*ncols_x=*/ne00,
        /*nrows_x=*/(int64_t)M,
        /*ncols_dst=*/ne_get_rows,
        /*stride_row_x=*/s01,
        /*ncols_y=*/ne_get_rows,
        /*nrows_dst=*/s1,
        /*nchannels_x=*/(int64_t)n_experts,
        /*nchannels_y=*/(int64_t)n_experts,
        /*stride_channel_x=*/s02,
        /*stride_channel_y=*/s12_mmq,
        /*stride_channel_dst=*/(int64_t)0,
        /*nsamples_x=*/1,
        /*nsamples_y=*/1,
        /*stride_sample_x=*/0,
        /*stride_sample_y=*/s13_mmq,
        /*stride_sample_dst=*/0,
        /*use_stream_k=*/use_stream_k,
        /*ncols_max=*/(int64_t)n_tokens,
    };

    mul_mat_q_case<type>(*ctx, args, stream);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: mul_mat_q_case (pair a) launch failed: %s\n", tag, cudaGetErrorString(err));
        return -4;
    }

    // Second matmul over the same activation buffer and same routing map.
    args.x   = (const char *)W_b;
    args.dst = out_b;
    mul_mat_q_case<type>(*ctx, args, stream);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: mul_mat_q_case (pair b) launch failed: %s\n", tag, cudaGetErrorString(err));
        return -5;
    }
    return 0;
}

} // anonymous namespace

extern "C" int ds4_mmq_q8_0_moe(
        const void * W, const float * X, const int32_t * ids, float * out,
        int M, int K, int n_tokens, int n_experts, int n_expert_used,
        cudaStream_t stream) {
    return ds4_mmq_moe_impl<GGML_TYPE_Q8_0>("ds4_mmq_q8_0_moe", W, X, ids, out, M, K,
                                            n_tokens, n_experts, n_expert_used, stream);
}

extern "C" int ds4_mmq_q2_K_moe(
        const void * W, const float * X, const int32_t * ids, float * out,
        int M, int K, int n_tokens, int n_experts, int n_expert_used,
        cudaStream_t stream) {
    return ds4_mmq_moe_impl<GGML_TYPE_Q2_K>("ds4_mmq_q2_K_moe", W, X, ids, out, M, K,
                                            n_tokens, n_experts, n_expert_used, stream);
}

extern "C" int ds4_mmq_iq2_xxs_moe(
        const void * W, const float * X, const int32_t * ids, float * out,
        int M, int K, int n_tokens, int n_experts, int n_expert_used,
        cudaStream_t stream) {
    return ds4_mmq_moe_impl<GGML_TYPE_IQ2_XXS>("ds4_mmq_iq2_xxs_moe", W, X, ids, out, M, K,
                                               n_tokens, n_experts, n_expert_used, stream);
}

extern "C" int ds4_mmq_q4_K_moe(
        const void * W, const float * X, const int32_t * ids, float * out,
        int M, int K, int n_tokens, int n_experts, int n_expert_used,
        cudaStream_t stream) {
    return ds4_mmq_moe_impl<GGML_TYPE_Q4_K>("ds4_mmq_q4_K_moe", W, X, ids, out, M, K,
                                            n_tokens, n_experts, n_expert_used, stream);
}

extern "C" int ds4_mmq_iq2_xxs_moe_pair(
        const void * W_a, const void * W_b,
        const float * X, const int32_t * ids, float * out_a, float * out_b,
        int M, int K, int n_tokens, int n_experts, int n_expert_used,
        cudaStream_t stream) {
    return ds4_mmq_moe_pair_impl<GGML_TYPE_IQ2_XXS>(
        "ds4_mmq_iq2_xxs_moe_pair", W_a, W_b, X, ids, out_a, out_b,
        M, K, n_tokens, n_experts, n_expert_used, stream);
}

extern "C" int ds4_mmq_q4_K_moe_pair(
        const void * W_a, const void * W_b,
        const float * X, const int32_t * ids, float * out_a, float * out_b,
        int M, int K, int n_tokens, int n_experts, int n_expert_used,
        cudaStream_t stream) {
    return ds4_mmq_moe_pair_impl<GGML_TYPE_Q4_K>(
        "ds4_mmq_q4_K_moe_pair", W_a, W_b, X, ids, out_a, out_b,
        M, K, n_tokens, n_experts, n_expert_used, stream);
}

// ----------------------------------------------------------------------------
// mmvq-backed entry points (Step 6 of the optimization plan).
//
// mmvq is upstream's matrix-vector matmul family, optimised for the
// n_tokens <= MMVQ_MAX_BATCH_SIZE=8 regime. Unlike mmq it consumes the
// CANONICAL block_q8_1 layout (via quantize_row_q8_1_cuda), not the
// interleaved block_q8_1_mmq that quantize_mmq_q8_1_cuda produces.
//
// The single-W _moe_vec entries cover:
//   - the down matmul at decode (treating [n_tokens=1, n_expert_used=6]
//     as [n_tokens=6, n_expert_used=1])
//   - dense attention projections at decode (n_tokens=1, no MoE)
//   - any small-batch path where mmvq's per-token grid wins over mmq's
//     tile-based approach
//
// The pair-fused _moe_pair_vec entries cover the gate+up matmuls at
// decode using mmvq's built-in fusion. fusion.gate is the up_w pointer
// and fusion.glu_op is GGML_GLU_OP_SWIGLU - the kernel computes
// silu(gate@x) * (up@x) in a single launch. mmvq's fusion is supported
// only at ncols_dst=1, so n_tokens=1 is the only valid case.
// ----------------------------------------------------------------------------

#include "mmvq.cuh"

namespace {

template <ggml_type type>
int ds4_mmq_moe_vec_impl(
        const char    * tag,
        const void    * W,
        const float   * X_f32,
        const int32_t * ids,
        float         * out_f32,
        int             M,
        int             K,
        int             n_tokens,
        int             n_experts,
        int             n_expert_used,
        cudaStream_t    stream) {

    if (!W || !X_f32 || !ids || !out_f32) {
        fprintf(stderr, "%s: null pointer\n", tag);
        return -1;
    }
    if (M <= 0 || K <= 0 || n_tokens <= 0 || n_experts <= 0 || n_expert_used <= 0) {
        fprintf(stderr, "%s: bad shape M=%d K=%d ntok=%d nexp=%d nused=%d\n",
                tag, M, K, n_tokens, n_experts, n_expert_used);
        return -1;
    }
    if (K % 256 != 0) {
        fprintf(stderr, "%s: K=%d must be a multiple of 256\n", tag, K);
        return -1;
    }
    if (n_expert_used > n_experts) {
        fprintf(stderr, "%s: n_expert_used=%d > n_experts=%d\n", tag, n_expert_used, n_experts);
        return -1;
    }
    // mmvq's per-arch batch cap. ncols_dst as computed below is
    // max(n_tokens, n_expert_used) depending on which dim we route into.
    // We follow upstream's convention: ne_y = n_tokens, ne_dst = n_expert_used.
    // So ncols_dst = n_tokens and nchannels_dst = n_expert_used.
    if (n_tokens > MMVQ_MAX_BATCH_SIZE) {
        fprintf(stderr, "%s: n_tokens=%d exceeds MMVQ_MAX_BATCH_SIZE=%d\n",
                tag, n_tokens, MMVQ_MAX_BATCH_SIZE);
        return -1;
    }

    const int dev = ggml_cuda_get_device();
    ggml_backend_cuda_context * ctx = get_ctx_for_device(dev);
    if (!ctx) {
        fprintf(stderr, "%s: failed to get cuda context for device %d\n", tag, dev);
        return -1;
    }

    // 1. Quantize X into CANONICAL Q8_1 (NOT the MMQ-interleaved variant).
    //    Layout: [ne13=1, ne12=n_tokens, ne11=1, ne10_padded blocks].
    const int64_t ne10_padded = GGML_PAD((int64_t)K, MATRIX_ROW_PADDING);
    const size_t  nbytes_q8_1 = (size_t)n_tokens * ne10_padded *
                                sizeof(block_q8_1) / QK8_1;
    ggml_cuda_pool_alloc<char> src1_q8_1(ctx->pool(), nbytes_q8_1);

    // s11 = stride between rows of an src1 channel in source-float units.
    //       Logical src1 [K, ne11=1, ne12=n_tokens, ne13=1] - K innermost.
    // s12 = stride between channels = K * ne11 = K.
    // s13 = stride between samples = K * ne11 * ne12 = K * n_tokens.
    quantize_row_q8_1_cuda(
        X_f32, /*ids=*/nullptr, (void *)src1_q8_1.get(),
        type, /*ne00=*/K,
        /*s11=*/(int64_t)K, /*s12=*/(int64_t)K, /*s13=*/(int64_t)K * n_tokens,
        /*ne0=*/ne10_padded, /*ne1=*/1, /*ne2=*/n_tokens, /*ne3=*/1,
        stream);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: quantize_row_q8_1_cuda failed: %s\n",
                tag, cudaGetErrorString(err));
        return -2;
    }

    // 2. mmvq stride setup. Mirror upstream's ggml_cuda_mul_mat_vec_q
    //    dispatch (mmvq.cu:1101-1136).
    //
    //    For MoE (ids != nullptr): per the dispatch math at line 1121-1130,
    //      ncols_dst          = ne2  = n_tokens
    //      nchannels_y        = ne11 = 1
    //      nchannels_dst      = ne1  = n_expert_used
    //      stride_col_y       = s12  = ne11 * (ne10_padded / QK8_1)
    //      stride_col_dst     = s2   = M (column stride in dst)
    //      stride_channel_y   = s11  = ne10_padded / QK8_1
    //      stride_channel_dst = s1   = M (channel stride in dst)
    //      ids_stride         = stride between rows of ids[] tensor
    const int64_t blck      = ggml_blck_size(type);
    const int64_t s01_row   = (int64_t)K / blck;            // weight row stride in blocks
    const int64_t s02_chan  = (int64_t)M * s01_row;         // expert-stack stride
    const int64_t s11_y     = ne10_padded / QK8_1;          // src1 channel stride in blocks
    const int64_t s12_y     = (int64_t)1 * s11_y;           // ne11 * s11
    const int64_t s1_dst    = (int64_t)M;                   // dst col stride

    // ids_stride: stride between rows of the ids tensor in int32 elements.
    // Caller passes ids[t * n_expert_used + s], so stride between tokens
    // is n_expert_used.
    const int ids_stride = n_expert_used;

    ggml_cuda_mm_fusion_args_device fusion = {};

    mul_mat_vec_q_switch_type(
        /*vx=*/W, /*type_x=*/type,
        /*vy=*/(const void *)src1_q8_1.get(),
        /*ids=*/ids, /*fusion=*/fusion,
        /*dst=*/out_f32,
        /*ncols_x=*/K, /*nrows_x=*/M, /*ncols_dst=*/n_tokens,
        /*stride_row_x=*/(int)s01_row,
        /*stride_col_y=*/(int)s12_y,
        /*stride_col_dst=*/(int)s1_dst,
        /*nchannels_x=*/n_experts,
        /*nchannels_y=*/1,
        /*nchannels_dst=*/n_expert_used,
        /*stride_channel_x=*/(int)s02_chan,
        /*stride_channel_y=*/(int)s11_y,
        /*stride_channel_dst=*/(int)s1_dst,
        /*nsamples_x=*/1, /*nsamples_dst=*/1,
        /*stride_sample_x=*/0, /*stride_sample_y=*/0, /*stride_sample_dst=*/0,
        /*ids_stride=*/ids_stride, stream);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: mul_mat_vec_q_switch_type launch failed: %s\n",
                tag, cudaGetErrorString(err));
        return -3;
    }
    return 0;
}

template <ggml_type type>
int ds4_mmq_moe_pair_vec_impl(
        const char    * tag,
        const void    * W_a,
        const void    * W_b,
        const float   * X_f32,
        const int32_t * ids,
        float         * out_silu,
        int             M,
        int             K,
        int             n_experts,
        int             n_expert_used,
        cudaStream_t    stream) {

    if (!W_a || !W_b || !X_f32 || !ids || !out_silu) {
        fprintf(stderr, "%s: null pointer\n", tag);
        return -1;
    }
    if (M <= 0 || K <= 0 || n_experts <= 0 || n_expert_used <= 0) {
        fprintf(stderr, "%s: bad shape M=%d K=%d nexp=%d nused=%d\n",
                tag, M, K, n_experts, n_expert_used);
        return -1;
    }
    if (K % 256 != 0) {
        fprintf(stderr, "%s: K=%d must be a multiple of 256\n", tag, K);
        return -1;
    }
    if (n_expert_used > n_experts) {
        fprintf(stderr, "%s: n_expert_used=%d > n_experts=%d\n", tag, n_expert_used, n_experts);
        return -1;
    }

    const int dev = ggml_cuda_get_device();
    ggml_backend_cuda_context * ctx = get_ctx_for_device(dev);
    if (!ctx) {
        fprintf(stderr, "%s: failed to get cuda context for device %d\n", tag, dev);
        return -1;
    }

    const int n_tokens = 1;  // fusion only supported at ncols_dst=1.

    // Quantize X (single token) into canonical Q8_1.
    const int64_t ne10_padded = GGML_PAD((int64_t)K, MATRIX_ROW_PADDING);
    const size_t  nbytes_q8_1 = (size_t)n_tokens * ne10_padded *
                                sizeof(block_q8_1) / QK8_1;
    ggml_cuda_pool_alloc<char> src1_q8_1(ctx->pool(), nbytes_q8_1);

    quantize_row_q8_1_cuda(
        X_f32, /*ids=*/nullptr, (void *)src1_q8_1.get(),
        type, /*ne00=*/K,
        /*s11=*/(int64_t)K, /*s12=*/(int64_t)K, /*s13=*/(int64_t)K * n_tokens,
        /*ne0=*/ne10_padded, /*ne1=*/1, /*ne2=*/n_tokens, /*ne3=*/1,
        stream);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: quantize_row_q8_1_cuda failed: %s\n",
                tag, cudaGetErrorString(err));
        return -2;
    }

    const int64_t blck      = ggml_blck_size(type);
    const int64_t s01_row   = (int64_t)K / blck;
    const int64_t s02_chan  = (int64_t)M * s01_row;
    const int64_t s11_y     = ne10_padded / QK8_1;
    const int64_t s12_y     = (int64_t)1 * s11_y;
    const int64_t s1_dst    = (int64_t)M;
    const int ids_stride    = n_expert_used;

    // Configure fusion: gate=W_b (up weights), glu_op=SWIGLU.
    // mmvq's kernel will compute, for each (channel_dst, row):
    //   a = vec_dot(W_a, x); b = vec_dot(W_b, x);
    //   dst = silu(a) * b
    ggml_cuda_mm_fusion_args_device fusion = {};
    fusion.gate   = W_b;
    fusion.glu_op = GGML_GLU_OP_SWIGLU;

    mul_mat_vec_q_switch_type(
        /*vx=*/W_a, /*type_x=*/type,
        /*vy=*/(const void *)src1_q8_1.get(),
        /*ids=*/ids, /*fusion=*/fusion,
        /*dst=*/out_silu,
        /*ncols_x=*/K, /*nrows_x=*/M, /*ncols_dst=*/n_tokens,
        /*stride_row_x=*/(int)s01_row,
        /*stride_col_y=*/(int)s12_y,
        /*stride_col_dst=*/(int)s1_dst,
        /*nchannels_x=*/n_experts,
        /*nchannels_y=*/1,
        /*nchannels_dst=*/n_expert_used,
        /*stride_channel_x=*/(int)s02_chan,
        /*stride_channel_y=*/(int)s11_y,
        /*stride_channel_dst=*/(int)s1_dst,
        /*nsamples_x=*/1, /*nsamples_dst=*/1,
        /*stride_sample_x=*/0, /*stride_sample_y=*/0, /*stride_sample_dst=*/0,
        /*ids_stride=*/ids_stride, stream);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: mul_mat_vec_q_switch_type (fused) launch failed: %s\n",
                tag, cudaGetErrorString(err));
        return -3;
    }
    return 0;
}

template <ggml_type type>
int ds4_mmq_dense_vec_impl(
        const char  * tag,
        const void  * W,
        const float * X_f32,
        float       * out_f32,
        int           M,
        int           N,
        int           K,
        cudaStream_t  stream) {

    if (!W || !X_f32 || !out_f32) {
        fprintf(stderr, "%s: null pointer\n", tag);
        return -1;
    }
    if (M <= 0 || N <= 0 || K <= 0) {
        fprintf(stderr, "%s: bad shape M=%d N=%d K=%d\n", tag, M, N, K);
        return -1;
    }
    if (K % 256 != 0) {
        fprintf(stderr, "%s: K=%d must be a multiple of 256\n", tag, K);
        return -1;
    }
    if (N > MMVQ_MAX_BATCH_SIZE) {
        fprintf(stderr, "%s: N=%d exceeds MMVQ_MAX_BATCH_SIZE=%d\n",
                tag, N, MMVQ_MAX_BATCH_SIZE);
        return -1;
    }

    const int dev = ggml_cuda_get_device();
    ggml_backend_cuda_context * ctx = get_ctx_for_device(dev);
    if (!ctx) {
        fprintf(stderr, "%s: failed to get cuda context for device %d\n", tag, dev);
        return -1;
    }

    // Dense: no MoE, ids=null. Layout [K, N, 1, 1] for src1.
    const int64_t ne10_padded = GGML_PAD((int64_t)K, MATRIX_ROW_PADDING);
    const size_t  nbytes_q8_1 = (size_t)N * ne10_padded *
                                sizeof(block_q8_1) / QK8_1;
    ggml_cuda_pool_alloc<char> src1_q8_1(ctx->pool(), nbytes_q8_1);

    // Dense src1 layout: K innermost, N next; ne11=N, ne12=1, ne13=1.
    quantize_row_q8_1_cuda(
        X_f32, /*ids=*/nullptr, (void *)src1_q8_1.get(),
        type, /*ne00=*/K,
        /*s11=*/(int64_t)K, /*s12=*/(int64_t)K * N, /*s13=*/(int64_t)K * N,
        /*ne0=*/ne10_padded, /*ne1=*/N, /*ne2=*/1, /*ne3=*/1,
        stream);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: quantize_row_q8_1_cuda failed: %s\n",
                tag, cudaGetErrorString(err));
        return -2;
    }

    // Dense (no ids): per upstream dispatch (mmvq.cu:1121-1127),
    //   ncols_dst          = ne1  = N
    //   nchannels_y        = ne12 = 1
    //   nchannels_dst      = ne2  = 1
    //   stride_col_y       = s11  = ne10_padded / QK8_1
    //   stride_channel_y   = s12  = N * (ne10_padded / QK8_1)
    const int64_t blck      = ggml_blck_size(type);
    const int64_t s01_row   = (int64_t)K / blck;
    const int64_t s11_y     = ne10_padded / QK8_1;
    const int64_t s12_y     = (int64_t)N * s11_y;
    const int64_t s1_dst    = (int64_t)M;

    ggml_cuda_mm_fusion_args_device fusion = {};

    mul_mat_vec_q_switch_type(
        /*vx=*/W, /*type_x=*/type,
        /*vy=*/(const void *)src1_q8_1.get(),
        /*ids=*/nullptr, /*fusion=*/fusion,
        /*dst=*/out_f32,
        /*ncols_x=*/K, /*nrows_x=*/M, /*ncols_dst=*/N,
        /*stride_row_x=*/(int)s01_row,
        /*stride_col_y=*/(int)s11_y,
        /*stride_col_dst=*/(int)s1_dst,
        /*nchannels_x=*/1,
        /*nchannels_y=*/1,
        /*nchannels_dst=*/1,
        /*stride_channel_x=*/0,
        /*stride_channel_y=*/(int)s12_y,
        /*stride_channel_dst=*/0,
        /*nsamples_x=*/1, /*nsamples_dst=*/1,
        /*stride_sample_x=*/0, /*stride_sample_y=*/0, /*stride_sample_dst=*/0,
        /*ids_stride=*/0, stream);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: mul_mat_vec_q_switch_type (dense) launch failed: %s\n",
                tag, cudaGetErrorString(err));
        return -3;
    }
    return 0;
}

} // anonymous namespace

extern "C" int ds4_mmq_q8_0_moe_vec(
        const void * W, const float * X, const int32_t * ids, float * out,
        int M, int K, int n_tokens, int n_experts, int n_expert_used,
        cudaStream_t stream) {
    return ds4_mmq_moe_vec_impl<GGML_TYPE_Q8_0>(
        "ds4_mmq_q8_0_moe_vec", W, X, ids, out, M, K,
        n_tokens, n_experts, n_expert_used, stream);
}

extern "C" int ds4_mmq_q2_K_moe_vec(
        const void * W, const float * X, const int32_t * ids, float * out,
        int M, int K, int n_tokens, int n_experts, int n_expert_used,
        cudaStream_t stream) {
    return ds4_mmq_moe_vec_impl<GGML_TYPE_Q2_K>(
        "ds4_mmq_q2_K_moe_vec", W, X, ids, out, M, K,
        n_tokens, n_experts, n_expert_used, stream);
}

extern "C" int ds4_mmq_iq2_xxs_moe_vec(
        const void * W, const float * X, const int32_t * ids, float * out,
        int M, int K, int n_tokens, int n_experts, int n_expert_used,
        cudaStream_t stream) {
    return ds4_mmq_moe_vec_impl<GGML_TYPE_IQ2_XXS>(
        "ds4_mmq_iq2_xxs_moe_vec", W, X, ids, out, M, K,
        n_tokens, n_experts, n_expert_used, stream);
}

extern "C" int ds4_mmq_q4_K_moe_vec(
        const void * W, const float * X, const int32_t * ids, float * out,
        int M, int K, int n_tokens, int n_experts, int n_expert_used,
        cudaStream_t stream) {
    return ds4_mmq_moe_vec_impl<GGML_TYPE_Q4_K>(
        "ds4_mmq_q4_K_moe_vec", W, X, ids, out, M, K,
        n_tokens, n_experts, n_expert_used, stream);
}

extern "C" int ds4_mmq_iq2_xxs_moe_pair_vec(
        const void * W_a, const void * W_b,
        const float * X, const int32_t * ids, float * out_silu,
        int M, int K, int n_experts, int n_expert_used,
        cudaStream_t stream) {
    return ds4_mmq_moe_pair_vec_impl<GGML_TYPE_IQ2_XXS>(
        "ds4_mmq_iq2_xxs_moe_pair_vec", W_a, W_b, X, ids, out_silu,
        M, K, n_experts, n_expert_used, stream);
}

extern "C" int ds4_mmq_q4_K_moe_pair_vec(
        const void * W_a, const void * W_b,
        const float * X, const int32_t * ids, float * out_silu,
        int M, int K, int n_experts, int n_expert_used,
        cudaStream_t stream) {
    return ds4_mmq_moe_pair_vec_impl<GGML_TYPE_Q4_K>(
        "ds4_mmq_q4_K_moe_pair_vec", W_a, W_b, X, ids, out_silu,
        M, K, n_experts, n_expert_used, stream);
}

extern "C" int ds4_mmq_q8_0_dense_vec(
        const void * W, const float * X, float * out,
        int M, int N, int K, cudaStream_t stream) {
    return ds4_mmq_dense_vec_impl<GGML_TYPE_Q8_0>(
        "ds4_mmq_q8_0_dense_vec", W, X, out, M, N, K, stream);
}

// Explicit instantiations. One per quant type the public API exposes.
// Each instantiation drags in the load_tiles_<type> + vec_dot_<type>_*
// device functions from mmq.cuh, so the .o objects below contain everything
// needed to link against the public C entries.
template void mul_mat_q_case<GGML_TYPE_Q8_0>(
    ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream);
template void mul_mat_q_case<GGML_TYPE_Q2_K>(
    ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream);
template void mul_mat_q_case<GGML_TYPE_IQ2_XXS>(
    ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream);
template void mul_mat_q_case<GGML_TYPE_Q4_K>(
    ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream);
