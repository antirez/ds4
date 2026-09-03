/* Metal-4 TensorOps probe for the gathered-QSA attention shapes.
 *
 * Answers three questions before a production kernel is designed:
 *   1. Do threadgroup-staged matmul2d ops at the attention geometry
 *      (QK^T: M=16 padded query heads x N=64 gathered tokens x K=256
 *      chunked by 32; PV: M=16 x N=256 dims x K=64 tokens) validate
 *      against an exact CPU reference?
 *   2. Is a MIXED float-LEFT / half-RIGHT operand pair exact (the KV
 *      cache is BF16, which converts losslessly to F16; Q and P stay
 *      F32 so the historical F16-P rounding failure cannot recur), or
 *      does the half operand drag the products to binary16 class?
 *   3. What raw rate do the two chunked matmuls + staging achieve at a
 *      dispatch shaped like the production grid (no gather, broadcast
 *      inputs), as an upper-bound estimate for ms/layer?
 *
 * Operand layout contract used by the repo's NT-mode kernels (see
 * kernel_mul_mm_id_mpp and the dsv4 score tile): LEFT staged [m][k]
 * with k contiguous, RIGHT staged [n][k] with k contiguous, extents
 * (NK_chunk, rows); destination stored through an explicitly strided
 * threadgroup tensor with the M dim contiguous.
 */
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

constant constexpr int PR_M = 16;   /* padded query-head rows   */
constant constexpr int PR_D = 256;  /* head dim                 */
constant constexpr int PR_BK = 64;  /* gathered tokens per tile */
constant constexpr int PR_NK = 32;  /* k chunk                  */
constant constexpr int PR_T = 128;  /* threads (4 simdgroups)   */
constant constexpr int PR_NH = 128; /* PV N half (two of them)  */

/* NOTE: with a cooperative-tensor destination the two input operand
 * types must MATCH (static_assert in the driver headers), so a mixed
 * float/half operand pair is not expressible on the in-register path;
 * the exact route is all-fp32 staging.  half/half is kept as the
 * known-loose F16-class anchor. */
template <typename T>
kernel void t2d_qkpv(
        device const float *q [[buffer(0)]],   /* [M][D]  */
        device const float *kc [[buffer(1)]],  /* [BK][D] */
        device const float *p [[buffer(2)]],   /* [M][BK] */
        device const float *vc [[buffer(3)]],  /* [BK][D] */
        device float *so [[buffer(4)]],        /* [M][BK] */
        device float *oo [[buffer(5)]],        /* [M][D]  */
        device float *dbg [[buffer(6)]],
        threadgroup uchar *tgm [[threadgroup(0)]],
        uint tid [[thread_index_in_threadgroup]],
        uint2 tpg [[threadgroup_position_in_grid]]) {
    threadgroup T *qb = (threadgroup T *)tgm;                 /* M*NK   */
    threadgroup T *kb = (threadgroup T *)(qb + PR_M * PR_NK); /* BK*NK */
    threadgroup T *pb = (threadgroup T *)(kb + PR_BK * PR_NK); /* M*NK */
    threadgroup T *vb = (threadgroup T *)(pb + PR_M * PR_NK); /* NH*NK */
    threadgroup float *sb =
        (threadgroup float *)(vb + PR_NH * PR_NK);            /* M*BK   */

    /* ---- QK^T: S(M,BK) = Q(M,D) @ K(BK,D)^T ---- */
    if ((int)dbg[1] & 1) { /* skipped */ }
    else {
        auto tQ = tensor(qb, dextents<int32_t, 2>(PR_NK, PR_M));
        auto tK = tensor(kb, dextents<int32_t, 2>(PR_NK, PR_BK));
        matmul2d<
            matmul2d_descriptor(
                PR_M, PR_BK, PR_NK, false, true, false,
                matmul2d_descriptor::mode::multiply_accumulate),
            execution_simdgroups<4>> mm;
        auto cS =
            mm.get_destination_cooperative_tensor<decltype(tK),
                                                  decltype(tQ), float>();
        #pragma unroll
        for (uint16_t i = 0; i < cS.get_capacity(); ++i) {
            if (cS.is_valid_element(i)) cS[i] = 0.0f;
        }
        for (int c = 0; c < PR_D / PR_NK; ++c) {
            for (int i = tid; i < PR_M * PR_NK; i += PR_T) {
                const int m = i / PR_NK, k = i % PR_NK;
                qb[i] = (T)q[m * PR_D + c * PR_NK + k];
            }
            for (int i = tid; i < PR_BK * PR_NK; i += PR_T) {
                const int t = i / PR_NK, k = i % PR_NK;
                kb[i] = (T)kc[t * PR_D + c * PR_NK + k];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            auto sQ = tQ.slice(0, 0);
            auto sK = tK.slice(0, 0);
            mm.run(sQ, sK, cS);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        /* Destination store: extents (N, M) with strides {1, N} land the
         * tile as plain row-major [m=head][n=token] at m*PR_BK + n. */
        auto tS = tensor(sb, dextents<int32_t, 2>(PR_BK, PR_M),
                         array<int, 2>({1, PR_BK}));
        cS.store(tS);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        /* Plain row-major [h][t] read-through. */
        for (int i = tid; i < PR_M * PR_BK; i += PR_T) so[i] = sb[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    /* ---- PV: O(M,D) = P(M,BK) @ V(BK,D)^T, in two N=128 halves so the
     * fp32 V tile stays at 16 KB ---- */
    for (int nhalf = 0; nhalf < ((int)dbg[1] & 2 ? 0 : 2); ++nhalf) {
        auto tP = tensor(pb, dextents<int32_t, 2>(PR_NK, PR_M));
        auto tV = tensor(vb, dextents<int32_t, 2>(PR_NK, PR_NH));
        matmul2d<
            matmul2d_descriptor(
                PR_M, PR_NH, PR_NK, false, true, false,
                matmul2d_descriptor::mode::multiply_accumulate),
            execution_simdgroups<4>> mm;
        auto cO =
            mm.get_destination_cooperative_tensor<decltype(tV),
                                                  decltype(tP), float>();
        #pragma unroll
        for (uint16_t i = 0; i < cO.get_capacity(); ++i) {
            if (cO.is_valid_element(i)) cO[i] = 0.0f;
        }
        for (int c = 0; c < PR_BK / PR_NK; ++c) {
            for (int i = tid; i < PR_M * PR_NK; i += PR_T) {
                const int m = i / PR_NK, k = i % PR_NK;
                pb[i] = (T)p[m * PR_BK + c * PR_NK + k];
            }
            /* V stages transposed: [n = dim][k = token]. */
            for (int i = tid; i < PR_NH * PR_NK; i += PR_T) {
                const int d = i / PR_NK, k = i % PR_NK;
                vb[i] = (T)vc[(c * PR_NK + k) * PR_D + nhalf * PR_NH + d];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            auto sP = tP.slice(0, 0);
            auto sV = tV.slice(0, 0);
            mm.run(sP, sV, cO);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        /* Register epilogue through coordinates: element (h, d).  In
         * marker mode dumps ids[0]*100 + i so the host can decode the
         * coordinate convention of get_multidimensional_index. */
        #pragma unroll
        for (uint16_t i = 0; i < cO.get_capacity(); ++i) {
            if (!cO.is_valid_element(i)) continue;
            const auto ids = cO.get_multidimensional_index(i);
            if (dbg[0] > 0.5f) {
                oo[nhalf * PR_M * PR_NH + (int)i] =
                    (float)(100 * (int)ids[0] + (int)ids[1]);
            } else {
                const int h = (int)ids[1], d = (int)ids[0];
                oo[h * PR_D + nhalf * PR_NH + d] = cO[i];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tpg.x == 0 && tpg.y == 0 && tid == 0)
        dbg[0] = 1.0f;
}

typedef decltype(t2d_qkpv<float>) t2d_qkpv_ff_t;
typedef decltype(t2d_qkpv<half>) t2d_qkpv_hh_t;

template [[host_name("t2d_qkpv_ff")]] kernel t2d_qkpv_ff_t t2d_qkpv<float>;
template [[host_name("t2d_qkpv_hh")]] kernel t2d_qkpv_hh_t t2d_qkpv<half>;
