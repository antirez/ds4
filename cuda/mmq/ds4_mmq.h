// SPDX-License-Identifier: MIT
// ds4_mmq.h - public C ABI for ds4's quantized matmul kernels.
//
// All functions are extern "C" so ds4.c / ds4_cuda.cu can call them
// without C++ compilation. Functions return 0 on success and non-zero on
// failure (with stderr error message). Device pointers are caller-owned.
//
// Phase 0: skeleton only. Q8_0 dense entry compiles and instantiates
// mul_mat_q_case<Q8_0> but is not yet wired into ds4_cuda.cu.
// Phase 1: Q8_1 activation quantizer wrapper added.
// Phase 2: Q8_0 dense entry verified against cublas+dequant baseline.
// Phase 3: Q2_K + IQ2_XXS dense entries.
// Phase 4: MoE _id variants of all three.

#pragma once

#include <cuda_runtime.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// One-time init. Sets the current CUDA device and triggers lazy population
// of the device-info singleton. Safe to call repeatedly.
//
//   device: CUDA device ordinal (0 for the primary GPU).
// Returns 0 on success.
int ds4_mmq_init(int device);
void ds4_mmq_set_aligned_q81_scratch(void *ptr, size_t bytes);

// Query whether ds4_mmq is willing to handle a given matmul. Returns
//   1 if mmq is faster than dequant+cublas for this shape on this device,
//   0 otherwise (caller should fall back to its existing dequant+cublas path).
//
// Wraps ggml_cuda_should_use_mmq. type_x uses ds4 quant codes which match
// ggml's enum:
//   8  = Q8_0
//   10 = Q2_K
//   16 = IQ2_XXS
//
//   ne11:      batch dimension (number of activation columns / tokens).
//   n_experts: 0 for dense matmul, >0 for MoE (e.g. 256 for V4 Flash).
int ds4_mmq_should_use(int type_x, int64_t ne11, int64_t n_experts);

// Dense matmul entry points. Per-type wrappers that all share the same
// underlying mul_mat_q template, parameterised by the weight quant type.
//
// All three variants compute:
//
//   out[col, row] = sum_k W[row, k] * X[k, col]      0 <= row < M, 0 <= col < N
//
// Layouts (matching ggml + llama.cpp mmq conventions, all on device):
//   W:       [M rows, K cols], row-major, packed in the type-specific block
//            format. K must be a multiple of 256.
//   X_f32:   [N rows, K cols] F32 row-major (logical [K, N] with K
//            innermost - i.e. for each "column" col of the logical [K, N]
//            matrix, K contiguous floats live at X[col*K .. col*K + K]).
//   out_f32: caller-allocated, M*N floats. mmq writes in column-major:
//            out[col*M + row]. Callers expecting row-major must transpose.
//
// Returns 0 on success, non-zero on validation or launch failure.

int ds4_mmq_q8_0_dense(
    const void  * W_q8_0,
    const float * X_f32,
    float       * out_f32,
    int           M,
    int           N,
    int           K,
    cudaStream_t  stream);

// v0.5 flat-pool p5a: same contract as ds4_mmq_q8_0_dense but the
// activation arrives PRE-QUANTIZED in the block_q8_1_mmq D4 layout
// (producer-emitted, op-for-op equal to quantize_mmq_q8_1<D4>; the fused
// own out_a kernel is the producer).  Y_q8_mmq must hold N*(K/128)
// 144-byte blocks (ib = kseg*N + row) plus mmq_x_max tail slack, which
// this entry zeroes.  Requires K such that GGML_PAD(K, MATRIX_ROW_PADDING)
// == K.  Returns 0 on success; callers fall back to the quantizing entry
// on any non-zero.
int ds4_mmq_q8_0_dense_preq(
    const void  * W_q8_0,
    const void  * Y_q8_mmq,
    size_t        y_bytes,
    float       * out_f32,
    int           M,
    int           N,
    int           K,
    cudaStream_t  stream);

// p5a verify instrument: reference quantize (dense_impl parameters) into a
// caller buffer, for byte-diffing producer emits.
int ds4_mmq_q8_0_quantize_ref(
    const float * X,
    void        * y,
    size_t        y_bytes,
    int           N,
    int           K,
    cudaStream_t  stream);

// Dense Q8_0 D2R on the kind-5 aligned artifact (weight server
// --repack-q8-aligned).  Same in/out contract as ds4_mmq_q8_0_dense but W is
// the ALIGNED artifact base ([half dq[nblk]][pad64][int8 qs]), and the shape
// must satisfy M % 128 == 0 && K % 1024 == 0.  Callers gate on n_tok scale
// and K <= 4096 (o_proj's K=8192 measured faster on mmq).
int ds4_mmq_q8_0_dense_d2r(
    const void  * W_aligned,
    const float * X_f32,
    float       * out_f32,
    int           M,
    int           N,
    int           K,
    cudaStream_t  stream);

// flat-pool p5c: D2R dense over a producer-quantized token-major Y
// (block_q8_1_mmq D4, ib = kseg*N + row, requires GGML_PAD(K) == K).
// y_bytes must cover the payload plus max(mmq_x_max, 128) slack blocks;
// the slack is zeroed here each call (S1.1a).  Returns 0 on success;
// callers fall back to the quantizing entry on any non-zero.
int ds4_mmq_q8_0_dense_d2r_preq(
    const void  * W_aligned,
    const void  * Y_q8_mmq,
    size_t        y_bytes,
    float       * out_f32,
    int           M,
    int           N,
    int           K,
    cudaStream_t  stream);

int ds4_mmq_q2_K_dense(
    const void  * W_q2_K,
    const float * X_f32,
    float       * out_f32,
    int           M,
    int           N,
    int           K,
    cudaStream_t  stream);

int ds4_mmq_iq2_xxs_dense(
    const void  * W_iq2_xxs,
    const float * X_f32,
    float       * out_f32,
    int           M,
    int           N,
    int           K,
    cudaStream_t  stream);

int ds4_mmq_q4_K_dense(
    const void  * W_q4_K,
    const float * X_f32,
    float       * out_f32,
    int           M,
    int           N,
    int           K,
    cudaStream_t  stream);

int ds4_mmq_mxfp4_dense(
    const void  * W_mxfp4,
    const float * X_f32,
    float       * out_f32,
    int           M,
    int           N,
    int           K,
    cudaStream_t  stream);

// MoE matmul entry points. For each (token, slot-within-token's-top-k) pair
// the kernel computes:
//
//   out[col, row] = sum_k W[ids[token, slot], row, k] * X[token, k]
//
// where col = token * n_expert_used + slot, row in [0, M).  The caller is
// responsible for any downstream sum-weighted-by-router-weights reduction
// across the n_expert_used dimension (Phase 5 wires this into ds4's
// existing moe_sum_kernel).
//
// Layouts:
//   W:       device pointer, [n_experts, M rows, K cols] in the
//            type-specific block format.  Per-expert slab is M*K/blck
//            blocks stored contiguously; experts are stacked.
//   X_f32:   device pointer, [n_tokens, K] F32 row-major (K innermost).
//   ids:     device pointer, [n_tokens, n_expert_used] int32_t row-major.
//            ids[t*n_expert_used + s] is the expert id for token t's
//            s-th routing slot.  Values must be in [0, n_experts).
//   out_f32: caller-allocated, M * n_tokens * n_expert_used floats.
//            Column-major: out[col*M + row].
//
// K must be a multiple of 256.  n_expert_used must be one of the values
// the vendored mm_ids_helper template specialises on: 2, 4, 6, 8, 16, 32
// (or any other value, which falls back to the generic path).  For V4
// Flash, n_expert_used = 6.
//
// Returns 0 on success, non-zero on validation or launch failure.

int ds4_mmq_q8_0_moe(
    const void    * W,
    const float   * X_f32,
    const int32_t * ids,
    float         * out_f32,
    int             M,
    int             K,
    int             n_tokens,
    int             n_experts,
    int             n_expert_used,
    cudaStream_t    stream);

int ds4_mmq_q2_K_moe(
    const void    * W,
    const float   * X_f32,
    const int32_t * ids,
    float         * out_f32,
    int             M,
    int             K,
    int             n_tokens,
    int             n_experts,
    int             n_expert_used,
    cudaStream_t    stream);

int ds4_mmq_iq2_xxs_moe(
    const void    * W,
    const float   * X_f32,
    const int32_t * ids,
    float         * out_f32,
    int             M,
    int             K,
    int             n_tokens,
    int             n_experts,
    int             n_expert_used,
    cudaStream_t    stream);

// ds4 (P4 Inc3): same contract as ds4_mmq_q2_K_moe but W_soa is the aligned
// row-pair-SoA artifact (weight server --repack-q2k-aligned, layout in
// ds4_mmq_q2_k_aligned_bytes' comment) instead of the raw block stream.  The
// tile loader reads the SoA sections directly -- bit-identical output to the
// raw path, no derepack scratch.
// CONTRACT DIFFERENCE (P3): unlike the raw entries, the output is NOT
// nonfinite-sanitized; the routed-MoE consumers (moe_mmq_swiglu / moe_sum
// with guard_nonfinite=1) sanitize at read, so the standalone whole-buffer
// pass is skipped.  New callers must sanitize at consumption.
int ds4_mmq_q2_K_moe_soa(
    const void    * W_soa,
    const float   * X_f32,
    const int32_t * ids,
    float         * out_f32,
    int             M,
    int             K,
    int             n_tokens,
    int             n_experts,
    int             n_expert_used,
    cudaStream_t    stream);

int ds4_mmq_q4_K_moe(
    const void    * W,
    const float   * X_f32,
    const int32_t * ids,
    float         * out_f32,
    int             M,
    int             K,
    int             n_tokens,
    int             n_experts,
    int             n_expert_used,
    cudaStream_t    stream);

int ds4_mmq_mxfp4_moe(
    const void    * W,
    const float   * X_f32,
    const int32_t * ids,
    float         * out_f32,
    int             M,
    int             K,
    int             n_tokens,
    int             n_experts,
    int             n_expert_used,
    cudaStream_t    stream);

// Paired MoE entries. Compute gate AND up over the same activation in a
// single call so the Q8_1 quantize of X (and the mm_ids_helper bookkeeping)
// happens once instead of twice. Both weights must be the same quant type
// and the same shape (M, K, n_experts); out_a / out_b have the same layout
// as a single ds4_mmq_<type>_moe call. Saves one launch of
// quantize_mmq_q8_1_cuda and one ggml_cuda_launch_mm_ids_helper per MoE
// block. See ds4_mmq.cu / routed_moe_launch for the wiring.
//
// Returns 0 on success; on error neither output is guaranteed valid.

int ds4_mmq_iq2_xxs_moe_pair(
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
    cudaStream_t    stream);

// ds4 (P4 Inc3): same contract as ds4_mmq_iq2_xxs_moe_pair but over the
// aligned-SoA artifacts (weight server --repack-iq2-aligned); see
// ds4_mmq_q2_K_moe_soa.
int ds4_mmq_iq2_xxs_moe_pair_soa(
    const void    * Wa_soa,
    const void    * Wb_soa,
    const float   * X_f32,
    const int32_t * ids,
    float         * out_a,
    float         * out_b,
    int             M,
    int             K,
    int             n_tokens,
    int             n_experts,
    int             n_expert_used,
    cudaStream_t    stream);

/* v0.5 inc-9 (F7): fused target-prefill pipeline over the aligned-SoA
 * IQ2_XXS gate/up and Q2_K down artifacts.  Builds the expert-major
 * assignment map once, runs the paired gate/up MMQs, computes clamp +
 * SwiGLU + router weighting in mid_f32, then gathers and quantizes those
 * rows for the Q2_K down MMQ through the same ids_dst/expert_bounds.  No
 * second mm_ids_helper; gate/up/mid/down keep the pair-major layout. */
int ds4_mmq_iq2_xxs_q2_K_moe_fused_soa(
    const void    * W_gate_soa,
    const void    * W_up_soa,
    const void    * W_down_soa,
    const float   * X_f32,
    const int32_t * ids,
    const float   * router_weights,
    float         * gate_f32,
    float         * up_f32,
    float         * mid_f32,
    float         * down_f32,
    int             expert_mid_dim,
    int             expert_in_dim,
    int             out_dim,
    int             n_tokens,
    int             n_experts,
    int             n_expert_used,
    float           clamp,
    cudaStream_t    stream);

/* Aligned-artifact production fast path: gate/up stay in registers, weighted
 * SwiGLU is quantized directly into down_q8_scratch, and only the pair-major
 * down output is materialized.  Caller-owned scratch keeps this hot path free
 * of stream-ordered allocations; all three ranges must be distinct, remain
 * live until return, and be sized to their LOGICAL segments (never an owning
 * arena's capacity). */
/* flat-pool p5c: input_q8_ext, when non-NULL, is a producer-emitted
 * TOKEN-COMPACT block_q8_1_mmq buffer of X (ib = kseg*n_tokens + row,
 * n_tokens rows) and the internal activation quantize is skipped; the
 * gate/up D2R kernel reads it through the p5b ids_src indirection.
 * Ignored (classic gathered quantize) when DS4_MMQ_NO_YIND is set. */
int ds4_mmq_iq2_xxs_q2_K_moe_fused_direct_soa(
    const void    * W_gate_soa,
    const void    * W_up_soa,
    const void    * W_down_soa,
    const float   * X_f32,
    const int32_t * ids,
    const float   * router_weights,
    void          * input_q8_scratch,
    size_t          input_q8_scratch_bytes,
    void          * down_q8_scratch,
    size_t          down_q8_scratch_bytes,
    void          * work_scratch,
    size_t          work_scratch_bytes,
    const void    * input_q8_ext,
    size_t          input_q8_ext_bytes,
    float         * down_f32,
    int             expert_mid_dim,
    int             expert_in_dim,
    int             out_dim,
    int             n_tokens,
    int             n_experts,
    int             n_expert_used,
    float           clamp,
    cudaStream_t    stream);

int ds4_mmq_q4_K_moe_pair(
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
    cudaStream_t    stream);

int ds4_mmq_mxfp4_moe_pair(
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
    cudaStream_t    stream);

// MoE vector matmul entries (Step 6). Same signature and semantics as the
// ds4_mmq_<type>_moe entries above, but route through llama.cpp's mmvq
// kernels instead of mmq. mmvq is structurally optimised for small batch
// counts (single-token decode, short prefill), where mmq's tile-based
// approach wastes work on empty columns.
//
// Constraints:
//   - n_tokens * something must fit under mmvq's per-arch batch cap
//     (MMVQ_MAX_BATCH_SIZE = 8 on Blackwell). Specifically, ncols_dst as
//     computed by the wrapper must be <= 8. The wrapper rejects with -1
//     if the request is too large.
//   - K must be a multiple of 256 (same as the mmq path).
//
// Unlike the mmq path, mmvq consumes a CANONICAL block_q8_1 buffer (not
// the interleaved block_q8_1_mmq the mmq path uses). The wrapper builds
// the canonical buffer internally; callers cannot reuse a Q8_1 buffer
// previously built for the mmq path.
//
// Returns 0 on success, non-zero on validation or launch failure.

int ds4_mmq_q8_0_moe_vec(
    const void    * W,
    const float   * X_f32,
    const int32_t * ids,
    float         * out_f32,
    int             M,
    int             K,
    int             n_tokens,
    int             n_experts,
    int             n_expert_used,
    cudaStream_t    stream);

int ds4_mmq_q2_K_moe_vec(
    const void    * W,
    const float   * X_f32,
    const int32_t * ids,
    float         * out_f32,
    int             M,
    int             K,
    int             n_tokens,
    int             n_experts,
    int             n_expert_used,
    cudaStream_t    stream);

int ds4_mmq_iq2_xxs_moe_vec(
    const void    * W,
    const float   * X_f32,
    const int32_t * ids,
    float         * out_f32,
    int             M,
    int             K,
    int             n_tokens,
    int             n_experts,
    int             n_expert_used,
    cudaStream_t    stream);

int ds4_mmq_q4_K_moe_vec(
    const void    * W,
    const float   * X_f32,
    const int32_t * ids,
    float         * out_f32,
    int             M,
    int             K,
    int             n_tokens,
    int             n_experts,
    int             n_expert_used,
    cudaStream_t    stream);

int ds4_mmq_mxfp4_moe_vec(
    const void    * W,
    const float   * X_f32,
    const int32_t * ids,
    float         * out_f32,
    int             M,
    int             K,
    int             n_tokens,
    int             n_experts,
    int             n_expert_used,
    cudaStream_t    stream);

// Aligned-SoA IQ2_XXS decode matvec (megakernel program M1-Inc1).
//
// block_iq2_xxs is 66 bytes, so the raw expert stream is only 2-byte aligned
// and every 32-bit code word costs two 16-bit loads.  W_aligned is a repacked
// copy of the SAME bytes with the block scales split out and the code stream
// 64B-aligned:
//
//   [ __half dq[nblk] ][ pad to 64B ][ uint2 qs[nblk * 8] ]
//
// where nblk = n_experts * M * (K / 256) and block linear order matches the
// raw tensor byte order (expert-major, then row, then block).  The weight
// server builds this layout (--repack-iq2-aligned); use
// ds4_mmq_iq2_xxs_aligned_bytes to size or validate an artifact.
//
// Semantics and output layout are identical to ds4_mmq_iq2_xxs_moe_vec at
// n_tokens == 1 (the only supported width; other widths return non-zero so
// the caller can fall back).
uint64_t ds4_mmq_iq2_xxs_aligned_bytes(int M, int K, int n_experts);

// M1-Inc2b: exact inverse of the weight-server repack.  Fills raw_out
// (nblk * 66 bytes, raw block_iq2_xxs byte stream) from an aligned artifact,
// device->device on `stream`.  Lets the batched/mmq raw-layout consumers run
// from a device scratch instead of the client mmap when the raw spans were
// excluded from the upload.
int ds4_mmq_iq2_xxs_aligned_derepack(
    const void    * W_aligned,
    void          * raw_out,
    int             M,
    int             K,
    int             n_experts,
    cudaStream_t    stream);

// M2 moe-down: aligned row-pair-SoA Q2_K routed-expert decode matvec.  Twin
// of mul_mat_vec_q_moe<GGML_TYPE_Q2_K, 2> at the down-leg call shape
// (n_expert_used == 1: each (token, slot) assignment is its own "token") with
// bit-identical outputs; only the weight layout changes.  W_aligned is the
// weight-server --repack-q2k-aligned artifact (DERIVED_Q2_K_ALIGNED_MOE,
// REPLACES the raw range; byte-neutral):
//
//   npair = n_experts * (M/2) * (K/256)
//   [ uint2 dm2[npair] ][ pad 64B ][ int4 sc4[npair*2] ][ pad 64B ]
//   [ uint2 qs2[npair*16] ]
//
// keyed to the kernel's rows_per_block == 2 (see ds4_mmq.cu for the exact
// field packing).  Use ds4_mmq_q2_k_aligned_bytes to size or validate an
// artifact.  Unsupported shapes return non-zero so the caller can fall back.
uint64_t ds4_mmq_q2_k_aligned_bytes(int M, int K, int n_experts);

int ds4_mmq_q2_K_aligned_moe_vec(
    const void    * W_aligned,
    const float   * X_f32,
    const int32_t * ids,
    float         * out_f32,
    int             M,
    int             K,
    int             n_tokens,
    int             n_experts,
    int             n_expert_used,
    cudaStream_t    stream);

// Exact inverse of the weight-server repack: fills raw_out (nblk * 84 bytes,
// raw block_q2_K stream) from an aligned artifact, device->device on
// `stream`, for the batched/mmq raw-layout consumers (same role as
// ds4_mmq_iq2_xxs_aligned_derepack).
int ds4_mmq_q2_K_aligned_derepack(
    const void    * W_aligned,
    void          * raw_out,
    int             M,
    int             K,
    int             n_experts,
    cudaStream_t    stream);

// M1-Inc3: aligned-SoA Q8_0 dense decode matvec.  Artifact layout
// [__half dq[nblk]][pad to 64B][int8 qs[nblk*32]], nblk = M * (K/32), block
// order equal to the raw tensor byte order (weight server
// --repack-q8-aligned; raw spans stay served — dense artifacts duplicate,
// they do not replace).  n_tokens == 1 and K % 1024 == 0 only; other shapes
// return non-zero so the caller can fall back to ds4_mmq_q8_0_dense_vec.
uint64_t ds4_mmq_q8_0_aligned_bytes(int M, int K);

int ds4_mmq_q8_0_aligned_dense_vec(
    const void  * W_aligned,
    const float * X_f32,
    float       * out_f32,
    int           M,
    int           N,
    int           K,
    cudaStream_t  stream);

int ds4_mmq_iq2_xxs_aligned_moe_vec(
    const void    * W_aligned,
    const float   * X_f32,
    const int32_t * ids,
    float         * out_f32,
    int             M,
    int             K,
    int             n_tokens,
    int             n_experts,
    int             n_expert_used,
    cudaStream_t    stream);

// M1-Inc2 variants over the same aligned artifacts (n_tokens == 1 only).
//
// _pair_vec: one activation quantize + one launch computes both raw gate and
// up outputs (nonfinite zeroed in-kernel; no sanitize pass needed).  The
// caller still runs its clamp-aware SwiGLU.
//
// _gate_up_mid_vec: additionally folds the clamp/SwiGLU/router-weight
// epilogue (identical semantics to ds4_mmq_moe_gate_up_mid_q8_1_qwarp32) and
// writes mid[slot * M + row] directly.
int ds4_mmq_iq2_xxs_aligned_moe_pair_vec(
    const void    * W_gate_aligned,
    const void    * W_up_aligned,
    const float   * X_f32,
    const int32_t * ids,
    float         * gate_out,
    float         * up_out,
    int             M,
    int             K,
    int             n_tokens,
    int             n_experts,
    int             n_expert_used,
    cudaStream_t    stream);

int ds4_mmq_iq2_xxs_aligned_moe_gate_up_mid_vec(
    const void    * W_gate_aligned,
    const void    * W_up_aligned,
    const float   * X_f32,
    const int32_t * ids,
    const float   * weights,
    float         * mid_f32,
    int             M,
    int             K,
    int             n_tokens,
    int             n_experts,
    int             n_expert_used,
    float           clamp,
    cudaStream_t    stream);

// Fused down+sum vector entries for routed MoE with top_k=6. These preserve
// the canonical Q8_1 activation quantization used by the regular _moe_vec
// path, but avoid materializing [token, slot, out_dim] down results and avoid
// the separate slot-sum kernel. The caller must already have baked router
// weights into X_f32, so the helper computes:
//
//   out[token, row] = sum_slot W[ids[token, slot], row, :] @ X[token, slot, :]
//
// Layouts:
//   W:       [n_experts, M rows, K cols] in Q2_K or Q4_K blocks
//   X_f32:   [n_tokens * 6, K] F32 row-major
//   ids:     [n_tokens, 6] int32 row-major
//   out_f32: [n_tokens, M] F32 row-major
//
// Constraints:
//   - n_expert_used must be 6
//   - K must be a multiple of 256
//
// Returns 0 on success, non-zero on validation or launch failure.

int ds4_mmq_q2_K_moe_down_sum6_vec(
    const void    * W,
    const float   * X_f32,
    const int32_t * ids,
    float         * out_f32,
    int             M,
    int             K,
    int             n_tokens,
    int             n_experts,
    int             n_expert_used,
    cudaStream_t    stream);

int ds4_mmq_q4_K_moe_down_sum6_vec(
    const void    * W,
    const float   * X_f32,
    const int32_t * ids,
    float         * out_f32,
    int             M,
    int             K,
    int             n_tokens,
    int             n_experts,
    int             n_expert_used,
    cudaStream_t    stream);

int ds4_mmq_mxfp4_moe_down_sum6_vec(
    const void    * W,
    const float   * X_f32,
    const int32_t * ids,
    float         * out_f32,
    int             M,
    int             K,
    int             n_tokens,
    int             n_experts,
    int             n_expert_used,
    cudaStream_t    stream);

// Fused gate+up+SwiGLU vector entries for routed MoE with top_k=6. These use
// canonical Q8_1 activation quantization and compute weighted mid directly:
//
//   mid[token, slot, row] =
//       silu(clamp_gate(W_gate[expert,row] @ X[token]))
//     * clamp_up(W_up[expert,row] @ X[token])
//     * router_weight[token, slot]
//
// Clamp semantics match ds4_cuda.cu: gate is capped above, up is clamped to
// [-clamp, clamp], and no clamp is applied when clamp <= 1e-6.
//
// Layouts:
//   W_gate/W_up: [n_experts, M rows, K cols] in IQ2_XXS or Q4_K blocks
//   X_f32:       [n_tokens, K] F32 row-major
//   ids:         [n_tokens, 6] int32 row-major
//   weights:     [n_tokens, 6] F32 row-major
//   mid_f32:     [n_tokens * 6, M] F32 row-major
//
// Constraints:
//   - n_expert_used must be 6
//   - K must be a multiple of 256
//
// Returns 0 on success, non-zero on validation or launch failure.

int ds4_mmq_iq2_xxs_moe_gate_up_mid_vec(
    const void    * W_gate,
    const void    * W_up,
    const float   * X_f32,
    const int32_t * ids,
    const float   * weights,
    float         * mid_f32,
    int             M,
    int             K,
    int             n_tokens,
    int             n_experts,
    int             n_expert_used,
    float           clamp,
    cudaStream_t    stream);

int ds4_mmq_q4_K_moe_gate_up_mid_vec(
    const void    * W_gate,
    const void    * W_up,
    const float   * X_f32,
    const int32_t * ids,
    const float   * weights,
    float         * mid_f32,
    int             M,
    int             K,
    int             n_tokens,
    int             n_experts,
    int             n_expert_used,
    float           clamp,
    cudaStream_t    stream);

int ds4_mmq_mxfp4_moe_gate_up_mid_vec(
    const void    * W_gate,
    const void    * W_up,
    const float   * X_f32,
    const int32_t * ids,
    const float   * weights,
    float         * mid_f32,
    int             M,
    int             K,
    int             n_tokens,
    int             n_experts,
    int             n_expert_used,
    float           clamp,
    cudaStream_t    stream);

// Pair-fused MoE vector matmul entries (Step 6). Computes
//
//   out[col, row] = (W_a[ids, row, :] @ X[token, :])
//                 * silu(W_b[ids, row, :] @ X[token, :])
//
// in a SINGLE mmvq launch via mmvq's built-in fusion (fusion.gate = W_b,
// fusion.glu_op = GGML_GLU_OP_SWIGLU). The kernel applies silu to the
// fusion.gate matmul and multiplies into the main matmul: pass the
// SwiGLU "up" weights as W_a and the SwiGLU "gate" weights as W_b to
// match ds4's expected silu(gate)*up semantics. The DeepSeek V4 clamp
// and router-weight multiplication are NOT applied by the kernel - the
// caller is expected to apply them as a small post-process (or to skip
// clamp if clamp==0).
//
// Constraints:
//   - n_tokens = 1 ONLY. mmvq supports fusion only at ncols_dst = 1.
//   - K must be a multiple of 256.
//
// Returns 0 on success, non-zero on validation or launch failure.

int ds4_mmq_iq2_xxs_moe_pair_vec(
    const void    * W_a,
    const void    * W_b,
    const float   * X_f32,
    const int32_t * ids,
    float         * out_silu,
    int             M,
    int             K,
    int             n_experts,
    int             n_expert_used,
    cudaStream_t    stream);

int ds4_mmq_q4_K_moe_pair_vec(
    const void    * W_a,
    const void    * W_b,
    const float   * X_f32,
    const int32_t * ids,
    float         * out_silu,
    int             M,
    int             K,
    int             n_experts,
    int             n_expert_used,
    cudaStream_t    stream);

// Raw pair MoE vector entries. They quantize X to canonical Q8_1 once, then
// run the same mmvq matvec kernel twice to produce the unfused gate and up
// outputs. This preserves the caller's clamp-aware SwiGLU epilogue while
// avoiding the duplicate Q8_1 quantize/scratch setup of two separate
// ds4_mmq_*_moe_vec calls.
//
// Layout and constraints match ds4_mmq_*_moe_vec:
//   out_[token * n_expert_used + slot, row]
//   K must be a multiple of 256.

int ds4_mmq_iq2_xxs_moe_pair_raw_vec(
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
    cudaStream_t    stream);

int ds4_mmq_q4_K_moe_pair_raw_vec(
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
    cudaStream_t    stream);

// Dense vector matmul entry (Step 6). Same shape semantics as
// ds4_mmq_q8_0_dense but routed through mmvq for batch counts that
// favour the vec path (n_tokens <= 8 on Blackwell).
//
// Returns 0 on success, non-zero on validation or launch failure.

int ds4_mmq_q8_0_dense_vec(
    const void  * W_q8_0,
    const float * X_f32,
    float       * out_f32,
    int           M,
    int           N,
    int           K,
    cudaStream_t  stream);

// Set the thread-local stream that the internal cuda pool uses for
// cudaMallocAsync / cudaFreeAsync.  Defaults to cudaStreamPerThread.
// Step 8 (CUDA Graphs) calls this with the capture stream so pool
// allocations land on the captured stream and don't invalidate capture.
// Pass NULL to reset to cudaStreamPerThread.
void ds4_pool_set_stream(cudaStream_t stream);

#ifdef __cplusplus
} // extern "C"
#endif
