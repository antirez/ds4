#ifndef DS4_DEEPSEEK41_GPU_H
#define DS4_DEEPSEEK41_GPU_H

#include <stdbool.h>
#include <stdint.h>

#ifndef DS4_GPU_TENSOR_DEFINED
#define DS4_GPU_TENSOR_DEFINED
typedef struct ds4_gpu_tensor ds4_gpu_tensor;
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* V4.1 activation/cache formats. Buffers are float-addressable but the
 * rounded values follow the released BF16/FP8/FP4 inference graph. */
typedef enum {
    DS4_V41_BF16 = 0,
    DS4_V41_FP8_E8M0 = 1,
    DS4_V41_FP4_E8M0 = 2,
    DS4_V41_FP4_E4M3 = 3,
} ds4_v41_activation_format;
int ds4_gpu_dsv41_quantize(ds4_gpu_tensor *x, uint32_t width, uint32_t rows,
                          ds4_v41_activation_format format);
#if !defined(__APPLE__) && !defined(DS4_ROCM_BUILD) && !defined(DS4_NO_GPU)
/* CUDA scalar Q8 shared expert. 1: queued; 0: unsupported, no work queued;
 * -1: failure. After 1, keep the input/output tensors alive and unchanged
 * until join, which orders the result before subsequent main-stream work.
 * Other work may run between start and join using separate tensors. */
int ds4_gpu_dsv41_shared_start(
        ds4_gpu_tensor *out, ds4_gpu_tensor *gate, ds4_gpu_tensor *up,
        ds4_gpu_tensor *mid, const ds4_gpu_tensor *x,
        const void *model_map, uint64_t model_size,
        uint64_t gate_offset, uint64_t up_offset, uint64_t down_offset,
        uint32_t width, uint32_t hidden, float clamp);
int ds4_gpu_dsv41_shared_join(void);
#endif
/* Full-head prefill, with BF16 rounding between the two Q8 projections. */
int ds4_gpu_dsv41_attention_output_batch(
        ds4_gpu_tensor *out, ds4_gpu_tensor *low,
        const void *model_map, uint64_t model_size,
        uint64_t out_a_offset, uint64_t out_b_offset,
        const ds4_gpu_tensor *heads, uint32_t n_tokens);
/* Packed local 32-head input and BF16 low projection; output is an unrounded
 * rank partial. The graph sums ranks before rounding the attention block. */
int ds4_gpu_dsv41_attention_output_tp_batch(
        ds4_gpu_tensor *out, ds4_gpu_tensor *low,
        const void *model_map, uint64_t model_size,
        uint64_t out_a_offset, uint64_t out_b_offset,
        const ds4_gpu_tensor *heads, uint32_t n_tokens, uint32_t tp_rank);
/* V4.1 output projections, independently Q8_0 or Q4_K. Offsets address the
 * complete matrices; heads/low contain only this rank's contiguous groups.
 * Rounds low to BF16, then writes F32 output (a rank partial for world=2).
 * The caller owns the final TP sum and output BF16 boundary.
 * CUDA Q4 batches require a stream outside graph capture. */
int ds4_gpu_dsv41_attention_output_typed_batch(
        ds4_gpu_tensor *out, ds4_gpu_tensor *low,
        const void *model_map, uint64_t model_size,
        uint64_t out_a_offset, uint64_t out_b_offset,
        uint32_t out_a_type, uint32_t out_b_type,
        const ds4_gpu_tensor *heads, uint32_t n_tokens,
        uint32_t tp_world, uint32_t tp_rank);
/* HC expansion at width 5120 with four streams. Round block[+add] to BF16
 * before the ordered sums, then round each output. Output must be disjoint
 * from all used input ranges; buffers require 16-byte alignment. CUDA requires
 * an initialized single-device backend with quality mode disabled. */
int ds4_gpu_dsv41_hc_expand_bf16(ds4_gpu_tensor *out,
        const ds4_gpu_tensor *block, const ds4_gpu_tensor *add,
        const ds4_gpu_tensor *residual, const ds4_gpu_tensor *split,
        uint32_t rows);
/* Adjacent-pair, unit-magnitude RoPE with the released V4.1 frequencies. */
int ds4_gpu_dsv41_rope(ds4_gpu_tensor *x, uint32_t width, uint32_t heads,
                      uint32_t rows, uint32_t start, bool compressed, bool inverse);
/* Compressed pairs advance two absolute token positions per stored row. */
int ds4_gpu_dsv41_rope_stride(ds4_gpu_tensor *x, uint32_t width, uint32_t heads,
                             uint32_t rows, uint32_t start, uint32_t stride,
                             bool compressed, bool inverse);
int ds4_gpu_dsv41_engram_add(ds4_gpu_tensor *residual,
                           const ds4_gpu_tensor *kv,
                           const ds4_gpu_tensor *q_weight,
                           const ds4_gpu_tensor *k_weight,
                           const ds4_gpu_tensor *mask,
                           uint32_t width, uint32_t rows, float eps);
/* Pool complete pairs and retain the last unpaired projection in previous_*.
 * start is the absolute token position, including earlier chunks. */
int ds4_gpu_dsv41_pool2(ds4_gpu_tensor *out,
                      const ds4_gpu_tensor *kv, const ds4_gpu_tensor *scores,
                      ds4_gpu_tensor *previous_kv, ds4_gpu_tensor *previous_scores,
                       uint32_t width, uint32_t rows, uint32_t start);
/* Candidate blocks contain eight compressed positions. Produce causal block
 * maxima, pinning the newest block; filter consumes a 0/-inf block mask. */
int ds4_gpu_dsv41_candidate_blocks(ds4_gpu_tensor *blocks,
                                  const ds4_gpu_tensor *scores,
                                  uint32_t width, uint32_t rows,
                                  uint32_t start, uint32_t ratio);
int ds4_gpu_dsv41_candidate_filter(ds4_gpu_tensor *scores,
                                  const ds4_gpu_tensor *block_mask,
                                  uint32_t width, uint32_t rows,
                                  uint32_t start, uint32_t ratio);
/* Causal index scores over ratio-1/2 compressed keys, without an extra cast
 * of the already quantized FP4 queries/keys. Scores have source_rows stride. */
int ds4_gpu_dsv41_indexer_scores_batch(ds4_gpu_tensor *scores,
                                     const ds4_gpu_tensor *q,
                                     const ds4_gpu_tensor *weights,
                                     const ds4_gpu_tensor *keys,
                                     uint32_t source_rows, uint32_t rows,
                                     uint32_t start, uint32_t ratio);
int ds4_gpu_dsv41_tensor_ops_available(void);
/* Reuse exact BF16 views of FP4 queries/keys across score tiles. Invalid
 * casts retain the F32 arithmetic for the affected tile. */
uint64_t ds4_gpu_dsv41_indexer_packed_bytes(uint32_t source_rows, uint32_t rows);
int ds4_gpu_dsv41_indexer_pack(ds4_gpu_tensor *packed,
                              const ds4_gpu_tensor *q, const ds4_gpu_tensor *keys,
                              uint32_t source_rows, uint32_t rows);
int ds4_gpu_dsv41_indexer_scores_packed(ds4_gpu_tensor *scores,
                                      const ds4_gpu_tensor *q,
                                      const ds4_gpu_tensor *weights,
                                      const ds4_gpu_tensor *keys,
                                      const ds4_gpu_tensor *packed,
                                      uint32_t source_rows, uint32_t rows,
                                      uint32_t start, uint32_t ratio,
                                      uint32_t packed_rows, uint32_t offset);
/* Exact row-sort ordering with independent causal widths; at least 1024
 * visible keys per row. Output stride is 512 indices. */
int ds4_gpu_dsv41_indexer_topk_batch(ds4_gpu_tensor *selected,
                                    const ds4_gpu_tensor *scores,
                                    uint32_t width, uint32_t rows,
                                    uint32_t start, uint32_t ratio);
enum { DS4_V41_CARRY_BF16, DS4_V41_CARRY_MASK, DS4_V41_CARRY_F32 };
/* Lossless storage for already-BF16 activations or 0/-inf candidate masks.
 * Packed rows are padded to whole uint32_t words. Plain rows remain F32. */
int ds4_gpu_dsv41_carry_copy(ds4_gpu_tensor *packed, uint32_t row_offset,
                            ds4_gpu_tensor *plain, uint32_t width, uint32_t rows,
                            uint32_t format, bool pack);
/* Batched F16 projections with the same arithmetic as individual matvecs. */
int ds4_gpu_dsv41_projection_rows(ds4_gpu_tensor *out,
                                 const void *model_map, uint64_t model_size,
                                 uint64_t weight_offset, uint32_t width,
                                 uint32_t outputs, uint32_t rows,
                                 const ds4_gpu_tensor *in);
/* Gather 512-wide F32 KV rows; IDs must come from top-k over source_rows. */
int ds4_gpu_dsv41_gather_kv(ds4_gpu_tensor *out, const ds4_gpu_tensor *source,
                           const ds4_gpu_tensor *ids, uint32_t source_rows,
                           uint32_t selected_rows);

#ifdef __APPLE__
/* Metal V4.1 specializations; other backends retain the shared graph APIs. */
/* Round low in place to BF16 (F32 storage), then write its F16 RHS in one
 * pass. Used ranges must be disjoint and 16-byte aligned; count <= UINT32_MAX.
 * The two conversions retain the F32 -> BF16 -> F16 rounding order. */
int ds4_gpu_dsv41_bf16_f16_rhs(ds4_gpu_tensor *low, ds4_gpu_tensor *rhs,
                              uint32_t width, uint32_t rows);
/* F32 weighted RMSNorm with the V4.1 BF16 output boundary. Activation views
 * and weight offsets must be 16-byte aligned; in-place normalization is safe. */
int ds4_gpu_dsv41_norm_rows(ds4_gpu_tensor *out, const ds4_gpu_tensor *x,
                           const void *model_map, uint64_t model_size,
                           uint64_t weight_offset, uint32_t width, uint32_t rows,
                           float eps);
/* Decode/small-row Q8 projections with the scalar reduction and BF16 output.
 * The output width must be even. Large prefill batches should retain GEMM. */
int ds4_gpu_dsv41_q8_bf16_rows(
        ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
        uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim,
        const ds4_gpu_tensor *x, uint32_t n_rows);
/* Single-row Q8 shared gate/up with BF16 boundaries before SwiGLU and at
 * its output. Admits K5120/M2304, NSG4, non-quality, single-device execution.
 * Returns 1 on success, 0 before encoding when unsupported, -1 on GPU error. */
int ds4_gpu_dsv41_shared_swiglu(
        ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
        uint64_t gate_offset, uint64_t up_offset,
        uint64_t in_dim, uint64_t out_dim,
        const ds4_gpu_tensor *x, float clamp);
/* V4.1 elementwise epilogues, single-device non-quality execution. Inputs
 * and output must have disjoint used ranges and 16-byte aligned offsets.
 * SwiGLU has width 2304 and rounds gate/up before clamping, then its output.
 * HC has width 5120 and four streams; expand rounds block[+add] before the
 * existing ordered sums, then each output. Sum rounds the four-stream sum;
 * split selects a 24-float weight row instead of a packed four-float row. */
int ds4_gpu_dsv41_swiglu_bf16(ds4_gpu_tensor *out,
        const ds4_gpu_tensor *gate, const ds4_gpu_tensor *up,
        uint32_t rows, float clamp);
int ds4_gpu_dsv41_hc_sum_bf16(ds4_gpu_tensor *out,
        const ds4_gpu_tensor *residual, const ds4_gpu_tensor *weights,
        uint32_t rows, bool split);
/* Q4_K Q-B (K1280, M32768), with raw F32 output. Dead workspace must hold
 * rows*1280*2 bytes; an additional 80 MiB permits transient FP16 weights for
 * larger prefill batches. The caller owns the following BF16/RoPE boundary.
 * Used ranges must be disjoint with 16-byte aligned offsets. Smaller views
 * retain RHS-only reuse or native Q4; no persistent weight cache is allocated. */
int ds4_gpu_dsv41_q4_qb_rows(
        ds4_gpu_tensor *out, ds4_gpu_tensor *rhs_scratch,
        const void *model_map, uint64_t model_size, uint64_t weight_offset,
        const ds4_gpu_tensor *x, uint32_t n_rows);
/* Optional dead graph storage for transient output-B Q4->F16 expansion.
 * Needs 80 MiB + n_tokens*8192*2 bytes; smaller views use the native path.
 * Admitted storage must not overlap heads, low, output, or either matrix. */
int ds4_gpu_dsv41_attention_output_typed_workspace_batch(
        ds4_gpu_tensor *out, ds4_gpu_tensor *low, ds4_gpu_tensor *workspace,
        const void *model_map, uint64_t model_size,
        uint64_t out_a_offset, uint64_t out_b_offset,
        uint32_t out_a_type, uint32_t out_b_type,
        const ds4_gpu_tensor *heads, uint32_t n_tokens,
        uint32_t tp_world, uint32_t tp_rank);
/* Round the complete head to BF16 before RoPE, retaining the tail's second
 * BF16 boundary. Equivalent to quantize(BF16) followed by dsv41_rope. */
int ds4_gpu_dsv41_bf16_rope(ds4_gpu_tensor *x, uint32_t width, uint32_t heads,
                           uint32_t rows, uint32_t start, bool compressed, bool inverse);
/* Ordered GPU publication of the all-visible candidate mask prefix. */
int ds4_gpu_dsv41_candidate_mask_all(ds4_gpu_tensor *mask, uint32_t n_comp);
#endif

#ifdef __cplusplus
}
#endif
#endif
