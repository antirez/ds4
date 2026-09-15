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
#if !defined(__APPLE__) && !defined(DS4_ROCM_BUILD) && !defined(__HIP_PLATFORM_AMD__) && !defined(DS4_NO_GPU)
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
#ifdef DS4_ROCM_BUILD
int ds4_gpu_dsv41_indexer_scores_one(ds4_gpu_tensor *scores,
        const ds4_gpu_tensor *q, const ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *keys, uint32_t source_rows);
#endif
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

#if defined(DS4_ROCM_BUILD) || defined(__HIP_PLATFORM_AMD__)
/* V4.1 IQ2_XXS/Q2_K resident rank reference: global routing IDs, one
 * contiguous half of384 experts, and F32 partial output. */
int ds4_gpu_dsv41_routed_moe_tp_tensor(
        ds4_gpu_tensor *out, ds4_gpu_tensor *gate, ds4_gpu_tensor *up,
        ds4_gpu_tensor *mid, ds4_gpu_tensor *scratch,
        const void *model_map, uint64_t model_size,
        uint64_t gate_offset, uint64_t up_offset, uint64_t down_offset,
        const ds4_gpu_tensor *selected, const ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *x, uint32_t n_tokens, uint32_t tp_rank);
/* Bulk owned gate/up operator. Global routing is preserved; unowned output
 * rows are zero. The caller retains the weighted activation/down boundary. */
int ds4_gpu_dsv41_moe_tp_gate_up(
        ds4_gpu_tensor *gate, ds4_gpu_tensor *up,
        const void *model_map, uint64_t model_size,
        uint64_t gate_offset, uint64_t up_offset,
        const ds4_gpu_tensor *selected, const ds4_gpu_tensor *x,
        uint32_t n_tokens, uint32_t tp_rank);
/* Owned Q2_K down projection with the inherited hot F16-mid and per-expert
 * F16-output boundaries. Scratch retains six slots, zeroing unowned ones. */
int ds4_gpu_dsv41_moe_tp_down(
        ds4_gpu_tensor *out, ds4_gpu_tensor *scratch,
        const ds4_gpu_tensor *mid, const ds4_gpu_tensor *selected,
        const void *model_map, uint64_t model_size, uint64_t down_offset,
        uint32_t n_tokens, uint32_t tp_rank);
/* HC2048 F32 SGEMM: 1 enqueued, 0 unsupported, -1 failure.
 * Never fall back after failure. Full heads scratch remains live on stream0;
 * free the plan before graph tensors. Input is already RMS-normalized;
 * output remains F32 for Sinkhorn. */
typedef struct ds4_gpu_dsv41_hc_plan ds4_gpu_dsv41_hc_plan;
int ds4_gpu_dsv41_hc_project(ds4_gpu_dsv41_hc_plan **plan,
                            ds4_gpu_tensor *out, const void *model_map,
                            uint64_t model_size, uint64_t weight_offset,
                            uint32_t rows, const ds4_gpu_tensor *input,
                            ds4_gpu_tensor *full_heads_scratch);
void ds4_gpu_dsv41_hc_plan_free(ds4_gpu_dsv41_hc_plan *plan);

/* Preserve V4.1 activation formats while applying Q8 weights. */
int ds4_gpu_dsv41_q8_projection_rows(ds4_gpu_tensor *out,
                                    const void *model_map, uint64_t model_size,
                                    uint64_t weight_offset, uint32_t width,
                                    uint32_t outputs, uint32_t rows,
                                    const ds4_gpu_tensor *in);
#endif

#ifdef __cplusplus
}
#endif
#endif
