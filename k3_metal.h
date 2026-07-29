#ifndef K3_METAL_H
#define K3_METAL_H

#include <stdint.h>

#include "ds4_gpu.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Kimi K3 checkpoint-native Metal operations. Model tensor offsets are
 * absolute offsets in the mmaped SafeTensors shard.
 */
int k3_gpu_embedding_bf16(
        ds4_gpu_tensor       *out,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        const ds4_gpu_tensor *token_ids,
        uint32_t              n_tokens,
        uint32_t              n_embd,
        uint32_t              n_vocab);

int k3_gpu_expand_bf16(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *in,
        uint64_t              count);

int k3_gpu_matmul_bf16(
        ds4_gpu_tensor       *out,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint32_t              in_dim,
        uint32_t              out_dim,
        const ds4_gpu_tensor *x,
        uint32_t              n_rows);

int k3_gpu_matmul4_bf16_enabled(void);
int k3_gpu_matmul4_bf16(
        ds4_gpu_tensor * const outputs[4],
        const void            *model_map,
        uint64_t               model_size,
        const uint64_t         weight_offsets[4],
        uint32_t               in_dim,
        uint32_t               out_dim,
        const ds4_gpu_tensor  *x,
        uint32_t               n_rows);

int k3_gpu_matmul_q8_0_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *weights,
        uint64_t              weight_offset,
        uint32_t              in_dim,
        uint32_t              out_dim,
        const ds4_gpu_tensor *x,
        uint32_t              n_rows);

int k3_gpu_gguf_mxfp4_pair_situ_f32(
        ds4_gpu_tensor       *mid,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              gate_offset,
        uint64_t              up_offset,
        const ds4_gpu_tensor *x,
        uint32_t              in_dim,
        uint32_t              out_dim,
        float                 beta,
        float                 linear_beta);

int k3_gpu_request_static_residency(
        ds4_gpu_tensor * const *weights,
        uint32_t                count);
int k3_gpu_pin_static_maps(
        const void * const *maps,
        const uint64_t     *sizes,
        uint32_t            count);
int k3_gpu_pin_model_spans(
        const void     *model_map,
        uint64_t        model_size,
        const uint64_t *offsets,
        const uint64_t *sizes,
        uint32_t        count);
void k3_gpu_clear_static_residency(void);

int k3_gpu_zero_f32_ordered(
        ds4_gpu_tensor *tensor,
        uint64_t        count);

int k3_gpu_rms_norm_bf16(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *x,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint32_t              width,
        uint32_t              n_rows,
        float                 eps);

int k3_gpu_rms_norm_bf16_strided(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *x,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint32_t              width,
        uint32_t              n_rows,
        uint32_t              row_stride,
        float                 eps);

int k3_gpu_situ(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *gate,
        const ds4_gpu_tensor *up,
        uint32_t              width,
        uint32_t              n_rows,
        float                 beta,
        float                 linear_beta);

/*
 * MXFP8 is stored as E4M3FN values plus one E8M0 scale per group of 32.
 * Values and scales are separate because that is also the checkpoint layout
 * for MXFP4 weights and avoids repacking streamed expert tensors.
 */
int k3_gpu_mxfp8_quantize(
        ds4_gpu_tensor       *values,
        ds4_gpu_tensor       *scales,
        const ds4_gpu_tensor *x,
        uint32_t              width,
        uint32_t              n_rows);

int k3_gpu_mxfp4_pair_situ(
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              gate_packed_offset,
        uint64_t              gate_scale_offset,
        uint64_t              up_packed_offset,
        uint64_t              up_scale_offset,
        const ds4_gpu_tensor *x_values,
        const ds4_gpu_tensor *x_scales,
        uint32_t              in_dim,
        uint32_t              out_dim,
        uint32_t              n_rows,
        float                 beta,
        float                 linear_beta);

int k3_gpu_mxfp4_pair_situ_tensor(
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        const ds4_gpu_tensor *weights,
        uint64_t              gate_packed_offset,
        uint64_t              gate_scale_offset,
        uint64_t              up_packed_offset,
        uint64_t              up_scale_offset,
        const ds4_gpu_tensor *x_values,
        const ds4_gpu_tensor *x_scales,
        uint32_t              in_dim,
        uint32_t              out_dim,
        uint32_t              n_rows,
        float                 beta,
        float                 linear_beta);

int k3_gpu_mxfp4_down(
        ds4_gpu_tensor       *out,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              packed_offset,
        uint64_t              scale_offset,
        const ds4_gpu_tensor *x_values,
        const ds4_gpu_tensor *x_scales,
        uint32_t              in_dim,
        uint32_t              out_dim,
        uint32_t              n_rows,
        float                 route_weight,
        int                   accumulate);

int k3_gpu_mxfp4_down_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *weights,
        uint64_t              packed_offset,
        uint64_t              scale_offset,
        const ds4_gpu_tensor *x_values,
        const ds4_gpu_tensor *x_scales,
        uint32_t              in_dim,
        uint32_t              out_dim,
        uint32_t              n_rows,
        float                 route_weight,
        int                   accumulate);

int k3_gpu_gguf_mxfp4_pair_situ(
        ds4_gpu_tensor       *mid,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              gate_offset,
        uint64_t              up_offset,
        const ds4_gpu_tensor *x_values,
        const ds4_gpu_tensor *x_scales,
        uint32_t              in_dim,
        uint32_t              out_dim,
        uint32_t              n_rows,
        float                 beta,
        float                 linear_beta);

int k3_gpu_gguf_mxfp4_down(
        ds4_gpu_tensor       *out,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        const ds4_gpu_tensor *x_values,
        const ds4_gpu_tensor *x_scales,
        uint32_t              in_dim,
        uint32_t              out_dim,
        uint32_t              n_rows);

int k3_gpu_gather_rows(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *x,
        const ds4_gpu_tensor *row_ids,
        uint32_t              n_rows,
        uint32_t              width);

int k3_gpu_scatter_add_rows(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *x,
        const ds4_gpu_tensor *row_ids,
        const ds4_gpu_tensor *row_weights,
        uint32_t              n_rows,
        uint32_t              width);

int k3_gpu_router_topk(
        ds4_gpu_tensor       *selected_ids,
        ds4_gpu_tensor       *selected_weights,
        const ds4_gpu_tensor *logits,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              correction_bias_offset,
        uint32_t              n_rows,
        uint32_t              n_experts,
        uint32_t              top_k);

/*
 * blocks layout is [row, block, width]. Prefix updates are explicitly rounded
 * to BF16 before they become an AttnRes source, matching the reference path.
 * Set block_write_idx to -1 when no block is written, preserve_prefix for a
 * read-only predictor pass, and output_norm_offset to UINT64_MAX when the mixed
 * result should not be normalized.
 */
int k3_gpu_attn_res(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *prefix,
        const ds4_gpu_tensor *delta,
        ds4_gpu_tensor       *blocks,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              norm_weight_offset,
        uint64_t              proj_weight_offset,
        uint64_t              output_norm_offset,
        uint32_t              width,
        uint32_t              n_rows,
        uint32_t              block_capacity,
        uint32_t              num_blocks,
        int32_t               block_write_idx,
        int                   preserve_prefix,
        float                 eps,
        float                 output_norm_eps);

/*
 * Fused single-token KDA. q/k/v/raw_gate/output_gate are [row, head, 128],
 * beta is [row, head], conv_state is [row, qkv, history, head, 128] with
 * history=3, and recurrent_state is [row, head, value, key]. Both states are
 * FP32 to avoid cumulative loss over long contexts.
 */
int k3_gpu_kda_decode(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *conv_state,
        ds4_gpu_tensor       *recurrent_state,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *k,
        const ds4_gpu_tensor *v,
        const ds4_gpu_tensor *raw_gate,
        const ds4_gpu_tensor *raw_beta,
        const ds4_gpu_tensor *output_gate,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              q_conv_offset,
        uint64_t              k_conv_offset,
        uint64_t              v_conv_offset,
        uint64_t              a_log_offset,
        uint64_t              dt_bias_offset,
        uint64_t              output_norm_offset,
        uint32_t              n_heads,
        uint32_t              n_rows,
        float                 gate_lower_bound,
        float                 norm_eps);

/*
 * Prompt-specialized KDA for one sequence. q/k/v/raw_gate are destructive
 * scratch: the convolution/preparation stage replaces them with normalized
 * Q/K, convolved V, and decay. State rows then remain in registers across the
 * whole prompt.
 */
int k3_gpu_kda_prefill(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *conv_state,
        ds4_gpu_tensor       *recurrent_state,
        ds4_gpu_tensor       *q,
        ds4_gpu_tensor       *k,
        ds4_gpu_tensor       *v,
        ds4_gpu_tensor       *raw_gate,
        const ds4_gpu_tensor *raw_beta,
        const ds4_gpu_tensor *output_gate,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              q_conv_offset,
        uint64_t              k_conv_offset,
        uint64_t              v_conv_offset,
        uint64_t              a_log_offset,
        uint64_t              dt_bias_offset,
        uint64_t              output_norm_offset,
        uint32_t              n_heads,
        uint32_t              n_tokens,
        float                 gate_lower_bound,
        float                 norm_eps);

/*
 * Latent MLA decode with a BF16 [capacity, 576] cache. q is
 * [row=1, head, 192], normalized_kv is [512] followed by the shared 64-wide
 * positional key, and output_gate is [head, 128]. The two partial tensors are
 * reusable decode scratch; max_partials must cover ceil((position+1)/1024).
 */
int k3_gpu_mla_decode(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *cache,
        ds4_gpu_tensor       *q_latent,
        ds4_gpu_tensor       *partial_values,
        ds4_gpu_tensor       *partial_stats,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *normalized_kv,
        const ds4_gpu_tensor *output_gate,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              kv_b_proj_offset,
        uint32_t              n_heads,
        uint32_t              position,
        uint32_t              cache_capacity,
        uint32_t              max_partials);

/*
 * Layer-major causal MLA prefill for a contiguous token chunk. q_latent must
 * hold [n_tokens, n_heads, 512]. The BF16 cache may already contain a prefix;
 * start_position is the absolute position of the first row.
 */
int k3_gpu_mla_prefill(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *cache,
        ds4_gpu_tensor       *q_latent,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *normalized_kv,
        const ds4_gpu_tensor *output_gate,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              kv_b_proj_offset,
        uint32_t              n_heads,
        uint32_t              start_position,
        uint32_t              n_tokens,
        uint32_t              cache_capacity);

#ifdef __cplusplus
}
#endif

#endif
