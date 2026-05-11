#ifndef DS4_METAL_H
#define DS4_METAL_H

#include <stdbool.h>
#include <stdint.h>

/* =========================================================================
 * Metal Tensor and Command Lifetime.
 * =========================================================================
 *
 * Opaque device tensor used by the DS4-specific Metal executor.
 *
 * The public Metal API is tensor-resident: activations, KV state, and scratch
 * buffers stay device-owned across the whole prefill/decode command sequence.
 */
typedef struct ds4_metal_tensor ds4_metal_tensor;

int ds4_metal_init(void);
void ds4_metal_cleanup(void);

ds4_metal_tensor *ds4_metal_tensor_alloc(uint64_t bytes);
ds4_metal_tensor *ds4_metal_tensor_view(const ds4_metal_tensor *base, uint64_t offset, uint64_t bytes);
void ds4_metal_tensor_free(ds4_metal_tensor *tensor);
uint64_t ds4_metal_tensor_bytes(const ds4_metal_tensor *tensor);
void *ds4_metal_tensor_contents(ds4_metal_tensor *tensor);
int ds4_metal_tensor_write(ds4_metal_tensor *tensor, uint64_t offset, const void *data, uint64_t bytes);
int ds4_metal_tensor_read(const ds4_metal_tensor *tensor, uint64_t offset, void *data, uint64_t bytes);
int ds4_metal_tensor_copy(ds4_metal_tensor *dst, uint64_t dst_offset,
                          const ds4_metal_tensor *src, uint64_t src_offset,
                          uint64_t bytes);

int ds4_metal_begin_commands(void);
int ds4_metal_flush_commands(void);
int ds4_metal_end_commands(void);
int ds4_metal_synchronize(void);

int ds4_metal_set_model_map(const void *model_map, uint64_t model_size);
int ds4_metal_set_model_map_range(const void *model_map, uint64_t model_size, uint64_t map_offset, uint64_t map_size);
void ds4_metal_set_quality(bool quality);
void ds4_metal_print_memory_report(const char *label);

/* Low-memory streaming buffer pool for weight loading from disk */
#define DS4_STREAM_BUF_BYTES (210 * 1024 * 1024)
#define DS4_ATTN_BUF_IDX 2
#define DS4_LMHEAD_BUF_IDX 3
#define DS4_LMHEAD_BUF_BYTES (600 * 1024 * 1024)
int ds4_metal_stream_buf_alloc(void);
void *ds4_metal_stream_mem(int buf_idx);
void ds4_metal_stream_fill(int buf_idx, const void *src, size_t offset, size_t n);

/* =========================================================================
 * Embeddings and Indexer Helpers.
 * =========================================================================
 *
 * These kernels seed HC state from token embeddings and implement the ratio-4
 * compressed-attention indexer that chooses visible compressed rows.
 */

int ds4_metal_embed_token_hc_tensor(
        ds4_metal_tensor *out_hc,
        const void       *model_map,
        uint64_t          model_size,
        uint64_t          weight_offset,
        uint32_t          n_vocab,
        uint32_t          token,
        uint32_t          n_embd,
        uint32_t          n_hc);

int ds4_metal_embed_tokens_hc_tensor(
        ds4_metal_tensor       *out_hc,
        const ds4_metal_tensor *tokens,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                n_vocab,
        uint32_t                n_tokens,
        uint32_t                n_embd,
        uint32_t                n_hc);

int ds4_metal_indexer_score_one_tensor(
        ds4_metal_tensor       *scores,
        const ds4_metal_tensor *q,
        const ds4_metal_tensor *weights,
        const ds4_metal_tensor *index_comp,
        uint32_t                n_comp,
        uint32_t                n_head,
        uint32_t                head_dim,
        float                   scale);

int ds4_metal_indexer_scores_prefill_tensor(
        ds4_metal_tensor       *scores,
        const ds4_metal_tensor *q,
        const ds4_metal_tensor *weights,
        const ds4_metal_tensor *index_comp,
        uint32_t                n_comp,
        uint32_t                n_tokens,
        uint32_t                n_head,
        uint32_t                head_dim,
        uint32_t                ratio,
        float                   scale);

int ds4_metal_indexer_scores_decode_batch_tensor(
        ds4_metal_tensor       *scores,
        const ds4_metal_tensor *q,
        const ds4_metal_tensor *weights,
        const ds4_metal_tensor *index_comp,
        uint32_t                n_comp,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_head,
        uint32_t                head_dim,
        uint32_t                ratio,
        float                   scale);

int ds4_metal_indexer_topk_tensor(
        ds4_metal_tensor       *selected,
        const ds4_metal_tensor *scores,
        uint32_t                n_comp,
        uint32_t                n_tokens,
        uint32_t                top_k);

int ds4_metal_dsv4_topk_mask_tensor(
        ds4_metal_tensor       *mask,
        const ds4_metal_tensor *topk,
        uint32_t                n_comp,
        uint32_t                n_tokens,
        uint32_t                top_k);

/* =========================================================================
 * Dense Projections, Norms, RoPE, and KV Rounding.
 * =========================================================================
 *
 * The graph uses these primitives for Q/KV projections, HC/output projections,
 * attention output projections, and DS4's tail-only RoPE.
 */

int ds4_metal_matmul_q8_0_tensor(
        ds4_metal_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_metal_tensor *x,
        uint64_t                n_tok);

int ds4_metal_shared_gate_up_swiglu_q8_0_tensor(
        ds4_metal_tensor       *gate,
        ds4_metal_tensor       *up,
        ds4_metal_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_metal_tensor *x);

int ds4_metal_matmul_f16_tensor(
        ds4_metal_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_metal_tensor *x,
        uint64_t                n_tok);

int ds4_metal_matmul_f16_pair_tensor(
        ds4_metal_tensor       *out_a,
        ds4_metal_tensor       *out_b,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_a_offset,
        uint64_t                weight_b_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_metal_tensor *x,
        uint64_t                n_tok);

int ds4_metal_matmul_f32_tensor(
        ds4_metal_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_metal_tensor *x,
        uint64_t                n_tok);

int ds4_metal_repeat_hc_tensor(
        ds4_metal_tensor       *out,
        const ds4_metal_tensor *row,
        uint32_t                n_embd,
        uint32_t                n_hc);

int ds4_metal_rms_norm_plain_tensor(
        ds4_metal_tensor       *out,
        const ds4_metal_tensor *x,
        uint32_t                n,
        float                   eps);

int ds4_metal_rms_norm_plain_rows_tensor(
        ds4_metal_tensor       *out,
        const ds4_metal_tensor *x,
        uint32_t                n,
        uint32_t                rows,
        float                   eps);

int ds4_metal_rms_norm_weight_tensor(
        ds4_metal_tensor       *out,
        const ds4_metal_tensor *x,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                n,
        float                   eps);

int ds4_metal_rms_norm_weight_rows_tensor(
        ds4_metal_tensor       *out,
        const ds4_metal_tensor *x,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                n,
        uint32_t                rows,
        float                   eps);

int ds4_metal_dsv4_qkv_rms_norm_rows_tensor(
        ds4_metal_tensor       *q_out,
        const ds4_metal_tensor *q,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                q_weight_offset,
        uint32_t                q_n,
        ds4_metal_tensor       *kv_out,
        const ds4_metal_tensor *kv,
        uint64_t                kv_weight_offset,
        uint32_t                kv_n,
        uint32_t                rows,
        float                   eps);

int ds4_metal_head_rms_norm_tensor(
        ds4_metal_tensor *x,
        uint32_t          n_tok,
        uint32_t          n_head,
        uint32_t          head_dim,
        float             eps);

int ds4_metal_dsv4_fp8_kv_quantize_tensor(
        ds4_metal_tensor *x,
        uint32_t          n_tok,
        uint32_t          head_dim,
        uint32_t          n_rot);

int ds4_metal_rope_tail_tensor(
        ds4_metal_tensor *x,
        uint32_t          n_tok,
        uint32_t          n_head,
        uint32_t          head_dim,
        uint32_t          n_rot,
        uint32_t          pos0,
        uint32_t          n_ctx_orig,
        bool              inverse,
        float             freq_base,
        float             freq_scale,
        float             ext_factor,
        float             attn_factor,
        float             beta_fast,
        float             beta_slow);

/* Release decode fused KV finalizer: after the standalone RoPE kernel, this
 * performs DS4's FP8 non-RoPE KV round trip and writes the F16-rounded raw
 * attention cache row in one dispatch. */
int ds4_metal_kv_fp8_store_raw_tensor(
        ds4_metal_tensor *kv,
        ds4_metal_tensor *raw_cache,
        uint32_t          raw_cap,
        uint32_t          row,
        uint32_t          head_dim,
        uint32_t          n_rot);

/* Reference/raw-cache primitive kept for prefill and diagnostics.  Decode uses
 * ds4_metal_kv_fp8_store_raw_tensor unless a diagnostic reference path is
 * explicitly selected by the graph driver. */
int ds4_metal_store_raw_kv_tensor(
        ds4_metal_tensor       *raw_cache,
        const ds4_metal_tensor *kv,
        uint32_t                raw_cap,
        uint32_t                row,
        uint32_t                head_dim);

int ds4_metal_store_raw_kv_batch_tensor(
        ds4_metal_tensor       *raw_cache,
        const ds4_metal_tensor *kv,
        uint32_t                raw_cap,
        uint32_t                pos0,
        uint32_t                n_tokens,
        uint32_t                head_dim);

/* =========================================================================
 * KV Compression and Attention.
 * =========================================================================
 *
 * Compressed layers maintain rolling score/KV state and append pooled rows at
 * ratio boundaries.  Attention kernels consume raw SWA rows, compressed rows,
 * and optional indexer masks.
 */

int ds4_metal_compressor_update_tensor(
        const ds4_metal_tensor *kv_cur,
        const ds4_metal_tensor *sc_cur,
        ds4_metal_tensor       *state_kv,
        ds4_metal_tensor       *state_score,
        ds4_metal_tensor       *comp_cache,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint64_t                norm_offset,
        uint32_t                norm_type,
        uint32_t                head_dim,
        uint32_t                ratio,
        uint32_t                pos,
        uint32_t                comp_row,
        uint32_t                n_rot,
        uint32_t                n_ctx_orig,
        float                   freq_base,
        float                   freq_scale,
        float                   ext_factor,
        float                   attn_factor,
        float                   beta_fast,
        float                   beta_slow,
        float                   rms_eps);

int ds4_metal_compressor_store_batch_tensor(
        const ds4_metal_tensor *kv,
        const ds4_metal_tensor *sc,
        ds4_metal_tensor       *state_kv,
        ds4_metal_tensor       *state_score,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint32_t                head_dim,
        uint32_t                ratio,
        uint32_t                pos0,
        uint32_t                n_tokens);

int ds4_metal_compressor_prefill_tensor(
        ds4_metal_tensor       *comp_cache,
        ds4_metal_tensor       *state_kv,
        ds4_metal_tensor       *state_score,
        const ds4_metal_tensor *kv,
        const ds4_metal_tensor *sc,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint64_t                norm_offset,
        uint32_t                norm_type,
        uint32_t                head_dim,
        uint32_t                ratio,
        uint32_t                pos0,
        uint32_t                n_tokens,
        uint32_t                n_rot,
        uint32_t                n_ctx_orig,
        bool                    quantize_fp8,
        float                   freq_base,
        float                   freq_scale,
        float                   ext_factor,
        float                   attn_factor,
        float                   beta_fast,
        float                   beta_slow,
        float                   rms_eps);

int ds4_metal_compressor_prefill_ratio4_replay_tensor(
        ds4_metal_tensor       *comp_cache,
        ds4_metal_tensor       *state_kv,
        ds4_metal_tensor       *state_score,
        const ds4_metal_tensor *kv,
        const ds4_metal_tensor *sc,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint64_t                norm_offset,
        uint32_t                norm_type,
        uint32_t                head_dim,
        uint32_t                pos0,
        uint32_t                n_tokens,
        uint32_t                n_rot,
        uint32_t                n_ctx_orig,
        bool                    quantize_fp8,
        float                   freq_base,
        float                   freq_scale,
        float                   ext_factor,
        float                   attn_factor,
        float                   beta_fast,
        float                   beta_slow,
        float                   rms_eps);

int ds4_metal_compressor_prefill_state_ratio4_tensor(
        ds4_metal_tensor       *state_kv,
        ds4_metal_tensor       *state_score,
        const ds4_metal_tensor *kv_tail,
        const ds4_metal_tensor *sc_tail,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint32_t                head_dim,
        uint32_t                pos0);

int ds4_metal_attention_decode_heads_tensor(
        ds4_metal_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_metal_tensor *q,
        const ds4_metal_tensor *raw_kv,
        uint32_t                n_raw,
        uint32_t                raw_cap,
        uint32_t                raw_start,
        const ds4_metal_tensor *comp_kv,
        uint32_t                n_comp,
        const ds4_metal_tensor *comp_mask,
        uint32_t                use_mask,
        uint32_t                n_head,
        uint32_t                head_dim);

int ds4_metal_attention_prefill_raw_heads_tensor(
        ds4_metal_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_metal_tensor *q,
        const ds4_metal_tensor *raw_kv,
        uint32_t                n_tokens,
        uint32_t                window,
        uint32_t                n_head,
        uint32_t                head_dim);

int ds4_metal_attention_decode_raw_batch_heads_tensor(
        ds4_metal_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_metal_tensor *q,
        const ds4_metal_tensor *raw_kv,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_raw,
        uint32_t                raw_cap,
        uint32_t                raw_start,
        uint32_t                window,
        uint32_t                n_head,
        uint32_t                head_dim);

int ds4_metal_attention_decode_mixed_batch_heads_tensor(
        ds4_metal_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_metal_tensor *q,
        const ds4_metal_tensor *raw_kv,
        const ds4_metal_tensor *comp_kv,
        const ds4_metal_tensor *comp_mask,
        uint32_t                use_comp_mask,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_raw,
        uint32_t                raw_cap,
        uint32_t                raw_start,
        uint32_t                n_comp,
        uint32_t                window,
        uint32_t                ratio,
        uint32_t                n_head,
        uint32_t                head_dim);

int ds4_metal_attention_indexed_mixed_batch_heads_tensor(
        ds4_metal_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_metal_tensor *q,
        const ds4_metal_tensor *raw_kv,
        const ds4_metal_tensor *comp_kv,
        const ds4_metal_tensor *topk,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_raw,
        uint32_t                raw_cap,
        uint32_t                raw_start,
        uint32_t                n_comp,
        uint32_t                top_k,
        uint32_t                window,
        uint32_t                ratio,
        uint32_t                n_head,
        uint32_t                head_dim);

int ds4_metal_attention_prefill_static_mixed_heads_tensor(
        ds4_metal_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_metal_tensor *q,
        const ds4_metal_tensor *raw_kv,
        const ds4_metal_tensor *comp_kv,
        uint32_t                n_tokens,
        uint32_t                n_comp,
        uint32_t                window,
        uint32_t                ratio,
        uint32_t                n_head,
        uint32_t                head_dim);

int ds4_metal_attention_prefill_masked_mixed_heads_tensor(
        ds4_metal_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_metal_tensor *q,
        const ds4_metal_tensor *raw_kv,
        const ds4_metal_tensor *comp_kv,
        const ds4_metal_tensor *comp_mask,
        uint32_t                n_tokens,
        uint32_t                n_comp,
        uint32_t                window,
        uint32_t                ratio,
        uint32_t                n_head,
        uint32_t                head_dim);

int ds4_metal_attention_output_q8_batch_tensor(
        ds4_metal_tensor       *out,
        ds4_metal_tensor       *low,
        ds4_metal_tensor       *group_tmp,
        ds4_metal_tensor       *low_tmp,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                out_a_offset,
        uint64_t                out_b_offset,
        uint64_t                group_dim,
        uint64_t                rank,
        uint32_t                n_groups,
        uint64_t                out_dim,
        const ds4_metal_tensor *heads,
        uint32_t                n_tokens);

int ds4_metal_attention_output_low_q8_tensor(
        ds4_metal_tensor       *low,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                out_a_offset,
        uint64_t                group_dim,
        uint64_t                rank,
        uint32_t                n_groups,
        const ds4_metal_tensor *heads);

/* =========================================================================
 * Router, Shared Expert, and Routed MoE.
 * =========================================================================
 *
 * These kernels implement the FFN body: router probabilities/top-k or hash
 * routing, shared SwiGLU, and the IQ2_XXS/Q2_K/Q4_K routed experts.
 */

int ds4_metal_swiglu_tensor(
        ds4_metal_tensor       *out,
        const ds4_metal_tensor *gate,
        const ds4_metal_tensor *up,
        uint32_t                n,
        float                   clamp,
        float                   weight);

int ds4_metal_add_tensor(
        ds4_metal_tensor       *out,
        const ds4_metal_tensor *a,
        const ds4_metal_tensor *b,
        uint32_t                n);

int ds4_metal_router_select_tensor(
        ds4_metal_tensor       *selected,
        ds4_metal_tensor       *weights,
        ds4_metal_tensor       *probs,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                bias_offset,
        uint64_t                hash_offset,
        uint32_t                hash_rows,
        uint32_t                token,
        uint32_t                n_expert_groups,
        uint32_t                n_group_used,
        bool                    has_bias,
        bool                    hash_mode,
        const ds4_metal_tensor *logits);

int ds4_metal_router_select_batch_tensor(
        ds4_metal_tensor       *selected,
        ds4_metal_tensor       *weights,
        ds4_metal_tensor       *probs,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                bias_offset,
        uint64_t                hash_offset,
        uint32_t                hash_rows,
        uint32_t                n_expert_groups,
        uint32_t                n_group_used,
        bool                    has_bias,
        bool                    hash_mode,
        const ds4_metal_tensor *logits,
        const ds4_metal_tensor *tokens,
        uint32_t                n_tokens);

int ds4_metal_routed_moe_one_tensor(
        ds4_metal_tensor       *out,
        ds4_metal_tensor       *gate,
        ds4_metal_tensor       *up,
        ds4_metal_tensor       *mid,
        ds4_metal_tensor       *experts,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                down_offset,
        uint32_t                gate_type,
        uint32_t                down_type,
        uint64_t                gate_expert_bytes,
        uint64_t                gate_row_bytes,
        uint64_t                down_expert_bytes,
        uint64_t                down_row_bytes,
        uint32_t                expert_in_dim,
        uint32_t                expert_mid_dim,
        uint32_t                out_dim,
        const ds4_metal_tensor *selected,
        const ds4_metal_tensor *weights,
        uint32_t                n_expert,
        float                   clamp,
        const ds4_metal_tensor *x);

int ds4_metal_routed_moe_batch_tensor(
        ds4_metal_tensor       *out,
        ds4_metal_tensor       *gate,
        ds4_metal_tensor       *up,
        ds4_metal_tensor       *mid,
        ds4_metal_tensor       *experts,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                down_offset,
        uint32_t                gate_type,
        uint32_t                down_type,
        uint64_t                gate_expert_bytes,
        uint64_t                gate_row_bytes,
        uint64_t                down_expert_bytes,
        uint64_t                down_row_bytes,
        uint32_t                expert_in_dim,
        uint32_t                expert_mid_dim,
        uint32_t                out_dim,
        const ds4_metal_tensor *selected,
        const ds4_metal_tensor *weights,
        uint32_t                n_expert,
        float                   clamp,
        const ds4_metal_tensor *x,
        uint32_t                n_tokens);

/* =========================================================================
 * Hyper-Connection Kernels.
 * =========================================================================
 *
 * HC kernels reduce four residual streams before a sublayer and expand the
 * sublayer output back into four streams afterward.
 */

int ds4_metal_hc_split_sinkhorn_tensor(
        ds4_metal_tensor       *out,
        const ds4_metal_tensor *mix,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                scale_offset,
        uint64_t                base_offset,
        uint32_t                n_hc,
        uint32_t                sinkhorn_iters,
        float                   eps);

int ds4_metal_hc_weighted_sum_tensor(
        ds4_metal_tensor       *out,
        const ds4_metal_tensor *residual_hc,
        const ds4_metal_tensor *weights,
        uint32_t                n_embd,
        uint32_t                n_hc);

int ds4_metal_hc_weighted_sum_split_tensor(
        ds4_metal_tensor       *out,
        const ds4_metal_tensor *residual_hc,
        const ds4_metal_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc);

/* Release decode fused HC pre-sublayer operation: split the HC mixer and
 * immediately reduce four HC streams into the active 4096-wide sublayer row. */
int ds4_metal_hc_split_weighted_sum_tensor(
        ds4_metal_tensor       *out,
        ds4_metal_tensor       *split,
        const ds4_metal_tensor *mix,
        const ds4_metal_tensor *residual_hc,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                scale_offset,
        uint64_t                base_offset,
        uint32_t                n_embd,
        uint32_t                n_hc,
        uint32_t                sinkhorn_iters,
        float                   eps);

int ds4_metal_hc_split_weighted_sum_norm_tensor(
        ds4_metal_tensor       *out,
        ds4_metal_tensor       *norm_out,
        ds4_metal_tensor       *split,
        const ds4_metal_tensor *mix,
        const ds4_metal_tensor *residual_hc,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                scale_offset,
        uint64_t                base_offset,
        uint64_t                norm_weight_offset,
        uint32_t                n_embd,
        uint32_t                n_hc,
        uint32_t                sinkhorn_iters,
        float                   eps,
        float                   norm_eps);

int ds4_metal_output_hc_weights_tensor(
        ds4_metal_tensor       *out,
        const ds4_metal_tensor *pre,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                scale_offset,
        uint64_t                base_offset,
        uint32_t                n_hc,
        float                   eps);

int ds4_metal_hc_expand_tensor(
        ds4_metal_tensor       *out_hc,
        const ds4_metal_tensor *block_out,
        const ds4_metal_tensor *residual_hc,
        const ds4_metal_tensor *post,
        const ds4_metal_tensor *comb,
        uint32_t                n_embd,
        uint32_t                n_hc);

int ds4_metal_hc_expand_split_tensor(
        ds4_metal_tensor       *out_hc,
        const ds4_metal_tensor *block_out,
        const ds4_metal_tensor *residual_hc,
        const ds4_metal_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc);

int ds4_metal_hc_expand_add_split_tensor(
        ds4_metal_tensor       *out_hc,
        const ds4_metal_tensor *block_out,
        const ds4_metal_tensor *block_add,
        const ds4_metal_tensor *residual_hc,
        const ds4_metal_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc);

int ds4_metal_shared_down_hc_expand_q8_0_tensor(
        ds4_metal_tensor       *out_hc,
        ds4_metal_tensor       *shared_out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_metal_tensor *shared_mid,
        const ds4_metal_tensor *routed_out,
        const ds4_metal_tensor *residual_hc,
        const ds4_metal_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc);

int ds4_metal_matmul_q8_0_hc_expand_tensor(
        ds4_metal_tensor       *out_hc,
        ds4_metal_tensor       *block_out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_metal_tensor *x,
        const ds4_metal_tensor *residual_hc,
        const ds4_metal_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc);

#endif
