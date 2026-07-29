/*
 * DeepSeek V4 graph state, allocation, decode, prefill, and diagnostics.
 *
 * Included exactly once inside ds4.c's graph-backend conditional. This keeps
 * all custom GPU orchestration concrete and statically linked.
 */

/*
 * Apple Metal stores the persistent attention-compressed KV cache in F16.  The
 * compressor still pools, normalizes, RoPEs, and FP8-rounds rows in F32 staging
 * before writing the cache, while checkpoints and debug dumps expand back to
 * F32 for the stable external format.  This is a storage optimization rather
 * than a semantic approximation: all Metal attention consumers already run the
 * compressed K/V rows through F16 FlashAttention/indexed-attention paths.
 */
#if defined(__APPLE__)
#define DS4_GPU_ATTN_COMP_CACHE_F16 1
#else
#define DS4_GPU_ATTN_COMP_CACHE_F16 0
#endif

#define DS4_GPU_GLM_COMPACT_CACHE_F16 DS4_GPU_ATTN_COMP_CACHE_F16

/* =========================================================================
 * Metal Release Graph State.
 * =========================================================================
 *
 * The release Metal executor owns one fixed set of tensors for single-token
 * decode and another for batched prefill.  The structure is DS4-specific:
 * tensor names follow the model stages rather than generic graph nodes.
 */

enum { DS4_STREAMING_PREFILL_CACHE_SEED_MAX_TOKENS = 64 };

typedef struct {
    /* Class P — per-tier replicated kernel scratch buffers.
     * Each used tier has its own copy; active_tier names the slot the
     * current dispatch step reads/writes. Single-tier paths leave
     * active_tier == 0; multi-tier dispatch updates active_tier in B6.
     *
     * Decode hidden-state buffers. A generated token enters as an embedding
     * in cur_hc and leaves as logits after all 43 layers update their
     * raw/compressed/indexer caches. The hc_pre / hc_post / hc_comb views
     * are derived from hc_split per tier (see metal_graph_alloc_raw_cap). */
    ds4_gpu_tensor *cur_hc_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *flat_hc_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *hc_mix_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *hc_split_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *hc_pre_by_tier[DS4_MAX_GPUS];   /* views of hc_split */
    ds4_gpu_tensor *hc_post_by_tier[DS4_MAX_GPUS];  /* views of hc_split */
    ds4_gpu_tensor *hc_comb_by_tier[DS4_MAX_GPUS];  /* views of hc_split */
    ds4_gpu_tensor *attn_cur_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *attn_norm_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *qr_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *qr_norm_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *q_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *kv_raw_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *kv_by_tier[DS4_MAX_GPUS];
    int active_tier;
    /* cached engine placement[] (length DS4_N_LAYER + 2) for the
     * dispatch loops. NULL in single-tier mode — active_tier stays 0 and
     * dispatch wrappers no-op the tier-switch + cross-device copy. The
     * pointer aliases e->placement; the engine outlives the graph so this
     * is safe. */
    const int *placement;

    /* Persistent KV state.  Raw KV is a sliding-window ring per layer.  Ratio-4
     * layers also keep an indexer-compressed cache; ratio-128 layers keep only
     * the attention-compressed cache.  The small state tensors are compressor
     * frontiers for the next compressed row, so they must be snapshotted with
     * the row counters whenever a checkpoint is saved or partially rewound. */
    ds4_gpu_tensor *layer_raw_cache[DS4_MAX_LAYER];
    ds4_gpu_tensor *layer_attn_comp_cache[DS4_MAX_LAYER];
    ds4_gpu_tensor *layer_attn_state_kv[DS4_MAX_LAYER];
    ds4_gpu_tensor *layer_attn_state_score[DS4_MAX_LAYER];
    ds4_gpu_tensor *layer_index_comp_cache[DS4_MAX_LAYER];
    ds4_gpu_tensor *layer_index_state_kv[DS4_MAX_LAYER];
    ds4_gpu_tensor *layer_index_state_score[DS4_MAX_LAYER];
    ds4_gpu_tensor *layer_raw_cache_tp[DS4_MAX_LAYER];
    ds4_gpu_tensor *layer_attn_comp_cache_tp[DS4_MAX_LAYER];

    /* Speculative decoding scratch.  MTP is allowed to mutate graph state only
     * if the target verifier can either commit it or restore the saved
     * frontiers.  The prefix1 buffers are the cheap partial-accept state for the
     * common N=2 case. */
    ds4_gpu_tensor *spec_attn_state_kv[DS4_MAX_LAYER];
    ds4_gpu_tensor *spec_attn_state_score[DS4_MAX_LAYER];
    ds4_gpu_tensor *spec_index_state_kv[DS4_MAX_LAYER];
    ds4_gpu_tensor *spec_index_state_score[DS4_MAX_LAYER];
    ds4_gpu_tensor *spec_prefix1_attn_state_kv[DS4_MAX_LAYER];
    ds4_gpu_tensor *spec_prefix1_attn_state_score[DS4_MAX_LAYER];
    ds4_gpu_tensor *spec_prefix1_index_state_kv[DS4_MAX_LAYER];
    ds4_gpu_tensor *spec_prefix1_index_state_score[DS4_MAX_LAYER];
    ds4_gpu_tensor *spec_logits;
    uint32_t layer_n_comp[DS4_MAX_LAYER];
    uint32_t layer_n_index_comp[DS4_MAX_LAYER];
    uint32_t spec_prefix1_n_comp[DS4_MAX_LAYER];
    uint32_t spec_prefix1_n_index_comp[DS4_MAX_LAYER];
    bool spec_capture_prefix1;
    uint32_t raw_cap;
    /* Maximum compressed-row capacity across layers.  Shared work buffers use
     * this worst-case size because ratio-4 indexer layers can still reach it. */
    uint32_t comp_cap;
    /* Persistent compressed caches are per layer, so size them from the actual
     * layer compression ratio instead of pessimistically using the ratio-4 cap
     * for every ratio-128 layer. */
    uint32_t layer_comp_cap[DS4_MAX_LAYER];
    uint32_t attn_comp_stage_cap;

    /* Class P (per-layer work tensors). Each used tier has its
     * own replica. They are reused in place by every layer instead of
     * allocating a generic graph arena. This is why the code is verbose but
     * predictable: each pointer names an actual DS4 stage. */
    ds4_gpu_tensor *comp_kv_cur_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *comp_sc_cur_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *attn_comp_stage_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *indexer_q_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *indexer_weights_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *indexer_scores_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *comp_mask_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *comp_selected_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *heads_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *attn_low_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *attn_out_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *after_attn_hc_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *ffn_cur_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *ffn_norm_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *shared_gate_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *shared_up_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *shared_mid_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *shared_out_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *router_logits_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *router_probs_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *router_selected_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *router_weights_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *routed_gate_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *routed_up_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *routed_mid_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *routed_down_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *routed_out_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *ffn_out_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *after_ffn_hc_by_tier[DS4_MAX_GPUS];
    /* Class H — output-head buffers and logits live on the
     * head tier only. head_tier is captured at metal_graph_alloc_raw_cap
     * time from placement[DS4_N_LAYER + 1] (or 0 in single-tier /
     * diagnostic paths). Non-head slots remain NULL. Readers go through
     * the metal_graph_logits / metal_graph_output_* accessors below. */
    ds4_gpu_tensor *output_pre_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *output_weights_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *output_embd_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *output_norm_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *logits_by_tier[DS4_MAX_GPUS];
    int head_tier;

    /* DSpark target features.  The proposer consumes mean-over-HC rows from
     * selected target layers; keeping them on-GPU avoids adding readbacks to
     * the target path. */
    ds4_gpu_tensor *dspark_hc_mean_weights;
    ds4_gpu_tensor *dspark_hc_mean_rows;
    ds4_gpu_tensor *dspark_target_hidden;
    ds4_gpu_tensor *dspark_target_hidden_batch;
    ds4_gpu_tensor *dspark_stage0_packed;
    ds4_gpu_tensor *dspark_stage0_proj;
    ds4_gpu_tensor *dspark_main_x;
    ds4_gpu_tensor *dspark_draft_tokens;
    ds4_gpu_tensor *dspark_draft_hc;
    ds4_gpu_tensor *dspark_target_hc;
    ds4_gpu_tensor *dspark_stage_input_hc;
    ds4_gpu_tensor *dspark_stage_output_hc;
    ds4_gpu_tensor *dspark_position_ids;
    ds4_gpu_tensor *dspark_raw_cache[DS4_DSPARK_MAX_STAGES];
    uint32_t dspark_cache_cap;
    uint32_t dspark_cache_start;
    uint32_t dspark_cache_token_start;
    uint32_t dspark_cache_len;
    uint32_t dspark_target_layer_count;
    uint32_t dspark_block_size;
    uint32_t dspark_target_layers[DS4_DSPARK_MAX_TARGET_LAYERS];
    uint32_t dspark_capture_mask;
    uint32_t dspark_capture_checkpoint_len;
    uint32_t dspark_capture_batch_mask;
    uint32_t dspark_capture_batch_start;
    uint32_t dspark_capture_batch_tokens;
    bool dspark_capture_valid;
    bool dspark_capture_batch_valid;
    int dspark_exec_tier;
    bool dspark_capture_enabled;
    bool verify_small_batch_tp;
    uint32_t pipeline_capture_chunk_start;
    uint32_t pipeline_capture_chunk_len;
    bool ssd_streaming; /* glm-branch SSD streaming; always false here */

    /* Optional MTP model state.  It has its own raw cache because the drafter
     * runs on speculative future tokens; target KV state is updated only after
     * verification accepts draft tokens. */
    ds4_gpu_tensor *mtp_embed;
    ds4_gpu_tensor *mtp_enorm;
    ds4_gpu_tensor *mtp_eproj;
    ds4_gpu_tensor *mtp_eproj_hc;
    ds4_gpu_tensor *mtp_hnorm_hc;
    ds4_gpu_tensor *mtp_hproj_hc;
    ds4_gpu_tensor *mtp_input_hc;
    ds4_gpu_tensor *mtp_state_hc;
    ds4_gpu_tensor *mtp_next_hc;
    ds4_gpu_tensor *mtp_raw_cache;
    uint32_t mtp_n_raw;
    uint32_t prefill_cap;
    uint32_t raw_window;
    uint32_t batch_token_offset;

    /* Batched prefill tensors.  Prefill is layer-major: a chunk of prompt
     * tokens moves through layer 0, then layer 1, and so on, updating the same
     * persistent caches used by decode.  Keeping this separate from decode
     * avoids a slow loop of one-token graph steps for long prompts. */
    /* Class E — embedding-tier-only prompt-token integer buffer.
     * Captured at metal_graph_alloc_raw_cap time from placement[0] (or 0 in
     * single-tier / diagnostic paths). Non-embedding slots stay NULL. Readers
     * go through metal_graph_prefill_tokens() below. */
    ds4_gpu_tensor *prefill_tokens_by_tier[DS4_MAX_GPUS];
    int emb_tier;
    /* Class P batch (chunked-prefill) scratch — per-tier
     * replicated. The cur/next pair is ping-ponged per layer step on the
     * layer's active tier; tier transitions copy the active buffer across
     * boundaries via ds4_gpu_tensor_copy_xdev (handled in B6). */
    ds4_gpu_tensor *batch_cur_hc_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *batch_next_hc_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *batch_flat_hc_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *batch_hc_mix_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *batch_hc_split_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *batch_attn_cur_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *batch_attn_norm_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *batch_qr_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *batch_qr_norm_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *batch_q_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *batch_kv_raw_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *batch_kv_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *batch_comp_kv_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *batch_comp_sc_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *batch_indexer_q_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *batch_indexer_weights_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *batch_heads_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *batch_attn_low_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *batch_attn_out_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *batch_group_tmp_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *batch_low_tmp_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *batch_after_attn_hc_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *batch_ffn_cur_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *batch_ffn_norm_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *batch_shared_gate_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *batch_shared_up_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *batch_shared_mid_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *batch_shared_out_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *batch_router_logits_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *batch_router_probs_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *batch_router_selected_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *batch_router_weights_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *batch_routed_gate_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *batch_routed_up_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *batch_routed_mid_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *batch_routed_down_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *batch_routed_out_by_tier[DS4_MAX_GPUS];
    bool batch_routed_mid_is_f16;
    ds4_gpu_tensor *batch_ffn_out_by_tier[DS4_MAX_GPUS];
    bool owns_prefill_workspace;
    bool materialize_ffn_out;
    /* Class P (replicated per tier — this is
     * consumed in per-layer attn/FFN kernels, NOT embedding-only). Read-only
     * after init; replicate by writing the same host directions buffer to
     * every used tier's slot during session setup. */
    ds4_gpu_tensor *directional_steering_dirs_by_tier[DS4_MAX_GPUS];
    float directional_steering_attn_scale;
    float directional_steering_ffn_scale;
    bool cuda_tp_decode;
    bool cuda_tp_attn;
    bool cuda_tp_attn_peer_read;
    bool cuda_tp_attn_heads;
    bool cuda_tp_attn_cache_dup;
    bool cuda_tp_moe;
    bool cuda_tp_ep;
    bool cuda_tp_ep_pack_exact;
    bool cuda_tp_moe_delay_reduce;
    bool cuda_tp_moe_copy3_handoff;
    bool cuda_tp_moe_pack_handoff;
    bool cuda_tp_moe_peer_read;
    bool cuda_tp_moe_peer_router;
    bool cuda_tp_shared;
    bool cuda_tp_shared_fold;
    bool cuda_tp_q;
    bool cuda_tp_output;
    bool cuda_tp_prefill_ffn;
    bool cuda_tp_prefill_attn_output;
    bool cuda_q_norm_rope_fuse;
    bool cuda_qkv_kv_rope_fuse;
    bool cuda_qkv_pair;
    bool cuda_tp_attn_out_hc_fuse;
    bool shared_gate_up_swiglu_fuse;
    bool decode_stage_profile;
    bool decode_index_stage_profile;
    bool output_stage_profile;
    ds4_gpu_tensor *tp_peer_tmp_by_tier[DS4_MAX_GPUS];
    uint32_t power_percent;
    double prefill_layer_avg_sec[DS4_MAX_LAYER];
    double decode_token_avg_sec;
    bool quality;
    bool mtp_enabled;
    /* Metal-only prefill helpers retained alongside the CUDA tiered workspace. */
    ds4_gpu_tensor *batch_q_half;
    ds4_gpu_tensor *prefill_seed_router_selected;
    uint32_t prefill_seed_tokens;
    uint64_t prefill_selected_profile_rows;
    uint64_t prefill_selected_profile_unique;
    uint64_t prefill_selected_profile_selected_bytes;
    uint64_t prefill_selected_profile_full_bytes;
    uint32_t prefill_selected_profile_layers;
    uint32_t prefill_selected_profile_min_unique;
    uint32_t prefill_selected_profile_max_unique;
    uint32_t streaming_preload_experts;
    bool ssd_streaming_cold;
    bool streaming_static_decode_map_current;
    float *cpu_router_norm;

    /* Metal network tensor parallelism. These views alias engine-owned
     * transport slabs except tp_logits_half, whose view object is session-owned. */
    uint32_t tp_world;
    uint32_t tp_rank;
    ds4_gpu_tensor **tp_out;
    ds4_gpu_tensor **tp_in;
    ds4_gpu_tensor **tp_batch_out;
    ds4_gpu_tensor **tp_batch_in;
    uint32_t tp_batch_rows;
    ds4_gpu_tensor *tp_zero;
    ds4_gpu_tensor *tp_logits_half;
} ds4_gpu_graph;

/* Tensors that are temporary for chunked prefill and grouped multi-session
 * decode. The batched server serializes every operation that uses them, so one
 * engine-owned set can be aliased by all resident session graphs. */
#define DS4_GPU_PREFILL_WORKSPACE_FIELDS(X) \
    X(prefill_tokens)                       \
    X(batch_ffn_out)                        \
    X(batch_routed_out)                     \
    X(batch_routed_down)                    \
    X(batch_routed_mid)                     \
    X(batch_routed_up)                      \
    X(batch_routed_gate)                    \
    X(batch_router_weights)                 \
    X(batch_router_selected)                \
    X(batch_router_probs)                   \
    X(batch_router_logits)                  \
    X(batch_shared_out)                     \
    X(batch_shared_mid)                     \
    X(batch_shared_up)                      \
    X(batch_shared_gate)                    \
    X(batch_ffn_norm)                       \
    X(batch_ffn_cur)                        \
    X(batch_after_attn_hc)                  \
    X(batch_low_tmp)                        \
    X(batch_group_tmp)                      \
    X(batch_attn_out)                       \
    X(batch_attn_low)                       \
    X(batch_heads)                          \
    X(batch_indexer_weights)                \
    X(batch_indexer_q)                      \
    X(batch_comp_sc)                        \
    X(batch_comp_kv)                        \
    X(batch_kv)                             \
    X(batch_kv_raw)                         \
    X(batch_q)                              \
    X(batch_qr_norm)                        \
    X(batch_qr)                             \
    X(batch_attn_norm)                      \
    X(batch_attn_cur)                       \
    X(batch_hc_split)                       \
    X(batch_hc_mix)                         \
    X(batch_flat_hc)                        \
    X(batch_next_hc)                        \
    X(batch_cur_hc)

/* Class H accessors. All reader sites for the output-head
 * tensors and the final logits route through these inlines, which read the
 * head_tier slot captured at allocation time. Single-tier paths set
 * head_tier == 0 and the slot is byte-identical to the legacy
 * metal_graph_logits(g) / g->output_* pointers. Multi-tier paths set head_tier
 * to placement[DS4_N_LAYER + 1]; other tier slots remain NULL. */
static inline ds4_gpu_tensor *metal_graph_logits(const ds4_gpu_graph *g) {
    return g->logits_by_tier[g->head_tier];
}
static inline ds4_gpu_tensor *metal_graph_output_pre(const ds4_gpu_graph *g) {
    return g->output_pre_by_tier[g->head_tier];
}
static inline ds4_gpu_tensor *metal_graph_output_weights(const ds4_gpu_graph *g) {
    return g->output_weights_by_tier[g->head_tier];
}
static inline ds4_gpu_tensor *metal_graph_output_embd(const ds4_gpu_graph *g) {
    return g->output_embd_by_tier[g->head_tier];
}
static inline ds4_gpu_tensor *metal_graph_output_norm(const ds4_gpu_graph *g) {
    return g->output_norm_by_tier[g->head_tier];
}

/* Class E accessor. The prompt-token integer buffer is
 * consumed by the embedding kernel on the embedding tier only. Single-tier
 * paths set emb_tier == 0 (byte-equivalent to the legacy single-tier
 * pointer). Multi-tier paths set emb_tier = placement[0]. */
static inline ds4_gpu_tensor *metal_graph_prefill_tokens(const ds4_gpu_graph *g) {
    return g->prefill_tokens_by_tier[g->emb_tier];
}

/* Class P accessors. Each Class P kernel-scratch buffer is
 * replicated across every tier the placement uses; the active_tier field
 * names the slot the current dispatch step reads/writes. Single-tier paths
 * leave active_tier == 0 (byte-equivalent to the legacy single-tier
 * pointer). Multi-tier dispatch (wired up in B6) updates active_tier with
 * the current layer's home tier before each kernel-dispatch wrapper runs. */
#define DS4_GPU_GRAPH_CLASS_P_ACCESSOR(name) \
static inline ds4_gpu_tensor *metal_graph_##name(const ds4_gpu_graph *g) { \
    return g->name##_by_tier[g->active_tier]; \
}

DS4_GPU_GRAPH_CLASS_P_ACCESSOR(cur_hc)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(flat_hc)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(hc_mix)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(hc_split)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(hc_pre)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(hc_post)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(hc_comb)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(attn_cur)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(attn_norm)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(qr)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(qr_norm)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(q)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(kv_raw)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(kv)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(comp_kv_cur)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(comp_sc_cur)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(attn_comp_stage)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(indexer_q)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(indexer_weights)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(indexer_scores)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(comp_mask)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(comp_selected)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(heads)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(attn_low)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(attn_out)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(after_attn_hc)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(ffn_cur)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(ffn_norm)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(shared_gate)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(shared_up)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(shared_mid)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(shared_out)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(router_logits)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(router_probs)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(router_selected)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(router_weights)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(routed_gate)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(routed_up)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(routed_mid)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(routed_down)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(routed_out)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(ffn_out)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(after_ffn_hc)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(batch_cur_hc)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(batch_next_hc)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(batch_flat_hc)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(batch_hc_mix)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(batch_hc_split)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(batch_attn_cur)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(batch_attn_norm)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(batch_qr)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(batch_qr_norm)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(batch_q)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(batch_kv_raw)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(batch_kv)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(batch_comp_kv)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(batch_comp_sc)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(batch_indexer_q)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(batch_indexer_weights)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(batch_heads)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(batch_attn_low)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(batch_attn_out)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(batch_group_tmp)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(batch_low_tmp)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(batch_after_attn_hc)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(batch_ffn_cur)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(batch_ffn_norm)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(batch_shared_gate)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(batch_shared_up)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(batch_shared_mid)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(batch_shared_out)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(batch_router_logits)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(batch_router_probs)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(batch_router_selected)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(batch_router_weights)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(batch_routed_gate)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(batch_routed_up)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(batch_routed_mid)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(batch_routed_down)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(batch_routed_out)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(batch_ffn_out)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(directional_steering_dirs)

/* dispatch-loop helpers for multi-tier per-layer execution.
 *
 * Single-tier (g->placement == NULL): all helpers are no-ops; active_tier
 * stays 0 from memset; behavior is byte-equivalent to legacy.
 *
 * Multi-tier: each helper switches g->active_tier to the requested tier
 * BEFORE the next kernel-dispatch wrapper reads any Class P accessor. If
 * the source-tier Class P cur_hc (or batch_cur_hc) differs from the new
 * tier's, ds4_gpu_tensor_copy_xdev ferries the active hidden state across
 * the boundary. copy_xdev returns 1 on success, 0 on failure. The
 * destination tensor's device_id was stamped at alloc_on time and is
 * immutable.
 *
 * For decode (one token at a time): metal_graph_set_active_tier_decode
 * swaps to the requested tier and copies cur_hc across the boundary.
 *
 * For batch (chunked prefill): metal_graph_set_active_tier_batch swaps
 * tier and copies batch_cur_hc across the boundary. The next/cur pair
 * is maintained per tier — after a copy, batch_next_hc on the destination
 * tier becomes the swap target for the next layer step on that tier.
 *
 * Helpers always invoke ds4_gpu_set_current_device(tier) so the next
 * kernel-launch sees the correct CUDA device. */

/* ds4_gpu_set_current_device is declared in ds4_gpu_mgpu.h — single-tier
 * (g_n_gpus <= 1) callers no-op. Returns 0 on success. */

#ifdef DS4_NO_GPU
static inline int ds4_gpu_set_current_device(int tier) { (void)tier; return 0; }
static inline int ds4_gpu_tensor_copy_xdev(ds4_gpu_tensor *dst,
                                            const ds4_gpu_tensor *src,
                                            uint64_t bytes) {
    (void)dst; (void)src; (void)bytes; return 1;
}
static inline int ds4_gpu_tensor_copy_xdev3(ds4_gpu_tensor       *dst0,
                                             const ds4_gpu_tensor *src0,
                                             uint64_t              bytes0,
                                             ds4_gpu_tensor       *dst1,
                                             const ds4_gpu_tensor *src1,
                                             uint64_t              bytes1,
                                             ds4_gpu_tensor       *dst2,
                                             const ds4_gpu_tensor *src2,
                                             uint64_t              bytes2) {
    (void)dst0; (void)src0; (void)bytes0;
    (void)dst1; (void)src1; (void)bytes1;
    (void)dst2; (void)src2; (void)bytes2;
    return 1;
}
static inline int ds4_gpu_tensor_copy_xdev_ordered(ds4_gpu_tensor *dst,
                                                    const ds4_gpu_tensor *src,
                                                    uint64_t bytes) {
    (void)dst; (void)src; (void)bytes; return 1;
}
static inline int ds4_gpu_tensor_wait_xdev(const ds4_gpu_tensor *src, int dst_tier) {
    (void)src; (void)dst_tier; return 1;
}
static inline int ds4_gpu_moe_handoff_pack_tensor(
        ds4_gpu_tensor       *packed,
        const ds4_gpu_tensor *ffn_norm,
        const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *weights,
        uint32_t              n_embd,
        uint32_t              n_expert) {
    (void)packed; (void)ffn_norm; (void)selected; (void)weights;
    (void)n_embd; (void)n_expert; return 1;
}
static inline int ds4_gpu_q8_cache_suppressed(void) { return 0; }
static inline void ds4_gpu_set_q8_cache_suppressed(int suppressed) { (void)suppressed; }
static inline int ds4_gpu_set_decode_fast_attention(int enabled) {
    (void)enabled;
    return 0;
}
static inline int ds4_gpu_set_decode_score_vec4(int enabled) {
    (void)enabled;
    return 0;
}
static inline int ds4_gpu_dsv4_qkv_rms_norm_rows_kv_rope_tensor(
        ds4_gpu_tensor       *q_out,
        const ds4_gpu_tensor *q,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                q_weight_offset,
        uint32_t                q_n,
        ds4_gpu_tensor       *kv_out,
        const ds4_gpu_tensor *kv,
        uint64_t                kv_weight_offset,
        uint32_t                kv_n,
        uint32_t                rows,
        uint32_t                kv_n_head,
        uint32_t                kv_head_dim,
        uint32_t                n_rot,
        uint32_t                pos0,
        uint32_t                n_ctx_orig,
        bool                    inverse,
        float                   freq_base,
        float                   freq_scale,
        float                   ext_factor,
        float                   attn_factor,
        float                   beta_fast,
        float                   beta_slow,
        float                   eps) {
    (void)q_out; (void)q; (void)model_map; (void)model_size;
    (void)q_weight_offset; (void)q_n; (void)kv_out; (void)kv;
    (void)kv_weight_offset; (void)kv_n; (void)rows; (void)kv_n_head;
    (void)kv_head_dim; (void)n_rot; (void)pos0; (void)n_ctx_orig;
    (void)inverse; (void)freq_base; (void)freq_scale; (void)ext_factor;
    (void)attn_factor; (void)beta_fast; (void)beta_slow; (void)eps;
    return 0;
}
#endif

/* Returns true on success. Single-tier: no-op success. Multi-tier:
 * sets the CUDA device, then if tier differs from current active_tier,
 * copies cur_hc to the destination tier and updates active_tier. */
static bool metal_graph_set_active_tier_decode(ds4_gpu_graph *g, int tier) {
    if (!g->placement) {
        /* Single-tier: just keep active_tier at 0; no device switch needed. */
        (void)tier;
        return true;
    }
    if (tier < 0 || tier >= DS4_MAX_GPUS) return false;
    if (tier == g->active_tier) return true;
    if (ds4_gpu_set_current_device(tier) != 0) return false;
    /* Boundary hop: copy cur_hc from source-tier to destination-tier slot. */
    if (g->active_tier >= 0) {
        ds4_gpu_tensor *src = g->cur_hc_by_tier[g->active_tier];
        ds4_gpu_tensor *dst = g->cur_hc_by_tier[tier];
        if (src && dst) {
            const uint64_t hc_bytes = (uint64_t)DS4_N_HC * DS4_N_EMBD * sizeof(float);
            if (!ds4_gpu_tensor_copy_xdev(dst, src, hc_bytes)) return false;
        }
    }
    g->active_tier = tier;
    return true;
}

/* Returns true on success. Same semantics as the decode helper but ferries
 * batch_cur_hc (which contains chunk_tokens * hc_dim floats — variable per
 * prefill call). The caller passes the chunk size in tokens; single-tier
 * paths ignore the argument. */
static bool metal_graph_set_active_tier_batch(ds4_gpu_graph *g, int tier, uint32_t chunk_tokens) {
    if (!g->placement) {
        (void)tier;
        (void)chunk_tokens;
        return true;
    }
    if (tier < 0 || tier >= DS4_MAX_GPUS) return false;
    if (tier == g->active_tier) return true;
    if (ds4_gpu_set_current_device(tier) != 0) return false;
    if (g->active_tier >= 0) {
        ds4_gpu_tensor *src = g->batch_cur_hc_by_tier[g->active_tier];
        ds4_gpu_tensor *dst = g->batch_cur_hc_by_tier[tier];
        if (src && dst) {
            const uint64_t hc_bytes =
                (uint64_t)chunk_tokens * DS4_N_HC * DS4_N_EMBD * sizeof(float);
            if (!ds4_gpu_tensor_copy_xdev(dst, src, hc_bytes)) return false;
        }
    }
    g->active_tier = tier;
    return true;
}

static bool metal_graph_set_active_tier_no_copy(ds4_gpu_graph *g, int tier) {
    if (!g->placement) {
        (void)tier;
        return true;
    }
    if (tier < 0 || tier >= DS4_MAX_GPUS) return false;
    if (tier == g->active_tier) return true;
    if (ds4_gpu_set_current_device(tier) != 0) return false;
    g->active_tier = tier;
    return true;
}

/* Upstream: --power N GPU duty-cycle throttling helpers. The single-tier
 * --power=100 path is a no-op; multi-tier inherits the same helpers via
 * graph_power_note_prefill_layer / graph_power_note_decode_token which we
 * call from the shared encode / decode loops. */

static bool graph_power_throttle_enabled(const ds4_gpu_graph *g) {
    return g && g->power_percent > 0 && g->power_percent < 100;
}

static double graph_power_update_avg(double avg, double sample) {
    if (sample <= 0.0 || !isfinite(sample)) return avg;
    if (avg <= 0.0 || !isfinite(avg)) return sample;
    return avg * 0.875 + sample * 0.125;
}

static void graph_power_sleep(double work_sec, uint32_t power_percent) {
    if (power_percent == 0 || power_percent >= 100) return;
    /* Target duty cycle: work / (work + sleep) = power / 100.
     * At --power 50 this sleeps for one measured work interval; at 25 it
     * sleeps for three. */
    const double sleep = work_sec * (100.0 - (double)power_percent) /
                         (double)power_percent;
    sleep_sec(sleep);
}

static void graph_power_note_prefill_layer(ds4_gpu_graph *g,
                                           uint32_t il,
                                           double elapsed_sec) {
    if (!graph_power_throttle_enabled(g)) return;
    if (il >= DS4_N_LAYER) return;
    g->prefill_layer_avg_sec[il] =
        graph_power_update_avg(g->prefill_layer_avg_sec[il], elapsed_sec);
    graph_power_sleep(g->prefill_layer_avg_sec[il], g->power_percent);
}

static void graph_power_note_decode_token(ds4_gpu_graph *g, double elapsed_sec) {
    if (!graph_power_throttle_enabled(g)) return;
    g->decode_token_avg_sec =
        graph_power_update_avg(g->decode_token_avg_sec, elapsed_sec);
    graph_power_sleep(g->decode_token_avg_sec, g->power_percent);
}

static void metal_graph_copy_prefill_workspace_pointers(
        ds4_gpu_graph       *dst,
        const ds4_gpu_graph *src) {
#define DS4_COPY_PREFILL_FIELD(name) \
    memcpy(dst->name##_by_tier, src->name##_by_tier, \
           sizeof(dst->name##_by_tier));
    DS4_GPU_PREFILL_WORKSPACE_FIELDS(DS4_COPY_PREFILL_FIELD)
#undef DS4_COPY_PREFILL_FIELD
    dst->batch_q_half = src->batch_q_half;
    dst->prefill_seed_router_selected = src->prefill_seed_router_selected;
}

static void metal_graph_transfer_prefill_workspace(
        ds4_gpu_graph *dst,
        ds4_gpu_graph *src) {
    memset(dst, 0, sizeof(*dst));
    dst->prefill_cap = src->prefill_cap;
    dst->emb_tier = src->emb_tier;
    dst->owns_prefill_workspace = true;
    metal_graph_copy_prefill_workspace_pointers(dst, src);
    src->owns_prefill_workspace = false;
}

static uint64_t metal_graph_prefill_workspace_bytes(const ds4_gpu_graph *g) {
    uint64_t total = 0;
    for (int t = 0; t < DS4_MAX_GPUS; t++) {
#define DS4_COUNT_PREFILL_FIELD(name) \
        total += ds4_gpu_tensor_bytes(g->name##_by_tier[t]);
        DS4_GPU_PREFILL_WORKSPACE_FIELDS(DS4_COUNT_PREFILL_FIELD)
#undef DS4_COUNT_PREFILL_FIELD
    }
    total += ds4_gpu_tensor_bytes(g->batch_q_half);
    total += ds4_gpu_tensor_bytes(g->prefill_seed_router_selected);
    return total;
}

static void metal_graph_free_prefill_workspace(ds4_gpu_graph *g) {
    if (!g || !g->owns_prefill_workspace) return;
    for (int t = 0; t < DS4_MAX_GPUS; t++) {
#define DS4_FREE_PREFILL_FIELD(name)                 \
        ds4_gpu_tensor_free(g->name##_by_tier[t]);   \
        g->name##_by_tier[t] = NULL;
        DS4_GPU_PREFILL_WORKSPACE_FIELDS(DS4_FREE_PREFILL_FIELD)
#undef DS4_FREE_PREFILL_FIELD
    }
    ds4_gpu_tensor_free(g->batch_q_half);
    ds4_gpu_tensor_free(g->prefill_seed_router_selected);
    g->batch_q_half = NULL;
    g->prefill_seed_router_selected = NULL;
    g->owns_prefill_workspace = false;
}

/* Release every Metal tensor owned by the whole-model graph runtime. */
static void metal_graph_free(ds4_gpu_graph *g) {
    /* free every Class P slot across all DS4_MAX_GPUS tier
     * slots. Unallocated slots are NULL and ds4_gpu_tensor_free(NULL) is a
     * no-op. The hc_pre / hc_post / hc_comb views must be freed BEFORE
     * their parent hc_split — view destruction releases its own struct
     * but does not touch the parent's memory. */
    metal_graph_free_prefill_workspace(g);
    for (int t = 0; t < DS4_MAX_GPUS; t++) {
        ds4_gpu_tensor_free(g->directional_steering_dirs_by_tier[t]);
        g->directional_steering_dirs_by_tier[t] = NULL;
    }
    /* Class H free across all tier slots. Non-head slots are
     * NULL and ds4_gpu_tensor_free(NULL) is a no-op. */
    for (int t = 0; t < DS4_MAX_GPUS; t++) {
        ds4_gpu_tensor_free(g->logits_by_tier[t]);
        g->logits_by_tier[t] = NULL;
    }
    ds4_gpu_tensor_free(g->mtp_raw_cache);
    ds4_gpu_tensor_free(g->mtp_next_hc);
    ds4_gpu_tensor_free(g->mtp_state_hc);
    ds4_gpu_tensor_free(g->mtp_input_hc);
    ds4_gpu_tensor_free(g->mtp_hproj_hc);
    ds4_gpu_tensor_free(g->mtp_hnorm_hc);
    ds4_gpu_tensor_free(g->mtp_eproj_hc);
    ds4_gpu_tensor_free(g->mtp_eproj);
    ds4_gpu_tensor_free(g->mtp_enorm);
    ds4_gpu_tensor_free(g->mtp_embed);
    ds4_gpu_tensor_free(g->spec_logits);
    /* Class H output-head free across all tier slots. */
    for (int t = 0; t < DS4_MAX_GPUS; t++) {
        ds4_gpu_tensor_free(g->output_norm_by_tier[t]);
        g->output_norm_by_tier[t] = NULL;
        ds4_gpu_tensor_free(g->output_embd_by_tier[t]);
        g->output_embd_by_tier[t] = NULL;
        ds4_gpu_tensor_free(g->output_weights_by_tier[t]);
        g->output_weights_by_tier[t] = NULL;
        ds4_gpu_tensor_free(g->output_pre_by_tier[t]);
        g->output_pre_by_tier[t] = NULL;
    }
    /* Class P decode scratch + routed-FFN free across all
     * tier slots. ffn_out is also a Class P field freed here. */
    for (int t = 0; t < DS4_MAX_GPUS; t++) {
        ds4_gpu_tensor_free(g->after_ffn_hc_by_tier[t]);
        ds4_gpu_tensor_free(g->ffn_out_by_tier[t]);
        ds4_gpu_tensor_free(g->routed_out_by_tier[t]);
        ds4_gpu_tensor_free(g->routed_down_by_tier[t]);
        ds4_gpu_tensor_free(g->routed_mid_by_tier[t]);
        ds4_gpu_tensor_free(g->routed_up_by_tier[t]);
        ds4_gpu_tensor_free(g->routed_gate_by_tier[t]);
        ds4_gpu_tensor_free(g->tp_peer_tmp_by_tier[t]);
        ds4_gpu_tensor_free(g->router_weights_by_tier[t]);
        ds4_gpu_tensor_free(g->router_selected_by_tier[t]);
        ds4_gpu_tensor_free(g->router_probs_by_tier[t]);
        ds4_gpu_tensor_free(g->router_logits_by_tier[t]);
        ds4_gpu_tensor_free(g->shared_out_by_tier[t]);
        ds4_gpu_tensor_free(g->shared_mid_by_tier[t]);
        ds4_gpu_tensor_free(g->shared_up_by_tier[t]);
        ds4_gpu_tensor_free(g->shared_gate_by_tier[t]);
        ds4_gpu_tensor_free(g->ffn_norm_by_tier[t]);
        ds4_gpu_tensor_free(g->ffn_cur_by_tier[t]);
        ds4_gpu_tensor_free(g->after_attn_hc_by_tier[t]);
        ds4_gpu_tensor_free(g->attn_out_by_tier[t]);
        ds4_gpu_tensor_free(g->attn_low_by_tier[t]);
        ds4_gpu_tensor_free(g->heads_by_tier[t]);
        ds4_gpu_tensor_free(g->comp_sc_cur_by_tier[t]);
        ds4_gpu_tensor_free(g->comp_kv_cur_by_tier[t]);
        ds4_gpu_tensor_free(g->attn_comp_stage_by_tier[t]);
        ds4_gpu_tensor_free(g->comp_mask_by_tier[t]);
        ds4_gpu_tensor_free(g->comp_selected_by_tier[t]);
        ds4_gpu_tensor_free(g->indexer_scores_by_tier[t]);
        ds4_gpu_tensor_free(g->indexer_weights_by_tier[t]);
        ds4_gpu_tensor_free(g->indexer_q_by_tier[t]);
    }
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        ds4_gpu_tensor_free(g->layer_raw_cache[il]);
    }
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        ds4_gpu_tensor_free(g->layer_raw_cache_tp[il]);
    }
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        ds4_gpu_tensor_free(g->layer_attn_comp_cache[il]);
    }
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        ds4_gpu_tensor_free(g->layer_attn_comp_cache_tp[il]);
    }
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        ds4_gpu_tensor_free(g->layer_attn_state_kv[il]);
    }
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        ds4_gpu_tensor_free(g->layer_attn_state_score[il]);
    }
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        ds4_gpu_tensor_free(g->layer_index_comp_cache[il]);
    }
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        ds4_gpu_tensor_free(g->layer_index_state_kv[il]);
    }
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        ds4_gpu_tensor_free(g->layer_index_state_score[il]);
    }
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        ds4_gpu_tensor_free(g->spec_attn_state_kv[il]);
        ds4_gpu_tensor_free(g->spec_attn_state_score[il]);
        ds4_gpu_tensor_free(g->spec_index_state_kv[il]);
        ds4_gpu_tensor_free(g->spec_index_state_score[il]);
        ds4_gpu_tensor_free(g->spec_prefix1_attn_state_kv[il]);
        ds4_gpu_tensor_free(g->spec_prefix1_attn_state_score[il]);
        ds4_gpu_tensor_free(g->spec_prefix1_index_state_kv[il]);
        ds4_gpu_tensor_free(g->spec_prefix1_index_state_score[il]);
    }
    /* Class P decode-step scratch + decode HC group free across
     * all tier slots. hc_pre / hc_post / hc_comb are VIEWS of hc_split — free
     * them before hc_split so the view struct release happens with the parent
     * still pointer-valid (view free does not touch parent memory). */
    for (int t = 0; t < DS4_MAX_GPUS; t++) {
        ds4_gpu_tensor_free(g->kv_by_tier[t]);
        ds4_gpu_tensor_free(g->kv_raw_by_tier[t]);
        ds4_gpu_tensor_free(g->q_by_tier[t]);
        ds4_gpu_tensor_free(g->qr_norm_by_tier[t]);
        ds4_gpu_tensor_free(g->qr_by_tier[t]);
        ds4_gpu_tensor_free(g->attn_norm_by_tier[t]);
        ds4_gpu_tensor_free(g->attn_cur_by_tier[t]);
        ds4_gpu_tensor_free(g->hc_comb_by_tier[t]);
        ds4_gpu_tensor_free(g->hc_post_by_tier[t]);
        ds4_gpu_tensor_free(g->hc_pre_by_tier[t]);
        ds4_gpu_tensor_free(g->hc_split_by_tier[t]);
        ds4_gpu_tensor_free(g->hc_mix_by_tier[t]);
        ds4_gpu_tensor_free(g->flat_hc_by_tier[t]);
        ds4_gpu_tensor_free(g->cur_hc_by_tier[t]);
    }
    ds4_gpu_tensor_free(g->dspark_position_ids);
    ds4_gpu_tensor_free(g->dspark_stage_output_hc);
    ds4_gpu_tensor_free(g->dspark_stage_input_hc);
    ds4_gpu_tensor_free(g->dspark_target_hc);
    ds4_gpu_tensor_free(g->dspark_draft_hc);
    ds4_gpu_tensor_free(g->dspark_draft_tokens);
    for (uint32_t stage = 0; stage < DS4_DSPARK_MAX_STAGES; stage++) {
        ds4_gpu_tensor_free(g->dspark_raw_cache[stage]);
    }
    ds4_gpu_tensor_free(g->dspark_main_x);
    ds4_gpu_tensor_free(g->dspark_stage0_proj);
    ds4_gpu_tensor_free(g->dspark_stage0_packed);
    ds4_gpu_tensor_free(g->dspark_target_hidden_batch);
    ds4_gpu_tensor_free(g->dspark_target_hidden);
    ds4_gpu_tensor_free(g->dspark_hc_mean_rows);
    ds4_gpu_tensor_free(g->dspark_hc_mean_weights);
    ds4_gpu_tensor_free(g->tp_logits_half);
    free(g->cpu_router_norm);
    memset(g, 0, sizeof(*g));
}

static bool metal_tensor_fill_f32(ds4_gpu_tensor *t, float v, uint64_t n) {
    return ds4_gpu_tensor_fill_f32(t, v, n) != 0;
}

/* =========================================================================
 * Directional Steering.
 * =========================================================================
 *
 * A steering file contains one normalized 4096-wide direction per layer.  When
 * enabled, the Metal graph edits selected block outputs in-place:
 *
 *     y = y - scale * v * dot(v, y)
 *
 * Positive scales remove the represented direction from the activation.
 * Negative scales add it.  This is deliberately explicit and opt-in; with zero
 * scales, the release graph does not allocate the direction tensor and follows
 * the normal inference path.
 */

/* directional_steering_dirs is Class P — replicated per tier.
 * The same host directions buffer is written to every tier slot the engine's
 * placement uses, then the load buffer is freed. Read-only after init, so
 * the per-tier replicas stay byte-identical and never re-sync. */
static bool metal_graph_load_directional_steering(
        ds4_gpu_graph *g,
        const char      *path,
        float            attn_scale,
        float            ffn_scale) {
    if (attn_scale == 0.0f && ffn_scale == 0.0f) return true;

    if (!path || !path[0]) {
        fprintf(stderr, "ds4: directional steering needs --dir-steering-file\n");
        return false;
    }

    const uint64_t n = (uint64_t)DS4_N_LAYER * DS4_N_EMBD;
    float *dirs = xmalloc((size_t)n * sizeof(dirs[0]));
    bool ok = read_f32_binary_file(path, dirs, n);
    if (ok) {
        /* Replicate the directions buffer onto every Class P tier slot that
         * has any other Class P scratch allocated (used_tier marker is the
         * presence of g->cur_hc_by_tier[t]). Single-tier: only slot 0. */
        bool any = false;
        for (int t = 0; ok && t < DS4_MAX_GPUS; t++) {
            if (!g->cur_hc_by_tier[t]) continue;
            g->directional_steering_dirs_by_tier[t] =
                ds4_gpu_tensor_alloc_ptr_on(t, n * sizeof(dirs[0]));
            ok = g->directional_steering_dirs_by_tier[t] != NULL &&
                 ds4_gpu_tensor_write(g->directional_steering_dirs_by_tier[t],
                                      0, dirs, n * sizeof(dirs[0])) != 0;
            if (ok) any = true;
        }
        if (ok && !any) {
            /* No used tiers — graph not allocated yet. This shouldn't happen
             * given the call site ordering, but bail cleanly. */
            ok = false;
        }
    }
    free(dirs);

    if (!ok) {
        fprintf(stderr, "ds4: failed to load directional steering vectors from %s\n", path);
        return false;
    }
    g->directional_steering_attn_scale = attn_scale;
    g->directional_steering_ffn_scale = ffn_scale;
    fprintf(stderr, "ds4: directional steering enabled: %s attn=%g ffn=%g\n",
            path, (double)attn_scale, (double)ffn_scale);
    return true;
}

static bool metal_graph_directional_steering_attn_enabled(const ds4_gpu_graph *g) {
    return g && metal_graph_directional_steering_dirs(g) &&
           g->directional_steering_attn_scale != 0.0f;
}

static bool metal_graph_directional_steering_ffn_enabled(const ds4_gpu_graph *g) {
    return g && metal_graph_directional_steering_dirs(g) &&
           g->directional_steering_ffn_scale != 0.0f;
}

static bool metal_graph_apply_directional_steering(
        ds4_gpu_graph  *g,
        ds4_gpu_tensor *x,
        uint32_t          il,
        uint32_t          rows,
        float             scale) {
    if (!g || !metal_graph_directional_steering_dirs(g) || scale == 0.0f) return true;
    return ds4_gpu_directional_steering_project_tensor(x,
                                            metal_graph_directional_steering_dirs(g),
                                            il,
                                            DS4_N_EMBD,
                                            rows,
                                            scale) != 0;
}

static bool metal_graph_apply_directional_steering_attn(
        ds4_gpu_graph  *g,
        ds4_gpu_tensor *x,
        uint32_t          il,
        uint32_t          rows) {
    return metal_graph_apply_directional_steering(g, x, il, rows, g ? g->directional_steering_attn_scale : 0.0f);
}

static bool metal_graph_apply_directional_steering_ffn(
        ds4_gpu_graph  *g,
        ds4_gpu_tensor *x,
        uint32_t          il,
        uint32_t          rows) {
    return metal_graph_apply_directional_steering(g, x, il, rows, g ? g->directional_steering_ffn_scale : 0.0f);
}

static bool metal_graph_configure_dspark_capture(
        ds4_gpu_graph            *g,
        const ds4_dspark_weights *dw) {
    if (!g || !dw || dw->target_layer_count == 0) return true;
    if (dw->target_layer_count > DS4_DSPARK_MAX_TARGET_LAYERS ||
        DS4_N_HC == 0 ||
        DS4_N_HC > DS4_MAX_HC) {
        return false;
    }

    g->dspark_hc_mean_weights =
        ds4_gpu_tensor_alloc((uint64_t)DS4_N_HC * sizeof(float));
    g->dspark_hc_mean_rows =
        ds4_gpu_tensor_alloc((uint64_t)g->prefill_cap *
                             DS4_N_HC * sizeof(float));
    g->dspark_target_hidden =
        ds4_gpu_tensor_alloc((uint64_t)dw->target_layer_count *
                             DS4_N_EMBD * sizeof(float));
    g->dspark_target_hidden_batch =
        ds4_gpu_tensor_alloc((uint64_t)dw->target_layer_count *
                             g->prefill_cap *
                             DS4_N_EMBD * sizeof(float));
    if (dw->block_size != 0 && dw->block_size <= DS4_DSPARK_MAX_BLOCK_SIZE) {
        g->dspark_stage0_packed =
            ds4_gpu_tensor_alloc(((uint64_t)dw->block_size + 1u) *
                                 dw->target_layer_count *
                                 DS4_N_EMBD * sizeof(float));
    }
    g->dspark_stage0_proj =
        ds4_gpu_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
    g->dspark_main_x =
        ds4_gpu_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
    if (!g->dspark_hc_mean_weights || !g->dspark_hc_mean_rows ||
        !g->dspark_target_hidden || !g->dspark_target_hidden_batch ||
        !g->dspark_stage0_proj || !g->dspark_main_x) {
        return false;
    }
    if (dw->block_size != 0 && dw->block_size <= DS4_DSPARK_MAX_BLOCK_SIZE) {
        const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
        g->dspark_draft_tokens =
            ds4_gpu_tensor_alloc((uint64_t)dw->block_size * sizeof(int32_t));
        g->dspark_draft_hc =
            ds4_gpu_tensor_alloc((uint64_t)dw->block_size * hc_dim * sizeof(float));
        g->dspark_target_hc =
            ds4_gpu_tensor_alloc(hc_dim * sizeof(float));
        g->dspark_stage_input_hc =
            ds4_gpu_tensor_alloc((uint64_t)(dw->block_size + 1u) *
                                 hc_dim * sizeof(float));
        g->dspark_stage_output_hc =
            ds4_gpu_tensor_alloc((uint64_t)dw->block_size *
                                 hc_dim * sizeof(float));
        g->dspark_position_ids =
            ds4_gpu_tensor_alloc((uint64_t)(dw->block_size + 1u) *
                                 sizeof(int32_t));
        if (!g->dspark_draft_tokens || !g->dspark_draft_hc ||
            !g->dspark_target_hc || !g->dspark_stage_input_hc ||
            !g->dspark_stage_output_hc || !g->dspark_position_ids) {
            return false;
        }
        if (dw->n_stages != 0 && g->raw_cap != 0) {
            for (uint32_t stage = 0; stage < dw->n_stages; stage++) {
                g->dspark_raw_cache[stage] =
                    ds4_gpu_tensor_alloc((uint64_t)g->raw_cap *
                                         DS4_N_HEAD_DIM * sizeof(float));
                if (!g->dspark_raw_cache[stage]) return false;
            }
            g->dspark_cache_cap = g->raw_cap;
            g->dspark_cache_start = 0;
            g->dspark_cache_token_start = 0;
            g->dspark_cache_len = 0;
        }
        g->dspark_block_size = dw->block_size;
    }

    float mean[DS4_MAX_HC] = {0};
    const float inv_hc = 1.0f / (float)DS4_N_HC;
    for (uint32_t i = 0; i < DS4_N_HC; i++) mean[i] = inv_hc;
    if (ds4_gpu_tensor_write(g->dspark_hc_mean_weights,
                             0,
                             mean,
                             (uint64_t)DS4_N_HC * sizeof(mean[0])) == 0) {
        return false;
    }
    const uint64_t mean_rows_count = (uint64_t)g->prefill_cap * DS4_N_HC;
    if (mean_rows_count == 0 || mean_rows_count > (uint64_t)SIZE_MAX / sizeof(float)) {
        return false;
    }
    float *mean_rows = xmalloc((size_t)mean_rows_count * sizeof(mean_rows[0]));
    for (uint64_t i = 0; i < mean_rows_count; i++) mean_rows[i] = inv_hc;
    const bool mean_rows_ok =
        ds4_gpu_tensor_write(g->dspark_hc_mean_rows,
                             0,
                             mean_rows,
                             mean_rows_count * sizeof(mean_rows[0])) != 0;
    free(mean_rows);
    if (!mean_rows_ok) return false;

    g->dspark_target_layer_count = dw->target_layer_count;
    memcpy(g->dspark_target_layers,
           dw->target_layers,
           (size_t)dw->target_layer_count * sizeof(g->dspark_target_layers[0]));
    g->dspark_capture_mask = 0;
    g->dspark_capture_checkpoint_len = 0;
    g->dspark_capture_batch_mask = 0;
    g->dspark_capture_batch_start = 0;
    g->dspark_capture_batch_tokens = 0;
    g->dspark_capture_valid = false;
    g->dspark_capture_batch_valid = false;
    g->dspark_capture_enabled = true;
    return true;
}

static uint64_t metal_graph_kv_cache_bytes_for_context(uint32_t ctx_size, uint32_t raw_cap) {
    uint64_t bytes = (uint64_t)DS4_N_LAYER *
                     raw_cap *
                     DS4_N_HEAD_DIM *
                     sizeof(float);

    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (ratio == 0) continue;
        const uint64_t comp_cap = (uint64_t)(ctx_size / ratio + 2u);
        bytes += comp_cap * DS4_N_HEAD_DIM *
                 (DS4_GPU_ATTN_COMP_CACHE_F16 ? sizeof(uint16_t) : sizeof(float));
        if (ratio == 4) {
            bytes += comp_cap * DS4_N_INDEXER_HEAD_DIM * sizeof(float);
        }
    }
    return bytes;
}

static uint64_t metal_graph_context_bytes_for_kv_policy(
        uint32_t  ctx_size,
        uint32_t  raw_cap,
        uint32_t  prefill_cap,
        uint64_t *kv_cache_bytes_out) {
    uint32_t min_ratio = UINT32_MAX;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (ratio != 0 && ratio < min_ratio) min_ratio = ratio;
    }
    if (min_ratio == UINT32_MAX) min_ratio = ctx_size ? ctx_size : 1u;
    uint64_t comp_cap = (uint64_t)(ctx_size / min_ratio + 2u);
    if (comp_cap < 2u) comp_cap = 2u;
    const uint64_t kv_cache_bytes = metal_graph_kv_cache_bytes_for_context(ctx_size, raw_cap);
    if (kv_cache_bytes_out) *kv_cache_bytes_out = kv_cache_bytes;
    uint64_t bytes = kv_cache_bytes +
                     2ull * comp_cap * prefill_cap * sizeof(float);
    if (DS4_GPU_ATTN_COMP_CACHE_F16) {
        uint64_t attn_stage_cap = (uint64_t)(prefill_cap / min_ratio + 2u);
        if (attn_stage_cap < 2u) attn_stage_cap = 2u;
        bytes += attn_stage_cap * DS4_N_HEAD_DIM * sizeof(float);
    }
    return bytes;
}

static ds4_gpu_tensor *metal_graph_alloc_kv_cache_tensor_on(
        bool managed,
        int tier,
        uint64_t bytes) {
    if (g_n_gpus <= 1) {
        return managed ? ds4_gpu_tensor_alloc_managed(bytes)
                       : ds4_gpu_tensor_alloc(bytes);
    }
    return managed ? ds4_gpu_tensor_alloc_managed_on(tier, bytes)
                   : ds4_gpu_tensor_alloc_ptr_on(tier, bytes);
}

static ds4_gpu_tensor *metal_graph_alloc_kv_cache_tensor(bool managed, uint64_t bytes) {
    return metal_graph_alloc_kv_cache_tensor_on(managed, 0, bytes);
}

/* =========================================================================
 * Metal Diagnostic Dump Hooks.
 * =========================================================================
 *
 * The release path calls these after important stages, but they are no-ops
 * unless DS4_METAL_GRAPH_DUMP_PREFIX or DS4_ROCM_GRAPH_DUMP_PREFIX is set.
 * Dumping synchronizes and restarts the command batch, so it is intentionally
 * isolated here.
 */

typedef struct {
    int init;
    const char *prefix;
    const char *name;
    int layer_set;
    uint32_t layer;
    int pos_set;
    uint32_t pos;
} metal_graph_debug_config;

static const metal_graph_debug_config *metal_graph_debug_get_config(void) {
    static metal_graph_debug_config cfg;
    if (!cfg.init) {
        cfg.init = 1;
        cfg.prefix = glm_graph_env_value("DS4_ROCM_GRAPH_DUMP_PREFIX",
                                         "DS4_METAL_GRAPH_DUMP_PREFIX");
        if (cfg.prefix && !cfg.prefix[0]) cfg.prefix = NULL;
        cfg.name = glm_graph_env_value("DS4_ROCM_GRAPH_DUMP_NAME",
                                       "DS4_METAL_GRAPH_DUMP_NAME");
        if (cfg.name && !cfg.name[0]) cfg.name = NULL;

        const char *layer_env = glm_graph_env_value("DS4_ROCM_GRAPH_DUMP_LAYER",
                                                    "DS4_METAL_GRAPH_DUMP_LAYER");
        if (layer_env && layer_env[0] && strcmp(layer_env, "all") != 0) {
            cfg.layer_set = 1;
            cfg.layer = (uint32_t)strtoul(layer_env, NULL, 10);
        }

        const char *pos_env = glm_graph_env_value("DS4_ROCM_GRAPH_DUMP_POS",
                                                  "DS4_METAL_GRAPH_DUMP_POS");
        if (pos_env && pos_env[0]) {
            cfg.pos_set = 1;
            cfg.pos = (uint32_t)strtoul(pos_env, NULL, 10);
        }
    }
    return &cfg;
}

static const char *metal_graph_debug_prefix_for(const char *name, uint32_t il, uint32_t pos) {
    const metal_graph_debug_config *cfg = metal_graph_debug_get_config();
    if (!cfg->prefix) return NULL;
    if (cfg->name && strstr(cfg->name, name) == NULL) return NULL;
    if (cfg->layer_set && cfg->layer != il) return NULL;
    if (cfg->pos_set && cfg->pos != pos) return NULL;
    return cfg->prefix;
}

static bool metal_graph_debug_wants(const char *name, uint32_t il, uint32_t pos) {
    return metal_graph_debug_prefix_for(name, il, pos) != NULL;
}

static void metal_graph_debug_dump_tensor(
        const char       *name,
        ds4_gpu_tensor *t,
        uint64_t          n_f32,
        uint32_t          il,
        uint32_t          pos) {
    const char *prefix = metal_graph_debug_prefix_for(name, il, pos);
    if (glm_graph_env_present("DS4_ROCM_GRAPH_DUMP_TRACE",
                              "DS4_METAL_GRAPH_DUMP_TRACE"))
        fprintf(stderr, "ds4: dump? name=%s il=%u pos=%u t=%p n=%llu wants=%d\n",
                name, il, pos, (void *)t, (unsigned long long)n_f32,
                metal_graph_debug_wants(name, il, pos));
    if (!t || n_f32 == 0 || !metal_graph_debug_wants(name, il, pos)) return;

    if (ds4_gpu_synchronize() == 0) {
        fprintf(stderr, "ds4: failed to synchronize before dumping %s layer %u pos %u\n", name, il, pos);
        return;
    }

    float *buf = xmalloc((size_t)n_f32 * sizeof(buf[0]));
    if (ds4_gpu_tensor_read(t, 0, buf, n_f32 * sizeof(buf[0])) != 0) {
        char path[1024];
        snprintf(path, sizeof(path), "%s_%s-%u_pos%u.bin", prefix, name, il, pos);
        if (write_f32_binary_file(path, buf, n_f32)) {
            fprintf(stderr, "ds4: dumped %s layer %u pos %u to %s\n", name, il, pos, path);
        }
    }
    free(buf);

    if (ds4_gpu_begin_commands() == 0) {
        fprintf(stderr, "ds4: failed to resume Metal command batch after dumping %s layer %u pos %u\n", name, il, pos);
    }
}

static void metal_graph_debug_dump_f16_tensor(
        const char       *name,
        ds4_gpu_tensor *t,
        uint64_t          n_f16,
        uint32_t          il,
        uint32_t          pos) {
    const char *prefix = glm_graph_env_value("DS4_ROCM_GRAPH_DUMP_PREFIX",
                                             "DS4_METAL_GRAPH_DUMP_PREFIX");
    if (!t || n_f16 == 0 || !metal_graph_debug_wants(name, il, pos)) return;

    if (ds4_gpu_synchronize() == 0) {
        fprintf(stderr, "ds4: failed to synchronize before dumping %s layer %u pos %u\n", name, il, pos);
        return;
    }

    uint16_t *hbuf = xmalloc((size_t)n_f16 * sizeof(hbuf[0]));
    float *fbuf = xmalloc((size_t)n_f16 * sizeof(fbuf[0]));
    if (ds4_gpu_tensor_read(t, 0, hbuf, n_f16 * sizeof(hbuf[0])) != 0) {
        for (uint64_t i = 0; i < n_f16; i++) fbuf[i] = f16_to_f32(hbuf[i]);
        char path[1024];
        snprintf(path, sizeof(path), "%s_%s-%u_pos%u.bin", prefix, name, il, pos);
        if (write_f32_binary_file(path, fbuf, n_f16)) {
            fprintf(stderr, "ds4: dumped %s layer %u pos %u to %s\n", name, il, pos, path);
        }
    }
    free(fbuf);
    free(hbuf);

    if (ds4_gpu_begin_commands() == 0) {
        fprintf(stderr, "ds4: failed to resume Metal command batch after dumping %s layer %u pos %u\n", name, il, pos);
    }
}

static void metal_graph_debug_dump_i32_tensor(
        const char       *name,
        ds4_gpu_tensor *t,
        uint64_t          n_i32,
        uint32_t          il,
        uint32_t          pos) {
    if (!t || n_i32 == 0) return;
    const char *prefix = metal_graph_debug_prefix_for(name, il, pos);
    if (!prefix) return;

    if (ds4_gpu_synchronize() == 0) {
        fprintf(stderr, "ds4: failed to synchronize before dumping %s layer %u pos %u\n", name, il, pos);
        return;
    }

    int32_t *buf = xmalloc((size_t)n_i32 * sizeof(buf[0]));
    if (ds4_gpu_tensor_read(t, 0, buf, n_i32 * sizeof(buf[0])) != 0) {
        char path[1024];
        snprintf(path, sizeof(path), "%s_%s-%u_pos%u.i32", prefix, name, il, pos);
        FILE *fp = fopen(path, "wb");
        if (fp) {
            if (fwrite(buf, sizeof(buf[0]), (size_t)n_i32, fp) == (size_t)n_i32) {
                fprintf(stderr, "ds4: dumped %s layer %u pos %u to %s\n", name, il, pos, path);
            }
            fclose(fp);
        }
    }
    free(buf);

    if (ds4_gpu_begin_commands() == 0) {
        fprintf(stderr, "ds4: failed to resume Metal command batch after dumping %s layer %u pos %u\n", name, il, pos);
    }
}

static bool metal_graph_needs_ffn_out(const ds4_gpu_graph *g, uint32_t il, uint32_t pos) {
    return metal_graph_directional_steering_ffn_enabled(g) ||
           g->materialize_ffn_out ||
           metal_graph_debug_wants("ffn_out", il, pos);
}

/* tier-aware lazy allocator. The Class P ffn_out scratch is
 * created on demand the first time a layer that materializes ffn_out runs
 * on a tier; subsequent visits to the same tier reuse the existing slot.
 * Single-tier paths: active_tier == 0 always, behavior unchanged. */
static bool metal_graph_ensure_ffn_out(ds4_gpu_graph *g) {
    const int t = g->active_tier;
    if (!g->ffn_out_by_tier[t]) {
        g->ffn_out_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(
                t, (uint64_t)DS4_N_EMBD * sizeof(float));
    }
    return g->ffn_out_by_tier[t] != NULL;
}

static bool metal_graph_ensure_batch_ffn_out_on(ds4_gpu_graph *g, int t) {
    if (t < 0 || t >= DS4_MAX_GPUS) return false;
    if (!g->batch_ffn_out_by_tier[t]) {
        g->batch_ffn_out_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(
                t, (uint64_t)g->prefill_cap * DS4_N_EMBD * sizeof(float));
    }
    return g->batch_ffn_out_by_tier[t] != NULL;
}

static bool metal_graph_ensure_batch_ffn_out(ds4_gpu_graph *g) {
    return metal_graph_ensure_batch_ffn_out_on(g, g->active_tier);
}

static bool metal_graph_tp_env_flag(const char *name, bool dflt) {
    const char *env = getenv(name);
    if (!env || !env[0]) return dflt;
    return strcmp(env, "0") != 0;
}

static bool metal_graph_cuda_tp_attn_requested(void) {
#if defined(__APPLE__)
    return false;
#else
    return metal_graph_tp_env_flag("DS4_CUDA_TP_ATTN", true);
#endif
}

static bool metal_graph_cuda_tp_attn_peer_read_requested(void) {
#if defined(__APPLE__)
    return false;
#else
    return metal_graph_tp_env_flag("DS4_CUDA_TP_ATTN_PEER_READ", true);
#endif
}

static bool metal_graph_cuda_tp_attn_heads_requested(void) {
#if defined(__APPLE__)
    return false;
#else
    return metal_graph_tp_env_flag("DS4_CUDA_TP_ATTN_HEADS", false);
#endif
}

static bool metal_graph_cuda_tp_attn_cache_dup_requested(void) {
#if defined(__APPLE__)
    return false;
#else
    return metal_graph_tp_env_flag("DS4_CUDA_TP_ATTN_CACHE_DUP", false);
#endif
}

static bool metal_graph_cuda_tp_moe_requested(void) {
#if defined(__APPLE__)
    return false;
#else
    return metal_graph_tp_env_flag("DS4_CUDA_TP_MOE", true);
#endif
}

static bool metal_graph_cuda_tp_ep_pack_exact_requested(void) {
#if defined(__APPLE__)
    return false;
#else
    return metal_graph_tp_env_flag("DS4_CUDA_TP_EP_PACK_EXACT", true);
#endif
}

static bool metal_graph_cuda_tp_ep_direct_return_requested(void) {
#if defined(__APPLE__)
    return false;
#else
    return metal_graph_tp_env_flag("DS4_CUDA_TP_EP_DIRECT_RETURN", true);
#endif
}

static bool metal_graph_cuda_tp_ep_delay_reduce_requested(void) {
#if defined(__APPLE__)
    return false;
#else
    return metal_graph_tp_env_flag("DS4_CUDA_TP_EP_DELAY_REDUCE", true);
#endif
}

static bool metal_graph_cuda_tp_ep_fused_hc_reduce_requested(void) {
#if defined(__APPLE__)
    return false;
#else
    return metal_graph_tp_env_flag("DS4_CUDA_TP_EP_FUSED_HC_REDUCE", true);
#endif
}

static DS4_MAYBE_UNUSED bool metal_graph_cuda_tp_ep_fused_shared_mid_requested(void) {
#if defined(__APPLE__)
    return false;
#else
    return metal_graph_tp_env_flag("DS4_CUDA_TP_EP_FUSED_SHARED_MID", true);
#endif
}

static DS4_MAYBE_UNUSED bool metal_graph_cuda_tp_ep_balanced_shared_mid_requested(void) {
#if defined(__APPLE__)
    return false;
#else
    return metal_graph_tp_env_flag(
            "DS4_CUDA_TP_EP_BALANCED_SHARED_MID", true);
#endif
}

static DS4_MAYBE_UNUSED bool metal_graph_cuda_tp_ep_dual_prequant_requested(void) {
#if defined(__APPLE__)
    return false;
#else
    return metal_graph_tp_env_flag(
            "DS4_CUDA_TP_EP_DUAL_PREQUANT", true);
#endif
}

static bool metal_graph_cuda_tp_moe_delay_reduce_requested(void) {
#if defined(__APPLE__)
    return false;
#else
    return metal_graph_tp_env_flag("DS4_CUDA_TP_MOE_DELAY_REDUCE", true);
#endif
}

static bool metal_graph_cuda_tp_moe_pack_handoff_requested(void) {
#if defined(__APPLE__)
    return false;
#else
    return metal_graph_tp_env_flag("DS4_CUDA_TP_MOE_PACK", false);
#endif
}

static bool metal_graph_cuda_tp_moe_copy3_handoff_requested(void) {
#if defined(__APPLE__)
    return false;
#else
    return metal_graph_tp_env_flag("DS4_CUDA_TP_MOE_COPY3", false);
#endif
}

static bool metal_graph_cuda_tp_moe_peer_read_requested(void) {
#if defined(__APPLE__)
    return false;
#else
    return metal_graph_tp_env_flag("DS4_CUDA_TP_MOE_PEER_READ", false);
#endif
}

static bool metal_graph_cuda_tp_moe_peer_router_requested(void) {
#if defined(__APPLE__)
    return false;
#else
    return metal_graph_tp_env_flag("DS4_CUDA_TP_MOE_PEER_ROUTER", false);
#endif
}

static bool metal_graph_cuda_tp_shared_requested(void) {
#if defined(__APPLE__)
    return false;
#else
    return metal_graph_tp_env_flag("DS4_CUDA_TP_SHARED", false);
#endif
}

static bool metal_graph_cuda_tp_shared_fold_requested(void) {
#if defined(__APPLE__)
    return false;
#else
    return metal_graph_tp_env_flag("DS4_CUDA_TP_SHARED_FOLD", true);
#endif
}

static bool metal_graph_cuda_tp_q_requested(void) {
#if defined(__APPLE__)
    return false;
#else
    return metal_graph_tp_env_flag("DS4_CUDA_TP_Q", false);
#endif
}

static bool metal_graph_cuda_tp_output_requested(void) {
#if defined(__APPLE__)
    return false;
#else
    return metal_graph_tp_env_flag("DS4_CUDA_TP_OUTPUT", true);
#endif
}

static bool metal_graph_cuda_greedy_split_top1_requested(void) {
#if defined(__APPLE__)
    return false;
#else
    const char *no = getenv("DS4_CUDA_NO_GREEDY_SPLIT_TOP1");
    if (no && no[0] && strcmp(no, "0") != 0) return false;
    return metal_graph_tp_env_flag("DS4_CUDA_GREEDY_SPLIT_TOP1", false);
#endif
}

static bool metal_graph_cuda_output_fused_top1_requested(void) {
#if defined(__APPLE__)
    return false;
#else
    return metal_graph_tp_env_flag("DS4_CUDA_OUTPUT_FUSED_TOP1", false);
#endif
}

static bool metal_graph_cuda_verify_decode2_split_top1_requested(void) {
#if defined(__APPLE__)
    return false;
#else
    const char *no = getenv("DS4_CUDA_NO_VERIFY_DECODE2_SPLIT_TOP1");
    if (no && no[0] && strcmp(no, "0") != 0) return false;
    return metal_graph_tp_env_flag("DS4_CUDA_VERIFY_DECODE2_SPLIT_TOP1", false);
#endif
}

static bool metal_graph_cuda_greedy_splitkv_requested(void) {
#if defined(__APPLE__)
    return false;
#else
    const char *no = getenv("DS4_CUDA_NO_GREEDY_SPLITKV");
    if (no && no[0] && strcmp(no, "0") != 0) return false;
    return metal_graph_tp_env_flag("DS4_CUDA_GREEDY_SPLITKV", false);
#endif
}

static bool metal_graph_cuda_greedy_vec4_requested(void) {
#if defined(__APPLE__)
    return false;
#else
    const char *no = getenv("DS4_CUDA_NO_GREEDY_VEC4");
    if (no && no[0] && strcmp(no, "0") != 0) return false;
    return metal_graph_tp_env_flag("DS4_CUDA_GREEDY_VEC4", false);
#endif
}

static bool metal_graph_cuda_splitkv_spec_requested(void) {
#if defined(__APPLE__)
    return false;
#else
    const char *no = getenv("DS4_CUDA_NO_SPLITKV_SPEC");
    if (no && no[0] && strcmp(no, "0") != 0) return false;
    return metal_graph_tp_env_flag("DS4_CUDA_SPLITKV_SPEC", false);
#endif
}

static bool metal_graph_cuda_splitkv_spec_toponly_row0_requested(void) {
#if defined(__APPLE__)
    return false;
#else
    const char *no = getenv("DS4_CUDA_NO_SPLITKV_SPEC_TOPONLY_ROW0");
    if (no && no[0] && strcmp(no, "0") != 0) return false;
    return metal_graph_tp_env_flag("DS4_CUDA_SPLITKV_SPEC_TOPONLY_ROW0", false);
#endif
}

static bool metal_graph_cuda_splitkv_spec_batch_verify_requested(void) {
#if defined(__APPLE__)
    return false;
#else
    const char *no = getenv("DS4_CUDA_NO_SPLITKV_SPEC_BATCH_VERIFY");
    if (no && no[0] && strcmp(no, "0") != 0) return false;
    return metal_graph_tp_env_flag("DS4_CUDA_SPLITKV_SPEC_BATCH_VERIFY", false);
#endif
}

static float metal_graph_cuda_greedy_vec4_margin_threshold(void) {
#if defined(__APPLE__)
    return 0.0f;
#else
    const char *env = getenv("DS4_CUDA_GREEDY_VEC4_MARGIN");
    if (env && env[0]) {
        char *end = NULL;
        double v = strtod(env, &end);
        while (end && isspace((unsigned char)*end)) end++;
        if (end != env && end && *end == '\0' && isfinite(v) && v >= 0.0) {
            return (float)v;
        }
        fprintf(stderr,
                "ds4: invalid DS4_CUDA_GREEDY_VEC4_MARGIN=%s; using 0.25\n",
                env);
    }
    return 0.25f;
#endif
}

static bool metal_graph_cuda_greedy_vec4_fallback_requested(void) {
#if defined(__APPLE__)
    return false;
#else
    const char *no = getenv("DS4_CUDA_NO_GREEDY_VEC4_FALLBACK");
    if (no && no[0] && strcmp(no, "0") != 0) return false;
    return metal_graph_cuda_greedy_vec4_margin_threshold() > 0.0f;
#endif
}

static float metal_graph_cuda_greedy_splitkv_margin_threshold(void) {
#if defined(__APPLE__)
    return 0.0f;
#else
    const char *env = getenv("DS4_CUDA_GREEDY_SPLITKV_MARGIN");
    if (env && env[0]) {
        char *end = NULL;
        double v = strtod(env, &end);
        while (end && isspace((unsigned char)*end)) end++;
        if (end != env && end && *end == '\0' && isfinite(v) && v >= 0.0) {
            return (float)v;
        }
        fprintf(stderr,
                "ds4: invalid DS4_CUDA_GREEDY_SPLITKV_MARGIN=%s; using 0.25\n",
                env);
    }
    return 0.25f;
#endif
}

static bool metal_graph_cuda_greedy_splitkv_fallback_requested(void) {
#if defined(__APPLE__)
    return false;
#else
    const char *no = getenv("DS4_CUDA_NO_GREEDY_SPLITKV_FALLBACK");
    if (no && no[0] && strcmp(no, "0") != 0) return false;
    return metal_graph_cuda_greedy_splitkv_margin_threshold() > 0.0f;
#endif
}

static bool metal_graph_cuda_greedy_splitkv_top2_requested(void) {
#if defined(__APPLE__)
    return false;
#else
    const char *no = getenv("DS4_CUDA_NO_GREEDY_SPLITKV_TOP2");
    if (no && no[0] && strcmp(no, "0") != 0) return false;
    return metal_graph_tp_env_flag("DS4_CUDA_GREEDY_SPLITKV_TOP2", true);
#endif
}

static bool metal_graph_cuda_greedy_splitkv_trust_replay_requested(void) {
#if defined(__APPLE__)
    return false;
#else
    return metal_graph_tp_env_flag("DS4_CUDA_GREEDY_SPLITKV_TRUST_REPLAY", false);
#endif
}

static bool metal_graph_cuda_greedy_splitkv_pair_replay_requested(void) {
#if defined(__APPLE__)
    return false;
#else
    const char *no = getenv("DS4_CUDA_NO_GREEDY_SPLITKV_PAIR_REPLAY");
    if (no && no[0] && strcmp(no, "0") != 0) return false;
    return metal_graph_tp_env_flag("DS4_CUDA_GREEDY_SPLITKV_PAIR_REPLAY", false);
#endif
}

static DS4_MAYBE_UNUSED uint32_t metal_graph_cuda_greedy_max_segment(const char *name) {
    const char *env = getenv(name);
    if (env && env[0]) {
        char *end = NULL;
        unsigned long v = strtoul(env, &end, 10);
        while (end && isspace((unsigned char)*end)) end++;
        if (end != env && end && *end == '\0' && v <= INT32_MAX) {
            return (uint32_t)v;
        }
        fprintf(stderr,
                "ds4: invalid %s=%s; expected 0..%d, using disabled\n",
                name,
                env,
                INT32_MAX);
    }
    return 0;
}

static uint32_t metal_graph_cuda_greedy_splitkv_max_segment(void) {
#if defined(__APPLE__)
    return 0;
#else
    return metal_graph_cuda_greedy_max_segment("DS4_CUDA_GREEDY_SPLITKV_MAX_SEGMENT");
#endif
}

static uint32_t metal_graph_cuda_greedy_vec4_max_segment(void) {
#if defined(__APPLE__)
    return 0;
#else
    return metal_graph_cuda_greedy_max_segment("DS4_CUDA_GREEDY_VEC4_MAX_SEGMENT");
#endif
}

static uint32_t metal_graph_cuda_greedy_splitkv_min_score(void) {
    const char *env = getenv("DS4_CUDA_SPLITKV_MIN_SCORE");
    if (env && env[0]) {
        char *end = NULL;
        unsigned long v = strtoul(env, &end, 10);
        while (end && isspace((unsigned char)*end)) end++;
        if (end != env && end && *end == '\0' && v <= UINT32_MAX) {
            return (uint32_t)v;
        }
    }
    return metal_graph_tp_env_flag("DS4_CUDA_SPLITKV_DECODE", false) ? 0u : 512u;
}

static bool metal_graph_cuda_q_norm_rope_fuse_requested(void) {
#if defined(__APPLE__)
    return false;
#else
    return metal_graph_tp_env_flag("DS4_CUDA_Q_NORM_ROPE_FUSE", true);
#endif
}

static bool metal_graph_cuda_qkv_kv_rope_fuse_requested(void) {
#if defined(__APPLE__)
    return false;
#else
    const char *no = getenv("DS4_CUDA_NO_QKV_KV_ROPE_FUSE");
    if (no && no[0] && strcmp(no, "0") != 0) return false;
    if (getenv("DS4_CUDA_DISABLE_QKV_RMS_FUSED") != NULL) return false;
    return metal_graph_tp_env_flag("DS4_CUDA_QKV_KV_ROPE_FUSE", true);
#endif
}

static bool metal_graph_cuda_tp_prefill_ffn_requested(void) {
#if defined(__APPLE__)
    return false;
#else
    return metal_graph_tp_env_flag("DS4_CUDA_TP_PREFILL_FFN", true);
#endif
}

static bool metal_graph_cuda_tp_prefill_attn_output_requested(void) {
#if defined(__APPLE__)
    return false;
#else
    return metal_graph_tp_env_flag("DS4_CUDA_TP_PREFILL_ATTN_OUTPUT", true);
#endif
}

static bool metal_graph_cuda_prefill_pipeline_requested(const ds4_gpu_graph *g) {
#if defined(__APPLE__)
    (void)g;
    return false;
#else
    const char *env = getenv("DS4_CUDA_PREFILL_PIPELINE");
    if (env && env[0]) return strcmp(env, "0") != 0;
    return g && g->cuda_tp_decode;
#endif
}

static bool metal_graph_cuda_prefill_pipeline_q8_cache_requested(void) {
#if defined(__APPLE__)
    return false;
#else
    return metal_graph_tp_env_flag("DS4_CUDA_PREFILL_PIPELINE_Q8_CACHE", false);
#endif
}

static uint32_t metal_graph_cuda_prefill_pipeline_microbatch(void) {
    const char *env = getenv("DS4_CUDA_PREFILL_PIPELINE_MB");
    if (env && env[0]) {
        char *end = NULL;
        unsigned long v = strtoul(env, &end, 10);
        if (end != env && v > 0 && v <= UINT32_MAX) return (uint32_t)v;
    }
    return 512;
}

static int metal_graph_cuda_tp_partner_tier(int tier) {
    if (g_n_gpus < 2 || (g_n_gpus & 1) != 0) return -1;
    const int half = g_n_gpus / 2;
    if (tier < 0 || tier >= half) return -1;
    return tier + half;
}

static uint32_t metal_graph_cuda_tp_output_tiers(
        const ds4_gpu_graph *g,
        int                  tiers[DS4_MAX_GPUS]) {
    if (!g) return 0;
    return metal_graph_cuda_tp_output_tiers_for_head(g->head_tier,
                                                    g->cuda_tp_output,
                                                    g_n_gpus,
                                                    tiers);
}

static uint64_t metal_graph_q8_0_row_bytes(uint64_t in_dim) {
    return ((in_dim + 31u) / 32u) * 34u;
}

/* =========================================================================
 * Metal Release Graph Allocation.
 * ========================================================================= */

/* Allocate the Metal graph state for a chosen raw-cache capacity.  The model
 * weights are not copied here; tensors reference the mapped GGUF.
 *
 * tier-aware per-layer allocation.
 *   placement: when non-NULL, an array of DS4_N_LAYER + 2 logical tiers
 *     (embedding, per-layer..., head). The per-layer KV / state allocations
 *     in this function use placement[il + 1] as the home tier for each
 *     layer il. When NULL (single-tier callers, diagnostic paths), all
 *     per-layer allocations land on tier 0 — byte-equivalent to legacy.
 *
 * Single-tier (g_n_gpus <= 1) is byte-equivalent regardless of placement,
 * because metal_graph_alloc_kv_cache_tensor_on short-circuits to the
 * legacy 1-arg helpers when g_n_gpus <= 1. */
static bool metal_graph_alloc_raw_cap(
        ds4_gpu_graph *g,
        const ds4_weights     *weights,
        const ds4_layer_weights *layer,
        uint32_t                raw_cap,
        uint32_t                ctx_size,
        uint32_t                prefill_cap,
        bool                    enable_mtp,
        const int              *placement,
        bool                    cuda_tensor_parallel,
        const ds4_gpu_graph    *shared_prefill_workspace) {
    const int saved_dspark_exec_tier = g->dspark_exec_tier;
    memset(g, 0, sizeof(*g));
    g->dspark_exec_tier = saved_dspark_exec_tier;
    g->owns_prefill_workspace = shared_prefill_workspace == NULL;
    g->cpu_router_norm = xmalloc((size_t)DS4_N_EMBD * sizeof(g->cpu_router_norm[0]));
    g->active_tier = placement ? -1 : 0;
    /* cache placement on the graph so the dispatch loops can
     * walk it without threading the engine pointer through every
     * kernel-dispatch wrapper. NULL in single-tier callers (placement
     * was already NULL on entry). */
    g->placement = placement;
    g->cuda_tp_decode = placement && cuda_tensor_parallel;
    g->cuda_tp_attn = g->cuda_tp_decode && metal_graph_cuda_tp_attn_requested();
    g->cuda_tp_attn_peer_read = metal_graph_cuda_tp_attn_peer_read_requested();
    g->cuda_tp_attn_heads = g->cuda_tp_decode && metal_graph_cuda_tp_attn_heads_requested();
    g->cuda_tp_attn_cache_dup = g->cuda_tp_attn_heads &&
                                metal_graph_cuda_tp_attn_cache_dup_requested();
    g->cuda_tp_moe = g->cuda_tp_decode && metal_graph_cuda_tp_moe_requested();
    g->cuda_tp_ep = g->cuda_tp_moe && cuda_tensor_parallel;
    g->cuda_tp_ep_pack_exact =
        g->cuda_tp_ep && metal_graph_cuda_tp_ep_pack_exact_requested();
    g->cuda_tp_moe_delay_reduce = metal_graph_cuda_tp_moe_delay_reduce_requested();
    g->cuda_tp_moe_copy3_handoff = metal_graph_cuda_tp_moe_copy3_handoff_requested();
    g->cuda_tp_moe_pack_handoff = metal_graph_cuda_tp_moe_pack_handoff_requested();
    g->cuda_tp_moe_peer_read = metal_graph_cuda_tp_moe_peer_read_requested();
    g->cuda_tp_moe_peer_router = metal_graph_cuda_tp_moe_peer_router_requested();
    g->cuda_tp_shared = g->cuda_tp_decode && metal_graph_cuda_tp_shared_requested();
    g->cuda_tp_shared_fold = metal_graph_cuda_tp_shared_fold_requested();
    g->cuda_tp_q = g->cuda_tp_decode && metal_graph_cuda_tp_q_requested();
    g->cuda_tp_output = g->cuda_tp_decode && metal_graph_cuda_tp_output_requested();
    g->cuda_tp_prefill_ffn = g->cuda_tp_decode && metal_graph_cuda_tp_prefill_ffn_requested();
    g->cuda_tp_prefill_attn_output =
        g->cuda_tp_decode && metal_graph_cuda_tp_prefill_attn_output_requested();
    g->cuda_q_norm_rope_fuse = metal_graph_cuda_q_norm_rope_fuse_requested();
    g->cuda_qkv_kv_rope_fuse = metal_graph_cuda_qkv_kv_rope_fuse_requested();
    g->cuda_qkv_pair = getenv("DS4_CUDA_NO_QKV_PAIR") == NULL;
    g->cuda_tp_attn_out_hc_fuse =
        getenv("DS4_CUDA_TP_ATTN_OUT_HC_FUSE") != NULL &&
        getenv("DS4_CUDA_NO_TP_ATTN_OUT_HC_FUSE") == NULL;
    g->shared_gate_up_swiglu_fuse =
        getenv("DS4_METAL_DISABLE_SHARED_GATE_UP_SWIGLU_FUSION") == NULL;
    g->decode_stage_profile = getenv("DS4_METAL_DECODE_STAGE_PROFILE") != NULL;
    g->decode_index_stage_profile = getenv("DS4_METAL_INDEXER_STAGE_PROFILE") != NULL;
    g->output_stage_profile = getenv("DS4_METAL_OUTPUT_STAGE_PROFILE") != NULL;
    const bool enable_splitkv_spec = metal_graph_cuda_splitkv_spec_requested();
    const bool enable_splitkv_batch_verify =
        enable_splitkv_spec && metal_graph_cuda_splitkv_spec_batch_verify_requested();
    const bool enable_spec_logits = enable_mtp || enable_splitkv_batch_verify;
    const bool enable_prefix1_snapshot = enable_mtp || enable_splitkv_spec;
    const bool enable_frontier_snapshot =
        enable_mtp ||
        enable_splitkv_spec ||
        (metal_graph_cuda_greedy_splitkv_requested() &&
         metal_graph_cuda_greedy_splitkv_fallback_requested()) ||
        (metal_graph_cuda_greedy_vec4_requested() &&
         metal_graph_cuda_greedy_vec4_fallback_requested());
    if (g->cuda_tp_decode && metal_graph_cuda_tp_partner_tier(0) < 0) {
        fprintf(stderr,
                "ds4: CUDA tensor parallelism requires an even multi-GPU placement; "
                "have %d GPU tiers\n",
                g_n_gpus);
        metal_graph_free(g);
        return false;
    }
    if (g->cuda_tp_ep &&
        (g_ds4_shape.family != DS4_MODEL_FAMILY_DEEPSEEK4 ||
         (DS4_N_EXPERT & 1u) != 0u)) {
        fprintf(stderr,
                "ds4: CUDA tensor parallelism requires an even-expert DeepSeek model\n");
        metal_graph_free(g);
        return false;
    }
    if (g->cuda_tp_ep) {
        fprintf(stderr,
                "ds4: CUDA routed MoE expert ownership enabled "
                "(half-resident decode and prefill)\n");
    }
    g->mtp_enabled = enable_mtp;
    if (raw_cap == 0) raw_cap = 1;
    if (ctx_size == 0) ctx_size = raw_cap;
    if (prefill_cap == 0) prefill_cap = 1;
    uint32_t raw_window = DS4_N_SWA;
    if (raw_window > ctx_size) raw_window = ctx_size;
    if (raw_window == 0) raw_window = 1;
    if (raw_cap < raw_window) raw_cap = raw_window;
    if (raw_cap > ctx_size) raw_cap = ctx_size;
    if (raw_cap == 0) raw_cap = 1;
    g->raw_cap = raw_cap;
    g->raw_window = raw_window;
    g->prefill_cap = prefill_cap;
    uint32_t min_ratio = UINT32_MAX;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        if (!weights_layer_has_required(&weights->layer[il], il)) continue;
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (ratio != 0 && ratio < min_ratio) min_ratio = ratio;
    }
    if (min_ratio == UINT32_MAX) min_ratio = ctx_size ? ctx_size : 1u;
    g->comp_cap = ctx_size / min_ratio + 2u;
    if (g->comp_cap < 2u) g->comp_cap = 2u;
    if (DS4_GPU_ATTN_COMP_CACHE_F16) {
        g->attn_comp_stage_cap = prefill_cap / min_ratio + 2u;
        if (g->attn_comp_stage_cap < 2u) g->attn_comp_stage_cap = 2u;
    }
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        if (!weights_layer_has_required(&weights->layer[il], il)) {
            g->layer_comp_cap[il] = 0;
            continue;
        }
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (ratio == 0) {
            g->layer_comp_cap[il] = 0;
        } else {
            g->layer_comp_cap[il] = ctx_size / ratio + 2u;
            if (g->layer_comp_cap[il] < 2u) g->layer_comp_cap[il] = 2u;
        }
    }

    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const uint64_t mix_hc = 2ull * DS4_N_HC + (uint64_t)DS4_N_HC * DS4_N_HC;
    const uint64_t q_rank = layer->attn_q_a->dim[1];
    const uint64_t q_dim = (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM;
    const uint64_t low_dim = (uint64_t)DS4_N_OUT_GROUP * DS4_N_LORA_O;
    const uint64_t group_dim = (uint64_t)DS4_N_HEAD_DIM * (DS4_N_HEAD / DS4_N_OUT_GROUP);
    const uint64_t shared_dim = layer->ffn_gate_shexp->dim[1];
    const uint64_t routed_mid_dim = layer->ffn_gate_exps->dim[1];
    /* Distributed coordinators do not normally own the output head. The
     * logits workspace still has a fixed model-vocabulary shape, while the
     * actual head is encoded only on a node that bound its tensors. */
    const uint64_t vocab_dim =
        weights->output ? weights->output->dim[1] : DS4_N_VOCAB;
    const uint64_t comp_width_max = 2ull * (DS4_N_HEAD_DIM > DS4_N_INDEXER_HEAD_DIM
        ? DS4_N_HEAD_DIM
        : DS4_N_INDEXER_HEAD_DIM);
    const uint64_t indexer_q_dim = (uint64_t)DS4_N_INDEXER_HEAD * DS4_N_INDEXER_HEAD_DIM;
    const uint64_t pc = prefill_cap;
    uint64_t kv_cache_bytes = 0;
    const uint64_t context_bytes =
        metal_graph_context_bytes_for_kv_policy(ctx_size, raw_cap, prefill_cap, &kv_cache_bytes);
    const bool managed_kv_cache =
        ds4_gpu_should_use_managed_kv_cache(kv_cache_bytes, context_bytes) != 0;
    if (managed_kv_cache) {
        /*
         * CUDA device allocations are fastest, but a million-token KV cache is
         * large enough to starve DGX Spark's unified CPU/GPU memory once the
         * model cache and driver allocations are present.  For this one
         * long-lived cache class, managed memory restores the old demand-paged
         * behavior.  It can be slower, but it keeps oversized contexts from
         * turning memory pressure into a machine-wide lockup.
         */
        fprintf(stderr,
                "ds4: CUDA using managed KV cache for ctx=%u "
                "(kv cache %.2f GiB, context buffers %.2f GiB); "
                "this may degrade performance but is needed for very large contexts\n",
                ctx_size,
                (double)kv_cache_bytes / 1073741824.0,
                (double)context_bytes / 1073741824.0);
    }

    /* Class P decode HC scratch — replicated across every tier
     * the placement uses (per-tier kernel-scratch). Single-tier path
     * (placement == NULL) collapses to tier 0 only; _ptr_on(0, ...) short-
     * circuits to legacy ds4_gpu_tensor_alloc when g_n_gpus <= 1 — byte-
     * equivalent. The hc_pre/hc_post/hc_comb buffers are VIEWS of hc_split
     * and therefore allocated per tier alongside their parent. */
    bool used_tier[DS4_MAX_GPUS] = {0};
    used_tier[0] = true; /* single-tier baseline always uses tier 0 */
    if (placement) {
        for (uint32_t i = 0; i < (uint32_t)DS4_N_LAYER + 2u; i++) {
            const int p = placement[i];
            if (p >= 0 && p < DS4_MAX_GPUS) used_tier[p] = true;
        }
    }
    if (g->cuda_tp_decode) {
        const int half = g_n_gpus / 2;
        for (int t = half; t < g_n_gpus; t++) {
            if (used_tier[t]) {
                fprintf(stderr,
                        "ds4: CUDA tensor parallelism expects layer homes in lower-half "
                        "tiers; placement already uses tier %d\n",
                        t);
                metal_graph_free(g);
                return false;
            }
        }
        for (int t = 0; t < half; t++) {
            if (used_tier[t]) used_tier[t + half] = true;
        }
        fprintf(stderr,
                "ds4: CUDA decode TP enabled: pairing lower-half tiers with "
                "upper-half tiers\n");
    }
    for (int t = 0; t < DS4_MAX_GPUS; t++) {
        if (!used_tier[t]) continue;
        g->cur_hc_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, hc_dim * sizeof(float));
        g->flat_hc_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, hc_dim * sizeof(float));
        g->hc_mix_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, mix_hc * sizeof(float));
        g->hc_split_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, mix_hc * sizeof(float));
        g->hc_pre_by_tier[t] = ds4_gpu_tensor_view(g->hc_split_by_tier[t],
                                                    0,
                                                    (uint64_t)DS4_N_HC * sizeof(float));
        g->hc_post_by_tier[t] = ds4_gpu_tensor_view(g->hc_split_by_tier[t],
                                                     (uint64_t)DS4_N_HC * sizeof(float),
                                                     (uint64_t)DS4_N_HC * sizeof(float));
        g->hc_comb_by_tier[t] = ds4_gpu_tensor_view(g->hc_split_by_tier[t],
                                                     2ull * DS4_N_HC * sizeof(float),
                                                     (uint64_t)DS4_N_HC * DS4_N_HC * sizeof(float));
        g->attn_cur_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, (uint64_t)DS4_N_EMBD * sizeof(float));
        g->attn_norm_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, (uint64_t)DS4_N_EMBD * sizeof(float));
        g->qr_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, q_rank * sizeof(float));
        g->qr_norm_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, q_rank * sizeof(float));
        g->q_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, q_dim * sizeof(float));
        g->kv_raw_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, (uint64_t)DS4_N_HEAD_DIM * sizeof(float));
        g->kv_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, (uint64_t)DS4_N_HEAD_DIM * sizeof(float));
    }
    bool state_init_ok = true;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        /* A distributed process owns only its bound layer slice. Persistent
         * KV state must follow that ownership just like the model tensors;
         * allocating every model layer here defeats split-model residency. */
        if (!weights_layer_has_required(&weights->layer[il], il)) continue;
        /* per-layer Class L allocations land on the layer's
         * home tier. placement is NULL on single-tier / diagnostic paths
         * (all-tier-0); non-NULL on the engine path that opted into
         * multi-tier. layer_tier == 0 in single-tier mode is the
         * byte-equivalent path through metal_graph_alloc_kv_cache_tensor_on
         * and ds4_gpu_tensor_alloc_ptr_on. */
        const int layer_tier = placement ? placement[il + 1] : 0;
        g->layer_raw_cache[il] = metal_graph_alloc_kv_cache_tensor_on(
                managed_kv_cache,
                layer_tier,
                (uint64_t)raw_cap * DS4_N_HEAD_DIM * sizeof(float));
        const int layer_tp_partner = g->cuda_tp_attn_cache_dup
            ? metal_graph_cuda_tp_partner_tier(layer_tier) : -1;
        if (layer_tp_partner >= 0) {
            g->layer_raw_cache_tp[il] = metal_graph_alloc_kv_cache_tensor_on(
                    managed_kv_cache,
                    layer_tp_partner,
                    (uint64_t)raw_cap * DS4_N_HEAD_DIM * sizeof(float));
        }
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (ratio != 0) {
            const uint32_t coff = ratio == 4 ? 2u : 1u;
            const uint64_t attn_width = (uint64_t)coff * DS4_N_HEAD_DIM;
            const uint64_t attn_rows = (uint64_t)coff * ratio;
            g->layer_attn_comp_cache[il] = metal_graph_alloc_kv_cache_tensor_on(
                    managed_kv_cache,
                    layer_tier,
                    (uint64_t)g->layer_comp_cap[il] * DS4_N_HEAD_DIM *
                    (DS4_GPU_ATTN_COMP_CACHE_F16 ? sizeof(uint16_t) : sizeof(float)));
            if (layer_tp_partner >= 0) {
                g->layer_attn_comp_cache_tp[il] = metal_graph_alloc_kv_cache_tensor_on(
                        managed_kv_cache,
                        layer_tp_partner,
                        (uint64_t)g->layer_comp_cap[il] * DS4_N_HEAD_DIM *
                        (DS4_GPU_ATTN_COMP_CACHE_F16 ? sizeof(uint16_t) : sizeof(float)));
            }
            g->layer_attn_state_kv[il] = ds4_gpu_tensor_alloc_ptr_on(layer_tier, attn_width * attn_rows * sizeof(float));
            g->layer_attn_state_score[il] = ds4_gpu_tensor_alloc_ptr_on(layer_tier, attn_width * attn_rows * sizeof(float));
            if (enable_frontier_snapshot) {
                g->spec_attn_state_kv[il] =
                    ds4_gpu_tensor_alloc_ptr_on(layer_tier, attn_width * attn_rows * sizeof(float));
                g->spec_attn_state_score[il] =
                    ds4_gpu_tensor_alloc_ptr_on(layer_tier, attn_width * attn_rows * sizeof(float));
                if (enable_prefix1_snapshot) {
                    g->spec_prefix1_attn_state_kv[il] =
                        ds4_gpu_tensor_alloc_ptr_on(layer_tier, attn_width * attn_rows * sizeof(float));
                    g->spec_prefix1_attn_state_score[il] =
                        ds4_gpu_tensor_alloc_ptr_on(layer_tier, attn_width * attn_rows * sizeof(float));
                }
            }
            if (g->layer_attn_state_kv[il]) {
                state_init_ok = state_init_ok &&
                                metal_tensor_fill_f32(g->layer_attn_state_kv[il], 0.0f, attn_width * attn_rows);
            }
            if (g->layer_attn_state_score[il]) {
                state_init_ok = state_init_ok &&
                                metal_tensor_fill_f32(g->layer_attn_state_score[il], DS4_NEG_INF, attn_width * attn_rows);
            }

            if (ratio == 4) {
                const uint64_t index_width = (uint64_t)coff * DS4_N_INDEXER_HEAD_DIM;
                const uint64_t index_rows = (uint64_t)coff * ratio;
                g->layer_index_comp_cache[il] = metal_graph_alloc_kv_cache_tensor_on(
                        managed_kv_cache,
                        layer_tier,
                        (uint64_t)g->layer_comp_cap[il] * DS4_N_INDEXER_HEAD_DIM * sizeof(float));
                g->layer_index_state_kv[il] = ds4_gpu_tensor_alloc_ptr_on(layer_tier, index_width * index_rows * sizeof(float));
                g->layer_index_state_score[il] = ds4_gpu_tensor_alloc_ptr_on(layer_tier, index_width * index_rows * sizeof(float));
                if (enable_frontier_snapshot) {
                    g->spec_index_state_kv[il] =
                        ds4_gpu_tensor_alloc_ptr_on(layer_tier, index_width * index_rows * sizeof(float));
                    g->spec_index_state_score[il] =
                        ds4_gpu_tensor_alloc_ptr_on(layer_tier, index_width * index_rows * sizeof(float));
                    if (enable_prefix1_snapshot) {
                        g->spec_prefix1_index_state_kv[il] =
                            ds4_gpu_tensor_alloc_ptr_on(layer_tier, index_width * index_rows * sizeof(float));
                        g->spec_prefix1_index_state_score[il] =
                            ds4_gpu_tensor_alloc_ptr_on(layer_tier, index_width * index_rows * sizeof(float));
                    }
                }
                if (g->layer_index_state_kv[il]) {
                    state_init_ok = state_init_ok &&
                                    metal_tensor_fill_f32(g->layer_index_state_kv[il], 0.0f, index_width * index_rows);
                }
                if (g->layer_index_state_score[il]) {
                    state_init_ok = state_init_ok &&
                                    metal_tensor_fill_f32(g->layer_index_state_score[il], DS4_NEG_INF, index_width * index_rows);
                }
            }
        }
    }
    /* Class P per-layer decode scratch + routed-expert state —
     * replicated across every used tier. ffn_out is lazily allocated by
     * metal_graph_ensure_ffn_out (per-tier on first touch). */
    for (int t = 0; t < DS4_MAX_GPUS; t++) {
        if (!used_tier[t]) continue;
        g->comp_kv_cur_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, comp_width_max * sizeof(float));
        g->comp_sc_cur_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, comp_width_max * sizeof(float));
        if (DS4_GPU_ATTN_COMP_CACHE_F16) {
            /* Upstream's F16-compressed attn staging buffer. Only allocated when
             * the F16-cache mode is enabled (the non-F16 path stages in-place). */
            g->attn_comp_stage_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t,
                    (uint64_t)g->attn_comp_stage_cap * DS4_N_HEAD_DIM * sizeof(float));
        }
        g->indexer_q_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, indexer_q_dim * sizeof(float));
        g->indexer_weights_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, (uint64_t)DS4_N_INDEXER_HEAD * sizeof(float));
        g->indexer_scores_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, (uint64_t)g->comp_cap * pc * sizeof(float));
        g->comp_mask_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, (uint64_t)g->comp_cap * pc * sizeof(float));
        g->comp_selected_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t,
                (uint64_t)(DS4_N_INDEXER_TOP_K ? DS4_N_INDEXER_TOP_K : 1u) * pc * sizeof(uint32_t));
        g->heads_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, q_dim * sizeof(float));
        g->attn_low_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, low_dim * sizeof(float));
        g->attn_out_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, (uint64_t)DS4_N_EMBD * sizeof(float));
        g->after_attn_hc_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, hc_dim * sizeof(float));
        g->ffn_cur_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, (uint64_t)DS4_N_EMBD * sizeof(float));
        g->ffn_norm_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, (uint64_t)DS4_N_EMBD * sizeof(float));
        g->shared_gate_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, shared_dim * sizeof(float));
        g->shared_up_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, shared_dim * sizeof(float));
        g->shared_mid_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, shared_dim * sizeof(float));
        g->shared_out_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, (uint64_t)DS4_N_EMBD * sizeof(float));
        g->router_logits_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, DS4_N_EXPERT * sizeof(float));
        g->router_probs_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, DS4_N_EXPERT * sizeof(float));
        g->router_selected_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, DS4_N_EXPERT_USED * sizeof(int));
        g->router_weights_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, DS4_N_EXPERT_USED * sizeof(float));
        g->routed_gate_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t,
                (uint64_t)DS4_N_EXPERT_USED * routed_mid_dim * sizeof(float));
        g->routed_up_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t,
                (uint64_t)DS4_N_EXPERT_USED * routed_mid_dim * sizeof(float));
        g->routed_mid_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t,
                (uint64_t)DS4_N_EXPERT_USED * routed_mid_dim * sizeof(float));
        g->routed_down_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t,
                (uint64_t)DS4_N_EXPERT_USED * DS4_N_EMBD * sizeof(float));
        g->routed_out_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, (uint64_t)DS4_N_EMBD * sizeof(float));
        if (g->cuda_tp_decode) {
            g->tp_peer_tmp_by_tier[t] =
                ds4_gpu_tensor_alloc_ptr_on(t, DS4_CUDA_TP_PEER_TMP_BYTES);
        }
        g->after_ffn_hc_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, hc_dim * sizeof(float));
    }
    /* Class H — head_tier captured from placement[DS4_N_LAYER + 1]
     * (or 0 in single-tier / diagnostic paths). Output-head tensors and the
     * final logits buffer allocate on head_tier only; other tier slots stay
     * NULL. The _ptr_on(0, ...) path short-circuits to the legacy
     * ds4_gpu_tensor_alloc when g_n_gpus <= 1 — byte-equivalent. */
    g->head_tier = placement ? placement[DS4_N_LAYER + 1] : 0;
    int output_tp_tiers[DS4_MAX_GPUS] = {0};
    const uint32_t output_tp_ways = g->cuda_tp_output
        ? metal_graph_cuda_tp_output_tiers(g, output_tp_tiers) : 0;
    if (g->cuda_tp_output && output_tp_ways < 2u) {
        fprintf(stderr,
                "ds4: CUDA output TP requires output head tier %d to be in "
                "the lower half of the CUDA placement\n",
                g->head_tier);
        metal_graph_free(g);
        return false;
    }
    uint64_t output_logits_elems = vocab_dim;
    if (enable_spec_logits && output_tp_ways >= 2u) {
        const uint64_t max_shard_vocab =
            (vocab_dim + output_tp_ways - 1u) / output_tp_ways;
        const uint64_t spec_shard_elems =
            (uint64_t)DS4_DSPARK_MAX_BLOCK_SIZE * max_shard_vocab;
        if (spec_shard_elems > output_logits_elems) {
            output_logits_elems = spec_shard_elems;
        }
    }
    g->output_pre_by_tier[g->head_tier] =
        ds4_gpu_tensor_alloc_ptr_on(g->head_tier, (uint64_t)DS4_N_HC * sizeof(float));
    g->output_weights_by_tier[g->head_tier] =
        ds4_gpu_tensor_alloc_ptr_on(g->head_tier, (uint64_t)DS4_N_HC * sizeof(float));
    g->output_embd_by_tier[g->head_tier] =
        ds4_gpu_tensor_alloc_ptr_on(g->head_tier, (uint64_t)DS4_N_EMBD * sizeof(float));
    g->output_norm_by_tier[g->head_tier] =
        ds4_gpu_tensor_alloc_ptr_on(g->head_tier, (uint64_t)DS4_N_EMBD * sizeof(float));
    g->logits_by_tier[g->head_tier] =
        ds4_gpu_tensor_alloc_ptr_on(g->head_tier,
                                    output_logits_elems * sizeof(float));
    for (uint32_t i = 1; i < output_tp_ways; i++) {
        const int t = output_tp_tiers[i];
        if (t < 0 || t >= DS4_MAX_GPUS || t == g->head_tier) continue;
        g->output_norm_by_tier[t] =
            ds4_gpu_tensor_alloc_ptr_on(t,
                                        (uint64_t)DS4_N_EMBD * sizeof(float));
        g->logits_by_tier[t] =
            ds4_gpu_tensor_alloc_ptr_on(t,
                                        output_logits_elems * sizeof(float));
    }
    /*
     * MTP is deliberately outside the normal graph footprint.  A session that
     * does not opt in with --mtp must allocate and execute exactly the same
     * buffers as the plain decoder: no support-model mapping, no draft logits,
     * and no MTP scratch hidden behind otherwise unused tensors.
     */
    if (enable_mtp) {
        g->mtp_embed = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
        g->mtp_enorm = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
        g->mtp_eproj = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
        g->mtp_eproj_hc = ds4_gpu_tensor_alloc(hc_dim * sizeof(float));
        g->mtp_hnorm_hc = ds4_gpu_tensor_alloc(hc_dim * sizeof(float));
        g->mtp_hproj_hc = ds4_gpu_tensor_alloc(hc_dim * sizeof(float));
        g->mtp_input_hc = ds4_gpu_tensor_alloc(hc_dim * sizeof(float));
        g->mtp_state_hc = ds4_gpu_tensor_alloc(hc_dim * sizeof(float));
        g->mtp_next_hc = ds4_gpu_tensor_alloc(hc_dim * sizeof(float));
        g->mtp_raw_cache = metal_graph_alloc_kv_cache_tensor(
                managed_kv_cache,
                (uint64_t)raw_cap * DS4_N_HEAD_DIM * sizeof(float));
        g->mtp_n_raw = 0;
    }
    if (enable_spec_logits) {
        const int spec_tier =
            g->dspark_exec_tier > 0 && g->dspark_exec_tier < DS4_MAX_GPUS
                ? g->dspark_exec_tier : 0;
        g->spec_logits = spec_tier
            ? ds4_gpu_tensor_alloc_ptr_on(spec_tier, (uint64_t)16 * DS4_N_VOCAB * sizeof(float))
            : ds4_gpu_tensor_alloc((uint64_t)16 * DS4_N_VOCAB * sizeof(float));
    }

    /* Class E — emb_tier captured from placement[0] (or 0 in
     * single-tier / diagnostic paths). _ptr_on(0, ...) short-circuits to the
     * legacy ds4_gpu_tensor_alloc when g_n_gpus <= 1 — byte-equivalent. */
    g->emb_tier = placement ? placement[0] : 0;
    /* Class P chunked-prefill batch scratch — replicated across
     * every used tier. The cur/next pair (batch_cur_hc / batch_next_hc) is
     * ping-ponged per layer step on each tier; tier transitions copy via
     * ds4_gpu_tensor_copy_xdev (handled in B6). batch_ffn_out is lazily
     * allocated by metal_graph_ensure_batch_ffn_out (per-tier on first touch)
     * and included in the CUDA scratch estimate because TP prefill can use it
     * as the combined routed+shared FFN buffer. */
    if (shared_prefill_workspace) {
        if (shared_prefill_workspace->prefill_cap < prefill_cap ||
            shared_prefill_workspace->emb_tier != g->emb_tier) {
            fprintf(stderr,
                    "ds4: shared prefill workspace is incompatible "
                    "(capacity %u/%u, embedding tier %d/%d)\n",
                    shared_prefill_workspace->prefill_cap,
                    prefill_cap,
                    shared_prefill_workspace->emb_tier,
                    g->emb_tier);
        } else {
            metal_graph_copy_prefill_workspace_pointers(
                    g, shared_prefill_workspace);
        }
    } else {
        g->prefill_tokens_by_tier[g->emb_tier] =
            ds4_gpu_tensor_alloc_ptr_on(g->emb_tier, pc * sizeof(int32_t));
        for (int t = 0; t < DS4_MAX_GPUS; t++) {
            if (!used_tier[t]) continue;
            g->batch_cur_hc_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, pc * hc_dim * sizeof(float));
            g->batch_next_hc_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, pc * hc_dim * sizeof(float));
            g->batch_flat_hc_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, pc * hc_dim * sizeof(float));
            g->batch_hc_mix_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, pc * mix_hc * sizeof(float));
            g->batch_hc_split_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, pc * mix_hc * sizeof(float));
            g->batch_attn_cur_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, pc * DS4_N_EMBD * sizeof(float));
            g->batch_attn_norm_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, pc * DS4_N_EMBD * sizeof(float));
            g->batch_qr_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, pc * q_rank * sizeof(float));
            g->batch_qr_norm_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, pc * q_rank * sizeof(float));
            g->batch_q_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, pc * q_dim * sizeof(float));
            g->batch_kv_raw_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, pc * DS4_N_HEAD_DIM * sizeof(float));
            g->batch_kv_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, pc * DS4_N_HEAD_DIM * sizeof(float));
            g->batch_comp_kv_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, pc * comp_width_max * sizeof(float));
            g->batch_comp_sc_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, pc * comp_width_max * sizeof(float));
            g->batch_indexer_q_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, pc * indexer_q_dim * sizeof(float));
            g->batch_indexer_weights_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, pc * DS4_N_INDEXER_HEAD * sizeof(float));
            g->batch_heads_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, pc * q_dim * sizeof(float));
            g->batch_attn_low_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, pc * low_dim * sizeof(float));
            g->batch_attn_out_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, pc * DS4_N_EMBD * sizeof(float));
            g->batch_group_tmp_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, pc * group_dim * sizeof(float));
            g->batch_low_tmp_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, pc * DS4_N_LORA_O * sizeof(float));
            g->batch_after_attn_hc_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, pc * hc_dim * sizeof(float));
            g->batch_ffn_cur_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, pc * DS4_N_EMBD * sizeof(float));
            g->batch_ffn_norm_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, pc * DS4_N_EMBD * sizeof(float));
            g->batch_shared_gate_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, pc * shared_dim * sizeof(float));
            g->batch_shared_up_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, pc * shared_dim * sizeof(float));
            g->batch_shared_mid_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, pc * shared_dim * sizeof(float));
            g->batch_shared_out_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, pc * DS4_N_EMBD * sizeof(float));
            g->batch_router_logits_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, pc * DS4_N_EXPERT * sizeof(float));
            g->batch_router_probs_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, pc * DS4_N_EXPERT * sizeof(float));
            g->batch_router_selected_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, pc * DS4_N_EXPERT_USED * sizeof(int));
            g->batch_router_weights_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, pc * DS4_N_EXPERT_USED * sizeof(float));
            g->batch_routed_gate_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, pc * DS4_N_EXPERT_USED * routed_mid_dim * sizeof(float));
            g->batch_routed_up_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, pc * DS4_N_EXPERT_USED * routed_mid_dim * sizeof(float));
            g->batch_routed_mid_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, pc * DS4_N_EXPERT_USED * routed_mid_dim * sizeof(float));
            g->batch_routed_down_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, pc * DS4_N_EXPERT_USED * DS4_N_EMBD * sizeof(float));
            g->batch_routed_out_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, pc * DS4_N_EMBD * sizeof(float));
        }
        if (DS4_GPU_ATTN_COMP_CACHE_F16) {
            g->batch_q_half = ds4_gpu_tensor_alloc(pc * q_dim * sizeof(uint16_t));
        }
        g->prefill_seed_router_selected = ds4_gpu_tensor_alloc(
                (uint64_t)DS4_N_LAYER * DS4_STREAMING_PREFILL_CACHE_SEED_MAX_TOKENS *
                DS4_N_EXPERT_USED * sizeof(int32_t));
    }

    bool layer_cache_ok = true;
    for (uint32_t il = 0; layer_cache_ok && il < DS4_N_LAYER; il++) {
        if (!weights_layer_has_required(&weights->layer[il], il)) continue;
        layer_cache_ok = g->layer_raw_cache[il] != NULL;
        if (layer_cache_ok && g->cuda_tp_attn_cache_dup) {
            layer_cache_ok = g->layer_raw_cache_tp[il] != NULL;
        }
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (layer_cache_ok && ratio != 0) {
            layer_cache_ok = g->layer_attn_comp_cache[il] != NULL &&
                             (!g->cuda_tp_attn_cache_dup ||
                              g->layer_attn_comp_cache_tp[il] != NULL) &&
                             g->layer_attn_state_kv[il] != NULL &&
                             g->layer_attn_state_score[il] != NULL &&
                             (!enable_frontier_snapshot ||
                              (g->spec_attn_state_kv[il] != NULL &&
                               g->spec_attn_state_score[il] != NULL)) &&
                             (!enable_prefix1_snapshot ||
                              (g->spec_prefix1_attn_state_kv[il] != NULL &&
                               g->spec_prefix1_attn_state_score[il] != NULL));
        }
        if (layer_cache_ok && ratio == 4) {
            layer_cache_ok = g->layer_index_comp_cache[il] != NULL &&
                             g->layer_index_state_kv[il] != NULL &&
                             g->layer_index_state_score[il] != NULL &&
                             (!enable_frontier_snapshot ||
                              (g->spec_index_state_kv[il] != NULL &&
                               g->spec_index_state_score[il] != NULL)) &&
                             (!enable_prefix1_snapshot ||
                              (g->spec_prefix1_index_state_kv[il] != NULL &&
                               g->spec_prefix1_index_state_score[il] != NULL));
        }
    }

    /* Class P validation — check every used tier's slot. */
    bool class_p_ok = true;
    for (int t = 0; class_p_ok && t < DS4_MAX_GPUS; t++) {
        if (!used_tier[t]) continue;
        class_p_ok =
            g->cur_hc_by_tier[t] && g->flat_hc_by_tier[t] && g->hc_mix_by_tier[t] && g->hc_split_by_tier[t] &&
            g->hc_pre_by_tier[t] && g->hc_post_by_tier[t] && g->hc_comb_by_tier[t] &&
            g->attn_cur_by_tier[t] && g->attn_norm_by_tier[t] && g->qr_by_tier[t] && g->qr_norm_by_tier[t] &&
            g->q_by_tier[t] && g->kv_raw_by_tier[t] && g->kv_by_tier[t] &&
            g->comp_kv_cur_by_tier[t] && g->comp_sc_cur_by_tier[t] &&
            (!DS4_GPU_ATTN_COMP_CACHE_F16 || g->attn_comp_stage_by_tier[t]) &&
            g->indexer_q_by_tier[t] && g->indexer_weights_by_tier[t] && g->indexer_scores_by_tier[t] &&
            g->comp_mask_by_tier[t] && g->comp_selected_by_tier[t] &&
            g->heads_by_tier[t] && g->attn_low_by_tier[t] && g->attn_out_by_tier[t] &&
            g->after_attn_hc_by_tier[t] && g->ffn_cur_by_tier[t] && g->ffn_norm_by_tier[t] &&
            g->shared_gate_by_tier[t] && g->shared_up_by_tier[t] && g->shared_mid_by_tier[t] &&
            g->shared_out_by_tier[t] &&
            g->router_logits_by_tier[t] && g->router_probs_by_tier[t] &&
            g->router_selected_by_tier[t] && g->router_weights_by_tier[t] &&
            g->routed_gate_by_tier[t] && g->routed_up_by_tier[t] && g->routed_mid_by_tier[t] &&
            g->routed_down_by_tier[t] && g->routed_out_by_tier[t] &&
            (!g->cuda_tp_decode || g->tp_peer_tmp_by_tier[t]) &&
            g->after_ffn_hc_by_tier[t] &&
            g->batch_cur_hc_by_tier[t] && g->batch_next_hc_by_tier[t] && g->batch_flat_hc_by_tier[t] &&
            g->batch_hc_mix_by_tier[t] && g->batch_hc_split_by_tier[t] &&
            g->batch_attn_cur_by_tier[t] && g->batch_attn_norm_by_tier[t] &&
            g->batch_qr_by_tier[t] && g->batch_qr_norm_by_tier[t] && g->batch_q_by_tier[t] &&
            g->batch_kv_raw_by_tier[t] && g->batch_kv_by_tier[t] &&
            g->batch_comp_kv_by_tier[t] && g->batch_comp_sc_by_tier[t] &&
            g->batch_indexer_q_by_tier[t] && g->batch_indexer_weights_by_tier[t] &&
            g->batch_heads_by_tier[t] && g->batch_attn_low_by_tier[t] && g->batch_attn_out_by_tier[t] &&
            g->batch_group_tmp_by_tier[t] && g->batch_low_tmp_by_tier[t] && g->batch_after_attn_hc_by_tier[t] &&
            g->batch_ffn_cur_by_tier[t] && g->batch_ffn_norm_by_tier[t] &&
            g->batch_shared_gate_by_tier[t] && g->batch_shared_up_by_tier[t] &&
            g->batch_shared_mid_by_tier[t] && g->batch_shared_out_by_tier[t] &&
            g->batch_router_logits_by_tier[t] && g->batch_router_probs_by_tier[t] &&
            g->batch_router_selected_by_tier[t] && g->batch_router_weights_by_tier[t] &&
            g->batch_routed_gate_by_tier[t] && g->batch_routed_up_by_tier[t] &&
            g->batch_routed_mid_by_tier[t] && g->batch_routed_down_by_tier[t] &&
            g->batch_routed_out_by_tier[t];
    }
    bool output_tp_ok = true;
    for (uint32_t i = 1; i < output_tp_ways; i++) {
        const int t = output_tp_tiers[i];
        if (t < 0 || t >= DS4_MAX_GPUS || t == g->head_tier) continue;
        output_tp_ok = output_tp_ok &&
                       g->output_norm_by_tier[t] != NULL &&
                       g->logits_by_tier[t] != NULL;
    }
    const bool ok = state_init_ok && layer_cache_ok && class_p_ok &&
                    /* Class H — validate the head_tier slot
                     * (single-tier: head_tier == 0, byte-equivalent). */
                    metal_graph_output_pre(g) && metal_graph_output_weights(g) &&
                    metal_graph_output_embd(g) && metal_graph_output_norm(g) &&
                    metal_graph_logits(g) && output_tp_ok &&
                    (!enable_mtp ||
                     (g->mtp_embed && g->mtp_enorm && g->mtp_eproj &&
                      g->mtp_eproj_hc && g->mtp_hnorm_hc && g->mtp_hproj_hc &&
                      g->mtp_input_hc && g->mtp_state_hc && g->mtp_next_hc &&
                      g->mtp_raw_cache)) &&
                    (!enable_spec_logits || g->spec_logits) &&
                    /* Class E — validate the emb_tier slot. */
                    metal_graph_prefill_tokens(g) &&
                    g->cpu_router_norm &&
                    (!DS4_GPU_ATTN_COMP_CACHE_F16 || g->batch_q_half) &&
                    g->prefill_seed_router_selected;
    if (!ok) metal_graph_free(g);
    return ok;
}

static bool metal_graph_alloc(
        ds4_gpu_graph *g,
        const ds4_weights     *weights,
        const ds4_layer_weights *layer) {
    /* single-tier convenience wrapper; placement=NULL routes
     * all per-layer allocations to tier 0. */
    return metal_graph_alloc_raw_cap(g, weights, layer, DS4_N_SWA, DS4_N_SWA,
                                     1, false, NULL, false, NULL);
}

static bool metal_graph_install_model_spans(
        const ds4_model              *model,
        const ds4_model_map_span_vec *spans,
        const char                   *label) {
    if (!model || !spans || spans->len == 0) return false;

    uint64_t *offsets = xmalloc((size_t)spans->len * sizeof(offsets[0]));
    uint64_t *sizes = xmalloc((size_t)spans->len * sizeof(sizes[0]));
    for (uint32_t i = 0; i < spans->len; i++) {
        offsets[i] = spans->v[i].off;
        sizes[i] = spans->v[i].end - spans->v[i].off;
    }

    const bool ok = ds4_gpu_set_model_map_spans(model->map,
                                                model->size,
                                                offsets,
                                                sizes,
                                                spans->len,
                                                spans->max_tensor_bytes) != 0;
    if (!ok) {
        fprintf(stderr,
                "ds4: Metal SSD streaming failed to map %s model spans\n",
                label ? label : "requested");
    }
    free(offsets);
    free(sizes);
    return ok;
}

static bool metal_graph_stream_readahead_enabled(void) {
    return glm_graph_env_present("DS4_ROCM_ENABLE_STREAMING_READAHEAD",
                                 "DS4_METAL_ENABLE_STREAMING_READAHEAD") &&
           !glm_graph_env_present("DS4_ROCM_DISABLE_STREAMING_READAHEAD",
                                  "DS4_METAL_DISABLE_STREAMING_READAHEAD");
}

static bool metal_graph_stream_madvise_willneed_enabled(void) {
    return glm_graph_env_present("DS4_ROCM_ENABLE_STREAMING_MADVISE_WILLNEED",
                                 "DS4_METAL_ENABLE_STREAMING_MADVISE_WILLNEED") &&
           !glm_graph_env_present("DS4_ROCM_DISABLE_STREAMING_MADVISE_WILLNEED",
                                  "DS4_METAL_DISABLE_STREAMING_MADVISE_WILLNEED");
}

static bool metal_graph_stream_decode_static_map_enabled(void) {
    if (glm_graph_env_present("DS4_ROCM_DISABLE_STREAMING_STATIC_DECODE_MAP",
                              "DS4_METAL_DISABLE_STREAMING_STATIC_DECODE_MAP")) {
        return false;
    }
#ifdef DS4_ROCM_BUILD
    return glm_graph_env_present("DS4_ROCM_ENABLE_STREAMING_STATIC_DECODE_MAP",
                                 "DS4_METAL_ENABLE_STREAMING_STATIC_DECODE_MAP");
#else
    return true;
#endif
}

static bool metal_graph_stream_decode_static_map_state_cache_enabled(void) {
    return !glm_graph_env_present("DS4_ROCM_DISABLE_STREAMING_STATIC_MAP_STATE_CACHE",
                                  "DS4_METAL_DISABLE_STREAMING_STATIC_MAP_STATE_CACHE");
}

static bool metal_graph_stream_decode_layer_batch_enabled(
        const ds4_gpu_graph *g) {
    return g &&
           g->ssd_streaming &&
           !g_expert_profile.active &&
           !glm_graph_env_present("DS4_ROCM_DISABLE_STREAMING_LAYER_BATCH",
                                  "DS4_METAL_DISABLE_STREAMING_LAYER_BATCH") &&
           (!glm_graph_env_present("DS4_ROCM_ENABLE_STREAMING_FULL_EXPERT_ADDR_TABLE",
                                   "DS4_METAL_ENABLE_STREAMING_FULL_EXPERT_ADDR_TABLE") ||
            glm_graph_env_present("DS4_ROCM_DISABLE_STREAMING_FULL_EXPERT_ADDR_TABLE",
                                  "DS4_METAL_DISABLE_STREAMING_FULL_EXPERT_ADDR_TABLE")) &&
           !glm_graph_env_present("DS4_ROCM_DECODE_STAGE_PROFILE",
                                  "DS4_METAL_DECODE_STAGE_PROFILE") &&
           !glm_graph_env_present("DS4_ROCM_GRAPH_DUMP_PREFIX",
                                  "DS4_METAL_GRAPH_DUMP_PREFIX");
}

static void metal_graph_stream_readahead_range_impl(
        const ds4_model *model,
        uint64_t         offset,
        uint64_t         size,
        bool             enabled) {
    if (!enabled ||
        !model ||
        model->fd < 0 ||
        !model->map ||
        offset > model->size ||
        size == 0 ||
        size > model->size - offset) {
        return;
    }

#if defined(F_RDADVISE)
    uint64_t pos = offset;
    uint64_t rem = size;
    while (rem > 0) {
        const uint64_t chunk64 =
            rem > (uint64_t)INT_MAX ? (uint64_t)INT_MAX : rem;
        if (pos > (uint64_t)LLONG_MAX) break;

        struct radvisory ra;
        ra.ra_offset = (off_t)pos;
        ra.ra_count = (int)chunk64;
        (void)fcntl(model->fd, F_RDADVISE, &ra);

        pos += chunk64;
        rem -= chunk64;
    }
#else
    (void)model;
    (void)offset;
    (void)size;
#endif
}

static bool metal_graph_stream_madvise_willneed_range_impl(
        const ds4_model *model,
        uint64_t         offset,
        uint64_t         size,
        bool             enabled,
        uint64_t        *advised) {
    if (!enabled ||
        !model ||
        !model->map ||
        offset > model->size ||
        size == 0 ||
        size > model->size - offset) {
        return !enabled;
    }

#if defined(POSIX_MADV_WILLNEED)
    const uint64_t page = (uint64_t)getpagesize();
    if (page == 0) return false;
    const uint64_t page_offset = offset & ~(page - 1u);
    const uint64_t leading = offset - page_offset;
    if (size > UINT64_MAX - leading ||
        leading + size > UINT64_MAX - (page - 1u)) {
        return false;
    }
    uint64_t advise_bytes = align_up(leading + size, page);
    if (advise_bytes > model->size - page_offset) {
        advise_bytes = model->size - page_offset;
    }
    if (advise_bytes == 0 || advise_bytes > (uint64_t)SIZE_MAX) {
        return false;
    }
    uint8_t *base = (uint8_t *)model->map;
    const int rc = posix_madvise((void *)(base + page_offset),
                                 (size_t)advise_bytes,
                                 POSIX_MADV_WILLNEED);
    if (rc != 0) return false;
    if (advised) {
        if (*advised > UINT64_MAX - advise_bytes) {
            *advised = UINT64_MAX;
        } else {
            *advised += advise_bytes;
        }
    }
    return true;
#else
    (void)model;
    (void)offset;
    (void)size;
    (void)advised;
    return true;
#endif
}

static void metal_graph_stream_readahead_range(
        const ds4_model *model,
        uint64_t         offset,
        uint64_t         size) {
    metal_graph_stream_readahead_range_impl(model,
                                            offset,
                                            size,
                                            metal_graph_stream_readahead_enabled());
    metal_graph_stream_madvise_willneed_range_impl(
            model,
            offset,
            size,
            metal_graph_stream_madvise_willneed_enabled(),
            NULL);
}

static void metal_graph_stream_readahead_spans(
        const ds4_model              *model,
        const ds4_model_map_span_vec *spans) {
    if (!spans) return;
    for (uint32_t i = 0; i < spans->len; i++) {
        metal_graph_stream_readahead_range(model,
                                           spans->v[i].off,
                                           spans->v[i].end - spans->v[i].off);
    }
}

typedef struct {
    uint64_t off;
    uint64_t size;
} metal_graph_stream_pagein_range;

typedef struct {
    pthread_t thread;
    const ds4_model *model;
    metal_graph_stream_pagein_range *ranges;
    pthread_t *threads;
    struct metal_graph_stream_pagein_worker *workers;
    uint32_t n_ranges;
    uint32_t n_threads;
    uint32_t layer;
    uint32_t n_tokens;
    uint32_t unique;
    uint64_t bytes;
    uint64_t touched;
    double read_ms;
    double thread_ms;
    bool profile;
    bool madvise_only;
    bool pread_only;
    bool readahead_only;
    bool started;
    bool ok;
    uint8_t sink;
} metal_graph_stream_pagein_job;

typedef struct metal_graph_stream_pagein_worker {
    metal_graph_stream_pagein_job *job;
    uint32_t first;
    uint32_t stride;
    uint64_t touched;
    double thread_ms;
    bool ok;
    uint8_t sink;
} metal_graph_stream_pagein_worker;

static bool metal_graph_stream_prefill_selected_pagein_enabled(
        const ds4_gpu_graph *g) {
    return g &&
           g->ssd_streaming &&
           glm_graph_env_present("DS4_ROCM_ENABLE_STREAMING_PREFILL_SELECTED_PAGEIN",
                                 "DS4_METAL_ENABLE_STREAMING_PREFILL_SELECTED_PAGEIN") &&
           !glm_graph_env_present("DS4_ROCM_DISABLE_STREAMING_PREFILL_SELECTED_PAGEIN",
                                  "DS4_METAL_DISABLE_STREAMING_PREFILL_SELECTED_PAGEIN");
}

static bool metal_graph_stream_prefill_selected_madvise_enabled(
        const ds4_gpu_graph *g) {
    return g &&
           g->ssd_streaming &&
           glm_graph_env_present("DS4_ROCM_ENABLE_STREAMING_PREFILL_SELECTED_MADVISE",
                                 "DS4_METAL_ENABLE_STREAMING_PREFILL_SELECTED_MADVISE") &&
           !glm_graph_env_present("DS4_ROCM_DISABLE_STREAMING_PREFILL_SELECTED_MADVISE",
                                  "DS4_METAL_DISABLE_STREAMING_PREFILL_SELECTED_MADVISE");
}

static bool metal_graph_stream_prefill_layer_pagein_enabled(
        const ds4_gpu_graph *g) {
    return g &&
           g->ssd_streaming &&
           glm_graph_env_present("DS4_ROCM_ENABLE_STREAMING_PREFILL_LAYER_PAGEIN",
                                 "DS4_METAL_ENABLE_STREAMING_PREFILL_LAYER_PAGEIN") &&
           !glm_graph_env_present("DS4_ROCM_DISABLE_STREAMING_PREFILL_LAYER_PAGEIN",
                                  "DS4_METAL_DISABLE_STREAMING_PREFILL_LAYER_PAGEIN");
}

static bool metal_graph_stream_prefill_layer_readahead_enabled(
        const ds4_gpu_graph *g) {
    return g &&
           g->ssd_streaming &&
           glm_graph_env_present("DS4_ROCM_ENABLE_STREAMING_PREFILL_LAYER_READAHEAD",
                                 "DS4_METAL_ENABLE_STREAMING_PREFILL_LAYER_READAHEAD") &&
           !glm_graph_env_present("DS4_ROCM_DISABLE_STREAMING_PREFILL_LAYER_READAHEAD",
                                  "DS4_METAL_DISABLE_STREAMING_PREFILL_LAYER_READAHEAD") &&
           !glm_graph_env_present("DS4_ROCM_DISABLE_STREAMING_PREFILL_LAYER_PREPARE",
                                  "DS4_METAL_DISABLE_STREAMING_PREFILL_LAYER_PREPARE");
}

static bool metal_graph_stream_prefill_layer_pread_enabled(
        const ds4_gpu_graph *g) {
    return g &&
           g->ssd_streaming &&
           !glm_graph_env_present("DS4_ROCM_DISABLE_STREAMING_PREFILL_LAYER_PREAD",
                                  "DS4_METAL_DISABLE_STREAMING_PREFILL_LAYER_PREAD") &&
           !glm_graph_env_present("DS4_ROCM_DISABLE_STREAMING_PREFILL_LAYER_PREPARE",
                                  "DS4_METAL_DISABLE_STREAMING_PREFILL_LAYER_PREPARE");
}

static bool metal_graph_stream_prefill_layer_madvise_enabled(
        const ds4_gpu_graph *g) {
    return g &&
           g->ssd_streaming &&
           !glm_graph_env_present("DS4_ROCM_DISABLE_STREAMING_PREFILL_LAYER_PREPARE",
                                  "DS4_METAL_DISABLE_STREAMING_PREFILL_LAYER_PREPARE") &&
           !glm_graph_env_present("DS4_ROCM_DISABLE_STREAMING_PREFILL_LAYER_MADVISE",
                                  "DS4_METAL_DISABLE_STREAMING_PREFILL_LAYER_MADVISE");
}

static uint32_t metal_graph_stream_prefill_batch_selected_addr_auto_max(void) {
    const char *env = glm_graph_env_value(
            "DS4_ROCM_STREAMING_PREFILL_BATCH_SELECTED_ADDR_MAX",
            "DS4_METAL_STREAMING_PREFILL_BATCH_SELECTED_ADDR_MAX");
    if (env && env[0]) {
        char *end = NULL;
        const long v = strtol(env, &end, 10);
        if (end != env) {
            if (v <= 0) return 0;
            if ((unsigned long)v > (unsigned long)UINT32_MAX) return UINT32_MAX;
            return (uint32_t)v;
        }
    }
#ifdef DS4_ROCM_BUILD
    if (DS4_MODEL_VARIANT == DS4_VARIANT_PRO ||
        DS4_MODEL_VARIANT == DS4_VARIANT_FLASH ||
        DS4_MODEL_VARIANT == DS4_VARIANT_GLM52) return UINT32_MAX;
#endif
    if (DS4_MODEL_VARIANT == DS4_VARIANT_PRO) return 800u;
    if (DS4_MODEL_VARIANT == DS4_VARIANT_FLASH) return 760u;
    return 0;
}

static uint32_t metal_graph_stream_prefill_batch_selected_addr_auto_min(void) {
    const char *env = glm_graph_env_value(
            "DS4_ROCM_STREAMING_PREFILL_BATCH_SELECTED_ADDR_MIN",
            "DS4_METAL_STREAMING_PREFILL_BATCH_SELECTED_ADDR_MIN");
    if (env && env[0]) {
        char *end = NULL;
        const long v = strtol(env, &end, 10);
        if (end != env) {
            if (v <= 0) return 0;
            if ((unsigned long)v > (unsigned long)UINT32_MAX) return UINT32_MAX;
            return (uint32_t)v;
        }
    }
#ifdef DS4_ROCM_BUILD
    if (DS4_MODEL_VARIANT == DS4_VARIANT_GLM52) return 2u;
#endif
    if (DS4_MODEL_VARIANT == DS4_VARIANT_PRO ||
        DS4_MODEL_VARIANT == DS4_VARIANT_FLASH) return 2u;
    return 0;
}

static bool metal_graph_stream_prefill_batch_selected_addr_enabled(
        const ds4_gpu_graph *g,
        const ds4_weights   *weights,
        uint32_t             n_tokens) {
    if (!g ||
        !g->ssd_streaming ||
        g->quality ||
        !weights ||
        n_tokens <= 1 ||
        glm_graph_env_present("DS4_ROCM_DISABLE_STREAMING_PREFILL_BATCH_SELECTED_ADDR",
                              "DS4_METAL_DISABLE_STREAMING_PREFILL_BATCH_SELECTED_ADDR") ||
        glm_graph_env_present("DS4_ROCM_DISABLE_STREAMING_EXPERT_ADDR_TABLE",
                              "DS4_METAL_DISABLE_STREAMING_EXPERT_ADDR_TABLE") ||
        glm_graph_env_present("DS4_ROCM_MOE_WRITE_CLAMPED_ACT",
                              "DS4_METAL_MOE_WRITE_CLAMPED_ACT") ||
        glm_graph_env_present("DS4_ROCM_DISABLE_ROUTED_PAIR_SWIGLU_FUSION",
                              "DS4_METAL_DISABLE_ROUTED_PAIR_SWIGLU_FUSION") ||
        DS4_N_LAYER == 0) {
        return false;
    }
    const uint32_t routed_il =
        DS4_N_LEADING_DENSE < DS4_N_LAYER ? DS4_N_LEADING_DENSE : 0u;
    const ds4_layer_weights *layer = &weights->layer[routed_il];
    if (!layer->ffn_gate_exps || !layer->ffn_up_exps || !layer->ffn_down_exps) {
        return false;
    }
#ifdef DS4_ROCM_BUILD
    const bool selected_iq2 =
        glm_stream_selected_expert_cache_supported(layer, routed_il);
    const bool selected_q2 =
        layer->ffn_gate_exps->type == DS4_TENSOR_Q2_K &&
        layer->ffn_up_exps->type == DS4_TENSOR_Q2_K &&
        layer->ffn_down_exps->type == DS4_TENSOR_Q2_K &&
        glm_stream_expert_cache_addr_layout_supported(weights, layer, routed_il);
    if (!selected_iq2 && !selected_q2) return false;
#else
    if (DS4_N_EXPERT_USED != 6 ||
        layer->ffn_gate_exps->type != DS4_TENSOR_IQ2_XXS ||
        layer->ffn_up_exps->type != DS4_TENSOR_IQ2_XXS ||
        layer->ffn_down_exps->type != DS4_TENSOR_Q2_K) {
        return false;
    }
#endif

    const uint32_t cache_configured =
        ds4_gpu_stream_expert_cache_configured_count();
#ifdef DS4_ROCM_BUILD
    if (cache_configured == 0) {
        return false;
    }
#else
    if (cache_configured < DS4_N_EXPERT) {
        return false;
    }
#endif

    if (glm_graph_env_present("DS4_ROCM_ENABLE_STREAMING_PREFILL_BATCH_SELECTED_ADDR",
                              "DS4_METAL_ENABLE_STREAMING_PREFILL_BATCH_SELECTED_ADDR")) {
        return true;
    }

    const uint32_t max_tokens =
        metal_graph_stream_prefill_batch_selected_addr_auto_max();
    const uint32_t min_tokens =
        metal_graph_stream_prefill_batch_selected_addr_auto_min();
    return max_tokens != 0 && n_tokens >= min_tokens && n_tokens <= max_tokens;
}

static bool metal_graph_cuda_stream_prefill_batch_selected_addr_enabled(
        const ds4_gpu_graph *g,
        const ds4_weights   *weights,
        uint32_t             n_tokens) {
#if !defined(DS4_ROCM_BUILD) && !defined(DS4_NO_GPU) && !defined(__APPLE__)
    if (!g ||
        !g->ssd_streaming ||
        g->quality ||
        !weights ||
        n_tokens <= 1 ||
        getenv("DS4_METAL_DISABLE_STREAMING_PREFILL_BATCH_SELECTED_ADDR") != NULL ||
        getenv("DS4_CUDA_DISABLE_STREAMING_PREFILL_BATCH_SELECTED_ADDR") != NULL ||
        getenv("DS4_METAL_DISABLE_STREAMING_EXPERT_ADDR_TABLE") != NULL ||
        getenv("DS4_METAL_MOE_WRITE_CLAMPED_ACT") != NULL ||
        getenv("DS4_METAL_DISABLE_ROUTED_PAIR_SWIGLU_FUSION") != NULL ||
        DS4_N_LAYER == 0 ||
        DS4_N_EXPERT < 128 ||
        DS4_N_EXPERT_USED != 6) {
        return false;
    }

    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const ds4_layer_weights *layer = &weights->layer[il];
        if (!layer->ffn_gate_exps || !layer->ffn_up_exps ||
            !layer->ffn_down_exps) {
            continue;
        }
        const bool q4 =
            layer->ffn_gate_exps->type == DS4_TENSOR_Q4_K &&
            layer->ffn_up_exps->type == DS4_TENSOR_Q4_K &&
            layer->ffn_down_exps->type == DS4_TENSOR_Q4_K;
        const bool iq2 =
            layer->ffn_gate_exps->type == DS4_TENSOR_IQ2_XXS &&
            layer->ffn_up_exps->type == DS4_TENSOR_IQ2_XXS &&
            layer->ffn_down_exps->type == DS4_TENSOR_Q2_K;
        if (q4 || iq2) return true;
    }
    return false;
#else
    (void)g;
    (void)weights;
    (void)n_tokens;
    return false;
#endif
}

#ifdef DS4_ROCM_BUILD
enum { DS4_ROCM_STREAM_PREFILL_FULL_LAYER_MIN_TOKENS = 1024 };
enum { DS4_ROCM_STREAM_PREFILL_FULL_LAYER_MAX_SEED_TOKENS = 8 };

typedef struct rocm_graph_stream_layer_expert_load {
    pthread_t                 thread;
    bool                      active;
    bool                      ok;
    const ds4_model          *model;
    const ds4_layer_weights  *layer;
    uint32_t                  il;
    uint64_t                  gate_expert_bytes;
    uint64_t                  down_expert_bytes;
} rocm_graph_stream_layer_expert_load;

static bool rocm_graph_stream_prefill_full_layer_enabled(
        const ds4_gpu_graph      *g,
        const ds4_layer_weights  *layer,
        uint32_t                  il,
        uint32_t                  n_tokens) {
    return g &&
           g->ssd_streaming &&
           !g->quality &&
           layer &&
           n_tokens >= DS4_ROCM_STREAM_PREFILL_FULL_LAYER_MIN_TOKENS &&
           glm_stream_resident_decode_layer_supported(layer, il);
}

static uint32_t rocm_graph_stream_prefill_full_layer_seed_tokens(void) {
    const uint32_t budget = ds4_gpu_stream_expert_cache_configured_count();
    const uint64_t entries_per_token =
        (uint64_t)DS4_N_LAYER * (uint64_t)DS4_N_EXPERT_USED;
    if (entries_per_token == 0) return 1;
    uint32_t seed_tokens = budget == 0 ? 1 : (uint32_t)(budget / entries_per_token);
    if (seed_tokens < 1) seed_tokens = 1;
    if (seed_tokens > DS4_ROCM_STREAM_PREFILL_FULL_LAYER_MAX_SEED_TOKENS) {
        seed_tokens = DS4_ROCM_STREAM_PREFILL_FULL_LAYER_MAX_SEED_TOKENS;
    }
    return seed_tokens;
}

static bool rocm_graph_stream_layer_expert_bytes(
        const ds4_layer_weights  *layer,
        uint64_t                 *gate_expert_bytes,
        uint64_t                 *down_expert_bytes) {
    return streaming_layer_gate_down_expert_bytes(layer,
                                                  gate_expert_bytes,
                                                  down_expert_bytes);
}

static bool rocm_graph_stream_layer_expert_load_sync(
        const ds4_model          *model,
        const ds4_layer_weights  *layer,
        uint32_t                  il,
        uint64_t                  gate_expert_bytes,
        uint64_t                  down_expert_bytes) {
    const ds4_gpu_stream_expert_table table =
        graph_stream_expert_table_make(model,
                                       layer,
                                       il,
                                       gate_expert_bytes,
                                       down_expert_bytes);
    return model &&
           layer &&
           ds4_gpu_stream_expert_cache_load_layer(&table) != 0;
}

static void *rocm_graph_stream_layer_expert_load_thread_main(void *arg) {
    rocm_graph_stream_layer_expert_load *job = arg;
    if (!job) return NULL;
    job->ok = rocm_graph_stream_layer_expert_load_sync(job->model,
                                                       job->layer,
                                                       job->il,
                                                       job->gate_expert_bytes,
                                                       job->down_expert_bytes);
    return NULL;
}

static bool rocm_graph_stream_layer_expert_load_join(
        rocm_graph_stream_layer_expert_load *job) {
    if (!job || !job->active) return true;
    const int rc = pthread_join(job->thread, NULL);
    const bool ok = rc == 0 && job->ok;
    if (rc != 0) {
        fprintf(stderr,
                "ds4: ROCm streaming full-layer expert load join failed: %s\n",
                strerror(rc));
    }
    memset(job, 0, sizeof(*job));
    return ok;
}

static bool rocm_graph_stream_layer_expert_load_start(
        rocm_graph_stream_layer_expert_load *job,
        const ds4_model                     *model,
        const ds4_layer_weights             *layer,
        uint32_t                             il,
        uint64_t                             gate_expert_bytes,
        uint64_t                             down_expert_bytes) {
    if (!job || job->active || !model || !layer) return false;
    memset(job, 0, sizeof(*job));
    job->model = model;
    job->layer = layer;
    job->il = il;
    job->gate_expert_bytes = gate_expert_bytes;
    job->down_expert_bytes = down_expert_bytes;
    const int rc = pthread_create(&job->thread,
                                  NULL,
                                  rocm_graph_stream_layer_expert_load_thread_main,
                                  job);
    if (rc != 0) {
        fprintf(stderr,
                "ds4: failed to start ROCm streaming full-layer expert load "
                "thread for layer %u: %s\n",
                il,
                strerror(rc));
        memset(job, 0, sizeof(*job));
        return false;
    }
    job->active = true;
    return true;
}

static bool rocm_graph_stream_layer_expert_load_start_next(
        rocm_graph_stream_layer_expert_load *job,
        const ds4_gpu_graph                 *g,
        const ds4_model                     *model,
        const ds4_weights                   *weights,
        uint32_t                             il,
        uint32_t                             n_tokens) {
    if (!job ||
        !model ||
        !weights ||
        il >= DS4_N_LAYER ||
        !rocm_graph_stream_prefill_full_layer_enabled(g,
                                                      &weights->layer[il],
                                                      il,
                                                      n_tokens)) {
        return true;
    }
    uint64_t gate_expert_bytes = 0;
    uint64_t down_expert_bytes = 0;
    if (!rocm_graph_stream_layer_expert_bytes(&weights->layer[il],
                                              &gate_expert_bytes,
                                              &down_expert_bytes)) {
        return false;
    }
    return rocm_graph_stream_layer_expert_load_start(job,
                                                     model,
                                                     &weights->layer[il],
                                                     il,
                                                     gate_expert_bytes,
                                                     down_expert_bytes);
}

static bool rocm_graph_stream_layer_expert_load_ready(
        rocm_graph_stream_layer_expert_load *job,
        const ds4_gpu_graph                 *g,
        const ds4_model                     *model,
        const ds4_weights                   *weights,
        uint32_t                             il,
        uint32_t                             n_tokens) {
    if (!model || !weights || il >= DS4_N_LAYER) return false;
    if (!rocm_graph_stream_prefill_full_layer_enabled(g,
                                                      &weights->layer[il],
                                                      il,
                                                      n_tokens)) {
        return true;
    }
    uint64_t gate_expert_bytes = 0;
    uint64_t down_expert_bytes = 0;
    if (!rocm_graph_stream_layer_expert_bytes(&weights->layer[il],
                                              &gate_expert_bytes,
                                              &down_expert_bytes)) {
        return false;
    }
    if (job && job->active) {
        if (job->il != il) {
            fprintf(stderr,
                    "ds4: ROCm streaming full-layer expert load expected layer "
                    "%u but pending job is layer %u\n",
                    il,
                    job->il);
            return false;
        }
        return rocm_graph_stream_layer_expert_load_join(job);
    }
    return rocm_graph_stream_layer_expert_load_sync(model,
                                                    &weights->layer[il],
                                                    il,
                                                    gate_expert_bytes,
                                                    down_expert_bytes);
}

static bool rocm_graph_stream_seed_full_layer_selected(
        ds4_gpu_graph           *g,
        const ds4_model         *model,
        const ds4_layer_weights *layer,
        uint32_t                 il,
        uint32_t                 n_tokens) {
    if (!rocm_graph_stream_prefill_full_layer_enabled(g, layer, il, n_tokens)) {
        return true;
    }
    uint64_t gate_expert_bytes = 0;
    uint64_t down_expert_bytes = 0;
    if (!rocm_graph_stream_layer_expert_bytes(layer,
                                              &gate_expert_bytes,
                                              &down_expert_bytes)) {
        return false;
    }
    const ds4_gpu_stream_expert_table table =
        graph_stream_expert_table_make(model,
                                       layer,
                                       il,
                                       gate_expert_bytes,
                                       down_expert_bytes);
    if (ds4_gpu_stream_expert_cache_seed_from_layer_selected(
                &table,
                metal_graph_batch_router_selected(g),
                n_tokens,
                rocm_graph_stream_prefill_full_layer_seed_tokens(),
                DS4_N_EXPERT_USED) == 0) {
        static bool warned = false;
        if (!warned) {
            fprintf(stderr,
                    "ds4: ROCm streaming full-layer prefill seed skipped; "
                    "decode may start with a colder expert cache\n");
            warned = true;
        }
    }
    return true;
}
#endif

static bool metal_graph_stream_prefill_selected_profile_enabled(
        const ds4_gpu_graph *g) {
    return g &&
           g->ssd_streaming &&
           glm_graph_env_present("DS4_ROCM_STREAMING_PREFILL_SELECTED_PROFILE",
                                 "DS4_METAL_STREAMING_PREFILL_SELECTED_PROFILE") &&
           !glm_graph_env_present("DS4_ROCM_DISABLE_STREAMING_PREFILL_SELECTED_PROFILE",
                                  "DS4_METAL_DISABLE_STREAMING_PREFILL_SELECTED_PROFILE");
}

static void metal_graph_stream_prefill_selected_profile_reset(
        ds4_gpu_graph *g) {
    if (!g) return;
    g->prefill_selected_profile_rows = 0;
    g->prefill_selected_profile_unique = 0;
    g->prefill_selected_profile_selected_bytes = 0;
    g->prefill_selected_profile_full_bytes = 0;
    g->prefill_selected_profile_layers = 0;
    g->prefill_selected_profile_min_unique = UINT32_MAX;
    g->prefill_selected_profile_max_unique = 0;
}

static uint64_t metal_graph_stream_prefill_selected_profile_add_bytes(
        uint64_t a,
        uint64_t b) {
    return a > UINT64_MAX - b ? UINT64_MAX : a + b;
}

static bool metal_graph_selected_profile_layer_impl(
        ds4_gpu_graph          *g,
        const ds4_layer_weights *layer,
        uint32_t                il,
        uint32_t                n_tokens,
        const char             *label) {
    if (!layer || !metal_graph_batch_router_selected(g) || n_tokens == 0 ||
        DS4_N_EXPERT == 0 || DS4_N_EXPERT > DS4_MAX_EXPERT ||
        DS4_N_EXPERT_USED == 0 || DS4_N_EXPERT_USED > DS4_MAX_EXPERT_USED) {
        return false;
    }

    const uint64_t n_ids = (uint64_t)n_tokens * DS4_N_EXPERT_USED;
    if (n_ids > SIZE_MAX / sizeof(int32_t)) return false;
    int32_t *selected = xmalloc((size_t)n_ids * sizeof(selected[0]));
    const bool read_ok = ds4_gpu_tensor_read(metal_graph_batch_router_selected(g),
                                             0,
                                             selected,
                                             n_ids * sizeof(selected[0])) != 0;
    if (!read_ok) {
        free(selected);
        return false;
    }

    bool seen[DS4_MAX_EXPERT] = { false };
    uint32_t unique = 0;
    for (uint64_t i = 0; i < n_ids; i++) {
        const int32_t expert = selected[i];
        if (expert < 0 || (uint32_t)expert >= DS4_N_EXPERT) {
            fprintf(stderr,
                    "ds4: Metal streaming prefill selected profile expert id %d is outside 0..%u at layer %u\n",
                    expert,
                    (uint32_t)DS4_N_EXPERT,
                    il);
            free(selected);
            return false;
        }
        if (!seen[expert]) {
            seen[expert] = true;
            unique++;
        }
    }
    free(selected);

    const uint64_t gate_row_bytes = routed_expert_row_bytes(layer->ffn_gate_exps);
    const uint64_t down_row_bytes = routed_expert_row_bytes(layer->ffn_down_exps);
    if (layer->ffn_gate_exps->dim[1] > UINT64_MAX / gate_row_bytes ||
        layer->ffn_down_exps->dim[1] > UINT64_MAX / down_row_bytes) {
        fprintf(stderr, "ds4: Metal streaming prefill selected profile byte size overflow at layer %u\n", il);
        return false;
    }
    const uint64_t gate_expert_bytes = layer->ffn_gate_exps->dim[1] * gate_row_bytes;
    const uint64_t down_expert_bytes = layer->ffn_down_exps->dim[1] * down_row_bytes;
    if (gate_expert_bytes > UINT64_MAX - gate_expert_bytes ||
        gate_expert_bytes + gate_expert_bytes > UINT64_MAX - down_expert_bytes) {
        fprintf(stderr, "ds4: Metal streaming prefill selected profile byte size overflow at layer %u\n", il);
        return false;
    }
    const uint64_t per_expert_bytes = gate_expert_bytes + gate_expert_bytes +
                                      down_expert_bytes;
    const uint64_t selected_bytes =
        unique > UINT64_MAX / per_expert_bytes ?
        UINT64_MAX : (uint64_t)unique * per_expert_bytes;
    const uint64_t full_bytes =
        (uint64_t)DS4_N_EXPERT > UINT64_MAX / per_expert_bytes ?
        UINT64_MAX : (uint64_t)DS4_N_EXPERT * per_expert_bytes;
    const double ratio = full_bytes == 0 ? 0.0 :
        (double)selected_bytes / (double)full_bytes;

    g->prefill_selected_profile_layers++;
    g->prefill_selected_profile_rows =
        metal_graph_stream_prefill_selected_profile_add_bytes(
                g->prefill_selected_profile_rows,
                n_ids);
    g->prefill_selected_profile_unique =
        metal_graph_stream_prefill_selected_profile_add_bytes(
                g->prefill_selected_profile_unique,
                unique);
    g->prefill_selected_profile_selected_bytes =
        metal_graph_stream_prefill_selected_profile_add_bytes(
                g->prefill_selected_profile_selected_bytes,
                selected_bytes);
    g->prefill_selected_profile_full_bytes =
        metal_graph_stream_prefill_selected_profile_add_bytes(
                g->prefill_selected_profile_full_bytes,
                full_bytes);
    if (unique < g->prefill_selected_profile_min_unique) {
        g->prefill_selected_profile_min_unique = unique;
    }
    if (unique > g->prefill_selected_profile_max_unique) {
        g->prefill_selected_profile_max_unique = unique;
    }

    fprintf(stderr,
            "ds4: %s layer=%u "
            "tokens=%u unique=%u/%u selected=%.2f GiB full=%.2f GiB ratio=%.3f\n",
            label ? label : "selected expert profile",
            il,
            n_tokens,
            unique,
            (uint32_t)DS4_N_EXPERT,
            (double)selected_bytes / (1024.0 * 1024.0 * 1024.0),
            (double)full_bytes / (1024.0 * 1024.0 * 1024.0),
            ratio);
    return true;
}

static bool metal_graph_stream_prefill_selected_profile_layer(
        ds4_gpu_graph          *g,
        const ds4_layer_weights *layer,
        uint32_t                il,
        uint32_t                n_tokens) {
    if (!metal_graph_stream_prefill_selected_profile_enabled(g)) return true;
    return metal_graph_selected_profile_layer_impl(
            g,
            layer,
            il,
            n_tokens,
            "Metal streaming prefill selected profile");
}

static void metal_graph_selected_profile_summary_impl(
        const ds4_gpu_graph *g,
        const char          *label) {
    if (!g || g->prefill_selected_profile_layers == 0) {
        return;
    }
    const double layers = (double)g->prefill_selected_profile_layers;
    const double avg_unique = (double)g->prefill_selected_profile_unique / layers;
    const double ratio = g->prefill_selected_profile_full_bytes == 0 ? 0.0 :
        (double)g->prefill_selected_profile_selected_bytes /
        (double)g->prefill_selected_profile_full_bytes;
    fprintf(stderr,
            "ds4: %s summary "
            "layers=%u avg_unique=%.1f min_unique=%u max_unique=%u "
            "selected=%.2f GiB full=%.2f GiB ratio=%.3f rows=%" PRIu64 "\n",
            label ? label : "selected expert profile",
            g->prefill_selected_profile_layers,
            avg_unique,
            g->prefill_selected_profile_min_unique == UINT32_MAX ?
                0 : g->prefill_selected_profile_min_unique,
            g->prefill_selected_profile_max_unique,
            (double)g->prefill_selected_profile_selected_bytes /
                (1024.0 * 1024.0 * 1024.0),
            (double)g->prefill_selected_profile_full_bytes /
                (1024.0 * 1024.0 * 1024.0),
            ratio,
            g->prefill_selected_profile_rows);
}

static void metal_graph_stream_prefill_selected_profile_summary(
        const ds4_gpu_graph *g) {
    if (!metal_graph_stream_prefill_selected_profile_enabled(g)) return;
    metal_graph_selected_profile_summary_impl(
            g,
            "Metal streaming prefill selected profile");
}

static bool metal_graph_stream_pagein_touch_range(
        const ds4_model *model,
        uint64_t         offset,
        uint64_t         size,
        uint64_t        *touched,
        uint8_t         *sink) {
    if (!model ||
        !model->map ||
        model->size == 0 ||
        offset > model->size ||
        size == 0 ||
        size > model->size - offset) {
        return false;
    }

    const uint64_t page = (uint64_t)getpagesize();
    const uint64_t page_offset = offset & ~(page - 1u);
    const uint64_t leading = offset - page_offset;
    if (size > UINT64_MAX - leading ||
        leading + size > UINT64_MAX - (page - 1u)) {
        return false;
    }
    uint64_t touch_bytes = align_up(leading + size, page);
    if (touch_bytes > model->size - page_offset) {
        touch_bytes = model->size - page_offset;
    }
    if (touch_bytes == 0 || touch_bytes > (uint64_t)SIZE_MAX) {
        return false;
    }

    const uint8_t *base = (const uint8_t *)model->map;
    const volatile uint8_t *p =
        (const volatile uint8_t *)(base + page_offset);

#if defined(POSIX_MADV_WILLNEED)
    (void)posix_madvise((void *)(base + page_offset),
                        (size_t)touch_bytes,
                        POSIX_MADV_WILLNEED);
#endif

    uint8_t s = sink ? *sink : 0;
    for (uint64_t off = 0; off < touch_bytes; off += page) {
        s ^= p[off];
    }
    s ^= p[touch_bytes - 1u];
    if (sink) *sink = s;
    if (touched) *touched += touch_bytes;
    return true;
}

static bool metal_graph_stream_pread_range(
        const ds4_model *model,
        uint64_t         offset,
        uint64_t         size,
        uint64_t        *read_bytes,
        uint8_t         *sink) {
    if (!model ||
        model->fd < 0 ||
        offset > model->size ||
        size == 0 ||
        size > model->size - offset) {
        return false;
    }
    if (offset > (uint64_t)LLONG_MAX) return false;

    const size_t chunk = 1024u * 1024u;
    uint8_t *buf = xmalloc(chunk);
    uint64_t pos = offset;
    uint64_t rem = size;
    uint8_t s = sink ? *sink : 0;
    bool ok = true;
    while (rem != 0) {
        const size_t want = rem > (uint64_t)chunk ? chunk : (size_t)rem;
        ssize_t nread;
        do {
            nread = pread(model->fd, buf, want, (off_t)pos);
        } while (nread < 0 && errno == EINTR);
        if (nread <= 0) {
            ok = false;
            break;
        }
        s ^= buf[0];
        s ^= buf[(size_t)nread - 1u];
        pos += (uint64_t)nread;
        rem -= (uint64_t)nread;
        if (read_bytes) {
            *read_bytes = *read_bytes > UINT64_MAX - (uint64_t)nread ?
                UINT64_MAX : *read_bytes + (uint64_t)nread;
        }
    }
    if (sink) *sink = s;
    free(buf);
    return ok;
}

static bool metal_graph_stream_prepare_range(
        const metal_graph_stream_pagein_job *job,
        uint64_t                             offset,
        uint64_t                             size,
        uint64_t                            *touched,
        uint8_t                             *sink) {
    if (!job) return false;
    if (job->pread_only) {
        return metal_graph_stream_pread_range(job->model,
                                              offset,
                                              size,
                                              touched,
                                              sink);
    }
    if (job->readahead_only) {
        metal_graph_stream_readahead_range_impl(job->model,
                                                offset,
                                                size,
                                                true);
        if (touched) {
            *touched = *touched > UINT64_MAX - size ?
                UINT64_MAX : *touched + size;
        }
        return true;
    }
    if (job->madvise_only) {
        return metal_graph_stream_madvise_willneed_range_impl(job->model,
                                                              offset,
                                                              size,
                                                              true,
                                                              touched);
    }
    return metal_graph_stream_pagein_touch_range(job->model,
                                                offset,
                                                size,
                                                touched,
                                                sink);
}

static void *metal_graph_stream_pagein_thread_main(void *arg) {
    metal_graph_stream_pagein_job *job = arg;
    const double t0 = job->profile ? now_sec() : 0.0;
    job->ok = true;
    for (uint32_t i = 0; i < job->n_ranges; i++) {
        const bool ok = metal_graph_stream_prepare_range(job,
                                                         job->ranges[i].off,
                                                         job->ranges[i].size,
                                                         &job->touched,
                                                         &job->sink);
        if (!ok) {
            job->ok = false;
            break;
        }
    }
    if (job->profile) {
        job->thread_ms = (now_sec() - t0) * 1000.0;
    }
    return NULL;
}

static void *metal_graph_stream_pagein_worker_main(void *arg) {
    metal_graph_stream_pagein_worker *worker = arg;
    metal_graph_stream_pagein_job *job = worker ? worker->job : NULL;
    const double t0 = job && job->profile ? now_sec() : 0.0;
    worker->ok = true;
    if (!job || worker->stride == 0) {
        worker->ok = false;
        return NULL;
    }
    for (uint32_t i = worker->first; i < job->n_ranges; i += worker->stride) {
        const bool ok = metal_graph_stream_prepare_range(job,
                                                         job->ranges[i].off,
                                                         job->ranges[i].size,
                                                         &worker->touched,
                                                         &worker->sink);
        if (!ok) {
            worker->ok = false;
            break;
        }
    }
    if (job->profile) {
        worker->thread_ms = (now_sec() - t0) * 1000.0;
    }
    return NULL;
}

static uint32_t metal_graph_stream_prefill_layer_pagein_threads(void) {
    const char *env = glm_graph_env_value(
            "DS4_ROCM_STREAMING_PREFILL_LAYER_PREPARE_THREADS",
            "DS4_METAL_STREAMING_PREFILL_LAYER_PREPARE_THREADS");
    if (!env || !env[0]) {
        env = glm_graph_env_value(
                "DS4_ROCM_STREAMING_PREFILL_LAYER_PAGEIN_THREADS",
                "DS4_METAL_STREAMING_PREFILL_LAYER_PAGEIN_THREADS");
    }
    if (!env || !env[0]) return 8;
    char *end = NULL;
    unsigned long v = strtoul(env, &end, 10);
    if (end == env || *end != '\0' || v == 0) return 1;
    return v > 16 ? 16u : (uint32_t)v;
}

static uint32_t metal_graph_stream_prefill_selected_prepare_threads(
        bool madvise_only) {
    if (!madvise_only) return 1;
    const char *env = glm_graph_env_value(
            "DS4_ROCM_STREAMING_PREFILL_SELECTED_PREPARE_THREADS",
            "DS4_METAL_STREAMING_PREFILL_SELECTED_PREPARE_THREADS");
    if (!env || !env[0]) {
        env = glm_graph_env_value(
                "DS4_ROCM_STREAMING_PREFILL_SELECTED_MADVISE_THREADS",
                "DS4_METAL_STREAMING_PREFILL_SELECTED_MADVISE_THREADS");
    }
    if (!env || !env[0]) return metal_graph_stream_prefill_layer_pagein_threads();
    char *end = NULL;
    unsigned long v = strtoul(env, &end, 10);
    if (end == env || *end != '\0' || v == 0) return 1;
    return v > 16 ? 16u : (uint32_t)v;
}

static uint32_t metal_graph_stream_prefill_selected_prepare_gap(void) {
    const char *env = glm_graph_env_value(
            "DS4_ROCM_STREAMING_PREFILL_SELECTED_PREPARE_GAP",
            "DS4_METAL_STREAMING_PREFILL_SELECTED_PREPARE_GAP");
    if (!env || !env[0]) return 0;
    char *end = NULL;
    unsigned long v = strtoul(env, &end, 10);
    if (end == env || *end != '\0') return 0;
    return v > 8 ? 8u : (uint32_t)v;
}

static bool metal_graph_stream_prefill_layer_pagein_overlap_enabled(void) {
    return !glm_graph_env_present("DS4_ROCM_STREAMING_PREFILL_LAYER_PREPARE_NO_OVERLAP",
                                  "DS4_METAL_STREAMING_PREFILL_LAYER_PREPARE_NO_OVERLAP") &&
           !glm_graph_env_present("DS4_ROCM_STREAMING_PREFILL_LAYER_PAGEIN_NO_OVERLAP",
                                  "DS4_METAL_STREAMING_PREFILL_LAYER_PAGEIN_NO_OVERLAP") &&
           !glm_graph_env_present("DS4_ROCM_DISABLE_STREAMING_PREFILL_LAYER_PREPARE_OVERLAP",
                                  "DS4_METAL_DISABLE_STREAMING_PREFILL_LAYER_PREPARE_OVERLAP") &&
           !glm_graph_env_present("DS4_ROCM_DISABLE_STREAMING_PREFILL_LAYER_PAGEIN_OVERLAP",
                                  "DS4_METAL_DISABLE_STREAMING_PREFILL_LAYER_PAGEIN_OVERLAP");
}

enum { DS4_STREAM_PREFILL_MAX_PREPARE_AHEAD = 4 };

static uint32_t metal_graph_stream_prefill_layer_prepare_ahead(void) {
    const char *env = glm_graph_env_value(
            "DS4_ROCM_STREAMING_PREFILL_LAYER_PREPARE_AHEAD",
            "DS4_METAL_STREAMING_PREFILL_LAYER_PREPARE_AHEAD");
    if (!env || !env[0]) return 1;
    char *end = NULL;
    unsigned long v = strtoul(env, &end, 10);
    if (end == env || *end != '\0' || v == 0) return 1;
    if (v > DS4_STREAM_PREFILL_MAX_PREPARE_AHEAD) {
        return DS4_STREAM_PREFILL_MAX_PREPARE_AHEAD;
    }
    return (uint32_t)v;
}

static bool metal_graph_stream_prefill_selected_pagein_start(
        ds4_gpu_graph                 *g,
        const ds4_model               *model,
        const ds4_layer_weights       *layer,
        uint32_t                       il,
        uint32_t                       n_tokens,
        uint64_t                       gate_expert_bytes,
        uint64_t                       down_expert_bytes,
        metal_graph_stream_pagein_job *job) {
    if (!job) return false;
    memset(job, 0, sizeof(*job));
    job->ok = true;
    job->profile =
        glm_graph_env_present("DS4_ROCM_STREAMING_PREFILL_SELECTED_PAGEIN_PROFILE",
                              "DS4_METAL_STREAMING_PREFILL_SELECTED_PAGEIN_PROFILE") ||
        glm_graph_env_present("DS4_ROCM_STREAMING_PREFILL_SELECTED_MADVISE_PROFILE",
                              "DS4_METAL_STREAMING_PREFILL_SELECTED_MADVISE_PROFILE");
    job->layer = il;
    job->n_tokens = n_tokens;

    const bool madvise_only =
        metal_graph_stream_prefill_selected_madvise_enabled(g);
    job->madvise_only = madvise_only;
    if (!metal_graph_stream_prefill_selected_pagein_enabled(g) &&
        !madvise_only) return true;
    if (!model || !layer || !metal_graph_batch_router_selected(g) || n_tokens == 0) {
        return false;
    }

    const uint64_t n_ids = (uint64_t)n_tokens * DS4_N_EXPERT_USED;
    if (n_ids > SIZE_MAX / sizeof(int32_t)) return false;
    int32_t *selected = xmalloc((size_t)n_ids * sizeof(selected[0]));

    const double t_read0 = job->profile ? now_sec() : 0.0;
    bool ok = ds4_gpu_tensor_read(metal_graph_batch_router_selected(g),
                                  0,
                                  selected,
                                  n_ids * sizeof(selected[0])) != 0;
    if (job->profile) {
        job->read_ms = (now_sec() - t_read0) * 1000.0;
    }

    bool seen[DS4_MAX_EXPERT] = { false };
    if (ok) {
        for (uint64_t i = 0; i < n_ids; i++) {
            const int32_t expert = selected[i];
            if (expert < 0 || (uint32_t)expert >= DS4_N_EXPERT) {
                fprintf(stderr,
                        "ds4: Metal streaming prefill selected page-in expert id %d is outside 0..%u at layer %u\n",
                        expert,
                        (uint32_t)DS4_N_EXPERT,
                        il);
                ok = false;
                break;
            }
            if (seen[expert]) continue;
            seen[expert] = true;
            job->unique++;
        }
    }
    free(selected);

    metal_graph_stream_pagein_range *ranges = NULL;
    uint32_t n_ranges = 0;
    if (ok && job->unique != 0) {
        ranges = xmalloc((size_t)DS4_N_EXPERT * 3u * sizeof(ranges[0]));
        const uint32_t gap = madvise_only ?
            metal_graph_stream_prefill_selected_prepare_gap() : 0;
        uint32_t e = 0;
        while (e < DS4_N_EXPERT) {
            while (e < DS4_N_EXPERT && !seen[e]) e++;
            if (e >= DS4_N_EXPERT) break;
            const uint32_t first = e;
            uint32_t last = e;
            uint32_t skipped = 0;
            e++;
            while (e < DS4_N_EXPERT) {
                if (seen[e]) {
                    last = e;
                    skipped = 0;
                } else if (skipped < gap) {
                    skipped++;
                } else {
                    break;
                }
                e++;
            }

            const uint64_t first_id = first;
            const uint64_t n_experts = (uint64_t)last - (uint64_t)first + 1u;
            if (first_id > UINT64_MAX / gate_expert_bytes ||
                first_id > UINT64_MAX / down_expert_bytes ||
                n_experts > UINT64_MAX / gate_expert_bytes ||
                n_experts > UINT64_MAX / down_expert_bytes) {
                fprintf(stderr, "ds4: Metal streaming prefill selected page-in offset overflow\n");
                ok = false;
                break;
            }
            const uint64_t gate_rel = first_id * gate_expert_bytes;
            const uint64_t down_rel = first_id * down_expert_bytes;
            const uint64_t gate_bytes = n_experts * gate_expert_bytes;
            const uint64_t down_bytes = n_experts * down_expert_bytes;
            if (gate_rel > UINT64_MAX - layer->ffn_gate_exps->abs_offset ||
                gate_rel > UINT64_MAX - layer->ffn_up_exps->abs_offset ||
                down_rel > UINT64_MAX - layer->ffn_down_exps->abs_offset) {
                fprintf(stderr, "ds4: Metal streaming prefill selected page-in offset overflow\n");
                ok = false;
                break;
            }
            ranges[n_ranges++] = (metal_graph_stream_pagein_range){
                layer->ffn_gate_exps->abs_offset + gate_rel,
                gate_bytes,
            };
            ranges[n_ranges++] = (metal_graph_stream_pagein_range){
                layer->ffn_up_exps->abs_offset + gate_rel,
                gate_bytes,
            };
            ranges[n_ranges++] = (metal_graph_stream_pagein_range){
                layer->ffn_down_exps->abs_offset + down_rel,
                down_bytes,
            };
            uint64_t run_bytes = UINT64_MAX;
            if (gate_bytes <= (UINT64_MAX - down_bytes) / 2ull) {
                run_bytes = gate_bytes * 2ull + down_bytes;
            }
            if (run_bytes == UINT64_MAX ||
                job->bytes > UINT64_MAX - run_bytes) {
                job->bytes = UINT64_MAX;
            } else {
                job->bytes += run_bytes;
            }
        }
    }

    if (!ok || n_ranges == 0) {
        free(ranges);
        return ok;
    }

    job->model = model;
    job->ranges = ranges;
    job->n_ranges = n_ranges;
    job->n_threads = metal_graph_stream_prefill_selected_prepare_threads(madvise_only);
    if (job->n_threads <= 1) {
        const int rc = pthread_create(&job->thread,
                                      NULL,
                                      metal_graph_stream_pagein_thread_main,
                                      job);
        if (rc != 0) {
            fprintf(stderr,
                    "ds4: Metal streaming prefill selected page-in thread failed: %s\n",
                    strerror(rc));
            free(ranges);
            memset(job, 0, sizeof(*job));
            return false;
        }
    } else {
        job->threads = xcalloc(job->n_threads, sizeof(job->threads[0]));
        job->workers = xcalloc(job->n_threads, sizeof(job->workers[0]));
        for (uint32_t t = 0; t < job->n_threads; t++) {
            job->workers[t].job = job;
            job->workers[t].first = t;
            job->workers[t].stride = job->n_threads;
            const int rc = pthread_create(&job->threads[t],
                                          NULL,
                                          metal_graph_stream_pagein_worker_main,
                                          &job->workers[t]);
            if (rc != 0) {
                fprintf(stderr,
                        "ds4: Metal streaming prefill selected page-in worker failed: %s\n",
                        strerror(rc));
                for (uint32_t j = 0; j < t; j++) {
                    (void)pthread_join(job->threads[j], NULL);
                }
                free(job->workers);
                free(job->threads);
                free(ranges);
                memset(job, 0, sizeof(*job));
                return false;
            }
        }
    }
    job->started = true;
    return true;
}

static bool metal_graph_stream_prefill_selected_pagein_join(
        metal_graph_stream_pagein_job *job) {
    if (!job || !job->started) return true;
    const double t0 = job->profile ? now_sec() : 0.0;
    int rc = 0;
    bool ok = true;
    if (job->n_threads <= 1) {
        rc = pthread_join(job->thread, NULL);
        ok = rc == 0 && job->ok;
    } else {
        job->touched = 0;
        job->thread_ms = 0.0;
        job->sink = 0;
        for (uint32_t t = 0; t < job->n_threads; t++) {
            const int trc = pthread_join(job->threads[t], NULL);
            if (trc != 0 && rc == 0) rc = trc;
            if (trc != 0 || !job->workers[t].ok) ok = false;
            if (job->touched > UINT64_MAX - job->workers[t].touched) {
                job->touched = UINT64_MAX;
            } else {
                job->touched += job->workers[t].touched;
            }
            if (job->workers[t].thread_ms > job->thread_ms) {
                job->thread_ms = job->workers[t].thread_ms;
            }
            job->sink ^= job->workers[t].sink;
        }
    }
    const double wait_ms = job->profile ? (now_sec() - t0) * 1000.0 : 0.0;
    if (job->profile) {
        const char *kind = job->madvise_only ? "madvise" : "page-in";
        const char *bytes_label = job->madvise_only ? "advised" : "touched";
        fprintf(stderr,
                "ds4: Metal streaming prefill selected %s layer=%u "
                "tokens=%u unique=%u ranges=%u bytes=%.2f GiB "
                "read=%.3f ms wait=%.3f ms thread=%.3f ms %s=%.2f GiB ok=%d\n",
                kind,
                job->layer,
                job->n_tokens,
                job->unique,
                job->n_ranges,
                (double)job->bytes / (1024.0 * 1024.0 * 1024.0),
                job->read_ms,
                wait_ms,
                job->thread_ms,
                bytes_label,
                (double)job->touched / (1024.0 * 1024.0 * 1024.0),
                ok ? 1 : 0);
    }
    if (rc != 0) {
        fprintf(stderr,
                "ds4: Metal streaming prefill selected page-in join failed: %s\n",
                strerror(rc));
    }
    free(job->workers);
    free(job->threads);
    free(job->ranges);
    memset(job, 0, sizeof(*job));
    return ok;
}

static bool metal_graph_stream_prefill_layer_pagein_start(
        const ds4_gpu_graph           *g,
        const ds4_model               *model,
        const ds4_weights             *weights,
        uint32_t                       il,
        uint32_t                       n_tokens,
        bool                           madvise_only,
        bool                           pread_only,
        bool                           readahead_only,
        bool                           decode_only,
        metal_graph_stream_pagein_job *job) {
    if (!job) return false;
    memset(job, 0, sizeof(*job));
    job->ok = true;
    job->madvise_only = madvise_only;
    job->pread_only = pread_only;
    job->readahead_only = readahead_only;
    job->profile =
        glm_graph_env_present("DS4_ROCM_STREAMING_PREFILL_LAYER_PAGEIN_PROFILE",
                              "DS4_METAL_STREAMING_PREFILL_LAYER_PAGEIN_PROFILE") ||
        glm_graph_env_present("DS4_ROCM_STREAMING_PREFILL_LAYER_PREAD_PROFILE",
                              "DS4_METAL_STREAMING_PREFILL_LAYER_PREAD_PROFILE") ||
        glm_graph_env_present("DS4_ROCM_STREAMING_PREFILL_LAYER_MADVISE_PROFILE",
                              "DS4_METAL_STREAMING_PREFILL_LAYER_MADVISE_PROFILE") ||
        glm_graph_env_present("DS4_ROCM_STREAMING_PREFILL_LAYER_READAHEAD_PROFILE",
                              "DS4_METAL_STREAMING_PREFILL_LAYER_READAHEAD_PROFILE");
    job->layer = il;
    job->n_tokens = n_tokens;

    if (pread_only) {
        if (g) {
            if (!metal_graph_stream_prefill_layer_pread_enabled(g)) return true;
        } else if (glm_graph_env_present("DS4_ROCM_DISABLE_STREAMING_PREFILL_LAYER_PREAD",
                                         "DS4_METAL_DISABLE_STREAMING_PREFILL_LAYER_PREAD") ||
                   glm_graph_env_present("DS4_ROCM_DISABLE_STREAMING_PREFILL_LAYER_PREPARE",
                                         "DS4_METAL_DISABLE_STREAMING_PREFILL_LAYER_PREPARE")) {
            return true;
        }
    } else if (readahead_only) {
        if (g) {
            if (!metal_graph_stream_prefill_layer_readahead_enabled(g)) return true;
        } else if (!glm_graph_env_present("DS4_ROCM_ENABLE_STREAMING_PREFILL_LAYER_READAHEAD",
                                          "DS4_METAL_ENABLE_STREAMING_PREFILL_LAYER_READAHEAD") ||
                   glm_graph_env_present("DS4_ROCM_DISABLE_STREAMING_PREFILL_LAYER_READAHEAD",
                                         "DS4_METAL_DISABLE_STREAMING_PREFILL_LAYER_READAHEAD") ||
                   glm_graph_env_present("DS4_ROCM_DISABLE_STREAMING_PREFILL_LAYER_PREPARE",
                                         "DS4_METAL_DISABLE_STREAMING_PREFILL_LAYER_PREPARE")) {
            return true;
        }
    } else if (madvise_only) {
        if (g) {
            if (!metal_graph_stream_prefill_layer_madvise_enabled(g)) return true;
        } else if (glm_graph_env_present("DS4_ROCM_DISABLE_STREAMING_PREFILL_LAYER_PREPARE",
                                         "DS4_METAL_DISABLE_STREAMING_PREFILL_LAYER_PREPARE") ||
                   glm_graph_env_present("DS4_ROCM_DISABLE_STREAMING_PREFILL_LAYER_MADVISE",
                                         "DS4_METAL_DISABLE_STREAMING_PREFILL_LAYER_MADVISE")) {
            return true;
        }
    } else {
        if (g) {
            if (!metal_graph_stream_prefill_layer_pagein_enabled(g)) return true;
        } else if (!glm_graph_env_present("DS4_ROCM_ENABLE_STREAMING_PREFILL_LAYER_PAGEIN",
                                          "DS4_METAL_ENABLE_STREAMING_PREFILL_LAYER_PAGEIN") ||
                   glm_graph_env_present("DS4_ROCM_DISABLE_STREAMING_PREFILL_LAYER_PAGEIN",
                                         "DS4_METAL_DISABLE_STREAMING_PREFILL_LAYER_PAGEIN")) {
            return true;
        }
    }
    if (!model || !weights || il >= DS4_N_LAYER) return false;

    const uint32_t n_threads =
        metal_graph_stream_prefill_layer_pagein_threads();
    ds4_model_map_span_vec spans;
    const bool spans_ok = decode_only ?
        weights_model_map_decode_layer_spans(weights, il, &spans) :
        weights_model_map_spans(weights, il, il, false, &spans);
    if (!spans_ok) return false;
    metal_graph_stream_pagein_range *ranges =
        xmalloc((size_t)spans.len * n_threads * sizeof(ranges[0]));
    uint32_t n_ranges = 0;
    const uint64_t page = (uint64_t)getpagesize();
    for (uint32_t i = 0; i < spans.len; i++) {
        const uint64_t size = spans.v[i].end - spans.v[i].off;
        uint64_t consumed = 0;
        uint64_t chunk = size / n_threads;
        if (chunk > page) chunk = (chunk / page) * page;
        if (chunk == 0) chunk = size;
        for (uint32_t t = 0; t < n_threads && consumed < size; t++) {
            uint64_t this_size =
                (t + 1u == n_threads || size - consumed <= chunk) ?
                size - consumed : chunk;
            ranges[n_ranges++] = (metal_graph_stream_pagein_range){
                spans.v[i].off + consumed,
                this_size,
            };
            consumed += this_size;
        }
        if (job->bytes > UINT64_MAX - size) {
            job->bytes = UINT64_MAX;
        } else {
            job->bytes += size;
        }
    }
    job->unique = spans.len;
    free(spans.v);

    job->model = model;
    job->ranges = ranges;
    job->n_ranges = n_ranges;
    job->n_threads = n_threads;
    if (n_threads == 1) {
        const int rc = pthread_create(&job->thread,
                                      NULL,
                                      metal_graph_stream_pagein_thread_main,
                                      job);
        if (rc != 0) {
            fprintf(stderr,
                    "ds4: Metal streaming prefill layer page-in thread failed: %s\n",
                    strerror(rc));
            free(ranges);
            memset(job, 0, sizeof(*job));
            return false;
        }
    } else {
        job->threads = xcalloc(n_threads, sizeof(job->threads[0]));
        job->workers = xcalloc(n_threads, sizeof(job->workers[0]));
        for (uint32_t t = 0; t < n_threads; t++) {
            job->workers[t].job = job;
            job->workers[t].first = t;
            job->workers[t].stride = n_threads;
            const int rc = pthread_create(&job->threads[t],
                                          NULL,
                                          metal_graph_stream_pagein_worker_main,
                                          &job->workers[t]);
            if (rc != 0) {
                fprintf(stderr,
                        "ds4: Metal streaming prefill layer page-in worker failed: %s\n",
                        strerror(rc));
                for (uint32_t j = 0; j < t; j++) {
                    (void)pthread_join(job->threads[j], NULL);
                }
                free(job->workers);
                free(job->threads);
                free(ranges);
                memset(job, 0, sizeof(*job));
                return false;
            }
        }
    }
    job->started = true;
    return true;
}

static bool metal_graph_stream_prefill_layer_pagein_join(
        metal_graph_stream_pagein_job *job) {
    if (!job || !job->started) return true;
    const double t0 = job->profile ? now_sec() : 0.0;
    int rc = 0;
    bool ok = true;
    if (job->n_threads <= 1) {
        rc = pthread_join(job->thread, NULL);
        ok = rc == 0 && job->ok;
    } else {
        job->touched = 0;
        job->thread_ms = 0.0;
        job->sink = 0;
        for (uint32_t t = 0; t < job->n_threads; t++) {
            const int trc = pthread_join(job->threads[t], NULL);
            if (trc != 0 && rc == 0) rc = trc;
            if (trc != 0 || !job->workers[t].ok) ok = false;
            if (job->touched > UINT64_MAX - job->workers[t].touched) {
                job->touched = UINT64_MAX;
            } else {
                job->touched += job->workers[t].touched;
            }
            if (job->workers[t].thread_ms > job->thread_ms) {
                job->thread_ms = job->workers[t].thread_ms;
            }
            job->sink ^= job->workers[t].sink;
        }
    }
    const double wait_ms = job->profile ? (now_sec() - t0) * 1000.0 : 0.0;
    if (job->profile) {
        const char *kind = job->pread_only ? "pread" :
                           job->readahead_only ? "readahead" :
                           job->madvise_only ? "madvise" : "page-in";
        const char *bytes_label = job->pread_only ? "read" :
                                  job->readahead_only ? "requested" :
                                  job->madvise_only ? "advised" : "touched";
        fprintf(stderr,
                "ds4: Metal streaming prefill layer %s layer=%u "
                "tokens=%u threads=%u ranges=%u bytes=%.2f GiB wait=%.3f ms "
                "thread=%.3f ms %s=%.2f GiB ok=%d\n",
                kind,
                job->layer,
                job->n_tokens,
                job->n_threads ? job->n_threads : 1u,
                job->n_ranges,
                (double)job->bytes / (1024.0 * 1024.0 * 1024.0),
                wait_ms,
                job->thread_ms,
                bytes_label,
                (double)job->touched / (1024.0 * 1024.0 * 1024.0),
                ok ? 1 : 0);
    }
    if (rc != 0) {
        fprintf(stderr,
                "ds4: Metal streaming prefill layer page-in join failed: %s\n",
                strerror(rc));
    }
    free(job->workers);
    free(job->threads);
    free(job->ranges);
    memset(job, 0, sizeof(*job));
    return ok;
}

typedef struct {
    metal_graph_stream_pagein_job job;
    uint32_t layer;
    bool active;
} metal_graph_stream_prepare_slot;

static metal_graph_stream_prepare_slot *metal_graph_stream_prepare_slot_find(
        metal_graph_stream_prepare_slot *slots,
        uint32_t                         n_slots,
        uint32_t                         layer) {
    for (uint32_t i = 0; i < n_slots; i++) {
        if (slots[i].active && slots[i].layer == layer) return &slots[i];
    }
    return NULL;
}

static metal_graph_stream_prepare_slot *metal_graph_stream_prepare_slot_free(
        metal_graph_stream_prepare_slot *slots,
        uint32_t                         n_slots) {
    for (uint32_t i = 0; i < n_slots; i++) {
        if (!slots[i].active) return &slots[i];
    }
    return NULL;
}

static bool metal_graph_stream_prepare_start_if_needed(
        const ds4_gpu_graph          *g,
        const ds4_model              *model,
        const ds4_weights            *weights,
        uint32_t                      layer,
        uint32_t                      n_tokens,
        bool                          madvise_only,
        bool                          pread_only,
        bool                          readahead_only,
        bool                          decode_only,
        metal_graph_stream_prepare_slot *slots,
        uint32_t                      n_slots) {
    if (layer >= DS4_N_LAYER) return true;
    if (metal_graph_stream_prepare_slot_find(slots, n_slots, layer)) {
        return true;
    }
    metal_graph_stream_prepare_slot *slot =
        metal_graph_stream_prepare_slot_free(slots, n_slots);
    if (!slot) {
        fprintf(stderr,
                "ds4: Metal streaming prefill prepare queue is full before layer %u\n",
                layer);
        return false;
    }
    memset(slot, 0, sizeof(*slot));
    slot->layer = layer;
    if (!metal_graph_stream_prefill_layer_pagein_start(g,
                                                       model,
                                                       weights,
                                                       layer,
                                                       n_tokens,
                                                       madvise_only,
                                                       pread_only,
                                                       readahead_only,
                                                       decode_only,
                                                       &slot->job)) {
        memset(slot, 0, sizeof(*slot));
        return false;
    }
    slot->active = slot->job.started;
    return true;
}

static bool metal_graph_stream_prepare_join_layer(
        const ds4_gpu_graph          *g,
        const ds4_model              *model,
        const ds4_weights            *weights,
        uint32_t                      layer,
        uint32_t                      n_tokens,
        bool                          madvise_only,
        bool                          pread_only,
        bool                          readahead_only,
        bool                          decode_only,
        metal_graph_stream_prepare_slot *slots,
        uint32_t                      n_slots) {
    metal_graph_stream_prepare_slot *slot =
        metal_graph_stream_prepare_slot_find(slots, n_slots, layer);
    if (!slot) {
        metal_graph_stream_pagein_job job;
        memset(&job, 0, sizeof(job));
        if (!metal_graph_stream_prefill_layer_pagein_start(g,
                                                           model,
                                                           weights,
                                                           layer,
                                                           n_tokens,
                                                           madvise_only,
                                                           pread_only,
                                                           readahead_only,
                                                           decode_only,
                                                           &job)) {
            return false;
        }
        return metal_graph_stream_prefill_layer_pagein_join(&job);
    }
    const bool ok = metal_graph_stream_prefill_layer_pagein_join(&slot->job);
    memset(slot, 0, sizeof(*slot));
    return ok;
}

static bool metal_graph_stream_prepare_join_all(
        metal_graph_stream_prepare_slot *slots,
        uint32_t                         n_slots) {
    bool ok = true;
    for (uint32_t i = 0; i < n_slots; i++) {
        if (!slots[i].active) continue;
        if (!metal_graph_stream_prefill_layer_pagein_join(&slots[i].job)) {
            ok = false;
        }
        memset(&slots[i], 0, sizeof(slots[i]));
    }
    return ok;
}

static void metal_graph_stream_readahead_layer(
        const ds4_model   *model,
        const ds4_weights *weights,
        uint32_t           il) {
    ds4_model_map_span_vec spans;
    if (!weights_model_map_spans(weights, il, il, false, &spans)) return;
    metal_graph_stream_readahead_spans(model, &spans);
    free(spans.v);
}

static void metal_graph_stream_readahead_layer_decode(
        const ds4_model   *model,
        const ds4_weights *weights,
        uint32_t           il) {
    ds4_model_map_span_vec spans;
    if (!weights_model_map_decode_layer_spans(weights, il, &spans)) return;
    metal_graph_stream_readahead_spans(model, &spans);
    free(spans.v);
}

static void metal_graph_stream_readahead_output(
        const ds4_model   *model,
        const ds4_weights *weights) {
    ds4_model_map_span_vec spans;
    if (!weights_model_map_output_spans(weights, &spans)) return;
    metal_graph_stream_readahead_spans(model, &spans);
    free(spans.v);
}

static bool metal_graph_stream_prefill_selected_readahead_enabled(
        const ds4_gpu_graph *g) {
    return g &&
           g->ssd_streaming &&
           (glm_graph_env_present("DS4_ROCM_ENABLE_STREAMING_PREFILL_SELECTED_READAHEAD",
                                  "DS4_METAL_ENABLE_STREAMING_PREFILL_SELECTED_READAHEAD") ||
            glm_graph_env_present("DS4_ROCM_ENABLE_STREAMING_PREFILL_SELECTED_READAHEAD_SHARED",
                                  "DS4_METAL_ENABLE_STREAMING_PREFILL_SELECTED_READAHEAD_SHARED")) &&
           !glm_graph_env_present("DS4_ROCM_DISABLE_STREAMING_PREFILL_SELECTED_READAHEAD",
                                  "DS4_METAL_DISABLE_STREAMING_PREFILL_SELECTED_READAHEAD");
}

static bool metal_graph_stream_prefill_selected_readahead_shared_enabled(
        const ds4_gpu_graph *g) {
    return g &&
           g->ssd_streaming &&
           glm_graph_env_present("DS4_ROCM_ENABLE_STREAMING_PREFILL_SELECTED_READAHEAD_SHARED",
                                 "DS4_METAL_ENABLE_STREAMING_PREFILL_SELECTED_READAHEAD_SHARED") &&
           !glm_graph_env_present("DS4_ROCM_DISABLE_STREAMING_PREFILL_SELECTED_READAHEAD_SHARED",
                                  "DS4_METAL_DISABLE_STREAMING_PREFILL_SELECTED_READAHEAD_SHARED") &&
           !glm_graph_env_present("DS4_ROCM_DISABLE_STREAMING_PREFILL_SELECTED_READAHEAD",
                                  "DS4_METAL_DISABLE_STREAMING_PREFILL_SELECTED_READAHEAD");
}

static uint32_t metal_graph_stream_prefill_selected_readahead_gap(void) {
    const char *env = glm_graph_env_value(
            "DS4_ROCM_STREAMING_PREFILL_SELECTED_READAHEAD_GAP",
            "DS4_METAL_STREAMING_PREFILL_SELECTED_READAHEAD_GAP");
    if (!env || !env[0]) return 0;
    char *end = NULL;
    unsigned long v = strtoul(env, &end, 10);
    if (end == env || *end != '\0') return 0;
    return v > 8 ? 8u : (uint32_t)v;
}

static bool metal_graph_stream_readahead_selected_run(
        const ds4_model         *model,
        const ds4_layer_weights *layer,
        uint32_t                 first,
        uint32_t                 last,
        uint64_t                 gate_expert_bytes,
        uint64_t                 down_expert_bytes,
        uint64_t                *hint_bytes) {
    if (!model || !layer || first > last || last >= DS4_N_EXPERT) return false;

    const uint64_t first_id = first;
    const uint64_t n_experts = (uint64_t)last - (uint64_t)first + 1u;
    if (first_id > UINT64_MAX / gate_expert_bytes ||
        first_id > UINT64_MAX / down_expert_bytes ||
        n_experts > UINT64_MAX / gate_expert_bytes ||
        n_experts > UINT64_MAX / down_expert_bytes) {
        fprintf(stderr, "ds4: Metal streaming prefill selected expert readahead overflow\n");
        return false;
    }

    const uint64_t gate_rel = first_id * gate_expert_bytes;
    const uint64_t down_rel = first_id * down_expert_bytes;
    const uint64_t gate_bytes = n_experts * gate_expert_bytes;
    const uint64_t down_bytes = n_experts * down_expert_bytes;
    if (gate_rel > UINT64_MAX - layer->ffn_gate_exps->abs_offset ||
        gate_rel > UINT64_MAX - layer->ffn_up_exps->abs_offset ||
        down_rel > UINT64_MAX - layer->ffn_down_exps->abs_offset) {
        fprintf(stderr, "ds4: Metal streaming prefill selected expert readahead overflow\n");
        return false;
    }

    metal_graph_stream_readahead_range_impl(model,
                                            layer->ffn_gate_exps->abs_offset + gate_rel,
                                            gate_bytes,
                                            true);
    metal_graph_stream_readahead_range_impl(model,
                                            layer->ffn_up_exps->abs_offset + gate_rel,
                                            gate_bytes,
                                            true);
    metal_graph_stream_readahead_range_impl(model,
                                            layer->ffn_down_exps->abs_offset + down_rel,
                                            down_bytes,
                                            true);
    if (hint_bytes) {
        if (*hint_bytes > UINT64_MAX - gate_bytes ||
            *hint_bytes + gate_bytes > UINT64_MAX - gate_bytes ||
            *hint_bytes + gate_bytes * 2u > UINT64_MAX - down_bytes) {
            *hint_bytes = UINT64_MAX;
        } else {
            *hint_bytes += gate_bytes * 2u + down_bytes;
        }
    }
    return true;
}

static bool metal_graph_stream_readahead_selected_experts_from_gpu(
        ds4_gpu_graph           *g,
        const ds4_model         *model,
        const ds4_layer_weights *layer,
        uint32_t                 il,
        uint32_t                 n_tokens,
        uint64_t                 gate_expert_bytes,
        uint64_t                 down_expert_bytes) {
    if (!metal_graph_stream_prefill_selected_readahead_enabled(g)) return true;
    if (!model || !layer || !g || !metal_graph_batch_router_selected(g) || n_tokens == 0) {
        return false;
    }
    if (sizeof(int) != sizeof(int32_t) || DS4_N_EXPERT > DS4_MAX_EXPERT) {
        return false;
    }

    const uint64_t n_ids = (uint64_t)n_tokens * DS4_N_EXPERT_USED;
    if (n_ids > SIZE_MAX / sizeof(int32_t)) return false;
    int32_t *selected = xmalloc((size_t)n_ids * sizeof(selected[0]));

    const bool profile =
        glm_graph_env_present("DS4_ROCM_STREAMING_PREFILL_SELECTED_READAHEAD_PROFILE",
                              "DS4_METAL_STREAMING_PREFILL_SELECTED_READAHEAD_PROFILE");
    const double t0 = profile ? now_sec() : 0.0;
    bool ok = ds4_gpu_tensor_read(metal_graph_batch_router_selected(g),
                                  0,
                                  selected,
                                  n_ids * sizeof(selected[0])) != 0;
    bool seen[DS4_MAX_EXPERT] = { false };
    uint32_t unique = 0;
    uint32_t ranges = 0;
    uint64_t hint_bytes = 0;
    const uint32_t gap = metal_graph_stream_prefill_selected_readahead_gap();
    if (ok) {
        for (uint64_t i = 0; i < n_ids; i++) {
            const int32_t expert = selected[i];
            if (expert < 0 || (uint32_t)expert >= DS4_N_EXPERT) {
                fprintf(stderr,
                        "ds4: Metal streaming prefill selected expert id %d is outside 0..%u at layer %u\n",
                        expert,
                        (uint32_t)DS4_N_EXPERT,
                        il);
                ok = false;
                break;
            }
            if (seen[expert]) continue;
            seen[expert] = true;
            unique++;
        }
    }

    if (ok) {
        uint32_t e = 0;
        while (e < DS4_N_EXPERT) {
            while (e < DS4_N_EXPERT && !seen[e]) e++;
            if (e >= DS4_N_EXPERT) break;
            const uint32_t first = e;
            uint32_t last = e;
            uint32_t skipped = 0;
            e++;
            while (e < DS4_N_EXPERT) {
                if (seen[e]) {
                    last = e;
                    skipped = 0;
                } else if (skipped < gap) {
                    skipped++;
                } else {
                    break;
                }
                e++;
            }

            if (!metal_graph_stream_readahead_selected_run(model,
                                                           layer,
                                                           first,
                                                           last,
                                                           gate_expert_bytes,
                                                           down_expert_bytes,
                                                           &hint_bytes)) {
                ok = false;
                break;
            }
            ranges++;
        }
    }
    if (profile) {
        fprintf(stderr,
                "ds4: Metal streaming prefill selected readahead layer=%u "
                "tokens=%u unique=%u ranges=%u gap=%u hint=%.2f GiB time=%.3f ms\n",
                il,
                n_tokens,
                unique,
                ranges,
                gap,
                (double)hint_bytes / (1024.0 * 1024.0 * 1024.0),
                (now_sec() - t0) * 1000.0);
    }
    free(selected);
    return ok;
}

static bool metal_graph_stream_map_token(
        const ds4_model   *model,
        const ds4_weights *weights) {
    ds4_model_map_span_vec spans;
    if (!weights_model_map_token_spans(weights, &spans)) {
        fprintf(stderr, "ds4: Metal SSD streaming could not build token embedding span\n");
        return false;
    }
    const bool ok = metal_graph_install_model_spans(model, &spans, "token embedding");
    free(spans.v);
    return ok;
}

static bool metal_graph_stream_map_decode_static_all(
        const ds4_model   *model,
        const ds4_weights *weights) {
    ds4_model_map_span_vec spans;
    if (!weights_model_map_decode_static_spans(weights, true, true, &spans)) {
        fprintf(stderr, "ds4: Metal SSD streaming could not build static decode spans\n");
        return false;
    }
    const bool ok = metal_graph_install_model_spans(model, &spans, "static decode");
    free(spans.v);
    return ok;
}

static bool metal_graph_stream_map_layer(
        const ds4_model   *model,
        const ds4_weights *weights,
        uint32_t           il) {
    ds4_model_map_span_vec spans;
    if (!weights_model_map_spans(weights, il, il, false, &spans)) {
        fprintf(stderr, "ds4: Metal SSD streaming could not build layer %u spans\n", il);
        return false;
    }
    const bool ok = metal_graph_install_model_spans(model, &spans, "layer");
    free(spans.v);
    return ok;
}

static bool metal_graph_stream_map_layer_decode(
        const ds4_model   *model,
        const ds4_weights *weights,
        uint32_t           il) {
    ds4_model_map_span_vec spans;
    if (!weights_model_map_decode_layer_spans(weights, il, &spans)) {
        fprintf(stderr, "ds4: Metal SSD streaming could not build decode layer %u spans\n", il);
        return false;
    }
    const bool ok = metal_graph_install_model_spans(model, &spans, "decode layer");
    free(spans.v);
    return ok;
}

static bool metal_graph_stream_map_output(
        const ds4_model   *model,
        const ds4_weights *weights) {
    ds4_model_map_span_vec spans;
    if (!weights_model_map_output_spans(weights, &spans)) {
        fprintf(stderr, "ds4: Metal SSD streaming could not build output head spans\n");
        return false;
    }
    const bool ok = metal_graph_install_model_spans(model, &spans, "output head");
    free(spans.v);
    return ok;
}

static uint32_t metal_graph_raw_span_for_batch(
        const ds4_gpu_graph *g,
        uint32_t               pos0,
        uint32_t               n_tokens) {
    if (!g || g->raw_cap == 0 || n_tokens == 0) return 0;

    const uint32_t window = g->raw_window ? g->raw_window : DS4_N_SWA;
    const uint32_t last_pos = pos0 + n_tokens - 1u;
    uint64_t needed = (uint64_t)n_tokens;
    if (window != 0) {
        needed += n_tokens == 1 ? (uint64_t)window - 1u : (uint64_t)window;
    }
    uint64_t available = (uint64_t)last_pos + 1u;
    if (needed > available) needed = available;
    if (needed > g->raw_cap) needed = g->raw_cap;
    return (uint32_t)needed;
}

static uint32_t metal_graph_raw_start_for_span(
        const ds4_gpu_graph *g,
        uint32_t               last_pos,
        uint32_t               n_raw) {
    if (!g || g->raw_cap == 0 || n_raw == 0) return 0;
    const uint32_t first_raw_pos = last_pos + 1u - n_raw;
    return first_raw_pos % g->raw_cap;
}

static uint32_t metal_graph_decode_raw_score_count(
        const ds4_gpu_graph *g,
        uint32_t               pos,
        uint32_t               n_raw,
        uint32_t               ratio) {
    if (!g || n_raw == 0) return 0;
    if (ratio == 0) return n_raw > 256u ? 256u : n_raw;

    const uint32_t first_raw_pos = pos + 1u - n_raw;
    const uint32_t raw_last_pos = first_raw_pos + n_raw - 1u;
    uint32_t lo = first_raw_pos;
    const uint32_t window = g->raw_window ? g->raw_window : DS4_N_SWA;
    if (window != 0 && pos + 1u > window) {
        const uint32_t wlo = pos + 1u - window;
        if (wlo > lo) lo = wlo;
    }
    const uint32_t hi = pos < raw_last_pos ? pos : raw_last_pos;
    if (hi < lo) return 0;
    uint32_t raw_count = hi - lo + 1u;
    if (raw_count > 256u) raw_count = 256u;
    return raw_count;
}

static bool metal_graph_cuda_splitkv_score_may_engage(
        const ds4_gpu_graph *g,
        uint32_t               pos) {
    if (!g) return false;

    const uint32_t min_score = metal_graph_cuda_greedy_splitkv_min_score();
    const uint32_t n_raw = metal_graph_raw_span_for_batch(g, pos, 1);
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        uint32_t visible_comp = 0;
        const uint32_t n_comp = g->layer_n_comp[il];
        if (n_comp != 0) {
            visible_comp = ratio == 0 ? n_comp : (pos + 1u) / ratio;
            if (visible_comp > n_comp) visible_comp = n_comp;
        }
        const uint32_t raw_count =
            metal_graph_decode_raw_score_count(g, pos, n_raw, ratio);
        const uint32_t n_score = raw_count + visible_comp;
        if (n_score > 1u && n_score >= min_score) return true;
    }
    return false;
}

static bool metal_graph_cuda_greedy_splitkv_may_engage(
        const ds4_gpu_graph *g,
        uint32_t               pos) {
    if (!metal_graph_cuda_greedy_splitkv_requested()) return false;
    return metal_graph_cuda_splitkv_score_may_engage(g, pos);
}

/* Capture the verifier prefix after the first speculative token.
 *
 * Exact MTP speculation is only profitable if partial accepts are cheap.  The
 * target verifier computes two draft tokens together; if only the first token
 * is accepted, replaying a one-token verifier throws away most of the gain.
 * For compressed-attention layers the mutable frontier is just the small
 * compressor state plus append counters, so we save that prefix-1 state while
 * the N=2 verifier is already stepping the compressor token by token.
 *
 * Raw SWA rows are not captured here.  This graph uses a raw ring larger than
 * the 128-token logical SWA window, so writing speculative future rows does
 * not evict visible raw rows.  If the raw cache is ever reduced to a strict
 * 128-row ring, speculative raw rows must become shadow rows and be copied
 * into the ring only on commit. */
static bool metal_graph_capture_prefix1_attn_state(ds4_gpu_graph *g, uint32_t il) {
    if (!g->spec_capture_prefix1 || !g->spec_prefix1_attn_state_kv[il]) return true;
    const uint64_t bytes = ds4_gpu_tensor_bytes(g->layer_attn_state_kv[il]);
    g->spec_prefix1_n_comp[il] = g->layer_n_comp[il];
    return ds4_gpu_tensor_copy(g->spec_prefix1_attn_state_kv[il], 0,
                                 g->layer_attn_state_kv[il], 0, bytes) != 0 &&
           ds4_gpu_tensor_copy(g->spec_prefix1_attn_state_score[il], 0,
                                 g->layer_attn_state_score[il], 0, bytes) != 0;
}

static bool metal_graph_capture_prefix1_index_state(ds4_gpu_graph *g, uint32_t il) {
    if (!g->spec_capture_prefix1 || !g->spec_prefix1_index_state_kv[il]) return true;
    const uint64_t bytes = ds4_gpu_tensor_bytes(g->layer_index_state_kv[il]);
    g->spec_prefix1_n_index_comp[il] = g->layer_n_index_comp[il];
    return ds4_gpu_tensor_copy(g->spec_prefix1_index_state_kv[il], 0,
                                 g->layer_index_state_kv[il], 0, bytes) != 0 &&
           ds4_gpu_tensor_copy(g->spec_prefix1_index_state_score[il], 0,
                                 g->layer_index_state_score[il], 0, bytes) != 0;
}

static uint32_t metal_graph_decode_indexer_sparse_threshold(const ds4_gpu_graph *g) {
    (void)g;
    static int parsed = -1;
    static uint32_t cached = 0;
    if (parsed < 0) {
        parsed = 0;
#ifndef DS4_ROCM_BUILD
        const char *env = getenv("DS4_METAL_DECODE_INDEXER_SPARSE_THRESHOLD");
        if (env && env[0]) {
            char *end = NULL;
            unsigned long v = strtoul(env, &end, 10);
            while (end && isspace((unsigned char)*end)) end++;
            if (end != env && end && *end == '\0' &&
                (v == 64ul || v == 128ul || v == 256ul || v == 512ul ||
                 v == 1024ul || v == 2048ul || v == 4096ul)) {
                cached = (uint32_t)v;
                parsed = 1;
            } else {
                fprintf(stderr,
                        "ds4: invalid DS4_METAL_DECODE_INDEXER_SPARSE_THRESHOLD=%s; "
                        "expected 64, 128, 256, 512, 1024, 2048, or 4096\n",
                        env);
            }
        }
#endif
    }
    if (parsed > 0) return cached;

    /* Keep dense attention longer than the legacy 512-row window by default.
     * Around the 2K frontier the sparse path's score/top-k setup dominates
     * the smaller attention scan, while larger contexts benefit from sparse
     * indexed attention.  This threshold changes only the implementation used
     * to consume the compressed rows; it must not lower the 512-row indexer
     * selection defined by DS4_N_INDEXER_TOP_K. */
    return 1024u;
}

/* =========================================================================
 * Metal Decode Release Helpers and Reference Fallbacks.
 * =========================================================================
 *
 * The normal generation path uses the fused helpers below.  The older unfused
 * kernels remain available as diagnostic reference paths selected only by the
 * DS4_METAL_DISABLE_*_FUSION environment switches.
 */

static bool metal_graph_env_flag(const char *name, int *cache) {
    if (*cache == -1) {
#ifdef DS4_ROCM_BUILD
        (void)name;
        *cache = 0;
#else
        const char *env = getenv(name);
        *cache = env && env[0] && strcmp(env, "0") != 0;
#endif
    }
    return *cache != 0;
}

static bool metal_graph_use_reference_hc_decode(void) {
    static int cache = -1;
    return metal_graph_env_flag("DS4_METAL_DISABLE_HC_FUSION", &cache);
}

static bool metal_graph_use_reference_kv_decode(void) {
    static int cache = -1;
    return metal_graph_env_flag("DS4_METAL_DISABLE_KV_FUSION", &cache);
}

static bool metal_graph_use_reference_qkv_norm(void) {
    static int cache = -1;
    return metal_graph_env_flag("DS4_METAL_DISABLE_QKV_NORM_FUSION", &cache);
}

static bool metal_graph_use_reference_qkv_pair_proj(void) {
    static int cache = -1;
    return metal_graph_env_flag("DS4_METAL_DISABLE_QKV_PAIR_PROJ", &cache);
}

static bool metal_graph_use_reference_compressor_pair_proj(void) {
    static int cache = -1;
    return metal_graph_env_flag("DS4_METAL_DISABLE_COMPRESSOR_PAIR_PROJ", &cache);
}

static bool metal_graph_use_reference_hc_norm_decode(void) {
    static int cache = -1;
    return metal_graph_env_flag("DS4_METAL_DISABLE_HC_NORM_FUSION", &cache);
}

static bool metal_graph_enable_batch_hc_norm_fusion(void) {
    static int cache = -1;
    if (metal_graph_use_reference_hc_norm_decode()) return false;
    if (cache == -1) {
        const char *disable = getenv("DS4_METAL_DISABLE_BATCH_HC_NORM_FUSION");
        if (disable && disable[0] && strcmp(disable, "0") != 0) {
            cache = 0;
        } else {
            const char *legacy_enable =
                getenv("DS4_METAL_ENABLE_BATCH_HC_NORM_FUSION");
            cache = (!legacy_enable || !legacy_enable[0] ||
                     strcmp(legacy_enable, "0") != 0) ? 1 : 0;
        }
    }
    return cache != 0;
}

static bool metal_graph_use_reference_shared_down_hc(void) {
    static int cache = -1;
    return metal_graph_env_flag("DS4_METAL_DISABLE_SHARED_DOWN_HC_FUSION", &cache);
}

static bool metal_graph_use_reference_attn_out_hc(void) {
    static int cache = -1;
    return metal_graph_env_flag("DS4_METAL_DISABLE_ATTN_OUT_HC_FUSION", &cache);
}

static bool metal_graph_decode_hc_pre(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *split,
        const ds4_gpu_tensor *mix,
        const ds4_gpu_tensor *residual_hc,
        const ds4_model        *model,
        uint64_t                scale_offset,
        uint64_t                base_offset) {
    if (metal_graph_use_reference_hc_decode()) {
        return ds4_gpu_hc_split_sinkhorn_tensor(split,
                                                  mix,
                                                  model->map,
                                                  model->size,
                                                  scale_offset,
                                                  base_offset,
                                                  DS4_N_HC,
                                                  DS4_N_HC_SINKHORN_ITER,
                                                  DS4_HC_EPS) != 0 &&
               ds4_gpu_hc_weighted_sum_tensor(out,
                                                 residual_hc,
                                                 split,
                                                 DS4_N_EMBD,
                                                 DS4_N_HC) != 0;
    }

    return ds4_gpu_hc_split_weighted_sum_tensor(out,
                                                  split,
                                                  mix,
                                                  residual_hc,
                                                  model->map,
                                                  model->size,
                                                  scale_offset,
                                                  base_offset,
                                                  DS4_N_EMBD,
                                                  DS4_N_HC,
                                                  DS4_N_HC_SINKHORN_ITER,
                                                  DS4_HC_EPS) != 0;
}

static bool metal_graph_hc_norm_fusion_check_enabled(void) {
    static int cache = -1;
    return metal_graph_env_flag("DS4_METAL_HC_NORM_FUSION_CHECK", &cache);
}

static float metal_graph_hc_norm_fusion_check_tolerance(void) {
    static int initialized;
    static float tolerance;
    if (initialized) return tolerance;
    tolerance = 2.0e-4f;
#ifndef DS4_ROCM_BUILD
    const char *env = getenv("DS4_METAL_HC_NORM_FUSION_CHECK_TOL");
    if (env && env[0]) {
        char *end = NULL;
        const float v = strtof(env, &end);
        if (end != env && isfinite(v) && v > 0.0f) tolerance = v;
    }
#endif
    initialized = 1;
    return tolerance;
}

static bool metal_graph_check_hc_norm_fusion(
        const char            *label,
        ds4_gpu_tensor        *fused_out,
        ds4_gpu_tensor        *fused_norm,
        const ds4_gpu_tensor  *mix,
        const ds4_gpu_tensor  *residual_hc,
        const ds4_model       *model,
        uint64_t               scale_offset,
        uint64_t               base_offset,
        uint64_t               norm_weight_offset,
        uint32_t               il,
        uint32_t               pos) {
    if (!metal_graph_hc_norm_fusion_check_enabled()) return true;
    if (!fused_out || !fused_norm || !mix || !residual_hc || !model) return false;

    const uint64_t n_embd = DS4_N_EMBD;
    const uint64_t mix_hc = 2ull * DS4_N_HC + (uint64_t)DS4_N_HC * DS4_N_HC;
    ds4_gpu_tensor *ref_split = ds4_gpu_tensor_alloc(mix_hc * sizeof(float));
    ds4_gpu_tensor *ref_out = ds4_gpu_tensor_alloc(n_embd * sizeof(float));
    ds4_gpu_tensor *ref_norm = ds4_gpu_tensor_alloc(n_embd * sizeof(float));
    bool ok = ref_split && ref_out && ref_norm;

    if (ok) {
        ok = ds4_gpu_hc_split_sinkhorn_tensor(ref_split,
                                              mix,
                                              model->map,
                                              model->size,
                                              scale_offset,
                                              base_offset,
                                              DS4_N_HC,
                                              DS4_N_HC_SINKHORN_ITER,
                                              DS4_HC_EPS) != 0 &&
             ds4_gpu_hc_weighted_sum_tensor(ref_out,
                                            residual_hc,
                                            ref_split,
                                            DS4_N_EMBD,
                                            DS4_N_HC) != 0 &&
             ds4_gpu_rms_norm_weight_tensor(ref_norm,
                                            ref_out,
                                            model->map,
                                            model->size,
                                            norm_weight_offset,
                                            DS4_N_EMBD,
                                            DS4_RMS_EPS) != 0;
    }

    if (ok) ok = ds4_gpu_end_commands() != 0;

    float *fused_out_cpu = NULL;
    float *ref_out_cpu = NULL;
    float *fused_norm_cpu = NULL;
    float *ref_norm_cpu = NULL;
    if (ok) {
        fused_out_cpu = xmalloc((size_t)n_embd * sizeof(float));
        ref_out_cpu = xmalloc((size_t)n_embd * sizeof(float));
        fused_norm_cpu = xmalloc((size_t)n_embd * sizeof(float));
        ref_norm_cpu = xmalloc((size_t)n_embd * sizeof(float));
        ok = ds4_gpu_tensor_read(fused_out, 0, fused_out_cpu, n_embd * sizeof(float)) != 0 &&
             ds4_gpu_tensor_read(ref_out, 0, ref_out_cpu, n_embd * sizeof(float)) != 0 &&
             ds4_gpu_tensor_read(fused_norm, 0, fused_norm_cpu, n_embd * sizeof(float)) != 0 &&
             ds4_gpu_tensor_read(ref_norm, 0, ref_norm_cpu, n_embd * sizeof(float)) != 0;
    }

    if (ok) {
        const float out_max = max_abs_diff(fused_out_cpu, ref_out_cpu, n_embd);
        const float out_rms = rms_abs_diff(fused_out_cpu, ref_out_cpu, n_embd);
        const float norm_max = max_abs_diff(fused_norm_cpu, ref_norm_cpu, n_embd);
        const float norm_rms = rms_abs_diff(fused_norm_cpu, ref_norm_cpu, n_embd);
        const float tol = metal_graph_hc_norm_fusion_check_tolerance();
        fprintf(stderr,
                "ds4: Metal HC norm fusion check %s layer=%u pos=%u "
                "out_max=%g out_rms=%g norm_max=%g norm_rms=%g tol=%g\n",
                label ? label : "hc",
                il,
                pos,
                out_max,
                out_rms,
                norm_max,
                norm_rms,
                tol);
        if (out_max > tol || norm_max > tol) {
            fprintf(stderr,
                    "ds4: Metal HC norm fusion check failed for %s layer=%u pos=%u\n",
                    label ? label : "hc",
                    il,
                    pos);
            ok = false;
        }
    }

    free(fused_out_cpu);
    free(ref_out_cpu);
    free(fused_norm_cpu);
    free(ref_norm_cpu);
    ds4_gpu_tensor_free(ref_norm);
    ds4_gpu_tensor_free(ref_out);
    ds4_gpu_tensor_free(ref_split);

    const bool restart_ok = ds4_gpu_begin_commands() != 0;
    return ok && restart_ok;
}

static bool metal_graph_decode_kv_store(
        ds4_gpu_tensor *kv,
        ds4_gpu_tensor *raw_cache,
        uint32_t          raw_cap,
        uint32_t          raw_row) {
    if (metal_graph_use_reference_kv_decode()) {
        return ds4_gpu_dsv4_fp8_kv_quantize_tensor(kv, 1, DS4_N_HEAD_DIM, DS4_N_ROT) != 0 &&
               ds4_gpu_store_raw_kv_tensor(raw_cache, kv, raw_cap, raw_row, DS4_N_HEAD_DIM) != 0;
    }

    return ds4_gpu_kv_fp8_store_raw_tensor(kv,
                                             raw_cache,
                                             raw_cap,
                                             raw_row,
                                             DS4_N_HEAD_DIM,
                                             DS4_N_ROT) != 0;
}

static uint64_t metal_graph_attn_comp_cache_row_bytes(void) {
    return (uint64_t)DS4_N_HEAD_DIM *
           (DS4_GPU_ATTN_COMP_CACHE_F16 ? sizeof(uint16_t) : sizeof(float));
}

static uint32_t metal_graph_attn_comp_cache_is_f16(void) {
    return DS4_GPU_ATTN_COMP_CACHE_F16 ? 1u : 0u;
}

static bool metal_graph_store_attn_comp_stage(
        ds4_gpu_graph *g,
        uint32_t       il,
        uint32_t       first_row,
        uint32_t       rows) {
    if (!g || il >= DS4_N_LAYER) return false;
    if (rows == 0) return true;
    if (!g->layer_attn_comp_cache[il] || !metal_graph_attn_comp_stage(g)) return false;
    if (rows > g->attn_comp_stage_cap || first_row > g->layer_comp_cap[il] ||
        rows > g->layer_comp_cap[il] - first_row) {
        return false;
    }

    const uint64_t count = (uint64_t)rows * DS4_N_HEAD_DIM;
    const uint64_t dst_offset = (uint64_t)first_row *
                                metal_graph_attn_comp_cache_row_bytes();
    if (DS4_GPU_ATTN_COMP_CACHE_F16) {
        return ds4_gpu_tensor_copy_f32_to_f16(g->layer_attn_comp_cache[il],
                                               dst_offset,
                                               metal_graph_attn_comp_stage(g),
                                               0,
                                               count) != 0;
    }

    return ds4_gpu_tensor_copy(g->layer_attn_comp_cache[il],
                               dst_offset,
                               metal_graph_attn_comp_stage(g),
                               0,
                               count * sizeof(float)) != 0;
}

static ds4_gpu_tensor *metal_graph_attn_comp_update_target(
        ds4_gpu_graph *g,
        uint32_t       il) {
    return DS4_GPU_ATTN_COMP_CACHE_F16
        ? metal_graph_attn_comp_stage(g)
        : g->layer_attn_comp_cache[il];
}

static uint32_t metal_graph_attn_comp_update_row(uint32_t row) {
    return DS4_GPU_ATTN_COMP_CACHE_F16 ? 0u : row;
}

static bool metal_graph_commit_attn_comp_stage(
        ds4_gpu_graph *g,
        uint32_t       il,
        uint32_t       first_row,
        uint32_t       rows) {
    if (!DS4_GPU_ATTN_COMP_CACHE_F16) return true;
    return metal_graph_store_attn_comp_stage(g, il, first_row, rows);
}

static ds4_gpu_tensor *metal_graph_attn_comp_row_view(
        ds4_gpu_graph *g,
        uint32_t       il,
        uint32_t       row) {
    if (DS4_GPU_ATTN_COMP_CACHE_F16) {
        return ds4_gpu_tensor_view(metal_graph_attn_comp_stage(g),
                                   0,
                                   (uint64_t)DS4_N_HEAD_DIM * sizeof(float));
    }
    return ds4_gpu_tensor_view(g->layer_attn_comp_cache[il],
                               (uint64_t)row * DS4_N_HEAD_DIM * sizeof(float),
                               (uint64_t)DS4_N_HEAD_DIM * sizeof(float));
}

static ds4_gpu_tensor *metal_graph_attn_comp_prefill_target(
        ds4_gpu_graph *g,
        uint32_t       il,
        uint32_t       first_row,
        uint32_t       rows) {
    if (DS4_GPU_ATTN_COMP_CACHE_F16) return metal_graph_attn_comp_stage(g);
    const uint32_t view_rows = rows ? rows : 1u;
    return ds4_gpu_tensor_view(g->layer_attn_comp_cache[il],
                               (uint64_t)first_row * DS4_N_HEAD_DIM * sizeof(float),
                               (uint64_t)view_rows * DS4_N_HEAD_DIM * sizeof(float));
}

static void metal_graph_attn_comp_prefill_target_free(ds4_gpu_tensor *t) {
    if (!DS4_GPU_ATTN_COMP_CACHE_F16) ds4_gpu_tensor_free(t);
}

static bool metal_graph_cuda_tp_attn_cache_dup_layer_ready(
        const ds4_gpu_graph *g,
        uint32_t             il) {
    if (!g || il >= DS4_N_LAYER || !g->cuda_tp_attn_cache_dup) return false;
    if (!g->placement || !g->layer_raw_cache[il] || !g->layer_raw_cache_tp[il]) {
        return false;
    }
    const int layer_tier = g->placement[il + 1];
    if (metal_graph_cuda_tp_partner_tier(layer_tier) < 0) return false;
    const uint32_t ratio = ds4_layer_compress_ratio(il);
    if (ratio != 0 &&
        (!g->layer_attn_comp_cache[il] ||
         !g->layer_attn_comp_cache_tp[il])) {
        return false;
    }
    return true;
}

static bool metal_graph_cuda_tp_attn_cache_copy_row(
        ds4_gpu_tensor       *dst_base,
        const ds4_gpu_tensor *src_base,
        uint64_t              offset,
        uint64_t              bytes) {
    if (bytes == 0) return true;
    ds4_gpu_tensor *dst = ds4_gpu_tensor_view(dst_base, offset, bytes);
    ds4_gpu_tensor *src = ds4_gpu_tensor_view(src_base, offset, bytes);
    bool ok = dst && src && ds4_gpu_tensor_copy_xdev(dst, src, bytes) != 0;
    ds4_gpu_tensor_free(src);
    ds4_gpu_tensor_free(dst);
    return ok;
}

static bool metal_graph_cuda_tp_attn_cache_sync_raw_row(
        ds4_gpu_graph *g,
        uint32_t       il,
        uint32_t       raw_row) {
    if (!metal_graph_cuda_tp_attn_cache_dup_layer_ready(g, il)) return true;
    if (raw_row >= g->raw_cap) return false;
    const uint64_t row_bytes = (uint64_t)DS4_N_HEAD_DIM * sizeof(float);
    return metal_graph_cuda_tp_attn_cache_copy_row(
            g->layer_raw_cache_tp[il],
            g->layer_raw_cache[il],
            (uint64_t)raw_row * row_bytes,
            row_bytes);
}

static bool metal_graph_cuda_tp_attn_cache_sync_all(ds4_gpu_graph *g) {
    if (!g || !g->cuda_tp_attn_cache_dup) return true;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        if (!metal_graph_cuda_tp_attn_cache_dup_layer_ready(g, il)) return false;
        const uint64_t raw_bytes =
            (uint64_t)g->raw_cap * DS4_N_HEAD_DIM * sizeof(float);
        if (!ds4_gpu_tensor_copy_xdev(g->layer_raw_cache_tp[il],
                                      g->layer_raw_cache[il],
                                      raw_bytes)) {
            return false;
        }
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        const uint32_t n_comp = g->layer_n_comp[il];
        if (ratio != 0 && n_comp != 0) {
            const uint64_t comp_bytes =
                (uint64_t)n_comp * metal_graph_attn_comp_cache_row_bytes();
            if (!ds4_gpu_tensor_copy_xdev(g->layer_attn_comp_cache_tp[il],
                                          g->layer_attn_comp_cache[il],
                                          comp_bytes)) {
                return false;
            }
        }
    }
    return true;
}

/* Encode one DS4 decode layer on Metal.  This is the release single-token
 * layer path; diagnostics reuse it so they compare exactly what generation
 * runs. */
static bool metal_graph_indexer_stage_profile_boundary(
        const char *stage,
        uint32_t    il,
        uint32_t    pos0,
        uint32_t    n_tokens,
        uint32_t    n_comp,
        double     *stage_t0);
static bool metal_graph_layer_stage_profile_boundary(
        const char *part,
        const char *stage,
        uint32_t    il,
        uint32_t    pos0,
        uint32_t    n_tokens,
        double     *stage_t0);
static bool metal_graph_decode_stage_profile_enabled(uint32_t il);
static bool metal_graph_matmul_plain_tensor(
        ds4_gpu_tensor       *out,
        const ds4_model        *model,
        const ds4_tensor       *w,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok);
static bool metal_graph_matmul_dense_quant_tensor(
        ds4_gpu_tensor       *out,
        const ds4_model        *model,
        const ds4_tensor       *w,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok);
static bool metal_graph_dense_quant_row_bytes(
        const ds4_tensor *w,
        uint64_t          in_dim,
        uint64_t         *row_bytes);
static bool metal_graph_matmul_dense_quant_abs(
        ds4_gpu_tensor       *out,
        const ds4_model        *model,
        const ds4_tensor       *w,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok);
static bool metal_graph_matmul_dense_quant_kslice(
        ds4_gpu_tensor       *out,
        const ds4_model        *model,
        const ds4_tensor       *w,
        uint64_t                full_in_dim,
        uint64_t                k_off,
        uint64_t                k_cnt,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                x_elem_off);
static bool metal_graph_attention_output_dense_quant_low(
        ds4_gpu_tensor       *low,
        ds4_gpu_graph        *g,
        const ds4_model        *model,
        const ds4_tensor       *out_a,
        uint64_t                group_dim,
        uint64_t                rank,
        uint32_t                group0,
        uint32_t                group_cnt,
        const ds4_gpu_tensor *heads);
static bool metal_graph_attention_output_dense_quant_tp(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *low,
        ds4_gpu_graph        *g,
        const ds4_model        *model,
        const ds4_tensor       *out_a,
        const ds4_tensor       *out_b,
        uint64_t                group_dim,
        uint64_t                rank,
        uint32_t                n_groups_total,
        uint32_t                group0,
        uint32_t                group_cnt,
        uint64_t                out_dim,
        const ds4_gpu_tensor *heads);
static bool metal_graph_attention_output_dense_quant_batch(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *low,
        ds4_gpu_graph        *g,
        const ds4_model        *model,
        const ds4_tensor       *out_a,
        const ds4_tensor       *out_b,
        uint64_t                group_dim,
        uint64_t                rank,
        uint32_t                n_groups,
        uint64_t                out_dim,
        const ds4_gpu_tensor *heads,
        uint32_t                n_tokens);

static bool metal_graph_use_pro_q4_cpu_router(void) {
    static int cache = -1;
    return metal_graph_env_flag("DS4_METAL_PRO_Q4_CPU_ROUTER", &cache);
}

static bool metal_graph_use_streaming_iq2_cpu_router(void) {
    return getenv("DS4_METAL_ENABLE_STREAMING_IQ2_CPU_ROUTER") != NULL &&
           getenv("DS4_METAL_DISABLE_STREAMING_IQ2_CPU_ROUTER") == NULL;
}

static bool metal_graph_use_q4_selected_shared_overlap(void) {
    static int cache = -1;
    return metal_graph_env_flag("DS4_METAL_Q4_SELECTED_OVERLAP_SHARED", &cache);
}

static bool metal_graph_use_cuda_selected_shared_overlap(const ds4_gpu_graph *g) {
#if !defined(DS4_ROCM_BUILD) && !defined(DS4_NO_GPU) && !defined(__APPLE__)
    return g &&
           g->ssd_streaming &&
           getenv("DS4_CUDA_DISABLE_STREAMING_SELECTED_SHARED_OVERLAP") == NULL;
#else
    (void)g;
    return false;
#endif
}

static bool metal_graph_q4_non_streaming_opt_in_enabled(void) {
    return getenv("DS4_METAL_ENABLE_Q4_SELECTED_EXPERT_VIEWS") != NULL ||
           getenv("DS4_METAL_ENABLE_PRO_Q4_SELECTED_EXPERT_VIEWS") != NULL ||
           getenv("DS4_METAL_ENABLE_Q4_EXPERT_TABLE") != NULL ||
           getenv("DS4_METAL_ENABLE_Q4_EXPERT_ADDRESS_TABLE") != NULL ||
           getenv("DS4_METAL_ENABLE_PRO_Q4_EXPERT_TABLE_AUTO") != NULL ||
           getenv("DS4_METAL_ENABLE_PRO_Q4_EXPERT_ADDRESS_AUTO") != NULL;
}

static bool metal_graph_q4_selected_paths_allowed(const ds4_gpu_graph *g) {
    if (!g) return false;
    if (g->ssd_streaming) return true;
    if (DS4_MODEL_FAMILY == DS4_MODEL_FAMILY_GLM_DSA) return false;
    return metal_graph_q4_non_streaming_opt_in_enabled();
}

static bool metal_graph_use_iq2_selected_shared_overlap(const ds4_gpu_graph *g) {
    return g &&
           g->ssd_streaming &&
           getenv("DS4_METAL_DISABLE_STREAMING_SELECTED_SHARED_OVERLAP") == NULL &&
           getenv("DS4_METAL_DISABLE_IQ2_SELECTED_SHARED_OVERLAP") == NULL;
}

static bool metal_graph_use_iq2_selected_async_load(const ds4_gpu_graph *g) {
    return g &&
           g->ssd_streaming &&
#ifndef DS4_ROCM_BUILD
           getenv("DS4_METAL_DISABLE_STREAMING_SELECTED_ASYNC_LOAD") == NULL;
#else
           true;
#endif
}

static bool metal_graph_use_iq2_selected_async_early_commit(
        const ds4_gpu_graph *g) {
    return g &&
           g->ssd_streaming &&
#ifndef DS4_ROCM_BUILD
           getenv("DS4_METAL_DISABLE_STREAMING_SELECTED_ASYNC_EARLY_COMMIT") == NULL;
#else
           false;
#endif
}

static bool metal_graph_use_pro_q4_expert_table_auto(const ds4_gpu_graph *g) {
    if (getenv("DS4_METAL_DISABLE_PRO_Q4_EXPERT_TABLE_AUTO") != NULL ||
        getenv("DS4_METAL_DISABLE_Q4_EXPERT_TABLE") != NULL) {
        return false;
    }
    if (!g || (!g->ssd_streaming &&
               getenv("DS4_METAL_ENABLE_PRO_Q4_EXPERT_TABLE_AUTO") == NULL)) {
        return false;
    }
#ifndef DS4_NO_GPU
    return ds4_gpu_pro_q4_expert_table_auto_available() != 0;
#else
    return false;
#endif
}

static bool metal_graph_decode_cpu_router_applicable(
        const ds4_gpu_graph     *g,
        const ds4_layer_weights *layer) {
    const bool pro_q4 =
        DS4_MODEL_VARIANT == DS4_VARIANT_PRO &&
        metal_graph_use_pro_q4_cpu_router() &&
        layer->ffn_gate_exps->type == DS4_TENSOR_Q4_K &&
        layer->ffn_up_exps->type == DS4_TENSOR_Q4_K &&
        layer->ffn_down_exps->type == DS4_TENSOR_Q4_K;
    const bool streaming_iq2 =
        g &&
        g->ssd_streaming &&
        !g->quality &&
        metal_graph_use_streaming_iq2_cpu_router() &&
        layer->ffn_gate_tid2eid == NULL &&
        layer->ffn_gate_exps->type == DS4_TENSOR_IQ2_XXS &&
        layer->ffn_up_exps->type == DS4_TENSOR_IQ2_XXS &&
        layer->ffn_down_exps->type == DS4_TENSOR_Q2_K &&
        DS4_N_EXPERT_USED == 6 &&
        DS4_N_EXPERT >= 128 &&
        !glm_graph_env_present("DS4_ROCM_MOE_WRITE_CLAMPED_ACT",
                               "DS4_METAL_MOE_WRITE_CLAMPED_ACT") &&
        !glm_graph_env_present("DS4_ROCM_DISABLE_ROUTED_PAIR_SWIGLU_FUSION",
                               "DS4_METAL_DISABLE_ROUTED_PAIR_SWIGLU_FUSION") &&
        !glm_graph_env_present("DS4_ROCM_DISABLE_IQ2_SELECTED_EXPERT_VIEWS",
                               "DS4_METAL_DISABLE_IQ2_SELECTED_EXPERT_VIEWS");
    return pro_q4 || streaming_iq2;
}

static bool metal_graph_decode_pro_q4_expert_table_expected(
        const ds4_gpu_graph     *g,
        const ds4_layer_weights *layer,
        uint64_t                 gate_tensor_bytes,
        uint64_t                 down_tensor_bytes) {
    const uint64_t q4_selected_min_tensor_bytes = 2ull * 1024ull * 1024ull * 1024ull;
    return !g->quality &&
           DS4_MODEL_VARIANT == DS4_VARIANT_PRO &&
           metal_graph_q4_selected_paths_allowed(g) &&
           layer->ffn_gate_exps->type == DS4_TENSOR_Q4_K &&
           layer->ffn_up_exps->type == DS4_TENSOR_Q4_K &&
           layer->ffn_down_exps->type == DS4_TENSOR_Q4_K &&
           DS4_N_EXPERT == 384 &&
           DS4_N_EXPERT_USED == 6 &&
           gate_tensor_bytes >= q4_selected_min_tensor_bytes &&
           down_tensor_bytes >= q4_selected_min_tensor_bytes &&
           !glm_graph_env_present("DS4_ROCM_MOE_WRITE_CLAMPED_ACT",
                                  "DS4_METAL_MOE_WRITE_CLAMPED_ACT") &&
           !glm_graph_env_present("DS4_ROCM_DISABLE_ROUTED_PAIR_SWIGLU_FUSION",
                                  "DS4_METAL_DISABLE_ROUTED_PAIR_SWIGLU_FUSION") &&
           (metal_graph_use_pro_q4_expert_table_auto(g) ||
            getenv("DS4_METAL_ENABLE_Q4_EXPERT_TABLE") != NULL) &&
           getenv("DS4_METAL_DISABLE_PRO_Q4_EXPERT_TABLE_AUTO") == NULL &&
           getenv("DS4_METAL_DISABLE_Q4_EXPERT_TABLE") == NULL;
}

static bool metal_graph_decode_q4_selected_slots_expected(
        const ds4_gpu_graph     *g,
        const ds4_layer_weights *layer,
        uint64_t                 gate_tensor_bytes,
        uint64_t                 down_tensor_bytes) {
    if (metal_graph_decode_pro_q4_expert_table_expected(g, layer,
                                                        gate_tensor_bytes,
                                                        down_tensor_bytes)) {
        return false;
    }
    const uint64_t q4_selected_min_tensor_bytes = 2ull * 1024ull * 1024ull * 1024ull;
    return !g->quality &&
           metal_graph_q4_selected_paths_allowed(g) &&
           layer->ffn_gate_exps->type == DS4_TENSOR_Q4_K &&
           layer->ffn_up_exps->type == DS4_TENSOR_Q4_K &&
           layer->ffn_down_exps->type == DS4_TENSOR_Q4_K &&
           DS4_N_EXPERT_USED == 6 &&
           DS4_N_EXPERT >= 128 &&
           (g->ssd_streaming ||
            (gate_tensor_bytes >= q4_selected_min_tensor_bytes &&
             down_tensor_bytes >= q4_selected_min_tensor_bytes)) &&
           !glm_graph_env_present("DS4_ROCM_MOE_WRITE_CLAMPED_ACT",
                                  "DS4_METAL_MOE_WRITE_CLAMPED_ACT") &&
           !glm_graph_env_present("DS4_ROCM_DISABLE_ROUTED_PAIR_SWIGLU_FUSION",
                                  "DS4_METAL_DISABLE_ROUTED_PAIR_SWIGLU_FUSION") &&
           !glm_graph_env_present("DS4_ROCM_DISABLE_Q4_SELECTED_EXPERT_VIEWS",
                                  "DS4_METAL_DISABLE_Q4_SELECTED_EXPERT_VIEWS");
}

static bool metal_graph_decode_iq2_selected_slots_expected(
        const ds4_gpu_graph     *g,
        const ds4_layer_weights *layer) {
    return g &&
           g->ssd_streaming &&
           !g->quality &&
           layer->ffn_gate_exps->type == DS4_TENSOR_IQ2_XXS &&
           layer->ffn_up_exps->type == DS4_TENSOR_IQ2_XXS &&
           layer->ffn_down_exps->type == DS4_TENSOR_Q2_K &&
           DS4_N_EXPERT_USED == 6 &&
           DS4_N_EXPERT >= 128 &&
           !glm_graph_env_present("DS4_ROCM_MOE_WRITE_CLAMPED_ACT",
                                  "DS4_METAL_MOE_WRITE_CLAMPED_ACT") &&
           !glm_graph_env_present("DS4_ROCM_DISABLE_ROUTED_PAIR_SWIGLU_FUSION",
                                  "DS4_METAL_DISABLE_ROUTED_PAIR_SWIGLU_FUSION") &&
           !glm_graph_env_present("DS4_ROCM_DISABLE_IQ2_SELECTED_EXPERT_VIEWS",
                                  "DS4_METAL_DISABLE_IQ2_SELECTED_EXPERT_VIEWS");
}

static bool metal_graph_streaming_expert_cache_seed_layer_expected(
        const ds4_gpu_graph     *g,
        const ds4_layer_weights *layer) {
    if (!g ||
        !g->ssd_streaming ||
        !layer ||
        !layer->ffn_gate_exps ||
        !layer->ffn_up_exps ||
        !layer->ffn_down_exps) {
        return false;
    }
    if (metal_graph_decode_iq2_selected_slots_expected(g, layer)) return true;
    if (DS4_MODEL_FAMILY == DS4_MODEL_FAMILY_GLM_DSA &&
        !g->quality &&
        layer->ffn_gate_exps->type == DS4_TENSOR_IQ2_XXS &&
        layer->ffn_up_exps->type == DS4_TENSOR_IQ2_XXS &&
        layer->ffn_down_exps->type == DS4_TENSOR_IQ2_XXS &&
        DS4_N_EXPERT_USED != 0 &&
        DS4_N_EXPERT_USED <= 8 &&
        DS4_N_EXPERT >= 128 &&
        !glm_graph_env_present("DS4_ROCM_GLM_DISABLE_STREAMING_EXPERT_CACHE",
                               "DS4_METAL_GLM_DISABLE_STREAMING_EXPERT_CACHE")) {
        return true;
    }
    if (DS4_MODEL_FAMILY != DS4_MODEL_FAMILY_GLM_DSA ||
        g->quality ||
        layer->ffn_gate_exps->type != layer->ffn_up_exps->type ||
        layer->ffn_gate_exps->type != layer->ffn_down_exps->type ||
        DS4_N_EXPERT_USED == 0 ||
        DS4_N_EXPERT_USED > 8 ||
        DS4_N_EXPERT < 128 ||
        glm_graph_env_present("DS4_ROCM_GLM_DISABLE_STREAMING_EXPERT_CACHE",
                              "DS4_METAL_GLM_DISABLE_STREAMING_EXPERT_CACHE")) {
        return false;
    }
    const uint32_t type = layer->ffn_gate_exps->type;
    return type == DS4_TENSOR_Q2_K || type == DS4_TENSOR_Q4_K;
}

static bool metal_graph_decode_cuda_selected_slots_expected(
        const ds4_gpu_graph     *g,
        const ds4_layer_weights *layer) {
#if !defined(DS4_ROCM_BUILD) && !defined(DS4_NO_GPU) && !defined(__APPLE__)
    if (!g ||
        !g->ssd_streaming ||
        g->quality ||
        !layer ||
        !layer->ffn_gate_exps ||
        !layer->ffn_up_exps ||
        !layer->ffn_down_exps ||
        DS4_N_EXPERT_USED != 6 ||
        DS4_N_EXPERT < 128 ||
        getenv("DS4_METAL_MOE_WRITE_CLAMPED_ACT") != NULL ||
        getenv("DS4_METAL_DISABLE_ROUTED_PAIR_SWIGLU_FUSION") != NULL) {
        return false;
    }
    const bool q4 =
        layer->ffn_gate_exps->type == DS4_TENSOR_Q4_K &&
        layer->ffn_up_exps->type == DS4_TENSOR_Q4_K &&
        layer->ffn_down_exps->type == DS4_TENSOR_Q4_K &&
        getenv("DS4_METAL_DISABLE_Q4_SELECTED_EXPERT_VIEWS") == NULL;
    const bool iq2 =
        layer->ffn_gate_exps->type == DS4_TENSOR_IQ2_XXS &&
        layer->ffn_up_exps->type == DS4_TENSOR_IQ2_XXS &&
        layer->ffn_down_exps->type == DS4_TENSOR_Q2_K &&
        getenv("DS4_METAL_DISABLE_IQ2_SELECTED_EXPERT_VIEWS") == NULL;
    return q4 || iq2;
#else
    (void)g;
    (void)layer;
    return false;
#endif
}

static uint32_t metal_graph_streaming_prefill_cache_seed_k(const ds4_gpu_graph *g) {
    const bool enabled =
        glm_graph_env_present("DS4_ROCM_ENABLE_STREAMING_PREFILL_CACHE_SEED",
                              "DS4_METAL_ENABLE_STREAMING_PREFILL_CACHE_SEED");
    if (!g ||
        !g->ssd_streaming ||
        !enabled) {
        return 0;
    }

    uint32_t k = 1;
    const char *env = glm_graph_env_value("DS4_ROCM_STREAMING_PREFILL_CACHE_SEED_K",
                                          "DS4_METAL_STREAMING_PREFILL_CACHE_SEED_K");
    if (env && env[0]) {
        char *end = NULL;
        unsigned long v = strtoul(env, &end, 10);
        if (end != env && *end == '\0') {
            if (v == 0) return 0;
            k = v > DS4_STREAMING_PREFILL_CACHE_SEED_MAX_TOKENS ?
                DS4_STREAMING_PREFILL_CACHE_SEED_MAX_TOKENS : (uint32_t)v;
        }
    }
    return k;
}

static bool metal_graph_streaming_prefill_cache_seed_enabled(const ds4_gpu_graph *g) {
    return metal_graph_streaming_prefill_cache_seed_k(g) != 0;
}

static bool metal_graph_streaming_expert_hotlist_enabled(const ds4_gpu_graph *g) {
    return g &&
           g->ssd_streaming &&
           !g->ssd_streaming_cold &&
           !glm_graph_env_present("DS4_ROCM_DISABLE_STREAMING_EXPERT_HOTLIST",
                                  "DS4_METAL_DISABLE_STREAMING_EXPERT_HOTLIST");
}

static bool metal_graph_streaming_expert_hotlist_add(
        uint32_t    layer,
        uint32_t    expert,
        uint32_t    priority,
        int32_t     experts[DS4_MAX_LAYER][DS4_MAX_EXPERT],
        uint32_t    priorities[DS4_MAX_LAYER][DS4_MAX_EXPERT],
        uint32_t    counts[DS4_MAX_LAYER],
        bool        seen[DS4_MAX_LAYER][DS4_MAX_EXPERT],
        uint32_t   *loaded) {
    if (layer >= DS4_N_LAYER || expert >= DS4_N_EXPERT) return true;
    if (layer >= DS4_MAX_LAYER || expert >= DS4_MAX_EXPERT) return true;
    if (seen[layer][expert]) return true;
    if (counts[layer] >= DS4_MAX_EXPERT) return false;
    seen[layer][expert] = true;
    if (priority == 0) priority = 1;
    priorities[layer][counts[layer]] = priority;
    experts[layer][counts[layer]++] = (int32_t)expert;
    (*loaded)++;
    return true;
}

static bool metal_graph_streaming_expert_hotlist_load_file(
        const char *path,
        uint32_t    max_entries,
        int32_t     experts[DS4_MAX_LAYER][DS4_MAX_EXPERT],
        uint32_t    priorities[DS4_MAX_LAYER][DS4_MAX_EXPERT],
        uint32_t    counts[DS4_MAX_LAYER],
        bool        seen[DS4_MAX_LAYER][DS4_MAX_EXPERT],
        uint32_t   *loaded_out) {
    if (!path || !path[0] || max_entries == 0 ||
        !experts || !priorities || !counts || !seen || !loaded_out) {
        return false;
    }
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr,
                "ds4: failed to open streaming expert hotlist %s: %s\n",
                path,
                strerror(errno));
        return false;
    }

    char line[256];
    uint64_t lineno = 0;
    uint32_t loaded = 0;
    while (fgets(line, sizeof(line), fp)) {
        if (loaded >= max_entries) break;
        lineno++;
        char *p = line;
        while (*p && isspace((unsigned char)*p)) p++;
        if (*p == '\0' || *p == '#') continue;

        errno = 0;
        char *end = NULL;
        unsigned long layer = strtoul(p, &end, 10);
        if (end == p || errno != 0) goto bad_line;
        p = end;
        while (*p && isspace((unsigned char)*p)) p++;

        errno = 0;
        unsigned long expert = strtoul(p, &end, 10);
        if (end == p || errno != 0) goto bad_line;
        p = end;
        while (*p && isspace((unsigned char)*p)) p++;

        errno = 0;
        unsigned long long hits = strtoull(p, &end, 10);
        if (end == p || errno != 0) goto bad_line;
        if (hits == 0) continue;
        const uint32_t priority =
            hits > UINT32_MAX ? UINT32_MAX : (uint32_t)hits;
        if (!metal_graph_streaming_expert_hotlist_add((uint32_t)layer,
                                                      (uint32_t)expert,
                                                      priority,
                                                      experts,
                                                      priorities,
                                                      counts,
                                                      seen,
                                                      &loaded)) {
            goto bad_line;
        }
        continue;

bad_line:
        fprintf(stderr,
                "ds4: invalid streaming expert hotlist line %" PRIu64 " in %s\n",
                lineno,
                path);
        fclose(fp);
        return false;
    }
    if (ferror(fp)) {
        fprintf(stderr,
                "ds4: failed to read streaming expert hotlist %s: %s\n",
                path,
                strerror(errno));
        fclose(fp);
        return false;
    }
    fclose(fp);

    if (loaded == 0) {
        fprintf(stderr, "ds4: streaming expert hotlist %s had no usable nonzero entries\n", path);
    }
    *loaded_out = loaded;
    return true;
}

static bool metal_graph_streaming_expert_hotlist_load_default(
        uint32_t    max_entries,
        int32_t     experts[DS4_MAX_LAYER][DS4_MAX_EXPERT],
        uint32_t    priorities[DS4_MAX_LAYER][DS4_MAX_EXPERT],
        uint32_t    counts[DS4_MAX_LAYER],
        bool        seen[DS4_MAX_LAYER][DS4_MAX_EXPERT],
        uint32_t   *loaded_out) {
    if (max_entries == 0 || !experts || !priorities || !counts || !seen || !loaded_out) {
        return false;
    }
    const uint16_t (*hotlist)[2] = NULL;
    uint32_t hotlist_count = 0;
    if (g_ds4_shape.variant == DS4_VARIANT_PRO) {
        hotlist = ds4_default_streaming_hotlist_pro;
        hotlist_count = ds4_default_streaming_hotlist_pro_count;
    } else if (g_ds4_shape.variant == DS4_VARIANT_FLASH) {
        hotlist = ds4_default_streaming_hotlist_flash;
        hotlist_count = ds4_default_streaming_hotlist_flash_count;
    } else if (g_ds4_shape.variant == DS4_VARIANT_GLM52) {
        hotlist = ds4_default_streaming_hotlist_glm52;
        hotlist_count = ds4_default_streaming_hotlist_glm52_count;
    } else {
        *loaded_out = 0;
        return true;
    }
    uint32_t loaded = 0;
    for (uint32_t i = 0;
         i < hotlist_count && loaded < max_entries;
         i++) {
        if (!metal_graph_streaming_expert_hotlist_add(
                hotlist[i][0],
                hotlist[i][1],
                max_entries - loaded,
                experts,
                priorities,
                counts,
                seen,
                &loaded)) {
            return false;
        }
    }
    *loaded_out = loaded;
    return true;
}

static uint32_t metal_graph_streaming_expert_preload_count(
        const ds4_gpu_graph *g,
        uint32_t             cache_budget) {
    if (!g || cache_budget == 0) return 0;
    uint32_t preload = g->streaming_preload_experts;
    if (preload == 0) {
        preload = cache_budget;
        const char *env = glm_graph_env_value(
                "DS4_ROCM_STREAMING_EXPERT_AUTO_PRELOAD_CAP",
                "DS4_METAL_STREAMING_EXPERT_AUTO_PRELOAD_CAP");
#ifdef DS4_ROCM_BUILD
        if (g_ds4_shape.variant == DS4_VARIANT_GLM52 &&
            (!env || !env[0])) {
            return 0;
        }
#endif
        /* Auto mode is a hot seed, not a request to synchronously fill the
         * whole cache. Large Flash caches can otherwise spend startup doing
         * thousands of preads into shared Metal buffers and trip the system
         * watchdog before decode begins. ROCm GLM52 uses indexed batch prefill
         * by default, which already populates the cache; explicit CLI preload
         * counts and auto-preload env caps bypass that default. */
        uint32_t cap = 4096;
        if (env && env[0]) {
            char *end = NULL;
            unsigned long v = strtoul(env, &end, 10);
            if (end != env && *end == '\0') {
                cap = v > UINT32_MAX ? UINT32_MAX : (uint32_t)v;
            }
        }
        if (cap != 0 && preload > cap) preload = cap;
    }
    if (preload > cache_budget) preload = cache_budget;
    const uint64_t max_possible = (uint64_t)DS4_N_LAYER * DS4_N_EXPERT;
    if ((uint64_t)preload > max_possible) preload = (uint32_t)max_possible;
    return preload;
}

static bool metal_graph_decode_set_hash_selected_override(
        const ds4_model         *model,
        const ds4_layer_weights *layer,
        uint32_t                 il,
        uint32_t                 token,
        uint64_t                 gate_tensor_bytes,
        uint64_t                 down_tensor_bytes,
        const ds4_gpu_graph     *g) {
    if (!layer->ffn_gate_tid2eid) return true;

    const bool q4_selected =
        metal_graph_decode_q4_selected_slots_expected(g,
                                                      layer,
                                                      gate_tensor_bytes,
                                                      down_tensor_bytes);
    const bool iq2_selected =
        metal_graph_decode_iq2_selected_slots_expected(g, layer);
    if (!q4_selected && !iq2_selected) {
        return true;
    }

    int selected[DS4_MAX_EXPERT_USED];
    int32_t selected_i32[DS4_MAX_EXPERT_USED];
    layer_hash_selected_experts(selected, model, layer, (int)token);
    for (uint32_t i = 0; i < DS4_N_EXPERT_USED; i++) {
        selected_i32[i] = (int32_t)selected[i];
    }
    if (g && g->ssd_streaming) {
        if (DS4_N_EXPERT == 0 ||
            gate_tensor_bytes % DS4_N_EXPERT != 0 ||
            down_tensor_bytes % DS4_N_EXPERT != 0) {
            return false;
        }
        const uint64_t gate_expert_bytes = gate_tensor_bytes / DS4_N_EXPERT;
        const uint64_t down_expert_bytes = down_tensor_bytes / DS4_N_EXPERT;
        const ds4_gpu_stream_expert_table table =
            graph_stream_expert_table_make(model,
                                           layer,
                                           il,
                                           gate_expert_bytes,
                                           down_expert_bytes);
        if (ds4_gpu_stream_expert_cache_begin_selected_load(
                    &table,
                    selected_i32,
                    DS4_N_EXPERT_USED) == 0) {
            return false;
        }
    }
    return ds4_gpu_routed_moe_set_selected_override(selected_i32, DS4_N_EXPERT_USED) != 0;
}

static bool metal_graph_decode_cpu_router(
        ds4_gpu_graph          *g,
        const ds4_model        *model,
        const ds4_layer_weights *layer,
        uint32_t                il,
        uint32_t                token) {
    const bool profile =
        getenv("DS4_METAL_PRO_Q4_CPU_ROUTER_PROFILE") != NULL ||
        getenv("DS4_METAL_STREAMING_IQ2_CPU_ROUTER_PROFILE") != NULL;
    const double t0 = profile ? now_sec() : 0.0;
    if (ds4_gpu_end_commands() == 0) return false;
    const double t_sync = profile ? now_sec() : 0.0;
    if (ds4_gpu_tensor_read(metal_graph_ffn_norm(g),
                            0,
                            g->cpu_router_norm,
                            (uint64_t)DS4_N_EMBD * sizeof(g->cpu_router_norm[0])) == 0) {
        return false;
    }
    const double t_read = profile ? now_sec() : 0.0;

    float logits[DS4_MAX_EXPERT];
    float probs[DS4_MAX_EXPERT];
    int selected[DS4_MAX_EXPERT_USED];
    int32_t selected_i32[DS4_MAX_EXPERT_USED];
    float weights[DS4_MAX_EXPERT_USED];

    matvec_any(logits, model, layer->ffn_gate_inp, g->cpu_router_norm);
    for (uint32_t i = 0; i < DS4_N_EXPERT; i++) {
        probs[i] = sqrtf(softplus_stable(logits[i]));
    }
    if (layer->ffn_gate_tid2eid) {
        layer_hash_selected_experts(selected, model, layer, (int)token);
        layer_hash_router_weights_from_probs(weights, probs, selected);
    } else {
        layer_topk_selected_experts_from_probs(selected, weights, model, layer, probs);
    }
    for (uint32_t i = 0; i < DS4_N_EXPERT_USED; i++) {
        selected_i32[i] = (int32_t)selected[i];
    }
    const double t_cpu = profile ? now_sec() : 0.0;

    if (ds4_gpu_tensor_write(metal_graph_router_logits(g),
                             0,
                             logits,
                             (uint64_t)DS4_N_EXPERT * sizeof(logits[0])) == 0 ||
        ds4_gpu_tensor_write(metal_graph_router_probs(g),
                             0,
                             probs,
                             (uint64_t)DS4_N_EXPERT * sizeof(probs[0])) == 0 ||
        ds4_gpu_tensor_write(metal_graph_router_selected(g),
                             0,
                             selected_i32,
                             (uint64_t)DS4_N_EXPERT_USED * sizeof(selected_i32[0])) == 0 ||
        ds4_gpu_tensor_write(metal_graph_router_weights(g),
                             0,
                             weights,
                             (uint64_t)DS4_N_EXPERT_USED * sizeof(weights[0])) == 0) {
        return false;
    }
    const double t_write = profile ? now_sec() : 0.0;
    if (ds4_gpu_begin_commands() == 0) return false;
    if (ds4_gpu_routed_moe_set_selected_override(selected_i32, DS4_N_EXPERT_USED) == 0) return false;

    if (profile) {
        fprintf(stderr,
                "ds4: Metal CPU router layer=%u gate=%s down=%s sync=%.3f ms read=%.3f ms cpu=%.3f ms write=%.3f ms total=%.3f ms\n",
                il,
                tensor_type_name(layer->ffn_gate_exps->type),
                tensor_type_name(layer->ffn_down_exps->type),
                (t_sync - t0) * 1000.0,
                (t_read - t_sync) * 1000.0,
                (t_cpu - t_read) * 1000.0,
                (t_write - t_cpu) * 1000.0,
                (t_write - t0) * 1000.0);
    }
    return true;
}

static bool metal_graph_use_iq2_selected_readahead_shared_delay(
        const ds4_gpu_graph *g) {
    return g &&
           g->ssd_streaming &&
           getenv("DS4_METAL_ENABLE_STREAMING_SELECTED_READAHEAD_SHARED_DELAY") != NULL &&
           getenv("DS4_METAL_DISABLE_STREAMING_SELECTED_READAHEAD_SHARED_DELAY") == NULL;
}

static bool metal_graph_decode_selected_readahead_override(
        ds4_gpu_graph            *g,
        const ds4_model          *model,
        const ds4_layer_weights  *layer,
        uint32_t                  il,
        uint64_t                  gate_expert_bytes,
        uint64_t                  down_expert_bytes) {
    if (!g || !model || !layer || !metal_graph_router_selected(g) ||
        DS4_N_EXPERT == 0 || DS4_N_EXPERT > DS4_MAX_EXPERT ||
        DS4_N_EXPERT_USED == 0 || DS4_N_EXPERT_USED > DS4_MAX_EXPERT_USED) {
        return false;
    }

    const bool profile =
        getenv("DS4_METAL_STREAMING_SELECTED_READAHEAD_PROFILE") != NULL;
    const double t0 = profile ? now_sec() : 0.0;
    if (ds4_gpu_end_commands() == 0) return false;
    const double t_sync = profile ? now_sec() : 0.0;

    int32_t selected_ids[DS4_MAX_EXPERT_USED] = {0};
    if (ds4_gpu_tensor_read(metal_graph_router_selected(g),
                            0,
                            selected_ids,
                            (uint64_t)DS4_N_EXPERT_USED * sizeof(selected_ids[0])) == 0) {
        return false;
    }
    const double t_read = profile ? now_sec() : 0.0;

    bool seen[DS4_MAX_EXPERT] = {0};
    uint32_t unique = 0;
    for (uint32_t i = 0; i < DS4_N_EXPERT_USED; i++) {
        if (selected_ids[i] < 0 || (uint32_t)selected_ids[i] >= DS4_N_EXPERT) {
            fprintf(stderr,
                    "ds4: Metal streaming selected readahead expert id %d is outside 0..%u at layer %u\n",
                    selected_ids[i],
                    DS4_N_EXPERT,
                    il);
            return false;
        }
        const uint32_t expert = (uint32_t)selected_ids[i];
        if (seen[expert]) continue;
        seen[expert] = true;
        unique++;

        const uint64_t expert_id = (uint64_t)expert;
        if (expert_id > UINT64_MAX / gate_expert_bytes ||
            expert_id > UINT64_MAX / down_expert_bytes) {
            fprintf(stderr, "ds4: Metal streaming selected readahead offset overflow\n");
            return false;
        }
        const uint64_t gate_rel = expert_id * gate_expert_bytes;
        const uint64_t down_rel = expert_id * down_expert_bytes;
        if (gate_rel > UINT64_MAX - layer->ffn_gate_exps->abs_offset ||
            gate_rel > UINT64_MAX - layer->ffn_up_exps->abs_offset ||
            down_rel > UINT64_MAX - layer->ffn_down_exps->abs_offset) {
            fprintf(stderr, "ds4: Metal streaming selected readahead offset overflow\n");
            return false;
        }

        metal_graph_stream_readahead_range_impl(model,
                                                layer->ffn_gate_exps->abs_offset + gate_rel,
                                                gate_expert_bytes,
                                                true);
        metal_graph_stream_readahead_range_impl(model,
                                                layer->ffn_up_exps->abs_offset + gate_rel,
                                                gate_expert_bytes,
                                                true);
        metal_graph_stream_readahead_range_impl(model,
                                                layer->ffn_down_exps->abs_offset + down_rel,
                                                down_expert_bytes,
                                                true);
    }
    const double t_hint = profile ? now_sec() : 0.0;

    if (ds4_gpu_routed_moe_set_selected_override(selected_ids,
                                                 DS4_N_EXPERT_USED) == 0) {
        return false;
    }
    const ds4_gpu_stream_expert_table table =
        graph_stream_expert_table_make(model,
                                       layer,
                                       il,
                                       gate_expert_bytes,
                                       down_expert_bytes);
    if (ds4_gpu_stream_expert_cache_begin_selected_load(
                &table,
                selected_ids,
                DS4_N_EXPERT_USED) == 0) {
        return false;
    }
    if (ds4_gpu_begin_commands() == 0) return false;
    const double t_done = profile ? now_sec() : 0.0;

    if (profile) {
        fprintf(stderr,
                "ds4: Metal streaming selected readahead layer=%u unique=%u sync=%.3f ms read=%.3f ms hint=%.3f ms resume=%.3f ms total=%.3f ms\n",
                il,
                unique,
                (t_sync - t0) * 1000.0,
                (t_read - t_sync) * 1000.0,
                (t_hint - t_read) * 1000.0,
                (t_done - t_hint) * 1000.0,
                (t_done - t0) * 1000.0);
    }
    return true;
}

static bool metal_graph_decode_cuda_selected_load(
        ds4_gpu_graph            *g,
        const ds4_model          *model,
        const ds4_layer_weights  *layer,
        uint32_t                  il,
        uint64_t                  gate_expert_bytes,
        uint64_t                  down_expert_bytes) {
#if !defined(DS4_ROCM_BUILD) && !defined(DS4_NO_GPU) && !defined(__APPLE__)
    if (!metal_graph_decode_cuda_selected_slots_expected(g, layer) ||
        !model ||
        !metal_graph_router_selected(g) ||
        DS4_N_EXPERT == 0 ||
        DS4_N_EXPERT > DS4_MAX_EXPERT ||
        DS4_N_EXPERT_USED == 0 ||
        DS4_N_EXPERT_USED > DS4_MAX_EXPERT_USED) {
        return false;
    }

    const bool profile =
        getenv("DS4_CUDA_STREAMING_EXPERT_CACHE_PROFILE") != NULL;
    const double t0 = profile ? now_sec() : 0.0;

    if (ds4_gpu_end_commands() == 0) return false;
    const double t_sync = profile ? now_sec() : 0.0;

    int32_t selected_ids[DS4_MAX_EXPERT_USED] = {0};
    bool ok = ds4_gpu_tensor_read(metal_graph_router_selected(g),
                                  0,
                                  selected_ids,
                                  (uint64_t)DS4_N_EXPERT_USED *
                                      sizeof(selected_ids[0])) != 0;
    const double t_read = profile ? now_sec() : 0.0;

    if (ok) {
        const ds4_gpu_stream_expert_table table =
            graph_stream_expert_table_make(model,
                                           layer,
                                           il,
                                           gate_expert_bytes,
                                           down_expert_bytes);
        ok = ds4_gpu_stream_expert_cache_begin_selected_load(
                    &table,
                    selected_ids,
                    DS4_N_EXPERT_USED) != 0;
    }
    const double t_load = profile ? now_sec() : 0.0;

    if (ds4_gpu_begin_commands() == 0) ok = false;
    const double t_done = profile ? now_sec() : 0.0;

    if (profile) {
        fprintf(stderr,
                "ds4: CUDA streaming selected load layer=%u sync=%.3f ms read=%.3f ms load=%.3f ms resume=%.3f ms total=%.3f ms\n",
                il,
                (t_sync - t0) * 1000.0,
                (t_read - t_sync) * 1000.0,
                (t_load - t_read) * 1000.0,
                (t_done - t_load) * 1000.0,
                (t_done - t0) * 1000.0);
    }
    return ok;
#else
    (void)g;
    (void)model;
    (void)layer;
    (void)il;
    (void)gate_expert_bytes;
    (void)down_expert_bytes;
    return false;
#endif
}

static bool metal_graph_cuda_stream_prefill_batch_selected_load(
        ds4_gpu_graph            *g,
        const ds4_model          *model,
        const ds4_layer_weights  *layer,
        uint32_t                  il,
        uint32_t                  n_tokens,
        uint64_t                  gate_expert_bytes,
        uint64_t                  down_expert_bytes) {
#if !defined(DS4_ROCM_BUILD) && !defined(DS4_NO_GPU) && !defined(__APPLE__)
    if (!metal_graph_decode_cuda_selected_slots_expected(g, layer) ||
        !model ||
        !metal_graph_batch_router_selected(g) ||
        n_tokens <= 1 ||
        DS4_N_EXPERT == 0 ||
        DS4_N_EXPERT_USED == 0 ||
        getenv("DS4_CUDA_DISABLE_STREAMING_PREFILL_BATCH_SELECTED_LOAD") != NULL) {
        return true;
    }

    if ((uint64_t)n_tokens > UINT64_MAX / (uint64_t)DS4_N_EXPERT_USED) {
        fprintf(stderr, "ds4: CUDA streaming prefill selected-id count overflow at layer %u\n", il);
        return false;
    }
    const uint64_t n_ids64 = (uint64_t)n_tokens * DS4_N_EXPERT_USED;
    if (n_ids64 == 0 || n_ids64 > SIZE_MAX / sizeof(int32_t)) {
        fprintf(stderr, "ds4: CUDA streaming prefill selected-id byte size overflow at layer %u\n", il);
        return false;
    }

    const bool profile =
        getenv("DS4_CUDA_STREAMING_PREFILL_BATCH_SELECTED_PROFILE") != NULL;
    const double t0 = profile ? now_sec() : 0.0;

    if (ds4_gpu_end_commands() == 0) return false;
    const double t_sync = profile ? now_sec() : 0.0;

    int32_t *selected_ids = xmalloc((size_t)n_ids64 * sizeof(selected_ids[0]));
    bool ok = ds4_gpu_tensor_read(metal_graph_batch_router_selected(g),
                                  0,
                                  selected_ids,
                                  n_ids64 * sizeof(selected_ids[0])) != 0;
    const double t_read = profile ? now_sec() : 0.0;
    if (ok) {
        const ds4_gpu_stream_expert_table table =
            graph_stream_expert_table_make(model,
                                           layer,
                                           il,
                                           gate_expert_bytes,
                                           down_expert_bytes);
        ok = ds4_gpu_stream_expert_cache_prepare_selected_batch(
                    &table,
                    selected_ids,
                    n_tokens,
                    DS4_N_EXPERT_USED) != 0;
    }
    free(selected_ids);
    const double t_load = profile ? now_sec() : 0.0;

    if (ds4_gpu_begin_commands() == 0) ok = false;
    const double t_done = profile ? now_sec() : 0.0;

    if (profile) {
        fprintf(stderr,
                "ds4: CUDA streaming prefill batch selected load layer=%u tokens=%u sync=%.3f ms read=%.3f ms load=%.3f ms resume=%.3f ms total=%.3f ms\n",
                il,
                n_tokens,
                (t_sync - t0) * 1000.0,
                (t_read - t_sync) * 1000.0,
                (t_load - t_read) * 1000.0,
                (t_done - t_load) * 1000.0,
                (t_done - t0) * 1000.0);
    }
    return ok;
#else
    (void)g;
    (void)model;
    (void)layer;
    (void)il;
    (void)n_tokens;
    (void)gate_expert_bytes;
    (void)down_expert_bytes;
    return true;
#endif
}

typedef struct metal_graph_selected_async_load {
    bool                      active;
    bool                      ok;
    /* Selected ids remain usable for a synchronous retry if the service
     * thread cannot stage the cache load without waiting on GPU work. */
    bool                      ids_ok;
    ds4_gpu_tensor           *router_selected;
    const ds4_model          *model;
    const ds4_layer_weights  *layer;
    uint32_t                  il;
    uint64_t                  event_value;
    uint64_t                  gate_expert_bytes;
    uint64_t                  down_expert_bytes;
    int32_t                   selected_ids[DS4_MAX_EXPERT_USED];
} metal_graph_selected_async_load;

static pthread_mutex_t g_metal_graph_selected_async_load_mutex =
    PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_metal_graph_selected_async_load_cond =
    PTHREAD_COND_INITIALIZER;
static pthread_cond_t g_metal_graph_selected_async_load_done_cond =
    PTHREAD_COND_INITIALIZER;
static pthread_t g_metal_graph_selected_async_load_thread;
static bool g_metal_graph_selected_async_load_thread_started = false;
static bool g_metal_graph_selected_async_load_has_job = false;
static bool g_metal_graph_selected_async_load_done = false;
static metal_graph_selected_async_load g_metal_graph_selected_async_load_job;

static void metal_graph_selected_async_load_run(
        metal_graph_selected_async_load *job) {
    job->ok = false;

    if (!job->router_selected || !job->model || !job->layer ||
        DS4_N_EXPERT == 0 || DS4_N_EXPERT > DS4_MAX_EXPERT ||
        DS4_N_EXPERT_USED == 0 || DS4_N_EXPERT_USED > DS4_MAX_EXPERT_USED) {
        return;
    }
    if (job->event_value != 0) {
#ifdef DS4_ROCM_BUILD
        if (ds4_gpu_tensor_read_after_selected_event(
                    job->router_selected,
                    0,
                    job->selected_ids,
                    (uint64_t)DS4_N_EXPERT_USED *
                        sizeof(job->selected_ids[0]),
                    job->event_value,
                    "selected-id async expert load") == 0) {
            return;
        }
#else
        if (ds4_gpu_wait_selected_readback_ready(job->event_value,
                                                 "selected-id async expert load") == 0) {
            return;
        }
        if (ds4_gpu_tensor_read(job->router_selected,
                                0,
                                job->selected_ids,
                                (uint64_t)DS4_N_EXPERT_USED *
                                    sizeof(job->selected_ids[0])) == 0) {
            return;
        }
#endif
    }
    for (uint32_t i = 0; i < DS4_N_EXPERT_USED; i++) {
        if (job->selected_ids[i] < 0 ||
            (uint32_t)job->selected_ids[i] >= DS4_N_EXPERT) {
            fprintf(stderr,
                    "ds4: Metal streaming async selected expert id %d is outside 0..%u at layer %u\n",
                    job->selected_ids[i],
                    DS4_N_EXPERT,
                    job->il);
            return;
        }
    }
    job->ids_ok = true;
    const ds4_gpu_stream_expert_table table =
        graph_stream_expert_table_make(job->model,
                                       job->layer,
                                       job->il,
                                       job->gate_expert_bytes,
                                       job->down_expert_bytes);
    if (ds4_gpu_stream_expert_cache_begin_selected_load(
                &table,
                job->selected_ids,
                DS4_N_EXPERT_USED) == 0) {
        return;
    }

    job->ok = true;
}

static void *metal_graph_selected_async_load_worker_main(void *arg) {
    (void)arg;
#ifdef __APPLE__
    /* The Metal cache paths must never wait on command buffers from this
     * thread while the main thread is encoding; register it so those waits
     * turn into load failures that the caller retries synchronously. */
    ds4_gpu_stream_expert_cache_note_service_thread();
#endif
    for (;;) {
        pthread_mutex_lock(&g_metal_graph_selected_async_load_mutex);
        while (!g_metal_graph_selected_async_load_has_job) {
            pthread_cond_wait(&g_metal_graph_selected_async_load_cond,
                              &g_metal_graph_selected_async_load_mutex);
        }
        metal_graph_selected_async_load job =
            g_metal_graph_selected_async_load_job;
        pthread_mutex_unlock(&g_metal_graph_selected_async_load_mutex);

        metal_graph_selected_async_load_run(&job);

        pthread_mutex_lock(&g_metal_graph_selected_async_load_mutex);
        g_metal_graph_selected_async_load_job = job;
        g_metal_graph_selected_async_load_has_job = false;
        g_metal_graph_selected_async_load_done = true;
        pthread_cond_signal(&g_metal_graph_selected_async_load_done_cond);
        pthread_mutex_unlock(&g_metal_graph_selected_async_load_mutex);
    }
    return NULL;
}

static bool metal_graph_selected_async_load_ensure_worker(void) {
    pthread_mutex_lock(&g_metal_graph_selected_async_load_mutex);
    if (g_metal_graph_selected_async_load_thread_started) {
        pthread_mutex_unlock(&g_metal_graph_selected_async_load_mutex);
        return true;
    }
    const int rc = pthread_create(&g_metal_graph_selected_async_load_thread,
                                  NULL,
                                  metal_graph_selected_async_load_worker_main,
                                  NULL);
    if (rc != 0) {
        pthread_mutex_unlock(&g_metal_graph_selected_async_load_mutex);
        fprintf(stderr,
                "ds4: failed to start Metal streaming async selected load worker: %s\n",
                strerror(rc));
        return false;
    }
    g_metal_graph_selected_async_load_thread_started = true;
    pthread_mutex_unlock(&g_metal_graph_selected_async_load_mutex);
    return true;
}

static DS4_MAYBE_UNUSED bool metal_graph_selected_async_load_start_tensor(
        metal_graph_selected_async_load *job,
        ds4_gpu_tensor                  *router_selected,
        const ds4_model                 *model,
        const ds4_layer_weights         *layer,
        uint32_t                         il,
        uint64_t                         event_value,
        uint64_t                         gate_expert_bytes,
        uint64_t                         down_expert_bytes) {
    if (!job || !router_selected || event_value == 0) return false;
    if (!metal_graph_selected_async_load_ensure_worker()) return false;
    memset(job, 0, sizeof(*job));
    job->router_selected = router_selected;
    job->model = model;
    job->layer = layer;
    job->il = il;
    job->event_value = event_value;
    job->gate_expert_bytes = gate_expert_bytes;
    job->down_expert_bytes = down_expert_bytes;

    pthread_mutex_lock(&g_metal_graph_selected_async_load_mutex);
    if (g_metal_graph_selected_async_load_has_job ||
        g_metal_graph_selected_async_load_done) {
        pthread_mutex_unlock(&g_metal_graph_selected_async_load_mutex);
        return false;
    }
    g_metal_graph_selected_async_load_job = *job;
    g_metal_graph_selected_async_load_job.ok = false;
    g_metal_graph_selected_async_load_has_job = true;
    pthread_cond_signal(&g_metal_graph_selected_async_load_cond);
    pthread_mutex_unlock(&g_metal_graph_selected_async_load_mutex);
    job->active = true;
    return true;
}

static DS4_MAYBE_UNUSED bool metal_graph_selected_async_load_start(
        metal_graph_selected_async_load *job,
        ds4_gpu_graph                   *g,
        const ds4_model                 *model,
        const ds4_layer_weights         *layer,
        uint32_t                         il,
        uint64_t                         event_value,
        uint64_t                         gate_expert_bytes,
        uint64_t                         down_expert_bytes) {
    return metal_graph_selected_async_load_start_tensor(
            job,
            g ? metal_graph_router_selected(g) : NULL,
            model,
            layer,
            il,
            event_value,
            gate_expert_bytes,
            down_expert_bytes);
}

static bool metal_graph_selected_async_load_finish(
        metal_graph_selected_async_load *job) {
    if (!job || !job->active) return false;
    pthread_mutex_lock(&g_metal_graph_selected_async_load_mutex);
    while (!g_metal_graph_selected_async_load_done) {
        pthread_cond_wait(&g_metal_graph_selected_async_load_done_cond,
                          &g_metal_graph_selected_async_load_mutex);
    }
    *job = g_metal_graph_selected_async_load_job;
    g_metal_graph_selected_async_load_done = false;
    pthread_mutex_unlock(&g_metal_graph_selected_async_load_mutex);
    job->active = false;
    if (!job->ok) return false;
    return ds4_gpu_routed_moe_set_selected_override(job->selected_ids,
                                                   DS4_N_EXPERT_USED) != 0;
}

#ifdef DS4_ROCM_BUILD
typedef struct rocm_graph_batch_selected_async_load {
    bool                      active;
    bool                      ok;
    const ds4_gpu_tensor     *selected;
    const ds4_model          *model;
    const ds4_layer_weights  *layer;
    uint32_t                  il;
    uint32_t                  n_tokens;
    uint64_t                  event_value;
    uint64_t                  gate_expert_bytes;
    uint64_t                  down_expert_bytes;
    int32_t                  *selected_ids;
} rocm_graph_batch_selected_async_load;

static pthread_mutex_t g_rocm_graph_batch_selected_async_load_mutex =
    PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_rocm_graph_batch_selected_async_load_cond =
    PTHREAD_COND_INITIALIZER;
static pthread_cond_t g_rocm_graph_batch_selected_async_load_done_cond =
    PTHREAD_COND_INITIALIZER;
static pthread_t g_rocm_graph_batch_selected_async_load_thread;
static bool g_rocm_graph_batch_selected_async_load_thread_started = false;
static bool g_rocm_graph_batch_selected_async_load_has_job = false;
static bool g_rocm_graph_batch_selected_async_load_done = false;
static rocm_graph_batch_selected_async_load
    g_rocm_graph_batch_selected_async_load_job;

static void rocm_graph_batch_selected_async_load_run(
        rocm_graph_batch_selected_async_load *job) {
    job->ok = false;
    if (!job->selected || !job->model || !job->layer || !job->selected_ids ||
        job->n_tokens <= 1 ||
        DS4_N_EXPERT == 0 || DS4_N_EXPERT > DS4_MAX_EXPERT ||
        DS4_N_EXPERT_USED == 0 || DS4_N_EXPERT_USED > DS4_MAX_EXPERT_USED) {
        return;
    }
    if (DS4_N_EXPERT_USED != 0 &&
        job->n_tokens > UINT64_MAX / DS4_N_EXPERT_USED) {
        return;
    }
    const uint64_t n_ids = (uint64_t)job->n_tokens * DS4_N_EXPERT_USED;
    if (n_ids > SIZE_MAX / sizeof(job->selected_ids[0])) return;
    if (ds4_gpu_tensor_read_after_selected_event(
                job->selected,
                0,
                job->selected_ids,
                n_ids * sizeof(job->selected_ids[0]),
                job->event_value,
                "prefill selected-id async expert load") == 0) {
        return;
    }
    for (uint64_t i = 0; i < n_ids; i++) {
        if (job->selected_ids[i] < 0 ||
            (uint32_t)job->selected_ids[i] >= DS4_N_EXPERT) {
            fprintf(stderr,
                    "ds4: ROCm streaming async batch selected expert id %d "
                    "is outside 0..%u at layer %u\n",
                    job->selected_ids[i],
                    DS4_N_EXPERT,
                    job->il);
            return;
        }
    }
    const ds4_gpu_stream_expert_table table =
        graph_stream_expert_table_make(job->model,
                                       job->layer,
                                       job->il,
                                       job->gate_expert_bytes,
                                       job->down_expert_bytes);
    if (ds4_gpu_stream_expert_cache_prepare_selected_batch(
                &table,
                job->selected_ids,
                job->n_tokens,
                DS4_N_EXPERT_USED) == 0) {
        return;
    }
    job->ok = true;
}

static void *rocm_graph_batch_selected_async_load_worker_main(void *arg) {
    (void)arg;
    for (;;) {
        pthread_mutex_lock(&g_rocm_graph_batch_selected_async_load_mutex);
        while (!g_rocm_graph_batch_selected_async_load_has_job) {
            pthread_cond_wait(&g_rocm_graph_batch_selected_async_load_cond,
                              &g_rocm_graph_batch_selected_async_load_mutex);
        }
        rocm_graph_batch_selected_async_load job =
            g_rocm_graph_batch_selected_async_load_job;
        pthread_mutex_unlock(&g_rocm_graph_batch_selected_async_load_mutex);

        rocm_graph_batch_selected_async_load_run(&job);

        pthread_mutex_lock(&g_rocm_graph_batch_selected_async_load_mutex);
        g_rocm_graph_batch_selected_async_load_job = job;
        g_rocm_graph_batch_selected_async_load_has_job = false;
        g_rocm_graph_batch_selected_async_load_done = true;
        pthread_cond_signal(&g_rocm_graph_batch_selected_async_load_done_cond);
        pthread_mutex_unlock(&g_rocm_graph_batch_selected_async_load_mutex);
    }
    return NULL;
}

static bool rocm_graph_batch_selected_async_load_ensure_worker(void) {
    pthread_mutex_lock(&g_rocm_graph_batch_selected_async_load_mutex);
    if (g_rocm_graph_batch_selected_async_load_thread_started) {
        pthread_mutex_unlock(&g_rocm_graph_batch_selected_async_load_mutex);
        return true;
    }
    const int rc = pthread_create(&g_rocm_graph_batch_selected_async_load_thread,
                                  NULL,
                                  rocm_graph_batch_selected_async_load_worker_main,
                                  NULL);
    if (rc != 0) {
        pthread_mutex_unlock(&g_rocm_graph_batch_selected_async_load_mutex);
        fprintf(stderr,
                "ds4: failed to start ROCm streaming async batch selected "
                "load worker: %s\n",
                strerror(rc));
        return false;
    }
    g_rocm_graph_batch_selected_async_load_thread_started = true;
    pthread_mutex_unlock(&g_rocm_graph_batch_selected_async_load_mutex);
    return true;
}

static bool rocm_graph_batch_selected_async_load_start(
        rocm_graph_batch_selected_async_load *job,
        const ds4_gpu_tensor                 *selected,
        const ds4_model                      *model,
        const ds4_layer_weights              *layer,
        uint32_t                              il,
        uint32_t                              n_tokens,
        uint64_t                              event_value,
        uint64_t                              gate_expert_bytes,
        uint64_t                              down_expert_bytes) {
    if (!job || !selected || event_value == 0 || n_tokens <= 1) return false;
    if (!rocm_graph_batch_selected_async_load_ensure_worker()) return false;
    if (DS4_N_EXPERT_USED != 0 &&
        n_tokens > UINT64_MAX / DS4_N_EXPERT_USED) {
        return false;
    }
    const uint64_t n_ids = (uint64_t)n_tokens * DS4_N_EXPERT_USED;
    if (n_ids > SIZE_MAX / sizeof(int32_t)) return false;
    memset(job, 0, sizeof(*job));
    job->selected_ids = xmalloc((size_t)n_ids * sizeof(job->selected_ids[0]));
    job->selected = selected;
    job->model = model;
    job->layer = layer;
    job->il = il;
    job->n_tokens = n_tokens;
    job->event_value = event_value;
    job->gate_expert_bytes = gate_expert_bytes;
    job->down_expert_bytes = down_expert_bytes;

    pthread_mutex_lock(&g_rocm_graph_batch_selected_async_load_mutex);
    if (g_rocm_graph_batch_selected_async_load_has_job ||
        g_rocm_graph_batch_selected_async_load_done) {
        pthread_mutex_unlock(&g_rocm_graph_batch_selected_async_load_mutex);
        free(job->selected_ids);
        memset(job, 0, sizeof(*job));
        return false;
    }
    g_rocm_graph_batch_selected_async_load_job = *job;
    g_rocm_graph_batch_selected_async_load_job.ok = false;
    g_rocm_graph_batch_selected_async_load_has_job = true;
    pthread_cond_signal(&g_rocm_graph_batch_selected_async_load_cond);
    pthread_mutex_unlock(&g_rocm_graph_batch_selected_async_load_mutex);
    job->active = true;
    return true;
}

static bool rocm_graph_batch_selected_async_load_finish(
        rocm_graph_batch_selected_async_load *job) {
    if (!job || !job->active) return false;
    pthread_mutex_lock(&g_rocm_graph_batch_selected_async_load_mutex);
    while (!g_rocm_graph_batch_selected_async_load_done) {
        pthread_cond_wait(&g_rocm_graph_batch_selected_async_load_done_cond,
                          &g_rocm_graph_batch_selected_async_load_mutex);
    }
    *job = g_rocm_graph_batch_selected_async_load_job;
    g_rocm_graph_batch_selected_async_load_done = false;
    pthread_mutex_unlock(&g_rocm_graph_batch_selected_async_load_mutex);
    const bool ok = job->ok;
    free(job->selected_ids);
    memset(job, 0, sizeof(*job));
    return ok;
}
#endif

static bool metal_graph_profile_router_selection(
        ds4_gpu_graph            *g,
        const ds4_layer_weights  *layer,
        uint32_t                  il,
        uint32_t                  pos) {
    if (!g_expert_profile.active) return true;
    if (!g || !layer || !metal_graph_router_selected(g) || !metal_graph_router_weights(g)) return false;

    if (ds4_gpu_end_commands() == 0) {
        fprintf(stderr,
                "ds4: failed to end Metal command batch for expert profile readback\n");
        return false;
    }

    int32_t selected[DS4_MAX_EXPERT_USED] = {0};
    float weights[DS4_MAX_EXPERT_USED] = {0};
    const bool read_ok =
        ds4_gpu_tensor_read(metal_graph_router_selected(g),
                            0,
                            selected,
                            (uint64_t)DS4_N_EXPERT_USED * sizeof(selected[0])) != 0 &&
        ds4_gpu_tensor_read(metal_graph_router_weights(g),
                            0,
                            weights,
                            (uint64_t)DS4_N_EXPERT_USED * sizeof(weights[0])) != 0;

    if (ds4_gpu_begin_commands() == 0) {
        fprintf(stderr,
                "ds4: failed to resume Metal command batch after expert profile readback\n");
        return false;
    }
    if (!read_ok) {
        fprintf(stderr, "ds4: failed to read Metal router tensors for expert profile\n");
        return false;
    }

    ds4_expert_profile_record(il,
                              pos,
                              selected,
                              weights,
                              layer->ffn_gate_tid2eid != NULL);
    return true;
}

/* Diagnostic skip-ablation for TP profiling (DS4_TP_ABLATE=chain[,chain]):
 * drops whole encode chains so their true in-situ cost shows up as a t/s
 * delta.  Output is semantically wrong while enabled; both ranks must set
 * the same value.  Chains: hcpre, router, kv, compidx. */
static bool metal_graph_tp_ablate(const char *chain) {
    static const char *env = NULL;
    static int init = 0;
    if (!init) {
        env = getenv("DS4_TP_ABLATE");
        init = 1;
    }
    return env && strstr(env, chain) != NULL;
}

static bool metal_graph_borrow_tensor_view(
        ds4_gpu_tensor       *view,
        const ds4_gpu_tensor *base,
        uint64_t              offset,
        uint64_t              bytes) {
    if (!view || !base || offset > base->bytes || bytes > base->bytes - offset) {
        return false;
    }
    memset(view, 0, sizeof(*view));
    view->ptr = (char *)base->ptr + offset;
    view->bytes = bytes;
    view->owner = 0;
    view->device_id = base->device_id;
    return true;
}

static bool metal_graph_cuda_tp_ep_finish_reduce(
        ds4_gpu_graph *g,
        int            home_tier,
        int            partner_tier,
        bool           direct_return,
        uint64_t       return_bytes,
        bool           combine) {
    bool ok;
    if (direct_return) {
        ok = ds4_gpu_tensor_wait_xdev_default(
                g->routed_down_by_tier[partner_tier], home_tier) != 0;
    } else {
        ok = ds4_gpu_tensor_copy_xdev_default(
                g->tp_peer_tmp_by_tier[home_tier],
                g->routed_down_by_tier[partner_tier],
                return_bytes) != 0;
    }
    if (!combine) return ok;
    if (ok && g->cuda_tp_ep_pack_exact) {
        ok = ds4_gpu_routed_moe_owned_packed_combine_tensor(
                metal_graph_routed_out(g),
                metal_graph_routed_down(g),
                g->tp_peer_tmp_by_tier[home_tier],
                metal_graph_router_selected(g),
                DS4_N_EMBD,
                DS4_N_EXPERT / 2u) != 0;
    } else if (ok) {
        ok = ds4_gpu_routed_moe_owned_slots_combine_tensor(
                metal_graph_routed_out(g),
                metal_graph_routed_down(g),
                g->tp_peer_tmp_by_tier[home_tier],
                metal_graph_router_selected(g),
                DS4_N_EMBD,
                DS4_N_EXPERT / 2u) != 0;
    }
    return ok;
}

typedef enum {
    METAL_DECODE_LAYER_FULL = 0,
    METAL_DECODE_LAYER_TO_FFN,
    METAL_DECODE_LAYER_FROM_ATTN_PRE_TO_FFN,
    METAL_DECODE_LAYER_FROM_ATTN_PRE_TO_ATTN,
    METAL_DECODE_LAYER_TO_QKV,
    METAL_DECODE_LAYER_FROM_QKV_TO_ATTN,
    METAL_DECODE_LAYER_FROM_QA_KV_RAW_TO_SHARED_MID,
    METAL_DECODE_LAYER_FROM_KV_STORE_TO_ATTN,
    METAL_DECODE_LAYER_FROM_ATTN_TO_FFN,
    METAL_DECODE_LAYER_TO_ROUTER,
    METAL_DECODE_LAYER_TO_SHARED_MID,
    METAL_DECODE_LAYER_FROM_ROUTER,
} metal_decode_layer_phase;

static bool metal_graph_encode_decode_layer_phase(
        ds4_gpu_graph  *g,
        const ds4_model        *model,
        const ds4_layer_weights *layer,
        uint32_t                il,
        uint32_t                pos,
        ds4_gpu_tensor       *raw_cache,
        uint32_t                raw_cap,
        uint32_t                raw_row,
        uint32_t                n_raw,
        int                     token,
        metal_decode_layer_phase phase) {
    /* switch to this layer's home tier before any Class P
     * accessor reads. Single-tier (placement == NULL): no-op. */
    if (g->placement) {
        const int this_tier = g->placement[il + 1];
        if (!metal_graph_set_active_tier_decode(g, this_tier)) return false;
    }
    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const uint64_t mix_hc = 2ull * DS4_N_HC + (uint64_t)DS4_N_HC * DS4_N_HC;
    const uint64_t q_rank = layer->attn_q_a->dim[1];
    const uint64_t q_dim = (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM;
    const uint32_t n_groups = DS4_N_OUT_GROUP;
    const uint32_t group_heads = DS4_N_HEAD / n_groups;
    const uint32_t group_dim = DS4_N_HEAD_DIM * group_heads;
    const uint32_t rank = DS4_N_LORA_O;
    const uint32_t shared_dim = (uint32_t)layer->ffn_gate_shexp->dim[1];
    const uint64_t expert_in_dim = layer->ffn_gate_exps->dim[0];
    const uint64_t expert_mid_dim = layer->ffn_gate_exps->dim[1];
    const uint64_t down_in_dim = layer->ffn_down_exps->dim[0];
    const uint64_t routed_out_dim = layer->ffn_down_exps->dim[1];
    const uint64_t gate_row_bytes = routed_expert_row_bytes(layer->ffn_gate_exps);
    const uint64_t gate_expert_bytes = expert_mid_dim * gate_row_bytes;
    const uint64_t down_row_bytes = routed_expert_row_bytes(layer->ffn_down_exps);
    const uint64_t down_expert_bytes = routed_out_dim * down_row_bytes;
    const bool compressed = ds4_layer_compress_ratio(il) != 0;
    const float freq_base = layer_rope_freq_base(il);
    const float freq_scale = layer_rope_freq_scale(il);
    const float ext_factor = compressed && DS4_ROPE_SCALE_FACTOR > 1.0f ? 1.0f : 0.0f;
    float attn_factor = 1.0f;
    if (ext_factor != 0.0f && freq_scale > 0.0f) {
        attn_factor /= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    const bool qkv_rms_fused = !metal_graph_use_reference_qkv_norm();
    const int cuda_tp_home_tier = g->active_tier;
    const int cuda_tp_partner_tier = g->cuda_tp_decode
        ? metal_graph_cuda_tp_partner_tier(cuda_tp_home_tier) : -1;
    const bool tp_split_attn = g->tp_world == 2;
    const uint32_t tp_heads = tp_split_attn ?
        (uint32_t)DS4_N_HEAD / 2u : (uint32_t)DS4_N_HEAD;
    const uint32_t tp_head0 = tp_split_attn ? g->tp_rank * tp_heads : 0;

    bool ok = true;
    const bool decode_stage_profile = metal_graph_decode_stage_profile_enabled(il);
    double decode_stage_t0 = decode_stage_profile ? now_sec() : 0.0;
#define DS4_METAL_PROFILE_DECODE_STAGE(name) do { \
        if (ok && decode_stage_profile) { \
            ok = metal_graph_layer_stage_profile_boundary("decode", (name), il, pos, 1, &decode_stage_t0); \
        } \
    } while (0)
    const bool tp_ablate_hcpre = metal_graph_tp_ablate("hcpre");
    if (phase != METAL_DECODE_LAYER_FROM_ROUTER) {
    const bool fuse_hc_norm =
        DS4_N_HC == 4 &&
        !metal_graph_use_reference_hc_decode() &&
        !metal_graph_use_reference_hc_norm_decode();
    const bool stop_before_attn =
        phase == METAL_DECODE_LAYER_FROM_ATTN_PRE_TO_ATTN ||
        phase == METAL_DECODE_LAYER_FROM_QKV_TO_ATTN ||
        phase == METAL_DECODE_LAYER_FROM_KV_STORE_TO_ATTN;
    const bool resume_after_qkv =
        phase == METAL_DECODE_LAYER_FROM_QKV_TO_ATTN ||
        phase == METAL_DECODE_LAYER_FROM_KV_STORE_TO_ATTN;
    const bool resume_after_qa_kv_raw =
        phase == METAL_DECODE_LAYER_FROM_QA_KV_RAW_TO_SHARED_MID;
    const bool resume_after_kv_store =
        phase == METAL_DECODE_LAYER_FROM_KV_STORE_TO_ATTN;
    const bool resume_after_attn =
        phase == METAL_DECODE_LAYER_FROM_ATTN_TO_FFN;
    bool cuda_tp_attn_heads_active = false;
    bool attn_inv_rope_done = resume_after_attn;
    if (phase != METAL_DECODE_LAYER_FROM_ATTN_PRE_TO_FFN &&
        phase != METAL_DECODE_LAYER_FROM_ATTN_PRE_TO_ATTN &&
        phase != METAL_DECODE_LAYER_FROM_QKV_TO_ATTN &&
        phase != METAL_DECODE_LAYER_FROM_QA_KV_RAW_TO_SHARED_MID &&
        phase != METAL_DECODE_LAYER_FROM_KV_STORE_TO_ATTN &&
        !resume_after_attn) {
    if (ok && !tp_ablate_hcpre) {
        ok = ds4_gpu_rms_norm_plain_tensor(metal_graph_flat_hc(g), metal_graph_cur_hc(g), (uint32_t)hc_dim, DS4_RMS_EPS) != 0;
        if (ok) ok = metal_graph_matmul_plain_tensor(metal_graph_hc_mix(g), model, layer->hc_attn_fn,
                                                     hc_dim, mix_hc, metal_graph_flat_hc(g), 1);
    }
    if (ok && fuse_hc_norm) {
        ok = ds4_gpu_hc_split_weighted_sum_norm_tensor(metal_graph_attn_cur(g),
                                                         metal_graph_attn_norm(g),
                                                         metal_graph_hc_split(g),
                                                         metal_graph_hc_mix(g),
                                                         metal_graph_cur_hc(g),
                                                         model->map,
                                                         model->size,
                                                         layer->hc_attn_scale->abs_offset,
                                                         layer->hc_attn_base->abs_offset,
                                                         layer->attn_norm->abs_offset,
                                                         DS4_N_EMBD,
                                                         DS4_N_HC,
                                                         DS4_N_HC_SINKHORN_ITER,
                                                         DS4_HC_EPS,
                                                         DS4_RMS_EPS) != 0;
        if (ok) {
            ok = metal_graph_check_hc_norm_fusion("attn",
                                                  metal_graph_attn_cur(g),
                                                  metal_graph_attn_norm(g),
                                                  metal_graph_hc_mix(g),
                                                  metal_graph_cur_hc(g),
                                                  model,
                                                  layer->hc_attn_scale->abs_offset,
                                                  layer->hc_attn_base->abs_offset,
                                                  layer->attn_norm->abs_offset,
                                                  il,
                                                  pos);
        }
    } else if (ok) {
        ok = metal_graph_decode_hc_pre(metal_graph_attn_cur(g),
                                       metal_graph_hc_split(g),
                                       metal_graph_hc_mix(g),
                                       metal_graph_cur_hc(g),
                                       model,
                                       layer->hc_attn_scale->abs_offset,
                                       layer->hc_attn_base->abs_offset);
    }
    DS4_METAL_PROFILE_DECODE_STAGE("attn_hc_pre");
    if (ok) {
        metal_graph_debug_dump_tensor("hc_attn_pre_mixes", metal_graph_hc_mix(g), mix_hc, il, pos);
        metal_graph_debug_dump_tensor("hc_attn_pre_weights", metal_graph_hc_pre(g), DS4_N_HC, il, pos);
        metal_graph_debug_dump_tensor("hc_attn_pre_post_weights", metal_graph_hc_post(g), DS4_N_HC, il, pos);
        metal_graph_debug_dump_tensor("hc_attn_pre_comb", metal_graph_hc_comb(g), (uint64_t)DS4_N_HC * DS4_N_HC, il, pos);
    }
    if (ok) {
        metal_graph_debug_dump_tensor("hc_attn_pre", metal_graph_attn_cur(g), DS4_N_EMBD, il, pos);
    }
    if (ok && !fuse_hc_norm) ok = ds4_gpu_rms_norm_weight_tensor(metal_graph_attn_norm(g), metal_graph_attn_cur(g),
                                                                   model->map, model->size,
                                                                   layer->attn_norm->abs_offset,
                                                                   DS4_N_EMBD, DS4_RMS_EPS) != 0;
    DS4_METAL_PROFILE_DECODE_STAGE("attn_norm");
    if (ok) {
        metal_graph_debug_dump_tensor("attn_norm", metal_graph_attn_norm(g), DS4_N_EMBD, il, pos);
    }
    if (phase == METAL_DECODE_LAYER_TO_QKV) return ok;
    }
    if (!resume_after_attn) {
    if (!resume_after_qkv) {
    bool qkv_pair_projected = resume_after_qa_kv_raw;
    if (!resume_after_qa_kv_raw && ok && qkv_rms_fused &&
        g->cuda_qkv_pair && !metal_graph_use_reference_qkv_pair_proj()) {
        qkv_pair_projected = ds4_gpu_matmul_q8_0_pair_tensor(
                metal_graph_qr(g),
                metal_graph_kv_raw(g),
                model->map,
                model->size,
                layer->attn_q_a->abs_offset,
                layer->attn_kv->abs_offset,
                DS4_N_EMBD,
                q_rank,
                DS4_N_HEAD_DIM,
                metal_graph_attn_norm(g),
                1) != 0;
    }
    if (!resume_after_qa_kv_raw && ok && !qkv_pair_projected) ok = ds4_gpu_matmul_q8_0_tensor(metal_graph_qr(g), model->map, model->size,
                                                                     layer->attn_q_a->abs_offset,
                                                                     DS4_N_EMBD, q_rank,
                                                                     metal_graph_attn_norm(g), 1) != 0;
    if (ok) {
        metal_graph_debug_dump_tensor("q_lora", metal_graph_qr(g), q_rank, il, pos);
    }
    const bool kvnorm_dump = metal_graph_debug_wants("KVnorm", il, pos);
    bool kv_rope_fused = false;
    if (qkv_rms_fused) {
        if (!resume_after_qa_kv_raw && ok && !qkv_pair_projected) ok = ds4_gpu_matmul_q8_0_tensor(metal_graph_kv_raw(g), model->map, model->size,
                                                                         layer->attn_kv->abs_offset,
                                                                         DS4_N_EMBD, DS4_N_HEAD_DIM,
                                                                         metal_graph_attn_norm(g), 1) != 0;
        if (ok) {
            metal_graph_debug_dump_tensor("KVraw", metal_graph_kv_raw(g), DS4_N_HEAD_DIM, il, pos);
        }
        if (ok && g->cuda_qkv_kv_rope_fuse && !kvnorm_dump && DS4_N_HEAD_KV == 1u) {
            ok = ds4_gpu_dsv4_qkv_rms_norm_rows_kv_rope_tensor(
                    metal_graph_qr_norm(g),
                    metal_graph_qr(g),
                    model->map,
                    model->size,
                    layer->attn_q_a_norm->abs_offset,
                    (uint32_t)q_rank,
                    metal_graph_kv(g),
                    metal_graph_kv_raw(g),
                    layer->attn_kv_a_norm->abs_offset,
                    DS4_N_HEAD_DIM,
                    1,
                    DS4_N_HEAD_KV,
                    DS4_N_HEAD_DIM,
                    DS4_N_ROT,
                    pos,
                    compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                    false,
                    freq_base,
                    freq_scale,
                    ext_factor,
                    attn_factor,
                    DS4_ROPE_YARN_BETA_FAST,
                    DS4_ROPE_YARN_BETA_SLOW,
                    DS4_RMS_EPS) != 0;
            kv_rope_fused = ok;
        } else if (ok) {
            ok = ds4_gpu_dsv4_qkv_rms_norm_rows_tensor(metal_graph_qr_norm(g),
                                                         metal_graph_qr(g),
                                                         model->map,
                                                         model->size,
                                                         layer->attn_q_a_norm->abs_offset,
                                                         (uint32_t)q_rank,
                                                         metal_graph_kv(g),
                                                         metal_graph_kv_raw(g),
                                                         layer->attn_kv_a_norm->abs_offset,
                                                         DS4_N_HEAD_DIM,
                                                         1,
                                                         DS4_RMS_EPS) != 0;
        }
    } else {
        if (ok) ok = ds4_gpu_rms_norm_weight_tensor(metal_graph_qr_norm(g), metal_graph_qr(g),
                                                      model->map, model->size,
                                                      layer->attn_q_a_norm->abs_offset,
                                                      (uint32_t)q_rank, DS4_RMS_EPS) != 0;
    }
    if (ok) {
        metal_graph_debug_dump_tensor("q_lora_norm", metal_graph_qr_norm(g), q_rank, il, pos);
    }
    if (qkv_rms_fused && ok && !kv_rope_fused) {
        metal_graph_debug_dump_tensor("KVnorm", metal_graph_kv(g), DS4_N_HEAD_DIM, il, pos);
    }
    /* Phase B head slice: under the real TP split this rank computes only
     * its heads [tp_head0, tp_head0 + tp_heads) end to end — q_b rows, the
     * per-head norm/rope, the attention core and its owned output groups.
     * q and heads hold the owned half compactly at the buffer base; the
     * head range lines up with the output-group split (32 heads = 4 of the
     * 8 groups). */
    uint64_t tp_q_row_bytes = 0;
    if (ok) ok = metal_graph_dense_quant_row_bytes(layer->attn_q_b,
                                                   q_rank,
                                                   &tp_q_row_bytes);
    const uint64_t tp_q_rows_off =
        (uint64_t)tp_head0 * DS4_N_HEAD_DIM * tp_q_row_bytes;
    if (ok) ok = metal_graph_matmul_dense_quant_abs(metal_graph_q(g),
                                                    model,
                                                    layer->attn_q_b,
                                                    layer->attn_q_b->abs_offset + tp_q_rows_off,
                                                    q_rank,
                                                    (uint64_t)tp_heads * DS4_N_HEAD_DIM,
                                                    metal_graph_qr_norm(g),
                                                    1);
    if (ok) {
        metal_graph_debug_dump_tensor("Qraw", metal_graph_q(g), q_dim, il, pos);
    }
    const bool decode_q_norm_debug = metal_graph_debug_wants("Qnorm", il, pos);
    bool decode_q_norm_rope_fused = false;
    if (ok && !decode_q_norm_debug) {
        decode_q_norm_rope_fused =
            ds4_gpu_head_rms_norm_rope_tail_tensor(metal_graph_q(g),
                                                   1,
                                                   tp_heads,
                                                   DS4_N_HEAD_DIM,
                                                   DS4_N_ROT,
                                                   pos,
                                                   compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                                                   false,
                                                   freq_base,
                                                   freq_scale,
                                                   ext_factor,
                                                   attn_factor,
                                                   DS4_ROPE_YARN_BETA_FAST,
                                                   DS4_ROPE_YARN_BETA_SLOW,
                                                   DS4_RMS_EPS) != 0;
    }
    if (!decode_q_norm_rope_fused) {
        if (ok) ok = ds4_gpu_head_rms_norm_tensor(metal_graph_q(g), 1, tp_heads, DS4_N_HEAD_DIM, DS4_RMS_EPS) != 0;
        if (ok) {
            metal_graph_debug_dump_tensor("Qnorm", metal_graph_q(g), q_dim, il, pos);
        }
        if (ok) ok = ds4_gpu_rope_tail_tensor(metal_graph_q(g), 1, tp_heads, DS4_N_HEAD_DIM,
                                                DS4_N_ROT, pos,
                                                compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                                                false, freq_base, freq_scale, ext_factor, attn_factor,
                                                DS4_ROPE_YARN_BETA_FAST, DS4_ROPE_YARN_BETA_SLOW) != 0;
    }
    DS4_METAL_PROFILE_DECODE_STAGE("q_path");
    if (ok) {
        metal_graph_debug_dump_tensor("Qcur", metal_graph_q(g), q_dim, il, pos);
    }
    if (!qkv_rms_fused) {
        if (ok) ok = metal_graph_matmul_dense_quant_tensor(metal_graph_kv_raw(g),
                                                           model,
                                                           layer->attn_kv,
                                                           DS4_N_EMBD,
                                                           DS4_N_HEAD_DIM,
                                                           metal_graph_attn_norm(g),
                                                           1);
        if (ok) {
            metal_graph_debug_dump_tensor("KVraw", metal_graph_kv_raw(g), DS4_N_HEAD_DIM, il, pos);
        }
        if (ok) ok = ds4_gpu_rms_norm_weight_tensor(metal_graph_kv(g), metal_graph_kv_raw(g),
                                                      model->map, model->size,
                                                      layer->attn_kv_a_norm->abs_offset,
                                                      DS4_N_HEAD_DIM, DS4_RMS_EPS) != 0;
        if (ok) {
            metal_graph_debug_dump_tensor("KVnorm", metal_graph_kv(g), DS4_N_HEAD_DIM, il, pos);
        }
    }
    const bool tp_ablate_kv = metal_graph_tp_ablate("kv");
    if (ok && !tp_ablate_kv && !kv_rope_fused) {
        ok = ds4_gpu_rope_tail_tensor(metal_graph_kv(g), 1,
                                      DS4_N_HEAD_KV, DS4_N_HEAD_DIM,
                                      DS4_N_ROT, pos,
                                      compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                                      false, freq_base, freq_scale,
                                      ext_factor, attn_factor,
                                      DS4_ROPE_YARN_BETA_FAST,
                                      DS4_ROPE_YARN_BETA_SLOW) != 0;
    }
    if (ok) {
        metal_graph_debug_dump_tensor("KVrope", metal_graph_kv(g), DS4_N_HEAD_DIM, il, pos);
    }
    }
    if (!resume_after_kv_store) {
        /* The common no-debug path may fuse KV RMS with RoPE above. KV
         * storage starts here after metal_graph_kv(g) contains the RoPE row. */
        if (ok) ok = metal_graph_decode_kv_store(metal_graph_kv(g), raw_cache, raw_cap, raw_row);
        if (ok) ok = metal_graph_cuda_tp_attn_cache_sync_raw_row(g, il, raw_row);
        DS4_METAL_PROFILE_DECODE_STAGE("kv_path");
        if (ok) {
            metal_graph_debug_dump_tensor("KVcur", metal_graph_kv(g), DS4_N_HEAD_DIM, il, pos);
        }
    }

    uint32_t n_comp = 0;
    ds4_gpu_tensor *comp_cache = NULL;
    ds4_gpu_tensor *comp_selected = NULL;
    uint32_t n_selected = 0;
    double decode_index_stage_t0 = 0.0;
    const bool decode_index_stage_profile = g->decode_index_stage_profile;
    if (ok && compressed) {
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        const uint32_t coff = ratio == 4 ? 2u : 1u;
        const uint32_t comp_width = coff * DS4_N_HEAD_DIM;
        const bool emit = ((pos + 1u) % ratio) == 0u;
        if (!layer->attn_compressor_kv || !layer->attn_compressor_gate ||
            !layer->attn_compressor_ape || !layer->attn_compressor_norm ||
            layer->attn_compressor_kv->type != DS4_TENSOR_F16 ||
            layer->attn_compressor_gate->type != DS4_TENSOR_F16 ||
            layer->attn_compressor_kv->dim[0] != DS4_N_EMBD ||
            layer->attn_compressor_gate->dim[0] != DS4_N_EMBD ||
            layer->attn_compressor_kv->dim[1] != comp_width ||
            layer->attn_compressor_gate->dim[1] != comp_width) {
            fprintf(stderr, "ds4: Metal graph compressor expects paired F16 compressor projections\n");
            ok = false;
        }
        if (ok && emit && g->layer_n_comp[il] >= g->layer_comp_cap[il]) {
            fprintf(stderr, "ds4: Metal graph compressed KV cache capacity exceeded at layer %u\n", il);
            ok = false;
        }
        bool comp_state_already_stored = false;
        if (ok && !metal_graph_use_reference_compressor_pair_proj()) {
            const int fused_store =
                ds4_gpu_matmul_f16_pair_compressor_store_tensor(
                        metal_graph_comp_kv_cur(g),
                        metal_graph_comp_sc_cur(g),
                        g->layer_attn_state_kv[il],
                        g->layer_attn_state_score[il],
                        model->map,
                        model->size,
                        layer->attn_compressor_kv->abs_offset,
                        layer->attn_compressor_gate->abs_offset,
                        layer->attn_compressor_ape->abs_offset,
                        layer->attn_compressor_ape->type,
                        DS4_N_EMBD,
                        comp_width,
                        metal_graph_attn_norm(g),
                        ratio,
                        pos);
            if (fused_store < 0) {
                ok = false;
            } else if (fused_store > 0) {
                comp_state_already_stored = true;
            } else {
                ok = ds4_gpu_matmul_f16_pair_tensor(metal_graph_comp_kv_cur(g),
                                                      metal_graph_comp_sc_cur(g),
                                                      model->map,
                                                      model->size,
                                                      layer->attn_compressor_kv->abs_offset,
                                                      layer->attn_compressor_gate->abs_offset,
                                                      DS4_N_EMBD,
                                                      comp_width,
                                                      metal_graph_attn_norm(g),
                                                      1) != 0;
            }
        } else {
            if (ok) ok = ds4_gpu_matmul_f16_tensor(metal_graph_comp_kv_cur(g), model->map, model->size,
                                                     layer->attn_compressor_kv->abs_offset,
                                                     DS4_N_EMBD, comp_width,
                                                     metal_graph_attn_norm(g), 1) != 0;
            if (ok) ok = ds4_gpu_matmul_f16_tensor(metal_graph_comp_sc_cur(g), model->map, model->size,
                                                     layer->attn_compressor_gate->abs_offset,
                                                     DS4_N_EMBD, comp_width,
                                                     metal_graph_attn_norm(g), 1) != 0;
        }
        DS4_METAL_PROFILE_DECODE_STAGE("compressor_proj");
        const uint32_t comp_row = g->layer_n_comp[il];
        if (ok) ok = ds4_gpu_compressor_update_tensor(metal_graph_comp_kv_cur(g),
                                                        metal_graph_comp_sc_cur(g),
                                                        g->layer_attn_state_kv[il],
                                                        g->layer_attn_state_score[il],
                                                        metal_graph_attn_comp_update_target(g, il),
                                                        model->map,
                                                        model->size,
                                                        layer->attn_compressor_ape->abs_offset,
                                                        layer->attn_compressor_ape->type,
                                                        layer->attn_compressor_norm->abs_offset,
                                                        layer->attn_compressor_norm->type,
                                                        DS4_N_HEAD_DIM,
                                                        ratio,
                                                        pos,
                                                        metal_graph_attn_comp_update_row(comp_row),
                                                        DS4_N_ROT,
                                                        compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                                                        freq_base,
                                                        freq_scale,
                                                        ext_factor,
                                                        attn_factor,
                                                        DS4_ROPE_YARN_BETA_FAST,
                                                        DS4_ROPE_YARN_BETA_SLOW,
                                                        DS4_RMS_EPS,
                                                        comp_state_already_stored) != 0;
        DS4_METAL_PROFILE_DECODE_STAGE("compressor_update");
        if (ok && emit) {
            ds4_gpu_tensor *comp_row_view = metal_graph_attn_comp_row_view(g, il, comp_row);
            if (!comp_row_view) {
                ok = false;
            } else {
                ok = ds4_gpu_dsv4_fp8_kv_quantize_tensor(comp_row_view, 1, DS4_N_HEAD_DIM, DS4_N_ROT) != 0;
                if (ok) {
                    metal_graph_debug_dump_tensor("KVcompress", comp_row_view, DS4_N_HEAD_DIM, il, pos);
                }
            }
            ds4_gpu_tensor_free(comp_row_view);
            DS4_METAL_PROFILE_DECODE_STAGE("compressor_quantize");
            if (ok) ok = metal_graph_commit_attn_comp_stage(g, il, comp_row, 1);
            DS4_METAL_PROFILE_DECODE_STAGE("compressor_commit");
        }
        if (ok && emit) g->layer_n_comp[il]++;

        if (ok && ratio == 4) {
            const uint32_t index_width = coff * DS4_N_INDEXER_HEAD_DIM;
            if (!layer->indexer_compressor_kv || !layer->indexer_compressor_gate ||
                !layer->indexer_compressor_ape || !layer->indexer_compressor_norm ||
                layer->indexer_compressor_kv->type != DS4_TENSOR_F16 ||
                layer->indexer_compressor_gate->type != DS4_TENSOR_F16 ||
                layer->indexer_compressor_kv->dim[0] != DS4_N_EMBD ||
                layer->indexer_compressor_gate->dim[0] != DS4_N_EMBD ||
                layer->indexer_compressor_kv->dim[1] != index_width ||
                layer->indexer_compressor_gate->dim[1] != index_width) {
                fprintf(stderr, "ds4: Metal graph indexer compressor expects paired F16 projections\n");
                ok = false;
            }
            if (ok && emit && g->layer_n_index_comp[il] >= g->layer_comp_cap[il]) {
                fprintf(stderr, "ds4: Metal graph indexer compressed KV cache capacity exceeded at layer %u\n", il);
                ok = false;
            }
            bool index_state_already_stored = false;
            if (ok && !metal_graph_use_reference_compressor_pair_proj()) {
                const int fused_store =
                    ds4_gpu_matmul_f16_pair_compressor_store_tensor(
                            metal_graph_comp_kv_cur(g),
                            metal_graph_comp_sc_cur(g),
                            g->layer_index_state_kv[il],
                            g->layer_index_state_score[il],
                            model->map,
                            model->size,
                            layer->indexer_compressor_kv->abs_offset,
                            layer->indexer_compressor_gate->abs_offset,
                            layer->indexer_compressor_ape->abs_offset,
                            layer->indexer_compressor_ape->type,
                            DS4_N_EMBD,
                            index_width,
                            metal_graph_attn_norm(g),
                            ratio,
                            pos);
                if (fused_store < 0) {
                    ok = false;
                } else if (fused_store > 0) {
                    index_state_already_stored = true;
                } else {
                    ok = ds4_gpu_matmul_f16_pair_tensor(metal_graph_comp_kv_cur(g),
                                                          metal_graph_comp_sc_cur(g),
                                                          model->map,
                                                          model->size,
                                                          layer->indexer_compressor_kv->abs_offset,
                                                          layer->indexer_compressor_gate->abs_offset,
                                                          DS4_N_EMBD,
                                                          index_width,
                                                          metal_graph_attn_norm(g),
                                                          1) != 0;
                }
            } else {
                if (ok) ok = ds4_gpu_matmul_f16_tensor(metal_graph_comp_kv_cur(g), model->map, model->size,
                                                         layer->indexer_compressor_kv->abs_offset,
                                                         DS4_N_EMBD, index_width,
                                                         metal_graph_attn_norm(g), 1) != 0;
                if (ok) ok = ds4_gpu_matmul_f16_tensor(metal_graph_comp_sc_cur(g), model->map, model->size,
                                                         layer->indexer_compressor_gate->abs_offset,
                                                         DS4_N_EMBD, index_width,
                                                         metal_graph_attn_norm(g), 1) != 0;
            }
            DS4_METAL_PROFILE_DECODE_STAGE("indexer_compressor_proj");
            const uint32_t index_row = g->layer_n_index_comp[il];
            if (ok) ok = ds4_gpu_compressor_update_tensor(metal_graph_comp_kv_cur(g),
                                                            metal_graph_comp_sc_cur(g),
                                                            g->layer_index_state_kv[il],
                                                            g->layer_index_state_score[il],
                                                            g->layer_index_comp_cache[il],
                                                            model->map,
                                                            model->size,
                                                            layer->indexer_compressor_ape->abs_offset,
                                                            layer->indexer_compressor_ape->type,
                                                            layer->indexer_compressor_norm->abs_offset,
                                                            layer->indexer_compressor_norm->type,
                                                            DS4_N_INDEXER_HEAD_DIM,
                                                            ratio,
                                                            pos,
                                                            index_row,
                                                            DS4_N_ROT,
                                                            compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                                                            freq_base,
                                                            freq_scale,
                                                            ext_factor,
                                                            attn_factor,
                                                            DS4_ROPE_YARN_BETA_FAST,
                                                            DS4_ROPE_YARN_BETA_SLOW,
                                                            DS4_RMS_EPS,
                                                            index_state_already_stored) != 0;
            DS4_METAL_PROFILE_DECODE_STAGE("indexer_compressor_update");
            if (ok && emit) {
#if defined(__APPLE__)
                ds4_gpu_tensor *index_row_view = ds4_gpu_tensor_view(
                        g->layer_index_comp_cache[il],
                        (uint64_t)index_row * DS4_N_INDEXER_HEAD_DIM * sizeof(float),
                        (uint64_t)DS4_N_INDEXER_HEAD_DIM * sizeof(float));
                if (!index_row_view) {
                    ok = false;
                } else {
                    ok = ds4_gpu_dsv4_indexer_qat_tensor(index_row_view,
                                                          1,
                                                          DS4_N_INDEXER_HEAD_DIM) != 0;
                }
                ds4_gpu_tensor_free(index_row_view);
#else
                ds4_gpu_tensor index_row_view;
                if (!metal_graph_borrow_tensor_view(
                        &index_row_view,
                        g->layer_index_comp_cache[il],
                        (uint64_t)index_row * DS4_N_INDEXER_HEAD_DIM * sizeof(float),
                        (uint64_t)DS4_N_INDEXER_HEAD_DIM * sizeof(float))) {
                    ok = false;
                } else {
                    ok = ds4_gpu_dsv4_indexer_qat_tensor(&index_row_view,
                                                          1,
                                                          DS4_N_INDEXER_HEAD_DIM) != 0;
                }
#endif
                DS4_METAL_PROFILE_DECODE_STAGE("indexer_compressor_qat");
            }
            if (ok && emit) g->layer_n_index_comp[il]++;
            const uint32_t decode_sparse_threshold =
                metal_graph_decode_indexer_sparse_threshold(g);
            if (ok &&
                g->layer_n_comp[il] > decode_sparse_threshold &&
                g->layer_n_index_comp[il] > DS4_N_INDEXER_TOP_K) {
                const uint64_t indexer_q_dim = (uint64_t)DS4_N_INDEXER_HEAD * DS4_N_INDEXER_HEAD_DIM;
                if (!layer->indexer_attn_q_b ||
                    !tensor_type_is_f16_or_q8_0(layer->indexer_attn_q_b->type) ||
                    layer->indexer_attn_q_b->dim[0] != q_rank ||
                    layer->indexer_attn_q_b->dim[1] != indexer_q_dim) {
                    fprintf(stderr, "ds4: Metal graph indexer q projection expects F16 or Q8_0 weights\n");
                    ok = false;
                }
                if (ok && (!layer->indexer_proj ||
                           layer->indexer_proj->type != DS4_TENSOR_F16 ||
                           layer->indexer_proj->dim[0] != DS4_N_EMBD ||
                           layer->indexer_proj->dim[1] != DS4_N_INDEXER_HEAD)) {
                    fprintf(stderr, "ds4: Metal graph indexer weight projection expects F16 weights\n");
                    ok = false;
                }
                if (ok) ok = metal_graph_matmul_plain_tensor(metal_graph_indexer_q(g),
                                                              model,
                                                              layer->indexer_attn_q_b,
                                                              q_rank,
                                                              indexer_q_dim,
                                                              metal_graph_qr_norm(g),
                                                              1);
                if (ok) ok = ds4_gpu_rope_tail_tensor(metal_graph_indexer_q(g), 1,
                                                        DS4_N_INDEXER_HEAD,
                                                        DS4_N_INDEXER_HEAD_DIM,
                                                        DS4_N_ROT,
                                                        pos,
                                                        compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                                                        false,
                                                        freq_base,
                                                        freq_scale,
                                                        ext_factor,
                                                        attn_factor,
                                                        DS4_ROPE_YARN_BETA_FAST,
                                                        DS4_ROPE_YARN_BETA_SLOW) != 0;
                if (ok) ok = ds4_gpu_dsv4_indexer_qat_tensor(metal_graph_indexer_q(g),
                                                              DS4_N_INDEXER_HEAD,
                                                              DS4_N_INDEXER_HEAD_DIM) != 0;
                if (ok) ok = ds4_gpu_matmul_f16_tensor(metal_graph_indexer_weights(g), model->map, model->size,
                                                         layer->indexer_proj->abs_offset,
                                                         DS4_N_EMBD, DS4_N_INDEXER_HEAD,
                                                         metal_graph_attn_norm(g), 1) != 0;
                const float index_scale = 1.0f / sqrtf((float)(DS4_N_INDEXER_HEAD_DIM * DS4_N_INDEXER_HEAD));
                if (ok && decode_index_stage_profile) {
                    ok = metal_graph_indexer_stage_profile_boundary(NULL,
                                                                    il,
                                                                    pos,
                                                                    1,
                                                                    g->layer_n_index_comp[il],
                                                                    &decode_index_stage_t0);
                }
                if (ok) ok = ds4_gpu_indexer_score_one_tensor(metal_graph_indexer_scores(g),
                                                                metal_graph_indexer_q(g),
                                                                metal_graph_indexer_weights(g),
                                                                g->layer_index_comp_cache[il],
                                                                g->layer_n_index_comp[il],
                                                                DS4_N_INDEXER_HEAD,
                                                                DS4_N_INDEXER_HEAD_DIM,
                                                                index_scale) != 0;
                if (ok && decode_index_stage_profile) {
                    ok = metal_graph_indexer_stage_profile_boundary("decode_score",
                                                                    il,
                                                                    pos,
                                                                    1,
                                                                    g->layer_n_index_comp[il],
                                                                    &decode_index_stage_t0);
                }
                if (ok) ok = ds4_gpu_indexer_topk_tensor(metal_graph_comp_selected(g),
                                                           metal_graph_indexer_scores(g),
                                                           g->layer_n_index_comp[il],
                                                           1,
                                                           DS4_N_INDEXER_TOP_K) != 0;
                if (ok && decode_index_stage_profile) {
                    ok = metal_graph_indexer_stage_profile_boundary("decode_topk",
                                                                    il,
                                                                    pos,
                                                                    1,
                                                                    g->layer_n_index_comp[il],
                                                                    &decode_index_stage_t0);
                }
                /* Decode used to materialize a dense compressed-row mask and
                 * call the generic gathered FlashAttention wrapper below.
                 * That wrapper scans every compressed row and rejects long
                 * contexts once raw+compressed rows exceed 8192.  Ratio-4 DS4
                 * attention is sparse after indexer top-k, so use the private
                 * indexed attention kernel instead: it scans only SWA raw rows
                 * plus the selected compressed rows, matching prefill and
                 * avoiding the long-context decode failure. */
                if (ok) {
                    comp_selected = metal_graph_comp_selected(g);
                    /*
                     * Contract: the indexer top-k is fixed by the model config
                     * and must remain the full 512 rows.  Do not reduce this for
                     * throughput benchmarks.
                     *
                     * Why: the indexer is not just an implementation detail.  It
                     * decides which compressed memory rows are visible to the
                     * attention kernel.  If we keep only 128/256 rows, the later
                     * indexed-attention math may be perfectly computed, but it is
                     * computed over the wrong candidate set: rows ranked 257-512
                     * are removed before softmax/PV can use them.  Those rows may
                     * carry weak-but-necessary evidence for retrieval, name/number
                     * recall, or long-context disambiguation.  The error is
                     * therefore semantic/algorithmic, not the acceptable kind of
                     * local numerical drift caused by a different reduction order
                     * or Tensor/NAX precision.
                     *
                     * Short prompt tests, first-token agreement, or even a small
                     * official-vector set can miss this because many prompts do
                     * not need the tail of the 512 selected compressed rows.  The
                     * failure appears only when the model needs information that
                     * fell below the reduced cutoff.  Optimizations belong inside
                     * the score/top-k/attention implementation while preserving
                     * DS4_N_INDEXER_TOP_K.
                     */
                    n_selected = DS4_N_INDEXER_TOP_K < g->layer_n_index_comp[il]
                        ? DS4_N_INDEXER_TOP_K
                        : g->layer_n_index_comp[il];
                }
            }
        }

        n_comp = g->layer_n_comp[il];
        comp_cache = g->layer_attn_comp_cache[il];
    }
    DS4_METAL_PROFILE_DECODE_STAGE("compressor_indexer");

    if (stop_before_attn) return ok;
    if (ok) {
        const uint32_t raw_start = metal_graph_raw_start_for_span(g, pos, n_raw);
        const bool indexed_attention = n_comp != 0 && comp_selected != NULL && n_selected != 0;
        const bool cuda_tp_attn_heads_requested = g->cuda_tp_attn_heads;
        const uint32_t cuda_tp_heads = DS4_N_HEAD / 2u;
        const bool cuda_tp_attn_local_cache =
            metal_graph_cuda_tp_attn_cache_dup_layer_ready(g, il);
        cuda_tp_attn_heads_active =
            cuda_tp_attn_heads_requested &&
            cuda_tp_partner_tier >= 0 &&
            g_gpu_peer_ok[cuda_tp_partner_tier][cuda_tp_home_tier] &&
            (DS4_N_HEAD % 2u) == 0u &&
            (n_groups % 2u) == 0u &&
            g->q_by_tier[cuda_tp_partner_tier] &&
            g->heads_by_tier[cuda_tp_partner_tier] &&
            !metal_graph_debug_wants("kqv_out", il, pos) &&
            !metal_graph_debug_wants("kqv_back", il, pos);
        if (cuda_tp_attn_heads_requested && !cuda_tp_attn_heads_active) {
            fprintf(stderr,
                    "ds4: CUDA decode TP cannot split attention heads for tier %d "
                    "(partner=%d heads=%u peer=%d)\n",
                    cuda_tp_home_tier,
                    cuda_tp_partner_tier,
                    DS4_N_HEAD,
                    cuda_tp_partner_tier >= 0 ?
                        g_gpu_peer_ok[cuda_tp_partner_tier][cuda_tp_home_tier] : 0);
            ok = false;
        }
        if (ok && cuda_tp_attn_heads_active) {
            const uint64_t tp_head_bytes = (uint64_t)cuda_tp_heads * DS4_N_HEAD_DIM * sizeof(float);
            ds4_gpu_tensor q_peer_src;
            ds4_gpu_tensor q_peer_dst;
            ds4_gpu_tensor peer_heads_tail;
            ok = metal_graph_borrow_tensor_view(&q_peer_src,
                                                metal_graph_q(g),
                                                tp_head_bytes,
                                                tp_head_bytes) &&
                 metal_graph_borrow_tensor_view(&q_peer_dst,
                                                g->q_by_tier[cuda_tp_partner_tier],
                                                0,
                                                tp_head_bytes) &&
                 metal_graph_borrow_tensor_view(&peer_heads_tail,
                                                g->heads_by_tier[cuda_tp_partner_tier],
                                                tp_head_bytes,
                                                tp_head_bytes);
            if (ok) {
                ok = ds4_gpu_tensor_copy_xdev(&q_peer_dst, &q_peer_src,
                                              tp_head_bytes) != 0;
            }
            ds4_gpu_tensor *peer_raw_cache = cuda_tp_attn_local_cache
                ? g->layer_raw_cache_tp[il] : raw_cache;
            ds4_gpu_tensor *peer_comp_cache = cuda_tp_attn_local_cache
                ? g->layer_attn_comp_cache_tp[il] : comp_cache;
            ds4_gpu_tensor *peer_selected = comp_selected;
            if (ok && indexed_attention && cuda_tp_attn_local_cache) {
                peer_selected = g->comp_selected_by_tier[cuda_tp_partner_tier];
                ok = peer_selected &&
                     ds4_gpu_tensor_copy_xdev(peer_selected,
                                              comp_selected,
                                              (uint64_t)n_selected * sizeof(int32_t)) != 0;
            }
            if (ok && !cuda_tp_attn_local_cache) {
                ok = ds4_gpu_tensor_wait_xdev(raw_cache, cuda_tp_partner_tier) != 0;
            }
            if (ok) ok = ds4_gpu_set_current_device(cuda_tp_partner_tier) == 0;
            if (ok && indexed_attention) {
                ok = ds4_gpu_attention_indexed_mixed_batch_heads_tensor(
                        &peer_heads_tail,
                        model->map,
                        model->size,
                        layer->attn_sinks->abs_offset + (uint64_t)cuda_tp_heads * sizeof(float),
                        &q_peer_dst,
                        peer_raw_cache,
                        peer_comp_cache,
                        metal_graph_attn_comp_cache_is_f16(),
                        peer_selected,
                        1,
                        pos,
                        n_raw,
                        raw_cap,
                        raw_start,
                        n_comp,
                        n_selected,
                        g->raw_window,
                        ds4_layer_compress_ratio(il),
                        cuda_tp_heads,
                        DS4_N_HEAD_DIM) != 0;
            } else if (ok) {
                ok = ds4_gpu_attention_decode_heads_tensor(
                        &peer_heads_tail,
                        model->map,
                        model->size,
                        layer->attn_sinks->abs_offset + (uint64_t)cuda_tp_heads * sizeof(float),
                        &q_peer_dst,
                        peer_raw_cache,
                        n_raw,
                        raw_cap,
                        raw_start,
                        n_comp ? peer_comp_cache : NULL,
                        metal_graph_attn_comp_cache_is_f16(),
                        n_comp,
                        NULL,
                        0,
                        cuda_tp_heads,
                        DS4_N_HEAD_DIM) != 0;
            }
            if (ok) {
                ok = ds4_gpu_rope_tail_tensor(&peer_heads_tail,
                                              1, cuda_tp_heads, DS4_N_HEAD_DIM,
                                              DS4_N_ROT, pos,
                                              compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                                              true,
                                              freq_base,
                                              freq_scale,
                                              ext_factor,
                                              attn_factor,
                                              DS4_ROPE_YARN_BETA_FAST,
                                              DS4_ROPE_YARN_BETA_SLOW) != 0;
            }
            if (ok) ok = ds4_gpu_set_current_device(cuda_tp_home_tier) == 0;
            if (ok && indexed_attention) {
                ok = ds4_gpu_attention_indexed_mixed_batch_heads_tensor(
                        metal_graph_heads(g),
                        model->map,
                        model->size,
                        layer->attn_sinks->abs_offset,
                        metal_graph_q(g),
                        raw_cache,
                        g->layer_attn_comp_cache[il],
                        metal_graph_attn_comp_cache_is_f16(),
                        comp_selected,
                        1,
                        pos,
                        n_raw,
                        raw_cap,
                        raw_start,
                        n_comp,
                        n_selected,
                        g->raw_window,
                        ds4_layer_compress_ratio(il),
                        cuda_tp_heads,
                        DS4_N_HEAD_DIM) != 0;
            } else if (ok) {
                ok = ds4_gpu_attention_decode_heads_tensor(metal_graph_heads(g),
                                                             model->map, model->size,
                                                             layer->attn_sinks->abs_offset,
                                                             metal_graph_q(g), raw_cache, n_raw,
                                                             raw_cap,
                                                             raw_start,
                                                             n_comp ? comp_cache : NULL,
                                                             metal_graph_attn_comp_cache_is_f16(),
                                                             n_comp,
                                                             NULL,
                                                             0,
                                                             cuda_tp_heads, DS4_N_HEAD_DIM) != 0;
            }
            if (ok) {
                ok = ds4_gpu_rope_tail_tensor(metal_graph_heads(g),
                                                1, cuda_tp_heads, DS4_N_HEAD_DIM,
                                                DS4_N_ROT, pos,
                                                compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                                                true,
                                                freq_base,
                                                freq_scale,
                                                ext_factor,
                                                attn_factor,
                                                DS4_ROPE_YARN_BETA_FAST,
                                                DS4_ROPE_YARN_BETA_SLOW) != 0;
            }
            if (ok && indexed_attention && decode_index_stage_profile) {
                ok = metal_graph_indexer_stage_profile_boundary("decode_attention",
                                                                il,
                                                                pos,
                                                                1,
                                                                n_comp,
                                                                &decode_index_stage_t0);
            }
        } else if (ok && indexed_attention) {
            ok = ds4_gpu_attention_indexed_mixed_batch_heads_tensor(
                    metal_graph_heads(g),
                    model->map,
                    model->size,
                    layer->attn_sinks->abs_offset + (uint64_t)tp_head0 * (layer->attn_sinks->bytes / DS4_N_HEAD),
                    metal_graph_q(g),
                    raw_cache,
                    g->layer_attn_comp_cache[il],
                    metal_graph_attn_comp_cache_is_f16(),
                    comp_selected,
                    1,
                    pos,
                    n_raw,
                    raw_cap,
                    raw_start,
                    n_comp,
                    n_selected,
                    g->raw_window,
                    ds4_layer_compress_ratio(il),
                    tp_heads,
                    DS4_N_HEAD_DIM) != 0;
            if (ok && decode_index_stage_profile) {
                ok = metal_graph_indexer_stage_profile_boundary("decode_attention",
                                                                il,
                                                                pos,
                                                                1,
                                                                n_comp,
                                                                &decode_index_stage_t0);
            }
        } else {
            ok = ds4_gpu_attention_decode_heads_tensor(metal_graph_heads(g),
                                                         model->map, model->size,
                                                         layer->attn_sinks->abs_offset + (uint64_t)tp_head0 * (layer->attn_sinks->bytes / DS4_N_HEAD),
                                                         metal_graph_q(g), raw_cache, n_raw,
                                                         raw_cap,
                                                         raw_start,
                                                         n_comp ? comp_cache : NULL,
                                                         metal_graph_attn_comp_cache_is_f16(),
                                                         n_comp,
                                                         NULL,
                                                         0,
                                                         tp_heads, DS4_N_HEAD_DIM) != 0;
        }
    }
    }
    if (ok && !cuda_tp_attn_heads_active && !attn_inv_rope_done) {
        ok = ds4_gpu_rope_tail_tensor(metal_graph_heads(g),
                                      1, tp_heads, DS4_N_HEAD_DIM,
                                      DS4_N_ROT, pos,
                                      compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                                      true,
                                      freq_base,
                                      freq_scale,
                                      ext_factor,
                                      attn_factor,
                                      DS4_ROPE_YARN_BETA_FAST,
                                      DS4_ROPE_YARN_BETA_SLOW) != 0;
    }
    DS4_METAL_PROFILE_DECODE_STAGE("attn_inv_rope");
    if (ok && !cuda_tp_attn_heads_active) {
        metal_graph_debug_dump_tensor("kqv_back", metal_graph_heads(g), q_dim, il, pos);
    }
    ds4_gpu_tensor *cuda_tp_attn_peer = NULL;
    bool cuda_tp_attn_hc_fused = false;
    const bool cuda_tp_attn_requested = g->cuda_tp_attn;
    const bool cuda_tp_attn =
        cuda_tp_attn_requested &&
        !metal_graph_directional_steering_attn_enabled(g) &&
        cuda_tp_partner_tier >= 0 &&
        (n_groups % 2u) == 0u;
    ds4_gpu_tensor *tp_attn_a = NULL;   /* rank partials consumed directly */
    ds4_gpu_tensor *tp_attn_b = NULL;   /* by the HC expand */
    const bool fuse_attn_out_hc =
        !cuda_tp_attn &&
        g->tp_world < 2 &&
        layer->attn_output_a->type == DS4_TENSOR_Q8_0 &&
        layer->attn_output_b->type == DS4_TENSOR_Q8_0 &&
        !metal_graph_directional_steering_attn_enabled(g) &&
        !metal_graph_use_reference_attn_out_hc();
    const bool fuse_tp_attn_out_hc =
        cuda_tp_attn &&
        !metal_graph_use_reference_attn_out_hc() &&
        g->cuda_tp_attn_out_hc_fuse;
    if (ok && cuda_tp_attn_requested && !cuda_tp_attn) {
        fprintf(stderr,
                "ds4: CUDA decode TP cannot split attention output for tier %d "
                "(partner=%d groups=%u)\n",
                cuda_tp_home_tier, cuda_tp_partner_tier, n_groups);
        ok = false;
    }
    if (ok && cuda_tp_attn) {
        const uint32_t tp_groups = n_groups / 2u;
        const uint64_t tp_heads_bytes = (uint64_t)tp_groups * group_dim * sizeof(float);
        const uint64_t tp_heads_off = (uint64_t)tp_groups * group_dim * sizeof(float);
        const bool cuda_tp_attn_peer_read =
            !cuda_tp_attn_heads_active &&
            g->cuda_tp_attn_peer_read &&
            g_gpu_peer_ok[cuda_tp_partner_tier][cuda_tp_home_tier];
        ds4_gpu_tensor peer_heads_dst_view;
        ds4_gpu_tensor peer_heads_src_view;
        ds4_gpu_tensor *peer_heads_dst = NULL;
        ds4_gpu_tensor *peer_heads_src = NULL;
        if (cuda_tp_attn_heads_active) {
            peer_heads_src = g->heads_by_tier[cuda_tp_partner_tier];
        } else if (!cuda_tp_attn_peer_read) {
            ok = metal_graph_borrow_tensor_view(
                    &peer_heads_dst_view,
                    g->heads_by_tier[cuda_tp_partner_tier],
                    tp_heads_off,
                    tp_heads_bytes);
            if (ok) peer_heads_dst = &peer_heads_dst_view;
        }
        if (ok) {
            if (cuda_tp_attn_heads_active) {
                peer_heads_src = g->heads_by_tier[cuda_tp_partner_tier];
            } else if (cuda_tp_attn_peer_read) {
                peer_heads_src = metal_graph_heads(g);
            } else {
                ok = metal_graph_borrow_tensor_view(&peer_heads_src_view,
                                                    metal_graph_heads(g),
                                                    tp_heads_off,
                                                    tp_heads_bytes);
                if (ok) peer_heads_src = &peer_heads_src_view;
            }
        }
        ds4_gpu_tensor *peer_heads = cuda_tp_attn_peer_read ?
            metal_graph_heads(g) : g->heads_by_tier[cuda_tp_partner_tier];
        ok = (cuda_tp_attn_heads_active || cuda_tp_attn_peer_read || peer_heads_dst) &&
             peer_heads_src && peer_heads &&
             g->attn_low_by_tier[cuda_tp_partner_tier] &&
             g->attn_out_by_tier[cuda_tp_partner_tier] &&
             g->tp_peer_tmp_by_tier[cuda_tp_home_tier];
        if (ok && !cuda_tp_attn_heads_active && !cuda_tp_attn_peer_read) {
            ok = ds4_gpu_tensor_copy_xdev(peer_heads_dst, peer_heads_src,
                                          tp_heads_bytes) != 0;
        } else if (ok && cuda_tp_attn_peer_read) {
            ok = ds4_gpu_tensor_wait_xdev(peer_heads_src, cuda_tp_partner_tier) != 0;
        }
        if (ok) ok = ds4_gpu_set_current_device(cuda_tp_partner_tier) == 0;
        if (ok) {
            ok = ds4_gpu_attention_output_q8_tp_tensor(
                    g->attn_out_by_tier[cuda_tp_partner_tier],
                    g->attn_low_by_tier[cuda_tp_partner_tier],
                    model->map,
                    model->size,
                    layer->attn_output_a->abs_offset,
                    layer->attn_output_b->abs_offset,
                    group_dim,
                    rank,
                    n_groups,
                    tp_groups,
                    tp_groups,
                    DS4_N_EMBD,
                    peer_heads) != 0;
        }
        if (ok) ok = ds4_gpu_set_current_device(cuda_tp_home_tier) == 0;
        if (ok && fuse_tp_attn_out_hc) {
            ok = ds4_gpu_attention_output_low_q8_tensor(
                    metal_graph_attn_low(g),
                    model->map,
                    model->size,
                    layer->attn_output_a->abs_offset,
                    group_dim,
                    rank,
                    tp_groups,
                    metal_graph_heads(g)) != 0;
        } else if (ok) {
            ok = ds4_gpu_attention_output_q8_tp_tensor(
                    metal_graph_attn_out(g),
                    metal_graph_attn_low(g),
                    model->map,
                    model->size,
                    layer->attn_output_a->abs_offset,
                    layer->attn_output_b->abs_offset,
                    group_dim,
                    rank,
                    n_groups,
                    0,
                    tp_groups,
                    DS4_N_EMBD,
                    metal_graph_heads(g)) != 0;
        }
        if (ok) {
            ok = ds4_gpu_tensor_copy_xdev(g->tp_peer_tmp_by_tier[cuda_tp_home_tier],
                                          g->attn_out_by_tier[cuda_tp_partner_tier],
                                          (uint64_t)DS4_N_EMBD * sizeof(float)) != 0;
            if (ok) cuda_tp_attn_peer = g->tp_peer_tmp_by_tier[cuda_tp_home_tier];
        }
        if (ok && fuse_tp_attn_out_hc) {
            ok = ds4_gpu_matmul_q8_0_kslice_hc_expand_add_tensor(
                    metal_graph_after_attn_hc(g),
                    metal_graph_attn_out(g),
                    model->map,
                    model->size,
                    layer->attn_output_b->abs_offset,
                    (uint64_t)n_groups * rank,
                    DS4_N_EMBD,
                    0,
                    (uint64_t)tp_groups * rank,
                    metal_graph_attn_low(g),
                    cuda_tp_attn_peer,
                    metal_graph_cur_hc(g),
                    metal_graph_hc_split(g),
                    DS4_N_EMBD,
                    DS4_N_HC) != 0;
            if (ok) cuda_tp_attn_hc_fused = true;
        }
    } else if (ok && fuse_attn_out_hc) {
        ok = ds4_gpu_attention_output_low_q8_tensor(metal_graph_attn_low(g),
                                                      model->map,
                                                      model->size,
                                                      layer->attn_output_a->abs_offset,
                                                      group_dim,
                                                      rank,
                                                      n_groups,
                                                      metal_graph_heads(g)) != 0;
        if (ok) {
            ok = ds4_gpu_matmul_q8_0_hc_expand_tensor(metal_graph_after_attn_hc(g),
                                                        metal_graph_attn_out(g),
                                                        model->map,
                                                        model->size,
                                                        layer->attn_output_b->abs_offset,
                                                        (uint64_t)n_groups * rank,
                                                        DS4_N_EMBD,
                                                        metal_graph_attn_low(g),
                                                        metal_graph_cur_hc(g),
                                                        metal_graph_hc_split(g),
                                                        DS4_N_EMBD,
                                                        DS4_N_HC) != 0;
        }
    } else if (ok && g->tp_world == 2) {
        /* Group-sliced attention output: this rank computes its half of the
         * output groups and the matching k-window of the expand projection,
         * leaving a partial block output in the gate slot. */
        const uint32_t tp_groups = n_groups / 2;
        ok = metal_graph_attention_output_dense_quant_tp(
                g->tp_out[il * DS4_TP_GATES_PER_LAYER + DS4_TP_GATE_ATTN],
                metal_graph_attn_low(g),
                g,
                model,
                layer->attn_output_a,
                layer->attn_output_b,
                group_dim, rank,
                n_groups,
                g->tp_rank * tp_groups, tp_groups,
                DS4_N_EMBD,
                metal_graph_heads(g));
    } else if (ok && layer->attn_output_a->type != DS4_TENSOR_Q8_0) {
        ds4_gpu_tensor *attn_out_dst = g->tp_world == 2 ?
                g->tp_out[il * DS4_TP_GATES_PER_LAYER + DS4_TP_GATE_ATTN] : metal_graph_attn_out(g);
        ok = metal_graph_attention_output_dense_quant_low(metal_graph_attn_low(g),
                                                          g,
                                                          model,
                                                          layer->attn_output_a,
                                                          group_dim,
                                                          rank,
                                                          0,
                                                          n_groups,
                                                          metal_graph_heads(g));
        if (ok) ok = metal_graph_matmul_dense_quant_tensor(attn_out_dst,
                                                           model,
                                                           layer->attn_output_b,
                                                           (uint64_t)n_groups * rank,
                                                           DS4_N_EMBD,
                                                           metal_graph_attn_low(g),
                                                           1);
    } else if (ok) {
        ds4_gpu_tensor *attn_out_dst = g->tp_world == 2 ?
                g->tp_out[il * DS4_TP_GATES_PER_LAYER + DS4_TP_GATE_ATTN] : metal_graph_attn_out(g);
        ok = ds4_gpu_attention_output_q8_batch_tensor(attn_out_dst,
                                                        metal_graph_attn_low(g),
                                                        metal_graph_batch_group_tmp(g),
                                                        metal_graph_batch_low_tmp(g),
                                                        model->map,
                                                        model->size,
                                                        layer->attn_output_a->abs_offset,
                                                        layer->attn_output_b->abs_offset,
                                                        group_dim, rank,
                                                        n_groups, DS4_N_EMBD,
                                                        metal_graph_heads(g), 1) != 0;
    }
    if (ok && g->tp_world == 2) {
        /* Gate ATTN: exchange the attention block output with the peer and
         * rebuild the canonical sum (rank0 first, then rank1) in attn_out
         * on both ranks — identical expression on both machines keeps them
         * bit-exact. */
        const uint32_t slot = il * DS4_TP_GATES_PER_LAYER + DS4_TP_GATE_ATTN;
        ok = ds4_gpu_tp_gate_encode(il, DS4_TP_GATE_ATTN) != 0;
        if (ok) {
            ds4_gpu_tensor *first = g->tp_rank == 0 ? g->tp_out[slot] : g->tp_in[slot];
            if (metal_graph_directional_steering_attn_enabled(g)) {
                ds4_gpu_tensor *second = g->tp_rank == 0 ? g->tp_in[slot] : g->tp_out[slot];
                ok = ds4_gpu_add_tensor(metal_graph_attn_out(g), first, second, DS4_N_EMBD) != 0;
            } else {
                /* Combine folded into the HC expand below; attn_out is not
                 * materialized on this path. */
                tp_attn_a = first;
                tp_attn_b = g->tp_rank == 0 ? g->tp_in[slot] : g->tp_out[slot];
            }
        }
    }
    DS4_METAL_PROFILE_DECODE_STAGE("attn_output");
    if (ok) {
        metal_graph_debug_dump_tensor("attn_low", metal_graph_attn_low(g), (uint64_t)n_groups * rank, il, pos);
    }
    if (ok) {
        metal_graph_debug_dump_tensor("attn_out", metal_graph_attn_out(g), DS4_N_EMBD, il, pos);
    }
    if (ok && metal_graph_directional_steering_attn_enabled(g)) {
        ok = metal_graph_apply_directional_steering_attn(g, metal_graph_attn_out(g), il, 1);
    }
    if (ok && !fuse_attn_out_hc && !cuda_tp_attn_hc_fused) {
        if (tp_attn_a) {
            ok = ds4_gpu_hc_expand_add_tensor(metal_graph_after_attn_hc(g), tp_attn_a, tp_attn_b,
                                                metal_graph_cur_hc(g), metal_graph_hc_post(g), metal_graph_hc_comb(g),
                                                DS4_N_EMBD, DS4_N_HC) != 0;
        } else if (cuda_tp_attn_peer) {
            ok = ds4_gpu_hc_expand_add_tensor(
                    metal_graph_after_attn_hc(g),
                    metal_graph_attn_out(g),
                    cuda_tp_attn_peer,
                    metal_graph_cur_hc(g),
                    metal_graph_hc_post(g),
                    metal_graph_hc_comb(g),
                    DS4_N_EMBD,
                    DS4_N_HC) != 0;
        } else {
            ok = ds4_gpu_hc_expand_tensor(metal_graph_after_attn_hc(g), metal_graph_attn_out(g), metal_graph_cur_hc(g),
                                            metal_graph_hc_post(g), metal_graph_hc_comb(g), DS4_N_EMBD, DS4_N_HC) != 0;
        }
    }
    DS4_METAL_PROFILE_DECODE_STAGE("attn_hc_post");
    if (ok) {
        metal_graph_debug_dump_tensor("hc_attn_post", metal_graph_after_attn_hc(g), hc_dim, il, pos);
    }
    if (ok && !tp_ablate_hcpre) {
        ok = ds4_gpu_rms_norm_plain_tensor(metal_graph_flat_hc(g), metal_graph_after_attn_hc(g), (uint32_t)hc_dim, DS4_RMS_EPS) != 0;
        if (ok) ok = metal_graph_matmul_plain_tensor(metal_graph_hc_mix(g), model, layer->hc_ffn_fn,
                                                     hc_dim, mix_hc, metal_graph_flat_hc(g), 1);
    }
    if (ok && fuse_hc_norm) {
        ok = ds4_gpu_hc_split_weighted_sum_norm_tensor(metal_graph_ffn_cur(g),
                                                         metal_graph_ffn_norm(g),
                                                         metal_graph_hc_split(g),
                                                         metal_graph_hc_mix(g),
                                                         metal_graph_after_attn_hc(g),
                                                         model->map,
                                                         model->size,
                                                         layer->hc_ffn_scale->abs_offset,
                                                         layer->hc_ffn_base->abs_offset,
                                                         layer->ffn_norm->abs_offset,
                                                         DS4_N_EMBD,
                                                         DS4_N_HC,
                                                         DS4_N_HC_SINKHORN_ITER,
                                                         DS4_HC_EPS,
                                                         DS4_RMS_EPS) != 0;
        if (ok) {
            ok = metal_graph_check_hc_norm_fusion("ffn",
                                                  metal_graph_ffn_cur(g),
                                                  metal_graph_ffn_norm(g),
                                                  metal_graph_hc_mix(g),
                                                  metal_graph_after_attn_hc(g),
                                                  model,
                                                  layer->hc_ffn_scale->abs_offset,
                                                  layer->hc_ffn_base->abs_offset,
                                                  layer->ffn_norm->abs_offset,
                                                  il,
                                                  pos);
        }
    } else if (ok) {
        ok = metal_graph_decode_hc_pre(metal_graph_ffn_cur(g),
                                       metal_graph_hc_split(g),
                                       metal_graph_hc_mix(g),
                                       metal_graph_after_attn_hc(g),
                                       model,
                                       layer->hc_ffn_scale->abs_offset,
                                       layer->hc_ffn_base->abs_offset);
    }
    DS4_METAL_PROFILE_DECODE_STAGE("ffn_hc_pre");
    if (ok) {
        metal_graph_debug_dump_tensor("hc_ffn_pre_mixes", metal_graph_hc_mix(g), mix_hc, il, pos);
        metal_graph_debug_dump_tensor("hc_ffn_pre_weights", metal_graph_hc_pre(g), DS4_N_HC, il, pos);
        metal_graph_debug_dump_tensor("hc_ffn_pre_post_weights", metal_graph_hc_post(g), DS4_N_HC, il, pos);
        metal_graph_debug_dump_tensor("hc_ffn_pre_comb", metal_graph_hc_comb(g), (uint64_t)DS4_N_HC * DS4_N_HC, il, pos);
    }
    if (ok) {
        metal_graph_debug_dump_tensor("hc_ffn_pre", metal_graph_ffn_cur(g), DS4_N_EMBD, il, pos);
    }
    if (ok && !fuse_hc_norm) ok = ds4_gpu_rms_norm_weight_tensor(metal_graph_ffn_norm(g), metal_graph_ffn_cur(g),
                                                                   model->map, model->size,
                                                                   layer->ffn_norm->abs_offset,
                                                                   DS4_N_EMBD, DS4_RMS_EPS) != 0;
    DS4_METAL_PROFILE_DECODE_STAGE("ffn_norm");
    if (ok) {
        metal_graph_debug_dump_tensor("ffn_norm", metal_graph_ffn_norm(g), DS4_N_EMBD, il, pos);
    }
    const uint64_t gate_row_bytes = routed_expert_row_bytes(layer->ffn_gate_exps);
    const uint64_t gate_expert_bytes DS4_MAYBE_UNUSED = expert_mid_dim * gate_row_bytes;
    const uint64_t down_row_bytes = routed_expert_row_bytes(layer->ffn_down_exps);
    const uint64_t down_expert_bytes DS4_MAYBE_UNUSED = routed_out_dim * down_row_bytes;
    if (ok && metal_graph_decode_cpu_router_applicable(g, layer)) {
        ok = metal_graph_decode_cpu_router(g, model, layer, il, (uint32_t)token);
    } else {
        if (ok && !metal_graph_tp_ablate("router")) {
        ok = metal_graph_matmul_plain_tensor(metal_graph_router_logits(g), model, layer->ffn_gate_inp,
                                                     DS4_N_EMBD, DS4_N_EXPERT, metal_graph_ffn_norm(g), 1);
        if (ok) ok = ds4_gpu_router_select_tensor(metal_graph_router_selected(g), metal_graph_router_weights(g), metal_graph_router_probs(g),
                                                    model->map, model->size,
                                                    layer->ffn_exp_probs_b ? layer->ffn_exp_probs_b->abs_offset : 0,
                                                    layer->ffn_gate_tid2eid ? layer->ffn_gate_tid2eid->abs_offset : 0,
                                                    layer->ffn_gate_tid2eid ? (uint32_t)layer->ffn_gate_tid2eid->dim[1] : 0,
                                                    (uint32_t)token,
                                                    DS4_N_EXPERT,
                                                    DS4_N_EXPERT_USED,
                                                    DS4_EXPERT_WEIGHT_SCALE,
                                                    0,
                                                    0,
                                                    layer->ffn_exp_probs_b != NULL,
                                                    layer->ffn_gate_tid2eid != NULL,
                                                    metal_graph_router_logits(g)) != 0;
        }
        if (ok) ok = metal_graph_decode_set_hash_selected_override(model,
                                                                   layer,
                                                                   il,
                                                                   (uint32_t)token,
                                                                   layer->ffn_gate_exps->bytes,
                                                                   layer->ffn_down_exps->bytes,
                                                                   g);
    }
    DS4_METAL_PROFILE_DECODE_STAGE("router");
    if (ok) ok = metal_graph_profile_router_selection(g, layer, il, pos);
    if (ok) {
        metal_graph_debug_dump_tensor("ffn_moe_logits", metal_graph_router_logits(g), DS4_N_EXPERT, il, pos);
        metal_graph_debug_dump_tensor("ffn_moe_probs", metal_graph_router_probs(g), DS4_N_EXPERT, il, pos);
        metal_graph_debug_dump_i32_tensor("ffn_moe_topk", metal_graph_router_selected(g), DS4_N_EXPERT_USED, il, pos);
        metal_graph_debug_dump_tensor("ffn_moe_weights_scaled", metal_graph_router_weights(g), DS4_N_EXPERT_USED, il, pos);
    }
    if (phase == METAL_DECODE_LAYER_TO_ROUTER) return ok;
    }
    const bool external_routed = phase == METAL_DECODE_LAYER_FROM_ROUTER;
    const bool fuse_shared_gate_up =
        !g->quality &&
        g->tp_world < 2 &&
        layer->ffn_gate_shexp->type == DS4_TENSOR_Q8_0 &&
        layer->ffn_up_shexp->type == DS4_TENSOR_Q8_0 &&
        g->shared_gate_up_swiglu_fuse;
    const bool keep_ffn_out = metal_graph_needs_ffn_out(g, il, pos);
    const bool cuda_tp_shared_requested = g->cuda_tp_shared;
    const bool cuda_tp_moe_requested = !external_routed && g->cuda_tp_moe;
    const uint64_t shared_tp_local = shared_dim / 2u;
    const uint64_t shared_tp_peer = shared_dim - shared_tp_local;
    const uint64_t shared_q8_blocks = ((uint64_t)DS4_N_EMBD + 31u) / 32u;
    const uint64_t shared_q8_x_bytes = shared_q8_blocks * 32u;
    const uint64_t shared_q8_scale_offset =
        (shared_q8_x_bytes + 15u) & ~15ull;
    const uint64_t shared_q8_prequant_bytes DS4_MAYBE_UNUSED =
        shared_q8_scale_offset + shared_q8_blocks * sizeof(float);
    const bool cuda_tp_shared =
        cuda_tp_shared_requested &&
        fuse_shared_gate_up &&
        !metal_graph_use_reference_shared_down_hc() &&
        cuda_tp_partner_tier >= 0 &&
        shared_tp_local != 0 &&
        shared_tp_peer != 0 &&
        (shared_tp_local % 32u) == 0 &&
        (shared_tp_peer % 32u) == 0 &&
        g->ffn_norm_by_tier[cuda_tp_partner_tier] &&
        g->shared_gate_by_tier[cuda_tp_partner_tier] &&
        g->shared_up_by_tier[cuda_tp_partner_tier] &&
        g->shared_mid_by_tier[cuda_tp_partner_tier] &&
        g->shared_out_by_tier[cuda_tp_partner_tier] &&
        g->tp_peer_tmp_by_tier[cuda_tp_home_tier];
    const bool cuda_tp_shared_fold =
        cuda_tp_shared &&
        g->cuda_tp_shared_fold &&
        !g->cuda_tp_ep &&
        cuda_tp_moe_requested &&
        !keep_ffn_out &&
        !metal_graph_directional_steering_ffn_enabled(g) &&
        !metal_graph_debug_wants("ffn_moe_out", il, pos) &&
        !metal_graph_debug_wants("ffn_shexp", il, pos);
    const bool fuse_shared_down_hc =
        g->tp_world < 2 &&
        layer->ffn_down_shexp->type == DS4_TENSOR_Q8_0 &&
        !cuda_tp_shared &&
        !keep_ffn_out &&
        !metal_graph_use_reference_shared_down_hc();
    const bool cuda_tp_moe =
        cuda_tp_moe_requested &&
        cuda_tp_partner_tier >= 0 &&
        (DS4_N_EXPERT_USED % 2u) == 0u &&
        g->ffn_norm_by_tier[cuda_tp_partner_tier] &&
        g->router_selected_by_tier[cuda_tp_partner_tier] &&
        g->router_weights_by_tier[cuda_tp_partner_tier] &&
        g->routed_gate_by_tier[cuda_tp_partner_tier] &&
        g->routed_up_by_tier[cuda_tp_partner_tier] &&
        g->routed_mid_by_tier[cuda_tp_partner_tier] &&
        g->routed_down_by_tier[cuda_tp_partner_tier] &&
        g->routed_out_by_tier[cuda_tp_partner_tier] &&
        g->tp_peer_tmp_by_tier[cuda_tp_home_tier] &&
        g->tp_peer_tmp_by_tier[cuda_tp_partner_tier];
    const bool cuda_tp_ep = cuda_tp_moe && g->cuda_tp_ep;
    const bool cuda_tp_moe_delay_reduce =
        cuda_tp_moe &&
        g->cuda_tp_moe_delay_reduce &&
        fuse_shared_down_hc &&
        !metal_graph_debug_wants("ffn_moe_out", il, pos);
    bool cuda_tp_moe_peer_tmp = false;
    bool cuda_tp_moe_peer_copy_deferred = false;
    bool cuda_tp_ep_reduce_deferred = false;
    bool cuda_tp_ep_fused_hc_reduce = false;
    bool cuda_tp_ep_direct_return = false;
    bool cuda_tp_ep_balanced_shared_mid DS4_MAYBE_UNUSED = false;
    bool cuda_tp_ep_dual_prequant = false;
    uint64_t cuda_tp_ep_return_bytes = 0;
    bool cuda_tp_shared_fold_peer_tmp = false;
    if (ok && cuda_tp_moe_requested && !cuda_tp_moe) {
        fprintf(stderr,
                "ds4: CUDA decode TP cannot split routed MoE for tier %d "
                "(partner=%d experts=%u)\n",
                cuda_tp_home_tier, cuda_tp_partner_tier, DS4_N_EXPERT_USED);
        ok = false;
    }
    if (ok && cuda_tp_moe) {
        const uint32_t tp_experts = cuda_tp_ep
            ? DS4_N_EXPERT_USED : DS4_N_EXPERT_USED / 2u;
        const uint64_t tp_selected_bytes = (uint64_t)tp_experts * sizeof(int32_t);
        const uint64_t tp_weights_bytes = (uint64_t)tp_experts * sizeof(float);
        const uint64_t peer_selected_offset = cuda_tp_ep ? 0 : tp_selected_bytes;
        const uint64_t peer_weights_offset = cuda_tp_ep ? 0 : tp_weights_bytes;
        ds4_gpu_tensor local_selected;
        ds4_gpu_tensor local_weights;
        ds4_gpu_tensor peer_selected_src;
        ds4_gpu_tensor peer_weights_src;
        ds4_gpu_tensor packed_peer_ffn_norm;
        ds4_gpu_tensor packed_peer_selected;
        ds4_gpu_tensor packed_peer_weights;
        ds4_gpu_tensor direct_peer_down;
        ds4_gpu_tensor *peer_down_output = NULL;
        cuda_tp_ep_return_bytes =
            (uint64_t)(g->cuda_tp_ep_pack_exact ? 4u : DS4_N_EXPERT_USED) *
            DS4_N_EMBD * sizeof(float);
        cuda_tp_ep_direct_return =
            cuda_tp_ep &&
            metal_graph_cuda_tp_ep_direct_return_requested() &&
            g_gpu_peer_ok[cuda_tp_partner_tier][cuda_tp_home_tier] &&
            metal_graph_borrow_tensor_view(
                    &direct_peer_down,
                    g->tp_peer_tmp_by_tier[cuda_tp_home_tier],
                    0,
                    cuda_tp_ep_return_bytes);
        if (cuda_tp_ep_direct_return) peer_down_output = &direct_peer_down;
#if !defined(__APPLE__)
        /* Run the shared gate/up projection on the less-loaded EP rank.  The
         * two kernels use complementary predicates over the same top-k IDs;
         * a partner result is ordered by the existing direct-return event. */
        cuda_tp_ep_balanced_shared_mid =
            cuda_tp_ep_direct_return &&
            cuda_tp_moe_delay_reduce &&
            metal_graph_cuda_tp_ep_delay_reduce_requested() &&
            metal_graph_cuda_tp_ep_fused_shared_mid_requested() &&
            metal_graph_cuda_tp_ep_balanced_shared_mid_requested() &&
            fuse_shared_gate_up &&
            !cuda_tp_shared_requested &&
            !g->cuda_tp_moe_peer_read &&
            !g->cuda_tp_moe_peer_router &&
            !g->decode_stage_profile;
        cuda_tp_ep_dual_prequant =
            cuda_tp_ep_balanced_shared_mid &&
            metal_graph_cuda_tp_ep_dual_prequant_requested() &&
            metal_graph_shared_gate(g) &&
            metal_graph_shared_gate(g)->bytes >= shared_q8_prequant_bytes &&
            g->shared_gate_by_tier[cuda_tp_partner_tier] &&
            g->shared_gate_by_tier[cuda_tp_partner_tier]->bytes >=
                shared_q8_prequant_bytes;
#endif
        const bool cuda_tp_moe_peer_read =
            g->cuda_tp_moe_peer_read &&
            g_gpu_peer_ok[cuda_tp_partner_tier][cuda_tp_home_tier];
        const bool cuda_tp_moe_peer_router =
            !cuda_tp_moe_peer_read &&
            g->cuda_tp_moe_peer_router &&
            g_gpu_peer_ok[cuda_tp_partner_tier][cuda_tp_home_tier];
        const ds4_gpu_tensor *peer_selected = g->router_selected_by_tier[cuda_tp_partner_tier];
        const ds4_gpu_tensor *peer_weights = g->router_weights_by_tier[cuda_tp_partner_tier];
        const ds4_gpu_tensor *peer_ffn_norm = g->ffn_norm_by_tier[cuda_tp_partner_tier];
        ok = metal_graph_borrow_tensor_view(&local_selected,
                                            metal_graph_router_selected(g),
                                            0,
                                            tp_selected_bytes) &&
             metal_graph_borrow_tensor_view(&local_weights,
                                            metal_graph_router_weights(g),
                                            0,
                                            tp_weights_bytes) &&
             metal_graph_borrow_tensor_view(&peer_selected_src,
                                            metal_graph_router_selected(g),
                                            peer_selected_offset,
                                            tp_selected_bytes) &&
             metal_graph_borrow_tensor_view(&peer_weights_src,
                                            metal_graph_router_weights(g),
                                            peer_weights_offset,
                                            tp_weights_bytes);
        if (ok && cuda_tp_moe_peer_read) {
            ok = ds4_gpu_tensor_wait_xdev(metal_graph_router_weights(g),
                                          cuda_tp_partner_tier) != 0;
            peer_selected = &peer_selected_src;
            peer_weights = &peer_weights_src;
            peer_ffn_norm = metal_graph_ffn_norm(g);
        } else if (ok && cuda_tp_moe_peer_router) {
            ok = ds4_gpu_tensor_copy_xdev(g->ffn_norm_by_tier[cuda_tp_partner_tier],
                                          metal_graph_ffn_norm(g),
                                          (uint64_t)DS4_N_EMBD * sizeof(float)) != 0;
            peer_selected = &peer_selected_src;
            peer_weights = &peer_weights_src;
            peer_ffn_norm = g->ffn_norm_by_tier[cuda_tp_partner_tier];
        } else if (ok && g->cuda_tp_moe_pack_handoff) {
            const uint64_t norm_bytes = (uint64_t)DS4_N_EMBD * sizeof(float);
            const uint64_t packed_selected_off = norm_bytes;
            const uint64_t packed_weights_off = packed_selected_off + tp_selected_bytes;
            const uint64_t packed_bytes = packed_weights_off + tp_weights_bytes;
            ok = metal_graph_borrow_tensor_view(&packed_peer_ffn_norm,
                                                g->tp_peer_tmp_by_tier[cuda_tp_partner_tier],
                                                0,
                                                norm_bytes) &&
                 metal_graph_borrow_tensor_view(&packed_peer_selected,
                                                g->tp_peer_tmp_by_tier[cuda_tp_partner_tier],
                                                packed_selected_off,
                                                tp_selected_bytes) &&
                 metal_graph_borrow_tensor_view(&packed_peer_weights,
                                                g->tp_peer_tmp_by_tier[cuda_tp_partner_tier],
                                                packed_weights_off,
                                                tp_weights_bytes);
            if (ok) {
                ok = ds4_gpu_moe_handoff_pack_tensor(
                        g->tp_peer_tmp_by_tier[cuda_tp_home_tier],
                        metal_graph_ffn_norm(g),
                        &peer_selected_src,
                        &peer_weights_src,
                        DS4_N_EMBD,
                        tp_experts) != 0;
            }
            if (ok) {
                ok = ds4_gpu_tensor_copy_xdev(
                        g->tp_peer_tmp_by_tier[cuda_tp_partner_tier],
                        g->tp_peer_tmp_by_tier[cuda_tp_home_tier],
                        packed_bytes) != 0;
            }
            peer_selected = &packed_peer_selected;
            peer_weights = &packed_peer_weights;
            peer_ffn_norm = &packed_peer_ffn_norm;
        } else if (ok && g->cuda_tp_moe_copy3_handoff) {
            ok = ds4_gpu_tensor_copy_xdev3(
                    g->ffn_norm_by_tier[cuda_tp_partner_tier],
                    metal_graph_ffn_norm(g),
                    (uint64_t)DS4_N_EMBD * sizeof(float),
                    g->router_selected_by_tier[cuda_tp_partner_tier],
                    &peer_selected_src,
                    tp_selected_bytes,
                    g->router_weights_by_tier[cuda_tp_partner_tier],
                    &peer_weights_src,
                    tp_weights_bytes) != 0;
        } else if (ok) {
            ok = ds4_gpu_tensor_copy_xdev(g->ffn_norm_by_tier[cuda_tp_partner_tier],
                                          metal_graph_ffn_norm(g),
                                          (uint64_t)DS4_N_EMBD * sizeof(float)) != 0 &&
                 ds4_gpu_tensor_copy_xdev(g->router_selected_by_tier[cuda_tp_partner_tier],
                                          &peer_selected_src,
                                          tp_selected_bytes) != 0 &&
                 ds4_gpu_tensor_copy_xdev(g->router_weights_by_tier[cuda_tp_partner_tier],
                                          &peer_weights_src,
                                          tp_weights_bytes) != 0;
        }

        bool switched_to_partner = false;
        if (ok) {
            ok = ds4_gpu_set_current_device(cuda_tp_partner_tier) == 0;
            switched_to_partner = ok;
        }
        if (ok) {
            if (cuda_tp_ep) {
                ok = ds4_gpu_routed_moe_one_owned_tensor(
                        g->routed_out_by_tier[cuda_tp_partner_tier],
                        g->routed_gate_by_tier[cuda_tp_partner_tier],
                        g->routed_up_by_tier[cuda_tp_partner_tier],
                        g->routed_mid_by_tier[cuda_tp_partner_tier],
                        g->routed_down_by_tier[cuda_tp_partner_tier],
                        model->map, model->size,
                        layer->ffn_gate_exps->abs_offset,
                        layer->ffn_up_exps->abs_offset,
                        layer->ffn_down_exps->abs_offset,
                        layer->ffn_gate_exps->type,
                        layer->ffn_down_exps->type,
                        gate_expert_bytes, gate_row_bytes,
                        down_expert_bytes, down_row_bytes,
                        (uint32_t)expert_in_dim,
                        (uint32_t)down_in_dim,
                        (uint32_t)routed_out_dim,
                        peer_selected,
                        peer_weights,
                        DS4_N_EXPERT,
                        tp_experts,
                        DS4_N_EXPERT / 2u,
                        DS4_N_EXPERT - DS4_N_EXPERT / 2u,
                        DS4_SWIGLU_CLAMP_EXP,
                        peer_ffn_norm,
                        peer_down_output,
                        g->cuda_tp_ep_pack_exact,
                        cuda_tp_ep_dual_prequant
                            ? g->shared_gate_by_tier[cuda_tp_partner_tier]
                            : NULL) != 0;
            } else {
                ok = ds4_gpu_routed_moe_one_tensor(
                        g->routed_out_by_tier[cuda_tp_partner_tier],
                        g->routed_gate_by_tier[cuda_tp_partner_tier],
                        g->routed_up_by_tier[cuda_tp_partner_tier],
                        g->routed_mid_by_tier[cuda_tp_partner_tier],
                        g->routed_down_by_tier[cuda_tp_partner_tier],
                        model->map, model->size,
                        layer->ffn_gate_exps->abs_offset,
                        layer->ffn_up_exps->abs_offset,
                        layer->ffn_down_exps->abs_offset,
                        layer->ffn_gate_exps->type,
                        layer->ffn_down_exps->type,
                        gate_expert_bytes, gate_row_bytes,
                        down_expert_bytes, down_row_bytes,
                        (uint32_t)expert_in_dim,
                        (uint32_t)down_in_dim,
                        (uint32_t)routed_out_dim,
                        peer_selected,
                        peer_weights,
                        DS4_N_EXPERT,
                        tp_experts,
                        DS4_SWIGLU_CLAMP_EXP,
                        peer_ffn_norm, NULL, 0, false) != 0;
            }
        }
#if !defined(__APPLE__)
        if (ok && cuda_tp_ep_balanced_shared_mid) {
            ok = ds4_gpu_shared_mid_swiglu_q8_0_decode_exact_tensor(
                    metal_graph_shared_mid(g),
                    model->map,
                    model->size,
                    layer->ffn_gate_shexp->abs_offset,
                    layer->ffn_up_shexp->abs_offset,
                    DS4_N_EMBD,
                    shared_dim,
                    peer_ffn_norm,
                    DS4_SWIGLU_CLAMP_EXP,
                    peer_selected,
                    cuda_tp_ep_dual_prequant
                        ? g->shared_gate_by_tier[cuda_tp_partner_tier]
                        : NULL,
                    DS4_N_EXPERT / 2u,
                    false) != 0;
        }
#endif
        if (switched_to_partner && ds4_gpu_set_current_device(cuda_tp_home_tier) != 0) {
            ok = false;
        }
        if (ok) {
            if (cuda_tp_ep) {
                ok = ds4_gpu_routed_moe_one_owned_tensor(
                        metal_graph_routed_out(g),
                        metal_graph_routed_gate(g),
                        metal_graph_routed_up(g),
                        metal_graph_routed_mid(g),
                        metal_graph_routed_down(g),
                        model->map, model->size,
                        layer->ffn_gate_exps->abs_offset,
                        layer->ffn_up_exps->abs_offset,
                        layer->ffn_down_exps->abs_offset,
                        layer->ffn_gate_exps->type,
                        layer->ffn_down_exps->type,
                        gate_expert_bytes, gate_row_bytes,
                        down_expert_bytes, down_row_bytes,
                        (uint32_t)expert_in_dim,
                        (uint32_t)down_in_dim,
                        (uint32_t)routed_out_dim,
                        &local_selected,
                        &local_weights,
                        DS4_N_EXPERT,
                        tp_experts,
                        0,
                        DS4_N_EXPERT / 2u,
                        DS4_SWIGLU_CLAMP_EXP,
                        metal_graph_ffn_norm(g),
                        NULL,
                        false,
                        cuda_tp_ep_dual_prequant
                            ? metal_graph_shared_gate(g)
                            : NULL) != 0;
            } else {
                ok = ds4_gpu_routed_moe_one_tensor(
                        metal_graph_routed_out(g),
                        metal_graph_routed_gate(g),
                        metal_graph_routed_up(g),
                        metal_graph_routed_mid(g),
                        metal_graph_routed_down(g),
                        model->map, model->size,
                        layer->ffn_gate_exps->abs_offset,
                        layer->ffn_up_exps->abs_offset,
                        layer->ffn_down_exps->abs_offset,
                        layer->ffn_gate_exps->type,
                        layer->ffn_down_exps->type,
                        gate_expert_bytes, gate_row_bytes,
                        down_expert_bytes, down_row_bytes,
                        (uint32_t)expert_in_dim,
                        (uint32_t)down_in_dim,
                        (uint32_t)routed_out_dim,
                        &local_selected,
                        &local_weights,
                        DS4_N_EXPERT,
                        tp_experts,
                        DS4_SWIGLU_CLAMP_EXP,
                        metal_graph_ffn_norm(g), NULL, 0, false) != 0;
            }
        }
#if !defined(__APPLE__)
        if (ok && cuda_tp_ep_balanced_shared_mid) {
            ok = ds4_gpu_shared_mid_swiglu_q8_0_decode_exact_tensor(
                    metal_graph_shared_mid(g),
                    model->map,
                    model->size,
                    layer->ffn_gate_shexp->abs_offset,
                    layer->ffn_up_shexp->abs_offset,
                    DS4_N_EMBD,
                    shared_dim,
                    metal_graph_ffn_norm(g),
                    DS4_SWIGLU_CLAMP_EXP,
                    &local_selected,
                    cuda_tp_ep_dual_prequant
                        ? metal_graph_shared_gate(g)
                        : NULL,
                    DS4_N_EXPERT / 2u,
                    true) != 0;
        }
#endif
        if (ok) {
            if (cuda_tp_ep) {
                if (cuda_tp_moe_delay_reduce &&
                    metal_graph_cuda_tp_ep_delay_reduce_requested() &&
                    !g->decode_stage_profile) {
                    cuda_tp_ep_reduce_deferred = true;
                    cuda_tp_ep_fused_hc_reduce =
                        g->cuda_tp_ep_pack_exact &&
                        metal_graph_cuda_tp_ep_fused_hc_reduce_requested();
                } else {
                    ok = metal_graph_cuda_tp_ep_finish_reduce(
                            g,
                            cuda_tp_home_tier,
                            cuda_tp_partner_tier,
                            cuda_tp_ep_direct_return,
                            cuda_tp_ep_return_bytes,
                            true);
                }
            } else if (cuda_tp_moe_delay_reduce &&
                !cuda_tp_shared && !cuda_tp_shared_fold) {
                /* Defer the peer routed-half copy until after the shared
                 * expert gate/up launch below: that work depends only on
                 * ffn_norm, so the home stream computes it while waiting
                 * for the partner instead of idling at the copy. */
                cuda_tp_moe_peer_copy_deferred = true;
            } else if (cuda_tp_moe_delay_reduce) {
                ok = ds4_gpu_tensor_copy_xdev(
                        g->tp_peer_tmp_by_tier[cuda_tp_home_tier],
                        g->routed_out_by_tier[cuda_tp_partner_tier],
                        (uint64_t)DS4_N_EMBD * sizeof(float)) != 0;
                cuda_tp_moe_peer_tmp = ok;
            } else if (!cuda_tp_shared_fold) {
                ok = ds4_gpu_add_xdev_tensor(metal_graph_routed_out(g),
                                             metal_graph_routed_out(g),
                                             g->routed_out_by_tier[cuda_tp_partner_tier],
                                             g->tp_peer_tmp_by_tier[cuda_tp_home_tier],
                                             DS4_N_EMBD) != 0;
            }
        }
    }
    /* Real TP split slices the shared expert by intermediate lanes, which
     * needs the unfused gate/up/swiglu/down sequence. */
    const bool tp_split_shared = g->tp_world == 2;
    const bool q4_selected_shared_overlap =
        metal_graph_use_q4_selected_shared_overlap() &&
        metal_graph_decode_q4_selected_slots_expected(g,
                                                      layer,
                                                      layer->ffn_gate_exps->bytes,
                                                      layer->ffn_down_exps->bytes);
    const bool iq2_selected_shared_overlap =
        metal_graph_use_iq2_selected_shared_overlap(g) &&
        metal_graph_decode_iq2_selected_slots_expected(g, layer);
    const bool cuda_selected_shared_overlap =
        metal_graph_use_cuda_selected_shared_overlap(g) &&
        metal_graph_decode_cuda_selected_slots_expected(g, layer);
    const bool overlap_selected_shared =
        ok &&
        g->tp_world < 2 &&
        !decode_stage_profile &&
        !metal_graph_decode_cpu_router_applicable(g, layer) &&
        layer->ffn_gate_tid2eid == NULL &&
        getenv("DS4_MOE_REPLAY_SELECTED_IDS") == NULL &&
        (q4_selected_shared_overlap ||
         iq2_selected_shared_overlap ||
         cuda_selected_shared_overlap);
    const bool async_selected_load =
        overlap_selected_shared &&
        ((iq2_selected_shared_overlap &&
          metal_graph_use_iq2_selected_async_load(g)) ||
         cuda_selected_shared_overlap);
    const bool selected_readahead_shared_delay =
        ok &&
        g->tp_world < 2 &&
        !overlap_selected_shared &&
        !decode_stage_profile &&
        metal_graph_use_iq2_selected_readahead_shared_delay(g) &&
        metal_graph_decode_iq2_selected_slots_expected(g, layer) &&
        !metal_graph_decode_cpu_router_applicable(g, layer) &&
        layer->ffn_gate_tid2eid == NULL &&
        getenv("DS4_MOE_REPLAY_SELECTED_IDS") == NULL;
    const bool cuda_stream_selected_load =
        ok &&
        !overlap_selected_shared &&
        !selected_readahead_shared_delay &&
        g->ssd_streaming &&
        metal_graph_decode_cuda_selected_slots_expected(g, layer) &&
        layer->ffn_gate_tid2eid == NULL &&
        getenv("DS4_MOE_REPLAY_SELECTED_IDS") == NULL;
    if (cuda_stream_selected_load) {
        ok = metal_graph_decode_cuda_selected_load(g,
                                                   model,
                                                   layer,
                                                   il,
                                                   gate_expert_bytes,
                                                   down_expert_bytes);
    }
    if (selected_readahead_shared_delay) {
        if (ok) {
            ok = metal_graph_decode_selected_readahead_override(g,
                                                                model,
                                                                layer,
                                                                il,
                                                                gate_expert_bytes,
                                                                down_expert_bytes);
        }
        if (ok && fuse_shared_gate_up) {
            ok = ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(metal_graph_shared_gate(g),
                                                             metal_graph_shared_up(g),
                                                             metal_graph_shared_mid(g),
                                                             model->map,
                                                             model->size,
                                                             layer->ffn_gate_shexp->abs_offset,
                                                             layer->ffn_up_shexp->abs_offset,
                                                             DS4_N_EMBD,
                                                             shared_dim,
                                                             metal_graph_ffn_norm(g),
                                                             DS4_SWIGLU_CLAMP_EXP) != 0;
        } else if (ok) {
            if (ok) ok = metal_graph_matmul_dense_quant_tensor(metal_graph_shared_gate(g),
                                                               model,
                                                               layer->ffn_gate_shexp,
                                                               DS4_N_EMBD,
                                                               shared_dim,
                                                               metal_graph_ffn_norm(g),
                                                               1);
            if (ok) ok = metal_graph_matmul_dense_quant_tensor(metal_graph_shared_up(g),
                                                               model,
                                                               layer->ffn_up_shexp,
                                                               DS4_N_EMBD,
                                                               shared_dim,
                                                               metal_graph_ffn_norm(g),
                                                               1);
            if (ok) ok = ds4_gpu_swiglu_tensor(metal_graph_shared_mid(g), metal_graph_shared_gate(g), metal_graph_shared_up(g),
                                               shared_dim, DS4_SWIGLU_CLAMP_EXP, 1.0f) != 0;
        }
        DS4_METAL_PROFILE_DECODE_STAGE("shared_gate_up");
        if (ok) ok = ds4_gpu_routed_moe_one_tensor(metal_graph_routed_out(g),
                                                     metal_graph_routed_gate(g),
                                                     metal_graph_routed_up(g),
                                                     metal_graph_routed_mid(g),
                                                     metal_graph_routed_down(g),
                                                     model->map, model->size,
                                                     layer->ffn_gate_exps->abs_offset,
                                                     layer->ffn_up_exps->abs_offset,
                                                     layer->ffn_down_exps->abs_offset,
                                                     layer->ffn_gate_exps->type,
                                                     layer->ffn_down_exps->type,
                                                     gate_expert_bytes, gate_row_bytes,
                                                     down_expert_bytes, down_row_bytes,
                                                     (uint32_t)expert_in_dim,
                                                     (uint32_t)down_in_dim,
                                                     (uint32_t)routed_out_dim,
                                                     metal_graph_router_selected(g), metal_graph_router_weights(g),
                                                     DS4_N_EXPERT,
                                                     DS4_N_EXPERT_USED, DS4_SWIGLU_CLAMP_EXP, metal_graph_ffn_norm(g),
                                                     NULL,
                                                     il,
                                                     false) != 0;
        DS4_METAL_PROFILE_DECODE_STAGE("routed_moe");
        if (ok) {
            metal_graph_debug_dump_tensor("ffn_moe_gate_clamped", metal_graph_routed_gate(g),
                                          (uint64_t)DS4_N_EXPERT_USED * down_in_dim, il, pos);
            metal_graph_debug_dump_tensor("ffn_moe_up_clamped", metal_graph_routed_up(g),
                                          (uint64_t)DS4_N_EXPERT_USED * down_in_dim, il, pos);
            metal_graph_debug_dump_tensor("ffn_moe_weighted_swiglu", metal_graph_routed_mid(g),
                                          (uint64_t)DS4_N_EXPERT_USED * down_in_dim, il, pos);
            metal_graph_debug_dump_tensor("ffn_moe_down", metal_graph_routed_down(g),
                                          (uint64_t)DS4_N_EXPERT_USED * DS4_N_EMBD, il, pos);
            metal_graph_debug_dump_tensor("ffn_moe_out", metal_graph_routed_out(g), DS4_N_EMBD, il, pos);
        }
        if (ok && fuse_shared_down_hc) {
            ok = ds4_gpu_shared_down_hc_expand_q8_0_tensor(metal_graph_after_ffn_hc(g),
                                                             metal_graph_shared_out(g),
                                                             model->map,
                                                             model->size,
                                                             layer->ffn_down_shexp->abs_offset,
                                                             shared_dim,
                                                             DS4_N_EMBD,
                                                             metal_graph_shared_mid(g),
                                                             metal_graph_routed_out(g),
                                                             metal_graph_after_attn_hc(g),
                                                             metal_graph_hc_split(g),
                                                             DS4_N_EMBD,
                                                             DS4_N_HC) != 0;
        } else if (ok) {
            ok = metal_graph_matmul_dense_quant_tensor(metal_graph_shared_out(g),
                                                       model,
                                                       layer->ffn_down_shexp,
                                                       shared_dim,
                                                       DS4_N_EMBD,
                                                       metal_graph_shared_mid(g),
                                                       1);
        }
        DS4_METAL_PROFILE_DECODE_STAGE("shared_down");
        if (ok) {
            metal_graph_debug_dump_tensor("ffn_shexp", metal_graph_shared_out(g), DS4_N_EMBD, il, pos);
        }
        if (ok && keep_ffn_out) {
            ok = metal_graph_ensure_ffn_out(g) &&
                 ds4_gpu_add_tensor(metal_graph_ffn_out(g), metal_graph_shared_out(g), metal_graph_routed_out(g), DS4_N_EMBD) != 0;
        }
        if (ok && keep_ffn_out) {
            metal_graph_debug_dump_tensor("ffn_out", metal_graph_ffn_out(g), DS4_N_EMBD, il, pos);
        }
        if (ok && metal_graph_directional_steering_ffn_enabled(g)) {
            ok = metal_graph_apply_directional_steering_ffn(g, metal_graph_ffn_out(g), il, 1);
        }
        if (ok && metal_graph_directional_steering_ffn_enabled(g)) {
            ok = ds4_gpu_hc_expand_tensor(metal_graph_after_ffn_hc(g),
                                            metal_graph_ffn_out(g),
                                            metal_graph_after_attn_hc(g),
                                            metal_graph_hc_post(g),
                                            metal_graph_hc_comb(g),
                                            DS4_N_EMBD,
                                            DS4_N_HC) != 0;
        } else if (ok && !fuse_shared_down_hc) {
            ok = ds4_gpu_hc_expand_add_split_tensor(metal_graph_after_ffn_hc(g),
                                                      metal_graph_routed_out(g),
                                                      metal_graph_shared_out(g),
                                                      metal_graph_after_attn_hc(g),
                                                      metal_graph_hc_split(g),
                                                      DS4_N_EMBD,
                                                      DS4_N_HC) != 0;
        }
        DS4_METAL_PROFILE_DECODE_STAGE("ffn_hc_post");
        if (ok) {
            metal_graph_debug_dump_tensor("hc_ffn_post", metal_graph_after_ffn_hc(g), hc_dim, il, pos);
        }
        return ok;
    }
    if (overlap_selected_shared) {
        uint64_t selected_event = 0;
        if (ok) ok = ds4_gpu_signal_selected_readback_ready(&selected_event) != 0;
        metal_graph_selected_async_load async_load = {0};
        bool async_load_started = false;
        const bool async_early_commit =
            async_selected_load &&
            metal_graph_use_iq2_selected_async_early_commit(g);
        if (ok && async_selected_load) {
            ok = metal_graph_selected_async_load_start(&async_load,
                                                       g,
                                                       model,
                                                       layer,
                                                       il,
                                                       selected_event,
                                                       gate_expert_bytes,
                                                       down_expert_bytes);
            async_load_started = ok;
        }
        if (ok && async_early_commit) {
            ok = ds4_gpu_flush_commands() != 0;
        }
        if (ok && fuse_shared_gate_up) {
            ok = ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(metal_graph_shared_gate(g),
                                                             metal_graph_shared_up(g),
                                                             metal_graph_shared_mid(g),
                                                             model->map,
                                                             model->size,
                                                             layer->ffn_gate_shexp->abs_offset,
                                                             layer->ffn_up_shexp->abs_offset,
                                                             DS4_N_EMBD,
                                                             shared_dim,
                                                             metal_graph_ffn_norm(g),
                                                             DS4_SWIGLU_CLAMP_EXP) != 0;
        } else if (ok) {
            if (ok) ok = metal_graph_matmul_dense_quant_tensor(metal_graph_shared_gate(g),
                                                               model,
                                                               layer->ffn_gate_shexp,
                                                               DS4_N_EMBD,
                                                               shared_dim,
                                                               metal_graph_ffn_norm(g),
                                                               1);
            if (ok) ok = metal_graph_matmul_dense_quant_tensor(metal_graph_shared_up(g),
                                                               model,
                                                               layer->ffn_up_shexp,
                                                               DS4_N_EMBD,
                                                               shared_dim,
                                                               metal_graph_ffn_norm(g),
                                                               1);
            if (ok) ok = ds4_gpu_swiglu_tensor(metal_graph_shared_mid(g), metal_graph_shared_gate(g), metal_graph_shared_up(g),
                                               shared_dim, DS4_SWIGLU_CLAMP_EXP, 1.0f) != 0;
        }
        DS4_METAL_PROFILE_DECODE_STAGE("shared_gate_up");
        if (ok && !fuse_shared_down_hc) {
            ok = metal_graph_matmul_dense_quant_tensor(metal_graph_shared_out(g),
                                                       model,
                                                       layer->ffn_down_shexp,
                                                       shared_dim,
                                                       DS4_N_EMBD,
                                                       metal_graph_shared_mid(g),
                                                       1);
        }
        DS4_METAL_PROFILE_DECODE_STAGE("shared_down");
        if (async_load_started) {
            const bool flush_ok = ds4_gpu_flush_commands() != 0;
            bool finish_ok =
                metal_graph_selected_async_load_finish(&async_load);
            if (!finish_ok && async_load.ids_ok) {
                /* The worker read valid ids but could not stage the load
                 * (it is not allowed to wait on in-flight cache entries).
                 * This thread is, so retry the same load synchronously. */
                const ds4_gpu_stream_expert_table retry_table =
                    graph_stream_expert_table_make(model,
                                                   layer,
                                                   il,
                                                   gate_expert_bytes,
                                                   down_expert_bytes);
                finish_ok =
                    ds4_gpu_stream_expert_cache_begin_selected_load(
                            &retry_table,
                            async_load.selected_ids,
                            DS4_N_EXPERT_USED) != 0 &&
                    ds4_gpu_routed_moe_set_selected_override(
                            async_load.selected_ids,
                            DS4_N_EXPERT_USED) != 0;
            }
            ok = ok && flush_ok && finish_ok;
        } else if (ok) {
            ok = ds4_gpu_commit_and_wait_selected_readback(selected_event,
                                                           "selected-id shared-overlap") != 0;
        }
        if (ok && !async_load_started) {
            int32_t selected_ids[DS4_MAX_EXPERT_USED];
            ok = ds4_gpu_tensor_read(metal_graph_router_selected(g),
                                     0,
                                     selected_ids,
                                     (uint64_t)DS4_N_EXPERT_USED * sizeof(selected_ids[0])) != 0 &&
                 ds4_gpu_routed_moe_set_selected_override(selected_ids,
                                                          DS4_N_EXPERT_USED) != 0;
            if (ok) {
                const ds4_gpu_stream_expert_table table =
                    graph_stream_expert_table_make(model,
                                                   layer,
                                                   il,
                                                   gate_expert_bytes,
                                                   down_expert_bytes);
                ok = ds4_gpu_stream_expert_cache_begin_selected_load(
                            &table,
                            selected_ids,
                            DS4_N_EXPERT_USED) != 0;
            }
        }
        if (ok) ok = ds4_gpu_routed_moe_one_tensor(metal_graph_routed_out(g),
                                                     metal_graph_routed_gate(g),
                                                     metal_graph_routed_up(g),
                                                     metal_graph_routed_mid(g),
                                                     metal_graph_routed_down(g),
                                                     model->map, model->size,
                                                     layer->ffn_gate_exps->abs_offset,
                                                     layer->ffn_up_exps->abs_offset,
                                                     layer->ffn_down_exps->abs_offset,
                                                     layer->ffn_gate_exps->type,
                                                     layer->ffn_down_exps->type,
                                                     gate_expert_bytes, gate_row_bytes,
                                                     down_expert_bytes, down_row_bytes,
                                                     (uint32_t)expert_in_dim,
                                                     (uint32_t)down_in_dim,
                                                     (uint32_t)routed_out_dim,
                                                     metal_graph_router_selected(g), metal_graph_router_weights(g),
                                                     DS4_N_EXPERT,
                                                     DS4_N_EXPERT_USED, DS4_SWIGLU_CLAMP_EXP, metal_graph_ffn_norm(g),
                                                     NULL,
                                                     il,
                                                     false) != 0;
        DS4_METAL_PROFILE_DECODE_STAGE("routed_moe");
        if (ok) {
            metal_graph_debug_dump_tensor("ffn_moe_gate_clamped", metal_graph_routed_gate(g),
                                          (uint64_t)DS4_N_EXPERT_USED * down_in_dim, il, pos);
            metal_graph_debug_dump_tensor("ffn_moe_up_clamped", metal_graph_routed_up(g),
                                          (uint64_t)DS4_N_EXPERT_USED * down_in_dim, il, pos);
            metal_graph_debug_dump_tensor("ffn_moe_weighted_swiglu", metal_graph_routed_mid(g),
                                          (uint64_t)DS4_N_EXPERT_USED * down_in_dim, il, pos);
            metal_graph_debug_dump_tensor("ffn_moe_down", metal_graph_routed_down(g),
                                          (uint64_t)DS4_N_EXPERT_USED * DS4_N_EMBD, il, pos);
            metal_graph_debug_dump_tensor("ffn_moe_out", metal_graph_routed_out(g), DS4_N_EMBD, il, pos);
        }
        if (ok && fuse_shared_down_hc) {
            ok = ds4_gpu_shared_down_hc_expand_q8_0_tensor(metal_graph_after_ffn_hc(g),
                                                             metal_graph_shared_out(g),
                                                             model->map,
                                                             model->size,
                                                             layer->ffn_down_shexp->abs_offset,
                                                             shared_dim,
                                                             DS4_N_EMBD,
                                                             metal_graph_shared_mid(g),
                                                             metal_graph_routed_out(g),
                                                             metal_graph_after_attn_hc(g),
                                                             metal_graph_hc_split(g),
                                                             DS4_N_EMBD,
                                                             DS4_N_HC) != 0;
        }
        DS4_METAL_PROFILE_DECODE_STAGE("shared_down");
        if (ok) {
            metal_graph_debug_dump_tensor("ffn_shexp", metal_graph_shared_out(g), DS4_N_EMBD, il, pos);
        }
        if (ok && keep_ffn_out) {
            ok = metal_graph_ensure_ffn_out(g) &&
                 ds4_gpu_add_tensor(metal_graph_ffn_out(g), metal_graph_shared_out(g), metal_graph_routed_out(g), DS4_N_EMBD) != 0;
        }
        if (ok && keep_ffn_out) {
            metal_graph_debug_dump_tensor("ffn_out", metal_graph_ffn_out(g), DS4_N_EMBD, il, pos);
        }
        if (ok && metal_graph_directional_steering_ffn_enabled(g)) {
            ok = metal_graph_apply_directional_steering_ffn(g, metal_graph_ffn_out(g), il, 1);
        }
        if (ok && metal_graph_directional_steering_ffn_enabled(g)) {
            ok = ds4_gpu_hc_expand_tensor(metal_graph_after_ffn_hc(g),
                                            metal_graph_ffn_out(g),
                                            metal_graph_after_attn_hc(g),
                                            metal_graph_hc_post(g),
                                            metal_graph_hc_comb(g),
                                            DS4_N_EMBD,
                                            DS4_N_HC) != 0;
        } else if (ok && !fuse_shared_down_hc) {
            ok = ds4_gpu_hc_expand_add_split_tensor(metal_graph_after_ffn_hc(g),
                                                      metal_graph_routed_out(g),
                                                      metal_graph_shared_out(g),
                                                      metal_graph_after_attn_hc(g),
                                                      metal_graph_hc_split(g),
                                                      DS4_N_EMBD,
                                                      DS4_N_HC) != 0;
        }
        DS4_METAL_PROFILE_DECODE_STAGE("ffn_hc_post");
        if (ok) {
            metal_graph_debug_dump_tensor("hc_ffn_post", metal_graph_after_ffn_hc(g), hc_dim, il, pos);
        }
        return ok;
    }
    /* Under the TP split the routed experts run after the shared expert so
     * the sum6 kernel can fold the shared partial and write the slab slot
     * directly (no separate local add). */
    const bool tp_fold_ffn = tp_split_shared &&
                             !keep_ffn_out &&
                             !metal_graph_directional_steering_ffn_enabled(g);
    if (ok && !tp_fold_ffn && !cuda_tp_moe) ok = ds4_gpu_routed_moe_one_tensor(metal_graph_routed_out(g),
                                                 metal_graph_routed_gate(g),
                                                 metal_graph_routed_up(g),
                                                 metal_graph_routed_mid(g),
                                                 metal_graph_routed_down(g),
                                                 model->map, model->size,
                                                 layer->ffn_gate_exps->abs_offset,
                                                 layer->ffn_up_exps->abs_offset,
                                                 layer->ffn_down_exps->abs_offset,
                                                 layer->ffn_gate_exps->type,
                                                 layer->ffn_down_exps->type,
                                                 gate_expert_bytes, gate_row_bytes,
                                                 down_expert_bytes, down_row_bytes,
                                                 (uint32_t)expert_in_dim,
                                                 (uint32_t)down_in_dim,
                                                 (uint32_t)routed_out_dim,
                                                 metal_graph_router_selected(g), metal_graph_router_weights(g),
                                                 DS4_N_EXPERT,
                                                 DS4_N_EXPERT_USED, DS4_SWIGLU_CLAMP_EXP, metal_graph_ffn_norm(g),
                                                 NULL,
                                                 il,
                                                 false) != 0;
    DS4_METAL_PROFILE_DECODE_STAGE("routed_moe");
    if (ok) {
        metal_graph_debug_dump_tensor("ffn_moe_gate_clamped", metal_graph_routed_gate(g),
                                      (uint64_t)DS4_N_EXPERT_USED * down_in_dim, il, pos);
        metal_graph_debug_dump_tensor("ffn_moe_up_clamped", metal_graph_routed_up(g),
                                      (uint64_t)DS4_N_EXPERT_USED * down_in_dim, il, pos);
    }
    if (ok) {
        metal_graph_debug_dump_tensor("ffn_moe_weighted_swiglu", metal_graph_routed_mid(g),
                                      (uint64_t)DS4_N_EXPERT_USED * down_in_dim, il, pos);
    }
    if (ok) {
        metal_graph_debug_dump_tensor("ffn_moe_down", metal_graph_routed_down(g),
                                      (uint64_t)DS4_N_EXPERT_USED * DS4_N_EMBD, il, pos);
    }
    if (ok) {
        metal_graph_debug_dump_tensor("ffn_moe_out", metal_graph_routed_out(g), DS4_N_EMBD, il, pos);
    }
    if (phase == METAL_DECODE_LAYER_TO_SHARED_MID ||
        phase == METAL_DECODE_LAYER_FROM_QA_KV_RAW_TO_SHARED_MID) {
        return ok;
    }
    if (ok && tp_split_shared) {
        /* Shared expert lane slice: the fused gate/up/swiglu kernel covers
         * this rank's half of the intermediate (row slicing is pure offset
         * math), compact at the buffer base; the down k-slice below turns
         * it into a partial output. */
        const uint32_t tp_half = shared_dim / 2;
        uint64_t shexp_row_bytes = 0;
        ok = metal_graph_dense_quant_row_bytes(layer->ffn_gate_shexp,
                                               DS4_N_EMBD,
                                               &shexp_row_bytes) &&
             layer->ffn_gate_shexp->type == layer->ffn_up_shexp->type;
        const uint64_t tp_lane_off = (uint64_t)g->tp_rank * tp_half * shexp_row_bytes;
        ok = ok && (tp_half % 32u) == 0;
        if (!ok) {
            fprintf(stderr, "ds4: TP shared expert width %u is not sliceable\n", shared_dim);
        }
        if (ok && layer->ffn_gate_shexp->type == DS4_TENSOR_Q8_0) {
            ok = ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(metal_graph_shared_gate(g),
                                                           metal_graph_shared_up(g),
                                                           metal_graph_shared_mid(g),
                                                           model->map,
                                                           model->size,
                                                           layer->ffn_gate_shexp->abs_offset + tp_lane_off,
                                                           layer->ffn_up_shexp->abs_offset + tp_lane_off,
                                                           DS4_N_EMBD,
                                                           tp_half,
                                                           metal_graph_ffn_norm(g),
                                                           DS4_SWIGLU_CLAMP_EXP) != 0;
        } else if (ok) {
            ok = metal_graph_matmul_dense_quant_abs(metal_graph_shared_gate(g),
                                                    model,
                                                    layer->ffn_gate_shexp,
                                                    layer->ffn_gate_shexp->abs_offset + tp_lane_off,
                                                    DS4_N_EMBD,
                                                    tp_half,
                                                    metal_graph_ffn_norm(g),
                                                    1);
            if (ok) ok = metal_graph_matmul_dense_quant_abs(metal_graph_shared_up(g),
                                                            model,
                                                            layer->ffn_up_shexp,
                                                            layer->ffn_up_shexp->abs_offset + tp_lane_off,
                                                            DS4_N_EMBD,
                                                            tp_half,
                                                            metal_graph_ffn_norm(g),
                                                            1);
            if (ok) ok = ds4_gpu_swiglu_tensor(metal_graph_shared_mid(g), metal_graph_shared_gate(g), metal_graph_shared_up(g),
                                               tp_half, DS4_SWIGLU_CLAMP_EXP, 1.0f) != 0;
        }
    } else if (ok && fuse_shared_gate_up) {
        ok = ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(metal_graph_shared_gate(g),
                                                         metal_graph_shared_up(g),
                                                         metal_graph_shared_mid(g),
                                                         model->map,
                                                         model->size,
                                                         layer->ffn_gate_shexp->abs_offset,
                                                         layer->ffn_up_shexp->abs_offset,
                                                         DS4_N_EMBD,
                                                         shared_dim,
                                                         metal_graph_ffn_norm(g),
                                                         DS4_SWIGLU_CLAMP_EXP) != 0;
    } else {
        if (ok) ok = metal_graph_matmul_dense_quant_tensor(metal_graph_shared_gate(g),
                                                           model,
                                                           layer->ffn_gate_shexp,
                                                           DS4_N_EMBD,
                                                           shared_dim,
                                                           metal_graph_ffn_norm(g),
                                                           1);
        if (ok) ok = metal_graph_matmul_dense_quant_tensor(metal_graph_shared_up(g),
                                                           model,
                                                           layer->ffn_up_shexp,
                                                           DS4_N_EMBD,
                                                           shared_dim,
                                                           metal_graph_ffn_norm(g),
                                                           1);
        if (ok) ok = ds4_gpu_swiglu_tensor(metal_graph_shared_mid(g), metal_graph_shared_gate(g), metal_graph_shared_up(g),
                                           shared_dim, DS4_SWIGLU_CLAMP_EXP, 1.0f) != 0;
    }
    DS4_METAL_PROFILE_DECODE_STAGE("shared_gate_up");
    if (ok && cuda_tp_ep_reduce_deferred) {
        ok = metal_graph_cuda_tp_ep_finish_reduce(
                g,
                cuda_tp_home_tier,
                cuda_tp_partner_tier,
                cuda_tp_ep_direct_return,
                cuda_tp_ep_return_bytes,
                !cuda_tp_ep_fused_hc_reduce);
    }
    if (ok && cuda_tp_moe_peer_copy_deferred) {
        ok = ds4_gpu_tensor_copy_xdev(
                g->tp_peer_tmp_by_tier[cuda_tp_home_tier],
                g->routed_out_by_tier[cuda_tp_partner_tier],
                (uint64_t)DS4_N_EMBD * sizeof(float)) != 0;
        cuda_tp_moe_peer_tmp = ok;
    }
    if (ok && cuda_tp_shared_fold) {
        bool switched_to_partner = false;
        ok = ds4_gpu_set_current_device(cuda_tp_partner_tier) == 0;
        switched_to_partner = ok;
        if (ok) {
            ok = ds4_gpu_add_tensor(
                    g->routed_out_by_tier[cuda_tp_partner_tier],
                    g->routed_out_by_tier[cuda_tp_partner_tier],
                    g->shared_out_by_tier[cuda_tp_partner_tier],
                    DS4_N_EMBD) != 0;
        }
        if (switched_to_partner &&
            ds4_gpu_set_current_device(cuda_tp_home_tier) != 0) {
            ok = false;
        }
        if (ok) {
            ok = ds4_gpu_add_tensor(metal_graph_routed_out(g),
                                    metal_graph_routed_out(g),
                                    metal_graph_shared_out(g),
                                    DS4_N_EMBD) != 0;
        }
        if (ok) {
            ok = ds4_gpu_tensor_copy_xdev(
                    g->tp_peer_tmp_by_tier[cuda_tp_home_tier],
                    g->routed_out_by_tier[cuda_tp_partner_tier],
                    (uint64_t)DS4_N_EMBD * sizeof(float)) != 0;
            cuda_tp_shared_fold_peer_tmp = ok;
        }
        if (ok) {
            ok = cuda_tp_shared_fold_peer_tmp &&
                 ds4_gpu_hc_expand_add_split_tensor(
                         metal_graph_after_ffn_hc(g),
                         metal_graph_routed_out(g),
                         g->tp_peer_tmp_by_tier[cuda_tp_home_tier],
                         metal_graph_after_attn_hc(g),
                         metal_graph_hc_split(g),
                         DS4_N_EMBD,
                         DS4_N_HC) != 0;
        }
    } else if (ok && cuda_tp_shared) {
        /* shared_out already contains the reduced local and partner partials. */
    } else if (ok && cuda_tp_ep_fused_hc_reduce) {
        ok = ds4_gpu_shared_down_hc_expand_owned_q8_0_tensor(
                metal_graph_after_ffn_hc(g),
                metal_graph_shared_out(g),
                model->map,
                model->size,
                layer->ffn_down_shexp->abs_offset,
                shared_dim,
                DS4_N_EMBD,
                metal_graph_shared_mid(g),
                metal_graph_routed_down(g),
                g->tp_peer_tmp_by_tier[cuda_tp_home_tier],
                metal_graph_router_selected(g),
                DS4_N_EXPERT / 2u,
                metal_graph_after_attn_hc(g),
                metal_graph_hc_split(g),
                DS4_N_EMBD,
                DS4_N_HC) != 0;
    } else if (ok && fuse_shared_down_hc) {
        if (cuda_tp_moe_peer_tmp) {
            ok = ds4_gpu_shared_down_hc_expand_add_q8_0_tensor(
                    metal_graph_after_ffn_hc(g),
                    metal_graph_shared_out(g),
                    model->map,
                    model->size,
                    layer->ffn_down_shexp->abs_offset,
                    shared_dim,
                    DS4_N_EMBD,
                    metal_graph_shared_mid(g),
                    metal_graph_routed_out(g),
                    g->tp_peer_tmp_by_tier[cuda_tp_home_tier],
                    metal_graph_after_attn_hc(g),
                    metal_graph_hc_split(g),
                    DS4_N_EMBD,
                    DS4_N_HC) != 0;
        } else {
            ok = ds4_gpu_shared_down_hc_expand_q8_0_tensor(
                    metal_graph_after_ffn_hc(g),
                    metal_graph_shared_out(g),
                    model->map,
                    model->size,
                    layer->ffn_down_shexp->abs_offset,
                    shared_dim,
                    DS4_N_EMBD,
                    metal_graph_shared_mid(g),
                    metal_graph_routed_out(g),
                    metal_graph_after_attn_hc(g),
                    metal_graph_hc_split(g),
                    DS4_N_EMBD,
                    DS4_N_HC) != 0;
        }
    } else if (ok && tp_split_shared) {
        ok = metal_graph_matmul_dense_quant_kslice(metal_graph_shared_out(g),
                                                   model,
                                                   layer->ffn_down_shexp,
                                                   shared_dim,
                                                   (uint64_t)g->tp_rank * (shared_dim / 2),
                                                   shared_dim / 2,
                                                   DS4_N_EMBD,
                                                   metal_graph_shared_mid(g),
                                                   0);
    } else if (ok) {
        ok = metal_graph_matmul_dense_quant_tensor(metal_graph_shared_out(g),
                                                   model,
                                                   layer->ffn_down_shexp,
                                                   shared_dim,
                                                   DS4_N_EMBD,
                                                   metal_graph_shared_mid(g),
                                                   1);
    }
    DS4_METAL_PROFILE_DECODE_STAGE("shared_down");
    if (ok) {
        metal_graph_debug_dump_tensor("ffn_shexp", metal_graph_shared_out(g), DS4_N_EMBD, il, pos);
    }
    if (ok && tp_fold_ffn) {
        ok = ds4_gpu_routed_moe_one_tensor(
                g->tp_out[il * DS4_TP_GATES_PER_LAYER + DS4_TP_GATE_FFN],
                metal_graph_routed_gate(g),
                metal_graph_routed_up(g),
                metal_graph_routed_mid(g),
                metal_graph_routed_down(g),
                model->map, model->size,
                layer->ffn_gate_exps->abs_offset,
                layer->ffn_up_exps->abs_offset,
                layer->ffn_down_exps->abs_offset,
                layer->ffn_gate_exps->type,
                layer->ffn_down_exps->type,
                gate_expert_bytes, gate_row_bytes,
                down_expert_bytes, down_row_bytes,
                (uint32_t)expert_in_dim,
                (uint32_t)down_in_dim,
                (uint32_t)routed_out_dim,
                metal_graph_router_selected(g), metal_graph_router_weights(g),
                DS4_N_EXPERT,
                DS4_N_EXPERT_USED, DS4_SWIGLU_CLAMP_EXP, metal_graph_ffn_norm(g),
                metal_graph_shared_out(g),
                il,
                false) != 0;
        DS4_METAL_PROFILE_DECODE_STAGE("routed_moe_folded");
    }
    ds4_gpu_tensor *tp_ffn_a = NULL;    /* rank0/rank1 partials consumed */
    ds4_gpu_tensor *tp_ffn_b = NULL;    /* directly by the HC expand */
    if (ok && g->tp_world == 2) {
        /* Gate FFN: local partial = shared expert + owned routed experts.
         * The HC expand below already sums two block vectors, so after the
         * exchange the two rank partials feed it directly (canonical rank
         * order) with no separate combine dispatch.  The paths that need
         * the materialized sum (ffn_out consumers) still builds it in
         * routed_out. */
        const uint32_t tp_slot = il * DS4_TP_GATES_PER_LAYER + DS4_TP_GATE_FFN;
        if (!tp_fold_ffn) {
            ok = ds4_gpu_add_tensor(g->tp_out[tp_slot], metal_graph_shared_out(g), metal_graph_routed_out(g),
                                    DS4_N_EMBD) != 0;
        }
        if (ok) ok = ds4_gpu_tp_gate_encode(il, DS4_TP_GATE_FFN) != 0;
        if (ok) {
            ds4_gpu_tensor *first = g->tp_rank == 0 ? g->tp_out[tp_slot] : g->tp_in[tp_slot];
            ds4_gpu_tensor *second = g->tp_rank == 0 ? g->tp_in[tp_slot] : g->tp_out[tp_slot];
            if (keep_ffn_out || metal_graph_directional_steering_ffn_enabled(g)) {
                ok = ds4_gpu_add_tensor(metal_graph_routed_out(g), first, second, DS4_N_EMBD) != 0;
            } else {
                tp_ffn_a = first;
                tp_ffn_b = second;
            }
        }
    }
    if (ok && keep_ffn_out) {
        ok = metal_graph_ensure_ffn_out(g) &&
             ds4_gpu_add_tensor(metal_graph_ffn_out(g),
                                g->tp_world == 2 ? g->tp_zero : metal_graph_shared_out(g),
                                metal_graph_routed_out(g), DS4_N_EMBD) != 0;
    }
    if (ok && keep_ffn_out) {
        metal_graph_debug_dump_tensor("ffn_out", metal_graph_ffn_out(g), DS4_N_EMBD, il, pos);
    }
    if (ok && metal_graph_directional_steering_ffn_enabled(g)) {
        ok = metal_graph_apply_directional_steering_ffn(g, metal_graph_ffn_out(g), il, 1);
    }
    if (ok && metal_graph_directional_steering_ffn_enabled(g)) {
        ok = ds4_gpu_hc_expand_tensor(metal_graph_after_ffn_hc(g),
                                        metal_graph_ffn_out(g),
                                        metal_graph_after_attn_hc(g),
                                        metal_graph_hc_post(g),
                                        metal_graph_hc_comb(g),
                                        DS4_N_EMBD,
                                        DS4_N_HC) != 0;
    } else if (ok && !cuda_tp_shared_fold && !fuse_shared_down_hc) {
        ok = ds4_gpu_hc_expand_add_split_tensor(metal_graph_after_ffn_hc(g),
                                                  tp_ffn_a ? tp_ffn_a : metal_graph_routed_out(g),
                                                  tp_ffn_a ? tp_ffn_b :
                                                  (g->tp_world == 2 ? g->tp_zero : metal_graph_shared_out(g)),
                                                  metal_graph_after_attn_hc(g),
                                                  metal_graph_hc_split(g),
                                                  DS4_N_EMBD,
                                                  DS4_N_HC) != 0;
    }
    DS4_METAL_PROFILE_DECODE_STAGE("ffn_hc_post");
#undef DS4_METAL_PROFILE_DECODE_STAGE
    if (ok) {
        metal_graph_debug_dump_tensor("hc_ffn_post", metal_graph_after_ffn_hc(g), hc_dim, il, pos);
    }
    return ok;
}

static bool metal_graph_encode_decode_layer(
        ds4_gpu_graph  *g,
        const ds4_model        *model,
        const ds4_layer_weights *layer,
        uint32_t                il,
        uint32_t                pos,
        ds4_gpu_tensor       *raw_cache,
        uint32_t                raw_cap,
        uint32_t                raw_row,
        uint32_t                n_raw,
        int                     token) {
    return metal_graph_encode_decode_layer_phase(
            g, model, layer, il, pos, raw_cache, raw_cap, raw_row, n_raw,
            token, METAL_DECODE_LAYER_FULL);
}

static bool metal_graph_output_logits_head_matmul(
        ds4_gpu_graph        *g,
        const ds4_model      *model,
        const ds4_weights    *weights,
        ds4_gpu_tensor       *norm_full,
        ds4_gpu_tensor       *dst_logits,
        uint32_t              n_tokens,
        uint64_t              vocab_dim);

/* Encode the final HC collapse, output norm, and vocab projection on Metal. */
static bool metal_graph_encode_output_head(
        ds4_gpu_graph *g,
        const ds4_model       *model,
        const ds4_weights     *weights,
        uint64_t               vocab_dim) {
    /* switch to head_tier before the output-head pipeline.
     * Single-tier (placement == NULL): no-op (head_tier == 0 == active_tier).
     * Note: head_tier was captured in metal_graph_alloc_raw_cap; this
     * helper consults it directly (and also covers the case where the
     * preceding decode layer ran on a different tier — copy_xdev ferries
     * the active cur_hc across the boundary). */
    if (g->placement) {
        if (!metal_graph_set_active_tier_decode(g, g->head_tier)) return false;
    }
    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const bool output_stage_profile = g->output_stage_profile;
    double output_stage_t0 = output_stage_profile ? now_sec() : 0.0;
#define DS4_METAL_PROFILE_OUTPUT_STAGE(name) do { \
        if (ok && output_stage_profile) { \
            ok = metal_graph_layer_stage_profile_boundary("output", (name), DS4_N_LAYER, 0, 1, &output_stage_t0); \
        } \
    } while (0)
    bool ok = ds4_gpu_rms_norm_plain_tensor(metal_graph_flat_hc(g), metal_graph_cur_hc(g), (uint32_t)hc_dim, DS4_RMS_EPS) != 0;
    DS4_METAL_PROFILE_OUTPUT_STAGE("hc_flat_norm");
    if (ok) ok = ds4_gpu_matmul_f16_tensor(metal_graph_output_pre(g),
                                             model->map,
                                             model->size,
                                             weights->output_hc_fn->abs_offset,
                                             hc_dim,
                                             DS4_N_HC,
                                             metal_graph_flat_hc(g),
                                             1) != 0;
    DS4_METAL_PROFILE_OUTPUT_STAGE("hc_pre");
    if (ok) {
        metal_graph_debug_dump_tensor("result_hc_pre", metal_graph_output_pre(g), DS4_N_HC, DS4_N_LAYER, 0);
    }
    if (ok) ok = ds4_gpu_output_hc_weights_tensor(metal_graph_output_weights(g),
                                                    metal_graph_output_pre(g),
                                                    model->map,
                                                    model->size,
                                                    weights->output_hc_scale->abs_offset,
                                                    weights->output_hc_base->abs_offset,
                                                    DS4_N_HC,
                                                    DS4_HC_EPS) != 0;
    DS4_METAL_PROFILE_OUTPUT_STAGE("hc_weights");
    if (ok) {
        metal_graph_debug_dump_tensor("result_hc_weights", metal_graph_output_weights(g), DS4_N_HC, DS4_N_LAYER, 0);
    }
    bool output_sum_norm_fused = false;
#if defined(__APPLE__)
    if (ok) {
        output_sum_norm_fused =
            ds4_gpu_hc_weighted_sum_norm_tensor(
                metal_graph_output_embd(g),
                metal_graph_output_norm(g),
                metal_graph_cur_hc(g),
                metal_graph_output_weights(g),
                model->map,
                model->size,
                weights->output_norm->abs_offset,
                DS4_N_EMBD,
                DS4_N_HC,
                DS4_RMS_EPS) != 0;
        if (!output_sum_norm_fused &&
            getenv("DS4_METAL_REQUIRE_OUTPUT_HC_SUM_NORM_FUSION") != NULL) {
            ok = false;
        }
    }
#endif
    if (ok && !output_sum_norm_fused) {
        ok = ds4_gpu_hc_weighted_sum_tensor(metal_graph_output_embd(g),
                                              metal_graph_cur_hc(g),
                                              metal_graph_output_weights(g),
                                              DS4_N_EMBD,
                                              DS4_N_HC) != 0;
    }
    DS4_METAL_PROFILE_OUTPUT_STAGE("hc_weighted_sum");
    if (ok) {
        metal_graph_debug_dump_tensor("result_hc", metal_graph_output_embd(g), DS4_N_EMBD, DS4_N_LAYER, 0);
    }
    if (ok && !output_sum_norm_fused) {
        ok = ds4_gpu_rms_norm_weight_tensor(metal_graph_output_norm(g),
                                              metal_graph_output_embd(g),
                                              model->map,
                                              model->size,
                                              weights->output_norm->abs_offset,
                                              DS4_N_EMBD,
                                              DS4_RMS_EPS) != 0;
    }
    DS4_METAL_PROFILE_OUTPUT_STAGE("output_norm");
    if (ok) {
        metal_graph_debug_dump_tensor("result_norm", metal_graph_output_norm(g), DS4_N_EMBD, DS4_N_LAYER, 0);
    }
    if (ok && g->tp_world == 2 && g->tp_logits_half) {
        /* Vocab-split: this rank computes its half of the head rows into
         * its logits view; the halves are bit-identical to the full head
         * (same kernel, same rows) and the worker ships its half to the
         * leader after the eval. */
        const uint64_t tp_vhalf = vocab_dim / 2u;
        uint64_t head_row_bytes = 0;
        ok = metal_graph_dense_quant_row_bytes(weights->output,
                                               DS4_N_EMBD,
                                               &head_row_bytes);
        if (ok) ok = metal_graph_matmul_dense_quant_abs(g->tp_logits_half,
                                                        model,
                                                        weights->output,
                                                        weights->output->abs_offset +
                                                            (uint64_t)g->tp_rank * tp_vhalf * head_row_bytes,
                                                        DS4_N_EMBD,
                                                        tp_vhalf,
                                                        metal_graph_output_norm(g),
                                                        1);
    } else if (ok && g->cuda_tp_ep && g->cuda_tp_output) {
        ok = metal_graph_output_logits_head_matmul(
                g, model, weights, metal_graph_output_norm(g),
                metal_graph_logits(g), 1, vocab_dim);
    } else if (ok) {
        ok = metal_graph_matmul_dense_quant_tensor(metal_graph_logits(g),
                                                   model,
                                                   weights->output,
                                                   DS4_N_EMBD,
                                                   vocab_dim,
                                                   metal_graph_output_norm(g),
                                                   1);
    }
    if (ok) {
        metal_graph_debug_dump_tensor("result_output", metal_graph_logits(g), vocab_dim, DS4_N_LAYER, 0);
    }
#undef DS4_METAL_PROFILE_OUTPUT_STAGE
    return ok;
}

/* Greedy-only output head: compute one local top-1 candidate per output TP
 * split and leave the full split logits on their owning tiers.  This avoids
 * gathering the whole vocabulary row back to the head tier when the caller only
 * needs the next argmax token. */
static bool metal_graph_encode_output_head_split_top1(
        ds4_gpu_graph *g,
        const ds4_model       *model,
        const ds4_weights     *weights,
        uint64_t               vocab_dim,
        int                    cuda_tp_output_tiers[DS4_MAX_GPUS],
        uint32_t              *cuda_tp_output_ways_out) {
    if (!g || !model || !weights ||
        !cuda_tp_output_tiers || !cuda_tp_output_ways_out ||
        vocab_dim > UINT32_MAX) {
        return false;
    }
    *cuda_tp_output_ways_out = 0;

    if (g->placement) {
        if (!metal_graph_set_active_tier_decode(g, g->head_tier)) return false;
    }
    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    bool ok = ds4_gpu_rms_norm_plain_tensor(metal_graph_flat_hc(g),
                                            metal_graph_cur_hc(g),
                                            (uint32_t)hc_dim,
                                            DS4_RMS_EPS) != 0;
    if (ok) ok = ds4_gpu_matmul_f16_tensor(metal_graph_output_pre(g),
                                             model->map,
                                             model->size,
                                             weights->output_hc_fn->abs_offset,
                                             hc_dim,
                                             DS4_N_HC,
                                             metal_graph_flat_hc(g),
                                             1) != 0;
    if (ok) ok = ds4_gpu_output_hc_weights_tensor(metal_graph_output_weights(g),
                                                    metal_graph_output_pre(g),
                                                    model->map,
                                                    model->size,
                                                    weights->output_hc_scale->abs_offset,
                                                    weights->output_hc_base->abs_offset,
                                                    DS4_N_HC,
                                                    DS4_HC_EPS) != 0;
    if (ok) ok = ds4_gpu_hc_weighted_sum_tensor(metal_graph_output_embd(g),
                                                  metal_graph_cur_hc(g),
                                                  metal_graph_output_weights(g),
                                                  DS4_N_EMBD,
                                                  DS4_N_HC) != 0;
    if (ok) ok = ds4_gpu_rms_norm_weight_tensor(metal_graph_output_norm(g),
                                                  metal_graph_output_embd(g),
                                                  model->map,
                                                  model->size,
                                                  weights->output_norm->abs_offset,
                                                  DS4_N_EMBD,
                                                  DS4_RMS_EPS) != 0;
    if (!ok) return false;

    const uint32_t cuda_tp_output_ways =
        g->cuda_tp_output ? metal_graph_cuda_tp_output_tiers(g, cuda_tp_output_tiers) : 0;
    const bool cuda_tp_output =
        g->cuda_tp_output &&
        cuda_tp_output_ways >= 2u &&
        weights->output->type == DS4_TENSOR_Q8_0 &&
        weights->output->ndim == 2 &&
        weights->output->dim[0] == DS4_N_EMBD &&
        weights->output->dim[1] == vocab_dim &&
        vocab_dim >= 2;
    if (!cuda_tp_output) return false;

    for (uint32_t i = 0; i < cuda_tp_output_ways; i++) {
        const int t = cuda_tp_output_tiers[i];
        if (t < 0 || t >= DS4_MAX_GPUS ||
            !g->output_norm_by_tier[t] ||
            !g->logits_by_tier[t] ||
            !g->comp_selected_by_tier[t] ||
            !g->comp_mask_by_tier[t]) {
            return false;
        }
    }

    const bool fused_top1 = metal_graph_cuda_output_fused_top1_requested();
    const uint64_t row_bytes = metal_graph_q8_0_row_bytes(DS4_N_EMBD);
    uint64_t split_start[DS4_MAX_GPUS] = {0};
    uint64_t split_count[DS4_MAX_GPUS] = {0};
    ds4_gpu_tensor split_logits[DS4_MAX_GPUS];
    memset(split_logits, 0, sizeof(split_logits));

    for (uint32_t i = 0; ok && i < cuda_tp_output_ways; i++) {
        const int t = cuda_tp_output_tiers[i];
        split_start[i] = (vocab_dim * (uint64_t)i) / cuda_tp_output_ways;
        const uint64_t split_end =
            (vocab_dim * (uint64_t)(i + 1u)) / cuda_tp_output_ways;
        split_count[i] = split_end - split_start[i];
        if (split_count[i] == 0 || split_count[i] > UINT32_MAX) {
            ok = false;
            break;
        }
        if (fused_top1) {
            if (t != g->head_tier) {
                ok = ds4_gpu_tensor_copy_xdev(
                        g->output_norm_by_tier[t],
                        metal_graph_output_norm(g),
                        (uint64_t)DS4_N_EMBD * sizeof(float)) != 0;
            }
        } else if (t == g->head_tier) {
            ok = metal_graph_borrow_tensor_view(&split_logits[i],
                                                metal_graph_logits(g),
                                                split_start[i] * sizeof(float),
                                                split_count[i] * sizeof(float));
        } else {
            ok = metal_graph_borrow_tensor_view(&split_logits[i],
                                                g->logits_by_tier[t],
                                                0,
                                                split_count[i] * sizeof(float)) &&
                 ds4_gpu_tensor_copy_xdev(
                        g->output_norm_by_tier[t],
                        metal_graph_output_norm(g),
                        (uint64_t)DS4_N_EMBD * sizeof(float)) != 0;
        }
    }

    for (uint32_t i = 1; ok && i < cuda_tp_output_ways; i++) {
        const int t = cuda_tp_output_tiers[i];
        ok = ds4_gpu_set_current_device(t) == 0;
        if (ok && fused_top1) {
            ok = ds4_gpu_matmul_q8_0_top1_tensor(g->comp_selected_by_tier[t],
                                                 g->comp_mask_by_tier[t],
                                                 model->map,
                                                 model->size,
                                                 weights->output->abs_offset +
                                                     split_start[i] * row_bytes,
                                                 DS4_N_EMBD,
                                                 split_count[i],
                                                 g->output_norm_by_tier[t],
                                                 (uint32_t)split_start[i]) != 0;
        } else if (ok) {
            ok = ds4_gpu_matmul_q8_0_tensor(&split_logits[i],
                                            model->map,
                                            model->size,
                                            weights->output->abs_offset +
                                                split_start[i] * row_bytes,
                                            DS4_N_EMBD,
                                            split_count[i],
                                            g->output_norm_by_tier[t],
                                            1) != 0;
        }
        if (ok && !fused_top1) {
            ok = ds4_gpu_indexer_top1_value_tensor(g->comp_selected_by_tier[t],
                                                   g->comp_mask_by_tier[t],
                                                   &split_logits[i],
                                                   (uint32_t)split_count[i],
                                                   1,
                                                   (uint32_t)split_start[i]) != 0;
        }
    }
    if (ok) ok = ds4_gpu_set_current_device(g->head_tier) == 0;
    if (ok && fused_top1) {
        ok = ds4_gpu_matmul_q8_0_top1_tensor(g->comp_selected_by_tier[g->head_tier],
                                             g->comp_mask_by_tier[g->head_tier],
                                             model->map,
                                             model->size,
                                             weights->output->abs_offset,
                                             DS4_N_EMBD,
                                             split_count[0],
                                             metal_graph_output_norm(g),
                                             (uint32_t)split_start[0]) != 0;
    } else if (ok) {
        ok = ds4_gpu_matmul_q8_0_tensor(&split_logits[0],
                                        model->map,
                                        model->size,
                                        weights->output->abs_offset,
                                        DS4_N_EMBD,
                                        split_count[0],
                                        metal_graph_output_norm(g),
                                        1) != 0;
    }
    if (ok && !fused_top1) {
        ok = ds4_gpu_indexer_top1_value_tensor(g->comp_selected_by_tier[g->head_tier],
                                               g->comp_mask_by_tier[g->head_tier],
                                               &split_logits[0],
                                               (uint32_t)split_count[0],
                                               1,
                                               (uint32_t)split_start[0]) != 0;
    }
    for (uint32_t i = 1; ok && i < cuda_tp_output_ways; i++) {
        const int t = cuda_tp_output_tiers[i];
        ds4_gpu_tensor head_id_dst;
        ds4_gpu_tensor head_value_dst;
        ok = metal_graph_borrow_tensor_view(&head_id_dst,
                                            g->comp_selected_by_tier[g->head_tier],
                                            (uint64_t)i * sizeof(uint32_t),
                                            sizeof(uint32_t)) &&
             metal_graph_borrow_tensor_view(&head_value_dst,
                                            g->comp_mask_by_tier[g->head_tier],
                                            (uint64_t)i * sizeof(float),
                                            sizeof(float)) &&
             ds4_gpu_tensor_copy_xdev3(&head_id_dst,
                                        g->comp_selected_by_tier[t],
                                        sizeof(uint32_t),
                                        &head_value_dst,
                                        g->comp_mask_by_tier[t],
                                        sizeof(float),
                                        NULL,
                                        NULL,
                                        0) != 0;
    }
    if (ok) {
        ok = ds4_gpu_set_current_device(g->head_tier) == 0;
        *cuda_tp_output_ways_out = cuda_tp_output_ways;
    }
    return ok;
}

static bool metal_graph_read_output_split_top1(
        ds4_gpu_graph *g,
        uint32_t       output_ways,
        int           *top_id) {
    if (!g || !top_id || output_ways == 0 || output_ways > DS4_MAX_GPUS) {
        return false;
    }
    bool have_best = false;
    uint32_t best_id = 0;
    float best_value = 0.0f;
    uint32_t cand_ids[DS4_MAX_GPUS] = {0};
    float cand_values[DS4_MAX_GPUS] = {0.0f};
    bool ok = ds4_gpu_tensor_read(g->comp_selected_by_tier[g->head_tier],
                                  0,
                                  cand_ids,
                                  (uint64_t)output_ways * sizeof(cand_ids[0])) != 0 &&
              ds4_gpu_tensor_read(g->comp_mask_by_tier[g->head_tier],
                                  0,
                                  cand_values,
                                  (uint64_t)output_ways * sizeof(cand_values[0])) != 0;
    for (uint32_t i = 0; ok && i < output_ways; i++) {
        const uint32_t cand_id = cand_ids[i];
        const float cand_value = cand_values[i];
        if (!have_best ||
            cand_value > best_value ||
            (cand_value == best_value && cand_id < best_id)) {
            have_best = true;
            best_id = cand_id;
            best_value = cand_value;
        }
    }
    ok = ok && have_best && best_id <= (uint32_t)INT32_MAX;
    if (ok) *top_id = (int)best_id;
    return ok;
}

/* Batched output head for speculative verification.
 *
 * A target verifier only needs top-1 ids for intermediate draft rows and full
 * logits for the last accepted row.  Running the normal one-row output head in
 * a loop serializes the HC collapse, output norm, and Q8 vocab projection.  For
 * tiny MTP suffixes we instead process all rows together and let the GPU reduce
 * each row to a top id; the CPU reads back just those ids plus the last row's
 * logits needed to continue the exact target stream. */
/* Shared vocab-head matmul: pads small batches to 8 rows for the exact-mma Q8
 * kernel and shards the vocabulary across output-TP tiers. */
static bool metal_graph_output_logits_head_matmul(
        ds4_gpu_graph        *g,
        const ds4_model      *model,
        const ds4_weights    *weights,
        ds4_gpu_tensor       *norm_full,
        ds4_gpu_tensor       *dst_logits,
        uint32_t              n_tokens,
        uint64_t              vocab_dim) {
    if (!g || !model || !weights || !norm_full || n_tokens == 0 ||
        !dst_logits ||
        ds4_gpu_tensor_bytes(dst_logits) <
            (uint64_t)n_tokens * vocab_dim * sizeof(float)) {
        return false;
    }
    const uint32_t head_rows =
        (n_tokens > 1 && n_tokens < 8 &&
         ds4_gpu_tensor_bytes(dst_logits) >= 8u * vocab_dim * sizeof(float) &&
         ds4_gpu_tensor_bytes(norm_full) >=
             8u * DS4_N_EMBD * sizeof(float)) ? 8u : n_tokens;
    ds4_gpu_tensor *output_norm =
        ds4_gpu_tensor_view(norm_full,
                            0,
                            (uint64_t)head_rows * DS4_N_EMBD * sizeof(float));
    ds4_gpu_tensor *logits =
        ds4_gpu_tensor_view(dst_logits,
                            0,
                            (uint64_t)head_rows * vocab_dim * sizeof(float));
    bool ok = output_norm && logits;
    if (ok && head_rows > n_tokens) {
        ds4_gpu_tensor *pad =
            ds4_gpu_tensor_view(norm_full,
                                (uint64_t)n_tokens * DS4_N_EMBD * sizeof(float),
                                (uint64_t)(head_rows - n_tokens) * DS4_N_EMBD *
                                    sizeof(float));
        ok = pad &&
             ds4_gpu_tensor_fill_f32(pad,
                                     0.0f,
                                     (uint64_t)(head_rows - n_tokens) *
                                         DS4_N_EMBD) != 0;
        ds4_gpu_tensor_free(pad);
    }
    /* Output TP for the speculative batch, mirroring the decode head: each
     * device matmuls its VRAM-resident vocab shard. Shard outputs land
     * compactly in logits_by_tier[t] ([head_rows x split]) and are gathered
     * into spec_logits rows. */
    int tp_tiers[DS4_MAX_GPUS] = {0};
    const uint32_t tp_ways = (ok && g->cuda_tp_output)
        ? metal_graph_cuda_tp_output_tiers(g, tp_tiers) : 0;
    bool tp_ok = ok && tp_ways >= 2u &&
                 weights->output->type == DS4_TENSOR_Q8_0 &&
                 weights->output->ndim == 2 &&
                 weights->output->dim[0] == DS4_N_EMBD &&
                 weights->output->dim[1] == vocab_dim &&
                 head_rows <= DS4_DSPARK_MAX_BLOCK_SIZE &&
                 getenv("DS4_DSPARK_VERIFY_HEAD_NO_TP") == NULL;
    for (uint32_t i = 0; tp_ok && i < tp_ways; i++) {
        const int t = tp_tiers[i];
        tp_ok = t >= 0 && t < DS4_MAX_GPUS &&
                g->logits_by_tier[t] &&
                ds4_gpu_tensor_bytes(g->logits_by_tier[t]) >=
                    (uint64_t)head_rows *
                        ((vocab_dim + tp_ways - 1u) / tp_ways) *
                        sizeof(float) &&
                (t == g->active_tier ||
                 (g->batch_ffn_norm_by_tier[t] &&
                  ds4_gpu_tensor_bytes(g->batch_ffn_norm_by_tier[t]) >=
                      (uint64_t)head_rows * DS4_N_EMBD * sizeof(float)));
    }
    if (tp_ok) {
        const uint64_t row_bytes = metal_graph_q8_0_row_bytes(DS4_N_EMBD);
        const int home_tier = g->active_tier;
        uint64_t split_start[DS4_MAX_GPUS] = {0};
        uint64_t split_count[DS4_MAX_GPUS] = {0};
        for (uint32_t i = 0; ok && i < tp_ways; i++) {
            const int t = tp_tiers[i];
            split_start[i] = (vocab_dim * (uint64_t)i) / tp_ways;
            const uint64_t split_end =
                (vocab_dim * (uint64_t)(i + 1u)) / tp_ways;
            split_count[i] = split_end - split_start[i];
            if (split_count[i] == 0) { ok = false; break; }
            if (t != home_tier) {
                ok = ds4_gpu_tensor_copy_xdev(
                        g->batch_ffn_norm_by_tier[t],
                        output_norm,
                        (uint64_t)head_rows * DS4_N_EMBD *
                            sizeof(float)) != 0;
            }
        }
        for (uint32_t i = 0; ok && i < tp_ways; i++) {
            const int t = tp_tiers[i];
            ok = ds4_gpu_set_current_device(t) == 0;
            if (!ok) break;
            ds4_gpu_tensor *shard_out =
                ds4_gpu_tensor_view(g->logits_by_tier[t],
                                    0,
                                    (uint64_t)head_rows * split_count[i] *
                                        sizeof(float));
            ds4_gpu_tensor *shard_in = t == home_tier ?
                NULL :
                ds4_gpu_tensor_view(g->batch_ffn_norm_by_tier[t],
                                    0,
                                    (uint64_t)head_rows * DS4_N_EMBD *
                                        sizeof(float));
            ok = shard_out && (t == home_tier || shard_in) &&
                 ds4_gpu_matmul_q8_0_tensor(shard_out,
                                            model->map,
                                            model->size,
                                            weights->output->abs_offset +
                                                split_start[i] * row_bytes,
                                            DS4_N_EMBD,
                                            split_count[i],
                                            t == home_tier ? output_norm :
                                                             shard_in,
                                            head_rows) != 0;
            ds4_gpu_tensor_free(shard_in);
            ds4_gpu_tensor_free(shard_out);
        }
        if (ok) ok = ds4_gpu_set_current_device(home_tier) == 0;
        for (uint32_t i = 0; ok && i < tp_ways; i++) {
            const int t = tp_tiers[i];
            for (uint32_t r = 0; ok && r < n_tokens; r++) {
                ds4_gpu_tensor *dst =
                    ds4_gpu_tensor_view(dst_logits,
                                        ((uint64_t)r * vocab_dim +
                                         split_start[i]) * sizeof(float),
                                        split_count[i] * sizeof(float));
                ds4_gpu_tensor *src =
                    ds4_gpu_tensor_view(g->logits_by_tier[t],
                                        (uint64_t)r * split_count[i] *
                                            sizeof(float),
                                        split_count[i] * sizeof(float));
                ok = dst && src &&
                     ds4_gpu_tensor_copy_xdev(dst,
                                              src,
                                              split_count[i] *
                                                  sizeof(float)) != 0;
                ds4_gpu_tensor_free(src);
                ds4_gpu_tensor_free(dst);
            }
        }
    } else if (ok && !(g->cuda_tp_ep && g->cuda_tp_output)) {
        ok = ds4_gpu_matmul_q8_0_tensor(logits,
                                        model->map,
                                        model->size,
                                        weights->output->abs_offset,
                                        DS4_N_EMBD,
                                        vocab_dim,
                                        output_norm,
                                        head_rows) != 0;
    } else if (ok) {
        /* The expert-parallel cache stores only output vocabulary shards, so
         * a single-device full-head fallback would access uncached weights. */
        ok = false;
    }
    ds4_gpu_tensor_free(logits);
    ds4_gpu_tensor_free(output_norm);
    return ok;
}

static bool metal_graph_encode_output_head_batch(
        ds4_gpu_graph *g,
        const ds4_model       *model,
        const ds4_weights     *weights,
        uint32_t               n_tokens,
        uint64_t               vocab_dim) {
    if (n_tokens == 0 || n_tokens > g->prefill_cap || !g->spec_logits) return false;

    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    ds4_gpu_tensor *output_pre = NULL;
    ds4_gpu_tensor *output_weights = NULL;
    ds4_gpu_tensor *output_embd = NULL;
    ds4_gpu_tensor *output_norm = NULL;

    bool ok = true;
    output_pre = ds4_gpu_tensor_view(metal_graph_batch_hc_mix(g),
                                       0,
                                       (uint64_t)n_tokens * DS4_N_HC * sizeof(float));
    output_weights = ds4_gpu_tensor_view(metal_graph_batch_hc_split(g),
                                           0,
                                           (uint64_t)n_tokens * DS4_N_HC * sizeof(float));
    output_embd = ds4_gpu_tensor_view(metal_graph_batch_ffn_cur(g),
                                        0,
                                        (uint64_t)n_tokens * DS4_N_EMBD * sizeof(float));
    output_norm = ds4_gpu_tensor_view(metal_graph_batch_ffn_norm(g),
                                        0,
                                        (uint64_t)n_tokens * DS4_N_EMBD * sizeof(float));
    ok = output_pre && output_weights && output_embd && output_norm;

    if (ok) ok = ds4_gpu_rms_norm_plain_rows_tensor(metal_graph_batch_flat_hc(g),
                                                      metal_graph_batch_cur_hc(g),
                                                      (uint32_t)hc_dim,
                                                      n_tokens,
                                                      DS4_RMS_EPS) != 0;
    if (ok) ok = ds4_gpu_matmul_f16_tensor(output_pre,
                                             model->map,
                                             model->size,
                                             weights->output_hc_fn->abs_offset,
                                             hc_dim,
                                             DS4_N_HC,
                                             metal_graph_batch_flat_hc(g),
                                             n_tokens) != 0;
    if (ok) ok = ds4_gpu_output_hc_weights_tensor(output_weights,
                                                    output_pre,
                                                    model->map,
                                                    model->size,
                                                    weights->output_hc_scale->abs_offset,
                                                    weights->output_hc_base->abs_offset,
                                                    DS4_N_HC,
                                                    DS4_HC_EPS) != 0;
    if (ok) ok = ds4_gpu_hc_weighted_sum_tensor(output_embd,
                                                  metal_graph_batch_cur_hc(g),
                                                  output_weights,
                                                  DS4_N_EMBD,
                                                  DS4_N_HC) != 0;
    if (ok) ok = ds4_gpu_rms_norm_weight_rows_tensor(output_norm,
                                                       output_embd,
                                                       model->map,
                                                       model->size,
                                                       weights->output_norm->abs_offset,
                                                       DS4_N_EMBD,
                                                       n_tokens,
                                                       DS4_RMS_EPS) != 0;
    if (ok) ok = metal_graph_output_logits_head_matmul(
            g, model, weights, metal_graph_batch_ffn_norm(g),
            g->spec_logits, n_tokens, vocab_dim);

    ds4_gpu_tensor_free(output_norm);
    ds4_gpu_tensor_free(output_embd);
    ds4_gpu_tensor_free(output_weights);
    ds4_gpu_tensor_free(output_pre);
    return ok;
}

static bool metal_graph_matmul_plain_tensor(
        ds4_gpu_tensor       *out,
        const ds4_model        *model,
        const ds4_tensor       *w,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok) {
    if (w->type == DS4_TENSOR_F16) {
        return ds4_gpu_matmul_f16_tensor(out, model->map, model->size,
                                           w->abs_offset, in_dim, out_dim, x, n_tok) != 0;
    }
    if (w->type == DS4_TENSOR_F32) {
        return ds4_gpu_matmul_f32_tensor(out, model->map, model->size,
                                           w->abs_offset, in_dim, out_dim, x, n_tok) != 0;
    }
    if (w->type == DS4_TENSOR_Q8_0) {
        return ds4_gpu_matmul_q8_0_tensor(out, model->map, model->size,
                                            w->abs_offset, in_dim, out_dim, x, n_tok) != 0;
    }
    if (tensor_type_is_dense_quant(w->type)) {
        return ds4_gpu_matmul_quant_tensor(out,
                                           model->map,
                                           model->size,
                                           w->abs_offset,
                                           w->type,
                                           in_dim,
                                           out_dim,
                                           x,
                                           n_tok) != 0;
    }
    fprintf(stderr, "ds4: Metal plain matmul does not support %s\n", tensor_type_name(w->type));
    return false;
}

static bool metal_graph_dense_quant_row_bytes(
        const ds4_tensor *w,
        uint64_t          in_dim,
        uint64_t         *row_bytes) {
    if (row_bytes) *row_bytes = 0;
    if (!w || !row_bytes || !tensor_type_is_dense_quant(w->type)) return false;
    return tensor_nbytes(w->type, in_dim, row_bytes);
}

static bool metal_graph_matmul_dense_quant_abs(
        ds4_gpu_tensor       *out,
        const ds4_model        *model,
        const ds4_tensor       *w,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok) {
    if (!w || !tensor_type_is_dense_quant(w->type)) return false;
    return ds4_gpu_matmul_quant_tensor(out,
                                       model->map,
                                       model->size,
                                       weight_offset,
                                       w->type,
                                       in_dim,
                                       out_dim,
                                       x,
                                       n_tok) != 0;
}

static bool metal_graph_matmul_dense_quant_tensor(
        ds4_gpu_tensor       *out,
        const ds4_model        *model,
        const ds4_tensor       *w,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok) {
    if (!w) return false;
    return metal_graph_matmul_dense_quant_abs(out,
                                             model,
                                             w,
                                             w->abs_offset,
                                             in_dim,
                                             out_dim,
                                             x,
                                             n_tok);
}

static bool metal_graph_matmul_dense_quant_kslice(
        ds4_gpu_tensor       *out,
        const ds4_model        *model,
        const ds4_tensor       *w,
        uint64_t                full_in_dim,
        uint64_t                k_off,
        uint64_t                k_cnt,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                x_elem_off) {
    if (!w || !tensor_type_is_dense_quant(w->type)) return false;
    return ds4_gpu_matmul_quant_kslice_tensor(out,
                                             model->map,
                                             model->size,
                                             w->abs_offset,
                                             w->type,
                                             full_in_dim,
                                             k_off,
                                             k_cnt,
                                             out_dim,
                                             x,
                                             x_elem_off) != 0;
}

static bool metal_graph_attention_output_dense_quant_low(
        ds4_gpu_tensor       *low,
        ds4_gpu_graph        *g,
        const ds4_model        *model,
        const ds4_tensor       *out_a,
        uint64_t                group_dim,
        uint64_t                rank,
        uint32_t                group0,
        uint32_t                group_cnt,
        const ds4_gpu_tensor *heads) {
    (void)g;
    if (!low || !model || !out_a || !heads ||
        group_dim == 0 || rank == 0 || group_cnt == 0) {
        return false;
    }
    if (out_a->type == DS4_TENSOR_Q8_0 && group0 == 0) {
        return ds4_gpu_attention_output_low_q8_tensor(low,
                                                      model->map,
                                                      model->size,
                                                      out_a->abs_offset,
                                                      group_dim,
                                                      rank,
                                                      group_cnt,
                                                      heads) != 0;
    }
    if (out_a->type == DS4_TENSOR_Q4_K) {
        return ds4_gpu_attention_output_low_q4_K_slice_tensor(low,
                                                             model->map,
                                                             model->size,
                                                             out_a->abs_offset,
                                                             group_dim,
                                                             rank,
                                                             group0,
                                                             group_cnt,
                                                             heads) != 0;
    }
    uint64_t row_bytes = 0;
    if (!metal_graph_dense_quant_row_bytes(out_a, group_dim, &row_bytes)) return false;
    const uint64_t group_weight_bytes = rank * row_bytes;
    bool ok = true;
    for (uint32_t i = 0; ok && i < group_cnt; i++) {
        ds4_gpu_tensor *head_view = ds4_gpu_tensor_view(
                heads,
                (uint64_t)i * group_dim * sizeof(float),
                group_dim * sizeof(float));
        ds4_gpu_tensor *low_view = ds4_gpu_tensor_view(
                low,
                (uint64_t)i * rank * sizeof(float),
                rank * sizeof(float));
        ok = head_view && low_view &&
             metal_graph_matmul_dense_quant_abs(low_view,
                                                model,
                                                out_a,
                                                out_a->abs_offset +
                                                    (uint64_t)(group0 + i) * group_weight_bytes,
                                                group_dim,
                                                rank,
                                                head_view,
                                                1);
        ds4_gpu_tensor_free(low_view);
        ds4_gpu_tensor_free(head_view);
    }
    return ok;
}

static bool metal_graph_attention_output_dense_quant_tp(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *low,
        ds4_gpu_graph        *g,
        const ds4_model        *model,
        const ds4_tensor       *out_a,
        const ds4_tensor       *out_b,
        uint64_t                group_dim,
        uint64_t                rank,
        uint32_t                n_groups_total,
        uint32_t                group0,
        uint32_t                group_cnt,
        uint64_t                out_dim,
        const ds4_gpu_tensor *heads) {
    if (!out || !low || !g || !model || !out_a || !out_b || !heads ||
        group0 + group_cnt > n_groups_total) {
        return false;
    }
    if (out_a->type == DS4_TENSOR_Q8_0 && out_b->type == DS4_TENSOR_Q8_0) {
        return ds4_gpu_attention_output_q8_tp_tensor(out,
                                                     low,
                                                     model->map,
                                                     model->size,
                                                     out_a->abs_offset,
                                                     out_b->abs_offset,
                                                     group_dim,
                                                     rank,
                                                     n_groups_total,
                                                     group0,
                                                     group_cnt,
                                                     out_dim,
                                                     heads) != 0;
    }
    if (!metal_graph_attention_output_dense_quant_low(low,
                                                     g,
                                                     model,
                                                     out_a,
                                                     group_dim,
                                                     rank,
                                                     group0,
                                                     group_cnt,
                                                     heads)) {
        return false;
    }
    return metal_graph_matmul_dense_quant_kslice(out,
                                                model,
                                                out_b,
                                                (uint64_t)n_groups_total * rank,
                                                (uint64_t)group0 * rank,
                                                (uint64_t)group_cnt * rank,
                                                out_dim,
                                                low,
                                                0);
}

static bool metal_graph_attention_output_dense_quant_batch(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *low,
        ds4_gpu_graph        *g,
        const ds4_model        *model,
        const ds4_tensor       *out_a,
        const ds4_tensor       *out_b,
        uint64_t                group_dim,
        uint64_t                rank,
        uint32_t                n_groups,
        uint64_t                out_dim,
        const ds4_gpu_tensor *heads,
        uint32_t                n_tokens) {
    if (!out || !low || !g || !model || !out_a || !out_b || !heads ||
        n_groups == 0 || n_tokens == 0) {
        return false;
    }
    if (out_a->type == DS4_TENSOR_Q8_0 && out_b->type == DS4_TENSOR_Q8_0) {
        return ds4_gpu_attention_output_q8_batch_tensor(out,
                                                        low,
                                                        metal_graph_batch_group_tmp(g),
                                                        metal_graph_batch_low_tmp(g),
                                                        model->map,
                                                        model->size,
                                                        out_a->abs_offset,
                                                        out_b->abs_offset,
                                                        group_dim,
                                                        rank,
                                                        n_groups,
                                                        out_dim,
                                                        heads,
                                                        n_tokens) != 0;
    }
    if (out_a->type == DS4_TENSOR_Q4_K && n_tokens >= 32u) {
        if (ds4_gpu_attention_output_q4_K_batch_tensor(out,
                                                       low,
                                                       metal_graph_batch_group_tmp(g),
                                                       metal_graph_batch_low_tmp(g),
                                                       model->map,
                                                       model->size,
                                                       out_a->abs_offset,
                                                       out_b->abs_offset,
                                                       out_b->type,
                                                       group_dim,
                                                       rank,
                                                       n_groups,
                                                       out_dim,
                                                       heads,
                                                       n_tokens) != 0) {
            return true;
        }
    }

    const uint64_t heads_row_elems = (uint64_t)n_groups * group_dim;
    const uint64_t low_row_elems = (uint64_t)n_groups * rank;
    bool ok = true;
    for (uint32_t t = 0; ok && t < n_tokens; t++) {
        ds4_gpu_tensor *heads_row = ds4_gpu_tensor_view(
                heads,
                (uint64_t)t * heads_row_elems * sizeof(float),
                heads_row_elems * sizeof(float));
        ds4_gpu_tensor *low_row = ds4_gpu_tensor_view(
                low,
                (uint64_t)t * low_row_elems * sizeof(float),
                low_row_elems * sizeof(float));
        ds4_gpu_tensor *out_row = ds4_gpu_tensor_view(
                out,
                (uint64_t)t * out_dim * sizeof(float),
                out_dim * sizeof(float));
        ok = heads_row && low_row && out_row &&
             metal_graph_attention_output_dense_quant_low(low_row,
                                                          g,
                                                          model,
                                                          out_a,
                                                          group_dim,
                                                          rank,
                                                          0,
                                                          n_groups,
                                                          heads_row);
        if (ok) ok = metal_graph_matmul_dense_quant_tensor(out_row,
                                                           model,
                                                           out_b,
                                                           low_row_elems,
                                                           out_dim,
                                                           low_row,
                                                           1);
        ds4_gpu_tensor_free(out_row);
        ds4_gpu_tensor_free(low_row);
        ds4_gpu_tensor_free(heads_row);
    }
    return ok;
}

static bool metal_graph_matmul_q8_0_named_tensor(
        const char             *module,
        uint32_t                il,
        uint32_t                pos0,
        ds4_gpu_tensor       *out,
        const ds4_model        *model,
        const ds4_tensor       *w,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok) {
    (void)module;
    (void)il;
    (void)pos0;
    return metal_graph_matmul_dense_quant_tensor(out,
                                                model,
                                                w,
                                                in_dim,
                                                out_dim,
                                                x,
                                                n_tok);
}

static bool metal_graph_encode_output_head_mtp(
        ds4_gpu_graph       *g,
        const ds4_model       *base_model,
        const ds4_weights     *base_weights,
        const ds4_model       *mtp_model,
        const ds4_mtp_weights *mtp,
        uint64_t               vocab_dim) {
    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    bool ok = ds4_gpu_rms_norm_plain_tensor(metal_graph_flat_hc(g), metal_graph_cur_hc(g), (uint32_t)hc_dim, DS4_RMS_EPS) != 0;
    if (ok) ok = metal_graph_matmul_plain_tensor(metal_graph_output_pre(g), mtp_model, mtp->hc_head_fn,
                                                 hc_dim, DS4_N_HC, metal_graph_flat_hc(g), 1);
    if (ok) ok = ds4_gpu_output_hc_weights_tensor(metal_graph_output_weights(g),
                                                    metal_graph_output_pre(g),
                                                    mtp_model->map,
                                                    mtp_model->size,
                                                    mtp->hc_head_scale->abs_offset,
                                                    mtp->hc_head_base->abs_offset,
                                                    DS4_N_HC,
                                                    DS4_HC_EPS) != 0;
    if (ok) ok = ds4_gpu_hc_weighted_sum_tensor(metal_graph_output_embd(g),
                                                  metal_graph_cur_hc(g),
                                                  metal_graph_output_weights(g),
                                                  DS4_N_EMBD,
                                                  DS4_N_HC) != 0;
    if (ok) ok = ds4_gpu_rms_norm_weight_tensor(metal_graph_output_norm(g),
                                                  metal_graph_output_embd(g),
                                                  mtp_model->map,
                                                  mtp_model->size,
                                                  mtp->norm->abs_offset,
                                                  DS4_N_EMBD,
                                                  DS4_RMS_EPS) != 0;
    if (ok) ok = metal_graph_matmul_dense_quant_tensor(metal_graph_logits(g),
                                                       base_model,
                                                       base_weights->output,
                                                       DS4_N_EMBD,
                                                       vocab_dim,
                                                       metal_graph_output_norm(g),
                                                       1);
    return ok;
}

/* =========================================================================
 * Metal Diagnostic Comparisons.
 * =========================================================================
 *
 * These routines deliberately allocate CPU-side reference buffers and read
 * Metal tensors back.  They are not part of generation; command-line tests use
 * them to localize drift against the C reference pipeline.
 */

static void metal_graph_trace_layer_stages(
        ds4_gpu_graph  *g,
        const ds4_model        *model,
        const ds4_layer_weights *layer,
        const float            *cpu_in_hc,
        uint32_t                il,
        int                     token) {
    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const uint64_t q_rank = layer->attn_q_a->dim[1];
    const uint64_t q_dim = (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM;
    const uint64_t shared_in_dim = layer->ffn_gate_shexp->dim[0];
    const uint64_t shared_dim = layer->ffn_gate_shexp->dim[1];
    const uint64_t expert_in_dim = layer->ffn_gate_exps->dim[0];
    const uint64_t down_in_dim = layer->ffn_down_exps->dim[0];
    const bool routed_q8_0 =
        layer->ffn_gate_exps->type == DS4_TENSOR_Q8_0 &&
        layer->ffn_up_exps->type == DS4_TENSOR_Q8_0 &&
        layer->ffn_down_exps->type == DS4_TENSOR_Q8_0;
    const uint64_t routed_q8_x_blocks = expert_in_dim / 32u;
    const uint64_t routed_q8_mid_blocks = down_in_dim / 32u;

    float *cpu_attn_cur = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *cpu_attn_norm = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *cpu_q = xmalloc((size_t)q_dim * sizeof(float));
    float *cpu_qr_norm = xmalloc((size_t)q_rank * sizeof(float));
    float *cpu_kv = xmalloc((size_t)DS4_N_HEAD_DIM * sizeof(float));
    float *cpu_heads = xmalloc((size_t)q_dim * sizeof(float));
    float *cpu_attn_out = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *cpu_after_attn_hc = xmalloc((size_t)hc_dim * sizeof(float));
    float *cpu_ffn_cur = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *cpu_ffn_norm = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *cpu_shared_gate = xmalloc((size_t)shared_dim * sizeof(float));
    float *cpu_shared_up = xmalloc((size_t)shared_dim * sizeof(float));
    float *cpu_shared_mid = xmalloc((size_t)shared_dim * sizeof(float));
    float *cpu_shared = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *cpu_routed = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *cpu_ffn_out = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *cpu_after_ffn_hc = xmalloc((size_t)hc_dim * sizeof(float));
    float post[4];
    float comb[16];
    float ffn_post[4];
    float ffn_comb[16];
    int selected[DS4_MAX_EXPERT_USED];
    float expert_weight[DS4_MAX_EXPERT_USED];
    const uint64_t shared_blocks = (shared_in_dim + 31) / 32;
    int8_t *shared_xq = xmalloc((size_t)shared_blocks * 32);
    float *shared_xscale = xmalloc((size_t)shared_blocks * sizeof(float));
    float *routed_mid_all = xmalloc((size_t)DS4_N_EXPERT_USED * down_in_dim * sizeof(float));
    block_q8_K *routed_xq = xmalloc((size_t)(expert_in_dim / QK_K) * sizeof(block_q8_K));
    block_q8_K *routed_midq = xmalloc((size_t)DS4_N_EXPERT_USED * (down_in_dim / QK_K) * sizeof(block_q8_K));
    int8_t *routed_q8_xq = routed_q8_0 ? xmalloc((size_t)routed_q8_x_blocks * 32u) : NULL;
    float *routed_q8_xscale = routed_q8_0 ? xmalloc((size_t)routed_q8_x_blocks * sizeof(routed_q8_xscale[0])) : NULL;
    int8_t *routed_q8_midq = routed_q8_0 ? xmalloc((size_t)DS4_N_EXPERT_USED * routed_q8_mid_blocks * 32u) : NULL;
    float *routed_q8_midscale = routed_q8_0 ? xmalloc((size_t)DS4_N_EXPERT_USED * routed_q8_mid_blocks * sizeof(routed_q8_midscale[0])) : NULL;

    hc_pre_from_state_one(model,
                          layer->hc_attn_fn,
                          layer->hc_attn_scale,
                          layer->hc_attn_base,
                          cpu_in_hc, cpu_attn_cur, post, comb);
    layer_attn_norm_one(cpu_attn_norm, model, layer, cpu_attn_cur);
    layer_q_projection_with_lora_one(model, layer, cpu_attn_norm, cpu_q, cpu_qr_norm);
    layer_kv_projection_normed_one(model, layer, cpu_attn_norm, cpu_kv);
    rope_tail_layer_inplace(cpu_q, DS4_N_HEAD, DS4_N_HEAD_DIM, DS4_N_ROT, 0, il, false);
    rope_tail_layer_inplace(cpu_kv, DS4_N_HEAD_KV, DS4_N_HEAD_DIM, DS4_N_ROT, 0, il, false);
    dsv4_fp8_kv_quantize_row_inplace_cpu(cpu_kv, DS4_N_HEAD_DIM, DS4_N_ROT);
    f16_round_inplace_cpu(cpu_kv, DS4_N_HEAD_DIM);
    layer_attention_one(cpu_heads, model, layer, cpu_q, cpu_kv);
    rope_tail_layer_inplace(cpu_heads, DS4_N_HEAD, DS4_N_HEAD_DIM, DS4_N_ROT, 0, il, true);
    layer_grouped_out_one(cpu_attn_out, model, layer, cpu_heads);
    hc_post_one(cpu_after_attn_hc, cpu_attn_out, cpu_in_hc, post, comb, DS4_N_EMBD, DS4_N_HC);
    hc_pre_from_state_one(model,
                          layer->hc_ffn_fn,
                          layer->hc_ffn_scale,
                          layer->hc_ffn_base,
                          cpu_after_attn_hc, cpu_ffn_cur, ffn_post, ffn_comb);
    rms_norm_weight(cpu_ffn_norm, cpu_ffn_cur, tensor_data(model, layer->ffn_norm), DS4_N_EMBD, DS4_RMS_EPS);
    quantize_q8_0_activation(cpu_ffn_norm, shared_xq, shared_xscale, shared_in_dim);
    matvec_q8_0_pair_prequant(cpu_shared_gate,
                              cpu_shared_up,
                              model,
                              layer->ffn_gate_shexp,
                              layer->ffn_up_shexp,
                              shared_xq,
                              shared_xscale);
    swiglu(cpu_shared_mid, cpu_shared_gate, cpu_shared_up, shared_dim, DS4_SWIGLU_CLAMP_EXP);
    matvec_q8_0(cpu_shared, model, layer->ffn_down_shexp, cpu_shared_mid);
    layer_routed_moe_one_prealloc(cpu_routed,
                                  model,
                                  layer,
                                  cpu_ffn_norm,
                                  il,
                                  token,
                                  DS4_SWIGLU_CLAMP_EXP,
                                  routed_mid_all,
                                  routed_xq,
                                  routed_midq,
                                  routed_q8_xq,
                                  routed_q8_xscale,
                                  routed_q8_midq,
                                  routed_q8_midscale);
    if (layer->ffn_gate_tid2eid) {
        layer_hash_selected_experts(selected, model, layer, token);
        layer_hash_router_weights_one(expert_weight, model, layer, cpu_ffn_norm, selected);
    } else {
        layer_topk_selected_experts(selected, expert_weight, model, layer, cpu_ffn_norm);
    }
    for (uint32_t i = 0; i < DS4_N_EMBD; i++) cpu_ffn_out[i] = cpu_shared[i] + cpu_routed[i];
    hc_post_one(cpu_after_ffn_hc, cpu_ffn_out, cpu_after_attn_hc, ffn_post, ffn_comb, DS4_N_EMBD, DS4_N_HC);

    float *gpu_attn_cur = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *gpu_attn_norm = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *gpu_q = xmalloc((size_t)q_dim * sizeof(float));
    float *gpu_kv = xmalloc((size_t)DS4_N_HEAD_DIM * sizeof(float));
    float *gpu_attn_out = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *gpu_after_attn_hc = xmalloc((size_t)hc_dim * sizeof(float));
    float *gpu_ffn_cur = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *gpu_ffn_norm = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *gpu_shared_gate = xmalloc((size_t)shared_dim * sizeof(float));
    float *gpu_shared_up = xmalloc((size_t)shared_dim * sizeof(float));
    float *gpu_shared_mid = xmalloc((size_t)shared_dim * sizeof(float));
    float *gpu_shared = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *gpu_routed_mid_all = xmalloc((size_t)DS4_N_EXPERT_USED * down_in_dim * sizeof(float));
    float *gpu_routed = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *gpu_ffn_out = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *gpu_after_ffn_hc = xmalloc((size_t)hc_dim * sizeof(float));
    int gpu_selected[DS4_MAX_EXPERT_USED];
    float gpu_expert_weight[DS4_MAX_EXPERT_USED];

    bool ok = ds4_gpu_tensor_read(metal_graph_attn_cur(g), 0, gpu_attn_cur, (uint64_t)DS4_N_EMBD * sizeof(float)) != 0 &&
              ds4_gpu_tensor_read(metal_graph_attn_norm(g), 0, gpu_attn_norm, (uint64_t)DS4_N_EMBD * sizeof(float)) != 0 &&
              ds4_gpu_tensor_read(metal_graph_q(g), 0, gpu_q, q_dim * sizeof(float)) != 0 &&
              ds4_gpu_tensor_read(metal_graph_kv(g), 0, gpu_kv, (uint64_t)DS4_N_HEAD_DIM * sizeof(float)) != 0 &&
              ds4_gpu_tensor_read(metal_graph_attn_out(g), 0, gpu_attn_out, (uint64_t)DS4_N_EMBD * sizeof(float)) != 0 &&
              ds4_gpu_tensor_read(metal_graph_after_attn_hc(g), 0, gpu_after_attn_hc, hc_dim * sizeof(float)) != 0 &&
              ds4_gpu_tensor_read(metal_graph_ffn_cur(g), 0, gpu_ffn_cur, (uint64_t)DS4_N_EMBD * sizeof(float)) != 0 &&
              ds4_gpu_tensor_read(metal_graph_ffn_norm(g), 0, gpu_ffn_norm, (uint64_t)DS4_N_EMBD * sizeof(float)) != 0 &&
              ds4_gpu_tensor_read(metal_graph_shared_gate(g), 0, gpu_shared_gate, shared_dim * sizeof(float)) != 0 &&
              ds4_gpu_tensor_read(metal_graph_shared_up(g), 0, gpu_shared_up, shared_dim * sizeof(float)) != 0 &&
              ds4_gpu_tensor_read(metal_graph_shared_mid(g), 0, gpu_shared_mid, shared_dim * sizeof(float)) != 0 &&
              ds4_gpu_tensor_read(metal_graph_shared_out(g), 0, gpu_shared, (uint64_t)DS4_N_EMBD * sizeof(float)) != 0 &&
              ds4_gpu_tensor_read(metal_graph_router_selected(g), 0, gpu_selected, sizeof(gpu_selected)) != 0 &&
              ds4_gpu_tensor_read(metal_graph_router_weights(g), 0, gpu_expert_weight, sizeof(gpu_expert_weight)) != 0 &&
              ds4_gpu_tensor_read(metal_graph_routed_mid(g), 0, gpu_routed_mid_all, (uint64_t)DS4_N_EXPERT_USED * down_in_dim * sizeof(float)) != 0 &&
              ds4_gpu_tensor_read(metal_graph_routed_out(g), 0, gpu_routed, (uint64_t)DS4_N_EMBD * sizeof(float)) != 0 &&
              ds4_gpu_tensor_read(metal_graph_ffn_out(g), 0, gpu_ffn_out, (uint64_t)DS4_N_EMBD * sizeof(float)) != 0 &&
              ds4_gpu_tensor_read(metal_graph_cur_hc(g), 0, gpu_after_ffn_hc, hc_dim * sizeof(float)) != 0;

    if (ok) {
        fprintf(stderr,
                "ds4: Metal stage layer %u attn_cur=%g/%g attn_norm=%g/%g q=%g/%g kv=%g/%g attn_out=%g/%g after_attn_hc=%g/%g ffn_cur=%g/%g ffn_norm=%g/%g shared=%g/%g router_w=%g routed=%g/%g ffn_out=%g/%g after_ffn_hc=%g/%g\n",
                il,
                max_abs_diff(cpu_attn_cur, gpu_attn_cur, DS4_N_EMBD), rms_abs_diff(cpu_attn_cur, gpu_attn_cur, DS4_N_EMBD),
                max_abs_diff(cpu_attn_norm, gpu_attn_norm, DS4_N_EMBD), rms_abs_diff(cpu_attn_norm, gpu_attn_norm, DS4_N_EMBD),
                max_abs_diff(cpu_q, gpu_q, q_dim), rms_abs_diff(cpu_q, gpu_q, q_dim),
                max_abs_diff(cpu_kv, gpu_kv, DS4_N_HEAD_DIM), rms_abs_diff(cpu_kv, gpu_kv, DS4_N_HEAD_DIM),
                max_abs_diff(cpu_attn_out, gpu_attn_out, DS4_N_EMBD), rms_abs_diff(cpu_attn_out, gpu_attn_out, DS4_N_EMBD),
                max_abs_diff(cpu_after_attn_hc, gpu_after_attn_hc, hc_dim), rms_abs_diff(cpu_after_attn_hc, gpu_after_attn_hc, hc_dim),
                max_abs_diff(cpu_ffn_cur, gpu_ffn_cur, DS4_N_EMBD), rms_abs_diff(cpu_ffn_cur, gpu_ffn_cur, DS4_N_EMBD),
                max_abs_diff(cpu_ffn_norm, gpu_ffn_norm, DS4_N_EMBD), rms_abs_diff(cpu_ffn_norm, gpu_ffn_norm, DS4_N_EMBD),
                max_abs_diff(cpu_shared, gpu_shared, DS4_N_EMBD), rms_abs_diff(cpu_shared, gpu_shared, DS4_N_EMBD),
                max_abs_diff(expert_weight, gpu_expert_weight, DS4_N_EXPERT_USED),
                max_abs_diff(cpu_routed, gpu_routed, DS4_N_EMBD), rms_abs_diff(cpu_routed, gpu_routed, DS4_N_EMBD),
                max_abs_diff(cpu_ffn_out, gpu_ffn_out, DS4_N_EMBD), rms_abs_diff(cpu_ffn_out, gpu_ffn_out, DS4_N_EMBD),
                max_abs_diff(cpu_after_ffn_hc, gpu_after_ffn_hc, hc_dim), rms_abs_diff(cpu_after_ffn_hc, gpu_after_ffn_hc, hc_dim));
        fprintf(stderr,
                "ds4: Metal shared layer %u gate=%g/%g up=%g/%g mid=%g/%g down=%g/%g\n",
                il,
                max_abs_diff(cpu_shared_gate, gpu_shared_gate, shared_dim), rms_abs_diff(cpu_shared_gate, gpu_shared_gate, shared_dim),
                max_abs_diff(cpu_shared_up, gpu_shared_up, shared_dim), rms_abs_diff(cpu_shared_up, gpu_shared_up, shared_dim),
                max_abs_diff(cpu_shared_mid, gpu_shared_mid, shared_dim), rms_abs_diff(cpu_shared_mid, gpu_shared_mid, shared_dim),
                max_abs_diff(cpu_shared, gpu_shared, DS4_N_EMBD), rms_abs_diff(cpu_shared, gpu_shared, DS4_N_EMBD));
        fprintf(stderr,
                "ds4: Metal routed layer %u mid=%g/%g out=%g/%g\n",
                il,
                max_abs_diff(routed_mid_all, gpu_routed_mid_all, DS4_N_EXPERT_USED * down_in_dim),
                rms_abs_diff(routed_mid_all, gpu_routed_mid_all, DS4_N_EXPERT_USED * down_in_dim),
                max_abs_diff(cpu_routed, gpu_routed, DS4_N_EMBD),
                rms_abs_diff(cpu_routed, gpu_routed, DS4_N_EMBD));
        if (memcmp(selected, gpu_selected, sizeof(selected)) != 0) {
            fprintf(stderr,
                    "ds4: Metal stage layer %u router selected mismatch: cpu=[%d,%d,%d,%d,%d,%d] gpu=[%d,%d,%d,%d,%d,%d]\n",
                    il,
                    selected[0], selected[1], selected[2], selected[3], selected[4], selected[5],
                    gpu_selected[0], gpu_selected[1], gpu_selected[2], gpu_selected[3], gpu_selected[4], gpu_selected[5]);
        }
    }

    free(gpu_after_ffn_hc);
    free(gpu_ffn_out);
    free(gpu_routed);
    free(gpu_routed_mid_all);
    free(gpu_shared);
    free(gpu_shared_mid);
    free(gpu_shared_up);
    free(gpu_shared_gate);
    free(gpu_ffn_norm);
    free(gpu_ffn_cur);
    free(gpu_after_attn_hc);
    free(gpu_attn_out);
    free(gpu_kv);
    free(gpu_q);
    free(gpu_attn_norm);
    free(gpu_attn_cur);
    free(routed_q8_midscale);
    free(routed_q8_midq);
    free(routed_q8_xscale);
    free(routed_q8_xq);
    free(routed_midq);
    free(routed_xq);
    free(routed_mid_all);
    free(shared_xscale);
    free(shared_xq);
    free(cpu_after_ffn_hc);
    free(cpu_ffn_out);
    free(cpu_routed);
    free(cpu_shared);
    free(cpu_shared_mid);
    free(cpu_shared_up);
    free(cpu_shared_gate);
    free(cpu_ffn_norm);
    free(cpu_ffn_cur);
    free(cpu_after_attn_hc);
    free(cpu_attn_out);
    free(cpu_heads);
    free(cpu_kv);
    free(cpu_qr_norm);
    free(cpu_q);
    free(cpu_attn_norm);
    free(cpu_attn_cur);
}

static int metal_graph_decode_test(
        const ds4_model   *model,
        const ds4_weights *weights,
        const token_vec   *prompt,
        bool               quality) {
    if (prompt->len <= 0) {
        fprintf(stderr, "ds4: Metal graph test needs a non-empty prompt\n");
        return 1;
    }

    const int token = prompt->v[0];
    const ds4_layer_weights *layer = &weights->layer[0];
    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const uint64_t q_rank = layer->attn_q_a->dim[1];
    const uint64_t q_dim = (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM;
    const uint64_t expert_in_dim = layer->ffn_gate_exps->dim[0];
    const uint64_t down_in_dim = layer->ffn_down_exps->dim[0];
    const uint64_t vocab_dim = weights->output->dim[1];
    const bool routed_q8_0 =
        layer->ffn_gate_exps->type == DS4_TENSOR_Q8_0 &&
        layer->ffn_up_exps->type == DS4_TENSOR_Q8_0 &&
        layer->ffn_down_exps->type == DS4_TENSOR_Q8_0;
    const uint64_t routed_q8_x_blocks = expert_in_dim / 32u;
    const uint64_t routed_q8_mid_blocks = down_in_dim / 32u;

    float *plain = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *cpu_hc = xmalloc((size_t)hc_dim * sizeof(float));
    float *cpu_attn_cur = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *cpu_post = xmalloc((size_t)DS4_N_HC * sizeof(float));
    float *cpu_comb = xmalloc((size_t)DS4_N_HC * DS4_N_HC * sizeof(float));
    float *cpu_attn_norm = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *cpu_qr_norm = xmalloc((size_t)q_rank * sizeof(float));
    float *cpu_q = xmalloc((size_t)q_dim * sizeof(float));
    float *cpu_kv = xmalloc((size_t)DS4_N_HEAD_DIM * sizeof(float));
    float *cpu_heads = xmalloc((size_t)q_dim * sizeof(float));
    float *cpu_attn_out = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *cpu_after_attn_hc = xmalloc((size_t)hc_dim * sizeof(float));
    float *cpu_ffn_cur = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *cpu_ffn_post = xmalloc((size_t)DS4_N_HC * sizeof(float));
    float *cpu_ffn_comb = xmalloc((size_t)DS4_N_HC * DS4_N_HC * sizeof(float));
    float *cpu_ffn_norm = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *cpu_shared = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *cpu_routed = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *cpu_ffn_out = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *cpu_after_ffn_hc = xmalloc((size_t)hc_dim * sizeof(float));
    float *cpu_logits = xmalloc((size_t)vocab_dim * sizeof(float));
    float *gpu_hc = xmalloc((size_t)hc_dim * sizeof(float));
    float *gpu_attn_cur = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *gpu_attn_norm = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *gpu_q = xmalloc((size_t)q_dim * sizeof(float));
    float *gpu_kv = xmalloc((size_t)DS4_N_HEAD_DIM * sizeof(float));
    float *gpu_raw = xmalloc((size_t)DS4_N_HEAD_DIM * sizeof(float));
    float *gpu_attn_out = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *gpu_after_attn_hc = xmalloc((size_t)hc_dim * sizeof(float));
    float *gpu_ffn_cur = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *gpu_ffn_norm = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *gpu_shared = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *gpu_routed = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *gpu_ffn_out = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *gpu_after_ffn_hc = xmalloc((size_t)hc_dim * sizeof(float));
    float *gpu_logits = xmalloc((size_t)vocab_dim * sizeof(float));
    int gpu_selected[DS4_MAX_EXPERT_USED];
    float gpu_expert_weight[DS4_MAX_EXPERT_USED];
    float *routed_mid_all = xmalloc((size_t)DS4_N_EXPERT_USED * down_in_dim * sizeof(float));
    block_q8_K *routed_xq = xmalloc((size_t)(expert_in_dim / QK_K) * sizeof(block_q8_K));
    block_q8_K *routed_midq = xmalloc((size_t)DS4_N_EXPERT_USED * (down_in_dim / QK_K) * sizeof(block_q8_K));
    int8_t *routed_q8_xq = routed_q8_0 ? xmalloc((size_t)routed_q8_x_blocks * 32u) : NULL;
    float *routed_q8_xscale = routed_q8_0 ? xmalloc((size_t)routed_q8_x_blocks * sizeof(routed_q8_xscale[0])) : NULL;
    int8_t *routed_q8_midq = routed_q8_0 ? xmalloc((size_t)DS4_N_EXPERT_USED * routed_q8_mid_blocks * 32u) : NULL;
    float *routed_q8_midscale = routed_q8_0 ? xmalloc((size_t)DS4_N_EXPERT_USED * routed_q8_mid_blocks * sizeof(routed_q8_midscale[0])) : NULL;
    int selected[DS4_MAX_EXPERT_USED];
    float expert_weight[DS4_MAX_EXPERT_USED];

    embed_token_f16(model, weights, token, plain);
    hc_from_plain_embedding(cpu_hc, plain, DS4_N_EMBD, DS4_N_HC);
    hc_pre_from_state_one(model,
                          layer->hc_attn_fn,
                          layer->hc_attn_scale,
                          layer->hc_attn_base,
                          cpu_hc, cpu_attn_cur, cpu_post, cpu_comb);
    layer_attn_norm_one(cpu_attn_norm, model, layer, cpu_attn_cur);
    layer_q_projection_with_lora_one(model, layer, cpu_attn_norm, cpu_q, cpu_qr_norm);
    layer_kv_projection_normed_one(model, layer, cpu_attn_norm, cpu_kv);
    rope_tail_layer_inplace(cpu_q, DS4_N_HEAD, DS4_N_HEAD_DIM, DS4_N_ROT, 0, 0, false);
    rope_tail_layer_inplace(cpu_kv, DS4_N_HEAD_KV, DS4_N_HEAD_DIM, DS4_N_ROT, 0, 0, false);
    dsv4_fp8_kv_quantize_row_inplace_cpu(cpu_kv, DS4_N_HEAD_DIM, DS4_N_ROT);
    f16_round_inplace_cpu(cpu_kv, DS4_N_HEAD_DIM);
    layer_attention_rows_one(cpu_heads, model, layer, cpu_q, cpu_kv, 1);
    rope_tail_layer_inplace(cpu_heads, DS4_N_HEAD, DS4_N_HEAD_DIM, DS4_N_ROT, 0, 0, true);
    layer_grouped_out_one(cpu_attn_out, model, layer, cpu_heads);
    hc_post_one(cpu_after_attn_hc, cpu_attn_out, cpu_hc, cpu_post, cpu_comb, DS4_N_EMBD, DS4_N_HC);
    hc_pre_from_state_one(model,
                          layer->hc_ffn_fn,
                          layer->hc_ffn_scale,
                          layer->hc_ffn_base,
                          cpu_after_attn_hc, cpu_ffn_cur, cpu_ffn_post, cpu_ffn_comb);
    rms_norm_weight(cpu_ffn_norm, cpu_ffn_cur, tensor_data(model, layer->ffn_norm), DS4_N_EMBD, DS4_RMS_EPS);
    layer_shared_ffn_one(cpu_shared, model, layer, cpu_ffn_norm);
    layer_routed_moe_one_prealloc(cpu_routed,
                                  model,
                                  layer,
                                  cpu_ffn_norm,
                                  0,
                                  token,
                                  DS4_SWIGLU_CLAMP_EXP,
                                  routed_mid_all,
                                  routed_xq,
                                  routed_midq,
                                  routed_q8_xq,
                                  routed_q8_xscale,
                                  routed_q8_midq,
                                  routed_q8_midscale);
    if (layer->ffn_gate_tid2eid) {
        layer_hash_selected_experts(selected, model, layer, token);
        layer_hash_router_weights_one(expert_weight, model, layer, cpu_ffn_norm, selected);
    } else {
        layer_topk_selected_experts(selected, expert_weight, model, layer, cpu_ffn_norm);
    }
    for (uint32_t i = 0; i < DS4_N_EMBD; i++) cpu_ffn_out[i] = cpu_shared[i] + cpu_routed[i];
    hc_post_one(cpu_after_ffn_hc,
                cpu_ffn_out,
                cpu_after_attn_hc,
                cpu_ffn_post,
                cpu_ffn_comb,
                DS4_N_EMBD,
                DS4_N_HC);
    output_logits_one(cpu_logits, model, weights, cpu_after_ffn_hc);

    ds4_gpu_graph g;
    bool ok = metal_graph_alloc(&g, weights, layer);
    g.quality = quality;
    g.materialize_ffn_out = true;
    if (ok) ok = ds4_gpu_begin_commands() != 0;
    if (ok) ok = ds4_gpu_embed_token_hc_tensor(metal_graph_cur_hc(&g),
                                                 model->map,
                                                 model->size,
                                                 weights->token_embd->abs_offset,
                                                 (uint32_t)weights->token_embd->dim[1],
                                                 (uint32_t)token,
                                                     DS4_N_EMBD,
                                                     DS4_N_HC) != 0;
    if (ok) ok = metal_graph_encode_decode_layer(&g,
                                               model,
                                               layer,
                                               0,
                                               0,
                                               g.layer_raw_cache[0],
                                               g.raw_cap,
                                               0,
                                               1,
                                               token);
    if (ok) {
        /* Single-tier diagnostic: swap the active-tier slots so the head
         * pipeline reads the embedded hidden state from cur_hc. */
        ds4_gpu_tensor *embedded_hc = g.cur_hc_by_tier[g.active_tier];
        g.cur_hc_by_tier[g.active_tier] = g.after_ffn_hc_by_tier[g.active_tier];
        g.after_ffn_hc_by_tier[g.active_tier] = embedded_hc;
    }
    if (ok) ok = metal_graph_encode_output_head(&g, model, weights, vocab_dim);
    if (ok) ok = ds4_gpu_end_commands() != 0;

    if (ok) {
        ok = ds4_gpu_tensor_read(metal_graph_after_ffn_hc(&g), 0, gpu_hc, hc_dim * sizeof(float)) != 0 &&
             ds4_gpu_tensor_read(metal_graph_attn_cur(&g), 0, gpu_attn_cur, (uint64_t)DS4_N_EMBD * sizeof(float)) != 0 &&
             ds4_gpu_tensor_read(metal_graph_attn_norm(&g), 0, gpu_attn_norm, (uint64_t)DS4_N_EMBD * sizeof(float)) != 0 &&
             ds4_gpu_tensor_read(metal_graph_q(&g), 0, gpu_q, q_dim * sizeof(float)) != 0 &&
             ds4_gpu_tensor_read(metal_graph_kv(&g), 0, gpu_kv, (uint64_t)DS4_N_HEAD_DIM * sizeof(float)) != 0 &&
             ds4_gpu_tensor_read(g.layer_raw_cache[0], 0, gpu_raw, (uint64_t)DS4_N_HEAD_DIM * sizeof(float)) != 0 &&
             ds4_gpu_tensor_read(metal_graph_attn_out(&g), 0, gpu_attn_out, (uint64_t)DS4_N_EMBD * sizeof(float)) != 0 &&
             ds4_gpu_tensor_read(metal_graph_after_attn_hc(&g), 0, gpu_after_attn_hc, hc_dim * sizeof(float)) != 0 &&
             ds4_gpu_tensor_read(metal_graph_ffn_cur(&g), 0, gpu_ffn_cur, (uint64_t)DS4_N_EMBD * sizeof(float)) != 0 &&
             ds4_gpu_tensor_read(metal_graph_ffn_norm(&g), 0, gpu_ffn_norm, (uint64_t)DS4_N_EMBD * sizeof(float)) != 0 &&
             ds4_gpu_tensor_read(metal_graph_shared_out(&g), 0, gpu_shared, (uint64_t)DS4_N_EMBD * sizeof(float)) != 0 &&
             ds4_gpu_tensor_read(metal_graph_router_selected(&g), 0, gpu_selected, sizeof(gpu_selected)) != 0 &&
             ds4_gpu_tensor_read(metal_graph_router_weights(&g), 0, gpu_expert_weight, sizeof(gpu_expert_weight)) != 0 &&
             ds4_gpu_tensor_read(metal_graph_routed_out(&g), 0, gpu_routed, (uint64_t)DS4_N_EMBD * sizeof(float)) != 0 &&
             ds4_gpu_tensor_read(metal_graph_ffn_out(&g), 0, gpu_ffn_out, (uint64_t)DS4_N_EMBD * sizeof(float)) != 0 &&
             ds4_gpu_tensor_read(metal_graph_cur_hc(&g), 0, gpu_after_ffn_hc, hc_dim * sizeof(float)) != 0 &&
             ds4_gpu_tensor_read(metal_graph_logits(&g), 0, gpu_logits, vocab_dim * sizeof(float)) != 0;
    }

    if (ok) {
        fprintf(stderr,
                "ds4: Metal graph test layer0 diffs: embed_hc=%g hc_pre=%g attn_norm=%g q_rope=%g kv_rope=%g raw_cache=%g attn_out=%g after_attn_hc=%g ffn_cur=%g ffn_norm=%g shared=%g router_w=%g routed=%g ffn_out=%g after_ffn_hc=%g logits=%g\n",
                max_abs_diff(cpu_hc, gpu_hc, hc_dim),
                max_abs_diff(cpu_attn_cur, gpu_attn_cur, DS4_N_EMBD),
                max_abs_diff(cpu_attn_norm, gpu_attn_norm, DS4_N_EMBD),
                max_abs_diff(cpu_q, gpu_q, q_dim),
                max_abs_diff(cpu_kv, gpu_kv, DS4_N_HEAD_DIM),
                max_abs_diff(cpu_kv, gpu_raw, DS4_N_HEAD_DIM),
                max_abs_diff(cpu_attn_out, gpu_attn_out, DS4_N_EMBD),
                max_abs_diff(cpu_after_attn_hc, gpu_after_attn_hc, hc_dim),
                max_abs_diff(cpu_ffn_cur, gpu_ffn_cur, DS4_N_EMBD),
                max_abs_diff(cpu_ffn_norm, gpu_ffn_norm, DS4_N_EMBD),
                max_abs_diff(cpu_shared, gpu_shared, DS4_N_EMBD),
                max_abs_diff(expert_weight, gpu_expert_weight, DS4_N_EXPERT_USED),
                max_abs_diff(cpu_routed, gpu_routed, DS4_N_EMBD),
                max_abs_diff(cpu_ffn_out, gpu_ffn_out, DS4_N_EMBD),
                max_abs_diff(cpu_after_ffn_hc, gpu_after_ffn_hc, hc_dim),
                max_abs_diff(cpu_logits, gpu_logits, vocab_dim));
        if (memcmp(selected, gpu_selected, sizeof(selected)) != 0) {
            fprintf(stderr,
                    "ds4: Metal graph router selected mismatch: cpu=[%d,%d,%d,%d,%d,%d] gpu=[%d,%d,%d,%d,%d,%d]\n",
                    selected[0], selected[1], selected[2], selected[3], selected[4], selected[5],
                    gpu_selected[0], gpu_selected[1], gpu_selected[2], gpu_selected[3], gpu_selected[4], gpu_selected[5]);
        }
        print_vec_stats("metal graph q", gpu_q, q_dim);
        print_vec_stats("metal graph kv", gpu_kv, DS4_N_HEAD_DIM);
        print_vec_stats("metal graph routed", gpu_routed, DS4_N_EMBD);
    } else {
        fprintf(stderr, "ds4: Metal graph test failed while encoding first decode stages\n");
        if (ds4_gpu_synchronize() == 0) {
            fprintf(stderr, "ds4: Metal synchronize after graph test failure also failed\n");
        }
    }

    metal_graph_free(&g);
    free(routed_q8_midscale);
    free(routed_q8_midq);
    free(routed_q8_xscale);
    free(routed_q8_xq);
    free(routed_midq);
    free(routed_xq);
    free(routed_mid_all);
    free(gpu_logits);
    free(gpu_after_ffn_hc);
    free(gpu_ffn_out);
    free(gpu_routed);
    free(gpu_shared);
    free(gpu_ffn_norm);
    free(gpu_ffn_cur);
    free(gpu_after_attn_hc);
    free(gpu_attn_out);
    free(gpu_raw);
    free(gpu_kv);
    free(gpu_q);
    free(gpu_attn_norm);
    free(gpu_attn_cur);
    free(gpu_hc);
    free(cpu_kv);
    free(cpu_q);
    free(cpu_attn_out);
    free(cpu_heads);
    free(cpu_ffn_norm);
    free(cpu_routed);
    free(cpu_logits);
    free(cpu_after_ffn_hc);
    free(cpu_ffn_out);
    free(cpu_shared);
    free(cpu_ffn_comb);
    free(cpu_ffn_post);
    free(cpu_ffn_cur);
    free(cpu_after_attn_hc);
    free(cpu_qr_norm);
    free(cpu_attn_norm);
    free(cpu_comb);
    free(cpu_post);
    free(cpu_attn_cur);
    free(cpu_hc);
    free(plain);
    return ok ? 0 : 1;
}

static int metal_graph_first_token_full_test(
        const ds4_model   *model,
        const ds4_weights *weights,
        const token_vec   *prompt,
        bool               quality) {
    if (prompt->len <= 0) {
        fprintf(stderr, "ds4: full Metal graph test needs a non-empty prompt\n");
        return 1;
    }

    const int token = prompt->v[0];
    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const uint64_t vocab_dim = weights->output->dim[1];
    float *cpu_hc = xmalloc((size_t)hc_dim * sizeof(float));
    float *gpu_hc = xmalloc((size_t)hc_dim * sizeof(float));
    float *cpu_logits = xmalloc((size_t)vocab_dim * sizeof(float));
    float *gpu_logits = xmalloc((size_t)vocab_dim * sizeof(float));

    forward_first_token_cpu(cpu_hc, model, weights, token);
    output_logits_one(cpu_logits, model, weights, cpu_hc);

    ds4_gpu_graph g;
    bool ok = metal_graph_alloc(&g, weights, &weights->layer[0]);
    g.quality = quality;
    const bool trace_layers = getenv("DS4_METAL_GRAPH_TRACE_LAYERS") != NULL;
    if (trace_layers && ok) {
        g.materialize_ffn_out = true;
        const bool teacher_force = getenv("DS4_METAL_GRAPH_TEACHER_FORCE") != NULL;
        const char *stage_layer_env = getenv("DS4_METAL_GRAPH_TRACE_STAGE_LAYER");
        const long stage_layer = stage_layer_env ? strtol(stage_layer_env, NULL, 10) : -1;
        float *plain = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
        float *cpu_cur = xmalloc((size_t)hc_dim * sizeof(float));
        float *cpu_next = xmalloc((size_t)hc_dim * sizeof(float));

        embed_token_f16(model, weights, token, plain);
        hc_from_plain_embedding(cpu_cur, plain, DS4_N_EMBD, DS4_N_HC);
        ok = ds4_gpu_begin_commands() != 0;
        if (ok) ok = ds4_gpu_embed_token_hc_tensor(metal_graph_cur_hc(&g),
                                                     model->map,
                                                     model->size,
                                                     weights->token_embd->abs_offset,
                                                     (uint32_t)weights->token_embd->dim[1],
                                                     (uint32_t)token,
                                                     DS4_N_EMBD,
                                                     DS4_N_HC) != 0;
        if (ok) ok = ds4_gpu_end_commands() != 0;

        for (uint32_t il = 0; ok && il < DS4_N_LAYER; il++) {
            if (teacher_force) {
                ok = ds4_gpu_tensor_write(metal_graph_cur_hc(&g), 0, cpu_cur, hc_dim * sizeof(float)) != 0;
            }
            ok = ds4_gpu_begin_commands() != 0;
            if (ok) ok = metal_graph_encode_decode_layer(&g, model, &weights->layer[il],
                                                       il, 0, g.layer_raw_cache[il], g.raw_cap, 0, 1, token);
            ds4_gpu_tensor *tmp = metal_graph_cur_hc(&g);
            g.cur_hc_by_tier[g.active_tier] = metal_graph_after_ffn_hc(&g);
            g.after_ffn_hc_by_tier[g.active_tier] = tmp;
            if (ok) ok = ds4_gpu_end_commands() != 0;

            layer_forward_self_one(cpu_next, model, &weights->layer[il], cpu_cur, il, 0, token);
            if (ok) ok = ds4_gpu_tensor_read(metal_graph_cur_hc(&g), 0, gpu_hc, hc_dim * sizeof(float)) != 0;
            if (ok) {
                fprintf(stderr,
                        "ds4: Metal full graph layer %u%s hc_max=%g hc_rms=%g\n",
                        il,
                        teacher_force ? " teacher" : "",
                        max_abs_diff(cpu_next, gpu_hc, hc_dim),
                        rms_abs_diff(cpu_next, gpu_hc, hc_dim));
                if (stage_layer == (long)il) {
                    metal_graph_trace_layer_stages(&g, model, &weights->layer[il], cpu_cur, il, token);
                }
            }
            float *ctmp = cpu_cur;
            cpu_cur = cpu_next;
            cpu_next = ctmp;
        }

        if (ok) ok = ds4_gpu_begin_commands() != 0;
        if (ok) ok = metal_graph_encode_output_head(&g, model, weights, vocab_dim);
        if (ok) ok = ds4_gpu_end_commands() != 0;

        free(cpu_next);
        free(cpu_cur);
        free(plain);
    } else {
        if (ok) ok = ds4_gpu_begin_commands() != 0;
        if (ok) ok = ds4_gpu_embed_token_hc_tensor(metal_graph_cur_hc(&g),
                                                     model->map,
                                                     model->size,
                                                     weights->token_embd->abs_offset,
                                                     (uint32_t)weights->token_embd->dim[1],
                                                     (uint32_t)token,
                                                     DS4_N_EMBD,
                                                     DS4_N_HC) != 0;

        for (uint32_t il = 0; ok && il < DS4_N_LAYER; il++) {
            ok = metal_graph_encode_decode_layer(&g, model, &weights->layer[il],
                                                 il, 0, g.layer_raw_cache[il],
                                                 g.raw_cap, 0, 1, token);
            ds4_gpu_tensor *tmp = metal_graph_cur_hc(&g);
            g.cur_hc_by_tier[g.active_tier] = metal_graph_after_ffn_hc(&g);
            g.after_ffn_hc_by_tier[g.active_tier] = tmp;
        }

        if (ok) ok = metal_graph_encode_output_head(&g, model, weights, vocab_dim);
        if (ok) ok = ds4_gpu_end_commands() != 0;
    }

    if (ok) {
        ok = ds4_gpu_tensor_read(metal_graph_cur_hc(&g), 0, gpu_hc, hc_dim * sizeof(float)) != 0 &&
             ds4_gpu_tensor_read(metal_graph_logits(&g), 0, gpu_logits, vocab_dim * sizeof(float)) != 0;
    }

    if (ok) {
        const uint64_t cpu_top = argmax_f32(cpu_logits, vocab_dim);
        const uint64_t gpu_top = argmax_f32(gpu_logits, vocab_dim);
        fprintf(stderr,
                "ds4: Metal full first-token graph diffs: final_hc_max=%g final_hc_rms=%g logits_max=%g logits_rms=%g cpu_top=%llu gpu_top=%llu cpu_top_logit=%g gpu_top_logit=%g\n",
                max_abs_diff(cpu_hc, gpu_hc, hc_dim),
                rms_abs_diff(cpu_hc, gpu_hc, hc_dim),
                max_abs_diff(cpu_logits, gpu_logits, vocab_dim),
                rms_abs_diff(cpu_logits, gpu_logits, vocab_dim),
                (unsigned long long)cpu_top,
                (unsigned long long)gpu_top,
                cpu_logits[cpu_top],
                gpu_logits[gpu_top]);
    } else {
        fprintf(stderr, "ds4: Metal full first-token graph test failed\n");
        if (ds4_gpu_synchronize() == 0) {
            fprintf(stderr, "ds4: Metal synchronize after full graph failure also failed\n");
        }
    }

    metal_graph_free(&g);
    free(gpu_logits);
    free(cpu_logits);
    free(gpu_hc);
    free(cpu_hc);
    return ok ? 0 : 1;
}

/* =========================================================================
 * Metal Release Decode and Prefill.
 * =========================================================================
 *
 * Everything below is the user-facing Metal backend.  It uses the same layer
 * encoder as diagnostics, but diagnostics are not required for normal command
 * flow and their CPU reads stay outside these generation entry points.
 */

static uint32_t metal_graph_token_split_after_layers(void) {
    uint32_t split_after_layers = 4;
#ifndef DS4_ROCM_BUILD
    const char *split_env = getenv("DS4_METAL_GRAPH_TOKEN_SPLIT_LAYERS");
    if (split_env && split_env[0]) {
        char *end = NULL;
        unsigned long v = strtoul(split_env, &end, 10);
        if (end != split_env && v <= DS4_N_LAYER) split_after_layers = (uint32_t)v;
    }
#endif
    return split_after_layers;
}

static int metal_graph_dspark_target_slot(
        const ds4_gpu_graph *g,
        uint32_t             il) {
    if (!g || !g->dspark_capture_enabled) return -1;
    for (uint32_t i = 0; i < g->dspark_target_layer_count; i++) {
        if (g->dspark_target_layers[i] == il) return (int)i;
    }
    return -1;
}

static uint32_t metal_graph_dspark_capture_complete_mask(
        const ds4_gpu_graph *g) {
    if (!g || g->dspark_target_layer_count == 0) return 0;
    return g->dspark_target_layer_count >= 32u ?
        UINT32_MAX : ((1u << g->dspark_target_layer_count) - 1u);
}

static void metal_graph_dspark_capture_note_slot(ds4_gpu_graph *g,
                                                 uint32_t       slot) {
    if (!g || slot >= g->dspark_target_layer_count) return;
    g->dspark_capture_mask |= 1u << slot;
    g->dspark_capture_valid =
        g->dspark_capture_mask == metal_graph_dspark_capture_complete_mask(g);
}

static void metal_graph_dspark_capture_row_invalidate(ds4_gpu_graph *g) {
    if (!g || !g->dspark_capture_enabled) return;
    g->dspark_capture_mask = 0;
    g->dspark_capture_checkpoint_len = 0;
    g->dspark_capture_valid = false;
}

static void metal_graph_dspark_capture_batch_invalidate(ds4_gpu_graph *g) {
    if (!g || !g->dspark_capture_enabled) return;
    g->dspark_capture_batch_mask = 0;
    g->dspark_capture_batch_start = 0;
    g->dspark_capture_batch_tokens = 0;
    g->dspark_capture_batch_valid = false;
}

static void metal_graph_dspark_capture_invalidate(ds4_gpu_graph *g) {
    metal_graph_dspark_capture_row_invalidate(g);
    metal_graph_dspark_capture_batch_invalidate(g);
}

static void metal_graph_dspark_cache_reset(ds4_gpu_graph *g) {
    if (!g) return;
    g->dspark_cache_start = 0;
    g->dspark_cache_token_start = 0;
    g->dspark_cache_len = 0;
}

static bool metal_graph_dspark_cache_window_valid(
        const ds4_gpu_graph *g,
        uint32_t             token_start,
        uint32_t             raw_start,
        uint32_t             len) {
    if (!g || len > g->dspark_cache_cap) return false;
    if (len == 0) return true;
    if (g->dspark_cache_cap == 0 ||
        raw_start >= g->dspark_cache_cap ||
        token_start > UINT32_MAX - len ||
        raw_start != token_start % g->dspark_cache_cap) {
        return false;
    }
    return true;
}

static bool metal_graph_dspark_cache_current_window_valid(
        const ds4_gpu_graph *g) {
    if (!g) return false;
    return metal_graph_dspark_cache_window_valid(g,
                                                g->dspark_cache_token_start,
                                                g->dspark_cache_start,
                                                g->dspark_cache_len);
}

static bool metal_graph_dspark_cache_set_window(ds4_gpu_graph *g,
                                                uint32_t       token_start,
                                                uint32_t       len) {
    if (!g || len > g->dspark_cache_cap) return false;
    const uint32_t raw_start =
        len && g->dspark_cache_cap ? token_start % g->dspark_cache_cap : 0;
    if (!metal_graph_dspark_cache_window_valid(g,
                                               len ? token_start : 0,
                                               raw_start,
                                               len)) {
        return false;
    }
    g->dspark_cache_start = raw_start;
    g->dspark_cache_token_start = len ? token_start : 0;
    g->dspark_cache_len = len;
    return true;
}

static bool metal_graph_dspark_cache_crop_to_prefix(ds4_gpu_graph *g,
                                                    uint32_t       prefix_len) {
    if (!g) return false;
    if (g->dspark_cache_len == 0) return true;
    if (!metal_graph_dspark_cache_current_window_valid(g)) return false;

    const uint32_t start = g->dspark_cache_token_start;
    const uint32_t end = start + g->dspark_cache_len;
    if (prefix_len <= start || prefix_len > end) {
        metal_graph_dspark_cache_reset(g);
        return true;
    }
    g->dspark_cache_len = prefix_len - start;
    return true;
}

static bool metal_graph_dspark_cache_ends_at(const ds4_gpu_graph *g,
                                             uint32_t             pos) {
    if (!metal_graph_dspark_cache_current_window_valid(g)) return false;
    if (g->dspark_cache_len == 0) return true;
    return g->dspark_cache_token_start <= UINT32_MAX - g->dspark_cache_len &&
           g->dspark_cache_token_start + g->dspark_cache_len == pos;
}

static bool metal_graph_dspark_cache_claim_appended_row(ds4_gpu_graph *g,
                                                        uint32_t pos) {
    if (!g || g->dspark_cache_len == 0 ||
        !metal_graph_dspark_cache_ends_at(g, pos)) return false;
    g->dspark_cache_len += 1u;
    if (g->dspark_cache_len > g->dspark_cache_cap) {
        const uint32_t excess = g->dspark_cache_len - g->dspark_cache_cap;
        g->dspark_cache_token_start += excess;
        g->dspark_cache_len = g->dspark_cache_cap;
        g->dspark_cache_start =
            g->dspark_cache_token_start % g->dspark_cache_cap;
    }
    return true;
}

bool ds4_test_dspark_cache_window_crop(void) {
    ds4_gpu_graph g;
    memset(&g, 0, sizeof(g));
    g.dspark_cache_cap = 8;

    if (!metal_graph_dspark_cache_set_window(&g, 10, 5)) return false;
    if (g.dspark_cache_token_start != 10 ||
        g.dspark_cache_start != 2 ||
        g.dspark_cache_len != 5) return false;
    if (!metal_graph_dspark_cache_ends_at(&g, 15)) return false;
    if (metal_graph_dspark_cache_ends_at(&g, 14)) return false;
    if (metal_graph_dspark_cache_window_valid(&g, 10, 3, 5)) return false;

    if (!metal_graph_dspark_cache_crop_to_prefix(&g, 13)) return false;
    if (g.dspark_cache_token_start != 10 ||
        g.dspark_cache_start != 2 ||
        g.dspark_cache_len != 3) return false;
    if (!metal_graph_dspark_cache_ends_at(&g, 13)) return false;
    if (!metal_graph_dspark_cache_claim_appended_row(&g, 13)) return false;
    if (g.dspark_cache_token_start != 10 ||
        g.dspark_cache_start != 2 ||
        g.dspark_cache_len != 4) return false;
    if (!metal_graph_dspark_cache_ends_at(&g, 14)) return false;
    if (metal_graph_dspark_cache_claim_appended_row(&g, 13)) return false;

    if (!metal_graph_dspark_cache_crop_to_prefix(&g, 20)) return false;
    if (g.dspark_cache_token_start != 0 ||
        g.dspark_cache_start != 0 ||
        g.dspark_cache_len != 0) return false;
    if (!metal_graph_dspark_cache_ends_at(&g, 20)) return false;

    if (metal_graph_dspark_cache_set_window(&g, UINT32_MAX - 1u, 2)) {
        return false;
    }
    return true;
}

static void metal_graph_dspark_capture_begin(ds4_gpu_graph *g) {
    metal_graph_dspark_capture_row_invalidate(g);
}

static void metal_graph_dspark_capture_begin_prefill(ds4_gpu_graph *g) {
    metal_graph_dspark_capture_invalidate(g);
}

static bool metal_graph_dspark_capture_hc(
        ds4_gpu_graph       *g,
        const ds4_gpu_tensor *hc,
        uint32_t             slot) {
    if (!g || !hc || !g->dspark_target_hidden ||
        !g->dspark_hc_mean_weights ||
        slot >= g->dspark_target_layer_count) {
        return false;
    }

    ds4_gpu_tensor *dst =
        ds4_gpu_tensor_view(g->dspark_target_hidden,
                            (uint64_t)slot * DS4_N_EMBD * sizeof(float),
                            (uint64_t)DS4_N_EMBD * sizeof(float));
    if (!dst) return false;
    const bool ok = ds4_gpu_hc_weighted_sum_tensor(dst,
                                                   hc,
                                                   g->dspark_hc_mean_weights,
                                                   DS4_N_EMBD,
                                                   DS4_N_HC) != 0;
    ds4_gpu_tensor_free(dst);
    if (ok) metal_graph_dspark_capture_note_slot(g, slot);
    return ok;
}

static bool metal_graph_dspark_capture_batch_note_slot(
        ds4_gpu_graph *g,
        uint32_t       slot,
        uint32_t       start,
        uint32_t       n_tokens) {
    if (!g || slot >= g->dspark_target_layer_count || n_tokens == 0) {
        return false;
    }
    if (g->dspark_capture_batch_mask == 0) {
        g->dspark_capture_batch_start = start;
        g->dspark_capture_batch_tokens = n_tokens;
    } else if (g->dspark_capture_batch_start != start ||
               g->dspark_capture_batch_tokens != n_tokens) {
        metal_graph_dspark_capture_batch_invalidate(g);
        return false;
    }
    g->dspark_capture_batch_mask |= 1u << slot;
    g->dspark_capture_batch_valid =
        g->dspark_capture_batch_mask ==
        metal_graph_dspark_capture_complete_mask(g);
    return true;
}

static bool metal_graph_dspark_capture_decode_layer(
        ds4_gpu_graph *g,
        uint32_t       il) {
    const int slot = metal_graph_dspark_target_slot(g, il);
    if (slot < 0) return true;
    return metal_graph_dspark_capture_hc(g, metal_graph_cur_hc(g), (uint32_t)slot);
}

static bool metal_graph_dspark_capture_prefill_layer(
        ds4_gpu_graph *g,
        uint32_t       il,
        uint32_t       start,
        uint32_t       n_tokens) {
    const int slot = metal_graph_dspark_target_slot(g, il);
    if (slot < 0) return true;
    if (n_tokens == 0) return false;

    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const uint64_t embd_bytes = (uint64_t)DS4_N_EMBD * sizeof(float);
    if (g->dspark_target_hidden_batch &&
        g->dspark_hc_mean_rows &&
        n_tokens <= g->prefill_cap) {
        ds4_gpu_tensor *batch_dst =
            ds4_gpu_tensor_view(g->dspark_target_hidden_batch,
                                ((uint64_t)slot * g->prefill_cap *
                                 DS4_N_EMBD) * sizeof(float),
                                (uint64_t)n_tokens * embd_bytes);
        ds4_gpu_tensor *last_src =
            batch_dst ?
                ds4_gpu_tensor_view(batch_dst,
                                    (uint64_t)(n_tokens - 1u) * embd_bytes,
                                    embd_bytes) : NULL;
        ds4_gpu_tensor *last_dst =
            ds4_gpu_tensor_view(g->dspark_target_hidden,
                                (uint64_t)slot * embd_bytes,
                                embd_bytes);
        bool ok = batch_dst && last_src && last_dst &&
                  ds4_gpu_hc_weighted_sum_tensor(batch_dst,
                                                 metal_graph_batch_cur_hc(g),
                                                 g->dspark_hc_mean_rows,
                                                 DS4_N_EMBD,
                                                 DS4_N_HC) != 0 &&
                  ds4_gpu_tensor_copy(last_dst,
                                      0,
                                      last_src,
                                      0,
                                      embd_bytes) != 0;
        ds4_gpu_tensor_free(last_dst);
        ds4_gpu_tensor_free(last_src);
        ds4_gpu_tensor_free(batch_dst);
        if (ok) {
            metal_graph_dspark_capture_note_slot(g, (uint32_t)slot);
            ok = metal_graph_dspark_capture_batch_note_slot(g,
                                                            (uint32_t)slot,
                                                            start,
                                                            n_tokens);
        }
        return ok;
    }

    ds4_gpu_tensor *last_hc =
        ds4_gpu_tensor_view(metal_graph_batch_cur_hc(g),
                            (uint64_t)(n_tokens - 1u) * hc_dim * sizeof(float),
                            hc_dim * sizeof(float));
    if (!last_hc) return false;
    const bool ok = metal_graph_dspark_capture_hc(g, last_hc, (uint32_t)slot);
    ds4_gpu_tensor_free(last_hc);
    return ok;
}

static bool metal_graph_dspark_capture_prefill_rows(
        ds4_gpu_graph *g,
        uint32_t       il,
        uint32_t       chunk_start,
        uint32_t       chunk_len,
        uint32_t       pos0,
        uint32_t       n_tokens) {
    const int slot = metal_graph_dspark_target_slot(g, il);
    if (slot < 0) return true;
    if (!g->dspark_target_hidden_batch ||
        !g->dspark_target_hidden ||
        !g->dspark_hc_mean_rows ||
        n_tokens == 0 ||
        chunk_len == 0 ||
        chunk_len > g->prefill_cap ||
        pos0 < chunk_start) {
        return true;
    }
    const uint32_t row0 = pos0 - chunk_start;
    if (row0 > chunk_len || n_tokens > chunk_len - row0) return true;

    const uint64_t embd_bytes = (uint64_t)DS4_N_EMBD * sizeof(float);
    ds4_gpu_tensor *batch_dst =
        ds4_gpu_tensor_view(g->dspark_target_hidden_batch,
                            (((uint64_t)(uint32_t)slot * g->prefill_cap +
                              row0) * DS4_N_EMBD) * sizeof(float),
                            (uint64_t)n_tokens * embd_bytes);
    bool ok = batch_dst &&
              ds4_gpu_hc_weighted_sum_tensor(batch_dst,
                                             metal_graph_batch_cur_hc(g),
                                             g->dspark_hc_mean_rows,
                                             DS4_N_EMBD,
                                             DS4_N_HC) != 0;
    if (!ok) fprintf(stderr, "ds4: pipeline capture rows FAIL il=%u row0=%u n=%u dst=%d\n",
                     il, row0, n_tokens, batch_dst != NULL);
    if (ok && row0 + n_tokens == chunk_len) {
        ds4_gpu_tensor *last_src =
            ds4_gpu_tensor_view(batch_dst,
                                (uint64_t)(n_tokens - 1u) * embd_bytes,
                                embd_bytes);
        ds4_gpu_tensor *last_dst =
            ds4_gpu_tensor_view(g->dspark_target_hidden,
                                (uint64_t)(uint32_t)slot * embd_bytes,
                                embd_bytes);
        ok = last_src && last_dst &&
             ds4_gpu_tensor_copy(last_dst, 0, last_src, 0, embd_bytes) != 0;
        ds4_gpu_tensor_free(last_dst);
        ds4_gpu_tensor_free(last_src);
        if (ok) {
            metal_graph_dspark_capture_note_slot(g, (uint32_t)slot);
            ok = metal_graph_dspark_capture_batch_note_slot(g,
                                                            (uint32_t)slot,
                                                            chunk_start,
                                                            chunk_len);
        }
    }
    ds4_gpu_tensor_free(batch_dst);
    return ok;
}

static bool metal_graph_dspark_capture_verified_suffix_begin(
        ds4_gpu_graph *g,
        uint32_t       start,
        uint32_t       n_tokens,
        bool           commands_open) {
    if (!g || !g->dspark_capture_enabled ||
        !g->dspark_target_hidden ||
        !g->dspark_target_hidden_batch ||
        start == 0 ||
        n_tokens == 0 ||
        n_tokens + 1u < n_tokens ||
        n_tokens + 1u > g->prefill_cap ||
        !g->dspark_capture_valid ||
        g->dspark_capture_checkpoint_len != start) {
        metal_graph_dspark_capture_invalidate(g);
        return false;
    }

    const uint64_t embd_bytes = (uint64_t)DS4_N_EMBD * sizeof(float);
    metal_graph_dspark_capture_batch_invalidate(g);
    bool ok = commands_open || ds4_gpu_begin_commands() != 0;
    for (uint32_t slot = 0; ok && slot < g->dspark_target_layer_count; slot++) {
        ds4_gpu_tensor *dst =
            ds4_gpu_tensor_view(g->dspark_target_hidden_batch,
                                ((uint64_t)slot * g->prefill_cap *
                                 DS4_N_EMBD) * sizeof(float),
                                embd_bytes);
        ds4_gpu_tensor *src =
            ds4_gpu_tensor_view(g->dspark_target_hidden,
                                (uint64_t)slot * embd_bytes,
                                embd_bytes);
        ok = dst && src &&
             ds4_gpu_tensor_copy(dst, 0, src, 0, embd_bytes) != 0;
        ds4_gpu_tensor_free(src);
        ds4_gpu_tensor_free(dst);
    }
    if (ok && !commands_open) ok = ds4_gpu_end_commands() != 0;
    else if (!ok && !commands_open) (void)ds4_gpu_synchronize();
    if (!ok) {
        metal_graph_dspark_capture_invalidate(g);
        return false;
    }

    metal_graph_dspark_capture_row_invalidate(g);
    return true;
}

static bool metal_graph_dspark_capture_verified_suffix_layer(
        ds4_gpu_graph *g,
        uint32_t       il,
        uint32_t       start,
        uint32_t       n_tokens) {
    const int slot = metal_graph_dspark_target_slot(g, il);
    if (slot < 0) return true;
    if (!g || !g->dspark_target_hidden_batch ||
        !g->dspark_target_hidden ||
        !g->dspark_hc_mean_rows ||
        start == 0 ||
        n_tokens == 0 ||
        n_tokens + 1u < n_tokens ||
        n_tokens + 1u > g->prefill_cap) {
        return false;
    }

    const uint64_t embd_bytes = (uint64_t)DS4_N_EMBD * sizeof(float);
    ds4_gpu_tensor *batch_dst =
        ds4_gpu_tensor_view(g->dspark_target_hidden_batch,
                            (((uint64_t)(uint32_t)slot * g->prefill_cap +
                              1u) * DS4_N_EMBD) * sizeof(float),
                            (uint64_t)n_tokens * embd_bytes);
    ds4_gpu_tensor *last_src =
        batch_dst ?
            ds4_gpu_tensor_view(batch_dst,
                                (uint64_t)(n_tokens - 1u) * embd_bytes,
                                embd_bytes) : NULL;
    ds4_gpu_tensor *last_dst =
        ds4_gpu_tensor_view(g->dspark_target_hidden,
                            (uint64_t)(uint32_t)slot * embd_bytes,
                            embd_bytes);
    bool ok = batch_dst && last_src && last_dst &&
              ds4_gpu_hc_weighted_sum_tensor(batch_dst,
                                             metal_graph_batch_cur_hc(g),
                                             g->dspark_hc_mean_rows,
                                             DS4_N_EMBD,
                                             DS4_N_HC) != 0 &&
              ds4_gpu_tensor_copy(last_dst,
                                  0,
                                  last_src,
                                  0,
                                  embd_bytes) != 0;
    ds4_gpu_tensor_free(last_dst);
    ds4_gpu_tensor_free(last_src);
    ds4_gpu_tensor_free(batch_dst);
    if (ok) {
        metal_graph_dspark_capture_note_slot(g, (uint32_t)slot);
        ok = metal_graph_dspark_capture_batch_note_slot(g,
                                                        (uint32_t)slot,
                                                        start - 1u,
                                                        n_tokens + 1u);
    }
    return ok;
}

/* Encode a full single-token decode step on Metal.  This is the generation
 * hot path: update caches, run all layers, then produce logits. */
static bool metal_graph_encode_token_raw_swa(
        ds4_gpu_graph *g,
        const ds4_model       *model,
        const ds4_weights     *weights,
        int                    token,
        uint32_t               pos,
        bool                   need_logits,
        bool                   allow_split_flush) {
    if (g->raw_cap == 0) {
        fprintf(stderr, "ds4: Metal graph raw KV cache is not allocated\n");
        return false;
    }
    /* Under the vocab split both ranks materialize their logits half. */
    if (g->tp_world == 2 && g->tp_rank == 1 &&
        !g->tp_logits_half) need_logits = false;
    const uint32_t raw_row = pos % g->raw_cap;
    const uint32_t n_raw = metal_graph_raw_span_for_batch(g, pos, 1);
    metal_graph_dspark_capture_begin(g);

    /* write the embedded token on the embedding tier. Single-
     * tier: emb_tier == 0 == active_tier; no-op. Multi-tier: switch to
     * emb_tier (no cross-device copy needed — embed writes from scratch). */
    if (g->placement) {
        if (!metal_graph_set_active_tier_decode(g, g->emb_tier)) return false;
    }
    bool ok = ds4_gpu_embed_token_hc_tensor(metal_graph_cur_hc(g),
                                              model->map,
                                              model->size,
                                              weights->token_embd->abs_offset,
                                              (uint32_t)weights->token_embd->dim[1],
                                              (uint32_t)token,
                                              DS4_N_EMBD,
                                              DS4_N_HC) != 0;

    /*
     * Start executing the prefix of the decode graph while the CPU is still
     * encoding the rest. The split point is layer-based because this executor is
     * a fixed DS4 tape, not a dynamic node graph; four layers is the measured
     * point where the prefix is large enough to hide useful work without
     * starving the second command buffer.
     */
    const uint32_t split_after_layers = metal_graph_token_split_after_layers();

    for (uint32_t il = 0; ok && il < DS4_N_LAYER; il++) {
        ok = metal_graph_encode_decode_layer(g,
                                             model,
                                             &weights->layer[il],
                                             il,
                                             pos,
                                             g->layer_raw_cache[il],
                                             g->raw_cap,
                                             raw_row,
                                             n_raw,
                                             token);
        ds4_gpu_tensor *tmp = metal_graph_cur_hc(g);
        g->cur_hc_by_tier[g->active_tier] = metal_graph_after_ffn_hc(g);
        g->after_ffn_hc_by_tier[g->active_tier] = tmp;
        if (ok) ok = metal_graph_dspark_capture_decode_layer(g, il);
        /* A TP gate uses one monotonic shared event for the whole token. A
         * later command buffer may signal a higher value while the prefix is
         * blocked at an earlier gate, making the transport consume a slab
         * slot before its payload is ready. Keep each TP token in one command
         * buffer; non-TP decode retains the encode/execute overlap. */
        if (ok && allow_split_flush && g->tp_world != 2 &&
            split_after_layers != 0 && il + 1u == split_after_layers) {
            ok = ds4_gpu_flush_commands() != 0;
        }
    }

    if (ok && need_logits) {
        ok = metal_graph_encode_output_head(g, model, weights, weights->output->dim[1]);
    }
    return ok;
}

static ds4_gpu_tensor *metal_graph_tensor_row_view(
        ds4_gpu_tensor *base,
        uint32_t          row,
        uint64_t          row_values) {
    return ds4_gpu_tensor_view(base,
                                 (uint64_t)row * row_values * sizeof(float),
                                 row_values * sizeof(float));
}

/* Upload prompt token ids for kernels that need token-aware hash routing. */
static bool metal_graph_upload_prompt_tokens(
        ds4_gpu_tensor *out_tokens,
        const token_vec  *prompt,
        uint32_t          pos0,
        uint32_t          n_tokens) {
    if (!out_tokens || pos0 > (uint32_t)prompt->len || n_tokens > (uint32_t)prompt->len - pos0) {
        return false;
    }

    int32_t *tokens = xmalloc((size_t)n_tokens * sizeof(tokens[0]));
    for (uint32_t i = 0; i < n_tokens; i++) tokens[i] = prompt->v[pos0 + i];

    const bool ok = ds4_gpu_tensor_write(out_tokens,
                                           0,
                                           tokens,
                                           (uint64_t)n_tokens * sizeof(tokens[0])) != 0;
    free(tokens);
    return ok;
}

/* Rebuild ratio-4 compressor state after chunked prefill so a following decode
 * token sees the same rolling compression window. */
static bool metal_graph_refresh_ratio4_compressor_state(
        ds4_gpu_graph  *g,
        const ds4_model  *model,
        ds4_gpu_tensor *state_kv,
        ds4_gpu_tensor *state_score,
        const ds4_tensor *kv_weight,
        const ds4_tensor *score_weight,
        const ds4_tensor *ape,
        uint32_t          head_dim,
        uint32_t          width,
        uint32_t          pos0,
        uint32_t          n_tokens) {
    if (n_tokens < 4) {
        return true;
    }
    if (!g || !model || !state_kv || !state_score || !kv_weight || !score_weight || !ape ||
        head_dim == 0 || width == 0) {
        return false;
    }

    /*
     * The recurrent ratio-4 state is intentionally rebuilt from the last
     * four tokens using the small-batch projection kernel. The full-chunk
     * projection is already available, but it uses the matrix-matrix path;
     * mixing those two accumulation orders changes a few FP8 rounding
     * decisions in later chunks.
     */
    ds4_gpu_tensor *tail_hc = ds4_gpu_tensor_view(
            metal_graph_batch_attn_norm(g),
            (uint64_t)(n_tokens - 4u) * DS4_N_EMBD * sizeof(float),
            4ull * DS4_N_EMBD * sizeof(float));
    bool ok = tail_hc != NULL;
    if (!ok) {
        fprintf(stderr, "ds4: ratio-4 compressor tail view creation failed\n");
    }
    if (ok) {
#if defined(__APPLE__)
        ok = ds4_gpu_matmul_f16_tensor(metal_graph_batch_comp_kv(g),
                                        model->map,
                                        model->size,
                                        kv_weight->abs_offset,
                                        DS4_N_EMBD,
                                        width,
                                        tail_hc,
                                        4) != 0;
        if (ok) {
            ok = ds4_gpu_matmul_f16_tensor(metal_graph_batch_comp_sc(g),
                                            model->map,
                                            model->size,
                                            score_weight->abs_offset,
                                            DS4_N_EMBD,
                                            width,
                                            tail_hc,
                                            4) != 0;
        }
#else
        ok = ds4_gpu_matmul_f16_pair_tensor(metal_graph_batch_comp_kv(g),
                                             metal_graph_batch_comp_sc(g),
                                             model->map,
                                             model->size,
                                             kv_weight->abs_offset,
                                             score_weight->abs_offset,
                                             DS4_N_EMBD,
                                             width,
                                             tail_hc,
                                             4) != 0;
#endif
        if (!ok) {
            fprintf(stderr, "ds4: ratio-4 compressor tail projection failed\n");
        }
    }
    if (ok) {
        ok = ds4_gpu_compressor_prefill_state_ratio4_tensor(state_kv,
                                                              state_score,
                                                              metal_graph_batch_comp_kv(g),
                                                              metal_graph_batch_comp_sc(g),
                                                              model->map,
                                                              model->size,
                                                              ape->abs_offset,
                                                              ape->type,
                                                              head_dim,
                                                              pos0 + n_tokens - 4u) != 0;
        if (!ok) {
            fprintf(stderr, "ds4: ratio-4 compressor state refresh failed\n");
        }
    }
    ds4_gpu_tensor_free(tail_hc);
    return ok;
}

/* CPU fallback for seeding batched HC state from token embeddings.  It is still
 * useful for tiny speculative verifier batches where a separate GPU embedding
 * command buffer costs more than the small host write. */
static bool metal_graph_upload_prompt_embeddings_hc_cpu(
        ds4_gpu_tensor   *out_hc,
        const ds4_model    *model,
        const ds4_weights  *weights,
        const token_vec    *prompt,
        uint32_t            pos0,
        uint32_t            n_tokens) {
    if (pos0 > (uint32_t)prompt->len || n_tokens > (uint32_t)prompt->len - pos0) return false;
    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const uint64_t total = (uint64_t)n_tokens * hc_dim;
    float *hc = xmalloc((size_t)total * sizeof(hc[0]));
    float *plain = xmalloc((size_t)DS4_N_EMBD * sizeof(plain[0]));

    for (uint32_t t = 0; t < n_tokens; t++) {
        embed_token_f16(model, weights, prompt->v[pos0 + t], plain);
        float *dst = hc + (uint64_t)t * hc_dim;
        for (uint32_t h = 0; h < DS4_N_HC; h++) {
            memcpy(dst + (uint64_t)h * DS4_N_EMBD,
                   plain,
                   (size_t)DS4_N_EMBD * sizeof(plain[0]));
        }
    }

    const bool ok = ds4_gpu_tensor_write(out_hc, 0, hc, total * sizeof(hc[0])) != 0;
    free(plain);
    free(hc);
    return ok;
}

/* Seed the batched HC state from token ids: every HC stream starts as the same
 * 4096-wide embedding.  Long prefill chunks use the Metal get-rows/repeat
 * kernel so the CPU does not build and upload a large [token, HC, dim] tensor. */
static bool metal_graph_upload_prompt_embeddings_hc(
        ds4_gpu_tensor   *out_hc,
        ds4_gpu_tensor   *tokens,
        const ds4_model    *model,
        const ds4_weights  *weights,
        const token_vec    *prompt,
        uint32_t            pos0,
        uint32_t            n_tokens) {
    if (pos0 > (uint32_t)prompt->len || n_tokens > (uint32_t)prompt->len - pos0) return false;

    uint32_t gpu_min = 512;
#ifndef DS4_ROCM_BUILD
    const char *gpu_min_env = getenv("DS4_METAL_GPU_BATCH_EMBED_MIN");
    if (gpu_min_env && gpu_min_env[0]) {
        char *end = NULL;
        unsigned long v = strtoul(gpu_min_env, &end, 10);
        if (end != gpu_min_env && v <= UINT32_MAX) gpu_min = (uint32_t)v;
    }
#endif

    if (tokens && n_tokens >= gpu_min) {
        return ds4_gpu_embed_tokens_hc_tensor(out_hc,
                                                tokens,
                                                model->map,
                                                model->size,
                                                weights->token_embd->abs_offset,
                                                (uint32_t)weights->token_embd->dim[1],
                                                n_tokens,
                                                DS4_N_EMBD,
                                                DS4_N_HC) != 0;
    }

    return metal_graph_upload_prompt_embeddings_hc_cpu(out_hc,
                                                       model,
                                                       weights,
                                                       prompt,
                                                       pos0,
                                                       n_tokens);
}

static bool metal_graph_hc_rms_scale_project(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *norm_scratch,
        const ds4_model        *model,
        const ds4_tensor       *weight,
        const ds4_gpu_tensor *x,
        uint64_t                in_dim,
        uint32_t                n_tokens) {
    if (!out || !norm_scratch || !model || !weight || !x ||
        in_dim > UINT32_MAX) {
        return false;
    }
#if defined(__APPLE__)
    return ds4_gpu_hc_rms_scale_project_f16_tensor(
               out,
               norm_scratch,
               model->map,
               model->size,
               weight->abs_offset,
               (uint32_t)in_dim,
               2u * DS4_N_HC + DS4_N_HC * DS4_N_HC,
               x,
               n_tokens,
               DS4_RMS_EPS) != 0;
#else
    bool ok = ds4_gpu_rms_norm_plain_rows_tensor(
                  norm_scratch,
                  x,
                  (uint32_t)in_dim,
                  n_tokens,
                  DS4_RMS_EPS) != 0;
    if (ok) {
        ok = ds4_gpu_matmul_f16_tensor(
                 out,
                 model->map,
                 model->size,
                 weight->abs_offset,
                 in_dim,
                 2u * DS4_N_HC + DS4_N_HC * DS4_N_HC,
                 norm_scratch,
                 n_tokens) != 0;
    }
    return ok;
#endif
}

static bool metal_graph_warmup_prefill_kernels(
        ds4_gpu_graph   *g,
        const ds4_model   *model,
        const ds4_weights *weights,
        uint32_t           n_tokens) {
    static bool warmed = false;
    if (g && g->ssd_streaming) return true;
    if (warmed) return true;
#ifndef DS4_ROCM_BUILD
    if (getenv("DS4_METAL_NO_PREFILL_KERNEL_WARMUP") != NULL) return true;
#endif

    /*
     * The first batched F16 matmul can pay Metal's one-time pipeline execution
     * cost. Run the same HC attention projection on scratch storage before the
     * measured prefill. The output is overwritten by the real graph.
     */
    if (n_tokens <= 8) return true;

    /* (B6 fix, ): warm-up uses layer-0's hc_attn_fn
     * weight, which in multi-tier is resolved on placement[1]'s tier.
     * Switch active_tier so the F16 matmul reads/writes the correct
     * Class P scratch and resolves the weight on the right device.
     * Single-tier (g->placement == NULL): no-op. */
    if (g->placement) {
        if (!metal_graph_set_active_tier_batch(g, g->placement[1], n_tokens)) return false;
    }
    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;

    bool ok = ds4_gpu_begin_commands() != 0;
    if (ok) {
        ok = metal_graph_hc_rms_scale_project(
                 metal_graph_batch_hc_mix(g),
                 metal_graph_batch_flat_hc(g),
                 model,
                 weights->layer[0].hc_attn_fn,
                 metal_graph_batch_cur_hc(g),
                 hc_dim,
                 n_tokens);
    }
    if (ok) ok = ds4_gpu_end_commands() != 0;
    if (!ok) {
        fprintf(stderr, "ds4: Metal prefill kernel warmup failed\n");
        return false;
    }

    warmed = true;
    return true;
}

/* Encode the batched prefill attention half for one layer.  It mirrors the CPU
 * layer-major path: HC pre/norm, Q/KV, cache/compression, prefix attention. */
static bool metal_graph_indexer_stage_profile_boundary(
        const char *stage,
        uint32_t    il,
        uint32_t    pos0,
        uint32_t    n_tokens,
        uint32_t    n_comp,
        double     *stage_t0) {
    if (ds4_gpu_end_commands() == 0) return false;
    const double now = now_sec();
    if (stage != NULL) {
        fprintf(stderr,
                "ds4: metal indexer stage layer=%u pos=%u tokens=%u comp=%u %s=%.3f ms\n",
                il,
                pos0,
                n_tokens,
                n_comp,
                stage,
                (now - *stage_t0) * 1000.0);
    }
    *stage_t0 = now;
    return ds4_gpu_begin_commands() != 0;
}

static bool metal_graph_env_value_eq(const char *v,
                                     size_t      n,
                                     const char *literal) {
    const size_t m = strlen(literal);
    if (n != m) return false;
    for (size_t i = 0; i < n; i++) {
        if (tolower((unsigned char)v[i]) !=
            tolower((unsigned char)literal[i])) {
            return false;
        }
    }
    return true;
}

static const char *metal_graph_env_trim(const char *v, size_t *len_out) {
    if (!v) {
        if (len_out) *len_out = 0;
        return NULL;
    }
    while (isspace((unsigned char)*v)) v++;
    size_t n = strlen(v);
    while (n > 0 && isspace((unsigned char)v[n - 1])) n--;
    if (len_out) *len_out = n;
    return v;
}

static bool metal_graph_profile_layer_value_match(const char *layer_env,
                                                  uint32_t    il) {
    size_t n = 0;
    layer_env = metal_graph_env_trim(layer_env, &n);
    if (!layer_env || n == 0) return true;

    char *end = NULL;
    const unsigned long layer = strtoul(layer_env, &end, 10);
    return end != layer_env &&
           (size_t)(end - layer_env) == n &&
           layer <= UINT32_MAX &&
           (uint32_t)layer == il;
}

static bool metal_graph_stage_profile_enabled_for_layer(
        const char *flag_env_name,
        const char *layer_env_name,
        uint32_t    il) {
    size_t flag_len = 0;
    const char *flag = metal_graph_env_trim(getenv(flag_env_name), &flag_len);
    if (!flag) return false;

    const char *layer_env = getenv(layer_env_name);
    const bool has_layer_filter = layer_env && layer_env[0];

    if (flag_len != 0) {
        if (metal_graph_env_value_eq(flag, flag_len, "0") ||
            metal_graph_env_value_eq(flag, flag_len, "false") ||
            metal_graph_env_value_eq(flag, flag_len, "no") ||
            metal_graph_env_value_eq(flag, flag_len, "off")) {
            return false;
        }
        if (!has_layer_filter &&
            !metal_graph_env_value_eq(flag, flag_len, "1") &&
            !metal_graph_env_value_eq(flag, flag_len, "true") &&
            !metal_graph_env_value_eq(flag, flag_len, "yes") &&
            !metal_graph_env_value_eq(flag, flag_len, "on") &&
            !metal_graph_env_value_eq(flag, flag_len, "all")) {
            return metal_graph_profile_layer_value_match(flag, il);
        }
    }

    return metal_graph_profile_layer_value_match(layer_env, il);
}

static bool metal_graph_layer_stage_profile_enabled(uint32_t il) {
    return metal_graph_stage_profile_enabled_for_layer(
                "DS4_ROCM_LAYER_STAGE_PROFILE",
                "DS4_ROCM_LAYER_STAGE_PROFILE_LAYER",
                il) ||
           metal_graph_stage_profile_enabled_for_layer(
                "DS4_METAL_LAYER_STAGE_PROFILE",
                "DS4_METAL_LAYER_STAGE_PROFILE_LAYER",
                il);
}

static bool metal_graph_decode_stage_profile_enabled(uint32_t il) {
    return metal_graph_stage_profile_enabled_for_layer(
                "DS4_ROCM_DECODE_STAGE_PROFILE",
                "DS4_ROCM_DECODE_STAGE_PROFILE_LAYER",
                il) ||
           metal_graph_stage_profile_enabled_for_layer(
                "DS4_METAL_DECODE_STAGE_PROFILE",
                "DS4_METAL_DECODE_STAGE_PROFILE_LAYER",
                il);
}

static bool metal_graph_layer_stage_profile_start(uint32_t il) {
    if (!metal_graph_layer_stage_profile_enabled(il)) return true;
    if (ds4_gpu_end_commands() == 0) return false;
    return ds4_gpu_begin_commands() != 0;
}

/* Optional prefill stage profiler. It intentionally ends the current Metal
 * command buffer and waits, so the printed number includes encoding plus GPU
 * execution for the stage just emitted. This is disabled by default because it
 * adds synchronization points and changes scheduling. */
static bool metal_graph_layer_stage_profile_boundary(
        const char *part,
        const char *stage,
        uint32_t    il,
        uint32_t    pos0,
        uint32_t    n_tokens,
        double     *stage_t0) {
    if (ds4_gpu_end_commands() == 0) return false;
    const double now = now_sec();
    if (stage != NULL) {
        fprintf(stderr,
                "ds4: metal layer stage part=%s layer=%u pos=%u tokens=%u %s=%.3f ms\n",
                part,
                il,
                pos0,
                n_tokens,
                stage,
                (now - *stage_t0) * 1000.0);
    }
    *stage_t0 = now;
    return ds4_gpu_begin_commands() != 0;
}

static bool metal_graph_q_stage_profile_boundary(
        const char *stage,
        uint32_t    il,
        uint32_t    pos0,
        uint32_t    n_tokens,
        double     *stage_t0) {
    if (ds4_gpu_end_commands() == 0) return false;
    const double now = now_sec();
    fprintf(stderr,
            "ds4: metal Q path stage layer=%u pos=%u tokens=%u %s=%.3f ms\n",
            il,
            pos0,
            n_tokens,
            stage,
            (now - *stage_t0) * 1000.0);
    *stage_t0 = now;
    return ds4_gpu_begin_commands() != 0;
}

static ds4_gpu_tensor *metal_graph_tensor_row_range_view(
        ds4_gpu_tensor *base,
        uint32_t          row0,
        uint32_t          rows,
        uint64_t          row_values) {
    return ds4_gpu_tensor_view(base,
                                 (uint64_t)row0 * row_values * sizeof(float),
                                 (uint64_t)rows * row_values * sizeof(float));
}

/* TP prefill threshold for row-splitting the replicated shared expert.
 * Routed experts remain ownership-split at every batch size. */
static uint32_t metal_graph_tp_prefill_split_min(void) {
    static int cached = -1;
    if (cached < 0) {
        cached = 32;
        const char *env = getenv("DS4_TP_PREFILL_SPLIT_MIN");
        if (env && env[0]) cached = atoi(env);
        if (cached < 2) cached = 2;
    }
    return (uint32_t)cached;
}

/* Opt-in sub-chunk gate pipelining for the TP prefill row swaps.  Must be
 * set on BOTH ranks (it changes the per-layer gate count; asymmetric
 * settings deadlock the big gates).  Default off: measured net-negative
 * on the M5 Max pair, see the pipelined blocks for the numbers. */
static bool metal_graph_tp_subgate_pipeline(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *env = getenv("DS4_TP_SUBGATE_PIPELINE");
        cached = env && env[0] && atoi(env) != 0;
    }
    return cached != 0;
}

static bool metal_graph_encode_layer_attention_batch(
        ds4_gpu_graph  *g,
        const ds4_model        *model,
        const ds4_layer_weights *layer,
        uint32_t                il,
        uint32_t                pos0,
        uint32_t                n_tokens) {
    if (n_tokens == 0 || n_tokens > g->prefill_cap) return false;

    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const uint64_t mix_hc = 2ull * DS4_N_HC + (uint64_t)DS4_N_HC * DS4_N_HC;
    const uint64_t q_rank = layer->attn_q_a->dim[1];
    const uint64_t q_dim = (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM;
    const uint32_t n_groups = DS4_N_OUT_GROUP;
    const uint32_t group_heads = DS4_N_HEAD / n_groups;
    const uint32_t group_dim = DS4_N_HEAD_DIM * group_heads;
    const uint32_t rank = DS4_N_LORA_O;
    const uint32_t ratio = ds4_layer_compress_ratio(il);
    const bool compressed = ratio != 0;
    const bool zero_prefix = pos0 == 0;
    /* TP attention row split for large zero-prefix chunks: q_a and the KV
     * path stay full (both ranks need every row's KV, and the compressor/
     * indexer keep updating their state from full rows), q_b onward runs on
     * this rank's half of the chunk rows, and the computed row halves of
     * batch_attn_out are swapped in place through one big gate per layer.
     * Three chunk shapes split: full-raw (uncompressed layer at pos0 == 0,
     * or a compressed layer whose chunk is too short to emit compressed
     * keys, i.e. raw_prefix_tokens == n_tokens), static-mixed (compressed
     * layer whose whole chunk attends through the one-shot mixed kernel
     * over the full raw keys plus n_tokens / ratio compressed keys, without
     * indexer top-k), and indexed (ratio-4 layer with indexer top-k, whose
     * per-token score/top-k selection stays replicated while the attention
     * consumption splits by rows).  Every condition derives from
     * pos0/n_tokens/ratio/model shape so both ranks stay in lockstep. */
    const bool tp_attn_full_raw = zero_prefix &&
        (ratio == 0 || (n_tokens < ratio && n_tokens <= g->raw_cap));
    const uint32_t tp_attn_n_comp = ratio != 0 ? n_tokens / ratio : 0;
    const bool tp_attn_static_mixed = zero_prefix && ratio != 0 &&
        tp_attn_n_comp != 0 &&
        !(ratio == 4 && tp_attn_n_comp > DS4_N_INDEXER_TOP_K);
    const bool tp_attn_indexed = zero_prefix && ratio == 4 &&
        tp_attn_n_comp > DS4_N_INDEXER_TOP_K;
    const bool tp_row_split_attn =
        g->tp_world == 2 &&
        g->tp_batch_rows != n_tokens &&
        (tp_attn_full_raw || tp_attn_static_mixed || tp_attn_indexed) &&
        !metal_graph_directional_steering_attn_enabled(g) &&
        n_tokens >= metal_graph_tp_prefill_split_min();
    const uint32_t tp_half_rows = (n_tokens + 1u) / 2u;
    const uint32_t tp_row0 = (tp_row_split_attn && g->tp_rank != 0) ? tp_half_rows : 0;
    const uint32_t tp_rows = tp_row_split_attn ?
        (g->tp_rank == 0 ? tp_half_rows : n_tokens - tp_half_rows) : n_tokens;
    const bool index_stage_profile =
        glm_graph_env_present("DS4_ROCM_INDEXER_STAGE_PROFILE",
                              "DS4_METAL_INDEXER_STAGE_PROFILE");
    const bool layer_stage_profile = metal_graph_layer_stage_profile_enabled(il);
    const bool q_stage_profile =
        glm_graph_env_present("DS4_ROCM_Q_STAGE_PROFILE",
                              "DS4_METAL_Q_STAGE_PROFILE");
    double layer_stage_t0 = layer_stage_profile ? now_sec() : 0.0;
    double q_stage_t0 = q_stage_profile ? now_sec() : 0.0;
#define DS4_METAL_PROFILE_ATTN_STAGE(name) do { \
        if (ok && layer_stage_profile) { \
            ok = metal_graph_layer_stage_profile_boundary("attn", (name), il, pos0, n_tokens, &layer_stage_t0); \
        } \
    } while (0)
#define DS4_METAL_PROFILE_Q_STAGE(name) do { \
        if (ok && q_stage_profile) { \
            ok = metal_graph_q_stage_profile_boundary((name), il, pos0, n_tokens, &q_stage_t0); \
        } \
    } while (0)
    const float freq_base = layer_rope_freq_base(il);
    const float freq_scale = layer_rope_freq_scale(il);
    const float ext_factor = compressed && DS4_ROPE_SCALE_FACTOR > 1.0f ? 1.0f : 0.0f;
    float attn_factor = 1.0f;
    if (ext_factor != 0.0f && freq_scale > 0.0f) {
        attn_factor /= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    enum { stack_count_cap = 16 };
    uint32_t comp_counts_stack[stack_count_cap];
    uint32_t index_counts_stack[stack_count_cap];
    uint32_t *comp_counts = NULL;
    uint32_t *index_counts = NULL;
    if (compressed) {
        if (n_tokens <= stack_count_cap) {
            memset(comp_counts_stack, 0,
                   (size_t)n_tokens * sizeof(comp_counts_stack[0]));
            comp_counts = comp_counts_stack;
        } else {
            comp_counts = xcalloc(n_tokens, sizeof(comp_counts[0]));
        }
    }
    if (ratio == 4) {
        if (n_tokens <= stack_count_cap) {
            memset(index_counts_stack, 0,
                   (size_t)n_tokens * sizeof(index_counts_stack[0]));
            index_counts = index_counts_stack;
        } else {
            index_counts = xcalloc(n_tokens, sizeof(index_counts[0]));
        }
    }
    const bool qkv_rms_fused = !metal_graph_use_reference_qkv_norm();
    ds4_gpu_tensor *hc_mix_view = ds4_gpu_tensor_view(
            metal_graph_batch_hc_mix(g), 0, (uint64_t)n_tokens * mix_hc * sizeof(float));
    ds4_gpu_tensor *hc_split_view = ds4_gpu_tensor_view(
            metal_graph_batch_hc_split(g), 0, (uint64_t)n_tokens * mix_hc * sizeof(float));
    ds4_gpu_tensor *attn_cur_view = ds4_gpu_tensor_view(
            metal_graph_batch_attn_cur(g), 0, (uint64_t)n_tokens * DS4_N_EMBD * sizeof(float));
    ds4_gpu_tensor *after_attn_hc_view = ds4_gpu_tensor_view(
            metal_graph_batch_after_attn_hc(g), 0, (uint64_t)n_tokens * hc_dim * sizeof(float));
    bool ok = hc_mix_view && hc_split_view && attn_cur_view && after_attn_hc_view;
    const bool fuse_hc_norm = n_tokens > 1 &&
                              DS4_N_HC == 4 &&
                              !metal_graph_use_reference_hc_decode() &&
                              metal_graph_enable_batch_hc_norm_fusion();
    if (ok) ok = metal_graph_hc_rms_scale_project(hc_mix_view,
                                                    metal_graph_batch_flat_hc(g),
                                                    model,
                                                    layer->hc_attn_fn,
                                                    metal_graph_batch_cur_hc(g),
                                                    hc_dim,
                                                    n_tokens);
    if (metal_graph_use_reference_hc_decode()) {
        if (ok) ok = ds4_gpu_hc_split_sinkhorn_tensor(hc_split_view,
                                                        hc_mix_view,
                                                        model->map,
                                                        model->size,
                                                        layer->hc_attn_scale->abs_offset,
                                                        layer->hc_attn_base->abs_offset,
                                                        DS4_N_HC,
                                                        DS4_N_HC_SINKHORN_ITER,
                                                        DS4_HC_EPS) != 0;
        if (ok) ok = ds4_gpu_hc_weighted_sum_split_tensor(attn_cur_view,
                                                            metal_graph_batch_cur_hc(g),
                                                            hc_split_view,
                                                            DS4_N_EMBD,
                                                            DS4_N_HC) != 0;
    } else if (fuse_hc_norm) {
        if (ok) ok = ds4_gpu_hc_split_weighted_sum_norm_tensor(attn_cur_view,
                                                                 metal_graph_batch_attn_norm(g),
                                                                 hc_split_view,
                                                                 hc_mix_view,
                                                                 metal_graph_batch_cur_hc(g),
                                                                 model->map,
                                                                 model->size,
                                                                 layer->hc_attn_scale->abs_offset,
                                                                 layer->hc_attn_base->abs_offset,
                                                                 layer->attn_norm->abs_offset,
                                                                 DS4_N_EMBD,
                                                                 DS4_N_HC,
                                                                 DS4_N_HC_SINKHORN_ITER,
                                                                 DS4_HC_EPS,
                                                                 DS4_RMS_EPS) != 0;
    } else {
        if (ok) ok = ds4_gpu_hc_split_weighted_sum_tensor(attn_cur_view,
                                                            hc_split_view,
                                                            hc_mix_view,
                                                            metal_graph_batch_cur_hc(g),
                                                            model->map,
                                                            model->size,
                                                            layer->hc_attn_scale->abs_offset,
                                                            layer->hc_attn_base->abs_offset,
                                                            DS4_N_EMBD,
                                                            DS4_N_HC,
                                                            DS4_N_HC_SINKHORN_ITER,
                                                            DS4_HC_EPS) != 0;
    }
    if (ok) {
        metal_graph_debug_dump_tensor("hc_attn_pre", metal_graph_batch_attn_cur(g),
                                      (uint64_t)n_tokens * DS4_N_EMBD, il, pos0);
    }
    DS4_METAL_PROFILE_ATTN_STAGE("hc_pre");
    if (ok && !fuse_hc_norm) {
        ok = ds4_gpu_rms_norm_weight_rows_tensor(metal_graph_batch_attn_norm(g),
                                                  metal_graph_batch_attn_cur(g),
                                                  model->map,
                                                  model->size,
                                                  layer->attn_norm->abs_offset,
                                                  DS4_N_EMBD,
                                                  n_tokens,
                                                  DS4_RMS_EPS) != 0;
    }
    if (ok) {
        metal_graph_debug_dump_tensor("attn_norm", metal_graph_batch_attn_norm(g),
                                      (uint64_t)n_tokens * DS4_N_EMBD, il, pos0);
    }
    DS4_METAL_PROFILE_ATTN_STAGE("norm");
    DS4_METAL_PROFILE_Q_STAGE("pre_q");
    if (ok) ok = metal_graph_matmul_q8_0_named_tensor("attn_q_a",
                                                      il,
                                                      pos0,
                                                      metal_graph_batch_qr(g),
                                                      model,
                                                      layer->attn_q_a,
                                                      DS4_N_EMBD,
                                                      q_rank,
                                                      metal_graph_batch_attn_norm(g),
                                                      n_tokens);
    if (ok) {
        metal_graph_debug_dump_tensor("q_lora", metal_graph_batch_qr(g),
                                      (uint64_t)n_tokens * q_rank, il, pos0);
    }
    DS4_METAL_PROFILE_Q_STAGE("q_a");
    if (qkv_rms_fused) {
        if (ok) ok = metal_graph_matmul_q8_0_named_tensor("attn_kv",
                                                          il,
                                                          pos0,
                                                          metal_graph_batch_kv_raw(g),
                                                          model,
                                                          layer->attn_kv,
                                                          DS4_N_EMBD,
                                                          DS4_N_HEAD_DIM,
                                                          metal_graph_batch_attn_norm(g),
                                                          n_tokens);
        if (ok) {
            metal_graph_debug_dump_tensor("KVraw", metal_graph_batch_kv_raw(g),
                                          (uint64_t)n_tokens * DS4_N_HEAD_DIM, il, pos0);
        }
        if (ok) ok = ds4_gpu_dsv4_qkv_rms_norm_rows_tensor(metal_graph_batch_qr_norm(g),
                                                             metal_graph_batch_qr(g),
                                                             model->map,
                                                             model->size,
                                                             layer->attn_q_a_norm->abs_offset,
                                                             (uint32_t)q_rank,
                                                             metal_graph_batch_kv(g),
                                                             metal_graph_batch_kv_raw(g),
                                                             layer->attn_kv_a_norm->abs_offset,
                                                             DS4_N_HEAD_DIM,
                                                             n_tokens,
                                                             DS4_RMS_EPS) != 0;
    } else {
        if (ok) ok = ds4_gpu_rms_norm_weight_rows_tensor(metal_graph_batch_qr_norm(g),
                                                           metal_graph_batch_qr(g),
                                                           model->map,
                                                           model->size,
                                                           layer->attn_q_a_norm->abs_offset,
                                                           (uint32_t)q_rank,
                                                           n_tokens,
                                                           DS4_RMS_EPS) != 0;
    }
    if (ok) {
        metal_graph_debug_dump_tensor("q_lora_norm", metal_graph_batch_qr_norm(g),
                                      (uint64_t)n_tokens * q_rank, il, pos0);
    }
    if (qkv_rms_fused && ok) {
        metal_graph_debug_dump_tensor("KVnorm", metal_graph_batch_kv(g),
                                      (uint64_t)n_tokens * DS4_N_HEAD_DIM, il, pos0);
    }
    DS4_METAL_PROFILE_Q_STAGE("q_a_norm");
    const bool q_path_debug =
        metal_graph_debug_wants("Qraw", il, pos0) ||
        metal_graph_debug_wants("Qnorm", il, pos0);
    /* Under the TP row split everything from q_b to the output projection
     * runs on this rank's rows only, through row-range views of the batch
     * tensors (batch_q_half is F16, so its view is built directly). */
    ds4_gpu_tensor *tp_q = tp_row_split_attn ?
        metal_graph_tensor_row_range_view(metal_graph_batch_q(g), tp_row0, tp_rows, q_dim) : NULL;
    ds4_gpu_tensor *tp_q_half = tp_row_split_attn ?
        ds4_gpu_tensor_view(g->batch_q_half,
                            (uint64_t)tp_row0 * q_dim * sizeof(uint16_t),
                            (uint64_t)tp_rows * q_dim * sizeof(uint16_t)) : NULL;
    ds4_gpu_tensor *tp_qr_norm = tp_row_split_attn ?
        metal_graph_tensor_row_range_view(metal_graph_batch_qr_norm(g), tp_row0, tp_rows, q_rank) : NULL;
    ds4_gpu_tensor *tp_heads = tp_row_split_attn ?
        metal_graph_tensor_row_range_view(metal_graph_batch_heads(g), tp_row0, tp_rows, q_dim) : NULL;
    ds4_gpu_tensor *tp_attn_out = tp_row_split_attn ?
        metal_graph_tensor_row_range_view(metal_graph_batch_attn_out(g), tp_row0, tp_rows,
                                          DS4_N_EMBD) : NULL;
    if (tp_row_split_attn &&
        (!tp_q || !tp_q_half || !tp_qr_norm || !tp_heads || !tp_attn_out)) {
        ok = false;
    }
    bool q_b_f16_out = false;
    if (ok && !q_path_debug && layer->attn_q_b->type == DS4_TENSOR_Q8_0) {
        q_b_f16_out = ds4_gpu_attn_q_b_f16_head_rms_rope_tail_tensor(tp_q ? tp_q : metal_graph_batch_q(g),
                                                                     tp_q_half ? tp_q_half : g->batch_q_half,
                                                                     model->map,
                                                                     model->size,
                                                                     layer->attn_q_b->abs_offset,
                                                                     q_rank,
                                                                     q_dim,
                                                                     tp_qr_norm ? tp_qr_norm : metal_graph_batch_qr_norm(g),
                                                                     tp_rows,
                                                                     DS4_N_HEAD,
                                                                     DS4_N_HEAD_DIM,
                                                                     DS4_N_ROT,
                                                                     pos0 + tp_row0,
                                                                     compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                                                                     false,
                                                                     freq_base,
                                                                     freq_scale,
                                                                     ext_factor,
                                                                     attn_factor,
                                                                     DS4_ROPE_YARN_BETA_FAST,
                                                                     DS4_ROPE_YARN_BETA_SLOW,
                                                                     DS4_RMS_EPS) != 0;
    }
    if (q_b_f16_out) {
        DS4_METAL_PROFILE_Q_STAGE("q_b");
        DS4_METAL_PROFILE_Q_STAGE("head_norm");
        if (ok) {
            metal_graph_debug_dump_tensor("Qcur", metal_graph_batch_q(g),
                                          (uint64_t)n_tokens * q_dim, il, pos0);
        }
        DS4_METAL_PROFILE_Q_STAGE("rope");
    } else {
        if (ok) ok = metal_graph_matmul_q8_0_named_tensor("attn_q_b",
                                                          il,
                                                          pos0,
                                                          tp_q ? tp_q : metal_graph_batch_q(g),
                                                          model,
                                                          layer->attn_q_b,
                                                          q_rank,
                                                          q_dim,
                                                          tp_qr_norm ? tp_qr_norm : metal_graph_batch_qr_norm(g),
                                                          tp_rows);
        if (ok) {
            metal_graph_debug_dump_tensor("Qraw", metal_graph_batch_q(g),
                                          (uint64_t)n_tokens * q_dim, il, pos0);
        }
        DS4_METAL_PROFILE_Q_STAGE("q_b");
        if (ok) ok = ds4_gpu_head_rms_norm_tensor(tp_q ? tp_q : metal_graph_batch_q(g),
                                                    tp_rows,
                                                    DS4_N_HEAD,
                                                    DS4_N_HEAD_DIM,
                                                    DS4_RMS_EPS) != 0;
        if (ok) {
            metal_graph_debug_dump_tensor("Qnorm", metal_graph_batch_q(g),
                                          (uint64_t)n_tokens * q_dim, il, pos0);
        }
        DS4_METAL_PROFILE_Q_STAGE("head_norm");
        if (ok) ok = ds4_gpu_rope_tail_tensor(tp_q ? tp_q : metal_graph_batch_q(g),
                                                tp_rows,
                                                DS4_N_HEAD,
                                                DS4_N_HEAD_DIM,
                                                DS4_N_ROT,
                                                pos0 + tp_row0,
                                                compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                                                false,
                                                freq_base,
                                                freq_scale,
                                                ext_factor,
                                                attn_factor,
                                                DS4_ROPE_YARN_BETA_FAST,
                                                DS4_ROPE_YARN_BETA_SLOW) != 0;
        if (ok) {
            metal_graph_debug_dump_tensor("Qcur", metal_graph_batch_q(g),
                                          (uint64_t)n_tokens * q_dim, il, pos0);
        }
        DS4_METAL_PROFILE_Q_STAGE("rope");
    }
    DS4_METAL_PROFILE_ATTN_STAGE("q_path");
    if (!qkv_rms_fused) {
        if (ok) ok = metal_graph_matmul_q8_0_named_tensor("attn_kv",
                                                          il,
                                                          pos0,
                                                          metal_graph_batch_kv_raw(g),
                                                          model,
                                                          layer->attn_kv,
                                                          DS4_N_EMBD,
                                                          DS4_N_HEAD_DIM,
                                                          metal_graph_batch_attn_norm(g),
                                                          n_tokens);
        if (ok) {
            metal_graph_debug_dump_tensor("KVraw", metal_graph_batch_kv_raw(g),
                                          (uint64_t)n_tokens * DS4_N_HEAD_DIM, il, pos0);
        }
        if (ok) ok = ds4_gpu_rms_norm_weight_rows_tensor(metal_graph_batch_kv(g),
                                                           metal_graph_batch_kv_raw(g),
                                                           model->map,
                                                           model->size,
                                                           layer->attn_kv_a_norm->abs_offset,
                                                           DS4_N_HEAD_DIM,
                                                           n_tokens,
                                                           DS4_RMS_EPS) != 0;
        if (ok) {
            metal_graph_debug_dump_tensor("KVnorm", metal_graph_batch_kv(g),
                                          (uint64_t)n_tokens * DS4_N_HEAD_DIM, il, pos0);
        }
    }
    if (ok) ok = ds4_gpu_rope_tail_tensor(metal_graph_batch_kv(g),
                                            n_tokens,
                                            DS4_N_HEAD_KV,
                                            DS4_N_HEAD_DIM,
                                            DS4_N_ROT,
                                            pos0,
                                            compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                                            false,
                                            freq_base,
                                            freq_scale,
                                            ext_factor,
                                            attn_factor,
                                            DS4_ROPE_YARN_BETA_FAST,
                                            DS4_ROPE_YARN_BETA_SLOW) != 0;
    if (ok) {
        metal_graph_debug_dump_tensor("KVrope", metal_graph_batch_kv(g),
                                      (uint64_t)n_tokens * DS4_N_HEAD_DIM, il, pos0);
    }
    if (ok) ok = ds4_gpu_dsv4_fp8_kv_quantize_tensor(metal_graph_batch_kv(g),
                                                       n_tokens,
                                                       DS4_N_HEAD_DIM,
                                                       DS4_N_ROT) != 0;
    if (ok) {
        metal_graph_debug_dump_tensor("KVcur", metal_graph_batch_kv(g),
                                      (uint64_t)n_tokens * DS4_N_HEAD_DIM, il, pos0);
    }
    DS4_METAL_PROFILE_ATTN_STAGE("kv_path");
    /*
     * Static graph order is q, kv, cpy_k(raw SWA), then attention. For a
     * zero-prefix batch it is safe to store the whole batch at once: attention
     * reads the contiguous batch KV, and the ring only has to end with the last
     * SWA rows for later chunks/decode. For nonzero chunks the physical ring is
     * sized to hold the current chunk plus the previous SWA window, while the
     * attention mask still enforces the 128-token logical window.
     */
    if (ok && zero_prefix) ok = ds4_gpu_store_raw_kv_batch_tensor(g->layer_raw_cache[il],
                                                                    metal_graph_batch_kv(g),
                                                                    g->raw_cap,
                                                                    pos0,
                                                                    n_tokens,
                                                                    DS4_N_HEAD_DIM) != 0;
    if (!ok) {
        fprintf(stderr, "ds4: gpu layer %u raw KV batch store failed\n", il);
    }
    const bool raw_batch_attention = zero_prefix && ratio == 0;
    bool batch_attention_done = false;

    if (ok && raw_batch_attention) {
        if (tp_row_split_attn) {
            ok = ds4_gpu_attention_prefill_raw_heads_range_tensor(tp_heads,
                                                                    model->map,
                                                                    model->size,
                                                                    layer->attn_sinks->abs_offset,
                                                                    tp_q,
                                                                    metal_graph_batch_kv(g),
                                                                    tp_row0,
                                                                    tp_rows,
                                                                    n_tokens,
                                                                    g->raw_window,
                                                                    DS4_N_HEAD,
                                                                    DS4_N_HEAD_DIM) != 0;
        } else {
            ok = ds4_gpu_attention_prefill_raw_heads_tensor(metal_graph_batch_heads(g),
                                                              model->map,
                                                              model->size,
                                                              layer->attn_sinks->abs_offset,
                                                              metal_graph_batch_q(g),
                                                              metal_graph_batch_kv(g),
                                                              n_tokens,
                                                              g->raw_window,
                                                              DS4_N_HEAD,
                                                              DS4_N_HEAD_DIM) != 0;
        }
        if (ok) batch_attention_done = true;
    } else if (ok && !zero_prefix && ratio == 0 && n_tokens <= g->raw_cap) {
        /*
         * The ubatch path stores the whole batch in the SWA cache, then runs
         * one batched attention kernel with an absolute-position causal/window
         * mask.  This avoids mixing prefill with the different single-token
         * attention path.
         */
        const uint32_t n_raw = metal_graph_raw_span_for_batch(g, pos0, n_tokens);
        /* Nonzero prompt chunks read the SWA cache as a ring.  FlashAttention
         * receives a linearized window starting at raw_start, not physical row
         * zero; otherwise wrapped chunks silently miss recent raw keys. */
        const uint32_t raw_start = metal_graph_raw_start_for_span(g,
                                                                  pos0 + n_tokens - 1u,
                                                                  n_raw);
        ok = ds4_gpu_store_raw_kv_batch_tensor(g->layer_raw_cache[il],
                                                 metal_graph_batch_kv(g),
                                                 g->raw_cap,
                                                 pos0,
                                                 n_tokens,
                                                 DS4_N_HEAD_DIM) != 0;
        if (ok) {
            metal_graph_debug_dump_tensor("raw_cache",
                                          g->layer_raw_cache[il],
                                          (uint64_t)n_raw * DS4_N_HEAD_DIM,
                                          il,
                                          pos0);
        }
        if (ok) {
            ok = ds4_gpu_attention_decode_raw_batch_heads_tensor(metal_graph_batch_heads(g),
                                                                   model->map,
                                                                   model->size,
                                                                   layer->attn_sinks->abs_offset,
                                                                   metal_graph_batch_q(g),
                                                                   g->layer_raw_cache[il],
                                                                   n_tokens,
                                                                   pos0,
                                                                   n_raw,
                                                                   g->raw_cap,
                                                                   raw_start,
                                                                   g->raw_window,
                                                                   DS4_N_HEAD,
                                                                   DS4_N_HEAD_DIM) != 0;
        }
        if (ok) batch_attention_done = true;
    } else if (ok && ratio != 0) {
        const uint32_t coff = ratio == 4 ? 2u : 1u;
        const uint32_t comp_width = coff * DS4_N_HEAD_DIM;
        const bool have_attn_comp = layer->attn_compressor_kv && layer->attn_compressor_gate &&
                                    layer->attn_compressor_ape && layer->attn_compressor_norm;
        if (!have_attn_comp) {
            fprintf(stderr, "ds4: Metal layer-major prefill needs attention compressor weights\n");
            ok = false;
        }
        if (ok) {
            ok = ds4_gpu_matmul_f16_tensor(metal_graph_batch_comp_kv(g),
                                             model->map,
                                             model->size,
                                             layer->attn_compressor_kv->abs_offset,
                                             DS4_N_EMBD,
                                             comp_width,
                                             metal_graph_batch_attn_norm(g),
                                             n_tokens) != 0;
            if (!ok) {
                fprintf(stderr, "ds4: gpu layer %u attention compressor KV projection failed\n", il);
            }
            if (ok) {
                ok = ds4_gpu_matmul_f16_tensor(metal_graph_batch_comp_sc(g),
                                                model->map,
                                                model->size,
                                                layer->attn_compressor_gate->abs_offset,
                                                DS4_N_EMBD,
                                                comp_width,
                                                metal_graph_batch_attn_norm(g),
                                                n_tokens) != 0;
                if (!ok) {
                    fprintf(stderr, "ds4: gpu layer %u attention compressor score projection failed\n", il);
                }
            }
        }
        if (ok) metal_graph_debug_dump_tensor("attn_comp_kv_raw",
                                              metal_graph_batch_comp_kv(g),
                                              (uint64_t)comp_width * n_tokens,
                                              il,
                                              pos0);
        if (ok) metal_graph_debug_dump_tensor("attn_comp_score_raw",
                                              metal_graph_batch_comp_sc(g),
                                              (uint64_t)comp_width * n_tokens,
                                              il,
                                              pos0);
        uint32_t n_comp = g->layer_n_comp[il];
        if (zero_prefix) {
            n_comp = n_tokens / ratio;
            if (ok && n_comp > g->layer_comp_cap[il]) {
                fprintf(stderr, "ds4: Metal layer-major compressed KV cache capacity exceeded at layer %u\n", il);
                ok = false;
            }
            if (ok && DS4_GPU_ATTN_COMP_CACHE_F16 && n_comp > g->attn_comp_stage_cap) {
                fprintf(stderr, "ds4: Metal graph compressed KV staging capacity exceeded at layer %u\n", il);
                ok = false;
            }
            ds4_gpu_tensor *attn_comp_target = NULL;
            if (ok) {
                attn_comp_target = metal_graph_attn_comp_prefill_target(g, il, 0, n_comp);
                if (!attn_comp_target) {
                    fprintf(stderr, "ds4: gpu layer %u attention compressor target creation failed\n", il);
                    ok = false;
                }
                if (ok) ok = ds4_gpu_compressor_prefill_tensor(attn_comp_target,
                                                         g->layer_attn_state_kv[il],
                                                         g->layer_attn_state_score[il],
                                                         metal_graph_batch_comp_kv(g),
                                                         metal_graph_batch_comp_sc(g),
                                                         model->map,
                                                         model->size,
                                                         layer->attn_compressor_ape->abs_offset,
                                                         layer->attn_compressor_ape->type,
                                                         layer->attn_compressor_norm->abs_offset,
                                                         layer->attn_compressor_norm->type,
                                                         DS4_N_HEAD_DIM,
                                                         ratio,
                                                         pos0,
                                                         n_tokens,
                                                         DS4_N_ROT,
                                                         compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                                                         true,
                                                         freq_base,
                                                         freq_scale,
                                                         ext_factor,
                                                         attn_factor,
                                                         DS4_ROPE_YARN_BETA_FAST,
                                                         DS4_ROPE_YARN_BETA_SLOW,
                                                         DS4_RMS_EPS) != 0;
                if (!ok) {
                    fprintf(stderr, "ds4: gpu layer %u attention compressor prefill failed\n", il);
                }
                DS4_METAL_PROFILE_ATTN_STAGE("compressor_prefill");
                if (ok && n_comp != 0) {
                    ok = metal_graph_commit_attn_comp_stage(g, il, 0, n_comp);
                }
                DS4_METAL_PROFILE_ATTN_STAGE("compressor_commit");
                if (ok && ratio == 4) {
                    ok = metal_graph_refresh_ratio4_compressor_state(g,
                                                                     model,
                                                                     g->layer_attn_state_kv[il],
                                                                     g->layer_attn_state_score[il],
                                                                     layer->attn_compressor_kv,
                                                                     layer->attn_compressor_gate,
                                                                     layer->attn_compressor_ape,
                                                                     DS4_N_HEAD_DIM,
                                                                     comp_width,
                                                                     pos0,
                                                                     n_tokens);
                }
                DS4_METAL_PROFILE_ATTN_STAGE("compressor_refresh");
            }
            if (ok) {
                g->layer_n_comp[il] = n_comp;
                for (uint32_t t = 0; t < n_tokens; t++) {
                    comp_counts[t] = (pos0 + t + 1u) / ratio;
                }
                if (n_comp != 0) {
                    metal_graph_debug_dump_tensor("KVcompress",
                                                  attn_comp_target,
                                                  (uint64_t)n_comp * DS4_N_HEAD_DIM,
                                                  il,
                                                  pos0);
                }
                metal_graph_debug_dump_tensor("attn_state_kv",
                                              g->layer_attn_state_kv[il],
                                              (uint64_t)comp_width * coff * ratio,
                                              il,
                                              pos0);
                metal_graph_debug_dump_tensor("attn_state_score",
                                              g->layer_attn_state_score[il],
                                              (uint64_t)comp_width * coff * ratio,
                                              il,
                                              pos0);
            }
            metal_graph_attn_comp_prefill_target_free(attn_comp_target);
        } else {
            const bool aligned_chunk =
                getenv("DS4_CUDA_NO_COMPRESSOR_PREFILL_BATCH") == NULL &&
                (pos0 % ratio) == 0u && (n_tokens % ratio) == 0u;
            if (aligned_chunk) {
                const uint32_t comp_before = g->layer_n_comp[il];
                const uint32_t comp_chunk = n_tokens / ratio;
                if (comp_before + comp_chunk > g->layer_comp_cap[il]) {
                    fprintf(stderr, "ds4: Metal graph compressed KV cache capacity exceeded at layer %u\n", il);
                    ok = false;
                }
                if (ok && DS4_GPU_ATTN_COMP_CACHE_F16 && comp_chunk > g->attn_comp_stage_cap) {
                    fprintf(stderr, "ds4: Metal graph compressed KV staging capacity exceeded at layer %u\n", il);
                    ok = false;
                }
                ds4_gpu_tensor *attn_comp_target =
                    ok ? metal_graph_attn_comp_prefill_target(g, il, comp_before, comp_chunk) : NULL;
                if (ok && !attn_comp_target) ok = false;
                if (ok && ratio == 4) {
                    ok = ds4_gpu_compressor_prefill_ratio4_replay_tensor(
                            attn_comp_target,
                            g->layer_attn_state_kv[il],
                            g->layer_attn_state_score[il],
                            metal_graph_batch_comp_kv(g),
                            metal_graph_batch_comp_sc(g),
                            model->map,
                            model->size,
                            layer->attn_compressor_ape->abs_offset,
                            layer->attn_compressor_ape->type,
                            layer->attn_compressor_norm->abs_offset,
                            layer->attn_compressor_norm->type,
                            DS4_N_HEAD_DIM,
                            pos0,
                            n_tokens,
                            DS4_N_ROT,
                            compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                            true,
                            freq_base,
                            freq_scale,
                            ext_factor,
                            attn_factor,
                            DS4_ROPE_YARN_BETA_FAST,
                            DS4_ROPE_YARN_BETA_SLOW,
                            DS4_RMS_EPS) != 0;
                } else if (ok) {
                    ok = ds4_gpu_compressor_prefill_tensor(
                            attn_comp_target,
                            g->layer_attn_state_kv[il],
                            g->layer_attn_state_score[il],
                            metal_graph_batch_comp_kv(g),
                            metal_graph_batch_comp_sc(g),
                            model->map,
                            model->size,
                            layer->attn_compressor_ape->abs_offset,
                            layer->attn_compressor_ape->type,
                            layer->attn_compressor_norm->abs_offset,
                            layer->attn_compressor_norm->type,
                            DS4_N_HEAD_DIM,
                            ratio,
                            pos0,
                            n_tokens,
                            DS4_N_ROT,
                            compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                            true,
                            freq_base,
                            freq_scale,
                            ext_factor,
                            attn_factor,
                            DS4_ROPE_YARN_BETA_FAST,
                            DS4_ROPE_YARN_BETA_SLOW,
                            DS4_RMS_EPS) != 0;
                }
                if (ok && comp_chunk != 0) {
                    ok = metal_graph_commit_attn_comp_stage(g, il, comp_before, comp_chunk);
                }
                if (ok && ratio == 4) {
                    ok = metal_graph_refresh_ratio4_compressor_state(g,
                                                                     model,
                                                                     g->layer_attn_state_kv[il],
                                                                     g->layer_attn_state_score[il],
                                                                     layer->attn_compressor_kv,
                                                                     layer->attn_compressor_gate,
                                                                     layer->attn_compressor_ape,
                                                                     DS4_N_HEAD_DIM,
                                                                     comp_width,
                                                                     pos0,
                                                                     n_tokens);
                }
                if (ok) {
                    g->layer_n_comp[il] = comp_before + comp_chunk;
                    if (comp_counts) {
                        for (uint32_t t = 0; t < n_tokens; t++) {
                            comp_counts[t] = (pos0 + t + 1u) / ratio;
                        }
                    }
                    metal_graph_debug_dump_tensor("KVcompress",
                                                  attn_comp_target,
                                                  (uint64_t)comp_chunk * DS4_N_HEAD_DIM,
                                                  il,
                                                  pos0);
                    metal_graph_debug_dump_tensor("attn_state_kv",
                                                  g->layer_attn_state_kv[il],
                                                  (uint64_t)comp_width * coff * ratio,
                                                  il,
                                                  pos0);
                    metal_graph_debug_dump_tensor("attn_state_score",
                                                  g->layer_attn_state_score[il],
                                                  (uint64_t)comp_width * coff * ratio,
                                                  il,
                                                  pos0);
                }
                metal_graph_attn_comp_prefill_target_free(attn_comp_target);
            } else {
                for (uint32_t t = 0; ok && t < n_tokens; t++) {
                    const uint32_t pos = pos0 + t;
                    const bool emit = ((pos + 1u) % ratio) == 0u;
                    if (emit && g->layer_n_comp[il] >= g->layer_comp_cap[il]) {
                        fprintf(stderr, "ds4: Metal graph compressed KV cache capacity exceeded at layer %u\n", il);
                        ok = false;
                        break;
                    }
                    ds4_gpu_tensor *kv_view = metal_graph_tensor_row_view(metal_graph_batch_comp_kv(g), t, comp_width);
                    ds4_gpu_tensor *sc_view = metal_graph_tensor_row_view(metal_graph_batch_comp_sc(g), t, comp_width);
                    const uint32_t comp_row = g->layer_n_comp[il];
                    ok = kv_view && sc_view &&
                         ds4_gpu_compressor_update_tensor(kv_view,
                                                            sc_view,
                                                            g->layer_attn_state_kv[il],
                                                            g->layer_attn_state_score[il],
                                                            metal_graph_attn_comp_update_target(g, il),
                                                            model->map,
                                                            model->size,
                                                            layer->attn_compressor_ape->abs_offset,
                                                            layer->attn_compressor_ape->type,
                                                            layer->attn_compressor_norm->abs_offset,
                                                            layer->attn_compressor_norm->type,
                                                            DS4_N_HEAD_DIM,
                                                            ratio,
                                                            pos,
                                                            metal_graph_attn_comp_update_row(comp_row),
                                                            DS4_N_ROT,
                                                            compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                                                            freq_base,
                                                            freq_scale,
                                                            ext_factor,
                                                            attn_factor,
                                                            DS4_ROPE_YARN_BETA_FAST,
                                                            DS4_ROPE_YARN_BETA_SLOW,
                                                            DS4_RMS_EPS,
                                                            false) != 0;
                    if (ok && emit) {
                        ds4_gpu_tensor *comp_row_view = metal_graph_attn_comp_row_view(g, il, comp_row);
                        ok = comp_row_view &&
                             ds4_gpu_dsv4_fp8_kv_quantize_tensor(comp_row_view,
                                                                   1,
                                                                   DS4_N_HEAD_DIM,
                                                                   DS4_N_ROT) != 0;
                        if (ok) {
                            metal_graph_debug_dump_tensor("KVcompress",
                                                          comp_row_view,
                                                          DS4_N_HEAD_DIM,
                                                          il,
                                                          pos);
                        }
                        ds4_gpu_tensor_free(comp_row_view);
                        if (ok) ok = metal_graph_commit_attn_comp_stage(g, il, comp_row, 1);
                    }
                    if (ok && emit) g->layer_n_comp[il]++;
                    if (comp_counts) comp_counts[t] = g->layer_n_comp[il];
                    if (ok && t == 0) ok = metal_graph_capture_prefix1_attn_state(g, il);
                    ds4_gpu_tensor_free(sc_view);
                    ds4_gpu_tensor_free(kv_view);
                }
            }
            n_comp = g->layer_n_comp[il];
        }
        DS4_METAL_PROFILE_ATTN_STAGE("compressor");

        if (ok && ratio == 4) {
            const uint32_t index_width = coff * DS4_N_INDEXER_HEAD_DIM;
            if (!layer->indexer_compressor_kv || !layer->indexer_compressor_gate ||
                !layer->indexer_compressor_ape || !layer->indexer_compressor_norm ||
                !layer->indexer_attn_q_b || !layer->indexer_proj) {
                fprintf(stderr, "ds4: Metal layer-major prefill needs indexer weights\n");
                ok = false;
            }
            if (ok) {
                ok = ds4_gpu_matmul_f16_tensor(metal_graph_batch_comp_kv(g),
                                                 model->map,
                                                 model->size,
                                                 layer->indexer_compressor_kv->abs_offset,
                                                 DS4_N_EMBD,
                                                 index_width,
                                                 metal_graph_batch_attn_norm(g),
                                                 n_tokens) != 0;
                if (ok) ok = ds4_gpu_matmul_f16_tensor(metal_graph_batch_comp_sc(g),
                                                         model->map,
                                                         model->size,
                                                         layer->indexer_compressor_gate->abs_offset,
                                                         DS4_N_EMBD,
                                                         index_width,
                                                         metal_graph_batch_attn_norm(g),
                                                         n_tokens) != 0;
            }
            if (ok) metal_graph_debug_dump_tensor("indexer_comp_kv_raw",
                                                  metal_graph_batch_comp_kv(g),
                                                  (uint64_t)index_width * n_tokens,
                                                  il,
                                                  pos0);
            if (ok) metal_graph_debug_dump_tensor("indexer_comp_score_raw",
                                                  metal_graph_batch_comp_sc(g),
                                                  (uint64_t)index_width * n_tokens,
                                                  il,
                                                  pos0);
            if (ok) ok = metal_graph_matmul_plain_tensor(metal_graph_batch_indexer_q(g),
                                                          model,
                                                          layer->indexer_attn_q_b,
                                                          q_rank,
                                                          (uint64_t)DS4_N_INDEXER_HEAD * DS4_N_INDEXER_HEAD_DIM,
                                                          metal_graph_batch_qr_norm(g),
                                                          n_tokens);
            if (ok) ok = ds4_gpu_rope_tail_tensor(metal_graph_batch_indexer_q(g),
                                                    n_tokens,
                                                    DS4_N_INDEXER_HEAD,
                                                    DS4_N_INDEXER_HEAD_DIM,
                                                    DS4_N_ROT,
                                                    pos0,
                                                    compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                                                    false,
                                                    freq_base,
                                                    freq_scale,
                                                    ext_factor,
                                                    attn_factor,
                                                    DS4_ROPE_YARN_BETA_FAST,
                                                    DS4_ROPE_YARN_BETA_SLOW) != 0;
            if (ok) ok = ds4_gpu_dsv4_indexer_qat_tensor(metal_graph_batch_indexer_q(g),
                                                          n_tokens * DS4_N_INDEXER_HEAD,
                                                          DS4_N_INDEXER_HEAD_DIM) != 0;
            if (ok) ok = ds4_gpu_matmul_f16_tensor(metal_graph_batch_indexer_weights(g),
                                                     model->map,
                                                     model->size,
                                                     layer->indexer_proj->abs_offset,
                                                     DS4_N_EMBD,
                                                     DS4_N_INDEXER_HEAD,
                                                     metal_graph_batch_attn_norm(g),
                                                     n_tokens) != 0;
            if (zero_prefix) {
                if (ok && n_comp > g->layer_comp_cap[il]) {
                    fprintf(stderr, "ds4: Metal layer-major indexer cache capacity exceeded at layer %u\n", il);
                    ok = false;
                }
                if (ok) {
                    ok = ds4_gpu_compressor_prefill_tensor(g->layer_index_comp_cache[il],
                                                             g->layer_index_state_kv[il],
                                                             g->layer_index_state_score[il],
                                                             metal_graph_batch_comp_kv(g),
                                                             metal_graph_batch_comp_sc(g),
                                                             model->map,
                                                             model->size,
                                                             layer->indexer_compressor_ape->abs_offset,
                                                             layer->indexer_compressor_ape->type,
                                                             layer->indexer_compressor_norm->abs_offset,
                                                             layer->indexer_compressor_norm->type,
                                                             DS4_N_INDEXER_HEAD_DIM,
                                                             ratio,
                                                             pos0,
                                                             n_tokens,
                                                             DS4_N_ROT,
                                                             compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                                                             false,
                                                             freq_base,
                                                             freq_scale,
                                                             ext_factor,
                                                             attn_factor,
                                                             DS4_ROPE_YARN_BETA_FAST,
                                                             DS4_ROPE_YARN_BETA_SLOW,
                                                             DS4_RMS_EPS) != 0;
                }
                if (ok && n_comp != 0) {
                    ok = ds4_gpu_dsv4_indexer_qat_tensor(g->layer_index_comp_cache[il],
                                                          n_comp,
                                                          DS4_N_INDEXER_HEAD_DIM) != 0;
                }
                if (ok) {
                    ok = metal_graph_refresh_ratio4_compressor_state(g,
                                                                     model,
                                                                     g->layer_index_state_kv[il],
                                                                     g->layer_index_state_score[il],
                                                                     layer->indexer_compressor_kv,
                                                                     layer->indexer_compressor_gate,
                                                                     layer->indexer_compressor_ape,
                                                                     DS4_N_INDEXER_HEAD_DIM,
                                                                     index_width,
                                                                     pos0,
                                                                     n_tokens);
                }
                if (ok) {
                    g->layer_n_index_comp[il] = n_comp;
                    for (uint32_t t = 0; t < n_tokens; t++) {
                        index_counts[t] = (pos0 + t + 1u) / ratio;
                    }
                    if (n_comp != 0) {
                        metal_graph_debug_dump_tensor("indexer_KVcompress",
                                                      g->layer_index_comp_cache[il],
                                                      (uint64_t)n_comp * DS4_N_INDEXER_HEAD_DIM,
                                                      il,
                                                      pos0);
                    }
                    metal_graph_debug_dump_tensor("indexer_state_kv",
                                                  g->layer_index_state_kv[il],
                                                  (uint64_t)index_width * coff * ratio,
                                                  il,
                                                  pos0);
                    metal_graph_debug_dump_tensor("indexer_state_score",
                                                  g->layer_index_state_score[il],
                                                  (uint64_t)index_width * coff * ratio,
                                                  il,
                                                  pos0);
                }
            } else {
                const bool aligned_chunk =
                    getenv("DS4_CUDA_NO_COMPRESSOR_PREFILL_BATCH") == NULL &&
                    (pos0 % ratio) == 0u && (n_tokens % ratio) == 0u;
                if (aligned_chunk) {
                    const uint32_t index_before = g->layer_n_index_comp[il];
                    const uint32_t index_chunk = n_tokens / ratio;
                    if (index_before + index_chunk > g->layer_comp_cap[il]) {
                        fprintf(stderr, "ds4: Metal graph indexer compressed KV cache capacity exceeded at layer %u\n", il);
                        ok = false;
                    }
                    ds4_gpu_tensor *index_view = NULL;
                    if (ok) {
                        index_view = ds4_gpu_tensor_view(
                                g->layer_index_comp_cache[il],
                                (uint64_t)index_before * DS4_N_INDEXER_HEAD_DIM * sizeof(float),
                                (uint64_t)index_chunk * DS4_N_INDEXER_HEAD_DIM * sizeof(float));
                        ok = index_view != NULL;
                    }
                    if (ok) {
                        ok = ds4_gpu_compressor_prefill_ratio4_replay_tensor(
                                index_view,
                                g->layer_index_state_kv[il],
                                g->layer_index_state_score[il],
                                metal_graph_batch_comp_kv(g),
                                metal_graph_batch_comp_sc(g),
                                model->map,
                                model->size,
                                layer->indexer_compressor_ape->abs_offset,
                                layer->indexer_compressor_ape->type,
                                layer->indexer_compressor_norm->abs_offset,
                                layer->indexer_compressor_norm->type,
                                DS4_N_INDEXER_HEAD_DIM,
                                pos0,
                                n_tokens,
                                DS4_N_ROT,
                                compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                                false,
                                freq_base,
                                freq_scale,
                                ext_factor,
                                attn_factor,
                                DS4_ROPE_YARN_BETA_FAST,
                                DS4_ROPE_YARN_BETA_SLOW,
                                DS4_RMS_EPS) != 0;
                    }
                    if (ok && index_chunk != 0) {
                        ok = ds4_gpu_dsv4_indexer_qat_tensor(index_view,
                                                              index_chunk,
                                                              DS4_N_INDEXER_HEAD_DIM) != 0;
                    }
                    if (ok) {
                        ok = metal_graph_refresh_ratio4_compressor_state(g,
                                                                         model,
                                                                         g->layer_index_state_kv[il],
                                                                         g->layer_index_state_score[il],
                                                                         layer->indexer_compressor_kv,
                                                                         layer->indexer_compressor_gate,
                                                                         layer->indexer_compressor_ape,
                                                                         DS4_N_INDEXER_HEAD_DIM,
                                                                         index_width,
                                                                         pos0,
                                                                         n_tokens);
                    }
                    if (ok) {
                        g->layer_n_index_comp[il] = index_before + index_chunk;
                        if (index_counts) {
                            for (uint32_t t = 0; t < n_tokens; t++) {
                                index_counts[t] = (pos0 + t + 1u) / ratio;
                            }
                        }
                        metal_graph_debug_dump_tensor("indexer_KVcompress",
                                                      index_view,
                                                      (uint64_t)index_chunk * DS4_N_INDEXER_HEAD_DIM,
                                                      il,
                                                      pos0);
                        metal_graph_debug_dump_tensor("indexer_state_kv",
                                                      g->layer_index_state_kv[il],
                                                      (uint64_t)index_width * coff * ratio,
                                                      il,
                                                      pos0);
                        metal_graph_debug_dump_tensor("indexer_state_score",
                                                      g->layer_index_state_score[il],
                                                      (uint64_t)index_width * coff * ratio,
                                                      il,
                                                      pos0);
                    }
                    ds4_gpu_tensor_free(index_view);
                } else {
                    for (uint32_t t = 0; ok && t < n_tokens; t++) {
                        const uint32_t pos = pos0 + t;
                        const bool emit = ((pos + 1u) % ratio) == 0u;
                        if (emit && g->layer_n_index_comp[il] >= g->layer_comp_cap[il]) {
                            fprintf(stderr, "ds4: Metal graph indexer compressed KV cache capacity exceeded at layer %u\n", il);
                            ok = false;
                            break;
                        }
                        ds4_gpu_tensor *kv_view = metal_graph_tensor_row_view(metal_graph_batch_comp_kv(g), t, index_width);
                        ds4_gpu_tensor *sc_view = metal_graph_tensor_row_view(metal_graph_batch_comp_sc(g), t, index_width);
                        const uint32_t index_row = g->layer_n_index_comp[il];
                        ok = kv_view && sc_view &&
                             ds4_gpu_compressor_update_tensor(kv_view,
                                                                sc_view,
                                                                g->layer_index_state_kv[il],
                                                                g->layer_index_state_score[il],
                                                                g->layer_index_comp_cache[il],
                                                                model->map,
                                                                model->size,
                                                                layer->indexer_compressor_ape->abs_offset,
                                                                layer->indexer_compressor_ape->type,
                                                                layer->indexer_compressor_norm->abs_offset,
                                                                layer->indexer_compressor_norm->type,
                                                                DS4_N_INDEXER_HEAD_DIM,
                                                                ratio,
                                                                pos,
                                                                index_row,
                                                                DS4_N_ROT,
                                                                compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                                                                freq_base,
                                                                freq_scale,
                                                                ext_factor,
                                                                attn_factor,
                                                                DS4_ROPE_YARN_BETA_FAST,
                                                                DS4_ROPE_YARN_BETA_SLOW,
                                                                DS4_RMS_EPS,
                                                                false) != 0;
                        if (ok && emit) {
                            ds4_gpu_tensor *index_row_view = ds4_gpu_tensor_view(
                                    g->layer_index_comp_cache[il],
                                    (uint64_t)index_row * DS4_N_INDEXER_HEAD_DIM * sizeof(float),
                                    (uint64_t)DS4_N_INDEXER_HEAD_DIM * sizeof(float));
                            if (!index_row_view) {
                                ok = false;
                            } else {
                                ok = ds4_gpu_dsv4_indexer_qat_tensor(index_row_view,
                                                                      1,
                                                                      DS4_N_INDEXER_HEAD_DIM) != 0;
                                ds4_gpu_tensor_free(index_row_view);
                            }
                        }
                        if (ok && emit) g->layer_n_index_comp[il]++;
                        if (index_counts) index_counts[t] = g->layer_n_index_comp[il];
                        if (ok && t == 0) ok = metal_graph_capture_prefix1_index_state(g, il);
                        ds4_gpu_tensor_free(sc_view);
                        ds4_gpu_tensor_free(kv_view);
                    }
                }
            }
        }
        if (ratio == 4) DS4_METAL_PROFILE_ATTN_STAGE("indexer_setup");

        if (ok && !zero_prefix && n_tokens <= g->raw_cap) {
            const uint32_t n_raw = metal_graph_raw_span_for_batch(g, pos0, n_tokens);
            /* See the raw-only branch above: batched mixed attention also
             * consumes a logical raw window, linearized out of the ring. */
            const uint32_t raw_start = metal_graph_raw_start_for_span(g,
                                                                      pos0 + n_tokens - 1u,
                                                                      n_raw);
            uint32_t use_comp_mask = 0;
            bool use_indexed_comp = false;
            double index_stage_t0 = 0.0;

            ok = ds4_gpu_store_raw_kv_batch_tensor(g->layer_raw_cache[il],
                                                     metal_graph_batch_kv(g),
                                                     g->raw_cap,
                                                     pos0,
                                                     n_tokens,
                                                     DS4_N_HEAD_DIM) != 0;
            if (ok && ratio == 4 && n_comp > DS4_N_INDEXER_TOP_K) {
                const float index_scale = 1.0f / sqrtf((float)(DS4_N_INDEXER_HEAD_DIM * DS4_N_INDEXER_HEAD));
                if (index_stage_profile) {
                    ok = metal_graph_indexer_stage_profile_boundary(NULL,
                                                                    il,
                                                                    pos0,
                                                                    n_tokens,
                                                                    n_comp,
                                                                    &index_stage_t0);
                }
                ok = ds4_gpu_indexer_scores_decode_batch_tensor(metal_graph_indexer_scores(g),
                                                                  metal_graph_batch_indexer_q(g),
                                                                  metal_graph_batch_indexer_weights(g),
                                                                  g->layer_index_comp_cache[il],
                                                                  n_comp,
                                                                  n_tokens,
                                                                  pos0,
                                                                  DS4_N_INDEXER_HEAD,
                                                                  DS4_N_INDEXER_HEAD_DIM,
                                                                  ratio,
                                                                  index_scale) != 0;
                if (ok && index_stage_profile) {
                    ok = metal_graph_indexer_stage_profile_boundary("score",
                                                                    il,
                                                                    pos0,
                                                                    n_tokens,
                                                                    n_comp,
                                                                    &index_stage_t0);
                }
                if (ok) {
                    metal_graph_debug_dump_tensor("indexer_scores",
                                                  metal_graph_indexer_scores(g),
                                                  (uint64_t)n_comp * n_tokens,
                                                  il,
                                                  pos0);
                }
                if (ok) {
                    ok = ds4_gpu_indexer_topk_tensor(metal_graph_comp_selected(g),
                                                       metal_graph_indexer_scores(g),
                                                       n_comp,
                                                       n_tokens,
                                                       DS4_N_INDEXER_TOP_K) != 0;
                    if (ok && index_stage_profile) {
                        ok = metal_graph_indexer_stage_profile_boundary("topk",
                                                                        il,
                                                                        pos0,
                                                                        n_tokens,
                                                                        n_comp,
                                                                        &index_stage_t0);
                    }
                    if (ok) {
                        metal_graph_debug_dump_i32_tensor("indexer_topk",
                                                          metal_graph_comp_selected(g),
                                                          (uint64_t)n_tokens * DS4_N_INDEXER_TOP_K,
                                                          il,
                                                          pos0);
                    }
                }
                if (ok) {
                    use_indexed_comp = true;
                }
                use_comp_mask = 1;
            }
            if (ok) {
                if (use_indexed_comp) {
                    ok = ds4_gpu_attention_indexed_mixed_batch_heads_tensor(metal_graph_batch_heads(g),
                                                                              model->map,
                                                                              model->size,
                                                                              layer->attn_sinks->abs_offset,
                                                                              metal_graph_batch_q(g),
                                                                              g->layer_raw_cache[il],
                                                                              g->layer_attn_comp_cache[il],
                                                                              metal_graph_attn_comp_cache_is_f16(),
                                                                              metal_graph_comp_selected(g),
                                                                              n_tokens,
                                                                              pos0,
                                                                              n_raw,
                                                                              g->raw_cap,
                                                                              raw_start,
                                                                              n_comp,
                                                                              DS4_N_INDEXER_TOP_K,
                                                                              g->raw_window,
                                                                              ratio,
                                                                              DS4_N_HEAD,
                                                                              DS4_N_HEAD_DIM) != 0;
                    if (ok && index_stage_profile) {
                        ok = metal_graph_indexer_stage_profile_boundary("attention",
                                                                        il,
                                                                        pos0,
                                                                        n_tokens,
                                                                        n_comp,
                                                                        &index_stage_t0);
                    }
                } else {
                    ok = ds4_gpu_attention_decode_mixed_batch_heads_tensor(metal_graph_batch_heads(g),
                                                                             model->map,
                                                                             model->size,
                                                                             layer->attn_sinks->abs_offset,
                                                                             metal_graph_batch_q(g),
                                                                             g->layer_raw_cache[il],
                                                                             g->layer_attn_comp_cache[il],
                                                                             metal_graph_attn_comp_cache_is_f16(),
                                                                             use_comp_mask ? metal_graph_comp_mask(g) : NULL,
                                                                             use_comp_mask,
                                                                             n_tokens,
                                                                             pos0,
                                                                             n_raw,
                                                                             g->raw_cap,
                                                                             raw_start,
                                                                             n_comp,
                                                                             g->raw_window,
                                                                             ratio,
                                                                             DS4_N_HEAD,
                                                                             DS4_N_HEAD_DIM) != 0;
                }
            }
            if (ok) batch_attention_done = true;
        }

        const bool topk_prefill_needed = ratio == 4 && n_comp > DS4_N_INDEXER_TOP_K;
        if (ok && zero_prefix && topk_prefill_needed && n_comp != 0) {
            const float index_scale = 1.0f / sqrtf((float)(DS4_N_INDEXER_HEAD_DIM * DS4_N_INDEXER_HEAD));
            double index_stage_t0 = 0.0;
            if (index_stage_profile) {
                ok = metal_graph_indexer_stage_profile_boundary(NULL,
                                                                il,
                                                                pos0,
                                                                n_tokens,
                                                                n_comp,
                                                                &index_stage_t0);
            }
            ok = ds4_gpu_indexer_scores_prefill_tensor(metal_graph_indexer_scores(g),
                                                         metal_graph_batch_indexer_q(g),
                                                         metal_graph_batch_indexer_weights(g),
                                                         g->layer_index_comp_cache[il],
                                                         n_comp,
                                                         n_tokens,
                                                         DS4_N_INDEXER_HEAD,
                                                         DS4_N_INDEXER_HEAD_DIM,
                                                         ratio,
                                                         index_scale) != 0;
            if (ok && index_stage_profile) {
                ok = metal_graph_indexer_stage_profile_boundary("score",
                                                                il,
                                                                pos0,
                                                                n_tokens,
                                                                n_comp,
                                                                &index_stage_t0);
            }
            if (ok) {
                metal_graph_debug_dump_tensor("indexer_scores",
                                              metal_graph_indexer_scores(g),
                                              (uint64_t)n_comp * n_tokens,
                                              il,
                                              pos0);
            }
            if (ok) {
                ok = ds4_gpu_indexer_topk_tensor(metal_graph_comp_selected(g),
                                                   metal_graph_indexer_scores(g),
                                                   n_comp,
                                                   n_tokens,
                                                   DS4_N_INDEXER_TOP_K) != 0;
                if (ok && index_stage_profile) {
                    ok = metal_graph_indexer_stage_profile_boundary("topk",
                                                                    il,
                                                                    pos0,
                                                                    n_tokens,
                                                                    n_comp,
                                                                    &index_stage_t0);
                }
                if (ok) {
                    metal_graph_debug_dump_i32_tensor("indexer_topk",
                                                      metal_graph_comp_selected(g),
                                                      (uint64_t)n_tokens * DS4_N_INDEXER_TOP_K,
                                                      il,
                                                      pos0);
                }
            }
            if (ok && tp_row_split_attn) {
                /* Score/top-k selection above ran replicated over all rows;
                 * only the attention consumption splits.  Passing the row
                 * offset through pos0 and clamping n_raw to the rows this
                 * rank can see keeps the kernel's first_raw_pos at the
                 * chunk origin, so the raw ring mapping is unchanged. */
                ds4_gpu_tensor *tp_topk = metal_graph_tensor_row_range_view(
                        metal_graph_comp_selected(g), tp_row0, tp_rows, DS4_N_INDEXER_TOP_K);
                ok = tp_topk &&
                     ds4_gpu_attention_indexed_mixed_batch_heads_tensor(tp_heads,
                                                                          model->map,
                                                                          model->size,
                                                                          layer->attn_sinks->abs_offset,
                                                                          tp_q,
                                                                          g->layer_raw_cache[il],
                                                                          g->layer_attn_comp_cache[il],
                                                                          metal_graph_attn_comp_cache_is_f16(),
                                                                          tp_topk,
                                                                          tp_rows,
                                                                          pos0 + tp_row0,
                                                                          tp_row0 + tp_rows,
                                                                          g->raw_cap,
                                                                          0,
                                                                          n_comp,
                                                                          DS4_N_INDEXER_TOP_K,
                                                                          g->raw_window,
                                                                          ratio,
                                                                          DS4_N_HEAD,
                                                                          DS4_N_HEAD_DIM) != 0;
                ds4_gpu_tensor_free(tp_topk);
                if (ok && index_stage_profile) {
                    ok = metal_graph_indexer_stage_profile_boundary("attention",
                                                                    il,
                                                                    pos0,
                                                                    n_tokens,
                                                                    n_comp,
                                                                    &index_stage_t0);
                }
            } else if (ok) {
                ok = ds4_gpu_attention_indexed_mixed_batch_heads_tensor(metal_graph_batch_heads(g),
                                                                          model->map,
                                                                          model->size,
                                                                          layer->attn_sinks->abs_offset,
                                                                          metal_graph_batch_q(g),
                                                                          g->layer_raw_cache[il],
                                                                          g->layer_attn_comp_cache[il],
                                                                          metal_graph_attn_comp_cache_is_f16(),
                                                                          metal_graph_comp_selected(g),
                                                                          n_tokens,
                                                                          pos0,
                                                                          n_tokens,
                                                                          g->raw_cap,
                                                                          0,
                                                                          n_comp,
                                                                          DS4_N_INDEXER_TOP_K,
                                                                          g->raw_window,
                                                                          ratio,
                                                                          DS4_N_HEAD,
                                                                          DS4_N_HEAD_DIM) != 0;
                if (ok && index_stage_profile) {
                    ok = metal_graph_indexer_stage_profile_boundary("attention",
                                                                    il,
                                                                    pos0,
                                                                    n_tokens,
                                                                    n_comp,
                                                                    &index_stage_t0);
                }
            }
            if (ok) batch_attention_done = true;
        }
        if (ok && zero_prefix && !topk_prefill_needed && n_comp != 0) {
            if (tp_row_split_attn) {
                ok = ds4_gpu_attention_prefill_static_mixed_heads_range_tensor(tp_heads,
                                                                                 model->map,
                                                                                 model->size,
                                                                                 layer->attn_sinks->abs_offset,
                                                                                 tp_q,
                                                                                 metal_graph_batch_kv(g),
                                                                                 g->layer_attn_comp_cache[il],
                                                                                 metal_graph_attn_comp_cache_is_f16(),
                                                                                 tp_row0,
                                                                                 tp_rows,
                                                                                 n_tokens,
                                                                                 n_comp,
                                                                                 g->raw_window,
                                                                                 ratio,
                                                                                 DS4_N_HEAD,
                                                                                 DS4_N_HEAD_DIM) != 0;
            } else {
                ok = ds4_gpu_attention_prefill_static_mixed_heads_tensor(metal_graph_batch_heads(g),
                                                                           model->map,
                                                                           model->size,
                                                                           layer->attn_sinks->abs_offset,
                                                                           metal_graph_batch_q(g),
                                                                           metal_graph_batch_kv(g),
                                                                           g->layer_attn_comp_cache[il],
                                                                           metal_graph_attn_comp_cache_is_f16(),
                                                                           n_tokens,
                                                                           n_comp,
                                                                           g->raw_window,
                                                                           ratio,
                                                                           DS4_N_HEAD,
                                                                           DS4_N_HEAD_DIM) != 0;
            }
            if (ok) batch_attention_done = true;
        }
    }

    if (ok && !raw_batch_attention && !batch_attention_done) {
        uint32_t raw_prefix_tokens = 0;
        if (zero_prefix && ratio != 0 && n_tokens <= g->raw_cap && comp_counts != NULL) {
            while (raw_prefix_tokens < n_tokens && comp_counts[raw_prefix_tokens] == 0u) {
                raw_prefix_tokens++;
            }
        }

        if (raw_prefix_tokens != 0) {
            if (tp_row_split_attn && raw_prefix_tokens == n_tokens) {
                /* tp_attn_full_raw guarantees the whole chunk stays raw
                 * (n_tokens < ratio), so the split covers every row. */
                ok = ds4_gpu_attention_prefill_raw_heads_range_tensor(tp_heads,
                                                                        model->map,
                                                                        model->size,
                                                                        layer->attn_sinks->abs_offset,
                                                                        tp_q,
                                                                        metal_graph_batch_kv(g),
                                                                        tp_row0,
                                                                        tp_rows,
                                                                        n_tokens,
                                                                        g->raw_window,
                                                                        DS4_N_HEAD,
                                                                        DS4_N_HEAD_DIM) != 0;
            } else {
                ok = ds4_gpu_attention_prefill_raw_heads_tensor(metal_graph_batch_heads(g),
                                                                  model->map,
                                                                  model->size,
                                                                  layer->attn_sinks->abs_offset,
                                                                  metal_graph_batch_q(g),
                                                                  metal_graph_batch_kv(g),
                                                                  raw_prefix_tokens,
                                                                  g->raw_window,
                                                                  DS4_N_HEAD,
                                                                  DS4_N_HEAD_DIM) != 0;
            }
        }
        if (raw_prefix_tokens < n_tokens) {
            for (uint32_t t = raw_prefix_tokens; ok && t < n_tokens; t++) {
                const uint32_t pos = pos0 + t;
                const uint32_t n_raw = metal_graph_raw_span_for_batch(g, pos, 1);
                const uint32_t raw_start = metal_graph_raw_start_for_span(g, pos, n_raw);
                const uint32_t cur_comp = comp_counts ? comp_counts[t] : 0u;
                const uint32_t cur_index = index_counts ? index_counts[t] : 0u;
                uint32_t n_selected = 0;
                ds4_gpu_tensor *comp_mask = NULL;

                if (ratio == 4 && cur_comp > DS4_N_INDEXER_TOP_K) {
                    const float index_scale = 1.0f / sqrtf((float)(DS4_N_INDEXER_HEAD_DIM * DS4_N_INDEXER_HEAD));
                    ds4_gpu_tensor *indexer_q_view = metal_graph_tensor_row_view(
                            metal_graph_batch_indexer_q(g), t, (uint64_t)DS4_N_INDEXER_HEAD * DS4_N_INDEXER_HEAD_DIM);
                    ds4_gpu_tensor *indexer_w_view = metal_graph_tensor_row_view(
                            metal_graph_batch_indexer_weights(g), t, DS4_N_INDEXER_HEAD);
                    ok = indexer_q_view && indexer_w_view &&
                         ds4_gpu_indexer_score_one_tensor(metal_graph_indexer_scores(g),
                                                            indexer_q_view,
                                                            indexer_w_view,
                                                            g->layer_index_comp_cache[il],
                                                            cur_index,
                                                            DS4_N_INDEXER_HEAD,
                                                            DS4_N_INDEXER_HEAD_DIM,
                                                            index_scale) != 0 &&
                         ds4_gpu_indexer_topk_tensor(metal_graph_comp_selected(g),
                                                       metal_graph_indexer_scores(g),
                                                       cur_index,
                                                       1,
                                                       DS4_N_INDEXER_TOP_K) != 0 &&
                         ds4_gpu_dsv4_topk_mask_tensor(metal_graph_comp_mask(g),
                                                         metal_graph_comp_selected(g),
                                                         cur_index,
                                                         1,
                                                         DS4_N_INDEXER_TOP_K) != 0;
                    ds4_gpu_tensor_free(indexer_w_view);
                    ds4_gpu_tensor_free(indexer_q_view);
                    if (ok) {
                        comp_mask = metal_graph_comp_mask(g);
                        n_selected = DS4_N_INDEXER_TOP_K < cur_index
                            ? DS4_N_INDEXER_TOP_K
                            : cur_index;
                    }
                }

                ds4_gpu_tensor *q_view = metal_graph_tensor_row_view(metal_graph_batch_q(g), t, q_dim);
                ds4_gpu_tensor *kv_cache_view = metal_graph_tensor_row_view(metal_graph_batch_kv(g), t, DS4_N_HEAD_DIM);
                ds4_gpu_tensor *heads_view = metal_graph_tensor_row_view(metal_graph_batch_heads(g), t, q_dim);
                ok = ok && q_view && kv_cache_view && heads_view;
                if (ok && !zero_prefix) {
                    ok = ds4_gpu_store_raw_kv_tensor(g->layer_raw_cache[il],
                                                       kv_cache_view,
                                                       g->raw_cap,
                                                       pos % g->raw_cap,
                                                       DS4_N_HEAD_DIM) != 0;
                }
                if (ok && comp_mask != NULL && n_selected != 0) {
                    ok = ds4_gpu_attention_indexed_mixed_batch_heads_tensor(heads_view,
                                                                              model->map,
                                                                              model->size,
                                                                              layer->attn_sinks->abs_offset,
                                                                              q_view,
                                                                              g->layer_raw_cache[il],
                                                                              g->layer_attn_comp_cache[il],
                                                                              metal_graph_attn_comp_cache_is_f16(),
                                                                              metal_graph_comp_selected(g),
                                                                              1,
                                                                              pos,
                                                                              n_raw,
                                                                              g->raw_cap,
                                                                              raw_start,
                                                                              cur_comp,
                                                                              n_selected,
                                                                              g->raw_window,
                                                                              ratio,
                                                                              DS4_N_HEAD,
                                                                              DS4_N_HEAD_DIM) != 0;
                } else if (ok) {
                    ok = ds4_gpu_attention_decode_heads_tensor(heads_view,
                                                                 model->map,
                                                                 model->size,
                                                                 layer->attn_sinks->abs_offset,
                                                                 q_view,
                                                                 g->layer_raw_cache[il],
                                                                 n_raw,
                                                                 g->raw_cap,
                                                                 raw_start,
                                                                 cur_comp ? g->layer_attn_comp_cache[il] : NULL,
                                                                 metal_graph_attn_comp_cache_is_f16(),
                                                                 cur_comp,
                                                                 comp_mask,
                                                                 n_selected,
                                                                 DS4_N_HEAD,
                                                                 DS4_N_HEAD_DIM) != 0;
                }
                ds4_gpu_tensor_free(heads_view);
                ds4_gpu_tensor_free(kv_cache_view);
                ds4_gpu_tensor_free(q_view);
            }
        }
    }
    DS4_METAL_PROFILE_ATTN_STAGE("attention");

    if (ok) {
        metal_graph_debug_dump_tensor("kqv_out", metal_graph_batch_heads(g),
                                      (uint64_t)n_tokens * q_dim, il, pos0);
    }
    if (ok) ok = ds4_gpu_rope_tail_tensor(tp_heads ? tp_heads : metal_graph_batch_heads(g),
                                            tp_rows,
                                            DS4_N_HEAD,
                                            DS4_N_HEAD_DIM,
                                            DS4_N_ROT,
                                            pos0 + tp_row0,
                                            compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                                            true,
                                            freq_base,
                                            freq_scale,
                                            ext_factor,
                                            attn_factor,
                                            DS4_ROPE_YARN_BETA_FAST,
                                            DS4_ROPE_YARN_BETA_SLOW) != 0;
    if (ok) {
        metal_graph_debug_dump_tensor("kqv_back", metal_graph_batch_heads(g),
                                      (uint64_t)n_tokens * q_dim, il, pos0);
    }
    DS4_METAL_PROFILE_ATTN_STAGE("inv_rope");
    const bool attn_out_debug =
        metal_graph_debug_wants("attn_low", il, pos0) ||
        metal_graph_debug_wants("attn_out", il, pos0);
    bool attn_out_f16 = false;
    if (ok &&
        !attn_out_debug &&
        !tp_row_split_attn &&
        layer->attn_output_a->type == DS4_TENSOR_Q8_0 &&
        layer->attn_output_b->type == DS4_TENSOR_Q8_0 &&
        !metal_graph_directional_steering_attn_enabled(g)) {
        attn_out_f16 = ds4_gpu_attention_output_q8_batch_f16_tensor(g->batch_q_half,
                                                                    metal_graph_batch_attn_low(g),
                                                                    model->map,
                                                                    model->size,
                                                                    layer->attn_output_a->abs_offset,
                                                                    layer->attn_output_b->abs_offset,
                                                                    group_dim,
                                                                    rank,
                                                                    n_groups,
                                                                    DS4_N_EMBD,
                                                                    metal_graph_batch_heads(g),
                                                                    n_tokens) != 0;
    }
    uint64_t tp_attn_gate_seq = 0;
    /* Opt-in sub-chunk gate pipelining (see metal_graph_tp_subgate_pipeline;
     * measured net-negative on the M5 Max pair, kept for slower wires).
     * Kernel-path constraint: the output projection picks its kernel by row
     * count (direct low below 32 rows, the 64-token-tile TensorOps path at
     * multiples of 64, ids-cache fallback otherwise), and the paths are not
     * bit-identical per row.  Parity with the single-node reference
     * therefore requires every sub-call to land on the same path as the
     * full-chunk call: n_tokens % 256 == 0 puts the chunk, the rank halves,
     * and the quarter sub-calls all on the TensorOps path.  Other sizes
     * keep the proven single-gate swap. */
    const bool tp_attn_pipeline =
        tp_row_split_attn && (n_tokens % 256u) == 0u &&
        metal_graph_tp_subgate_pipeline();
    if (!attn_out_f16) {
        if (ok && tp_attn_pipeline) {
            /* Sub-chunk pipelined swap: the output projection runs in two
             * sub-halves of this rank's rows and each sub-half's row swap
             * is kicked as soon as its rows land in batch_attn_out, so the
             * first wire exchange overlaps the second sub-half's compute.
             * The two kicks use opposite flag-slot parities; the wait below
             * (before the HC post expand) covers both. */
            const uint32_t tp_sub1 = (tp_half_rows + 1u) / 2u;
            const uint32_t tp_c1 = tp_rows < tp_sub1 ? tp_rows : tp_sub1;
            const uint32_t tp_own_base = g->tp_rank == 0 ? 0u : tp_half_rows;
            const uint32_t tp_peer_base = g->tp_rank == 0 ? tp_half_rows : 0u;
            for (uint32_t sub = 0; ok && sub < 2u; sub++) {
                const uint32_t coff = sub == 0 ? 0u : tp_c1;
                const uint32_t crows = sub == 0 ? tp_c1 : tp_rows - tp_c1;
                const uint32_t soff = sub == 0 ? 0u : tp_sub1;
                const uint32_t srows = sub == 0 ? tp_sub1 : tp_half_rows - tp_sub1;
                if (crows != 0) {
                    ds4_gpu_tensor *sub_heads = metal_graph_tensor_row_range_view(
                            metal_graph_batch_heads(g), tp_row0 + coff, crows, q_dim);
                    ds4_gpu_tensor *sub_out = metal_graph_tensor_row_range_view(
                            metal_graph_batch_attn_out(g), tp_row0 + coff, crows, DS4_N_EMBD);
                    ok = sub_heads && sub_out &&
                         metal_graph_attention_output_dense_quant_batch(sub_out,
                                                                        metal_graph_batch_attn_low(g),
                                                                        g,
                                                                        model,
                                                                        layer->attn_output_a,
                                                                        layer->attn_output_b,
                                                                        group_dim,
                                                                        rank,
                                                                        n_groups,
                                                                        DS4_N_EMBD,
                                                                        sub_heads,
                                                                        crows);
                    ds4_gpu_tensor_free(sub_out);
                    ds4_gpu_tensor_free(sub_heads);
                }
                if (ok && srows != 0) {
                    ds4_gpu_tensor *send_sub = metal_graph_tensor_row_range_view(
                            metal_graph_batch_attn_out(g), tp_own_base + soff, srows, DS4_N_EMBD);
                    ds4_gpu_tensor *recv_sub = metal_graph_tensor_row_range_view(
                            metal_graph_batch_attn_out(g), tp_peer_base + soff, srows, DS4_N_EMBD);
                    uint64_t seq = 0;
                    if (send_sub && recv_sub) {
                        seq = ds4_gpu_tp_big_gate_kick(il, n_tokens,
                                                       send_sub, recv_sub,
                                                       (uint64_t)srows * DS4_N_EMBD * sizeof(float));
                    }
                    ok = seq != 0;
                    if (ok) tp_attn_gate_seq = seq;
                    ds4_gpu_tensor_free(recv_sub);
                    ds4_gpu_tensor_free(send_sub);
                }
            }
        } else if (ok) {
            ok = metal_graph_attention_output_dense_quant_batch(tp_attn_out ? tp_attn_out : metal_graph_batch_attn_out(g),
                                                                metal_graph_batch_attn_low(g),
                                                                g,
                                                                model,
                                                                layer->attn_output_a,
                                                                layer->attn_output_b,
                                                                group_dim,
                                                                rank,
                                                                n_groups,
                                                                DS4_N_EMBD,
                                                                tp_heads ? tp_heads : metal_graph_batch_heads(g),
                                                                tp_rows);
        }
        if (ok) {
            metal_graph_debug_dump_tensor("attn_low", metal_graph_batch_attn_low(g),
                                          (uint64_t)n_tokens * n_groups * rank,
                                          il,
                                          pos0);
        }
        if (ok) {
            metal_graph_debug_dump_tensor("attn_out", metal_graph_batch_attn_out(g),
                                          (uint64_t)n_tokens * DS4_N_EMBD, il, pos0);
        }
    }
    DS4_METAL_PROFILE_ATTN_STAGE("output_proj");
    if (ok && tp_row_split_attn) {
        /* Release point for the pipelined row swaps of batch_attn_out: both
         * ranks reach the HC post expand with identical full tensors. */
        if (tp_attn_pipeline) {
            ok = tp_attn_gate_seq != 0 &&
                 ds4_gpu_tp_big_gate_wait(tp_attn_gate_seq) != 0;
        } else {
            const uint64_t half_bytes =
                (uint64_t)tp_half_rows * DS4_N_EMBD * sizeof(float);
            ds4_gpu_tensor *send_half = metal_graph_tensor_row_range_view(
                    metal_graph_batch_attn_out(g),
                    g->tp_rank == 0 ? 0 : tp_half_rows, tp_half_rows, DS4_N_EMBD);
            ds4_gpu_tensor *recv_half = metal_graph_tensor_row_range_view(
                    metal_graph_batch_attn_out(g),
                    g->tp_rank == 0 ? tp_half_rows : 0, tp_half_rows, DS4_N_EMBD);
            ok = send_half && recv_half &&
                 ds4_gpu_tp_big_gate_encode(il, n_tokens,
                                            send_half, recv_half,
                                            half_bytes) != 0;
            ds4_gpu_tensor_free(send_half);
            ds4_gpu_tensor_free(recv_half);
        }
        if (!ok) fprintf(stderr, "ds4: TP prefill attention row gate failed (layer %u)\n", il);
    }
    if (ok && !attn_out_f16 && metal_graph_directional_steering_attn_enabled(g)) {
        ok = metal_graph_apply_directional_steering_attn(g, metal_graph_batch_attn_out(g), il, n_tokens);
    }
    if (ok && attn_out_f16) {
        ok = ds4_gpu_hc_expand_split_half_tensor(after_attn_hc_view,
                                                 g->batch_q_half,
                                                 metal_graph_batch_cur_hc(g),
                                                 hc_split_view,
                                                 DS4_N_EMBD,
                                                 DS4_N_HC) != 0;
    } else if (ok) {
        ok = ds4_gpu_hc_expand_split_tensor(after_attn_hc_view,
                                            metal_graph_batch_attn_out(g),
                                            metal_graph_batch_cur_hc(g),
                                            hc_split_view,
                                            DS4_N_EMBD,
                                            DS4_N_HC) != 0;
    }
    if (ok) {
        metal_graph_debug_dump_tensor("hc_attn_post", metal_graph_batch_after_attn_hc(g),
                                      (uint64_t)n_tokens * hc_dim, il, pos0);
    }
    DS4_METAL_PROFILE_ATTN_STAGE("hc_post");
    ds4_gpu_tensor_free(tp_attn_out);
    ds4_gpu_tensor_free(tp_heads);
    ds4_gpu_tensor_free(tp_qr_norm);
    ds4_gpu_tensor_free(tp_q_half);
    ds4_gpu_tensor_free(tp_q);
    ds4_gpu_tensor_free(after_attn_hc_view);
    ds4_gpu_tensor_free(attn_cur_view);
    ds4_gpu_tensor_free(hc_split_view);
    ds4_gpu_tensor_free(hc_mix_view);
    if (index_counts != index_counts_stack) free(index_counts);
    if (comp_counts != comp_counts_stack) free(comp_counts);
#undef DS4_METAL_PROFILE_ATTN_STAGE
#undef DS4_METAL_PROFILE_Q_STAGE
    return ok;
}

static bool metal_graph_encode_mixed_routed_rows(
        ds4_gpu_graph *g,
        ds4_decode_item *decode_items,
        int decode_count,
        const ds4_model *model,
        const ds4_layer_weights *layer,
        uint32_t il,
        uint32_t prefill_rows);

/* Encode the batched prefill FFN half: HC pre/norm, shared expert, routed
 * experts, sum, and HC post.  A non-empty decode tail has already been
 * prepared in rows [n_tokens, n_tokens + decode_count); only the routed
 * expert dispatch is shared between the two arithmetic paths. */
static bool metal_graph_encode_layer_ffn_batch(
        ds4_gpu_graph  *g,
        const ds4_model        *model,
        const ds4_layer_weights *layer,
        uint32_t                il,
        uint32_t                pos0,
        uint32_t                n_tokens,
        ds4_decode_item         *decode_items,
        int                     decode_count) {
    if (n_tokens == 0 || n_tokens > g->prefill_cap) return false;
    if (decode_count < 0 ||
        (decode_count > 0 &&
         (!decode_items || (uint64_t)n_tokens + (uint32_t)decode_count > g->prefill_cap))) {
        return false;
    }

    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const uint64_t mix_hc = 2ull * DS4_N_HC + (uint64_t)DS4_N_HC * DS4_N_HC;
    const uint64_t shared_dim = layer->ffn_gate_shexp->dim[1];
    const uint64_t expert_in_dim = layer->ffn_gate_exps->dim[0];
    const uint64_t expert_mid_dim = layer->ffn_gate_exps->dim[1];
    const uint64_t down_in_dim = layer->ffn_down_exps->dim[0];
    const uint64_t routed_out_dim = layer->ffn_down_exps->dim[1];
    const uint64_t gate_row_bytes = routed_expert_row_bytes(layer->ffn_gate_exps);
    const uint64_t gate_expert_bytes = expert_mid_dim * gate_row_bytes;
    const uint64_t down_row_bytes = routed_expert_row_bytes(layer->ffn_down_exps);
    const uint64_t down_expert_bytes = routed_out_dim * down_row_bytes;
    const bool layer_stage_profile = metal_graph_layer_stage_profile_enabled(il);
    double layer_stage_t0 = layer_stage_profile ? now_sec() : 0.0;
#define DS4_METAL_PROFILE_FFN_STAGE(name) do { \
        if (ok && layer_stage_profile) { \
            ok = metal_graph_layer_stage_profile_boundary("ffn", (name), il, pos0, n_tokens, &layer_stage_t0); \
        } \
    } while (0)

    ds4_gpu_tensor *hc_mix_view = ds4_gpu_tensor_view(
            metal_graph_batch_hc_mix(g), 0, (uint64_t)n_tokens * mix_hc * sizeof(float));
    ds4_gpu_tensor *hc_split_view = ds4_gpu_tensor_view(
            metal_graph_batch_hc_split(g), 0, (uint64_t)n_tokens * mix_hc * sizeof(float));
    ds4_gpu_tensor *ffn_cur_view = ds4_gpu_tensor_view(
            metal_graph_batch_ffn_cur(g), 0, (uint64_t)n_tokens * DS4_N_EMBD * sizeof(float));
    ds4_gpu_tensor *next_hc_view = ds4_gpu_tensor_view(
            metal_graph_batch_next_hc(g), 0, (uint64_t)n_tokens * hc_dim * sizeof(float));
    bool ok = hc_mix_view && hc_split_view && ffn_cur_view && next_hc_view;
    const bool fuse_hc_norm = n_tokens > 1 &&
                              DS4_N_HC == 4 &&
                              !metal_graph_use_reference_hc_decode() &&
                              metal_graph_enable_batch_hc_norm_fusion();
    if (ok) ok = metal_graph_hc_rms_scale_project(hc_mix_view,
                                                    metal_graph_batch_flat_hc(g),
                                                    model,
                                                    layer->hc_ffn_fn,
                                                    metal_graph_batch_after_attn_hc(g),
                                                    hc_dim,
                                                    n_tokens);
    if (metal_graph_use_reference_hc_decode()) {
        if (ok) ok = ds4_gpu_hc_split_sinkhorn_tensor(hc_split_view,
                                                        hc_mix_view,
                                                        model->map,
                                                        model->size,
                                                        layer->hc_ffn_scale->abs_offset,
                                                        layer->hc_ffn_base->abs_offset,
                                                        DS4_N_HC,
                                                        DS4_N_HC_SINKHORN_ITER,
                                                        DS4_HC_EPS) != 0;
        if (ok) ok = ds4_gpu_hc_weighted_sum_split_tensor(ffn_cur_view,
                                                            metal_graph_batch_after_attn_hc(g),
                                                            hc_split_view,
                                                            DS4_N_EMBD,
                                                            DS4_N_HC) != 0;
    } else if (fuse_hc_norm) {
        if (ok) ok = ds4_gpu_hc_split_weighted_sum_norm_tensor(ffn_cur_view,
                                                                 metal_graph_batch_ffn_norm(g),
                                                                 hc_split_view,
                                                                 hc_mix_view,
                                                                 metal_graph_batch_after_attn_hc(g),
                                                                 model->map,
                                                                 model->size,
                                                                 layer->hc_ffn_scale->abs_offset,
                                                                 layer->hc_ffn_base->abs_offset,
                                                                 layer->ffn_norm->abs_offset,
                                                                 DS4_N_EMBD,
                                                                 DS4_N_HC,
                                                                 DS4_N_HC_SINKHORN_ITER,
                                                                 DS4_HC_EPS,
                                                                 DS4_RMS_EPS) != 0;
    } else {
        if (ok) ok = ds4_gpu_hc_split_weighted_sum_tensor(ffn_cur_view,
                                                            hc_split_view,
                                                            hc_mix_view,
                                                            metal_graph_batch_after_attn_hc(g),
                                                            model->map,
                                                            model->size,
                                                            layer->hc_ffn_scale->abs_offset,
                                                            layer->hc_ffn_base->abs_offset,
                                                            DS4_N_EMBD,
                                                            DS4_N_HC,
                                                            DS4_N_HC_SINKHORN_ITER,
                                                            DS4_HC_EPS) != 0;
    }
    if (ok) {
        metal_graph_debug_dump_tensor("hc_ffn_pre", metal_graph_batch_ffn_cur(g),
                                      (uint64_t)n_tokens * DS4_N_EMBD, il, pos0);
    }
    DS4_METAL_PROFILE_FFN_STAGE("hc_pre");
    if (ok && !fuse_hc_norm) {
        ok = ds4_gpu_rms_norm_weight_rows_tensor(metal_graph_batch_ffn_norm(g),
                                                  metal_graph_batch_ffn_cur(g),
                                                  model->map,
                                                  model->size,
                                                  layer->ffn_norm->abs_offset,
                                                  DS4_N_EMBD,
                                                  n_tokens,
                                                  DS4_RMS_EPS) != 0;
    }
    if (ok) {
        metal_graph_debug_dump_tensor("ffn_norm", metal_graph_batch_ffn_norm(g),
                                      (uint64_t)n_tokens * DS4_N_EMBD, il, pos0);
    }
    DS4_METAL_PROFILE_FFN_STAGE("norm");
    if (ok) ok = metal_graph_matmul_plain_tensor(metal_graph_batch_router_logits(g),
                                                 model,
                                                 layer->ffn_gate_inp,
                                                 DS4_N_EMBD,
                                                 DS4_N_EXPERT,
                                                 metal_graph_batch_ffn_norm(g),
                                                 n_tokens);

    ds4_gpu_tensor *router_tokens = NULL;
    if (ok) {
        router_tokens = ds4_gpu_tensor_view(metal_graph_prefill_tokens(g),
                                              (uint64_t)g->batch_token_offset * sizeof(int32_t),
                                              (uint64_t)n_tokens * sizeof(int32_t));
        ok = router_tokens != NULL;
    }
    if (ok) ok = ds4_gpu_router_select_batch_tensor(metal_graph_batch_router_selected(g),
                                                      metal_graph_batch_router_weights(g),
                                                      metal_graph_batch_router_probs(g),
                                                      model->map,
                                                      model->size,
                                                      layer->ffn_exp_probs_b ? layer->ffn_exp_probs_b->abs_offset : 0,
                                                      layer->ffn_gate_tid2eid ? layer->ffn_gate_tid2eid->abs_offset : 0,
                                                      layer->ffn_gate_tid2eid ? (uint32_t)layer->ffn_gate_tid2eid->dim[1] : 0,
                                                      0,
                                                      0,
                                                      layer->ffn_exp_probs_b != NULL,
                                                      layer->ffn_gate_tid2eid != NULL,
                                                      metal_graph_batch_router_logits(g),
                                                      metal_graph_prefill_tokens(g),
                                                      DS4_N_EXPERT,
                                                      DS4_N_EXPERT_USED,
                                                      DS4_EXPERT_WEIGHT_SCALE,
                                                      n_tokens) != 0;
    ds4_gpu_tensor_free(router_tokens);
    if (ok) {
        metal_graph_debug_dump_tensor("ffn_moe_logits", metal_graph_batch_router_logits(g),
                                      (uint64_t)n_tokens * DS4_N_EXPERT, il, pos0);
        metal_graph_debug_dump_tensor("ffn_moe_probs", metal_graph_batch_router_probs(g),
                                      (uint64_t)n_tokens * DS4_N_EXPERT, il, pos0);
        metal_graph_debug_dump_i32_tensor("ffn_moe_topk", metal_graph_batch_router_selected(g),
                                          (uint64_t)n_tokens * DS4_N_EXPERT_USED, il, pos0);
        metal_graph_debug_dump_tensor("ffn_moe_weights_scaled", metal_graph_batch_router_weights(g),
                                      (uint64_t)n_tokens * DS4_N_EXPERT_USED, il, pos0);
    }
    DS4_METAL_PROFILE_FFN_STAGE("router");

    if (ok) {
        ok = metal_graph_cuda_stream_prefill_batch_selected_load(g,
                                                                 model,
                                                                 layer,
                                                                 il,
                                                                 n_tokens,
                                                                 gate_expert_bytes,
                                                                 down_expert_bytes);
    }

#ifdef DS4_ROCM_BUILD
    rocm_graph_batch_selected_async_load rocm_batch_selected_async = {0};
    bool rocm_batch_selected_async_started = false;
    const bool rocm_batch_selected_shared_overlap =
        ok &&
        g->ssd_streaming &&
        !g->quality &&
        n_tokens > 1 &&
        DS4_N_EXPERT_USED == 6 &&
        !rocm_graph_stream_prefill_full_layer_enabled(g, layer, il, n_tokens) &&
        layer->ffn_gate_exps->type == DS4_TENSOR_IQ2_XXS &&
        layer->ffn_up_exps->type == DS4_TENSOR_IQ2_XXS &&
        layer->ffn_down_exps->type == DS4_TENSOR_Q2_K;
    if (rocm_batch_selected_shared_overlap) {
        uint64_t selected_event = 0;
        if (ds4_gpu_signal_selected_readback_ready(&selected_event) == 0) {
            ok = false;
        } else {
            ok = rocm_graph_batch_selected_async_load_start(
                    &rocm_batch_selected_async,
                    metal_graph_batch_router_selected(g),
                    model,
                    layer,
                    il,
                    n_tokens,
                    selected_event,
                    gate_expert_bytes,
                    down_expert_bytes);
            rocm_batch_selected_async_started = ok;
        }
    }
#endif

    const bool selected_readahead_shared =
        metal_graph_stream_prefill_selected_readahead_shared_enabled(g)
#ifdef DS4_ROCM_BUILD
        && !rocm_batch_selected_async_started
#endif
        ;
    if (ok &&
        metal_graph_stream_prefill_selected_readahead_enabled(g) &&
#ifdef DS4_ROCM_BUILD
        !rocm_batch_selected_async_started &&
#endif
        !selected_readahead_shared) {
        if (ds4_gpu_end_commands() == 0) {
            ok = false;
        } else {
            ok = metal_graph_stream_readahead_selected_experts_from_gpu(g,
                                                                        model,
                                                                        layer,
                                                                        il,
                                                                        n_tokens,
                                                                        gate_expert_bytes,
                                                                        down_expert_bytes) &&
                 ds4_gpu_begin_commands() != 0;
        }
    }

    const bool keep_ffn_out = metal_graph_needs_ffn_out(g, il, pos0);
    bool shared_down_f16 = false;

#define DS4_METAL_TRY_SHARED_DOWN_F16() do { \
        if (ok && !tp_row_split_ffn && !keep_ffn_out && \
            !metal_graph_debug_wants("ffn_shexp", il, pos0)) { \
            shared_down_f16 = ds4_gpu_matmul_q8_0_f16_out_tensor(g->batch_q_half, \
                                                                 model->map, \
                                                                 model->size, \
                                                                 layer->ffn_down_shexp->abs_offset, \
                                                                 shared_dim, \
                                                                 DS4_N_EMBD, \
                                                                 metal_graph_batch_shared_mid(g), \
                                                                 n_tokens) != 0; \
        } \
    } while (0)

#define DS4_METAL_ENCODE_PREFILL_SHARED_EXPERT() do { \
        if (ok) ok = metal_graph_matmul_q8_0_named_tensor("shared_gate", \
                                                          il, \
                                                          pos0, \
                                                          metal_graph_batch_shared_gate(g), \
                                                          model, \
                                                          layer->ffn_gate_shexp, \
                                                          DS4_N_EMBD, \
                                                          shared_dim, \
                                                          tp_ffn_x ? tp_ffn_x : metal_graph_batch_ffn_norm(g), \
                                                          tp_rows); \
        if (ok) ok = metal_graph_matmul_q8_0_named_tensor("shared_up", \
                                                          il, \
                                                          pos0, \
                                                          metal_graph_batch_shared_up(g), \
                                                          model, \
                                                          layer->ffn_up_shexp, \
                                                          DS4_N_EMBD, \
                                                          shared_dim, \
                                                          tp_ffn_x ? tp_ffn_x : metal_graph_batch_ffn_norm(g), \
                                                          tp_rows); \
        DS4_METAL_PROFILE_FFN_STAGE("shared_gate_up"); \
        if (ok) ok = ds4_gpu_swiglu_tensor(metal_graph_batch_shared_mid(g), \
                                             metal_graph_batch_shared_gate(g), \
                                             metal_graph_batch_shared_up(g), \
                                             (uint32_t)((uint64_t)tp_rows * shared_dim), \
                                             DS4_SWIGLU_CLAMP_EXP, \
                                             1.0f) != 0; \
        DS4_METAL_TRY_SHARED_DOWN_F16(); \
        if (ok && !shared_down_f16) ok = metal_graph_matmul_q8_0_named_tensor("shared_down", \
                                                                              il, \
                                                                              pos0, \
                                                                              metal_graph_batch_shared_out(g), \
                                                                              model, \
                                                                              layer->ffn_down_shexp, \
                                                                              shared_dim, \
                                                                              DS4_N_EMBD, \
                                                                              metal_graph_batch_shared_mid(g), \
                                                                              tp_rows); \
        DS4_METAL_PROFILE_FFN_STAGE("shared_down"); \
        if (ok && !shared_down_f16) { \
            metal_graph_debug_dump_tensor("ffn_shexp", metal_graph_batch_shared_out(g), \
                                          (uint64_t)n_tokens * DS4_N_EMBD, il, pos0); \
        } \
    } while (0)

    bool shared_done = false;
    /* With 50/50 expert residency, every rank evaluates every prompt row
     * against its local expert half.  For large chunks the replicated shared
     * expert remains row-split; its rows are folded into the local routed
     * partial before the one all-reduce-style bulk exchange. */
    const bool tp_split_ffn = g->tp_world == 2;
    const bool tp_row_split_ffn =
        tp_split_ffn && g->tp_batch_rows != n_tokens && !keep_ffn_out &&
        !metal_graph_directional_steering_ffn_enabled(g) &&
        n_tokens >= metal_graph_tp_prefill_split_min();
    const uint32_t tp_half_rows = (n_tokens + 1u) / 2u;
    const uint32_t tp_row0 = (tp_row_split_ffn && g->tp_rank != 0) ? tp_half_rows : 0;
    const uint32_t tp_rows = tp_row_split_ffn ?
        (g->tp_rank == 0 ? tp_half_rows : n_tokens - tp_half_rows) : n_tokens;
    ds4_gpu_tensor *tp_ffn_x = tp_row_split_ffn ?
        metal_graph_tensor_row_range_view(metal_graph_batch_ffn_norm(g), tp_row0, tp_rows,
                                          DS4_N_EMBD) : NULL;
    if (tp_row_split_ffn && !tp_ffn_x) ok = false;
    if (ok && selected_readahead_shared) {
        if (ds4_gpu_end_commands() == 0) {
            ok = false;
        } else {
            ok = metal_graph_stream_readahead_selected_experts_from_gpu(g,
                                                                        model,
                                                                        layer,
                                                                        il,
                                                                        n_tokens,
                                                                        gate_expert_bytes,
                                                                        down_expert_bytes) &&
                 ds4_gpu_begin_commands() != 0;
        }
        if (ok) {
            DS4_METAL_ENCODE_PREFILL_SHARED_EXPERT();
            shared_done = ok;
        }
        if (ok) {
            if (ds4_gpu_end_commands() == 0) {
                ok = false;
            } else {
                ok = ds4_gpu_begin_commands() != 0;
            }
        }
    }

    if (ok &&
        !shared_done &&
        (metal_graph_stream_prefill_selected_pagein_enabled(g) ||
         metal_graph_stream_prefill_selected_madvise_enabled(g))) {
        metal_graph_stream_pagein_job pagein_job;
        memset(&pagein_job, 0, sizeof(pagein_job));
        bool pagein_commands_open = false;
        if (ds4_gpu_end_commands() == 0) {
            ok = false;
        } else {
            ok = metal_graph_stream_prefill_selected_pagein_start(g,
                                                                  model,
                                                                  layer,
                                                                  il,
                                                                  n_tokens,
                                                                  gate_expert_bytes,
                                                                  down_expert_bytes,
                                                                  &pagein_job);
        }
        if (ok) {
            if (ds4_gpu_begin_commands() == 0) {
                ok = false;
            } else {
                pagein_commands_open = true;
            }
        }
        if (ok) {
            DS4_METAL_ENCODE_PREFILL_SHARED_EXPERT();
            shared_done = ok;
        }
        if (pagein_commands_open) {
            if (ds4_gpu_end_commands() == 0) ok = false;
        }
        if (!metal_graph_stream_prefill_selected_pagein_join(&pagein_job)) {
            ok = false;
        }
        if (ok) ok = ds4_gpu_begin_commands() != 0;
    }

#ifdef DS4_ROCM_BUILD
    if (rocm_batch_selected_async_started) {
        if (ok && !shared_done) {
            DS4_METAL_ENCODE_PREFILL_SHARED_EXPERT();
            shared_done = ok;
        }
        const bool finish_ok =
            rocm_graph_batch_selected_async_load_finish(&rocm_batch_selected_async);
        ok = ok && finish_ok;
    }
#endif

    const bool tp_split_batch_moe =
        g->tp_batch_rows == n_tokens && n_tokens > 0 &&
        g->tp_world == 2 &&
        g->tp_batch_out && g->tp_batch_in;
    const bool cuda_tp_owned_batch_moe =
        g->cuda_tp_ep && g->cuda_tp_prefill_ffn;
    if (ok && cuda_tp_owned_batch_moe) {
        ok = metal_graph_encode_mixed_routed_rows(
                g, decode_items, decode_count, model, layer, il, n_tokens);
    } else if (ok && tp_split_batch_moe) {
        /* Verify-block expert split: run the contiguous-half split
         * single-token routed kernels per row into the slab batch-out
         * rows, exchange all rows with one gate, then materialize the
         * combined routed output.  The add is commutative, so both ranks
         * compute bit-identical sums and stay in lockstep. */
        const uint64_t vec_bytes = (uint64_t)DS4_N_EMBD * sizeof(float);
        for (uint32_t r = 0; ok && r < n_tokens; r++) {
            ds4_gpu_tensor *out_row = ds4_gpu_tensor_view(
                    g->tp_batch_out[il], (uint64_t)r * vec_bytes, vec_bytes);
            ds4_gpu_tensor *x_row = ds4_gpu_tensor_view(
                    metal_graph_batch_ffn_norm(g), (uint64_t)r * vec_bytes, vec_bytes);
            ds4_gpu_tensor *sel_row = ds4_gpu_tensor_view(
                    metal_graph_batch_router_selected(g),
                    (uint64_t)r * DS4_N_EXPERT_USED * sizeof(int32_t),
                    (uint64_t)DS4_N_EXPERT_USED * sizeof(int32_t));
            ds4_gpu_tensor *w_row = ds4_gpu_tensor_view(
                    metal_graph_batch_router_weights(g),
                    (uint64_t)r * DS4_N_EXPERT_USED * sizeof(float),
                    (uint64_t)DS4_N_EXPERT_USED * sizeof(float));
            ok = out_row && x_row && sel_row && w_row &&
                 ds4_gpu_routed_moe_one_tensor(out_row,
                                               metal_graph_routed_gate(g),
                                               metal_graph_routed_up(g),
                                               metal_graph_routed_mid(g),
                                               metal_graph_routed_down(g),
                                               model->map, model->size,
                                               layer->ffn_gate_exps->abs_offset,
                                               layer->ffn_up_exps->abs_offset,
                                               layer->ffn_down_exps->abs_offset,
                                               layer->ffn_gate_exps->type,
                                               layer->ffn_down_exps->type,
                                               gate_expert_bytes,
                                               gate_row_bytes,
                                               down_expert_bytes,
                                               down_row_bytes,
                                               (uint32_t)expert_in_dim,
                                               (uint32_t)down_in_dim,
                                               (uint32_t)routed_out_dim,
                                               sel_row, w_row,
                                               DS4_N_EXPERT,
                                               DS4_N_EXPERT_USED,
                                               DS4_SWIGLU_CLAMP_EXP,
                                               x_row,
                                               NULL,
                                               il,
                                               false) != 0;
            ds4_gpu_tensor_free(w_row);
            ds4_gpu_tensor_free(sel_row);
            ds4_gpu_tensor_free(x_row);
            ds4_gpu_tensor_free(out_row);
        }
        if (ok) ok = ds4_gpu_tp_batch_gate_encode(il, n_tokens) != 0;
        if (ok) {
            ok = ds4_gpu_add_tensor(metal_graph_batch_routed_out(g),
                                    g->tp_batch_out[il],
                                    g->tp_batch_in[il],
                                    (uint32_t)((uint64_t)n_tokens * DS4_N_EMBD)) != 0;
        }
    } else if (ok) {
        ok = ds4_gpu_routed_moe_batch_tensor(metal_graph_batch_routed_out(g),
                                               metal_graph_batch_routed_gate(g),
                                               metal_graph_batch_routed_up(g),
                                               metal_graph_batch_routed_mid(g),
                                               metal_graph_batch_routed_down(g),
                                               model->map,
                                               model->size,
                                               layer->ffn_gate_exps->abs_offset,
                                               layer->ffn_up_exps->abs_offset,
                                               layer->ffn_down_exps->abs_offset,
                                               layer->ffn_gate_exps->type,
                                               layer->ffn_down_exps->type,
                                               gate_expert_bytes,
                                               gate_row_bytes,
                                               down_expert_bytes,
                                               down_row_bytes,
                                               (uint32_t)expert_in_dim,
                                               (uint32_t)down_in_dim,
                                               (uint32_t)routed_out_dim,
                                               metal_graph_batch_router_selected(g),
                                               metal_graph_batch_router_weights(g),
                                               DS4_N_EXPERT,
                                               DS4_N_EXPERT_USED,
                                               DS4_SWIGLU_CLAMP_EXP,
                                               metal_graph_batch_ffn_norm(g),
                                               il,
                                               n_tokens,
                                               &g->batch_routed_mid_is_f16,
                                               false) != 0;
    }
    if (ok) {
        metal_graph_debug_dump_tensor("ffn_moe_gate_clamped", metal_graph_batch_routed_gate(g),
                                      (uint64_t)n_tokens * DS4_N_EXPERT_USED * down_in_dim, il, pos0);
        metal_graph_debug_dump_tensor("ffn_moe_up_clamped", metal_graph_batch_routed_up(g),
                                      (uint64_t)n_tokens * DS4_N_EXPERT_USED * down_in_dim, il, pos0);
    }
    if (ok) {
        const uint64_t routed_mid_elems = (uint64_t)n_tokens * DS4_N_EXPERT_USED * down_in_dim;
        if (g->batch_routed_mid_is_f16) {
            metal_graph_debug_dump_f16_tensor("ffn_moe_weighted_swiglu", metal_graph_batch_routed_mid(g),
                                              routed_mid_elems, il, pos0);
        } else {
            metal_graph_debug_dump_tensor("ffn_moe_weighted_swiglu", metal_graph_batch_routed_mid(g),
                                          routed_mid_elems, il, pos0);
        }
    }
    if (ok) {
        metal_graph_debug_dump_tensor("ffn_moe_down", metal_graph_batch_routed_down(g),
                                      (uint64_t)n_tokens * DS4_N_EXPERT_USED * DS4_N_EMBD, il, pos0);
    }
    if (ok) {
        metal_graph_debug_dump_tensor("ffn_moe_out", metal_graph_batch_routed_out(g),
                                      (uint64_t)n_tokens * DS4_N_EMBD, il, pos0);
    }
    DS4_METAL_PROFILE_FFN_STAGE("routed_moe");
    if (!shared_done) {
        DS4_METAL_ENCODE_PREFILL_SHARED_EXPERT();
    }
#undef DS4_METAL_ENCODE_PREFILL_SHARED_EXPERT
#undef DS4_METAL_TRY_SHARED_DOWN_F16

    if (ok && tp_row_split_ffn) {
        /* Each shared-expert row must appear exactly once in the all-reduce.
         * Fold this rank's shared rows into its full-row routed partial; the
         * peer does the same for the complementary rows. */
        ds4_gpu_tensor *own_rows =
            metal_graph_tensor_row_range_view(metal_graph_batch_routed_out(g), tp_row0,
                                              tp_rows, DS4_N_EMBD);
        ok = own_rows &&
             ds4_gpu_add_tensor(own_rows, own_rows, metal_graph_batch_shared_out(g),
                                (uint32_t)((uint64_t)tp_rows * DS4_N_EMBD)) != 0;
        ds4_gpu_tensor_free(own_rows);
    }

    if (ok && tp_split_ffn && !tp_split_batch_moe) {
        /* All rows contain this rank's routed-expert partial.  Exchange that
         * matrix in one bulk gate, then add in canonical rank order.  The
         * batch verify path above already performed the equivalent slab gate. */
        const uint64_t bytes =
            (uint64_t)n_tokens * DS4_N_EMBD * sizeof(float);
        ok = metal_graph_ensure_batch_ffn_out(g) &&
             ds4_gpu_tp_big_gate_encode(il, n_tokens,
                                        metal_graph_batch_routed_out(g),
                                        metal_graph_batch_ffn_out(g),
                                        bytes) != 0;
        if (ok) {
            ds4_gpu_tensor *first = g->tp_rank == 0 ?
                metal_graph_batch_routed_out(g) : metal_graph_batch_ffn_out(g);
            ds4_gpu_tensor *second = g->tp_rank == 0 ?
                metal_graph_batch_ffn_out(g) : metal_graph_batch_routed_out(g);
            ok = ds4_gpu_add_tensor(metal_graph_batch_routed_out(g), first, second,
                                    (uint32_t)((uint64_t)n_tokens * DS4_N_EMBD)) != 0;
        }
        if (!ok) {
            fprintf(stderr, "ds4: TP prefill FFN all-reduce failed (layer %u)\n", il);
        }
    }

    if (ok && keep_ffn_out) {
        ok = metal_graph_ensure_batch_ffn_out(g) &&
             ds4_gpu_add_tensor(metal_graph_batch_ffn_out(g),
                                  metal_graph_batch_shared_out(g),
                                  metal_graph_batch_routed_out(g),
                                  (uint32_t)((uint64_t)n_tokens * DS4_N_EMBD)) != 0;
    }
    if (ok && keep_ffn_out) {
        metal_graph_debug_dump_tensor("ffn_out", metal_graph_batch_ffn_out(g),
                                      (uint64_t)n_tokens * DS4_N_EMBD, il, pos0);
    }
    if (ok && metal_graph_directional_steering_ffn_enabled(g)) {
        ok = metal_graph_apply_directional_steering_ffn(g, metal_graph_batch_ffn_out(g), il, n_tokens);
    }
    if (ok && metal_graph_directional_steering_ffn_enabled(g)) {
        ok = ds4_gpu_hc_expand_split_tensor(next_hc_view,
                                              metal_graph_batch_ffn_out(g),
                                              metal_graph_batch_after_attn_hc(g),
                                              hc_split_view,
                                              DS4_N_EMBD,
                                              DS4_N_HC) != 0;
    }
    else if (ok && shared_down_f16) {
        ok = ds4_gpu_hc_expand_add_split_half_add_tensor(next_hc_view,
                                                         metal_graph_batch_routed_out(g),
                                                         g->batch_q_half,
                                                         metal_graph_batch_after_attn_hc(g),
                                                         hc_split_view,
                                                         DS4_N_EMBD,
                                                         DS4_N_HC) != 0;
    }
    else if (ok && tp_row_split_ffn) {
        /* Shared expert already folded into the exchanged routed rows. */
        ok = ds4_gpu_hc_expand_split_tensor(next_hc_view,
                                              metal_graph_batch_routed_out(g),
                                              metal_graph_batch_after_attn_hc(g),
                                              hc_split_view,
                                              DS4_N_EMBD,
                                              DS4_N_HC) != 0;
    }
    else if (ok) {
        ok = ds4_gpu_hc_expand_add_split_tensor(next_hc_view,
                                                  metal_graph_batch_routed_out(g),
                                                  metal_graph_batch_shared_out(g),
                                                  metal_graph_batch_after_attn_hc(g),
                                                  hc_split_view,
                                                  DS4_N_EMBD,
                                                  DS4_N_HC) != 0;
    }
    DS4_METAL_PROFILE_FFN_STAGE("hc_post");
    if (ok) {
        metal_graph_debug_dump_tensor("hc_ffn_post", metal_graph_batch_next_hc(g),
                                      (uint64_t)n_tokens * hc_dim, il, pos0);
    }
    DS4_METAL_PROFILE_FFN_STAGE("hc_post");
    ds4_gpu_tensor_free(tp_ffn_x);
    ds4_gpu_tensor_free(next_hc_view);
    ds4_gpu_tensor_free(ffn_cur_view);
    ds4_gpu_tensor_free(hc_split_view);
    ds4_gpu_tensor_free(hc_mix_view);
#undef DS4_METAL_PROFILE_FFN_STAGE
    return ok;
}

/* Encode one complete layer for prefill by chaining attention and FFN batches. */
static bool metal_graph_encode_layer_batch(
        ds4_gpu_graph  *g,
        const ds4_model        *model,
        const ds4_layer_weights *layer,
        uint32_t                il,
        uint32_t                pos0,
        uint32_t                n_tokens) {
    if (g->placement) {
        const int this_tier = g->placement[il + 1u];
        if (!metal_graph_set_active_tier_batch(g, this_tier, n_tokens)) {
            return false;
        }
    }
    bool ok = metal_graph_layer_stage_profile_start(il);
    if (ok) {
        ok = metal_graph_encode_layer_attention_batch(g, model, layer, il, pos0, n_tokens);
    }
    if (!ok) {
        fprintf(stderr, "ds4: gpu layer %u attention batch encode failed\n", il);
    }
    if (ok) {
        ok = metal_graph_encode_layer_ffn_batch(g, model, layer, il, pos0,
                                                 n_tokens, NULL, 0);
        if (!ok) {
            fprintf(stderr, "ds4: gpu layer %u ffn batch encode failed\n", il);
        }
    }
    if (ok) {
        ds4_gpu_tensor *tmp = metal_graph_batch_cur_hc(g);
        g->batch_cur_hc_by_tier[g->active_tier] = metal_graph_batch_next_hc(g);
        g->batch_next_hc_by_tier[g->active_tier] = tmp;
    }
    return ok;
}

static bool metal_graph_eval_token_raw_swa_streaming(
        ds4_gpu_graph *g,
        const ds4_model       *model,
        const ds4_weights     *weights,
        int                    token,
        uint32_t               pos,
        float                 *logits) {
    if (g->raw_cap == 0) {
        fprintf(stderr, "ds4: Metal graph raw KV cache is not allocated\n");
        return false;
    }

    const bool profile =
        glm_graph_env_present("DS4_ROCM_GRAPH_TOKEN_PROFILE",
                              "DS4_METAL_GRAPH_TOKEN_PROFILE");
    const bool throttle = graph_power_throttle_enabled(g);
    const double t0 = (profile || throttle) ? now_sec() : 0.0;
    const uint32_t raw_row = pos % g->raw_cap;
    const uint32_t n_raw = metal_graph_raw_span_for_batch(g, pos, 1);
    metal_graph_dspark_capture_begin(g);

    const bool static_decode_map = metal_graph_stream_decode_static_map_enabled();
    const bool static_map_state_cache =
        static_decode_map && metal_graph_stream_decode_static_map_state_cache_enabled();
    const bool batch_static_decode =
        static_decode_map && metal_graph_stream_decode_layer_batch_enabled(g);
    bool ok = true;
    if (static_decode_map) {
        if (!static_map_state_cache || !g->streaming_static_decode_map_current) {
            ok = metal_graph_stream_map_decode_static_all(model, weights);
            if (ok) g->streaming_static_decode_map_current = static_map_state_cache;
        }
    } else {
        g->streaming_static_decode_map_current = false;
        ok = metal_graph_stream_map_token(model, weights);
    }
    if (ok && !static_decode_map && DS4_N_LAYER > 0) {
        metal_graph_stream_readahead_layer_decode(model, weights, 0);
    }
    if (ok) ok = ds4_gpu_begin_commands() != 0;
    if (ok) {
        ok = ds4_gpu_embed_token_hc_tensor(metal_graph_cur_hc(g),
                                           model->map,
                                           model->size,
                                           weights->token_embd->abs_offset,
                                           (uint32_t)weights->token_embd->dim[1],
                                           (uint32_t)token,
                                           DS4_N_EMBD,
                                           DS4_N_HC) != 0;
    }
    if (batch_static_decode) {
        for (uint32_t il = 0; ok && il < DS4_N_LAYER; il++) {
            ok = metal_graph_encode_decode_layer(g,
                                                 model,
                                                 &weights->layer[il],
                                                 il,
                                                 pos,
                                                 g->layer_raw_cache[il],
                                                 g->raw_cap,
                                                 raw_row,
                                                 n_raw,
                                                 token);
            if (ok) {
                ds4_gpu_tensor *tmp = metal_graph_cur_hc(g);
                g->cur_hc_by_tier[g->active_tier] = metal_graph_after_ffn_hc(g);
                g->after_ffn_hc_by_tier[g->active_tier] = tmp;
                ok = metal_graph_dspark_capture_decode_layer(g, il);
            }
        }
        if (ok && logits) {
            ok = metal_graph_encode_output_head(g, model, weights, weights->output->dim[1]);
        }
        const double t_encoded = (profile || throttle) ? now_sec() : 0.0;
        if (ok) ok = ds4_gpu_end_commands() != 0;
        const double t_done = (profile || throttle) ? now_sec() : 0.0;
        if (ok && logits) {
            ok = ds4_gpu_tensor_read(metal_graph_logits(g), 0, logits, (uint64_t)DS4_N_VOCAB * sizeof(float)) != 0;
        }
        const double t_read = (profile || throttle) ? now_sec() : 0.0;
        if (profile) {
            fprintf(stderr,
                    "ds4: metal SSD streaming batched token pos=%u encode=%.3f ms execute=%.3f ms read=%.3f ms total=%.3f ms logits=%d\n",
                    pos,
                    (t_encoded - t0) * 1000.0,
                    (t_done - t_encoded) * 1000.0,
                    (t_read - t_done) * 1000.0,
                    (t_read - t0) * 1000.0,
                    logits != NULL);
        }
        if (ok && throttle) {
            graph_power_note_decode_token(g, t_read - t0);
        }
        if (!ok) {
            if (ds4_gpu_synchronize() == 0) {
                fprintf(stderr, "ds4: Metal synchronize after batched SSD streaming graph eval failure also failed\n");
            }
        }
        return ok;
    }
    if (ok) ok = ds4_gpu_end_commands() != 0;

    double encode_s = 0.0;
    double execute_s = 0.0;
    for (uint32_t il = 0; ok && il < DS4_N_LAYER; il++) {
        const double tl0 = profile ? now_sec() : 0.0;
        if (!static_decode_map && !metal_graph_stream_map_layer_decode(model, weights, il)) {
            ok = false;
            break;
        }
        if (!static_decode_map && il + 1 < DS4_N_LAYER) {
            metal_graph_stream_readahead_layer_decode(model, weights, il + 1);
        } else if (!static_decode_map && logits) {
            metal_graph_stream_readahead_output(model, weights);
        }
        if (ok) ok = ds4_gpu_begin_commands() != 0;
        bool encoded_layer = false;
        if (ok) {
            ok = metal_graph_encode_decode_layer(g,
                                                 model,
                                                 &weights->layer[il],
                                                 il,
                                                 pos,
                                                 g->layer_raw_cache[il],
                                                 g->raw_cap,
                                                 raw_row,
                                                 n_raw,
                                                 token);
            encoded_layer = true;
        }
        if (encoded_layer) {
            ds4_gpu_tensor *tmp = metal_graph_cur_hc(g);
            g->cur_hc_by_tier[g->active_tier] = metal_graph_after_ffn_hc(g);
            g->after_ffn_hc_by_tier[g->active_tier] = tmp;
            if (ok) ok = metal_graph_dspark_capture_decode_layer(g, il);
        }
        const double tl_encoded = profile ? now_sec() : 0.0;
        if (ok) ok = ds4_gpu_end_commands() != 0;
        const double tl_done = profile ? now_sec() : 0.0;
        if (profile) {
            encode_s += tl_encoded - tl0;
            execute_s += tl_done - tl_encoded;
        }
    }

    if (ok && logits && !static_decode_map) ok = metal_graph_stream_map_output(model, weights);
    const double t_head0 = profile ? now_sec() : 0.0;
    if (ok && logits) ok = ds4_gpu_begin_commands() != 0;
    if (ok && logits) ok = metal_graph_encode_output_head(g, model, weights, weights->output->dim[1]);
    const double t_head_encoded = profile ? now_sec() : 0.0;
    if (ok && logits) ok = ds4_gpu_end_commands() != 0;
    const double t_done = (profile || throttle) ? now_sec() : 0.0;
    if (ok && logits) {
        ok = ds4_gpu_tensor_read(metal_graph_logits(g), 0, logits, (uint64_t)DS4_N_VOCAB * sizeof(float)) != 0;
    }
    const double t_read = (profile || throttle) ? now_sec() : 0.0;

    if (profile) {
        if (logits) {
            encode_s += t_head_encoded - t_head0;
            execute_s += t_done - t_head_encoded;
        }
        fprintf(stderr,
                "ds4: metal SSD streaming token pos=%u encode=%.3f ms execute=%.3f ms read=%.3f ms total=%.3f ms logits=%d\n",
                pos,
                encode_s * 1000.0,
                execute_s * 1000.0,
                (t_read - t_done) * 1000.0,
                (t_read - t0) * 1000.0,
                logits != NULL);
    }
    if (ok) graph_power_note_decode_token(g, t_read - t0);
    if (!ok) {
        if (ds4_gpu_synchronize() == 0) {
            fprintf(stderr, "ds4: Metal synchronize after SSD streaming graph eval failure also failed\n");
        }
    }
    return ok;
}

/* Execute one Metal decode token and read back logits. */
static bool metal_graph_eval_token_raw_swa(
        ds4_gpu_graph *g,
        const ds4_model       *model,
        const ds4_weights     *weights,
        int                    token,
        uint32_t               pos,
        float                 *logits) {
    if (g && g->ssd_streaming) {
        return metal_graph_eval_token_raw_swa_streaming(g, model, weights, token, pos, logits);
    }

    const bool profile =
        glm_graph_env_present("DS4_ROCM_GRAPH_TOKEN_PROFILE",
                              "DS4_METAL_GRAPH_TOKEN_PROFILE");
    const bool throttle = graph_power_throttle_enabled(g);
    const double t0 = (profile || throttle) ? now_sec() : 0.0;

    bool ok = ds4_gpu_begin_commands() != 0;
    if (ok) ok = metal_graph_encode_token_raw_swa(g, model, weights, token, pos, logits != NULL, true);
    const double t_encoded = (profile || throttle) ? now_sec() : 0.0;
    if (ok) ok = ds4_gpu_end_commands() != 0;
    const double t_done = (profile || throttle) ? now_sec() : 0.0;

    if (ok && logits && g->tp_world == 2 && g->tp_logits_half) {
        const uint64_t tp_vhalf = (uint64_t)DS4_N_VOCAB / 2u;
        const uint64_t off = (uint64_t)g->tp_rank * tp_vhalf * sizeof(float);
        ok = ds4_gpu_tensor_read(metal_graph_logits(g), off, logits + g->tp_rank * tp_vhalf,
                                 tp_vhalf * sizeof(float)) != 0;
    } else if (ok && logits && !(g->tp_world == 2 && g->tp_rank == 1)) {
        ok = ds4_gpu_tensor_read(metal_graph_logits(g), 0, logits, (uint64_t)DS4_N_VOCAB * sizeof(float)) != 0;
    }
    const double t_read = (profile || throttle) ? now_sec() : 0.0;
    if (profile) {
        fprintf(stderr,
                "ds4: metal graph token pos=%u encode=%.3f ms execute=%.3f ms read=%.3f ms total=%.3f ms logits=%d\n",
                pos,
                (t_encoded - t0) * 1000.0,
                (t_done - t_encoded) * 1000.0,
                (t_read - t_done) * 1000.0,
                (t_read - t0) * 1000.0,
                logits != NULL);
    }
    if (ok) graph_power_note_decode_token(g, t_read - t0);
    if (!ok) {
        if (ds4_gpu_synchronize() == 0) {
            fprintf(stderr, "ds4: Metal synchronize after graph eval failure also failed\n");
        }
    }
    return ok;
}

static bool metal_graph_streaming_decode_prefill_wide_default(
        const ds4_weights *weights) {
    return DS4_MODEL_VARIANT == DS4_VARIANT_FLASH &&
           weights &&
           DS4_N_LAYER > 0 &&
           weights->layer[0].ffn_gate_exps->type == DS4_TENSOR_Q4_K &&
           weights->layer[0].ffn_up_exps->type == DS4_TENSOR_Q4_K &&
           weights->layer[0].ffn_down_exps->type == DS4_TENSOR_Q4_K;
}

static uint32_t metal_graph_streaming_decode_prefill_max_tokens(
        const ds4_gpu_graph *g,
        const ds4_weights   *weights) {
    (void)g;
    if (glm_graph_env_present("DS4_ROCM_DISABLE_STREAMING_DECODE_PREFILL",
                              "DS4_METAL_DISABLE_STREAMING_DECODE_PREFILL")) {
        return 0;
    }

    const char *env = glm_graph_env_value(
            "DS4_ROCM_STREAMING_DECODE_PREFILL_MAX",
            "DS4_METAL_STREAMING_DECODE_PREFILL_MAX");
    if (env && env[0]) {
        char *end = NULL;
        const long v = strtol(env, &end, 10);
        if (end != env) {
            if (v <= 0) return 0;
            if ((unsigned long)v > (unsigned long)UINT32_MAX) return UINT32_MAX;
            return (uint32_t)v;
        }
    }

    if (DS4_MODEL_VARIANT != DS4_VARIANT_PRO &&
        DS4_MODEL_VARIANT != DS4_VARIANT_FLASH) {
        return 0u;
    }
    return metal_graph_streaming_decode_prefill_wide_default(weights) ? 64u : 18u;
}

static bool metal_graph_use_streaming_decode_prefill(
        const ds4_gpu_graph *g,
        const ds4_weights   *weights,
        uint32_t             n_tokens) {
    const uint32_t max_tokens =
        metal_graph_streaming_decode_prefill_max_tokens(g, weights);
    return g &&
           g->ssd_streaming &&
           !g->quality &&
           n_tokens != 0 &&
           max_tokens != 0 &&
           n_tokens <= max_tokens;
}

static bool metal_graph_use_streaming_decode_prefill_range(
        const ds4_gpu_graph *g,
        const ds4_weights   *weights,
        uint32_t             start,
        uint32_t             n_tokens) {
    /*
     * Short streamed prefill is latency-sensitive.  Use the decode-style path
     * by default for SSD streaming, while keeping a cold-only escape hatch for
     * strict-vector tests that need canonical layer-major prefill semantics.
     */
    if (start == 0) {
        if (glm_graph_env_present(
                    "DS4_ROCM_DISABLE_STREAMING_COLD_DECODE_PREFILL",
                    "DS4_METAL_DISABLE_STREAMING_COLD_DECODE_PREFILL")) {
            return false;
        }
    }
    return metal_graph_use_streaming_decode_prefill(g, weights, n_tokens);
}

static bool metal_graph_prefill_decode_streaming_range(
        ds4_gpu_graph *g,
        const ds4_model       *model,
        const ds4_weights     *weights,
        const token_vec       *prompt,
        uint32_t               start,
        uint32_t               n_tokens,
        float                 *logits,
        bool                   show_progress,
        ds4_session_progress_fn progress,
        void                  *progress_ud,
        ds4_session_progress_fn display_progress,
        void                  *display_progress_ud,
        ds4_session_cancel_fn  cancel,
        void                  *cancel_ud,
        bool                  *cancelled) {
    if (!metal_graph_use_streaming_decode_prefill(g, weights, n_tokens)) return false;
    if (!prompt || start > (uint32_t)prompt->len ||
        n_tokens > (uint32_t)prompt->len - start) return false;
    if (start == 0) {
        ds4_gpu_stream_expert_cache_reset_route_hotness();
    }

    const bool profile =
        glm_graph_env_present("DS4_ROCM_GRAPH_PREFILL_PROFILE",
                              "DS4_METAL_GRAPH_PREFILL_PROFILE");
    const double t0 = profile ? now_sec() : 0.0;

    /*
     * `prefill_chunk` is not just UI progress: ds4_session_sync() wraps it to
     * advance the live checkpoint, and ds4-server may save that checkpoint.
     * Decode-style prefill only reads logits for the final token, so report one
     * cacheable chunk at the end. `prefill_display` remains per-token UI only.
     */
    if (progress) progress(progress_ud, "prefill_chunk", (int)start, prompt->len);
    if (display_progress) {
        display_progress(display_progress_ud, "prefill_display", (int)start, prompt->len);
    }

    for (uint32_t i = 0; i < n_tokens; i++) {
        if (cancel && cancel(cancel_ud)) {
            if (cancelled) *cancelled = true;
            return true;
        }
        const uint32_t pos = start + i;
        const bool last = i + 1u == n_tokens;
        float *token_logits = (last && logits) ? logits : NULL;
        if (!metal_graph_eval_token_raw_swa(g,
                                            model,
                                            weights,
                                            prompt->v[pos],
                                            pos,
                                            token_logits)) {
            if (ds4_gpu_synchronize() == 0) {
                fprintf(stderr, "ds4: Metal synchronize after decode-style streaming prefill failure also failed\n");
            }
            return false;
        }

        if (last && progress && logits) {
            progress(progress_ud, "prefill_chunk", (int)(pos + 1u), prompt->len);
        }
        if (display_progress) {
            display_progress(display_progress_ud, "prefill_display", (int)(pos + 1u), prompt->len);
        }
        if (cancel && cancel(cancel_ud)) {
            if (cancelled) *cancelled = true;
            return true;
        }
        if (show_progress) {
            fprintf(stderr, "ds4: gpu streaming prefill token %u/%u\r",
                    i + 1u,
                    n_tokens);
            fflush(stderr);
        }
    }
    if (show_progress) fputc('\n', stderr);

    if (profile) {
        const double t1 = now_sec();
        fprintf(stderr,
                "ds4: gpu decode-style streaming prefill start=%u tokens=%u total=%.3f ms\n",
                start,
                n_tokens,
                (t1 - t0) * 1000.0);
    }
    return true;
}

static bool metal_graph_capture_prefill_seed_router_selected(
        ds4_gpu_graph *g,
        uint32_t       il,
        uint32_t       n_tokens) {
    uint32_t k = metal_graph_streaming_prefill_cache_seed_k(g);
    if (k == 0) return true;
    if (k > n_tokens) k = n_tokens;
    g->prefill_seed_tokens = k;
    if (!g->prefill_seed_router_selected || !metal_graph_batch_router_selected(g) ||
        il >= DS4_N_LAYER || n_tokens == 0 || sizeof(int) != sizeof(int32_t)) {
        return false;
    }

    const uint64_t bytes = (uint64_t)k * DS4_N_EXPERT_USED * sizeof(int32_t);
    const uint64_t src_off = (uint64_t)(n_tokens - k) *
                             DS4_N_EXPERT_USED * sizeof(int);
    const uint64_t dst_off = (uint64_t)il *
                             DS4_STREAMING_PREFILL_CACHE_SEED_MAX_TOKENS *
                             DS4_N_EXPERT_USED * sizeof(int32_t);
    return ds4_gpu_tensor_copy(g->prefill_seed_router_selected,
                               dst_off,
                               metal_graph_batch_router_selected(g),
                               src_off,
                               bytes) != 0;
}

static bool metal_graph_seed_streaming_expert_cache_from_prefill(
        ds4_gpu_graph     *g,
        const ds4_model   *model,
        const ds4_weights *weights) {
    const uint32_t seed_tokens = g ? g->prefill_seed_tokens : 0;
    if (!metal_graph_streaming_prefill_cache_seed_enabled(g)) return true;
    if (!model || !weights || !g->prefill_seed_router_selected || seed_tokens == 0) {
        return false;
    }

    int32_t selected[DS4_MAX_LAYER *
                     DS4_STREAMING_PREFILL_CACHE_SEED_MAX_TOKENS *
                     DS4_MAX_EXPERT_USED];
    const uint64_t bytes = (uint64_t)DS4_N_LAYER *
                           DS4_STREAMING_PREFILL_CACHE_SEED_MAX_TOKENS *
                           DS4_N_EXPERT_USED * sizeof(selected[0]);
    if (ds4_gpu_tensor_read(g->prefill_seed_router_selected,
                            0,
                            selected,
                            bytes) == 0) {
        return false;
    }

    const bool profile =
        glm_graph_env_present("DS4_ROCM_STREAMING_PREFILL_CACHE_SEED_PROFILE",
                              "DS4_METAL_STREAMING_PREFILL_CACHE_SEED_PROFILE");
    const double t0 = profile ? now_sec() : 0.0;
    uint32_t seeded_layers = 0;
    uint32_t seeded_rows = 0;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const ds4_layer_weights *layer = &weights->layer[il];
        if (!metal_graph_streaming_expert_cache_seed_layer_expected(g, layer)) {
            continue;
        }

        const uint64_t gate_row_bytes = routed_expert_row_bytes(layer->ffn_gate_exps);
        const uint64_t down_row_bytes = routed_expert_row_bytes(layer->ffn_down_exps);
        if (layer->ffn_gate_exps->dim[1] > UINT64_MAX / gate_row_bytes ||
            layer->ffn_down_exps->dim[1] > UINT64_MAX / down_row_bytes) {
            fprintf(stderr, "ds4: Metal prefill expert-cache seed byte size overflow at layer %u\n", il);
            return false;
        }
        const uint64_t gate_expert_bytes = layer->ffn_gate_exps->dim[1] * gate_row_bytes;
        const uint64_t down_expert_bytes = layer->ffn_down_exps->dim[1] * down_row_bytes;
        const ds4_gpu_stream_expert_table table =
            graph_stream_expert_table_make(model,
                                           layer,
                                           il,
                                           gate_expert_bytes,
                                           down_expert_bytes);
        for (uint32_t row = 0; row < seed_tokens; row++) {
            const size_t sel_off = ((size_t)il *
                                    DS4_STREAMING_PREFILL_CACHE_SEED_MAX_TOKENS +
                                    row) * DS4_N_EXPERT_USED;
            if (ds4_gpu_stream_expert_cache_seed_selected(
                        &table,
                        selected + sel_off,
                        DS4_N_EXPERT_USED) == 0) {
                return false;
            }
            seeded_rows++;
        }
        seeded_layers++;
    }
    if (profile) {
        fprintf(stderr,
                "ds4: Metal streaming prefill expert-cache seed k=%u layers=%u rows=%u time=%.3f ms\n",
                seed_tokens,
                seeded_layers,
                seeded_rows,
                (now_sec() - t0) * 1000.0);
    }
    return true;
}

static bool metal_graph_seed_streaming_expert_cache_from_hotlist(
        ds4_gpu_graph     *g,
        const ds4_model   *model,
        const ds4_weights *weights) {
    if (!metal_graph_streaming_expert_hotlist_enabled(g)) return true;
    if (!model || !weights) return false;

    uint32_t cache_budget = 0;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const ds4_layer_weights *layer = &weights->layer[il];
        if (!metal_graph_streaming_expert_cache_seed_layer_expected(g, layer)) {
            continue;
        }

        const uint64_t gate_row_bytes = routed_expert_row_bytes(layer->ffn_gate_exps);
        const uint64_t down_row_bytes = routed_expert_row_bytes(layer->ffn_down_exps);
        if (layer->ffn_gate_exps->dim[1] > UINT64_MAX / gate_row_bytes ||
            layer->ffn_down_exps->dim[1] > UINT64_MAX / down_row_bytes) {
            fprintf(stderr, "ds4: streaming expert hotlist budget byte size overflow at layer %u\n", il);
            return false;
        }
        const uint64_t gate_expert_bytes = layer->ffn_gate_exps->dim[1] * gate_row_bytes;
        const uint64_t down_expert_bytes = layer->ffn_down_exps->dim[1] * down_row_bytes;
        cache_budget = ds4_gpu_stream_expert_cache_budget_for_expert_size(
                gate_expert_bytes,
                down_expert_bytes);
        break;
    }
    if (cache_budget == 0) return true;
    const uint32_t preload_count =
        metal_graph_streaming_expert_preload_count(g, cache_budget);
    if (preload_count == 0) return true;
    const uint32_t current_count =
        ds4_gpu_stream_expert_cache_current_count();
    const char *path = glm_graph_env_value("DS4_ROCM_STREAMING_EXPERT_HOTLIST",
                                           "DS4_METAL_STREAMING_EXPERT_HOTLIST");
    const bool from_file = path && path[0];
    const bool refresh_builtin_glm =
        !from_file && g_ds4_shape.variant == DS4_VARIANT_GLM52;
    const bool profile =
        glm_graph_env_present("DS4_ROCM_STREAMING_EXPERT_HOTLIST_PROFILE",
                              "DS4_METAL_STREAMING_EXPERT_HOTLIST_PROFILE");
    if (!from_file && !refresh_builtin_glm && current_count >= preload_count) {
        if (profile) {
            fprintf(stderr,
                    "ds4: streaming expert hotlist seed skipped preload=%u current=%u\n",
                    preload_count,
                    current_count);
        }
        return true;
    }

    int32_t experts[DS4_MAX_LAYER][DS4_MAX_EXPERT];
    uint32_t priorities[DS4_MAX_LAYER][DS4_MAX_EXPERT];
    uint32_t counts[DS4_MAX_LAYER];
    bool seen[DS4_MAX_LAYER][DS4_MAX_EXPERT];
    memset(experts, 0, sizeof(experts));
    memset(priorities, 0, sizeof(priorities));
    memset(counts, 0, sizeof(counts));
    memset(seen, 0, sizeof(seen));

    uint32_t loaded = 0;
    if (from_file) {
        if (!metal_graph_streaming_expert_hotlist_load_file(path,
                                                           preload_count,
                                                           experts,
                                                           priorities,
                                                           counts,
                                                           seen,
                                                           &loaded)) {
            return false;
        }
    } else if (!metal_graph_streaming_expert_hotlist_load_default(preload_count,
                                                                  experts,
                                                                  priorities,
                                                                  counts,
                                                                  seen,
                                                                  &loaded)) {
        return false;
    }
    if (loaded == 0) return true;

    const double t0 = profile ? now_sec() : 0.0;
    uint32_t seeded_layers = 0;
    uint32_t seeded_experts = 0;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const uint32_t n = counts[il];
        if (n == 0) continue;
        const ds4_layer_weights *layer = &weights->layer[il];
        if (!metal_graph_streaming_expert_cache_seed_layer_expected(g, layer)) {
            continue;
        }

        const uint64_t gate_row_bytes = routed_expert_row_bytes(layer->ffn_gate_exps);
        const uint64_t down_row_bytes = routed_expert_row_bytes(layer->ffn_down_exps);
        if (layer->ffn_gate_exps->dim[1] > UINT64_MAX / gate_row_bytes ||
            layer->ffn_down_exps->dim[1] > UINT64_MAX / down_row_bytes) {
            fprintf(stderr, "ds4: streaming expert hotlist seed byte size overflow at layer %u\n", il);
            return false;
        }
        const uint64_t gate_expert_bytes = layer->ffn_gate_exps->dim[1] * gate_row_bytes;
        const uint64_t down_expert_bytes = layer->ffn_down_exps->dim[1] * down_row_bytes;
        const ds4_gpu_stream_expert_table table =
            graph_stream_expert_table_make(model,
                                           layer,
                                           il,
                                           gate_expert_bytes,
                                           down_expert_bytes);
        if (ds4_gpu_stream_expert_cache_seed_experts(
                    &table,
                    experts[il],
                    priorities[il],
                    n) == 0) {
            return false;
        }
        seeded_layers++;
        seeded_experts += n;
    }
    if (profile) {
        const char *source_name = NULL;
        if (from_file) {
            source_name = path;
        } else if (g_ds4_shape.variant == DS4_VARIANT_GLM52) {
            source_name = "built-in-glm52";
        } else if (g_ds4_shape.variant == DS4_VARIANT_FLASH) {
            source_name = "built-in-flash";
        } else if (g_ds4_shape.variant == DS4_VARIANT_PRO) {
            source_name = "built-in-pro";
        } else {
            source_name = "built-in";
        }
        fprintf(stderr,
                "ds4: streaming expert hotlist seed source=%s preload=%u loaded=%u layers=%u experts=%u time=%.3f ms\n",
                source_name,
                preload_count,
                loaded,
                seeded_layers,
                seeded_experts,
                (now_sec() - t0) * 1000.0);
    }
    return true;
}

typedef struct {
    int id0;
    int id1;
    float value0;
    float value1;
    bool valid;
    bool fast_attention;
} metal_graph_top2_result;

/* Greedy verifier helper.  Speculative decoding only needs the target model's
 * top token after most accepted draft rows; the full vocabulary row is needed
 * once, for the final committed state that normal sampling will continue from.
 * Keeping intermediate rows device-resident avoids turning verification into a
 * sequence of large CPU readbacks. */
static bool metal_graph_eval_token_raw_swa_top(
        ds4_gpu_graph *g,
        const ds4_model       *model,
        const ds4_weights     *weights,
        int                    token,
        uint32_t               pos,
        int                   *top_id,
        float                 *logits,
        bool                   allow_split_top1,
        metal_graph_top2_result *top2,
        bool                   force_fast_attention) {
    if (!top_id) return false;
    if (top2) memset(top2, 0, sizeof(*top2));

    const bool fast_attention =
        allow_split_top1 &&
        logits == NULL &&
        (force_fast_attention || metal_graph_cuda_greedy_splitkv_requested());
    if (top2) top2->fast_attention = fast_attention;
    const int old_fast_attention =
        ds4_gpu_set_decode_fast_attention(fast_attention ? 1 : 0);
    const bool profile = getenv("DS4_METAL_GRAPH_TOKEN_PROFILE") != NULL;
    const double t0 = profile ? now_sec() : 0.0;
    const bool split_top1 =
        allow_split_top1 &&
        logits == NULL &&
        top2 == NULL &&
        g->cuda_tp_output &&
        metal_graph_cuda_greedy_split_top1_requested();
    if (split_top1) {
        int output_tiers[DS4_MAX_GPUS] = {0};
        uint32_t output_ways = 0;
        bool ok = ds4_gpu_begin_commands() != 0;
        if (ok) ok = metal_graph_encode_token_raw_swa(g, model, weights,
                                                      token, pos, false, true);
        if (ok) ok = metal_graph_encode_output_head_split_top1(g,
                                                               model,
                                                               weights,
                                                               weights->output->dim[1],
                                                               output_tiers,
                                                               &output_ways);
        const double t_encoded = profile ? now_sec() : 0.0;
        if (ok) ok = ds4_gpu_end_commands() != 0;
        const double t_done = profile ? now_sec() : 0.0;
        if (ok) {
            bool have_best = false;
            uint32_t best_id = 0;
            float best_value = 0.0f;
            uint32_t cand_ids[DS4_MAX_GPUS] = {0};
            float cand_values[DS4_MAX_GPUS] = {0.0f};
            ok = output_ways <= DS4_MAX_GPUS &&
                 ds4_gpu_tensor_read(g->comp_selected_by_tier[g->head_tier],
                                      0,
                                      cand_ids,
                                      (uint64_t)output_ways * sizeof(cand_ids[0])) != 0 &&
                 ds4_gpu_tensor_read(g->comp_mask_by_tier[g->head_tier],
                                      0,
                                      cand_values,
                                      (uint64_t)output_ways * sizeof(cand_values[0])) != 0;
            for (uint32_t i = 0; ok && i < output_ways; i++) {
                const uint32_t cand_id = cand_ids[i];
                const float cand_value = cand_values[i];
                if (ok &&
                    (!have_best ||
                     cand_value > best_value ||
                     (cand_value == best_value && cand_id < best_id))) {
                    have_best = true;
                    best_id = cand_id;
                    best_value = cand_value;
                }
            }
            ok = ok && have_best && best_id <= (uint32_t)INT32_MAX;
            if (ok) *top_id = (int)best_id;
        }
        const double t_read = profile ? now_sec() : 0.0;
        if (profile) {
            fprintf(stderr,
                    "ds4: metal graph token pos=%u encode=%.3f ms execute=%.3f ms read=%.3f ms total=%.3f ms logits=0 top=1 split_top1=1\n",
                    pos,
                    (t_encoded - t0) * 1000.0,
                    (t_done - t_encoded) * 1000.0,
                    (t_read - t_done) * 1000.0,
                    (t_read - t0) * 1000.0);
        }
        if (!ok) {
            if (ds4_gpu_synchronize() == 0) {
                fprintf(stderr, "ds4: Metal synchronize after split-top graph eval failure also failed\n");
            }
        }
        (void)ds4_gpu_set_decode_fast_attention(old_fast_attention);
        return ok;
    }

    bool ok = ds4_gpu_begin_commands() != 0;
    if (ok) ok = metal_graph_encode_token_raw_swa(g, model, weights,
                                                  token, pos, true, true);
    if (ok) {
        ok = ds4_gpu_argmax_tensor(metal_graph_comp_selected(g),
                                   metal_graph_logits(g),
                                   DS4_N_VOCAB) != 0;
    }
    const double t_encoded = profile ? now_sec() : 0.0;
    if (ok) ok = ds4_gpu_end_commands() != 0;
    const double t_done = profile ? now_sec() : 0.0;
    if (ok && top2) {
        uint32_t ids[2] = {0, 0};
        float values[2] = {0.0f, 0.0f};
        ok = ds4_gpu_tensor_read(metal_graph_comp_selected(g),
                                 0,
                                 ids,
                                 sizeof(ids)) != 0 &&
             ds4_gpu_tensor_read(metal_graph_comp_mask(g),
                                 0,
                                 values,
                                 sizeof(values)) != 0;
        if (ok && ids[0] <= (uint32_t)INT32_MAX && ids[1] <= (uint32_t)INT32_MAX) {
            top2->id0 = (int)ids[0];
            top2->id1 = (int)ids[1];
            top2->value0 = values[0];
            top2->value1 = values[1];
            top2->valid = isfinite(values[0]) && isfinite(values[1]);
            *top_id = top2->id0;
        } else {
            ok = false;
        }
    } else if (ok) {
        ok = ds4_gpu_tensor_read(metal_graph_comp_selected(g), 0, top_id, sizeof(*top_id)) != 0;
    }
    if (ok && logits) {
        ok = ds4_gpu_tensor_read(metal_graph_logits(g), 0, logits, (uint64_t)DS4_N_VOCAB * sizeof(float)) != 0;
    }
    const double t_read = profile ? now_sec() : 0.0;
    if (profile) {
        fprintf(stderr,
                "ds4: metal graph token pos=%u encode=%.3f ms execute=%.3f ms read=%.3f ms total=%.3f ms logits=%d top=1 split_top1=0\n",
                pos,
                (t_encoded - t0) * 1000.0,
                (t_done - t_encoded) * 1000.0,
                (t_read - t_done) * 1000.0,
                (t_read - t0) * 1000.0,
                logits != NULL);
    }
    if (!ok) {
        if (ds4_gpu_synchronize() == 0) {
            fprintf(stderr, "ds4: Metal synchronize after top-only graph eval failure also failed\n");
        }
    }
    (void)ds4_gpu_set_decode_fast_attention(old_fast_attention);
    return ok;
}

static bool dspark_stage0_weights_ready(
        const ds4_gpu_graph     *g,
        const ds4_dspark_weights *dw) {
    if (!g || !dw || dw->n_stages == 0 || dw->target_layer_count == 0 ||
        dw->target_layer_count != g->dspark_target_layer_count ||
        !g->dspark_target_hidden || !g->dspark_stage0_proj ||
        !g->dspark_main_x) {
        return false;
    }

    const ds4_dspark_stage_weights *stage0 = &dw->stage[0];
    const ds4_tensor *main_proj = stage0->main_proj;
    const ds4_tensor *main_norm = stage0->main_norm;
    const uint64_t in_dim = (uint64_t)dw->target_layer_count * DS4_N_EMBD;
    return main_proj &&
           main_norm &&
           dspark_tensor_type_matches(main_proj->type, DS4_DSPARK_LAYOUT_DENSE) &&
           main_norm->type == DS4_TENSOR_F32 &&
           main_proj->ndim == 2 &&
           main_proj->dim[0] == in_dim &&
           main_proj->dim[1] == DS4_N_EMBD &&
           main_norm->ndim == 1 &&
           main_norm->dim[0] == DS4_N_EMBD;
}

static bool metal_graph_eval_dspark_stage0(
        ds4_gpu_graph          *g,
        const ds4_model        *dspark_model,
        const ds4_dspark_weights *dw) {
    if (!g || !dspark_model || !dw || !dspark_stage0_weights_ready(g, dw)) {
        return false;
    }

    const ds4_dspark_stage_weights *stage0 = &dw->stage[0];
    const uint64_t in_dim = (uint64_t)dw->target_layer_count * DS4_N_EMBD;
    bool ok = ds4_gpu_begin_commands() != 0;
    if (ok) {
        ok = metal_graph_matmul_plain_tensor(g->dspark_stage0_proj,
                                             dspark_model,
                                             stage0->main_proj,
                                             in_dim,
                                             DS4_N_EMBD,
                                             g->dspark_target_hidden,
                                             1);
    }
    if (ok) {
        ok = ds4_gpu_rms_norm_weight_tensor(g->dspark_main_x,
                                            g->dspark_stage0_proj,
                                            dspark_model->map,
                                            dspark_model->size,
                                            stage0->main_norm->abs_offset,
                                            DS4_N_EMBD,
                                            DS4_RMS_EPS) != 0;
    }
    if (ok) ok = ds4_gpu_end_commands() != 0;
    if (!ok) (void)ds4_gpu_synchronize();
    return ok;
}

static bool dspark_stage0_batch_ready(
        const ds4_gpu_graph      *g,
        const ds4_dspark_weights *dw,
        uint32_t                  n_tokens) {
    if (!dspark_stage0_weights_ready(g, dw) ||
        n_tokens == 0 ||
        n_tokens > g->prefill_cap ||
        !g->dspark_target_hidden_batch ||
        !metal_graph_batch_ffn_cur(g) ||
        !metal_graph_batch_ffn_norm(g) ||
        !metal_graph_batch_cur_hc(g)) {
        return false;
    }
    const uint64_t in_dim = (uint64_t)dw->target_layer_count * DS4_N_EMBD;
    return ds4_gpu_tensor_bytes(metal_graph_batch_ffn_cur(g)) >=
               (uint64_t)n_tokens * DS4_N_EMBD * sizeof(float) &&
           ds4_gpu_tensor_bytes(metal_graph_batch_ffn_norm(g)) >=
               (uint64_t)n_tokens * DS4_N_EMBD * sizeof(float) &&
           ds4_gpu_tensor_bytes(metal_graph_batch_cur_hc(g)) >=
               (uint64_t)n_tokens * DS4_N_HC * DS4_N_EMBD * sizeof(float) &&
           in_dim <= SIZE_MAX / sizeof(float);
}

static bool metal_graph_pack_dspark_target_hidden_batch(
        ds4_gpu_graph     *g,
        const ds4_dspark_weights *dw,
        ds4_gpu_tensor    *packed,
        uint32_t           n_tokens) {
    if (!g || !dw || !packed || n_tokens == 0 ||
        n_tokens > g->prefill_cap ||
        dw->target_layer_count != g->dspark_target_layer_count) {
        return false;
    }

    const uint64_t in_dim = (uint64_t)dw->target_layer_count * DS4_N_EMBD;
    const uint64_t packed_count = (uint64_t)n_tokens * in_dim;
    if (packed_count == 0 ||
        packed_count > (uint64_t)SIZE_MAX / sizeof(float)) {
        return false;
    }
    return ds4_gpu_pack_slot_rows_f32_tensor(packed,
                                             g->dspark_target_hidden_batch,
                                             n_tokens,
                                             DS4_N_EMBD,
                                             dw->target_layer_count,
                                             g->prefill_cap) != 0;
}

static bool metal_graph_eval_dspark_stage0_batch(
        ds4_gpu_graph            *g,
        const ds4_model          *dspark_model,
        const ds4_dspark_weights *dw,
        uint32_t                  n_tokens,
        bool                      commands_open) {
    if (!g || !dspark_model || !dw ||
        !dspark_stage0_batch_ready(g, dw, n_tokens)) {
        return false;
    }

    const ds4_dspark_stage_weights *stage0 = &dw->stage[0];
    const uint64_t in_dim = (uint64_t)dw->target_layer_count * DS4_N_EMBD;
    const uint64_t packed_bytes = (uint64_t)n_tokens * in_dim * sizeof(float);
    bool packed_owned = false;
    ds4_gpu_tensor *packed = NULL;
    if (g->dspark_stage0_packed &&
        ds4_gpu_tensor_bytes(g->dspark_stage0_packed) >= packed_bytes) {
        packed = g->dspark_stage0_packed;
    } else {
        packed = ds4_gpu_tensor_alloc(packed_bytes);
        packed_owned = true;
    }
    if (!packed) return false;

    bool ok = metal_graph_pack_dspark_target_hidden_batch(g, dw, packed, n_tokens);
    if (ok && !commands_open) ok = ds4_gpu_begin_commands() != 0;
    if (ok) {
        ok = metal_graph_matmul_plain_tensor(metal_graph_batch_ffn_cur(g),
                                             dspark_model,
                                             stage0->main_proj,
                                             in_dim,
                                             DS4_N_EMBD,
                                             packed,
                                             n_tokens);
    }
    if (ok) {
        ok = ds4_gpu_rms_norm_weight_rows_tensor(metal_graph_batch_ffn_norm(g),
                                                 metal_graph_batch_ffn_cur(g),
                                                 dspark_model->map,
                                                 dspark_model->size,
                                                 stage0->main_norm->abs_offset,
                                                 DS4_N_EMBD,
                                                 n_tokens,
                                                 DS4_RMS_EPS) != 0;
    }
    if (ok && !commands_open) ok = ds4_gpu_end_commands() != 0;
    if (!ok && !commands_open) (void)ds4_gpu_synchronize();
    if (packed_owned) ds4_gpu_tensor_free(packed);
    return ok;
}

static bool dspark_draft_block_ready(
        const ds4_gpu_graph      *g,
        const ds4_weights        *base_weights,
        const ds4_dspark_weights *dw,
        int                       token) {
    if (!g || !base_weights || !dw || !base_weights->token_embd ||
        !g->dspark_draft_tokens || !g->dspark_draft_hc ||
        dw->block_size == 0 ||
        dw->block_size > DS4_DSPARK_MAX_BLOCK_SIZE ||
        g->dspark_block_size != dw->block_size ||
        !dw->has_noise_token_id) {
        return false;
    }
    const uint32_t n_vocab = (uint32_t)base_weights->token_embd->dim[1];
    return token >= 0 &&
           (uint32_t)token < n_vocab &&
           dw->noise_token_id < n_vocab;
}

static bool dspark_stage_input_ready(
        const ds4_gpu_graph      *g,
        const ds4_dspark_weights *dw) {
    if (!g || !dw ||
        dw->block_size == 0 ||
        dw->block_size > DS4_DSPARK_MAX_BLOCK_SIZE ||
        g->dspark_block_size != dw->block_size ||
        !g->dspark_main_x || !g->dspark_draft_hc ||
        !g->dspark_target_hc || !g->dspark_stage_input_hc ||
        !g->dspark_position_ids) {
        return false;
    }
    if (dw->block_size == UINT32_MAX) return false;
    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const uint64_t rows = (uint64_t)dw->block_size + 1u;
    return ds4_gpu_tensor_bytes(g->dspark_target_hc) >=
               hc_dim * sizeof(float) &&
           ds4_gpu_tensor_bytes(g->dspark_stage_input_hc) >=
               rows * hc_dim * sizeof(float) &&
           ds4_gpu_tensor_bytes(g->dspark_position_ids) >=
               rows * sizeof(int32_t);
}

static bool dspark_stage_cache_ready(
        const ds4_gpu_graph      *g,
        const ds4_dspark_weights *dw) {
    if (!g || !dw ||
        dw->n_stages == 0 ||
        dw->n_stages > DS4_DSPARK_MAX_STAGES ||
        g->dspark_cache_cap == 0 ||
        !metal_graph_dspark_cache_current_window_valid(g)) {
        return false;
    }
    const uint64_t bytes =
        (uint64_t)g->dspark_cache_cap * DS4_N_HEAD_DIM * sizeof(float);
    for (uint32_t stage = 0; stage < dw->n_stages; stage++) {
        if (!g->dspark_raw_cache[stage] ||
            ds4_gpu_tensor_bytes(g->dspark_raw_cache[stage]) < bytes) {
            return false;
        }
    }
    return true;
}

static bool dspark_noncausal_attention_probe_ready(
        const ds4_gpu_graph      *g,
        const ds4_dspark_weights *dw) {
    if (!g || !dw ||
        dw->n_stages == 0 ||
        dw->block_size == 0 ||
        dw->block_size > DS4_DSPARK_MAX_BLOCK_SIZE ||
        g->prefill_cap < dw->block_size + 1u ||
        !metal_graph_batch_q(g) || !metal_graph_batch_heads(g) ||
        !g->dspark_raw_cache[0]) {
        return false;
    }
    const ds4_layer_weights *block = &dw->stage[0].block;
    if (!block->attn_sinks) return false;

    const uint64_t rows = (uint64_t)dw->block_size + 1u;
    const uint64_t q_dim = (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM;
    return ds4_gpu_tensor_bytes(metal_graph_batch_q(g)) >=
               rows * q_dim * sizeof(float) &&
           ds4_gpu_tensor_bytes(metal_graph_batch_heads(g)) >=
               rows * q_dim * sizeof(float) &&
           ds4_gpu_tensor_bytes(g->dspark_raw_cache[0]) >=
               rows * DS4_N_HEAD_DIM * sizeof(float);
}

static bool metal_graph_probe_dspark_noncausal_attention(
        ds4_gpu_graph            *g,
        const ds4_model          *dspark_model,
        const ds4_dspark_weights *dw) {
    if (!g || !dspark_model || !dspark_noncausal_attention_probe_ready(g, dw)) {
        return false;
    }

    const uint32_t rows = dw->block_size + 1u;
    const uint64_t q_dim = (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM;
    const ds4_layer_weights *block = &dw->stage[0].block;
    bool ok = ds4_gpu_tensor_fill_f32(metal_graph_batch_q(g),
                                      0.0f,
                                      (uint64_t)rows * q_dim) != 0 &&
              ds4_gpu_tensor_fill_f32(g->dspark_raw_cache[0],
                                      0.0f,
                                      (uint64_t)rows * DS4_N_HEAD_DIM) != 0;
    if (ok) ok = ds4_gpu_begin_commands() != 0;
    if (ok) {
        ok = ds4_gpu_attention_noncausal_raw_batch_heads_tensor(
                metal_graph_batch_heads(g),
                dspark_model->map,
                dspark_model->size,
                block->attn_sinks->abs_offset,
                metal_graph_batch_q(g),
                g->dspark_raw_cache[0],
                rows,
                rows,
                rows,
                0,
                DS4_N_HEAD,
                DS4_N_HEAD_DIM) != 0;
    }
    if (ok) ok = ds4_gpu_end_commands() != 0;
    if (!ok) (void)ds4_gpu_synchronize();
    return ok;
}

static bool metal_graph_prepare_dspark_setup_block(
        ds4_gpu_graph            *g,
        const ds4_model          *base_model,
        const ds4_weights        *base_weights,
        const ds4_dspark_weights *dw,
        int                       token,
        uint32_t                  pos) {
    if (!g || !base_model ||
        !dspark_draft_block_ready(g, base_weights, dw, token) ||
        !dspark_stage_input_ready(g, dw)) {
        return false;
    }
    if (pos > (uint32_t)INT32_MAX ||
        dw->block_size > (uint32_t)INT32_MAX ||
        pos > (uint32_t)INT32_MAX - dw->block_size) {
        return false;
    }

    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const uint64_t hc_bytes = hc_dim * sizeof(float);
    int32_t positions[DS4_DSPARK_MAX_BLOCK_SIZE + 1u];
    positions[0] = (int32_t)pos;
    for (uint32_t i = 0; i < dw->block_size; i++) {
        positions[i + 1u] = (int32_t)(pos + i);
    }
    int32_t ids[DS4_DSPARK_MAX_BLOCK_SIZE];
    ids[0] = (int32_t)token;
    for (uint32_t i = 1; i < dw->block_size; i++) {
        ids[i] = (int32_t)dw->noise_token_id;
    }

    bool ok = ds4_gpu_tensor_write(g->dspark_draft_tokens,
                                   0,
                                   ids,
                                   (uint64_t)dw->block_size * sizeof(ids[0])) != 0 &&
              ds4_gpu_tensor_write(g->dspark_position_ids,
                                   0,
                                   positions,
                                   ((uint64_t)dw->block_size + 1u) *
                                       sizeof(positions[0])) != 0;
    if (ok) ok = ds4_gpu_begin_commands() != 0;
    if (ok) {
        ok = ds4_gpu_embed_tokens_hc_tensor(g->dspark_draft_hc,
                                            g->dspark_draft_tokens,
                                            base_model->map,
                                            base_model->size,
                                            base_weights->token_embd->abs_offset,
                                            (uint32_t)base_weights->token_embd->dim[1],
                                            dw->block_size,
                                            DS4_N_EMBD,
                                            DS4_N_HC) != 0;
    }
    if (ok) {
        ok = ds4_gpu_repeat_hc_tensor(g->dspark_target_hc,
                                      g->dspark_main_x,
                                      DS4_N_EMBD,
                                      DS4_N_HC) != 0;
    }
    if (ok) {
        ok = ds4_gpu_tensor_copy(g->dspark_stage_input_hc,
                                 0,
                                 g->dspark_target_hc,
                                 0,
                                 hc_bytes) != 0;
    }
    if (ok) {
        ok = ds4_gpu_tensor_copy(g->dspark_stage_input_hc,
                                 hc_bytes,
                                 g->dspark_draft_hc,
                                 0,
                                 (uint64_t)dw->block_size * hc_bytes) != 0;
    }
    if (ok) ok = ds4_gpu_end_commands() != 0;
    if (!ok) (void)ds4_gpu_synchronize();
    return ok;
}

static bool metal_graph_prepare_dspark_stage0_setup_block(
        ds4_gpu_graph            *g,
        const ds4_model          *base_model,
        const ds4_weights        *base_weights,
        const ds4_model          *dspark_model,
        const ds4_dspark_weights *dw,
        int                       token,
        uint32_t                  pos) {
    if (!g || !base_model || !dspark_model ||
        !dspark_stage0_weights_ready(g, dw) ||
        !dspark_draft_block_ready(g, base_weights, dw, token) ||
        !dspark_stage_input_ready(g, dw)) {
        return false;
    }
    if (pos > (uint32_t)INT32_MAX ||
        dw->block_size > (uint32_t)INT32_MAX ||
        pos > (uint32_t)INT32_MAX - dw->block_size) {
        return false;
    }

    const ds4_dspark_stage_weights *stage0 = &dw->stage[0];
    const uint64_t in_dim = (uint64_t)dw->target_layer_count * DS4_N_EMBD;
    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const uint64_t hc_bytes = hc_dim * sizeof(float);
    int32_t positions[DS4_DSPARK_MAX_BLOCK_SIZE + 1u];
    positions[0] = (int32_t)pos;
    for (uint32_t i = 0; i < dw->block_size; i++) {
        positions[i + 1u] = (int32_t)(pos + i);
    }
    int32_t ids[DS4_DSPARK_MAX_BLOCK_SIZE];
    ids[0] = (int32_t)token;
    for (uint32_t i = 1; i < dw->block_size; i++) {
        ids[i] = (int32_t)dw->noise_token_id;
    }

    /* DS4_DSPARK_PROP_PROFILE=1: break the setup block into phases to
     * localize the TP-only prop_setup inflation (26ms vs 1.3ms single). */
    const bool prop_profile = getenv("DS4_DSPARK_PROP_PROFILE") != NULL;
    const double pp_t0 = prop_profile ? now_sec() : 0.0;
    bool ok = ds4_gpu_tensor_write(g->dspark_draft_tokens,
                                   0,
                                   ids,
                                   (uint64_t)dw->block_size * sizeof(ids[0])) != 0 &&
              ds4_gpu_tensor_write(g->dspark_position_ids,
                                   0,
                                   positions,
                                   ((uint64_t)dw->block_size + 1u) *
                                       sizeof(positions[0])) != 0;
    const double pp_t1 = prop_profile ? now_sec() : 0.0;
    if (ok) ok = ds4_gpu_begin_commands() != 0;
    const double pp_t2 = prop_profile ? now_sec() : 0.0;
    if (ok) {
        ok = metal_graph_matmul_plain_tensor(g->dspark_stage0_proj,
                                             dspark_model,
                                             stage0->main_proj,
                                             in_dim,
                                             DS4_N_EMBD,
                                             g->dspark_target_hidden,
                                             1);
    }
    if (ok) {
        ok = ds4_gpu_rms_norm_weight_tensor(g->dspark_main_x,
                                            g->dspark_stage0_proj,
                                            dspark_model->map,
                                            dspark_model->size,
                                            stage0->main_norm->abs_offset,
                                            DS4_N_EMBD,
                                            DS4_RMS_EPS) != 0;
    }
    if (ok) {
        ok = ds4_gpu_embed_tokens_hc_tensor(g->dspark_draft_hc,
                                            g->dspark_draft_tokens,
                                            base_model->map,
                                            base_model->size,
                                            base_weights->token_embd->abs_offset,
                                            (uint32_t)base_weights->token_embd->dim[1],
                                            dw->block_size,
                                            DS4_N_EMBD,
                                            DS4_N_HC) != 0;
    }
    if (ok) {
        ok = ds4_gpu_repeat_hc_tensor(g->dspark_target_hc,
                                      g->dspark_main_x,
                                      DS4_N_EMBD,
                                      DS4_N_HC) != 0;
    }
    if (ok) {
        ok = ds4_gpu_tensor_copy(g->dspark_stage_input_hc,
                                 0,
                                 g->dspark_target_hc,
                                 0,
                                 hc_bytes) != 0;
    }
    if (ok) {
        ok = ds4_gpu_tensor_copy(g->dspark_stage_input_hc,
                                 hc_bytes,
                                 g->dspark_draft_hc,
                                 0,
                                 (uint64_t)dw->block_size * hc_bytes) != 0;
    }
    const double pp_t3 = prop_profile ? now_sec() : 0.0;
    if (ok) ok = ds4_gpu_end_commands() != 0;
    if (prop_profile) {
        const double pp_t4 = now_sec();
        fprintf(stderr,
                "ds4: DSpark prop-setup phases: writes=%.3fms begin=%.3fms "
                "encode=%.3fms end/wait=%.3fms\n",
                (pp_t1 - pp_t0) * 1000.0,
                (pp_t2 - pp_t1) * 1000.0,
                (pp_t3 - pp_t2) * 1000.0,
                (pp_t4 - pp_t3) * 1000.0);
    }
    if (!ok) (void)ds4_gpu_synchronize();
    return ok;
}

static bool dspark_stage_block_ready(
        const ds4_gpu_graph      *g,
        const ds4_dspark_weights *dw,
        uint32_t                  stage) {
    if (!g || !dw ||
        stage >= dw->n_stages ||
        dw->block_size == 0 ||
        dw->block_size > DS4_DSPARK_MAX_BLOCK_SIZE ||
        g->prefill_cap < dw->block_size + 1u ||
        !g->dspark_stage_output_hc ||
        !dspark_stage_input_ready(g, dw) ||
        !dspark_stage_cache_ready(g, dw)) {
        return false;
    }

    const ds4_layer_weights *l = &dw->stage[stage].block;
    if (!l->hc_attn_fn || !l->hc_attn_scale || !l->hc_attn_base ||
        !l->attn_norm || !l->attn_q_a || !l->attn_q_a_norm ||
        !l->attn_q_b || !l->attn_kv || !l->attn_kv_a_norm ||
        !l->attn_sinks || !l->attn_output_a || !l->attn_output_b ||
        !l->hc_ffn_fn || !l->hc_ffn_scale || !l->hc_ffn_base ||
        !l->ffn_norm || !l->ffn_gate_inp || !l->ffn_exp_probs_b ||
        !l->ffn_gate_exps || !l->ffn_up_exps || !l->ffn_down_exps ||
        !l->ffn_gate_shexp || !l->ffn_up_shexp || !l->ffn_down_shexp) {
        return false;
    }

    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const uint64_t mix_hc = 2ull * DS4_N_HC + (uint64_t)DS4_N_HC * DS4_N_HC;
    const uint64_t q_dim = (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM;
    const uint64_t rows = (uint64_t)dw->block_size + 1u;
    const uint64_t draft = dw->block_size;
    const uint64_t out_low_dim = (uint64_t)DS4_N_OUT_GROUP * DS4_N_LORA_O;
    const uint64_t group_dim =
        (uint64_t)DS4_N_HEAD_DIM * (DS4_N_HEAD / DS4_N_OUT_GROUP);

    return
        dspark_tensor_type_matches(l->hc_attn_fn->type, DS4_DSPARK_LAYOUT_PLAIN) &&
        l->hc_attn_scale->type == DS4_TENSOR_F32 &&
        l->hc_attn_base->type == DS4_TENSOR_F32 &&
        l->attn_norm->type == DS4_TENSOR_F32 &&
        dspark_tensor_type_matches(l->attn_q_a->type, DS4_DSPARK_LAYOUT_DENSE) &&
        l->attn_q_a_norm->type == DS4_TENSOR_F32 &&
        dspark_tensor_type_matches(l->attn_q_b->type, DS4_DSPARK_LAYOUT_DENSE) &&
        dspark_tensor_type_matches(l->attn_kv->type, DS4_DSPARK_LAYOUT_DENSE) &&
        l->attn_kv_a_norm->type == DS4_TENSOR_F32 &&
        l->attn_sinks->type == DS4_TENSOR_F32 &&
        l->attn_output_a->type == DS4_TENSOR_Q8_0 &&
        l->attn_output_b->type == DS4_TENSOR_Q8_0 &&
        dspark_tensor_type_matches(l->hc_ffn_fn->type, DS4_DSPARK_LAYOUT_PLAIN) &&
        l->hc_ffn_scale->type == DS4_TENSOR_F32 &&
        l->hc_ffn_base->type == DS4_TENSOR_F32 &&
        l->ffn_norm->type == DS4_TENSOR_F32 &&
        dspark_tensor_type_matches(l->ffn_gate_inp->type, DS4_DSPARK_LAYOUT_DENSE) &&
        l->ffn_exp_probs_b->type == DS4_TENSOR_F32 &&
        tensor_is_routed_expert_type(l->ffn_gate_exps->type) &&
        l->ffn_gate_exps->type == l->ffn_up_exps->type &&
        tensor_is_routed_expert_type(l->ffn_down_exps->type) &&
        l->ffn_gate_shexp->type == DS4_TENSOR_Q8_0 &&
        l->ffn_up_shexp->type == DS4_TENSOR_Q8_0 &&
        l->ffn_down_shexp->type == DS4_TENSOR_Q8_0 &&
        l->hc_attn_fn->ndim == 2 &&
        l->hc_attn_fn->dim[0] == hc_dim &&
        l->hc_attn_fn->dim[1] == mix_hc &&
        l->attn_q_a->ndim == 2 &&
        l->attn_q_a->dim[0] == DS4_N_EMBD &&
        l->attn_q_a->dim[1] == DS4_N_LORA_Q &&
        l->attn_q_b->ndim == 2 &&
        l->attn_q_b->dim[0] == DS4_N_LORA_Q &&
        l->attn_q_b->dim[1] == q_dim &&
        l->attn_kv->ndim == 2 &&
        l->attn_kv->dim[0] == DS4_N_EMBD &&
        l->attn_kv->dim[1] == DS4_N_HEAD_DIM &&
        l->attn_output_a->ndim == 2 &&
        l->attn_output_a->dim[0] == group_dim &&
        l->attn_output_a->dim[1] == out_low_dim &&
        l->attn_output_b->ndim == 2 &&
        l->attn_output_b->dim[0] == out_low_dim &&
        l->attn_output_b->dim[1] == DS4_N_EMBD &&
        ds4_gpu_tensor_bytes(metal_graph_batch_hc_mix(g)) >= rows * mix_hc * sizeof(float) &&
        ds4_gpu_tensor_bytes(metal_graph_batch_hc_split(g)) >= rows * mix_hc * sizeof(float) &&
        ds4_gpu_tensor_bytes(metal_graph_batch_flat_hc(g)) >= rows * hc_dim * sizeof(float) &&
        ds4_gpu_tensor_bytes(metal_graph_batch_attn_cur(g)) >= rows * DS4_N_EMBD * sizeof(float) &&
        ds4_gpu_tensor_bytes(metal_graph_batch_attn_norm(g)) >= rows * DS4_N_EMBD * sizeof(float) &&
        ds4_gpu_tensor_bytes(metal_graph_batch_qr(g)) >= draft * DS4_N_LORA_Q * sizeof(float) &&
        ds4_gpu_tensor_bytes(metal_graph_batch_qr_norm(g)) >= draft * DS4_N_LORA_Q * sizeof(float) &&
        ds4_gpu_tensor_bytes(metal_graph_batch_q(g)) >= draft * q_dim * sizeof(float) &&
        ds4_gpu_tensor_bytes(metal_graph_batch_kv_raw(g)) >= rows * DS4_N_HEAD_DIM * sizeof(float) &&
        ds4_gpu_tensor_bytes(metal_graph_batch_kv(g)) >= rows * DS4_N_HEAD_DIM * sizeof(float) &&
        ds4_gpu_tensor_bytes(metal_graph_batch_heads(g)) >= draft * q_dim * sizeof(float) &&
        ds4_gpu_tensor_bytes(metal_graph_batch_attn_out(g)) >= draft * DS4_N_EMBD * sizeof(float) &&
        ds4_gpu_tensor_bytes(metal_graph_batch_after_attn_hc(g)) >= draft * hc_dim * sizeof(float) &&
        ds4_gpu_tensor_bytes(metal_graph_batch_next_hc(g)) >= draft * hc_dim * sizeof(float) &&
        ds4_gpu_tensor_bytes(metal_graph_prefill_tokens(g)) >= draft * sizeof(int32_t) &&
        ds4_gpu_tensor_bytes(g->dspark_stage_output_hc) >= draft * hc_dim * sizeof(float);
}

static bool dspark_stage_target_cache_seed_ready(
        const ds4_gpu_graph      *g,
        const ds4_dspark_weights *dw,
        uint32_t                  stage,
        uint32_t                  n_tokens) {
    if (!g || !dw ||
        stage >= dw->n_stages ||
        n_tokens == 0 ||
        n_tokens > g->prefill_cap ||
        !dspark_stage_cache_ready(g, dw) ||
        !metal_graph_batch_cur_hc(g) ||
        !metal_graph_batch_hc_mix(g) ||
        !metal_graph_batch_hc_split(g) ||
        !metal_graph_batch_flat_hc(g) ||
        !metal_graph_batch_attn_cur(g) ||
        !metal_graph_batch_attn_norm(g) ||
        !metal_graph_batch_kv_raw(g) ||
        !metal_graph_batch_kv(g)) {
        return false;
    }

    const ds4_layer_weights *l = &dw->stage[stage].block;
    if (!l->hc_attn_fn || !l->hc_attn_scale || !l->hc_attn_base ||
        !l->attn_norm || !l->attn_kv || !l->attn_kv_a_norm) {
        return false;
    }

    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const uint64_t mix_hc = 2ull * DS4_N_HC + (uint64_t)DS4_N_HC * DS4_N_HC;
    return
        dspark_tensor_type_matches(l->hc_attn_fn->type, DS4_DSPARK_LAYOUT_PLAIN) &&
        l->hc_attn_scale->type == DS4_TENSOR_F32 &&
        l->hc_attn_base->type == DS4_TENSOR_F32 &&
        l->attn_norm->type == DS4_TENSOR_F32 &&
        dspark_tensor_type_matches(l->attn_kv->type, DS4_DSPARK_LAYOUT_DENSE) &&
        l->attn_kv_a_norm->type == DS4_TENSOR_F32 &&
        l->hc_attn_fn->ndim == 2 &&
        l->hc_attn_fn->dim[0] == hc_dim &&
        l->hc_attn_fn->dim[1] == mix_hc &&
        l->attn_kv->ndim == 2 &&
        l->attn_kv->dim[0] == DS4_N_EMBD &&
        l->attn_kv->dim[1] == DS4_N_HEAD_DIM &&
        ds4_gpu_tensor_bytes(metal_graph_batch_cur_hc(g)) >=
            (uint64_t)n_tokens * hc_dim * sizeof(float) &&
        ds4_gpu_tensor_bytes(metal_graph_batch_hc_mix(g)) >=
            (uint64_t)n_tokens * mix_hc * sizeof(float) &&
        ds4_gpu_tensor_bytes(metal_graph_batch_hc_split(g)) >=
            (uint64_t)n_tokens * mix_hc * sizeof(float) &&
        ds4_gpu_tensor_bytes(metal_graph_batch_flat_hc(g)) >=
            (uint64_t)n_tokens * hc_dim * sizeof(float) &&
        ds4_gpu_tensor_bytes(metal_graph_batch_attn_cur(g)) >=
            (uint64_t)n_tokens * DS4_N_EMBD * sizeof(float) &&
        ds4_gpu_tensor_bytes(metal_graph_batch_attn_norm(g)) >=
            (uint64_t)n_tokens * DS4_N_EMBD * sizeof(float) &&
        ds4_gpu_tensor_bytes(metal_graph_batch_kv_raw(g)) >=
            (uint64_t)n_tokens * DS4_N_HEAD_DIM * sizeof(float) &&
        ds4_gpu_tensor_bytes(metal_graph_batch_kv(g)) >=
            (uint64_t)n_tokens * DS4_N_HEAD_DIM * sizeof(float);
}

static bool metal_graph_seed_dspark_stage_target_cache(
        ds4_gpu_graph            *g,
        const ds4_model          *dspark_model,
        const ds4_dspark_weights *dw,
        uint32_t                  stage,
        uint32_t                  pos0,
        uint32_t                  n_tokens,
        bool                      commands_open) {
    if (!g || !dspark_model || !dw ||
        !dspark_stage_target_cache_seed_ready(g, dw, stage, n_tokens) ||
        n_tokens > g->dspark_cache_cap) {
        return false;
    }

    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const uint64_t mix_hc = 2ull * DS4_N_HC + (uint64_t)DS4_N_HC * DS4_N_HC;
    const ds4_layer_weights *block = &dw->stage[stage].block;
    const bool fuse_hc_norm = DS4_N_HC == 4 &&
                              !metal_graph_use_reference_hc_decode() &&
                              metal_graph_enable_batch_hc_norm_fusion();

    ds4_gpu_tensor *hc_mix_view =
        ds4_gpu_tensor_view(metal_graph_batch_hc_mix(g),
                            0,
                            (uint64_t)n_tokens * mix_hc * sizeof(float));
    ds4_gpu_tensor *hc_split_view =
        ds4_gpu_tensor_view(metal_graph_batch_hc_split(g),
                            0,
                            (uint64_t)n_tokens * mix_hc * sizeof(float));
    ds4_gpu_tensor *attn_cur_view =
        ds4_gpu_tensor_view(metal_graph_batch_attn_cur(g),
                            0,
                            (uint64_t)n_tokens * DS4_N_EMBD * sizeof(float));
    bool ok = hc_mix_view && hc_split_view && attn_cur_view;

    const float freq_base = DS4_ROPE_FREQ_BASE;
    const float freq_scale = 1.0f;
    const float ext_factor = 0.0f;
    const float attn_factor = 1.0f;

    if (ok && !commands_open) ok = ds4_gpu_begin_commands() != 0;
    if (ok) ok = ds4_gpu_rms_norm_plain_rows_tensor(metal_graph_batch_flat_hc(g),
                                                      metal_graph_batch_cur_hc(g),
                                                      (uint32_t)hc_dim,
                                                      n_tokens,
                                                      DS4_RMS_EPS) != 0;
    if (ok) ok = metal_graph_matmul_plain_tensor(hc_mix_view,
                                                 dspark_model,
                                                 block->hc_attn_fn,
                                                 hc_dim,
                                                 mix_hc,
                                                 metal_graph_batch_flat_hc(g),
                                                 n_tokens);
    if (fuse_hc_norm) {
        if (ok) ok = ds4_gpu_hc_split_weighted_sum_norm_tensor(attn_cur_view,
                                                                 metal_graph_batch_attn_norm(g),
                                                                 hc_split_view,
                                                                 hc_mix_view,
                                                                 metal_graph_batch_cur_hc(g),
                                                                 dspark_model->map,
                                                                 dspark_model->size,
                                                                 block->hc_attn_scale->abs_offset,
                                                                 block->hc_attn_base->abs_offset,
                                                                 block->attn_norm->abs_offset,
                                                                 DS4_N_EMBD,
                                                                 DS4_N_HC,
                                                                 DS4_N_HC_SINKHORN_ITER,
                                                                 DS4_HC_EPS,
                                                                 DS4_RMS_EPS) != 0;
    } else {
        if (ok) ok = ds4_gpu_hc_split_weighted_sum_tensor(attn_cur_view,
                                                            hc_split_view,
                                                            hc_mix_view,
                                                            metal_graph_batch_cur_hc(g),
                                                            dspark_model->map,
                                                            dspark_model->size,
                                                            block->hc_attn_scale->abs_offset,
                                                            block->hc_attn_base->abs_offset,
                                                            DS4_N_EMBD,
                                                            DS4_N_HC,
                                                            DS4_N_HC_SINKHORN_ITER,
                                                            DS4_HC_EPS) != 0;
        if (ok) ok = ds4_gpu_rms_norm_weight_rows_tensor(metal_graph_batch_attn_norm(g),
                                                          metal_graph_batch_attn_cur(g),
                                                          dspark_model->map,
                                                          dspark_model->size,
                                                          block->attn_norm->abs_offset,
                                                          DS4_N_EMBD,
                                                          n_tokens,
                                                          DS4_RMS_EPS) != 0;
    }
    if (ok) ok = metal_graph_matmul_plain_tensor(metal_graph_batch_kv_raw(g),
                                                 dspark_model,
                                                 block->attn_kv,
                                                 DS4_N_EMBD,
                                                 DS4_N_HEAD_DIM,
                                                 metal_graph_batch_attn_norm(g),
                                                 n_tokens);
    if (ok) ok = ds4_gpu_rms_norm_weight_rows_tensor(metal_graph_batch_kv(g),
                                                      metal_graph_batch_kv_raw(g),
                                                      dspark_model->map,
                                                      dspark_model->size,
                                                      block->attn_kv_a_norm->abs_offset,
                                                      DS4_N_HEAD_DIM,
                                                      n_tokens,
                                                      DS4_RMS_EPS) != 0;
    if (ok) ok = ds4_gpu_rope_tail_tensor(metal_graph_batch_kv(g),
                                           n_tokens,
                                           1,
                                           DS4_N_HEAD_DIM,
                                           DS4_N_ROT,
                                           pos0,
                                           0,
                                           false,
                                           freq_base,
                                           freq_scale,
                                           ext_factor,
                                           attn_factor,
                                           DS4_ROPE_YARN_BETA_FAST,
                                           DS4_ROPE_YARN_BETA_SLOW) != 0;
    if (ok) ok = ds4_gpu_dsv4_fp8_kv_quantize_tensor(metal_graph_batch_kv(g),
                                                       n_tokens,
                                                       DS4_N_HEAD_DIM,
                                                       DS4_N_ROT) != 0;
    if (ok) ok = ds4_gpu_store_raw_kv_batch_tensor(g->dspark_raw_cache[stage],
                                                    metal_graph_batch_kv(g),
                                                    g->dspark_cache_cap,
                                                    pos0,
                                                    n_tokens,
                                                    DS4_N_HEAD_DIM) != 0;
    if (ok && !commands_open) ok = ds4_gpu_end_commands() != 0;

    ds4_gpu_tensor_free(attn_cur_view);
    ds4_gpu_tensor_free(hc_split_view);
    ds4_gpu_tensor_free(hc_mix_view);
    if (!ok && !commands_open) (void)ds4_gpu_synchronize();
    return ok;
}

static bool metal_graph_seed_dspark_initial_cache_from_prefill(
        ds4_gpu_graph            *g,
        const ds4_model          *dspark_model,
        const ds4_dspark_weights *dw,
        uint32_t                  batch_start,
        uint32_t                  n_tokens,
        uint32_t                 *seeded_rows) {
    if (seeded_rows) *seeded_rows = 0;
    if (!g || !dspark_model || !dw ||
        n_tokens == 0 ||
        n_tokens > g->prefill_cap ||
        n_tokens > g->dspark_cache_cap ||
        dw->n_stages == 0 ||
        dw->n_stages > DS4_DSPARK_MAX_STAGES ||
        !dspark_stage0_batch_ready(g, dw, n_tokens) ||
        !dspark_stage_cache_ready(g, dw)) {
        return false;
    }

    bool ok = ds4_gpu_begin_commands() != 0;
    if (ok) {
        ok = metal_graph_eval_dspark_stage0_batch(g,
                                                  dspark_model,
                                                  dw,
                                                  n_tokens,
                                                  true);
    }
    if (ok) {
        ok = ds4_gpu_repeat_hc_rows_tensor(metal_graph_batch_cur_hc(g),
                                           metal_graph_batch_ffn_norm(g),
                                           n_tokens,
                                           DS4_N_EMBD,
                                           DS4_N_HC) != 0;
    }
    for (uint32_t stage = 0; ok && stage < dw->n_stages; stage++) {
        ok = metal_graph_seed_dspark_stage_target_cache(g,
                                                        dspark_model,
                                                        dw,
                                                        stage,
                                                        batch_start,
                                                        n_tokens,
                                                        true);
    }
    if (ok) ok = ds4_gpu_end_commands() != 0;
    if (!ok) {
        (void)ds4_gpu_synchronize();
        return false;
    }
    if (!metal_graph_dspark_cache_set_window(g, batch_start, n_tokens)) {
        return false;
    }
    if (seeded_rows) *seeded_rows = n_tokens;
    return true;
}

static bool metal_graph_encode_dspark_next_stage_draft_input_from(
        ds4_gpu_graph            *g,
        const ds4_dspark_weights *dw,
        const ds4_gpu_tensor     *draft_hc) {
    if (!g || !dw || !dspark_stage_input_ready(g, dw) ||
        !draft_hc) {
        return false;
    }

    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const uint64_t hc_bytes = hc_dim * sizeof(float);
    if (ds4_gpu_tensor_bytes(draft_hc) <
        (uint64_t)dw->block_size * hc_bytes) {
        return false;
    }

    return ds4_gpu_tensor_copy(g->dspark_stage_input_hc,
                               hc_bytes,
                               draft_hc,
                               0,
                               (uint64_t)dw->block_size * hc_bytes) != 0;
}

static bool metal_graph_profile_layer_env_match(const char *env_name, uint32_t il) {
    const char *layer_env = getenv(env_name);
    if (!layer_env || !layer_env[0]) return true;

    char *end = NULL;
    const unsigned long layer = strtoul(layer_env, &end, 10);
    return end != layer_env &&
           *end == '\0' &&
           layer <= UINT32_MAX &&
           (uint32_t)layer == il;
}

static bool metal_graph_dspark_stage_profile_enabled(uint32_t stage) {
    return getenv("DS4_DSPARK_STAGE_PROFILE") != NULL &&
           metal_graph_profile_layer_env_match("DS4_DSPARK_STAGE_PROFILE_STAGE",
                                               stage);
}

static bool metal_graph_dspark_stage_profile_boundary(
        const char *part,
        uint32_t    stage,
        uint32_t    pos,
        uint32_t    rows,
        double     *stage_t0) {
    if (ds4_gpu_end_commands() == 0) return false;
    const double now = now_sec();
    fprintf(stderr,
            "ds4: DSpark stage profile stage=%u pos=%u rows=%u %s=%.3f ms\n",
            stage,
            pos,
            rows,
            part,
            (now - *stage_t0) * 1000.0);
    *stage_t0 = now;
    return ds4_gpu_begin_commands() != 0;
}

static bool metal_graph_eval_dspark_stage_block(
        ds4_gpu_graph            *g,
        const ds4_model          *dspark_model,
        const ds4_dspark_weights *dw,
        uint32_t                  stage,
        uint32_t                  pos,
        uint32_t                  support_len,
        uint32_t                  raw_start,
        bool                      prepare_next_stage_input,
        bool                      commands_open) {
    if (!g || !dspark_model || !dw ||
        !dspark_stage_block_ready(g, dw, stage)) {
        return false;
    }

    const uint32_t draft = dw->block_size;
    const uint32_t rows = draft + 1u;
    if (support_len > g->dspark_cache_cap ||
        rows > g->dspark_cache_cap - support_len ||
        (support_len != 0 && raw_start >= g->dspark_cache_cap)) {
        return false;
    }
    const uint32_t visible_rows = support_len + rows;
    const uint32_t attention_raw_start =
        support_len ? raw_start : (pos % g->dspark_cache_cap);
    const uint32_t append_pos = support_len ?
        (uint32_t)(((uint64_t)raw_start + support_len) %
                   g->dspark_cache_cap) :
        attention_raw_start;
    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const uint64_t mix_hc = 2ull * DS4_N_HC + (uint64_t)DS4_N_HC * DS4_N_HC;
    const uint64_t q_dim = (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM;
    const uint64_t group_dim =
        (uint64_t)DS4_N_HEAD_DIM * (DS4_N_HEAD / DS4_N_OUT_GROUP);
    const ds4_layer_weights *block = &dw->stage[stage].block;
    const bool fuse_hc_norm = DS4_N_HC == 4 &&
                              !metal_graph_use_reference_hc_decode() &&
                              metal_graph_enable_batch_hc_norm_fusion();

    ds4_gpu_tensor *hc_mix_view =
        ds4_gpu_tensor_view(metal_graph_batch_hc_mix(g),
                            0,
                            (uint64_t)rows * mix_hc * sizeof(float));
    ds4_gpu_tensor *hc_split_view =
        ds4_gpu_tensor_view(metal_graph_batch_hc_split(g),
                            0,
                            (uint64_t)rows * mix_hc * sizeof(float));
    ds4_gpu_tensor *attn_cur_view =
        ds4_gpu_tensor_view(metal_graph_batch_attn_cur(g),
                            0,
                            (uint64_t)rows * DS4_N_EMBD * sizeof(float));
    ds4_gpu_tensor *draft_attn_norm_view =
        ds4_gpu_tensor_view(metal_graph_batch_attn_norm(g),
                            (uint64_t)DS4_N_EMBD * sizeof(float),
                            (uint64_t)draft * DS4_N_EMBD * sizeof(float));
    ds4_gpu_tensor *draft_hc_view =
        ds4_gpu_tensor_view(g->dspark_stage_input_hc,
                            hc_dim * sizeof(float),
                            (uint64_t)draft * hc_dim * sizeof(float));
    ds4_gpu_tensor *draft_hc_split_view =
        ds4_gpu_tensor_view(metal_graph_batch_hc_split(g),
                            mix_hc * sizeof(float),
                            (uint64_t)draft * mix_hc * sizeof(float));
    ds4_gpu_tensor *kv_target_view =
        ds4_gpu_tensor_view(metal_graph_batch_kv(g),
                            0,
                            (uint64_t)DS4_N_HEAD_DIM * sizeof(float));
    ds4_gpu_tensor *kv_draft_view =
        ds4_gpu_tensor_view(metal_graph_batch_kv(g),
                            (uint64_t)DS4_N_HEAD_DIM * sizeof(float),
                            (uint64_t)draft * DS4_N_HEAD_DIM * sizeof(float));
    ds4_gpu_tensor *after_attn_hc_view =
        ds4_gpu_tensor_view(metal_graph_batch_after_attn_hc(g),
                            0,
                            (uint64_t)draft * hc_dim * sizeof(float));

    bool ok = hc_mix_view && hc_split_view && attn_cur_view &&
              draft_attn_norm_view && draft_hc_view &&
              draft_hc_split_view && kv_target_view && kv_draft_view &&
              after_attn_hc_view;
    const bool saved_streaming = g->ssd_streaming;
    g->ssd_streaming = false;

    const float freq_base = DS4_ROPE_FREQ_BASE;
    const float freq_scale = 1.0f;
    const float ext_factor = 0.0f;
    const float attn_factor = 1.0f;
    const bool stage_profile =
        metal_graph_dspark_stage_profile_enabled(stage);
    double stage_t0 = stage_profile ? now_sec() : 0.0;
#define DS4_DSPARK_PROFILE_STAGE(part_) do {                               \
        if (ok && stage_profile) {                                          \
            ok = metal_graph_dspark_stage_profile_boundary((part_),         \
                                                           stage,           \
                                                           pos,             \
                                                           rows,            \
                                                           &stage_t0);      \
        }                                                                   \
    } while (0)

    if (ok && !commands_open) ok = ds4_gpu_begin_commands() != 0;
    if (ok) ok = ds4_gpu_rms_norm_plain_rows_tensor(metal_graph_batch_flat_hc(g),
                                                      g->dspark_stage_input_hc,
                                                      (uint32_t)hc_dim,
                                                      rows,
                                                      DS4_RMS_EPS) != 0;
    if (ok) ok = metal_graph_matmul_plain_tensor(hc_mix_view,
                                                 dspark_model,
                                                 block->hc_attn_fn,
                                                 hc_dim,
                                                 mix_hc,
                                                 metal_graph_batch_flat_hc(g),
                                                 rows);
    DS4_DSPARK_PROFILE_STAGE("attn_hc_pre");
    if (fuse_hc_norm) {
        if (ok) ok = ds4_gpu_hc_split_weighted_sum_norm_tensor(attn_cur_view,
                                                                 metal_graph_batch_attn_norm(g),
                                                                 hc_split_view,
                                                                 hc_mix_view,
                                                                 g->dspark_stage_input_hc,
                                                                 dspark_model->map,
                                                                 dspark_model->size,
                                                                 block->hc_attn_scale->abs_offset,
                                                                 block->hc_attn_base->abs_offset,
                                                                 block->attn_norm->abs_offset,
                                                                 DS4_N_EMBD,
                                                                 DS4_N_HC,
                                                                 DS4_N_HC_SINKHORN_ITER,
                                                                 DS4_HC_EPS,
                                                                 DS4_RMS_EPS) != 0;
    } else {
        if (ok) ok = ds4_gpu_hc_split_weighted_sum_tensor(attn_cur_view,
                                                            hc_split_view,
                                                            hc_mix_view,
                                                            g->dspark_stage_input_hc,
                                                            dspark_model->map,
                                                            dspark_model->size,
                                                            block->hc_attn_scale->abs_offset,
                                                            block->hc_attn_base->abs_offset,
                                                            DS4_N_EMBD,
                                                            DS4_N_HC,
                                                            DS4_N_HC_SINKHORN_ITER,
                                                            DS4_HC_EPS) != 0;
        if (ok) ok = ds4_gpu_rms_norm_weight_rows_tensor(metal_graph_batch_attn_norm(g),
                                                          metal_graph_batch_attn_cur(g),
                                                          dspark_model->map,
                                                          dspark_model->size,
                                                          block->attn_norm->abs_offset,
                                                          DS4_N_EMBD,
                                                          rows,
                                                          DS4_RMS_EPS) != 0;
    }
    DS4_DSPARK_PROFILE_STAGE("attn_norm");

    if (ok) ok = metal_graph_matmul_plain_tensor(metal_graph_batch_qr(g),
                                                 dspark_model,
                                                 block->attn_q_a,
                                                 DS4_N_EMBD,
                                                 DS4_N_LORA_Q,
                                                 draft_attn_norm_view,
                                                 draft);
    if (ok) ok = ds4_gpu_rms_norm_weight_rows_tensor(metal_graph_batch_qr_norm(g),
                                                      metal_graph_batch_qr(g),
                                                      dspark_model->map,
                                                      dspark_model->size,
                                                      block->attn_q_a_norm->abs_offset,
                                                      DS4_N_LORA_Q,
                                                      draft,
                                                      DS4_RMS_EPS) != 0;
    if (ok) ok = metal_graph_matmul_plain_tensor(metal_graph_batch_q(g),
                                                 dspark_model,
                                                 block->attn_q_b,
                                                 DS4_N_LORA_Q,
                                                 q_dim,
                                                 metal_graph_batch_qr_norm(g),
                                                 draft);
    if (ok) ok = ds4_gpu_head_rms_norm_tensor(metal_graph_batch_q(g),
                                               draft,
                                               DS4_N_HEAD,
                                               DS4_N_HEAD_DIM,
                                               DS4_RMS_EPS) != 0;
    if (ok) ok = ds4_gpu_rope_tail_tensor(metal_graph_batch_q(g),
                                           draft,
                                           DS4_N_HEAD,
                                           DS4_N_HEAD_DIM,
                                           DS4_N_ROT,
                                           pos,
                                           0,
                                           false,
                                           freq_base,
                                           freq_scale,
                                           ext_factor,
                                           attn_factor,
                                           DS4_ROPE_YARN_BETA_FAST,
                                           DS4_ROPE_YARN_BETA_SLOW) != 0;
    DS4_DSPARK_PROFILE_STAGE("q_path");

    if (ok) ok = metal_graph_matmul_plain_tensor(metal_graph_batch_kv_raw(g),
                                                 dspark_model,
                                                 block->attn_kv,
                                                 DS4_N_EMBD,
                                                 DS4_N_HEAD_DIM,
                                                 metal_graph_batch_attn_norm(g),
                                                 rows);
    if (ok) ok = ds4_gpu_rms_norm_weight_rows_tensor(metal_graph_batch_kv(g),
                                                      metal_graph_batch_kv_raw(g),
                                                      dspark_model->map,
                                                      dspark_model->size,
                                                      block->attn_kv_a_norm->abs_offset,
                                                      DS4_N_HEAD_DIM,
                                                      rows,
                                                      DS4_RMS_EPS) != 0;
    if (ok) ok = ds4_gpu_rope_tail_tensor(kv_target_view,
                                           1,
                                           1,
                                           DS4_N_HEAD_DIM,
                                           DS4_N_ROT,
                                           pos,
                                           0,
                                           false,
                                           freq_base,
                                           freq_scale,
                                           ext_factor,
                                           attn_factor,
                                           DS4_ROPE_YARN_BETA_FAST,
                                           DS4_ROPE_YARN_BETA_SLOW) != 0;
    if (ok) ok = ds4_gpu_rope_tail_tensor(kv_draft_view,
                                           draft,
                                           1,
                                           DS4_N_HEAD_DIM,
                                           DS4_N_ROT,
                                           pos,
                                           0,
                                           false,
                                           freq_base,
                                           freq_scale,
                                           ext_factor,
                                           attn_factor,
                                           DS4_ROPE_YARN_BETA_FAST,
                                           DS4_ROPE_YARN_BETA_SLOW) != 0;
    if (ok) ok = ds4_gpu_dsv4_fp8_kv_quantize_tensor(metal_graph_batch_kv(g),
                                                       rows,
                                                       DS4_N_HEAD_DIM,
                                                       DS4_N_ROT) != 0;
    if (ok) ok = ds4_gpu_store_raw_kv_batch_tensor(g->dspark_raw_cache[stage],
                                                    metal_graph_batch_kv(g),
                                                    g->dspark_cache_cap,
                                                    append_pos,
                                                    rows,
                                                    DS4_N_HEAD_DIM) != 0;
    DS4_DSPARK_PROFILE_STAGE("kv_path");

    if (ok) ok = ds4_gpu_attention_noncausal_raw_batch_heads_tensor(
                    metal_graph_batch_heads(g),
                    dspark_model->map,
                    dspark_model->size,
                    block->attn_sinks->abs_offset,
                    metal_graph_batch_q(g),
                    g->dspark_raw_cache[stage],
                    draft,
                    visible_rows,
                    g->dspark_cache_cap,
                    attention_raw_start,
                    DS4_N_HEAD,
                    DS4_N_HEAD_DIM) != 0;
    if (ok) ok = ds4_gpu_rope_tail_tensor(metal_graph_batch_heads(g),
                                           draft,
                                           DS4_N_HEAD,
                                           DS4_N_HEAD_DIM,
                                           DS4_N_ROT,
                                           pos,
                                           0,
                                           true,
                                           freq_base,
                                           freq_scale,
                                           ext_factor,
                                           attn_factor,
                                           DS4_ROPE_YARN_BETA_FAST,
                                           DS4_ROPE_YARN_BETA_SLOW) != 0;
    DS4_DSPARK_PROFILE_STAGE("attention");
    if (ok) ok = ds4_gpu_attention_output_q8_batch_tensor(
                    metal_graph_batch_attn_out(g),
                    metal_graph_batch_attn_low(g),
                    metal_graph_batch_group_tmp(g),
                    metal_graph_batch_low_tmp(g),
                    dspark_model->map,
                    dspark_model->size,
                    block->attn_output_a->abs_offset,
                    block->attn_output_b->abs_offset,
                    group_dim,
                    DS4_N_LORA_O,
                    DS4_N_OUT_GROUP,
                    DS4_N_EMBD,
                    metal_graph_batch_heads(g),
                    draft) != 0;
    if (ok) ok = ds4_gpu_hc_expand_split_tensor(after_attn_hc_view,
                                                 metal_graph_batch_attn_out(g),
                                                 draft_hc_view,
                                                 draft_hc_split_view,
                                                 DS4_N_EMBD,
                                                 DS4_N_HC) != 0;
    DS4_DSPARK_PROFILE_STAGE("attn_output_hc");

    if (ok) ok = metal_graph_encode_layer_ffn_batch(g,
                                                     dspark_model,
                                                     block,
                                                     stage,
                                                     pos,
                                                     draft,
                                                     NULL,
                                                     0);
    DS4_DSPARK_PROFILE_STAGE("ffn");
    if (ok &&
        !prepare_next_stage_input &&
        getenv("DS4_DSPARK_DISABLE_FINAL_OUTPUT_ALIAS") != NULL) {
        ok = ds4_gpu_tensor_copy(g->dspark_stage_output_hc,
                                  0,
                                  metal_graph_batch_next_hc(g),
                                  0,
                                  (uint64_t)draft * hc_dim * sizeof(float)) != 0;
    }
    DS4_DSPARK_PROFILE_STAGE("copy_output");
    if (ok && prepare_next_stage_input) {
        ok = metal_graph_encode_dspark_next_stage_draft_input_from(
                g, dw, metal_graph_batch_next_hc(g));
    }
    DS4_DSPARK_PROFILE_STAGE("next_input");
    if (ok && !commands_open) ok = ds4_gpu_end_commands() != 0;
    g->ssd_streaming = saved_streaming;

    ds4_gpu_tensor_free(after_attn_hc_view);
    ds4_gpu_tensor_free(kv_draft_view);
    ds4_gpu_tensor_free(kv_target_view);
    ds4_gpu_tensor_free(draft_hc_split_view);
    ds4_gpu_tensor_free(draft_hc_view);
    ds4_gpu_tensor_free(draft_attn_norm_view);
    ds4_gpu_tensor_free(attn_cur_view);
    ds4_gpu_tensor_free(hc_split_view);
    ds4_gpu_tensor_free(hc_mix_view);
    if (!ok && !commands_open) (void)ds4_gpu_synchronize();
#undef DS4_DSPARK_PROFILE_STAGE
    return ok;
}

static bool metal_graph_eval_dspark_stage_chain(
        ds4_gpu_graph            *g,
        const ds4_model          *dspark_model,
        const ds4_dspark_weights *dw,
        uint32_t                  pos,
        uint32_t                 *completed_stages,
        uint32_t                 *cache_start_out,
        uint32_t                 *cache_rows_out) {
    if (completed_stages) *completed_stages = 0;
    if (cache_start_out) *cache_start_out = 0;
    if (cache_rows_out) *cache_rows_out = 0;
    if (!g || !dspark_model || !dw ||
        dw->n_stages == 0 ||
        dw->n_stages > DS4_DSPARK_MAX_STAGES ||
        !dspark_stage_input_ready(g, dw) ||
        !dspark_stage_cache_ready(g, dw) ||
        !metal_graph_prefill_tokens(g) ||
        !g->dspark_draft_tokens) {
        return false;
    }

    const uint32_t rows = dw->block_size + 1u;
    const uint32_t support_len = g->dspark_cache_len;
    const uint32_t raw_start = support_len ? g->dspark_cache_start : 0;
    if (support_len > g->dspark_cache_cap ||
        rows > g->dspark_cache_cap - support_len ||
        (support_len != 0 && raw_start >= g->dspark_cache_cap) ||
        !metal_graph_dspark_cache_ends_at(g, pos)) {
        return false;
    }
    if (cache_start_out) {
        *cache_start_out = support_len ? raw_start :
            (pos % g->dspark_cache_cap);
    }
    if (cache_rows_out) *cache_rows_out = support_len + rows;

    for (uint32_t stage = 0; stage < dw->n_stages; stage++) {
        if (!dspark_stage_block_ready(g, dw, stage)) return false;
    }

    /* The support model runs only on the coordinator. Its generic layer
     * helpers share the base graph object, so temporarily disarm TP or they
     * would encode expert gates that the worker can never reach. The base
     * model's later verification restores and uses the normal 50/50 split. */
    const uint32_t saved_tp_world = g->tp_world;
    const uint32_t saved_tp_batch_rows = g->tp_batch_rows;
    g->tp_world = 0;
    g->tp_batch_rows = 0;
    const bool suspended_expert_sharding = saved_tp_world == 2;
    if (suspended_expert_sharding) {
        ds4_gpu_tp_suspend_expert_sharding(1);
    }

    bool ok = ds4_gpu_begin_commands() != 0;
    if (ok) {
        ok = ds4_gpu_tensor_copy(metal_graph_prefill_tokens(g),
                                  0,
                                  g->dspark_draft_tokens,
                                  0,
                                  (uint64_t)dw->block_size * sizeof(int32_t)) != 0;
    }
    for (uint32_t stage = 0; ok && stage < dw->n_stages; stage++) {
        const bool stage_ok =
            metal_graph_eval_dspark_stage_block(g,
                                                dspark_model,
                                                dw,
                                                stage,
                                                pos,
                                                support_len,
                                                raw_start,
                                                stage + 1u < dw->n_stages,
                                                true);
        if (!stage_ok) {
            ok = false;
            break;
        }
        if (completed_stages) *completed_stages = stage + 1u;
    }
    if (ok) ok = ds4_gpu_end_commands() != 0;
    if (suspended_expert_sharding) {
        ds4_gpu_tp_suspend_expert_sharding(0);
    }
    g->tp_world = saved_tp_world;
    g->tp_batch_rows = saved_tp_batch_rows;
    if (!ok) {
        (void)ds4_gpu_synchronize();
        return false;
    }
    return true;
}

/* Keep the support KV ring aligned while the scheduler skips proposals. */
static bool metal_graph_dspark_ring_maintain(
        ds4_gpu_graph            *g,
        const ds4_model          *dspark_model,
        const ds4_dspark_weights *dw,
        uint32_t                  pos) {
    if (!g || !dspark_model || !dw ||
        !g->dspark_capture_valid ||
        g->dspark_cache_len == 0 ||
        !metal_graph_dspark_cache_ends_at(g, pos) ||
        !dspark_stage0_weights_ready(g, dw) ||
        !dspark_stage_cache_ready(g, dw) ||
        !metal_graph_batch_kv_raw(g) || !metal_graph_batch_kv(g)) {
        return false;
    }
    for (uint32_t stage = 0; stage < dw->n_stages; stage++) {
        if (!dspark_stage_block_ready(g, dw, stage)) return false;
    }

    const ds4_dspark_stage_weights *stage0 = &dw->stage[0];
    const uint64_t in_dim = (uint64_t)dw->target_layer_count * DS4_N_EMBD;
    ds4_gpu_tensor *kv_raw_view =
        ds4_gpu_tensor_view(metal_graph_batch_kv_raw(g),
                            0,
                            (uint64_t)DS4_N_HEAD_DIM * sizeof(float));
    ds4_gpu_tensor *kv_view =
        ds4_gpu_tensor_view(metal_graph_batch_kv(g),
                            0,
                            (uint64_t)DS4_N_HEAD_DIM * sizeof(float));
    bool ok = kv_raw_view && kv_view && ds4_gpu_begin_commands() != 0;
    if (ok) {
        ok = metal_graph_matmul_plain_tensor(g->dspark_stage0_proj,
                                             dspark_model,
                                             stage0->main_proj,
                                             in_dim,
                                             DS4_N_EMBD,
                                             g->dspark_target_hidden,
                                             1);
    }
    if (ok) {
        ok = ds4_gpu_rms_norm_weight_tensor(g->dspark_main_x,
                                            g->dspark_stage0_proj,
                                            dspark_model->map,
                                            dspark_model->size,
                                            stage0->main_norm->abs_offset,
                                            DS4_N_EMBD,
                                            DS4_RMS_EPS) != 0;
    }
    for (uint32_t stage = 0; ok && stage < dw->n_stages; stage++) {
        const ds4_layer_weights *block = &dw->stage[stage].block;
        ok = metal_graph_matmul_plain_tensor(kv_raw_view,
                                             dspark_model,
                                             block->attn_kv,
                                             DS4_N_EMBD,
                                             DS4_N_HEAD_DIM,
                                             g->dspark_main_x,
                                             1);
        if (ok) ok = ds4_gpu_rms_norm_weight_rows_tensor(
                         kv_view,
                         kv_raw_view,
                         dspark_model->map,
                         dspark_model->size,
                         block->attn_kv_a_norm->abs_offset,
                         DS4_N_HEAD_DIM,
                         1,
                         DS4_RMS_EPS) != 0;
        if (ok) ok = ds4_gpu_rope_tail_tensor(kv_view,
                                               1,
                                               1,
                                               DS4_N_HEAD_DIM,
                                               DS4_N_ROT,
                                               pos,
                                               0,
                                               false,
                                               DS4_ROPE_FREQ_BASE,
                                               1.0f,
                                               0.0f,
                                               1.0f,
                                               DS4_ROPE_YARN_BETA_FAST,
                                               DS4_ROPE_YARN_BETA_SLOW) != 0;
        if (ok) ok = ds4_gpu_dsv4_fp8_kv_quantize_tensor(kv_view,
                                                          1,
                                                          DS4_N_HEAD_DIM,
                                                          DS4_N_ROT) != 0;
        if (ok) ok = ds4_gpu_store_raw_kv_batch_tensor(
                         g->dspark_raw_cache[stage],
                         kv_view,
                         g->dspark_cache_cap,
                         pos,
                         1,
                         DS4_N_HEAD_DIM) != 0;
    }
    if (ok) ok = ds4_gpu_end_commands() != 0;
    else (void)ds4_gpu_synchronize();
    ds4_gpu_tensor_free(kv_view);
    ds4_gpu_tensor_free(kv_raw_view);
    if (ok) (void)metal_graph_dspark_cache_claim_appended_row(g, pos);
    return ok;
}

static ds4_gpu_tensor *metal_graph_dspark_final_output_hc(const ds4_gpu_graph *g) {
    if (!g) return NULL;
    if (getenv("DS4_DSPARK_DISABLE_FINAL_OUTPUT_ALIAS") == NULL &&
        metal_graph_batch_next_hc(g)) {
        return metal_graph_batch_next_hc(g);
    }
    return g->dspark_stage_output_hc;
}

static bool dspark_final_head_ready(
        const ds4_gpu_graph      *g,
        const ds4_weights        *base_weights,
        const ds4_dspark_weights *dw) {
    if (!g || !base_weights || !dw ||
        dw->n_stages == 0 ||
        dw->n_stages > DS4_DSPARK_MAX_STAGES ||
        dw->block_size == 0 ||
        dw->block_size > DS4_DSPARK_MAX_BLOCK_SIZE ||
        !base_weights->output ||
        !metal_graph_dspark_final_output_hc(g) ||
        !metal_graph_batch_hc_mix(g) ||
        !metal_graph_batch_hc_split(g) ||
        !metal_graph_batch_flat_hc(g) ||
        !metal_graph_batch_ffn_cur(g) ||
        !metal_graph_batch_ffn_norm(g) ||
        !g->spec_logits) {
        return false;
    }

    const ds4_dspark_stage_weights *final =
        &dw->stage[dw->n_stages - 1u];
    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const uint64_t draft = dw->block_size;
    const uint64_t vocab_dim = base_weights->output->dim[1];
    if (!final->norm ||
        !final->hc_head_base ||
        !final->hc_head_fn ||
        !final->hc_head_scale ||
        final->norm->type != DS4_TENSOR_F32 ||
        final->hc_head_base->type != DS4_TENSOR_F32 ||
        !dspark_tensor_type_matches(final->hc_head_fn->type,
                                    DS4_DSPARK_LAYOUT_PLAIN) ||
        final->hc_head_scale->type != DS4_TENSOR_F32 ||
        !tensor_type_is_dense_quant(base_weights->output->type)) {
        return false;
    }

    return final->norm->ndim == 1 &&
           final->norm->dim[0] == DS4_N_EMBD &&
           final->hc_head_base->ndim == 1 &&
           final->hc_head_base->dim[0] == DS4_N_HC &&
           final->hc_head_fn->ndim == 2 &&
           final->hc_head_fn->dim[0] == hc_dim &&
           final->hc_head_fn->dim[1] == DS4_N_HC &&
           final->hc_head_scale->ndim == 1 &&
           final->hc_head_scale->dim[0] == 1 &&
           base_weights->output->ndim == 2 &&
           base_weights->output->dim[0] == DS4_N_EMBD &&
           vocab_dim == DS4_N_VOCAB &&
           ds4_gpu_tensor_bytes(
                   metal_graph_dspark_final_output_hc(g)) >=
               draft * hc_dim * sizeof(float) &&
           ds4_gpu_tensor_bytes(metal_graph_batch_flat_hc(g)) >=
               draft * hc_dim * sizeof(float) &&
           ds4_gpu_tensor_bytes(metal_graph_batch_hc_mix(g)) >=
               draft * DS4_N_HC * sizeof(float) &&
           ds4_gpu_tensor_bytes(metal_graph_batch_hc_split(g)) >=
               draft * DS4_N_HC * sizeof(float) &&
           ds4_gpu_tensor_bytes(metal_graph_batch_ffn_cur(g)) >=
               draft * DS4_N_EMBD * sizeof(float) &&
           ds4_gpu_tensor_bytes(metal_graph_batch_ffn_norm(g)) >=
               draft * DS4_N_EMBD * sizeof(float) &&
           ds4_gpu_tensor_bytes(g->spec_logits) >=
               draft * vocab_dim * sizeof(float);
}

static bool metal_graph_eval_dspark_base_logits(
        ds4_gpu_graph            *g,
        const ds4_model          *base_model,
        const ds4_weights        *base_weights,
        const ds4_model          *dspark_model,
        const ds4_dspark_weights *dw) {
    if (!g || !base_model || !base_weights || !dspark_model || !dw ||
        !dspark_final_head_ready(g, base_weights, dw)) {
        return false;
    }

    const ds4_dspark_stage_weights *final =
        &dw->stage[dw->n_stages - 1u];
    const uint32_t draft = dw->block_size;
    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const uint64_t vocab_dim = base_weights->output->dim[1];
    ds4_gpu_tensor *stage_output_hc = metal_graph_dspark_final_output_hc(g);
    ds4_gpu_tensor *output_pre =
        ds4_gpu_tensor_view(metal_graph_batch_hc_mix(g),
                            0,
                            (uint64_t)draft * DS4_N_HC * sizeof(float));
    ds4_gpu_tensor *output_weights =
        ds4_gpu_tensor_view(metal_graph_batch_hc_split(g),
                            0,
                            (uint64_t)draft * DS4_N_HC * sizeof(float));
    ds4_gpu_tensor *output_embd =
        ds4_gpu_tensor_view(metal_graph_batch_ffn_cur(g),
                            0,
                            (uint64_t)draft * DS4_N_EMBD * sizeof(float));
    ds4_gpu_tensor *output_norm =
        ds4_gpu_tensor_view(metal_graph_batch_ffn_norm(g),
                            0,
                            (uint64_t)draft * DS4_N_EMBD * sizeof(float));
    ds4_gpu_tensor *logits =
        ds4_gpu_tensor_view(g->spec_logits,
                            0,
                            (uint64_t)draft * vocab_dim * sizeof(float));

    bool ok = stage_output_hc && output_pre && output_weights && output_embd &&
              output_norm && logits;
    if (ok) ok = ds4_gpu_begin_commands() != 0;
    if (ok) ok = ds4_gpu_rms_norm_plain_rows_tensor(metal_graph_batch_flat_hc(g),
                                                     stage_output_hc,
                                                     (uint32_t)hc_dim,
                                                     draft,
                                                     DS4_RMS_EPS) != 0;
    if (ok) ok = metal_graph_matmul_plain_tensor(output_pre,
                                                 dspark_model,
                                                 final->hc_head_fn,
                                                 hc_dim,
                                                 DS4_N_HC,
                                                 metal_graph_batch_flat_hc(g),
                                                 draft);
    if (ok) ok = ds4_gpu_output_hc_weights_tensor(output_weights,
                                                   output_pre,
                                                   dspark_model->map,
                                                   dspark_model->size,
                                                   final->hc_head_scale->abs_offset,
                                                   final->hc_head_base->abs_offset,
                                                   DS4_N_HC,
                                                   DS4_HC_EPS) != 0;
    if (ok) ok = ds4_gpu_hc_weighted_sum_tensor(output_embd,
                                                 stage_output_hc,
                                                 output_weights,
                                                 DS4_N_EMBD,
                                                 DS4_N_HC) != 0;
    if (ok) ok = ds4_gpu_rms_norm_weight_rows_tensor(output_norm,
                                                      output_embd,
                                                      dspark_model->map,
                                                      dspark_model->size,
                                                      final->norm->abs_offset,
                                                      DS4_N_EMBD,
                                                      draft,
                                                      DS4_RMS_EPS) != 0;
    if (ok) ok = metal_graph_matmul_plain_tensor(logits,
                                                  base_model,
                                                  base_weights->output,
                                                  DS4_N_EMBD,
                                                  vocab_dim,
                                                  output_norm,
                                                  draft);
    if (ok) ok = ds4_gpu_end_commands() != 0;
    if (!ok) (void)ds4_gpu_synchronize();

    ds4_gpu_tensor_free(logits);
    ds4_gpu_tensor_free(output_norm);
    ds4_gpu_tensor_free(output_embd);
    ds4_gpu_tensor_free(output_weights);
    ds4_gpu_tensor_free(output_pre);
    return ok;
}

static bool metal_graph_eval_dspark_final_hidden(
        ds4_gpu_graph            *g,
        const ds4_model          *dspark_model,
        const ds4_dspark_weights *dw) {
    if (!g || !dspark_model || !dw ||
        dw->n_stages == 0 ||
        dw->n_stages > DS4_DSPARK_MAX_STAGES ||
        dw->block_size == 0 ||
        dw->block_size > DS4_DSPARK_MAX_BLOCK_SIZE ||
        !metal_graph_dspark_final_output_hc(g) ||
        !metal_graph_batch_hc_mix(g) ||
        !metal_graph_batch_hc_split(g) ||
        !metal_graph_batch_flat_hc(g) ||
        !metal_graph_batch_ffn_cur(g) ||
        !metal_graph_batch_ffn_norm(g)) {
        return false;
    }

    const ds4_dspark_stage_weights *final =
        &dw->stage[dw->n_stages - 1u];
    const uint32_t draft = dw->block_size;
    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    if (!final->norm ||
        !final->hc_head_base ||
        !final->hc_head_fn ||
        !final->hc_head_scale ||
        final->norm->type != DS4_TENSOR_F32 ||
        final->hc_head_base->type != DS4_TENSOR_F32 ||
        !dspark_tensor_type_matches(final->hc_head_fn->type,
                                    DS4_DSPARK_LAYOUT_PLAIN) ||
        final->hc_head_scale->type != DS4_TENSOR_F32 ||
        final->norm->ndim != 1 ||
        final->norm->dim[0] != DS4_N_EMBD ||
        final->hc_head_base->ndim != 1 ||
        final->hc_head_base->dim[0] != DS4_N_HC ||
        final->hc_head_fn->ndim != 2 ||
        final->hc_head_fn->dim[0] != hc_dim ||
        final->hc_head_fn->dim[1] != DS4_N_HC ||
        final->hc_head_scale->ndim != 1 ||
        final->hc_head_scale->dim[0] != 1 ||
        ds4_gpu_tensor_bytes(metal_graph_dspark_final_output_hc(g)) <
            (uint64_t)draft * hc_dim * sizeof(float) ||
        ds4_gpu_tensor_bytes(metal_graph_batch_flat_hc(g)) <
            (uint64_t)draft * hc_dim * sizeof(float) ||
        ds4_gpu_tensor_bytes(metal_graph_batch_hc_mix(g)) <
            (uint64_t)draft * DS4_N_HC * sizeof(float) ||
        ds4_gpu_tensor_bytes(metal_graph_batch_hc_split(g)) <
            (uint64_t)draft * DS4_N_HC * sizeof(float) ||
        ds4_gpu_tensor_bytes(metal_graph_batch_ffn_cur(g)) <
            (uint64_t)draft * DS4_N_EMBD * sizeof(float) ||
        ds4_gpu_tensor_bytes(metal_graph_batch_ffn_norm(g)) <
            (uint64_t)draft * DS4_N_EMBD * sizeof(float)) {
        return false;
    }

    ds4_gpu_tensor *output_pre =
        ds4_gpu_tensor_view(metal_graph_batch_hc_mix(g),
                            0,
                            (uint64_t)draft * DS4_N_HC * sizeof(float));
    ds4_gpu_tensor *output_weights =
        ds4_gpu_tensor_view(metal_graph_batch_hc_split(g),
                            0,
                            (uint64_t)draft * DS4_N_HC * sizeof(float));
    ds4_gpu_tensor *output_embd =
        ds4_gpu_tensor_view(metal_graph_batch_ffn_cur(g),
                            0,
                            (uint64_t)draft * DS4_N_EMBD * sizeof(float));
    ds4_gpu_tensor *output_norm =
        ds4_gpu_tensor_view(metal_graph_batch_ffn_norm(g),
                            0,
                            (uint64_t)draft * DS4_N_EMBD * sizeof(float));
    ds4_gpu_tensor *stage_output_hc = metal_graph_dspark_final_output_hc(g);

    bool ok = stage_output_hc && output_pre && output_weights &&
              output_embd && output_norm;
    if (ok) ok = ds4_gpu_begin_commands() != 0;
    if (ok) ok = ds4_gpu_rms_norm_plain_rows_tensor(metal_graph_batch_flat_hc(g),
                                                     stage_output_hc,
                                                     (uint32_t)hc_dim,
                                                     draft,
                                                     DS4_RMS_EPS) != 0;
    if (ok) ok = metal_graph_matmul_plain_tensor(output_pre,
                                                 dspark_model,
                                                 final->hc_head_fn,
                                                 hc_dim,
                                                 DS4_N_HC,
                                                 metal_graph_batch_flat_hc(g),
                                                 draft);
    if (ok) ok = ds4_gpu_output_hc_weights_tensor(output_weights,
                                                   output_pre,
                                                   dspark_model->map,
                                                   dspark_model->size,
                                                   final->hc_head_scale->abs_offset,
                                                   final->hc_head_base->abs_offset,
                                                   DS4_N_HC,
                                                   DS4_HC_EPS) != 0;
    if (ok) ok = ds4_gpu_hc_weighted_sum_tensor(output_embd,
                                                 stage_output_hc,
                                                 output_weights,
                                                 DS4_N_EMBD,
                                                 DS4_N_HC) != 0;
    if (ok) ok = ds4_gpu_rms_norm_weight_rows_tensor(output_norm,
                                                      output_embd,
                                                      dspark_model->map,
                                                      dspark_model->size,
                                                      final->norm->abs_offset,
                                                      DS4_N_EMBD,
                                                      draft,
                                                      DS4_RMS_EPS) != 0;
    if (ok) ok = ds4_gpu_end_commands() != 0;
    if (!ok) (void)ds4_gpu_synchronize();

    ds4_gpu_tensor_free(output_norm);
    ds4_gpu_tensor_free(output_embd);
    ds4_gpu_tensor_free(output_weights);
    ds4_gpu_tensor_free(output_pre);
    return ok;
}

static bool metal_graph_eval_dspark_base_logits_from_hidden(
        ds4_gpu_graph      *g,
        const ds4_model    *base_model,
        const ds4_weights  *base_weights,
        const ds4_dspark_weights *dw) {
    if (!g || !base_model || !base_weights || !dw ||
        dw->block_size == 0 ||
        dw->block_size > DS4_DSPARK_MAX_BLOCK_SIZE ||
        !base_weights->output ||
        !tensor_type_is_dense_quant(base_weights->output->type) ||
        base_weights->output->ndim != 2 ||
        base_weights->output->dim[0] != DS4_N_EMBD ||
        base_weights->output->dim[1] != DS4_N_VOCAB ||
        !metal_graph_batch_ffn_norm(g) ||
        !g->spec_logits ||
        ds4_gpu_tensor_bytes(metal_graph_batch_ffn_norm(g)) <
            (uint64_t)dw->block_size * DS4_N_EMBD * sizeof(float) ||
        ds4_gpu_tensor_bytes(g->spec_logits) <
            (uint64_t)dw->block_size * DS4_N_VOCAB * sizeof(float)) {
        return false;
    }

    ds4_gpu_tensor *output_norm =
        ds4_gpu_tensor_view(metal_graph_batch_ffn_norm(g),
                            0,
                            (uint64_t)dw->block_size *
                                DS4_N_EMBD * sizeof(float));
    ds4_gpu_tensor *logits =
        ds4_gpu_tensor_view(g->spec_logits,
                            0,
                            (uint64_t)dw->block_size *
                                DS4_N_VOCAB * sizeof(float));
    bool ok = output_norm && logits;
    if (ok) ok = ds4_gpu_begin_commands() != 0;
    if (ok) ok = metal_graph_matmul_plain_tensor(logits,
                                                  base_model,
                                                  base_weights->output,
                                                  DS4_N_EMBD,
                                                  DS4_N_VOCAB,
                                                  output_norm,
                                                  dw->block_size);
    if (ok) ok = ds4_gpu_end_commands() != 0;
    if (!ok) (void)ds4_gpu_synchronize();

    ds4_gpu_tensor_free(logits);
    ds4_gpu_tensor_free(output_norm);
    return ok;
}

static bool dspark_markov_probe_ready(
        const ds4_dspark_weights *dw) {
    if (!dw ||
        dw->n_stages == 0 ||
        dw->n_stages > DS4_DSPARK_MAX_STAGES ||
        dw->block_size == 0 ||
        dw->block_size > DS4_DSPARK_MAX_BLOCK_SIZE ||
        dw->markov_rank == 0) {
        return false;
    }

    const ds4_dspark_stage_weights *final =
        &dw->stage[dw->n_stages - 1u];
    if (!final->markov_w1 ||
        !final->markov_w2 ||
        !dspark_tensor_type_matches(final->markov_w1->type,
                                    DS4_DSPARK_LAYOUT_DENSE) ||
        !dspark_tensor_type_matches(final->markov_w2->type,
                                    DS4_DSPARK_LAYOUT_DENSE)) {
        return false;
    }

    return final->markov_w1->ndim == 2 &&
           final->markov_w1->dim[0] == dw->markov_rank &&
           final->markov_w1->dim[1] == DS4_N_VOCAB &&
           final->markov_w2->ndim == 2 &&
           final->markov_w2->dim[0] == dw->markov_rank &&
           final->markov_w2->dim[1] == DS4_N_VOCAB;
}

static bool dspark_dense_row_to_f32(
        float           *out,
        const ds4_model *model,
        const ds4_tensor *t,
        uint32_t         row) {
    if (!out || !model || !t || t->ndim != 2 || row >= t->dim[1]) {
        return false;
    }

    const uint64_t width = t->dim[0];
    if (t->type == DS4_TENSOR_F32) {
        const float *base = tensor_data(model, t);
        memcpy(out, base + (uint64_t)row * width, width * sizeof(out[0]));
        return true;
    }
    if (t->type == DS4_TENSOR_F16) {
        const uint16_t *base = tensor_data(model, t);
        const uint16_t *src = base + (uint64_t)row * width;
        for (uint64_t i = 0; i < width; i++) out[i] = f16_to_f32(src[i]);
        return true;
    }
    if (t->type == DS4_TENSOR_Q8_0) {
        const uint64_t blocks = (width + 31u) / 32u;
        const uint8_t *src =
            (const uint8_t *)tensor_data(model, t) +
            (uint64_t)row * blocks * 34u;
        for (uint64_t b = 0; b < blocks; b++) {
            uint16_t scale_bits;
            memcpy(&scale_bits, src + b * 34u, sizeof(scale_bits));
            const float scale = f16_to_f32(scale_bits);
            const int8_t *qs = (const int8_t *)(src + b * 34u + 2u);
            const uint64_t i0 = b * 32u;
            const uint64_t n = width - i0 < 32u ? width - i0 : 32u;
            for (uint64_t i = 0; i < n; i++) {
                out[i0 + i] = scale * (float)qs[i];
            }
        }
        return true;
    }
    return false;
}

static uint32_t dspark_argmax_f32(const float *x, uint32_t n) {
    uint32_t best = 0;
    float best_v = x[0];
    for (uint32_t i = 1; i < n; i++) {
        if (x[i] > best_v) {
            best_v = x[i];
            best = i;
        }
    }
    return best;
}

typedef struct {
    const uint8_t *data;
    const int8_t *xq;
    const float *xscale;
    const float *logits;
    uint64_t in_dim;
    uint64_t blocks;
    uint64_t rows_per_slot;
    uint32_t best_idx[DS4_MAX_THREADS];
    float best_val[DS4_MAX_THREADS];
} dspark_markov_q8_0_argmax_ctx;

static void dspark_markov_q8_0_argmax_worker(
        void *vctx,
        uint64_t row0,
        uint64_t row1) {
    dspark_markov_q8_0_argmax_ctx *ctx = vctx;
    uint64_t slot = ctx->rows_per_slot ? row0 / ctx->rows_per_slot : 0;
    if (slot >= DS4_MAX_THREADS) slot = DS4_MAX_THREADS - 1u;

    float best_v = -FLT_MAX;
    uint32_t best = (uint32_t)row0;
    for (uint64_t row = row0; row < row1; row++) {
        const uint8_t *wrow = ctx->data + row * ctx->blocks * 34u;
        const float score =
            ctx->logits[row] +
            dot_q8_0_row(wrow, ctx->xq, ctx->xscale, ctx->in_dim, ctx->blocks);
        if (score > best_v) {
            best_v = score;
            best = (uint32_t)row;
        }
    }

    ctx->best_idx[slot] = best;
    ctx->best_val[slot] = best_v;
}

static bool dspark_markov_q8_0_argmax(
        uint32_t        *token_out,
        const ds4_model *model,
        const ds4_tensor *w,
        const float     *state,
        const float     *logits) {
    if (!token_out ||
        !model ||
        !w ||
        !state ||
        !logits ||
        w->type != DS4_TENSOR_Q8_0 ||
        w->ndim != 2 ||
        w->dim[1] > UINT32_MAX) {
        return false;
    }

    const uint64_t in_dim = w->dim[0];
    const uint64_t out_dim = w->dim[1];
    const uint64_t blocks = (in_dim + 31u) / 32u;
    if (out_dim == 0 ||
        blocks == 0 ||
        blocks > (uint64_t)SIZE_MAX / 32u ||
        blocks > (uint64_t)SIZE_MAX / sizeof(float)) {
        return false;
    }

    enum { DSPARK_MARKOV_ARGMAX_STACK_BLOCKS = 32 };
    int8_t xq_stack[DSPARK_MARKOV_ARGMAX_STACK_BLOCKS * 32u];
    float xscale_stack[DSPARK_MARKOV_ARGMAX_STACK_BLOCKS];
    const bool use_stack = blocks <= DSPARK_MARKOV_ARGMAX_STACK_BLOCKS;
    int8_t *xq = use_stack ? xq_stack : xmalloc((size_t)blocks * 32u);
    float *xscale = use_stack ? xscale_stack :
        xmalloc((size_t)blocks * sizeof(xscale[0]));
    quantize_q8_0_activation(state, xq, xscale, in_dim);

    ds4_threads_init();
    const uint32_t n_slots =
        g_pool.n_threads == 0 ? 1u : g_pool.n_threads;
    const uint64_t rows_per_slot = (out_dim + n_slots - 1u) / n_slots;
    dspark_markov_q8_0_argmax_ctx ctx = {
        .data = tensor_data(model, w),
        .xq = xq,
        .xscale = xscale,
        .logits = logits,
        .in_dim = in_dim,
        .blocks = blocks,
        .rows_per_slot = rows_per_slot,
    };
    for (uint32_t i = 0; i < DS4_MAX_THREADS; i++) {
        ctx.best_idx[i] = 0;
        ctx.best_val[i] = -FLT_MAX;
    }

    ds4_parallel_for(out_dim, dspark_markov_q8_0_argmax_worker, &ctx);

    uint32_t best = 0;
    float best_v = -FLT_MAX;
    for (uint32_t slot = 0; slot < n_slots && slot < DS4_MAX_THREADS; slot++) {
        const uint64_t row0 = (uint64_t)slot * rows_per_slot;
        if (row0 >= out_dim) break;
        if (ctx.best_val[slot] > best_v) {
            best_v = ctx.best_val[slot];
            best = ctx.best_idx[slot];
        }
    }

    if (!use_stack) {
        free(xscale);
        free(xq);
    }
    *token_out = best;
    return true;
}

/* Exact target verification preserves correctness when this diagnostic mode
 * proposes directly from the support model's base logits. */
static bool dspark_markov_bias_disabled(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *env = getenv("DS4_DSPARK_NO_MARKOV");
        cached = (env && env[0] && strcmp(env, "0") != 0) ? 1 : 0;
    }
    return cached == 1;
}

static bool dspark_disable_fused_cpu_markov_argmax(void) {
    static int cache = -1;
    if (cache < 0) {
        const char *env = getenv("DS4_DSPARK_DISABLE_FUSED_CPU_MARKOV_ARGMAX");
        cache = (env && env[0] && strcmp(env, "0") != 0) ? 1 : 0;
    }
    return cache != 0;
}

static bool dspark_disable_reuse_confidence0_markov(void) {
    static int cache = -1;
    if (cache < 0) {
        const char *env = getenv("DS4_DSPARK_DISABLE_REUSE_CONFIDENCE0_MARKOV");
        cache = (env && env[0] && strcmp(env, "0") != 0) ? 1 : 0;
    }
    return cache != 0;
}

static bool dspark_apply_markov_greedy_probe(
        float                  *logits,
        const ds4_model        *dspark_model,
        const ds4_dspark_weights *dw,
        int                     first_prev_token,
        float                  *markov_state,
        float                  *markov_bias,
        int32_t                 proposal[DS4_DSPARK_MAX_BLOCK_SIZE],
        uint32_t               *proposal_len) {
    if (proposal_len) *proposal_len = 0;
    if (!logits ||
        !dspark_model ||
        !dw ||
        !markov_state ||
        !markov_bias ||
        !proposal ||
        first_prev_token < 0 ||
        (uint32_t)first_prev_token >= DS4_N_VOCAB ||
        !dspark_markov_probe_ready(dw)) {
        return false;
    }

    const ds4_dspark_stage_weights *final =
        &dw->stage[dw->n_stages - 1u];
    const bool no_bias = dspark_markov_bias_disabled();
    int32_t prev_token = first_prev_token;
    for (uint32_t draft = 0; draft < dw->block_size; draft++) {
        float *row = logits + (uint64_t)draft * DS4_N_VOCAB;
        if (!no_bias) {
            if (!dspark_dense_row_to_f32(markov_state,
                                         dspark_model,
                                         final->markov_w1,
                                         (uint32_t)prev_token)) {
                return false;
            }
            matvec_any(markov_bias, dspark_model, final->markov_w2, markov_state);
            for (uint32_t i = 0; i < DS4_N_VOCAB; i++) {
                row[i] += markov_bias[i];
            }
        }
        const uint32_t token = dspark_argmax_f32(row, DS4_N_VOCAB);
        proposal[draft] = (int32_t)token;
        prev_token = (int32_t)token;
    }

    if (proposal_len) *proposal_len = dw->block_size;
    return true;
}

static bool dspark_confidence_probe_ready(
        const ds4_dspark_weights *dw) {
    if (!dspark_markov_probe_ready(dw)) return false;
    const ds4_dspark_stage_weights *final =
        &dw->stage[dw->n_stages - 1u];
    if (!final->confidence_proj ||
        !dspark_tensor_type_matches(final->confidence_proj->type,
                                    DS4_DSPARK_LAYOUT_DENSE)) {
        return false;
    }
    return final->confidence_proj->ndim == 2 &&
           final->confidence_proj->dim[0] ==
               (uint64_t)DS4_N_EMBD + dw->markov_rank &&
           final->confidence_proj->dim[1] == 1;
}

static bool dspark_eval_confidence_probe(
        float                  *confidence_logits,
        const float            *hidden_rows,
        const ds4_model        *dspark_model,
        const ds4_dspark_weights *dw,
        int                     first_prev_token,
        const int32_t           proposal[DS4_DSPARK_MAX_BLOCK_SIZE],
        float                  *markov_state,
        float                  *features,
        uint32_t               *confidence_len) {
    if (confidence_len) *confidence_len = 0;
    if (!confidence_logits ||
        !hidden_rows ||
        !dspark_model ||
        !dw ||
        !proposal ||
        !markov_state ||
        !features ||
        first_prev_token < 0 ||
        (uint32_t)first_prev_token >= DS4_N_VOCAB ||
        !dspark_confidence_probe_ready(dw)) {
        return false;
    }

    const ds4_dspark_stage_weights *final =
        &dw->stage[dw->n_stages - 1u];
    int32_t prev_token = first_prev_token;
    for (uint32_t draft = 0; draft < dw->block_size; draft++) {
        if (prev_token < 0 || (uint32_t)prev_token >= DS4_N_VOCAB) {
            return false;
        }
        if (!dspark_dense_row_to_f32(markov_state,
                                     dspark_model,
                                     final->markov_w1,
                                     (uint32_t)prev_token)) {
            return false;
        }
        memcpy(features,
               hidden_rows + (uint64_t)draft * DS4_N_EMBD,
               (uint64_t)DS4_N_EMBD * sizeof(features[0]));
        memcpy(features + DS4_N_EMBD,
               markov_state,
               (uint64_t)dw->markov_rank * sizeof(features[0]));
        matvec_any(confidence_logits + draft,
                   dspark_model,
                   final->confidence_proj,
                   features);
        prev_token = proposal[draft];
    }

    if (confidence_len) *confidence_len = dw->block_size;
    return true;
}

static bool dspark_apply_markov_confidence_lazy_runtime(
        ds4_gpu_graph          *g,
        const ds4_model        *dspark_model,
        const ds4_dspark_weights *dw,
        int                     first_prev_token,
        float                   confidence_threshold,
        float                  *logits,
        float                  *markov_bias,
        float                  *features,
        size_t                  features_cap,
        int32_t                 proposal[DS4_DSPARK_MAX_BLOCK_SIZE],
        uint32_t               *proposal_len,
        uint32_t               *confidence_len,
        uint32_t               *confidence_prefix_len,
        bool                    reuse_first_confidence,
        float                  *confidence0) {
    if (proposal_len) *proposal_len = 0;
    if (confidence_len) *confidence_len = 0;
    if (confidence_prefix_len) *confidence_prefix_len = 0;
    if (confidence0 && !reuse_first_confidence) *confidence0 = 0.0f;
    if (!g ||
        !g->spec_logits ||
        !metal_graph_batch_ffn_norm(g) ||
        !dspark_model ||
        !dw ||
        !logits ||
        !markov_bias ||
        !features ||
        !proposal ||
        confidence_threshold <= 0.0f ||
        first_prev_token < 0 ||
        (uint32_t)first_prev_token >= DS4_N_VOCAB ||
        (reuse_first_confidence && !confidence0) ||
        !dspark_markov_probe_ready(dw) ||
        !dspark_confidence_probe_ready(dw)) {
        return false;
    }

    const uint64_t logits_bytes =
        (uint64_t)DS4_N_VOCAB * sizeof(float);
    const uint64_t hidden_bytes =
        (uint64_t)DS4_N_EMBD * sizeof(float);
    const uint64_t feature_count =
        (uint64_t)DS4_N_EMBD + (uint64_t)dw->markov_rank;
    if (feature_count > features_cap) return false;
    float *markov_state = features + DS4_N_EMBD;
    bool ok = true;

    const ds4_dspark_stage_weights *final =
        &dw->stage[dw->n_stages - 1u];
    int32_t prev_token = first_prev_token;
    uint32_t produced = 0;
    uint32_t confident = 0;
    for (uint32_t draft = 0; ok && draft < dw->block_size; draft++) {
        if (prev_token < 0 || (uint32_t)prev_token >= DS4_N_VOCAB) {
            ok = false;
            break;
        }
        float confidence_logit = 0.0f;
        if (draft == 0 && reuse_first_confidence) {
            confidence_logit = *confidence0;
        } else {
            ok = dspark_dense_row_to_f32(markov_state,
                                         dspark_model,
                                         final->markov_w1,
                                         (uint32_t)prev_token);
            if (!ok) break;

            ok = ds4_gpu_tensor_read(metal_graph_batch_ffn_norm(g),
                                     (uint64_t)draft * hidden_bytes,
                                     features,
                                     hidden_bytes) != 0;
            if (!ok) break;
            matvec_any(&confidence_logit,
                       dspark_model,
                       final->confidence_proj,
                       features);
        }
        if (draft == 0 && confidence0) *confidence0 = confidence_logit;
        if (confidence_len) *confidence_len = draft + 1u;
        if (sigmoid_stable(confidence_logit) < confidence_threshold) {
            ok = true;
            break;
        }

        int32_t token = -1;
#ifndef __APPLE__
        /* CUDA can apply the Markov bias and argmax without reading back the
         * full logits row. Metal currently falls through to the CPU path. */
        if (ok && !dspark_markov_bias_disabled() &&
            getenv("DS4_DSPARK_NO_GPU_MARKOV") == NULL &&
            g->dspark_draft_tokens &&
            dw->markov_rank != 0 && (dw->markov_rank & 31u) == 0 &&
            final->markov_w1->type == DS4_TENSOR_Q8_0 &&
            final->markov_w2->type == DS4_TENSOR_Q8_0) {
            ds4_gpu_tensor *row_view =
                ds4_gpu_tensor_view(g->spec_logits,
                                    (uint64_t)draft * logits_bytes,
                                    logits_bytes);
            uint64_t gpu_key = 0;
            bool gpu_ok = row_view &&
                ds4_gpu_dspark_markov_argmax_tensor(
                    g->dspark_draft_tokens,
                    row_view,
                    dspark_model->map,
                    dspark_model->size,
                    final->markov_w1->abs_offset,
                    final->markov_w2->abs_offset,
                    (uint32_t)prev_token,
                    DS4_N_VOCAB,
                    dw->markov_rank) != 0 &&
                ds4_gpu_tensor_read(g->dspark_draft_tokens,
                                    0,
                                    &gpu_key,
                                    sizeof(gpu_key)) != 0;
            ds4_gpu_tensor_free(row_view);
            const uint32_t gpu_token = ~(uint32_t)(gpu_key & 0xffffffffu);
            if (gpu_ok && gpu_key != 0 && gpu_token < DS4_N_VOCAB) {
                token = (int32_t)gpu_token;
                proposal[draft] = token;
                produced = draft + 1u;
                confident = produced;
                prev_token = token;
                continue;
            }
        }
#endif
        if (ok) {
            ok = ds4_gpu_tensor_read(g->spec_logits,
                                     (uint64_t)draft * logits_bytes,
                                     logits,
                                     logits_bytes) != 0;
            if (ok) {
                uint32_t fused_token = 0;
                if (dspark_markov_bias_disabled()) {
                    token = (int32_t)dspark_argmax_f32(logits, DS4_N_VOCAB);
                } else if (!dspark_disable_fused_cpu_markov_argmax() &&
                    dspark_markov_q8_0_argmax(&fused_token,
                                              dspark_model,
                                              final->markov_w2,
                                              markov_state,
                                              logits)) {
                    token = (int32_t)fused_token;
                } else {
                    matvec_any(markov_bias, dspark_model, final->markov_w2, markov_state);
                    for (uint32_t i = 0; i < DS4_N_VOCAB; i++) {
                        logits[i] += markov_bias[i];
                    }
                    token = (int32_t)dspark_argmax_f32(logits, DS4_N_VOCAB);
                }
            }
        }
        if (!ok || token < 0 || (uint32_t)token >= DS4_N_VOCAB) {
            ok = false;
            break;
        }
        proposal[draft] = token;
        produced = draft + 1u;
        confident = produced;
        prev_token = token;
    }

    if (ok) {
        if (proposal_len) *proposal_len = produced;
        if (confidence_prefix_len) *confidence_prefix_len = confident;
    }
    return ok;
}

static bool dspark_eval_confidence0_runtime(
        ds4_gpu_graph            *g,
        const ds4_model          *dspark_model,
        const ds4_dspark_weights *dw,
        int                       first_prev_token,
        float                    *features,
        size_t                    features_cap,
        float                    *confidence0) {
    if (confidence0) *confidence0 = 0.0f;
    if (!confidence0 ||
        !g ||
        !metal_graph_batch_ffn_norm(g) ||
        !dspark_model ||
        !dw ||
        !features ||
        first_prev_token < 0 ||
        (uint32_t)first_prev_token >= DS4_N_VOCAB ||
        !dspark_confidence_probe_ready(dw)) {
        return false;
    }

    const uint64_t hidden_bytes =
        (uint64_t)DS4_N_EMBD * sizeof(float);
    const uint64_t feature_count =
        (uint64_t)DS4_N_EMBD + (uint64_t)dw->markov_rank;
    if (feature_count > features_cap ||
        ds4_gpu_tensor_bytes(metal_graph_batch_ffn_norm(g)) < hidden_bytes) {
        return false;
    }

    float *markov_state = features + DS4_N_EMBD;
    bool ok = true;

    const ds4_dspark_stage_weights *final =
        &dw->stage[dw->n_stages - 1u];
    if (ok) {
        ok = dspark_dense_row_to_f32(markov_state,
                                     dspark_model,
                                     final->markov_w1,
                                     (uint32_t)first_prev_token);
    }
    if (ok) {
        ok = ds4_gpu_tensor_read(metal_graph_batch_ffn_norm(g),
                                 0,
                                 features,
                                 hidden_bytes) != 0;
    }
    if (ok) {
        matvec_any(confidence0, dspark_model, final->confidence_proj, features);
    }

    return ok;
}

static uint32_t dspark_confident_prefix_len(
        const float *confidence_logits,
        uint32_t     confidence_len,
        float        threshold) {
    if (!confidence_logits || confidence_len == 0 || threshold <= 0.0f) {
        return confidence_len;
    }
    for (uint32_t i = 0; i < confidence_len; i++) {
        if (sigmoid_stable(confidence_logits[i]) < threshold) return i;
    }
    return confidence_len;
}

static bool metal_graph_eval_mtp_draft_from_hc(
        ds4_gpu_graph       *g,
        const ds4_model       *base_model,
        const ds4_weights     *base_weights,
        const ds4_model       *mtp_model,
        const ds4_mtp_weights *mtp,
        ds4_gpu_tensor      *prev_hc,
        ds4_gpu_tensor      *out_hc,
        int                    token,
        uint32_t               pos,
        float                 *logits,
        int                   *top_id) {
    if (!mtp || !mtp->block.attn_q_a || !g->mtp_raw_cache || !prev_hc || !out_hc) return false;

    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const uint32_t raw_row = pos % g->raw_cap;
    uint32_t n_raw = g->mtp_n_raw + 1u;
    if (n_raw > g->raw_window) n_raw = g->raw_window;
    if (n_raw > g->raw_cap) n_raw = g->raw_cap;

    ds4_gpu_tensor *saved_cur = metal_graph_cur_hc(g);
    ds4_gpu_tensor *saved_after = metal_graph_after_ffn_hc(g);
    const uint32_t saved_tp_world = g->tp_world;
    const uint32_t saved_tp_batch_rows = g->tp_batch_rows;
    g->tp_world = 0;
    g->tp_batch_rows = 0;
    const bool suspended_expert_sharding = saved_tp_world == 2;
    if (suspended_expert_sharding) {
        ds4_gpu_tp_suspend_expert_sharding(1);
    }
    bool ok = ds4_gpu_begin_commands() != 0;
    if (ok) ok = ds4_gpu_embed_token_hc_tensor(g->mtp_embed,
                                                  base_model->map,
                                                  base_model->size,
                                                  base_weights->token_embd->abs_offset,
                                                  (uint32_t)base_weights->token_embd->dim[1],
                                                  (uint32_t)token,
                                                  DS4_N_EMBD,
                                                  1) != 0;
    if (ok) ok = ds4_gpu_rms_norm_weight_tensor(g->mtp_enorm,
                                                  g->mtp_embed,
                                                  mtp_model->map,
                                                  mtp_model->size,
                                                  mtp->enorm->abs_offset,
                                                  DS4_N_EMBD,
                                                  DS4_RMS_EPS) != 0;
    if (ok) ok = ds4_gpu_matmul_q8_0_tensor(g->mtp_eproj,
                                              mtp_model->map,
                                              mtp_model->size,
                                              mtp->e_proj->abs_offset,
                                              DS4_N_EMBD,
                                              DS4_N_EMBD,
                                              g->mtp_enorm,
                                              1) != 0;
    if (ok) ok = ds4_gpu_repeat_hc_tensor(g->mtp_eproj_hc,
                                            g->mtp_eproj,
                                            DS4_N_EMBD,
                                            DS4_N_HC) != 0;
    if (ok) ok = ds4_gpu_rms_norm_weight_rows_tensor(g->mtp_hnorm_hc,
                                                       prev_hc,
                                                       mtp_model->map,
                                                       mtp_model->size,
                                                       mtp->hnorm->abs_offset,
                                                       DS4_N_EMBD,
                                                       DS4_N_HC,
                                                       DS4_RMS_EPS) != 0;
    if (ok) ok = ds4_gpu_matmul_q8_0_tensor(g->mtp_hproj_hc,
                                              mtp_model->map,
                                              mtp_model->size,
                                              mtp->h_proj->abs_offset,
                                              DS4_N_EMBD,
                                              DS4_N_EMBD,
                                              g->mtp_hnorm_hc,
                                              DS4_N_HC) != 0;
    if (ok) ok = ds4_gpu_add_tensor(g->mtp_input_hc,
                                      g->mtp_eproj_hc,
                                      g->mtp_hproj_hc,
                                      (uint32_t)hc_dim) != 0;
    if (ok) {
        g->cur_hc_by_tier[g->active_tier] = g->mtp_input_hc;
        g->after_ffn_hc_by_tier[g->active_tier] = out_hc;
        ok = metal_graph_encode_decode_layer(g,
                                             mtp_model,
                                             &mtp->block,
                                             1,
                                             pos,
                                             g->mtp_raw_cache,
                                             g->raw_cap,
                                             raw_row,
                                             n_raw,
                                             token);
    }
    if (ok) g->cur_hc_by_tier[g->active_tier] = out_hc;
    if (ok) ok = metal_graph_encode_output_head_mtp(g,
                                                    base_model,
                                                    base_weights,
                                                    mtp_model,
                                                    mtp,
                                                    base_weights->output->dim[1]);
    if (ok && top_id) {
        ok = ds4_gpu_argmax_tensor(metal_graph_comp_selected(g),
                                   metal_graph_logits(g),
                                   DS4_N_VOCAB) != 0;
    }
    if (ok) ok = ds4_gpu_end_commands() != 0;
    if (suspended_expert_sharding) {
        ds4_gpu_tp_suspend_expert_sharding(0);
    }
    g->cur_hc_by_tier[g->active_tier] = saved_cur;
    g->after_ffn_hc_by_tier[g->active_tier] = saved_after;

    if (ok && logits) {
        ok = ds4_gpu_tensor_read(metal_graph_logits(g), 0, logits, (uint64_t)DS4_N_VOCAB * sizeof(float)) != 0;
    }
    if (ok && top_id) {
        ok = ds4_gpu_tensor_read(metal_graph_comp_selected(g), 0, top_id, sizeof(*top_id)) != 0;
    }
    if (ok && g->mtp_n_raw < g->raw_window) g->mtp_n_raw++;
    g->tp_world = saved_tp_world;
    g->tp_batch_rows = saved_tp_batch_rows;
    if (!ok) {
        (void)ds4_gpu_synchronize();
        g->cur_hc_by_tier[g->active_tier] = saved_cur;
        g->after_ffn_hc_by_tier[g->active_tier] = saved_after;
    }
    return ok;
}

static bool metal_graph_eval_mtp_draft(
        ds4_gpu_graph       *g,
        const ds4_model       *base_model,
        const ds4_weights     *base_weights,
        const ds4_model       *mtp_model,
        const ds4_mtp_weights *mtp,
        int                    token,
        uint32_t               pos,
        float                 *logits,
        int                   *top_id) {
    return metal_graph_eval_mtp_draft_from_hc(g,
                                              base_model,
                                              base_weights,
                                              mtp_model,
                                              mtp,
                                              metal_graph_cur_hc(g),
                                              g->mtp_state_hc,
                                              token,
                                              pos,
                                              logits,
                                              top_id);
}

/* =========================================================================
 * Imatrix Collection.
 * =========================================================================
 *
 * The 2-bit DS4 quants care most about routed MoE experts.  For expert gate
 * and up matrices the matmul input is the FFN-normalized activation row.  For
 * expert down matrices the matmul input is the routed SwiGLU row after route
 * weighting.  During Metal prefill those tensors are already materialized as
 * `batch_ffn_norm`, `batch_router_selected`, and `batch_routed_mid`, so the
 * collector observes the exact release graph without changing inference math.
 *
 * The output is llama.cpp's legacy imatrix `.dat` format.  Entries are packed
 * by expert: one tensor entry contains `n_expert * n_columns` floats and the
 * quantizer slices the vector for each expert.
 */
typedef struct {
    float *gate_up_sum2;   /* [active layer][active expert][hidden] */
    float *down_sum2;      /* [active layer][active expert][expert FFN] */
    uint32_t gate_up_count[DS4_MAX_LAYER][DS4_MAX_EXPERT];
    uint32_t down_count[DS4_MAX_LAYER][DS4_MAX_EXPERT];
    float *ffn_norm_buf;
    float *routed_mid_buf;
    uint16_t *routed_mid_f16_buf;
    int   *selected_buf;
    float *sq_tmp;
    uint32_t cap_tokens;
    uint64_t observed_tokens;
    uint64_t observed_routes;
    uint32_t chunks;
    const char *dataset_path;
} ds4_imatrix_collector;

static bool imatrix_collector_init(ds4_imatrix_collector *c, uint32_t cap_tokens, const char *dataset_path) {
    memset(c, 0, sizeof(*c));
    c->cap_tokens = cap_tokens ? cap_tokens : 1u;
    c->dataset_path = dataset_path;
    const size_t gate_n = (size_t)DS4_N_LAYER * DS4_N_EXPERT * DS4_N_EMBD;
    const size_t down_n = (size_t)DS4_N_LAYER * DS4_N_EXPERT * DS4_N_FF_EXP;
    c->gate_up_sum2 = xcalloc(gate_n, sizeof(c->gate_up_sum2[0]));
    c->down_sum2 = xcalloc(down_n, sizeof(c->down_sum2[0]));
    c->ffn_norm_buf = xmalloc((size_t)c->cap_tokens * DS4_N_EMBD * sizeof(c->ffn_norm_buf[0]));
    c->routed_mid_buf = xmalloc((size_t)c->cap_tokens * DS4_N_EXPERT_USED * DS4_N_FF_EXP * sizeof(c->routed_mid_buf[0]));
    c->routed_mid_f16_buf = xmalloc((size_t)c->cap_tokens * DS4_N_EXPERT_USED * DS4_N_FF_EXP * sizeof(c->routed_mid_f16_buf[0]));
    c->selected_buf = xmalloc((size_t)c->cap_tokens * DS4_N_EXPERT_USED * sizeof(c->selected_buf[0]));
    c->sq_tmp = xmalloc((size_t)DS4_N_EMBD * sizeof(c->sq_tmp[0]));
    return c->gate_up_sum2 && c->down_sum2 && c->ffn_norm_buf &&
           c->routed_mid_buf && c->routed_mid_f16_buf && c->selected_buf && c->sq_tmp;
}

static void imatrix_collector_free(ds4_imatrix_collector *c) {
    if (!c) return;
    free(c->gate_up_sum2);
    free(c->down_sum2);
    free(c->ffn_norm_buf);
    free(c->routed_mid_buf);
    free(c->routed_mid_f16_buf);
    free(c->selected_buf);
    free(c->sq_tmp);
    memset(c, 0, sizeof(*c));
}

static float *imatrix_gate_up_ptr(ds4_imatrix_collector *c, uint32_t il, uint32_t expert) {
    return c->gate_up_sum2 + ((size_t)il * DS4_N_EXPERT + expert) * DS4_N_EMBD;
}

static float *imatrix_down_ptr(ds4_imatrix_collector *c, uint32_t il, uint32_t expert) {
    return c->down_sum2 + ((size_t)il * DS4_N_EXPERT + expert) * DS4_N_FF_EXP;
}

static bool imatrix_collect_layer_batch(
        ds4_imatrix_collector *c,
        ds4_gpu_graph         *g,
        uint32_t               il,
        uint32_t               n_tokens) {
    if (!c || n_tokens == 0) return true;
    if (n_tokens > c->cap_tokens) return false;

    const uint64_t norm_bytes = (uint64_t)n_tokens * DS4_N_EMBD * sizeof(float);
    const uint64_t mid_elems = (uint64_t)n_tokens * DS4_N_EXPERT_USED * DS4_N_FF_EXP;
    const uint64_t mid_bytes = mid_elems * (g->batch_routed_mid_is_f16 ? sizeof(uint16_t) : sizeof(float));
    const uint64_t sel_bytes = (uint64_t)n_tokens * DS4_N_EXPERT_USED * sizeof(int);
    void *mid_dst = g->batch_routed_mid_is_f16
        ? (void *)c->routed_mid_f16_buf
        : (void *)c->routed_mid_buf;
    if (ds4_gpu_tensor_read(metal_graph_batch_ffn_norm(g), 0, c->ffn_norm_buf, norm_bytes) == 0 ||
        ds4_gpu_tensor_read(metal_graph_batch_routed_mid(g), 0, mid_dst, mid_bytes) == 0 ||
        ds4_gpu_tensor_read(metal_graph_batch_router_selected(g), 0, c->selected_buf, sel_bytes) == 0)
    {
        return false;
    }

    for (uint32_t t = 0; t < n_tokens; t++) {
        const float *x = c->ffn_norm_buf + (size_t)t * DS4_N_EMBD;
        for (uint32_t i = 0; i < DS4_N_EMBD; i++) c->sq_tmp[i] = x[i] * x[i];

        for (uint32_t slot = 0; slot < DS4_N_EXPERT_USED; slot++) {
            const int expert = c->selected_buf[(size_t)t * DS4_N_EXPERT_USED + slot];
            if (expert < 0 || (uint32_t)expert >= DS4_N_EXPERT) continue;

            float *gate_up = imatrix_gate_up_ptr(c, il, (uint32_t)expert);
            for (uint32_t i = 0; i < DS4_N_EMBD; i++) gate_up[i] += c->sq_tmp[i];
            c->gate_up_count[il][expert]++;

            float *down = imatrix_down_ptr(c, il, (uint32_t)expert);
            const size_t mid_off = ((size_t)t * DS4_N_EXPERT_USED + slot) * DS4_N_FF_EXP;
            if (g->batch_routed_mid_is_f16) {
                const uint16_t *mid = c->routed_mid_f16_buf + mid_off;
                for (uint32_t i = 0; i < DS4_N_FF_EXP; i++) {
                    const float v = f16_to_f32(mid[i]);
                    down[i] += v * v;
                }
            } else {
                const float *mid = c->routed_mid_buf + mid_off;
                for (uint32_t i = 0; i < DS4_N_FF_EXP; i++) down[i] += mid[i] * mid[i];
            }
            c->down_count[il][expert]++;
            c->observed_routes++;
        }
    }
    c->observed_tokens += n_tokens;
    c->chunks++;
    return true;
}

static void imatrix_write_i32(FILE *fp, int32_t v) {
    if (fwrite(&v, sizeof(v), 1, fp) != 1) ds4_die("failed to write imatrix");
}

static void imatrix_write_entry(
        FILE       *fp,
        const char *name,
        const float *sum2,
        const uint32_t *counts,
        uint32_t n_expert,
        uint32_t n_col) {
    const int32_t len = (int32_t)strlen(name);
    const int32_t ncall = 1;
    const int32_t nval = (int32_t)((uint64_t)n_expert * n_col);
    imatrix_write_i32(fp, len);
    if (fwrite(name, 1, (size_t)len, fp) != (size_t)len) ds4_die("failed to write imatrix name");
    imatrix_write_i32(fp, ncall);
    imatrix_write_i32(fp, nval);

    float *tmp = xmalloc((size_t)n_col * sizeof(tmp[0]));
    for (uint32_t e = 0; e < n_expert; e++) {
        const uint32_t count = counts[e];
        const float *src = sum2 + (size_t)e * n_col;
        if (count == 0) {
            for (uint32_t i = 0; i < n_col; i++) tmp[i] = 1.0f;
        } else {
            const float inv = 1.0f / (float)count;
            for (uint32_t i = 0; i < n_col; i++) tmp[i] = src[i] * inv;
        }
        if (fwrite(tmp, sizeof(tmp[0]), n_col, fp) != n_col) ds4_die("failed to write imatrix values");
    }
    free(tmp);
}

static bool imatrix_collector_save(
        const ds4_imatrix_collector *c,
        const ds4_weights           *weights,
        const char                  *path) {
    FILE *fp = fopen(path, "wb");
    if (!fp) {
        fprintf(stderr, "ds4: failed to open imatrix output %s: %s\n", path, strerror(errno));
        return false;
    }

    const int32_t entries = (int32_t)(DS4_N_LAYER * 3);
    imatrix_write_i32(fp, entries);
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const ds4_layer_weights *layer = &weights->layer[il];
        char name[256];
        snprintf(name, sizeof(name), "%.*s", (int)layer->ffn_gate_exps->name.len, layer->ffn_gate_exps->name.ptr);
        imatrix_write_entry(fp, name,
                            c->gate_up_sum2 + (size_t)il * DS4_N_EXPERT * DS4_N_EMBD,
                            c->gate_up_count[il],
                            DS4_N_EXPERT,
                            DS4_N_EMBD);
        snprintf(name, sizeof(name), "%.*s", (int)layer->ffn_up_exps->name.len, layer->ffn_up_exps->name.ptr);
        imatrix_write_entry(fp, name,
                            c->gate_up_sum2 + (size_t)il * DS4_N_EXPERT * DS4_N_EMBD,
                            c->gate_up_count[il],
                            DS4_N_EXPERT,
                            DS4_N_EMBD);
        snprintf(name, sizeof(name), "%.*s", (int)layer->ffn_down_exps->name.len, layer->ffn_down_exps->name.ptr);
        imatrix_write_entry(fp, name,
                            c->down_sum2 + (size_t)il * DS4_N_EXPERT * DS4_N_FF_EXP,
                            c->down_count[il],
                            DS4_N_EXPERT,
                            DS4_N_FF_EXP);
    }

    const int32_t chunks = (int32_t)c->chunks;
    imatrix_write_i32(fp, chunks);
    const char *dataset = c->dataset_path ? c->dataset_path : "";
    const int32_t dataset_len = (int32_t)strlen(dataset);
    imatrix_write_i32(fp, dataset_len);
    if (dataset_len && fwrite(dataset, 1, (size_t)dataset_len, fp) != (size_t)dataset_len) {
        ds4_die("failed to write imatrix dataset name");
    }

    if (fclose(fp) != 0) {
        fprintf(stderr, "ds4: failed to close imatrix output %s: %s\n", path, strerror(errno));
        return false;
    }
    return true;
}

static bool metal_graph_reset_prefill_state(ds4_gpu_graph *g) {
    memset(g->layer_n_comp, 0, sizeof(g->layer_n_comp));
    memset(g->layer_n_index_comp, 0, sizeof(g->layer_n_index_comp));
    g->mtp_n_raw = 0;
    metal_graph_dspark_cache_reset(g);
    metal_graph_dspark_capture_invalidate(g);
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        if (!g->layer_raw_cache[il]) continue;
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (ratio == 0) continue;
        const uint32_t coff = ratio == 4 ? 2u : 1u;
        const uint64_t attn_width = (uint64_t)coff * DS4_N_HEAD_DIM;
        const uint64_t attn_rows = (uint64_t)coff * ratio;
        if (!metal_tensor_fill_f32(g->layer_attn_state_kv[il], 0.0f, attn_width * attn_rows)) return false;
        if (!metal_tensor_fill_f32(g->layer_attn_state_score[il], DS4_NEG_INF, attn_width * attn_rows)) return false;
        if (ratio == 4) {
            const uint64_t index_width = (uint64_t)coff * DS4_N_INDEXER_HEAD_DIM;
            const uint64_t index_rows = (uint64_t)coff * ratio;
            if (!metal_tensor_fill_f32(g->layer_index_state_kv[il], 0.0f, index_width * index_rows)) return false;
            if (!metal_tensor_fill_f32(g->layer_index_state_score[il], DS4_NEG_INF, index_width * index_rows)) return false;
        }
    }
    return true;
}

/* Execute graph-backend prefill in layer-major order so intermediate
 * activations stay on the GPU and cache state is built exactly once. */
static void gpu_graph_report_prefill_display_progress(
        ds4_session_progress_fn display_progress,
        void                   *display_progress_ud,
        uint32_t                start,
        uint32_t                n_tokens,
        uint32_t                layer_done,
        int                     total) {
    if (!display_progress) return;
    if (layer_done > (uint32_t)DS4_N_LAYER) layer_done = (uint32_t)DS4_N_LAYER;
    uint64_t done = (uint64_t)n_tokens * layer_done / (uint32_t)DS4_N_LAYER;
    if (layer_done == (uint32_t)DS4_N_LAYER) done = n_tokens;
    display_progress(display_progress_ud, "prefill_display",
                     (int)(start + (uint32_t)done), total);
}

typedef struct {
    int tier;
    uint32_t first_layer;
    uint32_t end_layer;
} metal_graph_prefill_stage;

static bool metal_graph_build_prefill_stages(
        const ds4_gpu_graph          *g,
        metal_graph_prefill_stage    *stages,
        uint32_t                     *n_stages) {
    if (!g || !g->placement || !stages || !n_stages) return false;
    uint32_t ns = 0;
    int prev_tier = -1;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const int tier = g->placement[il + 1];
        if (tier < 0 || tier >= DS4_MAX_GPUS) return false;
        if (il == 0 || tier != prev_tier) {
            if (ns >= DS4_MAX_GPUS) return false;
            stages[ns].tier = tier;
            stages[ns].first_layer = il;
            stages[ns].end_layer = il + 1u;
            ns++;
            prev_tier = tier;
        } else {
            stages[ns - 1u].end_layer = il + 1u;
        }
    }
    *n_stages = ns;
    return ns != 0;
}

static bool metal_graph_encode_prefill_stage_batch(
        ds4_gpu_graph              *g,
        const ds4_model            *model,
        const ds4_weights          *weights,
        const metal_graph_prefill_stage *stage,
        uint32_t                    pos0,
        uint32_t                    n_tokens) {
    if (!g || !model || !weights || !stage || n_tokens == 0) return false;
    if (!metal_graph_set_active_tier_no_copy(g, stage->tier)) return false;
    for (uint32_t il = stage->first_layer; il < stage->end_layer; il++) {
        if (g->placement && g->placement[il + 1] != stage->tier) return false;
        if (!metal_graph_encode_layer_batch(g,
                                            model,
                                            &weights->layer[il],
                                            il,
                                            pos0,
                                            n_tokens)) {
            return false;
        }
        if (g->pipeline_capture_chunk_len != 0 &&
            !metal_graph_dspark_capture_prefill_rows(
                    g, il,
                    g->pipeline_capture_chunk_start,
                    g->pipeline_capture_chunk_len,
                    pos0,
                    n_tokens)) {
            return false;
        }
    }
    return true;
}

static bool metal_graph_prefill_pipeline_stage_major(
        ds4_gpu_graph *g,
        const ds4_model       *model,
        const ds4_weights     *weights,
        const token_vec       *prompt,
        uint32_t               start,
        uint32_t               n_tokens,
        float                 *logits,
        bool                   show_progress,
        ds4_session_progress_fn display_progress,
        void                  *display_progress_ud) {
    if (!g || !model || !weights || !prompt || !g->placement ||
        n_tokens == 0 || n_tokens > g->prefill_cap ||
        start > (uint32_t)prompt->len ||
        n_tokens > (uint32_t)prompt->len - start) {
        return false;
    }

    metal_graph_prefill_stage stages[DS4_MAX_GPUS];
    uint32_t n_stages = 0;
    g->pipeline_capture_chunk_start = start;
    g->pipeline_capture_chunk_len =
        g->dspark_capture_enabled ? n_tokens : 0;
    if (!metal_graph_build_prefill_stages(g, stages, &n_stages) || n_stages < 2) {
        return false;
    }
    if (stages[0].tier != g->emb_tier) {
        return false;
    }

    uint32_t mb_cap = metal_graph_cuda_prefill_pipeline_microbatch();
    if (mb_cap == 0 || mb_cap >= n_tokens) return false;
    if (mb_cap > g->prefill_cap) mb_cap = g->prefill_cap;
    const uint32_t n_mb = (n_tokens + mb_cap - 1u) / mb_cap;
    if (n_mb < 2) return false;

    if (display_progress)
        display_progress(display_progress_ud, "prefill_display", (int)start, prompt->len);

    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    bool ok = true;
    const double t0 = getenv("DS4_METAL_GRAPH_PREFILL_PROFILE") ? now_sec() : 0.0;

    const bool sequential = getenv("DS4_CUDA_PREFILL_PIPELINE_SEQUENTIAL") != NULL;
    const bool suppress_q8_cache =
        !metal_graph_cuda_prefill_pipeline_q8_cache_requested();
    const int saved_q8_cache_suppressed = ds4_gpu_q8_cache_suppressed();
    if (suppress_q8_cache) ds4_gpu_set_q8_cache_suppressed(1);
    ok = ds4_gpu_begin_commands() != 0;
    if (sequential) {
        for (uint32_t mb_i = 0; ok && mb_i < n_mb; mb_i++) {
            const uint32_t mb_off = mb_i * mb_cap;
            uint32_t mb_len = n_tokens - mb_off;
            if (mb_len > mb_cap) mb_len = mb_cap;
            const uint32_t pos0 = start + mb_off;

            for (uint32_t stage_i = 0; ok && stage_i < n_stages; stage_i++) {
                g->batch_token_offset = mb_off;
                if (stage_i == 0) {
                    ok = metal_graph_set_active_tier_no_copy(g, stages[0].tier);
                    ds4_gpu_tensor *tokens_view = NULL;
                    if (ok) {
                        tokens_view = ds4_gpu_tensor_view(metal_graph_prefill_tokens(g),
                                                           (uint64_t)mb_off * sizeof(int32_t),
                                                           (uint64_t)mb_len * sizeof(int32_t));
                        ok = tokens_view != NULL;
                    }
                    if (ok) {
                        ok = metal_graph_upload_prompt_embeddings_hc(
                                g->batch_cur_hc_by_tier[stages[0].tier],
                                tokens_view,
                                model,
                                weights,
                                prompt,
                                pos0,
                                mb_len);
                    }
                    ds4_gpu_tensor_free(tokens_view);
                }
                if (ok) {
                    ok = metal_graph_encode_prefill_stage_batch(g,
                                                                model,
                                                                weights,
                                                                &stages[stage_i],
                                                                pos0,
                                                                mb_len);
                }
                if (ok && stage_i + 1u < n_stages) {
                    ds4_gpu_tensor *src = g->batch_cur_hc_by_tier[stages[stage_i].tier];
                    ds4_gpu_tensor *dst = g->batch_cur_hc_by_tier[stages[stage_i + 1u].tier];
                    if (ok && getenv("DS4_CUDA_PREFILL_PIPELINE_SYNC_BOUNDARY") != NULL) {
                        ok = metal_graph_set_active_tier_no_copy(g, stages[stage_i].tier) &&
                             ds4_gpu_synchronize() != 0;
                    }
                    ok = src && dst &&
                         ds4_gpu_tensor_copy_xdev_ordered(dst,
                                                          src,
                                                          (uint64_t)mb_len * hc_dim * sizeof(float)) != 0;
                }
            }
            if (ok) ok = ds4_gpu_end_commands() != 0;
            else (void)ds4_gpu_synchronize();
            if (ok && display_progress) {
                uint32_t done = mb_off + mb_len;
                if (done > n_tokens) done = n_tokens;
                display_progress(display_progress_ud,
                                 "prefill_display",
                                 (int)(start + done),
                                 prompt->len);
            }
            if (show_progress) {
                fprintf(stderr, "ds4: gpu sequential pipeline prefill microbatch %u/%u\r",
                        mb_i + 1u,
                        n_mb);
                fflush(stderr);
            }
            if (ok && mb_i + 1u < n_mb) ok = ds4_gpu_begin_commands() != 0;
        }
    } else {
        for (uint32_t wave = 0; ok && wave < n_mb + n_stages - 1u; wave++) {
            uint32_t smax = wave < n_stages ? wave : n_stages - 1u;
            for (int si = (int)smax; ok && si >= 0; si--) {
                const uint32_t stage_i = (uint32_t)si;
                const uint32_t mb_i = wave - stage_i;
                if (mb_i >= n_mb) continue;
                const uint32_t mb_off = mb_i * mb_cap;
                uint32_t mb_len = n_tokens - mb_off;
                if (mb_len > mb_cap) mb_len = mb_cap;
                const uint32_t pos0 = start + mb_off;

                g->batch_token_offset = mb_off;
                if (stage_i == 0) {
                    ok = metal_graph_set_active_tier_no_copy(g, stages[0].tier);
                    ds4_gpu_tensor *tokens_view = NULL;
                    if (ok) {
                        tokens_view = ds4_gpu_tensor_view(metal_graph_prefill_tokens(g),
                                                           (uint64_t)mb_off * sizeof(int32_t),
                                                           (uint64_t)mb_len * sizeof(int32_t));
                        ok = tokens_view != NULL;
                    }
                    if (ok) {
                        ok = metal_graph_upload_prompt_embeddings_hc(
                                g->batch_cur_hc_by_tier[stages[0].tier],
                                tokens_view,
                                model,
                                weights,
                                prompt,
                                pos0,
                                mb_len);
                    }
                    ds4_gpu_tensor_free(tokens_view);
                }

                if (ok) {
                    ok = metal_graph_encode_prefill_stage_batch(g,
                                                                model,
                                                                weights,
                                                                &stages[stage_i],
                                                                pos0,
                                                                mb_len);
                }
                if (ok && stage_i + 1u < n_stages) {
                    ds4_gpu_tensor *src = g->batch_cur_hc_by_tier[stages[stage_i].tier];
                    ds4_gpu_tensor *dst = g->batch_cur_hc_by_tier[stages[stage_i + 1u].tier];
                    if (ok && getenv("DS4_CUDA_PREFILL_PIPELINE_SYNC_BOUNDARY") != NULL) {
                        ok = metal_graph_set_active_tier_no_copy(g, stages[stage_i].tier) &&
                             ds4_gpu_synchronize() != 0;
                    }
                    ok = src && dst &&
                         ds4_gpu_tensor_copy_xdev_ordered(dst,
                                                          src,
                                                          (uint64_t)mb_len * hc_dim * sizeof(float)) != 0;
                }
                if (ok && display_progress && stage_i + 1u == n_stages) {
                    uint32_t done = mb_off + mb_len;
                    if (done > n_tokens) done = n_tokens;
                    display_progress(display_progress_ud,
                                     "prefill_display",
                                     (int)(start + done),
                                     prompt->len);
                }
            }
            if (show_progress) {
                fprintf(stderr, "ds4: gpu pipeline prefill wave %u/%u\r",
                        wave + 1u,
                        n_mb + n_stages - 1u);
                fflush(stderr);
            }
        }
        if (ok) ok = ds4_gpu_end_commands() != 0;
        else (void)ds4_gpu_synchronize();
    }
    if (show_progress) fputc('\n', stderr);
    g->batch_token_offset = 0;
    if (!ok) {
        if (suppress_q8_cache) ds4_gpu_set_q8_cache_suppressed(saved_q8_cache_suppressed);
        return false;
    }

    const uint32_t final_len = n_tokens - (n_mb - 1u) * mb_cap;
    const int src_tier = stages[n_stages - 1u].tier;
    if (!metal_graph_set_active_tier_no_copy(g, src_tier)) {
        if (suppress_q8_cache) ds4_gpu_set_q8_cache_suppressed(saved_q8_cache_suppressed);
        return false;
    }

    ds4_gpu_tensor *saved_cur = g->cur_hc_by_tier[src_tier];
    ds4_gpu_tensor *last_hc = NULL;
    if (logits) {
        last_hc = metal_graph_tensor_row_view(g->batch_cur_hc_by_tier[src_tier],
                                              final_len - 1u,
                                              hc_dim);
        ok = last_hc != NULL;
    }
    if (ok && logits) {
        g->cur_hc_by_tier[src_tier] = last_hc;
        ok = ds4_gpu_begin_commands() != 0;
    }
    if (ok && logits) ok = metal_graph_encode_output_head(g, model, weights, weights->output->dim[1]);
    if (ok && logits) ok = ds4_gpu_end_commands() != 0;
    else if (!ok) (void)ds4_gpu_synchronize();
    g->cur_hc_by_tier[src_tier] = saved_cur;
    ds4_gpu_tensor_free(last_hc);
    if (g->placement && g->active_tier != src_tier) {
        ok = metal_graph_set_active_tier_no_copy(g, src_tier);
    }
    if (ok && logits) {
        ok = ds4_gpu_tensor_read(metal_graph_logits(g),
                                 0,
                                 logits,
                                 (uint64_t)DS4_N_VOCAB * sizeof(float)) != 0;
    }
    if (ok && display_progress)
        display_progress(display_progress_ud, "prefill_display",
                         (int)(start + n_tokens), prompt->len);
    if (ok && t0 != 0.0) {
        const double t1 = now_sec();
        fprintf(stderr,
                "ds4: gpu pipeline prefill total tokens=%u stages=%u mb=%u total=%.3f ms\n",
                n_tokens,
                n_stages,
                mb_cap,
                (t1 - t0) * 1000.0);
    }
    if (suppress_q8_cache) ds4_gpu_set_q8_cache_suppressed(saved_q8_cache_suppressed);
    return ok;
}

static bool metal_graph_prefill_layer_major(
        ds4_gpu_graph *g,
        const ds4_model       *model,
        const ds4_weights     *weights,
        const token_vec       *prompt,
        uint32_t               start,
        uint32_t               n_tokens,
        float                 *logits,
        bool                   show_progress,
        ds4_imatrix_collector *imatrix,
        ds4_session_progress_fn display_progress,
        void                  *display_progress_ud) {
    if (n_tokens == 0 || n_tokens > g->prefill_cap) return false;
    if (start > (uint32_t)prompt->len) return false;
    if (n_tokens > (uint32_t)prompt->len - start) return false;

    if (display_progress)
        display_progress(display_progress_ud, "prefill_display", (int)start, prompt->len);

    bool ok = metal_graph_upload_prompt_tokens(metal_graph_prefill_tokens(g), prompt, start, n_tokens);
    if (!ok) return false;

#ifdef DS4_ROCM_BUILD
    if (g->ssd_streaming &&
        DS4_MODEL_VARIANT == DS4_VARIANT_PRO &&
        n_tokens >= 1024u) {
        ds4_gpu_stream_expert_cache_release_resident();
    }
#endif

    if (!metal_graph_warmup_prefill_kernels(g, model, weights, n_tokens)) return false;
    if (g->placement &&
        !metal_graph_set_active_tier_no_copy(g, g->emb_tier)) {
        return false;
    }
    metal_graph_dspark_capture_begin_prefill(g);

    const bool split_profile =
        glm_graph_env_present("DS4_ROCM_GRAPH_PREFILL_SPLIT_PROFILE",
                              "DS4_METAL_GRAPH_PREFILL_SPLIT_PROFILE");
    /*
     * A full long-prompt prefill can keep the GPU busy for a long time. Split
     * non-tiny prefills when a frontend asked for display progress: completed
     * layer command buffers are real scheduling/keepalive points, while
     * callbacks emitted while encoding one huge command buffer would only be
     * cosmetic.
     */
    const bool throttle = graph_power_throttle_enabled(g);
    const bool callback_split = display_progress != NULL && n_tokens >= 32;
    const bool split_commands = g->ssd_streaming ||
                                split_profile || throttle || callback_split ||
                                n_tokens > 2048 || imatrix != NULL;
    const bool profile =
        glm_graph_env_present("DS4_ROCM_GRAPH_PREFILL_PROFILE",
                              "DS4_METAL_GRAPH_PREFILL_PROFILE") ||
        split_profile;
    const double t0 = profile ? now_sec() : 0.0;
    double encode_s = 0.0;
    double execute_s = 0.0;

    const uint32_t pipeline_mb = metal_graph_cuda_prefill_pipeline_microbatch();
    if (!split_commands &&
        !profile &&
        imatrix == NULL &&
        metal_graph_cuda_prefill_pipeline_requested(g) &&
        pipeline_mb != 0 &&
        pipeline_mb < n_tokens) {
        return metal_graph_prefill_pipeline_stage_major(g,
                                                        model,
                                                        weights,
                                                        prompt,
                                                        start,
                                                        n_tokens,
                                                        logits,
                                                        show_progress,
                                                        display_progress,
                                                        display_progress_ud);
    }

    if (!split_commands) {
        ok = metal_graph_upload_prompt_embeddings_hc(metal_graph_batch_cur_hc(g),
                                                     metal_graph_prefill_tokens(g),
                                                     model,
                                                     weights,
                                                     prompt,
                                                     start,
                                                     n_tokens);
        if (ok) ok = ds4_gpu_begin_commands() != 0;
        for (uint32_t il = 0; ok && il < DS4_N_LAYER; il++) {
            ok = metal_graph_encode_layer_batch(g,
                                                model,
                                                &weights->layer[il],
                                                il,
                                                start,
                                                n_tokens);
            if (!ok) {
                fprintf(stderr, "ds4: gpu whole-prefill layer %u encode failed\n", il);
            }
            if (ok) ok = metal_graph_dspark_capture_prefill_layer(g,
                                                                  il,
                                                                  start,
                                                                  n_tokens);
            if (show_progress) {
                fprintf(stderr, "ds4: gpu prefill layer %u/%u\r", il + 1, (uint32_t)DS4_N_LAYER);
                fflush(stderr);
            }
        }
        if (show_progress) fputc('\n', stderr);
        if (display_progress)
            display_progress(display_progress_ud, "prefill_display",
                             (int)(start + n_tokens), prompt->len);

        const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
        uint32_t output_row = (uint32_t)n_tokens - 1u;
        const char *output_row_env = glm_graph_env_value(
                "DS4_ROCM_GRAPH_OUTPUT_ROW",
                "DS4_METAL_GRAPH_OUTPUT_ROW");
        if (output_row_env && output_row_env[0]) {
            char *end = NULL;
            unsigned long v = strtoul(output_row_env, &end, 10);
            if (end != output_row_env && v < (unsigned long)n_tokens) {
                output_row = (uint32_t)v;
            }
        }
        const int src_tier = g->active_tier;
        ds4_gpu_tensor *saved_cur = g->cur_hc_by_tier[src_tier];
        ds4_gpu_tensor *last_hc = NULL;
        if (ok && logits) {
            last_hc = metal_graph_tensor_row_view(metal_graph_batch_cur_hc(g), output_row, hc_dim);
            ok = last_hc != NULL;
        }
        if (ok && logits) {
            g->cur_hc_by_tier[src_tier] = last_hc;
            ok = metal_graph_encode_output_head(g, model, weights, weights->output->dim[1]);
            g->cur_hc_by_tier[src_tier] = saved_cur;
        }

        const double t_encoded = profile ? now_sec() : 0.0;
        if (ok) ok = ds4_gpu_end_commands() != 0;
        const double t_done = profile ? now_sec() : 0.0;
        g->cur_hc_by_tier[src_tier] = saved_cur;
        if (last_hc) ds4_gpu_tensor_free(last_hc);
        if (!ok) {
            if (ds4_gpu_synchronize() == 0) {
                fprintf(stderr, "ds4: Metal synchronize after whole-prefill graph failure also failed\n");
            }
            return false;
        }
#ifdef __APPLE__
        ds4_gpu_release_zero_prefix_prefill_mask_cache();
#endif

        const double t_before_read = profile ? now_sec() : 0.0;
        if (logits) {
            ok = ds4_gpu_tensor_read(metal_graph_logits(g), 0, logits, (uint64_t)DS4_N_VOCAB * sizeof(float)) != 0;
        }
        if (profile) {
            const double t_read = now_sec();
            fprintf(stderr,
                    "ds4: gpu graph prefill total tokens=%u encode=%.3f ms execute=%.3f ms read=%.3f ms total=%.3f ms\n",
                    n_tokens,
                    (t_encoded - t0) * 1000.0,
                    (t_done - t_encoded) * 1000.0,
                    (t_read - t_before_read) * 1000.0,
                    (t_read - t0) * 1000.0);
        }
        return ok;
    }

    if (g->ssd_streaming) {
        g->streaming_static_decode_map_current = false;
        if (!metal_graph_stream_map_token(model, weights)) return false;
    }
    metal_graph_stream_prefill_selected_profile_reset(g);
    metal_graph_stream_prepare_slot layer_prepare_slots[DS4_STREAM_PREFILL_MAX_PREPARE_AHEAD];
    memset(layer_prepare_slots, 0, sizeof(layer_prepare_slots));
    const bool layer_pagein =
        metal_graph_stream_prefill_layer_pagein_enabled(g);
    const bool layer_readahead =
        !layer_pagein &&
        metal_graph_stream_prefill_layer_readahead_enabled(g);
    const bool layer_pread =
        !layer_pagein && !layer_readahead &&
        metal_graph_stream_prefill_layer_pread_enabled(g);
    const bool layer_madvise =
        !layer_pagein && !layer_pread && !layer_readahead &&
        metal_graph_stream_prefill_layer_madvise_enabled(g);
    const bool layer_prepare =
        layer_pagein || layer_pread || layer_readahead || layer_madvise;
    const bool layer_prepare_overlap =
        layer_prepare && metal_graph_stream_prefill_layer_pagein_overlap_enabled();
    const uint32_t layer_prepare_ahead =
        layer_prepare && layer_prepare_overlap ?
        metal_graph_stream_prefill_layer_prepare_ahead() : 1u;
    const bool batch_selected_addr =
        metal_graph_stream_prefill_batch_selected_addr_enabled(g, weights, n_tokens) ||
        metal_graph_cuda_stream_prefill_batch_selected_addr_enabled(g, weights, n_tokens);
#ifdef DS4_ROCM_BUILD
    rocm_graph_stream_layer_expert_load rocm_full_layer_load;
    memset(&rocm_full_layer_load, 0, sizeof(rocm_full_layer_load));
#endif
    if (g->ssd_streaming && DS4_N_LAYER > 0) {
        if (layer_prepare) {
            if (!metal_graph_stream_prepare_start_if_needed(g,
                                                            model,
                                                            weights,
                                                            0,
                                                            n_tokens,
                                                            layer_madvise,
                                                            layer_pread,
                                                            layer_readahead,
                                                            batch_selected_addr,
                                                            layer_prepare_slots,
                                                            layer_prepare_ahead)) {
                return false;
            }
        } else {
            if (batch_selected_addr) {
                metal_graph_stream_readahead_layer_decode(model, weights, 0);
            } else {
                metal_graph_stream_readahead_layer(model, weights, 0);
            }
        }
    }
#ifdef DS4_ROCM_BUILD
    if (g->ssd_streaming && DS4_N_LAYER > 0 &&
        !rocm_graph_stream_layer_expert_load_start_next(&rocm_full_layer_load,
                                                        g,
                                                        model,
                                                        weights,
                                                        0,
                                                        n_tokens)) {
        return false;
    }
#endif

    double t_layer0 = (profile || throttle) ? now_sec() : 0.0;
    ok = metal_graph_upload_prompt_embeddings_hc(metal_graph_batch_cur_hc(g),
                                                 metal_graph_prefill_tokens(g),
                                                 model,
                                                 weights,
                                                 prompt,
                                                 start,
                                                 n_tokens);
    const double t_embed_encoded = (profile || throttle) ? now_sec() : 0.0;
    const double t_embed_done = (profile || throttle) ? now_sec() : 0.0;
    if (profile) {
        encode_s += t_embed_encoded - t_layer0;
        execute_s += t_embed_done - t_embed_encoded;
        if (split_profile) {
            fprintf(stderr,
                    "ds4: metal layer-major prefill embed encode=%.3f ms execute=%.3f ms\n",
                    (t_embed_encoded - t_layer0) * 1000.0,
                    (t_embed_done - t_embed_encoded) * 1000.0);
        }
    }
    if (!ok) {
#ifdef DS4_ROCM_BUILD
        (void)rocm_graph_stream_layer_expert_load_join(&rocm_full_layer_load);
        (void)ds4_gpu_stream_expert_cache_release_layer_cache();
#endif
        if (layer_prepare) {
            (void)metal_graph_stream_prepare_join_all(layer_prepare_slots,
                                                      layer_prepare_ahead);
        }
        if (ds4_gpu_synchronize() == 0) {
            fprintf(stderr, "ds4: Metal synchronize after layer-major prefill embed failure also failed\n");
        }
        return false;
    }

    for (uint32_t il = 0; ok && il < DS4_N_LAYER; il++) {
        double layer_elapsed = 0.0;
        if (layer_prepare &&
            !metal_graph_stream_prepare_join_layer(g,
                                                   model,
                                                   weights,
                                                   il,
                                                   n_tokens,
                                                   layer_madvise,
                                                   layer_pread,
                                                   layer_readahead,
                                                   batch_selected_addr,
                                                   layer_prepare_slots,
                                                   layer_prepare_ahead)) {
            ok = false;
            break;
        }
#ifdef DS4_ROCM_BUILD
        const bool rocm_full_layer_stream_prefill =
            rocm_graph_stream_prefill_full_layer_enabled(g,
                                                         &weights->layer[il],
                                                         il,
                                                         n_tokens);
        if (rocm_full_layer_stream_prefill &&
            !rocm_graph_stream_layer_expert_load_ready(&rocm_full_layer_load,
                                                       g,
                                                       model,
                                                       weights,
                                                       il,
                                                       n_tokens)) {
            ok = false;
            break;
        }
        if (rocm_full_layer_stream_prefill &&
            !rocm_graph_stream_layer_expert_load_start_next(&rocm_full_layer_load,
                                                            g,
                                                            model,
                                                            weights,
                                                            il + 1u,
                                                            n_tokens)) {
            ok = false;
            break;
        }
#endif
        if (g->ssd_streaming) {
            g->streaming_static_decode_map_current = false;
            bool decode_only_map = batch_selected_addr;
#ifdef DS4_ROCM_BUILD
            decode_only_map = decode_only_map || rocm_full_layer_stream_prefill;
#endif
            const bool map_ok = decode_only_map ?
                metal_graph_stream_map_layer_decode(model, weights, il) :
                metal_graph_stream_map_layer(model, weights, il);
            if (!map_ok) {
                ok = false;
                break;
            }
        }
        if (g->ssd_streaming) {
            if (layer_prepare && layer_prepare_overlap) {
                bool started_future = false;
                for (uint32_t ahead = 1; ahead <= layer_prepare_ahead; ahead++) {
                    if (il + ahead >= DS4_N_LAYER) break;
                    started_future = true;
                    if (!metal_graph_stream_prepare_start_if_needed(g,
                                                                    model,
                                                                    weights,
                                                                    il + ahead,
                                                                    n_tokens,
                                                                    layer_madvise,
                                                                    layer_pread,
                                                                    layer_readahead,
                                                                    batch_selected_addr,
                                                                    layer_prepare_slots,
                                                                    layer_prepare_ahead)) {
                        ok = false;
                        break;
                    }
                }
                if (!ok) break;
                if (!started_future && logits) {
                    metal_graph_stream_readahead_output(model, weights);
                }
            } else if (!layer_prepare && il + 1 < DS4_N_LAYER) {
                if (batch_selected_addr) {
                    metal_graph_stream_readahead_layer_decode(model, weights, il + 1);
                } else {
                    metal_graph_stream_readahead_layer(model, weights, il + 1);
                }
            } else if (logits) {
                metal_graph_stream_readahead_output(model, weights);
            }
        }
        if (split_profile) {
            /* (B6 fix): split-profile diagnostic bypasses the
             * metal_graph_encode_layer_batch wrapper that normally does
             * the per-layer tier switch. Replicate the switch here so the
             * diagnostic / profile mode stays multi-tier-correct.
             * Single-tier (g->placement == NULL): no-op. */
            if (g->placement) {
                const int this_tier = g->placement[il + 1];
                if (!metal_graph_set_active_tier_batch(g, this_tier, (uint32_t)n_tokens)) {
                    ok = false;
                    break;
                }
            }
            const double t_attn0 = now_sec();
            ok = ds4_gpu_begin_commands() != 0;
            if (ok) ok = metal_graph_encode_layer_attention_batch(g,
                                                                  model,
                                                                  &weights->layer[il],
                                                                  il,
                                                                  start,
                                                                  n_tokens);
            if (!ok) {
                fprintf(stderr, "ds4: gpu layer-major prefill layer %u attention encode failed\n", il);
            }
            const double t_attn_encoded = now_sec();
            if (ok) ok = ds4_gpu_end_commands() != 0;
            const double t_attn_done = now_sec();

            const double t_ffn0 = now_sec();
            if (ok) ok = ds4_gpu_begin_commands() != 0;
            if (ok) ok = metal_graph_encode_layer_ffn_batch(g,
                                                            model,
                                                            &weights->layer[il],
                                                            il,
                                                            start,
                                                            n_tokens,
                                                            NULL,
                                                            0);
            if (!ok) {
                fprintf(stderr, "ds4: gpu layer-major prefill layer %u ffn encode failed\n", il);
            }
            if (ok) {
                ds4_gpu_tensor *tmp = metal_graph_batch_cur_hc(g);
                g->batch_cur_hc_by_tier[g->active_tier] = metal_graph_batch_next_hc(g);
                g->batch_next_hc_by_tier[g->active_tier] = tmp;
            }
            if (ok) ok = metal_graph_dspark_capture_prefill_layer(g,
                                                                  il,
                                                                  start,
                                                                  n_tokens);
            if (ok) ok = metal_graph_capture_prefill_seed_router_selected(g,
                                                                          il,
                                                                          n_tokens);
            const double t_ffn_encoded = now_sec();
            if (ok) ok = ds4_gpu_end_commands() != 0;
            const double t_ffn_done = now_sec();
#ifdef DS4_ROCM_BUILD
            if (ok) {
                ok = rocm_graph_stream_seed_full_layer_selected(g,
                                                                model,
                                                                &weights->layer[il],
                                                                il,
                                                                n_tokens);
            }
#endif
            if (ok) {
                ok = metal_graph_stream_prefill_selected_profile_layer(
                        g,
                        &weights->layer[il],
                        il,
                        n_tokens);
            }
            if (ok && imatrix) ok = imatrix_collect_layer_batch(imatrix, g, il, (uint32_t)n_tokens);
            layer_elapsed = (t_attn_done - t_attn0) + (t_ffn_done - t_ffn0);

            encode_s += (t_attn_encoded - t_attn0) + (t_ffn_encoded - t_ffn0);
            execute_s += (t_attn_done - t_attn_encoded) + (t_ffn_done - t_ffn_encoded);
            fprintf(stderr,
                    "ds4: metal layer-major prefill layer %u attn encode=%.3f execute=%.3f ms ffn encode=%.3f execute=%.3f ms\n",
                    il,
                    (t_attn_encoded - t_attn0) * 1000.0,
                    (t_attn_done - t_attn_encoded) * 1000.0,
                    (t_ffn_encoded - t_ffn0) * 1000.0,
                    (t_ffn_done - t_ffn_encoded) * 1000.0);
        } else {
            const double t_chunk0 = (profile || throttle) ? now_sec() : 0.0;
            ok = ds4_gpu_begin_commands() != 0;
            if (ok) ok = metal_graph_encode_layer_batch(g,
                                                        model,
                                                        &weights->layer[il],
                                                        il,
                                                        start,
                                                        n_tokens);
            if (!ok) {
                fprintf(stderr, "ds4: gpu layer-major prefill layer %u encode failed\n", il);
            }
            if (ok) ok = metal_graph_dspark_capture_prefill_layer(g,
                                                                  il,
                                                                  start,
                                                                  n_tokens);
            if (ok) ok = metal_graph_capture_prefill_seed_router_selected(g,
                                                                          il,
                                                                          n_tokens);
            const double t_encoded = (profile || throttle) ? now_sec() : 0.0;
            if (ok) ok = ds4_gpu_end_commands() != 0;
            const double t_done = (profile || throttle) ? now_sec() : 0.0;
#ifdef DS4_ROCM_BUILD
            if (ok) {
                ok = rocm_graph_stream_seed_full_layer_selected(g,
                                                                model,
                                                                &weights->layer[il],
                                                                il,
                                                                n_tokens);
            }
#endif
            if (ok) {
                ok = metal_graph_stream_prefill_selected_profile_layer(
                        g,
                        &weights->layer[il],
                        il,
                        n_tokens);
            }
            if (ok && imatrix) ok = imatrix_collect_layer_batch(imatrix, g, il, (uint32_t)n_tokens);
            layer_elapsed = t_done - t_chunk0;
            if (profile) {
                encode_s += t_encoded - t_chunk0;
                execute_s += t_done - t_encoded;
                fprintf(stderr,
                        "ds4: gpu layer-major prefill layer %u encode=%.3f ms execute=%.3f ms\n",
                        il,
                        (t_encoded - t_chunk0) * 1000.0,
                        (t_done - t_encoded) * 1000.0);
            }
        }
        if (ok &&
            g->ssd_streaming &&
            layer_prepare &&
            !layer_prepare_overlap) {
            if (il + 1 < DS4_N_LAYER) {
                if (!metal_graph_stream_prepare_start_if_needed(g,
                                                                model,
                                                                weights,
                                                                il + 1,
                                                                n_tokens,
                                                                layer_madvise,
                                                                layer_pread,
                                                                layer_readahead,
                                                                batch_selected_addr,
                                                                layer_prepare_slots,
                                                                layer_prepare_ahead)) {
                    ok = false;
                }
            } else if (logits) {
                metal_graph_stream_readahead_output(model, weights);
            }
        }
        if (!ok) {
#ifdef DS4_ROCM_BUILD
            (void)rocm_graph_stream_layer_expert_load_join(&rocm_full_layer_load);
            (void)ds4_gpu_stream_expert_cache_release_layer_cache();
#endif
            if (layer_prepare) {
                (void)metal_graph_stream_prepare_join_all(layer_prepare_slots,
                                                          layer_prepare_ahead);
            }
            if (ds4_gpu_synchronize() == 0) {
                fprintf(stderr, "ds4: Metal synchronize after layer-major prefill failure also failed\n");
            }
            return false;
        }
        graph_power_note_prefill_layer(g, il, layer_elapsed);
        gpu_graph_report_prefill_display_progress(display_progress,
                                                  display_progress_ud,
                                                  start,
                                                  n_tokens,
                                                  il + 1,
                                                  prompt->len);
        if (show_progress) {
            fprintf(stderr, "ds4: gpu prefill layer %u/%u\r", il + 1, (uint32_t)DS4_N_LAYER);
            fflush(stderr);
        }
    }
    if (!ok) {
#ifdef DS4_ROCM_BUILD
        (void)rocm_graph_stream_layer_expert_load_join(&rocm_full_layer_load);
        (void)ds4_gpu_stream_expert_cache_release_layer_cache();
#endif
        if (layer_prepare) {
            (void)metal_graph_stream_prepare_join_all(layer_prepare_slots,
                                                      layer_prepare_ahead);
        }
        if (ds4_gpu_synchronize() == 0) {
            fprintf(stderr, "ds4: Metal synchronize after layer-major prefill failure also failed\n");
        }
        return false;
    }
#ifdef __APPLE__
    /* Zero-prefix masks are shared across the 43 per-layer command batches,
     * then become dead weight. Release them before the output head and later
     * replay/decode chunks so the prefill win does not add residency pressure. */
    ds4_gpu_release_zero_prefix_prefill_mask_cache();
#endif
    if (show_progress) fputc('\n', stderr);
    metal_graph_stream_prefill_selected_profile_summary(g);
#ifdef DS4_ROCM_BUILD
    (void)ds4_gpu_stream_expert_cache_release_layer_cache();
    if (g->ssd_streaming) ds4_gpu_release_q8_f16_cache();
#endif
    if (!metal_graph_seed_streaming_expert_cache_from_hotlist(g, model, weights)) {
        return false;
    }
    if (!metal_graph_seed_streaming_expert_cache_from_prefill(g, model, weights)) {
        return false;
    }

    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    uint32_t output_row = (uint32_t)n_tokens - 1u;
    const char *output_row_env = glm_graph_env_value(
            "DS4_ROCM_GRAPH_OUTPUT_ROW",
            "DS4_METAL_GRAPH_OUTPUT_ROW");
    if (output_row_env && output_row_env[0]) {
        char *end = NULL;
        unsigned long v = strtoul(output_row_env, &end, 10);
        if (end != output_row_env && v < (unsigned long)n_tokens) {
            output_row = (uint32_t)v;
        }
    }
    ds4_gpu_tensor *saved_cur = metal_graph_cur_hc(g);
    ds4_gpu_tensor *last_hc = NULL;

    const double t_head0 = profile ? now_sec() : 0.0;
    if (logits) {
        last_hc = metal_graph_tensor_row_view(metal_graph_batch_cur_hc(g),
                                              output_row,
                                              hc_dim);
        ok = last_hc != NULL;
    }
    if (ok && logits && g->ssd_streaming) {
        const bool static_decode_map =
            metal_graph_stream_decode_static_map_enabled();
        const bool static_map_state_cache =
            static_decode_map &&
            metal_graph_stream_decode_static_map_state_cache_enabled();
        g->streaming_static_decode_map_current = false;
        if (static_map_state_cache) {
            ok = metal_graph_stream_map_decode_static_all(model, weights);
            if (ok) g->streaming_static_decode_map_current = true;
        } else {
            ok = metal_graph_stream_map_output(model, weights);
        }
    }
    if (ok && logits) {
        g->cur_hc_by_tier[g->active_tier] = last_hc;
        ok = ds4_gpu_begin_commands() != 0;
    }
    if (ok && logits) ok = metal_graph_encode_output_head(g, model, weights, weights->output->dim[1]);
    const double t_head_encoded = profile ? now_sec() : 0.0;
    if (ok && logits) ok = ds4_gpu_end_commands() != 0;
    const double t_head_done = profile ? now_sec() : 0.0;
    g->cur_hc_by_tier[g->active_tier] = saved_cur;
    if (last_hc) ds4_gpu_tensor_free(last_hc);
    if (!ok) return false;

    const double t_before_read = profile ? now_sec() : 0.0;
    if (logits) {
        ok = ds4_gpu_tensor_read(metal_graph_logits(g), 0, logits, (uint64_t)DS4_N_VOCAB * sizeof(float)) != 0;
    }
    if (profile) {
        const double t_read = now_sec();
        encode_s += t_head_encoded - t_head0;
        execute_s += t_head_done - t_head_encoded;
        if (split_profile) {
            fprintf(stderr,
                    "ds4: gpu layer-major prefill head encode=%.3f ms execute=%.3f ms\n",
                    (t_head_encoded - t_head0) * 1000.0,
                    (t_head_done - t_head_encoded) * 1000.0);
        }
        fprintf(stderr,
                "ds4: gpu layer-major prefill total tokens=%u encode=%.3f ms execute=%.3f ms read=%.3f ms total=%.3f ms\n",
                n_tokens,
                encode_s * 1000.0,
                execute_s * 1000.0,
                (t_read - t_before_read) * 1000.0,
                (t_read - t0) * 1000.0);
    }
    return ok;
}

static bool metal_graph_prefill_raw_swa(
        ds4_gpu_graph *g,
        const ds4_model       *model,
        const ds4_weights     *weights,
        const token_vec       *prompt,
        int                    n_tokens,
        float                 *logits,
        bool                   show_progress,
        ds4_session_progress_fn display_progress,
        void                  *display_progress_ud,
        ds4_session_cancel_fn  cancel,
        void                  *cancel_ud,
        bool                  *cancelled) {
    if (n_tokens <= 0 || n_tokens > prompt->len) return false;
    if ((uint32_t)n_tokens > g->prefill_cap) return false;
    if (metal_graph_use_streaming_decode_prefill_range(g, weights, 0,
                                                       (uint32_t)n_tokens)) {
        return metal_graph_prefill_decode_streaming_range(g,
                                                          model,
                                                          weights,
                                                          prompt,
                                                          0,
                                                          (uint32_t)n_tokens,
                                                          logits,
                                                          show_progress,
                                                          NULL,
                                                          NULL,
                                                          display_progress,
                                                          display_progress_ud,
                                                          cancel,
                                                          cancel_ud,
                                                          cancelled);
    }
    /* The layer-major fallback below may submit the whole short prefill as one
     * Metal command buffer.  Once that command is in flight there is no useful
     * safe prefix to expose: by the time cancellation can be observed again,
     * the prompt has already been fully read and the KV is valid.  Let the
     * caller observe the pending interrupt at generation time instead. */
    (void)cancel;
    (void)cancel_ud;
    (void)cancelled;
    return metal_graph_prefill_layer_major(g,
                                           model,
                                           weights,
                                           prompt,
                                           0,
                                           (uint32_t)n_tokens,
                                           logits,
                                           show_progress,
                                           NULL,
                                           display_progress,
                                           display_progress_ud);
}

/* Prefill a contiguous token range in fixed-size chunks.
 *
 * The common case starts at token zero, but server sessions also use this to
 * extend an existing KV cache with a long suffix.  Resumed chunks are aligned
 * to the same absolute prefill-cap boundaries used by a cold full prompt, so
 * compression windows and row finalization follow the same schedule after the
 * cached prefix.
 */
static bool metal_graph_prefill_chunked_range(
        ds4_gpu_graph *g,
        const ds4_model       *model,
        const ds4_weights     *weights,
        const token_vec       *prompt,
        uint32_t               start,
        uint32_t               n_tokens,
        float                 *logits,
        bool                   show_progress,
        ds4_session_progress_fn progress,
        void                  *progress_ud,
        ds4_session_progress_fn display_progress,
        void                  *display_progress_ud,
        ds4_imatrix_collector *imatrix,
        ds4_session_cancel_fn  cancel,
        void                  *cancel_ud,
        bool                  *cancelled) {
    if (n_tokens == 0 || g->prefill_cap == 0) return false;
    if (start > (uint32_t)prompt->len) return false;
    if (n_tokens > (uint32_t)prompt->len - start) return false;
    if (g->ssd_streaming && start == 0) {
        ds4_gpu_stream_expert_cache_reset_route_hotness();
    }
    if (!imatrix &&
        metal_graph_use_streaming_decode_prefill_range(g, weights,
                                                       start, n_tokens)) {
        return metal_graph_prefill_decode_streaming_range(g,
                                                          model,
                                                          weights,
                                                          prompt,
                                                          start,
                                                          n_tokens,
                                                          logits,
                                                          show_progress,
                                                          progress,
                                                          progress_ud,
                                                          display_progress,
                                                          display_progress_ud,
                                                          cancel,
                                                          cancel_ud,
                                                          cancelled);
    }

    uint32_t chunk_cap = g->prefill_cap;
    if (start != 0 && chunk_cap > g->raw_cap) chunk_cap = g->raw_cap;
    if (chunk_cap == 0) return false;

    const bool profile =
        glm_graph_env_present("DS4_ROCM_GRAPH_PREFILL_PROFILE",
                              "DS4_METAL_GRAPH_PREFILL_PROFILE");
    const double t0 = profile ? now_sec() : 0.0;
    const uint32_t end = start + n_tokens;

    if (progress) {
        progress(progress_ud, "prefill_chunk", (int)start, prompt->len);
    }
    if (display_progress) {
        display_progress(display_progress_ud, "prefill_display", (int)start, prompt->len);
    }

    for (uint32_t pos0 = start; pos0 < end; ) {
        if (cancel && cancel(cancel_ud)) {
            if (cancelled) *cancelled = true;
            return true;
        }
        const uint32_t remaining = end - pos0;
        uint32_t local_cap = chunk_cap;
        if (start != 0 && g->prefill_cap != 0) {
            const uint32_t mod = pos0 % g->prefill_cap;
            if (mod != 0) {
                const uint32_t to_boundary = g->prefill_cap - mod;
                if (to_boundary < local_cap) local_cap = to_boundary;
            }
        }
        const uint32_t chunk = remaining < local_cap ? remaining : local_cap;
        const uint32_t chunk_end = pos0 + chunk;
        float *chunk_logits = (progress || chunk_end == end) ? logits : NULL;
        bool ok = metal_graph_prefill_layer_major(g,
                                                  model,
                                                  weights,
                                                  prompt,
                                                  pos0,
                                                  chunk,
                                                  chunk_logits,
                                                  show_progress,
                                                  imatrix,
                                                  display_progress,
                                                  display_progress_ud);
        if (!ok) {
            if (ds4_gpu_synchronize() == 0) {
                fprintf(stderr, "ds4: Metal synchronize after chunked prefill failure also failed\n");
            }
            return false;
        }
        if (progress) {
            progress(progress_ud, "prefill_chunk", (int)chunk_end, prompt->len);
        }
        if (display_progress) {
            display_progress(display_progress_ud, "prefill_display", (int)chunk_end, prompt->len);
        }
        if (cancel && cancel(cancel_ud)) {
            if (cancelled) *cancelled = true;
            return true;
        }
        pos0 = chunk_end;
    }
    if (show_progress) fputc('\n', stderr);
    if (profile) {
        const double t_read = now_sec();
        fprintf(stderr,
                "ds4: gpu chunked prefill start=%u tokens=%u chunk=%u total=%.3f ms\n",
                start,
                n_tokens,
                chunk_cap,
                (t_read - t0) * 1000.0);
    }
    return true;
}

/* Long prompts are prefetched in fixed-size chunks.  Chunks bound transient
 * attention buffers while preserving the same final KV/cache state. */
static bool metal_graph_prefill_chunked(
        ds4_gpu_graph *g,
        const ds4_model       *model,
        const ds4_weights     *weights,
        const token_vec       *prompt,
        int                    n_tokens,
        float                 *logits,
        bool                   show_progress,
        ds4_session_progress_fn progress,
        void                  *progress_ud,
        ds4_session_progress_fn display_progress,
        void                  *display_progress_ud,
        ds4_session_cancel_fn  cancel,
        void                  *cancel_ud,
        bool                  *cancelled) {
    if (n_tokens <= 0) return false;
    return metal_graph_prefill_chunked_range(g,
                                             model,
                                             weights,
                                             prompt,
                                             0,
                                             (uint32_t)n_tokens,
                                             logits,
                                             show_progress,
                                             progress,
                                             progress_ud,
                                             display_progress,
                                             display_progress_ud,
                                             NULL,
                                             cancel,
                                             cancel_ud,
                                             cancelled);
}

typedef struct ds4_verify_suffix_timing {
    double upload_ms;
    double layer_ms;
    double head_ms;
    double read_ms;
    bool fused_head;
} ds4_verify_suffix_timing;

static bool metal_graph_dspark_verify_selected_profile_enabled(void) {
    return getenv("DS4_DSPARK_VERIFY_SELECTED_PROFILE") != NULL &&
           getenv("DS4_DSPARK_DISABLE_VERIFY_SELECTED_PROFILE") == NULL;
}

/* Layer-major speculative target verifier for tiny MTP suffixes.
 *
 * This is the first production-shaped verifier attempt: unlike repeated decode
 * it runs the target model layer-by-layer for the whole speculative suffix, and
 * unlike the diagnostic path it does not read back full logits for every row.
 * The verifier returns the row top-1 ids needed for acceptance.  The caller
 * then reads exactly one logits row: the row that becomes the new continuation
 * state.  It still reuses the existing batch layer kernels, so it is not yet
 * the final hand-written N=2/N=4 decode microbatch, but it exercises the right
 * verifier contract and removes the obvious diagnostic overheads first. */
static bool metal_graph_verify_suffix_tops_impl(
        ds4_gpu_graph *g,
        const ds4_model       *model,
        const ds4_weights     *weights,
        const token_vec       *prompt,
        uint32_t               start,
        uint32_t               n_tokens,
        bool                   capture_prefix1,
        bool                   capture_dspark_hidden,
        int                   *row_tops,
        float                 *row_logits,
        ds4_verify_suffix_timing *timing) {
    if (timing) memset(timing, 0, sizeof(*timing));
    if (n_tokens == 0 || n_tokens > g->prefill_cap || !g->spec_logits) return false;
    if (start > (uint32_t)prompt->len || n_tokens > (uint32_t)prompt->len - start) return false;
    const uint32_t top_rows = n_tokens > 1 ? n_tokens - 1 : 0;
    if (top_rows && !row_tops) return false;

    const double upload_t0 = timing ? now_sec() : 0.0;
    bool ok = metal_graph_upload_prompt_tokens(metal_graph_prefill_tokens(g), prompt, start, n_tokens);
    if (ok) ok = metal_graph_upload_prompt_embeddings_hc(metal_graph_batch_cur_hc(g),
                                                         metal_graph_prefill_tokens(g),
                                                         model,
                                                         weights,
                                                         prompt,
                                                         start,
                                                         n_tokens);
    if (!ok) return false;

    const bool saved_capture = g->spec_capture_prefix1;
    g->spec_capture_prefix1 = capture_prefix1 && n_tokens == 2;
    const char *split_head_env = getenv("DS4_DSPARK_VERIFY_SPLIT_HEAD");
    const bool fuse_head =
        !split_head_env || !split_head_env[0] ||
        strcmp(split_head_env, "0") == 0;
    if (timing) timing->fused_head = fuse_head;
    if (timing) timing->upload_ms += (now_sec() - upload_t0) * 1000.0;

    const bool selected_profile =
        metal_graph_dspark_verify_selected_profile_enabled();
    if (selected_profile) {
        metal_graph_stream_prefill_selected_profile_reset(g);
    }

    /* Under TP, verify every speculative block against the two resident
     * expert halves. Both ranks encode identically, so one batch gate per
     * layer reconstructs the routed result while preserving their KV state. */
    g->tp_batch_rows = (g->tp_world == 2 &&
                        g->tp_batch_out != NULL && g->tp_batch_in != NULL &&
                        n_tokens <= (uint32_t)DS4_TP_BATCH_MAX_ROWS)
                       ? n_tokens : 0;
    const double layer_t0 = timing ? now_sec() : 0.0;
    ok = ds4_gpu_begin_commands() != 0;
    const bool dspark_capture_active =
        ok &&
        capture_dspark_hidden &&
        metal_graph_dspark_capture_verified_suffix_begin(g,
                                                         start,
                                                         n_tokens,
                                                         true);
    static int verify_profile_left = -1;
    if (verify_profile_left < 0) {
        verify_profile_left =
            getenv("DS4_DSPARK_VERIFY_PROFILE") != NULL ? 1 : 0;
    }
    const bool verify_profile = verify_profile_left > 0 && ok;
    if (verify_profile) verify_profile_left--;
    for (uint32_t il = 0; ok && il < DS4_N_LAYER; il++) {
        if (verify_profile) {
            ok = ds4_gpu_end_commands() != 0;
            if (ok) (void)ds4_gpu_synchronize();
            if (ok) ok = ds4_gpu_begin_commands() != 0;
            if (!ok) break;
        }
        ok = metal_graph_encode_layer_batch(g,
                                            model,
                                            &weights->layer[il],
                                            il,
                                            start,
                                            n_tokens);
        if (ok && dspark_capture_active) {
            ok = metal_graph_dspark_capture_verified_suffix_layer(g,
                                                                  il,
                                                                  start,
                                                                  n_tokens);
        }
        if (ok && selected_profile) {
            ok = ds4_gpu_end_commands() != 0 &&
                 metal_graph_selected_profile_layer_impl(
                         g,
                         &weights->layer[il],
                         il,
                         n_tokens,
                         "DSpark verifier selected profile") &&
                 ds4_gpu_begin_commands() != 0;
        }
    }
    g->tp_batch_rows = 0;
    if (ok && fuse_head) {
        ok = metal_graph_encode_output_head_batch(g,
                                                  model,
                                                  weights,
                                                  n_tokens,
                                                  weights->output->dim[1]);
    }
    if (ok && fuse_head) {
        if (top_rows == 1) {
            ok = ds4_gpu_argmax_tensor(metal_graph_comp_selected(g),
                                       g->spec_logits,
                                       DS4_N_VOCAB) != 0;
        } else if (top_rows) {
            ok = ds4_gpu_indexer_topk_tensor(metal_graph_comp_selected(g),
                                             g->spec_logits,
                                             DS4_N_VOCAB,
                                             top_rows,
                                             1) != 0;
        }
    }
    if (ok) ok = ds4_gpu_end_commands() != 0;
    else (void)ds4_gpu_synchronize();
    g->spec_capture_prefix1 = saved_capture;
    if (!ok && dspark_capture_active) {
        metal_graph_dspark_capture_invalidate(g);
    }
    if (timing) timing->layer_ms += (now_sec() - layer_t0) * 1000.0;
    if (!ok) return false;
    if (selected_profile) {
        metal_graph_selected_profile_summary_impl(
                g,
                "DSpark verifier selected profile");
    }

    if (!fuse_head) {
        const double head_t0 = timing ? now_sec() : 0.0;
        ok = ds4_gpu_begin_commands() != 0;
        if (ok) ok = metal_graph_encode_output_head_batch(g,
                                                          model,
                                                          weights,
                                                          n_tokens,
                                                          weights->output->dim[1]);
        if (ok) {
            if (top_rows == 1) {
                /* Common K=2 verify case: top_k=1 over n_vocab → use the dedicated
                 * argmax kernel (single-block tree-reduce) instead of the legacy
                 * indexer_topk_kernel's single-thread O(n_vocab * top_k) fall-through. */
                ok = ds4_gpu_argmax_tensor(metal_graph_comp_selected(g),
                                           g->spec_logits,
                                           DS4_N_VOCAB) != 0;
            } else if (top_rows) {
                /* top-1 of each of the top_rows rows: n_tokens=top_rows, top_k=1.
                 * The order is transposed vs the indexer-score callers; a swap
                 * silently scores row 0's runner-ups instead of each row. */
                ok = ds4_gpu_indexer_topk_tensor(metal_graph_comp_selected(g),
                                                   g->spec_logits,
                                                   DS4_N_VOCAB,
                                                   top_rows,
                                                   1) != 0;
            }
        }
        if (ok) ok = ds4_gpu_end_commands() != 0;
        else (void)ds4_gpu_synchronize();
        if (timing) timing->head_ms += (now_sec() - head_t0) * 1000.0;
    }
    const double read_t0 = timing ? now_sec() : 0.0;
    if (ok && top_rows) {
        ok = ds4_gpu_tensor_read(metal_graph_comp_selected(g),
                                   0,
                                   row_tops,
                                   (uint64_t)top_rows * sizeof(row_tops[0])) != 0;
        if (ok && getenv("DS4_DSPARK_VERIFY_TOPS_CHECK") != NULL) {
            float *chk = xmalloc((size_t)DS4_N_VOCAB * sizeof(float));
            for (uint32_t r = 0; r < top_rows; r++) {
                if (ds4_gpu_tensor_read(g->spec_logits,
                                        (uint64_t)r * DS4_N_VOCAB *
                                            sizeof(float),
                                        chk,
                                        (uint64_t)DS4_N_VOCAB *
                                            sizeof(float)) == 0) break;
                uint32_t am = 0;
                for (uint32_t i = 1; i < DS4_N_VOCAB; i++) {
                    if (chk[i] > chk[am]) am = i;
                }
                fprintf(stderr,
                        "ds4: verify tops-check row=%u gpu_top=%d cpu_argmax=%u "
                        "cpu_max=%.3f\n",
                        r, row_tops[r], am, chk[am]);
            }
            free(chk);
        }
    }
    if (ok && row_logits) {
        ok = ds4_gpu_tensor_read(g->spec_logits,
                                   0,
                                   row_logits,
                                   (uint64_t)n_tokens * DS4_N_VOCAB * sizeof(row_logits[0])) != 0;
    }
    if (timing) timing->read_ms += (now_sec() - read_t0) * 1000.0;
    return ok;
}

/* The verify block keeps the GPU genuinely busy, so the TP DVFS
 * keep-alive is a pure parasite for its duration — pause it. */
static bool metal_graph_verify_suffix_tops(
        ds4_gpu_graph *g,
        const ds4_model       *model,
        const ds4_weights     *weights,
        const token_vec       *prompt,
        uint32_t               start,
        uint32_t               n_tokens,
        bool                   capture_prefix1,
        bool                   capture_dspark_hidden,
        int                   *row_tops,
        float                 *row_logits,
        ds4_verify_suffix_timing *timing) {
    ds4_gpu_tp_keepalive_pause(1);
    const bool ok = metal_graph_verify_suffix_tops_impl(g, model, weights,
                                                        prompt, start,
                                                        n_tokens,
                                                        capture_prefix1,
                                                        capture_dspark_hidden,
                                                        row_tops, row_logits,
                                                        timing);
    ds4_gpu_tp_keepalive_pause(0);
    return ok;
}

static bool metal_graph_read_spec_logits_row(ds4_gpu_graph *g, uint32_t row, float *logits) {
    if (!g || !g->spec_logits || !logits || row >= g->prefill_cap) return false;
    const uint64_t row_bytes = (uint64_t)DS4_N_VOCAB * sizeof(float);
    return ds4_gpu_tensor_read(g->spec_logits,
                                 (uint64_t)row * row_bytes,
                                 logits,
                                 row_bytes) != 0;
}

/* Exact N=2 target verifier for MTP.
 *
 * The generic batch prefill path is fast, but it is not a safe substitute for
 * autoregressive decode: small row-wise differences in HC/MoE/output kernels
 * are enough to flip future greedy tokens.  This verifier keeps the exact
 * decode kernels and cache update order, but encodes the two proposed tokens
 * layer-by-layer in one command stream.  It returns the exact target top after
 * token0, and exact logits after token1. */
static bool metal_graph_verify_decode2_exact(
        ds4_gpu_graph *g,
        const ds4_model       *model,
        const ds4_weights     *weights,
        int                    token0,
        int                    token1,
        uint32_t               start,
        int                   *top0,
        int                   *top1,
        float                 *logits0,
        float                 *logits1) {
    if (!g || !top0 || (!top1 && !logits1) || g->raw_cap == 0) return false;

    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const uint64_t hc_bytes = hc_dim * sizeof(float);
    ds4_gpu_tensor *cur0_by_tier[DS4_MAX_GPUS] = {0};
    ds4_gpu_tensor *cur1_by_tier[DS4_MAX_GPUS] = {0};
    ds4_gpu_tensor *next0_by_tier[DS4_MAX_GPUS] = {0};
    ds4_gpu_tensor *next1_by_tier[DS4_MAX_GPUS] = {0};
    ds4_gpu_tensor *saved_cur_by_tier[DS4_MAX_GPUS] = {0};
    ds4_gpu_tensor *saved_after_by_tier[DS4_MAX_GPUS] = {0};
    const int saved_active_tier = g->active_tier;
    const bool saved_capture = g->spec_capture_prefix1;

    bool ok = true;
    for (int t = 0; t < DS4_MAX_GPUS; t++) {
        saved_cur_by_tier[t] = g->cur_hc_by_tier[t];
        saved_after_by_tier[t] = g->after_ffn_hc_by_tier[t];
        if (!g->batch_cur_hc_by_tier[t] && !g->batch_next_hc_by_tier[t]) continue;
        if (!g->batch_cur_hc_by_tier[t] || !g->batch_next_hc_by_tier[t]) {
            ok = false;
            break;
        }
        cur0_by_tier[t] = metal_graph_tensor_row_view(g->batch_cur_hc_by_tier[t], 0, hc_dim);
        cur1_by_tier[t] = metal_graph_tensor_row_view(g->batch_cur_hc_by_tier[t], 1, hc_dim);
        next0_by_tier[t] = metal_graph_tensor_row_view(g->batch_next_hc_by_tier[t], 0, hc_dim);
        next1_by_tier[t] = metal_graph_tensor_row_view(g->batch_next_hc_by_tier[t], 1, hc_dim);
        if (!cur0_by_tier[t] || !cur1_by_tier[t] ||
            !next0_by_tier[t] || !next1_by_tier[t]) {
            ok = false;
            break;
        }
    }

    int cur_tier = g->emb_tier;
    if (cur_tier < 0 || cur_tier >= DS4_MAX_GPUS ||
        !cur0_by_tier[cur_tier] || !cur1_by_tier[cur_tier]) {
        ok = false;
    }
    if (ok) ok = metal_graph_set_active_tier_no_copy(g, cur_tier);
    if (ok) ok = ds4_gpu_embed_token_hc_tensor(cur0_by_tier[cur_tier],
                                               model->map,
                                               model->size,
                                               weights->token_embd->abs_offset,
                                               (uint32_t)weights->token_embd->dim[1],
                                               (uint32_t)token0,
                                               DS4_N_EMBD,
                                               DS4_N_HC) != 0;
    if (ok) ok = ds4_gpu_embed_token_hc_tensor(cur1_by_tier[cur_tier],
                                               model->map,
                                               model->size,
                                               weights->token_embd->abs_offset,
                                               (uint32_t)weights->token_embd->dim[1],
                                               (uint32_t)token1,
                                               DS4_N_EMBD,
                                               DS4_N_HC) != 0;

    g->spec_capture_prefix1 = true;
    if (ok) ok = ds4_gpu_begin_commands() != 0;
    for (uint32_t il = 0; ok && il < DS4_N_LAYER; il++) {
        const uint32_t pos0 = start;
        const uint32_t pos1 = start + 1u;
        const int this_tier = g->placement ? g->placement[il + 1] : cur_tier;
        if (this_tier < 0 || this_tier >= DS4_MAX_GPUS ||
            !cur0_by_tier[this_tier] || !cur1_by_tier[this_tier]) {
            ok = false;
            break;
        }
        if (this_tier != cur_tier) {
            ok = ds4_gpu_tensor_copy_xdev(cur0_by_tier[this_tier],
                                          cur0_by_tier[cur_tier],
                                          hc_bytes) != 0 &&
                 ds4_gpu_tensor_copy_xdev(cur1_by_tier[this_tier],
                                          cur1_by_tier[cur_tier],
                                          hc_bytes) != 0;
            if (!ok) break;
            cur_tier = this_tier;
        }
        ok = metal_graph_set_active_tier_no_copy(g, this_tier);
        if (!ok) break;

        g->cur_hc_by_tier[this_tier] = cur0_by_tier[this_tier];
        g->after_ffn_hc_by_tier[this_tier] = next0_by_tier[this_tier];
        ok = metal_graph_encode_decode_layer(g,
                                             model,
                                             &weights->layer[il],
                                             il,
                                             pos0,
                                             g->layer_raw_cache[il],
                                             g->raw_cap,
                                             pos0 % g->raw_cap,
                                             metal_graph_raw_span_for_batch(g, pos0, 1),
                                             token0);
        if (!ok) break;
        ok = metal_graph_capture_prefix1_attn_state(g, il) &&
             metal_graph_capture_prefix1_index_state(g, il);
        if (!ok) break;

        g->cur_hc_by_tier[this_tier] = cur1_by_tier[this_tier];
        g->after_ffn_hc_by_tier[this_tier] = next1_by_tier[this_tier];
        ok = metal_graph_encode_decode_layer(g,
                                             model,
                                             &weights->layer[il],
                                             il,
                                             pos1,
                                             g->layer_raw_cache[il],
                                             g->raw_cap,
                                             pos1 % g->raw_cap,
                                             metal_graph_raw_span_for_batch(g, pos1, 1),
                                             token1);
        if (!ok) break;

        ds4_gpu_tensor *tmp = cur0_by_tier[this_tier];
        cur0_by_tier[this_tier] = next0_by_tier[this_tier];
        next0_by_tier[this_tier] = tmp;
        tmp = cur1_by_tier[this_tier];
        cur1_by_tier[this_tier] = next1_by_tier[this_tier];
        next1_by_tier[this_tier] = tmp;
    }
    if (ok) ok = ds4_gpu_end_commands() != 0;
    else (void)ds4_gpu_synchronize();
    g->spec_capture_prefix1 = saved_capture;

    if (ok) {
        ok = metal_graph_set_active_tier_no_copy(g, cur_tier);
    }
    if (ok) {
        const bool split_top1 =
            logits0 == NULL &&
            g->cuda_tp_output &&
            metal_graph_cuda_verify_decode2_split_top1_requested();
        uint32_t output_ways = 0;
        g->cur_hc_by_tier[cur_tier] = cur0_by_tier[cur_tier];
        ok = ds4_gpu_begin_commands() != 0;
        if (ok) ok = metal_graph_encode_output_head(g, model, weights, weights->output->dim[1]);
        if (ok) ok = ds4_gpu_argmax_tensor(metal_graph_comp_selected(g),
                                           metal_graph_logits(g),
                                           DS4_N_VOCAB) != 0;
        if (ok) ok = ds4_gpu_end_commands() != 0;
        else (void)ds4_gpu_synchronize();
        if (ok && split_top1) {
            ok = metal_graph_read_output_split_top1(g, output_ways, top0);
        } else if (ok) {
            ok = ds4_gpu_tensor_read(metal_graph_comp_selected(g), 0, top0, sizeof(*top0)) != 0;
        }
        if (ok && logits0) {
            ok = ds4_gpu_tensor_read(metal_graph_logits(g),
                                       0,
                                       logits0,
                                       (uint64_t)DS4_N_VOCAB * sizeof(logits0[0])) != 0;
        }
    }

    if (ok) {
        ok = metal_graph_set_active_tier_no_copy(g, cur_tier);
    }
    if (ok) {
        const bool split_top1 =
            logits1 == NULL &&
            top1 != NULL &&
            g->cuda_tp_output &&
            metal_graph_cuda_verify_decode2_split_top1_requested();
        int output_tiers[DS4_MAX_GPUS] = {0};
        uint32_t output_ways = 0;
        g->cur_hc_by_tier[cur_tier] = cur1_by_tier[cur_tier];
        ok = ds4_gpu_begin_commands() != 0;
        if (ok && split_top1) {
            ok = metal_graph_encode_output_head_split_top1(g,
                                                           model,
                                                           weights,
                                                           weights->output->dim[1],
                                                           output_tiers,
                                                           &output_ways);
        } else if (ok) {
            ok = metal_graph_encode_output_head(g, model, weights, weights->output->dim[1]);
            if (ok && top1) {
                ok = ds4_gpu_indexer_topk_tensor(metal_graph_comp_selected(g),
                                                 metal_graph_logits(g),
                                                 DS4_N_VOCAB,
                                                 1,
                                                 1) != 0;
            }
        }
        if (ok) ok = ds4_gpu_end_commands() != 0;
        else (void)ds4_gpu_synchronize();
        if (ok && split_top1) {
            ok = metal_graph_read_output_split_top1(g, output_ways, top1);
        } else if (ok && top1) {
            ok = ds4_gpu_tensor_read(metal_graph_comp_selected(g), 0, top1, sizeof(*top1)) != 0;
        }
        if (ok) {
            if (logits1) {
                ok = ds4_gpu_tensor_read(metal_graph_logits(g),
                                           0,
                                           logits1,
                                           (uint64_t)DS4_N_VOCAB * sizeof(logits1[0])) != 0;
            }
        }
    }
    g->spec_capture_prefix1 = saved_capture;
    for (int t = 0; t < DS4_MAX_GPUS; t++) {
        g->cur_hc_by_tier[t] = saved_cur_by_tier[t];
        g->after_ffn_hc_by_tier[t] = saved_after_by_tier[t];
    }
    if (g->placement) {
        if (saved_active_tier >= 0) {
            (void)metal_graph_set_active_tier_no_copy(g, saved_active_tier);
        } else {
            g->active_tier = saved_active_tier;
        }
    }
    for (int t = 0; t < DS4_MAX_GPUS; t++) {
        ds4_gpu_tensor_free(next1_by_tier[t]);
        ds4_gpu_tensor_free(next0_by_tier[t]);
        ds4_gpu_tensor_free(cur1_by_tier[t]);
        ds4_gpu_tensor_free(cur0_by_tier[t]);
    }
    return ok;
}

/* Pick a raw SWA cache size for Metal.  During batched prefill it must cover
 * the previous window plus the current ubatch. */
static uint32_t metal_graph_raw_cap_for_context(int ctx_size, uint32_t prefill_cap) {
    uint32_t raw_window = DS4_N_SWA;
    if (raw_window > (uint32_t)ctx_size) raw_window = (uint32_t)ctx_size;
    if (raw_window == 0) raw_window = 1;

    /*
     * During batched prefill the SWA cache must hold the current ubatch plus
     * the previous logical window. The cache is padded to a 256-row multiple
     * so the physical row order and FlashAttention block grouping match the
     * model path we compare against.
     */
    uint64_t wanted = (uint64_t)raw_window + prefill_cap;
    if (wanted > (uint32_t)ctx_size) wanted = (uint32_t)ctx_size;
    if (wanted == 0) wanted = 1;
    wanted = align_up(wanted, 256u);
    if (wanted > 8192u) wanted = 8192u;
    uint32_t raw_cap = (uint32_t)wanted;
    if (raw_cap < raw_window) raw_cap = raw_window;

#ifndef DS4_ROCM_BUILD
    const char *env = getenv("DS4_METAL_GRAPH_RAW_CAP");
    if (env && env[0]) {
        char *endp = NULL;
        const long v = strtol(env, &endp, 10);
        if (endp != env && v > 0) {
            raw_cap = (uint32_t)v;
            if (raw_cap > (uint32_t)ctx_size) raw_cap = (uint32_t)ctx_size;
            if (raw_cap > 8192u) raw_cap = 8192u;
            if (raw_cap < raw_window) raw_cap = raw_window;
        }
    }
#endif

    return raw_cap;
}

/* Choose the prefill ubatch size. Whole-batch is fastest for normal prompts.
 * Long Flash prompts default to 4096-token chunks; PRO defaults to 8192. */
static uint32_t metal_graph_prefill_cap_for_prompt(int prompt_len,
                                                   uint32_t prefill_chunk) {
    return ds4_prefill_cap_for_prompt(prompt_len, prefill_chunk);
}

/* When a server request shares a large prefix with the live checkpoint, extend
 * the KV cache with batched prefill instead of single-token decode.  On an M3
 * Max, prefill is faster from 2-token suffixes upward; keep the default at 4
 * as a conservative crossover.  The env knob remains useful for retuning. */
static uint32_t metal_graph_resume_prefill_min_tokens(void) {
#ifndef DS4_ROCM_BUILD
    const char *env = getenv("DS4_METAL_RESUME_PREFILL_MIN");
    if (env && env[0]) {
        char *endp = NULL;
        const long v = strtol(env, &endp, 10);
        if (endp != env) {
            if (v <= 0) return UINT32_MAX;
            return (uint32_t)v;
        }
    }
#endif
    return 4u;
}

static uint32_t glm_graph_resume_prefill_min_tokens(void) {
#ifndef DS4_ROCM_BUILD
    const char *env = getenv("DS4_GLM_RESUME_PREFILL_MIN");
    if (env && env[0]) {
        char *endp = NULL;
        const long v = strtol(env, &endp, 10);
        if (endp != env) {
            if (v <= 0) return UINT32_MAX;
            return (uint32_t)v;
        }
    }
#endif
    return 4u;
}

#define DS4_GLM_METAL_FULL_ATTN_DEFAULT_CONTEXT 4096u
#define DS4_GLM_METAL_STREAMING_FULL_ATTN_CONTEXT 8192u
#define DS4_GLM_METAL_FULL_ATTN_LAYER_FLUSH_CONTEXT 2048u
#define DS4_GLM_METAL_DISPLAY_PROGRESS_LAYER_TOKENS 32u
#define DS4_GLM_METAL_SMALL_PREFILL_STAGE_SYNC_TOKENS 0u
#define DS4_GLM_METAL_LONG_CONTEXT_THRESHOLD 65536u
#define DS4_GLM_METAL_LONG_CONTEXT_FULL_ATTN_CONTEXT 4096u
#define DS4_GLM_METAL_INDEXED_PREFILL_CHUNK_TOKENS 4096u
#define DS4_GLM_METAL_INDEXED_PREFILL_SCORE_SCRATCH_MB 256u

static uint32_t glm_graph_full_attention_cap(uint32_t ctx_size,
                                             bool     ssd_streaming);
static uint32_t glm_graph_indexed_prefill_chunk_tokens(
        uint32_t full_attention_cap,
        uint32_t compact_cap);
static uint32_t glm_graph_indexed_prefill_score_tokens(
        uint32_t indexed_prefill_cap,
        uint32_t compact_cap);

static uint64_t glm_graph_compact_cache_elem_bytes(void) {
    return DS4_GPU_GLM_COMPACT_CACHE_F16 ? sizeof(uint16_t) : sizeof(float);
}

static uint32_t glm_graph_compact_cache_is_f16(void) {
    return DS4_GPU_GLM_COMPACT_CACHE_F16 ? 1u : 0u;
}

static bool glm_graph_expanded_kv_cache_enabled(bool ssd_streaming) {
    (void)ssd_streaming;
    return false;
}

static bool glm_graph_layer_uses_full_indexer(uint32_t il) {
    if (il < DS4_N_LEADING_DENSE) return true;
    return il >= 6u && ((il - 6u) % 4u) == 0u;
}

static uint32_t glm_graph_normal_layer_count(void) {
    if (DS4_N_LAYER <= DS4_N_NEXTN_PREDICT || DS4_N_LAYER > DS4_MAX_LAYER) {
        return 0;
    }
    return DS4_N_LAYER - DS4_N_NEXTN_PREDICT;
}

static uint32_t glm_graph_full_indexer_layer_count_range(uint32_t layer_start,
                                                         uint32_t layer_end) {
    if (layer_start > layer_end) return 0;
    uint32_t n = 0;
    for (uint32_t il = layer_start; il <= layer_end; il++) {
        if (glm_graph_layer_uses_full_indexer(il)) n++;
    }
    return n;
}

static uint64_t glm_graph_full_kv_cache_elem_bytes(void) {
    return sizeof(uint16_t);
}

static uint32_t glm_graph_indexer_top_k_limit(void) {
    return DS4_N_INDEXER_TOP_K;
}

static uint32_t glm_tp_head_split_min(void) {
    static int cached = -1;
    if (cached < 0) {
        cached = 64;
        const char *env = getenv("DS4_GLM_TP_HEAD_SPLIT_MIN");
        if (env && env[0]) cached = atoi(env);
        if (cached < 0) cached = 0;
    }
    return (uint32_t)cached;
}

/* Correctness isolation: dump a hidden row (pre-output-norm), overwriting
 * on each call — run with -n 0 so the file ends as the final prompt row. */
static void glm_debug_dump_hidden_row(const ds4_gpu_tensor *t, uint32_t row) {
    const char *path = getenv("DS4_GLM_HIDDEN_DUMP");
    if (!path || !path[0] || !t) return;
    float *buf = malloc((size_t)DS4_N_EMBD * sizeof(float));
    if (!buf) return;
    if (ds4_gpu_tensor_read((ds4_gpu_tensor *)t,
                            (uint64_t)row * DS4_N_EMBD * sizeof(float),
                            buf,
                            (uint64_t)DS4_N_EMBD * sizeof(float))) {
        FILE *f = fopen(path, "wb");
        if (f) {
            fwrite(buf, sizeof(float), (size_t)DS4_N_EMBD, f);
            fclose(f);
        }
    }
    free(buf);
}

/* Layer bisect: which layer's output hidden to dump (-1 = final/off,
 * -2 = every layer, one file per layer). */
static int glm_debug_hidden_dump_layer(void) {
    const char *v = getenv("DS4_GLM_HIDDEN_DUMP_LAYER");
    if (!v || !v[0]) return -1;
    if (strcmp(v, "all") == 0) return -2;
    return atoi(v);
}

static bool glm_debug_hidden_dump_layer_match(uint32_t il) {
    const int dl = glm_debug_hidden_dump_layer();
    return dl == -2 || dl == (int)il;
}

static void glm_debug_dump_raw_layer(const ds4_gpu_tensor *t,
                                     const char *tag,
                                     uint64_t bytes,
                                     uint32_t il,
                                     int pos) {
    const char *path = getenv("DS4_GLM_HIDDEN_DUMP");
    if (!path || !path[0] || !t) return;
    char full[1024];
    if (pos >= 0)
        snprintf(full, sizeof(full), "%s.%s.L%02u.T%02u", path, tag, il, pos);
    else
        snprintf(full, sizeof(full), "%s.%s.L%02u", path, tag, il);
    void *buf = malloc(bytes);
    if (!buf) return;
    if (ds4_gpu_tensor_read((ds4_gpu_tensor *)t, 0, buf, bytes)) {
        FILE *f = fopen(full, "wb");
        if (f) { fwrite(buf, 1, bytes, f); fclose(f); }
    }
    free(buf);
}

static void glm_debug_dump_hidden_layer(const ds4_gpu_tensor *t,
                                        uint32_t row,
                                        uint32_t il,
                                        uint32_t pos) {
    const char *path = getenv("DS4_GLM_HIDDEN_DUMP");
    if (!path || !path[0] || !t) return;
    char full[1024];
    snprintf(full, sizeof(full), "%s.L%02u.T%02u", path, il, pos);
    float *buf = malloc((size_t)DS4_N_EMBD * sizeof(float));
    if (!buf) return;
    if (ds4_gpu_tensor_read((ds4_gpu_tensor *)t,
                            (uint64_t)row * DS4_N_EMBD * sizeof(float),
                            buf,
                            (uint64_t)DS4_N_EMBD * sizeof(float))) {
        FILE *f = fopen(full, "wb");
        if (f) {
            fwrite(buf, sizeof(float), (size_t)DS4_N_EMBD, f);
            fclose(f);
        }
    }
    free(buf);
}

/* Correctness isolation: dump the post-prefill logits vector once. */
static void glm_debug_dump_prefill_logits(const float *logits) {
    static int dumped;
    const char *path = getenv("DS4_GLM_LOGIT_DUMP");
    if (!path || !path[0] || dumped || !logits) return;
    FILE *f = fopen(path, "wb");
    if (!f) return;
    fwrite(logits, sizeof(float), (size_t)DS4_N_VOCAB, f);
    fclose(f);
    dumped = 1;
    fprintf(stderr, "ds4: prefill logits dumped to %s\n", path);
}

static bool glm_graph_indexed_prefill_trace_enabled(void) {
    return false;
}

static bool glm_graph_indexed_prefill_trace_all(void) {
    return false;
}

static uint32_t glm_graph_indexed_prefill_trace_slow_ms(void) {
    return 100u;
}

static uint32_t glm_graph_indexed_prefill_drain_interval(void) {
    return 16u;
}

static bool glm_graph_full_prefill_trace_enabled(void) {
    return false;
}

static bool glm_graph_full_prefill_trace_all(void) {
    return false;
}

static uint32_t glm_graph_full_prefill_trace_slow_ms(void) {
    return 100u;
}

static uint32_t glm_graph_full_prefill_drain_interval(void) {
    return 16u;
}

static void glm_graph_full_prefill_tracef(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "ds4: GLM full prefill trace ");
    vfprintf(stderr, fmt, ap);
    fputc('\n', stderr);
    fflush(stderr);
    va_end(ap);
}

static void glm_graph_indexed_prefill_tracef(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "ds4: GLM indexed prefill trace ");
    vfprintf(stderr, fmt, ap);
    fputc('\n', stderr);
    fflush(stderr);
    va_end(ap);
}

static uint32_t glm_graph_compact_cache_initial_cap(
        uint32_t ctx_size,
        uint32_t full_attention_cap) {
    if (ctx_size == 0) return 0;
    if (ctx_size <= full_attention_cap) return ctx_size;

    uint32_t cap = ctx_size;
    if (cap == 0) cap = full_attention_cap ? full_attention_cap : 1u;
    if (cap > ctx_size) cap = ctx_size;
    return cap;
}

static uint64_t glm_graph_compact_cache_bytes_for_cap(
        uint32_t normal_layers,
        uint32_t indexer_layers,
        uint32_t compact_cap) {
    if (compact_cap == 0) return 0;
    const uint64_t elem = glm_graph_compact_cache_elem_bytes();
    uint64_t total =
        (uint64_t)normal_layers *
        compact_cap *
        ((uint64_t)DS4_N_KV_LORA + DS4_N_ROT) *
        elem;
    total +=
        (uint64_t)indexer_layers *
        compact_cap *
        DS4_N_INDEXER_HEAD_DIM *
        elem;
    return total;
}

static uint64_t glm_graph_indexed_scratch_bytes_for_cap(
        uint32_t full_attention_cap,
        uint32_t compact_cap) {
    if (compact_cap == 0) return 0;
    const uint64_t indexed_rows =
        glm_graph_indexed_prefill_chunk_tokens(full_attention_cap, compact_cap);
    const uint64_t indexed_score_rows =
        glm_graph_indexed_prefill_score_tokens((uint32_t)indexed_rows,
                                               compact_cap);
    const uint64_t indexer_q_elems =
        (uint64_t)DS4_N_INDEXER_HEAD * DS4_N_INDEXER_HEAD_DIM;
    const uint64_t qk_low_elems =
        (uint64_t)DS4_N_HEAD * DS4_N_KV_LORA;
    uint64_t bytes = (uint64_t)compact_cap * sizeof(float);
    const uint64_t indexer_top_k = glm_graph_indexer_top_k_limit();
    bytes += indexed_score_rows * (uint64_t)compact_cap * sizeof(float);
    bytes += indexed_rows * indexer_q_elems * sizeof(float);
    bytes += indexed_rows * DS4_N_INDEXER_HEAD * sizeof(float);
    bytes += indexed_rows * indexer_top_k * sizeof(uint32_t); /* batch_indexer_selected */
    bytes += indexed_rows * qk_low_elems * sizeof(float);     /* batch_qk_low */
    bytes += indexed_rows * qk_low_elems * sizeof(float);     /* batch_attn_lora */
    return bytes;
}

static uint64_t glm_graph_workspace_add_bytes(
        uint64_t total,
        uint64_t count,
        uint64_t elem_bytes) {
    return ds4_add_sat_u64(total, ds4_mul_sat_u64(count, elem_bytes));
}

static uint32_t glm_graph_indexed_decode_split_blocks(void);

static uint64_t glm_graph_workspace_bytes_for_cap(
        uint32_t full_attention_cap,
        uint32_t compact_cap,
        bool     ssd_streaming) {
    const bool expanded_kv =
        glm_graph_expanded_kv_cache_enabled(ssd_streaming);
    const uint64_t indexed_rows =
        compact_cap != 0 ?
            glm_graph_indexed_prefill_chunk_tokens(full_attention_cap,
                                                   compact_cap) :
            0;
    const uint64_t batch_rows =
        expanded_kv || indexed_rows == 0 ? full_attention_cap : indexed_rows;
    const uint64_t indexer_top_k = glm_graph_indexer_top_k_limit();
    const uint64_t q_dim = (uint64_t)DS4_N_HEAD * DS4_N_KEY_MLA;
    const uint64_t q_nope =
        DS4_N_KEY_MLA > DS4_N_ROT ? (uint64_t)DS4_N_KEY_MLA - DS4_N_ROT : 0;
    const uint64_t heads_dim = (uint64_t)DS4_N_HEAD * DS4_N_VALUE_MLA;
    const uint64_t kv_raw_dim = (uint64_t)DS4_N_KV_LORA + DS4_N_ROT;
    uint64_t dense_hidden_max =
        DS4_N_FF_DENSE > DS4_N_FF_EXP ? DS4_N_FF_DENSE : DS4_N_FF_EXP;
    if (dense_hidden_max == 0) dense_hidden_max = DS4_N_FF_EXP;
    const uint64_t sparse_mid_elems =
        (uint64_t)DS4_N_EXPERT_USED * DS4_N_FF_EXP;
    const uint64_t ffn_mid_elems =
        dense_hidden_max > sparse_mid_elems ?
            dense_hidden_max :
            sparse_mid_elems;
    const uint64_t qk_low_elems =
        (uint64_t)DS4_N_HEAD * DS4_N_KV_LORA;
    const uint64_t split_attn_blocks =
        glm_graph_indexed_decode_split_blocks();

    uint64_t bytes =
        glm_graph_indexed_scratch_bytes_for_cap(full_attention_cap,
                                                compact_cap);

    bytes = glm_graph_workspace_add_bytes(bytes, 3u, DS4_N_EMBD * sizeof(float));
    bytes = glm_graph_workspace_add_bytes(bytes, 2u, DS4_N_LORA_Q * sizeof(float));
    bytes = glm_graph_workspace_add_bytes(bytes, q_dim, sizeof(float));
    bytes = glm_graph_workspace_add_bytes(bytes, DS4_N_INDEXER_HEAD_DIM, sizeof(float));
    bytes = glm_graph_workspace_add_bytes(bytes,
                                          (uint64_t)DS4_N_INDEXER_HEAD *
                                          DS4_N_INDEXER_HEAD_DIM,
                                          sizeof(float));
    bytes = glm_graph_workspace_add_bytes(bytes, DS4_N_INDEXER_HEAD, sizeof(float));
    bytes = glm_graph_workspace_add_bytes(bytes, indexer_top_k, sizeof(uint32_t));
    bytes = glm_graph_workspace_add_bytes(bytes, qk_low_elems, sizeof(float));
    bytes = glm_graph_workspace_add_bytes(bytes,
                                          split_attn_blocks * qk_low_elems,
                                          sizeof(float));
    bytes = glm_graph_workspace_add_bytes(bytes,
                                          split_attn_blocks *
                                          DS4_N_HEAD * 2u,
                                          sizeof(float));
    bytes = glm_graph_workspace_add_bytes(bytes, kv_raw_dim, sizeof(float));
    bytes = glm_graph_workspace_add_bytes(bytes, DS4_N_KV_LORA, sizeof(float));
    bytes = glm_graph_workspace_add_bytes(bytes,
                                          DS4_N_HEAD * q_nope,
                                          sizeof(float));
    bytes = glm_graph_workspace_add_bytes(bytes, 2u * heads_dim, sizeof(float));
    bytes = glm_graph_workspace_add_bytes(bytes, 6u, DS4_N_EMBD * sizeof(float));
    bytes = glm_graph_workspace_add_bytes(bytes, 2u * dense_hidden_max, sizeof(float));
    bytes = glm_graph_workspace_add_bytes(bytes, ffn_mid_elems, sizeof(float));
    bytes = glm_graph_workspace_add_bytes(bytes, DS4_N_EXPERT * 2u, sizeof(float));
    bytes = glm_graph_workspace_add_bytes(bytes, DS4_N_EXPERT_USED, sizeof(int32_t));
    bytes = glm_graph_workspace_add_bytes(bytes, DS4_N_EXPERT_USED, sizeof(float));
    bytes = glm_graph_workspace_add_bytes(bytes, DS4_N_VOCAB, sizeof(float));
    bytes = glm_graph_workspace_add_bytes(bytes,
                                          (uint64_t)DS4_N_LAYER *
                                          DS4_STREAMING_PREFILL_CACHE_SEED_MAX_TOKENS *
                                          DS4_N_EXPERT_USED,
                                          sizeof(int32_t));

    bytes = glm_graph_workspace_add_bytes(bytes, batch_rows, sizeof(int32_t));
    bytes = glm_graph_workspace_add_bytes(bytes,
                                          batch_rows * DS4_N_EXPERT * 2u,
                                          sizeof(float));
    bytes = glm_graph_workspace_add_bytes(bytes,
                                          batch_rows * DS4_N_EXPERT_USED,
                                          sizeof(int32_t));
    bytes = glm_graph_workspace_add_bytes(bytes,
                                          batch_rows * DS4_N_EXPERT_USED,
                                          sizeof(float));
    bytes = glm_graph_workspace_add_bytes(bytes,
                                          batch_rows * DS4_N_EMBD * 7u,
                                          sizeof(float));
    bytes = glm_graph_workspace_add_bytes(bytes,
                                          batch_rows * DS4_N_LORA_Q * 2u,
                                          sizeof(float));
    bytes = glm_graph_workspace_add_bytes(bytes, batch_rows * q_dim, sizeof(float));
    bytes = glm_graph_workspace_add_bytes(bytes,
                                          batch_rows * DS4_N_INDEXER_HEAD_DIM,
                                          sizeof(float));
    bytes = glm_graph_workspace_add_bytes(bytes, batch_rows * kv_raw_dim, sizeof(float));
    bytes = glm_graph_workspace_add_bytes(bytes,
                                          batch_rows * DS4_N_KV_LORA,
                                          sizeof(float));
    bytes = glm_graph_workspace_add_bytes(bytes,
                                          batch_rows * DS4_N_HEAD * q_nope,
                                          sizeof(float));
    bytes = glm_graph_workspace_add_bytes(bytes, batch_rows * heads_dim * 2u, sizeof(float));
    bytes = glm_graph_workspace_add_bytes(bytes,
                                          batch_rows * dense_hidden_max * 2u,
                                          sizeof(float));
    bytes = glm_graph_workspace_add_bytes(bytes,
                                          batch_rows * DS4_N_FF_EXP,
                                          sizeof(float));
    bytes = glm_graph_workspace_add_bytes(bytes,
                                          batch_rows * ffn_mid_elems,
                                          sizeof(float));
    if (indexed_rows != 0) {
        bytes = glm_graph_workspace_add_bytes(bytes,
                                              indexed_rows * qk_low_elems,
                                              sizeof(float));
    }
    return bytes;
}

static ds4_context_memory glm_graph_context_memory_estimate_for_compact_cap_slice(
        uint32_t ctx,
        uint32_t work_ctx,
        uint32_t compact_cap,
        bool     ssd_streaming,
        uint32_t layer_start,
        uint32_t layer_end) {
    ds4_context_memory m = {0};
    const uint32_t normal_layers = glm_graph_normal_layer_count();
    if (normal_layers == 0 || layer_start >= normal_layers ||
        layer_end < layer_start) {
        return m;
    }
    if (layer_end >= normal_layers) layer_end = normal_layers - 1u;
    const uint32_t layer_count = layer_end - layer_start + 1u;
    if (compact_cap > ctx) compact_cap = ctx;

    const bool expanded_kv = glm_graph_expanded_kv_cache_enabled(ssd_streaming);
    const uint32_t indexed_rows =
        compact_cap != 0 ?
            glm_graph_indexed_prefill_chunk_tokens(work_ctx, compact_cap) :
            0;
    const uint32_t batch_rows =
        expanded_kv || indexed_rows == 0 ? work_ctx : indexed_rows;

    m.prefill_cap = batch_rows;
    m.raw_cap = expanded_kv ? work_ctx : 0;
    if (expanded_kv) {
        m.raw_bytes = (uint64_t)layer_count *
                      work_ctx *
                      ((uint64_t)DS4_N_HEAD * (DS4_N_KEY_MLA + DS4_N_VALUE_MLA)) *
                      glm_graph_full_kv_cache_elem_bytes();
    }
    m.scratch_bytes =
        glm_graph_workspace_bytes_for_cap(work_ctx,
                                          compact_cap,
                                          ssd_streaming);
    if (compact_cap != 0) {
        m.comp_cap = compact_cap;
        m.compressed_bytes =
            glm_graph_compact_cache_bytes_for_cap(
                    layer_count,
                    glm_graph_full_indexer_layer_count_range(layer_start,
                                                             layer_end),
                    compact_cap);
    }
    m.total_bytes = m.raw_bytes + m.compressed_bytes + m.scratch_bytes;
    return m;
}

static ds4_context_memory glm_graph_context_memory_estimate_for_compact_cap(
        uint32_t ctx,
        uint32_t work_ctx,
        uint32_t compact_cap,
        bool     ssd_streaming) {
    const uint32_t normal_layers = glm_graph_normal_layer_count();
    if (normal_layers == 0) {
        const ds4_context_memory empty = {0};
        return empty;
    }
    return glm_graph_context_memory_estimate_for_compact_cap_slice(
            ctx,
            work_ctx,
            compact_cap,
            ssd_streaming,
            0,
            normal_layers - 1u);
}

ds4_context_memory ds4_context_memory_estimate_with_prefill_mode(
        ds4_backend backend,
        int         ctx_size,
        uint32_t    prefill_chunk,
        bool        ssd_streaming) {
    ds4_context_memory m = {0};
    uint32_t ctx = ctx_size > 0 ? (uint32_t)ctx_size : 1u;

    if (ds4_backend_uses_graph(backend)) {
        if (DS4_MODEL_FAMILY == DS4_MODEL_FAMILY_GLM_DSA) {
            const uint32_t work_ctx =
                glm_graph_full_attention_cap(ctx, ssd_streaming);
            const uint32_t compact_cap =
                glm_graph_compact_cache_initial_cap(ctx, work_ctx);
            m = glm_graph_context_memory_estimate_for_compact_cap(ctx,
                                                                  work_ctx,
                                                                  compact_cap,
                                                                  ssd_streaming);
            return m;
        }
        m.prefill_cap = metal_graph_prefill_cap_for_prompt((int)ctx,
                                                           prefill_chunk);
        m.raw_cap = metal_graph_raw_cap_for_context((int)ctx, m.prefill_cap);

        uint32_t min_ratio = UINT32_MAX;
        for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
            const uint32_t ratio = ds4_layer_compress_ratio(il);
            if (ratio != 0 && ratio < min_ratio) min_ratio = ratio;
        }
        if (min_ratio == UINT32_MAX) min_ratio = ctx;
        m.comp_cap = ctx / min_ratio + 2u;
        if (m.comp_cap < 2u) m.comp_cap = 2u;

        m.raw_bytes = (uint64_t)DS4_N_LAYER *
                      m.raw_cap *
                      DS4_N_HEAD_DIM *
                      sizeof(float);
        for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
            const uint32_t ratio = ds4_layer_compress_ratio(il);
            if (ratio == 0) continue;
            const uint32_t layer_comp_cap = ctx / ratio + 2u;
            m.compressed_bytes += (uint64_t)layer_comp_cap *
                                  DS4_N_HEAD_DIM *
                                  (DS4_GPU_ATTN_COMP_CACHE_F16 ? sizeof(uint16_t) : sizeof(float));
            if (ratio == 4) {
                m.compressed_bytes += (uint64_t)layer_comp_cap *
                                      DS4_N_INDEXER_HEAD_DIM *
                                      sizeof(float);
            }
        }
        uint64_t attn_stage_cap = (uint64_t)(m.prefill_cap / min_ratio + 2u);
        if (attn_stage_cap < 2u) attn_stage_cap = 2u;
        m.scratch_bytes = 2ull *
                          m.comp_cap *
                          m.prefill_cap *
                          sizeof(float) +
                          attn_stage_cap * DS4_N_HEAD_DIM * sizeof(float);
    } else {
        m.raw_cap = ds4_default_raw_cap(ctx);
        m.raw_bytes = (uint64_t)DS4_N_LAYER *
                      m.raw_cap *
                      DS4_N_HEAD_DIM *
                      sizeof(float);
        for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
            const uint32_t ratio = ds4_layer_compress_ratio(il);
            if (ratio == 0) continue;
            const uint32_t comp_cap = ctx / ratio + 2u;
            if (ratio == 4) m.comp_cap = comp_cap;
            m.compressed_bytes += (uint64_t)comp_cap *
                                  DS4_N_HEAD_DIM *
                                  sizeof(float);
            if (ratio == 4) {
                m.compressed_bytes += (uint64_t)comp_cap *
                                      DS4_N_INDEXER_HEAD_DIM *
                                      sizeof(float);
            }
        }
        if (m.comp_cap == 0) m.comp_cap = ctx / 4u + 2u;
        m.scratch_bytes = ((uint64_t)(m.raw_cap + m.comp_cap) * sizeof(float)) +
                          ((uint64_t)m.comp_cap * sizeof(float)) +
                          ((uint64_t)m.comp_cap * sizeof(bool));
    }

    m.total_bytes = m.raw_bytes + m.compressed_bytes + m.scratch_bytes;
    return m;
}

ds4_context_memory ds4_context_memory_estimate_with_prefill(
        ds4_backend backend,
        int         ctx_size,
        uint32_t    prefill_chunk) {
    return ds4_context_memory_estimate_with_prefill_mode(backend,
                                                         ctx_size,
                                                         prefill_chunk,
                                                         false);
}

ds4_context_memory ds4_context_memory_estimate(ds4_backend backend,
                                               int         ctx_size) {
    return ds4_context_memory_estimate_with_prefill(backend, ctx_size, 0);
}

static int metal_graph_prompt_logits_test(
        const ds4_model   *model,
        const ds4_weights *weights,
        const token_vec   *prompt,
        int                ctx_size) {
    int n_test = prompt->len;
    const char *n_test_env = getenv("DS4_METAL_GRAPH_PROMPT_TOKENS");
    if (n_test_env && n_test_env[0]) {
        char *endp = NULL;
        const long v = strtol(n_test_env, &endp, 10);
        if (endp != n_test_env && v > 0 && v <= prompt->len) n_test = (int)v;
    }

    if (n_test <= 0 || n_test > ctx_size) {
        fprintf(stderr, "ds4: Metal graph prompt test needs 1..%d prompt tokens\n", ctx_size);
        return 1;
    }

    const uint32_t raw_cap = metal_graph_raw_cap_for_context(ctx_size, (uint32_t)n_test);

    ds4_gpu_graph g;
    /* diagnostic single-tier callsite; placement=NULL. */
    bool ok = metal_graph_alloc_raw_cap(&g, weights, &weights->layer[0],
                                        raw_cap, (uint32_t)ctx_size,
                                        (uint32_t)n_test, false, NULL, false, NULL);
    if (!ok) {
        metal_graph_free(&g);
        fprintf(stderr, "ds4: failed to initialize Metal graph prompt test runtime\n");
        return 1;
    }
    const bool memory_report = getenv("DS4_METAL_MEMORY_REPORT") != NULL;
    if (memory_report) ds4_gpu_print_memory_report("after graph alloc");

    ds4_kv_cache cpu_cache;
    kv_cache_init(&cpu_cache, (uint32_t)ctx_size, raw_cap);
    float *cpu_logits = xmalloc((size_t)DS4_N_VOCAB * sizeof(float));
    float *gpu_logits = xmalloc((size_t)DS4_N_VOCAB * sizeof(float));
    float *oracle_logits = NULL;

    const char *oracle_path = getenv("DS4_ORACLE_LOGITS");
    if (oracle_path && oracle_path[0]) {
        oracle_logits = xmalloc((size_t)DS4_N_VOCAB * sizeof(float));
        if (!read_f32_binary_file(oracle_path, oracle_logits, DS4_N_VOCAB)) {
            free(oracle_logits);
            oracle_logits = NULL;
        }
    }

    for (int t = 0; t < n_test; t++) {
        const bool last = t == n_test - 1;
        forward_token_raw_swa_cpu(last ? cpu_logits : NULL,
                                  model,
                                  weights,
                                  &cpu_cache,
                                  prompt->v[t],
                                  (uint32_t)t);
    }
    ok = metal_graph_prefill_raw_swa(&g, model, weights, prompt, n_test,
                                     gpu_logits, true, NULL, NULL,
                                     NULL, NULL, NULL);
    if (memory_report) ds4_gpu_print_memory_report("after prompt graph");

    if (ok) {
        const char *dump_gpu = getenv("DS4_METAL_GRAPH_DUMP_LOGITS");
        if (dump_gpu && dump_gpu[0]) {
            if (write_f32_binary_file(dump_gpu, gpu_logits, DS4_N_VOCAB)) {
                fprintf(stderr, "ds4: wrote Metal graph logits to %s\n", dump_gpu);
            }
        }
        const char *dump_cpu = getenv("DS4_CPU_DUMP_LOGITS");
        if (dump_cpu && dump_cpu[0]) {
            if (write_f32_binary_file(dump_cpu, cpu_logits, DS4_N_VOCAB)) {
                fprintf(stderr, "ds4: wrote CPU logits to %s\n", dump_cpu);
            }
        }
        if (getenv("DS4_METAL_GRAPH_TRACE_CACHE") != NULL ||
            getenv("DS4_METAL_GRAPH_TRACE_COMP") != NULL) {
            for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
                const uint32_t n_raw = cpu_cache.layer[il].n_raw;
                if (n_raw != 0) {
                    const uint64_t raw_phys_n = (uint64_t)raw_cap * DS4_N_HEAD_DIM;
                    const uint64_t raw_logical_n = (uint64_t)n_raw * DS4_N_HEAD_DIM;
                    const uint32_t raw_start = n_raw < raw_cap ? 0u : ((uint32_t)n_test % raw_cap);
                    float *gpu_raw_phys = xmalloc((size_t)raw_phys_n * sizeof(float));
                    float *gpu_raw_logical = xmalloc((size_t)raw_logical_n * sizeof(float));
                    if (ds4_gpu_tensor_read(g.layer_raw_cache[il], 0, gpu_raw_phys, raw_phys_n * sizeof(float)) != 0) {
                        for (uint32_t r = 0; r < n_raw; r++) {
                            const uint32_t phys = (raw_start + r) % raw_cap;
                            memcpy(gpu_raw_logical + (uint64_t)r * DS4_N_HEAD_DIM,
                                   gpu_raw_phys + (uint64_t)phys * DS4_N_HEAD_DIM,
                                   (size_t)DS4_N_HEAD_DIM * sizeof(float));
                        }
                        fprintf(stderr,
                                "ds4: cache trace layer %u raw_n=%u raw_start=%u raw_max=%g raw_rms=%g\n",
                                il, n_raw, raw_start,
                                max_abs_diff(cpu_cache.layer[il].raw_kv, gpu_raw_logical, raw_logical_n),
                                rms_abs_diff(cpu_cache.layer[il].raw_kv, gpu_raw_logical, raw_logical_n));
                    }
                    free(gpu_raw_logical);
                    free(gpu_raw_phys);
                }

                const uint32_t n_comp = cpu_cache.layer[il].n_comp;
                if (n_comp == 0) continue;
                const uint64_t n = (uint64_t)n_comp * DS4_N_HEAD_DIM;
                float *gpu_comp = xmalloc((size_t)n * sizeof(float));
                bool comp_read = false;
                if (DS4_GPU_ATTN_COMP_CACHE_F16) {
                    uint16_t *gpu_comp_h = xmalloc((size_t)n * sizeof(uint16_t));
                    if (ds4_gpu_tensor_read(g.layer_attn_comp_cache[il], 0,
                                            gpu_comp_h, n * sizeof(uint16_t)) != 0) {
                        for (uint64_t i = 0; i < n; i++) gpu_comp[i] = f16_to_f32(gpu_comp_h[i]);
                        comp_read = true;
                    }
                    free(gpu_comp_h);
                } else {
                    comp_read = ds4_gpu_tensor_read(g.layer_attn_comp_cache[il], 0,
                                                    gpu_comp, n * sizeof(float)) != 0;
                }
                if (comp_read) {
                    fprintf(stderr,
                            "ds4: comp trace layer %u n=%u attn_max=%g attn_rms=%g\n",
                            il, n_comp,
                            max_abs_diff(cpu_cache.layer[il].attn_comp_kv, gpu_comp, n),
                            rms_abs_diff(cpu_cache.layer[il].attn_comp_kv, gpu_comp, n));
                }
                free(gpu_comp);

                const uint32_t n_index = cpu_cache.layer[il].n_index_comp;
                if (n_index != 0 && g.layer_index_comp_cache[il]) {
                    const uint64_t ni = (uint64_t)n_index * DS4_N_INDEXER_HEAD_DIM;
                    float *gpu_index = xmalloc((size_t)ni * sizeof(float));
                    if (ds4_gpu_tensor_read(g.layer_index_comp_cache[il], 0, gpu_index, ni * sizeof(float)) != 0) {
                        fprintf(stderr,
                                "ds4: comp trace layer %u n=%u index_max=%g index_rms=%g\n",
                                il, n_index,
                                max_abs_diff(cpu_cache.layer[il].index_comp_kv, gpu_index, ni),
                                rms_abs_diff(cpu_cache.layer[il].index_comp_kv, gpu_index, ni));
                    }
                    free(gpu_index);
                }
            }
        }
        const uint64_t cpu_top = argmax_f32(cpu_logits, DS4_N_VOCAB);
        const uint64_t gpu_top = argmax_f32(gpu_logits, DS4_N_VOCAB);
        fprintf(stderr,
                "ds4: Metal prompt graph logits: tokens=%d logits_max=%g logits_rms=%g cpu_top=%llu gpu_top=%llu cpu_top_logit=%g gpu_top_logit=%g\n",
                n_test,
                max_abs_diff(cpu_logits, gpu_logits, DS4_N_VOCAB),
                rms_abs_diff(cpu_logits, gpu_logits, DS4_N_VOCAB),
                (unsigned long long)cpu_top,
                (unsigned long long)gpu_top,
                cpu_logits[cpu_top],
                gpu_logits[gpu_top]);
        if (oracle_logits) {
            const uint64_t oracle_top = argmax_f32(oracle_logits, DS4_N_VOCAB);
            fprintf(stderr,
                    "ds4: oracle logits: tokens=%d oracle_top=%llu oracle_top_logit=%g cpu_max=%g cpu_rms=%g metal_max=%g metal_rms=%g\n",
                    n_test,
                    (unsigned long long)oracle_top,
                    oracle_logits[oracle_top],
                    max_abs_diff(cpu_logits, oracle_logits, DS4_N_VOCAB),
                    rms_abs_diff(cpu_logits, oracle_logits, DS4_N_VOCAB),
                    max_abs_diff(gpu_logits, oracle_logits, DS4_N_VOCAB),
                    rms_abs_diff(gpu_logits, oracle_logits, DS4_N_VOCAB));
        }
    } else {
        fprintf(stderr, "ds4: Metal prompt graph logits test failed\n");
        if (ds4_gpu_synchronize() == 0) {
            fprintf(stderr, "ds4: Metal synchronize after prompt graph failure also failed\n");
        }
    }

    free(gpu_logits);
    free(cpu_logits);
    free(oracle_logits);
    kv_cache_free(&cpu_cache);
    metal_graph_free(&g);
    return ok ? 0 : 1;
}
