enum {
    DS4_METAL_TENSOR_Q4_0    = 2,
    DS4_METAL_TENSOR_Q8_0    = 8,
    DS4_METAL_TENSOR_Q2_K    = 10,
    DS4_METAL_TENSOR_Q4_K    = 12,
    DS4_METAL_TENSOR_Q5_K    = 13,
    DS4_METAL_TENSOR_Q6_K    = 14,
    DS4_METAL_TENSOR_Q8_K    = 15,
    DS4_METAL_TENSOR_IQ2_XXS = 16,
};

@class DS4MetalQ4ExpertTable;

static id<MTLDevice> g_device;
static id<MTLCommandQueue> g_queue;
static id<MTLLibrary> g_library;
static id<MTLCommandBuffer> g_batch_cb;
static id<MTLComputeCommandEncoder> g_batch_enc;
static BOOL g_batch_has_work;
static NSMutableArray<id<MTLCommandBuffer>> *g_pending_cbs;
static id<MTLSharedEvent> g_selected_readback_event;
static uint64_t g_selected_readback_event_value;
static id<MTLComputePipelineState> g_set_rows_f32_i32_pipeline;
static id<MTLComputePipelineState> g_get_rows_f32_pipeline;
static id<MTLComputePipelineState> g_get_rows_f16_pipeline;
static id<MTLComputePipelineState> g_get_rows_i32_pipeline;
static id<MTLComputePipelineState> g_get_rows_q8_0_pipeline;
static id<MTLComputePipelineState> g_get_rows_q4_0_pipeline;
static id<MTLComputePipelineState> g_get_rows_q4_K_pipeline;
static id<MTLComputePipelineState> g_repeat_f32_pipeline;
static id<MTLComputePipelineState> g_concat_pipeline;
static id<MTLComputePipelineState> g_cpy_f32_f32_pipeline;
static id<MTLComputePipelineState> g_cpy_f32_f16_pipeline;
static id<MTLComputePipelineState> g_cpy_contig_f32_f16_pipeline;
static id<MTLComputePipelineState> g_cpy_f16_f32_pipeline;
static id<MTLComputePipelineState> g_cpy_f16_f16_pipeline;
static id<MTLComputePipelineState> g_cpy_contig_f16_f32_pipeline;
static id<MTLComputePipelineState> g_cpy_contig_f16_f16_pipeline;
static id<MTLComputePipelineState> g_flash_kv_stage_f16_pipeline;
static id<MTLComputePipelineState> g_swiglu_pipeline;
static id<MTLComputePipelineState> g_swiglu_flat_pipeline;
static id<MTLComputePipelineState> g_add_pipeline;
static id<MTLComputePipelineState> g_add2_pipeline;
static id<MTLComputePipelineState> g_add3_pipeline;
static id<MTLComputePipelineState> g_moe_sum6_pipeline;
static id<MTLComputePipelineState> g_moe_sum8_pipeline;
static id<MTLComputePipelineState> g_mul_pipeline;
static id<MTLComputePipelineState> g_rms_norm_pipeline;
static id<MTLComputePipelineState> g_rms_norm_plain_pipeline;
static id<MTLComputePipelineState> g_add_rms_norm_pipeline;
static id<MTLComputePipelineState> g_rms_norm_scale_pipeline;
static id<MTLComputePipelineState> g_dsv4_qkv_rms_norm_pipeline;
static id<MTLComputePipelineState> g_hc_split_sinkhorn_pipeline;
static id<MTLComputePipelineState> g_hc_split_weighted_sum_pipeline;
static id<MTLComputePipelineState> g_hc_split_weighted_sum_norm_pipeline;
static id<MTLComputePipelineState> g_hc_weighted_sum_pipeline;
static id<MTLComputePipelineState> g_hc_weighted_sum_norm_pipeline;
static id<MTLComputePipelineState> g_output_hc_weights4_pipeline;
static id<MTLComputePipelineState> g_hc_expand_pipeline;
static id<MTLComputePipelineState> g_unary_sigmoid_pipeline;
static id<MTLComputePipelineState> g_unary_silu_pipeline;
static id<MTLComputePipelineState> g_unary_softplus_pipeline;
static id<MTLComputePipelineState> g_unary_sqrt_pipeline;
static id<MTLComputePipelineState> g_unary_clamp_pipeline;
static id<MTLComputePipelineState> g_unary_scale_pipeline;
static id<MTLComputePipelineState> g_unary_fill_pipeline;
static id<MTLComputePipelineState> g_unary_fill_f16_pipeline;
static id<MTLComputePipelineState> g_bin_mul_scalar_pipeline;
static id<MTLComputePipelineState> g_bin_div_row_pipeline;
static id<MTLComputePipelineState> g_moe_mul_mv_id_iq2_xxs_pipeline;
static id<MTLComputePipelineState> g_moe_mul_mv_id_iq2_xxs_pair_pipeline;
static id<MTLComputePipelineState> g_moe_mul_mv_id_iq2_xxs_pair_swiglu_pipeline;
static id<MTLComputePipelineState> g_moe_mul_mv_id_q2_k_pipeline;
static id<MTLComputePipelineState> g_moe_mul_mv_id_q2_k_sum6_pipeline;
static id<MTLComputePipelineState> g_moe_mul_mv_id_iq2_xxs_sum6_pipeline;
static id<MTLComputePipelineState> g_moe_mul_mv_id_q4_k_pipeline;
static id<MTLComputePipelineState> g_moe_mul_mv_id_q4_k_pair_pipeline;
static id<MTLComputePipelineState> g_moe_mul_mv_id_q4_k_pair_swiglu_pipeline;
static id<MTLComputePipelineState> g_moe_mul_mv_id_q4_k_sum6_pipeline;
static id<MTLComputePipelineState> g_moe_mul_mv_group_q4_k_pair_swiglu_pipeline;
static id<MTLComputePipelineState> g_moe_mul_mv_group_q4_k_sum6_pipeline;
static id<MTLComputePipelineState> g_moe_mul_mv_group6_q4_k_pair_swiglu_pipeline;
static id<MTLComputePipelineState> g_moe_mul_mv_group6_q4_k_sum6_pipeline;
static id<MTLComputePipelineState> g_moe_mul_mv_group8_q4_k_pair_swiglu_pipeline;
static id<MTLComputePipelineState> g_moe_mul_mv_group8_q4_k_sum6_pipeline;
static id<MTLComputePipelineState> g_moe_mul_mv_group24_q4_k_id_pipeline;
static id<MTLComputePipelineState> g_moe_mul_mv_group24_q4_k_sum6_pipeline;
static id<MTLComputePipelineState> g_moe_mul_mv_slots6_iq2_xxs_pair_swiglu_pipeline;
static id<MTLComputePipelineState> g_moe_mul_mv_slots6_q2_k_sum6_pipeline;
static id<MTLComputePipelineState> g_moe_mul_mv_slots6_q4_k_pair_swiglu_pipeline;
static id<MTLComputePipelineState> g_moe_mul_mv_slots6_q4_k_sum6_pipeline;
static id<MTLComputePipelineState> g_moe_mul_mv_addr_iq2_xxs_pair_swiglu_pipeline;
static id<MTLComputePipelineState> g_moe_mul_mv_addr_iq2_xxs_pipeline;
static id<MTLComputePipelineState> g_moe_mul_mv_addr_q2_k_sum6_pipeline;
static id<MTLComputePipelineState> g_moe_mul_mv_addr_iq2_xxs_pair_swiglu_masked_pipeline;
static id<MTLComputePipelineState> g_moe_mul_mv_addr_q2_k_sum6_masked_pipeline;
static id<MTLComputePipelineState> g_moe_stream_expert_cache_validate_pipeline;
static id<MTLComputePipelineState> g_moe_q4_gather_slots6_pipeline;
static id<MTLComputePipelineState> g_moe_mul_mv_table_q4_k_pair_swiglu_pipeline;
static id<MTLComputePipelineState> g_moe_mul_mv_table_q4_k_sum6_pipeline;
static id<MTLComputePipelineState> g_moe_mul_mv_addr_q4_k_pair_swiglu_pipeline;
static id<MTLComputePipelineState> g_moe_mul_mv_addr_q4_k_sum6_pipeline;
static id<MTLArgumentEncoder> g_moe_table_q4_pair_gate_encoder;
static id<MTLArgumentEncoder> g_moe_table_q4_pair_up_encoder;
static id<MTLArgumentEncoder> g_moe_table_q4_sum_down_encoder;
static id<MTLComputePipelineState> g_rope_tail_batch_pipeline;
static id<MTLComputePipelineState> g_rope_tail_inplace_pair_pipeline;
static id<MTLComputePipelineState> g_rope_tail_inplace_pair_shared4_pipeline;
static id<MTLComputePipelineState> g_rope_tail_inplace_pair_affine_pipeline;
static id<MTLComputePipelineState> g_dsv4_fp8_kv_quantize_pipeline;
static id<MTLComputePipelineState> g_dsv4_indexer_qat_pipeline;
static id<MTLComputePipelineState> g_dsv4_kv_fp8_store_pipeline;
static id<MTLComputePipelineState> g_dsv4_ratio4_shift_pipeline;
static id<MTLComputePipelineState> g_dsv4_compressor_pack_ratio4_pipeline;
static id<MTLComputePipelineState> g_dsv4_softmax_pool_ratio4_direct_pipeline;
static id<MTLComputePipelineState> g_dsv4_softmax_pool_pipeline;
static id<MTLComputePipelineState> g_soft_max_f32_pipeline;
static id<MTLComputePipelineState> g_soft_max_f32_4_pipeline;
static id<MTLComputePipelineState> g_argsort_f32_i32_desc_pipeline;
static id<MTLComputePipelineState> g_argsort_merge_f32_i32_desc_pipeline;
static id<MTLComputePipelineState> g_sum_rows_f32_f32_pipeline;
static id<MTLComputePipelineState> g_dsv4_topk_mask_pipeline;
static id<MTLComputePipelineState> g_dsv4_topk_mask_scatter_pipeline;
static id<MTLComputePipelineState> g_dsv4_indexer_weighted_sum_pipeline;
static id<MTLComputePipelineState> g_dsv4_indexer_score_one_direct_pipeline;
static id<MTLComputePipelineState> g_dsv4_compressor_store_one_pipeline;
static id<MTLComputePipelineState> g_dsv4_sort_i32_rows_asc_pipeline;
static id<MTLComputePipelineState> g_dsv4_indexed_attention_heads8_pipeline;
static id<MTLComputePipelineState> g_dsv4_indexed_attention_heads8_rb16_pipeline;
static id<MTLComputePipelineState> g_dsv4_softplus_sqrt_pipeline;
static id<MTLComputePipelineState> g_dsv4_router_finalize_one_pipeline;
static id<MTLComputePipelineState> g_dsv4_router_finalize_one_simd_pipeline;
static id<MTLComputePipelineState> g_dsv4_router_finalize_weights_one_simd_pipeline;
static id<MTLComputePipelineState> g_dsv4_router_weights_one_pipeline;
static id<MTLComputePipelineState> g_glm_router_select_one_pipeline;
static id<MTLComputePipelineState> g_glm_kv_lora_rms_norm_pipeline;
static id<MTLComputePipelineState> g_glm_k_b_project_pipeline;
static id<MTLComputePipelineState> g_glm_store_compact_kv_pipeline;
static id<MTLComputePipelineState> g_glm_qkv_norm_store_compact_kv_pipeline;
static id<MTLComputePipelineState> g_glm_store_indexer_k_pipeline;
static id<MTLComputePipelineState> g_glm_build_kv_cache_pipeline;
static id<MTLComputePipelineState> g_glm_build_kv_cache_decode_group4_pipeline;
static id<MTLComputePipelineState> g_glm_build_kv_cache_flash_pipeline;
static id<MTLComputePipelineState> g_glm_attention_full_pipeline;
static id<MTLComputePipelineState> g_glm_fill_selected_range_pipeline;
static id<MTLComputePipelineState> g_glm_fill_selected_range_batch_pipeline;
static id<MTLComputePipelineState> g_glm_indexer_rope_tail_pipeline;
static id<MTLComputePipelineState> g_glm_indexer_score_one_pipeline;
static id<MTLComputePipelineState> g_glm_indexer_score_one_direct_pipeline;
static id<MTLComputePipelineState> g_glm_indexer_scores_batch_pipeline;
static id<MTLComputePipelineState> g_glm_indexer_scores_tiled_pipeline;
static id<MTLComputePipelineState> g_glm_indexer_scores_tiled_f32_pipeline;
static id<MTLComputePipelineState> g_glm_qk_lowrank_pipeline;
static id<MTLComputePipelineState> g_glm_qk_lowrank_glm52_pipeline;
static id<MTLComputePipelineState> g_glm_qk_lowrank_glm52_sg_pipeline;
static id<MTLComputePipelineState> g_glm_qk_lowrank_batch_pipeline;
static id<MTLComputePipelineState> g_glm_qk_lowrank_batch_glm52_t4_pipeline;
static id<MTLComputePipelineState> g_glm_value_project_q8_0_pipeline;
static id<MTLComputePipelineState> g_glm_value_project_q8_0_batch_heads_pipeline;
static id<MTLComputePipelineState> g_glm_value_project_q8_0_batch_heads_mma_pipeline;
static id<MTLComputePipelineState> g_glm_attention_indexed_decode_pipeline;
static id<MTLComputePipelineState> g_glm_attention_indexed_decode_split_group8_partial_pipeline;
static id<MTLComputePipelineState> g_glm_attention_indexed_decode_split_group8_partial_valid_fullheads_pipeline;
static id<MTLComputePipelineState> g_glm_attention_indexed_decode_split_group8_reduce_pipeline;
static id<MTLComputePipelineState> g_glm_attention_indexed_decode_split_group8_reduce16_pipeline;
static id<MTLComputePipelineState> g_glm_attention_indexed_batch_pipeline;
static id<MTLComputePipelineState> g_glm_attention_indexed_batch_group2_pipeline;
static id<MTLComputePipelineState> g_glm_attention_indexed_batch_q2_group4_pipeline;
static id<MTLComputePipelineState> g_glm_attention_indexed_batch_group8_pipeline;
static id<MTLComputePipelineState> g_glm_attention_indexed_batch_lora_group8_vec_pipeline;
static id<MTLComputePipelineState> g_glm_attention_indexed_batch_lora_group8_vec_valid_pipeline;
static id<MTLComputePipelineState> g_glm_attention_indexed_batch_lora_group8_vec_valid_fullheads_pipeline;
static id<MTLComputePipelineState> g_glm_attention_indexed_batch_lora_group8_vec_causal_pipeline;
static id<MTLComputePipelineState> g_glm_attention_indexed_batch_lora_group8_vec_causal_fullheads_pipeline;
static id<MTLComputePipelineState> g_glm_q4_k_pair_swiglu_f32_pipeline;
static id<MTLComputePipelineState> g_glm_q4_k_pair_swiglu2_f32_pipeline;
static id<MTLComputePipelineState> g_glm_q4_k_pair_swiglu4_f32_pipeline;
static id<MTLComputePipelineState> g_glm_q4_k_pair_swiglu2_mapped_f32_pipeline;
static id<MTLComputePipelineState> g_glm_q4_k_pair_swiglu2_mapped_row_f32_pipeline;
static id<MTLComputePipelineState> g_glm_q2_k_pair_swiglu_f32_pipeline;
static id<MTLComputePipelineState> g_glm_q2_k_addr_pair_swiglu2_f32_pipeline;
static id<MTLComputePipelineState> g_glm_q2_k_addr_pair_swiglu2_masked_f32_pipeline;
static id<MTLComputePipelineState> g_glm_q4_k_addr_pair_swiglu_f32_pipeline;
static id<MTLComputePipelineState> g_glm_q4_k_addr_pair_swiglu_masked_f32_pipeline;
static id<MTLComputePipelineState> g_glm_q2_k_down_f32_pipeline;
static id<MTLComputePipelineState> g_glm_q4_k_down_f32_pipeline;
static id<MTLComputePipelineState> g_glm_q2_k_addr_down_f32_pipeline;
static id<MTLComputePipelineState> g_glm_q4_k_addr_down_f32_pipeline;
static id<MTLComputePipelineState> g_glm_q5_k_pair_swiglu_f32_pipeline;
static id<MTLComputePipelineState> g_glm_q5_k_pair_swiglu_mapped_f32_pipeline;
static id<MTLComputePipelineState> g_glm_q5_k_pair_swiglu_mapped_row_f32_pipeline;
static id<MTLComputePipelineState> g_glm_q5_k_down_f32_pipeline;
static id<MTLComputePipelineState> g_glm_q6_k_down_f32_pipeline;
static id<MTLComputePipelineState> g_dsv4_router_weights_batch_pipeline;
static id<MTLComputePipelineState> g_dsv4_hc_expand4_pipeline;
static NSMutableDictionary<NSString *, id<MTLComputePipelineState>> *g_pipeline_cache;
static NSMutableDictionary<NSString *, id<MTLBuffer>> *g_model_buffer_cache;
static NSMutableDictionary<NSString *, DS4MetalQ4ExpertTable *> *g_q4_expert_table_cache;
static NSMutableDictionary<NSString *, id> *g_q4_expert_layer_residency_cache;
static NSMutableArray<id<MTLBuffer>> *g_transient_buffers;
static id g_model_residency_set;

typedef struct {
    id<MTLBuffer> __strong mask;
    id<MTLBuffer> __strong blk;
    NSUInteger mask_bytes;
    NSUInteger blk_bytes;
    uint32_t kind;
    uint32_t n_tokens;
    uint32_t n_comp;
    uint32_t n_keys;
    uint32_t window;
    uint32_t ratio;
    uint32_t nqptg;
    uint32_t ncpsg;
    bool has_kvpad;
    bool bc_mask;
    bool valid;
    bool blk_ready;
} ds4_gpu_zero_prefix_prefill_mask_cache_entry;

enum {
    DS4_GPU_PREFILL_MASK_CACHE_RAW = 1,
    DS4_GPU_PREFILL_MASK_CACHE_RATIO4 = 2,
    DS4_GPU_PREFILL_MASK_CACHE_RATIO128 = 3,
    DS4_GPU_PREFILL_MASK_CACHE_SLOTS = 3,
};

static ds4_gpu_zero_prefix_prefill_mask_cache_entry
    g_zero_prefix_prefill_mask_cache[DS4_GPU_PREFILL_MASK_CACHE_SLOTS];
static void ds4_gpu_invalidate_zero_prefix_prefill_block_maps(void);
static id<MTLBuffer> g_flash_attn_mask_buffer;
static id<MTLBuffer> g_flash_attn_zero_mask_buffer;
static id<MTLBuffer> g_flash_attn_pad_buffer;
static id<MTLBuffer> g_flash_attn_tmp_buffer;
static id<MTLBuffer> g_flash_attn_blk_buffer;
static id<MTLBuffer> g_flash_attn_ring_buffer;
static id<MTLBuffer> g_flash_attn_kv_buffer;
static id<MTLBuffer> g_glm_flash_attn_mask_buffer;
static id<MTLBuffer> g_compressor_pool_kv_buffer;
static id<MTLBuffer> g_compressor_pool_score_buffer;
static id<MTLBuffer> g_compressor_pool_score_cont_buffer;
static id<MTLBuffer> g_compressor_pool_softmax_buffer;
static id<MTLBuffer> g_compressor_pool_product_buffer;
static id<MTLBuffer> g_compressor_store_ape_buffer;
static id<MTLBuffer> g_compressor_store_score_buffer;
static id<MTLBuffer> g_embed_rows_buffer;
static id<MTLBuffer> g_router_selection_buffer;
static id<MTLBuffer> g_router_weight_sum_buffer;
static id<MTLBuffer> g_indexer_head_scores_buffer;
static id<MTLBuffer> g_indexer_topk_buffer;
static id<MTLBuffer> g_indexed_topk_buffer;
static id<MTLBuffer> g_f16_round_scratch_buffer;
static id<MTLBuffer> g_raw_store_round_buffer;
static id<MTLBuffer> g_moe_gate_scratch_buffer;
static id<MTLBuffer> g_moe_down_scratch_buffer;
static id<MTLBuffer> g_moe_id_map_buffer;
static id<MTLBuffer> g_moe_q4_gate_slots_buffer;
static id<MTLBuffer> g_moe_q4_up_slots_buffer;
static id<MTLBuffer> g_moe_q4_down_slots_buffer;
static id<MTLBuffer> g_attn_out_group_ids_buffer;
static int g_model_fd = -1;
static const void *g_model_map_ptr;
static uint64_t g_model_map_size;
static uint64_t g_model_mapped_offset;
static uint64_t g_model_mapped_size;
static uint64_t g_model_mapped_max_tensor_bytes;
static uint64_t g_tensor_alloc_live_bytes;
static uint64_t g_tensor_alloc_peak_bytes;
static pthread_mutex_t g_tensor_mu = PTHREAD_MUTEX_INITIALIZER;
static uintptr_t *g_tensor_live_slots;
static size_t g_tensor_live_cap;
static size_t g_tensor_live_count;
static size_t g_tensor_live_tombs;
static uint64_t g_model_wrap_count;
static uint64_t g_model_wrap_bytes;
static uint64_t g_model_wrap_max_bytes;
static uint64_t g_model_buffer_cache_bytes;
static uint64_t g_model_buffer_cache_evictions;
static int g_model_buffer_cache_over_limit;
static uint64_t g_stream_expert_cache_bytes;
static uint64_t g_stream_expert_cache_expert_bytes;
static uint32_t g_stream_expert_cache_entry_count;
static uint32_t g_stream_expert_cache_budget_override;
static uint64_t g_stream_expert_cache_hits;
static uint64_t g_stream_expert_cache_misses;
static uint64_t g_stream_expert_cache_evictions;
static uint64_t g_stream_expert_cache_wraps;
static uint64_t g_stream_expert_cache_clock;
static uint64_t g_stream_expert_cache_evict_advise_bytes;
static uint64_t g_stream_expert_cache_willneed_advise_bytes;
static uint64_t g_stream_expert_cache_pread_bytes;
static double g_stream_expert_cache_pread_ms;
static uint64_t g_stream_expert_cache_buffer_allocs;
static uint64_t g_stream_expert_cache_buffer_reuses;
static uint64_t g_stream_expert_cache_decode_tokens;
static uint64_t g_stream_expert_cache_hotness_decay_token;
static uint64_t g_stream_expert_timing_selected_calls;
static double g_stream_expert_timing_selected_read_ms;
static double g_stream_expert_timing_selected_sync_ms;
static double g_stream_expert_timing_selected_copy_ms;
static double g_stream_expert_timing_selected_bind_ms;
static uint64_t g_stream_expert_timing_split_layers;
static uint64_t g_stream_expert_timing_split_resident_experts;
static uint64_t g_stream_expert_timing_split_missing_experts;
static double g_stream_expert_timing_split_resident_ms;
static double g_stream_expert_timing_split_missing_ms;
static double g_stream_expert_timing_split_missing_load_ms;
static double g_stream_expert_timing_split_missing_slot_ms;
static double g_stream_expert_timing_split_missing_prune_ms;
static double g_stream_expert_timing_split_missing_addr_ms;
static double g_stream_expert_timing_split_missing_wait_ms;
static uint64_t g_stream_expert_timing_load_calls;
static double g_stream_expert_timing_load_prepare_ms;
static double g_stream_expert_timing_load_pread_ms;
static double g_stream_expert_timing_load_modify_ms;
static double g_stream_expert_timing_load_install_ms;
static uint64_t g_stream_expert_timing_prepare_batch_reuse_calls;
static double g_stream_expert_timing_prepare_batch_reuse_ms;
static uint64_t g_stream_expert_timing_prepare_buffer_calls;
static double g_stream_expert_timing_prepare_buffer_ms;
static uint64_t g_stream_expert_timing_prepare_task_experts;
static double g_stream_expert_timing_prepare_task_ms;
static uint64_t g_stream_expert_timing_reuse_scan_calls;
static uint64_t g_stream_expert_timing_reuse_scan_entries;
static double g_stream_expert_timing_reuse_scan_ms;
static double g_stream_expert_timing_reuse_clear_ms;
static uint64_t g_stream_expert_timing_readahead_calls;
static uint64_t g_stream_expert_timing_readahead_bytes;
static double g_stream_expert_timing_readahead_ms;
static uint64_t g_stream_expert_timing_cache_all_resident_layers;
static uint64_t g_stream_expert_timing_cache_all_missing_layers;
static uint64_t g_stream_expert_timing_cache_mixed_layers;
static uint64_t g_stream_expert_timing_cache_resident_experts;
static uint64_t g_stream_expert_timing_cache_missing_experts;
typedef struct {
    uint64_t selected_calls;
    double selected_read_ms;
    double selected_sync_ms;
    double selected_copy_ms;
    double selected_bind_ms;
    uint64_t split_layers;
    uint64_t split_resident_experts;
    uint64_t split_missing_experts;
    double split_resident_ms;
    double split_missing_ms;
    double split_missing_load_ms;
    double split_missing_slot_ms;
    double split_missing_prune_ms;
    double split_missing_addr_ms;
    double split_missing_wait_ms;
    uint64_t load_calls;
    double load_prepare_ms;
    double load_pread_ms;
    double load_modify_ms;
    double load_install_ms;
    uint64_t prepare_batch_reuse_calls;
    double prepare_batch_reuse_ms;
    uint64_t prepare_buffer_calls;
    double prepare_buffer_ms;
    uint64_t prepare_task_experts;
    double prepare_task_ms;
    uint64_t reuse_scan_calls;
    uint64_t reuse_scan_entries;
    double reuse_scan_ms;
    double reuse_clear_ms;
    uint64_t readahead_calls;
    uint64_t readahead_bytes;
    double readahead_ms;
    uint64_t cache_all_resident_layers;
    uint64_t cache_all_missing_layers;
    uint64_t cache_mixed_layers;
    uint64_t cache_resident_experts;
    uint64_t cache_missing_experts;
} ds4_gpu_stream_expert_timing_snapshot;
static ds4_gpu_stream_expert_timing_snapshot g_stream_expert_timing_last_report;
static int g_stream_prefill_batch_selected_addr_building;
static int g_glm_stream_expert_addr_table_building;
static uint64_t g_model_residency_count;
static int g_model_residency_added_to_queue;
static int g_glm_model_mode;
static int g_ssd_streaming_mode;
static int g_glm_streaming_prefill_full_layer_runtime;
static int g_metal4_runtime_available;
static int g_metal4_family_supported;
static int g_metal4_queue_supported;
static int g_metal4_m5_neural_accelerators_hint;
static int g_metal4_tensor_api_enabled;
static int g_metal4_tensor_api_compile_supported;
static char g_metal_device_name[128];
static int ds4_gpu_model_map_log_enabled(void);
static int ds4_gpu_stream_expert_cache_note_expert_size(
        uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes);
static uint32_t ds4_gpu_stream_expert_cache_configured_budget(void);
static void ds4_gpu_stream_expert_cache_clear_all(int reset_stats);
static void ds4_gpu_stream_expert_pending_load_clear(void);
static void ds4_gpu_stream_expert_pread_pool_shutdown(void);
static int ds4_gpu_stream_expert_timing_summary_enabled(void);
static int ds4_gpu_stream_expert_cache_entry_protected(
        uint32_t       layer,
        uint32_t       expert,
        uint32_t       protect_layer,
        const int32_t *protect_ids,
        uint32_t       n_protect);

/* The async selected-load worker registers itself so cache paths that would
 * flush/wait on command buffers (a race against the encoding thread) fail
 * the load instead; the caller then retries on the main thread. */
static pthread_t g_stream_expert_service_thread;
static int g_stream_expert_service_thread_set;

void ds4_gpu_stream_expert_cache_note_service_thread(void) {
    g_stream_expert_service_thread = pthread_self();
    g_stream_expert_service_thread_set = 1;
}

static int ds4_gpu_stream_expert_cache_on_service_thread(void) {
    return g_stream_expert_service_thread_set &&
           pthread_equal(pthread_self(), g_stream_expert_service_thread);
}
static NSUInteger g_flash_attn_mask_bytes;
static NSUInteger g_flash_attn_zero_mask_bytes;
static NSUInteger g_flash_attn_pad_bytes;
static NSUInteger g_flash_attn_tmp_bytes;
static NSUInteger g_flash_attn_blk_bytes;
static NSUInteger g_flash_attn_ring_bytes;
static NSUInteger g_flash_attn_kv_bytes;
static NSUInteger g_glm_flash_attn_mask_bytes;
static uint32_t g_glm_flash_attn_mask_pos0;
static uint32_t g_glm_flash_attn_mask_tokens;
static uint32_t g_glm_flash_attn_mask_cache_len;
static int g_glm_flash_attn_mask_valid;
static NSUInteger g_compressor_pool_kv_bytes;
static NSUInteger g_compressor_pool_score_bytes;
static NSUInteger g_compressor_pool_score_cont_bytes;
static NSUInteger g_compressor_pool_softmax_bytes;
static NSUInteger g_compressor_pool_product_bytes;
static NSUInteger g_compressor_store_ape_bytes;
static NSUInteger g_compressor_store_score_bytes;
static NSUInteger g_embed_rows_bytes;
static NSUInteger g_router_selection_bytes;
static NSUInteger g_router_weight_sum_bytes;
static NSUInteger g_indexer_head_scores_bytes;
static NSUInteger g_indexer_topk_bytes;
static NSUInteger g_indexed_topk_bytes;
static NSUInteger g_f16_round_scratch_bytes;
static NSUInteger g_raw_store_round_bytes;
static NSUInteger g_moe_gate_scratch_bytes;
static NSUInteger g_moe_down_scratch_bytes;
static NSUInteger g_moe_id_map_bytes;
static NSUInteger g_moe_q4_gate_slots_bytes;
static NSUInteger g_moe_q4_up_slots_bytes;
static NSUInteger g_moe_q4_down_slots_bytes;
static NSUInteger g_attn_out_group_ids_bytes;
static int g_initialized;
static int g_quality_mode;
static int g_mpp_invalid_env_reported;
#define DS4_METAL_MAX_ROUTED_EXPERT_USED 8
static int32_t g_routed_moe_selected_override[DS4_METAL_MAX_ROUTED_EXPERT_USED];
static uint32_t g_routed_moe_selected_override_n;
static int g_moe_selected_trace_record_initialized;
static FILE *g_moe_selected_trace_record_fp;
static uint64_t g_moe_selected_trace_record_count;
static int g_moe_selected_trace_replay_initialized;
static int32_t *g_moe_selected_trace_replay_ids;
static uint64_t g_moe_selected_trace_replay_count;
static uint64_t g_moe_selected_trace_replay_pos;

static double ds4_gpu_gib(uint64_t bytes);

static uint64_t ds4_gpu_system_memory_bytes(void) {
    uint64_t bytes = 0;
    size_t len = sizeof(bytes);
    if (sysctlbyname("hw.memsize", &bytes, &len, NULL, 0) != 0) return 0;
    return len == sizeof(bytes) ? bytes : 0;
}

static void ds4_gpu_print_device_summary(void) {
    const char *name = g_device.name ? [g_device.name UTF8String] : "unknown Metal device";
    uint64_t mem = ds4_gpu_system_memory_bytes();
    if (mem) {
        double gib = (double)mem / 1024.0 / 1024.0 / 1024.0;
        fprintf(stderr, "ds4: Metal device %s, %.2f GiB RAM\n", name, gib);
    } else {
        fprintf(stderr, "ds4: Metal device %s\n", name);
    }
}

#define DS4_METAL_MAX_MODEL_VIEWS 4096
/* Compatibility fallback for callers that cannot provide a parsed GGUF tensor
 * span. The normal DS4 engine passes the exact maximum tensor byte size. */
#define DS4_METAL_FALLBACK_MAX_TENSOR_BYTES (4ull * 1024ull * 1024ull * 1024ull)

typedef struct {
    __strong id<MTLBuffer> buffer;
    const void *model_map;
    uint64_t model_size;
    uint64_t model_offset;
    uint64_t bytes;
} ds4_gpu_model_view;

static ds4_gpu_model_view g_model_views[DS4_METAL_MAX_MODEL_VIEWS];
static uint32_t g_model_view_count;

enum {
    DS4_METAL_STREAM_EXPERT_CACHE_MAX_LAYER = 80,
    DS4_METAL_STREAM_EXPERT_CACHE_MAX_EXPERT = 384,
    DS4_METAL_STREAM_EXPERT_CACHE_MAX_SELECTED = DS4_METAL_MAX_ROUTED_EXPERT_USED,
    DS4_METAL_STREAM_EXPERT_CACHE_MAX_ENTRIES =
        DS4_METAL_STREAM_EXPERT_CACHE_MAX_LAYER *
        DS4_METAL_STREAM_EXPERT_CACHE_MAX_EXPERT,
    DS4_METAL_STREAM_EXPERT_CACHE_MAX_SLABS = 256,
    DS4_METAL_STREAM_EXPERT_HOTNESS_DECAY_TOKENS = 16,
    DS4_METAL_STREAM_EXPERT_VALIDATE_WORDS = 16,
};

typedef struct {
    uint32_t layer;
    uint32_t expert;
    uint64_t hits;
} ds4_gpu_moe_selected_hotlist_entry;

static int g_moe_selected_hotlist_initialized;
static uint64_t
    g_moe_selected_hotlist_counts[DS4_METAL_STREAM_EXPERT_CACHE_MAX_LAYER][DS4_METAL_STREAM_EXPERT_CACHE_MAX_EXPERT];
static uint64_t g_moe_selected_hotlist_records;
static uint64_t g_moe_selected_hotlist_selections;

typedef struct {
    __strong id<MTLBuffer> gate_buffer;
    __strong id<MTLBuffer> up_buffer;
    __strong id<MTLBuffer> down_buffer;
    const void *model_map;
    uint64_t model_size;
    uint64_t gate_abs_offset;
    uint64_t up_abs_offset;
    uint64_t down_abs_offset;
    uint64_t gate_expert_bytes;
    uint64_t down_expert_bytes;
    uint64_t logical_bytes;
    uint64_t last_used;
    uint64_t use_count;
    NSUInteger gate_inner;
    NSUInteger up_inner;
    NSUInteger down_inner;
    uint64_t inflight_seq;
    uint32_t slab_slot;
    uint8_t valid;
    uint8_t slab_backed;
} ds4_gpu_stream_expert_cache_entry;

typedef struct {
    __strong id<MTLBuffer> gate_buffer;
    __strong id<MTLBuffer> up_buffer;
    __strong id<MTLBuffer> down_buffer;
    NSUInteger gate_inner;
    NSUInteger up_inner;
    NSUInteger down_inner;
} ds4_gpu_stream_expert_reusable_buffers;

static ds4_gpu_stream_expert_cache_entry
    g_stream_expert_cache[DS4_METAL_STREAM_EXPERT_CACHE_MAX_LAYER][DS4_METAL_STREAM_EXPERT_CACHE_MAX_EXPERT];
static ds4_gpu_stream_expert_cache_entry
    g_stream_full_expert_addr_entry[DS4_METAL_STREAM_EXPERT_CACHE_MAX_LAYER];
static uint32_t g_stream_expert_cache_layer_count[DS4_METAL_STREAM_EXPERT_CACHE_MAX_LAYER];
static uint64_t g_stream_expert_cache_layer_hits[DS4_METAL_STREAM_EXPERT_CACHE_MAX_LAYER];
static uint64_t g_stream_expert_cache_layer_misses[DS4_METAL_STREAM_EXPERT_CACHE_MAX_LAYER];
static uint64_t g_stream_expert_cache_layer_evictions[DS4_METAL_STREAM_EXPERT_CACHE_MAX_LAYER];
static uint64_t g_stream_expert_cache_layer_pread_bytes[DS4_METAL_STREAM_EXPERT_CACHE_MAX_LAYER];
static double g_stream_expert_cache_layer_pread_ms[DS4_METAL_STREAM_EXPERT_CACHE_MAX_LAYER];
static uint64_t g_stream_expert_cache_layer_last_hits[DS4_METAL_STREAM_EXPERT_CACHE_MAX_LAYER];
static uint64_t g_stream_expert_cache_layer_last_misses[DS4_METAL_STREAM_EXPERT_CACHE_MAX_LAYER];
static uint64_t g_stream_expert_cache_layer_last_evictions[DS4_METAL_STREAM_EXPERT_CACHE_MAX_LAYER];
static uint64_t g_stream_expert_cache_layer_last_pread_bytes[DS4_METAL_STREAM_EXPERT_CACHE_MAX_LAYER];
static double g_stream_expert_cache_layer_last_pread_ms[DS4_METAL_STREAM_EXPERT_CACHE_MAX_LAYER];
static uint32_t
    g_stream_expert_cache_route_hotness[DS4_METAL_STREAM_EXPERT_CACHE_MAX_LAYER][DS4_METAL_STREAM_EXPERT_CACHE_MAX_EXPERT];
static id<MTLBuffer> g_stream_expert_cache_gate_addr_buffers[DS4_METAL_STREAM_EXPERT_CACHE_MAX_LAYER];
static id<MTLBuffer> g_stream_expert_cache_up_addr_buffers[DS4_METAL_STREAM_EXPERT_CACHE_MAX_LAYER];
static id<MTLBuffer> g_stream_expert_cache_down_addr_buffers[DS4_METAL_STREAM_EXPERT_CACHE_MAX_LAYER];
static id<MTLBuffer> g_stream_expert_cache_slabs[DS4_METAL_STREAM_EXPERT_CACHE_MAX_SLABS];
static uint32_t g_stream_expert_cache_slab_start_slot[DS4_METAL_STREAM_EXPERT_CACHE_MAX_SLABS];
static uint32_t g_stream_expert_cache_slab_slot_count[DS4_METAL_STREAM_EXPERT_CACHE_MAX_SLABS];
static uint32_t g_stream_expert_cache_slab_slots_used[DS4_METAL_STREAM_EXPERT_CACHE_MAX_SLABS];
static uint32_t g_stream_expert_cache_slab_count;
static uint32_t g_stream_expert_cache_slab_total_slots;
static uint32_t g_stream_expert_cache_free_slots[DS4_METAL_STREAM_EXPERT_CACHE_MAX_ENTRIES];
static uint32_t g_stream_expert_cache_free_slot_count;
static uint64_t g_stream_expert_cache_slab_slot_bytes;
static uint64_t g_stream_expert_cache_cb_seq;
static uint64_t g_stream_expert_cache_done_seq;
static uint64_t g_stream_expert_cache_batch_seq;
static uint64_t g_stream_expert_cache_owned_seq;
static uint64_t g_stream_expert_cache_pending_max_seq;
static id<MTLBuffer> g_stream_compact_gate_addr_buffers[DS4_METAL_STREAM_EXPERT_CACHE_MAX_LAYER];
static id<MTLBuffer> g_stream_compact_up_addr_buffers[DS4_METAL_STREAM_EXPERT_CACHE_MAX_LAYER];
static id<MTLBuffer> g_stream_compact_down_addr_buffers[DS4_METAL_STREAM_EXPERT_CACHE_MAX_LAYER];
static id<MTLBuffer> g_stream_compact_selected_buffers[DS4_METAL_STREAM_EXPERT_CACHE_MAX_LAYER];
static id<MTLBuffer> g_stream_selected_id_buffers[DS4_METAL_STREAM_EXPERT_CACHE_MAX_LAYER];
static id<MTLBuffer> g_stream_expert_validate_status_buffer;

@interface DS4MetalTensor : NSObject
@property(nonatomic, strong) id<MTLBuffer> buffer;
@property(nonatomic, assign) uint64_t offset;
@property(nonatomic, assign) uint64_t bytes;
@property(nonatomic, assign) uint8_t owner;
@end

@implementation DS4MetalTensor
@end

@interface DS4MetalQ4ExpertTable : NSObject
@property(nonatomic, strong) id<MTLBuffer> argumentBuffer;
@property(nonatomic, strong) id<MTLBuffer> addressBuffer;
@property(nonatomic, strong) NSMutableArray<id<MTLBuffer>> *expertBuffers;
@property(nonatomic, strong) id residencySet;
@property(nonatomic, assign) BOOL residencySetAddedToQueue;
@property(nonatomic, assign) uint32_t nExpert;
@property(nonatomic, assign) uint64_t expertBytes;
@end

@implementation DS4MetalQ4ExpertTable
- (void)dealloc {
#if TARGET_OS_OSX
    if (@available(macOS 15.0, *)) {
        if (_residencySet) {
            if (_residencySetAddedToQueue &&
                g_queue &&
                [g_queue respondsToSelector:@selector(removeResidencySet:)]) {
                [g_queue removeResidencySet:_residencySet];
            }
            [_residencySet endResidency];
        }
    }
#endif
}
@end

@interface DS4MetalQ4LayerResidency : NSObject
@property(nonatomic, strong) id residencySet;
@property(nonatomic, assign) BOOL addedToQueue;
@end

@implementation DS4MetalQ4LayerResidency
- (void)dealloc {
#if TARGET_OS_OSX
    if (@available(macOS 15.0, *)) {
        if (_residencySet) {
            if (_addedToQueue &&
                g_queue &&
                [g_queue respondsToSelector:@selector(removeResidencySet:)]) {
                [g_queue removeResidencySet:_residencySet];
            }
            [_residencySet endResidency];
        }
    }
#endif
}
@end

static DS4MetalTensor *ds4_gpu_tensor_obj(ds4_gpu_tensor *tensor) {
    return (__bridge DS4MetalTensor *)tensor;
}

static const DS4MetalTensor *ds4_gpu_tensor_const_obj(const ds4_gpu_tensor *tensor) {
    return (__bridge const DS4MetalTensor *)tensor;
}

/* C code owns ds4_gpu_tensor handles as retained Objective-C objects.  Freeing
 * the same opaque handle twice would make the second __bridge_transfer release
 * an already-deallocated object, which macOS reports as malloc corruption.  The
 * live table lets free validate a handle before touching Objective-C state; the
 * same mutex also serializes the diagnostic allocation counters. */
static uint64_t ds4_gpu_tensor_ptr_hash(uintptr_t ptr) {
    uint64_t x = (uint64_t)(ptr >> 4);
    x ^= x >> 33;
    x *= UINT64_C(0xff51afd7ed558ccd);
    x ^= x >> 33;
    x *= UINT64_C(0xc4ceb9fe1a85ec53);
    x ^= x >> 33;
    return x;
}

static int ds4_gpu_tensor_live_resize_locked(size_t min_cap) {
    size_t new_cap = 1024;
    while (new_cap < min_cap) new_cap <<= 1;

    uintptr_t *new_slots = calloc(new_cap, sizeof(new_slots[0]));
    if (!new_slots) return 0;

    for (size_t i = 0; i < g_tensor_live_cap; i++) {
        const uintptr_t key = g_tensor_live_slots[i];
        if (key == 0 || key == UINTPTR_MAX) continue;

        size_t idx = (size_t)ds4_gpu_tensor_ptr_hash(key) & (new_cap - 1);
        while (new_slots[idx] != 0) idx = (idx + 1) & (new_cap - 1);
        new_slots[idx] = key;
    }

    free(g_tensor_live_slots);
    g_tensor_live_slots = new_slots;
    g_tensor_live_cap = new_cap;
    g_tensor_live_tombs = 0;
    return 1;
}

static int ds4_gpu_tensor_live_insert_locked(const void *ptr) {
    if (!ptr || (uintptr_t)ptr == UINTPTR_MAX) return 0;
    if ((g_tensor_live_count + g_tensor_live_tombs + 1) * 10 >=
        g_tensor_live_cap * 7)
    {
        const size_t min_cap = g_tensor_live_cap ? g_tensor_live_cap * 2 : 1024;
        if (!ds4_gpu_tensor_live_resize_locked(min_cap)) return 0;
    }

    const uintptr_t key = (uintptr_t)ptr;
    size_t idx = (size_t)ds4_gpu_tensor_ptr_hash(key) & (g_tensor_live_cap - 1);
    size_t tomb = (size_t)-1;
    for (;;) {
        const uintptr_t cur = g_tensor_live_slots[idx];
        if (cur == key) return 0;
        if (cur == UINTPTR_MAX) {
            if (tomb == (size_t)-1) tomb = idx;
        } else if (cur == 0) {
            if (tomb != (size_t)-1) {
                idx = tomb;
                g_tensor_live_tombs--;
            }
            g_tensor_live_slots[idx] = key;
            g_tensor_live_count++;
            return 1;
        }
        idx = (idx + 1) & (g_tensor_live_cap - 1);
    }
}

static int ds4_gpu_tensor_live_remove_locked(const void *ptr) {
    if (!ptr || g_tensor_live_cap == 0) return 0;

    const uintptr_t key = (uintptr_t)ptr;
    size_t idx = (size_t)ds4_gpu_tensor_ptr_hash(key) & (g_tensor_live_cap - 1);
    for (;;) {
        const uintptr_t cur = g_tensor_live_slots[idx];
        if (cur == 0) return 0;
        if (cur == key) {
            g_tensor_live_slots[idx] = UINTPTR_MAX;
            g_tensor_live_count--;
            g_tensor_live_tombs++;
            return 1;
        }
        idx = (idx + 1) & (g_tensor_live_cap - 1);
    }
}

static int ds4_gpu_tensor_track_alloc_locked(
        const void *ptr,
        uint64_t bytes,
        uint64_t *live_snap,
        uint64_t *peak_snap)
{
    if (!ds4_gpu_tensor_live_insert_locked(ptr)) return 0;

    g_tensor_alloc_live_bytes += bytes;
    if (g_tensor_alloc_live_bytes > g_tensor_alloc_peak_bytes) {
        g_tensor_alloc_peak_bytes = g_tensor_alloc_live_bytes;
    }
    if (live_snap) *live_snap = g_tensor_alloc_live_bytes;
    if (peak_snap) *peak_snap = g_tensor_alloc_peak_bytes;
    return 1;
}

static int ds4_gpu_tensor_track_view_locked(const void *ptr) {
    return ds4_gpu_tensor_live_insert_locked(ptr);
}

static int ds4_gpu_tensor_prepare_free(
        ds4_gpu_tensor *tensor,
        uint8_t *owner,
        uint64_t *bytes,
        uint64_t *live_snap,
        uint64_t *peak_snap)
{
    pthread_mutex_lock(&g_tensor_mu);
    if (!ds4_gpu_tensor_live_remove_locked(tensor)) {
        pthread_mutex_unlock(&g_tensor_mu);
        fprintf(stderr,
                "ds4: Metal tensor free ignored for unknown handle %p\n",
                (void *)tensor);
        return 0;
    }

    DS4MetalTensor *obj = ds4_gpu_tensor_obj(tensor);
    const uint8_t obj_owner = obj.owner;
    const uint64_t obj_bytes = obj.bytes;
    if (obj_owner) {
        if (obj_bytes <= g_tensor_alloc_live_bytes) {
            g_tensor_alloc_live_bytes -= obj_bytes;
        } else {
            g_tensor_alloc_live_bytes = 0;
        }
    }
    if (owner) *owner = obj_owner;
    if (bytes) *bytes = obj_bytes;
    if (live_snap) *live_snap = g_tensor_alloc_live_bytes;
    if (peak_snap) *peak_snap = g_tensor_alloc_peak_bytes;
    pthread_mutex_unlock(&g_tensor_mu);
    return 1;
}

static void ds4_gpu_tensor_tracking_reset(void) {
    pthread_mutex_lock(&g_tensor_mu);
    if (g_tensor_live_count != 0) {
        fprintf(stderr,
                "ds4: Metal cleanup discarded %zu live tensor handles\n",
                g_tensor_live_count);
    }
    free(g_tensor_live_slots);
    g_tensor_live_slots = NULL;
    g_tensor_live_cap = 0;
    g_tensor_live_count = 0;
    g_tensor_live_tombs = 0;
    g_tensor_alloc_live_bytes = 0;
    g_tensor_alloc_peak_bytes = 0;
    pthread_mutex_unlock(&g_tensor_mu);
}

static id<MTLBuffer> ds4_gpu_tensor_buffer(const ds4_gpu_tensor *tensor) {
    if (!tensor) return nil;
    const DS4MetalTensor *obj = ds4_gpu_tensor_const_obj(tensor);
    return obj.buffer;
}

static NSUInteger ds4_gpu_tensor_offset(const ds4_gpu_tensor *tensor) {
    if (!tensor) return 0;
    const DS4MetalTensor *obj = ds4_gpu_tensor_const_obj(tensor);
    return (NSUInteger)obj.offset;
}

static id<MTLCommandBuffer> ds4_gpu_new_command_buffer(void);
static void ds4_gpu_stream_expert_cache_note_owned_created(void);

static id<MTLCommandBuffer> ds4_gpu_command_buffer(int *owned) {
    if (g_batch_cb) {
        *owned = 0;
        return g_batch_cb;
    }
    *owned = 1;
    id<MTLCommandBuffer> cb = ds4_gpu_new_command_buffer();
    if (cb) ds4_gpu_stream_expert_cache_note_owned_created();
    return cb;
}

static id<MTLComputeCommandEncoder> ds4_gpu_compute_encoder(id<MTLCommandBuffer> cb) {
    if (g_batch_cb && cb == g_batch_cb) {
        g_batch_has_work = YES;
        if (!g_batch_enc) g_batch_enc = [cb computeCommandEncoder];
        return g_batch_enc;
    }
    return [cb computeCommandEncoder];
}

static void ds4_gpu_end_compute_encoder(id<MTLCommandBuffer> cb, id<MTLComputeCommandEncoder> enc) {
    if (!enc) return;
    if (g_batch_cb && cb == g_batch_cb && enc == g_batch_enc) return;
    [enc endEncoding];
}

static void ds4_gpu_close_batch_encoder(void) {
    if (!g_batch_enc) return;
    [g_batch_enc endEncoding];
    g_batch_enc = nil;
}

static double g_gpu_busy_accum;
static uint64_t g_gpu_busy_cbs;

static int ds4_gpu_wait_command_buffer(id<MTLCommandBuffer> cb, const char *label) {
    [cb waitUntilCompleted];
    if (getenv("DS4_METAL_GPU_BUSY_PROFILE")) {
        const double busy = cb.GPUEndTime - cb.GPUStartTime;
        if (busy > 0) g_gpu_busy_accum += busy;
        if ((++g_gpu_busy_cbs % 64u) == 0u) {
            fprintf(stderr, "ds4: gpu busy accum %.1f ms over %llu cbs\n",
                    g_gpu_busy_accum * 1000.0,
                    (unsigned long long)g_gpu_busy_cbs);
        }
    }
    if (cb.status == MTLCommandBufferStatusError) {
        fprintf(stderr, "ds4: Metal %s failed: %s\n",
                label, [[cb.error localizedDescription] UTF8String]);
        return 0;
    }
    return 1;
}

static id<MTLCommandBuffer> ds4_gpu_new_command_buffer(void) {
    static int initialized;
    static int use_unretained;
    if (!initialized) {
        use_unretained = getenv("DS4_METAL_UNRETAINED_COMMAND_BUFFERS") != NULL;
        initialized = 1;
    }
    if (use_unretained) {
        return [g_queue commandBufferWithUnretainedReferences];
    }
    return [g_queue commandBuffer];
}

static uint64_t ds4_gpu_exact_view_cache_limit_bytes(void) {
    static int initialized;
    static uint64_t limit_bytes;
    if (initialized) return limit_bytes;

    const uint64_t mib = 1024ull * 1024ull;
    const uint64_t gib = 1024ull * mib;
    limit_bytes = 64ull * gib;

    const char *gib_env = getenv("DS4_METAL_EXACT_VIEW_CACHE_GIB");
    if (gib_env && gib_env[0]) {
        char *end = NULL;
        unsigned long long v = strtoull(gib_env, &end, 10);
        if (end != gib_env && *end == '\0') {
            limit_bytes = v > UINT64_MAX / gib ? UINT64_MAX : (uint64_t)v * gib;
        }
    }

    const char *mib_env = getenv("DS4_METAL_EXACT_VIEW_CACHE_MIB");
    if (mib_env && mib_env[0]) {
        char *end = NULL;
        unsigned long long v = strtoull(mib_env, &end, 10);
        if (end != mib_env && *end == '\0') {
            limit_bytes = v > UINT64_MAX / mib ? UINT64_MAX : (uint64_t)v * mib;
        }
    }

    initialized = 1;
    return limit_bytes;
}

static void ds4_gpu_model_buffer_cache_note_insert(uint64_t bytes) {
    if (g_model_buffer_cache_bytes > UINT64_MAX - bytes) {
        g_model_buffer_cache_bytes = UINT64_MAX;
    } else {
        g_model_buffer_cache_bytes += bytes;
    }

    const uint64_t limit = ds4_gpu_exact_view_cache_limit_bytes();
    if (limit != 0 && g_model_buffer_cache_bytes > limit) {
        g_model_buffer_cache_over_limit = 1;
    }
}

static void ds4_gpu_model_buffer_cache_clear(const char *reason) {
    if (!g_model_buffer_cache) {
        g_model_buffer_cache_bytes = 0;
        g_model_buffer_cache_over_limit = 0;
        return;
    }

    const NSUInteger entries = [g_model_buffer_cache count];
    if (entries != 0) {
        if (getenv("DS4_METAL_EXACT_VIEW_CACHE_PROFILE") != NULL) {
            fprintf(stderr,
                    "ds4: Metal exact model view cache evict reason=%s entries=%lu bytes=%.2f GiB limit=%.2f GiB\n",
                    reason ? reason : "unknown",
                    (unsigned long)entries,
                    ds4_gpu_gib(g_model_buffer_cache_bytes),
                    ds4_gpu_gib(ds4_gpu_exact_view_cache_limit_bytes()));
        }
        [g_model_buffer_cache removeAllObjects];
        g_model_buffer_cache_evictions++;
    }
    g_model_buffer_cache_bytes = 0;
    g_model_buffer_cache_over_limit = 0;
}

static void ds4_gpu_model_buffer_cache_maybe_evict(const char *reason) {
    if (g_model_buffer_cache_over_limit) {
        ds4_gpu_model_buffer_cache_clear(reason);
    }
}

static uint64_t ds4_gpu_stream_expert_cache_next_cb_seq(void) {
    if (g_stream_expert_cache_cb_seq == UINT64_MAX) {
        /*
         * A real wrap would require an astronomical number of command buffers.
         * Resetting the epoch space is still safer than letting zero become a
         * valid in-flight marker.
         */
        g_stream_expert_cache_cb_seq = 0;
        g_stream_expert_cache_done_seq = 0;
        g_stream_expert_cache_batch_seq = 0;
        g_stream_expert_cache_owned_seq = 0;
        g_stream_expert_cache_pending_max_seq = 0;
        for (uint32_t layer = 0; layer < DS4_METAL_STREAM_EXPERT_CACHE_MAX_LAYER; layer++) {
            for (uint32_t expert = 0; expert < DS4_METAL_STREAM_EXPERT_CACHE_MAX_EXPERT; expert++) {
                g_stream_expert_cache[layer][expert].inflight_seq = 0;
            }
        }
    }
    return ++g_stream_expert_cache_cb_seq;
}

static void ds4_gpu_stream_expert_cache_note_batch_created(void) {
    g_stream_expert_cache_batch_seq =
        ds4_gpu_stream_expert_cache_next_cb_seq();
}

static void ds4_gpu_stream_expert_cache_note_batch_committed(void) {
    if (g_stream_expert_cache_batch_seq > g_stream_expert_cache_pending_max_seq) {
        g_stream_expert_cache_pending_max_seq = g_stream_expert_cache_batch_seq;
    }
    g_stream_expert_cache_batch_seq = 0;
}

static void ds4_gpu_stream_expert_cache_note_owned_created(void) {
    g_stream_expert_cache_owned_seq =
        ds4_gpu_stream_expert_cache_next_cb_seq();
}

static void ds4_gpu_stream_expert_cache_note_pending_completed(void) {
    if (g_stream_expert_cache_pending_max_seq > g_stream_expert_cache_done_seq) {
        g_stream_expert_cache_done_seq = g_stream_expert_cache_pending_max_seq;
    }
    g_stream_expert_cache_pending_max_seq = 0;
}

static void ds4_gpu_stream_expert_cache_note_owned_completed(void) {
    if (g_stream_expert_cache_owned_seq > g_stream_expert_cache_done_seq) {
        g_stream_expert_cache_done_seq = g_stream_expert_cache_owned_seq;
    }
    g_stream_expert_cache_owned_seq = 0;
}

static int ds4_gpu_stream_expert_cache_entry_inflight(
        const ds4_gpu_stream_expert_cache_entry *e) {
    return e && e->valid && e->inflight_seq > g_stream_expert_cache_done_seq;
}

static int ds4_gpu_stream_expert_cache_mark_inflight(
        ds4_gpu_stream_expert_cache_entry *e) {
    if (!e || !e->valid) return 0;
    const uint64_t seq = g_stream_expert_cache_batch_seq ?
                         g_stream_expert_cache_batch_seq :
                         g_stream_expert_cache_owned_seq;
    if (seq == 0) return 0;
    e->inflight_seq = seq;
    return 1;
}

static int ds4_gpu_stream_expert_cache_mark_entries_inflight(
        ds4_gpu_stream_expert_cache_entry * const *entries,
        uint32_t n_entries,
        uint32_t active_mask) {
    if (!entries || n_entries == 0) return 0;
    for (uint32_t i = 0; i < n_entries; i++) {
        if (active_mask != 0 && (active_mask & (1u << i)) == 0) continue;
        if (!ds4_gpu_stream_expert_cache_mark_inflight(entries[i])) return 0;
    }
    return 1;
}

static int ds4_gpu_stream_expert_cache_wait_inflight(const char *label);

static int ds4_gpu_wait_pending_command_buffers(const char *label) {
    int ok = 1;
    for (id<MTLCommandBuffer> pending in g_pending_cbs) {
        if (!ds4_gpu_wait_command_buffer(pending, label)) ok = 0;
    }
    [g_pending_cbs removeAllObjects];
    ds4_gpu_stream_expert_cache_note_pending_completed();
    if (!ok) ds4_gpu_invalidate_zero_prefix_prefill_block_maps();
    return ok;
}

static int ds4_gpu_finish_command_buffer(id<MTLCommandBuffer> cb, int owned, const char *label) {
    if (!owned) return 1;

    [cb commit];
    int ok = ds4_gpu_wait_pending_command_buffers(label);
    if (!ds4_gpu_wait_command_buffer(cb, label)) {
        ok = 0;
        ds4_gpu_invalidate_zero_prefix_prefill_block_maps();
    }
    ds4_gpu_stream_expert_cache_note_owned_completed();
    [g_transient_buffers removeAllObjects];
    ds4_gpu_model_buffer_cache_maybe_evict(label);
    return ok;
}

static int ds4_gpu_device_name_contains(const char *needle);

static int ds4_gpu_use_m5_private_scratch(void) {
    static int initialized;
    static int enabled;
    if (!initialized) {
        enabled = ds4_gpu_device_name_contains("M5");
        initialized = 1;
    }
    return enabled;
}

static int ds4_gpu_scratch_needs_cpu_access(const char *label) {
    if (!label) return 0;
    return strstr(label, "mask") != NULL ||
           strcmp(label, "ds4_attention_output_group_ids") == 0;
}

static MTLResourceOptions ds4_gpu_model_resource_options(void) {
    MTLResourceOptions options = MTLResourceStorageModeShared;
    if (getenv("DS4_METAL_MODEL_UNTRACKED") != NULL) {
        options |= MTLResourceHazardTrackingModeUntracked;
    }
    return options;
}

static int ds4_gpu_ensure_scratch_buffer(
        id<MTLBuffer> __strong *buffer,
        NSUInteger    *capacity,
        NSUInteger     bytes,
        const char    *label) {
    if (*buffer && *capacity >= bytes) return 1;
    if (bytes == 0) bytes = 1;
    if (bytes > NSUIntegerMax) return 0;

    MTLResourceOptions options = MTLResourceStorageModeShared;
    if (ds4_gpu_use_m5_private_scratch() &&
        !ds4_gpu_scratch_needs_cpu_access(label)) {
        /*
         * M5 scratch buffers that only flow between Metal kernels do not need
         * CPU-visible shared storage. This reduces shared-memory traffic and
         * residency pressure for the long prefill scratch pools without
         * changing the public buffer lifetime model. Keep default hazard
         * tracking because the graph reuses these buffers across dependent
         * compute encoders.
         */
        options = MTLResourceStorageModePrivate;
    }

    *buffer = [g_device newBufferWithLength:bytes options:options];
    if (!*buffer && options != MTLResourceStorageModeShared) {
        *buffer = [g_device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    }
    if (!*buffer) {
        fprintf(stderr, "ds4: failed to allocate Metal scratch buffer %s (%llu bytes)\n",
                label, (unsigned long long)bytes);
        *capacity = 0;
        return 0;
    }
    (*buffer).label = [NSString stringWithUTF8String:label];
    *capacity = bytes;
    return 1;
}

static int ds4_gpu_ensure_zero_attention_mask(NSUInteger bytes) {
    const NSUInteger capacity = 8192u * sizeof(uint16_t);
    if (bytes > capacity) return 0;
    if (g_flash_attn_zero_mask_buffer &&
        g_flash_attn_zero_mask_bytes >= capacity) {
        return 1;
    }
    if (!ds4_gpu_ensure_scratch_buffer(&g_flash_attn_zero_mask_buffer,
                                         &g_flash_attn_zero_mask_bytes,
                                         capacity,
                                         "ds4_flash_attn_zero_mask")) {
        return 0;
    }
    void *contents = [g_flash_attn_zero_mask_buffer contents];
    if (!contents) return 0;
    memset(contents, 0, g_flash_attn_zero_mask_bytes);
    return 1;
}

static uint64_t round_up_u64(uint64_t v, uint64_t align) {
    return (v + align - 1) & ~(align - 1);
}

static uint64_t ds4_gpu_effective_model_max_tensor_bytes(uint64_t map_size, uint64_t max_tensor_bytes) {
    if (max_tensor_bytes != 0) return max_tensor_bytes;
    return map_size < DS4_METAL_FALLBACK_MAX_TENSOR_BYTES ?
           map_size : DS4_METAL_FALLBACK_MAX_TENSOR_BYTES;
}

static id<MTLComputePipelineState> ds4_gpu_get_pipeline(const char *function_name);
static int ds4_gpu_warm_model_views(void);
static double ds4_gpu_gib(uint64_t bytes);

static double ds4_gpu_now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

static int ds4_gpu_moe_selected_hotlist_cmp(const void *a, const void *b) {
    const ds4_gpu_moe_selected_hotlist_entry *ea = a;
    const ds4_gpu_moe_selected_hotlist_entry *eb = b;
    if (ea->hits < eb->hits) return 1;
    if (ea->hits > eb->hits) return -1;
    if (ea->layer != eb->layer) return ea->layer < eb->layer ? -1 : 1;
    if (ea->expert != eb->expert) return ea->expert < eb->expert ? -1 : 1;
    return 0;
}

static int ds4_gpu_moe_selected_hotlist_merge_requested(void) {
    return getenv("DS4_MOE_RECORD_SELECTED_HOTLIST_MERGE") != NULL &&
           getenv("DS4_MOE_RECORD_SELECTED_HOTLIST_FRESH") == NULL;
}

static int ds4_gpu_moe_selected_hotlist_load_existing(const char *path) {
    if (!path || !path[0] || !ds4_gpu_moe_selected_hotlist_merge_requested()) {
        return 1;
    }

    FILE *fp = fopen(path, "rb");
    if (!fp) {
        if (errno == ENOENT) return 1;
        fprintf(stderr, "ds4: failed to open selected hotlist merge file %s\n", path);
        return 0;
    }

    char line[256];
    uint64_t lineno = 0;
    uint64_t loaded_entries = 0;
    uint64_t loaded_hits = 0;
    uint64_t header_records = UINT64_MAX;
    uint64_t header_selections = UINT64_MAX;
    while (fgets(line, sizeof(line), fp)) {
        lineno++;
        char *p = line;
        while (*p && isspace((unsigned char)*p)) p++;
        if (*p == '\0') continue;
        if (*p == '#') {
            unsigned long long value = 0;
            if (sscanf(p, "# layer_records %llu", &value) == 1) {
                header_records = (uint64_t)value;
            } else if (sscanf(p, "# selections %llu", &value) == 1) {
                header_selections = (uint64_t)value;
            }
            continue;
        }

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
        if (layer < DS4_METAL_STREAM_EXPERT_CACHE_MAX_LAYER &&
            expert < DS4_METAL_STREAM_EXPERT_CACHE_MAX_EXPERT &&
            hits != 0) {
            uint64_t *dst = &g_moe_selected_hotlist_counts[layer][expert];
            if (*dst > UINT64_MAX - (uint64_t)hits) {
                *dst = UINT64_MAX;
            } else {
                *dst += (uint64_t)hits;
            }
            loaded_entries++;
            if (loaded_hits > UINT64_MAX - (uint64_t)hits) {
                loaded_hits = UINT64_MAX;
            } else {
                loaded_hits += (uint64_t)hits;
            }
        }
        continue;

bad_line:
        fprintf(stderr,
                "ds4: invalid selected hotlist merge line %" PRIu64 " in %s\n",
                lineno,
                path);
        fclose(fp);
        return 0;
    }
    if (ferror(fp)) {
        fprintf(stderr, "ds4: failed to read selected hotlist merge file %s\n", path);
        fclose(fp);
        return 0;
    }
    fclose(fp);

    g_moe_selected_hotlist_records =
        header_records != UINT64_MAX ? header_records : loaded_hits / 6u;
    g_moe_selected_hotlist_selections =
        header_selections != UINT64_MAX ? header_selections : loaded_hits;
    fprintf(stderr,
            "ds4: merged selected-id hotlist %s "
            "(%" PRIu64 " entries, %" PRIu64 " hits)\n",
            path,
            loaded_entries,
            loaded_hits);
    return 1;
}

static void ds4_gpu_moe_selected_hotlist_close(void) {
    const char *path = getenv("DS4_MOE_RECORD_SELECTED_HOTLIST");
    if (!g_moe_selected_hotlist_initialized || !path || !path[0]) return;

    const size_t cap =
        (size_t)DS4_METAL_STREAM_EXPERT_CACHE_MAX_LAYER *
        DS4_METAL_STREAM_EXPERT_CACHE_MAX_EXPERT;
    ds4_gpu_moe_selected_hotlist_entry *entries =
        malloc(cap * sizeof(entries[0]));
    if (!entries) {
        fprintf(stderr, "ds4: failed to allocate selected hotlist entries\n");
        return;
    }

    size_t n = 0;
    for (uint32_t layer = 0;
         layer < DS4_METAL_STREAM_EXPERT_CACHE_MAX_LAYER;
         layer++) {
        for (uint32_t expert = 0;
             expert < DS4_METAL_STREAM_EXPERT_CACHE_MAX_EXPERT;
             expert++) {
            const uint64_t hits =
                g_moe_selected_hotlist_counts[layer][expert];
            if (hits == 0) continue;
            entries[n++] = (ds4_gpu_moe_selected_hotlist_entry) {
                .layer = layer,
                .expert = expert,
                .hits = hits,
            };
        }
    }
    qsort(entries, n, sizeof(entries[0]), ds4_gpu_moe_selected_hotlist_cmp);

    FILE *fp = fopen(path, "wb");
    if (!fp) {
        fprintf(stderr, "ds4: failed to open selected hotlist file %s\n", path);
        free(entries);
        return;
    }
    fprintf(fp,
            "# ds4 selected-id hotlist v1\n"
            "# layer_records %" PRIu64 "\n"
            "# selections %" PRIu64 "\n"
            "# columns: layer expert hits weight\n",
            g_moe_selected_hotlist_records,
            g_moe_selected_hotlist_selections);
    for (size_t i = 0; i < n; i++) {
        fprintf(fp,
                "%u %u %" PRIu64 " 0\n",
                entries[i].layer,
                entries[i].expert,
                entries[i].hits);
    }
    free(entries);

    if (fclose(fp) != 0) {
        fprintf(stderr, "ds4: failed to close selected hotlist file %s\n", path);
    } else {
        fprintf(stderr,
                "ds4: wrote selected-id hotlist to %s "
                "(%" PRIu64 " layer records, %" PRIu64 " selections)\n",
                path,
                g_moe_selected_hotlist_records,
                g_moe_selected_hotlist_selections);
    }
}

static int ds4_gpu_moe_selected_hotlist_record(
        uint32_t       layer,
        const int32_t *selected_ids,
        uint32_t       n_selected,
        uint32_t       n_total_expert) {
    const char *path = getenv("DS4_MOE_RECORD_SELECTED_HOTLIST");
    if (!path || !path[0]) return 1;
    if (!selected_ids ||
        n_selected == 0 ||
        n_selected > DS4_METAL_STREAM_EXPERT_CACHE_MAX_SELECTED) {
        return 0;
    }
    if (!g_moe_selected_hotlist_initialized) {
        g_moe_selected_hotlist_initialized = 1;
        if (!ds4_gpu_moe_selected_hotlist_load_existing(path)) return 0;
        atexit(ds4_gpu_moe_selected_hotlist_close);
    }
    if (layer >= DS4_METAL_STREAM_EXPERT_CACHE_MAX_LAYER) return 1;

    g_moe_selected_hotlist_records++;
    for (uint32_t i = 0; i < n_selected; i++) {
        if (selected_ids[i] < 0) continue;
        const uint32_t expert = (uint32_t)selected_ids[i];
        if (expert >= n_total_expert ||
            expert >= DS4_METAL_STREAM_EXPERT_CACHE_MAX_EXPERT) {
            continue;
        }
        g_moe_selected_hotlist_counts[layer][expert]++;
        g_moe_selected_hotlist_selections++;
    }
    return 1;
}

static void ds4_gpu_moe_selected_trace_record_close(void) {
    if (g_moe_selected_trace_record_fp) {
        const char *path = getenv("DS4_MOE_RECORD_SELECTED_IDS");
        fclose(g_moe_selected_trace_record_fp);
        g_moe_selected_trace_record_fp = NULL;
        fprintf(stderr,
                "ds4: recorded %" PRIu64 " routed-MoE selected-id entries to %s\n",
                g_moe_selected_trace_record_count,
                path && path[0] ? path : "(unknown)");
    }
}

static int ds4_gpu_moe_selected_trace_record(
        const int32_t selected_ids[6],
        uint32_t      n_selected) {
    const char *path = getenv("DS4_MOE_RECORD_SELECTED_IDS");
    if (!path || !path[0]) return 1;
    if (n_selected != 6) {
        fprintf(stderr, "ds4: selected-id recording expects exactly 6 selected experts\n");
        return 0;
    }

    if (!g_moe_selected_trace_record_initialized) {
        g_moe_selected_trace_record_initialized = 1;
        g_moe_selected_trace_record_fp = fopen(path, "wb");
        if (!g_moe_selected_trace_record_fp) {
            fprintf(stderr, "ds4: failed to open selected-id record file %s\n", path);
            return 0;
        }
        setvbuf(g_moe_selected_trace_record_fp, NULL, _IOFBF, 1u << 20);
        atexit(ds4_gpu_moe_selected_trace_record_close);
    }

    if (fwrite(selected_ids, sizeof(selected_ids[0]), n_selected, g_moe_selected_trace_record_fp) != n_selected) {
        fprintf(stderr, "ds4: failed to write selected-id record file %s\n", path);
        return 0;
    }
    if (fflush(g_moe_selected_trace_record_fp) != 0) {
        fprintf(stderr, "ds4: failed to flush selected-id record file %s\n", path);
        return 0;
    }
    g_moe_selected_trace_record_count++;
    return 1;
}

static int ds4_gpu_moe_selected_trace_replay(
        int32_t  selected_ids[6],
        uint32_t n_selected) {
    const char *path = getenv("DS4_MOE_REPLAY_SELECTED_IDS");
    if (!path || !path[0]) return 0;
    if (n_selected != 6) {
        fprintf(stderr, "ds4: selected-id replay expects exactly 6 selected experts\n");
        return -1;
    }

    if (!g_moe_selected_trace_replay_initialized) {
        g_moe_selected_trace_replay_initialized = 1;
        FILE *fp = fopen(path, "rb");
        if (!fp) {
            fprintf(stderr, "ds4: failed to open selected-id replay file %s\n", path);
            return -1;
        }
        if (fseeko(fp, 0, SEEK_END) != 0) {
            fprintf(stderr, "ds4: failed to seek selected-id replay file %s\n", path);
            fclose(fp);
            return -1;
        }
        const off_t end = ftello(fp);
        if (end < 0) {
            fprintf(stderr, "ds4: failed to size selected-id replay file %s\n", path);
            fclose(fp);
            return -1;
        }
        if (fseeko(fp, 0, SEEK_SET) != 0) {
            fprintf(stderr, "ds4: failed to rewind selected-id replay file %s\n", path);
            fclose(fp);
            return -1;
        }

        const uint64_t bytes = (uint64_t)end;
        const uint64_t entry_bytes = (uint64_t)n_selected * sizeof(selected_ids[0]);
        if (bytes == 0 || (bytes % entry_bytes) != 0) {
            fprintf(stderr,
                    "ds4: selected-id replay file %s has invalid size %" PRIu64 "\n",
                    path,
                    bytes);
            fclose(fp);
            return -1;
        }
        if (bytes > SIZE_MAX) {
            fprintf(stderr, "ds4: selected-id replay file %s is too large\n", path);
            fclose(fp);
            return -1;
        }
        g_moe_selected_trace_replay_count = bytes / entry_bytes;
        g_moe_selected_trace_replay_ids = malloc((size_t)bytes);
        if (!g_moe_selected_trace_replay_ids) {
            fprintf(stderr, "ds4: failed to allocate selected-id replay buffer\n");
            fclose(fp);
            return -1;
        }
        if (fread(g_moe_selected_trace_replay_ids, 1, (size_t)bytes, fp) != (size_t)bytes) {
            fprintf(stderr, "ds4: failed to read selected-id replay file %s\n", path);
            fclose(fp);
            free(g_moe_selected_trace_replay_ids);
            g_moe_selected_trace_replay_ids = NULL;
            return -1;
        }
        fclose(fp);
        fprintf(stderr,
                "ds4: loaded %" PRIu64 " routed-MoE selected-id entries from %s\n",
                g_moe_selected_trace_replay_count,
                path);
    }

    if (g_moe_selected_trace_replay_pos >= g_moe_selected_trace_replay_count) {
        fprintf(stderr,
                "ds4: selected-id replay exhausted after %" PRIu64 " entries\n",
                g_moe_selected_trace_replay_pos);
        return -1;
    }
    memcpy(selected_ids,
           g_moe_selected_trace_replay_ids + g_moe_selected_trace_replay_pos * n_selected,
           (size_t)n_selected * sizeof(selected_ids[0]));
    g_moe_selected_trace_replay_pos++;
    return 1;
}

static int ds4_gpu_progress_enabled(void) {
    return ds4_log_is_tty(stderr);
}

static void ds4_gpu_progress_begin(const char *what) {
    if (!ds4_gpu_progress_enabled()) return;
    fprintf(stderr, "ds4: %s...", what);
    fflush(stderr);
}

static void ds4_gpu_progress_done(void) {
    if (!ds4_gpu_progress_enabled()) return;
    fputs(" done\n", stderr);
    fflush(stderr);
}

static void ds4_gpu_progress_failed(void) {
    if (!ds4_gpu_progress_enabled()) return;
    fputs(" failed\n", stderr);
    fflush(stderr);
}

static void ds4_gpu_model_views_clear(void) {
    for (uint32_t i = 0; i < g_model_view_count; i++) {
        g_model_views[i].buffer = nil;
        g_model_views[i].model_map = NULL;
        g_model_views[i].model_size = 0;
        g_model_views[i].model_offset = 0;
        g_model_views[i].bytes = 0;
    }
    g_model_view_count = 0;
}

static void ds4_gpu_model_residency_clear(void) {
#if TARGET_OS_OSX
    if (@available(macOS 15.0, *)) {
        if (g_model_residency_set) {
            if (g_model_residency_added_to_queue &&
                g_queue &&
                [g_queue respondsToSelector:@selector(removeResidencySet:)]) {
                [g_queue removeResidencySet:g_model_residency_set];
            }
            [g_model_residency_set endResidency];
            [g_model_residency_set removeAllAllocations];
            g_model_residency_set = nil;
        }
    }
#endif
    g_model_residency_count = 0;
    g_model_residency_added_to_queue = 0;
}

/* TP sharding keeps only this rank's expert ranges warm,
 * so whole-view residency requests (which would page in the full file)
 * must be skipped; pages fault in lazily through the same view buffers,
 * exactly like ssd-streaming mode. */
static int g_model_residency_skipped;

void ds4_gpu_model_residency_skip(int skip) {
    g_model_residency_skipped = skip;
}

static int ds4_gpu_model_residency_request_views(void) {
    if (g_model_view_count == 0 ||
        g_ssd_streaming_mode ||
        g_model_residency_skipped ||
        getenv("DS4_METAL_NO_RESIDENCY") != NULL) {
        return 1;
    }

#if TARGET_OS_OSX
    if (@available(macOS 15.0, *)) {
        /*
         * Register all model views as one residency set before inference. This
         * is a GPU residency/budgeting hint, not a request to fault the whole
         * 80+ GB file into memory. Its purpose is to make the driver see the
         * complete set of large shared allocations during setup instead of
         * discovering them lazily from the first measured graph command, where
         * VM validation and residency accounting would look like model compute.
         */
        MTLResidencySetDescriptor *desc = [[MTLResidencySetDescriptor alloc] init];
        desc.label = @"ds4_model";
        desc.initialCapacity = g_model_view_count;

        NSError *error = nil;
        g_model_residency_set = [g_device newResidencySetWithDescriptor:desc error:&error];
        if (!g_model_residency_set) {
            fprintf(stderr, "ds4: Metal model residency set creation failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            return 0;
        }

        for (uint32_t i = 0; i < g_model_view_count; i++) {
            [g_model_residency_set addAllocation:g_model_views[i].buffer];
        }
        [g_model_residency_set commit];
        [g_model_residency_set requestResidency];
        if (getenv("DS4_METAL_DISABLE_QUEUE_RESIDENCY_SET") == NULL &&
            g_queue &&
            [g_queue respondsToSelector:@selector(addResidencySet:)]) {
            [g_queue addResidencySet:g_model_residency_set];
            g_model_residency_added_to_queue = 1;
        }
        g_model_residency_count = g_model_view_count;
    }
#endif

    return 1;
}

static int ds4_gpu_add_model_view_range(
        const void *model_map,
        uint64_t    model_size,
        uint64_t    map_offset,
        uint64_t    map_size,
        uint64_t    max_tensor_bytes,
        bool        use_default_view_cap,
        uint64_t   *mapped_model_size_out) {
    const uint64_t page = (uint64_t)getpagesize();
    const uintptr_t model_addr = (uintptr_t)model_map;

    if ((model_addr & (uintptr_t)(page - 1)) != 0) {
        fprintf(stderr, "ds4: Metal model mmap base is not page aligned\n");
        return 0;
    }
    if (map_offset > model_size || map_size > model_size - map_offset) {
        fprintf(stderr, "ds4: Metal model mapped range is outside the GGUF mapping\n");
        return 0;
    }
    const uint64_t page_model_offset = map_offset & ~(page - 1);
    const uint64_t leading = map_offset - page_model_offset;
    if (map_size > UINT64_MAX - leading ||
        leading + map_size > UINT64_MAX - (page - 1))
    {
        fprintf(stderr, "ds4: Metal model mapped range overflows page alignment\n");
        return 0;
    }
    const uint64_t mapped_model_size = round_up_u64(leading + map_size, page);
    uint64_t max_buffer = (uint64_t)[g_device maxBufferLength];
    max_buffer &= ~(page - 1);

    /*
     * Wrap only the tensor-data part of the GGUF file. Metadata is parsed by the
     * CPU and is never dereferenced by kernels, so exposing it to Metal only
     * grows the residency set and the VM range the driver must validate.
     *
     * Metal buffers have a device-specific maximum length, and this model is
     * larger than that maximum on the target machines. Creating one no-copy
     * buffer per tensor would avoid the length limit, but it would also move a
     * lot of VM-object creation and residency bookkeeping into graph setup. The
     * stable shape here is a tiny number of page-aligned views created once.
     *
     * Adjacent views intentionally overlap by more than the largest tensor, plus
     * one page for alignment. That invariant guarantees every tensor lies wholly
     * inside at least one view, so hot paths pass one buffer and one inner byte
     * offset. We never split a weight tensor across command encoders.
     */
    if (max_tensor_bytes > map_size) {
        fprintf(stderr, "ds4: Metal model max tensor span is larger than a mapped tensor span\n");
        return 0;
    }
    if (max_tensor_bytes > UINT64_MAX - (page - 1)) {
        fprintf(stderr, "ds4: Metal model max tensor span overflows page alignment\n");
        return 0;
    }
    const uint64_t max_tensor_rounded = round_up_u64(max_tensor_bytes, page);
    if (max_tensor_rounded > UINT64_MAX - page) {
        fprintf(stderr, "ds4: Metal model view overlap overflows page slack\n");
        return 0;
    }
    const uint64_t overlap = max_tensor_rounded + page;
    if (max_buffer == 0 || max_buffer <= overlap) {
        fprintf(stderr,
                "ds4: Metal maxBufferLength is too small for DS4 model views "
                "(max tensor %.2f GiB, max buffer %.2f GiB)\n",
                ds4_gpu_gib(max_tensor_bytes),
                ds4_gpu_gib(max_buffer));
        return 0;
    }

    uint64_t view_limit = max_buffer;
    const char *view_limit_env = getenv("DS4_METAL_MODEL_VIEW_MAX_GIB");
    if (view_limit_env && view_limit_env[0]) {
        char *end = NULL;
        unsigned long long gib = strtoull(view_limit_env, &end, 10);
        if (end != view_limit_env && gib > 0) {
            uint64_t env_limit = gib * 1024ull * 1024ull * 1024ull;
            env_limit &= ~(page - 1);
            if (env_limit > 0) view_limit = env_limit;
        }
    } else if (use_default_view_cap && mapped_model_size > max_buffer) {
        /*
         * Very large no-copy buffers can make Metal's VM validation dominate
         * startup or the first graph command on multi-hundred-GiB slices. Keep
         * ordinary contiguous model mappings unchanged, but let distributed
         * span maps use smaller overlapping views when a range already has to
         * be split.
         */
        const uint64_t default_limit = 128ull * 1024ull * 1024ull * 1024ull;
        if (view_limit > default_limit) view_limit = default_limit;
    }
    if (view_limit > max_buffer) view_limit = max_buffer;
    view_limit &= ~(page - 1);
    if (view_limit == 0 || view_limit <= overlap) {
        fprintf(stderr,
                "ds4: Metal model view cap is too small for DS4 model views "
                "(cap %.2f GiB, max tensor %.2f GiB)\n",
                ds4_gpu_gib(view_limit),
                ds4_gpu_gib(max_tensor_bytes));
        return 0;
    }

    const uint64_t step = view_limit - overlap;
    uint64_t off = 0;
    while (off < mapped_model_size) {
        if (g_model_view_count == DS4_METAL_MAX_MODEL_VIEWS) {
            fprintf(stderr, "ds4: Metal model needs more mapped views than expected\n");
            return 0;
        }

        uint64_t view_bytes = mapped_model_size - off;
        if (view_bytes > view_limit) view_bytes = view_limit;

        id<MTLBuffer> buffer = [g_device newBufferWithBytesNoCopy:(void *)(model_addr + page_model_offset + off)
                                                           length:(NSUInteger)view_bytes
                                                          options:ds4_gpu_model_resource_options()
                                                      deallocator:nil];
        if (!buffer) {
            fprintf(stderr,
                    "ds4: Metal could not wrap mmaped model view at %.2f GiB, size %.2f GiB\n",
                    (double)(page_model_offset + off) / (1024.0 * 1024.0 * 1024.0),
                    (double)view_bytes / (1024.0 * 1024.0 * 1024.0));
            return 0;
        }
        buffer.label = [NSString stringWithFormat:@"ds4_model_view_%u", g_model_view_count];

        g_model_views[g_model_view_count].buffer = buffer;
        g_model_views[g_model_view_count].model_map = model_map;
        g_model_views[g_model_view_count].model_size = model_size;
        g_model_views[g_model_view_count].model_offset = page_model_offset + off;
        g_model_views[g_model_view_count].bytes = view_bytes;
        g_model_view_count++;

        g_model_wrap_count++;
        g_model_wrap_bytes += view_bytes;
        if (view_bytes > g_model_wrap_max_bytes) g_model_wrap_max_bytes = view_bytes;

        if (off + view_bytes >= mapped_model_size) break;
        off += step;
    }

    if (mapped_model_size_out) *mapped_model_size_out += mapped_model_size;
    return 1;
}

static int ds4_gpu_finish_model_views(
        double t0,
        uint64_t mapped_model_size,
        uint64_t display_offset) {
    const double t_mapped = ds4_gpu_now_ms();
    const int request_residency =
        !g_ssd_streaming_mode &&
        getenv("DS4_METAL_NO_RESIDENCY") == NULL;
    if (request_residency) ds4_gpu_progress_begin("requesting Metal residency (may take tens of seconds)");
    if (!ds4_gpu_model_residency_request_views()) {
        if (request_residency) ds4_gpu_progress_failed();
        return 0;
    }
    if (request_residency) ds4_gpu_progress_done();
    const double t_resident = ds4_gpu_now_ms();
    int warmed = 1;
    const double t_warm0 = ds4_gpu_now_ms();
    const int warm_model_views = !g_ssd_streaming_mode &&
                                 getenv("DS4_METAL_NO_RESIDENCY") == NULL &&
                                 getenv("DS4_METAL_NO_MODEL_WARMUP") == NULL;
    if (warm_model_views) {
        /*
         * The first GPU command touching no-copy mmap storage can pay command
         * queue setup, page-table validation, and shared-allocation residency
         * costs. Sample each model view here so timed graph execution starts
         * after that one-time work. The stride is intentionally coarse: this is
         * a validation touch over the VM ranges, not a full model prefetch. A
         * dense prefetch would create exactly the kind of memory pressure and
         * startup stalls this path is designed to avoid.
         */
        if (g_model_residency_skipped) {
            /* TP sharding: a single command buffer binding every
             * view demands residency of them all and OOMs; the engine's
             * CPU-side sharded warm pre-faults the owned bytes instead. */
            warmed = 1;
        } else {
            ds4_gpu_progress_begin("warming Metal model views");
            warmed = ds4_gpu_warm_model_views();
            if (warmed) ds4_gpu_progress_done();
            else ds4_gpu_progress_failed();
        }
    }
    const double t_warm = ds4_gpu_now_ms();
    if (ds4_gpu_model_map_log_enabled()) {
        fprintf(stderr,
                "ds4: Metal model views created in %.3f ms, residency requested in %.3f ms, warmup %.3f ms (mapped %.2f MiB from offset %.2f MiB)\n",
                t_mapped - t0,
                t_resident - t_mapped,
                t_warm - t_warm0,
                mapped_model_size / 1024.0 / 1024.0,
                display_offset / 1024.0 / 1024.0);
    }
    if (!warmed) return 0;
    return 1;
}

static int ds4_gpu_map_model_views(
        const void *model_map,
        uint64_t    model_size,
        uint64_t    map_offset,
        uint64_t    map_size,
        uint64_t    max_tensor_bytes) {
    const double t0 = ds4_gpu_now_ms();
    uint64_t mapped_model_size = 0;
    if (!ds4_gpu_add_model_view_range(model_map,
                                      model_size,
                                      map_offset,
                                      map_size,
                                      max_tensor_bytes,
                                      false,
                                      &mapped_model_size)) {
        return 0;
    }
    return ds4_gpu_finish_model_views(t0, mapped_model_size, map_offset);
}

static id<MTLBuffer> ds4_gpu_new_transient_buffer(NSUInteger bytes, const char *label) {
    if (bytes == 0) bytes = 1;

    id<MTLBuffer> buffer = [g_device newBufferWithLength:bytes
                                                 options:MTLResourceStorageModeShared];
    if (!buffer) {
        fprintf(stderr, "ds4: failed to allocate Metal transient buffer %s (%llu bytes)\n",
                label ? label : "(unnamed)", (unsigned long long)bytes);
        return nil;
    }
    if (label) buffer.label = [NSString stringWithUTF8String:label];

    /*
     * CPU-filled buffers must survive until their command buffer completes.
     * A local ObjC strong variable is not enough when the encoder function
     * returns before the caller commits the command buffer.
     */
    [g_transient_buffers addObject:buffer];
    return buffer;
}

static int ds4_gpu_zero_prefix_prefill_mask_cache_enabled(void) {
    if (getenv("DS4_METAL_DISABLE_M3_ZERO_PREFIX_PREFILL_MASK_CACHE") != NULL ||
        getenv("DS4_METAL_FLASH_ATTN_STAGE_PROFILE") != NULL) {
        return 0;
    }
    return ds4_gpu_device_name_contains("M3") ||
           getenv("DS4_METAL_ENABLE_ZERO_PREFIX_PREFILL_MASK_CACHE") != NULL;
}

static ds4_gpu_zero_prefix_prefill_mask_cache_entry *
ds4_gpu_get_zero_prefix_prefill_mask_cache(
        uint32_t   kind,
        uint32_t   n_tokens,
        uint32_t   n_comp,
        uint32_t   n_keys,
        uint32_t   window,
        uint32_t   ratio,
        uint32_t   nqptg,
        uint32_t   ncpsg,
        bool       has_kvpad,
        bool       bc_mask,
        NSUInteger mask_bytes,
        NSUInteger blk_bytes,
        bool      *created) {
    if (created) *created = false;
    if (!created || !ds4_gpu_zero_prefix_prefill_mask_cache_enabled() ||
        mask_bytes == 0 || blk_bytes == 0) {
        return NULL;
    }

    uint32_t slot = UINT32_MAX;
    switch (kind) {
        case DS4_GPU_PREFILL_MASK_CACHE_RAW:      slot = 0; break;
        case DS4_GPU_PREFILL_MASK_CACHE_RATIO4:   slot = 1; break;
        case DS4_GPU_PREFILL_MASK_CACHE_RATIO128: slot = 2; break;
        default: return NULL;
    }

    ds4_gpu_zero_prefix_prefill_mask_cache_entry *entry =
        &g_zero_prefix_prefill_mask_cache[slot];
    if (entry->valid && entry->mask && entry->blk &&
        entry->mask_bytes == mask_bytes && entry->blk_bytes == blk_bytes &&
        entry->kind == kind && entry->n_tokens == n_tokens &&
        entry->n_comp == n_comp && entry->n_keys == n_keys &&
        entry->window == window && entry->ratio == ratio &&
        entry->nqptg == nqptg && entry->ncpsg == ncpsg &&
        entry->has_kvpad == has_kvpad && entry->bc_mask == bc_mask) {
        return entry;
    }

    id<MTLBuffer> mask = [g_device newBufferWithLength:mask_bytes
                                                options:MTLResourceStorageModeShared];
    id<MTLBuffer> blk = [g_device newBufferWithLength:blk_bytes
                                               options:MTLResourceStorageModePrivate];
    if (!blk) {
        blk = [g_device newBufferWithLength:blk_bytes
                                    options:MTLResourceStorageModeShared];
    }
    if (!mask || !blk) return NULL;

    mask.label = [NSString stringWithFormat:@"ds4_prefill_mask_cache_%u", kind];
    blk.label = [NSString stringWithFormat:@"ds4_prefill_blk_cache_%u", kind];

    /* A replaced entry may still be referenced by an uncommitted or in-flight
     * command buffer. Keep its resources alive with the other batch-scoped
     * buffers until command completion instead of mutating or releasing them. */
    if (entry->mask) [g_transient_buffers addObject:entry->mask];
    if (entry->blk) [g_transient_buffers addObject:entry->blk];

    entry->mask = mask;
    entry->blk = blk;
    entry->mask_bytes = mask_bytes;
    entry->blk_bytes = blk_bytes;
    entry->kind = kind;
    entry->n_tokens = n_tokens;
    entry->n_comp = n_comp;
    entry->n_keys = n_keys;
    entry->window = window;
    entry->ratio = ratio;
    entry->nqptg = nqptg;
    entry->ncpsg = ncpsg;
    entry->has_kvpad = has_kvpad;
    entry->bc_mask = bc_mask;
    /* The caller publishes the entry only after synchronously filling mask. */
    entry->valid = false;
    entry->blk_ready = false;
    *created = true;
    return entry;
}

static void ds4_gpu_invalidate_zero_prefix_prefill_block_maps(void) {
    for (uint32_t i = 0; i < DS4_GPU_PREFILL_MASK_CACHE_SLOTS; i++) {
        g_zero_prefix_prefill_mask_cache[i].blk_ready = false;
    }
}

static void ds4_gpu_clear_zero_prefix_prefill_mask_cache(void) {
    for (uint32_t i = 0; i < DS4_GPU_PREFILL_MASK_CACHE_SLOTS; i++) {
        ds4_gpu_zero_prefix_prefill_mask_cache_entry *entry =
            &g_zero_prefix_prefill_mask_cache[i];
        entry->mask = nil;
        entry->blk = nil;
        entry->mask_bytes = 0;
        entry->blk_bytes = 0;
        entry->kind = 0;
        entry->n_tokens = 0;
        entry->n_comp = 0;
        entry->n_keys = 0;
        entry->window = 0;
        entry->ratio = 0;
        entry->nqptg = 0;
        entry->ncpsg = 0;
        entry->has_kvpad = false;
        entry->bc_mask = false;
        entry->valid = false;
        entry->blk_ready = false;
    }
}

void ds4_gpu_release_zero_prefix_prefill_mask_cache(void) {
    /* Layer-major prefill waits each layer before advancing, so its final
     * release point has no outstanding cache users. Keep the guard here for
     * diagnostic callers that may have an open or asynchronously flushed CB. */
    if (!g_initialized || g_batch_cb ||
        (g_pending_cbs && [g_pending_cbs count] != 0)) {
        return;
    }
    ds4_gpu_clear_zero_prefix_prefill_mask_cache();
}

static id<MTLComputePipelineState> ds4_gpu_get_mul_mm_pipeline(
        const char *function_name,
        bool        bc_inp,
        bool        bc_out) {
    NSString *key = [NSString stringWithFormat:@"%s_bci=%d_bco=%d",
                     function_name, bc_inp ? 1 : 0, bc_out ? 1 : 0];
    id<MTLComputePipelineState> cached = [g_pipeline_cache objectForKey:key];
    if (cached) return cached;

    MTLFunctionConstantValues *constants = [[MTLFunctionConstantValues alloc] init];
    [constants setConstantValue:&bc_inp type:MTLDataTypeBool atIndex:700];
    [constants setConstantValue:&bc_out type:MTLDataTypeBool atIndex:701];

    NSError *error = nil;
    NSString *name = [NSString stringWithUTF8String:function_name];
    id<MTLFunction> fn = [g_library newFunctionWithName:name
                                         constantValues:constants
                                                  error:&error];
    if (!fn) {
        fprintf(stderr, "ds4: Metal %s function not found: %s\n",
                function_name, [[error localizedDescription] UTF8String]);
        return nil;
    }

    error = nil;
    id<MTLComputePipelineState> pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!pipeline) {
        fprintf(stderr, "ds4: Metal %s pipeline failed: %s\n",
                function_name, [[error localizedDescription] UTF8String]);
        return nil;
    }

    [g_pipeline_cache setObject:pipeline forKey:key];
    return pipeline;
}

static id<MTLComputePipelineState> ds4_gpu_get_mul_mm_id_pipeline(
        const char *function_name,
        bool        bc_inp) {
    NSString *key = [NSString stringWithFormat:@"%s_bci=%d",
                     function_name, bc_inp ? 1 : 0];
    id<MTLComputePipelineState> cached = [g_pipeline_cache objectForKey:key];
    if (cached) return cached;

    MTLFunctionConstantValues *constants = [[MTLFunctionConstantValues alloc] init];
    [constants setConstantValue:&bc_inp type:MTLDataTypeBool atIndex:700];

    NSError *error = nil;
    NSString *name = [NSString stringWithUTF8String:function_name];
    id<MTLFunction> fn = [g_library newFunctionWithName:name
                                         constantValues:constants
                                                  error:&error];
    if (!fn) {
        fprintf(stderr, "ds4: Metal %s function not found: %s\n",
                function_name, [[error localizedDescription] UTF8String]);
        return nil;
    }

    error = nil;
    id<MTLComputePipelineState> pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!pipeline) {
        fprintf(stderr, "ds4: Metal %s pipeline failed: %s\n",
                function_name, [[error localizedDescription] UTF8String]);
        return nil;
    }

    [g_pipeline_cache setObject:pipeline forKey:key];
    return pipeline;
}

static id<MTLComputePipelineState> ds4_gpu_get_pipeline(
        const char *function_name) {
    NSString *key = [NSString stringWithFormat:@"%s", function_name];
    id<MTLComputePipelineState> cached = [g_pipeline_cache objectForKey:key];
    if (cached) return cached;

    NSError *error = nil;
    NSString *name = [NSString stringWithUTF8String:function_name];
    id<MTLFunction> fn = [g_library newFunctionWithName:name];
    if (!fn) {
        fprintf(stderr, "ds4: Metal %s function not found\n", function_name);
        return nil;
    }

    id<MTLComputePipelineState> pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!pipeline) {
        fprintf(stderr, "ds4: Metal %s pipeline failed: %s\n",
                function_name, [[error localizedDescription] UTF8String]);
        return nil;
    }

    [g_pipeline_cache setObject:pipeline forKey:key];
    return pipeline;
}

static int ds4_gpu_disable_hot_pipeline_statics(void) {
    static int initialized;
    static int disabled;
    if (!initialized) {
        disabled = getenv("DS4_METAL_DISABLE_HOT_PIPELINE_STATICS") != NULL;
        initialized = 1;
    }
    return disabled;
}

static id<MTLComputePipelineState> ds4_gpu_hot_pipeline(
        id<MTLComputePipelineState> pipeline,
        const char *fallback_name) {
    if (!ds4_gpu_disable_hot_pipeline_statics()) return pipeline;
    return ds4_gpu_get_pipeline(fallback_name);
}

static int ds4_gpu_use_compressor_pair_nr4(void) {
    static int initialized;
    static int enabled;
    if (!initialized) {
        enabled = getenv("DS4_METAL_COMPRESSOR_PAIR_NR4") != NULL;
        initialized = 1;
    }
    return enabled;
}

static int ds4_gpu_device_name_contains(const char *needle);

static int ds4_gpu_env_value_eq(const char *v, size_t n, const char *literal) {
    size_t m = strlen(literal);
    if (n != m) return 0;
    for (size_t i = 0; i < n; i++) {
        if (tolower((unsigned char)v[i]) != tolower((unsigned char)literal[i])) return 0;
    }
    return 1;
}

static int ds4_gpu_env_bool(const char *name) {
    const char *v = getenv(name);
    if (!v) return -1;

    while (isspace((unsigned char)*v)) v++;
    size_t n = strlen(v);
    while (n > 0 && isspace((unsigned char)v[n - 1])) n--;
    if (n == 0) return 1;

    if (ds4_gpu_env_value_eq(v, n, "1") ||
        ds4_gpu_env_value_eq(v, n, "true") ||
        ds4_gpu_env_value_eq(v, n, "yes") ||
        ds4_gpu_env_value_eq(v, n, "on")) {
        return 1;
    }
    if (ds4_gpu_env_value_eq(v, n, "0") ||
        ds4_gpu_env_value_eq(v, n, "false") ||
        ds4_gpu_env_value_eq(v, n, "no") ||
        ds4_gpu_env_value_eq(v, n, "off")) {
        return 0;
    }

    if (!g_mpp_invalid_env_reported) {
        fprintf(stderr,
                "ds4: invalid Metal boolean environment value %s=%.*s; treating presence as enabled\n",
                name, (int)n, v);
        g_mpp_invalid_env_reported = 1;
    }
    return 1;
}

static uint64_t ds4_gpu_env_u64(const char *name,
                                uint64_t    fallback,
                                uint64_t    min_value,
                                uint64_t    max_value) {
    const char *v = getenv(name);
    if (!v) return fallback;
    while (isspace((unsigned char)*v)) v++;
    if (!*v) return fallback;

    errno = 0;
    char *end = NULL;
    unsigned long long parsed = strtoull(v, &end, 10);
    if (end == v || errno == ERANGE) return fallback;
    while (isspace((unsigned char)*end)) end++;
    if (*end) return fallback;

    if (parsed < min_value) return fallback;
    uint64_t value = (uint64_t)parsed;
    if (value > max_value) value = max_value;
    return value;
}

static uint32_t ds4_gpu_glm_full_attention_max_cache_len(void) {
    /*
     * kernel_glm_attention_full stores one score per visible token in
     * threadgroup memory, plus 256 reduction slots.  Keep the default under
     * the 32 KiB envelope used by current Apple GPUs while allowing short
     * dense decode to step past the 4096-token prefill boundary.
     */
    return 7680u;
}

static uint32_t ds4_gpu_glm_flash_attention_max_cache_len(void) {
    /*
     * Staged FlashAttention uses fixed-size threadgroup scratch and separate
     * KV staging buffers, so it is not bound by the legacy full-attention
     * kernel's one-score-per-token threadgroup-memory envelope.
     */
    return 8192u;
}

static int ds4_gpu_mpp_available(void) {
    return g_metal4_tensor_api_enabled && !g_quality_mode;
}

/*
 * Retained Metal4 defaults live here instead of behind user-visible options.
 * The public runtime has one automatic accelerated path plus the global
 * DS4_METAL_DISABLE_METAL4 comparison switch.  Benchmark-only alternatives that
 * lost during M5 work are removed or kept out of the dispatch path so future
 * changes do not accidentally turn old experiments into new modes.
 */
static int ds4_gpu_use_mpp_attn_out_low_matmul(void) {
    return ds4_gpu_mpp_available();
}

enum {
    DS4_METAL_ATTN_OUT_MPP_TILE_N = 64,
};

static void ds4_gpu_warn_mpp_fallback(void) {
    static int warned;
    if (!warned) {
        fprintf(stderr, "ds4: accelerated Metal prefill matmul unavailable; falling back to legacy kernel\n");
        warned = 1;
    }
}

static int ds4_gpu_device_name_contains(const char *needle) {
    return g_metal_device_name[0] != '\0' && strstr(g_metal_device_name, needle) != NULL;
}

static int ds4_gpu_compile_tensor_probe(void) {
#if defined(__MAC_OS_X_VERSION_MAX_ALLOWED) && __MAC_OS_X_VERSION_MAX_ALLOWED >= 260000
    if (!g_device) return 0;
    if (@available(macOS 26.0, *)) {
        const char *src =
            "#include <metal_stdlib>\n"
            "#include <metal_tensor>\n"
            "#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n"
            "using namespace metal;\n"
            "using namespace mpp::tensor_ops;\n"
            "kernel void ds4_tensor_probe(\n"
            "        tensor<device half,  dextents<int32_t, 2>> A [[buffer(0)]],\n"
            "        tensor<device half,  dextents<int32_t, 2>> B [[buffer(1)]],\n"
            "        device float *C [[buffer(2)]],\n"
            "        uint2 tgid [[threadgroup_position_in_grid]]) {\n"
            "    auto tA = A.slice(0, (int)tgid.y);\n"
            "    auto tB = B.slice((int)tgid.x, 0);\n"
            "    matmul2d<matmul2d_descriptor(16, 16, dynamic_extent), execution_simdgroups<4>> mm;\n"
            "    auto cT = mm.get_destination_cooperative_tensor<decltype(tA), decltype(tB), float>();\n"
            "    auto sA = tA.slice(0, 0);\n"
            "    auto sB = tB.slice(0, 0);\n"
            "    mm.run(sB, sA, cT);\n"
            "    auto tC = tensor<device float, dextents<int32_t, 2>, tensor_inline>(C, dextents<int32_t, 2>(16, 16));\n"
            "    cT.store(tC);\n"
            "}\n";

        NSError *error = nil;
        NSString *source = [NSString stringWithUTF8String:src];
        id<MTLLibrary> probe_library = [g_device newLibraryWithSource:source options:[MTLCompileOptions new] error:&error];
        if (!probe_library) {
            fprintf(stderr, "ds4: Metal 4 tensor API probe compile failed: %s\n",
                    error ? [[error localizedDescription] UTF8String] : "(unknown)");
            return 0;
        }
        id<MTLFunction> fn = [probe_library newFunctionWithName:@"ds4_tensor_probe"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal 4 tensor API probe function missing\n");
            return 0;
        }
        error = nil;
        id<MTLComputePipelineState> pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!pipeline) {
            fprintf(stderr, "ds4: Metal 4 tensor API probe pipeline failed: %s\n",
                    error ? [[error localizedDescription] UTF8String] : "(unknown)");
            return 0;
        }
        return 1;
    }
#endif
    return 0;
}

static void ds4_gpu_detect_metal4_features(void) {
    g_metal4_runtime_available = 0;
    g_metal4_family_supported = 0;
    g_metal4_queue_supported = 0;
    g_metal4_m5_neural_accelerators_hint = 0;
    g_metal4_tensor_api_enabled = 0;
    g_metal4_tensor_api_compile_supported = 0;
    g_metal_device_name[0] = '\0';

    if (!g_device) return;

    const char *name = [[g_device name] UTF8String];
    if (name) {
        snprintf(g_metal_device_name, sizeof(g_metal_device_name), "%s", name);
    }

    const int metal4_disabled = ds4_gpu_env_bool("DS4_METAL_DISABLE_METAL4") > 0;

#if defined(__MAC_OS_X_VERSION_MAX_ALLOWED) && __MAC_OS_X_VERSION_MAX_ALLOWED >= 260000
    if (@available(macOS 26.0, *)) {
        g_metal4_runtime_available = 1;
        g_metal4_family_supported =
            !metal4_disabled && [g_device supportsFamily:MTLGPUFamilyMetal4] ? 1 : 0;
        g_metal4_queue_supported = [g_device respondsToSelector:@selector(newMTL4CommandQueue)] ? 1 : 0;

        /*
         * Apple does not currently expose a separate "Neural Accelerator" bit
         * through Metal. On public M5 systems the hardware signal is the device
         * generation plus Metal 4 support, so keep this as a conservative hint.
         */
        if (g_metal4_family_supported && ds4_gpu_device_name_contains("M5")) {
            g_metal4_m5_neural_accelerators_hint = 1;
        }

        if (g_metal4_family_supported) {
            const int default_enable =
                ds4_gpu_device_name_contains("M5") ||
                ds4_gpu_device_name_contains("M6") ||
                ds4_gpu_device_name_contains("A19") ||
                ds4_gpu_device_name_contains("A20");

            /*
             * Metal 4 TensorOps are portable in source, but on pre-M5 hardware
             * they can map to ordinary shader fallbacks.  Keep the automatic
             * fast path restricted to hardware generations where the Neural
             * Accelerator/TensorOps path is expected to pay off; older Metal
             * machines continue to use the established kernels unless a future
             * device is explicitly added here.
             */
            if (default_enable) {
                g_metal4_tensor_api_compile_supported = ds4_gpu_compile_tensor_probe();
                g_metal4_tensor_api_enabled = g_metal4_tensor_api_compile_supported;
                if (!g_metal4_tensor_api_enabled) {
                    fprintf(stderr, "ds4: Metal 4 tensor API probe failed; using legacy Metal kernels\n");
                }
            } else {
                fprintf(stderr, "ds4: Metal 4 tensor API disabled for pre-M5/pre-A19 devices\n");
            }
        }
    }
#endif
}

static int ds4_gpu_warm_model_views(void) {
    if (g_model_view_count == 0) return 1;

    id<MTLComputePipelineState> pipeline = ds4_gpu_get_pipeline("kernel_touch_u8_stride");
    if (!pipeline) return 0;

    uint64_t stride = 1024ull * 1024ull;
    const char *stride_env = getenv("DS4_METAL_MODEL_WARMUP_STRIDE_MB");
    if (stride_env && stride_env[0]) {
        char *end = NULL;
        unsigned long long mb = strtoull(stride_env, &end, 10);
        if (end != stride_env && mb > 0 && mb <= 1024) {
            stride = mb * 1024ull * 1024ull;
        }
    }
    const char *stride_kb_env = getenv("DS4_METAL_MODEL_WARMUP_STRIDE_KB");
    if (stride_kb_env && stride_kb_env[0]) {
        char *end = NULL;
        unsigned long long kb = strtoull(stride_kb_env, &end, 10);
        if (end != stride_kb_env && kb > 0 && kb <= 1024ull * 1024ull) {
            stride = kb * 1024ull;
            const uint64_t page = (uint64_t)getpagesize();
            if (stride < page) stride = page;
        }
    }

    uint64_t total_touches = 0;
    for (uint32_t i = 0; i < g_model_view_count; i++) {
        total_touches += (g_model_views[i].bytes + stride - 1) / stride;
    }
    if (total_touches == 0 || total_touches > (uint64_t)NSUIntegerMax) return 0;

    const NSUInteger out_bytes = (NSUInteger)total_touches;
    id<MTLBuffer> out = [g_device newBufferWithLength:out_bytes
                                             options:MTLResourceStorageModeShared];
    if (!out) {
        fprintf(stderr, "ds4: Metal model warmup scratch allocation failed\n");
        return 0;
    }
    out.label = @"ds4_model_warmup";

    id<MTLCommandBuffer> cb = ds4_gpu_new_command_buffer();
    if (!cb) {
        fprintf(stderr, "ds4: Metal model warmup command buffer allocation failed\n");
        return 0;
    }

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:pipeline];
    uint64_t dst_offset = 0;
    for (uint32_t i = 0; i < g_model_view_count; i++) {
        const uint64_t bytes = g_model_views[i].bytes;
        const uint64_t n = (bytes + stride - 1) / stride;
        [enc setBuffer:g_model_views[i].buffer offset:0 atIndex:0];
        [enc setBuffer:out offset:0 atIndex:1];
        [enc setBytes:&stride length:sizeof(stride) atIndex:2];
        [enc setBytes:&bytes length:sizeof(bytes) atIndex:3];
        [enc setBytes:&dst_offset length:sizeof(dst_offset) atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)((n + 255) / 256), 1, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        dst_offset += n;
    }
    ds4_gpu_end_compute_encoder(cb, enc);

    [cb commit];
    [cb waitUntilCompleted];

    if (cb.status == MTLCommandBufferStatusError) {
        fprintf(stderr, "ds4: Metal model warmup failed: %s\n",
                [[cb.error localizedDescription] UTF8String]);
        return 0;
    }

    return 1;
}

static const char *ds4_gpu_mul_mm_id_map0_name(uint32_t ne20) {
    switch (ne20) {
        case 1:  return "kernel_mul_mm_id_map0_ne20_1";
        case 2:  return "kernel_mul_mm_id_map0_ne20_2";
        case 4:  return "kernel_mul_mm_id_map0_ne20_4";
        case 5:  return "kernel_mul_mm_id_map0_ne20_5";
        case 6:  return "kernel_mul_mm_id_map0_ne20_6";
        case 8:  return "kernel_mul_mm_id_map0_ne20_8";
        case 10: return "kernel_mul_mm_id_map0_ne20_10";
        case 16: return "kernel_mul_mm_id_map0_ne20_16";
        case 22: return "kernel_mul_mm_id_map0_ne20_22";
        default: return NULL;
    }
}

static id<MTLComputePipelineState> ds4_gpu_get_mul_mv_pipeline(
        const char *function_name,
        int16_t     nsg) {
    NSString *key = [NSString stringWithFormat:@"%s_nsg=%d", function_name, (int)nsg];
    id<MTLComputePipelineState> cached = [g_pipeline_cache objectForKey:key];
    if (cached) return cached;

    MTLFunctionConstantValues *constants = [[MTLFunctionConstantValues alloc] init];
    [constants setConstantValue:&nsg type:MTLDataTypeShort atIndex:600];

    NSError *error = nil;
    NSString *name = [NSString stringWithUTF8String:function_name];
    id<MTLFunction> fn = [g_library newFunctionWithName:name
                                         constantValues:constants
                                                  error:&error];
    if (!fn) {
        fprintf(stderr, "ds4: Metal %s function not found: %s\n",
                function_name, [[error localizedDescription] UTF8String]);
        return nil;
    }

    error = nil;
    id<MTLComputePipelineState> pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!pipeline) {
        fprintf(stderr, "ds4: Metal %s pipeline failed: %s\n",
                function_name, [[error localizedDescription] UTF8String]);
        return nil;
    }

    [g_pipeline_cache setObject:pipeline forKey:key];
    return pipeline;
}

static id<MTLComputePipelineState> ds4_gpu_get_mul_mv_ext_pipeline(
        const char *function_name,
        int16_t     nsg,
        int16_t     nxpsg) {
    NSString *key = [NSString stringWithFormat:@"%s_nsg=%d_nxpsg=%d",
                     function_name, (int)nsg, (int)nxpsg];
    id<MTLComputePipelineState> cached = [g_pipeline_cache objectForKey:key];
    if (cached) return cached;

    MTLFunctionConstantValues *constants = [[MTLFunctionConstantValues alloc] init];
    [constants setConstantValue:&nsg   type:MTLDataTypeShort atIndex:600];
    [constants setConstantValue:&nxpsg type:MTLDataTypeShort atIndex:601];

    NSError *error = nil;
    NSString *name = [NSString stringWithUTF8String:function_name];
    id<MTLFunction> fn = [g_library newFunctionWithName:name
                                         constantValues:constants
                                                  error:&error];
    if (!fn) {
        fprintf(stderr, "ds4: Metal %s function not found: %s\n",
                function_name, [[error localizedDescription] UTF8String]);
        return nil;
    }

    error = nil;
    id<MTLComputePipelineState> pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!pipeline) {
        fprintf(stderr, "ds4: Metal %s pipeline failed: %s\n",
                function_name, [[error localizedDescription] UTF8String]);
        return nil;
    }

    [g_pipeline_cache setObject:pipeline forKey:key];
    return pipeline;
}

static id<MTLComputePipelineState> ds4_gpu_get_flash_attn_pad_pipeline(
        bool    has_mask,
        int32_t ncpsg) {
    NSString *key = [NSString stringWithFormat:@"kernel_flash_attn_ext_pad_mask=%d_ncpsg=%d",
                     has_mask ? 1 : 0, (int)ncpsg];
    id<MTLComputePipelineState> cached = [g_pipeline_cache objectForKey:key];
    if (cached) return cached;

    MTLFunctionConstantValues *constants = [[MTLFunctionConstantValues alloc] init];
    [constants setConstantValue:&has_mask type:MTLDataTypeBool atIndex:100];
    [constants setConstantValue:&ncpsg type:MTLDataTypeInt atIndex:125];

    NSError *error = nil;
    id<MTLFunction> fn = [g_library newFunctionWithName:@"kernel_flash_attn_ext_pad"
                                         constantValues:constants
                                                  error:&error];
    if (!fn) {
        fprintf(stderr, "ds4: Metal kernel_flash_attn_ext_pad function not found: %s\n",
                [[error localizedDescription] UTF8String]);
        return nil;
    }

    error = nil;
    id<MTLComputePipelineState> pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!pipeline) {
        fprintf(stderr, "ds4: Metal kernel_flash_attn_ext_pad pipeline failed: %s\n",
                [[error localizedDescription] UTF8String]);
        return nil;
    }

    [g_pipeline_cache setObject:pipeline forKey:key];
    return pipeline;
}

static id<MTLComputePipelineState> ds4_gpu_get_flash_attn_blk_pipeline(
        int32_t nqptg,
        int32_t ncpsg) {
    NSString *key = [NSString stringWithFormat:@"kernel_flash_attn_ext_blk_nqptg=%d_ncpsg=%d",
                     (int)nqptg, (int)ncpsg];
    id<MTLComputePipelineState> cached = [g_pipeline_cache objectForKey:key];
    if (cached) return cached;

    MTLFunctionConstantValues *constants = [[MTLFunctionConstantValues alloc] init];
    [constants setConstantValue:&nqptg type:MTLDataTypeInt atIndex:224];
    [constants setConstantValue:&ncpsg type:MTLDataTypeInt atIndex:225];

    NSError *error = nil;
    id<MTLFunction> fn = [g_library newFunctionWithName:@"kernel_flash_attn_ext_blk"
                                         constantValues:constants
                                                  error:&error];
    if (!fn) {
        fprintf(stderr, "ds4: Metal kernel_flash_attn_ext_blk function not found: %s\n",
                [[error localizedDescription] UTF8String]);
        return nil;
    }

    error = nil;
    id<MTLComputePipelineState> pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!pipeline) {
        fprintf(stderr, "ds4: Metal kernel_flash_attn_ext_blk pipeline failed: %s\n",
                [[error localizedDescription] UTF8String]);
        return nil;
    }

    [g_pipeline_cache setObject:pipeline forKey:key];
    return pipeline;
}

static id<MTLComputePipelineState> ds4_gpu_get_flash_attn_pipeline(
        const char *function_name,
        bool        has_mask,
        bool        has_sinks,
        bool        has_bias,
        bool        has_scap,
        bool        has_kvpad,
        bool        bc_mask,
        int32_t     ns10,
        int32_t     ns20,
        int32_t     nsg) {
    NSString *key = [NSString stringWithFormat:@"%s_mask=%d_sinks=%d_bias=%d_scap=%d_kvpad=%d_bcm=%d_ns10=%d_ns20=%d_nsg=%d",
                     function_name,
                     has_mask ? 1 : 0,
                     has_sinks ? 1 : 0,
                     has_bias ? 1 : 0,
                     has_scap ? 1 : 0,
                     has_kvpad ? 1 : 0,
                     bc_mask ? 1 : 0,
                     (int)ns10,
                     (int)ns20,
                     (int)nsg];
    id<MTLComputePipelineState> cached = [g_pipeline_cache objectForKey:key];
    if (cached) return cached;

    MTLFunctionConstantValues *constants = [[MTLFunctionConstantValues alloc] init];
    [constants setConstantValue:&has_mask  type:MTLDataTypeBool atIndex:300];
    [constants setConstantValue:&has_sinks type:MTLDataTypeBool atIndex:301];
    [constants setConstantValue:&has_bias  type:MTLDataTypeBool atIndex:302];
    [constants setConstantValue:&has_scap  type:MTLDataTypeBool atIndex:303];
    [constants setConstantValue:&has_kvpad type:MTLDataTypeBool atIndex:304];
    [constants setConstantValue:&bc_mask   type:MTLDataTypeBool atIndex:310];
    [constants setConstantValue:&ns10 type:MTLDataTypeInt atIndex:320];
    [constants setConstantValue:&ns20 type:MTLDataTypeInt atIndex:321];
    [constants setConstantValue:&nsg  type:MTLDataTypeInt atIndex:322];

    NSError *error = nil;
    NSString *name = [NSString stringWithUTF8String:function_name];
    id<MTLFunction> fn = [g_library newFunctionWithName:name
                                         constantValues:constants
                                                  error:&error];
    if (!fn) {
        fprintf(stderr, "ds4: Metal %s function not found: %s\n",
                function_name, [[error localizedDescription] UTF8String]);
        return nil;
    }

    error = nil;
    id<MTLComputePipelineState> pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!pipeline) {
        fprintf(stderr, "ds4: Metal %s pipeline failed: %s\n",
                function_name, [[error localizedDescription] UTF8String]);
        return nil;
    }

    [g_pipeline_cache setObject:pipeline forKey:key];
    return pipeline;
}

static id<MTLComputePipelineState> ds4_gpu_get_flash_attn_vec_pipeline(
        const char *function_name,
        bool        has_mask,
        bool        has_sinks,
        bool        has_bias,
        bool        has_scap,
        bool        has_kvpad,
        bool        shared_kvpad,
        int32_t     ns10,
        int32_t     ns20,
        int32_t     nsg,
        int32_t     nwg) {
    /*
     * Decode calls this once per layer with identical arguments, so memoize
     * the last hit and skip the NSString key + dictionary lookup on the hot
     * path.  The generic cache below remains the fallback for new variants.
     */
    static struct {
        const char *fn;
        bool m, s, b, c, k, sp;
        int32_t n10, n20, sg, wg;
        id<MTLComputePipelineState> pipeline;
    } memo;
    if (memo.pipeline && memo.fn != NULL && strcmp(memo.fn, function_name) == 0 &&
        memo.m == has_mask && memo.s == has_sinks && memo.b == has_bias &&
        memo.c == has_scap && memo.k == has_kvpad && memo.sp == shared_kvpad &&
        memo.n10 == ns10 && memo.n20 == ns20 && memo.sg == nsg && memo.wg == nwg) {
        return memo.pipeline;
    }

    NSString *key = [NSString stringWithFormat:@"%s_mask=%d_sinks=%d_bias=%d_scap=%d_kvpad=%d_sharedpad=%d_ns10=%d_ns20=%d_nsg=%d_nwg=%d",
                     function_name,
                     has_mask ? 1 : 0,
                     has_sinks ? 1 : 0,
                     has_bias ? 1 : 0,
                     has_scap ? 1 : 0,
                     has_kvpad ? 1 : 0,
                     shared_kvpad ? 1 : 0,
                     (int)ns10,
                     (int)ns20,
                     (int)nsg,
                     (int)nwg];
    id<MTLComputePipelineState> cached = [g_pipeline_cache objectForKey:key];
    if (cached) {
        memo = (typeof(memo)){ function_name, has_mask, has_sinks, has_bias,
                               has_scap, has_kvpad, shared_kvpad, ns10, ns20,
                               nsg, nwg, cached };
        return cached;
    }

    MTLFunctionConstantValues *constants = [[MTLFunctionConstantValues alloc] init];
    [constants setConstantValue:&has_mask  type:MTLDataTypeBool atIndex:400];
    [constants setConstantValue:&has_sinks type:MTLDataTypeBool atIndex:401];
    [constants setConstantValue:&has_bias  type:MTLDataTypeBool atIndex:402];
    [constants setConstantValue:&has_scap  type:MTLDataTypeBool atIndex:403];
    [constants setConstantValue:&has_kvpad type:MTLDataTypeBool atIndex:404];
    [constants setConstantValue:&shared_kvpad type:MTLDataTypeBool atIndex:405];
    [constants setConstantValue:&ns10 type:MTLDataTypeInt atIndex:420];
    [constants setConstantValue:&ns20 type:MTLDataTypeInt atIndex:421];
    [constants setConstantValue:&nsg  type:MTLDataTypeInt atIndex:422];
    [constants setConstantValue:&nwg  type:MTLDataTypeInt atIndex:423];

    NSError *error = nil;
    NSString *name = [NSString stringWithUTF8String:function_name];
    id<MTLFunction> fn = [g_library newFunctionWithName:name
                                         constantValues:constants
                                                  error:&error];
    if (!fn) {
        fprintf(stderr, "ds4: Metal %s function not found: %s\n",
                function_name, [[error localizedDescription] UTF8String]);
        return nil;
    }

    error = nil;
    id<MTLComputePipelineState> pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!pipeline) {
        fprintf(stderr, "ds4: Metal %s pipeline failed: %s\n",
                function_name, [[error localizedDescription] UTF8String]);
        return nil;
    }

    [g_pipeline_cache setObject:pipeline forKey:key];
    memo = (typeof(memo)){ function_name, has_mask, has_sinks, has_bias,
                           has_scap, has_kvpad, shared_kvpad, ns10, ns20,
                           nsg, nwg, pipeline };
    return pipeline;
}

static id<MTLComputePipelineState> ds4_gpu_get_flash_attn_reduce_pipeline(
        int32_t dv,
        int32_t nwg) {
    /* Same per-layer memo pattern as the vec getter above. */
    static int32_t memo_dv, memo_nwg;
    static id<MTLComputePipelineState> memo_pipeline;
    if (memo_pipeline && memo_dv == dv && memo_nwg == nwg) {
        return memo_pipeline;
    }

    NSString *key = [NSString stringWithFormat:@"kernel_flash_attn_ext_vec_reduce_dv=%d_nwg=%d",
                     (int)dv, (int)nwg];
    id<MTLComputePipelineState> cached = [g_pipeline_cache objectForKey:key];
    if (cached) {
        memo_dv = dv; memo_nwg = nwg; memo_pipeline = cached;
        return cached;
    }

    MTLFunctionConstantValues *constants = [[MTLFunctionConstantValues alloc] init];
    [constants setConstantValue:&dv  type:MTLDataTypeInt atIndex:500];
    [constants setConstantValue:&nwg type:MTLDataTypeInt atIndex:501];

    NSError *error = nil;
    id<MTLFunction> fn = [g_library newFunctionWithName:@"kernel_flash_attn_ext_vec_reduce"
                                         constantValues:constants
                                                  error:&error];
    if (!fn) {
        fprintf(stderr, "ds4: Metal kernel_flash_attn_ext_vec_reduce function not found: %s\n",
                [[error localizedDescription] UTF8String]);
        return nil;
    }

    error = nil;
    id<MTLComputePipelineState> pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!pipeline) {
        fprintf(stderr, "ds4: Metal kernel_flash_attn_ext_vec_reduce pipeline failed: %s\n",
                [[error localizedDescription] UTF8String]);
        return nil;
    }

    [g_pipeline_cache setObject:pipeline forKey:key];
    memo_dv = dv; memo_nwg = nwg; memo_pipeline = pipeline;
    return pipeline;
}

static uint32_t ds4_gpu_flash_attn_vec_nsg(uint32_t n_keys, uint32_t nwg, uint32_t ncpsg) {
    uint32_t nsg = 1;
    while (2u * nwg * nsg * ncpsg < n_keys && nsg < 4u) {
        nsg *= 2u;
    }
    return nsg;
}

static int ds4_gpu_trace_allocs(void) {
    static int initialized;
    static int enabled;
    if (!initialized) {
        enabled = getenv("DS4_METAL_TRACE_ALLOCS") != NULL;
        initialized = 1;
    }
    return enabled;
}

static double ds4_gpu_mib(uint64_t bytes) {
    return (double)bytes / (1024.0 * 1024.0);
}

static double ds4_gpu_gib(uint64_t bytes) {
    return (double)bytes / (1024.0 * 1024.0 * 1024.0);
}

static ds4_gpu_stream_expert_timing_snapshot
ds4_gpu_stream_expert_timing_current(void) {
    return (ds4_gpu_stream_expert_timing_snapshot) {
        .selected_calls = g_stream_expert_timing_selected_calls,
        .selected_read_ms = g_stream_expert_timing_selected_read_ms,
        .selected_sync_ms = g_stream_expert_timing_selected_sync_ms,
        .selected_copy_ms = g_stream_expert_timing_selected_copy_ms,
        .selected_bind_ms = g_stream_expert_timing_selected_bind_ms,
        .split_layers = g_stream_expert_timing_split_layers,
        .split_resident_experts = g_stream_expert_timing_split_resident_experts,
        .split_missing_experts = g_stream_expert_timing_split_missing_experts,
        .split_resident_ms = g_stream_expert_timing_split_resident_ms,
        .split_missing_ms = g_stream_expert_timing_split_missing_ms,
        .split_missing_load_ms =
            g_stream_expert_timing_split_missing_load_ms,
        .split_missing_slot_ms =
            g_stream_expert_timing_split_missing_slot_ms,
        .split_missing_prune_ms =
            g_stream_expert_timing_split_missing_prune_ms,
        .split_missing_addr_ms =
            g_stream_expert_timing_split_missing_addr_ms,
        .split_missing_wait_ms =
            g_stream_expert_timing_split_missing_wait_ms,
        .load_calls = g_stream_expert_timing_load_calls,
        .load_prepare_ms = g_stream_expert_timing_load_prepare_ms,
        .load_pread_ms = g_stream_expert_timing_load_pread_ms,
        .load_modify_ms = g_stream_expert_timing_load_modify_ms,
        .load_install_ms = g_stream_expert_timing_load_install_ms,
        .prepare_batch_reuse_calls =
            g_stream_expert_timing_prepare_batch_reuse_calls,
        .prepare_batch_reuse_ms =
            g_stream_expert_timing_prepare_batch_reuse_ms,
        .prepare_buffer_calls =
            g_stream_expert_timing_prepare_buffer_calls,
        .prepare_buffer_ms =
            g_stream_expert_timing_prepare_buffer_ms,
        .prepare_task_experts =
            g_stream_expert_timing_prepare_task_experts,
        .prepare_task_ms =
            g_stream_expert_timing_prepare_task_ms,
        .reuse_scan_calls =
            g_stream_expert_timing_reuse_scan_calls,
        .reuse_scan_entries =
            g_stream_expert_timing_reuse_scan_entries,
        .reuse_scan_ms =
            g_stream_expert_timing_reuse_scan_ms,
        .reuse_clear_ms =
            g_stream_expert_timing_reuse_clear_ms,
        .readahead_calls =
            g_stream_expert_timing_readahead_calls,
        .readahead_bytes =
            g_stream_expert_timing_readahead_bytes,
        .readahead_ms =
            g_stream_expert_timing_readahead_ms,
        .cache_all_resident_layers =
            g_stream_expert_timing_cache_all_resident_layers,
        .cache_all_missing_layers =
            g_stream_expert_timing_cache_all_missing_layers,
        .cache_mixed_layers = g_stream_expert_timing_cache_mixed_layers,
        .cache_resident_experts =
            g_stream_expert_timing_cache_resident_experts,
        .cache_missing_experts =
            g_stream_expert_timing_cache_missing_experts,
    };
}

static uint64_t ds4_gpu_stream_expert_timing_delta_u64(
        uint64_t current,
        uint64_t previous) {
    return current >= previous ? current - previous : current;
}

static double ds4_gpu_stream_expert_timing_delta_f64(
        double current,
        double previous) {
    return current >= previous ? current - previous : current;
}

static ds4_gpu_stream_expert_timing_snapshot
ds4_gpu_stream_expert_timing_delta(
        ds4_gpu_stream_expert_timing_snapshot current,
        ds4_gpu_stream_expert_timing_snapshot previous) {
    return (ds4_gpu_stream_expert_timing_snapshot) {
        .selected_calls =
            ds4_gpu_stream_expert_timing_delta_u64(current.selected_calls,
                                                   previous.selected_calls),
        .selected_read_ms =
            ds4_gpu_stream_expert_timing_delta_f64(current.selected_read_ms,
                                                   previous.selected_read_ms),
        .selected_sync_ms =
            ds4_gpu_stream_expert_timing_delta_f64(current.selected_sync_ms,
                                                   previous.selected_sync_ms),
        .selected_copy_ms =
            ds4_gpu_stream_expert_timing_delta_f64(current.selected_copy_ms,
                                                   previous.selected_copy_ms),
        .selected_bind_ms =
            ds4_gpu_stream_expert_timing_delta_f64(current.selected_bind_ms,
                                                   previous.selected_bind_ms),
        .split_layers =
            ds4_gpu_stream_expert_timing_delta_u64(current.split_layers,
                                                   previous.split_layers),
        .split_resident_experts =
            ds4_gpu_stream_expert_timing_delta_u64(current.split_resident_experts,
                                                   previous.split_resident_experts),
        .split_missing_experts =
            ds4_gpu_stream_expert_timing_delta_u64(current.split_missing_experts,
                                                   previous.split_missing_experts),
        .split_resident_ms =
            ds4_gpu_stream_expert_timing_delta_f64(current.split_resident_ms,
                                                   previous.split_resident_ms),
        .split_missing_ms =
            ds4_gpu_stream_expert_timing_delta_f64(current.split_missing_ms,
                                                   previous.split_missing_ms),
        .split_missing_load_ms =
            ds4_gpu_stream_expert_timing_delta_f64(
                    current.split_missing_load_ms,
                    previous.split_missing_load_ms),
        .split_missing_slot_ms =
            ds4_gpu_stream_expert_timing_delta_f64(
                    current.split_missing_slot_ms,
                    previous.split_missing_slot_ms),
        .split_missing_prune_ms =
            ds4_gpu_stream_expert_timing_delta_f64(
                    current.split_missing_prune_ms,
                    previous.split_missing_prune_ms),
        .split_missing_addr_ms =
            ds4_gpu_stream_expert_timing_delta_f64(
                    current.split_missing_addr_ms,
                    previous.split_missing_addr_ms),
        .split_missing_wait_ms =
            ds4_gpu_stream_expert_timing_delta_f64(
                    current.split_missing_wait_ms,
                    previous.split_missing_wait_ms),
        .load_calls =
            ds4_gpu_stream_expert_timing_delta_u64(current.load_calls,
                                                   previous.load_calls),
        .load_prepare_ms =
            ds4_gpu_stream_expert_timing_delta_f64(current.load_prepare_ms,
                                                   previous.load_prepare_ms),
        .load_pread_ms =
            ds4_gpu_stream_expert_timing_delta_f64(current.load_pread_ms,
                                                   previous.load_pread_ms),
        .load_modify_ms =
            ds4_gpu_stream_expert_timing_delta_f64(current.load_modify_ms,
                                                   previous.load_modify_ms),
        .load_install_ms =
            ds4_gpu_stream_expert_timing_delta_f64(current.load_install_ms,
                                                   previous.load_install_ms),
        .prepare_batch_reuse_calls =
            ds4_gpu_stream_expert_timing_delta_u64(
                    current.prepare_batch_reuse_calls,
                    previous.prepare_batch_reuse_calls),
        .prepare_batch_reuse_ms =
            ds4_gpu_stream_expert_timing_delta_f64(
                    current.prepare_batch_reuse_ms,
                    previous.prepare_batch_reuse_ms),
        .prepare_buffer_calls =
            ds4_gpu_stream_expert_timing_delta_u64(
                    current.prepare_buffer_calls,
                    previous.prepare_buffer_calls),
        .prepare_buffer_ms =
            ds4_gpu_stream_expert_timing_delta_f64(
                    current.prepare_buffer_ms,
                    previous.prepare_buffer_ms),
        .prepare_task_experts =
            ds4_gpu_stream_expert_timing_delta_u64(
                    current.prepare_task_experts,
                    previous.prepare_task_experts),
        .prepare_task_ms =
            ds4_gpu_stream_expert_timing_delta_f64(
                    current.prepare_task_ms,
                    previous.prepare_task_ms),
        .reuse_scan_calls =
            ds4_gpu_stream_expert_timing_delta_u64(
                    current.reuse_scan_calls,
                    previous.reuse_scan_calls),
        .reuse_scan_entries =
            ds4_gpu_stream_expert_timing_delta_u64(
                    current.reuse_scan_entries,
                    previous.reuse_scan_entries),
        .reuse_scan_ms =
            ds4_gpu_stream_expert_timing_delta_f64(
                    current.reuse_scan_ms,
                    previous.reuse_scan_ms),
        .reuse_clear_ms =
            ds4_gpu_stream_expert_timing_delta_f64(
                    current.reuse_clear_ms,
                    previous.reuse_clear_ms),
        .readahead_calls =
            ds4_gpu_stream_expert_timing_delta_u64(
                    current.readahead_calls,
                    previous.readahead_calls),
        .readahead_bytes =
            ds4_gpu_stream_expert_timing_delta_u64(
                    current.readahead_bytes,
                    previous.readahead_bytes),
        .readahead_ms =
            ds4_gpu_stream_expert_timing_delta_f64(
                    current.readahead_ms,
                    previous.readahead_ms),
        .cache_all_resident_layers =
            ds4_gpu_stream_expert_timing_delta_u64(
                    current.cache_all_resident_layers,
                    previous.cache_all_resident_layers),
        .cache_all_missing_layers =
            ds4_gpu_stream_expert_timing_delta_u64(
                    current.cache_all_missing_layers,
                    previous.cache_all_missing_layers),
        .cache_mixed_layers =
            ds4_gpu_stream_expert_timing_delta_u64(current.cache_mixed_layers,
                                                   previous.cache_mixed_layers),
        .cache_resident_experts =
            ds4_gpu_stream_expert_timing_delta_u64(
                    current.cache_resident_experts,
                    previous.cache_resident_experts),
        .cache_missing_experts =
            ds4_gpu_stream_expert_timing_delta_u64(
                    current.cache_missing_experts,
                    previous.cache_missing_experts),
    };
}

static int ds4_gpu_stream_expert_timing_has_data(
        ds4_gpu_stream_expert_timing_snapshot s) {
    return s.selected_calls != 0 ||
           s.split_layers != 0 ||
           s.load_calls != 0 ||
           s.cache_all_resident_layers != 0 ||
           s.cache_all_missing_layers != 0 ||
           s.cache_mixed_layers != 0;
}

static void ds4_gpu_stream_expert_timing_print(
        const char *scope,
        ds4_gpu_stream_expert_timing_snapshot s) {
    if (!ds4_gpu_stream_expert_timing_has_data(s)) return;
    const double selected_calls = (double)s.selected_calls;
    const double split_layers = (double)s.split_layers;
    const double selected_read_avg =
        selected_calls != 0.0 ? s.selected_read_ms / selected_calls : 0.0;
    const double selected_sync_avg =
        selected_calls != 0.0 ? s.selected_sync_ms / selected_calls : 0.0;
    const double selected_copy_avg =
        selected_calls != 0.0 ? s.selected_copy_ms / selected_calls : 0.0;
    const double selected_bind_avg =
        selected_calls != 0.0 ? s.selected_bind_ms / selected_calls : 0.0;
    const double split_resident_avg =
        split_layers != 0.0 ? s.split_resident_ms / split_layers : 0.0;
    const double split_missing_avg =
        split_layers != 0.0 ? s.split_missing_ms / split_layers : 0.0;
    const double split_missing_load_avg =
        split_layers != 0.0 ?
            s.split_missing_load_ms / split_layers : 0.0;
    const double split_missing_slot_avg =
        split_layers != 0.0 ?
            s.split_missing_slot_ms / split_layers : 0.0;
    const double split_missing_prune_avg =
        split_layers != 0.0 ?
            s.split_missing_prune_ms / split_layers : 0.0;
    const double split_missing_addr_avg =
        split_layers != 0.0 ?
            s.split_missing_addr_ms / split_layers : 0.0;
    const double split_missing_wait_avg =
        split_layers != 0.0 ?
            s.split_missing_wait_ms / split_layers : 0.0;
    const double load_calls = (double)s.load_calls;
    const double load_prepare_avg =
        load_calls != 0.0 ? s.load_prepare_ms / load_calls : 0.0;
    const double load_pread_avg =
        load_calls != 0.0 ? s.load_pread_ms / load_calls : 0.0;
    const double load_modify_avg =
        load_calls != 0.0 ? s.load_modify_ms / load_calls : 0.0;
    const double load_install_avg =
        load_calls != 0.0 ? s.load_install_ms / load_calls : 0.0;
    const double prepare_batch_reuse_avg =
        s.prepare_batch_reuse_calls != 0 ?
            s.prepare_batch_reuse_ms /
                (double)s.prepare_batch_reuse_calls : 0.0;
    const double prepare_buffer_avg =
        s.prepare_buffer_calls != 0 ?
            s.prepare_buffer_ms / (double)s.prepare_buffer_calls : 0.0;
    const double prepare_task_avg =
        s.prepare_task_experts != 0 ?
            s.prepare_task_ms / (double)s.prepare_task_experts : 0.0;
    const double reuse_scan_avg =
        s.reuse_scan_calls != 0 ?
            s.reuse_scan_ms / (double)s.reuse_scan_calls : 0.0;
    const double reuse_scan_entries_avg =
        s.reuse_scan_calls != 0 ?
            (double)s.reuse_scan_entries / (double)s.reuse_scan_calls : 0.0;
    const double readahead_avg =
        s.readahead_calls != 0 ?
            s.readahead_ms / (double)s.readahead_calls : 0.0;
    const double split_resident_experts_avg =
        split_layers != 0.0 ?
            (double)s.split_resident_experts / split_layers : 0.0;
    const double split_missing_experts_avg =
        split_layers != 0.0 ?
            (double)s.split_missing_experts / split_layers : 0.0;
    const uint64_t cache_layers =
        s.cache_all_resident_layers +
        s.cache_all_missing_layers +
        s.cache_mixed_layers;
    const double cache_layer_count = (double)cache_layers;
    const double cache_resident_experts_avg =
        cache_layer_count != 0.0 ?
            (double)s.cache_resident_experts / cache_layer_count : 0.0;
    const double cache_missing_experts_avg =
        cache_layer_count != 0.0 ?
            (double)s.cache_missing_experts / cache_layer_count : 0.0;
    fprintf(stderr,
            "ds4:   streaming expert timing %s selected_calls=%llu read_avg=%.3f ms sync_avg=%.3f ms copy_avg=%.3f ms bind_avg=%.3f ms read_total=%.3f ms sync_total=%.3f ms copy_total=%.3f ms bind_total=%.3f ms split_layers=%llu resident_experts_avg=%.2f missing_experts_avg=%.2f resident_submit_avg=%.3f ms missing_bind_avg=%.3f ms resident_submit_total=%.3f ms missing_bind_total=%.3f ms missing_load_avg=%.3f ms missing_slot_avg=%.3f ms missing_prune_avg=%.3f ms missing_addr_avg=%.3f ms missing_wait_avg=%.3f ms missing_wait_total=%.3f ms load_calls=%llu load_prepare_avg=%.3f ms load_pread_avg=%.3f ms load_modify_avg=%.3f ms load_install_avg=%.3f ms prepare_batch_reuse_calls=%llu prepare_batch_reuse_avg=%.3f ms prepare_batch_reuse_total=%.3f ms prepare_buffer_calls=%llu prepare_buffer_avg=%.3f ms prepare_buffer_total=%.3f ms prepare_task_experts=%llu prepare_task_avg=%.3f ms prepare_task_total=%.3f ms reuse_scan_calls=%llu reuse_scan_entries_avg=%.1f reuse_scan_avg=%.3f ms reuse_scan_total=%.3f ms reuse_clear_total=%.3f ms readahead_calls=%llu readahead_avg=%.3f ms readahead_total=%.3f ms readahead_gib=%.2f cache_all_resident=%llu cache_all_missing=%llu cache_mixed=%llu cache_resident_avg=%.2f cache_missing_avg=%.2f\n",
            scope ? scope : "total",
            (unsigned long long)s.selected_calls,
            selected_read_avg,
            selected_sync_avg,
            selected_copy_avg,
            selected_bind_avg,
            s.selected_read_ms,
            s.selected_sync_ms,
            s.selected_copy_ms,
            s.selected_bind_ms,
            (unsigned long long)s.split_layers,
            split_resident_experts_avg,
            split_missing_experts_avg,
            split_resident_avg,
            split_missing_avg,
            s.split_resident_ms,
            s.split_missing_ms,
            split_missing_load_avg,
            split_missing_slot_avg,
            split_missing_prune_avg,
            split_missing_addr_avg,
            split_missing_wait_avg,
            s.split_missing_wait_ms,
            (unsigned long long)s.load_calls,
            load_prepare_avg,
            load_pread_avg,
            load_modify_avg,
            load_install_avg,
            (unsigned long long)s.prepare_batch_reuse_calls,
            prepare_batch_reuse_avg,
            s.prepare_batch_reuse_ms,
            (unsigned long long)s.prepare_buffer_calls,
            prepare_buffer_avg,
            s.prepare_buffer_ms,
            (unsigned long long)s.prepare_task_experts,
            prepare_task_avg,
            s.prepare_task_ms,
            (unsigned long long)s.reuse_scan_calls,
            reuse_scan_entries_avg,
            reuse_scan_avg,
            s.reuse_scan_ms,
            s.reuse_clear_ms,
            (unsigned long long)s.readahead_calls,
            readahead_avg,
            s.readahead_ms,
            ds4_gpu_gib(s.readahead_bytes),
            (unsigned long long)s.cache_all_resident_layers,
            (unsigned long long)s.cache_all_missing_layers,
            (unsigned long long)s.cache_mixed_layers,
            cache_resident_experts_avg,
            cache_missing_experts_avg);
}

static void ds4_gpu_print_task_memory_report(void) {
    task_vm_info_data_t info;
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    const kern_return_t kr = task_info(mach_task_self(),
                                       TASK_VM_INFO,
                                       (task_info_t)&info,
                                       &count);
    if (kr != KERN_SUCCESS) return;

    fprintf(stderr,
            "ds4:   macOS task memory footprint %.2f GiB, resident %.2f GiB, virtual %.2f GiB\n",
            ds4_gpu_gib((uint64_t)info.phys_footprint),
            ds4_gpu_gib((uint64_t)info.resident_size),
            ds4_gpu_gib((uint64_t)info.virtual_size));
}

void ds4_gpu_print_memory_report(const char *label) {
    uint64_t cached_prefill_mask_bytes = 0;
    uint64_t cached_prefill_blk_bytes = 0;
    for (uint32_t i = 0; i < DS4_GPU_PREFILL_MASK_CACHE_SLOTS; i++) {
        const ds4_gpu_zero_prefix_prefill_mask_cache_entry *entry =
            &g_zero_prefix_prefill_mask_cache[i];
        if (entry->mask) cached_prefill_mask_bytes += entry->mask_bytes;
        if (entry->blk) cached_prefill_blk_bytes += entry->blk_bytes;
    }
    const uint64_t scratch =
        (uint64_t)g_flash_attn_mask_bytes +
        (uint64_t)g_flash_attn_zero_mask_bytes +
        cached_prefill_mask_bytes +
        (uint64_t)g_flash_attn_pad_bytes +
        (uint64_t)g_flash_attn_tmp_bytes +
        (uint64_t)g_flash_attn_blk_bytes +
        cached_prefill_blk_bytes +
        (uint64_t)g_flash_attn_ring_bytes +
        (uint64_t)g_flash_attn_kv_bytes +
        (uint64_t)g_glm_flash_attn_mask_bytes +
        (uint64_t)g_compressor_pool_kv_bytes +
        (uint64_t)g_compressor_pool_score_bytes +
        (uint64_t)g_compressor_pool_score_cont_bytes +
        (uint64_t)g_compressor_pool_softmax_bytes +
        (uint64_t)g_compressor_pool_product_bytes +
        (uint64_t)g_compressor_store_ape_bytes +
        (uint64_t)g_compressor_store_score_bytes +
        (uint64_t)g_embed_rows_bytes +
        (uint64_t)g_router_selection_bytes +
        (uint64_t)g_router_weight_sum_bytes +
        (uint64_t)g_indexer_head_scores_bytes +
        (uint64_t)g_indexer_topk_bytes +
        (uint64_t)g_indexed_topk_bytes +
        (uint64_t)g_f16_round_scratch_bytes +
        (uint64_t)g_raw_store_round_bytes +
        (uint64_t)g_moe_gate_scratch_bytes +
        (uint64_t)g_moe_down_scratch_bytes +
        (uint64_t)g_moe_id_map_bytes +
        (uint64_t)g_moe_q4_gate_slots_bytes +
        (uint64_t)g_moe_q4_up_slots_bytes +
        (uint64_t)g_moe_q4_down_slots_bytes;

    pthread_mutex_lock(&g_tensor_mu);
    const uint64_t tensor_live_snap = g_tensor_alloc_live_bytes;
    const uint64_t tensor_peak_snap = g_tensor_alloc_peak_bytes;
    pthread_mutex_unlock(&g_tensor_mu);

    uint64_t tracked_live = tensor_live_snap;
    if (tracked_live > UINT64_MAX - g_stream_expert_cache_bytes) {
        tracked_live = UINT64_MAX;
    } else {
        tracked_live += g_stream_expert_cache_bytes;
    }

    const bool color = ds4_log_is_tty(stderr);
    const char *green = color ? "\x1b[32m" : "";
    const char *bright_green = color ? "\x1b[1;32m" : "";
    const char *reset = color ? "\x1b[0m" : "";
    fprintf(stderr,
            "%sds4: Metal memory%s%s: runtime %.2f GiB + streaming experts %.2f GiB = %s%.2f GiB tracked live%s\n",
            green,
            label && label[0] ? " " : "",
            label && label[0] ? label : "",
            ds4_gpu_gib(tensor_live_snap),
            ds4_gpu_gib(g_stream_expert_cache_bytes),
            bright_green,
            ds4_gpu_gib(tracked_live),
            reset);
    if (color) fputs(green, stderr);
    fprintf(stderr,
            "ds4:   runtime tensors live %.2f MiB peak %.2f MiB\n",
            ds4_gpu_mib(tensor_live_snap),
            ds4_gpu_mib(tensor_peak_snap));
    ds4_gpu_print_task_memory_report();
    fprintf(stderr,
            "ds4:   mmap model wrapper spans %llu buffers %.2f GiB total, %.2f GiB max (not copied)\n",
            (unsigned long long)g_model_wrap_count,
            ds4_gpu_gib(g_model_wrap_bytes),
            ds4_gpu_gib(g_model_wrap_max_bytes));
    if (g_model_buffer_cache && [g_model_buffer_cache count] != 0) {
        const uint64_t limit = ds4_gpu_exact_view_cache_limit_bytes();
        if (limit == 0) {
            fprintf(stderr,
                    "ds4:   exact model view cache %lu buffers %.2f GiB unlimited, %llu evictions (not copied)\n",
                    (unsigned long)[g_model_buffer_cache count],
                    ds4_gpu_gib(g_model_buffer_cache_bytes),
                    (unsigned long long)g_model_buffer_cache_evictions);
        } else {
            fprintf(stderr,
                    "ds4:   exact model view cache %lu buffers %.2f GiB / %.2f GiB, %llu evictions (not copied)\n",
                    (unsigned long)[g_model_buffer_cache count],
                    ds4_gpu_gib(g_model_buffer_cache_bytes),
                    ds4_gpu_gib(limit),
                    (unsigned long long)g_model_buffer_cache_evictions);
        }
    }
    if (g_stream_expert_cache_hits != 0 ||
        g_stream_expert_cache_misses != 0 ||
        g_stream_expert_cache_bytes != 0) {
        const uint64_t budget = ds4_gpu_stream_expert_cache_configured_budget();
        uint64_t target_bytes = 0;
        if (budget != 0 && g_stream_expert_cache_expert_bytes != 0) {
            target_bytes =
                budget > UINT64_MAX / g_stream_expert_cache_expert_bytes ?
                    UINT64_MAX :
                    budget * g_stream_expert_cache_expert_bytes;
        }
        const uint64_t lookups = g_stream_expert_cache_hits + g_stream_expert_cache_misses;
        const double hit_rate = lookups ?
            (double)g_stream_expert_cache_hits / (double)lookups : 0.0;
        if (g_stream_expert_cache_evict_advise_bytes != 0 ||
            g_stream_expert_cache_willneed_advise_bytes != 0 ||
            g_stream_expert_cache_pread_bytes != 0) {
            fprintf(stderr,
                    "ds4:   streaming expert cache budget=%llu experts entries=%u expert=%.2f MiB target=%.2f GiB live=%.2f GiB, hits=%llu misses=%llu hit_rate=%.3f wraps=%llu evictions=%llu buffer_allocs=%llu buffer_reuses=%llu evict_dontneed=%.2f GiB miss_willneed=%.2f GiB miss_pread=%.2f GiB pread_ms=%.3f\n",
                    (unsigned long long)budget,
                    g_stream_expert_cache_entry_count,
                    ds4_gpu_mib(g_stream_expert_cache_expert_bytes),
                    ds4_gpu_gib(target_bytes),
                    ds4_gpu_gib(g_stream_expert_cache_bytes),
                    (unsigned long long)g_stream_expert_cache_hits,
                    (unsigned long long)g_stream_expert_cache_misses,
                    hit_rate,
                    (unsigned long long)g_stream_expert_cache_wraps,
                    (unsigned long long)g_stream_expert_cache_evictions,
                    (unsigned long long)g_stream_expert_cache_buffer_allocs,
                    (unsigned long long)g_stream_expert_cache_buffer_reuses,
                    ds4_gpu_gib(g_stream_expert_cache_evict_advise_bytes),
                    ds4_gpu_gib(g_stream_expert_cache_willneed_advise_bytes),
                    ds4_gpu_gib(g_stream_expert_cache_pread_bytes),
                    g_stream_expert_cache_pread_ms);
        } else {
            fprintf(stderr,
                    "ds4:   streaming expert cache budget=%llu experts entries=%u expert=%.2f MiB target=%.2f GiB live=%.2f GiB, hits=%llu misses=%llu hit_rate=%.3f wraps=%llu evictions=%llu buffer_allocs=%llu buffer_reuses=%llu\n",
                    (unsigned long long)budget,
                    g_stream_expert_cache_entry_count,
                    ds4_gpu_mib(g_stream_expert_cache_expert_bytes),
                    ds4_gpu_gib(target_bytes),
                    ds4_gpu_gib(g_stream_expert_cache_bytes),
                    (unsigned long long)g_stream_expert_cache_hits,
                    (unsigned long long)g_stream_expert_cache_misses,
                    hit_rate,
                    (unsigned long long)g_stream_expert_cache_wraps,
                    (unsigned long long)g_stream_expert_cache_evictions,
                    (unsigned long long)g_stream_expert_cache_buffer_allocs,
                    (unsigned long long)g_stream_expert_cache_buffer_reuses);
        }
        if (ds4_gpu_stream_expert_timing_summary_enabled()) {
            const ds4_gpu_stream_expert_timing_snapshot total =
                ds4_gpu_stream_expert_timing_current();
            if (ds4_gpu_stream_expert_timing_has_data(total)) {
                const ds4_gpu_stream_expert_timing_snapshot delta =
                    ds4_gpu_stream_expert_timing_delta(
                            total,
                            g_stream_expert_timing_last_report);
                ds4_gpu_stream_expert_timing_print("total", total);
                ds4_gpu_stream_expert_timing_print("delta", delta);
                g_stream_expert_timing_last_report = total;
            }
        }
        if (getenv("DS4_METAL_STREAMING_EXPERT_LAYER_STATS") != NULL) {
            fprintf(stderr, "ds4:   streaming expert cache per-layer stats:\n");
            for (uint32_t layer = 0;
                 layer < DS4_METAL_STREAM_EXPERT_CACHE_MAX_LAYER;
                 layer++) {
                const uint64_t hits = g_stream_expert_cache_layer_hits[layer];
                const uint64_t misses = g_stream_expert_cache_layer_misses[layer];
                const uint64_t lookups = hits + misses;
                const uint64_t evictions =
                    g_stream_expert_cache_layer_evictions[layer];
                const uint64_t pread_bytes =
                    g_stream_expert_cache_layer_pread_bytes[layer];
                const double pread_ms =
                    g_stream_expert_cache_layer_pread_ms[layer];
                const uint32_t cached =
                    g_stream_expert_cache_layer_count[layer];
                const uint32_t layer_slots =
                    ds4_gpu_stream_expert_cache_configured_count() != 0 ?
                        DS4_METAL_STREAM_EXPERT_CACHE_MAX_EXPERT : 0;
                if (lookups == 0 && evictions == 0 && pread_bytes == 0 &&
                    cached == 0) {
                    continue;
                }
                const double layer_hit_rate = lookups ?
                    (double)hits / (double)lookups : 0.0;
                fprintf(stderr,
                        "ds4:     layer=%u layer_slots=%u cached=%u hits=%llu misses=%llu hit_rate=%.3f evictions=%llu miss_pread=%.2f GiB pread_ms=%.3f\n",
                        layer,
                        layer_slots,
                        cached,
                        (unsigned long long)hits,
                        (unsigned long long)misses,
                        layer_hit_rate,
                        (unsigned long long)evictions,
                        ds4_gpu_gib(pread_bytes),
                        pread_ms);
            }
            if (getenv("DS4_METAL_STREAMING_EXPERT_LAYER_STATS_DELTA") != NULL) {
                fprintf(stderr, "ds4:   streaming expert cache per-layer delta:\n");
                for (uint32_t layer = 0;
                     layer < DS4_METAL_STREAM_EXPERT_CACHE_MAX_LAYER;
                     layer++) {
                    const uint64_t hits =
                        g_stream_expert_cache_layer_hits[layer];
                    const uint64_t misses =
                        g_stream_expert_cache_layer_misses[layer];
                    const uint64_t evictions =
                        g_stream_expert_cache_layer_evictions[layer];
                    const uint64_t pread_bytes =
                        g_stream_expert_cache_layer_pread_bytes[layer];
                    const double pread_ms =
                        g_stream_expert_cache_layer_pread_ms[layer];
                    const uint64_t delta_hits =
                        hits - g_stream_expert_cache_layer_last_hits[layer];
                    const uint64_t delta_misses =
                        misses - g_stream_expert_cache_layer_last_misses[layer];
                    const uint64_t delta_evictions =
                        evictions -
                        g_stream_expert_cache_layer_last_evictions[layer];
                    const uint64_t delta_pread_bytes =
                        pread_bytes -
                        g_stream_expert_cache_layer_last_pread_bytes[layer];
                    const double delta_pread_ms =
                        pread_ms -
                        g_stream_expert_cache_layer_last_pread_ms[layer];
                    const uint64_t lookups = delta_hits + delta_misses;
                    if (lookups == 0 && delta_evictions == 0 &&
                        delta_pread_bytes == 0) {
                        continue;
                    }
                    const double hit_rate = lookups ?
                        (double)delta_hits / (double)lookups : 0.0;
                    fprintf(stderr,
                            "ds4:     layer=%u cached=%u hits=%llu misses=%llu hit_rate=%.3f evictions=%llu miss_pread=%.2f GiB pread_ms=%.3f\n",
                            layer,
                            g_stream_expert_cache_layer_count[layer],
                            (unsigned long long)delta_hits,
                            (unsigned long long)delta_misses,
                            hit_rate,
                            (unsigned long long)delta_evictions,
                            ds4_gpu_gib(delta_pread_bytes),
                            delta_pread_ms);
                    g_stream_expert_cache_layer_last_hits[layer] = hits;
                    g_stream_expert_cache_layer_last_misses[layer] = misses;
                    g_stream_expert_cache_layer_last_evictions[layer] =
                        evictions;
                    g_stream_expert_cache_layer_last_pread_bytes[layer] =
                        pread_bytes;
                    g_stream_expert_cache_layer_last_pread_ms[layer] =
                        pread_ms;
                }
            }
        }
    }
    fprintf(stderr,
            "ds4:   model residency requests %llu%s\n",
            (unsigned long long)g_model_residency_count,
            g_ssd_streaming_mode ? " (ssd-streaming)" :
            (getenv("DS4_METAL_NO_RESIDENCY") != NULL ? " (disabled)" : ""));
    fprintf(stderr,
            "ds4:   device %s, Metal 4 runtime %s, family %s, MTL4 queue %s, tensor API %s, M5 neural accelerators %s\n",
            g_metal_device_name[0] ? g_metal_device_name : "(unknown)",
            g_metal4_runtime_available ? "yes" : "no",
            g_metal4_family_supported ? "yes" : "no",
            g_metal4_queue_supported ? "yes" : "no",
            g_metal4_tensor_api_enabled ? "enabled" :
                (g_metal4_tensor_api_compile_supported ? "available" : "disabled"),
            g_metal4_m5_neural_accelerators_hint ? "likely" : "not detected");
    fprintf(stderr,
            "ds4:   accelerated Metal path %s%s\n",
            ds4_gpu_mpp_available() ? "enabled" : "disabled",
            g_quality_mode ? " by --quality" :
                (!g_metal4_tensor_api_enabled ? " (tensor API unavailable)" : ""));
    fprintf(stderr,
            "ds4:   device %s, Metal 4 runtime %s, family %s, MTL4 queue %s, tensor API %s, M5 neural accelerators %s\n",
            g_metal_device_name[0] ? g_metal_device_name : "(unknown)",
            g_metal4_runtime_available ? "yes" : "no",
            g_metal4_family_supported ? "yes" : "no",
            g_metal4_queue_supported ? "yes" : "no",
            g_metal4_tensor_api_enabled ? "enabled" :
                (g_metal4_tensor_api_compile_supported ? "available" : "disabled"),
            g_metal4_m5_neural_accelerators_hint ? "likely" : "not detected");
    fprintf(stderr,
            "ds4:   scratch %.2f MiB (flash mask %.2f, pad %.2f, tmp %.2f, blk %.2f, ring %.2f, kv %.2f, compressor %.2f, router %.2f, indexer %.2f, moe %.2f, f16 %.2f, raw-store %.2f)\n",
            ds4_gpu_mib(scratch),
            ds4_gpu_mib((uint64_t)g_flash_attn_mask_bytes +
                        (uint64_t)g_glm_flash_attn_mask_bytes +
                        (uint64_t)g_flash_attn_zero_mask_bytes +
                        cached_prefill_mask_bytes),
            ds4_gpu_mib((uint64_t)g_flash_attn_pad_bytes),
            ds4_gpu_mib((uint64_t)g_flash_attn_tmp_bytes),
            ds4_gpu_mib((uint64_t)g_flash_attn_blk_bytes +
                        cached_prefill_blk_bytes),
            ds4_gpu_mib((uint64_t)g_flash_attn_ring_bytes),
            ds4_gpu_mib((uint64_t)g_flash_attn_kv_bytes),
            ds4_gpu_mib((uint64_t)g_compressor_pool_kv_bytes +
                          (uint64_t)g_compressor_pool_score_bytes +
                          (uint64_t)g_compressor_pool_score_cont_bytes +
                          (uint64_t)g_compressor_pool_softmax_bytes +
                          (uint64_t)g_compressor_pool_product_bytes +
                          (uint64_t)g_compressor_store_ape_bytes +
                          (uint64_t)g_compressor_store_score_bytes +
                          (uint64_t)g_embed_rows_bytes),
            ds4_gpu_mib((uint64_t)g_router_selection_bytes +
                          (uint64_t)g_router_weight_sum_bytes),
            ds4_gpu_mib((uint64_t)g_indexer_head_scores_bytes +
                          (uint64_t)g_indexer_topk_bytes +
                          (uint64_t)g_indexed_topk_bytes),
            ds4_gpu_mib((uint64_t)g_moe_gate_scratch_bytes +
                          (uint64_t)g_moe_down_scratch_bytes +
                          (uint64_t)g_moe_id_map_bytes +
                          (uint64_t)g_moe_q4_gate_slots_bytes +
                          (uint64_t)g_moe_q4_up_slots_bytes +
                          (uint64_t)g_moe_q4_down_slots_bytes),
            ds4_gpu_mib((uint64_t)g_f16_round_scratch_bytes),
            ds4_gpu_mib((uint64_t)g_raw_store_round_bytes));
    if (color) fputs(reset, stderr);
}

void ds4_gpu_set_quality(bool quality) {
    g_quality_mode = quality ? 1 : 0;
}

void ds4_gpu_set_glm_model(bool enabled) {
    g_glm_model_mode = enabled ? 1 : 0;
}

void ds4_gpu_set_ssd_streaming(bool enabled) {
    g_ssd_streaming_mode = enabled ? 1 : 0;
    ds4_gpu_stream_expert_cache_clear_all(1);
    if (g_ssd_streaming_mode) {
        fprintf(stderr,
                "ds4: Metal SSD streaming mode enabled; full model residency and warmup are skipped\n");
    }
}

void ds4_gpu_set_glm_streaming_prefill_full_layer(bool enabled) {
    g_glm_streaming_prefill_full_layer_runtime = enabled ? 1 : 0;
}

void ds4_gpu_set_streaming_expert_cache_budget(uint32_t experts) {
    if (experts > DS4_METAL_STREAM_EXPERT_CACHE_MAX_ENTRIES) {
        experts = DS4_METAL_STREAM_EXPERT_CACHE_MAX_ENTRIES;
    }
    g_stream_expert_cache_budget_override = experts;
    ds4_gpu_stream_expert_cache_clear_all(1);
}

void ds4_gpu_set_streaming_expert_cache_expert_bytes(uint64_t bytes) {
    /*
     * Pre-seed the cache's single slab size class with the model's uniform
     * per-expert bytes (first routed layer). With a mixed-precision GGUF this
     * pins the class to the majority layers so the boosted ones are rejected
     * deterministically from startup, instead of depending on which layer
     * happens to touch the cache first.
     */
    g_stream_expert_cache_expert_bytes = bytes;
}

uint64_t ds4_gpu_recommended_working_set_size(void) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!g_device) return 0;
    return (uint64_t)[g_device recommendedMaxWorkingSetSize];
}

static int ds4_gpu_model_map_log_enabled(void) {
    if (!g_ssd_streaming_mode) return 1;
    const char *trace = getenv("DS4_METAL_STREAMING_MAP_TRACE");
    return trace && trace[0] && strcmp(trace, "0") != 0;
}

static id<MTLBuffer> ds4_gpu_wrap_model_range(
        const void *model_map,
        uint64_t    model_size,
        uint64_t    offset,
        uint64_t    len,
        uint64_t   *inner_offset);

static id<MTLBuffer> ds4_gpu_wrap_model_exact_range(
        const void *model_map,
        uint64_t    model_size,
        uint64_t    offset,
        uint64_t    len,
        uint64_t   *inner_offset);

static const char *ds4_gpu_source =
"#include <metal_stdlib>\n"
"#ifdef DS4_METAL_HAS_TENSOR\n"
"#include <metal_tensor>\n"
"#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n"
"#endif\n"
"using namespace metal;\n"
"#ifdef DS4_METAL_HAS_TENSOR\n"
"using namespace mpp::tensor_ops;\n"
"#endif\n"
"\n"
"#define MAX(x, y) ((x) > (y) ? (x) : (y))\n"
"#define MIN(x, y) ((x) < (y) ? (x) : (y))\n"
"#define SWAP(x, y) { auto tmp = (x); (x) = (y); (y) = tmp; }\n"
"#define QK8_0 32\n"
"#ifndef QK_K\n"
"#define QK_K 256\n"
"#endif\n"
"#define N_SIMDWIDTH 32\n"
"#define N_R0_Q8_0 2\n"
"#define N_SG_Q8_0 4\n"
"#define FC_MUL_MV 600\n"
"#define FC_MUL_MM 700\n"
"#define FC_BIN 1300\n"
"#define FOR_UNROLL(x) _Pragma(\"clang loop unroll(full)\") for (x)\n"
"#define M_PI_F 3.14159265358979323846f\n"
"\n"
"// Reads one byte per stride to warm model-backed pages without copying the\n"
"// model. This is outside inference and exists only to reduce first-use stalls.\n"
"kernel void kernel_touch_u8_stride(\n"
"        device const uchar    *src        [[buffer(0)]],\n"
"        device uchar          *dst        [[buffer(1)]],\n"
"        constant ulong        &stride     [[buffer(2)]],\n"
"        constant ulong        &bytes      [[buffer(3)]],\n"
"        constant ulong        &dst_offset [[buffer(4)]],\n"
"        uint gid [[thread_position_in_grid]]) {\n"
"    ulong off = (ulong)gid * stride;\n"
"    if (off >= bytes) return;\n"
"    dst[dst_offset + (ulong)gid] = src[off];\n"
"}\n"
"\n"
"enum ds4_sort_order {\n"
"    DS4_SORT_ORDER_ASC,\n"
"    DS4_SORT_ORDER_DESC,\n"
"};\n"
"\n"
"struct block_q8_0 {\n"
"    half d;\n"
"    int8_t qs[QK8_0];\n"
"};\n"
"\n"
"struct block_q8_K {\n"
"    float d;\n"
"    int8_t qs[QK_K];\n"
"    int16_t bsums[QK_K / 16];\n"
"};\n"
"\n"
"\n";

static NSString *ds4_gpu_full_source(void) {
    NSString *base = [NSString stringWithUTF8String:ds4_gpu_source];
    NSFileManager *fm = [NSFileManager defaultManager];
    /*
     * Kernels are kept as separate files for review, then concatenated into one
     * Metal library.  Environment overrides are still honored so a diagnostic
     * run can swap one source file without changing the executable.
     */
    NSArray<NSArray<NSString *> *> *required_sources = @[
        @[@"DS4_METAL_FLASH_ATTN_SOURCE", @"metal/flash_attn.metal"],
        @[@"DS4_METAL_DENSE_SOURCE",      @"metal/dense.metal"],
        @[@"DS4_METAL_MOE_SOURCE",        @"metal/moe.metal"],
        @[@"DS4_METAL_DSV4_HC_SOURCE",    @"metal/dsv4_hc.metal"],
        @[@"DS4_METAL_UNARY_SOURCE",      @"metal/unary.metal"],
        @[@"DS4_METAL_DSV4_KV_SOURCE",    @"metal/dsv4_kv.metal"],
        @[@"DS4_METAL_DSV4_ROPE_SOURCE",  @"metal/dsv4_rope.metal"],
        @[@"DS4_METAL_DSV4_MISC_SOURCE",  @"metal/dsv4_misc.metal"],
        @[@"DS4_METAL_ARGSORT_SOURCE",    @"metal/argsort.metal"],
        @[@"DS4_METAL_CPY_SOURCE",        @"metal/cpy.metal"],
        @[@"DS4_METAL_CONCAT_SOURCE",     @"metal/concat.metal"],
        @[@"DS4_METAL_GET_ROWS_SOURCE",   @"metal/get_rows.metal"],
        @[@"DS4_METAL_SUM_ROWS_SOURCE",   @"metal/sum_rows.metal"],
        @[@"DS4_METAL_SOFTMAX_SOURCE",    @"metal/softmax.metal"],
        @[@"DS4_METAL_REPEAT_SOURCE",     @"metal/repeat.metal"],
        @[@"DS4_METAL_GLU_SOURCE",        @"metal/glu.metal"],
        @[@"DS4_METAL_NORM_SOURCE",       @"metal/norm.metal"],
        @[@"DS4_METAL_BIN_SOURCE",        @"metal/bin.metal"],
        @[@"DS4_METAL_SET_ROWS_SOURCE",   @"metal/set_rows.metal"],
    ];

    NSMutableString *source = [NSMutableString stringWithString:base];
    for (NSArray<NSString *> *spec in required_sources) {
        const char *override_path = getenv([spec[0] UTF8String]);
        NSMutableArray<NSString *> *paths = [NSMutableArray array];
        if (override_path && override_path[0]) {
            [paths addObject:[NSString stringWithUTF8String:override_path]];
        }
        [paths addObject:spec[1]];
        [paths addObject:[@"./" stringByAppendingString:spec[1]]];

        NSString *loaded = nil;
        NSString *loaded_path = nil;
        for (NSString *path in paths) {
            if (![fm fileExistsAtPath:path]) continue;

            NSError *error = nil;
            loaded = [NSString stringWithContentsOfFile:path
                                               encoding:NSUTF8StringEncoding
                                                  error:&error];
            if (!loaded) {
                fprintf(stderr, "ds4: failed to read Metal source %s: %s\n",
                        [path UTF8String], [[error localizedDescription] UTF8String]);
                return nil;
            }
            loaded_path = path;
            break;
        }

        if (!loaded) {
            fprintf(stderr,
                    "ds4: Metal source %s not found (set %s to override)\n",
                    [spec[1] UTF8String], [spec[0] UTF8String]);
            return nil;
        }
        [source appendFormat:@"\n// appended %@\n%@\n", loaded_path, loaded];
    }
    return source;
}

typedef struct {
    int32_t  ne00t;
    int32_t  ne00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne10;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
} ds4_gpu_get_rows_args;

typedef struct {
    int32_t  n_embd;
    int32_t  n_vocab;
    int32_t  n_tokens;
    uint64_t src_row_bytes;
    uint64_t dst_row_bytes;
    uint64_t token_stride;
} ds4_gpu_get_rows_q8_0_args;

typedef struct {
    int32_t  ne00;
    int32_t  ne01;
    int32_t  ne02;
    int32_t  ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne0;
    int32_t  ne1;
    int32_t  ne2;
    int32_t  ne3;
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
} ds4_gpu_repeat_args;

typedef struct {
    int32_t  nk0;
    int32_t  ne01;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne11;
    int32_t  ne12;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
} ds4_gpu_set_rows_args;

typedef struct {
    int32_t  ne00;
    int32_t  ne01;
    int32_t  ne02;
    int32_t  ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne10;
    int32_t  ne11;
    int32_t  ne12;
    int32_t  ne13;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    int32_t  ne0;
    int32_t  ne1;
    int32_t  ne2;
    int32_t  ne3;
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
    int32_t  dim;
} ds4_gpu_concat_args;

typedef struct {
    int64_t  nk0;
    int64_t  ne00;
    int64_t  ne01;
    int64_t  ne02;
    int64_t  ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int64_t  ne0;
    int64_t  ne1;
    int64_t  ne2;
    int64_t  ne3;
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
} ds4_gpu_cpy_args;

static ds4_gpu_cpy_args ds4_gpu_make_cpy_1d_args(
        uint32_t n,
        uint64_t src_elem,
        uint64_t dst_elem) {
    return (ds4_gpu_cpy_args) {
        .nk0 = (int64_t)n,
        .ne00 = (int64_t)n,
        .ne01 = 1,
        .ne02 = 1,
        .ne03 = 1,
        .nb00 = src_elem,
        .nb01 = (uint64_t)n * src_elem,
        .nb02 = (uint64_t)n * src_elem,
        .nb03 = (uint64_t)n * src_elem,
        .ne0 = (int64_t)n,
        .ne1 = 1,
        .ne2 = 1,
        .ne3 = 1,
        .nb0 = dst_elem,
        .nb1 = (uint64_t)n * dst_elem,
        .nb2 = (uint64_t)n * dst_elem,
        .nb3 = (uint64_t)n * dst_elem,
    };
}

static NSUInteger ds4_gpu_cpy_threads(uint32_t n, id<MTLComputePipelineState> pipeline) {
    NSUInteger nth = 32u;
    const NSUInteger max_threads = pipeline.maxTotalThreadsPerThreadgroup;
    while (nth < (NSUInteger)n && nth < max_threads) nth *= 2u;
    if (nth > max_threads) nth = max_threads;
    if (nth > (NSUInteger)n) nth = (NSUInteger)n;
    return nth ? nth : 1u;
}

static float ds4_gpu_negative_infinity(void) {
    union { uint32_t u; float f; } v = { 0xff800000u };
    return v.f;
}

static float ds4_gpu_positive_infinity(void) {
    union { uint32_t u; float f; } v = { 0x7f800000u };
    return v.f;
}

static int ds4_gpu_encode_cpy_f32_f32_1d(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        src,
        NSUInteger           src_off,
        id<MTLBuffer>        dst,
        NSUInteger           dst_off,
        uint32_t             n);

static int ds4_gpu_encode_cpy_f32_f32_3d(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        src,
        NSUInteger           src_off,
        id<MTLBuffer>        dst,
        NSUInteger           dst_off,
        uint32_t             cols,
        uint32_t             rows,
        uint32_t             planes,
        uint64_t             src_row_stride,
        uint64_t             src_plane_stride,
        uint64_t             dst_row_stride,
        uint64_t             dst_plane_stride);

static int ds4_gpu_encode_cpy_f32_f32_3d_src_strided(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        src,
        NSUInteger           src_off,
        id<MTLBuffer>        dst,
        NSUInteger           dst_off,
        uint32_t             cols,
        uint32_t             rows,
        uint32_t             planes,
        uint64_t             src_col_stride,
        uint64_t             src_row_stride,
        uint64_t             src_plane_stride,
        uint64_t             dst_row_stride,
        uint64_t             dst_plane_stride);

static int ds4_gpu_encode_cpy_f32_f16_1d(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        src,
        NSUInteger           src_off,
        id<MTLBuffer>        dst,
        NSUInteger           dst_off,
        uint32_t             n);

static int ds4_gpu_encode_cpy_f32_f16_2d(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        src,
        NSUInteger           src_off,
        id<MTLBuffer>        dst,
        NSUInteger           dst_off,
        uint32_t             cols,
        uint32_t             rows,
        uint64_t             src_row_stride,
        uint64_t             dst_row_stride);

static int ds4_gpu_encode_cpy_f16_f32_1d(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        src,
        NSUInteger           src_off,
        id<MTLBuffer>        dst,
        NSUInteger           dst_off,
        uint32_t             n);

static int ds4_gpu_encode_fill_f32_rows(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        buf,
        NSUInteger           offset,
        uint32_t             width,
        uint32_t             rows,
        float                value);

static int ds4_gpu_encode_add_f32_1d(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        a,
        NSUInteger           a_off,
        id<MTLBuffer>        b,
        NSUInteger           b_off,
        id<MTLBuffer>        out,
        NSUInteger           out_off,
        uint32_t             n);

typedef struct {
    int32_t  ne00;
    uint64_t nb01;
    int32_t  ne10;
    uint64_t nb11;
    int32_t  ne0;
    uint64_t nb1;
    int32_t  i00;
    int32_t  i10;
    float    alpha;
    float    limit;
} ds4_gpu_glu_args;

typedef struct {
    uint32_t n;
} ds4_gpu_add_flat_args;

typedef struct {
    int32_t  ne00;
    int32_t  ne01;
    int32_t  ne02;
    int32_t  ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne10;
    int32_t  ne11;
    int32_t  ne12;
    int32_t  ne13;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    int32_t  ne0;
    int32_t  ne1;
    int32_t  ne2;
    int32_t  ne3;
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
    uint64_t offs;
    uint64_t o1[8];
} ds4_gpu_bin_args;

typedef struct {
    int32_t  ne00;
    int32_t  ne01;
    int32_t  ne02;
    int32_t  ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne0;
    int32_t  ne1;
    int32_t  ne2;
    int32_t  ne3;
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
    float    slope;
    float    scale;
    float    bias;
    float    val;
    float    min;
    float    max;
} ds4_gpu_unary_args;

static ds4_gpu_bin_args ds4_gpu_make_bin_rows_args(uint32_t n, uint32_t rows, uint32_t rhs_n) {
    const uint64_t row_bytes = (uint64_t)n * sizeof(float);
    const uint64_t rhs_row_bytes = (uint64_t)rhs_n * sizeof(float);
    return (ds4_gpu_bin_args) {
        .ne00 = (int32_t)n,
        .ne01 = (int32_t)rows,
        .ne02 = 1,
        .ne03 = 1,
        .nb00 = sizeof(float),
        .nb01 = row_bytes,
        .nb02 = row_bytes,
        .nb03 = row_bytes,
        .ne10 = (int32_t)rhs_n,
        .ne11 = 1,
        .ne12 = 1,
        .ne13 = 1,
        .nb10 = sizeof(float),
        .nb11 = rhs_row_bytes,
        .nb12 = rhs_row_bytes,
        .nb13 = rhs_row_bytes,
        .ne0 = (int32_t)n,
        .ne1 = (int32_t)rows,
        .ne2 = 1,
        .ne3 = 1,
        .nb0 = sizeof(float),
        .nb1 = row_bytes,
        .nb2 = row_bytes,
        .nb3 = row_bytes,
        .offs = 0,
        .o1 = { 0 },
    };
}

static ds4_gpu_unary_args ds4_gpu_make_unary_rows_args(
        uint32_t n,
        uint32_t rows,
        int      c4,
        float    scale,
        float    bias) {
    const uint64_t row_bytes = (uint64_t)n * sizeof(float);
    const uint32_t n_kernel = c4 ? n / 4u : n;
    return (ds4_gpu_unary_args) {
        .ne00 = (int32_t)n_kernel,
        .ne01 = (int32_t)rows,
        .ne02 = 1,
        .ne03 = 1,
        .nb00 = sizeof(float),
        .nb01 = row_bytes,
        .nb02 = row_bytes,
        .nb03 = row_bytes,
        .ne0 = (int32_t)n_kernel,
        .ne1 = (int32_t)rows,
        .ne2 = 1,
        .ne3 = 1,
        .nb0 = sizeof(float),
        .nb1 = row_bytes,
        .nb2 = row_bytes,
        .nb3 = row_bytes,
        .slope = 0.0f,
        .scale = scale,
        .bias = bias,
        .val = 0.0f,
        .min = 0.0f,
        .max = 0.0f,
    };
}

static ds4_gpu_bin_args ds4_gpu_make_bin_same_rows_args(uint32_t n, uint32_t rows) {
    const uint64_t row_bytes = (uint64_t)n * sizeof(float);
    return (ds4_gpu_bin_args) {
        .ne00 = (int32_t)n,
        .ne01 = (int32_t)rows,
        .ne02 = 1,
        .ne03 = 1,
        .nb00 = sizeof(float),
        .nb01 = row_bytes,
        .nb02 = (uint64_t)rows * row_bytes,
        .nb03 = (uint64_t)rows * row_bytes,
        .ne10 = (int32_t)n,
        .ne11 = (int32_t)rows,
        .ne12 = 1,
        .ne13 = 1,
        .nb10 = sizeof(float),
        .nb11 = row_bytes,
        .nb12 = (uint64_t)rows * row_bytes,
        .nb13 = (uint64_t)rows * row_bytes,
        .ne0 = (int32_t)n,
        .ne1 = (int32_t)rows,
        .ne2 = 1,
        .ne3 = 1,
        .nb0 = sizeof(float),
        .nb1 = row_bytes,
        .nb2 = (uint64_t)rows * row_bytes,
        .nb3 = (uint64_t)rows * row_bytes,
        .offs = 0,
        .o1 = { 0 },
    };
}

static int ds4_gpu_encode_bin_f32_rows(
        id<MTLCommandBuffer>        cb,
        id<MTLComputePipelineState> pipeline,
        const ds4_gpu_bin_args   *args,
        id<MTLBuffer>               a,
        NSUInteger                  a_off,
        id<MTLBuffer>               b,
        NSUInteger                  b_off,
        id<MTLBuffer>               out,
        NSUInteger                  out_off);

static int ds4_gpu_encode_sum_rows_f32(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        src,
        NSUInteger           src_off,
        id<MTLBuffer>        dst,
        NSUInteger           dst_off,
        uint32_t             width,
        uint32_t             rows);

typedef struct {
    int32_t  ne00;
    int32_t  ne01;
    int32_t  ne02;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne10;
    int32_t  ne11;
    int32_t  ne12;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    int32_t  ne0;
    int32_t  ne1;
    int32_t  nr0;
    int16_t  r2;
    int16_t  r3;
} ds4_gpu_q8_0_matvec_args;

typedef struct {
    int32_t  ne00;
    int32_t  ne02;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne12;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    int32_t  ne0;
    int32_t  ne1;
    int16_t  r2;
    int16_t  r3;
} ds4_gpu_mul_mm_args;

typedef struct {
    int32_t  ne00;
    int32_t  ne01;
    int32_t  ne02;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne10;
    int32_t  ne11;
    int32_t  ne12;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    int32_t  ne0;
    int32_t  ne1;
    int16_t  r2;
    int16_t  r3;
} ds4_gpu_mul_mv_ext_args;

typedef ds4_gpu_q8_0_matvec_args ds4_gpu_f16_matvec_args;

static ds4_gpu_q8_0_matvec_args ds4_gpu_make_q8_0_mv_args(uint64_t in_dim, uint64_t out_dim) {
    const uint64_t row_bytes = (in_dim / 32u) * 34u;
    return (ds4_gpu_q8_0_matvec_args) {
        .ne00 = (int32_t)in_dim,
        .ne01 = (int32_t)out_dim,
        .ne02 = 1,
        .nb00 = 34,
        .nb01 = row_bytes,
        .nb02 = row_bytes * out_dim,
        .nb03 = row_bytes * out_dim,
        .ne10 = (int32_t)in_dim,
        .ne11 = 1,
        .ne12 = 1,
        .nb10 = sizeof(float),
        .nb11 = in_dim * sizeof(float),
        .nb12 = in_dim * sizeof(float),
        .nb13 = in_dim * sizeof(float),
        .ne0 = (int32_t)out_dim,
        .ne1 = 1,
        .nr0 = 2,
        .r2 = 1,
        .r3 = 1,
    };
}

static ds4_gpu_f16_matvec_args ds4_gpu_make_f16_mv_args(uint64_t in_dim, uint64_t out_dim) {
    const uint64_t row_bytes = in_dim * sizeof(uint16_t);
    return (ds4_gpu_f16_matvec_args) {
        .ne00 = (int32_t)in_dim,
        .ne01 = (int32_t)out_dim,
        .ne02 = 1,
        .nb00 = sizeof(uint16_t),
        .nb01 = row_bytes,
        .nb02 = row_bytes * out_dim,
        .nb03 = row_bytes * out_dim,
        .ne10 = (int32_t)in_dim,
        .ne11 = 1,
        .ne12 = 1,
        .nb10 = sizeof(float),
        .nb11 = in_dim * sizeof(float),
        .nb12 = in_dim * sizeof(float),
        .nb13 = in_dim * sizeof(float),
        .ne0 = (int32_t)out_dim,
        .ne1 = 1,
        .nr0 = 2,
        .r2 = 1,
        .r3 = 1,
    };
}

static ds4_gpu_q8_0_matvec_args ds4_gpu_make_f32_mv_args(
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t n_vec) {
    const uint64_t row_bytes = in_dim * sizeof(float);
    return (ds4_gpu_q8_0_matvec_args) {
        .ne00 = (int32_t)in_dim,
        .ne01 = (int32_t)out_dim,
        .ne02 = 1,
        .nb00 = sizeof(float),
        .nb01 = row_bytes,
        .nb02 = row_bytes * out_dim,
        .nb03 = row_bytes * out_dim,
        .ne10 = (int32_t)in_dim,
        .ne11 = (int32_t)n_vec,
        .ne12 = 1,
        .nb10 = sizeof(float),
        .nb11 = in_dim * sizeof(float),
        .nb12 = in_dim * n_vec * sizeof(float),
        .nb13 = in_dim * n_vec * sizeof(float),
        .ne0 = (int32_t)out_dim,
        .ne1 = (int32_t)n_vec,
        .nr0 = 2,
        .r2 = 1,
        .r3 = 1,
    };
}

typedef struct {
    const char *function_name;
    int16_t     nsg;
    int32_t     nr0;
    NSUInteger  smem;
} ds4_gpu_mv_dispatch;

static int ds4_gpu_tp_world_is_two(void);

static ds4_gpu_mv_dispatch ds4_gpu_make_q8_0_mv_dispatch(void) {
    const uint64_t default_nsg = ds4_gpu_tp_world_is_two() ? 2u : 4u;
    const int16_t nsg =
        (int16_t)ds4_gpu_env_u64("DS4_METAL_Q8_MV_NSG", default_nsg, 1u, 8u);
    const uint64_t rows = ds4_gpu_env_u64("DS4_METAL_Q8_MV_ROWS", 2u, 2u, 4u);
    if (rows >= 4u) {
        return (ds4_gpu_mv_dispatch) {
            .function_name = "kernel_mul_mv_q8_0_f32_r4",
            .nsg = nsg,
            .nr0 = 4,
            .smem = 32u * 4u * sizeof(float),
        };
    }
    return (ds4_gpu_mv_dispatch) {
        .function_name = "kernel_mul_mv_q8_0_f32",
        .nsg = nsg,
        .nr0 = 2,
        .smem = 32u * 2u * sizeof(float),
    };
}

static ds4_gpu_mv_dispatch ds4_gpu_make_plain_mv_dispatch(
        uint64_t in_dim,
        int      f32_weights) {
    if (in_dim < 32) {
        return (ds4_gpu_mv_dispatch) {
            .function_name = f32_weights ? "kernel_mul_mv_f32_f32_short" : "kernel_mul_mv_f16_f32_short",
            .nsg = 1,
            .nr0 = 32,
            .smem = 0,
        };
    }

    const int16_t nsg = (int16_t)((in_dim + 127u) / 128u > 8u ? 8u : (in_dim + 127u) / 128u);
    const int use_4 = (in_dim % 4u) == 0;
    return (ds4_gpu_mv_dispatch) {
        .function_name = f32_weights
            ? (use_4 ? "kernel_mul_mv_f32_f32_4" : "kernel_mul_mv_f32_f32")
            : (use_4 ? "kernel_mul_mv_f16_f32_4" : "kernel_mul_mv_f16_f32"),
        .nsg = nsg,
        .nr0 = 2,
        .smem = 32u * 2u * sizeof(float),
    };
}

static ds4_gpu_mul_mm_args ds4_gpu_make_mm_args(
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t n_tok,
        uint64_t row_bytes) {
    return (ds4_gpu_mul_mm_args) {
        .ne00 = (int32_t)in_dim,
        .ne02 = 1,
        .nb01 = row_bytes,
        .nb02 = row_bytes * out_dim,
        .nb03 = row_bytes * out_dim,
        .ne12 = 1,
        .nb10 = sizeof(float),
        .nb11 = in_dim * sizeof(float),
        .nb12 = in_dim * n_tok * sizeof(float),
        .nb13 = in_dim * n_tok * sizeof(float),
        .ne0 = (int32_t)out_dim,
        .ne1 = (int32_t)n_tok,
        .r2 = 1,
        .r3 = 1,
    };
}

static ds4_gpu_mul_mv_ext_args ds4_gpu_make_mv_ext_args(
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t n_tok,
        uint64_t elem_bytes,
        uint64_t row_bytes) {
    return (ds4_gpu_mul_mv_ext_args) {
        .ne00 = (int32_t)in_dim,
        .ne01 = (int32_t)out_dim,
        .ne02 = 1,
        .nb00 = elem_bytes,
        .nb01 = row_bytes,
        .nb02 = row_bytes * out_dim,
        .nb03 = row_bytes * out_dim,
        .ne10 = (int32_t)in_dim,
        .ne11 = (int32_t)n_tok,
        .ne12 = 1,
        .nb10 = sizeof(float),
        .nb11 = in_dim * sizeof(float),
        .nb12 = in_dim * n_tok * sizeof(float),
        .nb13 = in_dim * n_tok * sizeof(float),
        .ne0 = (int32_t)out_dim,
        .ne1 = (int32_t)n_tok,
        .r2 = 1,
        .r3 = 1,
    };
}

static int16_t ds4_gpu_mv_ext_nxpsg(uint64_t in_dim, uint64_t n_tok) {
    if ((in_dim % 256u) == 0 && n_tok < 3) return 16;
    if ((in_dim % 128u) == 0) return 8;
    return 4;
}

static int16_t ds4_gpu_mv_ext_r1ptg(uint64_t n_tok) {
    switch (n_tok) {
    case 2: return 2;
    case 3:
    case 6: return 3;
    case 4:
    case 7:
    case 8: return 4;
    case 5: return 5;
    default: return n_tok > 8 ? 4 : 0;
    }
}

static const char *ds4_gpu_mv_ext_name(int q8, int16_t r1ptg) {
    if (q8) {
        switch (r1ptg) {
        case 2: return "kernel_mul_mv_ext_q8_0_f32_r1_2";
        case 3: return "kernel_mul_mv_ext_q8_0_f32_r1_3";
        case 4: return "kernel_mul_mv_ext_q8_0_f32_r1_4";
        case 5: return "kernel_mul_mv_ext_q8_0_f32_r1_5";
        default: return NULL;
        }
    }

    switch (r1ptg) {
    case 2: return "kernel_mul_mv_ext_f16_f32_r1_2";
    case 3: return "kernel_mul_mv_ext_f16_f32_r1_3";
    case 4: return "kernel_mul_mv_ext_f16_f32_r1_4";
    case 5: return "kernel_mul_mv_ext_f16_f32_r1_5";
    default: return NULL;
    }
}

static const char *ds4_gpu_mv_ext_f32_name(int16_t r1ptg) {
    switch (r1ptg) {
    case 2: return "kernel_mul_mv_ext_f32_f32_r1_2";
    case 3: return "kernel_mul_mv_ext_f32_f32_r1_3";
    case 4: return "kernel_mul_mv_ext_f32_f32_r1_4";
    case 5: return "kernel_mul_mv_ext_f32_f32_r1_5";
    default: return NULL;
    }
}

static const char *ds4_gpu_mv_ext_q8_pair_swiglu_name(int16_t r1ptg) {
    switch (r1ptg) {
    case 2: return "kernel_mul_mv_ext_q8_0_pair_swiglu_f32_r1_2";
    case 3: return "kernel_mul_mv_ext_q8_0_pair_swiglu_f32_r1_3";
    case 4: return "kernel_mul_mv_ext_q8_0_pair_swiglu_f32_r1_4";
    case 5: return "kernel_mul_mv_ext_q8_0_pair_swiglu_f32_r1_5";
    default: return NULL;
    }
}

typedef struct {
    int32_t  ne00;
    int32_t  ne00_t;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
    float    eps;
    int32_t  nef1[3];
    int32_t  nef2[3];
    int32_t  nef3[3];
    uint64_t nbf1[3];
    uint64_t nbf2[3];
    uint64_t nbf3[3];
} ds4_gpu_rms_norm_args;

typedef struct {
    int32_t  q_n;
    int32_t  q_n4;
    int32_t  kv_n;
    int32_t  kv_n4;
    uint64_t q_row_stride;
    uint64_t kv_row_stride;
    float    eps;
} ds4_gpu_qkv_rms_norm_args;

static ds4_gpu_rms_norm_args ds4_gpu_make_rms_norm_args(uint32_t n, uint32_t rows, float eps) {
    const uint64_t row_bytes = (uint64_t)n * sizeof(float);
    return (ds4_gpu_rms_norm_args) {
        .ne00 = (int32_t)n,
        .ne00_t = (int32_t)(n / 4u),
        .nb1 = row_bytes,
        .nb2 = row_bytes * rows,
        .nb3 = row_bytes * rows,
        .eps = eps,
        .nef1 = { (int32_t)rows, 1, 1 },
        .nef2 = { 1, 1, 1 },
        .nef3 = { 1, 1, 1 },
        .nbf1 = { row_bytes, row_bytes, row_bytes },
        .nbf2 = { row_bytes * rows, row_bytes, row_bytes },
        .nbf3 = { row_bytes * rows, row_bytes, row_bytes },
    };
}

static ds4_gpu_rms_norm_args ds4_gpu_make_rms_norm_3d_args(
        uint32_t n0,
        uint32_t n1,
        uint32_t n2,
        float    eps) {
    const uint64_t row_bytes = (uint64_t)n0 * sizeof(float);
    const uint64_t plane_bytes = row_bytes * n1;
    return (ds4_gpu_rms_norm_args) {
        .ne00 = (int32_t)n0,
        .ne00_t = (int32_t)(n0 / 4u),
        .nb1 = row_bytes,
        .nb2 = plane_bytes,
        .nb3 = plane_bytes * n2,
        .eps = eps,
        .nef1 = { (int32_t)n1, 1, 1 },
        .nef2 = { (int32_t)n2, 1, 1 },
        .nef3 = { 1, 1, 1 },
        .nbf1 = { row_bytes, row_bytes, row_bytes },
        .nbf2 = { plane_bytes, row_bytes, row_bytes },
        .nbf3 = { plane_bytes * n2, row_bytes, row_bytes },
    };
}

static NSUInteger ds4_gpu_rms_norm_threads(uint32_t n) {
    NSUInteger ne00_t = n / 4u;
    NSUInteger nth = 32u;
    while (nth < ne00_t && nth < 1024u) nth *= 2u;
    if (nth > ne00_t) nth = ne00_t;
    return nth ? nth : 1u;
}

static NSUInteger ds4_gpu_rms_norm_pipeline_threads(
        uint32_t                  n,
        id<MTLComputePipelineState> pipeline) {
    NSUInteger ne00_t = n / 4u;
    NSUInteger max_threads = pipeline ? [pipeline maxTotalThreadsPerThreadgroup] : 1024u;
    NSUInteger nth = 32u;
    while (nth < ne00_t && nth < max_threads) nth *= 2u;
    if (nth > max_threads) nth = max_threads;
    if (nth > ne00_t) nth = ne00_t;
    return nth ? nth : 1u;
}

typedef struct {
    int32_t  n_hc;
    int32_t  sinkhorn_iters;
    int64_t  n_rows;
    int64_t  mix_hc;
    uint64_t nb01;
    uint64_t nb1;
    float    eps;
} ds4_gpu_hc_split_args;

typedef struct {
    int64_t n_embd;
    int64_t n_hc;
    int64_t n_tokens;
    uint64_t nb_x0;
    uint64_t nb_x1;
    uint64_t nb_x2;
    uint64_t nb_w0;
    uint64_t nb_w1;
    uint64_t nb0;
    uint64_t nb1;
} ds4_gpu_hc_weighted_sum_args;

typedef struct {
    int64_t n_embd;
    int64_t n_hc;
    int64_t n_tokens;
    uint64_t nb_x0;
    uint64_t nb_x1;
    uint64_t nb_x2;
    uint64_t nb_w0;
    uint64_t nb_w1;
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb_norm1;
    float norm_eps;
} ds4_gpu_hc_weighted_sum_norm_args;

typedef struct {
    float post_scale;
    float eps;
} ds4_gpu_output_hc_weights4_args;

typedef struct {
    int64_t n_embd;
    int32_t n_hc;
    int32_t sinkhorn_iters;
    int64_t n_rows;
    int64_t mix_hc;
    uint64_t nb_mix1;
    uint64_t nb_split1;
    uint64_t nb_x0;
    uint64_t nb_x1;
    uint64_t nb_x2;
    uint64_t nb0;
    uint64_t nb1;
    float eps;
} ds4_gpu_hc_split_weighted_sum_args;

typedef struct {
    int64_t n_embd;
    int32_t n_hc;
    int32_t sinkhorn_iters;
    int64_t n_rows;
    int64_t mix_hc;
    uint64_t nb_mix1;
    uint64_t nb_split1;
    uint64_t nb_x0;
    uint64_t nb_x1;
    uint64_t nb_x2;
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb_norm1;
    float eps;
    float norm_eps;
} ds4_gpu_hc_split_weighted_sum_norm_args;

typedef struct {
    int64_t n_embd;
    int64_t n_hc;
    int64_t n_tokens;
    uint64_t nb_block0;
    uint64_t nb_block1;
    uint64_t nb_add0;
    uint64_t nb_add1;
    uint64_t nb_res0;
    uint64_t nb_res1;
    uint64_t nb_res2;
    uint64_t nb_post0;
    uint64_t nb_post1;
    uint64_t nb_comb0;
    uint64_t nb_comb1;
    uint64_t nb_comb2;
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb2;
    int32_t has_add;
} ds4_gpu_hc_expand_args;

typedef struct {
    int32_t  nei0;
    int32_t  nei1;
    uint64_t nbi1;
    int32_t  ne00;
    int32_t  ne01;
    int32_t  ne02;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    int32_t  ne10;
    int32_t  ne11;
    int32_t  ne12;
    int32_t  ne13;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    int32_t  ne0;
    int32_t  ne1;
    uint64_t nb1;
    int32_t  nr0;
    /* Tensor-parallel expert ownership; see ds4_metal_args_mul_mv_id in
     * metal/moe.metal.  Zero (from struct literals) means no split. */
    int32_t  tp_rank;
    int32_t  tp_world;
    int32_t  tp_addend;
    int32_t  tp_expert_base;
} ds4_gpu_mul_mv_id_args;

typedef struct {
    uint32_t n_total_expert;
    uint32_t n_expert;
} ds4_gpu_stream_expert_validate_args;

typedef struct {
    uint32_t active_mask;
    uint32_t accumulate;
} ds4_gpu_stream_expert_split_args;

typedef struct {
    int32_t  ne02;
    int32_t  ne10;
    int32_t  ne11;
    uint64_t nb11;
    uint64_t nb12;
    int32_t  ne21;
    int32_t  ne20;
    uint64_t nb21;
} ds4_gpu_mul_mm_id_map_args;

typedef struct {
    int32_t  ne00;
    int32_t  ne02;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne11;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    int32_t  ne20;
    int32_t  ne21;
    int32_t  ne0;
    int32_t  ne1;
    int16_t  r2;
    int16_t  r3;
    int32_t  tp_rank;
    int32_t  tp_world;
    int32_t  tp_expert_base;
} ds4_gpu_mul_mm_id_args;

static int ds4_gpu_encode_mul_mv_id(
        id<MTLCommandBuffer>        cb,
        id<MTLComputePipelineState> pipeline,
        const ds4_gpu_mul_mv_id_args *args,
        id<MTLBuffer>               src0,
        NSUInteger                  src0_off,
        id<MTLBuffer>               src1,
        NSUInteger                  src1_off,
        id<MTLBuffer>               dst,
        NSUInteger                  dst_off,
        id<MTLBuffer>               ids,
        NSUInteger                  ids_off,
        NSUInteger                  threadgroup_bytes,
        NSUInteger                  nsg,
        bool                        rows_per_group_is_nr0);

static int ds4_gpu_encode_attn_out_low_q8_direct(
        id<MTLCommandBuffer>        cb,
        id<MTLComputePipelineState> pipeline,
        const ds4_gpu_mul_mv_id_args *args,
        id<MTLBuffer>               src0,
        NSUInteger                  src0_off,
        id<MTLBuffer>               src1,
        NSUInteger                  src1_off,
        id<MTLBuffer>               dst,
        NSUInteger                  dst_off,
        NSUInteger                  threadgroup_bytes,
        NSUInteger                  nsg,
        bool                        rows_per_group_is_nr0);

static int ds4_gpu_encode_attn_out_low_q8_mpp(
        id<MTLCommandBuffer>           cb,
        id<MTLComputePipelineState>    pipeline,
        const ds4_gpu_mul_mm_id_args *mm_args,
        id<MTLBuffer>                  src0,
        NSUInteger                     src0_off,
        id<MTLBuffer>                  src1,
        NSUInteger                     src1_off,
        id<MTLBuffer>                  dst,
        NSUInteger                     dst_off);

static int ds4_gpu_encode_attn_out_low_q8_mpp(
        id<MTLCommandBuffer>           cb,
        id<MTLComputePipelineState>    pipeline,
        const ds4_gpu_mul_mm_id_args *mm_args,
        id<MTLBuffer>                  src0,
        NSUInteger                     src0_off,
        id<MTLBuffer>                  src1,
        NSUInteger                     src1_off,
        id<MTLBuffer>                  dst,
        NSUInteger                     dst_off);

static ds4_gpu_mul_mm_id_map_args ds4_gpu_make_mul_mm_id_map_args(
        uint32_t src0_cols,
        uint32_t src0_experts,
        uint32_t src1_expert_rows,
        uint32_t selected_experts,
        uint32_t n_tokens);

static ds4_gpu_mul_mm_id_args ds4_gpu_make_mul_mm_id_args(
        uint32_t src0_cols,
        uint32_t src0_rows,
        uint32_t src0_experts,
        uint64_t src0_row_bytes,
        uint64_t src0_expert_bytes,
        uint32_t src1_expert_rows,
        uint32_t selected_experts,
        uint32_t n_tokens);
static ds4_gpu_mul_mm_id_args ds4_gpu_make_mul_mm_id_args_src1_size(
        uint32_t src0_cols,
        uint32_t src0_rows,
        uint32_t src0_experts,
        uint64_t src0_row_bytes,
        uint64_t src0_expert_bytes,
        uint32_t src1_expert_rows,
        uint32_t selected_experts,
        uint32_t n_tokens,
        uint32_t src1_elem_size);

static int ds4_gpu_encode_mul_mm_id(
        id<MTLCommandBuffer>        cb,
        id<MTLComputePipelineState> map_pipeline,
        id<MTLComputePipelineState> mm_pipeline,
        const ds4_gpu_mul_mm_id_map_args *map_args,
        const ds4_gpu_mul_mm_id_args *mm_args,
        id<MTLBuffer>               src0,
        NSUInteger                  src0_off,
        id<MTLBuffer>               src1,
        NSUInteger                  src1_off,
        id<MTLBuffer>               dst,
        NSUInteger                  dst_off,
        id<MTLBuffer>               ids,
        NSUInteger                  ids_off);

static int ds4_gpu_encode_mul_mm_id_map(
        id<MTLCommandBuffer>        cb,
        id<MTLComputePipelineState> map_pipeline,
        const ds4_gpu_mul_mm_id_map_args *map_args,
        const ds4_gpu_mul_mm_id_args *mm_args,
        id<MTLBuffer>               ids,
        NSUInteger                  ids_off);

static int ds4_gpu_encode_mul_mm_id_mapped(
        id<MTLCommandBuffer>        cb,
        id<MTLComputePipelineState> mm_pipeline,
        const ds4_gpu_mul_mm_id_args *mm_args,
        id<MTLBuffer>               src0,
        NSUInteger                  src0_off,
        id<MTLBuffer>               src1,
        NSUInteger                  src1_off,
        id<MTLBuffer>               dst,
        NSUInteger                  dst_off);
static int ds4_gpu_encode_mul_mm_id_mapped_tile(
        id<MTLCommandBuffer>        cb,
        id<MTLComputePipelineState> mm_pipeline,
        const ds4_gpu_mul_mm_id_args *mm_args,
        id<MTLBuffer>               src0,
        NSUInteger                  src0_off,
        id<MTLBuffer>               src1,
        NSUInteger                  src1_off,
        id<MTLBuffer>               dst,
        NSUInteger                  dst_off,
        NSUInteger                  threadgroup_bytes);
static int ds4_gpu_encode_mul_mm_id_addr_mapped_tile(
        id<MTLCommandBuffer>        cb,
        id<MTLComputePipelineState> mm_pipeline,
        const ds4_gpu_mul_mm_id_args *mm_args,
        id<MTLBuffer>               src0_addrs,
        id<MTLBuffer>               src1,
        NSUInteger                  src1_off,
        id<MTLBuffer>               dst,
        NSUInteger                  dst_off,
        NSUInteger                  threadgroup_bytes,
        ds4_gpu_stream_expert_cache_entry * const *resources,
        uint32_t                    resource_count,
        uint32_t                    resource_kind,
        id<MTLBuffer>               overflow_resource);

typedef struct {
    int32_t  ne11;
    int32_t  ne_12_2;
    int32_t  ne_12_3;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    uint64_t nb21;
    uint64_t nb22;
    uint64_t nb23;
    int32_t  ne31;
    int32_t  ne32;
    int32_t  ne33;
    uint64_t nb31;
    uint64_t nb32;
    uint64_t nb33;
} ds4_gpu_flash_attn_pad_args;

typedef struct {
    uint32_t raw_cap;
    uint32_t raw_start;
    uint32_t n_raw;
    uint32_t n_comp;
    uint32_t pad_rows;
    uint32_t shared_pad;
} ds4_gpu_flash_kv_stage_f16_args;

typedef struct {
    int32_t  ne01;
    int32_t  ne30;
    int32_t  ne31;
    int32_t  ne32;
    int32_t  ne33;
    uint64_t nb31;
    uint64_t nb32;
    uint64_t nb33;
} ds4_gpu_flash_attn_blk_args;

typedef struct {
    int32_t  ne01;
    int32_t  ne02;
    int32_t  ne03;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne11;
    int32_t  ne_12_2;
    int32_t  ne_12_3;
    int32_t  ns10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    int32_t  ns20;
    uint64_t nb21;
    uint64_t nb22;
    uint64_t nb23;
    int32_t  ne31;
    int32_t  ne32;
    int32_t  ne33;
    uint64_t nb31;
    uint64_t nb32;
    uint64_t nb33;
    int32_t  ne1;
    int32_t  ne2;
    int32_t  ne3;
    float    scale;
    float    max_bias;
    float    m0;
    float    m1;
    int32_t  n_head_log2;
    float    logit_softcap;
} ds4_gpu_flash_attn_vec_args;

typedef struct {
    int32_t nrows;
} ds4_gpu_flash_attn_reduce_args;

typedef struct {
    int64_t ne00;
    int64_t ne01;
    int64_t ne02;
    int64_t ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
    int32_t n_dims;
    int32_t mode;
    int32_t n_ctx_orig;
    int32_t inverse;
    float freq_base;
    float freq_scale;
    float ext_factor;
    float attn_factor;
    float beta_fast;
    float beta_slow;
    bool src2;
} ds4_gpu_rope_tail_batch_args;

typedef struct {
    uint64_t row_bytes;
    uint64_t token_bytes;
    int32_t head_dim;
    int32_t n_dims;
    int32_t n_ctx_orig;
    int32_t inverse;
    uint32_t pos0;
    uint32_t pos_step;
    float freq_base;
    float freq_scale;
    float ext_factor;
    float attn_factor;
    float beta_fast;
    float beta_slow;
} ds4_gpu_rope_affine_pair_args;

_Static_assert(sizeof(ds4_gpu_rope_affine_pair_args) == 64,
               "Metal affine RoPE argument ABI changed");

static ds4_gpu_rope_tail_batch_args ds4_gpu_make_rope_tail_args(
        uint32_t n_tok,
        uint32_t n_head,
        uint32_t head_dim,
        uint32_t n_rot,
        uint32_t n_ctx_orig,
        bool     inverse,
        float    freq_base,
        float    freq_scale,
        float    ext_factor,
        float    attn_factor,
        float    beta_fast,
        float    beta_slow) {
    const uint64_t row_bytes = (uint64_t)head_dim * sizeof(float);
    const uint64_t tok_bytes = (uint64_t)n_head * row_bytes;
    return (ds4_gpu_rope_tail_batch_args) {
        .ne00 = head_dim,
        .ne01 = n_head,
        .ne02 = n_tok,
        .ne03 = 1,
        .nb00 = sizeof(float),
        .nb01 = row_bytes,
        .nb02 = tok_bytes,
        .nb03 = (uint64_t)n_tok * tok_bytes,
        .nb0 = sizeof(float),
        .nb1 = row_bytes,
        .nb2 = tok_bytes,
        .nb3 = (uint64_t)n_tok * tok_bytes,
        .n_dims = (int32_t)n_rot,
        .mode = 0,
        .n_ctx_orig = (int32_t)n_ctx_orig,
        .inverse = inverse ? 1 : 0,
        .freq_base = freq_base,
        .freq_scale = freq_scale,
        .ext_factor = ext_factor,
        .attn_factor = attn_factor,
        .beta_fast = beta_fast,
        .beta_slow = beta_slow,
        .src2 = false,
    };
}

static int ds4_gpu_encode_rope_tail_inplace(
        id<MTLCommandBuffer>                 cb,
        id<MTLBuffer>                        xbuf,
        NSUInteger                           xoff,
        const ds4_gpu_rope_tail_batch_args *args,
        uint32_t                             n_tok,
        uint32_t                             n_head,
        uint32_t                             head_dim,
        uint32_t                             pos0,
        uint32_t                             pos_step) {
    const uint32_t tail_threads = args->n_dims > 0 ? (uint32_t)args->n_dims : 0u;
    const bool lane_compatible =
        tail_threads <= head_dim && ((head_dim - tail_threads) & 31u) == 0u;
    const bool force_affine_position =
        getenv("DS4_METAL_ENABLE_AFFINE_ROPE_PAIR") != NULL;
    const bool use_inplace_pair =
        g_rope_tail_inplace_pair_pipeline != nil &&
        args->mode == 0 && !args->src2 &&
        lane_compatible &&
        (ds4_gpu_device_name_contains("M3") ||
         (ds4_gpu_device_name_contains("M5") && n_tok == 1u) ||
         getenv("DS4_METAL_ENABLE_INPLACE_ROPE_PAIR") != NULL ||
         force_affine_position) &&
        getenv("DS4_METAL_DISABLE_M3_INPLACE_ROPE_PAIR") == NULL;
    const bool use_shared_coeff =
        use_inplace_pair && !force_affine_position &&
        g_rope_tail_inplace_pair_shared4_pipeline != nil &&
        /* The 256-thread grouped schedule helps long prefill, but reduces the
         * per-head parallelism that short batches and decode rely on. */
        tail_threads == 64u && n_head >= 4u && n_tok >= 32u &&
        getenv("DS4_METAL_DISABLE_M3_SHARED_ROPE_COEFF") == NULL;
    if (force_affine_position &&
        g_rope_tail_inplace_pair_affine_pipeline == nil &&
        getenv("DS4_METAL_DISABLE_M3_INPLACE_ROPE_PAIR") == NULL &&
        getenv("DS4_METAL_DISABLE_M3_AFFINE_ROPE_PAIR") == NULL) {
        fprintf(stderr,
                "ds4: forced affine-position RoPE pipeline is unavailable\n");
        return 0;
    }
    /* Keep long prefill on the proven shared4 kernel. Reconstructing affine
     * positions inside its coefficient cohort perturbs YaRN fast-math codegen;
     * the compact affine specialization is exact for the decode pair schedule. */
    const bool use_affine_position =
        use_inplace_pair && !use_shared_coeff &&
        g_rope_tail_inplace_pair_affine_pipeline != nil &&
        (n_tok == 1u || force_affine_position) &&
        (ds4_gpu_device_name_contains("M3") ||
         ds4_gpu_device_name_contains("M5") ||
         force_affine_position) &&
        getenv("DS4_METAL_DISABLE_M3_AFFINE_ROPE_PAIR") == NULL;

    int32_t pos_stack[256];
    int32_t *pos = NULL;
    id<MTLBuffer> posbuf = nil;
    const NSUInteger pos_bytes = (NSUInteger)n_tok * sizeof(int32_t);
    if (!use_affine_position) {
        pos = pos_stack;
        if (n_tok > (uint32_t)(sizeof(pos_stack) / sizeof(pos_stack[0]))) {
            pos = malloc((size_t)n_tok * sizeof(*pos));
            if (!pos) {
                fprintf(stderr, "ds4: failed to allocate Metal RoPE position buffer\n");
                return 0;
            }
        }
        for (uint32_t t = 0; t < n_tok; t++) {
            pos[t] = (int32_t)(pos0 + t * pos_step);
        }

        if (pos_bytes > 4096u) {
            /*
             * Metal inline setBytes data is meant for small constants. Long
             * prefill RoPE calls need thousands of positions; passing that much
             * inline can make the Apple driver abort the process.
             */
            posbuf = ds4_gpu_new_transient_buffer(
                pos_bytes, "ds4_rope_positions");
            if (!posbuf) {
                if (pos != pos_stack) free(pos);
                return 0;
            }
            memcpy([posbuf contents], pos, pos_bytes);
        }
    }

    ds4_gpu_rope_affine_pair_args affine_args;
    if (use_affine_position) {
        const uint64_t row_bytes = (uint64_t)head_dim * sizeof(float);
        affine_args = (ds4_gpu_rope_affine_pair_args) {
            .row_bytes = row_bytes,
            .token_bytes = (uint64_t)n_head * row_bytes,
            .head_dim = (int32_t)head_dim,
            .n_dims = args->n_dims,
            .n_ctx_orig = args->n_ctx_orig,
            .inverse = args->inverse,
            .pos0 = pos0,
            .pos_step = pos_step,
            .freq_base = args->freq_base,
            .freq_scale = args->freq_scale,
            .ext_factor = args->ext_factor,
            .attn_factor = args->attn_factor,
            .beta_fast = args->beta_fast,
            .beta_slow = args->beta_slow,
        };
    }
    const NSUInteger reference_nth =
        (NSUInteger)(head_dim < 256u ? head_dim : 256u);
    const NSUInteger pair_nth =
        (NSUInteger)(tail_threads < 256u ? tail_threads : 256u);
    const NSUInteger nth = use_shared_coeff ? 256u :
        (use_inplace_pair ? pair_nth : reference_nth);
    const NSUInteger head_groups = use_shared_coeff ?
        (NSUInteger)((n_head + 3u) / 4u) : (NSUInteger)n_head;
    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    id<MTLComputePipelineState> pipeline = use_affine_position ?
        g_rope_tail_inplace_pair_affine_pipeline :
        (use_shared_coeff ?
            g_rope_tail_inplace_pair_shared4_pipeline :
            (use_inplace_pair ?
                g_rope_tail_inplace_pair_pipeline : g_rope_tail_batch_pipeline));
    [enc setComputePipelineState:pipeline];
    if (use_affine_position) {
        [enc setBytes:&affine_args length:sizeof(affine_args) atIndex:0];
    } else {
        [enc setBytes:args length:sizeof(*args) atIndex:0];
    }
    [enc setBuffer:xbuf offset:xoff atIndex:1];
    if (!use_affine_position) {
        if (posbuf) {
            [enc setBuffer:posbuf offset:0 atIndex:2];
        } else {
            [enc setBytes:pos length:pos_bytes atIndex:2];
        }
        [enc setBuffer:xbuf offset:xoff atIndex:3];
    }
    [enc setBuffer:xbuf offset:xoff atIndex:4];
    [enc dispatchThreadgroups:MTLSizeMake(head_groups, n_tok, 1)
         threadsPerThreadgroup:MTLSizeMake(nth ? nth : 1u, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);

    if (pos && pos != pos_stack) free(pos);
    return 1;
}

typedef struct {
    int64_t ne00;
    int64_t ne01;
    int64_t ne02;
    int64_t ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
    int32_t n_rot;
} ds4_gpu_dsv4_fp8_kv_quantize_args;

typedef struct {
    int32_t head_dim;
    int32_t n_rot;
    int32_t raw_row;
} ds4_gpu_dsv4_kv_fp8_store_args;

typedef struct {
    uint32_t n_rows;
    uint32_t head_dim;
    uint64_t row_stride;
} ds4_gpu_dsv4_indexer_qat_args;

typedef struct {
    uint32_t width;
} ds4_gpu_dsv4_ratio4_shift_args;

typedef struct {
    uint32_t head_dim;
    uint32_t n_comp;
    uint32_t replay;
    uint32_t n_threads;
} ds4_gpu_dsv4_compressor_pack_ratio4_args;

typedef struct {
    int64_t  n_rows;
    uint32_t head_dim;
    uint32_t n_comp;
    uint32_t replay;
    uint32_t pad;
} ds4_gpu_dsv4_softmax_pool_ratio4_direct_args;

typedef struct {
    uint32_t width;
    uint32_t ratio;
    uint32_t pos;
    uint32_t ape_type;
} ds4_gpu_dsv4_compressor_store_one_args;

typedef struct {
    int64_t  ne00;
    int64_t  ne01;
    int64_t  ne02;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    int64_t  ne0;
    int64_t  ne1;
    uint64_t nb0;
    uint64_t nb1;
} ds4_gpu_dsv4_softmax_pool_args;

typedef struct {
    uint32_t width;
    uint32_t ratio;
    uint32_t pos0;
    uint32_t n_tokens;
} ds4_gpu_dsv4_compressor_score_ape_args;

typedef struct {
    int32_t  ne00;
    int32_t  ne01;
    int32_t  ne02;
    int32_t  ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne0;
    int32_t  ne1;
    int32_t  ne2;
    int32_t  ne3;
    int32_t  top_k;
} ds4_gpu_kargs_argsort;

typedef struct {
    int64_t  ne00;
    int64_t  ne01;
    int64_t  ne02;
    int64_t  ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne0;
    int32_t  ne1;
    int32_t  ne2;
    int32_t  ne3;
    int32_t  top_k;
    int32_t  len;
} ds4_gpu_kargs_argsort_merge;

typedef struct {
    int64_t  ne00;
    int64_t  ne01;
    int64_t  ne02;
    int64_t  ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int64_t  ne0;
    int64_t  ne1;
    int64_t  ne2;
    int64_t  ne3;
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
} ds4_gpu_kargs_sum_rows;

typedef struct {
    int32_t  ne00;
    int32_t  ne01;
    int32_t  ne02;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne11;
    int32_t  ne12;
    int32_t  ne13;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
    float    scale;
    float    max_bias;
    float    m0;
    float    m1;
    int32_t  n_head_log2;
} ds4_gpu_softmax_args;

typedef struct {
    int64_t  ne00;
    int64_t  ne01;
    uint64_t nb00;
    uint64_t nb01;
    int64_t  ne0;
    int64_t  ne1;
    uint64_t nb0;
    uint64_t nb1;
} ds4_gpu_dsv4_topk_mask_args;

typedef struct {
    int64_t  ne00;
    int64_t  ne01;
    int64_t  ne02;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    int64_t  ne10;
    int64_t  ne11;
    uint64_t nb10;
    uint64_t nb11;
    int64_t  ne0;
    int64_t  ne1;
    uint64_t nb0;
    uint64_t nb1;
    float    scale;
} ds4_gpu_dsv4_indexer_weighted_sum_args;

typedef struct {
    uint32_t has_bias;
    uint32_t hash_mode;
    uint32_t use_token_buffer;
    uint32_t token;
    uint32_t hash_rows;
} ds4_gpu_dsv4_router_select_one_args;

typedef struct {
    uint32_t n_expert;
    uint32_t n_expert_used;
    float    expert_weight_scale;
    uint32_t pad0;
} ds4_gpu_glm_router_select_one_args;

typedef struct {
    uint32_t n_tokens;
    uint32_t kv_raw_dim;
    uint32_t kv_lora_dim;
    float    eps;
} ds4_gpu_glm_kv_lora_rms_norm_args;

typedef struct {
    uint32_t n_tokens;
    uint32_t kv_lora_dim;
    uint32_t qk_nope;
    uint32_t n_head;
    uint32_t row_bytes;
    uint32_t weight_type;
    uint32_t pad1;
    uint32_t pad2;
} ds4_gpu_glm_k_b_project_args;

typedef struct {
    uint32_t pos0;
    uint32_t n_tokens;
    uint32_t cache_cap;
    uint32_t kv_raw_dim;
    uint32_t kv_lora_dim;
    uint32_t qk_rope;
    uint32_t cache_f16;
    uint32_t pad1;
} ds4_gpu_glm_store_compact_kv_args;

typedef struct {
    uint32_t pos0;
    uint32_t n_tokens;
    uint32_t cache_cap;
    uint32_t q_n;
    uint32_t q_n4;
    uint32_t kv_raw_dim;
    uint32_t kv_lora_dim;
    uint32_t kv_lora_n4;
    uint32_t qk_rope;
    uint32_t cache_f16;
    float    eps;
    uint32_t pad0;
} ds4_gpu_glm_qkv_norm_store_compact_kv_args;

typedef struct {
    uint32_t pos0;
    uint32_t n_tokens;
    uint32_t cache_cap;
    uint32_t head_dim;
    uint32_t rot_dim;
    uint32_t n_ctx_orig;
    uint32_t cache_f16;
    uint32_t pad0;
    float    eps;
    float    freq_base;
    float    freq_scale;
    float    ext_factor;
    float    attn_factor;
    float    beta_fast;
    float    beta_slow;
    float    pad1;
} ds4_gpu_glm_store_indexer_k_args;

typedef struct {
    uint32_t pos0;
    uint32_t n_tokens;
    uint32_t cache_cap;
    uint32_t n_head;
    uint32_t kv_raw_dim;
    uint32_t kv_lora_dim;
    uint32_t qk_nope;
    uint32_t qk_rope;
    uint32_t value_dim;
    uint32_t n_ctx_orig;
    uint32_t cache_f16;
    uint32_t pad0;
    float    freq_base;
    float    freq_scale;
    float    ext_factor;
    float    attn_factor;
    float    beta_fast;
    float    beta_slow;
} ds4_gpu_glm_build_kv_cache_args;

typedef struct {
    uint32_t pos0;
    uint32_t n_tokens;
    uint32_t cache_len;
    uint32_t cache_cap;
    uint32_t n_head;
    uint32_t qk_dim;
    uint32_t value_dim;
    uint32_t pad0;
    uint32_t cache_f16;
    uint32_t pad1;
    uint32_t pad2;
    float    scale;
} ds4_gpu_glm_attention_full_args;

typedef struct {
    uint32_t n_selected;
} ds4_gpu_glm_fill_selected_range_args;

typedef struct {
    uint32_t n_tokens;
    uint32_t pos0;
    uint32_t n_selected;
    uint32_t pad_row;
} ds4_gpu_glm_fill_selected_range_batch_args;

typedef struct {
    uint32_t n_tokens;
    uint32_t n_head;
    uint32_t head_dim;
    uint32_t rot_dim;
    uint32_t rot_offset;
    uint32_t pos0;
    uint32_t n_ctx_orig;
    float    freq_base;
    float    freq_scale;
    float    ext_factor;
    float    attn_factor;
    float    beta_fast;
    float    beta_slow;
} ds4_gpu_glm_rope_tail_args;

typedef struct {
    uint32_t n_rows;
    uint32_t n_head;
    uint32_t head_dim;
    uint32_t cache_f16;
    float    scale;
} ds4_gpu_glm_indexer_score_one_args;

typedef struct {
    uint32_t n_rows;
    uint32_t n_tokens;
    uint32_t n_head;
    uint32_t head_dim;
    uint32_t pos0;
    uint32_t cache_f16;
    uint64_t q_token_stride;
    uint64_t q_head_stride;
    uint64_t weights_token_stride;
    uint64_t score_token_stride;
    float    scale;
} ds4_gpu_glm_indexer_scores_batch_args;

typedef struct {
    uint32_t n_head;
    uint32_t kv_lora_dim;
    uint32_t qk_nope;
    uint32_t qk_dim;
    uint32_t row_bytes;
    uint32_t weight_type;
    uint32_t pad1;
    uint32_t pad2;
} ds4_gpu_glm_qk_lowrank_args;

typedef struct {
    uint32_t n_tokens;
    uint32_t n_head;
    uint32_t kv_lora_dim;
    uint32_t qk_nope;
    uint32_t qk_dim;
    uint32_t row_bytes;
    uint32_t weight_type;
    uint32_t head_base;
} ds4_gpu_glm_qk_lowrank_batch_args;

typedef struct {
    uint32_t n_selected;
    uint32_t cache_cap;
    uint32_t cache_f16;
    uint32_t n_head;
    uint32_t kv_lora_dim;
    uint32_t qk_nope;
    uint32_t qk_rope;
    uint32_t value_dim;
    uint32_t n_ctx_orig;
    uint32_t value_row_bytes;
    float    scale;
    float    freq_base;
    float    freq_scale;
    float    ext_factor;
    float    attn_factor;
    float    beta_fast;
    float    beta_slow;
    uint32_t value_type;
} ds4_gpu_glm_attention_indexed_decode_args;

typedef struct {
    uint32_t n_selected;
    uint32_t cache_cap;
    uint32_t cache_f16;
    uint32_t n_head;
    uint32_t kv_lora_dim;
    uint32_t qk_nope;
    uint32_t qk_rope;
    uint32_t value_dim;
    uint32_t n_ctx_orig;
    uint32_t value_row_bytes;
    uint32_t block_rows;
    uint32_t n_blocks;
    float    scale;
    float    freq_base;
    float    freq_scale;
    float    ext_factor;
    float    attn_factor;
    float    beta_fast;
    float    beta_slow;
    uint32_t value_type;
} ds4_gpu_glm_attention_indexed_decode_split_args;

typedef struct {
    uint32_t n_tokens;
    uint32_t n_selected;
    uint32_t cache_cap;
    uint32_t cache_f16;
    uint32_t n_head;
    uint32_t kv_lora_dim;
    uint32_t qk_nope;
    uint32_t qk_rope;
    uint32_t value_dim;
    uint32_t n_ctx_orig;
    uint32_t value_row_bytes;
    uint32_t value_type;
    uint32_t pos0;
    float    scale;
    float    freq_base;
    float    freq_scale;
    float    ext_factor;
    float    attn_factor;
    float    beta_fast;
    float    beta_slow;
    uint32_t head_base;
} ds4_gpu_glm_attention_indexed_batch_args;

typedef struct {
    uint32_t in_dim;
    uint32_t mid_dim;
    uint32_t out_dim;
    uint32_t n_total_expert;
    uint32_t n_expert_used;
    uint32_t n_tokens;
    uint32_t mid_token_stride;
    uint32_t down_type;
    int32_t  tp_rank;
    int32_t  tp_world;
    int32_t  tp_expert_base;
    uint64_t gate_expert_bytes;
    uint64_t gate_row_bytes;
    uint64_t up_expert_bytes;
    uint64_t up_row_bytes;
    uint64_t down_expert_bytes;
    uint64_t down_row_bytes;
} ds4_gpu_glm_routed_moe_args;

typedef struct {
    uint32_t n_tokens;
    uint32_t n_head;
    uint32_t n_raw;
    uint32_t raw_cap;
    uint32_t raw_start;
    uint32_t n_comp;
    uint32_t top_k;
    uint32_t pos0;
    uint32_t window;
    uint32_t ratio;
    uint32_t comp_kv_f16;
    uint32_t pad0;
    uint64_t q_token_stride;
    uint64_t q_head_stride;
    uint64_t raw_row_stride;
    uint64_t comp_row_stride;
    uint64_t topk_token_stride;
    uint64_t dst_token_stride;
    uint64_t dst_head_stride;
    float    scale;
} ds4_gpu_dsv4_indexed_attention_args;

typedef struct {
    uint32_t n_comp;
    uint32_t n_tokens;
    uint32_t n_head;
    uint32_t head_dim;
    uint32_t pos0;
    uint32_t ratio;
    uint64_t q_token_stride;
    uint64_t q_head_stride;
    uint64_t weights_token_stride;
    uint64_t index_row_stride;
    uint64_t score_token_stride;
    float    scale;
} ds4_gpu_dsv4_indexer_scores_fused_args;

typedef struct {
    uint32_t width;
    uint32_t rows;
    uint64_t gate_row_stride;
    uint64_t up_row_stride;
    uint64_t mid_row_stride;
    uint64_t weight_stride;
    uint32_t write_clamped;
    float    clamp_value;
} ds4_gpu_dsv4_moe_swiglu_weight_args;

typedef struct {
    uint32_t expert_base;
    uint32_t expert_count;
    uint32_t accumulate;
    uint32_t pad0;
} ds4_gpu_moe_expert_group_args;

typedef struct {
    uint64_t expert_bytes;
    uint32_t group_size;
    uint32_t n_slots;
} ds4_gpu_q4_gather_slots6_args;

typedef struct {
    uint32_t width;
    uint32_t tokens;
    uint64_t src_token_stride;
    uint64_t dst_token_stride;
} ds4_gpu_dsv4_moe_sum6_args;

/* Compile the single in-repo Metal source and create the pipelines that every
 * session uses. Shape-dependent kernels with function constants are built
 * lazily by the small ds4_gpu_get_* caches, so startup stays predictable
 * while long-context prefill and decode can still pick specialized variants. */
int ds4_gpu_init(void) {
    if (g_initialized) return 1;

    @autoreleasepool {
        g_device = MTLCreateSystemDefaultDevice();
        if (!g_device) {
            fprintf(stderr, "ds4: Metal device not available\n");
            return 0;
        }
        ds4_gpu_print_device_summary();
        ds4_gpu_detect_metal4_features();

        g_queue = [g_device newCommandQueue];
        if (!g_queue) {
            fprintf(stderr, "ds4: failed to create Metal command queue\n");
            g_device = nil;
            return 0;
        }
        g_model_buffer_cache = [NSMutableDictionary dictionary];
        g_model_buffer_cache_bytes = 0;
        g_model_buffer_cache_evictions = 0;
        g_model_buffer_cache_over_limit = 0;
        g_q4_expert_table_cache = [NSMutableDictionary dictionary];
        g_q4_expert_layer_residency_cache = [NSMutableDictionary dictionary];
        g_pipeline_cache = [NSMutableDictionary dictionary];
        g_transient_buffers = [NSMutableArray array];
        g_pending_cbs = [NSMutableArray array];
        if (!g_model_buffer_cache || !g_q4_expert_table_cache ||
            !g_q4_expert_layer_residency_cache ||
            !g_pipeline_cache || !g_transient_buffers || !g_pending_cbs) {
            fprintf(stderr, "ds4: Metal bookkeeping allocation failed\n");
            g_pending_cbs = nil;
            g_transient_buffers = nil;
            g_pipeline_cache = nil;
            g_q4_expert_layer_residency_cache = nil;
            g_q4_expert_table_cache = nil;
            g_model_buffer_cache = nil;
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        NSError *error = nil;
        NSString *source = ds4_gpu_full_source();
        if (!source) {
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        MTLCompileOptions *options = [MTLCompileOptions new];
        NSMutableDictionary *macros = [NSMutableDictionary new];
        if (g_metal4_tensor_api_enabled) {
            macros[@"DS4_METAL_HAS_TENSOR"] = @"1";
            fprintf(stderr, "ds4: Metal 4 tensor API enabled for Tensor kernels\n");
        }

        const int drift_hc_stable        = ds4_gpu_env_bool("DS4_METAL_HC_STABLE")          != 0; // default ON
        const int drift_norm_unify       = ds4_gpu_env_bool("DS4_METAL_NORM_RSQRT_DISABLE") != 0; // default ON
        const int drift_kv_raw_f32       = ds4_gpu_env_bool("DS4_METAL_KV_RAW_F32")         >  0; // default OFF
        const int drift_rope_exp2_log2   = ds4_gpu_env_bool("DS4_METAL_ROPE_EXP2_LOG2")     >  0; // default OFF
        const int drift_math_safe        = ds4_gpu_env_bool("DS4_METAL_MATH_SAFE")          >  0; // default OFF

        if (drift_math_safe) {
            // MTLCompileOptions.fastMathEnabled defaults to YES and Apple's
            // headers explicitly say this "may violate the IEEE 754 standard".
            // Different fast-math optimizations get applied across the
            // matmul2d cooperative-tensor path and the legacy
            // simdgroup_multiply_accumulate path on M5, amplifying the
            // mismatch. MTLMathModeSafe pins the entire library to strict
            // IEEE-754 semantics. Diagnostic-only: useful to localize drift
            // sources but not to ship as a default.
            if (@available(macOS 15.0, *)) {
                options.mathMode = MTLMathModeSafe;
                fprintf(stderr, "ds4: Metal shader library math mode = safe (strict IEEE-754) by DS4_METAL_MATH_SAFE\n");
            } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
                options.fastMathEnabled = NO;
#pragma clang diagnostic pop
                fprintf(stderr, "ds4: Metal shader library fast-math disabled by DS4_METAL_MATH_SAFE (pre-macOS 15)\n");
            }
        }

        if (drift_hc_stable)      macros[@"DS4_METAL_HC_STABLE"]          = @"1";
        if (drift_norm_unify)     macros[@"DS4_METAL_NORM_RSQRT_DISABLE"] = @"1";
        if (drift_kv_raw_f32)     macros[@"DS4_METAL_KV_RAW_F32"]         = @"1";
        if (drift_rope_exp2_log2) macros[@"DS4_METAL_ROPE_EXP2_LOG2"]     = @"1";
        fprintf(stderr,
                "ds4: drift-patch flags hc_stable=%s norm_unify=%s kv_raw_f32=%s rope_exp2_log2=%s math_safe=%s tensor_matmul=%s\n",
                drift_hc_stable      ? "on"  : "off",
                drift_norm_unify     ? "on"  : "off",
                drift_kv_raw_f32     ? "on"  : "off",
                drift_rope_exp2_log2 ? "on"  : "off",
                drift_math_safe      ? "on"  : "off",
                g_metal4_tensor_api_enabled ? "on" : "off");
        options.preprocessorMacros = macros;
        id<MTLLibrary> library = [g_device newLibraryWithSource:source options:options error:&error];
        if (!library) {
            fprintf(stderr, "ds4: Metal shader compilation failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_library = library;

        id<MTLFunction> fn = [library newFunctionWithName:@"kernel_get_rows_f32"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_get_rows_f32 function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        g_get_rows_f32_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_get_rows_f32_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_get_rows_f32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_get_rows_f16"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_get_rows_f16 function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        g_get_rows_f16_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_get_rows_f16_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_get_rows_f16 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_get_rows_i32"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_get_rows_i32 function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        g_get_rows_i32_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_get_rows_i32_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_get_rows_i32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_get_rows_q8_0_f32"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_get_rows_q8_0_f32 function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        g_get_rows_q8_0_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_get_rows_q8_0_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_get_rows_q8_0_f32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_get_rows_q4_0_f32"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_get_rows_q4_0_f32 function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        g_get_rows_q4_0_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_get_rows_q4_0_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_get_rows_q4_0_f32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_get_rows_q4_K_f32"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_get_rows_q4_K_f32 function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        g_get_rows_q4_K_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_get_rows_q4_K_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_get_rows_q4_K_f32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_repeat_f32"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_repeat_f32 function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        g_repeat_f32_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_repeat_f32_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_repeat_f32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_set_rows_f32_i32"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_set_rows_f32_i32 function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        g_set_rows_f32_i32_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_set_rows_f32_i32_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_set_rows_f32_i32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_concat"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_concat function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        g_concat_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_concat_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_concat pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_cpy_f32_f32"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_cpy_f32_f32 function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        g_cpy_f32_f32_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_cpy_f32_f32_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_cpy_f32_f32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_cpy_f32_f16"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_cpy_f32_f16 function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        g_cpy_f32_f16_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_cpy_f32_f16_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_cpy_f32_f16 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_cpy_contig_f32_f16_4"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_cpy_contig_f32_f16_4 function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        g_cpy_contig_f32_f16_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_cpy_contig_f32_f16_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_cpy_contig_f32_f16_4 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_cpy_f16_f32"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_cpy_f16_f32 function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        g_cpy_f16_f32_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_cpy_f16_f32_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_cpy_f16_f32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_cpy_f16_f16"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_cpy_f16_f16 function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        g_cpy_f16_f16_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_cpy_f16_f16_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_cpy_f16_f16 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_cpy_contig_f16_f32_4"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_cpy_contig_f16_f32_4 function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        g_cpy_contig_f16_f32_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_cpy_contig_f16_f32_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_cpy_contig_f16_f32_4 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_cpy_contig_f16_f16_bits_4"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_cpy_contig_f16_f16_bits_4 function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        g_cpy_contig_f16_f16_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_cpy_contig_f16_f16_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_cpy_contig_f16_f16_bits_4 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        error = nil;
        fn = [library newFunctionWithName:@"kernel_dsv4_flash_kv_stage_f16"];
        if (fn) {
            g_flash_kv_stage_f16_pipeline =
                [g_device newComputePipelineStateWithFunction:fn error:&error];
            if (!g_flash_kv_stage_f16_pipeline) {
                fprintf(stderr,
                        "ds4: optional Metal gathered KV staging pipeline unavailable: %s\n",
                        [[error localizedDescription] UTF8String]);
            }
        } else {
            fprintf(stderr,
                    "ds4: optional Metal gathered KV staging kernel unavailable\n");
        }

        fn = [library newFunctionWithName:@"kernel_dsv4_fp8_kv_quantize_f32"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_dsv4_fp8_kv_quantize_f32 function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_dsv4_fp8_kv_quantize_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_dsv4_fp8_kv_quantize_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_dsv4_fp8_kv_quantize_f32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_dsv4_indexer_hadamard_fp4_f32"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_dsv4_indexer_hadamard_fp4_f32 function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_dsv4_indexer_qat_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_dsv4_indexer_qat_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_dsv4_indexer_hadamard_fp4_f32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_dsv4_kv_fp8_store_f32"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_dsv4_kv_fp8_store_f32 function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_dsv4_kv_fp8_store_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_dsv4_kv_fp8_store_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_dsv4_kv_fp8_store_f32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_dsv4_ratio4_shift_f32"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_dsv4_ratio4_shift_f32 function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_dsv4_ratio4_shift_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_dsv4_ratio4_shift_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_dsv4_ratio4_shift_f32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_swiglu_f32"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_swiglu_f32 function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        g_swiglu_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_swiglu_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_swiglu_f32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_swiglu_flat_f32"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_swiglu_flat_f32 function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        error = nil;
        g_swiglu_flat_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_swiglu_flat_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_swiglu_flat_f32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_dsv4_moe_sum6_f32"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_dsv4_moe_sum6_f32 function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        g_moe_sum6_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_moe_sum6_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_dsv4_moe_sum6_f32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_dsv4_moe_sum8_f32"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_dsv4_moe_sum8_f32 function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        g_moe_sum8_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_moe_sum8_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_dsv4_moe_sum8_f32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        MTLFunctionConstantValues *bin_constants = [[MTLFunctionConstantValues alloc] init];
        int16_t bin_op = 0;
        int16_t bin_f = 1;
        bool bin_rb = false;
        bool bin_cb = false;
        [bin_constants setConstantValue:&bin_op type:MTLDataTypeShort atIndex:1300];
        [bin_constants setConstantValue:&bin_f  type:MTLDataTypeShort atIndex:1301];
        [bin_constants setConstantValue:&bin_rb type:MTLDataTypeBool  atIndex:1302];
        [bin_constants setConstantValue:&bin_cb type:MTLDataTypeBool  atIndex:1303];

        error = nil;
        fn = [library newFunctionWithName:@"kernel_bin_fuse_f32_f32_f32"
                           constantValues:bin_constants
                                    error:&error];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_bin_fuse_f32_f32_f32 function not found: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        g_add_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_add_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_bin_fuse_f32_f32_f32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_add2_f32"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_add2_f32 function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        error = nil;
        g_add2_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_add2_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_add2_f32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_add3_f32"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_add3_f32 function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        error = nil;
        g_add3_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_add3_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_add3_f32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        MTLFunctionConstantValues *bin_mul_constants = [[MTLFunctionConstantValues alloc] init];
        int16_t bin_mul_plain_op = 2;
        int16_t bin_mul_plain_f = 1;
        bool bin_mul_plain_rb = false;
        bool bin_mul_plain_cb = false;
        [bin_mul_constants setConstantValue:&bin_mul_plain_op type:MTLDataTypeShort atIndex:1300];
        [bin_mul_constants setConstantValue:&bin_mul_plain_f  type:MTLDataTypeShort atIndex:1301];
        [bin_mul_constants setConstantValue:&bin_mul_plain_rb type:MTLDataTypeBool  atIndex:1302];
        [bin_mul_constants setConstantValue:&bin_mul_plain_cb type:MTLDataTypeBool  atIndex:1303];

        error = nil;
        fn = [library newFunctionWithName:@"kernel_bin_fuse_f32_f32_f32"
                           constantValues:bin_mul_constants
                                    error:&error];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_bin_fuse_f32_f32_f32 mul function not found: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        g_mul_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_mul_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_bin_fuse_f32_f32_f32 mul pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        MTLFunctionConstantValues *bin_mul_scalar_constants = [[MTLFunctionConstantValues alloc] init];
        int16_t bin_mul_op = 2;
        int16_t bin_mul_f = 1;
        bool bin_mul_rb = false;
        bool bin_mul_cb = true;
        [bin_mul_scalar_constants setConstantValue:&bin_mul_op type:MTLDataTypeShort atIndex:1300];
        [bin_mul_scalar_constants setConstantValue:&bin_mul_f  type:MTLDataTypeShort atIndex:1301];
        [bin_mul_scalar_constants setConstantValue:&bin_mul_rb type:MTLDataTypeBool  atIndex:1302];
        [bin_mul_scalar_constants setConstantValue:&bin_mul_cb type:MTLDataTypeBool  atIndex:1303];

        error = nil;
        fn = [library newFunctionWithName:@"kernel_bin_fuse_f32_f32_f32"
                           constantValues:bin_mul_scalar_constants
                                    error:&error];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_bin_fuse_f32_f32_f32 mul-scalar function not found: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        g_bin_mul_scalar_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_bin_mul_scalar_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_bin_fuse_f32_f32_f32 mul-scalar pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        MTLFunctionConstantValues *bin_div_row_constants = [[MTLFunctionConstantValues alloc] init];
        int16_t bin_div_op = 3;
        int16_t bin_div_f = 1;
        bool bin_div_rb = false;
        bool bin_div_cb = true;
        [bin_div_row_constants setConstantValue:&bin_div_op type:MTLDataTypeShort atIndex:1300];
        [bin_div_row_constants setConstantValue:&bin_div_f  type:MTLDataTypeShort atIndex:1301];
        [bin_div_row_constants setConstantValue:&bin_div_rb type:MTLDataTypeBool  atIndex:1302];
        [bin_div_row_constants setConstantValue:&bin_div_cb type:MTLDataTypeBool  atIndex:1303];

        error = nil;
        fn = [library newFunctionWithName:@"kernel_bin_fuse_f32_f32_f32"
                           constantValues:bin_div_row_constants
                                    error:&error];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_bin_fuse_f32_f32_f32 div-row function not found: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        g_bin_div_row_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_bin_div_row_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_bin_fuse_f32_f32_f32 div-row pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_rms_norm_mul_f32_4"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_rms_norm_mul_f32_4 function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_rms_norm_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_rms_norm_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_rms_norm_mul_f32_4 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_rms_norm_f32_4"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_rms_norm_f32_4 function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_rms_norm_plain_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_rms_norm_plain_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_rms_norm_f32_4 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_add_rms_norm_mul_f32_4"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_add_rms_norm_mul_f32_4 function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_add_rms_norm_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_add_rms_norm_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_add_rms_norm_mul_f32_4 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_dsv4_qkv_rms_norm_f32_4"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_dsv4_qkv_rms_norm_f32_4 function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_dsv4_qkv_rms_norm_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_dsv4_qkv_rms_norm_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_dsv4_qkv_rms_norm_f32_4 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        MTLFunctionConstantValues *moe_mv_id_constants = [[MTLFunctionConstantValues alloc] init];
        int16_t moe_mv_id_nsg = 2;
        [moe_mv_id_constants setConstantValue:&moe_mv_id_nsg type:MTLDataTypeShort atIndex:600];

        error = nil;
        fn = [library newFunctionWithName:@"kernel_mul_mv_id_iq2_xxs_f32"
                           constantValues:moe_mv_id_constants
                                    error:&error];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_mul_mv_id_iq2_xxs_f32 function not found: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_moe_mul_mv_id_iq2_xxs_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_moe_mul_mv_id_iq2_xxs_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_mul_mv_id_iq2_xxs_f32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        error = nil;
        fn = [library newFunctionWithName:@"kernel_mul_mv_id_iq2_xxs_pair_f32"
                           constantValues:moe_mv_id_constants
                                    error:&error];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_mul_mv_id_iq2_xxs_pair_f32 function not found: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_moe_mul_mv_id_iq2_xxs_pair_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_moe_mul_mv_id_iq2_xxs_pair_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_mul_mv_id_iq2_xxs_pair_f32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        error = nil;
        fn = [library newFunctionWithName:@"kernel_mul_mv_id_iq2_xxs_pair_swiglu_f32"
                           constantValues:moe_mv_id_constants
                                    error:&error];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_mul_mv_id_iq2_xxs_pair_swiglu_f32 function not found: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_moe_mul_mv_id_iq2_xxs_pair_swiglu_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_moe_mul_mv_id_iq2_xxs_pair_swiglu_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_mul_mv_id_iq2_xxs_pair_swiglu_f32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        error = nil;
        fn = [library newFunctionWithName:@"kernel_mul_mv_id_q2_K_f32"
                           constantValues:moe_mv_id_constants
                                    error:&error];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_mul_mv_id_q2_K_f32 function not found: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_moe_mul_mv_id_q2_k_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_moe_mul_mv_id_q2_k_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_mul_mv_id_q2_K_f32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        error = nil;
        fn = [library newFunctionWithName:@"kernel_mul_mv_id_q2_K_sum6_f32"
                           constantValues:moe_mv_id_constants
                                    error:&error];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_mul_mv_id_q2_K_sum6_f32 function not found: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_moe_mul_mv_id_q2_k_sum6_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_moe_mul_mv_id_q2_k_sum6_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_mul_mv_id_q2_K_sum6_f32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        error = nil;
        fn = [library newFunctionWithName:@"kernel_mul_mv_id_iq2_xxs_sum6_f32"
                           constantValues:moe_mv_id_constants
                                    error:&error];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_mul_mv_id_iq2_xxs_sum6_f32 function not found: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_moe_mul_mv_id_iq2_xxs_sum6_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_moe_mul_mv_id_iq2_xxs_sum6_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_mul_mv_id_iq2_xxs_sum6_f32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        error = nil;
        fn = [library newFunctionWithName:@"kernel_mul_mv_slots6_iq2_xxs_pair_swiglu_f32"
                           constantValues:moe_mv_id_constants
                                    error:&error];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_mul_mv_slots6_iq2_xxs_pair_swiglu_f32 function not found: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_moe_mul_mv_slots6_iq2_xxs_pair_swiglu_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_moe_mul_mv_slots6_iq2_xxs_pair_swiglu_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_mul_mv_slots6_iq2_xxs_pair_swiglu_f32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        error = nil;
        fn = [library newFunctionWithName:@"kernel_mul_mv_slots6_q2_K_sum6_f32"
                           constantValues:moe_mv_id_constants
                                    error:&error];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_mul_mv_slots6_q2_K_sum6_f32 function not found: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_moe_mul_mv_slots6_q2_k_sum6_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_moe_mul_mv_slots6_q2_k_sum6_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_mul_mv_slots6_q2_K_sum6_f32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        error = nil;
        fn = [library newFunctionWithName:@"kernel_mul_mv_addr_iq2_xxs_pair_swiglu_f32"
                           constantValues:moe_mv_id_constants
                                    error:&error];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_mul_mv_addr_iq2_xxs_pair_swiglu_f32 function not found: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_moe_mul_mv_addr_iq2_xxs_pair_swiglu_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_moe_mul_mv_addr_iq2_xxs_pair_swiglu_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_mul_mv_addr_iq2_xxs_pair_swiglu_f32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        error = nil;
        fn = [library newFunctionWithName:@"kernel_mul_mv_addr_iq2_xxs_f32"
                           constantValues:moe_mv_id_constants
                                    error:&error];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_mul_mv_addr_iq2_xxs_f32 function not found: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_moe_mul_mv_addr_iq2_xxs_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_moe_mul_mv_addr_iq2_xxs_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_mul_mv_addr_iq2_xxs_f32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        error = nil;
        fn = [library newFunctionWithName:@"kernel_mul_mv_addr_q2_K_sum6_f32"
                           constantValues:moe_mv_id_constants
                                    error:&error];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_mul_mv_addr_q2_K_sum6_f32 function not found: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_moe_mul_mv_addr_q2_k_sum6_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_moe_mul_mv_addr_q2_k_sum6_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_mul_mv_addr_q2_K_sum6_f32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        error = nil;
        fn = [library newFunctionWithName:@"kernel_mul_mv_addr_iq2_xxs_pair_swiglu_masked_f32"
                           constantValues:moe_mv_id_constants
                                    error:&error];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_mul_mv_addr_iq2_xxs_pair_swiglu_masked_f32 function not found: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_moe_mul_mv_addr_iq2_xxs_pair_swiglu_masked_pipeline =
            [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_moe_mul_mv_addr_iq2_xxs_pair_swiglu_masked_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_mul_mv_addr_iq2_xxs_pair_swiglu_masked_f32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        error = nil;
        fn = [library newFunctionWithName:@"kernel_mul_mv_addr_q2_K_sum6_masked_f32"
                           constantValues:moe_mv_id_constants
                                    error:&error];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_mul_mv_addr_q2_K_sum6_masked_f32 function not found: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_moe_mul_mv_addr_q2_k_sum6_masked_pipeline =
            [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_moe_mul_mv_addr_q2_k_sum6_masked_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_mul_mv_addr_q2_K_sum6_masked_f32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        error = nil;
        fn = [library newFunctionWithName:@"kernel_stream_expert_cache_validate"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_stream_expert_cache_validate function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_moe_stream_expert_cache_validate_pipeline =
            [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_moe_stream_expert_cache_validate_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_stream_expert_cache_validate pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        error = nil;
        fn = [library newFunctionWithName:@"kernel_mul_mv_id_q4_K_f32"
                           constantValues:moe_mv_id_constants
                                    error:&error];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_mul_mv_id_q4_K_f32 function not found: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_moe_mul_mv_id_q4_k_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_moe_mul_mv_id_q4_k_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_mul_mv_id_q4_K_f32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        error = nil;
        fn = [library newFunctionWithName:@"kernel_mul_mv_id_q4_K_pair_f32"
                           constantValues:moe_mv_id_constants
                                    error:&error];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_mul_mv_id_q4_K_pair_f32 function not found: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_moe_mul_mv_id_q4_k_pair_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_moe_mul_mv_id_q4_k_pair_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_mul_mv_id_q4_K_pair_f32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        error = nil;
        fn = [library newFunctionWithName:@"kernel_mul_mv_id_q4_K_pair_swiglu_f32"
                           constantValues:moe_mv_id_constants
                                    error:&error];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_mul_mv_id_q4_K_pair_swiglu_f32 function not found: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_moe_mul_mv_id_q4_k_pair_swiglu_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_moe_mul_mv_id_q4_k_pair_swiglu_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_mul_mv_id_q4_K_pair_swiglu_f32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        error = nil;
        fn = [library newFunctionWithName:@"kernel_mul_mv_id_q4_K_sum6_f32"
                           constantValues:moe_mv_id_constants
                                    error:&error];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_mul_mv_id_q4_K_sum6_f32 function not found: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_moe_mul_mv_id_q4_k_sum6_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_moe_mul_mv_id_q4_k_sum6_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_mul_mv_id_q4_K_sum6_f32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        error = nil;
        fn = [library newFunctionWithName:@"kernel_mul_mv_group_q4_K_pair_swiglu_f32"
                           constantValues:moe_mv_id_constants
                                    error:&error];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_mul_mv_group_q4_K_pair_swiglu_f32 function not found: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_moe_mul_mv_group_q4_k_pair_swiglu_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_moe_mul_mv_group_q4_k_pair_swiglu_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_mul_mv_group_q4_K_pair_swiglu_f32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        error = nil;
        fn = [library newFunctionWithName:@"kernel_mul_mv_group_q4_K_sum6_f32"
                           constantValues:moe_mv_id_constants
                                    error:&error];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_mul_mv_group_q4_K_sum6_f32 function not found: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_moe_mul_mv_group_q4_k_sum6_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_moe_mul_mv_group_q4_k_sum6_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_mul_mv_group_q4_K_sum6_f32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        error = nil;
        fn = [library newFunctionWithName:@"kernel_mul_mv_group6_q4_K_pair_swiglu_f32"
                           constantValues:moe_mv_id_constants
                                    error:&error];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_mul_mv_group6_q4_K_pair_swiglu_f32 function not found: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_moe_mul_mv_group6_q4_k_pair_swiglu_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_moe_mul_mv_group6_q4_k_pair_swiglu_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_mul_mv_group6_q4_K_pair_swiglu_f32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        error = nil;
        fn = [library newFunctionWithName:@"kernel_mul_mv_group6_q4_K_sum6_f32"
                           constantValues:moe_mv_id_constants
                                    error:&error];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_mul_mv_group6_q4_K_sum6_f32 function not found: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_moe_mul_mv_group6_q4_k_sum6_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_moe_mul_mv_group6_q4_k_sum6_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_mul_mv_group6_q4_K_sum6_f32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        error = nil;
        fn = [library newFunctionWithName:@"kernel_mul_mv_group8_q4_K_pair_swiglu_f32"
                           constantValues:moe_mv_id_constants
                                    error:&error];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_mul_mv_group8_q4_K_pair_swiglu_f32 function not found: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_moe_mul_mv_group8_q4_k_pair_swiglu_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_moe_mul_mv_group8_q4_k_pair_swiglu_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_mul_mv_group8_q4_K_pair_swiglu_f32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        error = nil;
        fn = [library newFunctionWithName:@"kernel_mul_mv_group8_q4_K_sum6_f32"
                           constantValues:moe_mv_id_constants
                                    error:&error];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_mul_mv_group8_q4_K_sum6_f32 function not found: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_moe_mul_mv_group8_q4_k_sum6_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_moe_mul_mv_group8_q4_k_sum6_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_mul_mv_group8_q4_K_sum6_f32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        error = nil;
        fn = [library newFunctionWithName:@"kernel_mul_mv_group24_q4_K_id_f32"
                           constantValues:moe_mv_id_constants
                                    error:&error];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_mul_mv_group24_q4_K_id_f32 function not found: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_moe_mul_mv_group24_q4_k_id_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_moe_mul_mv_group24_q4_k_id_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_mul_mv_group24_q4_K_id_f32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        error = nil;
        fn = [library newFunctionWithName:@"kernel_mul_mv_group24_q4_K_sum6_f32"
                           constantValues:moe_mv_id_constants
                                    error:&error];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_mul_mv_group24_q4_K_sum6_f32 function not found: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_moe_mul_mv_group24_q4_k_sum6_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_moe_mul_mv_group24_q4_k_sum6_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_mul_mv_group24_q4_K_sum6_f32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        error = nil;
        fn = [library newFunctionWithName:@"kernel_mul_mv_slots6_q4_K_pair_swiglu_f32"
                           constantValues:moe_mv_id_constants
                                    error:&error];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_mul_mv_slots6_q4_K_pair_swiglu_f32 function not found: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_moe_mul_mv_slots6_q4_k_pair_swiglu_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_moe_mul_mv_slots6_q4_k_pair_swiglu_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_mul_mv_slots6_q4_K_pair_swiglu_f32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        error = nil;
        fn = [library newFunctionWithName:@"kernel_mul_mv_slots6_q4_K_sum6_f32"
                           constantValues:moe_mv_id_constants
                                    error:&error];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_mul_mv_slots6_q4_K_sum6_f32 function not found: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_moe_mul_mv_slots6_q4_k_sum6_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_moe_mul_mv_slots6_q4_k_sum6_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_mul_mv_slots6_q4_K_sum6_f32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        error = nil;
        fn = [library newFunctionWithName:@"kernel_q4_gather_slots6"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_q4_gather_slots6 function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_moe_q4_gather_slots6_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_moe_q4_gather_slots6_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_q4_gather_slots6 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        error = nil;
        fn = [library newFunctionWithName:@"kernel_mul_mv_table_q4_K_pair_swiglu_f32"
                           constantValues:moe_mv_id_constants
                                    error:&error];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_mul_mv_table_q4_K_pair_swiglu_f32 function not found: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_moe_table_q4_pair_gate_encoder = [fn newArgumentEncoderWithBufferIndex:2];
        g_moe_table_q4_pair_up_encoder = [fn newArgumentEncoderWithBufferIndex:3];
        if (!g_moe_table_q4_pair_gate_encoder || !g_moe_table_q4_pair_up_encoder) {
            fprintf(stderr, "ds4: Metal Q4 expert-table pair argument encoder creation failed\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_moe_mul_mv_table_q4_k_pair_swiglu_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_moe_mul_mv_table_q4_k_pair_swiglu_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_mul_mv_table_q4_K_pair_swiglu_f32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        error = nil;
        fn = [library newFunctionWithName:@"kernel_mul_mv_table_q4_K_sum6_f32"
                           constantValues:moe_mv_id_constants
                                    error:&error];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_mul_mv_table_q4_K_sum6_f32 function not found: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_moe_table_q4_sum_down_encoder = [fn newArgumentEncoderWithBufferIndex:1];
        if (!g_moe_table_q4_sum_down_encoder) {
            fprintf(stderr, "ds4: Metal Q4 expert-table down argument encoder creation failed\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_moe_mul_mv_table_q4_k_sum6_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_moe_mul_mv_table_q4_k_sum6_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_mul_mv_table_q4_K_sum6_f32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        error = nil;
        fn = [library newFunctionWithName:@"kernel_mul_mv_addr_q4_K_pair_swiglu_f32"
                           constantValues:moe_mv_id_constants
                                    error:&error];
        if (fn) {
            g_moe_mul_mv_addr_q4_k_pair_swiglu_pipeline =
                [g_device newComputePipelineStateWithFunction:fn error:&error];
            if (!g_moe_mul_mv_addr_q4_k_pair_swiglu_pipeline) {
                fprintf(stderr, "ds4: Metal kernel_mul_mv_addr_q4_K_pair_swiglu_f32 pipeline unavailable: %s\n",
                        [[error localizedDescription] UTF8String]);
            }
        } else {
            fprintf(stderr, "ds4: Metal kernel_mul_mv_addr_q4_K_pair_swiglu_f32 function unavailable: %s\n",
                    [[error localizedDescription] UTF8String]);
        }

        error = nil;
        fn = [library newFunctionWithName:@"kernel_mul_mv_addr_q4_K_sum6_f32"
                           constantValues:moe_mv_id_constants
                                    error:&error];
        if (fn) {
            g_moe_mul_mv_addr_q4_k_sum6_pipeline =
                [g_device newComputePipelineStateWithFunction:fn error:&error];
            if (!g_moe_mul_mv_addr_q4_k_sum6_pipeline) {
                fprintf(stderr, "ds4: Metal kernel_mul_mv_addr_q4_K_sum6_f32 pipeline unavailable: %s\n",
                        [[error localizedDescription] UTF8String]);
            }
        } else {
            fprintf(stderr, "ds4: Metal kernel_mul_mv_addr_q4_K_sum6_f32 function unavailable: %s\n",
                    [[error localizedDescription] UTF8String]);
        }

        fn = [library newFunctionWithName:@"kernel_dsv4_rope_tail_f32"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_dsv4_rope_tail_f32 function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_rope_tail_batch_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_rope_tail_batch_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_dsv4_rope_tail_f32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        error = nil;
        fn = [library newFunctionWithName:@"kernel_dsv4_rope_tail_f32_inplace_pair"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_dsv4_rope_tail_f32_inplace_pair function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_rope_tail_inplace_pair_pipeline =
            [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_rope_tail_inplace_pair_pipeline) {
            fprintf(stderr,
                    "ds4: Metal kernel_dsv4_rope_tail_f32_inplace_pair pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        error = nil;
        fn = [library newFunctionWithName:@"kernel_dsv4_rope_tail_f32_inplace_pair_shared4"];
        if (!fn) {
            fprintf(stderr,
                    "ds4: optional Metal shared-head RoPE kernel unavailable; using per-head path\n");
        } else {
            g_rope_tail_inplace_pair_shared4_pipeline =
                [g_device newComputePipelineStateWithFunction:fn error:&error];
            if (!g_rope_tail_inplace_pair_shared4_pipeline) {
                fprintf(stderr,
                        "ds4: optional Metal shared-head RoPE pipeline unavailable; using per-head path: %s\n",
                        [[error localizedDescription] UTF8String]);
            }
        }

        error = nil;
        fn = [library newFunctionWithName:@"kernel_dsv4_rope_tail_f32_inplace_pair_affine"];
        if (fn) {
            g_rope_tail_inplace_pair_affine_pipeline =
                [g_device newComputePipelineStateWithFunction:fn error:&error];
            if (!g_rope_tail_inplace_pair_affine_pipeline) {
                fprintf(stderr,
                        "ds4: optional Metal affine-position RoPE pair pipeline unavailable: %s\n",
                        [[error localizedDescription] UTF8String]);
            }
        } else {
            fprintf(stderr,
                    "ds4: optional Metal affine-position RoPE pair kernel unavailable\n");
        }

        error = nil;
        fn = [library newFunctionWithName:@"kernel_dsv4_softmax_pool"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_dsv4_softmax_pool function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_dsv4_softmax_pool_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_dsv4_softmax_pool_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_dsv4_softmax_pool pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_soft_max_f32"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_soft_max_f32 function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_soft_max_f32_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_soft_max_f32_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_soft_max_f32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_soft_max_f32_4"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_soft_max_f32_4 function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_soft_max_f32_4_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_soft_max_f32_4_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_soft_max_f32_4 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_argsort_f32_i32_desc"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_argsort_f32_i32_desc function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_argsort_f32_i32_desc_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_argsort_f32_i32_desc_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_argsort_f32_i32_desc pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_argsort_merge_f32_i32_desc"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_argsort_merge_f32_i32_desc function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_argsort_merge_f32_i32_desc_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_argsort_merge_f32_i32_desc_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_argsort_merge_f32_i32_desc pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        MTLFunctionConstantValues *sum_rows_constants = [[MTLFunctionConstantValues alloc] init];
        int16_t sum_rows_op = 10;
        [sum_rows_constants setConstantValue:&sum_rows_op type:MTLDataTypeShort atIndex:1400];

        error = nil;
        fn = [library newFunctionWithName:@"kernel_sum_rows_f32_f32"
                           constantValues:sum_rows_constants
                                    error:&error];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_sum_rows_f32_f32 function not found: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_sum_rows_f32_f32_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_sum_rows_f32_f32_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_sum_rows_f32_f32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_dsv4_topk_mask"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_dsv4_topk_mask function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_dsv4_topk_mask_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_dsv4_topk_mask_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_dsv4_topk_mask pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_dsv4_topk_mask_scatter"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_dsv4_topk_mask_scatter function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_dsv4_topk_mask_scatter_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_dsv4_topk_mask_scatter_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_dsv4_topk_mask_scatter pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_dsv4_indexer_weighted_sum"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_dsv4_indexer_weighted_sum function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_dsv4_indexer_weighted_sum_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_dsv4_indexer_weighted_sum_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_dsv4_indexer_weighted_sum pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_dsv4_hc_split_sinkhorn"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_dsv4_hc_split_sinkhorn function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_hc_split_sinkhorn_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_hc_split_sinkhorn_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_dsv4_hc_split_sinkhorn pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_dsv4_hc_split_weighted_sum"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_dsv4_hc_split_weighted_sum function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_hc_split_weighted_sum_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_hc_split_weighted_sum_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_dsv4_hc_split_weighted_sum pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_dsv4_hc_split_weighted_sum_norm4"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_dsv4_hc_split_weighted_sum_norm4 function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_hc_split_weighted_sum_norm_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_hc_split_weighted_sum_norm_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_dsv4_hc_split_weighted_sum_norm4 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_dsv4_hc_weighted_sum"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_dsv4_hc_weighted_sum function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_hc_weighted_sum_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_hc_weighted_sum_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_dsv4_hc_weighted_sum pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        error = nil;
        fn = [library newFunctionWithName:@"kernel_dsv4_hc_weighted_sum_norm4"];
        if (fn) {
            g_hc_weighted_sum_norm_pipeline =
                [g_device newComputePipelineStateWithFunction:fn error:&error];
            if (!g_hc_weighted_sum_norm_pipeline) {
                fprintf(stderr,
                        "ds4: optional Metal output HC sum/RMSNorm pipeline unavailable: %s\n",
                        [[error localizedDescription] UTF8String]);
            }
        } else {
            fprintf(stderr,
                    "ds4: optional Metal output HC sum/RMSNorm kernel unavailable\n");
        }

        error = nil;
        fn = [library newFunctionWithName:@"kernel_dsv4_output_hc_weights4"];
        if (fn) {
            g_output_hc_weights4_pipeline =
                [g_device newComputePipelineStateWithFunction:fn error:&error];
            if (!g_output_hc_weights4_pipeline) {
                fprintf(stderr,
                        "ds4: optional Metal output HC weights4 pipeline unavailable: %s\n",
                        [[error localizedDescription] UTF8String]);
            }
        } else {
            fprintf(stderr,
                    "ds4: optional Metal output HC weights4 kernel unavailable\n");
        }

        MTLFunctionConstantValues *unary_sigmoid_constants = [[MTLFunctionConstantValues alloc] init];
        int16_t unary_sigmoid_op = 102;
        bool unary_cnt = false;
        [unary_sigmoid_constants setConstantValue:&unary_sigmoid_op type:MTLDataTypeShort atIndex:1200];
        [unary_sigmoid_constants setConstantValue:&unary_cnt        type:MTLDataTypeBool  atIndex:1201];

        error = nil;
        fn = [library newFunctionWithName:@"kernel_unary_f32_f32_4"
                           constantValues:unary_sigmoid_constants
                                    error:&error];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_unary_f32_f32_4 sigmoid function not found: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_unary_sigmoid_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_unary_sigmoid_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_unary_f32_f32_4 sigmoid pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        MTLFunctionConstantValues *unary_silu_constants = [[MTLFunctionConstantValues alloc] init];
        int16_t unary_silu_op = 106;
        [unary_silu_constants setConstantValue:&unary_silu_op type:MTLDataTypeShort atIndex:1200];
        [unary_silu_constants setConstantValue:&unary_cnt     type:MTLDataTypeBool  atIndex:1201];

        error = nil;
        fn = [library newFunctionWithName:@"kernel_unary_f32_f32_4"
                           constantValues:unary_silu_constants
                                    error:&error];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_unary_f32_f32_4 silu function not found: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_unary_silu_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_unary_silu_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_unary_f32_f32_4 silu pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        MTLFunctionConstantValues *unary_softplus_constants = [[MTLFunctionConstantValues alloc] init];
        int16_t unary_softplus_op = 115;
        [unary_softplus_constants setConstantValue:&unary_softplus_op type:MTLDataTypeShort atIndex:1200];
        [unary_softplus_constants setConstantValue:&unary_cnt         type:MTLDataTypeBool  atIndex:1201];

        error = nil;
        fn = [library newFunctionWithName:@"kernel_unary_f32_f32_4"
                           constantValues:unary_softplus_constants
                                    error:&error];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_unary_f32_f32_4 softplus function not found: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_unary_softplus_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_unary_softplus_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_unary_f32_f32_4 softplus pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        MTLFunctionConstantValues *unary_sqrt_constants = [[MTLFunctionConstantValues alloc] init];
        int16_t unary_sqrt_op = 14;
        [unary_sqrt_constants setConstantValue:&unary_sqrt_op type:MTLDataTypeShort atIndex:1200];
        [unary_sqrt_constants setConstantValue:&unary_cnt     type:MTLDataTypeBool  atIndex:1201];

        error = nil;
        fn = [library newFunctionWithName:@"kernel_unary_f32_f32_4"
                           constantValues:unary_sqrt_constants
                                    error:&error];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_unary_f32_f32_4 sqrt function not found: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_unary_sqrt_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_unary_sqrt_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_unary_f32_f32_4 sqrt pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        MTLFunctionConstantValues *unary_clamp_constants = [[MTLFunctionConstantValues alloc] init];
        int16_t unary_clamp_op = 12;
        [unary_clamp_constants setConstantValue:&unary_clamp_op type:MTLDataTypeShort atIndex:1200];
        [unary_clamp_constants setConstantValue:&unary_cnt      type:MTLDataTypeBool  atIndex:1201];

        error = nil;
        fn = [library newFunctionWithName:@"kernel_unary_f32_f32"
                           constantValues:unary_clamp_constants
                                    error:&error];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_unary_f32_f32 clamp function not found: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_unary_clamp_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_unary_clamp_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_unary_f32_f32 clamp pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        MTLFunctionConstantValues *unary_scale_constants = [[MTLFunctionConstantValues alloc] init];
        int16_t unary_scale_op = 10;
        [unary_scale_constants setConstantValue:&unary_scale_op type:MTLDataTypeShort atIndex:1200];
        [unary_scale_constants setConstantValue:&unary_cnt      type:MTLDataTypeBool  atIndex:1201];

        error = nil;
        fn = [library newFunctionWithName:@"kernel_unary_f32_f32_4"
                           constantValues:unary_scale_constants
                                    error:&error];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_unary_f32_f32_4 scale function not found: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_unary_scale_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_unary_scale_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_unary_f32_f32_4 scale pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        MTLFunctionConstantValues *unary_fill_constants = [[MTLFunctionConstantValues alloc] init];
        int16_t unary_fill_op = 11;
        [unary_fill_constants setConstantValue:&unary_fill_op type:MTLDataTypeShort atIndex:1200];
        [unary_fill_constants setConstantValue:&unary_cnt     type:MTLDataTypeBool  atIndex:1201];

        error = nil;
        fn = [library newFunctionWithName:@"kernel_unary_f32_f32_4"
                           constantValues:unary_fill_constants
                                    error:&error];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_unary_f32_f32_4 fill function not found: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_unary_fill_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_unary_fill_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_unary_f32_f32_4 fill pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        error = nil;
        fn = [library newFunctionWithName:@"kernel_unary_f16_f16"
                           constantValues:unary_fill_constants
                                    error:&error];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_unary_f16_f16 fill function not found: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_unary_fill_f16_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_unary_fill_f16_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_unary_f16_f16 fill pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_dsv4_hc_expand"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_dsv4_hc_expand function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_hc_expand_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_hc_expand_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_dsv4_hc_expand pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        g_dsv4_indexer_score_one_direct_pipeline =
            ds4_gpu_get_pipeline("kernel_dsv4_indexer_score_one_direct");
        g_dsv4_compressor_store_one_pipeline =
            ds4_gpu_get_pipeline("kernel_dsv4_compressor_store_one");
        g_dsv4_compressor_pack_ratio4_pipeline =
            ds4_gpu_get_pipeline("kernel_dsv4_compressor_pack_ratio4");
        g_dsv4_softmax_pool_ratio4_direct_pipeline =
            ds4_gpu_get_pipeline("kernel_dsv4_softmax_pool_ratio4_direct");
        g_rms_norm_scale_pipeline =
            ds4_gpu_get_pipeline("kernel_rms_norm_scale_f32_4");
        g_dsv4_sort_i32_rows_asc_pipeline =
            ds4_gpu_get_pipeline("kernel_dsv4_sort_i32_rows_asc");
        g_dsv4_indexed_attention_heads8_pipeline =
            ds4_gpu_get_pipeline("kernel_dsv4_indexed_mixed_attention_heads8");
        g_dsv4_indexed_attention_heads8_rb16_pipeline =
            ds4_gpu_get_pipeline("kernel_dsv4_indexed_mixed_attention_heads8_rb16");
        g_dsv4_softplus_sqrt_pipeline =
            ds4_gpu_get_pipeline("kernel_dsv4_softplus_sqrt_f32_4");
        g_dsv4_router_finalize_one_pipeline =
            ds4_gpu_get_pipeline("kernel_dsv4_router_finalize_one");
        g_dsv4_router_finalize_one_simd_pipeline =
            ds4_gpu_get_pipeline("kernel_dsv4_router_finalize_one_simd");
        g_dsv4_router_finalize_weights_one_simd_pipeline =
            ds4_gpu_get_pipeline("kernel_dsv4_router_finalize_weights_one_simd");
        g_dsv4_router_weights_one_pipeline =
            ds4_gpu_get_pipeline("kernel_dsv4_router_weights_one");
        g_glm_router_select_one_pipeline =
            ds4_gpu_get_pipeline("kernel_glm_router_select_one");
        g_glm_kv_lora_rms_norm_pipeline =
            ds4_gpu_get_pipeline("kernel_glm_kv_lora_rms_norm");
        g_glm_k_b_project_pipeline =
            ds4_gpu_get_pipeline("kernel_glm_k_b_project_q8_0");
        g_glm_store_compact_kv_pipeline =
            ds4_gpu_get_pipeline("kernel_glm_store_compact_kv");
        g_glm_qkv_norm_store_compact_kv_pipeline =
            ds4_gpu_get_pipeline("kernel_glm_qkv_norm_store_compact_kv");
        g_glm_store_indexer_k_pipeline =
            ds4_gpu_get_pipeline("kernel_glm_store_indexer_k");
        g_glm_build_kv_cache_pipeline =
            ds4_gpu_get_pipeline("kernel_glm_build_kv_cache");
        g_glm_build_kv_cache_decode_group4_pipeline =
            ds4_gpu_get_pipeline("kernel_glm_build_kv_cache_decode_group4");
        g_glm_build_kv_cache_flash_pipeline =
            ds4_gpu_get_pipeline("kernel_glm_build_kv_cache_flash");
        g_glm_attention_full_pipeline =
            ds4_gpu_get_pipeline("kernel_glm_attention_full");
        g_glm_fill_selected_range_pipeline =
            ds4_gpu_get_pipeline("kernel_glm_fill_selected_range");
        g_glm_fill_selected_range_batch_pipeline =
            ds4_gpu_get_pipeline("kernel_glm_fill_selected_range_batch");
        g_glm_indexer_rope_tail_pipeline =
            ds4_gpu_get_pipeline("kernel_glm_indexer_rope_tail_f32");
        g_glm_indexer_score_one_pipeline =
            ds4_gpu_get_pipeline("kernel_glm_indexer_score_one");
        g_glm_indexer_score_one_direct_pipeline =
            ds4_gpu_get_pipeline("kernel_glm_indexer_score_one_direct");
        g_glm_indexer_scores_batch_pipeline =
            ds4_gpu_get_pipeline("kernel_glm_indexer_scores_batch");
        g_glm_indexer_scores_tiled_pipeline =
            ds4_gpu_get_pipeline("kernel_glm_indexer_scores_tiled");
        g_glm_indexer_scores_tiled_f32_pipeline =
            ds4_gpu_get_pipeline("kernel_glm_indexer_scores_tiled_f32");
        g_glm_qk_lowrank_pipeline =
            ds4_gpu_get_pipeline("kernel_glm_qk_lowrank_q8_0");
        g_glm_qk_lowrank_glm52_pipeline =
            ds4_gpu_get_pipeline("kernel_glm_qk_lowrank_q8_0_glm52");
        g_glm_qk_lowrank_glm52_sg_pipeline =
            ds4_gpu_get_pipeline("kernel_glm_qk_lowrank_q8_0_glm52_sg");
        g_glm_qk_lowrank_batch_pipeline =
            ds4_gpu_get_pipeline("kernel_glm_qk_lowrank_q8_0_batch");
        g_glm_qk_lowrank_batch_glm52_t4_pipeline =
            ds4_gpu_get_pipeline("kernel_glm_qk_lowrank_q8_0_batch_glm52_t4");
        g_glm_value_project_q8_0_pipeline =
            ds4_gpu_get_pipeline("kernel_glm_value_project_q8_0");
        g_glm_value_project_q8_0_batch_heads_pipeline =
            ds4_gpu_get_pipeline("kernel_glm_value_project_q8_0_batch_heads");
        g_glm_value_project_q8_0_batch_heads_mma_pipeline =
            ds4_gpu_get_pipeline("kernel_glm_value_project_q8_0_batch_heads_mma");
        g_glm_attention_indexed_decode_pipeline =
            ds4_gpu_get_pipeline("kernel_glm_attention_indexed_decode");
        g_glm_attention_indexed_decode_split_group8_partial_pipeline =
            ds4_gpu_get_pipeline("kernel_glm_attention_indexed_decode_split_group8_partial");
        g_glm_attention_indexed_decode_split_group8_partial_valid_fullheads_pipeline =
            ds4_gpu_get_pipeline("kernel_glm_attention_indexed_decode_split_group8_partial_valid_fullheads");
        g_glm_attention_indexed_decode_split_group8_reduce_pipeline =
            ds4_gpu_get_pipeline("kernel_glm_attention_indexed_decode_split_group8_reduce");
        g_glm_attention_indexed_decode_split_group8_reduce16_pipeline =
            ds4_gpu_get_pipeline("kernel_glm_attention_indexed_decode_split_group8_reduce16");
        g_glm_attention_indexed_batch_pipeline =
            ds4_gpu_get_pipeline("kernel_glm_attention_indexed_batch");
        g_glm_attention_indexed_batch_group2_pipeline =
            ds4_gpu_get_pipeline("kernel_glm_attention_indexed_batch_group2");
        g_glm_attention_indexed_batch_q2_group4_pipeline =
            ds4_gpu_get_pipeline("kernel_glm_attention_indexed_batch_q2_group4");
        g_glm_attention_indexed_batch_group8_pipeline =
            ds4_gpu_get_pipeline("kernel_glm_attention_indexed_batch_group8");
        g_glm_attention_indexed_batch_lora_group8_vec_pipeline =
            ds4_gpu_get_pipeline("kernel_glm_attention_indexed_batch_lora_group8_vec");
        g_glm_attention_indexed_batch_lora_group8_vec_valid_pipeline =
            ds4_gpu_get_pipeline("kernel_glm_attention_indexed_batch_lora_group8_vec_valid");
        g_glm_attention_indexed_batch_lora_group8_vec_valid_fullheads_pipeline =
            ds4_gpu_get_pipeline("kernel_glm_attention_indexed_batch_lora_group8_vec_valid_fullheads");
        g_glm_attention_indexed_batch_lora_group8_vec_causal_pipeline =
            ds4_gpu_get_pipeline("kernel_glm_attention_indexed_batch_lora_group8_vec_causal");
        g_glm_attention_indexed_batch_lora_group8_vec_causal_fullheads_pipeline =
            ds4_gpu_get_pipeline("kernel_glm_attention_indexed_batch_lora_group8_vec_causal_fullheads");
        g_glm_q4_k_pair_swiglu_f32_pipeline =
            ds4_gpu_get_pipeline("kernel_glm_q4_K_pair_swiglu_f32");
        g_glm_q4_k_pair_swiglu2_f32_pipeline =
            ds4_gpu_get_pipeline("kernel_glm_q4_K_pair_swiglu2_f32");
        g_glm_q4_k_pair_swiglu4_f32_pipeline =
            ds4_gpu_get_pipeline("kernel_glm_q4_K_pair_swiglu4_f32");
        g_glm_q4_k_pair_swiglu2_mapped_f32_pipeline =
            ds4_gpu_get_pipeline("kernel_glm_q4_K_pair_swiglu2_mapped_f32");
        g_glm_q4_k_pair_swiglu2_mapped_row_f32_pipeline =
            ds4_gpu_get_pipeline("kernel_glm_q4_K_pair_swiglu2_mapped_row_f32");
        g_glm_q2_k_pair_swiglu_f32_pipeline =
            ds4_gpu_get_pipeline("kernel_glm_q2_K_pair_swiglu_f32");
        g_glm_q2_k_addr_pair_swiglu2_f32_pipeline =
            ds4_gpu_get_pipeline("kernel_glm_q2_K_addr_pair_swiglu2_f32");
        g_glm_q2_k_addr_pair_swiglu2_masked_f32_pipeline =
            ds4_gpu_get_pipeline("kernel_glm_q2_K_addr_pair_swiglu2_f32_masked");
        g_glm_q4_k_addr_pair_swiglu_f32_pipeline =
            ds4_gpu_get_pipeline("kernel_glm_q4_K_addr_pair_swiglu_f32");
        g_glm_q4_k_addr_pair_swiglu_masked_f32_pipeline =
            ds4_gpu_get_pipeline("kernel_glm_q4_K_addr_pair_swiglu_f32_masked");
        g_glm_q2_k_down_f32_pipeline =
            ds4_gpu_get_pipeline("kernel_glm_q2_K_down_f32");
        g_glm_q4_k_down_f32_pipeline =
            ds4_gpu_get_pipeline("kernel_glm_q4_K_down_simd_f32");
        g_glm_q2_k_addr_down_f32_pipeline =
            ds4_gpu_get_pipeline("kernel_glm_q2_K_addr_down_f32");
        g_glm_q4_k_addr_down_f32_pipeline =
            ds4_gpu_get_pipeline("kernel_glm_q4_K_addr_down_simd_f32");
        g_glm_q5_k_pair_swiglu_f32_pipeline =
            ds4_gpu_get_pipeline("kernel_glm_q5_K_pair_swiglu_f32");
        g_glm_q5_k_pair_swiglu_mapped_f32_pipeline =
            ds4_gpu_get_pipeline("kernel_glm_q5_K_pair_swiglu_mapped_f32");
        g_glm_q5_k_pair_swiglu_mapped_row_f32_pipeline =
            ds4_gpu_get_pipeline("kernel_glm_q5_K_pair_swiglu_mapped_row_f32");
        g_glm_q5_k_down_f32_pipeline =
            ds4_gpu_get_pipeline("kernel_glm_q5_K_down_f32");
        g_glm_q6_k_down_f32_pipeline =
            ds4_gpu_get_pipeline("kernel_glm_q6_K_down_f32");
        g_dsv4_router_weights_batch_pipeline =
            ds4_gpu_get_pipeline("kernel_dsv4_router_weights_batch");
        g_dsv4_hc_expand4_pipeline =
            ds4_gpu_get_pipeline("kernel_dsv4_hc_expand4");
        if (!g_dsv4_indexer_score_one_direct_pipeline ||
            !g_dsv4_compressor_store_one_pipeline ||
            !g_dsv4_sort_i32_rows_asc_pipeline ||
            !g_dsv4_indexed_attention_heads8_pipeline ||
            !g_dsv4_indexed_attention_heads8_rb16_pipeline ||
            !g_dsv4_softplus_sqrt_pipeline ||
            !g_dsv4_router_finalize_one_pipeline ||
            !g_dsv4_router_weights_one_pipeline ||
            !g_glm_router_select_one_pipeline ||
            !g_glm_kv_lora_rms_norm_pipeline ||
            !g_glm_k_b_project_pipeline ||
            !g_glm_store_compact_kv_pipeline ||
            !g_glm_qkv_norm_store_compact_kv_pipeline ||
            !g_glm_store_indexer_k_pipeline ||
            !g_glm_build_kv_cache_pipeline ||
            !g_glm_build_kv_cache_decode_group4_pipeline ||
            !g_glm_build_kv_cache_flash_pipeline ||
            !g_glm_attention_full_pipeline ||
            !g_glm_fill_selected_range_pipeline ||
            !g_glm_fill_selected_range_batch_pipeline ||
            !g_glm_indexer_rope_tail_pipeline ||
            !g_glm_indexer_score_one_pipeline ||
            !g_glm_indexer_score_one_direct_pipeline ||
            !g_glm_indexer_scores_batch_pipeline ||
            !g_glm_indexer_scores_tiled_pipeline ||
            !g_glm_indexer_scores_tiled_f32_pipeline ||
            !g_glm_qk_lowrank_pipeline ||
            !g_glm_qk_lowrank_glm52_pipeline ||
            !g_glm_qk_lowrank_glm52_sg_pipeline ||
            !g_glm_qk_lowrank_batch_pipeline ||
            !g_glm_qk_lowrank_batch_glm52_t4_pipeline ||
            !g_glm_value_project_q8_0_pipeline ||
            !g_glm_value_project_q8_0_batch_heads_pipeline ||
            !g_glm_value_project_q8_0_batch_heads_mma_pipeline ||
            !g_glm_attention_indexed_decode_pipeline ||
            !g_glm_attention_indexed_decode_split_group8_partial_pipeline ||
            !g_glm_attention_indexed_decode_split_group8_partial_valid_fullheads_pipeline ||
            !g_glm_attention_indexed_decode_split_group8_reduce_pipeline ||
            !g_glm_attention_indexed_decode_split_group8_reduce16_pipeline ||
            !g_glm_attention_indexed_batch_pipeline ||
            !g_glm_attention_indexed_batch_group2_pipeline ||
            !g_glm_attention_indexed_batch_q2_group4_pipeline ||
            !g_glm_attention_indexed_batch_group8_pipeline ||
            !g_glm_attention_indexed_batch_lora_group8_vec_pipeline ||
            !g_glm_attention_indexed_batch_lora_group8_vec_valid_pipeline ||
            !g_glm_attention_indexed_batch_lora_group8_vec_valid_fullheads_pipeline ||
            !g_glm_attention_indexed_batch_lora_group8_vec_causal_pipeline ||
            !g_glm_attention_indexed_batch_lora_group8_vec_causal_fullheads_pipeline ||
            !g_glm_q4_k_pair_swiglu_f32_pipeline ||
            !g_glm_q4_k_pair_swiglu2_f32_pipeline ||
            !g_glm_q4_k_pair_swiglu4_f32_pipeline ||
            !g_glm_q4_k_pair_swiglu2_mapped_f32_pipeline ||
            !g_glm_q4_k_pair_swiglu2_mapped_row_f32_pipeline ||
            !g_glm_q2_k_pair_swiglu_f32_pipeline ||
            !g_glm_q2_k_addr_pair_swiglu2_f32_pipeline ||
            !g_glm_q2_k_addr_pair_swiglu2_masked_f32_pipeline ||
            !g_glm_q4_k_addr_pair_swiglu_f32_pipeline ||
            !g_glm_q4_k_addr_pair_swiglu_masked_f32_pipeline ||
            !g_glm_q2_k_down_f32_pipeline ||
            !g_glm_q4_k_down_f32_pipeline ||
            !g_glm_q2_k_addr_down_f32_pipeline ||
            !g_glm_q4_k_addr_down_f32_pipeline ||
            !g_glm_q5_k_pair_swiglu_f32_pipeline ||
            !g_glm_q5_k_pair_swiglu_mapped_f32_pipeline ||
            !g_glm_q5_k_pair_swiglu_mapped_row_f32_pipeline ||
            !g_glm_q5_k_down_f32_pipeline ||
            !g_glm_q6_k_down_f32_pipeline ||
            !g_dsv4_hc_expand4_pipeline) {
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        g_initialized = 1;
    }

    return 1;
}

ds4_gpu_tensor *ds4_gpu_tensor_alloc(uint64_t bytes) {
    if (!g_initialized && !ds4_gpu_init()) return NULL;
    if (bytes == 0 || bytes > (uint64_t)NSUIntegerMax) return NULL;

    @autoreleasepool {
        DS4MetalTensor *tensor = [DS4MetalTensor new];
        tensor.buffer = [g_device newBufferWithLength:(NSUInteger)bytes
                                              options:MTLResourceStorageModeShared];
        if (!tensor.buffer) {
            return NULL;
        }
        tensor.offset = 0;
        tensor.bytes = bytes;
        tensor.owner = 1;
        uint64_t live_snap = 0;
        uint64_t peak_snap = 0;
        pthread_mutex_lock(&g_tensor_mu);
        const int tracked = ds4_gpu_tensor_track_alloc_locked(
                (__bridge const void *)tensor,
                bytes,
                &live_snap,
                &peak_snap);
        pthread_mutex_unlock(&g_tensor_mu);
        if (!tracked) {
            fprintf(stderr, "ds4: failed to track Metal tensor allocation\n");
            tensor.buffer = nil;
            return NULL;
        }
        if (ds4_gpu_trace_allocs()) {
            fprintf(stderr,
                    "ds4: Metal tensor alloc %.3f MiB live %.3f MiB peak %.3f MiB\n",
                    (double)bytes / (1024.0 * 1024.0),
                    (double)live_snap / (1024.0 * 1024.0),
                    (double)peak_snap / (1024.0 * 1024.0));
        }
        return (__bridge_retained ds4_gpu_tensor *)tensor;
    }
}

ds4_gpu_tensor *ds4_gpu_tensor_alloc_managed(uint64_t bytes) {
    return ds4_gpu_tensor_alloc(bytes);
}

int ds4_gpu_should_use_managed_kv_cache(uint64_t kv_cache_bytes, uint64_t context_bytes) {
    (void)kv_cache_bytes;
    (void)context_bytes;
    return 0;
}

ds4_gpu_tensor *ds4_gpu_tensor_view(const ds4_gpu_tensor *base, uint64_t offset, uint64_t bytes) {
    if (!base) return NULL;
    const DS4MetalTensor *base_obj = ds4_gpu_tensor_const_obj(base);
    if (offset > base_obj.bytes || bytes > base_obj.bytes - offset) return NULL;
    if (base_obj.offset > UINT64_MAX - offset) return NULL;
    const uint64_t absolute_offset = base_obj.offset + offset;
    if (absolute_offset > (uint64_t)NSUIntegerMax) return NULL;

    @autoreleasepool {
        DS4MetalTensor *view = [DS4MetalTensor new];
        view.buffer = base_obj.buffer;
        view.offset = absolute_offset;
        view.bytes = bytes;
        view.owner = 0;
        pthread_mutex_lock(&g_tensor_mu);
        const int tracked = ds4_gpu_tensor_track_view_locked((__bridge const void *)view);
        pthread_mutex_unlock(&g_tensor_mu);
        if (!tracked) {
            fprintf(stderr, "ds4: failed to track Metal tensor view\n");
            view.buffer = nil;
            return NULL;
        }
        return (__bridge_retained ds4_gpu_tensor *)view;
    }
}

void ds4_gpu_tensor_free(ds4_gpu_tensor *tensor) {
    if (!tensor) return;
    @autoreleasepool {
        uint8_t owner = 0;
        uint64_t bytes = 0;
        uint64_t live_snap = 0;
        uint64_t peak_snap = 0;
        if (!ds4_gpu_tensor_prepare_free(tensor,
                                         &owner,
                                         &bytes,
                                         &live_snap,
                                         &peak_snap)) {
            return;
        }
        DS4MetalTensor *obj = (__bridge_transfer DS4MetalTensor *)tensor;
        if (owner) {
            if (ds4_gpu_trace_allocs()) {
                fprintf(stderr,
                        "ds4: Metal tensor free %.3f MiB live %.3f MiB peak %.3f MiB\n",
                        (double)bytes / (1024.0 * 1024.0),
                        (double)live_snap / (1024.0 * 1024.0),
                        (double)peak_snap / (1024.0 * 1024.0));
            }
        }
        obj.buffer = nil;
        obj.offset = 0;
        obj.bytes = 0;
        obj.owner = 0;
    }
}

uint64_t ds4_gpu_tensor_bytes(const ds4_gpu_tensor *tensor) {
    if (!tensor) return 0;
    const DS4MetalTensor *obj = ds4_gpu_tensor_const_obj(tensor);
    return obj.bytes;
}

void *ds4_gpu_tensor_contents(ds4_gpu_tensor *tensor) {
    if (!tensor) return NULL;
    DS4MetalTensor *obj = ds4_gpu_tensor_obj(tensor);
    return (uint8_t *)[obj.buffer contents] + obj.offset;
}

int ds4_gpu_tensor_fill_f32(ds4_gpu_tensor *tensor, float value, uint64_t count) {
    if (!tensor || count > ds4_gpu_tensor_bytes(tensor) / sizeof(float)) return 0;
    float *p = ds4_gpu_tensor_contents(tensor);
    if (!p && count != 0) return 0;
    for (uint64_t i = 0; i < count; i++) p[i] = value;
    return 1;
}

int ds4_gpu_tensor_write(ds4_gpu_tensor *tensor, uint64_t offset, const void *data, uint64_t bytes) {
    if (!tensor || (!data && bytes != 0)) return 0;
    DS4MetalTensor *obj = ds4_gpu_tensor_obj(tensor);
    if (offset > obj.bytes || bytes > obj.bytes - offset) return 0;
    if (bytes != 0) {
        memcpy((uint8_t *)[obj.buffer contents] + obj.offset + offset, data, (size_t)bytes);
    }
    return 1;
}

int ds4_gpu_tensor_read(const ds4_gpu_tensor *tensor, uint64_t offset, void *data, uint64_t bytes) {
    if (!tensor || (!data && bytes != 0)) return 0;
    const DS4MetalTensor *obj = ds4_gpu_tensor_const_obj(tensor);
    if (offset > obj.bytes || bytes > obj.bytes - offset) return 0;
    if (bytes != 0) {
        memcpy(data, (const uint8_t *)[obj.buffer contents] + obj.offset + offset, (size_t)bytes);
    }
    return 1;
}

int ds4_gpu_tensor_copy(ds4_gpu_tensor *dst, uint64_t dst_offset,
                          const ds4_gpu_tensor *src, uint64_t src_offset,
                          uint64_t bytes) {
    if (!dst || !src) return 0;
    if (!g_initialized && !ds4_gpu_init()) return 0;
    DS4MetalTensor *d = ds4_gpu_tensor_obj(dst);
    const DS4MetalTensor *s = ds4_gpu_tensor_const_obj(src);
    if (dst_offset > d.bytes || bytes > d.bytes - dst_offset) return 0;
    if (src_offset > s.bytes || bytes > s.bytes - src_offset) return 0;
    if (bytes == 0) return 1;
    if (!g_batch_cb) return 0;

    ds4_gpu_close_batch_encoder();
    g_batch_has_work = YES;
    id<MTLBlitCommandEncoder> blit = [g_batch_cb blitCommandEncoder];
    if (!blit) return 0;
    [blit copyFromBuffer:s.buffer
            sourceOffset:(NSUInteger)(s.offset + src_offset)
                toBuffer:d.buffer
       destinationOffset:(NSUInteger)(d.offset + dst_offset)
                    size:(NSUInteger)bytes];
    [blit endEncoding];
    return 1;
}

int ds4_gpu_tensor_copy_f32_to_f16(ds4_gpu_tensor *dst, uint64_t dst_offset,
                                   const ds4_gpu_tensor *src, uint64_t src_offset,
                                   uint64_t count) {
    if (!dst || !src) return 0;
    if (!g_initialized && !ds4_gpu_init()) return 0;
    DS4MetalTensor *d = ds4_gpu_tensor_obj(dst);
    const DS4MetalTensor *s = ds4_gpu_tensor_const_obj(src);
    if (count == 0) return 1;
    if (count > UINT64_MAX / sizeof(float) ||
        count > UINT64_MAX / sizeof(uint16_t)) {
        return 0;
    }
    const uint64_t src_bytes = count * sizeof(float);
    const uint64_t dst_bytes = count * sizeof(uint16_t);
    if (src_offset > s.bytes || src_bytes > s.bytes - src_offset ||
        dst_offset > d.bytes || dst_bytes > d.bytes - dst_offset) {
        return 0;
    }

    @autoreleasepool {
        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        uint64_t done = 0;
        int ok = 1;
        while (done < count && ok) {
            uint64_t chunk64 = count - done;
            if (chunk64 > UINT32_MAX) chunk64 = UINT32_MAX;
            const uint32_t chunk = (uint32_t)chunk64;
            ok = ds4_gpu_encode_cpy_f32_f16_1d(
                    cb,
                    s.buffer,
                    (NSUInteger)(s.offset + src_offset + done * sizeof(float)),
                    d.buffer,
                    (NSUInteger)(d.offset + dst_offset + done * sizeof(uint16_t)),
                    chunk);
            done += chunk;
        }
        if (ok) ok = ds4_gpu_finish_command_buffer(cb, owned, "tensor f32 to f16 copy");
        return ok;
    }
}

int ds4_gpu_pack_slot_rows_f32_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *slots,
        uint32_t                n_rows,
        uint32_t                width,
        uint32_t                n_slots,
        uint32_t                slot_cap) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!out || !slots || n_rows == 0 || width == 0 || n_slots == 0 ||
        slot_cap == 0 || n_rows > slot_cap) {
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> slotsbuf = ds4_gpu_tensor_buffer(slots);
        id<MTLBuffer> outbuf = ds4_gpu_tensor_buffer(out);
        uint64_t row_bytes = 0;
        uint64_t slot_plane_bytes = 0;
        uint64_t slots_bytes = 0;
        uint64_t out_rows = 0;
        uint64_t out_bytes = 0;
        if ((uint64_t)width > UINT64_MAX / sizeof(float)) return 0;
        row_bytes = (uint64_t)width * sizeof(float);
        if ((uint64_t)slot_cap > UINT64_MAX / row_bytes) return 0;
        slot_plane_bytes = (uint64_t)slot_cap * row_bytes;
        if ((uint64_t)n_slots > UINT64_MAX / slot_plane_bytes) return 0;
        slots_bytes = (uint64_t)n_slots * slot_plane_bytes;
        if ((uint64_t)n_rows > UINT64_MAX / n_slots) return 0;
        out_rows = (uint64_t)n_rows * n_slots;
        if (out_rows > UINT64_MAX / row_bytes) return 0;
        out_bytes = out_rows * row_bytes;
        if (!slotsbuf || !outbuf ||
            ds4_gpu_tensor_bytes(slots) < slots_bytes ||
            ds4_gpu_tensor_bytes(out) < out_bytes) {
            fprintf(stderr, "ds4: Metal slot-row pack received undersized buffers\n");
            return 0;
        }

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;
        if (!ds4_gpu_encode_cpy_f32_f32_3d_src_strided(cb,
                                                        slotsbuf,
                                                        ds4_gpu_tensor_offset(slots),
                                                        outbuf,
                                                        ds4_gpu_tensor_offset(out),
                                                        width,
                                                        n_slots,
                                                        n_rows,
                                                        sizeof(float),
                                                        slot_plane_bytes,
                                                        row_bytes,
                                                        row_bytes,
                                                        (uint64_t)n_slots * row_bytes)) {
            return 0;
        }
        if (!ds4_gpu_finish_command_buffer(cb, owned, "slot-row pack")) return 0;
    }

    return 1;
}

int ds4_gpu_begin_commands(void) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (g_batch_cb) return 0;
    g_batch_cb = ds4_gpu_new_command_buffer();
    g_batch_has_work = NO;
    if (g_batch_cb) ds4_gpu_stream_expert_cache_note_batch_created();
    return g_batch_cb != nil;
}

int ds4_gpu_flush_encoder(void) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!g_batch_cb) return 0;
    ds4_gpu_close_batch_encoder();
    return 1;
}

int ds4_gpu_flush_commands(void) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!g_batch_cb) return 0;

    ds4_gpu_close_batch_encoder();
    id<MTLCommandBuffer> cb = g_batch_cb;
    g_batch_cb = nil;
    g_batch_has_work = NO;
    [cb commit];
    [g_pending_cbs addObject:cb];
    ds4_gpu_stream_expert_cache_note_batch_committed();

    g_batch_cb = ds4_gpu_new_command_buffer();
    g_batch_has_work = NO;
    if (g_batch_cb) ds4_gpu_stream_expert_cache_note_batch_created();
    if (!g_batch_cb) {
        (void)ds4_gpu_wait_pending_command_buffers("command batch");
        [g_transient_buffers removeAllObjects];
        return 0;
    }
    return 1;
}

int ds4_gpu_commands_active(void) {
    return g_batch_cb != nil;
}

static int ds4_gpu_stream_expert_cache_wait_inflight(const char *label) {
    const char *what = label ? label : "streaming expert cache in-flight";
    if (g_batch_cb && ds4_gpu_flush_commands() == 0) return 0;
    if ([g_pending_cbs count] != 0 &&
        ds4_gpu_wait_pending_command_buffers(what) == 0) {
        return 0;
    }
    return 1;
}

int ds4_gpu_signal_selected_readback_ready(uint64_t *event_value) {
    if (!event_value) return 0;
    *event_value = 0;
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!g_batch_cb) return 0;

    if (@available(macOS 12.0, *)) {
        if (!g_selected_readback_event) {
            g_selected_readback_event = [g_device newSharedEvent];
            if (!g_selected_readback_event) {
                fprintf(stderr, "ds4: failed to create Metal shared event for selected-id overlap\n");
                return 0;
            }
        }

        ds4_gpu_close_batch_encoder();
        const uint64_t value = ++g_selected_readback_event_value;
        [g_batch_cb encodeSignalEvent:g_selected_readback_event value:value];
        g_batch_has_work = YES;
        *event_value = value;
        return 1;
    }

    fprintf(stderr, "ds4: selected-id overlap requires MTLSharedEvent support\n");
    return 0;
}

int ds4_gpu_commit_and_wait_selected_readback(uint64_t event_value, const char *label) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!g_batch_cb || event_value == 0) return 0;

    if (@available(macOS 12.0, *)) {
        if (!g_selected_readback_event) return 0;

        ds4_gpu_close_batch_encoder();
        id<MTLCommandBuffer> cb = g_batch_cb;
        g_batch_cb = nil;
        g_batch_has_work = NO;
        [cb commit];
        ds4_gpu_stream_expert_cache_note_batch_committed();

        const char *what = label ? label : "selected-id overlap";
        const BOOL signaled =
            [g_selected_readback_event waitUntilSignaledValue:event_value timeoutMS:60000];
        [g_pending_cbs addObject:cb];
        if (!signaled) {
            fprintf(stderr, "ds4: timeout waiting for Metal shared event in %s\n", what);
            (void)ds4_gpu_wait_pending_command_buffers(what);
            [g_transient_buffers removeAllObjects];
            return 0;
        }
        if (cb.status == MTLCommandBufferStatusError) {
            fprintf(stderr, "ds4: Metal %s failed: %s\n",
                    what,
                    [[cb.error localizedDescription] UTF8String]);
            (void)ds4_gpu_wait_pending_command_buffers(what);
            [g_transient_buffers removeAllObjects];
            return 0;
        }

        g_batch_cb = ds4_gpu_new_command_buffer();
        g_batch_has_work = NO;
        if (g_batch_cb) ds4_gpu_stream_expert_cache_note_batch_created();
        if (!g_batch_cb) {
            (void)ds4_gpu_wait_pending_command_buffers(what);
            [g_transient_buffers removeAllObjects];
            return 0;
        }
        return 1;
    }

    fprintf(stderr, "ds4: selected-id overlap requires MTLSharedEvent support\n");
    return 0;
}

/*
 * Tensor-parallel gates.
 *
 * A TP gate is a mid-command-stream rendezvous with the peer machine: the
 * kernels ahead of the gate leave a partial block output in a slab slot,
 * the GPU signals g_tp_gpu_event, and the pre-encoded combine kernel waits
 * on g_tp_cpu_event.  A dedicated service thread bridges the two: it spins
 * until the GPU reaches the gate, runs the transport exchange (RDMA WRITE
 * plus flag poll, or a TCP write/read pair — behind the callback), and
 * CPU-signals the release.  On exchange failure the release is signaled
 * anyway so the GPU never deadlocks; the failure latches in g_tp_failed
 * and the eval aborts at the next command-buffer boundary.
 *
 * Gate sequence values increase monotonically per encoded gate.  Both ranks
 * encode the identical graph, so the values agree by construction and slots
 * never need resetting between tokens.
 */
typedef struct {
    uint32_t layer;
    uint32_t gate;
    uint32_t rows;  /* 0 = row gate; >0 = verify-block batch gate */
    uint32_t event_arrival;
    uint64_t seq;
    /* Big batch gates (prefill): exchange big_bytes from big_out into
     * big_in directly (CPU-visible bounce buffers), bypassing the slab. */
    const void *big_out;
    void *big_in;
    uint64_t big_bytes;
} ds4_gpu_tp_request;

enum { DS4_GPU_TP_QUEUE = 1024 };

static id<MTLSharedEvent> g_tp_gpu_event;   /* GPU -> service thread */
static id<MTLSharedEvent> g_tp_cpu_event;   /* service thread -> GPU */
/* Batch (verify-block) gates run on their own sequence space and release
 * event: the row-gate seq feeds the RDMA pre-posted recv accounting, which
 * requires consecutive values, and a shared release event would make a
 * small batch value satisfy waits armed against the larger row seq. */
static id<MTLSharedEvent> g_tp_batch_gpu_event;
static id<MTLSharedEvent> g_tp_batch_cpu_event;
static uint64_t g_tp_batch_seq;
/* Batch flag values are tagged so a stale row-gate seq in the reused FFN
 * flag word can never satisfy a batch arrival spin (and vice versa). */
#define DS4_TP_BATCH_FLAG_TAG 0x80000000u
/* Expert-ownership split parameters for routed kernels. World 1 means TP is
 * not bound; world 2 assigns each rank one contiguous expert range. */
static int32_t g_tp_split_rank;
static int32_t g_tp_split_world = 1;
static int32_t g_tp_session_batch_mode;

static int ds4_gpu_tp_world_is_two(void) {
    return g_tp_split_world == 2;
}

/* Return the contiguous routed-expert range backed by this process. Rank 1
 * owns the high range and receives any odd-count remainder. */
static void ds4_gpu_tp_expert_range(uint32_t n_total_expert,
                                    uint32_t *first_expert,
                                    uint32_t *n_expert) {
    *first_expert = 0;
    *n_expert = n_total_expert;
    if (g_tp_split_world != 2) return;

    const uint32_t low_experts = n_total_expert / 2u;
    if (g_tp_split_rank == 1) {
        *first_expert = low_experts;
        *n_expert = n_total_expert - low_experts;
    } else {
        *n_expert = low_experts;
    }
}

/* Attention head split for GLM batch prefill: each rank computes a
 * contiguous half of the heads in the qk-low / attention-lora /
 * value-project batch kernels; the caller zeroes the unowned head range
 * of the heads buffer and combines the attn-output partials over the
 * TP big-gate exchange. */
static int32_t g_tp_attn_head_split;

void ds4_gpu_tp_set_attn_head_split(int enabled) {
    g_tp_attn_head_split = enabled ? 1 : 0;
}

static void ds4_gpu_tp_attn_head_range(uint32_t n_head,
                                       uint32_t group,
                                       uint32_t *head_base,
                                       uint32_t *head_count) {
    *head_base = 0;
    *head_count = n_head;
    if (!g_tp_attn_head_split || g_tp_split_world != 2) return;
    const uint32_t half = n_head / 2u;
    if (half == 0u || (half % group) != 0u || (n_head % 2u) != 0u) return;
    *head_count = half;
    *head_base = g_tp_split_rank == 1 ? half : 0u;
}
/* Flag gates (DS4_TP_FLAG_GATES): the GPU publishes gate arrival by storing
 * the sequence number into a slab word instead of signaling the shared
 * event; the service thread spin-reads it from shared memory, which wakes
 * hundreds of microseconds earlier than signaledValue polling.  The
 * CPU->GPU release direction stays on the shared event. */
static bool g_tp_flag_gates;
static id<MTLBuffer> g_tp_slab_buffer;
static NSUInteger g_tp_slab_buffer_off;
static volatile uint32_t *g_tp_gpu_flags;   /* CPU view of the flag words */
static uint64_t g_tp_gpu_flags_off;
static uint64_t g_tp_seq;
static ds4_gpu_tp_exchange_fn g_tp_exchange_fn;
static ds4_gpu_tp_batch_exchange_fn g_tp_batch_exchange_fn;
static ds4_gpu_tp_big_exchange_fn g_tp_big_exchange_fn;

void ds4_gpu_tp_set_big_exchange(ds4_gpu_tp_big_exchange_fn fn) {
    g_tp_big_exchange_fn = fn;
}

static void *g_tp_exchange_ud;
static pthread_t g_tp_thread;
static int g_tp_thread_running;
static int g_tp_shutdown;
static int g_tp_failed_flag;
static pthread_mutex_t g_tp_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_tp_cond = PTHREAD_COND_INITIALIZER;
static ds4_gpu_tp_request g_tp_queue[DS4_GPU_TP_QUEUE];
static uint32_t g_tp_queue_head;
static uint32_t g_tp_queue_count;

static uint64_t g_tp_stat_gates;
static double g_tp_stat_gpu_wait_ms;
static double g_tp_stat_exchange_ms;

/* GPU keep-alive (see kernel_dsv4_tp_keepalive): its own queue and thread,
 * alive exactly as long as the TP gate machinery. */
static id<MTLCommandQueue> g_tp_keepalive_queue;
static id<MTLBuffer> g_tp_keepalive_buffer;
static pthread_t g_tp_keepalive_thread;
static int g_tp_keepalive_running;

/* Nonzero while a verify block runs: the GPU is genuinely busy
 * there, so the keep-alive is a pure parasite (~2.3ms per 5-row block
 * measured against the single-machine verify). */
static volatile int g_tp_keepalive_paused;

void ds4_gpu_tp_keepalive_pause(int paused) {
    g_tp_keepalive_paused = paused;
}

void ds4_gpu_tp_set_session_batch_mode(int enabled) {
    g_tp_session_batch_mode = enabled ? 1 : 0;
}

static uint32_t ds4_gpu_tp_keepalive_tgs_from_env(void) {
    uint32_t ka_tgs = 1;
    const char *tgs_env = getenv("DS4_TP_KEEPALIVE_TGS");
    if (tgs_env) {
        int v = atoi(tgs_env);
        if (v > 0 && v <= 2048) ka_tgs = (uint32_t)v;
    }
    return ka_tgs;
}

static void *ds4_gpu_tp_keepalive_thread(void *arg) {
    (void)arg;
    /* Swept on the M5 Max pair: too few iterations lets clocks sag.  Current
     * TP split-resident Flash runs show 1.2M is a small Q4/Q2 decode win over
     * 800k, while two threadgroups waste work. */
    uint32_t iters = 1200000;
    const char *env = getenv("DS4_TP_KEEPALIVE_ITERS");
    if (env) iters = (uint32_t)atoi(env);
    /* One ALU-only threadgroup keeps the GPU from power-gating but does
     * not push the frequency governor; solo-vs-engine kernel gaps
     * (~1.7x) suggest decode runs well below max clocks. More TGs raise
     * apparent utilization without eating memory bandwidth. */
    uint32_t ka_tgs = ds4_gpu_tp_keepalive_tgs_from_env();
    id<MTLComputePipelineState> pipeline =
        ds4_gpu_get_pipeline("kernel_dsv4_tp_keepalive");
    if (!pipeline) {
        fprintf(stderr, "ds4: TP keep-alive pipeline missing\n");
        return NULL;
    }
    while (!g_tp_shutdown) {
        if (g_tp_keepalive_paused) {
            usleep(200);
            continue;
        }
        @autoreleasepool {
            id<MTLCommandBuffer> cb = [g_tp_keepalive_queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:pipeline];
            [enc setBuffer:g_tp_keepalive_buffer offset:0 atIndex:0];
            [enc setBytes:&iters length:sizeof(iters) atIndex:1];
            [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)ka_tgs, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
        }
    }
    return NULL;
}

static void *ds4_gpu_tp_service_thread(void *arg) {
    (void)arg;
    const bool profile = getenv("DS4_TP_GATE_PROFILE") != NULL;
    while (1) {
        pthread_mutex_lock(&g_tp_mutex);
        while (g_tp_queue_count == 0 && !g_tp_shutdown)
            pthread_cond_wait(&g_tp_cond, &g_tp_mutex);
        if (g_tp_shutdown && g_tp_queue_count == 0) {
            pthread_mutex_unlock(&g_tp_mutex);
            break;
        }
        ds4_gpu_tp_request req = g_tp_queue[g_tp_queue_head];
        g_tp_queue_head = (g_tp_queue_head + 1) % DS4_GPU_TP_QUEUE;
        g_tp_queue_count--;
        pthread_mutex_unlock(&g_tp_mutex);

        /* Wait for the GPU to reach this gate.  Tight spin: gate arrival is
         * on the decode critical path and normally tens of microseconds
         * out; yielding here measurably delays the release wake-up. */
        const double t0 = profile ? ds4_gpu_now_ms() : 0.0;
        uint32_t spins = 0;
        if (req.big_bytes > 0) {
            /* Big gates always signal arrival through the batch shared
             * event (see ds4_gpu_tp_big_gate_kick): the event completion
             * semantics are what guarantee the bounce payload is visible
             * before the exchange reads it. */
            while (g_tp_batch_gpu_event.signaledValue < req.seq) {
                if (g_tp_shutdown) break;
                if (++spins > (1u << 16)) sched_yield();
            }
        } else if (!req.event_arrival) {
            const uint32_t slot = req.layer * 2u + req.gate;
            uint32_t want = (uint32_t)req.seq;
            if (req.rows > 0)
                want = DS4_TP_BATCH_FLAG_TAG | (uint32_t)req.seq;
            while (__atomic_load_n(&g_tp_gpu_flags[slot], __ATOMIC_ACQUIRE) != want) {
                if (g_tp_shutdown) break;
                if (++spins > (1u << 20)) {
                    sched_yield();
                    spins = 0;
                }
            }
        } else if (req.rows > 0) {
            while (g_tp_batch_gpu_event.signaledValue < req.seq) {
                if (g_tp_shutdown) break;
                if (++spins > (1u << 16)) sched_yield();
            }
        } else {
            while (g_tp_gpu_event.signaledValue < req.seq) {
                if (g_tp_shutdown) break;
                if (++spins > (1u << 16)) sched_yield();
            }
        }
        const double t1 = profile ? ds4_gpu_now_ms() : 0.0;
        int ok = 0;
        if (!g_tp_shutdown && !g_tp_failed_flag) {
            if (req.big_bytes > 0) {
                if (g_tp_big_exchange_fn)
                    ok = g_tp_big_exchange_fn(g_tp_exchange_ud, req.layer,
                                              req.seq, req.big_out,
                                              req.big_in, req.big_bytes);
            } else if (req.rows > 0) {
                if (g_tp_batch_exchange_fn)
                    ok = g_tp_batch_exchange_fn(g_tp_exchange_ud, req.layer,
                                                req.rows, req.seq);
            } else if (g_tp_exchange_fn) {
                ok = g_tp_exchange_fn(g_tp_exchange_ud, req.layer, req.gate,
                                      req.seq);
            }
        }
        if (!ok && !g_tp_shutdown) {
            if (!g_tp_failed_flag)
                fprintf(stderr, "ds4: TP gate exchange failed (layer %u gate %u seq %llu)\n",
                        req.layer, req.gate, (unsigned long long)req.seq);
            g_tp_failed_flag = 1;
        }
        /* Release the GPU even on failure so end_commands can drain. */
        if (req.rows > 0) g_tp_batch_cpu_event.signaledValue = req.seq;
        else g_tp_cpu_event.signaledValue = req.seq;
        if (profile) {
            g_tp_stat_gpu_wait_ms += t1 - t0;
            g_tp_stat_exchange_ms += ds4_gpu_now_ms() - t1;
            if (++g_tp_stat_gates % 860 == 0) {
                fprintf(stderr,
                        "ds4: TP gates %llu: avg gpu-wait %.1f us, avg exchange %.1f us\n",
                        (unsigned long long)g_tp_stat_gates,
                        g_tp_stat_gpu_wait_ms / (double)g_tp_stat_gates * 1000.0,
                        g_tp_stat_exchange_ms / (double)g_tp_stat_gates * 1000.0);
            }
        }
    }
    return NULL;
}

int ds4_gpu_tp_init(uint32_t rank,
                    ds4_gpu_tensor *slab, uint64_t gpu_flags_off,
                    ds4_gpu_tp_exchange_fn fn, void *ud) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (g_tp_thread_running || rank > 1) return 0;
    g_tp_split_rank = (int32_t)rank;
    g_tp_split_world = 2;
    g_tp_slab_buffer = slab ? ds4_gpu_tensor_buffer(slab) : nil;
    g_tp_slab_buffer_off = slab ? ds4_gpu_tensor_offset(slab) : 0;
    g_tp_gpu_flags_off = gpu_flags_off;
    g_tp_gpu_flags = slab ?
        (volatile uint32_t *)((uint8_t *)ds4_gpu_tensor_contents(slab) + gpu_flags_off) : NULL;
    /* Flag arrival is the default: the slab-word publish detects in ~1-3us
     * where signaledValue polling costs 10-20, worth +2.3 t/s on the pair
     * (A/B 2026-07-06, byte-identical output).  DS4_TP_EVENT_GATES falls
     * back to the shared-event arrival path. */
    g_tp_flag_gates = g_tp_gpu_flags != NULL && getenv("DS4_TP_EVENT_GATES") == NULL;
    g_tp_gpu_event = [g_device newSharedEvent];
    g_tp_cpu_event = [g_device newSharedEvent];
    g_tp_batch_gpu_event = [g_device newSharedEvent];
    g_tp_batch_cpu_event = [g_device newSharedEvent];
    if (!g_tp_gpu_event || !g_tp_cpu_event ||
        !g_tp_batch_gpu_event || !g_tp_batch_cpu_event) {
        fprintf(stderr, "ds4: failed to create TP shared events\n");
        return 0;
    }
    g_tp_exchange_fn = fn;
    g_tp_exchange_ud = ud;
    g_tp_seq = 0;
    g_tp_batch_seq = 0;
    g_tp_shutdown = 0;
    g_tp_failed_flag = 0;
    g_tp_queue_head = 0;
    g_tp_queue_count = 0;
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_set_qos_class_np(&attr, QOS_CLASS_USER_INTERACTIVE, 0);
    if (pthread_create(&g_tp_thread, &attr, ds4_gpu_tp_service_thread, NULL) != 0) {
        pthread_attr_destroy(&attr);
        fprintf(stderr, "ds4: failed to start TP gate service thread\n");
        return 0;
    }
    pthread_attr_destroy(&attr);
    g_tp_thread_running = 1;
    if (getenv("DS4_TP_NO_KEEPALIVE") == NULL) {
        uint32_t ka_tgs = ds4_gpu_tp_keepalive_tgs_from_env();
        g_tp_keepalive_queue = [g_device newCommandQueue];
        g_tp_keepalive_buffer = [g_device newBufferWithLength:(NSUInteger)ka_tgs * 256u * sizeof(float)
                                                      options:MTLResourceStorageModeShared];
        if (g_tp_keepalive_queue && g_tp_keepalive_buffer &&
            pthread_create(&g_tp_keepalive_thread, NULL,
                           ds4_gpu_tp_keepalive_thread, NULL) == 0) {
            g_tp_keepalive_running = 1;
        } else {
            fprintf(stderr, "ds4: TP keep-alive setup failed (continuing without)\n");
        }
    }
    return 1;
}

void ds4_gpu_tp_shutdown(void) {
    if (!g_tp_thread_running) return;
    pthread_mutex_lock(&g_tp_mutex);
    g_tp_shutdown = 1;
    pthread_cond_broadcast(&g_tp_cond);
    pthread_mutex_unlock(&g_tp_mutex);
    pthread_join(g_tp_thread, NULL);
    g_tp_thread_running = 0;
    if (g_tp_keepalive_running) {
        pthread_join(g_tp_keepalive_thread, NULL);
        g_tp_keepalive_running = 0;
        g_tp_keepalive_queue = nil;
        g_tp_keepalive_buffer = nil;
    }
    g_tp_exchange_fn = NULL;
    g_tp_batch_exchange_fn = NULL;
    g_tp_exchange_ud = NULL;
    g_tp_split_rank = 0;
    g_tp_split_world = 1;
    g_tp_session_batch_mode = 0;
}

void ds4_gpu_tp_suspend_expert_sharding(int suspend) {
    if (!g_tp_thread_running) return;
    g_tp_split_world = suspend ? 1 : 2;
}

int ds4_gpu_tp_gate_encode(uint32_t layer, uint32_t gate) {
    if (!g_batch_cb) {
        fprintf(stderr, "ds4: TP gate encode without an open command batch (layer %u gate %u)\n",
                layer, gate);
        return 0;
    }
    if (!g_tp_thread_running) {
        fprintf(stderr, "ds4: TP gate encode without the gate service (layer %u)\n", layer);
        return 0;
    }
    const uint64_t seq = ++g_tp_seq;
    const bool event_arrival = g_tp_session_batch_mode || !g_tp_flag_gates;
    if (!event_arrival) {
        /* Publish arrival through the slab word; the buffer hazard against
         * the partial-output kernels orders the store after the payload. */
        const uint32_t slot = layer * 2u + gate;
        const uint32_t value = (uint32_t)seq;
        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb || owned) return 0;
        id<MTLComputePipelineState> pipeline =
            ds4_gpu_get_pipeline("kernel_dsv4_tp_flag_set");
        if (!pipeline) return 0;
        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBuffer:g_tp_slab_buffer
                offset:(NSUInteger)(g_tp_slab_buffer_off + g_tp_gpu_flags_off + (uint64_t)slot * 4u)
               atIndex:0];
        [enc setBytes:&value length:sizeof(value) atIndex:1];
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);
        ds4_gpu_close_batch_encoder();
    } else {
        ds4_gpu_close_batch_encoder();
        [g_batch_cb encodeSignalEvent:g_tp_gpu_event value:seq];
    }
    [g_batch_cb encodeWaitForEvent:g_tp_cpu_event value:seq];
    pthread_mutex_lock(&g_tp_mutex);
    if (g_tp_queue_count >= DS4_GPU_TP_QUEUE) {
        pthread_mutex_unlock(&g_tp_mutex);
        fprintf(stderr, "ds4: TP gate queue overflow\n");
        return 0;
    }
    uint32_t tail = (g_tp_queue_head + g_tp_queue_count) % DS4_GPU_TP_QUEUE;
    g_tp_queue[tail].layer = layer;
    g_tp_queue[tail].gate = gate;
    g_tp_queue[tail].rows = 0;
    g_tp_queue[tail].event_arrival = event_arrival ? 1u : 0u;
    g_tp_queue[tail].seq = seq;
    g_tp_queue[tail].big_out = NULL;
    g_tp_queue[tail].big_in = NULL;
    g_tp_queue[tail].big_bytes = 0;
    g_tp_queue_count++;
    pthread_cond_signal(&g_tp_cond);
    pthread_mutex_unlock(&g_tp_mutex);
    return 1;
}

void ds4_gpu_tp_set_batch_exchange(ds4_gpu_tp_batch_exchange_fn fn) {
    g_tp_batch_exchange_fn = fn;
}

/* Verify-block batch gate: same arrival/release machinery as the row gate
 * (the FFN flag word and event pair are reused — a decode gate and a batch
 * gate are never in flight together, and seq values stay globally unique),
 * but the service thread runs the multi-row exchange callback. */
int ds4_gpu_tp_batch_gate_encode(uint32_t layer, uint32_t rows) {
    if (!g_batch_cb) return 0;
    if (!g_tp_thread_running || rows == 0) return 0;
    const uint64_t seq = ++g_tp_batch_seq;
    const bool event_arrival = g_tp_session_batch_mode || !g_tp_flag_gates;
    if (!event_arrival) {
        const uint32_t slot = layer * 2u + 1u; /* FFN gate slot */
        const uint32_t value = DS4_TP_BATCH_FLAG_TAG | (uint32_t)seq;
        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb || owned) return 0;
        id<MTLComputePipelineState> pipeline =
            ds4_gpu_get_pipeline("kernel_dsv4_tp_flag_set");
        if (!pipeline) return 0;
        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBuffer:g_tp_slab_buffer
                offset:(NSUInteger)(g_tp_slab_buffer_off + g_tp_gpu_flags_off + (uint64_t)slot * 4u)
               atIndex:0];
        [enc setBytes:&value length:sizeof(value) atIndex:1];
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);
        ds4_gpu_close_batch_encoder();
    } else {
        ds4_gpu_close_batch_encoder();
        [g_batch_cb encodeSignalEvent:g_tp_batch_gpu_event value:seq];
    }
    [g_batch_cb encodeWaitForEvent:g_tp_batch_cpu_event value:seq];
    pthread_mutex_lock(&g_tp_mutex);
    if (g_tp_queue_count >= DS4_GPU_TP_QUEUE) {
        pthread_mutex_unlock(&g_tp_mutex);
        fprintf(stderr, "ds4: TP gate queue overflow\n");
        return 0;
    }
    uint32_t tail = (g_tp_queue_head + g_tp_queue_count) % DS4_GPU_TP_QUEUE;
    g_tp_queue[tail].layer = layer;
    g_tp_queue[tail].gate = 1u; /* FFN */
    g_tp_queue[tail].rows = rows;
    g_tp_queue[tail].event_arrival = event_arrival ? 1u : 0u;
    g_tp_queue[tail].seq = seq;
    g_tp_queue[tail].big_out = NULL;
    g_tp_queue[tail].big_in = NULL;
    g_tp_queue[tail].big_bytes = 0;
    g_tp_queue_count++;
    pthread_cond_signal(&g_tp_cond);
    pthread_mutex_unlock(&g_tp_mutex);
    return 1;
}

/* Prefill batch gate kick: same seq space and release event as the verify
 * batch gate, but the service thread exchanges big_bytes directly between
 * the two shared bounce buffers instead of slab slots.  The kick only
 * publishes the GPU arrival marker and queues the exchange; the caller
 * encodes the release wait later through ds4_gpu_tp_big_gate_wait, which
 * lets it interleave more GPU work with the wire exchange.  Arrival always
 * uses the batch shared event, NOT the flag word: a flag write carries no
 * memory-visibility guarantee for the payload buffer, and once the GPU
 * keeps running past the kick (no event wait right behind it) the service
 * thread can observe the flag before the producing kernels' stores reach
 * CPU-visible memory (measured: stale rows in the first sub-kick).  The
 * shared-event signal only fires after every preceding command completes,
 * which is exactly the payload ordering the exchange needs; the ~10 us
 * slower arrival detection is noise against a multi-ms exchange. */
uint64_t ds4_gpu_tp_big_gate_kick(uint32_t layer, uint32_t rows,
                                  const ds4_gpu_tensor *out_t,
                                  ds4_gpu_tensor *in_t,
                                  uint64_t bytes) {
    if (!g_batch_cb) return 0;
    if (!g_tp_thread_running || rows == 0 || bytes == 0) return 0;
    const void *out_ptr = ds4_gpu_tensor_contents((ds4_gpu_tensor *)out_t);
    void *in_ptr = ds4_gpu_tensor_contents(in_t);
    if (!out_ptr || !in_ptr) {
        fprintf(stderr, "ds4: TP big gate needs CPU-visible bounce buffers\n");
        return 0;
    }
    const uint64_t seq = ++g_tp_batch_seq;
    ds4_gpu_close_batch_encoder();
    [g_batch_cb encodeSignalEvent:g_tp_batch_gpu_event value:seq];
    pthread_mutex_lock(&g_tp_mutex);
    if (g_tp_queue_count >= DS4_GPU_TP_QUEUE) {
        pthread_mutex_unlock(&g_tp_mutex);
        fprintf(stderr, "ds4: TP gate queue overflow\n");
        return 0;
    }
    uint32_t tail = (g_tp_queue_head + g_tp_queue_count) % DS4_GPU_TP_QUEUE;
    g_tp_queue[tail].layer = layer;
    g_tp_queue[tail].gate = 1u;
    g_tp_queue[tail].rows = rows;
    g_tp_queue[tail].event_arrival = 1u;
    g_tp_queue[tail].seq = seq;
    g_tp_queue[tail].big_out = out_ptr;
    g_tp_queue[tail].big_in = in_ptr;
    g_tp_queue[tail].big_bytes = bytes;
    g_tp_queue_count++;
    pthread_cond_signal(&g_tp_cond);
    pthread_mutex_unlock(&g_tp_mutex);
    return seq;
}

/* Encode the GPU-side release wait for a previously kicked big gate.  The
 * batch release event is monotonic and the service thread completes queued
 * exchanges in kick order, so waiting on the LAST kicked seq of a stage
 * also covers every earlier kick. */
int ds4_gpu_tp_big_gate_wait(uint64_t seq) {
    if (!g_batch_cb || seq == 0) return 0;
    ds4_gpu_close_batch_encoder();
    [g_batch_cb encodeWaitForEvent:g_tp_batch_cpu_event value:seq];
    return 1;
}

int ds4_gpu_tp_big_gate_encode(uint32_t layer, uint32_t rows,
                               const ds4_gpu_tensor *out_t,
                               ds4_gpu_tensor *in_t,
                               uint64_t bytes) {
    const uint64_t seq = ds4_gpu_tp_big_gate_kick(layer, rows, out_t, in_t, bytes);
    if (seq == 0) return 0;
    return ds4_gpu_tp_big_gate_wait(seq);
}

int ds4_gpu_tp_failed(void) {
    return g_tp_failed_flag;
}

int ds4_gpu_wait_selected_readback_ready(uint64_t event_value, const char *label) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (event_value == 0) return 0;

    if (@available(macOS 12.0, *)) {
        if (!g_selected_readback_event) return 0;

        const char *what = label ? label : "selected-id readback";
        const BOOL signaled =
            [g_selected_readback_event waitUntilSignaledValue:event_value timeoutMS:60000];
        if (!signaled) {
            fprintf(stderr, "ds4: timeout waiting for Metal shared event in %s\n", what);
            return 0;
        }
        return 1;
    }

    fprintf(stderr, "ds4: selected-id overlap requires MTLSharedEvent support\n");
    return 0;
}

static int ds4_gpu_signal_batch_and_wait_event(const char *label) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!g_batch_cb) return 0;

    if (@available(macOS 12.0, *)) {
        if (!g_selected_readback_event) {
            g_selected_readback_event = [g_device newSharedEvent];
            if (!g_selected_readback_event) {
                fprintf(stderr, "ds4: failed to create Metal shared event for %s\n",
                        label ? label : "selected-id readback");
                return 0;
            }
        }

        ds4_gpu_close_batch_encoder();
        id<MTLCommandBuffer> cb = g_batch_cb;
        g_batch_cb = nil;
        const uint64_t value = ++g_selected_readback_event_value;
        [cb encodeSignalEvent:g_selected_readback_event value:value];
        g_batch_has_work = YES;
        [cb commit];
        ds4_gpu_stream_expert_cache_note_batch_committed();

        const BOOL signaled = [g_selected_readback_event waitUntilSignaledValue:value timeoutMS:60000];
        [g_pending_cbs addObject:cb];
        if (!signaled) {
            fprintf(stderr, "ds4: timeout waiting for Metal shared event in %s\n",
                    label ? label : "selected-id readback");
            (void)ds4_gpu_wait_pending_command_buffers(label ? label : "selected-id readback");
            [g_transient_buffers removeAllObjects];
            return 0;
        }
        if (cb.status == MTLCommandBufferStatusError) {
            fprintf(stderr, "ds4: Metal %s failed: %s\n",
                    label ? label : "selected-id readback",
                    [[cb.error localizedDescription] UTF8String]);
            (void)ds4_gpu_wait_pending_command_buffers(label ? label : "selected-id readback");
            [g_transient_buffers removeAllObjects];
            return 0;
        }

        g_batch_cb = ds4_gpu_new_command_buffer();
        g_batch_has_work = NO;
        if (g_batch_cb) ds4_gpu_stream_expert_cache_note_batch_created();
        if (!g_batch_cb) {
            (void)ds4_gpu_wait_pending_command_buffers(label ? label : "selected-id readback");
            [g_transient_buffers removeAllObjects];
            return 0;
        }
        return 1;
    } else {
        ds4_gpu_close_batch_encoder();
        id<MTLCommandBuffer> cb = g_batch_cb;
        g_batch_cb = nil;
        g_batch_has_work = NO;
        if (ds4_gpu_finish_command_buffer(cb, 1, label ? label : "selected-id readback") == 0) {
            return 0;
        }
        return ds4_gpu_begin_commands();
    }
}

int ds4_gpu_end_commands(void) {
    if (!g_batch_cb) return 0;
    ds4_gpu_close_batch_encoder();
    id<MTLCommandBuffer> cb = g_batch_cb;
    g_batch_cb = nil;
    g_batch_has_work = NO;
    g_stream_expert_cache_owned_seq = g_stream_expert_cache_batch_seq;
    g_stream_expert_cache_batch_seq = 0;
    return ds4_gpu_finish_command_buffer(cb, 1, "command batch");
}

static int ds4_gpu_flash_attn_stage_profile_boundary(
        id<MTLCommandBuffer> __strong *cbp,
        const char           *mode,
        const char           *stage,
        uint32_t              n_tokens,
        uint32_t              n_comp,
        uint32_t              n_keys,
        uint32_t              n_head,
        uint32_t              head_dim,
        uint32_t              window,
        uint32_t              ratio,
        double               *stage_t0) {
    if (!cbp || !*cbp || !stage_t0 || !stage) return 0;
    if (ds4_gpu_end_commands() == 0) return 0;

    const double now_ms = ds4_gpu_now_ms();
    const char *filter = getenv("DS4_METAL_FLASH_ATTN_STAGE_PROFILE_FILTER");
    const int print_stage =
        !filter || !filter[0] ||
        strstr(stage, filter) != NULL ||
        (mode && strstr(mode, filter) != NULL);
    if (print_stage) {
        fprintf(stderr,
                "ds4: Metal FlashAttention prefill stage mode=%s tokens=%u comp=%u "
                "keys=%u heads=%u dim=%u window=%u ratio=%u %s=%.3f ms\n",
                mode ? mode : "unknown",
                n_tokens,
                n_comp,
                n_keys,
                n_head,
                head_dim,
                window,
                ratio,
                stage,
                now_ms - *stage_t0);
    }
    *stage_t0 = now_ms;

    if (ds4_gpu_begin_commands() == 0) return 0;
    int owned = 0;
    *cbp = ds4_gpu_command_buffer(&owned);
    return *cbp != nil && owned == 0;
}

int ds4_gpu_synchronize(void) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (g_batch_cb) return ds4_gpu_end_commands();
    if ([g_pending_cbs count] != 0) {
        int ok = ds4_gpu_wait_pending_command_buffers("synchronize");
        [g_transient_buffers removeAllObjects];
        ds4_gpu_model_buffer_cache_maybe_evict("synchronize");
        return ok;
    }

    id<MTLCommandBuffer> cb = ds4_gpu_new_command_buffer();
    if (!cb) return 0;
    return ds4_gpu_finish_command_buffer(cb, 1, "synchronize");
}

void ds4_gpu_cleanup(void) {
    if (!g_initialized) return;

    @autoreleasepool {
        if (g_batch_cb) {
            ds4_gpu_close_batch_encoder();
            [g_batch_cb commit];
            [g_batch_cb waitUntilCompleted];
            g_batch_cb = nil;
            if (g_stream_expert_cache_batch_seq > g_stream_expert_cache_done_seq) {
                g_stream_expert_cache_done_seq = g_stream_expert_cache_batch_seq;
            }
            g_stream_expert_cache_batch_seq = 0;
        }
        (void)ds4_gpu_wait_pending_command_buffers("cleanup");
        if (ds4_gpu_stream_expert_timing_summary_enabled() &&
            getenv("DS4_METAL_MEMORY_REPORT") == NULL) {
            ds4_gpu_print_memory_report("at cleanup");
        }
        g_selected_readback_event = nil;
        g_selected_readback_event_value = 0;
        [g_transient_buffers removeAllObjects];
        ds4_gpu_stream_expert_pread_pool_shutdown();
        ds4_gpu_stream_expert_cache_clear_all(1);
        for (uint32_t layer = 0; layer < DS4_METAL_STREAM_EXPERT_CACHE_MAX_LAYER; layer++) {
            g_stream_expert_cache_gate_addr_buffers[layer] = nil;
            g_stream_expert_cache_up_addr_buffers[layer] = nil;
            g_stream_expert_cache_down_addr_buffers[layer] = nil;
            g_stream_compact_gate_addr_buffers[layer] = nil;
            g_stream_compact_up_addr_buffers[layer] = nil;
            g_stream_compact_down_addr_buffers[layer] = nil;
            g_stream_compact_selected_buffers[layer] = nil;
            g_stream_selected_id_buffers[layer] = nil;
        }
        g_set_rows_f32_i32_pipeline = nil;
        g_get_rows_f32_pipeline = nil;
        g_get_rows_f16_pipeline = nil;
        g_get_rows_i32_pipeline = nil;
        g_get_rows_q8_0_pipeline = nil;
        g_get_rows_q4_0_pipeline = nil;
        g_get_rows_q4_K_pipeline = nil;
        g_repeat_f32_pipeline = nil;
        g_concat_pipeline = nil;
        g_cpy_f32_f32_pipeline = nil;
        g_cpy_f32_f16_pipeline = nil;
        g_cpy_contig_f32_f16_pipeline = nil;
        g_cpy_f16_f32_pipeline = nil;
        g_cpy_f16_f16_pipeline = nil;
        g_cpy_contig_f16_f32_pipeline = nil;
        g_cpy_contig_f16_f16_pipeline = nil;
        g_flash_kv_stage_f16_pipeline = nil;
        g_swiglu_pipeline = nil;
        g_swiglu_flat_pipeline = nil;
        g_add_pipeline = nil;
        g_add2_pipeline = nil;
        g_add3_pipeline = nil;
        g_moe_sum6_pipeline = nil;
        g_moe_sum8_pipeline = nil;
        g_mul_pipeline = nil;
        g_bin_mul_scalar_pipeline = nil;
        g_bin_div_row_pipeline = nil;
        g_unary_sigmoid_pipeline = nil;
        g_unary_silu_pipeline = nil;
        g_unary_softplus_pipeline = nil;
        g_unary_sqrt_pipeline = nil;
        g_unary_clamp_pipeline = nil;
        g_unary_scale_pipeline = nil;
        g_unary_fill_pipeline = nil;
        g_unary_fill_f16_pipeline = nil;
        g_rms_norm_pipeline = nil;
        g_rms_norm_plain_pipeline = nil;
        g_add_rms_norm_pipeline = nil;
        g_rms_norm_scale_pipeline = nil;
        g_dsv4_qkv_rms_norm_pipeline = nil;
        g_hc_split_sinkhorn_pipeline = nil;
        g_hc_split_weighted_sum_pipeline = nil;
        g_hc_split_weighted_sum_norm_pipeline = nil;
        g_hc_weighted_sum_pipeline = nil;
        g_hc_weighted_sum_norm_pipeline = nil;
        g_output_hc_weights4_pipeline = nil;
        g_hc_expand_pipeline = nil;
        g_moe_mul_mv_id_iq2_xxs_pipeline = nil;
        g_moe_mul_mv_id_iq2_xxs_pair_pipeline = nil;
        g_moe_mul_mv_id_iq2_xxs_pair_swiglu_pipeline = nil;
        g_moe_mul_mv_id_q2_k_pipeline = nil;
        g_moe_mul_mv_id_q2_k_sum6_pipeline = nil;
        g_moe_mul_mv_id_iq2_xxs_sum6_pipeline = nil;
        g_moe_mul_mv_id_q4_k_pipeline = nil;
        g_moe_mul_mv_id_q4_k_pair_pipeline = nil;
        g_moe_mul_mv_id_q4_k_pair_swiglu_pipeline = nil;
        g_moe_mul_mv_id_q4_k_sum6_pipeline = nil;
        g_moe_mul_mv_group_q4_k_pair_swiglu_pipeline = nil;
        g_moe_mul_mv_group_q4_k_sum6_pipeline = nil;
        g_moe_mul_mv_group6_q4_k_pair_swiglu_pipeline = nil;
        g_moe_mul_mv_group6_q4_k_sum6_pipeline = nil;
        g_moe_mul_mv_group8_q4_k_pair_swiglu_pipeline = nil;
        g_moe_mul_mv_group8_q4_k_sum6_pipeline = nil;
        g_moe_mul_mv_group24_q4_k_id_pipeline = nil;
        g_moe_mul_mv_group24_q4_k_sum6_pipeline = nil;
        g_moe_mul_mv_slots6_iq2_xxs_pair_swiglu_pipeline = nil;
        g_moe_mul_mv_slots6_q2_k_sum6_pipeline = nil;
        g_moe_mul_mv_slots6_q4_k_pair_swiglu_pipeline = nil;
        g_moe_mul_mv_slots6_q4_k_sum6_pipeline = nil;
        g_moe_mul_mv_addr_iq2_xxs_pair_swiglu_pipeline = nil;
        g_moe_mul_mv_addr_iq2_xxs_pipeline = nil;
        g_moe_mul_mv_addr_q2_k_sum6_pipeline = nil;
        g_moe_mul_mv_addr_iq2_xxs_pair_swiglu_masked_pipeline = nil;
        g_moe_mul_mv_addr_q2_k_sum6_masked_pipeline = nil;
        g_moe_stream_expert_cache_validate_pipeline = nil;
        g_moe_q4_gather_slots6_pipeline = nil;
        g_moe_mul_mv_table_q4_k_pair_swiglu_pipeline = nil;
        g_moe_mul_mv_table_q4_k_sum6_pipeline = nil;
        g_moe_mul_mv_addr_q4_k_pair_swiglu_pipeline = nil;
        g_moe_mul_mv_addr_q4_k_sum6_pipeline = nil;
        g_moe_table_q4_pair_gate_encoder = nil;
        g_moe_table_q4_pair_up_encoder = nil;
        g_moe_table_q4_sum_down_encoder = nil;
        g_rope_tail_batch_pipeline = nil;
        g_rope_tail_inplace_pair_pipeline = nil;
        g_rope_tail_inplace_pair_shared4_pipeline = nil;
        g_rope_tail_inplace_pair_affine_pipeline = nil;
        g_dsv4_fp8_kv_quantize_pipeline = nil;
        g_dsv4_indexer_qat_pipeline = nil;
        g_dsv4_kv_fp8_store_pipeline = nil;
        g_dsv4_ratio4_shift_pipeline = nil;
        g_dsv4_compressor_pack_ratio4_pipeline = nil;
        g_dsv4_softmax_pool_ratio4_direct_pipeline = nil;
        g_dsv4_softmax_pool_pipeline = nil;
        g_soft_max_f32_pipeline = nil;
        g_soft_max_f32_4_pipeline = nil;
        g_argsort_f32_i32_desc_pipeline = nil;
        g_argsort_merge_f32_i32_desc_pipeline = nil;
        g_sum_rows_f32_f32_pipeline = nil;
        g_dsv4_topk_mask_pipeline = nil;
        g_dsv4_topk_mask_scatter_pipeline = nil;
        g_dsv4_indexer_weighted_sum_pipeline = nil;
        g_dsv4_indexer_score_one_direct_pipeline = nil;
        g_dsv4_compressor_store_one_pipeline = nil;
        g_dsv4_sort_i32_rows_asc_pipeline = nil;
        g_dsv4_indexed_attention_heads8_pipeline = nil;
        g_dsv4_indexed_attention_heads8_rb16_pipeline = nil;
        g_dsv4_softplus_sqrt_pipeline = nil;
        g_dsv4_router_finalize_one_pipeline = nil;
        g_dsv4_router_finalize_one_simd_pipeline = nil;
        g_dsv4_router_finalize_weights_one_simd_pipeline = nil;
        g_dsv4_router_weights_one_pipeline = nil;
        g_glm_router_select_one_pipeline = nil;
        g_glm_kv_lora_rms_norm_pipeline = nil;
        g_glm_k_b_project_pipeline = nil;
        g_glm_store_compact_kv_pipeline = nil;
        g_glm_qkv_norm_store_compact_kv_pipeline = nil;
        g_glm_store_indexer_k_pipeline = nil;
        g_glm_build_kv_cache_pipeline = nil;
        g_glm_build_kv_cache_decode_group4_pipeline = nil;
        g_glm_build_kv_cache_flash_pipeline = nil;
        g_glm_attention_full_pipeline = nil;
        g_glm_fill_selected_range_pipeline = nil;
        g_glm_fill_selected_range_batch_pipeline = nil;
        g_glm_indexer_rope_tail_pipeline = nil;
        g_glm_indexer_score_one_pipeline = nil;
        g_glm_indexer_score_one_direct_pipeline = nil;
        g_glm_indexer_scores_batch_pipeline = nil;
        g_glm_indexer_scores_tiled_pipeline = nil;
        g_glm_indexer_scores_tiled_f32_pipeline = nil;
        g_glm_qk_lowrank_pipeline = nil;
        g_glm_qk_lowrank_glm52_pipeline = nil;
        g_glm_qk_lowrank_glm52_sg_pipeline = nil;
        g_glm_qk_lowrank_batch_pipeline = nil;
        g_glm_qk_lowrank_batch_glm52_t4_pipeline = nil;
        g_glm_value_project_q8_0_pipeline = nil;
        g_glm_value_project_q8_0_batch_heads_pipeline = nil;
        g_glm_value_project_q8_0_batch_heads_mma_pipeline = nil;
        g_glm_attention_indexed_decode_pipeline = nil;
        g_glm_attention_indexed_decode_split_group8_partial_pipeline = nil;
        g_glm_attention_indexed_decode_split_group8_partial_valid_fullheads_pipeline = nil;
        g_glm_attention_indexed_decode_split_group8_reduce_pipeline = nil;
        g_glm_attention_indexed_decode_split_group8_reduce16_pipeline = nil;
        g_glm_attention_indexed_batch_pipeline = nil;
        g_glm_attention_indexed_batch_group2_pipeline = nil;
        g_glm_attention_indexed_batch_q2_group4_pipeline = nil;
        g_glm_attention_indexed_batch_group8_pipeline = nil;
        g_glm_attention_indexed_batch_lora_group8_vec_pipeline = nil;
        g_glm_attention_indexed_batch_lora_group8_vec_valid_pipeline = nil;
        g_glm_attention_indexed_batch_lora_group8_vec_valid_fullheads_pipeline = nil;
        g_glm_attention_indexed_batch_lora_group8_vec_causal_pipeline = nil;
        g_glm_attention_indexed_batch_lora_group8_vec_causal_fullheads_pipeline = nil;
        g_glm_q4_k_pair_swiglu_f32_pipeline = nil;
        g_glm_q4_k_pair_swiglu2_f32_pipeline = nil;
        g_glm_q4_k_pair_swiglu4_f32_pipeline = nil;
        g_glm_q4_k_pair_swiglu2_mapped_f32_pipeline = nil;
        g_glm_q4_k_pair_swiglu2_mapped_row_f32_pipeline = nil;
        g_glm_q2_k_pair_swiglu_f32_pipeline = nil;
        g_glm_q2_k_addr_pair_swiglu2_f32_pipeline = nil;
        g_glm_q2_k_addr_pair_swiglu2_masked_f32_pipeline = nil;
        g_glm_q4_k_addr_pair_swiglu_f32_pipeline = nil;
        g_glm_q4_k_addr_pair_swiglu_masked_f32_pipeline = nil;
        g_glm_q2_k_down_f32_pipeline = nil;
        g_glm_q4_k_down_f32_pipeline = nil;
        g_glm_q2_k_addr_down_f32_pipeline = nil;
        g_glm_q4_k_addr_down_f32_pipeline = nil;
        g_glm_q5_k_pair_swiglu_f32_pipeline = nil;
        g_glm_q5_k_pair_swiglu_mapped_f32_pipeline = nil;
        g_glm_q5_k_pair_swiglu_mapped_row_f32_pipeline = nil;
        g_glm_q5_k_down_f32_pipeline = nil;
        g_glm_q6_k_down_f32_pipeline = nil;
        g_dsv4_router_weights_batch_pipeline = nil;
        g_dsv4_hc_expand4_pipeline = nil;
        g_flash_attn_mask_buffer = nil;
        g_flash_attn_zero_mask_buffer = nil;
        g_flash_attn_pad_buffer = nil;
        g_flash_attn_tmp_buffer = nil;
        g_flash_attn_blk_buffer = nil;
        ds4_gpu_clear_zero_prefix_prefill_mask_cache();
        g_flash_attn_ring_buffer = nil;
        g_flash_attn_kv_buffer = nil;
        g_glm_flash_attn_mask_buffer = nil;
        g_compressor_pool_kv_buffer = nil;
        g_compressor_pool_score_buffer = nil;
        g_compressor_pool_score_cont_buffer = nil;
        g_compressor_pool_softmax_buffer = nil;
        g_compressor_pool_product_buffer = nil;
        g_compressor_store_ape_buffer = nil;
        g_compressor_store_score_buffer = nil;
        g_embed_rows_buffer = nil;
        g_router_selection_buffer = nil;
        g_router_weight_sum_buffer = nil;
        g_indexer_head_scores_buffer = nil;
        g_indexer_topk_buffer = nil;
        g_indexed_topk_buffer = nil;
        g_stream_expert_validate_status_buffer = nil;
        g_f16_round_scratch_buffer = nil;
        g_raw_store_round_buffer = nil;
        g_moe_gate_scratch_buffer = nil;
        g_moe_down_scratch_buffer = nil;
        g_moe_id_map_buffer = nil;
        g_moe_q4_gate_slots_buffer = nil;
        g_moe_q4_up_slots_buffer = nil;
        g_moe_q4_down_slots_buffer = nil;
        g_attn_out_group_ids_buffer = nil;
        g_model_fd = -1;
        g_model_map_ptr = NULL;
        g_model_map_size = 0;
        g_model_mapped_offset = 0;
        g_model_mapped_size = 0;
        g_model_mapped_max_tensor_bytes = 0;
        ds4_gpu_tensor_tracking_reset();
        g_flash_attn_mask_bytes = 0;
        g_flash_attn_zero_mask_bytes = 0;
        g_flash_attn_pad_bytes = 0;
        g_flash_attn_tmp_bytes = 0;
        g_flash_attn_blk_bytes = 0;
        g_flash_attn_ring_bytes = 0;
        g_flash_attn_kv_bytes = 0;
        g_glm_flash_attn_mask_bytes = 0;
        g_glm_flash_attn_mask_valid = 0;
        g_glm_flash_attn_mask_pos0 = 0;
        g_glm_flash_attn_mask_tokens = 0;
        g_glm_flash_attn_mask_cache_len = 0;
        g_compressor_pool_kv_bytes = 0;
        g_compressor_pool_score_bytes = 0;
        g_compressor_pool_score_cont_bytes = 0;
        g_compressor_pool_softmax_bytes = 0;
        g_compressor_pool_product_bytes = 0;
        g_compressor_store_ape_bytes = 0;
        g_compressor_store_score_bytes = 0;
        g_embed_rows_bytes = 0;
        g_router_selection_bytes = 0;
        g_router_weight_sum_bytes = 0;
        g_indexer_head_scores_bytes = 0;
        g_indexer_topk_bytes = 0;
        g_indexed_topk_bytes = 0;
        g_f16_round_scratch_bytes = 0;
        g_raw_store_round_bytes = 0;
        g_moe_gate_scratch_bytes = 0;
        g_moe_down_scratch_bytes = 0;
        g_moe_id_map_bytes = 0;
        g_moe_q4_gate_slots_bytes = 0;
        g_moe_q4_up_slots_bytes = 0;
        g_moe_q4_down_slots_bytes = 0;
        g_attn_out_group_ids_bytes = 0;
        g_model_wrap_count = 0;
        g_model_wrap_bytes = 0;
        g_model_wrap_max_bytes = 0;
        g_model_buffer_cache_bytes = 0;
        g_model_buffer_cache_evictions = 0;
        g_model_buffer_cache_over_limit = 0;
        ds4_gpu_model_residency_clear();
        ds4_gpu_model_views_clear();
        [g_pipeline_cache removeAllObjects];
        g_pipeline_cache = nil;
        [g_q4_expert_layer_residency_cache removeAllObjects];
        g_q4_expert_layer_residency_cache = nil;
        [g_q4_expert_table_cache removeAllObjects];
        g_q4_expert_table_cache = nil;
        [g_model_buffer_cache removeAllObjects];
        g_model_buffer_cache = nil;
        g_transient_buffers = nil;
        g_pending_cbs = nil;
        g_library = nil;
        g_queue = nil;
        g_device = nil;
        g_initialized = 0;
    }
}
