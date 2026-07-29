struct ds4_metal_args_dsv4_topk_mask {
    int64_t  ne00;
    int64_t  ne01;
    uint64_t nb00;
    uint64_t nb01;
    int64_t  ne0;
    int64_t  ne1;
    uint64_t nb0;
    uint64_t nb1;
};

struct ds4_metal_args_dsv4_indexer_weighted_sum {
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
};

struct ds4_metal_args_dsv4_softmax_pool {
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
};

struct ds4_metal_args_dsv4_softmax_pool_ratio4_direct {
    int64_t  n_rows;
    uint32_t head_dim;
    uint32_t n_comp;
    uint32_t replay;
    uint32_t pad;
};

struct ds4_metal_args_dsv4_compressor_score_ape {
    uint32_t width;
    uint32_t ratio;
    uint32_t pos0;
    uint32_t n_tokens;
};

struct ds4_metal_args_dsv4_indexed_attention {
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
};

struct ds4_metal_args_dsv4_indexer_scores_fused {
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
};

struct ds4_metal_args_dsv4_router_select_one {
    uint32_t has_bias;
    uint32_t hash_mode;
    uint32_t use_token_buffer;
    uint32_t token;
    uint32_t hash_rows;
};

struct ds4_metal_args_glm_router_select_one {
    uint32_t n_expert;
    uint32_t n_expert_used;
    float    expert_weight_scale;
    uint32_t pad0;
};

struct ds4_metal_args_glm_kv_lora_rms_norm {
    uint32_t n_tokens;
    uint32_t kv_raw_dim;
    uint32_t kv_lora_dim;
    float    eps;
};

struct ds4_metal_args_glm_k_b_project {
    uint32_t n_tokens;
    uint32_t kv_lora_dim;
    uint32_t qk_nope;
    uint32_t n_head;
    uint32_t row_bytes;
    uint32_t weight_type;
    uint32_t pad1;
    uint32_t pad2;
};

struct ds4_metal_args_glm_build_kv_cache {
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
};

struct ds4_metal_args_glm_store_compact_kv {
    uint32_t pos0;
    uint32_t n_tokens;
    uint32_t cache_cap;
    uint32_t kv_raw_dim;
    uint32_t kv_lora_dim;
    uint32_t qk_rope;
    uint32_t cache_f16;
    uint32_t pad1;
};

struct ds4_metal_args_glm_qkv_norm_store_compact_kv {
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
};

struct ds4_metal_args_glm_store_indexer_k {
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
};

struct ds4_metal_args_glm_attention_full {
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
};

struct ds4_metal_args_glm_fill_selected_range {
    uint32_t n_selected;
};

struct ds4_metal_args_glm_fill_selected_range_batch {
    uint32_t n_tokens;
    uint32_t pos0;
    uint32_t n_selected;
    uint32_t pad_row;
};

struct ds4_metal_args_glm_indexer_rope_tail {
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
};

struct ds4_metal_args_glm_indexer_score_one {
    uint32_t n_rows;
    uint32_t n_head;
    uint32_t head_dim;
    uint32_t cache_f16;
    float    scale;
};

struct ds4_metal_args_glm_indexer_scores_batch {
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
};

struct ds4_metal_args_glm_qk_lowrank {
    uint32_t n_head;
    uint32_t kv_lora_dim;
    uint32_t qk_nope;
    uint32_t qk_dim;
    uint32_t row_bytes;
    uint32_t weight_type;
    uint32_t pad1;
    uint32_t pad2;
};

struct ds4_metal_args_glm_qk_lowrank_batch {
    uint32_t n_tokens;
    uint32_t n_head;
    uint32_t kv_lora_dim;
    uint32_t qk_nope;
    uint32_t qk_dim;
    uint32_t row_bytes;
    uint32_t weight_type;
    /* First head this dispatch computes: under tensor-parallel head split
     * each rank covers a contiguous half of the heads; buffers and weights
     * keep full-model layout and are indexed by absolute head. */
    uint32_t head_base;
};

struct ds4_metal_args_glm_attention_indexed_decode {
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
};

struct ds4_metal_args_glm_attention_indexed_decode_split {
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
};

struct ds4_metal_args_glm_attention_indexed_batch {
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
};

struct ds4_metal_args_dsv4_directional_steering_project {
    uint32_t width;
    uint32_t rows;
    uint32_t layer;
    uint32_t n_threads;
    float    scale;
};

// Optional directional steering projection.
//
// Each threadgroup owns one 4096-wide token row, computes
// dot(row, direction[layer]), then subtracts scale * direction * dot in-place.
// Positive scales remove a concept direction; negative scales amplify it.  The
// kernel is not used unless a steering file and nonzero scale are provided.
