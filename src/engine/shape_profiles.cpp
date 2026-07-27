#include "pulsar_engine_internal.h"



const pulsar_shape PULSAR_SHAPE_FLASH = {
    .name = "DeepSeek V4 Flash",
    .variant = PULSAR_VARIANT_FLASH,
    .n_layer = 43,
    .n_embd = 4096,
    .n_vocab = 129280,
    .n_head = 64,
    .n_head_kv = 1,
    .n_head_dim = 512,
    .n_value_dim = 512,
    .n_rot = 64,
    .n_out_group = 8,
    .n_lora_q = 1024,
    .n_lora_o = 1024,
    .n_expert = 256,
    .n_expert_used = 6,
    .n_expert_shared = 1,
    .n_ff_exp = 2048,
    .n_hash_layer = 3,
    .n_swa = 128,
    .n_indexer_head = 64,
    .n_indexer_head_dim = 128,
    .n_indexer_top_k = 512,
    .n_hc = 4,
    .n_hc_sinkhorn_iter = 20,
    .rms_eps = PULSAR_DEFAULT_RMS_EPS,
    .hc_eps = PULSAR_DEFAULT_HC_EPS,
    .expert_weight_scale = 1.5f,
    .swiglu_clamp_exp = PULSAR_DEFAULT_SWIGLU_CLAMP_EXP,
    .rope_freq_base = PULSAR_DEFAULT_ROPE_FREQ_BASE,
    .rope_scale_factor = PULSAR_DEFAULT_ROPE_SCALE_FACTOR,
    .rope_yarn_beta_fast = PULSAR_DEFAULT_ROPE_YARN_BETA_FAST,
    .rope_yarn_beta_slow = PULSAR_DEFAULT_ROPE_YARN_BETA_SLOW,
    .compress_rope_freq_base = PULSAR_DEFAULT_COMPRESS_ROPE_FREQ_BASE,
    .rope_orig_ctx = PULSAR_DEFAULT_ROPE_ORIG_CTX,
};



const pulsar_shape PULSAR_SHAPE_PRO = {
    .name = "DeepSeek V4 Pro",
    .variant = PULSAR_VARIANT_PRO,
    .n_layer = 61,
    .n_embd = 7168,
    .n_vocab = 129280,
    .n_head = 128,
    .n_head_kv = 1,
    .n_head_dim = 512,
    .n_value_dim = 512,
    .n_rot = 64,
    .n_out_group = 16,
    .n_lora_q = 1536,
    .n_lora_o = 1024,
    .n_expert = 384,
    .n_expert_used = 6,
    .n_expert_shared = 1,
    .n_ff_exp = 3072,
    .n_hash_layer = 3,
    .n_swa = 128,
    .n_indexer_head = 64,
    .n_indexer_head_dim = 128,
    .n_indexer_top_k = 1024,
    .n_hc = 4,
    .n_hc_sinkhorn_iter = 20,
    .rms_eps = PULSAR_DEFAULT_RMS_EPS,
    .hc_eps = PULSAR_DEFAULT_HC_EPS,
    .expert_weight_scale = 2.5f,
    .swiglu_clamp_exp = PULSAR_DEFAULT_SWIGLU_CLAMP_EXP,
    .rope_freq_base = PULSAR_DEFAULT_ROPE_FREQ_BASE,
    .rope_scale_factor = PULSAR_DEFAULT_ROPE_SCALE_FACTOR,
    .rope_yarn_beta_fast = PULSAR_DEFAULT_ROPE_YARN_BETA_FAST,
    .rope_yarn_beta_slow = PULSAR_DEFAULT_ROPE_YARN_BETA_SLOW,
    .compress_rope_freq_base = PULSAR_DEFAULT_COMPRESS_ROPE_FREQ_BASE,
    .rope_orig_ctx = PULSAR_DEFAULT_ROPE_ORIG_CTX,
};



/* Boot default: the Flash profile; gguf metadata re-selects at load. */
pulsar_shape g_pulsar_shape = PULSAR_SHAPE_FLASH;



uint32_t g_pulsar_compress_ratios[PULSAR_MAX_LAYER] = {0};
uint32_t g_pulsar_layer_expert_count[PULSAR_MAX_LAYER] = {0};


int g_pulsar_lock_fd = -1;

