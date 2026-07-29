#ifndef K3_MODEL_H
#define K3_MODEL_H

#include "k3_safetensors.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

enum {
    K3_N_LAYER = 93,
    K3_N_EMBD = 7168,
    K3_N_VOCAB = 163840,
    K3_N_HEAD = 96,
    K3_HEAD_DIM = 128,
    K3_KDA_PROJ = K3_N_HEAD * K3_HEAD_DIM,
    K3_CONV_WIDTH = 4,
    K3_Q_LORA = 1536,
    K3_KV_LORA = 512,
    K3_QK_NOPE = 128,
    K3_QK_ROPE = 64,
    K3_QK_HEAD = K3_QK_NOPE + K3_QK_ROPE,
    K3_V_HEAD = 128,
    K3_N_EXPERT = 896,
    K3_N_EXPERT_USED = 16,
    K3_ROUTED_HIDDEN = 3584,
    K3_EXPERT_FF = 3072,
    K3_SHARED_FF = 6144,
    K3_DENSE_FF = 33792,
    K3_MAX_CONTEXT = 1048576,
    K3_ATTN_RES_BLOCK = 12,
    K3_ATTN_RES_BLOCKS =
        (K3_N_LAYER + K3_ATTN_RES_BLOCK - 1) / K3_ATTN_RES_BLOCK,
    K3_BOUNDARY_VECTORS = 2 + K3_ATTN_RES_BLOCKS,
    K3_BOUNDARY_F32_VALUES = K3_BOUNDARY_VECTORS * K3_N_EMBD,
};

typedef enum {
    K3_LAYER_KDA,
    K3_LAYER_MLA,
} k3_attention_kind;

typedef struct {
    const k3_st_tensor *w1_packed;
    const k3_st_tensor *w1_scale;
    const k3_st_tensor *w2_packed;
    const k3_st_tensor *w2_scale;
    const k3_st_tensor *w3_packed;
    const k3_st_tensor *w3_scale;
} k3_expert_weights;

typedef struct {
    uint32_t index;
    k3_attention_kind attention;
    k3_st_shard *shard;
    uint64_t static_offset;
    uint64_t static_size;
    uint64_t layer_size;

    const k3_st_tensor *input_norm;
    const k3_st_tensor *post_attention_norm;
    const k3_st_tensor *self_attention_res_norm;
    const k3_st_tensor *self_attention_res_proj;
    const k3_st_tensor *mlp_res_norm;
    const k3_st_tensor *mlp_res_proj;

    /* KDA. A_log is padded to 128 values; inference consumes the first 96. */
    const k3_st_tensor *a_log;
    const k3_st_tensor *dt_bias;
    const k3_st_tensor *q_conv;
    const k3_st_tensor *k_conv;
    const k3_st_tensor *v_conv;
    const k3_st_tensor *o_norm;
    const k3_st_tensor *q_proj;
    const k3_st_tensor *k_proj;
    const k3_st_tensor *v_proj;
    const k3_st_tensor *f_a_proj;
    const k3_st_tensor *f_b_proj;
    const k3_st_tensor *b_proj;

    /* Shared by KDA's full-rank output gate and MLA's output gate. */
    const k3_st_tensor *g_proj;
    const k3_st_tensor *o_proj;

    /* MLA. */
    const k3_st_tensor *q_a_proj;
    const k3_st_tensor *q_a_norm;
    const k3_st_tensor *q_b_proj;
    const k3_st_tensor *kv_a_proj;
    const k3_st_tensor *kv_a_norm;
    const k3_st_tensor *kv_b_proj;

    /* Layer 0 dense MLP. */
    const k3_st_tensor *dense_gate;
    const k3_st_tensor *dense_up;
    const k3_st_tensor *dense_down;

    /* Stable LatentMoE, layers 1..92. */
    const k3_st_tensor *router;
    const k3_st_tensor *router_bias;
    const k3_st_tensor *routed_down;
    const k3_st_tensor *routed_norm;
    const k3_st_tensor *routed_up;
    const k3_st_tensor *shared_gate;
    const k3_st_tensor *shared_up;
    const k3_st_tensor *shared_down;
    const k3_st_tensor *expert_gate;
    const k3_st_tensor *expert_up;
    const k3_st_tensor *expert_down;
    k3_expert_weights experts[K3_N_EXPERT];
} k3_layer;

typedef struct {
    char *directory;
    bool gguf;
} k3_model;

typedef struct {
    k3_st_shard *shard;
    const k3_st_tensor *lm_head;
    const k3_st_tensor *token_embedding;
    const k3_st_tensor *norm;
    const k3_st_tensor *output_attn_res_norm;
    const k3_st_tensor *output_attn_res_proj;
} k3_global_weights;

typedef struct k3_layer_load k3_layer_load;

int k3_model_open(k3_model **out,
                  const char *directory,
                  char *err,
                  size_t errcap);
void k3_model_close(k3_model *model);

int k3_model_open_layer(const k3_model *model,
                        uint32_t index,
                        k3_layer **out,
                        char *err,
                        size_t errcap);
void k3_layer_close(k3_layer *layer);

int k3_model_open_globals(const k3_model *model,
                          k3_global_weights *out,
                          char *err,
                          size_t errcap);
void k3_global_weights_close(k3_global_weights *weights);

/*
 * Map the next shard and issue filesystem readahead for either its static
 * prefix or full layer. The loader deliberately does not CPU-touch the advised
 * pages; Metal faults only the tensors used by the current schedule.
 */
int k3_layer_load_start(k3_layer_load **out,
                        const k3_model *model,
                        uint32_t index,
                        bool full_layer,
                        char *err,
                        size_t errcap);
/* Start the same asynchronous readahead for an already-open descriptor. */
int k3_layer_prefetch_start(k3_layer_load **out,
                            k3_layer *layer,
                            bool full_layer,
                            bool pagein,
                            char *err,
                            size_t errcap);
int k3_layer_readahead_start(k3_layer_load **out,
                             k3_layer *layer,
                             bool full_layer,
                             char *err,
                             size_t errcap);
int k3_layer_load_join(k3_layer_load *load,
                       k3_layer **out,
                       char *err,
                       size_t errcap);
void k3_layer_load_cancel(k3_layer_load *load);

bool k3_layer_is_mla(uint32_t index);

#endif
