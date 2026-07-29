#define _POSIX_C_SOURCE 200809L

#include "k3_model.h"

#include <errno.h>
#include <inttypes.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

struct k3_layer_load {
    pthread_t thread;
    const k3_model *model;
    uint32_t index;
    bool full_layer;
    bool pagein;
    bool map_layer;
    bool owns_layer;
    bool started;
    k3_layer *layer;
    int ok;
    char error[256];
};

static void set_error(char *err, size_t cap, const char *fmt, ...) {
    if (!err || cap == 0) return;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(err, cap, fmt, ap);
    va_end(ap);
}

static char *copy_string(const char *value) {
    const size_t len = strlen(value);
    char *copy = malloc(len + 1u);
    if (copy) memcpy(copy, value, len + 1u);
    return copy;
}

static int make_path(char *out,
                     size_t cap,
                     const char *directory,
                     uint32_t shard) {
    const size_t len = strlen(directory);
    const char *slash = len && directory[len - 1u] == '/' ? "" : "/";
    const int n = snprintf(out,
                           cap,
                           "%s%smodel-%05u-of-000096.safetensors",
                           directory,
                           slash,
                           shard);
    return n > 0 && (size_t)n < cap;
}

static int make_model_path(char *out,
                           size_t cap,
                           const k3_model *model,
                           uint32_t shard) {
    if (!model) return 0;
    if (model->gguf) {
        const int n = snprintf(out, cap, "%s", model->directory);
        return n > 0 && (size_t)n < cap;
    }
    return make_path(out, cap, model->directory, shard);
}

static int make_tensor_name(char *out,
                            size_t cap,
                            uint32_t layer,
                            const char *suffix) {
    const int n = snprintf(out,
                           cap,
                           "language_model.model.layers.%u.%s",
                           layer,
                           suffix);
    return n > 0 && (size_t)n < cap;
}

static const k3_st_tensor *require_tensor(
        const k3_st_shard *shard,
        uint32_t layer,
        const char *suffix,
        k3_st_dtype dtype,
        uint32_t n_dims,
        const uint64_t *shape,
        char *err,
        size_t errcap) {
    char name[256];
    if (!make_tensor_name(name, sizeof(name), layer, suffix)) {
        set_error(err, errcap, "K3 tensor name is too long");
        return NULL;
    }
    const k3_st_tensor *tensor = k3_st_find(shard, name);
    if (!tensor) {
        set_error(err, errcap, "missing tensor %s", name);
        return NULL;
    }
    const bool quantized_matrix =
        k3_st_is_gguf(shard) &&
        dtype == K3_ST_DTYPE_BF16 &&
        n_dims == 2 &&
        (tensor->dtype == K3_ST_DTYPE_Q8_0 ||
         tensor->dtype == K3_ST_DTYPE_MXFP4);
    if ((tensor->dtype != dtype && !quantized_matrix) ||
        tensor->n_dims != n_dims) {
        set_error(err,
                  errcap,
                  "tensor %s has dtype/rank %s/%u, expected %s/%u",
                  name,
                  k3_st_dtype_name(tensor->dtype),
                  tensor->n_dims,
                  k3_st_dtype_name(dtype),
                  n_dims);
        return NULL;
    }
    for (uint32_t i = 0; i < n_dims; i++) {
        if (tensor->shape[i] != shape[i]) {
            set_error(err,
                      errcap,
                      "tensor %s dimension %u is %" PRIu64 ", expected %" PRIu64,
                      name,
                      i,
                      tensor->shape[i],
                      shape[i]);
            return NULL;
        }
    }
    return tensor;
}

#define REQUIRE(field_, suffix_, dtype_, ...) do { \
    const uint64_t shape_[] = { __VA_ARGS__ }; \
    layer->field_ = require_tensor(layer->shard, \
                                   index, \
                                   suffix_, \
                                   dtype_, \
                                   (uint32_t)(sizeof(shape_) / sizeof(shape_[0])), \
                                   shape_, \
                                   err, \
                                   errcap); \
    if (!layer->field_) goto fail; \
} while (0)

bool k3_layer_is_mla(uint32_t index) {
    return index < K3_N_LAYER && ((index + 1u) % 4u == 0u || index == 92u);
}

int k3_model_open(k3_model **out,
                  const char *directory,
                  char *err,
                  size_t errcap) {
    if (out) *out = NULL;
    if (!out || !directory || !directory[0]) {
        set_error(err, errcap, "missing Kimi K3 model directory");
        return 0;
    }
    struct stat st;
    if (stat(directory, &st) != 0) {
        set_error(err, errcap, "missing Kimi K3 model %s: %s",
                  directory, strerror(errno));
        return 0;
    }
    const bool gguf = S_ISREG(st.st_mode);
    char index_path[4096];
    const size_t len = strlen(directory);
    const char *slash = len && directory[len - 1u] == '/' ? "" : "/";
    const int n = gguf
        ? snprintf(index_path, sizeof(index_path), "%s", directory)
        : snprintf(index_path,
                   sizeof(index_path),
                   "%s%smodel.safetensors.index.json",
                   directory,
                   slash);
    if (n <= 0 || (size_t)n >= sizeof(index_path)) {
        set_error(err, errcap, "Kimi K3 model path is too long");
        return 0;
    }
    if (stat(index_path, &st) != 0) {
        set_error(err,
                  errcap,
                  "missing %s: %s",
                  index_path,
                  strerror(errno));
        return 0;
    }
    k3_model *model = calloc(1, sizeof(*model));
    if (!model) {
        set_error(err, errcap, "out of memory opening Kimi K3");
        return 0;
    }
    model->directory = copy_string(directory);
    if (!model->directory) {
        set_error(err, errcap, "out of memory copying Kimi K3 path");
        free(model);
        return 0;
    }
    model->gguf = gguf;
    if (gguf) {
        k3_st_shard *probe = NULL;
        if (!k3_st_open(&probe, directory, err, errcap) ||
            !k3_st_is_gguf(probe)) {
            k3_st_close(probe);
            free(model->directory);
            free(model);
            return 0;
        }
        k3_st_close(probe);
    }
    *out = model;
    return 1;
}

void k3_model_close(k3_model *model) {
    if (!model) return;
    free(model->directory);
    free(model);
}

static int bind_common(k3_layer *layer,
                       uint32_t index,
                       char *err,
                       size_t errcap) {
    REQUIRE(input_norm, "input_layernorm.weight",
            K3_ST_DTYPE_BF16, K3_N_EMBD);
    REQUIRE(post_attention_norm, "post_attention_layernorm.weight",
            K3_ST_DTYPE_BF16, K3_N_EMBD);
    REQUIRE(self_attention_res_norm, "self_attention_res_norm.weight",
            K3_ST_DTYPE_BF16, K3_N_EMBD);
    REQUIRE(self_attention_res_proj, "self_attention_res_proj.weight",
            K3_ST_DTYPE_BF16, 1, K3_N_EMBD);
    REQUIRE(mlp_res_norm, "mlp_res_norm.weight",
            K3_ST_DTYPE_BF16, K3_N_EMBD);
    REQUIRE(mlp_res_proj, "mlp_res_proj.weight",
            K3_ST_DTYPE_BF16, 1, K3_N_EMBD);
    REQUIRE(g_proj, "self_attn.g_proj.weight",
            K3_ST_DTYPE_BF16, K3_KDA_PROJ, K3_N_EMBD);
    REQUIRE(o_proj, "self_attn.o_proj.weight",
            K3_ST_DTYPE_BF16, K3_N_EMBD, K3_KDA_PROJ);
    return 1;
fail:
    return 0;
}

static int bind_kda(k3_layer *layer,
                    uint32_t index,
                    char *err,
                    size_t errcap) {
    REQUIRE(a_log, "self_attn.A_log",
            K3_ST_DTYPE_F32, K3_HEAD_DIM);
    REQUIRE(dt_bias, "self_attn.dt_bias",
            K3_ST_DTYPE_F32, K3_KDA_PROJ);
    REQUIRE(q_conv, "self_attn.q_conv1d.weight",
            K3_ST_DTYPE_F32, K3_KDA_PROJ, 1, K3_CONV_WIDTH);
    REQUIRE(k_conv, "self_attn.k_conv1d.weight",
            K3_ST_DTYPE_F32, K3_KDA_PROJ, 1, K3_CONV_WIDTH);
    REQUIRE(v_conv, "self_attn.v_conv1d.weight",
            K3_ST_DTYPE_F32, K3_KDA_PROJ, 1, K3_CONV_WIDTH);
    REQUIRE(o_norm, "self_attn.o_norm.weight",
            K3_ST_DTYPE_F32, K3_HEAD_DIM);
    REQUIRE(q_proj, "self_attn.q_proj.weight",
            K3_ST_DTYPE_BF16, K3_KDA_PROJ, K3_N_EMBD);
    REQUIRE(k_proj, "self_attn.k_proj.weight",
            K3_ST_DTYPE_BF16, K3_KDA_PROJ, K3_N_EMBD);
    REQUIRE(v_proj, "self_attn.v_proj.weight",
            K3_ST_DTYPE_BF16, K3_KDA_PROJ, K3_N_EMBD);
    REQUIRE(f_a_proj, "self_attn.f_a_proj.weight",
            K3_ST_DTYPE_BF16, K3_HEAD_DIM, K3_N_EMBD);
    REQUIRE(f_b_proj, "self_attn.f_b_proj.weight",
            K3_ST_DTYPE_BF16, K3_KDA_PROJ, K3_HEAD_DIM);
    REQUIRE(b_proj, "self_attn.b_proj.weight",
            K3_ST_DTYPE_BF16, K3_N_HEAD, K3_N_EMBD);
    return 1;
fail:
    return 0;
}

static int bind_mla(k3_layer *layer,
                    uint32_t index,
                    char *err,
                    size_t errcap) {
    REQUIRE(q_a_proj, "self_attn.q_a_proj.weight",
            K3_ST_DTYPE_BF16, K3_Q_LORA, K3_N_EMBD);
    REQUIRE(q_a_norm, "self_attn.q_a_layernorm.weight",
            K3_ST_DTYPE_BF16, K3_Q_LORA);
    REQUIRE(q_b_proj, "self_attn.q_b_proj.weight",
            K3_ST_DTYPE_BF16, K3_N_HEAD * K3_QK_HEAD, K3_Q_LORA);
    REQUIRE(kv_a_proj, "self_attn.kv_a_proj_with_mqa.weight",
            K3_ST_DTYPE_BF16, K3_KV_LORA + K3_QK_ROPE, K3_N_EMBD);
    REQUIRE(kv_a_norm, "self_attn.kv_a_layernorm.weight",
            K3_ST_DTYPE_BF16, K3_KV_LORA);
    REQUIRE(kv_b_proj, "self_attn.kv_b_proj.weight",
            K3_ST_DTYPE_BF16,
            K3_N_HEAD * (K3_QK_NOPE + K3_V_HEAD),
            K3_KV_LORA);
    return 1;
fail:
    return 0;
}

static int bind_dense_ffn(k3_layer *layer,
                          uint32_t index,
                          char *err,
                          size_t errcap) {
    REQUIRE(dense_gate, "mlp.gate_proj.weight",
            K3_ST_DTYPE_BF16, K3_DENSE_FF, K3_N_EMBD);
    REQUIRE(dense_up, "mlp.up_proj.weight",
            K3_ST_DTYPE_BF16, K3_DENSE_FF, K3_N_EMBD);
    REQUIRE(dense_down, "mlp.down_proj.weight",
            K3_ST_DTYPE_BF16, K3_N_EMBD, K3_DENSE_FF);
    return 1;
fail:
    return 0;
}

static int bind_one_expert(k3_layer *layer,
                           uint32_t index,
                           uint32_t expert,
                           char *err,
                           size_t errcap) {
    static const char *names[] = {
        "w1.weight_packed", "w1.weight_scale",
        "w2.weight_packed", "w2.weight_scale",
        "w3.weight_packed", "w3.weight_scale",
    };
    const uint64_t shapes[][2] = {
        {K3_EXPERT_FF, K3_ROUTED_HIDDEN / 2u},
        {K3_EXPERT_FF, K3_ROUTED_HIDDEN / 32u},
        {K3_ROUTED_HIDDEN, K3_EXPERT_FF / 2u},
        {K3_ROUTED_HIDDEN, K3_EXPERT_FF / 32u},
        {K3_EXPERT_FF, K3_ROUTED_HIDDEN / 2u},
        {K3_EXPERT_FF, K3_ROUTED_HIDDEN / 32u},
    };
    const k3_st_tensor **fields[] = {
        &layer->experts[expert].w1_packed,
        &layer->experts[expert].w1_scale,
        &layer->experts[expert].w2_packed,
        &layer->experts[expert].w2_scale,
        &layer->experts[expert].w3_packed,
        &layer->experts[expert].w3_scale,
    };
    for (uint32_t i = 0; i < 6; i++) {
        char suffix[128];
        const int n = snprintf(suffix,
                               sizeof(suffix),
                               "block_sparse_moe.experts.%u.%s",
                               expert,
                               names[i]);
        if (n <= 0 || (size_t)n >= sizeof(suffix)) {
            set_error(err, errcap, "expert tensor name is too long");
            return 0;
        }
        *fields[i] = require_tensor(layer->shard,
                                    index,
                                    suffix,
                                    K3_ST_DTYPE_U8,
                                    2,
                                    shapes[i],
                                    err,
                                    errcap);
        if (!*fields[i]) return 0;
    }
    return 1;
}

static const k3_st_tensor *require_gguf_expert_stack(
        k3_layer *layer,
        const char *part,
        k3_st_dtype dtype,
        uint64_t out_dim,
        uint64_t in_dim,
        char *err,
        size_t errcap) {
    char name[128];
    const int n = snprintf(name, sizeof(name),
                           "blk.%u.ffn_%s_exps.weight",
                           layer->index, part);
    if (n <= 0 || (size_t)n >= sizeof(name)) {
        set_error(err, errcap, "K3 GGUF expert tensor name is too long");
        return NULL;
    }
    const k3_st_tensor *tensor = k3_st_find(layer->shard, name);
    if (!tensor || tensor->dtype != dtype || tensor->n_dims != 3 ||
        tensor->shape[0] != K3_N_EXPERT ||
        tensor->shape[1] != out_dim ||
        tensor->shape[2] != in_dim) {
        set_error(err, errcap,
                  "missing or invalid K3 GGUF expert tensor %s", name);
        return NULL;
    }
    return tensor;
}

static int bind_moe(k3_layer *layer,
                    uint32_t index,
                    char *err,
                    size_t errcap) {
    REQUIRE(router, "block_sparse_moe.gate.weight",
            K3_ST_DTYPE_BF16, K3_N_EXPERT, K3_N_EMBD);
    REQUIRE(router_bias, "block_sparse_moe.gate.e_score_correction_bias",
            K3_ST_DTYPE_F32, K3_N_EXPERT);
    REQUIRE(routed_down, "block_sparse_moe.routed_expert_down_proj.weight",
            K3_ST_DTYPE_BF16, K3_ROUTED_HIDDEN, K3_N_EMBD);
    REQUIRE(routed_norm, "block_sparse_moe.routed_expert_norm.weight",
            K3_ST_DTYPE_BF16, K3_ROUTED_HIDDEN);
    REQUIRE(routed_up, "block_sparse_moe.routed_expert_up_proj.weight",
            K3_ST_DTYPE_BF16, K3_N_EMBD, K3_ROUTED_HIDDEN);
    REQUIRE(shared_gate, "block_sparse_moe.shared_experts.gate_proj.weight",
            K3_ST_DTYPE_BF16, K3_SHARED_FF, K3_N_EMBD);
    REQUIRE(shared_up, "block_sparse_moe.shared_experts.up_proj.weight",
            K3_ST_DTYPE_BF16, K3_SHARED_FF, K3_N_EMBD);
    REQUIRE(shared_down, "block_sparse_moe.shared_experts.down_proj.weight",
            K3_ST_DTYPE_BF16, K3_N_EMBD, K3_SHARED_FF);
    if (k3_st_is_gguf(layer->shard)) {
        layer->expert_gate = require_gguf_expert_stack(
            layer, "gate", K3_ST_DTYPE_IQ2_XXS,
            K3_EXPERT_FF, K3_ROUTED_HIDDEN, err, errcap);
        layer->expert_up = require_gguf_expert_stack(
            layer, "up", K3_ST_DTYPE_IQ2_XXS,
            K3_EXPERT_FF, K3_ROUTED_HIDDEN, err, errcap);
        layer->expert_down = require_gguf_expert_stack(
            layer, "down", K3_ST_DTYPE_Q2_K,
            K3_ROUTED_HIDDEN, K3_EXPERT_FF, err, errcap);
        if (!layer->expert_gate || !layer->expert_up ||
            !layer->expert_down) {
            return 0;
        }
    } else {
        for (uint32_t expert = 0; expert < K3_N_EXPERT; expert++) {
            if (!bind_one_expert(layer, index, expert, err, errcap)) return 0;
        }
    }
    return 1;
fail:
    return 0;
}

static int set_static_span(k3_layer *layer,
                           uint32_t index,
                           char *err,
                           size_t errcap) {
    const k3_st_tensor *static_tensors[] = {
        layer->input_norm,
        layer->post_attention_norm,
        layer->self_attention_res_norm,
        layer->self_attention_res_proj,
        layer->mlp_res_norm,
        layer->mlp_res_proj,
        layer->a_log,
        layer->dt_bias,
        layer->q_conv,
        layer->k_conv,
        layer->v_conv,
        layer->o_norm,
        layer->q_proj,
        layer->k_proj,
        layer->v_proj,
        layer->f_a_proj,
        layer->f_b_proj,
        layer->b_proj,
        layer->g_proj,
        layer->o_proj,
        layer->q_a_proj,
        layer->q_a_norm,
        layer->q_b_proj,
        layer->kv_a_proj,
        layer->kv_a_norm,
        layer->kv_b_proj,
        layer->router,
        layer->router_bias,
        layer->routed_down,
        layer->routed_norm,
        layer->routed_up,
        layer->shared_gate,
        layer->shared_up,
        layer->shared_down,
        layer->dense_gate,
        layer->dense_up,
        layer->dense_down,
    };

    if (k3_st_is_gguf(layer->shard)) {
        uint64_t static_begin = UINT64_MAX;
        uint64_t static_end = 0;
        for (uint32_t i = 0;
             i < sizeof(static_tensors) / sizeof(static_tensors[0]);
             i++) {
            const k3_st_tensor *tensor = static_tensors[i];
            if (!tensor) continue;
            if (tensor->size > UINT64_MAX - tensor->offset) {
                set_error(err, errcap, "K3 GGUF static tensor overflow");
                return 0;
            }
            if (tensor->offset < static_begin) static_begin = tensor->offset;
            if (tensor->offset + tensor->size > static_end) {
                static_end = tensor->offset + tensor->size;
            }
        }
        if (static_begin == UINT64_MAX || static_end <= static_begin) {
            set_error(err, errcap, "K3 GGUF layer has no static tensor span");
            return 0;
        }
        uint64_t layer_end = static_end;
        const k3_st_tensor *expert_tensors[] = {
            layer->expert_gate, layer->expert_up, layer->expert_down,
        };
        for (uint32_t i = 0; i < 3; i++) {
            const k3_st_tensor *tensor = expert_tensors[i];
            if (!tensor) continue;
            if (tensor->offset < static_end ||
                tensor->size > UINT64_MAX - tensor->offset) {
                set_error(err, errcap,
                          "K3 GGUF layer %u expert tensor order is invalid",
                          index);
                return 0;
            }
            if (tensor->offset + tensor->size > layer_end) {
                layer_end = tensor->offset + tensor->size;
            }
        }
        layer->static_offset = static_begin;
        layer->static_size = static_end - static_begin;
        layer->layer_size = layer_end - static_begin;
        return 1;
    }

    layer->static_offset = k3_st_data_offset(layer->shard);
    layer->layer_size =
        k3_st_file_size(layer->shard) - layer->static_offset;
    if (index == 0) {
        layer->static_size = layer->layer_size;
        return 1;
    }

    uint64_t first_expert = UINT64_MAX;
    for (uint32_t expert = 0; expert < K3_N_EXPERT; expert++) {
        const k3_expert_weights *weights = &layer->experts[expert];
        const k3_st_tensor *tensors[] = {
            weights->w1_packed, weights->w1_scale,
            weights->w2_packed, weights->w2_scale,
            weights->w3_packed, weights->w3_scale,
        };
        uint64_t expected = tensors[0]->offset;
        for (uint32_t i = 0; i < 6; i++) {
            if (tensors[i]->offset != expected ||
                tensors[i]->size > UINT64_MAX - expected) {
                set_error(err, errcap,
                          "K3 layer %u expert %u is not contiguous in "
                          "w1/w1_scale/w2/w2_scale/w3/w3_scale order",
                          index, expert);
                return 0;
            }
            expected += tensors[i]->size;
            if (tensors[i]->offset < first_expert) {
                first_expert = tensors[i]->offset;
            }
        }
    }
    if (first_expert <= layer->static_offset) {
        set_error(err, errcap, "invalid K3 expert data boundary");
        return 0;
    }

    for (uint32_t i = 0;
         i < sizeof(static_tensors) / sizeof(static_tensors[0]);
         i++) {
        const k3_st_tensor *tensor = static_tensors[i];
        if (tensor && (tensor->offset > first_expert ||
                       tensor->size > first_expert - tensor->offset)) {
            set_error(
                err, errcap,
                "K3 layer %u static tensor %s follows expert data",
                index, tensor->name);
            return 0;
        }
    }
    layer->static_size = first_expert - layer->static_offset;
    return 1;
}

int k3_model_open_layer(const k3_model *model,
                        uint32_t index,
                        k3_layer **out,
                        char *err,
                        size_t errcap) {
    if (out) *out = NULL;
    if (!model || !out || index >= K3_N_LAYER) {
        set_error(err, errcap, "invalid K3 layer index");
        return 0;
    }
    char path[4096];
    if (!make_model_path(path, sizeof(path), model, index + 1u)) {
        set_error(err, errcap, "K3 shard path is too long");
        return 0;
    }
    k3_layer *layer = calloc(1, sizeof(*layer));
    if (!layer) {
        set_error(err, errcap, "out of memory opening K3 layer");
        return 0;
    }
    layer->index = index;
    layer->attention = k3_layer_is_mla(index) ? K3_LAYER_MLA : K3_LAYER_KDA;
    if (!k3_st_open(&layer->shard, path, err, errcap) ||
        !k3_st_map(layer->shard, err, errcap) ||
        !bind_common(layer, index, err, errcap) ||
        (layer->attention == K3_LAYER_KDA ?
            !bind_kda(layer, index, err, errcap) :
            !bind_mla(layer, index, err, errcap)) ||
        (index == 0 ?
            !bind_dense_ffn(layer, index, err, errcap) :
            !bind_moe(layer, index, err, errcap)) ||
        !set_static_span(layer, index, err, errcap)) {
        goto fail;
    }
    *out = layer;
    return 1;

fail:
    k3_layer_close(layer);
    return 0;
}

void k3_layer_close(k3_layer *layer) {
    if (!layer) return;
    k3_st_close(layer->shard);
    free(layer);
}

static const k3_st_tensor *require_global(
        const k3_st_shard *shard,
        const char *name,
        uint32_t n_dims,
        const uint64_t *shape,
        char *err,
        size_t errcap) {
    const k3_st_tensor *tensor = k3_st_find(shard, name);
    if (!tensor || tensor->dtype != K3_ST_DTYPE_BF16 ||
        tensor->n_dims != n_dims) {
        set_error(err, errcap, "missing or invalid K3 global tensor %s", name);
        return NULL;
    }
    for (uint32_t i = 0; i < n_dims; i++) {
        if (tensor->shape[i] != shape[i]) {
            set_error(err, errcap, "invalid shape for K3 global tensor %s", name);
            return NULL;
        }
    }
    return tensor;
}

int k3_model_open_globals(const k3_model *model,
                          k3_global_weights *out,
                          char *err,
                          size_t errcap) {
    if (!model || !out) {
        set_error(err, errcap, "missing K3 globals output");
        return 0;
    }
    memset(out, 0, sizeof(*out));
    char path[4096];
    if (!make_model_path(path, sizeof(path), model, 94)) {
        set_error(err, errcap, "K3 global shard path is too long");
        return 0;
    }
    if (!k3_st_open(&out->shard, path, err, errcap) ||
        !k3_st_map(out->shard, err, errcap)) {
        goto fail;
    }
    const uint64_t matrix[] = {K3_N_VOCAB, K3_N_EMBD};
    const uint64_t vector[] = {K3_N_EMBD};
    out->lm_head = require_global(out->shard,
                                  "language_model.lm_head.weight",
                                  2,
                                  matrix,
                                  err,
                                  errcap);
    out->token_embedding = require_global(
            out->shard,
            "language_model.model.embed_tokens.weight",
            2,
            matrix,
            err,
            errcap);
    out->norm = require_global(out->shard,
                               "language_model.model.norm.weight",
                               1,
                               vector,
                               err,
                               errcap);
    out->output_attn_res_norm = require_global(
            out->shard,
            "language_model.model.output_attn_res_norm.weight",
            1,
            vector,
            err,
            errcap);
    out->output_attn_res_proj = require_global(
            out->shard,
            "language_model.model.output_attn_res_proj.weight",
            2,
            (const uint64_t[]){1, K3_N_EMBD},
            err,
            errcap);
    if (!out->lm_head || !out->token_embedding || !out->norm ||
        !out->output_attn_res_norm || !out->output_attn_res_proj) {
        goto fail;
    }
    return 1;

fail:
    k3_global_weights_close(out);
    return 0;
}

void k3_global_weights_close(k3_global_weights *weights) {
    if (!weights) return;
    k3_st_close(weights->shard);
    memset(weights, 0, sizeof(*weights));
}

static void *load_thread_main(void *arg) {
    k3_layer_load *load = arg;
    if (!load->layer) {
        load->ok = k3_model_open_layer(load->model,
                                       load->index,
                                       &load->layer,
                                       load->error,
                                       sizeof(load->error));
        if (!load->ok) return NULL;
        load->owns_layer = true;
    }
    if (load->map_layer &&
        !k3_st_map(load->layer->shard,
                   load->error,
                   sizeof(load->error))) {
        load->ok = 0;
        return NULL;
    }
    const uint64_t offset = load->layer->static_offset;
    const uint64_t size = load->full_layer ?
        load->layer->layer_size :
        load->layer->static_size;
    /*
     * Match the established Metal SSD-streaming path: ask the filesystem to
     * read ahead, but do not fault every page from the CPU. Metal accesses
     * fault only the tensor pages actually consumed by this layer.
     */
    load->ok = k3_st_prefetch(load->layer->shard,
                              offset,
                              size,
                              load->pagein,
                              load->error,
                              sizeof(load->error));
    return NULL;
}

int k3_layer_load_start(k3_layer_load **out,
                        const k3_model *model,
                        uint32_t index,
                        bool full_layer,
                        char *err,
                        size_t errcap) {
    if (out) *out = NULL;
    if (!out || !model || index >= K3_N_LAYER) {
        set_error(err, errcap, "invalid asynchronous K3 layer load");
        return 0;
    }
    k3_layer_load *load = calloc(1, sizeof(*load));
    if (!load) {
        set_error(err, errcap, "out of memory starting K3 layer load");
        return 0;
    }
    load->model = model;
    load->index = index;
    load->full_layer = full_layer;
    load->map_layer = true;
    load->ok = 1;
    const int rc = pthread_create(&load->thread, NULL, load_thread_main, load);
    if (rc != 0) {
        set_error(err, errcap, "pthread_create: %s", strerror(rc));
        free(load);
        return 0;
    }
    load->started = true;
    *out = load;
    return 1;
}

int k3_layer_prefetch_start(k3_layer_load **out,
                            k3_layer *layer,
                            bool full_layer,
                            bool pagein,
                            char *err,
                            size_t errcap) {
    if (out) *out = NULL;
    if (!out || !layer || !layer->shard) {
        set_error(err, errcap, "invalid asynchronous K3 layer prefetch");
        return 0;
    }
    k3_layer_load *load = calloc(1, sizeof(*load));
    if (!load) {
        set_error(err, errcap, "out of memory starting K3 layer prefetch");
        return 0;
    }
    load->index = layer->index;
    load->full_layer = full_layer;
    load->pagein = pagein;
    /*
     * A GGUF context keeps one canonical whole-file mapping through its
     * globals descriptor. The per-layer descriptor is needed only for
     * metadata and read-ahead, unless explicit CPU page touching was asked
     * for.
     */
    load->map_layer = !k3_st_is_gguf(layer->shard) || pagein;
    load->layer = layer;
    load->ok = 1;
    const int rc = pthread_create(&load->thread, NULL, load_thread_main, load);
    if (rc != 0) {
        set_error(err, errcap, "pthread_create: %s", strerror(rc));
        free(load);
        return 0;
    }
    load->started = true;
    *out = load;
    return 1;
}

int k3_layer_readahead_start(k3_layer_load **out,
                             k3_layer *layer,
                             bool full_layer,
                             char *err,
                             size_t errcap) {
    if (out) *out = NULL;
    if (!out || !layer || !layer->shard) {
        set_error(err, errcap, "invalid asynchronous K3 layer read-ahead");
        return 0;
    }
    k3_layer_load *load = calloc(1, sizeof(*load));
    if (!load) {
        set_error(err, errcap, "out of memory starting K3 layer read-ahead");
        return 0;
    }
    load->index = layer->index;
    load->full_layer = full_layer;
    load->layer = layer;
    load->ok = 1;
    const int rc = pthread_create(&load->thread, NULL, load_thread_main, load);
    if (rc != 0) {
        set_error(err, errcap, "pthread_create: %s", strerror(rc));
        free(load);
        return 0;
    }
    load->started = true;
    *out = load;
    return 1;
}

int k3_layer_load_join(k3_layer_load *load,
                       k3_layer **out,
                       char *err,
                       size_t errcap) {
    if (out) *out = NULL;
    if (!load || !out || !load->started) {
        set_error(err, errcap, "invalid K3 layer load join");
        return 0;
    }
    const int rc = pthread_join(load->thread, NULL);
    load->started = false;
    if (rc != 0) {
        set_error(err, errcap, "pthread_join: %s", strerror(rc));
        if (load->owns_layer) {
            k3_layer_close(load->layer);
        } else if (load->layer) {
            k3_st_unmap(load->layer->shard);
        }
        free(load);
        return 0;
    }
    if (!load->ok) {
        set_error(err,
                  errcap,
                  "%s",
                  load->error[0] ? load->error : "K3 layer load failed");
        if (load->owns_layer) {
            k3_layer_close(load->layer);
        } else if (load->layer) {
            k3_st_unmap(load->layer->shard);
        }
        free(load);
        return 0;
    }
    *out = load->layer;
    free(load);
    return 1;
}

void k3_layer_load_cancel(k3_layer_load *load) {
    if (!load) return;
    if (load->started) {
        (void)pthread_join(load->thread, NULL);
        load->started = false;
    }
    if (load->owns_layer) {
        k3_layer_close(load->layer);
    } else if (load->layer) {
        k3_st_unmap(load->layer->shard);
    }
    free(load);
}
