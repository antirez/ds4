/*
 * Kimi K3 SafeTensors -> GGUF converter.
 *
 * The converter is deliberately bounded: one source layer is open at a time,
 * regular tensors are copied/quantized in row chunks, and routed experts are
 * handled by a fixed worker pool. Completed GGUF tensors are journaled so an
 * interrupted conversion can resume without rewriting the full model.
 */

#define _DARWIN_C_SOURCE
#define _POSIX_C_SOURCE 200809L

#include "quants.h"
#include "../k3_model.h"

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

enum {
    K3Q_GGUF_VERSION = 3,
    K3Q_ALIGNMENT = 32,
    K3Q_KV_COUNT = 14,
    K3Q_COPY_CHUNK = 16 * 1024 * 1024,
    K3Q_FLOAT_CHUNK = 32 * 1024 * 1024,
};

typedef enum {
    K3Q_SOURCE_REGULAR,
    K3Q_SOURCE_EXPERT_GATE,
    K3Q_SOURCE_EXPERT_UP,
    K3Q_SOURCE_EXPERT_DOWN,
} k3q_source_kind;

typedef struct {
    char *name;
    char *source_name;
    uint32_t layer;
    k3q_source_kind source_kind;
    ds4q_type type;
    uint32_t n_dims;
    uint64_t ne[DS4Q_MAX_DIMS];
    uint64_t size;
    uint64_t offset;
} k3q_tensor;

typedef struct {
    k3q_tensor *tensors;
    uint32_t count;
    uint32_t cap;
    uint64_t data_offset;
    uint64_t file_size;
    uint64_t tensor_payload;
    char *tokenizer;
    size_t tokenizer_size;
} k3q_plan;

typedef struct {
    const char *model_dir;
    const char *output;
    uint32_t threads;
    bool dry_run;
    bool resume;
    bool overwrite;
} k3q_options;

typedef struct {
    uint32_t layer;
    float *expert_input;
    float *down_input;
} k3q_importance;

static void fail(const char *message) {
    fprintf(stderr, "k3-quantize: %s\n", message);
    exit(1);
}

static void fail_errno(const char *action, const char *path) {
    fprintf(stderr, "k3-quantize: %s %s: %s\n",
            action, path ? path : "", strerror(errno));
    exit(1);
}

static void unlink_if_exists(const char *path) {
    if (unlink(path) != 0 && errno != ENOENT) {
        fail_errno("remove", path);
    }
}

static void sync_file_durable(int fd, const char *path) {
    if (fsync(fd) != 0) fail_errno("sync", path);
#if defined(F_FULLFSYNC)
    if (fcntl(fd, F_FULLFSYNC) != 0) {
        fail_errno("durably sync", path);
    }
#endif
}

static void *xmalloc(size_t bytes) {
    void *ptr = malloc(bytes ? bytes : 1u);
    if (!ptr) fail("out of memory");
    return ptr;
}

static void *xcalloc(size_t count, size_t bytes) {
    void *ptr = calloc(count ? count : 1u, bytes ? bytes : 1u);
    if (!ptr) fail("out of memory");
    return ptr;
}

static void *xrealloc(void *old, size_t bytes) {
    void *ptr = realloc(old, bytes ? bytes : 1u);
    if (!ptr) fail("out of memory");
    return ptr;
}

static char *xstrdup(const char *text) {
    const size_t length = strlen(text);
    char *copy = xmalloc(length + 1u);
    memcpy(copy, text, length + 1u);
    return copy;
}

static bool starts_with(const char *text, const char *prefix) {
    return strncmp(text, prefix, strlen(prefix)) == 0;
}

static bool contains(const char *text, const char *needle) {
    return strstr(text, needle) != NULL;
}

static uint64_t align_u64(uint64_t value, uint64_t alignment) {
    if (value > UINT64_MAX - (alignment - 1u)) fail("size overflow");
    return (value + alignment - 1u) / alignment * alignment;
}

static char *join_path(const char *directory, const char *name) {
    const size_t nd = strlen(directory);
    const size_t nn = strlen(name);
    const bool slash = nd && directory[nd - 1u] == '/';
    char *path = xmalloc(nd + (slash ? 0u : 1u) + nn + 1u);
    memcpy(path, directory, nd);
    size_t pos = nd;
    if (!slash) path[pos++] = '/';
    memcpy(path + pos, name, nn + 1u);
    return path;
}

static char *append_suffix(const char *path, const char *suffix) {
    const size_t np = strlen(path);
    const size_t ns = strlen(suffix);
    char *result = xmalloc(np + ns + 1u);
    memcpy(result, path, np);
    memcpy(result + np, suffix, ns + 1u);
    return result;
}

static char *read_small_file(const char *path, size_t *size_out) {
    FILE *fp = fopen(path, "rb");
    if (!fp) fail_errno("open", path);
    if (fseeko(fp, 0, SEEK_END) != 0) fail_errno("seek", path);
    const off_t end = ftello(fp);
    if (end < 0 || (uint64_t)end > SIZE_MAX - 1u) {
        fail("small file is too large");
    }
    if (fseeko(fp, 0, SEEK_SET) != 0) fail_errno("seek", path);
    char *data = xmalloc((size_t)end + 1u);
    if (end && fread(data, 1, (size_t)end, fp) != (size_t)end) {
        fail_errno("read", path);
    }
    fclose(fp);
    data[end] = '\0';
    if (size_out) *size_out = (size_t)end;
    return data;
}

static int pread_full(int fd, void *buffer, size_t bytes, uint64_t offset) {
    uint8_t *dst = buffer;
    size_t done = 0;
    while (done < bytes) {
        if (offset + done > (uint64_t)LLONG_MAX) return 0;
        ssize_t n;
        do {
            n = pread(fd, dst + done, bytes - done,
                      (off_t)(offset + done));
        } while (n < 0 && errno == EINTR);
        if (n <= 0) return 0;
        done += (size_t)n;
    }
    return 1;
}

static int pwrite_full(int fd, const void *buffer, size_t bytes,
                       uint64_t offset) {
    const uint8_t *src = buffer;
    size_t done = 0;
    while (done < bytes) {
        if (offset + done > (uint64_t)LLONG_MAX) return 0;
        ssize_t n;
        do {
            n = pwrite(fd, src + done, bytes - done,
                       (off_t)(offset + done));
        } while (n < 0 && errno == EINTR);
        if (n <= 0) return 0;
        done += (size_t)n;
    }
    return 1;
}

static ds4q_type source_gguf_type(k3_st_dtype type) {
    switch (type) {
    case K3_ST_DTYPE_F32:  return DS4Q_TYPE_F32;
    case K3_ST_DTYPE_F16:  return DS4Q_TYPE_F16;
    case K3_ST_DTYPE_BF16: return DS4Q_TYPE_BF16;
    default:               fail("unsupported regular SafeTensors dtype");
    }
    return DS4Q_TYPE_COUNT;
}

static uint64_t tensor_size(ds4q_type type, const uint64_t *ne,
                            uint32_t n_dims) {
    if (n_dims == 0 || n_dims > DS4Q_MAX_DIMS ||
        ne[0] > INT64_MAX) {
        fail("invalid tensor shape");
    }
    const size_t row = ds4q_row_size(type, (int64_t)ne[0]);
    if (!row) fail("tensor row is incompatible with target type");
    uint64_t bytes = row;
    for (uint32_t i = 1; i < n_dims; i++) {
        if (ne[i] && bytes > UINT64_MAX / ne[i]) fail("tensor size overflow");
        bytes *= ne[i];
    }
    return bytes;
}

static void plan_append(k3q_plan *plan, const k3q_tensor *tensor) {
    if (plan->count == plan->cap) {
        const uint32_t cap = plan->cap ? plan->cap * 2u : 512u;
        plan->tensors =
            xrealloc(plan->tensors, (size_t)cap * sizeof(plan->tensors[0]));
        plan->cap = cap;
    }
    plan->tensors[plan->count++] = *tensor;
}

static ds4q_type regular_target_type(uint32_t layer,
                                     const k3_st_tensor *source) {
    if (contains(source->name, ".block_sparse_moe.shared_experts.")) {
        return DS4Q_TYPE_MXFP4;
    }
    if (contains(source->name,
                 ".block_sparse_moe.routed_expert_up_proj.weight")) {
        return DS4Q_TYPE_Q8_0;
    }
    if (layer == 0 &&
        (contains(source->name, ".mlp.gate_proj.weight") ||
         contains(source->name, ".mlp.up_proj.weight"))) {
        return DS4Q_TYPE_Q8_0;
    }
    return source_gguf_type(source->dtype);
}

static void plan_add_regular(k3q_plan *plan, uint32_t layer,
                             const k3_st_tensor *source) {
    k3q_tensor tensor = {
        .name = xstrdup(source->name),
        .source_name = xstrdup(source->name),
        .layer = layer,
        .source_kind = K3Q_SOURCE_REGULAR,
        .type = regular_target_type(layer, source),
        .n_dims = source->n_dims,
    };
    if (tensor.n_dims == 0 || tensor.n_dims > DS4Q_MAX_DIMS) {
        fail("unsupported K3 tensor rank");
    }
    for (uint32_t i = 0; i < tensor.n_dims; i++) {
        tensor.ne[i] = source->shape[tensor.n_dims - 1u - i];
    }
    tensor.size = tensor_size(tensor.type, tensor.ne, tensor.n_dims);
    plan_append(plan, &tensor);
}

static char *expert_tensor_name(uint32_t layer, const char *part) {
    char name[128];
    const int n = snprintf(name, sizeof(name),
                           "blk.%u.ffn_%s_exps.weight", layer, part);
    if (n <= 0 || (size_t)n >= sizeof(name)) fail("expert name overflow");
    return xstrdup(name);
}

static void plan_add_expert_stack(k3q_plan *plan, uint32_t layer,
                                  const char *part,
                                  k3q_source_kind source_kind,
                                  ds4q_type type,
                                  uint64_t in_dim,
                                  uint64_t out_dim) {
    k3q_tensor tensor = {
        .name = expert_tensor_name(layer, part),
        .source_name = NULL,
        .layer = layer,
        .source_kind = source_kind,
        .type = type,
        .n_dims = 3,
        .ne = {in_dim, out_dim, K3_N_EXPERT},
    };
    tensor.size = tensor_size(tensor.type, tensor.ne, tensor.n_dims);
    plan_append(plan, &tensor);
}

static bool is_routed_source_tensor(const char *name) {
    return contains(name, ".block_sparse_moe.experts.");
}

static void build_layer_plan(k3q_plan *plan, const k3_model *model,
                             uint32_t layer_index) {
    char error[512] = {0};
    k3_layer *layer = NULL;
    if (!k3_model_open_layer(model, layer_index, &layer,
                             error, sizeof(error))) {
        fprintf(stderr, "k3-quantize: layer %u: %s\n",
                layer_index, error[0] ? error : "open failed");
        exit(1);
    }
    const uint32_t count = k3_st_tensor_count(layer->shard);
    for (uint32_t i = 0; i < count; i++) {
        const k3_st_tensor *tensor = k3_st_tensor_at(layer->shard, i);
        if (!tensor || is_routed_source_tensor(tensor->name)) continue;
        if (!starts_with(tensor->name, "language_model.")) continue;
        plan_add_regular(plan, layer_index, tensor);
    }
    if (layer_index != 0) {
        plan_add_expert_stack(plan, layer_index, "gate",
                              K3Q_SOURCE_EXPERT_GATE,
                              DS4Q_TYPE_IQ2_XXS,
                              K3_ROUTED_HIDDEN, K3_EXPERT_FF);
        plan_add_expert_stack(plan, layer_index, "up",
                              K3Q_SOURCE_EXPERT_UP,
                              DS4Q_TYPE_IQ2_XXS,
                              K3_ROUTED_HIDDEN, K3_EXPERT_FF);
        plan_add_expert_stack(plan, layer_index, "down",
                              K3Q_SOURCE_EXPERT_DOWN,
                              DS4Q_TYPE_Q2_K,
                              K3_EXPERT_FF, K3_ROUTED_HIDDEN);
    }
    k3_layer_close(layer);
}

static void build_global_plan(k3q_plan *plan, const k3_model *model) {
    char error[512] = {0};
    k3_global_weights globals = {0};
    if (!k3_model_open_globals(model, &globals, error, sizeof(error))) {
        fprintf(stderr, "k3-quantize: globals: %s\n",
                error[0] ? error : "open failed");
        exit(1);
    }
    const uint32_t count = k3_st_tensor_count(globals.shard);
    for (uint32_t i = 0; i < count; i++) {
        const k3_st_tensor *tensor = k3_st_tensor_at(globals.shard, i);
        if (!tensor || !starts_with(tensor->name, "language_model.")) {
            continue;
        }
        plan_add_regular(plan, UINT32_MAX, tensor);
    }
    k3_global_weights_close(&globals);
}

static uint64_t gguf_string_size(size_t length) {
    return 8u + length;
}

static uint64_t kv_string_size(const char *key, size_t value_length) {
    return gguf_string_size(strlen(key)) + 4u +
           gguf_string_size(value_length);
}

static uint64_t kv_u32_size(const char *key) {
    return gguf_string_size(strlen(key)) + 4u + 4u;
}

static uint64_t kv_bool_size(const char *key) {
    return gguf_string_size(strlen(key)) + 4u + 1u;
}

static void finalize_plan(k3q_plan *plan) {
    uint64_t metadata = 4u + 4u + 8u + 8u;
    metadata += kv_string_size("general.architecture",
                               strlen("kimi-k3"));
    metadata += kv_string_size("general.name",
                               strlen("Kimi K3 DwarfStar Q2"));
    metadata += kv_u32_size("general.alignment");
    metadata += kv_u32_size("general.file_type");
    metadata += kv_u32_size("kimi-k3.block_count");
    metadata += kv_u32_size("kimi-k3.embedding_length");
    metadata += kv_u32_size("kimi-k3.context_length");
    metadata += kv_u32_size("kimi-k3.vocabulary_size");
    metadata += kv_u32_size("kimi-k3.expert_count");
    metadata += kv_u32_size("kimi-k3.expert_used_count");
    metadata += kv_u32_size("kimi-k3.routed_hidden_length");
    metadata += kv_u32_size("kimi-k3.expert_feed_forward_length");
    metadata += kv_bool_size("kimi-k3.vision");
    metadata += kv_string_size("tokenizer.kimi-k3.tiktoken",
                               plan->tokenizer_size);

    for (uint32_t i = 0; i < plan->count; i++) {
        const k3q_tensor *tensor = &plan->tensors[i];
        metadata += gguf_string_size(strlen(tensor->name));
        metadata += 4u + (uint64_t)tensor->n_dims * 8u + 4u + 8u;
    }
    plan->data_offset = align_u64(metadata, K3Q_ALIGNMENT);

    uint64_t offset = 0;
    for (uint32_t i = 0; i < plan->count; i++) {
        k3q_tensor *tensor = &plan->tensors[i];
        tensor->offset = offset;
        if (tensor->size > UINT64_MAX - plan->tensor_payload) {
            fail("payload size overflow");
        }
        plan->tensor_payload += tensor->size;
        offset += align_u64(tensor->size, K3Q_ALIGNMENT);
    }
    plan->file_size = plan->data_offset + offset;
}

static k3q_plan build_plan(const char *model_dir) {
    k3q_plan plan = {0};
    char *tokenizer_path = join_path(model_dir, "tiktoken.model");
    plan.tokenizer = read_small_file(tokenizer_path, &plan.tokenizer_size);
    free(tokenizer_path);

    char error[512] = {0};
    k3_model *model = NULL;
    if (!k3_model_open(&model, model_dir, error, sizeof(error))) {
        fprintf(stderr, "k3-quantize: %s\n",
                error[0] ? error : "could not open model");
        exit(1);
    }
    for (uint32_t layer = 0; layer < K3_N_LAYER; layer++) {
        fprintf(stderr, "k3-quantize: planning layer %u/%u\r",
                layer + 1u, (uint32_t)K3_N_LAYER);
        build_layer_plan(&plan, model, layer);
    }
    fprintf(stderr, "\n");
    build_global_plan(&plan, model);
    k3_model_close(model);
    finalize_plan(&plan);
    return plan;
}

static void free_plan(k3q_plan *plan) {
    for (uint32_t i = 0; i < plan->count; i++) {
        free(plan->tensors[i].name);
        free(plan->tensors[i].source_name);
    }
    free(plan->tensors);
    free(plan->tokenizer);
    memset(plan, 0, sizeof(*plan));
}

static void print_plan(const k3q_plan *plan) {
    uint64_t by_type[DS4Q_TYPE_COUNT] = {0};
    uint32_t count_by_type[DS4Q_TYPE_COUNT] = {0};
    uint64_t by_layer[K3_N_LAYER] = {0};
    uint64_t coordinator_globals = 0;
    uint64_t output_globals = 0;
    for (uint32_t i = 0; i < plan->count; i++) {
        const k3q_tensor *tensor = &plan->tensors[i];
        if ((uint32_t)tensor->type < DS4Q_TYPE_COUNT) {
            by_type[tensor->type] += tensor->size;
            count_by_type[tensor->type]++;
        }
        if (tensor->layer < K3_N_LAYER) {
            by_layer[tensor->layer] += tensor->size;
        } else if (!strcmp(
                       tensor->name,
                       "language_model.model.embed_tokens.weight")) {
            coordinator_globals += tensor->size;
        } else {
            output_globals += tensor->size;
        }
    }
    printf("tensors: %u\n", plan->count);
    printf("metadata_bytes: %" PRIu64 "\n", plan->data_offset);
    printf("payload_bytes: %" PRIu64 "\n", plan->tensor_payload);
    printf("file_bytes: %" PRIu64 "\n", plan->file_size);
    printf("file_gib: %.9f\n",
           (double)plan->file_size / 1073741824.0);
    for (uint32_t type = 0; type < DS4Q_TYPE_COUNT; type++) {
        if (!count_by_type[type]) continue;
        printf("type: %-8s tensors=%u bytes=%" PRIu64 " gib=%.9f\n",
               ds4q_type_name((ds4q_type)type),
               count_by_type[type],
               by_type[type],
               (double)by_type[type] / 1073741824.0);
    }
    uint64_t suffix = output_globals;
    for (uint32_t i = 0; i < K3_N_LAYER; i++) suffix += by_layer[i];
    uint64_t prefix = coordinator_globals;
    uint64_t best_difference = UINT64_MAX;
    uint64_t best_first = 0;
    uint64_t best_second = 0;
    uint32_t best_split = 1;
    for (uint32_t split = 1; split < K3_N_LAYER; split++) {
        prefix += by_layer[split - 1u];
        suffix -= by_layer[split - 1u];
        const uint64_t difference =
            prefix > suffix ? prefix - suffix : suffix - prefix;
        if (difference < best_difference) {
            best_difference = difference;
            best_first = prefix;
            best_second = suffix;
            best_split = split;
        }
    }
    printf("two_node_split: 0:%u bytes=%" PRIu64
           " gib=%.9f, %u:output bytes=%" PRIu64 " gib=%.9f\n",
           best_split - 1u,
           best_first,
           (double)best_first / 1073741824.0,
           best_split,
           best_second,
           (double)best_second / 1073741824.0);
}

static void write_u32(FILE *fp, uint32_t value) {
    if (fwrite(&value, sizeof(value), 1, fp) != 1) fail("write u32 failed");
}

static void write_u64(FILE *fp, uint64_t value) {
    if (fwrite(&value, sizeof(value), 1, fp) != 1) fail("write u64 failed");
}

static void write_string_n(FILE *fp, const char *value, size_t length) {
    write_u64(fp, length);
    if (length && fwrite(value, 1, length, fp) != length) {
        fail("write string failed");
    }
}

static void write_string(FILE *fp, const char *value) {
    write_string_n(fp, value, strlen(value));
}

static void write_kv_string(FILE *fp, const char *key,
                            const char *value, size_t length) {
    write_string(fp, key);
    write_u32(fp, 8); /* GGUF_TYPE_STRING */
    write_string_n(fp, value, length);
}

static void write_kv_u32(FILE *fp, const char *key, uint32_t value) {
    write_string(fp, key);
    write_u32(fp, 4); /* GGUF_TYPE_UINT32 */
    write_u32(fp, value);
}

static void write_kv_bool(FILE *fp, const char *key, bool value) {
    const uint8_t byte = value ? 1u : 0u;
    write_string(fp, key);
    write_u32(fp, 7); /* GGUF_TYPE_BOOL */
    if (fwrite(&byte, 1, 1, fp) != 1) fail("write bool failed");
}

static void write_zeros(FILE *fp, uint64_t bytes) {
    static const uint8_t zeros[4096] = {0};
    while (bytes) {
        const size_t chunk =
            bytes < sizeof(zeros) ? (size_t)bytes : sizeof(zeros);
        if (fwrite(zeros, 1, chunk, fp) != chunk) {
            fail("write padding failed");
        }
        bytes -= chunk;
    }
}

static void write_header_contents(FILE *fp, const k3q_plan *plan) {
    if (fwrite("GGUF", 1, 4, fp) != 4) fail("write GGUF magic failed");
    write_u32(fp, K3Q_GGUF_VERSION);
    write_u64(fp, plan->count);
    write_u64(fp, K3Q_KV_COUNT);
    write_kv_string(fp, "general.architecture",
                    "kimi-k3", strlen("kimi-k3"));
    write_kv_string(fp, "general.name",
                    "Kimi K3 DwarfStar Q2",
                    strlen("Kimi K3 DwarfStar Q2"));
    write_kv_u32(fp, "general.alignment", K3Q_ALIGNMENT);
    write_kv_u32(fp, "general.file_type", 10);
    write_kv_u32(fp, "kimi-k3.block_count", K3_N_LAYER);
    write_kv_u32(fp, "kimi-k3.embedding_length", K3_N_EMBD);
    write_kv_u32(fp, "kimi-k3.context_length", K3_MAX_CONTEXT);
    write_kv_u32(fp, "kimi-k3.vocabulary_size", K3_N_VOCAB);
    write_kv_u32(fp, "kimi-k3.expert_count", K3_N_EXPERT);
    write_kv_u32(fp, "kimi-k3.expert_used_count", K3_N_EXPERT_USED);
    write_kv_u32(fp, "kimi-k3.routed_hidden_length", K3_ROUTED_HIDDEN);
    write_kv_u32(fp, "kimi-k3.expert_feed_forward_length", K3_EXPERT_FF);
    write_kv_bool(fp, "kimi-k3.vision", false);
    write_kv_string(fp, "tokenizer.kimi-k3.tiktoken",
                    plan->tokenizer, plan->tokenizer_size);

    for (uint32_t i = 0; i < plan->count; i++) {
        const k3q_tensor *tensor = &plan->tensors[i];
        write_string(fp, tensor->name);
        write_u32(fp, tensor->n_dims);
        for (uint32_t d = 0; d < tensor->n_dims; d++) {
            write_u64(fp, tensor->ne[d]);
        }
        write_u32(fp, (uint32_t)tensor->type);
        write_u64(fp, tensor->offset);
    }
    const off_t position = ftello(fp);
    if (position < 0 || (uint64_t)position > plan->data_offset) {
        fail("GGUF metadata size mismatch");
    }
    write_zeros(fp, plan->data_offset - (uint64_t)position);
}

static void write_header(const char *path, const k3q_plan *plan) {
    FILE *fp = fopen(path, "wb");
    if (!fp) fail_errno("create", path);
    write_header_contents(fp, plan);
    if (fflush(fp) != 0 ||
        ftruncate(fileno(fp), (off_t)plan->file_size) != 0) {
        fail_errno("flush", path);
    }
    sync_file_durable(fileno(fp), path);
    if (fclose(fp) != 0) {
        fail_errno("close", path);
    }
}

static int header_matches_plan(const char *path, const k3q_plan *plan) {
    FILE *actual = fopen(path, "rb");
    FILE *expected = tmpfile();
    if (!actual || !expected) {
        if (actual) fclose(actual);
        if (expected) fclose(expected);
        return 0;
    }
    write_header_contents(expected, plan);
    int ok =
        fflush(expected) == 0 &&
        fseeko(expected, 0, SEEK_SET) == 0;
    uint8_t *actual_bytes = xmalloc(K3Q_COPY_CHUNK);
    uint8_t *expected_bytes = xmalloc(K3Q_COPY_CHUNK);
    uint64_t remaining = plan->data_offset;
    while (ok && remaining != 0) {
        const size_t chunk =
            remaining < K3Q_COPY_CHUNK
                ? (size_t)remaining
                : K3Q_COPY_CHUNK;
        ok =
            fread(actual_bytes, 1, chunk, actual) == chunk &&
            fread(expected_bytes, 1, chunk, expected) == chunk &&
            memcmp(actual_bytes, expected_bytes, chunk) == 0;
        remaining -= chunk;
    }
    free(expected_bytes);
    free(actual_bytes);
    if (fclose(expected) != 0) ok = 0;
    if (fclose(actual) != 0) ok = 0;
    return ok;
}

static void bf16_to_f32(const uint8_t *source, float *target,
                        uint64_t elements) {
    for (uint64_t i = 0; i < elements; i++) {
        uint16_t bits;
        memcpy(&bits, source + 2u * i, sizeof(bits));
        target[i] = ds4q_bf16_to_f32(bits);
    }
}

static int copy_regular_tensor(int output_fd, uint64_t output_offset,
                               int source_fd,
                               const k3_st_tensor *source) {
    uint8_t *buffer = xmalloc(K3Q_COPY_CHUNK);
    uint64_t done = 0;
    while (done < source->size) {
        const size_t chunk =
            source->size - done < K3Q_COPY_CHUNK
                ? (size_t)(source->size - done)
                : K3Q_COPY_CHUNK;
        if (!pread_full(source_fd, buffer, chunk, source->offset + done) ||
            !pwrite_full(output_fd, buffer, chunk, output_offset + done)) {
            free(buffer);
            return 0;
        }
        done += chunk;
    }
    free(buffer);
    return 1;
}

static int quantize_regular_tensor(int output_fd, uint64_t output_offset,
                                   int source_fd,
                                   const k3_st_tensor *source,
                                   const k3q_tensor *target) {
    if (source->dtype != K3_ST_DTYPE_BF16 ||
        source->n_dims < 2 || target->n_dims != source->n_dims) {
        errno = EINVAL;
        return 0;
    }
    const uint64_t ncols = target->ne[0];
    uint64_t nrows = 1;
    for (uint32_t i = 1; i < target->n_dims; i++) nrows *= target->ne[i];
    if (ncols == 0 || ncols > SIZE_MAX / sizeof(float)) {
        errno = EINVAL;
        return 0;
    }
    uint64_t rows_per_chunk = K3Q_FLOAT_CHUNK /
        (ncols * sizeof(float));
    if (rows_per_chunk == 0) rows_per_chunk = 1;
    if (rows_per_chunk > nrows) rows_per_chunk = nrows;

    const size_t source_row = (size_t)ncols * sizeof(uint16_t);
    const size_t target_row =
        ds4q_row_size(target->type, (int64_t)ncols);
    uint8_t *source_bytes =
        xmalloc((size_t)rows_per_chunk * source_row);
    float *values =
        xmalloc((size_t)rows_per_chunk * (size_t)ncols * sizeof(float));
    uint8_t *quantized =
        xmalloc((size_t)rows_per_chunk * target_row);
    int ok = 1;
    for (uint64_t row = 0; row < nrows && ok; row += rows_per_chunk) {
        uint64_t rows = nrows - row;
        if (rows > rows_per_chunk) rows = rows_per_chunk;
        const size_t source_size = (size_t)rows * source_row;
        const size_t target_size = (size_t)rows * target_row;
        if (!pread_full(source_fd, source_bytes, source_size,
                        source->offset + row * source_row)) {
            ok = 0;
            break;
        }
        bf16_to_f32(source_bytes, values, rows * ncols);
        const size_t written = ds4q_quantize_chunk(
            target->type, values, quantized, 0, (int64_t)rows,
            (int64_t)ncols, NULL);
        if (written != target_size) {
            errno = EINVAL;
            ok = 0;
        } else if (!pwrite_full(
                       output_fd, quantized, target_size,
                       output_offset + row * target_row)) {
            ok = 0;
        }
    }
    free(quantized);
    free(values);
    free(source_bytes);
    return ok;
}

static float e8m0_to_f32(uint8_t exponent) {
    const uint32_t bits =
        exponent == 0 ? UINT32_C(0x00400000)
                      : (uint32_t)exponent << 23;
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static int read_dequant_expert_matrix(
        const k3_st_shard *shard,
        const k3_st_tensor *packed,
        const k3_st_tensor *scale,
        uint64_t out_dim,
        uint64_t in_dim,
        float **values_out) {
    if (!shard || !packed || !scale || !values_out ||
        packed->size != out_dim * in_dim / 2u ||
        scale->size != out_dim * in_dim / 32u) {
        errno = EINVAL;
        return 0;
    }
    uint8_t *packed_bytes = xmalloc((size_t)packed->size);
    uint8_t *scales = xmalloc((size_t)scale->size);
    float *values = xmalloc((size_t)(out_dim * in_dim) * sizeof(float));
    const int fd = k3_st_fd(shard);
    int ok = pread_full(fd, packed_bytes, (size_t)packed->size,
                        packed->offset) &&
             pread_full(fd, scales, (size_t)scale->size,
                        scale->offset);
    if (ok) {
        for (uint64_t i = 0; i < scale->size; i++) {
            if (scales[i] == UINT8_MAX) {
                errno = EILSEQ;
                ok = 0;
                break;
            }
        }
    }
    static const float fp4[16] = {
         0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
         0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
    };
    if (ok) {
        const uint64_t blocks_per_row = in_dim / 32u;
        for (uint64_t row = 0; row < out_dim; row++) {
            for (uint64_t block = 0; block < blocks_per_row; block++) {
                const float factor =
                    e8m0_to_f32(scales[row * blocks_per_row + block]);
                const uint64_t source =
                    (row * blocks_per_row + block) * 16u;
                const uint64_t target = row * in_dim + block * 32u;
                for (uint64_t i = 0; i < 16u; i++) {
                    const uint8_t byte = packed_bytes[source + i];
                    values[target + 2u * i] = fp4[byte & 15u] * factor;
                    values[target + 2u * i + 1u] =
                        fp4[byte >> 4] * factor;
                }
            }
        }
    }
    free(scales);
    free(packed_bytes);
    if (!ok) {
        free(values);
        return 0;
    }
    *values_out = values;
    return 1;
}

typedef struct {
    const k3_layer *layer;
    const k3q_tensor *target;
    const k3q_importance *importance;
    int output_fd;
    uint64_t output_base;
    uint64_t per_expert;
    uint32_t next_expert;
    uint32_t completed;
    uint32_t thread_count;
    int error_number;
    int failed;
    pthread_mutex_t lock;
} expert_batch;

static void expert_source(const k3_layer *layer,
                          k3q_source_kind kind,
                          uint32_t expert,
                          const k3_st_tensor **packed,
                          const k3_st_tensor **scale,
                          uint64_t *out_dim,
                          uint64_t *in_dim) {
    const k3_expert_weights *weights = &layer->experts[expert];
    if (kind == K3Q_SOURCE_EXPERT_GATE) {
        *packed = weights->w1_packed;
        *scale = weights->w1_scale;
        *out_dim = K3_EXPERT_FF;
        *in_dim = K3_ROUTED_HIDDEN;
    } else if (kind == K3Q_SOURCE_EXPERT_UP) {
        *packed = weights->w3_packed;
        *scale = weights->w3_scale;
        *out_dim = K3_EXPERT_FF;
        *in_dim = K3_ROUTED_HIDDEN;
    } else {
        *packed = weights->w2_packed;
        *scale = weights->w2_scale;
        *out_dim = K3_ROUTED_HIDDEN;
        *in_dim = K3_EXPERT_FF;
    }
}

static void normalize_importance(float *values, uint64_t count,
                                 float floor_ratio, float cap_ratio) {
    double mean = 0.0;
    for (uint64_t i = 0; i < count; i++) mean += values[i];
    mean /= count;
    if (!(mean > 0.0) || !isfinite(mean)) {
        for (uint64_t i = 0; i < count; i++) values[i] = 1.0f;
        return;
    }
    const float inverse_mean = (float)(1.0 / mean);
    for (uint64_t i = 0; i < count; i++) {
        float value = values[i] * inverse_mean;
        if (!isfinite(value) || value < floor_ratio) value = floor_ratio;
        if (value > cap_ratio) value = cap_ratio;
        values[i] = value;
    }
}

typedef struct {
    const k3_layer *layer;
    float *down_input;
    double *input_partials;
    uint32_t next_expert;
    int failed;
    pthread_mutex_t lock;
} importance_batch;

typedef struct {
    importance_batch *batch;
    uint32_t thread_index;
} importance_worker_args;

static int collect_one_expert_importance(importance_worker_args *args,
                                         uint32_t expert) {
    importance_batch *batch = args->batch;
    const k3_st_tensor *gate_packed = NULL;
    const k3_st_tensor *gate_scale = NULL;
    const k3_st_tensor *up_packed = NULL;
    const k3_st_tensor *up_scale = NULL;
    uint64_t gate_rows = 0;
    uint64_t gate_columns = 0;
    uint64_t up_rows = 0;
    uint64_t up_columns = 0;
    expert_source(batch->layer, K3Q_SOURCE_EXPERT_GATE, expert,
                  &gate_packed, &gate_scale,
                  &gate_rows, &gate_columns);
    expert_source(batch->layer, K3Q_SOURCE_EXPERT_UP, expert,
                  &up_packed, &up_scale, &up_rows, &up_columns);
    if (gate_rows != K3_EXPERT_FF ||
        gate_columns != K3_ROUTED_HIDDEN ||
        up_rows != gate_rows || up_columns != gate_columns) {
        return 0;
    }

    float *gate = NULL;
    if (!read_dequant_expert_matrix(batch->layer->shard,
                                    gate_packed, gate_scale,
                                    gate_rows, gate_columns, &gate)) {
        return 0;
    }
    double *partial = batch->input_partials +
        (size_t)args->thread_index * K3_ROUTED_HIDDEN;
    double *gate_energy = xcalloc(K3_EXPERT_FF, sizeof(double));
    for (uint64_t row = 0; row < gate_rows; row++) {
        const float *source = gate + row * gate_columns;
        double row_energy = 0.0;
        for (uint64_t column = 0; column < gate_columns; column++) {
            const double square = (double)source[column] * source[column];
            partial[column] += square;
            row_energy += square;
        }
        gate_energy[row] = row_energy;
    }
    free(gate);

    float *up = NULL;
    if (!read_dequant_expert_matrix(batch->layer->shard,
                                    up_packed, up_scale,
                                    up_rows, up_columns, &up)) {
        free(gate_energy);
        return 0;
    }
    float *down = batch->down_input +
        (size_t)expert * K3_EXPERT_FF;
    for (uint64_t row = 0; row < up_rows; row++) {
        const float *source = up + row * up_columns;
        double row_energy = 0.0;
        for (uint64_t column = 0; column < up_columns; column++) {
            const double square = (double)source[column] * source[column];
            partial[column] += square;
            row_energy += square;
        }
        /*
         * The down matrix consumes SiTU(gate, up). With no calibration
         * activations available, the product of the paired projection
         * variances is the closest weight-only second-moment proxy.
         */
        down[row] = (float)(gate_energy[row] * row_energy);
    }
    free(up);
    free(gate_energy);
    normalize_importance(down, K3_EXPERT_FF, 0.02f, 50.0f);
    return 1;
}

static void *importance_worker(void *opaque) {
    importance_worker_args *args = opaque;
    importance_batch *batch = args->batch;
    for (;;) {
        pthread_mutex_lock(&batch->lock);
        const uint32_t expert = batch->next_expert++;
        const int stop = expert >= K3_N_EXPERT || batch->failed;
        pthread_mutex_unlock(&batch->lock);
        if (stop) break;
        if (!collect_one_expert_importance(args, expert)) {
            pthread_mutex_lock(&batch->lock);
            batch->failed = 1;
            pthread_mutex_unlock(&batch->lock);
            break;
        }
        if ((expert + 1u) % 64u == 0u) {
            fprintf(stderr,
                    "k3-quantize: layer %u importance %u/%u experts\r",
                    batch->layer->index, expert + 1u,
                    (uint32_t)K3_N_EXPERT);
        }
    }
    return NULL;
}

static void importance_close(k3q_importance *importance) {
    if (!importance) return;
    free(importance->expert_input);
    free(importance->down_input);
    memset(importance, 0, sizeof(*importance));
    importance->layer = UINT32_MAX;
}

static int importance_prepare(k3q_importance *importance,
                              const k3_layer *layer,
                              uint32_t thread_count) {
    importance_close(importance);
    importance->layer = layer->index;
    importance->expert_input =
        xcalloc(K3_ROUTED_HIDDEN, sizeof(float));
    importance->down_input =
        xcalloc((size_t)K3_N_EXPERT * K3_EXPERT_FF, sizeof(float));
    double *partials = xcalloc(
        (size_t)thread_count * K3_ROUTED_HIDDEN, sizeof(double));
    pthread_t *threads = xcalloc(thread_count, sizeof(threads[0]));
    importance_worker_args *args =
        xcalloc(thread_count, sizeof(args[0]));
    importance_batch batch = {
        .layer = layer,
        .down_input = importance->down_input,
        .input_partials = partials,
    };
    pthread_mutex_init(&batch.lock, NULL);
    int failed = 0;
    for (uint32_t i = 0; i < thread_count; i++) {
        args[i] = (importance_worker_args){
            .batch = &batch,
            .thread_index = i,
        };
    }
    for (uint32_t i = 1; i < thread_count; i++) {
        if (pthread_create(&threads[i], NULL,
                           importance_worker, &args[i]) != 0) {
            failed = 1;
            batch.failed = 1;
            thread_count = i;
            break;
        }
    }
    importance_worker(&args[0]);
    for (uint32_t i = 1; i < thread_count; i++) {
        pthread_join(threads[i], NULL);
    }
    failed |= batch.failed;
    fprintf(stderr, "\n");
    if (!failed) {
        for (uint32_t column = 0; column < K3_ROUTED_HIDDEN; column++) {
            double energy = 0.0;
            for (uint32_t thread = 0; thread < thread_count; thread++) {
                energy += partials[
                    (size_t)thread * K3_ROUTED_HIDDEN + column];
            }
            /*
             * Temper the weak weight-energy proxy. Gate and up consume the
             * same latent vector, so both receive the same importance.
             */
            importance->expert_input[column] = (float)sqrt(energy);
        }
        normalize_importance(
            importance->expert_input,
            K3_ROUTED_HIDDEN, 0.02f, 50.0f);
    }
    pthread_mutex_destroy(&batch.lock);
    free(args);
    free(threads);
    free(partials);
    if (failed) importance_close(importance);
    return !failed;
}

static int quantize_one_expert(expert_batch *batch, uint32_t expert) {
    const k3_st_tensor *packed = NULL;
    const k3_st_tensor *scale = NULL;
    uint64_t rows = 0;
    uint64_t columns = 0;
    expert_source(batch->layer, batch->target->source_kind, expert,
                  &packed, &scale, &rows, &columns);
    float *values = NULL;
    if (!read_dequant_expert_matrix(batch->layer->shard,
                                    packed, scale, rows, columns,
                                    &values)) {
        return 0;
    }
    const float *importance =
        batch->target->source_kind == K3Q_SOURCE_EXPERT_DOWN
            ? batch->importance->down_input +
                (size_t)expert * K3_EXPERT_FF
            : batch->importance->expert_input;
    uint8_t *quantized = xmalloc((size_t)batch->per_expert);
    const size_t written = ds4q_quantize_chunk(
        batch->target->type, values, quantized, 0,
        (int64_t)rows, (int64_t)columns, importance);
    int ok = written == batch->per_expert;
    if (!ok) {
        errno = EINVAL;
    } else {
        ok = pwrite_full(
            batch->output_fd, quantized, written,
            batch->output_base + (uint64_t)expert * batch->per_expert);
    }
    free(quantized);
    free(values);
    return ok;
}

static void *expert_worker(void *argument) {
    expert_batch *batch = argument;
    for (;;) {
        pthread_mutex_lock(&batch->lock);
        const uint32_t expert = batch->next_expert++;
        const int stop = expert >= K3_N_EXPERT || batch->failed;
        pthread_mutex_unlock(&batch->lock);
        if (stop) break;
        if (!quantize_one_expert(batch, expert)) {
            pthread_mutex_lock(&batch->lock);
            batch->failed = 1;
            batch->error_number = errno ? errno : EIO;
            pthread_mutex_unlock(&batch->lock);
            break;
        }
        pthread_mutex_lock(&batch->lock);
        const uint32_t completed = ++batch->completed;
        if ((completed % 16u) == 0u || completed == K3_N_EXPERT) {
            fprintf(stderr,
                    "k3-quantize: layer %u %s %u/%u experts\r",
                    batch->target->layer,
                    ds4q_type_name(batch->target->type),
                    completed, (uint32_t)K3_N_EXPERT);
        }
        pthread_mutex_unlock(&batch->lock);
    }
    return NULL;
}

static int quantize_expert_stack(int output_fd, uint64_t output_offset,
                                 const k3_layer *layer,
                                 const k3q_tensor *target,
                                 const k3q_importance *importance,
                                 uint32_t thread_count) {
    const uint64_t per_expert = target->size / K3_N_EXPERT;
    expert_batch batch = {
        .layer = layer,
        .target = target,
        .importance = importance,
        .output_fd = output_fd,
        .output_base = output_offset,
        .per_expert = per_expert,
        .thread_count = thread_count,
    };
    pthread_mutex_init(&batch.lock, NULL);
    /* Publish the IQ2 tables before workers read the lock-free fast path. */
    ds4q_quantize_init(target->type);
    pthread_t *threads =
        xcalloc(thread_count, sizeof(threads[0]));
    for (uint32_t i = 1; i < thread_count; i++) {
        if (pthread_create(&threads[i], NULL,
                           expert_worker, &batch) != 0) {
            batch.failed = 1;
            thread_count = i;
            break;
        }
    }
    expert_worker(&batch);
    for (uint32_t i = 1; i < thread_count; i++) {
        pthread_join(threads[i], NULL);
    }
    fprintf(stderr, "\n");
    free(threads);
    pthread_mutex_destroy(&batch.lock);
    if (batch.failed) {
        errno = batch.error_number ? batch.error_number : EIO;
    }
    return !batch.failed && batch.completed == K3_N_EXPERT;
}

static int read_progress(const char *path, uint32_t *next_out) {
    FILE *fp = fopen(path, "rb");
    if (!fp) return 0;
    char magic[8] = {0};
    uint32_t next = 0;
    int ok =
        fread(magic, 1, 7, fp) == 7 &&
        memcmp(magic, "K3QPRO1", 7) == 0 &&
        fread(&next, sizeof(next), 1, fp) == 1 &&
        fgetc(fp) == EOF &&
        !ferror(fp);
    if (fclose(fp) != 0) ok = 0;
    if (ok && next_out) *next_out = next;
    return ok;
}

static void sync_parent_directory(const char *path) {
    const char *slash = strrchr(path, '/');
    char *directory = NULL;
    if (!slash) {
        directory = xstrdup(".");
    } else if (slash == path) {
        directory = xstrdup("/");
    } else {
        const size_t length = (size_t)(slash - path);
        directory = xmalloc(length + 1u);
        memcpy(directory, path, length);
        directory[length] = '\0';
    }
    const int fd = open(directory, O_RDONLY);
    if (fd < 0 || fsync(fd) != 0) {
        const int saved = errno;
        if (fd >= 0) close(fd);
        errno = saved;
        fail_errno("sync directory", directory);
    }
    if (close(fd) != 0) fail_errno("close directory", directory);
    free(directory);
}

static void write_progress(const char *path, uint32_t next) {
    char *temporary = append_suffix(path, ".tmp");
    FILE *fp = fopen(temporary, "wb");
    if (!fp) fail_errno("create progress", temporary);
    if (fwrite("K3QPRO1", 1, 7, fp) != 7 ||
        fwrite(&next, sizeof(next), 1, fp) != 1 ||
        fflush(fp) != 0) {
        fail_errno("write progress", temporary);
    }
    sync_file_durable(fileno(fp), temporary);
    if (fclose(fp) != 0) fail_errno("close progress", temporary);
    if (rename(temporary, path) != 0) {
        fail_errno("rename progress", path);
    }
    sync_parent_directory(path);
    free(temporary);
}

static k3_layer *open_layer_checked(const k3_model *model,
                                    uint32_t layer_index) {
    char error[512] = {0};
    k3_layer *layer = NULL;
    if (!k3_model_open_layer(model, layer_index, &layer,
                             error, sizeof(error))) {
        fprintf(stderr, "k3-quantize: layer %u: %s\n",
                layer_index, error[0] ? error : "open failed");
        exit(1);
    }
    k3_st_unmap(layer->shard);
#if defined(F_NOCACHE)
    (void)fcntl(k3_st_fd(layer->shard), F_NOCACHE, 1);
#endif
    return layer;
}

static int write_one_tensor(int output_fd, const k3q_plan *plan,
                            const k3q_tensor *target,
                            const k3_layer *layer,
                            const k3_global_weights *globals,
                            const k3q_importance *importance,
                            uint32_t threads) {
    const uint64_t output_offset =
        plan->data_offset + target->offset;
    if (target->source_kind != K3Q_SOURCE_REGULAR) {
        return quantize_expert_stack(output_fd, output_offset,
                                     layer, target, importance, threads);
    }
    const k3_st_shard *shard =
        target->layer == UINT32_MAX ? globals->shard : layer->shard;
    const k3_st_tensor *source =
        k3_st_find(shard, target->source_name);
    if (!source) {
        errno = ENOENT;
        return 0;
    }
    const ds4q_type source_type = source_gguf_type(source->dtype);
    if (target->type == source_type) {
        if (target->size != source->size) {
            errno = EINVAL;
            return 0;
        }
        return copy_regular_tensor(output_fd, output_offset,
                                   k3_st_fd(shard), source);
    }
    return quantize_regular_tensor(output_fd, output_offset,
                                   k3_st_fd(shard), source, target);
}

static void convert_model(const k3q_options *options,
                          const k3q_plan *plan) {
    char *partial = append_suffix(options->output, ".partial");
    char *progress = append_suffix(options->output, ".progress");
    char *progress_tmp = append_suffix(progress, ".tmp");
    char *lock_path = append_suffix(options->output, ".lock");
    const int lock_fd = open(lock_path, O_RDWR | O_CREAT, 0600);
    if (lock_fd < 0) fail_errno("open lock", lock_path);
    if (flock(lock_fd, LOCK_EX | LOCK_NB) != 0) {
        fail_errno("lock output", lock_path);
    }
    struct stat st;
    uint32_t first = 0;
    if (options->resume) {
        if (stat(partial, &st) != 0 ||
            (uint64_t)st.st_size != plan->file_size) {
            fail("--resume needs a matching .partial file");
        }
        if (!header_matches_plan(partial, plan)) {
            fail("--resume partial header does not match this model and plan");
        }
        if (!read_progress(progress, &first)) {
            fail("--resume needs a valid .progress journal");
        }
        if (first > plan->count) fail("invalid progress journal");
    } else {
        if (!options->overwrite &&
            (stat(options->output, &st) == 0 ||
             stat(partial, &st) == 0 ||
             stat(progress, &st) == 0 ||
             stat(progress_tmp, &st) == 0)) {
            fail("output exists; use --overwrite or --resume");
        }
        if (options->overwrite) {
            unlink_if_exists(options->output);
            unlink_if_exists(partial);
            unlink_if_exists(progress);
            unlink_if_exists(progress_tmp);
        }
        /*
         * Reset the journal first. If header creation is interrupted, resume
         * rejects the absent/short partial instead of trusting stale progress.
         */
        write_progress(progress, 0);
        write_header(partial, plan);
    }

    int output_fd = open(partial, O_RDWR);
    if (output_fd < 0) fail_errno("open", partial);
#if defined(F_NOCACHE)
    (void)fcntl(output_fd, F_NOCACHE, 1);
#endif
    char error[512] = {0};
    k3_model *model = NULL;
    if (!k3_model_open(&model, options->model_dir,
                       error, sizeof(error))) {
        fail(error[0] ? error : "could not reopen model");
    }
    k3_global_weights globals = {0};
    k3_layer *layer = NULL;
    k3q_importance importance = {.layer = UINT32_MAX};
    uint32_t open_layer = UINT32_MAX;

    for (uint32_t i = first; i < plan->count; i++) {
        const k3q_tensor *target = &plan->tensors[i];
        if (target->layer != UINT32_MAX &&
            target->layer != open_layer) {
            importance_close(&importance);
            k3_layer_close(layer);
            layer = open_layer_checked(model, target->layer);
            open_layer = target->layer;
        } else if (target->layer == UINT32_MAX && !globals.shard) {
            if (!k3_model_open_globals(model, &globals,
                                       error, sizeof(error))) {
                fail(error[0] ? error : "could not open globals");
            }
            k3_st_unmap(globals.shard);
#if defined(F_NOCACHE)
            (void)fcntl(k3_st_fd(globals.shard), F_NOCACHE, 1);
#endif
        }
        fprintf(stderr,
                "k3-quantize: [%u/%u] %s -> %s (%.3f GiB)\n",
                i + 1u, plan->count, target->name,
                ds4q_type_name(target->type),
                (double)target->size / 1073741824.0);
        if (target->source_kind != K3Q_SOURCE_REGULAR &&
            importance.layer != target->layer &&
            !importance_prepare(
                &importance, layer, options->threads)) {
            fail("could not prepare routed-expert importance");
        }
        if (!write_one_tensor(output_fd, plan, target,
                              layer, &globals, &importance,
                              options->threads)) {
            fprintf(stderr, "k3-quantize: failed writing %s: %s\n",
                    target->name, strerror(errno));
            exit(1);
        }
        sync_file_durable(output_fd, partial);
        write_progress(progress, i + 1u);
    }

    importance_close(&importance);
    k3_layer_close(layer);
    k3_global_weights_close(&globals);
    k3_model_close(model);
    sync_file_durable(output_fd, partial);
    if (fstat(output_fd, &st) != 0) fail_errno("stat", partial);
    if (st.st_size < 0 || (uint64_t)st.st_size != plan->file_size) {
        fail("partial GGUF has the wrong final size");
    }
    if (close(output_fd) != 0) fail_errno("close", partial);
    if (rename(partial, options->output) != 0) {
        fail_errno("rename output", options->output);
    }
    sync_parent_directory(options->output);
    unlink_if_exists(progress);
    sync_parent_directory(options->output);
    if (close(lock_fd) != 0) fail_errno("close lock", lock_path);
    free(lock_path);
    free(progress_tmp);
    free(progress);
    free(partial);
}

static void usage(const char *program) {
    printf("usage: %s --model DIR --out FILE [options]\n", program);
    printf("  --threads N   routed expert quantization workers (default 8)\n");
    printf("  --dry-run     build and print the exact GGUF plan only\n");
    printf("  --resume      continue FILE.partial using FILE.progress\n");
    printf("  --overwrite   replace an existing output/partial conversion\n");
}

static const char *next_arg(int argc, char **argv, int *index,
                            const char *option) {
    if (++*index >= argc) {
        fprintf(stderr, "k3-quantize: missing value for %s\n", option);
        exit(1);
    }
    return argv[*index];
}

static k3q_options parse_options(int argc, char **argv) {
    k3q_options options = {.threads = 8};
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--model") || !strcmp(argv[i], "--hf")) {
            options.model_dir = next_arg(argc, argv, &i, argv[i]);
        } else if (!strcmp(argv[i], "--out")) {
            options.output = next_arg(argc, argv, &i, argv[i]);
        } else if (!strcmp(argv[i], "--threads")) {
            const char *value = next_arg(argc, argv, &i, argv[i]);
            char *end = NULL;
            const unsigned long parsed = strtoul(value, &end, 10);
            if (!value[0] || !end || *end || parsed < 1 || parsed > 32) {
                fail("--threads must be 1..32");
            }
            options.threads = (uint32_t)parsed;
        } else if (!strcmp(argv[i], "--dry-run")) {
            options.dry_run = true;
        } else if (!strcmp(argv[i], "--resume")) {
            options.resume = true;
        } else if (!strcmp(argv[i], "--overwrite")) {
            options.overwrite = true;
        } else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
            usage(argv[0]);
            exit(0);
        } else {
            fprintf(stderr, "k3-quantize: unknown option %s\n", argv[i]);
            exit(1);
        }
    }
    if (!options.model_dir) fail("--model is required");
    if (!options.dry_run && !options.output) fail("--out is required");
    if (options.resume && options.overwrite) {
        fail("--resume and --overwrite are mutually exclusive");
    }
    return options;
}

int main(int argc, char **argv) {
    const k3q_options options = parse_options(argc, argv);
    k3q_plan plan = build_plan(options.model_dir);
    print_plan(&plan);
    if (!options.dry_run) {
        convert_model(&options, &plan);
        fprintf(stderr, "k3-quantize: wrote %s\n", options.output);
    }
    free_plan(&plan);
    return 0;
}
