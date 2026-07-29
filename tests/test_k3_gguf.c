#include "k3_model.h"

#include <assert.h>
#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

enum {
    COMPARE_CHUNK = 16 * 1024 * 1024,
};

typedef struct {
    uint64_t retained_tensors;
    uint64_t retained_bytes;
    uint64_t transformed_tensors;
    uint64_t omitted_expert_tensors;
} compare_stats;

static int compare_tensor_offset(const void *lhs, const void *rhs) {
    const k3_st_tensor *a = *(const k3_st_tensor *const *)lhs;
    const k3_st_tensor *b = *(const k3_st_tensor *const *)rhs;
    if (a->offset < b->offset) return -1;
    if (a->offset > b->offset) return 1;
    return strcmp(a->name, b->name);
}

static int compare_tensor_name(const void *lhs, const void *rhs) {
    const k3_st_tensor *a = *(const k3_st_tensor *const *)lhs;
    const k3_st_tensor *b = *(const k3_st_tensor *const *)rhs;
    return strcmp(a->name, b->name);
}

static int validate_gguf_directory(const k3_st_shard *gguf,
                                   char *error,
                                   size_t error_cap) {
    const uint32_t count = k3_st_tensor_count(gguf);
    if (count != 2736u) {
        snprintf(error, error_cap,
                 "GGUF has %u tensors, expected 2736", count);
        return 0;
    }
    const k3_st_tensor **tensors =
        malloc((size_t)count * sizeof(tensors[0]));
    if (!tensors) {
        snprintf(error, error_cap,
                 "out of memory validating GGUF tensor directory");
        return 0;
    }
    for (uint32_t i = 0; i < count; i++) {
        tensors[i] = k3_st_tensor_at(gguf, i);
    }
    qsort(tensors, count, sizeof(tensors[0]), compare_tensor_name);
    for (uint32_t i = 1; i < count; i++) {
        if (strcmp(tensors[i - 1u]->name, tensors[i]->name) == 0) {
            snprintf(error, error_cap,
                     "duplicate GGUF tensor name %s", tensors[i]->name);
            free(tensors);
            return 0;
        }
    }
    qsort(tensors, count, sizeof(tensors[0]), compare_tensor_offset);
    uint64_t end = k3_st_data_offset(gguf);
    for (uint32_t i = 0; i < count; i++) {
        const k3_st_tensor *tensor = tensors[i];
        if ((tensor->offset & 31u) != 0u || tensor->offset < end) {
            snprintf(error, error_cap,
                     "unaligned or overlapping GGUF tensor %s",
                     tensor->name);
            free(tensors);
            return 0;
        }
        end = tensor->offset + tensor->size;
    }
    if (end != k3_st_file_size(gguf)) {
        snprintf(error, error_cap,
                 "GGUF payload ends at %" PRIu64
                 ", file size is %" PRIu64,
                 end, k3_st_file_size(gguf));
        free(tensors);
        return 0;
    }
    free(tensors);
    return 1;
}

static int pread_full(int fd, void *buffer, size_t bytes, uint64_t offset) {
    if (offset > (uint64_t)LLONG_MAX ||
        bytes > (uint64_t)LLONG_MAX - offset) {
        errno = EOVERFLOW;
        return 0;
    }
    size_t done = 0;
    while (done < bytes) {
        ssize_t n;
        do {
            n = pread(fd,
                      (uint8_t *)buffer + done,
                      bytes - done,
                      (off_t)(offset + done));
        } while (n < 0 && errno == EINTR);
        if (n <= 0) return 0;
        done += (size_t)n;
    }
    return 1;
}

static int compare_tensor_bytes(const k3_st_shard *source_shard,
                                const k3_st_tensor *source,
                                const k3_st_shard *gguf,
                                const k3_st_tensor *target,
                                uint8_t *source_bytes,
                                uint8_t *target_bytes) {
    uint64_t done = 0;
    while (done < source->size) {
        const size_t chunk =
            source->size - done < COMPARE_CHUNK
                ? (size_t)(source->size - done)
                : COMPARE_CHUNK;
        if (!pread_full(k3_st_fd(source_shard), source_bytes, chunk,
                        source->offset + done) ||
            !pread_full(k3_st_fd(gguf), target_bytes, chunk,
                        target->offset + done)) {
            fprintf(stderr, "read failed comparing %s: %s\n",
                    source->name, strerror(errno));
            return 0;
        }
        if (memcmp(source_bytes, target_bytes, chunk) != 0) {
            fprintf(stderr,
                    "retained tensor differs from SafeTensor source: %s "
                    "at byte %" PRIu64 "\n",
                    source->name, done);
            return 0;
        }
        done += chunk;
    }
    return 1;
}

static int compare_source_shard(const k3_st_shard *source_shard,
                                const k3_st_shard *gguf,
                                uint8_t *source_bytes,
                                uint8_t *target_bytes,
                                compare_stats *stats) {
    for (uint32_t i = 0; i < k3_st_tensor_count(source_shard); i++) {
        const k3_st_tensor *source =
            k3_st_tensor_at(source_shard, i);
        const k3_st_tensor *target =
            k3_st_find(gguf, source->name);
        if (!target) {
            if (strstr(source->name,
                       ".block_sparse_moe.experts.") == NULL) {
                fprintf(stderr, "GGUF omitted retained tensor %s\n",
                        source->name);
                return 0;
            }
            stats->omitted_expert_tensors++;
            continue;
        }
        if (target->n_dims != source->n_dims) {
            fprintf(stderr, "GGUF tensor rank differs from source: %s\n",
                    source->name);
            return 0;
        }
        for (uint32_t dim = 0; dim < source->n_dims; dim++) {
            if (target->shape[dim] != source->shape[dim]) {
                fprintf(stderr,
                        "GGUF tensor shape differs from source: %s\n",
                        source->name);
                return 0;
            }
        }
        if (target->dtype != source->dtype ||
            target->size != source->size) {
            stats->transformed_tensors++;
            continue;
        }
        if (!compare_tensor_bytes(
                source_shard, source, gguf, target,
                source_bytes, target_bytes)) {
            return 0;
        }
        stats->retained_tensors++;
        stats->retained_bytes += source->size;
    }
    return 1;
}

static int compare_retained_tensors(const char *gguf_path,
                                    const char *source_path,
                                    char *error,
                                    size_t error_cap) {
    k3_st_shard *gguf = NULL;
    k3_model *source_model = NULL;
    k3_global_weights globals = {0};
    uint8_t *source_bytes = NULL;
    uint8_t *target_bytes = NULL;
    compare_stats stats = {0};
    int ok = 0;
    if (!k3_st_open(&gguf, gguf_path, error, error_cap) ||
        !k3_st_is_gguf(gguf) ||
        !k3_model_open(
            &source_model, source_path, error, error_cap)) {
        goto done;
    }
    source_bytes = malloc(COMPARE_CHUNK);
    target_bytes = malloc(COMPARE_CHUNK);
    if (!source_bytes || !target_bytes) {
        snprintf(error, error_cap,
                 "out of memory comparing retained tensors");
        goto done;
    }
    for (uint32_t i = 0; i < K3_N_LAYER; i++) {
        k3_layer *layer = NULL;
        if (!k3_model_open_layer(
                source_model, i, &layer, error, error_cap)) {
            goto done;
        }
        k3_st_unmap(layer->shard);
        const int layer_ok = compare_source_shard(
            layer->shard, gguf, source_bytes, target_bytes, &stats);
        k3_layer_close(layer);
        if (!layer_ok) goto done;
    }
    if (!k3_model_open_globals(
            source_model, &globals, error, error_cap)) {
        goto done;
    }
    k3_st_unmap(globals.shard);
    if (!compare_source_shard(
            globals.shard, gguf, source_bytes, target_bytes, &stats)) {
        goto done;
    }
    const uint64_t expected_omitted =
        (uint64_t)(K3_N_LAYER - 1u) * K3_N_EXPERT * 6u;
    if (stats.retained_tensors != 2090u ||
        stats.retained_bytes != UINT64_C(83503490048) ||
        stats.transformed_tensors != 370u ||
        stats.omitted_expert_tensors != expected_omitted) {
        snprintf(error, error_cap,
                 "unexpected retained/quantized tensor accounting");
        goto done;
    }
    printf("test_k3_gguf: retained source bytes exact "
           "(%" PRIu64 " tensors, %.3f GiB; "
           "%" PRIu64 " transformed tensors, "
           "%" PRIu64 " routed source tensors replaced)\n",
           stats.retained_tensors,
           (double)stats.retained_bytes / 1073741824.0,
           stats.transformed_tensors,
           stats.omitted_expert_tensors);
    ok = 1;

done:
    free(target_bytes);
    free(source_bytes);
    k3_global_weights_close(&globals);
    k3_model_close(source_model);
    k3_st_close(gguf);
    return ok;
}

static int tensor_in_static_span(
        const k3_layer *layer,
        const k3_st_tensor *tensor) {
    if (!layer || !tensor || tensor->offset < layer->static_offset) return 0;
    const uint64_t relative = tensor->offset - layer->static_offset;
    return relative <= layer->static_size &&
           tensor->size <= layer->static_size - relative;
}

int main(int argc, char **argv) {
    if (argc < 2 || argc > 3 || !argv[1][0]) {
        fprintf(stderr,
                "usage: %s KIMI-K3.gguf [SOURCE-SAFETENSORS]\n",
                argv[0]);
        return 2;
    }
    char error[512] = {0};
    k3_model *model = NULL;
    k3_layer *dense = NULL;
    k3_layer *moe = NULL;
    k3_layer *last = NULL;
    k3_st_shard *gguf = NULL;
    k3_global_weights globals = {0};
    assert(k3_model_open(&model, argv[1], error, sizeof(error)));
    assert(model->gguf);
    if (!k3_st_open(&gguf, argv[1], error, sizeof(error)) ||
        !validate_gguf_directory(gguf, error, sizeof(error))) {
        fprintf(stderr, "test_k3_gguf: %s\n", error);
        k3_st_close(gguf);
        k3_model_close(model);
        return 1;
    }
    k3_st_close(gguf);

    assert(k3_model_open_layer(model, 0, &dense, error, sizeof(error)));
    assert(k3_st_is_gguf(dense->shard));
    assert(dense->dense_gate->dtype == K3_ST_DTYPE_Q8_0);
    assert(dense->dense_up->dtype == K3_ST_DTYPE_Q8_0);
    assert(dense->dense_down->dtype == K3_ST_DTYPE_BF16);
    assert(dense->static_size == dense->layer_size);
    assert(tensor_in_static_span(dense, dense->dense_gate));
    assert(tensor_in_static_span(dense, dense->dense_up));
    assert(tensor_in_static_span(dense, dense->dense_down));

    assert(k3_model_open_layer(model, 1, &moe, error, sizeof(error)));
    assert(moe->routed_down->dtype == K3_ST_DTYPE_BF16);
    assert(moe->routed_up->dtype == K3_ST_DTYPE_Q8_0);
    assert(moe->shared_gate->dtype == K3_ST_DTYPE_MXFP4);
    assert(moe->shared_up->dtype == K3_ST_DTYPE_MXFP4);
    assert(moe->shared_down->dtype == K3_ST_DTYPE_MXFP4);
    assert(moe->expert_gate->dtype == K3_ST_DTYPE_IQ2_XXS);
    assert(moe->expert_up->dtype == K3_ST_DTYPE_IQ2_XXS);
    assert(moe->expert_down->dtype == K3_ST_DTYPE_Q2_K);
    assert(moe->layer_size > moe->static_size);

    assert(k3_model_open_layer(model, K3_N_LAYER - 1u,
                               &last, error, sizeof(error)));
    assert(last->attention == K3_LAYER_MLA);
    assert(last->expert_down->shape[0] == K3_N_EXPERT);

    assert(k3_model_open_globals(model, &globals, error, sizeof(error)));
    assert(globals.lm_head->dtype == K3_ST_DTYPE_BF16);
    assert(globals.token_embedding->dtype == K3_ST_DTYPE_BF16);

    k3_global_weights_close(&globals);
    k3_layer_close(last);
    k3_layer_close(moe);
    k3_layer_close(dense);
    k3_model_close(model);
    if (argc == 3 &&
        !compare_retained_tensors(
            argv[1], argv[2], error, sizeof(error))) {
        fprintf(stderr, "test_k3_gguf: %s\n",
                error[0] ? error : "retained tensor comparison failed");
        return 1;
    }
    puts("test_k3_gguf: ok");
    return 0;
}
