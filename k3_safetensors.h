#ifndef K3_SAFETENSORS_H
#define K3_SAFETENSORS_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define K3_ST_MAX_DIMS 8

typedef enum {
    K3_ST_DTYPE_INVALID = 0,
    K3_ST_DTYPE_U8,
    K3_ST_DTYPE_F32,
    K3_ST_DTYPE_F16,
    K3_ST_DTYPE_BF16,
    K3_ST_DTYPE_Q8_0,
    K3_ST_DTYPE_Q2_K,
    K3_ST_DTYPE_IQ2_XXS,
    K3_ST_DTYPE_MXFP4,
} k3_st_dtype;

typedef struct {
    const char *name;
    k3_st_dtype dtype;
    uint32_t n_dims;
    uint64_t shape[K3_ST_MAX_DIMS];
    uint64_t offset;
    uint64_t size;
} k3_st_tensor;

typedef struct k3_st_shard k3_st_shard;

/*
 * Parse one SafeTensors shard. Tensor offsets returned by k3_st_find() are
 * absolute file offsets, so callers can pass them directly to pread or use
 * them relative to k3_st_map().
 */
int k3_st_open(k3_st_shard **out, const char *path, char *err, size_t errcap);
void k3_st_close(k3_st_shard *shard);

const char *k3_st_path(const k3_st_shard *shard);
int k3_st_fd(const k3_st_shard *shard);
bool k3_st_is_gguf(const k3_st_shard *shard);
uint64_t k3_st_file_size(const k3_st_shard *shard);
uint64_t k3_st_data_offset(const k3_st_shard *shard);
uint32_t k3_st_tensor_count(const k3_st_shard *shard);
const k3_st_tensor *k3_st_tensor_at(const k3_st_shard *shard, uint32_t index);
const k3_st_tensor *k3_st_find(const k3_st_shard *shard, const char *name);

int k3_st_map(k3_st_shard *shard, char *err, size_t errcap);
void k3_st_unmap(k3_st_shard *shard);
const void *k3_st_map_base(const k3_st_shard *shard);

/*
 * Bring a file range into the page cache. touch=true faults one byte from
 * every VM page and is deterministic; touch=false only submits asynchronous
 * OS readahead hints. The range must be inside the shard.
 */
int k3_st_prefetch(k3_st_shard *shard,
                   uint64_t offset,
                   uint64_t size,
                   bool touch,
                   char *err,
                   size_t errcap);

uint64_t k3_st_nelements(const k3_st_tensor *tensor);
uint32_t k3_st_dtype_size(k3_st_dtype dtype);
const char *k3_st_dtype_name(k3_st_dtype dtype);

int k3_st_gguf_read_string(const char *path,
                           const char *key,
                           uint8_t **value,
                           size_t *value_size,
                           char *err,
                           size_t errcap);

#endif
