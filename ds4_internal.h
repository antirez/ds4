#ifndef DS4_INTERNAL_H
#define DS4_INTERNAL_H

#include "ds4.h"

#define DS4_INTERNAL_MAX_LAYER 61

typedef struct {
    uint32_t n_layers;
    uint32_t first_layer;
    uint32_t last_layer;
    bool has_layers;
    bool has_output_head;
    bool has_token_embedding;
    uint64_t total_cache_bytes;
    uint64_t range_cache_bytes[DS4_INTERNAL_MAX_LAYER][DS4_INTERNAL_MAX_LAYER];
} ds4_internal_model_shard_info;

int ds4_internal_model_shard_info_from_file(
        const char *model_path,
        ds4_internal_model_shard_info *out);
bool ds4_internal_session_cancel_requested(ds4_session *s);

void ds4_internal_dist_set_adopt_listen_fd(
        ds4_distributed_options *opt,
        int fd);
void ds4_internal_dist_set_local_worker(
        ds4_distributed_options *opt,
        bool local_worker);
void ds4_internal_dist_copy_options_private(
        const ds4_distributed_options *src,
        const ds4_distributed_options *dst);
void ds4_internal_dist_clear_options_private(
        const ds4_distributed_options *opt);

#endif
