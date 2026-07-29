#define _POSIX_C_SOURCE 200809L

#include "ds4_gpu.h"
#include "k3_model.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

bool ds4_log_is_tty(FILE *fp) {
    (void)fp;
    return false;
}

static void fail(const char *message) {
    fprintf(stderr, "test_k3_gguf_metal_map: %s\n", message);
    exit(1);
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s K3.gguf\n", argv[0]);
        return 2;
    }

    char err[512] = {0};
    k3_model *model = NULL;
    k3_layer *layer = NULL;
    if (!k3_model_open(&model, argv[1], err, sizeof(err)) ||
        !model->gguf ||
        !k3_model_open_layer(model, 1, &layer, err, sizeof(err)) ||
        !k3_st_map(layer->shard, err, sizeof(err))) {
        fprintf(stderr, "test_k3_gguf_metal_map: %s\n",
                err[0] ? err : "model open failed");
        return 1;
    }

    const k3_st_tensor *experts[] = {
        layer->expert_gate,
        layer->expert_up,
        layer->expert_down,
    };
    uint64_t offsets[4] = {layer->static_offset};
    uint64_t sizes[4] = {layer->static_size};
    uint64_t max_tensor = layer->static_size;
    for (uint32_t i = 0; i < 3; i++) {
        if (!experts[i]) fail("missing routed expert stack");
        offsets[i + 1u] = experts[i]->offset;
        sizes[i + 1u] = experts[i]->size;
        if (sizes[i + 1u] > max_tensor) max_tensor = sizes[i + 1u];
    }

    ds4_gpu_set_ssd_streaming(true);
    if (!ds4_gpu_init() ||
        !ds4_gpu_set_model_fd_for_map(
            k3_st_fd(layer->shard), k3_st_map_base(layer->shard)) ||
        !ds4_gpu_set_model_map_spans(
            k3_st_map_base(layer->shard),
            k3_st_file_size(layer->shard),
            offsets,
            sizes,
            4,
            max_tensor)) {
        fail("Metal could not map the production K3 GGUF layer spans");
    }
    if (!ds4_gpu_release_model_views()) fail("Metal view release failed");

    ds4_gpu_cleanup();
    k3_st_unmap(layer->shard);
    k3_layer_close(layer);
    k3_model_close(model);
    puts("test_k3_gguf_metal_map: ok");
    return 0;
}
