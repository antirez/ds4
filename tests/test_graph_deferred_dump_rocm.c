#include "ds4.h"
#include "ds4_gpu.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

enum {
    FLASH_LAYERS = 43,
    FLASH_HC_F32 = 16384,
    FLASH_TOPK = 6,
};

static int read_exact(const char *path, void *buf, size_t bytes) {
    FILE *fp = fopen(path, "rb");
    if (!fp) return 0;
    const int ok = fread(buf, 1, bytes, fp) == bytes && fgetc(fp) == EOF;
    fclose(fp);
    return ok;
}

int main(void) {
    char prefix[256];
    snprintf(prefix, sizeof(prefix), "/tmp/ds4-deferred-dump-%ld", (long)getpid());
    const char *names =
        "hc_attn_post,ffn_moe_topk,ffn_moe_weights_scaled,hc_ffn_post";
    setenv("DS4_ROCM_GRAPH_DUMP_PREFIX", prefix, 1);
    setenv("DS4_METAL_GRAPH_DUMP_PREFIX", prefix, 1);
    setenv("DS4_ROCM_GRAPH_DUMP_NAME", names, 1);
    setenv("DS4_METAL_GRAPH_DUMP_NAME", names, 1);
    setenv("DS4_ROCM_GRAPH_DUMP_LAYER", "all", 1);
    setenv("DS4_METAL_GRAPH_DUMP_LAYER", "all", 1);
    setenv("DS4_ROCM_GRAPH_DUMP_DEFER_LAYER_STATE", "1", 1);
    setenv("DS4_METAL_GRAPH_DUMP_DEFER_LAYER_STATE", "1", 1);

    if (!ds4_gpu_init()) {
        fprintf(stderr, "test_graph_deferred_dump_rocm: GPU init failed\n");
        return 1;
    }
    int ok = ds4_test_graph_deferred_dump_roundtrip() != 0;
    float *hc = (float *)malloc(FLASH_HC_F32 * sizeof(float));
    float weights[FLASH_TOPK];
    int32_t topk[FLASH_TOPK];
    ok = ok && hc;

    for (uint32_t il = 0; il < FLASH_LAYERS; il++) {
        char path[512];
        const char *hc_names[] = {"hc_attn_post", "hc_ffn_post"};
        for (size_t ni = 0; ni < sizeof(hc_names) / sizeof(hc_names[0]); ni++) {
            snprintf(path, sizeof(path), "%s_%s-%u_pos0.bin", prefix, hc_names[ni], il);
            if (ok) {
                ok = read_exact(path, hc, FLASH_HC_F32 * sizeof(float));
#ifdef DS4_ROCM_BUILD
                const float expected = ni == 0 ?
                    1.25f + (float)il : -2.5f - (float)il;
#else
                const float expected = ni == 0 ? 1.25f : -2.5f;
#endif
                for (uint32_t i = 0; ok && i < FLASH_HC_F32; i++)
                    ok = hc[i] == expected;
            }
            unlink(path);
        }
        snprintf(path, sizeof(path), "%s_ffn_moe_topk-%u_pos0.i32", prefix, il);
        if (ok) {
            ok = read_exact(path, topk, sizeof(topk));
            for (uint32_t i = 0; ok && i < FLASH_TOPK; i++)
                ok = topk[i] == (int32_t)i;
        }
        unlink(path);

        snprintf(path, sizeof(path),
                 "%s_ffn_moe_weights_scaled-%u_pos0.bin", prefix, il);
        if (ok) {
            ok = read_exact(path, weights, sizeof(weights));
            for (uint32_t i = 0; ok && i < FLASH_TOPK; i++) {
#ifdef DS4_ROCM_BUILD
                ok = weights[i] == 0.125f + (float)il;
#else
                ok = weights[i] == 0.125f;
#endif
            }
        }
        unlink(path);
    }
    free(hc);
    ds4_gpu_cleanup();
    if (!ok) {
        fprintf(stderr, "test_graph_deferred_dump_rocm: FAIL\n");
        return 1;
    }
    printf("test_graph_deferred_dump_rocm: PASS (172 deferred captures)\n");
    return 0;
}
