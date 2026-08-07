#include "first_divergence_capture.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define REQUIRE(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition); \
        return 1; \
    } \
} while (0)

int main(void) {
    ds4_first_divergence_capture capture;
    const float cp1[] = {1.0f, -0.0f, 3.5f};
    const uint32_t n_comp = 17;

    REQUIRE(ds4_first_divergence_capture_init(&capture, "PASS_A"));
    REQUIRE(ds4_first_divergence_capture_f32(
        &capture, 0, 0, DS4_FIRST_DIVERGENCE_CP1, "attn_norm",
        cp1, sizeof(cp1) / sizeof(cp1[0])));
    REQUIRE(ds4_first_divergence_capture_u32(
        &capture, 0, 0, DS4_FIRST_DIVERGENCE_CP3_F, "layer_n_comp",
        &n_comp, 1));
    REQUIRE(capture.count == 2);
    REQUIRE(capture.snapshots[0].element_count == 3);
    REQUIRE(memcmp(capture.snapshots[0].data, cp1, sizeof(cp1)) == 0);
    REQUIRE(strcmp(ds4_first_divergence_checkpoint_name(
                       DS4_FIRST_DIVERGENCE_CP2_KV_R),
                   "CP2-KV-R") == 0);
    ds4_first_divergence_capture_free(&capture);
    REQUIRE(capture.count == 0);

    puts("first-divergence capture: OK");
    return 0;
}
