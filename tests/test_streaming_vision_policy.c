/* No model or GPU allocation: exercise the actual short-prefill admission. */
#include "../ds4.c"
#include <assert.h>

int main(void) {
    unsetenv("DS4_METAL_DISABLE_STREAMING_DECODE_PREFILL");
    unsetenv("DS4_ROCM_DISABLE_STREAMING_DECODE_PREFILL");
    unsetenv("DS4_METAL_DISABLE_STREAMING_COLD_DECODE_PREFILL");
    unsetenv("DS4_ROCM_DISABLE_STREAMING_COLD_DECODE_PREFILL");
    setenv("DS4_METAL_STREAMING_DECODE_PREFILL_MAX", "512", 1);
    setenv("DS4_ROCM_STREAMING_DECODE_PREFILL_MAX", "512", 1);
    ds4_vision_span images[] = {
        {.token_start = 20, .embedding = {.token_count = 81}},
        {.token_start = 200, .embedding = {.token_count = 81}}
    };
    ds4_gpu_graph g = {.ssd_streaming = true,
        .prefill_vision_spans = images, .prefill_vision_span_count = 2};
    assert(!metal_graph_use_streaming_decode_prefill_range(&g, NULL, 0, 150));
    assert(!metal_graph_use_streaming_decode_prefill_range(&g, NULL, 19, 2));
    assert(!metal_graph_use_streaming_decode_prefill_range(&g, NULL, 100, 2));
    assert(!metal_graph_use_streaming_decode_prefill_range(&g, NULL, 200, 81));
    assert(metal_graph_use_streaming_decode_prefill_range(&g, NULL, 0, 20));
    assert(metal_graph_use_streaming_decode_prefill_range(&g, NULL, 101, 99));
    assert(metal_graph_use_streaming_decode_prefill_range(&g, NULL, 281, 8));
    assert(!metal_graph_use_streaming_decode_prefill_range(&g, NULL, 281, 513));
    /* Empty spans are not image-conditioned tokens. */
    assert(!metal_graph_visual_range_overlaps(&g, 20, 0));
    assert(!metal_graph_visual_range_overlaps(NULL, 20, 1));
    /* Widen addition so range endpoints cannot wrap at UINT32_MAX. */
    images[0].token_start = UINT32_MAX - 4u;
    images[0].embedding.token_count = 8;
    assert(metal_graph_visual_range_overlaps(&g, UINT32_MAX - 2u, 4));
    assert(!metal_graph_visual_range_overlaps(&g, 0, 4));
    g.prefill_vision_span_count = 0;
    assert(metal_graph_use_streaming_decode_prefill_range(&g, NULL, 0, 150));
    g.quality = true;
    assert(!metal_graph_use_streaming_decode_prefill_range(&g, NULL, 0, 150));
    puts("streaming vision policy: OK");
    return 0;
}
