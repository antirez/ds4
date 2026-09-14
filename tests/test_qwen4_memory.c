#define DS4_NO_GPU
#include "../ds4.c"
#include <assert.h>

static ds4_context_memory estimate(int ctx, uint32_t chunk) {
    ds4_context_memory m = ds4_context_memory_estimate_with_prefill_mode(
        DS4_BACKEND_METAL, ctx, chunk, true);
    assert(m.total_bytes == m.raw_bytes + m.compressed_bytes + m.scratch_bytes);
    if (!m.prefill_cap || m.prefill_cap > m.raw_cap)
        fprintf(stderr, "invalid Qwen caps: ctx=%d chunk=%u family=%u prefill=%u raw=%u\n",
                ctx, chunk, (unsigned)DS4_MODEL_FAMILY, m.prefill_cap, m.raw_cap);
    assert(m.prefill_cap > 0 && m.prefill_cap <= m.raw_cap);
    return m;
}

static void monotonic(void) {
    const uint32_t chunks[] = {1, 2, 8, 128, 1024, 4096, 8192};
    uint64_t previous = 0;
    for (size_t i = 0; i < sizeof(chunks) / sizeof(chunks[0]); i++) {
        ds4_context_memory m = estimate(8192, chunks[i]);
        assert(m.prefill_cap == chunks[i]);
        assert(m.total_bytes > previous);
        previous = m.total_bytes;
    }
    const int contexts[] = {1, 16, 128, 512, 4096, 8192, 262144};
    previous = 0;
    for (size_t i = 0; i < sizeof(contexts) / sizeof(contexts[0]); i++) {
        ds4_context_memory m = estimate(contexts[i], 128);
        assert(m.total_bytes > previous);
        previous = m.total_bytes;
    }
    ds4_context_memory bounded = estimate(128, 128);
    ds4_context_memory oversized = estimate(128, UINT32_MAX);
    assert(oversized.prefill_cap == 128 && oversized.total_bytes == bounded.total_bytes);
    ds4_context_memory smallest = estimate(0, 128);
    assert(smallest.raw_cap == 1 && smallest.prefill_cap == 1);
    ds4_context_memory largest = estimate(INT_MAX, UINT32_MAX);
    assert(largest.prefill_cap == INT_MAX && largest.total_bytes > previous);
}

int main(void) {
    const ds4_shape saved = g_ds4_shape;
    g_ds4_shape = DS4_SHAPE_QWEN4_EXP;
    monotonic();

    /* Independent allocation lower bounds for the shipped 49-layer shape:
     * 36 GDN layers, 48 value heads, 128x128 state, three convolution rows;
     * one live state and three MTP snapshots. No tensor data is needed. */
    const uint64_t recurrent = UINT64_C(36) *
        (48u * 128u * 128u + 3u * 10240u) * 4u + 9u * 10240u * 4u;
    assert(recurrent == UINT64_C(118038528));
    const uint64_t split_attention = UINT64_C(3) * 24u * 64u * 258u * 4u;
    const uint64_t logit_buffers = UINT64_C(12) * 248320u * 4u;
    ds4_context_memory m = estimate(4096, 128);
    const ds4_context_memory large_ctx = estimate(INT_MAX, 1);
    assert(large_ctx.compressed_bytes == UINT64_C(13) *
        ((uint64_t)INT_MAX * 128u * 4u + ((uint64_t)INT_MAX / 4u + 1u) * 128u * 2u));
    assert(m.scratch_bytes >= 4u * recurrent + split_attention + logit_buffers +
                              UINT64_C(128) * (10240u + 4u) * 4u);
    /* The conservative estimate remains bounded; no complete weight tensor or
     * n-gram table belongs to the graph workspace. */
    assert(m.total_bytes < UINT64_C(1024) * 1024u * 1024u);
    ds4_context_memory two = estimate(4096, 2);
    ds4_context_memory one = estimate(4096, 1);
    assert(two.raw_bytes == one.raw_bytes && two.compressed_bytes == one.compressed_bytes);
    assert(two.scratch_bytes - one.scratch_bytes >= (10240u + 4u) * 4u);

    ds4_context_memory resident = ds4_context_memory_estimate_with_prefill_mode(
        DS4_BACKEND_METAL, 4096, 128, false);
    assert(resident.total_bytes == m.total_bytes);
    assert(setenv("DS4_QWEN4_PREFILL_CHUNK", "128", 1) == 0);
    ds4_context_memory from_env = estimate(4096, 0);
    assert(from_env.prefill_cap == 128 && from_env.total_bytes == m.total_bytes);
    assert(unsetenv("DS4_QWEN4_PREFILL_CHUNK") == 0);

    /* Main-only Qwen files have no predictor or verifier snapshots. */
    g_ds4_shape.n_layer--;
    g_ds4_shape.n_nextn_predict = 0;
    ds4_context_memory main_only = estimate(4096, 128);
    assert(main_only.total_bytes < m.total_bytes);
    assert(m.total_bytes - main_only.total_bytes >= 3u * recurrent);
    g_ds4_shape = DS4_SHAPE_QWEN4_MINI;
    monotonic();
    g_ds4_shape = saved;
    puts("Qwen memory: bounded workspace, recurrent/MTP reserves and monotonic budgets OK");
    return 0;
}
