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

static void streaming_staging(void) {
    const ds4_shape saved = g_ds4_shape;
    g_ds4_shape = DS4_SHAPE_QWEN4_EXP;
    /* Metadata only: Q2 trunk down stores 768 columns, although the logical
     * FFN width is 640. The differently quantized MTP layer uses that logical
     * width directly. No model payload or GPU allocation is needed. */
    ds4_tensor gate = { .type = DS4_TENSOR_IQ2_XXS, .ndim = 3, .dim = {2560, 640, 512} };
    ds4_tensor down = { .type = DS4_TENSOR_Q2_K, .ndim = 3, .dim = {768, 2560, 512} };
    ds4_tensor mtp_gate = { .type = DS4_TENSOR_Q4_K, .ndim = 3, .dim = {2560, 640, 512} };
    ds4_tensor mtp_down = { .type = DS4_TENSOR_MXFP4, .ndim = 3, .dim = {640, 2560, 512} };
    ds4_weights w = {0};
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const bool predictor = il == DS4_N_LAYER - 1u;
        w.layer[il].ffn_gate_exps = w.layer[il].ffn_up_exps = predictor ? &mtp_gate : &gate;
        w.layer[il].ffn_down_exps = predictor ? &mtp_down : &down;
    }
    const uint64_t trunk = UINT64_C(1489920) * 512u;
    const uint64_t predictor = UINT64_C(2713600) * 512u;
    uint64_t bytes = 0;
    assert(qwen4_streaming_staging_bytes(&w, true, &bytes));
    assert(bytes == 2u * trunk && bytes == UINT64_C(1525678080));
    assert(bytes >= predictor);  /* even no-address full-layer fallback fits */
    assert(2u * predictor - bytes == UINT64_C(1253048320));
    assert(qwen4_streaming_staging_bytes(&w, false, &bytes) && bytes == 2u * trunk);

    /* Disabled MTP must not inspect absent predictor tensors. A main-only
     * file has the same reserve with no predictor layer in its shape. */
    w.layer[48].ffn_down_exps = NULL;
    assert(qwen4_streaming_staging_bytes(&w, false, &bytes) && bytes == 2u * trunk);
    assert(!qwen4_streaming_staging_bytes(&w, true, &bytes) && bytes == 0);
    g_ds4_shape.n_layer = 48;
    g_ds4_shape.n_nextn_predict = 0;
    assert(qwen4_streaming_staging_bytes(&w, true, &bytes) && bytes == 2u * trunk);
    g_ds4_shape = DS4_SHAPE_QWEN4_EXP;
    w.layer[48].ffn_down_exps = &mtp_down;

    /* A larger off-size predictor makes the conservative full-layer fallback
     * dominate. It remains reserved even though normal MTP selects only ten. */
    mtp_gate.dim[1] *= 2u;
    mtp_down.dim[1] *= 2u;
    assert(qwen4_streaming_staging_bytes(&w, true, &bytes) && bytes == 2u * predictor);
    assert(qwen4_streaming_staging_bytes(&w, false, &bytes) && bytes == 2u * trunk);
    mtp_gate.dim[1] /= 2u;
    mtp_down.dim[1] /= 2u;

    /* A smaller expert population can make two compact windows dominate;
     * include all three address tables for both in-flight lifetimes. */
    g_ds4_shape.n_expert = 16;
    g_ds4_shape.n_expert_used = 10;
    assert(qwen4_streaming_staging_bytes(&w, true, &bytes));
    assert(bytes == 2u * (UINT64_C(2713600) * 10u + 3u * 16u * sizeof(uint64_t)));
    assert(bytes > UINT64_C(2713600) * 16u);
    g_ds4_shape.n_expert_used = 17;
    assert(!qwen4_streaming_staging_bytes(&w, true, &bytes) && bytes == 0);
    g_ds4_shape = DS4_SHAPE_QWEN4_EXP;

    /* Saturated totals cannot wrap into a small admissible cache reserve;
     * invalid per-expert products fail before any budget is returned. */
    ds4_tensor huge = { .type = DS4_TENSOR_Q8_0, .ndim = 3,
                       .dim = {32, UINT64_MAX / 102u, 512} };
    w.layer[0].ffn_gate_exps = w.layer[0].ffn_up_exps = w.layer[0].ffn_down_exps = &huge;
    assert(qwen4_streaming_staging_bytes(&w, true, &bytes) && bytes == UINT64_MAX);
    huge.dim[1] = UINT64_MAX;
    assert(!qwen4_streaming_staging_bytes(&w, true, &bytes) && bytes == 0);
    assert(!qwen4_streaming_staging_bytes(NULL, true, &bytes) && bytes == 0);
    assert(!qwen4_streaming_staging_bytes(&w, true, NULL));
    g_ds4_shape = saved;
}

int main(void) {
    const ds4_shape saved = g_ds4_shape;
    g_ds4_shape = DS4_SHAPE_QWEN4_EXP;
    monotonic();
    streaming_staging();

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
    puts("Qwen memory: bounded workspace, recurrent/MTP staging and monotonic budgets OK");
    return 0;
}
