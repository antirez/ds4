#include "pulsar_engine_internal.h"

/* Model-layout queries: per-layer attention compression ratios and routed
 * expert counts. Split out of util.c in the C++ port. */

bool pulsar_backend_uses_graph(pulsar_backend backend) {
    return backend == PULSAR_BACKEND_CUDA;
}



/* Attention compression is read from GGUF metadata after validating that it
 * matches the exact layout expected for the loaded model shape. */
uint32_t pulsar_layer_compress_ratio(uint32_t il) {
    if (il >= PULSAR_N_LAYER) pulsar_die("DeepSeek4 layer index is outside the loaded model layout");
    return g_pulsar_compress_ratios[il];
}



/* Physically-present routed-expert count for a layer. For an un-pruned model
 * (or any layer whose keep_count was not set) this is the full n_expert; for a
 * REAP ds4-compact-v1 model the pruned layers report their dense survivor
 * count. Only the expert *weight* tensors are trimmed to this; the router and
 * bias stay padded to n_expert. */
uint32_t pulsar_layer_n_expert(uint32_t il) {
    if (il >= PULSAR_N_LAYER) pulsar_die("DeepSeek4 layer index is outside the loaded model layout");
    const uint32_t v = g_pulsar_layer_expert_count[il];
    return v ? v : PULSAR_N_EXPERT;
}



uint32_t pulsar_expected_layer_compress_ratio(uint32_t il) {
    if (il >= PULSAR_N_LAYER) pulsar_die("DeepSeek4 layer index is outside the loaded model layout");

    switch (PULSAR_MODEL_VARIANT) {
    case PULSAR_VARIANT_FLASH:
        if (il < 2) return 0;
        return (il & 1u) == 0 ? 4u : 128u;
    case PULSAR_VARIANT_PRO:
        if (il < 2) return 128u;
        return (il & 1u) == 0 ? 4u : 128u;
    default:
        pulsar_die("unsupported DeepSeek4 model variant");
    }
    return 0;
}
