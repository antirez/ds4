#include "ds4_engine_internal.h"



bool gpu_graph_directional_steering_attn_enabled(const ds4_gpu_graph *g) {
    return g && g->directional_steering_dirs && g->directional_steering_attn_scale != 0.0f;
}



bool gpu_graph_directional_steering_ffn_enabled(const ds4_gpu_graph *g) {
    return g && g->directional_steering_dirs && g->directional_steering_ffn_scale != 0.0f;
}



static bool gpu_graph_apply_directional_steering(
        ds4_gpu_graph  *g,
        ds4_gpu_tensor *x,
        uint32_t          il,
        uint32_t          rows,
        float             scale) {
    if (!g || !g->directional_steering_dirs || scale == 0.0f) return true;
    return ds4_gpu_directional_steering_project_tensor(x,
                                            g->directional_steering_dirs,
                                            il,
                                            DS4_N_EMBD,
                                            rows,
                                            scale) != 0;
}



bool gpu_graph_apply_directional_steering_attn(
        ds4_gpu_graph  *g,
        ds4_gpu_tensor *x,
        uint32_t          il,
        uint32_t          rows) {
    return gpu_graph_apply_directional_steering(g, x, il, rows, g ? g->directional_steering_attn_scale : 0.0f);
}



bool gpu_graph_apply_directional_steering_ffn(
        ds4_gpu_graph  *g,
        ds4_gpu_tensor *x,
        uint32_t          il,
        uint32_t          rows) {
    return gpu_graph_apply_directional_steering(g, x, il, rows, g ? g->directional_steering_ffn_scale : 0.0f);
}



static uint64_t gpu_graph_kv_cache_bytes_for_context(uint32_t ctx_size, uint32_t raw_cap) {
    /* Must track the ACTUAL storage formats (raw f16, packed attn comp, MXFP4
     * indexer) — an f32-priced estimate overshoots ~3x under the default-on
     * packed formats and needlessly trips the managed-KV (demand-paged) path
     * at the 512k+ contexts where performance matters most. */
    const uint64_t raw_elem = gpu_graph_raw_f16_enabled() ? sizeof(uint16_t)
                                                          : sizeof(float);
    uint64_t bytes = (uint64_t)DS4_N_LAYER * raw_cap * DS4_N_HEAD_DIM * raw_elem;

    const uint64_t attn_row = gpu_graph_attn_comp_cache_row_bytes();
    const uint64_t idx_row = gpu_graph_idx_fp4_enabled()
        ? DS4_ENGINE_IDXFP4_ROWBYTES
        : (uint64_t)DS4_N_INDEXER_HEAD_DIM * sizeof(float);
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (ratio == 0) continue;
        const uint64_t comp_cap = (uint64_t)(ctx_size / ratio + 2u);
        bytes += comp_cap * attn_row;
        if (ratio == 4) {
            bytes += comp_cap * idx_row;
        }
    }
    return bytes;
}



uint64_t gpu_graph_context_bytes_for_kv_policy(
        uint32_t  ctx_size,
        uint32_t  raw_cap,
        uint32_t  prefill_cap,
        uint64_t *kv_cache_bytes_out) {
    uint32_t min_ratio = UINT32_MAX;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (ratio != 0 && ratio < min_ratio) min_ratio = ratio;
    }
    if (min_ratio == UINT32_MAX) min_ratio = ctx_size ? ctx_size : 1u;
    uint64_t comp_cap = (uint64_t)(ctx_size / min_ratio + 2u);
    if (comp_cap < 2u) comp_cap = 2u;
    const uint64_t kv_cache_bytes = gpu_graph_kv_cache_bytes_for_context(ctx_size, raw_cap);
    if (kv_cache_bytes_out) *kv_cache_bytes_out = kv_cache_bytes;
    /* indexer_scores/comp_mask token rows shrink to the slice size under
     * DS4_PREFILL_SLICE (see gpu_graph_prefill_slice / the gpu_diag alloc). */
    uint64_t score_rows = (uint64_t)prefill_cap;
    if (gpu_graph_prefill_slice() != 0u && (uint64_t)gpu_graph_prefill_slice() < score_rows) {
        score_rows = (uint64_t)gpu_graph_prefill_slice();
    }
    uint64_t bytes = kv_cache_bytes +
                     2ull * comp_cap * score_rows * sizeof(float);
    return bytes;
}



ds4_gpu_tensor *gpu_graph_alloc_kv_cache_tensor(bool managed, uint64_t bytes) {
    return managed ? ds4_gpu_tensor_alloc_managed(bytes) : ds4_gpu_tensor_alloc(bytes);
}



/* =========================================================================
 * GPU Diagnostic Dump Hooks.
 * =========================================================================
 *
 * The release path calls these after important stages, but they are no-ops
 * unless DS4_CUDA_GRAPH_DUMP_PREFIX is set.  Dumping synchronizes and restarts
 * the command batch, so it is intentionally isolated here.
 */

/* Is graph dumping enabled AT ALL this process?  Read once and cached.  Exposed
 * so graph allocation can skip buffers that exist ONLY to be dumped: the routed
 * MoE gate/up staging slabs are written by the qwarp32 kernels and read by
 * nothing except gpu_graph_debug_dump_tensor, yet cost ~400 MB per session at
 * the default prefill_cap and ~2-4 MB/token of pointless device writes. */
bool gpu_graph_debug_dump_enabled(void) {
    static int enabled = -1;
    if (enabled < 0) {
        const char *p = getenv("DS4_CUDA_GRAPH_DUMP_PREFIX");
        enabled = (p && p[0]) ? 1 : 0;
    }
    return enabled != 0;
}

bool gpu_graph_debug_wants(const char *name, uint32_t il, uint32_t pos) {
    /* Called ~60x/layer x 43 layers/token on the release decode path, where it
     * is a no-op unless dumping is enabled.  Cache the enabled flag so the
     * common (disabled) case costs one branch, not a linear environ scan per
     * call.  The dump env vars are read once at process start like every other
     * flag; caching does not change runtime behavior. */
    if (!gpu_graph_debug_dump_enabled()) return false;

    const char *name_env = getenv("DS4_CUDA_GRAPH_DUMP_NAME");
    if (name_env && name_env[0] && strstr(name_env, name) == NULL) return false;

    const char *layer_env = getenv("DS4_CUDA_GRAPH_DUMP_LAYER");
    if (layer_env && layer_env[0] && strcmp(layer_env, "all") != 0 &&
        (uint32_t)strtoul(layer_env, NULL, 10) != il) return false;

    const char *pos_env = getenv("DS4_CUDA_GRAPH_DUMP_POS");
    if (pos_env && pos_env[0] && (uint32_t)strtoul(pos_env, NULL, 10) != pos) return false;

    return true;
}

