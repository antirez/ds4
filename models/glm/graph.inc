/*
 * GLM DSA graph state, allocation, decode, prefill, and diagnostics.
 *
 * Included exactly once inside ds4.c's graph-backend conditional. No generic
 * operator layer is introduced between this model integration and its kernels.
 */

typedef struct {
    uint32_t ctx_size;
    uint32_t ctx_cap;
    uint32_t normal_layers;
    uint32_t layer_start;
    uint32_t layer_end;
    uint32_t layer_count;
    uint64_t q_dim;
    uint64_t q_nope;
    uint64_t heads_dim;
    uint64_t kv_raw_dim;
    uint64_t dense_hidden_max;
    uint64_t ffn_mid_elems;

    ds4_gpu_tensor *cur;
    ds4_gpu_tensor *next;
    ds4_gpu_tensor *attn_norm;
    ds4_gpu_tensor *q_rank;
    ds4_gpu_tensor *q_rank_norm;
    ds4_gpu_tensor *q;
    ds4_gpu_tensor *kv_raw;
    ds4_gpu_tensor *kv_norm;
    ds4_gpu_tensor *k_nope;
    ds4_gpu_tensor *value;
    ds4_gpu_tensor *heads;
    ds4_gpu_tensor *attn_out;
    ds4_gpu_tensor *after_attn;
    ds4_gpu_tensor *ffn_norm;
    ds4_gpu_tensor *ffn_gate;
    ds4_gpu_tensor *ffn_up;
    ds4_gpu_tensor *ffn_mid;
    ds4_gpu_tensor *routed_gate;
    ds4_gpu_tensor *routed_up;
    ds4_gpu_tensor *routed_down;
    ds4_gpu_tensor *ffn_out;
    ds4_gpu_tensor *ffn_sum;
    ds4_gpu_tensor *router_logits;
    ds4_gpu_tensor *router_probs;
    ds4_gpu_tensor *router_selected;
    ds4_gpu_tensor *router_weights;
    ds4_gpu_tensor *output_norm;
    ds4_gpu_tensor *logits;
    ds4_gpu_tensor *batch_router_logits;
    ds4_gpu_tensor *batch_router_probs;
    ds4_gpu_tensor *batch_router_selected;
    ds4_gpu_tensor *batch_router_weights;
    ds4_gpu_tensor *prefill_seed_router_selected;
    uint32_t prefill_seed_tokens;
    bool prefill_seed_layer_captured[DS4_MAX_LAYER];

    ds4_gpu_tensor *prefill_tokens;
    ds4_gpu_tensor *batch_cur;
    ds4_gpu_tensor *batch_next;
    ds4_gpu_tensor *batch_attn_norm;
    ds4_gpu_tensor *batch_q_rank;
    ds4_gpu_tensor *batch_q_rank_norm;
    ds4_gpu_tensor *batch_q;
    ds4_gpu_tensor *batch_kv_raw;
    ds4_gpu_tensor *batch_kv_norm;
    ds4_gpu_tensor *batch_k_nope;
    ds4_gpu_tensor *batch_value;
    ds4_gpu_tensor *batch_heads;
    ds4_gpu_tensor *batch_attn_out;
    ds4_gpu_tensor *batch_after_attn;
    ds4_gpu_tensor *batch_ffn_norm;
    ds4_gpu_tensor *batch_ffn_gate;
    ds4_gpu_tensor *batch_ffn_up;
    ds4_gpu_tensor *batch_shared_mid;
    ds4_gpu_tensor *batch_ffn_mid;
    ds4_gpu_tensor *batch_routed_gate;
    ds4_gpu_tensor *batch_routed_up;
    ds4_gpu_tensor *batch_routed_down;
    ds4_gpu_tensor *batch_ffn_out;
    bool batch_routed_mid_is_f16;

    uint32_t compact_cache_cap;
    uint32_t indexed_prefill_cap;
    uint32_t indexed_prefill_score_cap;
    uint32_t indexer_full_layers;
    ds4_gpu_tensor *indexer_k;
    ds4_gpu_tensor *indexer_q;
    ds4_gpu_tensor *indexer_weights;
    ds4_gpu_tensor *indexer_scores;
    ds4_gpu_tensor *indexer_selected;
    ds4_gpu_tensor *qk_low;
    ds4_gpu_tensor *attn_partial_lora;
    ds4_gpu_tensor *attn_partial_ms;
    ds4_gpu_tensor *batch_indexer_k;
    ds4_gpu_tensor *batch_indexer_q;
    ds4_gpu_tensor *batch_indexer_weights;
    ds4_gpu_tensor *batch_indexer_scores;
    ds4_gpu_tensor *batch_indexer_selected;
    ds4_gpu_tensor *batch_qk_low;
    ds4_gpu_tensor *batch_attn_lora;
    ds4_gpu_tensor *layer_kv_lora_cache[DS4_MAX_LAYER];
    ds4_gpu_tensor *layer_k_rope_cache[DS4_MAX_LAYER];
    /* GLM MTP (nextn block) drafting: private compact caches for the nextn
     * layer (slot = absolute position; only [mtp_min_pos..pos] is ever
     * selected) plus small scratch. Allocated lazily on first draft. */
    ds4_gpu_tensor *mtp_kv_lora_cache;
    ds4_gpu_tensor *mtp_k_rope_cache;
    ds4_gpu_tensor *mtp_concat;
    ds4_gpu_tensor *mtp_selected;
    float          *mtp_logits_host;
    int             mtp_ready;
    ds4_gpu_tensor *layer_indexer_key_cache[DS4_MAX_LAYER];
    ds4_gpu_tensor *layer_key_cache[DS4_MAX_LAYER];
    ds4_gpu_tensor *layer_value_cache[DS4_MAX_LAYER];
    bool full_kv_cache;
    bool has_token_embd;
    bool has_output_head;
    bool quality;
    bool ssd_streaming;
    bool ssd_streaming_cold;
    bool generic_routed_moe;
    bool streaming_static_decode_map_current;
    /* Tensor parallelism (50/50 expert sharding): tp_world 2 means
     * this rank computes only its contiguous half of the routed experts
     * and exchanges the 24KB routed-FFN partial at one gate per sparse
     * layer.  Views alias the engine's TP slab slots [layer*2 + FFN]. */
    uint32_t tp_world;
    uint32_t tp_rank;
    ds4_gpu_tensor **tp_out;
    ds4_gpu_tensor **tp_in;
    /* Prefill batch gate bounce buffers (shared storage; grow on demand). */
    ds4_gpu_tensor *tp_bounce_out;
    ds4_gpu_tensor *tp_bounce_in;
    /* CUDA multi-tier placement and device-local decode scratch mirrors. */
    const int *placement;
#define DS4_GLM_WS_SLOTS 29
    ds4_gpu_tensor *ws_mirror[DS4_MAX_GPUS][DS4_GLM_WS_SLOTS];
    ds4_gpu_tensor *ws_orig[DS4_GLM_WS_SLOTS];
    int ws_ready;
    int ws_tier;
#define DS4_GLM_VERIFY_WS_SLOTS 28
    ds4_gpu_tensor *verify_ws_mirror[DS4_MAX_GPUS][DS4_GLM_VERIFY_WS_SLOTS];
    ds4_gpu_tensor *verify_ws_orig[DS4_GLM_VERIFY_WS_SLOTS];
    int verify_ws_ready;
    int verify_ws_tier;
} ds4_glm_gpu_graph;

static uint32_t glm_graph_model_context_limit(void) {
    if (DS4_ROPE_ORIG_CTX > UINT32_MAX) return UINT32_MAX;
    return (uint32_t)DS4_ROPE_ORIG_CTX;
}

static double glm_graph_bytes_to_gib(uint64_t bytes) {
    return (double)bytes / (1024.0 * 1024.0 * 1024.0);
}

static uint64_t glm_graph_saturating_add_u64(uint64_t a, uint64_t b) {
    return a > UINT64_MAX - b ? UINT64_MAX : a + b;
}

static bool glm_graph_env_disabled(const char *name) {
    const char *env = getenv(name);
    if (!env || !env[0]) return false;
    return strcmp(env, "0") == 0 ||
           strcasecmp(env, "false") == 0 ||
           strcasecmp(env, "off") == 0 ||
           strcasecmp(env, "no") == 0;
}

static double glm_graph_env_double(
        const char *name,
        double      fallback,
        double      min_value,
        double      max_value) {
    const char *env = getenv(name);
    if (!env || !env[0]) return fallback;
    char *end = NULL;
    errno = 0;
    const double v = strtod(env, &end);
    if (end == env || errno != 0 || !isfinite(v)) return fallback;
    if (v < min_value) return min_value;
    if (v > max_value) return max_value;
    return v;
}

static uint64_t glm_graph_host_memory_bytes(void) {
#if defined(__APPLE__)
    uint64_t mem = 0;
    size_t len = sizeof(mem);
    if (sysctlbyname("hw.memsize", &mem, &len, NULL, 0) != 0) return 0;
    return mem;
#else
    return 0;
#endif
}

static uint64_t glm_graph_streaming_active_model_bytes(
        const ds4_weights *weights) {
    if (!weights) return 0;

    uint64_t max_bytes = 0;
    ds4_model_map_span_vec spans;

    if (weights_layer_has_required(&weights->layer[0], 0) &&
        weights_model_map_token_spans(weights, &spans)) {
        max_bytes = model_map_span_vec_total_bytes(&spans);
        free(spans.v);
    }
    if (weights_have_output_head(weights) &&
        weights_model_map_output_spans(weights, &spans)) {
        const uint64_t bytes = model_map_span_vec_total_bytes(&spans);
        if (bytes > max_bytes) max_bytes = bytes;
        free(spans.v);
    }
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        if (!weights_model_map_spans(weights, il, il, false, &spans)) {
            continue;
        }
        const uint64_t bytes = model_map_span_vec_total_bytes(&spans);
        if (bytes > max_bytes) max_bytes = bytes;
        free(spans.v);
    }

    return max_bytes;
}

/* TP shard bytes: dense weights plus this rank's routed-expert range.
 * Zero when not sharding.  Set during engine open, before the GLM memory
 * guard runs. */
static uint64_t g_tp_shard_model_bytes;

/* A user-raised iogpu.wired_limit_mb is an explicit GPU budget grant;
 * prefer it over the fraction/reserve heuristics. */
static uint64_t glm_graph_wired_limit_bytes(void) {
#ifdef __APPLE__
    int64_t mb = 0;
    size_t len = sizeof(mb);
    if (sysctlbyname("iogpu.wired_limit_mb", &mb, &len, NULL, 0) != 0) return 0;
    if (mb <= 0) return 0;
    return (uint64_t)mb * 1024ull * 1024ull;
#else
    return 0;
#endif
}

static uint64_t glm_graph_model_bytes_for_guard(
        const ds4_model   *model,
        const ds4_weights *weights,
        bool               ssd_streaming,
        bool               load_slice,
        uint32_t           layer_start,
        uint32_t           layer_end,
        bool               include_token,
        bool               include_output) {
    if (!model) return 0;
    /* Under TP, the sharded map bytes are authoritative regardless of
     * how the caller frames the request (TP excludes real layer slicing,
     * so any slice request here is the session's full-range accounting). */
    if (!ssd_streaming && g_tp_shard_model_bytes != 0) {
        return g_tp_shard_model_bytes;
    }
    if (load_slice && weights) {
        ds4_model_map_span_vec spans;
        bool ok = false;
        if (ssd_streaming) {
            ok = weights_model_map_decode_static_slice_spans(weights,
                                                             layer_start,
                                                             layer_end,
                                                             include_token,
                                                             include_output,
                                                             &spans);
        } else {
            ok = weights_model_map_spans(weights,
                                         layer_start,
                                         layer_end,
                                         include_output,
                                         &spans);
        }
        if (ok) {
            const uint64_t bytes = model_map_span_vec_total_bytes(&spans);
            free(spans.v);
            if (bytes != 0) return bytes;
        }
    }
    if (!ssd_streaming) {
        if (g_tp_shard_model_bytes != 0) return g_tp_shard_model_bytes;
        return model->size;
    }
    const uint64_t active_bytes = glm_graph_streaming_active_model_bytes(weights);
    return active_bytes != 0 ? active_bytes : model->size;
}

static double glm_graph_memory_guard_default_reserve_gib(
        uint64_t budget_base,
        uint64_t model_bytes) {
    const double base_gib = glm_graph_bytes_to_gib(budget_base);
    const double model_gib = glm_graph_bytes_to_gib(model_bytes);
    if (base_gib >= 480.0 &&
        base_gib <= 640.0 &&
        model_gib >= base_gib * 0.80) {
        return 24.0;
    }
    return 32.0;
}

static bool glm_graph_memory_guard_for_compact_cap(
        const ds4_model   *model,
        const ds4_weights *weights,
        bool               ssd_streaming,
        bool               load_slice,
        uint32_t           layer_start,
        uint32_t           layer_end,
        bool               include_token,
        bool               include_output,
        uint32_t           ctx_size,
        uint32_t           compact_cap,
        uint64_t           transient_extra_bytes,
        const char        *phase) {
    if (!model || glm_graph_env_disabled("DS4_GLM_MEMORY_GUARD")) return true;

    const uint64_t host_bytes = glm_graph_host_memory_bytes();
    uint64_t budget_base = host_bytes;
    if (budget_base == 0) {
        budget_base = ds4_gpu_recommended_working_set_size();
    }
    if (budget_base == 0) return true;
    const uint64_t wired_limit = glm_graph_wired_limit_bytes();

    const uint32_t work_ctx =
        glm_graph_full_attention_cap(ctx_size, ssd_streaming);
    const ds4_context_memory mem = load_slice ?
        glm_graph_context_memory_estimate_for_compact_cap_slice(
                ctx_size,
                work_ctx,
                compact_cap,
                ssd_streaming,
                layer_start,
                layer_end) :
        glm_graph_context_memory_estimate_for_compact_cap(
                ctx_size,
                work_ctx,
                compact_cap,
                ssd_streaming);
    const uint64_t graph_bytes = mem.total_bytes;
    const uint64_t model_bytes =
        glm_graph_model_bytes_for_guard(model,
                                        weights,
                                        ssd_streaming,
                                        load_slice,
                                        layer_start,
                                        layer_end,
                                        include_token,
                                        include_output);
    uint64_t required = glm_graph_saturating_add_u64(model_bytes, graph_bytes);
    required = glm_graph_saturating_add_u64(required, transient_extra_bytes);

    const double fraction =
        glm_graph_env_double("DS4_GLM_MEMORY_GUARD_FRACTION", 0.99, 0.50, 1.00);
    double default_reserve_gib =
        glm_graph_memory_guard_default_reserve_gib(budget_base, model_bytes);
#ifdef DS4_ROCM_BUILD
    if (load_slice && !ssd_streaming) {
        /* The original fixed reserve protects Metal's shared host/GPU heap.
         * A resident ROCm layer slice already accounts its exact model spans
         * and owned graph state above. Keep proportional backend headroom for
         * driver and temporary allocations without rejecting viable UMA
         * slices merely because the heap is smaller than a high-memory Mac. */
        double rocm_reserve_gib = glm_graph_bytes_to_gib(budget_base) / 16.0;
        if (rocm_reserve_gib < 8.0) rocm_reserve_gib = 8.0;
        if (rocm_reserve_gib < default_reserve_gib) {
            default_reserve_gib = rocm_reserve_gib;
        }
    }
#endif
    const double reserve_gib =
        glm_graph_env_double("DS4_GLM_MEMORY_GUARD_RESERVE_GB",
                             default_reserve_gib,
                             0.0,
                             1024.0);
    const uint64_t fraction_budget = (uint64_t)((double)budget_base * fraction);
    const uint64_t reserve_bytes =
        (uint64_t)(reserve_gib * 1024.0 * 1024.0 * 1024.0);
    const uint64_t reserve_budget =
        reserve_bytes >= budget_base ? 0 : budget_base - reserve_bytes;
    uint64_t budget = fraction_budget;
    if (reserve_bytes != 0 && reserve_budget < budget) budget = reserve_budget;
    if (wired_limit != 0) {
        /* An explicitly raised iogpu.wired_limit_mb is the user granting
         * the GPU that much wired memory; it overrides the heuristics
         * (keep a small margin for non-model GPU allocations). */
        const uint64_t margin = 2ull * 1024ull * 1024ull * 1024ull;
        const uint64_t wired_budget =
            wired_limit > margin ? wired_limit - margin : wired_limit;
        if (wired_budget > budget) budget = wired_budget;
    }

    if (required <= budget) {
        const char *report = getenv("DS4_GLM_MEMORY_GUARD_REPORT");
        if (report && report[0]) {
            fprintf(stderr,
                    "ds4: GLM memory guard ctx=%u compact_cap=%u required=%.2f GiB "
                    "budget=%.2f GiB (model %.2f GiB, graph %.2f GiB, transient %.2f GiB)\n",
                    ctx_size,
                    mem.comp_cap,
                    glm_graph_bytes_to_gib(required),
                    glm_graph_bytes_to_gib(budget),
                    glm_graph_bytes_to_gib(model_bytes),
                    glm_graph_bytes_to_gib(graph_bytes),
                    glm_graph_bytes_to_gib(transient_extra_bytes));
            if (ssd_streaming && model_bytes != model->size) {
                fprintf(stderr,
                        "ds4:   GLM streaming guard uses active model span %.2f GiB "
                        "(full GGUF %.2f GiB)\n",
                        glm_graph_bytes_to_gib(model_bytes),
                        glm_graph_bytes_to_gib(model->size));
            } else if (load_slice && model_bytes != model->size) {
                fprintf(stderr,
                        "ds4:   GLM memory guard uses sliced model span %.2f GiB "
                        "(full GGUF %.2f GiB)\n",
                        glm_graph_bytes_to_gib(model_bytes),
                        glm_graph_bytes_to_gib(model->size));
            }
        }
        return true;
    }

    fprintf(stderr,
            "ds4: GLM memory guard refused ctx=%u compact_cap=%u %s\n",
            ctx_size,
            mem.comp_cap,
            phase ? phase : "before Metal graph allocation");
    if (ssd_streaming && model_bytes != model->size) {
        fprintf(stderr,
                "ds4:   streamed active model map: %.2f GiB "
                "(full GGUF %.2f GiB)\n",
                glm_graph_bytes_to_gib(model_bytes),
                glm_graph_bytes_to_gib(model->size));
    } else if (load_slice && model_bytes != model->size) {
        fprintf(stderr,
                "ds4:   sliced model map: %.2f GiB "
                "(full GGUF %.2f GiB)\n",
                glm_graph_bytes_to_gib(model_bytes),
                glm_graph_bytes_to_gib(model->size));
    } else {
        fprintf(stderr,
                "ds4:   model map: %.2f GiB\n",
                glm_graph_bytes_to_gib(model_bytes));
    }
    fprintf(stderr,
            "ds4:   graph cache/scratch: %.2f GiB "
            "(full KV %.2f GiB, compact DSA %.2f GiB, scratch %.2f GiB)\n",
            glm_graph_bytes_to_gib(graph_bytes),
            glm_graph_bytes_to_gib(mem.raw_bytes),
            glm_graph_bytes_to_gib(mem.compressed_bytes),
            glm_graph_bytes_to_gib(mem.scratch_bytes));
    fprintf(stderr,
            "ds4:   required model+graph: %.2f GiB; guard budget: %.2f GiB "
            "(base %.2f GiB, fraction %.2f, reserve %.2f GiB, transient %.2f GiB)\n",
            glm_graph_bytes_to_gib(required),
            glm_graph_bytes_to_gib(budget),
            glm_graph_bytes_to_gib(budget_base),
            fraction,
            reserve_gib,
            glm_graph_bytes_to_gib(transient_extra_bytes));
    fprintf(stderr,
            "ds4:   set DS4_GLM_MEMORY_GUARD=0 to bypass, use a smaller --ctx, "
            "or use SSD streaming\n");
    return false;
}

static bool glm_graph_memory_guard(
        const ds4_model   *model,
        const ds4_weights *weights,
        bool               ssd_streaming,
        uint32_t           ctx_size) {
    const uint32_t work_ctx =
        glm_graph_full_attention_cap(ctx_size, ssd_streaming);
    const uint32_t compact_cap =
        glm_graph_compact_cache_initial_cap(ctx_size, work_ctx);
    return glm_graph_memory_guard_for_compact_cap(
            model,
            weights,
            ssd_streaming,
            false,
            0,
            0,
            true,
            true,
            ctx_size,
            compact_cap,
            0,
            "before GLM graph allocation");
}

static bool glm_graph_memory_guard_with_transient(
        const ds4_model   *model,
        const ds4_weights *weights,
        bool               ssd_streaming,
        uint32_t           ctx_size,
        uint64_t           transient_extra_bytes,
        const char        *phase) {
    const uint32_t work_ctx =
        glm_graph_full_attention_cap(ctx_size, ssd_streaming);
    const uint32_t compact_cap =
        glm_graph_compact_cache_initial_cap(ctx_size, work_ctx);
    return glm_graph_memory_guard_for_compact_cap(
            model,
            weights,
            ssd_streaming,
            false,
            0,
            0,
            true,
            true,
            ctx_size,
            compact_cap,
            transient_extra_bytes,
            phase);
}

static bool glm_graph_memory_guard_slice(
        const ds4_model   *model,
        const ds4_weights *weights,
        bool               ssd_streaming,
        uint32_t           layer_start,
        uint32_t           layer_end,
        bool               include_token,
        bool               include_output,
        uint32_t           ctx_size) {
    const uint32_t work_ctx =
        glm_graph_full_attention_cap(ctx_size, ssd_streaming);
    const uint32_t compact_cap =
        glm_graph_compact_cache_initial_cap(ctx_size, work_ctx);
    return glm_graph_memory_guard_for_compact_cap(
            model,
            weights,
            ssd_streaming,
            true,
            layer_start,
            layer_end,
            include_token,
            include_output,
            ctx_size,
            compact_cap,
            0,
            "before GLM graph allocation");
}

static bool glm_graph_memory_guard_slice_with_transient(
        const ds4_model   *model,
        const ds4_weights *weights,
        bool               ssd_streaming,
        uint32_t           layer_start,
        uint32_t           layer_end,
        bool               include_token,
        bool               include_output,
        uint32_t           ctx_size,
        uint64_t           transient_extra_bytes,
        const char        *phase) {
    const uint32_t work_ctx =
        glm_graph_full_attention_cap(ctx_size, ssd_streaming);
    const uint32_t compact_cap =
        glm_graph_compact_cache_initial_cap(ctx_size, work_ctx);
    return glm_graph_memory_guard_for_compact_cap(
            model,
            weights,
            ssd_streaming,
            true,
            layer_start,
            layer_end,
            include_token,
            include_output,
            ctx_size,
            compact_cap,
            transient_extra_bytes,
            phase);
}

static uint32_t glm_graph_full_attention_cap(uint32_t ctx_size,
                                             bool     ssd_streaming) {
    uint32_t cap = ssd_streaming ?
        DS4_GLM_METAL_STREAMING_FULL_ATTN_CONTEXT :
        DS4_GLM_METAL_FULL_ATTN_DEFAULT_CONTEXT;
    if (ctx_size >= DS4_GLM_METAL_LONG_CONTEXT_THRESHOLD &&
        cap > DS4_GLM_METAL_LONG_CONTEXT_FULL_ATTN_CONTEXT) {
        cap = DS4_GLM_METAL_LONG_CONTEXT_FULL_ATTN_CONTEXT;
    }
    if (ctx_size > 0 && cap > ctx_size) cap = ctx_size;
    if (cap == 0) cap = 1;
    return cap;
}

static uint32_t glm_graph_full_prefill_layer_flush_interval(
        uint32_t n_tokens,
        uint32_t command_rows,
        bool     logits_requested) {
    /* Tiny logits-bearing passes (MTP verify, short prefills) must NOT
     * flush per layer: 76 command-buffer round-trips cost ~35ms while the
     * whole pass is ~70ms of GPU work. Real prefill chunks keep the
     * interactive per-layer flush. */
    return (n_tokens > DS4_GLM_METAL_FULL_ATTN_LAYER_FLUSH_CONTEXT ||
            command_rows > DS4_GLM_METAL_FULL_ATTN_LAYER_FLUSH_CONTEXT ||
            (logits_requested && n_tokens > 8u)) ? 1u : 0u;
}

static uint32_t glm_graph_prefill_progress_flush_interval(
        uint32_t                layer_flush_interval,
        uint32_t                n_tokens,
        ds4_session_progress_fn display_progress,
        uint32_t                work_total) {
    if (layer_flush_interval != 0) return layer_flush_interval;
    (void)n_tokens;
    (void)display_progress;
    (void)work_total;
    return 0;
}

static void glm_graph_report_prefill_display_progress(
        ds4_session_progress_fn display_progress,
        void                   *display_progress_ud,
        uint32_t                absolute_base,
        uint32_t                work_done_base,
        uint32_t                n_tokens,
        uint32_t                layer_done,
        uint32_t                normal_layers,
        uint32_t                work_total,
        bool                    allow_complete) {
    if (!display_progress || work_total == 0) return;

    uint64_t chunk_done = 0;
    if (normal_layers == 0 || layer_done >= normal_layers) {
        chunk_done = n_tokens;
    } else {
        chunk_done = (uint64_t)n_tokens * (uint64_t)layer_done /
                     (uint64_t)normal_layers;
    }

    uint64_t done = (uint64_t)work_done_base + chunk_done;
    if (done > (uint64_t)work_total) done = work_total;
    if (!allow_complete && done >= (uint64_t)work_total) {
        done = work_total > 0 ? (uint64_t)work_total - 1u : 0u;
    }
    display_progress(display_progress_ud,
                     "prefill_display",
                     (int)((uint64_t)absolute_base + done),
                     (int)((uint64_t)absolute_base + (uint64_t)work_total));
}

static bool glm_graph_small_prefill_stage_sync(
        uint32_t n_tokens,
        bool     logits_requested) {
    return logits_requested &&
           n_tokens > 0 &&
           n_tokens <= DS4_GLM_METAL_SMALL_PREFILL_STAGE_SYNC_TOKENS;
}

static uint32_t glm_graph_indexed_decode_split_min_block_rows(void) {
    return 32u;
}

static uint32_t glm_graph_indexed_decode_split_blocks(void) {
    const uint32_t block_rows = glm_graph_indexed_decode_split_min_block_rows();
    const uint32_t top_k = glm_graph_indexer_top_k_limit();
    return (top_k + block_rows - 1u) / block_rows;
}

static uint32_t glm_graph_indexed_decode_split_block_rows_for(uint32_t n_selected) {
    return n_selected <= 1024u ? 32u : 128u;
}

static bool glm_graph_indexed_decode_split_group8_available(uint32_t n_selected) {
    const uint32_t block_rows = glm_graph_indexed_decode_split_block_rows_for(n_selected);
    const uint32_t needed_blocks =
        block_rows != 0u ? (n_selected + block_rows - 1u) / block_rows : 0u;
    return n_selected > 512u &&
           block_rows > 0 &&
           needed_blocks > 0 &&
           needed_blocks <= glm_graph_indexed_decode_split_blocks() &&
           glm_graph_indexed_decode_split_blocks() <= 64u &&
           (DS4_N_HEAD % 8u) == 0 &&
           DS4_N_KV_LORA == 512u &&
           DS4_N_ROT == 64u &&
           glm_graph_compact_cache_is_f16();
}

static bool glm_graph_prefill_stage_sync_boundary(void) {
    if (ds4_gpu_end_commands() == 0) return false;
    return ds4_gpu_begin_commands() != 0;
}

static bool glm_graph_indexed_prefill_attention_boundary(void) {
#ifdef DS4_ROCM_BUILD
    /*
     * ROCm launches in this path are ordered on the default stream. The Metal
     * backend still needs the encoder flush, but on ROCm it is a full-device
     * synchronize and stalls every indexed-prefill layer.
     */
    return true;
#else
    return ds4_gpu_flush_encoder() != 0;
#endif
}

static DS4_MAYBE_UNUSED bool glm_graph_env_truthy(const char *env) {
    return env &&
           env[0] &&
           strcmp(env, "0") != 0 &&
           strcasecmp(env, "false") != 0 &&
           strcasecmp(env, "off") != 0 &&
           strcasecmp(env, "no") != 0;
}

static bool glm_graph_streaming_prefill_sync_each_layer(
        bool full_layer_prefill) {
#ifdef DS4_ROCM_BUILD
    /*
     * ROCm command boundaries are full device synchronizes. Compact streaming
     * prefill can keep queued default-stream work alive across layer mappings:
     * streamed model-range eviction synchronizes before freeing ranges, while
     * selected-expert cache reuse/eviction is protected by reuse events. The
     * full-layer expert cache is only double-buffered, so keep its old boundary.
     */
    if (full_layer_prefill) return true;
    const char *env = glm_graph_env_value(
            "DS4_ROCM_GLM_STREAMING_PREFILL_SYNC_EACH_LAYER",
            "DS4_METAL_GLM_STREAMING_PREFILL_SYNC_EACH_LAYER");
    if (!env) env = getenv("DS4_GLM_STREAMING_PREFILL_SYNC_EACH_LAYER");
    return glm_graph_env_truthy(env);
#else
    (void)full_layer_prefill;
    return true;
#endif
}

static bool glm_graph_indexed_prefill_batch_available(
        const ds4_glm_gpu_graph *g) {
    return g &&
           g->compact_cache_cap != 0 &&
           g->indexed_prefill_cap != 0 &&
           g->indexed_prefill_score_cap != 0 &&
           g->batch_indexer_q &&
           g->batch_indexer_weights &&
           g->batch_indexer_scores &&
           g->batch_indexer_selected &&
           g->batch_qk_low &&
           g->batch_attn_lora;
}

static bool glm_graph_indexed_prefill_batch_ready(
        const ds4_glm_gpu_graph *g,
        uint32_t                 pos) {
    return glm_graph_indexed_prefill_batch_available(g) &&
           (!g->full_kv_cache || pos >= g->ctx_cap);
}

static uint32_t glm_graph_limit_indexed_prefill_chunk(
        uint32_t pos,
        uint32_t chunk) {
    const uint32_t top_k = glm_graph_indexer_top_k_limit();
    if (pos < top_k) {
        const uint32_t bridge = top_k - pos;
        if (bridge != 0 && chunk > bridge) chunk = bridge;
    }
    return chunk;
}

static uint32_t glm_graph_indexed_prefill_chunk_tokens(
        uint32_t full_attention_cap,
        uint32_t compact_cap) {
    (void)full_attention_cap;
    uint32_t chunk = DS4_GLM_METAL_INDEXED_PREFILL_CHUNK_TOKENS;
    if (compact_cap > 0 && chunk > compact_cap) chunk = compact_cap;
    if (chunk == 0) chunk = 1;
    return chunk;
}

static uint32_t glm_graph_indexed_prefill_score_tokens(
        uint32_t indexed_prefill_cap,
        uint32_t compact_cap) {
    if (indexed_prefill_cap == 0 || compact_cap == 0) return 0;
    const uint32_t scratch_mb = DS4_GLM_METAL_INDEXED_PREFILL_SCORE_SCRATCH_MB;
    const uint64_t budget_bytes = (uint64_t)scratch_mb * 1024ull * 1024ull;
    uint64_t budget_rows = budget_bytes / ((uint64_t)compact_cap * sizeof(float));
    if (budget_rows == 0) budget_rows = 1;
    if (budget_rows > indexed_prefill_cap) budget_rows = indexed_prefill_cap;
    if (budget_rows > UINT32_MAX) budget_rows = UINT32_MAX;
    return (uint32_t)budget_rows;
}

static bool glm_graph_context_request(int ctx_size, uint32_t *ctx_out) {
    if (!ctx_out || ctx_size <= 0) return false;
    const uint32_t model_ctx = glm_graph_model_context_limit();
    if ((uint64_t)(uint32_t)ctx_size > (uint64_t)model_ctx) {
        fprintf(stderr,
                "ds4: GLM context %d exceeds model context %u\n",
                ctx_size,
                model_ctx);
        return false;
    }
    *ctx_out = (uint32_t)ctx_size;
    return true;
}

static bool glm_graph_span_fits_context(
        const ds4_glm_gpu_graph *g,
        uint32_t                 pos0,
        uint32_t                 n_tokens) {
    return g && n_tokens > 0 && pos0 < g->ctx_size && n_tokens <= g->ctx_size - pos0;
}

static bool glm_graph_span_fits_full_attention(
        const ds4_glm_gpu_graph *g,
        uint32_t                 pos0,
        uint32_t                 n_tokens) {
    return g && n_tokens > 0 && pos0 < g->ctx_cap && n_tokens <= g->ctx_cap - pos0;
}

static void glm_graph_log_full_attention_limit(
        const ds4_glm_gpu_graph *g,
        uint32_t                 pos0,
        uint32_t                 n_tokens) {
    const uint32_t end = pos0 + n_tokens;
    fprintf(stderr,
            "ds4: GLM Metal full-attention work cap is %u tokens; "
            "requested span [%u,%u) in ctx %u needs compact indexed attention\n",
            g ? g->ctx_cap : 0,
            pos0,
            end,
            g ? g->ctx_size : 0);
}

static bool glm_graph_tensor_layout(
        const ds4_tensor *t,
        uint32_t          type,
        uint32_t          ndim,
        uint64_t          dim0,
        uint64_t          dim1,
        uint64_t          dim2) {
    if (!t || t->type != type || t->ndim != ndim) return false;
    if (ndim > 0 && t->dim[0] != dim0) return false;
    if (ndim > 1 && t->dim[1] != dim1) return false;
    if (ndim > 2 && t->dim[2] != dim2) return false;
    return true;
}

static bool glm_graph_dense_tensor_layout(
        const ds4_tensor *t,
        uint32_t          ndim,
        uint64_t          dim0,
        uint64_t          dim1,
        uint64_t          dim2) {
    if (!t || !tensor_type_is_glm_dense_quant(t->type) || t->ndim != ndim) return false;
    if (ndim > 0 && t->dim[0] != dim0) return false;
    if (ndim > 1 && t->dim[1] != dim1) return false;
    if (ndim > 2 && t->dim[2] != dim2) return false;
    return true;
}

static bool glm_graph_layer_uses_generic_routed_moe(
        const ds4_layer_weights *l) {
    return l &&
           l->ffn_gate_exps &&
           l->ffn_up_exps &&
           l->ffn_down_exps &&
           l->ffn_gate_exps->type == DS4_TENSOR_IQ2_XXS;
}

static bool glm_graph_stream_map_token(
        ds4_glm_gpu_graph *g,
        const ds4_model   *model,
        const ds4_weights *weights) {
    if (!g || !g->ssd_streaming) return true;
    g->streaming_static_decode_map_current = false;
    return metal_graph_stream_map_token(model, weights);
}

static bool glm_graph_stream_layer_expert_cache_supported(
        const ds4_weights       *weights,
        const ds4_layer_weights *l,
        uint32_t                 il) {
    if (!weights || !l) return false;
    if (il < DS4_N_LEADING_DENSE) return true;
    return glm_stream_decode_experts_are_streamed(weights, l, il);
}

static bool glm_graph_stream_prefill_expert_addr_supported(
        const ds4_weights       *weights,
        const ds4_layer_weights *l,
        uint32_t                 il,
        uint32_t                 n_tokens) {
    if (il < DS4_N_LEADING_DENSE) return true;
    if (n_tokens <= 1) return false;
#ifdef DS4_ROCM_BUILD
    /*
     * ROCm selected-address batch prefill has pointer kernels for the
     * IQ2-gate/Q2-down generic path and the uniform Q2_K GLM path. Q4_K still
     * maps the full layer until matching pointer kernels exist.
     */
    if (glm_stream_selected_expert_cache_supported(l, il)) return true;
    return l &&
           l->ffn_gate_exps &&
           l->ffn_up_exps &&
           l->ffn_down_exps &&
           l->ffn_gate_exps->type == DS4_TENSOR_Q2_K &&
           l->ffn_up_exps->type == DS4_TENSOR_Q2_K &&
           l->ffn_down_exps->type == DS4_TENSOR_Q2_K &&
           glm_stream_expert_cache_addr_layout_supported(weights, l, il);
#else
    return glm_stream_expert_cache_addr_supported(weights, l, il);
#endif
}

static bool rocm_graph_glm_stream_prefill_full_layer_enabled(
        const ds4_glm_gpu_graph *g,
        const ds4_layer_weights *layer,
        uint32_t                 il,
        uint32_t                 n_tokens);

static bool glm_graph_stream_map_decode_layer(
        ds4_glm_gpu_graph *g,
        const ds4_model   *model,
        const ds4_weights *weights,
        uint32_t           il) {
    if (!g || !g->ssd_streaming) return true;
    g->streaming_static_decode_map_current = false;
    if (glm_graph_env_present("DS4_ROCM_GLM_STREAMING_DECODE_FULL_LAYER_MAP",
                              "DS4_METAL_GLM_STREAMING_DECODE_FULL_LAYER_MAP") ||
        getenv("DS4_GLM_STREAMING_DECODE_FULL_LAYER_MAP") != NULL) {
        return metal_graph_stream_map_layer(model, weights, il);
    }
    if (weights && il < DS4_N_LAYER &&
        glm_stream_resident_decode_layer_enabled(&weights->layer[il], il)) {
        return metal_graph_stream_map_layer(model, weights, il);
    }
    if (weights && il < DS4_N_LAYER &&
        glm_graph_stream_layer_expert_cache_supported(weights,
                                                      &weights->layer[il],
                                                      il)) {
        return metal_graph_stream_map_layer_decode(model, weights, il);
    }
    return metal_graph_stream_map_layer(model, weights, il);
}

static bool glm_graph_stream_map_prefill_layer(
        ds4_glm_gpu_graph *g,
        const ds4_model   *model,
        const ds4_weights *weights,
        uint32_t           il,
        uint32_t           n_tokens,
        bool               full_layer_prefill) {
    if (!g || !g->ssd_streaming) return true;
    g->streaming_static_decode_map_current = false;
    if (full_layer_prefill) {
        const char *trace = glm_graph_env_value("DS4_ROCM_STREAMING_MAP_TRACE",
                                                "DS4_METAL_STREAMING_MAP_TRACE");
        if (trace && trace[0] && strcmp(trace, "0") != 0) {
            fprintf(stderr,
                    "ds4: GLM SSD prefill map layer=%u tokens=%u mode=full-prefill\n",
                    il,
                    n_tokens);
        }
#ifdef DS4_ROCM_BUILD
        if (weights &&
            il < DS4_N_LAYER &&
            rocm_graph_glm_stream_prefill_full_layer_enabled(g,
                                                             &weights->layer[il],
                                                             il,
                                                             n_tokens)) {
            return metal_graph_stream_map_layer_decode(model, weights, il);
        }
#endif
        return metal_graph_stream_map_layer(model, weights, il);
    }
    const bool addr_supported =
        weights && il < DS4_N_LAYER &&
        glm_graph_stream_prefill_expert_addr_supported(weights,
                                                       &weights->layer[il],
                                                       il,
                                                       n_tokens);
    const char *trace = glm_graph_env_value("DS4_ROCM_STREAMING_MAP_TRACE",
                                            "DS4_METAL_STREAMING_MAP_TRACE");
    if (trace && trace[0] && strcmp(trace, "0") != 0) {
        fprintf(stderr,
                "ds4: GLM SSD prefill map layer=%u tokens=%u mode=%s\n",
                il,
                n_tokens,
                addr_supported ? "decode-expert-cache" : "full-layer");
    }
    if (addr_supported) {
        return metal_graph_stream_map_layer_decode(model, weights, il);
    }
    return metal_graph_stream_map_layer(model, weights, il);
}

#ifdef DS4_ROCM_BUILD
enum { DS4_GLM_STREAM_PREFILL_FULL_LAYER_MIN_TOKENS = 1024 };
#else
enum { DS4_GLM_STREAM_PREFILL_FULL_LAYER_MIN_TOKENS = 64 };
#endif

static uint32_t glm_graph_stream_prefill_full_layer_min_tokens(void) {
    const char *env = glm_graph_env_value(
            "DS4_ROCM_GLM_STREAMING_PREFILL_FULL_LAYER_MIN_TOKENS",
            "DS4_METAL_GLM_STREAMING_PREFILL_FULL_LAYER_MIN_TOKENS");
    if (!env) return DS4_GLM_STREAM_PREFILL_FULL_LAYER_MIN_TOKENS;
    char *end = NULL;
    errno = 0;
    unsigned long v = strtoul(env, &end, 10);
    if (end == env || errno != 0 || v == 0 || v > UINT32_MAX) {
        return DS4_GLM_STREAM_PREFILL_FULL_LAYER_MIN_TOKENS;
    }
    return (uint32_t)v;
}

static bool glm_graph_stream_prefill_full_layer_enabled(
        const ds4_glm_gpu_graph *g,
        uint32_t                 n_tokens) {
    if (!g || !g->ssd_streaming) return false;
    if (glm_graph_env_present(
                "DS4_ROCM_DISABLE_GLM_STREAMING_PREFILL_FULL_LAYER",
                "DS4_METAL_DISABLE_GLM_STREAMING_PREFILL_FULL_LAYER")) {
        return false;
    }
    if (glm_graph_env_present("DS4_ROCM_GLM_STREAMING_PREFILL_FULL_LAYER",
                              "DS4_METAL_GLM_STREAMING_PREFILL_FULL_LAYER")) {
        return true;
    }
    return n_tokens >= glm_graph_stream_prefill_full_layer_min_tokens();
}

static bool glm_graph_stream_prefill_full_layer_prepare_enabled(
        const ds4_glm_gpu_graph *g,
        bool                     full_layer_prefill) {
    return g &&
           g->ssd_streaming &&
           full_layer_prefill &&
           !glm_graph_env_present(
                "DS4_ROCM_DISABLE_GLM_STREAMING_PREFILL_FULL_LAYER_PREPARE",
                "DS4_METAL_DISABLE_GLM_STREAMING_PREFILL_FULL_LAYER_PREPARE");
}

#ifdef DS4_ROCM_BUILD
static bool rocm_graph_glm_stream_prefill_full_layer_enabled(
        const ds4_glm_gpu_graph *g,
        const ds4_layer_weights *layer,
        uint32_t                 il,
        uint32_t                 n_tokens) {
    return glm_graph_stream_prefill_full_layer_enabled(g, n_tokens) &&
           layer &&
           glm_stream_resident_decode_layer_supported(layer, il);
}

static bool rocm_graph_glm_stream_layer_expert_load_start_next(
        rocm_graph_stream_layer_expert_load *job,
        const ds4_glm_gpu_graph             *g,
        const ds4_model                     *model,
        const ds4_weights                   *weights,
        uint32_t                             first_il,
        uint32_t                             last_il,
        uint32_t                             n_tokens) {
    if (!job || !model || !weights || first_il > last_il) return true;
    if (job->active) return true;
    for (uint32_t il = first_il; il <= last_il && il < DS4_N_LAYER; il++) {
        const ds4_layer_weights *layer = &weights->layer[il];
        if (!rocm_graph_glm_stream_prefill_full_layer_enabled(g,
                                                              layer,
                                                              il,
                                                              n_tokens)) {
            continue;
        }
        uint64_t gate_expert_bytes = 0;
        uint64_t down_expert_bytes = 0;
        if (!rocm_graph_stream_layer_expert_bytes(layer,
                                                  &gate_expert_bytes,
                                                  &down_expert_bytes)) {
            return false;
        }
        return rocm_graph_stream_layer_expert_load_start(job,
                                                         model,
                                                         layer,
                                                         il,
                                                         gate_expert_bytes,
                                                         down_expert_bytes);
    }
    return true;
}

static bool rocm_graph_glm_stream_layer_expert_load_ready(
        rocm_graph_stream_layer_expert_load *job,
        const ds4_glm_gpu_graph             *g,
        const ds4_model                     *model,
        const ds4_weights                   *weights,
        uint32_t                             il,
        uint32_t                             n_tokens) {
    if (!model || !weights || il >= DS4_N_LAYER) return false;
    const ds4_layer_weights *layer = &weights->layer[il];
    if (!rocm_graph_glm_stream_prefill_full_layer_enabled(g,
                                                          layer,
                                                          il,
                                                          n_tokens)) {
        return true;
    }
    uint64_t gate_expert_bytes = 0;
    uint64_t down_expert_bytes = 0;
    if (!rocm_graph_stream_layer_expert_bytes(layer,
                                              &gate_expert_bytes,
                                              &down_expert_bytes)) {
        return false;
    }
    if (job && job->active) {
        if (job->il != il) {
            fprintf(stderr,
                    "ds4: GLM ROCm streaming full-layer expert load expected "
                    "layer %u but pending job is layer %u\n",
                    il,
                    job->il);
            return false;
        }
        return rocm_graph_stream_layer_expert_load_join(job);
    }
    return rocm_graph_stream_layer_expert_load_sync(model,
                                                    layer,
                                                    il,
                                                    gate_expert_bytes,
                                                    down_expert_bytes);
}
#else
static bool rocm_graph_glm_stream_prefill_full_layer_enabled(
        const ds4_glm_gpu_graph *g,
        const ds4_layer_weights *layer,
        uint32_t                 il,
        uint32_t                 n_tokens) {
    (void)g;
    (void)layer;
    (void)il;
    (void)n_tokens;
    return false;
}
#endif

static bool glm_graph_stream_map_output(
        ds4_glm_gpu_graph *g,
        const ds4_model   *model,
        const ds4_weights *weights) {
    if (!g || !g->ssd_streaming) return true;
    g->streaming_static_decode_map_current = false;
    return metal_graph_stream_map_output(model, weights);
}

static bool glm_graph_validate_expert_layout(
        const ds4_model  *model,
        const ds4_tensor *gate,
        const ds4_tensor *up,
        const ds4_tensor *down,
        uint64_t         *gate_row_bytes,
        uint64_t         *up_row_bytes,
        uint64_t         *down_row_bytes) {
    if (!gate || !up || !down) return false;
    if (!glm_graph_gate_pair_type_supported(gate->type, up->type) ||
        !glm_graph_down_type_supported(down->type) ||
        gate->ndim != 3 || up->ndim != 3 || down->ndim != 3 ||
        gate->dim[0] != DS4_N_EMBD ||
        gate->dim[1] != DS4_N_FF_EXP ||
        gate->dim[2] != DS4_N_EXPERT ||
        up->dim[0] != DS4_N_EMBD ||
        up->dim[1] != DS4_N_FF_EXP ||
        up->dim[2] != DS4_N_EXPERT ||
        down->dim[0] != DS4_N_FF_EXP ||
        down->dim[1] != DS4_N_EMBD ||
        down->dim[2] != DS4_N_EXPERT) {
        return false;
    }

    uint64_t gate_in = 0, gate_out = 0;
    uint64_t up_in = 0, up_out = 0;
    uint64_t down_in = 0, down_out = 0;
    (void)tensor_expert_bytes(model, gate, 0, &gate_in, &gate_out, gate_row_bytes);
    (void)tensor_expert_bytes(model, up, 0, &up_in, &up_out, up_row_bytes);
    (void)tensor_expert_bytes(model, down, 0, &down_in, &down_out, down_row_bytes);
    return gate_in == DS4_N_EMBD &&
           up_in == DS4_N_EMBD &&
           down_in == DS4_N_FF_EXP &&
           gate_out == DS4_N_FF_EXP &&
           up_out == DS4_N_FF_EXP &&
           down_out == DS4_N_EMBD;
}

static bool glm_graph_validate_layer_layout(
        const ds4_model         *model,
        const ds4_layer_weights *l,
        uint32_t                 il,
        uint64_t                 q_dim,
        uint64_t                 q_nope,
        uint64_t                 heads_dim,
        uint64_t                *kv_raw_dim_out,
        uint64_t                *dense_hidden_max) {
    if (!l) return false;
    const uint64_t kv_raw_dim = l->attn_kv_a_mqa ? l->attn_kv_a_mqa->dim[1] : 0;
    const uint64_t indexer_q_dim = (uint64_t)DS4_N_INDEXER_HEAD * DS4_N_INDEXER_HEAD_DIM;
    if (!glm_graph_tensor_layout(l->attn_norm, DS4_TENSOR_F32, 1, DS4_N_EMBD, 0, 0) ||
        !glm_graph_dense_tensor_layout(l->attn_q_a, 2, DS4_N_EMBD, DS4_N_LORA_Q, 0) ||
        !glm_graph_tensor_layout(l->attn_q_a_norm, DS4_TENSOR_F32, 1, DS4_N_LORA_Q, 0, 0) ||
        !glm_graph_dense_tensor_layout(l->attn_q_b, 2, DS4_N_LORA_Q, q_dim, 0) ||
        !l->attn_kv_a_mqa ||
        !tensor_type_is_glm_dense_quant(l->attn_kv_a_mqa->type) ||
        l->attn_kv_a_mqa->ndim != 2 ||
        l->attn_kv_a_mqa->dim[0] != DS4_N_EMBD ||
        kv_raw_dim < (uint64_t)DS4_N_KV_LORA + DS4_N_ROT ||
        !glm_graph_tensor_layout(l->attn_kv_a_norm, DS4_TENSOR_F32, 1, DS4_N_KV_LORA, 0, 0) ||
        !glm_graph_dense_tensor_layout(l->attn_k_b, 3, q_nope, DS4_N_KV_LORA, DS4_N_HEAD) ||
        !glm_graph_dense_tensor_layout(l->attn_v_b, 3, DS4_N_KV_LORA, DS4_N_VALUE_MLA, DS4_N_HEAD) ||
        !glm_graph_dense_tensor_layout(l->attn_output, 2, heads_dim, DS4_N_EMBD, 0) ||
        !glm_graph_dense_tensor_layout(l->indexer_attn_q_b, 2, DS4_N_LORA_Q, indexer_q_dim, 0) ||
        !glm_graph_dense_tensor_layout(l->indexer_attn_k, 2, DS4_N_EMBD, DS4_N_INDEXER_HEAD_DIM, 0) ||
        !glm_graph_tensor_layout(l->indexer_k_norm, DS4_TENSOR_F32, 1, DS4_N_INDEXER_HEAD_DIM, 0, 0) ||
        !glm_graph_tensor_layout(l->indexer_k_norm_b, DS4_TENSOR_F32, 1, DS4_N_INDEXER_HEAD_DIM, 0, 0) ||
        !glm_graph_tensor_layout(l->indexer_proj, DS4_TENSOR_F32, 2, DS4_N_EMBD, DS4_N_INDEXER_HEAD, 0) ||
        !glm_graph_tensor_layout(l->ffn_norm, DS4_TENSOR_F32, 1, DS4_N_EMBD, 0, 0)) {
        fprintf(stderr, "ds4: GLM Metal graph found unexpected attention layout in layer %u\n", il);
        return false;
    }
    if (kv_raw_dim > *kv_raw_dim_out) *kv_raw_dim_out = kv_raw_dim;

    if (il < DS4_N_LEADING_DENSE) {
        const uint64_t hidden = l->ffn_gate ? l->ffn_gate->dim[1] : 0;
        if (!l->ffn_gate ||
            !glm_graph_dense_tensor_layout(l->ffn_gate, 2, DS4_N_EMBD, hidden, 0) ||
            !glm_graph_dense_tensor_layout(l->ffn_up, 2, DS4_N_EMBD, hidden, 0) ||
            !glm_graph_dense_tensor_layout(l->ffn_down, 2, hidden, DS4_N_EMBD, 0)) {
            fprintf(stderr, "ds4: GLM Metal graph found unexpected dense FFN layout in layer %u\n", il);
            return false;
        }
        if (hidden > *dense_hidden_max) *dense_hidden_max = hidden;
    } else {
        uint64_t gate_row_bytes = 0, up_row_bytes = 0, down_row_bytes = 0;
        if (!glm_graph_tensor_layout(l->ffn_gate_inp, DS4_TENSOR_F32, 2, DS4_N_EMBD, DS4_N_EXPERT, 0) ||
            !glm_graph_tensor_layout(l->ffn_exp_probs_b, DS4_TENSOR_F32, 1, DS4_N_EXPERT, 0, 0) ||
            !glm_graph_validate_expert_layout(model,
                                              l->ffn_gate_exps,
                                              l->ffn_up_exps,
                                              l->ffn_down_exps,
                                              &gate_row_bytes,
                                              &up_row_bytes,
                                              &down_row_bytes) ||
            !glm_graph_dense_tensor_layout(l->ffn_gate_shexp, 2, DS4_N_EMBD, DS4_N_FF_EXP, 0) ||
            !glm_graph_dense_tensor_layout(l->ffn_up_shexp, 2, DS4_N_EMBD, DS4_N_FF_EXP, 0) ||
            !glm_graph_dense_tensor_layout(l->ffn_down_shexp, 2, DS4_N_FF_EXP, DS4_N_EMBD, 0)) {
            fprintf(stderr, "ds4: GLM Metal graph found unexpected sparse FFN layout in layer %u\n", il);
            return false;
        }
        (void)gate_row_bytes;
        (void)up_row_bytes;
        (void)down_row_bytes;
    }
    return true;
}

static bool glm_graph_validate_layout(
        const ds4_model   *model,
        const ds4_weights *weights,
        ds4_glm_gpu_graph *g,
        uint32_t           layer_start,
        uint32_t           layer_end,
        bool               require_token_embd,
        bool               require_output_head) {
    if (!model || !weights || !g) return false;
    const uint32_t normal_layers = glm_graph_normal_layer_count();
    if (normal_layers == 0 || DS4_N_ROT >= DS4_N_KEY_MLA) {
        fprintf(stderr, "ds4: GLM Metal graph found unsupported layer/key dimensions\n");
        return false;
    }
    if (layer_end == UINT32_MAX) layer_end = normal_layers - 1u;
    if (layer_start > layer_end || layer_end >= normal_layers) {
        fprintf(stderr,
                "ds4: GLM Metal graph found invalid layer slice %u:%u for %u normal layers\n",
                layer_start,
                layer_end,
                normal_layers);
        return false;
    }

    g->has_token_embd = weights->token_embd != NULL;
    g->has_output_head = weights_have_output_head(weights);
    if (require_token_embd && !g->has_token_embd) {
        fprintf(stderr, "ds4: GLM Metal graph layer slice requires token embeddings\n");
        return false;
    }
    if (g->has_token_embd &&
        !glm_graph_dense_tensor_layout(weights->token_embd, 2,
                                       DS4_N_EMBD, DS4_N_VOCAB, 0)) {
        fprintf(stderr, "ds4: GLM Metal graph found unexpected token embedding layout\n");
        return false;
    }
    if (require_output_head && !g->has_output_head) {
        fprintf(stderr, "ds4: GLM Metal graph layer slice requires the output head\n");
        return false;
    }
    if (weights_have_partial_output_head(weights) && !g->has_output_head) {
        fprintf(stderr, "ds4: GLM Metal graph found partial output head\n");
        return false;
    }
    if (g->has_output_head &&
        (!glm_graph_tensor_layout(weights->output_norm, DS4_TENSOR_F32, 1, DS4_N_EMBD, 0, 0) ||
         !glm_graph_dense_tensor_layout(weights->output, 2,
                                        DS4_N_EMBD, DS4_N_VOCAB, 0))) {
        fprintf(stderr, "ds4: GLM Metal graph found unexpected output head layout\n");
        return false;
    }

    g->normal_layers = normal_layers;
    g->layer_start = layer_start;
    g->layer_end = layer_end;
    g->layer_count = layer_end - layer_start + 1u;
    g->q_dim = (uint64_t)DS4_N_HEAD * DS4_N_KEY_MLA;
    g->q_nope = (uint64_t)DS4_N_KEY_MLA - DS4_N_ROT;
    g->heads_dim = (uint64_t)DS4_N_HEAD * DS4_N_VALUE_MLA;
    g->dense_hidden_max = DS4_N_FF_EXP;
    g->kv_raw_dim = 0;
    for (uint32_t il = g->layer_start; il <= g->layer_end; il++) {
        if (!glm_graph_validate_layer_layout(model,
                                             &weights->layer[il],
                                             il,
                                             g->q_dim,
                                             g->q_nope,
                                             g->heads_dim,
                                             &g->kv_raw_dim,
                                             &g->dense_hidden_max)) {
            return false;
        }
        if (glm_graph_layer_uses_generic_routed_moe(&weights->layer[il])) {
            g->generic_routed_moe = true;
        }
    }
    const uint64_t sparse_mid_elems = (uint64_t)DS4_N_EXPERT_USED * DS4_N_FF_EXP;
    g->ffn_mid_elems =
        g->dense_hidden_max > sparse_mid_elems ? g->dense_hidden_max : sparse_mid_elems;
    return g->layer_count > 0 &&
           g->kv_raw_dim >= (uint64_t)DS4_N_KV_LORA + DS4_N_ROT;
}

/* Per-tier decode scratch keeps layer kernels off peer-mapped work buffers. */
#define DS4_GLM_WS_FIELDS(X) \
    X(cur) X(next) X(attn_norm) X(q_rank) X(q_rank_norm) X(q) X(kv_raw) \
    X(kv_norm) X(k_nope) X(value) X(heads) X(attn_out) X(after_attn) \
    X(ffn_norm) X(ffn_gate) X(ffn_up) X(ffn_mid) X(ffn_out) X(ffn_sum) \
    X(router_logits) X(router_probs) X(router_selected) X(router_weights) \
    X(indexer_k) X(indexer_q) X(indexer_weights) X(indexer_scores) \
    X(indexer_selected) X(qk_low)

static ds4_gpu_tensor **glm_graph_ws_slot(ds4_glm_gpu_graph *g, int i) {
    int n = 0;
#define DS4_GLM_WS_SLOT_CASE(field) if (n++ == i) return &g->field;
    DS4_GLM_WS_FIELDS(DS4_GLM_WS_SLOT_CASE)
#undef DS4_GLM_WS_SLOT_CASE
    return NULL;
}

static void glm_graph_ws_free(ds4_glm_gpu_graph *g) {
    if (!g || !g->ws_ready) return;
    for (int i = 0; i < DS4_GLM_WS_SLOTS; i++) {
        ds4_gpu_tensor **slot = glm_graph_ws_slot(g, i);
        if (slot) *slot = g->ws_orig[i];
    }
    for (int tier = 0; tier < DS4_MAX_GPUS; tier++) {
        for (int i = 0; i < DS4_GLM_WS_SLOTS; i++) {
            ds4_gpu_tensor_free(g->ws_mirror[tier][i]);
            g->ws_mirror[tier][i] = NULL;
        }
    }
    g->ws_ready = 0;
    g->ws_tier = -1;
}

static void glm_graph_ws_init(ds4_glm_gpu_graph *g) {
    g->ws_ready = 0;
    g->ws_tier = -1;
    if (!g->placement) return;
    bool used[DS4_MAX_GPUS] = { false };
    if (g->placement[0] >= 0 && g->placement[0] < DS4_MAX_GPUS) {
        used[g->placement[0]] = true;
    }
    for (uint32_t il = g->layer_start; il <= g->layer_end; il++) {
        const int tier = g->placement[il + 1u];
        if (tier >= 0 && tier < DS4_MAX_GPUS) used[tier] = true;
    }
    for (int i = 0; i < DS4_GLM_WS_SLOTS; i++) {
        ds4_gpu_tensor **slot = glm_graph_ws_slot(g, i);
        if (!slot) return;
        g->ws_orig[i] = *slot;
    }
    for (int tier = 0; tier < DS4_MAX_GPUS; tier++) {
        if (!used[tier]) continue;
        for (int i = 0; i < DS4_GLM_WS_SLOTS; i++) {
            const ds4_gpu_tensor *orig = g->ws_orig[i];
            if (!orig) continue;
            g->ws_mirror[tier][i] =
                ds4_gpu_tensor_alloc_ptr_on(tier, ds4_gpu_tensor_bytes(orig));
            if (!g->ws_mirror[tier][i]) {
                fprintf(stderr,
                        "ds4: GLM per-tier working-set alloc failed (tier %d); "
                        "falling back to base buffers\n",
                        tier);
                g->ws_ready = 1;
                glm_graph_ws_free(g);
                return;
            }
        }
    }
    g->ws_ready = 1;
}

static bool glm_graph_ws_switch(ds4_glm_gpu_graph *g,
                                int tier,
                                bool carry_hidden) {
    if (!g->placement) return true;
    if (tier < 0 || tier >= DS4_MAX_GPUS ||
        ds4_gpu_set_current_device_fenced(tier) != 0) {
        return false;
    }
    if (!g->ws_ready) return true;
    if (g->ws_tier == tier) return true;
    ds4_gpu_tensor *old_cur = g->cur;
    for (int i = 0; i < DS4_GLM_WS_SLOTS; i++) {
        ds4_gpu_tensor **slot = glm_graph_ws_slot(g, i);
        if (slot && g->ws_mirror[tier][i]) {
            *slot = g->ws_mirror[tier][i];
        }
    }
    if (carry_hidden && old_cur && g->cur && g->cur != old_cur) {
        const uint64_t bytes = (uint64_t)DS4_N_EMBD * sizeof(float);
        if (ds4_gpu_tensor_copy_async(g->cur, old_cur, bytes) == 0) {
            return false;
        }
    }
    g->ws_tier = tier;
    return true;
}

#define DS4_GLM_VERIFY_WS_FIELDS(X) \
    X(batch_cur) X(batch_next) X(batch_attn_norm) X(batch_q_rank) \
    X(batch_q_rank_norm) X(batch_q) X(batch_indexer_k) X(batch_kv_raw) \
    X(batch_kv_norm) X(batch_qk_low) X(batch_attn_lora) \
    X(batch_indexer_selected) X(batch_heads) X(batch_attn_out) \
    X(batch_after_attn) X(batch_ffn_norm) X(batch_ffn_gate) X(batch_ffn_up) \
    X(batch_shared_mid) X(batch_ffn_mid) X(batch_routed_gate) \
    X(batch_routed_up) X(batch_routed_down) X(batch_ffn_out) \
    X(batch_router_logits) X(batch_router_probs) X(batch_router_selected) \
    X(batch_router_weights)

static ds4_gpu_tensor **glm_graph_verify_ws_slot(ds4_glm_gpu_graph *g,
                                                  int i) {
    int n = 0;
#define DS4_GLM_VERIFY_WS_SLOT_CASE(field) if (n++ == i) return &g->field;
    DS4_GLM_VERIFY_WS_FIELDS(DS4_GLM_VERIFY_WS_SLOT_CASE)
#undef DS4_GLM_VERIFY_WS_SLOT_CASE
    return NULL;
}

static void glm_graph_verify_ws_restore(ds4_glm_gpu_graph *g) {
    if (!g || !g->verify_ws_ready) return;
    for (int i = 0; i < DS4_GLM_VERIFY_WS_SLOTS; i++) {
        ds4_gpu_tensor **slot = glm_graph_verify_ws_slot(g, i);
        if (slot) *slot = g->verify_ws_orig[i];
    }
    g->verify_ws_tier = -1;
}

static void glm_graph_verify_ws_free(ds4_glm_gpu_graph *g) {
    if (!g || !g->verify_ws_ready) return;
    glm_graph_verify_ws_restore(g);
    for (int tier = 0; tier < DS4_MAX_GPUS; tier++) {
        for (int i = 0; i < DS4_GLM_VERIFY_WS_SLOTS; i++) {
            ds4_gpu_tensor_free(g->verify_ws_mirror[tier][i]);
            g->verify_ws_mirror[tier][i] = NULL;
        }
    }
    memset(g->verify_ws_orig, 0, sizeof(g->verify_ws_orig));
    g->verify_ws_ready = 0;
}

static bool glm_graph_verify_ws_init(ds4_glm_gpu_graph *g) {
    if (!g) return false;
    if (g->verify_ws_ready) return true;
    if (!g->placement) {
        g->verify_ws_ready = 1;
        g->verify_ws_tier = -1;
        return true;
    }
    bool used[DS4_MAX_GPUS] = { false };
    for (uint32_t il = g->layer_start; il <= g->layer_end; il++) {
        const int tier = g->placement[il + 1u];
        if (tier < 0 || tier >= DS4_MAX_GPUS) return false;
        used[tier] = true;
    }
    for (int i = 0; i < DS4_GLM_VERIFY_WS_SLOTS; i++) {
        ds4_gpu_tensor **slot = glm_graph_verify_ws_slot(g, i);
        if (!slot) return false;
        g->verify_ws_orig[i] = *slot;
    }
    for (int tier = 0; tier < DS4_MAX_GPUS; tier++) {
        if (!used[tier]) continue;
        for (int i = 0; i < DS4_GLM_VERIFY_WS_SLOTS; i++) {
            const ds4_gpu_tensor *orig = g->verify_ws_orig[i];
            if (!orig) continue;
            const uint32_t cap = (i >= 9 && i <= 11) ?
                g->indexed_prefill_cap : g->ctx_cap;
            const uint64_t orig_bytes = ds4_gpu_tensor_bytes(orig);
            if (cap == 0 || orig_bytes % cap != 0 ||
                orig_bytes / cap > UINT64_MAX / 2u) {
                g->verify_ws_ready = 1;
                glm_graph_verify_ws_free(g);
                return false;
            }
            const uint64_t bytes = (orig_bytes / cap) * 2u;
            g->verify_ws_mirror[tier][i] =
                ds4_gpu_tensor_alloc_ptr_on(tier, bytes);
            if (!g->verify_ws_mirror[tier][i]) {
                g->verify_ws_ready = 1;
                glm_graph_verify_ws_free(g);
                return false;
            }
        }
    }
    g->verify_ws_ready = 1;
    g->verify_ws_tier = -1;
    return true;
}

static bool glm_graph_verify_ws_switch(ds4_glm_gpu_graph *g,
                                       int tier,
                                       bool carry_hidden,
                                       uint32_t n_rows) {
    if (!g || n_rows == 0 || n_rows > 2) return false;
    if (!g->placement) return true;
    if (!glm_graph_verify_ws_init(g) ||
        tier < 0 || tier >= DS4_MAX_GPUS ||
        ds4_gpu_set_current_device_fenced(tier) != 0) {
        return false;
    }
    if (g->verify_ws_tier == tier) return true;
    ds4_gpu_tensor *old_cur = g->batch_cur;
    for (int i = 0; i < DS4_GLM_VERIFY_WS_SLOTS; i++) {
        ds4_gpu_tensor **slot = glm_graph_verify_ws_slot(g, i);
        if (slot && g->verify_ws_mirror[tier][i]) {
            *slot = g->verify_ws_mirror[tier][i];
        }
    }
    if (carry_hidden && old_cur && g->batch_cur && old_cur != g->batch_cur) {
        const uint64_t bytes =
            (uint64_t)n_rows * DS4_N_EMBD * sizeof(float);
        if (ds4_gpu_tensor_copy_async(g->batch_cur, old_cur, bytes) == 0) {
            return false;
        }
    }
    g->verify_ws_tier = tier;
    return true;
}
#undef DS4_GLM_VERIFY_WS_FIELDS

static void glm_graph_free(ds4_glm_gpu_graph *g) {
    glm_graph_verify_ws_free(g);
    glm_graph_ws_free(g);
    if (!g) return;
    ds4_gpu_tensor_free(g->mtp_kv_lora_cache);
    ds4_gpu_tensor_free(g->mtp_k_rope_cache);
    ds4_gpu_tensor_free(g->mtp_concat);
    ds4_gpu_tensor_free(g->mtp_selected);
    free(g->mtp_logits_host);
    g->mtp_kv_lora_cache = NULL;
    g->mtp_k_rope_cache = NULL;
    g->mtp_concat = NULL;
    g->mtp_selected = NULL;
    g->mtp_logits_host = NULL;
    g->mtp_ready = 0;
    for (uint32_t il = 0; il < DS4_MAX_LAYER; il++) {
        ds4_gpu_tensor_free(g->layer_indexer_key_cache[il]);
        ds4_gpu_tensor_free(g->layer_k_rope_cache[il]);
        ds4_gpu_tensor_free(g->layer_kv_lora_cache[il]);
        ds4_gpu_tensor_free(g->layer_value_cache[il]);
        ds4_gpu_tensor_free(g->layer_key_cache[il]);
    }
    ds4_gpu_tensor_free(g->logits);
    ds4_gpu_tensor_free(g->batch_router_weights);
    ds4_gpu_tensor_free(g->prefill_seed_router_selected);
    ds4_gpu_tensor_free(g->batch_router_selected);
    ds4_gpu_tensor_free(g->batch_router_probs);
    ds4_gpu_tensor_free(g->batch_router_logits);
    ds4_gpu_tensor_free(g->batch_routed_down);
    ds4_gpu_tensor_free(g->batch_routed_up);
    ds4_gpu_tensor_free(g->batch_routed_gate);
    ds4_gpu_tensor_free(g->batch_ffn_out);
    ds4_gpu_tensor_free(g->batch_ffn_mid);
    ds4_gpu_tensor_free(g->batch_shared_mid);
    ds4_gpu_tensor_free(g->batch_ffn_up);
    ds4_gpu_tensor_free(g->batch_ffn_gate);
    ds4_gpu_tensor_free(g->batch_ffn_norm);
    ds4_gpu_tensor_free(g->batch_after_attn);
    ds4_gpu_tensor_free(g->batch_attn_out);
    ds4_gpu_tensor_free(g->batch_heads);
    ds4_gpu_tensor_free(g->batch_value);
    ds4_gpu_tensor_free(g->batch_k_nope);
    ds4_gpu_tensor_free(g->batch_kv_norm);
    ds4_gpu_tensor_free(g->batch_kv_raw);
    ds4_gpu_tensor_free(g->batch_attn_lora);
    ds4_gpu_tensor_free(g->batch_qk_low);
    ds4_gpu_tensor_free(g->batch_indexer_selected);
    ds4_gpu_tensor_free(g->batch_indexer_scores);
    ds4_gpu_tensor_free(g->batch_indexer_weights);
    ds4_gpu_tensor_free(g->batch_indexer_q);
    ds4_gpu_tensor_free(g->batch_indexer_k);
    ds4_gpu_tensor_free(g->batch_q);
    ds4_gpu_tensor_free(g->batch_q_rank_norm);
    ds4_gpu_tensor_free(g->batch_q_rank);
    ds4_gpu_tensor_free(g->batch_attn_norm);
    ds4_gpu_tensor_free(g->batch_next);
    ds4_gpu_tensor_free(g->batch_cur);
    ds4_gpu_tensor_free(g->prefill_tokens);
    ds4_gpu_tensor_free(g->output_norm);
    ds4_gpu_tensor_free(g->router_weights);
    ds4_gpu_tensor_free(g->router_selected);
    ds4_gpu_tensor_free(g->router_probs);
    ds4_gpu_tensor_free(g->router_logits);
    ds4_gpu_tensor_free(g->ffn_sum);
    ds4_gpu_tensor_free(g->ffn_out);
    ds4_gpu_tensor_free(g->routed_down);
    ds4_gpu_tensor_free(g->routed_up);
    ds4_gpu_tensor_free(g->tp_bounce_out);
    ds4_gpu_tensor_free(g->tp_bounce_in);
    ds4_gpu_tensor_free(g->routed_gate);
    ds4_gpu_tensor_free(g->ffn_mid);
    ds4_gpu_tensor_free(g->ffn_up);
    ds4_gpu_tensor_free(g->ffn_gate);
    ds4_gpu_tensor_free(g->ffn_norm);
    ds4_gpu_tensor_free(g->after_attn);
    ds4_gpu_tensor_free(g->attn_out);
    ds4_gpu_tensor_free(g->heads);
    ds4_gpu_tensor_free(g->value);
    ds4_gpu_tensor_free(g->k_nope);
    ds4_gpu_tensor_free(g->kv_norm);
    ds4_gpu_tensor_free(g->kv_raw);
    ds4_gpu_tensor_free(g->attn_partial_ms);
    ds4_gpu_tensor_free(g->attn_partial_lora);
    ds4_gpu_tensor_free(g->qk_low);
    ds4_gpu_tensor_free(g->indexer_selected);
    ds4_gpu_tensor_free(g->indexer_scores);
    ds4_gpu_tensor_free(g->indexer_weights);
    ds4_gpu_tensor_free(g->indexer_q);
    ds4_gpu_tensor_free(g->indexer_k);
    ds4_gpu_tensor_free(g->q);
    ds4_gpu_tensor_free(g->q_rank_norm);
    ds4_gpu_tensor_free(g->q_rank);
    ds4_gpu_tensor_free(g->attn_norm);
    ds4_gpu_tensor_free(g->next);
    ds4_gpu_tensor_free(g->cur);
    memset(g, 0, sizeof(*g));
}

static bool glm_graph_ensure_compact_cache(
        const ds4_glm_gpu_graph *g,
        uint32_t                 needed_rows) {
    if (!g || needed_rows == 0) return false;
    if (needed_rows <= g->ctx_cap && g->compact_cache_cap == 0) return true;
    if (needed_rows <= g->compact_cache_cap) return true;
    fprintf(stderr,
            "ds4: GLM compact DSA cache capacity %u is smaller than required row %u "
            "(ctx=%u, full_cap=%u)\n",
            g->compact_cache_cap,
            needed_rows,
            g->ctx_size,
            g->ctx_cap);
    return false;
}

static bool glm_graph_warm_compact_indexer_store(
        ds4_glm_gpu_graph *g,
        const ds4_model   *model,
        const ds4_weights *weights,
        uint32_t           warm_pos) {
    if (!g || !model || !weights) return false;
    if (g->compact_cache_cap == 0 || g->indexer_full_layers == 0) return true;
    if (!g->indexer_k) return false;
    if (warm_pos >= g->compact_cache_cap) warm_pos = g->compact_cache_cap - 1u;

    if (ds4_gpu_tensor_fill_f32(g->indexer_k,
                                0.0f,
                                DS4_N_INDEXER_HEAD_DIM) == 0) {
        return false;
    }

    const bool profile = false;
    const double t0 = 0.0;
    bool ok = ds4_gpu_begin_commands() != 0;
    uint32_t warmed = 0;
    for (uint32_t il = g->layer_start; ok && il <= g->layer_end; il++) {
        if (!glm_graph_layer_uses_full_indexer(il)) continue;
        const ds4_layer_weights *l = &weights->layer[il];
        if (!g->layer_indexer_key_cache[il] ||
            !l->indexer_k_norm ||
            !l->indexer_k_norm_b) {
            ok = false;
            break;
        }
        const float rope_base = layer_rope_freq_base(il);
        const float rope_scale = layer_rope_freq_scale(il);
        ok = ds4_gpu_glm_store_indexer_k_tensor(
                g->layer_indexer_key_cache[il],
                g->indexer_k,
                model->map,
                model->size,
                l->indexer_k_norm->abs_offset,
                l->indexer_k_norm_b->abs_offset,
                warm_pos,
                1,
                g->compact_cache_cap,
                DS4_N_INDEXER_HEAD_DIM,
                DS4_N_ROT,
                0,
                1.0e-6f,
                rope_base,
                rope_scale,
                0.0f,
                1.0f,
                DS4_ROPE_YARN_BETA_FAST,
                DS4_ROPE_YARN_BETA_SLOW,
                glm_graph_compact_cache_is_f16()) != 0;
        if (ok) warmed++;
    }
    if (ok) ok = ds4_gpu_end_commands() != 0;
    else (void)ds4_gpu_synchronize();

    if (profile) {
        fprintf(stderr,
                "ds4: GLM compact indexer warmup pos=%u layers=%u %.3f ms\n",
                warm_pos,
                warmed,
                (now_sec() - t0) * 1000.0);
    }
    return ok;
}

static bool glm_graph_alloc_slice(
        ds4_glm_gpu_graph *g,
        const ds4_model   *model,
        const ds4_weights *weights,
        int                ctx_size,
        bool               ssd_streaming,
        bool               ssd_streaming_cold,
        uint64_t           streaming_transient_guard_bytes,
        uint32_t           layer_start,
        uint32_t           layer_end,
        bool               require_token_embd,
        bool               require_output_head) {
    if (!g || !model || !weights || ctx_size <= 0) return false;
    const int *placement = g->placement;
    memset(g, 0, sizeof(*g));
    g->placement = placement;
    g->ssd_streaming = ssd_streaming;
    g->ssd_streaming_cold = ssd_streaming_cold;

    if (!glm_graph_context_request(ctx_size, &g->ctx_size)) return false;
    if (!glm_graph_memory_guard_slice_with_transient(
                model,
                weights,
                g->ssd_streaming,
                layer_start,
                layer_end,
                require_token_embd,
                require_output_head,
                g->ctx_size,
                streaming_transient_guard_bytes,
                "before GLM graph allocation")) {
        return false;
    }
    if (!glm_graph_validate_layout(model,
                                   weights,
                                   g,
                                   layer_start,
                                   layer_end,
                                   require_token_embd,
                                   require_output_head)) {
        return false;
    }
    g->ctx_cap = glm_graph_full_attention_cap(g->ctx_size,
                                              g->ssd_streaming);
    g->full_kv_cache = glm_graph_expanded_kv_cache_enabled(g->ssd_streaming);
    g->compact_cache_cap =
        glm_graph_compact_cache_initial_cap(g->ctx_size, g->ctx_cap);
    g->indexed_prefill_cap =
        g->compact_cache_cap != 0 ?
            glm_graph_indexed_prefill_chunk_tokens(g->ctx_cap, g->compact_cache_cap) :
            0;
    g->indexed_prefill_score_cap =
        glm_graph_indexed_prefill_score_tokens(g->indexed_prefill_cap,
                                               g->compact_cache_cap);
    g->indexer_full_layers =
        glm_graph_full_indexer_layer_count_range(g->layer_start,
                                                 g->layer_end);
    if (g->ctx_size > g->ctx_cap) {
        fprintf(stderr,
                "ds4: GLM Metal session ctx=%u (model max=%u); "
                "full-attention prefill/work cap=%u; compact indexed decode is used beyond the cap\n",
                g->ctx_size,
                glm_graph_model_context_limit(),
                g->ctx_cap);
    }

    const uint64_t emb_bytes = (uint64_t)DS4_N_EMBD * sizeof(float);
    const uint64_t q_rank_bytes = (uint64_t)DS4_N_LORA_Q * sizeof(float);
    const uint64_t q_bytes = g->q_dim * sizeof(float);
    const uint64_t kv_raw_bytes = g->kv_raw_dim * sizeof(float);
    const uint64_t kv_norm_bytes = (uint64_t)DS4_N_KV_LORA * sizeof(float);
    const uint64_t k_nope_bytes = (uint64_t)DS4_N_HEAD * g->q_nope * sizeof(float);
    const uint64_t heads_bytes = g->heads_dim * sizeof(float);
    const uint64_t indexer_k_bytes = (uint64_t)DS4_N_INDEXER_HEAD_DIM * sizeof(float);
    const uint64_t indexer_q_bytes =
        (uint64_t)DS4_N_INDEXER_HEAD * DS4_N_INDEXER_HEAD_DIM * sizeof(float);
    const uint64_t indexer_weights_bytes =
        (uint64_t)DS4_N_INDEXER_HEAD * sizeof(float);
    const uint64_t indexer_work_cap =
        g->compact_cache_cap != 0 ? g->compact_cache_cap : g->ctx_cap;
    const uint64_t indexer_scores_bytes = indexer_work_cap * sizeof(float);
    const uint32_t indexer_top_k = glm_graph_indexer_top_k_limit();
    const uint64_t indexer_selected_bytes =
        (uint64_t)indexer_top_k * sizeof(uint32_t);
    const uint64_t qk_low_bytes =
        (uint64_t)DS4_N_HEAD * DS4_N_KV_LORA * sizeof(float);
    const uint32_t split_attn_blocks = glm_graph_indexed_decode_split_blocks();
    const uint64_t attn_partial_lora_bytes =
        (uint64_t)split_attn_blocks * qk_low_bytes;
    const uint64_t attn_partial_ms_bytes =
        (uint64_t)split_attn_blocks * DS4_N_HEAD * 2u * sizeof(float);
    const uint64_t logits_bytes = (uint64_t)DS4_N_VOCAB * sizeof(float);
    const uint64_t full_kv_elem_bytes = glm_graph_full_kv_cache_elem_bytes();
    const uint64_t key_cache_bytes = g->full_kv_cache ?
        (uint64_t)g->ctx_cap * g->q_dim * full_kv_elem_bytes : 0;
    const uint64_t value_cache_bytes = g->full_kv_cache ?
        (uint64_t)g->ctx_cap * g->heads_dim * full_kv_elem_bytes : 0;
    const uint64_t compact_kv_lora_bytes =
        (uint64_t)g->compact_cache_cap * DS4_N_KV_LORA *
        glm_graph_compact_cache_elem_bytes();
    const uint64_t compact_k_rope_bytes =
        (uint64_t)g->compact_cache_cap * DS4_N_ROT *
        glm_graph_compact_cache_elem_bytes();
    const uint64_t compact_indexer_key_bytes =
        (uint64_t)g->compact_cache_cap * DS4_N_INDEXER_HEAD_DIM *
        glm_graph_compact_cache_elem_bytes();
    const uint64_t batch_rows =
        g->full_kv_cache || g->indexed_prefill_cap == 0 ?
            g->ctx_cap :
            g->indexed_prefill_cap;
    const uint64_t indexed_batch_rows = g->indexed_prefill_cap;
    const uint64_t indexed_score_rows = g->indexed_prefill_score_cap;
    const uint64_t batch_indexer_q_bytes = indexed_batch_rows * indexer_q_bytes;
    const uint64_t batch_indexer_weights_bytes = indexed_batch_rows * indexer_weights_bytes;
    const uint64_t batch_indexer_scores_bytes =
        indexed_score_rows * indexer_work_cap * sizeof(float);
    const uint64_t batch_indexer_selected_bytes =
        indexed_batch_rows * indexer_top_k * sizeof(uint32_t);
    const uint64_t batch_qk_low_bytes = indexed_batch_rows * qk_low_bytes;
    const uint64_t batch_attn_lora_bytes =
        indexed_batch_rows * (uint64_t)DS4_N_HEAD * DS4_N_KV_LORA * sizeof(float);
    const uint64_t routed_mid_bytes =
        (uint64_t)DS4_N_EXPERT_USED * DS4_N_FF_EXP * sizeof(float);
    const uint64_t routed_down_bytes =
        (uint64_t)DS4_N_EXPERT_USED * DS4_N_EMBD * sizeof(float);
    const double cache_gib =
        (double)(g->layer_count * (key_cache_bytes + value_cache_bytes)) /
        (1024.0 * 1024.0 * 1024.0);
    if (g->full_kv_cache) {
        fprintf(stderr,
                "ds4: GLM graph allocating full-attention KV cache: work_ctx=%u layers=%u:%u (%u) %s %.2f GiB\n",
                g->ctx_cap,
                g->layer_start,
                g->layer_end,
                g->layer_count,
                "f16",
                cache_gib);
    } else {
        fprintf(stderr,
                "ds4: GLM graph using compact DSA KV only; expanded full-attention KV cache is skipped\n");
    }
#ifdef DS4_ROCM_BUILD
    if (glm_graph_env_truthy(
            getenv("DS4_ROCM_GLM_LAYER_SLICE_TOKEN_DECODE"))) {
        fprintf(stderr,
                "ds4: ROCm GLM one-token layer slices use the optimized token graph\n");
    }
#endif
    if (g->compact_cache_cap != 0) {
        const uint64_t compact_kv_total =
            (uint64_t)g->layer_count * (compact_kv_lora_bytes + compact_k_rope_bytes);
        const uint64_t compact_indexer_total =
            (uint64_t)g->indexer_full_layers * compact_indexer_key_bytes;
        const double compact_gib =
            (double)(compact_kv_total + compact_indexer_total) /
            (1024.0 * 1024.0 * 1024.0);
        fprintf(stderr,
                "ds4: GLM graph allocating compact DSA cache: rows=%u logical_ctx=%u kv_layers=%u indexer_layers=%u %s %.2f GiB\n",
                g->compact_cache_cap,
                g->ctx_size,
                g->layer_count,
                g->indexer_full_layers,
                glm_graph_compact_cache_is_f16() ? "f16" : "f32",
                compact_gib);
        fprintf(stderr,
                "ds4: GLM compact indexed prefill chunk=%u score_rows=%u score_scratch=%.2f MiB\n",
                g->indexed_prefill_cap,
                g->indexed_prefill_score_cap,
                (double)batch_indexer_scores_bytes / (1024.0 * 1024.0));
    }

    bool ok = true;
#define DS4_GLM_GRAPH_ALLOC_TENSOR(var, bytes_) \
    do { \
        (var) = ds4_gpu_tensor_alloc((bytes_)); \
        if (!(var)) { \
            fprintf(stderr, "ds4: GLM Metal graph could not allocate %s\n", #var); \
            ok = false; \
        } \
    } while (0)

    DS4_GLM_GRAPH_ALLOC_TENSOR(g->cur, emb_bytes);
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->next, emb_bytes);
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->attn_norm, emb_bytes);
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->q_rank, q_rank_bytes);
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->q_rank_norm, q_rank_bytes);
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->q, q_bytes);
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->indexer_k, indexer_k_bytes);
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->indexer_q, indexer_q_bytes);
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->indexer_weights, indexer_weights_bytes);
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->indexer_scores, indexer_scores_bytes);
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->indexer_selected, indexer_selected_bytes);
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->qk_low, qk_low_bytes);
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->attn_partial_lora, attn_partial_lora_bytes);
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->attn_partial_ms, attn_partial_ms_bytes);
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->kv_raw, kv_raw_bytes);
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->kv_norm, kv_norm_bytes);
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->k_nope, k_nope_bytes);
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->value, heads_bytes);
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->heads, heads_bytes);
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->attn_out, emb_bytes);
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->after_attn, emb_bytes);
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->ffn_norm, emb_bytes);
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->ffn_gate, g->dense_hidden_max * sizeof(float));
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->ffn_up, g->dense_hidden_max * sizeof(float));
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->ffn_mid, g->ffn_mid_elems * sizeof(float));
    if (g->generic_routed_moe) {
        DS4_GLM_GRAPH_ALLOC_TENSOR(g->routed_gate, routed_mid_bytes);
        DS4_GLM_GRAPH_ALLOC_TENSOR(g->routed_up, routed_mid_bytes);
        DS4_GLM_GRAPH_ALLOC_TENSOR(g->routed_down, routed_down_bytes);
    }
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->ffn_out, emb_bytes);
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->ffn_sum, emb_bytes);
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->router_logits, (uint64_t)DS4_N_EXPERT * sizeof(float));
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->router_probs, (uint64_t)DS4_N_EXPERT * sizeof(float));
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->router_selected, (uint64_t)DS4_N_EXPERT_USED * sizeof(int32_t));
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->router_weights, (uint64_t)DS4_N_EXPERT_USED * sizeof(float));
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->output_norm, emb_bytes);
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->logits, logits_bytes);
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->batch_router_logits, batch_rows * DS4_N_EXPERT * sizeof(float));
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->batch_router_probs, batch_rows * DS4_N_EXPERT * sizeof(float));
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->batch_router_selected, batch_rows * DS4_N_EXPERT_USED * sizeof(int32_t));
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->batch_router_weights, batch_rows * DS4_N_EXPERT_USED * sizeof(float));
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->prefill_seed_router_selected,
                               (uint64_t)DS4_N_LAYER *
                               DS4_STREAMING_PREFILL_CACHE_SEED_MAX_TOKENS *
                               DS4_N_EXPERT_USED *
                               sizeof(int32_t));

    DS4_GLM_GRAPH_ALLOC_TENSOR(g->prefill_tokens, batch_rows * sizeof(int32_t));
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->batch_cur, batch_rows * emb_bytes);
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->batch_next, batch_rows * emb_bytes);
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->batch_attn_norm, batch_rows * emb_bytes);
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->batch_q_rank, batch_rows * q_rank_bytes);
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->batch_q_rank_norm, batch_rows * q_rank_bytes);
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->batch_q, batch_rows * q_bytes);
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->batch_indexer_k, batch_rows * indexer_k_bytes);
    if (g->compact_cache_cap != 0 && g->indexed_prefill_cap != 0) {
        DS4_GLM_GRAPH_ALLOC_TENSOR(g->batch_indexer_q, batch_indexer_q_bytes);
        DS4_GLM_GRAPH_ALLOC_TENSOR(g->batch_indexer_weights, batch_indexer_weights_bytes);
        DS4_GLM_GRAPH_ALLOC_TENSOR(g->batch_indexer_scores, batch_indexer_scores_bytes);
        DS4_GLM_GRAPH_ALLOC_TENSOR(g->batch_indexer_selected, batch_indexer_selected_bytes);
        DS4_GLM_GRAPH_ALLOC_TENSOR(g->batch_qk_low, batch_qk_low_bytes);
        DS4_GLM_GRAPH_ALLOC_TENSOR(g->batch_attn_lora, batch_attn_lora_bytes);
    }
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->batch_kv_raw, batch_rows * kv_raw_bytes);
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->batch_kv_norm, batch_rows * kv_norm_bytes);
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->batch_k_nope, batch_rows * k_nope_bytes);
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->batch_value, batch_rows * heads_bytes);
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->batch_heads, batch_rows * heads_bytes);
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->batch_attn_out, batch_rows * emb_bytes);
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->batch_after_attn, batch_rows * emb_bytes);
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->batch_ffn_norm, batch_rows * emb_bytes);
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->batch_ffn_gate, batch_rows * g->dense_hidden_max * sizeof(float));
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->batch_ffn_up, batch_rows * g->dense_hidden_max * sizeof(float));
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->batch_shared_mid, batch_rows * DS4_N_FF_EXP * sizeof(float));
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->batch_ffn_mid, batch_rows * g->ffn_mid_elems * sizeof(float));
    if (g->generic_routed_moe) {
        DS4_GLM_GRAPH_ALLOC_TENSOR(g->batch_routed_gate, batch_rows * routed_mid_bytes);
        DS4_GLM_GRAPH_ALLOC_TENSOR(g->batch_routed_up, batch_rows * routed_mid_bytes);
        DS4_GLM_GRAPH_ALLOC_TENSOR(g->batch_routed_down, batch_rows * routed_down_bytes);
    }
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->batch_ffn_out, batch_rows * emb_bytes);

    for (uint32_t il = g->layer_start; ok && il <= g->layer_end; il++) {
        int cache_tier = 0;
        if (g->placement && g->placement[il + 1u] >= 0 &&
            g->placement[il + 1u] < DS4_MAX_GPUS) {
            cache_tier = g->placement[il + 1u];
        }
#define DS4_GLM_GRAPH_ALLOC_TENSOR_TIER(var, bytes_) \
    do { \
        (var) = ds4_gpu_tensor_alloc_ptr_on(cache_tier, (bytes_)); \
        if (!(var)) { \
            fprintf(stderr, "ds4: GLM graph could not allocate %s on tier %d\n", \
                    #var, cache_tier); \
            ok = false; \
        } \
    } while (0)
        if (g->full_kv_cache) {
            DS4_GLM_GRAPH_ALLOC_TENSOR_TIER(g->layer_key_cache[il], key_cache_bytes);
            DS4_GLM_GRAPH_ALLOC_TENSOR_TIER(g->layer_value_cache[il], value_cache_bytes);
        }
        if (g->compact_cache_cap != 0) {
            DS4_GLM_GRAPH_ALLOC_TENSOR_TIER(g->layer_kv_lora_cache[il], compact_kv_lora_bytes);
            DS4_GLM_GRAPH_ALLOC_TENSOR_TIER(g->layer_k_rope_cache[il], compact_k_rope_bytes);
            if (glm_graph_layer_uses_full_indexer(il)) {
                DS4_GLM_GRAPH_ALLOC_TENSOR_TIER(g->layer_indexer_key_cache[il],
                                                compact_indexer_key_bytes);
            }
        }
#undef DS4_GLM_GRAPH_ALLOC_TENSOR_TIER
    }
#undef DS4_GLM_GRAPH_ALLOC_TENSOR

    if (!ok) {
        glm_graph_free(g);
        return false;
    }
    glm_graph_ws_init(g);
    return true;
}

static bool glm_graph_alloc(
        ds4_glm_gpu_graph *g,
        const ds4_model   *model,
        const ds4_weights *weights,
        int                ctx_size,
        bool               ssd_streaming,
        bool               ssd_streaming_cold) {
    const uint32_t normal_layers = glm_graph_normal_layer_count();
    if (normal_layers == 0) {
        fprintf(stderr, "ds4: GLM Metal graph found no normal transformer layers\n");
        return false;
    }
    return glm_graph_alloc_slice(g,
                                 model,
                                 weights,
                                 ctx_size,
                                 ssd_streaming,
                                 ssd_streaming_cold,
                                 0,
                                 0,
                                 normal_layers - 1u,
                                 true,
                                 true);
}

static uint32_t glm_graph_weight_type_for_offset(
        const ds4_model *model,
        uint64_t         weight_offset);

static int glm_graph_matmul_q8_0_decode_tensor(
        ds4_gpu_tensor       *out,
        const ds4_model      *model,
        uint64_t              weight_offset,
        uint64_t              in_dim,
        uint64_t              out_dim,
        const ds4_gpu_tensor *x,
        bool                  ssd_streaming) {
    if (!model) return 0;
    const uint32_t weight_type = glm_graph_weight_type_for_offset(model, weight_offset);
    if (ssd_streaming) {
        return ds4_gpu_matmul_quant_tensor(out,
                                           model->map,
                                           model->size,
                                           weight_offset,
                                           weight_type,
                                           in_dim,
                                           out_dim,
                                           x,
                                           1);
    }
    return ds4_gpu_matmul_quant_decode_mpp_model_view_tensor(out,
                                                             model->map,
                                                             model->size,
                                                             weight_offset,
                                                             weight_type,
                                                             in_dim,
                                                             out_dim,
                                                             x,
                                                             1);
}

static bool glm_graph_q8_decode_profile_enabled(uint32_t il, const char *label) {
    (void)il;
    (void)label;
    return false;
}

static uint32_t glm_graph_weight_type_for_offset(
        const ds4_model *model,
        uint64_t         weight_offset) {
    if (!model || !model->tensors) return DS4_TENSOR_Q8_0;
    for (uint64_t i = 0; i < model->n_tensors; i++) {
        const ds4_tensor *t = &model->tensors[i];
        if (t->abs_offset == weight_offset) return t->type;
    }
    return DS4_TENSOR_Q8_0;
}

static bool glm_graph_weights_are_q8_0(
        const ds4_model *model,
        uint64_t         offset_a,
        uint64_t         offset_b) {
    return glm_graph_weight_type_for_offset(model, offset_a) == DS4_TENSOR_Q8_0 &&
           glm_graph_weight_type_for_offset(model, offset_b) == DS4_TENSOR_Q8_0;
}

static int glm_graph_matmul_q8_0_decode_profiled_tensor(
        ds4_gpu_tensor       *out,
        const ds4_model      *model,
        uint64_t              weight_offset,
        uint64_t              in_dim,
        uint64_t              out_dim,
        const ds4_gpu_tensor *x,
        uint32_t              il,
        uint32_t              pos,
        const char           *label,
        bool                  ssd_streaming) {
    const bool profile = glm_graph_q8_decode_profile_enabled(il, label);
    if (profile) {
        if (ds4_gpu_end_commands() == 0) return 0;
        if (ds4_gpu_begin_commands() == 0) return 0;
    }
    const double t0 = profile ? now_sec() : 0.0;
    int ok = glm_graph_matmul_q8_0_decode_tensor(out,
                                                 model,
                                                 weight_offset,
                                                 in_dim,
                                                 out_dim,
                                                 x,
                                                 ssd_streaming);
    if (profile) {
        if (ok) ok = ds4_gpu_end_commands() != 0;
        else (void)ds4_gpu_synchronize();
        const double now = now_sec();
        fprintf(stderr,
                "ds4: GLM Q8 decode profile layer=%u pos=%u label=%s in=%llu out=%llu %.3f ms\n",
                il,
                pos,
                label ? label : "?",
                (unsigned long long)in_dim,
                (unsigned long long)out_dim,
                (now - t0) * 1000.0);
        if (ok) ok = ds4_gpu_begin_commands() != 0;
    }
    return ok;
}

static bool glm_graph_encode_output_head_from(
        ds4_glm_gpu_graph *g,
        const ds4_model   *model,
        const ds4_weights *weights,
        const ds4_gpu_tensor *hidden) {
    bool ok = ds4_gpu_rms_norm_weight_tensor(g->output_norm,
                                             hidden,
                                             model->map,
                                             model->size,
                                             weights->output_norm->abs_offset,
                                             DS4_N_EMBD,
                                             DS4_RMS_EPS) != 0;
    if (ok) ok = glm_graph_matmul_q8_0_decode_tensor(g->logits,
                                                     model,
                                                     weights->output->abs_offset,
                                                     DS4_N_EMBD,
                                                     DS4_N_VOCAB,
                                                     g->output_norm,
                                                     g->ssd_streaming) != 0;
    return ok;
}

static bool glm_graph_encode_output_head(
        ds4_glm_gpu_graph *g,
        const ds4_model   *model,
        const ds4_weights *weights) {
    return glm_graph_encode_output_head_from(g, model, weights, g->cur);
}

static bool glm_graph_forward_output_head(
        ds4_glm_gpu_graph    *g,
        const ds4_model      *model,
        const ds4_weights    *weights,
        const ds4_gpu_tensor *hidden,
        float                *logits_out) {
    if (!g || !model || !weights || !hidden || !logits_out) return false;
    bool ok = ds4_gpu_begin_commands() != 0;
    if (ok) ok = glm_graph_encode_output_head_from(g, model, weights, hidden);
    if (ok) ok = ds4_gpu_end_commands() != 0;
    else (void)ds4_gpu_synchronize();
    if (ok && glm_debug_hidden_dump_layer() < 0)
        glm_debug_dump_hidden_row(hidden, 0);
    if (ok) {
        ok = ds4_gpu_tensor_read(g->logits,
                                 0,
                                 logits_out,
                                 (uint64_t)DS4_N_VOCAB * sizeof(float)) != 0;
    }
    return ok;
}

static bool glm_graph_profile_stage(
        bool        enabled,
        const char *part,
        const char *stage,
        uint32_t    il,
        uint32_t    pos0,
        uint32_t    n_tokens,
        double     *stage_t0) {
    if (!enabled) return true;
    if (!stage_t0) return false;
    return metal_graph_layer_stage_profile_boundary(part, stage, il, pos0, n_tokens, stage_t0);
}

static bool glm_graph_profile_router_selection(
        ds4_glm_gpu_graph       *g,
        const ds4_layer_weights *layer,
        uint32_t                 il,
        uint32_t                 pos) {
    if (!g_expert_profile.active) return true;
    if (!g || !layer || !g->router_selected || !g->router_weights) return false;

    if (ds4_gpu_end_commands() == 0) {
        fprintf(stderr,
                "ds4: failed to end GLM Metal command batch for expert profile readback\n");
        return false;
    }

    int32_t selected[DS4_MAX_EXPERT_USED] = {0};
    float weights[DS4_MAX_EXPERT_USED] = {0};
    const bool read_ok =
        ds4_gpu_tensor_read(g->router_selected,
                            0,
                            selected,
                            (uint64_t)DS4_N_EXPERT_USED * sizeof(selected[0])) != 0 &&
        ds4_gpu_tensor_read(g->router_weights,
                            0,
                            weights,
                            (uint64_t)DS4_N_EXPERT_USED * sizeof(weights[0])) != 0;

    if (ds4_gpu_begin_commands() == 0) {
        fprintf(stderr,
                "ds4: failed to resume GLM Metal command batch after expert profile readback\n");
        return false;
    }
    if (!read_ok) {
        fprintf(stderr, "ds4: failed to read GLM Metal router tensors for expert profile\n");
        return false;
    }

    ds4_expert_profile_record(il,
                              pos,
                              selected,
                              weights,
                              layer->ffn_gate_tid2eid != NULL);
    return true;
}

static bool glm_graph_profile_router_selection_batch(
        ds4_glm_gpu_graph       *g,
        const ds4_layer_weights *layer,
        uint32_t                 il,
        uint32_t                 pos0,
        uint32_t                 n_tokens) {
    if (!g_expert_profile.active) return true;
    if (!g || !layer || !g->batch_router_selected ||
        !g->batch_router_weights || n_tokens == 0) {
        return false;
    }

    const size_t selected_count = (size_t)n_tokens * DS4_N_EXPERT_USED;
    if (n_tokens != 0 && selected_count / n_tokens != DS4_N_EXPERT_USED) {
        return false;
    }
    if (selected_count > SIZE_MAX / sizeof(int32_t) ||
        selected_count > SIZE_MAX / sizeof(float)) {
        return false;
    }

    if (ds4_gpu_end_commands() == 0) {
        fprintf(stderr,
                "ds4: failed to end GLM Metal command batch for batch expert profile readback\n");
        return false;
    }

    int32_t *selected = xmalloc(selected_count * sizeof(selected[0]));
    float *weights = xmalloc(selected_count * sizeof(weights[0]));
    const bool read_ok =
        ds4_gpu_tensor_read(g->batch_router_selected,
                            0,
                            selected,
                            (uint64_t)selected_count * sizeof(selected[0])) != 0 &&
        ds4_gpu_tensor_read(g->batch_router_weights,
                            0,
                            weights,
                            (uint64_t)selected_count * sizeof(weights[0])) != 0;

    if (ds4_gpu_begin_commands() == 0) {
        free(weights);
        free(selected);
        fprintf(stderr,
                "ds4: failed to resume GLM Metal command batch after batch expert profile readback\n");
        return false;
    }
    if (!read_ok) {
        free(weights);
        free(selected);
        fprintf(stderr, "ds4: failed to read GLM Metal batch router tensors for expert profile\n");
        return false;
    }

    for (uint32_t t = 0; t < n_tokens; t++) {
        const size_t off = (size_t)t * DS4_N_EXPERT_USED;
        ds4_expert_profile_record(il,
                                  pos0 + t,
                                  selected + off,
                                  weights + off,
                                  layer->ffn_gate_tid2eid != NULL);
    }

    free(weights);
    free(selected);
    return true;
}

static bool glm_graph_prefill_stage_boundary(
        bool        stage_profile,
        bool        stage_sync,
        const char *part,
        const char *stage,
        uint32_t    il,
        uint32_t    pos0,
        uint32_t    n_tokens,
        double     *stage_t0) {
    if (stage_profile) {
        return glm_graph_profile_stage(true, part, stage, il, pos0, n_tokens, stage_t0);
    }
    if (stage_sync) return glm_graph_prefill_stage_sync_boundary();
    return true;
}

static int glm_graph_routed_moe_one_dispatch(
        const ds4_glm_gpu_graph *g,
        const ds4_model         *model,
        const ds4_layer_weights *l,
        uint32_t                 il,
        ds4_gpu_tensor          *out,
        ds4_gpu_tensor          *mid,
        uint64_t                 gate_expert_bytes,
        uint64_t                 gate_row_bytes,
        uint64_t                 up_expert_bytes,
        uint64_t                 up_row_bytes,
        uint64_t                 down_expert_bytes,
        uint64_t                 down_row_bytes,
        const ds4_gpu_tensor    *selected,
        const ds4_gpu_tensor    *weights,
        const ds4_gpu_tensor    *x,
        bool                     force_resident) {
    if (!g || !model || !l) return 0;
    /* Under the TP expert split only the ownership-aware kernels may run:
     * the generic mul_mv_id family and the GLM q2_K resident pair/down.
     * Anything else would silently compute the full expert set. */
    if (g->tp_world == 2 &&
        !glm_graph_layer_uses_generic_routed_moe(l) &&
        l->ffn_gate_exps->type != DS4_TENSOR_Q2_K) {
        fprintf(stderr,
                "ds4: GLM TP split lacks ownership-aware kernels for expert type %u (layer %u)\n",
                l->ffn_gate_exps->type, il);
        return 0;
    }
    if (glm_graph_layer_uses_generic_routed_moe(l)) {
        if (!g->routed_gate || !g->routed_up || !g->routed_down ||
            l->ffn_gate_exps->type != l->ffn_up_exps->type) {
            if (getenv("DS4_GLM_TP_DEBUG")) {
                fprintf(stderr,
                        "ds4: glm dispatch guard: gate=%p up=%p down=%p types=%u/%u\n",
                        (void *)g->routed_gate, (void *)g->routed_up,
                        (void *)g->routed_down,
                        l->ffn_gate_exps->type, l->ffn_up_exps->type);
            }
            return 0;
        }
        return ds4_gpu_routed_moe_one_tensor(out,
                                             g->routed_gate,
                                             g->routed_up,
                                             mid,
                                             g->routed_down,
                                             model->map,
                                             model->size,
                                             l->ffn_gate_exps->abs_offset,
                                             l->ffn_up_exps->abs_offset,
                                             l->ffn_down_exps->abs_offset,
                                             l->ffn_gate_exps->type,
                                             l->ffn_down_exps->type,
                                             gate_expert_bytes,
                                             gate_row_bytes,
                                             down_expert_bytes,
                                             down_row_bytes,
                                             DS4_N_EMBD,
                                             DS4_N_FF_EXP,
                                             DS4_N_EMBD,
                                             selected,
                                             weights,
                                             DS4_N_EXPERT,
                                             DS4_N_EXPERT_USED,
                                             0.0f,
                                             x,
                                             NULL,
                                             il,
                                             force_resident);
    }

    return ds4_gpu_glm_routed_moe_one_tensor(out,
                                             mid,
                                             model->map,
                                             model->size,
                                             l->ffn_gate_exps->abs_offset,
                                             l->ffn_up_exps->abs_offset,
                                             l->ffn_down_exps->abs_offset,
                                             l->ffn_gate_exps->type,
                                             l->ffn_up_exps->type,
                                             l->ffn_down_exps->type,
                                             gate_expert_bytes,
                                             gate_row_bytes,
                                             up_expert_bytes,
                                             up_row_bytes,
                                             down_expert_bytes,
                                             down_row_bytes,
                                             DS4_N_EMBD,
                                             DS4_N_FF_EXP,
                                             DS4_N_EMBD,
                                             selected,
                                             weights,
                                             DS4_N_EXPERT,
                                             DS4_N_EXPERT_USED,
                                             il,
                                             x,
                                             force_resident);
}

/* Post-compute visibility for GLM TP debugging: the combine stashes the
 * selected-ids contents pointer; by exchange time the router kernels have
 * completed, so the service thread sees final ids. */
static const int32_t *g_glm_tp_debug_ids DS4_MAYBE_UNUSED;

/* After the TP ownership-split batch routed MoE, exchange the
 * per-token routed partial rows with the peer through shared bounce
 * buffers (one gate per sparse layer per chunk) and rebuild the full
 * routed output with a commutative add. */
/* The routed batch dispatch writes its local partial DIRECTLY into the
 * shared bounce buffer (graph scratch may be private/untracked on M5, so
 * a blit from it is not reliable); ensure capacity before dispatching. */
static bool glm_graph_tp_batch_bounce_ready(ds4_glm_gpu_graph *g,
                                            uint32_t n_tokens) {
    const uint64_t bytes = (uint64_t)n_tokens * DS4_N_EMBD * sizeof(float);
    if (!g->tp_bounce_out || ds4_gpu_tensor_bytes(g->tp_bounce_out) < bytes) {
        ds4_gpu_tensor_free(g->tp_bounce_out);
        ds4_gpu_tensor_free(g->tp_bounce_in);
        g->tp_bounce_out = ds4_gpu_tensor_alloc(bytes);
        g->tp_bounce_in = ds4_gpu_tensor_alloc(bytes);
    }
    return g->tp_bounce_out && g->tp_bounce_in;
}

static bool glm_graph_tp_batch_ffn_combine(
        ds4_glm_gpu_graph *g,
        uint32_t           il,
        ds4_gpu_tensor    *ffn_out,
        uint32_t           n_tokens) {
    const uint64_t bytes = (uint64_t)n_tokens * DS4_N_EMBD * sizeof(float);
    if (!g->tp_bounce_out || !g->tp_bounce_in) return false;
    if (getenv("DS4_GLM_ABLATE_COMBINE")) {
        /* Timing probe: local half only, no exchange (garbage output;
         * both ranks must set the env or the gates desync). */
        return ds4_gpu_add_tensor(ffn_out,
                                  g->tp_bounce_out,
                                  g->tp_bounce_out,
                                  (uint32_t)((uint64_t)n_tokens * DS4_N_EMBD)) != 0;
    }
    if (!ds4_gpu_tp_big_gate_encode(il, n_tokens,
                                    g->tp_bounce_out, g->tp_bounce_in,
                                    bytes)) {
        return false;
    }
    return ds4_gpu_add_tensor(ffn_out,
                              g->tp_bounce_out,
                              g->tp_bounce_in,
                              (uint32_t)((uint64_t)n_tokens * DS4_N_EMBD)) != 0;
}

static int glm_graph_routed_moe_batch_dispatch(
        ds4_glm_gpu_graph       *g,
        const ds4_model         *model,
        const ds4_layer_weights *l,
        uint32_t                 il,
        ds4_gpu_tensor          *out,
        ds4_gpu_tensor          *mid,
        uint64_t                 gate_expert_bytes,
        uint64_t                 gate_row_bytes,
        uint64_t                 up_expert_bytes,
        uint64_t                 up_row_bytes,
        uint64_t                 down_expert_bytes,
        uint64_t                 down_row_bytes,
        const ds4_gpu_tensor    *selected,
        const ds4_gpu_tensor    *weights,
        const ds4_gpu_tensor    *x,
        uint32_t                 n_tokens,
        uint32_t                 mid_token_stride,
        bool                     force_resident,
        bool                     direct_scalar_q4) {
    if (!g || !model || !l) return 0;
    g->batch_routed_mid_is_f16 = false;

    if (glm_graph_layer_uses_generic_routed_moe(l)) {
        if (!g->batch_routed_gate || !g->batch_routed_up || !g->batch_routed_down ||
            l->ffn_gate_exps->type != l->ffn_up_exps->type) {
            return 0;
        }
        return ds4_gpu_routed_moe_batch_tensor(out,
                                               g->batch_routed_gate,
                                               g->batch_routed_up,
                                               mid,
                                               g->batch_routed_down,
                                               model->map,
                                               model->size,
                                               l->ffn_gate_exps->abs_offset,
                                               l->ffn_up_exps->abs_offset,
                                               l->ffn_down_exps->abs_offset,
                                               l->ffn_gate_exps->type,
                                               l->ffn_down_exps->type,
                                               gate_expert_bytes,
                                               gate_row_bytes,
                                               down_expert_bytes,
                                               down_row_bytes,
                                               DS4_N_EMBD,
                                               DS4_N_FF_EXP,
                                               DS4_N_EMBD,
                                               selected,
                                               weights,
                                               DS4_N_EXPERT,
                                               DS4_N_EXPERT_USED,
                                               0.0f,
                                               x,
                                               il,
                                               n_tokens,
                                               &g->batch_routed_mid_is_f16,
                                               force_resident);
    }

    if (direct_scalar_q4) {
        return ds4_gpu_glm_routed_moe_batch_direct_scalar_q4_tensor(
                out,
                mid,
                model->map,
                model->size,
                l->ffn_gate_exps->abs_offset,
                l->ffn_up_exps->abs_offset,
                l->ffn_down_exps->abs_offset,
                l->ffn_gate_exps->type,
                l->ffn_up_exps->type,
                l->ffn_down_exps->type,
                gate_expert_bytes,
                gate_row_bytes,
                up_expert_bytes,
                up_row_bytes,
                down_expert_bytes,
                down_row_bytes,
                DS4_N_EMBD,
                DS4_N_FF_EXP,
                DS4_N_EMBD,
                selected,
                weights,
                DS4_N_EXPERT,
                DS4_N_EXPERT_USED,
                il,
                x,
                n_tokens,
                mid_token_stride);
    }

    return ds4_gpu_glm_routed_moe_batch_tensor(
            out,
            mid,
            model->map,
            model->size,
            l->ffn_gate_exps->abs_offset,
            l->ffn_up_exps->abs_offset,
            l->ffn_down_exps->abs_offset,
            l->ffn_gate_exps->type,
            l->ffn_up_exps->type,
            l->ffn_down_exps->type,
            gate_expert_bytes,
            gate_row_bytes,
            up_expert_bytes,
            up_row_bytes,
            down_expert_bytes,
            down_row_bytes,
            DS4_N_EMBD,
            DS4_N_FF_EXP,
            DS4_N_EMBD,
            selected,
            weights,
            DS4_N_EXPERT,
            DS4_N_EXPERT_USED,
            il,
            x,
            n_tokens,
            mid_token_stride,
            force_resident);
}

static bool glm_graph_disable_add3_residual(void);

static bool glm_graph_use_streaming_selected_async_load(
        const ds4_glm_gpu_graph *g) {
    if (!g || !g->ssd_streaming) return false;
    if (glm_graph_env_present("DS4_ROCM_DISABLE_STREAMING_SELECTED_ASYNC_LOAD",
                              "DS4_METAL_DISABLE_STREAMING_SELECTED_ASYNC_LOAD") ||
        glm_graph_env_present("DS4_ROCM_DISABLE_GLM_STREAMING_SELECTED_ASYNC_LOAD",
                              "DS4_METAL_DISABLE_GLM_STREAMING_SELECTED_ASYNC_LOAD")) {
        return false;
    }
#ifdef DS4_ROCM_BUILD
    return true;
#else
    return getenv("DS4_METAL_ENABLE_GLM_STREAMING_SELECTED_ASYNC_LOAD") != NULL;
#endif
}

typedef struct glm_graph_streaming_async_profile {
    uint64_t async_calls;
    uint64_t sync_calls;
    double async_total_ms;
    double sync_total_ms;
    double async_signal_start_ms;
    double async_flush_router_ms;
    double async_shared_ms;
    double async_flush_shared_ms;
    double async_finish_ms;
    double async_routed_ms;
    double async_post_ms;
    double sync_early_load_ms;
    double sync_shared_ms;
    double sync_routed_ms;
    double sync_post_ms;
} glm_graph_streaming_async_profile;

static glm_graph_streaming_async_profile g_glm_streaming_async_profile;
static bool g_glm_streaming_async_profile_registered;

static bool glm_graph_streaming_async_profile_enabled(void) {
    return glm_graph_env_present("DS4_ROCM_GLM_STREAMING_ASYNC_PROFILE",
                                 "DS4_METAL_GLM_STREAMING_ASYNC_PROFILE");
}

static void glm_graph_streaming_async_profile_print(void) {
    const glm_graph_streaming_async_profile *p =
        &g_glm_streaming_async_profile;
    if (p->async_calls == 0 && p->sync_calls == 0) return;
    const double async_calls = p->async_calls ? (double)p->async_calls : 1.0;
    const double sync_calls = p->sync_calls ? (double)p->sync_calls : 1.0;
    fprintf(stderr,
            "ds4: GLM streaming async profile async_calls=%llu "
            "total=%.3f ms avg=%.3f ms signal_start=%.3f ms "
            "flush_router=%.3f ms shared=%.3f ms flush_shared=%.3f ms "
            "finish=%.3f ms routed=%.3f ms post=%.3f ms\n",
            (unsigned long long)p->async_calls,
            p->async_total_ms,
            p->async_total_ms / async_calls,
            p->async_signal_start_ms,
            p->async_flush_router_ms,
            p->async_shared_ms,
            p->async_flush_shared_ms,
            p->async_finish_ms,
            p->async_routed_ms,
            p->async_post_ms);
    fprintf(stderr,
            "ds4: GLM streaming sync profile calls=%llu "
            "total=%.3f ms avg=%.3f ms early_load=%.3f ms "
            "shared=%.3f ms routed=%.3f ms post=%.3f ms\n",
            (unsigned long long)p->sync_calls,
            p->sync_total_ms,
            p->sync_total_ms / sync_calls,
            p->sync_early_load_ms,
            p->sync_shared_ms,
            p->sync_routed_ms,
            p->sync_post_ms);
}

static void glm_graph_streaming_async_profile_register(void) {
    if (g_glm_streaming_async_profile_registered) return;
    if (!glm_graph_streaming_async_profile_enabled()) return;
    atexit(glm_graph_streaming_async_profile_print);
    g_glm_streaming_async_profile_registered = true;
}

static double glm_graph_streaming_async_profile_ms(void) {
    return now_sec() * 1000.0;
}

/* Timing-only skip-ablation for the GLM decode layer (comma list in
 * DS4_GLM_DECODE_ABLATE): the skipped stage's output buffer keeps stale
 * contents, so the run produces garbage text but every remaining dispatch
 * (and every TP gate) still executes. Whole-token time deltas against a
 * baseline run are the only reliable per-stage cost measurement — the
 * stage profiler's per-stage command-buffer splits inflate small stages. */
#define DS4_GLM_ABLATE_ATTN_OUT  (1u << 0)
#define DS4_GLM_ABLATE_ATTN_CORE (1u << 1)
#define DS4_GLM_ABLATE_QPATH     (1u << 2)
#define DS4_GLM_ABLATE_INDEXER   (1u << 3)
#define DS4_GLM_ABLATE_ROUTED    (1u << 4)
#define DS4_GLM_ABLATE_SHARED    (1u << 5)
#define DS4_GLM_ABLATE_QKLOW     (1u << 6)

static uint32_t glm_decode_ablate_mask(void) {
    static int cached = -1;
    if (cached < 0) {
        uint32_t mask = 0;
        const char *env = getenv("DS4_GLM_DECODE_ABLATE");
        if (env) {
            if (strstr(env, "attn_out")) mask |= DS4_GLM_ABLATE_ATTN_OUT;
            if (strstr(env, "attn_core")) mask |= DS4_GLM_ABLATE_ATTN_CORE;
            if (strstr(env, "qpath")) mask |= DS4_GLM_ABLATE_QPATH;
            if (strstr(env, "indexer")) mask |= DS4_GLM_ABLATE_INDEXER;
            if (strstr(env, "routed")) mask |= DS4_GLM_ABLATE_ROUTED;
            if (strstr(env, "shared")) mask |= DS4_GLM_ABLATE_SHARED;
            if (strstr(env, "qklow")) mask |= DS4_GLM_ABLATE_QKLOW;
            if (mask) {
                fprintf(stderr, "ds4: GLM decode ablation active (mask 0x%x) — output is garbage, timing only\n", mask);
            }
        }
        cached = (int)mask;
    }
    return (uint32_t)cached;
}

static bool glm_graph_encode_shared_swiglu_one(
        ds4_gpu_tensor          *mid,
        ds4_gpu_tensor          *gate,
        ds4_gpu_tensor          *up,
        const ds4_model         *model,
        const ds4_layer_weights *l,
        uint32_t                 il,
        uint32_t                 pos,
        const ds4_gpu_tensor    *x,
        bool                     ssd_streaming,
        bool                     stage_profile,
        double                  *stage_t0) {
    if (!mid || !gate || !up || !model || !l || !x ||
        !l->ffn_gate_shexp || !l->ffn_up_shexp) {
        return false;
    }

    bool ok = true;
    if (glm_graph_weights_are_q8_0(model,
                                   l->ffn_gate_shexp->abs_offset,
                                   l->ffn_up_shexp->abs_offset)) {
        ok = ds4_gpu_shared_mid_swiglu_q8_0_tensor(
                mid,
                model->map,
                model->size,
                l->ffn_gate_shexp->abs_offset,
                l->ffn_up_shexp->abs_offset,
                DS4_N_EMBD,
                DS4_N_FF_EXP,
                x,
                0.0f) != 0;
        if (ok) ok = glm_graph_profile_stage(stage_profile,
                                             "glm_decode_ffn",
                                             "shared_gate_up_swiglu",
                                             il,
                                             pos,
                                             1,
                                             stage_t0);
        return ok;
    }

    ok = glm_graph_matmul_q8_0_decode_profiled_tensor(gate,
                                                      model,
                                                      l->ffn_gate_shexp->abs_offset,
                                                      DS4_N_EMBD,
                                                      DS4_N_FF_EXP,
                                                      x,
                                                      il,
                                                      pos,
                                                      "shared_gate",
                                                      ssd_streaming) != 0;
    if (ok) ok = glm_graph_matmul_q8_0_decode_profiled_tensor(up,
                                                              model,
                                                              l->ffn_up_shexp->abs_offset,
                                                              DS4_N_EMBD,
                                                              DS4_N_FF_EXP,
                                                              x,
                                                              il,
                                                              pos,
                                                              "shared_up",
                                                              ssd_streaming) != 0;
    if (ok) ok = glm_graph_profile_stage(stage_profile,
                                         "glm_decode_ffn",
                                         "shared_gate_up",
                                         il,
                                         pos,
                                         1,
                                         stage_t0);
    if (ok) ok = ds4_gpu_swiglu_tensor(mid,
                                       gate,
                                       up,
                                       DS4_N_FF_EXP,
                                       0.0f,
                                       1.0f) != 0;
    if (ok) ok = glm_graph_profile_stage(stage_profile,
                                         "glm_decode_ffn",
                                         "shared_swiglu",
                                         il,
                                         pos,
                                         1,
                                         stage_t0);
    return ok;
}

static bool glm_graph_encode_sparse_ffn_one(
        ds4_glm_gpu_graph       *g,
        const ds4_model         *model,
        const ds4_layer_weights *l,
        uint32_t                 il,
        uint32_t                 pos,
        const ds4_gpu_tensor    *ffn_norm,
        const ds4_gpu_tensor    *after_attn,
        ds4_gpu_tensor          *next,
        ds4_gpu_tensor          *ffn_gate,
        ds4_gpu_tensor          *ffn_up,
        ds4_gpu_tensor          *ffn_mid,
        ds4_gpu_tensor          *ffn_out,
        ds4_gpu_tensor          *ffn_sum,
        ds4_gpu_tensor          *tmp,
        bool                     stage_profile,
        double                  *stage_t0) {
    uint64_t gate_in = 0, gate_out = 0, gate_row_bytes = 0;
    uint64_t up_in = 0, up_out = 0, up_row_bytes = 0;
    uint64_t down_in = 0, down_out = 0, down_row_bytes = 0;
    (void)tensor_expert_bytes(model, l->ffn_gate_exps, 0, &gate_in, &gate_out, &gate_row_bytes);
    (void)tensor_expert_bytes(model, l->ffn_up_exps, 0, &up_in, &up_out, &up_row_bytes);
    (void)tensor_expert_bytes(model, l->ffn_down_exps, 0, &down_in, &down_out, &down_row_bytes);
    (void)gate_in;
    (void)up_in;
    (void)down_in;

    bool ok = ds4_gpu_matmul_f32_tensor(g->router_logits,
                                        model->map,
                                        model->size,
                                        l->ffn_gate_inp->abs_offset,
                                        DS4_N_EMBD,
                                        DS4_N_EXPERT,
                                        ffn_norm,
                                        1) != 0;
    if (ok) ok = ds4_gpu_glm_router_select_tensor(g->router_selected,
                                                  g->router_weights,
                                                  g->router_probs,
                                                  model->map,
                                                  model->size,
                                                  l->ffn_exp_probs_b->abs_offset,
                                                  g->router_logits,
                                                  DS4_N_EXPERT,
                                                  DS4_N_EXPERT_USED,
                                                  DS4_EXPERT_WEIGHT_SCALE) != 0;
    if (ok) ok = glm_graph_profile_stage(stage_profile,
                                         "glm_decode_ffn",
                                         "router",
                                         il,
                                         pos,
                                         1,
                                         stage_t0);
    if (ok) ok = glm_graph_profile_router_selection(g, l, il, pos);
    const bool resident_decode_layer =
        g->ssd_streaming && glm_stream_resident_decode_layer_enabled(l, il);
    const bool generic_streaming_selected_cache =
        g->ssd_streaming &&
        !resident_decode_layer &&
        glm_graph_layer_uses_generic_routed_moe(l);
    const bool uniform_streaming_selected_cache =
        g->ssd_streaming &&
        !resident_decode_layer &&
        l->ffn_gate_exps->type == l->ffn_up_exps->type &&
        l->ffn_gate_exps->type == l->ffn_down_exps->type &&
        (l->ffn_gate_exps->type == DS4_TENSOR_Q2_K ||
         l->ffn_gate_exps->type == DS4_TENSOR_Q4_K);
    const bool streaming_selected_cache =
        generic_streaming_selected_cache ||
        uniform_streaming_selected_cache;
    const bool shared_first = streaming_selected_cache;
    metal_graph_selected_async_load async_load = {0};
    bool async_load_started = false;
    const bool async_profile =
        streaming_selected_cache &&
        glm_graph_streaming_async_profile_enabled();
    if (async_profile) glm_graph_streaming_async_profile_register();
    const double stream_total_t0 =
        async_profile ? glm_graph_streaming_async_profile_ms() : 0.0;
    double stream_t0 = stream_total_t0;
    bool async_path_profiled = false;
    if (ok && streaming_selected_cache) {
        const ds4_gpu_stream_expert_table table = {
            .model_map = model->map,
            .model_size = model->size,
            .layer = il,
            .n_total_expert = DS4_N_EXPERT,
            .gate_offset = l->ffn_gate_exps->abs_offset,
            .up_offset = l->ffn_up_exps->abs_offset,
            .down_offset = l->ffn_down_exps->abs_offset,
            .gate_expert_bytes = gate_out * gate_row_bytes,
            .down_expert_bytes = down_out * down_row_bytes,
        };
        const bool async_selected_load =
#ifdef DS4_ROCM_BUILD
            streaming_selected_cache &&
#else
            glm_graph_layer_uses_generic_routed_moe(l) &&
#endif
            glm_graph_use_streaming_selected_async_load(g);
        async_path_profiled = false;
        uint64_t selected_event = 0;
        if (async_selected_load) {
            if (ds4_gpu_signal_selected_readback_ready(&selected_event) != 0) {
                async_load_started = metal_graph_selected_async_load_start_tensor(
                        &async_load,
                        g->router_selected,
                        model,
                        l,
                        il,
                        selected_event,
                        gate_out * gate_row_bytes,
                        down_out * down_row_bytes);
                async_path_profiled = async_profile && async_load_started;
            }
            if (async_profile) {
                g_glm_streaming_async_profile.async_signal_start_ms +=
                    glm_graph_streaming_async_profile_ms() - stream_t0;
                stream_t0 = glm_graph_streaming_async_profile_ms();
            }
#ifndef DS4_ROCM_BUILD
            if (ok && async_load_started) {
                ok = ds4_gpu_flush_commands() != 0;
            }
#endif
            if (async_profile) {
                g_glm_streaming_async_profile.async_flush_router_ms +=
                    glm_graph_streaming_async_profile_ms() - stream_t0;
                stream_t0 = glm_graph_streaming_async_profile_ms();
            }
        }
        if (!async_load_started) {
            if (async_selected_load && selected_event != 0) {
                ok = ds4_gpu_wait_selected_readback_ready(
                        selected_event,
                        "selected-id sync expert load fallback") != 0;
            }
            if (ok) {
                ok = ds4_gpu_glm_stream_expert_cache_begin_selected_load_tensor(
                        &table,
                        g->router_selected,
                        DS4_N_EXPERT_USED) != 0;
            }
            if (async_profile) {
                g_glm_streaming_async_profile.sync_early_load_ms +=
                    glm_graph_streaming_async_profile_ms() - stream_t0;
                stream_t0 = glm_graph_streaming_async_profile_ms();
            }
        }
    }
    if (ok && shared_first) {
        ok = glm_graph_encode_shared_swiglu_one(ffn_mid,
                                                ffn_gate,
                                                ffn_up,
                                                model,
                                                l,
                                                il,
                                                pos,
                                                ffn_norm,
                                                g->ssd_streaming,
                                                stage_profile,
                                                stage_t0);
        if (ok) ok = glm_graph_matmul_q8_0_decode_profiled_tensor(ffn_sum,
                                                                  model,
                                                                  l->ffn_down_shexp->abs_offset,
                                                                  DS4_N_FF_EXP,
                                                                  DS4_N_EMBD,
                                                                  ffn_mid,
                                                                  il,
                                                                  pos,
                                                                  "shared_down",
                                                                  g->ssd_streaming) != 0;
        if (ok) ok = glm_graph_profile_stage(stage_profile,
                                             "glm_decode_ffn",
                                             "shared_down",
                                             il,
                                             pos,
                                             1,
                                             stage_t0);
    }
    if (async_profile) {
        const double now_ms = glm_graph_streaming_async_profile_ms();
        if (async_path_profiled) {
            g_glm_streaming_async_profile.async_shared_ms +=
                now_ms - stream_t0;
        } else {
            g_glm_streaming_async_profile.sync_shared_ms +=
                now_ms - stream_t0;
        }
        stream_t0 = now_ms;
    }
    if (async_load_started) {
        bool flush_ok = true;
#ifndef DS4_ROCM_BUILD
        flush_ok = ds4_gpu_flush_commands() != 0;
#endif
        if (async_profile) {
            g_glm_streaming_async_profile.async_flush_shared_ms +=
                glm_graph_streaming_async_profile_ms() - stream_t0;
            stream_t0 = glm_graph_streaming_async_profile_ms();
        }
        const bool finish_ok = metal_graph_selected_async_load_finish(&async_load);
        ok = ok && flush_ok && finish_ok;
        if (async_profile) {
            g_glm_streaming_async_profile.async_finish_ms +=
                glm_graph_streaming_async_profile_ms() - stream_t0;
            stream_t0 = glm_graph_streaming_async_profile_ms();
        }
    }
    /* 50/50 TP: this rank's routed partial goes straight into the
     * slab out slot, the gate exchanges it with the peer's half, and the
     * commutative add rebuilds the full routed output on both ranks
     * bit-identically.  The shared expert and everything else stay
     * replicated, so no other exchange is needed. */
    const bool tp_split_ffn = g->tp_world == 2 && g->tp_out && g->tp_in;
    const uint32_t tp_ffn_slot = il * DS4_TP_GATES_PER_LAYER + DS4_TP_GATE_FFN;
    ds4_gpu_tensor *routed_dst = tp_split_ffn ? g->tp_out[tp_ffn_slot] : ffn_out;
    if (ok && tp_split_ffn && g->ssd_streaming) {
        fprintf(stderr, "ds4: GLM tensor parallelism requires resident weights\n");
        ok = false;
    }
    if (!ok && tp_split_ffn && getenv("DS4_GLM_TP_DEBUG")) {
        fprintf(stderr, "ds4: glm sparse ffn: failed before routed dispatch (layer %u)\n", il);
    }
    if (ok && !(glm_decode_ablate_mask() & DS4_GLM_ABLATE_ROUTED)) {
        ok = glm_graph_routed_moe_one_dispatch(
            g,
            model,
            l,
            il,
            routed_dst,
            ffn_mid,
            gate_out * gate_row_bytes,
            gate_row_bytes,
            up_out * up_row_bytes,
            up_row_bytes,
            down_out * down_row_bytes,
            down_row_bytes,
            g->router_selected,
            g->router_weights,
            ffn_norm,
            resident_decode_layer) != 0;
    }
    if (ok && tp_split_ffn) {
        ok = ds4_gpu_tp_gate_encode(il, DS4_TP_GATE_FFN) != 0;
        if (ok) ok = ds4_gpu_add_tensor(ffn_out,
                                        g->tp_out[tp_ffn_slot],
                                        g->tp_in[tp_ffn_slot],
                                        DS4_N_EMBD) != 0;
        if (!ok) fprintf(stderr, "ds4: GLM TP gate/combine failed (layer %u)\n", il);
    } else if (!ok && tp_split_ffn) {
        fprintf(stderr, "ds4: GLM TP routed dispatch failed before the gate (layer %u)\n", il);
    }
    if (async_profile) {
        const double now_ms = glm_graph_streaming_async_profile_ms();
        if (async_path_profiled) {
            g_glm_streaming_async_profile.async_routed_ms +=
                now_ms - stream_t0;
        } else {
            g_glm_streaming_async_profile.sync_routed_ms +=
                now_ms - stream_t0;
        }
        stream_t0 = now_ms;
    }
    if (ok) ok = glm_graph_profile_stage(stage_profile,
                                         "glm_decode_ffn",
                                         "routed_moe",
                                         il,
                                         pos,
                                         1,
                                         stage_t0);
    if (ok && !shared_first &&
        !(glm_decode_ablate_mask() & DS4_GLM_ABLATE_SHARED)) {
        ok = glm_graph_encode_shared_swiglu_one(ffn_mid,
                                                ffn_gate,
                                                ffn_up,
                                                model,
                                                l,
                                                il,
                                                pos,
                                                ffn_norm,
                                                g->ssd_streaming,
                                                stage_profile,
                                                stage_t0);
        if (ok) ok = glm_graph_matmul_q8_0_decode_profiled_tensor(ffn_sum,
                                                                  model,
                                                                  l->ffn_down_shexp->abs_offset,
                                                                  DS4_N_FF_EXP,
                                                                  DS4_N_EMBD,
                                                                  ffn_mid,
                                                                  il,
                                                                  pos,
                                                                  "shared_down",
                                                                  g->ssd_streaming) != 0;
        if (ok) ok = glm_graph_profile_stage(stage_profile,
                                             "glm_decode_ffn",
                                             "shared_down",
                                             il,
                                             pos,
                                             1,
                                             stage_t0);
    }
    if (ok && !glm_graph_disable_add3_residual()) {
        ok = ds4_gpu_add3_tensor(next,
                                 after_attn,
                                 ffn_out,
                                 ffn_sum,
                                 DS4_N_EMBD) != 0;
    } else if (ok) {
        ok = ds4_gpu_add_tensor(tmp,
                                ffn_out,
                                ffn_sum,
                                DS4_N_EMBD) != 0;
        if (ok) ok = ds4_gpu_add_tensor(next,
                                        after_attn,
                                        tmp,
                                        DS4_N_EMBD) != 0;
    }
    if (ok) ok = glm_graph_profile_stage(stage_profile,
                                         "glm_decode_ffn",
                                         "residual",
                                         il,
                                         pos,
                                         1,
                                         stage_t0);
    if (async_profile) {
        const double now_ms = glm_graph_streaming_async_profile_ms();
        if (async_path_profiled) {
            g_glm_streaming_async_profile.async_calls++;
            g_glm_streaming_async_profile.async_post_ms += now_ms - stream_t0;
            g_glm_streaming_async_profile.async_total_ms +=
                now_ms - stream_total_t0;
        } else {
            g_glm_streaming_async_profile.sync_calls++;
            g_glm_streaming_async_profile.sync_post_ms += now_ms - stream_t0;
            g_glm_streaming_async_profile.sync_total_ms +=
                now_ms - stream_total_t0;
        }
    }
    return ok;
}

static bool glm_graph_encode_ffn_one_normed_from(
        ds4_glm_gpu_graph       *g,
        const ds4_model         *model,
        const ds4_layer_weights *l,
        uint32_t                 il,
        uint32_t                 pos,
        const ds4_gpu_tensor    *ffn_norm,
        const ds4_gpu_tensor    *after_attn,
        ds4_gpu_tensor          *next,
        ds4_gpu_tensor          *ffn_gate,
        ds4_gpu_tensor          *ffn_up,
        ds4_gpu_tensor          *ffn_mid,
        ds4_gpu_tensor          *ffn_out,
        ds4_gpu_tensor          *ffn_sum,
        ds4_gpu_tensor          *tmp,
        bool                     stage_profile,
        double                  *stage_t0) {
    if (!g || !model || !l || !ffn_norm || !after_attn || !next ||
        !ffn_gate || !ffn_up || !ffn_mid || !ffn_out ||
        !ffn_sum || !tmp) {
        return false;
    }

    if (il < DS4_N_LEADING_DENSE) {
        const uint64_t hidden = l->ffn_gate->dim[1];
        const bool can_fuse_gate_up =
            glm_graph_weights_are_q8_0(model,
                                       l->ffn_gate->abs_offset,
                                       l->ffn_up->abs_offset);
        const bool fused_gate_up = can_fuse_gate_up &&
            ds4_gpu_shared_gate_up_swiglu_q8_0_model_view_tensor(
                    ffn_gate,
                    ffn_up,
                    ffn_mid,
                    model->map,
                    model->size,
                    l->ffn_gate->abs_offset,
                    l->ffn_up->abs_offset,
                    DS4_N_EMBD,
                    hidden,
                    ffn_norm,
                    0.0f) != 0;
        bool ok = fused_gate_up;
        if (fused_gate_up) {
            ok = glm_graph_profile_stage(stage_profile,
                                         "glm_decode_ffn",
                                         "dense_gate_up_swiglu",
                                         il,
                                         pos,
                                         1,
                                         stage_t0);
        } else {
            ok = glm_graph_matmul_q8_0_decode_profiled_tensor(ffn_gate,
                                                              model,
                                                              l->ffn_gate->abs_offset,
                                                              DS4_N_EMBD,
                                                              hidden,
                                                              ffn_norm,
                                                              il,
                                                              pos,
                                                              "dense_gate",
                                                              g->ssd_streaming) != 0;
            if (ok) ok = glm_graph_matmul_q8_0_decode_profiled_tensor(ffn_up,
                                                                      model,
                                                                      l->ffn_up->abs_offset,
                                                                      DS4_N_EMBD,
                                                                      hidden,
                                                                      ffn_norm,
                                                                      il,
                                                                      pos,
                                                                      "dense_up",
                                                                      g->ssd_streaming) != 0;
            if (ok) ok = glm_graph_profile_stage(stage_profile,
                                                 "glm_decode_ffn",
                                                 "dense_gate_up",
                                                 il,
                                                 pos,
                                                 1,
                                                 stage_t0);
            if (ok) ok = ds4_gpu_swiglu_tensor(ffn_mid,
                                               ffn_gate,
                                               ffn_up,
                                               (uint32_t)hidden,
                                               0.0f,
                                               1.0f) != 0;
            if (ok) ok = glm_graph_profile_stage(stage_profile,
                                                 "glm_decode_ffn",
                                                 "dense_swiglu",
                                                 il,
                                                 pos,
                                                 1,
                                                 stage_t0);
        }
        if (ok) ok = glm_graph_matmul_q8_0_decode_profiled_tensor(ffn_out,
                                                                  model,
                                                                  l->ffn_down->abs_offset,
                                                                  hidden,
                                                                  DS4_N_EMBD,
                                                                  ffn_mid,
                                                                  il,
                                                                  pos,
                                                                  "dense_down",
                                                                  g->ssd_streaming) != 0;
        if (ok) ok = glm_graph_profile_stage(stage_profile,
                                             "glm_decode_ffn",
                                             "dense_down",
                                             il,
                                             pos,
                                             1,
                                             stage_t0);
        if (ok) ok = ds4_gpu_add_tensor(next,
                                        after_attn,
                                        ffn_out,
                                        DS4_N_EMBD) != 0;
        if (ok) ok = glm_graph_profile_stage(stage_profile,
                                             "glm_decode_ffn",
                                             "residual",
                                             il,
                                             pos,
                                             1,
                                             stage_t0);
        return ok;
    }

    return glm_graph_encode_sparse_ffn_one(g,
                                           model,
                                           l,
                                           il,
                                           pos,
                                           ffn_norm,
                                           after_attn,
                                           next,
                                           ffn_gate,
                                           ffn_up,
                                           ffn_mid,
                                           ffn_out,
                                           ffn_sum,
                                           tmp,
                                           stage_profile,
                                           stage_t0);
}

static bool glm_graph_encode_ffn_one_from(
        ds4_glm_gpu_graph       *g,
        const ds4_model         *model,
        const ds4_layer_weights *l,
        uint32_t                 il,
        uint32_t                 pos,
        const ds4_gpu_tensor    *after_attn,
        ds4_gpu_tensor          *next,
        ds4_gpu_tensor          *ffn_norm,
        ds4_gpu_tensor          *ffn_gate,
        ds4_gpu_tensor          *ffn_up,
        ds4_gpu_tensor          *ffn_mid,
        ds4_gpu_tensor          *ffn_out,
        ds4_gpu_tensor          *ffn_sum,
        ds4_gpu_tensor          *tmp,
        bool                     stage_profile,
        double                  *stage_t0) {
    if (!g || !model || !l || !after_attn || !next ||
        !ffn_norm || !ffn_gate || !ffn_up || !ffn_mid || !ffn_out ||
        !ffn_sum || !tmp) {
        return false;
    }

    bool ok = ds4_gpu_rms_norm_weight_tensor(ffn_norm,
                                             after_attn,
                                             model->map,
                                             model->size,
                                             l->ffn_norm->abs_offset,
                                             DS4_N_EMBD,
                                             DS4_RMS_EPS) != 0;
    if (ok) ok = glm_graph_profile_stage(stage_profile,
                                         "glm_decode_ffn",
                                         "ffn_norm",
                                         il,
                                         pos,
                                         1,
                                         stage_t0);
    if (!ok) return false;

    return glm_graph_encode_ffn_one_normed_from(g,
                                                model,
                                                l,
                                                il,
                                                pos,
                                                ffn_norm,
                                                after_attn,
                                                next,
                                                ffn_gate,
                                                ffn_up,
                                                ffn_mid,
                                                ffn_out,
                                                ffn_sum,
                                                tmp,
                                                stage_profile,
                                                stage_t0);
}

static ds4_gpu_tensor *glm_graph_tensor_row_view_strided(
        ds4_gpu_tensor *base,
        uint32_t        row,
        uint64_t        stride_values,
        uint64_t        row_values) {
    return ds4_gpu_tensor_view(base,
                               (uint64_t)row * stride_values * sizeof(float),
                               row_values * sizeof(float));
}

static uint32_t glm_graph_q8_stripe_tokens(void) {
    return 2048u;
}

static bool glm_graph_flash_attention_prefill_enabled(void) {
    return getenv("DS4_GLM_DISABLE_FLASH_PREFILL") == NULL;
}

static uint32_t glm_graph_flash_attention_prefill_min_tokens(void) {
    return 24u;
}

static bool glm_graph_use_flash_attention_prefill(uint32_t n_tokens) {
    return glm_graph_flash_attention_prefill_enabled() &&
           n_tokens >= glm_graph_flash_attention_prefill_min_tokens();
}

static bool glm_graph_use_flash_attention_staged_kv(
        uint32_t pos0,
        uint32_t n_tokens,
        uint32_t cache_len) {
    return pos0 == 0 &&
           n_tokens == cache_len;
}

static bool glm_graph_force_indexed_decode(void) {
    return false;
}

static bool glm_graph_disable_indexed_decode(void) {
    return false;
}

static bool glm_graph_decode_uses_indexed_attention(const ds4_glm_gpu_graph *g,
                                                    uint32_t                 pos,
                                                    const float             *logits_out) {
    return g && g->compact_cache_cap != 0 &&
           (!g->full_kv_cache ||
            pos >= g->ctx_cap ||
            glm_graph_force_indexed_decode() ||
            (logits_out != NULL && !glm_graph_disable_indexed_decode()));
}

static bool glm_graph_decode_updates_dense_cache(const ds4_glm_gpu_graph *g,
                                                 uint32_t                 pos,
                                                 const float             *logits_out) {
    return g && pos < g->ctx_cap &&
           !glm_graph_decode_uses_indexed_attention(g, pos, logits_out);
}

static bool glm_graph_indexed_prefill_scalar_kernels(void) {
    return false;
}

static bool glm_graph_indexed_prefill_scalar_indexer(void) {
    return false;
}

static bool glm_graph_indexed_prefill_batch_indexer(void) {
    return true;
}

static bool glm_graph_indexed_prefill_scalar_attn(void) {
    return false;
}

static bool glm_graph_indexed_prefill_batch_qk_low(void) {
    return true;
}

static bool glm_graph_indexed_prefill_batch_attn_kernel(void) {
    return true;
}

static uint32_t glm_graph_indexed_prefill_batch_attn_slice_tokens(void) {
    return 2048u;
}

static bool glm_graph_indexer_qat(void) {
    return false;
}

static bool glm_graph_indexed_prefill_batch_ffn(void) {
    return true;
}

static bool glm_graph_indexed_prefill_batch_ffn_norm(void) {
    return true;
}

static bool glm_graph_indexed_prefill_batch_routed_moe(void) {
    return true;
}

static bool glm_graph_indexed_prefill_batch_router_select(void) {
    return true;
}

static bool glm_graph_indexed_prefill_batch_residual(void) {
    return true;
}

static bool glm_graph_indexed_prefill_batch_f32_rows(void) {
    return true;
}

static bool glm_graph_indexed_prefill_batch_q8_rows(void) {
    return true;
}

static bool glm_graph_indexed_prefill_batch_shared_expert(void) {
    return true;
}

static bool glm_graph_matmul_q8_0_tensor(
        ds4_gpu_tensor       *out,
        const ds4_model      *model,
        uint64_t              weight_offset,
        uint64_t              in_dim,
        uint64_t              out_dim,
        const ds4_gpu_tensor *x,
        uint32_t              n_tokens) {
    if (!out || !model || !x || n_tokens == 0) return false;

    const uint32_t q8_stripe_tokens = glm_graph_q8_stripe_tokens();
    if (n_tokens <= q8_stripe_tokens) {
        return ds4_gpu_matmul_quant_tensor(out,
                                           model->map,
                                           model->size,
                                           weight_offset,
                                           glm_graph_weight_type_for_offset(model, weight_offset),
                                           in_dim,
                                           out_dim,
                                           x,
                                           n_tokens) != 0;
    }
    if (in_dim > UINT64_MAX / sizeof(float) ||
        out_dim > UINT64_MAX / sizeof(float)) {
        return false;
    }

    uint32_t done = 0;
    while (done < n_tokens) {
        uint32_t chunk = n_tokens - done;
        if (chunk > q8_stripe_tokens) chunk = q8_stripe_tokens;
        /*
         * The Q8 prefill TensorOps path needs token counts divisible by 32.
         * For a final chunk like 1736 rows, keep the aligned 1728 rows on that
         * path and leave only the tiny tail to the small-batch kernel.  Avoid
         * splitting small batches where the extra launch would dominate.
         */
        if (chunk >= 256u && chunk == n_tokens - done) {
            const uint32_t tail = chunk & 31u;
            if (tail != 0u && tail <= 16u) chunk -= tail;
        }

        ds4_gpu_tensor *x_view = ds4_gpu_tensor_view(
                x,
                (uint64_t)done * in_dim * sizeof(float),
                (uint64_t)chunk * in_dim * sizeof(float));
        ds4_gpu_tensor *out_view = ds4_gpu_tensor_view(
                out,
                (uint64_t)done * out_dim * sizeof(float),
                (uint64_t)chunk * out_dim * sizeof(float));
        const bool ok = x_view && out_view &&
            ds4_gpu_matmul_quant_tensor(out_view,
                                        model->map,
                                        model->size,
                                        weight_offset,
                                        glm_graph_weight_type_for_offset(model, weight_offset),
                                        in_dim,
                                        out_dim,
                                        x_view,
                                        chunk) != 0;
        ds4_gpu_tensor_free(out_view);
        ds4_gpu_tensor_free(x_view);
        if (!ok) return false;
        done += chunk;
    }
    return true;
}

static bool glm_graph_matmul_q8_0_rows_scalar(
        ds4_gpu_tensor       *out,
        const ds4_model      *model,
        uint64_t              weight_offset,
        uint64_t              in_dim,
        uint64_t              out_dim,
        const ds4_gpu_tensor *x,
        uint32_t              n_tokens) {
    if (!out || !model || !x || n_tokens == 0) return false;
    if (in_dim > UINT64_MAX / sizeof(float) ||
        out_dim > UINT64_MAX / sizeof(float)) {
        return false;
    }

    if (glm_graph_indexed_prefill_batch_q8_rows() &&
        ds4_gpu_matmul_quant_rows_scalar_tensor(out,
                                                model->map,
                                                model->size,
                                                weight_offset,
                                                glm_graph_weight_type_for_offset(model, weight_offset),
                                                in_dim,
                                                out_dim,
                                                x,
                                                n_tokens) != 0) {
        return true;
    }

    for (uint32_t t = 0; t < n_tokens; t++) {
        ds4_gpu_tensor *x_view = ds4_gpu_tensor_view(
                (ds4_gpu_tensor *)x,
                (uint64_t)t * in_dim * sizeof(float),
                in_dim * sizeof(float));
        ds4_gpu_tensor *out_view = ds4_gpu_tensor_view(
                out,
                (uint64_t)t * out_dim * sizeof(float),
                out_dim * sizeof(float));
        const bool ok = x_view && out_view &&
            ds4_gpu_matmul_quant_tensor(out_view,
                                        model->map,
                                        model->size,
                                        weight_offset,
                                        glm_graph_weight_type_for_offset(model, weight_offset),
                                        in_dim,
                                        out_dim,
                                        x_view,
                                        1) != 0;
        ds4_gpu_tensor_free(out_view);
        ds4_gpu_tensor_free(x_view);
        if (!ok) return false;
    }
    return true;
}

static bool glm_graph_matmul_f32_rows_scalar(
        ds4_gpu_tensor       *out,
        const ds4_model      *model,
        uint64_t              weight_offset,
        uint64_t              in_dim,
        uint64_t              out_dim,
        const ds4_gpu_tensor *x,
        uint32_t              n_tokens) {
    if (!out || !model || !x || n_tokens == 0) return false;
    if (in_dim > UINT64_MAX / sizeof(float) ||
        out_dim > UINT64_MAX / sizeof(float)) {
        return false;
    }

    if (glm_graph_indexed_prefill_batch_f32_rows()) {
        return ds4_gpu_matmul_f32_tensor(out,
                                         model->map,
                                         model->size,
                                         weight_offset,
                                         in_dim,
                                         out_dim,
                                         x,
                                         n_tokens) != 0;
    }

    for (uint32_t t = 0; t < n_tokens; t++) {
        ds4_gpu_tensor *x_view = ds4_gpu_tensor_view(
                (ds4_gpu_tensor *)x,
                (uint64_t)t * in_dim * sizeof(float),
                in_dim * sizeof(float));
        ds4_gpu_tensor *out_view = ds4_gpu_tensor_view(
                out,
                (uint64_t)t * out_dim * sizeof(float),
                out_dim * sizeof(float));
        const bool ok = x_view && out_view &&
            ds4_gpu_matmul_f32_tensor(out_view,
                                      model->map,
                                      model->size,
                                      weight_offset,
                                      in_dim,
                                      out_dim,
                                      x_view,
                                      1) != 0;
        ds4_gpu_tensor_free(out_view);
        ds4_gpu_tensor_free(x_view);
        if (!ok) return false;
    }
    return true;
}

static bool glm_graph_shared_gate_up_swiglu_q8_0_tensor(
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        const ds4_model      *model,
        uint64_t              gate_offset,
        uint64_t              up_offset,
        uint64_t              in_dim,
        uint64_t              out_dim,
        const ds4_gpu_tensor *x,
        uint32_t              n_tokens,
        float                 clamp) {
    if (!gate || !up || !mid || !model || !x || n_tokens == 0) return false;
    if (!glm_graph_weights_are_q8_0(model, gate_offset, up_offset)) return false;

    const uint32_t q8_stripe_tokens = glm_graph_q8_stripe_tokens();
    if (n_tokens == 1) {
        return ds4_gpu_shared_gate_up_swiglu_q8_0_model_view_tensor(gate,
                                                                    up,
                                                                    mid,
                                                                    model->map,
                                                                    model->size,
                                                                    gate_offset,
                                                                    up_offset,
                                                                    in_dim,
                                                                    out_dim,
                                                                    x,
                                                                    clamp) != 0;
    }
    if (n_tokens <= q8_stripe_tokens) {
        return ds4_gpu_shared_gate_up_swiglu_q8_0_rows_tensor(gate,
                                                              up,
                                                              mid,
                                                              model->map,
                                                              model->size,
                                                              gate_offset,
                                                              up_offset,
                                                              in_dim,
                                                              out_dim,
                                                              x,
                                                              n_tokens,
                                                              clamp) != 0;
    }
    if (in_dim > UINT64_MAX / sizeof(float) ||
        out_dim > UINT64_MAX / sizeof(float)) {
        return false;
    }

    uint32_t done = 0;
    while (done < n_tokens) {
        uint32_t chunk = n_tokens - done;
        if (chunk > q8_stripe_tokens) chunk = q8_stripe_tokens;

        ds4_gpu_tensor *x_view = ds4_gpu_tensor_view(
                x,
                (uint64_t)done * in_dim * sizeof(float),
                (uint64_t)chunk * in_dim * sizeof(float));
        ds4_gpu_tensor *gate_view = ds4_gpu_tensor_view(
                gate,
                (uint64_t)done * out_dim * sizeof(float),
                (uint64_t)chunk * out_dim * sizeof(float));
        ds4_gpu_tensor *up_view = ds4_gpu_tensor_view(
                up,
                (uint64_t)done * out_dim * sizeof(float),
                (uint64_t)chunk * out_dim * sizeof(float));
        ds4_gpu_tensor *mid_view = ds4_gpu_tensor_view(
                mid,
                (uint64_t)done * out_dim * sizeof(float),
                (uint64_t)chunk * out_dim * sizeof(float));
        const bool ok = x_view && gate_view && up_view && mid_view &&
            ds4_gpu_shared_gate_up_swiglu_q8_0_rows_tensor(gate_view,
                                                           up_view,
                                                           mid_view,
                                                           model->map,
                                                           model->size,
                                                           gate_offset,
                                                           up_offset,
                                                           in_dim,
                                                           out_dim,
                                                           x_view,
                                                           chunk,
                                                           clamp) != 0;
        ds4_gpu_tensor_free(mid_view);
        ds4_gpu_tensor_free(up_view);
        ds4_gpu_tensor_free(gate_view);
        ds4_gpu_tensor_free(x_view);
        if (!ok) return false;
        done += chunk;
    }
    return true;
}

static bool glm_graph_indexed_prefill_grouped_moe_default(
        const ds4_glm_gpu_graph *g) {
    return g &&
           !g->quality;
}

static uint32_t glm_graph_streaming_prefill_cache_seed_k(
        const ds4_glm_gpu_graph *g) {
    const bool enabled =
        glm_graph_env_present("DS4_ROCM_ENABLE_STREAMING_PREFILL_CACHE_SEED",
                              "DS4_METAL_ENABLE_STREAMING_PREFILL_CACHE_SEED");
    if (!g ||
        !g->ssd_streaming ||
        !enabled) {
        return 0;
    }

    uint32_t k = 1;
    const char *env = glm_graph_env_value("DS4_ROCM_STREAMING_PREFILL_CACHE_SEED_K",
                                          "DS4_METAL_STREAMING_PREFILL_CACHE_SEED_K");
    if (env && env[0]) {
        char *end = NULL;
        unsigned long v = strtoul(env, &end, 10);
        if (end != env && *end == '\0') {
            if (v == 0) return 0;
            k = v > DS4_STREAMING_PREFILL_CACHE_SEED_MAX_TOKENS ?
                DS4_STREAMING_PREFILL_CACHE_SEED_MAX_TOKENS : (uint32_t)v;
        }
    }
    return k;
}

static bool glm_graph_streaming_prefill_cache_seed_enabled(
        const ds4_glm_gpu_graph *g) {
    return glm_graph_streaming_prefill_cache_seed_k(g) != 0;
}

static void glm_graph_reset_prefill_seed_capture(ds4_glm_gpu_graph *g) {
    if (!g) return;
    g->prefill_seed_tokens = 0;
    memset(g->prefill_seed_layer_captured,
           0,
           sizeof(g->prefill_seed_layer_captured));
}

static bool glm_graph_streaming_expert_cache_seed_layer_expected(
        const ds4_glm_gpu_graph *g,
        const ds4_weights       *weights,
        const ds4_layer_weights *layer,
        uint32_t                 il) {
    if (!g ||
        !g->ssd_streaming ||
        g->quality ||
        !weights ||
        !layer ||
        !layer->ffn_gate_exps ||
        !layer->ffn_up_exps ||
        !layer->ffn_down_exps) {
        return false;
    }
    if (glm_stream_resident_decode_layer_enabled(layer, il)) return false;
    return glm_stream_selected_expert_cache_supported(layer, il) ||
           glm_stream_expert_cache_addr_layout_supported(weights, layer, il);
}

static bool glm_graph_capture_prefill_seed_router_selected(
        ds4_glm_gpu_graph *g,
        uint32_t           il,
        uint32_t           n_tokens) {
    uint32_t k = glm_graph_streaming_prefill_cache_seed_k(g);
    if (k == 0) return true;
    if (!g->prefill_seed_router_selected ||
        !g->batch_router_selected ||
        il >= DS4_N_LAYER ||
        il >= DS4_MAX_LAYER ||
        n_tokens == 0 ||
        DS4_N_EXPERT_USED == 0 ||
        DS4_N_EXPERT_USED > DS4_MAX_EXPERT_USED) {
        return false;
    }
    if (k > n_tokens) k = n_tokens;

    const uint64_t bytes = (uint64_t)k * DS4_N_EXPERT_USED * sizeof(int32_t);
    const uint64_t src_off = (uint64_t)(n_tokens - k) *
                             DS4_N_EXPERT_USED * sizeof(int32_t);
    const uint64_t dst_off = (uint64_t)il *
                             DS4_STREAMING_PREFILL_CACHE_SEED_MAX_TOKENS *
                             DS4_N_EXPERT_USED * sizeof(int32_t);
    if (ds4_gpu_tensor_copy(g->prefill_seed_router_selected,
                            dst_off,
                            g->batch_router_selected,
                            src_off,
                            bytes) == 0) {
        return false;
    }
    g->prefill_seed_tokens = k;
    g->prefill_seed_layer_captured[il] = true;
    return true;
}

static bool glm_graph_seed_streaming_expert_cache_from_prefill(
        ds4_glm_gpu_graph *g,
        const ds4_model   *model,
        const ds4_weights *weights) {
    if (!glm_graph_streaming_prefill_cache_seed_enabled(g)) return true;
    const uint32_t seed_tokens = g ? g->prefill_seed_tokens : 0;
    if (seed_tokens == 0) return true;
    if (!g ||
        !model ||
        !weights ||
        !g->prefill_seed_router_selected ||
        seed_tokens > DS4_STREAMING_PREFILL_CACHE_SEED_MAX_TOKENS ||
        DS4_N_LAYER > DS4_MAX_LAYER ||
        DS4_N_EXPERT_USED == 0 ||
        DS4_N_EXPERT_USED > DS4_MAX_EXPERT_USED) {
        return false;
    }
    bool any_captured = false;
    for (uint32_t il = g->layer_start;
         il <= g->layer_end && il < DS4_N_LAYER && il < DS4_MAX_LAYER;
         il++) {
        if (g->prefill_seed_layer_captured[il]) {
            any_captured = true;
            break;
        }
    }
    if (!any_captured) return true;

    int32_t selected[DS4_MAX_LAYER *
                     DS4_STREAMING_PREFILL_CACHE_SEED_MAX_TOKENS *
                     DS4_MAX_EXPERT_USED];
    const uint64_t bytes = (uint64_t)DS4_N_LAYER *
                           DS4_STREAMING_PREFILL_CACHE_SEED_MAX_TOKENS *
                           DS4_N_EXPERT_USED * sizeof(selected[0]);
    if (ds4_gpu_tensor_read(g->prefill_seed_router_selected,
                            0,
                            selected,
                            bytes) == 0) {
        return false;
    }

    const bool profile =
        glm_graph_env_present("DS4_ROCM_STREAMING_PREFILL_CACHE_SEED_PROFILE",
                              "DS4_METAL_STREAMING_PREFILL_CACHE_SEED_PROFILE");
    const double t0 = profile ? now_sec() : 0.0;
    uint32_t seeded_layers = 0;
    uint32_t seeded_rows = 0;
    for (uint32_t il = g->layer_start;
         il <= g->layer_end && il < DS4_N_LAYER;
         il++) {
        if (il >= DS4_MAX_LAYER || !g->prefill_seed_layer_captured[il]) {
            continue;
        }
        const ds4_layer_weights *layer = &weights->layer[il];
        if (!glm_graph_streaming_expert_cache_seed_layer_expected(g,
                                                                  weights,
                                                                  layer,
                                                                  il)) {
            continue;
        }

        uint64_t gate_expert_bytes = 0;
        uint64_t down_expert_bytes = 0;
        if (!streaming_layer_gate_down_expert_bytes(layer,
                                                    &gate_expert_bytes,
                                                    &down_expert_bytes)) {
            fprintf(stderr,
                    "ds4: GLM prefill expert-cache seed byte size overflow at layer %u\n",
                    il);
            return false;
        }

        const ds4_gpu_stream_expert_table table =
            graph_stream_expert_table_make(model,
                                           layer,
                                           il,
                                           gate_expert_bytes,
                                           down_expert_bytes);
        for (uint32_t row = 0; row < seed_tokens; row++) {
            const size_t sel_off = ((size_t)il *
                                    DS4_STREAMING_PREFILL_CACHE_SEED_MAX_TOKENS +
                                    row) * DS4_N_EXPERT_USED;
            if (ds4_gpu_stream_expert_cache_seed_selected(
                        &table,
                        selected + sel_off,
                        DS4_N_EXPERT_USED) == 0) {
                return false;
            }
            seeded_rows++;
        }
        seeded_layers++;
    }
    if (profile) {
        fprintf(stderr,
                "ds4: GLM streaming prefill expert-cache seed k=%u layers=%u rows=%u time=%.3f ms\n",
                seed_tokens,
                seeded_layers,
                seeded_rows,
                (now_sec() - t0) * 1000.0);
    }
    return true;
}

static bool glm_graph_seed_streaming_expert_cache_from_full_layer(
        ds4_glm_gpu_graph       *g,
        const ds4_model         *model,
        const ds4_weights       *weights,
        const ds4_layer_weights *layer,
        uint32_t                 il,
        uint32_t                 n_tokens,
        uint64_t                 gate_expert_bytes,
        uint64_t                 down_expert_bytes,
        bool                     full_layer_prefill) {
#ifdef DS4_ROCM_BUILD
    uint32_t seed_tokens = glm_graph_streaming_prefill_cache_seed_k(g);
    if (seed_tokens == 0) return true;
    if (!full_layer_prefill ||
        !model ||
        !weights ||
        !layer ||
        !g ||
        !g->batch_router_selected ||
        n_tokens == 0 ||
        il >= DS4_N_LAYER ||
        il >= DS4_MAX_LAYER ||
        gate_expert_bytes == 0 ||
        down_expert_bytes == 0 ||
        !g->prefill_seed_layer_captured[il] ||
        !glm_graph_streaming_expert_cache_seed_layer_expected(g,
                                                              weights,
                                                              layer,
                                                              il)) {
        return true;
    }
    if (seed_tokens > n_tokens) seed_tokens = n_tokens;
    const ds4_gpu_stream_expert_table table =
        graph_stream_expert_table_make(model,
                                       layer,
                                       il,
                                       gate_expert_bytes,
                                       down_expert_bytes);
    if (ds4_gpu_stream_expert_cache_seed_from_layer_selected(
                &table,
                g->batch_router_selected,
                n_tokens,
                seed_tokens,
                DS4_N_EXPERT_USED) != 0) {
        g->prefill_seed_layer_captured[il] = false;
        return true;
    }

    static bool warned = false;
    if (!warned) {
        fprintf(stderr,
                "ds4: GLM ROCm full-layer prefill expert-cache seed skipped; "
                "falling back to end-of-prefill selected seed\n");
        warned = true;
    }
    return true;
#else
    (void)g;
    (void)model;
    (void)weights;
    (void)layer;
    (void)il;
    (void)n_tokens;
    (void)gate_expert_bytes;
    (void)down_expert_bytes;
    (void)full_layer_prefill;
    return true;
#endif
}

static bool glm_graph_disable_add3_residual(void);

static bool glm_graph_encode_sparse_ffn_indexed_batch_routed_moe(
        ds4_glm_gpu_graph       *g,
        const ds4_model         *model,
        const ds4_layer_weights *l,
        uint32_t                 il,
        uint32_t                 pos0,
        const ds4_gpu_tensor    *after_attn,
        ds4_gpu_tensor          *next,
        uint32_t                 n_tokens,
        bool                     stage_profile,
        bool                     stage_sync,
        double                  *stage_t0) {
    if (!g || !model || !l || !after_attn || !next ||
        !g->batch_ffn_norm ||
        !g->batch_router_logits ||
        !g->batch_router_probs ||
        !g->batch_router_selected ||
        !g->batch_router_weights ||
        !g->batch_ffn_out ||
        !g->batch_ffn_mid ||
        n_tokens <= 1 ||
        il < DS4_N_LEADING_DENSE ||
        g->ffn_mid_elems > UINT32_MAX) {
        return false;
    }

    uint64_t gate_in = 0, gate_out = 0, gate_row_bytes = 0;
    uint64_t up_in = 0, up_out = 0, up_row_bytes = 0;
    uint64_t down_in = 0, down_out = 0, down_row_bytes = 0;
    (void)tensor_expert_bytes(model, l->ffn_gate_exps, 0, &gate_in, &gate_out, &gate_row_bytes);
    (void)tensor_expert_bytes(model, l->ffn_up_exps, 0, &up_in, &up_out, &up_row_bytes);
    (void)tensor_expert_bytes(model, l->ffn_down_exps, 0, &down_in, &down_out, &down_row_bytes);
    (void)gate_in;
    (void)up_in;
    (void)down_in;

    bool ok = glm_graph_matmul_f32_rows_scalar(g->batch_router_logits,
                                               model,
                                               l->ffn_gate_inp->abs_offset,
                                               DS4_N_EMBD,
                                               DS4_N_EXPERT,
                                               g->batch_ffn_norm,
                                               n_tokens);
    const bool use_batch_router_select =
        glm_graph_indexed_prefill_batch_router_select();
    if (ok && use_batch_router_select) {
        ok = ds4_gpu_glm_router_select_batch_tensor(g->batch_router_selected,
                                                    g->batch_router_weights,
                                                    g->batch_router_probs,
                                                    model->map,
                                                    model->size,
                                                    l->ffn_exp_probs_b->abs_offset,
                                                    g->batch_router_logits,
                                                    DS4_N_EXPERT,
                                                    DS4_N_EXPERT_USED,
                                                    DS4_EXPERT_WEIGHT_SCALE,
                                                    n_tokens) != 0;
    }
    for (uint32_t t = 0; ok && !use_batch_router_select && t < n_tokens; t++) {
        ds4_gpu_tensor *logits_view =
            glm_graph_tensor_row_view_strided(g->batch_router_logits,
                                              t,
                                              DS4_N_EXPERT,
                                              DS4_N_EXPERT);
        ds4_gpu_tensor *probs_view =
            glm_graph_tensor_row_view_strided(g->batch_router_probs,
                                              t,
                                              DS4_N_EXPERT,
                                              DS4_N_EXPERT);
        ds4_gpu_tensor *selected_view =
            ds4_gpu_tensor_view(g->batch_router_selected,
                                (uint64_t)t * DS4_N_EXPERT_USED * sizeof(int32_t),
                                (uint64_t)DS4_N_EXPERT_USED * sizeof(int32_t));
        ds4_gpu_tensor *weights_view =
            glm_graph_tensor_row_view_strided(g->batch_router_weights,
                                              t,
                                              DS4_N_EXPERT_USED,
                                              DS4_N_EXPERT_USED);
        ok = logits_view && probs_view && selected_view && weights_view;
        if (ok) {
            ok = ds4_gpu_glm_router_select_tensor(selected_view,
                                                  weights_view,
                                                  probs_view,
                                                  model->map,
                                                  model->size,
                                                  l->ffn_exp_probs_b->abs_offset,
                                                  logits_view,
                                                  DS4_N_EXPERT,
                                                  DS4_N_EXPERT_USED,
                                                  DS4_EXPERT_WEIGHT_SCALE) != 0;
        }
        ds4_gpu_tensor_free(weights_view);
        ds4_gpu_tensor_free(selected_view);
        ds4_gpu_tensor_free(probs_view);
        ds4_gpu_tensor_free(logits_view);
    }
    if (ok) ok = glm_graph_prefill_stage_boundary(stage_profile,
                                                  stage_sync,
                                                  "glm_indexed_ffn",
                                                  "router",
                                                  il,
                                                  pos0,
                                                  n_tokens,
                                                  stage_t0);
    if (ok) ok = glm_graph_profile_router_selection_batch(g,
                                                          l,
                                                          il,
                                                          pos0,
                                                          n_tokens);
    if (ok) ok = glm_graph_capture_prefill_seed_router_selected(g,
                                                                il,
                                                                n_tokens);
    metal_graph_debug_dump_tensor("glm_indexed_router_logits",
                                  g->batch_router_logits,
                                  (uint64_t)n_tokens * DS4_N_EXPERT,
                                  il,
                                  pos0);
    metal_graph_debug_dump_i32_tensor("glm_indexed_router_selected",
                                      g->batch_router_selected,
                                      (uint64_t)n_tokens * DS4_N_EXPERT_USED,
                                      il,
                                      pos0);
    metal_graph_debug_dump_tensor("glm_indexed_router_weights",
                                  g->batch_router_weights,
                                  (uint64_t)n_tokens * DS4_N_EXPERT_USED,
                                  il,
                                  pos0);

    const bool tp_batch_split_ffn2 = g->tp_world == 2;
    if (ok && tp_batch_split_ffn2) {
        ok = glm_graph_tp_batch_bounce_ready(g, n_tokens);
    }
    if (ok) {
        const bool use_grouped_moe =
            glm_graph_indexed_prefill_grouped_moe_default(g);
        ok = glm_graph_routed_moe_batch_dispatch(
                g,
                model,
                l,
                il,
                tp_batch_split_ffn2 ? g->tp_bounce_out : g->batch_ffn_out,
                g->batch_ffn_mid,
                gate_out * gate_row_bytes,
                gate_row_bytes,
                up_out * up_row_bytes,
                up_row_bytes,
                down_out * down_row_bytes,
                down_row_bytes,
                g->batch_router_selected,
                g->batch_router_weights,
                g->batch_ffn_norm,
                n_tokens,
                (uint32_t)g->ffn_mid_elems,
                false,
                !use_grouped_moe) != 0;
    }
    if (ok) ok = glm_graph_prefill_stage_boundary(stage_profile,
                                                  stage_sync,
                                                  "glm_indexed_ffn",
                                                  "routed_moe",
                                                  il,
                                                  pos0,
                                                  n_tokens,
                                                  stage_t0);
    metal_graph_debug_dump_tensor("glm_indexed_routed_out",
                                  g->batch_ffn_out,
                                  (uint64_t)n_tokens * DS4_N_EMBD,
                                  il,
                                  pos0);

    const bool use_batch_residual =
        glm_graph_indexed_prefill_batch_residual();
    const uint64_t residual_elems = (uint64_t)n_tokens * DS4_N_EMBD;
    if (use_batch_residual && residual_elems > UINT32_MAX) return false;

    bool shared_expert_done = false;
    if (ok &&
        use_batch_residual &&
        glm_graph_indexed_prefill_batch_shared_expert() &&
        g->batch_ffn_gate &&
        g->batch_ffn_up &&
        g->batch_shared_mid &&
        glm_graph_weights_are_q8_0(model,
                                   l->ffn_gate_shexp->abs_offset,
                                   l->ffn_up_shexp->abs_offset) &&
        ds4_gpu_shared_gate_up_swiglu_q8_0_rows_scalar_tensor(
                g->batch_ffn_gate,
                g->batch_ffn_up,
                g->batch_shared_mid,
                model->map,
                model->size,
                l->ffn_gate_shexp->abs_offset,
                l->ffn_up_shexp->abs_offset,
                DS4_N_EMBD,
                DS4_N_FF_EXP,
                g->batch_ffn_norm,
                n_tokens,
                0.0f) != 0) {
        shared_expert_done =
            glm_graph_matmul_q8_0_rows_scalar(g->batch_attn_out,
                                              model,
                                              l->ffn_down_shexp->abs_offset,
                                              DS4_N_FF_EXP,
                                              DS4_N_EMBD,
                                              g->batch_shared_mid,
                                              n_tokens);
    }

    for (uint32_t t = 0; ok && !shared_expert_done && t < n_tokens; t++) {
        ds4_gpu_tensor *ffn_norm_view =
            glm_graph_tensor_row_view_strided(g->batch_ffn_norm,
                                              t,
                                              DS4_N_EMBD,
                                              DS4_N_EMBD);
        ds4_gpu_tensor *shared_out_view = use_batch_residual ?
            glm_graph_tensor_row_view_strided(g->batch_attn_out,
                                              t,
                                              DS4_N_EMBD,
                                              DS4_N_EMBD) :
            NULL;
        ds4_gpu_tensor *after_attn_view = !use_batch_residual ?
            glm_graph_tensor_row_view_strided((ds4_gpu_tensor *)after_attn,
                                              t,
                                              DS4_N_EMBD,
                                              DS4_N_EMBD) :
            NULL;
        ds4_gpu_tensor *routed_out_view = !use_batch_residual ?
            glm_graph_tensor_row_view_strided(g->batch_ffn_out,
                                              t,
                                              DS4_N_EMBD,
                                              DS4_N_EMBD) :
            NULL;
        ds4_gpu_tensor *next_view = !use_batch_residual ?
            glm_graph_tensor_row_view_strided(next,
                                              t,
                                              DS4_N_EMBD,
                                              DS4_N_EMBD) :
            NULL;
        ok = ffn_norm_view &&
             (use_batch_residual ? (shared_out_view != NULL) :
                                   (after_attn_view && routed_out_view && next_view));
        if (ok && glm_graph_weights_are_q8_0(model,
                                             l->ffn_gate_shexp->abs_offset,
                                             l->ffn_up_shexp->abs_offset)) {
            ok = ds4_gpu_shared_gate_up_swiglu_q8_0_model_view_tensor(
                    g->ffn_gate,
                    g->ffn_up,
                    g->ffn_mid,
                    model->map,
                    model->size,
                    l->ffn_gate_shexp->abs_offset,
                    l->ffn_up_shexp->abs_offset,
                    DS4_N_EMBD,
                    DS4_N_FF_EXP,
                    ffn_norm_view,
                    0.0f) != 0;
        } else if (ok) {
            ok = glm_graph_matmul_q8_0_tensor(g->ffn_gate,
                                              model,
                                              l->ffn_gate_shexp->abs_offset,
                                              DS4_N_EMBD,
                                              DS4_N_FF_EXP,
                                              ffn_norm_view,
                                              1);
            if (ok) ok = glm_graph_matmul_q8_0_tensor(g->ffn_up,
                                                      model,
                                                      l->ffn_up_shexp->abs_offset,
                                                      DS4_N_EMBD,
                                                      DS4_N_FF_EXP,
                                                      ffn_norm_view,
                                                      1);
            if (ok) ok = ds4_gpu_swiglu_tensor(g->ffn_mid,
                                               g->ffn_gate,
                                               g->ffn_up,
                                               DS4_N_FF_EXP,
                                               0.0f,
                                               1.0f) != 0;
        }
        if (ok) ok = glm_graph_matmul_q8_0_tensor(use_batch_residual ?
                                                      shared_out_view :
                                                      g->ffn_sum,
                                                  model,
                                                  l->ffn_down_shexp->abs_offset,
                                                  DS4_N_FF_EXP,
                                                  DS4_N_EMBD,
                                                  g->ffn_mid,
                                                  1);
        if (ok && !use_batch_residual) {
            ok = ds4_gpu_add_tensor(g->attn_out,
                                    routed_out_view,
                                    g->ffn_sum,
                                    DS4_N_EMBD) != 0;
        }
        if (ok && !use_batch_residual) {
            ok = ds4_gpu_add_tensor(next_view,
                                    after_attn_view,
                                    g->attn_out,
                                    DS4_N_EMBD) != 0;
        }

        ds4_gpu_tensor_free(next_view);
        ds4_gpu_tensor_free(routed_out_view);
        ds4_gpu_tensor_free(shared_out_view);
        ds4_gpu_tensor_free(ffn_norm_view);
        ds4_gpu_tensor_free(after_attn_view);
    }
    if (ok) ok = glm_graph_prefill_stage_boundary(stage_profile,
                                                  stage_sync,
                                                  "glm_indexed_ffn",
                                                  "shared_expert",
                                                  il,
                                                  pos0,
                                                  n_tokens,
                                                  stage_t0);
    if (ok && use_batch_residual) {
        if (!glm_graph_disable_add3_residual()) {
            ok = ds4_gpu_add3_tensor(next,
                                     after_attn,
                                     g->batch_ffn_out,
                                     g->batch_attn_out,
                                     (uint32_t)residual_elems) != 0;
        } else {
            ok = ds4_gpu_add_tensor(g->batch_heads,
                                    g->batch_ffn_out,
                                    g->batch_attn_out,
                                    (uint32_t)residual_elems) != 0;
            if (ok) ok = ds4_gpu_add_tensor(next,
                                            after_attn,
                                            g->batch_heads,
                                            (uint32_t)residual_elems) != 0;
        }
    }
    if (ok) ok = glm_graph_prefill_stage_boundary(stage_profile,
                                                  stage_sync,
                                                  "glm_indexed_ffn",
                                                  "residual",
                                                  il,
                                                  pos0,
                                                  n_tokens,
                                                  stage_t0);
    metal_graph_debug_dump_tensor("glm_indexed_next",
                                  next,
                                  (uint64_t)n_tokens * DS4_N_EMBD,
                                  il,
                                  pos0);

    (void)pos0;
    return ok;
}

static bool glm_graph_upload_tokens(
        ds4_gpu_tensor *out_tokens,
        const int      *tokens,
        uint32_t        n_tokens) {
    if (!out_tokens || !tokens || n_tokens == 0) return false;

    int32_t *ids = xmalloc((size_t)n_tokens * sizeof(ids[0]));
    for (uint32_t i = 0; i < n_tokens; i++) ids[i] = (int32_t)tokens[i];
    const bool ok = ds4_gpu_tensor_write(out_tokens,
                                         0,
                                         ids,
                                         (uint64_t)n_tokens * sizeof(ids[0])) != 0;
    free(ids);
    return ok;
}

static bool glm_graph_disable_add3_residual(void) {
    return false;
}

static bool glm_graph_encode_ffn_batch(
        ds4_glm_gpu_graph       *g,
        const ds4_model         *model,
        const ds4_weights       *weights,
        const ds4_layer_weights *l,
        uint32_t                 il,
        uint32_t                 pos0,
        ds4_gpu_tensor          *after_attn,
        ds4_gpu_tensor          *next,
        uint32_t                 n_tokens,
        bool                     full_layer_prefill,
        bool                     stage_profile,
        bool                     stage_sync,
        double                  *stage_t0) {
    if (!g || !model || !weights || !l || !after_attn || !next || n_tokens == 0) return false;

    bool ok = ds4_gpu_rms_norm_weight_rows_tensor(g->batch_ffn_norm,
                                                  after_attn,
                                                  model->map,
                                                  model->size,
                                                  l->ffn_norm->abs_offset,
                                                  DS4_N_EMBD,
                                                  n_tokens,
                                                  DS4_RMS_EPS) != 0;
    if (ok) ok = glm_graph_prefill_stage_boundary(stage_profile,
                                                  stage_sync,
                                                  "glm_ffn",
                                                  "ffn_norm",
                                                  il,
                                                  pos0,
                                                  n_tokens,
                                                  stage_t0);
    if (ok) {
        metal_graph_debug_dump_tensor("glm_ffn_norm",
                                      g->batch_ffn_norm,
                                      (uint64_t)n_tokens * DS4_N_EMBD,
                                      il,
                                      pos0);
    }
    if (!ok) return false;

    if (il < DS4_N_LEADING_DENSE) {
        const uint64_t hidden = l->ffn_gate->dim[1];
        if (hidden == 0 || hidden > UINT32_MAX / n_tokens) return false;
        const uint32_t mid_elems = (uint32_t)(hidden * n_tokens);
        const uint64_t residual_elems = (uint64_t)n_tokens * DS4_N_EMBD;
        if (residual_elems > UINT32_MAX) return false;

        const bool fused_gate_up = glm_graph_shared_gate_up_swiglu_q8_0_tensor(
                g->batch_ffn_gate,
                g->batch_ffn_up,
                g->batch_ffn_mid,
                model,
                l->ffn_gate->abs_offset,
                l->ffn_up->abs_offset,
                DS4_N_EMBD,
                hidden,
                g->batch_ffn_norm,
                n_tokens,
                0.0f);
        if (fused_gate_up) {
            ok = glm_graph_prefill_stage_boundary(stage_profile,
                                                  stage_sync,
                                                  "glm_ffn",
                                                  "dense_gate_up_swiglu",
                                                  il,
                                                  pos0,
                                                  n_tokens,
                                                  stage_t0);
        } else {
            ok = glm_graph_matmul_q8_0_tensor(g->batch_ffn_gate,
                                              model,
                                              l->ffn_gate->abs_offset,
                                              DS4_N_EMBD,
                                              hidden,
                                              g->batch_ffn_norm,
                                              n_tokens);
            if (ok) ok = glm_graph_matmul_q8_0_tensor(g->batch_ffn_up,
                                                      model,
                                                      l->ffn_up->abs_offset,
                                                      DS4_N_EMBD,
                                                      hidden,
                                                      g->batch_ffn_norm,
                                                      n_tokens);
            if (ok) ok = glm_graph_prefill_stage_boundary(stage_profile,
                                                          stage_sync,
                                                          "glm_ffn",
                                                          "dense_gate_up",
                                                          il,
                                                          pos0,
                                                          n_tokens,
                                                          stage_t0);
            if (ok) ok = ds4_gpu_swiglu_tensor(g->batch_ffn_mid,
                                               g->batch_ffn_gate,
                                               g->batch_ffn_up,
                                               mid_elems,
                                               0.0f,
                                               1.0f) != 0;
            if (ok) ok = glm_graph_prefill_stage_boundary(stage_profile,
                                                          stage_sync,
                                                          "glm_ffn",
                                                          "dense_swiglu",
                                                          il,
                                                          pos0,
                                                          n_tokens,
                                                          stage_t0);
        }
        if (ok) ok = glm_graph_matmul_q8_0_tensor(g->batch_ffn_out,
                                                  model,
                                                  l->ffn_down->abs_offset,
                                                  hidden,
                                                  DS4_N_EMBD,
                                                  g->batch_ffn_mid,
                                                  n_tokens);
        if (ok) ok = glm_graph_prefill_stage_boundary(stage_profile,
                                                      stage_sync,
                                                      "glm_ffn",
                                                      "dense_down",
                                                      il,
                                                      pos0,
                                                      n_tokens,
                                                      stage_t0);
        if (ok) ok = ds4_gpu_add_tensor(next,
                                        after_attn,
                                        g->batch_ffn_out,
                                        (uint32_t)residual_elems) != 0;
        if (ok) ok = glm_graph_prefill_stage_boundary(stage_profile,
                                                      stage_sync,
                                                      "glm_ffn",
                                                      "residual",
                                                      il,
                                                      pos0,
                                                      n_tokens,
                                                      stage_t0);
        return ok;
    }

    if (g->ffn_mid_elems > UINT32_MAX ||
        g->dense_hidden_max < DS4_N_FF_EXP ||
        (uint64_t)n_tokens > UINT32_MAX / DS4_N_FF_EXP) {
        return false;
    }
    const uint64_t residual_elems = (uint64_t)n_tokens * DS4_N_EMBD;
    if (residual_elems > UINT32_MAX) return false;

    uint64_t gate_in = 0, gate_out = 0, gate_row_bytes = 0;
    uint64_t up_in = 0, up_out = 0, up_row_bytes = 0;
    uint64_t down_in = 0, down_out = 0, down_row_bytes = 0;
    (void)tensor_expert_bytes(model, l->ffn_gate_exps, 0, &gate_in, &gate_out, &gate_row_bytes);
    (void)tensor_expert_bytes(model, l->ffn_up_exps, 0, &up_in, &up_out, &up_row_bytes);
    (void)tensor_expert_bytes(model, l->ffn_down_exps, 0, &down_in, &down_out, &down_row_bytes);
    (void)gate_in;
    (void)up_in;
    (void)down_in;

    ok = ds4_gpu_matmul_f32_tensor(g->batch_router_logits,
                                   model->map,
                                   model->size,
                                   l->ffn_gate_inp->abs_offset,
                                   DS4_N_EMBD,
                                   DS4_N_EXPERT,
                                   g->batch_ffn_norm,
                                   n_tokens) != 0;
    if (ok) ok = ds4_gpu_glm_router_select_batch_tensor(g->batch_router_selected,
                                                        g->batch_router_weights,
                                                        g->batch_router_probs,
                                                        model->map,
                                                        model->size,
                                                        l->ffn_exp_probs_b->abs_offset,
                                                        g->batch_router_logits,
                                                        DS4_N_EXPERT,
                                                        DS4_N_EXPERT_USED,
                                                        DS4_EXPERT_WEIGHT_SCALE,
                                                        n_tokens) != 0;
    if (ok) ok = glm_graph_prefill_stage_boundary(stage_profile,
                                                  stage_sync,
                                                  "glm_ffn",
                                                  "router",
                                                  il,
                                                  pos0,
                                                  n_tokens,
                                                  stage_t0);
    if (ok) {
        metal_graph_debug_dump_tensor("glm_ffn_router_logits",
                                      g->batch_router_logits,
                                      (uint64_t)n_tokens * DS4_N_EXPERT,
                                      il,
                                      pos0);
        metal_graph_debug_dump_tensor("glm_ffn_router_probs",
                                      g->batch_router_probs,
                                      (uint64_t)n_tokens * DS4_N_EXPERT,
                                      il,
                                      pos0);
        metal_graph_debug_dump_i32_tensor("glm_ffn_router_selected",
                                          g->batch_router_selected,
                                          (uint64_t)n_tokens * DS4_N_EXPERT_USED,
                                          il,
                                          pos0);
        metal_graph_debug_dump_tensor("glm_ffn_router_weights",
                                      g->batch_router_weights,
                                      (uint64_t)n_tokens * DS4_N_EXPERT_USED,
                                      il,
                                      pos0);
    }
    if (ok) ok = glm_graph_profile_router_selection_batch(g,
                                                          l,
                                                          il,
                                                          pos0,
                                                          n_tokens);
    const bool tp_batch_split_ffn = g->tp_world == 2;
    if (ok && tp_batch_split_ffn) {
        ok = glm_graph_tp_batch_bounce_ready(g, n_tokens);
    }
    if (ok) ok = glm_graph_capture_prefill_seed_router_selected(g,
                                                                il,
                                                                n_tokens);
    if (ok) ok = glm_graph_seed_streaming_expert_cache_from_full_layer(
            g,
            model,
            weights,
            l,
            il,
            n_tokens,
            gate_out * gate_row_bytes,
            down_out * down_row_bytes,
            full_layer_prefill);
    bool shared_done = false;
#define DS4_GLM_ENCODE_FFN_BATCH_SHARED() do { \
        if (ok) { \
            const bool fused_shared = glm_graph_shared_gate_up_swiglu_q8_0_tensor( \
                    g->batch_ffn_gate, \
                    g->batch_ffn_up, \
                    g->batch_shared_mid, \
                    model, \
                    l->ffn_gate_shexp->abs_offset, \
                    l->ffn_up_shexp->abs_offset, \
                    DS4_N_EMBD, \
                    DS4_N_FF_EXP, \
                    g->batch_ffn_norm, \
                    n_tokens, \
                    0.0f); \
            if (fused_shared) { \
                ok = glm_graph_prefill_stage_boundary(stage_profile, \
                                                      stage_sync, \
                                                      "glm_ffn", \
                                                      "shared_gate_up_swiglu", \
                                                      il, \
                                                      pos0, \
                                                      n_tokens, \
                                                      stage_t0); \
            } else { \
                ok = glm_graph_matmul_q8_0_tensor(g->batch_ffn_gate, \
                                                  model, \
                                                  l->ffn_gate_shexp->abs_offset, \
                                                  DS4_N_EMBD, \
                                                  DS4_N_FF_EXP, \
                                                  g->batch_ffn_norm, \
                                                  n_tokens); \
                if (ok) ok = glm_graph_matmul_q8_0_tensor(g->batch_ffn_up, \
                                                          model, \
                                                          l->ffn_up_shexp->abs_offset, \
                                                          DS4_N_EMBD, \
                                                          DS4_N_FF_EXP, \
                                                          g->batch_ffn_norm, \
                                                          n_tokens); \
                if (ok) ok = glm_graph_prefill_stage_boundary(stage_profile, \
                                                              stage_sync, \
                                                              "glm_ffn", \
                                                              "shared_gate_up", \
                                                              il, \
                                                              pos0, \
                                                              n_tokens, \
                                                              stage_t0); \
                if (ok) ok = ds4_gpu_swiglu_tensor(g->batch_shared_mid, \
                                                   g->batch_ffn_gate, \
                                                   g->batch_ffn_up, \
                                                   n_tokens * DS4_N_FF_EXP, \
                                                   0.0f, \
                                                   1.0f) != 0; \
                if (ok) ok = glm_graph_prefill_stage_boundary(stage_profile, \
                                                              stage_sync, \
                                                              "glm_ffn", \
                                                              "shared_swiglu", \
                                                              il, \
                                                              pos0, \
                                                              n_tokens, \
                                                              stage_t0); \
            } \
        } \
        if (ok) ok = glm_graph_matmul_q8_0_tensor(g->batch_attn_out, \
                                                  model, \
                                                  l->ffn_down_shexp->abs_offset, \
                                                  DS4_N_FF_EXP, \
                                                  DS4_N_EMBD, \
                                                  g->batch_shared_mid, \
                                                  n_tokens); \
        if (ok) ok = glm_graph_prefill_stage_boundary(stage_profile, \
                                                      stage_sync, \
                                                      "glm_ffn", \
                                                      "shared_down", \
                                                      il, \
                                                      pos0, \
                                                      n_tokens, \
                                                      stage_t0); \
        if (ok) shared_done = true; \
    } while (0)
#ifdef DS4_ROCM_BUILD
    rocm_graph_batch_selected_async_load rocm_batch_selected_async = {0};
    bool rocm_batch_selected_async_started = false;
    const bool rocm_batch_selected_shared_overlap =
        ok &&
        g->ssd_streaming &&
        !g->quality &&
        n_tokens > 1 &&
        !full_layer_prefill &&
        !glm_graph_env_present(
                "DS4_ROCM_DISABLE_GLM_STREAMING_PREFILL_SELECTED_ASYNC_LOAD",
                "DS4_METAL_DISABLE_GLM_STREAMING_PREFILL_SELECTED_ASYNC_LOAD") &&
        !glm_graph_env_present(
                "DS4_ROCM_DISABLE_STREAMING_PREFILL_SELECTED_ASYNC_LOAD",
                "DS4_METAL_DISABLE_STREAMING_PREFILL_SELECTED_ASYNC_LOAD") &&
        glm_graph_stream_prefill_expert_addr_supported(weights, l, il, n_tokens);
    if (rocm_batch_selected_shared_overlap) {
        uint64_t selected_event = 0;
        if (ds4_gpu_signal_selected_readback_ready(&selected_event) != 0) {
            rocm_batch_selected_async_started =
                rocm_graph_batch_selected_async_load_start(
                    &rocm_batch_selected_async,
                    g->batch_router_selected,
                    model,
                    l,
                    il,
                    n_tokens,
                    selected_event,
                    gate_out * gate_row_bytes,
                    down_out * down_row_bytes);
        }
    }
    if (rocm_batch_selected_async_started) {
        DS4_GLM_ENCODE_FFN_BATCH_SHARED();
        const bool finish_ok =
            rocm_graph_batch_selected_async_load_finish(&rocm_batch_selected_async);
        if (!finish_ok) rocm_batch_selected_async_started = false;
    }
#endif
    if (n_tokens <= 8u && (glm_decode_ablate_mask() & DS4_GLM_ABLATE_ROUTED)) { /* ablate: keep the gate */ } else
    if (ok) ok = glm_graph_routed_moe_batch_dispatch(
            g,
            model,
            l,
            il,
            tp_batch_split_ffn ? g->tp_bounce_out : g->batch_ffn_out,
            g->batch_ffn_mid,
            gate_out * gate_row_bytes,
            gate_row_bytes,
            up_out * up_row_bytes,
            up_row_bytes,
            down_out * down_row_bytes,
            down_row_bytes,
            g->batch_router_selected,
            g->batch_router_weights,
            g->batch_ffn_norm,
            n_tokens,
            (uint32_t)g->ffn_mid_elems,
            full_layer_prefill,
            false) != 0;
    if (ok && g->tp_world == 2) {
        ok = glm_graph_tp_batch_ffn_combine(g, il, g->batch_ffn_out, n_tokens);
        if (!ok) fprintf(stderr, "ds4: GLM TP batch gate failed (layer %u)\n", il);
    }
    if (ok) ok = glm_graph_prefill_stage_boundary(stage_profile,
                                                  stage_sync,
                                                  "glm_ffn",
                                                  "routed_moe",
                                                  il,
                                                  pos0,
                                                  n_tokens,
                                                  stage_t0);
    if (ok) {
        metal_graph_debug_dump_tensor("glm_ffn_routed_out",
                                      g->batch_ffn_out,
                                      (uint64_t)n_tokens * DS4_N_EMBD,
                                      il,
                                      pos0);
    }
    if (ok && !shared_done) DS4_GLM_ENCODE_FFN_BATCH_SHARED();
#undef DS4_GLM_ENCODE_FFN_BATCH_SHARED
    if (ok) {
        metal_graph_debug_dump_tensor("glm_ffn_shared_out",
                                      g->batch_attn_out,
                                      (uint64_t)n_tokens * DS4_N_EMBD,
                                      il,
                                      pos0);
    }
    if (ok && !glm_graph_disable_add3_residual()) {
        ok = ds4_gpu_add3_tensor(next,
                                 after_attn,
                                 g->batch_ffn_out,
                                 g->batch_attn_out,
                                 (uint32_t)residual_elems) != 0;
    } else if (ok) {
        ok = ds4_gpu_add_tensor(g->batch_heads,
                                g->batch_ffn_out,
                                g->batch_attn_out,
                                (uint32_t)residual_elems) != 0;
        if (ok) ok = ds4_gpu_add_tensor(next,
                                        after_attn,
                                        g->batch_heads,
                                        (uint32_t)residual_elems) != 0;
    }
    if (ok) ok = glm_graph_prefill_stage_boundary(stage_profile,
                                                  stage_sync,
                                                  "glm_ffn",
                                                  "residual",
                                                  il,
                                                  pos0,
                                                  n_tokens,
                                                  stage_t0);
    if (ok) {
        metal_graph_debug_dump_tensor("glm_ffn_next",
                                      next,
                                      (uint64_t)n_tokens * DS4_N_EMBD,
                                      il,
                                      pos0);
    }
    return ok;
}


static bool glm_graph_begin_commands_if_needed(void);

/* ------------------------------------------------------------------------
 * GLM MTP (nextn block) drafting.
 *
 * blk.(N_LAYER-1) is GLM 5.2's multi-token-prediction block: a full
 * attention+MoE layer fed with eh_proj(concat(enorm(embed(token[p+1])),
 * hnorm(h[p]))), predicting token[p+2] through the shared output head.
 * It keeps a private compact KV cache (slot = absolute position; only
 * positions >= mtp_min_pos are ever selected, so the unwritten prompt
 * range is never read). Under TP the routed experts are combined over
 * the BIG-gate exchange, never the decode row gate, so the RDMA row-gate
 * schedule stays intact.
 * --------------------------------------------------------------------- */
static bool glm_graph_mtp_ensure(ds4_glm_gpu_graph *g) {
    if (g->mtp_ready) return true;
    if (g->compact_cache_cap == 0) return false;
    const uint64_t elem = glm_graph_compact_cache_elem_bytes();
    const uint64_t kv_bytes = (uint64_t)g->compact_cache_cap * DS4_N_KV_LORA * elem;
    const uint64_t rope_bytes = (uint64_t)g->compact_cache_cap * DS4_N_ROT * elem;
    g->mtp_kv_lora_cache = ds4_gpu_tensor_alloc(kv_bytes);
    g->mtp_k_rope_cache = ds4_gpu_tensor_alloc(rope_bytes);
    g->mtp_concat = ds4_gpu_tensor_alloc(2ull * DS4_N_EMBD * sizeof(float));
    g->mtp_selected = ds4_gpu_tensor_alloc((uint64_t)g->compact_cache_cap * sizeof(int32_t));
    g->mtp_logits_host = malloc((size_t)DS4_N_VOCAB * sizeof(float));
    if (!g->mtp_kv_lora_cache || !g->mtp_k_rope_cache || !g->mtp_concat ||
        !g->mtp_selected || !g->mtp_logits_host) {
        return false;
    }
    g->mtp_ready = 1;
    return true;
}

/* One MTP step at (absolute) position pos: consumes the main model's last
 * hidden h[pos] (expected in g->cur, pre output-norm) and next_token
 * (= token[pos+1]), writes the nextn KV at slot pos, and returns the
 * drafted token[pos+2] by greedy argmax. Clobbers the decode scratch. */
static bool glm_graph_mtp_step(
        ds4_glm_gpu_graph *g,
        const ds4_model   *model,
        const ds4_weights *weights,
        int                next_token,
        uint32_t           pos,
        uint32_t           min_pos,
        int               *draft_out) {
    if (!g || !model || !weights || !draft_out) return false;
    if (DS4_N_NEXTN_PREDICT == 0) return false;
    if (pos >= g->compact_cache_cap || min_pos > pos) {
        fprintf(stderr, "ds4: glm mtp: pos %u/min %u out of range (cap %u)\n",
                pos, min_pos, g->compact_cache_cap);
        return false;
    }
    const uint32_t il = DS4_N_LAYER - DS4_N_NEXTN_PREDICT;
    const ds4_layer_weights *l = &weights->layer[il];
    if (!l->nextn_eh_proj || !l->nextn_enorm || !l->nextn_hnorm ||
        !l->nextn_shared_head_norm || !l->ffn_gate_exps) {
        fprintf(stderr, "ds4: glm mtp: nextn weights missing at layer %u\n", il);
        return false;
    }
    const uint32_t kv_raw_dim = (uint32_t)l->attn_kv_a_mqa->dim[1];
    const float rope_base = layer_rope_freq_base(il);
    const float rope_scale = layer_rope_freq_scale(il);
    const uint32_t n_selected = pos - min_pos + 1u;
    bool input_ready = false;

    if (g->placement) {
        const int embedding_tier = g->placement[0];
        const int mtp_tier = g->placement[il + 1u];
        bool handoff_ok = glm_graph_ws_switch(g, embedding_tier, true);
        if (handoff_ok) handoff_ok = glm_graph_begin_commands_if_needed();
        if (handoff_ok) {
            handoff_ok = ds4_gpu_embed_token_quant_tensor(
                    g->next,
                    model->map,
                    model->size,
                    weights->token_embd->abs_offset,
                    weights->token_embd->type,
                    DS4_N_VOCAB,
                    (uint32_t)next_token,
                    DS4_N_EMBD) != 0;
        }
        if (handoff_ok) handoff_ok = ds4_gpu_end_commands() != 0;
        else (void)ds4_gpu_synchronize();
        ds4_gpu_tensor *embedded_next = handoff_ok ? g->next : NULL;
        if (handoff_ok) handoff_ok = glm_graph_ws_switch(g, mtp_tier, true);
        if (handoff_ok) {
            handoff_ok = ds4_gpu_tensor_copy_async(
                    g->next, embedded_next,
                    (uint64_t)DS4_N_EMBD * sizeof(float)) != 0;
        }
        if (!handoff_ok) {
            fprintf(stderr, "ds4: glm mtp: multi-tier input handoff failed\n");
            return false;
        }
        input_ready = true;
    }
    if (!glm_graph_mtp_ensure(g)) {
        fprintf(stderr, "ds4: glm mtp: ensure failed (cap %u)\n", g->compact_cache_cap);
        return false;
    }

    /* Draft attention window: absolute cache slots [min_pos..pos]. */
    {
        int32_t *sel = malloc((size_t)n_selected * sizeof(int32_t));
        if (!sel) return false;
        for (uint32_t i = 0; i < n_selected; i++) sel[i] = (int32_t)(min_pos + i);
        const int wr = ds4_gpu_tensor_write(g->mtp_selected, 0, sel,
                                            (uint64_t)n_selected * sizeof(int32_t));
        free(sel);
        if (!wr) {
            fprintf(stderr, "ds4: glm mtp: selected write failed (%u)\n", n_selected);
            return false;
        }
    }

    ds4_gpu_tensor *enorm_view =
        ds4_gpu_tensor_view(g->mtp_concat, 0, (uint64_t)DS4_N_EMBD * sizeof(float));
    ds4_gpu_tensor *hnorm_view =
        ds4_gpu_tensor_view(g->mtp_concat,
                            (uint64_t)DS4_N_EMBD * sizeof(float),
                            (uint64_t)DS4_N_EMBD * sizeof(float));
    if (!enorm_view || !hnorm_view) {
        ds4_gpu_tensor_free(enorm_view);
        ds4_gpu_tensor_free(hnorm_view);
        fprintf(stderr, "ds4: glm mtp: concat views failed\n");
        return false;
    }

    const char *mtp_stage = "begin";
#define DS4_GLM_MTP_STAGE(name_) do { if (ok) mtp_stage = (name_); } while (0)
    bool ok = glm_graph_begin_commands_if_needed();
    /* MTP input: concat(enorm(embed(next_token)), hnorm(h)) -> eh_proj. */
    if (ok && !input_ready) {
        ok = ds4_gpu_embed_token_quant_tensor(g->next,
                                              model->map,
                                              model->size,
                                              weights->token_embd->abs_offset,
                                              weights->token_embd->type,
                                              DS4_N_VOCAB,
                                              (uint32_t)next_token,
                                              DS4_N_EMBD) != 0;
    }
    DS4_GLM_MTP_STAGE("enorm");
    if (ok) ok = ds4_gpu_rms_norm_weight_tensor(enorm_view,
                                                g->next,
                                                model->map,
                                                model->size,
                                                l->nextn_enorm->abs_offset,
                                                DS4_N_EMBD,
                                                DS4_RMS_EPS) != 0;
    DS4_GLM_MTP_STAGE("hnorm");
    if (ok) ok = ds4_gpu_rms_norm_weight_tensor(hnorm_view,
                                                g->cur,
                                                model->map,
                                                model->size,
                                                l->nextn_hnorm->abs_offset,
                                                DS4_N_EMBD,
                                                DS4_RMS_EPS) != 0;
    DS4_GLM_MTP_STAGE("eh_proj");
    if (ok) ok = glm_graph_matmul_q8_0_decode_tensor(g->cur,
                                                     model,
                                                     l->nextn_eh_proj->abs_offset,
                                                     2ull * DS4_N_EMBD,
                                                     DS4_N_EMBD,
                                                     g->mtp_concat,
                                                     false);
    /* nextn attention (full causal over the MTP window, no indexer). */
    DS4_GLM_MTP_STAGE("attn_norm");
    if (ok) ok = ds4_gpu_rms_norm_weight_tensor(g->attn_norm,
                                                g->cur,
                                                model->map,
                                                model->size,
                                                l->attn_norm->abs_offset,
                                                DS4_N_EMBD,
                                                DS4_RMS_EPS) != 0;
    DS4_GLM_MTP_STAGE("q_a");
    if (ok) ok = glm_graph_matmul_q8_0_decode_tensor(g->q_rank,
                                                     model,
                                                     l->attn_q_a->abs_offset,
                                                     DS4_N_EMBD,
                                                     DS4_N_LORA_Q,
                                                     g->attn_norm,
                                                     false);
    DS4_GLM_MTP_STAGE("q_a_norm");
    if (ok) ok = ds4_gpu_rms_norm_weight_tensor(g->q_rank_norm,
                                                g->q_rank,
                                                model->map,
                                                model->size,
                                                l->attn_q_a_norm->abs_offset,
                                                DS4_N_LORA_Q,
                                                DS4_RMS_EPS) != 0;
    DS4_GLM_MTP_STAGE("q_b");
    if (ok) ok = glm_graph_matmul_q8_0_decode_tensor(g->q,
                                                     model,
                                                     l->attn_q_b->abs_offset,
                                                     DS4_N_LORA_Q,
                                                     g->q_dim,
                                                     g->q_rank_norm,
                                                     false);
    DS4_GLM_MTP_STAGE("rope");
    if (ok) ok = ds4_gpu_glm_rope_tail_tensor(g->q,
                                              1,
                                              DS4_N_HEAD,
                                              DS4_N_KEY_MLA,
                                              DS4_N_ROT,
                                              pos,
                                              0,
                                              rope_base,
                                              rope_scale,
                                              0.0f,
                                              1.0f,
                                              DS4_ROPE_YARN_BETA_FAST,
                                              DS4_ROPE_YARN_BETA_SLOW) != 0;
    DS4_GLM_MTP_STAGE("kv_a");
    if (ok) ok = glm_graph_matmul_q8_0_decode_tensor(g->kv_raw,
                                                     model,
                                                     l->attn_kv_a_mqa->abs_offset,
                                                     DS4_N_EMBD,
                                                     kv_raw_dim,
                                                     g->attn_norm,
                                                     false);
    DS4_GLM_MTP_STAGE("kv_norm");
    if (ok) ok = ds4_gpu_glm_kv_lora_rms_norm_tensor(g->kv_norm,
                                                     g->kv_raw,
                                                     model->map,
                                                     model->size,
                                                     l->attn_kv_a_norm->abs_offset,
                                                     1,
                                                     kv_raw_dim,
                                                     DS4_N_KV_LORA,
                                                     DS4_RMS_EPS) != 0;
    DS4_GLM_MTP_STAGE("kv_store");
    if (ok) ok = ds4_gpu_glm_store_compact_kv_tensor(g->mtp_kv_lora_cache,
                                                     g->mtp_k_rope_cache,
                                                     g->kv_norm,
                                                     g->kv_raw,
                                                     pos,
                                                     1,
                                                     g->compact_cache_cap,
                                                     kv_raw_dim,
                                                     DS4_N_KV_LORA,
                                                     DS4_N_ROT,
                                                     glm_graph_compact_cache_is_f16()) != 0;
    DS4_GLM_MTP_STAGE("qk_low");
    if (ok) ok = ds4_gpu_glm_qk_lowrank_typed_tensor(g->qk_low,
                                                     g->q,
                                                     model->map,
                                                     model->size,
                                                     l->attn_k_b->abs_offset,
                                                     l->attn_k_b->type,
                                                     DS4_N_HEAD,
                                                     DS4_N_KV_LORA,
                                                     (uint32_t)g->q_nope,
                                                     DS4_N_KEY_MLA) != 0;
    DS4_GLM_MTP_STAGE("attention");
    if (ok) ok = ds4_gpu_glm_attention_indexed_decode_typed_tensor(g->heads,
                                                                   g->q,
                                                                   g->qk_low,
                                                                   g->mtp_kv_lora_cache,
                                                                   g->mtp_k_rope_cache,
                                                                   model->map,
                                                                   model->size,
                                                                   l->attn_v_b->abs_offset,
                                                                   l->attn_v_b->type,
                                                                   g->mtp_selected,
                                                                   n_selected,
                                                                   g->compact_cache_cap,
                                                                   glm_graph_compact_cache_is_f16(),
                                                                   DS4_N_HEAD,
                                                                   DS4_N_KV_LORA,
                                                                   (uint32_t)g->q_nope,
                                                                   DS4_N_ROT,
                                                                   DS4_N_VALUE_MLA,
                                                                   0,
                                                                   rope_base,
                                                                   rope_scale,
                                                                   0.0f,
                                                                   1.0f,
                                                                   DS4_ROPE_YARN_BETA_FAST,
                                                                   DS4_ROPE_YARN_BETA_SLOW) != 0;
    DS4_GLM_MTP_STAGE("attn_out");
    if (ok) ok = glm_graph_matmul_q8_0_decode_tensor(g->attn_out,
                                                     model,
                                                     l->attn_output->abs_offset,
                                                     g->heads_dim,
                                                     DS4_N_EMBD,
                                                     g->heads,
                                                     false);
    DS4_GLM_MTP_STAGE("ffn_norm");
    if (ok) ok = ds4_gpu_add_rms_norm_weight_tensor(g->ffn_norm,
                                                    g->after_attn,
                                                    g->cur,
                                                    g->attn_out,
                                                    model->map,
                                                    model->size,
                                                    l->ffn_norm->abs_offset,
                                                    DS4_N_EMBD,
                                                    DS4_RMS_EPS) != 0;
    /* nextn sparse FFN: router + split routed experts (BIG-gate combine
     * under TP) + shared expert. */
    DS4_GLM_MTP_STAGE("router");
    if (ok) ok = ds4_gpu_matmul_f32_tensor(g->router_logits,
                                           model->map,
                                           model->size,
                                           l->ffn_gate_inp->abs_offset,
                                           DS4_N_EMBD,
                                           DS4_N_EXPERT,
                                           g->ffn_norm,
                                           1) != 0;
    DS4_GLM_MTP_STAGE("router_select");
    if (ok) ok = ds4_gpu_glm_router_select_tensor(g->router_selected,
                                                  g->router_weights,
                                                  g->router_probs,
                                                  model->map,
                                                  model->size,
                                                  l->ffn_exp_probs_b->abs_offset,
                                                  g->router_logits,
                                                  DS4_N_EXPERT,
                                                  DS4_N_EXPERT_USED,
                                                  DS4_EXPERT_WEIGHT_SCALE) != 0;
    DS4_GLM_MTP_STAGE("routed");
    if (ok) {
        uint64_t gate_in = 0, gate_out = 0, gate_row_bytes = 0;
        uint64_t up_in = 0, up_out = 0, up_row_bytes = 0;
        uint64_t down_in = 0, down_out = 0, down_row_bytes = 0;
        (void)tensor_expert_bytes(model, l->ffn_gate_exps, 0, &gate_in, &gate_out, &gate_row_bytes);
        (void)tensor_expert_bytes(model, l->ffn_up_exps, 0, &up_in, &up_out, &up_row_bytes);
        (void)tensor_expert_bytes(model, l->ffn_down_exps, 0, &down_in, &down_out, &down_row_bytes);
#ifdef DS4_ROCM_BUILD
        if (g->ssd_streaming) {
            const ds4_gpu_stream_expert_table table = {
                .model_map = model->map,
                .model_size = model->size,
                .layer = il,
                .n_total_expert = DS4_N_EXPERT,
                .gate_offset = l->ffn_gate_exps->abs_offset,
                .up_offset = l->ffn_up_exps->abs_offset,
                .down_offset = l->ffn_down_exps->abs_offset,
                .gate_expert_bytes = gate_out * gate_row_bytes,
                .down_expert_bytes = down_out * down_row_bytes,
            };
            ok = ds4_gpu_glm_stream_expert_cache_begin_selected_load_tensor(
                    &table, g->router_selected, DS4_N_EXPERT_USED) != 0;
        }
#endif
        const bool tp_split = g->tp_world == 2 && g->tp_out && g->tp_in;
        ds4_gpu_tensor *routed_dst = g->ffn_out;
        if (tp_split) {
            ok = glm_graph_tp_batch_bounce_ready(g, 1);
            routed_dst = g->tp_bounce_out;
        }
        if (ok) ok = glm_graph_routed_moe_one_dispatch(g,
                                                       model,
                                                       l,
                                                       il,
                                                       routed_dst,
                                                       g->ffn_mid,
                                                       gate_out * gate_row_bytes,
                                                       gate_row_bytes,
                                                       up_out * up_row_bytes,
                                                       up_row_bytes,
                                                       down_out * down_row_bytes,
                                                       down_row_bytes,
                                                       g->router_selected,
                                                       g->router_weights,
                                                       g->ffn_norm,
                                                       false) != 0;
        if (ok && tp_split) {
            ok = glm_graph_tp_batch_ffn_combine(g, il, g->ffn_out, 1);
        }
    }
    DS4_GLM_MTP_STAGE("shared");
    if (ok) ok = glm_graph_encode_shared_swiglu_one(g->ffn_mid,
                                                    g->ffn_gate,
                                                    g->ffn_up,
                                                    model,
                                                    l,
                                                    il,
                                                    pos,
                                                    g->ffn_norm,
                                                    false,
                                                    false,
                                                    NULL);
    DS4_GLM_MTP_STAGE("shared_down");
    if (ok) ok = glm_graph_matmul_q8_0_decode_tensor(g->ffn_sum,
                                                     model,
                                                     l->ffn_down_shexp->abs_offset,
                                                     DS4_N_FF_EXP,
                                                     DS4_N_EMBD,
                                                     g->ffn_mid,
                                                     false);
    DS4_GLM_MTP_STAGE("residual");
    if (ok) ok = ds4_gpu_add3_tensor(g->next,
                                     g->after_attn,
                                     g->ffn_out,
                                     g->ffn_sum,
                                     DS4_N_EMBD) != 0;
    /* Shared output head behind the nextn head norm. */
    DS4_GLM_MTP_STAGE("head_norm");
    if (ok) ok = ds4_gpu_rms_norm_weight_tensor(g->output_norm,
                                                g->next,
                                                model->map,
                                                model->size,
                                                l->nextn_shared_head_norm->abs_offset,
                                                DS4_N_EMBD,
                                                DS4_RMS_EPS) != 0;
    DS4_GLM_MTP_STAGE("head");
    if (ok) ok = glm_graph_matmul_q8_0_decode_tensor(g->logits,
                                                     model,
                                                     weights->output->abs_offset,
                                                     DS4_N_EMBD,
                                                     DS4_N_VOCAB,
                                                     g->output_norm,
                                                     false);
    DS4_GLM_MTP_STAGE("end");
    if (ok) ok = ds4_gpu_end_commands() != 0;
    else (void)ds4_gpu_synchronize();
    if (ok) {
        ok = ds4_gpu_tensor_read(g->logits,
                                 0,
                                 g->mtp_logits_host,
                                 (uint64_t)DS4_N_VOCAB * sizeof(float)) != 0;
    }
    ds4_gpu_tensor_free(enorm_view);
    ds4_gpu_tensor_free(hnorm_view);
    if (!ok) {
        fprintf(stderr, "ds4: glm mtp step failed at stage '%s' (pos %u)\n",
                mtp_stage, pos);
        return false;
    }
#undef DS4_GLM_MTP_STAGE
    int best = 0;
    float best_v = g->mtp_logits_host[0];
    for (uint32_t i = 1; i < DS4_N_VOCAB; i++) {
        if (g->mtp_logits_host[i] > best_v) {
            best_v = g->mtp_logits_host[i];
            best = (int)i;
        }
    }
    *draft_out = best;
    return true;
}

static bool glm_graph_forward_token(
        ds4_glm_gpu_graph *g,
        const ds4_model   *model,
        const ds4_weights *weights,
        int                token,
        const float       *input_hc,
        uint32_t           pos,
        float             *output_hc,
        float             *logits_out,
        bool               defer_completion);


/* Decode-style verify pass for tiny row counts (MTP): the indexed batch
 * fn measures ~1.4ms/layer at n=2 (gate-profile: gpu-wait 1.22ms/layer)
 * while decode does the same math in 0.79ms. This pass mirrors the
 * decode encoders at n rows over the batch scratch, attends causally
 * over the compact caches (valid while pos+n fits the indexer window),
 * and reuses the batch FFN encoder (routed split + big-gate combine).
 * KV/indexer caches are updated exactly like the indexed batch path. */
static bool glm_graph_verify_rows(
        ds4_glm_gpu_graph *g,
        const ds4_model   *model,
        const ds4_weights *weights,
        const int         *tokens,
        uint32_t           pos,
        uint32_t           n,
        float             *output_hc,
        float             *logits_out) {
    if (!g || !model || !weights || !tokens || n == 0 ||
        g->compact_cache_cap == 0 ||
        pos + n > g->compact_cache_cap ||
        !g->batch_cur || !g->batch_next || !g->prefill_tokens) {
        return false;
    }
    const uint32_t executable = glm_graph_normal_layer_count();
    ds4_gpu_tensor *cur = g->batch_cur;
    ds4_gpu_tensor *nxt = g->batch_next;
    if (!ds4_gpu_tensor_write(g->prefill_tokens, 0, tokens,
                              (uint64_t)n * sizeof(int32_t))) {
        return false;
    }
    if (g->placement &&
        !glm_graph_verify_ws_switch(g, g->placement[0], false, n)) {
        glm_graph_verify_ws_restore(g);
        return false;
    }
    cur = g->batch_cur;
    nxt = g->batch_next;
    bool ok = glm_graph_begin_commands_if_needed();
    if (ok) ok = ds4_gpu_embed_tokens_quant_tensor(cur,
                                                   g->prefill_tokens,
                                                   model->map,
                                                   model->size,
                                                   weights->token_embd->abs_offset,
                                                   weights->token_embd->type,
                                                   DS4_N_VOCAB,
                                                   n,
                                                   DS4_N_EMBD) != 0;
    for (uint32_t il = 0; ok && il < executable; il++) {
        if (g->placement) {
            g->batch_cur = cur;
            g->batch_next = nxt;
            ok = glm_graph_verify_ws_switch(g,
                                             g->placement[il + 1u],
                                             il != 0u,
                                             n);
            if (ok) {
                ok = glm_graph_ws_switch(g,
                                         g->placement[il + 1u],
                                         false);
            }
            cur = g->batch_cur;
            nxt = g->batch_next;
        }
        const ds4_layer_weights *l = &weights->layer[il];
        const uint32_t kv_raw_dim = (uint32_t)l->attn_kv_a_mqa->dim[1];
        const float rope_base = layer_rope_freq_base(il);
        const float rope_scale = layer_rope_freq_scale(il);
        ok = ds4_gpu_rms_norm_weight_rows_tensor(g->batch_attn_norm,
                                                 cur,
                                                 model->map,
                                                 model->size,
                                                 l->attn_norm->abs_offset,
                                                 DS4_N_EMBD,
                                                 n,
                                                 DS4_RMS_EPS) != 0;
        if (ok) ok = glm_graph_matmul_q8_0_tensor(g->batch_q_rank,
                                                  model,
                                                  l->attn_q_a->abs_offset,
                                                  DS4_N_EMBD,
                                                  DS4_N_LORA_Q,
                                                  g->batch_attn_norm,
                                                  n);
        if (ok) ok = ds4_gpu_rms_norm_weight_rows_tensor(g->batch_q_rank_norm,
                                                         g->batch_q_rank,
                                                         model->map,
                                                         model->size,
                                                         l->attn_q_a_norm->abs_offset,
                                                         DS4_N_LORA_Q,
                                                         n,
                                                         DS4_RMS_EPS) != 0;
        if (ok) ok = glm_graph_matmul_q8_0_tensor(g->batch_q,
                                                  model,
                                                  l->attn_q_b->abs_offset,
                                                  DS4_N_LORA_Q,
                                                  g->q_dim,
                                                  g->batch_q_rank_norm,
                                                  n);
        if (ok) ok = ds4_gpu_glm_rope_tail_tensor(g->batch_q,
                                                  n,
                                                  DS4_N_HEAD,
                                                  DS4_N_KEY_MLA,
                                                  DS4_N_ROT,
                                                  pos,
                                                  0,
                                                  rope_base,
                                                  rope_scale,
                                                  0.0f,
                                                  1.0f,
                                                  DS4_ROPE_YARN_BETA_FAST,
                                                  DS4_ROPE_YARN_BETA_SLOW) != 0;
        if (ok && glm_graph_layer_uses_full_indexer(il)) {
            ok = glm_graph_matmul_q8_0_tensor(g->batch_indexer_k,
                                              model,
                                              l->indexer_attn_k->abs_offset,
                                              DS4_N_EMBD,
                                              DS4_N_INDEXER_HEAD_DIM,
                                              cur,
                                              n);
            if (ok) ok = ds4_gpu_glm_store_indexer_k_tensor(
                    g->layer_indexer_key_cache[il],
                    g->batch_indexer_k,
                    model->map,
                    model->size,
                    l->indexer_k_norm->abs_offset,
                    l->indexer_k_norm_b->abs_offset,
                    pos,
                    n,
                    g->compact_cache_cap,
                    DS4_N_INDEXER_HEAD_DIM,
                    DS4_N_ROT,
                    0,
                    1.0e-6f,
                    rope_base,
                    rope_scale,
                    0.0f,
                    1.0f,
                    DS4_ROPE_YARN_BETA_FAST,
                    DS4_ROPE_YARN_BETA_SLOW,
                    glm_graph_compact_cache_is_f16()) != 0;
        }
        if (ok) ok = glm_graph_matmul_q8_0_tensor(g->batch_kv_raw,
                                                  model,
                                                  l->attn_kv_a_mqa->abs_offset,
                                                  DS4_N_EMBD,
                                                  kv_raw_dim,
                                                  g->batch_attn_norm,
                                                  n);
        if (ok) ok = ds4_gpu_glm_kv_lora_rms_norm_tensor(g->batch_kv_norm,
                                                         g->batch_kv_raw,
                                                         model->map,
                                                         model->size,
                                                         l->attn_kv_a_norm->abs_offset,
                                                         n,
                                                         kv_raw_dim,
                                                         DS4_N_KV_LORA,
                                                         DS4_RMS_EPS) != 0;
        if (ok) ok = ds4_gpu_glm_store_compact_kv_tensor(g->layer_kv_lora_cache[il],
                                                         g->layer_k_rope_cache[il],
                                                         g->batch_kv_norm,
                                                         g->batch_kv_raw,
                                                         pos,
                                                         n,
                                                         g->compact_cache_cap,
                                                         kv_raw_dim,
                                                         DS4_N_KV_LORA,
                                                         DS4_N_ROT,
                                                         glm_graph_compact_cache_is_f16()) != 0;
        if (ok) ok = ds4_gpu_glm_qk_lowrank_typed_batch_tensor(g->batch_qk_low,
                                                               g->batch_q,
                                                               model->map,
                                                               model->size,
                                                               l->attn_k_b->abs_offset,
                                                               l->attn_k_b->type,
                                                               n,
                                                               DS4_N_HEAD,
                                                               DS4_N_KV_LORA,
                                                               (uint32_t)g->q_nope,
                                                               DS4_N_KEY_MLA) != 0;
        if (ok) ok = ds4_gpu_glm_attention_indexed_batch_lora_causal_tensor(
                g->batch_attn_lora,
                g->batch_q,
                g->batch_qk_low,
                g->layer_kv_lora_cache[il],
                g->layer_k_rope_cache[il],
                n,
                pos,
                pos + n,
                g->compact_cache_cap,
                glm_graph_compact_cache_is_f16(),
                DS4_N_HEAD,
                DS4_N_KV_LORA,
                (uint32_t)g->q_nope,
                DS4_N_ROT,
                0,
                rope_base,
                rope_scale,
                0.0f,
                1.0f,
                DS4_ROPE_YARN_BETA_FAST,
                DS4_ROPE_YARN_BETA_SLOW) != 0;
        if (ok) ok = ds4_gpu_glm_value_project_typed_batch_heads_tensor(
                g->batch_heads,
                g->batch_attn_lora,
                model->map,
                model->size,
                l->attn_v_b->abs_offset,
                l->attn_v_b->type,
                n,
                DS4_N_HEAD,
                DS4_N_KV_LORA,
                DS4_N_VALUE_MLA) != 0;
        if (ok) ok = glm_graph_matmul_q8_0_tensor(g->batch_attn_out,
                                                  model,
                                                  l->attn_output->abs_offset,
                                                  g->heads_dim,
                                                  DS4_N_EMBD,
                                                  g->batch_heads,
                                                  n);
        if (ok) ok = ds4_gpu_add_tensor(g->batch_after_attn,
                                        cur,
                                        g->batch_attn_out,
                                        (uint32_t)((uint64_t)n * DS4_N_EMBD)) != 0;
        if (ok) ok = glm_graph_encode_ffn_batch(g,
                                                model,
                                                weights,
                                                l,
                                                il,
                                                pos,
                                                g->batch_after_attn,
                                                nxt,
                                                n,
                                                false,
                                                false,
                                                false,
                                                NULL);
        if (ok) {
            ds4_gpu_tensor *tmp = cur;
            cur = nxt;
            nxt = tmp;
        }
    }
    if (ok) ok = ds4_gpu_end_commands() != 0;
    else (void)ds4_gpu_synchronize();
    if (ok && output_hc) {
        ok = ds4_gpu_tensor_read(cur,
                                 0,
                                 output_hc,
                                 (uint64_t)n * DS4_N_EMBD * sizeof(float)) != 0;
    }
    if (ok && logits_out) {
        ds4_gpu_tensor *last = glm_graph_tensor_row_view_strided(cur,
                                                                 n - 1u,
                                                                 DS4_N_EMBD,
                                                                 DS4_N_EMBD);
        ok = last != NULL &&
             glm_graph_forward_output_head(g, model, weights, last, logits_out);
        ds4_gpu_tensor_free(last);
    }
    glm_graph_verify_ws_restore(g);
    return ok;
}

static bool glm_graph_forward_tokens(
        ds4_glm_gpu_graph *g,
        const ds4_model   *model,
        const ds4_weights *weights,
        const int         *tokens,
        const float       *input_hc,
        uint32_t           pos0,
        uint32_t           n_tokens,
        float             *output_hc,
        float             *logits_out,
        ds4_session_progress_fn display_progress,
        void              *display_progress_ud,
        uint32_t           display_absolute_base,
        uint32_t           work_done_base,
        uint32_t           work_total) {
    if (!g || !model || !weights || !tokens ||
        n_tokens == 0 ||
        g->layer_count == 0 ||
        !glm_graph_span_fits_context(g, pos0, n_tokens)) {
        return false;
    }
    if (!glm_graph_span_fits_full_attention(g, pos0, n_tokens)) {
        glm_graph_log_full_attention_limit(g, pos0, n_tokens);
        return false;
    }
    if (!g->full_kv_cache) {
        fprintf(stderr,
                "ds4: GLM full-attention prefill was requested without an expanded KV cache\n");
        return false;
    }
    for (uint32_t i = 0; i < n_tokens; i++) {
        if (tokens[i] < 0 || tokens[i] >= (int)DS4_N_VOCAB) return false;
    }
    if (!input_hc && !g->has_token_embd) return false;
    if (logits_out && !g->has_output_head) return false;
    glm_graph_reset_prefill_seed_capture(g);
    const uint32_t n_rows = pos0 + n_tokens;
    const bool trace = glm_graph_full_prefill_trace_enabled();
    const bool trace_all = trace && glm_graph_full_prefill_trace_all();
    const double trace_slow_ms = trace ?
        (double)glm_graph_full_prefill_trace_slow_ms() : 0.0;
    const double trace_chunk_t0 = trace ? now_sec() : 0.0;
    if (trace) {
        glm_graph_full_prefill_tracef(
                "chunk begin pos=%u tokens=%u rows=%u compact_cap=%u work_base=%u work_total=%u",
                pos0,
                n_tokens,
                n_rows,
                g->compact_cache_cap,
                work_done_base,
                work_total);
    }
    if (g->compact_cache_cap != 0) {
        const double trace_cache_t0 = trace ? now_sec() : 0.0;
        if (!glm_graph_ensure_compact_cache(g, n_rows)) {
            if (trace) {
                glm_graph_full_prefill_tracef(
                        "ensure_cache failed pos=%u tokens=%u rows=%u compact_cap=%u",
                        pos0,
                        n_tokens,
                        n_rows,
                        g->compact_cache_cap);
            }
            return false;
        }
        if (trace) {
            const double ms = (now_sec() - trace_cache_t0) * 1000.0;
            if (trace_all || ms >= trace_slow_ms) {
                glm_graph_full_prefill_tracef(
                        "ensure_cache done pos=%u tokens=%u rows=%u compact_cap=%u %.3f ms",
                        pos0,
                        n_tokens,
                        n_rows,
                        g->compact_cache_cap,
                        ms);
            }
        }
    }

    const double trace_upload_t0 = trace ? now_sec() : 0.0;
    bool ok = glm_graph_upload_tokens(g->prefill_tokens, tokens, n_tokens);
    if (trace) {
        const double ms = (now_sec() - trace_upload_t0) * 1000.0;
        if (trace_all || ms >= trace_slow_ms || !ok) {
            glm_graph_full_prefill_tracef(
                    "upload_tokens %s pos=%u tokens=%u %.3f ms",
                    ok ? "done" : "failed",
                    pos0,
                    n_tokens,
                    ms);
        }
    }
    ds4_gpu_tensor *cur = g->batch_cur;
    ds4_gpu_tensor *next = g->batch_next;
    ds4_gpu_tensor *last_hidden = NULL;

    glm_graph_report_prefill_display_progress(display_progress,
                                              display_progress_ud,
                                              display_absolute_base,
                                              work_done_base,
                                              n_tokens,
                                              0,
                                              g->layer_count,
                                              work_total,
                                              true);

    const bool stage_sync =
        glm_graph_small_prefill_stage_sync(n_tokens, logits_out != NULL);
    const uint32_t layer_flush_interval = stage_sync ? 0u :
        glm_graph_full_prefill_layer_flush_interval(n_tokens,
                                                    n_rows,
                                                    logits_out != NULL);
    const uint32_t progress_flush_interval =
        glm_graph_prefill_progress_flush_interval(layer_flush_interval,
                                                  n_tokens,
                                                  display_progress,
                                                  work_total);
    const bool progress_requested = display_progress && work_total > 0;
    const uint32_t drain_interval =
        (progress_requested && progress_flush_interval != 0) ?
        glm_graph_full_prefill_drain_interval() : 0u;
    metal_graph_stream_prepare_slot layer_prepare_slots[DS4_STREAM_PREFILL_MAX_PREPARE_AHEAD];
    memset(layer_prepare_slots, 0, sizeof(layer_prepare_slots));
    const bool full_layer_prefill =
        glm_graph_stream_prefill_full_layer_enabled(g, n_tokens);
    ds4_gpu_set_glm_streaming_prefill_full_layer(full_layer_prefill);
    const bool streaming_prefill_sync_each_layer =
        !g->ssd_streaming ||
        glm_graph_streaming_prefill_sync_each_layer(full_layer_prefill);
#ifdef DS4_ROCM_BUILD
    rocm_graph_stream_layer_expert_load rocm_full_layer_load;
    memset(&rocm_full_layer_load, 0, sizeof(rocm_full_layer_load));
#endif
    const bool full_layer_prepare_base =
        glm_graph_stream_prefill_full_layer_prepare_enabled(g,
                                                            full_layer_prefill);
    const bool layer_pagein =
        full_layer_prepare_base &&
        glm_graph_env_present("DS4_ROCM_ENABLE_STREAMING_PREFILL_LAYER_PAGEIN",
                              "DS4_METAL_ENABLE_STREAMING_PREFILL_LAYER_PAGEIN") &&
        !glm_graph_env_present("DS4_ROCM_DISABLE_STREAMING_PREFILL_LAYER_PAGEIN",
                               "DS4_METAL_DISABLE_STREAMING_PREFILL_LAYER_PAGEIN");
    const bool layer_readahead =
        full_layer_prepare_base &&
        !layer_pagein &&
        glm_graph_env_present("DS4_ROCM_ENABLE_STREAMING_PREFILL_LAYER_READAHEAD",
                              "DS4_METAL_ENABLE_STREAMING_PREFILL_LAYER_READAHEAD") &&
        !glm_graph_env_present("DS4_ROCM_DISABLE_STREAMING_PREFILL_LAYER_READAHEAD",
                               "DS4_METAL_DISABLE_STREAMING_PREFILL_LAYER_READAHEAD") &&
        !glm_graph_env_present("DS4_ROCM_DISABLE_STREAMING_PREFILL_LAYER_PREPARE",
                               "DS4_METAL_DISABLE_STREAMING_PREFILL_LAYER_PREPARE");
    const bool layer_pread =
        full_layer_prepare_base &&
        !layer_pagein &&
        !layer_readahead &&
        !glm_graph_env_present("DS4_ROCM_DISABLE_STREAMING_PREFILL_LAYER_PREAD",
                               "DS4_METAL_DISABLE_STREAMING_PREFILL_LAYER_PREAD") &&
        !glm_graph_env_present("DS4_ROCM_DISABLE_STREAMING_PREFILL_LAYER_PREPARE",
                               "DS4_METAL_DISABLE_STREAMING_PREFILL_LAYER_PREPARE");
    const bool layer_madvise =
        full_layer_prepare_base &&
        !layer_pagein &&
        !layer_pread &&
        !layer_readahead &&
        !glm_graph_env_present("DS4_ROCM_DISABLE_STREAMING_PREFILL_LAYER_PREPARE",
                               "DS4_METAL_DISABLE_STREAMING_PREFILL_LAYER_PREPARE") &&
        !glm_graph_env_present("DS4_ROCM_DISABLE_STREAMING_PREFILL_LAYER_MADVISE",
                               "DS4_METAL_DISABLE_STREAMING_PREFILL_LAYER_MADVISE");
    const bool layer_prepare =
        layer_pagein || layer_pread || layer_readahead || layer_madvise;
    const bool layer_prepare_overlap =
        layer_prepare &&
        metal_graph_stream_prefill_layer_pagein_overlap_enabled();
    const bool full_layer_flush_intermediate = full_layer_prefill;
    const uint32_t layer_prepare_ahead =
        layer_prepare && layer_prepare_overlap ?
        metal_graph_stream_prefill_layer_prepare_ahead() : 1u;
    if (trace) {
        glm_graph_full_prefill_tracef(
                "mode pos=%u tokens=%u stage_sync=%u layer_flush_interval=%u progress_flush_interval=%u drain_interval=%u",
                pos0,
                n_tokens,
                stage_sync ? 1u : 0u,
                layer_flush_interval,
                progress_flush_interval,
                drain_interval);
    }
    if (ok && layer_prepare && g->layer_count > 0 &&
        !metal_graph_stream_prepare_start_if_needed(NULL,
                                                    model,
                                                    weights,
                                                    g->layer_start,
                                                    n_tokens,
                                                    layer_madvise,
                                                    layer_pread,
                                                    layer_readahead,
                                                    full_layer_prefill &&
                                                    rocm_graph_glm_stream_prefill_full_layer_enabled(
                                                            g,
                                                            &weights->layer[g->layer_start],
                                                            g->layer_start,
                                                            n_tokens),
                                                    layer_prepare_slots,
                                                    layer_prepare_ahead)) {
        ok = false;
    }
#ifdef DS4_ROCM_BUILD
    if (ok &&
        full_layer_prefill &&
        !rocm_graph_glm_stream_layer_expert_load_start_next(
                &rocm_full_layer_load,
                g,
                model,
                weights,
                g->layer_start,
                g->layer_end,
                n_tokens)) {
        ok = false;
    }
#endif
    if (ok) {
        const double t0 = trace ? now_sec() : 0.0;
        if (input_hc) {
            ok = ds4_gpu_tensor_write(cur,
                                      0,
                                      input_hc,
                                      (uint64_t)n_tokens * DS4_N_EMBD * sizeof(float)) != 0;
        } else {
            ok = glm_graph_stream_map_token(g, model, weights);
        }
        if (ok) ok = ds4_gpu_begin_commands() != 0;
        if (trace) {
            const double ms = (now_sec() - t0) * 1000.0;
            if (trace_all || ms >= trace_slow_ms || !ok) {
                glm_graph_full_prefill_tracef(
                        "begin_commands%s %s pos=%u tokens=%u %.3f ms",
                        input_hc ? "_from_hidden" : "",
                        ok ? "done" : "failed",
                        pos0,
                        n_tokens,
                        ms);
            }
        }
    }
    if (ok && !input_hc) {
        const double t0 = trace ? now_sec() : 0.0;
        ok = ds4_gpu_embed_tokens_quant_tensor(cur,
                                               g->prefill_tokens,
                                               model->map,
                                               model->size,
                                               weights->token_embd->abs_offset,
                                               weights->token_embd->type,
                                               DS4_N_VOCAB,
                                               n_tokens,
                                               DS4_N_EMBD) != 0;
        if (trace) {
            const double ms = (now_sec() - t0) * 1000.0;
            if (trace_all || ms >= trace_slow_ms || !ok) {
                glm_graph_full_prefill_tracef(
                        "embed %s pos=%u tokens=%u %.3f ms",
                        ok ? "done" : "failed",
                        pos0,
                        n_tokens,
                        ms);
            }
        }
    }
    if (ok && g->ssd_streaming && streaming_prefill_sync_each_layer) {
        ok = ds4_gpu_end_commands() != 0;
    }
#define DS4_GLM_PROFILE_PREFILL_STAGE(part_, name_) do { \
        if (ok && layer_stage_profile) { \
            ok = metal_graph_layer_stage_profile_boundary((part_), (name_), il, pos0, n_tokens, &layer_stage_t0); \
        } else if (ok && stage_sync) { \
            ok = glm_graph_prefill_stage_sync_boundary(); \
        } \
    } while (0)
    for (uint32_t il = g->layer_start; ok && il <= g->layer_end; il++) {
        const uint32_t slice_layer_done = il - g->layer_start + 1u;
        if (layer_prepare &&
            !metal_graph_stream_prepare_join_layer(NULL,
                                                   model,
                                                   weights,
                                                   il,
                                                   n_tokens,
                                                   layer_madvise,
                                                   layer_pread,
                                                   layer_readahead,
                                                   full_layer_prefill &&
                                                   rocm_graph_glm_stream_prefill_full_layer_enabled(
                                                           g,
                                                           &weights->layer[il],
                                                           il,
                                                           n_tokens),
                                                   layer_prepare_slots,
                                                   layer_prepare_ahead)) {
            ok = false;
            break;
        }
#ifdef DS4_ROCM_BUILD
        if (full_layer_prefill &&
            !rocm_graph_glm_stream_layer_expert_load_ready(
                    &rocm_full_layer_load,
                    g,
                    model,
                    weights,
                    il,
                    n_tokens)) {
            ok = false;
            break;
        }
        if (full_layer_prefill &&
            !rocm_graph_glm_stream_layer_expert_load_start_next(
                    &rocm_full_layer_load,
                    g,
                    model,
                    weights,
                    il + 1u,
                    g->layer_end,
                    n_tokens)) {
            ok = false;
            break;
        }
#endif
        if (g->ssd_streaming) {
            ok = glm_graph_stream_map_prefill_layer(g,
                                                    model,
                                                    weights,
                                                    il,
                                                    n_tokens,
                                                    full_layer_prefill);
            if (ok && layer_prepare && layer_prepare_overlap) {
                bool started_future = false;
                for (uint32_t ahead = 1; ahead <= layer_prepare_ahead; ahead++) {
                    if (il + ahead > g->layer_end) break;
                    started_future = true;
                    if (!metal_graph_stream_prepare_start_if_needed(NULL,
                                                                    model,
                                                                    weights,
                                                                    il + ahead,
                                                                    n_tokens,
                                                                    layer_madvise,
                                                                    layer_pread,
                                                                    layer_readahead,
                                                                    full_layer_prefill &&
                                                                    rocm_graph_glm_stream_prefill_full_layer_enabled(
                                                                            g,
                                                                            &weights->layer[il + ahead],
                                                                            il + ahead,
                                                                            n_tokens),
                                                                    layer_prepare_slots,
                                                                    layer_prepare_ahead)) {
                        ok = false;
                        break;
                    }
                }
                if (ok && !started_future && logits_out) {
                    metal_graph_stream_readahead_output(model, weights);
                }
            }
            if (ok && !ds4_gpu_commands_active()) {
                ok = ds4_gpu_begin_commands() != 0;
            }
        }
        const ds4_layer_weights *l = &weights->layer[il];
        const uint32_t kv_raw_dim = (uint32_t)l->attn_kv_a_mqa->dim[1];
        const float rope_base = layer_rope_freq_base(il);
        const float rope_scale = layer_rope_freq_scale(il);
        const uint32_t cache_len = pos0 + n_tokens;
        const uint64_t residual_elems = (uint64_t)n_tokens * DS4_N_EMBD;
        const bool layer_stage_profile = metal_graph_layer_stage_profile_enabled(il);
        double layer_stage_t0 = layer_stage_profile ? now_sec() : 0.0;
        const double trace_layer_t0 = trace ? now_sec() : 0.0;
        bool trace_layer_flushed = false;
        if (trace && trace_all) {
            glm_graph_full_prefill_tracef(
                    "layer begin layer=%u pos=%u tokens=%u rows=%u",
                    il,
                    pos0,
                    n_tokens,
                    n_rows);
        }
        if (residual_elems > UINT32_MAX) {
            ok = false;
            break;
        }
        if (layer_stage_profile) {
            ok = metal_graph_layer_stage_profile_boundary("glm_attn",
                                                          NULL,
                                                          il,
                                                          pos0,
                                                          n_tokens,
                                                          &layer_stage_t0);
        }

        if (ok) ok = ds4_gpu_rms_norm_weight_rows_tensor(g->batch_attn_norm,
                                                         cur,
                                                         model->map,
                                                         model->size,
                                                         l->attn_norm->abs_offset,
                                                         DS4_N_EMBD,
                                                         n_tokens,
                                                         DS4_RMS_EPS) != 0;
        DS4_GLM_PROFILE_PREFILL_STAGE("glm_attn", "attn_norm");
        if (ok) ok = glm_graph_matmul_q8_0_tensor(g->batch_q_rank,
                                                  model,
                                                  l->attn_q_a->abs_offset,
                                                  DS4_N_EMBD,
                                                  DS4_N_LORA_Q,
                                                  g->batch_attn_norm,
                                                  n_tokens);
        if (ok) ok = ds4_gpu_rms_norm_weight_rows_tensor(g->batch_q_rank_norm,
                                                         g->batch_q_rank,
                                                         model->map,
                                                         model->size,
                                                         l->attn_q_a_norm->abs_offset,
                                                         DS4_N_LORA_Q,
                                                         n_tokens,
                                                         DS4_RMS_EPS) != 0;
        if (ok) ok = glm_graph_matmul_q8_0_tensor(g->batch_q,
                                                  model,
                                                  l->attn_q_b->abs_offset,
                                                  DS4_N_LORA_Q,
                                                  g->q_dim,
                                                  g->batch_q_rank_norm,
                                                  n_tokens);
        if (ok) ok = ds4_gpu_rope_tail_tensor(g->batch_q,
                                              n_tokens,
                                              DS4_N_HEAD,
                                              DS4_N_KEY_MLA,
                                              DS4_N_ROT,
                                              pos0,
                                              0,
                                              false,
                                              rope_base,
                                              rope_scale,
                                              0.0f,
                                              1.0f,
                                              DS4_ROPE_YARN_BETA_FAST,
                                              DS4_ROPE_YARN_BETA_SLOW) != 0;
        DS4_GLM_PROFILE_PREFILL_STAGE("glm_attn", "q_path");
        if (ok && g->compact_cache_cap != 0 && glm_graph_layer_uses_full_indexer(il)) {
            ok = glm_graph_matmul_q8_0_tensor(g->batch_indexer_k,
                                              model,
                                              l->indexer_attn_k->abs_offset,
                                              DS4_N_EMBD,
                                              DS4_N_INDEXER_HEAD_DIM,
                                              cur,
                                              n_tokens);
            if (ok) {
                ok = ds4_gpu_glm_store_indexer_k_tensor(
                        g->layer_indexer_key_cache[il],
                        g->batch_indexer_k,
                        model->map,
                        model->size,
                        l->indexer_k_norm->abs_offset,
                        l->indexer_k_norm_b->abs_offset,
                        pos0,
                        n_tokens,
                        g->compact_cache_cap,
                        DS4_N_INDEXER_HEAD_DIM,
                        DS4_N_ROT,
                        0,
                        1.0e-6f,
                        rope_base,
                        rope_scale,
                        0.0f,
                        1.0f,
                        DS4_ROPE_YARN_BETA_FAST,
                        DS4_ROPE_YARN_BETA_SLOW,
                        glm_graph_compact_cache_is_f16()) != 0;
            }
        }
        DS4_GLM_PROFILE_PREFILL_STAGE("glm_attn", "indexer_k");
        if (ok) ok = glm_graph_matmul_q8_0_tensor(g->batch_kv_raw,
                                                  model,
                                                  l->attn_kv_a_mqa->abs_offset,
                                                  DS4_N_EMBD,
                                                  kv_raw_dim,
                                                  g->batch_attn_norm,
                                                  n_tokens);
        if (ok) ok = ds4_gpu_glm_kv_lora_rms_norm_tensor(g->batch_kv_norm,
                                                         g->batch_kv_raw,
                                                         model->map,
                                                         model->size,
                                                         l->attn_kv_a_norm->abs_offset,
                                                         n_tokens,
                                                         kv_raw_dim,
                                                         DS4_N_KV_LORA,
                                                         DS4_RMS_EPS) != 0;
        if (ok && g->compact_cache_cap != 0) {
            ok = ds4_gpu_glm_store_compact_kv_tensor(g->layer_kv_lora_cache[il],
                                                     g->layer_k_rope_cache[il],
                                                     g->batch_kv_norm,
                                                     g->batch_kv_raw,
                                                     pos0,
                                                     n_tokens,
                                                     g->compact_cache_cap,
                                                     kv_raw_dim,
                                                     DS4_N_KV_LORA,
                                                     DS4_N_ROT,
                                                     glm_graph_compact_cache_is_f16()) != 0;
        }
        if (ok) ok = ds4_gpu_glm_k_b_project_typed_tensor(g->batch_k_nope,
                                                          g->batch_kv_norm,
                                                          model->map,
                                                          model->size,
                                                          l->attn_k_b->abs_offset,
                                                          l->attn_k_b->type,
                                                          n_tokens,
                                                          DS4_N_KV_LORA,
                                                          (uint32_t)g->q_nope,
                                                          DS4_N_HEAD) != 0;
        if (ok) ok = glm_graph_matmul_q8_0_tensor(g->batch_value,
                                                  model,
                                                  l->attn_v_b->abs_offset,
                                                  DS4_N_KV_LORA,
                                                  g->heads_dim,
                                                  g->batch_kv_norm,
                                                  n_tokens);
        DS4_GLM_PROFILE_PREFILL_STAGE("glm_attn", "kv_path");
        const bool flash_requested = glm_graph_use_flash_attention_prefill(n_tokens);
        const bool use_staged_flash_kv =
            flash_requested && glm_graph_use_flash_attention_staged_kv(pos0, n_tokens, cache_len);
        const bool use_flash_attn = flash_requested;
        if (ok) {
            if (use_staged_flash_kv) {
                ok = ds4_gpu_glm_build_kv_cache_flash_tensor(g->layer_key_cache[il],
                                                             g->layer_value_cache[il],
                                                             g->batch_kv_raw,
                                                             g->batch_k_nope,
                                                             g->batch_value,
                                                             pos0,
                                                             n_tokens,
                                                             g->ctx_cap,
                                                             DS4_N_HEAD,
                                                             kv_raw_dim,
                                                             DS4_N_KV_LORA,
                                                             (uint32_t)g->q_nope,
                                                             DS4_N_ROT,
                                                             DS4_N_VALUE_MLA,
                                                             0,
                                                             rope_base,
                                                             rope_scale,
                                                             0.0f,
                                                             1.0f,
                                                             DS4_ROPE_YARN_BETA_FAST,
                                                             DS4_ROPE_YARN_BETA_SLOW,
                                                             true) != 0;
            } else {
                ok = ds4_gpu_glm_build_kv_cache_tensor(g->layer_key_cache[il],
                                                       g->layer_value_cache[il],
                                                       g->batch_kv_raw,
                                                       g->batch_k_nope,
                                                       g->batch_value,
                                                       pos0,
                                                       n_tokens,
                                                       g->ctx_cap,
                                                       DS4_N_HEAD,
                                                       kv_raw_dim,
                                                       DS4_N_KV_LORA,
                                                       (uint32_t)g->q_nope,
                                                       DS4_N_ROT,
                                                       DS4_N_VALUE_MLA,
                                                       0,
                                                       rope_base,
                                                       rope_scale,
                                                       0.0f,
                                                       1.0f,
                                                       DS4_ROPE_YARN_BETA_FAST,
                                                       DS4_ROPE_YARN_BETA_SLOW,
                                                       true) != 0;
            }
        }
        DS4_GLM_PROFILE_PREFILL_STAGE("glm_attn", "kv_cache");
        if (ok) {
            if (use_flash_attn) {
                if (use_staged_flash_kv) {
                    ok = ds4_gpu_glm_attention_flash_staged_tensor(g->batch_heads,
                                                                   g->batch_q,
                                                                   g->layer_key_cache[il],
                                                                   g->layer_value_cache[il],
                                                                   pos0,
                                                                   n_tokens,
                                                                   cache_len,
                                                                   g->ctx_cap,
                                                                   DS4_N_HEAD,
                                                                   DS4_N_KEY_MLA,
                                                                   DS4_N_VALUE_MLA,
                                                                   true) != 0;
                } else {
                    ok = ds4_gpu_glm_attention_flash_tensor(g->batch_heads,
                                                            g->batch_q,
                                                            g->layer_key_cache[il],
                                                            g->layer_value_cache[il],
                                                            pos0,
                                                            n_tokens,
                                                            cache_len,
                                                            g->ctx_cap,
                                                            DS4_N_HEAD,
                                                            DS4_N_KEY_MLA,
                                                            DS4_N_VALUE_MLA,
                                                            true) != 0;
                }
            } else {
                ok = ds4_gpu_glm_attention_full_tensor(g->batch_heads,
                                                       g->batch_q,
                                                       g->layer_key_cache[il],
                                                       g->layer_value_cache[il],
                                                       pos0,
                                                       n_tokens,
                                                       cache_len,
                                                       g->ctx_cap,
                                                       DS4_N_HEAD,
                                                       DS4_N_KEY_MLA,
                                                       DS4_N_VALUE_MLA,
                                                       true) != 0;
            }
        }
        DS4_GLM_PROFILE_PREFILL_STAGE("glm_attn", "attention");
        if (ok) ok = glm_graph_matmul_q8_0_tensor(g->batch_attn_out,
                                                  model,
                                                  l->attn_output->abs_offset,
                                                  g->heads_dim,
                                                  DS4_N_EMBD,
                                                  g->batch_heads,
                                                  n_tokens);
        if (ok) ok = ds4_gpu_add_tensor(g->batch_after_attn,
                                        cur,
                                        g->batch_attn_out,
                                        (uint32_t)residual_elems) != 0;
        DS4_GLM_PROFILE_PREFILL_STAGE("glm_attn", "attn_output");
        if (ok) ok = glm_graph_encode_ffn_batch(g,
                                                model,
                                                weights,
                                                l,
                                                il,
                                                pos0,
                                                g->batch_after_attn,
                                                next,
                                                n_tokens,
                                                full_layer_prefill,
                                                layer_stage_profile,
                                                stage_sync,
                                                layer_stage_profile ? &layer_stage_t0 : NULL);
        if (ok) {
            ds4_gpu_tensor *tmp = cur;
            cur = next;
            next = tmp;
        }
        if (ok && glm_debug_hidden_dump_layer_match(il)) {
            ok = ds4_gpu_end_commands() != 0;
            if (ok) {
                for (uint32_t r = 0; r < n_tokens; r++)
                    glm_debug_dump_hidden_layer(cur, r, il, pos0 + r);
                glm_debug_dump_raw_layer(g->batch_router_selected, "sel",
                                         (uint64_t)n_tokens * DS4_N_EXPERT_USED * sizeof(int32_t),
                                         il, -1);
                glm_debug_dump_raw_layer(g->batch_router_weights, "selw",
                                         (uint64_t)n_tokens * DS4_N_EXPERT_USED * sizeof(float),
                                         il, -1);
                ok = ds4_gpu_begin_commands() != 0;
            }
        }
        if (ok &&
            !g->ssd_streaming &&
            progress_flush_interval != 0 &&
            (il < g->layer_end || progress_requested) &&
            (slice_layer_done % progress_flush_interval) == 0) {
            const uint32_t work_done =
                work_done_base + (uint32_t)(((uint64_t)n_tokens * slice_layer_done) / g->layer_count);
            const bool drain_now =
                drain_interval != 0 &&
                il < g->layer_end &&
                (slice_layer_done % drain_interval) == 0;
            const char *command_action = drain_now ? "drain" : "flush";
            const double trace_command_t0 = trace ? now_sec() : 0.0;
            if (trace && (trace_all || drain_now)) {
                glm_graph_full_prefill_tracef(
                        "layer %s begin layer=%u pos=%u tokens=%u work=%u/%u",
                        command_action,
                        il,
                        pos0,
                        n_tokens,
                        work_done,
                        work_total);
            }
            if (drain_now) {
                ok = ds4_gpu_end_commands() != 0;
                if (ok) ok = ds4_gpu_begin_commands() != 0;
            } else {
                ok = ds4_gpu_flush_commands() != 0;
            }
            if (trace) {
                const double trace_command_done = now_sec();
                const double command_ms = (trace_command_done - trace_command_t0) * 1000.0;
                const double layer_ms = (trace_command_done - trace_layer_t0) * 1000.0;
                trace_layer_flushed = true;
                if (trace_all || drain_now ||
                    command_ms >= trace_slow_ms || layer_ms >= trace_slow_ms || !ok) {
                    glm_graph_full_prefill_tracef(
                            "layer %s %s layer=%u pos=%u tokens=%u command=%.3f ms layer_total=%.3f ms work=%u/%u",
                            command_action,
                            ok ? "done" : "failed",
                            il,
                            pos0,
                            n_tokens,
                            command_ms,
                            layer_ms,
                            work_done,
                            work_total);
                }
            }
            if (ok) {
                const bool progress_completed =
                    drain_interval == 0 || drain_now;
                if (progress_completed) {
                    glm_graph_report_prefill_display_progress(display_progress,
                                                              display_progress_ud,
                                                              display_absolute_base,
                                                              work_done_base,
                                                              n_tokens,
                                                              slice_layer_done,
                                                              g->layer_count,
                                                              work_total,
                                                              logits_out == NULL && output_hc == NULL);
                }
            }
        }
        if (g->ssd_streaming) {
            if (streaming_prefill_sync_each_layer) {
                if (ok && full_layer_flush_intermediate &&
                    il < g->layer_end) {
                    ok = ds4_gpu_flush_commands() != 0;
                } else if (ok) {
                    ok = ds4_gpu_end_commands() != 0;
                }
                else (void)ds4_gpu_synchronize();
            }
            else if (!ok) (void)ds4_gpu_synchronize();
            if (ok) {
                glm_graph_report_prefill_display_progress(display_progress,
                                                          display_progress_ud,
                                                          display_absolute_base,
                                                          work_done_base,
                                                          n_tokens,
                                                          slice_layer_done,
                                                          g->layer_count,
                                                          work_total,
                                                          logits_out == NULL && output_hc == NULL);
            }
        }
        if (ok &&
            g->ssd_streaming &&
            layer_prepare &&
            !layer_prepare_overlap) {
            if (il < g->layer_end) {
                if (!metal_graph_stream_prepare_start_if_needed(NULL,
                                                                model,
                                                                weights,
                                                                il + 1u,
                                                                n_tokens,
                                                                layer_madvise,
                                                                layer_pread,
                                                                layer_readahead,
                                                                full_layer_prefill &&
                                                                rocm_graph_glm_stream_prefill_full_layer_enabled(
                                                                        g,
                                                                        &weights->layer[il + 1u],
                                                                        il + 1u,
                                                                        n_tokens),
                                                                layer_prepare_slots,
                                                                layer_prepare_ahead)) {
                    ok = false;
                }
            } else if (logits_out) {
                metal_graph_stream_readahead_output(model, weights);
            }
        }
        if (trace && ok) {
            const double layer_ms = (now_sec() - trace_layer_t0) * 1000.0;
            if (trace_all || (!trace_layer_flushed && layer_ms >= trace_slow_ms)) {
                glm_graph_full_prefill_tracef(
                        "layer end layer=%u pos=%u tokens=%u flushed=%u layer_total=%.3f ms",
                        il,
                        pos0,
                        n_tokens,
                        trace_layer_flushed ? 1u : 0u,
                        layer_ms);
            }
        }
    }
#undef DS4_GLM_PROFILE_PREFILL_STAGE
    if (ok && !g->ssd_streaming) {
        const double trace_end_t0 = trace ? now_sec() : 0.0;
        if (trace) {
            glm_graph_full_prefill_tracef(
                    "chunk end_commands begin pos=%u tokens=%u",
                    pos0,
                    n_tokens);
        }
        ok = ds4_gpu_end_commands() != 0;
        if (trace) {
            const double end_ms = (now_sec() - trace_end_t0) * 1000.0;
            const double chunk_ms = (now_sec() - trace_chunk_t0) * 1000.0;
            glm_graph_full_prefill_tracef(
                    "chunk end_commands %s pos=%u tokens=%u end=%.3f ms chunk_total=%.3f ms",
                    ok ? "done" : "failed",
                    pos0,
                    n_tokens,
                    end_ms,
                    chunk_ms);
        }
    } else if (!ok) {
        if (trace) {
            glm_graph_full_prefill_tracef(
                    "chunk failed before end pos=%u tokens=%u elapsed=%.3f ms",
                    pos0,
                    n_tokens,
                    (now_sec() - trace_chunk_t0) * 1000.0);
        }
#ifdef DS4_ROCM_BUILD
        (void)rocm_graph_stream_layer_expert_load_join(&rocm_full_layer_load);
        if (full_layer_prefill) {
            (void)ds4_gpu_stream_expert_cache_release_layer_cache();
        }
#endif
        if (layer_prepare) {
            (void)metal_graph_stream_prepare_join_all(layer_prepare_slots,
                                                      layer_prepare_ahead);
        }
        (void)ds4_gpu_synchronize();
    }
    if (ok && layer_prepare &&
        !metal_graph_stream_prepare_join_all(layer_prepare_slots,
                                             layer_prepare_ahead)) {
        ok = false;
    }
#ifdef DS4_ROCM_BUILD
    if (!rocm_graph_stream_layer_expert_load_join(&rocm_full_layer_load)) {
        ok = false;
    }
    if (full_layer_prefill) {
        (void)ds4_gpu_stream_expert_cache_release_layer_cache();
    }
#endif
    if (ok &&
        g->ssd_streaming &&
        !streaming_prefill_sync_each_layer &&
        !output_hc &&
        !logits_out) {
        ok = ds4_gpu_end_commands() != 0;
    }
    if (ok) {
        glm_graph_report_prefill_display_progress(display_progress,
                                                  display_progress_ud,
                                                  display_absolute_base,
                                                  work_done_base,
                                                  n_tokens,
                                                  g->layer_count,
                                                  g->layer_count,
                                                  work_total,
                                                  logits_out == NULL && output_hc == NULL);
    }
    if (ok && output_hc) {
        ok = ds4_gpu_tensor_read(cur,
                                 0,
                                 output_hc,
                                 (uint64_t)n_tokens * DS4_N_EMBD * sizeof(float)) != 0;
    }
    if (ok && logits_out) {
        ok = glm_graph_seed_streaming_expert_cache_from_prefill(g,
                                                                model,
                                                                weights);
    }
    if (ok && logits_out) {
        last_hidden = glm_graph_tensor_row_view_strided(cur,
                                                        n_tokens - 1u,
                                                        DS4_N_EMBD,
                                                        DS4_N_EMBD);
        ok = last_hidden != NULL;
        if (ok && g->ssd_streaming) ok = glm_graph_stream_map_output(g, model, weights);
        if (ok) ok = glm_graph_forward_output_head(g, model, weights, last_hidden, logits_out);
        if (ok) {
            glm_graph_report_prefill_display_progress(display_progress,
                                                      display_progress_ud,
                                                      display_absolute_base,
                                                      work_done_base,
                                                      n_tokens,
                                                      g->layer_count,
                                                      g->layer_count,
                                                      work_total,
                                                      true);
        }
    }
    ds4_gpu_tensor_free(last_hidden);
    ds4_gpu_set_glm_streaming_prefill_full_layer(false);
    return ok;
}

static uint32_t glm_graph_prefill_chunk_tokens(uint32_t full_attention_cap) {
    return full_attention_cap ? full_attention_cap : DS4_GLM_METAL_FULL_ATTN_DEFAULT_CONTEXT;
}

static bool glm_graph_forward_indexed_tokens(
        ds4_glm_gpu_graph *g,
        const ds4_model   *model,
        const ds4_weights *weights,
        const int         *tokens,
        const float       *input_hc,
        uint32_t           pos0,
        uint32_t           n_tokens,
        float             *output_hc,
        float             *logits_out,
        ds4_session_progress_fn display_progress,
        void              *display_progress_ud,
        uint32_t           display_absolute_base,
        uint32_t           work_done_base,
        uint32_t           work_total) {
    if (!g || !model || !weights || !tokens ||
        g->compact_cache_cap == 0 ||
        g->indexed_prefill_cap == 0 ||
        g->indexed_prefill_score_cap == 0 ||
        !g->batch_indexer_q ||
        !g->batch_indexer_weights ||
        !g->batch_indexer_scores ||
        !g->batch_indexer_selected ||
        !g->batch_qk_low ||
        n_tokens == 0 ||
        g->layer_count == 0 ||
        n_tokens > g->indexed_prefill_cap ||
        !glm_graph_span_fits_context(g, pos0, n_tokens)) {
        return false;
    }
    const uint32_t n_rows = pos0 + n_tokens;
    const uint32_t indexer_top_k = glm_graph_indexer_top_k_limit();
    if (pos0 < indexer_top_k && n_rows > indexer_top_k) {
        return false;
    }
    const uint32_t indexed_selected_count =
        n_rows <= indexer_top_k ? n_rows : indexer_top_k;
    const bool use_causal_range_select = n_rows <= indexer_top_k;
    const bool trace = glm_graph_indexed_prefill_trace_enabled();
    const bool trace_all = trace && glm_graph_indexed_prefill_trace_all();
    const double trace_slow_ms = trace ?
        (double)glm_graph_indexed_prefill_trace_slow_ms() : 0.0;
    const double trace_chunk_t0 = trace ? now_sec() : 0.0;
    if (trace) {
        glm_graph_indexed_prefill_tracef(
                "chunk begin pos=%u tokens=%u rows=%u selected=%u compact_cap=%u score_cap=%u work_base=%u work_total=%u",
                pos0,
                n_tokens,
                n_rows,
                indexed_selected_count,
                g->compact_cache_cap,
                g->indexed_prefill_score_cap,
                work_done_base,
                work_total);
    }
    const double trace_cache_t0 = trace ? now_sec() : 0.0;
    if (!glm_graph_ensure_compact_cache(g, n_rows)) {
        if (trace) {
            glm_graph_indexed_prefill_tracef(
                    "ensure_cache failed pos=%u tokens=%u rows=%u compact_cap=%u",
                    pos0,
                    n_tokens,
                    n_rows,
                    g->compact_cache_cap);
        }
        return false;
    }
    if (trace) {
        const double ms = (now_sec() - trace_cache_t0) * 1000.0;
        if (trace_all || ms >= trace_slow_ms) {
            glm_graph_indexed_prefill_tracef(
                    "ensure_cache done pos=%u tokens=%u rows=%u compact_cap=%u %.3f ms",
                    pos0,
                    n_tokens,
                    n_rows,
                    g->compact_cache_cap,
                    ms);
        }
    }
    for (uint32_t i = 0; i < n_tokens; i++) {
        if (tokens[i] < 0 || tokens[i] >= (int)DS4_N_VOCAB) return false;
    }
    if (!input_hc && !g->has_token_embd) return false;
    if (logits_out && !g->has_output_head) return false;
    glm_graph_reset_prefill_seed_capture(g);

    const double trace_upload_t0 = trace ? now_sec() : 0.0;
    bool ok = glm_graph_upload_tokens(g->prefill_tokens, tokens, n_tokens);
    if (trace) {
        const double ms = (now_sec() - trace_upload_t0) * 1000.0;
        if (trace_all || ms >= trace_slow_ms || !ok) {
            glm_graph_indexed_prefill_tracef(
                    "upload_tokens %s pos=%u tokens=%u %.3f ms",
                    ok ? "done" : "failed",
                    pos0,
                    n_tokens,
                    ms);
        }
    }
    ds4_gpu_tensor *cur = g->batch_cur;
    ds4_gpu_tensor *next = g->batch_next;
    ds4_gpu_tensor *last_hidden = NULL;

    glm_graph_report_prefill_display_progress(display_progress,
                                              display_progress_ud,
                                              display_absolute_base,
                                              work_done_base,
                                              n_tokens,
                                              0,
                                              g->layer_count,
                                              work_total,
                                              true);

    const bool use_all_scalar_kernels =
        n_tokens == 1u && glm_graph_indexed_prefill_scalar_kernels();
    const bool use_scalar_indexer =
        use_all_scalar_kernels ||
        glm_graph_indexed_prefill_scalar_indexer() ||
        !glm_graph_indexed_prefill_batch_indexer();
    const bool force_scalar_attn =
        use_all_scalar_kernels || glm_graph_indexed_prefill_scalar_attn();
    const bool use_batch_qk_low =
        !force_scalar_attn && glm_graph_indexed_prefill_batch_qk_low();
    const bool use_batch_attn_kernel =
        !force_scalar_attn && glm_graph_indexed_prefill_batch_attn_kernel();
    const bool use_split_value_proj =
        use_batch_attn_kernel &&
        g->batch_attn_lora;
    /* Tensor-parallel attention head split: each rank computes half the
     * heads in the qk-low / attention-lora / value-project kernels, the
     * unowned half of batch_heads stays zero, and the full-width attn
     * output projection yields partials combined over the big-gate
     * exchange (same commutative add as the routed-FFN combine). Only the
     * split-value-proj batch chain has head ownership. */
    const bool tp_attn_head_split =
        g->tp_world == 2 &&
        use_batch_attn_kernel &&
        use_split_value_proj &&
        (DS4_N_HEAD % 16u) == 0u &&
        n_tokens >= glm_tp_head_split_min(); /* small batches replicate;
                          * the floor is env-tunable for correctness
                          * isolation (DS4_GLM_TP_HEAD_SPLIT_MIN). */
    const bool use_batch_q_rank_proj = true;
    const bool use_batch_q_proj = true;
    const bool use_batch_indexer_k_proj = true;
    const bool use_batch_kv_proj = true;
    const bool use_batch_indexer_q_proj = true;
    const bool use_batch_indexer_weights_proj = true;
    const bool use_batch_attn_out_proj = true;
    const bool use_batch_ffn = glm_graph_indexed_prefill_batch_ffn();
    const bool stage_sync =
        glm_graph_small_prefill_stage_sync(n_tokens, logits_out != NULL);
    const uint32_t layer_flush_interval = stage_sync ? 0u :
        glm_graph_full_prefill_layer_flush_interval(n_tokens,
                                                    n_tokens,
                                                    logits_out != NULL);
    const uint32_t progress_flush_interval =
        glm_graph_prefill_progress_flush_interval(layer_flush_interval,
                                                  n_tokens,
                                                  display_progress,
                                                  work_total);
    const uint32_t drain_interval =
        progress_flush_interval != 0 ? glm_graph_indexed_prefill_drain_interval() : 0u;
    const bool progress_requested = display_progress && work_total > 0;
    ds4_gpu_set_glm_streaming_prefill_full_layer(false);
    const bool streaming_prefill_sync_each_layer =
        !g->ssd_streaming ||
        glm_graph_streaming_prefill_sync_each_layer(false);

    if (trace) {
        glm_graph_indexed_prefill_tracef(
                "mode pos=%u tokens=%u scalar_indexer=%u batch_qk_low=%u batch_attn=%u split_value=%u batch_ffn=%u progress_flush_interval=%u drain_interval=%u",
                pos0,
                n_tokens,
                use_scalar_indexer ? 1u : 0u,
                use_batch_qk_low ? 1u : 0u,
                use_batch_attn_kernel ? 1u : 0u,
                use_split_value_proj ? 1u : 0u,
                use_batch_ffn ? 1u : 0u,
                progress_flush_interval,
                drain_interval);
    }

    if (ok) {
        const double t0 = trace ? now_sec() : 0.0;
        if (input_hc) {
            ok = ds4_gpu_tensor_write(cur,
                                      0,
                                      input_hc,
                                      (uint64_t)n_tokens * DS4_N_EMBD * sizeof(float)) != 0;
        } else {
            ok = glm_graph_stream_map_token(g, model, weights);
        }
        if (ok) ok = ds4_gpu_begin_commands() != 0;
        if (trace) {
            const double ms = (now_sec() - t0) * 1000.0;
            if (trace_all || ms >= trace_slow_ms || !ok) {
                glm_graph_indexed_prefill_tracef(
                        "begin_commands%s %s pos=%u tokens=%u %.3f ms",
                        input_hc ? "_from_hidden" : "",
                        ok ? "done" : "failed",
                        pos0,
                        n_tokens,
                        ms);
            }
        }
    }
    if (ok && !input_hc) {
        const double t0 = trace ? now_sec() : 0.0;
        ok = ds4_gpu_embed_tokens_quant_tensor(cur,
                                               g->prefill_tokens,
                                               model->map,
                                               model->size,
                                               weights->token_embd->abs_offset,
                                               weights->token_embd->type,
                                               DS4_N_VOCAB,
                                               n_tokens,
                                               DS4_N_EMBD) != 0;
        if (trace) {
            const double ms = (now_sec() - t0) * 1000.0;
            if (trace_all || ms >= trace_slow_ms || !ok) {
                glm_graph_indexed_prefill_tracef(
                        "embed %s pos=%u tokens=%u %.3f ms",
                        ok ? "done" : "failed",
                        pos0,
                        n_tokens,
                        ms);
            }
        }
    }
    if (ok && g->ssd_streaming && streaming_prefill_sync_each_layer) {
        ok = ds4_gpu_end_commands() != 0;
    }

#define DS4_GLM_PROFILE_INDEXED_STAGE(part_, name_) do { \
        if (ok && trace) { \
            const double _trace_stage_now = now_sec(); \
            const double _trace_stage_ms = (_trace_stage_now - trace_stage_t0) * 1000.0; \
            if (trace_all || _trace_stage_ms >= trace_slow_ms) { \
                glm_graph_indexed_prefill_tracef( \
                        "stage layer=%u pos=%u tokens=%u %s.%s encode %.3f ms", \
                        il, \
                        pos0, \
                        n_tokens, \
                        (part_), \
                        (name_), \
                        _trace_stage_ms); \
            } \
            trace_stage_t0 = _trace_stage_now; \
        } \
        if (ok && layer_stage_profile) { \
            ok = metal_graph_layer_stage_profile_boundary((part_), (name_), il, pos0, n_tokens, &layer_stage_t0); \
        } else if (ok && stage_sync) { \
            ok = glm_graph_prefill_stage_sync_boundary(); \
        } \
    } while (0)
    ds4_gpu_tensor *last_indexer_selected = NULL;
    uint32_t last_indexer_selected_count = 0;
    if (ok && tp_attn_head_split) {
        /* The unowned head range of batch_heads must be exactly zero so the
         * full-width attn-output matmul produces partial sums. Owned heads
         * are rewritten every layer, so one fill per chunk suffices. */
        ok = ds4_gpu_tensor_fill_f32(g->batch_heads, 0.0f,
                                     (uint64_t)n_tokens * g->heads_dim) != 0;
    }
    ds4_gpu_tp_set_attn_head_split(tp_attn_head_split ? 1 : 0);
    for (uint32_t il = g->layer_start; ok && il <= g->layer_end; il++) {
        const uint32_t slice_layer_done = il - g->layer_start + 1u;
        if (g->ssd_streaming) {
            ok = glm_graph_stream_map_prefill_layer(g,
                                                    model,
                                                    weights,
                                                    il,
                                                    n_tokens,
                                                    false);
            if (ok) ok = ds4_gpu_begin_commands() != 0;
        }
        const ds4_layer_weights *l = &weights->layer[il];
        const uint32_t kv_raw_dim = (uint32_t)l->attn_kv_a_mqa->dim[1];
        const float rope_base = layer_rope_freq_base(il);
        const float rope_scale = layer_rope_freq_scale(il);
        const uint64_t residual_elems = (uint64_t)n_tokens * DS4_N_EMBD;
        const bool layer_stage_profile = metal_graph_layer_stage_profile_enabled(il);
        double layer_stage_t0 = layer_stage_profile ? now_sec() : 0.0;
        double trace_stage_t0 = trace ? now_sec() : 0.0;
        const double trace_layer_t0 = trace_stage_t0;
        const bool trace_full_indexer = glm_graph_layer_uses_full_indexer(il);
        bool trace_layer_flushed = false;
        if (trace && (trace_all || trace_full_indexer)) {
            glm_graph_indexed_prefill_tracef(
                    "layer begin layer=%u pos=%u tokens=%u rows=%u selected=%u full_indexer=%u",
                    il,
                    pos0,
                    n_tokens,
                    n_rows,
                    indexed_selected_count,
                    trace_full_indexer ? 1u : 0u);
        }
        if (residual_elems > UINT32_MAX) {
            ok = false;
            break;
        }
        if (layer_stage_profile) {
            ok = metal_graph_layer_stage_profile_boundary("glm_indexed_attn",
                                                          NULL,
                                                          il,
                                                          pos0,
                                                          n_tokens,
                                                          &layer_stage_t0);
        }

        if (ok) ok = ds4_gpu_rms_norm_weight_rows_tensor(g->batch_attn_norm,
                                                         cur,
                                                         model->map,
                                                         model->size,
                                                         l->attn_norm->abs_offset,
                                                         DS4_N_EMBD,
                                                         n_tokens,
                                                         DS4_RMS_EPS) != 0;
        DS4_GLM_PROFILE_INDEXED_STAGE("glm_indexed_attn", "attn_norm");
        if (ok) {
            if (n_tokens <= 8u && (glm_decode_ablate_mask() & DS4_GLM_ABLATE_QPATH)) { /* ablate */ } else
            ok = (use_batch_q_rank_proj ?
                  glm_graph_matmul_q8_0_tensor(g->batch_q_rank,
                                               model,
                                               l->attn_q_a->abs_offset,
                                               DS4_N_EMBD,
                                               DS4_N_LORA_Q,
                                               g->batch_attn_norm,
                                               n_tokens) :
                  glm_graph_matmul_q8_0_rows_scalar(g->batch_q_rank,
                                                    model,
                                                    l->attn_q_a->abs_offset,
                                                    DS4_N_EMBD,
                                                    DS4_N_LORA_Q,
                                                    g->batch_attn_norm,
                                                    n_tokens));
        }
        if (ok) ok = ds4_gpu_rms_norm_weight_rows_tensor(g->batch_q_rank_norm,
                                                         g->batch_q_rank,
                                                         model->map,
                                                         model->size,
                                                         l->attn_q_a_norm->abs_offset,
                                                         DS4_N_LORA_Q,
                                                         n_tokens,
                                                         DS4_RMS_EPS) != 0;
        if (ok) {
            ok = (use_batch_q_proj ?
                  glm_graph_matmul_q8_0_tensor(g->batch_q,
                                               model,
                                               l->attn_q_b->abs_offset,
                                               DS4_N_LORA_Q,
                                               g->q_dim,
                                               g->batch_q_rank_norm,
                                               n_tokens) :
                  glm_graph_matmul_q8_0_rows_scalar(g->batch_q,
                                                    model,
                                                    l->attn_q_b->abs_offset,
                                                    DS4_N_LORA_Q,
                                                    g->q_dim,
                                                    g->batch_q_rank_norm,
                                                    n_tokens));
        }
        if (ok) ok = ds4_gpu_rope_tail_tensor(g->batch_q,
                                              n_tokens,
                                              DS4_N_HEAD,
                                              DS4_N_KEY_MLA,
                                              DS4_N_ROT,
                                              pos0,
                                              0,
                                              false,
                                              rope_base,
                                              rope_scale,
                                              0.0f,
                                              1.0f,
                                              DS4_ROPE_YARN_BETA_FAST,
                                              DS4_ROPE_YARN_BETA_SLOW) != 0;
        DS4_GLM_PROFILE_INDEXED_STAGE("glm_indexed_attn", "q_path");

        if (ok && glm_graph_layer_uses_full_indexer(il)) {
            ok = (use_batch_indexer_k_proj ?
                  glm_graph_matmul_q8_0_tensor(g->batch_indexer_k,
                                               model,
                                               l->indexer_attn_k->abs_offset,
                                               DS4_N_EMBD,
                                               DS4_N_INDEXER_HEAD_DIM,
                                               cur,
                                               n_tokens) :
                  glm_graph_matmul_q8_0_rows_scalar(g->batch_indexer_k,
                                                    model,
                                                    l->indexer_attn_k->abs_offset,
                                                    DS4_N_EMBD,
                                                    DS4_N_INDEXER_HEAD_DIM,
                                                    cur,
                                                    n_tokens));
            if (ok) {
                ok = ds4_gpu_glm_store_indexer_k_tensor(
                        g->layer_indexer_key_cache[il],
                        g->batch_indexer_k,
                        model->map,
                        model->size,
                        l->indexer_k_norm->abs_offset,
                        l->indexer_k_norm_b->abs_offset,
                        pos0,
                        n_tokens,
                        g->compact_cache_cap,
                        DS4_N_INDEXER_HEAD_DIM,
                        DS4_N_ROT,
                        0,
                        1.0e-6f,
                        rope_base,
                        rope_scale,
                        0.0f,
                        1.0f,
                        DS4_ROPE_YARN_BETA_FAST,
                        DS4_ROPE_YARN_BETA_SLOW,
                        glm_graph_compact_cache_is_f16()) != 0;
            }
        }
        DS4_GLM_PROFILE_INDEXED_STAGE("glm_indexed_attn", "indexer_k");

        if (ok) {
            ok = (use_batch_kv_proj ?
                  glm_graph_matmul_q8_0_tensor(g->batch_kv_raw,
                                               model,
                                               l->attn_kv_a_mqa->abs_offset,
                                               DS4_N_EMBD,
                                               kv_raw_dim,
                                               g->batch_attn_norm,
                                               n_tokens) :
                  glm_graph_matmul_q8_0_rows_scalar(g->batch_kv_raw,
                                                    model,
                                                    l->attn_kv_a_mqa->abs_offset,
                                                    DS4_N_EMBD,
                                                    kv_raw_dim,
                                                    g->batch_attn_norm,
                                                    n_tokens));
        }
        if (ok) ok = ds4_gpu_glm_kv_lora_rms_norm_tensor(g->batch_kv_norm,
                                                         g->batch_kv_raw,
                                                         model->map,
                                                         model->size,
                                                         l->attn_kv_a_norm->abs_offset,
                                                         n_tokens,
                                                         kv_raw_dim,
                                                         DS4_N_KV_LORA,
                                                         DS4_RMS_EPS) != 0;
        if (ok) {
            ok = ds4_gpu_glm_store_compact_kv_tensor(g->layer_kv_lora_cache[il],
                                                     g->layer_k_rope_cache[il],
                                                     g->batch_kv_norm,
                                                     g->batch_kv_raw,
                                                     pos0,
                                                     n_tokens,
                                                     g->compact_cache_cap,
                                                     kv_raw_dim,
                                                     DS4_N_KV_LORA,
                                                     DS4_N_ROT,
                                                     glm_graph_compact_cache_is_f16()) != 0;
        }
        DS4_GLM_PROFILE_INDEXED_STAGE("glm_indexed_attn", "kv_path");

        if (ok && glm_graph_layer_uses_full_indexer(il)) {
            if (ok && !use_causal_range_select) {
                ok = (use_batch_indexer_q_proj ?
                      glm_graph_matmul_q8_0_tensor(g->batch_indexer_q,
                                                   model,
                                                   l->indexer_attn_q_b->abs_offset,
                                                   DS4_N_LORA_Q,
                                                   (uint64_t)DS4_N_INDEXER_HEAD * DS4_N_INDEXER_HEAD_DIM,
                                                   g->batch_q_rank_norm,
                                                   n_tokens) :
                      glm_graph_matmul_q8_0_rows_scalar(g->batch_indexer_q,
                                                        model,
                                                        l->indexer_attn_q_b->abs_offset,
                                                        DS4_N_LORA_Q,
                                                        (uint64_t)DS4_N_INDEXER_HEAD * DS4_N_INDEXER_HEAD_DIM,
                                                        g->batch_q_rank_norm,
                                                        n_tokens));
                if (ok) ok = ds4_gpu_glm_indexer_rope_tail_tensor(g->batch_indexer_q,
                                                                  n_tokens,
                                                                  DS4_N_INDEXER_HEAD,
                                                                  DS4_N_INDEXER_HEAD_DIM,
                                                                  DS4_N_ROT,
                                                                  pos0,
                                                                  0,
                                                                  rope_base,
                                                                  rope_scale,
                                                                  0.0f,
                                                                  1.0f,
                                                                  DS4_ROPE_YARN_BETA_FAST,
                                                                  DS4_ROPE_YARN_BETA_SLOW) != 0;
                if (ok && glm_graph_indexer_qat()) {
                    ok = ds4_gpu_dsv4_indexer_qat_tensor(g->batch_indexer_q,
                                                         n_tokens * DS4_N_INDEXER_HEAD,
                                                         DS4_N_INDEXER_HEAD_DIM) != 0;
                }
                if (ok) {
                    ok = (use_batch_indexer_weights_proj ?
                          ds4_gpu_matmul_f32_tensor(g->batch_indexer_weights,
                                                    model->map,
                                                    model->size,
                                                    l->indexer_proj->abs_offset,
                                                    DS4_N_EMBD,
                                                    DS4_N_INDEXER_HEAD,
                                                    cur,
                                                    n_tokens) != 0 :
                          glm_graph_matmul_f32_rows_scalar(g->batch_indexer_weights,
                                                           model,
                                                           l->indexer_proj->abs_offset,
                                                           DS4_N_EMBD,
                                                           DS4_N_INDEXER_HEAD,
                                                           cur,
                                                           n_tokens));
                }
                DS4_GLM_PROFILE_INDEXED_STAGE("glm_indexed_attn", "indexer_q_weights");
            }
            if (ok) {
                if (use_causal_range_select) {
                    ok = ds4_gpu_glm_fill_selected_range_batch_tensor(
                                 g->batch_indexer_selected,
                                 n_tokens,
                                 pos0,
                                 indexed_selected_count,
                                 g->compact_cache_cap) != 0;
                    DS4_GLM_PROFILE_INDEXED_STAGE("glm_indexed_attn", "indexer_range");
                } else if (use_scalar_indexer) {
                    const float indexer_scale =
                        1.0f / sqrtf((float)(DS4_N_INDEXER_HEAD_DIM * DS4_N_INDEXER_HEAD));
                    for (uint32_t t = 0; ok && t < n_tokens; t++) {
                        const uint32_t visible = pos0 + t + 1u;
                        ds4_gpu_tensor *indexer_q_view =
                            glm_graph_tensor_row_view_strided(
                                    g->batch_indexer_q,
                                    t,
                                    (uint64_t)DS4_N_INDEXER_HEAD * DS4_N_INDEXER_HEAD_DIM,
                                    (uint64_t)DS4_N_INDEXER_HEAD * DS4_N_INDEXER_HEAD_DIM);
                        ds4_gpu_tensor *indexer_weights_view =
                            glm_graph_tensor_row_view_strided(g->batch_indexer_weights,
                                                              t,
                                                              DS4_N_INDEXER_HEAD,
                                                              DS4_N_INDEXER_HEAD);
                        ds4_gpu_tensor *scores_view =
                            ds4_gpu_tensor_view(g->batch_indexer_scores,
                                                0,
                                                (uint64_t)visible * sizeof(float));
                        ds4_gpu_tensor *selected_view =
                            ds4_gpu_tensor_view(g->batch_indexer_selected,
                                                (uint64_t)t * indexed_selected_count * sizeof(uint32_t),
                                                (uint64_t)indexed_selected_count * sizeof(uint32_t));
                        ok = indexer_q_view && indexer_weights_view && scores_view && selected_view;
                        if (!ok) fprintf(stderr, "ds4: GLM scalar indexed prefill failed to create indexer row views at layer %u token %u\n", il, t);
                        if (ok) {
                            int rc = ds4_gpu_glm_indexer_score_one_tensor(
                                             scores_view,
                                             indexer_q_view,
                                             indexer_weights_view,
                                             g->layer_indexer_key_cache[il],
                                             visible,
                                             DS4_N_INDEXER_HEAD,
                                             DS4_N_INDEXER_HEAD_DIM,
                                             indexer_scale,
                                             glm_graph_compact_cache_is_f16());
                            ok = rc != 0;
                            if (!ok) fprintf(stderr, "ds4: GLM scalar indexed prefill indexer scores failed at layer %u token %u\n", il, t);
                        }
                        DS4_GLM_PROFILE_INDEXED_STAGE("glm_indexed_attn", "indexer_score_scalar");
                        if (ok) {
                            int rc = ds4_gpu_indexer_topk_tensor(selected_view,
                                                                 scores_view,
                                                                 visible,
                                                                 1,
                                                                 indexed_selected_count);
                            ok = rc != 0;
                            if (!ok) fprintf(stderr, "ds4: GLM scalar indexed prefill topk failed at layer %u token %u\n", il, t);
                        }
                        DS4_GLM_PROFILE_INDEXED_STAGE("glm_indexed_attn", "indexer_topk_scalar");
                        ds4_gpu_tensor_free(selected_view);
                        ds4_gpu_tensor_free(scores_view);
                        ds4_gpu_tensor_free(indexer_weights_view);
                        ds4_gpu_tensor_free(indexer_q_view);
                    }
                } else {
                    const float indexer_scale =
                        1.0f / sqrtf((float)(DS4_N_INDEXER_HEAD_DIM * DS4_N_INDEXER_HEAD));
                    const uint32_t score_cap =
                        g->indexed_prefill_score_cap != 0 ?
                            g->indexed_prefill_score_cap :
                            g->indexed_prefill_cap;
                    const uint64_t indexer_q_row_bytes =
                        (uint64_t)DS4_N_INDEXER_HEAD *
                        DS4_N_INDEXER_HEAD_DIM *
                        sizeof(float);
                    const uint64_t indexer_weights_row_bytes =
                        (uint64_t)DS4_N_INDEXER_HEAD * sizeof(float);
                    const uint64_t selected_row_bytes =
                        (uint64_t)indexed_selected_count * sizeof(uint32_t);
                    for (uint32_t t0 = 0; ok && t0 < n_tokens; ) {
                        uint32_t slice = n_tokens - t0;
                        if (slice > score_cap) slice = score_cap;
                        if (slice == 0) {
                            ok = false;
                            break;
                        }

                        ds4_gpu_tensor *indexer_q_view =
                            ds4_gpu_tensor_view(g->batch_indexer_q,
                                                (uint64_t)t0 * indexer_q_row_bytes,
                                                (uint64_t)slice * indexer_q_row_bytes);
                        ds4_gpu_tensor *indexer_weights_view =
                            ds4_gpu_tensor_view(g->batch_indexer_weights,
                                                (uint64_t)t0 * indexer_weights_row_bytes,
                                                (uint64_t)slice * indexer_weights_row_bytes);
                        ds4_gpu_tensor *selected_view =
                            ds4_gpu_tensor_view(g->batch_indexer_selected,
                                                (uint64_t)t0 * selected_row_bytes,
                                                (uint64_t)slice * selected_row_bytes);
                        ok = indexer_q_view && indexer_weights_view && selected_view;
                        if (!ok) {
                            fprintf(stderr,
                                    "ds4: GLM indexed prefill failed to create indexer score slice views at layer %u token %u\n",
                                    il,
                                    t0);
                        }
                        if (ok) {
                            ok = ds4_gpu_glm_indexer_scores_batch_tensor(
                                         g->batch_indexer_scores,
                                         indexer_q_view,
                                         indexer_weights_view,
                                         g->layer_indexer_key_cache[il],
                                         n_rows,
                                         slice,
                                         pos0 + t0,
                                         DS4_N_INDEXER_HEAD,
                                         DS4_N_INDEXER_HEAD_DIM,
                                         indexer_scale,
                                         glm_graph_compact_cache_is_f16()) != 0;
                        }
                        DS4_GLM_PROFILE_INDEXED_STAGE("glm_indexed_attn", "indexer_score");
                        if (ok) {
                            ok = ds4_gpu_indexer_topk_tensor(selected_view,
                                                             g->batch_indexer_scores,
                                                             n_rows,
                                                             slice,
                                                             indexed_selected_count) != 0;
                        }
                        DS4_GLM_PROFILE_INDEXED_STAGE("glm_indexed_attn", "indexer_topk");
                        ds4_gpu_tensor_free(selected_view);
                        ds4_gpu_tensor_free(indexer_weights_view);
                        ds4_gpu_tensor_free(indexer_q_view);
                        t0 += slice;
                    }
                }
            }
            if (ok) {
                last_indexer_selected = g->batch_indexer_selected;
                last_indexer_selected_count = indexed_selected_count;
            }
        } else if (ok && (!last_indexer_selected || last_indexer_selected_count == 0)) {
            ok = false;
        }
        DS4_GLM_PROFILE_INDEXED_STAGE("glm_indexed_attn", "indexer_select");
        metal_graph_debug_dump_tensor("glm_indexed_q",
                                      g->batch_q,
                                      (uint64_t)n_tokens * DS4_N_HEAD * DS4_N_KEY_MLA,
                                      il,
                                      pos0);
        if (use_batch_qk_low) {
            if (n_tokens <= 8u && (glm_decode_ablate_mask() & DS4_GLM_ABLATE_ATTN_CORE)) { /* ablate */ } else
            if (ok) ok = ds4_gpu_glm_qk_lowrank_typed_batch_tensor(g->batch_qk_low,
                                                                   g->batch_q,
                                                                   model->map,
                                                                   model->size,
                                                                   l->attn_k_b->abs_offset,
                                                                   l->attn_k_b->type,
                                                                   n_tokens,
                                                                   DS4_N_HEAD,
                                                                   DS4_N_KV_LORA,
                                                                   (uint32_t)g->q_nope,
                                                                   DS4_N_KEY_MLA) != 0;
        } else {
            for (uint32_t t = 0; ok && t < n_tokens; t++) {
                ds4_gpu_tensor *q_view =
                    glm_graph_tensor_row_view_strided(g->batch_q,
                                                      t,
                                                      g->q_dim,
                                                      g->q_dim);
                ds4_gpu_tensor *qk_low_view =
                    glm_graph_tensor_row_view_strided(
                            g->batch_qk_low,
                            t,
                            (uint64_t)DS4_N_HEAD * DS4_N_KV_LORA,
                            (uint64_t)DS4_N_HEAD * DS4_N_KV_LORA);
                ok = q_view && qk_low_view;
                if (!ok) fprintf(stderr, "ds4: GLM scalar indexed prefill failed to create qk-low row views at layer %u token %u\n", il, t);
                if (ok) {
                    int rc = ds4_gpu_glm_qk_lowrank_typed_tensor(qk_low_view,
                                                                 q_view,
                                                                 model->map,
                                                                 model->size,
                                                                 l->attn_k_b->abs_offset,
                                                                 l->attn_k_b->type,
                                                                 DS4_N_HEAD,
                                                                 DS4_N_KV_LORA,
                                                                 (uint32_t)g->q_nope,
                                                                 DS4_N_KEY_MLA);
                    ok = rc != 0;
                    if (!ok) fprintf(stderr, "ds4: GLM scalar indexed prefill qk-low failed at layer %u token %u\n", il, t);
                }
                ds4_gpu_tensor_free(qk_low_view);
                ds4_gpu_tensor_free(q_view);
            }
        }
        DS4_GLM_PROFILE_INDEXED_STAGE("glm_indexed_attn", "qk_low");
        metal_graph_debug_dump_tensor("glm_indexed_qk_low",
                                      g->batch_qk_low,
                                      (uint64_t)n_tokens * DS4_N_HEAD * DS4_N_KV_LORA,
                                      il,
                                      pos0);
        if (ok && use_batch_attn_kernel) ok = glm_graph_indexed_prefill_attention_boundary();

        if (use_batch_attn_kernel) {
            const uint32_t attn_slice_cap =
                glm_graph_indexed_prefill_batch_attn_slice_tokens();
            for (uint32_t t0 = 0; ok && t0 < n_tokens; ) {
                uint32_t slice = n_tokens - t0;
                if (slice > attn_slice_cap) slice = attn_slice_cap;

                ds4_gpu_tensor *q_view =
                    ds4_gpu_tensor_view(g->batch_q,
                                        (uint64_t)t0 * g->q_dim * sizeof(float),
                                        (uint64_t)slice * g->q_dim * sizeof(float));
                ds4_gpu_tensor *qk_low_view =
                    ds4_gpu_tensor_view(g->batch_qk_low,
                                        (uint64_t)t0 * DS4_N_HEAD * DS4_N_KV_LORA * sizeof(float),
                                        (uint64_t)slice * DS4_N_HEAD * DS4_N_KV_LORA * sizeof(float));
                ds4_gpu_tensor *heads_view =
                    ds4_gpu_tensor_view(g->batch_heads,
                                        (uint64_t)t0 * g->heads_dim * sizeof(float),
                                        (uint64_t)slice * g->heads_dim * sizeof(float));
                ds4_gpu_tensor *attn_lora_view = use_split_value_proj ?
                    ds4_gpu_tensor_view(g->batch_attn_lora,
                                        (uint64_t)t0 * DS4_N_HEAD * DS4_N_KV_LORA * sizeof(float),
                                        (uint64_t)slice * DS4_N_HEAD * DS4_N_KV_LORA * sizeof(float)) :
                    NULL;
                ds4_gpu_tensor *selected_view =
                    ds4_gpu_tensor_view(last_indexer_selected,
                                        (uint64_t)t0 * last_indexer_selected_count * sizeof(uint32_t),
                                        (uint64_t)slice * last_indexer_selected_count * sizeof(uint32_t));
                ok = q_view && qk_low_view && heads_view && selected_view &&
                     (!use_split_value_proj || attn_lora_view);
                if (!ok) {
                    fprintf(stderr, "ds4: GLM sliced indexed prefill failed to create attention views at layer %u token %u\n", il, t0);
                }
                if (n_tokens <= 8u && (glm_decode_ablate_mask() & DS4_GLM_ABLATE_ATTN_CORE)) { /* ablate */ } else if (ok && use_split_value_proj) {
                    int rc = 0;
                    if (use_causal_range_select) {
                        rc = ds4_gpu_glm_attention_indexed_batch_lora_causal_tensor(
                                attn_lora_view,
                                q_view,
                                qk_low_view,
                                g->layer_kv_lora_cache[il],
                                g->layer_k_rope_cache[il],
                                slice,
                                pos0 + t0,
                                last_indexer_selected_count,
                                g->compact_cache_cap,
                                glm_graph_compact_cache_is_f16(),
                                DS4_N_HEAD,
                                DS4_N_KV_LORA,
                                (uint32_t)g->q_nope,
                                DS4_N_ROT,
                                0,
                                rope_base,
                                rope_scale,
                                0.0f,
                                1.0f,
                                DS4_ROPE_YARN_BETA_FAST,
                                DS4_ROPE_YARN_BETA_SLOW);
                    } else {
                        rc = ds4_gpu_glm_attention_indexed_batch_lora_valid_tensor(
                                attn_lora_view,
                                q_view,
                                qk_low_view,
                                g->layer_kv_lora_cache[il],
                                g->layer_k_rope_cache[il],
                                selected_view,
                                slice,
                                last_indexer_selected_count,
                                g->compact_cache_cap,
                                glm_graph_compact_cache_is_f16(),
                                DS4_N_HEAD,
                                DS4_N_KV_LORA,
                                (uint32_t)g->q_nope,
                                DS4_N_ROT,
                                0,
                                rope_base,
                                rope_scale,
                                0.0f,
                                1.0f,
                                DS4_ROPE_YARN_BETA_FAST,
                                DS4_ROPE_YARN_BETA_SLOW);
                    }
                    ok = rc != 0;
                    if (!ok) fprintf(stderr, "ds4: GLM sliced indexed prefill attention-lora failed at layer %u token %u\n", il, t0);
                    if (ok && layer_stage_profile) {
                        ok = metal_graph_layer_stage_profile_boundary("glm_indexed_attn",
                                                                      "attention_lora",
                                                                      il,
                                                                      pos0 + t0,
                                                                      slice,
                                                                      &layer_stage_t0);
                    }
                    if (ok) {
                        rc = ds4_gpu_glm_value_project_typed_batch_heads_tensor(
                                    heads_view,
                                    attn_lora_view,
                                    model->map,
                                    model->size,
                                    l->attn_v_b->abs_offset,
                                    l->attn_v_b->type,
                                    slice,
                                    DS4_N_HEAD,
                                    DS4_N_KV_LORA,
                                    DS4_N_VALUE_MLA);
                        ok = rc != 0;
                        if (!ok) fprintf(stderr, "ds4: GLM sliced indexed prefill value project failed at layer %u token %u\n", il, t0);
                    }
                    if (ok && layer_stage_profile) {
                        ok = metal_graph_layer_stage_profile_boundary("glm_indexed_attn",
                                                                      "value_project",
                                                                      il,
                                                                      pos0 + t0,
                                                                      slice,
                                                                      &layer_stage_t0);
                    }
                } else if (ok) {
                    int rc = ds4_gpu_glm_attention_indexed_batch_typed_tensor(heads_view,
                                                                              q_view,
                                                                              qk_low_view,
                                                                              g->layer_kv_lora_cache[il],
                                                                              g->layer_k_rope_cache[il],
                                                                              model->map,
                                                                              model->size,
                                                                              l->attn_v_b->abs_offset,
                                                                              l->attn_v_b->type,
                                                                              selected_view,
                                                                              slice,
                                                                              last_indexer_selected_count,
                                                                              g->compact_cache_cap,
                                                                              glm_graph_compact_cache_is_f16(),
                                                                              DS4_N_HEAD,
                                                                              DS4_N_KV_LORA,
                                                                              (uint32_t)g->q_nope,
                                                                              DS4_N_ROT,
                                                                              DS4_N_VALUE_MLA,
                                                                              0,
                                                                              rope_base,
                                                                              rope_scale,
                                                                              0.0f,
                                                                              1.0f,
                                                                              DS4_ROPE_YARN_BETA_FAST,
                                                                              DS4_ROPE_YARN_BETA_SLOW);
                    ok = rc != 0;
                    if (!ok) fprintf(stderr, "ds4: GLM sliced indexed prefill indexed attention failed at layer %u token %u\n", il, t0);
                    if (ok && layer_stage_profile) {
                        ok = metal_graph_layer_stage_profile_boundary("glm_indexed_attn",
                                                                      "attention_fused",
                                                                      il,
                                                                      pos0 + t0,
                                                                      slice,
                                                                      &layer_stage_t0);
                    }
                }
                ds4_gpu_tensor_free(selected_view);
                ds4_gpu_tensor_free(attn_lora_view);
                ds4_gpu_tensor_free(heads_view);
                ds4_gpu_tensor_free(qk_low_view);
                ds4_gpu_tensor_free(q_view);
                t0 += slice;
            }
            if (ok) ok = glm_graph_indexed_prefill_attention_boundary();
        } else {
            for (uint32_t t = 0; ok && t < n_tokens; t++) {
                ds4_gpu_tensor *q_view =
                    glm_graph_tensor_row_view_strided(g->batch_q,
                                                      t,
                                                      g->q_dim,
                                                      g->q_dim);
                ds4_gpu_tensor *qk_low_view =
                    glm_graph_tensor_row_view_strided(
                            g->batch_qk_low,
                            t,
                            (uint64_t)DS4_N_HEAD * DS4_N_KV_LORA,
                            (uint64_t)DS4_N_HEAD * DS4_N_KV_LORA);
                ds4_gpu_tensor *heads_view =
                    glm_graph_tensor_row_view_strided(g->batch_heads,
                                                      t,
                                                      g->heads_dim,
                                                      g->heads_dim);
                ds4_gpu_tensor *selected_view =
                    ds4_gpu_tensor_view(last_indexer_selected,
                                        (uint64_t)t * last_indexer_selected_count * sizeof(uint32_t),
                                        (uint64_t)last_indexer_selected_count * sizeof(uint32_t));
                ok = q_view && qk_low_view && heads_view && selected_view;
                if (!ok) fprintf(stderr, "ds4: GLM scalar indexed prefill failed to create attention row views at layer %u token %u\n", il, t);
                if (ok) {
                    int rc = ds4_gpu_glm_attention_indexed_decode_typed_tensor(heads_view,
                                                                               q_view,
                                                                               qk_low_view,
                                                                               g->layer_kv_lora_cache[il],
                                                                               g->layer_k_rope_cache[il],
                                                                               model->map,
                                                                               model->size,
                                                                               l->attn_v_b->abs_offset,
                                                                               l->attn_v_b->type,
                                                                               selected_view,
                                                                               last_indexer_selected_count,
                                                                               g->compact_cache_cap,
                                                                               glm_graph_compact_cache_is_f16(),
                                                                               DS4_N_HEAD,
                                                                               DS4_N_KV_LORA,
                                                                               (uint32_t)g->q_nope,
                                                                               DS4_N_ROT,
                                                                               DS4_N_VALUE_MLA,
                                                                               0,
                                                                               rope_base,
                                                                               rope_scale,
                                                                               0.0f,
                                                                               1.0f,
                                                                               DS4_ROPE_YARN_BETA_FAST,
                                                                               DS4_ROPE_YARN_BETA_SLOW);
                    ok = rc != 0;
                    if (!ok) fprintf(stderr, "ds4: GLM scalar indexed prefill indexed attention failed at layer %u token %u\n", il, t);
                }
                ds4_gpu_tensor_free(selected_view);
                ds4_gpu_tensor_free(heads_view);
                ds4_gpu_tensor_free(qk_low_view);
                ds4_gpu_tensor_free(q_view);
            }
        }
        DS4_GLM_PROFILE_INDEXED_STAGE("glm_indexed_attn", "attention");
        metal_graph_debug_dump_tensor("glm_indexed_heads",
                                      g->batch_heads,
                                      (uint64_t)n_tokens * g->heads_dim,
                                      il,
                                      pos0);
        if (ok && tp_attn_head_split) {
            ok = glm_graph_tp_batch_bounce_ready(g, n_tokens);
        }
        if (ok) {
            /* Under the head split the projection input has zeros in the
             * unowned head columns, so the result is this rank's partial;
             * it must land in the shared bounce tensor for the exchange. */
            ds4_gpu_tensor *attn_out_dst =
                tp_attn_head_split ? g->tp_bounce_out : g->batch_attn_out;
            if (n_tokens <= 8u && (glm_decode_ablate_mask() & DS4_GLM_ABLATE_ATTN_OUT)) { /* ablate */ } else
            ok = (use_batch_attn_out_proj ?
                  glm_graph_matmul_q8_0_tensor(attn_out_dst,
                                               model,
                                               l->attn_output->abs_offset,
                                               g->heads_dim,
                                               DS4_N_EMBD,
                                               g->batch_heads,
                                               n_tokens) :
                  glm_graph_matmul_q8_0_rows_scalar(attn_out_dst,
                                                    model,
                                                    l->attn_output->abs_offset,
                                                    g->heads_dim,
                                                    DS4_N_EMBD,
                                                    g->batch_heads,
                                                    n_tokens));
        }
        if (ok && tp_attn_head_split) {
            ok = glm_graph_tp_batch_ffn_combine(g, il, g->batch_attn_out, n_tokens);
        }
        if (ok) ok = ds4_gpu_add_tensor(g->batch_after_attn,
                                        cur,
                                        g->batch_attn_out,
                                        (uint32_t)residual_elems) != 0;
        DS4_GLM_PROFILE_INDEXED_STAGE("glm_indexed_attn", "attn_output");
        metal_graph_debug_dump_tensor("glm_indexed_after_attn",
                                      g->batch_after_attn,
                                      (uint64_t)n_tokens * DS4_N_EMBD,
                                      il,
                                      pos0);
        metal_graph_debug_dump_tensor("glm_indexed_attn_out",
                                      g->batch_attn_out,
                                      (uint64_t)n_tokens * DS4_N_EMBD,
                                      il,
                                      pos0);
        if (ok && use_batch_ffn) {
            ok = glm_graph_encode_ffn_batch(g,
                                            model,
                                            weights,
                                            l,
                                            il,
                                            pos0,
                                            g->batch_after_attn,
                                            next,
                                            n_tokens,
                                            false,
                                            layer_stage_profile,
                                            stage_sync,
                                            layer_stage_profile ? &layer_stage_t0 : NULL);
        } else if (ok) {
            const bool use_batch_ffn_norm =
                n_tokens > 1 && glm_graph_indexed_prefill_batch_ffn_norm();
            if (use_batch_ffn_norm) {
                ok = ds4_gpu_rms_norm_weight_rows_tensor(g->batch_ffn_norm,
                                                          g->batch_after_attn,
                                                          model->map,
                                                          model->size,
                                                          l->ffn_norm->abs_offset,
                                                          DS4_N_EMBD,
                                                          n_tokens,
                                                          DS4_RMS_EPS) != 0;
            }
            if (use_batch_ffn_norm) {
                DS4_GLM_PROFILE_INDEXED_STAGE("glm_indexed_ffn", "ffn_norm");
            }
            if (ok &&
                use_batch_ffn_norm &&
                il >= DS4_N_LEADING_DENSE &&
                glm_graph_indexed_prefill_batch_routed_moe()) {
                ok = glm_graph_encode_sparse_ffn_indexed_batch_routed_moe(
                        g,
                        model,
                        l,
                        il,
                        pos0,
                        g->batch_after_attn,
                        next,
                        n_tokens,
                        layer_stage_profile,
                        stage_sync,
                        layer_stage_profile ? &layer_stage_t0 : NULL);
            } else for (uint32_t t = 0; ok && t < n_tokens; t++) {
                ds4_gpu_tensor *after_attn_view =
                    glm_graph_tensor_row_view_strided(g->batch_after_attn,
                                                      t,
                                                      DS4_N_EMBD,
                                                      DS4_N_EMBD);
                ds4_gpu_tensor *ffn_norm_view = use_batch_ffn_norm ?
                    glm_graph_tensor_row_view_strided(g->batch_ffn_norm,
                                                      t,
                                                      DS4_N_EMBD,
                                                      DS4_N_EMBD) :
                    NULL;
                ds4_gpu_tensor *next_view =
                    glm_graph_tensor_row_view_strided(next,
                                                      t,
                                                      DS4_N_EMBD,
                                                      DS4_N_EMBD);
                ok = after_attn_view && next_view &&
                     (!use_batch_ffn_norm || ffn_norm_view);
                if (ok && use_batch_ffn_norm) {
                    ok = glm_graph_encode_ffn_one_normed_from(g,
                                                              model,
                                                              l,
                                                              il,
                                                              pos0 + t,
                                                              ffn_norm_view,
                                                              after_attn_view,
                                                              next_view,
                                                              g->ffn_gate,
                                                              g->ffn_up,
                                                              g->ffn_mid,
                                                              g->ffn_out,
                                                              g->ffn_sum,
                                                              g->attn_out,
                                                              false,
                                                              NULL);
                } else if (ok) {
                    ok = glm_graph_encode_ffn_one_from(g,
                                                       model,
                                                       l,
                                                       il,
                                                       pos0 + t,
                                                       after_attn_view,
                                                       next_view,
                                                       g->ffn_norm,
                                                       g->ffn_gate,
                                                       g->ffn_up,
                                                       g->ffn_mid,
                                                       g->ffn_out,
                                                       g->ffn_sum,
                                                       g->attn_out,
                                                       false,
                                                       NULL);
                }
                ds4_gpu_tensor_free(next_view);
                ds4_gpu_tensor_free(ffn_norm_view);
                ds4_gpu_tensor_free(after_attn_view);
            }
        }
        if (ok) {
            ds4_gpu_tensor *tmp = cur;
            cur = next;
            next = tmp;
        }
        if (ok && glm_debug_hidden_dump_layer_match(il)) {
            ok = ds4_gpu_end_commands() != 0;
            if (ok) {
                for (uint32_t r = 0; r < n_tokens; r++)
                    glm_debug_dump_hidden_layer(cur, r, il, pos0 + r);
                glm_debug_dump_raw_layer(g->batch_router_selected, "sel",
                                         (uint64_t)n_tokens * DS4_N_EXPERT_USED * sizeof(int32_t),
                                         il, -1);
                glm_debug_dump_raw_layer(g->batch_router_weights, "selw",
                                         (uint64_t)n_tokens * DS4_N_EXPERT_USED * sizeof(float),
                                         il, -1);
                ok = ds4_gpu_begin_commands() != 0;
            }
        }
        if (ok &&
            !g->ssd_streaming &&
            progress_flush_interval != 0 &&
            (il < g->layer_end || progress_requested) &&
            (slice_layer_done % progress_flush_interval) == 0) {
            const uint32_t work_done =
                work_done_base + (uint32_t)(((uint64_t)n_tokens * slice_layer_done) / g->layer_count);
            const bool drain_now =
                drain_interval != 0 &&
                il < g->layer_end &&
                (slice_layer_done % drain_interval) == 0;
            const char *command_action = drain_now ? "drain" : "flush";
            const double trace_command_t0 = trace ? now_sec() : 0.0;
            if (trace && (trace_all || trace_full_indexer || drain_now)) {
                glm_graph_indexed_prefill_tracef(
                        "layer %s begin layer=%u pos=%u tokens=%u work=%u/%u",
                        command_action,
                        il,
                        pos0,
                        n_tokens,
                        work_done,
                        work_total);
            }
            if (drain_now) {
                ok = ds4_gpu_end_commands() != 0;
                if (ok) ok = ds4_gpu_begin_commands() != 0;
            } else {
                ok = ds4_gpu_flush_commands() != 0;
            }
            if (trace) {
                const double trace_command_done = now_sec();
                const double command_ms = (trace_command_done - trace_command_t0) * 1000.0;
                const double layer_ms = (trace_command_done - trace_layer_t0) * 1000.0;
                trace_layer_flushed = true;
                if (trace_all || trace_full_indexer || drain_now ||
                    command_ms >= trace_slow_ms || layer_ms >= trace_slow_ms || !ok) {
                    glm_graph_indexed_prefill_tracef(
                            "layer %s %s layer=%u pos=%u tokens=%u command=%.3f ms layer_total=%.3f ms work=%u/%u",
                            command_action,
                            ok ? "done" : "failed",
                            il,
                            pos0,
                            n_tokens,
                            command_ms,
                            layer_ms,
                            work_done,
                            work_total);
                }
            }
            if (ok) {
                const bool progress_completed =
                    drain_interval == 0 || drain_now;
                if (progress_completed) {
                    glm_graph_report_prefill_display_progress(display_progress,
                                                              display_progress_ud,
                                                              display_absolute_base,
                                                              work_done_base,
                                                              n_tokens,
                                                              slice_layer_done,
                                                              g->layer_count,
                                                              work_total,
                                                              logits_out == NULL && output_hc == NULL);
                }
            }
        }
        if (g->ssd_streaming) {
            if (streaming_prefill_sync_each_layer) {
                if (ok) ok = ds4_gpu_end_commands() != 0;
                else (void)ds4_gpu_synchronize();
            } else if (!ok) {
                (void)ds4_gpu_synchronize();
            }
            if (ok) {
                glm_graph_report_prefill_display_progress(display_progress,
                                                          display_progress_ud,
                                                          display_absolute_base,
                                                          work_done_base,
                                                          n_tokens,
                                                          slice_layer_done,
                                                          g->layer_count,
                                                          work_total,
                                                          logits_out == NULL && output_hc == NULL);
            }
        }
        if (trace && ok) {
            const double layer_ms = (now_sec() - trace_layer_t0) * 1000.0;
            if (trace_all || (!trace_layer_flushed && layer_ms >= trace_slow_ms)) {
                glm_graph_indexed_prefill_tracef(
                        "layer end layer=%u pos=%u tokens=%u flushed=%u layer_total=%.3f ms",
                        il,
                        pos0,
                        n_tokens,
                        trace_layer_flushed ? 1u : 0u,
                        layer_ms);
            }
        }
    }
    ds4_gpu_tp_set_attn_head_split(0);
#undef DS4_GLM_PROFILE_INDEXED_STAGE
    if (ok && !g->ssd_streaming) {
        const double trace_end_t0 = trace ? now_sec() : 0.0;
        if (trace) {
            glm_graph_indexed_prefill_tracef(
                    "chunk end_commands begin pos=%u tokens=%u",
                    pos0,
                    n_tokens);
        }
        ok = ds4_gpu_end_commands() != 0;
        if (trace) {
            const double end_ms = (now_sec() - trace_end_t0) * 1000.0;
            const double chunk_ms = (now_sec() - trace_chunk_t0) * 1000.0;
            glm_graph_indexed_prefill_tracef(
                    "chunk end_commands %s pos=%u tokens=%u end=%.3f ms chunk_total=%.3f ms",
                    ok ? "done" : "failed",
                    pos0,
                    n_tokens,
                    end_ms,
                    chunk_ms);
        }
    } else if (!ok) {
        if (trace) {
            glm_graph_indexed_prefill_tracef(
                    "chunk failed before end pos=%u tokens=%u elapsed=%.3f ms",
                    pos0,
                    n_tokens,
                    (now_sec() - trace_chunk_t0) * 1000.0);
        }
        (void)ds4_gpu_synchronize();
    }
    if (ok &&
        g->ssd_streaming &&
        !streaming_prefill_sync_each_layer &&
        !output_hc &&
        !logits_out) {
        ok = ds4_gpu_end_commands() != 0;
    }
    if (ok) {
        glm_graph_report_prefill_display_progress(display_progress,
                                                  display_progress_ud,
                                                  display_absolute_base,
                                                  work_done_base,
                                                  n_tokens,
                                                  g->layer_count,
                                                  g->layer_count,
                                                  work_total,
                                                  logits_out == NULL && output_hc == NULL);
    }
    if (ok && output_hc) {
        ok = ds4_gpu_tensor_read(cur,
                                 0,
                                 output_hc,
                                 (uint64_t)n_tokens * DS4_N_EMBD * sizeof(float)) != 0;
    }
    if (ok && logits_out) {
        ok = glm_graph_seed_streaming_expert_cache_from_prefill(g,
                                                                model,
                                                                weights);
    }
    if (ok && logits_out) {
        last_hidden = glm_graph_tensor_row_view_strided(cur,
                                                        n_tokens - 1u,
                                                        DS4_N_EMBD,
                                                        DS4_N_EMBD);
        ok = last_hidden != NULL;
        if (ok && g->ssd_streaming) ok = glm_graph_stream_map_output(g, model, weights);
        if (ok) ok = glm_graph_forward_output_head(g, model, weights, last_hidden, logits_out);
        if (ok) {
            glm_graph_report_prefill_display_progress(display_progress,
                                                      display_progress_ud,
                                                      display_absolute_base,
                                                      work_done_base,
                                                      n_tokens,
                                                      g->layer_count,
                                                      g->layer_count,
                                                      work_total,
                                                      true);
        }
    }
    ds4_gpu_tensor_free(last_hidden);
    ds4_gpu_set_glm_streaming_prefill_full_layer(false);
    return ok;
}

static bool glm_graph_use_streaming_token_prefill(
        const ds4_glm_gpu_graph *g,
        uint32_t                 pos0,
        uint32_t                 n_tokens);
static uint32_t glm_graph_streaming_token_prefill_max_tokens(void);
static bool glm_graph_prefill_token_major(
        ds4_glm_gpu_graph      *g,
        const ds4_model        *model,
        const ds4_weights      *weights,
        const int              *tokens,
        uint32_t                pos0,
        uint32_t                n_tokens,
        float                  *logits_out,
        ds4_session_progress_fn display_progress,
        void                   *display_progress_ud,
        uint32_t                display_absolute_base,
        uint32_t                work_done_base,
        uint32_t                work_total);

static bool glm_graph_prefill_range(
        ds4_glm_gpu_graph      *g,
        const ds4_model        *model,
        const ds4_weights      *weights,
        const int              *tokens,
        uint32_t                pos0,
        uint32_t                n_tokens,
        float                  *logits_out,
        ds4_session_progress_fn progress,
        void                   *progress_ud,
        uint32_t                progress_total) {
    if (n_tokens == 0) return true;
    if (!glm_graph_span_fits_context(g, pos0, n_tokens)) return false;
    const uint32_t chunk_max = glm_graph_prefill_chunk_tokens(g->ctx_cap);
    uint32_t done = 0;
    while (done < n_tokens) {
        const uint32_t pos = pos0 + done;
        if (!g->full_kv_cache) {
            const uint32_t remaining = n_tokens - done;
            uint32_t chunk = 1;
            if (glm_graph_use_streaming_token_prefill(g, pos, remaining)) {
                chunk = remaining;
                const uint32_t token_prefill_max =
                    glm_graph_streaming_token_prefill_max_tokens();
                if (token_prefill_max != 0 && chunk > token_prefill_max) {
                    chunk = token_prefill_max;
                }
                float *dst_logits = (done + chunk == n_tokens) ? logits_out : NULL;
                if (!glm_graph_prefill_token_major(g,
                                                   model,
                                                   weights,
                                                   tokens + done,
                                                   pos,
                                                   chunk,
                                                   dst_logits,
                                                   progress,
                                                   progress_ud,
                                                   pos0,
                                                   done,
                                                   n_tokens)) {
                    return false;
                }
            } else if (glm_graph_indexed_prefill_batch_ready(g, pos)) {
                chunk = remaining;
                if (chunk > g->indexed_prefill_cap) chunk = g->indexed_prefill_cap;
                chunk = glm_graph_limit_indexed_prefill_chunk(pos, chunk);
                if (chunk == 0) chunk = 1;
                float *dst_logits = (done + chunk == n_tokens) ? logits_out : NULL;
                if (!glm_graph_forward_indexed_tokens(g,
                                                      model,
                                                      weights,
                                                      tokens + done,
                                                      NULL,
                                                      pos,
                                                      chunk,
                                                      NULL,
                                                      dst_logits,
                                                      progress,
                                                      progress_ud,
                                                      pos0,
                                                      done,
                                                      n_tokens)) {
                    return false;
                }
            } else {
                float *dst_logits = (done + 1u == n_tokens) ? logits_out : NULL;
                if (!glm_graph_forward_token(g,
                                             model,
                                             weights,
                                             tokens[done],
                                             NULL,
                                             pos,
                                             NULL,
                                             dst_logits,
                                             false)) {
                    return false;
                }
            }
            done += chunk;
            if (progress) {
                const uint32_t current = pos0 + done;
                progress(progress_ud,
                         "prefill_chunk",
                         current,
                         progress_total ? progress_total : pos0 + n_tokens);
            }
            continue;
        }
        if (pos >= g->ctx_cap) {
            if (g->compact_cache_cap == 0) {
                glm_graph_log_full_attention_limit(g, pos, n_tokens - done);
                return false;
            }
            while (done < n_tokens) {
                const uint32_t cur_pos = pos0 + done;
                const bool use_indexed_batch =
                    glm_graph_indexed_prefill_batch_ready(g, cur_pos);
                if (use_indexed_batch) {
                    uint32_t chunk = n_tokens - done;
                    if (chunk > g->indexed_prefill_cap) chunk = g->indexed_prefill_cap;
                    chunk = glm_graph_limit_indexed_prefill_chunk(cur_pos, chunk);
                    float *dst_logits = (done + chunk == n_tokens) ? logits_out : NULL;
                    if (!glm_graph_forward_indexed_tokens(g,
                                                          model,
                                                          weights,
                                                          tokens + done,
                                                          NULL,
                                                          cur_pos,
                                                          chunk,
                                                          NULL,
                                                          dst_logits,
                                                          progress,
                                                          progress_ud,
                                                          pos0,
                                                          done,
                                                          n_tokens)) {
                        return false;
                    }
                    done += chunk;
                } else {
                    float *dst_logits = (done + 1u == n_tokens) ? logits_out : NULL;
                    if (!glm_graph_forward_token(g,
                                                 model,
                                                 weights,
                                                 tokens[done],
                                                 NULL,
                                                 cur_pos,
                                                 NULL,
                                                 dst_logits,
                                                 false)) {
                        return false;
                    }
                    done++;
                }
                if (progress) {
                    const uint32_t current = pos0 + done;
                    progress(progress_ud,
                             "prefill_chunk",
                             current,
                             progress_total ? progress_total : pos0 + n_tokens);
                }
            }
            return true;
        }
        uint32_t chunk = n_tokens - done;
        const uint32_t full_remaining = g->ctx_cap - pos;
        if (chunk > full_remaining) chunk = full_remaining;
        if (chunk > chunk_max) chunk = chunk_max;
        float *dst_logits = (done + chunk == n_tokens) ? logits_out : NULL;
        if (glm_graph_use_streaming_token_prefill(g, pos, chunk)) {
            if (!glm_graph_prefill_token_major(g,
                                               model,
                                               weights,
                                               tokens + done,
                                               pos,
                                               chunk,
                                               dst_logits,
                                               progress,
                                               progress_ud,
                                               pos0,
                                               done,
                                               n_tokens)) {
                return false;
            }
        } else if (!glm_graph_forward_tokens(g,
                                             model,
                                             weights,
                                             tokens + done,
                                             NULL,
                                             pos,
                                             chunk,
                                             NULL,
                                             dst_logits,
                                             progress,
                                             progress_ud,
                                             pos0,
                                             done,
                                             n_tokens)) {
            return false;
        }
        done += chunk;
        if (progress) {
            const uint32_t current = pos0 + done;
            progress(progress_ud,
                     "prefill_chunk",
                     current,
                     progress_total ? progress_total : pos0 + n_tokens);
        }
    }
    return true;
}

/*
 * For very short GLM SSD-streaming prefills, Metal still benefits from the
 * token-major path because it reuses the normal decode graph and warms the
 * decode expert cache. On ROCm/Strix Halo the indexed batch prefill is faster
 * now that streamed batch routing and expert cache seeding are implemented, so
 * ROCm defaults to canonical batch prefill unless the env override below opts
 * token-major prefill back in.
 */
enum { DS4_GLM_STREAM_PREFILL_TOKEN_MAJOR_MAX_TOKENS = 64 };

static uint32_t glm_graph_streaming_token_prefill_default_max_tokens(void) {
#ifdef DS4_ROCM_BUILD
    return 0;
#else
    return DS4_GLM_STREAM_PREFILL_TOKEN_MAJOR_MAX_TOKENS;
#endif
}

static uint32_t glm_graph_streaming_token_prefill_max_tokens(void) {
    const char *env = glm_graph_env_value(
            "DS4_ROCM_GLM_STREAMING_TOKEN_PREFILL_MAX",
            "DS4_METAL_GLM_STREAMING_TOKEN_PREFILL_MAX");
    if (!env || !env[0]) env = getenv("DS4_GLM_STREAMING_TOKEN_PREFILL_MAX");
    const uint32_t default_max =
        glm_graph_streaming_token_prefill_default_max_tokens();
    if (!env || !env[0]) return default_max;
    char *end = NULL;
    errno = 0;
    unsigned long v = strtoul(env, &end, 10);
    if (end == env || errno != 0 || v > UINT32_MAX) {
        return default_max;
    }
    return (uint32_t)v;
}

static bool glm_graph_use_streaming_token_prefill(
        const ds4_glm_gpu_graph *g,
        uint32_t                 pos0,
        uint32_t                 n_tokens) {
    if (!g || !g->ssd_streaming || g->quality || n_tokens == 0) return false;
    if (getenv("DS4_GLM_DISABLE_STREAMING_TOKEN_PREFILL") != NULL ||
        glm_graph_env_present("DS4_ROCM_GLM_DISABLE_STREAMING_TOKEN_PREFILL",
                              "DS4_METAL_GLM_DISABLE_STREAMING_TOKEN_PREFILL")) {
        return false;
    }
    if (!glm_graph_span_fits_full_attention(g, pos0, n_tokens)) return false;
    const uint32_t max_tokens = glm_graph_streaming_token_prefill_max_tokens();
    return max_tokens != 0 && n_tokens <= max_tokens;
}

static bool glm_graph_prefill_token_major(
        ds4_glm_gpu_graph      *g,
        const ds4_model        *model,
        const ds4_weights      *weights,
        const int              *tokens,
        uint32_t                pos0,
        uint32_t                n_tokens,
        float                  *logits_out,
        ds4_session_progress_fn display_progress,
        void                   *display_progress_ud,
        uint32_t                display_absolute_base,
        uint32_t                work_done_base,
        uint32_t                work_total) {
    if (!g || !model || !weights || !tokens || n_tokens == 0) return false;
    for (uint32_t i = 0; i < n_tokens; i++) {
        const bool last = i + 1u == n_tokens;
        float *dst_logits = (last && logits_out) ? logits_out : NULL;
        if (!glm_graph_forward_token(g,
                                     model,
                                     weights,
                                     tokens[i],
                                     NULL,
                                     pos0 + i,
                                     NULL,
                                     dst_logits,
                                     false)) {
            return false;
        }
        glm_graph_report_prefill_display_progress(display_progress,
                                                  display_progress_ud,
                                                  display_absolute_base,
                                                  work_done_base + i + 1u,
                                                  n_tokens,
                                                  0,
                                                  1,
                                                  work_total,
                                                  false);
    }
    return true;
}

static bool glm_graph_maybe_warm_compact_indexer_after_prefill(
        ds4_glm_gpu_graph      *g,
        const ds4_model        *model,
        const ds4_weights      *weights,
        uint32_t                next_pos) {
    if (!g || !model || !weights) return false;
    if (g->compact_cache_cap == 0 ||
        g->indexer_full_layers == 0 ||
        next_pos < g->ctx_cap) {
        return true;
    }
    if (next_pos >= g->ctx_size) return true;
    return glm_graph_warm_compact_indexer_store(g, model, weights, next_pos);
}

static bool glm_graph_begin_commands_if_needed(void) {
    return ds4_gpu_commands_active() || ds4_gpu_begin_commands() != 0;
}

static bool glm_graph_end_commands_if_active(void) {
    return !ds4_gpu_commands_active() || ds4_gpu_end_commands() != 0;
}

static bool glm_graph_streaming_decode_sync_each_layer(void) {
#ifdef DS4_ROCM_BUILD
    const char *env = glm_graph_env_value(
            "DS4_ROCM_GLM_STREAMING_DECODE_SYNC_EACH_LAYER",
            "DS4_METAL_GLM_STREAMING_DECODE_SYNC_EACH_LAYER");
    if (!env) env = getenv("DS4_GLM_STREAMING_DECODE_SYNC_EACH_LAYER");
    return glm_graph_env_truthy(env);
#else
    return true;
#endif
}

static bool glm_graph_forward_token(
        ds4_glm_gpu_graph *g,
        const ds4_model   *model,
        const ds4_weights *weights,
        int                token,
        const float       *input_hc,
        uint32_t           pos,
        float             *output_hc,
        float             *logits_out,
        bool               defer_completion) {
#define DS4_GLM_FT_FAIL(why) do { \
        if (getenv("DS4_GLM_TP_DEBUG")) \
            fprintf(stderr, "ds4: glm forward_token fail pos=%u: %s\n", pos, why); \
    } while (0)
    if (!g || !model || !weights ||
        token < 0 || token >= (int)DS4_N_VOCAB ||
        g->layer_count == 0 ||
        pos >= g->ctx_size ||
        (defer_completion &&
         (g->ssd_streaming || !ds4_gpu_commands_active()))) {
        DS4_GLM_FT_FAIL("arg guard");
        return false;
    }
    if (!input_hc && !g->has_token_embd) { DS4_GLM_FT_FAIL("no token embd"); return false; }
    if (logits_out && !g->has_output_head) { DS4_GLM_FT_FAIL("no output head"); return false; }
    const bool use_indexed_attention =
        glm_graph_decode_uses_indexed_attention(g, pos, logits_out);
    uint32_t decode_layer_flush_interval = 0;
    if (logits_out != NULL) {
        decode_layer_flush_interval = use_indexed_attention ? 4u : 32u;
        const char *dfi = getenv("DS4_GLM_DECODE_FLUSH_INTERVAL");
        if (dfi && dfi[0]) {
            int v = atoi(dfi);
            decode_layer_flush_interval = v <= 0 ? 0u : (uint32_t)v;
        }
        if (decode_layer_flush_interval > g->layer_count) {
            decode_layer_flush_interval = g->layer_count;
        }
        if (defer_completion) decode_layer_flush_interval = 0;
    }
    if (pos >= g->ctx_cap && !use_indexed_attention) {
        glm_graph_log_full_attention_limit(g, pos, 1);
        DS4_GLM_FT_FAIL("full attention limit");
        return false;
    }
    if (g->compact_cache_cap != 0 &&
        !glm_graph_ensure_compact_cache(g, pos + 1u)) {
        DS4_GLM_FT_FAIL("compact cache ensure");
        return false;
    }

    const bool decode_output_profile = false;
    const bool merge_indexed_output =
        logits_out != NULL && use_indexed_attention && !decode_output_profile;
    double decode_output_stage_t0 = decode_output_profile ? now_sec() : 0.0;
    const bool decode_flush_profile = false;
    uint32_t decode_flush_layer0 = 0;
    double decode_flush_stage_t0 = decode_flush_profile ? now_sec() : 0.0;

    const bool static_decode_map =
        !input_hc &&
        g->has_token_embd &&
        g->ssd_streaming &&
        metal_graph_stream_decode_static_map_enabled();
    const bool static_map_state_cache =
        static_decode_map && metal_graph_stream_decode_static_map_state_cache_enabled();
    const bool streaming_decode_sync_each_layer =
        g->ssd_streaming &&
        !static_decode_map &&
        glm_graph_streaming_decode_sync_each_layer();
    bool ok = true;
    if (input_hc) {
        ok = ds4_gpu_tensor_write(g->cur,
                                  0,
                                  input_hc,
                                  (uint64_t)DS4_N_EMBD * sizeof(float)) != 0;
    } else if (static_decode_map) {
        if (!static_map_state_cache || !g->streaming_static_decode_map_current) {
            ok = metal_graph_stream_map_decode_static_all(model, weights);
            if (ok) g->streaming_static_decode_map_current = static_map_state_cache;
        }
    } else {
        ok = glm_graph_stream_map_token(g, model, weights);
    }
    if (ok) ok = glm_graph_begin_commands_if_needed();
    if (ok && !input_hc) {
        if (g->placement) {
            ok = glm_graph_ws_switch(g, g->placement[0], false);
        }
    }
    if (ok && !input_hc) {
        ok = ds4_gpu_embed_token_quant_tensor(g->cur,
                                              model->map,
                                              model->size,
                                              weights->token_embd->abs_offset,
                                              weights->token_embd->type,
                                              DS4_N_VOCAB,
                                              (uint32_t)token,
                                              DS4_N_EMBD) != 0;
    }
    if (ok && streaming_decode_sync_each_layer) {
        ok = ds4_gpu_end_commands() != 0;
    }
    const uint32_t indexer_top_k = glm_graph_indexer_top_k_limit();
    ds4_gpu_tensor *last_indexer_selected = NULL;
    uint32_t last_indexer_selected_count = 0;
#define DS4_GLM_PROFILE_DECODE_STAGE(part_, name_) do { \
        if (ok && decode_stage_profile) { \
            ok = metal_graph_layer_stage_profile_boundary((part_), (name_), il, pos, 1, &decode_stage_t0); \
        } \
    } while (0)
    uint32_t glm_ft_fail_il = UINT32_MAX;
    for (uint32_t il = g->layer_start; ok && il <= g->layer_end; il++) {
        if (g->placement) {
            ok = glm_graph_ws_switch(g, g->placement[il + 1u], true);
            if (!ok) break;
        }
        glm_ft_fail_il = il;
        const uint32_t slice_layer_done = il - g->layer_start + 1u;
        if (g->ssd_streaming) {
            if (!static_decode_map) {
                ok = glm_graph_stream_map_decode_layer(g, model, weights, il);
            }
            if (ok) ok = glm_graph_begin_commands_if_needed();
        }
        const ds4_layer_weights *l = &weights->layer[il];
        const uint32_t kv_raw_dim = (uint32_t)l->attn_kv_a_mqa->dim[1];
        const float rope_base = layer_rope_freq_base(il);
        const float rope_scale = layer_rope_freq_scale(il);
        const bool decode_stage_profile = metal_graph_decode_stage_profile_enabled(il);
        double decode_stage_t0 = decode_stage_profile ? now_sec() : 0.0;
        if (decode_stage_profile) {
            ok = metal_graph_layer_stage_profile_boundary("glm_decode_attn",
                                                          NULL,
                                                          il,
                                                          pos,
                                                          1,
                                                          &decode_stage_t0);
        }

        if (ok) ok = ds4_gpu_rms_norm_weight_tensor(g->attn_norm,
                                                    g->cur,
                                                    model->map,
                                                    model->size,
                                                    l->attn_norm->abs_offset,
                                                    DS4_N_EMBD,
                                                    DS4_RMS_EPS) != 0;
        DS4_GLM_PROFILE_DECODE_STAGE("glm_decode_attn", "attn_norm");
        const uint32_t decode_ablate = glm_decode_ablate_mask();
        if (ok && !(decode_ablate & DS4_GLM_ABLATE_QPATH)) {
            ok = glm_graph_matmul_q8_0_decode_profiled_tensor(g->q_rank,
                                                              model,
                                                              l->attn_q_a->abs_offset,
                                                              DS4_N_EMBD,
                                                              DS4_N_LORA_Q,
                                                              g->attn_norm,
                                                              il,
                                                              pos,
                                                              "attn_q_a",
                                                              g->ssd_streaming) != 0;
        }
        const bool fuse_qkv_norm_store = use_indexed_attention &&
                                         !decode_stage_profile &&
                                         g->compact_cache_cap != 0;
        const bool fuse_qkv_norm = !decode_stage_profile && !fuse_qkv_norm_store;
        if (ok && fuse_qkv_norm_store) {
            ok = glm_graph_matmul_q8_0_decode_profiled_tensor(g->kv_raw,
                                                              model,
                                                              l->attn_kv_a_mqa->abs_offset,
                                                              DS4_N_EMBD,
                                                              kv_raw_dim,
                                                              g->attn_norm,
                                                              il,
                                                              pos,
                                                              "attn_kv_a_store",
                                                              g->ssd_streaming) != 0;
            if (ok) {
                ok = ds4_gpu_glm_qkv_norm_store_compact_kv_tensor(
                        g->q_rank_norm,
                        g->q_rank,
                        model->map,
                        model->size,
                        l->attn_q_a_norm->abs_offset,
                        DS4_N_LORA_Q,
                        g->layer_kv_lora_cache[il],
                        g->layer_k_rope_cache[il],
                        g->kv_raw,
                        l->attn_kv_a_norm->abs_offset,
                        pos,
                        1,
                        g->compact_cache_cap,
                        kv_raw_dim,
                        DS4_N_KV_LORA,
                        DS4_N_ROT,
                        glm_graph_compact_cache_is_f16(),
                        DS4_RMS_EPS) != 0;
            }
        } else if (ok && fuse_qkv_norm) {
            ok = glm_graph_matmul_q8_0_decode_profiled_tensor(g->kv_raw,
                                                              model,
                                                              l->attn_kv_a_mqa->abs_offset,
                                                              DS4_N_EMBD,
                                                              kv_raw_dim,
                                                              g->attn_norm,
                                                              il,
                                                              pos,
                                                              "attn_kv_a_norm",
                                                              g->ssd_streaming) != 0;
            if (ok) {
                ok = ds4_gpu_dsv4_qkv_rms_norm_rows_tensor(g->q_rank_norm,
                                                           g->q_rank,
                                                           model->map,
                                                           model->size,
                                                           l->attn_q_a_norm->abs_offset,
                                                           DS4_N_LORA_Q,
                                                           g->kv_norm,
                                                           g->kv_raw,
                                                           l->attn_kv_a_norm->abs_offset,
                                                           DS4_N_KV_LORA,
                                                           1,
                                                           DS4_RMS_EPS) != 0;
            }
        } else if (ok) {
            ok = ds4_gpu_rms_norm_weight_tensor(g->q_rank_norm,
                                                g->q_rank,
                                                model->map,
                                                model->size,
                                                l->attn_q_a_norm->abs_offset,
                                                DS4_N_LORA_Q,
                                                DS4_RMS_EPS) != 0;
        }
        if (ok && !(decode_ablate & DS4_GLM_ABLATE_QPATH)) {
            ok = glm_graph_matmul_q8_0_decode_profiled_tensor(g->q,
                                                              model,
                                                              l->attn_q_b->abs_offset,
                                                              DS4_N_LORA_Q,
                                                              g->q_dim,
                                                              g->q_rank_norm,
                                                              il,
                                                              pos,
                                                              "attn_q_b",
                                                              g->ssd_streaming) != 0;
            if (ok) ok = ds4_gpu_glm_rope_tail_tensor(g->q,
                                                      1,
                                                      DS4_N_HEAD,
                                                      DS4_N_KEY_MLA,
                                                      DS4_N_ROT,
                                                      pos,
                                                      0,
                                                      rope_base,
                                                      rope_scale,
                                                      0.0f,
                                                      1.0f,
                                                      DS4_ROPE_YARN_BETA_FAST,
                                                      DS4_ROPE_YARN_BETA_SLOW) != 0;
        }
        DS4_GLM_PROFILE_DECODE_STAGE("glm_decode_attn", "q_path");
        if (ok) metal_graph_debug_dump_tensor("glm_decode_q",
                                              g->q,
                                              (uint64_t)DS4_N_HEAD * DS4_N_KEY_MLA,
                                              il,
                                              pos);
        if (ok && g->compact_cache_cap != 0 && glm_graph_layer_uses_full_indexer(il) &&
            !(decode_ablate & DS4_GLM_ABLATE_INDEXER)) {
            ok = glm_graph_matmul_q8_0_decode_profiled_tensor(g->indexer_k,
                                                              model,
                                                              l->indexer_attn_k->abs_offset,
                                                              DS4_N_EMBD,
                                                              DS4_N_INDEXER_HEAD_DIM,
                                                              g->cur,
                                                              il,
                                                              pos,
                                                              "indexer_k",
                                                              g->ssd_streaming) != 0;
            DS4_GLM_PROFILE_DECODE_STAGE("glm_decode_attn", "indexer_k_proj");
            if (ok) {
                ok = ds4_gpu_glm_store_indexer_k_tensor(
                        g->layer_indexer_key_cache[il],
                        g->indexer_k,
                        model->map,
                        model->size,
                        l->indexer_k_norm->abs_offset,
                        l->indexer_k_norm_b->abs_offset,
                        pos,
                        1,
                        g->compact_cache_cap,
                        DS4_N_INDEXER_HEAD_DIM,
                        DS4_N_ROT,
                        0,
                        1.0e-6f,
                        rope_base,
                        rope_scale,
                        0.0f,
                        1.0f,
                        DS4_ROPE_YARN_BETA_FAST,
                        DS4_ROPE_YARN_BETA_SLOW,
                        glm_graph_compact_cache_is_f16()) != 0;
            }
            DS4_GLM_PROFILE_DECODE_STAGE("glm_decode_attn", "indexer_k_store");
        }
        DS4_GLM_PROFILE_DECODE_STAGE("glm_decode_attn", "indexer_k");
        if (ok && !fuse_qkv_norm && !fuse_qkv_norm_store) {
            ok = glm_graph_matmul_q8_0_decode_profiled_tensor(g->kv_raw,
                                                              model,
                                                              l->attn_kv_a_mqa->abs_offset,
                                                              DS4_N_EMBD,
                                                              kv_raw_dim,
                                                              g->attn_norm,
                                                              il,
                                                              pos,
                                                              "attn_kv_a",
                                                              g->ssd_streaming) != 0;
            if (ok) ok = ds4_gpu_glm_kv_lora_rms_norm_tensor(g->kv_norm,
                                                             g->kv_raw,
                                                             model->map,
                                                             model->size,
                                                             l->attn_kv_a_norm->abs_offset,
                                                             1,
                                                             kv_raw_dim,
                                                             DS4_N_KV_LORA,
                                                             DS4_RMS_EPS) != 0;
        }
        if (ok && g->compact_cache_cap != 0 && !fuse_qkv_norm_store) {
            ok = ds4_gpu_glm_store_compact_kv_tensor(g->layer_kv_lora_cache[il],
                                                     g->layer_k_rope_cache[il],
                                                     g->kv_norm,
                                                     g->kv_raw,
                                                     pos,
                                                     1,
                                                     g->compact_cache_cap,
                                                     kv_raw_dim,
                                                     DS4_N_KV_LORA,
                                                     DS4_N_ROT,
                                                     glm_graph_compact_cache_is_f16()) != 0;
        }
        if (use_indexed_attention) {
            if (ok && glm_graph_layer_uses_full_indexer(il)) {
                const uint32_t visible = pos + 1u;
                if (ok && visible <= indexer_top_k) {
                    ok = ds4_gpu_glm_fill_selected_range_tensor(g->indexer_selected,
                                                                visible) != 0;
                    DS4_GLM_PROFILE_DECODE_STAGE("glm_decode_attn", "indexer_fill");
                    last_indexer_selected_count = visible;
                } else if (ok && (decode_ablate & DS4_GLM_ABLATE_INDEXER)) {
                    /* Ablation: valid selected ids without the score/topk
                     * chain, so downstream attention timing stays real. */
                    ok = ds4_gpu_glm_fill_selected_range_tensor(g->indexer_selected,
                                                                indexer_top_k) != 0;
                    last_indexer_selected_count = indexer_top_k;
                } else if (ok) {
                    ok = glm_graph_matmul_q8_0_decode_profiled_tensor(g->indexer_q,
                                                                      model,
                                                                      l->indexer_attn_q_b->abs_offset,
                                                                      DS4_N_LORA_Q,
                                                                      (uint64_t)DS4_N_INDEXER_HEAD * DS4_N_INDEXER_HEAD_DIM,
                                                                      g->q_rank_norm,
                                                                      il,
                                                                      pos,
                                                                      "indexer_q",
                                                                      g->ssd_streaming) != 0;
                    if (ok) ok = ds4_gpu_glm_indexer_rope_tail_tensor(g->indexer_q,
                                                                      1,
                                                                      DS4_N_INDEXER_HEAD,
                                                                      DS4_N_INDEXER_HEAD_DIM,
                                                                      DS4_N_ROT,
                                                                      pos,
                                                                      0,
                                                                      rope_base,
                                                                      rope_scale,
                                                                      0.0f,
                                                                      1.0f,
                                                                      DS4_ROPE_YARN_BETA_FAST,
                                                                      DS4_ROPE_YARN_BETA_SLOW) != 0;
                    if (ok && glm_graph_indexer_qat()) {
                        ok = ds4_gpu_dsv4_indexer_qat_tensor(g->indexer_q,
                                                             DS4_N_INDEXER_HEAD,
                                                             DS4_N_INDEXER_HEAD_DIM) != 0;
                    }
                    DS4_GLM_PROFILE_DECODE_STAGE("glm_decode_attn", "indexer_q");
                    if (ok) ok = ds4_gpu_matmul_f32_tensor(g->indexer_weights,
                                                           model->map,
                                                           model->size,
                                                           l->indexer_proj->abs_offset,
                                                           DS4_N_EMBD,
                                                           DS4_N_INDEXER_HEAD,
                                                           g->cur,
                                                           1) != 0;
                    DS4_GLM_PROFILE_DECODE_STAGE("glm_decode_attn", "indexer_weights");
                    const float indexer_scale =
                        1.0f / sqrtf((float)(DS4_N_INDEXER_HEAD_DIM * DS4_N_INDEXER_HEAD));
                    ok = ds4_gpu_glm_indexer_score_one_tensor(g->indexer_scores,
                                                              g->indexer_q,
                                                              g->indexer_weights,
                                                              g->layer_indexer_key_cache[il],
                                                              visible,
                                                              DS4_N_INDEXER_HEAD,
                                                              DS4_N_INDEXER_HEAD_DIM,
                                                              indexer_scale,
                                                              glm_graph_compact_cache_is_f16()) != 0;
                    DS4_GLM_PROFILE_DECODE_STAGE("glm_decode_attn", "indexer_scores");
                    if (ok) ok = ds4_gpu_indexer_topk_tensor(g->indexer_selected,
                                                             g->indexer_scores,
                                                             visible,
                                                             1,
                                                             indexer_top_k) != 0;
                    DS4_GLM_PROFILE_DECODE_STAGE("glm_decode_attn", "indexer_topk");
                    last_indexer_selected_count = indexer_top_k;
                }
                if (ok) last_indexer_selected = g->indexer_selected;
            } else if (ok && (!last_indexer_selected || last_indexer_selected_count == 0)) {
                ok = false;
            }
            DS4_GLM_PROFILE_DECODE_STAGE("glm_decode_attn", "indexer_select");
            if (ok && !(decode_ablate & (DS4_GLM_ABLATE_ATTN_CORE | DS4_GLM_ABLATE_QKLOW))) {
                ok = ds4_gpu_glm_qk_lowrank_typed_tensor(g->qk_low,
                                                         g->q,
                                                         model->map,
                                                         model->size,
                                                         l->attn_k_b->abs_offset,
                                                         l->attn_k_b->type,
                                                         DS4_N_HEAD,
                                                         DS4_N_KV_LORA,
                                                         (uint32_t)g->q_nope,
                                                         DS4_N_KEY_MLA) != 0;
                if (ok) metal_graph_debug_dump_tensor("glm_decode_qk_low",
                                                      g->qk_low,
                                                      (uint64_t)DS4_N_HEAD * DS4_N_KV_LORA,
                                                      il,
                                                      pos);
            }
            DS4_GLM_PROFILE_DECODE_STAGE("glm_decode_attn", "kv_path");
            if (ok && (decode_ablate & DS4_GLM_ABLATE_ATTN_CORE)) {
                /* Skip the indexed attention kernels; zero heads so the
                 * rest of the layer stays finite (timing-only). */
                ok = ds4_gpu_tensor_fill_f32(g->heads, 0.0f,
                                             (uint64_t)g->heads_dim) != 0;
            } else if (ok && glm_graph_indexed_decode_split_group8_available(last_indexer_selected_count)) {
                const uint32_t split_block_rows =
                    glm_graph_indexed_decode_split_block_rows_for(last_indexer_selected_count);
                const uint32_t split_blocks =
                    (last_indexer_selected_count + split_block_rows - 1u) / split_block_rows;
                ok = ds4_gpu_glm_attention_indexed_decode_split_group8_typed_tensor(g->heads,
                                                                                    g->attn_partial_lora,
                                                                                    g->attn_partial_ms,
                                                                                    g->q,
                                                                                    g->qk_low,
                                                                                    g->layer_kv_lora_cache[il],
                                                                                    g->layer_k_rope_cache[il],
                                                                                    model->map,
                                                                                    model->size,
                                                                                    l->attn_v_b->abs_offset,
                                                                                    l->attn_v_b->type,
                                                                                    last_indexer_selected,
                                                                                    last_indexer_selected_count,
                                                                                    true,
                                                                                    g->compact_cache_cap,
                                                                                    glm_graph_compact_cache_is_f16(),
                                                                                    DS4_N_HEAD,
                                                                                    DS4_N_KV_LORA,
                                                                                    (uint32_t)g->q_nope,
                                                                                    DS4_N_ROT,
                                                                                    DS4_N_VALUE_MLA,
                                                                                    0,
                                                                                    split_block_rows,
                                                                                    split_blocks,
                                                                                    rope_base,
                                                                                    rope_scale,
                                                                                    0.0f,
                                                                                    1.0f,
                                                                                    DS4_ROPE_YARN_BETA_FAST,
                                                                                    DS4_ROPE_YARN_BETA_SLOW) != 0;
            } else if (ok) {
                ok = ds4_gpu_glm_attention_indexed_decode_typed_tensor(g->heads,
                                                                       g->q,
                                                                       g->qk_low,
                                                                       g->layer_kv_lora_cache[il],
                                                                       g->layer_k_rope_cache[il],
                                                                       model->map,
                                                                       model->size,
                                                                       l->attn_v_b->abs_offset,
                                                                       l->attn_v_b->type,
                                                                       last_indexer_selected,
                                                                       last_indexer_selected_count,
                                                                       g->compact_cache_cap,
                                                                       glm_graph_compact_cache_is_f16(),
                                                                       DS4_N_HEAD,
                                                                       DS4_N_KV_LORA,
                                                                       (uint32_t)g->q_nope,
                                                                       DS4_N_ROT,
                                                                       DS4_N_VALUE_MLA,
                                                                       0,
                                                                       rope_base,
                                                                       rope_scale,
                                                                       0.0f,
                                                                       1.0f,
                                                                       DS4_ROPE_YARN_BETA_FAST,
                                                                       DS4_ROPE_YARN_BETA_SLOW) != 0;
            }
        } else {
            if (ok) ok = ds4_gpu_glm_k_b_project_typed_tensor(g->k_nope,
                                                              g->kv_norm,
                                                              model->map,
                                                              model->size,
                                                              l->attn_k_b->abs_offset,
                                                              l->attn_k_b->type,
                                                              1,
                                                              DS4_N_KV_LORA,
                                                              (uint32_t)g->q_nope,
                                                              DS4_N_HEAD) != 0;
            if (ok) ok = glm_graph_matmul_q8_0_decode_profiled_tensor(g->value,
                                                                      model,
                                                                      l->attn_v_b->abs_offset,
                                                                      DS4_N_KV_LORA,
                                                                      g->heads_dim,
                                                                      g->kv_norm,
                                                                      il,
                                                                      pos,
                                                                      "attn_v_b",
                                                                      g->ssd_streaming) != 0;
            DS4_GLM_PROFILE_DECODE_STAGE("glm_decode_attn", "kv_path");
            if (ok) ok = ds4_gpu_glm_build_kv_cache_tensor(g->layer_key_cache[il],
                                                           g->layer_value_cache[il],
                                                           g->kv_raw,
                                                           g->k_nope,
                                                           g->value,
                                                           pos,
                                                           1,
                                                           g->ctx_cap,
                                                           DS4_N_HEAD,
                                                           kv_raw_dim,
                                                           DS4_N_KV_LORA,
                                                           (uint32_t)g->q_nope,
                                                           DS4_N_ROT,
                                                           DS4_N_VALUE_MLA,
                                                           0,
                                                           rope_base,
                                                           rope_scale,
                                                           0.0f,
                                                           1.0f,
                                                           DS4_ROPE_YARN_BETA_FAST,
                                                           DS4_ROPE_YARN_BETA_SLOW,
                                                           true) != 0;
            DS4_GLM_PROFILE_DECODE_STAGE("glm_decode_attn", "kv_cache");
            if (ok) ok = ds4_gpu_glm_attention_full_tensor(g->heads,
                                                           g->q,
                                                           g->layer_key_cache[il],
                                                           g->layer_value_cache[il],
                                                           pos,
                                                           1,
                                                           pos + 1u,
                                                           g->ctx_cap,
                                                           DS4_N_HEAD,
                                                           DS4_N_KEY_MLA,
                                                           DS4_N_VALUE_MLA,
                                                           true) != 0;
        }
        DS4_GLM_PROFILE_DECODE_STAGE("glm_decode_attn", "attention");
        if (ok) metal_graph_debug_dump_tensor("glm_decode_heads",
                                              g->heads,
                                              g->heads_dim,
                                              il,
                                              pos);
        if (ok && !(decode_ablate & DS4_GLM_ABLATE_ATTN_OUT)) {
            ok = glm_graph_matmul_q8_0_decode_profiled_tensor(g->attn_out,
                                                              model,
                                                              l->attn_output->abs_offset,
                                                              g->heads_dim,
                                                              DS4_N_EMBD,
                                                              g->heads,
                                                              il,
                                                              pos,
                                                              "attn_o",
                                                              g->ssd_streaming) != 0;
        }
        DS4_GLM_PROFILE_DECODE_STAGE("glm_decode_attn", "attn_output");
        if (ok) ok = ds4_gpu_add_rms_norm_weight_tensor(g->ffn_norm,
                                                        g->after_attn,
                                                        g->cur,
                                                        g->attn_out,
                                                        model->map,
                                                        model->size,
                                                        l->ffn_norm->abs_offset,
                                                        DS4_N_EMBD,
                                                        DS4_RMS_EPS) != 0;
        DS4_GLM_PROFILE_DECODE_STAGE("glm_decode_ffn", "ffn_norm");
        if (ok) ok = glm_graph_encode_ffn_one_normed_from(g,
                                                          model,
                                                          l,
                                                          il,
                                                          pos,
                                                          g->ffn_norm,
                                                          g->after_attn,
                                                          g->next,
                                                          g->ffn_gate,
                                                          g->ffn_up,
                                                          g->ffn_mid,
                                                          g->ffn_out,
                                                          g->ffn_sum,
                                                          g->attn_out,
                                                          decode_stage_profile,
                                                          decode_stage_profile ? &decode_stage_t0 : NULL);
        if (ok) {
            ds4_gpu_tensor *tmp = g->cur;
            g->cur = g->next;
            g->next = tmp;
        }
        if (ok && glm_debug_hidden_dump_layer_match(il)) {
            ok = ds4_gpu_end_commands() != 0;
            if (ok) {
                glm_debug_dump_hidden_layer(g->cur, 0, il, pos);
                glm_debug_dump_raw_layer(g->router_selected, "sel",
                                         (uint64_t)DS4_N_EXPERT_USED * sizeof(int32_t),
                                         il, (int)pos);
                glm_debug_dump_raw_layer(g->router_weights, "selw",
                                         (uint64_t)DS4_N_EXPERT_USED * sizeof(float),
                                         il, (int)pos);
                ok = ds4_gpu_begin_commands() != 0;
            }
        }
        if (ok &&
            !g->ssd_streaming &&
            decode_layer_flush_interval != 0 &&
            il < g->layer_end &&
            (slice_layer_done % decode_layer_flush_interval) == 0) {
            if (decode_flush_profile) {
                ok = ds4_gpu_flush_commands() != 0;
                if (ok) ok = ds4_gpu_synchronize() != 0;
                if (ok) {
                    const double now = now_sec();
                    fprintf(stderr,
                            "ds4: GLM decode layer flush pos=%u layers=%u..%u %.3f ms\n",
                            pos,
                            decode_flush_layer0,
                            il,
                            (now - decode_flush_stage_t0) * 1000.0);
                    decode_flush_layer0 = il + 1u;
                    decode_flush_stage_t0 = now;
                    ok = ds4_gpu_begin_commands() != 0;
                }
            } else {
                ok = ds4_gpu_flush_commands() != 0;
            }
        }
        if (streaming_decode_sync_each_layer) {
            if (ok) ok = ds4_gpu_end_commands() != 0;
            else (void)ds4_gpu_synchronize();
        }
    }
#undef DS4_GLM_PROFILE_DECODE_STAGE
    if (ok && (merge_indexed_output ||
               (defer_completion && logits_out != NULL))) {
        if (g->ssd_streaming) {
            if (!static_decode_map) {
                ok = glm_graph_stream_map_output(g, model, weights);
            }
            if (ok) ok = glm_graph_begin_commands_if_needed();
        }
        ok = glm_graph_encode_output_head(g, model, weights);
        if (g->ssd_streaming) {
            if (ok) ok = glm_graph_end_commands_if_active();
            else (void)ds4_gpu_synchronize();
        }
    }
    if (!g->ssd_streaming && !defer_completion) {
        if (ok) ok = ds4_gpu_end_commands() != 0;
        else (void)ds4_gpu_synchronize();
    } else if (!ok) {
        (void)ds4_gpu_synchronize();
    }
    if (decode_output_profile) {
        const double now = now_sec();
        fprintf(stderr,
                "ds4: GLM decode output profile pos=%u layers=%.3f ms\n",
                pos,
                (now - decode_output_stage_t0) * 1000.0);
        decode_output_stage_t0 = now;
    }
    if (ok && output_hc && !defer_completion) {
        ok = ds4_gpu_tensor_read(g->cur,
                                 0,
                                 output_hc,
                                 (uint64_t)DS4_N_EMBD * sizeof(float)) != 0;
    }
    if (ok && logits_out && !defer_completion) {
        if (use_indexed_attention) {
            if (!merge_indexed_output) {
                if (g->ssd_streaming && !static_decode_map) {
                    ok = glm_graph_stream_map_output(g, model, weights);
                }
                if (ok) ok = glm_graph_begin_commands_if_needed();
                if (ok) ok = glm_graph_encode_output_head(g, model, weights);
                if (ok) ok = glm_graph_end_commands_if_active();
                else (void)ds4_gpu_synchronize();
                if (decode_output_profile) {
                    const double now = now_sec();
                    fprintf(stderr,
                            "ds4: GLM decode output profile pos=%u output_head=%.3f ms\n",
                            pos,
                            (now - decode_output_stage_t0) * 1000.0);
                    decode_output_stage_t0 = now;
                }
            }
            if (ok) {
                if (glm_debug_hidden_dump_layer() < 0)
                    glm_debug_dump_hidden_row(g->cur, 0);
                ok = ds4_gpu_tensor_read(g->logits,
                                         0,
                                         logits_out,
                                         (uint64_t)DS4_N_VOCAB * sizeof(float)) != 0;
                if (decode_output_profile) {
                    const double now = now_sec();
                    fprintf(stderr,
                            "ds4: GLM decode output profile pos=%u logits_read=%.3f ms\n",
                            pos,
                            (now - decode_output_stage_t0) * 1000.0);
                }
            }
        } else {
            if (g->ssd_streaming && !static_decode_map) {
                ok = glm_graph_stream_map_output(g, model, weights);
            }
            if (ok && ds4_gpu_commands_active()) {
                ok = ds4_gpu_end_commands() != 0;
            }
            if (ok) ok = glm_graph_forward_output_head(g, model, weights, g->cur, logits_out);
            if (decode_output_profile) {
                const double now = now_sec();
                fprintf(stderr,
                        "ds4: GLM decode output profile pos=%u fallback_output=%.3f ms\n",
                        pos,
                        (now - decode_output_stage_t0) * 1000.0);
            }
        }
    }
    if (ok &&
        !logits_out &&
        !output_hc &&
        g->ssd_streaming &&
        !streaming_decode_sync_each_layer) {
        ok = ds4_gpu_end_commands() != 0;
    } else if (ok && !logits_out && g->ssd_streaming) {
        ok = glm_graph_end_commands_if_active();
    } else if (!ok) {
        (void)ds4_gpu_synchronize();
    }
    if (!ok && getenv("DS4_GLM_TP_DEBUG")) {
        fprintf(stderr,
                "ds4: glm forward_token fail pos=%u around layer %u\n",
                pos, glm_ft_fail_il);
    }
    (void)glm_ft_fail_il;
    return ok;
#undef DS4_GLM_FT_FAIL
}

static int glm_metal_first_token_logits(
        const ds4_model   *model,
        const ds4_weights *weights,
        int                token,
        float             *logits_out) {
    if (!model || !weights || !logits_out) return 1;
    if (token < 0 || token >= (int)DS4_N_VOCAB) {
        fprintf(stderr, "ds4: GLM token %d is outside vocab\n", token);
        return 1;
    }
    if (!weights->token_embd || weights->token_embd->type != DS4_TENSOR_Q8_0 ||
        !weights->output_norm || weights->output_norm->type != DS4_TENSOR_F32 ||
        !weights->output || weights->output->type != DS4_TENSOR_Q8_0 ||
        weights->output_norm->dim[0] != DS4_N_EMBD ||
        weights->output->dim[0] != DS4_N_EMBD ||
        weights->output->dim[1] != DS4_N_VOCAB) {
        fprintf(stderr, "ds4: GLM Metal first-token path found unexpected embedding/output layout\n");
        return 1;
    }
    if (DS4_N_LAYER <= DS4_N_NEXTN_PREDICT) {
        fprintf(stderr, "ds4: GLM Metal first-token path has no normal transformer layers\n");
        return 1;
    }

    const uint32_t normal_layers = DS4_N_LAYER - DS4_N_NEXTN_PREDICT;
    const uint64_t heads_dim = (uint64_t)DS4_N_HEAD * DS4_N_VALUE_MLA;
    uint64_t kv_raw_dim = 0;
    uint64_t dense_hidden_max = DS4_N_FF_EXP;
    bool generic_routed_moe = false;
    for (uint32_t il = 0; il < normal_layers; il++) {
        const ds4_layer_weights *l = &weights->layer[il];
        if (l->attn_kv_a_mqa && l->attn_kv_a_mqa->dim[1] > kv_raw_dim) {
            kv_raw_dim = l->attn_kv_a_mqa->dim[1];
        }
        if (il < DS4_N_LEADING_DENSE && l->ffn_gate &&
            l->ffn_gate->dim[1] > dense_hidden_max) {
            dense_hidden_max = l->ffn_gate->dim[1];
        }
        if (glm_graph_layer_uses_generic_routed_moe(l)) generic_routed_moe = true;
    }
    if (kv_raw_dim < DS4_N_KV_LORA) {
        fprintf(stderr, "ds4: GLM Metal first-token path found no valid KV projection\n");
        return 1;
    }

    const uint64_t emb_bytes = (uint64_t)DS4_N_EMBD * sizeof(float);
    const uint64_t sparse_mid_elems = (uint64_t)DS4_N_EXPERT_USED * DS4_N_FF_EXP;
    const uint64_t ffn_mid_elems =
        dense_hidden_max > sparse_mid_elems ? dense_hidden_max : sparse_mid_elems;
    const uint64_t routed_mid_bytes =
        (uint64_t)DS4_N_EXPERT_USED * DS4_N_FF_EXP * sizeof(float);
    const uint64_t routed_down_bytes =
        (uint64_t)DS4_N_EXPERT_USED * DS4_N_EMBD * sizeof(float);
    const uint64_t logits_bytes = (uint64_t)DS4_N_VOCAB * sizeof(float);

    ds4_gpu_tensor *cur = NULL;
    ds4_gpu_tensor *attn_norm = NULL;
    ds4_gpu_tensor *kv_raw = NULL;
    ds4_gpu_tensor *kv_norm = NULL;
    ds4_gpu_tensor *heads = NULL;
    ds4_gpu_tensor *attn_out = NULL;
    ds4_gpu_tensor *after_attn = NULL;
    ds4_gpu_tensor *ffn_norm = NULL;
    ds4_gpu_tensor *ffn_gate = NULL;
    ds4_gpu_tensor *ffn_up = NULL;
    ds4_gpu_tensor *ffn_mid = NULL;
    ds4_gpu_tensor *routed_gate = NULL;
    ds4_gpu_tensor *routed_up = NULL;
    ds4_gpu_tensor *routed_down = NULL;
    ds4_gpu_tensor *ffn_out = NULL;
    ds4_gpu_tensor *ffn_sum = NULL;
    ds4_gpu_tensor *next = NULL;
    ds4_gpu_tensor *router_logits = NULL;
    ds4_gpu_tensor *router_probs = NULL;
    ds4_gpu_tensor *router_selected = NULL;
    ds4_gpu_tensor *router_weights = NULL;
    ds4_gpu_tensor *logits = NULL;

    int ok = 1;
#define DS4_GLM_FIRST_ALLOC_TENSOR(var, bytes_) \
    do { \
        (var) = ds4_gpu_tensor_alloc((bytes_)); \
        if (!(var)) { \
            fprintf(stderr, "ds4: GLM Metal first-token path could not allocate %s\n", #var); \
            ok = 0; \
        } \
    } while (0)

    DS4_GLM_FIRST_ALLOC_TENSOR(cur, emb_bytes);
    DS4_GLM_FIRST_ALLOC_TENSOR(attn_norm, emb_bytes);
    DS4_GLM_FIRST_ALLOC_TENSOR(kv_raw, kv_raw_dim * sizeof(float));
    DS4_GLM_FIRST_ALLOC_TENSOR(kv_norm, (uint64_t)DS4_N_KV_LORA * sizeof(float));
    DS4_GLM_FIRST_ALLOC_TENSOR(heads, heads_dim * sizeof(float));
    DS4_GLM_FIRST_ALLOC_TENSOR(attn_out, emb_bytes);
    DS4_GLM_FIRST_ALLOC_TENSOR(after_attn, emb_bytes);
    DS4_GLM_FIRST_ALLOC_TENSOR(ffn_norm, emb_bytes);
    DS4_GLM_FIRST_ALLOC_TENSOR(ffn_gate, dense_hidden_max * sizeof(float));
    DS4_GLM_FIRST_ALLOC_TENSOR(ffn_up, dense_hidden_max * sizeof(float));
    DS4_GLM_FIRST_ALLOC_TENSOR(ffn_mid, ffn_mid_elems * sizeof(float));
    if (generic_routed_moe) {
        DS4_GLM_FIRST_ALLOC_TENSOR(routed_gate, routed_mid_bytes);
        DS4_GLM_FIRST_ALLOC_TENSOR(routed_up, routed_mid_bytes);
        DS4_GLM_FIRST_ALLOC_TENSOR(routed_down, routed_down_bytes);
    }
    DS4_GLM_FIRST_ALLOC_TENSOR(ffn_out, emb_bytes);
    DS4_GLM_FIRST_ALLOC_TENSOR(ffn_sum, emb_bytes);
    DS4_GLM_FIRST_ALLOC_TENSOR(next, emb_bytes);
    DS4_GLM_FIRST_ALLOC_TENSOR(router_logits, (uint64_t)DS4_N_EXPERT * sizeof(float));
    DS4_GLM_FIRST_ALLOC_TENSOR(router_probs, (uint64_t)DS4_N_EXPERT * sizeof(float));
    DS4_GLM_FIRST_ALLOC_TENSOR(router_selected, (uint64_t)DS4_N_EXPERT_USED * sizeof(int32_t));
    DS4_GLM_FIRST_ALLOC_TENSOR(router_weights, (uint64_t)DS4_N_EXPERT_USED * sizeof(float));
    DS4_GLM_FIRST_ALLOC_TENSOR(logits, logits_bytes);
#undef DS4_GLM_FIRST_ALLOC_TENSOR

    if (ok) {
        ok = ds4_gpu_embed_token_q8_0_tensor(cur,
                                             model->map,
                                             model->size,
                                             weights->token_embd->abs_offset,
                                             DS4_N_VOCAB,
                                             (uint32_t)token,
                                             DS4_N_EMBD);
    }
    for (uint32_t il = 0; ok && il < normal_layers; il++) {
        const ds4_layer_weights *gl = &weights->layer[il];
        const uint64_t gl_kv_raw_dim = gl->attn_kv_a_mqa ? gl->attn_kv_a_mqa->dim[1] : 0;
        if (!gl->attn_norm ||
            !gl->attn_kv_a_mqa ||
            !gl->attn_kv_a_norm ||
            !gl->attn_v_b ||
            !gl->attn_output ||
            !gl->ffn_norm ||
            gl_kv_raw_dim < DS4_N_KV_LORA ||
            gl_kv_raw_dim > kv_raw_dim ||
            gl->attn_kv_a_mqa->type != DS4_TENSOR_Q8_0 ||
            gl->attn_kv_a_mqa->dim[0] != DS4_N_EMBD ||
            gl->attn_v_b->type != DS4_TENSOR_Q8_0 ||
            gl->attn_v_b->dim[0] != DS4_N_KV_LORA ||
            gl->attn_v_b->dim[1] != DS4_N_VALUE_MLA ||
            gl->attn_v_b->dim[2] != DS4_N_HEAD ||
            gl->attn_output->type != DS4_TENSOR_Q8_0 ||
            gl->attn_output->dim[0] != heads_dim ||
            gl->attn_output->dim[1] != DS4_N_EMBD) {
            fprintf(stderr,
                    "ds4: GLM Metal first-token path found unexpected attention layout in layer %u\n",
                    il);
            ok = 0;
            break;
        }

        if (ok) ok = ds4_gpu_rms_norm_weight_tensor(attn_norm, cur,
                                                    model->map, model->size,
                                                    gl->attn_norm->abs_offset,
                                                    DS4_N_EMBD, DS4_RMS_EPS);
        if (ok) ok = ds4_gpu_matmul_q8_0_tensor(kv_raw,
                                                model->map,
                                                model->size,
                                                gl->attn_kv_a_mqa->abs_offset,
                                                DS4_N_EMBD,
                                                gl_kv_raw_dim,
                                                attn_norm,
                                                1);
        if (ok) ok = ds4_gpu_rms_norm_weight_tensor(kv_norm, kv_raw,
                                                    model->map, model->size,
                                                    gl->attn_kv_a_norm->abs_offset,
                                                    DS4_N_KV_LORA, DS4_RMS_EPS);
        if (ok) ok = ds4_gpu_matmul_q8_0_tensor(heads,
                                                model->map,
                                                model->size,
                                                gl->attn_v_b->abs_offset,
                                                DS4_N_KV_LORA,
                                                heads_dim,
                                                kv_norm,
                                                1);
        if (ok) ok = ds4_gpu_matmul_q8_0_tensor(attn_out,
                                                model->map,
                                                model->size,
                                                gl->attn_output->abs_offset,
                                                heads_dim,
                                                DS4_N_EMBD,
                                                heads,
                                                1);
        if (ok) ok = ds4_gpu_add_tensor(after_attn, cur, attn_out, DS4_N_EMBD);
        if (ok) ok = ds4_gpu_rms_norm_weight_tensor(ffn_norm, after_attn,
                                                    model->map, model->size,
                                                    gl->ffn_norm->abs_offset,
                                                    DS4_N_EMBD, DS4_RMS_EPS);
        if (il < DS4_N_LEADING_DENSE) {
            const uint64_t gl_ffn_hidden = gl->ffn_gate ? gl->ffn_gate->dim[1] : 0;
            if (!gl->ffn_gate ||
                !gl->ffn_up ||
                !gl->ffn_down ||
                gl->ffn_gate->type != DS4_TENSOR_Q8_0 ||
                gl->ffn_up->type != DS4_TENSOR_Q8_0 ||
                gl->ffn_down->type != DS4_TENSOR_Q8_0 ||
                gl->ffn_gate->dim[0] != DS4_N_EMBD ||
                gl->ffn_up->dim[0] != DS4_N_EMBD ||
                gl->ffn_up->dim[1] != gl_ffn_hidden ||
                gl->ffn_down->dim[0] != gl_ffn_hidden ||
                gl->ffn_down->dim[1] != DS4_N_EMBD ||
                gl_ffn_hidden > dense_hidden_max) {
                fprintf(stderr,
                        "ds4: GLM Metal first-token path found unexpected dense FFN layout in layer %u\n",
                        il);
                ok = 0;
                break;
            }
            if (ok) ok = ds4_gpu_matmul_q8_0_tensor(ffn_gate,
                                                    model->map,
                                                    model->size,
                                                    gl->ffn_gate->abs_offset,
                                                    DS4_N_EMBD,
                                                    gl_ffn_hidden,
                                                    ffn_norm,
                                                    1);
            if (ok) ok = ds4_gpu_matmul_q8_0_tensor(ffn_up,
                                                    model->map,
                                                    model->size,
                                                    gl->ffn_up->abs_offset,
                                                    DS4_N_EMBD,
                                                    gl_ffn_hidden,
                                                    ffn_norm,
                                                    1);
            if (ok) ok = ds4_gpu_swiglu_tensor(ffn_mid, ffn_gate, ffn_up,
                                               (uint32_t)gl_ffn_hidden, 0.0f, 1.0f);
            if (ok) ok = ds4_gpu_matmul_q8_0_tensor(ffn_out,
                                                    model->map,
                                                    model->size,
                                                    gl->ffn_down->abs_offset,
                                                    gl_ffn_hidden,
                                                    DS4_N_EMBD,
                                                    ffn_mid,
                                                    1);
            if (ok) ok = ds4_gpu_add_tensor(next, after_attn, ffn_out, DS4_N_EMBD);
        } else {
            const uint32_t gl_gate_type = gl->ffn_gate_exps ? gl->ffn_gate_exps->type : 0;
            const uint32_t gl_up_type = gl->ffn_up_exps ? gl->ffn_up_exps->type : 0;
            const bool gl_gate_pair_supported =
                glm_graph_gate_pair_type_supported(gl_gate_type, gl_up_type);
            uint64_t gate_in = 0, gate_out = 0, gate_row_bytes = 0;
            uint64_t up_in = 0, up_out = 0, up_row_bytes = 0;
            uint64_t down_in = 0, down_out = 0, down_row_bytes = 0;

            if (!gl->ffn_gate_inp ||
                !gl->ffn_exp_probs_b ||
                !gl->ffn_gate_exps ||
                !gl->ffn_up_exps ||
                !gl->ffn_down_exps ||
                !gl->ffn_gate_shexp ||
                !gl->ffn_up_shexp ||
                !gl->ffn_down_shexp ||
                gl->ffn_gate_inp->type != DS4_TENSOR_F32 ||
                gl->ffn_gate_inp->dim[0] != DS4_N_EMBD ||
                gl->ffn_gate_inp->dim[1] != DS4_N_EXPERT ||
                gl->ffn_exp_probs_b->type != DS4_TENSOR_F32 ||
                gl->ffn_exp_probs_b->dim[0] != DS4_N_EXPERT ||
                !gl_gate_pair_supported ||
                !glm_graph_down_type_supported(gl->ffn_down_exps->type) ||
                gl->ffn_gate_shexp->type != DS4_TENSOR_Q8_0 ||
                gl->ffn_up_shexp->type != DS4_TENSOR_Q8_0 ||
                gl->ffn_down_shexp->type != DS4_TENSOR_Q8_0 ||
                gl->ffn_gate_shexp->dim[0] != DS4_N_EMBD ||
                gl->ffn_gate_shexp->dim[1] != DS4_N_FF_EXP ||
                gl->ffn_up_shexp->dim[0] != DS4_N_EMBD ||
                gl->ffn_up_shexp->dim[1] != DS4_N_FF_EXP ||
                gl->ffn_down_shexp->dim[0] != DS4_N_FF_EXP ||
                gl->ffn_down_shexp->dim[1] != DS4_N_EMBD ||
                sparse_mid_elems > ffn_mid_elems) {
                fprintf(stderr,
                        "ds4: GLM Metal first-token path found unexpected sparse FFN layout in layer %u\n",
                        il);
                ok = 0;
                break;
            }

            (void)tensor_expert_bytes(model, gl->ffn_gate_exps, 0,
                                      &gate_in, &gate_out, &gate_row_bytes);
            (void)tensor_expert_bytes(model, gl->ffn_up_exps, 0,
                                      &up_in, &up_out, &up_row_bytes);
            (void)tensor_expert_bytes(model, gl->ffn_down_exps, 0,
                                      &down_in, &down_out, &down_row_bytes);
            if (gate_in != DS4_N_EMBD ||
                up_in != DS4_N_EMBD ||
                down_in != DS4_N_FF_EXP ||
                gate_out != DS4_N_FF_EXP ||
                up_out != DS4_N_FF_EXP ||
                down_out != DS4_N_EMBD) {
                fprintf(stderr,
                        "ds4: GLM Metal first-token path found unexpected expert strides in layer %u\n",
                        il);
                ok = 0;
                break;
            }

            if (ok) ok = ds4_gpu_matmul_f32_tensor(router_logits,
                                                   model->map,
                                                   model->size,
                                                   gl->ffn_gate_inp->abs_offset,
                                                   DS4_N_EMBD,
                                                   DS4_N_EXPERT,
                                                   ffn_norm,
                                                   1);
            if (ok) ok = ds4_gpu_glm_router_select_tensor(router_selected,
                                                          router_weights,
                                                          router_probs,
                                                          model->map,
                                                          model->size,
                                                          gl->ffn_exp_probs_b->abs_offset,
                                                          router_logits,
                                                          DS4_N_EXPERT,
                                                          DS4_N_EXPERT_USED,
                                                          DS4_EXPERT_WEIGHT_SCALE);
            if (ok) {
                const ds4_gpu_stream_expert_table table = {
                    .model_map = model->map,
                    .model_size = model->size,
                    .layer = il,
                    .n_total_expert = DS4_N_EXPERT,
                    .gate_offset = gl->ffn_gate_exps->abs_offset,
                    .up_offset = gl->ffn_up_exps->abs_offset,
                    .down_offset = gl->ffn_down_exps->abs_offset,
                    .gate_expert_bytes = gate_out * gate_row_bytes,
                    .down_expert_bytes = down_out * down_row_bytes,
                };
                ok = ds4_gpu_glm_stream_expert_cache_begin_selected_load_tensor(
                        &table,
                        router_selected,
                        DS4_N_EXPERT_USED) != 0;
            }
            ds4_glm_gpu_graph route_g = {
                .routed_gate = routed_gate,
                .routed_up = routed_up,
                .routed_down = routed_down,
                .ssd_streaming = false,
            };
            if (ok) ok = glm_graph_routed_moe_one_dispatch(
                    &route_g,
                    model,
                    gl,
                    il,
                    ffn_out,
                    ffn_mid,
                    gate_out * gate_row_bytes,
                    gate_row_bytes,
                    up_out * up_row_bytes,
                    up_row_bytes,
                    down_out * down_row_bytes,
                    down_row_bytes,
                    router_selected,
                    router_weights,
                    ffn_norm,
                    false);
            if (ok) ok = ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(
                    ffn_gate,
                    ffn_up,
                    ffn_mid,
                    model->map,
                    model->size,
                    gl->ffn_gate_shexp->abs_offset,
                    gl->ffn_up_shexp->abs_offset,
                    DS4_N_EMBD,
                    DS4_N_FF_EXP,
                    ffn_norm,
                    0.0f);
            if (ok) ok = ds4_gpu_matmul_q8_0_tensor(ffn_sum,
                                                    model->map,
                                                    model->size,
                                                    gl->ffn_down_shexp->abs_offset,
                                                    DS4_N_FF_EXP,
                                                    DS4_N_EMBD,
                                                    ffn_mid,
                                                    1);
            if (ok) ok = ds4_gpu_add_tensor(attn_out, ffn_out, ffn_sum, DS4_N_EMBD);
            if (ok) ok = ds4_gpu_add_tensor(next, after_attn, attn_out, DS4_N_EMBD);
        }

        if (ok) {
            ds4_gpu_tensor *tmp = cur;
            cur = next;
            next = tmp;
        }
    }
    if (ok) ok = ds4_gpu_rms_norm_weight_tensor(ffn_norm, cur,
                                                model->map, model->size,
                                                weights->output_norm->abs_offset,
                                                DS4_N_EMBD, DS4_RMS_EPS);
    if (ok) ok = ds4_gpu_matmul_q8_0_tensor(logits,
                                            model->map,
                                            model->size,
                                            weights->output->abs_offset,
                                            DS4_N_EMBD,
                                            DS4_N_VOCAB,
                                            ffn_norm,
                                            1);
    if (ok) ok = ds4_gpu_tensor_read(logits, 0, logits_out, logits_bytes) != 0;

    ds4_gpu_tensor_free(router_weights);
    ds4_gpu_tensor_free(router_selected);
    ds4_gpu_tensor_free(router_probs);
    ds4_gpu_tensor_free(router_logits);
    ds4_gpu_tensor_free(logits);
    ds4_gpu_tensor_free(next);
    ds4_gpu_tensor_free(ffn_sum);
    ds4_gpu_tensor_free(ffn_out);
    ds4_gpu_tensor_free(routed_down);
    ds4_gpu_tensor_free(routed_up);
    ds4_gpu_tensor_free(routed_gate);
    ds4_gpu_tensor_free(ffn_mid);
    ds4_gpu_tensor_free(ffn_up);
    ds4_gpu_tensor_free(ffn_gate);
    ds4_gpu_tensor_free(ffn_norm);
    ds4_gpu_tensor_free(after_attn);
    ds4_gpu_tensor_free(attn_out);
    ds4_gpu_tensor_free(heads);
    ds4_gpu_tensor_free(kv_norm);
    ds4_gpu_tensor_free(kv_raw);
    ds4_gpu_tensor_free(attn_norm);
    ds4_gpu_tensor_free(cur);
    return ok ? 0 : 1;
}

static DS4_MAYBE_UNUSED int generate_glm_metal_first_token(
        const ds4_model   * model,
        const ds4_vocab   * vocab,
        const ds4_weights * weights,
        const token_vec   * prompt,
        int                 n_predict,
        int                 ctx_size,
        ds4_token_emit_fn   emit,
        ds4_generation_done_fn done,
        void              * emit_ud) {
    fprintf(stderr, "ds4: using GLM Metal first-token generation path\n");

    if (prompt->len != 1 || prompt->len > ctx_size) {
        fprintf(stderr,
                "ds4: GLM Metal generation currently supports exactly one prompt token; "
                "multi-token prefill needs the GLM KV/DSA graph\n");
        return 1;
    }
    if (n_predict <= 0) {
        if (done) done(emit_ud);
        return 0;
    }
    if (n_predict > 1) {
        fprintf(stderr,
                "ds4: GLM Metal generation currently emits only the first generated token; "
                "stopping after one token\n");
    }

    float *logits = xmalloc((size_t)DS4_N_VOCAB * sizeof(logits[0]));
    const double t0 = now_sec();
    const int rc = glm_metal_first_token_logits(model, weights, prompt->v[0], logits);
    const double t1 = now_sec();
    if (rc != 0) {
        free(logits);
        return 1;
    }

    if (getenv("DS4_TRACE_TOP") != NULL) {
        print_top_logits(stderr, "GLM first-token", vocab, logits, DS4_N_VOCAB, 10);
    }
    const int token = sample_argmax(logits, DS4_N_VOCAB);
    if (!vocab_token_is_generation_stop(vocab, token) && emit) emit(emit_ud, token);
    if (done) done(emit_ud);

    const double eval_s = t1 - t0;
    ds4_log(stderr,
            DS4_LOG_TIMING,
            "ds4: GLM first-token eval: %.2f t/s\n",
            eval_s > 0.0 ? 1.0 / eval_s : 0.0);

    free(logits);
    return 0;
}

static int generate_glm_metal_argmax(
        const ds4_model   * model,
        const ds4_vocab   * vocab,
        const ds4_weights * weights,
        const token_vec   * prompt,
        int                 n_predict,
        int                 ctx_size,
        bool                quality,
        bool                ssd_streaming,
        bool                ssd_streaming_cold,
        uint32_t            ssd_streaming_preload_experts,
        uint64_t            ssd_streaming_cache_bytes,
        uint64_t            ssd_streaming_prefill_headroom_bytes,
        ds4_token_emit_fn   emit,
        ds4_generation_done_fn done,
        void              * emit_ud,
        ds4_session_progress_fn progress,
        void              * progress_ud) {
    fprintf(stderr, "ds4: using GLM full-attention argmax generation path\n");

    if (!prompt || prompt->len <= 0 || prompt->len > ctx_size) {
        fprintf(stderr, "ds4: prompt is empty or exceeds context size\n");
        return 1;
    }
    if (n_predict <= 0) {
        if (done) done(emit_ud);
        return 0;
    }

    ds4_glm_gpu_graph g = {0};
    if (!glm_graph_alloc(&g,
                         model,
                         weights,
                         ctx_size,
                         ssd_streaming,
                         ssd_streaming_cold)) {
        fprintf(stderr, "ds4: failed to allocate GLM graph runtime\n");
        return 1;
    }
    g.quality = quality;
    if ((uint32_t)prompt->len >= g.ctx_size) {
        fprintf(stderr,
                "ds4: prompt length %d leaves no GLM context room (ctx %u)\n",
                prompt->len,
                g.ctx_size);
        glm_graph_free(&g);
        return 1;
    }
    const bool memory_report = getenv("DS4_METAL_MEMORY_REPORT") != NULL;
    if (memory_report) ds4_gpu_print_memory_report("after GLM graph alloc");

    float *logits = xmalloc((size_t)DS4_N_VOCAB * sizeof(logits[0]));
    bool ok = true;
    const bool seed_before_prefill =
        ssd_streaming &&
        !glm_graph_env_present("DS4_ROCM_GLM_DISABLE_STREAMING_SEED_BEFORE_PREFILL",
                               "DS4_METAL_GLM_DISABLE_STREAMING_SEED_BEFORE_PREFILL");
    const double t_prefill0 = now_sec();
    if (seed_before_prefill) {
        ds4_gpu_graph seed_graph;
        memset(&seed_graph, 0, sizeof(seed_graph));
        seed_graph.quality = quality;
        seed_graph.ssd_streaming = ssd_streaming;
        seed_graph.ssd_streaming_cold = ssd_streaming_cold;
        seed_graph.streaming_preload_experts = ssd_streaming_preload_experts;
        ok = metal_graph_seed_streaming_expert_cache_from_hotlist(&seed_graph,
                                                                  model,
                                                                  weights);
    }
    if (ok) {
        ok = glm_graph_prefill_range(&g,
                                     model,
                                     weights,
                                     prompt->v,
                                     0,
                                     (uint32_t)prompt->len,
                                     logits,
                                     progress,
                                     progress_ud,
                                     (uint32_t)prompt->len);
    }
    const double t_prefill1 = now_sec();
    if (memory_report) ds4_gpu_print_memory_report("after GLM prefill");
    if (!ok) {
        fprintf(stderr, "ds4: GLM prefill failed\n");
        free(logits);
        glm_graph_free(&g);
        return 1;
    }
#ifdef DS4_ROCM_BUILD
    /*
     * Decode is SSD-read bound, so the prefill expert headroom is worth more
     * as extra dynamic cache once prefill is done.  Opt out with
     * DS4_ROCM_GLM_STREAMING_GROW_CACHE_AFTER_PREFILL=0.
     */
    const char *grow_cache_env =
        getenv("DS4_ROCM_GLM_STREAMING_GROW_CACHE_AFTER_PREFILL");
    if (ssd_streaming &&
        ssd_streaming_cache_bytes != 0 &&
        ssd_streaming_prefill_headroom_bytes != 0 &&
        (grow_cache_env == NULL || glm_graph_env_truthy(grow_cache_env))) {
        uint64_t budget_bytes = 0;
        uint64_t per_expert_bytes = 0;
        if (ssd_streaming_cache_bytes <=
            UINT64_MAX - ssd_streaming_prefill_headroom_bytes) {
            budget_bytes =
                ssd_streaming_cache_bytes + ssd_streaming_prefill_headroom_bytes;
        }
        const uint32_t grown_budget =
            ds4_streaming_cache_experts_for_byte_budget(weights,
                                                        budget_bytes,
                                                        &per_expert_bytes);
        const uint32_t current_budget =
            ds4_gpu_stream_expert_cache_configured_count();
        if (grown_budget > current_budget) {
            ds4_gpu_set_streaming_expert_cache_budget(grown_budget);
            fprintf(stderr,
                    "ds4: ROCm GLM streaming expert cache grew after prefill: "
                    "%u -> %u experts (%.2f GiB)\n",
                    current_budget,
                    grown_budget,
                    (double)((uint64_t)grown_budget * per_expert_bytes) /
                        1073741824.0);
        }
    }
#else
    (void)ssd_streaming_cache_bytes;
    (void)ssd_streaming_prefill_headroom_bytes;
#endif

    int n_generated = 0;
    int n_decode_eval = 0;
    uint32_t pos = (uint32_t)prompt->len;
    const bool token_timing = getenv("DS4_TOKEN_TIMING") != NULL;
    const double t_decode0 = now_sec();
    for (int i = 0; i < n_predict && pos < g.ctx_size; i++) {
        if (getenv("DS4_TRACE_TOP") != NULL) {
            char label[64];
            snprintf(label, sizeof(label), "GLM step %d", i);
            print_top_logits(stderr, label, vocab, logits, DS4_N_VOCAB, 10);
        }
        const int token = sample_argmax(logits, DS4_N_VOCAB);
        if (vocab_token_is_generation_stop(vocab, token)) break;
        if (emit) emit(emit_ud, token);
        n_generated++;

        if (i == n_predict - 1 || pos + 1u >= g.ctx_size) {
            pos++;
            break;
        }

        const double t_eval0 = token_timing ? now_sec() : 0.0;
        ok = glm_graph_forward_token(&g, model, weights, token, NULL, pos,
                                     NULL, logits, false);
        if (!ok) {
            fprintf(stderr, "ds4: GLM decode failed at position %u\n", pos);
            free(logits);
            glm_graph_free(&g);
            return 1;
        }
        if (token_timing) {
            const double t_eval1 = now_sec();
            fprintf(stderr,
                    "ds4: GLM decode eval %d took %.3f ms\n",
                    n_decode_eval + 1,
                    (t_eval1 - t_eval0) * 1000.0);
        }
        n_decode_eval++;
        pos++;
    }
    const double t_decode1 = now_sec();
    if (done) done(emit_ud);

    const double prefill_s = t_prefill1 - t_prefill0;
    const double decode_s = t_decode1 - t_decode0;
    ds4_log(stderr,
            DS4_LOG_TIMING,
            "ds4: GLM prefill: %.2f t/s, generation: %.2f t/s\n",
            prefill_s > 0.0 ? (double)prompt->len / prefill_s : 0.0,
            decode_s > 0.0 ? (double)n_generated / decode_s : 0.0);

    if (memory_report) ds4_gpu_print_memory_report("before GLM graph free");
    free(logits);
    glm_graph_free(&g);
    return 0;
}
