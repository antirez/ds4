#include "pulsar_engine_internal.h"



/* Encode a full single-token decode step on GPU.  This is the generation
 * hot path: update caches, run all layers, then produce logits. */
static bool gpu_graph_encode_token_raw_swa(
        pulsar_gpu_graph *g,
        const pulsar_model       *model,
        const pulsar_weights     *weights,
        int                    token,
        uint32_t               pos,
        bool                   need_logits,
        bool                   allow_split_flush) {
    if (g->raw_cap == 0) {
        fprintf(stderr, "pulsar: GPU graph raw KV cache is not allocated\n");
        return false;
    }
    const uint32_t raw_row = pos % g->raw_cap;
    const uint32_t n_raw = gpu_graph_raw_span_for_batch(g, pos, 1);

    bool ok = pulsar_gpu_embed_token_hc_tensor(g->cur_hc,
                                              model->map,
                                              model->size,
                                              weights->token_embd->abs_offset,
                                              (uint32_t)weights->token_embd->dim[1],
                                              (uint32_t)token,
                                              PULSAR_N_EMBD,
                                              PULSAR_N_HC) != 0;

    /*
     * Start executing the prefix of the decode graph while the CPU is still
     * encoding the rest. The split point is layer-based because this executor is
     * a fixed DS4 tape, not a dynamic node graph; four layers is the measured
     * point where the prefix is large enough to hide useful work without
     * starving the second command buffer.
     */
    const uint32_t split_after_layers = gpu_graph_token_split_after_layers();

    for (uint32_t il = 0; ok && il < PULSAR_N_LAYER; il++) {
        ok = gpu_graph_encode_decode_layer(g,
                                             model,
                                             &weights->layer[il],
                                             il,
                                             pos,
                                             g->layer_raw_cache[il],
                                             g->raw_cap,
                                             raw_row,
                                             n_raw,
                                             token);
        pulsar_gpu_tensor *tmp = g->cur_hc;
        g->cur_hc = g->after_ffn_hc;
        g->after_ffn_hc = tmp;
        if (ok && allow_split_flush && split_after_layers != 0 && il + 1u == split_after_layers) {
            ok = pulsar_gpu_flush_commands() != 0;
        }
    }

    if (ok && need_logits) {
        ok = gpu_graph_encode_output_head(g, model, weights, weights->output->dim[1]);
    }
    return ok;
}



pulsar_gpu_tensor *gpu_graph_tensor_row_view(
        pulsar_gpu_tensor *base,
        uint32_t          row,
        uint64_t          row_values) {
    return pulsar_gpu_tensor_view(base,
                                 (uint64_t)row * row_values * sizeof(float),
                                 row_values * sizeof(float));
}


/* Row view into an HC residual CARRIER buffer (BF16 storage; task #62). Same as
 * gpu_graph_tensor_row_view but strides by PULSAR_HC_ELT_SIZE, not sizeof(float) —
 * use this (not the generic helper) for cur_hc/next_hc/after_*_hc bases. */
pulsar_gpu_tensor *gpu_graph_hc_row_view(
        pulsar_gpu_tensor *base,
        uint32_t          row,
        uint64_t          row_values) {
    return pulsar_gpu_tensor_view(base,
                                 (uint64_t)row * row_values * PULSAR_HC_ELT_SIZE,
                                 row_values * PULSAR_HC_ELT_SIZE);
}



/* Upload prompt token ids for kernels that need token-aware hash routing. */
bool gpu_graph_upload_prompt_tokens(
        pulsar_gpu_tensor *out_tokens,
        const token_vec  *prompt,
        uint32_t          pos0,
        uint32_t          n_tokens) {
    if (!out_tokens || pos0 > (uint32_t)prompt->len || n_tokens > (uint32_t)prompt->len - pos0) {
        return false;
    }

    int32_t *tokens = (int32_t *)xmalloc((size_t)n_tokens * sizeof(tokens[0]));
    for (uint32_t i = 0; i < n_tokens; i++) tokens[i] = prompt->v[pos0 + i];

    const bool ok = pulsar_gpu_tensor_write(out_tokens,
                                           0,
                                           tokens,
                                           (uint64_t)n_tokens * sizeof(tokens[0])) != 0;
    free(tokens);
    return ok;
}



/* Rebuild ratio-4 compressor state after chunked prefill so a following decode
 * token sees the same rolling compression window. */
static bool gpu_graph_refresh_ratio4_compressor_state(
        pulsar_gpu_graph  *g,
        const pulsar_model  *model,
        pulsar_gpu_tensor *state_kv,
        pulsar_gpu_tensor *state_score,
        const pulsar_tensor *kv_weight,
        const pulsar_tensor *score_weight,
        const pulsar_tensor *ape,
        uint32_t          head_dim,
        uint32_t          width,
        uint32_t          pos0,
        uint32_t          n_tokens) {
    if (n_tokens < 4) {
        return true;
    }
    if (!g || !model || !state_kv || !state_score || !kv_weight || !score_weight || !ape ||
        head_dim == 0 || width == 0) {
        return false;
    }

    /*
     * The recurrent ratio-4 state is intentionally rebuilt from the last
     * four tokens using the small-batch projection kernel. The full-chunk
     * projection is already available, but it uses the matrix-matrix path;
     * mixing those two accumulation orders changes a few FP8 rounding
     * decisions in later chunks.
     */
    pulsar_gpu_tensor *tail_hc = pulsar_gpu_tensor_view(
            g->batch_attn_norm,
            (uint64_t)(n_tokens - 4u) * PULSAR_N_EMBD * sizeof(float),
            4ull * PULSAR_N_EMBD * sizeof(float));
    bool ok = tail_hc != NULL;
    if (ok) {
        ok = gpu_graph_matmul_plain_tensor(g->batch_comp_kv,
                                              model,
                                              kv_weight,
                                         PULSAR_N_EMBD,
                                         width,
                                         tail_hc,
                                         4) != 0;
    }
    if (ok) {
        ok = gpu_graph_matmul_plain_tensor(g->batch_comp_sc,
                                             model,
                                             score_weight,
                                         PULSAR_N_EMBD,
                                         width,
                                         tail_hc,
                                         4) != 0;
    }
    if (ok) {
        ok = pulsar_gpu_compressor_prefill_state_ratio4_tensor(state_kv,
                                                              state_score,
                                                              g->batch_comp_kv,
                                                              g->batch_comp_sc,
                                                              model->map,
                                                              model->size,
                                                              ape->abs_offset,
                                                              ape->type,
                                                              head_dim,
                                                              pos0 + n_tokens - 4u) != 0;
    }
    pulsar_gpu_tensor_free(tail_hc);
    return ok;
}



/* CPU fallback for seeding batched HC state from token embeddings.  It is still
 * useful for tiny speculative verifier batches where a separate GPU embedding
 * command buffer costs more than the small host write. */
static bool gpu_graph_upload_prompt_embeddings_hc_cpu(
        pulsar_gpu_tensor   *out_hc,
        const pulsar_model    *model,
        const pulsar_weights  *weights,
        const token_vec    *prompt,
        uint32_t            pos0,
        uint32_t            n_tokens) {
    if (pos0 > (uint32_t)prompt->len || n_tokens > (uint32_t)prompt->len - pos0) return false;
    const uint64_t hc_dim = (uint64_t)PULSAR_N_HC * PULSAR_N_EMBD;
    const uint64_t total = (uint64_t)n_tokens * hc_dim;
    /* out_hc is an HC residual CARRIER (BF16 storage; task #62) — stage in the
     * carrier's element size, NOT f32, or this write overflows the device buffer
     * by 2x. Rounding matches the GPU store path (__float2bfloat16 = RNE). */
    unsigned char *hc = (unsigned char *)xmalloc((size_t)total * PULSAR_HC_ELT_SIZE);
    float *plain = (float *)xmalloc((size_t)PULSAR_N_EMBD * sizeof(plain[0]));

    for (uint32_t t = 0; t < n_tokens; t++) {
        embed_token_f16(model, weights, prompt->v[pos0 + t], plain);
        unsigned char *dst = hc + (uint64_t)t * hc_dim * PULSAR_HC_ELT_SIZE;
        for (uint32_t h = 0; h < PULSAR_N_HC; h++) {
            unsigned char *row = dst + (uint64_t)h * PULSAR_N_EMBD * PULSAR_HC_ELT_SIZE;
            pulsar_store_hc_carrier_f32(row, plain, PULSAR_N_EMBD);
        }
    }

    const bool ok = pulsar_gpu_tensor_write(out_hc, 0, hc, total * PULSAR_HC_ELT_SIZE) != 0;
    free(plain);
    free(hc);
    return ok;
}



/* Seed the batched HC state from token ids: every HC stream starts as the same
 * 4096-wide embedding.  Long prefill chunks use the GPU get-rows/repeat
 * kernel so the CPU does not build and upload a large [token, HC, dim] tensor. */
bool gpu_graph_upload_prompt_embeddings_hc(
        pulsar_gpu_tensor   *out_hc,
        pulsar_gpu_tensor   *tokens,
        const pulsar_model    *model,
        const pulsar_weights  *weights,
        const token_vec    *prompt,
        uint32_t            pos0,
        uint32_t            n_tokens) {
    if (pos0 > (uint32_t)prompt->len || n_tokens > (uint32_t)prompt->len - pos0) return false;

    static long gpu_min_cached = -1;
    if (gpu_min_cached < 0) {
        gpu_min_cached = 512;
        const char *gpu_min_env = getenv("PULSAR_CUDA_GPU_BATCH_EMBED_MIN");
        if (gpu_min_env && gpu_min_env[0]) {
            char *end = NULL;
            unsigned long v = strtoul(gpu_min_env, &end, 10);
            if (end != gpu_min_env && v <= UINT32_MAX) gpu_min_cached = (long)v;
        }
    }
    const uint32_t gpu_min = (uint32_t)gpu_min_cached;

    if (tokens && n_tokens >= gpu_min) {
        return pulsar_gpu_embed_tokens_hc_tensor(out_hc,
                                                tokens,
                                                model->map,
                                                model->size,
                                                weights->token_embd->abs_offset,
                                                (uint32_t)weights->token_embd->dim[1],
                                                n_tokens,
                                                PULSAR_N_EMBD,
                                                PULSAR_N_HC) != 0;
    }

    return gpu_graph_upload_prompt_embeddings_hc_cpu(out_hc,
                                                       model,
                                                       weights,
                                                       prompt,
                                                       pos0,
                                                       n_tokens);
}



bool gpu_graph_warmup_prefill_kernels(
        pulsar_gpu_graph   *g,
        const pulsar_model   *model,
        const pulsar_weights *weights,
        uint32_t           n_tokens) {
    static bool warmed = false;
    if (warmed) return true;
    if (getenv("PULSAR_CUDA_NO_PREFILL_KERNEL_WARMUP") != NULL) return true;

    /*
     * The first batched F16 matmul can pay GPU's one-time pipeline execution
     * cost. Run the same HC attention projection on scratch storage before the
     * measured prefill. The output is overwritten by the real graph.
     */
    if (n_tokens <= 8) return true;

    const uint64_t hc_dim = (uint64_t)PULSAR_N_HC * PULSAR_N_EMBD;
    const uint64_t mix_hc = 2ull * PULSAR_N_HC + (uint64_t)PULSAR_N_HC * PULSAR_N_HC;

    bool ok = pulsar_gpu_begin_commands() != 0;
    if (ok) {
        ok = gpu_graph_matmul_plain_tensor(g->batch_hc_mix,
                                             model,
                                             weights->layer[0].hc_attn_fn,
                                         hc_dim,
                                         mix_hc,
                                         g->batch_flat_hc,
                                         n_tokens) != 0;
    }
    if (ok) ok = pulsar_gpu_end_commands() != 0;
    if (!ok) {
        fprintf(stderr, "pulsar: GPU prefill kernel warmup failed\n");
        return false;
    }

    warmed = true;
    return true;
}



/* Encode the batched prefill attention half for one layer.  It mirrors the CPU
 * layer-major path: HC pre/norm, Q/KV, cache/compression, prefix attention. */
bool gpu_graph_indexer_stage_profile_boundary(
        const char *stage,
        uint32_t    il,
        uint32_t    pos0,
        uint32_t    n_tokens,
        uint32_t    n_comp,
        double     *stage_t0) {
    if (pulsar_gpu_end_commands() == 0) return false;
    const double now = now_sec();
    if (stage != NULL) {
        fprintf(stderr,
                "pulsar: GPU indexer stage layer=%u pos=%u tokens=%u comp=%u %s=%.3f ms\n",
                il,
                pos0,
                n_tokens,
                n_comp,
                stage,
                (now - *stage_t0) * 1000.0);
    }
    *stage_t0 = now;
    return pulsar_gpu_begin_commands() != 0;
}



static bool gpu_graph_profile_layer_env_match(const char *env_name, uint32_t il) {
    const char *layer_env = getenv(env_name);
    if (!layer_env || !layer_env[0]) return true;

    char *end = NULL;
    const unsigned long layer = strtoul(layer_env, &end, 10);
    return end != layer_env &&
           *end == '\0' &&
           layer <= UINT32_MAX &&
           (uint32_t)layer == il;
}



static bool gpu_graph_layer_stage_profile_enabled(uint32_t il) {
    static int cache = -1;
    return gpu_graph_env_flag("PULSAR_CUDA_LAYER_STAGE_PROFILE", &cache) &&
           gpu_graph_profile_layer_env_match("PULSAR_CUDA_LAYER_STAGE_PROFILE_LAYER", il);
}



bool gpu_graph_decode_stage_profile_enabled(uint32_t il) {
    static int cache = -1;
    return gpu_graph_env_flag("PULSAR_CUDA_DECODE_STAGE_PROFILE", &cache) &&
           gpu_graph_profile_layer_env_match("PULSAR_CUDA_DECODE_STAGE_PROFILE_LAYER", il);
}



/* Optional prefill stage profiler. It intentionally ends the current GPU
 * command buffer and waits, so the printed number includes encoding plus GPU
 * execution for the stage just emitted. This is disabled by default because it
 * adds synchronization points and changes scheduling. */
bool gpu_graph_layer_stage_profile_boundary(
        const char *part,
        const char *stage,
        uint32_t    il,
        uint32_t    pos0,
        uint32_t    n_tokens,
        double     *stage_t0) {
    if (pulsar_gpu_end_commands() == 0) return false;
    const double now = now_sec();
    fprintf(stderr,
            "pulsar: GPU layer stage part=%s layer=%u pos=%u tokens=%u %s=%.3f ms\n",
            part,
            il,
            pos0,
            n_tokens,
            stage,
            (now - *stage_t0) * 1000.0);
    *stage_t0 = now;
    return pulsar_gpu_begin_commands() != 0;
}



static bool gpu_graph_q_stage_profile_boundary(
        const char *stage,
        uint32_t    il,
        uint32_t    pos0,
        uint32_t    n_tokens,
        double     *stage_t0) {
    if (pulsar_gpu_end_commands() == 0) return false;
    const double now = now_sec();
    fprintf(stderr,
            "pulsar: GPU Q path stage layer=%u pos=%u tokens=%u %s=%.3f ms\n",
            il,
            pos0,
            n_tokens,
            stage,
            (now - *stage_t0) * 1000.0);
    *stage_t0 = now;
    return pulsar_gpu_begin_commands() != 0;
}



bool gpu_graph_encode_layer_attention_batch(
        pulsar_gpu_graph  *g,
        const pulsar_model        *model,
        const pulsar_layer_weights *layer,
        uint32_t                il,
        uint32_t                pos0,
        uint32_t                n_tokens) {
    if (n_tokens == 0 || n_tokens > g->prefill_cap) return false;

    const uint64_t hc_dim = (uint64_t)PULSAR_N_HC * PULSAR_N_EMBD;
    const uint64_t mix_hc = 2ull * PULSAR_N_HC + (uint64_t)PULSAR_N_HC * PULSAR_N_HC;
    const uint64_t q_rank = layer->attn_q_a->dim[1];
    const uint64_t q_dim = (uint64_t)PULSAR_N_HEAD * PULSAR_N_HEAD_DIM;
    const uint32_t n_groups = PULSAR_N_OUT_GROUP;
    const uint32_t group_heads = PULSAR_N_HEAD / n_groups;
    const uint32_t group_dim = PULSAR_N_HEAD_DIM * group_heads;
    const uint32_t rank = PULSAR_N_LORA_O;
    const uint32_t ratio = pulsar_layer_compress_ratio(il);
    const bool compressed = ratio != 0;
    /* Banked multiseq step (Tier-2): rows are independent sessions — per-row
     * position/bank from the host mirrors (gpu_graph_multiseq_step_begin),
     * per-bank compressor frontiers, banked kernel operands (whole pool +
     * device descriptor arrays), scalar counters = read-only supersets.
     * step_begin rejects position-0 rows, so zero_prefix is never multiseq. */
    const bool mseq = g->batch_multiseq;
    const uint32_t nb = gpu_graph_bank_pool_count(g);
    if (mseq && (pos0 == 0 || n_tokens > g->batch_multiseq_rows ||
                 (uint32_t)g->ms_positions[0] != pos0)) {
        fprintf(stderr, "pulsar: multiseq layer batch rejected: rows/pos0 do not "
                        "match the armed step (pos0=%u n_tokens=%u rows=%u)\n",
                pos0, n_tokens, g->batch_multiseq_rows);
        return false;
    }
    const bool zero_prefix = pos0 == 0;
    static int index_stage_env = -1, q_stage_env = -1;
    const bool index_stage_profile = gpu_graph_env_flag("PULSAR_CUDA_INDEXER_STAGE_PROFILE", &index_stage_env);
    const bool layer_stage_profile = gpu_graph_layer_stage_profile_enabled(il);
    const bool q_stage_profile = gpu_graph_env_flag("PULSAR_CUDA_Q_STAGE_PROFILE", &q_stage_env);
    double layer_stage_t0 = layer_stage_profile ? now_sec() : 0.0;
    double q_stage_t0 = q_stage_profile ? now_sec() : 0.0;
#define PULSAR_CUDA_PROFILE_ATTN_STAGE(name) do { \
        if (ok && layer_stage_profile) { \
            ok = gpu_graph_layer_stage_profile_boundary("attn", (name), il, pos0, n_tokens, &layer_stage_t0); \
        } \
    } while (0)
#define PULSAR_CUDA_PROFILE_Q_STAGE(name) do { \
        if (ok && q_stage_profile) { \
            ok = gpu_graph_q_stage_profile_boundary((name), il, pos0, n_tokens, &q_stage_t0); \
        } \
    } while (0)
    const float freq_base = layer_rope_freq_base(il);
    const float freq_scale = layer_rope_freq_scale(il);
    const float ext_factor = compressed && PULSAR_ROPE_SCALE_FACTOR > 1.0f ? 1.0f : 0.0f;
    float attn_factor = 1.0f;
    if (ext_factor != 0.0f && freq_scale > 0.0f) {
        attn_factor /= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    uint32_t *comp_counts = compressed ? (uint32_t *)xcalloc(n_tokens, sizeof(comp_counts[0])) : NULL;
    uint32_t *index_counts = ratio == 4 ? (uint32_t *)xcalloc(n_tokens, sizeof(index_counts[0])) : NULL;
    const bool qkv_rms_fused = !gpu_graph_use_reference_qkv_norm();
    pulsar_gpu_tensor *hc_mix_view = pulsar_gpu_tensor_view(
            g->batch_hc_mix, 0, (uint64_t)n_tokens * mix_hc * sizeof(float));
    pulsar_gpu_tensor *hc_split_view = pulsar_gpu_tensor_view(
            g->batch_hc_split, 0, (uint64_t)n_tokens * mix_hc * sizeof(float));
    pulsar_gpu_tensor *attn_cur_view = pulsar_gpu_tensor_view(
            g->batch_attn_cur, 0, (uint64_t)n_tokens * PULSAR_N_EMBD * sizeof(float));
    pulsar_gpu_tensor *after_attn_hc_view = pulsar_gpu_tensor_view(
            g->batch_after_attn_hc, 0, (uint64_t)n_tokens * hc_dim * PULSAR_HC_ELT_SIZE);   /* carrier */
    bool ok = hc_mix_view && hc_split_view && attn_cur_view && after_attn_hc_view;
    const bool fuse_hc_norm = PULSAR_N_HC == 4 &&
                              !gpu_graph_use_reference_hc_decode() &&
                              gpu_graph_enable_batch_hc_norm_fusion();
    if (ok) ok = pulsar_gpu_rms_norm_plain_rows_tensor(g->batch_flat_hc,
                                                      g->batch_cur_hc,
                                                      (uint32_t)hc_dim,
                                                      n_tokens,
                                                      PULSAR_RMS_EPS) != 0;
    if (ok) ok = gpu_graph_matmul_plain_tensor(hc_mix_view,
                                              model,
                                              layer->hc_attn_fn,
                                             hc_dim,
                                             mix_hc,
                                             g->batch_flat_hc,
                                             n_tokens) != 0;
    if (gpu_graph_use_reference_hc_decode()) {
        if (ok) ok = pulsar_gpu_hc_split_sinkhorn_tensor(hc_split_view,
                                                        hc_mix_view,
                                                        model->map,
                                                        model->size,
                                                        layer->hc_attn_scale->abs_offset,
                                                        layer->hc_attn_base->abs_offset,
                                                        PULSAR_N_HC,
                                                        PULSAR_N_HC_SINKHORN_ITER,
                                                        PULSAR_HC_EPS) != 0;
        if (ok) ok = pulsar_gpu_hc_weighted_sum_split_tensor(attn_cur_view,
                                                            g->batch_cur_hc,
                                                            hc_split_view,
                                                            PULSAR_N_EMBD,
                                                            PULSAR_N_HC) != 0;
    } else if (fuse_hc_norm) {
        if (ok) ok = pulsar_gpu_hc_split_weighted_sum_norm_tensor(attn_cur_view,
                                                                 g->batch_attn_norm,
                                                                 hc_split_view,
                                                                 hc_mix_view,
                                                                 g->batch_cur_hc,
                                                                 model->map,
                                                                 model->size,
                                                                 layer->hc_attn_scale->abs_offset,
                                                                 layer->hc_attn_base->abs_offset,
                                                                 layer->attn_norm->abs_offset,
                                                                 PULSAR_N_EMBD,
                                                                 PULSAR_N_HC,
                                                                 PULSAR_N_HC_SINKHORN_ITER,
                                                                 PULSAR_HC_EPS,
                                                                 PULSAR_RMS_EPS) != 0;
    } else {
        if (ok) ok = pulsar_gpu_hc_split_weighted_sum_tensor(attn_cur_view,
                                                            hc_split_view,
                                                            hc_mix_view,
                                                            g->batch_cur_hc,
                                                            model->map,
                                                            model->size,
                                                            layer->hc_attn_scale->abs_offset,
                                                            layer->hc_attn_base->abs_offset,
                                                            PULSAR_N_EMBD,
                                                            PULSAR_N_HC,
                                                            PULSAR_N_HC_SINKHORN_ITER,
                                                            PULSAR_HC_EPS) != 0;
    }
    if (ok) {
        gpu_graph_debug_dump_tensor("hc_attn_pre", g->batch_attn_cur,
                                      (uint64_t)n_tokens * PULSAR_N_EMBD, il, pos0);
    }
    PULSAR_CUDA_PROFILE_ATTN_STAGE("hc_pre");
    if (ok && !fuse_hc_norm) {
        ok = pulsar_gpu_rms_norm_weight_rows_tensor(g->batch_attn_norm,
                                                  g->batch_attn_cur,
                                                  model->map,
                                                  model->size,
                                                  layer->attn_norm->abs_offset,
                                                  PULSAR_N_EMBD,
                                                  n_tokens,
                                                  PULSAR_RMS_EPS) != 0;
    }
    if (ok) {
        gpu_graph_debug_dump_tensor("attn_norm", g->batch_attn_norm,
                                      (uint64_t)n_tokens * PULSAR_N_EMBD, il, pos0);
    }
    /* batch_attn_norm is now final for this layer and feeds up to seven MXFP8
     * projections below (q_a, kv, attn compressor kv+gate, indexer compressor
     * kv+gate, indexer_proj), each of which would otherwise re-quantize the
     * identical [n_tokens x n_embd] f32 tensor.  Arm the quantize-once cache
     * here -- immediately after the ONLY writes to this buffer (the fused-norm
     * and standalone-norm branches above), which is what keeps a later cache
     * hit coherent -- and disarm at the single exit below. */
    if (ok) pulsar_gpu_mxfp8_act_cache_arm(g->batch_attn_norm, n_tokens, PULSAR_N_EMBD);
    PULSAR_CUDA_PROFILE_ATTN_STAGE("norm");
    PULSAR_CUDA_PROFILE_Q_STAGE("pre_q");
    if (ok) ok = gpu_graph_matmul_mxfp8_named_tensor("attn_q_a",
                                                      il,
                                                      pos0,
                                                      g->batch_qr,
                                                      model,
                                                      layer->attn_q_a,
                                                      PULSAR_N_EMBD,
                                                      q_rank,
                                                      g->batch_attn_norm,
                                                      n_tokens);
    if (ok) {
        gpu_graph_debug_dump_tensor("q_lora", g->batch_qr,
                                      (uint64_t)n_tokens * q_rank, il, pos0);
    }
    PULSAR_CUDA_PROFILE_Q_STAGE("q_a");
    if (qkv_rms_fused) {
        if (ok) ok = gpu_graph_matmul_mxfp8_named_tensor("attn_kv",
                                                          il,
                                                          pos0,
                                                          g->batch_kv_raw,
                                                          model,
                                                          layer->attn_kv,
                                                          PULSAR_N_EMBD,
                                                          PULSAR_N_HEAD_DIM,
                                                          g->batch_attn_norm,
                                                          n_tokens);
        if (ok) {
            gpu_graph_debug_dump_tensor("KVraw", g->batch_kv_raw,
                                          (uint64_t)n_tokens * PULSAR_N_HEAD_DIM, il, pos0);
        }
        if (ok) ok = pulsar_gpu_dsv4_qkv_rms_norm_rows_tensor(g->batch_qr_norm,
                                                             g->batch_qr,
                                                             model->map,
                                                             model->size,
                                                             layer->attn_q_a_norm->abs_offset,
                                                             (uint32_t)q_rank,
                                                             g->batch_kv,
                                                             g->batch_kv_raw,
                                                             layer->attn_kv_a_norm->abs_offset,
                                                             PULSAR_N_HEAD_DIM,
                                                             n_tokens,
                                                             PULSAR_RMS_EPS) != 0;
    } else {
        if (ok) ok = pulsar_gpu_rms_norm_weight_rows_tensor(g->batch_qr_norm,
                                                           g->batch_qr,
                                                           model->map,
                                                           model->size,
                                                           layer->attn_q_a_norm->abs_offset,
                                                           (uint32_t)q_rank,
                                                           n_tokens,
                                                           PULSAR_RMS_EPS) != 0;
    }
    if (ok) {
        gpu_graph_debug_dump_tensor("q_lora_norm", g->batch_qr_norm,
                                      (uint64_t)n_tokens * q_rank, il, pos0);
    }
    if (qkv_rms_fused && ok) {
        gpu_graph_debug_dump_tensor("KVnorm", g->batch_kv,
                                      (uint64_t)n_tokens * PULSAR_N_HEAD_DIM, il, pos0);
    }
    PULSAR_CUDA_PROFILE_Q_STAGE("q_a_norm");
    {
        if (ok) ok = gpu_graph_matmul_mxfp8_named_tensor("attn_q_b",
                                                          il,
                                                          pos0,
                                                          g->batch_q,
                                                          model,
                                                          layer->attn_q_b,
                                                          q_rank,
                                                          q_dim,
                                                          g->batch_qr_norm,
                                                          n_tokens);
        if (ok) {
            gpu_graph_debug_dump_tensor("Qraw", g->batch_q,
                                          (uint64_t)n_tokens * q_dim, il, pos0);
        }
        PULSAR_CUDA_PROFILE_Q_STAGE("q_b");
        const bool prefill_q_norm_debug = gpu_graph_debug_wants("Qnorm", il, pos0);
        bool prefill_q_norm_rope_fused = false;
        if (ok && !prefill_q_norm_debug) {
            prefill_q_norm_rope_fused =
                pulsar_gpu_head_rms_norm_rope_tail_tensor(g->batch_q,
                                                       n_tokens,
                                                       PULSAR_N_HEAD,
                                                       PULSAR_N_HEAD_DIM,
                                                       PULSAR_N_ROT,
                                                       pos0,
                                                       compressed ? (uint32_t)PULSAR_ROPE_ORIG_CTX : 0,
                                                       false,
                                                       freq_base,
                                                       freq_scale,
                                                       ext_factor,
                                                       attn_factor,
                                                       PULSAR_ROPE_YARN_BETA_FAST,
                                                       PULSAR_ROPE_YARN_BETA_SLOW,
                                                       PULSAR_RMS_EPS,
                                                       mseq ? g->batch_positions : NULL) != 0;
        }
        if (!prefill_q_norm_rope_fused) {
            if (ok) ok = pulsar_gpu_head_rms_norm_tensor(g->batch_q,
                                                        n_tokens,
                                                        PULSAR_N_HEAD,
                                                        PULSAR_N_HEAD_DIM,
                                                        PULSAR_RMS_EPS) != 0;
            if (ok) {
                gpu_graph_debug_dump_tensor("Qnorm", g->batch_q,
                                              (uint64_t)n_tokens * q_dim, il, pos0);
            }
            PULSAR_CUDA_PROFILE_Q_STAGE("head_norm");
            if (ok) ok = pulsar_gpu_rope_tail_tensor(g->batch_q,
                                                    n_tokens,
                                                    PULSAR_N_HEAD,
                                                    PULSAR_N_HEAD_DIM,
                                                    PULSAR_N_ROT,
                                                    pos0,
                                                    compressed ? (uint32_t)PULSAR_ROPE_ORIG_CTX : 0,
                                                    false,
                                                    freq_base,
                                                    freq_scale,
                                                    ext_factor,
                                                    attn_factor,
                                                    PULSAR_ROPE_YARN_BETA_FAST,
                                                    PULSAR_ROPE_YARN_BETA_SLOW,
                                                    mseq ? g->batch_positions : NULL) != 0;
        } else {
            PULSAR_CUDA_PROFILE_Q_STAGE("head_norm");
        }
        if (ok) {
            gpu_graph_debug_dump_tensor("Qcur", g->batch_q,
                                          (uint64_t)n_tokens * q_dim, il, pos0);
        }
        PULSAR_CUDA_PROFILE_Q_STAGE("rope");
    }
    PULSAR_CUDA_PROFILE_ATTN_STAGE("q_path");
    if (!qkv_rms_fused) {
        if (ok) ok = gpu_graph_matmul_mxfp8_named_tensor("attn_kv",
                                                          il,
                                                          pos0,
                                                          g->batch_kv_raw,
                                                          model,
                                                          layer->attn_kv,
                                                          PULSAR_N_EMBD,
                                                          PULSAR_N_HEAD_DIM,
                                                          g->batch_attn_norm,
                                                          n_tokens);
        if (ok) {
            gpu_graph_debug_dump_tensor("KVraw", g->batch_kv_raw,
                                          (uint64_t)n_tokens * PULSAR_N_HEAD_DIM, il, pos0);
        }
        if (ok) ok = pulsar_gpu_rms_norm_weight_rows_tensor(g->batch_kv,
                                                           g->batch_kv_raw,
                                                           model->map,
                                                           model->size,
                                                           layer->attn_kv_a_norm->abs_offset,
                                                           PULSAR_N_HEAD_DIM,
                                                           n_tokens,
                                                           PULSAR_RMS_EPS) != 0;
        if (ok) {
            gpu_graph_debug_dump_tensor("KVnorm", g->batch_kv,
                                          (uint64_t)n_tokens * PULSAR_N_HEAD_DIM, il, pos0);
        }
    }
    if (ok) ok = pulsar_gpu_rope_tail_tensor(g->batch_kv,
                                            n_tokens,
                                            PULSAR_N_HEAD_KV,
                                            PULSAR_N_HEAD_DIM,
                                            PULSAR_N_ROT,
                                            pos0,
                                            compressed ? (uint32_t)PULSAR_ROPE_ORIG_CTX : 0,
                                            false,
                                            freq_base,
                                            freq_scale,
                                            ext_factor,
                                            attn_factor,
                                            PULSAR_ROPE_YARN_BETA_FAST,
                                            PULSAR_ROPE_YARN_BETA_SLOW,
                                            mseq ? g->batch_positions : NULL) != 0;
    if (ok) {
        gpu_graph_debug_dump_tensor("KVrope", g->batch_kv,
                                      (uint64_t)n_tokens * PULSAR_N_HEAD_DIM, il, pos0);
    }
    if (ok) ok = pulsar_gpu_dsv4_fp8_kv_quantize_tensor(g->batch_kv,
                                                       n_tokens,
                                                       PULSAR_N_HEAD_DIM,
                                                       PULSAR_N_ROT) != 0;
    if (ok) {
        gpu_graph_debug_dump_tensor("KVcur", g->batch_kv,
                                      (uint64_t)n_tokens * PULSAR_N_HEAD_DIM, il, pos0);
    }
    PULSAR_CUDA_PROFILE_ATTN_STAGE("kv_path");
    /*
     * Static graph order is q, kv, cpy_k(raw SWA), then attention. For a
     * zero-prefix batch it is safe to store the whole batch at once: attention
     * reads the contiguous batch KV, and the ring only has to end with the last
     * SWA rows for later chunks/decode. For nonzero chunks the physical ring is
     * sized to hold the current chunk plus the previous SWA window, while the
     * attention mask still enforces the 128-token logical window.
     */
    if (ok && zero_prefix) ok = pulsar_gpu_store_raw_kv_batch_tensor(g->layer_raw_cache[il],
                                                                    g->batch_kv,
                                                                    g->raw_cap,
                                                                    pos0,
                                                                    n_tokens,
                                                                    PULSAR_N_HEAD_DIM,
                                                                    (uint32_t)gpu_graph_raw_f16_enabled(),
                                                                    NULL, NULL, 1) != 0;
    const bool raw_batch_attention = zero_prefix && ratio == 0;
    bool batch_attention_done = false;

    if (ok && raw_batch_attention) {
        ok = pulsar_gpu_attention_prefill_raw_heads_tensor(g->batch_heads,
                                                          model->map,
                                                          model->size,
                                                          layer->attn_sinks->abs_offset,
                                                          g->batch_q,
                                                          g->batch_kv,
                                                          n_tokens,
                                                          g->raw_window,
                                                          PULSAR_N_HEAD,
                                                          PULSAR_N_HEAD_DIM,
                                                          0 /* batch_kv is f32 */) != 0;
        if (ok) batch_attention_done = true;
    } else if (ok && !zero_prefix && ratio == 0 && n_tokens <= g->raw_cap) {
        /*
         * The ubatch path stores the whole batch in the SWA cache, then runs
         * one batched attention kernel with an absolute-position causal/window
         * mask.  This avoids mixing prefill with the different single-token
         * attention path.
         */
        const uint32_t n_raw = gpu_graph_raw_span_for_batch(g, pos0, n_tokens);
        /* Nonzero prompt chunks read the SWA cache as a ring.  FlashAttention
         * receives a linearized window starting at raw_start, not physical row
         * zero; otherwise wrapped chunks silently miss recent raw keys. */
        const uint32_t raw_start = gpu_graph_raw_start_for_span(g,
                                                                  pos0 + n_tokens - 1u,
                                                                  n_raw);
        ok = pulsar_gpu_store_raw_kv_batch_tensor(mseq ? gpu_graph_bank_raw_pool(g, il)
                                                    : g->layer_raw_cache[il],
                                                 g->batch_kv,
                                                 g->raw_cap,
                                                 pos0,
                                                 n_tokens,
                                                 PULSAR_N_HEAD_DIM,
                                                 (uint32_t)gpu_graph_raw_f16_enabled(),
                                                 mseq ? g->batch_positions : NULL,
                                                 mseq ? g->batch_seq_id : NULL,
                                                 mseq ? nb : 1) != 0;
        if (ok && !gpu_graph_raw_f16_enabled()) {
            /* diag-only dump; the dumper reads f32 and would misinterpret a
             * __half ring — skip it under PULSAR_RAW_F16.  Under mseq the store
             * above scattered through the WHOLE bank pool while this classic
             * view shows only the current bank — annotate rather than let
             * the dump masquerade as the stored bytes (the tag itself is a
             * filename/filter key, so it stays "raw_cache"). */
            if (mseq && gpu_graph_debug_wants("raw_cache", il, pos0)) {
                fprintf(stderr, "pulsar: raw_cache dump layer %u pos %u: cur-bank "
                                "view; banked stores not shown\n", il, pos0);
            }
            gpu_graph_debug_dump_tensor("raw_cache",
                                          g->layer_raw_cache[il],
                                          (uint64_t)n_raw * PULSAR_N_HEAD_DIM,
                                          il,
                                          pos0);
        }
        if (ok) {
            ok = pulsar_gpu_attention_decode_raw_batch_heads_tensor(g->batch_heads,
                                                                   model->map,
                                                                   model->size,
                                                                   layer->attn_sinks->abs_offset,
                                                                   g->batch_q,
                                                                   mseq ? gpu_graph_bank_raw_pool(g, il)
                                                                        : g->layer_raw_cache[il],
                                                                   n_tokens,
                                                                   pos0,
                                                                   mseq ? 0 : n_raw,
                                                                   g->raw_cap,
                                                                   mseq ? 0 : raw_start,
                                                                    g->raw_window,
                                                                    PULSAR_N_HEAD,
                                                                    PULSAR_N_HEAD_DIM,
                                                                    0,
                                                                    (uint32_t)gpu_graph_raw_f16_enabled(),
                                                                    mseq ? g->batch_positions : NULL,
                                                                    mseq ? g->batch_seq_id : NULL,
                                                                    0,
                                                                    mseq ? nb : 1) != 0;
        }
        if (ok) batch_attention_done = true;
    } else if (ok && ratio != 0) {
        const uint32_t coff = ratio == 4 ? 2u : 1u;
        const uint32_t comp_width = coff * PULSAR_N_HEAD_DIM;
        const bool have_attn_comp = layer->attn_compressor_kv && layer->attn_compressor_gate &&
                                    layer->attn_compressor_ape && layer->attn_compressor_norm;
        if (!have_attn_comp) {
            fprintf(stderr, "pulsar: GPU layer-major prefill needs attention compressor weights\n");
            ok = false;
        }
        if (ok) {
            ok = gpu_graph_matmul_plain_tensor(g->batch_comp_kv,
                                              model,
                                              layer->attn_compressor_kv,
                                             PULSAR_N_EMBD,
                                             comp_width,
                                             g->batch_attn_norm,
                                             n_tokens) != 0;
            if (ok) ok = gpu_graph_matmul_plain_tensor(g->batch_comp_sc,
                                              model,
                                              layer->attn_compressor_gate,
                                                     PULSAR_N_EMBD,
                                                     comp_width,
                                                     g->batch_attn_norm,
                                                     n_tokens) != 0;
        }
        if (ok) gpu_graph_debug_dump_tensor("attn_comp_kv_raw",
                                              g->batch_comp_kv,
                                              (uint64_t)comp_width * n_tokens,
                                              il,
                                              pos0);
        if (ok) gpu_graph_debug_dump_tensor("attn_comp_score_raw",
                                              g->batch_comp_sc,
                                              (uint64_t)comp_width * n_tokens,
                                              il,
                                              pos0);
        /* Stage-B save: keep this batch's per-position compressor projections so
         * a partial spec accept can roll the pool state forward without a
         * transformer replay. Must run here -- the indexer section below reuses
         * batch_comp_kv/sc. */
        if (ok && g->spec_comp_save_n && g->spec_comp_kv_save[il]) {
            uint32_t sn = g->spec_comp_save_n;
            if (sn > n_tokens) sn = n_tokens;
            if (sn > 17u) sn = 17u;
            const uint64_t sb = (uint64_t)sn * comp_width * sizeof(float);
            ok = pulsar_gpu_tensor_copy(g->spec_comp_kv_save[il], 0, g->batch_comp_kv, 0, sb) != 0 &&
                 pulsar_gpu_tensor_copy(g->spec_comp_sc_save[il], 0, g->batch_comp_sc, 0, sb) != 0;
        }
        uint32_t n_comp = g->layer_n_comp[il];
        if (zero_prefix) {
            n_comp = n_tokens / ratio;
            if (ok && n_comp > g->layer_comp_cap[il]) {
                fprintf(stderr, "pulsar: GPU layer-major compressed KV cache capacity exceeded at layer %u\n", il);
                ok = false;
            }
            if (ok && gpu_graph_attn_pack_enabled() &&
                n_comp > g->attn_comp_stage_cap) {
                fprintf(stderr, "pulsar: GPU graph compressed KV staging capacity exceeded at layer %u\n", il);
                ok = false;
            }
            pulsar_gpu_tensor *attn_comp_target = NULL;
            if (ok) {
                attn_comp_target = gpu_graph_attn_comp_prefill_target(g, il, 0, n_comp);
                ok = attn_comp_target != NULL &&
                     pulsar_gpu_compressor_prefill_tensor(attn_comp_target,
                                                         g->layer_attn_state_kv[il],
                                                         g->layer_attn_state_score[il],
                                                         g->batch_comp_kv,
                                                         g->batch_comp_sc,
                                                         model->map,
                                                         model->size,
                                                         layer->attn_compressor_ape->abs_offset,
                                                         layer->attn_compressor_ape->type,
                                                         layer->attn_compressor_norm->abs_offset,
                                                         layer->attn_compressor_norm->type,
                                                         PULSAR_N_HEAD_DIM,
                                                         ratio,
                                                         pos0,
                                                         n_tokens,
                                                         PULSAR_N_ROT,
                                                         compressed ? (uint32_t)PULSAR_ROPE_ORIG_CTX : 0,
                                                         /* Under PULSAR_ATTN_PACK the commit's pack-store kernel
                                                          * is the single fp8 quantizer (same recipe); a second
                                                          * quantize is NOT bit-idempotent for borderline block
                                                          * amax (scale can shift, re-rounding small values). */
                                                         !gpu_graph_attn_pack_enabled(),
                                                         freq_base,
                                                         freq_scale,
                                                         ext_factor,
                                                         attn_factor,
                                                         PULSAR_ROPE_YARN_BETA_FAST,
                                                         PULSAR_ROPE_YARN_BETA_SLOW,
                                                         PULSAR_RMS_EPS) != 0;
                if (ok && n_comp != 0) {
                    ok = gpu_graph_commit_attn_comp_stage(g, il, 0, n_comp);
                }
                if (ok && ratio == 4) {
                    ok = gpu_graph_refresh_ratio4_compressor_state(g,
                                                                     model,
                                                                     g->layer_attn_state_kv[il],
                                                                     g->layer_attn_state_score[il],
                                                                     layer->attn_compressor_kv,
                                                                     layer->attn_compressor_gate,
                                                                     layer->attn_compressor_ape,
                                                                     PULSAR_N_HEAD_DIM,
                                                                     comp_width,
                                                                     pos0,
                                                                     n_tokens);
                }
            }
            if (ok) {
                g->layer_n_comp[il] = n_comp;
                for (uint32_t t = 0; t < n_tokens; t++) {
                    comp_counts[t] = (pos0 + t + 1u) / ratio;
                }
                if (n_comp != 0) {
                    gpu_graph_debug_dump_tensor("KVcompress",
                                                  attn_comp_target,
                                                  (uint64_t)n_comp * PULSAR_N_HEAD_DIM,
                                                  il,
                                                  pos0);
                }
                gpu_graph_debug_dump_tensor("attn_state_kv",
                                              g->layer_attn_state_kv[il],
                                              (uint64_t)comp_width * coff * ratio,
                                              il,
                                              pos0);
                gpu_graph_debug_dump_tensor("attn_state_score",
                                              g->layer_attn_state_score[il],
                                              (uint64_t)comp_width * coff * ratio,
                                              il,
                                              pos0);
            }
            gpu_graph_attn_comp_prefill_target_free(attn_comp_target);
        } else {
            /* Classic aligned-chunk fast path: one contiguous ratio-aligned run
             * on the single-session cache. */
            const bool aligned_chunk = !mseq &&
                                       (pos0 % ratio) == 0u && (n_tokens % ratio) == 0u;
            /* LEVER 2 (plan-34): a BANKED step whose whole batch is a single
             * same-bank contiguous run (step_begin guarantees contiguity, so
             * seq_id[0]==seq_id[last] => one run => one bank) that is ratio-
             * aligned reuses the SAME batched replay/pool kernels the classic
             * aligned path trusts, keyed at the bank's frontier — one launch
             * per stage instead of the per-row loop's N launches.  Byte-
             * identical by construction: the batched pool recomputes the same
             * candidate window the recurrent per-row state carries (proven
             * equivalent by the classic aligned-vs-per-row equivalence), and it
             * leaves the bank state in the exact configuration classic leaves
             * (compressor_prefill*_tensor's tail re-seed == the per-row shift).
             * A mixed step's decode rows are length-1 runs (seq_id[0]!=
             * seq_id[last]) and fall through to the per-row loop unchanged. */
            const uint32_t run_bank = mseq ? (uint32_t)g->ms_seq_id[0] : 0u;
            const bool mseq_aligned_run = mseq &&
                (uint32_t)g->ms_seq_id[n_tokens - 1u] == run_bank &&
                (pos0 % ratio) == 0u && (n_tokens % ratio) == 0u;
            if (aligned_chunk) {
                const uint32_t comp_before = g->layer_n_comp[il];
                const uint32_t comp_chunk = n_tokens / ratio;
                if (comp_before + comp_chunk > g->layer_comp_cap[il]) {
                    fprintf(stderr, "pulsar: GPU graph compressed KV cache capacity exceeded at layer %u\n", il);
                    ok = false;
                }
                if (ok && gpu_graph_attn_pack_enabled() &&
                    comp_chunk > g->attn_comp_stage_cap) {
                    fprintf(stderr, "pulsar: GPU graph compressed KV staging capacity exceeded at layer %u\n", il);
                    ok = false;
                }
                pulsar_gpu_tensor *attn_comp_target =
                    ok ? gpu_graph_attn_comp_prefill_target(g, il, comp_before, comp_chunk) : NULL;
                if (ok && !attn_comp_target) ok = false;
                if (ok && ratio == 4) {
                    ok = pulsar_gpu_compressor_prefill_ratio4_replay_tensor(
                            attn_comp_target,
                            g->layer_attn_state_kv[il],
                            g->layer_attn_state_score[il],
                            g->batch_comp_kv,
                            g->batch_comp_sc,
                            model->map,
                            model->size,
                            layer->attn_compressor_ape->abs_offset,
                            layer->attn_compressor_ape->type,
                            layer->attn_compressor_norm->abs_offset,
                            layer->attn_compressor_norm->type,
                            PULSAR_N_HEAD_DIM,
                            pos0,
                            n_tokens,
                            PULSAR_N_ROT,
                            compressed ? (uint32_t)PULSAR_ROPE_ORIG_CTX : 0,
                            /* pack: quantize once, in the commit (see above) */
                            !gpu_graph_attn_pack_enabled(),
                            freq_base,
                            freq_scale,
                            ext_factor,
                            attn_factor,
                            PULSAR_ROPE_YARN_BETA_FAST,
                            PULSAR_ROPE_YARN_BETA_SLOW,
                            PULSAR_RMS_EPS) != 0;
                } else if (ok) {
                    ok = pulsar_gpu_compressor_prefill_tensor(
                            attn_comp_target,
                            g->layer_attn_state_kv[il],
                            g->layer_attn_state_score[il],
                            g->batch_comp_kv,
                            g->batch_comp_sc,
                            model->map,
                            model->size,
                            layer->attn_compressor_ape->abs_offset,
                            layer->attn_compressor_ape->type,
                            layer->attn_compressor_norm->abs_offset,
                            layer->attn_compressor_norm->type,
                            PULSAR_N_HEAD_DIM,
                            ratio,
                            pos0,
                            n_tokens,
                            PULSAR_N_ROT,
                            compressed ? (uint32_t)PULSAR_ROPE_ORIG_CTX : 0,
                            /* pack: quantize once, in the commit (see above) */
                            !gpu_graph_attn_pack_enabled(),
                            freq_base,
                            freq_scale,
                            ext_factor,
                            attn_factor,
                            PULSAR_ROPE_YARN_BETA_FAST,
                            PULSAR_ROPE_YARN_BETA_SLOW,
                            PULSAR_RMS_EPS) != 0;
                }
                if (ok && comp_chunk != 0) {
                    ok = gpu_graph_commit_attn_comp_stage(g, il, comp_before, comp_chunk);
                }
                if (ok && ratio == 4) {
                    ok = gpu_graph_refresh_ratio4_compressor_state(g,
                                                                     model,
                                                                     g->layer_attn_state_kv[il],
                                                                     g->layer_attn_state_score[il],
                                                                     layer->attn_compressor_kv,
                                                                     layer->attn_compressor_gate,
                                                                     layer->attn_compressor_ape,
                                                                     PULSAR_N_HEAD_DIM,
                                                                     comp_width,
                                                                     pos0,
                                                                     n_tokens);
                }
                if (ok) {
                    g->layer_n_comp[il] = comp_before + comp_chunk;
                    if (comp_counts) {
                        for (uint32_t t = 0; t < n_tokens; t++) {
                            comp_counts[t] = (pos0 + t + 1u) / ratio;
                        }
                    }
                    gpu_graph_debug_dump_tensor("KVcompress",
                                                  attn_comp_target,
                                                  (uint64_t)comp_chunk * PULSAR_N_HEAD_DIM,
                                                  il,
                                                  pos0);
                    gpu_graph_debug_dump_tensor("attn_state_kv",
                                                  g->layer_attn_state_kv[il],
                                                  (uint64_t)comp_width * coff * ratio,
                                                  il,
                                                  pos0);
                    gpu_graph_debug_dump_tensor("attn_state_score",
                                                  g->layer_attn_state_score[il],
                                                  (uint64_t)comp_width * coff * ratio,
                                                  il,
                                                  pos0);
                }
                gpu_graph_attn_comp_prefill_target_free(attn_comp_target);
            } else if (mseq_aligned_run) {
                /* LEVER 2: batched banked attn-compressor emit for the single
                 * same-bank aligned run.  Mirrors the classic aligned branch
                 * above, keyed at the bank's frontier / bank state lanes /
                 * bank comp cache.  quantize_fp8 = !pack (the pack commit is the
                 * single fp8 quantizer, as in the classic path). */
                const uint32_t bank = run_bank;
                const uint32_t comp_before = g->ms_n_comp[bank][il];
                const uint32_t comp_chunk = n_tokens / ratio;
                const bool pack = gpu_graph_attn_pack_enabled();
                pulsar_gpu_tensor *bank_comp = NULL;
                pulsar_gpu_tensor *bank_st_kv = NULL, *bank_st_sc = NULL, *comp_target = NULL;
                if (comp_before + comp_chunk > g->layer_comp_cap[il]) {
                    fprintf(stderr, "pulsar: GPU graph compressed KV cache capacity exceeded at layer %u\n", il);
                    ok = false;
                }
                if (ok && pack && comp_chunk > g->attn_comp_stage_cap) {
                    fprintf(stderr, "pulsar: GPU graph compressed KV staging capacity exceeded at layer %u\n", il);
                    ok = false;
                }
                if (ok) {
                    bank_st_kv = gpu_graph_bank_attn_state_kv_view(g, il, bank);
                    bank_st_sc = gpu_graph_bank_attn_state_score_view(g, il, bank);
                    ok = bank_st_kv && bank_st_sc;
                }
                if (ok) {
                    if (pack) {
                        comp_target = pulsar_gpu_tensor_view(g->attn_comp_stage, 0,
                                (uint64_t)comp_chunk * PULSAR_N_HEAD_DIM * sizeof(float));
                    } else {
                        bank_comp = gpu_graph_bank_attn_comp_view(g, il, bank);
                        if (bank_comp) {
                            comp_target = pulsar_gpu_tensor_view(bank_comp,
                                    (uint64_t)comp_before * PULSAR_N_HEAD_DIM * sizeof(float),
                                    (uint64_t)comp_chunk * PULSAR_N_HEAD_DIM * sizeof(float));
                        }
                    }
                    ok = comp_target != NULL;
                }
                if (ok && ratio == 4) {
                    ok = pulsar_gpu_compressor_prefill_ratio4_replay_tensor(
                            comp_target, bank_st_kv, bank_st_sc,
                            g->batch_comp_kv, g->batch_comp_sc,
                            model->map, model->size,
                            layer->attn_compressor_ape->abs_offset,
                            layer->attn_compressor_ape->type,
                            layer->attn_compressor_norm->abs_offset,
                            layer->attn_compressor_norm->type,
                            PULSAR_N_HEAD_DIM, pos0, n_tokens, PULSAR_N_ROT,
                            compressed ? (uint32_t)PULSAR_ROPE_ORIG_CTX : 0,
                            !pack, freq_base, freq_scale, ext_factor, attn_factor,
                            PULSAR_ROPE_YARN_BETA_FAST, PULSAR_ROPE_YARN_BETA_SLOW,
                            PULSAR_RMS_EPS) != 0;
                } else if (ok) {
                    ok = pulsar_gpu_compressor_prefill_tensor(
                            comp_target, bank_st_kv, bank_st_sc,
                            g->batch_comp_kv, g->batch_comp_sc,
                            model->map, model->size,
                            layer->attn_compressor_ape->abs_offset,
                            layer->attn_compressor_ape->type,
                            layer->attn_compressor_norm->abs_offset,
                            layer->attn_compressor_norm->type,
                            PULSAR_N_HEAD_DIM, ratio, pos0, n_tokens, PULSAR_N_ROT,
                            compressed ? (uint32_t)PULSAR_ROPE_ORIG_CTX : 0,
                            !pack, freq_base, freq_scale, ext_factor, attn_factor,
                            PULSAR_ROPE_YARN_BETA_FAST, PULSAR_ROPE_YARN_BETA_SLOW,
                            PULSAR_RMS_EPS) != 0;
                }
                if (ok && comp_chunk != 0) {
                    ok = gpu_graph_commit_attn_comp_stage_bank(g, il, bank, comp_before, comp_chunk);
                }
                if (ok && ratio == 4) {
                    ok = gpu_graph_refresh_ratio4_compressor_state(g, model,
                            bank_st_kv, bank_st_sc,
                            layer->attn_compressor_kv, layer->attn_compressor_gate,
                            layer->attn_compressor_ape, PULSAR_N_HEAD_DIM, comp_width,
                            pos0, n_tokens);
                }
                if (ok) {
                    g->ms_n_comp[bank][il] = comp_before + comp_chunk;
                    if (comp_counts) {
                        for (uint32_t t = 0; t < n_tokens; t++) {
                            comp_counts[t] = (pos0 + t + 1u) / ratio;
                        }
                    }
                }
                pulsar_gpu_tensor_free(comp_target);
                pulsar_gpu_tensor_free(bank_comp);
                pulsar_gpu_tensor_free(bank_st_sc);
                pulsar_gpu_tensor_free(bank_st_kv);
            } else {
                /* Per-row compressor loop.  Multiseq: row t belongs to bank
                 * ms_seq_id[t] at absolute position ms_positions[t] — the
                 * pool/emit run against THAT bank's state lanes and comp
                 * cache at ITS frontier (ms_n_comp[bank][il]), and bump only
                 * that bank's counter; the scalar layer_n_comp stays the
                 * step-top superset (read-only here — the §6.1 race class is
                 * structurally unreachable).  Per-bank ratio groups are
                 * independent, and the pack-mode f32 stage row is safe to
                 * share across banks: each iteration's emit packs it before
                 * the next iteration's kernels run on the same stream. */
                for (uint32_t t = 0; ok && t < n_tokens; t++) {
                    const uint32_t pos = mseq ? (uint32_t)g->ms_positions[t] : pos0 + t;
                    const uint32_t bank = mseq ? (uint32_t)g->ms_seq_id[t] : 0u;
                    uint32_t *const n_comp_slot = mseq ? &g->ms_n_comp[bank][il]
                                                       : &g->layer_n_comp[il];
                    const bool emit = ((pos + 1u) % ratio) == 0u;
                    if (emit && *n_comp_slot >= g->layer_comp_cap[il]) {
                        fprintf(stderr, "pulsar: GPU graph compressed KV cache capacity exceeded at layer %u\n", il);
                        ok = false;
                        break;
                    }
                    pulsar_gpu_tensor *kv_view = gpu_graph_tensor_row_view(g->batch_comp_kv, t, comp_width);
                    pulsar_gpu_tensor *sc_view = gpu_graph_tensor_row_view(g->batch_comp_sc, t, comp_width);
                    const uint32_t comp_row = *n_comp_slot;
                    pulsar_gpu_tensor *ms_st_kv = mseq
                        ? gpu_graph_bank_attn_state_kv_view(g, il, bank) : NULL;
                    pulsar_gpu_tensor *ms_st_sc = mseq
                        ? gpu_graph_bank_attn_state_score_view(g, il, bank) : NULL;
                    /* f32 storage writes the persistent cache directly, so it
                     * needs the bank's comp view; pack mode stages in the
                     * shared f32 row and commits bank-aware below. */
                    pulsar_gpu_tensor *ms_target = (mseq && !gpu_graph_attn_pack_enabled())
                        ? gpu_graph_bank_attn_comp_view(g, il, bank) : NULL;
                    ok = kv_view && sc_view &&
                         (!mseq || (ms_st_kv && ms_st_sc &&
                                    (gpu_graph_attn_pack_enabled() || ms_target)));
                    if (ok) {
                        ok = pulsar_gpu_compressor_update_tensor(kv_view,
                                                            sc_view,
                                                            mseq ? ms_st_kv : g->layer_attn_state_kv[il],
                                                            mseq ? ms_st_sc : g->layer_attn_state_score[il],
                                                            ms_target ? ms_target
                                                                      : gpu_graph_attn_comp_update_target(g, il),
                                                            model->map,
                                                            model->size,
                                                            layer->attn_compressor_ape->abs_offset,
                                                            layer->attn_compressor_ape->type,
                                                            layer->attn_compressor_norm->abs_offset,
                                                            layer->attn_compressor_norm->type,
                                                            PULSAR_N_HEAD_DIM,
                                                            ratio,
                                                            pos,
                                                            gpu_graph_attn_comp_update_row(comp_row),
                                                            PULSAR_N_ROT,
                                                            compressed ? (uint32_t)PULSAR_ROPE_ORIG_CTX : 0,
                                                            freq_base,
                                                            freq_scale,
                                                            ext_factor,
                                                            attn_factor,
                                                            PULSAR_ROPE_YARN_BETA_FAST,
                                                            PULSAR_ROPE_YARN_BETA_SLOW,
                                                            PULSAR_RMS_EPS) != 0;
                    }
                    if (ok && emit) {
                        pulsar_gpu_tensor *comp_row_view = ms_target
                            ? pulsar_gpu_tensor_view(ms_target,
                                                  (uint64_t)comp_row * PULSAR_N_HEAD_DIM * sizeof(float),
                                                  (uint64_t)PULSAR_N_HEAD_DIM * sizeof(float))
                            : gpu_graph_attn_comp_row_view(g, il, comp_row);
                        if (gpu_graph_attn_pack_enabled()) {
                            /* comp_row_view aliases the f32 stage; commit below
                             * quantizes+packs and roundtrips the stage in place
                             * — dump after commit to match f32-mode values. */
                            ok = comp_row_view != NULL;
                        } else {
                            ok = comp_row_view &&
                                 pulsar_gpu_dsv4_fp8_kv_quantize_tensor(comp_row_view,
                                                                      1,
                                                                      PULSAR_N_HEAD_DIM,
                                                                      PULSAR_N_ROT) != 0;
                        }
                        if (ok && !gpu_graph_attn_pack_enabled()) {
                            gpu_graph_debug_dump_tensor("KVcompress",
                                                          comp_row_view,
                                                          PULSAR_N_HEAD_DIM,
                                                          il,
                                                          pos);
                        }
                        if (ok) {
                            ok = mseq
                                ? gpu_graph_commit_attn_comp_stage_bank(g, il, bank, comp_row, 1)
                                : gpu_graph_commit_attn_comp_stage(g, il, comp_row, 1);
                        }
                        if (ok && gpu_graph_attn_pack_enabled()) {
                            gpu_graph_debug_dump_tensor("KVcompress",
                                                          comp_row_view,
                                                          PULSAR_N_HEAD_DIM,
                                                          il,
                                                          pos);
                        }
                        pulsar_gpu_tensor_free(comp_row_view);
                    }
                    if (ok && emit) (*n_comp_slot)++;
                    if (comp_counts) comp_counts[t] = *n_comp_slot;
                    pulsar_gpu_tensor_free(ms_target);
                    pulsar_gpu_tensor_free(ms_st_sc);
                    pulsar_gpu_tensor_free(ms_st_kv);
                    pulsar_gpu_tensor_free(sc_view);
                    pulsar_gpu_tensor_free(kv_view);
                }
            }
            n_comp = g->layer_n_comp[il];
        }
        PULSAR_CUDA_PROFILE_ATTN_STAGE("compressor");

        if (ok && ratio == 4) {
            const uint32_t index_width = coff * PULSAR_N_INDEXER_HEAD_DIM;
            if (!layer->indexer_compressor_kv || !layer->indexer_compressor_gate ||
                !layer->indexer_compressor_ape || !layer->indexer_compressor_norm ||
                !layer->indexer_attn_q_b || !layer->indexer_proj) {
                fprintf(stderr, "pulsar: GPU layer-major prefill needs indexer weights\n");
                ok = false;
            }
            if (ok) {
                ok = gpu_graph_matmul_plain_tensor(g->batch_comp_kv,
                                              model,
                                              layer->indexer_compressor_kv,
                                                 PULSAR_N_EMBD,
                                                 index_width,
                                                 g->batch_attn_norm,
                                                 n_tokens) != 0;
                if (ok) ok = gpu_graph_matmul_plain_tensor(g->batch_comp_sc,
                                              model,
                                              layer->indexer_compressor_gate,
                                                         PULSAR_N_EMBD,
                                                         index_width,
                                                         g->batch_attn_norm,
                                                         n_tokens) != 0;
            }
            if (ok) gpu_graph_debug_dump_tensor("indexer_comp_kv_raw",
                                                  g->batch_comp_kv,
                                                  (uint64_t)index_width * n_tokens,
                                                  il,
                                                  pos0);
            if (ok) gpu_graph_debug_dump_tensor("indexer_comp_score_raw",
                                                  g->batch_comp_sc,
                                                  (uint64_t)index_width * n_tokens,
                                                  il,
                                                  pos0);
            /* Stage-B save (indexer variant; see the attn compressor hook). */
            if (ok && g->spec_comp_save_n && g->spec_icomp_kv_save[il]) {
                uint32_t sn = g->spec_comp_save_n;
                if (sn > n_tokens) sn = n_tokens;
                if (sn > 17u) sn = 17u;
                const uint64_t sb = (uint64_t)sn * index_width * sizeof(float);
                ok = pulsar_gpu_tensor_copy(g->spec_icomp_kv_save[il], 0, g->batch_comp_kv, 0, sb) != 0 &&
                     pulsar_gpu_tensor_copy(g->spec_icomp_sc_save[il], 0, g->batch_comp_sc, 0, sb) != 0;
            }
            if (ok) ok = gpu_graph_matmul_plain_tensor(g->batch_indexer_q,
                                                          model,
                                                          layer->indexer_attn_q_b,
                                                          q_rank,
                                                          (uint64_t)PULSAR_N_INDEXER_HEAD * PULSAR_N_INDEXER_HEAD_DIM,
                                                          g->batch_qr_norm,
                                                          n_tokens);
            if (ok) ok = pulsar_gpu_rope_tail_tensor(g->batch_indexer_q,
                                                    n_tokens,
                                                    PULSAR_N_INDEXER_HEAD,
                                                    PULSAR_N_INDEXER_HEAD_DIM,
                                                    PULSAR_N_ROT,
                                                    pos0,
                                                    compressed ? (uint32_t)PULSAR_ROPE_ORIG_CTX : 0,
                                                    false,
                                                    freq_base,
                                                    freq_scale,
                                                    ext_factor,
                                                    attn_factor,
                                                    PULSAR_ROPE_YARN_BETA_FAST,
                                                    PULSAR_ROPE_YARN_BETA_SLOW,
                                                    mseq ? g->batch_positions : NULL) != 0;
            if (ok) ok = pulsar_gpu_dsv4_indexer_qat_tensor(g->batch_indexer_q,
                                                          n_tokens * PULSAR_N_INDEXER_HEAD,
                                                          PULSAR_N_INDEXER_HEAD_DIM) != 0;
            if (ok) ok = gpu_graph_matmul_plain_tensor(g->batch_indexer_weights,
                                              model,
                                              layer->indexer_proj,
                                                     PULSAR_N_EMBD,
                                                     PULSAR_N_INDEXER_HEAD,
                                                     g->batch_attn_norm,
                                                     n_tokens) != 0;
            if (zero_prefix) {
                if (ok && n_comp > g->layer_comp_cap[il]) {
                    fprintf(stderr, "pulsar: GPU layer-major indexer cache capacity exceeded at layer %u\n", il);
                    ok = false;
                }
                const int idx_fp4 = gpu_graph_idx_fp4_enabled();
                if (ok) {
                    ok = pulsar_gpu_compressor_prefill_tensor(idx_fp4 ? g->idx_comp_stage
                                                                   : g->layer_index_comp_cache[il],
                                                             g->layer_index_state_kv[il],
                                                             g->layer_index_state_score[il],
                                                             g->batch_comp_kv,
                                                             g->batch_comp_sc,
                                                             model->map,
                                                             model->size,
                                                             layer->indexer_compressor_ape->abs_offset,
                                                             layer->indexer_compressor_ape->type,
                                                             layer->indexer_compressor_norm->abs_offset,
                                                             layer->indexer_compressor_norm->type,
                                                             PULSAR_N_INDEXER_HEAD_DIM,
                                                             ratio,
                                                             pos0,
                                                             n_tokens,
                                                             PULSAR_N_ROT,
                                                             compressed ? (uint32_t)PULSAR_ROPE_ORIG_CTX : 0,
                                                             false,
                                                             freq_base,
                                                             freq_scale,
                                                             ext_factor,
                                                             attn_factor,
                                                             PULSAR_ROPE_YARN_BETA_FAST,
                                                             PULSAR_ROPE_YARN_BETA_SLOW,
                                                             PULSAR_RMS_EPS) != 0;
                }
                if (ok && n_comp != 0) {
                    ok = idx_fp4
                        ? pulsar_gpu_dsv4_indexer_qat_pack_tensor(g->idx_comp_stage,
                                                                g->layer_index_comp_cache[il],
                                                                0,
                                                                n_comp,
                                                                PULSAR_N_INDEXER_HEAD_DIM) != 0
                        : pulsar_gpu_dsv4_indexer_qat_tensor(g->layer_index_comp_cache[il],
                                                          n_comp,
                                                          PULSAR_N_INDEXER_HEAD_DIM) != 0;
                    /* plan-33 inc C: boundary-row restore (whole-prefill site). */
                    if (ok) ok = gpu_graph_emit_keep_restore(g, il,
                            g->banks.n_banks ? g->banks.cur_bank : 0u, 0, n_comp, true);
                }
                if (ok) {
                    ok = gpu_graph_refresh_ratio4_compressor_state(g,
                                                                     model,
                                                                     g->layer_index_state_kv[il],
                                                                     g->layer_index_state_score[il],
                                                                     layer->indexer_compressor_kv,
                                                                     layer->indexer_compressor_gate,
                                                                     layer->indexer_compressor_ape,
                                                                     PULSAR_N_INDEXER_HEAD_DIM,
                                                                     index_width,
                                                                     pos0,
                                                                     n_tokens);
                }
                if (ok) {
                    g->layer_n_index_comp[il] = n_comp;
                    for (uint32_t t = 0; t < n_tokens; t++) {
                        index_counts[t] = (pos0 + t + 1u) / ratio;
                    }
                    if (n_comp != 0) {
                        gpu_graph_debug_dump_tensor("indexer_KVcompress",
                                                      idx_fp4 ? g->idx_comp_stage
                                                              : g->layer_index_comp_cache[il],
                                                      (uint64_t)n_comp * PULSAR_N_INDEXER_HEAD_DIM,
                                                      il,
                                                      pos0);
                    }
                    gpu_graph_debug_dump_tensor("indexer_state_kv",
                                                  g->layer_index_state_kv[il],
                                                  (uint64_t)index_width * coff * ratio,
                                                  il,
                                                  pos0);
                    gpu_graph_debug_dump_tensor("indexer_state_score",
                                                  g->layer_index_state_score[il],
                                                  (uint64_t)index_width * coff * ratio,
                                                  il,
                                                  pos0);
                }
            } else {
                /* Classic aligned fast path; LEVER 2 adds the banked single-
                 * same-bank-aligned-run variant (see the attn emit section). */
                const bool aligned_chunk = !mseq &&
                                           (pos0 % ratio) == 0u && (n_tokens % ratio) == 0u;
                const uint32_t run_bank = mseq ? (uint32_t)g->ms_seq_id[0] : 0u;
                const bool mseq_aligned_run = mseq &&
                    (uint32_t)g->ms_seq_id[n_tokens - 1u] == run_bank &&
                    (pos0 % ratio) == 0u && (n_tokens % ratio) == 0u;
                if (aligned_chunk) {
                    const uint32_t index_before = g->layer_n_index_comp[il];
                    const uint32_t index_chunk = n_tokens / ratio;
                    if (index_before + index_chunk > g->layer_comp_cap[il]) {
                        fprintf(stderr, "pulsar: GPU graph indexer compressed KV cache capacity exceeded at layer %u\n", il);
                        ok = false;
                    }
                    const int idx_fp4 = gpu_graph_idx_fp4_enabled();
                    pulsar_gpu_tensor *index_view = NULL;
                    if (ok) {
                        index_view = pulsar_gpu_tensor_view(
                                idx_fp4 ? g->idx_comp_stage : g->layer_index_comp_cache[il],
                                (uint64_t)index_before * PULSAR_N_INDEXER_HEAD_DIM * sizeof(float),
                                (uint64_t)index_chunk * PULSAR_N_INDEXER_HEAD_DIM * sizeof(float));
                        ok = index_view != NULL;
                    }
                    if (ok) {
                        ok = pulsar_gpu_compressor_prefill_ratio4_replay_tensor(
                                index_view,
                                g->layer_index_state_kv[il],
                                g->layer_index_state_score[il],
                                g->batch_comp_kv,
                                g->batch_comp_sc,
                                model->map,
                                model->size,
                                layer->indexer_compressor_ape->abs_offset,
                                layer->indexer_compressor_ape->type,
                                layer->indexer_compressor_norm->abs_offset,
                                layer->indexer_compressor_norm->type,
                                PULSAR_N_INDEXER_HEAD_DIM,
                                pos0,
                                n_tokens,
                                PULSAR_N_ROT,
                                compressed ? (uint32_t)PULSAR_ROPE_ORIG_CTX : 0,
                                false,
                                freq_base,
                                freq_scale,
                                ext_factor,
                                attn_factor,
                                PULSAR_ROPE_YARN_BETA_FAST,
                                PULSAR_ROPE_YARN_BETA_SLOW,
                                PULSAR_RMS_EPS) != 0;
                    }
                    if (ok && index_chunk != 0) {
                        ok = idx_fp4
                            ? pulsar_gpu_dsv4_indexer_qat_pack_tensor(index_view,
                                                                    g->layer_index_comp_cache[il],
                                                                    index_before,
                                                                    index_chunk,
                                                                    PULSAR_N_INDEXER_HEAD_DIM) != 0
                            : pulsar_gpu_dsv4_indexer_qat_tensor(index_view,
                                                              index_chunk,
                                                              PULSAR_N_INDEXER_HEAD_DIM) != 0;
                        /* plan-33 inc C: boundary-row restore (chunked emit site —
                         * the replay-from-R path that recomputes row R/4). */
                        if (ok) ok = gpu_graph_emit_keep_restore(g, il,
                                g->banks.n_banks ? g->banks.cur_bank : 0u,
                                index_before, index_chunk, true);
                    }
                    if (ok) {
                        ok = gpu_graph_refresh_ratio4_compressor_state(g,
                                                                         model,
                                                                         g->layer_index_state_kv[il],
                                                                         g->layer_index_state_score[il],
                                                                         layer->indexer_compressor_kv,
                                                                         layer->indexer_compressor_gate,
                                                                         layer->indexer_compressor_ape,
                                                                         PULSAR_N_INDEXER_HEAD_DIM,
                                                                         index_width,
                                                                         pos0,
                                                                         n_tokens);
                    }
                    if (ok) {
                        g->layer_n_index_comp[il] = index_before + index_chunk;
                        if (index_counts) {
                            for (uint32_t t = 0; t < n_tokens; t++) {
                                index_counts[t] = (pos0 + t + 1u) / ratio;
                            }
                        }
                        gpu_graph_debug_dump_tensor("indexer_KVcompress",
                                                      index_view,
                                                      (uint64_t)index_chunk * PULSAR_N_INDEXER_HEAD_DIM,
                                                      il,
                                                      pos0);
                        gpu_graph_debug_dump_tensor("indexer_state_kv",
                                                      g->layer_index_state_kv[il],
                                                      (uint64_t)index_width * coff * ratio,
                                                      il,
                                                      pos0);
                        gpu_graph_debug_dump_tensor("indexer_state_score",
                                                      g->layer_index_state_score[il],
                                                      (uint64_t)index_width * coff * ratio,
                                                      il,
                                                      pos0);
                    }
                    pulsar_gpu_tensor_free(index_view);
                } else if (mseq_aligned_run) {
                    /* LEVER 2: batched banked indexer-compressor emit for the
                     * single same-bank aligned run — mirrors the classic aligned
                     * indexer path, keyed at the bank frontier / bank index
                     * state lanes / bank index comp cache.  fp4 stages in the
                     * shared idx_comp_stage (one bank per step, no aliasing) and
                     * packs into the bank cache; f32 quantizes the bank cache in
                     * place. */
                    const uint32_t bank = run_bank;
                    const uint32_t index_before = g->ms_n_index_comp[bank][il];
                    const uint32_t index_chunk = n_tokens / ratio;
                    const int idx_fp4 = gpu_graph_idx_fp4_enabled();
                    pulsar_gpu_tensor *bank_idx = NULL, *bank_ist_kv = NULL, *bank_ist_sc = NULL;
                    pulsar_gpu_tensor *index_view = NULL;
                    if (index_before + index_chunk > g->layer_comp_cap[il]) {
                        fprintf(stderr, "pulsar: GPU graph indexer compressed KV cache capacity exceeded at layer %u\n", il);
                        ok = false;
                    }
                    if (ok) {
                        bank_idx = gpu_graph_bank_index_comp_view(g, il, bank);
                        bank_ist_kv = gpu_graph_bank_index_state_kv_view(g, il, bank);
                        bank_ist_sc = gpu_graph_bank_index_state_score_view(g, il, bank);
                        ok = bank_idx && bank_ist_kv && bank_ist_sc;
                    }
                    if (ok) {
                        index_view = idx_fp4
                            ? pulsar_gpu_tensor_view(g->idx_comp_stage,
                                    (uint64_t)index_before * PULSAR_N_INDEXER_HEAD_DIM * sizeof(float),
                                    (uint64_t)index_chunk * PULSAR_N_INDEXER_HEAD_DIM * sizeof(float))
                            : pulsar_gpu_tensor_view(bank_idx,
                                    (uint64_t)index_before * PULSAR_N_INDEXER_HEAD_DIM * sizeof(float),
                                    (uint64_t)index_chunk * PULSAR_N_INDEXER_HEAD_DIM * sizeof(float));
                        ok = index_view != NULL;
                    }
                    if (ok) {
                        ok = pulsar_gpu_compressor_prefill_ratio4_replay_tensor(
                                index_view, bank_ist_kv, bank_ist_sc,
                                g->batch_comp_kv, g->batch_comp_sc,
                                model->map, model->size,
                                layer->indexer_compressor_ape->abs_offset,
                                layer->indexer_compressor_ape->type,
                                layer->indexer_compressor_norm->abs_offset,
                                layer->indexer_compressor_norm->type,
                                PULSAR_N_INDEXER_HEAD_DIM, pos0, n_tokens, PULSAR_N_ROT,
                                compressed ? (uint32_t)PULSAR_ROPE_ORIG_CTX : 0,
                                false, freq_base, freq_scale, ext_factor, attn_factor,
                                PULSAR_ROPE_YARN_BETA_FAST, PULSAR_ROPE_YARN_BETA_SLOW,
                                PULSAR_RMS_EPS) != 0;
                    }
                    if (ok && index_chunk != 0) {
                        ok = idx_fp4
                            ? pulsar_gpu_dsv4_indexer_qat_pack_tensor(index_view,
                                                                    bank_idx,
                                                                    index_before,
                                                                    index_chunk,
                                                                    PULSAR_N_INDEXER_HEAD_DIM) != 0
                            : pulsar_gpu_dsv4_indexer_qat_tensor(index_view,
                                                              index_chunk,
                                                              PULSAR_N_INDEXER_HEAD_DIM) != 0;
                        if (ok) ok = gpu_graph_emit_keep_restore(g, il, bank,
                                index_before, index_chunk, true);
                    }
                    if (ok) {
                        ok = gpu_graph_refresh_ratio4_compressor_state(g, model,
                                bank_ist_kv, bank_ist_sc,
                                layer->indexer_compressor_kv, layer->indexer_compressor_gate,
                                layer->indexer_compressor_ape, PULSAR_N_INDEXER_HEAD_DIM,
                                index_width, pos0, n_tokens);
                    }
                    if (ok) {
                        g->ms_n_index_comp[bank][il] = index_before + index_chunk;
                        if (index_counts) {
                            for (uint32_t t = 0; t < n_tokens; t++) {
                                index_counts[t] = (pos0 + t + 1u) / ratio;
                            }
                        }
                    }
                    pulsar_gpu_tensor_free(index_view);
                    pulsar_gpu_tensor_free(bank_ist_sc);
                    pulsar_gpu_tensor_free(bank_ist_kv);
                    pulsar_gpu_tensor_free(bank_idx);
                } else {
                    /* Per-row indexer compressor loop; multiseq semantics as
                     * in the attn emit loop above (bank state lanes, bank
                     * frontier row, bank counter bump; scalar = read-only
                     * superset).  The fp4 stage rows are indexed by the
                     * bank-LOCAL frontier: two banks at the same frontier
                     * share a stage row safely because each iteration's emit
                     * packs it before the next iteration's kernels run. */
                    for (uint32_t t = 0; ok && t < n_tokens; t++) {
                        const uint32_t pos = mseq ? (uint32_t)g->ms_positions[t] : pos0 + t;
                        const uint32_t bank = mseq ? (uint32_t)g->ms_seq_id[t] : 0u;
                        uint32_t *const n_index_slot = mseq ? &g->ms_n_index_comp[bank][il]
                                                            : &g->layer_n_index_comp[il];
                        const bool emit = ((pos + 1u) % ratio) == 0u;
                        if (emit && *n_index_slot >= g->layer_comp_cap[il]) {
                            fprintf(stderr, "pulsar: GPU graph indexer compressed KV cache capacity exceeded at layer %u\n", il);
                            ok = false;
                            break;
                        }
                        pulsar_gpu_tensor *kv_view = gpu_graph_tensor_row_view(g->batch_comp_kv, t, index_width);
                        pulsar_gpu_tensor *sc_view = gpu_graph_tensor_row_view(g->batch_comp_sc, t, index_width);
                        const uint32_t index_row = *n_index_slot;
                        const int idx_fp4 = gpu_graph_idx_fp4_enabled();
                        pulsar_gpu_tensor *ms_st_kv = mseq
                            ? gpu_graph_bank_index_state_kv_view(g, il, bank) : NULL;
                        pulsar_gpu_tensor *ms_st_sc = mseq
                            ? gpu_graph_bank_index_state_score_view(g, il, bank) : NULL;
                        pulsar_gpu_tensor *ms_cache = mseq
                            ? gpu_graph_bank_index_comp_view(g, il, bank) : NULL;
                        ok = kv_view && sc_view &&
                             (!mseq || (ms_st_kv && ms_st_sc && ms_cache));
                        if (ok) {
                            ok = pulsar_gpu_compressor_update_tensor(kv_view,
                                                                sc_view,
                                                                mseq ? ms_st_kv : g->layer_index_state_kv[il],
                                                                mseq ? ms_st_sc : g->layer_index_state_score[il],
                                                                idx_fp4 ? g->idx_comp_stage
                                                                        : (mseq ? ms_cache
                                                                                : g->layer_index_comp_cache[il]),
                                                                model->map,
                                                                model->size,
                                                                layer->indexer_compressor_ape->abs_offset,
                                                                layer->indexer_compressor_ape->type,
                                                                layer->indexer_compressor_norm->abs_offset,
                                                                layer->indexer_compressor_norm->type,
                                                                PULSAR_N_INDEXER_HEAD_DIM,
                                                                ratio,
                                                                pos,
                                                                index_row,
                                                                PULSAR_N_ROT,
                                                                compressed ? (uint32_t)PULSAR_ROPE_ORIG_CTX : 0,
                                                                freq_base,
                                                                freq_scale,
                                                                ext_factor,
                                                                attn_factor,
                                                                PULSAR_ROPE_YARN_BETA_FAST,
                                                                PULSAR_ROPE_YARN_BETA_SLOW,
                                                                PULSAR_RMS_EPS) != 0;
                        }
                        if (ok && emit) {
                            pulsar_gpu_tensor *index_row_view = pulsar_gpu_tensor_view(
                                    idx_fp4 ? g->idx_comp_stage
                                            : (mseq ? ms_cache : g->layer_index_comp_cache[il]),
                                    (uint64_t)index_row * PULSAR_N_INDEXER_HEAD_DIM * sizeof(float),
                                    (uint64_t)PULSAR_N_INDEXER_HEAD_DIM * sizeof(float));
                            if (!index_row_view) {
                                ok = false;
                            } else if (idx_fp4) {
                                ok = pulsar_gpu_dsv4_indexer_qat_pack_tensor(index_row_view,
                                                                           mseq ? ms_cache
                                                                                : g->layer_index_comp_cache[il],
                                                                           index_row,
                                                                           1,
                                                                           PULSAR_N_INDEXER_HEAD_DIM) != 0;
                                pulsar_gpu_tensor_free(index_row_view);
                            } else {
                                ok = pulsar_gpu_dsv4_indexer_qat_tensor(index_row_view,
                                                                      1,
                                                                      PULSAR_N_INDEXER_HEAD_DIM) != 0;
                                pulsar_gpu_tensor_free(index_row_view);
                            }
                            /* plan-33 inc C: boundary-row restore (banked emit). */
                            if (ok) ok = gpu_graph_emit_keep_restore(g, il,
                                    mseq ? (uint32_t)bank
                                         : (g->banks.n_banks ? g->banks.cur_bank : 0u),
                                    index_row, 1, true);
                        }
                        if (ok && emit) (*n_index_slot)++;
                        if (index_counts) index_counts[t] = *n_index_slot;
                        pulsar_gpu_tensor_free(ms_cache);
                        pulsar_gpu_tensor_free(ms_st_sc);
                        pulsar_gpu_tensor_free(ms_st_kv);
                        pulsar_gpu_tensor_free(sc_view);
                        pulsar_gpu_tensor_free(kv_view);
                    }
                }
            }
        }
        if (ratio == 4) PULSAR_CUDA_PROFILE_ATTN_STAGE("indexer_setup");

        if (ok && !zero_prefix && n_tokens <= g->raw_cap) {
            const uint32_t n_raw = gpu_graph_raw_span_for_batch(g, pos0, n_tokens);
            /* See the raw-only branch above: batched mixed attention also
             * consumes a logical raw window, linearized out of the ring. */
            const uint32_t raw_start = gpu_graph_raw_start_for_span(g,
                                                                      pos0 + n_tokens - 1u,
                                                                      n_raw);
            double index_stage_t0 = 0.0;

            ok = pulsar_gpu_store_raw_kv_batch_tensor(mseq ? gpu_graph_bank_raw_pool(g, il)
                                                        : g->layer_raw_cache[il],
                                                     g->batch_kv,
                                                     g->raw_cap,
                                                     pos0,
                                                     n_tokens,
                                                     PULSAR_N_HEAD_DIM,
                                                     (uint32_t)gpu_graph_raw_f16_enabled(),
                                                     mseq ? g->batch_positions : NULL,
                                                     mseq ? g->batch_seq_id : NULL,
                                                     mseq ? nb : 1) != 0;
            if (ok && ratio == 4 && n_comp > PULSAR_N_INDEXER_TOP_K) {
                const float index_scale = 1.0f / sqrtf((float)(PULSAR_N_INDEXER_HEAD_DIM * PULSAR_N_INDEXER_HEAD));
                /* PULSAR_PREFILL_SLICE: run [score -> top-k -> indexed attention]
                 * over <=slice-token spans so indexer_scores only ever holds
                 * one span of rows.  Per-token math is keyed on the absolute
                 * position (pos0+t) and the raw window/comp visibility are
                 * recomputed per span exactly like a smaller chunk, so slicing
                 * is bit-identical; slice==0 (unset) is one full-chunk span
                 * with pointer-identical arguments. */
                const uint32_t slice = gpu_graph_prefill_slice();
                const uint32_t span = (slice != 0u && slice < n_tokens) ? slice : n_tokens;
                const uint64_t iq_row = (uint64_t)PULSAR_N_INDEXER_HEAD * PULSAR_N_INDEXER_HEAD_DIM;
                for (uint32_t s0 = 0; ok && s0 < n_tokens; s0 += span) {
                    const uint32_t sn = n_tokens - s0 < span ? n_tokens - s0 : span;
                    const uint32_t spos0 = pos0 + s0;
                    const uint32_t s_n_raw = gpu_graph_raw_span_for_batch(g, spos0, sn);
                    const uint32_t s_raw_start = gpu_graph_raw_start_for_span(g,
                                                                                spos0 + sn - 1u,
                                                                                s_n_raw);
                    pulsar_gpu_tensor *iq_view = pulsar_gpu_tensor_view(g->batch_indexer_q,
                            (uint64_t)s0 * iq_row * sizeof(float),
                            (uint64_t)sn * iq_row * sizeof(float));
                    pulsar_gpu_tensor *iw_view = pulsar_gpu_tensor_view(g->batch_indexer_weights,
                            (uint64_t)s0 * PULSAR_N_INDEXER_HEAD * sizeof(float),
                            (uint64_t)sn * PULSAR_N_INDEXER_HEAD * sizeof(float));
                    pulsar_gpu_tensor *sq_view = pulsar_gpu_tensor_view(g->batch_q,
                            (uint64_t)s0 * q_dim * sizeof(float),
                            (uint64_t)sn * q_dim * sizeof(float));
                    pulsar_gpu_tensor *sh_view = pulsar_gpu_tensor_view(g->batch_heads,
                            (uint64_t)s0 * q_dim * sizeof(float),
                            (uint64_t)sn * q_dim * sizeof(float));
                    /* Multiseq: per-span descriptor views (rows s0..s0+sn),
                     * whole-pool cache operands, packed comp reads (decode
                     * semantics — bit-identical to the f32 shadow by the
                     * pack-dot contract in pulsar_cuda_attention.cu), and the
                     * scalar raw span/start ignored (per-row derived). */
                    pulsar_gpu_tensor *sp_view = mseq
                        ? pulsar_gpu_tensor_view(g->batch_positions,
                                              (uint64_t)s0 * sizeof(int32_t),
                                              (uint64_t)sn * sizeof(int32_t))
                        : NULL;
                    pulsar_gpu_tensor *ss_view = mseq
                        ? pulsar_gpu_tensor_view(g->batch_seq_id,
                                              (uint64_t)s0 * sizeof(int32_t),
                                              (uint64_t)sn * sizeof(int32_t))
                        : NULL;
                    ok = iq_view && iw_view && sq_view && sh_view &&
                         (!mseq || (sp_view && ss_view));
                    if (ok && index_stage_profile) {
                        ok = gpu_graph_indexer_stage_profile_boundary(NULL,
                                                                        il,
                                                                        spos0,
                                                                        sn,
                                                                        n_comp,
                                                                        &index_stage_t0);
                    }
                    if (ok) ok = pulsar_gpu_indexer_scores_decode_batch_tensor(g->indexer_scores,
                                                                              iq_view,
                                                                              iw_view,
                                                                              mseq ? gpu_graph_bank_index_comp_pool(g, il)
                                                                                   : g->layer_index_comp_cache[il],
                                                                              n_comp,
                                                                              sn,
                                                                              spos0,
                                                                              PULSAR_N_INDEXER_HEAD,
                                                                              PULSAR_N_INDEXER_HEAD_DIM,
                                                                              ratio,
                                                                              index_scale,
                                                                              sp_view, ss_view,
                                                                              mseq ? gpu_graph_bank_index_comp_bases(g, il) : NULL,
                                                                              mseq ? g->layer_comp_cap[il] : 0,
                                                                              mseq ? nb : 1) != 0;
                    if (ok && index_stage_profile) {
                        ok = gpu_graph_indexer_stage_profile_boundary("score",
                                                                        il,
                                                                        spos0,
                                                                        sn,
                                                                        n_comp,
                                                                        &index_stage_t0);
                    }
                    if (ok) {
                        gpu_graph_debug_dump_tensor("indexer_scores",
                                                      g->indexer_scores,
                                                      (uint64_t)n_comp * sn,
                                                      il,
                                                      spos0);
                    }
                    if (ok) {
                        ok = pulsar_gpu_indexer_topk_tensor(g->comp_selected,
                                                           g->indexer_scores,
                                                           n_comp,
                                                           sn,
                                                           PULSAR_N_INDEXER_TOP_K) != 0;
                        if (ok && index_stage_profile) {
                            ok = gpu_graph_indexer_stage_profile_boundary("topk",
                                                                            il,
                                                                            spos0,
                                                                            sn,
                                                                            n_comp,
                                                                            &index_stage_t0);
                        }
                        if (ok) {
                            gpu_graph_debug_dump_i32_tensor("indexer_topk",
                                                              g->comp_selected,
                                                              (uint64_t)sn * PULSAR_N_INDEXER_TOP_K,
                                                              il,
                                                              spos0);
                        }
                    }
                    if (ok) {
                        ok = pulsar_gpu_attention_indexed_mixed_batch_heads_tensor(sh_view,
                                                                                  model->map,
                                                                                  model->size,
                                                                                  layer->attn_sinks->abs_offset,
                                                                                  sq_view,
                                                                                  mseq ? gpu_graph_bank_raw_pool(g, il)
                                                                                       : g->layer_raw_cache[il],
                                                                                  mseq ? gpu_graph_bank_attn_comp_pool(g, il)
                                                                                       : gpu_graph_attn_comp_read_cache(g, il, n_comp),
                                                                                  0u, 0u, /* comp f32 (f16/fp8 comp modes removed) */
                                                                                  mseq ? gpu_graph_attn_comp_cache_is_pack()
                                                                                       : 0 /* shadow is f32 */,
                                                                                  g->comp_selected,
                                                                                  sn,
                                                                                  spos0,
                                                                                  mseq ? 0 : s_n_raw,
                                                                                  g->raw_cap,
                                                                                  mseq ? 0 : s_raw_start,
                                                                                  n_comp,
                                                                                  PULSAR_N_INDEXER_TOP_K,
                                                                                  g->raw_window,
                                                                                  ratio,
                                                                                  PULSAR_N_HEAD,
                                                                                  PULSAR_N_HEAD_DIM,
                                                                                  (uint32_t)gpu_graph_raw_f16_enabled(),
                                                                                  sp_view, ss_view,
                                                                                  mseq ? gpu_graph_bank_attn_comp_bases(g, il) : NULL,
                                                                                  mseq ? g->layer_comp_cap[il] : 0,
                                                                                  mseq ? nb : 1) != 0;
                        if (ok && index_stage_profile) {
                            ok = gpu_graph_indexer_stage_profile_boundary("attention",
                                                                            il,
                                                                            spos0,
                                                                            sn,
                                                                            n_comp,
                                                                            &index_stage_t0);
                        }
                    }
                    pulsar_gpu_tensor_free(ss_view);
                    pulsar_gpu_tensor_free(sp_view);
                    pulsar_gpu_tensor_free(sh_view);
                    pulsar_gpu_tensor_free(sq_view);
                    pulsar_gpu_tensor_free(iw_view);
                    pulsar_gpu_tensor_free(iq_view);
                }
            } else if (ok) {
                ok = pulsar_gpu_attention_decode_mixed_batch_heads_tensor(g->batch_heads,
                                                                         model->map,
                                                                         model->size,
                                                                         layer->attn_sinks->abs_offset,
                                                                         g->batch_q,
                                                                         mseq ? gpu_graph_bank_raw_pool(g, il)
                                                                              : g->layer_raw_cache[il],
                                                                         mseq ? gpu_graph_bank_attn_comp_pool(g, il)
                                                                              : gpu_graph_attn_comp_read_cache(g, il, n_comp),
                                                                         0u, 0u, /* comp f32 (f16/fp8 comp modes removed) */
                                                                         mseq ? gpu_graph_attn_comp_cache_is_pack()
                                                                              : 0 /* shadow is f32 */,
                                                                         NULL,
                                                                         0,
                                                                         n_tokens,
                                                                         pos0,
                                                                         mseq ? 0 : n_raw,
                                                                         g->raw_cap,
                                                                         mseq ? 0 : raw_start,
                                                                         n_comp,
                                                                          g->raw_window,
                                                                          ratio,
                                                                          PULSAR_N_HEAD,
                                                                          PULSAR_N_HEAD_DIM,
                                                                          0,
                                                                          (uint32_t)gpu_graph_raw_f16_enabled(),
                                                                          mseq ? g->batch_positions : NULL,
                                                                          mseq ? g->batch_seq_id : NULL,
                                                                          mseq ? gpu_graph_bank_attn_comp_bases(g, il) : NULL,
                                                                          mseq ? g->layer_comp_cap[il] : 0,
                                                                          mseq ? nb : 1) != 0;
            }
            if (ok) batch_attention_done = true;
        }

        const bool topk_prefill_needed = ratio == 4 && n_comp > PULSAR_N_INDEXER_TOP_K;
        if (ok && zero_prefix && topk_prefill_needed && n_comp != 0) {
            const float index_scale = 1.0f / sqrtf((float)(PULSAR_N_INDEXER_HEAD_DIM * PULSAR_N_INDEXER_HEAD));
            double index_stage_t0 = 0.0;
            if (index_stage_profile) {
                ok = gpu_graph_indexer_stage_profile_boundary(NULL,
                                                                il,
                                                                pos0,
                                                                n_tokens,
                                                                n_comp,
                                                                &index_stage_t0);
            }
            /* PULSAR_PREFILL_SLICE: same span loop as the chunked branch.  The
             * historical pulsar_gpu_indexer_scores_prefill_tensor is exactly the
             * decode-batch entry with pos0 == 0 (zero_prefix means pos0 == 0,
             * same launcher, causal), so a span at offset s0 scores the same
             * per-token values with pos0 = s0.  Attention per span keeps
             * first_raw_pos == 0 by passing n_raw = s0 + sn with raw_start 0. */
            const uint32_t zslice = gpu_graph_prefill_slice();
            const uint32_t zspan = (zslice != 0u && zslice < n_tokens) ? zslice : n_tokens;
            const uint64_t ziq_row = (uint64_t)PULSAR_N_INDEXER_HEAD * PULSAR_N_INDEXER_HEAD_DIM;
            for (uint32_t s0 = 0; ok && s0 < n_tokens; s0 += zspan) {
                const uint32_t sn = n_tokens - s0 < zspan ? n_tokens - s0 : zspan;
                const uint32_t spos0 = pos0 + s0;
                pulsar_gpu_tensor *iq_view = pulsar_gpu_tensor_view(g->batch_indexer_q,
                        (uint64_t)s0 * ziq_row * sizeof(float),
                        (uint64_t)sn * ziq_row * sizeof(float));
                pulsar_gpu_tensor *iw_view = pulsar_gpu_tensor_view(g->batch_indexer_weights,
                        (uint64_t)s0 * PULSAR_N_INDEXER_HEAD * sizeof(float),
                        (uint64_t)sn * PULSAR_N_INDEXER_HEAD * sizeof(float));
                pulsar_gpu_tensor *sq_view = pulsar_gpu_tensor_view(g->batch_q,
                        (uint64_t)s0 * q_dim * sizeof(float),
                        (uint64_t)sn * q_dim * sizeof(float));
                pulsar_gpu_tensor *sh_view = pulsar_gpu_tensor_view(g->batch_heads,
                        (uint64_t)s0 * q_dim * sizeof(float),
                        (uint64_t)sn * q_dim * sizeof(float));
                ok = iq_view && iw_view && sq_view && sh_view;
                if (ok) ok = pulsar_gpu_indexer_scores_decode_batch_tensor(g->indexer_scores,
                                                                          iq_view,
                                                                          iw_view,
                                                                          g->layer_index_comp_cache[il],
                                                                          n_comp,
                                                                          sn,
                                                                          spos0,
                                                                          PULSAR_N_INDEXER_HEAD,
                                                                          PULSAR_N_INDEXER_HEAD_DIM,
                                                                          ratio,
                                                                          index_scale,
                                                                          NULL, NULL, NULL, 0, 1) != 0;
                if (ok && index_stage_profile) {
                    ok = gpu_graph_indexer_stage_profile_boundary("score",
                                                                    il,
                                                                    spos0,
                                                                    sn,
                                                                    n_comp,
                                                                    &index_stage_t0);
                }
                if (ok) {
                    gpu_graph_debug_dump_tensor("indexer_scores",
                                                  g->indexer_scores,
                                                  (uint64_t)n_comp * sn,
                                                  il,
                                                  spos0);
                }
                if (ok) {
                    ok = pulsar_gpu_indexer_topk_tensor(g->comp_selected,
                                                       g->indexer_scores,
                                                       n_comp,
                                                       sn,
                                                       PULSAR_N_INDEXER_TOP_K) != 0;
                    if (ok && index_stage_profile) {
                        ok = gpu_graph_indexer_stage_profile_boundary("topk",
                                                                        il,
                                                                        spos0,
                                                                        sn,
                                                                        n_comp,
                                                                        &index_stage_t0);
                    }
                    if (ok) {
                        gpu_graph_debug_dump_i32_tensor("indexer_topk",
                                                          g->comp_selected,
                                                          (uint64_t)sn * PULSAR_N_INDEXER_TOP_K,
                                                          il,
                                                          spos0);
                    }
                }
                if (ok) {
                    ok = pulsar_gpu_attention_indexed_mixed_batch_heads_tensor(sh_view,
                                                                              model->map,
                                                                              model->size,
                                                                              layer->attn_sinks->abs_offset,
                                                                              sq_view,
                                                                              g->layer_raw_cache[il],
                                                                              gpu_graph_attn_comp_read_cache(g, il, n_comp),
                                                                              0u, 0u, /* comp f32 (f16/fp8 comp modes removed) */
                                                                              0 /* shadow is f32 */,
                                                                              g->comp_selected,
                                                                              sn,
                                                                              spos0,
                                                                              s0 + sn,
                                                                              g->raw_cap,
                                                                              0,
                                                                              n_comp,
                                                                              PULSAR_N_INDEXER_TOP_K,
                                                                              g->raw_window,
                                                                              ratio,
                                                                              PULSAR_N_HEAD,
                                                                              PULSAR_N_HEAD_DIM,
                                                                              (uint32_t)gpu_graph_raw_f16_enabled(),
                                                                              NULL, NULL, NULL, 0, 1) != 0;
                    if (ok && index_stage_profile) {
                        ok = gpu_graph_indexer_stage_profile_boundary("attention",
                                                                        il,
                                                                        spos0,
                                                                        sn,
                                                                        n_comp,
                                                                        &index_stage_t0);
                    }
                }
                pulsar_gpu_tensor_free(sh_view);
                pulsar_gpu_tensor_free(sq_view);
                pulsar_gpu_tensor_free(iw_view);
                pulsar_gpu_tensor_free(iq_view);
            }
            if (ok) batch_attention_done = true;
        }
        if (ok && zero_prefix && !topk_prefill_needed && n_comp != 0) {
            ok = pulsar_gpu_attention_prefill_static_mixed_heads_tensor(g->batch_heads,
                                                                       model->map,
                                                                       model->size,
                                                                       layer->attn_sinks->abs_offset,
                                                                       g->batch_q,
                                                                       g->batch_kv,
                                                                       gpu_graph_attn_comp_read_cache(g, il, n_comp),
                                                                       0u, 0u, /* comp f32 (f16/fp8 comp modes removed) */
                                                                       n_tokens,
                                                                       n_comp,
                                                                       g->raw_window,
                                                                       ratio,
                                                                       PULSAR_N_HEAD,
                                                                       PULSAR_N_HEAD_DIM,
                                                                       0 /* batch_kv is f32 */) != 0;
            if (ok) batch_attention_done = true;
        }
    }

    if (ok && mseq && !batch_attention_done) {
        /* Every multiseq-legal shape is handled by the banked branches above;
         * the fallback below is classic single-session (shadow reads, scalar
         * spans).  Reaching it banked would silently compute the wrong rows. */
        fprintf(stderr, "pulsar: multiseq layer batch rejected: unsupported shape "
                        "(layer %u n_tokens=%u raw_cap=%u)\n",
                il, n_tokens, g->raw_cap);
        ok = false;
    }
    if (ok && !raw_batch_attention && !batch_attention_done) {
        uint32_t raw_prefix_tokens = 0;
        if (zero_prefix && ratio != 0 && n_tokens <= g->raw_cap && comp_counts != NULL) {
            while (raw_prefix_tokens < n_tokens && comp_counts[raw_prefix_tokens] == 0u) {
                raw_prefix_tokens++;
            }
        }

        if (raw_prefix_tokens != 0) {
            ok = pulsar_gpu_attention_prefill_raw_heads_tensor(g->batch_heads,
                                                              model->map,
                                                              model->size,
                                                              layer->attn_sinks->abs_offset,
                                                              g->batch_q,
                                                              g->batch_kv,
                                                              raw_prefix_tokens,
                                                              g->raw_window,
                                                              PULSAR_N_HEAD,
                                                              PULSAR_N_HEAD_DIM,
                                                              0 /* batch_kv is f32 */) != 0;
        }
        if (raw_prefix_tokens < n_tokens) {
            for (uint32_t t = raw_prefix_tokens; ok && t < n_tokens; t++) {
                const uint32_t pos = pos0 + t;
                const uint32_t n_raw = gpu_graph_raw_span_for_batch(g, pos, 1);
                const uint32_t raw_start = gpu_graph_raw_start_for_span(g, pos, n_raw);
                const uint32_t cur_comp = comp_counts ? comp_counts[t] : 0u;
                const uint32_t cur_index = index_counts ? index_counts[t] : 0u;
                uint32_t n_selected = 0;
                pulsar_gpu_tensor *comp_mask = NULL;

                if (ratio == 4 && cur_comp > PULSAR_N_INDEXER_TOP_K) {
                    const float index_scale = 1.0f / sqrtf((float)(PULSAR_N_INDEXER_HEAD_DIM * PULSAR_N_INDEXER_HEAD));
                    pulsar_gpu_tensor *indexer_q_view = gpu_graph_tensor_row_view(
                            g->batch_indexer_q, t, (uint64_t)PULSAR_N_INDEXER_HEAD * PULSAR_N_INDEXER_HEAD_DIM);
                    pulsar_gpu_tensor *indexer_w_view = gpu_graph_tensor_row_view(
                            g->batch_indexer_weights, t, PULSAR_N_INDEXER_HEAD);
                    ok = indexer_q_view && indexer_w_view &&
                         pulsar_gpu_indexer_score_one_tensor(g->indexer_scores,
                                                            indexer_q_view,
                                                            indexer_w_view,
                                                            g->layer_index_comp_cache[il],
                                                            cur_index,
                                                            PULSAR_N_INDEXER_HEAD,
                                                            PULSAR_N_INDEXER_HEAD_DIM,
                                                            index_scale) != 0 &&
                         pulsar_gpu_indexer_topk_tensor(g->comp_selected,
                                                       g->indexer_scores,
                                                       cur_index,
                                                       1,
                                                       PULSAR_N_INDEXER_TOP_K) != 0 &&
                         pulsar_gpu_dsv4_topk_mask_tensor(g->comp_mask,
                                                         g->comp_selected,
                                                         cur_index,
                                                         1,
                                                         PULSAR_N_INDEXER_TOP_K) != 0;
                    pulsar_gpu_tensor_free(indexer_w_view);
                    pulsar_gpu_tensor_free(indexer_q_view);
                    if (ok) {
                        comp_mask = g->comp_mask;
                        n_selected = PULSAR_N_INDEXER_TOP_K < cur_index
                            ? PULSAR_N_INDEXER_TOP_K
                            : cur_index;
                    }
                }

                pulsar_gpu_tensor *q_view = gpu_graph_tensor_row_view(g->batch_q, t, q_dim);
                pulsar_gpu_tensor *kv_cache_view = gpu_graph_tensor_row_view(g->batch_kv, t, PULSAR_N_HEAD_DIM);
                pulsar_gpu_tensor *heads_view = gpu_graph_tensor_row_view(g->batch_heads, t, q_dim);
                ok = ok && q_view && kv_cache_view && heads_view;
                if (ok && !zero_prefix) {
                    ok = pulsar_gpu_store_raw_kv_tensor(g->layer_raw_cache[il],
                                                       kv_cache_view,
                                                       g->raw_cap,
                                                       pos % g->raw_cap,
                                                       PULSAR_N_HEAD_DIM,
                                                       (uint32_t)gpu_graph_raw_f16_enabled()) != 0;
                }
                if (ok && comp_mask != NULL && n_selected != 0) {
                    ok = pulsar_gpu_attention_indexed_mixed_batch_heads_tensor(heads_view,
                                                                              model->map,
                                                                              model->size,
                                                                              layer->attn_sinks->abs_offset,
                                                                              q_view,
                                                                              g->layer_raw_cache[il],
                                                                              gpu_graph_attn_comp_read_cache(g, il, cur_comp),
                                                                              0u, 0u, /* comp f32 (f16/fp8 comp modes removed) */
                                                                              0 /* shadow is f32 */,
                                                                              g->comp_selected,
                                                                              1,
                                                                              pos,
                                                                              n_raw,
                                                                              g->raw_cap,
                                                                              raw_start,
                                                                              cur_comp,
                                                                              n_selected,
                                                                              g->raw_window,
                                                                              ratio,
                                                                              PULSAR_N_HEAD,
                                                                              PULSAR_N_HEAD_DIM,
                                                                              (uint32_t)gpu_graph_raw_f16_enabled(),
                                                                              NULL, NULL, NULL, 0, 1) != 0;
                } else if (ok) {
                    ok = pulsar_gpu_attention_decode_heads_tensor(heads_view,
                                                                 model->map,
                                                                 model->size,
                                                                 layer->attn_sinks->abs_offset,
                                                                 q_view,
                                                                 g->layer_raw_cache[il],
                                                                 n_raw,
                                                                 g->raw_cap,
                                                                 raw_start,
                                                                 cur_comp ? gpu_graph_attn_comp_read_cache(g, il, cur_comp) : NULL,
                                                                 0u, 0u, /* comp f32 (f16/fp8 comp modes removed) */
                                                                 0 /* shadow is f32 */,
                                                                 cur_comp,
                                                                 comp_mask,
                                                                 n_selected,
                                                                 PULSAR_N_HEAD,
                                                                 PULSAR_N_HEAD_DIM,
                                                                 (uint32_t)gpu_graph_raw_f16_enabled()) != 0;
                }
                pulsar_gpu_tensor_free(heads_view);
                pulsar_gpu_tensor_free(kv_cache_view);
                pulsar_gpu_tensor_free(q_view);
            }
        }
    }
    PULSAR_CUDA_PROFILE_ATTN_STAGE("attention");

    if (ok) {
        gpu_graph_debug_dump_tensor("kqv_out", g->batch_heads,
                                      (uint64_t)n_tokens * q_dim, il, pos0);
    }
    if (ok) ok = pulsar_gpu_rope_tail_tensor(g->batch_heads,
                                            n_tokens,
                                            PULSAR_N_HEAD,
                                            PULSAR_N_HEAD_DIM,
                                            PULSAR_N_ROT,
                                            pos0,
                                            compressed ? (uint32_t)PULSAR_ROPE_ORIG_CTX : 0,
                                            true,
                                            freq_base,
                                            freq_scale,
                                            ext_factor,
                                            attn_factor,
                                            PULSAR_ROPE_YARN_BETA_FAST,
                                            PULSAR_ROPE_YARN_BETA_SLOW,
                                            mseq ? g->batch_positions : NULL) != 0;
    if (ok) {
        gpu_graph_debug_dump_tensor("kqv_back", g->batch_heads,
                                      (uint64_t)n_tokens * q_dim, il, pos0);
    }
    PULSAR_CUDA_PROFILE_ATTN_STAGE("inv_rope");
    if (ok) {
        ok = pulsar_gpu_attention_output_batch_tensor(g->batch_attn_out,
                                                   g->batch_attn_low,
                                                   model->map,
                                                   model->size,
                                                   layer->attn_output_a->abs_offset,
                                                   layer->attn_output_b->abs_offset,
                                                   group_dim,
                                                   rank,
                                                   n_groups,
                                                   PULSAR_N_EMBD,
                                                   g->batch_heads,
                                                   n_tokens) != 0;
    }
    if (ok) {
        gpu_graph_debug_dump_tensor("attn_low", g->batch_attn_low,
                                      (uint64_t)n_tokens * n_groups * rank,
                                      il,
                                      pos0);
    }
    if (ok) {
        gpu_graph_debug_dump_tensor("attn_out", g->batch_attn_out,
                                      (uint64_t)n_tokens * PULSAR_N_EMBD, il, pos0);
    }
    PULSAR_CUDA_PROFILE_ATTN_STAGE("output_proj");
    if (ok && gpu_graph_directional_steering_attn_enabled(g)) {
        ok = gpu_graph_apply_directional_steering_attn(g, g->batch_attn_out, il, n_tokens);
    }
    if (ok) {
        ok = pulsar_gpu_hc_expand_split_tensor(after_attn_hc_view,
                                            g->batch_attn_out,
                                            g->batch_cur_hc,
                                            hc_split_view,
                                            PULSAR_N_EMBD,
                                            PULSAR_N_HC) != 0;
    }
    if (ok) {
        gpu_graph_debug_dump_hc_tensor("hc_attn_post", g->batch_after_attn_hc,
                                      (uint64_t)n_tokens * hc_dim, il, pos0);
    }
    PULSAR_CUDA_PROFILE_ATTN_STAGE("hc_post");
    pulsar_gpu_mxfp8_act_cache_disarm();
    pulsar_gpu_tensor_free(after_attn_hc_view);
    pulsar_gpu_tensor_free(attn_cur_view);
    pulsar_gpu_tensor_free(hc_split_view);
    pulsar_gpu_tensor_free(hc_mix_view);
    free(index_counts);
    free(comp_counts);
#undef PULSAR_CUDA_PROFILE_ATTN_STAGE
#undef PULSAR_CUDA_PROFILE_Q_STAGE
    return ok;
}



/* Encode the batched prefill FFN half: HC pre/norm, shared expert, routed
 * experts, sum, and HC post. */
bool gpu_graph_encode_layer_ffn_batch(
        pulsar_gpu_graph  *g,
        const pulsar_model        *model,
        const pulsar_layer_weights *layer,
        uint32_t                il,
        uint32_t                pos0,
        uint32_t                n_tokens) {
    if (n_tokens == 0 || n_tokens > g->prefill_cap) return false;

    const uint64_t hc_dim = (uint64_t)PULSAR_N_HC * PULSAR_N_EMBD;
    const uint64_t mix_hc = 2ull * PULSAR_N_HC + (uint64_t)PULSAR_N_HC * PULSAR_N_HC;
    const uint64_t shared_dim = layer->ffn_gate_shexp->dim[1];
    const uint64_t expert_in_dim = layer->ffn_gate_exps->dim[0];
    const uint64_t down_in_dim = layer->ffn_down_exps->dim[0];
    const uint64_t routed_out_dim = layer->ffn_down_exps->dim[1];
    uint64_t gate_expert_bytes = 0, gate_row_bytes = 0;
    uint64_t down_expert_bytes = 0, down_row_bytes = 0;
    if (!routed_expert_gate_down_layout(layer->ffn_gate_exps, layer->ffn_down_exps,
                                        &gate_expert_bytes, &gate_row_bytes,
                                        &down_expert_bytes, &down_row_bytes)) {
        return false;
    }
    const bool layer_stage_profile = gpu_graph_layer_stage_profile_enabled(il);
    double layer_stage_t0 = layer_stage_profile ? now_sec() : 0.0;
#define PULSAR_CUDA_PROFILE_FFN_STAGE(name) do { \
        if (ok && layer_stage_profile) { \
            ok = gpu_graph_layer_stage_profile_boundary("ffn", (name), il, pos0, n_tokens, &layer_stage_t0); \
        } \
    } while (0)

    pulsar_gpu_tensor *hc_mix_view = pulsar_gpu_tensor_view(
            g->batch_hc_mix, 0, (uint64_t)n_tokens * mix_hc * sizeof(float));
    pulsar_gpu_tensor *hc_split_view = pulsar_gpu_tensor_view(
            g->batch_hc_split, 0, (uint64_t)n_tokens * mix_hc * sizeof(float));
    pulsar_gpu_tensor *ffn_cur_view = pulsar_gpu_tensor_view(
            g->batch_ffn_cur, 0, (uint64_t)n_tokens * PULSAR_N_EMBD * sizeof(float));
    pulsar_gpu_tensor *next_hc_view = pulsar_gpu_tensor_view(
            g->batch_next_hc, 0, (uint64_t)n_tokens * hc_dim * PULSAR_HC_ELT_SIZE);   /* carrier */
    bool ok = hc_mix_view && hc_split_view && ffn_cur_view && next_hc_view;
    const bool fuse_hc_norm = PULSAR_N_HC == 4 &&
                              !gpu_graph_use_reference_hc_decode() &&
                              gpu_graph_enable_batch_hc_norm_fusion();
    if (ok) ok = pulsar_gpu_rms_norm_plain_rows_tensor(g->batch_flat_hc,
                                                      g->batch_after_attn_hc,
                                                      (uint32_t)hc_dim,
                                                      n_tokens,
                                                      PULSAR_RMS_EPS) != 0;
    if (ok) ok = gpu_graph_matmul_plain_tensor(hc_mix_view,
                                              model,
                                              layer->hc_ffn_fn,
                                             hc_dim,
                                             mix_hc,
                                             g->batch_flat_hc,
                                             n_tokens) != 0;
    if (gpu_graph_use_reference_hc_decode()) {
        if (ok) ok = pulsar_gpu_hc_split_sinkhorn_tensor(hc_split_view,
                                                        hc_mix_view,
                                                        model->map,
                                                        model->size,
                                                        layer->hc_ffn_scale->abs_offset,
                                                        layer->hc_ffn_base->abs_offset,
                                                        PULSAR_N_HC,
                                                        PULSAR_N_HC_SINKHORN_ITER,
                                                        PULSAR_HC_EPS) != 0;
        if (ok) ok = pulsar_gpu_hc_weighted_sum_split_tensor(ffn_cur_view,
                                                            g->batch_after_attn_hc,
                                                            hc_split_view,
                                                            PULSAR_N_EMBD,
                                                            PULSAR_N_HC) != 0;
    } else if (fuse_hc_norm) {
        if (ok) ok = pulsar_gpu_hc_split_weighted_sum_norm_tensor(ffn_cur_view,
                                                                 g->batch_ffn_norm,
                                                                 hc_split_view,
                                                                 hc_mix_view,
                                                                 g->batch_after_attn_hc,
                                                                 model->map,
                                                                 model->size,
                                                                 layer->hc_ffn_scale->abs_offset,
                                                                 layer->hc_ffn_base->abs_offset,
                                                                 layer->ffn_norm->abs_offset,
                                                                 PULSAR_N_EMBD,
                                                                 PULSAR_N_HC,
                                                                 PULSAR_N_HC_SINKHORN_ITER,
                                                                 PULSAR_HC_EPS,
                                                                 PULSAR_RMS_EPS) != 0;
    } else {
        if (ok) ok = pulsar_gpu_hc_split_weighted_sum_tensor(ffn_cur_view,
                                                            hc_split_view,
                                                            hc_mix_view,
                                                            g->batch_after_attn_hc,
                                                            model->map,
                                                            model->size,
                                                            layer->hc_ffn_scale->abs_offset,
                                                            layer->hc_ffn_base->abs_offset,
                                                            PULSAR_N_EMBD,
                                                            PULSAR_N_HC,
                                                            PULSAR_N_HC_SINKHORN_ITER,
                                                            PULSAR_HC_EPS) != 0;
    }
    if (ok) {
        gpu_graph_debug_dump_tensor("hc_ffn_pre", g->batch_ffn_cur,
                                      (uint64_t)n_tokens * PULSAR_N_EMBD, il, pos0);
    }
    PULSAR_CUDA_PROFILE_FFN_STAGE("hc_pre");
    if (ok && !fuse_hc_norm) {
        ok = pulsar_gpu_rms_norm_weight_rows_tensor(g->batch_ffn_norm,
                                                  g->batch_ffn_cur,
                                                  model->map,
                                                  model->size,
                                                  layer->ffn_norm->abs_offset,
                                                  PULSAR_N_EMBD,
                                                  n_tokens,
                                                  PULSAR_RMS_EPS) != 0;
    }
    if (ok) {
        gpu_graph_debug_dump_tensor("ffn_norm", g->batch_ffn_norm,
                                      (uint64_t)n_tokens * PULSAR_N_EMBD, il, pos0);
    }
    PULSAR_CUDA_PROFILE_FFN_STAGE("norm");
    if (ok) ok = gpu_graph_matmul_plain_tensor(g->batch_router_logits,
                                              model,
                                              layer->ffn_gate_inp,
                                             PULSAR_N_EMBD,
                                             PULSAR_N_EXPERT,
                                             g->batch_ffn_norm,
                                             n_tokens) != 0;

    if (ok) ok = pulsar_gpu_router_select_batch_tensor(g->batch_router_selected,
                                                      g->batch_router_weights,
                                                      g->batch_router_probs,
                                                      model->map,
                                                      model->size,
                                                      layer->ffn_exp_probs_b ? layer->ffn_exp_probs_b->abs_offset : 0,
                                                      layer->ffn_gate_tid2eid ? layer->ffn_gate_tid2eid->abs_offset : 0,
                                                      layer->ffn_gate_tid2eid ? (uint32_t)layer->ffn_gate_tid2eid->dim[1] : 0,
                                                      0,
                                                      0,
                                                      layer->ffn_exp_probs_b != NULL,
                                                      layer->ffn_gate_tid2eid != NULL,
                                                      g->batch_router_logits,
                                                      g->prefill_tokens,
                                                      PULSAR_N_EXPERT,
                                                      PULSAR_N_EXPERT_USED,
                                                      PULSAR_EXPERT_WEIGHT_SCALE,
                                                      n_tokens) != 0;
    if (ok) {
        gpu_graph_debug_dump_tensor("ffn_moe_logits", g->batch_router_logits,
                                      (uint64_t)n_tokens * PULSAR_N_EXPERT, il, pos0);
        gpu_graph_debug_dump_tensor("ffn_moe_probs", g->batch_router_probs,
                                      (uint64_t)n_tokens * PULSAR_N_EXPERT, il, pos0);
        gpu_graph_debug_dump_i32_tensor("ffn_moe_topk", g->batch_router_selected,
                                          (uint64_t)n_tokens * PULSAR_N_EXPERT_USED, il, pos0);
        gpu_graph_debug_dump_tensor("ffn_moe_weights_scaled", g->batch_router_weights,
                                      (uint64_t)n_tokens * PULSAR_N_EXPERT_USED, il, pos0);
    }
    PULSAR_CUDA_PROFILE_FFN_STAGE("router");

    const bool keep_ffn_out = gpu_graph_needs_ffn_out(g, il, pos0);

#define PULSAR_CUDA_ENCODE_PREFILL_SHARED_EXPERT() do { \
        if (ok) ok = gpu_graph_matmul_mxfp8_named_tensor("shared_gate", \
                                                          il, \
                                                          pos0, \
                                                          g->batch_shared_gate, \
                                                          model, \
                                                          layer->ffn_gate_shexp, \
                                                          PULSAR_N_EMBD, \
                                                          shared_dim, \
                                                          g->batch_ffn_norm, \
                                                          n_tokens); \
        if (ok) ok = gpu_graph_matmul_mxfp8_named_tensor("shared_up", \
                                                          il, \
                                                          pos0, \
                                                          g->batch_shared_up, \
                                                          model, \
                                                          layer->ffn_up_shexp, \
                                                          PULSAR_N_EMBD, \
                                                          shared_dim, \
                                                          g->batch_ffn_norm, \
                                                          n_tokens); \
        PULSAR_CUDA_PROFILE_FFN_STAGE("shared_gate_up"); \
        if (ok) ok = pulsar_gpu_swiglu_tensor(g->batch_shared_mid, \
                                             g->batch_shared_gate, \
                                             g->batch_shared_up, \
                                             (uint32_t)((uint64_t)n_tokens * shared_dim), \
                                             PULSAR_SWIGLU_CLAMP_EXP, \
                                             1.0f) != 0; \
        if (ok) ok = gpu_graph_matmul_mxfp8_named_tensor("shared_down", \
                                                                              il, \
                                                                              pos0, \
                                                                              g->batch_shared_out, \
                                                                              model, \
                                                                              layer->ffn_down_shexp, \
                                                                              shared_dim, \
                                                                              PULSAR_N_EMBD, \
                                                                              g->batch_shared_mid, \
                                                                              n_tokens); \
        PULSAR_CUDA_PROFILE_FFN_STAGE("shared_down"); \
        if (ok) { \
            gpu_graph_debug_dump_tensor("ffn_shexp", g->batch_shared_out, \
                                          (uint64_t)n_tokens * PULSAR_N_EMBD, il, pos0); \
        } \
    } while (0)

    if (ok) {
        ok = pulsar_gpu_routed_moe_batch_tensor(g->batch_routed_out,
                                               g->batch_routed_gate,
                                               g->batch_routed_up,
                                               g->batch_routed_mid,
                                               g->batch_routed_down,
                                               tensor_map_base(model, layer->ffn_gate_exps),
                                               tensor_map_size(model, layer->ffn_gate_exps),
                                               layer->ffn_gate_exps->abs_offset,
                                               layer->ffn_up_exps->abs_offset,
                                               layer->ffn_down_exps->abs_offset,
                                               layer->ffn_gate_exps->type,
                                               layer->ffn_down_exps->type,
                                               gate_expert_bytes,
                                               gate_row_bytes,
                                               down_expert_bytes,
                                               down_row_bytes,
                                               (uint32_t)expert_in_dim,
                                               (uint32_t)down_in_dim,
                                               (uint32_t)routed_out_dim,
                                               g->batch_router_selected,
                                               g->batch_router_weights,
                                               pulsar_layer_n_expert(il),
                                               PULSAR_N_EXPERT_USED,
                                               PULSAR_SWIGLU_CLAMP_EXP,
                                               g->batch_ffn_norm,
                                               il,
                                               n_tokens,
                                               &g->batch_routed_mid_is_f16) != 0;
    }
    if (ok) {
        gpu_graph_debug_dump_tensor("ffn_moe_gate_clamped", g->batch_routed_gate,
                                      (uint64_t)n_tokens * PULSAR_N_EXPERT_USED * down_in_dim, il, pos0);
        gpu_graph_debug_dump_tensor("ffn_moe_up_clamped", g->batch_routed_up,
                                      (uint64_t)n_tokens * PULSAR_N_EXPERT_USED * down_in_dim, il, pos0);
    }
    if (ok) {
        const uint64_t routed_mid_elems = (uint64_t)n_tokens * PULSAR_N_EXPERT_USED * down_in_dim;
        if (g->batch_routed_mid_is_f16) {
            gpu_graph_debug_dump_f16_tensor("ffn_moe_weighted_swiglu", g->batch_routed_mid,
                                              routed_mid_elems, il, pos0);
        } else {
            gpu_graph_debug_dump_tensor("ffn_moe_weighted_swiglu", g->batch_routed_mid,
                                          routed_mid_elems, il, pos0);
        }
    }
    if (ok) {
        gpu_graph_debug_dump_tensor("ffn_moe_down", g->batch_routed_down,
                                      (uint64_t)n_tokens * PULSAR_N_EXPERT_USED * PULSAR_N_EMBD, il, pos0);
    }
    if (ok) {
        gpu_graph_debug_dump_tensor("ffn_moe_out", g->batch_routed_out,
                                      (uint64_t)n_tokens * PULSAR_N_EMBD, il, pos0);
    }
    PULSAR_CUDA_PROFILE_FFN_STAGE("routed_moe");
    PULSAR_CUDA_ENCODE_PREFILL_SHARED_EXPERT();
#undef PULSAR_CUDA_ENCODE_PREFILL_SHARED_EXPERT

    if (ok && keep_ffn_out) {
        ok = gpu_graph_ensure_batch_ffn_out(g) &&
             pulsar_gpu_add_tensor(g->batch_ffn_out,
                                  g->batch_shared_out,
                                  g->batch_routed_out,
                                  (uint32_t)((uint64_t)n_tokens * PULSAR_N_EMBD)) != 0;
    }
    if (ok && keep_ffn_out) {
        gpu_graph_debug_dump_tensor("ffn_out", g->batch_ffn_out,
                                      (uint64_t)n_tokens * PULSAR_N_EMBD, il, pos0);
    }
    if (ok && gpu_graph_directional_steering_ffn_enabled(g)) {
        ok = gpu_graph_apply_directional_steering_ffn(g, g->batch_ffn_out, il, n_tokens);
    }
    if (ok && gpu_graph_directional_steering_ffn_enabled(g)) {
        ok = pulsar_gpu_hc_expand_split_tensor(next_hc_view,
                                              g->batch_ffn_out,
                                              g->batch_after_attn_hc,
                                              hc_split_view,
                                              PULSAR_N_EMBD,
                                              PULSAR_N_HC) != 0;
    }
    else if (ok) {
        ok = pulsar_gpu_hc_expand_add_split_tensor(next_hc_view,
                                                  g->batch_routed_out,
                                                  g->batch_shared_out,
                                                  g->batch_after_attn_hc,
                                                  hc_split_view,
                                                  PULSAR_N_EMBD,
                                                  PULSAR_N_HC) != 0;
    }
    if (ok) {
        gpu_graph_debug_dump_hc_tensor("hc_ffn_post", g->batch_next_hc,
                                      (uint64_t)n_tokens * hc_dim, il, pos0);
    }
    PULSAR_CUDA_PROFILE_FFN_STAGE("hc_post");
    pulsar_gpu_tensor_free(next_hc_view);
    pulsar_gpu_tensor_free(ffn_cur_view);
    pulsar_gpu_tensor_free(hc_split_view);
    pulsar_gpu_tensor_free(hc_mix_view);
#undef PULSAR_CUDA_PROFILE_FFN_STAGE
    return ok;
}



/* Encode one complete layer for prefill by chaining attention and FFN batches. */
bool gpu_graph_encode_layer_batch(
        pulsar_gpu_graph  *g,
        const pulsar_model        *model,
        const pulsar_layer_weights *layer,
        uint32_t                il,
        uint32_t                pos0,
        uint32_t                n_tokens) {
    bool ok = gpu_graph_encode_layer_attention_batch(g, model, layer, il, pos0, n_tokens);
    if (!ok) {
        fprintf(stderr, "pulsar: gpu layer %u attention batch encode failed\n", il);
    }
    if (ok) {
        ok = gpu_graph_encode_layer_ffn_batch(g, model, layer, il, pos0, n_tokens);
        if (!ok) {
            fprintf(stderr, "pulsar: gpu layer %u ffn batch encode failed\n", il);
        }
    }
    if (ok) {
        pulsar_gpu_tensor *tmp = g->batch_cur_hc;
        g->batch_cur_hc = g->batch_next_hc;
        g->batch_next_hc = tmp;
    }
    /* Fused spec loop (P2): when armed, capture the drafter's anchor hidden for
     * every batch position at the anchor layers, so the last-accepted position's
     * hidden is available without a replay decode. Off (0) during prefill and
     * plain decode. */
    if (ok && g->dspark_capture_batch_n) {
        for (int slot = 0; slot < 3; slot++) {
            if (il != g->dspark_target_layer_ids[slot]) continue;
            if (!g->dspark_target_h_batch[slot]) break;
            uint32_t cap_n = g->dspark_capture_batch_n;
            if (cap_n > n_tokens) cap_n = n_tokens;
            if (!pulsar_gpu_dspark_hc_mean_reduce_batch(g->dspark_target_h_batch[slot],
                                                     g->batch_cur_hc,
                                                     PULSAR_N_EMBD, PULSAR_N_HC, cap_n)) {
                ok = false;
            }
            break;
        }
    }
    /* Bulk prefill capture for drafter retraining (PULSAR_DSPARK_PREFILL_DUMP):
     * same reduction as the verify capture above, but over EVERY chunk position
     * into the per-layer bulk buffers. Armed only by the prefill path. */
    if (ok && g->dspark_bulk_n) {
        for (int slot = 0; slot < 3; slot++) {
            if (il != g->dspark_target_layer_ids[slot]) continue;
            if (!g->dspark_bulk_h[slot]) break;
            uint32_t cap_n = g->dspark_bulk_n;
            if (cap_n > n_tokens) cap_n = n_tokens;
            if (!pulsar_gpu_dspark_hc_mean_reduce_batch(g->dspark_bulk_h[slot],
                                                     g->batch_cur_hc,
                                                     PULSAR_N_EMBD, PULSAR_N_HC, cap_n)) {
                ok = false;
            }
            break;
        }
    }
    return ok;
}



/* Execute one GPU decode token and read back logits. */
bool gpu_graph_eval_token_raw_swa(
        pulsar_gpu_graph *g,
        const pulsar_model       *model,
        const pulsar_weights     *weights,
        int                    token,
        uint32_t               pos,
        float                 *logits) {

    const bool profile = getenv("PULSAR_CUDA_GRAPH_TOKEN_PROFILE") != NULL;
    const double t0 = profile ? now_sec() : 0.0;

    const int captured = pulsar_gpu_decode_graph_begin();
    bool ok = captured != 0 || pulsar_gpu_begin_commands() != 0;
    /* The split-flush prefix overlap is a direct-submission latency trick; a
     * mid-tape device sync is illegal during graph capture (and unnecessary:
     * the graph replays the whole tape in one launch). */
    if (ok) ok = gpu_graph_encode_token_raw_swa(g, model, weights, token, pos, logits != NULL,
                                                !captured);
    const double t_encoded = profile ? now_sec() : 0.0;
    if (ok) ok = (captured ? pulsar_gpu_decode_graph_end() : pulsar_gpu_end_commands()) != 0;
    const double t_done = profile ? now_sec() : 0.0;

    if (ok && logits) {
        ok = pulsar_gpu_tensor_read(g->logits, 0, logits, (uint64_t)PULSAR_N_VOCAB * sizeof(float)) != 0;
    }
    const double t_read = profile ? now_sec() : 0.0;
    if (profile) {
        fprintf(stderr,
                "pulsar: GPU graph token pos=%u encode=%.3f ms execute=%.3f ms read=%.3f ms total=%.3f ms logits=%d\n",
                pos,
                (t_encoded - t0) * 1000.0,
                (t_done - t_encoded) * 1000.0,
                (t_read - t_done) * 1000.0,
                (t_read - t0) * 1000.0,
                logits != NULL);
    }
    if (!ok) {
        if (pulsar_gpu_synchronize() == 0) {
            fprintf(stderr, "pulsar: GPU synchronize after graph eval failure also failed\n");
        }
    }
    return ok;
}



/* Greedy verifier helper.  Speculative decoding only needs the target model's
 * top token after most accepted draft rows; the full vocabulary row is needed
 * once, for the final committed state that normal sampling will continue from.
 * Keeping intermediate rows device-resident avoids turning verification into a
 * sequence of large CPU readbacks. */
bool gpu_graph_eval_token_raw_swa_top(
        pulsar_gpu_graph *g,
        const pulsar_model       *model,
        const pulsar_weights     *weights,
        int                    token,
        uint32_t               pos,
        int                   *top_id,
        float                 *logits) {
    if (!top_id) return false;

    bool ok = pulsar_gpu_begin_commands() != 0;
    if (ok) ok = gpu_graph_encode_token_raw_swa(g, model, weights,
                                                  token, pos, true, true);
    if (ok) {
        ok = pulsar_gpu_argmax_tensor(g->comp_selected,
                                   g->logits,
                                   PULSAR_N_VOCAB) != 0;
    }
    if (ok) ok = pulsar_gpu_end_commands() != 0;
    if (ok) ok = pulsar_gpu_tensor_read(g->comp_selected, 0, top_id, sizeof(*top_id)) != 0;
    if (ok && logits) {
        ok = pulsar_gpu_tensor_read(g->logits, 0, logits, (uint64_t)PULSAR_N_VOCAB * sizeof(float)) != 0;
    }
    if (!ok) {
        if (pulsar_gpu_synchronize() == 0) {
            fprintf(stderr, "pulsar: GPU synchronize after top-only graph eval failure also failed\n");
        }
    }
    return ok;
}



/* Stage-B no-replay rollback for the fused spec loop: after restoring the
 * pre-batch frontier snapshot, roll ONLY the recurrent compressor/indexer pool
 * state forward through the committed batch positions using the projections
 * saved during the verify batch (spec_comp_*_save). Bit-identical to what a
 * transformer replay would produce: the same per-token update kernels run on
 * the same input rows in the same order -- minus the 43-layer forward. The
 * comp-cache rows and raw KV need no work (the batch already wrote the
 * committed positions' rows from identical state; rejected rows are position-
 * addressed and get overwritten). Counters are set by formula. The pooled-row
 * emit goes to a scratch sink (cache rows are already correct). */
bool gpu_graph_dspark_compressor_rollforward(
        pulsar_gpu_graph  *g,
        const pulsar_model  *model,
        const pulsar_weights *weights,
        uint32_t          pos0,
        uint32_t          n_positions) {
    if (!g || !model || !weights) return false;
    if (n_positions == 0) return true;
    if (n_positions > 17u || !g->spec_comp_scratch_row) return false;
    for (uint32_t il = 0; il < PULSAR_N_LAYER; il++) {
        const uint32_t ratio = pulsar_layer_compress_ratio(il);
        if (ratio == 0) continue;
        const pulsar_layer_weights *layer = &weights->layer[il];
        const uint32_t coff = ratio == 4 ? 2u : 1u;
        const uint32_t comp_width = coff * PULSAR_N_HEAD_DIM;
        const uint32_t index_width = 2u * PULSAR_N_INDEXER_HEAD_DIM;
        const float freq_base = layer_rope_freq_base(il);
        const float freq_scale = layer_rope_freq_scale(il);
        const float ext_factor = PULSAR_ROPE_SCALE_FACTOR > 1.0f ? 1.0f : 0.0f;
        float attn_factor = 1.0f;
        if (ext_factor != 0.0f && freq_scale > 0.0f) {
            attn_factor /= 1.0f + 0.1f * logf(1.0f / freq_scale);
        }
        if (!g->spec_comp_kv_save[il] || !g->spec_comp_sc_save[il]) return false;
        for (uint32_t t = 0; t < n_positions; t++) {
            const uint32_t pos = pos0 + t;
            pulsar_gpu_tensor *kv_view = gpu_graph_tensor_row_view(g->spec_comp_kv_save[il], t, comp_width);
            pulsar_gpu_tensor *sc_view = gpu_graph_tensor_row_view(g->spec_comp_sc_save[il], t, comp_width);
            bool ok = kv_view && sc_view &&
                pulsar_gpu_compressor_update_tensor(kv_view, sc_view,
                        g->layer_attn_state_kv[il], g->layer_attn_state_score[il],
                        g->spec_comp_scratch_row,
                        model->map, model->size,
                        layer->attn_compressor_ape->abs_offset,
                        layer->attn_compressor_ape->type,
                        layer->attn_compressor_norm->abs_offset,
                        layer->attn_compressor_norm->type,
                        PULSAR_N_HEAD_DIM, ratio, pos, 0,
                        PULSAR_N_ROT, (uint32_t)PULSAR_ROPE_ORIG_CTX,
                        freq_base, freq_scale, ext_factor, attn_factor,
                        PULSAR_ROPE_YARN_BETA_FAST, PULSAR_ROPE_YARN_BETA_SLOW,
                        PULSAR_RMS_EPS) != 0;
            pulsar_gpu_tensor_free(sc_view);
            pulsar_gpu_tensor_free(kv_view);
            if (!ok) return false;
            if (ratio == 4 && g->spec_icomp_kv_save[il]) {
                pulsar_gpu_tensor *ikv = gpu_graph_tensor_row_view(g->spec_icomp_kv_save[il], t, index_width);
                pulsar_gpu_tensor *isc = gpu_graph_tensor_row_view(g->spec_icomp_sc_save[il], t, index_width);
                ok = ikv && isc &&
                    pulsar_gpu_compressor_update_tensor(ikv, isc,
                            g->layer_index_state_kv[il], g->layer_index_state_score[il],
                            g->spec_comp_scratch_row,
                            model->map, model->size,
                            layer->indexer_compressor_ape->abs_offset,
                            layer->indexer_compressor_ape->type,
                            layer->indexer_compressor_norm->abs_offset,
                            layer->indexer_compressor_norm->type,
                            PULSAR_N_INDEXER_HEAD_DIM, ratio, pos, 0,
                            PULSAR_N_ROT, (uint32_t)PULSAR_ROPE_ORIG_CTX,
                            freq_base, freq_scale, ext_factor, attn_factor,
                            PULSAR_ROPE_YARN_BETA_FAST, PULSAR_ROPE_YARN_BETA_SLOW,
                            PULSAR_RMS_EPS) != 0;
                pulsar_gpu_tensor_free(isc);
                pulsar_gpu_tensor_free(ikv);
                if (!ok) return false;
            }
        }
        g->layer_n_comp[il] = (pos0 + n_positions) / ratio;
        if (ratio == 4) g->layer_n_index_comp[il] = (pos0 + n_positions) / ratio;
    }
    return true;
}
