#include "pulsar_engine_internal.h"








int pulsar_engine::routed_quant_bits() {
    auto *e = this;
    if (!e) return 0;
    /* Report the routed-expert precision tier actually present, derived from
     * the loaded tensor types (was hardcoded 2, which under-reported the mixed
     * IQ2 + MXFP4/type-40 build as pure 2-bit). Any 4-bit routed format
     * (MXFP4 E2M1 / CUTLASS type-40) anywhere in gate/up/down makes this a
     * 4-bit-tier model; otherwise the 2-bit floor (IQ2_XXS / Q2_K); 0 if no
     * routed experts. The kvstore snapshot-compat guards accept {2,4} and
     * pulsar_engine_model_id() is a compile-time constant, so this is the only
     * model-variant discriminator in the disk-KV key — a value change
     * invalidates old snapshots (one-time re-prefill; fine in dev). */
    int bits = 0;
    for (uint32_t il = 0; il < PULSAR_N_LAYER; il++) {
        const pulsar_tensor *proj[3] = {
            e->weights.layer[il].ffn_gate_exps,
            e->weights.layer[il].ffn_up_exps,
            e->weights.layer[il].ffn_down_exps,
        };
        for (int k = 0; k < 3; k++) {
            const pulsar_tensor *t = proj[k];
            if (!t) continue;
            if (t->type == PULSAR_TENSOR_FP4_E2M1 ||
                t->type == PULSAR_TENSOR_CUTLASS_MXFP4)
                return 4;
            if (bits == 0) bits = 2;
        }
    }
    return bits;
}



bool pulsar_engine::has_output_head() {
    auto *e = this;
    return e && weights_have_output_head(&e->weights);
}





bool pulsar_engine::has_dspark() {
    auto *e = this;
    return e && e->dspark_ready;
}

int pulsar_engine_dspark_draft_tokens(pulsar_engine *e) {
    return e->has_dspark() ? e->dspark_draft_tokens : 0;
}



const pulsar_tokens *pulsar_session::tokens() {
    auto *s = this;
    return s ? &s->checkpoint : NULL;
}









void pulsar_engine::dump_tokens(const pulsar_tokens *tokens) {
    auto *e = this;
    e->vocab.dump_tokens(tokens);  /* the pulsar_vocab member */
}



int pulsar_dump_text_tokenization(const char *model_path, const char *text, FILE *fp) {
    pulsar_model model;
    pulsar_vocab vocab;
    token_vec tokens = {0};

    if (!fp) fp = stdout;
    model_open(&model, model_path, false, false);
    vocab.vocab_load(&model);
    vocab.tokenize_rendered_chat_vocab(text ? text : "", &tokens);

    dump_tokens_fp(fp, &vocab, &tokens);
    token_vec_free(&tokens);
    vocab.vocab_free();
    model_close(&model);
    return 0;
}



static bool imatrix_read_text_file(const char *path, char **out, size_t *len_out) {
    *out = NULL;
    *len_out = 0;
    struct stat st;
    if (stat(path, &st) != 0) {
        fprintf(stderr, "pulsar: failed to stat imatrix dataset %s: %s\n", path, strerror(errno));
        return false;
    }
    if (st.st_size < 0 || (uint64_t)st.st_size > SIZE_MAX - 1) {
        fprintf(stderr, "pulsar: imatrix dataset is too large: %s\n", path);
        return false;
    }
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "pulsar: failed to open imatrix dataset %s: %s\n", path, strerror(errno));
        return false;
    }
    size_t n = (size_t)st.st_size;
    char *buf = (char *)xmalloc(n + 1);
    if (n != 0 && fread(buf, 1, n, fp) != n) {
        fprintf(stderr, "pulsar: failed to read imatrix dataset %s\n", path);
        fclose(fp);
        free(buf);
        return false;
    }
    if (fclose(fp) != 0) {
        fprintf(stderr, "pulsar: failed to close imatrix dataset %s: %s\n", path, strerror(errno));
        free(buf);
        return false;
    }
    buf[n] = '\0';
    *out = buf;
    *len_out = n;
    return true;
}



static char *imatrix_trim_block(char *p, char *end) {
    while (p < end && isspace((unsigned char)*p)) p++;
    while (end > p && isspace((unsigned char)end[-1])) end--;
    *end = '\0';
    return p;
}



int pulsar_engine::collect_imatrix(const char *dataset_path,
                               const char *output_path,
                               int ctx_size,
                               int max_prompts,
                               int max_tokens) {
    auto *e = this;
    if (!e || !dataset_path || !output_path) return 1;
    if (e->backend != PULSAR_BACKEND_CUDA || !e->gpu_ready) {
        fprintf(stderr, "pulsar: imatrix collection currently requires --cuda\n");
        return 1;
    }
    if (ctx_size <= 0) ctx_size = 32768;

    char *dataset = NULL;
    size_t dataset_len = 0;
    if (!imatrix_read_text_file(dataset_path, &dataset, &dataset_len)) return 1;

    const pulsar_model *model = &e->model;
    const pulsar_weights *weights = &e->weights;
    const uint32_t prefill_cap =
        gpu_graph_prefill_cap_for_prompt(ctx_size, e->prefill_chunk);
    const uint32_t raw_cap = gpu_graph_raw_cap_for_context(ctx_size, prefill_cap);

    pulsar_gpu_graph g;
    bool ok = gpu_graph_alloc_raw_cap(&g, weights, &weights->layer[0],
                                        raw_cap, (uint32_t)ctx_size, prefill_cap, false);
    if (!ok) {
        fprintf(stderr, "pulsar: failed to allocate imatrix GPU graph runtime\n");
        free(dataset);
        return 1;
    }
    g.quality = e->quality;

    pulsar_imatrix_collector collector;
    if (!imatrix_collector_init(&collector, prefill_cap, dataset_path)) {
        fprintf(stderr, "pulsar: failed to allocate imatrix collector\n");
        gpu_graph_free(&g);
        free(dataset);
        return 1;
    }

    fprintf(stderr,
            "pulsar: collecting routed-MoE imatrix from %s (model=%s, layers=%u, experts=%u, ctx=%d, chunk=%u)\n",
            dataset_path, PULSAR_MODEL_SHAPE_NAME, PULSAR_N_LAYER, PULSAR_N_EXPERT, ctx_size, prefill_cap);

    int prompts_done = 0;
    int tokens_done = 0;
    char *cursor = dataset;
    const char *marker_lit = "===== DS4_IMATRIX_PROMPT";
    while (*cursor) {
        char *start = cursor;
        char *marker = strstr(cursor, marker_lit);
        if (marker) {
            char *nl = strchr(marker, '\n');
            if (!nl) break;
            start = nl + 1;
        } else if (prompts_done != 0) {
            break;
        }

        char *next = strstr(start, marker_lit);
        char *end = next ? next : dataset + dataset_len;
        char saved = *end;
        char *prompt_text = imatrix_trim_block(start, end);
        if (prompt_text[0] != '\0') {
            token_vec prompt = {0};
            pulsar_tokenize_rendered_chat(e, prompt_text, &prompt);
            if (prompt.len > ctx_size) prompt.len = ctx_size;
            if (max_tokens > 0 && prompt.len > max_tokens - tokens_done) {
                prompt.len = max_tokens - tokens_done;
            }
            if (prompt.len > 0) {
                if (!gpu_graph_reset_prefill_state(&g)) {
                    fprintf(stderr, "pulsar: failed to reset imatrix graph state\n");
                    ok = false;
                } else if ((uint32_t)prompt.len > prefill_cap) {
                    ok = gpu_graph_prefill_chunked_range(&g, model, weights,
                                                           &prompt, 0,
                                                           (uint32_t)prompt.len,
                                                           NULL, false,
                                                           NULL, NULL,
                                                           NULL, NULL,
                                                           &collector,
                                                           NULL, NULL, NULL);
                } else {
                    ok = gpu_graph_prefill_layer_major(&g, model, weights,
                                                         &prompt, 0,
                                                         (uint32_t)prompt.len,
                                                         NULL, false,
                                                         &collector,
                                                         NULL, NULL);
                }
                if (!ok) {
                    fprintf(stderr, "pulsar: imatrix prefill failed at prompt %d\n", prompts_done + 1);
                    token_vec_free(&prompt);
                    *end = saved;
                    break;
                }
                prompts_done++;
                tokens_done += prompt.len;
                if (prompts_done % 10 == 0) {
                    fprintf(stderr,
                            "pulsar: imatrix prompts=%d tokens=%d routes=%llu\r",
                            prompts_done,
                            tokens_done,
                            (unsigned long long)collector.observed_routes);
                    fflush(stderr);
                }
            }
            token_vec_free(&prompt);
        }
        *end = saved;
        if (!next) break;
        cursor = next;
        if (max_prompts > 0 && prompts_done >= max_prompts) break;
        if (max_tokens > 0 && tokens_done >= max_tokens) break;
    }
    fputc('\n', stderr);

    if (ok) {
        ok = imatrix_collector_save(&collector, weights, output_path);
        if (ok) {
            fprintf(stderr,
                    "pulsar: wrote imatrix %s from %d prompts, %d tokens, %llu routed expert observations\n",
                    output_path,
                    prompts_done,
                    tokens_done,
                    (unsigned long long)collector.observed_routes);
        }
    }

    imatrix_collector_free(&collector);
    gpu_graph_free(&g);
    free(dataset);
    return ok ? 0 : 1;
}



int pulsar_engine::generate_argmax(const pulsar_tokens  *prompt,
        int                n_predict,
        int                ctx_size,
        pulsar_token_emit_fn  emit,
        pulsar_generation_done_fn done,
        void              *emit_ud,
        pulsar_session_progress_fn progress,
        void              *progress_ud) {
    auto *e = this;
    const pulsar_model *model = &e->model;
    const pulsar_vocab *vocab = &e->vocab;
    const pulsar_weights *weights = &e->weights;

    if (pulsar_backend_uses_graph(e->backend)) {
        if (!e->gpu_ready) {
            fprintf(stderr, "pulsar: %s generation requested but the graph backend is unavailable\n",
                    pulsar_backend_name(e->backend));
            return 1;
        }
        return generate_gpu_graph_raw_swa(model, vocab, weights, prompt,
                                            n_predict, ctx_size, e->quality,
                                            e->prefill_chunk,
                                            e->directional_steering_file,
                                            e->directional_steering_attn_scale,
                                            e->directional_steering_ffn_scale,
                                            emit, done, emit_ud,
                                            progress, progress_ud);
    }
    return 1;
}



int pulsar_engine::gpu_graph_test(const pulsar_tokens *prompt) {
    auto *e = this;
    if (!e->gpu_ready) {
        fprintf(stderr, "pulsar: %s graph test requested but backend is unavailable\n",
                pulsar_backend_name(e->backend));
        return 1;
    }
    return gpu_graph_decode_test(&e->model, &e->weights, prompt, e->quality);
}



int pulsar_engine::head_test(const pulsar_tokens *prompt) {
    auto *e = this;
    if (!prompt || prompt->len <= 0) {
        fprintf(stderr, "pulsar: head test requires a non-empty prompt\n");
        return 1;
    }

    const pulsar_model *model = &e->model;
    const pulsar_vocab *vocab = &e->vocab;
    const pulsar_weights *weights = &e->weights;
    const pulsar_layer_weights *layer0 = &weights->layer[0];

    float *prompt_embd = (float *)xmalloc((size_t)prompt->len * PULSAR_N_EMBD * sizeof(prompt_embd[0]));
    embed_prompt(model, weights, prompt, PULSAR_N_EMBD, prompt_embd);

    const uint32_t n_hc = PULSAR_N_HC;
    float *hc0 = (float *)xmalloc((size_t)PULSAR_N_EMBD * sizeof(hc0[0]));
    float *residual_hc = (float *)xmalloc((size_t)n_hc * PULSAR_N_EMBD * sizeof(residual_hc[0]));
    float hc_post[4];
    float hc_comb[16];
    layer_attn_pre_one(model, layer0,
        prompt_embd + (uint64_t)(prompt->len - 1) * PULSAR_N_EMBD,
        hc0, residual_hc, hc_post, hc_comb);
    print_vec_stats("blk.0 attn_pre", hc0, PULSAR_N_EMBD);

    float *attn_norm0 = (float *)xmalloc((size_t)PULSAR_N_EMBD * sizeof(attn_norm0[0]));
    layer_attn_norm_one(attn_norm0, model, layer0, hc0);

    const uint64_t q_dim = (uint64_t)PULSAR_N_HEAD * PULSAR_N_HEAD_DIM;
    float *q0 = (float *)xmalloc((size_t)q_dim * sizeof(q0[0]));
    layer_q_projection_normed_one(model, layer0, attn_norm0, q0);
    print_vec_stats("blk.0 q", q0, q_dim);

    float *kv0 = (float *)xmalloc((size_t)PULSAR_N_HEAD_DIM * sizeof(kv0[0]));
    layer_kv_projection_normed_one(model, layer0, attn_norm0, kv0);
    print_vec_stats("blk.0 kv", kv0, PULSAR_N_HEAD_DIM);
    rope_tail_layer_inplace(q0, PULSAR_N_HEAD, PULSAR_N_HEAD_DIM, PULSAR_N_ROT, (uint32_t)(prompt->len - 1), 0, false);
    rope_tail_layer_inplace(kv0, PULSAR_N_HEAD_KV, PULSAR_N_HEAD_DIM, PULSAR_N_ROT, (uint32_t)(prompt->len - 1), 0, false);
    dsv4_fp8_kv_quantize_row_inplace_cpu(kv0, PULSAR_N_HEAD_DIM, PULSAR_N_ROT);
    f16_round_inplace_cpu(kv0, PULSAR_N_HEAD_DIM);

    float *attn_heads = (float *)xmalloc((size_t)q_dim * sizeof(attn_heads[0]));
    layer_attention_one(attn_heads, model, layer0, q0, kv0);
    print_vec_stats("blk.0 attn_heads", attn_heads, q_dim);
    rope_tail_layer_inplace(attn_heads, PULSAR_N_HEAD, PULSAR_N_HEAD_DIM, PULSAR_N_ROT, (uint32_t)(prompt->len - 1), 0, true);

    float *attn_out = (float *)xmalloc((size_t)PULSAR_N_EMBD * sizeof(attn_out[0]));
    layer_grouped_out_one(attn_out, model, layer0, attn_heads);
    print_vec_stats("blk.0 attn_out", attn_out, PULSAR_N_EMBD);

    float *after_attn_hc = (float *)xmalloc((size_t)n_hc * PULSAR_N_EMBD * sizeof(after_attn_hc[0]));
    hc_post_one(after_attn_hc, attn_out, residual_hc, hc_post, hc_comb, PULSAR_N_EMBD, n_hc);
    print_vec_stats("blk.0 after_attn_hc", after_attn_hc, (uint64_t)n_hc * PULSAR_N_EMBD);

    float *after_ffn_hc = (float *)xmalloc((size_t)n_hc * PULSAR_N_EMBD * sizeof(after_ffn_hc[0]));
    layer_ffn_one(after_ffn_hc, model, layer0, after_attn_hc, 0, prompt->v[prompt->len - 1],
                  NULL, 0.0f, true);
    print_vec_stats("blk.0 after_ffn_hc", after_ffn_hc, (uint64_t)n_hc * PULSAR_N_EMBD);

    float *logits = (float *)xmalloc((size_t)PULSAR_N_VOCAB * sizeof(logits[0]));
    output_logits_one(logits, model, weights, after_ffn_hc);
    print_vec_stats("logits", logits, PULSAR_N_VOCAB);

    int best[8];
    for (int i = 0; i < 8; i++) best[i] = -1;
    for (uint32_t i = 0; i < PULSAR_N_VOCAB; i++) {
        for (int j = 0; j < 8; j++) {
            if (best[j] < 0 || logits[i] > logits[best[j]]) {
                for (int k = 7; k > j; k--) best[k] = best[k - 1];
                best[j] = (int)i;
                break;
            }
        }
    }

    printf("top logits after native blk.0 slice:\n");
    for (int i = 0; i < 8; i++) {
        printf("  %6d  %9.4f  %.*s\n",
            best[i],
            logits[best[i]],
            (int)vocab->token[best[i]].len,
            vocab->token[best[i]].ptr);
    }

    free(logits);
    free(after_ffn_hc);
    free(after_attn_hc);
    free(attn_out);
    free(attn_heads);
    free(kv0);
    free(q0);
    free(attn_norm0);
    free(residual_hc);
    free(hc0);
    free(prompt_embd);
    return 0;
}






int pulsar_engine::open(pulsar_engine **out, const pulsar_engine_options *opt) {
    pulsar_engine *e = (pulsar_engine *)xcalloc(1, sizeof(*e));
    e->model.fd = -1;
    e->dspark_model.fd = -1;
    e->backend = opt->backend;
    e->quality = opt->quality;
    e->prefill_chunk = opt->prefill_chunk;
    /* Default draft depth 3: the measured v5mx optimum (2026-07-17 k-sweep on
     * the shipped ds4flash build at the tau=0.25 conf-sched default, quench
     * disarmed, conf-sched trimming active). k=3 beats k=5 by +15% structured
     * to +32% prose served decode; distribution-preserving (exact verify) —
     * byte-identical on structured, near-tie-equivalent on greedy prose (the
     * verify-width change flips ~1-ULP argmax ties, same class as yield-quench).
     * The DSpark drafter forward is autoregressive, so its cost scales with the
     * chain length ON TOP of the verify rows — ms/accepted-token stays flat
     * ~41-46 ms across k, i.e. depth never amortizes, so shallower wins.
     * The prior default 5 was a compact-model figure (2026-07-09, conf3 head,
     * tau 0.35) that does not hold on shipped v5mx at the tau=0.25 default. */
    e->dspark_draft_tokens = opt->dspark_draft_tokens > 0 ? opt->dspark_draft_tokens : 3;
    if (e->dspark_draft_tokens > 16) e->dspark_draft_tokens = 16;
    e->dspark_confidence = opt->dspark_confidence > 0.0f ? opt->dspark_confidence : 0.5f;
    if ((opt->directional_steering_attn != 0.0f || opt->directional_steering_ffn != 0.0f) &&
        (!opt->directional_steering_file || !opt->directional_steering_file[0]))
    {
        fprintf(stderr, "pulsar: directional steering needs --dir-steering-file\n");
        free(e);
        *out = NULL;
        return 1;
    }
    if (opt->directional_steering_file && opt->directional_steering_file[0]) {
        e->directional_steering_file = pulsar_strdup(opt->directional_steering_file);
        e->directional_steering_attn_scale = opt->directional_steering_attn;
        e->directional_steering_ffn_scale = opt->directional_steering_ffn;
    }
    if (opt->n_threads > 0) g_requested_threads = (uint32_t)opt->n_threads;
    pulsar_acquire_instance_lock();

    const bool graph_backend = pulsar_backend_uses_graph(opt->backend);
    if (graph_backend) pulsar_linux_graph_backend_set_oom_score(opt->backend);
    model_open(&e->model, opt->model_path, graph_backend, !opt->inspect_only);
    if (opt->warm_weights) model_warm_weights(&e->model);
    if (!opt->inspect_only) e->vocab.vocab_load(&e->model);
    config_validate_model(&e->model);
    if (opt->expert_overlay && opt->expert_overlay[0]) {
        const char *sep = strrchr(opt->expert_overlay, ':');
        if (!sep || sep == opt->expert_overlay || !sep[1]) {
            fprintf(stderr, "pulsar: --expert-overlay expects FILE:PREFIX (e.g. donor.gguf:blk.17.)\n");
            e->destroy();
            *out = NULL;
            return 1;
        }
        char overlay_path[4096];
        const size_t path_len = (size_t)(sep - opt->expert_overlay);
        if (path_len >= sizeof(overlay_path)) {
            fprintf(stderr, "pulsar: --expert-overlay path is too long\n");
            e->destroy();
            *out = NULL;
            return 1;
        }
        memcpy(overlay_path, opt->expert_overlay, path_len);
        overlay_path[path_len] = '\0';
        model_open(&e->overlay_model, overlay_path, graph_backend, false);
        e->overlay_ready = true;
        /* PREFIX is a comma-separated list so several layers can be swapped
         * in one run (e.g. compose "anchor + candidate" from a cheap base
         * without materializing the combined model as a file). */
        char prefixes[2048];
        const size_t plist_len = strlen(sep + 1);
        if (plist_len >= sizeof(prefixes)) {
            fprintf(stderr, "pulsar: --expert-overlay prefix list is too long\n");
            e->destroy();
            *out = NULL;
            return 1;
        }
        memcpy(prefixes, sep + 1, plist_len + 1);
        uint32_t swapped = 0;
        for (char *p = strtok(prefixes, ","); p; p = strtok(NULL, ",")) {
            const uint32_t n = model_apply_expert_overlay(&e->model, &e->overlay_model, p);
            if (n == 0) {
                fprintf(stderr, "pulsar: --expert-overlay prefix '%s' matched no routed-expert tensors\n",
                        p);
                e->destroy();
                *out = NULL;
                return 1;
            }
            swapped += n;
        }
        fprintf(stderr, "pulsar: expert overlay: %u tensors swapped in from %s (prefixes %s)\n",
                swapped, overlay_path, sep + 1);
    }
    weights_bind(&e->weights, &e->model);
    if (opt->inspect_only) {
        *out = e;
        return 0;
    }
    if (!opt->dspark_disable && model_find_tensor(&e->model, "dspark.main_proj.weight")) {
        /* Drafter merged into the main GGUF: bind from the main model and
         * alias dspark_model to it by value (same map/fd; every dspark call
         * site reads e->dspark_model, and close is guarded on dspark_external
         * so the shared mapping is only torn down once). */
        dspark_weights_bind(&e->dspark_weights, &e->model);
        e->dspark_model = e->model;
        e->dspark_external = false;
        e->dspark_ready = true;
        fprintf(stderr, "pulsar: DSpark drafter found in model (draft=%d, confidence=%.2f)\n",
                e->dspark_draft_tokens,
                (double)e->dspark_confidence);
    } else if (!opt->dspark_disable && opt->dspark_path && opt->dspark_path[0]) {
        model_open(&e->dspark_model, opt->dspark_path, graph_backend, true);
        dspark_weights_bind(&e->dspark_weights, &e->dspark_model);
        e->dspark_external = true;
        e->dspark_ready = true;
        fprintf(stderr, "pulsar: DSpark support model loaded: %s (draft=%d, confidence=%.2f)\n",
                opt->dspark_path,
                e->dspark_draft_tokens,
                (double)e->dspark_confidence);
    }

    if (graph_backend) {
        e->gpu_ready = pulsar_gpu_init() != 0;
        if (!e->gpu_ready) {
            fprintf(stderr, "pulsar: %s backend unavailable; aborting startup\n",
                    pulsar_backend_name(e->backend));
            e->destroy();
            *out = NULL;
            return 1;
        }
        pulsar_gpu_set_quality(e->quality);
        (void)pulsar_gpu_set_model_fd(e->model.fd);
        const int model_map_ok =
            pulsar_gpu_set_model_map_range(e->model.map,
                                        e->model.size,
                                        e->model.tensor_data_pos,
                                        e->model.size - e->model.tensor_data_pos,
                                        e->model.max_tensor_bytes);
        if (!model_map_ok) {
            fprintf(stderr,
                    "pulsar: %s failed to map model views; aborting startup. "
                    "This is commonly caused by insufficient memory or accelerator VM budget.\n",
                    pulsar_backend_name(e->backend));
            e->destroy();
            *out = NULL;
            return 1;
        }
        if (e->dspark_ready && e->dspark_external &&
            !pulsar_gpu_set_model_map_range(e->dspark_model.map,
                                           e->dspark_model.size,
                                           e->dspark_model.tensor_data_pos,
                                           e->dspark_model.size - e->dspark_model.tensor_data_pos,
                                           e->dspark_model.max_tensor_bytes))
        {
            fprintf(stderr,
                    "pulsar: %s failed to map DSpark model views; aborting startup. "
                    "This is commonly caused by insufficient memory or accelerator VM budget.\n",
                    pulsar_backend_name(e->backend));
            e->destroy();
            *out = NULL;
            return 1;
        }
        (void)pulsar_gpu_set_model_fd_for_map(e->model.fd, e->model.map);
        if (!accelerator_cache_model_tensors(e->backend, &e->model,
                                             NULL, NULL, 0,
                                             e->dspark_ready ? NULL : "dspark.")) {
            fprintf(stderr, "pulsar: %s failed to prepare optional model cache\n",
                    pulsar_backend_name(e->backend));
            e->destroy();
            *out = NULL;
            return 1;
        }
        if (e->dspark_ready && e->dspark_external) {
            (void)pulsar_gpu_set_model_fd_for_map(e->dspark_model.fd, e->dspark_model.map);
            if (!accelerator_cache_model_tensors(e->backend, &e->dspark_model,
                                                 NULL, NULL, 0, NULL)) {
                fprintf(stderr, "pulsar: %s failed to prepare optional DSpark model cache\n",
                        pulsar_backend_name(e->backend));
                e->destroy();
                *out = NULL;
                return 1;
            }
            (void)pulsar_gpu_set_model_fd_for_map(e->model.fd, e->model.map);
        }
        if (e->overlay_ready &&
            !accelerator_prepare_expert_overlay(e->backend, &e->model,
                                                &e->overlay_model)) {
            fprintf(stderr, "pulsar: %s failed to prepare expert-overlay spans\n",
                    pulsar_backend_name(e->backend));
            e->destroy();
            *out = NULL;
            return 1;
        }
        fprintf(stderr, "pulsar: %s backend initialized for graph diagnostics\n",
                pulsar_backend_name(e->backend));

        /* One MoE-tier boot line so a silent slow tier is no longer silent:
         * resolved expert weight type (grouped-CUTLASS type-40 vs per-expert
         * tiled type-39) + the resolved gate/up MXFP4 tile width (NT16/NT8).
         * Reuses the bound-layer expert tensors and the CUDA tile-width
         * accessor; log-only, runs once at open on every GPU serve. */
        {
            /* Count the tier PER LAYER.  Reporting the first bound layer's type
             * as "the" tier is actively misleading on a heterogeneous model:
             * v5mx has only 5 uniform-MXFP4 layers, so the old line printed
             * "per-expert-tiled" while a dozen layers were in fact running the
             * grouped CUTLASS path (2026-07-21 profile).  This observability
             * exists to catch a SILENT fall to a slow tier, so it has to be
             * truthful about the mix or it defeats its own purpose. */
            uint32_t n_grouped = 0, n_tiled = 0, n_routed = 0;
            const pulsar_layer_weights *ml = NULL;
            for (uint32_t il = 0; il < PULSAR_N_LAYER; il++) {
                const pulsar_layer_weights *l = &e->weights.layer[il];
                if (!l->ffn_gate_exps || !l->ffn_down_exps) continue;
                if (!ml) ml = l;                    /* first routed layer, for the type sample */
                n_routed++;
                /* The grouped/GEMV CUTLASS dispatch is entered only when BOTH
                 * gate and down experts are type-40 (see the moe.cu batch
                 * predicate); any other mix takes the per-expert tiled path. */
                if (l->ffn_gate_exps->type == PULSAR_TENSOR_CUTLASS_MXFP4 &&
                    l->ffn_down_exps->type == PULSAR_TENSOR_CUTLASS_MXFP4) n_grouped++;
                else n_tiled++;
            }
            if (ml) {
                const uint32_t gt = ml->ffn_gate_exps->type;
                const uint32_t dt = ml->ffn_down_exps->type;
                fprintf(stderr,
                        "pulsar: MoE expert tier: %u/%u layers grouped-CUTLASS, %u/%u per-expert-tiled "
                        "(first routed layer gate=%s(%u) down=%s(%u)) mxfp4 tile=NT%u\n",
                        n_grouped, n_routed, n_tiled, n_routed,
                        tensor_type_name(gt), gt, tensor_type_name(dt), dt,
                        pulsar_gpu_moe_mxfp4_tile_width());
            }
        }
    }

    *out = e;
    return 0;
}



void pulsar_engine::summary() {
    auto *e = this;
    model_summary(&e->model);
}



int pulsar_engine::vocab_size() {
    auto *e = this;
    return e ? e->vocab.n_vocab : 0;
}



/* The engine's logits ROW WIDTH — the shape profile's n_vocab, which is what
 * every logits buffer the engine writes is strided by.  This is NOT
 * pulsar_engine_vocab_size (the tokenizer table length): the loader never checks
 * the two against each other, and sizing a logits buffer from the tokenizer
 * length is exactly the mismatch that produced an unbounded-logits write. */
int pulsar_engine::logits_width() const {
    auto *e = this;
    return e ? (int)PULSAR_N_VOCAB : 0;
}



const char *pulsar_engine::model_name() {
    auto *e = this;
    (void)e;
    return PULSAR_MODEL_SHAPE_NAME;
}

void pulsar_engine::spec_metrics(pulsar_spec_metrics *out) {
    auto *e = this;
    if (!out) return;
    memset(out, 0, sizeof(*out));
    if (!e) return;
    out->accepted_tokens = e->spec_accepted_tokens;
    out->draft_tokens = e->spec_draft_tokens;
    out->num_drafts = e->spec_num_drafts;
    out->gen_tokens = e->spec_gen_tokens;
    for (int i = 0; i < 16; i++) out->accepted_per_pos[i] = e->spec_accepted_per_pos[i];
    out->max_draft = e->dspark_draft_tokens;
    out->has_dspark = e->dspark_ready;
}



/* Per-SESSION cumulative DSpark counters (accepted/draft/drafts/gen). Same
 * semantics as pulsar_engine_spec_metrics but scoped to one session, so a caller
 * can diff a snapshot across a single request for a per-response accept-rate.
 * accepted_per_pos is left zero (the per-position waterfall stays a /metrics,
 * cross-request concern). */
void pulsar_session::spec_metrics(pulsar_spec_metrics *out) const {
    auto *s = this;
    if (!out) return;
    memset(out, 0, sizeof(*out));
    if (!s) return;
    out->accepted_tokens = s->spec.spec_accepted_tokens;
    out->draft_tokens = s->spec.spec_draft_tokens;
    out->num_drafts = s->spec.spec_num_drafts;
    out->gen_tokens = s->spec.spec_gen_tokens;
    out->max_draft = s->engine ? s->engine->dspark_draft_tokens : 0;
    out->has_dspark = s->engine ? s->engine->dspark_ready : false;
}



int pulsar_engine::layer_count() {
    auto *e = this;
    (void)e;
    return (int)PULSAR_N_LAYER;
}



uint64_t pulsar_engine::weights_resident_bytes() {
    auto *e = this;
    if (!e) return 0;
    /* The GGUF(s) are mmap'd read-only and shared across every session, so this
     * is a single resident copy competing with per-session KV for the unified
     * memory budget.  A merged/embedded drafter lives inside e->model and is
     * already counted; an external drafter and an expert overlay map their own
     * files and are added when present. */
    uint64_t bytes = e->model.size;
    if (e->dspark_ready && e->dspark_external) bytes += e->dspark_model.size;
    if (e->overlay_ready) bytes += e->overlay_model.size;
    return bytes;
}



uint32_t pulsar_engine::layer_compress_ratio(uint32_t layer) {
    auto *e = this;
    (void)e;
    if (layer >= PULSAR_N_LAYER) return 0;
    return pulsar_layer_compress_ratio(layer);
}



uint64_t pulsar_engine::hidden_f32_values() {
    auto *e = this;
    (void)e;
    return (uint64_t)PULSAR_N_HC * PULSAR_N_EMBD;
}



int pulsar_engine::model_id() {
    auto *e = this;
    (void)e;
    return (int)PULSAR_MODEL_VARIANT;
}



void pulsar_engine::destroy() {
    auto *e = this;
    if (!e) return;
    weights_free(&e->weights);
    e->vocab.vocab_free();
    pulsar_threads_shutdown();
    /* Tear down GPU state (which cudaHostUnregisters the mmap'd weight ranges)
     * before munmap'ing the model — unmapping still-registered pages is UB. */
    pulsar_gpu_cleanup();
    if (e->dspark_ready && e->dspark_external) model_close(&e->dspark_model);
    if (e->overlay_ready) model_close(&e->overlay_model);
    model_close(&e->model);
    pulsar_release_instance_lock();
    free(e->directional_steering_dirs);
    free(e->directional_steering_file);
    free(e);
}



int pulsar_session::create(pulsar_session **out, pulsar_engine *e, int ctx_size) {
    if (!out || !e || ctx_size <= 0) return 1;
    if (!pulsar_backend_uses_graph(e->backend) || !e->gpu_ready) return 1;

    pulsar_session *s = (pulsar_session *)xcalloc(1, sizeof(*s));
    s->engine = e;
    s->ctx_size = ctx_size;
    s->prefill_cap = gpu_graph_prefill_cap_for_prompt(ctx_size,
                                                        e->prefill_chunk);
    const uint32_t raw_cap = gpu_graph_raw_cap_for_context(ctx_size, s->prefill_cap);
    const pulsar_layer_weights *shape_layer = weights_first_bound_layer(&e->weights);
    if (!shape_layer) {
        fprintf(stderr, "pulsar: no transformer layers are loaded\n");
        free(s);
        return 1;
    }
    /* Measure the true GPU cost of this session (allocator delta across the
     * create) so callers can reconcile admission estimates against reality. */
    const uint64_t alloc_before = pulsar_gpu_tensor_alloc_bytes_current();
    if (!gpu_graph_alloc_raw_cap(&s->graph, &e->weights, shape_layer,
                                   raw_cap, (uint32_t)ctx_size, s->prefill_cap,
                                   e->dspark_ready))
    {
        free(s);
        return 1;
    }
    s->graph.quality = e->quality;
    if (!gpu_graph_load_directional_steering(&s->graph,
                                               e->directional_steering_file,
                                               e->directional_steering_attn_scale,
                                               e->directional_steering_ffn_scale)) {
        gpu_graph_free(&s->graph);
        free(s);
        return 1;
    }
    s->logits = (float *)xmalloc((size_t)PULSAR_N_VOCAB * sizeof(s->logits[0]));
    if (e->dspark_ready) {
        if (!gpu_graph_init_dspark_target(&s->graph, e->dspark_weights.target_layer_ids)) {
            fprintf(stderr, "pulsar: failed to allocate DSpark graph buffers\n");
            gpu_graph_free(&s->graph);
            free(s->logits);
            free(s);
            return 1;
        }
    }
    s->resident_bytes = pulsar_gpu_tensor_alloc_bytes_current() - alloc_before;
    *out = s;
    return 0;
}



uint64_t pulsar_session_resident_bytes(const pulsar_session *s) {
    return s ? s->resident_bytes : 0;
}



/* TRUE total per-session GPU byte cost of pulsar_session_create at this context
 * size — the admission-control price of a session.  Derives prefill/raw caps
 * exactly like pulsar_session_create and includes the DSpark drafter graph state
 * when the engine has a drafter loaded, so callers cannot pass mismatched
 * parameters.  Built on the same sizing code as the allocator
 * (gpu_graph_session_bytes, gpu_diag.c); reconcile against
 * pulsar_session_resident_bytes after the create. */
uint64_t pulsar_engine::session_cost_bytes(int ctx_size) {
    auto *e = this;
    if (!e || ctx_size <= 0) return 0;
    if (!pulsar_backend_uses_graph(e->backend) || !e->gpu_ready) return 0;
    const uint32_t prefill_cap = gpu_graph_prefill_cap_for_prompt(ctx_size,
                                                                  e->prefill_chunk);
    const uint32_t raw_cap = gpu_graph_raw_cap_for_context(ctx_size, prefill_cap);
    const pulsar_layer_weights *shape_layer = weights_first_bound_layer(&e->weights);
    if (!shape_layer) return 0;
    return gpu_graph_session_bytes(&e->weights, shape_layer, raw_cap,
                                   (uint32_t)ctx_size, prefill_cap,
                                   e->dspark_ready);
}

uint64_t pulsar_engine::session_cost_bytes_banked(int ctx_size,
                                              int n_banks) {
    auto *e = this;
    if (!e || ctx_size <= 0 || n_banks < 1) return 0;
    if (!pulsar_backend_uses_graph(e->backend) || !e->gpu_ready) return 0;
    const uint32_t prefill_cap = gpu_graph_prefill_cap_for_prompt(ctx_size,
                                                                  e->prefill_chunk);
    const uint32_t raw_cap = gpu_graph_raw_cap_for_context(ctx_size, prefill_cap);
    const pulsar_layer_weights *shape_layer = weights_first_bound_layer(&e->weights);
    if (!shape_layer) return 0;
    return gpu_graph_session_bytes_banked(&e->weights, shape_layer, raw_cap,
                                          (uint32_t)ctx_size, prefill_cap,
                                          e->dspark_ready, (uint32_t)n_banks);
}

uint64_t pulsar_engine::demand_paged_bytes_per_bank(int ctx_size) {
    auto *e = this;
    if (!e || ctx_size <= 0) return 0;
    if (!pulsar_backend_uses_graph(e->backend) || !e->gpu_ready) return 0;
    return gpu_graph_demand_paged_bytes_per_bank((uint32_t)ctx_size);
}

uint64_t pulsar_session::touched_kv_bytes() const {
    auto *s = this;
    if (!s) return 0;
    return gpu_graph_touched_kv_bytes(&s->graph);
}




void pulsar_session::destroy() {
    auto *s = this;
    if (!s) return;
    gpu_graph_free(&s->graph);
    token_vec_free(&s->checkpoint);
    pulsar_sample_scratch_free(&s->sample_scratch);
    s->bank_carry_free();
    free(s->dspark_pending_qrows);
    free(s->spec_row_scratch);
    free(s->logits);
    free(s);
}



void pulsar_session::set_progress(pulsar_session_progress_fn fn, void *ud) {
    auto *s = this;
    if (!s) return;
    s->progress = fn;
    s->progress_ud = ud;
}



void pulsar_session::set_display_progress(pulsar_session_progress_fn fn, void *ud) {
    auto *s = this;
    if (!s) return;
    s->display_progress = fn;
    s->display_progress_ud = ud;
}



void pulsar_session::set_cancel(pulsar_session_cancel_fn fn, void *ud) {
    auto *s = this;
    if (!s) return;
    s->cancel = fn;
    s->cancel_ud = ud;
}



static bool pulsar_session_cancelled(pulsar_session *s) {
    return s && s->cancel && s->cancel(s->cancel_ud);
}



static bool pulsar_session_cancelled_cb(void *ud) {
    return pulsar_session_cancelled((pulsar_session *)ud);
}



void pulsar_session::report_progress(const char *event, int current, int total) {
    auto *s = this;
    if (!s || !s->progress || !event) return;
    s->progress(s->progress_ud, event, current, total);
}


static void pulsar_session_note_prefill_progress(void *ud, const char *event, int current, int total) {
    pulsar_sync_progress *p = (pulsar_sync_progress *)ud;
    if (!p || !p->session || !p->prompt) return;
    if (!strcmp(event, "prefill_chunk") && current > 0 && current <= p->prompt->len) {
        p->session->checkpoint.len = 0;
        for (int i = 0; i < current; i++) token_vec_push(&p->session->checkpoint, p->prompt->v[i]);
        p->session->checkpoint_valid = true;
    }
    if (p->user) p->user(p->user_ud, event, current, total);
}



/* Bring the live backend state to exactly the supplied token prefix.
 *
 * pulsar-server and the REPL are stateless at the text/API layer but stateful here:
 * they resend or rebuild the full transcript, and this function decides whether
 * the live checkpoint is a prefix.  A matching prefix is extended in one of two
 * ways:
 *
 *   - long suffix: batched layer-major prefill, aligned to absolute chunk
 *     boundaries so compressor/indexer rows finalize in the same order as a
 *     cold prompt;
 *   - short suffix: ordinary one-token decode, which is faster below the
 *     measured crossover and preserves exact autoregressive semantics.
 *
 * A non-matching prompt discards the checkpoint and prefills from token zero.
 */
int pulsar_session::sync(const pulsar_tokens *prompt, char *err, size_t errlen) {
    auto *s = this;
    if (!s || !prompt || prompt->len <= 0 || prompt->len >= s->ctx_size) {
        snprintf(err, errlen, "prompt exceeds context");
        return 1;
    }
    if (pulsar_session_cancelled(s)) {
        snprintf(err, errlen, "interrupted");
        return PULSAR_SESSION_SYNC_INTERRUPTED;
    }
    pulsar_engine *e = s->engine;
    const char *backend_name = pulsar_backend_name(e->backend);

    /* a sync begins a new request: any carry left by a max-tokens/stop-string
     * truncated generation belongs to the previous request's distribution.
     * (position stamping alone misses a same-length full rebuild.) */
    s->spec.spec_carry_valid = false;
    /* Same argument, same blind spot: the pendings' position stamp cannot see a
     * rebuild that lands on the same length, and a sampled draft's q belongs to
     * the previous request's distribution. Dropping them here costs one draft
     * round at the start of a request and is the only guard that covers it. */
    s->spec.dspark_n_pending = 0;
    /* A sync begins a new request: re-arm the terminal yield quench. */
    spec_quench_reset(s);

    if (s->checkpoint_valid &&
        prompt->len >= s->checkpoint.len &&
        pulsar_tokens_starts_with(prompt, &s->checkpoint))
    {
        const int suffix = prompt->len - s->checkpoint.len;
        const uint32_t resume_min = gpu_graph_resume_prefill_min_tokens();
        if (suffix > 0 && (uint32_t)suffix >= resume_min) {
            bool cancelled = false;
            pulsar_sync_progress progress = {
                .session = s,
                .prompt = prompt,
                .user = s->progress,
                .user_ud = s->progress_ud,
            };
            bool ok = gpu_graph_prefill_chunked_range(&s->graph,
                                                        &e->model,
                                                        &e->weights,
                                                        prompt,
                                                        (uint32_t)s->checkpoint.len,
                                                        (uint32_t)suffix,
                                                        s->logits,
                                                        false,
                                                        pulsar_session_note_prefill_progress,
                                                        &progress,
                                                        s->display_progress,
                                                        s->display_progress_ud,
                                                        NULL,
                                                        pulsar_session_cancelled_cb,
                                                        s,
                                                        &cancelled);
            if (cancelled) {
                snprintf(err, errlen, "interrupted");
                s->checkpoint_valid = true;
                return PULSAR_SESSION_SYNC_INTERRUPTED;
            }
            if (!ok) {
                snprintf(err, errlen, "%s resumed prefill failed while extending checkpoint", backend_name);
                s->checkpoint_valid = false;
                return 1;
            }
            pulsar_tokens_copy(&s->checkpoint, prompt);
            s->checkpoint_valid = true;
            return 0;
        }

        for (int i = s->checkpoint.len; i < prompt->len; i++) {
            if (pulsar_session_cancelled(s)) {
                snprintf(err, errlen, "interrupted");
                s->checkpoint_valid = true;
                return PULSAR_SESSION_SYNC_INTERRUPTED;
            }
            if (!gpu_graph_eval_token_raw_swa(&s->graph, &e->model, &e->weights,
                                                (uint32_t)prompt->v[i],
                                                (uint32_t)s->checkpoint.len,
                                                s->logits))
            {
                snprintf(err, errlen, "%s decode failed while extending checkpoint", backend_name);
                s->checkpoint_valid = false;
                return 1;
            }
            token_vec_push(&s->checkpoint, prompt->v[i]);
        }
        return 0;
    }

    bool ok;
    s->checkpoint_valid = false;
    s->checkpoint.len = 0;
    if (!gpu_graph_reset_prefill_state(&s->graph)) {
        snprintf(err, errlen, "%s prefill state reset failed", backend_name);
        return 1;
    }
    /* The rebuild path is the one place classic per-bank truth is legitimately
     * re-established: reset_prefill_state zeroes layer_n_comp /
     * layer_n_index_comp, so the scalars stop holding any multiseq superset
     * and the prefill below refills them from zero against the installed
     * bank.  (The prefix-resume path above cannot be reached while dirty —
     * decode_multiseq clears checkpoint_valid, which that path gates on.) */
    s->mseq_dirty = false;
    if (s->prefill_cap < (uint32_t)prompt->len) {
        bool cancelled = false;
        pulsar_sync_progress progress = {
            .session = s,
            .prompt = prompt,
            .user = s->progress,
            .user_ud = s->progress_ud,
        };
        ok = gpu_graph_prefill_chunked(&s->graph, &e->model, &e->weights,
                                         prompt, prompt->len, s->logits, false,
                                         pulsar_session_note_prefill_progress, &progress,
                                         s->display_progress,
                                         s->display_progress_ud,
                                         pulsar_session_cancelled_cb,
                                         s,
                                         &cancelled);
        if (cancelled) {
            snprintf(err, errlen, "interrupted");
            s->checkpoint_valid = s->checkpoint.len > 0;
            return PULSAR_SESSION_SYNC_INTERRUPTED;
        }
    } else {
        bool cancelled = false;
        ok = gpu_graph_prefill_raw_swa(&s->graph, &e->model, &e->weights,
                                         prompt, prompt->len, s->logits, false,
                                         s->display_progress,
                                         s->display_progress_ud,
                                         pulsar_session_cancelled_cb,
                                         s,
                                         &cancelled);
        if (cancelled) {
            snprintf(err, errlen, "interrupted");
            return PULSAR_SESSION_SYNC_INTERRUPTED;
        }
    }
    if (!ok) {
        snprintf(err, errlen, "%s prefill failed", backend_name);
        s->checkpoint_valid = false;
        return 1;
    }
    pulsar_tokens_copy(&s->checkpoint, prompt);
    s->checkpoint_valid = true;
    return 0;
}



/* Return true when canonicalization would replace already-sampled tokens.
 *
 * A DS4 session checkpoint is more than a token vector: the backend state also
 * contains raw SWA rows, compressed KV rows, indexer rows, and compressor
 * frontiers.  Replacing any part of the live tail requires restoring that whole
 * frontier first.  Extending exactly at the live end is safe; rewriting behind
 * it is not an in-place operation. */
bool pulsar_session_rewrite_requires_rebuild(int live_len, int canonical_len, int common) {
    if (live_len < 0 || canonical_len < 0 || common < 0) return true;
    if (common > live_len || common > canonical_len) return true;
    return common < live_len;
}



/* Replace the live suffix after a shared prefix.
 *
 * This is used after parsing a generated tool call.  The model may have emitted
 * DSML in an order that is semantically valid but not byte-for-byte equal to the
 * canonical prompt we will see on the next request.  Rewriting only the token
 * checkpoint is not enough: the backend still contains raw and compressed rows
 * for the old suffix.  Until we have a real frontier snapshot at the
 * rewrite point, any replacement behind the live end reports that a rebuild is
 * needed without mutating the session.  The server may still find an older disk KV
 * checkpoint before falling back to a full replay. */
pulsar_session_rewrite_result pulsar_session::rewrite_from_common(const pulsar_tokens *prompt, int common,
        char *err, size_t errlen) {
    auto *s = this;
    if (!s || !prompt || prompt->len <= 0 || prompt->len >= s->ctx_size) {
        snprintf(err, errlen, "prompt exceeds context");
        return PULSAR_SESSION_REWRITE_ERROR;
    }
    if (!s->checkpoint_valid) {
        snprintf(err, errlen, "session has no valid checkpoint");
        return PULSAR_SESSION_REWRITE_ERROR;
    }
    if (common < 0 || common > s->checkpoint.len || common > prompt->len) {
        snprintf(err, errlen, "invalid rewrite prefix");
        return PULSAR_SESSION_REWRITE_ERROR;
    }
    for (int i = 0; i < common; i++) {
        if (s->checkpoint.v[i] != prompt->v[i]) {
            snprintf(err, errlen, "rewrite prefix does not match live checkpoint");
            return PULSAR_SESSION_REWRITE_ERROR;
        }
    }

    if (common == s->checkpoint.len) {
        return s->sync(prompt, err, errlen) == 0 ?
            PULSAR_SESSION_REWRITE_OK : PULSAR_SESSION_REWRITE_ERROR;
    }

    if (pulsar_session_rewrite_requires_rebuild(s->checkpoint.len, prompt->len, common)) {
        snprintf(err, errlen, "rewrite needs rebuild: common=%d live=%d canonical=%d",
                 common, s->checkpoint.len, prompt->len);
        return PULSAR_SESSION_REWRITE_REBUILD_NEEDED;
    }

    snprintf(err, errlen, "unexpected canonical rewrite state");
    return PULSAR_SESSION_REWRITE_ERROR;
}



int pulsar_session::common_prefix(const pulsar_tokens *prompt) {
    auto *s = this;
    if (!s->checkpoint_valid) return 0;
    int n = s->checkpoint.len < prompt->len ? s->checkpoint.len : prompt->len;
    int i = 0;
    while (i < n && s->checkpoint.v[i] == prompt->v[i]) i++;
    return i;
}



int pulsar_session::argmax() {
    auto *s = this;
    return sample_argmax(s->logits, PULSAR_N_VOCAB);
}



int pulsar_session::argmax_excluding(int excluded_id) {
    auto *s = this;
    if (!s || !s->logits) return -1;
    int best = -1;
    float best_logit = PULSAR_NEG_INF;
    for (uint32_t i = 0; i < PULSAR_N_VOCAB; i++) {
        if ((int)i == excluded_id) continue;
        const float v = s->logits[i];
        if (best < 0 || v > best_logit) {
            best = (int)i;
            best_logit = v;
        }
    }
    return best;
}



int pulsar_sample_logits(const float *logits, int n_vocab, float temperature,
                      int top_k, float top_p, float min_p, uint64_t *rng) {
    if (!logits || n_vocab <= 0) return 0;
    return sample_top_p_min_p(logits, (uint32_t)n_vocab, temperature, top_k, top_p, min_p, rng);
}



int pulsar_session::sample(float temperature, int top_k, float top_p, float min_p, uint64_t *rng) {
    auto *s = this;
    return sample_top_p_min_p(s->logits, PULSAR_N_VOCAB, temperature, top_k, top_p, min_p, rng);
}



int pulsar_session::top_logprobs(pulsar_token_score *out, int k) {
    auto *s = this;
    if (!s || !out || k <= 0) return 0;
    if (k > (int)PULSAR_N_VOCAB) k = (int)PULSAR_N_VOCAB;
    for (int i = 0; i < k; i++) {
        out[i].id = -1;
        out[i].logit = PULSAR_NEG_INF;
        out[i].logprob = PULSAR_NEG_INF;
    }

    float max_logit = PULSAR_NEG_INF;
    for (uint32_t i = 0; i < PULSAR_N_VOCAB; i++) {
        const float v = s->logits[i];
        if (!isfinite(v)) continue;
        if (v > max_logit) max_logit = v;
        for (int j = 0; j < k; j++) {
            if (out[j].id < 0 || v > out[j].logit) {
                for (int l = k - 1; l > j; l--) out[l] = out[l - 1];
                out[j].id = (int)i;
                out[j].logit = v;
                break;
            }
        }
    }
    if (!isfinite(max_logit)) return 0;

    double sum = 0.0;
    for (uint32_t i = 0; i < PULSAR_N_VOCAB; i++) {
        const float v = s->logits[i];
        if (isfinite(v)) sum += exp((double)v - (double)max_logit);
    }
    const double logsum = (double)max_logit + log(sum);
    for (int i = 0; i < k && out[i].id >= 0; i++) {
        out[i].logprob = isfinite(out[i].logit) ? (float)((double)out[i].logit - logsum) : PULSAR_NEG_INF;
    }
    return k;
}



int pulsar_session::token_logprob(int token, pulsar_token_score *out) {
    auto *s = this;
    if (!s || !out || token < 0 || token >= (int)PULSAR_N_VOCAB) return 0;

    float max_logit = PULSAR_NEG_INF;
    for (uint32_t i = 0; i < PULSAR_N_VOCAB; i++) {
        const float v = s->logits[i];
        if (isfinite(v) && v > max_logit) max_logit = v;
    }
    if (!isfinite(max_logit)) return 0;

    double sum = 0.0;
    for (uint32_t i = 0; i < PULSAR_N_VOCAB; i++) {
        const float v = s->logits[i];
        if (isfinite(v)) sum += exp((double)v - (double)max_logit);
    }
    const double logsum = (double)max_logit + log(sum);
    out->id = token;
    out->logit = s->logits[token];
    out->logprob = isfinite(out->logit) ? (float)((double)out->logit - logsum) : PULSAR_NEG_INF;
    return 1;
}



int pulsar_session::copy_logits(float *out, int cap) {
    auto *s = this;
    if (!s || !out || cap < (int)PULSAR_N_VOCAB) return 0;
    memcpy(out, s->logits, (size_t)PULSAR_N_VOCAB * sizeof(out[0]));
    return (int)PULSAR_N_VOCAB;
}



int pulsar_session::set_logits(const float *logits, int n) {
    auto *s = this;
    if (!s || !logits || n != (int)PULSAR_N_VOCAB) return 1;
    memcpy(s->logits, logits, (size_t)PULSAR_N_VOCAB * sizeof(s->logits[0]));
    return 0;
}



int pulsar_session::eval(int token, char *err, size_t errlen) {
    auto *s = this;
    if (!s) return 1;
    /* Fail loud rather than corrupt: after a multiseq step the graph's scalar
     * frontier counters hold a cross-bank superset, so this decode would emit
     * its compressor row at the superset index and attend over another bank's
     * rows — wrong logits, silently.  pulsar_session_sync re-establishes per-bank
     * state (rebuild path) and clears the flag. */
    if (s->mseq_dirty) {
        snprintf(err, errlen,
                 "session eval after a multiseq decode step: classic per-bank "
                 "state is stale (frontier counters hold a cross-bank "
                 "superset); re-sync the session first");
        return 1;
    }
    pulsar_engine *e = s->engine;
    if (!gpu_graph_eval_token_raw_swa(&s->graph, &e->model, &e->weights,
                                        (uint32_t)token,
                                        (uint32_t)s->checkpoint.len,
                                        s->logits))
    {
        snprintf(err, errlen, "%s decode failed", pulsar_backend_name(e->backend));
        s->checkpoint_valid = false;
        return 1;
    }
    token_vec_push(&s->checkpoint, token);
    /* a token evaluated outside the speculative path (tool injection, plain
     * fallback loops) advances the state past any in-flight carry */
    s->spec.spec_carry_valid = false;
    return 0;
}







void pulsar_session::note_committed_tokens(const int *toks, int n) {
    auto *s = this;
    if (!s || !toks || n <= 0) return;
    for (int i = 0; i < n; i++) token_vec_push(&s->checkpoint, toks[i]);
}






void pulsar_session::invalidate() {
    auto *s = this;
    s->checkpoint_valid = false;
    s->checkpoint.len = 0;
    s->spec.dspark_n_pending = 0;
    s->spec.spec_carry_valid = false;
    spec_quench_reset(s);
    /* plan-33 inc C: an invalidated bank restarts from zero — a live keep
     * threshold would make the fresh prefill's early emits restore a STALE
     * boundary row over the new conversation's KV. Disarm it. */
    if (s->graph.banks.n_banks) {
        s->graph.ms_emit_keep[s->graph.banks.cur_bank] = 0u;
    }
    /* The drafter's context-KV ring must not survive into a new prompt: it was
     * never reset before, so in the server every request after the first
     * attended over the PREVIOUS request's window rows for its first ~128
     * generated tokens (and the drafter is near-useless without a valid
     * window: masked-window eval 4.7% vs 86% top-1). Positions are
     * drafter-relative, so restarting at 0 is exact. */
    for (int i = 0; i < 3; i++) s->graph.dspark_n_raw[i] = 0;
    s->graph.dspark_prompt_n = 0;
}



void pulsar_session::rewind(int pos) {
    auto *s = this;
    if (pos < 0) pos = 0;
    if (pos > s->checkpoint.len) pos = s->checkpoint.len;
    s->checkpoint.len = pos;
    s->spec.dspark_n_pending = 0;
    s->spec.spec_carry_valid = false;
    spec_quench_reset(s);
    /* Rewound positions' drafter rows are stale; empty the window (it refills
     * from the prompt capture on the next prefill, or from commits). */
    for (int i = 0; i < 3; i++) s->graph.dspark_n_raw[i] = 0;
    s->graph.dspark_prompt_n = 0;
}



int pulsar_session::pos() {
    auto *s = this;
    return s->checkpoint.len;
}



int pulsar_session::ctx() {
    auto *s = this;
    return s->ctx_size;
}



int pulsar_session_prefill_cap(pulsar_session *s) {
    return s ? (int)s->prefill_cap : 0;
}



/* Multi-session serving: is interrupting pulsar_session_sync() at a chunk
 * boundary (cancel callback) and re-issuing the sync bit-identical to letting
 * it run to completion?
 *
 * Two conditions must hold, and the return value encodes both:
 *
 *   - gpu_graph_prefill_chunked_range caps resumed (start != 0) chunks at
 *     raw_cap. If this session's cold chunks are larger (prefill_cap >
 *     raw_cap), a resumed prefill would re-chunk on different boundaries,
 *     changing batch shapes and therefore cuBLASLt algo selection; exact
 *     replay is lost. Return 0: the caller must not interrupt at all.
 *
 *   - Below gpu_graph_resume_prefill_min_tokens() remaining tokens,
 *     pulsar_session_sync extends the checkpoint by single-token decode evals
 *     instead of a final batched chunk. Interrupting with less than this left
 *     would change which path evaluates the tail. Return that minimum, so the
 *     caller only interrupts while (target - checkpoint) >= the returned
 *     value. (When resume is disabled via PULSAR_CUDA_RESUME_PREFILL_MIN<=0 the
 *     minimum is UINT32_MAX and the comparison never permits interruption.) */
uint32_t pulsar_session::prefill_quantum_min_suffix() const {
    auto *s = this;
    if (!s) return 0;
    if (s->graph.prefill_cap > s->graph.raw_cap) return 0;
    /* A cold (start==0) chunk loop trims each non-final chunk end DOWN to the
     * compress-ratio LCM, while a resumed (start!=0) loop snaps to absolute
     * prefill_cap boundaries. The two produce the same chunk ends only when
     * prefill_cap itself is LCM-aligned (true for the 4096/8192 defaults; a
     * hand-set --prefill-chunk may not be). */
    uint32_t align = 1;
    for (uint32_t il = 0; il < PULSAR_N_LAYER; il++) {
        const uint32_t r = pulsar_layer_compress_ratio(il);
        if (r > 1 && align % r != 0) {
            uint32_t a = align, b = r;
            while (b) { const uint32_t t = a % b; a = b; b = t; }
            align *= r / a;
        }
    }
    if (align > 1 && s->graph.prefill_cap % align != 0) return 0;
    return gpu_graph_resume_prefill_min_tokens();
}

