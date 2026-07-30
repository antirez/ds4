#include "pulsar_engine_internal.h"



bool imatrix_collector_init(pulsar_imatrix_collector *c, uint32_t cap_tokens, const char *dataset_path) {
    memset(c, 0, sizeof(*c));
    c->cap_tokens = cap_tokens ? cap_tokens : 1u;
    c->dataset_path = dataset_path;
    const size_t gate_n = (size_t)PULSAR_N_LAYER * PULSAR_N_EXPERT * PULSAR_N_EMBD;
    const size_t down_n = (size_t)PULSAR_N_LAYER * PULSAR_N_EXPERT * PULSAR_N_FF_EXP;
    c->gate_up_sum2 = (float *)xcalloc(gate_n, sizeof(c->gate_up_sum2[0]));
    c->down_sum2 = (float *)xcalloc(down_n, sizeof(c->down_sum2[0]));
    c->ffn_norm_buf = (float *)xmalloc((size_t)c->cap_tokens * PULSAR_N_EMBD * sizeof(c->ffn_norm_buf[0]));
    c->routed_mid_buf = (float *)xmalloc((size_t)c->cap_tokens * PULSAR_N_EXPERT_USED * PULSAR_N_FF_EXP * sizeof(c->routed_mid_buf[0]));
    c->routed_mid_f16_buf = (uint16_t *)xmalloc((size_t)c->cap_tokens * PULSAR_N_EXPERT_USED * PULSAR_N_FF_EXP * sizeof(c->routed_mid_f16_buf[0]));
    c->selected_buf = (int *)xmalloc((size_t)c->cap_tokens * PULSAR_N_EXPERT_USED * sizeof(c->selected_buf[0]));
    c->sq_tmp = (float *)xmalloc((size_t)PULSAR_N_EMBD * sizeof(c->sq_tmp[0]));
    return c->gate_up_sum2 && c->down_sum2 && c->ffn_norm_buf &&
           c->routed_mid_buf && c->routed_mid_f16_buf && c->selected_buf && c->sq_tmp;
}



void imatrix_collector_free(pulsar_imatrix_collector *c) {
    if (!c) return;
    free(c->gate_up_sum2);
    free(c->down_sum2);
    free(c->ffn_norm_buf);
    free(c->routed_mid_buf);
    free(c->routed_mid_f16_buf);
    free(c->selected_buf);
    free(c->sq_tmp);
    memset(c, 0, sizeof(*c));
}



static float *imatrix_gate_up_ptr(pulsar_imatrix_collector *c, uint32_t il, uint32_t expert) {
    return c->gate_up_sum2 + ((size_t)il * PULSAR_N_EXPERT + expert) * PULSAR_N_EMBD;
}



static float *imatrix_down_ptr(pulsar_imatrix_collector *c, uint32_t il, uint32_t expert) {
    return c->down_sum2 + ((size_t)il * PULSAR_N_EXPERT + expert) * PULSAR_N_FF_EXP;
}



static bool imatrix_collect_layer_batch(
        pulsar_imatrix_collector *c,
        pulsar_gpu_graph         *g,
        uint32_t               il,
        uint32_t               n_tokens) {
    if (!c || n_tokens == 0) return true;
    if (n_tokens > c->cap_tokens) return false;

    const uint64_t norm_bytes = (uint64_t)n_tokens * PULSAR_N_EMBD * sizeof(float);
    const uint64_t mid_elems = (uint64_t)n_tokens * PULSAR_N_EXPERT_USED * PULSAR_N_FF_EXP;
    const uint64_t mid_bytes = mid_elems * (g->batch_routed_mid_is_f16 ? sizeof(uint16_t) : sizeof(float));
    const uint64_t sel_bytes = (uint64_t)n_tokens * PULSAR_N_EXPERT_USED * sizeof(int);
    void *mid_dst = g->batch_routed_mid_is_f16
        ? (void *)c->routed_mid_f16_buf
        : (void *)c->routed_mid_buf;
    if (pulsar_gpu_tensor_read(g->batch_ffn_norm, 0, c->ffn_norm_buf, norm_bytes) == 0 ||
        pulsar_gpu_tensor_read(g->batch_routed_mid, 0, mid_dst, mid_bytes) == 0 ||
        pulsar_gpu_tensor_read(g->batch_router_selected, 0, c->selected_buf, sel_bytes) == 0)
    {
        return false;
    }

    for (uint32_t t = 0; t < n_tokens; t++) {
        const float *x = c->ffn_norm_buf + (size_t)t * PULSAR_N_EMBD;
        for (uint32_t i = 0; i < PULSAR_N_EMBD; i++) c->sq_tmp[i] = x[i] * x[i];

        for (uint32_t slot = 0; slot < PULSAR_N_EXPERT_USED; slot++) {
            const int expert = c->selected_buf[(size_t)t * PULSAR_N_EXPERT_USED + slot];
            if (expert < 0 || (uint32_t)expert >= PULSAR_N_EXPERT) continue;

            float *gate_up = imatrix_gate_up_ptr(c, il, (uint32_t)expert);
            for (uint32_t i = 0; i < PULSAR_N_EMBD; i++) gate_up[i] += c->sq_tmp[i];
            c->gate_up_count[il][expert]++;

            float *down = imatrix_down_ptr(c, il, (uint32_t)expert);
            const size_t mid_off = ((size_t)t * PULSAR_N_EXPERT_USED + slot) * PULSAR_N_FF_EXP;
            if (g->batch_routed_mid_is_f16) {
                const uint16_t *mid = c->routed_mid_f16_buf + mid_off;
                for (uint32_t i = 0; i < PULSAR_N_FF_EXP; i++) {
                    const float v = f16_to_f32(mid[i]);
                    down[i] += v * v;
                }
            } else {
                const float *mid = c->routed_mid_buf + mid_off;
                for (uint32_t i = 0; i < PULSAR_N_FF_EXP; i++) down[i] += mid[i] * mid[i];
            }
            c->down_count[il][expert]++;
            c->observed_routes++;
        }
    }
    c->observed_tokens += n_tokens;
    c->chunks++;
    return true;
}



static void imatrix_write_i32(FILE *fp, int32_t v) {
    if (fwrite(&v, sizeof(v), 1, fp) != 1) pulsar_die("failed to write imatrix");
}



static void imatrix_write_entry(
        FILE       *fp,
        const char *name,
        const float *sum2,
        const uint32_t *counts,
        uint32_t n_expert,
        uint32_t n_col) {
    const int32_t len = (int32_t)strlen(name);
    const int32_t ncall = 1;
    const int32_t nval = (int32_t)((uint64_t)n_expert * n_col);
    imatrix_write_i32(fp, len);
    if (fwrite(name, 1, (size_t)len, fp) != (size_t)len) pulsar_die("failed to write imatrix name");
    imatrix_write_i32(fp, ncall);
    imatrix_write_i32(fp, nval);

    float *tmp = (float *)xmalloc((size_t)n_col * sizeof(tmp[0]));
    for (uint32_t e = 0; e < n_expert; e++) {
        const uint32_t count = counts[e];
        const float *src = sum2 + (size_t)e * n_col;
        if (count == 0) {
            for (uint32_t i = 0; i < n_col; i++) tmp[i] = 1.0f;
        } else {
            const float inv = 1.0f / (float)count;
            for (uint32_t i = 0; i < n_col; i++) tmp[i] = src[i] * inv;
        }
        if (fwrite(tmp, sizeof(tmp[0]), n_col, fp) != n_col) pulsar_die("failed to write imatrix values");
    }
    free(tmp);
}



bool imatrix_collector_save(
        const pulsar_imatrix_collector *c,
        const pulsar_weights           *weights,
        const char                  *path) {
    FILE *fp = fopen(path, "wb");
    if (!fp) {
        fprintf(stderr, "pulsar: failed to open imatrix output %s: %s\n", path, strerror(errno));
        return false;
    }

    const int32_t entries = (int32_t)(PULSAR_N_LAYER * 3);
    imatrix_write_i32(fp, entries);
    for (uint32_t il = 0; il < PULSAR_N_LAYER; il++) {
        const pulsar_layer_weights *layer = &weights->layer[il];
        char name[256];
        snprintf(name, sizeof(name), "%.*s", (int)layer->ffn_gate_exps->name.len, layer->ffn_gate_exps->name.ptr);
        imatrix_write_entry(fp, name,
                            c->gate_up_sum2 + (size_t)il * PULSAR_N_EXPERT * PULSAR_N_EMBD,
                            c->gate_up_count[il],
                            PULSAR_N_EXPERT,
                            PULSAR_N_EMBD);
        snprintf(name, sizeof(name), "%.*s", (int)layer->ffn_up_exps->name.len, layer->ffn_up_exps->name.ptr);
        imatrix_write_entry(fp, name,
                            c->gate_up_sum2 + (size_t)il * PULSAR_N_EXPERT * PULSAR_N_EMBD,
                            c->gate_up_count[il],
                            PULSAR_N_EXPERT,
                            PULSAR_N_EMBD);
        snprintf(name, sizeof(name), "%.*s", (int)layer->ffn_down_exps->name.len, layer->ffn_down_exps->name.ptr);
        imatrix_write_entry(fp, name,
                            c->down_sum2 + (size_t)il * PULSAR_N_EXPERT * PULSAR_N_FF_EXP,
                            c->down_count[il],
                            PULSAR_N_EXPERT,
                            PULSAR_N_FF_EXP);
    }

    const int32_t chunks = (int32_t)c->chunks;
    imatrix_write_i32(fp, chunks);
    const char *dataset = c->dataset_path ? c->dataset_path : "";
    const int32_t dataset_len = (int32_t)strlen(dataset);
    imatrix_write_i32(fp, dataset_len);
    if (dataset_len && fwrite(dataset, 1, (size_t)dataset_len, fp) != (size_t)dataset_len) {
        pulsar_die("failed to write imatrix dataset name");
    }

    if (fclose(fp) != 0) {
        fprintf(stderr, "pulsar: failed to close imatrix output %s: %s\n", path, strerror(errno));
        return false;
    }
    return true;
}



bool gpu_graph_reset_prefill_state(pulsar_gpu_graph *g) {
    memset(g->layer_n_comp, 0, sizeof(g->layer_n_comp));
    memset(g->layer_n_index_comp, 0, sizeof(g->layer_n_index_comp));
    for (uint32_t il = 0; il < PULSAR_N_LAYER; il++) {
        const uint32_t ratio = pulsar_layer_compress_ratio(il);
        if (ratio == 0) continue;
        const uint32_t coff = ratio == 4 ? 2u : 1u;
        const uint64_t attn_width = (uint64_t)coff * PULSAR_N_HEAD_DIM;
        const uint64_t attn_rows = (uint64_t)coff * ratio;
        if (!gpu_tensor_fill_f32(g->layer_attn_state_kv[il], 0.0f, attn_width * attn_rows)) return false;
        if (!gpu_tensor_fill_f32(g->layer_attn_state_score[il], PULSAR_NEG_INF, attn_width * attn_rows)) return false;
        if (ratio == 4) {
            const uint64_t index_width = (uint64_t)coff * PULSAR_N_INDEXER_HEAD_DIM;
            const uint64_t index_rows = (uint64_t)coff * ratio;
            if (!gpu_tensor_fill_f32(g->layer_index_state_kv[il], 0.0f, index_width * index_rows)) return false;
            if (!gpu_tensor_fill_f32(g->layer_index_state_score[il], PULSAR_NEG_INF, index_width * index_rows)) return false;
        }
    }
    return true;
}



/* Execute graph-backend prefill in layer-major order so intermediate
 * activations stay on the GPU and cache state is built exactly once. */
static void gpu_graph_report_prefill_display_progress(
        pulsar_session_progress_fn display_progress,
        void                   *display_progress_ud,
        uint32_t                start,
        uint32_t                n_tokens,
        uint32_t                layer_done,
        int                     total) {
    if (!display_progress) return;
    if (layer_done > (uint32_t)PULSAR_N_LAYER) layer_done = (uint32_t)PULSAR_N_LAYER;
    uint64_t done = (uint64_t)n_tokens * layer_done / (uint32_t)PULSAR_N_LAYER;
    if (layer_done == (uint32_t)PULSAR_N_LAYER) done = n_tokens;
    display_progress(display_progress_ud, "prefill_display",
                     (int)(start + (uint32_t)done), total);
}



/* Bulk anchor-hidden dump for drafter retraining: after a chunk syncs, append
 * one record to PULSAR_DSPARK_PREFILL_DUMP -- {u32 n, u32 start, i32 ids[n],
 * f32 h0[n*4096], f32 h1[...], f32 h2[...]} -- and clear the arm flag. Records
 * with start==0 mark request boundaries for the trainer. Failure paths clear
 * the arm without writing so the verify batch path never sees a stale arm. */
static FILE *g_dspark_bulk_dump;

static void dspark_bulk_dump_close(void) {
    if (g_dspark_bulk_dump) {
        fclose(g_dspark_bulk_dump);
        g_dspark_bulk_dump = NULL;
    }
}

static void dspark_bulk_drain(pulsar_gpu_graph *g, const token_vec *prompt,
                              uint32_t start, uint32_t n, bool ok) {
    if (!g->dspark_bulk_n) return;
    const uint32_t cap_n = g->dspark_bulk_n < n ? g->dspark_bulk_n : n;
    g->dspark_bulk_n = 0;
    if (!ok) return;
    /* Prompt-window capture: keep the last <=128 positions' anchor hiddens in
     * the position%128 ring so generation can seed the drafter's context
     * window (contiguous positions -> at most two segment copies per layer). */
    if (g->dspark_prompt_h[0] && cap_n > 0) {
        const uint32_t win = PULSAR_DSPARK_DRAFT_WINDOW;
        const uint32_t take = cap_n < win ? cap_n : win;
        const uint32_t p0 = start + cap_n - take;
        for (int s2 = 0; s2 < 3; s2++) {
            uint32_t done = 0;
            while (done < take) {
                const uint32_t pos = p0 + done;
                const uint32_t slot = pos % win;
                uint32_t run = win - slot;
                if (run > take - done) run = take - done;
                (void)pulsar_gpu_tensor_copy(g->dspark_prompt_h[s2],
                                          (uint64_t)slot * PULSAR_N_EMBD * sizeof(float),
                                          g->dspark_bulk_h[s2],
                                          (uint64_t)(pos - start) * PULSAR_N_EMBD * sizeof(float),
                                          (uint64_t)run * PULSAR_N_EMBD * sizeof(float));
                done += run;
            }
        }
        /* Contiguity guard: after a rewind + partial re-prefill the ring may
         * hold rows from an older prompt below `start`; mark them invalid. */
        if (start == 0 || start != g->dspark_prompt_n) g->dspark_prompt_lo = start;
        g->dspark_prompt_n = start + cap_n;
        /* Any prefill replaces the generation context, so the drafter's
         * committed-row window is stale by definition: empty it and let the
         * next generation start reseed from the prompt ring. (The server's
         * between-request reset does not go through invalidate/rewind, so
         * this is the invariant that catches every new prompt.) */
        for (int i2 = 0; i2 < 3; i2++) g->dspark_n_raw[i2] = 0;
    }
    const char *path = getenv("PULSAR_DSPARK_PREFILL_DUMP");
    if (!path || !path[0] || !g->dspark_bulk_h[0]) return;
    static FILE *f = NULL;
    static float *host = NULL;
    if (!f) {
        f = fopen(path, "wb");
        /* Process-lifetime debug stream: close it at exit so the final
         * buffered record is not lost if a caller skips the fflush below. */
        if (f) atexit(dspark_bulk_dump_close);
    }
    if (!f) return;
    g_dspark_bulk_dump = f;
    if (!host) host = (float *)xmalloc((size_t)g->prefill_cap * PULSAR_N_EMBD * sizeof(float));
    uint32_t hdr[2] = { cap_n, start };
    fwrite(hdr, sizeof(uint32_t), 2, f);
    fwrite(prompt->v + start, sizeof(int32_t), cap_n, f);
    for (int s = 0; s < 3; s++) {
        if (!pulsar_gpu_tensor_read(g->dspark_bulk_h[s], 0, host,
                                 (uint64_t)cap_n * PULSAR_N_EMBD * sizeof(float)))
            return;
        fwrite(host, sizeof(float), (size_t)cap_n * PULSAR_N_EMBD, f);
    }
    fflush(f);
}

static bool gpu_graph_prefill_layer_major_inner(
        pulsar_gpu_graph *g,
        const pulsar_model       *model,
        const pulsar_weights     *weights,
        const token_vec       *prompt,
        uint32_t               start,
        uint32_t               n_tokens,
        float                 *logits,
        bool                   show_progress,
        pulsar_imatrix_collector *imatrix,
        pulsar_session_progress_fn display_progress,
        void                  *display_progress_ud);

bool gpu_graph_prefill_layer_major(
        pulsar_gpu_graph *g,
        const pulsar_model       *model,
        const pulsar_weights     *weights,
        const token_vec       *prompt,
        uint32_t               start,
        uint32_t               n_tokens,
        float                 *logits,
        bool                   show_progress,
        pulsar_imatrix_collector *imatrix,
        pulsar_session_progress_fn display_progress,
        void                  *display_progress_ud) {
    const bool ok = gpu_graph_prefill_layer_major_inner(g, model, weights, prompt,
                                                        start, n_tokens, logits,
                                                        show_progress, imatrix,
                                                        display_progress,
                                                        display_progress_ud);
    dspark_bulk_drain(g, prompt, start, n_tokens, ok);
    return ok;
}

static bool gpu_graph_prefill_layer_major_inner(
        pulsar_gpu_graph *g,
        const pulsar_model       *model,
        const pulsar_weights     *weights,
        const token_vec       *prompt,
        uint32_t               start,
        uint32_t               n_tokens,
        float                 *logits,
        bool                   show_progress,
        pulsar_imatrix_collector *imatrix,
        pulsar_session_progress_fn display_progress,
        void                  *display_progress_ud) {
    if (n_tokens == 0 || n_tokens > g->prefill_cap) return false;
    if (start > (uint32_t)prompt->len) return false;
    if (n_tokens > (uint32_t)prompt->len - start) return false;

    if (display_progress)
        display_progress(display_progress_ud, "prefill_display", (int)start, prompt->len);

    bool ok = gpu_graph_upload_prompt_tokens(g->prefill_tokens, prompt, start, n_tokens);
    if (!ok) return false;


    if (!gpu_graph_warmup_prefill_kernels(g, model, weights, n_tokens)) return false;

    /* Bulk anchor-hidden capture (drafter retraining): armed per chunk, after
     * warmup so warmup encodes don't pollute the buffers; drained (and cleared)
     * by gpu_graph_prefill_chunked_range after the chunk syncs. */
    g->dspark_bulk_n = g->dspark_bulk_h[0] ? n_tokens : 0;

    const bool split_profile = getenv("PULSAR_CUDA_GRAPH_PREFILL_SPLIT_PROFILE") != NULL;
    /*
     * A full long-prompt prefill can keep the GPU busy for a long time. Split
     * non-tiny prefills when a frontend asked for display progress: completed
     * layer command buffers are real scheduling/keepalive points, while
     * callbacks emitted while encoding one huge command buffer would only be
     * cosmetic.
     */
    /* Do NOT split (per-layer cudaDeviceSynchronize) just to fire a display
     * progress callback: on this backend end_commands() is a full device sync,
     * so per-layer progress drains the async launch pipeline 43x/chunk. Chunk-
     * level progress (gpu_graph_prefill_chunked_range) still fires; intra-chunk
     * progress is dropped in favor of throughput. */
    const bool split_commands = split_profile ||
                                n_tokens > 2048 || imatrix != NULL;
    const bool profile = getenv("PULSAR_CUDA_GRAPH_PREFILL_PROFILE") != NULL || split_profile;
    const double t0 = profile ? now_sec() : 0.0;
    double encode_s = 0.0;
    double execute_s = 0.0;

    if (!split_commands) {
        ok = gpu_graph_upload_prompt_embeddings_hc(g->batch_cur_hc,
                                                     g->prefill_tokens,
                                                     model,
                                                     weights,
                                                     prompt,
                                                     start,
                                                     n_tokens);
        if (ok) ok = pulsar_gpu_begin_commands() != 0;
        for (uint32_t il = 0; ok && il < PULSAR_N_LAYER; il++) {
            ok = gpu_graph_encode_layer_batch(g,
                                                model,
                                                &weights->layer[il],
                                                il,
                                                start,
                                                n_tokens);
            if (!ok) {
                fprintf(stderr, "pulsar: gpu whole-prefill layer %u encode failed\n", il);
            }
            if (show_progress) {
                fprintf(stderr, "pulsar: gpu prefill layer %u/%u\r", il + 1, (uint32_t)PULSAR_N_LAYER);
                fflush(stderr);
            }
        }
        if (show_progress) fputc('\n', stderr);
        if (display_progress)
            display_progress(display_progress_ud, "prefill_display",
                             (int)(start + n_tokens), prompt->len);

        const uint64_t hc_dim = (uint64_t)PULSAR_N_HC * PULSAR_N_EMBD;
        uint32_t output_row = (uint32_t)n_tokens - 1u;
        const char *output_row_env = getenv("PULSAR_CUDA_GRAPH_OUTPUT_ROW");
        if (output_row_env && output_row_env[0]) {
            char *end = NULL;
            unsigned long v = strtoul(output_row_env, &end, 10);
            if (end != output_row_env && v < (unsigned long)n_tokens) {
                output_row = (uint32_t)v;
            }
        }
        pulsar_gpu_tensor *saved_cur = g->cur_hc;
        pulsar_gpu_tensor *last_hc = NULL;
        if (ok && logits) {
            last_hc = gpu_graph_hc_row_view(g->batch_cur_hc, output_row, hc_dim);
            ok = last_hc != NULL;
        }
        if (ok && logits) {
            g->cur_hc = last_hc;
            ok = gpu_graph_encode_output_head(g, model, weights, weights->output->dim[1]);
            g->cur_hc = saved_cur;
        }

        const double t_encoded = profile ? now_sec() : 0.0;
        if (ok) ok = pulsar_gpu_end_commands() != 0;
        const double t_done = profile ? now_sec() : 0.0;
        g->cur_hc = saved_cur;
        if (last_hc) pulsar_gpu_tensor_free(last_hc);
        if (!ok) {
            if (pulsar_gpu_synchronize() == 0) {
                fprintf(stderr, "pulsar: GPU synchronize after whole-prefill graph failure also failed\n");
            }
            return false;
        }

        const double t_before_read = profile ? now_sec() : 0.0;
        if (logits) {
            ok = pulsar_gpu_tensor_read(g->logits, 0, logits, (uint64_t)PULSAR_N_VOCAB * sizeof(float)) != 0;
        }
        if (profile) {
            const double t_read = now_sec();
            fprintf(stderr,
                    "pulsar: gpu graph prefill total tokens=%u encode=%.3f ms execute=%.3f ms read=%.3f ms total=%.3f ms\n",
                    n_tokens,
                    (t_encoded - t0) * 1000.0,
                    (t_done - t_encoded) * 1000.0,
                    (t_read - t_before_read) * 1000.0,
                    (t_read - t0) * 1000.0);
        }
        return ok;
    }

    double t_layer0 = profile ? now_sec() : 0.0;
    ok = gpu_graph_upload_prompt_embeddings_hc(g->batch_cur_hc,
                                                 g->prefill_tokens,
                                                 model,
                                                 weights,
                                                 prompt,
                                                 start,
                                                 n_tokens);
    const double t_embed_encoded = profile ? now_sec() : 0.0;
    const double t_embed_done = profile ? now_sec() : 0.0;
    if (profile) {
        encode_s += t_embed_encoded - t_layer0;
        execute_s += t_embed_done - t_embed_encoded;
        if (split_profile) {
            fprintf(stderr,
                    "pulsar: GPU layer-major prefill embed encode=%.3f ms execute=%.3f ms\n",
                    (t_embed_encoded - t_layer0) * 1000.0,
                    (t_embed_done - t_embed_encoded) * 1000.0);
        }
    }
    if (!ok) {
        if (pulsar_gpu_synchronize() == 0) {
            fprintf(stderr, "pulsar: GPU synchronize after layer-major prefill embed failure also failed\n");
        }
        return false;
    }

    for (uint32_t il = 0; ok && il < PULSAR_N_LAYER; il++) {
        if (split_profile) {
            const double t_attn0 = now_sec();
            ok = pulsar_gpu_begin_commands() != 0;
            if (ok) ok = gpu_graph_encode_layer_attention_batch(g,
                                                                  model,
                                                                  &weights->layer[il],
                                                                  il,
                                                                  start,
                                                                  n_tokens);
            if (!ok) {
                fprintf(stderr, "pulsar: gpu layer-major prefill layer %u attention encode failed\n", il);
            }
            const double t_attn_encoded = now_sec();
            if (ok) ok = pulsar_gpu_end_commands() != 0;
            const double t_attn_done = now_sec();

            const double t_ffn0 = now_sec();
            if (ok) ok = pulsar_gpu_begin_commands() != 0;
            if (ok) ok = gpu_graph_encode_layer_ffn_batch(g,
                                                            model,
                                                            &weights->layer[il],
                                                            il,
                                                            start,
                                                            n_tokens);
            if (!ok) {
                fprintf(stderr, "pulsar: gpu layer-major prefill layer %u ffn encode failed\n", il);
            }
            if (ok) {
                pulsar_gpu_tensor *tmp = g->batch_cur_hc;
                g->batch_cur_hc = g->batch_next_hc;
                g->batch_next_hc = tmp;
            }
            const double t_ffn_encoded = now_sec();
            if (ok) ok = pulsar_gpu_end_commands() != 0;
            const double t_ffn_done = now_sec();
            if (ok && imatrix) ok = imatrix_collect_layer_batch(imatrix, g, il, (uint32_t)n_tokens);

            encode_s += (t_attn_encoded - t_attn0) + (t_ffn_encoded - t_ffn0);
            execute_s += (t_attn_done - t_attn_encoded) + (t_ffn_done - t_ffn_encoded);
            fprintf(stderr,
                    "pulsar: GPU layer-major prefill layer %u attn encode=%.3f execute=%.3f ms ffn encode=%.3f execute=%.3f ms\n",
                    il,
                    (t_attn_encoded - t_attn0) * 1000.0,
                    (t_attn_done - t_attn_encoded) * 1000.0,
                    (t_ffn_encoded - t_ffn0) * 1000.0,
                    (t_ffn_done - t_ffn_encoded) * 1000.0);
        } else {
            const double t_chunk0 = profile ? now_sec() : 0.0;
            ok = pulsar_gpu_begin_commands() != 0;
            if (ok) ok = gpu_graph_encode_layer_batch(g,
                                                        model,
                                                        &weights->layer[il],
                                                        il,
                                                        start,
                                                        n_tokens);
            if (!ok) {
                fprintf(stderr, "pulsar: gpu layer-major prefill layer %u encode failed\n", il);
            }
            const double t_encoded = profile ? now_sec() : 0.0;
            if (ok) ok = pulsar_gpu_end_commands() != 0;
            const double t_done = profile ? now_sec() : 0.0;
            if (ok && imatrix) ok = imatrix_collect_layer_batch(imatrix, g, il, (uint32_t)n_tokens);
            if (profile) {
                encode_s += t_encoded - t_chunk0;
                execute_s += t_done - t_encoded;
                fprintf(stderr,
                        "pulsar: gpu layer-major prefill layer %u encode=%.3f ms execute=%.3f ms\n",
                        il,
                        (t_encoded - t_chunk0) * 1000.0,
                        (t_done - t_encoded) * 1000.0);
            }
        }
        if (!ok) {
            if (pulsar_gpu_synchronize() == 0) {
                fprintf(stderr, "pulsar: GPU synchronize after layer-major prefill failure also failed\n");
            }
            return false;
        }
        gpu_graph_report_prefill_display_progress(display_progress,
                                                  display_progress_ud,
                                                  start,
                                                  n_tokens,
                                                  il + 1,
                                                  prompt->len);
        if (show_progress) {
            fprintf(stderr, "pulsar: gpu prefill layer %u/%u\r", il + 1, (uint32_t)PULSAR_N_LAYER);
            fflush(stderr);
        }
    }
    if (!ok) {
        if (pulsar_gpu_synchronize() == 0) {
            fprintf(stderr, "pulsar: GPU synchronize after layer-major prefill failure also failed\n");
        }
        return false;
    }
    if (show_progress) fputc('\n', stderr);

    const uint64_t hc_dim = (uint64_t)PULSAR_N_HC * PULSAR_N_EMBD;
    uint32_t output_row = (uint32_t)n_tokens - 1u;
    const char *output_row_env = getenv("PULSAR_CUDA_GRAPH_OUTPUT_ROW");
    if (output_row_env && output_row_env[0]) {
        char *end = NULL;
        unsigned long v = strtoul(output_row_env, &end, 10);
        if (end != output_row_env && v < (unsigned long)n_tokens) {
            output_row = (uint32_t)v;
        }
    }
    pulsar_gpu_tensor *saved_cur = g->cur_hc;
    pulsar_gpu_tensor *last_hc = NULL;

    const double t_head0 = profile ? now_sec() : 0.0;
    if (logits) {
        last_hc = gpu_graph_hc_row_view(g->batch_cur_hc,
                                          output_row,
                                          hc_dim);
        ok = last_hc != NULL;
    }
    if (ok && logits) {
        g->cur_hc = last_hc;
        ok = pulsar_gpu_begin_commands() != 0;
    }
    if (ok && logits) ok = gpu_graph_encode_output_head(g, model, weights, weights->output->dim[1]);
    const double t_head_encoded = profile ? now_sec() : 0.0;
    if (ok && logits) ok = pulsar_gpu_end_commands() != 0;
    const double t_head_done = profile ? now_sec() : 0.0;
    g->cur_hc = saved_cur;
    if (last_hc) pulsar_gpu_tensor_free(last_hc);
    if (!ok) return false;

    const double t_before_read = profile ? now_sec() : 0.0;
    if (logits) {
        ok = pulsar_gpu_tensor_read(g->logits, 0, logits, (uint64_t)PULSAR_N_VOCAB * sizeof(float)) != 0;
    }
    if (profile) {
        const double t_read = now_sec();
        encode_s += t_head_encoded - t_head0;
        execute_s += t_head_done - t_head_encoded;
        if (split_profile) {
            fprintf(stderr,
                    "pulsar: gpu layer-major prefill head encode=%.3f ms execute=%.3f ms\n",
                    (t_head_encoded - t_head0) * 1000.0,
                    (t_head_done - t_head_encoded) * 1000.0);
        }
        fprintf(stderr,
                "pulsar: gpu layer-major prefill total tokens=%u encode=%.3f ms execute=%.3f ms read=%.3f ms total=%.3f ms\n",
                n_tokens,
                encode_s * 1000.0,
                execute_s * 1000.0,
                (t_read - t_before_read) * 1000.0,
                (t_read - t0) * 1000.0);
    }
    return ok;
}



bool gpu_graph_prefill_raw_swa(
        pulsar_gpu_graph *g,
        const pulsar_model       *model,
        const pulsar_weights     *weights,
        const token_vec       *prompt,
        int                    n_tokens,
        float                 *logits,
        bool                   show_progress,
        pulsar_session_progress_fn display_progress,
        void                  *display_progress_ud,
        pulsar_session_cancel_fn  cancel,
        void                  *cancel_ud,
        bool                  *cancelled) {
    if (n_tokens <= 0 || n_tokens > prompt->len) return false;
    if ((uint32_t)n_tokens > g->prefill_cap) return false;
    /* The layer-major fallback below may submit the whole short prefill as one
     * GPU command buffer.  Once that command is in flight there is no useful
     * safe prefix to expose: by the time cancellation can be observed again,
     * the prompt has already been fully read and the KV is valid.  Let the
     * caller observe the pending interrupt at generation time instead. */
    (void)cancel;
    (void)cancel_ud;
    (void)cancelled;
    return gpu_graph_prefill_layer_major(g,
                                           model,
                                           weights,
                                           prompt,
                                           0,
                                           (uint32_t)n_tokens,
                                           logits,
                                           show_progress,
                                           NULL,
                                           display_progress,
                                           display_progress_ud);
}



/* Prefill a contiguous token range in fixed-size chunks.
 *
 * The common case starts at token zero, but server sessions also use this to
 * extend an existing KV cache with a long suffix.  Resumed chunks are aligned
 * to the same absolute prefill-cap boundaries used by a cold full prompt, so
 * compression windows and row finalization follow the same schedule after the
 * cached prefix.
 */
static uint32_t gcd_u32(uint32_t a, uint32_t b) {
    while (b) { uint32_t t = a % b; a = b; b = t; }
    return a;
}

bool gpu_graph_prefill_chunked_range(
        pulsar_gpu_graph *g,
        const pulsar_model       *model,
        const pulsar_weights     *weights,
        const token_vec       *prompt,
        uint32_t               start,
        uint32_t               n_tokens,
        float                 *logits,
        bool                   show_progress,
        pulsar_session_progress_fn progress,
        void                  *progress_ud,
        pulsar_session_progress_fn display_progress,
        void                  *display_progress_ud,
        pulsar_imatrix_collector *imatrix,
        pulsar_session_cancel_fn  cancel,
        void                  *cancel_ud,
        bool                  *cancelled) {
    if (n_tokens == 0 || g->prefill_cap == 0) return false;
    if (start > (uint32_t)prompt->len) return false;
    if (n_tokens > (uint32_t)prompt->len - start) return false;

    uint32_t chunk_cap = g->prefill_cap;
    if (start != 0 && chunk_cap > g->raw_cap) chunk_cap = g->raw_cap;
    if (chunk_cap == 0) return false;

    const bool profile = getenv("PULSAR_CUDA_GRAPH_PREFILL_PROFILE") != NULL;
    const double t0 = profile ? now_sec() : 0.0;
    const uint32_t end = start + n_tokens;

    if (progress) {
        progress(progress_ud, "prefill_chunk", (int)start, prompt->len);
    }
    if (display_progress) {
        display_progress(display_progress_ud, "prefill_display", (int)start, prompt->len);
    }

    for (uint32_t pos0 = start; pos0 < end; ) {
        if (cancel && cancel(cancel_ud)) {
            if (cancelled) *cancelled = true;
            return true;
        }
        const uint32_t remaining = end - pos0;
        uint32_t local_cap = chunk_cap;
        if (start != 0 && g->prefill_cap != 0) {
            const uint32_t mod = pos0 % g->prefill_cap;
            if (mod != 0) {
                const uint32_t to_boundary = g->prefill_cap - mod;
                if (to_boundary < local_cap) local_cap = to_boundary;
            }
        }
        uint32_t chunk = remaining < local_cap ? remaining : local_cap;
        /* Keep every NON-final chunk boundary aligned to the layer compress
         * ratios (LCM, i.e. 4 for the ratio-4 layers): one unaligned boundary
         * makes pos0 unaligned for every later chunk, and each of those takes
         * the per-token compressor fallback instead of the batched aligned
         * path for the whole rest of the prompt. The final chunk keeps its
         * exact remainder; an unaligned START (continuation from an arbitrary
         * position) pays the fallback for its first chunk only, because that
         * chunk still ENDS on an aligned boundary. */
        if (chunk < remaining) {
            uint32_t align = 1;
            for (uint32_t il = 0; il < PULSAR_N_LAYER; il++) {
                const uint32_t r = pulsar_layer_compress_ratio(il);
                if (r > 1 && align % r != 0) align *= r / gcd_u32(align, r);
            }
            if (align > 1) {
                const uint32_t aligned_end = ((pos0 + chunk) / align) * align;
                if (aligned_end > pos0) chunk = aligned_end - pos0;
            }
        }
        const uint32_t chunk_end = pos0 + chunk;
        /* Only the final chunk's logits are consumed (the progress callback below
         * reports position only, never reads logits). Running the full output
         * head + vocab GEMM + readback on every non-final chunk is wasted work
         * whose result is immediately overwritten. */
        float *chunk_logits = (chunk_end == end) ? logits : NULL;
        bool ok = gpu_graph_prefill_layer_major(g,
                                                  model,
                                                  weights,
                                                  prompt,
                                                  pos0,
                                                  chunk,
                                                  chunk_logits,
                                                  show_progress,
                                                  imatrix,
                                                  display_progress,
                                                  display_progress_ud);
        if (!ok) {
            if (pulsar_gpu_synchronize() == 0) {
                fprintf(stderr, "pulsar: GPU synchronize after chunked prefill failure also failed\n");
            }
            return false;
        }
        if (progress) {
            progress(progress_ud, "prefill_chunk", (int)chunk_end, prompt->len);
        }
        if (display_progress) {
            display_progress(display_progress_ud, "prefill_display", (int)chunk_end, prompt->len);
        }
        if (cancel && cancel(cancel_ud)) {
            if (cancelled) *cancelled = true;
            return true;
        }
        pos0 = chunk_end;
    }
    if (show_progress) fputc('\n', stderr);
    if (profile) {
        const double t_read = now_sec();
        fprintf(stderr,
                "pulsar: gpu chunked prefill start=%u tokens=%u chunk=%u total=%.3f ms\n",
                start,
                n_tokens,
                chunk_cap,
                (t_read - t0) * 1000.0);
    }
    return true;
}



/* Long prompts are prefetched in fixed-size chunks.  Chunks bound transient
 * attention buffers while preserving the same final KV/cache state. */
bool gpu_graph_prefill_chunked(
        pulsar_gpu_graph *g,
        const pulsar_model       *model,
        const pulsar_weights     *weights,
        const token_vec       *prompt,
        int                    n_tokens,
        float                 *logits,
        bool                   show_progress,
        pulsar_session_progress_fn progress,
        void                  *progress_ud,
        pulsar_session_progress_fn display_progress,
        void                  *display_progress_ud,
        pulsar_session_cancel_fn  cancel,
        void                  *cancel_ud,
        bool                  *cancelled) {
    if (n_tokens <= 0) return false;
    return gpu_graph_prefill_chunked_range(g,
                                             model,
                                             weights,
                                             prompt,
                                             0,
                                             (uint32_t)n_tokens,
                                             logits,
                                             show_progress,
                                             progress,
                                             progress_ud,
                                             display_progress,
                                             display_progress_ud,
                                             NULL,
                                             cancel,
                                             cancel_ud,
                                             cancelled);
}



/* Layer-major speculative target verifier for tiny draft suffixes.
 *
 * This is the first production-shaped verifier attempt: unlike repeated decode
 * it runs the target model layer-by-layer for the whole speculative suffix, and
 * unlike the diagnostic path it does not read back full logits for every row.
 * The verifier returns the row top-1 ids needed for acceptance.  The caller
 * then reads exactly one logits row: the row that becomes the new continuation
 * state.  It still reuses the existing batch layer kernels, so it is not yet
 * the final hand-written N=2/N=4 decode microbatch, but it exercises the right
 * verifier contract and removes the obvious diagnostic overheads first. */
bool gpu_graph_verify_suffix_tops(
        pulsar_gpu_graph *g,
        const pulsar_model       *model,
        const pulsar_weights     *weights,
        const token_vec       *prompt,
        uint32_t               start,
        uint32_t               n_tokens,
        int                   *row_tops,
        float                 *row_logits) {
    if (n_tokens == 0 || n_tokens > g->prefill_cap || !g->spec_logits) return false;
    if (start > (uint32_t)prompt->len || n_tokens > (uint32_t)prompt->len - start) return false;
    const uint32_t top_rows = n_tokens > 1 ? n_tokens - 1 : 0;
    if (top_rows && !row_tops) return false;

    /* Diagnostic flag: read the environment once per process (C has no
     * static-initializer form for this, so use the repo's -1 sentinel idiom). */
    static int timing_env = -1;
    if (timing_env < 0) timing_env = getenv("PULSAR_SPEC_VERIFY_TIMING") != NULL;
    const bool timing = timing_env != 0;
    const double t0 = timing ? now_sec() : 0.0;
    bool ok = gpu_graph_upload_prompt_tokens(g->prefill_tokens, prompt, start, n_tokens);
    if (ok) ok = gpu_graph_upload_prompt_embeddings_hc(g->batch_cur_hc,
                                                         g->prefill_tokens,
                                                         model,
                                                         weights,
                                                         prompt,
                                                         start,
                                                         n_tokens);
    const double t_uploaded = timing ? now_sec() : 0.0;
    if (!ok) return false;

    /* PULSAR_DECODE_DESCR level 2 (Tier-2 diagnostic): run this verify batch
     * through the banked multiseq machinery as a one-bank batch over the
     * current bank — per-bank frontiers, banked emit loop, banked kernels.
     * Byte-gated vs classic with the generic kernel tiers pinned (see
     * gpu_graph_decode_descr_enabled).  The spec rollback path restores the
     * scalar counters; the per-bank ms counters are re-captured from them at
     * every step_begin (capture_cur), so a partial-accept rollback between
     * verifies cannot leave the mirrors stale. */
    const bool mseq_diag = gpu_graph_decode_descr_enabled() >= 2;
    if (mseq_diag) {
        int32_t *dpos = (int32_t *)xmalloc((size_t)n_tokens * sizeof(int32_t));
        int32_t *dseq = (int32_t *)xmalloc((size_t)n_tokens * sizeof(int32_t));
        const int32_t bank = g->banks.n_banks ? (int32_t)g->banks.cur_bank : 0;
        for (uint32_t t = 0; t < n_tokens; t++) {
            dpos[t] = (int32_t)(start + t);
            dseq[t] = bank;
        }
        ok = gpu_graph_multiseq_step_begin(g, dpos, dseq, n_tokens, true);
        free(dseq);
        free(dpos);
        if (!ok) return false;
    }

    const double t_layers_encode0 = timing ? now_sec() : 0.0;
    ok = pulsar_gpu_begin_commands() != 0;
    for (uint32_t il = 0; ok && il < PULSAR_N_LAYER; il++) {
        ok = gpu_graph_encode_layer_batch(g,
                                            model,
                                            &weights->layer[il],
                                            il,
                                            start,
                                            n_tokens);
    }
    const double t_layers_encoded = timing ? now_sec() : 0.0;
    if (ok) {
        ok = pulsar_gpu_end_commands() != 0;
    } else {
        (void)pulsar_gpu_synchronize();
    }
    if (mseq_diag) {
        /* Disarm + frontier self-check even when the sweep failed. */
        const bool end_ok = gpu_graph_multiseq_step_end(g);
        if (ok) ok = end_ok;
    }
    const double t_layers_done = timing ? now_sec() : 0.0;
    if (!ok) return false;

    const double t_head_encode0 = timing ? now_sec() : 0.0;
    ok = pulsar_gpu_begin_commands() != 0;
    if (ok) ok = gpu_graph_encode_output_head_batch(g,
                                                      model,
                                                      weights,
                                                      n_tokens,
                                                      weights->output->dim[1]);
    if (ok) {
        if (top_rows == 1) {
            /* Common K=2 verify case: top_k=1 over n_vocab → use the dedicated
             * argmax kernel (single-block tree-reduce) instead of the legacy
             * indexer_topk_kernel's single-thread O(n_vocab * top_k) fall-through. */
            ok = pulsar_gpu_argmax_tensor(g->comp_selected,
                                       g->spec_logits,
                                       PULSAR_N_VOCAB) != 0;
        } else if (top_rows) {
            /* top-1 of each of the top_rows rows. The legacy
             * indexer_topk_tensor fall-through for this shape launches ONE
             * THREAD per row scanning the whole vocab (~14ms at 4 rows);
             * per-row argmax launches (1024-thread tree-reduce each) do the
             * same job in <1ms and write the same [top_rows] i32 layout. */
            for (uint32_t r = 0; ok && r < top_rows; r++) {
                pulsar_gpu_tensor *row = pulsar_gpu_tensor_view(
                        g->spec_logits,
                        (uint64_t)r * PULSAR_N_VOCAB * sizeof(float),
                        (uint64_t)PULSAR_N_VOCAB * sizeof(float));
                pulsar_gpu_tensor *dst = pulsar_gpu_tensor_view(
                        g->comp_selected,
                        (uint64_t)r * sizeof(int32_t),
                        sizeof(int32_t));
                ok = row && dst &&
                     pulsar_gpu_argmax_tensor(dst, row, PULSAR_N_VOCAB) != 0;
                pulsar_gpu_tensor_free(row);
                pulsar_gpu_tensor_free(dst);
            }
        }
    }
    const double t_head_encoded = timing ? now_sec() : 0.0;
    if (ok) {
        ok = pulsar_gpu_end_commands() != 0;
    } else {
        (void)pulsar_gpu_synchronize();
    }
    const double t_head_done = timing ? now_sec() : 0.0;
    const double t_read0 = timing ? now_sec() : 0.0;
    if (ok && top_rows) {
        ok = pulsar_gpu_tensor_read(g->comp_selected,
                                   0,
                                   row_tops,
                                   (uint64_t)top_rows * sizeof(row_tops[0])) != 0;
    }
    if (ok && row_logits) {
        ok = pulsar_gpu_tensor_read(g->spec_logits,
                                   0,
                                   row_logits,
                                   (uint64_t)n_tokens * PULSAR_N_VOCAB * sizeof(row_logits[0])) != 0;
    }
    if (timing) {
        const double t_done = now_sec();
        fprintf(stderr,
                "pulsar: spec verify suffix tokens=%u start=%u upload=%.3f ms layers_encode=%.3f ms layers_execute=%.3f ms head_encode=%.3f ms head_execute=%.3f ms read=%.3f ms total=%.3f ms ok=%d\n",
                n_tokens,
                start,
                (t_uploaded - t0) * 1000.0,
                (t_layers_encoded - t_layers_encode0) * 1000.0,
                (t_layers_done - t_layers_encoded) * 1000.0,
                (t_head_encoded - t_head_encode0) * 1000.0,
                (t_head_done - t_head_encoded) * 1000.0,
                (t_done - t_read0) * 1000.0,
                (t_done - t0) * 1000.0,
                ok ? 1 : 0);
    }
    return ok;
}



/* Tier-2 batched multi-session decode step: one current token per live bank
 * through ONE weight sweep (this is where the N-fold aggregate decode win
 * comes from — decode is weight-bandwidth-bound, and the batch reads the
 * weights once for all rows).  Structure mirrors gpu_graph_verify_suffix_tops
 * (upload -> layer sweep -> output head -> readback), but the rows are
 * independent sessions at unrelated positions rather than one session's
 * speculative suffix, so the sweep runs armed as a banked multiseq step
 * (per-row positions/seq_id descriptors, per-bank compressor frontiers,
 * emit-before-attention per the driver contract in pulsar_gpu.h).
 *
 * Inputs are packed row-major: row k carries session bank[k]'s current token
 * tokens[k] at absolute position pos[k].  Rows must satisfy the step_begin
 * contract (TRUE bank ids; per-bank runs contiguous with consecutive
 * positions; every bank's frontier position-true; no position-0 rows — for
 * plain decode each bank contributes exactly one row, trivially satisfying
 * the run rules).  The caller owns per-bank host bookkeeping: ms counters
 * current (gpu_graph_bank_counters_capture after any classic per-bank work),
 * and after this step the scalar layer_n_comp counters hold the CROSS-BANK
 * SUPERSET — gpu_graph_bank_counters_install(bank) before any classic
 * single-bank work resumes.  NEVER co-schedule speculation with n_active >= 2
 * (contract; the scheduler's alone->spec / shared->plain switch).
 *
 * logits: out [n_active * PULSAR_N_VOCAB] host rows, row k = bank[k]'s
 * next-token distribution; sampling stays per-session on the host with that
 * session's own sampler state.
 *
 * Returns 1 on success; 0 on a RECOVERABLE rejection (bad args, upload
 * failure, or step_begin contract rejection — no persistent graph state was
 * mutated, the caller may fix the batch and retry); -1 on a FATAL failure
 * (the armed sweep, its frontier self-check, or the output head failed —
 * per-bank KV state can no longer be trusted; the session must be torn
 * down). */
int gpu_graph_decode_multiseq_batch(
        pulsar_gpu_graph *g,
        const pulsar_model       *model,
        const pulsar_weights     *weights,
        const int             *tokens,
        const int32_t         *pos,
        const int32_t         *bank,
        uint32_t               n_active,
        float                 *logits,
        uint32_t              *out_n_rows,
        uint32_t               max_head_runs) {
    /* plan-34 inc 3: the ROW count (n_active) is bounded by prefill_cap (a K-row
     * prefill chunk rides this entry); PULSAR_MSEQ_MAX bounds only the BANK count,
     * enforced per-row in step_begin (seq[t] >= PULSAR_MSEQ_MAX). The pool-count
     * clause below is the bank ceiling; do NOT reinstate an n_active>PULSAR_MSEQ_MAX
     * row ceiling. */
    if (!g || !model || !weights || !tokens || !pos || !bank || !logits ||
        n_active == 0 ||
        n_active > gpu_graph_bank_pool_count(g) * g->prefill_cap ||
        n_active > g->prefill_cap || !g->spec_logits) {
        fprintf(stderr, "pulsar: multiseq decode rejected: bad args (n_active=%u"
                        " pool=%u prefill_cap=%u logits_slab=%s)\n", n_active,
                g ? gpu_graph_bank_pool_count(g) : 0u,
                g ? g->prefill_cap : 0u,
                (g && g->spec_logits) ? "ok" : "MISSING");
        return 0;
    }

    /* Gather each session's current token into the batch input rows.  The
     * token embedding is position/bank-independent, so the existing prompt
     * uploader runs unmodified over a stack copy of the caller's array (this
     * is exactly how classic decode feeds the graph, batched).  The copy
     * (<= PULSAR_MSEQ_MAX ints) keeps the caller's `const int *` honest — the
     * token_vec ABI is non-const and casting it away here would license a
     * write we do not make. */
    /* inc 3: heap the row-indexed token copy (was [PULSAR_MSEQ_MAX]) so a K-row
     * prefill chunk fits. n_active <= prefill_cap. */
    int *cur_tokens = (int *)xmalloc((size_t)n_active * sizeof(int));
    memcpy(cur_tokens, tokens, (size_t)n_active * sizeof(cur_tokens[0]));
    token_vec cur;
    memset(&cur, 0, sizeof(cur));
    cur.v = cur_tokens;
    cur.len = cur.cap = (int)n_active;
    if (!gpu_graph_upload_prompt_tokens(g->prefill_tokens, &cur, 0, n_active) ||
        !gpu_graph_upload_prompt_embeddings_hc(g->batch_cur_hc,
                                               g->prefill_tokens,
                                               model, weights, &cur,
                                               0, n_active)) {
        free(cur_tokens);
        return 0;   /* scratch-only writes so far — recoverable */
    }
    free(cur_tokens);   /* uploaded to device; host copy is dead */

    /* Arm the banked step (validates the driver contract; a rejection here
     * leaves the graph untouched — recoverable). */
    if (!gpu_graph_multiseq_step_begin(g, pos, bank, n_active, false)) return 0;

    /* plan-34 inc 3: emit logits only for the LAST ROW OF EACH per-bank RUN.
     * A K-row prefill run advances the KV by K but only its last row's logits
     * are consumed (the continuation); computing all K vocab GEMMs would be the
     * single biggest GEMM in the model AND overflow the 16-row spec_logits slab.
     * n_runs == n_banks <= PULSAR_MSEQ_MAX <= 16, so the head always fits.
     * DECODE-ONLY (every run length 1 => n_runs == n_active) keeps the identity
     * layout and the SINGLE-BLOCK layers+head path => byte-identical to before
     * (the inc-1/inc-2 gates must stay green). */
    int last_idx[PULSAR_MSEQ_MAX];
    uint32_t n_runs = 0;
    for (uint32_t t = 0; t < n_active; t++) {
        if (t + 1 == n_active || bank[t + 1] != bank[t]) {
            if (n_runs < PULSAR_MSEQ_MAX) last_idx[n_runs] = (int)t;
            n_runs++;
        }
    }
    /* plan-34 inc 5 (LEVER 1): the caller may emit logits for only the FIRST
     * `max_head_runs` runs (0 == all). A fused mixed step whose trailing PREFILL
     * run is not on its final chunk passes max_head_runs = n_dec (the decode banks
     * only): the prefill run's intermediate logits are NEVER consumed, so not
     * computing them changes nothing observable — and, crucially, when the emitted
     * runs are exactly the leading length-1 decode runs (last_idx[r] == r), the head
     * takes the SINGLE-BLOCK identity path (no extra end/gather/begin/end resync,
     * no wasted K-row prefill head GEMM). Decode-bank logits are byte-identical to a
     * decode-only step either way. max_head_runs == 0 (all runs) preserves inc-3/4
     * behavior exactly. */
    uint32_t head_runs = (max_head_runs == 0u || max_head_runs > n_runs)
                       ? n_runs : max_head_runs;
    bool head_single_block = true;
    for (uint32_t r = 0; r < head_runs; r++)
        if ((uint32_t)last_idx[r] != r) { head_single_block = false; break; }

    bool ok = pulsar_gpu_begin_commands() != 0;
    for (uint32_t il = 0; ok && il < PULSAR_N_LAYER; il++) {
        ok = gpu_graph_encode_layer_batch(g, model, &weights->layer[il], il,
                                          (uint32_t)pos[0], n_active);
    }
    if (head_single_block) {
        /* Hot path: the emitted runs are the leading identity rows [0,head_runs)
         * (all length-1) — head in the SAME block, no gather, no extra synchronize.
         * Decode-only (head_runs == n_active) is byte-identical to before; an
         * intermediate fused step (head_runs == n_dec < n_active) heads only the
         * decode banks and skips the whole prefill-head two-block. */
        if (ok) ok = gpu_graph_encode_output_head_batch(g, model, weights,
                                                        head_runs, weights->output->dim[1]);
        if (ok) ok = pulsar_gpu_end_commands() != 0; else (void)pulsar_gpu_synchronize();
    } else {
        /* Prefill/mixed final: close the layer block so batch_cur_hc is final,
         * GATHER each emitted run's LAST row to the front of batch_cur_hc, then run
         * the head on the compact head_runs rows in a second block. last_idx is
         * ascending with last_idx[r] >= r, so front-compaction never clobbers an
         * un-copied source. The extra synchronize is amortized over the prefill. */
        if (ok) ok = pulsar_gpu_end_commands() != 0; else (void)pulsar_gpu_synchronize();
        const uint64_t hc_row_bytes = (uint64_t)PULSAR_N_HC * PULSAR_N_EMBD * PULSAR_HC_ELT_SIZE;   /* carrier */
        for (uint32_t r = 0; ok && r < head_runs; r++) {
            if ((uint32_t)last_idx[r] != r)
                ok = pulsar_gpu_tensor_copy(g->batch_cur_hc, (uint64_t)r * hc_row_bytes,
                                         g->batch_cur_hc, (uint64_t)last_idx[r] * hc_row_bytes,
                                         hc_row_bytes) != 0;
        }
        if (ok) ok = pulsar_gpu_begin_commands() != 0;
        if (ok) ok = gpu_graph_encode_output_head_batch(g, model, weights,
                                                        head_runs, weights->output->dim[1]);
        if (ok) ok = pulsar_gpu_end_commands() != 0; else (void)pulsar_gpu_synchronize();
    }
    /* Disarm + per-bank frontier self-check even when the sweep failed. The
     * step_end check covers EVERY bank (incl. a prefill run whose head we skipped),
     * so a skipped-head run's KV frontier is still validated. */
    const bool end_ok = gpu_graph_multiseq_step_end(g);
    if (!ok || !end_ok) return -1;   /* armed sweep failed: session-fatal */

    /* Logits readback: head_runs rows (one per emitted run, in run order = ascending
     * first-appearance). Decode-only => head_runs == n_runs == n_active, row k ==
     * bank[k]. */
    if (out_n_rows) *out_n_rows = head_runs;
    ok = pulsar_gpu_tensor_read(g->spec_logits, 0, logits,
                             (uint64_t)head_runs * PULSAR_N_VOCAB * sizeof(float)) != 0;
    /* The banks' KV/frontiers committed correctly (step_end passed); a
     * readback failure still leaves the caller without this step's logits,
     * which desynchronizes its sampling from the committed KV — treat as
     * fatal rather than guess. */
    return ok ? 1 : -1;
}



bool gpu_graph_read_spec_logits_row(pulsar_gpu_graph *g, uint32_t row, float *logits) {
    if (!g || !g->spec_logits || !logits || row >= g->prefill_cap) return false;
    const uint64_t row_bytes = (uint64_t)PULSAR_N_VOCAB * sizeof(float);
    return pulsar_gpu_tensor_read(g->spec_logits,
                                 (uint64_t)row * row_bytes,
                                 logits,
                                 row_bytes) != 0;
}



/* Pick a raw SWA cache size for GPU.  During batched prefill it must cover
 * the previous window plus the current ubatch. */
uint32_t gpu_graph_raw_cap_for_context(int ctx_size, uint32_t prefill_cap) {
    uint32_t raw_window = PULSAR_N_SWA;
    if (raw_window > (uint32_t)ctx_size) raw_window = (uint32_t)ctx_size;
    if (raw_window == 0) raw_window = 1;

    /*
     * During batched prefill the SWA cache must hold the current ubatch plus
     * the previous logical window. The cache is padded to a 256-row multiple
     * so the physical row order and FlashAttention block grouping match the
     * model path we compare against.
     */
    uint64_t wanted = (uint64_t)raw_window + prefill_cap;
    if (wanted > (uint32_t)ctx_size) wanted = (uint32_t)ctx_size;
    if (wanted == 0) wanted = 1;
    wanted = align_up(wanted, 256u);
    if (wanted > 8192u) wanted = 8192u;
    uint32_t raw_cap = (uint32_t)wanted;
    if (raw_cap < raw_window) raw_cap = raw_window;

    const char *env = getenv("PULSAR_CUDA_GRAPH_RAW_CAP");
    if (env && env[0]) {
        char *endp = NULL;
        const long v = strtol(env, &endp, 10);
        if (endp != env && v > 0) {
            raw_cap = (uint32_t)v;
            if (raw_cap > (uint32_t)ctx_size) raw_cap = (uint32_t)ctx_size;
            if (raw_cap > 8192u) raw_cap = 8192u;
            if (raw_cap < raw_window) raw_cap = raw_window;
        }
    }

    return raw_cap;
}



/* Choose the prefill ubatch size. Whole-batch is fastest for normal prompts.
 * Long Flash prompts default to 4096-token chunks; PRO defaults to 8192. */
uint32_t gpu_graph_prefill_cap_for_prompt(int prompt_len,
                                                   uint32_t prefill_chunk) {
    return pulsar_prefill_cap_for_prompt(prompt_len, prefill_chunk);
}



/* When a server request shares a large prefix with the live checkpoint, extend
 * the KV cache with batched prefill instead of single-token decode.  On an M3
 * Max, prefill is faster from 2-token suffixes upward; keep the default at 4
 * as a conservative crossover.  The env knob remains useful for retuning. */
uint32_t gpu_graph_resume_prefill_min_tokens(void) {
    const char *env = getenv("PULSAR_CUDA_RESUME_PREFILL_MIN");
    if (env && env[0]) {
        char *endp = NULL;
        const long v = strtol(env, &endp, 10);
        if (endp != env) {
            if (v <= 0) return UINT32_MAX;
            return (uint32_t)v;
        }
    }
    return 4u;
}



pulsar_context_memory pulsar_context_memory_estimate_with_prefill(
        pulsar_backend backend,
        int         ctx_size,
        uint32_t    prefill_chunk) {
    pulsar_context_memory m = {0};
    uint32_t ctx = ctx_size > 0 ? (uint32_t)ctx_size : 1u;

    if (pulsar_backend_uses_graph(backend)) {
        m.prefill_cap = gpu_graph_prefill_cap_for_prompt((int)ctx,
                                                           prefill_chunk);
        m.raw_cap = gpu_graph_raw_cap_for_context((int)ctx, m.prefill_cap);

        uint32_t min_ratio = UINT32_MAX;
        for (uint32_t il = 0; il < PULSAR_N_LAYER; il++) {
            const uint32_t ratio = pulsar_layer_compress_ratio(il);
            if (ratio != 0 && ratio < min_ratio) min_ratio = ratio;
        }
        if (min_ratio == UINT32_MAX) min_ratio = ctx;
        m.comp_cap = ctx / min_ratio + 2u;
        if (m.comp_cap < 2u) m.comp_cap = 2u;

        m.raw_bytes = (uint64_t)PULSAR_N_LAYER *
                      m.raw_cap *
                      PULSAR_N_HEAD_DIM *
                      sizeof(float);
        for (uint32_t il = 0; il < PULSAR_N_LAYER; il++) {
            const uint32_t ratio = pulsar_layer_compress_ratio(il);
            if (ratio == 0) continue;
            const uint32_t layer_comp_cap = ctx / ratio + 2u;
            m.compressed_bytes += (uint64_t)layer_comp_cap *
                                  PULSAR_N_HEAD_DIM *
                                  sizeof(float);
            if (ratio == 4) {
                m.compressed_bytes += (uint64_t)layer_comp_cap *
                                      PULSAR_N_INDEXER_HEAD_DIM *
                                      sizeof(float);
            }
        }
        uint64_t attn_stage_cap = (uint64_t)(m.prefill_cap / min_ratio + 2u);
        if (attn_stage_cap < 2u) attn_stage_cap = 2u;
        /* indexer_scores/comp_mask token rows shrink to the slice size under
         * PULSAR_PREFILL_SLICE (see gpu_graph_prefill_slice / gpu_diag alloc). */
        uint64_t score_rows = (uint64_t)m.prefill_cap;
        if (gpu_graph_prefill_slice() != 0u &&
            (uint64_t)gpu_graph_prefill_slice() < score_rows) {
            score_rows = (uint64_t)gpu_graph_prefill_slice();
        }
        m.scratch_bytes = 2ull *
                          m.comp_cap *
                          score_rows *
                          sizeof(float) +
                          attn_stage_cap * PULSAR_N_HEAD_DIM * sizeof(float);
    } else {
        m.raw_cap = pulsar_default_raw_cap(ctx);
        m.raw_bytes = (uint64_t)PULSAR_N_LAYER *
                      m.raw_cap *
                      PULSAR_N_HEAD_DIM *
                      sizeof(float);
        for (uint32_t il = 0; il < PULSAR_N_LAYER; il++) {
            const uint32_t ratio = pulsar_layer_compress_ratio(il);
            if (ratio == 0) continue;
            const uint32_t comp_cap = ctx / ratio + 2u;
            if (ratio == 4) m.comp_cap = comp_cap;
            m.compressed_bytes += (uint64_t)comp_cap *
                                  PULSAR_N_HEAD_DIM *
                                  sizeof(float);
            if (ratio == 4) {
                m.compressed_bytes += (uint64_t)comp_cap *
                                      PULSAR_N_INDEXER_HEAD_DIM *
                                      sizeof(float);
            }
        }
        if (m.comp_cap == 0) m.comp_cap = ctx / 4u + 2u;
        m.scratch_bytes = ((uint64_t)(m.raw_cap + m.comp_cap) * sizeof(float)) +
                          ((uint64_t)m.comp_cap * sizeof(float)) +
                          ((uint64_t)m.comp_cap * sizeof(bool));
    }

    m.total_bytes = m.raw_bytes + m.compressed_bytes + m.scratch_bytes;
    return m;
}



pulsar_context_memory pulsar_context_memory_estimate(pulsar_backend backend,
                                               int         ctx_size) {
    return pulsar_context_memory_estimate_with_prefill(backend, ctx_size, 0);
}



/* Packed variant of the estimate: recompute the three persistent KV caches
 * (raw ring, attn comp, indexer comp) with the real element/row widths the
 * graph actually allocates (see gpu_graph_alloc_raw_cap in gpu_diag.c), rather
 * than the sizeof(float) pessimistic upper bound the base estimate uses.  The
 * f32 scratch working set (batch_* buffers etc.) is not packed and carries
 * through from the base estimate unchanged. */
pulsar_context_memory pulsar_context_memory_estimate_packed(
        pulsar_backend backend,
        int         ctx_size,
        uint32_t    prefill_chunk) {
    pulsar_context_memory m =
        pulsar_context_memory_estimate_with_prefill(backend, ctx_size, prefill_chunk);
    if (!pulsar_backend_uses_graph(backend)) return m;

    const uint32_t ctx = ctx_size > 0 ? (uint32_t)ctx_size : 1u;

    /* Raw SWA ring: f16 rows under PULSAR_RAW_F16 (default on), else f32. */
    const uint64_t raw_elem_bytes = gpu_graph_raw_f16_enabled() ? sizeof(uint16_t)
                                                                 : sizeof(float);
    m.raw_bytes = (uint64_t)PULSAR_N_LAYER * m.raw_cap * PULSAR_N_HEAD_DIM * raw_elem_bytes;

    /* Compressed caches: PULSAR_ATTN_PACK attn comp row + MXFP4 indexer row. */
    const uint64_t attn_row_bytes  = gpu_graph_attn_comp_cache_row_bytes();
    const uint64_t index_row_bytes = gpu_graph_idx_fp4_enabled()
        ? PULSAR_ENGINE_IDXFP4_ROWBYTES
        : (uint64_t)PULSAR_N_INDEXER_HEAD_DIM * sizeof(float);
    m.compressed_bytes = 0;
    for (uint32_t il = 0; il < PULSAR_N_LAYER; il++) {
        const uint32_t ratio = pulsar_layer_compress_ratio(il);
        if (ratio == 0) continue;
        const uint32_t layer_comp_cap = ctx / ratio + 2u;
        m.compressed_bytes += (uint64_t)layer_comp_cap * attn_row_bytes;
        if (ratio == 4) {
            m.compressed_bytes += (uint64_t)layer_comp_cap * index_row_bytes;
        }
    }

    m.total_bytes = m.raw_bytes + m.compressed_bytes + m.scratch_bytes;
    return m;
}






void embed_prompt(
        const pulsar_model   * model,
        const pulsar_weights * weights,
        const token_vec   * tokens,
        uint32_t            n_embd,
        float             * out) {
    for (int i = 0; i < tokens->len; i++) {
        embed_token_f16(model, weights, tokens->v[i], out + (uint64_t)i * n_embd);
    }
}

