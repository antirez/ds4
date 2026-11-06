/* Native V4.1 DSpark graph oracle. No target experts are made resident. */
#include "../ds4.c"

#define REQUIRE(x) do { if (!(x)) { fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); goto done; } } while (0)

static bool dump_tensor(const char *dir, const char *name, ds4_gpu_tensor *x, size_t floats) {
    char path[4096];
    if (snprintf(path, sizeof(path), "%s/%s.f32", dir, name) >= (int)sizeof(path)) return false;
    float *data = malloc(floats * sizeof(float));
    if (!data) return false;
    bool ok = ds4_gpu_tensor_read(x, 0, data, floats * sizeof(float));
    for (size_t i = 0; ok && i < floats; i++) ok = isfinite(data[i]);
    FILE *fp = ok ? fopen(path, "wb") : NULL;
    ok = fp && fwrite(data, sizeof(float), floats, fp) == floats;
    if (fp && fclose(fp)) ok = false;
    free(data);
    return ok;
}

static bool cancel_now(void *unused) {
    (void)unused;
    return true;
}

static int check_candidate_mask(void) {
    int rc = 1;
    ds41_gpu_graph g = {0};
    ds4_gpu_tensor *reference = NULL;
    const uint32_t widths[] = {1, 7, 8, 9, 127, 1025, 8191, 8192, 12307, 16383, 16384, 16385, 17017};
    const uint32_t capacity = 17017, max_blocks = (capacity + 7u) / 8u;
    float *scores = malloc(capacity * sizeof(float));
    float *expected = malloc(max_blocks * sizeof(float));
    float *actual = malloc(max_blocks * sizeof(float));
    g_ds4_shape = DS4_SHAPE_FLASH41;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++)
        g_ds4_compress_ratios[il] = ds4_expected_layer_compress_ratio(il);
    REQUIRE(scores && expected && actual && ds4_gpu_init());
    g.index_scores = ds4_gpu_tensor_alloc(capacity * sizeof(float));
    g.block_scores = ds4_gpu_tensor_alloc(max_blocks * sizeof(float));
    g.block_selected = ds4_gpu_tensor_alloc(2048u * sizeof(uint32_t));
    g.block_mask = ds4_gpu_tensor_alloc(max_blocks * sizeof(float));
    reference = ds4_gpu_tensor_alloc(max_blocks * sizeof(float));
    REQUIRE(g.index_scores && g.block_scores && g.block_selected && g.block_mask && reference);
    for (uint32_t i = 0; i < capacity; i++) scores[i] = (float)((i * 7919u) % 104729u) - 50000;
    REQUIRE(ds4_gpu_tensor_write(g.index_scores, 0, scores, capacity * sizeof(float)));
    for (unsigned wi = 0; wi < sizeof(widths) / sizeof(*widths); wi++) {
        const uint32_t n = widths[wi], blocks = (n + 7u) / 8u;
        const uint32_t top = blocks < 2048u ? blocks : 2048u;
        g.pos = n - 1u;
        REQUIRE(ds4_gpu_begin_commands());
        REQUIRE(ds4_gpu_dsv41_candidate_blocks(g.block_scores, g.index_scores, n, 1, g.pos, 1));
        REQUIRE(ds4_gpu_indexer_topk_tensor(g.block_selected, g.block_scores, blocks, 1, top));
        REQUIRE(ds4_gpu_dsv4_topk_mask_tensor(reference, g.block_selected, blocks, 1, top));
        REQUIRE(ds4_gpu_tensor_fill_f32(g.block_mask, -123, max_blocks));
        REQUIRE(ds41_attention_candidates(&g, 20));
        REQUIRE(ds4_gpu_end_commands());
        REQUIRE(ds4_gpu_tensor_read(reference, 0, expected, blocks * sizeof(float)));
        REQUIRE(ds4_gpu_tensor_read(g.block_mask, 0, actual, max_blocks * sizeof(float)));
        REQUIRE(!memcmp(expected, actual, blocks * sizeof(float)));
        for (uint32_t i = blocks; i < max_blocks; i++) REQUIRE(actual[i] == -123);
        fprintf(stderr, "V4.1 candidate mask width=%u: exact\n", n);
    }
    rc = 0;
done:
    if (ds4_gpu_commands_active()) (void)ds4_gpu_end_commands();
    ds4_gpu_tensor_free(reference);
    ds4_gpu_tensor_free(g.block_mask);
    ds4_gpu_tensor_free(g.block_selected);
    ds4_gpu_tensor_free(g.block_scores);
    ds4_gpu_tensor_free(g.index_scores);
    ds4_gpu_cleanup();
    free(actual); free(expected); free(scores);
    return rc;
}

static int check_runtime(const char *target, const char *draft, const ds4_tp_options *tp_opt) {
    int rc = 1;
    ds4_engine *engine = NULL;
    ds4_tp *tp = NULL;
    ds4_session *a = NULL, *b = NULL;
    ds4_tokens prompt = {0};
    FILE *snapshot = NULL;
    char err[256] = {0};
    ds4_engine_options opt = {.model_path = target, .mtp_path = draft,
#ifdef __APPLE__
        .backend = DS4_BACKEND_METAL,
#else
        .backend = DS4_BACKEND_CUDA,
#endif
        .context_size = 2048, .power_percent = 100, .ssd_streaming = true,
        .ssd_streaming_cache_bytes = UINT64_C(40) << 30, .dspark = true};
    if (tp_opt) { opt.tp = *tp_opt; opt.ssd_streaming = false; }
    REQUIRE(!ds4_engine_open(&engine, &opt));
    if (tp_opt) {
        ds4_tp_identity id = {.gguf_bytes = ds4_engine_model_bytes(engine),
            .model_id = ds4_engine_model_id(engine), .n_layer = DS4_N_LAYER,
            .n_embd = DS4_N_EMBD, .n_vocab = DS4_N_VOCAB,
            .quant_bits = ds4_engine_routed_quant_bits(engine), .ctx_size = 2048};
        ds4_engine_tp_gate_schedule(engine, &id.gate_slot_start, &id.gate_slot_step,
            &id.gates_per_token, id.gate_slot_mask);
        REQUIRE(ds4_tp_create(&tp, &opt.tp, &id, err, sizeof(err)));
        REQUIRE(ds4_engine_tp_bind(engine, tp, err, sizeof(err)));
    }
    REQUIRE(!ds4_session_create(&a, engine, 2048));
    REQUIRE(!ds4_session_create(&b, engine, 2048));
    for (unsigned i = 0; i < 127; i++) ds4_tokens_push(&prompt, 300 + i % 23);
    REQUIRE(!ds4_session_sync(a, &prompt, err, sizeof(err)));
    snapshot = tmpfile();
    REQUIRE(snapshot && !ds4_session_save_payload(a, snapshot, err, sizeof(err)));
    const uint64_t bytes = ds4_session_payload_bytes(a);
    for (unsigned keep = 0; keep <= 6; keep++) {
        if (tp) {
            ds4_session_invalidate(a); ds4_session_invalidate(b);
            REQUIRE(!ds4_session_sync(a, &prompt, err, sizeof(err)));
            REQUIRE(!ds4_session_sync(b, &prompt, err, sizeof(err)));
        } else {
            rewind(snapshot);
            REQUIRE(!ds4_session_load_payload(a, snapshot, bytes, err, sizeof(err)));
            rewind(snapshot);
            REQUIRE(!ds4_session_load_payload(b, snapshot, bytes, err, sizeof(err)));
        }
        int ta[6], tb[6];
        for (unsigned i = 0; i < 6; i++) {
            ta[i] = 700 + i;
            tb[i] = i < keep ? ta[i] : 900 + (int)i;
        }
        ds41_dspark *da = a->ds41_graph.dspark, *db = b->ds41_graph.dspark;
        REQUIRE(ds41_session_verify(a, ta, 6));
        REQUIRE(ds41_session_commit(a, ta, 6, 6));
        REQUIRE(ds41_session_verify(b, tb, 6));
        REQUIRE(ds41_session_commit(b, tb, 6, keep));
        double worst = 0;
        for (uint64_t i = 0; i < (uint64_t)keep * DS4_N_VOCAB; i++)
            worst = fmax(worst, fabs(da->host_verified[i] - db->host_verified[i]));
        fprintf(stderr, "verify prefix=%u causal max=%.8g\n", keep, worst);
        REQUIRE(worst < 0.0001);
        if (keep < 6) REQUIRE(ds4_session_rewind_speculative(a, 127 + keep));
        REQUIRE(a->checkpoint.len == b->checkpoint.len && a->checkpoint_valid);
        REQUIRE(!memcmp(a->logits, b->logits, DS4_N_VOCAB * sizeof(float)));
        for (unsigned i = 0; i < 8; i++) {
            REQUIRE(!ds4_session_eval(a, 1200 + i, err, sizeof(err)));
            REQUIRE(!ds4_session_eval(b, 1200 + i, err, sizeof(err)));
            worst = 0;
            for (uint32_t v = 0; v < DS4_N_VOCAB; v++)
                worst = fmax(worst, fabs(a->logits[v] - b->logits[v]));
            REQUIRE(worst < 0.0001);
        }
        REQUIRE(!ds4_session_rewind_speculative(a, a->checkpoint.len - 1));
        fprintf(stderr, "verify prefix=%u rollback and 8 continuation steps PASS\n", keep);
    }
    puts("V4.1 DSpark causal verification, all prefix commits/rewinds and snapshot load: PASS");
    for (unsigned exact = 0; exact < 2; exact++) {
        ds4_session_invalidate(a);
        REQUIRE(!ds4_session_sync(a, &prompt, err, sizeof(err)));
        engine->dspark_exact_sampling = exact != 0;
        engine->dspark_confidence_threshold = 0;
        uint64_t rng = 12345;
        for (unsigned cycle = 0; cycle < 8; cycle++) {
            const int seed = sample_argmax(a->logits, DS4_N_VOCAB);
            const int before = a->checkpoint.len;
            int accepted[6];
            const int count = ds4_session_eval_speculative(a, seed, 6, -1,
                0.7f, 0, 0.95f, 0, &rng, accepted, 6, err, sizeof(err));
            REQUIRE(count >= 1 && count <= 6 && accepted[0] == seed);
            REQUIRE(a->checkpoint.len == before + count &&
                    a->ds41_graph.pos == (uint32_t)a->checkpoint.len);
            for (int i = 0; i < count; i++)
                REQUIRE(a->checkpoint.v[before + i] == accepted[i]);
            if (count > 1) {
                REQUIRE(ds4_session_rewind_speculative(a, before + count - 1));
                REQUIRE(!ds4_session_eval(a, accepted[count - 1], err, sizeof(err)));
                REQUIRE(a->checkpoint.len == before + count);
            }
        }
        if (!tp) {
            rewind(snapshot);
            REQUIRE(!ds4_session_save_payload(a, snapshot, err, sizeof(err)));
            const uint64_t saved_bytes = ds4_session_payload_bytes(a);
            rewind(snapshot);
            REQUIRE(!ds4_session_load_payload(b, snapshot, saved_bytes, err, sizeof(err)));
            REQUIRE(!b->ds41_graph.dspark->kv_valid);
            for (unsigned i = 0; i < 3; i++) {
                REQUIRE(!ds4_session_eval(a, 1500 + i, err, sizeof(err)));
                REQUIRE(!ds4_session_eval(b, 1500 + i, err, sizeof(err)));
                REQUIRE(!memcmp(a->logits, b->logits, DS4_N_VOCAB * sizeof(float)));
            }
        }
        fprintf(stderr, "V4.1 DSpark exact=%u sampling and frontier%s: PASS\n", exact,
            tp ? " (physical TP)" : ", post-verify snapshot");
    }
    {
        int accepted[6];
        uint64_t rng = 71;
        const int pos = a->checkpoint.len;
        ds4_session_set_cancel(a, cancel_now, NULL);
        REQUIRE(ds4_session_eval_speculative(a, sample_argmax(a->logits, DS4_N_VOCAB),
            6, -1, 0.7f, 0, 0.95f, 0, &rng, accepted, 6, err, sizeof(err)) < 0);
        REQUIRE(a->checkpoint_valid && a->checkpoint.len == pos);
        ds4_session_set_cancel(a, NULL, NULL);
        err[0] = '\0';
        REQUIRE(!ds4_session_eval(a, 1600, err, sizeof(err)));
        while (prompt.len < 2043) ds4_tokens_push(&prompt, 300 + prompt.len % 23);
        ds4_session_invalidate(a);
        REQUIRE(!ds4_session_sync(a, &prompt, err, sizeof(err)));
        while (a->checkpoint.len < 2048) {
            const int before = a->checkpoint.len;
            const int n = ds4_session_eval_speculative(a, sample_argmax(a->logits, DS4_N_VOCAB),
                6, -1, 0.7f, 0, 0.95f, 0, &rng, accepted, 6, err, sizeof(err));
            REQUIRE(n >= 1 && n <= 2048 - before && a->checkpoint.len == before + n);
        }
        REQUIRE(ds4_session_eval_speculative(a, 1600, 6, -1, 0.7f, 0, 0.95f, 0,
            &rng, accepted, 6, err, sizeof(err)) == 0);
        fprintf(stderr, "V4.1 DSpark cancellation and context-limit clamping: PASS\n");
    }
    rc = 0;
done:
    if (err[0]) fprintf(stderr, "%s\n", err);
    if (snapshot) fclose(snapshot);
    ds4_tokens_free(&prompt);
    ds4_session_free(b); ds4_session_free(a);
    if (tp) (void)ds4_tp_send_stop(tp);
    ds4_engine_close(engine); ds4_tp_free(tp);
    return rc;
}

int main(int argc, char **argv) {
    if (argc == 2 && !strcmp(argv[1], "--candidate-mask")) return check_candidate_mask();
    if (argc == 4 && !strcmp(argv[3], "--runtime")) return check_runtime(argv[1], argv[2], NULL);
    if (argc == 8 && !strcmp(argv[3], "--runtime-tp")) {
        char *end;
        long port = strtol(argv[5], &end, 10);
        if (*end || port < 1 || port > 65535) return 2;
        long gid = strtol(argv[7], &end, 10);
        if (*end || gid < 0 || gid > 255) return 2;
        ds4_tp_options opt = {.role = DS4_TP_LEADER, .requested = true,
            .listen_host = argv[4], .listen_port = (int)port,
            .transport = DS4_TP_TRANSPORT_RDMA, .rdma_device = argv[6],
            .rdma_gid_index = (int)gid, .rdma_gid_index_set = true};
        return check_runtime(argv[1], argv[2], &opt);
    }
    if (argc != 4) {
        fprintf(stderr, "usage: %s TARGET_GGUF DSPARK_GGUF OUTPUT_DIRECTORY\n", argv[0]);
        return 2;
    }
    int rc = 1;
    ds4_model target = {.fd = -1}, draft = {.fd = -1};
    ds4_weights tw = {0};
    ds4_dspark_weights dw = {0};
    ds41_dspark *d = NULL;
    ds4_gpu_tensor *hc = NULL;
    float *source = NULL;
    model_open(&target, argv[1], true, false);
    config_validate_model(&target);
    REQUIRE(DS4_MODEL_FAMILY == DS4_MODEL_FAMILY_DEEPSEEK41);
    weights_bind(&tw, &target, false, 0, UINT32_MAX, true, false);
    model_open(&draft, argv[2], true, false);
    REQUIRE(support_model_checkpoint_compatible(&draft));
    ds4_dspark_summary summary = model_dspark_summary(&draft);
    dspark_weights_bind_optional(&dw, &draft, &summary);
    REQUIRE(dw.v41 && !dw.invalid_tensors && !dw.missing_tensors && !dw.metadata_errors);
    REQUIRE(ds4_gpu_init());
    const uint64_t offsets[] = {tw.token_embd->abs_offset, tw.output->abs_offset};
    const uint64_t sizes[] = {tw.token_embd->bytes, tw.output->bytes};
    REQUIRE(ds4_gpu_set_model_fd_for_map(target.fd, target.map));
    REQUIRE(ds4_gpu_set_model_map_spans(target.map, target.size, offsets, sizes, 2, tw.token_embd->bytes));
    REQUIRE(ds4_gpu_set_model_fd_for_map(draft.fd, draft.map));
    REQUIRE(ds4_gpu_set_model_map_range(draft.map, draft.size, draft.tensor_data_pos,
        draft.size - draft.tensor_data_pos, draft.max_tensor_bytes));
    REQUIRE(ds4_gpu_set_model_fd_for_map(target.fd, target.map));
    d = ds41_dspark_alloc(&dw);
    REQUIRE(d);
    enum { ROWS = 137 };
    const size_t elements = (size_t)ROWS * DS4_N_HC * DS4_N_EMBD;
    source = malloc(elements * sizeof(float));
    hc = ds4_gpu_tensor_alloc(elements * sizeof(float));
    REQUIRE(source && hc);
    for (unsigned stage = 0; stage < 3; stage++) {
        for (size_t i = 0; i < elements; i++)
            source[i] = ((int)((i * 17u + stage * 29u) % 257u) - 128) / 64.0f;
        REQUIRE(ds4_gpu_tensor_write(hc, 0, source, elements * sizeof(float)));
        REQUIRE(ds4_gpu_begin_commands());
        REQUIRE(ds41_dspark_capture(d, 37 + stage, 0, hc, ROWS));
        REQUIRE(ds4_gpu_end_commands());
    }
    REQUIRE(ds41_dspark_forward(d, &target, &tw, &draft, ROWS, 100, NULL));
    REQUIRE(d->kv_valid && d->kv_end == ROWS);
    REQUIRE(dump_tensor(argv[3], "main_x", d->main_x, DS41_DRAFT_WINDOW * DS4_N_EMBD));
    for (unsigned stage = 0; stage < 3; stage++) {
        char name[32];
        snprintf(name, sizeof(name), "window%u", stage);
        REQUIRE(dump_tensor(argv[3], name, d->window[stage], (DS41_DRAFT_WINDOW + DS41_DRAFT_ROWS) * DS4_N_HEAD_DIM));
    }
    REQUIRE(dump_tensor(argv[3], "hidden", d->batch.x, DS41_DRAFT_ROWS * DS4_N_EMBD));
    REQUIRE(dump_tensor(argv[3], "norm", d->batch.norm, DS41_DRAFT_ROWS * DS4_N_EMBD));
    REQUIRE(dump_tensor(argv[3], "logits", d->logits, DS41_DRAFT_ROWS * DS4_N_VOCAB));
    REQUIRE(ds4_gpu_tensor_read(d->logits, 0, d->host_logits, DS41_DRAFT_ROWS * DS4_N_VOCAB * sizeof(float)));
    REQUIRE(ds41_dspark_forward(d, &target, &tw, &draft, ROWS, 100, NULL));
    float *second = malloc(DS41_DRAFT_ROWS * DS4_N_VOCAB * sizeof(float));
    REQUIRE(second);
    bool repeated = ds4_gpu_tensor_read(d->logits, 0, second, DS41_DRAFT_ROWS * DS4_N_VOCAB * sizeof(float)) &&
        !memcmp(second, d->host_logits, DS41_DRAFT_ROWS * DS4_N_VOCAB * sizeof(float));
    free(second);
    REQUIRE(repeated);
    int previous = 100;
    for (unsigned row = 0; row < DS41_DRAFT_ROWS; row++) {
        float confidence;
        REQUIRE(ds41_dspark_markov(d, &draft, row, previous, &confidence));
        char name[32];
        snprintf(name, sizeof(name), "markov%u", row);
        REQUIRE(dump_tensor(argv[3], name, d->markov_logits, DS4_N_VOCAB));
        snprintf(name, sizeof(name), "confidence%u", row);
        REQUIRE(dump_tensor(argv[3], name, d->confidence, 1));
        previous = (int)dspark_argmax_f32(d->host_logits + (uint64_t)row * DS4_N_VOCAB, DS4_N_VOCAB);
        fprintf(stderr, "draft[%u]=%d confidence=%.7g\n", row, previous, confidence);
    }
    REQUIRE(dump_tensor(argv[3], "biased_logits", d->logits, DS41_DRAFT_ROWS * DS4_N_VOCAB));
    REQUIRE(ds4_gpu_begin_commands());
    const float pre[5][4] = {{1}, {1}, {1}, {1}, {1}};
    REQUIRE(ds4_gpu_tensor_write(d->batch.pre, 0, pre, sizeof(pre)));
    for (uint32_t row = 0; row < DS41_DRAFT_ROWS; row++) {
        const uint64_t bytes = (uint64_t)DS4_N_EMBD * DS4_N_HC * sizeof(float);
        ds4_gpu_tensor *out = ds4_gpu_tensor_view(d->batch.residual, row * bytes, bytes);
        bool ok = out && ds4_gpu_embed_token_hc_tensor(out, target.map, target.size,
            tw.token_embd->abs_offset, DS4_N_VOCAB, row ? dw.noise_token_id : 100,
            DS4_N_EMBD, DS4_N_HC);
        ds4_gpu_tensor_free(out);
        REQUIRE(ok);
    }
    REQUIRE(ds4_gpu_end_commands());
    REQUIRE(dump_tensor(argv[3], "embedded", d->batch.residual, 5u * 4u * DS4_N_EMBD));
    for (unsigned stage = 0; stage < 3; stage++) {
        REQUIRE(ds4_gpu_begin_commands());
        REQUIRE(ds41_dspark_stage(d, &draft, stage, ROWS));
        REQUIRE(ds4_gpu_end_commands());
#define STAGE_DUMP(field, count) do { \
    char name[64]; snprintf(name, sizeof(name), "stage%u_" #field, stage); \
    REQUIRE(dump_tensor(argv[3], name, d->batch.field, (count))); \
} while (0)
        STAGE_DUMP(q, 5u * DS4_N_HEAD * DS4_N_HEAD_DIM);
        STAGE_DUMP(kv, 5u * DS4_N_HEAD_DIM);
        STAGE_DUMP(heads, 5u * DS4_N_HEAD * DS4_N_HEAD_DIM);
        STAGE_DUMP(after_attn, 5u * DS4_N_HC * DS4_N_EMBD);
        STAGE_DUMP(norm, 5u * DS4_N_EMBD);
        STAGE_DUMP(route_logits, 5u * dw.n_experts);
        STAGE_DUMP(routed, 5u * DS4_N_EMBD);
        STAGE_DUMP(shared, 5u * DS4_N_EMBD);
        STAGE_DUMP(ffn_split, 5u * 24u);
        STAGE_DUMP(residual, 5u * DS4_N_HC * DS4_N_EMBD);
#undef STAGE_DUMP
    }
    puts("V4.1 draft capture, ring seed and forward: finite and repeatable");
    rc = 0;
done:
    if (ds4_gpu_commands_active()) (void)ds4_gpu_end_commands();
    ds4_gpu_tensor_free(hc);
    ds41_dspark_free(d);
    ds4_gpu_cleanup();
    model_close(&draft); model_close(&target);
    free(source);
    return rc;
}
