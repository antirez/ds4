/* Bounded SSD graph bring-up. Run only on a dedicated Metal test host. */
#include "../ds4.c"
#include <assert.h>
#include <sys/wait.h>

#define REQUIRE(x) do { if (!(x)) { \
    fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); goto done; \
} } while (0)

static int check_batch_admission(void) {
    int rc = 1;
    ds4_session *sessions = calloc(8, sizeof(*sessions));
    ds4_decode_item items[8] = {0};
    REQUIRE(sessions);
    for (int i = 0; i < 8; i++) {
        sessions[i].ds41_graph_ready = sessions[i].checkpoint_valid = true;
        sessions[i].ds41_graph.valid = true;
        sessions[i].ds41_graph.prefill_cap = 1;
        sessions[i].checkpoint.len = sessions[i].ds41_graph.pos = 1;
        items[i].session = &sessions[i];
    }
    REQUIRE(!ds41_sessions_batch_supported(NULL, 2));
    REQUIRE(!ds41_sessions_batch_supported(items, 1));
    REQUIRE(!ds41_sessions_batch_supported(items, 9));
    REQUIRE(!ds41_sessions_batch_supported(items, 2));
    sessions[1].ds41_graph.prefill_cap = 2;
    REQUIRE(ds41_sessions_batch_supported(items, 2));
    REQUIRE(!ds41_sessions_batch_supported(items, 8));
    sessions[7].ds41_graph.prefill_cap = 8;
    REQUIRE(ds41_sessions_batch_supported(items, 8));
    sessions[1].ds41_graph.tp_world = 2;
    REQUIRE(!ds41_sessions_batch_supported(items, 2));
    sessions[1].ds41_graph.prefill_cap = 3;
    REQUIRE(ds41_sessions_batch_supported(items, 3));
    sessions[1].ds41_graph.tp_world = 1;
    sessions[3].ds41_graph.streaming = true;
    REQUIRE(!ds41_sessions_batch_supported(items, 8));
    sessions[3].ds41_graph.streaming = false;
    sessions[3].ds41_graph.image_count = 1;
    REQUIRE(!ds41_sessions_batch_supported(items, 8));
    sessions[3].ds41_graph.image_count = 0;
    sessions[3].ds41_graph.quality = true;
    REQUIRE(!ds41_sessions_batch_supported(items, 8));
    sessions[3].ds41_graph.quality = false;
    sessions[3].ds41_graph.pos++;
    REQUIRE(!ds41_sessions_batch_supported(items, 8));
    sessions[3].ds41_graph.pos--;
    REQUIRE(setenv("DS4_METAL_DISABLE_V41_SESSION_BATCH", "1", 1) == 0);
    REQUIRE(!ds41_sessions_batch_supported(items, 8));
    REQUIRE(unsetenv("DS4_METAL_DISABLE_V41_SESSION_BATCH") == 0);
    REQUIRE(ds41_sessions_batch_supported(items, 8));
    ds41_gpu_graph *graphs[] = {&sessions[0].ds41_graph, &sessions[1].ds41_graph};
    graphs[0]->prefill_cap = graphs[1]->prefill_cap = 8192;
    graphs[0]->ctx = 65536;
    graphs[1]->ctx = 98304;
    REQUIRE(ds41_batch_workspace(graphs, 2) == graphs[1]);
    graphs[0]->ctx = 131072;
    REQUIRE(ds41_batch_workspace(graphs, 2) == graphs[0]);
    graphs[0]->prefill_cap = 1;
    REQUIRE(!ds41_batch_workspace(graphs, 2));
    puts("V4.1 native batch admission and bounded-workspace fallback: PASS");
    rc = 0;
done:
    unsetenv("DS4_METAL_DISABLE_V41_SESSION_BATCH");
    free(sessions);
    return rc;
}

static int check_batch_head(const char *path) {
    ds4_model model = {.fd = -1};
    ds4_weights weights = {0};
    ds4_gpu_tensor *input = NULL, *output = NULL, *scalar = NULL, *row = NULL;
    int rc = 1;
    model_open(&model, path, true, false);
    config_validate_model(&model);
    REQUIRE(DS4_MODEL_FAMILY == DS4_MODEL_FAMILY_DEEPSEEK41);
    weights_bind(&weights, &model, false, 0, UINT32_MAX, true, false);
    const ds4_tensor *head = weights.output;
    REQUIRE(head->type == DS4_TENSOR_Q8_0);
    ds4_gpu_model_residency_skip(1);
    REQUIRE(ds4_gpu_init());
    REQUIRE(ds4_gpu_set_model_map_range(model.map, model.size, head->abs_offset,
                                        head->bytes, head->bytes));
    input = ds4_gpu_tensor_alloc(8u * DS4_N_EMBD * sizeof(float));
    output = ds4_gpu_tensor_alloc(8u * DS4_N_VOCAB * sizeof(float));
    scalar = ds4_gpu_tensor_alloc(DS4_N_VOCAB * sizeof(float));
    REQUIRE(input && output && scalar);
    float *x = ds4_gpu_tensor_contents(input);
    float *batch = ds4_gpu_tensor_contents(output);
    float *single = ds4_gpu_tensor_contents(scalar);
    REQUIRE(x && batch && single);
    const uint32_t counts[] = {2, 3, 4, 8};
    for (unsigned fixture = 0; fixture < 3; fixture++) {
        for (uint32_t i = 0; i < 8u * DS4_N_EMBD; i++)
            x[i] = fixture == 0 ? 0 :
                (float)((int)((i * (fixture == 1 ? 37u : 97u) + 13u) % 257u) - 128) /
                (fixture == 1 ? 64.0f : 4096.0f);
        for (unsigned c = 0; c < sizeof(counts) / sizeof(*counts); c++) {
            const uint32_t count = counts[c];
            REQUIRE(ds41_matmul_batch(output, &model, head, input, count, false));
            double worst = 0, norm = 0, squared = 0;
            for (uint32_t r = 0; r < count; r++) {
                row = ds4_gpu_tensor_view(input, (uint64_t)r * DS4_N_EMBD * sizeof(float),
                                           DS4_N_EMBD * sizeof(float));
                REQUIRE(row && ds41_matmul(scalar, &model, head, row, false));
                ds4_gpu_tensor_free(row); row = NULL;
                for (uint32_t v = 0; v < DS4_N_VOCAB; v++) {
                    const float y = batch[(uint64_t)r * DS4_N_VOCAB + v];
                    REQUIRE(isfinite(y) && isfinite(single[v]));
                    const double d = (double)y - single[v];
                    worst = fmax(worst, fabs(d)); squared += d * d;
                    norm += (double)single[v] * single[v];
                }
                /* Independent double dot products on real quantized rows. */
                const uint32_t columns[] = {0, 517, DS4_N_VOCAB / 2, DS4_N_VOCAB - 1};
                for (unsigned j = 0; j < sizeof(columns) / sizeof(*columns); j++) {
                    const uint32_t v = columns[j];
                    const uint8_t *w = model.map + head->abs_offset +
                        (uint64_t)v * (DS4_N_EMBD / 32u) * 34u;
                    double sum = 0;
                    for (uint32_t k = 0; k < DS4_N_EMBD; k++) {
                        uint16_t scale;
                        memcpy(&scale, w + (k / 32u) * 34u, sizeof(scale));
                        const int8_t q = (int8_t)w[(k / 32u) * 34u + 2u + k % 32u];
                        sum += (double)f16_to_f32(scale) * q * x[r * DS4_N_EMBD + k];
                    }
                    REQUIRE(fabs(batch[(uint64_t)r * DS4_N_VOCAB + v] - sum) < 0.0002);
                }
            }
            const double relative = norm ? sqrt(squared / norm) : sqrt(squared);
            fprintf(stderr, "head fixture=%u rows=%u max=%.7g relative=%.7g\n",
                    fixture, count, worst, relative);
            REQUIRE(worst < 0.0002 && relative < 0.000002);
        }
    }
    puts("V4.1 batch head: scalar full logits and independent double reference PASS");
    rc = 0;
done:
    if (ds4_gpu_commands_active()) ds4_gpu_end_commands();
    ds4_gpu_tensor_free(row); ds4_gpu_tensor_free(scalar);
    ds4_gpu_tensor_free(output); ds4_gpu_tensor_free(input);
    ds4_gpu_cleanup(); model_close(&model);
    return rc;
}

static int check_attention_layouts(const char *path) {
    ds4_model model = {.fd = -1};
    ds4_weights weights = {0};
    model_open(&model, path, true, false);
    config_validate_model(&model);
    assert(DS4_MODEL_FAMILY == DS4_MODEL_FAMILY_DEEPSEEK41);
    weights_bind(&weights, &model, false, 0, UINT32_MAX, true, false);
    const uint32_t layers[] = {0, 39};
    const uint32_t types[] = {DS4_TENSOR_Q4_0, DS4_TENSOR_Q4_K};
    for (unsigned i = 0; i < 2; i++) {
        for (unsigned projection = 0; projection < 2; projection++) {
            for (unsigned type = 0; type < 2; type++) {
                const pid_t child = fork();
                assert(child >= 0);
                if (!child) {
                    ds4_layer_weights *l = &weights.layer[layers[i]];
                    ds4_tensor *w = projection ? l->attn_output_b : l->attn_output_a;
                    w->type = types[type];
                    weights_validate_layout(&weights, layers[i], layers[i], false, false);
                    _exit(0);
                }
                int status;
                pid_t done;
                do { done = waitpid(child, &status, 0); } while (done < 0 && errno == EINTR);
                assert(done == child && WIFEXITED(status) && WEXITSTATUS(status) == 1);
            }
        }
    }
    model_close(&model);
    puts("V4.1 Q8 attention admitted, unsupported Q4 output layouts rejected before GPU allocation: PASS");
    return 0;
}

static int check_rope_reference(const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) return 1;
    g_ds4_shape = DS4_SHAPE_FLASH41;
    for (uint32_t i = 0; i < DS4_N_LAYER; i++)
        g_ds4_compress_ratios[i] = ds4_expected_layer_compress_ratio(i);
    uint32_t count;
    int rc = 1;
    ds4_gpu_tensor *x = NULL;
    REQUIRE(fread(&count, sizeof(count), 1, fp) == 1 && count <= 10000);
    REQUIRE(ds4_gpu_init());
    x = ds4_gpu_tensor_alloc(4 * 512 * sizeof(float));
    REQUIRE(x);
    bool good = true;
    for (uint32_t i = 0; i < count; i++) {
        uint32_t args[3];
        float input[4 * 512], expected[4 * 512], actual[4 * 512];
        REQUIRE(fread(args, sizeof(args), 1, fp) == 1 && args[0] < 40 && args[2] <= 1);
        REQUIRE(fread(input, sizeof(input), 1, fp) == 1);
        REQUIRE(fread(expected, sizeof(expected), 1, fp) == 1);
        REQUIRE(ds4_gpu_tensor_write(x, 0, input, sizeof(input)));
        REQUIRE(ds4_gpu_begin_commands());
        REQUIRE(ds41_rope(x, 4, 512, args[0], args[1], args[2] != 0));
        REQUIRE(ds4_gpu_end_commands());
        REQUIRE(ds4_gpu_tensor_read(x, 0, actual, sizeof(actual)));
        double error = 0, norm = 0, worst = 0;
        for (uint32_t h = 0; h < 4; h++) for (uint32_t j = 0; j < 512; j++) {
            const uint32_t k = h * 512 + j;
            REQUIRE(isfinite(actual[k]));
            if (j < 448) REQUIRE(actual[k] == expected[k]);
            else {
                double d = actual[k] - expected[k];
                error += d * d; norm += (double)expected[k] * expected[k];
                worst = fmax(worst, fabs(d));
            }
        }
        const double relative = sqrt(error / norm);
        fprintf(stderr, "RoPE layer=%u pos=%u inverse=%u: relative RMS %.7g max %.7g\n",
                args[0], args[1], args[2], relative, worst);
        if (relative > 0.00001) good = false;
    }
    REQUIRE(good);
    rc = 0;
done:
    if (ds4_gpu_commands_active()) ds4_gpu_end_commands();
    ds4_gpu_tensor_free(x);
    ds4_gpu_cleanup();
    fclose(fp);
    return rc;
}

typedef struct {
    int progress, current, cancel_at;
    ds4_session *snapshot_session;
    bool snapshot_checked, snapshot_ready, expect_encoder;
} session_progress;

static void note_progress(void *ud, const char *event, int current, int total) {
    session_progress *p = ud;
    const bool display = strcmp(event, "prefill_display") == 0;
    assert((display || strcmp(event, "prefill_chunk") == 0) && current > 0 && current <= total);
    p->progress++;
    p->current = current;
    if (p->snapshot_session && !p->snapshot_checked) {
        ds4_session_snapshot snap = {0};
        char err[256];
        assert(!p->snapshot_session->ds41_graph.valid);
        assert(!p->expect_encoder || p->snapshot_session->ds41_graph.encoder_resident);
        assert(ds4_session_save_snapshot(p->snapshot_session, &snap, err, sizeof(err)) != 0);
        ds4_session_snapshot_free(&snap);
        p->snapshot_checked = true;
    }
    if (p->snapshot_session && !display && p->snapshot_session->checkpoint_valid) {
        ds4_session_snapshot snap = {0};
        char err[256];
        assert(p->snapshot_session->ds41_graph.valid);
        assert(ds4_session_pos(p->snapshot_session) == current);
        assert(ds4_session_save_snapshot(p->snapshot_session, &snap, err, sizeof(err)) == 0);
        ds4_session_snapshot_free(&snap);
        p->snapshot_ready = true;
    }
}

static bool cancel_progress(void *ud) {
    session_progress *p = ud;
    return p->cancel_at > 0 && p->progress >= p->cancel_at;
}

static int check_imatrix_inputs(void) {
    ds4_imatrix_collector c = {0};
    ds4_gpu_tensor *x = NULL, *mid = NULL, *selected = NULL;
    int rc = 1;
    REQUIRE(imatrix_collector_init(&c, 1, "synthetic"));
    x = ds4_gpu_tensor_alloc(DS4_N_EMBD * 4u);
    mid = ds4_gpu_tensor_alloc(DS4_N_EXPERT_USED * DS4_N_FF_EXP * 4u);
    selected = ds4_gpu_tensor_alloc(DS4_N_EXPERT_USED * 4u);
    REQUIRE(x && mid && selected);
    float *xp = ds4_gpu_tensor_contents(x), *mp = ds4_gpu_tensor_contents(mid);
    int *sp = ds4_gpu_tensor_contents(selected);
    for (uint32_t i = 0; i < DS4_N_EMBD; i++) xp[i] = (float)(i % 11u) - 5;
    for (uint32_t s = 0; s < DS4_N_EXPERT_USED; s++) {
        sp[s] = (int)(DS4_N_EXPERT - 1 - s * 17);
        for (uint32_t i = 0; i < DS4_N_FF_EXP; i++)
            mp[s * DS4_N_FF_EXP + i] = (float)(s + 1) * ((float)(i % 7u) - 3) / 8;
    }
    for (int i = 0; i < 2; i++)
        REQUIRE(imatrix_collect_tensor_batch(&c, x, mid, selected, false, 3, 1));
    for (uint32_t s = 0; s < DS4_N_EXPERT_USED; s++) {
        const uint32_t expert = (uint32_t)sp[s];
        REQUIRE(c.gate_up_count[3][expert] == 2 && c.down_count[3][expert] == 2);
        for (uint32_t i = 0; i < DS4_N_EMBD; i++)
            REQUIRE(imatrix_gate_up_ptr(&c, 3, expert)[i] == 2 * xp[i] * xp[i]);
        for (uint32_t i = 0; i < DS4_N_FF_EXP; i++) {
            const float value = mp[s * DS4_N_FF_EXP + i];
            REQUIRE(imatrix_down_ptr(&c, 3, expert)[i] == 2 * value * value);
        }
    }
    REQUIRE(c.gate_up_count[3][0] == 0 && c.observed_routes == 2 * DS4_N_EXPERT_USED);
    puts("V4.1 imatrix normalized inputs, weighted down rows and expert IDs: PASS");
    rc = 0;
done:
    ds4_gpu_tensor_free(x); ds4_gpu_tensor_free(mid); ds4_gpu_tensor_free(selected);
    imatrix_collector_free(&c);
    return rc;
}

static int check_sessions(const char *path) {
    int cwd_fd = open(".", O_RDONLY);
    ds4_engine *engine = NULL;
    ds4_session *s = NULL, *restored = NULL, *too_large = NULL;
    ds4_session_snapshot snap = {0};
    ds4_tokens tokens = {0};
    session_progress progress = {0};
    char err[256] = {0};
    int rc = 1;
    ds4_engine_options opt = {.model_path = path, .backend = DS4_BACKEND_METAL,
        .context_size = 256, .power_percent = 100, .ssd_streaming = true,
        .ssd_streaming_cache_experts = 512};
    REQUIRE(ds4_engine_open(&engine, &opt) == 0);
    REQUIRE(cwd_fd >= 0 && chdir("/") == 0);
    REQUIRE(ds4_session_create(&s, engine, 256) == 0);
    REQUIRE(ds4_session_create(&restored, engine, 256) == 0);
    REQUIRE(fchdir(cwd_fd) == 0);
    REQUIRE(check_imatrix_inputs() == 0);
    ds4_session_set_progress(s, note_progress, &progress);
    for (int i = 0; i < 129; i++) ds4_tokens_push(&tokens, 100 + i);
    REQUIRE(ds4_session_set_power(s, 50) != 0 && ds4_session_power(s) == 100);
    REQUIRE(ds4_session_set_power(s, 100) == 0);
    REQUIRE(ds4_engine_head_test(engine, &tokens) != 0);
    REQUIRE(ds4_engine_first_token_test(engine, &tokens) != 0);
    REQUIRE(ds4_engine_metal_graph_test(engine, &tokens) != 0);
    REQUIRE(ds4_engine_metal_graph_full_test(engine, &tokens) != 0);
    REQUIRE(ds4_engine_metal_graph_prompt_test(engine, &tokens, 256) != 0);
    const int positions[] = {1, 2, 3, 127, 128, 129};
    for (size_t i = 0; i < sizeof(positions) / sizeof(positions[0]); i++) {
        tokens.len = positions[i];
        REQUIRE(ds4_session_sync(s, &tokens, err, sizeof(err)) == 0);
        REQUIRE(ds4_session_pos(s) == tokens.len && progress.current == tokens.len);
        const int callbacks = progress.progress;
        REQUIRE(ds4_session_sync(s, &tokens, err, sizeof(err)) == 0);
        REQUIRE(progress.progress == callbacks);
        REQUIRE(ds4_session_save_snapshot(s, &snap, err, sizeof(err)) == 0);
        REQUIRE(snap.len == ds4_session_payload_bytes(s));
        REQUIRE(ds4_session_load_snapshot(restored, &snap, err, sizeof(err)) == 0);
        REQUIRE(ds4_session_pos(restored) == tokens.len);
        REQUIRE(memcmp(s->logits, restored->logits, DS4_N_VOCAB * sizeof(float)) == 0);
        REQUIRE(memcmp(&s->ds41_graph.history, &restored->ds41_graph.history,
                        sizeof(s->ds41_graph.history)) == 0);
        REQUIRE(ds4_session_eval(s, 987, err, sizeof(err)) == 0);
        REQUIRE(ds4_session_eval(restored, 987, err, sizeof(err)) == 0);
        REQUIRE(memcmp(s->logits, restored->logits, DS4_N_VOCAB * sizeof(float)) == 0);
        REQUIRE(ds4_session_load_snapshot(s, &snap, err, sizeof(err)) == 0);
        fprintf(stderr, "V4.1 session/snapshot parity at %d: PASS\n", tokens.len);
    }
    /* Decode computes every output head; sync skips intermediate ones. The
     * skipped scratch writes must not affect later tokens or continued sync. */
    ds4_session_invalidate(restored);
    tokens.len = 1;
    REQUIRE(ds4_session_sync(restored, &tokens, err, sizeof(err)) == 0);
    tokens.len = s->checkpoint.len;
    for (int i = 1; i < tokens.len; i++)
        REQUIRE(ds4_session_eval(restored, tokens.v[i], err, sizeof(err)) == 0);
    REQUIRE(memcmp(s->logits, restored->logits, DS4_N_VOCAB * sizeof(float)) == 0);
    REQUIRE(memcmp(&s->ds41_graph.history, &restored->ds41_graph.history,
                    sizeof(s->ds41_graph.history)) == 0);
    puts("V4.1 final-head prefill versus every-token head: PASS");
    /* Zero weights cannot reveal a missing cache span. Fill every saved byte
     * with a nonzero deterministic pattern, erase it, then restore exactly. */
    ds41_state_span spans[54];
    uint32_t span_count = ds41_state_spans(&s->ds41_graph, 129, spans);
    for (uint32_t i = 0; i < span_count; i++) {
        float *data = ds4_gpu_tensor_contents(spans[i].tensor);
        REQUIRE(data);
        for (uint64_t j = 0; j < spans[i].bytes / 4u; j++)
            data[j] = (float)(i + 1u) + (float)(j % 256u) / 256.0f;
    }
    REQUIRE(ds4_session_save_snapshot(s, &snap, err, sizeof(err)) == 0);
    for (uint32_t i = 0; i < span_count; i++)
        memset(ds4_gpu_tensor_contents(spans[i].tensor), 0, (size_t)spans[i].bytes);
    REQUIRE(ds4_session_load_snapshot(s, &snap, err, sizeof(err)) == 0);
    for (uint32_t i = 0; i < span_count; i++) {
        const float *data = ds4_gpu_tensor_contents(spans[i].tensor);
        for (uint64_t j = 0; j < spans[i].bytes / 4u; j++)
            REQUIRE(data[j] == (float)(i + 1u) + (float)(j % 256u) / 256.0f);
    }
    snap.ptr[48] ^= 1u;
    REQUIRE(ds4_session_load_snapshot(s, &snap, err, sizeof(err)) != 0);
    snap.ptr[48] ^= 1u;
    snap.len--;
    REQUIRE(ds4_session_load_snapshot(s, &snap, err, sizeof(err)) != 0);
    snap.len++;
    REQUIRE(ds4_session_load_snapshot(s, &snap, err, sizeof(err)) == 0);
    ds4_session_rewind(s, 2);
    tokens.len = 5;
    progress = (session_progress){.cancel_at = 2};
    ds4_session_set_cancel(s, cancel_progress, &progress);
    REQUIRE(ds4_session_sync(s, &tokens, err, sizeof(err)) == DS4_SESSION_SYNC_INTERRUPTED);
    REQUIRE(ds4_session_pos(s) == 2 && s->checkpoint_valid);
    REQUIRE(ds4_session_save_snapshot(s, &snap, err, sizeof(err)) == 0);
    ds4_session_invalidate(restored);
    tokens.len = 2;
    REQUIRE(ds4_session_sync(restored, &tokens, err, sizeof(err)) == 0);
    REQUIRE(memcmp(s->logits, restored->logits, DS4_N_VOCAB * sizeof(float)) == 0);
    REQUIRE(ds4_session_load_snapshot(restored, &snap, err, sizeof(err)) == 0);
    REQUIRE(ds4_session_eval(s, 987, err, sizeof(err)) == 0);
    REQUIRE(ds4_session_eval(restored, 987, err, sizeof(err)) == 0);
    REQUIRE(memcmp(s->logits, restored->logits, DS4_N_VOCAB * sizeof(float)) == 0);
    REQUIRE(ds4_session_load_snapshot(s, &snap, err, sizeof(err)) == 0);
    tokens.len = 5;
    ds4_session_set_cancel(s, NULL, NULL);
    REQUIRE(ds4_session_sync(s, &tokens, err, sizeof(err)) == 0 && progress.current == 5);
    REQUIRE(ds4_session_eval(s, -1, err, sizeof(err)) != 0);
    REQUIRE(!s->checkpoint_valid);
    REQUIRE(ds4_session_sync(s, &tokens, err, sizeof(err)) == 0);
    REQUIRE(ds4_session_create(&too_large, engine, 1048577) != 0 && too_large == NULL);
    puts("V4.1 public sessions, prefix reuse, snapshots, cancellation and bounds: PASS");
    rc = 0;
done:
    if (rc) fprintf(stderr, "session error: %s\n", err);
    ds4_session_snapshot_free(&snap);
    ds4_tokens_free(&tokens);
    ds4_session_free(restored);
    ds4_session_free(too_large);
    ds4_session_free(s);
    ds4_engine_close(engine);
    if (cwd_fd >= 0) { fchdir(cwd_fd); close(cwd_fd); }
    return rc;
}

static int check_long_sessions(const char *path, const char *prompt_path) {
    ds4_engine *engine = NULL;
    ds4_session *s = NULL, *restored = NULL;
    ds4_session_snapshot snap = {0};
    ds4_tokens tokens = {0};
    session_progress progress = {0};
    char *prompt = NULL, err[256] = {0};
    size_t prompt_bytes = 0;
    int rc = 1;
    ds4_engine_options opt = {.model_path = path, .backend = DS4_BACKEND_METAL,
        .context_size = 32768, .power_percent = 100, .ssd_streaming = true,
        .ssd_streaming_cache_bytes = UINT64_C(64) * 1024 * 1024 * 1024};
    REQUIRE(imatrix_read_text_file(prompt_path, &prompt, &prompt_bytes));
    REQUIRE(prompt_bytes > 0 && ds4_engine_open(&engine, &opt) == 0);
    ds4_encode_chat_prompt(engine, NULL, prompt, DS4_THINK_NONE, &tokens);
    const int total = tokens.len;
    REQUIRE(total >= 16385 && total < 32768);
    REQUIRE(ds4_session_create(&s, engine, 32768) == 0);
    REQUIRE(ds4_session_create(&restored, engine, 32768) == 0);
    ds4_session_set_progress(s, note_progress, &progress);
    const int positions[] = {127, 128, 129, 511, 512, 513,
                            1023, 1024, 1025, 16383, 16384, 16385};
    const double start = now_sec();
    for (size_t i = 0; i < sizeof(positions) / sizeof(*positions); i++) {
        tokens.len = positions[i];
        REQUIRE(ds4_session_sync(s, &tokens, err, sizeof(err)) == 0);
        REQUIRE(ds4_session_pos(s) == tokens.len);
        REQUIRE(progress.current == tokens.len);
        const int callbacks = progress.progress;
        REQUIRE(ds4_session_sync(s, &tokens, err, sizeof(err)) == 0);
        REQUIRE(progress.progress == callbacks);
        REQUIRE(ds4_session_save_snapshot(s, &snap, err, sizeof(err)) == 0);
        REQUIRE(snap.len == ds4_session_payload_bytes(s));
        ds4_session_invalidate(restored);
        REQUIRE(ds4_session_load_snapshot(restored, &snap, err, sizeof(err)) == 0);
        REQUIRE(memcmp(s->logits, restored->logits, DS4_N_VOCAB * sizeof(float)) == 0);
        REQUIRE(memcmp(&s->ds41_graph.history, &restored->ds41_graph.history,
                       sizeof(s->ds41_graph.history)) == 0);
        /* Continue beyond each restored boundary so latent KV, reused index
         * keys and a pending compression pair all affect the comparison. */
        const int token = tokens.len < total ? tokens.v[tokens.len] :
                          sample_argmax(s->logits, DS4_N_VOCAB);
        REQUIRE(ds4_session_eval(s, token, err, sizeof(err)) == 0);
        REQUIRE(ds4_session_eval(restored, token, err, sizeof(err)) == 0);
        REQUIRE(memcmp(s->logits, restored->logits, DS4_N_VOCAB * sizeof(float)) == 0);
        REQUIRE(ds4_session_load_snapshot(s, &snap, err, sizeof(err)) == 0);
        fprintf(stderr, "V4.1 long sync/restore at %d: PASS (%.1fs, %.2f MiB snapshot)\n",
                tokens.len, now_sec() - start, (double)snap.len / 1048576.0);
    }
    puts("V4.1 continued prefill and restored logits through 16385 tokens: PASS");
    rc = 0;
done:
    if (rc) fprintf(stderr, "long session error: %s\n", err);
    ds4_session_snapshot_free(&snap);
    ds4_tokens_free(&tokens);
    free(prompt);
    ds4_session_free(restored);
    ds4_session_free(s);
    ds4_engine_close(engine);
    return rc;
}

static int check_prefill(const char *path, const char *prompt_path,
                         bool encoder, bool cancel_only) {
    ds4_engine *engine = NULL;
    ds4_session *control = NULL, *candidate = NULL;
    ds4_session_snapshot snap = {0};
    ds4_tokens tokens = {0};
    char *prompt = NULL, err[256] = {0};
    size_t prompt_bytes;
    int rc = 1;
    ds4_engine_options opt = {.model_path = path, .backend = DS4_BACKEND_METAL,
        .context_size = encoder ? 17408 : 4096, .power_percent = 100, .ssd_streaming = true,
        .ssd_streaming_cache_bytes = (encoder ? UINT64_C(80) : UINT64_C(32)) << 30};
    const char *disable = encoder ? "DS4_METAL_DISABLE_V41_ENCODER_RESIDENCY" :
                                   "DS4_METAL_DISABLE_V41_LAYER_PREFILL";
    if (encoder) setenv("DS4_METAL_DISABLE_V41_WIDE_PREFILL", "1", 1);
    REQUIRE(!getenv("DS4_METAL_DISABLE_V41_LAYER_PREFILL"));
    REQUIRE(!getenv(disable));
    REQUIRE(imatrix_read_text_file(prompt_path, &prompt, &prompt_bytes));
    REQUIRE(ds4_engine_open(&engine, &opt) == 0);
    ds4_encode_chat_prompt(engine, NULL, prompt, DS4_THINK_NONE, &tokens);
    REQUIRE(tokens.len >= (encoder ? 16903 : 3588));
    REQUIRE(ds4_session_create(&control, engine, opt.context_size) == 0);
    REQUIRE(ds4_session_create(&candidate, engine, opt.context_size) == 0);
    const int short_frontiers[] = {257, 513, 1025, 1537, 3585};
    const int long_frontiers[] = {16385, 16900};
    const int *frontiers = encoder ? long_frontiers : short_frontiers;
    const size_t nfrontiers = encoder ? sizeof(long_frontiers) / sizeof(*long_frontiers) :
                                       sizeof(short_frontiers) / sizeof(*short_frontiers);
    for (size_t i = 0; i < nfrontiers && !cancel_only; i++) {
        tokens.len = frontiers[i];
        REQUIRE(setenv(disable, "1", 1) == 0);
        double start = now_sec();
        REQUIRE(ds4_session_sync(control, &tokens, err, sizeof(err)) == 0);
        const double serial = now_sec() - start;
        REQUIRE(unsetenv(disable) == 0);
        const bool check_live = i == (encoder ? 0u : 2u);
        session_progress live = {.snapshot_session = check_live ? candidate : NULL,
                                 .expect_encoder = encoder};
        ds4_session_set_progress(candidate, note_progress, &live);
        start = now_sec();
        REQUIRE(ds4_session_sync(candidate, &tokens, err, sizeof(err)) == 0);
        const double batch = now_sec() - start;
        ds4_session_set_progress(candidate, NULL, NULL);
        REQUIRE(!check_live || (live.snapshot_checked && live.snapshot_ready));
        REQUIRE(memcmp(control->logits, candidate->logits, DS4_N_VOCAB * 4u) == 0);
        ds41_state_span a[54], b[54];
        const uint32_t count = ds41_state_spans(&control->ds41_graph, tokens.len, a);
        REQUIRE(ds41_state_spans(&candidate->ds41_graph, tokens.len, b) == count);
        for (uint32_t j = 0; j < count; j++) {
            REQUIRE(a[j].bytes == b[j].bytes);
            REQUIRE(memcmp(ds4_gpu_tensor_contents(a[j].tensor),
                           ds4_gpu_tensor_contents(b[j].tensor), (size_t)a[j].bytes) == 0);
        }
        REQUIRE(ds4_session_save_snapshot(candidate, &snap, err, sizeof(err)) == 0);
        REQUIRE(ds4_session_load_snapshot(candidate, &snap, err, sizeof(err)) == 0);
        for (int j = 0; j < 3; j++) {
            const int token = tokens.v[tokens.len++];
            REQUIRE(ds4_session_eval(control, token, err, sizeof(err)) == 0);
            REQUIRE(ds4_session_eval(candidate, token, err, sizeof(err)) == 0);
            REQUIRE(memcmp(control->logits, candidate->logits, DS4_N_VOCAB * 4u) == 0);
        }
        fprintf(stderr, "V4.1 prefill frontier %d: exact logits/cache/continued decode, %.3fs %s / %.3fs default\n",
                frontiers[i], serial, encoder ? "two-layer" : "serial", batch);
    }
    ds4_session_invalidate(candidate);
    const uint32_t cache_budget = ds4_gpu_stream_expert_cache_configured_count();
    session_progress progress = {.cancel_at = 20, .snapshot_session = candidate,
                                 .expect_encoder = encoder};
    ds4_session_set_progress(candidate, note_progress, &progress);
    ds4_session_set_cancel(candidate, cancel_progress, &progress);
    tokens.len = encoder ? 16385 : 257;
    REQUIRE(ds4_session_sync(candidate, &tokens, err, sizeof(err)) == DS4_SESSION_SYNC_INTERRUPTED);
    REQUIRE(!candidate->checkpoint_valid && !candidate->ds41_graph.valid);
    REQUIRE(!candidate->ds41_graph.encoder_resident);
    REQUIRE(ds4_gpu_stream_expert_cache_configured_count() == cache_budget);
    REQUIRE(ds4_session_save_snapshot(candidate, &snap, err, sizeof(err)) != 0);
    ds4_session_set_cancel(candidate, NULL, NULL);
    ds4_session_set_progress(candidate, NULL, NULL);
    ds4_session_invalidate(control);
    tokens.len = 129;
    REQUIRE(ds4_session_sync(candidate, &tokens, err, sizeof(err)) == 0);
    REQUIRE(ds4_session_sync(control, &tokens, err, sizeof(err)) == 0);
    REQUIRE(memcmp(control->logits, candidate->logits, DS4_N_VOCAB * 4u) == 0);
    puts("V4.1 layer-major cancellation refuses partial state and rebuilds correctly: PASS");
    rc = 0;
done:
    unsetenv(disable);
    if (encoder) unsetenv("DS4_METAL_DISABLE_V41_WIDE_PREFILL");
    if (rc) fprintf(stderr, "prefill error: %s\n", err);
    ds4_session_snapshot_free(&snap);
    ds4_tokens_free(&tokens);
    free(prompt);
    ds4_session_free(candidate); ds4_session_free(control);
    ds4_engine_close(engine);
    return rc;
}

typedef struct {
    const char *model, *prompt;
    bool sessions;
    int result;
} prefill_thread_args;

static void *prefill_thread(void *ud) {
    prefill_thread_args *args = ud;
    args->result = args->sessions ? check_long_sessions(args->model, args->prompt) :
        check_prefill(args->model, args->prompt, false, false);
    return NULL;
}

static int check_thread_prefill(const char *model, const char *prompt, bool sessions) {
    pthread_attr_t attr;
    pthread_t thread;
    prefill_thread_args args = {model, prompt, sessions, 1};
    assert(pthread_attr_init(&attr) == 0);
    assert(pthread_attr_setstacksize(&attr, 512u * 1024u) == 0);
    const int rc = pthread_create(&thread, &attr, prefill_thread, &args);
    pthread_attr_destroy(&attr);
    assert(rc == 0);
    assert(pthread_join(thread, NULL) == 0);
    return args.result;
}

static bool cancel_encoder_load(void *ud) {
    unsigned *calls = ud;
    return ++*calls == 3;
}

static int check_encoder(const char *path, const char *prompt_path) {
    ds4_engine *engine = NULL;
    ds4_session *control = NULL, *candidate = NULL;
    ds4_tokens tokens = {0};
    ds41_encoder_residency residency = {0};
    char *prompt = NULL, err[256] = {0};
    size_t prompt_bytes;
    int rc = 1;
    ds4_engine_options opt = {.model_path = path, .backend = DS4_BACKEND_METAL,
        .context_size = 4096, .power_percent = 100, .ssd_streaming = true,
        .ssd_streaming_cache_bytes = UINT64_C(80) << 30};
    setenv("DS4_METAL_DISABLE_V41_WIDE_PREFILL", "1", 1);
    REQUIRE(imatrix_read_text_file(prompt_path, &prompt, &prompt_bytes));
    REQUIRE(ds4_engine_open(&engine, &opt) == 0);
    ds4_encode_chat_prompt(engine, NULL, prompt, DS4_THINK_NONE, &tokens);
    REQUIRE(tokens.len > 1025);
    REQUIRE(ds4_session_create(&control, engine, 4096) == 0);
    REQUIRE(ds4_session_create(&candidate, engine, 4096) == 0);
    const uint32_t budget = ds4_gpu_stream_expert_cache_configured_count();
    REQUIRE(budget >= 20u * DS4_N_EXPERT);
    ds41_encoder_acquire(&candidate->ds41_graph, &engine->model, &engine->weights,
                         16383, &residency, NULL, NULL);
    REQUIRE(!candidate->ds41_graph.encoder_resident && !residency.cache_budget);
    REQUIRE(ds4_gpu_stream_expert_cache_configured_count() == budget);
    ds4_gpu_set_streaming_expert_cache_budget(1);
    ds41_encoder_acquire(&candidate->ds41_graph, &engine->model, &engine->weights,
                         16384, &residency, NULL, NULL);
    REQUIRE(!candidate->ds41_graph.encoder_resident && !residency.cache_budget);
    REQUIRE(ds4_gpu_stream_expert_cache_configured_count() == 1);
    ds4_gpu_set_streaming_expert_cache_budget(budget);
    tokens.len = 1025;
    REQUIRE(ds4_session_sync(control, &tokens, err, sizeof(err)) == 0);
    tokens.len = 257;
    REQUIRE(ds4_session_sync(candidate, &tokens, err, sizeof(err)) == 0);
    unsigned calls = 0;
    ds4_gpu_print_memory_report("before encoder residency");
    ds41_encoder_acquire(&candidate->ds41_graph, &engine->model, &engine->weights,
                         16384, &residency, cancel_encoder_load, &calls);
    REQUIRE(calls == 3 && !candidate->ds41_graph.encoder_resident);
    REQUIRE(!residency.cache_budget && ds4_gpu_stream_expert_cache_configured_count() == budget);
    for (unsigned i = 0; i < 60; i++) REQUIRE(!residency.span[i].locked);
    ds4_gpu_print_memory_report("after cancelled encoder load");
    const double start = now_sec();
    ds41_encoder_acquire(&candidate->ds41_graph, &engine->model, &engine->weights,
                         16384, &residency, NULL, NULL);
    REQUIRE(candidate->ds41_graph.encoder_resident);
    REQUIRE(ds4_gpu_stream_expert_cache_configured_count() == 0);
    fprintf(stderr, "V4.1 encoder load: %.3fs\n", now_sec() - start);
    ds4_gpu_print_memory_report("encoder resident");
    REQUIRE(ds41_graph_prefill(&candidate->ds41_graph, &engine->model, &engine->weights,
                              tokens.v + 257, 768, NULL, NULL, 1025, NULL, NULL));
    REQUIRE(ds41_graph_logits(&candidate->ds41_graph, &engine->model,
                             &engine->weights, candidate->logits));
    REQUIRE(memcmp(control->logits, candidate->logits, DS4_N_VOCAB * 4u) == 0);
    ds41_state_span a[54], b[54];
    const uint32_t count = ds41_state_spans(&control->ds41_graph, 1025, a);
    REQUIRE(ds41_state_spans(&candidate->ds41_graph, 1025, b) == count);
    for (uint32_t j = 0; j < count; j++) {
        REQUIRE(a[j].bytes == b[j].bytes);
        REQUIRE(memcmp(ds4_gpu_tensor_contents(a[j].tensor),
                       ds4_gpu_tensor_contents(b[j].tensor), (size_t)a[j].bytes) == 0);
    }
    ds41_encoder_release(&candidate->ds41_graph, &engine->model, &residency);
    REQUIRE(!candidate->ds41_graph.encoder_resident);
    REQUIRE(ds4_gpu_stream_expert_cache_configured_count() == budget);
    ds4_gpu_print_memory_report("encoder released");
    REQUIRE(ds41_graph_step(&candidate->ds41_graph, &engine->model,
                           &engine->weights, tokens.v[1025], candidate->logits));
    REQUIRE(ds4_session_eval(control, tokens.v[1025], err, sizeof(err)) == 0);
    REQUIRE(memcmp(control->logits, candidate->logits, DS4_N_VOCAB * 4u) == 0);
    puts("V4.1 resident encoder: exact logits/cache/decode and bounded cancellation/release: PASS");
    rc = 0;
done:
    if (candidate && engine)
        ds41_encoder_release(&candidate->ds41_graph, &engine->model, &residency);
    unsetenv("DS4_METAL_DISABLE_V41_WIDE_PREFILL");
    if (rc) fprintf(stderr, "encoder error: %s\n", err);
    ds4_tokens_free(&tokens);
    free(prompt);
    ds4_session_free(candidate); ds4_session_free(control);
    ds4_engine_close(engine);
    return rc;
}

static int check_chat(const char *path, const char *level, const char *system, const char *prompt) {
    ds4_model model = {.fd = -1};
    ds4_engine engine = {0};
    ds4_tokens direct = {0}, incremental = {0};
    ds4_think_mode mode;
    int rc = 1;
    if (!strcmp(level, "high")) mode = DS4_THINK_HIGH;
    else if (!strcmp(level, "max")) mode = DS4_THINK_MAX;
    else if (!ds4_think_mode_parse_level(level, &mode)) return 2;
    model_open(&model, path, true, false);
    config_validate_model(&model);
    vocab_load(&engine.vocab, &model);
    REQUIRE(ds4_engine_is_deepseek41(&engine));
    REQUIRE(ds4_think_mode_for_context(mode, 256) == mode);
    ds4_encode_chat_prompt(&engine, system, prompt, mode, &direct);
    ds4_chat_begin(&engine, &incremental);
    ds4_chat_append_think_prefix(&engine, &incremental, mode);
    if (system[0]) ds4_chat_append_message(&engine, &incremental, "system", system);
    ds4_chat_append_message(&engine, &incremental, "user", prompt);
    ds4_chat_append_assistant_prefix(&engine, &incremental, mode);
    REQUIRE(direct.len == incremental.len && ds4_tokens_starts_with(&direct, &incremental));
    dump_tokens_fp(stdout, &engine.vocab, &direct);
    const int previous = incremental.len;
    ds4_chat_append_message(&engine, &incremental, "system", "A later instruction");
    REQUIRE(incremental.v[previous] == engine.vocab.system_id);
    const char *invalid[] = {"", "-1", "+1", "101", "1.5", "1e2", " 25", "25 ", "99999999999999999"};
    for (size_t i = 0; i < sizeof(invalid) / sizeof(*invalid); i++)
        REQUIRE(!ds4_think_mode_parse_level(invalid[i], &mode));
    REQUIRE(ds4_think_mode_level((ds4_think_mode)INT_MIN) == -1);
    rc = 0;
done:
    ds4_tokens_free(&direct); ds4_tokens_free(&incremental);
    vocab_free(&engine.vocab); model_close(&model);
    return rc;
}

static bool partition_close(const char *name, uint32_t layer,
                            const float *expected, const float *actual, uint32_t count) {
    double error = 0, norm = 0, worst = 0;
    for (uint32_t i = 0; i < count; i++) {
        if (!isfinite(expected[i]) || !isfinite(actual[i])) return false;
        const double d = (double)actual[i] - expected[i];
        error += d * d;
        norm += (double)expected[i] * expected[i];
        worst = fmax(worst, fabs(d));
    }
    const double relative = sqrt(error / fmax(norm, 1e-30));
    fprintf(stderr, "V4.1 %s layer %u: relative RMS %.9g max %.9g\n",
            name, layer, relative, worst);
    return relative < 1e-4;
}

static void *publish_engram_test_rows(void *arg) {
    ds41_engram_prefetch *p = arg;
    p->ok = true;
    for (uint32_t i = 0; i < p->count; i++) {
        p->out[i] = (float)i + 1;
        __atomic_store_n(&p->ready, i + 1u, __ATOMIC_RELEASE);
        if (i % 7u == 0) sched_yield();
    }
    __atomic_store_n(&p->done, true, __ATOMIC_RELEASE);
    return NULL;
}

static bool cancel_engram_wait(void *unused) {
    (void)unused;
    return true;
}

static int check_engram_prefetch_ready(void) {
    int rc = 1;
    float values[32];
    ds41_engram_prefetch p = {0};
    for (uint32_t trial = 0; trial < 1000; trial++) {
        memset(values, 0, sizeof(values));
        p = (ds41_engram_prefetch){.count = 32, .out = values};
        REQUIRE(pthread_create(&p.thread, NULL, publish_engram_test_rows, &p) == 0);
        p.active = true;
        for (uint32_t i = 1; i <= p.count; i++) {
            REQUIRE(ds41_engram_prefetch_wait(&p, i, NULL, NULL));
            REQUIRE(values[i - 1u] == (float)i);
        }
        REQUIRE(!ds41_engram_prefetch_wait(&p, 33, NULL, NULL));
        REQUIRE(ds41_engram_prefetch_join(&p, false));
    }
    ds41_engram_prefetch stopped = {.active = true, .count = 32, .ready = 16, .done = true};
    REQUIRE(ds41_engram_prefetch_wait(&stopped, 16, NULL, NULL));
    REQUIRE(!ds41_engram_prefetch_wait(&stopped, 17, NULL, NULL));
    stopped.done = false;
    REQUIRE(!ds41_engram_prefetch_wait(&stopped, 17, cancel_engram_wait, NULL));
    const ds4_engram_table bad = {.fd = -1, .rows = 1};
    const uint32_t ids[DS4_ENGRAM_COLS * 2] = {0};
    p = (ds41_engram_prefetch){.table = &bad, .ids = ids, .count = 1, .out = values};
    REQUIRE(pthread_create(&p.thread, NULL, ds41_engram_prefetch_read, &p) == 0);
    p.active = true;
    REQUIRE(!ds41_engram_prefetch_wait(&p, 1, NULL, NULL));
    REQUIRE(!ds41_engram_prefetch_join(&p, false));
    puts("V4.1 Engram prefix publication, completion race, read failure and cancellation: PASS");
    rc = 0;
done:
    if (p.active) ds41_engram_prefetch_join(&p, true);
    return rc;
}

static int check_wide_prefill(const char *path, const char *prompt_path,
                              bool cancel_only, bool decoder_cancel, const char *control_env) {
    ds4_engine *engine = NULL;
    ds4_session *control = NULL, *candidate = NULL;
    ds4_tokens tokens = {0};
    char *prompt = NULL;
    size_t prompt_bytes;
    int rc = 1;
    const bool engram_overlap = control_env &&
        (!strcmp(control_env, "DS4_METAL_DISABLE_V41_ENGRAM_PREFETCH") ||
         !strcmp(control_env, "DS4_METAL_DISABLE_V41_ENGRAM_PIPELINE"));
    const bool chunk_partition = control_env && !strcmp(control_env, "DS4_METAL_DISABLE_V41_WIDE_CHUNK");
    const bool compact_carry = control_env && !strcmp(control_env, "DS4_METAL_DISABLE_V41_COMPACT_CARRY");
    const bool prefill_alias = control_env && !strcmp(control_env, "DS4_METAL_DISABLE_V41_PREFILL_ALIAS");
    const bool packed_wide = control_env && !strcmp(control_env, "DS4_METAL_DISABLE_V41_PACKED_M32N128");
    const bool wide_case = chunk_partition || packed_wide || prefill_alias;
    const char *label = engram_overlap ? "Engram overlap" : control_env ? control_env : "wide";
    const uint32_t context = wide_case ? 34816u : 24576u;
    ds4_engine_options opt = {.model_path = path, .backend = DS4_BACKEND_METAL,
        .context_size = context, .power_percent = 100, .ssd_streaming = true,
        .ssd_streaming_cache_bytes = UINT64_C(64) << 30};
    if (!decoder_cancel && !control_env) setenv("DS4_METAL_DISABLE_V41_DECODER_SUFFIX", "1", 1);
    REQUIRE(imatrix_read_text_file(prompt_path, &prompt, &prompt_bytes));
    REQUIRE(ds4_engine_open(&engine, &opt) == 0);
    ds4_encode_chat_prompt(engine, NULL, prompt, DS4_THINK_NONE, &tokens);
    REQUIRE(tokens.len > (wide_case ? 26624 : 18433));
    if (chunk_partition || compact_carry || prefill_alias) setenv(control_env, "1", 1);
    REQUIRE(ds4_session_create(&control, engine, context) == 0);
    if (chunk_partition || compact_carry || prefill_alias) unsetenv(control_env);
    REQUIRE(ds4_session_create(&candidate, engine, context) == 0);
    ds41_gpu_graph *a = &control->ds41_graph, *b = &candidate->ds41_graph;
    REQUIRE(!chunk_partition || (a->prefill_cap == 2048u && b->prefill_cap == 8192u));
    REQUIRE(!packed_wide || (a->prefill_cap == 8192u && b->prefill_cap == 8192u));
    REQUIRE(!compact_carry || (!a->compact_carry && b->compact_carry));
    if (prefill_alias) {
        REQUIRE(!a->prefill_alias && b->prefill_alias);
        const uint64_t saved = (uint64_t)b->prefill_cap *
            ((2u * DS4_N_HC + 3u + DS4_N_EXPERT_USED) * DS4_N_EMBD +
             (2u * DS4_N_EXPERT_USED + 3u) * DS4_N_FF_EXP) * sizeof(float);
        REQUIRE(a->allocation_bytes == b->allocation_bytes + saved);
        REQUIRE(ds4_gpu_tensor_contents(b->batch.flat_norm) == ds4_gpu_tensor_contents(b->batch.q));
        REQUIRE(ds4_gpu_tensor_contents(b->batch.experts) == ds4_gpu_tensor_contents(b->batch.q));
        REQUIRE(ds4_gpu_tensor_contents(b->batch.engram_kv) == ds4_gpu_tensor_contents(b->batch.q));
        const uint64_t cap = b->prefill_cap;
        const float *heads = ds4_gpu_tensor_contents(b->batch.heads);
        const float *low = ds4_gpu_tensor_contents(b->batch.low);
        REQUIRE(ds4_gpu_tensor_contents(b->batch.gate) == heads);
        REQUIRE(ds4_gpu_tensor_contents(b->batch.up) == heads + cap * DS4_N_EXPERT_USED * DS4_N_FF_EXP);
        REQUIRE(ds4_gpu_tensor_contents(b->batch.shared_gate) == low);
        REQUIRE(ds4_gpu_tensor_contents(b->batch.shared_up) == low + cap * DS4_N_FF_EXP);
        REQUIRE(ds4_gpu_tensor_contents(b->batch.shared_mid) == low + cap * 2u * DS4_N_FF_EXP);
        REQUIRE(ds4_gpu_tensor_contents(b->batch.shared) == ds4_gpu_tensor_contents(b->batch.x));
        REQUIRE(ds4_gpu_tensor_contents(b->batch.routed) == ds4_gpu_tensor_contents(b->batch.block));
        REQUIRE(ds4_gpu_tensor_contents(a->batch.flat_norm) != ds4_gpu_tensor_contents(a->batch.q));
        REQUIRE(ds4_gpu_tensor_contents(a->batch.experts) != ds4_gpu_tensor_contents(a->batch.q));
        REQUIRE(ds4_gpu_tensor_contents(a->batch.engram_kv) != ds4_gpu_tensor_contents(a->batch.q));
    }
    const uint32_t counts[] = {4096, 6144, wide_case ? 16384u : 8192u};
    for (size_t i = 0; !cancel_only && i < sizeof(counts) / sizeof(*counts); i++) {
        const uint32_t start = a->pos, count = counts[i];
        double t0 = now_sec();
        if (control_env) {
            setenv(control_env, "1", 1);
            REQUIRE(ds41_graph_prefill(a, &engine->model, &engine->weights, tokens.v + start,
                count, NULL, NULL, (int)(start + count), NULL, NULL));
            unsetenv(control_env);
        } else {
            for (uint32_t off = 0; off < count; off += a->prefill_cap) {
                const uint32_t rows = count - off < a->prefill_cap ? count - off : a->prefill_cap;
                REQUIRE(ds41_graph_prefill(a, &engine->model, &engine->weights,
                    tokens.v + start + off, rows, NULL, NULL,
                (int)(start + count), NULL, NULL));
            }
        }
        const double reference_seconds = now_sec() - t0;
        session_progress progress = {0};
        t0 = now_sec();
        REQUIRE(ds41_graph_prefill(b, &engine->model, &engine->weights,
            tokens.v + start, count, note_progress, &progress, (int)(start + count), NULL, NULL));
        const double candidate_seconds = now_sec() - t0;
        REQUIRE(progress.current == (int)(start + count));
        const uint32_t chunk = ds41_encoder_chunk_cap(b, count);
        const uint32_t encoder_chunks = (count + chunk - 1u) / chunk;
        const uint32_t decoder_chunk = count > chunk && chunk > 2048u ? 2048u : chunk;
        const uint32_t decoder_chunks = (count + decoder_chunk - 1u) / decoder_chunk;
        REQUIRE(control_env || progress.progress == (int)(20u * (encoder_chunks + decoder_chunks)));
        REQUIRE(a->pos == b->pos && !memcmp(&a->history, &b->history, sizeof(a->history)));
        ds41_state_span sa[64], sb[64];
        const uint32_t n = ds41_state_spans(a, a->pos, sa);
        REQUIRE(n == ds41_state_spans(b, b->pos, sb));
        for (uint32_t j = 0; j < n; j++) {
            REQUIRE(sa[j].bytes == sb[j].bytes);
            if (memcmp(ds4_gpu_tensor_contents(sa[j].tensor),
                       ds4_gpu_tensor_contents(sb[j].tensor), (size_t)sa[j].bytes)) {
                fprintf(stderr, "wide prefill mismatch: frontier %u state span %u\n", a->pos, j);
                goto done;
            }
        }
        REQUIRE(ds41_graph_logits(a, &engine->model, &engine->weights, control->logits));
        REQUIRE(ds41_graph_logits(b, &engine->model, &engine->weights, candidate->logits));
        REQUIRE(!memcmp(control->logits, candidate->logits, DS4_N_VOCAB * sizeof(float)));
        fprintf(stderr, "V4.1 %s prefill start=%u count=%u: exact state/logits, %.2f -> %.2f t/s\n",
                label, start, count,
                count / reference_seconds, count / candidate_seconds);
    }
    if (!cancel_only) {
        REQUIRE(ds41_graph_step(a, &engine->model, &engine->weights, tokens.v[a->pos], control->logits));
        REQUIRE(ds41_graph_step(b, &engine->model, &engine->weights, tokens.v[b->pos], candidate->logits));
        REQUIRE(!memcmp(control->logits, candidate->logits, DS4_N_VOCAB * sizeof(float)));
    }
    if (chunk_partition) {
        REQUIRE(tokens.len >= 32768);
        ds4_session_invalidate(control);
        ds4_session_invalidate(candidate);
        tokens.len = 32768;
        char err[256];
        REQUIRE(ds4_session_sync(control, &tokens, err, sizeof(err)) == 0);
        REQUIRE(ds4_session_sync(candidate, &tokens, err, sizeof(err)) == 0);
        REQUIRE(a->carry_cap == b->carry_cap && a->pos == 32768u && b->pos == a->pos);
        ds41_state_span sa[64], sb[64];
        const uint32_t n = ds41_state_spans(a, a->pos, sa);
        REQUIRE(n == ds41_state_spans(b, b->pos, sb));
        for (uint32_t j = 0; j < n; j++) {
            REQUIRE(sa[j].bytes == sb[j].bytes);
            REQUIRE(!memcmp(ds4_gpu_tensor_contents(sa[j].tensor),
                            ds4_gpu_tensor_contents(sb[j].tensor), (size_t)sa[j].bytes));
        }
        REQUIRE(!memcmp(control->logits, candidate->logits, DS4_N_VOCAB * sizeof(float)));
        puts("V4.1 32k automatic prefill partition: exact state/logits PASS");
    }
    ds4_session_invalidate(candidate);
    ds4_session_invalidate(control);
    const uint32_t cancel_chunk = ds41_encoder_chunk_cap(b, 8192u);
    session_progress cancelled = {.cancel_at = decoder_cancel ?
        (int)(20u * ((8192u + cancel_chunk - 1u) / cancel_chunk) + 1u) : engram_overlap ? 1 : 7,
                                   .snapshot_session = candidate};
    ds4_session_set_progress(candidate, note_progress, &cancelled);
    ds4_session_set_cancel(candidate, cancel_progress, &cancelled);
    tokens.len = 8192;
    char err[256];
    REQUIRE(ds4_session_sync(candidate, &tokens, err, sizeof(err)) == DS4_SESSION_SYNC_INTERRUPTED);
    REQUIRE(cancelled.snapshot_checked && !b->valid && !candidate->checkpoint_valid);
    REQUIRE(!decoder_cancel || cancelled.current > 4096);
    ds4_session_set_progress(candidate, NULL, NULL);
    ds4_session_set_cancel(candidate, NULL, NULL);
    if (engram_overlap) {
        for (uint32_t table = 0; table < 2; table++) {
            ds4_session_invalidate(candidate);
            const int saved = b->table[table].fd;
            b->table[table].fd = -1;
            const bool failed = !ds41_graph_prefill(b, &engine->model, &engine->weights,
                tokens.v, 4096, NULL, NULL, 4096, NULL, NULL);
            b->table[table].fd = saved;
            REQUIRE(failed && !b->valid);
        }
    }
    tokens.len = 129;
    REQUIRE(ds4_session_sync(candidate, &tokens, err, sizeof(err)) == 0);
    REQUIRE(ds4_session_sync(control, &tokens, err, sizeof(err)) == 0);
    REQUIRE(!memcmp(control->logits, candidate->logits, DS4_N_VOCAB * sizeof(float)));
    puts(engram_overlap ? "V4.1 Engram overlap: exact state/logits, cancellation, read errors and rebuild PASS" :
        control_env ? "V4.1 batched prefill: exact state/logits, cancellation and rebuild PASS" :
        decoder_cancel ? "V4.1 decoder suffix cancellation and exact rebuild: PASS" :
        "V4.1 wide initial/continued prefill and following decode are byte-identical: PASS");
    rc = 0;
done:
    if (control_env) unsetenv(control_env);
    unsetenv("DS4_METAL_DISABLE_V41_DECODER_SUFFIX");
    if (ds4_gpu_commands_active()) ds4_gpu_end_commands();
    ds4_tokens_free(&tokens);
    free(prompt);
    ds4_session_free(candidate); ds4_session_free(control); ds4_engine_close(engine);
    return rc;
}

static int check_prefill_alias_fallback(const char *path, const char *prompt_path) {
    ds4_engine *engine = NULL;
    ds4_session *control = NULL, *candidate = NULL;
    ds4_tokens tokens = {0};
    char *prompt = NULL;
    size_t bytes;
    int rc = 1;
    ds4_engine_options opt = {.model_path = path, .backend = DS4_BACKEND_METAL,
        .context_size = 1024, .power_percent = 100, .ssd_streaming = true,
        .ssd_streaming_cache_bytes = UINT64_C(64) << 30};
    const char *flags[] = {"DS4_METAL_DISABLE_V41_BATCH_ATTN",
        "DS4_METAL_DISABLE_V41_BATCH_MOE", "DS4_METAL_DISABLE_V41_BATCH_HC"};
    REQUIRE(imatrix_read_text_file(prompt_path, &prompt, &bytes));
    REQUIRE(ds4_engine_open(&engine, &opt) == 0);
    ds4_encode_chat_prompt(engine, NULL, prompt, DS4_THINK_NONE, &tokens);
    REQUIRE(tokens.len > 50);
    setenv("DS4_METAL_DISABLE_V41_PREFILL_ALIAS", "1", 1);
    REQUIRE(ds4_session_create(&control, engine, 1024) == 0);
    unsetenv("DS4_METAL_DISABLE_V41_PREFILL_ALIAS");
    REQUIRE(ds4_session_create(&candidate, engine, 1024) == 0);
    ds41_gpu_graph *a = &control->ds41_graph, *b = &candidate->ds41_graph;
    REQUIRE(!a->prefill_alias && b->prefill_alias);
    for (uint32_t mode = 0; mode < 8; mode++) {
        for (uint32_t j = 0; j < 3; j++) {
            if (mode & (1u << j)) setenv(flags[j], "1", 1);
            else unsetenv(flags[j]);
        }
        ds41_graph_reset(a); ds41_graph_reset(b);
        for (uint32_t pass = 0; pass < 2; pass++) {
            const uint32_t start = a->pos, count = pass ? 29 : 19;
            REQUIRE(ds41_graph_prefill(a, &engine->model, &engine->weights,
                tokens.v + start, count, NULL, NULL, (int)(start + count), NULL, NULL));
            REQUIRE(ds41_graph_prefill(b, &engine->model, &engine->weights,
                tokens.v + start, count, NULL, NULL, (int)(start + count), NULL, NULL));
            ds41_state_span sa[64], sb[64];
            const uint32_t n = ds41_state_spans(a, a->pos, sa);
            REQUIRE(a->pos == b->pos && n == ds41_state_spans(b, b->pos, sb));
            REQUIRE(!memcmp(&a->history, &b->history, sizeof(a->history)));
            for (uint32_t j = 0; j < n; j++) {
                REQUIRE(sa[j].bytes == sb[j].bytes);
                REQUIRE(!memcmp(ds4_gpu_tensor_contents(sa[j].tensor),
                    ds4_gpu_tensor_contents(sb[j].tensor), (size_t)sa[j].bytes));
            }
            REQUIRE(ds41_graph_logits(a, &engine->model, &engine->weights, control->logits));
            REQUIRE(ds41_graph_logits(b, &engine->model, &engine->weights, candidate->logits));
            REQUIRE(!memcmp(control->logits, candidate->logits, DS4_N_VOCAB * sizeof(float)));
            fprintf(stderr, "V4.1 prefill storage aliases mode=%u count=%u: exact state/logits PASS\n",
                mode, count);
        }
        REQUIRE(ds41_graph_step(a, &engine->model, &engine->weights, tokens.v[a->pos], control->logits));
        REQUIRE(ds41_graph_step(b, &engine->model, &engine->weights, tokens.v[b->pos], candidate->logits));
        REQUIRE(!memcmp(control->logits, candidate->logits, DS4_N_VOCAB * sizeof(float)));
    }
    rc = 0;
done:
    if (ds4_gpu_commands_active()) ds4_gpu_end_commands();
    unsetenv("DS4_METAL_DISABLE_V41_PREFILL_ALIAS");
    for (uint32_t j = 0; j < 3; j++) unsetenv(flags[j]);
    ds4_tokens_free(&tokens); free(prompt);
    ds4_session_free(candidate); ds4_session_free(control); ds4_engine_close(engine);
    return rc;
}

static void note_deferred_progress(void *ud, const char *event, int current, int total) {
    session_progress *p = ud;
    assert(current >= p->current);
    note_progress(ud, event, current, total);
    if (p->cancel_at == -1 && !strcmp(event, "prefill_chunk") &&
        !p->snapshot_session->ds41_graph.valid) p->cancel_at = p->progress;
}

static int check_deferred_decoder(const char *path, const char *prompt_path) {
    ds4_engine *engine = NULL;
    ds4_session *control = NULL, *candidate = NULL;
    ds4_tokens tokens = {0};
    ds4_session_snapshot snap = {0};
    char *prompt = NULL, err[256] = {0};
    size_t bytes;
    int rc = 1;
    ds4_engine_options opt = {.model_path = path, .backend = DS4_BACKEND_METAL,
        .context_size = 57344, .power_percent = 100, .ssd_streaming = true,
        .ssd_streaming_cache_bytes = UINT64_C(48) << 30};
    REQUIRE(imatrix_read_text_file(prompt_path, &prompt, &bytes));
    REQUIRE(ds4_engine_open(&engine, &opt) == 0);
    ds4_encode_chat_prompt(engine, NULL, prompt, DS4_THINK_NONE, &tokens);
    REQUIRE(tokens.len >= 49157);
    REQUIRE(ds4_session_create(&control, engine, opt.context_size) == 0);
    REQUIRE(ds4_session_create(&candidate, engine, opt.context_size) == 0);
    ds41_gpu_graph *a = &control->ds41_graph, *b = &candidate->ds41_graph;
    REQUIRE(a->carry_cap >= 16384u && b->carry_cap >= 16384u);
    a->carry_cap = b->carry_cap = 16384u;
    /* A tiny tail must not force an extra encoder sweep. This callback would
     * interrupt if the decoder were left pending at a completed sweep. */
    session_progress short_tail = {.snapshot_session = candidate, .cancel_at = -1};
    ds4_session_set_progress(candidate, note_deferred_progress, &short_tail);
    ds4_session_set_cancel(candidate, cancel_progress, &short_tail);
    tokens.len = 16385;
    REQUIRE(ds4_session_sync(candidate, &tokens, err, sizeof(err)) == 0);
    REQUIRE(short_tail.cancel_at == -1 && b->valid && candidate->checkpoint_valid);
    ds4_session_set_cancel(candidate, NULL, NULL);
    ds4_session_set_progress(candidate, NULL, NULL);
    ds4_session_invalidate(candidate);
    const int frontiers[] = {24577, 49155};
    for (size_t i = 0; i < sizeof(frontiers) / sizeof(*frontiers); i++) {
        tokens.len = frontiers[i];
        const int count = tokens.len - control->checkpoint.len;
        setenv("DS4_METAL_DISABLE_V41_DEFER_DECODER", "1", 1);
        double start = now_sec();
        REQUIRE(ds4_session_sync(control, &tokens, err, sizeof(err)) == 0);
        const double before = now_sec() - start;
        unsetenv("DS4_METAL_DISABLE_V41_DEFER_DECODER");
        session_progress progress = {.snapshot_session = candidate};
        ds4_session_set_progress(candidate, note_deferred_progress, &progress);
        start = now_sec();
        REQUIRE(ds4_session_sync(candidate, &tokens, err, sizeof(err)) == 0);
        const double after = now_sec() - start;
        ds4_session_set_progress(candidate, NULL, NULL);
        REQUIRE(progress.current == tokens.len && progress.snapshot_checked && progress.snapshot_ready);
        REQUIRE(a->valid && b->valid && a->pos == b->pos);
        REQUIRE(!memcmp(&a->history, &b->history, sizeof(a->history)));
        REQUIRE(!memcmp(control->logits, candidate->logits, DS4_N_VOCAB * sizeof(float)));
        ds41_state_span sa[64], sb[64];
        const uint32_t spans = ds41_state_spans(a, a->pos, sa);
        REQUIRE(spans == ds41_state_spans(b, b->pos, sb));
        for (uint32_t j = 0; j < spans; j++) {
            REQUIRE(sa[j].bytes == sb[j].bytes);
            REQUIRE(!memcmp(ds4_gpu_tensor_contents(sa[j].tensor),
                ds4_gpu_tensor_contents(sb[j].tensor), (size_t)sa[j].bytes));
        }
        REQUIRE(ds4_session_save_snapshot(candidate, &snap, err, sizeof(err)) == 0);
        REQUIRE(ds4_session_load_snapshot(candidate, &snap, err, sizeof(err)) == 0);
        ds4_session_snapshot_free(&snap);
        const int token = tokens.v[tokens.len++];
        REQUIRE(ds4_session_eval(control, token, err, sizeof(err)) == 0);
        REQUIRE(ds4_session_eval(candidate, token, err, sizeof(err)) == 0);
        REQUIRE(!memcmp(control->logits, candidate->logits, DS4_N_VOCAB * sizeof(float)));
        fprintf(stderr, "V4.1 deferred decoder frontier=%d: exact state/logits/snapshot/nextdecode, %.2f -> %.2f t/s\n",
            frontiers[i], count / before, count / after);
    }
    ds4_session_invalidate(candidate);
    session_progress cancelled = {.snapshot_session = candidate, .cancel_at = -1};
    ds4_session_set_progress(candidate, note_deferred_progress, &cancelled);
    ds4_session_set_cancel(candidate, cancel_progress, &cancelled);
    tokens.len = 24577;
    REQUIRE(ds4_session_sync(candidate, &tokens, err, sizeof(err)) == DS4_SESSION_SYNC_INTERRUPTED);
    REQUIRE(cancelled.current == 16384 && cancelled.snapshot_checked && !cancelled.snapshot_ready);
    REQUIRE(!b->valid && !candidate->checkpoint_valid);
    REQUIRE(ds4_session_save_snapshot(candidate, &snap, err, sizeof(err)) != 0);
    ds4_session_set_cancel(candidate, NULL, NULL);
    ds4_session_set_progress(candidate, NULL, NULL);
    ds4_session_invalidate(control);
    tokens.len = 129;
    REQUIRE(ds4_session_sync(control, &tokens, err, sizeof(err)) == 0);
    REQUIRE(ds4_session_sync(candidate, &tokens, err, sizeof(err)) == 0);
    REQUIRE(!memcmp(control->logits, candidate->logits, DS4_N_VOCAB * sizeof(float)));
    puts("V4.1 deferred decoder: exact initial/continued state and interruption recovery PASS");
    rc = 0;
done:
    unsetenv("DS4_METAL_DISABLE_V41_DEFER_DECODER");
    if (rc) fprintf(stderr, "deferred decoder failure: %s\n", err);
    ds4_session_snapshot_free(&snap);
    ds4_tokens_free(&tokens); free(prompt);
    ds4_session_free(candidate); ds4_session_free(control); ds4_engine_close(engine);
    return rc;
}

static int check_sweep_partitions(const char *path, const char *prompt_path) {
    ds4_engine *engine = NULL;
    ds4_session *control = NULL, *candidate = NULL;
    ds4_tokens tokens = {0};
    char *prompt = NULL, err[256] = {0};
    size_t bytes;
    int rc = 1;
    ds4_engine_options opt = {.model_path = path, .backend = DS4_BACKEND_METAL,
        .context_size = 53248, .power_percent = 100, .ssd_streaming = true,
        .ssd_streaming_cache_bytes = UINT64_C(56) << 30};
    REQUIRE(imatrix_read_text_file(prompt_path, &prompt, &bytes));
    REQUIRE(ds4_engine_open(&engine, &opt) == 0);
    ds4_encode_chat_prompt(engine, NULL, prompt, DS4_THINK_NONE, &tokens);
    REQUIRE(tokens.len > 49152);
    REQUIRE(ds4_session_create(&control, engine, opt.context_size) == 0);
    REQUIRE(ds4_session_create(&candidate, engine, opt.context_size) == 0);
    ds41_gpu_graph *a = &control->ds41_graph, *b = &candidate->ds41_graph;
    REQUIRE(a->carry_cap >= 24576 && b->carry_cap >= 24576);
    a->carry_cap = 8192;
    const uint32_t frontiers[] = {24576, 49152};
    for (uint32_t i = 0; i < 2; i++) {
        tokens.len = (int)frontiers[i];
        REQUIRE(ds4_session_sync(control, &tokens, err, sizeof(err)) == 0);
        REQUIRE(ds4_session_sync(candidate, &tokens, err, sizeof(err)) == 0);
        ds41_state_span sa[64], sb[64];
        const uint32_t n = ds41_state_spans(a, a->pos, sa);
        REQUIRE(a->pos == b->pos && a->pos == frontiers[i]);
        REQUIRE(!memcmp(&a->history, &b->history, sizeof(a->history)));
        REQUIRE(n == ds41_state_spans(b, b->pos, sb));
        for (uint32_t j = 0; j < n; j++) {
            REQUIRE(sa[j].bytes == sb[j].bytes);
            REQUIRE(!memcmp(ds4_gpu_tensor_contents(sa[j].tensor),
                ds4_gpu_tensor_contents(sb[j].tensor), (size_t)sa[j].bytes));
        }
        REQUIRE(!memcmp(control->logits, candidate->logits, DS4_N_VOCAB * sizeof(float)));
        fprintf(stderr, "V4.1 sweep partitions at %u: exact state/logits PASS\n", a->pos);
    }
    for (uint32_t i = 0; i < 8; i++) {
        const int token = sample_argmax(control->logits, DS4_N_VOCAB);
        REQUIRE(ds4_session_eval(control, token, err, sizeof(err)) == 0);
        REQUIRE(ds4_session_eval(candidate, token, err, sizeof(err)) == 0);
        REQUIRE(!memcmp(control->logits, candidate->logits, DS4_N_VOCAB * sizeof(float)));
    }
    puts("V4.1 sweep partitions: initial/continued prefill and following decode exact PASS");
    rc = 0;
done:
    if (ds4_gpu_commands_active()) ds4_gpu_end_commands();
    ds4_tokens_free(&tokens);
    free(prompt);
    ds4_session_free(candidate); ds4_session_free(control); ds4_engine_close(engine);
    return rc;
}

static int check_decoder_suffix(const char *path, const char *prompt_path) {
    ds4_engine *engine = NULL;
    ds4_session *control = NULL, *candidate = NULL;
    ds4_tokens tokens = {0};
    char *prompt = NULL;
    size_t prompt_bytes;
    int rc = 1;
    ds4_engine_options opt = {.model_path = path, .backend = DS4_BACKEND_METAL,
        .context_size = 18432, .power_percent = 100, .ssd_streaming = true,
        .ssd_streaming_cache_bytes = UINT64_C(64) << 30};
    /* Hold arithmetic fixed to isolate dependency pruning from GEMM tiling. */
    setenv("DS4_METAL_DISABLE_V41_BATCH_MOE", "1", 1);
    setenv("DS4_METAL_DISABLE_V41_BATCH_ATTN", "1", 1);
    REQUIRE(imatrix_read_text_file(prompt_path, &prompt, &prompt_bytes));
    REQUIRE(ds4_engine_open(&engine, &opt) == 0);
    ds4_encode_chat_prompt(engine, NULL, prompt, DS4_THINK_NONE, &tokens);
    REQUIRE(tokens.len > 16386);
    REQUIRE(ds4_session_create(&control, engine, 18432) == 0);
    REQUIRE(ds4_session_create(&candidate, engine, 18432) == 0);
    ds41_gpu_graph *a = &control->ds41_graph, *b = &candidate->ds41_graph;
    for (uint32_t pass = 0; pass < 2; pass++) {
        const uint32_t start = a->pos, count = 8192;
        setenv("DS4_METAL_DISABLE_V41_DECODER_SUFFIX", "1", 1);
        double t0 = now_sec();
        REQUIRE(ds41_graph_prefill(a, &engine->model, &engine->weights,
            tokens.v + start, count, NULL, NULL, (int)(start + count), NULL, NULL));
        const double reference = now_sec() - t0;
        unsetenv("DS4_METAL_DISABLE_V41_DECODER_SUFFIX");
        t0 = now_sec();
        REQUIRE(ds41_graph_prefill(b, &engine->model, &engine->weights,
            tokens.v + start, count, NULL, NULL, (int)(start + count), NULL, NULL));
        const double candidate_time = now_sec() - t0;
        ds41_state_span sa[64], sb[64];
        const uint32_t n = ds41_state_spans(a, a->pos, sa);
        REQUIRE(a->pos == b->pos && n == ds41_state_spans(b, b->pos, sb));
        for (uint32_t j = 0; j < n; j++) {
            REQUIRE(sa[j].bytes == sb[j].bytes);
            if (memcmp(ds4_gpu_tensor_contents(sa[j].tensor),
                       ds4_gpu_tensor_contents(sb[j].tensor), (size_t)sa[j].bytes)) {
                fprintf(stderr, "decoder suffix mismatch: frontier %u span %u\n", a->pos, j);
                goto done;
            }
        }
        REQUIRE(!memcmp(&a->history, &b->history, sizeof(a->history)));
        REQUIRE(ds41_graph_logits(a, &engine->model, &engine->weights, control->logits));
        REQUIRE(ds41_graph_logits(b, &engine->model, &engine->weights, candidate->logits));
        REQUIRE(!memcmp(control->logits, candidate->logits, DS4_N_VOCAB * sizeof(float)));
        fprintf(stderr, "V4.1 decoder suffix start=%u count=%u: exact state/logits, %.2f -> %.2f t/s\n",
                start, count, count / reference, count / candidate_time);
        REQUIRE(ds41_graph_step(a, &engine->model, &engine->weights, tokens.v[a->pos], control->logits));
        REQUIRE(ds41_graph_step(b, &engine->model, &engine->weights, tokens.v[b->pos], candidate->logits));
        REQUIRE(!memcmp(control->logits, candidate->logits, DS4_N_VOCAB * sizeof(float)));
    }
    puts("V4.1 exact decoder dependency suffix: initial/continued prefill and decode PASS");
    rc = 0;
done:
    unsetenv("DS4_METAL_DISABLE_V41_BATCH_MOE");
    unsetenv("DS4_METAL_DISABLE_V41_BATCH_ATTN");
    unsetenv("DS4_METAL_DISABLE_V41_DECODER_SUFFIX");
    if (ds4_gpu_commands_active()) ds4_gpu_end_commands();
    ds4_tokens_free(&tokens);
    free(prompt);
    ds4_session_free(candidate); ds4_session_free(control); ds4_engine_close(engine);
    return rc;
}

static int check_decoder_publication(const char *path, bool encoder) {
    ds4_engine *engine = NULL;
    ds4_session *a = NULL, *b = NULL;
    int rc = 1;
    ds4_engine_options opt = {.model_path = path, .backend = DS4_BACKEND_METAL,
        .context_size = 19000, .power_percent = 100, .ssd_streaming = true,
        .ssd_streaming_cache_bytes = UINT64_C(32) << 30};
    REQUIRE(ds4_engine_open(&engine, &opt) == 0);
    REQUIRE(ds4_session_create(&a, engine, 19000) == 0);
    REQUIRE(ds4_session_create(&b, engine, 19000) == 0);
    ds41_gpu_graph *old = &a->ds41_graph, *fast = &b->ds41_graph;
    const struct { uint32_t start, count; } shapes[] = {
        {0, 1}, {0, 2}, {127, 1}, {127, 31}, {128, 129}, {129, 513}, {8191, 4096}, {16383, 2048}};
    const uint32_t layers[] = {20, 2, 8, 14};
    for (uint32_t li = 0; li < (encoder ? 4u : 1u); li++) {
    const uint32_t il = layers[li], ratio = ds4_layer_compress_ratio(il);
    const uint32_t owner = il < 8 ? 0 : il < 14 ? 1 : il < 20 ? 2 : 3;
    for (size_t shape = 0; shape < sizeof(shapes) / sizeof(*shapes); shape++) {
        const uint32_t start = shapes[shape].start, count = shapes[shape].count;
        if (count > fast->prefill_cap) continue;
        float *norm = ds4_gpu_tensor_contents(old->batch.norm);
        REQUIRE(norm);
        uint32_t rng = 41;
        for (uint64_t i = 0; i < (uint64_t)count * DS4_N_EMBD; i++) {
            rng = rng * 1664525u + 1013904223u;
            norm[i] = (float)((int)(rng >> 16) - 32768) / 16384.0f;
        }
        memcpy(ds4_gpu_tensor_contents(fast->batch.norm), norm,
               (size_t)count * DS4_N_EMBD * sizeof(float));
        ds4_gpu_tensor *expected[] = {old->compressed[owner], old->index_cache[owner]};
        ds4_gpu_tensor *actual[] = {fast->compressed[owner], fast->index_cache[owner]};
        const uint32_t widths[] = {DS4_N_HEAD_DIM, DS4_N_INDEXER_HEAD_DIM};
        for (uint32_t j = 0; j < 2; j++) {
            memset(ds4_gpu_tensor_contents(expected[j]), 0x3f, ds4_gpu_tensor_bytes(expected[j]));
            memset(ds4_gpu_tensor_contents(actual[j]), 0x3f, ds4_gpu_tensor_bytes(actual[j]));
        }
        for (uint32_t j = 0; j < DS4_N_HEAD_DIM; j++) {
            const float value = (float)((int)(j % 41u) - 20) / 32.0f;
            ((float *)ds4_gpu_tensor_contents(old->previous_kv[owner]))[j] = value;
            ((float *)ds4_gpu_tensor_contents(fast->previous_kv[owner]))[j] = value;
            ((float *)ds4_gpu_tensor_contents(old->previous_score[owner]))[j] = value;
            ((float *)ds4_gpu_tensor_contents(fast->previous_score[owner]))[j] = value;
        }
        ds41_gpu_graph row = *old;
        REQUIRE(ds4_gpu_begin_commands());
        for (uint32_t t = 0; t < count; t++) {
            row.pos = start + t;
            row.norm = old->rows_view[t].norm;
            REQUIRE(ds41_attention_publish(&row, &engine->model, &engine->weights.layer[il], il));
        }
        REQUIRE(ds4_gpu_end_commands());
        REQUIRE(ds4_gpu_begin_commands());
        REQUIRE(ds41_attention_publish_batch(fast, &fast->batch, &engine->model,
                                             &engine->weights.layer[il], il, start, count));
        REQUIRE(ds4_gpu_end_commands());
        for (uint32_t j = 0; j < 2; j++) {
            const float *x = ds4_gpu_tensor_contents(expected[j]);
            const float *y = ds4_gpu_tensor_contents(actual[j]);
            const uint64_t begin = (uint64_t)(start / ratio) * widths[j];
            const uint64_t end = (uint64_t)((start + count) / ratio) * widths[j];
            REQUIRE(!memcmp(x, y, (size_t)begin * sizeof(float)));
            REQUIRE(!memcmp(x + end, y + end,
                (size_t)ds4_gpu_tensor_bytes(expected[j]) - end * sizeof(float)));
            double error = 0, scale = 0;
            uint64_t changed = 0;
            for (uint64_t i = begin; i < end; i++) {
                REQUIRE(isfinite(x[i]) && isfinite(y[i]));
                const double d = (double)y[i] - x[i];
                error += d * d;
                scale += (double)x[i] * x[i];
                changed += x[i] != y[i];
            }
            const double relative = sqrt(error / fmax(scale, 1e-30));
            fprintf(stderr, "V4.1 publication layer=%u start=%u count=%u cache=%u RMS=%.9g changed=%llu/%llu\n",
                il, start, count, j, relative, (unsigned long long)changed,
                (unsigned long long)(end - begin));
            REQUIRE(changed == 0);
        }
        if (ratio == 2u) {
            ds4_gpu_tensor *previous[] = {old->previous_kv[owner], old->previous_score[owner]};
            ds4_gpu_tensor *current[] = {fast->previous_kv[owner], fast->previous_score[owner]};
            for (uint32_t j = 0; j < 2u; j++) {
                const float *x = ds4_gpu_tensor_contents(previous[j]);
                const float *y = ds4_gpu_tensor_contents(current[j]);
                double error = 0, scale = 0;
                for (uint32_t d = 0; d < DS4_N_HEAD_DIM; d++) {
                    REQUIRE(isfinite(x[d]) && isfinite(y[d]));
                    error += ((double)y[d] - x[d]) * ((double)y[d] - x[d]);
                    scale += (double)x[d] * x[d];
                }
                REQUIRE(sqrt(error / fmax(scale, 1e-30)) < 0.002);
                REQUIRE(!memcmp(x, y, DS4_N_HEAD_DIM * sizeof(float)));
            }
        }
    }
    }
    puts("V4.1 batched publication: byte-identical caches and intact guards PASS");
    rc = 0;
done:
    if (ds4_gpu_commands_active()) ds4_gpu_end_commands();
    ds4_session_free(b); ds4_session_free(a); ds4_engine_close(engine);
    return rc;
}

static int check_attention_batches(const char *path, bool batch_index) {
    ds4_engine *engine = NULL;
    ds4_session *a = NULL, *b = NULL;
    int rc = 1;
    setenv("DS4_METAL_DISABLE_V41_BATCH_COMPRESS", "1", 1);
    if (!batch_index) setenv("DS4_METAL_DISABLE_V41_BATCH_INDEX", "1", 1);
    ds4_engine_options opt = {.model_path = path, .backend = DS4_BACKEND_METAL,
        .context_size = 19000, .power_percent = 100, .ssd_streaming = true,
        .ssd_streaming_cache_bytes = UINT64_C(32) * 1024 * 1024 * 1024};
    REQUIRE(ds4_engine_open(&engine, &opt) == 0);
    REQUIRE(ds4_session_create(&a, engine, 19000) == 0);
    REQUIRE(ds4_session_create(&b, engine, 19000) == 0);
    ds41_gpu_graph *old = &a->ds41_graph, *fast = &b->ds41_graph;
    const struct { uint32_t start, count; } shapes[] = {
        {0, 31}, {127, 129}, {129, 513}, {16383, 1025}};
    const uint32_t layers[] = {0, 2, 3, 20, 24, 39};
    for (size_t shape = 0; shape < sizeof(shapes) / sizeof(*shapes); shape++) {
        const uint32_t start = shapes[shape].start, count = shapes[shape].count;
        for (size_t li = 0; li < sizeof(layers) / sizeof(*layers); li++) {
            const uint32_t il = layers[li], ratio = ds4_layer_compress_ratio(il);
            old->pos = fast->pos = start;
            ds41_state_span sa[64], sb[64];
            const uint32_t spans = ds41_state_spans(old, start + count, sa);
            REQUIRE(spans == ds41_state_spans(fast, start + count, sb));
            for (uint32_t j = 0; j < spans; j++) {
                REQUIRE(sa[j].bytes == sb[j].bytes);
                float *x = ds4_gpu_tensor_contents(sa[j].tensor);
                REQUIRE(x);
                for (uint64_t i = 0; i < sa[j].bytes / sizeof(float); i++)
                    x[i] = (float)((int)((i * 13u + j * 7u) % 101u) - 50) / 128.0f;
                memcpy(ds4_gpu_tensor_contents(sb[j].tensor), x, (size_t)sa[j].bytes);
            }
            ds4_gpu_tensor *inputs[] = {old->batch.norm, old->batch.qr, old->batch.q, old->batch.kv};
            ds4_gpu_tensor *copies[] = {fast->batch.norm, fast->batch.qr, fast->batch.q, fast->batch.kv};
            const uint32_t widths[] = {DS4_N_EMBD, DS4_N_LORA_Q,
                                       DS4_N_HEAD * DS4_N_HEAD_DIM, DS4_N_HEAD_DIM};
            for (size_t j = 0; j < sizeof(inputs) / sizeof(*inputs); j++) {
                float *x = ds4_gpu_tensor_contents(inputs[j]);
                REQUIRE(x);
                for (uint64_t i = 0; i < (uint64_t)count * widths[j]; i++)
                    x[i] = (float)((int)((i * 17u + j * 11u) % 97u) - 48) / 64.0f;
                memcpy(ds4_gpu_tensor_contents(copies[j]), x, (size_t)count * widths[j] * sizeof(float));
            }
            int32_t *ids = ds4_gpu_tensor_contents(old->batch.selected_comp);
            REQUIRE(ids);
            for (uint32_t t = 0; t < count; t++) {
                const uint32_t n = ratio ? (start + t + 1u) / ratio : 0;
                const uint32_t k = n < DS4_N_INDEXER_TOP_K ? n : DS4_N_INDEXER_TOP_K;
                for (uint32_t j = 0; j < DS4_N_INDEXER_TOP_K; j++)
                    ids[t * DS4_N_INDEXER_TOP_K + j] = j < k ? (int32_t)(k - 1u - j) : -1;
            }
            memcpy(ds4_gpu_tensor_contents(fast->batch.selected_comp), ids,
                   (size_t)count * DS4_N_INDEXER_TOP_K * sizeof(*ids));
            memset(ds4_gpu_tensor_contents(old->batch.block_mask), 0,
                   (size_t)ds4_gpu_tensor_bytes(old->batch.block_mask));
            memset(ds4_gpu_tensor_contents(fast->batch.block_mask), 0,
                   (size_t)ds4_gpu_tensor_bytes(fast->batch.block_mask));
            ds41_gpu_graph row = *old;
            REQUIRE(ds4_gpu_begin_commands());
            for (uint32_t t = 0; t < count; t++) {
                row.pos = start + t;
                row.norm = old->rows_view[t].norm;
                row.qr = old->rows_view[t].qr;
                row.q = old->rows_view[t].q;
                row.kv = old->rows_view[t].kv;
                row.heads = old->rows_view[t].heads;
                row.selected_comp = old->rows_view[t].selected_comp;
                row.block_mask = old->rows_view[t].block_mask;
                REQUIRE(ds41_attention(&row, &engine->model, &engine->weights.layer[il], il, true));
            }
            REQUIRE(ds4_gpu_end_commands());
            REQUIRE(ds4_gpu_begin_commands());
            REQUIRE(ds41_attention_batch(fast, &engine->model, &engine->weights.layer[il], il, count));
            REQUIRE(ds4_gpu_end_commands());
            for (uint32_t j = 0; j < spans; j++) {
                REQUIRE(memcmp(ds4_gpu_tensor_contents(sa[j].tensor),
                    ds4_gpu_tensor_contents(sb[j].tensor), (size_t)sa[j].bytes) == 0);
            }
            REQUIRE(memcmp(ds4_gpu_tensor_contents(old->batch.selected_comp),
                ds4_gpu_tensor_contents(fast->batch.selected_comp),
                (size_t)count * DS4_N_INDEXER_TOP_K * sizeof(int32_t)) == 0);
            const float *expected = ds4_gpu_tensor_contents(old->batch.heads);
            const float *actual = ds4_gpu_tensor_contents(fast->batch.heads);
            double error = 0, norm = 0, worst = 0;
            for (uint64_t i = 0; i < (uint64_t)count * DS4_N_HEAD * DS4_N_HEAD_DIM; i++) {
                REQUIRE(isfinite(actual[i]) && isfinite(expected[i]));
                const double d = (double)actual[i] - expected[i];
                error += d * d; norm += (double)expected[i] * expected[i];
                worst = fmax(worst, fabs(d));
            }
            const double relative = sqrt(error / fmax(norm, 1e-30));
            fprintf(stderr, "V4.1 batched attention layer=%u start=%u count=%u RMS=%.9g max=%.9g\n",
                    il, start, count, relative, worst);
            REQUIRE(relative < 0.002);
        }
    }
    puts("V4.1 batched causal attention: bounded output error and exact KV/index state: PASS");
    rc = 0;
done:
    unsetenv("DS4_METAL_DISABLE_V41_BATCH_INDEX");
    unsetenv("DS4_METAL_DISABLE_V41_BATCH_COMPRESS");
    if (ds4_gpu_commands_active()) ds4_gpu_end_commands();
    ds4_session_free(b); ds4_session_free(a); ds4_engine_close(engine);
    return rc;
}

/* Map only one real layer at a time. A null transport tests expert ownership
 * without network traffic; add both partials independently on the host. */
static int check_partitions(const char *path) {
    ds4_model model = {.fd = -1};
    ds4_weights weights = {0};
    ds41_gpu_graph g = {.table = {{.fd = -1}, {.fd = -1}}};
    ds4_model_map_span_vec spans = {0};
    uint64_t *offsets = NULL, *sizes = NULL;
    float *expected = NULL, *actual = NULL, *first = NULL;
    int rc = 1;
    model_open(&model, path, true, false);
    config_validate_model(&model);
    REQUIRE(DS4_MODEL_FAMILY == DS4_MODEL_FAMILY_DEEPSEEK41);
    weights_bind(&weights, &model, false, 0, UINT32_MAX, true, false);
    ds4_gpu_model_residency_skip(1);
    REQUIRE(ds4_gpu_init());
    ds4_gpu_set_ssd_streaming(true);
    ds4_gpu_set_streaming_expert_cache_budget(16);
    setenv("DS4_TP_NO_KEEPALIVE", "1", 1);
    REQUIRE(ds4_gpu_set_model_fd(model.fd));
    REQUIRE(ds41_graph_alloc(&g, &model, &weights, path, 1, true));
    const size_t bytes = DS4_N_EMBD * sizeof(float);
    const size_t head_bytes = DS4_N_HEAD * DS4_N_HEAD_DIM * sizeof(float);
    expected = malloc(head_bytes); actual = malloc(head_bytes); first = malloc(head_bytes);
    REQUIRE(expected && actual && first);
    float *x = ds4_gpu_tensor_contents(g.norm), *low = ds4_gpu_tensor_contents(g.low);
    REQUIRE(x && low);
    for (uint32_t i = 0; i < DS4_N_EMBD; i++) x[i] = (float)((int)(i * 37u % 257u) - 128) / 64;
    for (uint32_t i = 0; i < 8192; i++) low[i] = (float)((int)(i * 53u % 251u) - 125) / 64;
    REQUIRE(ds41_bf16(g.norm, DS4_N_EMBD) && ds41_bf16(g.low, 8192));
    float *qr = ds4_gpu_tensor_contents(g.qr), *heads = ds4_gpu_tensor_contents(g.heads);
    REQUIRE(qr && heads);
    for (uint32_t i = 0; i < DS4_N_LORA_Q; i++) qr[i] = x[i];
    for (uint32_t i = 0; i < DS4_N_HEAD * DS4_N_HEAD_DIM; i++) heads[i] = x[i % DS4_N_EMBD];
    const uint32_t layers[] = {0, 1, 14, 20, 39};
    const int32_t routes[][6] = {{0, 1, 63, 125, 190, 191},
        {192, 193, 255, 300, 382, 383}, {0, 383, 191, 192, 73, 267}};
    const float route_weights[] = {0.1f, 0.15f, 0.2f, 0.25f, 0.35f, 0.45f};
    REQUIRE(ds4_gpu_tensor_write(g.route_weights, 0, route_weights, sizeof(route_weights)));
    for (size_t i = 0; i < sizeof(layers) / sizeof(*layers); i++) {
        const uint32_t il = layers[i];
        const ds4_layer_weights *l = &weights.layer[il];
        free(spans.v); spans = (ds4_model_map_span_vec){0};
        model_map_span_vec_include_one(&spans, l->attn_output_b);
        model_map_span_vec_include_one(&spans, l->attn_output_a);
        model_map_span_vec_include_one(&spans, l->attn_q_b);
        model_map_span_vec_include_one(&spans, l->ffn_gate_exps);
        model_map_span_vec_include_one(&spans, l->ffn_up_exps);
        model_map_span_vec_include_one(&spans, l->ffn_down_exps);
        offsets = realloc(offsets, spans.len * sizeof(*offsets));
        sizes = realloc(sizes, spans.len * sizeof(*sizes));
        REQUIRE(offsets && sizes);
        for (uint32_t j = 0; j < spans.len; j++) {
            offsets[j] = spans.v[j].off;
            sizes[j] = spans.v[j].end - spans.v[j].off;
            REQUIRE(spans.v[j].end <= model.size);
        }
        REQUIRE(ds4_gpu_set_model_map_spans(model.map, model.size, offsets, sizes,
                                           spans.len, spans.max_tensor_bytes));
        REQUIRE(ds41_matmul(g.block, &model, l->attn_output_b, g.low, false));
        REQUIRE(ds4_gpu_tensor_read(g.block, 0, expected, bytes));
        for (uint32_t rank = 0; rank < 2; rank++) {
            REQUIRE(metal_graph_matmul_dense_quant_kslice(g.block, &model,
                l->attn_output_b, 8192, rank * 4096u, 4096, DS4_N_EMBD,
                g.low, rank * 4096u));
            REQUIRE(ds4_gpu_tensor_read(g.block, 0, rank ? actual : first, bytes));
        }
        for (uint32_t j = 0; j < DS4_N_EMBD; j++) actual[j] += first[j];
        REQUIRE(partition_close("attention K split", il, expected, actual, DS4_N_EMBD));
        REQUIRE(ds41_matmul(g.q, &model, l->attn_q_b, g.qr, true));
        REQUIRE(ds4_gpu_tensor_read(g.q, 0, expected, head_bytes));
        for (uint32_t rank = 0; rank < 2; rank++) {
            REQUIRE(ds41_matmul_rows(g.q, &model, l->attn_q_b, g.qr, rank * 16384u, 16384));
            REQUIRE(ds4_gpu_tensor_read(g.q, 0, actual + rank * 16384u, head_bytes / 2));
        }
        REQUIRE(partition_close("query row split", il, expected, actual, 32768));
        REQUIRE(ds4_gpu_attention_output_low_q8_tensor(g.low, model.map, model.size,
            l->attn_output_a->abs_offset, 4096, 1024, 8, g.heads));
        REQUIRE(ds4_gpu_tensor_read(g.low, 0, expected, 8192u * sizeof(float)));
        for (uint32_t rank = 0; rank < 2; rank++) {
            ds4_gpu_tensor *half = ds4_gpu_tensor_view(g.heads, rank * head_bytes / 2, head_bytes / 2);
            uint64_t row;
            REQUIRE(half && tensor_nbytes(l->attn_output_a->type, 4096, &row));
            const bool ok = ds4_gpu_attention_output_low_q8_tensor(g.low, model.map, model.size,
                l->attn_output_a->abs_offset + rank * 4096u * row, 4096, 1024, 4, half);
            ds4_gpu_tensor_free(half);
            REQUIRE(ok && ds4_gpu_tensor_read(g.low, 0, actual + rank * 4096u, 4096u * sizeof(float)));
        }
        REQUIRE(partition_close("output group split", il, expected, actual, 8192));
        for (uint32_t j = 0; j < 8192; j++) low[j] = (float)((int)(j * 53u % 251u) - 125) / 64;
        REQUIRE(ds41_bf16(g.low, 8192));
        uint64_t gate_row, down_row;
        REQUIRE(tensor_nbytes(l->ffn_gate_exps->type, DS4_N_EMBD, &gate_row));
        REQUIRE(tensor_nbytes(l->ffn_down_exps->type, DS4_N_FF_EXP, &down_row));
        ds4_gpu_set_streaming_expert_cache_expert_bytes(
            2 * gate_row * DS4_N_FF_EXP + down_row * DS4_N_EMBD);
        for (size_t r = 0; r < sizeof(routes) / sizeof(*routes); r++) {
            REQUIRE(ds4_gpu_tensor_write(g.selected, 0, routes[r], sizeof(routes[r])));
            /* resident, SSD miss, SSD hit, then each resident TP rank */
            for (int mode = 0; mode < 5; mode++) {
                if (mode >= 3) REQUIRE(ds4_gpu_tp_init(mode - 3, NULL, 0, 0, 0, NULL, NULL));
                REQUIRE(ds4_gpu_begin_commands());
                REQUIRE(ds4_gpu_routed_moe_one_tensor(g.routed, g.gate, g.up, g.mid, g.experts,
                    model.map, model.size, l->ffn_gate_exps->abs_offset,
                    l->ffn_up_exps->abs_offset, l->ffn_down_exps->abs_offset,
                    l->ffn_gate_exps->type, l->ffn_down_exps->type,
                    gate_row * DS4_N_FF_EXP, gate_row, down_row * DS4_N_EMBD, down_row,
                    DS4_N_EMBD, DS4_N_FF_EXP, DS4_N_EMBD, g.selected, g.route_weights,
                    DS4_N_EXPERT, DS4_N_EXPERT_USED, DS4_SWIGLU_CLAMP_EXP,
                    g.norm, NULL, il, mode == 0 || mode >= 3));
                REQUIRE(ds4_gpu_end_commands());
                REQUIRE(ds4_gpu_tensor_read(g.routed, 0,
                    mode == 0 ? expected : mode == 3 ? first : actual, bytes));
                if (mode >= 3) ds4_gpu_tp_shutdown();
                if (mode == 4) {
                    for (uint32_t j = 0; j < DS4_N_EMBD; j++) actual[j] += first[j];
                    REQUIRE(partition_close("expert split", il, expected, actual, DS4_N_EMBD));
                } else if (mode == 1 || mode == 2) {
                    REQUIRE(partition_close("SSD/resident", il, expected, actual, DS4_N_EMBD));
                }
            }
        }
    }
    puts("V4.1 real-weight attention and expert partitions: PASS");
    rc = 0;
done:
    if (ds4_gpu_commands_active()) ds4_gpu_end_commands();
    ds4_gpu_tp_shutdown();
    ds41_graph_free(&g); ds4_gpu_cleanup(); model_close(&model);
    free(spans.v); free(offsets); free(sizes);
    free(expected); free(actual); free(first);
    return rc;
}

static int check_session_accounting(const char *path) {
    ds4_engine *engine = NULL;
    ds4_session *small = NULL, *wide = NULL;
    int rc = 1;
    ds4_engine_options opt = {.model_path = path, .backend = DS4_BACKEND_METAL,
        .context_size = 16384, .power_percent = 100, .ssd_streaming = true,
        .ssd_streaming_cache_bytes = UINT64_C(16) << 30};
    REQUIRE(ds4_engine_open(&engine, &opt) == 0);
    setenv("DS4_METAL_DISABLE_V41_WIDE_CHUNK", "1", 1);
    REQUIRE(ds4_session_create(&small, engine, opt.context_size) == 0);
    const uint64_t first = engine->ds41_session_bytes;
    REQUIRE(first);
    unsetenv("DS4_METAL_DISABLE_V41_WIDE_CHUNK");
    REQUIRE(ds4_session_create(&wide, engine, opt.context_size) == 0);
    const uint64_t second = engine->ds41_session_bytes - first;
    REQUIRE(second > first);
    /* Debug settings and logical sweep limits may change after allocation. */
    small->ds41_graph.carry_cap = 8192;
    ds4_session_free(small); small = NULL;
    REQUIRE(engine->ds41_session_bytes == second);
    setenv("DS4_METAL_DISABLE_V41_WIDE_CHUNK", "1", 1);
    ds4_session_free(wide); wide = NULL;
    REQUIRE(engine->ds41_session_bytes == 0);
    REQUIRE(ds4_session_create(&small, engine, opt.context_size) == 0);
    REQUIRE(engine->ds41_session_bytes == first);
    ds4_session_free(small); small = NULL;
    REQUIRE(engine->ds41_session_bytes == 0);
    puts("V4.1 session accounting retains allocation sizes across debug settings: PASS");
    rc = 0;
done:
    unsetenv("DS4_METAL_DISABLE_V41_WIDE_CHUNK");
    ds4_session_free(small); ds4_session_free(wide); ds4_engine_close(engine);
    return rc;
}

static int check_memory_plan(const char *path) {
    const uint64_t gib = UINT64_C(1073741824);
    ds4_engine e = {.model = {.fd = -1}};
    const uint64_t saved_shard = g_tp_shard_model_bytes;
    int rc = 1;
    model_open(&e.model, path, true, false);
    config_validate_model(&e.model);
    REQUIRE(DS4_MODEL_FAMILY == DS4_MODEL_FAMILY_DEEPSEEK41);
    REQUIRE(ds41_prefill_limit(8191) == 2048u);
    REQUIRE(ds41_prefill_limit(8192) == 4096u);
    REQUIRE(ds41_prefill_limit(16383) == 4096u);
    REQUIRE(ds41_prefill_limit(16384) == 8192u);
    ds4_engine_options opt = {0};
    REQUIRE(!engine_warm_full_model(&opt));
    opt.warm_weights = true;
    REQUIRE(engine_warm_full_model(&opt));
    opt.inspect_only = true;
    REQUIRE(!engine_warm_full_model(&opt));
    opt.inspect_only = false;
    opt.ssd_streaming = true;
    REQUIRE(!engine_warm_full_model(&opt));
    opt.ssd_streaming = false;
    opt.cuda_tensor_parallel = true;
    REQUIRE(!engine_warm_full_model(&opt));
    opt.cuda_tensor_parallel = false;
    opt.tp.role = DS4_TP_LEADER;
    REQUIRE(!engine_warm_full_model(&opt));
    opt.tp.role = DS4_TP_WORKER;
    REQUIRE(!engine_warm_full_model(&opt));
    opt.tp.role = DS4_TP_NONE;
    opt.distributed.role = DS4_DISTRIBUTED_WORKER;
    REQUIRE(!engine_warm_full_model(&opt));
    opt.distributed.role = DS4_DISTRIBUTED_COORDINATOR;
    REQUIRE(!engine_warm_full_model(&opt));
    weights_bind(&e.weights, &e.model, false, 0, UINT32_MAX, true, false);
    REQUIRE(e.model.size < e.model.file_size);
    const bool q4 = e.weights.layer[0].ffn_gate_exps->type == DS4_TENSOR_Q4_K;
    REQUIRE(q4 || e.weights.layer[0].ffn_gate_exps->type == DS4_TENSOR_IQ2_XXS);
    g_tp_shard_model_bytes = 0;
    uint64_t graph = ds41_graph_bytes(1048576);
    REQUIRE(!ds41_memory_admit_for_host(&e, graph, true, 128 * gib, 110 * gib));
    REQUIRE(ds41_memory_admit_for_host(&e, graph, true, 256 * gib, 224 * gib) == !q4);
    REQUIRE(ds41_memory_admit_for_host(&e, graph, true, 512 * gib, 448 * gib));
    REQUIRE(!ds41_memory_admit_for_host(&e, graph, true, 512 * gib, 110 * gib));
    REQUIRE(!ds41_memory_admit_for_host(&e, graph, true, 0, 224 * gib));
    REQUIRE(!ds41_memory_admit_for_host(&e, graph, true, 256 * gib, 0));
    REQUIRE(!ds41_memory_admit_for_host(&e, UINT64_MAX, true, 512 * gib, 448 * gib));
    /* A generous real-shard estimate. No weight pages or GPU buffers are
     * allocated by this test, even for the simulated 512 GiB machine. */
    g_tp_shard_model_bytes = (q4 ? 153 : 81) * gib;
    REQUIRE(ds41_memory_admit_for_host(&e, graph, true, 128 * gib, 110 * gib) == !q4);
    REQUIRE(ds41_memory_admit_for_host(&e, graph, true, 256 * gib, 224 * gib));
    g_tp_shard_model_bytes = 0;
    e.ssd_streaming = true;
    e.ssd_streaming_prefill_headroom_bytes = (q4 ? 16 : 8) * gib;
    e.ssd_streaming_cache_experts = DS4_N_LAYER * DS4_N_EXPERT;
    uint64_t expert = 0, fixed = 0;
    REQUIRE(ds4_streaming_routed_expert_bytes(&e.weights, &expert) && expert);
    REQUIRE(weights_streaming_non_routed_bytes(&e.weights, &fixed));
    REQUIRE(ds41_memory_admit_for_host(&e, graph, true, 128 * gib, 110 * gib));
    uint64_t planned = fixed + graph + 2 * gib + e.ssd_streaming_cache_bytes;
    REQUIRE(e.ssd_streaming_cache_experts > 0 && planned <= 110 * gib);
    REQUIRE(planned + expert > 110 * gib);
    const uint32_t fitted = e.ssd_streaming_cache_experts;
    REQUIRE(ds41_memory_admit_for_host(&e, graph, false, 128 * gib, 110 * gib));
    REQUIRE(!ds41_memory_admit_for_host(&e, graph + expert, false, 128 * gib, 110 * gib));
    REQUIRE(e.ssd_streaming_cache_experts == fitted);
    REQUIRE(!ds41_memory_admit_for_host(&e, graph, true, 8 * gib, 6 * gib));
    puts("V4.1 disk-only Engram, 128/256/512 GiB residency, TP and SSD admission: PASS");
    rc = 0;
done:
    g_tp_shard_model_bytes = saved_shard;
    model_close(&e.model);
    return rc;
}

static int check_vision_routing(void) {
    enum { ROWS = 33, EXPERTS = 384, USED = 6 };
    ds41_gpu_graph *g = calloc(1, sizeof(*g));
    float *bias = NULL, *logits = NULL;
    int rc = 1;
    REQUIRE(g);
    g_ds4_shape = DS4_SHAPE_FLASH41;
    g->pos = 100;
    ds4_vision_span spans[] = {
        {.token_start = 98, .embedding.token_count = 5},
        {.token_start = 109, .embedding.token_count = 9},
        {.token_start = 130, .embedding.token_count = 8}
    };
    g->images = spans;
    g->image_count = sizeof(spans) / sizeof(*spans);
    for (uint32_t start = 90; start < 141; start++) {
        uint8_t mask[ROWS];
        ds41_text_mask(g, start, ROWS, mask);
        for (uint32_t i = 0; i < ROWS; i++) {
            const uint32_t pos = start + i;
            const bool image = (pos >= 98 && pos < 103) ||
                (pos >= 109 && pos < 118) || (pos >= 130 && pos < 138);
            REQUIRE(mask[i] == !image);
            REQUIRE((ds41_image_at(g, pos) != NULL) == image);
        }
    }
    REQUIRE(posix_memalign((void **)&bias, 16384, 16384) == 0);
    memset(bias, 0, 16384);
    logits = calloc(ROWS * EXPERTS, sizeof(float));
    REQUIRE(logits);
    for (int i = 0; i < EXPERTS; i++) {
        bias[i] = i >= 5 && i < 11 ? 10.0f : 0.0f;
        bias[EXPERTS + i] = i >= 310 && i < 316 ? 10.0f : 0.0f;
        for (int row = 0; row < ROWS; row++)
            logits[row * EXPERTS + i] = (float)i / 1000.0f + (float)row / 100.0f;
    }
    ds4_model model = {.map = (const uint8_t *)bias, .size = 16384};
    ds4_tensor text_bias = {.abs_offset = 0}, image_bias = {.abs_offset = EXPERTS * 4};
    ds4_layer_weights layer = {.ffn_exp_probs_b = &text_bias, .ffn_exp_probs_vl = &image_bias};
    REQUIRE(ds4_gpu_init() && ds4_gpu_set_model_map_range(bias, 16384, 0, 16384, 16384));
    g->prefill_tokens = ds4_gpu_tensor_alloc(ROWS * 4u);
    g->batch.route_logits = ds4_gpu_tensor_alloc(ROWS * EXPERTS * 4u);
    g->batch.route_probs = ds4_gpu_tensor_alloc(ROWS * EXPERTS * 4u);
    g->batch.route_weights = ds4_gpu_tensor_alloc(ROWS * USED * 4u);
    g->batch.selected = ds4_gpu_tensor_alloc(ROWS * USED * 4u);
    REQUIRE(g->prefill_tokens && g->batch.route_logits && g->batch.route_probs &&
            g->batch.route_weights && g->batch.selected);
    int tokens[ROWS] = {0};
    REQUIRE(ds4_gpu_tensor_write(g->prefill_tokens, 0, tokens, sizeof(tokens)));
    REQUIRE(ds4_gpu_tensor_write(g->batch.route_logits, 0, logits, ROWS * EXPERTS * 4u));
    const uint32_t counts[] = {1, 2, 3, 9, 17, 32, ROWS};
    for (unsigned mode = 0; mode < 2; mode++) {
        g->image_count = mode ? 0 : sizeof(spans) / sizeof(*spans);
        for (unsigned shape = 0; shape < sizeof(counts) / sizeof(*counts); shape++) {
            const uint32_t count = counts[shape];
            REQUIRE(ds4_gpu_begin_commands());
            REQUIRE(ds41_route_batch(g, &model, &layer, count));
            REQUIRE(ds4_gpu_end_commands());
            const int32_t *selected = ds4_gpu_tensor_contents(g->batch.selected);
            const float *weights = ds4_gpu_tensor_contents(g->batch.route_weights);
            REQUIRE(selected && weights);
            for (uint32_t row = 0; row < count; row++) {
                const int last = ds41_image_at(g, g->pos + row) ? 315 : 10;
                double sum = 0;
                for (int k = 0; k < USED; k++)
                    sum += sqrt(log1p(exp((double)logits[row * EXPERTS + last - k])));
                for (int k = 0; k < USED; k++) {
                    const int id = last - k;
                    REQUIRE(selected[row * USED + k] == id);
                    const double expected = DS4_EXPERT_WEIGHT_SCALE *
                        sqrt(log1p(exp((double)logits[row * EXPERTS + id]))) / sum;
                    REQUIRE(fabs(weights[row * USED + k] - expected) < 2e-6);
                }
            }
        }
    }
    puts("V4.1 image masks and 384-expert text/visual routing against double reference: PASS");
    rc = 0;
done:
    if (ds4_gpu_commands_active()) (void)ds4_gpu_end_commands();
    if (g) {
        ds4_gpu_tensor_free(g->batch.selected);
        ds4_gpu_tensor_free(g->batch.route_weights);
        ds4_gpu_tensor_free(g->batch.route_probs);
        ds4_gpu_tensor_free(g->batch.route_logits);
        ds4_gpu_tensor_free(g->prefill_tokens);
    }
    ds4_gpu_cleanup();
    free(logits);
    free(bias);
    free(g);
    return rc;
}

static int check_vision_encoder(const char *path, const char *image, const char *output) {
    ds4_engine *e = calloc(1, sizeof(*e));
    if (!e) return 1;
    e->vision_model.fd = -1;
    e->backend = DS4_BACKEND_METAL;
    e->vision_kind = DS4_VISION_DEEPSEEK4;
    e->vision_ready = true;
    g_ds4_shape = DS4_SHAPE_FLASH41;
    model_open(&e->vision_model, path, true, false);
    deepseek4_vision_weights_bind(&e->deepseek4_vision_weights, &e->vision_model);
    ds4_vision_embedding first = {0}, second = {0};
    FILE *fp = NULL;
    char error[256] = {0};
    int rc = 1;
    REQUIRE(ds4_engine_vision_encode_file(e, image, &first, error, sizeof(error)));
    REQUIRE(ds4_engine_vision_encode_file(e, image, &second, error, sizeof(error)));
    REQUIRE(first.token_count > 0 && first.token_count == second.token_count);
    const size_t count = (size_t)first.token_count * DS4_N_EMBD;
    for (size_t i = 0; i < count; i++) REQUIRE(isfinite(first.data[i]));
    REQUIRE(!memcmp(first.data, second.data, count * sizeof(float)));
    fp = fopen(output, "wb");
    REQUIRE(fp && fwrite(first.data, sizeof(float), count, fp) == count);
    const int closed = fclose(fp);
    fp = NULL;
    REQUIRE(closed == 0);
    printf("V4.1 vision: %ux%u grid, %u rows of %u, finite and repeatable\n",
           first.grid_width, first.grid_height, first.token_count, DS4_N_EMBD);
    rc = 0;
done:
    if (rc && error[0]) fprintf(stderr, "%s\n", error);
    if (fp) fclose(fp);
    ds4_vision_embedding_free(&first);
    ds4_vision_embedding_free(&second);
    ds4_gpu_cleanup();
    model_close(&e->vision_model);
    free(e);
    return rc;
}

int main(int argc, char **argv) {
    if (argc == 2 && !strcmp(argv[1], "--batch-admission"))
        return check_batch_admission();
    if (argc == 3 && !strcmp(argv[2], "--batch-head"))
        return check_batch_head(argv[1]);
    if (argc == 2 && !strcmp(argv[1], "--vision-routing"))
        return check_vision_routing();
    if (argc == 5 && !strcmp(argv[1], "--vision-encoder"))
        return check_vision_encoder(argv[2], argv[3], argv[4]);
    if (argc == 2 && !strcmp(argv[1], "--engram-ready"))
        return check_engram_prefetch_ready();
    /* Isolate scheduling from matrix-kernel rounding in the byte-exact
     * row/layer/residency comparisons. --long-sessions and --thread-sessions
     * exercise the default matrix path and require exact snapshot restoration;
     * its inference quality is checked against the official batch vectors. */
    if (argc == 4 && (!strcmp(argv[2], "--thread-prefill") ||
        !strcmp(argv[2], "--encoder-parity") || !strcmp(argv[2], "--prefill-parity") ||
        !strcmp(argv[2], "--encoder-long-parity"))) {
        assert(setenv("DS4_METAL_DISABLE_V41_BATCH_MOE", "1", 1) == 0);
        assert(setenv("DS4_METAL_DISABLE_V41_BATCH_ATTN", "1", 1) == 0);
    }
    if (argc == 4 && !strcmp(argv[2], "--thread-sessions"))
        return check_thread_prefill(argv[1], argv[3], true);
    if (argc == 4 && !strcmp(argv[2], "--thread-prefill"))
        return check_thread_prefill(argv[1], argv[3], false);
    if (argc == 4 && !strcmp(argv[2], "--encoder-parity"))
        return check_encoder(argv[1], argv[3]);
    if (argc == 4 && !strcmp(argv[2], "--prefill-parity"))
        return check_prefill(argv[1], argv[3], false, false);
    if (argc == 4 && !strcmp(argv[2], "--encoder-long-parity"))
        return check_prefill(argv[1], argv[3], true, false);
    if (argc == 4 && !strcmp(argv[2], "--encoder-cancel"))
        return check_prefill(argv[1], argv[3], true, true);
    if (argc == 4 && !strcmp(argv[2], "--long-sessions"))
        return check_long_sessions(argv[1], argv[3]);
    if (argc == 3 && !strcmp(argv[2], "--session-accounting"))
        return check_session_accounting(argv[1]);
    if (argc == 3 && !strcmp(argv[2], "--memory-plan"))
        return check_memory_plan(argv[1]);
    if (argc == 3 && !strcmp(argv[2], "--partitions"))
        return check_partitions(argv[1]);
    if (argc == 3 && !strcmp(argv[2], "--attention-layouts"))
        return check_attention_layouts(argv[1]);
    if (argc == 3 && !strcmp(argv[2], "--attention-batch"))
        return check_attention_batches(argv[1], false);
    if (argc == 3 && !strcmp(argv[2], "--attention-index-batch"))
        return check_attention_batches(argv[1], true);
    if (argc == 3 && !strcmp(argv[2], "--decoder-publication"))
        return check_decoder_publication(argv[1], false);
    if (argc == 3 && !strcmp(argv[2], "--compressor-batch"))
        return check_decoder_publication(argv[1], true);
    if (argc == 4 && !strcmp(argv[2], "--wide-prefill"))
        return check_wide_prefill(argv[1], argv[3], false, false, NULL);
    if (argc == 4 && !strcmp(argv[2], "--wide-cancel"))
        return check_wide_prefill(argv[1], argv[3], true, false, NULL);
    if (argc == 4 && !strcmp(argv[2], "--decoder-cancel"))
        return check_wide_prefill(argv[1], argv[3], true, true, NULL);
    if (argc == 4 && !strcmp(argv[2], "--engram-prefetch"))
        return check_wide_prefill(argv[1], argv[3], false, false, "DS4_METAL_DISABLE_V41_ENGRAM_PREFETCH");
    if (argc == 4 && !strcmp(argv[2], "--engram-pipeline"))
        return check_wide_prefill(argv[1], argv[3], false, false, "DS4_METAL_DISABLE_V41_ENGRAM_PIPELINE");
    if (argc == 4 && !strcmp(argv[2], "--compressor-prefill"))
        return check_wide_prefill(argv[1], argv[3], false, false, "DS4_METAL_DISABLE_V41_BATCH_COMPRESS");
    if (argc == 4 && !strcmp(argv[2], "--packed-index-prefill"))
        return check_wide_prefill(argv[1], argv[3], false, false, "DS4_METAL_DISABLE_V41_PACKED_INDEX");
    if (argc == 4 && !strcmp(argv[2], "--compact-carry-prefill"))
        return check_wide_prefill(argv[1], argv[3], false, false, "DS4_METAL_DISABLE_V41_COMPACT_CARRY");
    if (argc == 4 && !strcmp(argv[2], "--bf16-prefill"))
        return check_wide_prefill(argv[1], argv[3], false, false, "DS4_METAL_DISABLE_V41_LINEAR_BF16");
    if (argc == 4 && !strcmp(argv[2], "--hc-scaled-prefill"))
        return check_wide_prefill(argv[1], argv[3], false, false, "DS4_METAL_DISABLE_HC_RMS_SCALE_PROJ");
    if (argc == 4 && !strcmp(argv[2], "--topk-prefix-prefill"))
        return check_wide_prefill(argv[1], argv[3], false, false, "DS4_METAL_DISABLE_V41_TOPK_PREFIX");
    if (argc == 4 && !strcmp(argv[2], "--prefill-alias"))
        return check_wide_prefill(argv[1], argv[3], false, false, "DS4_METAL_DISABLE_V41_PREFILL_ALIAS");
    if (argc == 4 && !strcmp(argv[2], "--wide-expert-prefill"))
        return check_wide_prefill(argv[1], argv[3], false, false, "DS4_METAL_DISABLE_V41_PACKED_M32N128");
    if (argc == 4 && !strcmp(argv[2], "--prefill-alias-fallback"))
        return check_prefill_alias_fallback(argv[1], argv[3]);
    if (argc == 4 && !strcmp(argv[2], "--chunk-prefill"))
        return check_wide_prefill(argv[1], argv[3], false, false, "DS4_METAL_DISABLE_V41_WIDE_CHUNK");
    if (argc == 4 && !strcmp(argv[2], "--decoder-suffix"))
        return check_decoder_suffix(argv[1], argv[3]);
    if (argc == 4 && !strcmp(argv[2], "--sweep-partitions"))
        return check_sweep_partitions(argv[1], argv[3]);
    if (argc == 4 && !strcmp(argv[2], "--deferred-decoder"))
        return check_deferred_decoder(argv[1], argv[3]);
    if (argc == 6 && !strcmp(argv[1], "--chat-fixture"))
        return check_chat(argv[2], argv[3], argv[4], argv[5]);
    if (argc == 3 && strcmp(argv[1], "--rope-reference") == 0)
        return check_rope_reference(argv[2]);
    if (argc == 3 && strcmp(argv[2], "--session-fixture") == 0)
        return check_sessions(argv[1]);
    if (argc < 3 || argc > 4 ||
        (!strncmp(argv[2], "--", 2) && strcmp(argv[2], "--zero-fixture"))) {
        fprintf(stderr, "usage: %s MODEL (--zero-fixture | --session-fixture | "
                        "--long-sessions PROMPT_FILE | --prefill-parity PROMPT_FILE | "
                        "--thread-prefill PROMPT_FILE | --thread-sessions PROMPT_FILE | "
                        "--encoder-parity PROMPT_FILE | --encoder-long-parity PROMPT_FILE | "
                        "--encoder-cancel PROMPT_FILE | "
                        "--partitions | --attention-layouts | --attention-batch | --attention-index-batch | "
                        "--decoder-publication | --wide-prefill PROMPT_FILE | "
                        "--wide-cancel PROMPT_FILE | --decoder-cancel PROMPT_FILE | "
                        "--engram-prefetch PROMPT_FILE | --compressor-prefill PROMPT_FILE | "
                        "--engram-pipeline PROMPT_FILE | "
                        "--packed-index-prefill PROMPT_FILE | "
                        "--compact-carry-prefill PROMPT_FILE | "
                        "--bf16-prefill PROMPT_FILE | "
                        "--hc-scaled-prefill PROMPT_FILE | "
                        "--topk-prefix-prefill PROMPT_FILE | "
                        "--prefill-alias PROMPT_FILE | "
                        "--wide-expert-prefill PROMPT_FILE | "
                        "--prefill-alias-fallback PROMPT_FILE | "
                        "--sweep-partitions PROMPT_FILE | "
                        "--deferred-decoder PROMPT_FILE | "
                        "--decoder-suffix PROMPT_FILE | --session-accounting | --memory-plan | "
                        "RENDERED_PROMPT [GENERATE])\n", argv[0]);
        return 2;
    }
    int rc = 1;
    const bool zero = strcmp(argv[2], "--zero-fixture") == 0;
    int generate = argc == 4 ? atoi(argv[3]) : 32;
    if (generate < 0 || generate > 512) return 2;
    ds4_model model = {.fd = -1};
    ds4_weights weights = {0};
    ds4_vocab vocab = {0};
    ds41_gpu_graph graph = {.table = {{.fd = -1}, {.fd = -1}}};
    token_vec prompt = {0};
    float *logits = NULL;
    ds4_model_map_span_vec spans = {0};
    uint64_t *offsets = NULL, *sizes = NULL;
    model_open(&model, argv[1], true, false);
    config_validate_model(&model);
    REQUIRE(DS4_MODEL_FAMILY == DS4_MODEL_FAMILY_DEEPSEEK41);
    weights_bind(&weights, &model, false, 0, UINT32_MAX, true, false);
    vocab_load(&vocab, &model);
    if (zero) {
        for (int i = 0; i < 4; i++) token_vec_push(&prompt, 100 + i);
    } else {
        tokenize_rendered_chat_vocab(&vocab, argv[2], &prompt);
    }
    REQUIRE(prompt.len > 0 && prompt.len + generate <= 4096);
    ds4_gpu_model_residency_skip(1);
    REQUIRE(ds4_gpu_init());
    ds4_gpu_set_ssd_streaming(true);
    ds4_gpu_set_streaming_expert_cache_budget(512);
    uint64_t gate_row = 0, down_row = 0;
    REQUIRE(tensor_nbytes(weights.layer[0].ffn_gate_exps->type, DS4_N_EMBD, &gate_row));
    REQUIRE(tensor_nbytes(weights.layer[0].ffn_down_exps->type, DS4_N_FF_EXP, &down_row));
    ds4_gpu_set_streaming_expert_cache_expert_bytes(
        2u * gate_row * DS4_N_FF_EXP + down_row * DS4_N_EMBD);
    REQUIRE(ds4_gpu_set_model_fd(model.fd));
    model_map_span_vec_include_one(&spans, weights.token_embd);
    model_map_span_vec_include_output(&spans, &weights);
    for (uint32_t il = 0; il < DS4_N_LAYER; il++)
        model_map_span_vec_include_layer_decode_static(&spans, &weights.layer[il]);
    offsets = malloc(spans.len * sizeof(*offsets));
    sizes = malloc(spans.len * sizeof(*sizes));
    REQUIRE(offsets && sizes);
    for (uint32_t i = 0; i < spans.len; i++) {
        offsets[i] = spans.v[i].off;
        sizes[i] = spans.v[i].end - spans.v[i].off;
        REQUIRE(spans.v[i].end <= model.size);
    }
    REQUIRE(ds4_gpu_set_model_map_spans(model.map, model.size, offsets, sizes,
                                       spans.len, spans.max_tensor_bytes));
    REQUIRE(ds41_graph_alloc(&graph, &model, &weights, argv[1], zero ? 20000 : 4096, true));
    logits = malloc((size_t)DS4_N_VOCAB * sizeof(float));
    REQUIRE(logits);
    const double start = now_sec();
    for (int i = 0; i < prompt.len; i++) {
        REQUIRE(ds41_graph_step(&graph, &model, &weights, prompt.v[i], logits));
        for (uint32_t j = 0; j < DS4_N_VOCAB; j++)
            REQUIRE(isfinite(logits[j]) && (!zero || logits[j] == 0));
        fprintf(stderr, "V4.1 token %d/%d: %.3fs elapsed, argmax %d\n",
                i + 1, prompt.len, now_sec() - start, sample_argmax(logits, DS4_N_VOCAB));
    }
    if (zero) {
        ds41_graph_reset(&graph);
        REQUIRE(ds41_graph_step(&graph, &model, &weights, prompt.v[0], logits));
        for (uint32_t j = 0; j < DS4_N_VOCAB; j++) REQUIRE(logits[j] == 0);
        /* Exercise sparse attention beyond both the old 8192-row helper limit
         * and the 2048-block candidate frontier without a 17k-token prefill. */
        graph.pos = 17017;
        ds41_state_span cache[54];
        const uint32_t count = ds41_state_spans(&graph, graph.pos, cache);
        for (uint32_t i = 0; i < count; i++)
            memset(ds4_gpu_tensor_contents(cache[i].tensor), 0, (size_t)cache[i].bytes);
        REQUIRE(ds41_graph_step(&graph, &model, &weights, prompt.v[0], logits));
        REQUIRE(ds41_graph_step(&graph, &model, &weights, prompt.v[1], logits));
        for (uint32_t j = 0; j < DS4_N_VOCAB; j++) REQUIRE(logits[j] == 0);
        puts("V4.1 zero-weight SSD graph and reset: PASS");
    } else {
        ds4_engine output = {.vocab = vocab};
        for (int i = 0; i < generate; i++) {
            const int token = sample_argmax(logits, DS4_N_VOCAB);
            size_t len = 0;
            char *piece = ds4_token_text(&output, token, &len);
            REQUIRE(piece);
            fwrite(piece, 1, len, stdout);
            fflush(stdout);
            free(piece);
            if (vocab_token_is_generation_stop(&vocab, token)) break;
            REQUIRE(ds41_graph_step(&graph, &model, &weights, token, logits));
            for (uint32_t j = 0; j < DS4_N_VOCAB; j++) REQUIRE(isfinite(logits[j]));
        }
        putchar('\n');
    }
    rc = 0;
done:
    if (ds4_gpu_commands_active()) ds4_gpu_end_commands();
    ds41_graph_free(&graph);
    ds4_gpu_cleanup();
    vocab_free(&vocab);
    token_vec_free(&prompt);
    model_close(&model);
    free(logits); free(spans.v); free(offsets); free(sizes);
    return rc;
}
