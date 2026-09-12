/* Run model checks only on dedicated Metal hosts with enough memory. */
#include "../ds4.c"
#include <assert.h>

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); goto done; \
} } while (0)

static int check_dispatch(void) {
    int rc = 1;
    g_ds4_shape = DS4_SHAPE_FLASH41;
    ds41_gpu_graph g = {.ctx = 131072, .prefill_cap = 8192,
        .carry_cap = 32768, .streaming = true, .tp_world = 1};
    const uint32_t half = DS4_N_LAYER * DS4_N_EXPERT / 2u;
    const uint32_t saved = ds4_gpu_stream_expert_cache_configured_count();
    ds4_gpu_set_ssd_streaming(true);
    const uint32_t remaining[] = {1, 31, 32, 33, 127, 128, 255, 256, 257, 511, 512, 513, 1023, 1024,
        2047, 2048, 2049, 4095, 4096, 4097, 8191, 8192, 8193,
        16383, 16384, 16385, 32767, 32768, 32769, 65536};
    const uint32_t cold[] = {1, 1, 1, 1, 1, 1, 1, 256, 257, 511, 512, 513, 1023, 1024,
        2047, 2048, 2048, 2048, 4096, 4096, 6144, 8192, 8192,
        14336, 16384, 16384, 30720, 32768, 32768, 32768};
    for (uint32_t cache = half - 1; cache <= half; cache++) {
        ds4_gpu_set_streaming_expert_cache_budget(cache);
        for (uint32_t warm = 0; warm < 2; warm++) {
            g.pos = warm;
            for (size_t i = 0; i < sizeof(remaining) / sizeof(*remaining); i++) {
                const uint32_t expected = warm && cache == half && remaining[i] < 1024 ?
                    1 : cold[i];
                if (ds41_prefill_count(&g, remaining[i]) != expected)
                    fprintf(stderr, "dispatch cache=%u configured=%u warm=%u remaining=%u expected=%u actual=%u\n",
                        cache, ds4_gpu_stream_expert_cache_configured_count(), warm,
                        remaining[i], expected, ds41_prefill_count(&g, remaining[i]));
                CHECK(ds41_prefill_count(&g, remaining[i]) == expected);
            }
        }
    }
    g.pos = 0;
    g.tp_world = 2;
    g.streaming = false;
    for (size_t i = 0; i < sizeof(remaining) / sizeof(*remaining); i++)
        CHECK(ds41_prefill_count(&g, remaining[i]) ==
            (remaining[i] >= 32 && remaining[i] < 256 ? remaining[i] : cold[i]));
    CHECK(setenv("DS4_METAL_DISABLE_V41_TP_SMALL_PREFILL", "1", 1) == 0);
    CHECK(ds41_prefill_count(&g, 255) == 1);
    CHECK(unsetenv("DS4_METAL_DISABLE_V41_TP_SMALL_PREFILL") == 0);
    const char *ablations[] = {"DS4_METAL_DISABLE_V41_BATCH_ATTN",
        "DS4_METAL_DISABLE_V41_BATCH_CORE", "DS4_METAL_DISABLE_V41_BATCH_MOE",
        "DS4_METAL_DISABLE_V41_BATCH_HC", "DS4_METAL_DISABLE_V41_LAYER_PREFILL"};
    for (size_t i = 0; i < sizeof(ablations) / sizeof(*ablations); i++) {
        CHECK(setenv(ablations[i], "1", 1) == 0);
        CHECK(ds41_prefill_count(&g, 65536) == 1);
        CHECK(unsetenv(ablations[i]) == 0);
    }
    g.tp_world = 1;
    CHECK(ds41_prefill_count(&g, 7) == 1);
    CHECK(ds41_prefill_count(&g, 8) == 8);
    for (size_t i = 0; i < sizeof(remaining) / sizeof(*remaining); i++)
        CHECK(ds41_prefill_count(&g, remaining[i]) ==
            (remaining[i] >= 8 && remaining[i] < 256 ? remaining[i] : cold[i]));
    ds4_imatrix_collector imatrix = {0};
    g.imatrix = &imatrix;
    CHECK(ds41_prefill_count(&g, 65536) == 1);
    g.imatrix = NULL;
    g.carry_cap = 0;
    CHECK(ds41_prefill_count(&g, 65536) == 2048);
    g.prefill_cap = 1024;
    CHECK(ds41_prefill_count(&g, 4096) == 1024);
    g.prefill_cap = 8192;
    CHECK(ds41_encoder_chunk_cap(&g, 8191) == 2048);
    CHECK(ds41_encoder_chunk_cap(&g, 8192) == 4096);
    CHECK(ds41_encoder_chunk_cap(&g, 16383) == 4096);
    CHECK(ds41_encoder_chunk_cap(&g, 16384) == 8192);
    puts("V4.1 cold/warm and TP prefill dispatch, tile boundaries and debug/imatrix fallbacks: PASS");
    rc = 0;
done:
    ds4_gpu_set_streaming_expert_cache_budget(saved);
    ds4_gpu_set_ssd_streaming(false);
    return rc;
}

typedef struct {
    ds4_session *session;
    int target, current, frontier;
    unsigned callbacks, scalar, batches, deferred, displays;
    bool partial_checked, final_checked;
    double begin, first_display;
} prefill_progress;

static void progress_note(void *ud, const char *event, int current, int total) {
    prefill_progress *p = ud;
    ds4_session *s = p->session;
    assert(total == p->target && current >= p->current && current <= total);
    p->current = current;
    p->callbacks++;
    if (!strcmp(event, "prefill_display")) {
        if (!p->displays++) p->first_display = now_sec() - p->begin;
        assert(!s->ds41_graph.valid);
        if (!p->partial_checked) {
            ds4_session_snapshot snap = {0};
            char err[256];
            assert(ds4_session_save_snapshot(s, &snap, err, sizeof(err)) != 0);
            ds4_session_snapshot_free(&snap);
            p->partial_checked = true;
        }
    } else {
        assert(!strcmp(event, "prefill_chunk"));
        if (current != p->frontier) {
            const int count = current - p->frontier;
            ds41_gpu_graph before = s->ds41_graph;
            before.pos = (uint32_t)p->frontier;
            assert((uint32_t)count == ds41_prefill_count(&before,
                (uint32_t)(total - p->frontier)));
            if (count == 1) p->scalar++;
            else p->batches++;
            if (!s->ds41_graph.valid) p->deferred++;
            p->frontier = current;
        }
        if (s->checkpoint_valid) {
            assert(current == total && s->ds41_graph.valid);
            p->final_checked = true;
        }
    }
}

static bool state_equal(ds4_session *a, ds4_session *b) {
    if (!a->checkpoint_valid || !b->checkpoint_valid ||
        !a->ds41_graph.valid || !b->ds41_graph.valid ||
        ds4_session_pos(a) != ds4_session_pos(b) ||
        memcmp(&a->ds41_graph.history, &b->ds41_graph.history,
            sizeof(a->ds41_graph.history))) return false;
    ds41_state_span sa[64], sb[64];
    uint32_t n = ds41_state_spans(&a->ds41_graph, a->ds41_graph.pos, sa);
    if (n != ds41_state_spans(&b->ds41_graph, b->ds41_graph.pos, sb)) return false;
    for (uint32_t i = 0; i < n; i++) {
        if (sa[i].bytes != sb[i].bytes ||
            memcmp(ds4_gpu_tensor_contents(sa[i].tensor),
                ds4_gpu_tensor_contents(sb[i].tensor), (size_t)sa[i].bytes)) {
            fprintf(stderr, "frontier=%d cache span=%u differs (%llu bytes)\n",
                ds4_session_pos(a), i, (unsigned long long)sa[i].bytes);
            return false;
        }
    }
    for (uint32_t i = 0; i < DS4_N_VOCAB; i++) {
        if (!isfinite(a->logits[i]) || a->logits[i] != b->logits[i]) {
            fprintf(stderr, "frontier=%d logit=%u control=%.9g mixed=%.9g\n",
                ds4_session_pos(a), i, a->logits[i], b->logits[i]);
            return false;
        }
    }
    return true;
}

static int check_mixed(const char *model, const char *prompt_path,
                       const ds4_tp_options *tp_opt, bool resident) {
    ds4_engine *engine = NULL;
    ds4_tp *tp = NULL;
    ds4_session *control = NULL, *mixed = NULL;
    ds4_session_snapshot snap = {0};
    ds4_tokens tokens = {0};
    char *prompt = NULL, err[256] = {0};
    size_t bytes;
    int rc = 1;
    ds4_engine_options opt = {.model_path = model, .backend = DS4_BACKEND_METAL,
        .context_size = 131072, .power_percent = 100, .ssd_streaming = !tp_opt && !resident,
        .ssd_streaming_cache_bytes = tp_opt || resident ? 0 : UINT64_C(64) << 30};
    if (tp_opt) opt.tp = *tp_opt;
    CHECK(imatrix_read_text_file(prompt_path, &prompt, &bytes));
    CHECK(ds4_engine_open(&engine, &opt) == 0);
    if (tp_opt) {
        ds4_tp_identity id = {
            .gguf_bytes = ds4_engine_model_bytes(engine),
            .model_id = (uint32_t)ds4_engine_model_id(engine),
            .n_layer = (uint32_t)ds4_engine_layer_count(engine),
            .n_embd = (uint32_t)ds4_engine_embd_dim(engine),
            .n_vocab = (uint32_t)ds4_engine_vocab_size(engine),
            .quant_bits = (uint32_t)ds4_engine_routed_quant_bits(engine),
            .ctx_size = (uint32_t)opt.context_size,
        };
        ds4_engine_tp_gate_schedule(engine, &id.gate_slot_start,
            &id.gate_slot_step, &id.gates_per_token, id.gate_slot_mask);
        CHECK(ds4_tp_create(&tp, tp_opt, &id, err, sizeof(err)));
        CHECK(ds4_engine_tp_bind(engine, tp, err, sizeof(err)));
    }
    ds4_encode_chat_prompt(engine, NULL, prompt, DS4_THINK_NONE, &tokens);
    CHECK(tokens.len > (resident ? 131072 : 120000));
    CHECK(ds4_session_create(&control, engine, opt.context_size) == 0);
    CHECK(ds4_session_create(&mixed, engine, opt.context_size) == 0);
    const int appends[] = {127, 1, 255, 256, 257, 1023, 1024, 4095, 4096,
        8191, 8192, 16383, 16384, 49153, 129, 4096, 16383};
    const size_t n_appends = sizeof(appends) / sizeof(*appends) - (resident ? 0u : 1u);
    unsigned scalar = 0, batches = 0, deferred = 0;
    for (size_t i = 0; i < n_appends; i++) {
        const int start = ds4_session_pos(mixed);
        tokens.len = start + appends[i];
        /* The normal worker mirrors batch ordering. Keep TP partitions equal,
         * but compare queued decode against synchronous layer submission. */
        const char *ablation = tp ? "DS4_METAL_DISABLE_V41_TP_DECODE_QUEUE" :
            "DS4_METAL_DISABLE_V41_DEFER_DECODER";
        CHECK(setenv(ablation, "1", 1) == 0);
        CHECK(ds4_session_sync(control, &tokens, err, sizeof(err)) == 0);
        CHECK(unsetenv(ablation) == 0);
        prefill_progress p = {.session = mixed, .frontier = start,
            .current = start, .target = tokens.len, .begin = now_sec()};
        ds4_session_set_progress(mixed, progress_note, &p);
        CHECK(ds4_session_sync(mixed, &tokens, err, sizeof(err)) == 0);
        const double seconds = now_sec() - p.begin;
        CHECK(p.current == tokens.len && p.final_checked);
        CHECK(!p.batches || (p.displays && p.partial_checked));
        scalar += p.scalar; batches += p.batches; deferred += p.deferred;
        CHECK(state_equal(control, mixed));
        const unsigned callbacks = p.callbacks;
        CHECK(ds4_session_sync(mixed, &tokens, err, sizeof(err)) == 0);
        CHECK(callbacks == p.callbacks && state_equal(control, mixed));
        ds4_session_set_progress(mixed, NULL, NULL);
        CHECK(ds4_session_save_snapshot(mixed, &snap, err, sizeof(err)) == 0);
        const int token = tokens.v[tokens.len];
        CHECK(setenv("DS4_METAL_DISABLE_V41_TP_DECODE_QUEUE", "1", 1) == 0);
        CHECK(ds4_session_eval(control, token, err, sizeof(err)) == 0);
        CHECK(unsetenv("DS4_METAL_DISABLE_V41_TP_DECODE_QUEUE") == 0);
        CHECK(ds4_session_eval(mixed, token, err, sizeof(err)) == 0);
        CHECK(state_equal(control, mixed));
        if (!tp) {
            CHECK(ds4_session_load_snapshot(mixed, &snap, err, sizeof(err)) == 0);
            CHECK(ds4_session_eval(mixed, token, err, sizeof(err)) == 0);
            CHECK(state_equal(control, mixed));
        } else if (i + 1 == n_appends) {
            /* TP snapshots rebuild both ranks from tokens. Compare against
             * the same one-shot replay, not different batched reductions. */
            ds4_session_invalidate(control);
            CHECK(ds4_session_sync(control, &tokens, err, sizeof(err)) == 0);
            CHECK(ds4_session_load_snapshot(mixed, &snap, err, sizeof(err)) == 0);
            CHECK(state_equal(control, mixed));
            CHECK(ds4_session_eval(control, token, err, sizeof(err)) == 0);
            CHECK(ds4_session_eval(mixed, token, err, sizeof(err)) == 0);
            CHECK(state_equal(control, mixed));
            puts("TP snapshot rebuild, both-rank replay and next decode: PASS");
        }
        ds4_session_snapshot_free(&snap);
        fprintf(stderr, "mixed start=%d append=%d scalar=%u batches=%u deferred=%u "
            "display=%u first_display=%.3fs time=%.3fs rate=%.2f: exact state/decode PASS\n",
            start, appends[i], p.scalar, p.batches, p.deferred, p.displays,
            p.first_display, seconds, appends[i] / seconds);
    }
    CHECK(scalar && batches && deferred);
    puts("V4.1 mixed small/large continued prefill, dispatch, progress and restore: PASS");
    rc = 0;
done:
    unsetenv("DS4_METAL_DISABLE_V41_DEFER_DECODER");
    unsetenv("DS4_METAL_DISABLE_V41_TP_DECODE_QUEUE");
    if (rc) fprintf(stderr, "mixed prefill failure: %s\n", err);
    ds4_session_snapshot_free(&snap);
    ds4_tokens_free(&tokens); free(prompt);
    ds4_session_free(mixed); ds4_session_free(control);
    if (tp) ds4_tp_send_stop(tp);
    ds4_engine_close(engine); ds4_tp_free(tp);
    return rc;
}

int main(int argc, char **argv) {
    if (argc == 2 && !strcmp(argv[1], "--dispatch")) return check_dispatch();
    if (argc == 3) return check_mixed(argv[1], argv[2], NULL, false);
    if (argc == 4 && !strcmp(argv[1], "--resident"))
        return check_mixed(argv[2], argv[3], NULL, true);
    if (argc == 8 && !strcmp(argv[1], "--tensor-parallel")) {
        ds4_tp_options tp = {.role = DS4_TP_LEADER, .requested = true,
            .listen_host = argv[4], .listen_port = atoi(argv[5]),
            .transport = DS4_TP_TRANSPORT_RDMA, .rdma_device = argv[6],
            .rdma_gid_index = atoi(argv[7]), .rdma_gid_index_set = true};
        return check_mixed(argv[2], argv[3], &tp, false);
    }
    fprintf(stderr, "usage: %s --dispatch | MODEL LONG_PROMPT_FILE | "
        "--resident MODEL LONG_PROMPT_FILE | "
        "--tensor-parallel MODEL LONG_PROMPT_FILE LISTEN_HOST PORT RDMA_DEVICE GID\n", argv[0]);
    return 2;
}
