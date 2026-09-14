/* Run model checks only on dedicated GPU hosts with enough memory. */
#include "../ds4.c"
#include <assert.h>

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); goto done; \
} } while (0)

static int check_dispatch(void) {
    int rc = 1;
    g_ds4_shape = DS4_SHAPE_FLASH41;
    ds4_weights weights = {0};
    ds4_tensor gate = {.type = DS4_TENSOR_IQ2_XXS}, down = {.type = DS4_TENSOR_Q2_K};
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        weights.layer[il].ffn_gate_exps = weights.layer[il].ffn_up_exps = &gate;
        weights.layer[il].ffn_down_exps = &down;
    }
    ds41_gpu_graph g = {.ctx = 131072, .prefill_cap = 8192,
        .carry_cap = 32768, .streaming = true, .tp_world = 1};
    const uint32_t half = DS4_N_LAYER * DS4_N_EXPERT / 2u;
    const uint32_t saved = ds4_gpu_stream_expert_cache_configured_count();
    ds4_gpu_set_ssd_streaming(true);
    const uint32_t remaining[] = {1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65,
        127, 128, 255, 256, 257, 511, 512, 513, 1023, 1024,
        2047, 2048, 2049, 4095, 4096, 4097, 8191, 8192, 8193,
        16383, 16384, 16385, 32767, 32768, 32769, 65536};
    const uint32_t cold[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 256, 257, 511, 512, 513, 1023, 1024,
        2047, 2048, 2048, 2048, 4096, 4096, 6144, 8192, 8192,
        14336, 16384, 16384, 30720, 32768, 32768, 32768};
    _Static_assert(sizeof(remaining) == sizeof(cold), "prefill dispatch table sizes");
    for (uint32_t cache = half - 1; cache <= half; cache++) {
        ds4_gpu_set_streaming_expert_cache_budget(cache);
        for (uint32_t warm = 0; warm < 2; warm++) {
            g.pos = warm;
            for (size_t i = 0; i < sizeof(remaining) / sizeof(*remaining); i++) {
                uint32_t expected = cold[i];
#ifdef __APPLE__
                if (warm && cache == half && remaining[i] < 1024) expected = 1;
#elif !defined(DS4_ROCM_BUILD)
                if (remaining[i] > 2048 && remaining[i] < 8192 && remaining[i] % 2048 >= 256)
                    expected = remaining[i];
#endif
                if (ds41_prefill_count(&g, remaining[i]) != expected)
                    fprintf(stderr, "dispatch cache=%u configured=%u warm=%u remaining=%u expected=%u actual=%u\n",
                        cache, ds4_gpu_stream_expert_cache_configured_count(), warm,
                        remaining[i], expected, ds41_prefill_count(&g, remaining[i]));
                CHECK(ds41_prefill_count(&g, remaining[i]) == expected);
                uint32_t small = 0;
#ifndef __APPLE__
                if (remaining[i] >= 2 && remaining[i] < 256)
                    small = remaining[i] < 8 ? remaining[i] : 8;
#endif
                CHECK(ds41_short_prefill_count(&g, &weights, remaining[i]) == small);
            }
        }
    }
    g.pos = 0;
#if !defined(__APPLE__) && !defined(DS4_ROCM_BUILD)
    CHECK(ds41_prefill_count(&g, 2303) == 2048);
    CHECK(ds41_prefill_count(&g, 2304) == 2304);
    CHECK(ds41_prefill_count(&g, 3241) == 3241);
    CHECK(setenv("DS4_CUDA_DISABLE_SSD_MEDIUM_SWEEP", "1", 1) == 0);
    CHECK(ds41_prefill_count(&g, 3241) == 2048);
    CHECK(unsetenv("DS4_CUDA_DISABLE_SSD_MEDIUM_SWEEP") == 0);
    g.prefill_cap = 1024;
    CHECK(ds41_prefill_count(&g, 3241) == 1024);
    g.prefill_cap = 8192;
#endif
    g.tp_world = 2;
    g.streaming = false;
    for (size_t i = 0; i < sizeof(remaining) / sizeof(*remaining); i++) {
        uint32_t expected = cold[i], small = 0;
#ifdef __APPLE__
        if (remaining[i] >= 32 && remaining[i] < 256) expected = remaining[i];
#else
        if (remaining[i] >= 2 && remaining[i] < 256)
            small = remaining[i] < 8 ? remaining[i] : 8;
#endif
        CHECK(ds41_prefill_count(&g, remaining[i]) == expected);
        CHECK(ds41_short_prefill_count(&g, &weights, remaining[i]) == small);
    }
    CHECK(setenv("DS4_METAL_DISABLE_V41_TP_SMALL_PREFILL", "1", 1) == 0);
    CHECK(ds41_prefill_count(&g, 255) == 1);
    CHECK(ds41_short_prefill_count(&g, &weights, 255) == 0);
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
    unsigned callbacks, scalar, batches, short_batches, deferred, displays;
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
            const uint32_t small = ds41_short_prefill_count(&before, &s->engine->weights,
                (uint32_t)(total - p->frontier));
            assert((uint32_t)count == (small ? small : ds41_prefill_count(&before,
                (uint32_t)(total - p->frontier))));
            if (count == 1) p->scalar++;
            else if (small) p->short_batches++;
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

static bool cancel_after_display(void *ud) {
    return ((prefill_progress *)ud)->displays >= 2;
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
        const size_t bytes = (size_t)sa[i].bytes;
        void *left = malloc(bytes), *right = malloc(bytes);
        const bool readable = left && right && sa[i].bytes == sb[i].bytes &&
            ds4_gpu_tensor_read(sa[i].tensor, 0, left, bytes) &&
            ds4_gpu_tensor_read(sb[i].tensor, 0, right, bytes);
        const bool equal = readable && memcmp(left, right, bytes) == 0;
        if (!equal) {
            fprintf(stderr, "frontier=%d cache span=%u differs (%llu bytes)\n",
                ds4_session_pos(a), i, (unsigned long long)sa[i].bytes);
            if (readable) {
                unsigned different = 0;
                float worst = 0;
                const float *x = left, *y = right;
                for (size_t j = 0; j < bytes / sizeof(float); j++) {
                    if (x[j] == y[j]) continue;
                    if (different++ < 4) fprintf(stderr, "  cache[%zu] %.9g != %.9g\n", j, x[j], y[j]);
                    const float gap = fabsf(x[j] - y[j]);
                    if (!isfinite(gap) || gap > worst) worst = gap;
                }
                fprintf(stderr, "  differing=%u worst=%g\n", different, worst);
            }
            free(left); free(right);
            return false;
        }
        free(left); free(right);
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

typedef enum {
    PREFILL_METAL, PREFILL_CUDA, PREFILL_CUDA_LONG,
    PREFILL_CUDA_DEFERRED, PREFILL_CUDA_SMALL
} prefill_test_mode;

static int check_mixed(const char *model, const char *prompt_path,
                       const ds4_tp_options *tp_opt, bool resident, prefill_test_mode mode) {
    const bool cuda = mode != PREFILL_METAL;
    const bool cuda_long = mode == PREFILL_CUDA_LONG || mode == PREFILL_CUDA_DEFERRED;
    const bool deferred_only = mode == PREFILL_CUDA_DEFERRED;
    const bool cuda_small = mode == PREFILL_CUDA_SMALL;
    ds4_engine *engine = NULL;
    ds4_tp *tp = NULL;
    ds4_session *control = NULL, *mixed = NULL;
    ds4_session_snapshot snap = {0};
    ds4_tokens tokens = {0};
    char *prompt = NULL, err[256] = {0};
    size_t bytes;
    int rc = 1;
    ds4_engine_options opt = {.model_path = model,
        .backend = cuda ? DS4_BACKEND_CUDA : DS4_BACKEND_METAL,
        .context_size = cuda ? (cuda_long ? 65536 : 16384) : 131072, .power_percent = 100,
        .ssd_streaming = !tp_opt && !resident,
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
    CHECK(tokens.len > (cuda ? opt.context_size : resident ? 131072 : 120000));
    CHECK(ds4_session_create(&control, engine, opt.context_size) == 0);
    CHECK(ds4_session_create(&mixed, engine, opt.context_size) == 0);
    const int ordinary_appends[] = {127, 1, 255, 256, 257, 1023, 1024, 4095, 4096,
        8191, 8192, 16383, 16384, 49153, 129, 4096, 16383};
    const int small_appends[] = {7, 1, 8, 9, 15, 16, 17, 31, 1, 32, 63, 64, 65, 127, 128, 129,
        255, 256, 257, 511, 512, 513, 1023, 1024};
    const int *appends = cuda_small ? small_appends : ordinary_appends;
    const size_t n_appends = cuda_small ? sizeof(small_appends) / sizeof(*small_appends) :
        cuda ? (cuda_long ? 14u : 9u) :
        sizeof(ordinary_appends) / sizeof(*ordinary_appends) - (resident ? 0u : 1u);
    unsigned scalar = 0, batches = 0, deferred = 0;
    for (size_t i = deferred_only ? 13u : 0u; i < n_appends; i++) {
        /* A single 49K append crosses the deferred-decoder threshold. Reset
         * both sessions to cover it without allocating two 128K graphs. */
        if (cuda_long && i == 13) {
            ds4_session_invalidate(control);
            ds4_session_invalidate(mixed);
        }
        const int start = ds4_session_pos(mixed);
        tokens.len = start + appends[i];
        /* The normal worker mirrors batch ordering. Keep TP partitions equal,
         * but compare queued decode against synchronous layer submission. */
        const char *ablation = tp ? "DS4_METAL_DISABLE_V41_TP_DECODE_QUEUE" :
            "DS4_METAL_DISABLE_V41_DEFER_DECODER";
        CHECK(setenv(ablation, "1", 1) == 0);
        if (cuda && !tp) {
            CHECK(setenv("DS4_CUDA_SESSION_BATCH_MOE", "0", 1) == 0);
            CHECK(setenv("DS4_CUDA_DISABLE_SSD_PREFETCH", "1", 1) == 0);
            CHECK(setenv("DS4_CUDA_DISABLE_SSD_MEDIUM_SWEEP", "1", 1) == 0);
        }
        if (cuda && tp && appends[i] < 256) {
            /* Mirror scalar control execution on the worker too. */
            ds4_tokens prefix = tokens;
            for (prefix.len = start + 1; prefix.len <= tokens.len; prefix.len++)
                CHECK(ds4_session_sync(control, &prefix, err, sizeof(err)) == 0);
        } else {
            CHECK(ds4_session_sync(control, &tokens, err, sizeof(err)) == 0);
        }
        if (cuda && !tp) {
            CHECK(unsetenv("DS4_CUDA_SESSION_BATCH_MOE") == 0);
            CHECK(unsetenv("DS4_CUDA_DISABLE_SSD_PREFETCH") == 0);
            CHECK(unsetenv("DS4_CUDA_DISABLE_SSD_MEDIUM_SWEEP") == 0);
        }
        CHECK(unsetenv(ablation) == 0);
        prefill_progress p = {.session = mixed, .frontier = start,
            .current = start, .target = tokens.len, .begin = now_sec()};
        ds4_session_set_progress(mixed, progress_note, &p);
        CHECK(ds4_session_sync(mixed, &tokens, err, sizeof(err)) == 0);
        const double seconds = now_sec() - p.begin;
        CHECK(p.current == tokens.len && p.final_checked);
        CHECK(!p.batches || (p.displays && p.partial_checked));
        scalar += p.scalar; batches += p.batches + p.short_batches; deferred += p.deferred;
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
        fprintf(stderr, "mixed start=%d append=%d scalar=%u batches=%u short_batches=%u deferred=%u "
            "display=%u first_display=%.3fs time=%.3fs rate=%.2f: exact state/decode PASS\n",
            start, appends[i], p.scalar, p.batches, p.short_batches, p.deferred, p.displays,
            p.first_display, seconds, appends[i] / seconds);
        if (cuda && i == 3) {
            CHECK(ds4_session_save_snapshot(mixed, &snap, err, sizeof(err)) == 0);
            const int frontier = ds4_session_pos(mixed);
            tokens.len = frontier + 1024;
            prefill_progress cancelled = {.session = mixed, .frontier = frontier,
                .current = frontier, .target = tokens.len, .begin = now_sec()};
            ds4_session_set_progress(mixed, progress_note, &cancelled);
            ds4_session_set_cancel(mixed, cancel_after_display, &cancelled);
            CHECK(ds4_session_sync(mixed, &tokens, err, sizeof(err)) == DS4_SESSION_SYNC_INTERRUPTED);
            CHECK(cancelled.partial_checked && !mixed->checkpoint_valid);
            /* TP invalidates both ranks and resets to an empty valid graph;
             * the local path leaves its unfinished graph explicitly invalid. */
            CHECK(tp ? (mixed->ds41_graph.valid && mixed->ds41_graph.pos == 0 &&
                         mixed->checkpoint.len == 0) : !mixed->ds41_graph.valid);
            ds4_session_set_cancel(mixed, NULL, NULL);
            ds4_session_set_progress(mixed, NULL, NULL);
            if (tp) {
                /* TP restore replays the prefix; compare the same partition. */
                ds4_session_invalidate(control);
                tokens.len = frontier;
                CHECK(ds4_session_sync(control, &tokens, err, sizeof(err)) == 0);
            }
            CHECK(ds4_session_load_snapshot(mixed, &snap, err, sizeof(err)) == 0);
            CHECK(state_equal(control, mixed));
            CHECK(ds4_session_eval(control, tokens.v[frontier], err, sizeof(err)) == 0);
            CHECK(ds4_session_eval(mixed, tokens.v[frontier], err, sizeof(err)) == 0);
            CHECK(state_equal(control, mixed));
            ds4_session_snapshot_free(&snap);
            puts("V4.1 cancelled layer prefill, restore and next decode: exact PASS");
        }
    }
    CHECK(scalar && batches && ((cuda && !cuda_long) || deferred));
    puts(deferred_only ? "V4.1 deferred decoder, progress and restore: PASS" :
        "V4.1 mixed small/large continued prefill, dispatch, progress and restore: PASS");
    rc = 0;
done:
    if (cuda) unsetenv("DS4_CUDA_SESSION_BATCH_MOE");
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
    if (argc == 3) return check_mixed(argv[1], argv[2], NULL, false, PREFILL_METAL);
    if (argc == 4 && !strcmp(argv[1], "--resident"))
        return check_mixed(argv[2], argv[3], NULL, true, PREFILL_METAL);
    if (argc == 4 && !strcmp(argv[1], "--cuda"))
        return check_mixed(argv[2], argv[3], NULL, false, PREFILL_CUDA);
    if (argc == 4 && !strcmp(argv[1], "--cuda-small"))
        return check_mixed(argv[2], argv[3], NULL, false, PREFILL_CUDA_SMALL);
    if (argc == 4 && !strcmp(argv[1], "--cuda-long"))
        return check_mixed(argv[2], argv[3], NULL, false, PREFILL_CUDA_LONG);
    if (argc == 4 && !strcmp(argv[1], "--cuda-deferred"))
        return check_mixed(argv[2], argv[3], NULL, false, PREFILL_CUDA_DEFERRED);
    if (argc == 8 && (!strcmp(argv[1], "--tensor-parallel") ||
                     !strcmp(argv[1], "--tensor-parallel-cuda") ||
                     !strcmp(argv[1], "--tensor-parallel-cuda-small"))) {
        ds4_tp_options tp = {.role = DS4_TP_LEADER, .requested = true,
            .listen_host = argv[4], .listen_port = atoi(argv[5]),
            .transport = DS4_TP_TRANSPORT_RDMA, .rdma_device = argv[6],
            .rdma_gid_index = atoi(argv[7]), .rdma_gid_index_set = true};
        return check_mixed(argv[2], argv[3], &tp, false,
            !strcmp(argv[1], "--tensor-parallel-cuda-small") ? PREFILL_CUDA_SMALL :
            !strcmp(argv[1], "--tensor-parallel-cuda") ? PREFILL_CUDA : PREFILL_METAL);
    }
    fprintf(stderr, "usage: %s --dispatch | MODEL LONG_PROMPT_FILE | "
        "--resident MODEL LONG_PROMPT_FILE | --cuda[-small|-long|-deferred] MODEL LONG_PROMPT_FILE | "
        "--tensor-parallel[-cuda[-small]] MODEL LONG_PROMPT_FILE LISTEN_HOST PORT RDMA_DEVICE GID\n", argv[0]);
    return 2;
}
