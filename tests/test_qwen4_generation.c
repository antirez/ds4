/* Real-model one-shot/session parity with identical effective prefill chunks.
 * --prefill-reference compares automatic dispatch with the diagnostic
 * SSD legacy, unsplit prefill path, then uses automatic dispatch for both decodes.
 * Usage: test_qwen4_generation MODEL [--ssd-streaming] [--prefill-reference]
 *        [--chunk N --ctx N [--tokens N]]
 * SSD mode bounds the expert cache to 1024 entries. At most two graphs are
 * live; the public one-shot graph is released before numerical comparison. */
#include "../ds4.c"

#ifdef DS4_HAS_QWEN4_METAL
enum { GENERATION_STEPS = 8 };
static char generation_error[256];

static void generation_need(bool ok, const char *what) {
    if (ok) return;
    fprintf(stderr, "Qwen generation: %s: %s\n", what, generation_error);
    exit(1);
}

typedef struct {
    const char *path;
    uint32_t cap;
    int length, last, calls;
    int tokens[GENERATION_STEPS], count, done;
} generation_capture;

static void generation_progress(void *opaque, const char *event, int current, int total) {
    if (strcmp(event, "prefill_chunk")) return;
    generation_capture *c = opaque;
    const int remaining = c->length - c->last;
    const int expected = c->last + (remaining > (int)c->cap ? (int)c->cap : remaining);
    if (remaining <= 0 || total != c->length || current != expected) {
        fprintf(stderr, "Qwen generation: %s chunk %d: frontier=%d/%d, expected=%d/%d cap=%u\n",
                c->path, c->calls, current, total, expected, c->length, c->cap);
        exit(1);
    }
    c->last = current;
    c->calls++;
    if (c->calls == 1 || current == total || current % 2048 == 0)
        fprintf(stderr, "Qwen generation: %s prefill %d/%d cap=%u\n",
                c->path, current, total, c->cap);
}

static void generation_emit(void *opaque, int token) {
    generation_capture *c = opaque;
    generation_need(!c->done && c->count < GENERATION_STEPS, "bounded ordered token emission");
    c->tokens[c->count++] = token;
}

static void generation_done(void *opaque) {
    generation_capture *c = opaque;
    generation_need(c->done == 0, "one completion callback");
    c->done++;
}

static void generation_logits(const float *direct, const float *session, uint32_t cap, int step) {
    const uint32_t vocab = DS4_N_VOCAB;
    uint32_t changed = 0, first = 0, direct_bits = 0, session_bits = 0;
    double worst = 0;
    for (uint32_t i = 0; i < vocab; i++) {
        if (!isfinite(direct[i]) || !isfinite(session[i])) {
            fprintf(stderr, "Qwen generation: cap=%u step=%d token=%u nonfinite logits direct=%g session=%g\n",
                    cap, step, i, direct[i], session[i]);
            exit(1);
        }
        uint32_t a, b;
        memcpy(&a, direct + i, sizeof(a));
        memcpy(&b, session + i, sizeof(b));
        if (a != b) {
            if (!changed) { first = i; direct_bits = a; session_bits = b; }
            changed++;
            worst = fmax(worst, fabs((double)direct[i] - session[i]));
        }
    }
    if (!changed) return;
    fprintf(stderr, "Qwen generation: cap=%u step=%d logits differ: %u/%u, max_error=%.9g, "
            "top1=%d/%d; first token=%u direct=%.9g (0x%08x) session=%.9g (0x%08x)\n",
            cap, step, changed, vocab, worst, sample_argmax(direct, vocab),
            sample_argmax(session, vocab), first, direct[first], direct_bits, session[first], session_bits);
    exit(1);
}

static void generation_case(const char *model, bool streaming, bool prefill_reference,
                            uint32_t requested, int ctx, int tokens) {
    /* Compute the expected public behavior independently of the resolver
     * under test: an explicit request wins over the environment, then clamps. */
    const uint32_t cap = requested < (uint32_t)ctx ? requested : (uint32_t)ctx;
    generation_need(qwen4_prefill_chunk_resolve((uint32_t)ctx, requested) == cap,
                    "explicit prefill chunk overrides environment and clamps to context");
    const int length = tokens ? tokens : (2u * cap + 3u < (uint32_t)(ctx - GENERATION_STEPS - 1) ?
        (int)(2u * cap + 3u) : ctx - GENERATION_STEPS - 1);
    ds4_engine_options opt = {.model_path = model, .backend = DS4_BACKEND_METAL,
        .context_size = ctx, .prefill_chunk = requested, .glm_mtp = false,
        .ssd_streaming = streaming, .ssd_streaming_cache_experts = streaming ? 1024u : 0u};
    ds4_engine *engine = NULL;
    generation_need(ds4_engine_open(&engine, &opt) == 0 && ds4_engine_is_qwen4(engine), "open Qwen model");
    ds4_tokens seed = {0}, prompt = {0};
    ds4_tokenize_text(engine, "Continue this counting sequence: one, two, three, four, five, six, seven, eight. ", &seed);
    generation_need(seed.len > 0, "tokenize fixture");
    for (int i = 0; i < length; i++) ds4_tokens_push(&prompt, seed.v[i % seed.len]);

    generation_capture one_shot = {.path = "one-shot", .cap = cap, .length = length};
    generation_need(ds4_engine_generate_argmax(engine, &prompt, GENERATION_STEPS, ctx,
        generation_emit, generation_done, &one_shot, generation_progress, &one_shot) == 0,
        "public one-shot generation");
    generation_need(one_shot.done == 1 && one_shot.last == length && one_shot.calls > 0,
                    "complete one-shot callbacks");

    ds4_session *session = NULL;
    generation_need(ds4_session_create(&session, engine, ctx) == 0, "create comparison session");
    generation_need(session->qwen4_graph.cap_tokens == cap && !session->qwen4_graph.mtp_R,
                    "session effective chunk and disabled MTP");
    generation_capture session_trace = {.path = "session", .cap = cap, .length = length};
    ds4_session_set_progress(session, generation_progress, &session_trace);
    generation_need(ds4_session_sync(session, &prompt, generation_error, sizeof(generation_error)) == 0,
                    "session prefill");
    ds4_session_set_progress(session, NULL, NULL);
    generation_need(session_trace.last == length && session_trace.calls == one_shot.calls,
                    "matching session chunk frontiers");

    ds4_qwen4_gpu_graph *direct = calloc(1, sizeof(*direct));
    generation_need(direct != NULL, "allocate direct graph descriptor");
    const ds4_context_memory memory = ds4_context_memory_estimate_with_prefill_mode(
        DS4_BACKEND_METAL, ctx, cap, streaming);
    if (streaming) {
        generation_need(qwen4_streaming_memory_admit(engine,
            ds4_add_sat_u64(engine->qwen4_session_bytes, memory.total_bytes), false),
            "admit both bounded comparison graphs");
        engine->qwen4_session_bytes += memory.total_bytes;
    }
    generation_need(qwen4_graph_alloc(direct, &engine->weights, (uint32_t)ctx,
        qwen4_prefill_chunk_resolve((uint32_t)ctx, requested), false, NULL, NULL, NULL, 0),
        "allocate independent direct graph");
    direct->ssd_streaming = streaming;
    qwen4_graph_reset(direct);
    float *direct_logits = malloc((size_t)DS4_N_VOCAB * sizeof(float));
    float *session_logits = malloc((size_t)DS4_N_VOCAB * sizeof(float));
    generation_need(direct_logits && session_logits, "allocate vocabulary comparisons");
    /* Match one-shot's omission of intermediate output heads. Session sync
     * computes each head for its progress/checkpoint contract. The optional
     * reference disables the two arithmetic changes implicated in prefill
     * drift independently of the graph's automatic phase dispatch. */
    if (prefill_reference) {
        generation_need(setenv("DS4_QWEN4_DENSE_MM_LEGACY", "1", 1) == 0 &&
                        setenv("DS4_QWEN4_NO_DENSE_MM_KSPLIT", "1", 1) == 0,
                        "select legacy unsplit reference prefill");
    }
    for (int pos = 0; pos < length;) {
        uint32_t rows = (uint32_t)(length - pos);
        if (rows > direct->cap_tokens) rows = direct->cap_tokens;
        generation_need(qwen4_graph_forward_tokens(direct, &engine->model, &engine->weights,
            prompt.v + pos, rows, pos + (int)rows == length ? direct_logits : NULL, false),
            "direct prefill");
        pos += (int)rows;
    }
    if (prefill_reference) {
        generation_need(unsetenv("DS4_QWEN4_DENSE_MM_LEGACY") == 0 &&
                        unsetenv("DS4_QWEN4_NO_DENSE_MM_KSPLIT") == 0,
                        "restore automatic dispatch for both decodes");
    }

    int emitted = 0;
    bool stopped = false;
    for (int step = 0; step <= GENERATION_STEPS; step++) {
        generation_need(ds4_session_copy_logits(session, session_logits, (int)DS4_N_VOCAB) == (int)DS4_N_VOCAB,
                        "read complete session vocabulary");
        generation_logits(direct_logits, session_logits, cap, step);
        generation_need(direct->pos == (uint32_t)ds4_session_pos(session), "matching decode frontier positions");
        if (step == GENERATION_STEPS) break;
        const int token = ds4_session_argmax(session);
        if (!stopped && ds4_token_is_stop(engine, token)) stopped = true;
        if (!stopped) {
            if (emitted >= one_shot.count || token != one_shot.tokens[emitted]) {
                fprintf(stderr, "Qwen generation: cap=%u step=%d emitted token mismatch: one-shot=%d session=%d\n",
                        cap, step, emitted < one_shot.count ? one_shot.tokens[emitted] : -1, token);
                exit(1);
            }
            emitted++;
        }
        /* Keep all nine vocabulary checks even if this fixture emits a stop.
         * Subsequent rows are teacher-forced ordinary fixture tokens. */
        const int next = stopped ? seed.v[step % seed.len] : token;
        generation_need(ds4_session_eval(session, next, generation_error, sizeof(generation_error)) == 0,
                        "teacher-forced session decode");
        generation_need(qwen4_graph_forward_token(direct, &engine->model, &engine->weights, next, direct_logits),
                        "teacher-forced direct decode");
    }
    generation_need(emitted == one_shot.count, "same public and session greedy token count");
    printf("PASS Qwen generation requested=%u cap=%u ctx=%d prompt=%d chunks=%d tokens=%d: "
           "%d complete vocabulary frontiers bit-exact (%s%s)\n", requested, cap, ctx, length,
           one_shot.calls, emitted, GENERATION_STEPS + 1, streaming ? "SSD" : "resident",
           prefill_reference ? ", legacy unsplit prefill reference" : "");
    free(session_logits); free(direct_logits);
    qwen4_graph_free(direct); free(direct);
    if (streaming) engine->qwen4_session_bytes -= memory.total_bytes;
    ds4_session_free(session);
    ds4_tokens_free(&prompt); ds4_tokens_free(&seed);
    ds4_engine_close(engine);
}

static int generation_number(const char *text, int low, int high) {
    char *end = NULL;
    const long value = strtol(text, &end, 10);
    if (!*text || *end || value < low || value > high) return 0;
    return (int)value;
}

int main(int argc, char **argv) {
    bool streaming = false, prefill_reference = false;
    int chunk = 0, ctx = 0, tokens = 0;
    if (argc < 2) goto usage;
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--ssd-streaming")) streaming = true;
        else if (!strcmp(argv[i], "--prefill-reference")) prefill_reference = true;
        else if (!strcmp(argv[i], "--chunk") && i + 1 < argc) {
            chunk = generation_number(argv[++i], 1, 65536);
            if (!chunk) goto usage;
        } else if (!strcmp(argv[i], "--ctx") && i + 1 < argc) {
            ctx = generation_number(argv[++i], 16, 32768);
            if (!ctx) goto usage;
        } else if (!strcmp(argv[i], "--tokens") && i + 1 < argc) {
            tokens = generation_number(argv[++i], 1, 32768);
            if (!tokens) goto usage;
        } else goto usage;
    }
    if (prefill_reference && !streaming) {
        fputs("Qwen generation: --prefill-reference requires --ssd-streaming; "
              "resident arithmetic uses the main reference, not SSD legacy unsplit\n", stderr);
        return 2;
    }
    if ((chunk == 0) != (ctx == 0)) goto usage;
    if (tokens && (!ctx || tokens > ctx - GENERATION_STEPS - 1)) goto usage;
    /* An inherited diagnostic override could make both sides take the
     * reference path and hide an automatic-dispatch regression. */
    if (prefill_reference && (getenv("DS4_QWEN4_DENSE_MM_LEGACY") ||
                             getenv("DS4_QWEN4_NO_DENSE_MM_KSPLIT"))) {
        fputs("Qwen generation: --prefill-reference requires unsetting "
              "DS4_QWEN4_DENSE_MM_LEGACY and DS4_QWEN4_NO_DENSE_MM_KSPLIT\n", stderr);
        return 2;
    }
    const char *env = getenv("DS4_QWEN4_PREFILL_CHUNK");
    char *saved = env ? strdup(env) : NULL;
    generation_need(!env || saved, "save prefill environment");
    generation_need(setenv("DS4_QWEN4_PREFILL_CHUNK", "32", 1) == 0, "set conflicting chunk fallback");
    if (chunk) generation_case(argv[1], streaming, prefill_reference, (uint32_t)chunk, ctx, tokens);
    else if (prefill_reference) {
        generation_case(argv[1], streaming, true, 32u, 512, 29);
        generation_case(argv[1], streaming, true, 128u, 1024, 575); /* four chunks + a 63-row tail */
        generation_case(argv[1], streaming, true, 128u, 512, 256); /* two full chunks */
    }
    else {
        generation_case(argv[1], streaming, false, 32u, 512, 0);
        generation_case(argv[1], streaming, false, 128u, 512, 0);
        /* An explicit chunk above the context must clamp to ctx, never fall
         * back to the conflicting environment value of 32. */
        generation_case(argv[1], streaming, false, 512u, 256, 0);
    }
    generation_need((saved ? setenv("DS4_QWEN4_PREFILL_CHUNK", saved, 1) :
        unsetenv("DS4_QWEN4_PREFILL_CHUNK")) == 0, "restore prefill environment");
    free(saved);
    return 0;
usage:
    fprintf(stderr, "usage: %s QWEN_GGUF [--ssd-streaming] [--prefill-reference] "
            "[--chunk 1..65536 --ctx 16..32768 [--tokens N]]\n", argv[0]);
    return 2;
}
#else
int main(void) { fputs("Qwen generation test requires Metal\n", stderr); return 2; }
#endif
