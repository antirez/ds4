/* Real-model PLE prefetch, cancellation and fallback regression.
 * Usage: test_qwen4_ple_prefetch QWEN_GGUF */
#include <pthread.h>
static int test_pthread_create(pthread_t *, const pthread_attr_t *, void *(*)(void *), void *);
#define pthread_create test_pthread_create
#include "../ds4.c"
#undef pthread_create
#include <assert.h>

enum { CHUNK = 256, PROMPT = 2 * CHUNK + 17, FRONTIERS = 3 };
enum { CONTROL, PREFETCH, CANCEL, READ_FAILURE, THREAD_FAILURE };
static int mode, started, finished, failed;

static void *prefetch_worker(void *arg) {
    qwen4_ple_prefetch *p = arg;
    const ds4_model *model = p->model;
    ds4_model bad = *model;
    /* Fail only the first background read. The session's model and fd stay
     * untouched, so the real synchronous retry can succeed. */
    if (mode == READ_FAILURE && finished == 0) {
        bad.ngram_fd = INT_MAX;
        p->model = &bad;
    }
    qwen4_ple_prefetch_run(p);
    p->model = model;
    if (!p->ok) failed++;
    finished++;
    return NULL;
}

static int test_pthread_create(pthread_t *thread, const pthread_attr_t *attr,
                               void *(*fn)(void *), void *arg) {
    /* All unrelated engine and n-gram reader threads use the real API. */
    if (fn != qwen4_ple_prefetch_run) return pthread_create(thread, attr, fn, arg);
    if (mode == THREAD_FAILURE) { failed++; return EAGAIN; }
    int rc = pthread_create(thread, attr, prefetch_worker, arg);
    if (!rc) started++;
    return rc;
}

typedef struct {
    ds4_session *session;
    float *expected;
    size_t bytes;
    int calls;
    bool cancel;
} progress;

static void report(void *ud, const char *event, int current, int total) {
    if (strcmp(event, "prefill_chunk")) return;
    progress *p = ud;
    const int frontiers[] = {CHUNK, 2 * CHUNK, PROMPT};
    assert(p->calls < FRONTIERS && current == frontiers[p->calls] && total == PROMPT);
    assert(p->session->checkpoint_valid && ds4_session_pos(p->session) == current);
    /* The session must join its worker before exposing a durable frontier. */
    assert(started == finished);
    for (uint32_t i = 0; i < DS4_N_VOCAB; i++) assert(isfinite(p->session->logits[i]));
    float *expected = p->expected + (size_t)p->calls * DS4_N_VOCAB;
    if (mode == CONTROL) memcpy(expected, p->session->logits, p->bytes);
    else assert(!memcmp(expected, p->session->logits, p->bytes));
    p->calls++;
    if (mode == CANCEL && p->calls == 1) p->cancel = true;
}

static bool cancelled(void *ud) { return ((progress *)ud)->cancel; }

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s QWEN_GGUF\n", argv[0]);
        return 2;
    }
    ds4_engine_options opt = {.model_path = argv[1], .context_size = PROMPT + 8,
        .prefill_chunk = CHUNK,
#ifdef __APPLE__
        .backend = DS4_BACKEND_METAL
#else
        .backend = DS4_BACKEND_CUDA
#endif
    };
    ds4_engine *engine = NULL;
    assert(ds4_engine_open(&engine, &opt) == 0);
    assert(ds4_engine_is_qwen4(engine) && engine->model.ngram_tensor);
    ds4_tokens tokens = {0};
    ds4_tokenize_text(engine, "The quick brown fox counts 17 green apples, then 29 blue boats.\n", &tokens);
    const int seed = tokens.len;
    assert(seed > 0 && seed < CHUNK);
    while (tokens.len < PROMPT) ds4_tokens_push(&tokens, tokens.v[tokens.len % seed]);
    const size_t bytes = DS4_N_VOCAB * sizeof(float);
    float *expected = xmalloc((FRONTIERS + 1) * bytes);
    int next = -1;
    const char *names[] = {"rollback", "prefetch", "cancel/resume", "read fallback", "thread fallback"};
    for (mode = CONTROL; mode <= THREAD_FAILURE; mode++) {
        if (mode == CONTROL) assert(setenv("DS4_QWEN4_NO_PLE_PREFETCH", "1", 1) == 0);
        else assert(unsetenv("DS4_QWEN4_NO_PLE_PREFETCH") == 0);
        ds4_session *session = NULL;
        assert(ds4_session_create(&session, engine, PROMPT + 8) == 0);
        assert(session->qwen4_graph.cap_tokens == CHUNK);
        started = finished = failed = 0;
        progress p = {.session = session, .expected = expected, .bytes = bytes};
        ds4_session_set_progress(session, report, &p);
        ds4_session_set_cancel(session, cancelled, &p);
        char error[256] = {0};
        int rc = ds4_session_sync(session, &tokens, error, sizeof(error));
        if (mode == CANCEL) {
            assert(rc == DS4_SESSION_SYNC_INTERRUPTED && p.calls == 1);
            assert(started == 1 && finished == 1);
            assert(session->checkpoint_valid && ds4_session_pos(session) == CHUNK);
            p.cancel = false;
            rc = ds4_session_sync(session, &tokens, error, sizeof(error));
        }
        if (rc) fprintf(stderr, "%s: %s\n", names[mode], error);
        assert(rc == 0 && p.calls == FRONTIERS);
        assert(started == (mode == CONTROL || mode == THREAD_FAILURE ? 0 : 2));
        assert(finished == started);
        assert(failed == (mode == READ_FAILURE ? 1 : mode == THREAD_FAILURE ? 2 : 0));
        ds4_session_set_progress(session, NULL, NULL);
        ds4_session_set_cancel(session, NULL, NULL);
        /* Exact continuation also checks the live PLE/recurrent history after
         * consuming prefetched rows, discarding them on cancel, or retrying. */
        if (mode == CONTROL) next = ds4_session_argmax(session);
        assert(ds4_session_eval(session, next, error, sizeof(error)) == 0);
        float *decode = expected + (size_t)FRONTIERS * DS4_N_VOCAB;
        if (mode == CONTROL) memcpy(decode, session->logits, bytes);
        else assert(!memcmp(decode, session->logits, bytes));
        ds4_session_free(session);
        printf("Qwen PLE %s: exact chunk and decode logits OK\n", names[mode]);
        fflush(stdout);
    }
    free(expected);
    ds4_tokens_free(&tokens);
    ds4_engine_close(engine);
    return 0;
}
