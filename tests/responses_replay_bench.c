/*
 * End-to-end /v1/responses continuation ingress benchmark.
 *
 * The timed operation uses a socketpair, client_main(), the resident worker,
 * generate_job(), and server_session_sync(). A synthetic byte-vocabulary
 * engine invokes the production rendered-chat tokenizer; its checkpoint-only
 * test session preserves and appends exact token prefixes but intentionally
 * does not evaluate model weights. The reported host_ingress_ns/op boundary
 * is therefore CPU-side endpoint ingress rather than model or GPU latency; it
 * retains HTTP dispatch, response serialization, response-id lookup, and
 * saved-prefix synchronization.
 *
 * Before response-id support is available, each operation posts the full
 * visible replay. A response-id build first posts that same initial history,
 * then follows the response ids produced by real endpoint replies and posts
 * only a user delta. Without the optional restart probe, the first timed
 * request in both forms represents the same logical conversation state; only
 * the wire form and historical parsing work differ. The restart probe adds one
 * unmeasured continuation solely to validate disk restoration.
 */
#define DS4_SERVER_TEST
#define DS4_SERVER_TEST_NO_MAIN
#include "../ds4_server.c"

#include <dirent.h>
#include <errno.h>

static volatile uint64_t responses_replay_bench_sink;
static int responses_replay_bench_diag_fd = -1;

typedef struct {
    int turns;
    int iterations;
    int disk_cache_mb;
    int continued_interval_tokens;
    int checkpoint_tokens;
    uint64_t checkpoint_bytes;
    bool restart_check;
    bool checkpoint_expectation_set;
    bool checkpoint_expect_saved;
} responses_replay_bench_config;

typedef struct {
    server srv;
    pthread_t worker;
    bool worker_started;
} responses_replay_bench_server;

static void responses_replay_bench_fail(const char *message) {
    int fd = responses_replay_bench_diag_fd >= 0 ?
        responses_replay_bench_diag_fd : STDERR_FILENO;
    dprintf(fd, "responses-replay-bench: %s\n", message);
    exit(1);
}

static double responses_replay_bench_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1000000000.0;
}

static int responses_replay_bench_positive_int(const char *text, const char *name) {
    char *end = NULL;
    errno = 0;
    long value = strtol(text, &end, 10);
    if (errno || !text[0] || !end || *end || value <= 0 || value > INT_MAX) {
        fprintf(stderr, "responses-replay-bench: invalid %s: %s\n", name, text);
        exit(2);
    }
    return (int)value;
}

static uint64_t responses_replay_bench_positive_u64(const char *text,
                                                     const char *name) {
    char *end = NULL;
    errno = 0;
    unsigned long long value = strtoull(text, &end, 10);
    if (errno || !text[0] || !end || *end || value == 0 ||
        (uint64_t)value != value)
    {
        fprintf(stderr, "responses-replay-bench: invalid %s: %s\n", name, text);
        exit(2);
    }
    return (uint64_t)value;
}

static responses_replay_bench_config responses_replay_bench_parse_options(
        int argc, char **argv) {
    responses_replay_bench_config cfg = {
        .turns = 512,
        .iterations = 100,
        .continued_interval_tokens = 8192,
    };
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--turns") && i + 1 < argc) {
            cfg.turns = responses_replay_bench_positive_int(argv[++i], "--turns");
        } else if (!strcmp(argv[i], "--iterations") && i + 1 < argc) {
            cfg.iterations =
                responses_replay_bench_positive_int(argv[++i], "--iterations");
        } else if (!strcmp(argv[i], "--disk-cache-mb") && i + 1 < argc) {
            cfg.disk_cache_mb =
                responses_replay_bench_positive_int(argv[++i], "--disk-cache-mb");
        } else if (!strcmp(argv[i], "--continued-interval-tokens") && i + 1 < argc) {
            cfg.continued_interval_tokens = responses_replay_bench_positive_int(
                argv[++i], "--continued-interval-tokens");
        } else if (!strcmp(argv[i], "--checkpoint-tokens") && i + 1 < argc) {
            cfg.checkpoint_tokens = responses_replay_bench_positive_int(
                argv[++i], "--checkpoint-tokens");
        } else if (!strcmp(argv[i], "--checkpoint-bytes") && i + 1 < argc) {
            cfg.checkpoint_bytes = responses_replay_bench_positive_u64(
                argv[++i], "--checkpoint-bytes");
        } else if (!strcmp(argv[i], "--restart-check")) {
            cfg.restart_check = true;
        } else if (!strcmp(argv[i], "--expect-checkpoint-saved")) {
            if (cfg.checkpoint_expectation_set && !cfg.checkpoint_expect_saved) {
                fprintf(stderr, "responses-replay-bench: conflicting checkpoint expectations\n");
                exit(2);
            }
            cfg.checkpoint_expectation_set = true;
            cfg.checkpoint_expect_saved = true;
        } else if (!strcmp(argv[i], "--expect-checkpoint-skipped")) {
            if (cfg.checkpoint_expectation_set && cfg.checkpoint_expect_saved) {
                fprintf(stderr, "responses-replay-bench: conflicting checkpoint expectations\n");
                exit(2);
            }
            cfg.checkpoint_expectation_set = true;
            cfg.checkpoint_expect_saved = false;
        } else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
            puts("usage: responses_replay_bench [--turns N] [--iterations N]");
            puts("       [--disk-cache-mb N --continued-interval-tokens N]");
            puts("       [--checkpoint-tokens N --checkpoint-bytes N --restart-check]");
            puts("       [--expect-checkpoint-saved|--expect-checkpoint-skipped]");
            exit(0);
        } else {
            fprintf(stderr, "responses-replay-bench: unknown option: %s\n", argv[i]);
            exit(2);
        }
    }
    if ((cfg.checkpoint_tokens || cfg.checkpoint_bytes || cfg.restart_check ||
         cfg.checkpoint_expectation_set) && !cfg.disk_cache_mb)
    {
        fprintf(stderr, "responses-replay-bench: disk checkpoint options require --disk-cache-mb\n");
        exit(2);
    }
    if (cfg.restart_check && cfg.checkpoint_expectation_set &&
        !cfg.checkpoint_expect_saved)
    {
        fprintf(stderr, "responses-replay-bench: restart check requires a retained checkpoint\n");
        exit(2);
    }
    return cfg;
}

static char *responses_replay_bench_full_input(int turns) {
    buf input = {0};
    buf_putc(&input, '[');
    for (int i = 0; i < turns; i++) {
        if (i) buf_putc(&input, ',');
        buf_printf(&input,
                   "{\"type\":\"message\",\"role\":\"user\",\"content\":[{"
                   "\"type\":\"input_text\",\"text\":\"turn %d: explain cache reuse\"}]},"
                   "{\"type\":\"message\",\"role\":\"assistant\",\"content\":[{"
                   "\"type\":\"output_text\",\"text\":\"acknowledged %d\"}]}",
                   i, i);
    }
    buf_putc(&input, ']');
    return buf_take(&input);
}

static void responses_replay_bench_append_delta_user(buf *input, bool *has_item,
                                                      int padding_bytes) {
    if (*has_item) buf_putc(input, ',');
    buf_puts(input,
             "{\"type\":\"message\",\"role\":\"user\",\"content\":[{"
             "\"type\":\"input_text\",\"text\":\"next question");
    for (int i = 0; i < padding_bytes; i++) buf_putc(input, 'x');
    buf_puts(input, "\"}]}");
    *has_item = true;
}

static void responses_replay_bench_append_fixture_assistant(buf *input, bool *has_item) {
    if (*has_item) buf_putc(input, ',');
    buf_puts(input,
             "{\"type\":\"message\",\"role\":\"assistant\",\"content\":[{"
             "\"type\":\"output_text\",\"text\":\"A\"}]}");
    *has_item = true;
}

/* The checkpoint-only session deterministically emits A for one output token.
 * This builds the matching visible replay while keeping request construction
 * outside the timed server round trip. */
static char *responses_replay_bench_growing_input(const char *history,
                                                   int completed_deltas,
                                                   int initial_padding_bytes) {
    size_t len = history ? strlen(history) : 0;
    if (len < 2 || history[0] != '[' || history[len - 1] != ']') {
        responses_replay_bench_fail("invalid full replay fixture");
    }
    buf input = {0};
    buf_append(&input, history, len - 1);
    bool has_item = len > 2;
    for (int i = 0; i < completed_deltas; i++) {
        responses_replay_bench_append_delta_user(
            &input, &has_item, i == 0 ? initial_padding_bytes : 0);
        responses_replay_bench_append_fixture_assistant(&input, &has_item);
    }
    responses_replay_bench_append_delta_user(
        &input, &has_item, completed_deltas == 0 ? initial_padding_bytes : 0);
    buf_putc(&input, ']');
    return buf_take(&input);
}

static char *responses_replay_bench_request_body(const char *input,
                                                  const char *previous_response_id) {
    buf body = {0};
    buf_puts(&body,
             "{\"max_output_tokens\":1,\"reasoning\":{\"effort\":\"none\"},");
    if (previous_response_id && previous_response_id[0]) {
        buf_puts(&body, "\"previous_response_id\":");
        json_escape(&body, previous_response_id);
        buf_putc(&body, ',');
    }
    buf_puts(&body, "\"input\":");
    buf_puts(&body, input ? input : "[]");
    buf_putc(&body, '}');
    return buf_take(&body);
}

static const char *responses_replay_bench_delta_input(void) {
    return "[{\"type\":\"message\",\"role\":\"user\",\"content\":[{"
           "\"type\":\"input_text\",\"text\":\"next question\"}]}]";
}

static void responses_replay_bench_write_all(int fd, const char *data, size_t len) {
    while (len > 0) {
        ssize_t wrote = write(fd, data, len);
        if (wrote < 0 && errno == EINTR) continue;
        if (wrote <= 0) responses_replay_bench_fail("failed to send HTTP request");
        data += wrote;
        len -= (size_t)wrote;
    }
}

static char *responses_replay_bench_http_request(server *s, const char *body) {
    int pair[2] = {-1, -1};
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, pair) != 0) {
        responses_replay_bench_fail("socketpair failed");
    }

    client_arg *arg = xmalloc(sizeof(*arg));
    arg->srv = s;
    arg->fd = pair[1];
    pthread_mutex_lock(&s->mu);
    s->clients++;
    pthread_mutex_unlock(&s->mu);

    pthread_t client;
    if (pthread_create(&client, NULL, client_main, arg) != 0) {
        free(arg);
        pthread_mutex_lock(&s->mu);
        s->clients--;
        pthread_mutex_unlock(&s->mu);
        close(pair[0]);
        close(pair[1]);
        responses_replay_bench_fail("failed to start HTTP client thread");
    }

    buf wire = {0};
    buf_printf(&wire,
               "POST /v1/responses HTTP/1.1\r\n"
               "Host: responses-replay-bench\r\n"
               "Content-Type: application/json\r\n"
               "Content-Length: %zu\r\n"
               "Connection: close\r\n\r\n",
               strlen(body));
    buf_puts(&wire, body);
    responses_replay_bench_write_all(pair[0], wire.ptr, wire.len);
    buf_free(&wire);
    /* Content-Length already delimits the request. A write-side half-close is
     * a client disconnect to the server, so keep this end open until the reply
     * has been read. */

    buf reply = {0};
    char chunk[4096];
    for (;;) {
        ssize_t n = read(pair[0], chunk, sizeof(chunk));
        if (n < 0 && errno == EINTR) continue;
        if (n < 0) {
            close(pair[0]);
            pthread_join(client, NULL);
            responses_replay_bench_fail("failed to read HTTP response");
        }
        if (n == 0) break;
        buf_append(&reply, chunk, (size_t)n);
    }
    close(pair[0]);
    pthread_join(client, NULL);
    return buf_take(&reply);
}

static char *responses_replay_bench_response_id(char *reply) {
    if (!reply || !strstr(reply, " 200 ")) {
        free(reply);
        responses_replay_bench_fail("endpoint did not return HTTP 200");
    }
    const char *body = strstr(reply, "\r\n\r\n");
    body = body ? body + 4 : reply;
    const char *id = strstr(body, "\"id\":\"resp_");
    if (!id) {
        free(reply);
        responses_replay_bench_fail("endpoint reply did not contain a response id");
    }
    id += strlen("\"id\":\"");
    const char *end = strchr(id, '\"');
    if (!end) {
        free(reply);
        responses_replay_bench_fail("endpoint reply had a malformed response id");
    }
    char *result = xstrndup(id, (size_t)(end - id));
    responses_replay_bench_sink += (uint64_t)strlen(reply) + (uint64_t)strlen(result);
    free(reply);
    return result;
}

static char *responses_replay_bench_issue(server *s, const char *body) {
    return responses_replay_bench_response_id(
        responses_replay_bench_http_request(s, body));
}

static char *responses_replay_bench_issue_timed(server *s, const char *body,
                                                 double *elapsed) {
    const double start = responses_replay_bench_now();
    char *reply = responses_replay_bench_http_request(s, body);
    *elapsed += responses_replay_bench_now() - start;
    return responses_replay_bench_response_id(reply);
}

static void responses_replay_bench_server_init(responses_replay_bench_server *bench,
                                                const responses_replay_bench_config *cfg,
                                                const char *disk_dir) {
    memset(bench, 0, sizeof(*bench));
    server *s = &bench->srv;
    s->engine = ds4_test_engine_create_byte_tokenizer();
    if (!s->engine) responses_replay_bench_fail("failed to create tokenizer fixture");
    s->ctx_size = 262144;
    s->default_tokens = 1;
    s->slot_count = 1;
    s->slots = calloc(1, sizeof(*s->slots));
    if (!s->slots) responses_replay_bench_fail("failed to allocate server slot");
    s->slots[0].srv = s;
    s->slots[0].id = 0;
#if defined(DS4_RESPONSE_STATE_DEFAULT_MAX_IDS)
    s->slots[0].frontier_epoch = 1;
#endif
    s->slots[0].session =
        ds4_test_session_create_token_only(s->engine, s->ctx_size);
    if (!s->slots[0].session) responses_replay_bench_fail("failed to create session fixture");
    if (cfg->checkpoint_bytes &&
        !ds4_test_session_set_payload_bytes(s->slots[0].session,
                                            cfg->checkpoint_bytes))
    {
        responses_replay_bench_fail("failed to configure checkpoint payload fixture");
    }

    pthread_mutex_init(&s->tool_mu, NULL);
    pthread_mutex_init(&s->kv_mu, NULL);
    pthread_mutex_init(&s->inference_mu, NULL);
    pthread_mutex_init(&s->model_mu, NULL);
    pthread_mutex_init(&s->mu, NULL);
    pthread_mutex_init(&s->trace_mu, NULL);
    pthread_cond_init(&s->model_cv, NULL);
    pthread_cond_init(&s->cv, NULL);
    pthread_cond_init(&s->clients_cv, NULL);
    if (cfg->disk_cache_mb) {
        kv_cache_options opt = kv_cache_default_options();
        /* The benchmark wants the response-id checkpoint itself, not an
         * unrelated cold checkpoint of the same synthetic test session. */
        opt.cold_max_tokens = 0;
        opt.continued_interval_tokens = cfg->continued_interval_tokens;
        if (!disk_dir || !ds4_kvstore_open(&s->kv, disk_dir,
                                            (uint64_t)cfg->disk_cache_mb,
                                            false, opt,
                                            "responses-replay-bench", NULL, NULL))
        {
            responses_replay_bench_fail("failed to open disk cache fixture");
        }
    }
    if (pthread_create(&bench->worker, NULL, worker_main, s) != 0) {
        responses_replay_bench_fail("failed to start server worker");
    }
    bench->worker_started = true;
}

static void responses_replay_bench_server_close(responses_replay_bench_server *bench) {
    server *s = &bench->srv;
    if (bench->worker_started) {
        pthread_mutex_lock(&s->mu);
        s->stopping = true;
        pthread_cond_broadcast(&s->cv);
        pthread_mutex_unlock(&s->mu);
        pthread_join(bench->worker, NULL);
    }
#if defined(DS4_RESPONSE_STATE_DEFAULT_MAX_IDS)
    response_state_index_free(&s->response_states);
#endif
    if (s->slots) {
        for (int i = 0; i < s->slot_count; i++) ds4_session_free(s->slots[i].session);
    }
    free(s->slots);
    if (s->kv.enabled) ds4_kvstore_close(&s->kv);
    pthread_mutex_destroy(&s->tool_mu);
    pthread_mutex_destroy(&s->kv_mu);
    pthread_mutex_destroy(&s->inference_mu);
    pthread_mutex_destroy(&s->model_mu);
    pthread_mutex_destroy(&s->trace_mu);
    pthread_cond_destroy(&s->model_cv);
    pthread_cond_destroy(&s->clients_cv);
    pthread_cond_destroy(&s->cv);
    pthread_mutex_destroy(&s->mu);
    ds4_test_engine_free_byte_tokenizer(s->engine);
    memset(bench, 0, sizeof(*bench));
}

static void responses_replay_bench_remove_disk_dir(const char *dir) {
    if (!dir || !dir[0]) return;
    DIR *d = opendir(dir);
    if (d) {
        struct dirent *de;
        while ((de = readdir(d)) != NULL) {
            if (!strcmp(de->d_name, ".") || !strcmp(de->d_name, "..")) continue;
            char *path = ds4_kvstore_path_join(dir, de->d_name);
            (void)unlink(path);
            free(path);
        }
        closedir(d);
    }
    (void)rmdir(dir);
}

static int responses_replay_bench_prompt_tokens(server *s, const char *body) {
    request r;
    char err[160] = {0};
    if (!parse_responses_request(s->engine, s, body, s->default_tokens,
                                 s->ctx_size, &r, err, sizeof(err)))
    {
        dprintf(responses_replay_bench_diag_fd >= 0 ? responses_replay_bench_diag_fd :
                STDERR_FILENO,
                "responses-replay-bench: failed to count fixture prompt: %s\n",
                err[0] ? err : "unknown error");
        exit(1);
    }
    const int tokens = r.prompt.len;
    request_free(&r);
    return tokens;
}

static int responses_replay_bench_initial_padding(server *s,
                                                   const char *history,
                                                   int target_tokens) {
    if (target_tokens == 0) return 0;
    int padding = 0;
    for (int attempt = 0; attempt < 3; attempt++) {
        char *input = responses_replay_bench_growing_input(history, 0, padding);
        char *body = responses_replay_bench_request_body(input, NULL);
        const int actual = responses_replay_bench_prompt_tokens(s, body) + 1;
        free(body);
        free(input);
        if (actual == target_tokens) return padding;
        if (actual > target_tokens || target_tokens - actual > INT_MAX - padding) {
            responses_replay_bench_fail("requested checkpoint token shape is smaller than the fixture");
        }
        /* The byte tokenizer has no merge table, so each x appends exactly one
         * rendered token. Retain the short retry for future fixture changes. */
        padding += target_tokens - actual;
    }
    responses_replay_bench_fail("failed to shape checkpoint token fixture");
    return 0;
}

static uint64_t responses_replay_bench_payload_save_count(const server *s) {
    return s && s->slots && s->slots[0].session ?
        ds4_test_session_payload_save_count(s->slots[0].session) : 0;
}

static int responses_replay_bench_check_initial_checkpoint(
        server *s, const char *id, const responses_replay_bench_config *cfg) {
#if defined(DS4_RESPONSE_STATE_DEFAULT_MAX_IDS)
    response_state_entry *entry = response_state_acquire(s, id);
    if (!entry) responses_replay_bench_fail("initial response state was not indexed");
    const bool saved = response_state_checkpoint_saved(s, entry);
    const int live_tokens = entry->live_tokens;
    response_state_release(s, entry);
    if (cfg->checkpoint_tokens && live_tokens != cfg->checkpoint_tokens) {
        responses_replay_bench_fail("initial response did not reach requested checkpoint token shape");
    }
    if (!cfg->disk_cache_mb) return 0;
    const bool expected = cfg->checkpoint_expectation_set ?
        cfg->checkpoint_expect_saved : true;
    if (saved != expected) {
        responses_replay_bench_fail(expected ?
            "expected retained response-id checkpoint was not saved" :
            "oversized response-id checkpoint unexpectedly persisted");
    }
    const uint64_t payload_saves = responses_replay_bench_payload_save_count(s);
    if (payload_saves != (expected ? 1u : 0u)) {
        responses_replay_bench_fail(expected ?
            "retained response-id checkpoint staged an unexpected payload count" :
            "oversized response-id checkpoint staged a payload before budget rejection");
    }
    const int boundary = s->slots[0].response_state_last_checkpoint_tokens;
    if (boundary <= 0) {
        responses_replay_bench_fail("response-id checkpoint did not record its cadence boundary");
    }
    return boundary;
#else
    (void)s;
    (void)id;
    (void)cfg;
    return 0;
#endif
}

static void responses_replay_bench_check_tail_live_only(
        server *s, const char *id, int initial_boundary,
        uint64_t expected_payload_saves,
        const responses_replay_bench_config *cfg) {
#if defined(DS4_RESPONSE_STATE_DEFAULT_MAX_IDS)
    if (!cfg->disk_cache_mb) return;
    response_state_entry *entry = response_state_acquire(s, id);
    if (!entry) responses_replay_bench_fail("tail response state was not indexed");
    const bool saved = response_state_checkpoint_saved(s, entry);
    response_state_release(s, entry);
    if (saved) {
        responses_replay_bench_fail("short response unexpectedly wrote another response-id checkpoint");
    }
    if (s->slots[0].response_state_last_checkpoint_tokens != initial_boundary) {
        responses_replay_bench_fail("short response unexpectedly advanced response-id checkpoint cadence");
    }
    if (responses_replay_bench_payload_save_count(s) != expected_payload_saves) {
        responses_replay_bench_fail("short response unexpectedly staged another checkpoint payload");
    }
#else
    (void)s;
    (void)id;
    (void)initial_boundary;
    (void)expected_payload_saves;
    (void)cfg;
#endif
}

static int responses_replay_bench_quiet_stderr(void) {
    fflush(stderr);
    int saved = dup(STDERR_FILENO);
    int nullfd = open("/dev/null", O_WRONLY);
    if (saved < 0 || nullfd < 0 || dup2(nullfd, STDERR_FILENO) < 0) {
        if (saved >= 0) close(saved);
        if (nullfd >= 0) close(nullfd);
        responses_replay_bench_fail("failed to silence endpoint logs");
    }
    close(nullfd);
    responses_replay_bench_diag_fd = saved;
    return saved;
}

static void responses_replay_bench_restore_stderr(int saved) {
    fflush(stderr);
    if (saved >= 0) {
        (void)dup2(saved, STDERR_FILENO);
        close(saved);
    }
    responses_replay_bench_diag_fd = -1;
}

static double responses_replay_bench_full_replay(server *s, const char *history,
                                                  int initial_padding_bytes,
                                                  int iterations) {
    char *warmup_input = responses_replay_bench_growing_input(
        history, 0, initial_padding_bytes);
    char *warmup_body = responses_replay_bench_request_body(warmup_input, NULL);
    char *warmup = responses_replay_bench_issue(s, warmup_body);
    free(warmup);
    free(warmup_body);
    free(warmup_input);

    int saved_stderr = responses_replay_bench_quiet_stderr();
    double elapsed = 0.0;
    for (int i = 0; i < iterations; i++) {
        char *input = responses_replay_bench_growing_input(
            history, i + 1, initial_padding_bytes);
        char *body = responses_replay_bench_request_body(input, NULL);
        char *response_id = responses_replay_bench_issue_timed(s, body, &elapsed);
        free(response_id);
        free(body);
        free(input);
    }
    responses_replay_bench_restore_stderr(saved_stderr);
    return elapsed;
}

#if defined(DS4_RESPONSE_STATE_DEFAULT_MAX_IDS) && !defined(DS4_RESPONSES_REPLAY_BENCH_FORCE_FULL)
static char *responses_replay_bench_issue_initial(server *s, const char *initial_body,
                                                   const responses_replay_bench_config *cfg,
                                                   int *checkpoint_boundary,
                                                   uint64_t *payload_saves) {
    char *id = responses_replay_bench_issue(s, initial_body);
    if (checkpoint_boundary) {
        *checkpoint_boundary = responses_replay_bench_check_initial_checkpoint(s, id, cfg);
    }
    if (payload_saves) {
        *payload_saves = responses_replay_bench_payload_save_count(s);
    }
    return id;
}

static char *responses_replay_bench_resume_after_restart(
        server *s, const char *previous_id, int checkpoint_boundary,
        const responses_replay_bench_config *cfg) {
    const int before = ds4_session_pos(s->slots[0].session);
    if (before != 0) responses_replay_bench_fail("restarted fixture retained an in-memory session");
    char *body = responses_replay_bench_request_body(
        responses_replay_bench_delta_input(), previous_id);
    char *next_id = responses_replay_bench_issue(s, body);
    free(body);
    if (ds4_session_pos(s->slots[0].session) <= before) {
        free(next_id);
        responses_replay_bench_fail("retained response id did not reload after restart");
    }
    responses_replay_bench_check_tail_live_only(
        s, next_id, checkpoint_boundary, 0, cfg);
    return next_id;
}

static double responses_replay_bench_response_id_tail(
        server *s, char *previous_id, int checkpoint_boundary,
        uint64_t expected_payload_saves,
        const responses_replay_bench_config *cfg) {
    int saved_stderr = responses_replay_bench_quiet_stderr();
    double elapsed = 0.0;
    for (int i = 0; i < cfg->iterations; i++) {
        char *body = responses_replay_bench_request_body(
            responses_replay_bench_delta_input(), previous_id);
        const int before = ds4_session_pos(s->slots[0].session);
        char *next_id = responses_replay_bench_issue_timed(s, body, &elapsed);
        free(body);
        free(previous_id);
        previous_id = next_id;
        if (ds4_session_pos(s->slots[0].session) <= before) {
            free(previous_id);
            responses_replay_bench_restore_stderr(saved_stderr);
            responses_replay_bench_fail("response-id hit did not advance the saved prefix");
        }
        responses_replay_bench_check_tail_live_only(
            s, previous_id, checkpoint_boundary, expected_payload_saves, cfg);
    }
    responses_replay_bench_restore_stderr(saved_stderr);
    free(previous_id);
    return elapsed;
}
#endif

int main(int argc, char **argv) {
    responses_replay_bench_config cfg = responses_replay_bench_parse_options(argc, argv);
    char disk_template[] = "/tmp/ds4-responses-replay-XXXXXX";
    char *disk_dir = NULL;
    if (cfg.disk_cache_mb) {
        if (!mkdtemp(disk_template)) responses_replay_bench_fail("failed to create disk cache directory");
        disk_dir = xstrdup(disk_template);
    }

    char *full_input = responses_replay_bench_full_input(cfg.turns);
    responses_replay_bench_server bench;
    responses_replay_bench_server_init(&bench, &cfg, disk_dir);
    const int initial_padding = responses_replay_bench_initial_padding(
        &bench.srv, full_input, cfg.checkpoint_tokens);
    char *initial_input = responses_replay_bench_growing_input(
        full_input, 0, initial_padding);
    char *initial_body = responses_replay_bench_request_body(initial_input, NULL);

    double elapsed;
#if defined(DS4_RESPONSE_STATE_DEFAULT_MAX_IDS) && !defined(DS4_RESPONSES_REPLAY_BENCH_FORCE_FULL)
    int checkpoint_boundary = 0;
    uint64_t payload_saves = 0;
    char *previous_id = responses_replay_bench_issue_initial(
        &bench.srv, initial_body, &cfg, &checkpoint_boundary, &payload_saves);
    if (cfg.restart_check) {
        responses_replay_bench_server_close(&bench);
        responses_replay_bench_server_init(&bench, &cfg, disk_dir);
        char *resumed_id = responses_replay_bench_resume_after_restart(
            &bench.srv, previous_id, checkpoint_boundary, &cfg);
        free(previous_id);
        previous_id = resumed_id;
        payload_saves = 0;
    }
    elapsed = responses_replay_bench_response_id_tail(
        &bench.srv, previous_id, checkpoint_boundary, payload_saves, &cfg);
#else
    elapsed = responses_replay_bench_full_replay(
        &bench.srv, full_input, initial_padding, cfg.iterations);
#endif

    responses_replay_bench_server_close(&bench);
    responses_replay_bench_remove_disk_dir(disk_dir);
    free(disk_dir);
    free(initial_body);
    free(initial_input);
    free(full_input);
    if (responses_replay_bench_sink == 0) return 1;
    printf("{\"metric\":\"host_ingress_ns/op\",\"value\":%.0f}\n",
           elapsed * 1000000000.0 / (double)cfg.iterations);
    return 0;
}
