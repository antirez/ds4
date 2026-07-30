#include "pulsar_server_internal.h"
#include "pulsar_lock.hpp"



static void http_request_free(http_request *r) {
    free(r->body);
    memset(r, 0, sizeof(*r));
}



static ssize_t header_end(const char *p, size_t n) {
    for (size_t i = 3; i < n; i++) {
        if (p[i - 3] == '\r' && p[i - 2] == '\n' && p[i - 1] == '\r' && p[i] == '\n') return (ssize_t)(i + 1);
    }
    for (size_t i = 1; i < n; i++) {
        if (p[i - 1] == '\n' && p[i] == '\n') return (ssize_t)(i + 1);
    }
    return -1;
}



/* Body length from the request headers. Returns -1 (reject) on a duplicate
 * Content-Length, a non-numeric value, or any Transfer-Encoding header: each
 * connection is Connection: close today so smuggling is only latent, but the
 * framing must stay unambiguous if keep-alive is ever added. */
static long content_length(const char *h, size_t n) {
    const char *p = h, *end = h + n;
    long clen = 0;
    int seen = 0;
    while (p < end) {
        const char *line = p;
        while (p < end && *p != '\n') p++;
        size_t len = (size_t)(p - line);
        if (len && line[len - 1] == '\r') len--;
        if (len >= 18 && strncasecmp(line, "Transfer-Encoding:", 18) == 0) {
            return -1;
        }
        if (len >= 15 && strncasecmp(line, "Content-Length:", 15) == 0) {
            const char *v = line + 15;
            while (v < line + len && isspace((unsigned char)*v)) v++;
            char *vend = NULL;
            long parsed = strtol(v, &vend, 10);
            if (vend == v || parsed < 0) return -1;
            if (seen && parsed != clen) return -1;
            seen = 1;
            clen = parsed;
        }
        if (p < end) p++;
    }
    return clen;
}



static bool read_http_request(int fd, http_request *r) {
    buf b = {0};
    ssize_t hend = -1;
    const size_t max_header = 64 * 1024;
    const size_t max_body = 64 * 1024 * 1024;
    /* Whole-request read deadline: SO_RCVTIMEO only bounds a single recv, so
     * a client trickling one byte per interval could hold this thread
     * forever. The deadline bounds total arrival time for headers + body. */
    const time_t deadline = time(NULL) + PULSAR_SERVER_REQUEST_READ_DEADLINE_SEC;

    while (hend < 0 && b.len < max_header) {
        char tmp[4096];
        ssize_t n = recv(fd, tmp, sizeof(tmp), 0);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) goto fail;
        if (time(NULL) > deadline) goto fail;
        buf_append(&b, tmp, (size_t)n);
        hend = header_end(b.ptr, b.len);
    }
    if (hend < 0) goto fail;

    char line[512];
    size_t i;
    i = 0;
    while (i < b.len && b.ptr[i] != '\n' && i + 1 < sizeof(line)) {
        line[i] = b.ptr[i];
        i++;
    }
    line[i] = '\0';
    if (sscanf(line, "%7s %255s", r->method, r->path) != 2) goto fail;
    char *q;
    q = strchr(r->path, '?');
    if (q) *q = '\0';

    long clen;
    clen = content_length(b.ptr, (size_t)hend);
    if (clen < 0 || (size_t)clen > max_body) goto fail;
    while (b.len < (size_t)hend + (size_t)clen) {
        char tmp[8192];
        ssize_t n = recv(fd, tmp, sizeof(tmp), 0);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) goto fail;
        if (time(NULL) > deadline) goto fail;
        buf_append(&b, tmp, (size_t)n);
    }

    r->body_len = (size_t)clen;
    r->body = (char *)server_xmalloc(r->body_len + 1);
    memcpy(r->body, b.ptr + hend, r->body_len);
    r->body[r->body_len] = '\0';
    buf_free(&b);
    return true;
fail:
    buf_free(&b);
    return false;
}



void append_model_json_values(buf *b, const char *id, const char *name,
                                     int ctx, int default_tokens) {
    const int max_completion = default_tokens < ctx ? default_tokens : ctx;
    buf_printf(b,
        "{\"id\":");
    json_escape(b, id);
    /* vLLM convention: "root" is the model path HF-tooling resolves (e.g.
     * llama-benchy uses it/the id as the tokenizer id). id == root here. */
    buf_puts(b, ",\"root\":");
    json_escape(b, id);
    buf_puts(b,
        ",\"object\":\"model\","
        "\"created\":1767225600,"
        "\"owned_by\":\"pulsar\","
        "\"name\":");
    json_escape(b, name);
    buf_printf(b,
        ","
        "\"context_length\":%d,"
        "\"top_provider\":{"
            "\"context_length\":%d,"
            "\"max_completion_tokens\":%d,"
            "\"is_moderated\":false},"
        "\"supported_parameters\":["
            "\"tools\","
            "\"tool_choice\","
            "\"max_tokens\","
            "\"temperature\","
            "\"top_p\","
            "\"top_k\","
            "\"min_p\","
            "\"stop\","
            "\"seed\","
            "\"stream\","
            "\"reasoning_effort\"]}",
        ctx,
        ctx,
        max_completion);
}



static void append_model_json(buf *b, const server *s, const char *id) {
    append_model_json_values(b,
                             id,
                             server_served_model_name(s),
                             pulsar_session_ctx(s->sess),
                             s->default_tokens);
}



bool server::send_model(int fd, const char *id) {
    auto *s = this;
    buf b = {0};
    append_model_json(&b, s, id);
    buf_putc(&b, '\n');
    bool ok = http_response(fd, s->enable_cors, 200, "application/json", b.ptr);
    buf_free(&b);
    return ok;
}



bool server::send_models(int fd) {
    auto *s = this;
    /* Advertise only the model actually loaded (its shape id), not both
     * flash/pro aliases — the server serves one GGUF at a time. */
    buf b = {0};
    buf_puts(&b, "{\"object\":\"list\",\"data\":[");
    append_model_json(&b, s, server_served_model_id(s));
    buf_puts(&b, "]}\n");
    bool ok = http_response(fd, s->enable_cors, 200, "application/json", b.ptr);
    buf_free(&b);
    return ok;
}

/* Liveness probe (/healthz, /ping): is the process alive at all? Always 200
 * while the process runs — deliberately independent of readiness/drain state,
 * so a k8s liveness probe never restarts a server that is merely draining.
 * Lock-free, engine-free (safe on a client thread). */
bool server::send_liveness(int fd) {
    auto *s = this;
    return http_response(fd, s->enable_cors, 200, "application/json",
                         "{\"status\":\"ok\"}\n");
}

/* Readiness + status (/health): is the server ready to accept work, and what
 * is it doing right now? 200 {"status":"ok",...} when serving; 503
 * {"status":"draining",...} once shutdown has been requested so a load
 * balancer stops routing to it. Reads only the worker-published snapshot under
 * mu (same discipline as /metrics — no engine calls on the client thread). */
bool server::send_health(int fd) {
    auto *s = this;
    const char *model = server_served_model_id(s);
    bool draining;
    int n_slots, running, waiting;
    time_t started;
    double kv = 0.0; /* max KV utilization across provisioned slots */
    pthread_mutex_lock(&s->mu);
    draining = s->stopping;
    n_slots  = s->n_slots;
    running  = s->n_generating;
    waiting  = s->n_queued;
    started  = s->started;
    for (int i = 0; i < n_slots; i++) {
        const int pos = s->m_slot_pos[i];
        const int ctx = s->m_slot_ctx[i];
        double u = (ctx > 0 && pos > 0) ? (double)pos / (double)ctx : 0.0;
        if (u > 1.0) u = 1.0;
        if (u > kv) kv = u;
    }
    pthread_mutex_unlock(&s->mu);
    if (running < 0) running = 0;
    if (waiting < 0) waiting = 0;
    const long uptime = started ? (long)(time(NULL) - started) : 0;

    buf b = {0};
    buf_printf(&b,
        "{\"status\":\"%s\",\"version\":\"%s\",\"model\":\"%s\","
        "\"uptime_s\":%ld,\"slots\":{\"total\":%d,\"running\":%d,\"waiting\":%d},"
        "\"kv_cache_usage\":%.6f}\n",
        draining ? "draining" : "ok", PULSAR_VERSION_STR, model,
        uptime, n_slots, running, waiting, kv);
    bool ok = http_response(fd, s->enable_cors, draining ? 503 : 200,
                            "application/json", b.ptr);
    buf_free(&b);
    return ok;
}

/* Version + build identity (/version), vLLM/OpenAI convention. Version is the
 * git-describe string baked in at build time (see Makefile). */
bool server::send_version(int fd) {
    auto *s = this;
    buf b = {0};
    buf_printf(&b,
        "{\"version\":\"%s\",\"engine\":\"pulsar\",\"cuda_arch\":\"sm_120f\","
        "\"model\":\"%s\",\"model_name\":\"%s\",\"context\":%d}\n",
        PULSAR_VERSION_STR,
        server_served_model_id(s),
        server_served_model_name(s),
        pulsar_session_ctx(s->sess));
    bool ok = http_response(fd, s->enable_cors, 200, "application/json", b.ptr);
    buf_free(&b);
    return ok;
}

/* Root banner so a bare GET / (browsers, uptime probes) gets a 200 with the
 * version and a pointer to the real endpoints instead of a 404. */
bool server::send_root(int fd) {
    auto *s = this;
    buf b = {0};
    buf_printf(&b,
        "{\"service\":\"pulsar-server\",\"version\":\"%s\",\"status\":\"ok\","
        "\"endpoints\":[\"/health\",\"/version\",\"/v1/models\","
        "\"/v1/chat/completions\",\"/v1/completions\",\"/v1/messages\","
        "\"/v1/responses\",\"/metrics\"]}\n",
        PULSAR_VERSION_STR);
    bool ok = http_response(fd, s->enable_cors, 200, "application/json", b.ptr);
    buf_free(&b);
    return ok;
}

/* Prometheus /metrics — DSpark speculative-decode counters in vLLM naming, so
 * tool-eval-bench --spec-live (and any vLLM-oriented scraper) reads acceptance
 * rate, acceptance length, and the per-position waterfall unchanged. All
 * counters are cumulative since engine open; gauges are point-in-time. */
bool server::send_metrics(int fd) {
    auto *s = this;
    /* This runs on a client thread: it must not call into the engine
     * (CUDA-state audit, pulsar_server_internal.h). Everything below reads the
     * snapshots the worker publishes under mu (m_spec/m_slot_pos/m_slot_ctx,
     * server_publish_metrics_snapshot — refreshed at bind time and once per
     * quantum, so gauges lag live state by at most one quantum). */
    const char *model = server_served_model_id(s);
    pulsar_spec_metrics m;
    int n_slots;
    double slot_kv[PULSAR_SESSION_POOL_CAP];
    pthread_mutex_lock(&s->mu);
    m = s->m_spec;
    n_slots = s->n_slots;
    for (int i = 0; i < n_slots; i++) {
        const int pos = s->m_slot_pos[i];
        const int ctx = s->m_slot_ctx[i];
        double kv = (ctx > 0 && pos > 0) ? (double)pos / (double)ctx : 0.0;
        slot_kv[i] = kv > 1.0 ? 1.0 : kv;
    }
    int running = s->n_generating;
    int waiting = s->n_queued;
    unsigned long long prompt_toks = (unsigned long long)s->m_prompt_tokens;
    unsigned long long pfx_queries = (unsigned long long)s->m_prefix_queries;
    unsigned long long pfx_hits = (unsigned long long)s->m_prefix_hits;
    pthread_mutex_unlock(&s->mu);
    if (running < 0) running = 0;
    if (waiting < 0) waiting = 0;
    double kv = 0.0; /* unlabeled gauge: max across provisioned slots */
    for (int i = 0; i < n_slots; i++) {
        if (slot_kv[i] > kv) kv = slot_kv[i];
    }

    buf b = {0};
    /* Spec-decode counters (the core of --spec-live). model_name label lets the
     * scraper detect the drafter identity and the spec_decode method. */
    buf_puts(&b, "# HELP vllm:spec_decode_num_draft_tokens_total Cumulative draft tokens proposed.\n");
    buf_puts(&b, "# TYPE vllm:spec_decode_num_draft_tokens_total counter\n");
    buf_printf(&b, "vllm:spec_decode_num_draft_tokens_total{model_name=\"%s\"} %llu\n",
               model, (unsigned long long)m.draft_tokens);
    buf_puts(&b, "# HELP vllm:spec_decode_num_accepted_tokens_total Cumulative draft tokens accepted.\n");
    buf_puts(&b, "# TYPE vllm:spec_decode_num_accepted_tokens_total counter\n");
    buf_printf(&b, "vllm:spec_decode_num_accepted_tokens_total{model_name=\"%s\"} %llu\n",
               model, (unsigned long long)m.accepted_tokens);
    buf_puts(&b, "# HELP vllm:spec_decode_num_drafts_total Cumulative draft rounds.\n");
    buf_puts(&b, "# TYPE vllm:spec_decode_num_drafts_total counter\n");
    buf_printf(&b, "vllm:spec_decode_num_drafts_total{model_name=\"%s\"} %llu\n",
               model, (unsigned long long)m.num_drafts);
    /* Per-position accepted counters -> the scraper derives per-position
     * acceptance = count/num_drafts (the waterfall). Emit 0..max_draft-1 so the
     * chart has a full row even before every position has fired. */
    if (m.max_draft > 0) {
        int np = m.max_draft > 16 ? 16 : m.max_draft;
        buf_puts(&b, "# HELP vllm:spec_decode_num_accepted_tokens_per_pos_total Accepted count per draft position.\n");
        buf_puts(&b, "# TYPE vllm:spec_decode_num_accepted_tokens_per_pos_total counter\n");
        for (int i = 0; i < np; i++)
            buf_printf(&b, "vllm:spec_decode_num_accepted_tokens_per_pos_total{model_name=\"%s\",position=\"%d\"} %llu\n",
                       model, i, (unsigned long long)m.accepted_per_pos[i]);
    }
    /* Token-throughput counters (the scraper derives prompt/gen t/s from deltas). */
    buf_puts(&b, "# HELP vllm:prompt_tokens_total Cumulative prompt tokens prefilled.\n");
    buf_puts(&b, "# TYPE vllm:prompt_tokens_total counter\n");
    buf_printf(&b, "vllm:prompt_tokens_total %llu\n", prompt_toks);
    buf_puts(&b, "# HELP vllm:generation_tokens_total Cumulative tokens emitted.\n");
    buf_puts(&b, "# TYPE vllm:generation_tokens_total counter\n");
    buf_printf(&b, "vllm:generation_tokens_total %llu\n", (unsigned long long)m.gen_tokens);
    /* Prefix-cache hit rate (scraper computes hits/queries). */
    buf_puts(&b, "# HELP vllm:prefix_cache_queries_total Cumulative prompt tokens looked up in the prefix cache.\n");
    buf_puts(&b, "# TYPE vllm:prefix_cache_queries_total counter\n");
    buf_printf(&b, "vllm:prefix_cache_queries_total %llu\n", pfx_queries);
    buf_puts(&b, "# HELP vllm:prefix_cache_hits_total Cumulative prompt tokens served from the prefix cache.\n");
    buf_puts(&b, "# TYPE vllm:prefix_cache_hits_total counter\n");
    buf_printf(&b, "vllm:prefix_cache_hits_total %llu\n", pfx_hits);
    /* Scheduler + KV gauges. The unlabeled kv_cache_usage_perc series is the
     * MAX across provisioned slots (single-gauge scrapers keep working and
     * see the most-loaded session); the slot-labeled series break it out
     * per session. */
    buf_puts(&b, "# HELP vllm:kv_cache_usage_perc KV cache utilization (0-1); unlabeled = max across slots.\n");
    buf_puts(&b, "# TYPE vllm:kv_cache_usage_perc gauge\n");
    buf_printf(&b, "vllm:kv_cache_usage_perc %.6f\n", kv);
    for (int i = 0; i < n_slots; i++) {
        buf_printf(&b, "vllm:kv_cache_usage_perc{slot=\"%d\"} %.6f\n", i, slot_kv[i]);
    }
    buf_puts(&b, "# HELP vllm:num_requests_running Requests currently generating.\n");
    buf_puts(&b, "# TYPE vllm:num_requests_running gauge\n");
    buf_printf(&b, "vllm:num_requests_running %d\n", running);
    buf_puts(&b, "# HELP vllm:num_requests_waiting Requests queued and not yet started.\n");
    buf_puts(&b, "# TYPE vllm:num_requests_waiting gauge\n");
    buf_printf(&b, "vllm:num_requests_waiting %d\n", waiting);

    bool ok = http_response(fd, s->enable_cors, 200, "text/plain; version=0.0.4", b.ptr);
    buf_free(&b);
    return ok;
}



void server::client_done() {
    auto *s = this;
    pthread_mutex_lock(&s->mu);
    if (s->clients > 0) s->clients--;
    pthread_cond_broadcast(&s->clients_cv);
    pthread_mutex_unlock(&s->mu);
}



void set_client_socket_nonblocking(int fd);



void *client_main(void *arg) {
    client_arg *ca = (client_arg *)arg;
    server *s = ca->srv;
    int fd = ca->fd;
    free(ca);

    http_request hr = {0};
    if (!read_http_request(fd, &hr)) {
        http_error(fd, s->enable_cors, 400, "bad HTTP request");
        goto done;
    }

    if (!strcmp(hr.method, "OPTIONS")) {
        http_response(fd, s->enable_cors, 204, NULL, "");
        http_request_free(&hr);
        goto done;
    }

    if (!strcmp(hr.method, "GET") && !strcmp(hr.path, "/health")) {
        s->send_health(fd);
        http_request_free(&hr);
        goto done;
    }
    if (!strcmp(hr.method, "GET") &&
        (!strcmp(hr.path, "/healthz") || !strcmp(hr.path, "/ping"))) {
        s->send_liveness(fd);
        http_request_free(&hr);
        goto done;
    }
    if (!strcmp(hr.method, "GET") && !strcmp(hr.path, "/version")) {
        s->send_version(fd);
        http_request_free(&hr);
        goto done;
    }
    if (!strcmp(hr.method, "GET") && !strcmp(hr.path, "/")) {
        s->send_root(fd);
        http_request_free(&hr);
        goto done;
    }
    if (!strcmp(hr.method, "GET") && !strcmp(hr.path, "/v1/models")) {
        s->send_models(fd);
        http_request_free(&hr);
        goto done;
    }
    if (!strcmp(hr.method, "GET") && !strcmp(hr.path, "/metrics")) {
        s->send_metrics(fd);
        http_request_free(&hr);
        goto done;
    }
    const char *model_path_prefix;
    model_path_prefix = "/v1/models/";
    size_t model_path_prefix_len;
    model_path_prefix_len = strlen(model_path_prefix);
    if (!strcmp(hr.method, "GET") &&
        !strncmp(hr.path, model_path_prefix, model_path_prefix_len) &&
        !strcmp(hr.path + model_path_prefix_len,
                server_served_model_id(s)))
    {
        s->send_model(fd, hr.path + model_path_prefix_len);
        http_request_free(&hr);
        goto done;
    }

    request req;
    char err[160];
    bool ok;
    ok = false;
    int ctx_size;
    ctx_size = pulsar_session_ctx(s->sess);
    if (!strcmp(hr.method, "POST") && !strcmp(hr.path, "/v1/messages")) {
        ok = parse_anthropic_request(s->engine, s, hr.body, s->default_tokens,
                                     ctx_size, &req, err, sizeof(err));
    } else if (!strcmp(hr.method, "POST") && !strcmp(hr.path, "/v1/chat/completions")) {
        ok = parse_chat_request(s->engine, s, hr.body, s->default_tokens,
                                ctx_size, &req, err, sizeof(err));
    } else if (!strcmp(hr.method, "POST") && !strcmp(hr.path, "/v1/responses")) {
        ok = parse_responses_request(s->engine, s, hr.body, s->default_tokens,
                                     ctx_size, &req, err, sizeof(err));
    } else if (!strcmp(hr.method, "POST") && !strcmp(hr.path, "/v1/completions")) {
        ok = parse_completion_request(s->engine, hr.body, s->default_tokens,
                                      ctx_size, &req, err, sizeof(err));
    } else {
        http_error(fd, s->enable_cors, 404, "unknown endpoint");
        http_request_free(&hr);
        goto done;
    }
    if (ok) req.raw_body = xstrndup(hr.body, hr.body_len);
    http_request_free(&hr);
    if (!ok) {
        http_error(fd, s->enable_cors, 400, err);
        goto done;
    }
    if (!req.model_from_request) {
        free(req.model);
        req.model = xstrdup(server_served_model_id(s));
    }
    if (request_exceeds_context(&req, ctx_size)) {
        http_error_context_length_exceeded(fd, s->enable_cors, &req, req.prompt.len, ctx_size);
        request_free(&req);
        goto done;
    }

    set_client_socket_nonblocking(fd);
    job j;
    memset(&j, 0, sizeof(j));
    j.fd = fd;
    j.req = req;
    pthread_mutex_init(&j.mu, NULL);
    pthread_cond_init(&j.cv, NULL);

    bool enqueued;
    {
        /* Hold j.mu across enqueue + the wait for the worker's completion
         * signal. The ScopedLock MUST release before the destroy below (you
         * cannot pthread_mutex_destroy a held mutex), so the lock lives in this
         * inner scope and the single shared teardown runs after it unlocks. */
        pulsar::ScopedLock lk(&j.mu);
        enqueued = s->enqueue(&j);
        if (enqueued) {
            while (!j.done) pthread_cond_wait(&j.cv, &j.mu);
        }
    }
    if (!enqueued) http_error(fd, s->enable_cors, 503, "server shutting down");
    pthread_cond_destroy(&j.cv);
    pthread_mutex_destroy(&j.mu);
    request_free(&j.req);
done:
    close(fd);
    s->client_done();
    return NULL;
}



int listen_on(const char *host, int port) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -1;
    int yes = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));

    struct sockaddr_in sa;
    memset(&sa, 0, sizeof(sa));
    sa.sin_family = AF_INET;
    sa.sin_port = htons((uint16_t)port);
    if (!strcmp(host, "localhost")) host = "127.0.0.1";
    if (inet_pton(AF_INET, host, &sa.sin_addr) != 1) {
        close(fd);
        errno = EINVAL;
        return -1;
    }
    if (bind(fd, (struct sockaddr *)&sa, sizeof(sa)) != 0) {
        close(fd);
        return -1;
    }
    if (listen(fd, 128) != 0) {
        close(fd);
        return -1;
    }
    return fd;
}



void configure_client_socket(int fd) {
    struct timeval tv;
    tv.tv_sec = PULSAR_SERVER_IO_TIMEOUT_SEC;
    tv.tv_usec = 0;
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
}



void set_client_socket_nonblocking(int fd) {
    /* The inference worker writes streaming responses itself.  Once a request is
     * queued, a blocked socket would block every other request too, so slow
     * clients are failed instead of back-pressuring the model session. */
    int flags = fcntl(fd, F_GETFL, 0);
    if (flags >= 0) (void)fcntl(fd, F_SETFL, flags | O_NONBLOCK);
}

