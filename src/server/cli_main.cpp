#include "pulsar_server_internal.h"
#include "pulsar_gpu.h"

#include <malloc.h>


/* =========================================================================
 * F1 memory diagnostics (env-gated sampler).
 *
 * PULSAR_SERVER_MEMDIAG=<path>|1|stderr — read ONCE at startup (no-hot-path-
 * flags rule) — starts a low-rate (200 ms) sampler thread that emits one
 * line per sample: process RSS/HWM/swap, system MemAvailable/Free/Cached/
 * Dirty/Writeback, CUDA free/total, the live pulsar_gpu tensor byte counter,
 * the admission ledger, and the glibc heap footprint (mallinfo2).  All
 * values in MiB.  Purely observational: it takes s->mu only to read the
 * ledger, exactly like the scheduler's own logging.
 * ========================================================================= */

typedef struct {
    server *s;
    FILE   *out;
    pthread_t thread;
    bool    running;
} memdiag;

/* Sum of the given keys' kB values from a /proc status-style file; keys[i]
 * match line prefixes ("VmRSS:").  Values land in out_kib[i] (0 if absent). */
static void memdiag_scan_kib(const char *path, const char *const *keys,
                             uint64_t *out_kib, int n) {
    for (int i = 0; i < n; i++) out_kib[i] = 0;
    FILE *fp = fopen(path, "r");
    if (!fp) return;
    char line[256];
    while (fgets(line, sizeof(line), fp)) {
        for (int i = 0; i < n; i++) {
            const size_t klen = strlen(keys[i]);
            if (!strncmp(line, keys[i], klen)) {
                unsigned long long v = 0;
                if (sscanf(line + klen, " %llu kB", &v) == 1) out_kib[i] = v;
            }
        }
    }
    fclose(fp);
}

static void *memdiag_main(void *ud) {
    memdiag *d = (memdiag *)ud;
    server *s = d->s;
    struct timespec t0;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    const double MIB = 1024.0 * 1024.0;
    for (;;) {
        pthread_mutex_lock(&s->mu);
        const bool stop = s->stopping;
        const uint64_t ledger = s->kv_committed_bytes;
        pthread_mutex_unlock(&s->mu);
        if (stop) break;

        static const char *const skeys[] = {"VmRSS:", "VmHWM:", "VmSwap:"};
        static const char *const mkeys[] = {"MemAvailable:", "MemFree:",
                                            "Cached:", "Dirty:", "Writeback:"};
        uint64_t sv[3], mv[5];
        memdiag_scan_kib("/proc/self/status", skeys, sv, 3);
        memdiag_scan_kib("/proc/meminfo", mkeys, mv, 5);
        uint64_t cuda_free = 0, cuda_total = 0;
        pulsar_gpu_mem_info(&cuda_free, &cuda_total);
        struct mallinfo2 mi = mallinfo2();
        struct timespec t1;
        clock_gettime(CLOCK_MONOTONIC, &t1);
        const double ms = (t1.tv_sec - t0.tv_sec) * 1e3 +
                          (t1.tv_nsec - t0.tv_nsec) / 1e6;
        fprintf(d->out,
                "memdiag ms=%.0f rss=%.1f hwm=%.1f swap=%.1f "
                "avail=%.1f memfree=%.1f cached=%.1f dirty=%.1f wb=%.1f "
                "cudafree=%.1f cudatotal=%.1f gputensor=%.1f ledger=%.1f "
                "heap=%.1f heapmmap=%.1f heapused=%.1f\n",
                ms, sv[0] / 1024.0, sv[1] / 1024.0, sv[2] / 1024.0,
                mv[0] / 1024.0, mv[1] / 1024.0, mv[2] / 1024.0,
                mv[3] / 1024.0, mv[4] / 1024.0,
                (double)cuda_free / MIB, (double)cuda_total / MIB,
                (double)pulsar_gpu_tensor_alloc_bytes_current() / MIB,
                (double)ledger / MIB,
                (double)mi.arena / MIB, (double)mi.hblkhd / MIB,
                (double)mi.uordblks / MIB);
        fflush(d->out);
        struct timespec nap = {0, 200 * 1000 * 1000};
        nanosleep(&nap, NULL);
    }
    if (d->out != stderr) fclose(d->out);
    return NULL;
}

/* Env is read here, once, before any request runs; returns false when the
 * diagnostic is off.  Caller joins d->thread at shutdown iff d->running. */
static bool memdiag_start(memdiag *d, server *s) {
    memset(d, 0, sizeof(*d));
    const char *v = getenv("DS4_SERVER_MEMDIAG");
    if (!v || !v[0]) return false;
    d->s = s;
    d->out = stderr;
    if (strcmp(v, "1") != 0 && strcmp(v, "stderr") != 0) {
        d->out = fopen(v, "w");
        if (!d->out) {
            server_log(PULSAR_LOG_WARNING,
                       "pulsar-server: DS4_SERVER_MEMDIAG: cannot open %s: %s "
                       "(sampling to stderr)", v, strerror(errno));
            d->out = stderr;
        }
    }
    if (pthread_create(&d->thread, NULL, memdiag_main, d) != 0) {
        server_log(PULSAR_LOG_WARNING,
                   "pulsar-server: DS4_SERVER_MEMDIAG: sampler thread failed to start");
        if (d->out != stderr) fclose(d->out);
        return false;
    }
    d->running = true;
    server_log(PULSAR_LOG_DEFAULT, "pulsar-server: memory diagnostics sampler on (%s)",
               d->out == stderr ? "stderr" : v);
    return true;
}



/* =========================================================================
 * Startup warmup generation (F1 unledgered-spike fix, task #32).
 *
 * The CUDA stack allocates a large first-generation working set LAZILY —
 * cuBLASLt workspaces, FP8/MXFP4 staging, GEMV activation buffers — on the
 * first prefill/decode that exercises each kernel family (measured on the
 * GB10, 2026-07-17: ~9.2 GiB, materializing in ~1.5 s on the first client
 * burst; see pulsar_server_internal.h, PULSAR_SERVER_PROCESS_OVERHEAD_BYTES).
 * Every MemAvailable-based admission check made before that burst is
 * therefore stale-high by that amount: the floor check legally admits
 * sessions whose room the first generation then consumes, and the box
 * craters straight through watchdog floors mid-request (F1 symptom B:
 * 6.3 -> 2.9 GiB in ~3 s with the session ledger clean).
 *
 * Running one throwaway generation here — a full prefill chunk plus a tail,
 * then a few tokens through the production decode path (speculative when
 * the drafter is loaded, so its buffers materialize too) — moves that cliff
 * to startup, BEFORE the listener opens and before the admission budget is
 * derived.  After this returns, MemAvailable is the truth every later
 * check needs it to be.
 *
 * Best-effort: a warmup failure is logged and serving continues (the cost
 * is only that the first real generation pays the materialization instead).
 * The session state is shrunk to a single token afterwards so the warmup
 * transcript can never win slot routing, hit the disk KV min-tokens floor,
 * or collide with a real conversation. */
static void server_warmup_generation(pulsar_engine *engine, pulsar_session *session,
                                     int ctx_size) {
    const double t0 = server_now_sec();
    const uint64_t avail_before = server_mem_available_bytes();
    /* One full prefill-cap chunk plus a tail: exercises the widened chunk
     * prefill shape AND the sub-chunk tail shape.  The session's resolved
     * cap is used (the config value can be 0 = engine default). */
    const int prefill_cap = pulsar_session_prefill_cap(session);
    int want = prefill_cap + 64;
    if (want < 64) want = 64;
    if (want > ctx_size - 256) want = ctx_size - 256;
    if (want < prefill_cap) {
        /* A small --ctx-size truncates the warmup below one full prefill
         * chunk: the chunked-prefill working set is NOT materialized here
         * (single-shot path only) and the measured budget will read
         * stale-high — still bounded by the static formula's min(). */
        server_log(PULSAR_LOG_WARNING,
                   "pulsar-server: warmup prompt truncated to %d tokens "
                   "(< prefill cap %d): chunked-prefill workspaces are not "
                   "materialized at startup; first long prompt pays them",
                   want, prefill_cap);
    }
    pulsar_tokens body = {0}, prompt = {0};
    pulsar_tokenize_text(engine, "The quick brown fox jumps over the lazy dog. ",
                      &body);
    if (body.len > 0 && want > 0) {
        while (prompt.len < want)
            for (int i = 0; i < body.len && prompt.len < want; i++)
                pulsar_tokens_push(&prompt, body.v[i]);
    }
    char err[256];
    err[0] = '\0';
    bool ok = prompt.len > 0 &&
              pulsar_session_sync(session, &prompt, err, sizeof(err)) == 0;
    int emitted = 0;
    if (ok) {
        uint64_t rng = 0x5eed5eed5eed5eedULL;
        int toks[17];
        while (emitted < 12) {
            if (pulsar_engine_has_dspark(engine)) {
                const int n = pulsar_session_generate_speculative(
                        session, 0.0f, 0, 1.0f, 0.0f, &rng, 12 - emitted,
                        pulsar_token_eos(engine), toks,
                        (int)(sizeof(toks) / sizeof(toks[0])),
                        err, sizeof(err));
                /* Contract (session.c): eos arrives as a returned token; 0 means
                 * a rejected step (bad args / dirty multiseq state), not eos. */
                if (n <= 0) { ok = false; break; }
                emitted += n;
                if (toks[n - 1] == pulsar_token_eos(engine)) break;
            } else {
                const int tok = pulsar_session_argmax(session);
                if (tok == pulsar_token_eos(engine)) break;
                if (pulsar_session_eval(session, tok, err, sizeof(err)) != 0) {
                    ok = false;
                    break;
                }
                emitted++;
            }
        }
    }
    /* Shrink the live state to one token: below every routing/storage
     * threshold, and the next real prompt rebuilds from zero anyway. */
    if (prompt.len > 0) {
        prompt.len = 1;
        char shrink_err[256];
        (void)pulsar_session_sync(session, &prompt, shrink_err,
                               sizeof(shrink_err));
    }
    pulsar_tokens_free(&prompt);
    pulsar_tokens_free(&body);
    const uint64_t avail_after = server_mem_available_bytes();
    if (ok) {
        server_log(PULSAR_LOG_DEFAULT,
                   "pulsar-server: warmup generation: %d prompt + %d decode tokens "
                   "in %.1f s; MemAvailable %.2f -> %.2f GiB "
                   "(first-generation working set %.2f GiB materialized)",
                   want, emitted, server_now_sec() - t0,
                   (double)avail_before / (1024.0 * 1024.0 * 1024.0),
                   (double)avail_after / (1024.0 * 1024.0 * 1024.0),
                   (double)(avail_before > avail_after
                                ? avail_before - avail_after : 0) /
                       (1024.0 * 1024.0 * 1024.0));
    } else {
        server_log(PULSAR_LOG_WARNING,
                   "pulsar-server: warmup generation failed (%s); the first real "
                   "generation will pay the deferred CUDA allocations instead",
                   err[0] ? err : "empty warmup prompt");
    }
}



static int parse_int_arg(const char *s, const char *opt) {
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (!s[0] || *end || v <= 0 || v > INT_MAX) {
        server_log(PULSAR_LOG_DEFAULT, "pulsar-server: invalid value for %s: %s", opt, s);
        exit(2);
    }
    return (int)v;
}



static int parse_nonneg_int_arg(const char *s, const char *opt) {
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (!s[0] || *end || v < 0 || v > INT_MAX) {
        server_log(PULSAR_LOG_DEFAULT, "pulsar-server: invalid value for %s: %s", opt, s);
        exit(2);
    }
    return (int)v;
}



static float parse_float_arg(const char *s, const char *opt, float minv, float maxv) {
    char *end = NULL;
    float v = strtof(s, &end);
    if (!s[0] || *end || v < minv || v > maxv) {
        server_log(PULSAR_LOG_DEFAULT, "pulsar-server: invalid value for %s: %s", opt, s);
        exit(2);
    }
    return v;
}



static const char *need_arg(int *i, int argc, char **argv, const char *opt) {
    if (*i + 1 >= argc) {
        server_log(PULSAR_LOG_DEFAULT, "pulsar-server: missing value for %s", opt);
        exit(2);
    }
    return argv[++(*i)];
}



/* Tier-2 overcommit (task #55). PULSAR_OVERCOMMIT switches the pool auto-size from
 * "N-bank pool must fit the budget in FULL" to "only the EAGER floor (raw ring +
 * state lanes + shared) for N banks must fit; the ctx-scaled comp/index are
 * VA-only, demand-paged, and NOT charged at admission". This lets a large --ctx
 * (e.g. the 1M default) come up with N>1 banks all 1M-CAPABLE, each paying only
 * for the KV it actually touches. DEFAULT ON as of v0.3.0: the increment-2
 * proactive eviction guard has landed (generate.c worker_batched_decode_quantum),
 * so banks that grow toward 1M are bounded by LRU-idle eviction before the
 * physical budget is breached. PULSAR_OVERCOMMIT=0/off/false reverts to classic
 * full-charge admission; PULSAR_MSEQ_BANKS still pins and bypasses auto-sizing.
 * Read once at startup, never on a hot path. */
static bool server_overcommit_enabled(void) {
    const char *v = getenv("DS4_OVERCOMMIT");
    if (!v || !v[0]) return true; /* default ON (v0.3.0) */
    return !(v[0] == '0' || !strcasecmp(v, "off") || !strcasecmp(v, "false"));
}

/* Optional per-bank touched-KV reservation for the overcommit fit: the ctx whose
 * demand-paged comp/index cost is charged per bank at admission (headroom so a
 * bank that grows to this depth was pre-admitted). Default 0 => pure overcommit
 * (charge only the eager floor). Clamped to non-negative; garbage => 0. */
static int server_overcommit_reserve_ctx(void) {
    const char *v = getenv("DS4_OVERCOMMIT_RESERVE_CTX");
    if (!v || !v[0]) return 0;
    char *end = NULL;
    long n = strtol(v, &end, 10);
    if (end == v || (end && *end != '\0') || n < 0 || n > INT_MAX) return 0;
    return (int)n;
}



static void log_context_memory(pulsar_backend backend,
                               int         ctx_size,
                               uint32_t    prefill_chunk) {
    pulsar_context_memory m =
        pulsar_context_memory_estimate_with_prefill(backend,
                                                 ctx_size,
                                                 prefill_chunk);
    server_log(PULSAR_LOG_DEFAULT,
               "pulsar-server: context buffers %.2f MiB (ctx=%d, backend=%s, prefill_chunk=%u, raw_kv_rows=%u, compressed_kv_rows=%u)",
               (double)m.total_bytes / (1024.0 * 1024.0),
               ctx_size,
               pulsar_backend_name(backend),
               m.prefill_cap,
               m.raw_cap,
               m.comp_cap);
}



/* Admission-control budget (Tier 1 §1.4). The session budget is the
 * unified-memory headroom left for per-session GPU state after the shared
 * resident weights, a fixed process-overhead reserve, and the free-memory
 * floor (PULSAR_SERVER_MEM_FLOOR_BYTES) — committing the full budget must still
 * leave the floor free. Returns 0 if the reserves already exceed usable. */
static uint64_t server_kv_budget_bytes(uint64_t weights_resident_bytes) {
    uint64_t reserved = weights_resident_bytes +
                        PULSAR_SERVER_PROCESS_OVERHEAD_BYTES +
                        PULSAR_SERVER_MEM_FLOOR_BYTES;
    if (reserved >= PULSAR_SERVER_USABLE_BYTES) return 0;
    return PULSAR_SERVER_USABLE_BYTES - reserved;
}

/* Admission rule: the already-committed live-session cost plus this request's
 * estimated cost must fit under the budget. Gates the startup session here and
 * every lazy slot provisioning in the scheduler (generate.c); an over-budget
 * provisioning attempt leaves the job queued until a slot frees (plan Tier 1
 * §1.4). Non-static: the scheduler calls it; unit tests live in this TU. */
bool server_kv_admits(uint64_t kv_budget_bytes,
                             uint64_t committed_bytes,
                             uint64_t incoming_bytes) {
    if (incoming_bytes > kv_budget_bytes) return false;      /* guards overflow + lone fit */
    return committed_bytes <= kv_budget_bytes - incoming_bytes;
}

/* Live MemAvailable floor: refuse a session create unless the kernel still
 * reports room for the estimated cost plus the free floor. The floor is
 * kernel/OS breathing room only — process-fixed costs live in the overhead
 * reserve the ledger already subtracted (see PULSAR_SERVER_MEM_FLOOR_BYTES in
 * pulsar_server_internal.h for the 2026-07-15 sizing). avail == 0 means
 * /proc/meminfo was unreadable: fail closed — this guard exists precisely
 * for when other accounting is wrong, so it must not silently disarm
 * itself. Non-static: provision_slot and the eviction precheck (generate.c)
 * call it; unit tests live in this TU. */
bool server_mem_floor_admits(uint64_t avail_bytes, uint64_t est_bytes) {
    if (avail_bytes == 0) return false;
    if (est_bytes > UINT64_MAX - PULSAR_SERVER_MEM_FLOOR_BYTES) return false;
    return avail_bytes >= est_bytes + PULSAR_SERVER_MEM_FLOOR_BYTES;
}



static void server_close_resources(server *s) {
    if (s->trace) {
        fclose(s->trace);
        s->trace = NULL;
    }
    kv_cache_close(&s->kv);
    tool_memory_free(&s->tool_mem);
    pthread_mutex_destroy(&s->tool_mu);
    pthread_mutex_destroy(&s->trace_mu);
    pthread_cond_destroy(&s->clients_cv);
    pthread_cond_destroy(&s->cv);
    pthread_mutex_destroy(&s->mu);
    /* The ONE session (server.sess — classic mode == slot 0 alone, pool mode
     * == N banks over it) is freed exactly once here; slots are pure bank
     * descriptors and own nothing to free. */
    for (int i = 0; i < PULSAR_SESSION_POOL_CAP; i++) {
        live_tool_state_free(&s->slots[i].responses_live);
        live_tool_state_free(&s->slots[i].anthropic_live);
        visible_live_free(&s->slots[i].thinking_live);
        s->slots[i].provisioned = false;
    }
    if (s->sess) pulsar_session_free(s->sess);
    s->sess = NULL;
    /* Tier-2 guard spill files are per-bank snapshots (server_spill_bank in
     * generate.c writes <spill_dir>/spill-bank-<bank>.kv).  A bank that is
     * still spilled when the server exits would otherwise leave a stale
     * multi-GiB file behind forever, so sweep every provisioned slot's bank
     * on the way out.  Best-effort: a missing file is the normal case. */
    if (s->spill_dir[0]) {
        for (int i = 0; i < s->n_slots && i < PULSAR_SESSION_POOL_CAP; i++) {
            char spath[600];
            snprintf(spath, sizeof spath, "%s/spill-bank-%u.kv",
                     s->spill_dir, (unsigned)s->slots[i].bank);
            (void)remove(spath);
        }
    }
    pulsar_engine_close(s->engine);
    memset(s, 0, sizeof(*s));
}



void usage(FILE *fp, const char *topic) {
    pulsar_help_print(fp, PULSAR_HELP_SERVER, topic);
}



static pulsar_backend parse_backend_arg(const char *s, const char *arg) {
    if (!strcmp(s, "cuda")) return PULSAR_BACKEND_CUDA;
    server_log(PULSAR_LOG_DEFAULT, "pulsar-server: invalid %s value: %s", arg, s);
    server_log(PULSAR_LOG_DEFAULT, "pulsar-server: valid server backends are: cuda");
    exit(2);
}



static pulsar_backend default_server_backend(void) {
    return PULSAR_BACKEND_CUDA;
}



/* Default gguf resolution, in order: the project gguf/ directory (canonical
 * production filename, then the generic name), the current directory, then
 * $PULSAR_MODEL_DIR/<canonical> when that env var is set.  No store path is
 * baked into the binary.  Returns NULL if nothing readable — callers decide
 * whether that is fatal (main model) or a warning (drafter).  Heap results
 * intentionally live for the process lifetime (they become engine paths). */
static const char *resolve_gguf_at(const char *dir, const char *name) {
    size_t n = strlen(dir) + 1 + strlen(name) + 1;
    char *p = (char *)malloc(n);
    if (!p) return NULL;
    snprintf(p, n, "%s/%s", dir, name);
    if (access(p, R_OK) == 0) return p;
    free(p);
    return NULL;
}

/* Naming convention: gguf/ holds immutable versioned artifacts
 * (ds4flash-<variant>-<mods>-vN.gguf, dspark-<variant>-vN.gguf) plus two
 * ACTIVE-POINTER symlinks — model.gguf and dspark.gguf — that select what a
 * bare `pulsar-server` runs.  Deploy = repoint the symlink. */
static const char *resolve_default_gguf(const char *pointer) {
    const char *p;
    if ((p = resolve_gguf_at("gguf", pointer)) != NULL) return p;
    if (access(pointer, R_OK) == 0) return pointer;
    const char *dir = getenv("DS4_MODEL_DIR");
    if (dir && dir[0] && (p = resolve_gguf_at(dir, pointer)) != NULL) return p;
    return NULL;
}

/* Default directory for the disk KV cache, which is ON by default: the
 * conversation-bounce restore (sub-second vs full re-prefill TTFT) must not
 * depend on remembering --kv-disk-dir.  Placement follows XDG:
 * $XDG_CACHE_HOME/ds4/kv-<model>, else ~/.cache/ds4/kv-<model>.  The user
 * cache dir is chosen over <dirname(model)>/ because the model store may be
 * read-only or a network mount, while checkpoints are latency-sensitive,
 * regenerable, per-user state.
 *
 * <model> keys the directory by model identity.  The store header only
 * self-validates the model VARIANT (Flash/Pro, pulsar_engine_model_id) and the
 * routed-expert quant bits — NOT the exact weight artifact — so two different
 * ggufs of the same shape sharing a directory could cross-restore each
 * other's checkpoints.  gguf/model.gguf is an ACTIVE-POINTER symlink by
 * convention; its realpath basename is the versioned artifact name and
 * therefore the right identity key.  Returns a heap path (process lifetime),
 * or NULL when no cache home can be resolved. */
static char *server_default_kv_disk_dir(const char *model_path) {
    char resolved[PATH_MAX];
    const char *src = model_path ? model_path : "";
    if (model_path && realpath(model_path, resolved)) src = resolved;
    const char *base = strrchr(src, '/');
    base = base ? base + 1 : src;
    size_t len = strlen(base);
    if (len > 5 && !strcasecmp(base + len - 5, ".gguf")) len -= 5;
    char key[128];
    if (len == 0) {
        strcpy(key, "model");
    } else {
        if (len > sizeof(key) - 1) len = sizeof(key) - 1;
        for (size_t j = 0; j < len; j++) {
            unsigned char ch = (unsigned char)base[j];
            key[j] = (isalnum(ch) || ch == '.' || ch == '_' || ch == '-') ?
                     (char)ch : '_';
        }
        key[len] = '\0';
    }
    buf b = {0};
    /* Per the XDG spec a non-absolute XDG_CACHE_HOME is ignored (a relative
     * value would key the cache off whatever the post---chdir cwd happens to
     * be). */
    const char *xdg = getenv("XDG_CACHE_HOME");
    if (xdg && xdg[0] == '/') {
        buf_printf(&b, "%s/ds4/kv-%s", xdg, key);
    } else {
        const char *home = getenv("HOME");
        if (!home || !home[0]) return NULL;
        buf_printf(&b, "%s/.cache/ds4/kv-%s", home, key);
    }
    return buf_take(&b);
}



/* Resolve the effective disk KV cache directory after option parsing (and
 * after --chdir, so a relative model path realpaths from the project tree).
 * Explicit --kv-disk-dir PATH behaves exactly as before; --kv-disk-dir "" or
 * --no-kv-disk opts out; otherwise the default above is filled in.  An
 * unresolvable or unusable directory must never be fatal — the server runs
 * with the disk cache disabled (kv_cache_open handles the unusable case). */
static void server_resolve_kv_disk_dir(server_config *c) {
    if (c->kv_disk_disable) {
        c->kv_disk_dir = NULL;
        server_log(PULSAR_LOG_DEFAULT, "pulsar-server: disk KV cache disabled by flag");
        return;
    }
    if (c->kv_disk_dir) return; /* explicit path: unchanged behavior */
    c->kv_disk_dir = server_default_kv_disk_dir(c->engine.model_path);
    if (c->kv_disk_dir) {
        server_log(PULSAR_LOG_DEFAULT,
                   "pulsar-server: disk KV cache default dir %s "
                   "(--kv-disk-dir PATH overrides; --no-kv-disk disables)",
                   c->kv_disk_dir);
    } else {
        server_log(PULSAR_LOG_DEFAULT,
                   "pulsar-server: disk KV cache disabled (neither XDG_CACHE_HOME "
                   "nor HOME is set, no default directory)");
    }
}



static server_config parse_options(int argc, char **argv) {
    server_config c = {
        .engine = {
            .model_path = "ds4flash.gguf",
            .backend = default_server_backend(),
        },
        .host = "0.0.0.0",
        .port = 8000,
        /* v0.3.0 default: 1M context with overcommit ON (see
         * server_overcommit_enabled). Every session is grow-to-1M-capable, but
         * the ctx-scaled KV is demand-paged, so a short session pays only for
         * what it touches — admission charges just the eager floor (~4 banks come
         * up at ~6.4 GiB, NOT 4x the 1M worst case), and the increment-2 eviction
         * guard bounds banks that actually grow toward 1M. An operator who wants
         * a hard per-session cap passes a smaller --ctx; PULSAR_OVERCOMMIT=0 reverts
         * to classic full-charge admission (1M => N=1 single session). */
        .ctx_size = 1048576,
        .default_tokens = 393216,
        .tool_memory_max_ids = PULSAR_TOOL_MEMORY_DEFAULT_MAX_IDS,
    };
    c.kv_cache = kv_cache_default_options();

    bool directional_steering_scale_set = false;
    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        if (!strcmp(arg, "-h") || !strcmp(arg, "--help")) {
            const char *topic = (i + 1 < argc && argv[i + 1][0] != '-') ?
                argv[i + 1] : NULL;
            usage(stdout, topic);
            exit(0);
        }
        if (!strcmp(arg, "-m") || !strcmp(arg, "--model")) {
            c.engine.model_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--served-model-id")) {
            c.served_model_id = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--served-model-name")) {
            c.served_model_name = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--no-dspark")) {
            c.engine.dspark_disable = true;
        } else if (!strcmp(arg, "--dspark")) {
            c.engine.dspark_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--dspark-draft")) {
            c.engine.dspark_draft_tokens = parse_int_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--dspark-confidence")) {
            c.engine.dspark_confidence = parse_float_arg(need_arg(&i, argc, argv, arg), arg, 0.0f, 1.0f);
        } else if (!strcmp(arg, "-c") || !strcmp(arg, "--ctx")) {
            c.ctx_size = parse_int_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "-n") || !strcmp(arg, "--tokens")) {
            c.default_tokens = parse_int_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "-t") || !strcmp(arg, "--threads")) {
            c.engine.n_threads = parse_int_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--chdir")) {
            c.chdir_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--host")) {
            c.host = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--port")) {
            c.port = parse_int_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--cors")) {
            c.enable_cors = true;
        } else if (!strcmp(arg, "--trace")) {
            c.trace_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--kv-disk-dir")) {
            /* An empty value opts out of the default-on disk cache; the last
             * kv-disk flag on the command line wins. */
            c.kv_disk_dir = need_arg(&i, argc, argv, arg);
            c.kv_disk_disable = !c.kv_disk_dir[0];
        } else if (!strcmp(arg, "--no-kv-disk")) {
            c.kv_disk_dir = NULL;
            c.kv_disk_disable = true;
        } else if (!strcmp(arg, "--kv-disk-space-mb")) {
            c.kv_disk_space_mb = (uint64_t)parse_int_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--kv-cache-min-tokens")) {
            c.kv_cache.min_tokens = parse_int_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--kv-cache-cold-max-tokens")) {
            c.kv_cache.cold_max_tokens = parse_nonneg_int_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--kv-cache-continued-interval-tokens")) {
            c.kv_cache.continued_interval_tokens = parse_nonneg_int_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--kv-cache-boundary-trim-tokens")) {
            c.kv_cache.boundary_trim_tokens = parse_nonneg_int_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--kv-cache-boundary-align-tokens")) {
            c.kv_cache.boundary_align_tokens = parse_nonneg_int_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--kv-cache-reject-different-quant")) {
            c.kv_cache_reject_different_quant = true;
        } else if (!strcmp(arg, "--disable-exact-dsml-tool-replay")) {
            c.disable_exact_dsml_tool_replay = true;
        } else if (!strcmp(arg, "--tool-memory-max-ids")) {
            c.tool_memory_max_ids = parse_int_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--quality")) {
            c.engine.quality = true;
        } else if (!strcmp(arg, "--prefill-chunk")) {
            int v = parse_int_arg(need_arg(&i, argc, argv, arg), arg);
            if (v <= 0) {
                server_log(PULSAR_LOG_DEFAULT,
                           "pulsar-server: --prefill-chunk must be positive");
                exit(2);
            }
            c.engine.prefill_chunk = (uint32_t)v;
        } else if (!strcmp(arg, "--dir-steering-file")) {
            c.engine.directional_steering_file = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--dir-steering-ffn")) {
            c.engine.directional_steering_ffn = parse_float_arg(need_arg(&i, argc, argv, arg), arg, -100.0f, 100.0f);
            directional_steering_scale_set = true;
        } else if (!strcmp(arg, "--dir-steering-attn")) {
            c.engine.directional_steering_attn = parse_float_arg(need_arg(&i, argc, argv, arg), arg, -100.0f, 100.0f);
            directional_steering_scale_set = true;
        } else if (!strcmp(arg, "--warm-weights")) {
            c.engine.warm_weights = true;
        } else if (!strcmp(arg, "--cuda")) {
            c.engine.backend = PULSAR_BACKEND_CUDA;
        } else if (!strcmp(arg, "--backend")) {
            c.engine.backend = parse_backend_arg(need_arg(&i, argc, argv, arg), arg);
        } else {
            server_log(PULSAR_LOG_DEFAULT, "pulsar-server: unknown option: %s", arg);
            usage(stderr, NULL);
            exit(2);
        }
    }
    if (c.kv_cache.cold_max_tokens > 0 &&
        c.kv_cache.cold_max_tokens < c.kv_cache.min_tokens)
    {
        server_log(PULSAR_LOG_DEFAULT,
                   "pulsar-server: --kv-cache-cold-max-tokens must be 0 or >= --kv-cache-min-tokens");
        exit(2);
    }
    if (c.engine.directional_steering_file && !directional_steering_scale_set) {
        c.engine.directional_steering_ffn = 1.0f;
    }
    /* Production defaults: when -m/--dspark are not given, resolve the
     * canonical ggufs (cwd first, then the model store) so a bare
     * `pulsar-server` is the full validated launch. */
    if (!strcmp(c.engine.model_path, "ds4flash.gguf") &&
        access(c.engine.model_path, R_OK) != 0) {
        const char *m = resolve_default_gguf("model.gguf");
        if (m) {
            c.engine.model_path = m;
            server_log(PULSAR_LOG_DEFAULT, "pulsar-server: default model %s", m);
        }
    }
    if (!c.engine.dspark_path) {
        const char *d = resolve_default_gguf("dspark.gguf");
        if (d) {
            c.engine.dspark_path = d;
            server_log(PULSAR_LOG_DEFAULT, "pulsar-server: default drafter %s", d);
        } else {
            server_log(PULSAR_LOG_DEFAULT,
                       "pulsar-server: no drafter found (gguf/dspark.gguf); "
                       "running without speculative decoding");
        }
    }
    return c;
}



#ifndef PULSAR_SERVER_TEST

int main(int argc, char **argv) {
    signal(SIGPIPE, SIG_IGN);
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = stop_signal_handler;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGINT, &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);

    server_config cfg = parse_options(argc, argv);
    if (cfg.chdir_path && chdir(cfg.chdir_path) != 0) {
        server_log(PULSAR_LOG_DEFAULT, "pulsar-server: failed to chdir to %s: %s",
                   cfg.chdir_path, strerror(errno));
        return 1;
    }
    server_resolve_kv_disk_dir(&cfg);

    pulsar_engine *engine = NULL;
    if (pulsar_engine_open(&engine, &cfg.engine) != 0) return 1;

    log_context_memory(cfg.engine.backend,
                       cfg.ctx_size,
                       cfg.engine.prefill_chunk);

    /* Admission control (Tier 1 §1.4): compute the session budget from the
     * real resident weight footprint and the TRUE per-session cost (full graph
     * + prefill working set + drafter state, pulsar_engine_session_cost_bytes),
     * then gate the slot's graph allocation on it. The scheduler runs the same
     * predicate for every lazily provisioned slot. */
    const uint64_t weights_resident = pulsar_engine_weights_resident_bytes(engine);
    const uint64_t kv_budget = server_kv_budget_bytes(weights_resident);

    /* Tier-2 pool auto-sizing (batching ON by default, auto-fit). Batching is
     * the default: the bank count is DERIVED from --ctx so a moderate ctx yields
     * a multi-bank pool while a huge ctx (e.g. 1M) correctly yields N=1 (exact
     * classic behavior, no startup failure). PULSAR_MSEQ_BANKS, when set, PINS the
     * count and skips auto-sizing. This runs BEFORE the first
     * pulsar_engine_session_cost_bytes call (gpu_graph_bank_pool_n caches
     * PULSAR_MSEQ_BANKS on first read); the fit probe uses the _banked cost variant,
     * which prices an explicit N without touching that cache. */
    const int pool_auto_max = PULSAR_SESSION_POOL_CAP; /* throughput saturates ~4 */

    /* Tier-2 overcommit (task #55, increment 1). Precompute the per-bank split of
     * the banked session cost into an EAGER floor (raw ring + state lanes +
     * dspark-banked) and a DEMAND-PAGED comp/index term (VA-only, physical on
     * touch). The cost is exactly affine in N: banked(N) = shared + N*(eager +
     * demand), so eager = (banked(2)-banked(1)) - demand and shared = banked(1) -
     * (banked(2)-banked(1)). Overcommit charges only shared + N*eager (+ optional
     * touched reserve) at admission, so a huge --ctx yields N>1 banks instead of
     * N=1. DEFAULT ON as of v0.3.0 — the increment-2 proactive-eviction guard has
     * landed and was stress-validated (banks growing toward 1M are bounded by
     * LRU-idle eviction before the physical budget is breached; see
     * server_overcommit_enabled). PULSAR_OVERCOMMIT=0 reverts to full-charge N=1. */
    const bool overcommit = server_overcommit_enabled();
    uint64_t oc_shared = 0, oc_eager_pb = 0, oc_expect_pb = 0, oc_demand_pb = 0;
    if (overcommit) {
        const uint64_t banked1 = pulsar_engine_session_cost_bytes_banked(engine, cfg.ctx_size, 1);
        const uint64_t banked2 = pulsar_engine_session_cost_bytes_banked(engine, cfg.ctx_size, 2);
        const uint64_t marginal = banked2 > banked1 ? banked2 - banked1 : 0;
        oc_demand_pb = pulsar_engine_demand_paged_bytes_per_bank(engine, cfg.ctx_size);
        oc_eager_pb = marginal > oc_demand_pb ? marginal - oc_demand_pb : 0;
        oc_shared = banked1 > marginal ? banked1 - marginal : 0;
        const int reserve_ctx = server_overcommit_reserve_ctx();
        oc_expect_pb = reserve_ctx > 0
            ? pulsar_engine_demand_paged_bytes_per_bank(engine, reserve_ctx) : 0;
    }

    if (getenv("DS4_MSEQ_BANKS")) {
        server_log(PULSAR_LOG_DEFAULT,
                   "pulsar-server: Tier-2 pool: DS4_MSEQ_BANKS pinned by operator "
                   "(auto-sizing skipped)");
    } else {
        /* Fit table across reference contexts — how many banks (1..cap) fit
         * within the admission budget at each ctx. Operator-facing sizing aid. */
        static const int ref_ctx[] = {32768, 65536, 131072, 262144, 524288, 1048576};
        server_log(PULSAR_LOG_DEFAULT,
                   "pulsar-server: Tier-2 pool fit table (budget %.1f GiB, cap %d banks):",
                   (double)kv_budget / (1024.0 * 1024.0 * 1024.0), pool_auto_max);
        for (size_t ci = 0; ci < sizeof(ref_ctx) / sizeof(ref_ctx[0]); ci++) {
            const double cost1 =
                (double)pulsar_engine_session_cost_bytes_banked(engine, ref_ctx[ci], 1)
                / (1024.0 * 1024.0 * 1024.0);
            int fitN = 1;
            double costN = cost1;
            for (int N = pool_auto_max; N >= 1; N--) {
                const uint64_t c =
                    pulsar_engine_session_cost_bytes_banked(engine, ref_ctx[ci], N);
                if (c > 0 && server_kv_admits(kv_budget, 0, c)) {
                    fitN = N;
                    costN = (double)c / (1024.0 * 1024.0 * 1024.0);
                    break;
                }
            }
            server_log(PULSAR_LOG_DEFAULT,
                       "pulsar-server:   ctx %7d: fits %d bank(s) (1-bank %.2f GiB, "
                       "%d-bank %.2f GiB)",
                       ref_ctx[ci], fitN, cost1, fitN, costN);
        }
        /* Auto-derive N for the requested ctx: largest N (1..cap) whose cost fits
         * the budget (which already reserves overhead + the 4 GiB floor). Always
         * at least 1, so no ctx ever fails to start here. Overcommit charges only
         * the eager floor + reserve; the classic path charges the full N-bank
         * cost (so a huge ctx yields N=1). */
        const double gib = 1024.0 * 1024.0 * 1024.0;
        int chosen = 1;
        if (overcommit) {
            for (int N = pool_auto_max; N >= 1; N--) {
                const uint64_t cost = oc_shared + (uint64_t)N * (oc_eager_pb + oc_expect_pb);
                if (server_kv_admits(kv_budget, 0, cost)) { chosen = N; break; }
            }
        } else {
            for (int N = pool_auto_max; N >= 1; N--) {
                const uint64_t c =
                    pulsar_engine_session_cost_bytes_banked(engine, cfg.ctx_size, N);
                if (c > 0 && server_kv_admits(kv_budget, 0, c)) { chosen = N; break; }
            }
        }
        char banks[8];
        snprintf(banks, sizeof(banks), "%d", chosen);
        setenv("DS4_MSEQ_BANKS", banks, 1);
        if (overcommit) {
            server_log(PULSAR_LOG_DEFAULT,
                       "pulsar-server: Tier-2 OVERCOMMIT auto-sized to %d bank(s) for --ctx %d: "
                       "eager floor %.2f GiB/bank + reserve %.2f GiB/bank charged; "
                       "demand-paged VA %.2f GiB/bank reserved (physical on touch, NOT "
                       "charged); shared %.2f GiB; admission est %.2f GiB (batching %s)",
                       chosen, cfg.ctx_size,
                       (double)oc_eager_pb / gib, (double)oc_expect_pb / gib,
                       (double)oc_demand_pb / gib, (double)oc_shared / gib,
                       (double)(oc_shared + (uint64_t)chosen * (oc_eager_pb + oc_expect_pb)) / gib,
                       chosen > 1 ? "ON" : "OFF");
        } else {
            server_log(PULSAR_LOG_DEFAULT,
                       "pulsar-server: Tier-2 pool auto-sized to %d bank(s) for --ctx %d "
                       "(batching %s)", chosen, cfg.ctx_size,
                       chosen > 1 ? "ON" : "OFF (single-session, ctx too large for a pool)");
        }
    }

    /* Full N-bank cost prices what the allocator will ACTUALLY reserve (incl. the
     * demand-paged VA, which cudaMallocManaged counts at full size). Under
     * overcommit, admission is gated on the eager-floor est instead so the pool
     * comes up at N>1; the full est is kept for the est-vs-actual reconcile below
     * (it matches the measured resident, so no false drift warning). */
    const uint64_t full_banked_est = pulsar_engine_session_cost_bytes(engine, cfg.ctx_size);
    uint64_t overcommit_admission_est = 0;
    if (overcommit) {
        const char *nb = getenv("DS4_MSEQ_BANKS");
        long finalN = nb && nb[0] ? strtol(nb, NULL, 10) : 1;
        if (finalN < 1) finalN = 1;
        overcommit_admission_est = oc_shared + (uint64_t)finalN * (oc_eager_pb + oc_expect_pb);
    }
    const uint64_t session_est = overcommit ? overcommit_admission_est : full_banked_est;
    server_log(PULSAR_LOG_DEFAULT,
               "pulsar-server: session admission: usable=%.1f GiB, weights_resident=%.1f GiB, "
               "overhead=%.1f GiB, floor=%.1f GiB, budget=%.1f GiB",
               (double)PULSAR_SERVER_USABLE_BYTES / (1024.0 * 1024.0 * 1024.0),
               (double)weights_resident / (1024.0 * 1024.0 * 1024.0),
               (double)PULSAR_SERVER_PROCESS_OVERHEAD_BYTES / (1024.0 * 1024.0 * 1024.0),
               (double)PULSAR_SERVER_MEM_FLOOR_BYTES / (1024.0 * 1024.0 * 1024.0),
               (double)kv_budget / (1024.0 * 1024.0 * 1024.0));
    server_log(PULSAR_LOG_DEFAULT,
               "pulsar-server: session admission: per-session true cost est=%.2f GiB (ctx=%d)",
               (double)session_est / (1024.0 * 1024.0 * 1024.0),
               cfg.ctx_size);
    if (session_est == 0 || !server_kv_admits(kv_budget, 0, session_est)) {
        server_log(PULSAR_LOG_DEFAULT,
                   "pulsar-server: session admission REJECTED: est %.2f GiB exceeds budget %.2f GiB "
                   "(reduce --ctx-size)",
                   (double)session_est / (1024.0 * 1024.0 * 1024.0),
                   (double)kv_budget / (1024.0 * 1024.0 * 1024.0));
        pulsar_engine_close(engine);
        return 1;
    }

    pulsar_session *session = NULL;
    if (pulsar_session_create(&session, engine, cfg.ctx_size) != 0) {
        server_log(PULSAR_LOG_DEFAULT, "pulsar-server: failed to create %s session",
                   pulsar_backend_name(cfg.engine.backend));
        pulsar_engine_close(engine);
        return 1;
    }
    /* Reconcile the estimate with what the allocator really did and commit the
     * ACTUAL to the ledger (>10% drift means the sizing code in gpu_diag.c has
     * fallen out of sync with the allocator — fix that, not the ledger). Under
     * overcommit the admission est is the eager floor only; reconcile against the
     * FULL banked est so it still matches the measured resident (which counts the
     * demand-paged managed VA at full size) — no false drift warning. */
    const uint64_t session_actual =
        server_reconciled_session_cost(0, cfg.ctx_size,
                                       overcommit ? full_banked_est : session_est,
                                       pulsar_session_resident_bytes(session));

    /* Materialize the lazy first-generation CUDA working set BEFORE deriving
     * the admission budget or opening the listener (F1, task #32). */
    server_warmup_generation(engine, session, cfg.ctx_size);

    /* Re-derive the admission budget from MEASURED post-warmup MemAvailable
     * (the est==actual discipline applied to the budget itself): the static
     * USABLE/OVERHEAD constants drift as kernels and workspaces change
     * (measured drift ~1 GiB by 2026-07-17, two days after the constants
     * were sized), while the measured number cannot.  The static formula is
     * kept as an upper bound — it still protects against a page-cache- or
     * UVM-inflated MemAvailable reading — and slot 0's committed actual is
     * added back because its bytes are already resident (and therefore
     * already excluded from MemAvailable).  A parse failure keeps the
     * static budget: the live floor check in provision_slot fails closed on
     * its own read. */
    uint64_t kv_budget_final = kv_budget;
    {
        const uint64_t avail_now = server_mem_available_bytes();
        if (avail_now > 0) {
            const uint64_t headroom =
                avail_now > PULSAR_SERVER_MEM_FLOOR_BYTES
                    ? avail_now - PULSAR_SERVER_MEM_FLOOR_BYTES : 0;
            uint64_t measured = session_actual + headroom;
            /* The measured budget is one-shot and PERMANENT, while the
             * MemAvailable it derives from can read transiently low at
             * startup (a previous model process still releasing memory;
             * UVM/MemTotal shrink on driver 610).  A budget refusal is
             * classed "eviction helps" by the scheduler, so a collapsed
             * budget would evict-thrash slot 0 forever instead of ever
             * provisioning slot 1.  Clamp: the budget always structurally
             * permits at least one more session (slot 0's actual + one
             * session estimate — the extra-slot ctx is capped at slot 0's,
             * so its est can never exceed session_est); when memory is
             * GENUINELY tight, the LIVE per-attempt floor check
             * (server_mem_floor_admits — truthful post-warmup) is the guard,
             * and it recovers naturally when memory frees. */
            const uint64_t budget_min = session_actual + session_est;
            if (measured < budget_min) {
                server_log(PULSAR_LOG_WARNING,
                           "pulsar-server: session admission: measured budget "
                           "%.2f GiB clamped up to %.2f GiB (slot 0 actual + "
                           "one session est): MemAvailable %.2f GiB reads low "
                           "at startup; the live MemAvailable floor check "
                           "remains the per-attempt guard",
                           (double)measured / (1024.0 * 1024.0 * 1024.0),
                           (double)budget_min / (1024.0 * 1024.0 * 1024.0),
                           (double)avail_now / (1024.0 * 1024.0 * 1024.0));
                measured = budget_min;
            }
            if (measured < kv_budget_final) kv_budget_final = measured;
            server_log(PULSAR_LOG_DEFAULT,
                       "pulsar-server: session admission: measured budget %.2f GiB "
                       "(MemAvailable %.2f GiB post-warmup - floor %.2f GiB "
                       "+ slot 0 committed %.2f GiB; static bound %.2f GiB)",
                       (double)kv_budget_final / (1024.0 * 1024.0 * 1024.0),
                       (double)avail_now / (1024.0 * 1024.0 * 1024.0),
                       (double)PULSAR_SERVER_MEM_FLOOR_BYTES / (1024.0 * 1024.0 * 1024.0),
                       (double)session_actual / (1024.0 * 1024.0 * 1024.0),
                       (double)kv_budget / (1024.0 * 1024.0 * 1024.0));
        }
    }

    server s;
    memset(&s, 0, sizeof(s));
    s.engine = engine;
    s.served_model_id = cfg.served_model_id;      /* --served-model-id override (id) */
    s.served_model_name = cfg.served_model_name;  /* --served-model-name override (name) */
    s.started = time(NULL);          /* uptime origin reported by /health */
    /* Slot 0 is provisioned here at the configured --ctx-size over the ONE
     * session (s.sess). Tier-2: if the created session is bank-pooled
     * (PULSAR_MSEQ_BANKS>1), every other slot maps to one of its banks; the
     * scheduler provisions banks 1..pool_banks-1 lazily as pure host
     * bookkeeping. In classic mode (pool_banks==0) slot 0 is the only slot —
     * a job needing another one queues. */
    const int pool_banks = pulsar_session_bank_count(session);
    s.pool_banks = pool_banks > 1 ? pool_banks : 0;
    s.live_bank = 0;
    /* Three-way scheduler knob (worker_main): decode banks <= spec_max_live run
     * the per-bank spec/plain time-slice lane; more than that batch. DATA-SET
     * from the multiseq gate on THIS v5mx build (2026-07-18, PULSAR_MSEQ_BANKS=3,
     * 512 tok/session, engine-level): batched N=1/2/3 = 16.2/22.85/27.92 t/s
     * aggregate vs spec-time-slice 16.88/16.86/16.65. v5mx's ~55-62% spec accept
     * means time-slicing yields no aggregate gain past N=1, so batching wins from
     * N=2 (+36% at N=2). The crossover is therefore at 1, NOT the plan's stale
     * §2.2 "3" (that was the 86%-accept compact model). Default 1 = spec only
     * when alone (N=1 byte-identical), batch at N>=2. Env PULSAR_SERVER_SPEC_MAX_LIVE
     * retunes without a rebuild (e.g. =2 to force spec through N=2). */
    {
        const char *sm = getenv("DS4_SERVER_SPEC_MAX_LIVE");
        int v = sm ? atoi(sm) : 1;
        if (v < 1) v = 1;
        if (s.pool_banks > 0 && v > s.pool_banks) v = s.pool_banks;
        s.spec_max_live = v;
    }
    /* plan-34 phase-2 inc 5: fused mixed-batch lane. Default OFF (OPT-IN via
     * PULSAR_MIXED_BATCH=1). It is a workload-dependent TRADE, not a safe blanket
     * default — three measured regimes:
     *   (a) shallow / synchronized concurrent  -> NEUTRAL (teb --perf-only 2026-07-27:
     *       ON==OFF within noise at d0/d4096, all concurrencies).
     *   (b) one big prefill overlapping a few shallow decoders (plan-34, K=8000
     *       ndec=3, chunk 8) -> better tail/throughput (the reason it exists).
     *   (c) MULTIPLE DEEP concurrent streams -> REGRESSES HARD. teb --perf-only
     *       d8192/c4 reproduced ON 265 pp / 16.0 tg vs OFF 373 / 30.6 = ~-29%
     *       prefill, ~-48% decode (the deep KV decode step is already bandwidth-
     *       saturated, so folding prefill in displaces decode catastrophically).
     * Was briefly flipped default-ON 2026-07-26 on the (a)/(b) reasoning; the
     * teb A/B (2026-07-27) found regime (c) and Tyler reverted it to opt-in —
     * a default that can halve decode t/s at deep-context concurrency is a
     * footgun on a 1M-capable box. Re-enable per-deployment only for a workload
     * verified to be shallow-bursty. DO NOT re-flip default-on without a
     * depth/concurrency guard (auto-disable in regime (c)). PULSAR_MIXED_CHUNK
     * overrides the chunk (c8 = plan-34 tail point). Phase-1 warm-fork TTFT is
     * v0.3.0's headline continuous-batching win. One startup read, no hot-path
     * getenv. Only engages in pool mode. */
    {
        /* Default ON in pool mode WITH the deep-concurrent guard below — the
         * condition the opt-in revert set for re-flipping ("do NOT re-flip
         * default-on without a depth/concurrency guard"). The guard bounds the
         * aggregate committed depth of the active decode set: regime (c)
         * measured NEUTRAL at 4x4096=16384 aggregate rows and -29% pp/-48% tg
         * at 4x8192=32768, so the default threshold 16384 stops fusing exactly
         * where fusing stops paying, and regimes (a)/(b) keep their win.
         * DS4_MIXED_BATCH=0 still forces the lane fully off;
         * DS4_MIXED_DEEP_GUARD_ROWS overrides the threshold (0 = no guard). */
        const char *mb = getenv("DS4_MIXED_BATCH");
        s.mixed_batch_enabled = s.pool_banks > 0 &&
                                !(mb && (mb[0] == '0' || !strcasecmp(mb, "off")));
        const char *mc = getenv("DS4_MIXED_CHUNK");
        int kc = mc ? atoi(mc) : 8;
        if (kc < 1) kc = 1;
        s.mixed_chunk_tokens = kc;
        const char *mg = getenv("DS4_MIXED_DEEP_GUARD_ROWS");
        s.mixed_deep_guard_rows = mg ? atoi(mg) : 16384;
        if (s.mixed_deep_guard_rows < 0) s.mixed_deep_guard_rows = 0;
        server_log(PULSAR_LOG_DEFAULT,
                   "pulsar-server: fused mixed-batch lane %s (chunk=%d/step, deep guard=%d rows)",
                   s.mixed_batch_enabled ? "ENABLED (default; opt out with DS4_MIXED_BATCH=0)"
                                         : "disabled (DS4_MIXED_BATCH)",
                   s.mixed_chunk_tokens,
                   s.mixed_deep_guard_rows);
    }
    /* plan-33 inc B: warm full-prefix fork routing kill-switch. Default ON in
     * pool mode; PULSAR_WARM_FORK=0 restores today's in-place-continuation routing
     * exactly. One startup read — never on a hot path. */
    {
        const char *wf = getenv("DS4_WARM_FORK");
        s.warm_fork_enabled = s.pool_banks > 0 &&
                              !(wf && (wf[0] == '0' || !strcasecmp(wf, "off")));
        /* inc D: partial-cut floor. Default 256 tokens (2 ratio-4 groups of
         * reuse); floor to 128 so a cut always aligns to >= one group of R. */
        s.warm_partial_min = 256;
        const char *wpm = getenv("DS4_WARM_PARTIAL_MIN");
        if (wpm && *wpm) {
            int v = atoi(wpm);
            if (v < 128) v = 128;
            s.warm_partial_min = v;
        }
        server_log(PULSAR_LOG_DEFAULT, "pulsar-server: warm-fork routing %s (partial-min %d)",
                   s.warm_fork_enabled ? "ENABLED" : "disabled", s.warm_partial_min);
    }
    s.n_slots = 1;
    s.kv_budget_bytes = kv_budget_final;
    for (int i = 0; i < PULSAR_SESSION_POOL_CAP; i++) s.slots[i].bank = (uint32_t)i;
    if (s.pool_banks > 0) {
        /* Even-split per-bank ledger charge: the whole pool's admitted cost
         * spread across its banks (conservative — demand-paged reality is
         * smaller). Startup commits bank 0's share; each provisioned bank adds
         * one share, so committed == session_est only when all banks are live,
         * and that sum was already verified <= budget above. */
        s.bank_marginal_bytes = session_est / (uint64_t)pool_banks;
        s.kv_committed_bytes = s.bank_marginal_bytes;
        s.slots[0].est_cost_bytes = s.bank_marginal_bytes;
        server_log(PULSAR_LOG_DEFAULT,
                   "pulsar-server: Tier-2 shared pool ACTIVE: %d banks, "
                   "spec_max_live=%d, per-bank marginal %.2f GiB "
                   "(pool resident actual %.2f GiB)",
                   pool_banks, s.spec_max_live,
                   (double)s.bank_marginal_bytes / (1024.0 * 1024.0 * 1024.0),
                   (double)session_actual / (1024.0 * 1024.0 * 1024.0));
    } else {
        s.kv_committed_bytes = session_actual;
        s.slots[0].est_cost_bytes = session_actual;
    }
    /* Tier-2 task #55 increment 2b — proactive-eviction guard. Enabled only under
     * overcommit with N>1 banks (banks are 1M-capable but total physical is
     * bounded). The guard keeps touched (demand-paged) KV under budget − eager by
     * spilling LRU-idle banks to local fast disk + cudaFree; a returning bank
     * reloads bit-identically. Decisions use the deterministic touched accounting,
     * NEVER the coarse cudaMemGetInfo gauge. PULSAR_SERVER_KV_BUDGET_OVERRIDE (bytes)
     * lowers the budget for the memory-safety smoke so the guard fires at modest
     * fills while the box stays far from real OOM. */
    if (overcommit && s.pool_banks > 0) {
        uint64_t budget = kv_budget_final;
        const char *ov = getenv("DS4_SERVER_KV_BUDGET_OVERRIDE");
        if (ov && ov[0]) {
            unsigned long long b = strtoull(ov, NULL, 10);
            if (b > 0) { budget = (uint64_t)b;
                server_log(PULSAR_LOG_WARNING,
                    "pulsar-server: guard: KV budget OVERRIDDEN to %.2f GiB (test hook)",
                    (double)budget / (1024.0*1024.0*1024.0)); }
        }
        s.guard_eager_bytes = overcommit_admission_est;   /* eager floor resident */
        s.guard_touched_budget = budget > s.guard_eager_bytes
                               ? budget - s.guard_eager_bytes : 0;
        /* Direct touched-budget override (bytes) for the memory-safety smoke: sets
         * the resident demand-paged-KV ceiling the guard keeps under, so it fires
         * at a precise modest fill while the box stays far from real OOM. */
        const char *tb = getenv("DS4_SERVER_GUARD_TOUCHED_BUDGET");
        if (tb && tb[0]) {
            unsigned long long b = strtoull(tb, NULL, 10);
            if (b > 0) { s.guard_touched_budget = (uint64_t)b;
                server_log(PULSAR_LOG_WARNING,
                    "pulsar-server: guard: touched budget OVERRIDDEN to %.3f GiB (test hook)",
                    (double)s.guard_touched_budget / (1024.0*1024.0*1024.0)); }
        }
        const char *sd = getenv("DS4_SERVER_SPILL_DIR");
        snprintf(s.spill_dir, sizeof s.spill_dir, "%s",
                 (sd && sd[0]) ? sd : "./ds4-spill");
        (void)mkdir(s.spill_dir, 0700);                   /* best-effort; may exist */
        s.guard_enabled = (s.guard_touched_budget > 0);
        server_log(PULSAR_LOG_DEFAULT,
                   "pulsar-server: Tier-2 2b guard %s: touched budget %.2f GiB "
                   "(kv budget %.2f − eager %.2f), spill dir %s",
                   s.guard_enabled ? "ENABLED" : "DISABLED (no touched headroom)",
                   (double)s.guard_touched_budget / (1024.0*1024.0*1024.0),
                   (double)budget / (1024.0*1024.0*1024.0),
                   (double)s.guard_eager_bytes / (1024.0*1024.0*1024.0),
                   s.spill_dir);
    }
    s.sess = session;                /* the one session; slot 0 describes it */
    s.slots[0].provisioned = true;
    s.slots[0].state = SLOT_IDLE;
    s.slots[0].ctx_size = cfg.ctx_size;
    /* Slot-routing trivial-match threshold (choose_slot_for_job): the token
     * depth at which a shared prefix stops meaning "same rendered template
     * header" and starts meaning "same conversation". Derived once per model
     * rather than hardcoded: tokenize the largest template-injected text two
     * UNRELATED conversations can share — BOS plus the fixed think-max
     * preamble (prompt_render.c renders both before any client content; the
     * preamble only appears at PULSAR_THINK_MAX, unreachable below a 384K
     * context, but a threshold sized for it stays correct on boxes where it
     * IS reachable and costs nothing here) — then allow
     * PULSAR_SERVER_SLOT_TRIVIAL_ALLOWANCE_TOKENS of incidental prologue
     * overlap on top (see the constant's comment for the sizing evidence). */
    {
        buf hdr = {0};
        buf_puts(&hdr, PULSAR_SERVER_RENDER_BOS);
        buf_puts(&hdr, pulsar_think_max_prefix());
        pulsar_tokens hdr_tokens = {0};
        pulsar_tokenize_rendered_chat(engine, hdr.ptr, &hdr_tokens);
        s.slot_trivial_common_tokens =
            hdr_tokens.len + PULSAR_SERVER_SLOT_TRIVIAL_ALLOWANCE_TOKENS;
        server_log(PULSAR_LOG_DEFAULT,
                   "pulsar-server: slot routing: trivial-match threshold %d tokens "
                   "(template header %d + incidental allowance %d)",
                   s.slot_trivial_common_tokens, hdr_tokens.len,
                   PULSAR_SERVER_SLOT_TRIVIAL_ALLOWANCE_TOKENS);
        pulsar_tokens_free(&hdr_tokens);
        buf_free(&hdr);
    }
    s.default_tokens = cfg.default_tokens;
    s.disable_exact_dsml_tool_replay = cfg.disable_exact_dsml_tool_replay;
    s.tool_mem.max_entries = cfg.tool_memory_max_ids;
    s.enable_cors = cfg.enable_cors;
    if (cfg.kv_disk_dir &&
        !kv_cache_open(&s.kv, cfg.kv_disk_dir, cfg.kv_disk_space_mb,
                       cfg.kv_cache_reject_different_quant, cfg.kv_cache))
    {
        /* Never fatal: an uncreatable/read-only directory logs its reason in
         * kv_cache_open; state the consequence once and serve without disk
         * restore. */
        server_log(PULSAR_LOG_DEFAULT,
                   "pulsar-server: disk KV cache disabled (directory %s unusable); "
                   "serving without disk restore",
                   cfg.kv_disk_dir);
    }
    if (s.disable_exact_dsml_tool_replay) {
        server_log(PULSAR_LOG_DEFAULT,
                   "pulsar-server: exact DSML tool replay disabled; tool history uses canonical JSON rendering");
    }
    pthread_mutex_init(&s.mu, NULL);
    pthread_cond_init(&s.cv, NULL);
    pthread_cond_init(&s.clients_cv, NULL);
    pthread_mutex_init(&s.tool_mu, NULL);
    pthread_mutex_init(&s.trace_mu, NULL);
    if (cfg.trace_path) {
        s.trace = fopen(cfg.trace_path, "w");
        if (!s.trace) {
            server_log(PULSAR_LOG_DEFAULT, "pulsar-server: failed to open trace file %s: %s",
                       cfg.trace_path, strerror(errno));
            server_close_resources(&s);
            return 1;
        }
        setvbuf(s.trace, NULL, _IONBF, 0);
        server_log(PULSAR_LOG_DEFAULT, "pulsar-server: tracing session to %s", cfg.trace_path);
    }

    /* Seed the /metrics snapshots (slot-0 position, spec-decode config like
     * max_draft) before any client thread can scrape: send_metrics never
     * calls into the engine (CUDA-state audit, pulsar_server_internal.h). This
     * runs before the worker thread starts, so it is still single-threaded
     * engine access. */
    server_publish_metrics_snapshot(&s);

    memdiag mdiag;
    const bool mdiag_on = memdiag_start(&mdiag, &s);

    pthread_t worker;
    if (pthread_create(&worker, NULL, worker_main, &s) != 0) die("failed to start worker");

    int lfd = listen_on(cfg.host, cfg.port);
    if (lfd < 0) {
        server_log(PULSAR_LOG_DEFAULT, "pulsar-server: failed to listen on %s:%d: %s", cfg.host, cfg.port, strerror(errno));
        pthread_mutex_lock(&s.mu);
        s.stopping = true;
        pthread_cond_broadcast(&s.cv);
        pthread_mutex_unlock(&s.mu);
        pthread_join(worker, NULL);
        if (mdiag_on) pthread_join(mdiag.thread, NULL);
        server_close_resources(&s);
        return 1;
    }
    g_listen_fd = lfd;
    server_log(PULSAR_LOG_DEFAULT, "pulsar-server: listening on http://%s:%d (pulsar %s)",
               cfg.host, cfg.port, PULSAR_VERSION_STR);

    while (!g_stop_requested) {
        int fd = accept(lfd, NULL, NULL);
        if (fd < 0) {
            if (g_stop_requested) break;
            if (errno == EINTR) continue;
            server_log(PULSAR_LOG_DEFAULT, "pulsar-server: accept failed: %s", strerror(errno));
            continue;
        }
        if (g_stop_requested) {
            close(fd);
            break;
        }

        configure_client_socket(fd);
        pthread_mutex_lock(&s.mu);
        const int at_cap = s.clients >= PULSAR_SERVER_MAX_CLIENTS;
        if (!at_cap) s.clients++;
        pthread_mutex_unlock(&s.mu);
        if (at_cap) {
            http_error(fd, s.enable_cors, 503, "too many connections");
            close(fd);
            continue;
        }
        client_arg *ca = (client_arg *)server_xmalloc(sizeof(*ca));
        ca->srv = &s;
        ca->fd = fd;
        pthread_t th;
        if (pthread_create(&th, NULL, client_main, ca) != 0) {
            pthread_mutex_lock(&s.mu);
            s.clients--;
            pthread_cond_broadcast(&s.clients_cv);
            pthread_mutex_unlock(&s.mu);
            free(ca);
            close(fd);
            continue;
        }
        pthread_detach(th);
    }
    if (g_listen_fd >= 0) {
        close(lfd);
        g_listen_fd = -1;
    }

    server_log(PULSAR_LOG_DEFAULT, "pulsar-server: shutdown requested, draining requests");
    pthread_mutex_lock(&s.mu);
    s.stopping = true;
    pthread_cond_broadcast(&s.cv);
    pthread_mutex_unlock(&s.mu);
    pthread_join(worker, NULL);
    if (mdiag_on) pthread_join(mdiag.thread, NULL);
    pthread_mutex_lock(&s.mu);
    while (s.clients > 0) pthread_cond_wait(&s.clients_cv, &s.mu);
    pthread_mutex_unlock(&s.mu);

    for (int i = 0; i < s.n_slots; i++) {
        session_slot *sl = &s.slots[i];
        if (!sl->provisioned) continue;
        /* Tier-2: install this slot's bank so tokens/store read ITS frontier,
         * not the pool's last-live cursor (worker has joined; single-threaded).
         * Must go through server_bank_switch, NOT a bare state_restore: a bank
         * parked by the guard spill has had its comp/index slabs cudaFree'd, and
         * only server_bank_switch reloads them from spill-bank-N.kv first.  The
         * bare restore left gpu_graph_bank_repoint to fail partway through the
         * layer loop, which both lost THIS slot's snapshot and left the graph
         * half-repointed with cur_bank stale — so the next slot whose bank
         * matched that stale cur_bank early-returned "success" over mixed views
         * and silently lost its snapshot too.  Skip the slot if it can't be
         * installed rather than persisting another conversation's frontier. */
        if (s.pool_banks > 0 && !server_bank_switch(&s, sl->bank)) {
            server_log(PULSAR_LOG_DEFAULT,
                       "pulsar-server: slot %d bank %d could not be installed at shutdown; "
                       "skipping its KV persist", i, sl->bank);
            continue;
        }
        const pulsar_tokens *tokens = pulsar_session_tokens(s.sess);
        if (s.kv.enabled && tokens && tokens->len >= s.kv.opt.min_tokens) {
            server_log(PULSAR_LOG_KVCACHE,
                       "pulsar-server: persisting slot %d KV cache before shutdown tokens=%d",
                       i, tokens->len);
            kv_cache_store_current(&s, sl, "shutdown");
        }
    }
    server_close_resources(&s);
    return 0;
}


#endif /* PULSAR_SERVER_TEST */
