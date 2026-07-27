#include "pulsar.h"
#include "pulsar_help.h"

/* Purpose-built throughput benchmark.
 *
 * The benchmark walks one fixed token sequence to configurable context
 * frontiers, measuring only the newest prefill interval at each frontier.  It
 * then snapshots the live session in memory, performs a fixed greedy decode
 * run without allowing EOS, restores the snapshot, and continues to the next
 * frontier.  Snapshot save/restore time is intentionally outside both timing
 * windows.
 */

#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
    const char *model_path;
    const char *prompt_path;
    const char *chat_prompt_path;
    const char *system;
    const char *csv_path;
    pulsar_backend backend;
    int threads;
    int ctx_start;
    int ctx_max;
    int ctx_alloc;
    int step_incr;
    int gen_tokens;
    uint32_t prefill_chunk;
    double step_mul;
    const char *dump_frontier_logits_dir;
    bool warm_weights;
    bool quality;
} bench_config;

static double bench_now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1000000000.0;
}

static void usage(FILE *fp, const char *topic) {
    pulsar_help_print(fp, PULSAR_HELP_BENCH, topic);
}

static int parse_int(const char *s, const char *opt) {
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (s[0] == '\0' || *end != '\0' || v <= 0 || v > INT_MAX) {
        fprintf(stderr, "pulsar-bench: invalid value for %s: %s\n", opt, s);
        exit(2);
    }
    return (int)v;
}

static int parse_nonnegative_int(const char *s, const char *opt) {
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (s[0] == '\0' || *end != '\0' || v < 0 || v > INT_MAX) {
        fprintf(stderr, "pulsar-bench: invalid value for %s: %s\n", opt, s);
        exit(2);
    }
    return (int)v;
}

static double parse_double_arg(const char *s, const char *opt) {
    char *end = NULL;
    double v = strtod(s, &end);
    if (s[0] == '\0' || *end != '\0' || !isfinite(v)) {
        fprintf(stderr, "pulsar-bench: invalid value for %s: %s\n", opt, s);
        exit(2);
    }
    return v;
}

static const char *need_arg(int *i, int argc, char **argv, const char *opt) {
    if (*i + 1 >= argc) {
        fprintf(stderr, "pulsar-bench: %s requires an argument\n", opt);
        exit(2);
    }
    return argv[++*i];
}

static pulsar_backend parse_backend(const char *s, const char *opt) {
    if (!strcmp(s, "cuda")) return PULSAR_BACKEND_CUDA;
    fprintf(stderr, "pulsar-bench: invalid value for %s: %s\n", opt, s);
    fprintf(stderr, "pulsar-bench: valid backends are: cuda\n");
    exit(2);
}

static pulsar_backend default_backend(void) {
    return PULSAR_BACKEND_CUDA;
}

static char *read_file(const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "pulsar-bench: failed to open %s: %s\n", path, strerror(errno));
        exit(1);
    }
    if (fseek(fp, 0, SEEK_END) != 0) {
        fprintf(stderr, "pulsar-bench: failed to seek %s\n", path);
        fclose(fp);
        exit(1);
    }
    long n = ftell(fp);
    if (n < 0) {
        fprintf(stderr, "pulsar-bench: failed to tell %s\n", path);
        fclose(fp);
        exit(1);
    }
    if (fseek(fp, 0, SEEK_SET) != 0) {
        fprintf(stderr, "pulsar-bench: failed to rewind %s\n", path);
        fclose(fp);
        exit(1);
    }
    char *buf = (char *)malloc((size_t)n + 1);
    if (!buf) {
        fprintf(stderr, "pulsar-bench: out of memory reading %s\n", path);
        fclose(fp);
        exit(1);
    }
    if (fread(buf, 1, (size_t)n, fp) != (size_t)n) {
        fprintf(stderr, "pulsar-bench: failed to read %s\n", path);
        free(buf);
        fclose(fp);
        exit(1);
    }
    fclose(fp);
    buf[n] = '\0';
    return buf;
}

static bench_config parse_options(int argc, char **argv) {
    bench_config c = {
        .model_path = "ds4flash.gguf",
        .system = "You are a helpful assistant.",
        .backend = default_backend(),
        .ctx_start = 2048,
        .ctx_max = 32768,
        .step_incr = 2048,
        .gen_tokens = 128,
        .step_mul = 1.0,
    };

    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        if (!strcmp(arg, "-h") || !strcmp(arg, "--help")) {
            const char *topic = (i + 1 < argc && argv[i + 1][0] != '-') ?
                argv[i + 1] : NULL;
            usage(stdout, topic);
            exit(0);
        }
        if (!strcmp(arg, "-m") || !strcmp(arg, "--model")) {
            c.model_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--prompt-file")) {
            c.prompt_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--chat-prompt-file")) {
            c.chat_prompt_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "-sys") || !strcmp(arg, "--system")) {
            c.system = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--ctx-start")) {
            c.ctx_start = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--ctx-max")) {
            c.ctx_max = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--ctx-alloc")) {
            c.ctx_alloc = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--step-incr")) {
            c.step_incr = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--step-mul")) {
            c.step_mul = parse_double_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--gen-tokens") || !strcmp(arg, "--tokens") || !strcmp(arg, "-n")) {
            c.gen_tokens = parse_nonnegative_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--csv")) {
            c.csv_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--dump-frontier-logits-dir")) {
            c.dump_frontier_logits_dir = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "-t") || !strcmp(arg, "--threads")) {
            c.threads = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--backend")) {
            c.backend = parse_backend(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--cuda")) {
            c.backend = PULSAR_BACKEND_CUDA;
        } else if (!strcmp(arg, "--quality")) {
            c.quality = true;
        } else if (!strcmp(arg, "--prefill-chunk")) {
            c.prefill_chunk = (uint32_t)parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--warm-weights")) {
            c.warm_weights = true;
        } else {
            fprintf(stderr, "pulsar-bench: unknown option: %s\n", arg);
            usage(stderr, NULL);
            exit(2);
        }
    }

    if (!!c.prompt_path == !!c.chat_prompt_path) {
        fprintf(stderr, "pulsar-bench: specify exactly one of --prompt-file or --chat-prompt-file\n");
        exit(2);
    }
    if (c.ctx_start > c.ctx_max) {
        fprintf(stderr, "pulsar-bench: --ctx-start must be <= --ctx-max\n");
        exit(2);
    }
    if (c.step_mul < 1.0) {
        fprintf(stderr, "pulsar-bench: --step-mul must be >= 1\n");
        exit(2);
    }
    if (c.step_mul == 1.0 && c.step_incr <= 0) {
        fprintf(stderr, "pulsar-bench: --step-incr must be positive when --step-mul is 1\n");
        exit(2);
    }
    if (c.ctx_max > INT_MAX - c.gen_tokens - 1) {
        fprintf(stderr, "pulsar-bench: requested context is too large\n");
        exit(2);
    }
    if (c.ctx_alloc == 0) c.ctx_alloc = c.ctx_max + c.gen_tokens + 1;
    if (c.ctx_alloc <= c.ctx_max + c.gen_tokens) {
        fprintf(stderr, "pulsar-bench: --ctx-alloc must be greater than ctx-max + gen-tokens\n");
        exit(2);
    }
    return c;
}

static void json_write_string(FILE *fp, const char *s) {
    fputc('"', fp);
    if (s) {
        for (const unsigned char *p = (const unsigned char *)s; *p; p++) {
            switch (*p) {
            case '"':  fputs("\\\"", fp); break;
            case '\\': fputs("\\\\", fp); break;
            case '\b': fputs("\\b", fp); break;
            case '\f': fputs("\\f", fp); break;
            case '\n': fputs("\\n", fp); break;
            case '\r': fputs("\\r", fp); break;
            case '\t': fputs("\\t", fp); break;
            default:
                if (*p < 0x20) fprintf(fp, "\\u%04x", (unsigned)*p);
                else fputc((char)*p, fp);
                break;
            }
        }
    }
    fputc('"', fp);
}

static int write_frontier_logits_json(
        const bench_config *cfg,
        pulsar_engine         *engine,
        pulsar_session        *session,
        int                 frontier,
        int                 previous) {
    if (!cfg->dump_frontier_logits_dir) return 0;

    const int vocab = pulsar_engine_vocab_size(engine);
    float *logits = (float *)malloc((size_t)vocab * sizeof(logits[0]));
    if (!logits) {
        fprintf(stderr, "pulsar-bench: out of memory copying frontier logits\n");
        return 1;
    }
    if (pulsar_session_copy_logits(session, logits, vocab) != vocab) {
        fprintf(stderr, "pulsar-bench: failed to copy frontier logits at %d\n", frontier);
        free(logits);
        return 1;
    }

    char path[PATH_MAX];
    const int n = snprintf(path,
                           sizeof(path),
                           "%s/frontier_%06d.logits.json",
                           cfg->dump_frontier_logits_dir,
                           frontier);
    if (n <= 0 || (size_t)n >= sizeof(path)) {
        fprintf(stderr, "pulsar-bench: frontier logits path is too long\n");
        free(logits);
        return 1;
    }

    FILE *fp = fopen(path, "wb");
    if (!fp) {
        fprintf(stderr, "pulsar-bench: failed to open %s: %s\n", path, strerror(errno));
        free(logits);
        return 1;
    }

    const int argmax = pulsar_session_argmax(session);
    fprintf(fp, "{\n  \"source\":\"pulsar-bench\",\n  \"model\":");
    json_write_string(fp, cfg->model_path);
    fprintf(fp,
            ",\n  \"backend\":\"%s\",\n  \"quality\":%s,\n"
            "  \"quant_bits\":%d,\n  \"prompt_tokens\":%d,\n"
            "  \"frontier_tokens\":%d,\n  \"prefill_tokens\":%d,\n"
            "  \"ctx\":%d,\n  \"vocab\":%d,\n"
            "  \"argmax_id\":%d,\n  \"argmax_logit\":%.9g,\n  \"logits\":[",
            pulsar_backend_name(cfg->backend),
            cfg->quality ? "true" : "false",
            pulsar_engine_routed_quant_bits(engine),
            frontier,
            frontier,
            frontier - previous,
            cfg->ctx_alloc,
            vocab,
            argmax,
            logits[argmax]);
    for (int i = 0; i < vocab; i++) {
        if (i) fputc(',', fp);
        if ((i % 8) == 0) fputs("\n    ", fp);
        if (isfinite(logits[i])) fprintf(fp, "%.9g", logits[i]);
        else fputs("null", fp);
    }
    fputs("\n  ]\n}\n", fp);
    if (fclose(fp) != 0) {
        fprintf(stderr, "pulsar-bench: failed to close %s\n", path);
        free(logits);
        return 1;
    }
    free(logits);
    return 0;
}

static int next_frontier(const bench_config *c, int cur) {
    if (cur >= c->ctx_max) return c->ctx_max;
    int next;
    if (c->step_mul == 1.0) {
        if (cur > INT_MAX - c->step_incr) next = c->ctx_max;
        else next = cur + c->step_incr;
    } else {
        const double v = ceil((double)cur * c->step_mul);
        next = v > (double)INT_MAX ? c->ctx_max : (int)v;
        if (next <= cur) next = cur + 1;
    }
    if (next > c->ctx_max) next = c->ctx_max;
    return next;
}

static void log_context_memory(pulsar_backend backend,
                               int         ctx_size,
                               uint32_t    prefill_chunk) {
    pulsar_context_memory m =
        pulsar_context_memory_estimate_with_prefill(backend,
                                                 ctx_size,
                                                 prefill_chunk);
    fprintf(stderr,
            "pulsar-bench: context buffers %.2f MiB (ctx=%d, backend=%s, prefill_chunk=%u, raw_kv_rows=%u, compressed_kv_rows=%u)\n",
            (double)m.total_bytes / (1024.0 * 1024.0),
            ctx_size,
            pulsar_backend_name(backend),
            m.prefill_cap,
            m.raw_cap,
            m.comp_cap);
}

int main(int argc, char **argv) {
    bench_config cfg = parse_options(argc, argv);

    pulsar_engine_options opt = {
        .model_path = cfg.model_path,
        /* bench measures the plain decode baseline; never bind a merged
         * drafter (spec-decode throughput has its own timed-server protocol) */
        .dspark_disable = true,
        .backend = cfg.backend,
        .n_threads = cfg.threads,
        .prefill_chunk = cfg.prefill_chunk,
        .warm_weights = cfg.warm_weights,
        .quality = cfg.quality,
    };
    pulsar_engine *engine = NULL;
    if (pulsar_engine_open(&engine, &opt) != 0) return 1;
    log_context_memory(cfg.backend, cfg.ctx_alloc, cfg.prefill_chunk);

    char *text = read_file(cfg.prompt_path ? cfg.prompt_path : cfg.chat_prompt_path);
    pulsar_tokens prompt = {0};
    if (cfg.chat_prompt_path) {
        pulsar_encode_chat_prompt(engine, cfg.system, text, PULSAR_THINK_NONE, &prompt);
    } else {
        pulsar_tokenize_text(engine, text, &prompt);
    }
    free(text);

    if (prompt.len < cfg.ctx_max) {
        fprintf(stderr,
                "pulsar-bench: prompt has %d tokens, need at least --ctx-max=%d\n",
                prompt.len,
                cfg.ctx_max);
        pulsar_tokens_free(&prompt);
        pulsar_engine_close(engine);
        return 1;
    }

    pulsar_session *session = NULL;
    if (pulsar_session_create(&session, engine, cfg.ctx_alloc) != 0) {
        fprintf(stderr, "pulsar-bench: failed to create session\n");
        pulsar_tokens_free(&prompt);
        pulsar_engine_close(engine);
        return 1;
    }
    FILE *out = stdout;
    if (cfg.csv_path) {
        out = fopen(cfg.csv_path, "wb");
        if (!out) {
            fprintf(stderr, "pulsar-bench: failed to open %s: %s\n", cfg.csv_path, strerror(errno));
            pulsar_session_free(session);
            pulsar_tokens_free(&prompt);
            pulsar_engine_close(engine);
            return 1;
        }
    }
    fprintf(out, "ctx_tokens,prefill_tokens,prefill_tps,gen_tokens,gen_tps,kvcache_bytes\n");
    fflush(out);

    const int eos = pulsar_token_eos(engine);
    pulsar_session_snapshot snap = {0};
    char err[256];
    int previous = 0;
    int rc = 0;

    for (int frontier = cfg.ctx_start; ; frontier = next_frontier(&cfg, frontier)) {
        pulsar_tokens prefix = {
            .v = prompt.v,
            .len = frontier,
            .cap = frontier,
        };

        const double prefill_t0 = bench_now_sec();
        if (pulsar_session_sync(session, &prefix, err, sizeof(err)) != 0) {
            fprintf(stderr, "pulsar-bench: prefill to %d failed: %s\n", frontier, err);
            rc = 1;
            break;
        }
        const double prefill_t1 = bench_now_sec();
        const double prefill_sec = prefill_t1 - prefill_t0;
        const int prefill_tokens = frontier - previous;

        if (write_frontier_logits_json(&cfg, engine, session, frontier, previous) != 0) {
            rc = 1;
            break;
        }

        if (cfg.gen_tokens > 0) {
            if (pulsar_session_save_snapshot(session, &snap, err, sizeof(err)) != 0) {
                fprintf(stderr, "pulsar-bench: snapshot at %d failed: %s\n", frontier, err);
                rc = 1;
                break;
            }
        }

        const double gen_t0 = bench_now_sec();
        for (int i = 0; i < cfg.gen_tokens; i++) {
            if (pulsar_session_pos(session) + 1 >= pulsar_session_ctx(session)) {
                fprintf(stderr, "pulsar-bench: generation would exceed allocated context at frontier %d\n", frontier);
                rc = 1;
                break;
            }
            const int token = pulsar_session_argmax_excluding(session, eos);
            if (token < 0) {
                fprintf(stderr, "pulsar-bench: failed to choose non-EOS token at frontier %d\n", frontier);
                rc = 1;
                break;
            }
            if (pulsar_session_eval(session, token, err, sizeof(err)) != 0) {
                fprintf(stderr, "pulsar-bench: decode at frontier %d failed: %s\n", frontier, err);
                rc = 1;
                break;
            }
        }
        const double gen_t1 = bench_now_sec();
        if (rc != 0) break;

        if (cfg.gen_tokens == 0) {
            /* Pure prefill benchmark: leave the live session at the frontier. */
        } else {
            if (pulsar_session_load_snapshot(session, &snap, err, sizeof(err)) != 0) {
                fprintf(stderr, "pulsar-bench: restore at %d failed: %s\n", frontier, err);
                rc = 1;
                break;
            }
        }

        const double gen_sec = gen_t1 - gen_t0;
        fprintf(out,
                "%d,%d,%.2f,%d,%.2f,%llu\n",
                frontier,
                prefill_tokens,
                prefill_sec > 0.0 ? (double)prefill_tokens / prefill_sec : 0.0,
                cfg.gen_tokens,
                gen_sec > 0.0 ? (double)cfg.gen_tokens / gen_sec : 0.0,
                (unsigned long long)snap.len);
        fflush(out);

        previous = frontier;
        if (frontier >= cfg.ctx_max) break;
    }

    if (out != stdout) fclose(out);
    pulsar_session_snapshot_free(&snap);
    pulsar_session_free(session);
    pulsar_tokens_free(&prompt);
    pulsar_engine_close(engine);
    return rc;
}
