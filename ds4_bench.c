#include "ds4.h"

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
    const char *mtp_path;
    ds4_backend backend;
    int threads;
    int ctx_start;
    int ctx_max;
    int ctx_alloc;
    int step_incr;
    int gen_tokens;
    int mtp_draft_tokens;
    int power_percent;
    float mtp_margin;
    double step_mul;
    const char *dump_frontier_logits_dir;
    bool warm_weights;
    bool quality;
    bool suffix_decoding;
    uint32_t suffix_max_depth;
    uint64_t suffix_memory_budget;
    float suffix_spec_factor;
    float suffix_spec_offset;
    float suffix_min_prob;
} bench_config;

typedef struct {
    int generated;
    int mtp_draft_tokens;
    uint64_t mtp_steps;
    uint64_t mtp_draft_slots;
    uint64_t mtp_accepted_tokens;
    uint64_t mtp_accepted_extra_tokens;
    uint64_t spec_steps;
    double decode_eval_sec;
    double mtp_eval_sec;
    ds4_suffix_stats suffix;
} bench_decode_stats;

static double bench_now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1000000000.0;
}

static void usage(FILE *fp) {
    fprintf(fp,
        "Usage: ds4-bench --prompt-file FILE [options]\n"
        "\n"
        "Benchmarks instantaneous prefill and generation throughput at context\n"
        "frontiers such as 2048, 4096, 6144, ... . Generation is always greedy,\n"
        "skips EOS on the base path, and runs for --gen-tokens tokens unless an\n"
        "accepted speculative draft reaches EOS first.\n"
        "\n"
        "Input:\n"
        "  --prompt-file FILE\n"
        "      Raw benchmark text. The fixed token sequence is sliced at each frontier.\n"
        "  --chat-prompt-file FILE\n"
        "      Render FILE as one no-thinking chat user message, then slice that sequence.\n"
        "  -sys, --system TEXT\n"
        "      System prompt used only with --chat-prompt-file.\n"
        "\n"
        "Model and backend:\n"
        "  -m, --model FILE       GGUF model path. Default: ds4flash.gguf\n"
        "  --mtp FILE             Optional MTP support GGUF for speculative decode.\n"
        "  --mtp-draft N          Maximum autoregressive MTP draft tokens. Default: 1\n"
        "  --mtp-margin F         MTP verifier margin. Default: 3\n"
        "  --metal | --cuda | --cpu | --backend NAME\n"
        "      Select backend explicitly. Defaults to Metal on macOS, CUDA elsewhere.\n"
        "  -t, --threads N        CPU helper threads.\n"
        "  --quality              Prefer exact kernels where applicable.\n"
        "  --warm-weights         Touch mapped tensor pages before benchmarking.\n"
        "  --suffix-decoding      Enable suffix tree speculative decoding.\n"
        "  --suffix-max-depth N   Max sequence depth for suffix tree. Default: 32\n"
        "  --suffix-memory-budget MB\n"
        "      Max tree memory in MB. Default: 64\n"
        "  --suffix-spec-factor F Draft cap = match_len * F + spec_offset. Default: 1.0\n"
        "  --suffix-spec-offset F Additive offset for the draft cap. Default: 0\n"
        "  --suffix-min-prob F    Stop drafting when prob drops below F. Default: 0\n"
        "  --power N              Target GPU duty cycle percentage, 1..100. Default: 100\n"
        "\n"
        "Sweep:\n"
        "  --ctx-start N          First measured frontier. Default: 2048\n"
        "  --ctx-max N            Last measured frontier. Default: 32768\n"
        "  --ctx-alloc N          Allocated context. Default: ctx-max + gen-tokens + 1\n"
        "  --step-mul F           Multiplicative step. Default: 1\n"
        "  --step-incr N          Linear step when --step-mul is 1. Default: 2048\n"
        "  --gen-tokens N         Greedy decode tokens per frontier. Default: 128\n"
        "\n"
        "Output:\n"
        "  --csv FILE             Write CSV there instead of stdout.\n"
        "  --dump-frontier-logits-dir DIR\n"
        "      Write one full-logit JSON file per measured frontier. DIR must exist.\n"
        "  -h, --help             Show this help.\n");
}

static int parse_int(const char *s, const char *opt) {
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (s[0] == '\0' || *end != '\0' || v <= 0 || v > INT_MAX) {
        fprintf(stderr, "ds4-bench: invalid value for %s: %s\n", opt, s);
        exit(2);
    }
    return (int)v;
}

static double parse_double_arg(const char *s, const char *opt) {
    char *end = NULL;
    double v = strtod(s, &end);
    if (s[0] == '\0' || *end != '\0' || !isfinite(v)) {
        fprintf(stderr, "ds4-bench: invalid value for %s: %s\n", opt, s);
        exit(2);
    }
    return v;
}

static const char *need_arg(int *i, int argc, char **argv, const char *opt) {
    if (*i + 1 >= argc) {
        fprintf(stderr, "ds4-bench: %s requires an argument\n", opt);
        exit(2);
    }
    return argv[++*i];
}

static ds4_backend parse_backend(const char *s, const char *opt) {
    if (!strcmp(s, "metal")) return DS4_BACKEND_METAL;
    if (!strcmp(s, "cuda")) return DS4_BACKEND_CUDA;
    if (!strcmp(s, "cpu")) return DS4_BACKEND_CPU;
    fprintf(stderr, "ds4-bench: invalid value for %s: %s\n", opt, s);
    fprintf(stderr, "ds4-bench: valid backends are: metal, cuda, cpu\n");
    exit(2);
}

static ds4_backend default_backend(void) {
#ifdef DS4_NO_GPU
    return DS4_BACKEND_CPU;
#elif defined(__APPLE__)
    return DS4_BACKEND_METAL;
#else
    return DS4_BACKEND_CUDA;
#endif
}

static char *read_file(const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "ds4-bench: failed to open %s: %s\n", path, strerror(errno));
        exit(1);
    }
    if (fseek(fp, 0, SEEK_END) != 0) {
        fprintf(stderr, "ds4-bench: failed to seek %s\n", path);
        fclose(fp);
        exit(1);
    }
    long n = ftell(fp);
    if (n < 0) {
        fprintf(stderr, "ds4-bench: failed to tell %s\n", path);
        fclose(fp);
        exit(1);
    }
    if (fseek(fp, 0, SEEK_SET) != 0) {
        fprintf(stderr, "ds4-bench: failed to rewind %s\n", path);
        fclose(fp);
        exit(1);
    }
    char *buf = malloc((size_t)n + 1);
    if (!buf) {
        fprintf(stderr, "ds4-bench: out of memory reading %s\n", path);
        fclose(fp);
        exit(1);
    }
    if (fread(buf, 1, (size_t)n, fp) != (size_t)n) {
        fprintf(stderr, "ds4-bench: failed to read %s\n", path);
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
        .mtp_draft_tokens = 1,
        .mtp_margin = 3.0f,
        .step_mul = 1.0,
        .suffix_max_depth = 32,
        .suffix_memory_budget = 64ULL * 1024ULL * 1024ULL,
        .suffix_spec_factor = 1.0f,
        .suffix_spec_offset = 0.0f,
        .suffix_min_prob = 0.0f,
    };

    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        if (!strcmp(arg, "-h") || !strcmp(arg, "--help")) {
            usage(stdout);
            exit(0);
        } else if (!strcmp(arg, "-m") || !strcmp(arg, "--model")) {
            c.model_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--mtp")) {
            c.mtp_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--mtp-draft")) {
            c.mtp_draft_tokens = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--mtp-margin")) {
            const double v = parse_double_arg(need_arg(&i, argc, argv, arg), arg);
            if (v < 0.0 || v > 1000.0) {
                fprintf(stderr, "ds4-bench: --mtp-margin must be between 0 and 1000\n");
                exit(2);
            }
            c.mtp_margin = (float)v;
        } else if (!strcmp(arg, "--suffix-decoding")) {
            c.suffix_decoding = true;
        } else if (!strcmp(arg, "--suffix-max-depth")) {
            c.suffix_max_depth = (uint32_t)parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--suffix-memory-budget")) {
            c.suffix_memory_budget = (uint64_t)parse_int(need_arg(&i, argc, argv, arg), arg) * 1024ULL * 1024ULL;
        } else if (!strcmp(arg, "--suffix-spec-factor")) {
            const double v = parse_double_arg(need_arg(&i, argc, argv, arg), arg);
            if (v <= 0.0 || v > 100.0) {
                fprintf(stderr, "ds4-bench: --suffix-spec-factor must be greater than 0 and at most 100\n");
                exit(2);
            }
            c.suffix_spec_factor = (float)v;
        } else if (!strcmp(arg, "--suffix-spec-offset")) {
            const double v = parse_double_arg(need_arg(&i, argc, argv, arg), arg);
            if (v < 0.0 || v > 100.0) {
                fprintf(stderr, "ds4-bench: --suffix-spec-offset must be between 0 and 100\n");
                exit(2);
            }
            c.suffix_spec_offset = (float)v;
        } else if (!strcmp(arg, "--suffix-min-prob")) {
            const double v = parse_double_arg(need_arg(&i, argc, argv, arg), arg);
            if (v < 0.0 || v > 1.0) {
                fprintf(stderr, "ds4-bench: --suffix-min-prob must be between 0 and 1\n");
                exit(2);
            }
            c.suffix_min_prob = (float)v;
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
            c.gen_tokens = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--csv")) {
            c.csv_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--dump-frontier-logits-dir")) {
            c.dump_frontier_logits_dir = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "-t") || !strcmp(arg, "--threads")) {
            c.threads = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--backend")) {
            c.backend = parse_backend(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--metal")) {
            c.backend = DS4_BACKEND_METAL;
        } else if (!strcmp(arg, "--cuda")) {
            c.backend = DS4_BACKEND_CUDA;
        } else if (!strcmp(arg, "--cpu")) {
            c.backend = DS4_BACKEND_CPU;
        } else if (!strcmp(arg, "--quality")) {
            c.quality = true;
        } else if (!strcmp(arg, "--power")) {
            c.power_percent = parse_int(need_arg(&i, argc, argv, arg), arg);
            if (c.power_percent < 1 || c.power_percent > 100) {
                fprintf(stderr, "ds4-bench: --power must be between 1 and 100\n");
                exit(2);
            }
        } else if (!strcmp(arg, "--warm-weights")) {
            c.warm_weights = true;
        } else {
            fprintf(stderr, "ds4-bench: unknown option: %s\n", arg);
            usage(stderr);
            exit(2);
        }
    }

    if (!!c.prompt_path == !!c.chat_prompt_path) {
        fprintf(stderr, "ds4-bench: specify exactly one of --prompt-file or --chat-prompt-file\n");
        exit(2);
    }
    if (c.ctx_start > c.ctx_max) {
        fprintf(stderr, "ds4-bench: --ctx-start must be <= --ctx-max\n");
        exit(2);
    }
    if (c.step_mul < 1.0) {
        fprintf(stderr, "ds4-bench: --step-mul must be >= 1\n");
        exit(2);
    }
    if (c.step_mul == 1.0 && c.step_incr <= 0) {
        fprintf(stderr, "ds4-bench: --step-incr must be positive when --step-mul is 1\n");
        exit(2);
    }
    if (c.ctx_max > INT_MAX - c.gen_tokens - 1) {
        fprintf(stderr, "ds4-bench: requested context is too large\n");
        exit(2);
    }
    if (c.ctx_alloc == 0) c.ctx_alloc = c.ctx_max + c.gen_tokens + 1;
    if (c.ctx_alloc <= c.ctx_max + c.gen_tokens) {
        fprintf(stderr, "ds4-bench: --ctx-alloc must be greater than ctx-max + gen-tokens\n");
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
        ds4_engine         *engine,
        ds4_session        *session,
        int                 frontier,
        int                 previous) {
    if (!cfg->dump_frontier_logits_dir) return 0;

    const int vocab = ds4_engine_vocab_size(engine);
    float *logits = malloc((size_t)vocab * sizeof(logits[0]));
    if (!logits) {
        fprintf(stderr, "ds4-bench: out of memory copying frontier logits\n");
        return 1;
    }
    if (ds4_session_copy_logits(session, logits, vocab) != vocab) {
        fprintf(stderr, "ds4-bench: failed to copy frontier logits at %d\n", frontier);
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
        fprintf(stderr, "ds4-bench: frontier logits path is too long\n");
        free(logits);
        return 1;
    }

    FILE *fp = fopen(path, "wb");
    if (!fp) {
        fprintf(stderr, "ds4-bench: failed to open %s: %s\n", path, strerror(errno));
        free(logits);
        return 1;
    }

    const int argmax = ds4_session_argmax(session);
    fprintf(fp, "{\n  \"source\":\"ds4-bench\",\n  \"model\":");
    json_write_string(fp, cfg->model_path);
    fprintf(fp,
            ",\n  \"backend\":\"%s\",\n  \"quality\":%s,\n"
            "  \"quant_bits\":%d,\n  \"prompt_tokens\":%d,\n"
            "  \"frontier_tokens\":%d,\n  \"prefill_tokens\":%d,\n"
            "  \"ctx\":%d,\n  \"vocab\":%d,\n"
            "  \"argmax_id\":%d,\n  \"argmax_logit\":%.9g,\n  \"logits\":[",
            ds4_backend_name(cfg->backend),
            cfg->quality ? "true" : "false",
            ds4_engine_routed_quant_bits(engine),
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
        fprintf(stderr, "ds4-bench: failed to close %s\n", path);
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

static void log_context_memory(ds4_backend backend, int ctx_size, const ds4_context_memory *m) {
    fprintf(stderr,
            "ds4-bench: context buffers %.2f MiB "
            "(ctx=%d, backend=%s, raw=%.2f MiB, compressed=%.2f MiB, scratch=%.2f MiB, "
            "prefill_chunk=%u, raw_kv_rows=%u, compressed_kv_rows=%u)\n",
            (double)m->total_bytes / (1024.0 * 1024.0),
            ctx_size,
            ds4_backend_name(backend),
            (double)m->raw_bytes / (1024.0 * 1024.0),
            (double)m->compressed_bytes / (1024.0 * 1024.0),
            (double)m->scratch_bytes / (1024.0 * 1024.0),
            m->prefill_cap,
            m->raw_cap,
            m->comp_cap);
}

int main(int argc, char **argv) {
    bench_config cfg = parse_options(argc, argv);

    ds4_engine_options opt = {
        .model_path = cfg.model_path,
        .mtp_path = cfg.mtp_path,
        .backend = cfg.backend,
        .n_threads = cfg.threads,
        .mtp_draft_tokens = cfg.mtp_draft_tokens,
        .mtp_margin = cfg.mtp_margin,
        .power_percent = cfg.power_percent,
        .warm_weights = cfg.warm_weights,
        .quality = cfg.quality,
        .suffix_decoding = cfg.suffix_decoding,
        .suffix_max_depth = cfg.suffix_max_depth,
        .suffix_memory_budget = cfg.suffix_memory_budget,
        .suffix_spec_factor = cfg.suffix_spec_factor,
        .suffix_spec_offset = cfg.suffix_spec_offset,
        .suffix_min_prob = cfg.suffix_min_prob,
    };
    ds4_engine *engine = NULL;
    if (ds4_engine_open(&engine, &opt) != 0) return 1;
    ds4_context_memory context_memory = ds4_context_memory_estimate(cfg.backend, cfg.ctx_alloc);
    log_context_memory(cfg.backend, cfg.ctx_alloc, &context_memory);
    if (ds4_engine_mtp_draft_tokens(engine) > 1) {
        fprintf(stderr,
                "ds4-bench: MTP speculative decode enabled (draft_tokens=%d, margin=%.3g)\n",
                ds4_engine_mtp_draft_tokens(engine),
                cfg.mtp_margin);
    }
    if (cfg.suffix_decoding) {
        fprintf(stderr,
                "ds4-bench: suffix decoding enabled (max_depth=%u, memory_budget=%llu bytes, "
                "spec_factor=%.2f, spec_offset=%.2f, min_prob=%.4f)\n",
                cfg.suffix_max_depth,
                (unsigned long long)cfg.suffix_memory_budget,
                cfg.suffix_spec_factor,
                cfg.suffix_spec_offset,
                cfg.suffix_min_prob);
    }

    char *text = read_file(cfg.prompt_path ? cfg.prompt_path : cfg.chat_prompt_path);
    ds4_tokens prompt = {0};
    if (cfg.chat_prompt_path) {
        ds4_encode_chat_prompt(engine, cfg.system, text, DS4_THINK_NONE, &prompt);
    } else {
        ds4_tokenize_text(engine, text, &prompt);
    }
    free(text);

    if (prompt.len < cfg.ctx_max) {
        fprintf(stderr,
                "ds4-bench: prompt has %d tokens, need at least --ctx-max=%d\n",
                prompt.len,
                cfg.ctx_max);
        ds4_tokens_free(&prompt);
        ds4_engine_close(engine);
        return 1;
    }

    ds4_session *session = NULL;
    if (ds4_session_create(&session, engine, cfg.ctx_alloc) != 0) {
        fprintf(stderr, "ds4-bench: failed to create session\n");
        ds4_tokens_free(&prompt);
        ds4_engine_close(engine);
        return 1;
    }

    FILE *out = stdout;
    if (cfg.csv_path) {
        out = fopen(cfg.csv_path, "wb");
        if (!out) {
            fprintf(stderr, "ds4-bench: failed to open %s: %s\n", cfg.csv_path, strerror(errno));
            ds4_session_free(session);
            ds4_tokens_free(&prompt);
            ds4_engine_close(engine);
            return 1;
        }
    }
    fprintf(out,
            "ctx_tokens,prefill_tokens,prefill_tps,gen_tokens,gen_tps,kvcache_bytes,"
            "context_total_bytes,context_raw_bytes,context_compressed_bytes,context_scratch_bytes,"
            "context_prefill_cap,context_raw_cap,context_comp_cap,"
            "decode_eval_sec,spec_steps,mtp_draft_tokens,mtp_spec_steps,mtp_draft_slots,"
            "mtp_accepted_tokens,mtp_accepted_extra_tokens,mtp_extra_accept_rate,mtp_eval_sec,"
            "suffix_tree_nodes,suffix_tree_bytes,suffix_draft_attempts,suffix_draft_hits,"
            "suffix_accepted_tokens,suffix_avg_draft_len\n");
    fflush(out);

    const int eos = ds4_token_eos(engine);
    ds4_session_snapshot snap = {0};
    char err[256];
    int previous = 0;
    int rc = 0;

    for (int frontier = cfg.ctx_start; ; frontier = next_frontier(&cfg, frontier)) {
        ds4_tokens prefix = {
            .v = prompt.v,
            .len = frontier,
            .cap = frontier,
        };

        const double prefill_t0 = bench_now_sec();
        if (ds4_session_sync(session, &prefix, err, sizeof(err)) != 0) {
            fprintf(stderr, "ds4-bench: prefill to %d failed: %s\n", frontier, err);
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

        if (ds4_session_save_snapshot(session, &snap, err, sizeof(err)) != 0) {
            fprintf(stderr, "ds4-bench: snapshot at %d failed: %s\n", frontier, err);
            rc = 1;
            break;
        }

        bench_decode_stats decode = {
            .mtp_draft_tokens = ds4_engine_mtp_draft_tokens(engine),
        };
        const bool suffix_spec =
            cfg.suffix_decoding && cfg.backend != DS4_BACKEND_CPU;
        const bool use_spec =
            (decode.mtp_draft_tokens > 1 || suffix_spec) &&
            getenv("DS4_MTP_SPEC_DISABLE") == NULL &&
            getenv("DS4_SPEC_DISABLE") == NULL;
        const double gen_t0 = bench_now_sec();
        while (decode.generated < cfg.gen_tokens) {
            if (ds4_session_pos(session) + 1 >= ds4_session_ctx(session)) {
                fprintf(stderr, "ds4-bench: generation would exceed allocated context at frontier %d\n", frontier);
                rc = 1;
                break;
            }
            const int token = ds4_session_argmax_excluding(session, eos);
            if (token < 0) {
                fprintf(stderr, "ds4-bench: failed to choose non-EOS token at frontier %d\n", frontier);
                rc = 1;
                break;
            }
            int toks[17];
            int ntok = 0;
            const int remaining = cfg.gen_tokens - decode.generated;
            const double eval_t0 = bench_now_sec();
            if (use_spec) {
                decode.spec_steps++;
                if (decode.mtp_draft_tokens > 1) decode.mtp_steps++;
                if (decode.mtp_draft_tokens > 1 && remaining > 1) {
                    int slots = decode.mtp_draft_tokens;
                    if (slots > (int)(sizeof(toks) / sizeof(toks[0])) - 1) {
                        slots = (int)(sizeof(toks) / sizeof(toks[0])) - 1;
                    }
                    if (slots > remaining - 1) slots = remaining - 1;
                    decode.mtp_draft_slots += (uint64_t)slots;
                }
                ntok = ds4_session_eval_speculative_argmax(session,
                                                           token,
                                                           remaining,
                                                           eos,
                                                           toks,
                                                           (int)(sizeof(toks) / sizeof(toks[0])),
                                                           err,
                                                           sizeof(err));
                const double eval_t1 = bench_now_sec();
                decode.decode_eval_sec += eval_t1 - eval_t0;
                if (decode.mtp_draft_tokens > 1) {
                    decode.mtp_eval_sec += eval_t1 - eval_t0;
                }
                if (ntok < 0) {
                    fprintf(stderr, "ds4-bench: speculative decode at frontier %d failed: %s\n", frontier, err);
                    rc = 1;
                    break;
                }
                if (decode.mtp_draft_tokens > 1) {
                    decode.mtp_accepted_tokens += (uint64_t)ntok;
                    if (ntok > 1) decode.mtp_accepted_extra_tokens += (uint64_t)(ntok - 1);
                }
            } else {
                if (ds4_session_eval(session, token, err, sizeof(err)) != 0) {
                    const double eval_t1 = bench_now_sec();
                    decode.decode_eval_sec += eval_t1 - eval_t0;
                    fprintf(stderr, "ds4-bench: decode at frontier %d failed: %s\n", frontier, err);
                    rc = 1;
                    break;
                }
                const double eval_t1 = bench_now_sec();
                decode.decode_eval_sec += eval_t1 - eval_t0;
                toks[0] = token;
                ntok = 1;
            }
            if (ntok <= 0) {
                fprintf(stderr, "ds4-bench: decode at frontier %d accepted no tokens\n", frontier);
                rc = 1;
                break;
            }
            for (int j = 0; j < ntok && decode.generated < cfg.gen_tokens; j++) {
                decode.generated++;
                if (toks[j] == eos) break;
            }
        }
        const double gen_t1 = bench_now_sec();
        if (rc != 0) break;

        ds4_session_suffix_stats(session, &decode.suffix);

        if (ds4_session_load_snapshot(session, &snap, err, sizeof(err)) != 0) {
            fprintf(stderr, "ds4-bench: restore at %d failed: %s\n", frontier, err);
            rc = 1;
            break;
        }

        const double gen_sec = gen_t1 - gen_t0;
        const double mtp_extra_accept_rate =
            decode.mtp_draft_slots != 0
                ? (double)decode.mtp_accepted_extra_tokens / (double)decode.mtp_draft_slots
                : 0.0;
        fprintf(out,
                "%d,%d,%.2f,%d,%.2f,%llu,"
                "%llu,%llu,%llu,%llu,%u,%u,%u,"
                "%.6f,%llu,%d,%llu,%llu,%llu,%llu,%.6f,%.6f,"
                "%llu,%llu,%llu,%llu,%llu,%.6f\n",
                frontier,
                prefill_tokens,
                prefill_sec > 0.0 ? (double)prefill_tokens / prefill_sec : 0.0,
                decode.generated,
                gen_sec > 0.0 ? (double)decode.generated / gen_sec : 0.0,
                (unsigned long long)snap.len,
                (unsigned long long)context_memory.total_bytes,
                (unsigned long long)context_memory.raw_bytes,
                (unsigned long long)context_memory.compressed_bytes,
                (unsigned long long)context_memory.scratch_bytes,
                context_memory.prefill_cap,
                context_memory.raw_cap,
                context_memory.comp_cap,
                decode.decode_eval_sec,
                (unsigned long long)decode.spec_steps,
                decode.mtp_draft_tokens,
                (unsigned long long)decode.mtp_steps,
                (unsigned long long)decode.mtp_draft_slots,
                (unsigned long long)decode.mtp_accepted_tokens,
                (unsigned long long)decode.mtp_accepted_extra_tokens,
                mtp_extra_accept_rate,
                decode.mtp_eval_sec,
                (unsigned long long)decode.suffix.node_count,
                (unsigned long long)decode.suffix.total_bytes,
                (unsigned long long)decode.suffix.query_count,
                (unsigned long long)decode.suffix.query_hits,
                (unsigned long long)decode.suffix.draft_tokens_accepted,
                decode.suffix.query_hits > 0
                    ? (double)decode.suffix.draft_tokens_produced / (double)decode.suffix.query_hits
                    : 0.0);
        fflush(out);

        previous = frontier;
        if (frontier >= cfg.ctx_max) break;
    }

    if (out != stdout) fclose(out);
    ds4_session_snapshot_free(&snap);
    ds4_session_free(session);
    ds4_tokens_free(&prompt);
    ds4_engine_close(engine);
    return rc;
}
