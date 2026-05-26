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
#include <float.h>
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
    ds4_backend backend;
    int threads;
    int ctx_start;
    int ctx_max;
    int ctx_alloc;
    int step_incr;
    int gen_tokens;
    int power_percent;
    double step_mul;
    const char *dump_frontier_logits_dir;
    bool warm_weights;
    bool quality;
    /* KV cache compression simulation; fp8 is the historical default. */
    ds4_kv_dtype kv_dtype;
    /* PPL teacher-forced quality measurement.  When set, skips the
     * throughput sweep and instead tokenizes the file, walks token by
     * token, accumulates -log P(token_t | tokens_<t), and prints
     * nll_avg / ppl / scored_tokens. */
    const char *ppl_prompt_path;
    int          ppl_max_tokens;
    /* Quality validation: KLD + top-K agreement vs a baseline run.
     * --quality-emit FILE writes per-position full-vocab logits during a PPL
     * run; --quality-baseline FILE reads such a dump and compares each
     * position's logit vector to the current run's, reporting KL divergence
     * (mean / max), top-1 agreement %, and top-5 agreement %. */
    const char *quality_emit_path;
    const char *quality_baseline_path;
} bench_config;

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
        "runs for exactly --gen-tokens tokens, and skips EOS so every row is\n"
        "comparable.\n"
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
        "  --metal | --cuda | --cpu | --backend NAME\n"
        "      Select backend explicitly. Defaults to Metal on macOS, CUDA elsewhere.\n"
        "  -t, --threads N        CPU helper threads.\n"
        "  --quality              Prefer exact kernels where applicable.\n"
        "  --kv-cache fp8|turbo3  KV cache compression simulation. Default: fp8 (historical path).\n"
        "  --warm-weights         Touch mapped tensor pages before benchmarking.\n"
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
        .step_mul = 1.0,
    };

    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        if (!strcmp(arg, "-h") || !strcmp(arg, "--help")) {
            usage(stdout);
            exit(0);
        } else if (!strcmp(arg, "-m") || !strcmp(arg, "--model")) {
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
        } else if (!strcmp(arg, "--kv-cache")) {
            const char *kv_name = need_arg(&i, argc, argv, arg);
            if (!ds4_kv_dtype_from_name(kv_name, &c.kv_dtype)) {
                fprintf(stderr, "ds4-bench: unknown --kv-cache value '%s' (expected fp8 or turbo3)\n", kv_name);
                exit(2);
            }
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
            "  \"kv_cache\":\"%s\",\n"
            "  \"quant_bits\":%d,\n  \"prompt_tokens\":%d,\n"
            "  \"frontier_tokens\":%d,\n  \"prefill_tokens\":%d,\n"
            "  \"ctx\":%d,\n  \"vocab\":%d,\n"
            "  \"argmax_id\":%d,\n  \"argmax_logit\":%.9g,\n  \"logits\":[",
            ds4_backend_name(cfg->backend),
            cfg->quality ? "true" : "false",
            ds4_kv_dtype_name(cfg->kv_dtype),
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

static void log_context_memory(ds4_backend backend, int ctx_size) {
    ds4_context_memory m = ds4_context_memory_estimate(backend, ctx_size);
    fprintf(stderr,
            "ds4-bench: context buffers %.2f MiB (ctx=%d, backend=%s, prefill_chunk=%u, raw_kv_rows=%u, compressed_kv_rows=%u)\n",
            (double)m.total_bytes / (1024.0 * 1024.0),
            ctx_size,
            ds4_backend_name(backend),
            m.prefill_cap,
            m.raw_cap,
            m.comp_cap);
}

/* Side-by-side KV footprint for fp8 vs turbo3 at the given backend/ctx.
 * The active dtype is highlighted; the other one is printed for comparison
 * so users can see the packed-byte savings at a glance. */
static void log_kv_footprint_compare(ds4_backend backend, int ctx_size, ds4_kv_dtype active) {
    const ds4_kv_footprint fp8 = ds4_kv_footprint_estimate(backend, ctx_size, DS4_KV_FP8);
    const ds4_kv_footprint t3  = ds4_kv_footprint_estimate(backend, ctx_size, DS4_KV_TURBO3);
    const double mib = 1.0 / (1024.0 * 1024.0);
    const double raw_ratio = (t3.raw_bytes > 0)
            ? ((double)fp8.raw_bytes / (double)t3.raw_bytes) : 0.0;
    /* Print the SWA ring (the only pool that swaps to packed bytes) plus
     * the compressed pools (kept float / F16 because the compressor pool
     * integrates softmax-weighted accumulations that need an original-basis
     * read). */
    fprintf(stderr,
            "ds4-bench: KV footprint @ ctx=%d:\n"
            "  fp8     raw=%.2f MiB  compressed=%.2f MiB  total=%.2f MiB%s\n"
            "  turbo3  raw=%.2f MiB  compressed=%.2f MiB  total=%.2f MiB%s\n"
            "  raw shrink: %.2fx  (turbo3 saves %.2f MiB on the SWA ring)\n",
            ctx_size,
            (double)fp8.raw_bytes * mib,
            (double)fp8.compressed_bytes * mib,
            (double)fp8.total_bytes * mib,
            active == DS4_KV_FP8 ? "  <-- active" : "",
            (double)t3.raw_bytes * mib,
            (double)t3.compressed_bytes * mib,
            (double)t3.total_bytes * mib,
            active == DS4_KV_TURBO3 ? "  <-- active" : "",
            raw_ratio,
            (double)(fp8.raw_bytes - t3.raw_bytes) * mib);
}

/* Quality-dump binary format.  Magic "DS4Q" | u32 vocab | u32 scored
 * | (scored * vocab * float32 logits).  Logits are written RAW, not
 * softmaxed; the comparator runs log-sum-exp on read. */
#define DS4_QDUMP_MAGIC "DS4Q"

/* Streaming softmax: returns log Z so callers can compute log_p = logit - logZ. */
static double ds4_log_sum_exp(const float *logits, int n) {
    float m = logits[0];
    for (int i = 1; i < n; i++) if (logits[i] > m) m = logits[i];
    double s = 0.0;
    for (int i = 0; i < n; i++) s += exp((double)(logits[i] - m));
    return (double)m + log(s);
}

/* Find top-K indices of a logit vector via partial selection.  K small (<=5). */
static void ds4_top_k_indices(const float *logits, int n, int k, int *out_idx) {
    for (int i = 0; i < k; i++) out_idx[i] = -1;
    float out_val[8]; /* k<=8 enforced by callers */
    for (int i = 0; i < k; i++) out_val[i] = -FLT_MAX;
    for (int i = 0; i < n; i++) {
        float v = logits[i];
        if (v <= out_val[k - 1]) continue;
        int j = k - 1;
        while (j > 0 && out_val[j - 1] < v) {
            out_val[j] = out_val[j - 1];
            out_idx[j] = out_idx[j - 1];
            j--;
        }
        out_val[j] = v;
        out_idx[j] = i;
    }
}

/* Teacher-forced perplexity on a token sequence.  For each position i in
 * [0, n-1) feed tokens[i], then read log P(tokens[i+1] | tokens[0..i])
 * from the current logits.  Accumulate -logprob; report mean NLL and
 * exp(mean_NLL).  Compares quality across --kv-cache dtypes apples-to-
 * apples (deterministic, no sampling).
 *
 * Optional modes:
 *   --quality-emit FILE      Dump per-position raw logits to FILE.
 *   --quality-baseline FILE  Read baseline FILE, compare every position's
 *                            logits to current run.  Reports:
 *                              - KLD(baseline || current)  full vocab, mean / max
 *                              - top-1 agreement (target argmax == baseline argmax)
 *                              - top-5 agreement (target argmax in baseline top-5)
 */
static int run_ppl_mode(const bench_config *cfg) {
    ds4_engine_options opt = {
        .model_path = cfg->model_path,
        .backend = cfg->backend,
        .n_threads = cfg->threads,
        .warm_weights = cfg->warm_weights,
        .quality = cfg->quality,
        .kv_dtype = cfg->kv_dtype,
    };
    ds4_engine *engine = NULL;
    if (ds4_engine_open(&engine, &opt) != 0) return 1;

    char *text = read_file(cfg->prompt_path);
    ds4_tokens prompt = {0};
    ds4_tokenize_text(engine, text, &prompt);
    free(text);

    int score_limit = cfg->ppl_max_tokens;
    if (score_limit <= 0 || score_limit > prompt.len) score_limit = prompt.len;
    if (score_limit < 2) {
        fprintf(stderr, "ds4-bench: --ppl-prompt needs at least 2 tokens\n");
        ds4_tokens_free(&prompt);
        ds4_engine_close(engine);
        return 1;
    }

    int ctx_size = score_limit + 16;
    ds4_session *session = NULL;
    if (ds4_session_create(&session, engine, ctx_size) != 0) {
        fprintf(stderr, "ds4-bench: failed to create session\n");
        ds4_tokens_free(&prompt);
        ds4_engine_close(engine);
        return 1;
    }

    const int vocab = ds4_engine_vocab_size(engine);
    float *cur_logits = NULL;
    float *bl_logits  = NULL;
    FILE  *emit_fp    = NULL;
    FILE  *bl_fp      = NULL;
    int    bl_vocab   = 0;
    int    bl_scored  = 0;
    bool   quality_on = (cfg->quality_emit_path || cfg->quality_baseline_path);

    if (quality_on) {
        cur_logits = (float *)malloc((size_t)vocab * sizeof(float));
        if (!cur_logits) {
            fprintf(stderr, "ds4-bench: oom for logit scratch\n");
            ds4_session_free(session); ds4_tokens_free(&prompt); ds4_engine_close(engine);
            return 1;
        }
    }

    if (cfg->quality_emit_path) {
        emit_fp = fopen(cfg->quality_emit_path, "wb");
        if (!emit_fp) {
            fprintf(stderr, "ds4-bench: cannot open --quality-emit '%s': %s\n",
                    cfg->quality_emit_path, strerror(errno));
            free(cur_logits); ds4_session_free(session); ds4_tokens_free(&prompt); ds4_engine_close(engine);
            return 1;
        }
        uint32_t hdr[3] = { 0, (uint32_t)vocab, 0 /* scored placeholder */ };
        memcpy(&hdr[0], DS4_QDUMP_MAGIC, 4);
        fwrite(hdr, sizeof(uint32_t), 3, emit_fp);
    }

    if (cfg->quality_baseline_path) {
        bl_fp = fopen(cfg->quality_baseline_path, "rb");
        if (!bl_fp) {
            fprintf(stderr, "ds4-bench: cannot open --quality-baseline '%s': %s\n",
                    cfg->quality_baseline_path, strerror(errno));
            if (emit_fp) fclose(emit_fp);
            free(cur_logits); ds4_session_free(session); ds4_tokens_free(&prompt); ds4_engine_close(engine);
            return 1;
        }
        uint32_t hdr[3];
        if (fread(hdr, sizeof(uint32_t), 3, bl_fp) != 3 ||
            memcmp(&hdr[0], DS4_QDUMP_MAGIC, 4) != 0) {
            fprintf(stderr, "ds4-bench: '%s' is not a DS4Q baseline dump\n",
                    cfg->quality_baseline_path);
            fclose(bl_fp); if (emit_fp) fclose(emit_fp);
            free(cur_logits); ds4_session_free(session); ds4_tokens_free(&prompt); ds4_engine_close(engine);
            return 1;
        }
        bl_vocab  = (int)hdr[1];
        bl_scored = (int)hdr[2];
        if (bl_vocab != vocab) {
            fprintf(stderr, "ds4-bench: baseline vocab=%d, current vocab=%d (mismatch)\n",
                    bl_vocab, vocab);
            fclose(bl_fp); if (emit_fp) fclose(emit_fp);
            free(cur_logits); ds4_session_free(session); ds4_tokens_free(&prompt); ds4_engine_close(engine);
            return 1;
        }
        bl_logits = (float *)malloc((size_t)vocab * sizeof(float));
        if (!bl_logits) {
            fprintf(stderr, "ds4-bench: oom for baseline scratch\n");
            fclose(bl_fp); if (emit_fp) fclose(emit_fp);
            free(cur_logits); ds4_session_free(session); ds4_tokens_free(&prompt); ds4_engine_close(engine);
            return 1;
        }
    }

    char err[256];
    double nll_sum = 0.0;
    int    scored  = 0;
    /* Quality-vs-baseline accumulators (only used when --quality-baseline). */
    double kld_sum = 0.0;
    double kld_max = 0.0;
    int    top1_match = 0;
    int    top5_match = 0;
    int    qcompared  = 0;

    double t0 = bench_now_sec();
    for (int i = 0; i + 1 < score_limit; i++) {
        if (ds4_session_eval(session, prompt.v[i], err, sizeof err) != 0) {
            fprintf(stderr, "ds4-bench: ppl eval failed at pos %d: %s\n", i, err);
            break;
        }
        ds4_token_score sc;
        if (!ds4_session_token_logprob(session, prompt.v[i + 1], &sc)) {
            fprintf(stderr, "ds4-bench: token_logprob failed at pos %d\n", i);
            break;
        }
        if (!isfinite(sc.logprob)) continue;
        nll_sum += -(double)sc.logprob;
        scored++;

        if (!quality_on) continue;

        int copied = ds4_session_copy_logits(session, cur_logits, vocab);
        if (copied != vocab) {
            fprintf(stderr, "ds4-bench: copy_logits returned %d, expected %d\n", copied, vocab);
            break;
        }

        if (emit_fp) {
            if (fwrite(cur_logits, sizeof(float), (size_t)vocab, emit_fp) != (size_t)vocab) {
                fprintf(stderr, "ds4-bench: quality-emit write failed at pos %d\n", i);
                break;
            }
        }

        if (bl_fp) {
            if (qcompared >= bl_scored) {
                fprintf(stderr, "ds4-bench: baseline has only %d positions, current at %d\n",
                        bl_scored, qcompared);
                break;
            }
            if (fread(bl_logits, sizeof(float), (size_t)vocab, bl_fp) != (size_t)vocab) {
                fprintf(stderr, "ds4-bench: baseline read failed at pos %d\n", qcompared);
                break;
            }
            /* KLD(baseline || current) = sum_v p_b(v) * (log p_b(v) - log p_c(v))
             *                          = sum_v p_b(v) * (logit_b - logZ_b - logit_c + logZ_c) */
            double logZb = ds4_log_sum_exp(bl_logits,  vocab);
            double logZc = ds4_log_sum_exp(cur_logits, vocab);
            double kld   = 0.0;
            for (int v = 0; v < vocab; v++) {
                double pb = exp((double)bl_logits[v] - logZb);
                if (pb <= 0.0) continue;
                double diff = (double)bl_logits[v] - logZb
                            - (double)cur_logits[v] + logZc;
                kld += pb * diff;
            }
            if (kld < 0.0) kld = 0.0; /* numerical floor */
            kld_sum += kld;
            if (kld > kld_max) kld_max = kld;

            /* Top-1/top-5 agreement: target argmax vs baseline top-5 set. */
            int bl_top5[5], cur_top1[1];
            ds4_top_k_indices(bl_logits,  vocab, 5, bl_top5);
            ds4_top_k_indices(cur_logits, vocab, 1, cur_top1);
            if (cur_top1[0] == bl_top5[0]) top1_match++;
            for (int j = 0; j < 5; j++) {
                if (cur_top1[0] == bl_top5[j]) { top5_match++; break; }
            }
            qcompared++;
        }
    }
    double elapsed = bench_now_sec() - t0;

    /* Patch scored count into emit header. */
    if (emit_fp) {
        uint32_t s = (uint32_t)scored;
        fseek(emit_fp, 8, SEEK_SET);
        fwrite(&s, sizeof(uint32_t), 1, emit_fp);
        fclose(emit_fp);
    }
    if (bl_fp) fclose(bl_fp);
    free(cur_logits);
    free(bl_logits);

    const double avg_nll = scored > 0 ? (nll_sum / (double)scored) : 0.0;
    const double ppl     = scored > 0 ? exp(avg_nll) : 0.0;
    const char  *kv_name = ds4_kv_dtype_name(cfg->kv_dtype);
    fprintf(stdout,
            "ds4-bench: PPL teacher-forced  kv_cache=%s  tokens=%d  scored=%d  "
            "elapsed=%.2fs\n"
            "ds4-bench:   nll_avg=%.6f  ppl=%.6f\n",
            kv_name, score_limit, scored, elapsed, avg_nll, ppl);

    if (cfg->quality_emit_path) {
        fprintf(stdout,
                "ds4-bench: quality-emit wrote %d positions x vocab=%d to %s\n",
                scored, vocab, cfg->quality_emit_path);
    }
    if (cfg->quality_baseline_path && qcompared > 0) {
        const double kld_mean = kld_sum / (double)qcompared;
        const double top1_pct = 100.0 * (double)top1_match / (double)qcompared;
        const double top5_pct = 100.0 * (double)top5_match / (double)qcompared;
        fprintf(stdout,
                "ds4-bench: quality vs baseline (%s)  positions=%d\n"
                "ds4-bench:   KLD(baseline||current)  mean=%.6f nats  max=%.6f nats\n"
                "ds4-bench:   top-1 agreement=%.2f%%   top-5 agreement=%.2f%%\n",
                cfg->quality_baseline_path, qcompared,
                kld_mean, kld_max, top1_pct, top5_pct);
    }

    ds4_session_free(session);
    ds4_tokens_free(&prompt);
    ds4_engine_close(engine);
    return 0;
}

int main(int argc, char **argv) {
    bench_config cfg = parse_options(argc, argv);
    log_context_memory(cfg.backend, cfg.ctx_alloc);
    log_kv_footprint_compare(cfg.backend, cfg.ctx_alloc, cfg.kv_dtype);

    ds4_engine_options opt = {
        .model_path = cfg.model_path,
        .backend = cfg.backend,
        .n_threads = cfg.threads,
        .power_percent = cfg.power_percent,
        .warm_weights = cfg.warm_weights,
        .quality = cfg.quality,
        .kv_dtype = cfg.kv_dtype,
    };
    ds4_engine *engine = NULL;
    if (ds4_engine_open(&engine, &opt) != 0) return 1;
    log_context_memory(cfg.backend, cfg.ctx_alloc);

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
    fprintf(out, "ctx_tokens,prefill_tokens,prefill_tps,gen_tokens,gen_tps,kvcache_bytes\n");
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

        const double gen_t0 = bench_now_sec();
        for (int i = 0; i < cfg.gen_tokens; i++) {
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
            if (ds4_session_eval(session, token, err, sizeof(err)) != 0) {
                fprintf(stderr, "ds4-bench: decode at frontier %d failed: %s\n", frontier, err);
                rc = 1;
                break;
            }
        }
        const double gen_t1 = bench_now_sec();
        if (rc != 0) break;

        if (ds4_session_load_snapshot(session, &snap, err, sizeof(err)) != 0) {
            fprintf(stderr, "ds4-bench: restore at %d failed: %s\n", frontier, err);
            rc = 1;
            break;
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
    ds4_session_snapshot_free(&snap);
    ds4_session_free(session);
    ds4_tokens_free(&prompt);
    ds4_engine_close(engine);
    return rc;
}
