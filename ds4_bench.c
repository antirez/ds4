#include "ds4.h"
#include "ds4_distributed.h"
#include "ds4_gpu_args.h"
#include "ds4_help.h"

/* Purpose-built throughput benchmark.
 *
 * The benchmark walks one fixed token sequence to configurable context
 * frontiers, measuring only the newest prefill interval at each frontier.  It
 * then snapshots the live session in memory when the payload is small enough,
 * performs a fixed greedy decode run without allowing EOS, restores the
 * snapshot or replays the prefix, and continues to the next frontier.  Snapshot
 * save/restore time is intentionally outside both timing windows.
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

#define DS4_BENCH_DEFAULT_SNAPSHOT_MAX_BYTES (UINT64_C(1) << 30)
#define DS4_BENCH_SPEC_ACCEPT_CAP 17

typedef struct {
    const char *model_path;
    const char *mtp_path;
    const char *prompt_path;
    const char *chat_prompt_path;
    const char *system;
    const char *csv_path;
    const char *expert_profile_path;
    const char *gpu_vram_arg;
    const char *gpu_devices_arg;
    ds4_backend backend;
    int threads;
    int ctx_start;
    int ctx_max;
    int ctx_alloc;
    int step_incr;
    int gen_tokens;
    int mtp_draft_tokens;
    int power_percent;
    uint32_t prefill_chunk;
    uint32_t ssd_streaming_cache_experts;
    uint64_t ssd_streaming_cache_bytes;
    uint32_t ssd_streaming_full_layers;
    uint32_t ssd_streaming_preload_experts;
    uint64_t simulate_used_memory_bytes;
    double step_mul;
    float mtp_margin;
    const char *dump_frontier_logits_dir;
    ds4_dist_options dist;
    bool warm_weights;
    bool quality;
    bool ssd_streaming;
    bool ssd_streaming_cold;
    bool ssd_streaming_full_layers_set;
    bool cuda_tensor_parallel;
    bool show_output;
    bool mtp_options_set;
} bench_config;

static double bench_now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1000000000.0;
}

static uint64_t bench_snapshot_max_bytes(void) {
    const char *env = getenv("DS4_BENCH_SNAPSHOT_MAX_BYTES");
    if (!env || env[0] == '\0') return DS4_BENCH_DEFAULT_SNAPSHOT_MAX_BYTES;
    if (!strcmp(env, "unlimited") || !strcmp(env, "UNLIMITED") ||
        !strcmp(env, "inf") || !strcmp(env, "INF")) {
        return UINT64_MAX;
    }
    char *end = NULL;
    unsigned long long v = strtoull(env, &end, 10);
    if (env[0] == '\0' || !end || *end != '\0') {
        fprintf(stderr,
                "ds4-bench: invalid DS4_BENCH_SNAPSHOT_MAX_BYTES=%s; using default %llu\n",
                env,
                (unsigned long long)DS4_BENCH_DEFAULT_SNAPSHOT_MAX_BYTES);
        return DS4_BENCH_DEFAULT_SNAPSHOT_MAX_BYTES;
    }
    return (uint64_t)v;
}

static double bytes_to_gib(uint64_t bytes) {
    return (double)bytes / (1024.0 * 1024.0 * 1024.0);
}

static void usage(FILE *fp, const char *topic) {
    ds4_help_print(fp, DS4_HELP_BENCH, topic);
    if (!topic || !strcmp(topic, "all") || !strcmp(topic, "runtime") ||
        !strcmp(topic, "benchmark")) {
        fprintf(fp,
                "MTP Decode\n"
                "  --mtp FILE                 Optional MTP support GGUF used for speculative greedy decode.\n"
                "  --mtp-draft N              Maximum autoregressive MTP draft tokens. Default: 1\n"
                "  --mtp-margin F             Verifier confidence margin, 0..1000. Default: 3\n\n");
    }
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

static int parse_nonnegative_int(const char *s, const char *opt) {
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (s[0] == '\0' || *end != '\0' || v < 0 || v > INT_MAX) {
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
#ifdef DS4_ROCM_BUILD
    if (!strcmp(s, "rocm")) return DS4_BACKEND_CUDA;
#else
    if (!strcmp(s, "cuda")) return DS4_BACKEND_CUDA;
#endif
    if (!strcmp(s, "cpu")) return DS4_BACKEND_CPU;
    fprintf(stderr, "ds4-bench: invalid value for %s: %s\n", opt, s);
#ifdef DS4_ROCM_BUILD
    fprintf(stderr, "ds4-bench: valid backends are: metal, rocm, cpu\n");
#else
    fprintf(stderr, "ds4-bench: valid backends are: metal, cuda, cpu\n");
#endif
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
        .step_mul = 1.0,
        .mtp_margin = 3.0f,
    };

    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        if (!strcmp(arg, "-h") || !strcmp(arg, "--help")) {
            const char *topic = (i + 1 < argc && argv[i + 1][0] != '-') ?
                argv[i + 1] : NULL;
            usage(stdout, topic);
            exit(0);
        }
        char dist_parse_err[256] = {0};
        ds4_dist_cli_parse_result dist_parse =
            ds4_dist_parse_cli_arg(arg,
                                   &i,
                                   argc,
                                   argv,
                                   &c.dist,
                                   dist_parse_err,
                                   sizeof(dist_parse_err));
        if (dist_parse == DS4_DIST_CLI_ERROR) {
            fprintf(stderr,
                    "ds4-bench: %s\n",
                    dist_parse_err[0] ? dist_parse_err : "invalid distributed option");
            exit(2);
        }
        if (dist_parse == DS4_DIST_CLI_MATCHED) continue;

        if (!strcmp(arg, "-m") || !strcmp(arg, "--model")) {
            c.model_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--mtp")) {
            c.mtp_path = need_arg(&i, argc, argv, arg);
            c.mtp_options_set = true;
        } else if (!strcmp(arg, "--mtp-draft")) {
            c.mtp_draft_tokens = parse_int(need_arg(&i, argc, argv, arg), arg);
            c.mtp_options_set = true;
        } else if (!strcmp(arg, "--mtp-margin")) {
            const double v = parse_double_arg(need_arg(&i, argc, argv, arg), arg);
            if (v < 0.0 || v > 1000.0) {
                fprintf(stderr, "ds4-bench: invalid value for %s: %.17g\n", arg, v);
                exit(2);
            }
            c.mtp_margin = (float)v;
            c.mtp_options_set = true;
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
        } else if (!strcmp(arg, "--expert-profile")) {
            c.expert_profile_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "-t") || !strcmp(arg, "--threads")) {
            c.threads = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--backend")) {
            c.backend = parse_backend(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--metal")) {
            c.backend = DS4_BACKEND_METAL;
#ifdef DS4_ROCM_BUILD
        } else if (!strcmp(arg, "--rocm")) {
            c.backend = DS4_BACKEND_CUDA;
#else
        } else if (!strcmp(arg, "--cuda")) {
            c.backend = DS4_BACKEND_CUDA;
#endif
        } else if (!strcmp(arg, "--gpu-vram")) {
            c.gpu_vram_arg = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--gpu-devices")) {
            c.gpu_devices_arg = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--cuda-tensor-parallel")) {
            c.cuda_tensor_parallel = true;
        } else if (!strcmp(arg, "--cpu")) {
            c.backend = DS4_BACKEND_CPU;
        } else if (!strcmp(arg, "--quality")) {
            c.quality = true;
        } else if (!strcmp(arg, "--ssd-streaming")) {
            c.ssd_streaming = true;
        } else if (!strcmp(arg, "--ssd-streaming-cold")) {
            c.ssd_streaming_cold = true;
        } else if (!strcmp(arg, "--ssd-streaming-cache-experts")) {
            uint32_t experts = 0;
            uint64_t bytes = 0;
            if (!ds4_parse_streaming_cache_experts_arg(
                    need_arg(&i, argc, argv, arg), &experts, &bytes)) {
                fprintf(stderr,
                        "ds4-bench: --ssd-streaming-cache-experts must be a positive count or <number>GB\n");
                exit(2);
            }
            c.ssd_streaming_cache_experts = experts;
            c.ssd_streaming_cache_bytes = bytes;
        } else if (!strcmp(arg, "--ssd-streaming-full-layers")) {
            int v = parse_nonnegative_int(need_arg(&i, argc, argv, arg), arg);
            c.ssd_streaming_full_layers = (uint32_t)v;
            c.ssd_streaming_full_layers_set = true;
        } else if (!strcmp(arg, "--ssd-streaming-preload-experts")) {
            int v = parse_int(need_arg(&i, argc, argv, arg), arg);
            if (v <= 0) {
                fprintf(stderr, "ds4-bench: --ssd-streaming-preload-experts must be positive\n");
                exit(2);
            }
            c.ssd_streaming_preload_experts = (uint32_t)v;
        } else if (!strcmp(arg, "--simulate-used-memory")) {
            if (!ds4_parse_gib_arg(need_arg(&i, argc, argv, arg),
                                   &c.simulate_used_memory_bytes)) {
                fprintf(stderr,
                        "ds4-bench: --simulate-used-memory must be a positive GiB value, e.g. 64GB\n");
                exit(2);
            }
        } else if (!strcmp(arg, "--prefill-chunk")) {
            c.prefill_chunk = (uint32_t)parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--power")) {
            c.power_percent = parse_int(need_arg(&i, argc, argv, arg), arg);
            if (c.power_percent < 1 || c.power_percent > 100) {
                fprintf(stderr, "ds4-bench: --power must be between 1 and 100\n");
                exit(2);
            }
        } else if (!strcmp(arg, "--warm-weights")) {
            c.warm_weights = true;
        } else if (!strcmp(arg, "--show-output")) {
            c.show_output = true;
        } else {
            fprintf(stderr, "ds4-bench: unknown option: %s\n", arg);
            usage(stderr, NULL);
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
    char dist_err[256];
    if (ds4_dist_prepare_engine_options(&c.dist, NULL, dist_err, sizeof(dist_err)) != 0) {
        fprintf(stderr, "ds4-bench: %s\n", dist_err);
        exit(2);
    }
    if (c.dist.role == DS4_DISTRIBUTED_WORKER) {
        fprintf(stderr, "ds4-bench: --role worker is a serving mode; start workers with ./ds4\n");
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

static void log_context_memory(ds4_backend backend,
                               int         ctx_size,
                               uint32_t    prefill_chunk,
                               bool        ssd_streaming) {
    ds4_context_memory m =
        ds4_context_memory_estimate_with_prefill_mode(backend,
                                                      ctx_size,
                                                      prefill_chunk,
                                                      ssd_streaming);
    fprintf(stderr,
            "ds4-bench: context buffers %.2f MiB (ctx=%d, backend=%s, prefill_chunk=%u, raw_kv_rows=%u, compressed_kv_rows=%u)\n",
            (double)m.total_bytes / (1024.0 * 1024.0),
            ctx_size,
            ds4_backend_name(backend),
            m.prefill_cap,
            m.raw_cap,
            m.comp_cap);
}

static int wait_distributed_route(ds4_session *session) {
    char err[256] = {0};
    char last[256] = {0};
    unsigned ticks = 0;
    const struct timespec delay = {0, 250000000L};

    for (;;) {
        int ready = ds4_session_distributed_route_ready(session, err, sizeof(err));
        if (ready > 0) {
            if (ticks) fprintf(stderr, "ds4-bench: distributed route ready\n");
            return 0;
        }
        if (ready < 0) {
            fprintf(stderr,
                    "ds4-bench: distributed route readiness failed: %s\n",
                    err[0] ? err : "unknown error");
            return 1;
        }
        const char *why = err[0] ? err : "route incomplete";
        if (strcmp(last, why) != 0 || (ticks % 20u) == 0) {
            fprintf(stderr, "ds4-bench: waiting for distributed route: %s\n", why);
            snprintf(last, sizeof(last), "%s", why);
        }
        nanosleep(&delay, NULL);
        ticks++;
    }
}

static void maybe_warn_distributed_step_shape(const bench_config *cfg, ds4_session *session) {
    if (!cfg || !session || cfg->dist.role != DS4_DISTRIBUTED_COORDINATOR) return;
    uint32_t chunk = cfg->dist.prefill_chunk;
    if (chunk == 0) {
        const int cap = ds4_session_prefill_cap(session);
        if (cap > 0) chunk = (uint32_t)cap;
    }
    if (chunk == 0) return;
    if (cfg->step_mul == 1.0 &&
        cfg->step_incr > 0 &&
        (uint32_t)cfg->step_incr < chunk &&
        cfg->ctx_start < cfg->ctx_max)
    {
        fprintf(stderr,
                "ds4-bench: note: --step-incr=%d is smaller than distributed prefill chunk %u; "
                "suffix rows will not show multi-chunk pipeline overlap\n",
                cfg->step_incr,
                chunk);
    }
}

int main(int argc, char **argv) {
    bench_config cfg = parse_options(argc, argv);

    /* Hint the packer at the largest ctx this bench run will exercise
     * so per-layer KV bytes are priced for the real session size, not
     * a stale 4096 default. Single-tier and CPU paths ignore this. */
    int placement_ctx_hint = cfg.ctx_max;
    if (cfg.ctx_alloc > placement_ctx_hint) placement_ctx_hint = cfg.ctx_alloc;

    ds4_gpu_config gpu_cfg = {0};
    bool skip_cuda = false;
    const bool have_gpu_config = cfg.gpu_vram_arg || cfg.gpu_devices_arg;
    if (have_gpu_config) {
        char gpu_err[256];
        if (parse_gpu_vram_arg(cfg.gpu_vram_arg, cfg.gpu_devices_arg,
                               &gpu_cfg, &skip_cuda,
                               gpu_err, sizeof(gpu_err)) != 0) {
            fprintf(stderr, "ds4-bench: %s\n", gpu_err);
            return 2;
        }
        cfg.backend = skip_cuda ? DS4_BACKEND_CPU : DS4_BACKEND_CUDA;
    }

    ds4_engine_options opt = {
        .model_path = cfg.model_path,
        .mtp_path = cfg.mtp_path,
        .backend = cfg.backend,
        .n_threads = cfg.threads,
        .context_size = cfg.ctx_alloc,
        .prefill_chunk = cfg.prefill_chunk,
        .mtp_draft_tokens = cfg.mtp_draft_tokens,
        .mtp_margin = cfg.mtp_margin,
        .ssd_streaming_cache_experts = cfg.ssd_streaming_cache_experts,
        .ssd_streaming_cache_bytes = cfg.ssd_streaming_cache_bytes,
        .ssd_streaming_full_layers = cfg.ssd_streaming_full_layers,
        .ssd_streaming_preload_experts = cfg.ssd_streaming_preload_experts,
        .simulate_used_memory_bytes = cfg.simulate_used_memory_bytes,
        .power_percent = cfg.power_percent,
        .warm_weights = cfg.warm_weights,
        .quality = cfg.quality,
        .cuda_tensor_parallel = cfg.cuda_tensor_parallel,
        .ssd_streaming = cfg.ssd_streaming,
        .ssd_streaming_cold = cfg.ssd_streaming_cold,
        .ssd_streaming_full_layers_set = cfg.ssd_streaming_full_layers_set,
        .expert_profile_path = cfg.expert_profile_path,
        .distributed = cfg.dist,
    };
    char dist_err[256];
    if (ds4_dist_prepare_engine_options(&cfg.dist, &opt, dist_err, sizeof(dist_err)) != 0) {
        fprintf(stderr, "ds4-bench: %s\n", dist_err);
        return 2;
    }
    ds4_engine *engine = NULL;
    if (have_gpu_config && !skip_cuda) {
        const bool was_auto =
            (cfg.gpu_vram_arg && !strcmp(cfg.gpu_vram_arg, "auto")) ||
            (!cfg.gpu_vram_arg && cfg.gpu_devices_arg);
        char layout[256];
        if (format_gpu_layout_line(&gpu_cfg, was_auto,
                                   layout, sizeof(layout)) > 0) {
            fprintf(stdout, "%s\n", layout);
            fflush(stdout);
        }
        if (ds4_engine_create_with_gpu_config(
                &engine, &opt, &gpu_cfg) != 0) return 1;
    } else if (ds4_engine_open(&engine, &opt) != 0) {
        return 1;
    }
    const int effective_mtp_draft = ds4_engine_mtp_draft_tokens(engine);
    const bool mtp_disabled_by_env = getenv("DS4_MTP_SPEC_DISABLE") != NULL;
    const bool mtp_active = effective_mtp_draft > 1 && !mtp_disabled_by_env;
    if (cfg.mtp_options_set) {
        if (mtp_active) {
            fprintf(stderr,
                    "ds4-bench: MTP speculative greedy decode enabled "
                    "(effective draft=%d, margin=%.3g)\n",
                    effective_mtp_draft,
                    (double)cfg.mtp_margin);
        } else if (mtp_disabled_by_env && effective_mtp_draft > 1) {
            fprintf(stderr,
                    "ds4-bench: MTP support loaded but speculation is disabled by "
                    "DS4_MTP_SPEC_DISABLE\n");
        } else {
            fprintf(stderr,
                    "ds4-bench: MTP speculative decode is inactive "
                    "(effective draft=%d; provide support with --mtp and set --mtp-draft above 1)\n",
                    effective_mtp_draft);
        }
    }
    log_context_memory(opt.backend,
                       cfg.ctx_alloc,
                       ds4_engine_prefill_chunk(engine),
                       cfg.ssd_streaming);

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
    if (cfg.dist.role == DS4_DISTRIBUTED_COORDINATOR &&
        wait_distributed_route(session) != 0)
    {
        ds4_session_free(session);
        ds4_tokens_free(&prompt);
        ds4_engine_close(engine);
        return 1;
    }
    maybe_warn_distributed_step_shape(&cfg, session);

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
    fprintf(out, "ctx_tokens,prefill_tokens,prefill_tps,gen_tokens,gen_tps,gen_first_ms,gen_steady_tokens,gen_steady_tps,kvcache_bytes");
    if (cfg.mtp_options_set) {
        fprintf(out,
                ",gen_cycles,gen_first_chunk_tokens,mtp_committed_tokens,"
                "mtp_accepted_extras,mtp_max_committed_chunk,mtp_extra_slots,"
                "mtp_slot_acceptance_pct,mtp_eos_recoveries");
    }
    fputc('\n', out);
    fflush(out);

    const int eos = ds4_token_eos(engine);
    const bool distributed = cfg.dist.role == DS4_DISTRIBUTED_COORDINATOR;
    ds4_session_snapshot snap = {0};
    const uint64_t snapshot_max_bytes = bench_snapshot_max_bytes();
    bool warned_large_snapshot = false;
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

        const bool need_restore_after_generation =
            cfg.gen_tokens > 0 && frontier < cfg.ctx_max;
        bool have_snapshot = false;
        if (need_restore_after_generation && !distributed &&
            getenv("DS4_BENCH_DISABLE_SNAPSHOT") == NULL) {
            const uint64_t payload_bytes = ds4_session_payload_bytes(session);
            const bool large_snapshot_forced =
                getenv("DS4_BENCH_FORCE_SNAPSHOT") != NULL;
            if (payload_bytes > snapshot_max_bytes && !large_snapshot_forced) {
                if (!warned_large_snapshot) {
                    fprintf(stderr,
                            "ds4-bench: session payload snapshot is %.2f GiB, above the %.2f GiB benchmark limit; "
                            "replaying prefixes instead (set DS4_BENCH_FORCE_SNAPSHOT=1 to force snapshots)\n",
                            bytes_to_gib(payload_bytes),
                            bytes_to_gib(snapshot_max_bytes));
                    warned_large_snapshot = true;
                }
            } else if (payload_bytes > 0) {
                if (ds4_session_save_snapshot(session, &snap, err, sizeof(err)) != 0) {
                    fprintf(stderr, "ds4-bench: snapshot at %d failed: %s\n", frontier, err);
                    rc = 1;
                    break;
                }
                have_snapshot = true;
            }
        }

        double gen_first_sec = 0.0;
        double gen_steady_sec = 0.0;
        int gen_done = 0;
        int gen_cycles = 0;
        int gen_first_chunk_tokens = 0;
        int mtp_accepted_extras = 0;
        int mtp_max_committed_chunk = 0;
        int mtp_extra_slots = 0;
        int mtp_eos_recoveries = 0;
        bool mtp_acceptance_defensible = true;
        int *gen_token_buf = (cfg.show_output || mtp_active) && cfg.gen_tokens > 0
            ? malloc((size_t)cfg.gen_tokens * sizeof(gen_token_buf[0]))
            : NULL;
        if ((cfg.show_output || mtp_active) && cfg.gen_tokens > 0 && !gen_token_buf) {
            fprintf(stderr, "ds4-bench: out of memory recording generated tokens\n");
            rc = 1;
            break;
        }
        const double gen_t0 = bench_now_sec();
        int gen_token_count = 0;
        while (gen_done < cfg.gen_tokens) {
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

            int committed[DS4_BENCH_SPEC_ACCEPT_CAP];
            int committed_n = 1;
            int accepted_extras_this_cycle = 0;
            const int remaining = cfg.gen_tokens - gen_done;
            const double token_t0 = bench_now_sec();
            if (mtp_active) {
                int extra_slots = effective_mtp_draft;
                if (extra_slots > remaining - 1) extra_slots = remaining - 1;
                if (extra_slots > DS4_BENCH_SPEC_ACCEPT_CAP - 1) {
                    extra_slots = DS4_BENCH_SPEC_ACCEPT_CAP - 1;
                }
                if (extra_slots > 0) mtp_extra_slots += extra_slots;

                committed_n = ds4_session_eval_speculative_argmax(
                        session,
                        token,
                        remaining,
                        eos,
                        committed,
                        DS4_BENCH_SPEC_ACCEPT_CAP,
                        err,
                        sizeof(err));
                if (committed_n <= 0 || committed_n > remaining ||
                    committed_n > DS4_BENCH_SPEC_ACCEPT_CAP) {
                    if (committed_n < 0) {
                        fprintf(stderr,
                                "ds4-bench: speculative decode at frontier %d failed: %s\n",
                                frontier,
                                err);
                    } else {
                        fprintf(stderr,
                                "ds4-bench: speculative decode at frontier %d returned invalid token count %d\n",
                                frontier,
                                committed_n);
                    }
                    rc = 1;
                    break;
                }

                int eos_index = -1;
                for (int i = 0; i < committed_n; i++) {
                    if (committed[i] == eos) {
                        eos_index = i;
                        break;
                    }
                }
                if (eos_index >= 0) {
                    /* The benchmark historically excludes EOS and always emits
                     * exactly --gen-tokens tokens. Rebuild the state immediately
                     * before the accepted EOS, then evaluate the non-EOS runner. */
                    const int canonical_len = frontier + gen_token_count + eos_index;
                    int *canonical_v = malloc((size_t)canonical_len * sizeof(canonical_v[0]));
                    if (!canonical_v) {
                        fprintf(stderr, "ds4-bench: out of memory recovering from speculative EOS\n");
                        rc = 1;
                        break;
                    }
                    memcpy(canonical_v,
                           prefix.v,
                           (size_t)frontier * sizeof(canonical_v[0]));
                    memcpy(canonical_v + frontier,
                           gen_token_buf,
                           (size_t)gen_token_count * sizeof(canonical_v[0]));
                    memcpy(canonical_v + frontier + gen_token_count,
                           committed,
                           (size_t)eos_index * sizeof(canonical_v[0]));
                    ds4_tokens canonical = {
                        .v = canonical_v,
                        .len = canonical_len,
                        .cap = canonical_len,
                    };
                    if (ds4_session_sync(session, &canonical, err, sizeof(err)) != 0) {
                        fprintf(stderr,
                                "ds4-bench: speculative EOS recovery at frontier %d failed: %s\n",
                                frontier,
                                err);
                        free(canonical_v);
                        rc = 1;
                        break;
                    }
                    free(canonical_v);

                    const int non_eos = ds4_session_argmax_excluding(session, eos);
                    if (non_eos < 0 ||
                        ds4_session_eval(session, non_eos, err, sizeof(err)) != 0) {
                        fprintf(stderr,
                                "ds4-bench: non-EOS decode recovery at frontier %d failed: %s\n",
                                frontier,
                                non_eos < 0 ? "no non-EOS token" : err);
                        rc = 1;
                        break;
                    }
                    committed[eos_index] = non_eos;
                    committed_n = eos_index + 1;
                    accepted_extras_this_cycle = eos_index > 1 ? eos_index - 1 : 0;
                    mtp_eos_recoveries++;
                    mtp_acceptance_defensible = false;
                } else {
                    accepted_extras_this_cycle = committed_n - 1;
                }
            } else {
                committed[0] = token;
                if (ds4_session_eval(session, token, err, sizeof(err)) != 0) {
                    fprintf(stderr, "ds4-bench: decode at frontier %d failed: %s\n", frontier, err);
                    rc = 1;
                    break;
                }
            }
            const double token_t1 = bench_now_sec();
            if (gen_cycles == 0) {
                gen_first_sec = token_t1 - token_t0;
                gen_first_chunk_tokens = committed_n;
            }
            else gen_steady_sec += token_t1 - token_t0;
            for (int i = 0; i < committed_n; i++) {
                if (gen_token_buf) gen_token_buf[gen_token_count++] = committed[i];
            }
            gen_done += committed_n;
            gen_cycles++;
            mtp_accepted_extras += accepted_extras_this_cycle;
            if (committed_n > mtp_max_committed_chunk) {
                mtp_max_committed_chunk = committed_n;
            }
        }
        const double gen_t1 = bench_now_sec();
        if (cfg.show_output && gen_token_buf && gen_token_count > 0) {
            fprintf(stderr, "ds4-bench: gen[ctx=%d] decoded text: \"", frontier);
            for (int i = 0; i < gen_token_count; i++) {
                size_t tlen = 0;
                char *txt = ds4_token_text(engine, gen_token_buf[i], &tlen);
                if (txt) {
                    fwrite(txt, 1, tlen, stderr);
                    free(txt);
                }
            }
            fprintf(stderr, "\"\n");
            fflush(stderr);
        }
        free(gen_token_buf);
        if (rc != 0) break;

        if (!need_restore_after_generation) {
            /* Nothing later depends on the frontier state. */
        } else if (distributed || !have_snapshot) {
            if (ds4_session_sync(session, &prefix, err, sizeof(err)) != 0) {
                fprintf(stderr, "ds4-bench: replay restore at %d failed: %s\n", frontier, err);
                rc = 1;
                break;
            }
        } else {
            if (ds4_session_load_snapshot(session, &snap, err, sizeof(err)) != 0) {
                fprintf(stderr, "ds4-bench: restore at %d failed: %s\n", frontier, err);
                rc = 1;
                break;
            }
        }

        const double gen_sec = gen_t1 - gen_t0;
        const int gen_steady_tokens = gen_done > gen_first_chunk_tokens
            ? gen_done - gen_first_chunk_tokens
            : 0;
        fprintf(out,
                "%d,%d,%.2f,%d,%.2f,%.3f,%d,%.2f,%llu",
                frontier,
                prefill_tokens,
                prefill_sec > 0.0 ? (double)prefill_tokens / prefill_sec : 0.0,
                gen_done,
                gen_sec > 0.0 ? (double)gen_done / gen_sec : 0.0,
                gen_first_sec * 1000.0,
                gen_steady_tokens,
                gen_steady_sec > 0.0 ? (double)gen_steady_tokens / gen_steady_sec : 0.0,
                (unsigned long long)(have_snapshot ? snap.len : 0));
        if (cfg.mtp_options_set) {
            fprintf(out,
                    ",%d,%d,%d,%d,%d,%d,",
                    gen_cycles,
                    gen_first_chunk_tokens,
                    gen_done,
                    mtp_accepted_extras,
                    mtp_max_committed_chunk,
                    mtp_extra_slots);
            if (mtp_acceptance_defensible && mtp_extra_slots > 0) {
                fprintf(out,
                        "%.2f",
                        100.0 * (double)mtp_accepted_extras / (double)mtp_extra_slots);
            }
            fprintf(out, ",%d", mtp_eos_recoveries);
        }
        fputc('\n', out);
        fflush(out);

        if (cfg.mtp_options_set) {
            fprintf(stderr,
                    "ds4-bench: mtp[ctx=%d] active=%s cycles=%d committed=%d "
                    "accepted-extras=%d max-chunk=%d slots=%d slot-acceptance=",
                    frontier,
                    mtp_active ? "yes" : "no",
                    gen_cycles,
                    gen_done,
                    mtp_accepted_extras,
                    mtp_max_committed_chunk,
                    mtp_extra_slots);
            if (mtp_acceptance_defensible && mtp_extra_slots > 0) {
                fprintf(stderr,
                        "%.2f%%",
                        100.0 * (double)mtp_accepted_extras / (double)mtp_extra_slots);
            } else {
                fputs("n/a", stderr);
            }
            fprintf(stderr, " eos-recoveries=%d\n", mtp_eos_recoveries);
        }

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
