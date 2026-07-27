#include "pulsar_agent_internal.h"



/* ============================================================================
 * Small Utilities And Command-Line Parsing
 * ============================================================================
 */

void agent_sigint_handler(int sig) {
    (void)sig;
    agent_sigint = 1;
}



void *agent_xmalloc(size_t n) {
    void *p = (void *)malloc(n ? n : 1);
    if (!p) {
        perror("pulsar-agent: malloc");
        exit(1);
    }
    return p;
}



char *xstrdup(const char *s) {
    if (!s) s = "";
    size_t n = strlen(s);
    char *p = (char *)agent_xmalloc(n + 1);
    memcpy(p, s, n + 1);
    return p;
}



char *xstrndup(const char *s, size_t n) {
    char *p = (char *)agent_xmalloc(n + 1);
    memcpy(p, s, n);
    p[n] = '\0';
    return p;
}



void *agent_xrealloc(void *ptr, size_t n) {
    void *p = (void *)realloc(ptr, n ? n : 1);
    if (!p) {
        perror("pulsar-agent: realloc");
        exit(1);
    }
    return p;
}



void write_all(int fd, const char *p, size_t n) {
    while (n) {
        ssize_t wr = write(fd, p, n);
        if (wr < 0) {
            if (errno == EINTR) continue;
            return;
        }
        p += wr;
        n -= (size_t)wr;
    }
}



void agent_input_buf_append(agent_input_buf *b, const char *s, size_t n) {
    if (!n) return;
    if (b->len + n + 1 > b->cap) {
        size_t cap = b->cap ? b->cap * 2 : 4096;
        while (cap < b->len + n + 1) cap *= 2;
        b->ptr = (char *)agent_xrealloc(b->ptr, cap);
        b->cap = cap;
    }
    memcpy(b->ptr + b->len, s, n);
    b->len += n;
    b->ptr[b->len] = '\0';
}



char *agent_input_buf_take(agent_input_buf *b) {
    if (!b->ptr) return xstrdup("");
    char *p = b->ptr;
    memset(b, 0, sizeof(*b));
    return p;
}



void agent_input_buf_free(agent_input_buf *b) {
    free(b->ptr);
    memset(b, 0, sizeof(*b));
}



static int parse_int(const char *s, const char *opt) {
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (s[0] == '\0' || *end != '\0' || v <= 0 || v > INT32_MAX) {
        fprintf(stderr, "pulsar-agent: invalid value for %s: %s\n", opt, s);
        exit(2);
    }
    return (int)v;
}



static bool agent_slash_command_with_args(const char *cmd, const char *name) {
    size_t len = strlen(name);
    return !strncmp(cmd, name, len) &&
           (cmd[len] == '\0' || isspace((unsigned char)cmd[len]));
}



bool agent_slash_command_known(const char *cmd) {
    return !strcmp(cmd, "/help") ||
           !strcmp(cmd, "/save") ||
           !strcmp(cmd, "/compact") ||
           !strcmp(cmd, "/list") ||
           !strcmp(cmd, "/quit") ||
           !strcmp(cmd, "/exit") ||
           !strcmp(cmd, "/new") ||
           agent_slash_command_with_args(cmd, "/switch") ||
           agent_slash_command_with_args(cmd, "/del") ||
           agent_slash_command_with_args(cmd, "/strip") ||
           agent_slash_command_with_args(cmd, "/history");
}



static uint64_t parse_u64(const char *s, const char *opt) {
    char *end = NULL;
    unsigned long long v = strtoull(s, &end, 10);
    if (s[0] == '\0' || *end != '\0' || v == 0) {
        fprintf(stderr, "pulsar-agent: invalid value for %s: %s\n", opt, s);
        exit(2);
    }
    return (uint64_t)v;
}



static float parse_float_range(const char *s, const char *opt, float min, float max) {
    char *end = NULL;
    float v = strtof(s, &end);
    if (s[0] == '\0' || *end != '\0' || !isfinite(v) || v < min || v > max) {
        fprintf(stderr, "pulsar-agent: invalid value for %s: %s\n", opt, s);
        exit(2);
    }
    return v;
}



static pulsar_backend parse_backend(const char *s) {
    if (!strcmp(s, "cuda")) return PULSAR_BACKEND_CUDA;
    fprintf(stderr, "pulsar-agent: invalid backend: %s\n", s);
    exit(2);
}



static pulsar_backend default_backend(void) {
    return PULSAR_BACKEND_CUDA;
}



double agent_now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1.0e-9;
}



void usage(FILE *fp, const char *topic) {
    pulsar_help_print(fp, PULSAR_HELP_AGENT, topic);
}



static const char *need_arg(int *i, int argc, char **argv, const char *opt) {
    if (*i + 1 >= argc) {
        fprintf(stderr, "pulsar-agent: missing value for %s\n", opt);
        exit(2);
    }
    return argv[++(*i)];
}



agent_config parse_options(int argc, char **argv) {
    agent_config c = {
        .engine = {
            .model_path = "ds4flash.gguf",
            /* agent's per-token DSML/forcing loop can't consume speculative
             * chunks yet; disable the merged drafter until that is wired */
            .dspark_disable = true,
            .backend = default_backend(),
        },
        .gen = {
            .system = "You are a helpful coding assistant running inside pulsar-agent.",
            .n_predict = 50000,
            .ctx_size = 100000,
            .temperature = PULSAR_DEFAULT_TEMPERATURE,
            .top_p = PULSAR_DEFAULT_TOP_P,
            .min_p = PULSAR_DEFAULT_MIN_P,
            .think_mode = PULSAR_THINK_HIGH,
        },
    };

    bool steering_scale_set = false;
    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        if (!strcmp(arg, "-h") || !strcmp(arg, "--help")) {
            const char *topic = (i + 1 < argc && argv[i + 1][0] != '-') ?
                argv[i + 1] : NULL;
            usage(stdout, topic);
            exit(0);
        }
        if (!strcmp(arg, "-p") || !strcmp(arg, "--prompt")) {
            c.gen.prompt = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--non-interactive")) {
            c.non_interactive = true;
        } else if (!strcmp(arg, "-sys") || !strcmp(arg, "--system")) {
            c.gen.system = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--trace")) {
            c.gen.trace_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "-m") || !strcmp(arg, "--model")) {
            c.engine.model_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "-c") || !strcmp(arg, "--ctx")) {
            c.gen.ctx_size = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "-n") || !strcmp(arg, "--tokens")) {
            c.gen.n_predict = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--temp")) {
            c.gen.temperature = parse_float_range(need_arg(&i, argc, argv, arg), arg, 0.0f, 100.0f);
        } else if (!strcmp(arg, "--top-p")) {
            c.gen.top_p = parse_float_range(need_arg(&i, argc, argv, arg), arg, 0.0f, 1.0f);
        } else if (!strcmp(arg, "--min-p")) {
            c.gen.min_p = parse_float_range(need_arg(&i, argc, argv, arg), arg, 0.0f, 1.0f);
        } else if (!strcmp(arg, "--seed")) {
            c.gen.seed = parse_u64(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--think")) {
            c.gen.think_mode = PULSAR_THINK_HIGH;
        } else if (!strcmp(arg, "--think-max")) {
            c.gen.think_mode = PULSAR_THINK_MAX;
        } else if (!strcmp(arg, "--nothink")) {
            c.gen.think_mode = PULSAR_THINK_NONE;
        } else if (!strcmp(arg, "--backend")) {
            c.engine.backend = parse_backend(need_arg(&i, argc, argv, arg));
        } else if (!strcmp(arg, "--cuda")) {
            c.engine.backend = PULSAR_BACKEND_CUDA;
        } else if (!strcmp(arg, "-t") || !strcmp(arg, "--threads")) {
            c.engine.n_threads = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--chdir")) {
            c.chdir_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--quality")) {
            c.engine.quality = true;
        } else if (!strcmp(arg, "--prefill-chunk")) {
            int v = parse_int(need_arg(&i, argc, argv, arg), arg);
            if (v <= 0) {
                fprintf(stderr, "pulsar-agent: --prefill-chunk must be positive\n");
                exit(2);
            }
            c.engine.prefill_chunk = (uint32_t)v;
        } else if (!strcmp(arg, "--warm-weights")) {
            c.engine.warm_weights = true;
        } else if (!strcmp(arg, "--dir-steering-file")) {
            c.engine.directional_steering_file = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--dir-steering-ffn")) {
            c.engine.directional_steering_ffn = parse_float_range(need_arg(&i, argc, argv, arg), arg, -100.0f, 100.0f);
            steering_scale_set = true;
        } else if (!strcmp(arg, "--dir-steering-attn")) {
            c.engine.directional_steering_attn = parse_float_range(need_arg(&i, argc, argv, arg), arg, -100.0f, 100.0f);
            steering_scale_set = true;
        } else {
            fprintf(stderr, "pulsar-agent: unknown option: %s\n", arg);
            usage(stderr, NULL);
            exit(2);
        }
    }

    if (c.engine.directional_steering_file && !steering_scale_set)
        c.engine.directional_steering_ffn = 1.0f;
    return c;
}



void log_context_memory(pulsar_backend backend,
                               int         ctx_size,
                               uint32_t    prefill_chunk) {
    pulsar_context_memory m =
        pulsar_context_memory_estimate_with_prefill(backend,
                                                 ctx_size,
                                                 prefill_chunk);
    fprintf(stderr,
            "pulsar-agent: context buffers %.2f MiB (ctx=%d, backend=%s, prefill_chunk=%u, raw_kv_rows=%u, compressed_kv_rows=%u)\n",
            (double)m.total_bytes / (1024.0 * 1024.0),
            ctx_size,
            pulsar_backend_name(backend),
            m.prefill_cap,
            m.raw_cap,
            m.comp_cap);
}



pulsar_think_mode effective_think_mode(const agent_config *cfg) {
    return pulsar_think_mode_for_context(cfg->gen.think_mode, cfg->gen.ctx_size);
}

