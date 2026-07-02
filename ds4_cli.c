#include "ds4.h"
#include "ds4_distributed.h"
#include "ds4_help.h"
#include "ds4_internal.h"
#include "linenoise.h"

/* ds4 CLI.
 *
 * One-shot mode builds a single DeepSeek chat prompt and exits.  Interactive
 * mode keeps a rendered token transcript plus one ds4_session, so follow-up
 * turns reuse the live Metal KV checkpoint just like the server does.  The CLI
 * deliberately keeps policy here and leaves graph/cache mechanics inside the
 * engine API. */

#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <signal.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <time.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/wait.h>
#ifdef __linux__
#include <sys/prctl.h>
#endif

enum {
    CLI_LOCAL_GPU_MAX = 64,
    CLI_MODEL_PATH_MAX = 16,
    CLI_LOCAL_ASSIGNMENT_MAX = CLI_LOCAL_GPU_MAX + CLI_MODEL_PATH_MAX
};

typedef struct {
    const char *prompt;
    const char *system;
    int n_predict;
    int ctx_size;
    float temperature;
    float top_p;
    float min_p;
    uint64_t seed;
    bool dump_tokens;
    const char *dump_logits_path;
    const char *dump_logprobs_path;
    int dump_logprobs_top_k;
    const char *perplexity_file_path;
    const char *imatrix_dataset_path;
    const char *imatrix_output_path;
    int imatrix_max_prompts;
    int imatrix_max_tokens;
    ds4_think_mode think_mode;
    bool head_test;
    bool first_token_test;
    bool metal_graph_test;
    bool metal_graph_full_test;
    bool metal_graph_prompt_test;
} cli_generation_options;

typedef struct {
    const char *path;
    ds4_internal_model_shard_info info;
} cli_model_shard;

typedef struct {
    const char *model_path;
    int gpu_id;
    uint32_t start;
    uint32_t end;
    bool has_output;
    uint64_t cache_bytes;
} cli_local_gpu_assignment;

typedef struct {
    bool enabled;
    int ids[CLI_LOCAL_GPU_MAX];
    uint32_t n;
    uint32_t n_layers;
    cli_model_shard shards[CLI_MODEL_PATH_MAX];
    uint32_t n_shards;
    cli_local_gpu_assignment assignments[CLI_LOCAL_ASSIGNMENT_MAX];
    uint32_t assignment_count;
    int coordinator_port;
    int coordinator_listen_fd;
    pid_t worker_pids[CLI_LOCAL_ASSIGNMENT_MAX];
    uint32_t worker_count;
} cli_local_gpus;

typedef struct {
    ds4_engine_options engine;
    ds4_dist_options *dist;
    cli_generation_options gen;
    cli_local_gpus local_gpus;
    const char *model_paths[CLI_MODEL_PATH_MAX];
    uint32_t model_path_count;
    char *prompt_owned;
    bool inspect;
    bool prefill_chunk_set;
} cli_config;

static int cli_local_gpus_reap_exited(const cli_config *cfg);

static volatile sig_atomic_t cli_interrupted;
static volatile sig_atomic_t cli_dist_busy;
static volatile sig_atomic_t cli_dist_notice_printed;

static const char cli_dist_drain_msg[] =
    "\nds4: interrupt received; stopping distributed generation...\n";

static void cli_sigint_handler(int sig) {
    (void)sig;
    cli_interrupted = 1;
    if (cli_dist_busy && !cli_dist_notice_printed) {
        cli_dist_notice_printed = 1;
        ssize_t ignored = write(STDERR_FILENO,
                                cli_dist_drain_msg,
                                sizeof(cli_dist_drain_msg) - 1u);
        (void)ignored;
    }
}

static bool cli_interrupt_requested(void) {
    return cli_interrupted != 0;
}

static bool cli_session_cancel_cb(void *ud) {
    (void)ud;
    return cli_interrupt_requested();
}

static void cli_interrupt_clear(void) {
    cli_interrupted = 0;
    cli_dist_notice_printed = 0;
}

static void cli_local_gpus_close_listener(cli_local_gpus *g) {
    if (!g || g->coordinator_listen_fd < 0) return;
    close(g->coordinator_listen_fd);
    g->coordinator_listen_fd = -1;
}

static void cli_local_gpus_release_listener(const cli_config *cfg) {
    if (!cfg || !cfg->local_gpus.enabled) return;
    cli_local_gpus *g = (cli_local_gpus *)&cfg->local_gpus;
    cli_local_gpus_close_listener(g);
}

static bool cli_distributed_coordinator(const cli_config *cfg) {
    return cfg && cfg->engine.distributed.role == DS4_DISTRIBUTED_COORDINATOR;
}

static void cli_dist_busy_set(const cli_config *cfg, bool busy) {
    if (!cli_distributed_coordinator(cfg)) return;
    cli_dist_busy = busy ? 1 : 0;
    if (!busy) cli_dist_notice_printed = 0;
}

static int cli_wait_distributed_route(const cli_config *cfg, ds4_session *session) {
    if (!cli_distributed_coordinator(cfg)) return 0;
    cli_local_gpus_release_listener(cfg);

    char err[256] = {0};
    char last[256] = {0};
    unsigned ticks = 0;
    const struct timespec delay = {0, 250000000L};

    for (;;) {
        if (cli_interrupt_requested()) {
            fprintf(stderr, "ds4: interrupted while waiting for distributed route\n");
            return 1;
        }
        if (cli_local_gpus_reap_exited(cfg) != 0) {
            return 1;
        }
        int ready = ds4_session_distributed_route_ready(session, err, sizeof(err));
        if (ready > 0) {
            if (ticks) fprintf(stderr, "ds4: distributed route ready\n");
            return 0;
        }
        if (ready < 0) {
            fprintf(stderr,
                    "ds4: distributed route readiness failed: %s\n",
                    err[0] ? err : "unknown error");
            return 1;
        }

        const char *why = err[0] ? err : "route incomplete";
        if (strcmp(last, why) != 0 || (ticks % 20u) == 0) {
            fprintf(stderr, "ds4: waiting for distributed route: %s\n", why);
            snprintf(last, sizeof(last), "%s", why);
        }
        nanosleep(&delay, NULL);
        ticks++;
    }
}

static void usage(FILE *fp, const char *topic) {
    ds4_help_print(fp, DS4_HELP_DS4, topic);
}

static int parse_int(const char *s, const char *opt) {
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (s[0] == '\0' || *end != '\0' || v <= 0 || v > INT32_MAX) {
        fprintf(stderr, "ds4: invalid value for %s: %s\n", opt, s);
        exit(2);
    }
    return (int)v;
}

static uint64_t parse_u64(const char *s, const char *opt) {
    char *end = NULL;
    unsigned long long v = strtoull(s, &end, 10);
    if (s[0] == '\0' || *end != '\0' || v == 0) {
        fprintf(stderr, "ds4: invalid value for %s: %s\n", opt, s);
        exit(2);
    }
    return (uint64_t)v;
}

static float parse_float_range(const char *s, const char *opt, float min, float max) {
    char *end = NULL;
    float v = strtof(s, &end);
    if (s[0] == '\0' || *end != '\0' || !isfinite(v) || v < min || v > max) {
        fprintf(stderr, "ds4: invalid value for %s: %s\n", opt, s);
        exit(2);
    }
    return v;
}

static ds4_backend parse_backend(const char *s) {
    if (!strcmp(s, "metal")) return DS4_BACKEND_METAL;
#ifdef DS4_ROCM_BUILD
    if (!strcmp(s, "rocm")) return DS4_BACKEND_CUDA;
#else
    if (!strcmp(s, "cuda")) return DS4_BACKEND_CUDA;
#endif
    if (!strcmp(s, "cpu")) return DS4_BACKEND_CPU;
    fprintf(stderr, "ds4: invalid backend: %s\n", s);
#ifdef DS4_ROCM_BUILD
    fprintf(stderr, "ds4: valid backends are: metal, rocm, cpu\n");
#else
    fprintf(stderr, "ds4: valid backends are: metal, cuda, cpu\n");
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

static void log_context_memory(ds4_backend backend,
                               int         ctx_size,
                               uint32_t    prefill_chunk) {
    ds4_context_memory m =
        ds4_context_memory_estimate_with_prefill(backend,
                                                 ctx_size,
                                                 prefill_chunk);
    fprintf(stderr,
            "ds4: context buffers %.2f MiB (ctx=%d, backend=%s, prefill_chunk=%u, raw_kv_rows=%u, compressed_kv_rows=%u)\n",
            (double)m.total_bytes / (1024.0 * 1024.0),
            ctx_size,
            ds4_backend_name(backend),
            m.prefill_cap,
            m.raw_cap,
            m.comp_cap);
}

static ds4_think_mode cli_effective_think_mode(const cli_generation_options *gen) {
    return ds4_think_mode_for_context(gen->think_mode, gen->ctx_size);
}

static bool cli_think_max_downgraded(const cli_generation_options *gen) {
    return gen->think_mode == DS4_THINK_MAX &&
           cli_effective_think_mode(gen) != DS4_THINK_MAX;
}

static void cli_warn_think_max_downgraded(const cli_generation_options *gen, const char *name) {
    if (!cli_think_max_downgraded(gen)) return;
    ds4_log(stderr,
        DS4_LOG_WARNING,
        "ds4: warning: %s needs --ctx >= %u; ctx=%d uses normal thinking instead\n",
        name,
        ds4_think_max_min_context(),
        gen->ctx_size);
}

static double cli_now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1.0e-9;
}

static void cli_local_gpus_parse(cli_local_gpus *g, const char *spec) {
    if (!g || !spec || !spec[0]) {
        fprintf(stderr, "ds4: --gpus requires a comma-separated GPU list\n");
        exit(2);
    }

    char *copy = malloc(strlen(spec) + 1u);
    if (!copy) {
        fprintf(stderr, "ds4: out of memory parsing --gpus\n");
        exit(1);
    }
    strcpy(copy, spec);

    char *p = copy;
    while (*p) {
        while (*p == ',' || isspace((unsigned char)*p)) p++;
        if (!*p) break;
        char *start = p;
        while (*p && *p != ',') p++;
        char saved = *p;
        *p = '\0';

        char *end_trim = start + strlen(start);
        while (end_trim > start && isspace((unsigned char)end_trim[-1])) *--end_trim = '\0';
        errno = 0;
        char *end = NULL;
        long id = strtol(start, &end, 10);
        if (errno != 0 || start[0] == '\0' || *end != '\0' || id < 0 || id > INT32_MAX) {
            fprintf(stderr, "ds4: invalid GPU id in --gpus: %s\n", start);
            free(copy);
            exit(2);
        }
        if (g->n >= CLI_LOCAL_GPU_MAX) {
            fprintf(stderr, "ds4: --gpus supports at most %u entries\n", (unsigned)CLI_LOCAL_GPU_MAX);
            free(copy);
            exit(2);
        }
        for (uint32_t i = 0; i < g->n; i++) {
            if (g->ids[i] == (int)id) {
                fprintf(stderr, "ds4: duplicate GPU id in --gpus: %ld\n", id);
                free(copy);
                exit(2);
            }
        }
        g->ids[g->n++] = (int)id;
        if (!saved) break;
        p++;
    }
    free(copy);

    if (g->n == 0) {
        fprintf(stderr, "ds4: --gpus requires at least one GPU id\n");
        exit(2);
    }
    g->enabled = true;
}

static void cli_local_gpus_open_listener(cli_local_gpus *g) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        fprintf(stderr, "ds4: failed to create local launcher socket: %s\n", strerror(errno));
        exit(1);
    }

    int one = 1;
    (void)setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = 0;
    if (bind(fd, (const struct sockaddr *)&addr, sizeof(addr)) != 0) {
        fprintf(stderr, "ds4: failed to bind local distributed listener: %s\n", strerror(errno));
        close(fd);
        exit(1);
    }
    if (listen(fd, 64) != 0) {
        fprintf(stderr, "ds4: failed to listen on local distributed socket: %s\n", strerror(errno));
        close(fd);
        exit(1);
    }

    socklen_t len = sizeof(addr);
    if (getsockname(fd, (struct sockaddr *)&addr, &len) != 0) {
        fprintf(stderr, "ds4: failed to read local distributed port: %s\n", strerror(errno));
        close(fd);
        exit(1);
    }
    int port = (int)ntohs(addr.sin_port);
    if (port <= 0) {
        fprintf(stderr, "ds4: failed to allocate a local distributed port\n");
        close(fd);
        exit(1);
    }
    g->coordinator_port = port;
    g->coordinator_listen_fd = fd;
}

static uint32_t cli_model_shard_layer_count(const cli_model_shard *s) {
    if (!s || !s->info.has_layers || s->info.last_layer < s->info.first_layer) return 0;
    return s->info.last_layer - s->info.first_layer + 1u;
}

static uint64_t cli_gib_bytes(void) {
    return 1073741824ull;
}

static bool cli_read_u64_file(const char *path, uint64_t *out) {
    if (out) *out = 0;
    if (!path || !out) return false;
    FILE *fp = fopen(path, "r");
    if (!fp) return false;
    char buf[64];
    if (!fgets(buf, sizeof(buf), fp)) {
        fclose(fp);
        return false;
    }
    fclose(fp);

    errno = 0;
    char *end = NULL;
    unsigned long long v = strtoull(buf, &end, 10);
    if (errno != 0 || end == buf || v == 0) return false;
    *out = (uint64_t)v;
    return true;
}

static uint64_t cli_local_gpus_min_vram_total_bytes(void) {
#ifdef __linux__
    DIR *dir = opendir("/sys/class/drm");
    if (!dir) return 0;
    uint64_t min_total = UINT64_MAX;
    struct dirent *de = NULL;
    while ((de = readdir(dir)) != NULL) {
        if (strncmp(de->d_name, "card", 4) != 0) continue;
        char path[PATH_MAX];
        int n = snprintf(path,
                         sizeof(path),
                         "/sys/class/drm/%s/device/mem_info_vram_total",
                         de->d_name);
        if (n < 0 || (size_t)n >= sizeof(path)) continue;
        uint64_t total = 0;
        if (!cli_read_u64_file(path, &total) || total == 0) continue;
        if (total < min_total) min_total = total;
    }
    closedir(dir);
    return min_total == UINT64_MAX ? 0 : min_total;
#else
    return 0;
#endif
}

static uint64_t cli_model_shard_range_cache_bytes(
        const cli_model_shard *s,
        uint32_t start,
        uint32_t end) {
    if (!s || !s->info.has_layers || end < start) return 0;
    if (start < s->info.first_layer || end > s->info.last_layer) return 0;
    if (start < DS4_INTERNAL_MAX_LAYER && end < DS4_INTERNAL_MAX_LAYER) {
        const uint64_t exact = s->info.range_cache_bytes[start][end];
        if (exact != 0) return exact;
    }

    const uint32_t shard_layers = cli_model_shard_layer_count(s);
    const uint32_t range_layers = end - start + 1u;
    if (s->info.total_cache_bytes != 0 && shard_layers != 0) {
        return (s->info.total_cache_bytes * range_layers + shard_layers - 1u) /
               shard_layers;
    }
    return (uint64_t)range_layers;
}

static uint64_t cli_model_range_cache_bytes(
        const cli_local_gpus *g,
        uint32_t start,
        uint32_t end) {
    if (!g || end < start) return 0;
    uint64_t total = 0;
    for (uint32_t si = 0; si < g->n_shards; si++) {
        const cli_model_shard *s = &g->shards[si];
        if (end < s->info.first_layer || start > s->info.last_layer) continue;
        const uint32_t lo = start > s->info.first_layer ? start : s->info.first_layer;
        const uint32_t hi = end < s->info.last_layer ? end : s->info.last_layer;
        const uint64_t bytes = cli_model_shard_range_cache_bytes(s, lo, hi);
        if (total > UINT64_MAX - bytes) return UINT64_MAX;
        total += bytes;
    }
    return total;
}

static uint64_t cli_model_total_cache_bytes(const cli_local_gpus *g) {
    if (!g || g->n_layers == 0) return 0;
    return cli_model_range_cache_bytes(g, 0, g->n_layers - 1u);
}

static void cli_local_gpus_model_budgets(
        const cli_config *cfg,
        uint64_t budgets[CLI_LOCAL_GPU_MAX]) {
    for (uint32_t i = 0; i < CLI_LOCAL_GPU_MAX; i++) budgets[i] = 0;
    if (!cfg || !cfg->local_gpus.enabled) return;

    const uint64_t total = cli_local_gpus_min_vram_total_bytes();
    if (total == 0) return;

    const uint64_t model_reserve = 2ull * cli_gib_bytes();
    uint64_t base = total > model_reserve ? total - model_reserve : 0;
    for (uint32_t i = 0; i < cfg->local_gpus.n; i++) budgets[i] = base;

    ds4_context_memory ctx =
        ds4_context_memory_estimate_with_prefill(cfg->engine.backend,
                                                 cfg->gen.ctx_size,
                                                 cfg->engine.prefill_chunk);
    budgets[0] = budgets[0] > ctx.total_bytes ? budgets[0] - ctx.total_bytes : 0;
    for (uint32_t i = 0; i < cfg->local_gpus.n; i++) {
        if (budgets[i] == 0) budgets[i] = 1;
    }
}

static int cli_model_shard_cmp(const void *a, const void *b) {
    const cli_model_shard *sa = (const cli_model_shard *)a;
    const cli_model_shard *sb = (const cli_model_shard *)b;
    const uint32_t la = sa->info.has_layers ? sa->info.first_layer : UINT32_MAX;
    const uint32_t lb = sb->info.has_layers ? sb->info.first_layer : UINT32_MAX;
    if (la < lb) return -1;
    if (la > lb) return 1;
    if (sa->info.has_token_embedding != sb->info.has_token_embedding) {
        return sa->info.has_token_embedding ? -1 : 1;
    }
    if (sa->info.has_output_head != sb->info.has_output_head) {
        return sa->info.has_output_head ? 1 : -1;
    }
    return 0;
}

static void cli_add_model_path(cli_config *cfg, const char *path) {
    if (!cfg || !path || !path[0]) {
        fprintf(stderr, "ds4: -m/--model requires a non-empty path\n");
        exit(2);
    }
    if (cfg->model_path_count >= CLI_MODEL_PATH_MAX) {
        fprintf(stderr,
                "ds4: at most %u model shards may be passed with repeated -m\n",
                (unsigned)CLI_MODEL_PATH_MAX);
        exit(2);
    }
    cfg->model_paths[cfg->model_path_count++] = path;
    cfg->engine.model_path = path;
}

static void cli_local_gpus_set_visible(int gpu_id) {
    char id[32];
    snprintf(id, sizeof(id), "%d", gpu_id);
#ifdef DS4_ROCM_BUILD
    setenv("ROCR_VISIBLE_DEVICES", id, 1);
    setenv("HIP_VISIBLE_DEVICES", "0", 1);
    setenv("CUDA_VISIBLE_DEVICES", "0", 1);
#else
    setenv("CUDA_VISIBLE_DEVICES", id, 1);
#endif
}

static void cli_local_gpus_load_shards(cli_config *cfg) {
    cli_local_gpus *g = &cfg->local_gpus;
    g->n_shards = cfg->model_path_count;
    if (g->n_shards == 0 || g->n_shards > CLI_MODEL_PATH_MAX) {
        fprintf(stderr, "ds4: invalid local GPU model shard count\n");
        exit(2);
    }

    uint32_t expected_start = 0;
    uint32_t n_layers = 0;
    for (uint32_t i = 0; i < g->n_shards; i++) {
        cli_model_shard *s = &g->shards[i];
        s->path = cfg->model_paths[i];
        if (ds4_internal_model_shard_info_from_file(s->path, &s->info) != 0) {
            fprintf(stderr, "ds4: failed to inspect model shard: %s\n", s->path);
            exit(2);
        }
        if (i == 0) {
            n_layers = s->info.n_layers;
        } else if (s->info.n_layers != n_layers) {
            fprintf(stderr, "ds4: model shards disagree on layer count\n");
            exit(2);
        }
        if (!s->info.has_layers) {
            fprintf(stderr, "ds4: model shard contains no transformer layers: %s\n", s->path);
            exit(2);
        }
    }

    qsort(g->shards, g->n_shards, sizeof(g->shards[0]), cli_model_shard_cmp);
    for (uint32_t i = 0; i < g->n_shards; i++) {
        cli_model_shard *s = &g->shards[i];
        if (i == 0 && !s->info.has_token_embedding) {
            fprintf(stderr,
                    "ds4: model shard containing layer 0 must contain token_embd.weight: %s\n",
                    s->path);
            exit(2);
        }
        if (s->info.first_layer != expected_start) {
            fprintf(stderr,
                    "ds4: model shards must be contiguous; expected layer %u but %s starts at %u\n",
                    expected_start,
                    s->path,
                    s->info.first_layer);
            exit(2);
        }
        if (i + 1u < g->n_shards && s->info.has_output_head) {
            fprintf(stderr, "ds4: only the final model shard may contain the output head: %s\n", s->path);
            exit(2);
        }
        expected_start = s->info.last_layer + 1u;
    }

    if (expected_start != n_layers) {
        fprintf(stderr,
                "ds4: model shards cover %u layers but model metadata requires %u\n",
                expected_start,
                n_layers);
        exit(2);
    }
    if (!g->shards[g->n_shards - 1u].info.has_output_head) {
        fprintf(stderr, "ds4: final model shard must contain the output head\n");
        exit(2);
    }
    if (g->n > n_layers) {
        fprintf(stderr, "ds4: --gpus has %u entries but the model has only %u layers\n",
                g->n, n_layers);
        exit(2);
    }
    g->n_layers = n_layers;
}

static void cli_local_gpus_add_assignment(
        cli_local_gpus *g,
        const cli_model_shard *s,
        int gpu_id,
        uint32_t start,
        uint32_t end) {
    if (g->assignment_count >= CLI_LOCAL_ASSIGNMENT_MAX) {
        fprintf(stderr,
                "ds4: local GPU split produced too many route entries (%u max)\n",
                (unsigned)CLI_LOCAL_ASSIGNMENT_MAX);
        exit(2);
    }
    cli_local_gpu_assignment *a = &g->assignments[g->assignment_count++];
    memset(a, 0, sizeof(*a));
    a->model_path = s->path;
    a->gpu_id = gpu_id;
    a->start = start;
    a->end = end;
    a->has_output = s->info.has_output_head && end == s->info.last_layer;
    a->cache_bytes = cli_model_shard_range_cache_bytes(s, start, end);
}

static bool cli_local_gpus_plan_ranges(
        const cli_local_gpus *g,
        const uint64_t budgets[CLI_LOCAL_GPU_MAX],
        uint32_t starts[CLI_LOCAL_GPU_MAX],
        uint32_t ends[CLI_LOCAL_GPU_MAX]) {
    const uint32_t n = g->n;
    const uint32_t layers = g->n_layers;
    double dp[CLI_LOCAL_GPU_MAX + 1u][DS4_INTERNAL_MAX_LAYER + 1u];
    int split[CLI_LOCAL_GPU_MAX + 1u][DS4_INTERNAL_MAX_LAYER + 1u];

    for (uint32_t p = 0; p <= CLI_LOCAL_GPU_MAX; p++) {
        for (uint32_t e = 0; e <= DS4_INTERNAL_MAX_LAYER; e++) {
            dp[p][e] = 1.0e300;
            split[p][e] = -1;
        }
    }
    dp[0][0] = 0.0;

    for (uint32_t p = 1; p <= n; p++) {
        for (uint32_t e = p; e <= layers; e++) {
            for (uint32_t k = p - 1u; k < e; k++) {
                if (dp[p - 1u][k] >= 1.0e299) continue;
                const uint64_t cost = cli_model_range_cache_bytes(g, k, e - 1u);
                if (budgets[p - 1u] != 0 && cost > budgets[p - 1u]) continue;
                const double load = budgets[p - 1u] != 0 ?
                    (double)cost / (double)budgets[p - 1u] :
                    (double)cost;
                const double candidate = dp[p - 1u][k] > load ? dp[p - 1u][k] : load;
                if (candidate < dp[p][e]) {
                    dp[p][e] = candidate;
                    split[p][e] = (int)k;
                }
            }
        }
    }

    if (split[n][layers] < 0) return false;

    uint32_t e = layers;
    for (uint32_t p = n; p > 0; p--) {
        const int k = split[p][e];
        if (k < 0) return false;
        starts[p - 1u] = (uint32_t)k;
        ends[p - 1u] = e - 1u;
        e = (uint32_t)k;
    }
    return e == 0;
}

static void cli_local_gpus_build_assignments(cli_config *cfg) {
    cli_local_gpus *g = &cfg->local_gpus;
    uint64_t budgets[CLI_LOCAL_GPU_MAX];
    uint32_t starts[CLI_LOCAL_GPU_MAX];
    uint32_t ends[CLI_LOCAL_GPU_MAX];

    cli_local_gpus_model_budgets(cfg, budgets);
    if (!cli_local_gpus_plan_ranges(g, budgets, starts, ends)) {
        const uint64_t total = cli_model_total_cache_bytes(g);
        fprintf(stderr,
                "ds4: --gpus could not fit a byte-balanced local split "
                "within estimated per-GPU model-cache budgets\n");
        fprintf(stderr,
                "ds4: model cache %.2f GiB, coordinator budget %.2f GiB, worker budget %.2f GiB\n",
                (double)total / (double)cli_gib_bytes(),
                (double)budgets[0] / (double)cli_gib_bytes(),
                (g->n > 1 ? (double)budgets[1] : (double)budgets[0]) /
                    (double)cli_gib_bytes());
        exit(2);
    }

    g->assignment_count = 0;
    for (uint32_t pi = 0; pi < g->n; pi++) {
        for (uint32_t si = 0; si < g->n_shards; si++) {
            const cli_model_shard *s = &g->shards[si];
            if (ends[pi] < s->info.first_layer || starts[pi] > s->info.last_layer) continue;
            const uint32_t lo = starts[pi] > s->info.first_layer ? starts[pi] : s->info.first_layer;
            const uint32_t hi = ends[pi] < s->info.last_layer ? ends[pi] : s->info.last_layer;
            cli_local_gpus_add_assignment(g, s, g->ids[pi], lo, hi);
        }
    }
}

static bool cli_local_gpus_tight_model_split(const cli_local_gpus *g) {
    if (!g) return false;
    const uint64_t tight = 160ull * 1073741824ull;
    for (uint32_t i = 0; i < g->assignment_count; i++) {
        if (g->assignments[i].cache_bytes >= tight) return true;
    }
    return false;
}

static void cli_local_gpus_configure(cli_config *cfg) {
    if (!cfg || !cfg->local_gpus.enabled) return;
    cli_local_gpus *g = &cfg->local_gpus;

    if (cfg->engine.backend != DS4_BACKEND_CUDA) {
#ifdef DS4_ROCM_BUILD
        fprintf(stderr, "ds4: --gpus requires the ROCm backend in this build\n");
#else
        fprintf(stderr, "ds4: --gpus requires the CUDA backend in this build\n");
#endif
        exit(2);
    }
    if (cfg->engine.mtp_path) {
        fprintf(stderr, "ds4: --gpus is not compatible with --mtp yet\n");
        exit(2);
    }
    if (cfg->inspect || cfg->gen.dump_tokens || cfg->gen.head_test ||
        cfg->gen.first_token_test || cfg->gen.metal_graph_test ||
        cfg->gen.metal_graph_full_test || cfg->gen.metal_graph_prompt_test ||
        cfg->gen.perplexity_file_path || cfg->gen.imatrix_dataset_path ||
        cfg->gen.imatrix_output_path) {
        fprintf(stderr, "ds4: --gpus is for normal generation/server-style distributed runs, not diagnostics\n");
        exit(2);
    }
    if (cfg->dist->role != DS4_DISTRIBUTED_NONE ||
        cfg->dist->layers.set ||
        cfg->dist->listen_host ||
        cfg->dist->listen_port ||
        cfg->dist->coordinator_host ||
        cfg->dist->coordinator_port) {
        fprintf(stderr, "ds4: --gpus cannot be combined with explicit --role/--layers/--listen/--coordinator\n");
        exit(2);
    }

    cli_local_gpus_load_shards(cfg);
    cli_local_gpus_build_assignments(cfg);
    if (!cfg->prefill_chunk_set && cli_local_gpus_tight_model_split(g)) {
        cfg->engine.prefill_chunk = 1024;
        fprintf(stderr,
                "ds4: local GPU launcher defaulting --prefill-chunk 1024 "
                "for a tight VRAM split; pass --prefill-chunk to override\n");
        cli_local_gpus_build_assignments(cfg);
    }
    if (g->assignment_count == 0) {
        fprintf(stderr, "ds4: local GPU split produced no route entries\n");
        exit(2);
    }
    if (g->assignment_count == 1) {
        cfg->engine.model_path = g->assignments[0].model_path;
        return;
    }
    cli_local_gpus_open_listener(g);

    const cli_local_gpu_assignment *a = &g->assignments[0];
    cfg->engine.model_path = a->model_path;
    cfg->dist->role = DS4_DISTRIBUTED_COORDINATOR;
    cfg->dist->layers.start = a->start;
    cfg->dist->layers.end = a->end;
    cfg->dist->layers.has_output = a->has_output;
    cfg->dist->layers.set = true;
    cfg->dist->listen_host = "127.0.0.1";
    cfg->dist->listen_port = g->coordinator_port;
    ds4_internal_dist_set_adopt_listen_fd(cfg->dist, g->coordinator_listen_fd);
    if (!getenv("DS4_DIST_ROUTE_WAIT_SEC")) setenv("DS4_DIST_ROUTE_WAIT_SEC", "300", 0);
}

static int cli_local_gpu_worker_run(cli_config cfg, uint32_t idx) {
    cli_local_gpus *g = &cfg.local_gpus;
    const cli_local_gpu_assignment *a = &g->assignments[idx];

    cli_local_gpus_close_listener(g);
    ds4_internal_dist_set_adopt_listen_fd(cfg.dist, 0);

    struct sigaction ignore_int;
    memset(&ignore_int, 0, sizeof(ignore_int));
    ignore_int.sa_handler = SIG_IGN;
    sigemptyset(&ignore_int.sa_mask);
    (void)sigaction(SIGINT, &ignore_int, NULL);

#ifdef __linux__
    (void)prctl(PR_SET_PDEATHSIG, SIGTERM);
    if (getppid() == 1) return 1;
#endif

    cli_local_gpus_set_visible(a->gpu_id);
    if (!getenv("DS4_DIST_ENABLE_WORKER_PREFETCH") &&
        !getenv("DS4_DIST_DISABLE_WORKER_PREFETCH")) {
        setenv("DS4_DIST_DISABLE_WORKER_PREFETCH", "1", 0);
    }
    if (!getenv("DS4_DIST_ENABLE_WORKER_FORWARD_PIPELINE") &&
        !getenv("DS4_DIST_WORKER_FORWARD_SYNC")) {
        setenv("DS4_DIST_WORKER_FORWARD_SYNC", "1", 0);
    }
    cfg.engine.model_path = a->model_path;
    cfg.dist->role = DS4_DISTRIBUTED_WORKER;
    cfg.dist->layers.start = a->start;
    cfg.dist->layers.end = a->end;
    cfg.dist->layers.has_output = a->has_output;
    cfg.dist->layers.set = true;
    cfg.dist->listen_host = NULL;
    cfg.dist->listen_port = 0;
    cfg.dist->coordinator_host = "127.0.0.1";
    cfg.dist->coordinator_port = g->coordinator_port;
    cfg.dist->prefill_chunk = 0;
    cfg.dist->prefill_window = 0;
    cfg.dist->activation_bits = 0;
    cfg.dist->replay_check = false;
    ds4_internal_dist_set_local_worker(cfg.dist, true);

    char dist_err[256];
    if (ds4_dist_prepare_engine_options(cfg.dist, &cfg.engine, dist_err, sizeof(dist_err)) != 0) {
        fprintf(stderr, "ds4: local GPU worker %u: %s\n", idx, dist_err);
        return 2;
    }

    char end_buf[32];
    if (a->has_output) snprintf(end_buf, sizeof(end_buf), "output");
    else snprintf(end_buf, sizeof(end_buf), "%u", a->end);
    fprintf(stderr,
            "ds4: local GPU worker %u pid=%ld gpu=%d model=%s layers %u:%s connecting to 127.0.0.1:%d\n",
            idx,
            (long)getpid(),
            a->gpu_id,
            a->model_path,
            a->start,
            end_buf,
            g->coordinator_port);

    ds4_engine *engine = NULL;
    if (ds4_engine_open(&engine, &cfg.engine) != 0) return 1;
    ds4_dist_generation_options dist_gen = {
        .prompt = cfg.gen.prompt,
        .system = cfg.gen.system,
        .dump_logits_path = cfg.gen.dump_logits_path,
        .dump_logprobs_path = cfg.gen.dump_logprobs_path,
        .dump_logprobs_top_k = cfg.gen.dump_logprobs_top_k,
        .n_predict = cfg.gen.n_predict,
        .ctx_size = cfg.gen.ctx_size,
        .temperature = cfg.gen.temperature,
        .top_p = cfg.gen.top_p,
        .min_p = cfg.gen.min_p,
        .seed = cfg.gen.seed,
        .think_mode = cfg.gen.think_mode,
    };
    int rc = ds4_dist_run(engine, cfg.dist, &dist_gen);
    ds4_engine_close(engine);
    return rc;
}

static void cli_local_gpus_stop(cli_local_gpus *g) {
    if (!g) return;
    cli_local_gpus_close_listener(g);
    if (g->worker_count == 0) return;
    for (uint32_t i = 0; i < g->worker_count; i++) {
        if (g->worker_pids[i] > 0) kill(g->worker_pids[i], SIGTERM);
    }

    bool left = true;
    for (int spin = 0; spin < 50 && left; spin++) {
        left = false;
        for (uint32_t i = 0; i < g->worker_count; i++) {
            pid_t pid = g->worker_pids[i];
            if (pid <= 0) continue;
            int status = 0;
            pid_t got = waitpid(pid, &status, WNOHANG);
            if (got == pid) g->worker_pids[i] = 0;
            else if (got == 0) left = true;
        }
        if (left) {
            const struct timespec delay = {0, 100000000L};
            nanosleep(&delay, NULL);
        }
    }

    for (uint32_t i = 0; i < g->worker_count; i++) {
        pid_t pid = g->worker_pids[i];
        if (pid <= 0) continue;
        kill(pid, SIGKILL);
        (void)waitpid(pid, NULL, 0);
        g->worker_pids[i] = 0;
    }
    g->worker_count = 0;
}

static void cli_local_gpus_status_text(int status, char *buf, size_t len) {
    if (!buf || len == 0) return;
    if (WIFEXITED(status)) {
        snprintf(buf, len, "exit status %d", WEXITSTATUS(status));
    } else if (WIFSIGNALED(status)) {
        snprintf(buf, len, "signal %d", WTERMSIG(status));
    } else {
        snprintf(buf, len, "status %d", status);
    }
}

static int cli_local_gpus_reap_exited(const cli_config *cfg) {
    if (!cfg || !cfg->local_gpus.enabled || cfg->local_gpus.worker_count == 0) return 0;
    cli_local_gpus *g = (cli_local_gpus *)&cfg->local_gpus;
    for (uint32_t i = 0; i < g->worker_count; i++) {
        const pid_t pid = g->worker_pids[i];
        if (pid <= 0) continue;
        int status = 0;
        const pid_t got = waitpid(pid, &status, WNOHANG);
        if (got == pid) {
            char detail[64];
            cli_local_gpus_status_text(status, detail, sizeof(detail));
            fprintf(stderr,
                    "ds4: local GPU worker %u pid=%ld exited unexpectedly during startup (%s)\n",
                    i + 1u,
                    (long)pid,
                    detail);
            g->worker_pids[i] = 0;
            return 1;
        }
        if (got < 0 && errno == ECHILD) {
            fprintf(stderr,
                    "ds4: local GPU worker %u pid=%ld is no longer available\n",
                    i + 1u,
                    (long)pid);
            g->worker_pids[i] = 0;
            return 1;
        }
    }
    return 0;
}

static int cli_local_gpus_start(cli_config *cfg) {
    if (!cfg || !cfg->local_gpus.enabled) return 0;
    cli_local_gpus *g = &cfg->local_gpus;
    if (g->assignment_count == 0) return 0;
    cli_local_gpus_set_visible(g->assignments[0].gpu_id);
    if (g->assignment_count == 1) {
        fprintf(stderr, "ds4: local GPU launcher using gpu=%d\n", g->assignments[0].gpu_id);
        return 0;
    }

    const cli_local_gpu_assignment *a = &g->assignments[0];
    char end_buf[32];
    if (a->has_output) snprintf(end_buf, sizeof(end_buf), "output");
    else snprintf(end_buf, sizeof(end_buf), "%u", a->end);
    fprintf(stderr,
            "ds4: local GPU launcher coordinator gpu=%d model=%s layers %u:%s listen=127.0.0.1:%d workers=%u\n",
            a->gpu_id,
            a->model_path,
            a->start,
            end_buf,
            g->coordinator_port,
            g->assignment_count - 1u);

    fflush(NULL);
    for (uint32_t i = 1; i < g->assignment_count; i++) {
        pid_t pid = fork();
        if (pid < 0) {
            fprintf(stderr, "ds4: failed to fork local GPU worker %u: %s\n", i, strerror(errno));
            cli_local_gpus_stop(g);
            return 1;
        }
        if (pid == 0) {
            int rc = cli_local_gpu_worker_run(*cfg, i);
            exit(rc);
        }
        g->worker_pids[g->worker_count++] = pid;
    }
    return 0;
}

static char *read_prompt_file(const char *path, bool fatal);

typedef struct {
    int base_tokens;
    int input_tokens;
    bool use_color;
    bool finished;
} cli_prefill_progress;

static void cli_prefill_progress_cb(void *ud, const char *event, int current, int total) {
    (void)total;
    cli_prefill_progress *p = ud;
    if (!p || !event || p->input_tokens <= 0) return;
    const bool is_display = strcmp(event, "prefill_display") == 0;
    if (strcmp(event, "prefill_chunk") && !is_display) return;
    if (is_display && !p->use_color) return;

    int processed = current - p->base_tokens;
    if (processed < 0) processed = 0;
    if (processed > p->input_tokens) processed = p->input_tokens;
    double pct = 100.0 * (double)processed / (double)p->input_tokens;
    if (pct > 100.0) pct = 100.0;

    const bool complete = processed >= p->input_tokens;
    if (complete && p->finished) return;

    if (p->use_color) {
        fputc('\r', stderr);
        ds4_log(stderr,
                DS4_LOG_PREFILL,
                "processing %d input tokens: %d/%d (%.1f%%)",
                p->input_tokens,
                processed,
                p->input_tokens,
                pct);
        fputs("\x1b[K", stderr);
        if (complete) fputc('\n', stderr);
    } else {
        fprintf(stderr,
                "processing %d input tokens: %d/%d (%.1f%%)\n",
                p->input_tokens,
                processed,
                p->input_tokens,
                pct);
    }
    if (complete) p->finished = true;
    fflush(stderr);
}

static bool is_rendered_chat_prompt(const char *prompt) {
    const char *bos = "<｜begin▁of▁sentence｜>";
    return prompt && strncmp(prompt, bos, strlen(bos)) == 0;
}

typedef struct {
    ds4_engine *engine;
    FILE *fp;
    bool format_thinking;
    bool in_think;
    bool color_open;
    bool use_color;
    bool last_output_newline;
    char pending[16];
    size_t pending_len;
} token_printer;

static bool bytes_has_prefix(const char *p, size_t n, const char *prefix) {
    size_t plen = strlen(prefix);
    return n >= plen && memcmp(p, prefix, plen) == 0;
}

static bool bytes_is_partial_prefix(const char *p, size_t n, const char *prefix) {
    size_t plen = strlen(prefix);
    return n < plen && memcmp(prefix, p, n) == 0;
}

static void token_printer_set_grey(token_printer *p) {
    if (p->use_color && !p->color_open) {
        fputs("\x1b[90m", p->fp);
        p->color_open = true;
    }
}

static void token_printer_reset_color(token_printer *p) {
    if (p->use_color && p->color_open) {
        fputs("\x1b[0m", p->fp);
        p->color_open = false;
    }
}

static void token_printer_write_char(token_printer *p, char c) {
    if (p->in_think) token_printer_set_grey(p);
    fputc((unsigned char)c, p->fp);
    p->last_output_newline = c == '\n';
}

static void token_printer_process(token_printer *p, const char *text, size_t len, bool finish) {
    const char *think_open = "<think>";
    const char *think_close = "</think>";
    size_t total = p->pending_len + len;
    char *buf = malloc(total ? total : 1);
    if (!buf) return;
    if (p->pending_len) memcpy(buf, p->pending, p->pending_len);
    if (len) memcpy(buf + p->pending_len, text, len);
    p->pending_len = 0;

    size_t i = 0;
    while (i < total) {
        const char *cur = buf + i;
        const size_t rem = total - i;
        if (bytes_has_prefix(cur, rem, think_open)) {
            p->in_think = true;
            i += strlen(think_open);
            continue;
        }
        if (bytes_has_prefix(cur, rem, think_close)) {
            p->in_think = false;
            token_printer_reset_color(p);
            if (!p->last_output_newline) {
                fputc('\n', p->fp);
                p->last_output_newline = true;
            }
            i += strlen(think_close);
            continue;
        }
        if (!finish && cur[0] == '<' &&
            (bytes_is_partial_prefix(cur, rem, think_open) ||
             bytes_is_partial_prefix(cur, rem, think_close)))
        {
            if (rem < sizeof(p->pending)) {
                memcpy(p->pending, cur, rem);
                p->pending_len = rem;
            }
            break;
        }
        token_printer_write_char(p, cur[0]);
        i++;
    }

    free(buf);
}

static void token_printer_finish(token_printer *p) {
    if (p->format_thinking) {
        token_printer_process(p, NULL, 0, true);
        token_printer_reset_color(p);
    }
    fflush(p->fp);
}

static void generation_done(void *ud) {
    token_printer *p = ud;
    token_printer_finish(p);
    if (!p->last_output_newline) {
        fputc('\n', p->fp);
        p->last_output_newline = true;
    }
    fflush(p->fp);
}

static void token_printer_write_text(token_printer *p, const char *text, size_t len) {
    if (p->format_thinking) {
        token_printer_process(p, text, len, false);
    } else if (len) {
        fwrite(text, 1, len, p->fp);
        p->last_output_newline = text[len - 1] == '\n';
    }
}

static void print_generated_token(void *ud, int token) {
    token_printer *p = ud;
    size_t len = 0;
    char *text = ds4_token_text(p->engine, token, &len);
    token_printer_write_text(p, text, len);
    fflush(p->fp);
    free(text);
}

static void build_prompt(ds4_engine *engine, const cli_generation_options *gen, ds4_tokens *out) {
    if (is_rendered_chat_prompt(gen->prompt)) {
        ds4_tokenize_rendered_chat(engine, gen->prompt, out);
    } else {
        ds4_encode_chat_prompt(engine, gen->system, gen->prompt,
                               cli_effective_think_mode(gen), out);
    }
}

static int run_sampled_generation(ds4_engine *engine, const cli_config *cfg, const ds4_tokens *prompt) {
    ds4_session *session = NULL;
    if (ds4_session_create(&session, engine, cfg->gen.ctx_size) != 0) {
        fprintf(stderr, "ds4: sampled CLI generation requires a session backend\n");
        return 1;
    }
    if (cli_wait_distributed_route(cfg, session) != 0) {
        ds4_session_free(session);
        return 1;
    }

    char err[160];
    ds4_think_mode think_mode = cli_effective_think_mode(&cfg->gen);
    token_printer printer = {
        .engine = engine,
        .fp = stdout,
        .format_thinking = ds4_think_mode_enabled(think_mode),
        .in_think = ds4_think_mode_enabled(think_mode),
        .use_color = isatty(fileno(stdout)) != 0,
        .last_output_newline = true,
    };
    cli_prefill_progress progress = {
        .base_tokens = 0,
        .input_tokens = prompt->len,
        .use_color = ds4_log_is_tty(stderr),
    };

    const double t_prefill0 = cli_now_sec();
    ds4_session_set_progress(session, cli_prefill_progress_cb, &progress);
    ds4_session_set_display_progress(session,
                                     progress.use_color ? cli_prefill_progress_cb : NULL,
                                     progress.use_color ? &progress : NULL);
    cli_dist_busy_set(cfg, true);
    int sync_rc = ds4_session_sync(session, prompt, err, sizeof(err));
    cli_dist_busy_set(cfg, false);
    if (sync_rc != 0) {
        ds4_session_set_progress(session, NULL, NULL);
        ds4_session_set_display_progress(session, NULL, NULL);
        fprintf(stderr, "ds4: prompt processing failed: %s\n", err);
        ds4_session_free(session);
        return 1;
    }
    ds4_session_set_progress(session, NULL, NULL);
    ds4_session_set_display_progress(session, NULL, NULL);
    const double t_prefill1 = cli_now_sec();

    int max_tokens = cfg->gen.n_predict;
    int room = ds4_session_ctx(session) - ds4_session_pos(session);
    if (room <= 1) max_tokens = 0;
    else if (max_tokens > room - 1) max_tokens = room - 1;

    uint64_t rng = cfg->gen.seed ? cfg->gen.seed :
        ((uint64_t)time(NULL) ^ ((uint64_t)getpid() << 32) ^ (uint64_t)clock());
    int generated = 0;
    const double t_decode0 = cli_now_sec();
    while (generated < max_tokens && !cli_interrupt_requested()) {
        int token = ds4_session_sample(session, cfg->gen.temperature, 0,
                                       cfg->gen.top_p, cfg->gen.min_p, &rng);
        if (token == ds4_token_eos(engine)) break;

        int toks[17];
        int ntok = 0;
        if (cfg->gen.temperature <= 0.0f && ds4_engine_mtp_draft_tokens(engine) > 1 &&
            getenv("DS4_MTP_SPEC_DISABLE") == NULL) {
            cli_dist_busy_set(cfg, true);
            ntok = ds4_session_eval_speculative_argmax(session,
                                                       token,
                                                       max_tokens - generated,
                                                       ds4_token_eos(engine),
                                                       toks,
                                                       (int)(sizeof(toks) / sizeof(toks[0])),
                                                       err,
                                                       sizeof(err));
            cli_dist_busy_set(cfg, false);
            if (ntok < 0) {
                fprintf(stderr, "ds4: decode failed: %s\n", err);
                ds4_session_free(session);
                return 1;
            }
        } else {
            cli_dist_busy_set(cfg, true);
            int eval_rc = ds4_session_eval(session, token, err, sizeof(err));
            cli_dist_busy_set(cfg, false);
            if (eval_rc != 0) {
                fprintf(stderr, "ds4: decode failed: %s\n", err);
                ds4_session_free(session);
                return 1;
            }
            toks[0] = token;
            ntok = 1;
        }

        bool stop = false;
        for (int j = 0; j < ntok; j++) {
            if (toks[j] == ds4_token_eos(engine)) {
                stop = true;
                break;
            }
            size_t piece_len = 0;
            char *piece = ds4_token_text(engine, toks[j], &piece_len);
            token_printer_write_text(&printer, piece, piece_len);
            fflush(stdout);
            free(piece);
            generated++;
            if (generated >= max_tokens) break;
        }
        if (stop) break;
    }
    const double t_decode1 = cli_now_sec();
    generation_done(&printer);
    if (cli_interrupt_requested()) cli_interrupt_clear();

    const double prefill_s = t_prefill1 - t_prefill0;
    const double decode_s = t_decode1 - t_decode0;
    ds4_log(stderr,
            DS4_LOG_TIMING,
            "ds4: prefill: %.2f t/s, generation: %.2f t/s\n",
            prefill_s > 0.0 ? (double)prompt->len / prefill_s : 0.0,
            decode_s > 0.0 ? (double)generated / decode_s : 0.0);

    ds4_session_free(session);
    return 0;
}

static bool json_utf8_valid(const char *s, size_t n) {
    size_t i = 0;
    while (i < n) {
        unsigned char c = (unsigned char)s[i++];
        if (c < 0x80) continue;
        int need = 0;
        if (c >= 0xc2 && c <= 0xdf) need = 1;
        else if (c >= 0xe0 && c <= 0xef) need = 2;
        else if (c >= 0xf0 && c <= 0xf4) need = 3;
        else return false;
        if (i + (size_t)need > n) return false;
        unsigned char c1 = (unsigned char)s[i];
        if (c == 0xe0 && c1 < 0xa0) return false;
        if (c == 0xed && c1 >= 0xa0) return false;
        if (c == 0xf0 && c1 < 0x90) return false;
        if (c == 0xf4 && c1 >= 0x90) return false;
        for (int j = 0; j < need; j++) {
            unsigned char cc = (unsigned char)s[i + (size_t)j];
            if ((cc & 0xc0) != 0x80) return false;
        }
        i += (size_t)need;
    }
    return true;
}

static void json_write_string(FILE *fp, const char *s, size_t n) {
    bool valid_utf8 = json_utf8_valid(s, n);
    fputc('"', fp);
    for (size_t i = 0; i < n; i++) {
        unsigned char c = (unsigned char)s[i];
        if (c == '"' || c == '\\') {
            fputc('\\', fp);
            fputc((char)c, fp);
        } else if (c == '\n') {
            fputs("\\n", fp);
        } else if (c == '\r') {
            fputs("\\r", fp);
        } else if (c == '\t') {
            fputs("\\t", fp);
        } else if (c < 0x20) {
            fprintf(fp, "\\u%04x", (unsigned)c);
        } else if (!valid_utf8 && c >= 0x80) {
            /* Tokenizer pieces can be arbitrary byte fragments.  The bytes
             * array is authoritative; this escape keeps the JSON valid. */
            fprintf(fp, "\\u%04x", (unsigned)c);
        } else {
            fputc((char)c, fp);
        }
    }
    fputc('"', fp);
}

static void json_write_token(FILE *fp, ds4_engine *engine, int token) {
    size_t n = 0;
    char *text = ds4_token_text(engine, token, &n);
    fprintf(fp, "{\"id\":%d,\"text\":", token);
    json_write_string(fp, text, n);
    fputs(",\"bytes\":[", fp);
    for (size_t i = 0; i < n; i++) {
        if (i) fputc(',', fp);
        fprintf(fp, "%u", (unsigned)(unsigned char)text[i]);
    }
    fputc(']', fp);
    fputc('}', fp);
    free(text);
}

static int run_logits_dump(ds4_engine *engine, const cli_config *cfg, const ds4_tokens *prompt) {
    ds4_session *session = NULL;
    if (ds4_session_create(&session, engine, cfg->gen.ctx_size) != 0) {
        fprintf(stderr, "ds4: --dump-logits requires a graph session backend\n");
        return 1;
    }
    if (cli_wait_distributed_route(cfg, session) != 0) {
        ds4_session_free(session);
        return 1;
    }

    char err[160];
    cli_prefill_progress progress = {
        .base_tokens = 0,
        .input_tokens = prompt->len,
        .use_color = ds4_log_is_tty(stderr),
    };
    ds4_session_set_progress(session, cli_prefill_progress_cb, &progress);
    ds4_session_set_display_progress(session,
                                     progress.use_color ? cli_prefill_progress_cb : NULL,
                                     progress.use_color ? &progress : NULL);
    if (ds4_session_sync(session, prompt, err, sizeof(err)) != 0) {
        ds4_session_set_progress(session, NULL, NULL);
        ds4_session_set_display_progress(session, NULL, NULL);
        fprintf(stderr, "ds4: prompt processing failed: %s\n", err);
        ds4_session_free(session);
        return 1;
    }
    ds4_session_set_progress(session, NULL, NULL);
    ds4_session_set_display_progress(session, NULL, NULL);

    const int vocab = ds4_engine_vocab_size(engine);
    float *logits = malloc((size_t)vocab * sizeof(logits[0]));
    if (!logits) {
        ds4_session_free(session);
        return 1;
    }
    if (ds4_session_copy_logits(session, logits, vocab) != vocab) {
        fprintf(stderr, "ds4: failed to copy session logits\n");
        free(logits);
        ds4_session_free(session);
        return 1;
    }

    FILE *fp = fopen(cfg->gen.dump_logits_path, "wb");
    if (!fp) {
        fprintf(stderr, "ds4: failed to open --dump-logits file: %s\n", cfg->gen.dump_logits_path);
        free(logits);
        ds4_session_free(session);
        return 1;
    }

    fprintf(fp, "{\n  \"source\":\"ds4\",\n  \"model\":");
    json_write_string(fp, cfg->engine.model_path, strlen(cfg->engine.model_path));
    fprintf(fp,
            ",\n  \"backend\":\"%s\",\n  \"quant_bits\":%d,\n"
            "  \"prompt_tokens\":%d,\n  \"ctx\":%d,\n  \"vocab\":%d,\n",
            ds4_backend_name(cfg->engine.backend),
            ds4_engine_routed_quant_bits(engine),
            prompt->len,
            cfg->gen.ctx_size,
            vocab);
    const int argmax = ds4_session_argmax(session);
    fputs("  \"argmax_token\":", fp);
    json_write_token(fp, engine, argmax);
    fprintf(fp, ",\n  \"argmax_logit\":%.9g,\n  \"logits\":[", logits[argmax]);
    for (int i = 0; i < vocab; i++) {
        if (i) fputc(',', fp);
        if ((i % 8) == 0) fputs("\n    ", fp);
        if (isfinite(logits[i])) {
            fprintf(fp, "%.9g", logits[i]);
        } else {
            fputs("null", fp);
        }
    }
    fputs("\n  ]\n}\n", fp);
    if (fclose(fp) != 0) {
        fprintf(stderr, "ds4: failed to close --dump-logits file: %s\n", cfg->gen.dump_logits_path);
        free(logits);
        ds4_session_free(session);
        return 1;
    }

    free(logits);
    ds4_session_free(session);
    return 0;
}

static int run_logprob_dump(ds4_engine *engine, const cli_config *cfg, const ds4_tokens *prompt) {
    ds4_session *session = NULL;
    if (ds4_session_create(&session, engine, cfg->gen.ctx_size) != 0) {
        fprintf(stderr, "ds4: --dump-logprobs requires a graph session backend\n");
        return 1;
    }
    if (cli_wait_distributed_route(cfg, session) != 0) {
        ds4_session_free(session);
        return 1;
    }

    char err[160];
    cli_prefill_progress progress = {
        .base_tokens = 0,
        .input_tokens = prompt->len,
        .use_color = ds4_log_is_tty(stderr),
    };
    ds4_session_set_progress(session, cli_prefill_progress_cb, &progress);
    ds4_session_set_display_progress(session,
                                     progress.use_color ? cli_prefill_progress_cb : NULL,
                                     progress.use_color ? &progress : NULL);
    if (ds4_session_sync(session, prompt, err, sizeof(err)) != 0) {
        ds4_session_set_progress(session, NULL, NULL);
        ds4_session_set_display_progress(session, NULL, NULL);
        fprintf(stderr, "ds4: prompt processing failed: %s\n", err);
        ds4_session_free(session);
        return 1;
    }
    ds4_session_set_progress(session, NULL, NULL);
    ds4_session_set_display_progress(session, NULL, NULL);

    FILE *fp = fopen(cfg->gen.dump_logprobs_path, "wb");
    if (!fp) {
        fprintf(stderr, "ds4: failed to open --dump-logprobs file: %s\n", cfg->gen.dump_logprobs_path);
        ds4_session_free(session);
        return 1;
    }

    int k = cfg->gen.dump_logprobs_top_k > 0 ? cfg->gen.dump_logprobs_top_k : 20;
    if (k > 128) k = 128;
    ds4_token_score *scores = calloc((size_t)k, sizeof(scores[0]));
    if (!scores) {
        fclose(fp);
        ds4_session_free(session);
        return 1;
    }

    fprintf(fp, "{\n  \"source\":\"ds4\",\n  \"prompt_tokens\":%d,\n  \"ctx\":%d,\n  \"top_k\":%d,\n  \"steps\":[\n",
            prompt->len, cfg->gen.ctx_size, k);
    int generated = 0;
    int max_tokens = cfg->gen.n_predict;
    int room = ds4_session_ctx(session) - ds4_session_pos(session);
    if (room <= 1) max_tokens = 0;
    else if (max_tokens > room - 1) max_tokens = room - 1;
    for (; generated < max_tokens; generated++) {
        int n = ds4_session_top_logprobs(session, scores, k);
        int token = ds4_session_argmax(session);
        if (generated) fputs(",\n", fp);
        fprintf(fp, "    {\"step\":%d,\"selected\":", generated);
        json_write_token(fp, engine, token);
        fputs(",\"top_logprobs\":[", fp);
        for (int i = 0; i < n && scores[i].id >= 0; i++) {
            if (i) fputc(',', fp);
            fputs("{\"token\":", fp);
            json_write_token(fp, engine, scores[i].id);
            fprintf(fp, ",\"logit\":%.9g,\"logprob\":%.9g}", scores[i].logit, scores[i].logprob);
        }
        fputs("]}", fp);

        if (token == ds4_token_eos(engine)) break;
        if (ds4_session_eval(session, token, err, sizeof(err)) != 0) {
            fprintf(stderr, "ds4: decode failed while dumping logprobs: %s\n", err);
            free(scores);
            fclose(fp);
            ds4_session_free(session);
            return 1;
        }
    }
    fputs("\n  ]\n}\n", fp);
    if (fclose(fp) != 0) {
        fprintf(stderr, "ds4: failed to close --dump-logprobs file: %s\n", cfg->gen.dump_logprobs_path);
        free(scores);
        ds4_session_free(session);
        return 1;
    }
    free(scores);
    ds4_session_free(session);
    return 0;
}

static int run_perplexity_file(ds4_engine *engine, const cli_config *cfg) {
    char *text = read_prompt_file(cfg->gen.perplexity_file_path, true);
    ds4_tokens tokens = {0};
    ds4_tokenize_text(engine, text, &tokens);
    free(text);

    /* Seed the graph with enough real context to stay on the normal Metal
     * prefill path; scoring starts immediately after this fixed prefix. */
    const int prefix_len = 32;
    if (tokens.len <= prefix_len) {
        fprintf(stderr, "ds4: --perplexity-file needs more than %d tokens\n", prefix_len);
        ds4_tokens_free(&tokens);
        return 1;
    }

    int scored = tokens.len - prefix_len;
    if (cfg->gen.n_predict > 0 && scored > cfg->gen.n_predict) scored = cfg->gen.n_predict;
    if (scored > cfg->gen.ctx_size - prefix_len) scored = cfg->gen.ctx_size - prefix_len;
    if (scored <= 0) {
        fprintf(stderr, "ds4: context too small for perplexity scoring\n");
        ds4_tokens_free(&tokens);
        return 1;
    }

    ds4_session *session = NULL;
    if (ds4_session_create(&session, engine, cfg->gen.ctx_size) != 0) {
        fprintf(stderr, "ds4: --perplexity-file requires a graph session backend\n");
        ds4_tokens_free(&tokens);
        return 1;
    }
    if (cli_wait_distributed_route(cfg, session) != 0) {
        ds4_session_free(session);
        ds4_tokens_free(&tokens);
        return 1;
    }

    ds4_tokens prefix = {0};
    for (int i = 0; i < prefix_len; i++) ds4_tokens_push(&prefix, tokens.v[i]);
    char err[160];
    if (ds4_session_sync(session, &prefix, err, sizeof(err)) != 0) {
        fprintf(stderr, "ds4: perplexity initial token failed: %s\n", err);
        ds4_tokens_free(&prefix);
        ds4_session_free(session);
        ds4_tokens_free(&tokens);
        return 1;
    }
    ds4_tokens_free(&prefix);

    double nll = 0.0;
    for (int j = 0; j < scored; j++) {
        const int i = prefix_len + j;
        ds4_token_score score;
        if (!ds4_session_token_logprob(session, tokens.v[i], &score)) {
            fprintf(stderr, "ds4: failed to score token %d\n", i);
            ds4_session_free(session);
            ds4_tokens_free(&tokens);
            return 1;
        }
        nll -= (double)score.logprob;

        if (((j + 1) % 256) == 0 || j + 1 == scored) {
            fprintf(stderr, "ds4: perplexity scored %d/%d\r", j + 1, scored);
            fflush(stderr);
        }

        if (j + 1 < scored && ds4_session_eval(session, tokens.v[i], err, sizeof(err)) != 0) {
            fprintf(stderr, "\nds4: perplexity decode failed at token %d: %s\n", i, err);
            ds4_session_free(session);
            ds4_tokens_free(&tokens);
            return 1;
        }
    }
    fputc('\n', stderr);

    const double avg_nll = nll / (double)scored;
    printf("tokens=%d scored=%d nll=%.9f avg_nll=%.9f ppl=%.9f\n",
           tokens.len, scored, nll, avg_nll, exp(avg_nll));

    ds4_session_free(session);
    ds4_tokens_free(&tokens);
    return 0;
}

static int run_generation(ds4_engine *engine, const cli_config *cfg) {
    ds4_tokens prompt = {0};
    build_prompt(engine, &cfg->gen, &prompt);

    int rc = 0;
    if (cfg->gen.metal_graph_test) {
        rc = ds4_engine_metal_graph_test(engine, &prompt);
        ds4_tokens_free(&prompt);
        return rc;
    }
    if (cfg->gen.metal_graph_full_test) {
        rc = ds4_engine_metal_graph_full_test(engine, &prompt);
        ds4_tokens_free(&prompt);
        return rc;
    }
    if (cfg->gen.metal_graph_prompt_test) {
        rc = ds4_engine_metal_graph_prompt_test(engine, &prompt, cfg->gen.ctx_size);
        ds4_tokens_free(&prompt);
        return rc;
    }
    if (cfg->gen.dump_logits_path) {
        rc = run_logits_dump(engine, cfg, &prompt);
        ds4_tokens_free(&prompt);
        return rc;
    }
    if (cfg->gen.dump_logprobs_path) {
        rc = run_logprob_dump(engine, cfg, &prompt);
        ds4_tokens_free(&prompt);
        return rc;
    }

    const bool diagnostic = cfg->gen.dump_tokens ||
                            cfg->gen.head_test ||
                            cfg->gen.first_token_test;
    if (cfg->gen.head_test) {
        rc = ds4_engine_head_test(engine, &prompt);
    }
    if (rc == 0 && cfg->gen.first_token_test) {
        rc = ds4_engine_first_token_test(engine, &prompt);
    }
    if (cfg->gen.dump_tokens) {
        ds4_engine_dump_tokens(engine, &prompt);
    }

    if (diagnostic) {
        if (rc == 0) {
            fprintf(stderr, "ds4: diagnostic run completed on the native %s path.\n",
                    ds4_backend_name(cfg->engine.backend));
        }
    } else if (cfg->engine.distributed.role == DS4_DISTRIBUTED_COORDINATOR ||
               cfg->gen.temperature > 0.0f ||
               ds4_engine_mtp_draft_tokens(engine) > 1) {
        rc = run_sampled_generation(engine, cfg, &prompt);
    } else {
        token_printer printer = {
            .engine = engine,
            .fp = stdout,
            .format_thinking = ds4_think_mode_enabled(cli_effective_think_mode(&cfg->gen)),
            .in_think = ds4_think_mode_enabled(cli_effective_think_mode(&cfg->gen)),
            .use_color = isatty(fileno(stdout)) != 0,
            .last_output_newline = true,
        };
        cli_prefill_progress progress = {
            .base_tokens = 0,
            .input_tokens = prompt.len,
            .use_color = ds4_log_is_tty(stderr),
        };
        rc = ds4_engine_generate_argmax(engine, &prompt, cfg->gen.n_predict,
                                        cfg->gen.ctx_size,
                                        print_generated_token,
                                        generation_done,
                                        &printer,
                                        cli_prefill_progress_cb,
                                        &progress);
    }

    ds4_tokens_free(&prompt);
    return rc;
}

static char *trim_inplace(char *s) {
    while (*s && isspace((unsigned char)*s)) s++;
    char *end = s + strlen(s);
    while (end > s && isspace((unsigned char)end[-1])) end--;
    *end = '\0';
    return s;
}

static void print_repl_help(void) {
    puts("Commands:");
    puts("  /help          Show this help.");
    puts("  /think         Use normal thinking mode.");
    puts("  /think-max     Use Think Max only when context is at least 393216 tokens.");
    puts("  /nothink       Disable thinking mode.");
    puts("  /ctx N         Set context size for following prompts.");
    puts("  /power N       Set GPU duty cycle percentage, 1..100.");
    puts("  /read FILE     Read a prompt from FILE and run it.");
    puts("  /quit, /exit   Leave the prompt.");
    puts("  Ctrl+C         Stop generation; at the prompt, exit.");
}

static bool parse_power_percent(const char *arg, int *out) {
    char *end = NULL;
    long v = strtol(arg, &end, 10);
    if (!arg[0] || *end != '\0' || v < 1 || v > 100) return false;
    *out = (int)v;
    return true;
}

static void history_file_path(char *buf, size_t len) {
    const char *home = getenv("HOME");
    if (!home || !home[0]) home = ".";
    snprintf(buf, len, "%s/.ds4_history", home);
}

typedef struct {
    ds4_session *session;
    ds4_tokens transcript;
    int ctx_size;
    int max_prefix_tokens;
} repl_chat;

static void tokens_insert(ds4_tokens *dst, int pos, const ds4_tokens *src) {
    if (!src || src->len <= 0) return;
    if (pos < 0) pos = 0;
    if (pos > dst->len) pos = dst->len;
    while (dst->len + src->len > dst->cap) {
        dst->cap = dst->cap ? dst->cap * 2 : 64;
        int *next = realloc(dst->v, (size_t)dst->cap * sizeof(dst->v[0]));
        if (!next) {
            perror("ds4: realloc");
            exit(1);
        }
        dst->v = next;
    }
    memmove(dst->v + pos + src->len, dst->v + pos,
            (size_t)(dst->len - pos) * sizeof(dst->v[0]));
    memcpy(dst->v + pos, src->v, (size_t)src->len * sizeof(src->v[0]));
    dst->len += src->len;
}

static void tokens_remove(ds4_tokens *dst, int pos, int n) {
    if (n <= 0 || pos < 0 || pos >= dst->len) return;
    if (pos + n > dst->len) n = dst->len - pos;
    memmove(dst->v + pos, dst->v + pos + n,
            (size_t)(dst->len - pos - n) * sizeof(dst->v[0]));
    dst->len -= n;
}

/* Insert/remove the Think Max prefix inside the existing transcript.  The
 * prefix lives after BOS, before any system/developer text, which mirrors the
 * API rendering path.  Changing it invalidates the session because every later
 * token position would otherwise refer to the wrong prefix. */
static void repl_chat_apply_max_prefix(ds4_engine *engine, repl_chat *chat, bool enable) {
    if (enable && chat->max_prefix_tokens == 0) {
        ds4_tokens prefix = {0};
        ds4_chat_append_max_effort_prefix(engine, &prefix);
        tokens_insert(&chat->transcript, 1, &prefix);
        chat->max_prefix_tokens = prefix.len;
        ds4_tokens_free(&prefix);
        if (chat->session) ds4_session_invalidate(chat->session);
    } else if (!enable && chat->max_prefix_tokens > 0) {
        tokens_remove(&chat->transcript, 1, chat->max_prefix_tokens);
        chat->max_prefix_tokens = 0;
        if (chat->session) ds4_session_invalidate(chat->session);
    }
}

static int repl_chat_create_session(ds4_engine *engine,
                                    repl_chat *chat,
                                    const cli_config *cfg,
                                    int ctx_size) {
    ds4_session *session = NULL;
    if (ds4_session_create(&session, engine, ctx_size) != 0) {
        fprintf(stderr, "ds4: interactive chat KV cache requires a session backend\n");
        return 1;
    }
    if (cli_wait_distributed_route(cfg, session) != 0) {
        ds4_session_free(session);
        return 1;
    }
    if (chat->session) ds4_session_free(chat->session);
    chat->session = session;
    chat->ctx_size = ctx_size;
    return 0;
}

static int repl_chat_init(ds4_engine *engine, repl_chat *chat, const cli_config *cfg) {
    memset(chat, 0, sizeof(*chat));
    ds4_chat_begin(engine, &chat->transcript);
    repl_chat_apply_max_prefix(engine, chat,
                               cli_effective_think_mode(&cfg->gen) == DS4_THINK_MAX);
    if (cfg->gen.system && cfg->gen.system[0]) {
        ds4_chat_append_message(engine, &chat->transcript, "system", cfg->gen.system);
    }
    return repl_chat_create_session(engine, chat, cfg, cfg->gen.ctx_size);
}

static void repl_chat_free(repl_chat *chat) {
    if (!chat) return;
    ds4_session_free(chat->session);
    ds4_tokens_free(&chat->transcript);
    memset(chat, 0, sizeof(*chat));
}

static int repl_chat_set_ctx(ds4_engine *engine,
                             repl_chat *chat,
                             const cli_config *cfg,
                             int ctx_size) {
    ds4_session_free(chat->session);
    chat->session = NULL;
    chat->ctx_size = 0;
    return repl_chat_create_session(engine, chat, cfg, ctx_size);
}

enum {
    CLI_CHAT_OK = 0,
    CLI_CHAT_FATAL = 1,
    CLI_CHAT_INTERRUPTED = 2,
};

/* Run one interactive turn.  The transcript is tentatively extended with user
 * and assistant markers, then ds4_session_sync() decides whether this is a KV
 * continuation.  If prompt processing fails, the transcript rolls back before
 * returning to the prompt. */
static int run_chat_turn(ds4_engine *engine, cli_config *cfg, repl_chat *chat, const char *user_text) {
    if (!chat->session) {
        fprintf(stderr, "ds4: no active interactive KV cache\n");
        return CLI_CHAT_FATAL;
    }
    ds4_session_set_cancel(chat->session, cli_session_cancel_cb, NULL);

    ds4_think_mode think_mode = ds4_think_mode_for_context(cfg->gen.think_mode,
                                                           chat->ctx_size);
    repl_chat_apply_max_prefix(engine, chat, think_mode == DS4_THINK_MAX);
    const int rollback_len = chat->transcript.len;
    ds4_chat_append_message(engine, &chat->transcript, "user", user_text);
    ds4_chat_append_assistant_prefix(engine, &chat->transcript, think_mode);

    const int old_pos = ds4_session_pos(chat->session);
    const int common = ds4_session_common_prefix(chat->session, &chat->transcript);
    const int cached = common == old_pos && chat->transcript.len >= old_pos ? common : 0;
    const int suffix = chat->transcript.len - cached;

    char err[160];
    cli_prefill_progress progress = {
        .base_tokens = cached,
        .input_tokens = suffix,
        .use_color = ds4_log_is_tty(stderr),
    };
    const double t_prefill0 = cli_now_sec();
    ds4_session_set_progress(chat->session, cli_prefill_progress_cb, &progress);
    ds4_session_set_display_progress(chat->session,
                                     progress.use_color ? cli_prefill_progress_cb : NULL,
                                     progress.use_color ? &progress : NULL);
    cli_dist_busy_set(cfg, true);
    int sync_rc = ds4_session_sync(chat->session, &chat->transcript, err, sizeof(err));
    cli_dist_busy_set(cfg, false);
    if (sync_rc != 0) {
        ds4_session_set_progress(chat->session, NULL, NULL);
        ds4_session_set_display_progress(chat->session, NULL, NULL);
        chat->transcript.len = rollback_len;
        if (sync_rc == DS4_SESSION_SYNC_INTERRUPTED || cli_interrupt_requested()) {
            fprintf(stderr, "ds4: prompt processing interrupted\n");
            ds4_session_invalidate(chat->session);
            cli_interrupt_clear();
            ds4_session_set_cancel(chat->session, NULL, NULL);
            return CLI_CHAT_INTERRUPTED;
        } else {
            fprintf(stderr, "ds4: prompt processing failed: %s\n", err);
        }
        ds4_session_set_cancel(chat->session, NULL, NULL);
        return CLI_CHAT_FATAL;
    }
    ds4_session_set_progress(chat->session, NULL, NULL);
    ds4_session_set_display_progress(chat->session, NULL, NULL);
    const double t_prefill1 = cli_now_sec();

    token_printer printer = {
        .engine = engine,
        .fp = stdout,
        .format_thinking = ds4_think_mode_enabled(think_mode),
        .in_think = ds4_think_mode_enabled(think_mode),
        .use_color = isatty(fileno(stdout)) != 0,
        .last_output_newline = true,
    };

    int max_tokens = cfg->gen.n_predict;
    int room = ds4_session_ctx(chat->session) - ds4_session_pos(chat->session);
    if (room <= 1) max_tokens = 0;
    else if (max_tokens > room - 1) max_tokens = room - 1;

    uint64_t rng = cfg->gen.seed ? cfg->gen.seed :
        ((uint64_t)time(NULL) ^ ((uint64_t)getpid() << 32) ^ (uint64_t)clock());
    int generated = 0;
    const double t_decode0 = cli_now_sec();
    while (generated < max_tokens && !cli_interrupt_requested()) {
        int token = ds4_session_sample(chat->session,
                                       cfg->gen.temperature,
                                       0,
                                       cfg->gen.top_p,
                                       cfg->gen.min_p,
                                       &rng);
        if (token == ds4_token_eos(engine)) break;

        int toks[17];
        int ntok = 0;
        if (cfg->gen.temperature <= 0.0f && ds4_engine_mtp_draft_tokens(engine) > 1 &&
            getenv("DS4_MTP_SPEC_DISABLE") == NULL) {
            cli_dist_busy_set(cfg, true);
            ntok = ds4_session_eval_speculative_argmax(chat->session,
                                                       token,
                                                       max_tokens - generated,
                                                       ds4_token_eos(engine),
                                                       toks,
                                                       (int)(sizeof(toks) / sizeof(toks[0])),
                                                       err,
                                                       sizeof(err));
            cli_dist_busy_set(cfg, false);
            if (ntok < 0) {
                if (cli_interrupt_requested()) {
                    fprintf(stderr, "ds4: generation interrupted\n");
                    if (generated == 0) chat->transcript.len = rollback_len;
                    else ds4_tokens_push(&chat->transcript, ds4_token_eos(engine));
                    ds4_session_invalidate(chat->session);
                    cli_interrupt_clear();
                    ds4_session_set_cancel(chat->session, NULL, NULL);
                    return CLI_CHAT_INTERRUPTED;
                } else {
                    fprintf(stderr, "ds4: decode failed: %s\n", err);
                }
                ds4_session_set_cancel(chat->session, NULL, NULL);
                return CLI_CHAT_FATAL;
            }
        } else {
            cli_dist_busy_set(cfg, true);
            int eval_rc = ds4_session_eval(chat->session, token, err, sizeof(err));
            cli_dist_busy_set(cfg, false);
            if (eval_rc != 0) {
                if (cli_interrupt_requested()) {
                    fprintf(stderr, "ds4: generation interrupted\n");
                    if (generated == 0) chat->transcript.len = rollback_len;
                    else ds4_tokens_push(&chat->transcript, ds4_token_eos(engine));
                    ds4_session_invalidate(chat->session);
                    cli_interrupt_clear();
                    ds4_session_set_cancel(chat->session, NULL, NULL);
                    return CLI_CHAT_INTERRUPTED;
                } else {
                    fprintf(stderr, "ds4: decode failed: %s\n", err);
                }
                ds4_session_set_cancel(chat->session, NULL, NULL);
                return CLI_CHAT_FATAL;
            }
            toks[0] = token;
            ntok = 1;
        }

        bool stop = false;
        for (int j = 0; j < ntok; j++) {
            if (toks[j] == ds4_token_eos(engine)) {
                stop = true;
                break;
            }
            size_t piece_len = 0;
            char *piece = ds4_token_text(engine, toks[j], &piece_len);
            ds4_tokens_push(&chat->transcript, toks[j]);
            token_printer_write_text(&printer, piece, piece_len);
            fflush(stdout);
            free(piece);
            generated++;
            if (generated >= max_tokens) break;
        }
        if (stop) break;
    }
    const double t_decode1 = cli_now_sec();
    generation_done(&printer);

    const bool interrupted = cli_interrupt_requested();
    if (interrupted && generated == 0) {
        chat->transcript.len = rollback_len;
        ds4_session_invalidate(chat->session);
    } else {
        ds4_tokens_push(&chat->transcript, ds4_token_eos(engine));
    }

    const double prefill_s = t_prefill1 - t_prefill0;
    const double decode_s = t_decode1 - t_decode0;
    if (interrupted) cli_interrupt_clear();
    ds4_log(stderr,
            DS4_LOG_TIMING,
            "ds4: prefill: %.2f t/s, generation: %.2f t/s\n",
            prefill_s > 0.0 ? (double)suffix / prefill_s : 0.0,
            decode_s > 0.0 ? (double)generated / decode_s : 0.0);
    ds4_session_set_cancel(chat->session, NULL, NULL);
    return CLI_CHAT_OK;
}

static int run_repl(ds4_engine *engine, cli_config *cfg) {
    struct sigaction old_int;
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sigemptyset(&sa.sa_mask);
    sa.sa_handler = cli_sigint_handler;
    bool sigint_installed = sigaction(SIGINT, &sa, &old_int) == 0;
    cli_interrupt_clear();

    repl_chat chat;
    if (repl_chat_init(engine, &chat, cfg) != 0) {
        if (sigint_installed) sigaction(SIGINT, &old_int, NULL);
        return 1;
    }

    char hist[PATH_MAX];
    history_file_path(hist, sizeof(hist));
    linenoiseSetMultiLine(1);
    linenoiseHistorySetMaxLen(512);
    linenoiseHistoryLoad(hist);
    print_repl_help();

    int rc = 0;
    for (;;) {
        errno = 0;
        char *line = linenoise("ds4> ");
        if (!line) {
            if (errno == EAGAIN || cli_interrupt_requested()) {
                cli_interrupt_clear();
                break;
            }
            break;
        }
        char *cmd = trim_inplace(line);
        if (!cmd[0]) {
            linenoiseFree(line);
            continue;
        }
        linenoiseHistoryAdd(cmd);
        linenoiseHistorySave(hist);

        if (!strcmp(cmd, "/help")) {
            print_repl_help();
        } else if (!strcmp(cmd, "/think")) {
            cfg->gen.think_mode = DS4_THINK_HIGH;
            repl_chat_apply_max_prefix(engine, &chat, false);
            puts("Thinking mode: high.");
        } else if (!strcmp(cmd, "/think-max")) {
            cfg->gen.think_mode = DS4_THINK_MAX;
            bool active = ds4_think_mode_for_context(cfg->gen.think_mode,
                                                     chat.ctx_size) == DS4_THINK_MAX;
            repl_chat_apply_max_prefix(engine, &chat, active);
            cli_warn_think_max_downgraded(&cfg->gen, "/think-max");
            printf("Thinking mode: %s.\n", active ? "max" : "high (ctx below 393216)");
        } else if (!strcmp(cmd, "/nothink")) {
            cfg->gen.think_mode = DS4_THINK_NONE;
            repl_chat_apply_max_prefix(engine, &chat, false);
            puts("Thinking mode: none.");
        } else if (!strncmp(cmd, "/power", 6) && (cmd[6] == '\0' || isspace((unsigned char)cmd[6]))) {
            char *arg = trim_inplace(cmd + 6);
            if (!arg[0]) {
                printf("Power: %d%%.\n", ds4_session_power(chat.session));
            } else {
                int power = 0;
                if (!parse_power_percent(arg, &power)) {
                    fprintf(stderr, "ds4: /power must be between 1 and 100\n");
                } else if (ds4_session_set_power(chat.session, power) != 0) {
                    fprintf(stderr, "ds4: failed to set /power\n");
                } else {
                    cfg->engine.power_percent = power;
                    printf("Power: %d%%.\n", power);
                }
            }
        } else if (!strncmp(cmd, "/ctx", 4) && (cmd[4] == '\0' || isspace((unsigned char)cmd[4]))) {
            char *arg = trim_inplace(cmd + 4);
            if (!arg[0]) {
                fprintf(stderr, "ds4: /ctx needs a positive integer\n");
            } else {
                cfg->gen.ctx_size = parse_int(arg, "/ctx");
                log_context_memory(cfg->engine.backend,
                                   cfg->gen.ctx_size,
                                   cfg->engine.prefill_chunk);
                rc = repl_chat_set_ctx(engine, &chat, cfg, cfg->gen.ctx_size);
                if (rc != 0) {
                    linenoiseFree(line);
                    break;
                }
                bool active = ds4_think_mode_for_context(cfg->gen.think_mode,
                                                         chat.ctx_size) == DS4_THINK_MAX;
                repl_chat_apply_max_prefix(engine, &chat, active);
                cli_warn_think_max_downgraded(&cfg->gen, "/ctx");
            }
        } else if (!strcmp(cmd, "/quit") || !strcmp(cmd, "/exit")) {
            linenoiseFree(line);
            break;
        } else if (!strncmp(cmd, "/read", 5) && (cmd[5] == '\0' || isspace((unsigned char)cmd[5]))) {
            char *path = trim_inplace(cmd + 5);
            if (!path[0]) {
                fprintf(stderr, "ds4: /read needs a file path\n");
            } else {
                char *prompt = read_prompt_file(path, false);
                if (prompt) {
                    rc = run_chat_turn(engine, cfg, &chat, prompt);
                    free(prompt);
                }
            }
        } else if (cmd[0] == '/') {
            fprintf(stderr, "ds4: unknown command: %s\n", cmd);
            fprintf(stderr, "ds4: type /help for commands\n");
        } else {
            rc = run_chat_turn(engine, cfg, &chat, cmd);
        }
        linenoiseFree(line);
        if (rc == CLI_CHAT_INTERRUPTED) {
            rc = 0;
            continue;
        }
        if (rc != 0 && cfg->local_gpus.enabled) break;
    }
    if (sigint_installed) sigaction(SIGINT, &old_int, NULL);
    repl_chat_free(&chat);
    return rc;
}

static const char *need_arg(int *i, int argc, char **argv, const char *opt) {
    if (*i + 1 >= argc) {
        fprintf(stderr, "ds4: missing value for %s\n", opt);
        exit(2);
    }
    return argv[++(*i)];
}

static char *read_prompt_file(const char *path, bool fatal) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "ds4: failed to open prompt file: %s\n", path);
        if (fatal) exit(2);
        return NULL;
    }
    if (fseek(fp, 0, SEEK_END) != 0) {
        fprintf(stderr, "ds4: failed to seek prompt file: %s\n", path);
        fclose(fp);
        if (fatal) exit(2);
        return NULL;
    }
    long len = ftell(fp);
    if (len < 0) {
        fprintf(stderr, "ds4: failed to size prompt file: %s\n", path);
        fclose(fp);
        if (fatal) exit(2);
        return NULL;
    }
    rewind(fp);

    char *buf = malloc((size_t)len + 1);
    if (!buf) {
        fprintf(stderr, "ds4: out of memory reading prompt file: %s\n", path);
        fclose(fp);
        if (fatal) exit(2);
        return NULL;
    }
    size_t nread = fread(buf, 1, (size_t)len, fp);
    if (nread != (size_t)len) {
        fprintf(stderr, "ds4: failed to read prompt file: %s\n", path);
        free(buf);
        fclose(fp);
        if (fatal) exit(2);
        return NULL;
    }
    if (fclose(fp) != 0) {
        fprintf(stderr, "ds4: failed to close prompt file: %s\n", path);
        free(buf);
        if (fatal) exit(2);
        return NULL;
    }
    buf[len] = '\0';
    return buf;
}

static cli_config parse_options(int argc, char **argv) {
    cli_config c = {
        .engine = {
            .model_path = "ds4flash.gguf",
            .backend = default_backend(),
            .mtp_draft_tokens = 1,
            .mtp_margin = 3.0f,
        },
        .gen = {
            .prompt = NULL,
            .system = "You are a helpful assistant",
            .n_predict = 50000,
            .ctx_size = 32768,
            .temperature = DS4_DEFAULT_TEMPERATURE,
            .top_p = DS4_DEFAULT_TOP_P,
            .min_p = DS4_DEFAULT_MIN_P,
            .dump_logprobs_top_k = 20,
            .think_mode = DS4_THINK_HIGH,
        },
        .local_gpus = {
            .coordinator_listen_fd = -1,
        },
    };

    c.dist = ds4_dist_options_create();
    if (!c.dist) {
        fprintf(stderr, "ds4: out of memory creating distributed options\n");
        exit(1);
    }

    bool directional_steering_scale_set = false;
    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        if (!strcmp(arg, "-h") || !strcmp(arg, "--help")) {
            const char *topic = (i + 1 < argc && argv[i + 1][0] != '-') ?
                argv[i + 1] : NULL;
            usage(stdout, topic);
            exit(0);
        }
        char dist_parse_err[256] = {0};
        ds4_dist_cli_parse_result dist_parse = ds4_dist_parse_cli_arg(arg,
                                                                      &i,
                                                                      argc,
                                                                      argv,
                                                                      c.dist,
                                                                      dist_parse_err,
                                                                      sizeof(dist_parse_err));
        if (dist_parse == DS4_DIST_CLI_ERROR) {
            fprintf(stderr, "ds4: %s\n", dist_parse_err[0] ? dist_parse_err : "invalid distributed option");
            exit(2);
        }
        if (dist_parse == DS4_DIST_CLI_MATCHED) continue;

        if (!strcmp(arg, "-p") || !strcmp(arg, "--prompt")) {
            if (c.gen.prompt) {
                fprintf(stderr, "ds4: specify only one prompt source\n");
                exit(2);
            }
            c.gen.prompt = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--prompt-file")) {
            if (c.gen.prompt) {
                fprintf(stderr, "ds4: specify only one prompt source\n");
                exit(2);
            }
            c.prompt_owned = read_prompt_file(need_arg(&i, argc, argv, arg), true);
            c.gen.prompt = c.prompt_owned;
        } else if (!strcmp(arg, "-sys") || !strcmp(arg, "--system")) {
            c.gen.system = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "-m") || !strcmp(arg, "--model")) {
            cli_add_model_path(&c, need_arg(&i, argc, argv, arg));
        } else if (!strcmp(arg, "--mtp")) {
            c.engine.mtp_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--mtp-draft")) {
            c.engine.mtp_draft_tokens = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--mtp-margin")) {
            c.engine.mtp_margin = parse_float_range(need_arg(&i, argc, argv, arg), arg, 0.0f, 1000.0f);
        } else if (!strcmp(arg, "-n") || !strcmp(arg, "--tokens")) {
            c.gen.n_predict = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "-c") || !strcmp(arg, "--ctx")) {
            c.gen.ctx_size = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--gpus")) {
            cli_local_gpus_parse(&c.local_gpus, need_arg(&i, argc, argv, arg));
        } else if (!strcmp(arg, "--temp")) {
            c.gen.temperature = parse_float_range(need_arg(&i, argc, argv, arg), arg, 0.0f, 100.0f);
        } else if (!strcmp(arg, "--top-p")) {
            c.gen.top_p = parse_float_range(need_arg(&i, argc, argv, arg), arg, 0.0f, 1.0f);
        } else if (!strcmp(arg, "--min-p")) {
            c.gen.min_p = parse_float_range(need_arg(&i, argc, argv, arg), arg, 0.0f, 1.0f);
        } else if (!strcmp(arg, "--seed")) {
            c.gen.seed = parse_u64(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--quality")) {
            c.engine.quality = true;
        } else if (!strcmp(arg, "--ssd-streaming")) {
            c.engine.ssd_streaming = true;
        } else if (!strcmp(arg, "--ssd-streaming-cold")) {
            c.engine.ssd_streaming_cold = true;
        } else if (!strcmp(arg, "--ssd-streaming-cache-experts")) {
            uint32_t experts = 0;
            uint64_t bytes = 0;
            if (!ds4_parse_streaming_cache_experts_arg(
                    need_arg(&i, argc, argv, arg), &experts, &bytes)) {
                fprintf(stderr,
                        "ds4: --ssd-streaming-cache-experts must be a positive count or <number>GB\n");
                exit(2);
            }
            c.engine.ssd_streaming_cache_experts = experts;
            c.engine.ssd_streaming_cache_bytes = bytes;
        } else if (!strcmp(arg, "--ssd-streaming-preload-experts")) {
            int v = parse_int(need_arg(&i, argc, argv, arg), arg);
            if (v <= 0) {
                fprintf(stderr, "ds4: --ssd-streaming-preload-experts must be positive\n");
                exit(2);
            }
            c.engine.ssd_streaming_preload_experts = (uint32_t)v;
        } else if (!strcmp(arg, "--simulate-used-memory")) {
            if (!ds4_parse_gib_arg(need_arg(&i, argc, argv, arg),
                                   &c.engine.simulate_used_memory_bytes)) {
                fprintf(stderr,
                        "ds4: --simulate-used-memory must be a positive GiB value, e.g. 64GB\n");
                exit(2);
            }
        } else if (!strcmp(arg, "--prefill-chunk")) {
            int v = parse_int(need_arg(&i, argc, argv, arg), arg);
            if (v <= 0) {
                fprintf(stderr, "ds4: --prefill-chunk must be positive\n");
                exit(2);
            }
            c.engine.prefill_chunk = (uint32_t)v;
            c.prefill_chunk_set = true;
        } else if (!strcmp(arg, "--power")) {
            c.engine.power_percent = parse_int(need_arg(&i, argc, argv, arg), arg);
            if (c.engine.power_percent < 1 || c.engine.power_percent > 100) {
                fprintf(stderr, "ds4: --power must be between 1 and 100\n");
                exit(2);
            }
        } else if (!strcmp(arg, "--dir-steering-file")) {
            c.engine.directional_steering_file = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--expert-profile")) {
            c.engine.expert_profile_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--dir-steering-ffn")) {
            c.engine.directional_steering_ffn = parse_float_range(need_arg(&i, argc, argv, arg), arg, -100.0f, 100.0f);
            directional_steering_scale_set = true;
        } else if (!strcmp(arg, "--dir-steering-attn")) {
            c.engine.directional_steering_attn = parse_float_range(need_arg(&i, argc, argv, arg), arg, -100.0f, 100.0f);
            directional_steering_scale_set = true;
        } else if (!strcmp(arg, "-t") || !strcmp(arg, "--threads")) {
            c.engine.n_threads = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--backend")) {
            c.engine.backend = parse_backend(need_arg(&i, argc, argv, arg));
        } else if (!strcmp(arg, "--cpu")) {
            c.engine.backend = DS4_BACKEND_CPU;
        } else if (!strcmp(arg, "--metal")) {
            c.engine.backend = DS4_BACKEND_METAL;
#ifdef DS4_ROCM_BUILD
        } else if (!strcmp(arg, "--rocm")) {
            c.engine.backend = DS4_BACKEND_CUDA;
#else
        } else if (!strcmp(arg, "--cuda")) {
            c.engine.backend = DS4_BACKEND_CUDA;
#endif
        } else if (!strcmp(arg, "--dump-tokens")) {
            c.gen.dump_tokens = true;
        } else if (!strcmp(arg, "--dump-logits")) {
            c.gen.dump_logits_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--dump-logprobs")) {
            c.gen.dump_logprobs_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--logprobs-top-k")) {
            c.gen.dump_logprobs_top_k = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--perplexity-file")) {
            c.gen.perplexity_file_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--imatrix-dataset")) {
            c.gen.imatrix_dataset_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--imatrix-out")) {
            c.gen.imatrix_output_path = need_arg(&i, argc, argv, arg);
            c.engine.backend = DS4_BACKEND_METAL;
        } else if (!strcmp(arg, "--imatrix-max-prompts")) {
            c.gen.imatrix_max_prompts = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--imatrix-max-tokens")) {
            c.gen.imatrix_max_tokens = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--think")) {
            c.gen.think_mode = DS4_THINK_HIGH;
        } else if (!strcmp(arg, "--think-max")) {
            c.gen.think_mode = DS4_THINK_MAX;
        } else if (!strcmp(arg, "--nothink")) {
            c.gen.think_mode = DS4_THINK_NONE;
        } else if (!strcmp(arg, "--head-test")) {
            c.gen.head_test = true;
        } else if (!strcmp(arg, "--first-token-test")) {
            c.gen.first_token_test = true;
        } else if (!strcmp(arg, "--metal-graph-test")) {
            c.gen.metal_graph_test = true;
#ifdef DS4_ROCM_BUILD
            c.engine.backend = DS4_BACKEND_CUDA;
#else
            c.engine.backend = DS4_BACKEND_METAL;
#endif
        } else if (!strcmp(arg, "--metal-graph-full-test")) {
            c.gen.metal_graph_full_test = true;
#ifdef DS4_ROCM_BUILD
            c.engine.backend = DS4_BACKEND_CUDA;
#else
            c.engine.backend = DS4_BACKEND_METAL;
#endif
        } else if (!strcmp(arg, "--metal-graph-prompt-test")) {
            c.gen.metal_graph_prompt_test = true;
#ifdef DS4_ROCM_BUILD
            c.engine.backend = DS4_BACKEND_CUDA;
#else
            c.engine.backend = DS4_BACKEND_METAL;
#endif
        } else if (!strcmp(arg, "--metal-graph-generate")) {
            fprintf(stderr, "ds4: --metal-graph-generate was removed; --metal is the graph path\n");
            exit(2);
        } else if (!strcmp(arg, "--inspect")) {
            c.inspect = true;
        } else if (!strcmp(arg, "--warm-weights")) {
            c.engine.warm_weights = true;
        } else if (!strcmp(arg, "--server")) {
            fprintf(stderr, "ds4: use ds4-server for the HTTP server\n");
            exit(2);
        } else {
            fprintf(stderr, "ds4: unknown option: %s\n", arg);
            usage(stderr, NULL);
            exit(2);
        }
    }

    if (c.engine.directional_steering_file && !directional_steering_scale_set) {
        c.engine.directional_steering_ffn = 1.0f;
    }
    if (c.gen.imatrix_output_path && !c.gen.imatrix_dataset_path) {
        fprintf(stderr, "ds4: --imatrix-out requires --imatrix-dataset\n");
        exit(2);
    }
    if (c.gen.imatrix_dataset_path && !c.gen.imatrix_output_path) {
        fprintf(stderr, "ds4: --imatrix-dataset requires --imatrix-out\n");
        exit(2);
    }
    if (c.gen.perplexity_file_path && c.gen.prompt) {
        fprintf(stderr, "ds4: --perplexity-file does not use -p/--prompt-file\n");
        exit(2);
    }
    if (c.model_path_count == 0) {
        c.model_paths[c.model_path_count++] = c.engine.model_path;
    }
    if (c.model_path_count > 1 && !c.local_gpus.enabled) {
        fprintf(stderr, "ds4: repeated -m/--model is currently supported only with --gpus\n");
        exit(2);
    }
    cli_local_gpus_configure(&c);
    char dist_err[256];
    if (ds4_dist_prepare_engine_options(c.dist, &c.engine, dist_err, sizeof(dist_err)) != 0) {
        fprintf(stderr, "ds4: %s\n", dist_err);
        exit(2);
    }

    return c;
}

int main(int argc, char **argv) {
    cli_config cfg = parse_options(argc, argv);
    ds4_internal_dist_copy_options_private(cfg.dist, &cfg.engine.distributed);
    if (cfg.gen.dump_tokens) {
        if (cfg.gen.prompt == NULL) {
            fprintf(stderr, "ds4: --dump-tokens requires -p or --prompt-file\n");
            free(cfg.prompt_owned);
            return 2;
        }
        int rc = ds4_dump_text_tokenization(cfg.engine.model_path,
                                            cfg.gen.prompt,
                                            stdout);
        ds4_dist_options_free(cfg.dist);
        free(cfg.prompt_owned);
        return rc;
    }
    cfg.engine.inspect_only = cfg.inspect;
    if (cli_local_gpus_start(&cfg) != 0) {
        ds4_dist_options_free(cfg.dist);
        free(cfg.prompt_owned);
        return 1;
    }
    ds4_engine *engine = NULL;
    if (ds4_engine_open(&engine, &cfg.engine) != 0) {
        cli_local_gpus_stop(&cfg.local_gpus);
        ds4_dist_options_free(cfg.dist);
        free(cfg.prompt_owned);
        return 1;
    }
    if (cfg.dist && cfg.dist->role == DS4_DISTRIBUTED_WORKER) {
        ds4_dist_generation_options dist_gen = {
            .prompt = cfg.gen.prompt,
            .system = cfg.gen.system,
            .dump_logits_path = cfg.gen.dump_logits_path,
            .dump_logprobs_path = cfg.gen.dump_logprobs_path,
            .dump_logprobs_top_k = cfg.gen.dump_logprobs_top_k,
            .n_predict = cfg.gen.n_predict,
            .ctx_size = cfg.gen.ctx_size,
            .temperature = cfg.gen.temperature,
            .top_p = cfg.gen.top_p,
            .min_p = cfg.gen.min_p,
            .seed = cfg.gen.seed,
            .think_mode = cfg.gen.think_mode,
        };
        int rc = ds4_dist_run(engine, cfg.dist, &dist_gen);
        ds4_engine_close(engine);
        cli_local_gpus_stop(&cfg.local_gpus);
        ds4_dist_options_free(cfg.dist);
        free(cfg.prompt_owned);
        return rc;
    }
    if (!cfg.inspect) {
        log_context_memory(cfg.engine.backend,
                           cfg.gen.ctx_size,
                           cfg.engine.prefill_chunk);
        cli_warn_think_max_downgraded(&cfg.gen, "--think-max");
    }
    int rc = 0;
    if (cfg.inspect) {
        ds4_engine_summary(engine);
    } else if (cfg.gen.imatrix_output_path) {
        rc = ds4_engine_collect_imatrix(engine,
                                        cfg.gen.imatrix_dataset_path,
                                        cfg.gen.imatrix_output_path,
                                        cfg.gen.ctx_size,
                                        cfg.gen.imatrix_max_prompts,
                                        cfg.gen.imatrix_max_tokens);
    } else if (cfg.gen.perplexity_file_path) {
        rc = run_perplexity_file(engine, &cfg);
    } else if (cfg.gen.prompt == NULL) {
        rc = run_repl(engine, &cfg);
    } else {
        rc = run_generation(engine, &cfg);
    }
    ds4_engine_close(engine);
    cli_local_gpus_stop(&cfg.local_gpus);
    ds4_dist_options_free(cfg.dist);
    free(cfg.prompt_owned);
    return rc;
}
