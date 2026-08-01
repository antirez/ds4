#include "ds4.h"
#include "ds4_distributed.h"
#include "ds4_gpu_args.h"
#include "ds4_help.h"
#include "ds4_kvstore.h"
#include "ds4_web.h"
#include "linenoise.h"

#include <errno.h>
#include <ctype.h>
#include <dirent.h>
#include <fnmatch.h>
#include <fcntl.h>
#include <limits.h>
#include <math.h>
#include <poll.h>
#include <pthread.h>
#include <regex.h>
#include <signal.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

/* This is intentionally not in linenoise.h, but it is part of the existing
 * multiplexed editor implementation.  The agent uses it only to restore text
 * after Enter is pressed while the model is still busy. */
int linenoiseEditInsert(struct linenoiseState *l, const char *c, size_t clen);

static int set_nonblock(int fd, bool on, int *old_flags);
static bool agent_parse_bool_default(const char *s, bool def);

/* ============================================================================
 * Configuration, Worker State, And Streaming Types
 * ============================================================================
 *
 * The agent is intentionally a single process: the UI thread owns terminal
 * input/output, while the worker thread owns the live DS4 session and KV state.
 * These types define the shared state and the small streaming state machines
 * used to render sampled assistant text and DSML tool calls as they arrive.
 */

typedef struct {
    const char *prompt;
    const char *system;
    const char *trace_path;
    bool raw_prompt;
    int n_predict;
    int ctx_size;
    float temperature;
    float top_p;
    float min_p;
    bool temperature_set;
    bool top_p_set;
    bool min_p_set;
    uint64_t seed;
    ds4_think_mode think_mode;
} agent_generation_options;

typedef struct {
    ds4_engine_options engine;
    agent_generation_options gen;
    const char *gpu_vram_arg;
    const char *gpu_devices_arg;
    const char *chdir_path;
    bool non_interactive;
} agent_config;

typedef enum {
    AGENT_WORKER_IDLE,
    AGENT_WORKER_PREFILL,
    AGENT_WORKER_GENERATING,
    AGENT_WORKER_COMPACTING,
    AGENT_WORKER_DRAINING,
    AGENT_WORKER_SAVING,
    AGENT_WORKER_ERROR,
    AGENT_WORKER_STOPPED,
} agent_worker_state;

typedef struct {
    agent_worker_state state;
    int prefill_done;
    int prefill_total;
    unsigned prefill_label;
    double prefill_tps;
    int generated;
    double gen_tps;
    bool greedy_sampling;
    int ctx_used;
    int ctx_size;
    int power_percent;
    char error[256];
} agent_status;

typedef struct agent_bash_job agent_bash_job;

typedef struct {
    ds4_engine *engine;
    agent_config *cfg;
    ds4_session *session;
    ds4_tokens transcript;
    char *cache_dir;
    char *sysprompt_path;
    char session_sha[41];
    char *session_title;
    uint64_t session_created_at;
    char *legacy_session_path_to_delete;
    bool user_activity;
    bool session_dirty;
    pthread_t thread;
    pthread_mutex_t mu;
    pthread_cond_t cond;
    int wake_fd[2];
    FILE *trace;
    bool wake_pending;
    bool stop;
    bool interrupt;
    bool initialized;
    bool save_requested;
    bool compact_requested;
    bool power_requested;
    int requested_power;
    int progress_base;
    bool progress_direct;
    double progress_started_at;
    char *cmd_text;
    agent_status status;
    char *out;
    size_t out_len;
    size_t out_cap;
    ds4_web *web;
    bool web_approval_pending;
    bool web_approval_answered;
    bool web_approval_result;
    char web_approval_message[256];
    char web_approval_error[160];
    bool queued_user_drain_pending;
    bool queued_user_drain_answered;
    char *queued_user_drain_text;
    bool datetime_context_injected;
    int last_system_prompt_reminder_at;
    char more_path[PATH_MAX];
    int more_next_line;
    bool more_bare;
    bool more_valid;
    agent_bash_job *bash_jobs;
    int next_bash_job_id;
    bool raw_mode_needs_restore;
} agent_worker;

static unsigned agent_next_prefill_label(void);

typedef struct agent_tail_capture {
    char *buf;
    size_t cap;
    size_t start;
    size_t len;
    size_t total;
} agent_tail_capture;

typedef enum {
    AGENT_MD_PENDING_NONE,
    AGENT_MD_PENDING_STAR,
    AGENT_MD_PENDING_BACKTICK,
} agent_markdown_pending;

typedef struct agent_syntax agent_syntax;

typedef struct {
    ds4_engine *engine;
    agent_worker *worker;
    bool format_thinking;
    bool format_markdown;
    bool in_think;
    bool color_open;
    bool use_color;
    bool last_output_newline;
    bool wrote_visible_output;
    bool md_bold;
    bool md_italic;
    bool md_inline_code;
    bool md_code_block;
    bool md_fence_info;
    bool md_code_line_start;
    bool md_code_in_ml_comment;
    bool md_syntax_silent;
    bool md_syntax_has_highlight;
    agent_markdown_pending md_pending;
    size_t md_pending_len;
    const agent_syntax *md_syntax;
    char md_fence_lang[32];
    size_t md_fence_lang_len;
    const char *md_code_line_prefix;
    const char *md_code_line_prefix_color;
    bool md_code_highlight_upto;
    char *md_code_line;
    size_t md_code_line_len;
    size_t md_code_line_cap;
    char pending[16];
    size_t pending_len;
    char utf8_pending[4];
    size_t utf8_pending_len;
    size_t utf8_pending_need;
    agent_tail_capture *capture;
} agent_token_renderer;

typedef struct {
    char *name;
    char *value;
    bool is_string;
} agent_tool_arg;

typedef struct {
    char *name;
    agent_tool_arg *args;
    int argc;
    int argcap;
} agent_tool_call;

typedef struct {
    agent_tool_call *v;
    int len;
    int cap;
} agent_tool_calls;

typedef enum {
    AGENT_TOOL_SYNTAX_DSML,
    AGENT_TOOL_SYNTAX_GLM,
} agent_tool_syntax;

typedef enum {
    AGENT_DSML_SEARCH,
    AGENT_DSML_STRUCTURAL,
    AGENT_DSML_PARAM_VALUE,
    AGENT_DSML_DONE,
    AGENT_DSML_ERROR,
} agent_dsml_state;

typedef struct {
    agent_tool_syntax syntax;
    agent_dsml_state state;
    char search_tail[64];
    size_t search_len;
    char *raw;
    size_t raw_len;
    size_t raw_cap;
    size_t parse_pos;
    agent_tool_call current;
    char *param_name;
    bool param_is_string;
    size_t param_value_start;
    bool param_close_prefix;
    bool glm_after_call;
    agent_tool_calls calls;
    char error[160];
} agent_dsml_parser;

typedef enum {
    AGENT_TOOL_PARAM_NORMAL,
    AGENT_TOOL_PARAM_PATH,
    AGENT_TOOL_PARAM_OFFSET,
    AGENT_TOOL_PARAM_CONTENT,
    AGENT_TOOL_PARAM_DIFF_OLD,
    AGENT_TOOL_PARAM_DIFF_NEW,
    AGENT_TOOL_PARAM_BASH_COMMAND,
} agent_tool_param_kind;

typedef struct {
    bool active;
    bool tool_announced;
    bool param_active;
    bool at_line_start;
    bool last_output_newline;
    agent_tool_param_kind param_kind;
    char tool_name[64];
    char param_name[64];
    char param_end_tail[64];
    size_t param_end_len;
    bool read_style;
    bool read_prefix_rendered;
    bool read_line_rendered;
    char read_path[512];
    char read_start[32];
    char read_max[32];
    char read_whole[8];
    char tool_path[512];
    bool code_param_active;
} agent_tool_visualizer;

typedef struct {
    char tail[32];
    size_t len;
} agent_dsml_marker_detector;

typedef struct {
    agent_token_renderer *renderer;
    agent_dsml_parser *parser;
    agent_tool_syntax syntax;
    agent_tool_visualizer viz;
    bool in_think;
    bool dsml_active;
    bool dsml_ignored;
    bool replay;
    char pending[16];
    size_t pending_len;
    char dsml_start_tail[64];
    size_t dsml_start_len;
    agent_dsml_marker_detector plain_dsml;
    agent_dsml_marker_detector think_dsml;
    bool dsml_in_think;
    bool dsml_in_think_reported;
    bool post_think_gap;
    bool tool_preflight_error;
    char tool_preflight_error_msg[256];
} agent_stream_renderer;

typedef struct {
    bool active;
    bool done;
} agent_edit_upto_forcer;

static volatile sig_atomic_t agent_sigint;
static agent_worker *agent_completion_worker;

static void worker_apply_pending_power(agent_worker *w);
static void agent_trace(agent_worker *w, const char *fmt, ...);
static void agent_trace_text(agent_worker *w, const char *label,
                             const char *text, size_t len);
static bool agent_stream_wants_greedy_sampling(const agent_stream_renderer *sr);
static void agent_publish_system_status(agent_worker *w, const char *msg);
static int agent_web_confirm(void *privdata, const char *message,
                             char *err, size_t err_len);
static void agent_web_log(void *privdata, const char *message);
static bool agent_preflight_edit_old(agent_worker *w, const agent_tool_call *call,
                                     char *err, size_t err_len);
static int agent_worker_sync_tokens(agent_worker *w, const ds4_tokens *tokens,
                                    bool publish_progress,
                                    char *err, size_t err_len);
static int agent_read_default_lines(agent_worker *w);

static agent_tool_syntax agent_tool_syntax_for_engine(ds4_engine *engine) {
    return ds4_engine_is_glm_dsa(engine) ? AGENT_TOOL_SYNTAX_GLM
                                         : AGENT_TOOL_SYNTAX_DSML;
}

static bool agent_tool_syntax_assistant_turn_uses_eos(agent_tool_syntax syntax) {
    return syntax != AGENT_TOOL_SYNTAX_GLM;
}

static void agent_worker_append_assistant_turn_end(agent_worker *w) {
    if (agent_tool_syntax_assistant_turn_uses_eos(
            agent_tool_syntax_for_engine(w->engine)))
        ds4_tokens_push(&w->transcript, ds4_token_eos(w->engine));
}

static int agent_worker_effective_ctx_size(const agent_worker *w) {
    int ctx = 0;
    if (w && w->session) ctx = ds4_session_ctx(w->session);
    if (ctx <= 0 && w && w->cfg) ctx = w->cfg->gen.ctx_size;
    return ctx;
}

/* ============================================================================
 * Small Utilities And Command-Line Parsing
 * ============================================================================
 */

static void agent_sigint_handler(int sig) {
    (void)sig;
    agent_sigint = 1;
}

static void *xmalloc(size_t n) {
    void *p = malloc(n ? n : 1);
    if (!p) {
        perror("ds4-agent: malloc");
        exit(1);
    }
    return p;
}

static char *xstrdup(const char *s) {
    if (!s) s = "";
    size_t n = strlen(s);
    char *p = xmalloc(n + 1);
    memcpy(p, s, n + 1);
    return p;
}

static char *xstrndup(const char *s, size_t n) {
    char *p = xmalloc(n + 1);
    memcpy(p, s, n);
    p[n] = '\0';
    return p;
}

static void *xrealloc(void *ptr, size_t n) {
    void *p = realloc(ptr, n ? n : 1);
    if (!p) {
        perror("ds4-agent: realloc");
        exit(1);
    }
    return p;
}

static void write_all(int fd, const char *p, size_t n) {
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

typedef struct {
    char *ptr;
    size_t len;
    size_t cap;
} agent_input_buf;

static void agent_input_buf_append(agent_input_buf *b, const char *s, size_t n) {
    if (!n) return;
    if (b->len + n + 1 > b->cap) {
        size_t cap = b->cap ? b->cap * 2 : 4096;
        while (cap < b->len + n + 1) cap *= 2;
        b->ptr = xrealloc(b->ptr, cap);
        b->cap = cap;
    }
    memcpy(b->ptr + b->len, s, n);
    b->len += n;
    b->ptr[b->len] = '\0';
}

static char *agent_input_buf_take(agent_input_buf *b) {
    if (!b->ptr) return xstrdup("");
    char *p = b->ptr;
    memset(b, 0, sizeof(*b));
    return p;
}

static void agent_input_buf_free(agent_input_buf *b) {
    free(b->ptr);
    memset(b, 0, sizeof(*b));
}

static int parse_int(const char *s, const char *opt) {
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (s[0] == '\0' || *end != '\0' || v <= 0 || v > INT32_MAX) {
        fprintf(stderr, "ds4-agent: invalid value for %s: %s\n", opt, s);
        exit(2);
    }
    return (int)v;
}

static int parse_nonnegative_int(const char *s, const char *opt) {
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (s[0] == '\0' || *end != '\0' || v < 0 || v > INT32_MAX) {
        fprintf(stderr, "ds4-agent: invalid value for %s: %s\n", opt, s);
        exit(2);
    }
    return (int)v;
}

static bool parse_power_percent(const char *arg, int *out) {
    char *end = NULL;
    long v = strtol(arg, &end, 10);
    if (!arg[0] || *end != '\0' || v < 1 || v > 100) return false;
    *out = (int)v;
    return true;
}

static bool agent_slash_command_with_args(const char *cmd, const char *name) {
    size_t len = strlen(name);
    return !strncmp(cmd, name, len) &&
           (cmd[len] == '\0' || isspace((unsigned char)cmd[len]));
}

static bool agent_slash_command_known(const char *cmd) {
    return !strcmp(cmd, "/help") ||
           !strcmp(cmd, "/save") ||
           !strcmp(cmd, "/compact") ||
           !strcmp(cmd, "/list") ||
           !strcmp(cmd, "/quit") ||
           !strcmp(cmd, "/exit") ||
           !strcmp(cmd, "/new") ||
           agent_slash_command_with_args(cmd, "/power") ||
           agent_slash_command_with_args(cmd, "/switch") ||
           agent_slash_command_with_args(cmd, "/del") ||
           agent_slash_command_with_args(cmd, "/strip") ||
           agent_slash_command_with_args(cmd, "/history");
}

static uint64_t parse_u64(const char *s, const char *opt) {
    char *end = NULL;
    unsigned long long v = strtoull(s, &end, 10);
    if (s[0] == '\0' || *end != '\0' || v == 0) {
        fprintf(stderr, "ds4-agent: invalid value for %s: %s\n", opt, s);
        exit(2);
    }
    return (uint64_t)v;
}

static float parse_float_range(const char *s, const char *opt, float min, float max) {
    char *end = NULL;
    float v = strtof(s, &end);
    if (s[0] == '\0' || *end != '\0' || !isfinite(v) || v < min || v > max) {
        fprintf(stderr, "ds4-agent: invalid value for %s: %s\n", opt, s);
        exit(2);
    }
    return v;
}

static ds4_backend parse_backend(const char *s) {
    if (!strcmp(s, "metal")) return DS4_BACKEND_METAL;
    if (!strcmp(s, "cuda")) return DS4_BACKEND_CUDA;
    if (!strcmp(s, "cpu")) return DS4_BACKEND_CPU;
    fprintf(stderr, "ds4-agent: invalid backend: %s\n", s);
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

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1.0e-9;
}

static void usage(FILE *fp, const char *topic) {
    ds4_help_print(fp, DS4_HELP_AGENT, topic);
}

static const char *need_arg(int *i, int argc, char **argv, const char *opt) {
    if (*i + 1 >= argc) {
        fprintf(stderr, "ds4-agent: missing value for %s\n", opt);
        exit(2);
    }
    return argv[++(*i)];
}

static agent_config parse_options(int argc, char **argv) {
    agent_config c = {
        .engine = {
            .model_path = "ds4flash.gguf",
            .backend = default_backend(),
            .mtp_draft_tokens = 1,
            .mtp_margin = 3.0f,
        },
        .gen = {
            .system = "You are a helpful coding assistant running inside ds4-agent.",
            .n_predict = 50000,
            .ctx_size = 100000,
            .temperature = DS4_DEFAULT_TEMPERATURE,
            .top_p = DS4_DEFAULT_TOP_P,
            .min_p = DS4_DEFAULT_MIN_P,
            .think_mode = DS4_THINK_HIGH,
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
        char dist_parse_err[256] = {0};
        ds4_dist_cli_parse_result dist_parse =
            ds4_dist_parse_cli_arg(arg,
                                   &i,
                                   argc,
                                   argv,
                                   &c.engine.distributed,
                                   dist_parse_err,
                                   sizeof(dist_parse_err));
        if (dist_parse == DS4_DIST_CLI_ERROR) {
            fprintf(stderr,
                    "ds4-agent: %s\n",
                    dist_parse_err[0] ? dist_parse_err : "invalid distributed option");
            exit(2);
        }
        if (dist_parse == DS4_DIST_CLI_MATCHED) continue;

        if (!strcmp(arg, "-p") || !strcmp(arg, "--prompt")) {
            c.gen.prompt = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--non-interactive")) {
            c.non_interactive = true;
        } else if (!strcmp(arg, "--raw") || !strcmp(arg, "--raw-prompt")) {
            c.gen.raw_prompt = true;
        } else if (!strcmp(arg, "-sys") || !strcmp(arg, "--system")) {
            c.gen.system = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--trace")) {
            c.gen.trace_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "-m") || !strcmp(arg, "--model")) {
            c.engine.model_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--mtp")) {
            c.engine.mtp_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--mtp-draft")) {
            c.engine.mtp_draft_tokens = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--mtp-margin")) {
            c.engine.mtp_margin = parse_float_range(need_arg(&i, argc, argv, arg), arg, 0.0f, 1000.0f);
        } else if (!strcmp(arg, "--glm-mtp")) {
            c.engine.glm_mtp = true;
        } else if (!strcmp(arg, "--glm-mtp-timing")) {
            c.engine.glm_mtp = true;
            c.engine.glm_mtp_timing = true;
        } else if (!strcmp(arg, "--dspark")) {
            c.engine.dspark = true;
        } else if (!strcmp(arg, "--dspark-confidence")) {
            c.engine.dspark = true;
            c.engine.dspark_confidence_threshold =
                parse_float_range(need_arg(&i, argc, argv, arg), arg, 0.0f, 1.0f);
            c.engine.dspark_confidence_threshold_set = true;
        } else if (!strcmp(arg, "--dspark-strict")) {
            c.engine.dspark = true;
            c.engine.dspark_strict = true;
        } else if (!strcmp(arg, "-c") || !strcmp(arg, "--ctx")) {
            c.gen.ctx_size = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "-n") || !strcmp(arg, "--tokens")) {
            c.gen.n_predict = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--temp")) {
            c.gen.temperature = parse_float_range(need_arg(&i, argc, argv, arg), arg, 0.0f, 100.0f);
            c.gen.temperature_set = true;
        } else if (!strcmp(arg, "--top-p")) {
            c.gen.top_p = parse_float_range(need_arg(&i, argc, argv, arg), arg, 0.0f, 1.0f);
            c.gen.top_p_set = true;
        } else if (!strcmp(arg, "--min-p")) {
            c.gen.min_p = parse_float_range(need_arg(&i, argc, argv, arg), arg, 0.0f, 1.0f);
            c.gen.min_p_set = true;
        } else if (!strcmp(arg, "--seed")) {
            c.gen.seed = parse_u64(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--think")) {
            c.gen.think_mode = DS4_THINK_HIGH;
        } else if (!strcmp(arg, "--think-max")) {
            c.gen.think_mode = DS4_THINK_MAX;
        } else if (!strcmp(arg, "--nothink")) {
            c.gen.think_mode = DS4_THINK_NONE;
        } else if (!strcmp(arg, "--backend")) {
            c.engine.backend = parse_backend(need_arg(&i, argc, argv, arg));
        } else if (!strcmp(arg, "--metal")) {
            c.engine.backend = DS4_BACKEND_METAL;
        } else if (!strcmp(arg, "--cuda")) {
            c.engine.backend = DS4_BACKEND_CUDA;
        } else if (!strcmp(arg, "--gpu-vram")) {
            c.gpu_vram_arg = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--gpu-devices")) {
            c.gpu_devices_arg = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--cuda-tensor-parallel")) {
            c.engine.cuda_tensor_parallel = true;
        } else if (!strcmp(arg, "--cpu")) {
            c.engine.backend = DS4_BACKEND_CPU;
        } else if (!strcmp(arg, "-t") || !strcmp(arg, "--threads")) {
            c.engine.n_threads = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--chdir")) {
            c.chdir_path = need_arg(&i, argc, argv, arg);
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
                        "ds4-agent: --ssd-streaming-cache-experts must be a positive count or <number>GB\n");
                exit(2);
            }
            c.engine.ssd_streaming_cache_experts = experts;
            c.engine.ssd_streaming_cache_bytes = bytes;
        } else if (!strcmp(arg, "--ssd-streaming-full-layers")) {
            int v = parse_nonnegative_int(need_arg(&i, argc, argv, arg), arg);
            c.engine.ssd_streaming_full_layers = (uint32_t)v;
            c.engine.ssd_streaming_full_layers_set = true;
        } else if (!strcmp(arg, "--ssd-streaming-preload-experts")) {
            int v = parse_int(need_arg(&i, argc, argv, arg), arg);
            if (v <= 0) {
                fprintf(stderr, "ds4-agent: --ssd-streaming-preload-experts must be positive\n");
                exit(2);
            }
            c.engine.ssd_streaming_preload_experts = (uint32_t)v;
        } else if (!strcmp(arg, "--simulate-used-memory")) {
            if (!ds4_parse_gib_arg(need_arg(&i, argc, argv, arg),
                                   &c.engine.simulate_used_memory_bytes)) {
                fprintf(stderr,
                        "ds4-agent: --simulate-used-memory must be a positive GiB value, e.g. 64GB\n");
                exit(2);
            }
        } else if (!strcmp(arg, "--prefill-chunk")) {
            int v = parse_int(need_arg(&i, argc, argv, arg), arg);
            if (v <= 0) {
                fprintf(stderr, "ds4-agent: --prefill-chunk must be positive\n");
                exit(2);
            }
            c.engine.prefill_chunk = (uint32_t)v;
        } else if (!strcmp(arg, "--power")) {
            c.engine.power_percent = parse_int(need_arg(&i, argc, argv, arg), arg);
            if (c.engine.power_percent < 1 || c.engine.power_percent > 100) {
                fprintf(stderr, "ds4-agent: --power must be between 1 and 100\n");
                exit(2);
            }
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
            fprintf(stderr, "ds4-agent: unknown option: %s\n", arg);
            usage(stderr, NULL);
            exit(2);
        }
    }

    if (c.engine.directional_steering_file && !steering_scale_set)
        c.engine.directional_steering_ffn = 1.0f;
    char dist_err[256];
    if (ds4_dist_prepare_engine_options(&c.engine.distributed,
                                        &c.engine,
                                        dist_err,
                                        sizeof(dist_err)) != 0) {
        fprintf(stderr, "ds4-agent: %s\n", dist_err);
        exit(2);
    }
    if (c.engine.distributed.role == DS4_DISTRIBUTED_WORKER) {
        fprintf(stderr, "ds4-agent: --role worker is a serving mode; start workers with ./ds4\n");
        exit(2);
    }
    if (c.gen.raw_prompt && (!c.non_interactive || !c.gen.prompt)) {
        fprintf(stderr,
                "ds4-agent: --raw-prompt is only supported with --non-interactive -p\n");
        exit(2);
    }
    return c;
}

static void agent_apply_model_sampling_defaults(
        ds4_engine               *engine,
        agent_generation_options *gen) {
    if (!engine || !gen || !ds4_engine_is_glm_dsa(engine)) return;

    if (!gen->temperature_set) gen->temperature = 1.0f;
    if (!gen->top_p_set) gen->top_p = 0.95f;
    if (!gen->min_p_set) gen->min_p = 0.0f;
}

static ds4_think_mode effective_think_mode(const agent_config *cfg) {
    return ds4_think_mode_for_context(cfg->gen.think_mode, cfg->gen.ctx_size);
}

/* ============================================================================
 * System Prompt Rendering And Worker Output Queues
 * ============================================================================
 */

static const char agent_tools_prompt_intro[] =
    "You are a coding agent running in a local workspace. Use tools for local file and system work. "
    "Avoid printing large file contents or large code blocks as answers; create or edit files with tools, "
    "then summarize results briefly.\n\n"
    "## Tools\n\n"
    "You have access to native DSML tools. Invoke tools by writing exactly this shape:\n\n"
    "<｜DSML｜tool_calls>\n"
    "<｜DSML｜invoke name=\"$TOOL_NAME\">\n"
    "<｜DSML｜parameter name=\"$PARAMETER_NAME\" string=\"true|false\">$PARAMETER_VALUE</｜DSML｜parameter>\n"
    "</｜DSML｜invoke>\n"
    "</｜DSML｜tool_calls>\n\n"
    "Tool calls are not allowed inside <think></think>; finish thinking before emitting DSML.\n\n"
    "String parameters use raw text and string=\"true\". Numbers and booleans use JSON text and string=\"false\".\n\n"
    "Read defaults to a context-sized bounded chunk, not the whole file. "
    "For first looks at large files, prefer read with explicit max_lines around 80-160; "
    "if read says more lines are available, call more with count=<lines> to read the next chunk. "
    "The read result also reports continue_offset=N, which is the next start_line if you need to jump manually. "
    "If the user explicitly asks you to read a complete file into context, call read with whole=true. "
    "A whole-file read may fail if the result would not fit the current context; then explain that and use chunks.\n\n";

#define AGENT_EDIT_TARGET_RULE \
    "When editing files, state the target filename before the edit; for the edit tool, put path first."

static const char agent_tools_prompt_edit_line[] =
    "## Editing files\n\n"
    AGENT_EDIT_TARGET_RULE "\n"
    "Use write for new files or deliberate whole-file replacement. Use edit with path, old, and new for changes. "
    "The old text must match exactly once in the current file; otherwise edit fails for safety.\n"
    "For large replacements, prefer anchored old text: write the first lines, then [upto], then the final lines. "
    "The tool replaces everything from the head through the tail. If the head or tail is ambiguous, the edit fails.\n"
    "After [upto], always write unique final lines before closing old; never close old immediately after [upto].\n"
    "Do not use a generic tail anchor like:\n"
    "- BigNum bignum_add(BigNum *a, BigNum *b) {\n"
    "- [upto]\n"
    "- }\n"
    "because the closing brace may match many functions. Instead include final lines that are unique near that function, "
    "for example its last calculation and return line before the brace.\n"
    "Example anchored edit:\n"
    "<｜DSML｜tool_calls>\n"
    "<｜DSML｜invoke name=\"edit\">\n"
    "<｜DSML｜parameter name=\"path\" string=\"true\">/tmp/example.c</｜DSML｜parameter>\n"
    "<｜DSML｜parameter name=\"old\" string=\"true\">static int parse(void) {\n"
    "    int ok = 0;\n"
    "[upto]\n"
    "    return ok;\n"
    "}</｜DSML｜parameter>\n"
    "<｜DSML｜parameter name=\"new\" string=\"true\">static int parse(void) {\n"
    "    return parse_impl();\n"
    "}</｜DSML｜parameter>\n"
    "</｜DSML｜invoke>\n"
    "</｜DSML｜tool_calls>\n"
    "To insert text, use edit with old set to an exact unique anchor and new set to that anchor plus the added text.\n"
    "Use read raw=true only when you need plain file text without line numbers or read annotations.\n\n";

static const char agent_tools_prompt_after_edit[] =
    "For long-running bash commands, pass refresh_sec. If a bash job is still running, use "
    "bash_status to check it early or bash_stop to terminate it.\n\n"
    "Use google_search to find web pages. Use visit_page to read a known URL with a visible browser. "
    "The first web call may ask the user for permission to start Chrome.\n\n"
    "### Available Tool Schemas\n\n"
    "{\n"
    "  \"type\": \"function\",\n"
    "  \"function\": {\n"
    "    \"name\": \"google_search\",\n"
    "    \"description\": \"Search Google in a visible browser and return compact Markdown links.\",\n"
    "    \"parameters\": {\n"
    "      \"type\": \"object\",\n"
    "      \"properties\": {\n"
    "        \"query\": {\"type\": \"string\"}\n"
    "      },\n"
    "      \"required\": [\"query\"]\n"
    "    }\n"
    "  }\n"
    "}\n\n"
    "{\n"
    "  \"type\": \"function\",\n"
    "  \"function\": {\n"
    "    \"name\": \"visit_page\",\n"
    "    \"description\": \"Open a URL in a visible browser and return rendered page Markdown.\",\n"
    "    \"parameters\": {\n"
    "      \"type\": \"object\",\n"
    "      \"properties\": {\n"
    "        \"url\": {\"type\": \"string\"}\n"
    "      },\n"
    "      \"required\": [\"url\"]\n"
    "    }\n"
    "  }\n"
    "}\n\n"
    "{\n"
    "  \"type\": \"function\",\n"
    "  \"function\": {\n"
    "    \"name\": \"bash\",\n"
    "    \"description\": \"Run a shell command.\",\n"
    "    \"parameters\": {\n"
    "      \"type\": \"object\",\n"
    "      \"properties\": {\n"
    "        \"command\": {\"type\": \"string\"},\n"
    "        \"timeout_sec\": {\"type\": \"number\"},\n"
    "        \"refresh_sec\": {\"type\": \"number\"}\n"
    "      },\n"
    "      \"required\": [\"command\"]\n"
    "    }\n"
    "  }\n"
    "}\n\n"
    "{\n"
    "  \"type\": \"function\",\n"
    "  \"function\": {\n"
    "    \"name\": \"bash_status\",\n"
    "    \"description\": \"Report current status and new output for a bash job.\",\n"
    "    \"parameters\": {\n"
    "      \"type\": \"object\",\n"
    "      \"properties\": {\n"
    "        \"job\": {\"type\": \"number\"},\n"
    "        \"pid\": {\"type\": \"number\"},\n"
    "        \"refresh_sec\": {\"type\": \"number\"}\n"
    "      },\n"
    "      \"required\": [\"job\"]\n"
    "    }\n"
    "  }\n"
    "}\n\n"
    "{\n"
    "  \"type\": \"function\",\n"
    "  \"function\": {\n"
    "    \"name\": \"bash_stop\",\n"
    "    \"description\": \"Terminate a running bash job and report its final output.\",\n"
    "    \"parameters\": {\n"
    "      \"type\": \"object\",\n"
    "      \"properties\": {\n"
    "        \"job\": {\"type\": \"number\"},\n"
    "        \"pid\": {\"type\": \"number\"},\n"
    "        \"refresh_sec\": {\"type\": \"number\"}\n"
    "      },\n"
    "      \"required\": [\"job\"]\n"
    "    }\n"
    "  }\n"
    "}\n\n"
    "{\n"
    "  \"type\": \"function\",\n"
    "  \"function\": {\n"
    "    \"name\": \"read\",\n"
    "    \"description\": \"Read a text file or a range of lines.\",\n"
    "    \"parameters\": {\n"
    "      \"type\": \"object\",\n"
    "      \"properties\": {\n"
    "        \"path\": {\"type\": \"string\"},\n"
    "        \"start_line\": {\"type\": \"number\"},\n"
    "        \"max_lines\": {\"type\": \"number\"},\n"
    "        \"whole\": {\"type\": \"boolean\"},\n"
    "        \"raw\": {\"type\": \"boolean\"}\n"
    "      },\n"
    "      \"required\": [\"path\"]\n"
    "    }\n"
    "  }\n"
    "}\n\n"
    "{\n"
    "  \"type\": \"function\",\n"
    "  \"function\": {\n"
    "    \"name\": \"more\",\n"
    "    \"description\": \"Continue the previous read-like output.\",\n"
    "    \"parameters\": {\n"
    "      \"type\": \"object\",\n"
    "      \"properties\": {\n"
    "        \"count\": {\"type\": \"number\"}\n"
    "      }\n"
    "    }\n"
    "  }\n"
    "}\n\n"
    "{\n"
    "  \"type\": \"function\",\n"
    "  \"function\": {\n"
    "    \"name\": \"write\",\n"
    "    \"description\": \"Create or overwrite a text file.\",\n"
    "    \"parameters\": {\n"
    "      \"type\": \"object\",\n"
    "      \"properties\": {\n"
    "        \"path\": {\"type\": \"string\"},\n"
    "        \"content\": {\"type\": \"string\"}\n"
    "      },\n"
    "      \"required\": [\"path\", \"content\"]\n"
    "    }\n"
    "  }\n"
    "}\n\n"
    "{\n"
    "  \"type\": \"function\",\n"
    "  \"function\": {\n"
    "    \"name\": \"edit\",\n"
    "    \"description\": \"Replace exactly one old text match; old may contain [upto] between unique head and tail anchors.\",\n"
    "    \"parameters\": {\n"
    "      \"type\": \"object\",\n"
    "      \"properties\": {\n"
    "        \"path\": {\"type\": \"string\"},\n"
    "        \"old\": {\"type\": \"string\"},\n"
    "        \"new\": {\"type\": \"string\"}\n"
    "      },\n"
    "      \"required\": [\"path\", \"old\", \"new\"]\n"
    "    }\n"
    "  }\n"
    "}\n\n"
    "{\n"
    "  \"type\": \"function\",\n"
    "  \"function\": {\n"
    "    \"name\": \"search\",\n"
    "    \"description\": \"Search files and return compact edit-friendly matches.\",\n"
    "    \"parameters\": {\n"
    "      \"type\": \"object\",\n"
    "      \"properties\": {\n"
    "        \"query\": {\"type\": \"string\"},\n"
    "        \"path\": {\"type\": \"string\"},\n"
    "        \"mode\": {\"type\": \"string\"},\n"
    "        \"glob\": {\"type\": \"string\"},\n"
    "        \"context\": {\"type\": \"number\"},\n"
    "        \"max_results\": {\"type\": \"number\"},\n"
    "        \"case_sensitive\": {\"type\": \"boolean\"}\n"
    "      },\n"
    "      \"required\": [\"query\"]\n"
    "    }\n"
    "  }\n"
    "}\n\n"
    "{\n"
    "  \"type\": \"function\",\n"
    "  \"function\": {\n"
    "    \"name\": \"list\",\n"
    "    \"description\": \"List one directory compactly.\",\n"
    "    \"parameters\": {\n"
    "      \"type\": \"object\",\n"
    "      \"properties\": {\n"
    "        \"path\": {\"type\": \"string\"}\n"
    "      },\n"
    "      \"required\": [\"path\"]\n"
    "    }\n"
    "  }\n"
    "}\n"
    "\n"
    "# Rules\n\n"
    "- Always use strict syntax for DSML tool stanzas.\n"
    "- This system runs on local inference of a few hundred tokens/s of prefill, "
    "and a few tens of tokens/s decoding speed. Use read/search to get the "
    "anchors you need, then use anchored edit to avoid having to "
    "retype large text.\n"
    "- Write code that is reliable and works well; always have a mental model of "
    "what is going on in complex parts of the code.\n"
    "- Work in a way that preserves the current system configuration integrity, "
    "unless explicitly asked otherwise by the user.\n";

static char *agent_build_dsml_tools_prompt(void) {
    const char *edit = agent_tools_prompt_edit_line;
    size_t a = strlen(agent_tools_prompt_intro);
    size_t b = strlen(edit);
    size_t c = strlen(agent_tools_prompt_after_edit);
    char *out = xmalloc(a + b + c + 1);
    memcpy(out, agent_tools_prompt_intro, a);
    memcpy(out + a, edit, b);
    memcpy(out + a + b, agent_tools_prompt_after_edit, c + 1);
    return out;
}

static const char agent_glm_tools_prompt_intro[] =
    "You are a coding agent running in a local workspace. Use tools for local file and system work. "
    "Avoid printing large file contents or large code blocks as answers; create or edit files with tools, "
    "then summarize results briefly.\n\n"
    "# Tools\n\n"
    "You may call one or more functions to assist with the user query.\n\n"
    "You are provided with function signatures within <tools></tools> XML tags:\n"
    "<tools>\n";

static const char agent_glm_tools_prompt_after_schemas[] =
    "</tools>\n\n"
    "For a function call, output the function name and arguments within exactly this XML format:\n"
    "<tool_call>{function-name}<arg_key>{arg-key-1}</arg_key><arg_value>{arg-value-1}</arg_value>"
    "<arg_key>{arg-key-2}</arg_key><arg_value>{arg-value-2}</arg_value>...</tool_call>\n\n"
    "Tool calls are not allowed inside <think></think>; finish thinking before emitting <tool_call>.\n\n"
    "# Rules\n\n"
    "- Use strict native GLM tool-call syntax with <tool_call>, <arg_key>, and <arg_value> tags.\n"
    "- read path alone returns a context-sized bounded chunk, not the whole file; for first looks at large files, prefer max_lines around 80-160.\n"
    "- If read says more lines are available, call more with count=<lines> to read the next chunk.\n"
    "- Use whole=true only when the user explicitly asks for the complete file contents or when bounded chunks are insufficient for the task; add raw=true only when line numbers would corrupt the payload.\n"
    "- " AGENT_EDIT_TARGET_RULE "\n"
    "- Use edit with exact old text and replacement new text; old may contain one [upto] marker between unique anchors.\n"
    "- For long bash jobs, pass refresh_sec and then poll with bash_status or stop with bash_stop.\n"
    "- Preserve the current system configuration unless the user explicitly asks otherwise.\n";

static const char agent_glm_tool_schemas[] =
    "{\"type\":\"function\",\"function\":{\"name\":\"google_search\",\"description\":\"Search web pages.\",\"parameters\":{\"type\":\"object\",\"properties\":{\"query\":{\"type\":\"string\"}},\"required\":[\"query\"]}}}\n"
    "{\"type\":\"function\",\"function\":{\"name\":\"visit_page\",\"description\":\"Read a URL in browser.\",\"parameters\":{\"type\":\"object\",\"properties\":{\"url\":{\"type\":\"string\"}},\"required\":[\"url\"]}}}\n"
    "{\"type\":\"function\",\"function\":{\"name\":\"bash\",\"description\":\"Run a shell command.\",\"parameters\":{\"type\":\"object\",\"properties\":{\"command\":{\"type\":\"string\"},\"timeout_sec\":{\"type\":\"number\"},\"refresh_sec\":{\"type\":\"number\"}},\"required\":[\"command\"]}}}\n"
    "{\"type\":\"function\",\"function\":{\"name\":\"bash_status\",\"description\":\"Check a bash job.\",\"parameters\":{\"type\":\"object\",\"properties\":{\"job\":{\"type\":\"number\"},\"pid\":{\"type\":\"number\"},\"refresh_sec\":{\"type\":\"number\"}},\"required\":[\"job\"]}}}\n"
    "{\"type\":\"function\",\"function\":{\"name\":\"bash_stop\",\"description\":\"Stop a bash job.\",\"parameters\":{\"type\":\"object\",\"properties\":{\"job\":{\"type\":\"number\"},\"pid\":{\"type\":\"number\"},\"refresh_sec\":{\"type\":\"number\"}},\"required\":[\"job\"]}}}\n"
    "{\"type\":\"function\",\"function\":{\"name\":\"read\",\"description\":\"Read a text file/range.\",\"parameters\":{\"type\":\"object\",\"properties\":{\"path\":{\"type\":\"string\"},\"start_line\":{\"type\":\"number\"},\"max_lines\":{\"type\":\"number\"},\"whole\":{\"type\":\"boolean\"},\"raw\":{\"type\":\"boolean\"}},\"required\":[\"path\"]}}}\n"
    "{\"type\":\"function\",\"function\":{\"name\":\"more\",\"description\":\"Continue previous read-like output.\",\"parameters\":{\"type\":\"object\",\"properties\":{\"count\":{\"type\":\"number\"}}}}}\n"
    "{\"type\":\"function\",\"function\":{\"name\":\"write\",\"description\":\"Create or overwrite a file.\",\"parameters\":{\"type\":\"object\",\"properties\":{\"path\":{\"type\":\"string\"},\"content\":{\"type\":\"string\"}},\"required\":[\"path\",\"content\"]}}}\n"
    "{\"type\":\"function\",\"function\":{\"name\":\"edit\",\"description\":\"Replace one exact old text match; old may use [upto].\",\"parameters\":{\"type\":\"object\",\"properties\":{\"path\":{\"type\":\"string\"},\"old\":{\"type\":\"string\"},\"new\":{\"type\":\"string\"}},\"required\":[\"path\",\"old\",\"new\"]}}}\n"
    "{\"type\":\"function\",\"function\":{\"name\":\"search\",\"description\":\"Search files.\",\"parameters\":{\"type\":\"object\",\"properties\":{\"query\":{\"type\":\"string\"},\"path\":{\"type\":\"string\"},\"mode\":{\"type\":\"string\"},\"glob\":{\"type\":\"string\"},\"context\":{\"type\":\"number\"},\"max_results\":{\"type\":\"number\"},\"case_sensitive\":{\"type\":\"boolean\"}},\"required\":[\"query\"]}}}\n"
    "{\"type\":\"function\",\"function\":{\"name\":\"list\",\"description\":\"List one directory.\",\"parameters\":{\"type\":\"object\",\"properties\":{\"path\":{\"type\":\"string\"}},\"required\":[\"path\"]}}}\n";

static char *agent_build_glm_tools_prompt(void) {
    size_t schemas_len = strlen(agent_glm_tool_schemas);
    const char *schemas = agent_glm_tool_schemas;
    size_t a = strlen(agent_glm_tools_prompt_intro);
    size_t b = schemas_len;
    size_t c = strlen(agent_glm_tools_prompt_after_schemas);
    char *out = xmalloc(a + b + c + 1);
    memcpy(out, agent_glm_tools_prompt_intro, a);
    memcpy(out + a, schemas, b);
    memcpy(out + a + b, agent_glm_tools_prompt_after_schemas, c + 1);
    return out;
}

static char *agent_build_tools_prompt(ds4_engine *engine) {
    if (agent_tool_syntax_for_engine(engine) == AGENT_TOOL_SYNTAX_GLM)
        return agent_build_glm_tools_prompt();
    return agent_build_dsml_tools_prompt();
}

static const char agent_dsml_syntax_reminder[] =
    "DSML syntax reminder:\n"
    "<｜DSML｜tool_calls>\n"
    "<｜DSML｜invoke name=\"$TOOL_NAME\">\n"
    "<｜DSML｜parameter name=\"$PARAMETER_NAME\" string=\"true|false\">$PARAMETER_VALUE</｜DSML｜parameter>\n"
    "</｜DSML｜invoke>\n"
    "</｜DSML｜tool_calls>\n";

static const char agent_glm_syntax_reminder[] =
    "GLM tool-call syntax reminder:\n"
    "<tool_call>$TOOL_NAME<arg_key>$PARAMETER_NAME</arg_key>"
    "<arg_value>$PARAMETER_VALUE</arg_value></tool_call>\n";

#define AGENT_SYSTEM_PROMPT_REMINDER_TOKENS 50000

static char *agent_build_system_prompt_reminder(ds4_engine *engine) {
    char *tools = agent_build_tools_prompt(engine);
    const char *start = "\n\n[System prompt reminder follows.]\n";
    const char *end = "[End system prompt reminder.]\n\n";
    const size_t len = strlen(start) + strlen(tools) + strlen(end) + 1;
    char *out = xmalloc(len);
    snprintf(out, len, "%s%s%s", start, tools, end);
    free(tools);
    return out;
}

static void agent_append_system_prompt(ds4_engine *engine, ds4_tokens *tokens,
                                       const char *extra) {
    /* The built-in tool prompt is trusted DS4 control text.  Tokenize it like a
     * rendered chat prompt so the literal ｜DSML｜ markers in the examples become
     * the model's dedicated DSML token.  Do not apply that tokenizer to user
     * supplied -sys text: arbitrary user text containing <｜User｜>, <think>, or
     * ｜DSML｜ must remain plain content, not control tokens. */
    char *tools_prompt = agent_build_tools_prompt(engine);
    if (agent_tool_syntax_for_engine(engine) == AGENT_TOOL_SYNTAX_GLM)
        ds4_chat_append_message(engine, tokens, "system", tools_prompt);
    else
        ds4_tokenize_rendered_chat(engine, tools_prompt, tokens);
    free(tools_prompt);

    if (!extra || !extra[0]) return;
    size_t n = strlen(extra);
    char *plain = xmalloc(n + 3);
    memcpy(plain, "\n\n", 2);
    memcpy(plain + 2, extra, n + 1);
    ds4_chat_append_message(engine, tokens, "system", plain);
    free(plain);
}

static void agent_worker_note_system_prompt_seen(agent_worker *w) {
    w->last_system_prompt_reminder_at = w->transcript.len;
}

static void agent_worker_maybe_append_datetime_context(agent_worker *w) {
    if (w->datetime_context_injected) return;

    time_t now = time(NULL);
    struct tm tm;
    localtime_r(&now, &tm);

    char when[128];
    if (strftime(when, sizeof(when), "%Y-%m-%d %H:%M:%S %Z", &tm) == 0)
        snprintf(when, sizeof(when), "%lld", (long long)now);

    char msg[256];
    snprintf(msg, sizeof(msg),
             "Current local date and time at session start: %s. "
             "Use this only when date or time matters.", when);
    ds4_chat_append_message(w->engine, &w->transcript, "system", msg);
    agent_trace_text(w, "datetime-context", msg, strlen(msg));
    w->datetime_context_injected = true;
}

static void agent_worker_maybe_append_system_prompt_reminder(agent_worker *w) {
    if (w->last_system_prompt_reminder_at <= 0) {
        agent_worker_note_system_prompt_seen(w);
        return;
    }
    if (w->transcript.len - w->last_system_prompt_reminder_at <
        AGENT_SYSTEM_PROMPT_REMINDER_TOKENS) {
        return;
    }

    char *reminder = agent_build_system_prompt_reminder(w->engine);
    agent_publish_system_status(w, "Re-injecting system prompt reminder...");
    agent_trace(w, "system prompt reminder injected at transcript=%d",
                w->transcript.len);
    if (agent_tool_syntax_for_engine(w->engine) == AGENT_TOOL_SYNTAX_GLM) {
        ds4_chat_append_message(w->engine, &w->transcript, "system", reminder);
    } else {
        ds4_tokenize_rendered_chat(w->engine, reminder, &w->transcript);
    }
    free(reminder);

    if (w->cfg->gen.system && w->cfg->gen.system[0]) {
        ds4_chat_append_message(w->engine, &w->transcript, "system",
                                w->cfg->gen.system);
    }
    agent_worker_note_system_prompt_seen(w);
}

/* Wake the UI thread after changing worker-visible state.  The byte in
 * wake_fd is level-triggered with wake_pending so bursts of sampled tokens do
 * not flood the pipe. */
static void agent_wake_locked(agent_worker *w) {
    if (w->wake_pending) return;
    w->wake_pending = true;
    char c = 'x';
    ssize_t wr = write(w->wake_fd[1], &c, 1);
    (void)wr;
}

/* Queue rendered output for the UI thread.  The worker never writes directly
 * to the terminal, which keeps linenoise redraws serialized in one place. */
static void agent_publish(agent_worker *w, const char *s, size_t n) {
    if (!n) return;
    pthread_mutex_lock(&w->mu);
    if (w->out_len + n + 1 > w->out_cap) {
        size_t cap = w->out_cap ? w->out_cap * 2 : 4096;
        while (cap < w->out_len + n + 1) cap *= 2;
        char *p = realloc(w->out, cap);
        if (!p) {
            pthread_mutex_unlock(&w->mu);
            return;
        }
        w->out = p;
        w->out_cap = cap;
    }
    memcpy(w->out + w->out_len, s, n);
    w->out_len += n;
    w->out[w->out_len] = '\0';
    agent_wake_locked(w);
    pthread_mutex_unlock(&w->mu);
}

static void agent_publishf(agent_worker *w, const char *fmt, ...) {
    char stack[1024];
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(stack, sizeof(stack), fmt, ap);
    va_end(ap);
    if (n <= 0) return;
    if ((size_t)n < sizeof(stack)) {
        agent_publish(w, stack, (size_t)n);
        return;
    }

    char *heap = xmalloc((size_t)n + 1);
    va_start(ap, fmt);
    vsnprintf(heap, (size_t)n + 1, fmt, ap);
    va_end(ap);
    agent_publish(w, heap, (size_t)n);
    free(heap);
}

static bool worker_is_idle(agent_worker *w);

static void agent_set_status(agent_worker *w, agent_worker_state state) {
    pthread_mutex_lock(&w->mu);
    w->status.state = state;
    if (state != AGENT_WORKER_PREFILL)
        w->status.prefill_tps = 0.0;
    if (state != AGENT_WORKER_GENERATING)
        w->status.greedy_sampling = false;
    agent_wake_locked(w);
    pthread_mutex_unlock(&w->mu);
}

static void agent_set_error(agent_worker *w, const char *msg) {
    pthread_mutex_lock(&w->mu);
    w->status.state = AGENT_WORKER_ERROR;
    w->status.prefill_tps = 0.0;
    w->status.greedy_sampling = false;
    snprintf(w->status.error, sizeof(w->status.error), "%s", msg ? msg : "unknown error");
    agent_wake_locked(w);
    pthread_mutex_unlock(&w->mu);
}

/* ============================================================================
 * Trace Logging
 * ============================================================================
 */

static void agent_trace_time(FILE *fp) {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    struct tm tm;
    localtime_r(&ts.tv_sec, &tm);
    fprintf(fp, "%04d-%02d-%02d %02d:%02d:%02d.%03ld",
            tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday,
            tm.tm_hour, tm.tm_min, tm.tm_sec, ts.tv_nsec / 1000000);
}

static void agent_trace(agent_worker *w, const char *fmt, ...) {
    if (!w || !w->trace) return;
    pthread_mutex_lock(&w->mu);
    agent_trace_time(w->trace);
    fputs(" ", w->trace);
    va_list ap;
    va_start(ap, fmt);
    vfprintf(w->trace, fmt, ap);
    va_end(ap);
    fputc('\n', w->trace);
    fflush(w->trace);
    pthread_mutex_unlock(&w->mu);
}

static void agent_trace_escaped(FILE *fp, const char *s, size_t n) {
    for (size_t i = 0; i < n; i++) {
        unsigned char c = (unsigned char)s[i];
        switch (c) {
        case '\\': fputs("\\\\", fp); break;
        case '\n': fputs("\\n", fp); break;
        case '\r': fputs("\\r", fp); break;
        case '\t': fputs("\\t", fp); break;
        case '"': fputs("\\\"", fp); break;
        default:
            if (c < 32 || c == 127) fprintf(fp, "\\x%02x", c);
            else fputc(c, fp);
            break;
        }
    }
}

static void agent_trace_token(agent_worker *w, int token, const char *text,
                              size_t text_len, int index) {
    if (!w || !w->trace) return;
    pthread_mutex_lock(&w->mu);
    agent_trace_time(w->trace);
    fprintf(w->trace, " token index=%d id=%d bytes=%zu text=\"",
            index, token, text_len);
    agent_trace_escaped(w->trace, text ? text : "", text_len);
    fputs("\" hex=", w->trace);
    for (size_t i = 0; i < text_len; i++)
        fprintf(w->trace, "%02x", (unsigned char)text[i]);
    fputc('\n', w->trace);
    fflush(w->trace);
    pthread_mutex_unlock(&w->mu);
}

static void agent_trace_tokens(agent_worker *w, const char *label,
                               const ds4_tokens *tokens, int start) {
    if (!w || !w->trace || !tokens) return;
    if (start < 0) start = 0;
    if (start > tokens->len) start = tokens->len;
    agent_trace(w, "tokens label=%s start=%d len=%d", label ? label : "",
                start, tokens->len);
    for (int i = start; i < tokens->len; i++) {
        size_t text_len = 0;
        char *text = ds4_token_text(w->engine, tokens->v[i], &text_len);
        agent_trace_token(w, tokens->v[i], text, text_len, i);
        free(text);
    }
}

static void agent_trace_text(agent_worker *w, const char *label,
                             const char *text, size_t len) {
    if (!w || !w->trace) return;
    pthread_mutex_lock(&w->mu);
    agent_trace_time(w->trace);
    fprintf(w->trace, " %s=\"", label ? label : "text");
    agent_trace_escaped(w->trace, text ? text : "", len);
    fputs("\"\n", w->trace);
    fflush(w->trace);
    pthread_mutex_unlock(&w->mu);
}

/* ============================================================================
 * DSML Tool-Call Parser
 * ============================================================================
 *
 * The model streams raw text tokens.  This parser recognizes completed DSML
 * tool stanzas and keeps a copy of the raw stanza for diagnostics.  It is
 * deliberately strict after the opening marker: typo recovery belongs to the
 * streaming detector so the actual tool parser stays small and predictable.
 */

static bool bytes_has_prefix(const char *p, size_t n, const char *prefix) {
    size_t plen = strlen(prefix);
    return n >= plen && memcmp(p, prefix, plen) == 0;
}

static bool bytes_is_partial_prefix(const char *p, size_t n, const char *prefix) {
    size_t plen = strlen(prefix);
    return n < plen && memcmp(prefix, p, n) == 0;
}

static void agent_tool_call_free(agent_tool_call *c) {
    if (!c) return;
    free(c->name);
    for (int i = 0; i < c->argc; i++) {
        free(c->args[i].name);
        free(c->args[i].value);
    }
    free(c->args);
    memset(c, 0, sizeof(*c));
}

static void agent_tool_calls_free(agent_tool_calls *calls) {
    if (!calls) return;
    for (int i = 0; i < calls->len; i++) agent_tool_call_free(&calls->v[i]);
    free(calls->v);
    memset(calls, 0, sizeof(*calls));
}

static void agent_tool_call_add_arg(agent_tool_call *c, const char *name,
                                    const char *value, size_t value_len,
                                    bool is_string) {
    if (c->argc == c->argcap) {
        c->argcap = c->argcap ? c->argcap * 2 : 4;
        c->args = xrealloc(c->args, (size_t)c->argcap * sizeof(c->args[0]));
    }
    c->args[c->argc++] = (agent_tool_arg){
        .name = xstrdup(name),
        .value = xstrndup(value, value_len),
        .is_string = is_string,
    };
}

static void agent_tool_calls_push(agent_tool_calls *calls, agent_tool_call *call) {
    if (!call->name) return;
    if (calls->len == calls->cap) {
        calls->cap = calls->cap ? calls->cap * 2 : 2;
        calls->v = xrealloc(calls->v, (size_t)calls->cap * sizeof(calls->v[0]));
    }
    calls->v[calls->len++] = *call;
    memset(call, 0, sizeof(*call));
}

static const char *agent_tool_arg_value(const agent_tool_call *call, const char *name) {
    for (int i = 0; i < call->argc; i++) {
        if (call->args[i].name && !strcmp(call->args[i].name, name))
            return call->args[i].value ? call->args[i].value : "";
    }
    return NULL;
}

static void agent_dsml_parser_free(agent_dsml_parser *p) {
    if (!p) return;
    agent_tool_syntax syntax = p->syntax;
    free(p->raw);
    agent_tool_call_free(&p->current);
    free(p->param_name);
    agent_tool_calls_free(&p->calls);
    memset(p, 0, sizeof(*p));
    p->syntax = syntax;
}

static void agent_dsml_parser_reset(agent_dsml_parser *p) {
    agent_dsml_parser_free(p);
    p->state = AGENT_DSML_SEARCH;
}

static void agent_dsml_raw_append(agent_dsml_parser *p, const char *s, size_t n) {
    if (!n) return;
    if (p->raw_len + n + 1 > p->raw_cap) {
        size_t cap = p->raw_cap ? p->raw_cap * 2 : 512;
        while (cap < p->raw_len + n + 1) cap *= 2;
        p->raw = xrealloc(p->raw, cap);
        p->raw_cap = cap;
    }
    memcpy(p->raw + p->raw_len, s, n);
    p->raw_len += n;
    p->raw[p->raw_len] = '\0';
}

static char *agent_parse_attr(const char *tag, const char *name) {
    char pat[64];
    snprintf(pat, sizeof(pat), "%s=\"", name);
    const char *p = strstr(tag, pat);
    if (!p) return NULL;
    p += strlen(pat);
    const char *end = strchr(p, '"');
    if (!end) return NULL;
    return xstrndup(p, (size_t)(end - p));
}

static void agent_dsml_set_error(agent_dsml_parser *p, const char *msg) {
    p->state = AGENT_DSML_ERROR;
    snprintf(p->error, sizeof(p->error), "%s", msg);
}

static const char *agent_skip_ascii_space(const char *p, const char *end) {
    while (p < end && (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n')) p++;
    return p;
}

static void agent_trim_span(const char **p, const char **end) {
    *p = agent_skip_ascii_space(*p, *end);
    while (*end > *p) {
        char c = (*end)[-1];
        if (c != ' ' && c != '\t' && c != '\r' && c != '\n') break;
        (*end)--;
    }
}

static bool agent_dsml_open_tag_is(const char *tag, const char *name) {
    char prefix[64];
    snprintf(prefix, sizeof(prefix), "<｜DSML｜%s", name);
    size_t prefix_len = strlen(prefix);
    if (strncmp(tag, prefix, prefix_len) != 0) return false;
    char c = tag[prefix_len];
    return c == '>' || c == ' ' || c == '\t' || c == '\r' || c == '\n';
}

static bool agent_dsml_close_tag_at(const char *s, const char *name, size_t *tag_len) {
    char prefix[64];
    static const char dsml_bar[] = "｜";
    snprintf(prefix, sizeof(prefix), "</｜DSML｜%s", name);
    size_t prefix_len = strlen(prefix);
    if (strncmp(s, prefix, prefix_len) != 0) return false;
    const char *p = s + prefix_len;
    while (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n') p++;
    if (strncmp(p, dsml_bar, strlen(dsml_bar)) == 0) p += strlen(dsml_bar);
    while (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n') p++;
    if (*p != '>') return false;
    if (tag_len) *tag_len = (size_t)(p - s) + 1;
    return true;
}

/* Recognize a streamed parameter close tag prefix.  Full close detection is
 * handled by agent_dsml_close_tag_at(); this helper exists for online behavior:
 * terminal rendering must hide partial close tags without waiting for the whole
 * parameter to finish. */
static bool agent_dsml_parameter_close_tail(const char *tail, size_t len,
                                            bool *complete) {
    static const char prefix[] = "</｜DSML｜parameter";
    static const char dsml_bar[] = "｜";
    const size_t prefix_len = sizeof(prefix) - 1;
    const size_t bar_len = sizeof(dsml_bar) - 1;
    *complete = false;
    if (len <= prefix_len) return memcmp(prefix, tail, len) == 0;
    if (memcmp(prefix, tail, prefix_len) != 0) return false;
    size_t i = prefix_len;
    while (i < len && (tail[i] == ' ' || tail[i] == '\t' ||
                       tail[i] == '\r' || tail[i] == '\n')) i++;
    if (i < len && len - i <= bar_len) {
        if (memcmp(dsml_bar, tail + i, len - i) == 0) return true;
    }
    if (i + bar_len <= len && memcmp(tail + i, dsml_bar, bar_len) == 0)
        i += bar_len;
    for (; i < len; i++) {
        if (tail[i] == '>') {
            *complete = i == len - 1;
            return *complete;
        }
        if (tail[i] != ' ' && tail[i] != '\t' && tail[i] != '\r' && tail[i] != '\n')
            return false;
    }
    return true;
}

static bool agent_glm_arg_value_close_tail(const char *tail, size_t len,
                                           bool *complete) {
    static const char close[] = "</arg_value>";
    *complete = false;
    size_t close_len = sizeof(close) - 1;
    if (len <= close_len && memcmp(close, tail, len) == 0) {
        *complete = len == close_len;
        return true;
    }
    return false;
}

static bool agent_tool_value_close_tail(agent_tool_syntax syntax,
                                        const char *tail, size_t len,
                                        bool *complete) {
    if (syntax == AGENT_TOOL_SYNTAX_GLM)
        return agent_glm_arg_value_close_tail(tail, len, complete);
    return agent_dsml_parameter_close_tail(tail, len, complete);
}

static void agent_dsml_update_param_close_prefix(agent_dsml_parser *p) {
    p->param_close_prefix = false;
    if (p->state != AGENT_DSML_PARAM_VALUE || p->raw_len <= p->param_value_start)
        return;

    const char *value = p->raw + p->param_value_start;
    const char *end = p->raw + p->raw_len;
    const char *lt = end;
    while (lt > value) {
        lt--;
        if (*lt == '<') break;
    }
    if (lt < value || *lt != '<') return;

    size_t tail_len = (size_t)(end - lt);
    if (tail_len > 64) return;
    bool complete = false;
    p->param_close_prefix =
        agent_tool_value_close_tail(p->syntax, lt, tail_len, &complete) &&
        !complete;
}

/* Find a DSML closing tag while accepting the few harmless closing-tag variants
 * the model has been observed to emit.  Opening tags stay strict so accidental
 * prose does not become a tool call. */
static char *agent_dsml_find_close_tag(const char *s, const char *name, size_t *tag_len) {
    const char *p = s;
    while ((p = strstr(p, "</｜DSML｜")) != NULL) {
        if (agent_dsml_close_tag_at(p, name, tag_len)) return (char *)p;
        p++;
    }
    return NULL;
}

static bool agent_bytes_starts_with(const char *p, const char *end,
                                    const char *prefix) {
    size_t n = strlen(prefix);
    return (size_t)(end - p) >= n && memcmp(p, prefix, n) == 0;
}

static bool agent_bytes_partial_prefix_at(const char *p, const char *end,
                                          const char *prefix) {
    return bytes_is_partial_prefix(p, (size_t)(end - p), prefix);
}

static void agent_glm_tool_parse(agent_dsml_parser *p) {
    static const char start[] = "<tool_call>";
    static const char close[] = "</tool_call>";
    static const char arg_key[] = "<arg_key>";
    static const char arg_key_close[] = "</arg_key>";
    static const char arg_value[] = "<arg_value>";
    static const char arg_value_close[] = "</arg_value>";

    if (p->raw_len < sizeof(start) - 1 ||
        memcmp(p->raw, start, sizeof(start) - 1) != 0) {
        return;
    }

    while (p->state == AGENT_DSML_STRUCTURAL ||
           p->state == AGENT_DSML_PARAM_VALUE)
    {
        const char *raw = p->raw;
        const char *end = p->raw + p->raw_len;
        if (p->state == AGENT_DSML_PARAM_VALUE) {
            const char *value_end = strstr(raw + p->param_value_start, arg_value_close);
            if (!value_end) {
                const char *call_end = strstr(raw + p->param_value_start, close);
                if (call_end) agent_dsml_set_error(p, "unterminated <arg_value> in GLM tool call");
                return;
            }
            agent_tool_call_add_arg(&p->current, p->param_name ? p->param_name : "",
                                    raw + p->param_value_start,
                                    (size_t)(value_end - (raw + p->param_value_start)),
                                    true);
            free(p->param_name);
            p->param_name = NULL;
            p->param_close_prefix = false;
            p->parse_pos = (size_t)(value_end - raw) + sizeof(arg_value_close) - 1;
            p->state = AGENT_DSML_STRUCTURAL;
            continue;
        }

        while (p->parse_pos < p->raw_len &&
               (p->raw[p->parse_pos] == ' ' || p->raw[p->parse_pos] == '\t' ||
                p->raw[p->parse_pos] == '\r' || p->raw[p->parse_pos] == '\n'))
            p->parse_pos++;
        if (p->parse_pos >= p->raw_len) return;

        const char *cur = raw + p->parse_pos;
        if (p->glm_after_call) {
            if (agent_bytes_starts_with(cur, end, start)) {
                p->parse_pos += sizeof(start) - 1;
                p->glm_after_call = false;
                continue;
            }
            if (agent_bytes_partial_prefix_at(cur, end, start)) return;
            p->glm_after_call = false;
            p->state = AGENT_DSML_DONE;
            return;
        }

        if (!p->current.name) {
            const char *name_start = agent_skip_ascii_space(cur, end);
            const char *lt = memchr(name_start, '<', (size_t)(end - name_start));
            if (!lt) return;
            bool at_arg = agent_bytes_starts_with(lt, end, arg_key);
            bool at_close = agent_bytes_starts_with(lt, end, close);
            if (!at_arg && !at_close) {
                if (agent_bytes_partial_prefix_at(lt, end, arg_key) ||
                    agent_bytes_partial_prefix_at(lt, end, close))
                    return;
                agent_dsml_set_error(p, "expected <arg_key> or </tool_call> in GLM tool call");
                return;
            }
            const char *name_end = lt;
            agent_trim_span(&name_start, &name_end);
            if (name_start >= name_end) {
                agent_dsml_set_error(p, "GLM tool call without function name");
                return;
            }
            agent_tool_call_free(&p->current);
            p->current.name = xstrndup(name_start, (size_t)(name_end - name_start));
            p->parse_pos = (size_t)(lt - raw);
            cur = raw + p->parse_pos;
        }

        if (agent_bytes_starts_with(cur, end, close)) {
            p->parse_pos += sizeof(close) - 1;
            agent_tool_calls_push(&p->calls, &p->current);
            p->glm_after_call = true;
            continue;
        }
        if (agent_bytes_partial_prefix_at(cur, end, close)) return;

        if (!agent_bytes_starts_with(cur, end, arg_key)) {
            if (agent_bytes_partial_prefix_at(cur, end, arg_key)) return;
            agent_dsml_set_error(p, "expected <arg_key> in GLM tool call");
            return;
        }
        cur += sizeof(arg_key) - 1;
        const char *key_end_mut = strstr(cur, arg_key_close);
        if (!key_end_mut) {
            const char *call_end = strstr(cur, close);
            if (call_end) agent_dsml_set_error(p, "unterminated <arg_key> in GLM tool call");
            return;
        }
        const char *key_start = cur;
        const char *key_end = key_end_mut;
        agent_trim_span(&key_start, &key_end);
        if (key_start >= key_end) {
            agent_dsml_set_error(p, "empty <arg_key> in GLM tool call");
            return;
        }
        char *key = xstrndup(key_start, (size_t)(key_end - key_start));
        cur = key_end_mut + sizeof(arg_key_close) - 1;
        cur = agent_skip_ascii_space(cur, end);
        if (!agent_bytes_starts_with(cur, end, arg_value)) {
            if (agent_bytes_partial_prefix_at(cur, end, arg_value)) {
                free(key);
                return;
            }
            free(key);
            agent_dsml_set_error(p, "expected <arg_value> in GLM tool call");
            return;
        }
        cur += sizeof(arg_value) - 1;
        free(p->param_name);
        p->param_name = key;
        p->param_is_string = true;
        p->param_value_start = (size_t)(cur - raw);
        p->parse_pos = p->param_value_start;
        p->param_close_prefix = false;
        p->state = AGENT_DSML_PARAM_VALUE;
    }
}

static void agent_dsml_finish(agent_dsml_parser *p) {
    if (!p || p->state == AGENT_DSML_DONE || p->state == AGENT_DSML_ERROR)
        return;
    if (p->syntax != AGENT_TOOL_SYNTAX_GLM || !p->glm_after_call)
        return;

    while (p->parse_pos < p->raw_len &&
           (p->raw[p->parse_pos] == ' ' || p->raw[p->parse_pos] == '\t' ||
            p->raw[p->parse_pos] == '\r' || p->raw[p->parse_pos] == '\n'))
        p->parse_pos++;
    if (p->parse_pos >= p->raw_len) {
        p->glm_after_call = false;
        p->state = AGENT_DSML_DONE;
    }
}

/* Parse as much of the accumulated DSML buffer as possible.  The parser can be
 * called after every streamed byte: incomplete input leaves state unchanged
 * until enough bytes arrive, while malformed completed input switches to
 * AGENT_DSML_ERROR so the model gets a retryable tool error. */
static void agent_dsml_parse(agent_dsml_parser *p) {
    if (p->syntax == AGENT_TOOL_SYNTAX_GLM) {
        agent_glm_tool_parse(p);
        return;
    }

    while (p->state == AGENT_DSML_STRUCTURAL || p->state == AGENT_DSML_PARAM_VALUE) {
        if (p->state == AGENT_DSML_PARAM_VALUE) {
            size_t end_tag_len = 0;
            char *end = agent_dsml_find_close_tag(p->raw + p->param_value_start,
                                                  "parameter", &end_tag_len);
            if (!end) return;
            agent_tool_call_add_arg(&p->current, p->param_name ? p->param_name : "",
                                    p->raw + p->param_value_start,
                                    (size_t)(end - (p->raw + p->param_value_start)),
                                    p->param_is_string);
            p->param_close_prefix = false;
            free(p->param_name);
            p->param_name = NULL;
            p->parse_pos = (size_t)(end - p->raw) + end_tag_len;
            p->state = AGENT_DSML_STRUCTURAL;
            continue;
        }

        while (p->parse_pos < p->raw_len &&
               (p->raw[p->parse_pos] == ' ' || p->raw[p->parse_pos] == '\t' ||
                p->raw[p->parse_pos] == '\r' || p->raw[p->parse_pos] == '\n'))
            p->parse_pos++;
        if (p->parse_pos >= p->raw_len) return;

        size_t close_len = 0;
        if (agent_dsml_close_tag_at(p->raw + p->parse_pos, "tool_calls", &close_len)) {
            agent_tool_calls_push(&p->calls, &p->current);
            p->parse_pos += close_len;
            p->state = AGENT_DSML_DONE;
            return;
        }
        if (agent_dsml_close_tag_at(p->raw + p->parse_pos, "invoke", &close_len)) {
            agent_tool_calls_push(&p->calls, &p->current);
            p->parse_pos += close_len;
            continue;
        }

        char *tag_end = strchr(p->raw + p->parse_pos, '>');
        if (!tag_end) return;
        size_t tag_len = (size_t)(tag_end - (p->raw + p->parse_pos)) + 1;
        char *tag = xstrndup(p->raw + p->parse_pos, tag_len);

        if (agent_dsml_open_tag_is(tag, "invoke")) {
            agent_tool_call_free(&p->current);
            p->current.name = agent_parse_attr(tag, "name");
            if (!p->current.name) {
                free(tag);
                agent_dsml_set_error(p, "tool invoke without name");
                return;
            }
            p->parse_pos += tag_len;
        } else if (agent_dsml_open_tag_is(tag, "parameter")) {
            free(p->param_name);
            p->param_name = agent_parse_attr(tag, "name");
            char *is_string = agent_parse_attr(tag, "string");
            p->param_is_string = is_string && !strcmp(is_string, "true");
            free(is_string);
            if (!p->param_name) {
                free(tag);
                agent_dsml_set_error(p, "tool parameter without name");
                return;
            }
            p->parse_pos += tag_len;
            p->param_value_start = p->parse_pos;
            p->param_close_prefix = false;
            p->state = AGENT_DSML_PARAM_VALUE;
        } else {
            snprintf(p->error, sizeof(p->error), "unexpected DSML tag: %.*s",
                     (int)(tag_len > 80 ? 80 : tag_len), tag);
            free(tag);
            p->state = AGENT_DSML_ERROR;
            return;
        }
        free(tag);
    }
}

static void agent_dsml_start(agent_dsml_parser *p) {
    const char *start = p->syntax == AGENT_TOOL_SYNTAX_GLM ?
        "<tool_call>" : "<｜DSML｜tool_calls>";
    p->state = AGENT_DSML_STRUCTURAL;
    p->search_len = 0;
    agent_dsml_raw_append(p, start, strlen(start));
    p->parse_pos = strlen(start);
}

static void agent_dsml_feed(agent_dsml_parser *p, const char *s, size_t n) {
    const char *start = p->syntax == AGENT_TOOL_SYNTAX_GLM ?
        "<tool_call>" : "<｜DSML｜tool_calls>";
    const size_t start_len = strlen(start);
    if (p->state == AGENT_DSML_DONE || p->state == AGENT_DSML_ERROR) return;

    for (size_t i = 0; i < n; i++) {
        char c = s[i];
        if (p->state == AGENT_DSML_SEARCH) {
            if (p->search_len == sizeof(p->search_tail)) {
                memmove(p->search_tail, p->search_tail + 1, --p->search_len);
            }
            p->search_tail[p->search_len++] = c;
            if (p->search_len >= start_len &&
                memcmp(p->search_tail + p->search_len - start_len, start, start_len) == 0)
                agent_dsml_start(p);
            continue;
        }

        agent_dsml_raw_append(p, &c, 1);
        agent_dsml_parse(p);
        if (p->state == AGENT_DSML_PARAM_VALUE)
            agent_dsml_update_param_close_prefix(p);
        else
            p->param_close_prefix = false;
    }
}

/* ============================================================================
 * Assistant Markdown Rendering
 * ============================================================================
 *
 * This renderer handles only the cheap markdown cues that make terminal output
 * readable: **bold**, *italic*, inline code, and fenced code blocks.  It is a
 * streaming parser, so it buffers only ambiguous marker bytes long enough to
 * decide whether they are formatting or literal text.
 */

static void agent_tail_capture_append(agent_tail_capture *t,
                                      const char *s, size_t n) {
    if (!t || !n) return;
    if (!t->cap) return;
    if (!t->buf) t->buf = xmalloc(t->cap);
    t->total += n;

    if (n >= t->cap) {
        memcpy(t->buf, s + n - t->cap, t->cap);
        t->start = 0;
        t->len = t->cap;
        return;
    }

    if (t->len < t->cap) {
        size_t free_tail = t->cap - t->len;
        size_t first = n < free_tail ? n : free_tail;
        size_t pos = (t->start + t->len) % t->cap;
        size_t right = t->cap - pos;
        size_t chunk = first < right ? first : right;
        memcpy(t->buf + pos, s, chunk);
        if (first > chunk) memcpy(t->buf, s + chunk, first - chunk);
        t->len += first;
        s += first;
        n -= first;
    }

    while (n) {
        size_t pos = (t->start + t->len) % t->cap;
        size_t right = t->cap - pos;
        size_t chunk = n < right ? n : right;
        memcpy(t->buf + pos, s, chunk);
        t->start = (t->start + chunk) % t->cap;
        s += chunk;
        n -= chunk;
    }
}

static char *agent_tail_capture_take(agent_tail_capture *t, size_t *len) {
    size_t n = t ? t->len : 0;
    char *out = xmalloc(n + 1);
    if (n) {
        size_t right = t->cap - t->start;
        size_t first = n < right ? n : right;
        memcpy(out, t->buf + t->start, first);
        if (n > first) memcpy(out + first, t->buf, n - first);
    }
    out[n] = '\0';
    if (len) *len = n;
    free(t->buf);
    memset(t, 0, sizeof(*t));
    return out;
}

static void renderer_write(agent_token_renderer *r, const char *s, size_t n) {
    if (r->capture) agent_tail_capture_append(r->capture, s, n);
    else agent_publish(r->worker, s, n);
}

static void renderer_set_grey(agent_token_renderer *r) {
    if (r->use_color) renderer_write(r, "\x1b[38;5;245m", 11);
}

static void renderer_reset_color(agent_token_renderer *r) {
    if (r->use_color) renderer_write(r, "\x1b[0m", 4);
    r->color_open = false;
}

static size_t renderer_utf8_need(unsigned char c) {
    if (c < 0x80) return 1;
    if (c >= 0xc2 && c <= 0xdf) return 2;
    if (c >= 0xe0 && c <= 0xef) return 3;
    if (c >= 0xf0 && c <= 0xf4) return 4;
    return 1;
}

static bool renderer_has_text_attrs(agent_token_renderer *r) {
    return r->in_think || r->md_bold || r->md_italic ||
           r->md_inline_code || r->md_code_block;
}

static void renderer_set_text_attrs(agent_token_renderer *r) {
    if (!r->use_color) return;
    if (r->in_think) {
        renderer_set_grey(r);
        return;
    }
    if (r->md_code_block) {
        renderer_write(r, "\x1b[38;5;75m", 10);
        return;
    } else if (r->md_inline_code) {
        renderer_write(r, "\x1b[36m", 5);
    }
    if (r->md_bold) renderer_write(r, "\x1b[1m", 4);
    if (r->md_italic) renderer_write(r, "\x1b[3m", 4);
}

static void renderer_restore_text_attrs(agent_token_renderer *r) {
    if (!r->use_color || !r->color_open || !renderer_has_text_attrs(r)) return;
    renderer_set_text_attrs(r);
}

static void renderer_write_complete_char_raw(agent_token_renderer *r, const char *s, size_t n) {
    bool styled = r->use_color && renderer_has_text_attrs(r);
    if (styled && !r->color_open) {
        renderer_set_text_attrs(r);
        r->color_open = true;
    } else if (!styled && r->color_open) {
        renderer_reset_color(r);
    }
    renderer_write(r, s, n);
    if (n) r->wrote_visible_output = true;
    r->last_output_newline = n == 1 && s[0] == '\n';
}

static void renderer_flush_utf8(agent_token_renderer *r) {
    if (!r->utf8_pending_len) return;
    renderer_write_complete_char_raw(r, r->utf8_pending, r->utf8_pending_len);
    r->utf8_pending_len = 0;
    r->utf8_pending_need = 0;
}

static void renderer_write_char_raw(agent_token_renderer *r, char c) {
    unsigned char uc = (unsigned char)c;

    if (r->utf8_pending_len) {
        if ((uc & 0xc0) == 0x80 && r->utf8_pending_len < sizeof(r->utf8_pending)) {
            r->utf8_pending[r->utf8_pending_len++] = c;
            if (r->utf8_pending_len == r->utf8_pending_need) renderer_flush_utf8(r);
            return;
        }
        renderer_flush_utf8(r);
    }

    size_t need = renderer_utf8_need(uc);
    if (need == 1) {
        renderer_write_complete_char_raw(r, &c, 1);
        return;
    }
    r->utf8_pending[0] = c;
    r->utf8_pending_len = 1;
    r->utf8_pending_need = need;
}

static void renderer_write_plain_byte(agent_token_renderer *r, char c) {
    bool old_bold = r->md_bold;
    bool old_italic = r->md_italic;
    bool old_inline_code = r->md_inline_code;
    bool old_code_block = r->md_code_block;

    /* Code blocks are streamed immediately in plain text, then repainted with
     * syntax colors when a complete terminal-safe line is available.  Disable
     * markdown attributes only for this byte; renderer_write_char_raw() will
     * reset any tracked manual color once if needed. */
    r->md_bold = false;
    r->md_italic = false;
    r->md_inline_code = false;
    r->md_code_block = false;
    renderer_write_char_raw(r, c);
    r->md_bold = old_bold;
    r->md_italic = old_italic;
    r->md_inline_code = old_inline_code;
    r->md_code_block = old_code_block;
}

/* Poor man's code highlighter inspired by antirez/kilo: a tiny language table
 * plus one line-oriented tokenizer for comments, strings, numbers, and
 * separator-bounded keywords.  This is deliberately not a full parser; it is
 * only for making fenced Markdown code readable in the terminal. */
#define AGENT_HL_NORMAL 0
#define AGENT_HL_COMMENT 1
#define AGENT_HL_KEYWORD1 2
#define AGENT_HL_KEYWORD2 3
#define AGENT_HL_STRING 4
#define AGENT_HL_NUMBER 5

#define AGENT_SYNTAX_NUMBERS (1u<<0)
#define AGENT_SYNTAX_STRINGS (1u<<1)
#define AGENT_SYNTAX_BACKTICK_STRINGS (1u<<2)
#define AGENT_SYNTAX_CASE_INSENSITIVE (1u<<3)

struct agent_syntax {
    const char *name;
    const char *aliases;
    const char **keywords;
    const char *singleline_comments[3];
    const char *multiline_start;
    const char *multiline_end;
    unsigned flags;
};

static const char *agent_kw_generic[] = {
    "if","else","for","while","do","switch","case","default","break",
    "continue","return","try","catch","finally","throw","throws","class",
    "struct","enum","interface","trait","impl","fn","func","function",
    "def","lambda","let","var","const","static","public","private",
    "protected","import","include","from","export","package","module",
    "namespace","new","delete","async","await","yield","match","type",
    "true|","false|","null|","nil|","none|","None|","NULL|","void|",
    "int|","long|","float|","double|","char|","bool|","string|",
    "String|","usize|","isize|","u8|","u16|","u32|","u64|","i8|",
    "i16|","i32|","i64|",NULL
};

static const char *agent_kw_c[] = {
    "auto","break","case","continue","default","do","else","enum",
    "extern","for","goto","if","register","return","sizeof","static",
    "struct","switch","typedef","union","volatile","while",
    "alignas","alignof","and","and_eq","asm","bitand","bitor","class",
    "compl","constexpr","const_cast","decltype","delete","dynamic_cast",
    "explicit","export","false","friend","inline","mutable","namespace",
    "new","noexcept","not","not_eq","nullptr","operator","or","or_eq",
    "private","protected","public","reinterpret_cast","static_assert",
    "static_cast","template","this","thread_local","throw","true","try",
    "typeid","typename","virtual","xor","xor_eq",
    "NULL|","bool|","char|","const|","double|","float|","int|","long|",
    "short|","signed|","size_t|","ssize_t|","uint8_t|","uint16_t|",
    "uint32_t|","uint64_t|","unsigned|","void|",NULL
};

static const char *agent_kw_python[] = {
    "and","as","assert","async","await","break","case","class","continue",
    "def","del","elif","else","except","finally","for","from","global",
    "if","import","in","is","lambda","match","nonlocal","not","or","pass",
    "raise","return","try","while","with","yield",
    "False|","None|","True|","bool|","bytes|","dict|","float|","int|",
    "list|","object|","set|","str|","tuple|",NULL
};

static const char *agent_kw_js[] = {
    "async","await","break","case","catch","class","const","continue",
    "debugger","default","delete","do","else","export","extends",
    "finally","for","from","function","get","if","import","in",
    "instanceof","let","new","of","return","set","static","super",
    "switch","this","throw","try","typeof","var","void","while","with",
    "yield","abstract","as","declare","enum","implements","interface",
    "keyof","namespace","private","protected","public","readonly","type",
    "any|","boolean|","false|","never|","null|","number|","string|",
    "symbol|","true|","undefined|","unknown|","void|",NULL
};

static const char *agent_kw_java[] = {
    "abstract","assert","break","case","catch","class","const","continue",
    "default","do","else","enum","extends","final","finally","for","goto",
    "if","implements","import","instanceof","interface","native","new",
    "package","private","protected","public","return","static","strictfp",
    "super","switch","synchronized","this","throw","throws","transient",
    "try","volatile","while",
    "boolean|","byte|","char|","double|","false|","float|","int|","long|",
    "null|","short|","true|","void|",NULL
};

static const char *agent_kw_csharp[] = {
    "abstract","as","base","break","case","catch","checked","class","const",
    "continue","default","delegate","do","else","enum","event","explicit",
    "extern","finally","fixed","for","foreach","goto","if","implicit","in",
    "interface","internal","is","lock","namespace","new","operator","out",
    "override","params","private","protected","public","readonly","ref",
    "return","sealed","sizeof","stackalloc","static","struct","switch",
    "this","throw","try","typeof","unchecked","unsafe","using","virtual",
    "volatile","while","async","await","get","init","record","set","var",
    "bool|","byte|","char|","decimal|","double|","false|","float|","int|",
    "long|","null|","object|","sbyte|","short|","string|","true|","uint|",
    "ulong|","ushort|","void|",NULL
};

static const char *agent_kw_go[] = {
    "break","case","chan","const","continue","default","defer","else",
    "fallthrough","for","func","go","goto","if","import","interface",
    "map","package","range","return","select","struct","switch","type",
    "var","bool|","byte|","complex64|","complex128|","error|","false|",
    "float32|","float64|","int|","int8|","int16|","int32|","int64|",
    "nil|","rune|","string|","true|","uint|","uint8|","uint16|",
    "uint32|","uint64|","uintptr|",NULL
};

static const char *agent_kw_rust[] = {
    "as","async","await","break","const","continue","crate","dyn","else",
    "enum","extern","fn","for","if","impl","in","let","loop","match",
    "mod","move","mut","pub","ref","return","self","Self","static",
    "struct","super","trait","type","unsafe","use","where","while",
    "bool|","char|","false|","f32|","f64|","i8|","i16|","i32|","i64|",
    "i128|","isize|","str|","String|","true|","u8|","u16|","u32|",
    "u64|","u128|","usize|",NULL
};

static const char *agent_kw_shell[] = {
    "case","do","done","elif","else","esac","fi","for","function","if",
    "in","select","then","time","until","while","break","continue",
    "return","export","local","readonly","source","test","true|","false|",
    "echo|","printf|","cd|","pwd|","read|","set|","unset|","shift|",NULL
};

static const char *agent_kw_sql[] = {
    "add","alter","and","as","asc","between","by","case","check","column",
    "constraint","create","delete","desc","distinct","drop","else","end",
    "exists","foreign","from","group","having","in","index","insert",
    "into","is","join","key","left","like","limit","not","null","on",
    "or","order","outer","primary","references","right","select","set",
    "table","then","union","unique","update","values","view","where",
    "bigint|","boolean|","date|","decimal|","false|","int|","integer|",
    "numeric|","real|","text|","timestamp|","true|","varchar|",NULL
};

static const char *agent_kw_ruby[] = {
    "BEGIN","END","alias","and","begin","break","case","class","def",
    "defined?","do","else","elsif","end","ensure","for","if","in",
    "module","next","not","or","redo","rescue","retry","return","self",
    "super","then","undef","unless","until","when","while","yield",
    "false|","nil|","true|",NULL
};

static const char *agent_kw_php[] = {
    "abstract","and","array","as","break","callable","case","catch","class",
    "clone","const","continue","declare","default","die","do","echo","else",
    "elseif","empty","enddeclare","endfor","endforeach","endif","endswitch",
    "endwhile","eval","exit","extends","final","finally","fn","for",
    "foreach","function","global","goto","if","implements","include",
    "include_once","instanceof","insteadof","interface","isset","list",
    "match","namespace","new","or","print","private","protected","public",
    "readonly","require","require_once","return","static","switch","throw",
    "trait","try","unset","use","var","while","xor","bool|","false|",
    "float|","int|","null|","string|","true|","void|",NULL
};

static const char *agent_kw_swift[] = {
    "actor","as","associatedtype","async","await","break","case","catch",
    "class","continue","default","defer","do","else","enum","extension",
    "fallthrough","for","func","guard","if","import","in","init","inout",
    "is","let","nonisolated","operator","private","protocol","public",
    "repeat","return","self","Self","static","struct","subscript","super",
    "switch","throw","throws","try","typealias","var","where","while",
    "Any|","Bool|","Double|","false|","Float|","Int|","nil|","String|",
    "true|","Void|",NULL
};

static const char *agent_kw_kotlin[] = {
    "as","break","class","continue","do","else","false","for","fun","if",
    "in","interface","is","null","object","package","return","super",
    "this","throw","true","try","typealias","typeof","val","var","when",
    "while","actual","annotation","by","catch","companion","const",
    "constructor","crossinline","data","enum","expect","external","final",
    "finally","import","infix","init","inline","inner","internal","lateinit",
    "noinline","open","operator","out","override","private","protected",
    "public","reified","sealed","suspend","tailrec","vararg",
    "Any|","Boolean|","Byte|","Char|","Double|","Float|","Int|","Long|",
    "Short|","String|","Unit|",NULL
};

static const char *agent_kw_zig[] = {
    "addrspace","align","allowzero","and","anyframe","anytype","asm",
    "async","await","break","callconv","catch","comptime","const",
    "continue","defer","else","enum","errdefer","error","export","extern",
    "fn","for","if","inline","linksection","noalias","noinline","nosuspend",
    "opaque","or","orelse","packed","pub","resume","return","struct",
    "suspend","switch","test","threadlocal","try","union","unreachable",
    "usingnamespace","var","volatile","while",
    "bool|","false|","f32|","f64|","i32|","i64|","null|","true|","u8|",
    "u16|","u32|","u64|","usize|","void|",NULL
};

static const char *agent_kw_lua[] = {
    "and","break","do","else","elseif","end","false","for","function",
    "goto","if","in","local","nil","not","or","repeat","return","then",
    "true","until","while",NULL
};

static const char *agent_kw_html[] = {
    "a","body","button","div","doctype","form","h1","h2","h3","head",
    "html","input","label","li","link","main","meta","ol","option","p",
    "script","section","select","span","style","table","tbody","td","th",
    "thead","title","tr","ul","class|","href|","id|","name|","rel|",
    "src|","type|","value|",NULL
};

static const char *agent_kw_css[] = {
    "align-items","background","border","bottom","color","display","flex",
    "font","font-size","gap","grid","height","justify-content","left",
    "margin","max-width","min-width","padding","position","right","top",
    "transform","width","z-index","absolute|","auto|","block|","flex|",
    "grid|","hidden|","inline|","none|","relative|","solid|",NULL
};

static const agent_syntax agent_syntaxes[] = {
    {"generic", " text txt", agent_kw_generic, {"//","#",NULL}, "/*", "*/",
        AGENT_SYNTAX_NUMBERS | AGENT_SYNTAX_STRINGS | AGENT_SYNTAX_BACKTICK_STRINGS},
    {"c", " c h cpp c++ cc cxx hpp hxx objc objective-c", agent_kw_c, {"//",NULL,NULL}, "/*", "*/",
        AGENT_SYNTAX_NUMBERS | AGENT_SYNTAX_STRINGS},
    {"python", " py python py3", agent_kw_python, {"#",NULL,NULL}, NULL, NULL,
        AGENT_SYNTAX_NUMBERS | AGENT_SYNTAX_STRINGS},
    {"javascript", " js jsx javascript typescript ts tsx node mjs cjs", agent_kw_js, {"//",NULL,NULL}, "/*", "*/",
        AGENT_SYNTAX_NUMBERS | AGENT_SYNTAX_STRINGS | AGENT_SYNTAX_BACKTICK_STRINGS},
    {"java", " java", agent_kw_java, {"//",NULL,NULL}, "/*", "*/",
        AGENT_SYNTAX_NUMBERS | AGENT_SYNTAX_STRINGS},
    {"csharp", " cs c# csharp dotnet", agent_kw_csharp, {"//",NULL,NULL}, "/*", "*/",
        AGENT_SYNTAX_NUMBERS | AGENT_SYNTAX_STRINGS},
    {"go", " go golang", agent_kw_go, {"//",NULL,NULL}, "/*", "*/",
        AGENT_SYNTAX_NUMBERS | AGENT_SYNTAX_STRINGS | AGENT_SYNTAX_BACKTICK_STRINGS},
    {"rust", " rs rust", agent_kw_rust, {"//",NULL,NULL}, "/*", "*/",
        AGENT_SYNTAX_NUMBERS | AGENT_SYNTAX_STRINGS},
    {"shell", " sh bash zsh shell fish ksh", agent_kw_shell, {"#",NULL,NULL}, NULL, NULL,
        AGENT_SYNTAX_NUMBERS | AGENT_SYNTAX_STRINGS | AGENT_SYNTAX_BACKTICK_STRINGS},
    {"sql", " sql postgres mysql sqlite", agent_kw_sql, {"--",NULL,NULL}, "/*", "*/",
        AGENT_SYNTAX_NUMBERS | AGENT_SYNTAX_STRINGS | AGENT_SYNTAX_CASE_INSENSITIVE},
    {"ruby", " rb ruby", agent_kw_ruby, {"#",NULL,NULL}, NULL, NULL,
        AGENT_SYNTAX_NUMBERS | AGENT_SYNTAX_STRINGS},
    {"php", " php", agent_kw_php, {"//","#",NULL}, "/*", "*/",
        AGENT_SYNTAX_NUMBERS | AGENT_SYNTAX_STRINGS},
    {"swift", " swift", agent_kw_swift, {"//",NULL,NULL}, "/*", "*/",
        AGENT_SYNTAX_NUMBERS | AGENT_SYNTAX_STRINGS},
    {"kotlin", " kt kts kotlin", agent_kw_kotlin, {"//",NULL,NULL}, "/*", "*/",
        AGENT_SYNTAX_NUMBERS | AGENT_SYNTAX_STRINGS},
    {"zig", " zig", agent_kw_zig, {"//",NULL,NULL}, NULL, NULL,
        AGENT_SYNTAX_NUMBERS | AGENT_SYNTAX_STRINGS},
    {"lua", " lua", agent_kw_lua, {"--",NULL,NULL}, NULL, NULL,
        AGENT_SYNTAX_NUMBERS | AGENT_SYNTAX_STRINGS},
    {"html", " html htm xml svg", agent_kw_html, {NULL,NULL,NULL}, "<!--", "-->",
        AGENT_SYNTAX_NUMBERS | AGENT_SYNTAX_STRINGS},
    {"css", " css scss sass", agent_kw_css, {NULL,NULL,NULL}, "/*", "*/",
        AGENT_SYNTAX_NUMBERS | AGENT_SYNTAX_STRINGS},
    {"json", " json jsonc", NULL, {"//",NULL,NULL}, "/*", "*/",
        AGENT_SYNTAX_NUMBERS | AGENT_SYNTAX_STRINGS},
    {"yaml", " yaml yml toml ini", NULL, {"#",NULL,NULL}, NULL, NULL,
        AGENT_SYNTAX_NUMBERS | AGENT_SYNTAX_STRINGS},
    {"markdown", " md markdown", agent_kw_generic, {NULL,NULL,NULL}, "<!--", "-->",
        AGENT_SYNTAX_NUMBERS | AGENT_SYNTAX_STRINGS},
    {NULL, NULL, NULL, {NULL,NULL,NULL}, NULL, NULL, 0}
};

static bool agent_syntax_alias_match(const char *aliases, const char *lang) {
    if (!aliases || !lang || !lang[0]) return false;
    size_t llen = strlen(lang);
    const char *p = aliases;
    while (*p) {
        while (*p == ' ') p++;
        const char *start = p;
        while (*p && *p != ' ') p++;
        if ((size_t)(p - start) == llen && !strncasecmp(start, lang, llen))
            return true;
    }
    return false;
}

static const agent_syntax *agent_syntax_for_lang(const char *lang) {
    if (lang && lang[0]) {
        for (const agent_syntax *s = agent_syntaxes; s->name; s++) {
            if (!strcasecmp(s->name, lang) ||
                agent_syntax_alias_match(s->aliases, lang))
                return s;
        }
    }
    return &agent_syntaxes[0];
}

static const agent_syntax *agent_syntax_for_path(const char *path) {
    if (!path || !path[0]) return agent_syntax_for_lang(NULL);
    const char *base = strrchr(path, '/');
    base = base ? base + 1 : path;
    if (!strcasecmp(base, "Dockerfile")) return agent_syntax_for_lang("sh");
    if (!strcasecmp(base, "Makefile") || !strcasecmp(base, "makefile"))
        return agent_syntax_for_lang("sh");
    const char *dot = strrchr(base, '.');
    if (!dot || !dot[1]) return agent_syntax_for_lang(NULL);
    return agent_syntax_for_lang(dot + 1);
}

static bool agent_syntax_separator(char c) {
    unsigned char uc = (unsigned char)c;
    return c == '\0' || isspace(uc) || strchr(",.()+-/*=~%[]{}<>:;!&|^?", c) != NULL;
}

static const char *agent_syntax_line_comment(const agent_syntax *syn,
                                             const char *p) {
    if (!syn) return NULL;
    for (int i = 0; i < 3 && syn->singleline_comments[i]; i++) {
        const char *m = syn->singleline_comments[i];
        size_t mlen = strlen(m);
        if (mlen && !strncmp(p, m, mlen)) return m;
    }
    return NULL;
}

static int agent_syntax_color(int hl) {
    switch (hl) {
    case AGENT_HL_COMMENT: return 244;
    case AGENT_HL_KEYWORD1: return 214;
    case AGENT_HL_KEYWORD2: return 81;
    case AGENT_HL_STRING: return 150;
    case AGENT_HL_NUMBER: return 203;
    default: return 252;
    }
}

static void renderer_syntax_write(agent_token_renderer *r, int hl,
                                  const char *s, size_t n) {
    if (!n) return;
    if (hl != AGENT_HL_NORMAL) r->md_syntax_has_highlight = true;
    if (r->md_syntax_silent) return;
    if (r->use_color && hl != AGENT_HL_NORMAL) {
        char seq[32];
        snprintf(seq, sizeof(seq), "\x1b[38;5;%dm", agent_syntax_color(hl));
        renderer_write(r, seq, strlen(seq));
    }
    renderer_write(r, s, n);
    if (r->use_color && hl != AGENT_HL_NORMAL) renderer_write(r, "\x1b[0m", 4);
    r->wrote_visible_output = true;
    r->last_output_newline = false;
}

static void renderer_syntax_write_upto_marker(agent_token_renderer *r) {
    static const char marker[] = "[upto]";
    r->md_syntax_has_highlight = true;
    if (r->md_syntax_silent) return;
    if (r->use_color) {
        renderer_write(r, "\x1b[38;5;244m[", strlen("\x1b[38;5;244m["));
        renderer_write(r, "\x1b[1;38;5;177mupto",
                       strlen("\x1b[1;38;5;177mupto"));
        renderer_write(r, "\x1b[38;5;244m]\x1b[0m",
                       strlen("\x1b[38;5;244m]\x1b[0m"));
    } else {
        renderer_write(r, marker, sizeof(marker) - 1);
    }
    r->wrote_visible_output = true;
    r->last_output_newline = false;
}

static size_t agent_syntax_keyword_len(const char *kw, bool *secondary) {
    size_t len = strlen(kw);
    *secondary = len && kw[len - 1] == '|';
    return *secondary ? len - 1 : len;
}

static bool agent_syntax_match_keyword(const agent_syntax *syn,
                                       const char *p,
                                       const char *line_end,
                                       size_t *out_len,
                                       int *out_hl) {
    if (!syn || !syn->keywords) return false;
    for (int i = 0; syn->keywords[i]; i++) {
        bool secondary = false;
        size_t klen = agent_syntax_keyword_len(syn->keywords[i], &secondary);
        if ((size_t)(line_end - p) < klen) continue;
        bool match = (syn->flags & AGENT_SYNTAX_CASE_INSENSITIVE) ?
            !strncasecmp(p, syn->keywords[i], klen) :
            !strncmp(p, syn->keywords[i], klen);
        if (!match) continue;
        if (!agent_syntax_separator(p[klen])) continue;
        *out_len = klen;
        *out_hl = secondary ? AGENT_HL_KEYWORD2 : AGENT_HL_KEYWORD1;
        return true;
    }
    return false;
}

static bool agent_syntax_number_start(const char *p, const char *line,
                                      bool prev_sep, int prev_hl) {
    unsigned char c = (unsigned char)*p;
    if (isdigit(c) && (prev_sep || prev_hl == AGENT_HL_NUMBER)) return true;
    if (*p == '.' && p > line && prev_hl == AGENT_HL_NUMBER) return true;
    return false;
}

static size_t agent_syntax_number_len(const char *p, const char *line_end) {
    const char *q = p;
    while (q < line_end) {
        unsigned char c = (unsigned char)*q;
        if (isalnum(c) || *q == '_' || *q == '.' || *q == '+' || *q == '-') q++;
        else break;
    }
    return (size_t)(q - p);
}

static void renderer_syntax_emit_line(agent_token_renderer *r,
                                      const char *line, size_t len) {
    const agent_syntax *syn = r->md_syntax ? r->md_syntax : agent_syntax_for_lang(NULL);
    const char *p = line;
    const char *end = line + len;
    bool prev_sep = true;
    int prev_hl = AGENT_HL_NORMAL;
    int in_string = 0;

    while (p < end) {
        if (r->md_code_highlight_upto &&
            (size_t)(end - p) >= strlen("[upto]") &&
            !strncmp(p, "[upto]", strlen("[upto]")))
        {
            renderer_syntax_write_upto_marker(r);
            p += strlen("[upto]");
            prev_sep = true;
            prev_hl = AGENT_HL_NORMAL;
            continue;
        }

        if (r->md_code_in_ml_comment) {
            const char *mce = syn->multiline_end;
            if (mce && *mce) {
                size_t mlen = strlen(mce);
                const char *q = p;
                while (q < end && ((size_t)(end - q) < mlen ||
                       strncmp(q, mce, mlen))) q++;
                if (q < end) {
                    q += mlen;
                    renderer_syntax_write(r, AGENT_HL_COMMENT, p, (size_t)(q - p));
                    p = q;
                    r->md_code_in_ml_comment = false;
                    prev_sep = true;
                    prev_hl = AGENT_HL_COMMENT;
                    continue;
                }
            }
            renderer_syntax_write(r, AGENT_HL_COMMENT, p, (size_t)(end - p));
            return;
        }

        const char *scs = agent_syntax_line_comment(syn, p);
        if (!in_string && scs) {
            renderer_syntax_write(r, AGENT_HL_COMMENT, p, (size_t)(end - p));
            return;
        }

        if (!in_string && syn->multiline_start && syn->multiline_end &&
            !strncmp(p, syn->multiline_start, strlen(syn->multiline_start))) {
            size_t mlen = strlen(syn->multiline_start);
            const char *q = p + mlen;
            size_t elen = strlen(syn->multiline_end);
            while (q < end && ((size_t)(end - q) < elen ||
                   strncmp(q, syn->multiline_end, elen))) q++;
            if (q < end) q += elen;
            else r->md_code_in_ml_comment = true;
            renderer_syntax_write(r, AGENT_HL_COMMENT, p, (size_t)(q - p));
            p = q;
            prev_sep = false;
            prev_hl = AGENT_HL_COMMENT;
            continue;
        }

        if ((syn->flags & AGENT_SYNTAX_STRINGS) && in_string) {
            const char *q = p;
            while (q < end) {
                if (*q == '\\' && q + 1 < end) {
                    q += 2;
                    continue;
                }
                q++;
                if (q[-1] == in_string) {
                    in_string = 0;
                    break;
                }
            }
            renderer_syntax_write(r, AGENT_HL_STRING, p, (size_t)(q - p));
            p = q;
            prev_sep = false;
            prev_hl = AGENT_HL_STRING;
            continue;
        }

        if ((syn->flags & AGENT_SYNTAX_STRINGS) &&
            (*p == '"' || *p == '\'' ||
             ((syn->flags & AGENT_SYNTAX_BACKTICK_STRINGS) && *p == '`'))) {
            int quote = *p;
            const char *q = p + 1;
            while (q < end) {
                if (*q == '\\' && q + 1 < end) {
                    q += 2;
                    continue;
                }
                q++;
                if (q[-1] == quote) {
                    break;
                }
            }
            renderer_syntax_write(r, AGENT_HL_STRING, p, (size_t)(q - p));
            p = q;
            prev_sep = false;
            prev_hl = AGENT_HL_STRING;
            continue;
        }

        if ((syn->flags & AGENT_SYNTAX_NUMBERS) &&
            agent_syntax_number_start(p, line, prev_sep, prev_hl)) {
            size_t nlen = agent_syntax_number_len(p, end);
            renderer_syntax_write(r, AGENT_HL_NUMBER, p, nlen);
            p += nlen;
            prev_sep = false;
            prev_hl = AGENT_HL_NUMBER;
            continue;
        }

        if (prev_sep) {
            size_t klen = 0;
            int khl = AGENT_HL_NORMAL;
            if (agent_syntax_match_keyword(syn, p, end, &klen, &khl)) {
                renderer_syntax_write(r, khl, p, klen);
                p += klen;
                prev_sep = false;
                prev_hl = khl;
                continue;
            }
        }

        renderer_syntax_write(r, AGENT_HL_NORMAL, p, 1);
        prev_sep = agent_syntax_separator(*p);
        prev_hl = AGENT_HL_NORMAL;
        p++;
    }
}

static void renderer_code_line_append(agent_token_renderer *r,
                                      const char *s, size_t n) {
    if (!n) return;
    if (r->md_code_line_len + n + 1 > r->md_code_line_cap) {
        size_t cap = r->md_code_line_cap ? r->md_code_line_cap * 2 : 256;
        while (cap < r->md_code_line_len + n + 1) cap *= 2;
        r->md_code_line = xrealloc(r->md_code_line, cap);
        r->md_code_line_cap = cap;
    }
    memcpy(r->md_code_line + r->md_code_line_len, s, n);
    r->md_code_line_len += n;
    r->md_code_line[r->md_code_line_len] = '\0';
}

static int renderer_terminal_cols(void) {
    struct winsize ws;
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0 && ws.ws_col > 0)
        return ws.ws_col;
    return 80;
}

static bool renderer_code_line_can_repaint(agent_token_renderer *r) {
    if (!r->use_color || r->capture || r->md_code_line_len == 0) return false;
    int cols = renderer_terminal_cols();
    size_t prefix_len = r->md_code_line_prefix ?
                        strlen(r->md_code_line_prefix) : 0;
    if (cols <= 1 || prefix_len + r->md_code_line_len >= (size_t)cols)
        return false;
    for (size_t i = 0; i < r->md_code_line_len; i++) {
        unsigned char c = (unsigned char)r->md_code_line[i];
        if (c == '\t' || c == 0x1b || c >= 0x80 || (c < 0x20 && c != '\r'))
            return false;
    }
    return true;
}

static void renderer_code_write_line_prefix(agent_token_renderer *r) {
    if (!r->md_code_line_prefix) return;
    if (r->use_color && r->md_code_line_prefix_color)
        renderer_write(r, r->md_code_line_prefix_color,
                       strlen(r->md_code_line_prefix_color));
    renderer_write(r, r->md_code_line_prefix,
                   strlen(r->md_code_line_prefix));
    if (r->use_color && r->md_code_line_prefix_color)
        renderer_write(r, "\x1b[0m", 4);
    r->color_open = false;
}

/* Run the syntax highlighter in silent mode to learn whether the already
 * streamed line would change if repainted, while preserving the multiline
 * comment state until the caller decides whether repaint is safe. */
static bool renderer_code_scan_line(agent_token_renderer *r,
                                    bool *final_ml_comment) {
    bool old_silent = r->md_syntax_silent;
    bool old_highlight = r->md_syntax_has_highlight;
    bool old_ml_comment = r->md_code_in_ml_comment;

    r->md_syntax_silent = true;
    r->md_syntax_has_highlight = false;
    renderer_syntax_emit_line(r, r->md_code_line, r->md_code_line_len);
    bool changed = r->md_syntax_has_highlight;
    *final_ml_comment = r->md_code_in_ml_comment;

    r->md_code_in_ml_comment = old_ml_comment;
    r->md_syntax_silent = old_silent;
    r->md_syntax_has_highlight = old_highlight;
    return changed;
}

/* Code is shown as soon as bytes arrive.  At end-of-line we can cheaply
 * replace only that terminal row with syntax-highlighted text, but only for
 * simple one-row ASCII lines; long, tabbed, escaped, or UTF-8 lines are left
 * as streamed and only advance the highlighter state. */
static void renderer_code_emit_buffered_line(agent_token_renderer *r,
                                             bool with_newline) {
    bool final_ml_comment = r->md_code_in_ml_comment;
    bool changed = renderer_code_scan_line(r, &final_ml_comment);
    bool repaint = changed && renderer_code_line_can_repaint(r);
    if (repaint) {
        renderer_reset_color(r);
        renderer_write(r, "\r\x1b[0K", 5);
        renderer_code_write_line_prefix(r);
        renderer_syntax_emit_line(r, r->md_code_line, r->md_code_line_len);
    } else {
        r->md_code_in_ml_comment = final_ml_comment;
    }
    r->md_code_line_len = 0;
    if (with_newline) {
        renderer_write_plain_byte(r, '\n');
        r->wrote_visible_output = true;
        r->last_output_newline = true;
        r->md_code_line_start = true;
    }
}

static void renderer_code_byte(agent_token_renderer *r, char c) {
    if (c == '\n') {
        renderer_code_emit_buffered_line(r, true);
        return;
    }
    renderer_code_line_append(r, &c, 1);
    renderer_write_plain_byte(r, c);
    if (c != ' ' && c != '\t' && c != '\r') r->md_code_line_start = false;
}

static void renderer_code_emit_backtick_literals(agent_token_renderer *r,
                                                 size_t count) {
    for (size_t i = 0; i < count; i++) renderer_code_byte(r, '`');
}

static void renderer_code_begin(agent_token_renderer *r) {
    renderer_reset_color(r);
    r->md_code_block = true;
    r->md_inline_code = false;
    r->md_fence_info = true;
    r->md_code_line_start = true;
    r->md_code_in_ml_comment = false;
    r->md_syntax = agent_syntax_for_lang(NULL);
    r->md_fence_lang_len = 0;
    r->md_fence_lang[0] = '\0';
    r->md_code_line_prefix = NULL;
    r->md_code_line_prefix_color = NULL;
    r->md_code_highlight_upto = false;
    r->md_code_line_len = 0;
}

static void renderer_code_stream_begin(agent_token_renderer *r,
                                       const agent_syntax *syntax) {
    renderer_reset_color(r);
    r->md_code_block = true;
    r->md_inline_code = false;
    r->md_fence_info = false;
    r->md_code_line_start = true;
    r->md_code_in_ml_comment = false;
    r->md_syntax = syntax ? syntax : agent_syntax_for_lang(NULL);
    r->md_fence_lang_len = 0;
    r->md_fence_lang[0] = '\0';
    r->md_code_line_prefix = NULL;
    r->md_code_line_prefix_color = NULL;
    r->md_code_highlight_upto = false;
    r->md_code_line_len = 0;
}

static void renderer_code_stream_set_prefix(agent_token_renderer *r,
                                            const char *prefix,
                                            const char *color) {
    r->md_code_line_prefix = prefix;
    r->md_code_line_prefix_color = color;
}

static void renderer_code_stream_set_upto_marker(agent_token_renderer *r,
                                                 bool enabled) {
    r->md_code_highlight_upto = enabled;
}

static void renderer_code_end(agent_token_renderer *r) {
    bool only_space = true;
    for (size_t i = 0; i < r->md_code_line_len; i++) {
        if (r->md_code_line[i] != ' ' && r->md_code_line[i] != '\t' &&
            r->md_code_line[i] != '\r') {
            only_space = false;
            break;
        }
    }
    if (r->md_code_line_len && !only_space)
        renderer_code_emit_buffered_line(r, false);
    else
        r->md_code_line_len = 0;
    r->md_code_block = false;
    r->md_inline_code = false;
    r->md_fence_info = false;
    r->md_code_line_start = true;
    r->md_code_in_ml_comment = false;
    r->md_syntax = NULL;
    r->md_fence_lang_len = 0;
    r->md_fence_lang[0] = '\0';
    r->md_code_line_prefix = NULL;
    r->md_code_line_prefix_color = NULL;
}

/* Tiny streaming Markdown highlighter for assistant prose.  It deliberately
 * recognizes only delimiters that the model commonly emits in short answers:
 * **bold**, *italic*, `inline code`, ``inline code`` and fenced code blocks.
 * The state machine holds only possible delimiter bytes; once a byte is known
 * to be ordinary text it is sent to the raw UTF-8 writer above.  Tool
 * visualization and redirected output bypass this layer. */
static void renderer_markdown_clear_pending(agent_token_renderer *r) {
    r->md_pending = AGENT_MD_PENDING_NONE;
    r->md_pending_len = 0;
}

static void renderer_markdown_emit_pending_literals(agent_token_renderer *r) {
    char c;
    if (r->md_pending == AGENT_MD_PENDING_STAR) {
        c = '*';
    } else if (r->md_pending == AGENT_MD_PENDING_BACKTICK) {
        c = '`';
    } else {
        return;
    }
    size_t count = r->md_pending_len;
    renderer_markdown_clear_pending(r);
    if (r->md_code_block) {
        if (c == '`') renderer_code_emit_backtick_literals(r, count);
        else for (size_t i = 0; i < count; i++) renderer_code_byte(r, c);
        return;
    }
    for (size_t i = 0; i < count; i++) renderer_write_char_raw(r, c);
}

static void renderer_markdown_commit_backticks(agent_token_renderer *r) {
    size_t count = r->md_pending_len;
    renderer_markdown_clear_pending(r);
    if (count >= 3) {
        for (size_t i = 0; i < count; i++) renderer_write_plain_byte(r, '`');
        if (r->md_code_block) renderer_code_end(r);
        else renderer_code_begin(r);
        return;
    }
    if (r->md_code_block) {
        renderer_code_emit_backtick_literals(r, count);
        return;
    }
    /* Support both `code` and ``code``.  The latter is uncommon in model
     * replies, but accepting it costs nothing and avoids leaking delimiters. */
    r->md_inline_code = !r->md_inline_code;
}

static bool renderer_space_byte(char c) {
    return c == ' ' || c == '\t' || c == '\r' || c == '\n';
}

/* Consume one byte of markdown-aware assistant output.  Backticks and stars
 * are held in r->pending until the parser knows whether they form a marker;
 * all ordinary text is emitted with the current terminal attributes. */
static void renderer_markdown_feed(agent_token_renderer *r, char c) {
    if (r->md_fence_info) {
        if (c == '\n') {
            if (r->md_code_block) {
                r->md_fence_lang[r->md_fence_lang_len] = '\0';
                r->md_syntax = agent_syntax_for_lang(r->md_fence_lang);
            }
            renderer_write_plain_byte(r, '\n');
            r->md_fence_info = false;
        } else if (r->md_code_block) {
            unsigned char uc = (unsigned char)c;
            if (r->md_fence_lang_len + 1 < sizeof(r->md_fence_lang) &&
                (isalnum(uc) || c == '_' || c == '-' || c == '+' || c == '#'))
            {
                r->md_fence_lang[r->md_fence_lang_len++] = c;
            }
            renderer_write_plain_byte(r, c);
        }
        return;
    }

    if (r->md_pending == AGENT_MD_PENDING_BACKTICK) {
        if (c == '`') {
            r->md_pending_len++;
            return;
        }
        renderer_markdown_commit_backticks(r);
        renderer_markdown_feed(r, c);
        return;
    }

    if (r->md_pending == AGENT_MD_PENDING_STAR) {
        renderer_markdown_clear_pending(r);
        if (!r->md_inline_code && !r->md_code_block && c == '*') {
            r->md_bold = !r->md_bold;
            return;
        }
        if (!r->md_inline_code && !r->md_code_block &&
            (r->md_italic || !renderer_space_byte(c)))
        {
            r->md_italic = !r->md_italic;
            renderer_markdown_feed(r, c);
            return;
        }
        renderer_write_char_raw(r, '*');
        renderer_markdown_feed(r, c);
        return;
    }

    if (c == '`' && (!r->md_code_block || r->md_code_line_start)) {
        r->md_pending = AGENT_MD_PENDING_BACKTICK;
        r->md_pending_len = 1;
        return;
    }
    if (r->md_code_block) {
        renderer_code_byte(r, c);
        return;
    }
    if (!r->md_inline_code && !r->md_code_block && c == '*') {
        r->md_pending = AGENT_MD_PENDING_STAR;
        r->md_pending_len = 1;
        return;
    }
    renderer_write_char_raw(r, c);
}

static void renderer_markdown_finish(agent_token_renderer *r) {
    /* A closing code fence can be the final bytes of the assistant reply.  In
     * that case no following character arrives to force the pending backticks
     * through the normal streaming path, so commit a full fence here instead of
     * leaking the literal ``` marker to the terminal. */
    if (r->md_pending == AGENT_MD_PENDING_BACKTICK && r->md_pending_len >= 3)
        renderer_markdown_commit_backticks(r);
    else
        renderer_markdown_emit_pending_literals(r);
    if (r->md_code_block && r->md_code_line_len)
        renderer_code_emit_buffered_line(r, false);
    r->md_bold = false;
    r->md_italic = false;
    r->md_inline_code = false;
    r->md_code_block = false;
    r->md_fence_info = false;
    r->md_code_line_start = false;
    r->md_code_in_ml_comment = false;
    r->md_syntax = NULL;
    r->md_fence_lang_len = 0;
    r->md_fence_lang[0] = '\0';
    r->md_code_line_prefix = NULL;
    r->md_code_line_prefix_color = NULL;
    r->md_code_highlight_upto = false;
    free(r->md_code_line);
    r->md_code_line = NULL;
    r->md_code_line_len = 0;
    r->md_code_line_cap = 0;
}

static void renderer_write_char(agent_token_renderer *r, char c) {
    if (!r->format_markdown || r->in_think) {
        renderer_markdown_emit_pending_literals(r);
        renderer_write_char_raw(r, c);
        return;
    }
    renderer_markdown_feed(r, c);
}

/* Render assistant text while hiding <think> tags and dimming thinking text.
 * The function is also responsible for not prematurely emitting a partial
 * control tag split across model tokens. */
static void renderer_process(agent_token_renderer *r, const char *text, size_t len, bool finish) {
    const char *think_open = "<think>";
    const char *think_close = "</think>";
    size_t total = r->pending_len + len;
    char *buf = xmalloc(total ? total : 1);
    if (r->pending_len) memcpy(buf, r->pending, r->pending_len);
    if (len) memcpy(buf + r->pending_len, text, len);
    r->pending_len = 0;

    size_t i = 0;
    while (i < total) {
        const char *cur = buf + i;
        size_t rem = total - i;
        if (bytes_has_prefix(cur, rem, think_open)) {
            r->in_think = true;
            i += strlen(think_open);
            continue;
        }
        if (bytes_has_prefix(cur, rem, think_close)) {
            r->in_think = false;
            renderer_reset_color(r);
            if (!r->last_output_newline) renderer_write(r, "\n", 1);
            renderer_write(r, "\n", 1);
            r->last_output_newline = true;
            i += strlen(think_close);
            continue;
        }
        if (!finish && cur[0] == '<' &&
            (bytes_is_partial_prefix(cur, rem, think_open) ||
             bytes_is_partial_prefix(cur, rem, think_close)))
        {
            if (rem < sizeof(r->pending)) {
                memcpy(r->pending, cur, rem);
                r->pending_len = rem;
            }
            break;
        }
        renderer_write_char(r, cur[0]);
        i++;
    }
    free(buf);
}

static void renderer_finish(agent_token_renderer *r) {
    if (r->format_thinking) {
        renderer_process(r, NULL, 0, true);
    }
    renderer_markdown_finish(r);
    renderer_flush_utf8(r);
    renderer_reset_color(r);
    if (r->wrote_visible_output) {
        if (!r->last_output_newline) renderer_write(r, "\n", 1);
        renderer_write(r, "\n", 1);
        r->last_output_newline = true;
    }
}

static void renderer_color(agent_token_renderer *r, const char *seq) {
    renderer_markdown_emit_pending_literals(r);
    renderer_flush_utf8(r);
    bool reset = !seq || !seq[0] || !strcmp(seq, "\x1b[0m");
    if (r->use_color && seq && seq[0]) renderer_write(r, seq, strlen(seq));
    r->color_open = r->use_color && !reset;
}

static void renderer_plain(agent_token_renderer *r, const char *s, size_t n) {
    renderer_markdown_emit_pending_literals(r);
    renderer_flush_utf8(r);
    renderer_write(r, s, n);
    if (n) r->wrote_visible_output = true;
    if (n) r->last_output_newline = s[n - 1] == '\n';
}

/* ============================================================================
 * Streaming Tool Visualization
 * ============================================================================
 *
 * Tool calls are parsed for execution later, but they are also visualized while
 * the model is still sampling.  This state machine suppresses raw DSML and
 * prints compact, tool-specific progress such as "$ command" or
 * "Reading file 1:500...".
 */

static bool streq_any(const char *s, const char *a, const char *b,
                      const char *c, const char *d) {
    return (a && !strcmp(s, a)) || (b && !strcmp(s, b)) ||
           (c && !strcmp(s, c)) || (d && !strcmp(s, d));
}

static agent_tool_param_kind agent_tool_param_kind_for(const char *tool, const char *param) {
    if (!tool) tool = "";
    if (!param) param = "";
    if (!strcmp(tool, "bash") && !strcmp(param, "command"))
        return AGENT_TOOL_PARAM_BASH_COMMAND;
    if (!strcmp(tool, "edit") && !strcmp(param, "old"))
        return AGENT_TOOL_PARAM_DIFF_OLD;
    if (!strcmp(tool, "edit") && !strcmp(param, "new"))
        return AGENT_TOOL_PARAM_DIFF_NEW;
    if (streq_any(param, "path", "file", "filename", NULL))
        return AGENT_TOOL_PARAM_PATH;
    if (streq_any(param, "line", "start_line", "end_line", "offset") ||
        streq_any(param, "start", "end", "count", "max_lines") ||
        streq_any(param, "timeout_sec", "refresh_sec", NULL, NULL))
        return AGENT_TOOL_PARAM_OFFSET;
    if (streq_any(param, "content", "text", NULL, NULL))
        return AGENT_TOOL_PARAM_CONTENT;
    return AGENT_TOOL_PARAM_NORMAL;
}

static const char *agent_tool_param_color(agent_tool_param_kind kind) {
    switch (kind) {
    case AGENT_TOOL_PARAM_PATH: return "\x1b[32m";
    case AGENT_TOOL_PARAM_OFFSET: return "\x1b[33m";
    case AGENT_TOOL_PARAM_CONTENT: return "\x1b[34m";
    case AGENT_TOOL_PARAM_DIFF_OLD: return "\x1b[31m";
    case AGENT_TOOL_PARAM_DIFF_NEW: return "\x1b[32m";
    case AGENT_TOOL_PARAM_BASH_COMMAND: return "\x1b[1;36m";
    default: return "\x1b[37m";
    }
}

static void agent_tool_viz_write(agent_stream_renderer *sr, const char *s, size_t n) {
    renderer_plain(sr->renderer, s, n);
    for (size_t i = 0; i < n; i++) sr->viz.last_output_newline = s[i] == '\n';
}

static void agent_tool_viz_puts(agent_stream_renderer *sr, const char *s) {
    agent_tool_viz_write(sr, s, strlen(s));
}

static void agent_tool_viz_start(agent_stream_renderer *sr) {
    agent_tool_visualizer *v = &sr->viz;
    bool line_open = !sr->renderer->last_output_newline;
    memset(v, 0, sizeof(*v));
    v->active = true;
    v->at_line_start = true;
    v->last_output_newline = true;
    if (sr->replay) {
        if (line_open) agent_tool_viz_puts(sr, "\n");
    } else if (sr->renderer->use_color) {
        /* The raw DSML start marker may arrive after ordinary text on the
         * current row.  Clear that row only for the live terminal UI; plain
         * stdout mode must never leak cursor-control escapes into pipes. */
        agent_tool_viz_puts(sr, "\r\x1b[2K");
    } else if (line_open) {
        agent_tool_viz_puts(sr, "\n");
    }
    v->last_output_newline = true;
}

static void agent_tool_viz_line_prefix(agent_stream_renderer *sr) {
    agent_tool_visualizer *v = &sr->viz;
    if (!v->last_output_newline) agent_tool_viz_puts(sr, "\n");
    agent_tool_viz_puts(sr, "🛠️ ");
    v->at_line_start = false;
}

static const char *agent_tool_viz_prefix(const char *name) {
    if (!strcmp(name, "bash")) return "$ ";
    if (!strcmp(name, "read")) return "read ";
    if (!strcmp(name, "write")) return "write ";
    if (!strcmp(name, "edit")) return "edit ";
    if (!strcmp(name, "search")) return "search ";
    if (!strcmp(name, "google_search")) return "google ";
    if (!strcmp(name, "visit_page")) return "visit ";
    return NULL;
}

static void agent_tool_viz_tool(agent_stream_renderer *sr, const char *name) {
    agent_tool_visualizer *v = &sr->viz;
    if (v->tool_announced && !strcmp(v->tool_name, name)) return;
    if (v->tool_announced && !v->last_output_newline) agent_tool_viz_puts(sr, "\n");
    snprintf(v->tool_name, sizeof(v->tool_name), "%s", name ? name : "tool");
    v->tool_announced = true;
    v->read_style = !strcmp(v->tool_name, "read");
    agent_tool_viz_line_prefix(sr);
    if (v->read_style) {
        renderer_color(sr->renderer, "\x1b[1;37m");
        agent_tool_viz_puts(sr, "Reading ");
        renderer_color(sr->renderer, "\x1b[32m");
        v->read_prefix_rendered = true;
        return;
    }
    renderer_color(sr->renderer, !strcmp(v->tool_name, "bash") ?
                                "\x1b[1;36m" : "\x1b[1;37m");
    const char *prefix = agent_tool_viz_prefix(v->tool_name);
    if (prefix) {
        agent_tool_viz_puts(sr, prefix);
    } else {
        agent_tool_viz_puts(sr, v->tool_name);
        agent_tool_viz_puts(sr, " ");
    }
    renderer_color(sr->renderer, "\x1b[0m");
}

static void agent_tool_viz_append(char *dst, size_t cap, char c) {
    size_t len = strlen(dst);
    if (len + 1 >= cap) return;
    dst[len] = c;
    dst[len + 1] = '\0';
}

static void agent_tool_viz_read_value_byte(agent_stream_renderer *sr, char c) {
    agent_tool_visualizer *v = &sr->viz;
    if (!strcmp(v->param_name, "path")) {
        agent_tool_viz_append(v->read_path, sizeof(v->read_path), c);
        if (v->read_prefix_rendered) agent_tool_viz_write(sr, &c, 1);
    } else if (!strcmp(v->param_name, "start_line")) {
        agent_tool_viz_append(v->read_start, sizeof(v->read_start), c);
    } else if (!strcmp(v->param_name, "max_lines")) {
        agent_tool_viz_append(v->read_max, sizeof(v->read_max), c);
    } else if (!strcmp(v->param_name, "whole")) {
        agent_tool_viz_append(v->read_whole, sizeof(v->read_whole), c);
    }
}

static void agent_tool_viz_render_read(agent_stream_renderer *sr) {
    agent_tool_visualizer *v = &sr->viz;
    if (!v->read_style || v->read_line_rendered) return;

    if (!v->read_prefix_rendered) {
        agent_tool_viz_line_prefix(sr);
        renderer_color(sr->renderer, "\x1b[1;37m");
        agent_tool_viz_puts(sr, "Reading ");
        renderer_color(sr->renderer, "\x1b[32m");
        agent_tool_viz_puts(sr, v->read_path[0] ? v->read_path : "<unknown>");
    } else if (!v->read_path[0]) {
        renderer_color(sr->renderer, "\x1b[32m");
        agent_tool_viz_puts(sr, "<unknown>");
    }
    renderer_color(sr->renderer, "\x1b[33m");
    bool whole = agent_parse_bool_default(v->read_whole, false);
    if (whole && (!v->read_start[0] || !strcmp(v->read_start, "1"))) {
        agent_tool_viz_puts(sr, " (whole file)");
    } else if (whole) {
        agent_tool_viz_puts(sr, " ");
        agent_tool_viz_puts(sr, v->read_start);
        agent_tool_viz_puts(sr, ":EOF");
    } else {
        char default_lines[16];
        snprintf(default_lines, sizeof(default_lines), "%d",
                 agent_read_default_lines(sr->renderer ?
                                          sr->renderer->worker : NULL));
        agent_tool_viz_puts(sr, " ");
        agent_tool_viz_puts(sr, v->read_start[0] ? v->read_start : "1");
        agent_tool_viz_puts(sr, ":");
        agent_tool_viz_puts(sr, v->read_max[0] ? v->read_max : default_lines);
    }
    renderer_color(sr->renderer, "\x1b[1;37m");
    agent_tool_viz_puts(sr, "...");
    renderer_color(sr->renderer, "\x1b[0m");
    agent_tool_viz_puts(sr, "\n");
    v->read_line_rendered = true;
}

static bool agent_tool_viz_param_is_code_body(agent_tool_visualizer *v) {
    if (!strcmp(v->tool_name, "write") &&
        v->param_kind == AGENT_TOOL_PARAM_CONTENT)
        return true;
    if (!strcmp(v->tool_name, "edit") &&
        (v->param_kind == AGENT_TOOL_PARAM_DIFF_OLD ||
         v->param_kind == AGENT_TOOL_PARAM_DIFF_NEW ||
         v->param_kind == AGENT_TOOL_PARAM_CONTENT))
        return true;
    return false;
}

static const char *agent_tool_viz_diff_prefix(agent_tool_param_kind kind,
                                              const char **color) {
    if (color) *color = NULL;
    const char *prefix = NULL;
    if (kind == AGENT_TOOL_PARAM_DIFF_OLD) {
        prefix = "- ";
        if (color) *color = "\x1b[31m";
    } else if (kind == AGENT_TOOL_PARAM_DIFF_NEW) {
        prefix = "+ ";
        if (color) *color = "\x1b[32m";
    }
    return prefix;
}

static void agent_tool_viz_code_prefix(agent_stream_renderer *sr) {
    agent_tool_visualizer *v = &sr->viz;
    if (!v->at_line_start) return;
    const char *color = NULL;
    const char *prefix = agent_tool_viz_diff_prefix(v->param_kind, &color);
    if (!prefix) return;
    renderer_color(sr->renderer, color);
    renderer_write(sr->renderer, prefix, strlen(prefix));
    renderer_color(sr->renderer, "\x1b[0m");
    sr->renderer->wrote_visible_output = true;
    sr->renderer->last_output_newline = false;
    v->last_output_newline = false;
    v->at_line_start = false;
}

static void agent_tool_viz_code_begin(agent_stream_renderer *sr) {
    agent_tool_visualizer *v = &sr->viz;
    const agent_syntax *syntax = agent_syntax_for_path(v->tool_path);
    renderer_code_stream_begin(sr->renderer, syntax);
    renderer_code_stream_set_upto_marker(sr->renderer,
        !strcmp(v->tool_name, "edit") &&
        v->param_kind == AGENT_TOOL_PARAM_DIFF_OLD);
    v->code_param_active = true;
    if (v->param_kind == AGENT_TOOL_PARAM_DIFF_OLD ||
        v->param_kind == AGENT_TOOL_PARAM_DIFF_NEW)
    {
        const char *color = NULL;
        const char *prefix = agent_tool_viz_diff_prefix(v->param_kind, &color);
        /* Diff prefixes are terminal UI, not code.  Keep them outside the
         * syntax buffer so a later row repaint preserves their red/green color
         * while highlighting only the actual edited line. */
        renderer_code_stream_set_prefix(sr->renderer, prefix, color);
        agent_tool_viz_code_prefix(sr);
    }
}

static void agent_tool_viz_code_end(agent_stream_renderer *sr) {
    agent_tool_visualizer *v = &sr->viz;
    if (!v->code_param_active) return;
    renderer_code_end(sr->renderer);
    v->code_param_active = false;
    v->at_line_start = true;
    v->last_output_newline = sr->renderer->last_output_newline;
}

static void agent_tool_viz_code_byte(agent_stream_renderer *sr, char c) {
    agent_tool_visualizer *v = &sr->viz;
    agent_tool_viz_code_prefix(sr);
    renderer_code_byte(sr->renderer, c);
    v->last_output_newline = c == '\n';
    v->at_line_start = c == '\n';
}

static void agent_tool_viz_param_begin(agent_stream_renderer *sr, const char *name) {
    agent_tool_visualizer *v = &sr->viz;
    if (!v->tool_announced && sr->parser->current.name)
        agent_tool_viz_tool(sr, sr->parser->current.name);
    snprintf(v->param_name, sizeof(v->param_name), "%s", name ? name : "");
    v->param_kind = agent_tool_param_kind_for(v->tool_name, v->param_name);
    v->param_active = true;
    v->param_end_len = 0;

    if (v->read_style) return;

    if (v->param_kind == AGENT_TOOL_PARAM_DIFF_OLD ||
        v->param_kind == AGENT_TOOL_PARAM_DIFF_NEW)
    {
        if (!v->last_output_newline) agent_tool_viz_puts(sr, "\n");
        v->at_line_start = true;
        agent_tool_viz_code_begin(sr);
        return;
    }

    if (v->param_kind == AGENT_TOOL_PARAM_CONTENT) {
        if (!v->last_output_newline) agent_tool_viz_puts(sr, "\n");
        if (strcmp(v->tool_name, "write")) {
            renderer_color(sr->renderer, "\x1b[1;37m");
            agent_tool_viz_puts(sr, v->param_name);
            agent_tool_viz_puts(sr, ":\n");
        }
        v->at_line_start = true;
        if (agent_tool_viz_param_is_code_body(v)) {
            agent_tool_viz_code_begin(sr);
        } else {
            renderer_color(sr->renderer, "\x1b[34m");
        }
        return;
    }

    if (v->param_kind != AGENT_TOOL_PARAM_BASH_COMMAND) {
        if (!v->at_line_start) agent_tool_viz_puts(sr, " ");
        renderer_color(sr->renderer, "\x1b[1;37m");
        agent_tool_viz_puts(sr, v->param_name);
        agent_tool_viz_puts(sr, "=");
    } else {
        renderer_color(sr->renderer, agent_tool_param_color(AGENT_TOOL_PARAM_BASH_COMMAND));
        return;
    }
    renderer_color(sr->renderer, agent_tool_param_color(v->param_kind));
}

static void agent_tool_viz_param_end(agent_stream_renderer *sr) {
    agent_tool_visualizer *v = &sr->viz;
    v->param_end_len = 0;
    if (v->code_param_active) agent_tool_viz_code_end(sr);
    if (!v->read_style) renderer_color(sr->renderer, "\x1b[0m");
    v->param_active = false;
    v->param_name[0] = '\0';
}

static void agent_tool_viz_param_raw_byte(agent_stream_renderer *sr, char c) {
    agent_tool_visualizer *v = &sr->viz;
    if (v->read_style) {
        agent_tool_viz_read_value_byte(sr, c);
        return;
    }
    if (v->param_kind == AGENT_TOOL_PARAM_PATH) {
        agent_tool_viz_append(v->tool_path, sizeof(v->tool_path), c);
    }
    if (v->code_param_active) {
        agent_tool_viz_code_byte(sr, c);
        return;
    }
    if (v->param_kind == AGENT_TOOL_PARAM_BASH_COMMAND) {
        agent_tool_viz_write(sr, &c, 1);
        v->at_line_start = c == '\n';
        return;
    }
    if (v->param_kind == AGENT_TOOL_PARAM_DIFF_OLD ||
        v->param_kind == AGENT_TOOL_PARAM_DIFF_NEW)
    {
        agent_tool_viz_code_begin(sr);
        agent_tool_viz_code_byte(sr, c);
        return;
    }
    agent_tool_viz_write(sr, &c, 1);
    v->at_line_start = c == '\n';
}

static void agent_tool_viz_restore_param_color(agent_stream_renderer *sr) {
    agent_tool_visualizer *v = &sr->viz;
    if (!v->active || !v->param_active || v->read_style) return;
    renderer_color(sr->renderer, agent_tool_param_color(v->param_kind));
}

/* Stream one DSML parameter byte into the visualizer.  The visualizer must not
 * wait for the whole parameter: large write/edit contents should show progress
 * as the model emits them, while still detecting the closing parameter tag. */
static void agent_tool_viz_param_value_byte(agent_stream_renderer *sr, char c) {
    agent_tool_visualizer *v = &sr->viz;

    if (v->param_end_len || c == '<') {
        if (v->param_end_len == sizeof(v->param_end_tail)) {
            size_t keep = v->param_end_len;
            v->param_end_len = 0;
            for (size_t i = 0; i < keep; i++)
                agent_tool_viz_param_raw_byte(sr, v->param_end_tail[i]);
            if (c != '<') {
                agent_tool_viz_param_raw_byte(sr, c);
                return;
            }
        }
        if (v->param_end_len < sizeof(v->param_end_tail))
            v->param_end_tail[v->param_end_len++] = c;
        bool complete = false;
        if (agent_tool_value_close_tail(sr->parser->syntax,
                                        v->param_end_tail,
                                        v->param_end_len,
                                        &complete)) {
            if (complete) agent_tool_viz_param_end(sr);
            return;
        }
        size_t keep = v->param_end_len;
        v->param_end_len = 0;
        for (size_t i = 0; i < keep; i++)
            agent_tool_viz_param_raw_byte(sr, v->param_end_tail[i]);
        return;
    }
    agent_tool_viz_param_raw_byte(sr, c);
}

static void agent_tool_viz_finish(agent_stream_renderer *sr, const char *status) {
    agent_tool_visualizer *v = &sr->viz;
    if (!v->active) return;
    if (v->param_active) agent_tool_viz_param_end(sr);
    if (!status || !status[0]) agent_tool_viz_render_read(sr);
    if (status && status[0]) {
        if (!v->last_output_newline) agent_tool_viz_puts(sr, "\n");
        renderer_color(sr->renderer, "\x1b[90m");
        agent_tool_viz_puts(sr, status);
        renderer_color(sr->renderer, "\x1b[0m");
    }
    if (!v->last_output_newline) agent_tool_viz_puts(sr, "\n");
    v->active = false;
}

static void agent_tool_viz_dump_invalid_dsml(agent_stream_renderer *sr) {
    agent_tool_visualizer *v = &sr->viz;
    if (!v->active) return;

    /* The normal path hides DSML and paints a friendly semantic projection.  If
     * parsing fails, show the exact bytes we rejected so the next fix is based
     * on evidence instead of guessing from the projection. */
    if (v->param_active) {
        v->param_active = false;
        v->param_end_len = 0;
        v->param_name[0] = '\0';
    }
    if (!v->last_output_newline) agent_tool_viz_puts(sr, "\n");
    renderer_color(sr->renderer, "\x1b[1;31m");
    if (sr->parser->raw && sr->parser->raw_len) {
        agent_tool_viz_write(sr, sr->parser->raw, sr->parser->raw_len);
    } else {
        agent_tool_viz_puts(sr, "<empty DSML>");
    }
    renderer_color(sr->renderer, "\x1b[0m");
    if (!v->last_output_newline) agent_tool_viz_puts(sr, "\n");
}

static void agent_stream_finish_ignored_dsml(agent_stream_renderer *sr, const char *detail) {
    const char *msg =
        detail && detail[0] ? detail :
        "tool calling is not allowed inside <think></think>";
    sr->dsml_in_think = true;
    sr->dsml_in_think_reported = true;
    agent_trace(sr->renderer->worker, "dsml ignored inside thinking: %s", msg);
    if (!sr->renderer->last_output_newline)
        renderer_plain(sr->renderer, "\n", 1);
    renderer_color(sr->renderer, "\x1b[1;31m");
    renderer_plain(sr->renderer, "[tool call ignored: ", 20);
    renderer_plain(sr->renderer, msg, strlen(msg));
    renderer_plain(sr->renderer, "]\n", 2);
    renderer_color(sr->renderer, "\x1b[0m");
    agent_dsml_parser_reset(sr->parser);
    sr->dsml_active = false;
    sr->dsml_ignored = false;
}

static void agent_stream_malformed_dsml(agent_stream_renderer *sr,
                                        const char *detail) {
    const char *msg = detail && detail[0] ? detail :
        "DSML markup outside a valid tool_calls block";
    if (sr->parser->state == AGENT_DSML_ERROR) return;
    agent_dsml_set_error(sr->parser, msg);
    agent_trace(sr->renderer->worker, "malformed dsml in assistant output: %s", msg);
    if (!sr->renderer->last_output_newline)
        renderer_plain(sr->renderer, "\n", 1);
    renderer_color(sr->renderer, "\x1b[1;31m");
    renderer_plain(sr->renderer, "[invalid tool call: ", 20);
    renderer_plain(sr->renderer, msg, strlen(msg));
    renderer_plain(sr->renderer, "]\n", 2);
    renderer_color(sr->renderer, "\x1b[0m");
}

/* Mirror parser progress into the terminal visualizer.  Parser state is the
 * source of truth; this function only decides what the user should see. */
static void agent_stream_tool_events(agent_stream_renderer *sr) {
    agent_dsml_parser *p = sr->parser;
    agent_tool_visualizer *v = &sr->viz;
    if (!v->tool_announced && p->current.name)
        agent_tool_viz_tool(sr, p->current.name);
    if (v->tool_announced && !p->current.name && !v->param_active) {
        agent_tool_viz_render_read(sr);
        if (!v->last_output_newline) agent_tool_viz_puts(sr, "\n");
        v->read_style = false;
        v->read_prefix_rendered = false;
        v->read_line_rendered = false;
        v->read_path[0] = '\0';
        v->read_start[0] = '\0';
        v->read_max[0] = '\0';
        v->read_whole[0] = '\0';
        v->tool_announced = false;
    }
    if (!v->param_active && p->state == AGENT_DSML_PARAM_VALUE && p->param_name)
        agent_tool_viz_param_begin(sr, p->param_name);
}

static void agent_stream_preflight_closed_param(agent_stream_renderer *sr) {
    if (!sr || sr->replay || sr->dsml_ignored || sr->tool_preflight_error)
        return;
    agent_dsml_parser *p = sr->parser;
    agent_tool_visualizer *v = &sr->viz;
    if (!p || !v->param_active || strcmp(v->param_name, "old") != 0)
        return;
    if (!p->current.name || strcmp(p->current.name, "edit") != 0)
        return;

    char err[256] = {0};
    if (agent_preflight_edit_old(sr->renderer->worker, &p->current,
                                 err, sizeof(err)))
        return;

    sr->tool_preflight_error = true;
    snprintf(sr->tool_preflight_error_msg, sizeof(sr->tool_preflight_error_msg),
             "edit old selector failed before new was generated: %s",
             err[0] ? err : "old text is not a unique match");
    agent_trace(sr->renderer->worker, "edit old preflight failed: %s",
                sr->tool_preflight_error_msg);
}

static void agent_stream_feed_dsml_byte(agent_stream_renderer *sr, char c) {
    bool was_param = !sr->dsml_ignored && sr->viz.param_active;
    agent_dsml_feed(sr->parser, &c, 1);
    if (!sr->dsml_ignored) {
        agent_stream_tool_events(sr);
        if (was_param) agent_tool_viz_param_value_byte(sr, c);
        if (was_param && sr->parser->state != AGENT_DSML_PARAM_VALUE &&
            sr->viz.param_active)
        {
            agent_stream_preflight_closed_param(sr);
            agent_tool_viz_param_end(sr);
        }
    }
    if (sr->parser->state == AGENT_DSML_DONE) {
        if (sr->dsml_ignored) {
            agent_stream_finish_ignored_dsml(
                sr, "tool calling is not allowed inside <think></think>");
        } else {
            agent_trace(sr->renderer->worker, "dsml done calls=%d",
                        sr->parser->calls.len);
            agent_tool_viz_finish(sr, NULL);
            sr->dsml_active = false;
        }
    } else if (sr->parser->state == AGENT_DSML_ERROR) {
        if (sr->dsml_ignored) {
            agent_stream_finish_ignored_dsml(
                sr, "malformed tool call inside <think></think>");
        } else {
            char status[220];
            snprintf(status, sizeof(status), "[invalid tool call: %s]\n",
                     sr->parser->error[0] ? sr->parser->error : "parse error");
            agent_trace(sr->renderer->worker, "dsml error %s",
                        sr->parser->error[0] ? sr->parser->error : "parse error");
            agent_tool_viz_dump_invalid_dsml(sr);
            agent_tool_viz_finish(sr, status);
            sr->dsml_active = false;
        }
    }
}

/* Start a DSML block from the streaming detector.  The detector may accept a
 * known malformed opening form for robustness, but the parser is seeded with
 * canonical bytes so all later parsing remains strict. */
static void agent_stream_start_dsml(agent_stream_renderer *sr, bool ignored) {
    sr->dsml_active = true;
    sr->dsml_ignored = ignored;
    if (ignored) sr->dsml_in_think = true;
    sr->dsml_start_len = 0;
    sr->post_think_gap = false;
    agent_trace(sr->renderer->worker, "%s tool start detected%s",
                sr->syntax == AGENT_TOOL_SYNTAX_GLM ? "glm" : "dsml",
                ignored ? " inside thinking" : "");
    agent_dsml_start(sr->parser);
    if (!ignored) {
        agent_tool_viz_start(sr);
        agent_stream_tool_events(sr);
    }
}

static void agent_stream_note_plain_dsml_byte(agent_stream_renderer *sr, char c);

static void agent_stream_flush_start_tail(agent_stream_renderer *sr) {
    if (!sr->dsml_start_len) return;
    sr->post_think_gap = false;
    for (size_t i = 0; i < sr->dsml_start_len; i++) {
        renderer_write_char(sr->renderer, sr->dsml_start_tail[i]);
        agent_stream_note_plain_dsml_byte(sr, sr->dsml_start_tail[i]);
        if (sr->parser->state == AGENT_DSML_ERROR) break;
    }
    sr->dsml_start_len = 0;
}

static bool agent_stream_dsml_start_match(agent_tool_syntax syntax,
                                          const char *tail, size_t len,
                                          bool *complete,
                                          bool *implicit_invoke) {
    if (syntax == AGENT_TOOL_SYNTAX_GLM) {
        static const char glm_call[] = "<tool_call>";
        size_t form_len = sizeof(glm_call) - 1;
        *complete = false;
        *implicit_invoke = false;
        if (len <= form_len && memcmp(glm_call, tail, len) == 0) {
            *complete = len == form_len;
            return true;
        }
        return false;
    }

    static const char canonical[] = "<｜DSML｜tool_calls>";
    static const char missing_bar[] = "<DSML｜tool_calls>";
    static const char invoke[] = "<｜DSML｜invoke";
    static const char invoke_missing_bar[] = "<DSML｜invoke";
    struct {
        const char *text;
        bool implicit_invoke;
    } forms[] = {
        {canonical, false},
        {missing_bar, false},
        {invoke, true},
        {invoke_missing_bar, true},
    };
    *complete = false;
    *implicit_invoke = false;
    for (size_t i = 0; i < sizeof(forms)/sizeof(forms[0]); i++) {
        size_t form_len = strlen(forms[i].text);
        if (len <= form_len && memcmp(forms[i].text, tail, len) == 0) {
            *complete = len == form_len;
            *implicit_invoke = forms[i].implicit_invoke;
            return true;
        }
    }
    return false;
}

static bool agent_tail_matches(const char *tail, size_t len,
                               const char *needle, size_t needle_len) {
    return len >= needle_len &&
           memcmp(tail + len - needle_len, needle, needle_len) == 0;
}

/* Detect DSML-looking control markers in text that is not currently owned by
 * the executable DSML parser.  This helper intentionally has no policy: inside
 * <think> the marker means "tool call attempted too early", while in normal
 * assistant output it means malformed DSML that the model should see as a tool
 * error. */
static bool agent_dsml_marker_detector_feed(agent_dsml_marker_detector *d,
                                            char c) {
    if (d->len == sizeof(d->tail)) {
        memmove(d->tail, d->tail + 1, sizeof(d->tail) - 1);
        d->len--;
    }
    d->tail[d->len++] = c;

    static const char fullwidth_marker[] = "｜DSML｜";
    static const char ascii_marker[] = "|DSML|";
    static const char missing_open[] = "<DSML｜";
    static const char missing_close[] = "</DSML｜";
    return agent_tail_matches(d->tail, d->len,
                              fullwidth_marker, sizeof(fullwidth_marker) - 1) ||
           agent_tail_matches(d->tail, d->len,
                              ascii_marker, sizeof(ascii_marker) - 1) ||
           agent_tail_matches(d->tail, d->len,
                              missing_open, sizeof(missing_open) - 1) ||
           agent_tail_matches(d->tail, d->len,
                              missing_close, sizeof(missing_close) - 1);
}

static void agent_stream_note_thinking_dsml_byte(agent_stream_renderer *sr,
                                                 char c) {
    if (!sr->in_think || sr->dsml_in_think) return;
    if (agent_dsml_marker_detector_feed(&sr->think_dsml, c))
        sr->dsml_in_think = true;
}

static void agent_stream_note_plain_dsml_byte(agent_stream_renderer *sr,
                                              char c) {
    if (sr->parser->state == AGENT_DSML_ERROR) return;
    if (sr->dsml_active || sr->in_think || sr->dsml_in_think) return;
    if (agent_dsml_marker_detector_feed(&sr->plain_dsml, c)) {
        agent_stream_malformed_dsml(
            sr, "DSML markup outside a valid tool_calls block");
    }
}

/* Route ordinary assistant bytes either to normal markdown rendering or into
 * the DSML detector.  The detector must hold short prefixes because the model
 * can split "<｜DSML｜tool_calls>" across arbitrary tokens. */
static void agent_stream_normal_byte(agent_stream_renderer *sr, char c) {
    static const char canonical_invoke[] = "<｜DSML｜invoke";
    const char *start = sr->syntax == AGENT_TOOL_SYNTAX_GLM ?
        "<tool_call>" : "<｜DSML｜tool_calls>";
    if (sr->parser->state == AGENT_DSML_ERROR) return;
    agent_stream_note_thinking_dsml_byte(sr, c);

    /* DeepSeek usually emits one or more blank lines after </think> before
     * either prose or a DSML tool stanza.  At that point the bytes are just a
     * visual gap between the hidden thinking phase and the real answer, and
     * printing them makes tool calls appear after odd empty lines.  We only
     * suppress whitespace in this very narrow post-thinking window; once the
     * first non-space byte arrives, normal rendering resumes. */
    if (sr->post_think_gap &&
        (c == ' ' || c == '\t' || c == '\r' || c == '\n'))
    {
        return;
    }

    if (sr->dsml_start_len || c == start[0]) {
        if (sr->dsml_start_len < sizeof(sr->dsml_start_tail))
            sr->dsml_start_tail[sr->dsml_start_len++] = c;
        bool complete = false, implicit_invoke = false;
        if (agent_stream_dsml_start_match(sr->syntax,
                                          sr->dsml_start_tail, sr->dsml_start_len,
                                          &complete, &implicit_invoke))
        {
            if (complete) {
                /* Accept the common missing-leading-bar typo
                 * "<DSML｜tool_calls>" here, but seed the parser with the
                 * canonical marker so the rest of the DSML parser stays
                 * strict and simple.  Also accept a direct invoke opener as an
                 * implicit tool_calls block; the model often knows it wants a
                 * tool but forgets the outer wrapper. */
                agent_stream_start_dsml(sr, sr->in_think);
                if (sr->syntax == AGENT_TOOL_SYNTAX_DSML && implicit_invoke) {
                    for (size_t i = 0; i < sizeof(canonical_invoke) - 1; i++)
                        agent_stream_feed_dsml_byte(sr, canonical_invoke[i]);
                }
            }
            return;
        }
        if (sr->dsml_start_len > 1 &&
            sr->dsml_start_tail[sr->dsml_start_len - 1] == start[0])
        {
            sr->post_think_gap = false;
            size_t flush = sr->dsml_start_len - 1;
            for (size_t i = 0; i < flush; i++) {
                renderer_write_char(sr->renderer, sr->dsml_start_tail[i]);
                agent_stream_note_plain_dsml_byte(sr, sr->dsml_start_tail[i]);
                if (sr->parser->state == AGENT_DSML_ERROR) break;
            }
            if (sr->parser->state == AGENT_DSML_ERROR) {
                sr->dsml_start_len = 0;
                return;
            }
            sr->dsml_start_tail[0] = start[0];
            sr->dsml_start_len = 1;
            return;
        }
        agent_stream_flush_start_tail(sr);
        return;
    }

    sr->post_think_gap = false;
    renderer_write_char(sr->renderer, c);
    agent_stream_note_plain_dsml_byte(sr, c);
}

/* This is the single streaming display state machine for assistant output.  It
 * hides raw DSML as soon as the tool_calls marker is complete, lets the DSML
 * parser continue building executable calls, and paints semantic tool output
 * from parser state changes.  The sampled transcript remains unchanged: only
 * the terminal projection is rewritten. */
static void agent_stream_text(agent_stream_renderer *sr, const char *text, size_t len, bool finish) {
    const char *think_open = "<think>";
    const char *think_close = "</think>";
    size_t total = sr->pending_len + len;
    char *buf = xmalloc(total ? total : 1);
    if (sr->pending_len) memcpy(buf, sr->pending, sr->pending_len);
    if (len) memcpy(buf + sr->pending_len, text, len);
    sr->pending_len = 0;

    /* The UI may reset terminal attributes while redrawing the editable prompt
     * between generated chunks.  If a DSML parameter is still streaming, make
     * each new token fragment self-contained by restoring the active parameter
     * color before visible bytes are projected.  This keeps the prompt normal
     * without sacrificing long write/edit content coloring. */
    if (len) agent_tool_viz_restore_param_color(sr);
    if (len && !sr->dsml_active) renderer_restore_text_attrs(sr->renderer);

    size_t i = 0;
    while (i < total) {
        char *cur = buf + i;
        size_t rem = total - i;
        if (!sr->dsml_active && bytes_has_prefix(cur, rem, think_open)) {
            agent_stream_flush_start_tail(sr);
            sr->post_think_gap = false;
            sr->in_think = true;
            sr->renderer->in_think = true;
            i += strlen(think_open);
            continue;
        }
        if (!sr->dsml_active && bytes_has_prefix(cur, rem, think_close)) {
            agent_stream_flush_start_tail(sr);
            sr->in_think = false;
            sr->renderer->in_think = false;
            renderer_reset_color(sr->renderer);
            if (!sr->renderer->last_output_newline)
                renderer_write(sr->renderer, "\n", 1);
            renderer_write(sr->renderer, "\n", 1);
            sr->renderer->last_output_newline = true;
            sr->post_think_gap = true;
            i += strlen(think_close);
            continue;
        }
        if (!finish && !sr->dsml_active && cur[0] == '<' &&
            (bytes_is_partial_prefix(cur, rem, think_open) ||
             bytes_is_partial_prefix(cur, rem, think_close)))
        {
            if (rem < sizeof(sr->pending)) {
                memcpy(sr->pending, cur, rem);
                sr->pending_len = rem;
            }
            break;
        }

        if (sr->dsml_active) {
            agent_stream_feed_dsml_byte(sr, cur[0]);
        } else if (sr->in_think) {
            /* Tool calls are executable only after thinking has closed.  Still
             * route thinking bytes through the DSML start detector so an
             * accidental in-think tool stanza can be suppressed cleanly instead
             * of being shown as raw markup or, worse, executed. */
            agent_stream_normal_byte(sr, cur[0]);
        } else {
            agent_stream_normal_byte(sr, cur[0]);
        }
        i++;
    }
    free(buf);

    if (finish) {
        agent_stream_flush_start_tail(sr);
        sr->post_think_gap = false;
        if (sr->dsml_active) agent_dsml_finish(sr->parser);
        if (sr->dsml_active) {
            if (sr->parser->state == AGENT_DSML_DONE) {
                if (sr->dsml_ignored) {
                    agent_stream_finish_ignored_dsml(
                        sr, "tool calling is not allowed inside <think></think>");
                } else {
                    agent_trace(sr->renderer->worker, "dsml done calls=%d",
                                sr->parser->calls.len);
                    agent_tool_viz_finish(sr, NULL);
                    sr->dsml_active = false;
                }
            } else if (sr->dsml_ignored) {
                agent_stream_finish_ignored_dsml(
                    sr, "tool calling is not allowed inside <think></think>");
            } else {
                agent_tool_viz_finish(sr, sr->tool_preflight_error ?
                                      "[tool call stopped: edit old selector failed]\n" :
                                      "[tool call interrupted]\n");
                sr->dsml_active = false;
            }
        }
        if (sr->dsml_in_think && !sr->dsml_in_think_reported) {
            agent_stream_finish_ignored_dsml(
                sr, "tool calling is not allowed inside <think></think>");
        }
    }
}

/* ============================================================================
 * Worker Progress And Generic Buffers
 * ============================================================================
 */

static void worker_progress_cb(void *ud, const char *event, int current, int total) {
    agent_worker *w = ud;
    if (!w || !event) return;
    (void)total;
    worker_apply_pending_power(w);
    pthread_mutex_lock(&w->mu);
    int done = 0;
    int progress_total = w->status.prefill_total;
    if (!strcmp(event, "prefill_work")) {
        /* GLM reports intra-chunk work relative to the suffix being synced.
         * It is display progress only: checkpoint progress still arrives as
         * absolute prefill_chunk positions after each completed chunk. */
        done = current;
    } else if (!strcmp(event, "prefill_chunk") ||
               !strcmp(event, "prefill_display")) {
        if (w->progress_direct) {
            pthread_mutex_unlock(&w->mu);
            return;
        }
        done = current - w->progress_base;
    } else {
        pthread_mutex_unlock(&w->mu);
        return;
    }
    if (progress_total < 0) progress_total = 0;
    if (done < 0) done = 0;
    if (done > progress_total) done = progress_total;
    w->status.prefill_total = progress_total;
    w->status.prefill_done = done;
    double elapsed = now_sec() - w->progress_started_at;
    w->status.prefill_tps =
        done > 0 && elapsed > 0.0 ? (double)done / elapsed : 0.0;
    agent_wake_locked(w);
    pthread_mutex_unlock(&w->mu);
}

static bool worker_should_interrupt(agent_worker *w) {
    pthread_mutex_lock(&w->mu);
    bool interrupt = w->interrupt || w->stop;
    pthread_mutex_unlock(&w->mu);
    return interrupt;
}

/* Ctrl+C is a latched request consumed by the worker.  Once an interrupted
 * operation has reached a stable append-only boundary and is about to publish
 * IDLE, the request must be acknowledged; otherwise the editor can observe an
 * idle worker with a stale interrupt still pending. */
static void worker_clear_interrupt(agent_worker *w) {
    pthread_mutex_lock(&w->mu);
    w->interrupt = false;
    pthread_mutex_unlock(&w->mu);
}

static bool agent_err_is_interrupted(const char *err) {
    return err && !strcmp(err, "interrupted");
}

static bool worker_cancel_session_cb(void *ud) {
    return worker_should_interrupt(ud);
}

typedef struct {
    char *ptr;
    size_t len;
    size_t cap;
    bool truncated;
} agent_buf;

static void agent_buf_append(agent_buf *b, const char *s, size_t n) {
    if (!n || b->truncated) return;
    const size_t max = 128 * 1024;
    if (b->len + n > max) {
        n = max > b->len ? max - b->len : 0;
        b->truncated = true;
    }
    if (!n) return;
    if (b->len + n + 1 > b->cap) {
        size_t cap = b->cap ? b->cap * 2 : 4096;
        while (cap < b->len + n + 1) cap *= 2;
        b->ptr = xrealloc(b->ptr, cap);
        b->cap = cap;
    }
    memcpy(b->ptr + b->len, s, n);
    b->len += n;
    b->ptr[b->len] = '\0';
}

static void agent_buf_puts(agent_buf *b, const char *s) {
    agent_buf_append(b, s, strlen(s));
}

static char *agent_buf_take(agent_buf *b) {
    if (!b->ptr) return xstrdup("");
    char *p = b->ptr;
    memset(b, 0, sizeof(*b));
    return p;
}

static bool agent_tokens_equal(const ds4_tokens *a, const ds4_tokens *b) {
    if (!a || !b || a->len != b->len) return false;
    for (int i = 0; i < a->len; i++) {
        if (a->v[i] != b->v[i]) return false;
    }
    return true;
}

static bool agent_mkdir_p(const char *path) {
    if (!path || !path[0]) return false;
    char *tmp = xstrdup(path);
    for (char *p = tmp + 1; *p; p++) {
        if (*p != '/') continue;
        *p = '\0';
        if (mkdir(tmp, 0700) != 0 && errno != EEXIST) {
            free(tmp);
            return false;
        }
        *p = '/';
    }
    bool ok = mkdir(tmp, 0700) == 0 || errno == EEXIST;
    free(tmp);
    return ok;
}

static char *agent_default_cache_dir(void) {
    const char *home = getenv("HOME");
    if (!home || !home[0]) home = ".";
    agent_buf b = {0};
    agent_buf_puts(&b, home);
    if (b.len == 0 || b.ptr[b.len - 1] != '/') agent_buf_puts(&b, "/");
    agent_buf_puts(&b, ".ds4/kvcache");
    return agent_buf_take(&b);
}

static char *agent_kv_path_for_sha(const char *dir, const char sha[41]) {
    char name[44];
    memcpy(name, sha, 40);
    memcpy(name + 40, ".kv", 4);
    return ds4_kvstore_path_join(dir, name);
}

static void agent_le_put64(uint8_t *p, uint64_t v) {
    for (int i = 0; i < 8; i++) p[i] = (uint8_t)(v >> (8 * i));
}

/* Agent session IDs are intentionally independent from the rendered transcript:
 * once a session has a title and creation time, resaving it keeps the same file
 * name while the transcript and KV payload evolve. */
static void agent_session_identity_sha(const char *title, uint64_t created_at,
                                       char sha_out[41]) {
    size_t title_len = title ? strlen(title) : 0;
    agent_buf b = {0};
    agent_buf_append(&b, title ? title : "", title_len);
    uint8_t ts[8];
    agent_le_put64(ts, created_at);
    agent_buf_append(&b, (const char *)ts, sizeof(ts));
    ds4_kvstore_sha1_bytes_hex(b.ptr ? b.ptr : "", b.len, sha_out);
    free(b.ptr);
}

static void agent_worker_clear_session_identity(agent_worker *w) {
    w->session_sha[0] = '\0';
    free(w->session_title);
    w->session_title = NULL;
    w->session_created_at = 0;
    free(w->legacy_session_path_to_delete);
    w->legacy_session_path_to_delete = NULL;
}

typedef struct {
    bool has_title_trailer;
    bool legacy_identity;
    char *title;
    uint64_t created_at;
    char sha[41];
} agent_kv_session_meta;

static void agent_kv_session_meta_free(agent_kv_session_meta *m) {
    free(m->title);
    memset(m, 0, sizeof(*m));
}

/* ============================================================================
 * Agent KV Store And Session Persistence
 * ============================================================================
 */

static char *agent_session_title_from_text(const char *text, size_t text_len,
                                           size_t max_bytes);

/* Agent sessions deliberately use a different policy from ds4-server:
 *
 * - sysprompt.kv is a fixed bootstrap checkpoint for the current tool/system
 *   prompt.  Because its name is fixed, the current rendered text is compared
 *   with the text stored in the file before loading.  A mismatch simply rebuilds
 *   and overwrites the file.
 * - conversation sessions are explicit saves only.  Their stable file name is
 *   SHA1(title || created_at_le64).kv, where title is the first user prompt and
 *   created_at is preserved across future saves.  The title is stored in an
 *   agent-only trailer after the KV payload.
 *
 * The DS4 payload stores the exact token sequence and graph state.  The rendered
 * text is retained for listing, history rendering, and stripped-session rebuilds. */
static bool agent_kv_read_text(FILE *fp, uint32_t text_bytes,
                               char **text_out, char *err, size_t err_len) {
    char *text = xmalloc((size_t)text_bytes + 1);
    if (fread(text, 1, text_bytes, fp) != text_bytes) {
        if (err && err_len) snprintf(err, err_len, "truncated cached text");
        free(text);
        return false;
    }
    text[text_bytes] = '\0';
    *text_out = text;
    return true;
}

static bool agent_kv_write_title_trailer(FILE *fp, const char *title,
                                         char *err, size_t err_len) {
    size_t title_len = title ? strlen(title) : 0;
    if (title_len > UINT32_MAX) {
        snprintf(err, err_len, "agent session title is too large");
        return false;
    }
    uint8_t tb[4];
    ds4_kvstore_le_put32(tb, (uint32_t)title_len);
    return fwrite(tb, 1, sizeof(tb), fp) == sizeof(tb) &&
           fwrite(title ? title : "", 1, title_len, fp) == title_len;
}

/* Read the optional agent title trailer without disturbing the payload cursor.
 * The caller is positioned just after rendered text, which is also the payload
 * start expected by ds4_session_load_payload(). */
static bool agent_kv_read_title_trailer(FILE *fp, const ds4_kvstore_entry *hdr,
                                        char **title_out,
                                        char *err, size_t err_len) {
    off_t payload_pos = ftello(fp);
    if (payload_pos < 0) {
        if (err && err_len) snprintf(err, err_len, "%s", strerror(errno));
        return false;
    }
    if (hdr->payload_bytes > (uint64_t)LLONG_MAX ||
        fseeko(fp, (off_t)hdr->payload_bytes, SEEK_CUR) != 0)
    {
        if (err && err_len) snprintf(err, err_len, "%s", strerror(errno));
        return false;
    }

    uint8_t tb[4];
    if (fread(tb, 1, sizeof(tb), fp) != sizeof(tb)) {
        if (err && err_len) snprintf(err, err_len, "missing agent session title trailer");
        fseeko(fp, payload_pos, SEEK_SET);
        return false;
    }
    uint32_t title_bytes = ds4_kvstore_le_get32(tb);
    char *title = xmalloc((size_t)title_bytes + 1);
    if (fread(title, 1, title_bytes, fp) != title_bytes) {
        if (err && err_len) snprintf(err, err_len, "truncated agent session title trailer");
        free(title);
        fseeko(fp, payload_pos, SEEK_SET);
        return false;
    }
    title[title_bytes] = '\0';
    if (fseeko(fp, payload_pos, SEEK_SET) != 0) {
        if (err && err_len) snprintf(err, err_len, "%s", strerror(errno));
        free(title);
        return false;
    }
    *title_out = title;
    return true;
}

static void agent_kv_identity_sha(const ds4_kvstore_entry *hdr,
                                  const char *text, uint32_t text_bytes,
                                  const char *title,
                                  char sha_out[41]) {
    if (hdr->ext_flags & DS4_KVSTORE_EXT_SESSION_TITLE) {
        agent_session_identity_sha(title ? title : "", hdr->created_at, sha_out);
    } else {
        ds4_kvstore_sha1_bytes_hex(text, text_bytes, sha_out);
    }
}

/* Load a KV file and optionally verify either its session identity or exact
 * rendered text.  sysprompt.kv uses exact text because the file name is fixed;
 * saved sessions use their filename SHA: modern agent sessions hash the title
 * trailer plus created_at, while legacy sessions still hash rendered text. */
static bool agent_kv_load_path(agent_worker *w, const char *path,
                               const char *expected_sha,
                               const char *expected_text,
                               size_t expected_text_len,
                               ds4_tokens *loaded_tokens,
                               agent_kv_session_meta *meta_out,
                               char *err, size_t err_len) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        snprintf(err, err_len, "%s", strerror(errno));
        return false;
    }

    ds4_kvstore_entry hdr = {0};
    uint32_t text_bytes = 0;
    bool ok = ds4_kvstore_read_header(fp, &hdr, &text_bytes);
    if (!ok) snprintf(err, err_len, "invalid KV header");

    char *text = NULL;
    if (ok) ok = agent_kv_read_text(fp, text_bytes, &text, err, err_len);
    char *title = NULL;
    bool has_title = ok && (hdr.ext_flags & DS4_KVSTORE_EXT_SESSION_TITLE);
    if (has_title)
        ok = agent_kv_read_title_trailer(fp, &hdr, &title, err, err_len);
    uint32_t expected_tokens = hdr.tokens;
    if (ok && hdr.payload_bytes != 0 &&
        hdr.model_id != (uint8_t)ds4_engine_model_id(w->engine))
    {
        snprintf(err, err_len, "KV checkpoint was written for a different model");
        ok = false;
    }
    if (ok && hdr.payload_bytes != 0 &&
        hdr.quant_bits != (uint8_t)ds4_engine_routed_quant_bits(w->engine))
    {
        snprintf(err, err_len, "KV checkpoint was written for a different quantization");
        ok = false;
    }
    if (ok && expected_text) {
        if ((size_t)text_bytes != expected_text_len ||
            memcmp(text, expected_text, expected_text_len) != 0)
        {
            snprintf(err, err_len, "cached text does not match current system prompt");
            ok = false;
        }
    }
    if (ok && expected_sha) {
        char actual_sha[41];
        agent_kv_identity_sha(&hdr, text, text_bytes, title, actual_sha);
        if (strcmp(actual_sha, expected_sha)) {
            snprintf(err, err_len, "cached session identity does not match file name");
            ok = false;
        }
    }

    char load_err[160] = {0};
    if (ok && hdr.payload_bytes == 0) {
        ds4_tokens rebuilt = {0};
        ds4_tokenize_rendered_chat(w->engine, text, &rebuilt);
        expected_tokens = (uint32_t)rebuilt.len;
        if (agent_worker_sync_tokens(w, &rebuilt, true, err, err_len) != 0) {
            ds4_session_invalidate(w->session);
            ok = false;
        }
        ds4_tokens_free(&rebuilt);
    } else if (ok &&
               ds4_session_load_payload(w->session, fp, hdr.payload_bytes,
                                        load_err, sizeof(load_err)) != 0)
    {
        snprintf(err, err_len, "%s", load_err[0] ? load_err : "failed to load KV payload");
        ds4_session_invalidate(w->session);
        ok = false;
    }
    fclose(fp);

    if (ok) {
        const ds4_tokens *live = ds4_session_tokens(w->session);
        if (!live || live->len != (int)expected_tokens) {
            snprintf(err, err_len, "KV payload token count mismatch");
            ds4_session_invalidate(w->session);
            ok = false;
        } else if (loaded_tokens) {
            ds4_tokens_free(loaded_tokens);
            ds4_tokens_copy(loaded_tokens, live);
        }
        if (meta_out) {
            agent_kv_session_meta_free(meta_out);
            meta_out->has_title_trailer = has_title;
            meta_out->legacy_identity = !has_title;
            meta_out->created_at = hdr.created_at;
            agent_kv_identity_sha(&hdr, text, text_bytes, title, meta_out->sha);
            meta_out->title = has_title ?
                xstrdup(title) :
                agent_session_title_from_text(text, text_bytes, 0);
        }
    }
    free(title);
    free(text);
    return ok;
}

/* Save the current live KV under the rendered transcript identity.  The caller
 * decides the policy: fixed sysprompt path or SHA-named session path. */
static bool agent_kv_save_path(agent_worker *w, const char *path,
                               const ds4_tokens *tokens,
                               const char *reason,
                               char sha_out[41],
                               const char *session_title,
                               uint64_t session_created_at,
                               char *err, size_t err_len) {
    const ds4_tokens *live = ds4_session_tokens(w->session);
    if (!agent_tokens_equal(live, tokens)) {
        snprintf(err, err_len, "live KV state does not match session transcript");
        return false;
    }
    const int quant_bits = ds4_engine_routed_quant_bits(w->engine);
    if (quant_bits != 2 && quant_bits != 4) {
        snprintf(err, err_len, "unsupported routed quantization for KV save");
        return false;
    }
    const int model_id = ds4_engine_model_id(w->engine);

    size_t text_len = 0;
    char *text = ds4_kvstore_render_tokens_text(w->engine, tokens, &text_len);
    if (!text) {
        snprintf(err, err_len, "failed to render KV text key");
        return false;
    }
    if (text_len > UINT32_MAX) {
        snprintf(err, err_len, "rendered KV text key is too large");
        free(text);
        return false;
    }
    const bool session_identity = session_title != NULL;
    uint64_t now = (uint64_t)time(NULL);
    uint64_t created_at = session_identity && session_created_at ?
        session_created_at : now;
    char sha[41];
    if (session_identity)
        agent_session_identity_sha(session_title, created_at, sha);
    else
        ds4_kvstore_sha1_bytes_hex(text, text_len, sha);
    if (sha_out) memcpy(sha_out, sha, sizeof(sha));

    ds4_session_payload_file staged = {0};
    char save_err[160] = {0};
    if (ds4_session_stage_payload(w->session, &staged,
                                  save_err, sizeof(save_err)) != 0) {
        snprintf(err, err_len, "%s",
                 save_err[0] ? save_err : "session has no valid KV payload");
        free(text);
        return false;
    }
    uint64_t payload_bytes = staged.bytes;

    agent_buf tmpl = {0};
    agent_buf_puts(&tmpl, path);
    agent_buf_puts(&tmpl, ".tmp.XXXXXX");
    char *tmp = agent_buf_take(&tmpl);
    int fd = mkstemp(tmp);
    if (fd < 0) {
        snprintf(err, err_len, "%s", strerror(errno));
        ds4_session_payload_file_free(&staged);
        free(tmp);
        free(text);
        return false;
    }

    FILE *fp = fdopen(fd, "wb");
    if (!fp) {
        snprintf(err, err_len, "%s", strerror(errno));
        close(fd);
        unlink(tmp);
        ds4_session_payload_file_free(&staged);
        free(tmp);
        free(text);
        return false;
    }

    uint8_t h[DS4_KVSTORE_FIXED_HEADER];
    ds4_kvstore_fill_header(h, (uint8_t)model_id, (uint8_t)quant_bits,
                            ds4_kvstore_reason_code(reason),
                            session_identity ? DS4_KVSTORE_EXT_SESSION_TITLE : 0,
                            (uint32_t)tokens->len, 0,
                            (uint32_t)ds4_session_ctx(w->session),
                            created_at, now, payload_bytes);
    uint8_t tb[4];
    ds4_kvstore_le_put32(tb, (uint32_t)text_len);

    errno = 0;
    bool ok = fwrite(h, 1, sizeof(h), fp) == sizeof(h) &&
              fwrite(tb, 1, sizeof(tb), fp) == sizeof(tb) &&
              fwrite(text, 1, text_len, fp) == text_len &&
              ds4_session_write_staged_payload(&staged, fp,
                                               save_err, sizeof(save_err)) == 0 &&
              (!session_identity ||
               agent_kv_write_title_trailer(fp, session_title,
                                            save_err, sizeof(save_err))) &&
              fflush(fp) == 0;
    int saved_errno = errno;
    if (fclose(fp) != 0) {
        if (!saved_errno) saved_errno = errno;
        ok = false;
    }
    if (ok && rename(tmp, path) != 0) {
        saved_errno = errno;
        ok = false;
    }
    if (!ok) {
        snprintf(err, err_len, "%s",
                 saved_errno ? strerror(saved_errno) :
                 (save_err[0] ? save_err : "failed to write KV file"));
        unlink(tmp);
    }

    ds4_session_payload_file_free(&staged);
    free(tmp);
    free(text);
    return ok;
}

static void agent_worker_build_system_tokens(agent_worker *w, ds4_tokens *out) {
    ds4_chat_begin(w->engine, out);
    ds4_think_mode think_mode = effective_think_mode(w->cfg);
    if (agent_tool_syntax_for_engine(w->engine) == AGENT_TOOL_SYNTAX_GLM) {
        const char *effort = ds4_glm_reasoning_effort_text(think_mode);
        if (effort) ds4_chat_append_message(w->engine, out, "system", effort);
    } else if (w->cfg->gen.think_mode == DS4_THINK_MAX &&
               think_mode == DS4_THINK_MAX) {
        ds4_chat_append_max_effort_prefix(w->engine, out);
    }
    agent_append_system_prompt(w->engine, out, w->cfg->gen.system);
}

static void agent_publish_system_status(agent_worker *w, const char *msg) {
    if (w->cfg->non_interactive) return;
    if (isatty(STDOUT_FILENO)) {
        static const char marker[] = "\x1b[33m✦ \x1b[38;5;218m";
        agent_publish(w, marker, sizeof(marker) - 1);
        agent_publish(w, msg, strlen(msg));
        agent_publish(w, "\x1b[0m\n", strlen("\x1b[0m\n"));
    } else {
        agent_publish(w, "✦ ", strlen("✦ "));
        agent_publish(w, msg, strlen(msg));
        agent_publish(w, "\n", 1);
    }
}

static void agent_publishf_system_status(agent_worker *w, const char *fmt, ...) {
    char stack[1024];
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(stack, sizeof(stack), fmt, ap);
    va_end(ap);
    if (n <= 0) return;
    if ((size_t)n < sizeof(stack)) {
        agent_publish_system_status(w, stack);
        return;
    }

    char *heap = xmalloc((size_t)n + 1);
    va_start(ap, fmt);
    vsnprintf(heap, (size_t)n + 1, fmt, ap);
    va_end(ap);
    agent_publish_system_status(w, heap);
    free(heap);
}

static int agent_web_confirm(void *privdata, const char *message,
                             char *err, size_t err_len) {
    agent_worker *w = privdata;
    if (!w || w->cfg->non_interactive) {
        snprintf(err, err_len,
                 "visible Chrome browser startup requires interactive approval");
        return 0;
    }

    pthread_mutex_lock(&w->mu);
    w->web_approval_pending = true;
    w->web_approval_answered = false;
    w->web_approval_result = false;
    w->web_approval_error[0] = '\0';
    snprintf(w->web_approval_message, sizeof(w->web_approval_message),
             "%s", message ? message : "Start visible Chrome browser? (y/n) ");
    agent_wake_locked(w);
    while (!w->stop && !w->interrupt && !w->web_approval_answered)
        pthread_cond_wait(&w->cond, &w->mu);
    bool ok = w->web_approval_result;
    if (!w->web_approval_answered && (w->stop || w->interrupt)) {
        ok = false;
        w->web_approval_pending = false;
        snprintf(w->web_approval_error, sizeof(w->web_approval_error),
                 "interrupted");
    }
    if (!ok) {
        snprintf(err, err_len, "%s",
                 w->web_approval_error[0] ? w->web_approval_error :
                 "user denied Chrome browser start");
    }
    pthread_mutex_unlock(&w->mu);
    return ok ? 1 : 0;
}

static void agent_web_log(void *privdata, const char *message) {
    agent_worker *w = privdata;
    if (!w || !message || !message[0]) return;
    agent_trace(w, "web: %s", message);
}

static bool agent_web_cancel(void *privdata) {
    return worker_should_interrupt(privdata);
}

static bool worker_take_web_approval_request(agent_worker *w,
                                             char *message, size_t message_len) {
    pthread_mutex_lock(&w->mu);
    bool pending = w->web_approval_pending;
    if (pending) {
        snprintf(message, message_len, "%s", w->web_approval_message);
        w->web_approval_pending = false;
    }
    pthread_mutex_unlock(&w->mu);
    return pending;
}

static void worker_answer_web_approval(agent_worker *w, bool allow,
                                       const char *deny_error) {
    pthread_mutex_lock(&w->mu);
    w->web_approval_result = allow;
    w->web_approval_answered = true;
    if (!allow)
        snprintf(w->web_approval_error, sizeof(w->web_approval_error),
                 "%s", deny_error && deny_error[0] ? deny_error :
                 "user denied Chrome browser start");
    pthread_cond_signal(&w->cond);
    agent_wake_locked(w);
    pthread_mutex_unlock(&w->mu);
}

/* When a model turn finishes with a tool call, queued user messages should not
 * preempt that tool.  The worker asks the UI thread for the queue contents only
 * after the tool result is appended, so the next model input can contain both
 * the tool observation and the user's pending correction. */
static char *worker_request_queued_user_drain(agent_worker *w) {
    pthread_mutex_lock(&w->mu);
    w->queued_user_drain_pending = true;
    w->queued_user_drain_answered = false;
    free(w->queued_user_drain_text);
    w->queued_user_drain_text = NULL;
    agent_wake_locked(w);
    pthread_cond_signal(&w->cond);
    while (!w->stop && !w->queued_user_drain_answered)
        pthread_cond_wait(&w->cond, &w->mu);
    char *text = w->queued_user_drain_text;
    w->queued_user_drain_text = NULL;
    w->queued_user_drain_pending = false;
    w->queued_user_drain_answered = false;
    pthread_mutex_unlock(&w->mu);
    return text;
}

static bool worker_take_queued_user_drain_request(agent_worker *w) {
    pthread_mutex_lock(&w->mu);
    bool pending = w->queued_user_drain_pending;
    if (pending) w->queued_user_drain_pending = false;
    pthread_mutex_unlock(&w->mu);
    return pending;
}

static void worker_answer_queued_user_drain(agent_worker *w, char *text) {
    pthread_mutex_lock(&w->mu);
    free(w->queued_user_drain_text);
    w->queued_user_drain_text = text;
    w->queued_user_drain_answered = true;
    pthread_cond_signal(&w->cond);
    agent_wake_locked(w);
    pthread_mutex_unlock(&w->mu);
}

/* Synchronize the live DS4 session to a transcript.  This is the agent's main
 * cache-saving operation: if the requested transcript extends the live session,
 * only the suffix is prefetched; otherwise the DS4 session rebuilds from the
 * longest common prefix it can retain. */
static int agent_worker_sync_tokens(agent_worker *w, const ds4_tokens *tokens,
                                    bool publish_progress,
                                    char *err, size_t err_len) {
    int old_pos = ds4_session_pos(w->session);
    int common = ds4_session_common_prefix(w->session, tokens);
    int cached = common == old_pos && tokens->len >= old_pos ? common : 0;
    int suffix = tokens->len - cached;
    if (suffix < 0) suffix = tokens->len;

    if (publish_progress) {
        pthread_mutex_lock(&w->mu);
        unsigned prefill_label = w->status.state == AGENT_WORKER_PREFILL ?
            w->status.prefill_label : agent_next_prefill_label();
        w->status.state = AGENT_WORKER_PREFILL;
        w->progress_base = cached;
        w->progress_direct = false;
        w->progress_started_at = now_sec();
        w->status.prefill_done = 0;
        w->status.prefill_total = suffix;
        w->status.prefill_label = prefill_label;
        w->status.prefill_tps = 0.0;
        w->status.generated = 0;
        w->status.gen_tps = 0.0;
        agent_wake_locked(w);
        pthread_mutex_unlock(&w->mu);
    }

    ds4_session_set_progress(w->session, publish_progress ? worker_progress_cb : NULL,
                             publish_progress ? w : NULL);
    ds4_session_set_display_progress(w->session,
                                     publish_progress ? worker_progress_cb : NULL,
                                     publish_progress ? w : NULL);
    ds4_session_set_cancel(w->session, worker_cancel_session_cb, w);
    double t_sync0 = now_sec();
    int rc = ds4_session_sync(w->session, tokens, err, err_len);
    double t_sync1 = now_sec();
    ds4_session_set_cancel(w->session, NULL, NULL);
    ds4_session_set_progress(w->session, NULL, NULL);
    ds4_session_set_display_progress(w->session, NULL, NULL);
    agent_trace(w, "prefill sync done prompt=%d cached=%d suffix=%d rc=%d %.3f ms",
                tokens->len, cached, suffix, rc, (t_sync1 - t_sync0) * 1000.0);
    return rc;
}

/* Start a new session at the system/tool prompt.  A fixed sysprompt.kv
 * checkpoint avoids paying this prefill cost repeatedly, but only when the
 * rendered prompt text still matches the file.  The same fixed path is shared
 * by Flash and Pro; agent_kv_load_path() checks the model id, so switching
 * model families rebuilds this cache instead of restoring incompatible KV. */
static bool agent_worker_reset_to_sysprompt(agent_worker *w, char *err, size_t err_len) {
    ds4_tokens sys = {0};
    agent_worker_build_system_tokens(w, &sys);

    size_t text_len = 0;
    char *text = ds4_kvstore_render_tokens_text(w->engine, &sys, &text_len);
    if (!text) {
        snprintf(err, err_len, "failed to render system prompt");
        ds4_tokens_free(&sys);
        return false;
    }

    bool loaded = false;
    char load_err[160] = {0};
    if (w->sysprompt_path) {
        loaded = agent_kv_load_path(w, w->sysprompt_path, NULL,
                                    text, text_len, &w->transcript,
                                    NULL,
                                    load_err, sizeof(load_err));
        if (loaded) {
            agent_trace(w, "sysprompt kv hit file=%s tokens=%d",
                        w->sysprompt_path, w->transcript.len);
        }
    }

    if (!loaded) {
        if (w->sysprompt_path)
            agent_publish_system_status(w, "Updating system prompt cache...");
        ds4_tokens_free(&w->transcript);
        ds4_tokens_copy(&w->transcript, &sys);
        if (agent_worker_sync_tokens(w, &w->transcript, true, err, err_len) != 0) {
            free(text);
            ds4_tokens_free(&sys);
            return false;
        }
        if (w->sysprompt_path) {
            char save_err[160] = {0};
            char ignored_sha[41];
            if (!agent_kv_save_path(w, w->sysprompt_path, &w->transcript,
                                    "agent-system", ignored_sha,
                                    NULL, 0,
                                    save_err, sizeof(save_err)))
            {
                if (w->cfg->non_interactive) {
                    fprintf(stderr, "ds4-agent: failed to save system prompt KV: %s\n",
                            save_err);
                } else {
                    agent_buf b = {0};
                    agent_buf_puts(&b, "\nds4-agent: failed to save system prompt KV: ");
                    agent_buf_puts(&b, save_err);
                    agent_buf_puts(&b, "\n");
                    char *msg = agent_buf_take(&b);
                    agent_publish(w, msg, strlen(msg));
                    free(msg);
                }
            } else {
                agent_trace(w, "sysprompt kv stored file=%s tokens=%d",
                            w->sysprompt_path, w->transcript.len);
            }
        }
    }

    agent_worker_note_system_prompt_seen(w);
    pthread_mutex_lock(&w->mu);
    w->user_activity = false;
    w->session_dirty = false;
    w->status.state = AGENT_WORKER_IDLE;
    w->progress_direct = false;
    w->status.prefill_done = 0;
    w->status.prefill_total = 0;
    w->status.prefill_tps = 0.0;
    w->status.generated = 0;
    w->status.gen_tps = 0.0;
    w->status.greedy_sampling = false;
    w->status.error[0] = '\0';
    agent_wake_locked(w);
    pthread_mutex_unlock(&w->mu);
    w->datetime_context_injected = false;
    agent_worker_clear_session_identity(w);
    free(text);
    ds4_tokens_free(&sys);
    return true;
}

static bool agent_worker_should_stop(agent_worker *w) {
    pthread_mutex_lock(&w->mu);
    bool stop = w->stop;
    pthread_mutex_unlock(&w->mu);
    return stop;
}

static bool agent_worker_wait_distributed_route(agent_worker *w, char *err, size_t err_len) {
    if (!w || !w->cfg ||
        w->cfg->engine.distributed.role != DS4_DISTRIBUTED_COORDINATOR)
        return true;

    char last[160] = {0};
    unsigned ticks = 0;
    const struct timespec delay = {0, 250000000L};
    for (;;) {
        int ready = ds4_session_distributed_route_ready(w->session, err, err_len);
        if (ready > 0) {
            if (ticks != 0) {
                if (w->cfg->non_interactive)
                    fprintf(stderr, "ds4-agent: distributed route ready\n");
                else
                    agent_publish_system_status(w, "Distributed route ready.");
            }
            if (err_len) err[0] = '\0';
            return true;
        }
        if (ready < 0) return false;

        const char *why = err && err[0] ? err : "route incomplete";
        if (strcmp(last, why) != 0 || (ticks % 20u) == 0) {
            if (w->cfg->non_interactive) {
                fprintf(stderr, "ds4-agent: waiting for distributed route: %s\n", why);
            } else {
                char msg[224];
                snprintf(msg, sizeof(msg), "Waiting for distributed route: %s", why);
                agent_publish_system_status(w, msg);
            }
            snprintf(last, sizeof(last), "%s", why);
        }
        if (agent_worker_should_stop(w)) {
            snprintf(err, err_len, "agent stopped while waiting for distributed route");
            return false;
        }
        nanosleep(&delay, NULL);
        ticks++;
    }
}

static bool agent_worker_has_user_session(agent_worker *w) {
    pthread_mutex_lock(&w->mu);
    bool yes = w->user_activity;
    pthread_mutex_unlock(&w->mu);
    return yes;
}

static bool agent_worker_needs_save(agent_worker *w) {
    pthread_mutex_lock(&w->mu);
    bool yes = w->user_activity && w->session_dirty;
    pthread_mutex_unlock(&w->mu);
    return yes;
}

/* Save the current session under its stable agent identity.  The worker owns
 * the live KV, so busy /save requests are deferred until a stable append-only
 * point and then executed by the worker thread. */
static bool agent_worker_save_session_now(agent_worker *w, char sha_out[41],
                                          int *tokens_out,
                                          char *err, size_t err_len) {
    if (!agent_worker_has_user_session(w)) {
        snprintf(err, err_len, "nothing to save");
        return false;
    }

    if (agent_worker_sync_tokens(w, &w->transcript, false, err, err_len) != 0)
        return false;
    if (!agent_mkdir_p(w->cache_dir)) {
        snprintf(err, err_len, "failed to create %s", w->cache_dir);
        return false;
    }

    size_t text_len = 0;
    char *text = ds4_kvstore_render_tokens_text(w->engine, &w->transcript,
                                                &text_len);
    if (!text) {
        snprintf(err, err_len, "failed to render session text");
        return false;
    }
    if (!w->session_title) {
        w->session_title = agent_session_title_from_text(text, text_len, 0);
    }
    if (w->session_created_at == 0)
        w->session_created_at = (uint64_t)time(NULL);

    char sha[41];
    agent_session_identity_sha(w->session_title, w->session_created_at, sha);
    char *path = agent_kv_path_for_sha(w->cache_dir, sha);

    bool ok = agent_kv_save_path(w, path, &w->transcript,
                                 "agent-session", sha_out,
                                 w->session_title, w->session_created_at,
                                 err, err_len);
    if (ok) {
        memcpy(w->session_sha, sha, sizeof(w->session_sha));
        if (w->legacy_session_path_to_delete &&
            strcmp(w->legacy_session_path_to_delete, path) != 0)
        {
            unlink(w->legacy_session_path_to_delete);
        }
        free(w->legacy_session_path_to_delete);
        w->legacy_session_path_to_delete = NULL;
        pthread_mutex_lock(&w->mu);
        w->session_dirty = false;
        agent_wake_locked(w);
        pthread_mutex_unlock(&w->mu);
        if (tokens_out) *tokens_out = w->transcript.len;
    }
    free(path);
    free(text);
    return ok;
}

static bool agent_worker_save_session(agent_worker *w, char *err, size_t err_len) {
    if (!worker_is_idle(w)) {
        snprintf(err, err_len, "model is busy");
        return false;
    }
    char sha[41];
    int tokens = 0;
    bool ok = agent_worker_save_session_now(w, sha, &tokens, err, err_len);
    if (ok) printf("saved session %.8s (%d tokens)\n", sha, tokens);
    return ok;
}

/* ============================================================================
 * Session Listing, History Rendering, And Completion
 * ============================================================================
 */

static void agent_format_age(uint64_t when, char *buf, size_t len) {
    uint64_t now = (uint64_t)time(NULL);
    uint64_t age = when && now > when ? now - when : 0;
    if (age < 60) snprintf(buf, len, "%llus ago", (unsigned long long)age);
    else if (age < 3600) snprintf(buf, len, "%llum ago", (unsigned long long)(age / 60));
    else if (age < 86400) snprintf(buf, len, "%lluh ago", (unsigned long long)(age / 3600));
    else snprintf(buf, len, "%llud ago", (unsigned long long)(age / 86400));
}

static char *agent_session_title_from_span(const char *p, const char *end,
                                           size_t max_bytes,
                                           const char *empty_title) {
    bool limited = max_bytes != 0;
    if (limited && max_bytes < 4) max_bytes = 4;
    while (p < end && isspace((unsigned char)*p)) p++;
    while (end > p && isspace((unsigned char)end[-1])) end--;

    agent_buf b = {0};
    bool space = false;
    bool truncated = false;
    for (const char *s = p; s < end; s++) {
        unsigned char c = (unsigned char)*s;
        if (isspace(c)) {
            space = b.len != 0;
            continue;
        }
        if (space && (!limited || b.len + 4 < max_bytes)) {
            agent_buf_puts(&b, " ");
            space = false;
        }
        if (limited && b.len + 4 > max_bytes) {
            truncated = true;
            break;
        }
        agent_buf_append(&b, s, 1);
    }
    if (truncated) agent_buf_puts(&b, "...");
    if (!b.ptr || !b.len) {
        free(b.ptr);
        return xstrdup(empty_title);
    }
    return agent_buf_take(&b);
}

static char *agent_session_title_from_prompt(const char *prompt,
                                             size_t max_bytes) {
    const char *p = prompt ? prompt : "";
    return agent_session_title_from_span(p, p + strlen(p), max_bytes,
                                         "(empty user prompt)");
}

/* Extract a human-readable title from the first user turn stored in the
 * rendered transcript.  max_bytes==0 means "full normalized title"; callers
 * that render to the terminal pass an explicit display budget. */
static char *agent_session_title_from_text(const char *text, size_t text_len,
                                           size_t max_bytes) {
    static const char user_mark[] = "<｜User｜>";
    static const char assistant_mark[] = "<｜Assistant｜>";
    const char *p = text ? strstr(text, user_mark) : NULL;
    if (!p) return xstrdup("(no user prompt)");
    p += strlen(user_mark);
    const char *end = text + text_len;
    const char *assistant = strstr(p, assistant_mark);
    const char *next_user = strstr(p, user_mark);
    if (assistant && assistant < end) end = assistant;
    if (next_user && next_user < end) end = next_user;
    return agent_session_title_from_span(p, end, max_bytes,
                                         "(empty user prompt)");
}

static char *agent_session_title_clip(const char *title, size_t max_bytes) {
    if (!title) return xstrdup("(no user prompt)");
    size_t len = strlen(title);
    if (max_bytes == 0 || len <= max_bytes) return xstrdup(title);
    if (max_bytes < 4) max_bytes = 4;
    agent_buf b = {0};
    agent_buf_append(&b, title, max_bytes - 3);
    agent_buf_puts(&b, "...");
    return agent_buf_take(&b);
}

static char *agent_session_title_from_file(const char *path, size_t max_bytes) {
    FILE *fp = fopen(path, "rb");
    if (!fp) return xstrdup("(unreadable session)");
    ds4_kvstore_entry hdr = {0};
    uint32_t text_bytes = 0;
    char *text = NULL;
    char *trailer_title = NULL;
    bool ok = ds4_kvstore_read_header(fp, &hdr, &text_bytes) &&
              agent_kv_read_text(fp, text_bytes, &text, NULL, 0);
    if (ok && (hdr.ext_flags & DS4_KVSTORE_EXT_SESSION_TITLE))
        ok = agent_kv_read_title_trailer(fp, &hdr, &trailer_title, NULL, 0);
    fclose(fp);
    char *title = ok ?
        (trailer_title ?
            agent_session_title_clip(trailer_title, max_bytes) :
            agent_session_title_from_text(text, text_bytes, max_bytes)) :
        xstrdup("(unreadable session)");
    free(trailer_title);
    free(text);
    return title;
}

#define AGENT_HISTORY_DEFAULT_TURNS 3
#define AGENT_HISTORY_MAX_TURNS 200
#define AGENT_HISTORY_ASSISTANT_MAX_LINES 80
#define AGENT_HISTORY_ASSISTANT_MAX_BYTES 12000

typedef enum {
    AGENT_HISTORY_MARK_NONE,
    AGENT_HISTORY_MARK_USER,
    AGENT_HISTORY_MARK_ASSISTANT,
    AGENT_HISTORY_MARK_EOS,
} agent_history_mark;

typedef struct {
    const char **v;
    agent_history_mark *mark;
    int len;
    int cap;
} agent_history_ptrs;

static void agent_history_ptrs_push(agent_history_ptrs *p, const char *s,
                                    agent_history_mark mark) {
    if (p->len == p->cap) {
        p->cap = p->cap ? p->cap * 2 : 16;
        p->v = xrealloc(p->v, (size_t)p->cap * sizeof(p->v[0]));
        p->mark = xrealloc(p->mark, (size_t)p->cap * sizeof(p->mark[0]));
    }
    p->v[p->len] = s;
    p->mark[p->len] = mark;
    p->len++;
}

static const char *agent_memmem(const char *hay, size_t hay_len,
                                const char *needle, size_t needle_len) {
    if (!needle_len) return hay;
    if (needle_len > hay_len) return NULL;
    const char first = needle[0];
    const char *end = hay + hay_len - needle_len + 1;
    for (const char *p = hay; p < end; p++) {
        if (*p == first && memcmp(p, needle, needle_len) == 0) return p;
    }
    return NULL;
}

static const char *agent_history_next_marker(const char *p, const char *end,
                                             agent_history_mark *mark,
                                             size_t *mark_len) {
    static const char user_mark[] = "<｜User｜>";
    static const char assistant_mark[] = "<｜Assistant｜>";
    static const char eos_mark[] = "<｜end▁of▁sentence｜>";
    const char *u = agent_memmem(p, (size_t)(end - p),
                                 user_mark, sizeof(user_mark) - 1);
    const char *a = agent_memmem(p, (size_t)(end - p),
                                 assistant_mark, sizeof(assistant_mark) - 1);
    const char *e = agent_memmem(p, (size_t)(end - p),
                                 eos_mark, sizeof(eos_mark) - 1);
    if (!u && !a && !e) return NULL;
    if (u && (!a || u < a) && (!e || u < e)) {
        if (mark) *mark = AGENT_HISTORY_MARK_USER;
        if (mark_len) *mark_len = sizeof(user_mark) - 1;
        return u;
    }
    if (a && (!e || a < e)) {
        if (mark) *mark = AGENT_HISTORY_MARK_ASSISTANT;
        if (mark_len) *mark_len = sizeof(assistant_mark) - 1;
        return a;
    }
    if (mark) *mark = AGENT_HISTORY_MARK_EOS;
    if (mark_len) *mark_len = sizeof(eos_mark) - 1;
    return e;
}

static void agent_history_trim(const char **p, const char **end) {
    while (*p < *end && isspace((unsigned char)**p)) (*p)++;
    while (*end > *p && isspace((unsigned char)(*end)[-1])) (*end)--;
}

static bool agent_history_has_prefix(const char *p, const char *end,
                                     const char *prefix) {
    size_t n = strlen(prefix);
    return (size_t)(end - p) >= n && memcmp(p, prefix, n) == 0;
}

/* Tool messages are rendered as user turns in the transcript.  Return the
 * inner payload for the current <tool_result> wrapper so /history skips these
 * pseudo-user turns and displays their content without leaking the wrapper. */
static bool agent_history_tool_result_payload(const char **p, const char **end) {
    const char *s = *p, *e = *end;
    agent_history_trim(&s, &e);

    const char *open = "<tool_result>";
    const char *close = "</tool_result>";
    const size_t open_len = strlen(open);
    const size_t close_len = strlen(close);
    if (!agent_history_has_prefix(s, e, open)) return false;

    s += open_len;
    if ((size_t)(e - s) >= close_len &&
        memcmp(e - close_len, close, close_len) == 0)
    {
        e -= close_len;
    }
    *p = s;
    *end = e;
    return true;
}

static bool agent_history_is_tool_user(const char *p, const char *end) {
    agent_history_trim(&p, &end);
    return agent_history_tool_result_payload(&p, &end) ||
           agent_history_has_prefix(p, end, "Tool:") ||
           agent_history_has_prefix(p, end, "Tool result");
}

static void agent_history_ptrs_free(agent_history_ptrs *p) {
    free(p->v);
    free(p->mark);
    memset(p, 0, sizeof(*p));
}

/* Find the oldest rendered-chat marker needed to show the last N user turns.
 * Tool-result pseudo-user turns are skipped while human turns exist, so
 * /history stays centered on the human conversation.  Compacted sessions can
 * legitimately have a tail made only of tool result turns; in that case we
 * fall back to recent tool/assistant events instead of showing an empty
 * history. */
static const char *agent_history_start_for_turns(const char *text, size_t len,
                                                 int user_turns,
                                                 bool *tool_only) {
    const char *end = text + len;
    agent_history_ptrs marks = {0};
    agent_history_ptrs users = {0};
    agent_history_ptrs all_users = {0};
    const char *p = text;
    while (p < end) {
        agent_history_mark mark = AGENT_HISTORY_MARK_NONE;
        size_t mark_len = 0;
        const char *m = agent_history_next_marker(p, end, &mark, &mark_len);
        if (!m) break;
        agent_history_ptrs_push(&marks, m, mark);
        const char *content = m + mark_len;
        agent_history_mark next_mark = AGENT_HISTORY_MARK_NONE;
        size_t next_len = 0;
        const char *next = agent_history_next_marker(content, end,
                                                     &next_mark, &next_len);
        const char *content_end = next ? next : end;
        if (mark == AGENT_HISTORY_MARK_USER) {
            agent_history_ptrs_push(&all_users, m, mark);
            if (!agent_history_is_tool_user(content, content_end))
                agent_history_ptrs_push(&users, m, mark);
        }
        p = content_end;
    }

    const char *start = end;
    if (tool_only) *tool_only = false;
    if (users.len > 0) {
        int idx = users.len - user_turns;
        if (idx < 0) idx = 0;
        start = users.v[idx];
    } else if (all_users.len > 0) {
        int idx = all_users.len - user_turns;
        if (idx < 0) idx = 0;
        start = all_users.v[idx];
        if (tool_only) *tool_only = true;

        /* Tool result messages are stored as user-role turns after the
         * assistant DSML stanza that produced them.  Include that preceding
         * assistant marker when it is still in the retained tail, otherwise
         * replay shows the result but hides the call that caused it. */
        for (int i = marks.len - 1; i >= 0; i--) {
            if (marks.v[i] >= start) continue;
            if (marks.mark[i] == AGENT_HISTORY_MARK_USER) break;
            if (marks.mark[i] == AGENT_HISTORY_MARK_ASSISTANT) {
                start = marks.v[i];
                break;
            }
        }
    }
    agent_history_ptrs_free(&marks);
    agent_history_ptrs_free(&users);
    agent_history_ptrs_free(&all_users);
    return start;
}

static bool agent_history_latest_compaction_summary(const char *text,
                                                    size_t len,
                                                    const char **sum_start,
                                                    const char **sum_end) {
    static const char start_mark[] =
        "[ds4-agent compacted earlier conversation. Durable task-state summary follows.]";
    static const char end_mark[] =
        "[End compacted summary. Recent conversation continues verbatim below.]";
    const char *end = text + len;
    const char *scan = text;
    const char *best_start = NULL;
    const char *best_end = NULL;
    while (scan < end) {
        const char *s = agent_memmem(scan, (size_t)(end - scan),
                                     start_mark, sizeof(start_mark) - 1);
        if (!s) break;
        const char *content = s + sizeof(start_mark) - 1;
        const char *e = agent_memmem(content, (size_t)(end - content),
                                     end_mark, sizeof(end_mark) - 1);
        if (!e) break;
        best_start = content;
        best_end = e;
        scan = e + sizeof(end_mark) - 1;
    }
    if (!best_start || !best_end) return false;
    agent_history_trim(&best_start, &best_end);
    if (best_start >= best_end) return false;
    if (sum_start) *sum_start = best_start;
    if (sum_end) *sum_end = best_end;
    return true;
}

static void agent_history_publish_limited(agent_worker *w, const char *p,
                                          const char *end, int max_lines,
                                          size_t max_bytes);

static void agent_history_render_compaction_summary(agent_worker *w,
                                                    const char *text,
                                                    size_t len) {
    const char *p = NULL, *end = NULL;
    if (!agent_history_latest_compaction_summary(text, len, &p, &end)) return;
    bool color = isatty(STDOUT_FILENO) != 0;
    if (color) {
        const char *s = "\n\x1b[1;95mCompacted Summary:\x1b[0m\n";
        agent_publish(w, s, strlen(s));
    } else {
        agent_publish(w, "\nCompacted Summary:\n",
                      strlen("\nCompacted Summary:\n"));
    }
    agent_history_publish_limited(w, p, end, 80, 12000);
}

static const char *agent_history_skip_utf8_continuation(const char *p,
                                                        const char *end) {
    while (p < end && (((unsigned char)*p) & 0xc0) == 0x80) p++;
    return p;
}

static const char *agent_history_tail_start(const char *p, const char *end,
                                            int max_lines, size_t max_bytes,
                                            bool *truncated) {
    *truncated = false;
    if (p >= end) return p;

    const char *start = p;
    size_t len = (size_t)(end - p);
    if (max_bytes && len > max_bytes) {
        start = end - max_bytes;
        *truncated = true;
    }

    if (max_lines > 0) {
        const char *scan = end;
        if (scan > p && scan[-1] == '\n') scan--;
        const char *line_start = p;
        int lines = 0;
        while (scan > p) {
            scan--;
            if (*scan == '\n' && ++lines == max_lines) {
                line_start = scan + 1;
                break;
            }
        }
        if (line_start > p) *truncated = true;
        if (line_start > start) start = line_start;
    }

    return agent_history_skip_utf8_continuation(start, end);
}

static void agent_history_publish_limited(agent_worker *w, const char *p,
                                          const char *end, int max_lines,
                                          size_t max_bytes) {
    bool truncated = false;
    const char *start = agent_history_tail_start(p, end, max_lines, max_bytes,
                                                 &truncated);
    if (truncated)
        agent_publish(w, "\n... earlier history truncated; showing tail ...\n",
                      strlen("\n... earlier history truncated; showing tail ...\n"));
    agent_publish(w, start, (size_t)(end - start));
    if (end > start && end[-1] != '\n') agent_publish(w, "\n", 1);
}

static void agent_history_render_assistant(agent_worker *w,
                                           const char *p, const char *end) {
    agent_history_trim(&p, &end);
    if (p >= end) return;
    bool source_truncated = false;
    (void)agent_history_tail_start(p, end,
                                   AGENT_HISTORY_ASSISTANT_MAX_LINES,
                                   AGENT_HISTORY_ASSISTANT_MAX_BYTES,
                                   &source_truncated);
    bool use_color = isatty(STDOUT_FILENO) != 0;
    agent_tail_capture tail = {
        .cap = source_truncated ? AGENT_HISTORY_ASSISTANT_MAX_BYTES : 0,
    };
    agent_token_renderer renderer = {
        .engine = w->engine,
        .worker = w,
        .format_thinking = true,
        /* History replay should look like the original live output: the user is
         * switching back to a session, not reading a different transcript
         * format.  Tool calls are still dry-rendered below, so replay never
         * executes tools or mutates transcript state. */
        .format_markdown = true,
        .use_color = use_color && !source_truncated,
        .last_output_newline = true,
        .capture = source_truncated ? &tail : NULL,
    };
    agent_tool_syntax tool_syntax = agent_tool_syntax_for_engine(w->engine);
    agent_dsml_parser dsml = {
        .syntax = tool_syntax,
        .state = AGENT_DSML_SEARCH,
    };
    agent_stream_renderer stream = {
        .renderer = &renderer,
        .parser = &dsml,
        .syntax = tool_syntax,
        .replay = true,
    };

    /* Dry-run replay: the same streaming projection hides DSML and renders
     * semantic tool lines, but no tool is executed and no transcript state is
     * changed.  The saved KV payload remains the only authority for resume. */
    agent_stream_text(&stream, p, (size_t)(end - p), true);
    renderer_finish(&renderer);
    agent_dsml_parser_free(&dsml);

    if (source_truncated) {
        size_t tail_len = 0;
        char *tail_text = agent_tail_capture_take(&tail, &tail_len);
        bool rendered_truncated = tail.total > tail_len;
        bool line_truncated = false;
        const char *tail_start =
            agent_history_tail_start(tail_text, tail_text + tail_len,
                                     AGENT_HISTORY_ASSISTANT_MAX_LINES,
                                     AGENT_HISTORY_ASSISTANT_MAX_BYTES,
                                     &line_truncated);
        if (use_color) agent_publish(w, "\x1b[90m", 5);
        agent_publish(w,
                      "\n... earlier assistant history truncated; showing tail ...\n",
                      strlen("\n... earlier assistant history truncated; showing tail ...\n"));
        (void)rendered_truncated;
        agent_publish(w, tail_start, (size_t)(tail_text + tail_len - tail_start));
        if (tail_len && tail_text[tail_len - 1] != '\n') agent_publish(w, "\n", 1);
        if (use_color) agent_publish(w, "\x1b[0m", 4);
        free(tail_text);
    }
}

/* Re-render saved transcript text for /history and /switch.  It intentionally
 * uses the same assistant/token renderer as live output, so restored history
 * looks like the original terminal stream instead of raw rendered-chat text. */
static void agent_history_render_text(agent_worker *w, const char *text,
                                      size_t len, int user_turns) {
    if (user_turns <= 0) return;
    if (user_turns > AGENT_HISTORY_MAX_TURNS)
        user_turns = AGENT_HISTORY_MAX_TURNS;

    const char *end = text + len;
    agent_history_render_compaction_summary(w, text, len);

    bool tool_only = false;
    const char *p = agent_history_start_for_turns(text, len, user_turns,
                                                  &tool_only);
    if (p >= end) {
        agent_publish(w, "\n(no user history)\n", strlen("\n(no user history)\n"));
        return;
    }

    bool color = isatty(STDOUT_FILENO) != 0;
    if (color) agent_publish(w, "\n\x1b[90m", strlen("\n\x1b[90m"));
    else agent_publish(w, "\n", 1);
    if (tool_only) {
        agent_publishf(w, "--- session history: recent tool/assistant events ---\n");
    } else {
        agent_publishf(w, "--- session history: last %d user turn%s ---\n",
                       user_turns, user_turns == 1 ? "" : "s");
    }
    if (color) agent_publish(w, "\x1b[0m", 4);

    while (p < end) {
        agent_history_mark mark = AGENT_HISTORY_MARK_NONE;
        size_t mark_len = 0;
        const char *m = agent_history_next_marker(p, end, &mark, &mark_len);
        if (!m) break;
        const char *content = m + mark_len;
        agent_history_mark next_mark = AGENT_HISTORY_MARK_NONE;
        size_t next_len = 0;
        const char *next = agent_history_next_marker(content, end,
                                                     &next_mark, &next_len);
        const char *content_end = next ? next : end;
        const char *tp = content, *te = content_end;
        agent_history_trim(&tp, &te);

        if (mark == AGENT_HISTORY_MARK_USER) {
            if (agent_history_is_tool_user(tp, te)) {
                const char *payload_start = tp;
                const char *payload_end = te;
                (void)agent_history_tool_result_payload(&payload_start,
                                                        &payload_end);
                if (color) {
                    const char *s = "\x1b[90mTool result:\n";
                    agent_publish(w, s, strlen(s));
                } else {
                    agent_publish(w, "Tool result:\n", strlen("Tool result:\n"));
                }
                agent_history_publish_limited(w, payload_start, payload_end,
                                              12, 3000);
                if (color) agent_publish(w, "\x1b[0m", 4);
            } else {
                if (color) {
                    const char *s = "\x1b[1;32mUser:\x1b[0m\n";
                    agent_publish(w, s, strlen(s));
                } else {
                    agent_publish(w, "User:\n", strlen("User:\n"));
                }
                agent_history_publish_limited(w, tp, te, 24, 6000);
            }
        } else if (mark == AGENT_HISTORY_MARK_ASSISTANT) {
            if (color) {
                const char *s = "\x1b[1;37mAssistant:\x1b[0m\n";
                agent_publish(w, s, strlen(s));
            } else {
                agent_publish(w, "Assistant:\n", strlen("Assistant:\n"));
            }
            agent_history_render_assistant(w, tp, te);
        }
        p = content_end;
    }

    if (color) {
        const char *s = "\x1b[90m--- end history ---\x1b[0m\n";
        agent_publish(w, s, strlen(s));
    } else {
        agent_publish(w, "--- end history ---\n", strlen("--- end history ---\n"));
    }
}

/* Render recent saved transcript text without mutating the live session. */
static bool agent_worker_show_history(agent_worker *w, int user_turns,
                                      char *err, size_t err_len) {
    if (!worker_is_idle(w)) {
        snprintf(err, err_len, "model is busy");
        return false;
    }
    size_t text_len = 0;
    char *text = ds4_kvstore_render_tokens_text(w->engine, &w->transcript,
                                                &text_len);
    if (!text) {
        snprintf(err, err_len, "failed to render session text");
        return false;
    }
    agent_history_render_text(w, text, text_len, user_turns);
    free(text);
    return true;
}

typedef struct {
    ds4_kvstore_entry entry;
    char *title;
} agent_session_list_item;

static int agent_session_list_cmp_recent(const void *a, const void *b) {
    const agent_session_list_item *sa = a, *sb = b;
    uint64_t ta = sa->entry.last_used ? sa->entry.last_used : sa->entry.created_at;
    uint64_t tb = sb->entry.last_used ? sb->entry.last_used : sb->entry.created_at;
    if (ta < tb) return 1;
    if (ta > tb) return -1;
    return strcmp(sa->entry.sha, sb->entry.sha);
}

static void agent_session_list_free(agent_session_list_item *v, int n) {
    for (int i = 0; i < n; i++) {
        ds4_kvstore_entry_free(&v[i].entry);
        free(v[i].title);
    }
    free(v);
}

static void agent_session_list_push(agent_session_list_item **v, int *len,
                                    int *cap, ds4_kvstore_entry entry,
                                    char *title) {
    if (*len == *cap) {
        *cap = *cap ? *cap * 2 : 16;
        *v = xrealloc(*v, (size_t)*cap * sizeof((*v)[0]));
    }
    (*v)[(*len)++] = (agent_session_list_item){
        .entry = entry,
        .title = title,
    };
}

/* Print resumable sessions from ~/.ds4/kvcache.  sysprompt.kv is intentionally
 * ignored because it is an implementation cache, not a user session. */
static void agent_worker_list_sessions(agent_worker *w) {
    DIR *d = opendir(w->cache_dir);
    if (!d) {
        printf("no sessions: %s\n", strerror(errno));
        return;
    }

    int cols = renderer_terminal_cols();
    size_t title_budget = cols > 16 ? (size_t)(cols - 12) : 20;
    if (title_budget > 160) title_budget = 160;

    agent_session_list_item *sessions = NULL;
    int sessions_len = 0, sessions_cap = 0;
    const uint8_t model_id = (uint8_t)ds4_engine_model_id(w->engine);
    struct dirent *de;
    while ((de = readdir(d)) != NULL) {
        char sha[41];
        if (!ds4_kvstore_sha_hex_name(de->d_name, sha)) continue;
        char *path = ds4_kvstore_path_join(w->cache_dir, de->d_name);
        ds4_kvstore_entry e = {0};
        if (ds4_kvstore_read_entry_file(path, sha, &e)) {
            if (e.model_id == model_id) {
                char *title = agent_session_title_from_file(path, title_budget);
                agent_session_list_push(&sessions, &sessions_len, &sessions_cap,
                                        e, title);
            } else {
                ds4_kvstore_entry_free(&e);
            }
        }
        free(path);
    }
    closedir(d);
    if (!sessions_len) {
        printf("no saved sessions\n");
        return;
    }

    qsort(sessions, (size_t)sessions_len, sizeof(sessions[0]),
          agent_session_list_cmp_recent);

    bool color = isatty(STDOUT_FILENO) != 0;
    const char *sha_on = color ? "\x1b[1;96m" : "";
    const char *title_on = color ? "\x1b[1;97m" : "";
    const char *help_on = color ? "\x1b[97m" : "";
    const char *dim = color ? "\x1b[90m" : "";
    const char *reset = color ? "\x1b[0m" : "";

    for (int i = 0; i < sessions_len; i++) {
        ds4_kvstore_entry *e = &sessions[i].entry;
        char age[32];
        agent_format_age(e->last_used ? e->last_used : e->created_at,
                         age, sizeof(age));
        printf("%s%.8s%s %s>%s %s%s%s\n",
               sha_on, e->sha, reset, dim, reset,
               title_on, sessions[i].title, reset);
        printf("         %s> %s, %u tokens, %.2f MB%s%s\n\n",
               dim, age, e->tokens,
               (double)e->file_size / (1024.0 * 1024.0),
               e->payload_bytes == 0 ? ", stripped" : "",
               reset);
    }
    printf("%sUse /switch <id> to select a session, /del <id> to remove, "
           "/strip <id> to strip KV cache.%s\n",
           help_on, reset);
    agent_session_list_free(sessions, sessions_len);
}

typedef struct {
    char sha[41];
    uint64_t last_used;
} agent_completion_session;

typedef struct {
    agent_completion_session *v;
    int len;
    int cap;
} agent_completion_sessions;

static void agent_completion_sessions_push(agent_completion_sessions *s,
                                           const char sha[41],
                                           uint64_t last_used) {
    if (s->len == s->cap) {
        s->cap = s->cap ? s->cap * 2 : 16;
        s->v = xrealloc(s->v, (size_t)s->cap * sizeof(s->v[0]));
    }
    memcpy(s->v[s->len].sha, sha, 41);
    s->v[s->len].last_used = last_used;
    s->len++;
}

static int agent_completion_session_cmp(const void *a, const void *b) {
    const agent_completion_session *sa = a, *sb = b;
    if (sa->last_used < sb->last_used) return 1;
    if (sa->last_used > sb->last_used) return -1;
    return strcmp(sa->sha, sb->sha);
}

/* Tab completion for /switch.  Suggestions are sorted by recent use and accept
 * either an empty prefix or any unambiguous hex prefix. */
static void agent_switch_completion_callback(const char *buf,
                                             linenoiseCompletions *lc) {
    agent_worker *w = agent_completion_worker;
    static const char cmd[] = "/switch";
    const size_t cmd_len = sizeof(cmd) - 1;
    if (!w || !buf || strncmp(buf, cmd, cmd_len) != 0) return;

    const char *p = buf + cmd_len;
    if (*p && *p != ' ' && *p != '\t') return;
    while (*p == ' ' || *p == '\t') p++;

    const char *prefix = p;
    size_t prefix_len = strlen(prefix);
    for (size_t i = 0; i < prefix_len; i++) {
        if (!isxdigit((unsigned char)prefix[i])) return;
    }
    if (prefix_len > 40) return;

    DIR *d = opendir(w->cache_dir);
    if (!d) return;

    agent_completion_sessions sessions = {0};
    const uint8_t model_id = (uint8_t)ds4_engine_model_id(w->engine);
    struct dirent *de;
    while ((de = readdir(d)) != NULL) {
        char sha[41];
        if (!ds4_kvstore_sha_hex_name(de->d_name, sha)) continue;
        if (prefix_len && strncasecmp(sha, prefix, prefix_len) != 0) continue;

        uint64_t last_used = 0;
        char *path = ds4_kvstore_path_join(w->cache_dir, de->d_name);
        ds4_kvstore_entry e = {0};
        if (ds4_kvstore_read_entry_file(path, sha, &e)) {
            if (e.model_id == model_id) last_used = e.last_used;
            else last_used = UINT64_MAX;
            ds4_kvstore_entry_free(&e);
        } else {
            last_used = UINT64_MAX;
        }
        free(path);
        if (last_used == UINT64_MAX) continue;
        agent_completion_sessions_push(&sessions, sha, last_used);
    }
    closedir(d);

    qsort(sessions.v, (size_t)sessions.len, sizeof(sessions.v[0]),
          agent_completion_session_cmp);
    for (int i = 0; i < sessions.len; i++) {
        char line[64];
        int sha_chars = prefix_len > 8 ? 40 : 8;
        snprintf(line, sizeof(line), "/switch %.*s",
                 sha_chars, sessions.v[i].sha);
        linenoiseAddCompletion(lc, line);
    }
    free(sessions.v);
}

/* Resolve a user-provided SHA prefix to exactly one saved session file. */
static bool agent_worker_find_session(agent_worker *w, const char *prefix,
                                      char sha_out[41], char **path_out,
                                      char *err, size_t err_len) {
    size_t plen = strlen(prefix);
    if (plen == 0 || plen > 40) {
        snprintf(err, err_len, "invalid session SHA prefix");
        return false;
    }
    for (size_t i = 0; i < plen; i++) {
        if (!isxdigit((unsigned char)prefix[i])) {
            snprintf(err, err_len, "invalid session SHA prefix");
            return false;
        }
    }

    DIR *d = opendir(w->cache_dir);
    if (!d) {
        snprintf(err, err_len, "%s", strerror(errno));
        return false;
    }
    int matches = 0;
    char match_sha[41] = {0};
    char *match_path = NULL;
    const uint8_t model_id = (uint8_t)ds4_engine_model_id(w->engine);
    struct dirent *de;
    while ((de = readdir(d)) != NULL) {
        char sha[41];
        if (!ds4_kvstore_sha_hex_name(de->d_name, sha)) continue;
        if (strncasecmp(sha, prefix, plen) != 0) continue;
        char *path = ds4_kvstore_path_join(w->cache_dir, de->d_name);
        ds4_kvstore_entry e = {0};
        bool same_model = ds4_kvstore_read_entry_file(path, sha, &e) &&
                          e.model_id == model_id;
        ds4_kvstore_entry_free(&e);
        if (!same_model) {
            free(path);
            continue;
        }
        matches++;
        if (matches == 1) {
            memcpy(match_sha, sha, sizeof(match_sha));
            match_path = path;
        } else {
            free(path);
        }
    }
    closedir(d);
    if (matches == 0) {
        snprintf(err, err_len, "no saved session matches %.40s", prefix);
        return false;
    }
    if (matches > 1) {
        snprintf(err, err_len, "session prefix %.40s is ambiguous", prefix);
        free(match_path);
        return false;
    }
    memcpy(sha_out, match_sha, 41);
    *path_out = match_path;
    return true;
}

static bool agent_worker_delete_session(agent_worker *w, const char *prefix,
                                        char sha_out[41],
                                        char *err, size_t err_len) {
    char sha[41];
    char *path = NULL;
    if (!agent_worker_find_session(w, prefix, sha, &path, err, err_len))
        return false;
    if (unlink(path) != 0) {
        snprintf(err, err_len, "%s", strerror(errno));
        free(path);
        return false;
    }
    if (sha_out) memcpy(sha_out, sha, 41);
    free(path);
    return true;
}

/* Strip the heavy backend payload from a saved session while preserving its
 * rendered transcript. Loading such a file later tokenizes the text and
 * rebuilds the live KV with a full prefill. */
static bool agent_worker_strip_session(agent_worker *w, const char *prefix,
                                       char sha_out[41],
                                       uint32_t *tokens_out,
                                       char *err, size_t err_len) {
    if (err && err_len) err[0] = '\0';
    char sha[41];
    char *path = NULL;
    if (!agent_worker_find_session(w, prefix, sha, &path, err, err_len))
        return false;

    FILE *fp = fopen(path, "rb");
    if (!fp) {
        snprintf(err, err_len, "%s", strerror(errno));
        free(path);
        return false;
    }

    ds4_kvstore_entry hdr = {0};
    uint32_t text_bytes = 0;
    char *text = NULL;
    char *title = NULL;
    bool ok = ds4_kvstore_read_header(fp, &hdr, &text_bytes) &&
              agent_kv_read_text(fp, text_bytes, &text, err, err_len);
    if (ok && (hdr.ext_flags & DS4_KVSTORE_EXT_SESSION_TITLE))
        ok = agent_kv_read_title_trailer(fp, &hdr, &title, err, err_len);
    fclose(fp);
    if (!ok) {
        if (!err[0]) snprintf(err, err_len, "failed to read session");
        free(title);
        free(text);
        free(path);
        return false;
    }

    char actual_sha[41];
    agent_kv_identity_sha(&hdr, text, text_bytes, title, actual_sha);
    if (strcmp(actual_sha, sha)) {
        snprintf(err, err_len, "cached session identity does not match file name");
        free(title);
        free(text);
        free(path);
        return false;
    }

    ds4_tokens stripped_tokens = {0};
    ds4_tokenize_rendered_chat(w->engine, text, &stripped_tokens);
    uint32_t stripped_token_count = (uint32_t)stripped_tokens.len;
    ds4_tokens_free(&stripped_tokens);

    agent_buf tmpl = {0};
    agent_buf_puts(&tmpl, path);
    agent_buf_puts(&tmpl, ".tmp.XXXXXX");
    char *tmp = agent_buf_take(&tmpl);
    int fd = mkstemp(tmp);
    if (fd < 0) {
        snprintf(err, err_len, "%s", strerror(errno));
        free(tmp);
        free(text);
        free(path);
        return false;
    }

    fp = fdopen(fd, "wb");
    if (!fp) {
        snprintf(err, err_len, "%s", strerror(errno));
        close(fd);
        unlink(tmp);
        free(tmp);
        free(text);
        free(path);
        return false;
    }

    uint8_t h[DS4_KVSTORE_FIXED_HEADER];
    uint64_t now = (uint64_t)time(NULL);
    ds4_kvstore_fill_header(h, hdr.model_id, hdr.quant_bits, hdr.reason, hdr.ext_flags,
                            stripped_token_count, hdr.hits, hdr.ctx_size,
                            hdr.created_at, now, 0);
    uint8_t tb[4];
    ds4_kvstore_le_put32(tb, text_bytes);

    errno = 0;
    ok = fwrite(h, 1, sizeof(h), fp) == sizeof(h) &&
         fwrite(tb, 1, sizeof(tb), fp) == sizeof(tb) &&
         fwrite(text, 1, text_bytes, fp) == text_bytes &&
         (!(hdr.ext_flags & DS4_KVSTORE_EXT_SESSION_TITLE) ||
          agent_kv_write_title_trailer(fp, title, err, err_len)) &&
         fflush(fp) == 0;
    int saved_errno = errno;
    if (fclose(fp) != 0) {
        if (!saved_errno) saved_errno = errno;
        ok = false;
    }
    if (ok && rename(tmp, path) != 0) {
        saved_errno = errno;
        ok = false;
    }
    if (!ok) {
        snprintf(err, err_len, "%s",
                 saved_errno ? strerror(saved_errno) : "failed to write stripped session");
        unlink(tmp);
    } else {
        if (sha_out) memcpy(sha_out, sha, 41);
        if (tokens_out) *tokens_out = stripped_token_count;
    }

    free(tmp);
    free(title);
    free(text);
    free(path);
    return ok;
}

/* Load a saved session KV into the live transcript and optionally replay recent
 * history for the human. */
static bool agent_worker_switch_session(agent_worker *w, const char *prefix,
                                        int history_turns,
                                        char *err, size_t err_len) {
    if (!worker_is_idle(w)) {
        snprintf(err, err_len, "model is busy");
        return false;
    }
    char sha[41];
    char *path = NULL;
    if (!agent_worker_find_session(w, prefix, sha, &path, err, err_len))
        return false;

    bool stripped = false;
    ds4_kvstore_entry entry = {0};
    if (ds4_kvstore_read_entry_file(path, sha, &entry)) {
        stripped = entry.payload_bytes == 0;
        ds4_kvstore_entry_free(&entry);
    }
    if (stripped) {
        printf("rebuilding stripped session %.8s from rendered text...\n", sha);
        fflush(stdout);
    }

    ds4_tokens loaded = {0};
    agent_kv_session_meta meta = {0};
    bool ok = agent_kv_load_path(w, path, sha, NULL, 0, &loaded, &meta,
                                 err, err_len);
    if (ok) {
        ds4_tokens_free(&w->transcript);
        w->transcript = loaded;
        free(w->session_title);
        w->session_title = meta.title ? xstrdup(meta.title) : xstrdup("(no user prompt)");
        w->session_created_at = meta.created_at ? meta.created_at : (uint64_t)time(NULL);
        memcpy(w->session_sha, sha, sizeof(w->session_sha));
        free(w->legacy_session_path_to_delete);
        w->legacy_session_path_to_delete = meta.legacy_identity ? xstrdup(path) : NULL;
        w->datetime_context_injected = true;
        pthread_mutex_lock(&w->mu);
        w->user_activity = true;
        w->session_dirty = false;
        w->status.state = AGENT_WORKER_IDLE;
        w->status.ctx_used = w->transcript.len;
        w->status.ctx_size = agent_worker_effective_ctx_size(w);
        w->status.prefill_tps = 0.0;
        w->status.greedy_sampling = false;
        w->status.error[0] = '\0';
        agent_wake_locked(w);
        pthread_mutex_unlock(&w->mu);
        printf("switched to session %.8s (%d tokens%s)\n",
               sha, w->transcript.len, stripped ? ", rebuilt from text" : "");
        if (history_turns > 0)
            (void)agent_worker_show_history(w, history_turns, err, err_len);
    } else {
        ds4_tokens_free(&loaded);
    }
    agent_kv_session_meta_free(&meta);
    free(path);
    return ok;
}

/* ============================================================================
 * Tool Argument Parsing And File Tool Helpers
 * ============================================================================
 */

static int agent_parse_timeout(const char *s) {
    if (!s || !s[0]) return 3600;
    char *end = NULL;
    double v = strtod(s, &end);
    if (end == s || v <= 0.0 || !isfinite(v)) return 3600;
    if (v < 1.0) v = 1.0;
    if (v > 24.0 * 3600.0) v = 24.0 * 3600.0;
    return (int)v;
}

static int agent_parse_int_default(const char *s, int def, int min, int max) {
    if (!s || !s[0]) return def;
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (end == s) return def;
    while (*end == ' ' || *end == '\t' || *end == '\r' || *end == '\n') end++;
    if (*end) return def;
    if (v < min) v = min;
    if (v > max) v = max;
    return (int)v;
}

static bool agent_parse_bool_default(const char *s, bool def) {
    if (!s || !s[0]) return def;
    if (!strcasecmp(s, "true") || !strcasecmp(s, "yes") || !strcmp(s, "1"))
        return true;
    if (!strcasecmp(s, "false") || !strcasecmp(s, "no") || !strcmp(s, "0"))
        return false;
    return def;
}

#define AGENT_FILE_MAX_BYTES (16*1024*1024)
#define AGENT_READ_DEFAULT_LINES_SMALL 120
#define AGENT_READ_DEFAULT_LINES_MEDIUM 240
#define AGENT_READ_DEFAULT_LINES_LARGE 500
#define AGENT_READ_SMALL_CONTEXT_MAX 8192
#define AGENT_READ_MEDIUM_CONTEXT_MAX 16384
#define AGENT_TOOL_RESULT_RESERVE_TOKENS 1024
#define AGENT_EDIT_UPTO_MIN_PREFIX_BYTES 64
#define AGENT_EDIT_UPTO_MIN_PREFIX_LINES 2
#define AGENT_COMPACT_SOFT_PERCENT 85
#define AGENT_COMPACT_MIN_FREE_TOKENS 8192
#define AGENT_COMPACT_TAIL_DIVISOR 10
#define AGENT_COMPACT_TAIL_CAP_TOKENS 50000
#define AGENT_COMPACT_SUMMARY_MAX_TOKENS 4096

typedef struct {
    size_t start;
    size_t content_end;
    size_t end;
} agent_line_span;

typedef struct {
    agent_line_span *v;
    int len;
    int cap;
} agent_line_spans;

static void agent_line_spans_free(agent_line_spans *spans) {
    free(spans->v);
    memset(spans, 0, sizeof(*spans));
}

static void agent_line_spans_push(agent_line_spans *spans, agent_line_span span) {
    if (spans->len == spans->cap) {
        spans->cap = spans->cap ? spans->cap * 2 : 128;
        spans->v = xrealloc(spans->v, (size_t)spans->cap * sizeof(spans->v[0]));
    }
    spans->v[spans->len++] = span;
}

/* Split a text buffer into line spans.  content_end excludes CR/LF so callers
 * can print or compare line content without newline spelling differences. */
static void agent_split_lines(const char *data, size_t len, agent_line_spans *spans) {
    size_t pos = 0;
    while (pos < len) {
        size_t start = pos;
        while (pos < len && data[pos] != '\n' && data[pos] != '\r') pos++;
        size_t content_end = pos;
        if (pos < len) {
            if (data[pos] == '\r' && pos + 1 < len && data[pos + 1] == '\n')
                pos += 2;
            else
                pos++;
        }
        agent_line_spans_push(spans, (agent_line_span){
            .start = start,
            .content_end = content_end,
            .end = pos,
        });
    }
}

static int agent_read_file_bytes(const char *path, char **data, size_t *len,
                                 char *err, size_t errlen) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        snprintf(err, errlen, "open %s: %s", path, strerror(errno));
        return -1;
    }
    char *buf = NULL;
    size_t used = 0, cap = 0;
    char tmp[8192];
    while (true) {
        size_t n = fread(tmp, 1, sizeof(tmp), fp);
        if (n) {
            if (used + n > AGENT_FILE_MAX_BYTES) {
                fclose(fp);
                free(buf);
                snprintf(err, errlen, "file too large: %s exceeds %d bytes",
                         path, AGENT_FILE_MAX_BYTES);
                return -1;
            }
            if (used + n + 1 > cap) {
                cap = cap ? cap * 2 : 8192;
                while (cap < used + n + 1) cap *= 2;
                buf = xrealloc(buf, cap);
            }
            memcpy(buf + used, tmp, n);
            used += n;
            buf[used] = '\0';
        }
        if (n < sizeof(tmp)) {
            if (ferror(fp)) {
                snprintf(err, errlen, "read %s: %s", path, strerror(errno));
                fclose(fp);
                free(buf);
                return -1;
            }
            break;
        }
    }
    fclose(fp);
    if (!buf) buf = xstrdup("");
    *data = buf;
    *len = used;
    return 0;
}

static int agent_line_for_offset(const agent_line_spans *spans, size_t offset) {
    if (!spans || spans->len <= 0) return 1;
    for (int i = 0; i < spans->len; i++) {
        if (offset < spans->v[i].end) return i + 1;
    }
    return spans->len;
}

static bool agent_old_new_line_effect(const char *old_data, size_t old_len,
                                      const char *new_data, size_t new_len,
                                      size_t edit_offset, size_t replaced_len,
                                      int *start_line, int *end_line,
                                      int *delta) {
    agent_line_spans old_spans = {0};
    agent_line_spans new_spans = {0};
    agent_split_lines(old_data, old_len, &old_spans);
    agent_split_lines(new_data, new_len, &new_spans);
    bool ok = old_spans.len > 0;
    if (ok) {
        size_t old_last = edit_offset;
        if (replaced_len > 0) old_last = edit_offset + replaced_len - 1;
        if (old_last >= old_len) old_last = old_len ? old_len - 1 : 0;
        if (start_line) *start_line = agent_line_for_offset(&old_spans, edit_offset);
        if (end_line) *end_line = agent_line_for_offset(&old_spans, old_last);
        if (delta) *delta = new_spans.len - old_spans.len;
    }
    agent_line_spans_free(&old_spans);
    agent_line_spans_free(&new_spans);
    return ok;
}

static void agent_edit_result_append_context(agent_buf *b,
                                             const char *path,
                                             const char *data, size_t len,
                                             int anchor_start,
                                             int anchor_end);

static char *agent_edit_result(const char *path,
                                       int start_line, int end_line, int delta,
                                       const char *new_data, size_t new_len,
                                       const char *kind) {
    agent_buf b = {0};
    char msg[PATH_MAX + 180];
    snprintf(msg, sizeof(msg), "Edited %s using %s\n", path, kind);
    agent_buf_puts(&b, msg);
    if (start_line > 0 && end_line >= start_line) {
        snprintf(msg, sizeof(msg),
                 "Touched old lines %d-%d; current post-edit context follows.\n",
                 start_line, end_line);
        agent_buf_puts(&b, msg);
        if (delta != 0) {
            snprintf(msg, sizeof(msg),
                     "Line shift: old lines after %d moved by %+d (old line %d is now line %d). Re-read before relying on old line numbers there.\n",
                     end_line, delta, end_line + 1, end_line + 1 + delta);
            agent_buf_puts(&b, msg);
        }
    }
    if (start_line > 0 && end_line >= start_line) {
        int new_anchor_end = end_line + delta;
        if (new_anchor_end < start_line) new_anchor_end = start_line;
        agent_edit_result_append_context(&b, path, new_data, new_len,
                                         start_line, new_anchor_end);
    }
    return agent_buf_take(&b);
}

static void agent_worker_set_more(agent_worker *w, const char *path,
                                  int next_line, bool bare) {
    snprintf(w->more_path, sizeof(w->more_path), "%s", path ? path : "");
    w->more_next_line = next_line;
    w->more_bare = bare;
    w->more_valid = path && path[0] && next_line > 0;
}

static bool agent_tool_result_fits_context(agent_worker *w, const char *result,
                                           int reserve_tokens,
                                           int *tokens_out) {
    ds4_tokens tmp = {0};
    ds4_tokens_copy(&tmp, &w->transcript);
    ds4_chat_append_message(w->engine, &tmp, "tool", result ? result : "");
    int tokens = tmp.len;
    ds4_tokens_free(&tmp);
    if (tokens_out) *tokens_out = tokens;
    int ctx = agent_worker_effective_ctx_size(w);
    return ctx > 0 && tokens + reserve_tokens < ctx;
}

static int agent_tool_result_reserve_tokens(agent_worker *w) {
    int ctx = agent_worker_effective_ctx_size(w);
    int reserve = AGENT_TOOL_RESULT_RESERVE_TOKENS;
    if (ctx > 0) {
        int proportional = ctx / 8;
        if (proportional < 16) proportional = 16;
        if (reserve > proportional) reserve = proportional;
    }
    return reserve;
}

static int agent_read_default_lines(agent_worker *w) {
    int ctx = agent_worker_effective_ctx_size(w);
    if (ctx > 0 && ctx <= AGENT_READ_SMALL_CONTEXT_MAX)
        return AGENT_READ_DEFAULT_LINES_SMALL;
    if (ctx > 0 && ctx <= AGENT_READ_MEDIUM_CONTEXT_MAX)
        return AGENT_READ_DEFAULT_LINES_MEDIUM;
    return AGENT_READ_DEFAULT_LINES_LARGE;
}

/* Read file text for the model.  Normal mode shows plain line numbers.  Raw
 * mode is reserved for cases where line decoration would corrupt the payload
 * being inspected. */
static char *agent_read_range(agent_worker *w, const char *path, int start_line,
                              int max_lines, bool whole_file, bool bare,
                              bool set_more) {
    char err[256];
    char *data = NULL;
    size_t len = 0;
    if (!path || !path[0]) return xstrdup("Tool error: read requires path\n");
    if (agent_read_file_bytes(path, &data, &len, err, sizeof(err)) != 0) {
        agent_buf b = {0};
        agent_buf_puts(&b, "Tool error: ");
        agent_buf_puts(&b, err);
        agent_buf_puts(&b, "\n");
        return agent_buf_take(&b);
    }

    agent_line_spans spans = {0};
    agent_split_lines(data, len, &spans);
    if (start_line < 1) start_line = 1;
    int start_idx = start_line - 1;
    if (start_idx > spans.len) start_idx = spans.len;
    if (whole_file) {
        max_lines = spans.len - start_idx;
    } else {
        if (max_lines <= 0) max_lines = agent_read_default_lines(w);
    }
    int end_idx = start_idx + max_lines;
    if (end_idx > spans.len) end_idx = spans.len;

    agent_buf out = {0};
    if (bare) {
        size_t start = start_idx < spans.len ? spans.v[start_idx].start : len;
        size_t end = end_idx > start_idx ? spans.v[end_idx - 1].end : start;
        agent_buf_append(&out, data + start, end - start);
        if (end > start && out.ptr[out.len - 1] != '\n') agent_buf_puts(&out, "\n");
        if (end_idx < spans.len) {
            char note[160];
            snprintf(note, sizeof(note),
                     "[Read truncated at line %d of %d. continue_offset=%d. "
                     "Call more with count=%d to read the next chunk.]\n",
                     end_idx, spans.len, end_idx + 1,
                     max_lines > 0 ? max_lines : agent_read_default_lines(w));
            agent_buf_puts(&out, note);
        }
    } else {
        char hdr[PATH_MAX + 160];
        if (end_idx < spans.len) {
            snprintf(hdr, sizeof(hdr),
                     "%s: lines %d-%d of %d; continue_offset=%d; "
                     "call more with count=%d to read the next chunk\n",
                     path, spans.len ? start_idx + 1 : 0, end_idx, spans.len,
                     end_idx + 1, max_lines > 0 ? max_lines : agent_read_default_lines(w));
        } else {
            snprintf(hdr, sizeof(hdr), "%s: lines %d-%d of %d\n",
                     path, spans.len ? start_idx + 1 : 0, end_idx, spans.len);
        }
        agent_buf_puts(&out, hdr);
        for (int i = start_idx; i < end_idx; i++) {
            agent_line_span sp = spans.v[i];
            char prefix[64];
            snprintf(prefix, sizeof(prefix), "%d ", i + 1);
            agent_buf_puts(&out, prefix);
            agent_buf_append(&out, data + sp.start, sp.content_end - sp.start);
            agent_buf_puts(&out, "\n");
        }
    }
    if (set_more) {
        if (end_idx < spans.len) agent_worker_set_more(w, path, end_idx + 1, bare);
        else agent_worker_set_more(w, NULL, 0, false);
    }
    agent_line_spans_free(&spans);
    free(data);
    return agent_buf_take(&out);
}

static char *agent_tool_read(agent_worker *w, const agent_tool_call *call) {
    const char *path = agent_tool_arg_value(call, "path");
    bool whole = agent_parse_bool_default(agent_tool_arg_value(call, "whole"), false);
    int start = agent_parse_int_default(agent_tool_arg_value(call, "start_line"),
                                        1, 1, INT_MAX);
    int count = agent_parse_int_default(agent_tool_arg_value(call, "max_lines"),
                                        agent_read_default_lines(w), 1, INT_MAX);
    bool raw = agent_parse_bool_default(agent_tool_arg_value(call, "raw"), false);
    return agent_read_range(w, path, start, count, whole, raw, true);
}

static char *agent_tool_more(agent_worker *w, const agent_tool_call *call) {
    int count = agent_parse_int_default(agent_tool_arg_value(call, "count"),
                                        agent_read_default_lines(w), 1, INT_MAX);
    if (!w->more_valid) return xstrdup("Tool error: no previous output to continue\n");
    return agent_read_range(w, w->more_path, w->more_next_line, count, false,
                            w->more_bare, true);
}

static char *agent_tool_write(agent_worker *w, const agent_tool_call *call) {
    (void)w;
    const char *path = agent_tool_arg_value(call, "path");
    const char *content = agent_tool_arg_value(call, "content");
    if (!path || !path[0]) return xstrdup("Tool error: write requires path\n");
    if (!content) return xstrdup("Tool error: write requires content\n");
    FILE *fp = fopen(path, "wb");
    if (!fp) {
        agent_buf b = {0};
        agent_buf_puts(&b, "Tool error: open for write failed: ");
        agent_buf_puts(&b, strerror(errno));
        agent_buf_puts(&b, "\n");
        return agent_buf_take(&b);
    }
    size_t len = strlen(content);
    size_t wr = fwrite(content, 1, len, fp);
    int close_rc = fclose(fp);
    if (wr != len || close_rc != 0) {
        agent_buf b = {0};
        agent_buf_puts(&b, "Tool error: write failed: ");
        agent_buf_puts(&b, strerror(errno));
        agent_buf_puts(&b, "\n");
        return agent_buf_take(&b);
    }
    char msg[PATH_MAX + 160];
    snprintf(msg, sizeof(msg), "Wrote %zu bytes to %s\n", len, path);
    return xstrdup(msg);
}

static char *agent_tool_list(const agent_tool_call *call) {
    const char *path = agent_tool_arg_value(call, "path");
    if (!path || !path[0]) path = ".";
    DIR *dir = opendir(path);
    if (!dir) {
        agent_buf b = {0};
        agent_buf_puts(&b, "Tool error: opendir failed: ");
        agent_buf_puts(&b, strerror(errno));
        agent_buf_puts(&b, "\n");
        return agent_buf_take(&b);
    }
    agent_buf out = {0};
    char hdr[PATH_MAX + 64];
    snprintf(hdr, sizeof(hdr), "%s:\n", path);
    agent_buf_puts(&out, hdr);
    struct dirent *de;
    int shown = 0;
    while ((de = readdir(dir)) != NULL && shown < 300) {
        if (!strcmp(de->d_name, ".") || !strcmp(de->d_name, "..")) continue;
        char full[PATH_MAX];
        snprintf(full, sizeof(full), "%s/%s", path, de->d_name);
        struct stat st;
        if (lstat(full, &st) != 0) continue;
        char type = S_ISDIR(st.st_mode) ? 'd' :
                    S_ISLNK(st.st_mode) ? 'l' :
                    S_ISREG(st.st_mode) ? '-' : '?';
        char line[PATH_MAX + 96];
        snprintf(line, sizeof(line), "%c %10lld %s%s\n", type,
                 (long long)st.st_size, de->d_name, S_ISDIR(st.st_mode) ? "/" : "");
        agent_buf_puts(&out, line);
        shown++;
    }
    if (de) agent_buf_puts(&out, "... more entries omitted ...\n");
    closedir(dir);
    return agent_buf_take(&out);
}

/* ============================================================================
 * Edit And Search Tools
 * ============================================================================
 */

static int agent_write_file_bytes(const char *path, const char *data, size_t len,
                                  char *err, size_t errlen) {
    FILE *fp = fopen(path, "wb");
    if (!fp) {
        snprintf(err, errlen, "open %s: %s", path, strerror(errno));
        return -1;
    }
    size_t wr = fwrite(data, 1, len, fp);
    if (wr != len) {
        snprintf(err, errlen, "write %s: %s", path, strerror(errno));
        fclose(fp);
        return -1;
    }
    if (fclose(fp) != 0) {
        snprintf(err, errlen, "close %s: %s", path, strerror(errno));
        return -1;
    }
    return 0;
}

static void agent_edit_result_append_line(agent_buf *b, const char *data,
                                          const agent_line_span *sp,
                                          int line) {
    char prefix[64];
    snprintf(prefix, sizeof(prefix), "%d ", line);
    agent_buf_puts(b, prefix);
    agent_buf_append(b, data + sp->start, sp->content_end - sp->start);
    agent_buf_puts(b, "\n");
}

/* Successful edits return the nearby post-edit file shape.  This spends cheap
 * prefill tokens to save expensive model retries: the model immediately sees
 * shifted line numbers, braces, semicolons, and accidental duplication. */
static void agent_edit_result_append_context(agent_buf *b,
                                             const char *path,
                                             const char *data, size_t len,
                                             int anchor_start,
                                             int anchor_end) {
    enum {
        CONTEXT_BEFORE = 5,
        CONTEXT_AFTER = 8,
        EDITED_CONTEXT_HEAD = 18,
        EDITED_CONTEXT_TAIL = 18
    };

    agent_line_spans spans = {0};
    agent_split_lines(data, len, &spans);
    if (spans.len <= 0) {
        agent_line_spans_free(&spans);
        return;
    }

    if (anchor_start < 1) anchor_start = 1;
    if (anchor_start > spans.len) anchor_start = spans.len;
    if (anchor_end < anchor_start) anchor_end = anchor_start;
    if (anchor_end > spans.len) anchor_end = spans.len;

    int ctx_start = anchor_start - CONTEXT_BEFORE;
    if (ctx_start < 1) ctx_start = 1;
    int ctx_end = anchor_end + CONTEXT_AFTER;
    if (ctx_end > spans.len) ctx_end = spans.len;

    char hdr[PATH_MAX + 160];
    snprintf(hdr, sizeof(hdr),
             "Current file around edit: %s lines %d-%d of %d\n",
             path, ctx_start, ctx_end, spans.len);
    agent_buf_puts(b, hdr);

    int edited_lines = anchor_end - anchor_start + 1;
    if (edited_lines <= EDITED_CONTEXT_HEAD + EDITED_CONTEXT_TAIL) {
        for (int line = ctx_start; line <= ctx_end; line++)
            agent_edit_result_append_line(b, data, &spans.v[line - 1], line);
    } else {
        int head_end = anchor_start + EDITED_CONTEXT_HEAD - 1;
        int tail_start = anchor_end - EDITED_CONTEXT_TAIL + 1;
        for (int line = ctx_start; line <= head_end; line++)
            agent_edit_result_append_line(b, data, &spans.v[line - 1], line);
        snprintf(hdr, sizeof(hdr),
                 "... %d edited lines omitted ...\n",
                 tail_start - head_end - 1);
        agent_buf_puts(b, hdr);
        for (int line = tail_start; line <= ctx_end; line++)
            agent_edit_result_append_line(b, data, &spans.v[line - 1], line);
    }

    agent_line_spans_free(&spans);
}

static const char *agent_memmem_simple(const char *hay, size_t hay_len,
                                       const char *needle, size_t needle_len) {
    if (!needle_len) return hay;
    if (needle_len > hay_len) return NULL;
    size_t last = hay_len - needle_len;
    for (size_t i = 0; i <= last; i++) {
        if (hay[i] == needle[0] && !memcmp(hay + i, needle, needle_len))
            return hay + i;
    }
    return NULL;
}

static bool agent_find_unique(const char *data, size_t len,
                              const char *needle, size_t needle_len,
                              const char **match, const char *label,
                              char *err, size_t err_len) {
    if (!needle || needle_len == 0) {
        snprintf(err, err_len, "%s anchor is empty", label);
        return false;
    }
    const char *first = agent_memmem_simple(data, len, needle, needle_len);
    if (!first) {
        snprintf(err, err_len, "%s anchor not found", label);
        return false;
    }
    size_t after_first = (size_t)(first - data) + 1;
    const char *second = after_first <= len ?
        agent_memmem_simple(data + after_first, len - after_first,
                            needle, needle_len) : NULL;
    if (second) {
        snprintf(err, err_len, "%s anchor is not unique", label);
        return false;
    }
    *match = first;
    return true;
}

/* Find an anchor only in the suffix after start.
 *
 * Anchored edits use "head [upto] tail": the head fixes the edit start, and
 * the tail should delimit the first unique end point after that start. A tail
 * may legitimately appear earlier in the file, so checking global uniqueness
 * would reject valid edits.
 */
static bool agent_find_unique_after(const char *data, size_t len,
                                    const char *start,
                                    const char *needle, size_t needle_len,
                                    const char **match, const char *label,
                                    char *err, size_t err_len) {
    if (!needle || needle_len == 0) {
        snprintf(err, err_len, "%s anchor is empty", label);
        return false;
    }
    if (start < data || start > data + len) {
        snprintf(err, err_len, "%s search starts outside file", label);
        return false;
    }
    size_t off = (size_t)(start - data);
    const char *first = agent_memmem_simple(data + off, len - off,
                                            needle, needle_len);
    if (!first) {
        snprintf(err, err_len, "%s anchor not found after old head", label);
        return false;
    }
    size_t after_first = (size_t)(first - data) + 1;
    const char *second = after_first <= len ?
        agent_memmem_simple(data + after_first, len - after_first,
                            needle, needle_len) : NULL;
    if (second) {
        snprintf(err, err_len, "%s anchor is not unique after old head", label);
        return false;
    }
    *match = first;
    return true;
}

static bool agent_edit_old_may_be_closing_tag(const char *text, size_t len) {
    size_t i = 0;
    while (i < len && (text[i] == ' ' || text[i] == '\t' ||
                       text[i] == '\r' || text[i] == '\n'))
        i++;
    return i < len && text[i] == '<';
}

static bool agent_edit_old_is_only_space(const char *text, size_t len) {
    for (size_t i = 0; i < len; i++) {
        if (text[i] != ' ' && text[i] != '\t' &&
            text[i] != '\r' && text[i] != '\n')
            return false;
    }
    return true;
}

static bool agent_span_has_nonspace(const char *s, size_t len) {
    for (size_t i = 0; i < len; i++) {
        if (!isspace((unsigned char)s[i])) return true;
    }
    return false;
}

static bool agent_edit_old_prefix_mature_for_upto(const char *old, size_t old_len) {
    if (old_len < AGENT_EDIT_UPTO_MIN_PREFIX_BYTES) return false;
    int nonempty_lines = 0;
    bool line_has_text = false;
    for (size_t i = 0; i < old_len; i++) {
        if (old[i] == '\n') {
            if (line_has_text) nonempty_lines++;
            line_has_text = false;
        } else if (!isspace((unsigned char)old[i])) {
            line_has_text = true;
        }
    }
    return nonempty_lines >= AGENT_EDIT_UPTO_MIN_PREFIX_LINES;
}

static bool agent_edit_old_ready_for_upto(const char *old, size_t old_len) {
    if (!old_len || strstr(old, "[upto]")) return false;
    if (!agent_edit_old_prefix_mature_for_upto(old, old_len)) return false;
    size_t end = old_len;
    while (end > 0 && (old[end - 1] == ' ' || old[end - 1] == '\t' ||
                       old[end - 1] == '\r'))
        end--;
    return end > 0 && old[end - 1] == '\n';
}

/* While the model streams an edit old=... argument, stop it from retyping a
 * large exact old block once the emitted prefix is already a unique file
 * anchor.  The next sampled token is inspected before eval: if it would keep
 * writing old text rather than close the parameter, the caller evaluates a
 * complete "[upto]" marker line instead. */
static bool agent_edit_upto_forcer_should_replace(agent_edit_upto_forcer *forcer,
                                                  agent_dsml_parser *p,
                                                  const char *next_text,
                                                  size_t next_len) {
    if (!forcer || !p) return false;
    bool in_edit_old = p->state == AGENT_DSML_PARAM_VALUE &&
        p->current.name && strcmp(p->current.name, "edit") == 0 &&
        p->param_name && strcmp(p->param_name, "old") == 0;
    if (!in_edit_old) {
        forcer->active = false;
        forcer->done = false;
        return false;
    }
    if (!forcer->active) {
        forcer->active = true;
        forcer->done = false;
    }
    if (forcer->done)
        return false;
    if (agent_edit_old_may_be_closing_tag(next_text, next_len) ||
        agent_edit_old_is_only_space(next_text, next_len))
        return false;

    const char *path = agent_tool_arg_value(&p->current, "path");
    if (!path || !path[0] || p->param_value_start > p->raw_len) return false;

    const char *old = p->raw + p->param_value_start;
    size_t old_len = p->raw_len - p->param_value_start;
    if (!agent_edit_old_ready_for_upto(old, old_len)) return false;

    char err[256];
    char *data = NULL;
    size_t len = 0;
    if (agent_read_file_bytes(path, &data, &len, err, sizeof(err)) != 0)
        return false;

    const char *match = NULL;
    bool unique = agent_find_unique(data, len, old, old_len, &match,
                                    "old prefix", err, sizeof(err));
    free(data);
    if (unique) {
        forcer->done = true;
        return true;
    }
    return false;
}

static bool agent_edit_find_old_span(const char *data, size_t len,
                                     const char *old, const char **match,
                                     size_t *match_len, bool *anchored,
                                     char *err, size_t err_len) {
    static const char marker[] = "[upto]";
    size_t old_len = strlen(old);
    const char *upto = strstr(old, marker);
    if (!upto) {
        *anchored = false;
        if (!agent_find_unique(data, len, old, old_len, match, "old text",
                               err, err_len))
            return false;
        *match_len = old_len;
        return true;
    }
    if (strstr(upto + strlen(marker), marker)) {
        snprintf(err, err_len, "old text contains more than one [upto] marker");
        return false;
    }
    size_t head_len = (size_t)(upto - old);
    const char *tail = upto + strlen(marker);
    size_t tail_len = old_len - head_len - strlen(marker);
    /* Strip leading newline/CR from tail before searching.  The head already
     * includes the newline at its end, so the extra \n that follows [upto] in
     * the old text (whether injected by the forcer or written by the model)
     * must not be part of the tail needle -- the file after the head has no
     * duplicate newline. */
    while (tail_len > 0 && (*tail == '\n' || *tail == '\r')) {
        tail++;
        tail_len--;
    }
    if (!agent_span_has_nonspace(tail, tail_len)) {
        snprintf(err, err_len,
                 "old text after [upto] must include a unique tail anchor");
        return false;
    }
    const char *head_pos = NULL;
    const char *tail_pos = NULL;
    if (!agent_find_unique(data, len, old, head_len, &head_pos, "old head",
                           err, err_len))
        return false;
    if (!agent_find_unique_after(data, len, head_pos + head_len,
                                 tail, tail_len, &tail_pos, "old tail",
                                 err, err_len))
        return false;
    *anchored = true;
    *match = head_pos;
    *match_len = (size_t)(tail_pos - head_pos) + tail_len;
    return true;
}

#ifdef DS4_AGENT_TEST
static int agent_test_failures;

static void agent_test_assert(bool cond, const char *expr,
                              const char *file, int line) {
    if (cond) return;
    fprintf(stderr, "%s:%d: assertion failed: %s\n", file, line, expr);
    agent_test_failures++;
}

#define AGENT_TEST_ASSERT(expr) \
    agent_test_assert((expr), #expr, __FILE__, __LINE__)

static void test_agent_edit_upto_tail_newline_is_not_part_of_anchor(void) {
    const char *data =
        "CFLAGS = -Wall -Wextra -g\n"
        "LDFLAGS =\n"
        "\n"
        "all: bc\n"
        "\n"
        "bc: main.c\n"
        "\t$(CC) $(CFLAGS) -o bc main.c $(LDFLAGS)\n"
        "\n"
        "clean:\n"
        "\trm -f bc\n";
    const char *old =
        "CFLAGS = -Wall -Wextra -g\n"
        "LDFLAGS =\n"
        "\n"
        "all: bc\n"
        "\n"
        "bc: main.c\n"
        "\t$(CC) $(CFLAGS) -o bc main.c $(LDFLAGS)\n"
        "\n"
        "[upto]\n"
        "clean:\n";

    const char *match = NULL;
    size_t match_len = 0;
    bool anchored = false;
    char err[128] = {0};
    AGENT_TEST_ASSERT(agent_edit_find_old_span(data, strlen(data), old,
                                              &match, &match_len, &anchored,
                                              err, sizeof(err)));
    AGENT_TEST_ASSERT(anchored);
    AGENT_TEST_ASSERT(match == data);
    AGENT_TEST_ASSERT(match_len == strlen(data) - strlen("\trm -f bc\n"));
}

static void test_agent_edit_upto_requires_tail_after_newline_strip(void) {
    const char *data = "head\nbody\ntail\n";
    const char *old = "head\n[upto]\n";
    const char *match = NULL;
    size_t match_len = 0;
    bool anchored = false;
    char err[128] = {0};

    AGENT_TEST_ASSERT(!agent_edit_find_old_span(data, strlen(data), old,
                                               &match, &match_len, &anchored,
                                               err, sizeof(err)));
    AGENT_TEST_ASSERT(strstr(err, "must include a unique tail anchor") != NULL);
}

static char *agent_test_stream_capture(agent_tool_syntax syntax,
                                       const char **chunks,
                                       size_t chunk_count,
                                       agent_dsml_parser *p,
                                       bool *dsml_in_think) {
    agent_tail_capture tail = {
        .cap = 16384,
    };
    agent_token_renderer renderer = {
        .format_thinking = true,
        .format_markdown = false,
        .last_output_newline = true,
        .capture = &tail,
    };
    *p = (agent_dsml_parser){
        .syntax = syntax,
        .state = AGENT_DSML_SEARCH,
    };
    agent_stream_renderer stream = {
        .renderer = &renderer,
        .parser = p,
        .syntax = syntax,
    };

    for (size_t i = 0; i < chunk_count; i++)
        agent_stream_text(&stream, chunks[i], strlen(chunks[i]), false);
    agent_stream_text(&stream, NULL, 0, true);
    renderer_finish(&renderer);
    if (dsml_in_think) *dsml_in_think = stream.dsml_in_think;
    return agent_tail_capture_take(&tail, NULL);
}

static void test_agent_glm_tool_parser_single_arg(void) {
    const char *text =
        "prose before <tool_call>list"
        "<arg_key>path</arg_key><arg_value>.</arg_value>"
        "</tool_call>";
    agent_dsml_parser p = {
        .syntax = AGENT_TOOL_SYNTAX_GLM,
        .state = AGENT_DSML_SEARCH,
    };

    agent_dsml_feed(&p, text, strlen(text));
    agent_dsml_finish(&p);

    AGENT_TEST_ASSERT(p.state == AGENT_DSML_DONE);
    AGENT_TEST_ASSERT(p.calls.len == 1);
    AGENT_TEST_ASSERT(p.calls.v[0].name && !strcmp(p.calls.v[0].name, "list"));
    AGENT_TEST_ASSERT(p.calls.v[0].argc == 1);
    AGENT_TEST_ASSERT(p.calls.v[0].args[0].name &&
                      !strcmp(p.calls.v[0].args[0].name, "path"));
    AGENT_TEST_ASSERT(p.calls.v[0].args[0].value &&
                      !strcmp(p.calls.v[0].args[0].value, "."));

    agent_dsml_parser_free(&p);
}

static void test_agent_glm_tool_parser_chunked_multi_arg(void) {
    const char *a = "<tool_call>bash<arg_key>command</arg_key>";
    const char *b = "<arg_value>printf hi</arg_value>";
    const char *c =
        "<arg_key>refresh_sec</arg_key><arg_value>1</arg_value></tool_call>";
    agent_dsml_parser p = {
        .syntax = AGENT_TOOL_SYNTAX_GLM,
        .state = AGENT_DSML_SEARCH,
    };

    agent_dsml_feed(&p, a, strlen(a));
    AGENT_TEST_ASSERT(p.state == AGENT_DSML_STRUCTURAL);
    agent_dsml_feed(&p, b, strlen(b));
    AGENT_TEST_ASSERT(p.state == AGENT_DSML_STRUCTURAL);
    agent_dsml_feed(&p, c, strlen(c));
    agent_dsml_finish(&p);

    AGENT_TEST_ASSERT(p.state == AGENT_DSML_DONE);
    AGENT_TEST_ASSERT(p.calls.len == 1);
    AGENT_TEST_ASSERT(p.calls.v[0].name && !strcmp(p.calls.v[0].name, "bash"));
    AGENT_TEST_ASSERT(p.calls.v[0].argc == 2);
    AGENT_TEST_ASSERT(!strcmp(agent_tool_arg_value(&p.calls.v[0], "command"),
                              "printf hi"));
    AGENT_TEST_ASSERT(!strcmp(agent_tool_arg_value(&p.calls.v[0], "refresh_sec"),
                              "1"));

    agent_dsml_parser_free(&p);
}

static void test_agent_glm_tool_parser_streams_param_state(void) {
    const char *a =
        "<tool_call>bash<arg_key>command</arg_key><arg_value>printf hi";
    const char *b = "</arg";
    const char *c =
        "_value><arg_key>refresh_sec</arg_key><arg_value>1</arg_value></tool_call>";
    agent_dsml_parser p = {
        .syntax = AGENT_TOOL_SYNTAX_GLM,
        .state = AGENT_DSML_SEARCH,
    };

    agent_dsml_feed(&p, a, strlen(a));
    AGENT_TEST_ASSERT(p.state == AGENT_DSML_PARAM_VALUE);
    AGENT_TEST_ASSERT(p.current.name && !strcmp(p.current.name, "bash"));
    AGENT_TEST_ASSERT(p.param_name && !strcmp(p.param_name, "command"));
    AGENT_TEST_ASSERT(!p.param_close_prefix);

    agent_dsml_feed(&p, b, strlen(b));
    AGENT_TEST_ASSERT(p.state == AGENT_DSML_PARAM_VALUE);
    AGENT_TEST_ASSERT(p.param_close_prefix);

    agent_dsml_feed(&p, c, strlen(c));
    agent_dsml_finish(&p);
    AGENT_TEST_ASSERT(p.state == AGENT_DSML_DONE);
    AGENT_TEST_ASSERT(p.calls.len == 1);
    AGENT_TEST_ASSERT(!strcmp(agent_tool_arg_value(&p.calls.v[0], "command"),
                              "printf hi"));
    AGENT_TEST_ASSERT(!strcmp(agent_tool_arg_value(&p.calls.v[0], "refresh_sec"),
                              "1"));

    agent_dsml_parser_free(&p);
}

static void test_agent_glm_tool_parser_multiple_adjacent_calls(void) {
    const char *text =
        "<tool_call>list<arg_key>path</arg_key><arg_value>.</arg_value></tool_call>\n"
        "<tool_call>bash<arg_key>command</arg_key><arg_value>pwd</arg_value></tool_call>";
    agent_dsml_parser p = {
        .syntax = AGENT_TOOL_SYNTAX_GLM,
        .state = AGENT_DSML_SEARCH,
    };

    agent_dsml_feed(&p, text, strlen(text));
    agent_dsml_finish(&p);

    AGENT_TEST_ASSERT(p.state == AGENT_DSML_DONE);
    AGENT_TEST_ASSERT(p.calls.len == 2);
    AGENT_TEST_ASSERT(p.calls.v[0].name && !strcmp(p.calls.v[0].name, "list"));
    AGENT_TEST_ASSERT(!strcmp(agent_tool_arg_value(&p.calls.v[0], "path"), "."));
    AGENT_TEST_ASSERT(p.calls.v[1].name && !strcmp(p.calls.v[1].name, "bash"));
    AGENT_TEST_ASSERT(!strcmp(agent_tool_arg_value(&p.calls.v[1], "command"), "pwd"));

    agent_dsml_parser_free(&p);
}

static void test_agent_glm_stream_tool_call_chunked(void) {
    const char *chunks[] = {
        "intro ",
        "<to",
        "ol_call>bash<arg_key>command</arg_key><arg_value>printf hi</arg",
        "_value><arg_key>refresh_sec</arg_key><arg_value>1</arg_value></tool_call>",
    };
    agent_dsml_parser p;
    char *out = agent_test_stream_capture(AGENT_TOOL_SYNTAX_GLM,
                                          chunks,
                                          sizeof(chunks)/sizeof(chunks[0]),
                                          &p,
                                          NULL);

    AGENT_TEST_ASSERT(p.state == AGENT_DSML_DONE);
    AGENT_TEST_ASSERT(p.calls.len == 1);
    AGENT_TEST_ASSERT(p.calls.v[0].name && !strcmp(p.calls.v[0].name, "bash"));
    AGENT_TEST_ASSERT(!strcmp(agent_tool_arg_value(&p.calls.v[0], "command"),
                              "printf hi"));
    AGENT_TEST_ASSERT(strstr(out, "intro ") != NULL);
    AGENT_TEST_ASSERT(strstr(out, "printf hi") != NULL);
    AGENT_TEST_ASSERT(strstr(out, "<tool_call>") == NULL);
    AGENT_TEST_ASSERT(strstr(out, "<arg_key>") == NULL);
    AGENT_TEST_ASSERT(strstr(out, "</arg_value>") == NULL);

    free(out);
    agent_dsml_parser_free(&p);
}

static void test_agent_glm_stream_ignores_tool_inside_think(void) {
    const char *chunks[] = {
        "<think>plan <tool",
        "_call>bash<arg_key>command</arg_key><arg_value>printf hi</arg_value></tool_call>",
        "</think>done",
    };
    agent_dsml_parser p;
    bool dsml_in_think = false;
    char *out = agent_test_stream_capture(AGENT_TOOL_SYNTAX_GLM,
                                          chunks,
                                          sizeof(chunks)/sizeof(chunks[0]),
                                          &p,
                                          &dsml_in_think);

    AGENT_TEST_ASSERT(dsml_in_think);
    AGENT_TEST_ASSERT(p.calls.len == 0);
    AGENT_TEST_ASSERT(strstr(out, "[tool call ignored:") != NULL);
    AGENT_TEST_ASSERT(strstr(out, "<tool_call>") == NULL);

    free(out);
    agent_dsml_parser_free(&p);
}

static void test_agent_glm_stream_greedy_sampling_boundaries(void) {
    agent_tail_capture tail = {
        .cap = 4096,
    };
    agent_token_renderer renderer = {
        .format_thinking = true,
        .format_markdown = false,
        .last_output_newline = true,
        .capture = &tail,
    };
    agent_dsml_parser p = {
        .syntax = AGENT_TOOL_SYNTAX_GLM,
        .state = AGENT_DSML_SEARCH,
    };
    agent_stream_renderer stream = {
        .renderer = &renderer,
        .parser = &p,
        .syntax = AGENT_TOOL_SYNTAX_GLM,
    };

    agent_stream_text(&stream, "<tool", strlen("<tool"), false);
    AGENT_TEST_ASSERT(agent_stream_wants_greedy_sampling(&stream));

    const char *body =
        "_call>bash<arg_key>command</arg_key><arg_value>printf hi";
    agent_stream_text(&stream, body, strlen(body), false);
    AGENT_TEST_ASSERT(p.state == AGENT_DSML_PARAM_VALUE);
    AGENT_TEST_ASSERT(!agent_stream_wants_greedy_sampling(&stream));

    agent_stream_text(&stream, "</arg", strlen("</arg"), false);
    AGENT_TEST_ASSERT(p.param_close_prefix);
    AGENT_TEST_ASSERT(agent_stream_wants_greedy_sampling(&stream));

    const char *close = "_value></tool_call>";
    agent_stream_text(&stream, close, strlen(close), false);
    agent_stream_text(&stream, NULL, 0, true);
    renderer_finish(&renderer);
    char *out = agent_tail_capture_take(&tail, NULL);

    AGENT_TEST_ASSERT(p.state == AGENT_DSML_DONE);
    AGENT_TEST_ASSERT(p.calls.len == 1);
    AGENT_TEST_ASSERT(strstr(out, "</arg_value>") == NULL);

    free(out);
    agent_dsml_parser_free(&p);
}

static void test_agent_dsml_stream_tool_call_chunked(void) {
    const char *chunks[] = {
        "<｜D",
        "SML｜tool_calls><｜DSML｜invoke name=\"read\">"
        "<｜DSML｜parameter name=\"path\" string=\"true\">ds4_agent.c</｜DSML｜parameter>"
        "</｜DSML｜invoke></｜DSML｜tool_calls>",
    };
    agent_dsml_parser p;
    char *out = agent_test_stream_capture(AGENT_TOOL_SYNTAX_DSML,
                                          chunks,
                                          sizeof(chunks)/sizeof(chunks[0]),
                                          &p,
                                          NULL);

    AGENT_TEST_ASSERT(p.state == AGENT_DSML_DONE);
    AGENT_TEST_ASSERT(p.calls.len == 1);
    AGENT_TEST_ASSERT(p.calls.v[0].name && !strcmp(p.calls.v[0].name, "read"));
    AGENT_TEST_ASSERT(!strcmp(agent_tool_arg_value(&p.calls.v[0], "path"),
                              "ds4_agent.c"));
    AGENT_TEST_ASSERT(strstr(out, "Reading ") != NULL);
    AGENT_TEST_ASSERT(strstr(out, "ds4_agent.c") != NULL);
    AGENT_TEST_ASSERT(strstr(out, "DSML") == NULL);

    free(out);
    agent_dsml_parser_free(&p);
}

static void test_agent_glm_tool_parser_rejects_missing_value(void) {
    const char *text = "<tool_call>list<arg_key>path</arg_key></tool_call>";
    agent_dsml_parser p = {
        .syntax = AGENT_TOOL_SYNTAX_GLM,
        .state = AGENT_DSML_SEARCH,
    };

    agent_dsml_feed(&p, text, strlen(text));

    AGENT_TEST_ASSERT(p.state == AGENT_DSML_ERROR);
    AGENT_TEST_ASSERT(strstr(p.error, "expected <arg_value>") != NULL);

    agent_dsml_parser_free(&p);
}

static void test_agent_glm_tools_prompt_is_native(void) {
    char *prompt = agent_build_glm_tools_prompt();

    AGENT_TEST_ASSERT(strstr(prompt, "<tools>") != NULL);
    AGENT_TEST_ASSERT(strstr(prompt, "<tool_call>") != NULL);
    AGENT_TEST_ASSERT(strstr(prompt, "<arg_key>") != NULL);
    AGENT_TEST_ASSERT(strstr(prompt, "\"name\":\"bash\"") != NULL);
    AGENT_TEST_ASSERT(strstr(prompt, "\"command\"") != NULL);
    AGENT_TEST_ASSERT(strstr(prompt, "put path first") != NULL);
    AGENT_TEST_ASSERT(strstr(prompt, "Emit at most one") == NULL);
    AGENT_TEST_ASSERT(strstr(prompt, "<｜DSML｜") == NULL);

    free(prompt);
}

static void test_agent_glm_template_policy(void) {
    AGENT_TEST_ASSERT(!agent_tool_syntax_assistant_turn_uses_eos(
        AGENT_TOOL_SYNTAX_GLM));
    AGENT_TEST_ASSERT(agent_tool_syntax_assistant_turn_uses_eos(
        AGENT_TOOL_SYNTAX_DSML));
    AGENT_TEST_ASSERT(!strcmp(ds4_glm_reasoning_effort_text(DS4_THINK_HIGH),
                              "Reasoning Effort: High"));
    AGENT_TEST_ASSERT(!strcmp(ds4_glm_reasoning_effort_text(DS4_THINK_MAX),
                              "Reasoning Effort: Max"));
    AGENT_TEST_ASSERT(ds4_glm_reasoning_effort_text(DS4_THINK_NONE) == NULL);
}

static void test_agent_read_default_lines_follow_context(void) {
    agent_config cfg = {0};
    agent_worker w = {
        .cfg = &cfg,
    };

    cfg.gen.ctx_size = 4096;
    AGENT_TEST_ASSERT(agent_read_default_lines(&w) ==
                      AGENT_READ_DEFAULT_LINES_SMALL);

    cfg.gen.ctx_size = AGENT_READ_SMALL_CONTEXT_MAX + 1;
    AGENT_TEST_ASSERT(agent_read_default_lines(&w) ==
                      AGENT_READ_DEFAULT_LINES_MEDIUM);

    cfg.gen.ctx_size = AGENT_READ_MEDIUM_CONTEXT_MAX + 1;
    AGENT_TEST_ASSERT(agent_read_default_lines(&w) ==
                      AGENT_READ_DEFAULT_LINES_LARGE);
}

static void ds4_agent_unit_tests_run(void) {
    test_agent_edit_upto_tail_newline_is_not_part_of_anchor();
    test_agent_edit_upto_requires_tail_after_newline_strip();
    test_agent_read_default_lines_follow_context();
    test_agent_glm_template_policy();
    test_agent_glm_tools_prompt_is_native();
    test_agent_glm_tool_parser_single_arg();
    test_agent_glm_tool_parser_chunked_multi_arg();
    test_agent_glm_tool_parser_streams_param_state();
    test_agent_glm_tool_parser_multiple_adjacent_calls();
    test_agent_glm_stream_tool_call_chunked();
    test_agent_glm_stream_ignores_tool_inside_think();
    test_agent_glm_stream_greedy_sampling_boundaries();
    test_agent_dsml_stream_tool_call_chunked();
    test_agent_glm_tool_parser_rejects_missing_value();
}
#endif

static bool agent_preflight_edit_old(agent_worker *w, const agent_tool_call *call,
                                     char *err, size_t err_len) {
    (void)w;
    const char *path = agent_tool_arg_value(call, "path");
    if (!path || !path[0]) return true; /* Cannot preflight until path is known. */

    const char *old = agent_tool_arg_value(call, "old");
    if (!old || !old[0]) {
        snprintf(err, err_len, "edit requires non-empty old text");
        return false;
    }

    char *data = NULL;
    size_t len = 0;
    if (agent_read_file_bytes(path, &data, &len, err, err_len) != 0)
        return false;

    const char *match = NULL;
    size_t match_len = 0;
    bool anchored = false;
    bool ok = agent_edit_find_old_span(data, len, old, &match, &match_len,
                                       &anchored, err, err_len);
    free(data);
    return ok;
}

static char *agent_apply_file_splice(const char *path,
                                     const char *data, size_t len,
                                     size_t offset, size_t remove_len,
                                     const char *insert, const char *kind) {
    char err[256];
    if (!insert) insert = "";
    size_t insert_len = strlen(insert);
    size_t out_len = offset + insert_len + (len - offset - remove_len);
    char *out = xmalloc(out_len + 1);
    memcpy(out, data, offset);
    memcpy(out + offset, insert, insert_len);
    memcpy(out + offset + insert_len, data + offset + remove_len,
           len - offset - remove_len);
    out[out_len] = '\0';

    int rc = agent_write_file_bytes(path, out, out_len, err, sizeof(err));
    if (rc != 0) {
        free(out);
        agent_buf b = {0};
        agent_buf_puts(&b, "Tool error: ");
        agent_buf_puts(&b, err);
        agent_buf_puts(&b, "\n");
        return agent_buf_take(&b);
    }

    int start_line = 0, end_line = 0, delta = 0;
    agent_old_new_line_effect(data, len, out, out_len, offset, remove_len,
                              &start_line, &end_line, &delta);
    char *result = agent_edit_result(path, start_line, end_line, delta,
                                     out, out_len, kind);
    free(out);
    return result;
}

/* Old/new editing is intentionally conservative: exact old text must be unique.
 * For large replacements, old may contain one [upto] marker: the head must be
 * unique, and the tail must be unique after that head before the whole span is
 * replaced. */
static char *agent_tool_edit(agent_worker *w, const agent_tool_call *call) {
    (void)w;
    const char *path = agent_tool_arg_value(call, "path");
    if (!path || !path[0]) return xstrdup("Tool error: edit requires path\n");
    const char *old = agent_tool_arg_value(call, "old");
    const char *new_text = agent_tool_arg_value(call, "new");
    if (!old || !old[0]) return xstrdup("Tool error: edit requires non-empty old text\n");
    if (!new_text) return xstrdup("Tool error: edit requires new text\n");

    char err[256];
    char *data = NULL;
    size_t len = 0;
    if (agent_read_file_bytes(path, &data, &len, err, sizeof(err)) != 0) {
        agent_buf b = {0};
        agent_buf_puts(&b, "Tool error: ");
        agent_buf_puts(&b, err);
        agent_buf_puts(&b, "\n");
        return agent_buf_take(&b);
    }

    const char *match = NULL;
    size_t match_len = 0;
    bool anchored = false;
    if (!agent_edit_find_old_span(data, len, old, &match, &match_len,
                                  &anchored, err, sizeof(err)))
    {
        free(data);
        agent_buf b = {0};
        agent_buf_puts(&b, "Tool error: ");
        agent_buf_puts(&b, err);
        agent_buf_puts(&b, "\n");
        return agent_buf_take(&b);
    }

    char *result = agent_apply_file_splice(path, data, len,
                                           (size_t)(match - data), match_len,
                                           new_text,
                                           anchored ? "anchored old/new replacement"
                                                    : "old/new replacement");
    free(data);
    return result;
}

typedef struct {
    const char *query;
    const char *glob;
    regex_t regex;
    bool use_regex;
    bool regex_ready;
    bool case_sensitive;
    int context;
    int max_results;
    int results;
    agent_buf out;
} agent_search_ctx;

static bool agent_literal_match(const char *s, size_t n, const char *q,
                                bool case_sensitive) {
    size_t qn = strlen(q);
    if (!qn) return true;
    if (qn > n) return false;
    for (size_t i = 0; i + qn <= n; i++) {
        bool ok = true;
        for (size_t j = 0; j < qn; j++) {
            unsigned char a = (unsigned char)s[i + j];
            unsigned char b = (unsigned char)q[j];
            if (!case_sensitive) {
                a = (unsigned char)tolower(a);
                b = (unsigned char)tolower(b);
            }
            if (a != b) {
                ok = false;
                break;
            }
        }
        if (ok) return true;
    }
    return false;
}

static bool agent_search_line_matches(agent_search_ctx *ctx, const char *s, size_t n) {
    if (ctx->use_regex) {
        char *line = xstrndup(s, n);
        int rc = regexec(&ctx->regex, line, 0, NULL, 0);
        free(line);
        return rc == 0;
    }
    return agent_literal_match(s, n, ctx->query, ctx->case_sensitive);
}

static void agent_search_emit_line(agent_search_ctx *ctx, const char *data,
                                   agent_line_span sp, int line_no) {
    char prefix[64];
    snprintf(prefix, sizeof(prefix), "  %d ", line_no);
    agent_buf_puts(&ctx->out, prefix);
    agent_buf_append(&ctx->out, data + sp.start, sp.content_end - sp.start);
    agent_buf_puts(&ctx->out, "\n");
}

/* Search one text file and emit matching lines with plain line numbers. */
static void agent_search_file(agent_search_ctx *ctx, const char *path) {
    if (ctx->results >= ctx->max_results) return;
    if (ctx->glob && ctx->glob[0]) {
        const char *base = strrchr(path, '/');
        base = base ? base + 1 : path;
        if (fnmatch(ctx->glob, base, 0) != 0 && fnmatch(ctx->glob, path, 0) != 0)
            return;
    }
    char err[256];
    char *data = NULL;
    size_t len = 0;
    if (agent_read_file_bytes(path, &data, &len, err, sizeof(err)) != 0) return;
    if (memchr(data, '\0', len)) {
        free(data);
        return;
    }
    agent_line_spans spans = {0};
    agent_split_lines(data, len, &spans);
    bool printed_file = false;
    int last_context_line = -1;
    for (int i = 0; i < spans.len && ctx->results < ctx->max_results; i++) {
        agent_line_span sp = spans.v[i];
        if (!agent_search_line_matches(ctx, data + sp.start, sp.content_end - sp.start))
            continue;
        if (!printed_file) {
            agent_buf_puts(&ctx->out, path);
            agent_buf_puts(&ctx->out, "\n");
            printed_file = true;
        }
        int from = i - ctx->context;
        int to = i + ctx->context;
        if (from < 0) from = 0;
        if (to >= spans.len) to = spans.len - 1;
        if (from <= last_context_line) from = last_context_line + 1;
        for (int j = from; j <= to; j++) {
            agent_search_emit_line(ctx, data, spans.v[j], j + 1);
            last_context_line = j;
        }
        ctx->results++;
    }
    if (printed_file) agent_buf_puts(&ctx->out, "\n");
    agent_line_spans_free(&spans);
    free(data);
}

/* Recursively search a file or directory, avoiding .git and stopping once the
 * result cap is reached. */
static void agent_search_path(agent_search_ctx *ctx, const char *path, int depth) {
    if (ctx->results >= ctx->max_results || depth > 24) return;
    struct stat st;
    if (lstat(path, &st) != 0) return;
    if (S_ISREG(st.st_mode)) {
        agent_search_file(ctx, path);
        return;
    }
    if (!S_ISDIR(st.st_mode)) return;
    DIR *dir = opendir(path);
    if (!dir) return;
    struct dirent *de;
    while ((de = readdir(dir)) != NULL && ctx->results < ctx->max_results) {
        if (!strcmp(de->d_name, ".") || !strcmp(de->d_name, "..")) continue;
        if (!strcmp(de->d_name, ".git")) continue;
        char child[PATH_MAX];
        snprintf(child, sizeof(child), "%s/%s", path, de->d_name);
        agent_search_path(ctx, child, depth + 1);
    }
    closedir(dir);
}

/* Implement the search tool using either literal matching or POSIX regex. */
static char *agent_tool_search(agent_worker *w, const agent_tool_call *call) {
    (void)w;
    const char *query = agent_tool_arg_value(call, "query");
    if (!query || !query[0]) return xstrdup("Tool error: search requires query\n");
    const char *path = agent_tool_arg_value(call, "path");
    if (!path || !path[0]) path = ".";
    const char *mode = agent_tool_arg_value(call, "mode");
    agent_search_ctx ctx = {
        .query = query,
        .glob = agent_tool_arg_value(call, "glob"),
        .use_regex = mode && !strcmp(mode, "regex"),
        .case_sensitive = agent_parse_bool_default(agent_tool_arg_value(call, "case_sensitive"), true),
        .context = agent_parse_int_default(agent_tool_arg_value(call, "context"), 0, 0, 5),
        .max_results = agent_parse_int_default(agent_tool_arg_value(call, "max_results"), 50, 1, 500),
    };
    if (ctx.use_regex) {
        int flags = REG_EXTENDED | REG_NOSUB;
        if (!ctx.case_sensitive) flags |= REG_ICASE;
        int rc = regcomp(&ctx.regex, query, flags);
        if (rc != 0) {
            char msg[256];
            regerror(rc, &ctx.regex, msg, sizeof(msg));
            agent_buf b = {0};
            agent_buf_puts(&b, "Tool error: invalid regex: ");
            agent_buf_puts(&b, msg);
            agent_buf_puts(&b, "\n");
            return agent_buf_take(&b);
        }
        ctx.regex_ready = true;
    }
    agent_search_path(&ctx, path, 0);
    if (ctx.regex_ready) regfree(&ctx.regex);
    if (!ctx.out.ptr) agent_buf_puts(&ctx.out, "No matches\n");
    else {
        char hdr[96];
        snprintf(hdr, sizeof(hdr), "%d match%s shown\n\n",
                 ctx.results, ctx.results == 1 ? "" : "es");
        size_t hdr_len = strlen(hdr);
        if (ctx.out.len + hdr_len + 1 > ctx.out.cap) {
            ctx.out.cap = ctx.out.len + hdr_len + 1;
            ctx.out.ptr = xrealloc(ctx.out.ptr, ctx.out.cap);
        }
        memmove(ctx.out.ptr + hdr_len, ctx.out.ptr, ctx.out.len + 1);
        memcpy(ctx.out.ptr, hdr, hdr_len);
        ctx.out.len += hdr_len;
    }
    return agent_buf_take(&ctx.out);
}

/* ============================================================================
 * Browser Web Tools
 * ============================================================================
 *
 * The browser subsystem lives in ds4_web.c: it owns visible Chrome and CDP.  The
 * agent side only asks for permission, dispatches tools, and caps visit_page
 * output using the same "head plus temp file" shape as bash.
 */

#define AGENT_WEB_HEAD_BYTES (8*1024)
#define AGENT_WEB_HEAD_LINES 100

static int agent_count_lines(const char *s) {
    if (!s || !s[0]) return 0;
    int lines = 0;
    for (const char *p = s; *p; p++) {
        if (*p == '\n') lines++;
    }
    if (s[strlen(s) - 1] != '\n') lines++;
    return lines;
}

static char *agent_string_head(const char *s, int max_lines, size_t max_bytes,
                               int *lines_read, bool *byte_limited) {
    if (lines_read) *lines_read = 0;
    if (byte_limited) *byte_limited = false;
    if (!s) return xstrdup("");
    size_t used = 0;
    int lines = 0;
    while (s[used] && used < max_bytes && lines < max_lines) {
        if (s[used++] == '\n') lines++;
    }
    if (s[used] && used >= max_bytes && byte_limited) *byte_limited = true;
    if (used && s[used - 1] != '\n' && lines < max_lines) lines++;
    if (lines_read) *lines_read = lines;
    return xstrndup(s, used);
}

static bool agent_write_temp_text(const char *prefix, const char *text,
                                  char *path, size_t path_len,
                                  char *err, size_t err_len) {
    char tmpl[PATH_MAX];
    snprintf(tmpl, sizeof(tmpl), "/tmp/%s_XXXXXX", prefix);
    int fd = mkstemp(tmpl);
    if (fd < 0) {
        snprintf(err, err_len, "failed to create temporary file: %s", strerror(errno));
        return false;
    }
    size_t len = text ? strlen(text) : 0;
    const char *p = text ? text : "";
    size_t left = len;
    while (left) {
        ssize_t n = write(fd, p, left);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) {
            snprintf(err, err_len, "failed to write temporary file: %s", strerror(errno));
            close(fd);
            unlink(tmpl);
            return false;
        }
        p += n;
        left -= (size_t)n;
    }
    if (close(fd) != 0) {
        snprintf(err, err_len, "failed to close temporary file: %s", strerror(errno));
        unlink(tmpl);
        return false;
    }
    snprintf(path, path_len, "%s", tmpl);
    return true;
}

static char *agent_tool_google_search(agent_worker *w, const agent_tool_call *call) {
    const char *query = agent_tool_arg_value(call, "query");
    if (!query || !query[0]) return xstrdup("Tool error: google_search requires query\n");
    char err[256] = {0};
    agent_publishf_system_status(w, "Searching Google for %s...", query);
    char *md = ds4_web_google_search(w->web, query, err, sizeof(err));
    if (!md) {
        agent_buf b = {0};
        agent_buf_puts(&b, "Tool error: google_search failed: ");
        agent_buf_puts(&b, err[0] ? err : "unknown error");
        agent_buf_puts(&b, "\n");
        return agent_buf_take(&b);
    }
    return md;
}

static char *agent_tool_visit_page(agent_worker *w, const agent_tool_call *call) {
    const char *url = agent_tool_arg_value(call, "url");
    if (!url || !url[0]) return xstrdup("Tool error: visit_page requires url\n");
    char err[256] = {0};
    agent_publishf_system_status(w, "Opening page %s...", url);
    char *md = ds4_web_visit_page(w->web, url, err, sizeof(err));
    if (!md) {
        agent_buf b = {0};
        agent_buf_puts(&b, "Tool error: visit_page failed: ");
        agent_buf_puts(&b, err[0] ? err : "unknown error");
        agent_buf_puts(&b, "\n");
        return agent_buf_take(&b);
    }

    char path[PATH_MAX];
    if (!agent_write_temp_text("ds4_agent_web", md, path, sizeof(path),
                               err, sizeof(err)))
    {
        free(md);
        agent_buf b = {0};
        agent_buf_puts(&b, "Tool error: visit_page failed: ");
        agent_buf_puts(&b, err[0] ? err : "could not store rendered page");
        agent_buf_puts(&b, "\n");
        return agent_buf_take(&b);
    }

    int total_lines = agent_count_lines(md);
    int shown_lines = 0;
    bool byte_limited = false;
    char *head = agent_string_head(md, AGENT_WEB_HEAD_LINES, AGENT_WEB_HEAD_BYTES,
                                   &shown_lines, &byte_limited);
    bool truncated = byte_limited || shown_lines < total_lines;
    agent_buf out = {0};
    char line[PATH_MAX + 256];
    snprintf(line, sizeof(line),
             "visit_page url=%s\noutput_path=%s (%zu bytes, %d lines)\n",
             url, path, strlen(md), total_lines);
    agent_buf_puts(&out, line);
    if (truncated) {
        snprintf(line, sizeof(line), "<head -%d %s>\n",
                 AGENT_WEB_HEAD_LINES, path);
        agent_buf_puts(&out, line);
        agent_buf_puts(&out, head);
        if (head[0] && head[strlen(head) - 1] != '\n') agent_buf_puts(&out, "\n");
        agent_buf_puts(&out, "</head>\n");
        agent_buf_puts(&out,
            "Use read path=<output_path> start_line=<line> max_lines=<count> raw=true to inspect more rendered Markdown.\n");
    } else {
        agent_buf_puts(&out, "<markdown>\n");
        agent_buf_puts(&out, head);
        if (head[0] && head[strlen(head) - 1] != '\n') agent_buf_puts(&out, "\n");
        agent_buf_puts(&out, "</markdown>\n");
    }
    free(head);
    free(md);
    return agent_buf_take(&out);
}

/* ============================================================================
 * Asynchronous Bash Jobs
 * ============================================================================
 *
 * Bash commands are tracked jobs, not blocking one-shot calls.  Each job owns a
 * process, a pipe, and a secure /tmp output file.  The first observation is
 * head-biased so headers and early errors are visible; later progress updates
 * are tail-biased and report how much output was added since the previous
 * observation.
 */

#define AGENT_BASH_HEAD_BYTES (8*1024)
#define AGENT_BASH_HEAD_LINES 100
#define AGENT_BASH_TAIL_BYTES (32*1024)
#define AGENT_BASH_PROGRESS_TAIL_LINES 4
#define AGENT_BASH_FINAL_TAIL_LINES 20

struct agent_bash_job {
    int id;
    pid_t pid;
    int pipe_fd;
    int tmp_fd;
    char path[PATH_MAX];
    char *cmd;
    double start_time;
    double timeout_sec;
    size_t bytes;
    int newline_count;
    char last_byte;
    size_t observed_bytes;
    int observed_display_lines;
    bool observed_once;
    int exit_status;
    bool running;
    bool timed_out;
    struct agent_bash_job *next;
    agent_worker *worker;  /* back-pointer for terminal state restoration */
};

static int agent_bash_display_lines(const agent_bash_job *job) {
    if (!job || job->bytes == 0) return 0;
    return job->newline_count + (job->last_byte != '\n');
}

static void agent_bash_note_output(agent_bash_job *job, const char *s, size_t n) {
    for (size_t i = 0; i < n; i++) {
        if (s[i] == '\n') job->newline_count++;
    }
    if (n) job->last_byte = s[n - 1];
    job->bytes += n;
}

static void agent_bash_job_free(agent_bash_job *job) {
    if (!job) return;
    if (job->running && job->pid > 0) {
        kill(-job->pid, SIGKILL);
        kill(job->pid, SIGKILL);
        waitpid(job->pid, NULL, 0);
    }
    if (job->pipe_fd >= 0) close(job->pipe_fd);
    if (job->tmp_fd >= 0) close(job->tmp_fd);
    free(job->cmd);
    free(job);
}

static void agent_bash_jobs_free(agent_worker *w) {
    agent_bash_job *job = w->bash_jobs;
    while (job) {
        agent_bash_job *next = job->next;
        agent_bash_job_free(job);
        job = next;
    }
    w->bash_jobs = NULL;
}

static agent_bash_job *agent_bash_find_job(agent_worker *w, int id, pid_t pid) {
    for (agent_bash_job *job = w->bash_jobs; job; job = job->next) {
        if ((id > 0 && job->id == id) || (id <= 0 && pid > 0 && job->pid == pid))
            return job;
    }
    return NULL;
}

static void agent_bash_remove_job(agent_worker *w, agent_bash_job *target) {
    agent_bash_job **link = &w->bash_jobs;
    while (*link) {
        if (*link == target) {
            *link = target->next;
            target->next = NULL;
            agent_bash_job_free(target);
            return;
        }
        link = &(*link)->next;
    }
}

static void agent_bash_drain(agent_bash_job *job) {
    if (!job || job->pipe_fd < 0) return;
    char tmp[4096];
    for (;;) {
        ssize_t n = read(job->pipe_fd, tmp, sizeof(tmp));
        if (n > 0) {
            agent_bash_note_output(job, tmp, (size_t)n);
            if (job->tmp_fd >= 0) write_all(job->tmp_fd, tmp, (size_t)n);
            continue;
        }
        if (n < 0 && errno == EINTR) continue;
        break;
    }
}

static void agent_worker_note_terminal_mode_may_have_changed(agent_worker *w) {
    if (!w) return;
    pthread_mutex_lock(&w->mu);
    w->raw_mode_needs_restore = true;
    pthread_mutex_unlock(&w->mu);
}

static void agent_bash_finalize(agent_bash_job *job, int status) {
    agent_bash_drain(job);
    if (job->pipe_fd >= 0) {
        close(job->pipe_fd);
        job->pipe_fd = -1;
    }
    if (job->tmp_fd >= 0) {
        close(job->tmp_fd);
        job->tmp_fd = -1;
    }
    if (WIFEXITED(status)) job->exit_status = WEXITSTATUS(status);
    else if (WIFSIGNALED(status)) job->exit_status = 128 + WTERMSIG(status);
    else job->exit_status = -1;
    job->running = false;
    /* A child can still open /dev/tty directly and alter terminal state even
     * though its stdin is /dev/null.  Ask the UI thread to verify raw mode at
     * a safe point instead of touching linenoise from the worker path. */
    agent_worker_note_terminal_mode_may_have_changed(job->worker);
}

/* Drain available output, notice process exit, and enforce timeout.  This is
 * called opportunistically by status/wait/compaction instead of a background
 * reaper thread, keeping all bash job state owned by the agent worker. */
static void agent_bash_poll(agent_bash_job *job) {
    if (!job || !job->running) return;
    agent_bash_drain(job);

    int status = 0;
    pid_t rc = waitpid(job->pid, &status, WNOHANG);
    if (rc == job->pid) {
        agent_bash_finalize(job, status);
        return;
    }
    if (rc < 0 && errno != EINTR) {
        job->exit_status = -1;
        job->running = false;
        if (job->pipe_fd >= 0) {
            close(job->pipe_fd);
            job->pipe_fd = -1;
        }
        if (job->tmp_fd >= 0) {
            close(job->tmp_fd);
            job->tmp_fd = -1;
        }
        agent_worker_note_terminal_mode_may_have_changed(job->worker);
        return;
    }
    if (now_sec() - job->start_time >= job->timeout_sec) {
        job->timed_out = true;
        kill(-job->pid, SIGKILL);
        kill(job->pid, SIGKILL);
        while (waitpid(job->pid, &status, 0) < 0 && errno == EINTR) {}
        agent_bash_finalize(job, status);
    }
}

/* Spawn a shell command into its own process group so bash_stop/timeout can
 * kill grandchildren created by the shell, not just the /bin/sh wrapper. */
static agent_bash_job *agent_bash_start(agent_worker *w, const char *cmd,
                                        int timeout_sec, char *err, size_t err_len) {
    char tmp_path[] = "/tmp/ds4_agent_output_XXXXXX";
    int tmpfd = mkstemp(tmp_path);
    if (tmpfd < 0) {
        snprintf(err, err_len, "failed to create temporary output file: %s", strerror(errno));
        return NULL;
    }

    int pipefd[2];
    if (pipe(pipefd) != 0) {
        snprintf(err, err_len, "failed to create pipe: %s", strerror(errno));
        close(tmpfd);
        unlink(tmp_path);
        return NULL;
    }
    pid_t pid = fork();
    if (pid < 0) {
        snprintf(err, err_len, "failed to fork: %s", strerror(errno));
        close(pipefd[0]);
        close(pipefd[1]);
        close(tmpfd);
        unlink(tmp_path);
        return NULL;
    }
    if (pid == 0) {
        setpgid(0, 0);
        close(tmpfd);
        /* The bash tool is not interactive.  Give the shell /dev/null as
         * stdin so it does not inherit the live linenoise terminal and reset
         * it from raw mode to cooked mode behind the agent's back. */
        int null_fd = open("/dev/null", O_RDONLY);
        if (null_fd >= 0) {
            if (dup2(null_fd, STDIN_FILENO) < 0)
                close(STDIN_FILENO);
            if (null_fd != STDIN_FILENO)
                close(null_fd);
        } else {
            close(STDIN_FILENO);
        }
        close(pipefd[0]);
        dup2(pipefd[1], STDOUT_FILENO);
        dup2(pipefd[1], STDERR_FILENO);
        close(pipefd[1]);
        execl("/bin/sh", "sh", "-c", cmd ? cmd : "", (char *)NULL);
        _exit(127);
    }

    close(pipefd[1]);
    setpgid(pid, pid);
    int old_flags;
    set_nonblock(pipefd[0], true, &old_flags);

    agent_bash_job *job = xmalloc(sizeof(*job));
    memset(job, 0, sizeof(*job));
    if (w->next_bash_job_id <= 0) w->next_bash_job_id = 1;
    job->id = w->next_bash_job_id++;
    job->pid = pid;
    job->pipe_fd = pipefd[0];
    job->tmp_fd = tmpfd;
    snprintf(job->path, sizeof(job->path), "%s", tmp_path);
    job->cmd = xstrdup(cmd);
    job->start_time = now_sec();
    job->timeout_sec = timeout_sec;
    job->exit_status = -1;
    job->running = true;
    job->worker = w;
    job->next = w->bash_jobs;
    w->bash_jobs = job;
    return job;
}

static void agent_tail_append(agent_buf *b, const char *s, size_t n, size_t max) {
    if (!n) return;
    agent_buf_append(b, s, n);
    if (b->len > max) {
        size_t drop = b->len - max;
        memmove(b->ptr, b->ptr + drop, b->len - drop + 1);
        b->len -= drop;
    }
}

/* Read the first max_lines from the output file, with a byte cap to avoid a
 * pathological single long line flooding the next model turn. */
static char *agent_bash_read_head(const agent_bash_job *job, int max_lines,
                                  size_t max_bytes, int *lines_read,
                                  bool *byte_limited) {
    if (lines_read) *lines_read = 0;
    if (byte_limited) *byte_limited = false;
    if (!job || !job->path[0] || job->bytes == 0) return xstrdup("");
    FILE *fp = fopen(job->path, "rb");
    if (!fp) return xstrdup("<failed to reopen output file>\n");

    agent_buf out = {0};
    int lines = 0;
    while (lines < max_lines && out.len < max_bytes) {
        int c = fgetc(fp);
        if (c == EOF) {
            if (ferror(fp) && errno == EINTR) {
                clearerr(fp);
                continue;
            }
            break;
        }
        char ch = (char)c;
        agent_buf_append(&out, &ch, 1);
        if (ch == '\n') lines++;
    }
    if (out.len >= max_bytes && !feof(fp) && byte_limited) *byte_limited = true;
    fclose(fp);
    if (lines_read) *lines_read = lines + (out.len && out.ptr[out.len - 1] != '\n');
    if (!out.ptr) return xstrdup("");
    return agent_buf_take(&out);
}

/* Read the last max_lines from the full output file.  The model-visible label
 * says "tail -N <file>" so it is clear this is not the complete output. */
static char *agent_bash_read_tail_lines(const agent_bash_job *job, int max_lines) {
    if (!job || !job->path[0] || job->bytes == 0) return xstrdup("");
    FILE *fp = fopen(job->path, "rb");
    if (!fp) return xstrdup("<failed to reopen output file>\n");

    agent_buf tail = {0};
    char tmp[2048];
    for (;;) {
        size_t n = fread(tmp, 1, sizeof(tmp), fp);
        if (n) agent_tail_append(&tail, tmp, n, AGENT_BASH_TAIL_BYTES);
        if (n < sizeof(tmp)) {
            if (ferror(fp) && errno == EINTR) {
                clearerr(fp);
                continue;
            }
            break;
        }
    }
    fclose(fp);
    if (!tail.ptr) return xstrdup("");

    char *start = tail.ptr;
    int newlines = 0;
    for (char *p = tail.ptr + tail.len; p > tail.ptr; p--) {
        if (p[-1] == '\n' && ++newlines > max_lines) {
            start = p;
            break;
        }
    }
    char *out = xstrdup(start);
    free(tail.ptr);
    return out;
}

/* Build the tool result for a bash job.  mark_observed advances the per-job
 * cursor so the next status reports only fresh output. */
static char *agent_bash_observation(agent_bash_job *job, bool mark_observed) {
    agent_bash_poll(job);
    bool first_observation = !job->observed_once;
    int display_lines = agent_bash_display_lines(job);
    double elapsed = now_sec() - job->start_time;

    agent_buf out = {0};
    char line[PATH_MAX + 256];
    if (job->running) {
        snprintf(line, sizeof(line),
            "bash job=%d pid=%ld status=running elapsed_sec=%.1f timeout_sec=%.0f\n",
            job->id, (long)job->pid, elapsed, job->timeout_sec);
    } else {
        snprintf(line, sizeof(line),
            "bash job=%d pid=%ld status=done elapsed_sec=%.1f timed_out=%d\n",
            job->id, (long)job->pid, elapsed, job->timed_out ? 1 : 0);
    }
    agent_buf_puts(&out, line);
    if (!job->running) {
        snprintf(line, sizeof(line), "exit_status=%d\n", job->exit_status);
        agent_buf_puts(&out, line);
    }

    if (job->bytes == 0) {
        agent_buf_puts(&out, "<output>\n</output>\n");
    } else if (first_observation) {
        int shown_lines = 0;
        bool byte_limited = false;
        char *head = agent_bash_read_head(job, AGENT_BASH_HEAD_LINES,
                                          AGENT_BASH_HEAD_BYTES,
                                          &shown_lines, &byte_limited);
        bool truncated = byte_limited || display_lines > shown_lines;
        if (!job->running && !truncated) {
            agent_buf_puts(&out, "<output>\n");
            agent_buf_puts(&out, head);
            if (head[0] && head[strlen(head) - 1] != '\n') agent_buf_puts(&out, "\n");
            agent_buf_puts(&out, "</output>\n");
        } else {
            snprintf(line, sizeof(line),
                     "output_path=%s (%zu bytes, %d lines)\n",
                     job->path[0] ? job->path : "<unavailable>",
                     job->bytes, display_lines);
            agent_buf_puts(&out, line);
            snprintf(line, sizeof(line), "<head -%d %s>\n",
                     AGENT_BASH_HEAD_LINES, job->path);
            agent_buf_puts(&out, line);
            agent_buf_puts(&out, head);
            if (head[0] && head[strlen(head) - 1] != '\n') agent_buf_puts(&out, "\n");
            agent_buf_puts(&out, "</head>\n");
        }
        free(head);
    } else {
        int tail_lines = job->running ? AGENT_BASH_PROGRESS_TAIL_LINES :
                                        AGENT_BASH_FINAL_TAIL_LINES;
        char *tail = agent_bash_read_tail_lines(job, tail_lines);
        snprintf(line, sizeof(line),
                 "output_path=%s (%zu bytes, %d lines)\n",
                 job->path[0] ? job->path : "<unavailable>",
                 job->bytes, display_lines);
        agent_buf_puts(&out, line);
        snprintf(line, sizeof(line), "<tail -%d %s>\n", tail_lines, job->path);
        agent_buf_puts(&out, line);
        agent_buf_puts(&out, tail);
        if (tail[0] && tail[strlen(tail) - 1] != '\n') agent_buf_puts(&out, "\n");
        snprintf(line, sizeof(line), "</tail>\n");
        agent_buf_puts(&out, line);
        free(tail);
    }
    if (job->running) {
        snprintf(line, sizeof(line),
            "\nUse bash_status job=%d to get info before refresh time; use bash_stop job=%d to stop execution\n",
            job->id, job->id);
        agent_buf_puts(&out, line);
    }

    if (mark_observed) {
        job->observed_bytes = job->bytes;
        job->observed_display_lines = display_lines;
        job->observed_once = true;
    }
    return agent_buf_take(&out);
}

static void agent_bash_publish_observation(agent_worker *w, const char *obs) {
    if (!obs || !obs[0]) return;
    const char *body = NULL;
    const char *label = strstr(obs, "\n<head ");
    const char *close = NULL;
    if (label) {
        close = "</head>";
    } else {
        label = strstr(obs, "\n<tail ");
        if (label) close = "</tail>";
    }
    if (label) {
        const char *tag_end = strstr(label, ">\n");
        if (tag_end) {
            agent_publish(w, "\x1b[90m", 5);
            if (strstr(label, "\n<head ") == label)
                agent_publish(w, "[showing first output lines]\n",
                              strlen("[showing first output lines]\n"));
            else
                agent_publish(w, "[showing last output lines]\n",
                              strlen("[showing last output lines]\n"));
            agent_publish(w, "\x1b[0m", 4);
            body = tag_end + 2;
        }
    } else {
        label = strstr(obs, "\n<output>\n");
        if (label) {
            body = label + strlen("\n<output>\n");
            close = "</output>";
        }
    }
    if (!body || !body[0]) return;
    const char *end = close ? strstr(body, close) : NULL;
    size_t n = end ? (size_t)(end - body) : strlen(body);
    if (n) {
        bool failed = strstr(obs, "status=done") && !strstr(obs, "exit_status=0\n");
        if (failed) agent_publish(w, "\x1b[38;5;208m", 11);
        agent_publish(w, body, n);
        if (body[n - 1] != '\n') agent_publish(w, "\n", 1);
        if (failed) agent_publish(w, "\x1b[0m", 4);
    }
}

static void agent_bash_refresh_for(agent_worker *w, agent_bash_job *job,
                                   int refresh_sec) {
    double start = now_sec();
    while (job->running && now_sec() - start < refresh_sec) {
        if (worker_should_interrupt(w)) break;
        agent_bash_poll(job);
        if (!job->running) break;
        struct pollfd pfd = {.fd = job->pipe_fd, .events = POLLIN};
        poll(&pfd, 1, 100);
    }
    agent_bash_poll(job);
}

/* Common implementation for bash, bash_status, and bash_stop. */
static char *agent_bash_job_tool_result(agent_worker *w, agent_bash_job *job,
                                        bool wait, int refresh_sec,
                                        bool stop, bool remove_if_done) {
    if (stop && job->running) {
        kill(-job->pid, SIGTERM);
        kill(job->pid, SIGTERM);
        double start = now_sec();
        while (job->running && now_sec() - start < 1.0) {
            agent_bash_poll(job);
            if (!job->running) break;
            usleep(20000);
        }
        if (job->running) {
            kill(-job->pid, SIGKILL);
            kill(job->pid, SIGKILL);
        }
    }
    if (wait || stop) agent_bash_refresh_for(w, job, refresh_sec);
    else agent_bash_poll(job);

    char *obs = agent_bash_observation(job, true);
    agent_bash_publish_observation(w, obs);
    if (remove_if_done && !job->running) agent_bash_remove_job(w, job);
    return obs;
}

static int agent_tool_job_id(const agent_tool_call *call) {
    return agent_parse_int_default(agent_tool_arg_value(call, "job"), 0, 0, INT_MAX);
}

static pid_t agent_tool_pid(const agent_tool_call *call) {
    return (pid_t)agent_parse_int_default(agent_tool_arg_value(call, "pid"), 0, 0, INT_MAX);
}

/* ============================================================================
 * Tool Dispatch
 * ============================================================================
 */

/* Execute one parsed DSML tool call and return the text that will be appended as
 * the tool-role result.  UI visualization already happened while streaming; this
 * function is only about side effects and the model-visible observation. */
static char *agent_execute_tool_call(agent_worker *w, const agent_tool_call *call) {
    agent_buf result = {0};
    if (!call->name) return xstrdup("Tool error: missing tool name\n");

    if (!strcmp(call->name, "read")) return agent_tool_read(w, call);
    if (!strcmp(call->name, "more")) return agent_tool_more(w, call);
    if (!strcmp(call->name, "write")) return agent_tool_write(w, call);
    if (!strcmp(call->name, "list")) return agent_tool_list(call);
    if (!strcmp(call->name, "edit")) return agent_tool_edit(w, call);
    if (!strcmp(call->name, "search")) return agent_tool_search(w, call);
    if (!strcmp(call->name, "google_search")) return agent_tool_google_search(w, call);
    if (!strcmp(call->name, "visit_page")) return agent_tool_visit_page(w, call);

    if (!strcmp(call->name, "bash")) {
        const char *cmd = agent_tool_arg_value(call, "command");
        if (!cmd || !cmd[0]) return xstrdup("Tool error: bash requires command\n");
        int timeout = agent_parse_timeout(agent_tool_arg_value(call, "timeout_sec"));
        int refresh = agent_parse_int_default(agent_tool_arg_value(call, "refresh_sec"),
                                              60, 1, 3600);
        char err[160] = {0};
        agent_bash_job *job = agent_bash_start(w, cmd, timeout, err, sizeof(err));
        if (!job) {
            agent_buf_puts(&result, "Tool error: bash failed to start: ");
            agent_buf_puts(&result, err[0] ? err : "unknown error");
            agent_buf_puts(&result, "\n");
            return agent_buf_take(&result);
        }
        return agent_bash_job_tool_result(w, job, true, refresh, false, true);
    }

    if (!strcmp(call->name, "bash_status") ||
        !strcmp(call->name, "bash_stop"))
    {
        int job_id = agent_tool_job_id(call);
        pid_t pid = agent_tool_pid(call);
        agent_bash_job *job = agent_bash_find_job(w, job_id, pid);
        if (!job) {
            char msg[128];
            snprintf(msg, sizeof(msg), "Tool error: bash job not found: job=%d pid=%ld\n",
                     job_id, (long)pid);
            return xstrdup(msg);
        }
        int refresh = agent_parse_int_default(agent_tool_arg_value(call, "refresh_sec"),
                                              60, 1, 3600);
        bool stop = !strcmp(call->name, "bash_stop");
        bool wait = stop;
        return agent_bash_job_tool_result(w, job, wait, refresh, stop, true);
    }

    {
        char header[256];
        snprintf(header, sizeof(header), "\n[tool:%s] unknown tool\n", call->name);
        agent_publish(w, header, strlen(header));
        agent_buf_puts(&result, "Tool error: unknown tool: ");
        agent_buf_puts(&result, call->name);
        agent_buf_puts(&result, "\n");
        return agent_buf_take(&result);
    }
}

/* Execute all tool calls from one DSML block, preserving per-call labels in the
 * combined result so the model can associate observations with calls. */
static char *agent_execute_tool_calls(agent_worker *w, const agent_tool_calls *calls) {
    agent_buf all = {0};
    for (int i = 0; i < calls->len; i++) {
        char *res = agent_execute_tool_call(w, &calls->v[i]);
        char hdr[128];
        snprintf(hdr, sizeof(hdr), "Tool result %d (%s):\n", i + 1,
                 calls->v[i].name ? calls->v[i].name : "unknown");
        agent_buf_puts(&all, hdr);
        agent_buf_puts(&all, res);
        if (res[0] && res[strlen(res) - 1] != '\n') agent_buf_puts(&all, "\n");
        free(res);
    }
    if (calls->len == 0) agent_buf_puts(&all, "Tool error: empty tool call block\n");
    return agent_buf_take(&all);
}

/* If compaction happens while a bash process is still alive, inject a small
 * tool-role reminder into the rebuilt transcript.  Otherwise the summary could
 * preserve the user's task but lose the fact that an external process still
 * needs status/wait/stop handling. */
static char *agent_bash_jobs_compaction_observation(agent_worker *w) {
    if (!w->bash_jobs) return NULL;
    agent_buf out = {0};
    agent_buf_puts(&out,
        "Bash job update after context compaction. Running jobs still need explicit bash_status or bash_stop if relevant.\n");
    for (agent_bash_job *job = w->bash_jobs, *next = NULL; job; job = next) {
        next = job->next;
        char *obs = agent_bash_observation(job, true);
        char hdr[64];
        snprintf(hdr, sizeof(hdr), "\nJob %d:\n", job->id);
        agent_buf_puts(&out, hdr);
        agent_buf_puts(&out, obs);
        free(obs);
        if (!job->running) agent_bash_remove_job(w, job);
    }
    return agent_buf_take(&out);
}

/* ============================================================================
 * Context Compaction
 * ============================================================================
 *
 * Compaction asks the model for durable task state, then rebuilds the live
 * transcript as: system prompt + summary + recent verbatim tail.  This keeps
 * the active KV usable while avoiding unbounded transcript growth.
 */

/* Decide when to compact before an ordinary turn or before appending a large
 * tool result.  The fixed free-token threshold is capped proportionally for
 * smaller contexts so tests with tiny contexts still compact rather than fail. */
static bool agent_worker_should_compact(agent_worker *w) {
    int ctx = agent_worker_effective_ctx_size(w);
    int used = w->transcript.len;
    if (ctx <= 0 || used <= 0) return false;
    if (used >= (ctx * AGENT_COMPACT_SOFT_PERCENT) / 100) return true;
    int free_threshold = AGENT_COMPACT_MIN_FREE_TOKENS;
    int proportional = ctx / 8;
    if (free_threshold > proportional) free_threshold = proportional;
    return ctx - used <= free_threshold;
}

static int agent_special_token_id(ds4_engine *engine, const char *rendered) {
    ds4_tokens t = {0};
    ds4_tokenize_rendered_chat(engine, rendered, &t);
    int id = t.len == 1 ? t.v[0] : -1;
    ds4_tokens_free(&t);
    return id;
}

/* Pick a recent verbatim tail for the compacted transcript.  Prefer a user
 * boundary inside the budget so the rebuilt context starts at a natural turn. */
static int agent_compact_tail_start(agent_worker *w, int bottom, int sys_len) {
    int ctx = agent_worker_effective_ctx_size(w);
    int tail_budget = ctx / AGENT_COMPACT_TAIL_DIVISOR;
    if (tail_budget > AGENT_COMPACT_TAIL_CAP_TOKENS)
        tail_budget = AGENT_COMPACT_TAIL_CAP_TOKENS;
    if (tail_budget < 1) tail_budget = 1;

    int target = bottom - tail_budget;
    if (target < sys_len) target = sys_len;

    int user_id = agent_special_token_id(w->engine,
        ds4_engine_is_glm_dsa(w->engine) ? "<|user|>" : "<｜User｜>");
    if (user_id < 0) return target;

    for (int i = target; i < bottom; i++) {
        if (w->transcript.v[i] == user_id) return i;
    }
    return target;
}

static void agent_tokens_append_range(ds4_tokens *dst, const ds4_tokens *src,
                                      int start, int end) {
    if (start < 0) start = 0;
    if (end > src->len) end = src->len;
    for (int i = start; i < end; i++) ds4_tokens_push(dst, src->v[i]);
}

/* Build the private prompt used to ask the model for durable state.  The prompt
 * explicitly forbids tool calls because the result is consumed internally, not
 * delivered as an assistant turn. */
static char *agent_compact_make_prompt(const char *reason) {
    agent_buf b = {0};
    agent_buf_puts(&b,
        "Internal ds4-agent context compaction request. This is not a user request.\n"
        "Write a durable task-state summary of the conversation so far. Preserve only facts that matter for continuing the work:\n"
        "- user goals, constraints, and preferences\n"
        "- files inspected or edited\n"
        "- commands run and important results\n"
        "- decisions, rejected approaches, known bugs, and pending next steps\n"
        "- reloadable bulky data with exact paths/ranges/commands when available\n\n"
        "Do not invent facts. Do not include generic narration. Do not include raw file contents unless they were essential to a conclusion.\n"
        "After the summary, stop. Do not continue the user task, do not call tools, and do not output thinking tags or DSML markup.\n"
        "Output only the compact summary.\n");
    if (reason && reason[0]) {
        agent_buf_puts(&b, "\nCompaction reason: ");
        agent_buf_puts(&b, reason);
        agent_buf_puts(&b, "\n");
    }
    return agent_buf_take(&b);
}

/* Perform the full compaction exchange and rebuild the live DS4 session from
 * the compacted transcript.  Any failure invalidates live KV because the model
 * may have just seen private compaction instructions that are not part of the
 * real conversation. */
static bool agent_worker_compact(agent_worker *w, const char *reason,
                                 char *err, size_t err_len) {
    const int bottom = w->transcript.len;
    if (bottom <= 0) return true;

    ds4_tokens sys = {0};
    agent_worker_build_system_tokens(w, &sys);
    if (bottom <= sys.len) {
        ds4_tokens_free(&sys);
        return true;
    }

    agent_publishf(w,
        "\n\x1b[1;95mCOMPACTING\x1b[0m %s: summarizing durable task state\n\x1b[38;5;245m",
        reason && reason[0] ? reason : "context");

    char *prompt_text = agent_compact_make_prompt(reason);
    ds4_tokens prompt = {0};
    ds4_tokens_copy(&prompt, &w->transcript);
    ds4_chat_append_message(w->engine, &prompt, "user", prompt_text);
    free(prompt_text);
    ds4_chat_append_assistant_prefix(w->engine, &prompt, DS4_THINK_NONE);

    pthread_mutex_lock(&w->mu);
    w->status.state = AGENT_WORKER_COMPACTING;
    w->progress_direct = false;
    w->progress_started_at = now_sec();
    w->status.prefill_done = 0;
    w->status.prefill_total = 0;
    w->status.prefill_tps = 0.0;
    w->status.generated = 0;
    w->status.gen_tps = 0.0;
    w->status.greedy_sampling = false;
    agent_wake_locked(w);
    pthread_mutex_unlock(&w->mu);

    int summary_room = agent_worker_effective_ctx_size(w) - prompt.len - 1;
    if (summary_room < 256) {
        snprintf(err, err_len, "not enough context left to request compaction summary");
        ds4_tokens_free(&prompt);
        ds4_tokens_free(&sys);
        agent_publish(w, "\x1b[0m\n", 5);
        return false;
    }
    int summary_max = summary_room < AGENT_COMPACT_SUMMARY_MAX_TOKENS ?
                      summary_room : AGENT_COMPACT_SUMMARY_MAX_TOKENS;

    ds4_session_set_progress(w->session, worker_progress_cb, w);
    ds4_session_set_display_progress(w->session, worker_progress_cb, w);
    ds4_session_set_cancel(w->session, worker_cancel_session_cb, w);
    int sync_rc = ds4_session_sync(w->session, &prompt, err, err_len);
    ds4_session_set_cancel(w->session, NULL, NULL);
    ds4_session_set_progress(w->session, NULL, NULL);
    ds4_session_set_display_progress(w->session, NULL, NULL);
    if (sync_rc == DS4_SESSION_SYNC_INTERRUPTED) {
        ds4_session_invalidate(w->session);
        snprintf(err, err_len, "interrupted");
        agent_publish_system_status(
            w, "Compaction interrupted; keeping the previous conversation state.");
        ds4_tokens_free(&prompt);
        ds4_tokens_free(&sys);
        agent_publish(w, "\x1b[0m\n", 5);
        worker_clear_interrupt(w);
        return false;
    }
    if (sync_rc != 0) {
        ds4_session_invalidate(w->session);
        ds4_tokens_free(&prompt);
        ds4_tokens_free(&sys);
        agent_publish(w, "\x1b[0m\n", 5);
        return false;
    }

    /* From here until the final rebuild, the live KV contains the internal
     * compaction prompt/summary, while w->transcript still contains the real
     * conversation.  If anything fails, invalidate live KV so the next turn
     * cannot accidentally continue from the private compaction exchange. */
    agent_buf summary = {0};
    char eval_err[160] = {0};
    int dsml_id = agent_special_token_id(w->engine, "｜DSML｜");
    double t0 = now_sec();
    for (int i = 0; i < summary_max; i++) {
        if (worker_should_interrupt(w)) {
            snprintf(err, err_len, "interrupted");
            ds4_session_invalidate(w->session);
            ds4_tokens_free(&prompt);
            ds4_tokens_free(&sys);
            free(summary.ptr);
            agent_publish(w, "\x1b[0m\n", 5);
            agent_publish_system_status(
                w, "Compaction interrupted; keeping the previous conversation state.");
            worker_clear_interrupt(w);
            return false;
        }
        int token = ds4_session_argmax(w->session);
        if (ds4_token_is_stop_for_think_mode(w->engine,
                                             token,
                                             DS4_THINK_NONE) ||
            token == dsml_id) {
            if (token == dsml_id && summary.len && summary.ptr[summary.len - 1] == '<') {
                summary.ptr[--summary.len] = '\0';
            }
            agent_trace(w, "compaction summary stopped before control token id=%d", token);
            break;
        }
        if (ds4_session_eval(w->session, token, eval_err, sizeof(eval_err)) != 0) {
            snprintf(err, err_len, "%s", eval_err);
            ds4_session_invalidate(w->session);
            ds4_tokens_free(&prompt);
            ds4_tokens_free(&sys);
            free(summary.ptr);
            agent_publish(w, "\x1b[0m\n", 5);
            return false;
        }

        size_t text_len = 0;
        char *text = ds4_token_text(w->engine, token, &text_len);
        agent_buf_append(&summary, text, text_len);
        agent_publish(w, text, text_len);
        free(text);

        double dt = now_sec() - t0;
        pthread_mutex_lock(&w->mu);
        w->status.generated = i + 1;
        w->status.gen_tps = dt > 0.0 ? (double)(i + 1) / dt : 0.0;
        w->status.greedy_sampling = false;
        agent_wake_locked(w);
        pthread_mutex_unlock(&w->mu);
    }
    agent_publish(w, "\x1b[0m\n", 5);
    ds4_tokens_free(&prompt);

    if (!summary.ptr || !summary.ptr[0]) {
        snprintf(err, err_len, "compaction summary was empty");
        ds4_session_invalidate(w->session);
        ds4_tokens_free(&sys);
        free(summary.ptr);
        return false;
    }

    int tail_start = agent_compact_tail_start(w, bottom, sys.len);
    ds4_tokens compacted = {0};
    ds4_tokens_copy(&compacted, &sys);

    agent_buf summary_msg = {0};
    agent_buf_puts(&summary_msg,
        "\n\n[ds4-agent compacted earlier conversation. Durable task-state summary follows.]\n");
    agent_buf_puts(&summary_msg, summary.ptr);
    if (summary_msg.len && summary_msg.ptr[summary_msg.len - 1] != '\n')
        agent_buf_puts(&summary_msg, "\n");
    agent_buf_puts(&summary_msg, "[End compacted summary. Recent conversation continues verbatim below.]\n\n");
    ds4_chat_append_message(w->engine, &compacted, "system", summary_msg.ptr);
    free(summary_msg.ptr);
    free(summary.ptr);

    agent_tokens_append_range(&compacted, &w->transcript, tail_start, bottom);

    agent_publishf(w,
        "\x1b[1;95mCOMPACTING\x1b[0m rebuilding context: old=%d summary+tail=%d tail=%d\n",
        bottom, compacted.len, bottom - tail_start);

    ds4_tokens old_transcript = {0};
    ds4_tokens_copy(&old_transcript, &w->transcript);
    ds4_tokens_free(&w->transcript);
    w->transcript = compacted;
    if (agent_worker_sync_tokens(w, &w->transcript, true, err, err_len) != 0) {
        ds4_session_invalidate(w->session);
        ds4_tokens_free(&w->transcript);
        w->transcript = old_transcript;
        ds4_tokens_free(&sys);
        return false;
    }
    agent_worker_note_system_prompt_seen(w);
    ds4_tokens_free(&old_transcript);
    ds4_tokens_free(&sys);
    char *bash_update = agent_bash_jobs_compaction_observation(w);
    if (bash_update) {
        ds4_chat_append_message(w->engine, &w->transcript, "tool", bash_update);
        w->session_dirty = true;
        agent_trace_text(w, "tool-after-compaction", bash_update, strlen(bash_update));
        agent_publish(w, "\x1b[90mCOMPACTING added bash job update after rebuild\x1b[0m\n",
                      strlen("\x1b[90mCOMPACTING added bash job update after rebuild\x1b[0m\n"));
        free(bash_update);
    }
    agent_trace(w, "compacted reason=\"%s\" old=%d new=%d tail_start=%d tail=%d",
                reason ? reason : "", bottom, w->transcript.len,
                tail_start, bottom - tail_start);
    return true;
}

static bool agent_worker_compact_if_needed(agent_worker *w, const char *reason,
                                           char *err, size_t err_len) {
    if (!agent_worker_should_compact(w)) return true;
    return agent_worker_compact(w, reason, err, err_len);
}

static int worker_accept_generated_token(agent_worker *w,
                                         int token,
                                         int *generated,
                                         double t0,
                                         agent_stream_renderer *stream,
                                         char *err,
                                         size_t err_len) {
    ds4_tokens_push(&w->transcript, token);

    size_t text_len = 0;
    char *text = ds4_token_text(w->engine, token, &text_len);
    agent_trace_token(w, token, text, text_len, *generated + 1);
    agent_stream_text(stream, text, text_len, false);
    free(text);
    (*generated)++;

    if (ds4_session_eval(w->session, token, err, err_len) != 0) {
        ds4_session_invalidate(w->session);
        return 1;
    }

    double dt = now_sec() - t0;
    pthread_mutex_lock(&w->mu);
    w->status.generated = *generated;
    w->status.gen_tps = dt > 0.0 ? (double)*generated / dt : 0.0;
    agent_wake_locked(w);
    pthread_mutex_unlock(&w->mu);
    return 0;
}

static int worker_force_generated_text(agent_worker *w,
                                       const char *text,
                                       int max_tokens,
                                       int *generated,
                                       double t0,
                                       agent_stream_renderer *stream,
                                       char *err,
                                       size_t err_len) {
    ds4_tokens tokens = {0};
    ds4_tokenize_text(w->engine, text, &tokens);
    if (tokens.len > max_tokens - *generated) {
        snprintf(err, err_len, "not enough generation room to force %s", text);
        ds4_tokens_free(&tokens);
        return 1;
    }
    for (int i = 0; i < tokens.len && *generated < max_tokens; i++) {
        if (worker_accept_generated_token(w, tokens.v[i], generated, t0,
                                          stream, err, err_len) != 0) {
            ds4_tokens_free(&tokens);
            return 1;
        }
    }
    ds4_tokens_free(&tokens);
    return 0;
}

/* ============================================================================
 * Model Worker Thread
 * ============================================================================
 */

/* DSML structure is a machine-readable grammar, so once the model has clearly
 * started a tool stanza we decode grammar bytes greedily.  Parameter values are
 * different: they can be shell commands, code, file contents, or edit bodies,
 * and should keep the configured sampling behavior.  The only exception is a
 * parameter closing tag once it is clearly DSML syntax, not ordinary text such
 * as HTML/XML/code containing "</".
 *
 * This helper is intentionally derived only from the current streaming parser
 * state.  The state object is local to one assistant round, so malformed output,
 * EOS, Ctrl+C, or the next turn cannot accidentally leave sampling greedy. */
static bool agent_stream_wants_greedy_sampling(const agent_stream_renderer *sr) {
    if (!sr || !sr->parser) return false;
    if (sr->parser->state == AGENT_DSML_ERROR ||
        sr->parser->state == AGENT_DSML_DONE)
        return false;

    /* A possible opening marker is being held back by the start detector.  A
     * single '<' is too common in prose/code to justify forcing argmax; after
     * the second byte, the buffered prefix still matching here is specifically
     * DSML-shaped ("<｜..." or the tolerated "<D..." typo). */
    if (sr->dsml_start_len > 1) return true;
    if (!sr->dsml_active) return false;

    if (sr->parser->state == AGENT_DSML_STRUCTURAL)
        return true;
    if (sr->parser->state != AGENT_DSML_PARAM_VALUE)
        return false;

    return sr->parser->param_close_prefix;
}

static int worker_sample_with_mode(agent_worker *w, const agent_config *cfg,
                                   bool greedy, uint64_t *rng) {
    return ds4_session_sample(w->session,
                              greedy ? 0.0f : cfg->gen.temperature,
                              0,
                              greedy ? 1.0f : cfg->gen.top_p,
                              greedy ? 0.0f : cfg->gen.min_p,
                              rng);
}

static void worker_set_greedy_sampling(agent_worker *w, bool greedy) {
    pthread_mutex_lock(&w->mu);
    if (w->status.greedy_sampling != greedy) {
        w->status.greedy_sampling = greedy;
        agent_wake_locked(w);
    }
    pthread_mutex_unlock(&w->mu);
}

/* Run one user turn until the assistant stops or returns a tool call.  Tool
 * results are appended to the transcript and the loop continues, which gives
 * the model native DSML tool iteration without a client/server protocol. */
static int worker_run_turn(agent_worker *w, const char *user_text) {
    agent_config *cfg = w->cfg;
    ds4_think_mode think_mode = effective_think_mode(cfg);
    pthread_mutex_lock(&w->mu);
    w->interrupt = false;
    w->status.error[0] = '\0';
    agent_wake_locked(w);
    pthread_mutex_unlock(&w->mu);

    char compact_err[160] = {0};
    if (!agent_worker_compact_if_needed(w, "soft limit before user turn",
                                        compact_err, sizeof(compact_err)))
    {
        if (agent_err_is_interrupted(compact_err)) {
            worker_clear_interrupt(w);
            agent_set_status(w, AGENT_WORKER_IDLE);
            return 0;
        }
        agent_set_error(w, compact_err[0] ? compact_err : "context compaction failed");
        return 1;
    }
    agent_worker_maybe_append_datetime_context(w);
    agent_trace_text(w, "user", user_text ? user_text : "",
                     user_text ? strlen(user_text) : 0);
    if (!w->session_title) {
        w->session_title = agent_session_title_from_prompt(user_text, 0);
        w->session_created_at = (uint64_t)time(NULL);
        agent_session_identity_sha(w->session_title, w->session_created_at,
                                   w->session_sha);
    }
    ds4_chat_append_message(w->engine, &w->transcript, "user", user_text);

    uint64_t rng = cfg->gen.seed ? cfg->gen.seed :
        ((uint64_t)time(NULL) ^ ((uint64_t)getpid() << 32) ^ (uint64_t)clock());
    pthread_mutex_lock(&w->mu);
    w->user_activity = true;
    w->session_dirty = true;
    agent_wake_locked(w);
    pthread_mutex_unlock(&w->mu);

    /* A user turn may contain any number of assistant/tool/assistant rounds.
     * Coding agents naturally perform long read/edit/test loops, so there is
     * deliberately no artificial "too many tool calls" ceiling here: context
     * pressure, compaction, user Ctrl+C, and the model's final answer are the
     * real stopping conditions.  The transcript is the single source of truth:
     * after a DSML stanza completes we terminate that assistant message, append
     * the tool result as a tool message, then ask the model to continue. */
    for (int tool_round = 0; ; tool_round++) {
        if (tool_round > 0 &&
            !agent_worker_compact_if_needed(w, "soft limit before tool continuation",
                                            compact_err, sizeof(compact_err)))
        {
            if (agent_err_is_interrupted(compact_err)) {
                worker_clear_interrupt(w);
                agent_set_status(w, AGENT_WORKER_IDLE);
                return 0;
            }
            agent_set_error(w, compact_err[0] ? compact_err : "context compaction failed");
            return 1;
        }
        agent_worker_maybe_append_system_prompt_reminder(w);
        ds4_chat_append_assistant_prefix(w->engine, &w->transcript, think_mode);

        const ds4_tokens *prompt_for_sync = &w->transcript;
        int old_pos = ds4_session_pos(w->session);
        int common = ds4_session_common_prefix(w->session, &w->transcript);
        int cached = common == old_pos && w->transcript.len >= old_pos ? common : 0;

        int suffix = prompt_for_sync->len - cached;
        agent_trace(w, "prefill tool_round=%d transcript=%d prompt=%d cached=%d suffix=%d think=%s",
                    tool_round, w->transcript.len, prompt_for_sync->len,
                    cached, suffix, ds4_think_mode_name(think_mode));
        agent_trace_tokens(w, "prefill_suffix", prompt_for_sync, cached);

        pthread_mutex_lock(&w->mu);
        unsigned prefill_label = w->status.state == AGENT_WORKER_PREFILL ?
            w->status.prefill_label : agent_next_prefill_label();
        w->status.state = AGENT_WORKER_PREFILL;
        w->progress_base = cached;
        w->progress_direct = false;
        w->progress_started_at = now_sec();
        w->status.prefill_done = 0;
        w->status.prefill_total = suffix;
        w->status.prefill_label = prefill_label;
        w->status.prefill_tps = 0.0;
        w->status.generated = 0;
        w->status.gen_tps = 0.0;
        w->status.greedy_sampling = false;
        agent_wake_locked(w);
        pthread_mutex_unlock(&w->mu);

        char err[160];
        ds4_session_set_progress(w->session, worker_progress_cb, w);
        ds4_session_set_display_progress(w->session, worker_progress_cb, w);
        ds4_session_set_cancel(w->session, worker_cancel_session_cb, w);
        double t_sync0 = now_sec();
        int sync_rc = ds4_session_sync(w->session, prompt_for_sync, err, sizeof(err));
        double t_sync1 = now_sec();
        ds4_session_set_cancel(w->session, NULL, NULL);
        ds4_session_set_progress(w->session, NULL, NULL);
        ds4_session_set_display_progress(w->session, NULL, NULL);
        agent_trace(w, "prefill sync done tool_round=%d prompt=%d cached=%d suffix=%d rc=%d %.3f ms",
                    tool_round, prompt_for_sync->len, cached, suffix,
                    sync_rc, (t_sync1 - t_sync0) * 1000.0);
        if (sync_rc == DS4_SESSION_SYNC_INTERRUPTED) {
            agent_publish_system_status(
                w, "Model reading interrupted; the model may only be aware of the prefix processed so far.");
            agent_worker_append_assistant_turn_end(w);
            worker_clear_interrupt(w);
            agent_set_status(w, AGENT_WORKER_IDLE);
            return 0;
        }
        if (sync_rc != 0) {
            agent_set_error(w, err);
            return 1;
        }

        int max_tokens = cfg->gen.n_predict;
        int room = ds4_session_ctx(w->session) - ds4_session_pos(w->session);
        if (room <= 1) max_tokens = 0;
        else if (max_tokens > room - 1) max_tokens = room - 1;

        bool use_color = isatty(STDOUT_FILENO) != 0;
        agent_token_renderer renderer = {
            .engine = w->engine,
            .worker = w,
            .format_thinking = ds4_think_mode_enabled(think_mode),
            .format_markdown = use_color,
            .in_think = ds4_think_mode_enabled(think_mode),
            .use_color = use_color,
            .last_output_newline = true,
        };
        agent_tool_syntax tool_syntax = agent_tool_syntax_for_engine(w->engine);
        agent_dsml_parser dsml = {
            .syntax = tool_syntax,
            .state = AGENT_DSML_SEARCH,
        };
        agent_stream_renderer stream = {
            .renderer = &renderer,
            .parser = &dsml,
            .syntax = tool_syntax,
            .in_think = ds4_think_mode_enabled(think_mode),
        };
        agent_edit_upto_forcer upto_forcer = {0};
        bool got_tool = false;
        bool malformed_tool = false;
        bool early_tool_error = false;
        int generated = 0;
        double t0 = now_sec();
        pthread_mutex_lock(&w->mu);
        w->status.state = AGENT_WORKER_GENERATING;
        w->status.prefill_tps = 0.0;
        w->status.greedy_sampling = false;
        agent_wake_locked(w);
        pthread_mutex_unlock(&w->mu);

        bool status_greedy_sampling = false;
        while (generated < max_tokens && !worker_should_interrupt(w)) {
            worker_apply_pending_power(w);
            bool greedy_sampling = agent_stream_wants_greedy_sampling(&stream);
            if (greedy_sampling != status_greedy_sampling) {
                worker_set_greedy_sampling(w, greedy_sampling);
                status_greedy_sampling = greedy_sampling;
            }
            int token = worker_sample_with_mode(w, cfg, greedy_sampling, &rng);
            if (ds4_token_is_stop_for_think_mode(w->engine, token, think_mode)) {
                if (tool_syntax == AGENT_TOOL_SYNTAX_GLM &&
                    token != ds4_token_eos(w->engine)) {
                    agent_trace(w, "glm assistant generation stopped before control token id=%d", token);
                }
                break;
            }

            size_t text_len = 0;
            char *text = ds4_token_text(w->engine, token, &text_len);
            if (agent_edit_upto_forcer_should_replace(&upto_forcer, &dsml,
                                                       text, text_len))
            {
                agent_trace(w, "edit old auto-upto replaced token=%d text=%.*s",
                            token, (int)(text_len > 80 ? 80 : text_len), text);
                free(text);
                if (worker_force_generated_text(w, "[upto]\n", max_tokens,
                                                &generated, t0, &stream,
                                                err, sizeof(err)) != 0) {
                    agent_dsml_parser_free(&dsml);
                    agent_set_error(w, err);
                    return 1;
                }
            } else {
                free(text);
                if (worker_accept_generated_token(w, token, &generated, t0,
                                                  &stream, err, sizeof(err)) != 0) {
                    agent_dsml_parser_free(&dsml);
                    agent_set_error(w, err);
                    return 1;
                }
            }

            greedy_sampling = agent_stream_wants_greedy_sampling(&stream);
            if (greedy_sampling != status_greedy_sampling) {
                worker_set_greedy_sampling(w, greedy_sampling);
                status_greedy_sampling = greedy_sampling;
            }

            if (dsml.state == AGENT_DSML_DONE) {
                got_tool = true;
                break;
            }
            if (stream.tool_preflight_error) {
                early_tool_error = true;
                break;
            }
            if (dsml.state == AGENT_DSML_ERROR) {
                malformed_tool = true;
                break;
            }
            if (stream.dsml_in_think) {
                malformed_tool = true;
                break;
            }
        }

        bool interrupted = worker_should_interrupt(w);
        agent_stream_text(&stream, NULL, 0, true);
        renderer_finish(&renderer);
        worker_set_greedy_sampling(w, false);
        if (!got_tool && dsml.state == AGENT_DSML_DONE && dsml.calls.len > 0)
            got_tool = true;
        if (interrupted) {
            agent_worker_append_assistant_turn_end(w);
            agent_dsml_parser_free(&dsml);
            agent_publish_system_status(w, "Stopped by user");
            worker_clear_interrupt(w);
            agent_set_status(w, AGENT_WORKER_IDLE);
            return 0;
        }
        if (stream.dsml_in_think) {
            got_tool = false;
            malformed_tool = true;
            early_tool_error = false;
            snprintf(dsml.error, sizeof(dsml.error),
                     "tool calling is not allowed inside <think></think>");
        } else if (!malformed_tool && dsml.state == AGENT_DSML_ERROR) {
            malformed_tool = true;
        } else if (!got_tool && !malformed_tool && !early_tool_error &&
                   !interrupted &&
                   (dsml.state == AGENT_DSML_STRUCTURAL ||
                    dsml.state == AGENT_DSML_PARAM_VALUE))
        {
            malformed_tool = true;
            snprintf(dsml.error, sizeof(dsml.error),
                     tool_syntax == AGENT_TOOL_SYNTAX_GLM ?
                     "incomplete GLM tool call" :
                     "incomplete DSML tool call");
        }

        agent_worker_append_assistant_turn_end(w);

        if (!got_tool && !malformed_tool && !early_tool_error) {
            agent_dsml_parser_free(&dsml);
            agent_set_status(w, AGENT_WORKER_IDLE);
            return 0;
        }

        char *tool_result;
        if (early_tool_error) {
            agent_buf b = {0};
            agent_buf_puts(&b, "Tool error: ");
            agent_buf_puts(&b, stream.tool_preflight_error_msg[0] ?
                           stream.tool_preflight_error_msg :
                           "edit old selector failed before new was generated");
            agent_buf_puts(&b, "\n");
            tool_result = agent_buf_take(&b);
        } else if (malformed_tool) {
            agent_buf b = {0};
            agent_buf_puts(&b, tool_syntax == AGENT_TOOL_SYNTAX_GLM ?
                           "Tool error: invalid GLM tool call: " :
                           "Tool error: invalid DSML tool call: ");
            agent_buf_puts(&b, dsml.error[0] ? dsml.error : "parse error");
            agent_buf_puts(&b, "\n");
            agent_buf_puts(&b, tool_syntax == AGENT_TOOL_SYNTAX_GLM ?
                           agent_glm_syntax_reminder :
                           agent_dsml_syntax_reminder);
            tool_result = agent_buf_take(&b);
        } else {
            tool_result = agent_execute_tool_calls(w, &dsml.calls);
        }
        int projected_tokens = 0;
        int result_reserve = agent_tool_result_reserve_tokens(w);
        if (!agent_tool_result_fits_context(w, tool_result, result_reserve,
                                            &projected_tokens))
        {
            if (!agent_worker_compact(w, "tool result would exceed context",
                                      compact_err, sizeof(compact_err)))
            {
                free(tool_result);
                agent_dsml_parser_free(&dsml);
                if (agent_err_is_interrupted(compact_err)) {
                    worker_clear_interrupt(w);
                    agent_set_status(w, AGENT_WORKER_IDLE);
                    return 0;
                }
                agent_set_error(w, compact_err[0] ? compact_err : "context compaction failed");
                return 1;
            }
            if (!agent_tool_result_fits_context(w, tool_result, result_reserve,
                                                &projected_tokens))
            {
                free(tool_result);
                agent_buf b = {0};
                char msg[256];
                snprintf(msg, sizeof(msg),
                         "Tool error: tool result still does not fit after context compaction "
                         "(projected_prompt=%d tokens, ctx=%d, reserve=%d). "
                         "Retry with a smaller read/search/bash output.\n",
                         projected_tokens, agent_worker_effective_ctx_size(w),
                         result_reserve);
                agent_buf_puts(&b, msg);
                tool_result = agent_buf_take(&b);
                if (!agent_tool_result_fits_context(w, tool_result, 16, NULL)) {
                    free(tool_result);
                    agent_dsml_parser_free(&dsml);
                    agent_set_error(w, "context full after compaction");
                    return 1;
                }
            }
        }
        ds4_chat_append_message(w->engine, &w->transcript, "tool", tool_result);
        free(tool_result);
        agent_dsml_parser_free(&dsml);

        char *queued_user = worker_request_queued_user_drain(w);
        if (queued_user && queued_user[0]) {
            agent_trace_text(w, "queued_user", queued_user, strlen(queued_user));
            ds4_chat_append_message(w->engine, &w->transcript, "user", queued_user);
            pthread_mutex_lock(&w->mu);
            w->user_activity = true;
            w->session_dirty = true;
            agent_wake_locked(w);
            pthread_mutex_unlock(&w->mu);
        }
        free(queued_user);
    }
}

static int worker_run_raw_prompt(agent_worker *w, const char *user_text) {
    agent_config *cfg = w->cfg;
    pthread_mutex_lock(&w->mu);
    w->interrupt = false;
    w->status.error[0] = '\0';
    agent_wake_locked(w);
    pthread_mutex_unlock(&w->mu);

    ds4_tokens prompt = {0};
    ds4_tokenize_text(w->engine, user_text ? user_text : "", &prompt);
    if (prompt.len <= 0) {
        ds4_tokens_free(&prompt);
        agent_set_error(w, "raw prompt tokenized to no tokens");
        return 1;
    }

    agent_trace_text(w, "raw-prompt", user_text ? user_text : "",
                     user_text ? strlen(user_text) : 0);
    agent_trace_tokens(w, "raw_prompt_tokens", &prompt, 0);

    char err[160] = {0};
    int sync_rc = agent_worker_sync_tokens(w, &prompt, true, err, sizeof(err));
    if (sync_rc == DS4_SESSION_SYNC_INTERRUPTED) {
        worker_clear_interrupt(w);
        agent_set_status(w, AGENT_WORKER_IDLE);
        ds4_tokens_free(&prompt);
        return 0;
    }
    if (sync_rc != 0) {
        agent_set_error(w, err[0] ? err : "raw prompt processing failed");
        ds4_tokens_free(&prompt);
        return 1;
    }

    ds4_tokens_free(&w->transcript);
    ds4_tokens_copy(&w->transcript, &prompt);

    int max_tokens = cfg->gen.n_predict;
    int room = ds4_session_ctx(w->session) - ds4_session_pos(w->session);
    if (room <= 1) max_tokens = 0;
    else if (max_tokens > room - 1) max_tokens = room - 1;

    uint64_t rng = cfg->gen.seed ? cfg->gen.seed :
        ((uint64_t)time(NULL) ^ ((uint64_t)getpid() << 32) ^ (uint64_t)clock());
    int generated = 0;
    double t0 = now_sec();

    pthread_mutex_lock(&w->mu);
    w->status.state = AGENT_WORKER_GENERATING;
    w->status.prefill_tps = 0.0;
    w->status.generated = 0;
    w->status.gen_tps = 0.0;
    w->status.greedy_sampling = cfg->gen.temperature <= 0.0f;
    agent_wake_locked(w);
    pthread_mutex_unlock(&w->mu);

    while (generated < max_tokens && !worker_should_interrupt(w)) {
        int token = ds4_session_sample(w->session,
                                       cfg->gen.temperature,
                                       0,
                                       cfg->gen.top_p,
                                       cfg->gen.min_p,
                                       &rng);
        if (ds4_token_is_stop(w->engine, token)) break;

        size_t text_len = 0;
        char *text = ds4_token_text(w->engine, token, &text_len);
        agent_trace_token(w, token, text, text_len, generated + 1);

        if (ds4_session_eval(w->session, token, err, sizeof(err)) != 0) {
            free(text);
            agent_set_error(w, err[0] ? err : "raw decode failed");
            ds4_tokens_free(&prompt);
            return 1;
        }

        ds4_tokens_push(&w->transcript, token);
        agent_publish(w, text, text_len);
        free(text);
        generated++;

        double dt = now_sec() - t0;
        pthread_mutex_lock(&w->mu);
        w->status.generated = generated;
        w->status.gen_tps = dt > 0.0 ? (double)generated / dt : 0.0;
        agent_wake_locked(w);
        pthread_mutex_unlock(&w->mu);
    }

    if (worker_should_interrupt(w)) {
        worker_clear_interrupt(w);
        agent_set_status(w, AGENT_WORKER_IDLE);
        ds4_tokens_free(&prompt);
        return 0;
    }

    agent_set_status(w, AGENT_WORKER_IDLE);
    ds4_tokens_free(&prompt);
    return 0;
}

static void worker_request_save(agent_worker *w) {
    pthread_mutex_lock(&w->mu);
    w->save_requested = true;
    pthread_cond_signal(&w->cond);
    agent_wake_locked(w);
    pthread_mutex_unlock(&w->mu);
}

static void worker_request_compact(agent_worker *w) {
    pthread_mutex_lock(&w->mu);
    w->compact_requested = true;
    pthread_cond_signal(&w->cond);
    agent_wake_locked(w);
    pthread_mutex_unlock(&w->mu);
}

static void worker_request_power(agent_worker *w, int power) {
    pthread_mutex_lock(&w->mu);
    w->requested_power = power;
    w->power_requested = true;
    w->status.power_percent = power;
    pthread_cond_signal(&w->cond);
    agent_wake_locked(w);
    pthread_mutex_unlock(&w->mu);
}

static bool worker_take_save_requested(agent_worker *w) {
    pthread_mutex_lock(&w->mu);
    bool requested = w->save_requested;
    w->save_requested = false;
    pthread_mutex_unlock(&w->mu);
    return requested;
}

static bool worker_take_compact_requested(agent_worker *w) {
    pthread_mutex_lock(&w->mu);
    bool requested = w->compact_requested;
    w->compact_requested = false;
    pthread_mutex_unlock(&w->mu);
    return requested;
}

static bool worker_take_power_requested(agent_worker *w, int *power) {
    pthread_mutex_lock(&w->mu);
    bool requested = w->power_requested;
    if (requested) {
        if (power) *power = w->requested_power;
        w->power_requested = false;
    }
    pthread_mutex_unlock(&w->mu);
    return requested;
}

static void worker_apply_pending_power(agent_worker *w) {
    int power = 0;
    if (!worker_take_power_requested(w, &power)) return;
    if (ds4_session_set_power(w->session, power) != 0) {
        agent_publishf(w, "\npower change failed\n");
        return;
    }
    pthread_mutex_lock(&w->mu);
    w->cfg->engine.power_percent = power;
    w->status.power_percent = power;
    agent_wake_locked(w);
    pthread_mutex_unlock(&w->mu);
}

static void worker_run_deferred_save(agent_worker *w) {
    if (!worker_take_save_requested(w)) return;
    agent_set_status(w, AGENT_WORKER_SAVING);
    char err[160] = {0};
    char sha[41];
    int tokens = 0;
    if (agent_worker_save_session_now(w, sha, &tokens, err, sizeof(err)))
        agent_publishf(w, "\nsaved session %.8s (%d tokens)\n", sha, tokens);
    else
        agent_publishf(w, "\nsave failed: %s\n", err[0] ? err : "unknown error");
    agent_set_status(w, AGENT_WORKER_IDLE);
}

static void worker_run_deferred_compact(agent_worker *w) {
    if (!worker_take_compact_requested(w)) return;
    if (!agent_worker_has_user_session(w)) {
        agent_publishf(w, "\ncompact skipped: nothing to compact\n");
        return;
    }

    int before = w->transcript.len;
    char err[160] = {0};
    if (agent_worker_compact(w, "user requested compaction", err, sizeof(err))) {
        if (w->transcript.len != before) {
            pthread_mutex_lock(&w->mu);
            w->session_dirty = true;
            agent_wake_locked(w);
            pthread_mutex_unlock(&w->mu);
        } else {
            agent_publishf(w, "\ncompact skipped: nothing to compact\n");
        }
        agent_set_status(w, AGENT_WORKER_IDLE);
    } else {
        if (agent_err_is_interrupted(err)) {
            worker_clear_interrupt(w);
            agent_set_status(w, AGENT_WORKER_IDLE);
            return;
        }
        agent_set_error(w, err[0] ? err : "context compaction failed");
    }
}

/* Worker thread entry point.  The UI thread submits plain user text; this
 * thread owns all DS4 session mutation, tool execution, and compaction. */
static void *worker_main(void *arg) {
    agent_worker *w = arg;
    agent_trace(w, "agent worker start ctx=%d backend=%s model=%s trace=%s",
                agent_worker_effective_ctx_size(w),
                ds4_backend_name(w->cfg->engine.backend),
                w->cfg->engine.model_path ? w->cfg->engine.model_path : "",
                w->cfg->gen.trace_path ? w->cfg->gen.trace_path : "");
    char init_err[160] = {0};
    bool init_ok = agent_worker_wait_distributed_route(w, init_err, sizeof(init_err));
    if (init_ok && !w->cfg->gen.raw_prompt)
        init_ok = agent_worker_reset_to_sysprompt(w, init_err, sizeof(init_err));
    if (!init_ok) {
        agent_set_error(w, init_err[0] ? init_err : "failed to initialize system prompt");
    }
    if (w->cfg->gen.raw_prompt)
        agent_trace(w, "raw one-shot prompt mode: skipping agent system prompt");
    else
        agent_trace_tokens(w, "initial_system_prompt", &w->transcript, 0);
    pthread_mutex_lock(&w->mu);
    w->initialized = true;
    agent_wake_locked(w);
    pthread_mutex_unlock(&w->mu);

    while (true) {
        pthread_mutex_lock(&w->mu);
        while (!w->stop && !w->cmd_text && !w->save_requested &&
               !w->compact_requested && !w->power_requested)
            pthread_cond_wait(&w->cond, &w->mu);
        if (w->stop) {
            pthread_mutex_unlock(&w->mu);
            break;
        }
        if (w->power_requested) {
            pthread_mutex_unlock(&w->mu);
            worker_apply_pending_power(w);
            continue;
        }
        if (!w->cmd_text && w->save_requested) {
            pthread_mutex_unlock(&w->mu);
            worker_run_deferred_save(w);
            continue;
        }
        if (!w->cmd_text && w->compact_requested) {
            pthread_mutex_unlock(&w->mu);
            worker_run_deferred_compact(w);
            continue;
        }
        char *cmd = w->cmd_text;
        w->cmd_text = NULL;
        pthread_mutex_unlock(&w->mu);

        if (w->cfg->gen.raw_prompt)
            worker_run_raw_prompt(w, cmd);
        else
            worker_run_turn(w, cmd);
        free(cmd);
        worker_apply_pending_power(w);
        worker_run_deferred_compact(w);
        worker_run_deferred_save(w);
    }

    agent_set_status(w, AGENT_WORKER_STOPPED);
    return NULL;
}

/* ============================================================================
 * Worker/UI Synchronization Helpers
 * ============================================================================
 */

static int set_nonblock(int fd, bool on, int *old_flags) {
    int flags = fcntl(fd, F_GETFL, 0);
    if (flags < 0) return -1;
    if (old_flags) *old_flags = flags;
    int next = on ? (flags | O_NONBLOCK) : (flags & ~O_NONBLOCK);
    return fcntl(fd, F_SETFL, next);
}

/* Check and clear the raw_mode_needs_restore flag under the worker mutex.
 * Returns true if the UI thread should verify/reapply linenoise raw mode. */
static bool worker_check_raw_mode_restore(agent_worker *w) {
    bool needs = false;
    pthread_mutex_lock(&w->mu);
    if (w->raw_mode_needs_restore) {
        w->raw_mode_needs_restore = false;
        needs = true;
    }
    pthread_mutex_unlock(&w->mu);
    return needs;
}

static void drain_wake_fd(int fd) {
    char buf[128];
    for (;;) {
        ssize_t n = read(fd, buf, sizeof(buf));
        if (n > 0) continue;
        if (n < 0 && errno == EINTR) continue;
        break;
    }
}

/* Submit one user turn if the worker is idle.  Busy submissions are rejected so
 * the UI can keep the typed text editable instead of silently queueing it. */
static bool worker_submit(agent_worker *w, const char *text) {
    pthread_mutex_lock(&w->mu);
    bool ok = w->initialized && w->status.state == AGENT_WORKER_IDLE && !w->cmd_text;
    if (ok) {
        w->cmd_text = xstrdup(text);
        /* A submitted turn is no longer idle, even if the worker thread has
         * not yet reached its real prefill accounting.  Non-interactive mode
         * depends on this to avoid exiting in the small handoff window between
         * accepting stdin and starting generation. */
        w->status.state = AGENT_WORKER_PREFILL;
        w->progress_direct = false;
        w->status.prefill_done = 0;
        w->status.prefill_total = 0;
        w->status.prefill_label = agent_next_prefill_label();
        w->status.prefill_tps = 0.0;
        w->status.generated = 0;
        w->status.gen_tps = 0.0;
        w->status.greedy_sampling = false;
        pthread_cond_signal(&w->cond);
    }
    pthread_mutex_unlock(&w->mu);
    return ok;
}

static int worker_status_power_locked(agent_worker *w) {
    if (w->power_requested) return w->requested_power;
    int power = w->cfg->engine.power_percent;
    return power > 0 ? power : 100;
}

/* Request interruption at the next model/tool polling point. */
static void worker_interrupt(agent_worker *w) {
    pthread_mutex_lock(&w->mu);
    w->interrupt = true;
    if (w->cfg &&
        w->cfg->engine.distributed.role == DS4_DISTRIBUTED_COORDINATOR &&
        (w->status.state == AGENT_WORKER_PREFILL ||
         w->status.state == AGENT_WORKER_GENERATING ||
         w->status.state == AGENT_WORKER_COMPACTING))
    {
        w->status.state = AGENT_WORKER_DRAINING;
        w->status.prefill_tps = 0.0;
        w->status.greedy_sampling = false;
        agent_wake_locked(w);
    }
    pthread_mutex_unlock(&w->mu);
}

/* Stop the worker thread. */
static void worker_stop(agent_worker *w) {
    pthread_mutex_lock(&w->mu);
    w->stop = true;
    pthread_cond_signal(&w->cond);
    pthread_mutex_unlock(&w->mu);
}

/* The UI thread consumes output in batches.  Taking ownership of w->out under
 * the mutex keeps terminal writes outside the lock while preserving order. */
static void worker_consume(agent_worker *w, char **out, size_t *out_len, agent_status *status) {
    pthread_mutex_lock(&w->mu);
    if (out) {
        *out = w->out;
        *out_len = w->out_len;
        w->out = NULL;
        w->out_len = 0;
        w->out_cap = 0;
    }
    w->status.ctx_used = w->transcript.len;
    w->status.ctx_size = agent_worker_effective_ctx_size(w);
    w->status.power_percent = worker_status_power_locked(w);
    if (status) *status = w->status;
    w->wake_pending = false;
    pthread_mutex_unlock(&w->mu);
}

static void worker_get_status(agent_worker *w, agent_status *status) {
    pthread_mutex_lock(&w->mu);
    w->status.ctx_used = w->transcript.len;
    w->status.ctx_size = agent_worker_effective_ctx_size(w);
    w->status.power_percent = worker_status_power_locked(w);
    *status = w->status;
    pthread_mutex_unlock(&w->mu);
}

static bool worker_is_idle(agent_worker *w) {
    pthread_mutex_lock(&w->mu);
    bool idle = w->initialized &&
        (w->status.state == AGENT_WORKER_IDLE ||
         w->status.state == AGENT_WORKER_ERROR);
    pthread_mutex_unlock(&w->mu);
    return idle;
}

static bool worker_is_initialized(agent_worker *w, agent_status *status) {
    pthread_mutex_lock(&w->mu);
    w->status.ctx_used = w->transcript.len;
    w->status.ctx_size = agent_worker_effective_ctx_size(w);
    w->status.power_percent = worker_status_power_locked(w);
    if (status) *status = w->status;
    bool initialized = w->initialized;
    pthread_mutex_unlock(&w->mu);
    return initialized;
}

static bool stdout_is_tty(void) {
    return isatty(STDOUT_FILENO) != 0;
}

static char *agent_format_user_prompt_echo(const char *text) {
    agent_buf b = {0};
    if (stdout_is_tty()) {
        agent_buf_puts(&b, "\x1b[1;91m*\x1b[1;97m ");
        agent_buf_puts(&b, text);
        agent_buf_puts(&b, "\x1b[0m\n\n");
    } else {
        agent_buf_puts(&b, "* ");
        agent_buf_puts(&b, text);
        agent_buf_puts(&b, "\n\n");
    }
    return agent_buf_take(&b);
}

static void agent_echo_user_prompt(const char *text) {
    char *msg = agent_format_user_prompt_echo(text);
    printf("%s", msg);
    fflush(stdout);
    free(msg);
}

/* ============================================================================
 * Terminal Prompt, Status Footer, And Async Output Rendering
 * ============================================================================
 */

static void agent_format_ctx_size(int ctx_size, char *buf, size_t len);
#define AGENT_INPUT_INITIAL_BUFLEN 4096
#define AGENT_INPUT_MAX_BUFLEN (1024*1024)
#define AGENT_STATUS_STYLE_START "\x1b[48;5;238;38;5;252m"
#define AGENT_STATUS_STYLE_END "\x1b[0m"
#define AGENT_STATUS_BAR_FILL "\x1b[48;5;238;38;5;201;1m"
#define AGENT_QUEUE_STYLE "\x1b[38;5;87;1m"
#define AGENT_STATUS_REDRAW_INTERVAL_SEC 0.20
#define AGENT_PROGRESS_BAR_WIDTH 32
#define AGENT_PROGRESS_BAR_MAX_BYTES 256

static void agent_progress_append(char *buf, size_t len, size_t *pos,
                                  const char *s) {
    if (len == 0 || *pos >= len - 1) return;
    size_t avail = len - *pos;
    int n = snprintf(buf + *pos, avail, "%s", s);
    if (n <= 0) return;
    if ((size_t)n >= avail) *pos = len - 1;
    else *pos += (size_t)n;
}

static void build_prompt_text(const agent_status *st, char *buf, size_t len) {
    (void)st;
    snprintf(buf, len, "ds4-agent> ");
}

static void agent_progress_bar(int done, int total, double tps,
                               char *buf, size_t len, bool color) {
    if (len == 0) return;
    if (total <= 0) total = 1;
    if (done < 0) done = 0;
    if (done > total) done = total;
    int filled = (int)(((long long)done * AGENT_PROGRESS_BAR_WIDTH) / total);
    if (filled < 0) filled = 0;
    if (filled > AGENT_PROGRESS_BAR_WIDTH) filled = AGENT_PROGRESS_BAR_WIDTH;
    if (color && filled == 0 && done < total) filled = 1;
    char rate[32] = {0};
    size_t rate_len = 0;
    if (tps > 0.0 && filled < AGENT_PROGRESS_BAR_WIDTH) {
        snprintf(rate, sizeof(rate), " %.0ft/s", tps);
        rate_len = strlen(rate);
    }
    size_t pos = 0;
    agent_progress_append(buf, len, &pos, "[");
    if (color) agent_progress_append(buf, len, &pos, AGENT_STATUS_BAR_FILL);
    for (int i = 0; i < AGENT_PROGRESS_BAR_WIDTH && pos + 1 < len; i++) {
        if (color && i == filled) {
            agent_progress_append(buf, len, &pos, AGENT_STATUS_STYLE_START);
        }
        if (i >= filled && rate_len > 0 && (size_t)(i - filled) < rate_len) {
            char ch[2] = {rate[i - filled], '\0'};
            agent_progress_append(buf, len, &pos, ch);
        } else {
            agent_progress_append(buf, len, &pos, i < filled ? "▶" : "·");
        }
    }
    if (color) agent_progress_append(buf, len, &pos, AGENT_STATUS_STYLE_START);
    agent_progress_append(buf, len, &pos, "]");
    buf[pos < len ? pos : len - 1] = '\0';
}

static void agent_power_status_suffix(const agent_status *st,
                                      char *buf, size_t len) {
    if (len == 0) return;
    if (st->power_percent > 0 && st->power_percent < 100)
        snprintf(buf, len, " | ⚡ %d%%", st->power_percent);
    else
        buf[0] = '\0';
}

static unsigned agent_next_prefill_label(void) {
    static unsigned next;
    return next++;
}

/* Keep each prefill operation on a single playful label so the footer does not
 * visually churn while progress updates stream in. */
static const char *agent_prefill_label(const agent_status *st) {
    static const char *labels[] = {
        "reading",
        "absorbing",
        "studying",
        "gathering",
        "crunching",
        "scrutinizing",
    };
    size_t n = sizeof(labels) / sizeof(labels[0]);
    return labels[(st ? st->prefill_label : 0u) % n];
}

/* Build the one-line footer shown below the prompt.  It is intentionally compact
 * because linenoise redraws it on every progress update. */
static void build_status_text(const agent_status *st, char *buf, size_t len) {
    char used[32], total_ctx[32];
    char power[32];
    agent_format_ctx_size(st->ctx_used, used, sizeof(used));
    agent_format_ctx_size(st->ctx_size, total_ctx, sizeof(total_ctx));
    agent_power_status_suffix(st, power, sizeof(power));

    switch (st->state) {
    case AGENT_WORKER_PREFILL: {
        int done = st->prefill_done;
        int total = st->prefill_total > 0 ? st->prefill_total : 1;
        if (done > total) done = total;
        double pct = 100.0 * (double)done / (double)total;
        char bar[AGENT_PROGRESS_BAR_MAX_BYTES];
        agent_progress_bar(done, total, st->prefill_tps, bar, sizeof(bar),
                           stdout_is_tty());
        snprintf(buf, len, "ctx %s/%s | %s %s %d/%d %.1f%%%s",
                 used, total_ctx, agent_prefill_label(st), bar,
                 done, total, pct, power);
        break;
    }
    case AGENT_WORKER_GENERATING:
        snprintf(buf, len, "ctx %s/%s | generation %d tokens%s %.1f t/s%s",
                 used, total_ctx, st->generated,
                 st->greedy_sampling ? " ❄️" : "", st->gen_tps, power);
        break;
    case AGENT_WORKER_COMPACTING:
        snprintf(buf, len, "ctx %s/%s | COMPACTING summary %d tokens %.1f t/s%s",
                 used, total_ctx, st->generated, st->gen_tps, power);
        break;
    case AGENT_WORKER_DRAINING:
        snprintf(buf, len, "ctx %s/%s | stopping after distributed cluster drains%s",
                 used, total_ctx, power);
        break;
    case AGENT_WORKER_SAVING:
        snprintf(buf, len, "ctx %s/%s | saving session%s", used, total_ctx, power);
        break;
    case AGENT_WORKER_ERROR:
        snprintf(buf, len, "ctx %s/%s | error: %s%s", used, total_ctx,
                 st->error[0] ? st->error : "unknown error", power);
        break;
    case AGENT_WORKER_STOPPED:
        snprintf(buf, len, "ctx %s/%s | interrupted%s", used, total_ctx, power);
        break;
    default:
        snprintf(buf, len, "ctx %s/%s | idle%s", used, total_ctx, power);
        break;
    }
}

typedef struct {
    char **v;
    size_t len;
    size_t cap;
} agent_prompt_queue;

static void agent_prompt_queue_push(agent_prompt_queue *q, const char *text) {
    if (q->len == q->cap) {
        q->cap = q->cap ? q->cap * 2 : 4;
        q->v = xrealloc(q->v, q->cap * sizeof(q->v[0]));
    }
    q->v[q->len++] = xstrdup(text ? text : "");
}

static char *agent_prompt_queue_pop(agent_prompt_queue *q) {
    if (!q->len) return NULL;
    char *text = q->v[0];
    memmove(q->v, q->v + 1, (q->len - 1) * sizeof(q->v[0]));
    q->len--;
    return text;
}

static void agent_prompt_queue_push_front(agent_prompt_queue *q, char *text) {
    if (q->len == q->cap) {
        q->cap = q->cap ? q->cap * 2 : 4;
        q->v = xrealloc(q->v, q->cap * sizeof(q->v[0]));
    }
    memmove(q->v + 1, q->v, q->len * sizeof(q->v[0]));
    q->v[0] = text;
    q->len++;
}

static char *agent_prompt_queue_take_all(agent_prompt_queue *q) {
    if (!q->len) return NULL;
    if (q->len == 1) return agent_prompt_queue_pop(q);

    agent_buf b = {0};
    for (size_t i = 0; i < q->len; i++) {
        char hdr[64];
        if (i) agent_buf_puts(&b, "\n\n");
        snprintf(hdr, sizeof(hdr), "Queued user message %zu:\n", i + 1);
        agent_buf_puts(&b, hdr);
        agent_buf_puts(&b, q->v[i]);
        free(q->v[i]);
    }
    q->len = 0;
    return agent_buf_take(&b);
}

static char *agent_prompt_queue_take_all_echo(agent_prompt_queue *q) {
    if (!q->len) return NULL;
    agent_buf b = {0};
    for (size_t i = 0; i < q->len; i++) {
        char *echo = agent_format_user_prompt_echo(q->v[i]);
        agent_buf_puts(&b, echo);
        free(echo);
    }
    return agent_buf_take(&b);
}

static const char *agent_prompt_queue_peek(const agent_prompt_queue *q) {
    return q->len ? q->v[0] : NULL;
}

static void agent_prompt_queue_free(agent_prompt_queue *q) {
    for (size_t i = 0; i < q->len; i++) free(q->v[i]);
    free(q->v);
    memset(q, 0, sizeof(*q));
}

static bool agent_footer_is_multiline(const char *status) {
    return status && strchr(status, '\n');
}

/* Build the editable footer.  With queued prompts, the footer becomes multiple
 * rows: a compact queue preview first, then the normal status row. */
static void build_footer_text(const agent_status *st, const agent_prompt_queue *queue,
                              int cols, char *buf, size_t len) {
    char status[512];
    build_status_text(st, status, sizeof(status));
    if (!queue || !queue->len) {
        snprintf(buf, len, "%s", status);
        return;
    }

    const char *queued = agent_prompt_queue_peek(queue);
    if (cols < 40) cols = 40;
    int max_rows = 3;
    size_t budget = (size_t)cols * (size_t)max_rows;
    const char *plain_suffix = " (ctrl+x to edit, ESC to send ASAP)";
    size_t queued_len = strlen(queued);
    char more_suffix[160];
    const char *suffix = plain_suffix;
    size_t take = queued_len;
    if (queued_len + strlen(plain_suffix) > budget) {
        size_t reserve = 72;
        take = budget > reserve ? budget - reserve : budget / 2;
        snprintf(more_suffix, sizeof(more_suffix),
                 "... %zu characters more ..., (ctrl+x to edit, ESC to send ASAP)",
                 queued_len - take);
        suffix = more_suffix;
    }

    agent_buf msg = {0};
    agent_buf_puts(&msg, "queued: ");
    for (size_t i = 0; i < take; i++) {
        char c = queued[i];
        if (c == '\n' || c == '\r' || c == '\t') c = ' ';
        agent_buf_append(&msg, &c, 1);
    }
    agent_buf_puts(&msg, suffix);
    char *preview = agent_buf_take(&msg);

    agent_buf out = {0};
    size_t pos = 0, preview_len = strlen(preview);
    for (int row = 0; row < max_rows && pos < preview_len; row++) {
        if (row) agent_buf_puts(&out, "\n");
        if (stdout_is_tty()) agent_buf_puts(&out, AGENT_QUEUE_STYLE);
        size_t part = preview_len - pos;
        if (part > (size_t)cols) part = (size_t)cols;
        agent_buf_append(&out, preview + pos, part);
        if (stdout_is_tty()) agent_buf_puts(&out, "\x1b[0m");
        pos += part;
    }
    agent_buf_puts(&out, "\n");
    if (stdout_is_tty()) agent_buf_puts(&out, AGENT_STATUS_STYLE_START);
    agent_buf_puts(&out, status);
    snprintf(buf, len, "%s", out.ptr ? out.ptr : "");
    free(preview);
    free(out.ptr);
}

typedef struct {
    struct linenoiseState edit;
    char *input;
    char prompt[160];
    char status[4096];
    int old_stdin_flags;
    bool active;
    bool hidden;
    bool output_line_open;
    bool prompt_below_output;
    int output_col;
    bool scroll_region;
    int term_rows;
    int term_cols;
    int output_bottom;
    int prompt_row;
    int reserved_rows;
    bool output_cursor_saved;
    bool output_at_scroll_boundary;
    double last_prompt_redraw_time;
    char cpr_buf[32];
    size_t cpr_len;
    bool paste_open;
    bool paste_start_pending;
    char paste_tail[6];
    size_t paste_tail_len;
} agent_editor;

static void editor_queue_bytes(agent_editor *ed, const char *buf, size_t len);
static void editor_hide(agent_editor *ed);
static void editor_show(agent_editor *ed);

typedef enum {
    CPR_INVALID,
    CPR_PARTIAL,
    CPR_COMPLETE,
} cpr_state;

/* Classify a possible terminal cursor-position reply (ESC[row;colR).  User
 * keystrokes can arrive interleaved with these replies, so we only swallow bytes
 * when they are definitely part of a complete CPR sequence. */
static cpr_state cpr_candidate_state(const char *buf, size_t len) {
    if (len == 0) return CPR_PARTIAL;
    if ((unsigned char)buf[0] != 0x1b) return CPR_INVALID;
    if (len == 1) return CPR_PARTIAL;
    if (buf[1] != '[') return CPR_INVALID;
    if (len == 2) return CPR_PARTIAL;

    size_t p = 2;
    if (buf[p] < '0' || buf[p] > '9') return CPR_INVALID;
    while (p < len && buf[p] >= '0' && buf[p] <= '9') p++;
    if (p == len) return CPR_PARTIAL;
    if (buf[p++] != ';') return CPR_INVALID;
    if (p == len) return CPR_PARTIAL;
    if (buf[p] < '0' || buf[p] > '9') return CPR_INVALID;
    while (p < len && buf[p] >= '0' && buf[p] <= '9') p++;
    if (p == len) return CPR_PARTIAL;
    return p + 1 == len && buf[p] == 'R' ? CPR_COMPLETE : CPR_INVALID;
}

static void editor_flush_cpr_candidate(agent_editor *ed) {
    if (!ed->cpr_len) return;
    linenoiseEditQueueInput(&ed->edit, ed->cpr_buf, ed->cpr_len);
    ed->cpr_len = 0;
}

static bool agent_tail_ends_with(const char *tail, size_t tail_len,
                                 const char *seq, size_t seq_len) {
    return tail_len >= seq_len &&
           memcmp(tail + tail_len - seq_len, seq, seq_len) == 0;
}

static bool agent_tail_has_seq_prefix(const char *tail, size_t tail_len,
                                      const char *seq, size_t seq_len) {
    size_t max = tail_len < seq_len - 1 ? tail_len : seq_len - 1;
    for (size_t n = max; n > 0; n--) {
        if (memcmp(tail + tail_len - n, seq, n) == 0) return true;
    }
    return false;
}

/* Track bracketed paste markers outside linenoise.  The nonblocking event loop
 * may receive a paste in chunks; pausing linenoiseEditFeed() until ESC[201~
 * arrives prevents pasted newlines from being interpreted as Enter. */
static void editor_track_bracketed_paste(agent_editor *ed, char c) {
    static const char start[] = "\x1b[200~";
    static const char end[] = "\x1b[201~";

    if (ed->paste_tail_len == sizeof(ed->paste_tail)) {
        memmove(ed->paste_tail, ed->paste_tail + 1, sizeof(ed->paste_tail) - 1);
        ed->paste_tail_len--;
    }
    ed->paste_tail[ed->paste_tail_len++] = c;

    /* The blocking linenoise() path waits inside linenoiseEditPaste() until it
     * sees ESC[201~. In the agent the outer event loop reads stdin in
     * non-blocking chunks; if we let linenoise start parsing ESC[200~ before
     * the closing marker has arrived, pasted newlines can be interpreted as
     * Enter. Keep feeding bytes into linenoise's queue, but don't call
     * linenoiseEditFeed() while the terminal paste envelope is still open. */
    if (agent_tail_ends_with(ed->paste_tail, ed->paste_tail_len,
                             start, sizeof(start) - 1))
    {
        ed->paste_open = true;
        ed->paste_start_pending = false;
    } else if (agent_tail_ends_with(ed->paste_tail, ed->paste_tail_len,
                                    end, sizeof(end) - 1))
    {
        ed->paste_open = false;
        ed->paste_start_pending = false;
    } else {
        ed->paste_start_pending =
            !ed->paste_open &&
            agent_tail_has_seq_prefix(ed->paste_tail, ed->paste_tail_len,
                                      start, sizeof(start) - 1);
    }
}

/* Separate late CPR replies from real user input before handing bytes to
 * linenoise. */
static void editor_filter_input_byte(agent_editor *ed, char c) {
    if (ed->cpr_len || (unsigned char)c == 0x1b) {
        if (ed->cpr_len == sizeof(ed->cpr_buf)) {
            editor_flush_cpr_candidate(ed);
        }
        ed->cpr_buf[ed->cpr_len++] = c;
        cpr_state st = cpr_candidate_state(ed->cpr_buf, ed->cpr_len);
        if (st == CPR_COMPLETE) {
            ed->cpr_len = 0; /* Late terminal cursor report: discard it. */
        } else if (st == CPR_INVALID) {
            editor_flush_cpr_candidate(ed);
        }
        return;
    }
    linenoiseEditQueueInput(&ed->edit, &c, 1);
}

/* Queue raw terminal bytes into linenoise while preserving paste envelopes and
 * filtering cursor-position replies. */
static void editor_queue_bytes(agent_editor *ed, const char *buf, size_t len) {
    for (size_t i = 0; i < len; i++) {
        editor_track_bracketed_paste(ed, buf[i]);
        editor_filter_input_byte(ed, buf[i]);
    }
}

/* Drain stdin in nonblocking mode.  The outer event loop decides when queued
 * bytes are fed to linenoiseEditFeed(). */
static void editor_read_stdin(agent_editor *ed) {
    char buf[256];
    for (;;) {
        ssize_t n = read(STDIN_FILENO, buf, sizeof(buf));
        if (n > 0) {
            editor_queue_bytes(ed, buf, (size_t)n);
            continue;
        }
        if (n < 0 && errno == EINTR) continue;
        break;
    }
}

static bool editor_take_queued_byte(agent_editor *ed, unsigned char byte) {
    struct linenoiseState *l = &ed->edit;
    for (size_t i = l->queued_input_pos; i < l->queued_input_len; i++) {
        if ((unsigned char)l->queued_input[i] != byte) continue;
        memmove(l->queued_input + i, l->queued_input + i + 1,
                l->queued_input_len - i - 1);
        l->queued_input_len--;
        if (l->queued_input_pos > l->queued_input_len)
            l->queued_input_pos = l->queued_input_len;
        return true;
    }
    return false;
}

static bool editor_take_bare_escape(agent_editor *ed) {
    if (ed->cpr_len == 1 && (unsigned char)ed->cpr_buf[0] == 0x1b) {
        ed->cpr_len = 0;
        return true;
    }
    return false;
}

static void editor_replace_input(agent_editor *ed, const char *text) {
    if (ed->hidden) editor_show(ed);
    linenoiseEditClear(&ed->edit);
    if (text && text[0]) linenoiseEditInsert(&ed->edit, text, strlen(text));
}

/* Fallback cursor tracking for terminals that do not answer CPR quickly.  It is
 * intentionally approximate for wide Unicode; the CPR path handles exact
 * positioning in normal interactive terminals. */
static void editor_note_output(agent_editor *ed, const char *text, size_t len) {
    int cols = ed->edit.cols > 0 ? (int)ed->edit.cols : 80;
    for (size_t i = 0; i < len; i++) {
        size_t start = i;
        unsigned char c = (unsigned char)text[i];
        if (c == 0x1b && i + 1 < len && text[i + 1] == '[') {
            (void)start;
            i += 2;
            while (i < len) {
                unsigned char e = (unsigned char)text[i];
                if (e >= 0x40 && e <= 0x7e) break;
                i++;
            }
            continue;
        }
        if (c == '\n') {
            ed->output_col = 0;
            ed->output_line_open = false;
            continue;
        }
        if (c == '\r') {
            ed->output_col = 0;
            continue;
        }
        if (c == '\b') {
            if (ed->output_col > 0) ed->output_col--;
            continue;
        }

        int width = 1;
        if (c == '\t') {
            width = 8 - (ed->output_col & 7);
        } else if (c < 0x20 || c == 0x7f) {
            width = 0;
        } else if (c >= 0xc0) {
            while (i + 1 < len && (((unsigned char)text[i + 1]) & 0xc0) == 0x80)
                i++;
        } else if ((c & 0xc0) == 0x80) {
            width = 0;
        }

        if (width > 0) {
            ed->output_col = (ed->output_col + width) % cols;
            ed->output_line_open = true;
        }
    }
}

/* Normalize generated LF to CRLF for terminal output without changing the text
 * stored in the transcript. */
static void editor_write_terminal_text(const char *text, size_t len) {
    size_t start = 0;
    for (size_t i = 0; i < len; i++) {
        if (text[i] != '\n') continue;
        if (i > start) write_all(STDOUT_FILENO, text + start, i - start);
        write_all(STDOUT_FILENO, "\r\n", 2);
        start = i + 1;
    }
    if (start < len) write_all(STDOUT_FILENO, text + start, len - start);
}

/* Locate a CPR reply inside a mixed stdin buffer.  Bytes before/after the reply
 * are user input and must be queued back into linenoise. */
static bool find_cpr_reply(const char *buf, size_t len, size_t *start, size_t *end,
                           int *row, int *col) {
    for (size_t i = 0; i + 5 < len; i++) {
        if ((unsigned char)buf[i] != 0x1b || buf[i + 1] != '[') continue;
        size_t p = i + 2;
        int r = 0, c = 0;
        if (p >= len || buf[p] < '0' || buf[p] > '9') continue;
        while (p < len && buf[p] >= '0' && buf[p] <= '9') {
            r = r * 10 + (buf[p++] - '0');
        }
        if (p >= len || buf[p++] != ';') continue;
        if (p >= len || buf[p] < '0' || buf[p] > '9') continue;
        while (p < len && buf[p] >= '0' && buf[p] <= '9') {
            c = c * 10 + (buf[p++] - '0');
        }
        if (p >= len || buf[p] != 'R') continue;
        *start = i;
        *end = p + 1;
        *row = r;
        *col = c;
        return true;
    }
    return false;
}

/* Ask the terminal for the cursor column after writing model output.  Any user
 * bytes read while waiting for the CPR reply are queued back into linenoise so
 * typing during generation is not lost. */
static bool editor_query_cursor(agent_editor *ed, int *col_out) {
    if (!isatty(STDIN_FILENO) || !isatty(STDOUT_FILENO)) return false;

    char buf[512];
    size_t len = 0, start = 0, end = 0;
    int row = 0, col = 0;
    write_all(STDOUT_FILENO, "\x1b[6n", 4);

    for (int attempt = 0; attempt < 8; attempt++) {
        struct pollfd pfd = {.fd = STDIN_FILENO, .events = POLLIN};
        int rc = poll(&pfd, 1, 5);
        if (rc < 0 && errno == EINTR) continue;
        if (rc <= 0) continue;
        for (;;) {
            ssize_t n = read(STDIN_FILENO, buf + len, sizeof(buf) - len);
            if (n > 0) {
                len += (size_t)n;
                if (find_cpr_reply(buf, len, &start, &end, &row, &col)) {
                    if (start) editor_queue_bytes(ed, buf, start);
                    if (end < len) editor_queue_bytes(ed, buf + end, len - end);
                    (void)row;
                    *col_out = col;
                    return col > 0;
                }
                if (len == sizeof(buf)) break;
                continue;
            }
            if (n < 0 && errno == EINTR) continue;
            break;
        }
    }

    if (len) editor_queue_bytes(ed, buf, len);
    return false;
}

static void editor_move_to_output_cursor(agent_editor *ed) {
    char seq[64];
    write_all(STDOUT_FILENO, "\x1b[1A", 4);
    int n = snprintf(seq, sizeof(seq), "\x1b[%dG", ed->output_col + 1);
    if (n > 0) write_all(STDOUT_FILENO, seq, (size_t)n);
}

static bool editor_get_terminal_size(int *rows, int *cols) {
    struct winsize ws;
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) != 0) return false;
    if (ws.ws_row < 1 || ws.ws_col < 1) return false;
    *rows = ws.ws_row;
    *cols = ws.ws_col;
    return true;
}

static void editor_csi_cursor(int row, int col) {
    char seq[64];
    int n = snprintf(seq, sizeof(seq), "\x1b[%d;%dH", row, col);
    if (n > 0) write_all(STDOUT_FILENO, seq, (size_t)n);
}

static void editor_save_output_cursor(agent_editor *ed) {
    if (!ed->scroll_region) return;
    write_all(STDOUT_FILENO, "\0337", 2);
    ed->output_cursor_saved = true;
}

static void editor_restore_output_cursor(agent_editor *ed) {
    if (!ed->scroll_region) return;
    if (ed->output_cursor_saved) {
        write_all(STDOUT_FILENO, "\0338", 2);
    } else {
        editor_csi_cursor(ed->output_bottom, 1);
    }
}

static void editor_move_to_prompt_row(agent_editor *ed) {
    if (!ed->scroll_region) return;
    editor_csi_cursor(ed->prompt_row, 1);
}

static void editor_move_to_prompt_cursor(agent_editor *ed) {
    if (!ed->scroll_region) return;
    if (ed->edit.screen_cursor_row > 0 && ed->edit.screen_cursor_col > 0) {
        editor_csi_cursor(ed->edit.screen_cursor_row, ed->edit.screen_cursor_col);
    } else {
        editor_move_to_prompt_row(ed);
    }
}

static void editor_clear_row(int row) {
    editor_csi_cursor(row, 1);
    write_all(STDOUT_FILENO, "\r\x1b[0K", 5);
}

static void editor_clear_prompt_region(agent_editor *ed) {
    if (!ed->scroll_region) return;
    for (int row = ed->prompt_row; row <= ed->term_rows; row++)
        editor_clear_row(row);

    /* In scroll-region mode ds4-agent owns the absolute prompt/status rows.
     * Clearing them directly is more reliable than asking linenoise to clean
     * relative to whatever cursor position the last worker/status transition
     * left behind.  Reset linenoise's render bookkeeping so the next show is a
     * pure write into the reserved rows. */
    ed->edit.oldrows = 0;
    ed->edit.oldstatusrows = 0;
    ed->edit.oldrpos = 1;
    ed->edit.oldpos = ed->edit.pos;
}

static void editor_set_scroll_margin(int bottom) {
    char seq[96];
    int n = snprintf(seq, sizeof(seq), "\x1b[1;%dr", bottom);
    if (n > 0) write_all(STDOUT_FILENO, seq, (size_t)n);
}

static void editor_scroll_output_up(int bottom, int lines) {
    if (lines <= 0) return;
    editor_set_scroll_margin(bottom);
    editor_csi_cursor(bottom, 1);
    for (int i = 0; i < lines; i++)
        write_all(STDOUT_FILENO, "\n", 1);
}

static bool editor_set_scroll_layout(agent_editor *ed, int reserved_rows,
                                     bool allow_shrink,
                                     bool scroll_on_grow) {
    if (!ed->scroll_region) return false;

    int rows = 0, cols = 0;
    if (!editor_get_terminal_size(&rows, &cols)) return false;
    if (rows < 8 || cols < 20) return false;
    if (reserved_rows < 2) reserved_rows = 2;
    if (reserved_rows > rows - 2) reserved_rows = rows - 2;
    if (!allow_shrink && ed->reserved_rows > 0 &&
        ed->term_rows == rows && ed->term_cols == cols &&
        reserved_rows < ed->reserved_rows)
    {
        reserved_rows = ed->reserved_rows;
    }

    int output_bottom = rows - reserved_rows;
    int prompt_row = output_bottom + 1;
    bool changed = ed->term_rows != rows ||
                   ed->term_cols != cols ||
                   ed->output_bottom != output_bottom ||
                   ed->prompt_row != prompt_row ||
                   ed->reserved_rows != reserved_rows;
    if (!changed) return true;

    /* If the prompt grows, rows that were output rows become prompt rows.  Do
     * not simply clear them: first scroll the old output region upward by the
     * number of newly reserved rows, exactly as if the model had printed more
     * lines.  If the prompt shrinks, no output is restored; the output region
     * simply grows downward and the prompt/status block remains bottom
     * anchored. */
    bool scrolled_output = false;
    if (scroll_on_grow &&
        ed->term_rows == rows && ed->term_cols == cols &&
        ed->output_bottom > 0 && output_bottom < ed->output_bottom)
    {
        editor_scroll_output_up(ed->output_bottom,
                                ed->output_bottom - output_bottom);
        scrolled_output = true;
    }

    editor_set_scroll_margin(output_bottom);

    ed->term_rows = rows;
    ed->term_cols = cols;
    ed->output_bottom = output_bottom;
    ed->prompt_row = prompt_row;
    ed->reserved_rows = reserved_rows;
    ed->output_cursor_saved = false;
    ed->output_at_scroll_boundary = scrolled_output;

    for (int row = prompt_row; row <= rows; row++)
        editor_clear_row(row);

    /* If the prompt grew while generated output was in the middle of a line,
     * the scroll above moved that partial line up with its column intact.
     * Preserve that column when saving the new output cursor; otherwise the
     * next token resumes at column 1 and overwrites the line it was extending. */
    int output_col = ed->output_line_open ? ed->output_col + 1 : 1;
    if (output_col < 1) output_col = 1;
    if (output_col > cols) output_col = cols;
    editor_csi_cursor(output_bottom, output_col);
    editor_save_output_cursor(ed);
    editor_move_to_prompt_row(ed);
    return true;
}

static int editor_linenoise_layout_changed(struct linenoiseState *l,
                                           size_t prompt_rows,
                                           size_t status_rows,
                                           void *privdata) {
    (void)l;
    agent_editor *ed = privdata;
    if (!ed || !ed->scroll_region) return 0;
    if (prompt_rows < 1) prompt_rows = 1;
    int reserved = (int)(prompt_rows + status_rows);
    if (!editor_set_scroll_layout(ed, reserved, true, true)) return 0;
    return ed->prompt_row;
}

/* Keep generated output inside a scroll region that excludes the live prompt
 * and status footer.  This lets terminals scroll model/tool output naturally
 * without rewriting the prompt on every streamed token, which is especially
 * important over SSH where full redraws are visibly expensive. */
static bool editor_configure_scroll_region(agent_editor *ed) {
    if (ed->scroll_region) return true;
    if (!isatty(STDIN_FILENO) || !isatty(STDOUT_FILENO)) return false;

    int rows = 0, cols = 0;
    if (!editor_get_terminal_size(&rows, &cols)) return false;
    if (rows < 8 || cols < 20) return false;

    ed->term_rows = 0;
    ed->term_cols = 0;
    ed->output_bottom = 0;
    ed->prompt_row = 0;
    ed->reserved_rows = 0;
    ed->output_cursor_saved = false;
    ed->output_at_scroll_boundary = false;
    ed->scroll_region = true;
    if (!editor_set_scroll_layout(ed, 2, true, false)) return false;

    /* The agent prints backend startup lines before the editor exists.  Once
     * the scroll region is installed, create an append line at the bottom of
     * that region instead of guessing that the old terminal cursor was already
     * there.  Without this first scroll, the first agent/model output can
     * overwrite the last visible startup line. */
    editor_scroll_output_up(ed->output_bottom, 1);
    ed->output_cursor_saved = false;
    editor_csi_cursor(ed->output_bottom, 1);
    editor_save_output_cursor(ed);
    editor_move_to_prompt_row(ed);
    return true;
}

static void editor_restore_terminal_layout(agent_editor *ed) {
    if (!ed->scroll_region) return;
    write_all(STDOUT_FILENO, "\x1b[0m", 4);
    write_all(STDOUT_FILENO, "\x1b[r", 3);
    editor_csi_cursor(ed->term_rows, 1);
    write_all(STDOUT_FILENO, "\r\x1b[0K\r\n", 7);
    ed->scroll_region = false;
    ed->output_cursor_saved = false;
    ed->term_rows = ed->term_cols = 0;
    ed->output_bottom = ed->prompt_row = 0;
    ed->reserved_rows = 0;
    ed->output_at_scroll_boundary = false;
}

/* Start linenoise in nonblocking mode and install the status footer. */
static int editor_start(agent_editor *ed, const char *prompt,
                        const char *status, const char *initial) {
    char *input = xmalloc(AGENT_INPUT_INITIAL_BUFLEN);
    snprintf(ed->prompt, sizeof(ed->prompt), "%s", prompt);
    snprintf(ed->status, sizeof(ed->status), "%s", status ? status : "");
    bool had_scroll_region = ed->scroll_region;
    bool use_scroll_region = editor_configure_scroll_region(ed);
    if (use_scroll_region) {
        if (had_scroll_region)
            editor_set_scroll_layout(ed, 2, true, false);
        editor_move_to_prompt_row(ed);
    }
    if (linenoiseEditStart(&ed->edit, STDIN_FILENO, STDOUT_FILENO,
                           input, AGENT_INPUT_INITIAL_BUFLEN, ed->prompt) != 0)
    {
        editor_restore_terminal_layout(ed);
        free(input);
        return -1;
    }
    bool embedded_status = agent_footer_is_multiline(ed->status);
    const char *status_start = stdout_is_tty() && !embedded_status ?
        AGENT_STATUS_STYLE_START : "";
    const char *status_end = stdout_is_tty() && ed->status[0] ?
        AGENT_STATUS_STYLE_END : "";
    linenoiseEditSetStatus(&ed->edit, ed->status,
                           status_start, status_end);
    linenoiseEditSetLayoutCallback(&ed->edit, editor_linenoise_layout_changed, ed);
    if (isatty(ed->edit.ifd) || getenv("LINENOISE_ASSUME_TTY")) {
        linenoiseHide(&ed->edit);
        linenoiseShow(&ed->edit);
    }
    ed->input = input;
    ed->edit.buflen_max = AGENT_INPUT_MAX_BUFLEN;
    ed->active = true;
    if (set_nonblock(STDIN_FILENO, true, &ed->old_stdin_flags) != 0)
        ed->old_stdin_flags = -1;
    if (initial && initial[0]) linenoiseEditInsert(&ed->edit, initial, strlen(initial));
    ed->hidden = false;
    ed->output_line_open = false;
    ed->prompt_below_output = false;
    ed->output_col = 0;
    ed->cpr_len = 0;
    ed->paste_open = false;
    ed->paste_start_pending = false;
    ed->paste_tail_len = 0;
    return 0;
}

/* Stop the live editor and restore stdin flags. */
static void editor_stop(agent_editor *ed) {
    if (!ed->active) return;
    /* ds4-agent treats linenoise as a live input widget, not as persistent
     * command scrollback.  Clear it before shutdown so submitting a line and
     * immediately reopening the editor does not leave the accepted
     * prompt+input duplicated above the fresh prompt. */
    if (!ed->hidden && (isatty(ed->edit.ifd) || getenv("LINENOISE_ASSUME_TTY")))
        editor_hide(ed);
    linenoiseEditStop(&ed->edit);
    if (ed->old_stdin_flags >= 0) fcntl(STDIN_FILENO, F_SETFL, ed->old_stdin_flags);
    free(ed->edit.buf);
    ed->input = NULL;
    ed->active = false;
    ed->hidden = false;
    ed->output_line_open = false;
    ed->prompt_below_output = false;
    ed->output_col = 0;
    ed->cpr_len = 0;
    ed->paste_open = false;
    ed->paste_start_pending = false;
    ed->paste_tail_len = 0;
}

/* Hide the live prompt before model output is written.  In scroll-region mode
 * the output cursor was saved before the prompt was drawn, so restoring it is
 * enough to append more model/tool bytes without touching the prompt rows. */
static void editor_hide(agent_editor *ed) {
    if (!ed->active || ed->hidden) return;
    if (ed->scroll_region) {
        editor_clear_prompt_region(ed);
        editor_restore_output_cursor(ed);
        ed->hidden = true;
        return;
    }
    linenoiseHide(&ed->edit);
    if (ed->prompt_below_output) {
        editor_move_to_output_cursor(ed);
        ed->prompt_below_output = false;
    }
    ed->hidden = true;
}

/* Restore the live prompt after output.  The primary path draws it in the
 * reserved bottom rows; the fallback path keeps the older one-row-below-output
 * trick for terminals where scroll regions are unavailable. */
static void editor_show(agent_editor *ed) {
    if (!ed->active || !ed->hidden) return;
    if (ed->scroll_region) {
        editor_save_output_cursor(ed);
        editor_move_to_prompt_row(ed);
        write_all(STDOUT_FILENO, "\x1b[0m", 4);
        linenoiseShow(&ed->edit);
        ed->hidden = false;
        return;
    }
    if (ed->output_line_open) {
        write_all(STDOUT_FILENO, "\r\n", 2);
        ed->prompt_below_output = true;
    } else {
        ed->prompt_below_output = false;
    }
    /* Model/tool output can leave SGR attributes active while it streams.
     * Redrawing linenoise always starts from normal attributes; tool rendering
     * re-emits its own color on the next streamed byte if it is still inside a
     * colored parameter. */
    write_all(STDOUT_FILENO, "\x1b[0m", 4);
    linenoiseShow(&ed->edit);
    ed->hidden = false;
}

static void editor_update_prompt(agent_editor *ed, const char *prompt) {
    snprintf(ed->prompt, sizeof(ed->prompt), "%s", prompt);
    ed->edit.prompt = ed->prompt;
    ed->edit.plen = strlen(ed->prompt);
}

static void editor_update_status(agent_editor *ed, const char *status) {
    snprintf(ed->status, sizeof(ed->status), "%s", status ? status : "");
    bool embedded_status = agent_footer_is_multiline(ed->status);
    const char *status_start = stdout_is_tty() && !embedded_status ?
        AGENT_STATUS_STYLE_START : "";
    const char *status_end = stdout_is_tty() && ed->status[0] ?
        AGENT_STATUS_STYLE_END : "";
    linenoiseEditSetStatus(&ed->edit, ed->status,
                           status_start, status_end);
}

static void editor_set_prompt_status(agent_editor *ed, const char *prompt,
                                     const char *status) {
    bool prompt_changed = strcmp(ed->prompt, prompt) != 0;
    bool status_changed = strcmp(ed->status, status ? status : "") != 0;
    if (!ed->active || (!prompt_changed && !status_changed)) return;
    if (ed->hidden) {
        if (prompt_changed) editor_update_prompt(ed, prompt);
        if (status_changed) editor_update_status(ed, status);
        return;
    }
    editor_hide(ed);
    if (prompt_changed) editor_update_prompt(ed, prompt);
    if (status_changed) editor_update_status(ed, status);
    editor_show(ed);
}

static void editor_redraw_visible_prompt(agent_editor *ed) {
    if (!ed->active || !ed->scroll_region) return;
    editor_clear_prompt_region(ed);
    editor_move_to_prompt_row(ed);
    write_all(STDOUT_FILENO, "\x1b[0m", 4);
    linenoiseShow(&ed->edit);
    ed->last_prompt_redraw_time = now_sec();
}

static bool editor_prompt_redraw_due(agent_editor *ed) {
    double now = now_sec();
    if (ed->last_prompt_redraw_time <= 0.0 ||
        now - ed->last_prompt_redraw_time >= AGENT_STATUS_REDRAW_INTERVAL_SEC)
    {
        return true;
    }
    return false;
}

static void editor_write_scroll_output_preserve_prompt(agent_editor *ed,
                                                       const char *text,
                                                       size_t len) {
    static const char sync_start[] = "\x1b[?2026h";
    static const char sync_end[] = "\x1b[?2026l";
    if (!len) return;

    write_all(STDOUT_FILENO, sync_start, sizeof(sync_start) - 1);
    editor_restore_output_cursor(ed);
    editor_write_terminal_text(text, len);
    editor_note_output(ed, text, len);
    editor_save_output_cursor(ed);
    write_all(STDOUT_FILENO, "\x1b[0m", 4);
    editor_move_to_prompt_cursor(ed);
    write_all(STDOUT_FILENO, sync_end, sizeof(sync_end) - 1);
    ed->output_at_scroll_boundary = true;
}

/* Serialize async model/tool output with linenoise.  This is the central
 * terminal contract.  In scroll-region mode the live prompt stays painted:
 * output is appended in the upper scroll area, then the cursor is returned to
 * linenoise's remembered prompt position.  The fallback path still hides and
 * redraws because it has no protected prompt rows. */
static void editor_write_async(agent_editor *ed, const char *text, size_t len,
                               const char *prompt, const char *status,
                               bool force_show) {
    if (ed->scroll_region && ed->active && !ed->hidden && len) {
        bool prompt_changed = strcmp(ed->prompt, prompt) != 0;
        bool status_changed = strcmp(ed->status, status ? status : "") != 0;

        editor_write_scroll_output_preserve_prompt(ed, text, len);
        if (prompt_changed) editor_update_prompt(ed, prompt);
        if (status_changed) editor_update_status(ed, status);
        if ((force_show || editor_prompt_redraw_due(ed)) &&
            (prompt_changed || status_changed))
        {
            editor_redraw_visible_prompt(ed);
        }
        return;
    }

    editor_hide(ed);
    if (len) {
        editor_write_terminal_text(text, len);
        if (ed->scroll_region) ed->output_at_scroll_boundary = true;
        if (!ed->scroll_region) {
            if (text[len - 1] == '\n' || text[len - 1] == '\r') {
                ed->output_col = 0;
                ed->output_line_open = false;
            } else {
                int col = 0;
                if (editor_query_cursor(ed, &col)) {
                    int cols = ed->edit.cols > 0 ? (int)ed->edit.cols : 80;
                    ed->output_col = col > 0 ? col - 1 : 0;
                    ed->output_line_open = true;
                    if (ed->output_col + 1 >= cols) {
                        write_all(STDOUT_FILENO, "\r\n", 2);
                        ed->output_col = 0;
                    }
                } else {
                    editor_note_output(ed, text, len);
                }
            }
        }
    }
    if (ed->active) {
        editor_update_prompt(ed, prompt);
        editor_update_status(ed, status);
        /* In scroll-region mode this saves the current output cursor and
         * redraws linenoise in the fixed prompt rows.  In fallback mode it may
         * put the prompt below an unfinished generated line. */
        if (force_show || len) editor_show(ed);
    }
}

/* Ctrl+C while idle is an edit-cancel key, not an exit key.  Clear the real
 * linenoise buffer so stale text cannot be submitted later, then leave a short
 * visible hint about the explicit EOF exit path. */
static void editor_cancel_input_with_hint(agent_editor *ed,
                                          const char *prompt,
                                          const char *status) {
    if (!ed->active) return;
    if (ed->hidden) editor_show(ed);
    linenoiseEditClear(&ed->edit);
    const char *msg = stdout_is_tty() ?
        "\x1b[1;33mpress Ctrl+D to exit\x1b[0m\n" :
        "press Ctrl+D to exit\n";
    editor_write_async(ed, msg, strlen(msg), prompt, status, true);
}

static void runtime_help(void) {
    puts("Commands:");
    puts("  /help        Show this help.");
    puts("  /save        Save the current session.");
    puts("  /compact     Compact the current session context now.");
    puts("  /list        List saved sessions.");
    puts("  /switch SHA  Load a saved session and show recent history.");
    puts("  /del SHA     Delete a saved session.");
    puts("  /strip SHA   Strip KV payload; /switch rebuilds it by prefill.");
    puts("  /history [N] Show N recent user turns from the current session.");
    puts("  /power N     Set GPU duty cycle percentage, 1..100.");
    puts("  /new         Start a fresh session from the system prompt.");
    puts("  /quit, /exit Exit.");
    puts("  Ctrl+C       Interrupt generation; clear edited text.");
    puts("  Enter        Queue text while the agent is busy.");
    puts("  Ctrl+X       Edit the first queued prompt.");
    puts("  ESC          Interrupt and send queued prompt immediately.");
    puts("  Ctrl+D       Exit from an empty prompt.");
}

static void agent_format_ctx_size(int ctx_size, char *buf, size_t len) {
    if (ctx_size >= 1000) {
        if (ctx_size % 1000 == 0) snprintf(buf, len, "%dk", ctx_size / 1000);
        else snprintf(buf, len, "%.1fk", (double)ctx_size / 1000.0);
    } else {
        snprintf(buf, len, "%d", ctx_size);
    }
}

static void agent_format_welcome_banner(const agent_config *cfg,
                                        char *buf, size_t len) {
    char ctx[32];
    agent_format_ctx_size(cfg->gen.ctx_size, ctx, sizeof(ctx));
    if (stdout_is_tty()) {
        snprintf(buf, len,
                 "\x1b[1;97mDwarf\x1b[1;94mStar\x1b[0m 🐋 Agent, context %s tokens\n\n",
                 ctx);
    } else {
        snprintf(buf, len, "DwarfStar Agent, context %s tokens\n\n", ctx);
    }
}

static void editor_write_welcome_banner(agent_editor *editor,
                                        const agent_config *cfg,
                                        const char *prompt,
                                        const char *statusline) {
    char banner[256];
    agent_format_welcome_banner(cfg, banner, sizeof(banner));
    editor_write_async(editor, banner, strlen(banner), prompt, statusline, true);
}

/* Initialize the worker, cache directory, sysprompt checkpoint path, trace file,
 * and model thread.  After this returns, all DS4 session mutation happens on
 * the worker thread. */
static int agent_worker_init(agent_worker *w, ds4_engine *engine, agent_config *cfg) {
    memset(w, 0, sizeof(*w));
    w->engine = engine;
    w->cfg = cfg;
    w->wake_fd[0] = -1;
    w->wake_fd[1] = -1;
    pthread_mutex_init(&w->mu, NULL);
    pthread_cond_init(&w->cond, NULL);
    w->status.state = AGENT_WORKER_IDLE;
    if (pipe(w->wake_fd) != 0) return -1;
    int old_flags;
    set_nonblock(w->wake_fd[0], true, &old_flags);
    set_nonblock(w->wake_fd[1], true, &old_flags);
    if (ds4_session_create(&w->session, engine, cfg->gen.ctx_size) != 0) {
        fprintf(stderr, "ds4-agent: session backend is required\n");
        return -1;
    }
    w->cache_dir = agent_default_cache_dir();
    if (!agent_mkdir_p(w->cache_dir)) {
        fprintf(stderr, "ds4-agent: failed to create %s: %s\n",
                w->cache_dir, strerror(errno));
        return -1;
    }
    ds4_web_config web_cfg = {
        .home_dir = getenv("HOME"),
        .port = 9333,
        .confirm = agent_web_confirm,
        .confirm_privdata = w,
        .log = agent_web_log,
        .log_privdata = w,
        .cancel = agent_web_cancel,
        .cancel_privdata = w,
    };
    w->web = ds4_web_create(&web_cfg);
    w->sysprompt_path = ds4_kvstore_path_join(w->cache_dir, "sysprompt.kv");
    if (cfg->gen.trace_path && cfg->gen.trace_path[0]) {
        w->trace = fopen(cfg->gen.trace_path, "ab");
        if (!w->trace) {
            fprintf(stderr, "ds4-agent: failed to open trace %s: %s\n",
                    cfg->gen.trace_path, strerror(errno));
            return -1;
        }
    }
    if (pthread_create(&w->thread, NULL, worker_main, w) != 0) return -1;
    return 0;
}

/* Shut down the worker and release owned resources, including any live bash
 * process groups. */
static void agent_worker_free(agent_worker *w) {
    worker_stop(w);
    if (w->thread) pthread_join(w->thread, NULL);
    agent_bash_jobs_free(w);
    ds4_web_free(w->web);
    ds4_session_free(w->session);
    ds4_tokens_free(&w->transcript);
    free(w->cache_dir);
    free(w->sysprompt_path);
    free(w->session_title);
    free(w->legacy_session_path_to_delete);
    free(w->queued_user_drain_text);
    if (w->wake_fd[0] >= 0) close(w->wake_fd[0]);
    if (w->wake_fd[1] >= 0) close(w->wake_fd[1]);
    if (w->trace) fclose(w->trace);
    free(w->cmd_text);
    free(w->out);
    pthread_cond_destroy(&w->cond);
    pthread_mutex_destroy(&w->mu);
}

typedef enum {
    AGENT_YES_NO_AUTO_NONE,
    AGENT_YES_NO_AUTO_NO,
    AGENT_YES_NO_AUTO_YES,
} agent_yes_no_auto;

typedef struct {
    int timeout_sec;
    agent_yes_no_auto timeout_answer;
} agent_yes_no_options;

static const char *agent_yes_no_auto_name(agent_yes_no_auto answer) {
    switch (answer) {
    case AGENT_YES_NO_AUTO_NO: return "no";
    case AGENT_YES_NO_AUTO_YES: return "yes";
    default: return "";
    }
}

/* Shared y/n prompt.  By default it blocks forever like the historical helper;
 * callers that cannot safely stall the agent can request an automatic answer
 * after timeout_sec seconds. */
static bool agent_prompt_yes_no_ex(const char *prompt,
                                   const agent_yes_no_options *opts,
                                   bool *timed_out) {
    char buf[32];
    int timeout_sec = opts ? opts->timeout_sec : 0;
    agent_yes_no_auto auto_answer = opts ?
        opts->timeout_answer : AGENT_YES_NO_AUTO_NONE;
    bool use_timeout = timeout_sec > 0 && auto_answer != AGENT_YES_NO_AUTO_NONE;
    double deadline = use_timeout ? now_sec() + timeout_sec : 0.0;

    if (timed_out) *timed_out = false;
    for (;;) {
        printf("%s", prompt);
        if (use_timeout) {
            int rem = (int)(deadline - now_sec() + 0.999);
            if (rem < 0) rem = 0;
            printf("[auto-%s in %ds] ", agent_yes_no_auto_name(auto_answer), rem);
        }
        fflush(stdout);
        if (use_timeout) {
            double rem_sec = deadline - now_sec();
            if (rem_sec <= 0.0) {
                if (timed_out) *timed_out = true;
                printf("\n");
                return auto_answer == AGENT_YES_NO_AUTO_YES;
            }
            struct pollfd pfd = {.fd = STDIN_FILENO, .events = POLLIN};
            int timeout_ms = (int)(rem_sec * 1000.0) + 1;
            int rc;
            do {
                rc = poll(&pfd, 1, timeout_ms);
            } while (rc < 0 && errno == EINTR);
            if (rc == 0) {
                if (timed_out) *timed_out = true;
                printf("\n");
                return auto_answer == AGENT_YES_NO_AUTO_YES;
            }
            if (rc < 0) return false;
        }
        /* stdin may be in non-blocking mode (set by editor_start).
         * Temporarily switch to blocking so fgets can wait for input. */
        int saved_flags = fcntl(STDIN_FILENO, F_GETFL, 0);
        if (saved_flags >= 0 && (saved_flags & O_NONBLOCK)) {
            fcntl(STDIN_FILENO, F_SETFL, saved_flags & ~O_NONBLOCK);
        }
        bool got_line = fgets(buf, sizeof(buf), stdin) != NULL;
        if (saved_flags >= 0 && (saved_flags & O_NONBLOCK)) {
            fcntl(STDIN_FILENO, F_SETFL, saved_flags);
        }
        if (!got_line) return false;
        char *p = buf;
        while (*p == ' ' || *p == '\t') p++;
        if (*p == 'y' || *p == 'Y') return true;
        if (*p == 'n' || *p == 'N') return false;
    }
}

static bool agent_prompt_yes_no(const char *prompt) {
    return agent_prompt_yes_no_ex(prompt, NULL, NULL);
}

/* Ask before discarding a dirty user session.  Fresh sessions that contain only
 * the system prompt are deliberately ignored. */
static bool agent_maybe_save_before_leaving_session(agent_worker *w) {
    if (!agent_worker_needs_save(w)) return true;
    if (!agent_prompt_yes_no("Save current session? (y/n) ")) return true;
    char err[160] = {0};
    if (agent_worker_save_session(w, err, sizeof(err))) return true;
    printf("save failed: %s\n", err);
    return agent_prompt_yes_no("Continue anyway? (y/n) ");
}

typedef enum {
    AGENT_EXIT_CANCEL,
    AGENT_EXIT_CLEAN,
    AGENT_EXIT_NOW,
} agent_exit_save_result;

/* Process exit is different from /new or /switch: once the terminal is already
 * restored, declining the save can terminate immediately and let the OS reclaim
 * model/Metal resources instead of waiting for orderly teardown. */
static agent_exit_save_result agent_maybe_save_before_exiting(agent_worker *w) {
    if (!agent_worker_needs_save(w)) return AGENT_EXIT_CLEAN;
    if (!agent_prompt_yes_no("Save current session? (y/n) ")) return AGENT_EXIT_NOW;
    char err[160] = {0};
    if (agent_worker_save_session(w, err, sizeof(err))) return AGENT_EXIT_CLEAN;
    printf("save failed: %s\n", err);
    return agent_prompt_yes_no("Continue anyway? (y/n) ") ?
        AGENT_EXIT_NOW : AGENT_EXIT_CANCEL;
}

/* ============================================================================
 * Interactive Runtime Loop
 * ============================================================================
 */

static void agent_noninteractive_marker(const char *msg) {
    write_all(STDERR_FILENO, msg, strlen(msg));
    write_all(STDERR_FILENO, "\n", 1);
}

static int agent_read_stdin_available(agent_input_buf *in, bool *eof) {
    char buf[4096];
    for (;;) {
        ssize_t n = read(STDIN_FILENO, buf, sizeof(buf));
        if (n > 0) {
            agent_input_buf_append(in, buf, (size_t)n);
            continue;
        }
        if (n == 0) {
            *eof = true;
            return 0;
        }
        if (errno == EINTR) continue;
        if (errno == EAGAIN || errno == EWOULDBLOCK) return 0;
        perror("ds4-agent: read stdin");
        return -1;
    }
}

/* Headless mode is intentionally just another front-end for the same worker.
 * With -p/--prompt it is a one-shot execution.  Without -p it becomes a small
 * stdin protocol: announce readiness on stderr, collect bytes until stdin has
 * been quiet for 200 ms, submit that buffer as one prompt, and keep reading so
 * later input can be queued while the model is still working. */
static int run_agent_non_interactive(ds4_engine *engine, agent_config *cfg) {
    agent_worker worker;
    if (agent_worker_init(&worker, engine, cfg) != 0) return 1;

    const bool one_shot = cfg->gen.prompt != NULL;
    bool one_shot_submitted = false;
    bool stdin_eof = false;
    bool waiting_announced = false;
    bool stdin_nonblock = false;
    int old_stdin_flags = 0;
    agent_input_buf input = {0};
    agent_prompt_queue queue = {0};
    double quiet_deadline = 0.0;
    int rc = 0;

    if (!one_shot) {
        if (set_nonblock(STDIN_FILENO, true, &old_stdin_flags) != 0) {
            perror("ds4-agent: nonblocking stdin");
            agent_worker_free(&worker);
            return 1;
        }
        stdin_nonblock = true;
    }

    while (true) {
        bool initialized = worker_is_initialized(&worker, NULL);
        bool idle = worker_is_idle(&worker);

        if (one_shot && !one_shot_submitted && initialized) {
            if (worker_submit(&worker, cfg->gen.prompt))
                one_shot_submitted = true;
            idle = false;
        }

        if (!one_shot && queue.len && idle) {
            char *queued = agent_prompt_queue_take_all(&queue);
            if (worker_submit(&worker, queued)) {
                idle = false;
            } else {
                agent_prompt_queue_push_front(&queue, queued);
                queued = NULL;
            }
            free(queued);
        }

        if (!one_shot && initialized && idle && !queue.len &&
            input.len == 0 && !stdin_eof && !waiting_announced)
        {
            agent_noninteractive_marker("+DWARFSTAR_WAITING");
            waiting_announced = true;
        }

        int timeout_ms = -1;
        if (!one_shot && input.len > 0) {
            double rem = quiet_deadline - now_sec();
            timeout_ms = rem <= 0.0 ? 0 : (int)(rem * 1000.0) + 1;
        }

        struct pollfd pfd[2];
        int nfds = 0;
        int wake_idx = nfds;
        pfd[nfds++] = (struct pollfd){.fd = worker.wake_fd[0], .events = POLLIN};
        int stdin_idx = -1;
        if (!one_shot && initialized && !stdin_eof) {
            stdin_idx = nfds;
            pfd[nfds++] = (struct pollfd){.fd = STDIN_FILENO, .events = POLLIN};
        }

        int prc = poll(pfd, (nfds_t)nfds, timeout_ms);
        if (prc < 0) {
            if (errno == EINTR) continue;
            perror("ds4-agent: poll");
            rc = 1;
            break;
        }
        if (pfd[wake_idx].revents & POLLIN) drain_wake_fd(worker.wake_fd[0]);
        if (stdin_idx >= 0 && (pfd[stdin_idx].revents & (POLLIN | POLLHUP))) {
            size_t old_len = input.len;
            if (agent_read_stdin_available(&input, &stdin_eof) != 0) {
                rc = 1;
                break;
            }
            if (input.len != old_len) {
                quiet_deadline = now_sec() + 0.200;
                waiting_announced = false;
            }
        }

        char *out = NULL;
        size_t out_len = 0;
        agent_status st = {0};
        worker_consume(&worker, &out, &out_len, &st);
        if (out && out_len) {
            write_all(STDOUT_FILENO, out, out_len);
            fflush(stdout);
        }
        free(out);

        if (worker_take_queued_user_drain_request(&worker)) {
            char *queued = agent_prompt_queue_take_all(&queue);
            worker_answer_queued_user_drain(&worker, queued);
        }

        if (st.state == AGENT_WORKER_ERROR) {
            fprintf(stderr, "ds4-agent: %s\n",
                    st.error[0] ? st.error : "worker error");
            rc = 1;
            break;
        }

        if (!one_shot && input.len > 0 &&
            (stdin_eof || now_sec() >= quiet_deadline))
        {
            char *prompt = agent_input_buf_take(&input);
            if (worker_is_idle(&worker) && queue.len == 0) {
                if (!worker_submit(&worker, prompt)) {
                    agent_prompt_queue_push(&queue, prompt);
                    agent_noninteractive_marker("+DWARFSTAR_QUEUED");
                }
            } else {
                agent_prompt_queue_push(&queue, prompt);
                agent_noninteractive_marker("+DWARFSTAR_QUEUED");
            }
            free(prompt);
            waiting_announced = false;
        }

        if (one_shot && one_shot_submitted && worker_is_idle(&worker)) break;
        if (!one_shot && stdin_eof && input.len == 0 &&
            queue.len == 0 && worker_is_idle(&worker))
            break;
    }

    /* Drain anything published between the final status transition and the
     * loop exit.  This keeps stdout complete without adding another protocol. */
    char *out = NULL;
    size_t out_len = 0;
    worker_consume(&worker, &out, &out_len, NULL);
    if (out && out_len) {
        write_all(STDOUT_FILENO, out, out_len);
        fflush(stdout);
    }
    free(out);

    if (stdin_nonblock) fcntl(STDIN_FILENO, F_SETFL, old_stdin_flags);
    agent_input_buf_free(&input);
    agent_prompt_queue_free(&queue);
    agent_worker_free(&worker);
    return rc;
}

/* Main UI loop.  poll() multiplexes stdin with the worker wake pipe; all
 * terminal writes go through editor_write_async() so linenoise, status footer,
 * model output, and tool output never race each other. */
static int run_agent(ds4_engine *engine, agent_config *cfg) {
    agent_worker worker;
    if (agent_worker_init(&worker, engine, cfg) != 0) return 1;

    char hist[PATH_MAX];
    const char *home = getenv("HOME");
    if (!home || !home[0]) home = ".";
    snprintf(hist, sizeof(hist), "%s/.ds4_agent_history", home);
    /* The agent uses ANSI scroll regions when possible: model/tool output
     * scrolls above the live linenoise prompt and status footer, so streaming
     * tokens do not require repainting the bottom rows.  Terminals without
     * scroll-region support fall back to the older prompt-below-output path. */
    linenoiseSetMultiLine(1);
    linenoiseHistorySetMaxLen(512);
    linenoiseHistoryLoad(hist);
    agent_completion_worker = &worker;
    linenoiseSetCompletionCallback(agent_switch_completion_callback);

    agent_status st;
    worker_get_status(&worker, &st);
    char prompt[160];
    char statusline[4096];
    build_prompt_text(&st, prompt, sizeof(prompt));
    build_footer_text(&st, NULL, 80, statusline, sizeof(statusline));

    agent_editor editor = {0};
    agent_prompt_queue queue = {0};
    if (editor_start(&editor, prompt, statusline, NULL) != 0) {
        fprintf(stderr, "ds4-agent: failed to start line editor\n");
        agent_worker_free(&worker);
        return 1;
    }
    editor_write_welcome_banner(&editor, cfg, prompt, statusline);

    char *initial_pending = cfg->gen.prompt && cfg->gen.prompt[0] ?
                            xstrdup(cfg->gen.prompt) : NULL;

    bool running = true;
    bool exit_save_handled = false;
    bool show_welcome_after_restart = false;
    bool force_status_redraw_after_restart = false;
    char *restore_line = NULL;
    while (running) {
        /* If a bash child process changed the terminal mode (e.g., from raw
         * to cooked), restore raw mode so linenoise continues to work. */
        if (worker_check_raw_mode_restore(&worker)) {
            linenoiseRestoreRawMode();
        }
        struct pollfd pfd[2] = {
            {.fd = STDIN_FILENO, .events = POLLIN},
            {.fd = worker.wake_fd[0], .events = POLLIN},
        };
        int timeout = (!editor.paste_open && !editor.paste_start_pending &&
                       linenoiseEditQueuedInput(&editor.edit) > 0) ? 0 : 100;
        int rc = poll(pfd, 2, timeout);
        if (rc < 0 && errno != EINTR) break;

        if (agent_sigint) {
            agent_sigint = 0;
            if (worker_is_idle(&worker)) {
                editor_cancel_input_with_hint(&editor, prompt, statusline);
            } else {
                worker_interrupt(&worker);
            }
        }

        if (rc > 0 && (pfd[0].revents & POLLIN)) editor_read_stdin(&editor);

        /* Linenoise runs the terminal in raw mode, so Ctrl+C normally arrives
         * as byte 3 instead of SIGINT.  Handle it before worker output is
         * drained and repainted; otherwise a busy decoding stream can leave the
         * interrupt waiting behind a large terminal-output backlog. */
        if (editor_take_queued_byte(&editor, 3)) { /* Ctrl+C */
            if (!worker_is_idle(&worker)) {
                worker_interrupt(&worker);
            } else {
                editor_cancel_input_with_hint(&editor, prompt, statusline);
            }
        }

        if (rc > 0 && (pfd[1].revents & POLLIN)) drain_wake_fd(worker.wake_fd[0]);

        char *out = NULL;
        size_t out_len = 0;
        worker_consume(&worker, &out, &out_len, &st);
        build_prompt_text(&st, prompt, sizeof(prompt));
        int footer_cols = editor.edit.cols > 0 ? (int)editor.edit.cols : 80;
        build_footer_text(&st, &queue, footer_cols, statusline, sizeof(statusline));
        if (out && out_len) {
            bool force_show = st.state == AGENT_WORKER_IDLE ||
                              st.state == AGENT_WORKER_ERROR ||
                              st.state == AGENT_WORKER_STOPPED;
            editor_write_async(&editor, out, out_len, prompt, statusline, force_show);
        } else {
            editor_set_prompt_status(&editor, prompt, statusline);
            if (editor.hidden && (st.state == AGENT_WORKER_IDLE ||
                                  st.state == AGENT_WORKER_ERROR ||
                                  st.state == AGENT_WORKER_STOPPED))
                editor_show(&editor);
        }
        if (st.state == AGENT_WORKER_ERROR && st.error[0]) {
            char msg[320];
            int n = snprintf(msg, sizeof(msg), "\nds4-agent: %s\n", st.error);
            editor_write_async(&editor, msg, n > 0 ? (size_t)n : 0,
                               prompt, statusline, true);
            pthread_mutex_lock(&worker.mu);
            worker.status.state = AGENT_WORKER_IDLE;
            worker.status.prefill_tps = 0.0;
            worker.status.greedy_sampling = false;
            worker.status.error[0] = '\0';
            pthread_mutex_unlock(&worker.mu);
        }
        free(out);

        if (worker_take_queued_user_drain_request(&worker)) {
            char *echo = agent_prompt_queue_take_all_echo(&queue);
            char *queued = agent_prompt_queue_take_all(&queue);
            if (echo) {
                build_footer_text(&st, &queue, footer_cols, statusline, sizeof(statusline));
                editor_write_async(&editor, echo, strlen(echo), prompt, statusline, true);
                free(echo);
            }
            worker_answer_queued_user_drain(&worker, queued);
            continue;
        }

        char web_approval_msg[256];
        if (worker_take_web_approval_request(&worker, web_approval_msg,
                                             sizeof(web_approval_msg)))
        {
            char *saved_input = NULL;
            if (editor.active && editor.edit.buf && editor.edit.len)
                saved_input = xstrndup(editor.edit.buf, editor.edit.len);
            editor_stop(&editor);
            editor_restore_terminal_layout(&editor);
            agent_yes_no_options approval_opts = {
                .timeout_sec = 30,
                .timeout_answer = AGENT_YES_NO_AUTO_NO,
            };
            bool approval_timed_out = false;
            bool allow = agent_prompt_yes_no_ex(web_approval_msg,
                                                &approval_opts,
                                                &approval_timed_out);
            worker_answer_web_approval(&worker, allow,
                approval_timed_out ? "Chrome browser start approval timed out" : NULL);
            worker_get_status(&worker, &st);
            build_prompt_text(&st, prompt, sizeof(prompt));
            int restart_cols = editor.edit.cols > 0 ? (int)editor.edit.cols : 80;
            build_footer_text(&st, &queue, restart_cols, statusline, sizeof(statusline));
            editor_start(&editor, prompt, statusline, saved_input);
            free(saved_input);
            continue;
        }

        if (initial_pending && worker_is_idle(&worker)) {
            if (worker_submit(&worker, initial_pending)) {
                free(initial_pending);
                initial_pending = NULL;
            }
        }

        if (!initial_pending && queue.len && worker_is_idle(&worker)) {
            char *echo = agent_prompt_queue_take_all_echo(&queue);
            char *queued = agent_prompt_queue_take_all(&queue);
            if (worker_submit(&worker, queued)) {
                linenoiseHistoryAdd(queued);
                linenoiseHistorySave(hist);
                build_footer_text(&st, &queue, footer_cols, statusline, sizeof(statusline));
                if (echo)
                    editor_write_async(&editor, echo, strlen(echo), prompt, statusline, true);
            } else {
                agent_prompt_queue_push_front(&queue, queued);
                queued = NULL;
            }
            free(echo);
            free(queued);
        }

        if (queue.len && editor_take_queued_byte(&editor, 24)) { /* Ctrl+X */
            char *queued = agent_prompt_queue_pop(&queue);
            editor_replace_input(&editor, queued);
            worker_get_status(&worker, &st);
            build_prompt_text(&st, prompt, sizeof(prompt));
            footer_cols = editor.edit.cols > 0 ? (int)editor.edit.cols : 80;
            build_footer_text(&st, &queue, footer_cols, statusline, sizeof(statusline));
            editor_set_prompt_status(&editor, prompt, statusline);
            free(queued);
        }
        if (queue.len && !worker_is_idle(&worker) && editor_take_bare_escape(&editor)) {
            worker_interrupt(&worker);
        }

        if (!editor.paste_open && !editor.paste_start_pending &&
            linenoiseEditQueuedInput(&editor.edit) > 0)
        {
            if (editor.hidden) {
                /* A user key while the model is in the middle of a partial
                 * output line means the prompt must become visible again. End
                 * the model line explicitly; otherwise linenoise would redraw
                 * on top of generated text. */
                editor_show(&editor);
            }
            errno = 0;
            char *line = linenoiseEditFeed(&editor.edit);
            if (line == linenoiseEditMore) {
                /* Still editing. */
            } else if (!line) {
                if (errno == EAGAIN) {
                    if (!worker_is_idle(&worker)) {
                        worker_interrupt(&worker);
                    } else {
                        editor_cancel_input_with_hint(&editor, prompt, statusline);
                    }
                } else {
                    running = false;
                }
            } else {
                char *cmd = line;
                while (*cmd == ' ' || *cmd == '\t' || *cmd == '\r' || *cmd == '\n') cmd++;
                char *end = cmd + strlen(cmd);
                while (end > cmd && (end[-1] == ' ' || end[-1] == '\t' || end[-1] == '\r' || end[-1] == '\n')) end--;
                *end = '\0';

                bool was_below_output = editor.prompt_below_output;
                bool had_output_line_open = editor.output_line_open;
                int saved_output_col = editor.output_col;
                editor_stop(&editor);
                bool busy = !worker_is_idle(&worker);
                if (!cmd[0]) {
                    /* Empty input: just reopen the editor. */
                } else if (!strcmp(cmd, "/help")) {
                    runtime_help();
                } else if (!strcmp(cmd, "/save")) {
                    if (busy) {
                        worker_request_save(&worker);
                        printf("save scheduled at next safe point\n");
                    } else {
                        char err[160] = {0};
                        if (!agent_worker_save_session(&worker, err, sizeof(err)))
                            printf("save failed: %s\n", err);
                    }
                } else if (!strcmp(cmd, "/compact")) {
                    worker_request_compact(&worker);
                    if (busy)
                        printf("compaction scheduled at next safe point\n");
                } else if (!strcmp(cmd, "/list")) {
                    agent_worker_list_sessions(&worker);
                } else if (!strncmp(cmd, "/power", 6) &&
                           (cmd[6] == '\0' || cmd[6] == ' ' || cmd[6] == '\t')) {
                    char *arg = cmd + 6;
                    while (*arg == ' ' || *arg == '\t') arg++;
                    if (!arg[0]) {
                        printf("usage: /power <1..100>\n");
                    } else {
                        int power = 0;
                        if (!parse_power_percent(arg, &power)) {
                            printf("usage: /power <1..100>\n");
                        } else {
                            worker_request_power(&worker, power);
                        }
                    }
                } else if (cmd[0] == '/' && !agent_slash_command_known(cmd)) {
                    ssize_t ignored = write(STDOUT_FILENO, "\a", 1);
                    (void)ignored;
                    restore_line = xstrdup(cmd);
                } else if (cmd[0] == '/' && busy) {
                    printf("command requires the model to be idle: %s\n", cmd);
                } else if (!strcmp(cmd, "/quit") || !strcmp(cmd, "/exit")) {
                    /* Stop the editor so raw mode and non-blocking stdin are
                     * disabled before we prompt the user.  Then restore the
                     * ANSI scroll region too; AGENT_EXIT_NOW exits directly. */
                    editor_stop(&editor);
                    editor_restore_terminal_layout(&editor);
                    agent_exit_save_result exit_save =
                        agent_maybe_save_before_exiting(&worker);
                    if (exit_save == AGENT_EXIT_NOW) {
                        exit(0);
                    } else if (exit_save == AGENT_EXIT_CLEAN) {
                        exit_save_handled = true;
                        running = false;
                    } else {
                        /* AGENT_EXIT_CANCEL: user declined to proceed after a
                         * save failure.  Reopen the editor and continue. */
                        editor_start(&editor, prompt, statusline, NULL);
                    }
                } else if (!strcmp(cmd, "/new")) {
                    editor_restore_terminal_layout(&editor);
                    if (agent_maybe_save_before_leaving_session(&worker)) {
                        char err[160] = {0};
                        if (!agent_worker_reset_to_sysprompt(&worker, err, sizeof(err))) {
                            printf("new session failed: %s\n", err);
                        } else {
                            show_welcome_after_restart = true;
                        }
                    }
                } else if (!strncmp(cmd, "/switch", 7) &&
                           (cmd[7] == '\0' || cmd[7] == ' ' || cmd[7] == '\t')) {
                    char *arg = cmd + 7;
                    while (*arg == ' ' || *arg == '\t') arg++;
                    if (!arg[0]) {
                        printf("usage: /switch <sha-prefix>\n");
                    } else {
                        editor_restore_terminal_layout(&editor);
                        if (agent_maybe_save_before_leaving_session(&worker)) {
                            char *sha = arg;
                            while (*arg && *arg != ' ' && *arg != '\t') arg++;
                            if (*arg) *arg = '\0';
                            char err[160] = {0};
                            if (!agent_worker_switch_session(&worker, sha,
                                                             AGENT_HISTORY_DEFAULT_TURNS,
                                                             err, sizeof(err)))
                                printf("switch failed: %s\n", err);
                            else
                                force_status_redraw_after_restart = true;
                        }
                    }
                } else if (!strncmp(cmd, "/del", 4) &&
                           (cmd[4] == '\0' || cmd[4] == ' ' || cmd[4] == '\t')) {
                    char *arg = cmd + 4;
                    while (*arg == ' ' || *arg == '\t') arg++;
                    if (!arg[0]) {
                        printf("usage: /del <sha-prefix>\n");
                    } else {
                        char *sha_arg = arg;
                        while (*arg && *arg != ' ' && *arg != '\t') arg++;
                        if (*arg) *arg = '\0';
                        char sha[41] = {0};
                        char err[160] = {0};
                        if (agent_worker_delete_session(&worker, sha_arg,
                                                        sha, err, sizeof(err)))
                            printf("deleted session %.8s\n", sha);
                        else
                            printf("delete failed: %s\n", err);
                    }
                } else if (!strncmp(cmd, "/strip", 6) &&
                           (cmd[6] == '\0' || cmd[6] == ' ' || cmd[6] == '\t')) {
                    char *arg = cmd + 6;
                    while (*arg == ' ' || *arg == '\t') arg++;
                    if (!arg[0]) {
                        printf("usage: /strip <sha-prefix>\n");
                    } else {
                        char *sha_arg = arg;
                        while (*arg && *arg != ' ' && *arg != '\t') arg++;
                        if (*arg) *arg = '\0';
                        char sha[41] = {0};
                        uint32_t tokens = 0;
                        char err[160] = {0};
                        if (agent_worker_strip_session(&worker, sha_arg,
                                                       sha, &tokens,
                                                       err, sizeof(err)))
                            printf("stripped session %.8s (%u tokens)\n",
                                   sha, tokens);
                        else
                            printf("strip failed: %s\n", err);
                    }
                } else if (!strncmp(cmd, "/history", 8) &&
                           (cmd[8] == '\0' || cmd[8] == ' ' || cmd[8] == '\t')) {
                    char *arg = cmd + 8;
                    while (*arg == ' ' || *arg == '\t') arg++;
                    int history_turns = arg[0] ?
                        agent_parse_int_default(arg, AGENT_HISTORY_DEFAULT_TURNS,
                                                1, AGENT_HISTORY_MAX_TURNS) :
                        AGENT_HISTORY_DEFAULT_TURNS;
                    char err[160] = {0};
                    if (!agent_worker_show_history(&worker, history_turns,
                                                   err, sizeof(err)))
                        printf("history failed: %s\n", err);
                } else if (busy) {
                    agent_prompt_queue_push(&queue, cmd);
                } else {
                    linenoiseHistoryAdd(cmd);
                    linenoiseHistorySave(hist);
                    if (worker_submit(&worker, cmd)) {
                        agent_echo_user_prompt(cmd);
                    } else {
                        restore_line = xstrdup(cmd);
                    }
                }
                linenoiseFree(line);

                if (running) {
                    worker_get_status(&worker, &st);
                    build_prompt_text(&st, prompt, sizeof(prompt));
                    int restart_cols = editor.edit.cols > 0 ? (int)editor.edit.cols : 80;
                    build_footer_text(&st, &queue, restart_cols, statusline, sizeof(statusline));
                    editor_start(&editor, prompt, statusline, restore_line);
                    if (!editor.scroll_region && was_below_output) {
                        editor.output_line_open = had_output_line_open;
                        editor.prompt_below_output = was_below_output;
                        editor.output_col = saved_output_col;
                    }
                    if (show_welcome_after_restart) {
                        editor_write_welcome_banner(&editor, cfg, prompt, statusline);
                        show_welcome_after_restart = false;
                    }
                    if (force_status_redraw_after_restart) {
                        editor_write_async(&editor, "", 0, prompt, statusline, true);
                        force_status_redraw_after_restart = false;
                    }
                    free(restore_line);
                    restore_line = NULL;
                }
            }
        }
    }

    free(initial_pending);
    free(restore_line);
    agent_prompt_queue_free(&queue);
    editor_stop(&editor);
    editor_restore_terminal_layout(&editor);
    linenoiseSetCompletionCallback(NULL);
    agent_completion_worker = NULL;
    if (!exit_save_handled) {
        agent_exit_save_result exit_save =
            agent_maybe_save_before_exiting(&worker);
        if (exit_save == AGENT_EXIT_NOW) exit(0);
    }
    agent_worker_free(&worker);
    return 0;
}

#ifndef DS4_AGENT_TEST_NO_MAIN
int main(int argc, char **argv) {
    agent_config cfg = parse_options(argc, argv);
    if (cfg.chdir_path && chdir(cfg.chdir_path) != 0) {
        fprintf(stderr, "ds4-agent: failed to chdir to %s: %s\n",
                cfg.chdir_path, strerror(errno));
        return 1;
    }
    cfg.engine.context_size = cfg.gen.ctx_size;
    cfg.engine.placement_ctx_hint = cfg.gen.ctx_size;
    if (cfg.gpu_vram_arg || cfg.gpu_devices_arg) {
        cfg.engine.backend = cfg.gpu_vram_arg &&
                             !strcmp(cfg.gpu_vram_arg, "0")
            ? DS4_BACKEND_CPU
            : DS4_BACKEND_CUDA;
    }
    ds4_engine *engine = NULL;
    if (cfg.gpu_vram_arg || cfg.gpu_devices_arg) {
        ds4_gpu_config gpu_cfg = {0};
        bool skip_cuda = false;
        char gpu_err[256];
        if (parse_gpu_vram_arg(cfg.gpu_vram_arg, cfg.gpu_devices_arg,
                               &gpu_cfg, &skip_cuda,
                               gpu_err, sizeof(gpu_err)) != 0) {
            fprintf(stderr, "ds4-agent: %s\n", gpu_err);
            return 2;
        }
        if (skip_cuda) {
            cfg.engine.backend = DS4_BACKEND_CPU;
            if (ds4_engine_open(&engine, &cfg.engine) != 0) return 1;
        } else {
            const bool was_auto =
                (cfg.gpu_vram_arg && !strcmp(cfg.gpu_vram_arg, "auto")) ||
                (!cfg.gpu_vram_arg && cfg.gpu_devices_arg);
            char layout[256];
            if (format_gpu_layout_line(&gpu_cfg, was_auto,
                                       layout, sizeof(layout)) > 0) {
                fprintf(stdout, "%s\n", layout);
                fflush(stdout);
            }
            cfg.engine.backend = DS4_BACKEND_CUDA;
            if (ds4_engine_create_with_gpu_config(
                    &engine, &cfg.engine, &gpu_cfg) != 0) return 1;
        }
    } else if (ds4_engine_open(&engine, &cfg.engine) != 0) {
        return 1;
    }
    agent_apply_model_sampling_defaults(engine, &cfg.gen);

    struct sigaction old_int;
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sigemptyset(&sa.sa_mask);
    sa.sa_handler = agent_sigint_handler;
    bool sigint_installed = !cfg.non_interactive &&
        sigaction(SIGINT, &sa, &old_int) == 0;

    int rc = cfg.non_interactive ?
        run_agent_non_interactive(engine, &cfg) :
        run_agent(engine, &cfg);

    if (sigint_installed) sigaction(SIGINT, &old_int, NULL);
    ds4_engine_close(engine);
    return rc;
}
#endif
