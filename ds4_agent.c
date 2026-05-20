#include "ds4.h"
#include "ds4_kvstore.h"
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

typedef enum {
    AGENT_EDIT_MODE_TAG,
    AGENT_EDIT_MODE_CRC,
} agent_edit_mode;

typedef struct {
    const char *prompt;
    const char *system;
    const char *trace_path;
    int n_predict;
    int ctx_size;
    float temperature;
    float top_p;
    float min_p;
    uint64_t seed;
    ds4_think_mode think_mode;
} agent_generation_options;

typedef struct {
    ds4_engine_options engine;
    agent_generation_options gen;
    agent_edit_mode edit_mode;
} agent_config;

typedef enum {
    AGENT_WORKER_IDLE,
    AGENT_WORKER_PREFILL,
    AGENT_WORKER_GENERATING,
    AGENT_WORKER_COMPACTING,
    AGENT_WORKER_ERROR,
    AGENT_WORKER_STOPPED,
} agent_worker_state;

typedef struct {
    agent_worker_state state;
    int prefill_done;
    int prefill_total;
    int generated;
    double gen_tps;
    int ctx_used;
    int ctx_size;
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
    int progress_base;
    char *cmd_text;
    agent_status status;
    char *out;
    size_t out_len;
    size_t out_cap;
    char more_path[PATH_MAX];
    int more_next_line;
    bool more_bare;
    bool more_valid;
    agent_bash_job *bash_jobs;
    int next_bash_job_id;
} agent_worker;

typedef enum {
    AGENT_MD_PENDING_NONE,
    AGENT_MD_PENDING_STAR,
    AGENT_MD_PENDING_BACKTICK,
} agent_markdown_pending;

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
    agent_markdown_pending md_pending;
    size_t md_pending_len;
    char pending[16];
    size_t pending_len;
    char utf8_pending[4];
    size_t utf8_pending_len;
    size_t utf8_pending_need;
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
    AGENT_DSML_SEARCH,
    AGENT_DSML_STRUCTURAL,
    AGENT_DSML_PARAM_VALUE,
    AGENT_DSML_DONE,
    AGENT_DSML_ERROR,
} agent_dsml_state;

typedef struct {
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
} agent_tool_visualizer;

typedef struct {
    agent_token_renderer *renderer;
    agent_dsml_parser *parser;
    agent_tool_visualizer viz;
    bool in_think;
    bool dsml_active;
    bool dsml_ignored;
    bool replay;
    char pending[16];
    size_t pending_len;
    char dsml_start_tail[64];
    size_t dsml_start_len;
    bool post_think_gap;
} agent_stream_renderer;

static volatile sig_atomic_t agent_sigint;
static agent_worker *agent_completion_worker;

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

static int parse_int(const char *s, const char *opt) {
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (s[0] == '\0' || *end != '\0' || v <= 0 || v > INT32_MAX) {
        fprintf(stderr, "ds4-agent: invalid value for %s: %s\n", opt, s);
        exit(2);
    }
    return (int)v;
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

static const char *agent_edit_mode_name(agent_edit_mode mode) {
    return mode == AGENT_EDIT_MODE_CRC ? "crc" : "tag";
}

static bool agent_parse_edit_mode(const char *s, agent_edit_mode *mode) {
    if (!s) return false;
    if (!strcmp(s, "tag")) {
        *mode = AGENT_EDIT_MODE_TAG;
        return true;
    }
    if (!strcmp(s, "crc")) {
        *mode = AGENT_EDIT_MODE_CRC;
        return true;
    }
    return false;
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

static void usage(FILE *fp) {
    fprintf(fp,
        "Usage: ds4-agent [options]\n"
        "\n"
        "This is an experimental native DS4 agent MVP. It keeps the terminal\n"
        "responsive with linenoise's multiplexed API while a model worker owns\n"
        "the live KV session.\n"
        "\n"
        "Options:\n"
        "  -m, --model FILE        GGUF model path. Default: ds4flash.gguf\n"
        "  --mtp FILE             Optional MTP support GGUF.\n"
        "  --mtp-draft N          Maximum MTP draft tokens. Default: 1\n"
        "  --mtp-margin F         MTP verifier margin. Default: 3\n"
        "  -c, --ctx N            Context size. Default: 100000\n"
        "  -n, --tokens N         Max generated tokens per turn. Default: 50000\n"
        "  -p, --prompt TEXT      Submit an initial prompt after startup.\n"
        "  -sys, --system TEXT    Extra system prompt. Empty disables extra text.\n"
        "  --trace FILE           Write prompt, token, and DSML debug trace.\n"
        "  --temp F               Sampling temperature. Default: 1\n"
        "  --top-p F              Nucleus sampling probability. Default: 1\n"
        "  --min-p F              Min-p sampling threshold. Default: 0.05\n"
        "  --seed N               Sampling seed.\n"
        "  --think                Use normal thinking mode. Default.\n"
        "  --think-max            Use Think Max when context is large enough.\n"
        "  --nothink              Disable thinking.\n"
        "  --backend NAME         metal, cuda, or cpu.\n"
        "  --metal, --cuda, --cpu Select backend explicitly.\n"
        "  -t, --threads N        CPU helper threads.\n"
        "  --quality              Prefer exact kernels where available.\n"
        "  --warm-weights         Touch mapped tensor pages before generation.\n"
        "  --dir-steering-file FILE\n"
        "  --dir-steering-ffn F\n"
        "  --dir-steering-attn F\n"
        "  -h, --help             Show this help.\n"
        "\n"
        "Commands:\n"
        "  /help                  Show runtime help.\n"
        "  /save                  Save the current agent session.\n"
        "  /set [NAME VALUE]      Show or change settings for new chats.\n"
        "  /list                  List saved sessions in ~/.ds4/kvcache.\n"
        "  /switch SHA            Load a saved session and show recent history.\n"
        "  /history [N]           Show N recent user turns from the current session.\n"
        "  /new                   Start a fresh session from the system prompt.\n"
        "  /quit, /exit           Exit.\n");
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
        .edit_mode = AGENT_EDIT_MODE_CRC,
    };

    bool steering_scale_set = false;
    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        if (!strcmp(arg, "-h") || !strcmp(arg, "--help")) {
            usage(stdout);
            exit(0);
        } else if (!strcmp(arg, "-p") || !strcmp(arg, "--prompt")) {
            c.gen.prompt = need_arg(&i, argc, argv, arg);
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
        } else if (!strcmp(arg, "--cpu")) {
            c.engine.backend = DS4_BACKEND_CPU;
        } else if (!strcmp(arg, "-t") || !strcmp(arg, "--threads")) {
            c.engine.n_threads = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--quality")) {
            c.engine.quality = true;
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
            usage(stderr);
            exit(2);
        }
    }

    if (c.engine.directional_steering_file && !steering_scale_set)
        c.engine.directional_steering_ffn = 1.0f;
    return c;
}

static void log_context_memory(ds4_backend backend, int ctx_size) {
    ds4_context_memory m = ds4_context_memory_estimate(backend, ctx_size);
    fprintf(stderr,
            "ds4-agent: context buffers %.2f MiB (ctx=%d, backend=%s, prefill_chunk=%u, raw_kv_rows=%u, compressed_kv_rows=%u)\n",
            (double)m.total_bytes / (1024.0 * 1024.0),
            ctx_size,
            ds4_backend_name(backend),
            m.prefill_cap,
            m.raw_cap,
            m.comp_cap);
}

static ds4_think_mode effective_think_mode(const agent_config *cfg) {
    return ds4_think_mode_for_context(cfg->gen.think_mode, cfg->gen.ctx_size);
}

/* ============================================================================
 * System Prompt Rendering And Worker Output Queues
 * ============================================================================
 */

static const char agent_tools_prompt_intro[] =
    "You are a coding agent running in a local workspace. When the user asks you to inspect, create, "
    "modify, build, test, or otherwise operate on local files, use tools instead of printing large file "
    "contents as the answer.\n\n"
    "## Tools\n\n"
    "You have access to native DSML tools. Invoke tools by writing exactly this shape:\n\n"
    "<｜DSML｜tool_calls>\n"
    "<｜DSML｜invoke name=\"$TOOL_NAME\">\n"
    "<｜DSML｜parameter name=\"$PARAMETER_NAME\" string=\"true|false\">$PARAMETER_VALUE</｜DSML｜parameter>\n"
    "</｜DSML｜invoke>\n"
    "</｜DSML｜tool_calls>\n\n"
    "Tool calls are not allowed inside <think></think>; finish thinking before emitting DSML.\n\n"
    "String parameters use raw text and string=\"true\". Numbers and booleans use JSON text and string=\"false\". "
    "For coding tasks, prefer a tool call over printing a complete source file inline. After tools run, summarize "
    "the result briefly.\n\n"
    "Read defaults to a bounded chunk: path alone returns the first 500 lines, not the whole file. "
    "If read says more lines are available, call the more tool with count=500 to read the next chunk. "
    "The read result also reports continue_offset=N, which is the next start_line if you need to jump manually. "
    "If the user explicitly asks you to read a complete file into context, call read with whole=true. "
    "A whole-file read may fail if the result would not fit the current context; then explain that and use chunks.\n\n";

static const char agent_tools_prompt_edit_tag[] =
    "## Edit mode: tag\n\n"
    "Read output is edit-friendly by default: each line is rendered as LINE:TAG> text, where TAG is a four-character "
    "checksum for the exact current line. Prefer line+tag edits after read/search. For a single tagged line, use "
    "line+tag+new only. Example: after `16:rA3_> count = 10;`, edit with line=16 tag=\"rA3_\" new=\"count = 11;\". "
    "For multiple tagged lines, copy one LINE:TAG entry per replaced line into `lines` and put the full replacement "
    "block in `new`; for example lines=\"16:rA3_\\n17:Kq9z\". Do not use start/end tags for bulk edits. "
    "When replacing many contiguous lines, prefer one bulk tagged edit instead of many single-line edits. Example:\n"
    "<｜DSML｜tool_calls>\n"
    "<｜DSML｜invoke name=\"edit\">\n"
    "<｜DSML｜parameter name=\"path\" string=\"true\">/tmp/example.c</｜DSML｜parameter>\n"
    "<｜DSML｜parameter name=\"lines\" string=\"true\">16:rA3_\n17:Kq9z\n18:PX0b</｜DSML｜parameter>\n"
    "<｜DSML｜parameter name=\"new\" string=\"true\">if (count > limit) {\n    return limit;\n}</｜DSML｜parameter>\n"
    "</｜DSML｜invoke>\n"
    "</｜DSML｜tool_calls>\n"
    "Use old/new only when you have no tag and the old text is unique. Never guess tags; if a tag edit fails, "
    "read or search again and retry with fresh tags. Use read raw=true only when you need undecorated file text.\n\n";

static const char agent_tools_prompt_edit_crc[] =
    "## Edit mode: crc\n\n"
    "Read and search output use plain line numbers and an eight-hex-digit whole-file CRC32 called file_tag. The file_tag covers the "
    "entire current file contents. When editing with line numbers, ranges, or old/new in this mode, include the "
    "file_tag you saw in the latest read/search result so the edit tool can reject stale edits if the file changed. "
    "Line output is rendered as `LINE text`, without per-line tags. Prefer line/range edits with file_tag for simple "
    "changes, or old/new plus file_tag when the old text is unique. A range may be passed as range=\"10:20\" or as "
    "start_line=10 end_line=20. Example:\n"
    "<｜DSML｜tool_calls>\n"
    "<｜DSML｜invoke name=\"edit\">\n"
    "<｜DSML｜parameter name=\"path\" string=\"true\">/tmp/example.c</｜DSML｜parameter>\n"
    "<｜DSML｜parameter name=\"file_tag\" string=\"true\">9e3a41bf</｜DSML｜parameter>\n"
    "<｜DSML｜parameter name=\"range\" string=\"true\">16:18</｜DSML｜parameter>\n"
    "<｜DSML｜parameter name=\"new\" string=\"true\">if (count > limit) {\n    return limit;\n}</｜DSML｜parameter>\n"
    "</｜DSML｜invoke>\n"
    "</｜DSML｜tool_calls>\n"
    "Use read raw=true only when you need undecorated file text.\n\n";

static const char agent_tools_prompt_after_edit[] =
    "For long-running bash commands, pass refresh_sec to receive updates. If a bash job is still running, use "
    "bash_wait, bash_status, or bash_stop with the returned job number.\n\n"
    "### Available Tool Schemas\n\n"
    "{\"type\":\"function\",\"function\":{\"name\":\"bash\",\"description\":\"Run a shell command.\","
    "\"parameters\":{\"type\":\"object\",\"properties\":{\"command\":{\"type\":\"string\"},"
    "\"timeout_sec\":{\"type\":\"number\"},\"refresh_sec\":{\"type\":\"number\"}},"
    "\"required\":[\"command\"]}}}\n"
    "{\"type\":\"function\",\"function\":{\"name\":\"bash_status\",\"description\":\"Report current status and new output for a bash job.\","
    "\"parameters\":{\"type\":\"object\",\"properties\":{\"job\":{\"type\":\"number\"},"
    "\"pid\":{\"type\":\"number\"}},\"required\":[\"job\"]}}}\n"
    "{\"type\":\"function\",\"function\":{\"name\":\"bash_wait\",\"description\":\"Wait for a bash job to finish or produce another refresh.\","
    "\"parameters\":{\"type\":\"object\",\"properties\":{\"job\":{\"type\":\"number\"},"
    "\"pid\":{\"type\":\"number\"},\"refresh_sec\":{\"type\":\"number\"}},\"required\":[\"job\"]}}}\n"
    "{\"type\":\"function\",\"function\":{\"name\":\"bash_stop\",\"description\":\"Terminate a running bash job and report its final output.\","
    "\"parameters\":{\"type\":\"object\",\"properties\":{\"job\":{\"type\":\"number\"},"
    "\"pid\":{\"type\":\"number\"}},\"required\":[\"job\"]}}}\n"
    "{\"type\":\"function\",\"function\":{\"name\":\"read\",\"description\":\"Read a text file or a range of lines.\","
    "\"parameters\":{\"type\":\"object\",\"properties\":{\"path\":{\"type\":\"string\"},"
    "\"start_line\":{\"type\":\"number\"},\"max_lines\":{\"type\":\"number\"},"
    "\"whole\":{\"type\":\"boolean\"},\"raw\":{\"type\":\"boolean\"}},\"required\":[\"path\"]}}}\n"
    "{\"type\":\"function\",\"function\":{\"name\":\"more\",\"description\":\"Continue the previous read-like output.\","
    "\"parameters\":{\"type\":\"object\",\"properties\":{\"count\":{\"type\":\"number\"}}}}}\n"
    "{\"type\":\"function\",\"function\":{\"name\":\"write\",\"description\":\"Create or overwrite a text file.\","
    "\"parameters\":{\"type\":\"object\",\"properties\":{\"path\":{\"type\":\"string\"},"
    "\"content\":{\"type\":\"string\"}},\"required\":[\"path\",\"content\"]}}}\n"
    "{\"type\":\"function\",\"function\":{\"name\":\"edit\",\"description\":\"Edit a file using the active edit mode or old/new text.\","
    "\"parameters\":{\"type\":\"object\",\"properties\":{\"path\":{\"type\":\"string\"},"
    "\"line\":{\"type\":\"number\"},\"start_line\":{\"type\":\"number\"},\"end_line\":{\"type\":\"number\"},"
    "\"range\":{\"type\":\"string\"},\"tag\":{\"type\":\"string\"},\"lines\":{\"type\":\"string\"},"
    "\"file_tag\":{\"type\":\"string\"},\"old\":{\"type\":\"string\"},\"new\":{\"type\":\"string\"}},"
    "\"required\":[\"path\"]}}}\n"
    "{\"type\":\"function\",\"function\":{\"name\":\"search\",\"description\":\"Search files and return compact edit-friendly matches.\","
    "\"parameters\":{\"type\":\"object\",\"properties\":{\"query\":{\"type\":\"string\"},"
    "\"path\":{\"type\":\"string\"},\"mode\":{\"type\":\"string\"},\"glob\":{\"type\":\"string\"},"
    "\"context\":{\"type\":\"number\"},\"max_results\":{\"type\":\"number\"},"
    "\"case_sensitive\":{\"type\":\"boolean\"}},\"required\":[\"query\"]}}}\n"
    "{\"type\":\"function\",\"function\":{\"name\":\"list\",\"description\":\"List one directory compactly.\","
    "\"parameters\":{\"type\":\"object\",\"properties\":{\"path\":{\"type\":\"string\"}},\"required\":[\"path\"]}}}\n";

static char *agent_build_tools_prompt(agent_edit_mode mode) {
    const char *edit = mode == AGENT_EDIT_MODE_CRC ?
                       agent_tools_prompt_edit_crc : agent_tools_prompt_edit_tag;
    size_t a = strlen(agent_tools_prompt_intro);
    size_t b = strlen(edit);
    size_t c = strlen(agent_tools_prompt_after_edit);
    char *out = xmalloc(a + b + c + 1);
    memcpy(out, agent_tools_prompt_intro, a);
    memcpy(out + a, edit, b);
    memcpy(out + a + b, agent_tools_prompt_after_edit, c + 1);
    return out;
}

static agent_edit_mode agent_edit_mode_from_rendered_text(const char *text) {
    if (text && strstr(text, "## Edit mode: crc")) return AGENT_EDIT_MODE_CRC;
    return AGENT_EDIT_MODE_TAG;
}

static void agent_append_system_prompt(ds4_engine *engine, ds4_tokens *tokens,
                                       const char *extra, agent_edit_mode edit_mode) {
    /* The built-in tool prompt is trusted DS4 control text.  Tokenize it like a
     * rendered chat prompt so the literal ｜DSML｜ markers in the examples become
     * the model's dedicated DSML token.  Do not apply that tokenizer to user
     * supplied -sys text: arbitrary user text containing <｜User｜>, <think>, or
     * ｜DSML｜ must remain plain content, not control tokens. */
    char *tools_prompt = agent_build_tools_prompt(edit_mode);
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
    agent_wake_locked(w);
    pthread_mutex_unlock(&w->mu);
}

static void agent_set_error(agent_worker *w, const char *msg) {
    pthread_mutex_lock(&w->mu);
    w->status.state = AGENT_WORKER_ERROR;
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
    free(p->raw);
    agent_tool_call_free(&p->current);
    free(p->param_name);
    agent_tool_calls_free(&p->calls);
    memset(p, 0, sizeof(*p));
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

/* Parse as much of the accumulated DSML buffer as possible.  The parser can be
 * called after every streamed byte: incomplete input leaves state unchanged
 * until enough bytes arrive, while malformed completed input switches to
 * AGENT_DSML_ERROR so the model gets a retryable tool error. */
static void agent_dsml_parse(agent_dsml_parser *p) {
    static const char invoke_start[] = "<｜DSML｜invoke";
    static const char param_start[] = "<｜DSML｜parameter";

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

        if (!strncmp(tag, invoke_start, strlen(invoke_start))) {
            agent_tool_call_free(&p->current);
            p->current.name = agent_parse_attr(tag, "name");
            if (!p->current.name) {
                free(tag);
                agent_dsml_set_error(p, "tool invoke without name");
                return;
            }
            p->parse_pos += tag_len;
        } else if (!strncmp(tag, param_start, strlen(param_start))) {
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
    static const char start[] = "<｜DSML｜tool_calls>";
    p->state = AGENT_DSML_STRUCTURAL;
    p->search_len = 0;
    agent_dsml_raw_append(p, start, strlen(start));
    p->parse_pos = strlen(start);
}

static void agent_dsml_feed(agent_dsml_parser *p, const char *s, size_t n) {
    static const char start[] = "<｜DSML｜tool_calls>";
    const size_t start_len = sizeof(start) - 1;
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

static void renderer_write(agent_token_renderer *r, const char *s, size_t n) {
    agent_publish(r->worker, s, n);
}

static void renderer_set_grey(agent_token_renderer *r) {
    if (r->use_color) renderer_write(r, "\x1b[90m", 5);
}

static void renderer_reset_color(agent_token_renderer *r) {
    if (r->use_color) renderer_write(r, "\x1b[0m", 4);
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

static void renderer_write_complete_char_raw(agent_token_renderer *r, const char *s, size_t n) {
    bool styled = r->use_color && renderer_has_text_attrs(r);
    if (styled) renderer_set_text_attrs(r);
    renderer_write(r, s, n);
    if (styled) renderer_reset_color(r);
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
    for (size_t i = 0; i < count; i++) renderer_write_char_raw(r, c);
}

static void renderer_markdown_commit_backticks(agent_token_renderer *r) {
    size_t count = r->md_pending_len;
    renderer_markdown_clear_pending(r);
    if (count >= 3) {
        r->md_code_block = !r->md_code_block;
        r->md_inline_code = false;
        /* Hide the optional language marker in ```c / ```python fences and
         * the rest of the closing-fence line.  The code itself starts at the
         * next real line, already colored by md_code_block. */
        r->md_fence_info = true;
        return;
    }
    if (r->md_code_block) {
        for (size_t i = 0; i < count; i++) renderer_write_char_raw(r, '`');
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
            if (!r->last_output_newline) renderer_write_char_raw(r, '\n');
            r->md_fence_info = false;
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

    if (c == '`') {
        r->md_pending = AGENT_MD_PENDING_BACKTICK;
        r->md_pending_len = 1;
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
    r->md_bold = false;
    r->md_italic = false;
    r->md_inline_code = false;
    r->md_code_block = false;
    r->md_fence_info = false;
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
    if (r->use_color) renderer_write(r, seq, strlen(seq));
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
    if (!strcmp(tool, "edit") && !strcmp(param, "lines"))
        return AGENT_TOOL_PARAM_OFFSET;
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
    } else {
        /* The raw DSML start marker may arrive after ordinary text on the
         * current row.  Clear that row and repaint it as a semantic tool call
         * line instead of letting the XML-ish control syntax leak into the live
         * user interface.  Replay mode must not emit this cursor control,
         * because it may erase already-rendered backlog. */
        agent_tool_viz_puts(sr, "\r\x1b[2K");
    }
    agent_tool_viz_puts(sr, "🛠️ ");
}

static const char *agent_tool_viz_prefix(const char *name) {
    if (!strcmp(name, "bash")) return "$ ";
    if (!strcmp(name, "read")) return "read ";
    if (!strcmp(name, "write")) return "write ";
    if (!strcmp(name, "edit")) return "edit ";
    if (!strcmp(name, "search")) return "search ";
    return NULL;
}

static void agent_tool_viz_tool(agent_stream_renderer *sr, const char *name) {
    agent_tool_visualizer *v = &sr->viz;
    if (v->tool_announced && !strcmp(v->tool_name, name)) return;
    if (v->tool_announced && !v->last_output_newline) agent_tool_viz_puts(sr, "\n");
    snprintf(v->tool_name, sizeof(v->tool_name), "%s", name ? name : "tool");
    v->tool_announced = true;
    v->read_style = !strcmp(v->tool_name, "read");
    if (v->read_style) {
        if (!v->last_output_newline) agent_tool_viz_puts(sr, "\n");
        renderer_color(sr->renderer, "\x1b[1;37m");
        agent_tool_viz_puts(sr, "Reading ");
        renderer_color(sr->renderer, "\x1b[32m");
        v->read_prefix_rendered = true;
        return;
    }
    v->at_line_start = false;
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
        if (!v->last_output_newline) agent_tool_viz_puts(sr, "\n");
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
        agent_tool_viz_puts(sr, " ");
        agent_tool_viz_puts(sr, v->read_start[0] ? v->read_start : "1");
        agent_tool_viz_puts(sr, ":");
        agent_tool_viz_puts(sr, v->read_max[0] ? v->read_max : "500");
    }
    renderer_color(sr->renderer, "\x1b[1;37m");
    agent_tool_viz_puts(sr, "...");
    renderer_color(sr->renderer, "\x1b[0m");
    agent_tool_viz_puts(sr, "\n");
    v->read_line_rendered = true;
}

static void agent_tool_viz_diff_prefix(agent_stream_renderer *sr) {
    agent_tool_visualizer *v = &sr->viz;
    if (!v->at_line_start) return;
    renderer_color(sr->renderer, v->param_kind == AGENT_TOOL_PARAM_DIFF_OLD ?
                                "\x1b[31m" : "\x1b[32m");
    agent_tool_viz_puts(sr, v->param_kind == AGENT_TOOL_PARAM_DIFF_OLD ? "- " : "+ ");
    v->at_line_start = false;
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
        agent_tool_viz_diff_prefix(sr);
        return;
    }

    if (v->param_kind == AGENT_TOOL_PARAM_CONTENT) {
        if (!v->last_output_newline) agent_tool_viz_puts(sr, "\n");
        renderer_color(sr->renderer, "\x1b[1;37m");
        agent_tool_viz_puts(sr, v->param_name);
        agent_tool_viz_puts(sr, ":\n");
        renderer_color(sr->renderer, "\x1b[34m");
        v->at_line_start = true;
        return;
    }

    if (v->param_kind != AGENT_TOOL_PARAM_BASH_COMMAND) {
        if (!v->at_line_start) agent_tool_viz_puts(sr, " ");
        renderer_color(sr->renderer, "\x1b[1;37m");
        agent_tool_viz_puts(sr, v->param_name);
        agent_tool_viz_puts(sr, "=");
    }
    renderer_color(sr->renderer, agent_tool_param_color(v->param_kind));
}

static void agent_tool_viz_param_end(agent_stream_renderer *sr) {
    agent_tool_visualizer *v = &sr->viz;
    v->param_end_len = 0;
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
    if (v->param_kind == AGENT_TOOL_PARAM_DIFF_OLD ||
        v->param_kind == AGENT_TOOL_PARAM_DIFF_NEW)
    {
        agent_tool_viz_diff_prefix(sr);
        agent_tool_viz_write(sr, &c, 1);
        if (c == '\n') v->at_line_start = true;
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

static bool agent_tool_viz_param_end_tail(const char *tail, size_t len, bool *complete) {
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

/* Stream one DSML parameter byte into the visualizer.  The visualizer must not
 * wait for the whole parameter: large write/edit contents should show progress
 * as the model emits them, while still detecting the closing parameter tag. */
static void agent_tool_viz_param_value_byte(agent_stream_renderer *sr, char c) {
    agent_tool_visualizer *v = &sr->viz;

    if (v->param_end_len || c == '<') {
        if (v->param_end_len < sizeof(v->param_end_tail))
            v->param_end_tail[v->param_end_len++] = c;
        bool complete = false;
        if (agent_tool_viz_param_end_tail(v->param_end_tail, v->param_end_len, &complete)) {
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

static void agent_stream_feed_dsml_byte(agent_stream_renderer *sr, char c) {
    bool was_param = !sr->dsml_ignored && sr->viz.param_active;
    agent_dsml_feed(sr->parser, &c, 1);
    if (!sr->dsml_ignored) {
        agent_stream_tool_events(sr);
        if (was_param) agent_tool_viz_param_value_byte(sr, c);
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
    sr->dsml_start_len = 0;
    sr->post_think_gap = false;
    agent_trace(sr->renderer->worker, "dsml start detected%s",
                ignored ? " inside thinking" : "");
    agent_dsml_start(sr->parser);
    if (!ignored) {
        agent_tool_viz_start(sr);
        agent_stream_tool_events(sr);
    }
}

static void agent_stream_flush_start_tail(agent_stream_renderer *sr) {
    if (!sr->dsml_start_len) return;
    sr->post_think_gap = false;
    for (size_t i = 0; i < sr->dsml_start_len; i++)
        renderer_write_char(sr->renderer, sr->dsml_start_tail[i]);
    sr->dsml_start_len = 0;
}

static bool agent_stream_dsml_start_match(const char *tail, size_t len,
                                          bool *complete) {
    static const char canonical[] = "<｜DSML｜tool_calls>";
    static const char missing_bar[] = "<DSML｜tool_calls>";
    const char *forms[] = {canonical, missing_bar};
    *complete = false;
    for (size_t i = 0; i < sizeof(forms)/sizeof(forms[0]); i++) {
        size_t form_len = strlen(forms[i]);
        if (len <= form_len && memcmp(forms[i], tail, len) == 0) {
            *complete = len == form_len;
            return true;
        }
    }
    return false;
}

/* Route ordinary assistant bytes either to normal markdown rendering or into
 * the DSML detector.  The detector must hold short prefixes because the model
 * can split "<｜DSML｜tool_calls>" across arbitrary tokens. */
static void agent_stream_normal_byte(agent_stream_renderer *sr, char c) {
    static const char start[] = "<｜DSML｜tool_calls>";

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
        bool complete = false;
        if (agent_stream_dsml_start_match(sr->dsml_start_tail, sr->dsml_start_len,
                                          &complete))
        {
            if (complete) {
                /* Accept the common missing-leading-bar typo
                 * "<DSML｜tool_calls>" here, but seed the parser with the
                 * canonical marker so the rest of the DSML parser stays
                 * strict and simple. */
                agent_stream_start_dsml(sr, sr->in_think);
            }
            return;
        }
        if (sr->dsml_start_len > 1 &&
            sr->dsml_start_tail[sr->dsml_start_len - 1] == start[0])
        {
            sr->post_think_gap = false;
            size_t flush = sr->dsml_start_len - 1;
            for (size_t i = 0; i < flush; i++)
                renderer_write_char(sr->renderer, sr->dsml_start_tail[i]);
            sr->dsml_start_tail[0] = start[0];
            sr->dsml_start_len = 1;
            return;
        }
        agent_stream_flush_start_tail(sr);
        return;
    }

    sr->post_think_gap = false;
    renderer_write_char(sr->renderer, c);
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
        if (sr->dsml_active) {
            if (sr->dsml_ignored) {
                agent_stream_finish_ignored_dsml(
                    sr, "unfinished tool call inside <think></think>");
            } else {
                agent_tool_viz_finish(sr, "[tool call interrupted]\n");
                sr->dsml_active = false;
            }
        }
    }
}

/* ============================================================================
 * Worker Progress And Generic Buffers
 * ============================================================================
 */

static void worker_progress_cb(void *ud, const char *event, int current, int total) {
    (void)total;
    agent_worker *w = ud;
    if (!w || !event || strcmp(event, "prefill_chunk")) return;
    pthread_mutex_lock(&w->mu);
    int done = current - w->progress_base;
    if (done < 0) done = 0;
    if (done > w->status.prefill_total) done = w->status.prefill_total;
    w->status.prefill_done = done;
    agent_wake_locked(w);
    pthread_mutex_unlock(&w->mu);
}

static bool worker_should_interrupt(agent_worker *w) {
    pthread_mutex_lock(&w->mu);
    bool interrupt = w->interrupt || w->stop;
    pthread_mutex_unlock(&w->mu);
    return interrupt;
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

/* ============================================================================
 * Agent KV Store And Session Persistence
 * ============================================================================
 */

/* Agent sessions deliberately use a different policy from ds4-server:
 *
 * - sysprompt.kv is a fixed bootstrap checkpoint for the current tool/system
 *   prompt.  Because its name is fixed, the current rendered text is compared
 *   with the text stored in the file before loading.  A mismatch simply rebuilds
 *   and overwrites the file.
 * - conversation sessions are explicit saves only.  Their file name is still
 *   SHA1(rendered transcript).kv, which keeps the existing KV file format and
 *   lets /list and /switch identify sessions by the first hex characters.
 *
 * The DS4 payload stores the exact token sequence and graph state.  The rendered
 * text is only the stable external identity of that state. */
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

/* Load a KV file and optionally verify either its SHA identity or exact
 * rendered text.  sysprompt.kv uses exact text because the file name is fixed;
 * saved sessions use the SHA because their file name already carries it. */
static bool agent_kv_load_path(agent_worker *w, const char *path,
                               const char *expected_sha,
                               const char *expected_text,
                               size_t expected_text_len,
                               ds4_tokens *loaded_tokens,
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
    if (ok && hdr.quant_bits != (uint8_t)ds4_engine_routed_quant_bits(w->engine)) {
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
        ds4_kvstore_sha1_bytes_hex(text, text_bytes, actual_sha);
        if (strcmp(actual_sha, expected_sha)) {
            snprintf(err, err_len, "cached text hash does not match file name");
            ok = false;
        }
    }

    char load_err[160] = {0};
    if (ok &&
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
        if (!live || live->len != (int)hdr.tokens) {
            snprintf(err, err_len, "KV payload token count mismatch");
            ds4_session_invalidate(w->session);
            ok = false;
        } else if (loaded_tokens) {
            ds4_tokens_free(loaded_tokens);
            ds4_tokens_copy(loaded_tokens, live);
        }
    }
    free(text);
    return ok;
}

/* Save the current live KV under the rendered transcript identity.  The caller
 * decides the policy: fixed sysprompt path or SHA-named session path. */
static bool agent_kv_save_path(agent_worker *w, const char *path,
                               const ds4_tokens *tokens,
                               const char *reason,
                               char sha_out[41],
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
    char sha[41];
    ds4_kvstore_sha1_bytes_hex(text, text_len, sha);
    if (sha_out) memcpy(sha_out, sha, sizeof(sha));

    uint64_t payload_bytes = ds4_session_payload_bytes(w->session);
    if (payload_bytes == 0) {
        snprintf(err, err_len, "session has no valid KV payload");
        free(text);
        return false;
    }

    agent_buf tmpl = {0};
    agent_buf_puts(&tmpl, path);
    agent_buf_puts(&tmpl, ".tmp.XXXXXX");
    char *tmp = agent_buf_take(&tmpl);
    int fd = mkstemp(tmp);
    if (fd < 0) {
        snprintf(err, err_len, "%s", strerror(errno));
        free(tmp);
        free(text);
        return false;
    }

    FILE *fp = fdopen(fd, "wb");
    if (!fp) {
        snprintf(err, err_len, "%s", strerror(errno));
        close(fd);
        unlink(tmp);
        free(tmp);
        free(text);
        return false;
    }

    const uint64_t now = (uint64_t)time(NULL);
    uint8_t h[DS4_KVSTORE_FIXED_HEADER];
    ds4_kvstore_fill_header(h, (uint8_t)quant_bits,
                            ds4_kvstore_reason_code(reason),
                            0, (uint32_t)tokens->len, 0,
                            (uint32_t)ds4_session_ctx(w->session),
                            now, now, payload_bytes);
    uint8_t tb[4];
    ds4_kvstore_le_put32(tb, (uint32_t)text_len);

    char save_err[160] = {0};
    errno = 0;
    bool ok = fwrite(h, 1, sizeof(h), fp) == sizeof(h) &&
              fwrite(tb, 1, sizeof(tb), fp) == sizeof(tb) &&
              fwrite(text, 1, text_len, fp) == text_len &&
              ds4_session_save_payload(w->session, fp,
                                       save_err, sizeof(save_err)) == 0 &&
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

    free(tmp);
    free(text);
    return ok;
}

/* Drop older session files that are an exact prefix of the session being saved.
 * Those files are just previous checkpoints of the same conversation and would
 * clutter /list without providing a distinct resumable branch. */
static void agent_kv_delete_prefix_sessions(agent_worker *w,
                                            const char *current_sha,
                                            const char *current_text,
                                            size_t current_text_len) {
    DIR *d = opendir(w->cache_dir);
    if (!d) return;
    struct dirent *de;
    while ((de = readdir(d)) != NULL) {
        char sha[41];
        if (!ds4_kvstore_sha_hex_name(de->d_name, sha)) continue;
        if (current_sha && !strcmp(sha, current_sha)) continue;

        char *path = ds4_kvstore_path_join(w->cache_dir, de->d_name);
        FILE *fp = fopen(path, "rb");
        if (!fp) {
            free(path);
            continue;
        }
        ds4_kvstore_entry hdr = {0};
        uint32_t text_bytes = 0;
        char *old_text = NULL;
        bool ok = ds4_kvstore_read_header(fp, &hdr, &text_bytes) &&
                  agent_kv_read_text(fp, text_bytes, &old_text, NULL, 0);
        fclose(fp);
        if (ok &&
            text_bytes < current_text_len &&
            memcmp(current_text, old_text, text_bytes) == 0)
        {
            unlink(path);
        }
        free(old_text);
        free(path);
    }
    closedir(d);
}

static void agent_worker_build_system_tokens(agent_worker *w, ds4_tokens *out) {
    ds4_chat_begin(w->engine, out);
    if (w->cfg->gen.think_mode == DS4_THINK_MAX &&
        effective_think_mode(w->cfg) == DS4_THINK_MAX)
        ds4_chat_append_max_effort_prefix(w->engine, out);
    agent_append_system_prompt(w->engine, out, w->cfg->gen.system,
                               w->cfg->edit_mode);
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
        w->status.state = AGENT_WORKER_PREFILL;
        w->progress_base = cached;
        w->status.prefill_done = 0;
        w->status.prefill_total = suffix;
        w->status.generated = 0;
        w->status.gen_tps = 0.0;
        agent_wake_locked(w);
        pthread_mutex_unlock(&w->mu);
    }

    ds4_session_set_progress(w->session, publish_progress ? worker_progress_cb : NULL,
                             publish_progress ? w : NULL);
    int rc = ds4_session_sync(w->session, tokens, err, err_len);
    ds4_session_set_progress(w->session, NULL, NULL);
    return rc;
}

/* Start a new session at the system/tool prompt.  A fixed sysprompt.kv
 * checkpoint avoids paying this prefill cost repeatedly, but only when the
 * rendered prompt text still matches the file. */
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
                                    load_err, sizeof(load_err));
        if (loaded) {
            agent_trace(w, "sysprompt kv hit file=%s tokens=%d",
                        w->sysprompt_path, w->transcript.len);
        }
    }

    if (!loaded) {
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
                                    save_err, sizeof(save_err)))
            {
                agent_buf b = {0};
                agent_buf_puts(&b, "\nds4-agent: failed to save system prompt KV: ");
                agent_buf_puts(&b, save_err);
                agent_buf_puts(&b, "\n");
                char *msg = agent_buf_take(&b);
                agent_publish(w, msg, strlen(msg));
                free(msg);
            } else {
                agent_trace(w, "sysprompt kv stored file=%s tokens=%d",
                            w->sysprompt_path, w->transcript.len);
            }
        }
    }

    pthread_mutex_lock(&w->mu);
    w->user_activity = false;
    w->session_dirty = false;
    w->status.state = AGENT_WORKER_IDLE;
    w->status.prefill_done = 0;
    w->status.prefill_total = 0;
    w->status.generated = 0;
    w->status.gen_tps = 0.0;
    w->status.error[0] = '\0';
    agent_wake_locked(w);
    pthread_mutex_unlock(&w->mu);

    free(text);
    ds4_tokens_free(&sys);
    return true;
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

/* Save the current session under its rendered-text SHA.  Before saving, sync the
 * live KV to the transcript so the file contains both the exact text and a
 * resumable checkpoint. */
static bool agent_worker_save_session(agent_worker *w, char *err, size_t err_len) {
    if (!worker_is_idle(w)) {
        snprintf(err, err_len, "model is busy");
        return false;
    }
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
    char sha[41];
    ds4_kvstore_sha1_bytes_hex(text, text_len, sha);
    char *path = agent_kv_path_for_sha(w->cache_dir, sha);

    bool ok = agent_kv_save_path(w, path, &w->transcript,
                                 "agent-session", NULL, err, err_len);
    if (ok) {
        agent_kv_delete_prefix_sessions(w, sha, text, text_len);
        pthread_mutex_lock(&w->mu);
        w->session_dirty = false;
        agent_wake_locked(w);
        pthread_mutex_unlock(&w->mu);
        printf("saved session %.8s (%d tokens)\n", sha, w->transcript.len);
    }
    free(path);
    free(text);
    return ok;
}

static void agent_print_settings(const agent_config *cfg) {
    printf("edit-mode %s  (new chats only; /set edit-mode tag|crc before the first user turn)\n",
           agent_edit_mode_name(cfg->edit_mode));
}

static bool agent_worker_set_edit_mode(agent_worker *w, agent_edit_mode mode,
                                       char *err, size_t err_len) {
    if (w->cfg->edit_mode == mode) return true;
    if (!worker_is_idle(w)) {
        snprintf(err, err_len, "model is busy");
        return false;
    }
    if (agent_worker_has_user_session(w)) {
        snprintf(err, err_len,
                 "edit-mode changes the system prompt; use /new before changing it");
        return false;
    }

    agent_edit_mode old = w->cfg->edit_mode;
    w->cfg->edit_mode = mode;
    if (!agent_worker_reset_to_sysprompt(w, err, err_len)) {
        w->cfg->edit_mode = old;
        return false;
    }
    return true;
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

/* Extract a human-readable /list title from the first user turn stored in the
 * rendered transcript.  The session file has no separate metadata by design. */
static char *agent_session_title_from_text(const char *text, size_t text_len) {
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

    while (p < end && isspace((unsigned char)*p)) p++;
    while (end > p && isspace((unsigned char)end[-1])) end--;

    agent_buf b = {0};
    bool space = false;
    bool truncated = false;
    const size_t max_bytes = 70;
    for (const char *s = p; s < end; s++) {
        unsigned char c = (unsigned char)*s;
        if (isspace(c)) {
            space = b.len != 0;
            continue;
        }
        if (space && b.len + 1 < max_bytes) {
            agent_buf_puts(&b, " ");
            space = false;
        }
        if (b.len >= max_bytes) {
            truncated = true;
            break;
        }
        agent_buf_append(&b, s, 1);
    }
    if (truncated) agent_buf_puts(&b, "...");
    if (!b.ptr || !b.len) {
        free(b.ptr);
        return xstrdup("(empty user prompt)");
    }
    return agent_buf_take(&b);
}

static char *agent_session_title_from_file(const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) return xstrdup("(unreadable session)");
    ds4_kvstore_entry hdr = {0};
    uint32_t text_bytes = 0;
    char *text = NULL;
    bool ok = ds4_kvstore_read_header(fp, &hdr, &text_bytes) &&
              agent_kv_read_text(fp, text_bytes, &text, NULL, 0);
    fclose(fp);
    char *title = ok ? agent_session_title_from_text(text, text_bytes) :
                        xstrdup("(unreadable session)");
    free(text);
    return title;
}

#define AGENT_HISTORY_DEFAULT_TURNS 3
#define AGENT_HISTORY_MAX_TURNS 200

typedef enum {
    AGENT_HISTORY_MARK_NONE,
    AGENT_HISTORY_MARK_USER,
    AGENT_HISTORY_MARK_ASSISTANT,
    AGENT_HISTORY_MARK_EOS,
} agent_history_mark;

typedef struct {
    const char **v;
    int len;
    int cap;
} agent_history_ptrs;

static void agent_history_ptrs_push(agent_history_ptrs *p, const char *s) {
    if (p->len == p->cap) {
        p->cap = p->cap ? p->cap * 2 : 16;
        p->v = xrealloc(p->v, (size_t)p->cap * sizeof(p->v[0]));
    }
    p->v[p->len++] = s;
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

static bool agent_history_is_tool_user(const char *p, const char *end) {
    agent_history_trim(&p, &end);
    return agent_history_has_prefix(p, end, "Tool:") ||
           agent_history_has_prefix(p, end, "Tool result");
}

/* Find the oldest rendered-chat marker needed to show the last N user turns.
 * Tool-result pseudo-user turns are skipped so /history remains centered on
 * the human conversation rather than internal tool plumbing. */
static const char *agent_history_start_for_turns(const char *text, size_t len,
                                                 int user_turns) {
    const char *end = text + len;
    agent_history_ptrs users = {0};
    const char *p = text;
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
        if (mark == AGENT_HISTORY_MARK_USER &&
            !agent_history_is_tool_user(content, content_end))
        {
            agent_history_ptrs_push(&users, m);
        }
        p = content_end;
    }

    const char *start = end;
    if (users.len > 0) {
        int idx = users.len - user_turns;
        if (idx < 0) idx = 0;
        start = users.v[idx];
    }
    free(users.v);
    return start;
}

static void agent_history_publish_limited(agent_worker *w, const char *p,
                                          const char *end, int max_lines,
                                          size_t max_bytes) {
    size_t bytes = 0;
    int lines = 0;
    const char *s = p;
    bool truncated = false;
    while (s < end && bytes < max_bytes && lines < max_lines) {
        const char *line = memchr(s, '\n', (size_t)(end - s));
        const char *line_end = line ? line + 1 : end;
        size_t n = (size_t)(line_end - s);
        bool cut_line = false;
        if (bytes + n > max_bytes) {
            n = max_bytes - bytes;
            cut_line = true;
        }
        agent_publish(w, s, n);
        bytes += n;
        if (cut_line) {
            s += n;
            truncated = true;
            break;
        }
        if (line) lines++;
        s = line_end;
    }
    if (s < end || truncated)
        agent_publish(w, "\n... history output truncated ...\n",
                      strlen("\n... history output truncated ...\n"));
    else if (end > p && end[-1] != '\n')
        agent_publish(w, "\n", 1);
}

static void agent_history_render_assistant(agent_worker *w,
                                           const char *p, const char *end) {
    agent_history_trim(&p, &end);
    if (p >= end) return;
    bool use_color = isatty(STDOUT_FILENO) != 0;
    agent_token_renderer renderer = {
        .engine = w->engine,
        .worker = w,
        .format_thinking = true,
        /* History replay should look like the original live output: the user is
         * switching back to a session, not reading a different transcript
         * format.  Tool calls are still dry-rendered below, so replay never
         * executes tools or mutates transcript state. */
        .format_markdown = use_color,
        .use_color = use_color,
        .last_output_newline = true,
    };
    agent_dsml_parser dsml = {.state = AGENT_DSML_SEARCH};
    agent_stream_renderer stream = {
        .renderer = &renderer,
        .parser = &dsml,
        .replay = true,
    };

    /* Dry-run replay: the same streaming projection hides DSML and renders
     * semantic tool lines, but no tool is executed and no transcript state is
     * changed.  The saved KV payload remains the only authority for resume. */
    agent_stream_text(&stream, p, (size_t)(end - p), true);
    renderer_finish(&renderer);
    agent_dsml_parser_free(&dsml);
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
    const char *p = agent_history_start_for_turns(text, len, user_turns);
    if (p >= end) {
        agent_publish(w, "\n(no user history)\n", strlen("\n(no user history)\n"));
        return;
    }

    bool color = isatty(STDOUT_FILENO) != 0;
    if (color) agent_publish(w, "\n\x1b[90m", strlen("\n\x1b[90m"));
    else agent_publish(w, "\n", 1);
    agent_publishf(w, "--- session history: last %d user turn%s ---\n",
                   user_turns, user_turns == 1 ? "" : "s");
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
                if (color) {
                    const char *s = "\x1b[90mTool result:\n";
                    agent_publish(w, s, strlen(s));
                } else {
                    agent_publish(w, "Tool result:\n", strlen("Tool result:\n"));
                }
                agent_history_publish_limited(w, tp, te, 12, 3000);
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

/* Print resumable sessions from ~/.ds4/kvcache.  sysprompt.kv is intentionally
 * ignored because it is an implementation cache, not a user session. */
static void agent_worker_list_sessions(agent_worker *w) {
    DIR *d = opendir(w->cache_dir);
    if (!d) {
        printf("no sessions: %s\n", strerror(errno));
        return;
    }
    printf("saved sessions in %s:\n", w->cache_dir);
    int n = 0;
    struct dirent *de;
    while ((de = readdir(d)) != NULL) {
        char sha[41];
        if (!ds4_kvstore_sha_hex_name(de->d_name, sha)) continue;
        char *path = ds4_kvstore_path_join(w->cache_dir, de->d_name);
        ds4_kvstore_entry e = {0};
        if (ds4_kvstore_read_entry_file(path, sha, &e)) {
            char age[32];
            agent_format_age(e.last_used, age, sizeof(age));
            char *title = agent_session_title_from_file(path);
            printf("  %.8s (%s) %s  [%u tokens, %.1f MiB]\n",
                   sha, age, title, e.tokens,
                   (double)e.file_size / (1024.0 * 1024.0));
            free(title);
            n++;
            ds4_kvstore_entry_free(&e);
        }
        free(path);
    }
    closedir(d);
    if (!n) printf("  (none)\n");
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
    struct dirent *de;
    while ((de = readdir(d)) != NULL) {
        char sha[41];
        if (!ds4_kvstore_sha_hex_name(de->d_name, sha)) continue;
        if (prefix_len && strncasecmp(sha, prefix, prefix_len) != 0) continue;

        uint64_t last_used = 0;
        char *path = ds4_kvstore_path_join(w->cache_dir, de->d_name);
        ds4_kvstore_entry e = {0};
        if (ds4_kvstore_read_entry_file(path, sha, &e)) {
            last_used = e.last_used;
            ds4_kvstore_entry_free(&e);
        }
        free(path);
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
    struct dirent *de;
    while ((de = readdir(d)) != NULL) {
        char sha[41];
        if (!ds4_kvstore_sha_hex_name(de->d_name, sha)) continue;
        if (strncasecmp(sha, prefix, plen) != 0) continue;
        matches++;
        if (matches == 1) {
            memcpy(match_sha, sha, sizeof(match_sha));
            match_path = ds4_kvstore_path_join(w->cache_dir, de->d_name);
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

    ds4_tokens loaded = {0};
    bool ok = agent_kv_load_path(w, path, sha, NULL, 0, &loaded, err, err_len);
    if (ok) {
        char *text = ds4_kvstore_render_tokens_text(w->engine, &loaded, NULL);
        if (text) {
            w->cfg->edit_mode = agent_edit_mode_from_rendered_text(text);
            free(text);
        }
        ds4_tokens_free(&w->transcript);
        w->transcript = loaded;
        pthread_mutex_lock(&w->mu);
        w->user_activity = true;
        w->session_dirty = false;
        w->status.state = AGENT_WORKER_IDLE;
        w->status.error[0] = '\0';
        agent_wake_locked(w);
        pthread_mutex_unlock(&w->mu);
        printf("switched to session %.8s (%d tokens, edit-mode=%s)\n",
               sha, w->transcript.len, agent_edit_mode_name(w->cfg->edit_mode));
        if (history_turns > 0)
            (void)agent_worker_show_history(w, history_turns, err, err_len);
    } else {
        ds4_tokens_free(&loaded);
    }
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
#define AGENT_READ_DEFAULT_LINES 500
#define AGENT_TOOL_RESULT_RESERVE_TOKENS 1024
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

/* Split a text buffer into editable line spans.  content_end excludes CR/LF so
 * line tags remain stable across the common newline spellings. */
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

/* Four base64-ish characters identify the exact rendered line the model saw.
 * This is not cryptographic; it is a compact stale-edit guard that catches
 * accidental edits after the human or another tool changed the file. */
static void agent_line_tag(const char *s, size_t n, char out[5]) {
    static const char alphabet[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
    uint32_t h = 2166136261u;
    for (size_t i = 0; i < n; i++) {
        h ^= (unsigned char)s[i];
        h *= 16777619u;
    }
    uint32_t v = h & 0xffffffu;
    out[0] = alphabet[(v >> 18) & 63];
    out[1] = alphabet[(v >> 12) & 63];
    out[2] = alphabet[(v >> 6) & 63];
    out[3] = alphabet[v & 63];
    out[4] = '\0';
}

/* CRC edit mode uses a whole-file CRC32 as a compact stale-file guard.  It is a
 * user-visible compatibility checksum, not a cryptographic identity. */
static uint32_t agent_crc32_bytes(const char *s, size_t n) {
    uint32_t crc = 0xffffffffu;
    for (size_t i = 0; i < n; i++) {
        crc ^= (unsigned char)s[i];
        for (int j = 0; j < 8; j++)
            crc = (crc >> 1) ^ (0xedb88320u & (0u - (crc & 1u)));
    }
    return crc ^ 0xffffffffu;
}

static void agent_file_tag(const char *s, size_t n, char out[9]) {
    snprintf(out, 9, "%08x", agent_crc32_bytes(s, n));
}

static bool agent_normalize_file_tag(const char *s, char out[9]) {
    if (!s) return false;
    while (*s == ' ' || *s == '\t') s++;
    if (s[0] == '0' && (s[1] == 'x' || s[1] == 'X')) s += 2;
    for (int i = 0; i < 8; i++) {
        if (!isxdigit((unsigned char)s[i])) return false;
        out[i] = (char)tolower((unsigned char)s[i]);
    }
    out[8] = '\0';
    s += 8;
    while (*s == ' ' || *s == '\t' || *s == '\r' || *s == '\n') s++;
    return *s == '\0';
}

static bool agent_verify_file_tag_arg(const char *file_tag, const char *data,
                                      size_t len, char *err, size_t err_len) {
    char expected[9], actual[9];
    if (!agent_normalize_file_tag(file_tag, expected)) {
        snprintf(err, err_len, "invalid file_tag; expected 8 hex CRC32 digits");
        return false;
    }
    agent_file_tag(data, len, actual);
    if (strcmp(expected, actual)) {
        snprintf(err, err_len,
                 "file_tag mismatch; expected %s current %s. Re-read/search and retry",
                 expected, actual);
        return false;
    }
    return true;
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
    return tokens + reserve_tokens < w->cfg->gen.ctx_size;
}

/* Read file text for the model.  Decorated mode follows the active edit mode:
 * per-line tags in tag mode, or plain line numbers plus a whole-file CRC in crc
 * mode.  Raw mode is reserved for cases where decoration would corrupt the
 * payload being inspected. */
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
        if (max_lines <= 0) max_lines = AGENT_READ_DEFAULT_LINES;
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
                     "Call the more tool with count=%d to read the next chunk.]\n",
                     end_idx, spans.len, end_idx + 1,
                     max_lines > 0 ? max_lines : AGENT_READ_DEFAULT_LINES);
            agent_buf_puts(&out, note);
        }
    } else {
        char hdr[PATH_MAX + 160];
        char file_tag[9];
        bool crc_mode = w->cfg->edit_mode == AGENT_EDIT_MODE_CRC;
        if (crc_mode) agent_file_tag(data, len, file_tag);
        if (end_idx < spans.len) {
            if (crc_mode) {
                snprintf(hdr, sizeof(hdr),
                         "%s: file_tag=%s lines %d-%d of %d; continue_offset=%d; "
                         "call the more tool with count=%d to read the next chunk\n",
                         path, file_tag, spans.len ? start_idx + 1 : 0,
                         end_idx, spans.len, end_idx + 1,
                         max_lines > 0 ? max_lines : AGENT_READ_DEFAULT_LINES);
            } else {
                snprintf(hdr, sizeof(hdr),
                         "%s: lines %d-%d of %d; continue_offset=%d; "
                         "call the more tool with count=%d to read the next chunk\n",
                         path, spans.len ? start_idx + 1 : 0, end_idx, spans.len,
                         end_idx + 1, max_lines > 0 ? max_lines : AGENT_READ_DEFAULT_LINES);
            }
        } else {
            if (crc_mode) {
                snprintf(hdr, sizeof(hdr), "%s: file_tag=%s lines %d-%d of %d\n",
                         path, file_tag, spans.len ? start_idx + 1 : 0,
                         end_idx, spans.len);
            } else {
                snprintf(hdr, sizeof(hdr), "%s: lines %d-%d of %d\n",
                         path, spans.len ? start_idx + 1 : 0, end_idx, spans.len);
            }
        }
        agent_buf_puts(&out, hdr);
        for (int i = start_idx; i < end_idx; i++) {
            agent_line_span sp = spans.v[i];
            char prefix[64];
            if (crc_mode) {
                snprintf(prefix, sizeof(prefix), "%d ", i + 1);
            } else {
                char tag[5];
                agent_line_tag(data + sp.start, sp.content_end - sp.start, tag);
                snprintf(prefix, sizeof(prefix), "%d:%s> ", i + 1, tag);
            }
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
                                        AGENT_READ_DEFAULT_LINES, 1, INT_MAX);
    bool raw = agent_parse_bool_default(agent_tool_arg_value(call, "raw"), false);
    return agent_read_range(w, path, start, count, whole, raw, true);
}

static char *agent_tool_more(agent_worker *w, const agent_tool_call *call) {
    int count = agent_parse_int_default(agent_tool_arg_value(call, "count"),
                                        AGENT_READ_DEFAULT_LINES, 1, INT_MAX);
    if (!w->more_valid) return xstrdup("Tool error: no previous output to continue\n");
    return agent_read_range(w, w->more_path, w->more_next_line, count, false,
                            w->more_bare, true);
}

static char *agent_tool_write(const agent_tool_call *call) {
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
    char msg[PATH_MAX + 128];
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

/* Classic old/new editing path.  It is intentionally conservative: the old
 * text must occur exactly once, otherwise the model should read more context
 * or switch to line-guarded editing.  In crc edit mode, file_tag adds a stale
 * whole-file guard before the old/new replacement is attempted. */
static char *agent_tool_edit_old_new(const char *path, const char *old,
                                     const char *new_text,
                                     const char *file_tag) {
    if (!old || !old[0]) return xstrdup("Tool error: edit old/new requires non-empty old text\n");
    if (!new_text) new_text = "";
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
    if (file_tag && file_tag[0] &&
        !agent_verify_file_tag_arg(file_tag, data, len, err, sizeof(err)))
    {
        agent_buf b = {0};
        agent_buf_puts(&b, "Tool error: ");
        agent_buf_puts(&b, err);
        agent_buf_puts(&b, "\n");
        free(data);
        return agent_buf_take(&b);
    }
    size_t old_len = strlen(old);
    const char *first = agent_memmem_simple(data, len, old, old_len);
    if (!first) {
        free(data);
        return xstrdup("Tool error: old text not found\n");
    }
    const char *second = agent_memmem_simple(first + old_len,
                                             len - (size_t)(first - data) - old_len,
                                             old, old_len);
    if (second) {
        free(data);
        return xstrdup("Tool error: old text is not unique; use line tags or file_tag plus line/range\n");
    }
    size_t new_len = strlen(new_text);
    size_t prefix = (size_t)(first - data);
    size_t out_len = prefix + new_len + (len - prefix - old_len);
    char *out = xmalloc(out_len + 1);
    memcpy(out, data, prefix);
    memcpy(out + prefix, new_text, new_len);
    memcpy(out + prefix + new_len, first + old_len, len - prefix - old_len);
    out[out_len] = '\0';
    int rc = agent_write_file_bytes(path, out, out_len, err, sizeof(err));
    free(data);
    free(out);
    if (rc != 0) {
        agent_buf b = {0};
        agent_buf_puts(&b, "Tool error: ");
        agent_buf_puts(&b, err);
        agent_buf_puts(&b, "\n");
        return agent_buf_take(&b);
    }
    char msg[PATH_MAX + 96];
    snprintf(msg, sizeof(msg), "Edited %s using old/new replacement\n", path);
    return xstrdup(msg);
}

/* Replace one or more lines after verifying endpoint tags.  This remains for
 * compatibility, but the system prompt steers the model toward the safer
 * bulk-lines form where every replaced line carries its own tag. */
static char *agent_tool_edit_tagged(const char *path, int start_line,
                                    const char *start_tag, int end_line,
                                    const char *end_tag, const char *text) {
    if (!text) text = "";
    if (start_line <= 0 || end_line <= 0 || end_line < start_line)
        return xstrdup("Tool error: invalid edit line range\n");
    if (!start_tag || !start_tag[0] || !end_tag || !end_tag[0])
        return xstrdup("Tool error: tagged edit requires endpoint tags\n");

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
    agent_line_spans spans = {0};
    agent_split_lines(data, len, &spans);
    if (start_line > spans.len || end_line > spans.len) {
        agent_line_spans_free(&spans);
        free(data);
        return xstrdup("Tool error: edit line range is outside the file\n");
    }
    agent_line_span s = spans.v[start_line - 1];
    agent_line_span e = spans.v[end_line - 1];
    char actual_start[5], actual_end[5];
    agent_line_tag(data + s.start, s.content_end - s.start, actual_start);
    agent_line_tag(data + e.start, e.content_end - e.start, actual_end);
    if (strcmp(actual_start, start_tag) || strcmp(actual_end, end_tag)) {
        agent_buf b = {0};
        agent_buf_puts(&b, "Tool error: line tag mismatch; re-read the file and retry\n");
        char detail[128];
        snprintf(detail, sizeof(detail), "current start=%s end=%s\n",
                 actual_start, actual_end);
        agent_buf_puts(&b, detail);
        agent_line_spans_free(&spans);
        free(data);
        return agent_buf_take(&b);
    }

    size_t text_len = strlen(text);
    bool replaced_had_newline = e.end > e.content_end;
    bool add_newline = replaced_had_newline && (text_len == 0 || text[text_len - 1] != '\n');
    size_t out_len = s.start + text_len + (add_newline ? 1 : 0) + (len - e.end);
    char *out = xmalloc(out_len + 1);
    size_t pos = 0;
    memcpy(out + pos, data, s.start);
    pos += s.start;
    memcpy(out + pos, text, text_len);
    pos += text_len;
    if (add_newline) out[pos++] = '\n';
    memcpy(out + pos, data + e.end, len - e.end);
    pos += len - e.end;
    out[pos] = '\0';

    int rc = agent_write_file_bytes(path, out, out_len, err, sizeof(err));
    agent_line_spans_free(&spans);
    free(data);
    free(out);
    if (rc != 0) {
        agent_buf b = {0};
        agent_buf_puts(&b, "Tool error: ");
        agent_buf_puts(&b, err);
        agent_buf_puts(&b, "\n");
        return agent_buf_take(&b);
    }
    char msg[PATH_MAX + 128];
    snprintf(msg, sizeof(msg), "Edited %s lines %d-%d\n", path, start_line, end_line);
    return xstrdup(msg);
}

typedef struct {
    int line;
    char tag[5];
} agent_line_ref;

typedef struct {
    agent_line_ref *v;
    int len;
    int cap;
} agent_line_refs;

static void agent_line_refs_free(agent_line_refs *refs) {
    free(refs->v);
    memset(refs, 0, sizeof(*refs));
}

static void agent_line_refs_push(agent_line_refs *refs, int line, const char tag[5]) {
    if (refs->len == refs->cap) {
        refs->cap = refs->cap ? refs->cap * 2 : 8;
        refs->v = xrealloc(refs->v, (size_t)refs->cap * sizeof(refs->v[0]));
    }
    refs->v[refs->len].line = line;
    memcpy(refs->v[refs->len].tag, tag, 5);
    refs->len++;
}

static bool agent_parse_line_ref(const char *s, int *line, char tag[5]) {
    while (*s == ' ' || *s == '\t') s++;
    if (!isdigit((unsigned char)*s)) return false;
    long ln = 0;
    while (isdigit((unsigned char)*s)) {
        ln = ln * 10 + (*s++ - '0');
        if (ln > INT_MAX) return false;
    }
    if (*s++ != ':') return false;
    for (int i = 0; i < 4; i++) {
        unsigned char c = (unsigned char)s[i];
        if (!(isalnum(c) || c == '-' || c == '_')) return false;
        tag[i] = (char)c;
    }
    tag[4] = '\0';
    *line = (int)ln;
    return true;
}

/* Parse the edit tool's bulk `lines` argument.  The model is shown only the
 * compact LINE:TAG format, but accepting "LINE:TAG> current text" makes the
 * tool robust when it copies read output verbatim. */
static bool agent_parse_line_refs(const char *lines, agent_line_refs *refs) {
    const char *p = lines;
    while (p && *p) {
        const char *eol = strpbrk(p, "\r\n");
        size_t n = eol ? (size_t)(eol - p) : strlen(p);
        char *one = xstrndup(p, n);
        int line = 0;
        char tag[5];
        bool blank = true;
        for (size_t i = 0; i < n; i++) {
            if (one[i] != ' ' && one[i] != '\t') {
                blank = false;
                break;
            }
        }
        if (!blank) {
            if (!agent_parse_line_ref(one, &line, tag)) {
                free(one);
                return false;
            }
            agent_line_refs_push(refs, line, tag);
        }
        free(one);
        if (!eol) break;
        p = eol + 1;
        if (*eol == '\r' && *p == '\n') p++;
    }
    return refs->len > 0;
}

/* Safest multi-line edit path: every line being replaced must be listed with
 * the exact tag the model saw.  This prevents silently overwriting human edits
 * inside a start/end range. */
static char *agent_tool_edit_bulk_lines(const char *path, const char *lines,
                                        const char *new_text) {
    if (!new_text) new_text = "";
    if (!lines || !lines[0])
        return xstrdup("Tool error: bulk tagged edit requires lines plus new\n");

    agent_line_refs refs = {0};
    if (!agent_parse_line_refs(lines, &refs)) {
        agent_line_refs_free(&refs);
        return xstrdup("Tool error: invalid lines format; use one LINE:TAG entry per line being replaced\n");
    }
    for (int i = 1; i < refs.len; i++) {
        if (refs.v[i].line != refs.v[i - 1].line + 1) {
            agent_line_refs_free(&refs);
            return xstrdup("Tool error: bulk tagged edit lines must be contiguous\n");
        }
    }

    char err[256];
    char *data = NULL;
    size_t len = 0;
    if (agent_read_file_bytes(path, &data, &len, err, sizeof(err)) != 0) {
        agent_buf b = {0};
        agent_buf_puts(&b, "Tool error: ");
        agent_buf_puts(&b, err);
        agent_buf_puts(&b, "\n");
        agent_line_refs_free(&refs);
        return agent_buf_take(&b);
    }
    agent_line_spans spans = {0};
    agent_split_lines(data, len, &spans);
    int first_line = refs.v[0].line;
    int last_line = refs.v[refs.len - 1].line;
    if (first_line <= 0 || last_line > spans.len) {
        agent_line_refs_free(&refs);
        agent_line_spans_free(&spans);
        free(data);
        return xstrdup("Tool error: bulk edit line range is outside the file\n");
    }
    for (int i = 0; i < refs.len; i++) {
        agent_line_span sp = spans.v[refs.v[i].line - 1];
        char actual[5];
        agent_line_tag(data + sp.start, sp.content_end - sp.start, actual);
        if (strcmp(actual, refs.v[i].tag)) {
            agent_buf b = {0};
            char detail[160];
            snprintf(detail, sizeof(detail),
                     "Tool error: line tag mismatch at line %d; expected %s current %s. Re-read and retry with fresh lines.\n",
                     refs.v[i].line, refs.v[i].tag, actual);
            agent_buf_puts(&b, detail);
            agent_line_refs_free(&refs);
            agent_line_spans_free(&spans);
            free(data);
            return agent_buf_take(&b);
        }
    }

    agent_line_span s = spans.v[first_line - 1];
    agent_line_span e = spans.v[last_line - 1];
    size_t new_len = strlen(new_text);
    bool replaced_had_newline = e.end > e.content_end;
    bool add_newline = replaced_had_newline && (new_len == 0 || new_text[new_len - 1] != '\n');
    size_t out_len = s.start + new_len + (add_newline ? 1 : 0) + (len - e.end);
    char *out = xmalloc(out_len + 1);
    size_t pos = 0;
    memcpy(out + pos, data, s.start);
    pos += s.start;
    memcpy(out + pos, new_text, new_len);
    pos += new_len;
    if (add_newline) out[pos++] = '\n';
    memcpy(out + pos, data + e.end, len - e.end);
    pos += len - e.end;
    out[pos] = '\0';

    int rc = agent_write_file_bytes(path, out, out_len, err, sizeof(err));
    agent_line_refs_free(&refs);
    agent_line_spans_free(&spans);
    free(data);
    free(out);
    if (rc != 0) {
        agent_buf b = {0};
        agent_buf_puts(&b, "Tool error: ");
        agent_buf_puts(&b, err);
        agent_buf_puts(&b, "\n");
        return agent_buf_take(&b);
    }
    char msg[PATH_MAX + 128];
    snprintf(msg, sizeof(msg), "Edited %s lines %d-%d\n", path, first_line, last_line);
    return xstrdup(msg);
}

/* Hybrid edit path for "replace this substring on the tagged line".  It keeps
 * compatibility with old/new editing while adding the stale-line guard. */
static char *agent_tool_edit_tagged_old_new(const char *path, int line,
                                            const char *tag, const char *old,
                                            const char *new_text) {
    if (!old || !old[0]) return xstrdup("Tool error: tagged old/new edit requires non-empty old text\n");
    if (!new_text) new_text = "";
    if (line <= 0 || !tag || !tag[0])
        return xstrdup("Tool error: tagged old/new edit requires line and tag\n");

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

    agent_line_spans spans = {0};
    agent_split_lines(data, len, &spans);
    if (line > spans.len) {
        agent_line_spans_free(&spans);
        free(data);
        return xstrdup("Tool error: edit line is outside the file\n");
    }

    agent_line_span sp = spans.v[line - 1];
    char actual[5];
    agent_line_tag(data + sp.start, sp.content_end - sp.start, actual);
    if (strcmp(actual, tag)) {
        agent_buf b = {0};
        agent_buf_puts(&b, "Tool error: line tag mismatch; re-read the file and retry\n");
        char detail[64];
        snprintf(detail, sizeof(detail), "current tag=%s\n", actual);
        agent_buf_puts(&b, detail);
        agent_line_spans_free(&spans);
        free(data);
        return agent_buf_take(&b);
    }

    size_t old_len = strlen(old);
    size_t line_len = sp.content_end - sp.start;
    const char *line_data = data + sp.start;
    const char *first = agent_memmem_simple(line_data, line_len, old, old_len);
    if (!first) {
        agent_line_spans_free(&spans);
        free(data);
        return xstrdup("Tool error: old text not found on the tagged line\n");
    }
    const char *second = agent_memmem_simple(first + old_len,
                                             line_len - (size_t)(first - line_data) - old_len,
                                             old, old_len);
    if (second) {
        agent_line_spans_free(&spans);
        free(data);
        return xstrdup("Tool error: old text is not unique on the tagged line\n");
    }

    size_t new_len = strlen(new_text);
    size_t old_off = (size_t)(first - data);
    size_t out_len = old_off + new_len + (len - old_off - old_len);
    char *out = xmalloc(out_len + 1);
    memcpy(out, data, old_off);
    memcpy(out + old_off, new_text, new_len);
    memcpy(out + old_off + new_len, data + old_off + old_len, len - old_off - old_len);
    out[out_len] = '\0';

    int rc = agent_write_file_bytes(path, out, out_len, err, sizeof(err));
    agent_line_spans_free(&spans);
    free(data);
    free(out);
    if (rc != 0) {
        agent_buf b = {0};
        agent_buf_puts(&b, "Tool error: ");
        agent_buf_puts(&b, err);
        agent_buf_puts(&b, "\n");
        return agent_buf_take(&b);
    }
    char msg[PATH_MAX + 128];
    snprintf(msg, sizeof(msg), "Edited %s line %d using tag-restricted old/new\n",
             path, line);
    return xstrdup(msg);
}

static bool agent_parse_line_range_arg(const char *s, int *start, int *end) {
    if (!s || !s[0]) return false;
    char *ep = NULL;
    long a = strtol(s, &ep, 10);
    if (a <= 0 || a > INT_MAX || ep == s) return false;
    while (*ep == ' ' || *ep == '\t') ep++;
    if (*ep != ':' && *ep != '-') return false;
    ep++;
    while (*ep == ' ' || *ep == '\t') ep++;
    char *ep2 = NULL;
    long b = strtol(ep, &ep2, 10);
    if (b <= 0 || b > INT_MAX || ep2 == ep) return false;
    while (*ep2 == ' ' || *ep2 == '\t' || *ep2 == '\r' || *ep2 == '\n') ep2++;
    if (*ep2 != '\0' || b < a) return false;
    *start = (int)a;
    *end = (int)b;
    return true;
}

/* CRC edit mode replaces one line or a contiguous range after verifying the
 * whole-file file_tag.  It trades per-line checks for a cheaper prompt shape:
 * if any byte in the file changed since read/search, the edit is rejected. */
static char *agent_tool_edit_crc_range(const char *path, const char *file_tag,
                                       int start_line, int end_line,
                                       const char *new_text) {
    if (!new_text) new_text = "";
    if (start_line <= 0 || end_line <= 0 || end_line < start_line)
        return xstrdup("Tool error: invalid edit line range\n");
    if (!file_tag || !file_tag[0])
        return xstrdup("Tool error: crc edit requires file_tag from the latest read/search\n");

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
    if (!agent_verify_file_tag_arg(file_tag, data, len, err, sizeof(err))) {
        agent_buf b = {0};
        agent_buf_puts(&b, "Tool error: ");
        agent_buf_puts(&b, err);
        agent_buf_puts(&b, "\n");
        free(data);
        return agent_buf_take(&b);
    }

    agent_line_spans spans = {0};
    agent_split_lines(data, len, &spans);
    if (start_line > spans.len || end_line > spans.len) {
        agent_line_spans_free(&spans);
        free(data);
        return xstrdup("Tool error: edit line range is outside the file\n");
    }

    agent_line_span s = spans.v[start_line - 1];
    agent_line_span e = spans.v[end_line - 1];
    size_t text_len = strlen(new_text);
    bool replaced_had_newline = e.end > e.content_end;
    bool add_newline = replaced_had_newline && (text_len == 0 || new_text[text_len - 1] != '\n');
    size_t out_len = s.start + text_len + (add_newline ? 1 : 0) + (len - e.end);
    char *out = xmalloc(out_len + 1);
    size_t pos = 0;
    memcpy(out + pos, data, s.start);
    pos += s.start;
    memcpy(out + pos, new_text, text_len);
    pos += text_len;
    if (add_newline) out[pos++] = '\n';
    memcpy(out + pos, data + e.end, len - e.end);
    pos += len - e.end;
    out[pos] = '\0';

    int rc = agent_write_file_bytes(path, out, out_len, err, sizeof(err));
    agent_line_spans_free(&spans);
    free(data);
    free(out);
    if (rc != 0) {
        agent_buf b = {0};
        agent_buf_puts(&b, "Tool error: ");
        agent_buf_puts(&b, err);
        agent_buf_puts(&b, "\n");
        return agent_buf_take(&b);
    }
    char msg[PATH_MAX + 160];
    snprintf(msg, sizeof(msg), "Edited %s lines %d-%d using file_tag\n",
             path, start_line, end_line);
    return xstrdup(msg);
}

/* Pick the safest edit mode supported by the tool arguments.  The preferred
 * multi-line shape is `lines` plus `new`, while old/new remains available for
 * compatibility with common coding-agent habits. */
static char *agent_tool_edit(agent_worker *w, const agent_tool_call *call) {
    const char *path = agent_tool_arg_value(call, "path");
    if (!path || !path[0]) return xstrdup("Tool error: edit requires path\n");
    const char *new_text = agent_tool_arg_value(call, "new");
    const char *lines = agent_tool_arg_value(call, "lines");
    if (lines) return agent_tool_edit_bulk_lines(path, lines, new_text);

    const char *file_tag = agent_tool_arg_value(call, "file_tag");
    int line = agent_parse_int_default(agent_tool_arg_value(call, "line"), 0, 0, INT_MAX);
    int range_start = 0, range_end = 0;
    const char *range = agent_tool_arg_value(call, "range");
    bool have_range = agent_parse_line_range_arg(range, &range_start, &range_end);
    if (range && !have_range)
        return xstrdup("Tool error: invalid range; use START:END, for example 10:20\n");
    int start = have_range ? range_start : line ? line :
        agent_parse_int_default(agent_tool_arg_value(call, "start_line"),
                                0, 0, INT_MAX);
    int end = have_range ? range_end : line ? line :
        agent_parse_int_default(agent_tool_arg_value(call, "end_line"),
                                start, 0, INT_MAX);
    const char *tag = agent_tool_arg_value(call, "tag");
    const char *start_tag = tag ? tag : agent_tool_arg_value(call, "start_tag");
    const char *end_tag = tag ? tag : agent_tool_arg_value(call, "end_tag");
    const char *old = agent_tool_arg_value(call, "old");
    bool has_usable_line_tag = (line && tag && tag[0]) ||
        (start && end && start_tag && start_tag[0] && end_tag && end_tag[0]);
    if (w->cfg->edit_mode == AGENT_EDIT_MODE_CRC &&
        (!file_tag || !file_tag[0]) && !has_usable_line_tag && (old || start || end))
    {
        return xstrdup("Tool error: crc edit mode requires file_tag from the latest read/search\n");
    }
    if (file_tag && file_tag[0]) {
        if (old) return agent_tool_edit_old_new(path, old, new_text, file_tag);
        if (start && end)
            return agent_tool_edit_crc_range(path, file_tag, start, end, new_text);
        return xstrdup("Tool error: file_tag edit requires old/new or line/start_line/range plus new\n");
    }
    if (old && line && tag)
        return agent_tool_edit_tagged_old_new(path, line, tag, old, new_text);
    if (old) return agent_tool_edit_old_new(path, old, new_text, NULL);
    if (start && end && start != end)
        return xstrdup("Tool error: bulk tagged edit requires `lines` with one LINE:TAG entry for every line being replaced. Endpoint tags are not enough; re-read the range and retry with `lines` plus `new`.\n");

    return agent_tool_edit_tagged(path, start, start_tag, end, end_tag, new_text);
}

typedef struct {
    const char *query;
    const char *glob;
    agent_edit_mode edit_mode;
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
    if (ctx->edit_mode == AGENT_EDIT_MODE_CRC) {
        snprintf(prefix, sizeof(prefix), "  %d ", line_no);
    } else {
        char tag[5];
        agent_line_tag(data + sp.start, sp.content_end - sp.start, tag);
        snprintf(prefix, sizeof(prefix), "  %d:%s> ", line_no, tag);
    }
    agent_buf_puts(&ctx->out, prefix);
    agent_buf_append(&ctx->out, data + sp.start, sp.content_end - sp.start);
    agent_buf_puts(&ctx->out, "\n");
}

/* Search one text file and emit matching lines with edit-friendly tags. */
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
            if (ctx->edit_mode == AGENT_EDIT_MODE_CRC) {
                char tag[9];
                char hdr[PATH_MAX + 64];
                agent_file_tag(data, len, tag);
                snprintf(hdr, sizeof(hdr), "%s file_tag=%s\n", path, tag);
                agent_buf_puts(&ctx->out, hdr);
            } else {
                agent_buf_puts(&ctx->out, path);
                agent_buf_puts(&ctx->out, "\n");
            }
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
    const char *query = agent_tool_arg_value(call, "query");
    if (!query || !query[0]) return xstrdup("Tool error: search requires query\n");
    const char *path = agent_tool_arg_value(call, "path");
    if (!path || !path[0]) path = ".";
    const char *mode = agent_tool_arg_value(call, "mode");
    agent_search_ctx ctx = {
        .query = query,
        .glob = agent_tool_arg_value(call, "glob"),
        .edit_mode = w->cfg->edit_mode,
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
 * Asynchronous Bash Jobs
 * ============================================================================
 *
 * Bash commands are tracked jobs, not blocking one-shot calls.  Each job owns a
 * process, a pipe, and a secure /tmp output file.  The first observation is
 * head-biased so headers and early errors are visible; later progress updates
 * are tail-biased and report how much output was added since the previous
 * observation.
 */

#define AGENT_BASH_HEAD_BYTES (64*1024)
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
    size_t new_bytes = job->bytes >= job->observed_bytes ?
                       job->bytes - job->observed_bytes : 0;
    int new_lines = display_lines >= job->observed_display_lines ?
                    display_lines - job->observed_display_lines : 0;
    double elapsed = now_sec() - job->start_time;

    agent_buf out = {0};
    char line[512];
    snprintf(line, sizeof(line),
        "bash %s job=%d pid=%ld status=%s elapsed_sec=%.1f timeout_sec=%.0f%s\n",
        first_observation ? "initial" : (job->running ? "update" : "final"),
        job->id, (long)job->pid, job->running ? "running" : "done",
        elapsed, job->timeout_sec,
        job->timed_out ? " timed_out=true" : "");
    agent_buf_puts(&out, line);
    if (!job->running) {
        snprintf(line, sizeof(line), "exit_status=%d\n", job->exit_status);
        agent_buf_puts(&out, line);
    }
    snprintf(line, sizeof(line),
        "output_path=%s\ntotal_bytes=%zu total_lines=%d\n"
        "new_bytes_since_last_update=%zu new_lines_since_last_update=%d\n",
        job->path[0] ? job->path : "<unavailable>",
        job->bytes, display_lines, new_bytes, new_lines);
    agent_buf_puts(&out, line);

    if (job->bytes == 0) {
        agent_buf_puts(&out, first_observation ?
                       "output: <no output yet>\n" :
                       "no new output since previous update\n");
    } else if (first_observation) {
        int shown_lines = 0;
        bool byte_limited = false;
        char *head = agent_bash_read_head(job, AGENT_BASH_HEAD_LINES,
                                          AGENT_BASH_HEAD_BYTES,
                                          &shown_lines, &byte_limited);
        bool truncated = byte_limited || display_lines > shown_lines;
        if (truncated) {
            snprintf(line, sizeof(line),
                     "head -%d %s (initial output; truncated, full output at output_path):\n",
                     AGENT_BASH_HEAD_LINES, job->path);
        } else {
            snprintf(line, sizeof(line), "output (complete):\n");
        }
        agent_buf_puts(&out, line);
        agent_buf_puts(&out, head);
        if (head[0] && head[strlen(head) - 1] != '\n') agent_buf_puts(&out, "\n");
        free(head);
    } else if (job->running) {
        if (new_bytes) {
            char *tail = agent_bash_read_tail_lines(job, AGENT_BASH_PROGRESS_TAIL_LINES);
            snprintf(line, sizeof(line),
                     "tail -%d %s (last lines of full output, not complete output):\n",
                     AGENT_BASH_PROGRESS_TAIL_LINES, job->path);
            agent_buf_puts(&out, line);
            agent_buf_puts(&out, tail);
            if (tail[0] && tail[strlen(tail) - 1] != '\n') agent_buf_puts(&out, "\n");
            free(tail);
        } else {
            agent_buf_puts(&out, "no new output since previous update\n");
        }
    } else {
        char *tail = agent_bash_read_tail_lines(job, AGENT_BASH_FINAL_TAIL_LINES);
        snprintf(line, sizeof(line),
                 "tail -%d %s (final last lines of full output, not complete output):\n",
                 AGENT_BASH_FINAL_TAIL_LINES, job->path);
        agent_buf_puts(&out, line);
        agent_buf_puts(&out, tail);
        if (tail[0] && tail[strlen(tail) - 1] != '\n') agent_buf_puts(&out, "\n");
        free(tail);
    }
    if (job->running) {
        snprintf(line, sizeof(line),
            "next: use bash_wait job=%d refresh_sec=60, bash_status job=%d, or bash_stop job=%d\n",
            job->id, job->id, job->id);
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
    const char *line_end = strchr(obs, '\n');
    size_t first_len = line_end ? (size_t)(line_end - obs + 1) : strlen(obs);
    agent_publish(w, "\x1b[90m", 5);
    agent_publish(w, obs, first_len);
    agent_publish(w, "\x1b[0m", 4);

    const char *body = NULL;
    const char *label = strstr(obs, "\nhead -");
    if (!label) label = strstr(obs, "\ntail -");
    if (label) {
        const char *colon = strstr(label, ":\n");
        if (colon) body = colon + 2;
    } else {
        label = strstr(obs, "\noutput (complete):\n");
        if (label) body = label + strlen("\noutput (complete):\n");
    }
    if (!body || !body[0]) return;
    const char *end = strstr(body, "\nnext:");
    size_t n = end ? (size_t)(end - body + 1) : strlen(body);
    if (n) {
        bool failed = strstr(obs, "status=done") && !strstr(obs, "exit_status=0\n");
        if (failed) agent_publish(w, "\x1b[38;5;208m", 11);
        agent_publish(w, body, n);
        if (failed) agent_publish(w, "\x1b[0m", 4);
    }
}

static void agent_bash_wait_refresh(agent_worker *w, agent_bash_job *job,
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

/* Common implementation for bash, bash_status, bash_wait, and bash_stop. */
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
    if (wait || stop) agent_bash_wait_refresh(w, job, refresh_sec);
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
    if (!strcmp(call->name, "write")) return agent_tool_write(call);
    if (!strcmp(call->name, "list")) return agent_tool_list(call);
    if (!strcmp(call->name, "edit")) return agent_tool_edit(w, call);
    if (!strcmp(call->name, "search")) return agent_tool_search(w, call);

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
        !strcmp(call->name, "bash_wait") ||
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
        bool wait = !strcmp(call->name, "bash_wait");
        bool stop = !strcmp(call->name, "bash_stop");
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
        "Bash job update after context compaction. Running jobs still need explicit bash_wait, bash_status, or bash_stop if relevant.\n");
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
    int ctx = w->cfg->gen.ctx_size;
    int used = w->transcript.len;
    if (ctx <= 0 || used <= 0) return false;
    if (used >= (ctx * AGENT_COMPACT_SOFT_PERCENT) / 100) return true;
    int free_threshold = AGENT_COMPACT_MIN_FREE_TOKENS;
    int proportional = ctx / 4;
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
    int tail_budget = w->cfg->gen.ctx_size / AGENT_COMPACT_TAIL_DIVISOR;
    if (tail_budget > AGENT_COMPACT_TAIL_CAP_TOKENS)
        tail_budget = AGENT_COMPACT_TAIL_CAP_TOKENS;
    if (tail_budget < 1) tail_budget = 1;

    int target = bottom - tail_budget;
    if (target < sys_len) target = sys_len;

    int user_id = agent_special_token_id(w->engine, "<｜User｜>");
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
        "\n\x1b[1;95mCOMPACTING\x1b[0m %s: summarizing durable task state\n\x1b[90m",
        reason && reason[0] ? reason : "context");

    char *prompt_text = agent_compact_make_prompt(reason);
    ds4_tokens prompt = {0};
    ds4_tokens_copy(&prompt, &w->transcript);
    ds4_chat_append_message(w->engine, &prompt, "user", prompt_text);
    free(prompt_text);
    ds4_chat_append_assistant_prefix(w->engine, &prompt, DS4_THINK_NONE);

    pthread_mutex_lock(&w->mu);
    w->status.state = AGENT_WORKER_COMPACTING;
    w->status.prefill_done = 0;
    w->status.prefill_total = 0;
    w->status.generated = 0;
    w->status.gen_tps = 0.0;
    agent_wake_locked(w);
    pthread_mutex_unlock(&w->mu);

    int summary_room = w->cfg->gen.ctx_size - prompt.len - 1;
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
    if (ds4_session_sync(w->session, &prompt, err, err_len) != 0) {
        ds4_session_set_progress(w->session, NULL, NULL);
        ds4_session_invalidate(w->session);
        ds4_tokens_free(&prompt);
        ds4_tokens_free(&sys);
        agent_publish(w, "\x1b[0m\n", 5);
        return false;
    }
    ds4_session_set_progress(w->session, NULL, NULL);

    /* From here until the final rebuild, the live KV contains the internal
     * compaction prompt/summary, while w->transcript still contains the real
     * conversation.  If anything fails, invalidate live KV so the next turn
     * cannot accidentally continue from the private compaction exchange. */
    agent_buf summary = {0};
    char eval_err[160] = {0};
    int think_end_id = agent_special_token_id(w->engine, "</think>");
    int dsml_id = agent_special_token_id(w->engine, "｜DSML｜");
    double t0 = now_sec();
    for (int i = 0; i < summary_max; i++) {
        if (worker_should_interrupt(w)) {
            snprintf(err, err_len, "compaction interrupted");
            ds4_session_invalidate(w->session);
            ds4_tokens_free(&prompt);
            ds4_tokens_free(&sys);
            free(summary.ptr);
            agent_publish(w, "\x1b[0m\n", 5);
            return false;
        }
        int token = ds4_session_argmax(w->session);
        if (token == ds4_token_eos(w->engine)) break;
        if (token == think_end_id || token == dsml_id) {
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

/* ============================================================================
 * Model Worker Thread
 * ============================================================================
 */

/* Run one user turn until the assistant stops or returns a tool call.  Tool
 * results are appended to the transcript and the loop continues, which gives
 * the model native DSML tool iteration without a client/server protocol. */
static int worker_run_turn(agent_worker *w, const char *user_text) {
    agent_config *cfg = w->cfg;
    ds4_think_mode think_mode = effective_think_mode(cfg);
    char compact_err[160] = {0};
    if (!agent_worker_compact_if_needed(w, "soft limit before user turn",
                                        compact_err, sizeof(compact_err)))
    {
        agent_set_error(w, compact_err[0] ? compact_err : "context compaction failed");
        return 1;
    }
    int turn_start_len = w->transcript.len;
    agent_trace_text(w, "user", user_text ? user_text : "",
                     user_text ? strlen(user_text) : 0);
    ds4_chat_append_message(w->engine, &w->transcript, "user", user_text);

    uint64_t rng = cfg->gen.seed ? cfg->gen.seed :
        ((uint64_t)time(NULL) ^ ((uint64_t)getpid() << 32) ^ (uint64_t)clock());
    pthread_mutex_lock(&w->mu);
    w->interrupt = false;
    w->user_activity = true;
    w->session_dirty = true;
    w->status.error[0] = '\0';
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
            agent_set_error(w, compact_err[0] ? compact_err : "context compaction failed");
            return 1;
        }
        int round_start_len = w->transcript.len;
        int rollback_len = tool_round == 0 ? turn_start_len : round_start_len;
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
        w->status.state = AGENT_WORKER_PREFILL;
        w->progress_base = cached;
        w->status.prefill_done = 0;
        w->status.prefill_total = suffix;
        w->status.generated = 0;
        w->status.gen_tps = 0.0;
        agent_wake_locked(w);
        pthread_mutex_unlock(&w->mu);

        char err[160];
        ds4_session_set_progress(w->session, worker_progress_cb, w);
        if (ds4_session_sync(w->session, prompt_for_sync, err, sizeof(err)) != 0) {
            ds4_session_set_progress(w->session, NULL, NULL);
            w->transcript.len = rollback_len;
            agent_set_error(w, err);
            return 1;
        }
        ds4_session_set_progress(w->session, NULL, NULL);

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
        agent_dsml_parser dsml = {.state = AGENT_DSML_SEARCH};
        agent_stream_renderer stream = {
            .renderer = &renderer,
            .parser = &dsml,
            .in_think = ds4_think_mode_enabled(think_mode),
        };
        bool got_tool = false;
        bool malformed_tool = false;
        int generated = 0;
        double t0 = now_sec();

        pthread_mutex_lock(&w->mu);
        w->status.state = AGENT_WORKER_GENERATING;
        agent_wake_locked(w);
        pthread_mutex_unlock(&w->mu);

        while (generated < max_tokens && !worker_should_interrupt(w)) {
            int token = ds4_session_sample(w->session, cfg->gen.temperature, 0,
                                           cfg->gen.top_p, cfg->gen.min_p, &rng);
            if (token == ds4_token_eos(w->engine)) break;

            if (ds4_session_eval(w->session, token, err, sizeof(err)) != 0) {
                agent_dsml_parser_free(&dsml);
                agent_set_error(w, err);
                return 1;
            }

            ds4_tokens_push(&w->transcript, token);

            size_t text_len = 0;
            char *text = ds4_token_text(w->engine, token, &text_len);
            agent_trace_token(w, token, text, text_len, generated + 1);
            agent_stream_text(&stream, text, text_len, false);
            free(text);
            generated++;

            double dt = now_sec() - t0;
            pthread_mutex_lock(&w->mu);
            w->status.generated = generated;
            w->status.gen_tps = dt > 0.0 ? (double)generated / dt : 0.0;
            agent_wake_locked(w);
            pthread_mutex_unlock(&w->mu);

            if (dsml.state == AGENT_DSML_DONE) {
                got_tool = true;
                break;
            }
            if (dsml.state == AGENT_DSML_ERROR) {
                malformed_tool = true;
                break;
            }
        }

        agent_stream_text(&stream, NULL, 0, true);
        renderer_finish(&renderer);

        if (generated == 0 && worker_should_interrupt(w)) {
            agent_dsml_parser_free(&dsml);
            /* Ctrl+C can arrive during a prefill that follows already completed
             * tool work in the same user turn.  Only the current assistant
             * prefix is speculative at that point; rolling back to the original
             * user-turn start would erase real tool calls/results that the user
             * already saw and may refer to in the next prompt. */
            w->transcript.len = rollback_len;
            ds4_session_invalidate(w->session);
            agent_set_status(w, AGENT_WORKER_IDLE);
            return 0;
        }
        ds4_tokens_push(&w->transcript, ds4_token_eos(w->engine));

        if (!got_tool && !malformed_tool) {
            agent_dsml_parser_free(&dsml);
            agent_set_status(w, AGENT_WORKER_IDLE);
            return 0;
        }

        char *tool_result;
        if (malformed_tool) {
            agent_buf b = {0};
            agent_buf_puts(&b, "Tool error: invalid DSML tool call: ");
            agent_buf_puts(&b, dsml.error[0] ? dsml.error : "parse error");
            agent_buf_puts(&b, "\n");
            tool_result = agent_buf_take(&b);
        } else {
            tool_result = agent_execute_tool_calls(w, &dsml.calls);
        }
        int projected_tokens = 0;
        if (!agent_tool_result_fits_context(w, tool_result,
                                            AGENT_TOOL_RESULT_RESERVE_TOKENS,
                                            &projected_tokens))
        {
            if (!agent_worker_compact(w, "tool result would exceed context",
                                      compact_err, sizeof(compact_err)))
            {
                free(tool_result);
                agent_dsml_parser_free(&dsml);
                agent_set_error(w, compact_err[0] ? compact_err : "context compaction failed");
                return 1;
            }
            if (!agent_tool_result_fits_context(w, tool_result,
                                                AGENT_TOOL_RESULT_RESERVE_TOKENS,
                                                &projected_tokens))
            {
                free(tool_result);
                agent_buf b = {0};
                char msg[256];
                snprintf(msg, sizeof(msg),
                         "Tool error: tool result still does not fit after context compaction "
                         "(projected_prompt=%d tokens, ctx=%d, reserve=%d). "
                         "Retry with a smaller read/search/bash output.\n",
                         projected_tokens, w->cfg->gen.ctx_size,
                         AGENT_TOOL_RESULT_RESERVE_TOKENS);
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
    }
}

/* Worker thread entry point.  The UI thread submits plain user text; this
 * thread owns all DS4 session mutation, tool execution, and compaction. */
static void *worker_main(void *arg) {
    agent_worker *w = arg;
    agent_trace(w, "agent worker start ctx=%d backend=%s model=%s trace=%s",
                w->cfg->gen.ctx_size,
                ds4_backend_name(w->cfg->engine.backend),
                w->cfg->engine.model_path ? w->cfg->engine.model_path : "",
                w->cfg->gen.trace_path ? w->cfg->gen.trace_path : "");
    char init_err[160] = {0};
    if (!agent_worker_reset_to_sysprompt(w, init_err, sizeof(init_err))) {
        agent_set_error(w, init_err[0] ? init_err : "failed to initialize system prompt");
    }
    agent_trace_tokens(w, "initial_system_prompt", &w->transcript, 0);

    while (true) {
        pthread_mutex_lock(&w->mu);
        while (!w->stop && !w->cmd_text) pthread_cond_wait(&w->cond, &w->mu);
        if (w->stop) {
            pthread_mutex_unlock(&w->mu);
            break;
        }
        char *cmd = w->cmd_text;
        w->cmd_text = NULL;
        pthread_mutex_unlock(&w->mu);

        worker_run_turn(w, cmd);
        free(cmd);
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
    bool ok = w->status.state == AGENT_WORKER_IDLE && !w->cmd_text;
    if (ok) {
        w->cmd_text = xstrdup(text);
        pthread_cond_signal(&w->cond);
    }
    pthread_mutex_unlock(&w->mu);
    return ok;
}

/* Request interruption at the next model/tool polling point. */
static void worker_interrupt(agent_worker *w) {
    pthread_mutex_lock(&w->mu);
    w->interrupt = true;
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
    w->status.ctx_size = w->cfg->gen.ctx_size;
    if (status) *status = w->status;
    w->wake_pending = false;
    pthread_mutex_unlock(&w->mu);
}

static void worker_get_status(agent_worker *w, agent_status *status) {
    pthread_mutex_lock(&w->mu);
    w->status.ctx_used = w->transcript.len;
    w->status.ctx_size = w->cfg->gen.ctx_size;
    *status = w->status;
    pthread_mutex_unlock(&w->mu);
}

static bool worker_is_idle(agent_worker *w) {
    agent_status st;
    worker_get_status(w, &st);
    return st.state == AGENT_WORKER_IDLE || st.state == AGENT_WORKER_ERROR;
}

static bool stdout_is_tty(void) {
    return isatty(STDOUT_FILENO) != 0;
}

static void agent_echo_user_prompt(const char *text) {
    if (stdout_is_tty()) {
        printf("\x1b[1;91m*\x1b[1;97m %s\x1b[0m\n\n", text);
    } else {
        printf("* %s\n\n", text);
    }
    fflush(stdout);
}

/* ============================================================================
 * Terminal Prompt, Status Footer, And Async Output Rendering
 * ============================================================================
 */

static void agent_format_ctx_size(int ctx_size, char *buf, size_t len);

#define AGENT_INPUT_INITIAL_BUFLEN 4096
#define AGENT_INPUT_MAX_BUFLEN (1024*1024)
#define AGENT_STATUS_STYLE_START "\x1b[7;90m"
#define AGENT_STATUS_STYLE_END "\x1b[0m"
#define AGENT_STATUS_BAR_FILL "\x1b[1;95m"

static void build_prompt_text(const agent_status *st, char *buf, size_t len) {
    (void)st;
    snprintf(buf, len, "ds4-agent> ");
}

static void agent_progress_bar(int done, int total, char *buf, size_t len,
                               bool color) {
    const int width = 20;
    if (len == 0) return;
    if (total <= 0) total = 1;
    if (done < 0) done = 0;
    if (done > total) done = total;
    int filled = (int)(((long long)done * width) / total);
    if (filled < 0) filled = 0;
    if (filled > width) filled = width;
    if (color && filled == 0 && done < total) filled = 1;
    size_t pos = 0;
    int n = snprintf(buf + pos, pos < len ? len - pos : 0, "[");
    if (n > 0) pos += (size_t)n;
    if (color) {
        n = snprintf(buf + pos, pos < len ? len - pos : 0, "%s", AGENT_STATUS_BAR_FILL);
        if (n > 0) pos += (size_t)n;
    }
    for (int i = 0; i < width && pos + 1 < len; i++) {
        if (color && i == filled) {
            n = snprintf(buf + pos, pos < len ? len - pos : 0, "%s", AGENT_STATUS_STYLE_START);
            if (n > 0) pos += (size_t)n;
        }
        if (pos + 1 < len) buf[pos++] = i < filled ? '#' : '-';
    }
    if (color) {
        n = snprintf(buf + pos, pos < len ? len - pos : 0, "%s", AGENT_STATUS_STYLE_START);
        if (n > 0) pos += (size_t)n;
    }
    if (pos + 1 < len) buf[pos++] = ']';
    buf[pos < len ? pos : len - 1] = '\0';
}

/* Build the one-line footer shown below the prompt.  It is intentionally compact
 * because linenoise redraws it on every progress update. */
static void build_status_text(const agent_status *st, char *buf, size_t len) {
    char used[32], total_ctx[32];
    agent_format_ctx_size(st->ctx_used, used, sizeof(used));
    agent_format_ctx_size(st->ctx_size, total_ctx, sizeof(total_ctx));

    switch (st->state) {
    case AGENT_WORKER_PREFILL: {
        int done = st->prefill_done;
        int total = st->prefill_total > 0 ? st->prefill_total : 1;
        if (done > total) done = total;
        double pct = 100.0 * (double)done / (double)total;
        char bar[128];
        agent_progress_bar(done, total, bar, sizeof(bar), stdout_is_tty());
        snprintf(buf, len, "ctx %s/%s | prefill %s %d/%d %.1f%%",
                 used, total_ctx, bar, done, total, pct);
        break;
    }
    case AGENT_WORKER_GENERATING:
        snprintf(buf, len, "ctx %s/%s | generation %d tokens %.1f t/s",
                 used, total_ctx, st->generated, st->gen_tps);
        break;
    case AGENT_WORKER_COMPACTING:
        snprintf(buf, len, "ctx %s/%s | COMPACTING summary %d tokens %.1f t/s",
                 used, total_ctx, st->generated, st->gen_tps);
        break;
    case AGENT_WORKER_ERROR:
        snprintf(buf, len, "ctx %s/%s | error: %s", used, total_ctx,
                 st->error[0] ? st->error : "unknown error");
        break;
    case AGENT_WORKER_STOPPED:
        snprintf(buf, len, "ctx %s/%s | interrupted", used, total_ctx);
        break;
    default:
        snprintf(buf, len, "ctx %s/%s | idle", used, total_ctx);
        break;
    }
}

typedef struct {
    struct linenoiseState edit;
    char *input;
    char prompt[160];
    char status[256];
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
    char cpr_buf[32];
    size_t cpr_len;
    bool paste_open;
    bool paste_start_pending;
    char paste_tail[6];
    size_t paste_tail_len;
} agent_editor;

static void editor_queue_bytes(agent_editor *ed, const char *buf, size_t len);
static void editor_hide(agent_editor *ed);

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

static void editor_clear_row(int row) {
    editor_csi_cursor(row, 1);
    write_all(STDOUT_FILENO, "\r\x1b[0K", 5);
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
    if (scroll_on_grow && ed->output_at_scroll_boundary &&
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
    editor_csi_cursor(output_bottom, 1);
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
    linenoiseEditSetStatus(&ed->edit, ed->status,
                           stdout_is_tty() ? AGENT_STATUS_STYLE_START : "",
                           stdout_is_tty() ? AGENT_STATUS_STYLE_END : "");
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
    linenoiseHide(&ed->edit);
    if (ed->scroll_region) {
        editor_restore_output_cursor(ed);
        ed->hidden = true;
        return;
    }
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
    linenoiseEditSetStatus(&ed->edit, ed->status,
                           stdout_is_tty() ? AGENT_STATUS_STYLE_START : "",
                           stdout_is_tty() ? AGENT_STATUS_STYLE_END : "");
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

/* Serialize async model/tool output with linenoise.  This is the central
 * terminal contract: hide prompt, write output, update output cursor state,
 * then redraw prompt/status if needed. */
static void editor_write_async(agent_editor *ed, const char *text, size_t len,
                               const char *prompt, const char *status,
                               bool force_show) {
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

static void runtime_help(void) {
    puts("Commands:");
    puts("  /help        Show this help.");
    puts("  /save        Save the current session.");
    puts("  /set [NAME VALUE]");
    puts("               Show settings or change settings for a new chat.");
    puts("  /list        List saved sessions.");
    puts("  /switch SHA  Load a saved session and show recent history.");
    puts("  /history [N] Show N recent user turns from the current session.");
    puts("  /new         Start a fresh session from the system prompt.");
    puts("  /quit, /exit Exit.");
    puts("  Ctrl+C       Interrupt generation; exit from an idle prompt.");
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
    ds4_session_free(w->session);
    ds4_tokens_free(&w->transcript);
    free(w->cache_dir);
    free(w->sysprompt_path);
    if (w->wake_fd[0] >= 0) close(w->wake_fd[0]);
    if (w->wake_fd[1] >= 0) close(w->wake_fd[1]);
    if (w->trace) fclose(w->trace);
    free(w->cmd_text);
    free(w->out);
    pthread_cond_destroy(&w->cond);
    pthread_mutex_destroy(&w->mu);
}

static bool agent_prompt_yes_no(const char *prompt) {
    char buf[32];
    for (;;) {
        printf("%s", prompt);
        fflush(stdout);
        if (!fgets(buf, sizeof(buf), stdin)) return false;
        char *p = buf;
        while (*p == ' ' || *p == '\t') p++;
        if (*p == 'y' || *p == 'Y') return true;
        if (*p == 'n' || *p == 'N') return false;
    }
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

/* ============================================================================
 * Interactive Runtime Loop
 * ============================================================================
 */

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
    char statusline[256];
    build_prompt_text(&st, prompt, sizeof(prompt));
    build_status_text(&st, statusline, sizeof(statusline));

    agent_editor editor = {0};
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
    char *restore_line = NULL;
    while (running) {
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
                running = false;
            } else {
                worker_interrupt(&worker);
            }
        }

        if (rc > 0 && (pfd[1].revents & POLLIN)) drain_wake_fd(worker.wake_fd[0]);

        char *out = NULL;
        size_t out_len = 0;
        worker_consume(&worker, &out, &out_len, &st);
        build_prompt_text(&st, prompt, sizeof(prompt));
        build_status_text(&st, statusline, sizeof(statusline));
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
            worker.status.error[0] = '\0';
            pthread_mutex_unlock(&worker.mu);
        }
        free(out);

        if (initial_pending && worker_is_idle(&worker)) {
            if (worker_submit(&worker, initial_pending)) {
                free(initial_pending);
                initial_pending = NULL;
            }
        }

        if (rc > 0 && (pfd[0].revents & POLLIN)) editor_read_stdin(&editor);

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
                    if (!worker_is_idle(&worker)) worker_interrupt(&worker);
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
                if (!cmd[0]) {
                    /* Empty input: just reopen the editor. */
                } else if (!worker_is_idle(&worker)) {
                    /* The model is still busy.  For the MVP, keep the same
                     * text editable instead of submitting it.  Later this is
                     * where queued user messages should be implemented. */
                    restore_line = xstrdup(cmd);
                } else if (!strcmp(cmd, "/quit") || !strcmp(cmd, "/exit")) {
                    editor_restore_terminal_layout(&editor);
                    if (agent_maybe_save_before_leaving_session(&worker)) {
                        exit_save_handled = true;
                        running = false;
                    }
                } else if (!strcmp(cmd, "/help")) {
                    runtime_help();
                } else if (!strcmp(cmd, "/save")) {
                    char err[160] = {0};
                    if (!agent_worker_save_session(&worker, err, sizeof(err)))
                        printf("save failed: %s\n", err);
                } else if (!strncmp(cmd, "/set", 4) &&
                           (cmd[4] == '\0' || cmd[4] == ' ' || cmd[4] == '\t')) {
                    char *arg = cmd + 4;
                    while (*arg == ' ' || *arg == '\t') arg++;
                    if (!arg[0]) {
                        agent_print_settings(cfg);
                    } else {
                        char *name = arg;
                        while (*arg && *arg != ' ' && *arg != '\t') arg++;
                        if (*arg) *arg++ = '\0';
                        while (*arg == ' ' || *arg == '\t') arg++;
                        if (!strcmp(name, "edit-mode")) {
                            char *value = arg;
                            while (*arg && *arg != ' ' && *arg != '\t') arg++;
                            if (*arg) *arg++ = '\0';
                            while (*arg == ' ' || *arg == '\t') arg++;
                            agent_edit_mode mode;
                            if (*arg || !agent_parse_edit_mode(value, &mode)) {
                                printf("usage: /set edit-mode tag|crc\n");
                            } else {
                                char err[160] = {0};
                                if (!agent_worker_set_edit_mode(&worker, mode,
                                                                err, sizeof(err)))
                                {
                                    printf("set failed: %s\n", err);
                                } else {
                                    printf("edit-mode %s\n",
                                           agent_edit_mode_name(cfg->edit_mode));
                                }
                            }
                        } else {
                            printf("unknown setting: %s\n", name);
                        }
                    }
                } else if (!strcmp(cmd, "/list")) {
                    agent_worker_list_sessions(&worker);
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
                        }
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
                } else if (cmd[0] == '/') {
                    printf("unknown command: %s\n", cmd);
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
                    build_status_text(&st, statusline, sizeof(statusline));
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
                    free(restore_line);
                    restore_line = NULL;
                }
            }
        }
    }

    free(initial_pending);
    free(restore_line);
    editor_stop(&editor);
    editor_restore_terminal_layout(&editor);
    linenoiseSetCompletionCallback(NULL);
    agent_completion_worker = NULL;
    if (!exit_save_handled)
        (void)agent_maybe_save_before_leaving_session(&worker);
    agent_worker_free(&worker);
    return 0;
}

int main(int argc, char **argv) {
    agent_config cfg = parse_options(argc, argv);
    log_context_memory(cfg.engine.backend, cfg.gen.ctx_size);

    ds4_engine *engine = NULL;
    if (ds4_engine_open(&engine, &cfg.engine) != 0) return 1;

    struct sigaction old_int;
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sigemptyset(&sa.sa_mask);
    sa.sa_handler = agent_sigint_handler;
    bool sigint_installed = sigaction(SIGINT, &sa, &old_int) == 0;

    int rc = run_agent(engine, &cfg);

    if (sigint_installed) sigaction(SIGINT, &old_int, NULL);
    ds4_engine_close(engine);
    return rc;
}
