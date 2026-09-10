#define _DARWIN_C_SOURCE
#include "ds4.h"
#include "ds4_gpu.h"

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

/* Profile actual session dispatches without enabling stage-profile gates.
 * Timeline instrumentation changes encoder boundaries: these wall times are
 * diagnostic, not an uninstrumented throughput benchmark. */
typedef struct {
    const char *model, *prompt, *timeline, *csv;
    int prefix, ctx, warmup, tokens, chunk;
    int trace_mode;
    bool ssd;
} config;

static void usage(FILE *out, const char *program) {
    fprintf(out,
        "usage: %s -m MODEL --csv NEW_FILE (--timeline NEW_FILE | --no-trace) [options]\n"
        "  --prompt-file PATH    text to tokenize (default speed-bench/promessi_sposi.txt)\n"
        "  --prefix-tokens N     exact prefill length (default 2048)\n"
        "  --ctx N               context allocation (default 4096)\n"
        "  --warmup N            untallied decode steps (default 4)\n"
        "  --tokens N            measured decode steps (default 16)\n"
        "  --prefill-chunk N     prefill chunk size (default 128)\n"
        "  --ssd-streaming       stream experts from SSD (default resident)\n"
        "  --no-trace            control run to assess instrumentation overhead\n"
        "Outputs must not already exist. Selection excludes EOS to ensure the\n"
        "requested step count. The prompt is raw text, without a chat template.\n",
        program);
}

static void fail(const char *message) {
    fprintf(stderr, "metal-decode-profile: %s\n", message);
    exit(2);
}

static const char *argument(int *i, int argc, char **argv) {
    if (++*i >= argc) fail("missing option argument");
    return argv[*i];
}

static int number(const char *value, int minimum) {
    char *end = NULL;
    errno = 0;
    long n = strtol(value, &end, 10);
    if (errno || !value[0] || *end || n < minimum || n > INT_MAX)
        fail("invalid integer option");
    return (int)n;
}

static config options(int argc, char **argv) {
    config c = {.prompt = "speed-bench/promessi_sposi.txt", .prefix = 2048,
                .ctx = 4096, .warmup = 4, .tokens = 16, .chunk = 128};
    for (int i = 1; i < argc; ++i) {
        const char *a = argv[i];
        if (!strcmp(a, "--help") || !strcmp(a, "-h")) { usage(stdout, argv[0]); exit(0); }
        else if (!strcmp(a, "-m") || !strcmp(a, "--model")) c.model = argument(&i, argc, argv);
        else if (!strcmp(a, "--prompt-file")) c.prompt = argument(&i, argc, argv);
        else if (!strcmp(a, "--csv")) c.csv = argument(&i, argc, argv);
        else if (!strcmp(a, "--timeline")) {
            if (c.trace_mode) fail("choose exactly one of --timeline and --no-trace");
            c.timeline = argument(&i, argc, argv); c.trace_mode = 1;
        } else if (!strcmp(a, "--no-trace")) {
            if (c.trace_mode) fail("choose exactly one of --timeline and --no-trace");
            c.trace_mode = -1;
        } else if (!strcmp(a, "--ssd-streaming")) c.ssd = true;
        else if (!strcmp(a, "--prefix-tokens")) c.prefix = number(argument(&i, argc, argv), 1);
        else if (!strcmp(a, "--ctx")) c.ctx = number(argument(&i, argc, argv), 1);
        else if (!strcmp(a, "--warmup")) c.warmup = number(argument(&i, argc, argv), 0);
        else if (!strcmp(a, "--tokens")) c.tokens = number(argument(&i, argc, argv), 1);
        else if (!strcmp(a, "--prefill-chunk")) c.chunk = number(argument(&i, argc, argv), 1);
        else fail("unknown option (use --help)");
    }
    if (!c.model || !c.csv || !c.trace_mode) fail("model, CSV and trace mode are required");
    if ((int64_t)c.prefix + c.warmup + c.tokens >= c.ctx)
        fail("context must exceed prefix + warmup + measured tokens");
    return c;
}

static double seconds(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + t.tv_nsec * 1e-9;
}

static FILE *new_file(const char *path) {
    const int fd = open(path, O_WRONLY | O_CREAT | O_EXCL, 0600);
    if (fd < 0) { perror(path); return NULL; }
    FILE *f = fdopen(fd, "w");
    if (!f) close(fd);
    return f;
}

static char *read_prompt(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return NULL; }
    char *text = NULL;
    if (fseek(f, 0, SEEK_END)) goto done;
    const long length = ftell(f);
    if (length < 0 || (uint64_t)length > SIZE_MAX - 1 || fseek(f, 0, SEEK_SET)) goto done;
    text = malloc((size_t)length + 1);
    if (!text) goto done;
    if (fread(text, 1, (size_t)length, f) != (size_t)length) { free(text); text = NULL; }
    else text[length] = '\0';
done:
    fclose(f);
    return text;
}

static int phase(const config *c, const char *name, uint32_t position) {
    if (!c->timeline) return 1;
    if (ds4_gpu_timeline_set_phase(name, position) == 1) return 1;
    fprintf(stderr, "metal-decode-profile: cannot mark %s; timeline unsupported or batch still open\n", name);
    return 0;
}

int main(int argc, char **argv) {
    const config c = options(argc, argv);
    /* Stage profiling changes fusion eligibility, unlike the encoder timeline. */
    const char *conflicts[] = {"DS4_METAL_DECODE_STAGE_PROFILE", "DS4_METAL_LAYER_STAGE_PROFILE",
        "DS4_METAL_INDEXER_STAGE_PROFILE", "DS4_METAL_OUTPUT_STAGE_PROFILE",
        "DS4_METAL_GRAPH_DUMP_PREFIX", "DS4_METAL_MOE_ONE_STAGE_PROFILE",
        "DS4_METAL_MOE_STAGE_PROFILE", "DS4_METAL_FLASH_ATTN_STAGE_PROFILE",
        "DS4_METAL_ATTN_OUT_STAGE_PROFILE", "DS4_METAL_Q_STAGE_PROFILE",
        "DS4_METAL_Q8_PREFILL_PROFILE", "DS4_ROCM_LAYER_STAGE_PROFILE",
        "DS4_ROCM_DECODE_STAGE_PROFILE", "DS4_ROCM_INDEXER_STAGE_PROFILE",
        "DS4_ROCM_Q_STAGE_PROFILE"};
    for (unsigned i = 0; i < sizeof(conflicts)/sizeof(*conflicts); ++i)
        if (getenv(conflicts[i])) fail("unset stage profiling/graph dumps before measuring actual dispatches");
    if (access(c.model, R_OK)) { perror(c.model); return 2; }
    char *text = read_prompt(c.prompt);
    if (!text) fail("could not read prompt");
    if (c.timeline) {
        FILE *trace = new_file(c.timeline);
        if (!trace) { free(text); return 2; }
        fclose(trace);
        if (setenv("DS4_METAL_ENCODER_TIMELINE", c.timeline, 1)) fail("cannot set timeline path");
    } else if (unsetenv("DS4_METAL_ENCODER_TIMELINE")) fail("cannot disable timeline for control run");
    FILE *csv = new_file(c.csv);
    if (!csv) { free(text); return 2; }
    fprintf(csv, "phase,step,position,token,seconds,select_seconds,eval_seconds\n");
    fprintf(stderr,
        "metal-decode-profile: model=%s prompt=%s prefix=%d ctx=%d warmup=%d tokens=%d "
        "prefill_chunk=%d mode=%s timeline=%s\n",
        c.model, c.prompt, c.prefix, c.ctx, c.warmup, c.tokens, c.chunk,
        c.ssd ? "ssd-streaming" : "resident", c.timeline ? c.timeline : "off");
    ds4_engine_options opt = {.model_path = c.model, .backend = DS4_BACKEND_METAL,
        .context_size = c.ctx, .prefill_chunk = (uint32_t)c.chunk, .power_percent = 100,
        .warm_weights = !c.ssd, .ssd_streaming = c.ssd};
    ds4_engine *engine = NULL;
    ds4_session *session = NULL;
    ds4_tokens tokens = {0};
    char error[256] = {0};
    int result = 1;
    if (ds4_engine_open(&engine, &opt)) goto done;
    fprintf(stderr, "metal-decode-profile: loaded=%s bytes=%llu\n",
        ds4_engine_model_name(engine), (unsigned long long)ds4_engine_model_bytes(engine));
    ds4_tokenize_text(engine, text, &tokens);
    if (tokens.len < c.prefix) { fprintf(stderr, "prompt has %d tokens; need %d\n", tokens.len, c.prefix); goto done; }
    if (!phase(&c, "setup", 0) || ds4_session_create(&session, engine, c.ctx)) goto done;
    ds4_tokens prefix = {.v = tokens.v, .len = c.prefix, .cap = c.prefix};
    if (!phase(&c, "prefill", (uint32_t)c.prefix)) goto done;
    double t0 = seconds();
    if (ds4_session_sync(session, &prefix, error, sizeof(error))) goto done;
    double duration = seconds() - t0;
    fprintf(csv, "prefill,0,%d,-1,%.9f,0,%.9f\n", c.prefix, duration, duration);
    fflush(csv);
    const int eos = ds4_token_eos(engine);
    double measured = 0;
    for (int i = 0; i < c.warmup + c.tokens; ++i) {
        const char *name = i < c.warmup ? "warmup" : "decode";
        const int step = i < c.warmup ? i : i - c.warmup;
        const uint32_t position = (uint32_t)ds4_session_tokens(session)->len;
        if (!phase(&c, "selection", position)) goto done;
        t0 = seconds();
        const int token = ds4_session_argmax_excluding(session, eos);
        const double select_seconds = seconds() - t0;
        if (token < 0 || !phase(&c, name, position)) goto done;
        const double eval_start = seconds();
        if (ds4_session_eval(session, token, error, sizeof(error))) goto done;
        const double eval_seconds = seconds() - eval_start;
        duration = seconds() - t0;
        fprintf(csv, "%s,%d,%u,%d,%.9f,%.9f,%.9f\n", name, step, position, token,
            duration, select_seconds, eval_seconds);
        fflush(csv);
        if (i >= c.warmup) measured += duration;
        fprintf(stderr, "metal-decode-profile: %s step=%d pos=%u token=%d wall=%.3f ms\n",
            name, step, position, token, duration * 1e3);
    }
    fprintf(stderr, "metal-decode-profile: %d measured steps, wall=%.6f s (%s)\n",
        c.tokens, measured, c.timeline ? "instrumented" : "control");
    result = 0;
done:
    if (result) fprintf(stderr, "metal-decode-profile: failed: %s\n", error[0] ? error : "see diagnostics above");
    if (session) {
        (void)phase(&c, "cleanup", 0);
        ds4_session_free(session);
    }
    if (engine) ds4_engine_close(engine);
    ds4_tokens_free(&tokens);
    free(text);
    if (fclose(csv)) result = 1;
    return result;
}
