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

enum {
    DS4_BENCH_N_VOCAB = 129280,
};

#define DS4_BENCH_FORCED_LOGIT_DRIFT_EPS 1e-5f

typedef struct {
    const char *model_path;
    const char *prompt_path;
    const char *chat_prompt_path;
    const char *system;
    const char *csv_path;
    const char *mtp_path;
    const char *exact_dump_prefix;
    const char *replay_dump_prefix;
    ds4_backend backend;
    int threads;
    int ctx_start;
    int ctx_max;
    int ctx_alloc;
    int step_incr;
    int gen_tokens;
    int mtp_draft_tokens;
    int spec_probe_ngram;
    int spec_probe_draft;
    int exact_replay_runs;
    int exact_replay_topk;
    double step_mul;
    float mtp_margin;
    bool warm_weights;
    bool quality;
    bool spec_probe;
    bool exact_replay_probe;
    bool exact_replay_fresh_session;
    bool exact_replay_snapshot;
    bool exact_replay_snapshot_fresh_session;
    bool exact_replay_forced_logit_diff;
    bool exact_replay_snapshot_roundtrip;
} bench_config;

static double bench_now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1000000000.0;
}

static void bench_set_dump_prefix(const char *prefix) {
    if (prefix && prefix[0]) setenv("DS4_METAL_GRAPH_DUMP_PREFIX", prefix, 1);
    else unsetenv("DS4_METAL_GRAPH_DUMP_PREFIX");
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
        "  --mtp FILE             Optional MTP GGUF used for exact speculative decode.\n"
        "  --mtp-draft N          Maximum MTP draft tokens per verifier cycle. Default: 2\n"
        "  --mtp-margin F         MTP confidence margin. Default: 3\n"
        "      For exact MTP probes, set DS4_MTP_RESTORE_DRAFT_FRONTIER=1.\n"
        "  --metal | --cuda | --cpu | --backend NAME\n"
        "      Select backend explicitly. Defaults to Metal on macOS, CUDA elsewhere.\n"
        "  -t, --threads N        CPU helper threads.\n"
        "  --quality              Prefer exact kernels where applicable.\n"
        "  --warm-weights         Touch mapped tensor pages before benchmarking.\n"
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
        "  --spec-probe           Report prefix-retrieval speculation upper-bound tok/s.\n"
        "  --spec-ngram N         Match length for --spec-probe. Default: 4\n"
        "  --spec-draft N         Max draft tokens for --spec-probe. Default: 4\n"
        "  --exact-replay-probe   Compare exact greedy decode after full exact replay prefill.\n"
        "  --exact-replay-fresh-session\n"
        "      Use a separate fresh session for the replay side of --exact-replay-probe.\n"
        "      Note: exact replay may load MTP with --mtp FILE without running MTP decode.\n"
        "  --exact-replay-snapshot\n"
        "      Restore the saved pre-generation snapshot for replay instead of refilling.\n"
        "  --exact-replay-snapshot-fresh-session\n"
        "      Restore the saved snapshot into a new session for each replay run.\n"
        "  --exact-replay-runs N\n"
        "      Repeat exact replay N times after one exact decode.\n"
        "  --exact-replay-topk N\n"
        "      Capture top-N logits for exact replay mismatch diagnostics. Default: 2\n"
        "  --exact-replay-forced-logit-diff\n"
        "      Force replay through exact tokens and report post-token logit drift.\n"
        "  --exact-replay-snapshot-roundtrip\n"
        "      After restoring a snapshot, re-save it and compare payload bytes before decode.\n"
        "  --exact-dump-prefix PREFIX\n"
        "      Dump exact decode graph tensors with this prefix.\n"
        "  --replay-dump-prefix PREFIX\n"
        "      Dump replay decode graph tensors with this prefix.\n"
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
        .mtp_draft_tokens = 2,
        .spec_probe_ngram = 4,
        .spec_probe_draft = 4,
        .exact_replay_runs = 1,
        .exact_replay_topk = 2,
        .step_mul = 1.0,
        .mtp_margin = 3.0f,
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
            c.mtp_margin = (float)parse_double_arg(need_arg(&i, argc, argv, arg), arg);
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
        } else if (!strcmp(arg, "--spec-probe")) {
            c.spec_probe = true;
        } else if (!strcmp(arg, "--spec-ngram")) {
            c.spec_probe_ngram = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--spec-draft")) {
            c.spec_probe_draft = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--exact-replay-probe")) {
            c.exact_replay_probe = true;
        } else if (!strcmp(arg, "--exact-replay-fresh-session")) {
            c.exact_replay_probe = true;
            c.exact_replay_fresh_session = true;
        } else if (!strcmp(arg, "--exact-replay-snapshot")) {
            c.exact_replay_probe = true;
            c.exact_replay_snapshot = true;
            c.exact_replay_fresh_session = false;
        } else if (!strcmp(arg, "--exact-replay-snapshot-fresh-session")) {
            c.exact_replay_probe = true;
            c.exact_replay_snapshot = true;
            c.exact_replay_snapshot_fresh_session = true;
            c.exact_replay_fresh_session = false;
        } else if (!strcmp(arg, "--exact-replay-runs")) {
            c.exact_replay_probe = true;
            if (!c.exact_replay_snapshot) c.exact_replay_fresh_session = true;
            c.exact_replay_runs = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--exact-replay-topk")) {
            c.exact_replay_probe = true;
            c.exact_replay_topk = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--exact-replay-forced-logit-diff")) {
            c.exact_replay_probe = true;
            c.exact_replay_forced_logit_diff = true;
        } else if (!strcmp(arg, "--exact-replay-snapshot-roundtrip")) {
            c.exact_replay_probe = true;
            c.exact_replay_snapshot = true;
            c.exact_replay_snapshot_roundtrip = true;
        } else if (!strcmp(arg, "--exact-dump-prefix")) {
            c.exact_dump_prefix = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--replay-dump-prefix")) {
            c.replay_dump_prefix = need_arg(&i, argc, argv, arg);
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
    if (c.spec_probe_ngram <= 0 || c.spec_probe_draft <= 0) {
        fprintf(stderr, "ds4-bench: --spec-ngram and --spec-draft must be positive\n");
        exit(2);
    }
    if (c.mtp_path && c.mtp_draft_tokens <= 0) {
        fprintf(stderr, "ds4-bench: --mtp-draft must be positive\n");
        exit(2);
    }
    if (c.mtp_margin < 0.0f) {
        fprintf(stderr, "ds4-bench: --mtp-margin must be non-negative\n");
        exit(2);
    }
    if (c.exact_replay_runs > 1 &&
        !c.exact_replay_fresh_session &&
        !c.exact_replay_snapshot) {
        fprintf(stderr, "ds4-bench: --exact-replay-runs requires fresh-session or snapshot replay\n");
        exit(2);
    }
    if (c.exact_replay_snapshot && c.exact_replay_fresh_session) {
        fprintf(stderr, "ds4-bench: choose only one of --exact-replay-snapshot and --exact-replay-fresh-session\n");
        exit(2);
    }
    if (c.exact_replay_snapshot_fresh_session && !c.exact_replay_snapshot) {
        fprintf(stderr, "ds4-bench: --exact-replay-snapshot-fresh-session requires --exact-replay-snapshot\n");
        exit(2);
    }
    if (c.exact_replay_topk < 2 || c.exact_replay_topk > 64) {
        fprintf(stderr, "ds4-bench: --exact-replay-topk must be between 2 and 64\n");
        exit(2);
    }
    return c;
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

static int bench_token_at(const ds4_tokens *prefix, const int *generated, int index) {
    if (index < prefix->len) return prefix->v[index];
    return generated[index - prefix->len];
}

static int spec_probe_prefix_retrieval(
        const ds4_tokens *prefix,
        const int        *generated,
        int               generated_len,
        int               generated_pos,
        int               ngram,
        int               draft_cap,
        int              *drafted) {
    *drafted = 0;
    if (!prefix || !generated || generated_pos < 0 ||
        generated_pos >= generated_len || ngram <= 0 || draft_cap <= 0) {
        return 0;
    }

    const int ctx_len = prefix->len + generated_pos;
    if (ctx_len < ngram) return 0;
    const int query = ctx_len - ngram;
    int best = -1;
    for (int i = query - 1; i >= 0; i--) {
        bool match = true;
        for (int j = 0; j < ngram; j++) {
            if (bench_token_at(prefix, generated, i + j) !=
                bench_token_at(prefix, generated, query + j)) {
                match = false;
                break;
            }
        }
        if (match) {
            best = i;
            break;
        }
    }
    if (best < 0) return 0;

    int accepted = 0;
    const int draft_start = best + ngram;
    for (int d = 0; d < draft_cap && generated_pos + d < generated_len; d++) {
        if (draft_start + d >= ctx_len) break;
        (*drafted)++;
        if (bench_token_at(prefix, generated, draft_start + d) != generated[generated_pos + d]) {
            break;
        }
        accepted++;
    }
    return accepted;
}

static void spec_probe_report(
        int               frontier,
        const ds4_tokens *prefix,
        const int        *generated,
        int               generated_len,
        int               ngram,
        int               draft_cap,
        double            gen_tps) {
    int cycles = 0;
    int matched_cycles = 0;
    int drafted_tokens = 0;
    int accepted_drafts = 0;
    int accepted_extra = 0;

    for (int i = 0; i < generated_len; ) {
        int drafted = 0;
        int accepted = spec_probe_prefix_retrieval(prefix,
                                                   generated,
                                                   generated_len,
                                                   i,
                                                   ngram,
                                                   draft_cap,
                                                   &drafted);
        cycles++;
        drafted_tokens += drafted;
        accepted_drafts += accepted;
        if (accepted > 0) {
            matched_cycles++;
            accepted_extra += accepted > 1 ? accepted - 1 : 0;
            i += accepted;
        } else {
            i++;
        }
    }

    const double avg_commit = cycles > 0 ? (double)generated_len / (double)cycles : 1.0;
    const double draft_accept =
        drafted_tokens > 0 ? (double)accepted_drafts / (double)drafted_tokens : 0.0;
    const double effective_tps_upper = gen_tps * avg_commit;

    fprintf(stderr,
            "ds4-bench: spec probe frontier=%d kind=prefix_retrieval ngram=%d draft=%d generated=%d cycles=%d matched_cycles=%d drafted=%d accepted=%d accepted_extra=%d avg_commit=%.3f draft_accept=%.3f effective_tps_upper=%.2f\n",
            frontier,
            ngram,
            draft_cap,
            generated_len,
            cycles,
            matched_cycles,
            drafted_tokens,
            accepted_drafts,
            accepted_extra,
            avg_commit,
            draft_accept,
            effective_tps_upper);
}

typedef struct {
    int generated;
    int cycles;
    int spec_opportunity_cycles;
    int full_accept_cycles;
    int partial_accept_cycles;
    int reject_cycles;
    int accepted_extra;
    int possible_extra;
    int exact_match;
    int first_mismatch;
    double exact_sec;
    double spec_sec;
} mtp_spec_stats;

static int bench_decode_exact_tokens(
        ds4_session *session,
        int          eos,
        int          max_tokens,
        int         *tokens,
        ds4_token_score *top2,
        int          topk,
        float       *after_logits,
        int          after_logits_stride,
        char        *err,
        size_t       errlen) {
    for (int i = 0; i < max_tokens; i++) {
        if (ds4_session_pos(session) + 1 >= ds4_session_ctx(session)) {
            snprintf(err, errlen, "generation would exceed allocated context");
            return -1;
        }
        if (top2) {
            int n = ds4_session_top_logprobs(session, &top2[(size_t)i * (size_t)topk], topk);
            if (n < topk) {
                snprintf(err, errlen, "failed to read top logits");
                return -1;
            }
        }
        const int token = ds4_session_argmax_excluding(session, eos);
        if (token < 0) {
            snprintf(err, errlen, "failed to choose non-EOS token");
            return -1;
        }
        tokens[i] = token;
        if (ds4_session_eval_exact(session, token, err, errlen) != 0) return -1;
        if (after_logits) {
            if (ds4_session_copy_logits(session,
                                        &after_logits[(size_t)i * (size_t)after_logits_stride],
                                        after_logits_stride) == 0)
            {
                snprintf(err, errlen, "failed to copy logits");
                return -1;
            }
        }
    }
    return max_tokens;
}

static int bench_score_rank(const ds4_token_score *scores, int k, int token) {
    if (!scores || k <= 0) return -1;
    for (int i = 0; i < k; i++) {
        if (scores[i].id == token) return i;
    }
    return -1;
}

static float bench_score_logit_or_nan(const ds4_token_score *scores, int k, int token) {
    const int rank = bench_score_rank(scores, k, token);
    return rank >= 0 ? scores[rank].logit : NAN;
}

static void bench_logits_diff(
        const float *a,
        const float *b,
        int          n,
        float       *max_abs_out,
        float       *rms_out,
        int         *max_id_out) {
    double sum_sq = 0.0;
    float max_abs = 0.0f;
    int max_id = -1;
    for (int i = 0; i < n; i++) {
        const float d = fabsf(a[i] - b[i]);
        sum_sq += (double)d * (double)d;
        if (d > max_abs) {
            max_abs = d;
            max_id = i;
        }
    }
    if (max_abs_out) *max_abs_out = max_abs;
    if (rms_out) *rms_out = n > 0 ? (float)sqrt(sum_sq / (double)n) : 0.0f;
    if (max_id_out) *max_id_out = max_id;
}

static int bench_decode_mtp_tokens(
        ds4_session    *session,
        int             eos,
        int             max_tokens,
        int             mtp_draft_tokens,
        int            *tokens,
        mtp_spec_stats *stats,
        char           *err,
        size_t          errlen) {
    int generated = 0;
    while (generated < max_tokens) {
        if (ds4_session_pos(session) + 1 >= ds4_session_ctx(session)) {
            snprintf(err, errlen, "generation would exceed allocated context");
            return -1;
        }
        const int token = ds4_session_argmax_excluding(session, eos);
        if (token < 0) {
            snprintf(err, errlen, "failed to choose non-EOS token");
            return -1;
        }

        int accepted[17];
        const int remaining = max_tokens - generated;
        int cap = remaining < (int)(sizeof(accepted) / sizeof(accepted[0]))
            ? remaining
            : (int)(sizeof(accepted) / sizeof(accepted[0]));
        int ntok = ds4_session_eval_speculative_argmax(session,
                                                       token,
                                                       remaining,
                                                       eos,
                                                       accepted,
                                                       cap,
                                                       err,
                                                       errlen);
        if (ntok <= 0) return -1;
        if (ntok > remaining) ntok = remaining;

        int possible_extra = remaining > 1
            ? (remaining - 1 < mtp_draft_tokens ? remaining - 1 : mtp_draft_tokens)
            : 0;
        if (possible_extra > cap - 1) possible_extra = cap - 1;
        stats->cycles++;
        if (possible_extra > 0) {
            stats->spec_opportunity_cycles++;
            stats->possible_extra += possible_extra;
            const int extra = ntok > 1 ? ntok - 1 : 0;
            stats->accepted_extra += extra;
            if (extra == possible_extra) stats->full_accept_cycles++;
            else if (extra > 0) stats->partial_accept_cycles++;
            else stats->reject_cycles++;
        }

        for (int i = 0; i < ntok; i++) tokens[generated++] = accepted[i];
    }
    stats->generated = generated;
    return generated;
}

static void mtp_spec_report(
        int                   frontier,
        int                   draft_tokens,
        const mtp_spec_stats *stats) {
    const double target_cycles_per_sec =
        stats->spec_sec > 0.0 ? (double)stats->cycles / stats->spec_sec : 0.0;
    const double accepted_tokens_per_sec =
        stats->spec_sec > 0.0 ? (double)stats->generated / stats->spec_sec : 0.0;
    const double avg_accepted_per_cycle =
        stats->cycles > 0 ? (double)stats->generated / (double)stats->cycles : 0.0;
    const double draft_accept_rate =
        stats->possible_extra > 0 ? (double)stats->accepted_extra / (double)stats->possible_extra : 0.0;
    const double rollback_partial_rate =
        stats->spec_opportunity_cycles > 0
            ? (double)(stats->partial_accept_cycles + stats->reject_cycles) /
                (double)stats->spec_opportunity_cycles
            : 0.0;
    const double exact_tps =
        stats->exact_sec > 0.0 ? (double)stats->generated / stats->exact_sec : 0.0;

    fprintf(stderr,
            "ds4-bench: mtp spec frontier=%d draft=%d generated=%d cycles=%d target_cycles_per_sec=%.2f accepted_tokens_per_sec=%.2f avg_accepted_per_cycle=%.3f draft_accept_rate=%.3f rollback_partial_rate=%.3f full_accept_cycles=%d partial_accept_cycles=%d reject_cycles=%d mtp_exact_match=%d first_mismatch=%d exact_tps=%.2f effective_tps=%.2f\n",
            frontier,
            draft_tokens,
            stats->generated,
            stats->cycles,
            target_cycles_per_sec,
            accepted_tokens_per_sec,
            avg_accepted_per_cycle,
            draft_accept_rate,
            rollback_partial_rate,
            stats->full_accept_cycles,
            stats->partial_accept_cycles,
            stats->reject_cycles,
            stats->exact_match,
            stats->first_mismatch,
            exact_tps,
            accepted_tokens_per_sec);
}

int main(int argc, char **argv) {
    bench_config cfg = parse_options(argc, argv);
    log_context_memory(cfg.backend, cfg.ctx_alloc);

    ds4_engine_options opt = {
        .model_path = cfg.model_path,
        .mtp_path = cfg.mtp_path,
        .backend = cfg.backend,
        .n_threads = cfg.threads,
        .mtp_draft_tokens = cfg.mtp_draft_tokens,
        .mtp_margin = cfg.mtp_margin,
        .warm_weights = cfg.warm_weights,
        .quality = cfg.quality,
    };
    ds4_engine *engine = NULL;
    if (ds4_engine_open(&engine, &opt) != 0) return 1;

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
    const bool mtp_bench = cfg.mtp_path && !cfg.exact_replay_probe;
    if (mtp_bench) {
        fprintf(out, "ctx_tokens,prefill_tokens,prefill_tps,gen_tokens,gen_tps,kvcache_bytes,mtp_cycles,mtp_cycles_per_sec,mtp_avg_accepted_per_cycle,mtp_draft_accept_rate,mtp_rollback_partial_rate,mtp_exact_match,mtp_exact_tps\n");
    } else if (cfg.exact_replay_probe) {
        fprintf(out, "ctx_tokens,prefill_tokens,prefill_tps,gen_tokens,replay_tps,exact_tps,kvcache_bytes,exact_replay_match,exact_replay_first_mismatch,exact_replay_runs,exact_replay_matches\n");
    } else {
        fprintf(out, "ctx_tokens,prefill_tokens,prefill_tps,gen_tokens,gen_tps,kvcache_bytes\n");
    }
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

        if (ds4_session_save_snapshot(session, &snap, err, sizeof(err)) != 0) {
            fprintf(stderr, "ds4-bench: snapshot at %d failed: %s\n", frontier, err);
            rc = 1;
            break;
        }

        int *generated = NULL;
        if (cfg.spec_probe || mtp_bench || cfg.exact_replay_probe) {
            generated = malloc((size_t)cfg.gen_tokens * sizeof(generated[0]));
            if (!generated) {
                fprintf(stderr, "ds4-bench: out of memory allocating generation tokens\n");
                rc = 1;
                break;
            }
        }

        int *exact_tokens = NULL;
        ds4_token_score *exact_top2 = NULL;
        ds4_token_score *replay_top2 = NULL;
        float *exact_after_logits = NULL;
        float *replay_after_logits = NULL;
        int exact_replay_match = 1;
        int exact_replay_first_mismatch = -1;
        double exact_replay_sec = 0.0;
        ds4_session *replay_session = NULL;
        mtp_spec_stats mtp_stats = {0};
        if (mtp_bench || cfg.exact_replay_probe) {
            exact_tokens = malloc((size_t)cfg.gen_tokens * sizeof(exact_tokens[0]));
            if (!exact_tokens) {
                fprintf(stderr, "ds4-bench: out of memory allocating exact comparison tokens\n");
                free(generated);
                rc = 1;
                break;
            }
            if (cfg.exact_replay_probe) {
                exact_top2 = malloc((size_t)cfg.gen_tokens * (size_t)cfg.exact_replay_topk * sizeof(exact_top2[0]));
                replay_top2 = malloc((size_t)cfg.gen_tokens * (size_t)cfg.exact_replay_topk * sizeof(replay_top2[0]));
                if (cfg.exact_replay_forced_logit_diff) {
                    exact_after_logits = malloc((size_t)cfg.gen_tokens * (size_t)DS4_BENCH_N_VOCAB * sizeof(exact_after_logits[0]));
                    replay_after_logits = malloc((size_t)DS4_BENCH_N_VOCAB * sizeof(replay_after_logits[0]));
                }
                if (!exact_top2 || !replay_top2 ||
                    (cfg.exact_replay_forced_logit_diff && (!exact_after_logits || !replay_after_logits))) {
                    fprintf(stderr, "ds4-bench: out of memory allocating exact replay top logits\n");
                    free(replay_after_logits);
                    free(exact_after_logits);
                    free(replay_top2);
                    free(exact_top2);
                    free(exact_tokens);
                    free(generated);
                    rc = 1;
                    break;
                }
            }

            const double exact_t0 = bench_now_sec();
            if (cfg.exact_dump_prefix || cfg.replay_dump_prefix) {
                bench_set_dump_prefix(cfg.exact_dump_prefix);
            }
            int nexact = bench_decode_exact_tokens(session,
                                                   eos,
                                                   cfg.gen_tokens,
                                                   exact_tokens,
                                                   exact_top2,
                                                   cfg.exact_replay_topk,
                                                   exact_after_logits,
                                                   DS4_BENCH_N_VOCAB,
                                                   err,
                                                   sizeof(err));
            const double exact_t1 = bench_now_sec();
            mtp_stats.exact_sec = exact_t1 - exact_t0;
            exact_replay_sec = exact_t1 - exact_t0;
            if (nexact != cfg.gen_tokens) {
                fprintf(stderr, "ds4-bench: exact decode at frontier %d failed: %s\n", frontier, err);
                free(replay_after_logits);
                free(exact_after_logits);
                free(replay_top2);
                free(exact_top2);
                free(exact_tokens);
                free(generated);
                rc = 1;
                break;
            }
            if (mtp_bench) {
                replay_session = session;
                if (ds4_session_load_snapshot(session, &snap, err, sizeof(err)) != 0) {
                    fprintf(stderr,
                            "ds4-bench: MTP comparison snapshot restore at %d failed: %s\n",
                            frontier,
                            err);
                    free(replay_after_logits);
                    free(exact_after_logits);
                    free(replay_top2);
                    free(exact_top2);
                    free(exact_tokens);
                    free(generated);
                    rc = 1;
                    break;
                }
            }
        }

        int exact_replay_runs_completed = 0;
        int exact_replay_matches = 0;
        double exact_replay_decode_sec = 0.0;

        const double gen_t0 = bench_now_sec();
        if (mtp_bench) {
            int nspec = bench_decode_mtp_tokens(session,
                                                eos,
                                                cfg.gen_tokens,
                                                cfg.mtp_draft_tokens,
                                                generated,
                                                &mtp_stats,
                                                err,
                                                sizeof(err));
            if (nspec != cfg.gen_tokens) {
                fprintf(stderr, "ds4-bench: MTP decode at frontier %d failed: %s\n", frontier, err);
                rc = 1;
            }
        } else if (cfg.exact_replay_probe) {
            exact_replay_match = 1;
            exact_replay_first_mismatch = -1;
            for (int run = 0; run < cfg.exact_replay_runs; run++) {
                ds4_session *run_session = NULL;
                ds4_session_snapshot roundtrip = {0};
                int snapshot_roundtrip_match = -1;
                if (cfg.exact_replay_snapshot) {
                    if (cfg.exact_replay_snapshot_fresh_session) {
                        if (ds4_session_create(&run_session, engine, cfg.ctx_alloc) != 0) {
                            fprintf(stderr, "ds4-bench: failed to create exact snapshot replay session run=%d\n", run + 1);
                            rc = 1;
                            break;
                        }
                    } else {
                        run_session = session;
                    }
                    if (ds4_session_load_snapshot(run_session, &snap, err, sizeof(err)) != 0) {
                        fprintf(stderr,
                                "ds4-bench: exact replay snapshot restore at frontier %d failed run=%d: %s\n",
                                frontier,
                                run + 1,
                                err);
                        if (run_session && run_session != session) ds4_session_free(run_session);
                        rc = 1;
                        break;
                    }
                    if (cfg.exact_replay_snapshot_roundtrip) {
                        if (ds4_session_save_snapshot(run_session,
                                                      &roundtrip,
                                                      err,
                                                      sizeof(err)) != 0) {
                            fprintf(stderr,
                                    "ds4-bench: exact replay snapshot roundtrip save at frontier %d failed run=%d: %s\n",
                                    frontier,
                                    run + 1,
                                    err);
                            ds4_session_snapshot_free(&roundtrip);
                            if (run_session && run_session != session) ds4_session_free(run_session);
                            rc = 1;
                            break;
                        }
                        snapshot_roundtrip_match =
                            roundtrip.len == snap.len &&
                            memcmp(roundtrip.ptr, snap.ptr, (size_t)snap.len) == 0;
                        fprintf(stderr,
                                "ds4-bench: exact replay snapshot_roundtrip_match frontier=%d run=%d match=%d orig_bytes=%llu roundtrip_bytes=%llu\n",
                                frontier,
                                run + 1,
                                snapshot_roundtrip_match,
                                (unsigned long long)snap.len,
                                (unsigned long long)roundtrip.len);
                        ds4_session_snapshot_free(&roundtrip);
                    }
                } else if (cfg.exact_replay_fresh_session) {
                    if (ds4_session_create(&run_session, engine, cfg.ctx_alloc) != 0) {
                        fprintf(stderr, "ds4-bench: failed to create exact replay session run=%d\n", run + 1);
                        rc = 1;
                        break;
                    }
                } else {
                    ds4_session_invalidate(session);
                    run_session = session;
                }

                if (!cfg.exact_replay_snapshot &&
                    ds4_session_sync(run_session, &prefix, err, sizeof(err)) != 0) {
                    fprintf(stderr,
                            "ds4-bench: exact replay prefill to %d failed run=%d: %s\n",
                            frontier,
                            run + 1,
                            err);
                    if (run_session && run_session != session) ds4_session_free(run_session);
                    rc = 1;
                    break;
                }

                if (cfg.exact_dump_prefix || cfg.replay_dump_prefix) {
                    bench_set_dump_prefix(cfg.replay_dump_prefix);
                }
                const double replay_t0 = bench_now_sec();
                int nreplay = 0;
                float forced_max_abs = 0.0f;
                float forced_max_rms = 0.0f;
                int forced_max_step = -1;
                int forced_max_id = -1;
                int first_drift_step = -1;
                float first_drift_max_abs = 0.0f;
                float first_drift_rms = 0.0f;
                int first_drift_id = -1;
                if (cfg.exact_replay_forced_logit_diff) {
                    for (; nreplay < cfg.gen_tokens; nreplay++) {
                        generated[nreplay] = exact_tokens[nreplay];
                        if (replay_top2) {
                            int n = ds4_session_top_logprobs(run_session,
                                                             &replay_top2[(size_t)nreplay *
                                                                          (size_t)cfg.exact_replay_topk],
                                                             cfg.exact_replay_topk);
                            if (n < cfg.exact_replay_topk) {
                                snprintf(err, sizeof(err), "failed to read top logits");
                                break;
                            }
                        }
                        if (ds4_session_eval_exact(run_session, exact_tokens[nreplay],
                                                   err, sizeof(err)) != 0) {
                            break;
                        }
                        if (ds4_session_copy_logits(run_session,
                                                    replay_after_logits,
                                                    DS4_BENCH_N_VOCAB) == 0) {
                            snprintf(err, sizeof(err), "failed to copy replay logits");
                            break;
                        }
                        float step_max = 0.0f;
                        float step_rms = 0.0f;
                        int step_id = -1;
                        bench_logits_diff(&exact_after_logits[(size_t)nreplay *
                                                              (size_t)DS4_BENCH_N_VOCAB],
                                          replay_after_logits,
                                          DS4_BENCH_N_VOCAB,
                                          &step_max,
                                          &step_rms,
                                          &step_id);
                        if (step_max > forced_max_abs) {
                            forced_max_abs = step_max;
                            forced_max_rms = step_rms;
                            forced_max_step = nreplay;
                            forced_max_id = step_id;
                        }
                        if (first_drift_step < 0 &&
                            step_max > DS4_BENCH_FORCED_LOGIT_DRIFT_EPS) {
                            first_drift_step = nreplay;
                            first_drift_max_abs = step_max;
                            first_drift_rms = step_rms;
                            first_drift_id = step_id;
                        }
                    }
                } else {
                    nreplay = bench_decode_exact_tokens(run_session,
                                                        eos,
                                                        cfg.gen_tokens,
                                                        generated,
                                                        replay_top2,
                                                        cfg.exact_replay_topk,
                                                        NULL,
                                                        0,
                                                        err,
                                                        sizeof(err));
                }
                const double replay_t1 = bench_now_sec();
                exact_replay_decode_sec += replay_t1 - replay_t0;
                exact_replay_runs_completed++;
                if (nreplay != cfg.gen_tokens) {
                    fprintf(stderr,
                            "ds4-bench: exact replay decode at frontier %d failed run=%d: %s\n",
                            frontier,
                            run + 1,
                            err);
                    if (run_session && run_session != session) ds4_session_free(run_session);
                    rc = 1;
                    break;
                }
                if (cfg.exact_replay_forced_logit_diff) {
                    fprintf(stderr,
                            "ds4-bench: exact replay forced_logit_diff frontier=%d run=%d max_abs=%.9f rms=%.9f step=%d token_id=%d first_drift_step=%d first_drift_max_abs=%.9f first_drift_rms=%.9f first_drift_token_id=%d\n",
                            frontier,
                            run + 1,
                            forced_max_abs,
                            forced_max_rms,
                            forced_max_step,
                            forced_max_id,
                            first_drift_step,
                            first_drift_max_abs,
                            first_drift_rms,
                            first_drift_id);
                }

                int run_match = 1;
                int run_first_mismatch = -1;
                for (int i = 0; i < cfg.gen_tokens; i++) {
                    if (generated[i] != exact_tokens[i]) {
                        run_match = 0;
                        run_first_mismatch = i;
                        break;
                    }
                }
                if (run_match) {
                    exact_replay_matches++;
                } else {
                    if (exact_replay_match) {
                        exact_replay_match = 0;
                        exact_replay_first_mismatch = run_first_mismatch;
                    }
                    fprintf(stderr,
                            "ds4-bench: exact replay mismatch at frontier %d run=%d step=%d first=%d replay=%d\n",
                            frontier,
                            run + 1,
                            run_first_mismatch,
                            exact_tokens[run_first_mismatch],
                            generated[run_first_mismatch]);
                    const ds4_token_score *a = &exact_top2[(size_t)run_first_mismatch *
                                                            (size_t)cfg.exact_replay_topk];
                    const ds4_token_score *b = &replay_top2[(size_t)run_first_mismatch *
                                                             (size_t)cfg.exact_replay_topk];
                    fprintf(stderr,
                            "ds4-bench: exact_replay_mismatch_top run=%d step=%d first_top0=%d first_logit0=%.6f first_top1=%d first_logit1=%.6f first_margin=%.6f replay_top0=%d replay_logit0=%.6f replay_top1=%d replay_logit1=%.6f replay_margin=%.6f\n",
                            run + 1,
                            run_first_mismatch,
                            a[0].id,
                            a[0].logit,
                            a[1].id,
                            a[1].logit,
                            a[0].logit - a[1].logit,
                            b[0].id,
                            b[0].logit,
                            b[1].id,
                            b[1].logit,
                            b[0].logit - b[1].logit);
                    const int first_token_replay_rank =
                        bench_score_rank(b, cfg.exact_replay_topk, exact_tokens[run_first_mismatch]);
                    const int replay_token_first_rank =
                        bench_score_rank(a, cfg.exact_replay_topk, generated[run_first_mismatch]);
                    const float first_token_replay_logit =
                        bench_score_logit_or_nan(b, cfg.exact_replay_topk, exact_tokens[run_first_mismatch]);
                    const float replay_token_first_logit =
                        bench_score_logit_or_nan(a, cfg.exact_replay_topk, generated[run_first_mismatch]);
                    fprintf(stderr,
                            "ds4-bench: exact_replay_mismatch_overlap run=%d step=%d topk=%d first_token_replay_rank=%d first_token_replay_logit=%.6f replay_token_first_rank=%d replay_token_first_logit=%.6f\n",
                            run + 1,
                            run_first_mismatch,
                            cfg.exact_replay_topk,
                            first_token_replay_rank,
                            first_token_replay_logit,
                            replay_token_first_rank,
                            replay_token_first_logit);
                }

                fprintf(stderr,
                        "ds4-bench: exact replay frontier=%d generated=%d run=%d/%d exact_replay_match=%d first_mismatch=%d replay_tps=%.2f\n",
                        frontier,
                        cfg.gen_tokens,
                        run + 1,
                        cfg.exact_replay_runs,
                        run_match,
                        run_first_mismatch,
                        replay_t1 > replay_t0 ? (double)cfg.gen_tokens / (replay_t1 - replay_t0) : 0.0);

                if (run_session && run_session != session) {
                    ds4_session_free(run_session);
                }
            }
        } else {
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
                if (generated) generated[i] = token;
                if (ds4_session_eval(session, token, err, sizeof(err)) != 0) {
                    fprintf(stderr, "ds4-bench: decode at frontier %d failed: %s\n", frontier, err);
                    rc = 1;
                    break;
                }
            }
        }
        const double gen_t1 = bench_now_sec();
        mtp_stats.spec_sec = cfg.exact_replay_probe ? exact_replay_decode_sec : gen_t1 - gen_t0;
        if (rc != 0) {
            if (replay_session && replay_session != session) ds4_session_free(replay_session);
            free(replay_after_logits);
            free(exact_after_logits);
            free(replay_top2);
            free(exact_top2);
            free(exact_tokens);
            free(generated);
            break;
        }

        if (mtp_bench) {
            mtp_stats.exact_match = 1;
            mtp_stats.first_mismatch = -1;
            for (int i = 0; i < cfg.gen_tokens; i++) {
                if (generated[i] != exact_tokens[i]) {
                    mtp_stats.exact_match = 0;
                    mtp_stats.first_mismatch = i;
                    break;
                }
            }
            mtp_spec_report(frontier, cfg.mtp_draft_tokens, &mtp_stats);
            if (!mtp_stats.exact_match) {
                fprintf(stderr,
                        "ds4-bench: MTP token mismatch at frontier %d step %d exact=%d mtp=%d\n",
                        frontier,
                        mtp_stats.first_mismatch,
                        exact_tokens[mtp_stats.first_mismatch],
                        generated[mtp_stats.first_mismatch]);
                free(replay_after_logits);
                free(exact_after_logits);
                free(exact_tokens);
                free(generated);
                rc = 1;
                break;
            }
        }

        if (cfg.exact_replay_probe) {
            if (exact_replay_runs_completed != cfg.exact_replay_runs ||
                exact_replay_matches != cfg.exact_replay_runs) {
                exact_replay_match = 0;
            }
            const double replay_tps = exact_replay_decode_sec > 0.0
                ? (double)cfg.gen_tokens * (double)exact_replay_runs_completed / exact_replay_decode_sec
                : 0.0;
            const double exact_tps = exact_replay_sec > 0.0
                ? (double)cfg.gen_tokens / exact_replay_sec
                : 0.0;
            fprintf(stderr,
                    "ds4-bench: exact replay summary frontier=%d generated=%d runs=%d matches=%d exact_replay_match=%d first_mismatch=%d exact_tps=%.2f replay_tps=%.2f\n",
                    frontier,
                    cfg.gen_tokens,
                    exact_replay_runs_completed,
                    exact_replay_matches,
                    exact_replay_match,
                    exact_replay_first_mismatch,
                    exact_tps,
                    replay_tps);
        }

        if (mtp_bench || cfg.exact_replay_probe) {
            if (replay_session && replay_session != session) {
                ds4_session_free(replay_session);
            } else {
                ds4_session_invalidate(session);
            }
        } else {
            if (ds4_session_load_snapshot(session, &snap, err, sizeof(err)) != 0) {
                fprintf(stderr, "ds4-bench: restore at %d failed: %s\n", frontier, err);
                free(replay_after_logits);
                free(exact_after_logits);
                free(replay_top2);
                free(exact_top2);
                free(exact_tokens);
                free(generated);
                rc = 1;
                break;
            }
        }

        const double gen_sec = cfg.exact_replay_probe ? exact_replay_decode_sec : gen_t1 - gen_t0;
        const double measured_gen_tokens = cfg.exact_replay_probe
            ? (double)cfg.gen_tokens * (double)exact_replay_runs_completed
            : (double)cfg.gen_tokens;
        const double gen_tps = gen_sec > 0.0 ? measured_gen_tokens / gen_sec : 0.0;
        if (cfg.spec_probe && generated) {
            spec_probe_report(frontier,
                              &prefix,
                              generated,
                              cfg.gen_tokens,
                              cfg.spec_probe_ngram,
                              cfg.spec_probe_draft,
                              gen_tps);
        }
        free(replay_after_logits);
        free(exact_after_logits);
        free(replay_top2);
        free(exact_top2);
        free(generated);
        if (mtp_bench) {
            const double target_cycles_per_sec =
                mtp_stats.spec_sec > 0.0 ? (double)mtp_stats.cycles / mtp_stats.spec_sec : 0.0;
            const double avg_accepted_per_cycle =
                mtp_stats.cycles > 0 ? (double)mtp_stats.generated / (double)mtp_stats.cycles : 0.0;
            const double draft_accept_rate =
                mtp_stats.possible_extra > 0 ? (double)mtp_stats.accepted_extra / (double)mtp_stats.possible_extra : 0.0;
            const double rollback_partial_rate =
                mtp_stats.spec_opportunity_cycles > 0
                    ? (double)(mtp_stats.partial_accept_cycles + mtp_stats.reject_cycles) /
                        (double)mtp_stats.spec_opportunity_cycles
                    : 0.0;
            const double exact_tps =
                mtp_stats.exact_sec > 0.0 ? (double)cfg.gen_tokens / mtp_stats.exact_sec : 0.0;
            fprintf(out,
                    "%d,%d,%.2f,%d,%.2f,%llu,%d,%.2f,%.3f,%.3f,%.3f,%d,%.2f\n",
                    frontier,
                    prefill_tokens,
                    prefill_sec > 0.0 ? (double)prefill_tokens / prefill_sec : 0.0,
                    cfg.gen_tokens,
                    gen_tps,
                    (unsigned long long)snap.len,
                    mtp_stats.cycles,
                    target_cycles_per_sec,
                    avg_accepted_per_cycle,
                    draft_accept_rate,
                    rollback_partial_rate,
                    mtp_stats.exact_match,
                    exact_tps);
        } else if (cfg.exact_replay_probe) {
            const double exact_tps =
                exact_replay_sec > 0.0 ? (double)cfg.gen_tokens / exact_replay_sec : 0.0;
            fprintf(out,
                    "%d,%d,%.2f,%d,%.2f,%.2f,%llu,%d,%d,%d,%d\n",
                    frontier,
                    prefill_tokens,
                    prefill_sec > 0.0 ? (double)prefill_tokens / prefill_sec : 0.0,
                    cfg.gen_tokens,
                    gen_tps,
                    exact_tps,
                    (unsigned long long)snap.len,
                    exact_replay_match,
                    exact_replay_first_mismatch,
                    exact_replay_runs_completed,
                    exact_replay_matches);
        } else {
            fprintf(out,
                    "%d,%d,%.2f,%d,%.2f,%llu\n",
                    frontier,
                    prefill_tokens,
                    prefill_sec > 0.0 ? (double)prefill_tokens / prefill_sec : 0.0,
                    cfg.gen_tokens,
                    gen_tps,
                    (unsigned long long)snap.len);
        }
        fflush(out);

        free(exact_tokens);
        previous = frontier;
        if (cfg.exact_replay_probe && !exact_replay_match) {
            rc = 1;
            break;
        }
        if (frontier >= cfg.ctx_max) break;
    }

    if (out != stdout) fclose(out);
    ds4_session_snapshot_free(&snap);
    ds4_session_free(session);
    ds4_tokens_free(&prompt);
    ds4_engine_close(engine);
    return rc;
}
