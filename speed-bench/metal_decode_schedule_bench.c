#include "ds4.h"

#include <errno.h>
#include <limits.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define FIRST_SPLIT_ENV "DS4_METAL_GRAPH_TOKEN_SPLIT_LAYERS"
#define SECOND_SPLIT_ENV "DS4_METAL_GRAPH_TOKEN_SECOND_SPLIT_LAYERS"

enum {
    VARIANT_COUNT = 2,
    DEFAULT_PREFIX_TOKENS = 2048,
    DEFAULT_CTX = 4096,
    DEFAULT_WARMUP = 16,
    DEFAULT_MEASURED = 512,
};

typedef struct {
    int first;
    int second;
} decode_schedule;

typedef struct {
    const char *model_path;
    const char *prompt_path;
    const char *candidate_env;
    int prefix_tokens;
    int ctx;
    int warmup;
    int measured;
    bool include_selection;
    decode_schedule control;
    decode_schedule candidate;
} bench_config;

static void usage(FILE *fp, const char *argv0) {
    fprintf(fp,
            "usage: %s [options]\n"
            "\n"
            "  -m, --model PATH       GGUF path (default: ds4flash.gguf)\n"
            "  --prompt-file PATH     token source (default: ds4.c)\n"
            "  --prefix-tokens N      prefill length (default: 2048)\n"
            "  --ctx N                session allocation (default: 4096)\n"
            "  --warmup N             untimed steps per variant (default: 16)\n"
            "  --tokens N             measured steps per variant (default: 512)\n"
            "  --control-first N      control first split (default: 2)\n"
            "  --control-second N     control second split (default: 32)\n"
            "  --candidate-first N    candidate first split (default: 1; control with --candidate-env)\n"
            "  --candidate-second N   candidate second split (default: 32; control with --candidate-env)\n"
            "  --candidate-env NAME   unset NAME for control, set NAME=1 for candidate\n"
            "  --include-selection    include one non-EOS argmax in each timed step\n",
            argv0);
}

static const char *need_arg(int *i, int argc, char **argv, const char *opt) {
    if (*i + 1 >= argc) {
        fprintf(stderr,
                "metal-decode-schedule-bench: %s requires an argument\n",
                opt);
        exit(2);
    }
    return argv[++*i];
}

static int parse_int_arg(const char *value, const char *opt, int minimum) {
    char *end = NULL;
    errno = 0;
    long parsed = strtol(value, &end, 10);
    if (errno != 0 || value[0] == '\0' || !end || *end != '\0' ||
        parsed < minimum || parsed > INT_MAX) {
        fprintf(stderr,
                "metal-decode-schedule-bench: invalid value for %s: %s\n",
                opt,
                value);
        exit(2);
    }
    return (int)parsed;
}

static bench_config parse_options(int argc, char **argv) {
    bench_config cfg = {
        .model_path = "ds4flash.gguf",
        .prompt_path = "ds4.c",
        .candidate_env = NULL,
        .prefix_tokens = DEFAULT_PREFIX_TOKENS,
        .ctx = DEFAULT_CTX,
        .warmup = DEFAULT_WARMUP,
        .measured = DEFAULT_MEASURED,
        .include_selection = false,
        .control = {.first = 2, .second = 32},
        .candidate = {.first = 1, .second = 32},
    };
    bool candidate_first_explicit = false;
    bool candidate_second_explicit = false;

    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        if (!strcmp(arg, "-h") || !strcmp(arg, "--help")) {
            usage(stdout, argv[0]);
            exit(0);
        } else if (!strcmp(arg, "-m") || !strcmp(arg, "--model")) {
            cfg.model_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--prompt-file")) {
            cfg.prompt_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--candidate-env")) {
            cfg.candidate_env = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--include-selection")) {
            cfg.include_selection = true;
        } else if (!strcmp(arg, "--prefix-tokens")) {
            cfg.prefix_tokens =
                parse_int_arg(need_arg(&i, argc, argv, arg), arg, 1);
        } else if (!strcmp(arg, "--ctx")) {
            cfg.ctx = parse_int_arg(need_arg(&i, argc, argv, arg), arg, 1);
        } else if (!strcmp(arg, "--warmup")) {
            cfg.warmup =
                parse_int_arg(need_arg(&i, argc, argv, arg), arg, 0);
        } else if (!strcmp(arg, "--tokens") || !strcmp(arg, "--measured")) {
            cfg.measured =
                parse_int_arg(need_arg(&i, argc, argv, arg), arg, 1);
        } else if (!strcmp(arg, "--control-first")) {
            cfg.control.first =
                parse_int_arg(need_arg(&i, argc, argv, arg), arg, 0);
        } else if (!strcmp(arg, "--control-second")) {
            cfg.control.second =
                parse_int_arg(need_arg(&i, argc, argv, arg), arg, 0);
        } else if (!strcmp(arg, "--candidate-first")) {
            cfg.candidate.first =
                parse_int_arg(need_arg(&i, argc, argv, arg), arg, 0);
            candidate_first_explicit = true;
        } else if (!strcmp(arg, "--candidate-second")) {
            cfg.candidate.second =
                parse_int_arg(need_arg(&i, argc, argv, arg), arg, 0);
            candidate_second_explicit = true;
        } else {
            fprintf(stderr,
                    "metal-decode-schedule-bench: unknown option: %s\n",
                    arg);
            usage(stderr, argv[0]);
            exit(2);
        }
    }

    /*
     * An environment-gated comparison isolates that feature by default.
     * Explicit candidate split arguments can still combine a feature and
     * schedule experiment when desired.
     */
    if (cfg.candidate_env) {
        if (!candidate_first_explicit) {
            cfg.candidate.first = cfg.control.first;
        }
        if (!candidate_second_explicit) {
            cfg.candidate.second = cfg.control.second;
        }
    }

    const int64_t needed =
        (int64_t)cfg.prefix_tokens + cfg.warmup + cfg.measured + 1;
    if (needed > cfg.ctx) {
        fprintf(stderr,
                "metal-decode-schedule-bench: --ctx must exceed "
                "prefix + warmup + measured (minimum %lld)\n",
                (long long)needed);
        exit(2);
    }
    return cfg;
}

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1.0e9;
}

static char *read_text(const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr,
                "metal-decode-schedule-bench: failed to open %s: %s\n",
                path,
                strerror(errno));
        return NULL;
    }
    if (fseek(fp, 0, SEEK_END) != 0) {
        fprintf(stderr,
                "metal-decode-schedule-bench: failed to seek %s\n",
                path);
        fclose(fp);
        return NULL;
    }
    long len = ftell(fp);
    if (len < 0 || fseek(fp, 0, SEEK_SET) != 0) {
        fprintf(stderr,
                "metal-decode-schedule-bench: failed to size %s\n",
                path);
        fclose(fp);
        return NULL;
    }
    char *text = malloc((size_t)len + 1u);
    if (!text) {
        fprintf(stderr,
                "metal-decode-schedule-bench: out of memory reading %s\n",
                path);
        fclose(fp);
        return NULL;
    }
    if (fread(text, 1, (size_t)len, fp) != (size_t)len) {
        fprintf(stderr,
                "metal-decode-schedule-bench: failed to read %s\n",
                path);
        free(text);
        fclose(fp);
        return NULL;
    }
    text[len] = '\0';
    fclose(fp);
    return text;
}

static uint32_t float_bits(float value) {
    uint32_t bits = 0;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static int select_variant(const bench_config *cfg, int variant) {
    const decode_schedule *schedule =
        variant == 0 ? &cfg->control : &cfg->candidate;
    char first[16];
    char second[16];
    snprintf(first, sizeof(first), "%d", schedule->first);
    snprintf(second, sizeof(second), "%d", schedule->second);
    if (setenv(FIRST_SPLIT_ENV, first, 1) != 0 ||
        setenv(SECOND_SPLIT_ENV, second, 1) != 0) {
        fprintf(stderr,
                "metal-decode-schedule-bench: failed to select "
                "%s schedule %d/%d: %s\n",
                variant == 0 ? "control" : "candidate",
                schedule->first,
                schedule->second,
                strerror(errno));
        return 1;
    }
    if (cfg->candidate_env &&
        (variant == 0
             ? unsetenv(cfg->candidate_env)
             : setenv(cfg->candidate_env, "1", 1)) != 0) {
        fprintf(stderr,
                "metal-decode-schedule-bench: failed to select candidate "
                "environment %s for %s: %s\n",
                cfg->candidate_env,
                variant == 0 ? "control" : "candidate",
                strerror(errno));
        return 1;
    }
    return 0;
}

/* Unset leftover output-geometry experiment knobs so A/B runs start clean.
 * Keep only names that still have production getenv sites (or their M3
 * rollback aliases). Dead allowlist-only names from abandoned NR8/NR4X2 /
 * EXACT_VIEW / MAX_BYTES probes were removed after the mxfp4-m3 env audit. */
static int normalize_output_experiment_env(void) {
    static const char *const conflicting[] = {
        "DS4_METAL_ENABLE_OUTPUT_Q8_NR4",
        "DS4_METAL_DISABLE_M3_OUTPUT_Q8_NR4",
        "DS4_METAL_ENABLE_Q8_DECODE_EXACT_VIEWS",
        "DS4_METAL_DISABLE_Q8_DECODE_EXACT_VIEWS",
        "DS4_METAL_Q8_DECODE_EXACT_VIEW_MAX_MIB",
        "DS4_METAL_Q8_MV_ROWS",
        "DS4_METAL_ENABLE_OUTPUT_HC_SUM_NORM_FUSION",
        "DS4_METAL_DISABLE_M3_OUTPUT_HC_SUM_NORM_FUSION",
        "DS4_METAL_REQUIRE_OUTPUT_HC_SUM_NORM_FUSION",
        "DS4_METAL_ENABLE_OUTPUT_HC_WEIGHTS4",
        "DS4_METAL_DISABLE_M3_OUTPUT_HC_WEIGHTS4",
        "DS4_METAL_REQUIRE_OUTPUT_HC_WEIGHTS4",
    };
    for (size_t i = 0; i < sizeof(conflicting) / sizeof(conflicting[0]); i++) {
        if (unsetenv(conflicting[i]) != 0) {
            fprintf(stderr,
                    "metal-decode-schedule-bench: failed to unset %s: %s\n",
                    conflicting[i],
                    strerror(errno));
            return 1;
        }
    }
    return 0;
}

static int compare_frontier(
        ds4_session *session0,
        ds4_session *session1,
        float       *logits0,
        float       *logits1,
        int          vocab,
        int          eos,
        size_t       frontier,
        int         *token_out) {
    const size_t row_bytes = (size_t)vocab * sizeof(logits0[0]);
    const int pos0 = ds4_session_pos(session0);
    const int pos1 = ds4_session_pos(session1);
    if (pos0 != pos1) {
        fprintf(stderr,
                "metal-decode-schedule-bench: position mismatch at frontier %zu: "
                "session0=%d session1=%d\n",
                frontier,
                pos0,
                pos1);
        return 1;
    }

    memset(logits0, 0xa5, row_bytes);
    memset(logits1, 0x5a, row_bytes);
    if (ds4_session_copy_logits(session0, logits0, vocab) != vocab ||
        ds4_session_copy_logits(session1, logits1, vocab) != vocab) {
        fprintf(stderr,
                "metal-decode-schedule-bench: failed to copy logits at frontier %zu\n",
                frontier);
        return 1;
    }

    if (memcmp(logits0, logits1, row_bytes) != 0) {
        size_t first = SIZE_MAX;
        size_t differing = 0;
        for (int i = 0; i < vocab; i++) {
            if (memcmp(&logits0[i], &logits1[i], sizeof(float)) != 0) {
                if (first == SIZE_MAX) first = (size_t)i;
                differing++;
            }
        }
        const int top0 = ds4_session_argmax(session0);
        const int top1 = ds4_session_argmax(session1);
        fprintf(stderr,
                "metal-decode-schedule-bench: raw logit mismatch at frontier %zu: "
                "differing=%zu/%d top0=%d top1=%d\n",
                frontier,
                differing,
                vocab,
                top0,
                top1);
        if (first != SIZE_MAX) {
            fprintf(stderr,
                    "metal-decode-schedule-bench: first mismatch id=%zu "
                    "session0=%a (0x%08x) session1=%a (0x%08x)\n",
                    first,
                    logits0[first],
                    (unsigned)float_bits(logits0[first]),
                    logits1[first],
                    (unsigned)float_bits(logits1[first]));
        }
        return 1;
    }

    const int token0 = ds4_session_argmax_excluding(session0, eos);
    const int token1 = ds4_session_argmax_excluding(session1, eos);
    if (token0 < 0 || token1 < 0 || token0 != token1) {
        fprintf(stderr,
                "metal-decode-schedule-bench: non-EOS token mismatch at "
                "frontier %zu: session0=%d session1=%d\n",
                frontier,
                token0,
                token1);
        return 1;
    }
    if (token_out) *token_out = token0;
    return 0;
}

int main(int argc, char **argv) {
    const bench_config cfg = parse_options(argc, argv);
    char *text = read_text(cfg.prompt_path);
    if (!text) return 1;
    if (normalize_output_experiment_env() != 0 ||
        select_variant(&cfg, 0) != 0) {
        free(text);
        return 1;
    }

    ds4_engine_options opt = {
        .model_path = cfg.model_path,
        .backend = DS4_BACKEND_METAL,
        .context_size = cfg.ctx,
        .prefill_chunk = 4096,
        .power_percent = 100,
        .warm_weights = true,
    };
    ds4_engine *engine = NULL;
    ds4_session *sessions[VARIANT_COUNT] = {0};
    ds4_tokens tokens = {0};
    float *logits[VARIANT_COUNT] = {0};
    char err[256] = {0};
    double elapsed[VARIANT_COUNT] = {0};
    size_t measured_tokens[VARIANT_COUNT] = {0};
    size_t exact_rows = 0;
    size_t exact_floats = 0;
    size_t exact_selected_ids = 0;
    int rc = 1;

    if (ds4_engine_open(&engine, &opt) != 0) goto done;
    ds4_tokenize_text(engine, text, &tokens);
    free(text);
    text = NULL;
    if (tokens.len < cfg.prefix_tokens) {
        fprintf(stderr,
                "metal-decode-schedule-bench: prompt has %d tokens; need %d\n",
                tokens.len,
                cfg.prefix_tokens);
        goto done;
    }

    const int layers = ds4_engine_layer_count(engine);
    if (cfg.control.first > layers || cfg.control.second > layers ||
        cfg.candidate.first > layers || cfg.candidate.second > layers) {
        fprintf(stderr,
                "metal-decode-schedule-bench: split layers must be between "
                "0 and %d (control=%d/%d candidate=%d/%d)\n",
                layers,
                cfg.control.first,
                cfg.control.second,
                cfg.candidate.first,
                cfg.candidate.second);
        goto done;
    }
    const int vocab = ds4_engine_vocab_size(engine);

    ds4_tokens prefix = {
        .v = tokens.v,
        .len = cfg.prefix_tokens,
        .cap = cfg.prefix_tokens,
    };
    for (int i = 0; i < VARIANT_COUNT; i++) {
        if (ds4_session_create(&sessions[i], engine, cfg.ctx) != 0 ||
            ds4_session_sync(sessions[i], &prefix, err, sizeof(err)) != 0) {
            fprintf(stderr,
                    "metal-decode-schedule-bench: session %d prefill failed: %s\n",
                    i,
                    err[0] ? err : "unknown error");
            goto done;
        }
        logits[i] = malloc((size_t)vocab * sizeof(logits[i][0]));
        if (!logits[i]) {
            fprintf(stderr,
                    "metal-decode-schedule-bench: logit buffer allocation failed\n");
            goto done;
        }
    }

    fprintf(stderr,
            "metal-decode-schedule-bench: model=%s prompt=%s prefix=%d "
            "ctx=%d warmup=%d measured=%d control=%d/%d candidate=%d/%d "
            "candidate_env=%s include_selection=%s\n",
            cfg.model_path,
            cfg.prompt_path,
            cfg.prefix_tokens,
            cfg.ctx,
            cfg.warmup,
            cfg.measured,
            cfg.control.first,
            cfg.control.second,
            cfg.candidate.first,
            cfg.candidate.second,
            cfg.candidate_env ? cfg.candidate_env : "(none)",
            cfg.include_selection ? "yes" : "no");

    const int eos = ds4_token_eos(engine);
    const int total_steps = cfg.warmup + cfg.measured;
    for (int step = 0; step < total_steps; step++) {
        int token = -1;
        int selected[VARIANT_COUNT] = {-1, -1};
        if (compare_frontier(sessions[0],
                             sessions[1],
                             logits[0],
                             logits[1],
                             vocab,
                             eos,
                             (size_t)step,
                             &token) != 0) {
            goto done;
        }
        exact_rows++;
        exact_floats += (size_t)vocab;

        /*
         * Even steps evaluate the control on session 0, then the candidate on
         * session 1. Odd steps reverse that pairing and candidate order.
         * This alternates both candidate order and candidate/session pairing.
         */
        for (int order = 0; order < VARIANT_COUNT; order++) {
            const int session_i = order;
            const int variant_i = (order + step) & 1;
            if (select_variant(&cfg, variant_i) != 0) goto done;

            const double t0 = now_sec();
            if (ds4_session_eval(sessions[session_i],
                                 token,
                                 err,
                                 sizeof(err)) != 0) {
                fprintf(stderr,
                        "metal-decode-schedule-bench: decode failed at step=%d "
                        "variant=%s schedule=%d/%d session=%d: %s\n",
                        step,
                        variant_i == 0 ? "control" : "candidate",
                        variant_i == 0
                            ? cfg.control.first
                            : cfg.candidate.first,
                        variant_i == 0
                            ? cfg.control.second
                            : cfg.candidate.second,
                        session_i,
                        err[0] ? err : "unknown error");
                goto done;
            }
            if (cfg.include_selection) {
                selected[variant_i] =
                    ds4_session_argmax_excluding(sessions[session_i], eos);
                if (selected[variant_i] < 0) {
                    fprintf(stderr,
                            "metal-decode-schedule-bench: timed selection failed "
                            "at step=%d variant=%s session=%d\n",
                            step,
                            variant_i == 0 ? "control" : "candidate",
                            session_i);
                    goto done;
                }
            }
            const double t1 = now_sec();
            if (step >= cfg.warmup) {
                elapsed[variant_i] += t1 - t0;
                measured_tokens[variant_i]++;
            }
        }
        if (cfg.include_selection) {
            if (selected[0] != selected[1]) {
                fprintf(stderr,
                        "metal-decode-schedule-bench: timed selection mismatch "
                        "at step=%d control=%d candidate=%d\n",
                        step,
                        selected[0],
                        selected[1]);
                goto done;
            }
            exact_selected_ids++;
        }
    }

    {
        int final_token = -1;
        if (compare_frontier(sessions[0],
                             sessions[1],
                             logits[0],
                             logits[1],
                             vocab,
                             eos,
                             (size_t)total_steps,
                             &final_token) != 0) {
            goto done;
        }
        exact_rows++;
        exact_floats += (size_t)vocab;
    }

    for (int variant = 0; variant < VARIANT_COUNT; variant++) {
        const decode_schedule *schedule =
            variant == 0 ? &cfg.control : &cfg.candidate;
        printf("variant=%s first_split=%d second_split=%d tokens=%zu "
               "seconds=%.6f tokens_per_second=%.4f\n",
               variant == 0 ? "control" : "candidate",
               schedule->first,
               schedule->second,
               measured_tokens[variant],
               elapsed[variant],
               elapsed[variant] > 0.0
                   ? (double)measured_tokens[variant] / elapsed[variant]
                   : 0.0);
    }
    printf("exact_rows=%zu exact_floats=%zu exact_selected_ids=%zu vocab=%d\n",
           exact_rows,
           exact_floats,
           exact_selected_ids,
           vocab);
    rc = 0;

done:
    free(text);
    for (int i = 0; i < VARIANT_COUNT; i++) {
        free(logits[i]);
        if (sessions[i]) ds4_session_free(sessions[i]);
    }
    ds4_tokens_free(&tokens);
    if (engine) ds4_engine_close(engine);
    return rc;
}
