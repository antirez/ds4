#include "ds4.h"

#include <errno.h>
#include <limits.h>
#include <stdbool.h>
#include <stdint.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define BENCH_NAME "metal-prefill-variant-bench"

enum {
    VARIANT_COUNT = 2,
    RUNS_PER_REPEAT = 4,
    DEFAULT_PREFIX_TOKENS = 8192,
    DEFAULT_WARMUP_TOKENS = 32,
    DEFAULT_REPEATS = 2,
};

typedef struct {
    const char *model_path;
    const char *prompt_path;
    const char *candidate_env;
    const char *candidate_value;
    const char *control_value;
    const char *extra_env[16];    /* candidate-only NAME=VALUE settings */
    const char *control_env[16];  /* control-only NAME=VALUE settings */
    int n_extra;
    int n_control;
    bool interleave;
    int prefill_chunk;
    int prefix_tokens;
    int initial_tokens;
    bool tolerate_drift;     /* report drift statistics instead of failing on a logit mismatch */
    int warmup_tokens;
    int ctx;
    int repeats;
} bench_config;

typedef struct {
    double seconds;
    size_t runs;
    size_t tokens;
} variant_result;

static void usage(FILE *fp, const char *argv0) {
    fprintf(fp,
            "usage: %s --candidate-env NAME [options]\n"
            "\n"
            "  -m, --model PATH       GGUF path (default: ds4flash.gguf)\n"
            "  --prompt-file PATH     token source (default: ds4.c)\n"
            "  --candidate-env NAME   unset NAME for control, set it for candidate\n"
            "  --candidate-value TEXT candidate env value (default: 1)\n"
            "  --control-value TEXT   explicit control env value (default: unset)\n"
            "  --extra-env NAME=VALUE additional candidate-only setting; repeatable\n"
            "  --control-env NAME=VALUE control-only setting; repeatable\n"
            "  --prefill-chunk N      tokens per chunk (default: 4096)\n"
            "  --prefix-tokens N      final prefill length (default: 8192)\n"
            "  --initial-tokens N     untimed live prefix before appending to that length\n"
            "  --tolerate-drift       report max/mean |delta|, top-1 agreement instead of failing on a mismatch\n"
            "  --warmup-tokens N      untimed tokens per variant (default: 32; min: 32)\n"
            "  --ctx N                session allocation (default: max lengths + 1)\n"
            "  --repeats N            alternating ABBA/BAAB pairs (default: 2)\n"
            "  --interleave           one session per repeat, prefilled chunk by chunk with the\n"
            "                         variant alternating per chunk (and per repeat), so both\n"
            "                         variants share the machine state and every position\n",
            argv0);
}

static const char *need_arg(int *i, int argc, char **argv, const char *opt) {
    if (*i + 1 >= argc) {
        fprintf(stderr, "%s: %s requires an argument\n", BENCH_NAME, opt);
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
                "%s: invalid value for %s: %s\n",
                BENCH_NAME,
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
        .candidate_value = "1",
        .prefill_chunk = 4096,
        .prefix_tokens = DEFAULT_PREFIX_TOKENS,
        .warmup_tokens = DEFAULT_WARMUP_TOKENS,
        .ctx = 0,
        .repeats = DEFAULT_REPEATS,
    };

    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        if (!strcmp(arg, "-h") || !strcmp(arg, "--help")) {
            usage(stdout, argv[0]);
            exit(0);
        } else if (!strcmp(arg, "-m") || !strcmp(arg, "--model")) {
            cfg.model_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--prompt-file")) {
            cfg.prompt_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--control-value")) {
            cfg.control_value = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--candidate-value")) {
            cfg.candidate_value = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--prefill-chunk")) {
            cfg.prefill_chunk = parse_int_arg(need_arg(&i, argc, argv, arg), arg, 1);
        } else if (!strcmp(arg, "--candidate-env")) {
            cfg.candidate_env = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--extra-env") || !strcmp(arg, "--control-env")) {
            const char *spec = need_arg(&i, argc, argv, arg);
            const char **list = arg[2] == 'e' ? cfg.extra_env : cfg.control_env;
            int *n = arg[2] == 'e' ? &cfg.n_extra : &cfg.n_control;
            if (!strchr(spec, '=') || spec[0] == '=' || *n >= 16) {
                fprintf(stderr, "%s: %s needs NAME=VALUE (at most 16)\n", BENCH_NAME, arg);
                exit(2);
            }
            list[(*n)++] = spec;
        } else if (!strcmp(arg, "--interleave")) {
            cfg.interleave = true;
        } else if (!strcmp(arg, "--prefix-tokens")) {
            cfg.prefix_tokens =
                parse_int_arg(need_arg(&i, argc, argv, arg), arg, 1);
        } else if (!strcmp(arg, "--warmup-tokens")) {
            cfg.warmup_tokens = parse_int_arg(
                need_arg(&i, argc, argv, arg), arg, DEFAULT_WARMUP_TOKENS);
        } else if (!strcmp(arg, "--tolerate-drift")) {
            cfg.tolerate_drift = true;
        } else if (!strcmp(arg, "--initial-tokens")) {
            cfg.initial_tokens = parse_int_arg(need_arg(&i, argc, argv, arg), arg, 0);
        } else if (!strcmp(arg, "--ctx")) {
            cfg.ctx = parse_int_arg(need_arg(&i, argc, argv, arg), arg, 2);
        } else if (!strcmp(arg, "--repeats")) {
            cfg.repeats =
                parse_int_arg(need_arg(&i, argc, argv, arg), arg, 1);
        } else {
            fprintf(stderr, "%s: unknown option: %s\n", BENCH_NAME, arg);
            usage(stderr, argv[0]);
            exit(2);
        }
    }

    if (!cfg.candidate_env || cfg.candidate_env[0] == '\0' ||
        strchr(cfg.candidate_env, '=') != NULL) {
        fprintf(stderr, "%s: --candidate-env requires a valid name\n", BENCH_NAME);
        exit(2);
    }
    if (cfg.initial_tokens >= cfg.prefix_tokens) {
        fprintf(stderr, "%s: --initial-tokens must be less than --prefix-tokens\n", BENCH_NAME);
        exit(2);
    }
    const int longest =
        cfg.prefix_tokens > cfg.warmup_tokens
            ? cfg.prefix_tokens
            : cfg.warmup_tokens;
    if (longest == INT_MAX) {
        fprintf(stderr, "%s: requested token length is too large\n", BENCH_NAME);
        exit(2);
    }
    if (cfg.ctx == 0) cfg.ctx = longest + 1;
    if (cfg.ctx <= longest) {
        fprintf(stderr,
                "%s: --ctx must exceed prefix and warmup lengths "
                "(minimum %d)\n",
                BENCH_NAME,
                longest + 1);
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
                "%s: failed to open %s: %s\n",
                BENCH_NAME,
                path,
                strerror(errno));
        return NULL;
    }
    if (fseek(fp, 0, SEEK_END) != 0) {
        fprintf(stderr, "%s: failed to seek %s\n", BENCH_NAME, path);
        fclose(fp);
        return NULL;
    }
    const long len = ftell(fp);
    if (len < 0 || fseek(fp, 0, SEEK_SET) != 0) {
        fprintf(stderr, "%s: failed to size %s\n", BENCH_NAME, path);
        fclose(fp);
        return NULL;
    }
    char *text = malloc((size_t)len + 1u);
    if (!text) {
        fprintf(stderr, "%s: out of memory reading %s\n", BENCH_NAME, path);
        fclose(fp);
        return NULL;
    }
    if (fread(text, 1, (size_t)len, fp) != (size_t)len) {
        fprintf(stderr, "%s: failed to read %s\n", BENCH_NAME, path);
        free(text);
        fclose(fp);
        return NULL;
    }
    text[len] = '\0';
    fclose(fp);
    return text;
}

static int apply_env_list(const char *const *list, int n, bool set) {
    for (int i = 0; i < n; i++) {
        const char *eq = strchr(list[i], '=');
        char name[128];
        const size_t len = (size_t)(eq - list[i]);
        if (len >= sizeof(name)) return 1;
        memcpy(name, list[i], len);
        name[len] = '\0';
        if ((set ? setenv(name, eq + 1, 1) : unsetenv(name)) != 0) return 1;
    }
    return 0;
}

static int select_variant(const bench_config *cfg, int variant) {
    /* Control: candidate settings unset (or set to --control-value),
     * control-only settings applied.  Candidate: control-only settings
     * unset, candidate settings applied. */
    int env_rc = apply_env_list(cfg->control_env, cfg->n_control, variant == 0);
    env_rc |= apply_env_list(cfg->extra_env, cfg->n_extra, variant != 0);
    env_rc |= variant == 0
            ? (cfg->control_value ? setenv(cfg->candidate_env, cfg->control_value, 1) : unsetenv(cfg->candidate_env))
            : setenv(cfg->candidate_env, cfg->candidate_value, 1);
    if (env_rc != 0) {
        fprintf(stderr,
                "%s: failed to select %s with %s: %s\n",
                BENCH_NAME,
                variant == 0 ? "control" : "candidate",
                cfg->candidate_env,
                strerror(errno));
        return 1;
    }
    return 0;
}

static uint32_t float_bits(float value) {
    uint32_t bits = 0;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static bool g_tolerate_drift = false;

/* Drift statistics over one full-vocabulary row: max and mean |delta|, the
 * top-1 ids of both rows, and the reference top-1 margin (top-1 minus top-2). */
static void report_drift(const float *reference, const float *observed, int vocab, size_t run) {
    double max_d = 0.0, sum_d = 0.0;
    int r0 = 0, o0 = 0, r1 = -1;
    for (int i = 0; i < vocab; i++) {
        const double d = fabs((double)reference[i] - (double)observed[i]);
        if (d > max_d) max_d = d;
        sum_d += d;
        if (reference[i] > reference[r0]) { r1 = r0; r0 = i; }
        else if (r1 < 0 || reference[i] > reference[r1]) r1 = i;
        if (observed[i] > observed[o0]) o0 = i;
    }
    printf("run=%zu drift max_abs=%.6g mean_abs=%.6g top1_reference=%d top1_observed=%d top1_agree=%s reference_margin=%.6g\n",
           run, max_d, sum_d / (double)vocab, r0, o0, r0 == o0 ? "yes" : "no",
           r1 >= 0 ? (double)reference[r0] - (double)reference[r1] : 0.0);
}

static int compare_logits(
        const float *reference,
        const float *observed,
        int          vocab,
        size_t       run) {
    const size_t bytes = (size_t)vocab * sizeof(reference[0]);
    if (memcmp(reference, observed, bytes) == 0) return 0;
    if (g_tolerate_drift) { report_drift(reference, observed, vocab, run); return 0; }

    size_t first = SIZE_MAX;
    size_t differing = 0;
    for (int i = 0; i < vocab; i++) {
        if (memcmp(&reference[i], &observed[i], sizeof(float)) != 0) {
            if (first == SIZE_MAX) first = (size_t)i;
            differing++;
        }
    }
    fprintf(stderr,
            "%s: raw logit mismatch at run %zu: differing=%zu/%d\n",
            BENCH_NAME,
            run,
            differing,
            vocab);
    if (first != SIZE_MAX) {
        fprintf(stderr,
                "%s: first mismatch id=%zu reference=%a (0x%08x) "
                "observed=%a (0x%08x)\n",
                BENCH_NAME,
                first,
                reference[first],
                (unsigned)float_bits(reference[first]),
                observed[first],
                (unsigned)float_bits(observed[first]));
    }
    return 1;
}

static int warm_variant(
        ds4_engine         *engine,
        const bench_config *cfg,
        const ds4_tokens   *all_tokens,
        int                 variant,
        char               *err,
        size_t              errlen) {
    ds4_session *session = NULL;
    ds4_tokens warmup = {
        .v = all_tokens->v,
        .len = cfg->warmup_tokens,
        .cap = cfg->warmup_tokens,
    };
    if (select_variant(cfg, variant) != 0) return 1;
    if (ds4_session_create(&session, engine, cfg->ctx) != 0) {
        fprintf(stderr,
                "%s: failed to create %s warmup session\n",
                BENCH_NAME,
                variant == 0 ? "control" : "candidate");
        return 1;
    }
    const int rc = ds4_session_sync(session, &warmup, err, errlen);
    if (rc != 0) {
        fprintf(stderr,
                "%s: %s warmup failed: %s\n",
                BENCH_NAME,
                variant == 0 ? "control" : "candidate",
                err[0] ? err : "unknown error");
    }
    ds4_session_free(session);
    return rc != 0;
}

int main(int argc, char **argv) {
    static const int orders[2][RUNS_PER_REPEAT] = {
        {0, 1, 1, 0},
        {1, 0, 0, 1},
    };
    const bench_config cfg = parse_options(argc, argv);
    g_tolerate_drift = cfg.tolerate_drift;
    char *text = read_text(cfg.prompt_path);
    if (!text) return 1;
    if (select_variant(&cfg, 0) != 0) {
        free(text);
        return 1;
    }

    ds4_engine_options opt = {
        .model_path = cfg.model_path,
        .backend = DS4_BACKEND_METAL,
        .context_size = cfg.ctx,
        .prefill_chunk = (uint32_t)cfg.prefill_chunk,
        .power_percent = 100,
        .warm_weights = true,
    };
    ds4_engine *engine = NULL;
    ds4_tokens tokens = {0};
    float *reference = NULL;
    float *observed = NULL;
    variant_result results[VARIANT_COUNT] = {0};
    char err[256] = {0};
    bool have_reference = false;
    size_t exact_runs = 0;
    int rc = 1;

    if (ds4_engine_open(&engine, &opt) != 0) goto done;
    ds4_tokenize_text(engine, text, &tokens);
    free(text);
    text = NULL;

    const int needed =
        cfg.prefix_tokens > cfg.warmup_tokens
            ? cfg.prefix_tokens
            : cfg.warmup_tokens;
    if (tokens.len < needed) {
        fprintf(stderr,
                "%s: prompt has %d tokens; need %d\n",
                BENCH_NAME,
                tokens.len,
                needed);
        goto done;
    }
    const int vocab = ds4_engine_vocab_size(engine);
    const size_t logit_bytes = (size_t)vocab * sizeof(float);
    reference = malloc(logit_bytes);
    observed = malloc(logit_bytes);
    if (!reference || !observed) {
        fprintf(stderr, "%s: logit buffer allocation failed\n", BENCH_NAME);
        goto done;
    }

    fprintf(stderr,
            "%s: model=%s prompt=%s prefix=%d initial=%d warmup=%d ctx=%d repeats=%d "
            "candidate_env=%s candidate_value=%s control_value=%s prefill_chunk=%d\n",
            BENCH_NAME,
            cfg.model_path,
            cfg.prompt_path,
            cfg.prefix_tokens,
            cfg.initial_tokens,
            cfg.warmup_tokens,
            cfg.ctx,
            cfg.repeats,
            cfg.candidate_env,
            cfg.candidate_value,
            cfg.control_value ? cfg.control_value : "<unset>",
            cfg.prefill_chunk);

    for (int variant = 0; variant < VARIANT_COUNT; variant++) {
        err[0] = '\0';
        if (warm_variant(
                engine, &cfg, &tokens, variant, err, sizeof(err)) != 0) {
            goto done;
        }
    }

    ds4_tokens prefix = {
        .v = tokens.v,
        .len = cfg.prefix_tokens,
        .cap = cfg.prefix_tokens,
    };
    ds4_tokens initial = { .v = tokens.v, .len = cfg.initial_tokens, .cap = cfg.initial_tokens };
    const int timed_tokens = cfg.prefix_tokens - cfg.initial_tokens;
    size_t run = 0;
    if (cfg.interleave) {
        /* Sustained load drifts the GPU by several percent within a minute,
         * and single fresh-session runs carry positional outliers, so
         * measure both variants inside the same session: every chunk after
         * --initial-tokens is timed under the variant of its parity, and the
         * next repeat swaps the parity.  Final logits must match across
         * repeats, which pins every chunk of both variants. */
        const int chunk = cfg.prefill_chunk;
        if (timed_tokens % chunk != 0 || cfg.initial_tokens % chunk != 0) {
            fprintf(stderr, "%s: --interleave needs --initial-tokens and --prefix-tokens as multiples of --prefill-chunk\n", BENCH_NAME);
            goto done;
        }
        /* The first session that reaches a context length pays first-use
         * costs on its buffers for every chunk after the first (measured at
         * -25% on an M5 Max), so pass 0 walks the whole prefix untimed. */
        for (int repeat = -1; repeat < cfg.repeats; repeat++) {
            const bool timed = repeat >= 0;
            ds4_session *session = NULL;
            if (select_variant(&cfg, 0) != 0 || ds4_session_create(&session, engine, cfg.ctx) != 0) {
                fprintf(stderr, "%s: failed to create interleaved session %d\n", BENCH_NAME, repeat + 1);
                goto done;
            }
            err[0] = '\0';
            if (initial.len && ds4_session_sync(session, &initial, err, sizeof(err)) != 0) {
                fprintf(stderr, "%s: initial prefix failed: %s\n", BENCH_NAME, err);
                ds4_session_free(session);
                goto done;
            }
            for (int c = 0; c < timed_tokens / chunk; c++) {
                const int variant = (c + repeat) & 1;
                ds4_tokens part = { .v = tokens.v, .len = cfg.initial_tokens + (c + 1) * chunk,
                                    .cap = cfg.initial_tokens + (c + 1) * chunk };
                if (select_variant(&cfg, variant) != 0) { ds4_session_free(session); goto done; }
                const double t0 = now_sec();
                const int sync_rc = ds4_session_sync(session, &part, err, sizeof(err));
                const double seconds = now_sec() - t0;
                if (sync_rc != 0 || ds4_session_pos(session) != part.len) {
                    fprintf(stderr, "%s: interleaved chunk %d failed: %s\n", BENCH_NAME, c + 1, err[0] ? err : "position");
                    ds4_session_free(session);
                    goto done;
                }
                if (timed) {
                    results[variant].seconds += seconds;
                    results[variant].runs++;
                    results[variant].tokens += (size_t)chunk;
                }
                printf("run=%zu repeat=%d pattern=%s slot=%d variant=%s tokens=%d seconds=%.6f tokens_per_second=%.4f exact=%s\n",
                       run + 1, repeat + 1, !timed ? "warm" : repeat & 1 ? "BABA" : "ABAB", c + 1,
                       variant == 0 ? "control" : "candidate", chunk, seconds,
                       seconds > 0.0 ? (double)chunk / seconds : 0.0, timed ? "pending" : "untimed");
                fflush(stdout);
                if (timed) run++;
            }
            memset(observed, (repeat & 1) == 0 ? 0xa5 : 0x5a, logit_bytes);
            if (ds4_session_copy_logits(session, observed, vocab) != vocab) {
                fprintf(stderr, "%s: failed to copy logits after repeat %d\n", BENCH_NAME, repeat + 1);
                ds4_session_free(session);
                goto done;
            }
            if (!have_reference) {
                memcpy(reference, observed, logit_bytes);
                have_reference = true;
            } else if (compare_logits(reference, observed, vocab, (size_t)repeat + 1) != 0) {
                ds4_session_free(session);
                goto done;
            }
            if (timed) exact_runs++;
            printf("repeat=%d final logits exact=yes\n", repeat + 1);
            ds4_session_free(session);
        }
    } else
    for (int repeat = 0; repeat < cfg.repeats; repeat++) {
        const int *order = orders[repeat & 1];
        const char *pattern = (repeat & 1) == 0 ? "ABBA" : "BAAB";
        for (int slot = 0; slot < RUNS_PER_REPEAT; slot++) {
            const int variant = order[slot];
            ds4_session *session = NULL;
            if (select_variant(&cfg, variant) != 0 ||
                ds4_session_create(&session, engine, cfg.ctx) != 0) {
                fprintf(stderr,
                        "%s: failed to create run %zu %s session\n",
                        BENCH_NAME,
                        run + 1,
                        variant == 0 ? "control" : "candidate");
                if (session) ds4_session_free(session);
                goto done;
            }

            err[0] = '\0';
            if (initial.len && ds4_session_sync(session, &initial, err, sizeof(err)) != 0) {
                fprintf(stderr, "%s: initial prefix failed: %s\n", BENCH_NAME, err);
                ds4_session_free(session);
                goto done;
            }
            const double t0 = now_sec();
            const int sync_rc =
                ds4_session_sync(session, &prefix, err, sizeof(err));
            const double seconds = now_sec() - t0;
            if (sync_rc != 0) {
                fprintf(stderr,
                        "%s: run %zu %s prefill failed: %s\n",
                        BENCH_NAME,
                        run + 1,
                        variant == 0 ? "control" : "candidate",
                        err[0] ? err : "unknown error");
                ds4_session_free(session);
                goto done;
            }
            if (ds4_session_pos(session) != cfg.prefix_tokens) {
                fprintf(stderr,
                        "%s: run %zu ended at position %d; expected %d\n",
                        BENCH_NAME,
                        run + 1,
                        ds4_session_pos(session),
                        cfg.prefix_tokens);
                ds4_session_free(session);
                goto done;
            }

            memset(observed, (run & 1u) == 0 ? 0xa5 : 0x5a, logit_bytes);
            if (ds4_session_copy_logits(session, observed, vocab) != vocab) {
                fprintf(stderr,
                        "%s: failed to copy logits at run %zu\n",
                        BENCH_NAME,
                        run + 1);
                ds4_session_free(session);
                goto done;
            }
            if (!have_reference) {
                if (variant != 0) {
                    fprintf(stderr,
                            "%s: first measured run must be control\n",
                            BENCH_NAME);
                    ds4_session_free(session);
                    goto done;
                }
                memset(reference, 0x5a, logit_bytes);
                if (ds4_session_copy_logits(session, reference, vocab) != vocab ||
                    compare_logits(reference, observed, vocab, run + 1) != 0) {
                    fprintf(stderr,
                            "%s: failed to establish control reference\n",
                            BENCH_NAME);
                    ds4_session_free(session);
                    goto done;
                }
                have_reference = true;
            } else if (compare_logits(
                           reference, observed, vocab, run + 1) != 0) {
                ds4_session_free(session);
                goto done;
            }

            results[variant].seconds += seconds;
            results[variant].runs++;
            results[variant].tokens += (size_t)timed_tokens;
            exact_runs++;
            printf("run=%zu repeat=%d pattern=%s slot=%d variant=%s "
                   "tokens=%d seconds=%.6f tokens_per_second=%.4f exact=yes\n",
                   run + 1,
                   repeat + 1,
                   pattern,
                   slot + 1,
                   variant == 0 ? "control" : "candidate",
                   timed_tokens,
                   seconds,
                   seconds > 0.0
                       ? (double)timed_tokens / seconds
                       : 0.0);
            fflush(stdout);
            ds4_session_free(session);
            run++;
        }
    }

    for (int variant = 0; variant < VARIANT_COUNT; variant++) {
        printf("aggregate variant=%s runs=%zu tokens=%zu seconds=%.6f "
               "tokens_per_second=%.4f\n",
               variant == 0 ? "control" : "candidate",
               results[variant].runs,
               results[variant].tokens,
               results[variant].seconds,
               results[variant].seconds > 0.0
                   ? (double)results[variant].tokens /
                         results[variant].seconds
                   : 0.0);
    }
    {
        const double control_tps =
            results[0].seconds > 0.0
                ? (double)results[0].tokens / results[0].seconds
                : 0.0;
        const double candidate_tps =
            results[1].seconds > 0.0
                ? (double)results[1].tokens / results[1].seconds
                : 0.0;
        printf("candidate_delta_percent=%.4f exact_runs=%zu "
               "exact_floats=%zu vocab=%d\n",
               control_tps > 0.0
                   ? (candidate_tps / control_tps - 1.0) * 100.0
                   : 0.0,
               exact_runs,
               exact_runs * (size_t)vocab,
               vocab);
    }
    rc = 0;

done:
    free(text);
    free(reference);
    free(observed);
    ds4_tokens_free(&tokens);
    if (engine) ds4_engine_close(engine);
    return rc;
}
