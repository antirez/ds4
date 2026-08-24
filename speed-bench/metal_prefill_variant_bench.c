#include "ds4.h"

#include <errno.h>
#include <limits.h>
#include <stdbool.h>
#include <stdint.h>
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
    int prefix_tokens;
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
            "  --candidate-env NAME   unset NAME for control, set NAME=1 for candidate\n"
            "  --prefix-tokens N      timed prefill length (default: 8192)\n"
            "  --warmup-tokens N      untimed tokens per variant (default: 32; min: 32)\n"
            "  --ctx N                session allocation (default: max lengths + 1)\n"
            "  --repeats N            alternating ABBA/BAAB pairs (default: 2)\n",
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
        } else if (!strcmp(arg, "--candidate-env")) {
            cfg.candidate_env = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--prefix-tokens")) {
            cfg.prefix_tokens =
                parse_int_arg(need_arg(&i, argc, argv, arg), arg, 1);
        } else if (!strcmp(arg, "--warmup-tokens")) {
            cfg.warmup_tokens = parse_int_arg(
                need_arg(&i, argc, argv, arg), arg, DEFAULT_WARMUP_TOKENS);
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

static int select_variant(const bench_config *cfg, int variant) {
    const int env_rc =
        variant == 0
            ? unsetenv(cfg->candidate_env)
            : setenv(cfg->candidate_env, "1", 1);
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

static int compare_logits(
        const float *reference,
        const float *observed,
        int          vocab,
        size_t       run) {
    const size_t bytes = (size_t)vocab * sizeof(reference[0]);
    if (memcmp(reference, observed, bytes) == 0) return 0;

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
        .prefill_chunk = 4096,
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
            "%s: model=%s prompt=%s prefix=%d warmup=%d ctx=%d repeats=%d "
            "candidate_env=%s\n",
            BENCH_NAME,
            cfg.model_path,
            cfg.prompt_path,
            cfg.prefix_tokens,
            cfg.warmup_tokens,
            cfg.ctx,
            cfg.repeats,
            cfg.candidate_env);

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
    size_t run = 0;
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
            results[variant].tokens += (size_t)cfg.prefix_tokens;
            exact_runs++;
            printf("run=%zu repeat=%d pattern=%s slot=%d variant=%s "
                   "tokens=%d seconds=%.6f tokens_per_second=%.4f exact=yes\n",
                   run + 1,
                   repeat + 1,
                   pattern,
                   slot + 1,
                   variant == 0 ? "control" : "candidate",
                   cfg.prefix_tokens,
                   seconds,
                   seconds > 0.0
                       ? (double)cfg.prefix_tokens / seconds
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
