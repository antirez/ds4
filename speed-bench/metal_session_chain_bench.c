#define _DARWIN_C_SOURCE
#include "ds4.h"

#include <errno.h>
#include <limits.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define BENCH_NAME "metal-session-chain-bench"

typedef struct {
    const char *model_path;
    const char *prompt_path;
    int prefix_tokens;
    int tokens;
    int burst;
    int warmup;
    int ctx;
    bool control_chain;
} bench_config;

typedef struct {
    int *ids;
    int count;
    int capacity;
    bool overflow;
} token_capture;

static void usage(FILE *fp, const char *argv0) {
    fprintf(fp,
            "usage: %s [options]\n"
            "  -m, --model PATH    GGUF path (default: ds4flash.gguf)\n"
            "  --prompt-file PATH  token source (default: ds4.c)\n"
            "  --prefix-tokens N   prefill length (default: 2048)\n"
            "  --tokens N          evaluated tokens per variant (default: 256; min: 2)\n"
            "  --burst N           maximum tokens per AB/BA pair (default: 16; min: 2)\n"
            "  --warmup N          untimed tokens per variant (default: 16; 0 or >=2)\n"
            "  --ctx N             allocation (default: prefix + warmup + tokens + 1)\n"
            "  --control-chain     compare raw-prefix chain against lookahead1 chain\n"
            "                      set DS4_METAL_REQUIRE_SESSION_CHAIN_RAW_PREFIX=1\n"
            "                      to assert the default raw-prefix path is selected\n"
            "EOS is evaluated normally. Both variants evaluate the seed AND last token.\n",
            argv0);
}

static int parse_number(const char *text, const char *option, int minimum) {
    char *end = NULL;
    errno = 0;
    long value = strtol(text, &end, 10);
    if (errno || !text[0] || !end || *end || value < minimum || value > INT_MAX) {
        fprintf(stderr, "%s: invalid %s: %s\n", BENCH_NAME, option, text);
        exit(2);
    }
    return (int)value;
}

static bench_config parse_options(int argc, char **argv) {
    bench_config cfg = {
        .model_path = "ds4flash.gguf", .prompt_path = "ds4.c",
        .prefix_tokens = 2048, .tokens = 256, .burst = 16, .warmup = 16,
    };
    for (int i = 1; i < argc; ++i) {
        const char *option = argv[i];
        if (!strcmp(option, "--help") || !strcmp(option, "-h")) {
            usage(stdout, argv[0]);
            exit(0);
        }
        if (!strcmp(option, "--control-chain")) {
            cfg.control_chain = true;
            continue;
        }
        if (i + 1 >= argc) {
            fprintf(stderr, "%s: missing value for %s\n", BENCH_NAME, option);
            exit(2);
        }
        const char *value = argv[++i];
        if (!strcmp(option, "--model") || !strcmp(option, "-m")) cfg.model_path = value;
        else if (!strcmp(option, "--prompt-file")) cfg.prompt_path = value;
        else if (!strcmp(option, "--prefix-tokens")) cfg.prefix_tokens = parse_number(value, option, 1);
        else if (!strcmp(option, "--tokens")) cfg.tokens = parse_number(value, option, 2);
        else if (!strcmp(option, "--burst")) cfg.burst = parse_number(value, option, 2);
        else if (!strcmp(option, "--warmup")) cfg.warmup = parse_number(value, option, 0);
        else if (!strcmp(option, "--ctx")) cfg.ctx = parse_number(value, option, 2);
        else {
            fprintf(stderr, "%s: unknown option: %s\n", BENCH_NAME, option);
            exit(2);
        }
    }
    /* A session-chain call needs at least two tokens. Avoid silently mixing
     * classic single-token fallbacks into the candidate's measured tail. */
    if (cfg.warmup == 1 ||
        (cfg.burst == 2 && ((cfg.tokens & 1) || (cfg.warmup & 1)))) {
        fprintf(stderr, "%s: warmup must be 0 or >=2; burst=2 requires even token counts\n", BENCH_NAME);
        exit(2);
    }
    const int64_t needed = (int64_t)cfg.prefix_tokens + cfg.warmup + cfg.tokens + 1;
    if (needed > INT_MAX || (cfg.ctx != 0 && cfg.ctx < needed)) {
        fprintf(stderr, "%s: context must be >= prefix + warmup + tokens + 1 (%lld)\n",
                BENCH_NAME, (long long)needed);
        exit(2);
    }
    if (cfg.ctx == 0) cfg.ctx = (int)needed;
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
        fprintf(stderr, "%s: cannot open %s: %s\n", BENCH_NAME, path, strerror(errno));
        return NULL;
    }
    char *text = NULL;
    if (fseek(fp, 0, SEEK_END) != 0) goto done;
    const long length = ftell(fp);
    if (length < 0 || fseek(fp, 0, SEEK_SET) != 0) goto done;
    text = malloc((size_t)length + 1u);
    if (!text) goto done;
    if (fread(text, 1, (size_t)length, fp) != (size_t)length) {
        free(text);
        text = NULL;
    } else {
        text[length] = '\0';
    }
done:
    fclose(fp);
    if (!text) fprintf(stderr, "%s: failed reading %s\n", BENCH_NAME, path);
    return text;
}

static bool capture_token(void *ctx, int token) {
    token_capture *capture = ctx;
    if (capture->count >= capture->capacity) {
        capture->overflow = true;
        return false;
    }
    capture->ids[capture->count++] = token;
    return true; /* Including EOS: this measures a fixed number of evals. */
}

static bool run_burst(ds4_session *session, int variant, bool control_chain, int count, int vocab,
                      token_capture *capture, double *seconds, char *err, size_t errlen) {
    capture->count = 0;
    capture->capacity = count;
    capture->overflow = false;
    const int before = ds4_session_pos(session);
    const double start = now_sec();
    bool ok = true;
    if (variant == 0 && !control_chain) {
        for (int i = 0; ok && i < count; ++i) {
            const int token = ds4_session_argmax(session);
            ok = token >= 0 && token < vocab && capture_token(capture, token) &&
                 ds4_session_eval(session, token, err, errlen) == 0;
        }
    } else {
        bool completed = false;
        const int approved = ds4_session_eval_chain_greedy(
            session, count, capture_token, capture, &completed, err, errlen);
        ok = approved == count && completed;
    }
    *seconds = now_sec() - start;
    const ds4_tokens *checkpoint = ds4_session_tokens(session);
    ok = ok && !capture->overflow && capture->count == count && checkpoint &&
         checkpoint->len == before + count && ds4_session_pos(session) == before + count &&
         memcmp(checkpoint->v + before, capture->ids, (size_t)count * sizeof(int)) == 0;
    if (!ok) fprintf(stderr,
        "%s: %s burst failed pos=%d count=%d captured=%d final=%d error=%s\n",
        BENCH_NAME, variant ? "chain" : control_chain ? "chain-fallback" : "classic",
        before, count, capture->count,
        ds4_session_pos(session), err[0] ? err : "state/count mismatch");
    return ok;
}

static bool compare_frontier(ds4_session **sessions, float **logits, int vocab,
                             int expected_pos) {
    const ds4_tokens *a = ds4_session_tokens(sessions[0]);
    const ds4_tokens *b = ds4_session_tokens(sessions[1]);
    if (!a || !b || a->len != expected_pos || b->len != expected_pos ||
        ds4_session_pos(sessions[0]) != expected_pos ||
        ds4_session_pos(sessions[1]) != expected_pos ||
        memcmp(a->v, b->v, (size_t)expected_pos * sizeof(int)) != 0) {
        fprintf(stderr, "%s: token checkpoint mismatch at pos=%d\n", BENCH_NAME, expected_pos);
        return false;
    }
    const size_t bytes = (size_t)vocab * sizeof(float);
    for (int variant = 0; variant < 2; ++variant) {
        memset(logits[variant], variant ? 0x5a : 0xa5, bytes);
        if (ds4_session_copy_logits(sessions[variant], logits[variant], vocab) != vocab) {
            fprintf(stderr, "%s: logits unavailable variant=%d pos=%d\n", BENCH_NAME, variant, expected_pos);
            return false;
        }
    }
    int different = 0, first = -1;
    for (int i = 0; i < vocab; ++i) {
        uint32_t bits[2];
        memcpy(&bits[0], &logits[0][i], sizeof(bits[0]));
        memcpy(&bits[1], &logits[1][i], sizeof(bits[1]));
        /* Bit tests remain meaningful under the project's -ffast-math. */
        if ((bits[0] & 0x7f800000u) == 0x7f800000u ||
            (bits[1] & 0x7f800000u) == 0x7f800000u) {
            fprintf(stderr, "%s: non-finite logits pos=%d id=%d\n", BENCH_NAME, expected_pos, i);
            return false;
        }
        if (bits[0] != bits[1]) {
            if (first < 0) first = i;
            ++different;
        }
    }
    if (different) fprintf(stderr,
        "%s: logits mismatch pos=%d different=%d/%d first=%d session0=%a session1=%a\n",
        BENCH_NAME, expected_pos, different, vocab, first, logits[0][first], logits[1][first]);
    return different == 0;
}

int main(int argc, char **argv) {
    const bench_config cfg = parse_options(argc, argv);
    const char *disable_prefix = "DS4_METAL_DISABLE_SESSION_CHAIN_RAW_PREFIX";
    const char *saved_disable_env = getenv(disable_prefix);
    char *saved_disable = saved_disable_env ? strdup(saved_disable_env) : NULL;
    if (saved_disable_env && !saved_disable) return 1;
    char *text = read_text(cfg.prompt_path);
    if (!text) { free(saved_disable); return 1; }
    ds4_engine *engine = NULL;
    ds4_session *sessions[2] = {NULL, NULL};
    ds4_tokens tokens = {0};
    float *logits[2] = {NULL, NULL};
    token_capture captures[2] = {{0}, {0}};
    char err[256] = {0};
    int status = 1;
    const ds4_engine_options options = {
        .model_path = cfg.model_path, .backend = DS4_BACKEND_METAL,
        .context_size = cfg.ctx, .power_percent = 100, .warm_weights = true,
    };
    if (ds4_engine_open(&engine, &options) != 0) goto done;
    ds4_tokenize_text(engine, text, &tokens);
    if (tokens.len < cfg.prefix_tokens) {
        fprintf(stderr, "%s: prompt has %d tokens; need %d\n", BENCH_NAME, tokens.len, cfg.prefix_tokens);
        goto done;
    }
    const int vocab = ds4_engine_vocab_size(engine);
    if (vocab <= 0) goto done;
    ds4_tokens prefix = {.v = tokens.v, .len = cfg.prefix_tokens, .cap = cfg.prefix_tokens};
    for (int variant = 0; variant < 2; ++variant) {
        logits[variant] = malloc((size_t)vocab * sizeof(float));
        captures[variant].ids = malloc((size_t)cfg.burst * sizeof(int));
        if (!logits[variant] || !captures[variant].ids ||
            ds4_session_create(&sessions[variant], engine, cfg.ctx) != 0 ||
            ds4_session_sync(sessions[variant], &prefix, err, sizeof(err)) != 0) {
            fprintf(stderr, "%s: setup failed: %s\n", BENCH_NAME, err);
            goto done;
        }
    }
    if (!ds4_session_chain_greedy_supported(sessions[0]) ||
        !ds4_session_chain_greedy_supported(sessions[1])) {
        fprintf(stderr, "%s: session chain unsupported; check backend/model/diagnostic settings\n", BENCH_NAME);
        goto done;
    }
    if (!compare_frontier(sessions, logits, vocab, cfg.prefix_tokens)) goto done;
    fprintf(stderr,
        "%s: model=%s prefix=%d warmup=%d tokens=%d burst=%d ctx=%d control=%s selection=argmax-including-EOS\n",
        BENCH_NAME, cfg.model_path, cfg.prefix_tokens, cfg.warmup, cfg.tokens, cfg.burst, cfg.ctx,
        cfg.control_chain ? "chain-fallback" : "classic");
    double total_seconds[2] = {0, 0};
    int total_tokens[2] = {0, 0};
    uint64_t checked_frontiers = 1;
    for (int phase = 0; phase < 2; ++phase) {
        const int limit = phase ? cfg.tokens : cfg.warmup;
        int evaluated = 0;
        for (int pair = 0; evaluated < limit; ++pair) {
            const int remaining = limit - evaluated;
            int count = remaining < cfg.burst ? remaining : cfg.burst;
            if (remaining - count == 1) --count;
            const int pos = cfg.prefix_tokens + (phase ? cfg.warmup : 0) + evaluated;
            double seconds[2] = {0, 0};
            /* Balance method order every pair, and physical-session binding
             * every two pairs, so both effects are crossed over four pairs. */
            const int session_xor = (pair >> 1) & 1;
            for (int slot = 0; slot < 2; ++slot) {
                const int variant = slot ^ (pair & 1);
                if (cfg.control_chain &&
                    (variant ? unsetenv(disable_prefix) : setenv(disable_prefix, "1", 1))) {
                    fprintf(stderr, "%s: failed setting chain variant environment\n", BENCH_NAME);
                    goto done;
                }
                err[0] = '\0';
                if (!run_burst(sessions[variant ^ session_xor], variant, cfg.control_chain, count, vocab,
                               &captures[variant], &seconds[variant], err, sizeof(err))) goto done;
            }
            if (memcmp(captures[0].ids, captures[1].ids, (size_t)count * sizeof(int)) != 0) {
                fprintf(stderr, "%s: generated token IDs differ at pos=%d\n", BENCH_NAME, pos);
                goto done;
            }
            if (!compare_frontier(sessions, logits, vocab, pos + count)) goto done;
            ++checked_frontiers;
            if (phase) {
                for (int variant = 0; variant < 2; ++variant) {
                    total_seconds[variant] += seconds[variant];
                    total_tokens[variant] += count;
                }
                printf("pair=%d order=%s %s_session=%d pos=%d tokens=%d %s_seconds=%.6f chain_seconds=%.6f exact=yes\n",
                       pair + 1, (pair & 1) ? "BA" : "AB",
                       cfg.control_chain ? "fallback_chain" : "classic", session_xor,
                       pos, count, cfg.control_chain ? "fallback_chain" : "classic",
                       seconds[0], seconds[1]);
                fflush(stdout);
            }
            evaluated += count;
        }
    }
    double tps[2];
    for (int variant = 0; variant < 2; ++variant) {
        tps[variant] = total_seconds[variant] > 0 ? total_tokens[variant] / total_seconds[variant] : 0;
        printf("aggregate variant=%s tokens=%d seconds=%.6f tokens_per_second=%.4f\n",
               variant ? "chain" : cfg.control_chain ? "chain-fallback" : "classic",
               total_tokens[variant], total_seconds[variant], tps[variant]);
    }
    printf("chain_delta_percent=%.4f exact_frontiers=%llu exact_floats=%llu vocab=%d final_pos=%d\n",
           tps[0] > 0 ? 100.0 * (tps[1] / tps[0] - 1.0) : 0.0,
           (unsigned long long)checked_frontiers,
           (unsigned long long)(checked_frontiers * (uint64_t)vocab), vocab,
           ds4_session_pos(sessions[0]));
    status = 0;
done:
    if (cfg.control_chain) {
        if (saved_disable) setenv(disable_prefix, saved_disable, 1);
        else unsetenv(disable_prefix);
    }
    free(saved_disable);
    for (int variant = 0; variant < 2; ++variant) {
        ds4_session_free(sessions[variant]);
        free(logits[variant]);
        free(captures[variant].ids);
    }
    ds4_tokens_free(&tokens);
    ds4_engine_close(engine);
    free(text);
    return status;
}
