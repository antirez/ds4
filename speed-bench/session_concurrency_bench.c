#include "ds4.h"

/* Session-concurrency benchmark.
 *
 * Serving several requests at once is a different regime from the single
 * session walk ds4-bench measures.  The numbers that matter here are the
 * aggregate decode throughput at a given batch width, how much per-stream
 * speed is lost as the width grows, and how badly a resumed prefill disturbs
 * the decoders that share the engine.  This harness drives the engine API
 * directly so the result carries no HTTP, tokenizer or scheduler noise: it is
 * the ceiling the server can aim at, and the signal to optimize against.
 *
 * The grid is concurrency x context, and memory, not time, is what bounds its
 * wide/long corner.  A cell that does not fit cannot simply be attempted:
 * macOS compresses and swaps long before an allocation fails, which wedges the
 * machine instead of reporting an error.  So the harness prices every cell
 * from the engine's own accounting (the per-session estimate the server uses
 * for admission, plus a margin for what the estimate does not see) against a
 * budget sampled once at startup, and skips what it cannot afford.  System
 * readings taken while running are not usable for this: the process footprint
 * understates a session, because caches only become resident once prefill
 * touches them, and freed pages the allocator recycles never show up.
 * Streams read the corpus at staggered offsets: identical prompts would give
 * every stream the same MoE routing and flatter any batched expert path.
 */

#include <errno.h>
#include <limits.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#if defined(__APPLE__)
#include <mach/mach.h>
#include <mach/mach_host.h>
#endif

#define BENCH "session-concurrency-bench"

enum {
    MAX_STREAMS = 64,
    MAX_LIST = 32,
    /* Prefill seed for the mixed-phase victim: enough to own a valid
     * checkpoint that the timed chunks can extend. */
    MIXED_SEED_TOKENS = 16,
    DEFAULT_GEN = 64,
    DEFAULT_WARMUP = 8,
    DEFAULT_MIXED_STEPS = 64,
    DEFAULT_MIXED_QUANTUM = 128,
    CTX_MARGIN = 16,
};

/* Held back so the sweep never pushes the machine into compression or swap. */
#define DEFAULT_RESERVE_GIB 8.0
/* The context accounting covers the caches and the prefill transients but not
 * the recurrent state, the host staging rows, or the logit buffers.  Those run
 * to a few percent of a session at the default chunk. */
#define SESSION_EXTRAS 1.10

typedef struct {
    const char *model_path;
    const char *prompt_path;
    const char *csv_path;
    int concurrency[MAX_LIST];
    int n_concurrency;
    int ctx[MAX_LIST];
    int n_ctx;
    int gen;
    int warmup;
    int mixed_steps;
    int mixed_quantum;
    uint32_t prefill_chunk;
    double budget_gib;
    double reserve_gib;
    uint64_t budget_bytes;
    bool mixed;
    bool spec;        /* batched speculative decode (Qwen3.8 MTP, greedy) */
    bool verify;
    bool warm_weights;
    bool force;
    bool private_transients;
    const char *dump_tokens_path;
    const char *dump_logits_dir;
    const char *candidate_env;
    const char *candidate_value;
    int repeat;
} bench_config;

typedef struct {
    int    concurrency;
    int    ctx;
    bool   ran;
    bool   oom;
    double predicted_gib;
    double prefill_wall_s;      /* whole wave, first stream start to last done */
    double prefill_last_ttft_s; /* the unlucky stream's wait */
    double prefill_tps;         /* aggregate prompt tokens per second */
    double decode_agg_tps;      /* eval-only, comparable with ds4-bench */
    double decode_wall_tps;     /* including token selection */
    double step_ms_mean;
    double step_ms_p50;
    double step_ms_p95;
    double step_ms_max;
    double mixed_step_ms_p50;
    double mixed_step_ms_p95;
    double mixed_decode_tps;
    double mixed_prefill_tps;
    double footprint_gib;
    double estimate_gib;
    bool   verify_ok;
    double control_tps;
    double candidate_tps;
} cell_result;

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

static double bytes_to_gib(uint64_t bytes) {
    return (double)bytes / (1024.0 * 1024.0 * 1024.0);
}

/* Memory the machine can still hand out, sampled once the weights are
 * resident.  Sessions are freed between cells, so this stays the budget for
 * every cell of the sweep. */
static uint64_t available_memory_bytes(void) {
#if defined(__APPLE__)
    vm_size_t page = 0;
    vm_statistics64_data_t vm;
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    if (host_page_size(mach_host_self(), &page) != KERN_SUCCESS) return 0;
    if (host_statistics64(mach_host_self(), HOST_VM_INFO64,
                          (host_info64_t)&vm, &count) != KERN_SUCCESS) {
        return 0;
    }
    const uint64_t reclaimable = (uint64_t)vm.free_count +
                                 (uint64_t)vm.inactive_count +
                                 (uint64_t)vm.purgeable_count +
                                 (uint64_t)vm.external_page_count;
    return reclaimable * (uint64_t)page;
#else
    FILE *fp = fopen("/proc/meminfo", "r");
    if (!fp) return 0;
    char line[256];
    uint64_t kib = 0;
    while (fgets(line, sizeof(line), fp)) {
        if (sscanf(line, "MemAvailable: %llu kB",
                   (unsigned long long *)&kib) == 1) {
            break;
        }
    }
    fclose(fp);
    return kib * 1024ull;
#endif
}

/* Sampled once, with the weights resident and no session yet alive.  Sampling
 * again later would read low and drift down over a sweep: prefill fills the
 * unified buffer cache with n-gram pages, which are reclaimable but which the
 * system counts as used. */
static uint64_t session_budget_bytes(const bench_config *cfg) {
    const double gib = 1024.0 * 1024.0 * 1024.0;
    if (cfg->budget_gib > 0.0) return (uint64_t)(cfg->budget_gib * gib);
    const uint64_t available = available_memory_bytes();
    const uint64_t reserve = (uint64_t)(cfg->reserve_gib * gib);
    return available > reserve ? available - reserve : 0;
}

static void usage(FILE *fp, const char *argv0) {
    fprintf(fp,
            "usage: %s [options]\n"
            "\n"
            "  -m, --model PATH      GGUF path (default: ds4flash.gguf)\n"
            "  --prompt-file PATH    token source (default: speed-bench/promessi_sposi.txt)\n"
            "  --concurrency LIST    stream counts (default: 1,2,4,8,16)\n"
            "  --ctx LIST            prompt lengths (default: 0,4096,16384,32768,65536,131072)\n"
            "  --gen N               measured decode steps per stream (default: %d)\n"
            "  --warmup N            untimed decode steps (default: %d)\n"
            "  --prefill-chunk N     engine prefill chunk\n"
            "  --warm-weights        touch the weights before the first cell\n"
            "  --mixed               also measure decode while one stream prefills\n"
            "  --mixed-steps N       timed mixed steps (default: %d)\n"
            "  --mixed-quantum N     prefill tokens per mixed step (default: %d)\n"
            "  --verify              check batched decode against sequential decode\n"
            "  --spec                batched speculative decode (Qwen3.8 MTP, greedy acceptance)\n"
            "  --budget-gib X        memory the sessions may use (default: measured)\n"
            "  --reserve-gib X       held back from that budget (default: %.0f)\n"
            "  --force               run every cell, including ones that will not fit\n"
            "  --private-transients  give every session its own prefill arena\n"
            "  --dump-tokens PATH    append the tokens each stream generated\n"
            "  --dump-logits-dir DIR write each stream's final logit row, for bit comparison\n"
            "  --candidate-env NAME  paired A/B: unset NAME for control, set it for candidate\n"
            "  --candidate-value V   value the candidate sets (default 1)\n"
            "  --repeat N            ABBA blocks per cell in A/B mode (default 2)\n"
            "  --csv PATH            append one row per cell\n"
            "\n"
            "Cells are priced from the engine's context accounting; the ones\n"
            "that would exceed the budget are skipped rather than attempted.\n",
            argv0, DEFAULT_GEN, DEFAULT_WARMUP,
            DEFAULT_MIXED_STEPS, DEFAULT_MIXED_QUANTUM, DEFAULT_RESERVE_GIB);
}

static const char *need_arg(int *i, int argc, char **argv, const char *opt) {
    if (*i + 1 >= argc) {
        fprintf(stderr, BENCH ": %s requires an argument\n", opt);
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
        fprintf(stderr, BENCH ": invalid value for %s: %s\n", opt, value);
        exit(2);
    }
    return (int)parsed;
}

/* "1,2,4,8" into out[]; values are kept in the order given so a sweep can be
 * driven from the largest cell down when memory is the concern. */
static int parse_list(const char *value, const char *opt, int *out, int minimum) {
    int n = 0;
    const char *p = value;
    while (*p) {
        char *end = NULL;
        errno = 0;
        long parsed = strtol(p, &end, 10);
        if (errno != 0 || end == p || parsed < minimum || parsed > INT_MAX) {
            fprintf(stderr, BENCH ": invalid list for %s: %s\n", opt, value);
            exit(2);
        }
        if (n == MAX_LIST) {
            fprintf(stderr, BENCH ": %s accepts at most %d values\n", opt, MAX_LIST);
            exit(2);
        }
        out[n++] = (int)parsed;
        p = end;
        if (*p == ',') p++;
        else if (*p != '\0') {
            fprintf(stderr, BENCH ": invalid list for %s: %s\n", opt, value);
            exit(2);
        }
    }
    if (n == 0) {
        fprintf(stderr, BENCH ": empty list for %s\n", opt);
        exit(2);
    }
    return n;
}

static char *read_file(const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, BENCH ": failed to open %s: %s\n", path, strerror(errno));
        return NULL;
    }
    if (fseek(fp, 0, SEEK_END) != 0) {
        fprintf(stderr, BENCH ": failed to seek %s\n", path);
        fclose(fp);
        return NULL;
    }
    long len = ftell(fp);
    if (len < 0 || fseek(fp, 0, SEEK_SET) != 0) {
        fprintf(stderr, BENCH ": failed to size %s\n", path);
        fclose(fp);
        return NULL;
    }
    char *text = malloc((size_t)len + 1u);
    if (!text) {
        fprintf(stderr, BENCH ": out of memory reading %s\n", path);
        fclose(fp);
        return NULL;
    }
    if (fread(text, 1, (size_t)len, fp) != (size_t)len) {
        fprintf(stderr, BENCH ": failed to read %s\n", path);
        free(text);
        fclose(fp);
        return NULL;
    }
    text[len] = '\0';
    fclose(fp);
    return text;
}

static int compare_double(const void *a, const void *b) {
    const double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : (x > y ? 1 : 0);
}

/* Nearest-rank percentile over an already sorted array. */
static double percentile(const double *sorted, int n, double fraction) {
    if (n <= 0) return 0.0;
    int idx = (int)(fraction * (double)n);
    if (idx >= n) idx = n - 1;
    if (idx < 0) idx = 0;
    return sorted[idx];
}

static double mean(const double *values, int n) {
    if (n <= 0) return 0.0;
    double sum = 0.0;
    for (int i = 0; i < n; i++) sum += values[i];
    return sum / (double)n;
}

static void free_sessions(ds4_session **sessions, int count) {
    for (int i = 0; i < count; i++) {
        if (sessions[i]) ds4_session_free(sessions[i]);
        sessions[i] = NULL;
    }
}

/* Stream i reads the corpus from a different offset so the streams do not
 * share MoE routing.  With a short corpus the offsets collapse to zero and the
 * caller is told, because identical contexts make batched expert work look
 * better than it is. */
static int stream_offset(int stream, int concurrency, int span, int need) {
    if (concurrency <= 1 || span <= need) return 0;
    return (int)((int64_t)stream * (span - need) / (concurrency - 1));
}

typedef struct {
    ds4_engine  *engine;
    ds4_tokens  *corpus;
    int          eos;
    int          vocab;
    bool         spec;
} bench_env;

/* One decode wave: every stream advances by one token per step, which is what
 * the server's decode worker does once its slots are all generating. */
/* produced_out counts the tokens the streams committed: one per stream and
 * step, plus every accepted draft in speculative mode. */
static int measure_decode(const bench_env *env,
                          ds4_session **sessions,
                          int count,
                          int steps,
                          bool timed,
                          double *step_ms,
                          double *eval_sec_out,
                          double *wall_sec_out,
                          int *tokens_out,
                          long *produced_out) {
    ds4_decode_item items[MAX_STREAMS];
    int accepted[MAX_STREAMS][2], n_accepted[MAX_STREAMS];
    char err[256] = {0};
    double eval_sec = 0.0;
    long produced = 0;
    const double wall_t0 = now_sec();

    for (int step = 0; step < steps; step++) {
        for (int i = 0; i < count; i++) {
            ds4_session *s = sessions[i];
            if (ds4_session_pos(s) + (env->spec ? 2 : 1) >= ds4_session_ctx(s)) {
                fprintf(stderr, BENCH ": stream %d reached its context limit\n", i);
                return 1;
            }
            const int token = ds4_session_argmax_excluding(s, env->eos);
            if (token < 0) {
                fprintf(stderr, BENCH ": stream %d has no non-EOS token\n", i);
                return 1;
            }
            items[i].session = s;
            items[i].token = token;
            if (tokens_out) tokens_out[step * count + i] = token;
        }
        const double t0 = now_sec();
        if (env->spec) {
            if (ds4_sessions_eval_batch_speculative_argmax(items, count, accepted, n_accepted,
                                                           err, sizeof(err)) != 0) {
                fprintf(stderr, BENCH ": batched speculative decode failed: %s\n", err);
                return 1;
            }
            for (int i = 0; i < count; i++) produced += n_accepted[i];
        } else {
            if (ds4_sessions_eval_batch(items, count, err, sizeof(err)) != 0) {
                fprintf(stderr, BENCH ": batched decode failed: %s\n", err);
                return 1;
            }
            produced += count;
        }
        const double elapsed = now_sec() - t0;
        eval_sec += elapsed;
        if (timed && step_ms) step_ms[step] = elapsed * 1e3;
    }

    if (eval_sec_out) *eval_sec_out = eval_sec;
    if (wall_sec_out) *wall_sec_out = now_sec() - wall_t0;
    if (produced_out) *produced_out = produced;
    return 0;
}

/* Greedy speculation is lossless: the tokens a speculative batch commits
 * must be the ones plain greedy decode picks.  Both sides start from the
 * same prefilled state; the reference advances one token at a time with
 * the batched non-speculative path. */
static bool verify_spec_matches_plain(const bench_env *env,
                                      ds4_session **spec,
                                      ds4_session **reference,
                                      int count,
                                      int cycles) {
    ds4_decode_item items[MAX_STREAMS];
    int accepted[MAX_STREAMS][2], n_accepted[MAX_STREAMS];
    char err[256] = {0};
    int *seq = calloc((size_t)count * (size_t)cycles * 2u, sizeof(int));
    int len[MAX_STREAMS] = {0};
    long committed = 0;
    if (!seq) return false;
    for (int c = 0; c < cycles; c++) {
        for (int i = 0; i < count; i++) {
            items[i].session = spec[i];
            items[i].token = ds4_session_argmax_excluding(spec[i], env->eos);
        }
        if (ds4_sessions_eval_batch_speculative_argmax(items, count, accepted, n_accepted, err, sizeof(err)) != 0) {
            fprintf(stderr, BENCH ": speculative batch failed: %s\n", err);
            free(seq);
            return false;
        }
        for (int i = 0; i < count; i++) {
            for (int k = 0; k < n_accepted[i]; k++) seq[(size_t)i * cycles * 2 + len[i]++] = accepted[i][k];
            committed += n_accepted[i];
        }
    }
    printf("  spec: %.2f tokens per stream and cycle\n", (double)committed / ((double)count * cycles));
    /* the reference walks every stream to the same length */
    int max_len = 0;
    for (int i = 0; i < count; i++) if (len[i] > max_len) max_len = len[i];
    bool ok = true;
    for (int k = 0; k < max_len && ok; k++) {
        int n = 0;
        for (int i = 0; i < count; i++) {
            if (k >= len[i]) continue;
            const int token = ds4_session_argmax_excluding(reference[i], env->eos);
            const int want = seq[(size_t)i * cycles * 2 + k];
            if (token != want) {
                fprintf(stderr, BENCH ": spec verify failed: stream %d token %d: speculative %d, plain %d\n",
                        i, k, want, token);
                ok = false;
                break;
            }
            items[n].session = reference[i];
            items[n].token = token;
            n++;
        }
        if (ok && n > 0 && ds4_sessions_eval_batch(items, n, err, sizeof(err)) != 0) {
            fprintf(stderr, BENCH ": reference decode failed: %s\n", err);
            ok = false;
        }
    }
    free(seq);
    return ok;
}

/* Batched decode must pick the same tokens as one-at-a-time decode.  Native
 * grouping may reorder floating point reductions, so the contract is token
 * agreement, not bit-identical logits. */
static bool verify_batch_matches_sequential(const bench_env *env,
                                            ds4_session **batched,
                                            ds4_session **reference,
                                            int count,
                                            int steps) {
    ds4_decode_item items[MAX_STREAMS];
    char err[256] = {0};
    /* Token agreement is the contract, but report how far the logits are from
     * bit-identical: that is what says whether a bit-exact batched path is
     * within reach or a different kernel selection away. */
    float *la = malloc((size_t)env->vocab * sizeof(*la));
    float *lb = malloc((size_t)env->vocab * sizeof(*lb));
    uint64_t differing = 0, compared = 0;
    double worst_abs = 0.0, worst_rel = 0.0;
    double min_margin = 0.0;
    bool have_margin = false;

    for (int step = 0; step < steps; step++) {
        for (int i = 0; i < count; i++) {
            if (la && lb &&
                ds4_session_copy_logits(batched[i], la, env->vocab) == env->vocab &&
                ds4_session_copy_logits(reference[i], lb, env->vocab) == env->vocab) {
                double top1 = -1e30, top2 = -1e30;
                for (int v = 0; v < env->vocab; v++) {
                    compared++;
                    if (la[v] > top1) { top2 = top1; top1 = la[v]; }
                    else if (la[v] > top2) { top2 = la[v]; }
                    if (memcmp(&la[v], &lb[v], sizeof(float)) == 0) continue;
                    differing++;
                    const double d = fabs((double)la[v] - (double)lb[v]);
                    if (d > worst_abs) worst_abs = d;
                    const double scale = fabs((double)lb[v]);
                    if (scale > 1e-6 && d / scale > worst_rel) worst_rel = d / scale;
                }
                const double margin = top1 - top2;
                if (!have_margin || margin < min_margin) {
                    min_margin = margin;
                    have_margin = true;
                }
            }
            const int a = ds4_session_argmax_excluding(batched[i], env->eos);
            const int b = ds4_session_argmax_excluding(reference[i], env->eos);
            if (a != b) {
                fprintf(stderr,
                        BENCH ": verify failed at step %d stream %d: "
                        "batched token %d, sequential token %d\n",
                        step, i, a, b);
                /* A near-tie flipped by rounding looks completely different
                 * from a wrong result: print both candidates as each side
                 * scored them. */
                if (la && lb && a >= 0 && b >= 0 && a < env->vocab && b < env->vocab) {
                    fprintf(stderr,
                            BENCH ":   batched    logit[%d]=%.9g logit[%d]=%.9g gap %.3e\n"
                            BENCH ":   sequential logit[%d]=%.9g logit[%d]=%.9g gap %.3e\n",
                            a, (double)la[a], b, (double)la[b],
                            (double)la[a] - (double)la[b],
                            a, (double)lb[a], b, (double)lb[b],
                            (double)lb[b] - (double)lb[a]);
                }
                if (compared) {
                    printf("  logits so far: %llu of %llu differ, worst absolute %.3e\n",
                           (unsigned long long)differing, (unsigned long long)compared, worst_abs);
                }
                free(la);
                free(lb);
                return false;
            }
            items[i].session = batched[i];
            items[i].token = a;
        }
        if (ds4_sessions_eval_batch(items, count, err, sizeof(err)) != 0) {
            fprintf(stderr, BENCH ": verify batched decode failed: %s\n", err);
            return false;
        }
        for (int i = 0; i < count; i++) {
            if (ds4_session_eval(reference[i], items[i].token,
                                 err, sizeof(err)) != 0) {
                fprintf(stderr, BENCH ": verify sequential decode failed: %s\n", err);
                free(la);
                free(lb);
                return false;
            }
        }
    }
    if (compared) {
        printf("  logits: %llu of %llu differ (%.4f%%), worst absolute %.3e\n",
               (unsigned long long)differing, (unsigned long long)compared,
               100.0 * (double)differing / (double)compared, worst_abs);
        if (have_margin) {
            printf("  closest selection: top1 - top2 = %.3e, %.0fx the worst "
                   "logit difference\n",
                   min_margin, worst_abs > 0.0 ? min_margin / worst_abs : 0.0);
        }
    }
    (void)worst_rel;
    free(la);
    free(lb);
    return true;
}

/* Decode under a resumed prefill: the pattern that decides whether one long
 * prompt stalls every other user.  Each step advances the decoders by one
 * token and the victim by one prefill quantum. */
static int measure_mixed(const bench_env *env,
                         ds4_session **sessions,
                         int count,
                         ds4_session *victim,
                         const ds4_tokens *victim_slice,
                         int steps,
                         int quantum,
                         double *step_ms,
                         int *steps_done,
                         int *prefill_tokens)
{
    ds4_decode_item items[MAX_STREAMS];
    char err[256] = {0};
    int frontier = MIXED_SEED_TOKENS;
    int done = 0;

    for (int step = 0; step < steps; step++) {
        if (frontier >= victim_slice->len) break;
        int next = frontier + quantum;
        if (next > victim_slice->len) next = victim_slice->len;

        for (int i = 0; i < count; i++) {
            ds4_session *s = sessions[i];
            if (ds4_session_pos(s) + 1 >= ds4_session_ctx(s)) break;
            const int token = ds4_session_argmax_excluding(s, env->eos);
            if (token < 0) return 1;
            items[i].session = s;
            items[i].token = token;
        }
        const ds4_tokens partial = {
            .v = victim_slice->v,
            .len = next,
            .cap = next,
        };
        const double t0 = now_sec();
        if (ds4_sessions_eval_batch_with_prefill(items, count, victim, &partial,
                                                 err, sizeof(err)) != 0) {
            fprintf(stderr, BENCH ": mixed step failed: %s\n", err);
            return 1;
        }
        step_ms[done++] = (now_sec() - t0) * 1e3;
        frontier = next;
    }

    *steps_done = done;
    *prefill_tokens = frontier - MIXED_SEED_TOKENS;
    return 0;
}

/* Paired A/B inside one engine.  Comparing two processes is hopeless here:
 * each pays a different price to make 70 GiB of weights resident against
 * whatever the last run left cached, and the GPU's thermal state drifts over a
 * long sweep.  Alternating the variants in ABBA blocks on the same prefilled
 * sessions cancels both, which is what the other harnesses in this directory
 * do. */
static int select_variant(const bench_config *cfg, bool candidate) {
    if (!cfg->candidate_env) return 0;
    if (candidate
            ? setenv(cfg->candidate_env, cfg->candidate_value, 1)
            : unsetenv(cfg->candidate_env)) {
        fprintf(stderr, BENCH ": cannot select %s for %s: %s\n",
                cfg->candidate_env, candidate ? "candidate" : "control",
                strerror(errno));
        return 1;
    }
    return 0;
}

static void run_cell(const bench_config *cfg,
                     const bench_env *env,
                     int concurrency,
                     int ctx_tokens,
                     cell_result *out)
{
    memset(out, 0, sizeof(*out));
    out->concurrency = concurrency;
    out->ctx = ctx_tokens;

    const int prompt_tokens = ctx_tokens > 0 ? ctx_tokens : 1;
    /* A/B mode decodes gen steps per variant per block, on top of the plain
     * measured run, and every one of them advances the streams. */
    const int ab_steps = cfg->candidate_env ? cfg->gen * 2 * cfg->repeat : 0;
    const int decode_budget = cfg->warmup + cfg->gen + ab_steps +
                              (cfg->mixed ? cfg->mixed_steps : 0);
    /* a speculative cycle can commit two tokens per stream */
    const int ctx_alloc = prompt_tokens + decode_budget * (cfg->spec ? 2 : 1) + CTX_MARGIN;
    const int session_count = concurrency + (cfg->mixed ? 1 : 0);
    const int total_sessions = cfg->verify ? session_count * 2 : session_count;

    const ds4_context_memory mem =
        ds4_context_memory_estimate_with_prefill(DS4_BACKEND_METAL, ctx_alloc,
                                                 cfg->prefill_chunk);
    out->estimate_gib = bytes_to_gib(mem.total_bytes) * (double)total_sessions;

    const uint64_t budget_bytes = cfg->budget_bytes;
    /* With a shared arena only the caches repeat; the transients are paid
     * once for the whole engine. */
    const double caches = (double)(mem.raw_bytes + mem.compressed_bytes);
    const double transients = (double)mem.scratch_bytes;
    const double per_session =
        (caches + (cfg->private_transients ? transients : 0.0)) * SESSION_EXTRAS;
    const double predicted = per_session * (double)total_sessions +
        (cfg->private_transients ? 0.0 : transients * SESSION_EXTRAS);
    out->predicted_gib = bytes_to_gib((uint64_t)predicted);

    printf("\n== concurrency %d, context %d ==\n", concurrency, ctx_tokens);
    printf("  sessions %d x %d tokens, %.2f GiB each%s, %.2f GiB against a "
           "%.2f GiB budget\n",
           total_sessions, ctx_alloc, bytes_to_gib((uint64_t)per_session),
           cfg->private_transients ? "" : " plus one shared arena",
           out->predicted_gib, bytes_to_gib(budget_bytes));
    fflush(stdout);

    if (!cfg->force && predicted > (double)budget_bytes) {
        printf("  skipped: the sessions do not fit\n");
        out->oom = true;
        return;
    }

    ds4_session *sessions[MAX_STREAMS * 2];
    memset(sessions, 0, sizeof(sessions));
    for (int i = 0; i < total_sessions; i++) {
        if (ds4_session_create(&sessions[i], env->engine, ctx_alloc) != 0 ||
            !sessions[i]) {
            printf("  skipped: session %d of %d did not fit\n", i + 1, total_sessions);
            free_sessions(sessions, total_sessions);
            out->oom = true;
            return;
        }
    }



    /* Prefill wave.  Prefills are serialized by the engine exactly as the
     * server serializes them, so the last stream's wait is the honest
     * time-to-first-token under load. */
    char err[256] = {0};
    const int span = env->corpus->len;
    const double wave_t0 = now_sec();
    double last_done = wave_t0;
    for (int i = 0; i < total_sessions; i++) {
        const int stream = cfg->verify ? i % session_count : i;
        const int off = stream_offset(stream, session_count, span, prompt_tokens);
        const ds4_tokens slice = {
            .v = env->corpus->v + off,
            .len = prompt_tokens,
            .cap = prompt_tokens,
        };
        if (ds4_session_sync(sessions[i], &slice, err, sizeof(err)) != 0) {
            fprintf(stderr, BENCH ": prefill of stream %d failed: %s\n", i, err);
            free_sessions(sessions, total_sessions);
            out->oom = true;
            return;
        }
        last_done = now_sec();
        if (i == concurrency - 1) out->prefill_last_ttft_s = last_done - wave_t0;
    }
    out->prefill_wall_s = last_done - wave_t0;
    if (out->prefill_wall_s > 0.0) {
        out->prefill_tps = (double)prompt_tokens * (double)total_sessions /
                           out->prefill_wall_s;
    }

    out->footprint_gib = out->predicted_gib;

    if (cfg->verify) {
        out->verify_ok = cfg->spec
            ? verify_spec_matches_plain(env, sessions, sessions + session_count, concurrency, cfg->gen)
            : verify_batch_matches_sequential(env, sessions, sessions + session_count, concurrency, cfg->gen);
        printf("  verify: %s\n", out->verify_ok ? (cfg->spec ? "speculative == plain greedy" : "batched == sequential")
                                                : "MISMATCH");
        out->ran = out->verify_ok;
        free_sessions(sessions, total_sessions);
        return;
    }

    if (measure_decode(env, sessions, concurrency, cfg->warmup,
                       false, NULL, NULL, NULL, NULL, NULL) != 0) {
        free_sessions(sessions, total_sessions);
        return;
    }

    double *step_ms = malloc((size_t)cfg->gen * sizeof(*step_ms));
    if (!step_ms) {
        free_sessions(sessions, total_sessions);
        return;
    }
    double eval_sec = 0.0, wall_sec = 0.0;
    long produced_tokens = 0;
    int *tokens = cfg->dump_tokens_path
        ? malloc((size_t)cfg->gen * (size_t)concurrency * sizeof(*tokens))
        : NULL;
    if (measure_decode(env, sessions, concurrency, cfg->gen,
                       true, step_ms, &eval_sec, &wall_sec, tokens, &produced_tokens) != 0) {
        free(tokens);
        free(step_ms);
        free_sessions(sessions, total_sessions);
        return;
    }
    /* Greedy decode, so any two runs of the same cell must agree token for
     * token.  Comparing dumps is how a memory-policy change proves it did not
     * disturb the arithmetic. */
    if (tokens) {
        FILE *fp = fopen(cfg->dump_tokens_path, "a");
        if (fp) {
            for (int i = 0; i < concurrency; i++) {
                fprintf(fp, "conc=%d ctx=%d stream=%d:", concurrency, ctx_tokens, i);
                for (int step = 0; step < cfg->gen; step++) {
                    fprintf(fp, " %d", tokens[step * concurrency + i]);
                }
                fputc('\n', fp);
            }
            fclose(fp);
        }
        free(tokens);
    }

    const double produced = (double)produced_tokens;
    out->decode_agg_tps = eval_sec > 0.0 ? produced / eval_sec : 0.0;
    if (cfg->spec) {
        printf("  spec: %.2f tokens per stream and cycle\n",
               produced / ((double)concurrency * (double)cfg->gen));
    }
    if (cfg->candidate_env) {
        double sec[2] = {0.0, 0.0}, tok[2] = {0.0, 0.0};
        for (int r = 0; r < cfg->repeat; r++) {
            /* ABBA: the second block reverses the order so any drift over the
             * pair cancels instead of favouring whichever ran first. */
            for (int slot = 0; slot < 2; slot++) {
                const bool candidate = (r % 2 == 0) ? slot == 1 : slot == 0;
                double block = 0.0;
                long block_tokens = 0;
                if (select_variant(cfg, candidate) != 0 ||
                    measure_decode(env, sessions, concurrency, cfg->gen,
                                   false, NULL, &block, NULL, NULL, &block_tokens) != 0) {
                    free(step_ms);
                    free_sessions(sessions, total_sessions);
                    return;
                }
                sec[candidate ? 1 : 0] += block;
                tok[candidate ? 1 : 0] += (double)block_tokens;
            }
        }
        out->control_tps = sec[0] > 0.0 ? tok[0] / sec[0] : 0.0;
        out->candidate_tps = sec[1] > 0.0 ? tok[1] / sec[1] : 0.0;
        (void)select_variant(cfg, false);
    }
    out->decode_wall_tps = wall_sec > 0.0 ? produced / wall_sec : 0.0;
    out->step_ms_mean = mean(step_ms, cfg->gen);
    qsort(step_ms, (size_t)cfg->gen, sizeof(*step_ms), compare_double);
    out->step_ms_p50 = percentile(step_ms, cfg->gen, 0.50);
    out->step_ms_p95 = percentile(step_ms, cfg->gen, 0.95);
    out->step_ms_max = step_ms[cfg->gen - 1];
    free(step_ms);

    if (cfg->mixed) {
        ds4_session *victim = sessions[concurrency];
        const int off = stream_offset(concurrency, session_count, span, prompt_tokens);
        const ds4_tokens slice = {
            .v = env->corpus->v + off,
            .len = prompt_tokens,
            .cap = prompt_tokens,
        };
        double *mixed_ms = malloc((size_t)cfg->mixed_steps * sizeof(*mixed_ms));
        if (mixed_ms) {
            /* The victim was prefilled with the whole slice above; rewind it
             * to the seed so the timed chunks have work to resume. */
            ds4_session_rewind(victim, MIXED_SEED_TOKENS);
            const ds4_tokens seed = {
                .v = slice.v,
                .len = MIXED_SEED_TOKENS,
                .cap = MIXED_SEED_TOKENS,
            };
            int done = 0, prefilled = 0;
            if (ds4_session_sync(victim, &seed, err, sizeof(err)) == 0 &&
                measure_mixed(env, sessions, concurrency, victim, &slice,
                              cfg->mixed_steps, cfg->mixed_quantum,
                              mixed_ms, &done, &prefilled) == 0 && done > 0) {
                double total = 0.0;
                for (int i = 0; i < done; i++) total += mixed_ms[i];
                qsort(mixed_ms, (size_t)done, sizeof(*mixed_ms), compare_double);
                out->mixed_step_ms_p50 = percentile(mixed_ms, done, 0.50);
                out->mixed_step_ms_p95 = percentile(mixed_ms, done, 0.95);
                if (total > 0.0) {
                    out->mixed_decode_tps =
                        (double)concurrency * (double)done / (total / 1e3);
                    out->mixed_prefill_tps = (double)prefilled / (total / 1e3);
                }
            } else {
                fprintf(stderr, BENCH ": mixed phase skipped\n");
            }
            free(mixed_ms);
        }
    }

    /* Full logit rows after a fixed number of greedy steps.  A change that is
     * meant to be arithmetically neutral - a memory policy, a buffer layout -
     * must leave these bit-identical; comparing two runs is how that is
     * checked. */
    if (cfg->dump_logits_dir) {
        float *row = malloc((size_t)env->vocab * sizeof(*row));
        for (int i = 0; row && i < concurrency; i++) {
            char path[1024];
            snprintf(path, sizeof(path), "%s/c%d_ctx%d_s%d.f32",
                     cfg->dump_logits_dir, concurrency, ctx_tokens, i);
            if (ds4_session_copy_logits(sessions[i], row, env->vocab) != env->vocab) continue;
            FILE *fp = fopen(path, "wb");
            if (!fp) continue;
            fwrite(row, sizeof(*row), (size_t)env->vocab, fp);
            fclose(fp);
        }
        free(row);
    }

    out->ran = true;
    free_sessions(sessions, total_sessions);

    printf("  prefill  %8.1f tok/s aggregate, last stream ready after %.2f s\n",
           out->prefill_tps, out->prefill_last_ttft_s);
    printf("  decode   %8.1f tok/s aggregate, %7.2f tok/s per stream\n",
           out->decode_agg_tps, out->decode_agg_tps / (double)concurrency);
    printf("  step     mean %.2f ms, p50 %.2f ms, p95 %.2f ms, max %.2f ms\n",
           out->step_ms_mean, out->step_ms_p50, out->step_ms_p95, out->step_ms_max);
    if (cfg->mixed && out->mixed_step_ms_p50 > 0.0) {
        printf("  mixed    step p50 %.2f ms, p95 %.2f ms; decode %.1f tok/s, "
               "prefill %.1f tok/s\n",
               out->mixed_step_ms_p50, out->mixed_step_ms_p95,
               out->mixed_decode_tps, out->mixed_prefill_tps);
    }
    if (cfg->candidate_env) {
        printf("  %s: control %.1f tok/s, candidate %.1f tok/s, %.2fx\n",
               cfg->candidate_env, out->control_tps, out->candidate_tps,
               out->control_tps > 0.0 ? out->candidate_tps / out->control_tps : 0.0);
    }
    printf("  %.2f GiB of session state\n", out->footprint_gib);
    fflush(stdout);
}

static void write_csv(const char *path,
                      const char *model,
                      const cell_result *cells,
                      int count) {
    const bool existed = access(path, F_OK) == 0;
    FILE *fp = fopen(path, "a");
    if (!fp) {
        fprintf(stderr, BENCH ": failed to open %s: %s\n", path, strerror(errno));
        return;
    }
    if (!existed) {
        fprintf(fp, "model,concurrency,ctx,status,prefill_tps,last_ttft_s,"
                    "decode_agg_tps,decode_stream_tps,decode_wall_tps,"
                    "step_ms_mean,step_ms_p50,step_ms_p95,step_ms_max,"
                    "mixed_step_ms_p50,mixed_step_ms_p95,mixed_decode_tps,"
                    "mixed_prefill_tps,estimate_gib,predicted_gib,footprint_gib\n");
    }
    for (int i = 0; i < count; i++) {
        const cell_result *c = &cells[i];
        fprintf(fp,
                "%s,%d,%d,%s,%.2f,%.3f,%.2f,%.2f,%.2f,"
                "%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.2f,%.2f,%.2f,%.2f,%.2f\n",
                model, c->concurrency, c->ctx,
                c->oom ? "nofit" : (c->ran ? "ok" : "error"),
                c->prefill_tps, c->prefill_last_ttft_s,
                c->decode_agg_tps,
                c->concurrency > 0 ? c->decode_agg_tps / (double)c->concurrency : 0.0,
                c->decode_wall_tps,
                c->step_ms_mean, c->step_ms_p50, c->step_ms_p95, c->step_ms_max,
                c->mixed_step_ms_p50, c->mixed_step_ms_p95,
                c->mixed_decode_tps, c->mixed_prefill_tps,
                c->estimate_gib, c->predicted_gib, c->footprint_gib);
    }
    fclose(fp);
}

static void print_summary(const cell_result *cells, int count) {
    printf("\n%-6s %-9s %-12s %-12s %-11s %-10s %s\n",
           "conc", "ctx", "decode t/s", "per stream", "step p95", "prefill", "state");
    for (int i = 0; i < count; i++) {
        const cell_result *c = &cells[i];
        if (!c->ran) {
            printf("%-6d %-9d %-12s %-12s %-11s %-10s %s\n",
                   c->concurrency, c->ctx, "-", "-", "-", "-",
                   c->oom ? "no fit" : "error");
            continue;
        }
        char agg[16], per[16], p95[16], pf[16];
        snprintf(agg, sizeof(agg), "%.1f", c->decode_agg_tps);
        snprintf(per, sizeof(per), "%.2f",
                 c->decode_agg_tps / (double)c->concurrency);
        snprintf(p95, sizeof(p95), "%.1f ms", c->step_ms_p95);
        snprintf(pf, sizeof(pf), "%.0f t/s", c->prefill_tps);
        printf("%-6d %-9d %-12s %-12s %-11s %-10s ok\n",
               c->concurrency, c->ctx, agg, per, p95, pf);
    }
}

int main(int argc, char **argv) {
    bench_config cfg = {
        .model_path = "ds4flash.gguf",
        .prompt_path = "speed-bench/promessi_sposi.txt",
        .csv_path = NULL,
        .gen = DEFAULT_GEN,
        .warmup = DEFAULT_WARMUP,
        .mixed_steps = DEFAULT_MIXED_STEPS,
        .mixed_quantum = DEFAULT_MIXED_QUANTUM,
        .reserve_gib = DEFAULT_RESERVE_GIB,
        .candidate_value = "1",
        .repeat = 2,
    };
    cfg.n_concurrency = parse_list("1,2,4,8,16", "--concurrency",
                                   cfg.concurrency, 1);
    cfg.n_ctx = parse_list("0,4096,16384,32768,65536,131072", "--ctx",
                           cfg.ctx, 0);

    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        if (!strcmp(arg, "-h") || !strcmp(arg, "--help")) {
            usage(stdout, argv[0]);
            return 0;
        } else if (!strcmp(arg, "-m") || !strcmp(arg, "--model")) {
            cfg.model_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--prompt-file")) {
            cfg.prompt_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--csv")) {
            cfg.csv_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--concurrency")) {
            cfg.n_concurrency = parse_list(need_arg(&i, argc, argv, arg), arg,
                                           cfg.concurrency, 1);
        } else if (!strcmp(arg, "--ctx")) {
            cfg.n_ctx = parse_list(need_arg(&i, argc, argv, arg), arg,
                                   cfg.ctx, 0);
        } else if (!strcmp(arg, "--gen")) {
            cfg.gen = parse_int_arg(need_arg(&i, argc, argv, arg), arg, 1);
        } else if (!strcmp(arg, "--warmup")) {
            cfg.warmup = parse_int_arg(need_arg(&i, argc, argv, arg), arg, 0);
        } else if (!strcmp(arg, "--prefill-chunk")) {
            cfg.prefill_chunk =
                (uint32_t)parse_int_arg(need_arg(&i, argc, argv, arg), arg, 1);
        } else if (!strcmp(arg, "--mixed")) {
            cfg.mixed = true;
        } else if (!strcmp(arg, "--mixed-steps")) {
            cfg.mixed_steps = parse_int_arg(need_arg(&i, argc, argv, arg), arg, 1);
        } else if (!strcmp(arg, "--mixed-quantum")) {
            cfg.mixed_quantum = parse_int_arg(need_arg(&i, argc, argv, arg), arg, 1);
        } else if (!strcmp(arg, "--verify")) {
            cfg.verify = true;
        } else if (!strcmp(arg, "--spec")) {
            cfg.spec = true;
        } else if (!strcmp(arg, "--warm-weights")) {
            cfg.warm_weights = true;
        } else if (!strcmp(arg, "--force")) {
            cfg.force = true;
        } else if (!strcmp(arg, "--private-transients")) {
            cfg.private_transients = true;
        } else if (!strcmp(arg, "--dump-tokens")) {
            cfg.dump_tokens_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--dump-logits-dir")) {
            cfg.dump_logits_dir = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--candidate-env")) {
            cfg.candidate_env = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--candidate-value")) {
            cfg.candidate_value = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--repeat")) {
            cfg.repeat = parse_int_arg(need_arg(&i, argc, argv, arg), arg, 1);
        } else if (!strcmp(arg, "--budget-gib")) {
            cfg.budget_gib = atof(need_arg(&i, argc, argv, arg));
        } else if (!strcmp(arg, "--reserve-gib")) {
            cfg.reserve_gib = atof(need_arg(&i, argc, argv, arg));
        } else {
            fprintf(stderr, BENCH ": unknown option %s\n", arg);
            usage(stderr, argv[0]);
            return 2;
        }
    }

    int max_conc = 0, max_ctx = 0;
    for (int i = 0; i < cfg.n_concurrency; i++) {
        if (cfg.concurrency[i] > max_conc) max_conc = cfg.concurrency[i];
    }
    for (int i = 0; i < cfg.n_ctx; i++) {
        if (cfg.ctx[i] > max_ctx) max_ctx = cfg.ctx[i];
    }
    if (max_conc > MAX_STREAMS) {
        fprintf(stderr, BENCH ": concurrency above %d is not supported\n", MAX_STREAMS);
        return 2;
    }

    const int64_t decode_budget = (int64_t)cfg.warmup + cfg.gen +
        (cfg.candidate_env ? (int64_t)cfg.gen * 2 * cfg.repeat : 0) +
        (cfg.mixed ? cfg.mixed_steps : 0);
    if (decode_budget > INT_MAX ||
        decode_budget * (cfg.spec ? 2 : 1) + (max_ctx > 0 ? max_ctx : 1) + CTX_MARGIN > INT_MAX) {
        fprintf(stderr, BENCH ": requested context and generation budget are too large\n");
        return 2;
    }
    const int max_alloc = (max_ctx > 0 ? max_ctx : 1) +
                          (int)decode_budget * (cfg.spec ? 2 : 1) + CTX_MARGIN;
    ds4_engine_options opt = {
        .model_path = cfg.model_path,
        .backend = DS4_BACKEND_METAL,
        .context_size = max_alloc,
        .prefill_chunk = cfg.prefill_chunk,
        .warm_weights = cfg.warm_weights,
        /* Mirror the server's batched startup so placement pricing and any
         * shared prefill scratch match what serving actually does. */
        .placement_session_count_hint = max_conc + (cfg.mixed ? 1 : 0),
        .share_session_prefill_workspace = !cfg.private_transients,
        .glm_mtp = cfg.spec,
    };
    ds4_engine *engine = NULL;
    if (ds4_engine_open(&engine, &opt) != 0) {
        fprintf(stderr, BENCH ": failed to open %s\n", cfg.model_path);
        return 1;
    }

    char *text = read_file(cfg.prompt_path);
    if (!text) {
        ds4_engine_close(engine);
        return 1;
    }
    ds4_tokens corpus = {0};
    ds4_tokenize_text(engine, text, &corpus);
    free(text);
    if (corpus.len < (max_ctx > 0 ? max_ctx : 1)) {
        fprintf(stderr,
                BENCH ": %s holds %d tokens, the sweep needs at least %d\n",
                cfg.prompt_path, corpus.len, max_ctx);
        ds4_tokens_free(&corpus);
        ds4_engine_close(engine);
        return 1;
    }

    const bench_env env = {
        .engine = engine,
        .corpus = &corpus,
        .eos = ds4_token_eos(engine),
        .vocab = ds4_engine_vocab_size(engine),
        .spec = cfg.spec,
    };

    cfg.budget_bytes = session_budget_bytes(&cfg);
    printf(BENCH ": %s, corpus %d tokens, %.2f GiB free for sessions "
           "after loading the weights\n",
           ds4_engine_model_name(engine), corpus.len,
           bytes_to_gib(cfg.budget_bytes));

    const int cell_count = cfg.n_ctx * cfg.n_concurrency;
    cell_result *cells = calloc((size_t)cell_count, sizeof(*cells));
    if (!cells) {
        ds4_tokens_free(&corpus);
        ds4_engine_close(engine);
        return 1;
    }

    int n = 0;
    for (int c = 0; c < cfg.n_ctx; c++) {
        const int prompt_tokens = cfg.ctx[c] > 0 ? cfg.ctx[c] : 1;
        if (corpus.len <= prompt_tokens) {
            fprintf(stderr,
                    BENCH ": corpus is not longer than context %d; every stream "
                    "will share one prompt and MoE routing\n", cfg.ctx[c]);
        }
        for (int k = 0; k < cfg.n_concurrency; k++) {
            run_cell(&cfg, &env, cfg.concurrency[k], cfg.ctx[c], &cells[n++]);
        }
    }

    print_summary(cells, n);
    if (cfg.csv_path) {
        write_csv(cfg.csv_path, ds4_engine_model_name(engine), cells, n);
    }

    bool failed = false;
    for (int i = 0; i < n; i++) {
        if (!cells[i].oom && !cells[i].ran) failed = true;
        if (cfg.verify && !cells[i].verify_ok) failed = true;
    }
    free(cells);
    ds4_tokens_free(&corpus);
    ds4_engine_close(engine);
    return failed ? 1 : 0;
}
