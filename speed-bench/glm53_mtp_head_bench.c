#define _POSIX_C_SOURCE 200809L
#include "ds4.h"

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum {
    BENCH_BLOCKS = 3,
    BENCH_TOKENS = 512,
    BENCH_CHUNK = 64,
};

static const char *const rollback_env =
    "DS4_GLM_MTP_DISABLE_DEFERRED_ROW1_HEAD";

typedef struct {
    double elapsed;
    int singles;
    int doubles;
    int rejections;
    int cycle_count;
    unsigned char cycle_sizes[BENCH_TOKENS];
    int ids[BENCH_TOKENS];
} bench_arm;

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static bool select_rollback(bool rollback) {
    return rollback ? setenv(rollback_env, "1", 1) == 0 :
                      unsetenv(rollback_env) == 0;
}

static bool advance_arm(ds4_session *session,
                        bool rollback,
                        int offset,
                        int count,
                        bench_arm *arm) {
    if (!select_rollback(rollback)) return false;
    char err[256] = {0};
    const double start = now_sec();
    int generated = 0;
    while (generated < count) {
        const int first = ds4_session_argmax(session);
        int accepted[2] = {0};
        const int cap = count - generated < 2 ? 1 : 2;
        const int n = ds4_session_eval_speculative_argmax(
            session, first, cap, -1, accepted, 2, err, sizeof(err));
        if (n <= 0 || n > cap) {
            fprintf(stderr, "glm53-mtp-head-bench: generation failed at %d: %s\n",
                    offset + generated, err[0] ? err : "unknown error");
            return false;
        }
        const bool draft_was_seeded = arm->cycle_count != 0;
        if (arm->cycle_count >= BENCH_TOKENS) return false;
        arm->cycle_sizes[arm->cycle_count++] = (unsigned char)n;
        arm->singles += n == 1;
        arm->doubles += n == 2;
        arm->rejections += draft_was_seeded && cap == 2 && n == 1;
        for (int i = 0; i < n; i++) {
            arm->ids[offset + generated + i] = accepted[i];
        }
        generated += n;
    }
    arm->elapsed += now_sec() - start;
    return true;
}

static bool run_block(ds4_engine *engine,
                      const ds4_tokens *prompt,
                      int block,
                      bench_arm *optimized,
                      bench_arm *rollback) {
    ds4_session *optimized_session = NULL;
    ds4_session *rollback_session = NULL;
    float *optimized_logits = NULL;
    float *rollback_logits = NULL;
    char err[256] = {0};
    bool ok = ds4_session_create(&optimized_session, engine, 8192) == 0 &&
              ds4_session_create(&rollback_session, engine, 8192) == 0 &&
              ds4_session_sync(optimized_session, prompt,
                               err, sizeof(err)) == 0 &&
              ds4_session_sync(rollback_session, prompt,
                               err, sizeof(err)) == 0;
    if (!ok) {
        fprintf(stderr, "glm53-mtp-head-bench: session setup failed: %s\n",
                err[0] ? err : "unknown error");
        goto done;
    }

    const int vocab = ds4_engine_vocab_size(engine);
    if (vocab <= 0) {
        fprintf(stderr, "glm53-mtp-head-bench: invalid vocabulary size\n");
        ok = false;
        goto done;
    }
    optimized_logits = malloc((size_t)vocab * sizeof(optimized_logits[0]));
    rollback_logits = malloc((size_t)vocab * sizeof(rollback_logits[0]));
    if (!optimized_logits || !rollback_logits) {
        fprintf(stderr, "glm53-mtp-head-bench: logit allocation failed\n");
        ok = false;
        goto done;
    }

    for (int offset = 0, chunk = 0; offset < BENCH_TOKENS;
         offset += BENCH_CHUNK, chunk++) {
        const bool rollback_first = ((block + chunk) & 1) != 0;
        if (rollback_first) {
            ok = advance_arm(rollback_session, true, offset, BENCH_CHUNK,
                             rollback) &&
                 advance_arm(optimized_session, false, offset, BENCH_CHUNK,
                             optimized);
        } else {
            ok = advance_arm(optimized_session, false, offset, BENCH_CHUNK,
                             optimized) &&
                 advance_arm(rollback_session, true, offset, BENCH_CHUNK,
                             rollback);
        }
        if (!ok) goto done;

        const size_t generated_bytes =
            (size_t)(offset + BENCH_CHUNK) * sizeof(optimized->ids[0]);
        ok = memcmp(optimized->ids, rollback->ids, generated_bytes) == 0 &&
             optimized->cycle_count == rollback->cycle_count &&
             memcmp(optimized->cycle_sizes, rollback->cycle_sizes,
                    (size_t)optimized->cycle_count *
                        sizeof(optimized->cycle_sizes[0])) == 0 &&
             optimized->singles == rollback->singles &&
             optimized->doubles == rollback->doubles &&
             optimized->rejections == rollback->rejections &&
             ds4_session_pos(optimized_session) ==
                 ds4_session_pos(rollback_session) &&
             ds4_session_copy_logits(optimized_session,
                                     optimized_logits, vocab) == vocab &&
             ds4_session_copy_logits(rollback_session,
                                     rollback_logits, vocab) == vocab &&
             memcmp(optimized_logits, rollback_logits,
                    (size_t)vocab * sizeof(optimized_logits[0])) == 0;
        if (!ok) {
            fprintf(stderr,
                    "glm53-mtp-head-bench: paired state mismatch "
                    "block=%d offset=%d\n",
                    block + 1, offset + BENCH_CHUNK);
            goto done;
        }
        printf("block=%d chunk=%d order=%s optimized_s=%.6f "
               "rollback_s=%.6f\n",
               block + 1, chunk + 1,
               rollback_first ? "rollback/optimized" :
                                "optimized/rollback",
               optimized->elapsed, rollback->elapsed);
        fflush(stdout);
    }
    if (optimized->rejections == 0 || optimized->doubles == 0) {
        fprintf(stderr,
                "glm53-mtp-head-bench: block %d did not exercise both "
                "deferred-head outcomes\n",
                block + 1);
        ok = false;
    }

done:
    free(rollback_logits);
    free(optimized_logits);
    ds4_session_free(rollback_session);
    ds4_session_free(optimized_session);
    return ok;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s MODEL\n", argv[0]);
        return 2;
    }
    const char *old_rollback = getenv(rollback_env);
    char *saved_rollback = old_rollback ? strdup(old_rollback) : NULL;
    if (old_rollback && !saved_rollback) return 1;

    ds4_engine_options opt = {
        .model_path = argv[1],
        .backend = DS4_BACKEND_METAL,
        .prefill_chunk = 0,
        .power_percent = 100,
        .warm_weights = true,
        .glm_mtp = true,
    };
    ds4_engine *engine = NULL;
    ds4_tokens prompt = {0};
    int rc = 1;
    if (ds4_engine_open(&engine, &opt) != 0) goto done;
    ds4_encode_chat_prompt(
        engine, NULL,
        "Explain, in detailed numbered steps, how an append-only write-ahead "
        "log, checkpoints, and idempotent replay recover a stateful service "
        "after a crash. Continue until every invariant and edge case is covered.",
        DS4_THINK_NONE, &prompt);
    if (prompt.len == 0) goto done;

    bench_arm optimized[BENCH_BLOCKS] = {0};
    bench_arm rollback[BENCH_BLOCKS] = {0};
    double optimized_total = 0.0;
    double rollback_total = 0.0;
    for (int block = 0; block < BENCH_BLOCKS; block++) {
        if (!run_block(engine, &prompt, block,
                       &optimized[block], &rollback[block])) {
            goto done;
        }
        optimized_total += optimized[block].elapsed;
        rollback_total += rollback[block].elapsed;
        const double optimized_tps =
            BENCH_TOKENS / optimized[block].elapsed;
        const double rollback_tps =
            BENCH_TOKENS / rollback[block].elapsed;
        printf("block=%d result optimized=%.6f rollback=%.6f "
               "percent=%.4f single=%d double=%d rejected=%d\n",
               block + 1, optimized_tps, rollback_tps,
               (optimized_tps / rollback_tps - 1.0) * 100.0,
               optimized[block].singles, optimized[block].doubles,
               optimized[block].rejections);
    }
    const double optimized_tps =
        (double)(BENCH_TOKENS * BENCH_BLOCKS) / optimized_total;
    const double rollback_tps =
        (double)(BENCH_TOKENS * BENCH_BLOCKS) / rollback_total;
    printf("aggregate optimized=%.6f rollback=%.6f percent=%.4f\n",
           optimized_tps, rollback_tps,
           (optimized_tps / rollback_tps - 1.0) * 100.0);
    rc = 0;

done:
    if (saved_rollback) {
        (void)setenv(rollback_env, saved_rollback, 1);
    } else {
        (void)unsetenv(rollback_env);
    }
    free(saved_rollback);
    ds4_tokens_free(&prompt);
    ds4_engine_close(engine);
    return rc;
}
