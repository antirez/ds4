/* Model-backed CUDA oracle for multi-session decode batching.
 *
 * Eight test sessions advance by default, alternating batch sizes 8, 4, and 2.
 * Odd session counts advance as one batch to exercise ragged row grids. Every
 * full-logit frontier is archived, the batched sessions are freed, and each
 * prompt is replayed
 * through one isolated control session. Batch order is reversed on alternate
 * steps to expose accidental row/slot coupling. This is intentionally not part
 * of `make test`: it requires the large DeepSeek model and eight CUDA devices
 * used by the TP/EP setup.
 *
 * Run with:
 *   DS4_TEST_MODEL=/path/to/model.gguf make test-cuda-session-batch
 */

#include "ds4.h"
#include "ds4_gpu_args.h"
#include "ds4_gpu_mgpu.h"

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define DEFAULT_SESSION_COUNT 8
#define MAX_SESSION_COUNT 16
#define DECODE_STEPS 24
#define TEST_CTX 1024

static const char *prompts[] = {
    "Write the integers from 1 to 200, separated by commas. Do not stop early.",
    "Write a compact C function that validates UTF-8, then explain each branch.",
    "List the first one hundred prime numbers and show no derivation.",
    "Describe how an LRU cache works using a detailed worked example.",
    "Write the first eighty Fibonacci numbers, one per line.",
    "Explain B-tree insertion with a concrete sequence of twenty keys.",
    "Generate SQL that creates and queries a small issue tracker schema.",
    "Compare TCP and UDP using six precise operational examples.",
};

static uint64_t hash_bytes(const void *data, size_t len) {
    const unsigned char *p = data;
    uint64_t hash = UINT64_C(1469598103934665603);
    for (size_t i = 0; i < len; i++) {
        hash ^= p[i];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

static void fail(const char *what, int session, int step) {
    fprintf(stderr, "FAIL: %s session=%d step=%d\n", what, session, step);
    exit(1);
}

static void compare_frontier(ds4_session *control, const float *expected,
                             int expected_argmax, float *actual, int vocab,
                             int session, int step,
                             float *worst_abs, int *nonexact) {
    if (ds4_session_copy_logits(control, actual, vocab) != vocab) {
        fail("copy logits", session, step);
    }

    float max_abs = 0.0f;
    int different = 0;
    for (int i = 0; i < vocab; i++) {
        float d = fabsf(actual[i] - expected[i]);
        if (memcmp(&actual[i], &expected[i], sizeof(float)) != 0) different++;
        if (!isfinite(d)) d = INFINITY;
        if (d > max_abs) max_abs = d;
    }
    if (max_abs > *worst_abs) *worst_abs = max_abs;
    *nonexact += different;

    int actual_argmax = ds4_session_argmax(control);
    if (different != 0 || actual_argmax != expected_argmax) {
        fprintf(stderr,
                "FAIL: logits mismatch session=%d step=%d control=%d batch=%d "
                "max_abs=%g differing=%d\n",
                session, step, actual_argmax, expected_argmax,
                max_abs, different);
        exit(1);
    }
}

int main(void) {
    const char *model = getenv("DS4_TEST_MODEL");
    if (!model || !model[0]) {
        fprintf(stderr, "FAIL: DS4_TEST_MODEL is not set\n");
        return 1;
    }

    int session_count = DEFAULT_SESSION_COUNT;
    const char *session_count_env = getenv("DS4_TEST_SESSION_COUNT");
    if (session_count_env && session_count_env[0]) {
        session_count = atoi(session_count_env);
    }
    if (session_count < 2 || session_count > MAX_SESSION_COUNT) {
        fprintf(stderr,
                "FAIL: DS4_TEST_SESSION_COUNT must be between 2 and %d\n",
                MAX_SESSION_COUNT);
        return 1;
    }

    int test_ctx = TEST_CTX;
    const char *test_ctx_env = getenv("DS4_TEST_CONTEXT");
    if (test_ctx_env && test_ctx_env[0]) test_ctx = atoi(test_ctx_env);
    if (test_ctx < TEST_CTX || test_ctx > 65536) {
        fprintf(stderr,
                "FAIL: DS4_TEST_CONTEXT must be between %d and 65536\n",
                TEST_CTX);
        return 1;
    }

    int long_words = 0;
    const char *long_words_env = getenv("DS4_TEST_LONG_WORDS");
    if (long_words_env && long_words_env[0]) long_words = atoi(long_words_env);
    if (long_words < 0 || long_words > test_ctx - 128) {
        fprintf(stderr,
                "FAIL: DS4_TEST_LONG_WORDS must leave at least 128 context tokens\n");
        return 1;
    }
    char *long_prompt = NULL;
    if (long_words != 0) {
        const char word[] = " token";
        const size_t word_len = sizeof(word) - 1u;
        long_prompt = malloc((size_t)long_words * word_len + 1u);
        if (!long_prompt) fail("long prompt allocation", -1, -1);
        char *dst = long_prompt;
        for (int i = 0; i < long_words; i++) {
            memcpy(dst, word, word_len);
            dst += word_len;
        }
        *dst = '\0';
    }

    setenv("DS4_CUDA_SESSION_BATCH_MOE", "1", 1);

    ds4_gpu_config gpu_cfg = {0};
    bool skip_cuda = false;
    char err[256] = {0};
    if (parse_gpu_vram_arg("auto", "0,2,4,6,1,3,5,7",
                           &gpu_cfg, &skip_cuda, err, sizeof(err)) != 0 ||
        skip_cuda) {
        fprintf(stderr, "FAIL: GPU configuration: %s\n", err);
        return 1;
    }

    ds4_engine_options opt = {
        .model_path = model,
        .backend = DS4_BACKEND_CUDA,
        .n_threads = 1,
        .cuda_tensor_parallel = true,
        .share_session_prefill_workspace = true,
        .placement_ctx_hint = (uint32_t)test_ctx,
    };
    ds4_engine *engine = NULL;
    if (ds4_engine_create_with_gpu_config(&engine, &opt, &gpu_cfg) != 0) {
        fprintf(stderr, "FAIL: engine open\n");
        return 1;
    }

    ds4_session *batched[MAX_SESSION_COUNT] = {0};
    ds4_tokens prompt[MAX_SESSION_COUNT] = {0};
    const int prompt_count = (int)(sizeof(prompts) / sizeof(prompts[0]));
    for (int i = 0; i < session_count; i++) {
        const char *prompt_text = long_prompt && (i & 1) == 0
            ? long_prompt : prompts[i % prompt_count];
        ds4_encode_chat_prompt(engine, NULL, prompt_text, DS4_THINK_NONE,
                               &prompt[i]);
        if (long_prompt && (i & 1) == 0) {
            fprintf(stderr,
                    "test_cuda_session_batch long prompt session=%d tokens=%d\n",
                    i, prompt[i].len);
            if (prompt[i].len + DECODE_STEPS >= test_ctx) {
                fail("long prompt exceeds decode context", i, -1);
            }
        }
        if (ds4_session_create(&batched[i], engine, (uint32_t)test_ctx) != 0) {
            fail("session create", i, -1);
        }
        if (ds4_session_sync(batched[i], &prompt[i], err, sizeof(err)) != 0) {
            fprintf(stderr, "FAIL: prefill session=%d: %s\n", i, err);
            return 1;
        }
    }

    const int vocab = ds4_engine_vocab_size(engine);
    const size_t frontiers = (size_t)(DECODE_STEPS + 1) * (size_t)session_count;
    float *expected = malloc(frontiers * (size_t)vocab * sizeof(*expected));
    int *expected_argmax = malloc(frontiers * sizeof(*expected_argmax));
    float *actual = malloc((size_t)vocab * sizeof(*actual));
    if (!expected || !expected_argmax || !actual) {
        fail("logit allocation", -1, -1);
    }

    double batch_ms[MAX_SESSION_COUNT + 1] = {0};
    int batch_calls[MAX_SESSION_COUNT + 1] = {0};
    uint64_t evaluated_rows = 0;
    double evaluated_ms = 0.0;
    const bool power_of_two = (session_count & (session_count - 1)) == 0;
    for (int step = 0; step <= DECODE_STEPS; step++) {
        int tokens[MAX_SESSION_COUNT];
        for (int i = 0; i < session_count; i++) {
            const size_t frontier =
                (size_t)step * (size_t)session_count + (size_t)i;
            if (ds4_session_copy_logits(
                    batched[i], expected + frontier * (size_t)vocab,
                    vocab) != vocab) {
                fail("archive logits", i, step);
            }
            tokens[i] = ds4_session_argmax(batched[i]);
            expected_argmax[frontier] = tokens[i];
        }
        if (step == DECODE_STEPS) break;

        const int group = !power_of_two ? session_count :
                          step % 3 == 0 ? session_count :
                          step % 3 == 1 ? session_count / 2 :
                                          session_count / 4 > 1
                                              ? session_count / 4 : 2;
        for (int base = 0; base < session_count; base += group) {
            ds4_decode_item items[MAX_SESSION_COUNT];
            const int rows = group < session_count - base
                ? group : session_count - base;
            for (int row = 0; row < rows; row++) {
                int i = (step & 1) ? base + rows - 1 - row : base + row;
                items[row].session = batched[i];
                items[row].token = tokens[i];
            }
            const double started = now_ms();
            const int eval_rc = ds4_sessions_eval_batch(items, rows,
                                                        err, sizeof(err));
            const double elapsed = now_ms() - started;
            if (eval_rc != 0) {
                fprintf(stderr,
                        "FAIL: batch eval size=%d base=%d step=%d: %s\n",
                        rows, base, step, err);
                return 1;
            }
            batch_ms[rows] += elapsed;
            batch_calls[rows]++;
            evaluated_rows += (uint64_t)rows;
            evaluated_ms += elapsed;
        }
    }
    fprintf(stderr,
            "decode batch timing: rows=%llu total=%.3f ms aggregate=%.1f tok/s\n",
            (unsigned long long)evaluated_rows, evaluated_ms,
            evaluated_ms > 0.0 ? (double)evaluated_rows * 1000.0 / evaluated_ms : 0.0);
    for (int rows = 2; rows <= session_count; rows++) {
        if (batch_calls[rows] != 0) {
            fprintf(stderr,
                    "  batch=%d calls=%d mean=%.3f ms aggregate=%.1f tok/s\n",
                    rows, batch_calls[rows],
                    batch_ms[rows] / batch_calls[rows],
                    (double)rows * 1000.0 * batch_calls[rows] / batch_ms[rows]);
        }
    }
    for (int i = 0; i < session_count; i++) {
        ds4_session_free(batched[i]);
        batched[i] = NULL;
    }

    if (getenv("DS4_TEST_BATCH_ONLY") != NULL) {
        const size_t logits_bytes =
            frontiers * (size_t)vocab * sizeof(*expected);
        fprintf(stderr,
                "test_cuda_session_batch PASS batch-only sessions=%d steps=%d "
                "frontier0_hash=%016llx logit_hash=%016llx "
                "argmax_hash=%016llx\n",
                session_count, DECODE_STEPS,
                (unsigned long long)hash_bytes(
                    expected,
                    (size_t)session_count * (size_t)vocab * sizeof(*expected)),
                (unsigned long long)hash_bytes(expected, logits_bytes),
                (unsigned long long)hash_bytes(
                    expected_argmax, frontiers * sizeof(*expected_argmax)));
        free(actual);
        free(expected_argmax);
        free(expected);
        for (int i = 0; i < session_count; i++) {
            ds4_tokens_free(&prompt[i]);
        }
        free(long_prompt);
        ds4_engine_close(engine);
        return 0;
    }

    float worst_abs = 0.0f;
    int nonexact = 0;
    for (int i = 0; i < session_count; i++) {
        ds4_session *control = NULL;
        if (ds4_session_create(&control, engine, (uint32_t)test_ctx) != 0 ||
            ds4_session_sync(control, &prompt[i], err, sizeof(err)) != 0) {
            fprintf(stderr, "FAIL: control prefill session=%d: %s\n", i, err);
            return 1;
        }
        for (int step = 0; step <= DECODE_STEPS; step++) {
            const size_t frontier =
                (size_t)step * (size_t)session_count + (size_t)i;
            compare_frontier(control,
                             expected + frontier * (size_t)vocab,
                             expected_argmax[frontier], actual, vocab,
                             i, step, &worst_abs, &nonexact);
            if (step < DECODE_STEPS &&
                ds4_session_eval(control, expected_argmax[frontier],
                                 err, sizeof(err)) != 0) {
                fprintf(stderr,
                        "FAIL: control eval session=%d step=%d: %s\n",
                        i, step, err);
                return 1;
            }
        }
        ds4_session_free(control);
    }

    fprintf(stderr,
            "test_cuda_session_batch PASS sessions=%d steps=%d "
            "worst_logit_abs=%g nonexact_logits=%d\n",
            session_count, DECODE_STEPS, worst_abs, nonexact);

    free(actual);
    free(expected_argmax);
    free(expected);
    for (int i = 0; i < session_count; i++) {
        ds4_tokens_free(&prompt[i]);
    }
    free(long_prompt);
    ds4_engine_close(engine);
    return 0;
}
