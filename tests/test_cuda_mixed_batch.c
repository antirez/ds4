/* Model-backed CUDA oracle for mixed prefill and multi-session decode.
 *
 * The control advances one prefill suffix and then a decode batch. The mixed
 * side advances the same rows through the native shared-MoE path. Every full
 * vocabulary logit must remain bit-exact. This is intentionally not part of
 * `make test`: it requires the large DeepSeek model and the eight-GPU TP/EP
 * configuration.
 *
 * Run with:
 *   DS4_TEST_MODEL=/path/to/model.gguf make test-cuda-mixed-batch
 *
 * Exercise compressed/indexed context with:
 *   DS4_TEST_CONTEXT=4096 DS4_TEST_MIXED_INITIAL=2048 \
 *   DS4_TEST_MIXED_ROUNDS=8 make test-cuda-mixed-batch
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
#define DEFAULT_CONTEXT 1024
#define DEFAULT_INITIAL 128
#define DEFAULT_QUANTUM 128
#define DEFAULT_ROUNDS 3

static const char *decode_prompts[] = {
    "Count upward using comma-separated integers.",
    "Write a C function and continue it in detail.",
    "List prime numbers without explanation.",
    "Describe a database transaction step by step.",
    "Produce a long JSON array of short strings.",
    "Explain TCP congestion control precisely.",
    "Write a long deterministic SQL migration.",
    "Enumerate powers of two with one per line.",
};

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

static void fail(const char *what, const char *err, int row, int round) {
    fprintf(stderr, "FAIL: %s row=%d round=%d%s%s\n",
            what, row, round, err && err[0] ? ": " : "",
            err && err[0] ? err : "");
    exit(1);
}

static int env_int(const char *name, int fallback, int min, int max) {
    const char *value = getenv(name);
    if (!value || !value[0]) return fallback;
    char *end = NULL;
    long parsed = strtol(value, &end, 10);
    if (end == value || *end != '\0' || parsed < min || parsed > max) {
        fprintf(stderr, "FAIL: %s must be in [%d, %d]\n", name, min, max);
        exit(1);
    }
    return (int)parsed;
}

static void compare_logits(ds4_session *mixed, ds4_session *control,
                           float *mixed_logits, float *control_logits,
                           int vocab, const char *kind, int row, int round) {
    if (ds4_session_copy_logits(mixed, mixed_logits, vocab) != vocab ||
        ds4_session_copy_logits(control, control_logits, vocab) != vocab) {
        fail("copy logits", NULL, row, round);
    }

    int different = 0;
    float worst = 0.0f;
    for (int i = 0; i < vocab; i++) {
        if (memcmp(&mixed_logits[i], &control_logits[i], sizeof(float)) != 0) {
            different++;
        }
        float delta = fabsf(mixed_logits[i] - control_logits[i]);
        if (!isfinite(delta)) delta = INFINITY;
        if (delta > worst) worst = delta;
    }
    const int mixed_argmax = ds4_session_argmax(mixed);
    const int control_argmax = ds4_session_argmax(control);
    if (different != 0 || mixed_argmax != control_argmax) {
        fprintf(stderr,
                "FAIL: %s logits row=%d round=%d differing=%d worst=%g "
                "argmax=%d/%d\n",
                kind, row, round, different, worst,
                mixed_argmax, control_argmax);
        exit(1);
    }
}

int main(void) {
    const char *model = getenv("DS4_TEST_MODEL");
    if (!model || !model[0]) fail("DS4_TEST_MODEL is not set", NULL, -1, -1);

    const int session_count = env_int(
            "DS4_TEST_SESSION_COUNT", DEFAULT_SESSION_COUNT, 3,
            MAX_SESSION_COUNT);
    const int context = env_int(
            "DS4_TEST_CONTEXT", DEFAULT_CONTEXT, DEFAULT_CONTEXT, 65536);
    const int initial = env_int(
            "DS4_TEST_MIXED_INITIAL", DEFAULT_INITIAL, 128, context - 1);
    const int quantum = env_int(
            "DS4_TEST_MIXED_QUANTUM", DEFAULT_QUANTUM, 1, context - 1);
    const int rounds = env_int(
            "DS4_TEST_MIXED_ROUNDS", DEFAULT_ROUNDS, 1, 64);
    if ((int64_t)initial + (int64_t)quantum * rounds >= context) {
        fail("mixed test frontier reaches context limit", NULL, -1, -1);
    }

    setenv("DS4_CUDA_MIXED_PREFILL_DECODE", "1", 1);

    const char *gpu_devices = getenv("DS4_TEST_GPU_DEVICES");
    if (!gpu_devices || !gpu_devices[0]) gpu_devices = "0,2,4,6,1,3,5,7";
    ds4_gpu_config gpu = {0};
    bool skip_cuda = false;
    char err[256] = {0};
    if (parse_gpu_vram_arg("auto", gpu_devices, &gpu, &skip_cuda,
                           err, sizeof(err)) != 0 || skip_cuda) {
        fail("GPU configuration", err, -1, -1);
    }

    ds4_engine_options options = {
        .model_path = model,
        .backend = DS4_BACKEND_CUDA,
        .n_threads = 1,
        .cuda_tensor_parallel = true,
        .share_session_prefill_workspace = true,
        .placement_ctx_hint = (uint32_t)context,
    };
    ds4_engine *engine = NULL;
    if (ds4_engine_create_with_gpu_config(&engine, &options, &gpu) != 0) {
        fail("engine open", NULL, -1, -1);
    }

    const char phrase[] = " numbered-token";
    const size_t repeat_count = (size_t)context * 2u;
    const size_t text_size = repeat_count * (sizeof(phrase) - 1u) + 1u;
    char *text = malloc(text_size);
    if (!text) fail("prefill text allocation", NULL, -1, -1);
    char *dst = text;
    for (size_t i = 0; i < repeat_count; i++) {
        memcpy(dst, phrase, sizeof(phrase) - 1u);
        dst += sizeof(phrase) - 1u;
    }
    *dst = '\0';

    ds4_tokens long_prompt = {0};
    ds4_tokenize_text(engine, text, &long_prompt);
    free(text);
    const int final_frontier = initial + quantum * rounds;
    if (long_prompt.len < final_frontier) {
        fail("generated prefill prompt is too short", NULL, -1, -1);
    }

    ds4_session *prefill_mixed = NULL;
    ds4_session *prefill_control = NULL;
    if (ds4_session_create(&prefill_mixed, engine, (uint32_t)context) != 0 ||
        ds4_session_create(&prefill_control, engine, (uint32_t)context) != 0) {
        fail("prefill session creation", NULL, -1, -1);
    }
    ds4_tokens prefix = long_prompt;
    prefix.len = initial;
    if (ds4_session_sync(prefill_mixed, &prefix, err, sizeof(err)) != 0 ||
        ds4_session_sync(prefill_control, &prefix, err, sizeof(err)) != 0) {
        fail("initial prefill", err, -1, -1);
    }

    ds4_session *mixed[MAX_SESSION_COUNT] = {0};
    ds4_session *control[MAX_SESSION_COUNT] = {0};
    ds4_tokens decode_prompt[MAX_SESSION_COUNT] = {0};
    const int prompt_count =
        (int)(sizeof(decode_prompts) / sizeof(decode_prompts[0]));
    for (int i = 0; i < session_count; i++) {
        ds4_encode_chat_prompt(
                engine, NULL, decode_prompts[i % prompt_count],
                DS4_THINK_NONE, &decode_prompt[i]);
        if (ds4_session_create(&mixed[i], engine, (uint32_t)context) != 0 ||
            ds4_session_create(&control[i], engine, (uint32_t)context) != 0 ||
            ds4_session_sync(mixed[i], &decode_prompt[i], err, sizeof(err)) != 0 ||
            ds4_session_sync(control[i], &decode_prompt[i], err, sizeof(err)) != 0) {
            fail("decode session setup", err, i, -1);
        }
    }

    const int vocab = ds4_engine_vocab_size(engine);
    float *mixed_logits = malloc((size_t)vocab * sizeof(*mixed_logits));
    float *control_logits = malloc((size_t)vocab * sizeof(*control_logits));
    if (!mixed_logits || !control_logits) {
        fail("logit allocation", NULL, -1, -1);
    }

    double separate_total = 0.0;
    double mixed_total = 0.0;
    const bool allow_fallback = getenv("DS4_TEST_ALLOW_FALLBACK") != NULL;
    bool all_native = true;
    for (int round = 0; round < rounds; round++) {
        ds4_decode_item mixed_items[MAX_SESSION_COUNT];
        ds4_decode_item control_items[MAX_SESSION_COUNT];
        for (int row = 0; row < session_count; row++) {
            const int i = (round & 1) != 0
                ? session_count - 1 - row : row;
            const int mixed_token = ds4_session_argmax(mixed[i]);
            const int control_token = ds4_session_argmax(control[i]);
            if (mixed_token != control_token) {
                fail("pre-step argmax mismatch", NULL, i, round);
            }
            mixed_items[row] = (ds4_decode_item){mixed[i], mixed_token};
            control_items[row] = (ds4_decode_item){control[i], control_token};
        }
        ds4_tokens target = long_prompt;
        target.len = initial + (round + 1) * quantum;

        double started = now_ms();
        if (ds4_session_sync(
                    prefill_control, &target, err, sizeof(err)) != 0 ||
            ds4_sessions_eval_batch(
                    control_items, session_count, err, sizeof(err)) != 0) {
            fail("serialized step", err, -1, round);
        }
        const double separate_ms = now_ms() - started;
        separate_total += separate_ms;

        const uint64_t native_before = ds4_test_mixed_native_count();
        started = now_ms();
        if (ds4_sessions_eval_batch_with_prefill(
                    mixed_items, session_count, prefill_mixed, &target,
                    err, sizeof(err)) != 0) {
            fail("mixed step", err, -1, round);
        }
        const double mixed_ms = now_ms() - started;
        mixed_total += mixed_ms;
        const uint64_t native_after = ds4_test_mixed_native_count();
        if (native_after != native_before + 1u) {
            all_native = false;
            if (!allow_fallback || native_after != native_before) {
                fail("mixed call used the serialized fallback", NULL, -1, round);
            }
        }

        compare_logits(prefill_mixed, prefill_control,
                       mixed_logits, control_logits, vocab,
                       "prefill", -1, round);
        for (int i = 0; i < session_count; i++) {
            compare_logits(mixed[i], control[i],
                           mixed_logits, control_logits, vocab,
                           "decode", i, round);
        }
        fprintf(stderr,
                "PASS: round=%d rows=%d+%d separate=%.3f ms mixed=%.3f ms "
                "speedup=%.3fx\n",
                round, quantum, session_count, separate_ms, mixed_ms,
                separate_ms / mixed_ms);
    }

    fprintf(stderr,
            "test_cuda_mixed_batch PASS context=%d initial=%d rounds=%d "
            "rows=%d+%d exact mode=%s speedup=%.3fx\n",
            context, initial, rounds, quantum, session_count,
            all_native ? "native" : "fallback",
            separate_total / mixed_total);

    free(control_logits);
    free(mixed_logits);
    for (int i = 0; i < session_count; i++) {
        ds4_tokens_free(&decode_prompt[i]);
        ds4_session_free(control[i]);
        ds4_session_free(mixed[i]);
    }
    ds4_session_free(prefill_control);
    ds4_session_free(prefill_mixed);
    ds4_tokens_free(&long_prompt);
    ds4_engine_close(engine);
    return 0;
}
