/* Model-backed correctness oracle for native Metal session batching.
 *
 * Run with:
 *   DS4_TEST_MODEL=/path/to/model.gguf make test-metal-session-batch
 */

#include "ds4.h"
#include "ds4_tp.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define MAX_SESSION_COUNT 16
#define DECODE_STEPS 6
#define MIXED_SUFFIX_TOKENS 8
#define TEST_CTX 512

static const char *prompts[MAX_SESSION_COUNT] = {
    "Write the integers from 1 to 80, separated by commas.",
    "Explain a binary search using one compact worked example.",
    "Give three concise reasons to test concurrent model sessions.",
    "Write a four-line description of merge sort.",
    "List five prime numbers and briefly define a prime number.",
    "Explain the difference between a stack and a queue in two sentences.",
    "Give a compact example of hexadecimal notation.",
    "Describe one invariant of a binary search tree.",
};

static void fail(const char *what, int session, int step) {
    fprintf(stderr, "FAIL: %s session=%d step=%d\n", what, session, step);
    exit(1);
}

static ds4_tp_transport tp_transport_from_env(void) {
    const char *value = getenv("DS4_TEST_TP_TRANSPORT");
    if (!value || !value[0] || strcmp(value, "auto") == 0) {
        return DS4_TP_TRANSPORT_AUTO;
    }
    if (strcmp(value, "tcp") == 0) return DS4_TP_TRANSPORT_TCP;
    if (strcmp(value, "rdma") == 0) return DS4_TP_TRANSPORT_RDMA;
    fprintf(stderr, "FAIL: invalid DS4_TEST_TP_TRANSPORT=%s\n", value);
    exit(1);
}

static int tp_port_from_env(void) {
    const char *value = getenv("DS4_TEST_TP_PORT");
    if (!value || !value[0]) return 19452;
    char *end = NULL;
    long port = strtol(value, &end, 10);
    if (end == value || *end != '\0' || port < 1 || port > 65535) {
        fprintf(stderr, "FAIL: invalid DS4_TEST_TP_PORT=%s\n", value);
        exit(1);
    }
    return (int)port;
}

static int session_count_from_env(void) {
    const char *value = getenv("DS4_TEST_SESSION_COUNT");
    if (!value || !value[0]) return 2;
    char *end = NULL;
    long count = strtol(value, &end, 10);
    if (end == value || *end != '\0' || count < 2 ||
        count > MAX_SESSION_COUNT) {
        fprintf(stderr, "FAIL: invalid DS4_TEST_SESSION_COUNT=%s\n", value);
        exit(1);
    }
    return (int)count;
}

static void archive_logits(ds4_session *session, float *dst, int vocab,
                           int session_id, int step) {
    if (ds4_session_copy_logits(session, dst, vocab) != vocab) {
        fail("copy logits", session_id, step);
    }
}

static void compare_logits(ds4_session *session, const float *expected,
                           float *actual, int vocab, int expected_argmax,
                           int session_id, int step) {
    archive_logits(session, actual, vocab, session_id, step);
    float max_abs = 0.0f;
    int different = 0;
    int low_different = 0;
    int high_different = 0;
    for (int i = 0; i < vocab; i++) {
        if (memcmp(&actual[i], &expected[i], sizeof(float)) != 0) {
            different++;
            if (i < vocab / 2) low_different++;
            else high_different++;
        }
        float d = fabsf(actual[i] - expected[i]);
        if (!isfinite(d)) d = FLT_MAX;
        if (d > max_abs) max_abs = d;
    }
    int actual_argmax = ds4_session_argmax(session);
    if (different != 0 || actual_argmax != expected_argmax) {
        fprintf(stderr,
                "FAIL: logits mismatch session=%d step=%d expected_top=%d "
                "actual_top=%d differing=%d low=%d high=%d max_abs=%g\n",
                session_id, step, expected_argmax, actual_argmax,
                different, low_different, high_different, max_abs);
        exit(1);
    }
}

int main(void) {
    const char *model = getenv("DS4_TEST_MODEL");
    if (!model || !model[0]) {
        fprintf(stderr, "FAIL: DS4_TEST_MODEL is not set\n");
        return 1;
    }
    setenv("DS4_METAL_SESSION_BATCH_LOG", "1", 1);
    const int session_count = session_count_from_env();

    const char *tp_mode = getenv("DS4_TEST_TP_MODE");
    const bool tp_leader = tp_mode && strcmp(tp_mode, "leader") == 0;
    const bool tp_worker = tp_mode && strcmp(tp_mode, "worker") == 0;
    if (tp_mode && tp_mode[0] && !tp_leader && !tp_worker) {
        fprintf(stderr, "FAIL: invalid DS4_TEST_TP_MODE=%s\n", tp_mode);
        return 1;
    }
    const int tp_port = tp_port_from_env();
    ds4_engine_options opt = {
        .model_path = model,
        .backend = DS4_BACKEND_METAL,
        .n_threads = 1,
        .context_size = TEST_CTX,
    };
    if (tp_leader) {
        opt.tp.role = DS4_TP_LEADER;
        opt.tp.listen_host = getenv("DS4_TEST_TP_LISTEN_HOST");
        if (!opt.tp.listen_host || !opt.tp.listen_host[0]) {
            opt.tp.listen_host = "0.0.0.0";
        }
        opt.tp.listen_port = tp_port;
        opt.tp.transport = tp_transport_from_env();
    } else if (tp_worker) {
        opt.tp.role = DS4_TP_WORKER;
        opt.tp.leader_host = getenv("DS4_TEST_TP_LEADER_HOST");
        if (!opt.tp.leader_host || !opt.tp.leader_host[0]) {
            fprintf(stderr, "FAIL: DS4_TEST_TP_LEADER_HOST is required for worker mode\n");
            return 1;
        }
        opt.tp.leader_port = tp_port;
        opt.tp.transport = tp_transport_from_env();
    }
    ds4_engine *engine = NULL;
    if (ds4_engine_open(&engine, &opt) != 0) fail("engine open", -1, -1);

    if (tp_worker) {
        const int worker_rc = ds4_tp_worker_run(engine, &opt.tp);
        ds4_engine_close(engine);
        return worker_rc;
    }

    ds4_tp *tp = NULL;
    if (tp_leader) {
        char tp_err[256] = "";
        ds4_tp_identity identity = {
            .gguf_bytes = ds4_engine_model_bytes(engine),
            .model_id = (uint32_t)ds4_engine_model_id(engine),
            .n_layer = (uint32_t)ds4_engine_layer_count(engine),
            .n_embd = (uint32_t)ds4_engine_embd_dim(engine),
            .n_vocab = (uint32_t)ds4_engine_vocab_size(engine),
            .quant_bits = (uint32_t)ds4_engine_routed_quant_bits(engine),
            .ctx_size = TEST_CTX,
        };
        ds4_engine_tp_gate_schedule(engine,
                                    &identity.gate_slot_start,
                                    &identity.gate_slot_step,
                                    &identity.gates_per_token);
        if (!ds4_tp_create(&tp, &opt.tp, &identity,
                           tp_err, sizeof(tp_err)) ||
            !ds4_engine_tp_bind(engine, tp, tp_err, sizeof(tp_err))) {
            fprintf(stderr, "FAIL: TP leader setup: %s\n", tp_err);
            return 1;
        }
    }

    ds4_tokens prompt[MAX_SESSION_COUNT] = {0};
    ds4_session *batched[MAX_SESSION_COUNT] = {0};
    char err[256] = {0};
    for (int i = 0; i < session_count; i++) {
        ds4_encode_chat_prompt(engine, NULL, prompts[i % 8], DS4_THINK_NONE,
                               &prompt[i]);
        if (ds4_session_create(&batched[i], engine, TEST_CTX) != 0) {
            fail("session create", i, -1);
        }
        if (ds4_session_sync(batched[i], &prompt[i], err, sizeof(err)) != 0) {
            fprintf(stderr, "FAIL: prefill session=%d: %s\n", i, err);
            return 1;
        }
    }

    if (tp_leader && getenv("DS4_TEST_TP_DISCONNECT")) {
        ds4_decode_item items[MAX_SESSION_COUNT];
        for (int i = 0; i < session_count; i++) {
            items[i].session = batched[i];
            items[i].token = ds4_session_argmax(batched[i]);
        }
        fprintf(stderr, "TP_DISCONNECT_READY\n");
        fflush(stderr);
        usleep(1000 * 1000);
        err[0] = '\0';
        if (ds4_sessions_eval_batch(items, session_count,
                                    err, sizeof(err)) == 0) {
            fail("disconnect batch unexpectedly succeeded", -1, -1);
        }
        for (int i = 0; i < session_count; i++) {
            if (ds4_session_pos(batched[i]) != 0) {
                fail("disconnect did not invalidate checkpoint", i, -1);
            }
            ds4_session_free(batched[i]);
            ds4_tokens_free(&prompt[i]);
        }
        ds4_engine_close(engine);
        ds4_tp_free(tp);
        fprintf(stderr,
                "test_metal_session_batch DISCONNECT PASS invalidated=%d err=%s\n",
                session_count, err[0] ? err : "unknown");
        return 0;
    }

    const int vocab = ds4_engine_vocab_size(engine);
    const size_t frontier_count =
        (size_t)session_count * (DECODE_STEPS + 1u);
    float *expected = malloc(frontier_count * (size_t)vocab * sizeof(float));
    float *actual = malloc((size_t)vocab * sizeof(float));
    int *argmax = malloc(frontier_count * sizeof(int));
    int generated[MAX_SESSION_COUNT][DECODE_STEPS];
    if (!expected || !actual || !argmax) fail("oracle allocation", -1, -1);

#define FRONTIER(step_, session_) \
    ((size_t)(step_) * (size_t)session_count + (size_t)(session_))
    for (int i = 0; i < session_count; i++) {
        size_t f = FRONTIER(0, i);
        archive_logits(batched[i], expected + f * (size_t)vocab,
                       vocab, i, 0);
        argmax[f] = ds4_session_argmax(batched[i]);
    }

    for (int step = 0; step < DECODE_STEPS; step++) {
        ds4_decode_item items[MAX_SESSION_COUNT];
        for (int row = 0; row < session_count; row++) {
            int i = (step & 1) ? session_count - 1 - row : row;
            int token = ds4_session_argmax(batched[i]);
            generated[i][step] = token;
            items[row].session = batched[i];
            items[row].token = token;
        }
        if (ds4_sessions_eval_batch(items, session_count,
                                    err, sizeof(err)) != 0) {
            fprintf(stderr, "FAIL: batch step=%d: %s\n", step, err);
            return 1;
        }
        for (int i = 0; i < session_count; i++) {
            size_t f = FRONTIER(step + 1, i);
            archive_logits(batched[i], expected + f * (size_t)vocab,
                           vocab, i, step + 1);
            argmax[f] = ds4_session_argmax(batched[i]);
        }
    }
    for (int i = 0; i < session_count; i++) {
        ds4_session_free(batched[i]);
    }

    ds4_tokens mixed_prompt = {0};
    ds4_tokens suffix = {0};
    ds4_tokens_copy(&mixed_prompt, &prompt[0]);
    ds4_tokenize_text(engine,
                      " Continue with a concise verification example and conclusion.",
                      &suffix);
    if (suffix.len < MIXED_SUFFIX_TOKENS) {
        fail("mixed suffix tokenization", -1, -1);
    }
    for (int i = 0; i < MIXED_SUFFIX_TOKENS; i++) {
        ds4_tokens_push(&mixed_prompt, suffix.v[i]);
    }

    ds4_session *mixed_prefill = NULL;
    ds4_session *mixed_decode[MAX_SESSION_COUNT] = {0};
    float *mixed_expected = malloc(
            (size_t)(session_count + 1) * (size_t)vocab * sizeof(float));
    int mixed_argmax[MAX_SESSION_COUNT + 1];
    if (!mixed_expected) fail("mixed oracle allocation", -1, -1);
    if (ds4_session_create(&mixed_prefill, engine, TEST_CTX) != 0) {
        fail("mixed prefill create", -1, -1);
    }
    if (ds4_session_sync(mixed_prefill, &prompt[0], err, sizeof(err)) != 0) {
        fprintf(stderr, "FAIL: mixed base prefill: %s\n", err);
        return 1;
    }
    ds4_decode_item mixed_items[MAX_SESSION_COUNT];
    for (int i = 0; i < session_count; i++) {
        if (ds4_session_create(&mixed_decode[i], engine, TEST_CTX) != 0) {
            fail("mixed decode create", i, -1);
        }
        if (ds4_session_sync(mixed_decode[i], &prompt[i], err, sizeof(err)) != 0) {
            fprintf(stderr, "FAIL: mixed decode prefill session=%d: %s\n", i, err);
            return 1;
        }
        mixed_items[i].session = mixed_decode[i];
        mixed_items[i].token = ds4_session_argmax(mixed_decode[i]);
    }
    if (ds4_sessions_eval_batch_with_prefill(
                mixed_items, session_count, mixed_prefill, &mixed_prompt,
                err, sizeof(err)) != 0) {
        fprintf(stderr, "FAIL: Metal mixed batch: %s\n", err);
        return 1;
    }

    archive_logits(mixed_prefill, mixed_expected, vocab, -1, -1);
    mixed_argmax[0] = ds4_session_argmax(mixed_prefill);
    for (int i = 0; i < session_count; i++) {
        archive_logits(mixed_decode[i],
                       mixed_expected + (size_t)(i + 1) * (size_t)vocab,
                       vocab, i, -1);
        mixed_argmax[i + 1] = ds4_session_argmax(mixed_decode[i]);
    }

    ds4_session *mixed_control = NULL;
    if (ds4_session_create(&mixed_control, engine, TEST_CTX) != 0) {
        fail("mixed prefill control create", -1, -1);
    }
    if (ds4_session_sync(mixed_control, &prompt[0], err, sizeof(err)) != 0 ||
        ds4_session_sync(mixed_control, &mixed_prompt, err, sizeof(err)) != 0) {
        fprintf(stderr, "FAIL: mixed prefill control: %s\n", err);
        return 1;
    }
    compare_logits(mixed_control, mixed_expected, actual, vocab,
                   mixed_argmax[0],
                   -1, -1);
    if (ds4_session_pos(mixed_prefill) != mixed_prompt.len ||
        ds4_session_pos(mixed_control) != mixed_prompt.len) {
        fail("mixed prefill checkpoint", -1, -1);
    }
    ds4_session_free(mixed_control);

    for (int i = 0; i < session_count; i++) {
        mixed_control = NULL;
        if (ds4_session_create(&mixed_control, engine, TEST_CTX) != 0) {
            fail("mixed decode control create", i, -1);
        }
        if (ds4_session_sync(mixed_control, &prompt[i], err, sizeof(err)) != 0 ||
            ds4_session_eval(mixed_control, mixed_items[i].token,
                             err, sizeof(err)) != 0) {
            fprintf(stderr, "FAIL: mixed decode control session=%d: %s\n", i, err);
            return 1;
        }
        compare_logits(mixed_control,
                       mixed_expected + (size_t)(i + 1) * (size_t)vocab,
                       actual, vocab, mixed_argmax[i + 1], i, -1);
        if (ds4_session_pos(mixed_decode[i]) != prompt[i].len + 1 ||
            ds4_session_pos(mixed_control) != prompt[i].len + 1) {
            fail("mixed decode checkpoint", i, -1);
        }
        ds4_session_free(mixed_control);
        ds4_session_free(mixed_decode[i]);
    }
    ds4_session_free(mixed_prefill);
    free(mixed_expected);
    ds4_tokens_free(&suffix);
    ds4_tokens_free(&mixed_prompt);

    for (int i = 0; i < session_count; i++) {
        ds4_session *control = NULL;
        if (ds4_session_create(&control, engine, TEST_CTX) != 0) {
            fail("control create", i, -1);
        }
        if (ds4_session_sync(control, &prompt[i], err, sizeof(err)) != 0) {
            fprintf(stderr, "FAIL: control prefill session=%d: %s\n", i, err);
            return 1;
        }
        for (int step = 0; step <= DECODE_STEPS; step++) {
            size_t f = FRONTIER(step, i);
            compare_logits(control, expected + f * (size_t)vocab,
                           actual, vocab, argmax[f], i, step);
            if (step < DECODE_STEPS &&
                ds4_session_eval(control, generated[i][step],
                                 err, sizeof(err)) != 0) {
                fprintf(stderr,
                        "FAIL: control eval session=%d step=%d: %s\n",
                        i, step, err);
                return 1;
            }
        }
        ds4_session_free(control);
        ds4_tokens_free(&prompt[i]);
    }

    free(argmax);
    free(actual);
    free(expected);
    if (tp) (void)ds4_tp_send_stop(tp);
    ds4_engine_close(engine);
    ds4_tp_free(tp);
    fprintf(stderr,
            "test_metal_session_batch PASS sessions=%d steps=%d mixed_suffix=%d exact_logits=1\n",
            session_count, DECODE_STEPS, MIXED_SUFFIX_TOKENS);
    return 0;
#undef FRONTIER
}
