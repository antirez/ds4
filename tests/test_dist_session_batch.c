/* Model-backed oracle for distributed multi-session decode batching.
 *
 * Proves that ds4_dist_sessions_eval() -- the path ds4_sessions_eval_batch()
 * takes when every member is a coordinator session -- produces exactly the
 * logits the sessions would have produced one at a time.  That matters because
 * the batch submits every member's WORK frame before collecting any result, so
 * a member's remote step overlaps its peers' local slices; if that overlap
 * coupled two sessions' KV or paired a result with the wrong request, the
 * logits would drift and nothing else in the system would notice.
 *
 * Four sessions advance by default, cycling batch sizes 4, 3 and 2 so ragged
 * counts and the count==1 single-session fallback are both exercised.  Batch
 * member order is reversed on alternate steps to expose accidental slot
 * coupling.  Every full-logit frontier is archived, the batched sessions are
 * freed, and each prompt is then replayed through one isolated control session
 * whose decodes go one at a time.  Frontiers must match bit for bit.
 *
 * This is not part of `make test`: it needs the large model and a live worker.
 * The coordinator listens and the worker dials in, so the worker must already
 * be running and serving the complementary layer slice before this starts.
 *
 * The worker holds one session per session id it is sent and does not reclaim
 * one when the coordinator frees its side, so this test needs room for twice
 * DS4_TEST_SESSION_COUNT: the batched sessions and then the control sessions.
 * Start the worker with --max-sessions 2N or more.
 *
 * Run with, on the coordinator host:
 *   DS4_TEST_MODEL=/path/to/model.gguf make test-dist-session-batch
 *
 * Environment:
 *   DS4_TEST_MODEL          model path (required)
 *   DS4_TEST_DIST_LISTEN    coordinator listen host (default 192.168.88.239)
 *   DS4_TEST_DIST_PORT      coordinator listen port (default 9010)
 *   DS4_TEST_DIST_LAYERS    coordinator layer slice (default 0:20)
 *   DS4_TEST_SESSION_COUNT  batched sessions (default 4, max 8)
 *   DS4_TEST_CONTEXT        context size (default 1024)
 *   DS4_TEST_ROUTE_TIMEOUT  seconds to wait for the worker (default 120)
 */

#include "ds4.h"
#include "ds4_distributed.h"

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define DEFAULT_SESSION_COUNT 4
#define MAX_SESSION_COUNT 8
#define DECODE_STEPS 12
#define TEST_CTX 1024
#define DEFAULT_ROUTE_TIMEOUT 120

static const char *prompts[] = {
    "List the first twenty prime numbers and show no derivation.",
    "Describe how an LRU cache works using a short worked example.",
    "Write the first thirty Fibonacci numbers, one per line.",
    "Compare TCP and UDP using four precise operational examples.",
    "Explain B-tree insertion with a concrete sequence of ten keys.",
    "Write a compact C function that validates UTF-8.",
    "Generate SQL that creates and queries a small issue tracker schema.",
    "Explain how a CPU branch predictor works, briefly.",
};

static void fail(const char *what, int session, int step) {
    fprintf(stderr, "FAIL: %s session=%d step=%d\n", what, session, step);
    exit(1);
}

/* A distributed frontier must be bit-identical, not merely close: the batch
 * and the single-session path run the same kernels on the same slice, so any
 * difference at all means the batch perturbed state rather than lost accuracy.
 */
static void compare_frontier(ds4_session *control, const float *expected,
                             int expected_argmax, float *actual, int vocab,
                             int session, int step, int *nonexact) {
    if (ds4_session_copy_logits(control, actual, vocab) != vocab) {
        fail("copy control logits", session, step);
    }

    int different = 0;
    float max_abs = 0.0f;
    for (int i = 0; i < vocab; i++) {
        if (memcmp(&actual[i], &expected[i], sizeof(float)) != 0) different++;
        float d = fabsf(actual[i] - expected[i]);
        if (!isfinite(d)) d = INFINITY;
        if (d > max_abs) max_abs = d;
    }
    *nonexact += different;

    const int actual_argmax = ds4_session_argmax(control);
    if (different != 0 || actual_argmax != expected_argmax) {
        fprintf(stderr,
                "FAIL: logits mismatch session=%d step=%d control=%d batch=%d "
                "max_abs=%g differing=%d\n",
                session, step, actual_argmax, expected_argmax,
                max_abs, different);
        exit(1);
    }
}

/* Build the distributed options through the shared CLI parser rather than
 * filling the struct by hand, so the layer-range grammar this test accepts
 * cannot drift from the one the binaries accept. */
static void configure_distributed(ds4_engine_options *opt, const char *layers,
                                  const char *host, int port) {
    char port_buf[16];
    snprintf(port_buf, sizeof(port_buf), "%d", port);
    char *argv[] = {
        (char *)"--role",   (char *)"coordinator",
        (char *)"--layers", (char *)layers,
        (char *)"--listen", (char *)host, port_buf,
    };
    const int argc = (int)(sizeof(argv) / sizeof(argv[0]));

    char err[256] = {0};
    for (int i = 0; i < argc; i++) {
        const ds4_dist_cli_parse_result rc =
            ds4_dist_parse_cli_arg(argv[i], &i, argc, argv, &opt->distributed,
                                   err, sizeof(err));
        if (rc != DS4_DIST_CLI_MATCHED) {
            fprintf(stderr, "FAIL: distributed option '%s': %s\n", argv[i],
                    err[0] ? err : "not recognised");
            exit(1);
        }
    }
}

/* The worker dials us, so readiness is "a worker covering the rest of the
 * model has registered", not "the listener is bound". */
static void wait_for_route(ds4_session *probe, int timeout_sec) {
    char err[256] = {0};
    for (int waited = 0; waited < timeout_sec; waited++) {
        const int ready = ds4_session_distributed_route_ready(probe, err,
                                                              sizeof(err));
        if (ready == 1) return;
        if (ready < 0) {
            fprintf(stderr, "FAIL: distributed route: %s\n",
                    err[0] ? err : "configuration error");
            exit(1);
        }
        if (waited == 0) {
            fprintf(stderr,
                    "test_dist_session_batch: waiting for a worker to register"
                    " (%s)\n", err[0] ? err : "route incomplete");
        }
        sleep(1);
    }
    fprintf(stderr,
            "FAIL: no worker registered within %d s. Start one serving the\n"
            "      layers this coordinator does not, dialling this host, with\n"
            "      --max-sessions at least twice DS4_TEST_SESSION_COUNT.\n",
            timeout_sec);
    exit(1);
}

static int env_int(const char *name, int fallback) {
    const char *v = getenv(name);
    return (v && v[0]) ? atoi(v) : fallback;
}

int main(void) {
    const char *model = getenv("DS4_TEST_MODEL");
    if (!model || !model[0]) {
        fprintf(stderr, "FAIL: DS4_TEST_MODEL is not set\n");
        return 1;
    }
    const char *listen_host = getenv("DS4_TEST_DIST_LISTEN");
    if (!listen_host || !listen_host[0]) listen_host = "192.168.88.239";
    const char *layers = getenv("DS4_TEST_DIST_LAYERS");
    if (!layers || !layers[0]) layers = "0:20";
    const int listen_port = env_int("DS4_TEST_DIST_PORT", 9010);
    const int route_timeout = env_int("DS4_TEST_ROUTE_TIMEOUT",
                                      DEFAULT_ROUTE_TIMEOUT);

    const int session_count = env_int("DS4_TEST_SESSION_COUNT",
                                      DEFAULT_SESSION_COUNT);
    if (session_count < 2 || session_count > MAX_SESSION_COUNT) {
        fprintf(stderr, "FAIL: DS4_TEST_SESSION_COUNT must be 2..%d\n",
                MAX_SESSION_COUNT);
        return 1;
    }
    const int test_ctx = env_int("DS4_TEST_CONTEXT", TEST_CTX);
    if (test_ctx < TEST_CTX || test_ctx > 65536) {
        fprintf(stderr, "FAIL: DS4_TEST_CONTEXT must be %d..65536\n", TEST_CTX);
        return 1;
    }

    ds4_engine_options opt = {
        .model_path = model,
        .backend = DS4_BACKEND_CUDA, /* ROCm shares this backend id */
        .n_threads = 1,
        .context_size = test_ctx,
        .placement_ctx_hint = test_ctx,
        .placement_session_count_hint = session_count,
        .share_session_prefill_workspace = true,
    };
    configure_distributed(&opt, layers, listen_host, listen_port);

    char err[256] = {0};
    if (ds4_dist_prepare_engine_options(&opt.distributed, &opt,
                                        err, sizeof(err)) != 0) {
        fprintf(stderr, "FAIL: %s\n", err);
        return 1;
    }
    ds4_engine *engine = NULL;
    if (ds4_engine_open(&engine, &opt) != 0) {
        fprintf(stderr, "FAIL: engine open\n");
        return 1;
    }

    ds4_session *batched[MAX_SESSION_COUNT] = {0};
    ds4_tokens prompt[MAX_SESSION_COUNT] = {0};
    const int prompt_count = (int)(sizeof(prompts) / sizeof(prompts[0]));
    for (int i = 0; i < session_count; i++) {
        if (ds4_session_create(&batched[i], engine, test_ctx) != 0) {
            fail("session create", i, -1);
        }
    }
    if (!ds4_session_is_distributed(batched[0])) {
        fprintf(stderr,
                "FAIL: sessions are not distributed; check --role/--layers\n");
        return 1;
    }
    wait_for_route(batched[0], route_timeout);

    for (int i = 0; i < session_count; i++) {
        ds4_encode_chat_prompt(engine, NULL, prompts[i % prompt_count],
                               DS4_THINK_NONE, &prompt[i]);
        if (ds4_session_sync(batched[i], &prompt[i], err, sizeof(err)) != 0) {
            fprintf(stderr, "FAIL: prefill session=%d: %s\n", i, err);
            return 1;
        }
    }

    const int vocab = ds4_engine_vocab_size(engine);
    if (vocab <= 0) fail("engine reports no vocabulary", -1, -1);
    const size_t frontiers = (size_t)(DECODE_STEPS + 1) * (size_t)session_count;
    float *expected = malloc(frontiers * (size_t)vocab * sizeof(*expected));
    int *expected_argmax = malloc(frontiers * sizeof(*expected_argmax));
    float *actual = malloc((size_t)vocab * sizeof(*actual));
    if (!expected || !expected_argmax || !actual) fail("logit allocation", -1, -1);

    /* Cycle the group size so a batch is sometimes ragged and sometimes falls
     * to the count==1 path, which routes through the single-session code the
     * batch is being checked against. */
    static const int group_cycle[] = {4, 3, 2};
    int batched_calls = 0;
    for (int step = 0; step <= DECODE_STEPS; step++) {
        int tokens[MAX_SESSION_COUNT];
        for (int i = 0; i < session_count; i++) {
            const size_t frontier =
                (size_t)step * (size_t)session_count + (size_t)i;
            if (ds4_session_copy_logits(batched[i],
                                        expected + frontier * (size_t)vocab,
                                        vocab) != vocab) {
                fail("archive logits", i, step);
            }
            tokens[i] = ds4_session_argmax(batched[i]);
            expected_argmax[frontier] = tokens[i];
        }
        if (step == DECODE_STEPS) break;

        int group = group_cycle[step % 3];
        if (group > session_count) group = session_count;
        for (int base = 0; base < session_count; base += group) {
            ds4_decode_item items[MAX_SESSION_COUNT];
            const int rows = group < session_count - base
                ? group : session_count - base;
            for (int row = 0; row < rows; row++) {
                const int i = (step & 1) ? base + rows - 1 - row : base + row;
                items[row].session = batched[i];
                items[row].token = tokens[i];
            }
            if (ds4_sessions_eval_batch(items, rows, err, sizeof(err)) != 0) {
                fprintf(stderr, "FAIL: batch eval rows=%d base=%d step=%d: %s\n",
                        rows, base, step, err);
                return 1;
            }
            if (rows > 1) batched_calls++;
        }
    }

    for (int i = 0; i < session_count; i++) {
        ds4_session_free(batched[i]);
        batched[i] = NULL;
    }

    /* Guard against a vacuous pass: if every group had collapsed to one row,
     * the comparison below would be the single-session path against itself. */
    if (batched_calls == 0) {
        fprintf(stderr,
                "FAIL: no multi-row batch ran; the oracle would compare the\n"
                "      single-session path against itself\n");
        return 1;
    }

    /* TODO: two members failing inside one batch is still uncovered.  The
     * batch collects every submitted member even after a peer has failed and
     * then recovers each one through dist_coordinator_rebuild_from_transcript,
     * and that combination cannot be provoked from here: the only faults this
     * process controls are killing the worker, which fails every member rather
     * than a chosen two, or closing a route descriptor, which the plan owns.
     * Covering it needs a diagnostic hook in ds4_distributed.c -- an env-read
     * bitmask consulted once inside dist_coordinator_finish_remote_on_fd(),
     * failing the collect for the named members of the next batch only.  That
     * is a diagnostic switch validating the one release path, not a semantic
     * variant, so it fits the rules in AGENT.md. */

    int nonexact = 0;
    for (int i = 0; i < session_count; i++) {
        ds4_session *control = NULL;
        if (ds4_session_create(&control, engine, test_ctx) != 0) {
            fail("control session create", i, -1);
        }
        if (ds4_session_sync(control, &prompt[i], err, sizeof(err)) != 0) {
            fprintf(stderr,
                    "FAIL: control prefill session=%d: %s\n"
                    "      If this reports a worker session cap, restart the\n"
                    "      worker with --max-sessions at least %d.\n",
                    i, err, 2 * session_count);
            return 1;
        }
        for (int step = 0; step <= DECODE_STEPS; step++) {
            const size_t frontier =
                (size_t)step * (size_t)session_count + (size_t)i;
            compare_frontier(control, expected + frontier * (size_t)vocab,
                             expected_argmax[frontier], actual, vocab,
                             i, step, &nonexact);
            if (step < DECODE_STEPS &&
                ds4_session_eval(control, expected_argmax[frontier],
                                 err, sizeof(err)) != 0) {
                fprintf(stderr, "FAIL: control eval session=%d step=%d: %s\n",
                        i, step, err);
                return 1;
            }
        }
        ds4_session_free(control);
    }

    fprintf(stderr,
            "test_dist_session_batch PASS sessions=%d steps=%d "
            "batched_calls=%d nonexact_logits=%d\n",
            session_count, DECODE_STEPS, batched_calls, nonexact);

    free(actual);
    free(expected_argmax);
    free(expected);
    for (int i = 0; i < session_count; i++) ds4_tokens_free(&prompt[i]);
    ds4_engine_close(engine);
    return 0;
}
