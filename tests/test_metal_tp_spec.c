/* Physical two-rank DSpark oracle; the peer runs the ordinary ./ds4 worker.
 * Check committed tokens against serial target logits, then append to the
 * live speculative cache and repeat across compression boundaries. */
#include "ds4.h"
#include "ds4_tp.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* A larger session must not borrow an undersized shared workspace. Freeing
 * its first borrower must not release the engine-owned scratch either. */
static int check_workspaces(ds4_engine *engine) {
    ds4_session *small = NULL, *large = NULL, *other = NULL;
    ds4_tokens prompt = {0};
    char err[256] = "";
    const int vocab = ds4_engine_vocab_size(engine);
    float *expected = malloc((size_t)vocab * sizeof(float));
    float *actual = malloc((size_t)vocab * sizeof(float));
    int ok = 0;
    ds4_encode_chat_prompt(engine, NULL, "Explain what a database transaction is.",
                           DS4_THINK_NONE, &prompt);
    if (!expected || !actual || !prompt.len || prompt.len >= 256 ||
        ds4_session_create(&small, engine, 256) ||
        ds4_session_create(&large, engine, 8192) ||
        ds4_session_create(&other, engine, 256)) goto done;
    if (ds4_session_sync(small, &prompt, err, sizeof(err)) ||
        ds4_session_copy_logits(small, expected, vocab) != vocab ||
        ds4_session_sync(large, &prompt, err, sizeof(err)) ||
        ds4_session_sync(other, &prompt, err, sizeof(err)) ||
        ds4_session_copy_logits(other, actual, vocab) != vocab ||
        memcmp(expected, actual, (size_t)vocab * sizeof(float))) goto done;
    ds4_session_free(small);
    small = NULL;
    const int token = ds4_session_argmax(other);
    if (ds4_session_eval(other, token, err, sizeof(err))) goto done;
    ds4_tokens_push(&prompt, token);
    if (ds4_session_create(&small, engine, 256) ||
        ds4_session_sync(small, &prompt, err, sizeof(err)) ||
        ds4_session_copy_logits(small, expected, vocab) != vocab ||
        ds4_session_sync(large, &prompt, err, sizeof(err)) ||
        ds4_session_copy_logits(large, expected, vocab) != vocab ||
        ds4_session_copy_logits(other, actual, vocab) != vocab) goto done;
    if (memcmp(expected, actual, (size_t)vocab * sizeof(float))) goto done;
    ok = 1;
done:
    fprintf(stderr, "TP shared workspace lifetime and mixed capacities: %s %s\n",
            ok ? "PASS" : "FAIL", err);
    ds4_session_free(other);
    ds4_session_free(large);
    ds4_session_free(small);
    ds4_tokens_free(&prompt);
    free(actual);
    free(expected);
    return ok;
}

static int check_prefix(ds4_engine *engine, int prefix) {
    ds4_session *spec = NULL, *ref = NULL;
    ds4_tokens prompt = {0}, text = {0}, filler = {0};
    char err[256] = "";
    float worst_gap = 0;
    int max_chunk = 0, generated = 0, ok = 0;
    ds4_encode_chat_prompt(engine, NULL,
        "Write a complete C hash table implementation with string keys, insert, "
        "find, delete, and a test main. Output only C code.", DS4_THINK_NONE, &text);
    ds4_tokenize_text(engine, "/* Handle collisions and release allocated memory. */\n", &filler);
    if (!text.len || text.len > prefix || !filler.len) goto done;
    while (prompt.len < prefix - text.len)
        ds4_tokens_push(&prompt, filler.v[prompt.len % filler.len]);
    for (int i = 0; i < text.len; i++) ds4_tokens_push(&prompt, text.v[i]);
    if (ds4_session_create(&spec, engine, 8192) ||
        ds4_session_create(&ref, engine, 8192)) goto done;
    for (int phase = 0; phase < 2; phase++) {
        if (phase) {
            ds4_tokens_free(&text);
            ds4_tokenize_text(engine, "\n/* Continue with deletion and cleanup tests. */\n", &text);
            for (int i = 0; i < text.len; i++) ds4_tokens_push(&prompt, text.v[i]);
        }
        if (ds4_session_sync(spec, &prompt, err, sizeof(err)) ||
            ds4_session_sync(ref, &prompt, err, sizeof(err))) goto done;
        int n = 0;
        while (n < 128) {
            int accepted[16];
            const int count = ds4_session_eval_speculative_argmax_ignoring_eos(
                spec, ds4_session_argmax(spec), 128 - n, ds4_token_eos(engine),
                DS4_THINK_NONE, accepted, 16, err, sizeof(err));
            if (count <= 0 || count > 128 - n || count > 16) goto done;
            if (count > max_chunk) max_chunk = count;
            for (int i = 0; i < count; i++) {
                ds4_token_score top, score;
                if (ds4_session_top_logprobs(ref, &top, 1) != 1 ||
                    ds4_session_token_logprob(ref, accepted[i], &score) != 1) goto done;
                const float gap = top.logit - score.logit;
                if (!isfinite(gap) || gap > 2.0f) {
                    fprintf(stderr, "FAIL prefix=%d phase=%d token=%d gap=%g\n", prefix, phase, n+i, gap);
                    goto done;
                }
                if (gap > worst_gap) worst_gap = gap;
                if (ds4_session_eval(ref, accepted[i], err, sizeof(err))) goto done;
                ds4_tokens_push(&prompt, accepted[i]);
            }
            n += count;
            generated += count;
        }
    }
    ok = max_chunk > 1;
done:
    fprintf(stderr, "TP DSpark prefix=%d generated=%d max_chunk=%d worst_gap=%g: %s %s\n",
            prefix, generated, max_chunk, worst_gap, ok ? "PASS" : "FAIL", err);
    ds4_session_free(ref);
    ds4_session_free(spec);
    ds4_tokens_free(&prompt);
    ds4_tokens_free(&text);
    ds4_tokens_free(&filler);
    return ok;
}

static int check_request_limits(ds4_engine *engine) {
    const int limits[] = {1, 2, 3, 4, 5, 9, 10};
    ds4_tokens prompt = {0}, filler = {0};
    ds4_session *session = NULL;
    char err[256] = "";
    int ok = 0;
    ds4_tokenize_text(engine, "Keep the explanation short. ", &filler);
    if (!filler.len) goto done;
    for (unsigned i = 0; i < sizeof(limits) / sizeof(*limits); i++) {
        for (int mode = 0; mode < 3; mode++) {
            prompt.len = 0;
            const int prefix = 256 - limits[i];
            for (int j = 0; j < prefix; j++)
                ds4_tokens_push(&prompt, filler.v[j % filler.len]);
            if (ds4_session_create(&session, engine, 256) ||
                ds4_session_sync(session, &prompt, err, sizeof(err))) goto done;
            const int seed = ds4_session_argmax(session);
            const int stop = mode == 2 ? seed : ds4_token_eos(engine);
            const int capacity = mode == 1 ? 1 : 16;
            int tokens[18];
            for (unsigned j = 0; j < 18; j++) tokens[j] = -1;
            const int n = ds4_session_eval_speculative_argmax(session, seed,
                limits[i], stop, tokens + 1, capacity, err, sizeof(err));
            if (n < 1 || n > capacity || n > limits[i] ||
                tokens[1] != seed || tokens[0] != -1 || tokens[n + 1] != -1 ||
                (mode != 0 && n != 1) || !ds4_session_checkpoint_valid(session) ||
                ds4_session_pos(session) != prefix + n) goto done;
            const ds4_tokens *history = ds4_session_tokens(session);
            if (!history || history->len != prefix + n ||
                memcmp(history->v, prompt.v, (size_t)prefix * sizeof(int)) ||
                memcmp(history->v + prefix, tokens + 1, (size_t)n * sizeof(int))) goto done;
            ds4_session_free(session);
            session = NULL;
        }
    }
    ok = 1;
done:
    fprintf(stderr, "TP DSpark request limits, stop tokens and context boundary: %s %s\n",
            ok ? "PASS" : "FAIL", err);
    ds4_session_free(session);
    ds4_tokens_free(&prompt);
    ds4_tokens_free(&filler);
    return ok;
}

int main(int argc, char **argv) {
    bool cuda = false, tcp = false;
    if (argc < 6 || argc > 9) {
        fprintf(stderr, "usage: %s MODEL SUPPORT LISTEN_HOST PORT RDMA_DEVICE [GID [--cuda] [--tcp]]\n", argv[0]);
        return 2;
    }
    for (int i = 7; i < argc; i++) {
        if (!strcmp(argv[i], "--cuda") && !cuda) cuda = true;
        else if (!strcmp(argv[i], "--tcp") && !tcp) tcp = true;
        else return 2;
    }
    char *end = NULL;
    const long port = strtol(argv[4], &end, 10);
    if (end == argv[4] || *end || port < 1 || port > 65535) {
        fprintf(stderr, "invalid port: %s\n", argv[4]);
        return 2;
    }
    long gid = 1;
    if (argc >= 7) {
        gid = strtol(argv[6], &end, 10);
        if (end == argv[6] || *end || gid < 0 || gid > 255) return 2;
    }
    ds4_engine_options opt = {
        .model_path = argv[1], .mtp_path = argv[2], .dspark = true,
        .backend = cuda ? DS4_BACKEND_CUDA : DS4_BACKEND_METAL,
        .n_threads = 1, .context_size = 8192,
        .share_session_prefill_workspace = true,
        .prefill_chunk = cuda ? 2048 : 0,
        .tp = {.role = DS4_TP_LEADER, .listen_host = argv[3],
               .listen_port = (int)port,
               .transport = tcp ? DS4_TP_TRANSPORT_TCP : DS4_TP_TRANSPORT_RDMA,
               .rdma_device = argv[5], .rdma_gid_index = (int)gid, .rdma_gid_index_set = true},
    };
    ds4_engine *engine = NULL;
    ds4_tp *tp = NULL;
    char err[256] = "";
    int ok = ds4_engine_open(&engine, &opt) == 0;
    if (ok) {
        ds4_tp_identity id = {
            .gguf_bytes = ds4_engine_model_bytes(engine),
            .model_id = ds4_engine_model_id(engine),
            .n_layer = ds4_engine_layer_count(engine),
            .n_embd = ds4_engine_embd_dim(engine),
            .n_vocab = ds4_engine_vocab_size(engine),
            .quant_bits = ds4_engine_routed_quant_bits(engine), .ctx_size = 8192,
        };
        ds4_engine_tp_gate_schedule(engine, &id.gate_slot_start, &id.gate_slot_step,
                                   &id.gates_per_token, id.gate_slot_mask);
        ok = ds4_tp_create(&tp, &opt.tp, &id, err, sizeof(err)) &&
             ds4_engine_tp_bind(engine, tp, err, sizeof(err));
    }
    /* Cross the indexer top-k boundary inside a verify batch and just after
     * prefill, as well as the short and longer compressed-attention cases. */
    if (ok) ok = check_workspaces(engine) && check_request_limits(engine) &&
                 check_prefix(engine, 127) && check_prefix(engine, 2046) &&
                 check_prefix(engine, 2048) && check_prefix(engine, 4095);
    if (tp) (void)ds4_tp_send_stop(tp);
    ds4_engine_close(engine);
    ds4_tp_free(tp);
    if (!ok) fprintf(stderr, "TP DSpark oracle failed: %s\n", err);
    return ok ? 0 : 1;
}
