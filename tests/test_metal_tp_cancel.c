/* Run on two dedicated Metal hosts; the peer is the ordinary ds4 worker. */
#include "../ds4.h"
#include "../ds4_tp.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "line %d: %s (%s)\n", __LINE__, #x, err); goto done; \
} } while (0)

typedef struct { unsigned events; int frontier; bool stopped; } interruption;

static void progress(void *ud, const char *event, int current, int total) {
    interruption *p = ud;
    (void)event; (void)total;
    p->events++;
    if (current >= p->frontier) p->stopped = true;
}

static bool cancelled(void *ud) { return ((interruption *)ud)->stopped; }

int main(int argc, char **argv) {
    if (argc != 7) {
        fprintf(stderr, "usage: %s MODEL PROMPT LISTEN_HOST PORT RDMA_DEVICE GID\n", argv[0]);
        return 2;
    }
    int rc = 1;
    char err[256] = "", *text = NULL;
    float *expected = NULL, *actual = NULL;
    FILE *file = NULL;
    ds4_engine *engine = NULL;
    ds4_session *control = NULL, *subject = NULL;
    ds4_tp *tp = NULL;
    ds4_tokens tokens = {0};
    ds4_session_snapshot snapshot = {0};
    ds4_engine_options opt = {.model_path = argv[1], .backend = DS4_BACKEND_METAL,
        .context_size = 65536, .power_percent = 100};
    opt.tp = (ds4_tp_options){.role = DS4_TP_LEADER, .requested = true,
        .listen_host = argv[3], .listen_port = atoi(argv[4]),
        .transport = DS4_TP_TRANSPORT_RDMA, .rdma_device = argv[5],
        .rdma_gid_index = atoi(argv[6]), .rdma_gid_index_set = true};
    CHECK((file = fopen(argv[2], "rb")) != NULL);
    CHECK(fseek(file, 0, SEEK_END) == 0);
    long bytes = ftell(file);
    CHECK(bytes > 0 && bytes < 100000000 && fseek(file, 0, SEEK_SET) == 0);
    CHECK((text = malloc((size_t)bytes + 1u)) != NULL);
    CHECK(fread(text, 1, (size_t)bytes, file) == (size_t)bytes);
    text[bytes] = '\0';
    fclose(file); file = NULL;
    CHECK(ds4_engine_open(&engine, &opt) == 0);
    ds4_tp_identity id = {.gguf_bytes = ds4_engine_model_bytes(engine),
        .model_id = (uint32_t)ds4_engine_model_id(engine),
        .n_layer = (uint32_t)ds4_engine_layer_count(engine),
        .n_embd = (uint32_t)ds4_engine_embd_dim(engine),
        .n_vocab = (uint32_t)ds4_engine_vocab_size(engine),
        .quant_bits = (uint32_t)ds4_engine_routed_quant_bits(engine), .ctx_size = 65536};
    ds4_engine_tp_gate_schedule(engine, &id.gate_slot_start, &id.gate_slot_step,
                                 &id.gates_per_token, id.gate_slot_mask);
    CHECK(ds4_tp_create(&tp, &opt.tp, &id, err, sizeof(err)));
    CHECK(ds4_engine_tp_bind(engine, tp, err, sizeof(err)));
    CHECK(ds4_session_create(&control, engine, opt.context_size) == 0);
    CHECK(ds4_session_create(&subject, engine, opt.context_size) == 0);
    ds4_encode_chat_prompt(engine, NULL, text, DS4_THINK_NONE, &tokens);
    CHECK(tokens.len > 32769);
    const int vocab = ds4_engine_vocab_size(engine);
    expected = malloc((size_t)vocab * sizeof(*expected));
    actual = malloc((size_t)vocab * sizeof(*actual));
    CHECK(expected && actual);
    const int lengths[] = {31, 32, 255, 256, 8192, 32768, 1026, 1057, 32768};
    for (unsigned i = 0; i < sizeof(lengths) / sizeof(*lengths); i++) {
        ds4_session_invalidate(subject);
        const int prefix = i >= 6 ? 1025 : 0;
        if (prefix) {
            tokens.len = prefix;
            CHECK(ds4_session_sync(subject, &tokens, err, sizeof(err)) == 0);
        }
        tokens.len = lengths[i];
        interruption p = {.stopped = i == 0,
            .frontier = prefix + (lengths[i] - prefix + 1) / 2};
        ds4_session_set_progress(subject, progress, &p);
        ds4_session_set_display_progress(subject, progress, &p);
        ds4_session_set_cancel(subject, cancelled, &p);
        CHECK(ds4_session_sync(subject, &tokens, err, sizeof(err)) == DS4_SESSION_SYNC_INTERRUPTED);
        CHECK(!ds4_tp_failed(tp) && p.stopped && (i == 0 || p.events > 0));
        CHECK(ds4_session_save_snapshot(subject, &snapshot, err, sizeof(err)) != 0);
        ds4_session_snapshot_free(&snapshot);
        ds4_session_set_cancel(subject, NULL, NULL);
        ds4_session_set_progress(subject, NULL, NULL);
        ds4_session_set_display_progress(subject, NULL, NULL);
        tokens.len = 1025;
        ds4_session_invalidate(control);
        CHECK(ds4_session_sync(control, &tokens, err, sizeof(err)) == 0);
        CHECK(ds4_session_sync(subject, &tokens, err, sizeof(err)) == 0);
        for (unsigned step = 0; step < 2; step++) {
            if (step) {
                CHECK(ds4_session_eval(control, tokens.v[tokens.len], err, sizeof(err)) == 0);
                CHECK(ds4_session_eval(subject, tokens.v[tokens.len], err, sizeof(err)) == 0);
            }
            CHECK(ds4_session_copy_logits(control, expected, vocab) == vocab);
            CHECK(ds4_session_copy_logits(subject, actual, vocab) == vocab);
            CHECK(memcmp(expected, actual, (size_t)vocab * sizeof(*actual)) == 0);
            for (int v = 0; v < vocab; v++) CHECK(isfinite(actual[v]));
        }
        printf("cancel %d -> %d tokens, same-connection rebuild and next decode: exact PASS\n",
               prefix, lengths[i]);
        fflush(stdout);
    }
    rc = 0;
done:
    if (subject) {
        ds4_session_set_cancel(subject, NULL, NULL);
        ds4_session_set_progress(subject, NULL, NULL);
        ds4_session_set_display_progress(subject, NULL, NULL);
    }
    if (file) fclose(file);
    ds4_session_snapshot_free(&snapshot);
    ds4_session_free(subject); ds4_session_free(control);
    if (tp && !ds4_tp_failed(tp)) ds4_tp_send_stop(tp);
    ds4_engine_close(engine); ds4_tp_free(tp);
    ds4_tokens_free(&tokens); free(text); free(expected); free(actual);
    return rc;
}
