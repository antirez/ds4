/* Full-depth graph, SSD and cache-continuation checks. Independent CPU
 * logit comparison is in check_deepseek41_native_logits.py. Only the fixture
 * narrows widths; production model admission stays fixed. */
#include "../ds4.c"

typedef struct {
    int seen, last;
    bool valid;
} prefill_progress;

static void check_prefill_progress(void *ud, const char *event, int current, int total) {
    prefill_progress *p = ud;
    if (strcmp(event, "prefill_display") || total != 129 ||
        current <= p->last || current > total) {
        if (p->valid) fprintf(stderr, "invalid native prefill progress: %s %d/%d\n", event, current, total);
        p->valid = false;
    }
    p->last = current;
    p->seen++;
}

static bool cancel_prefill(void *ud) {
    unsigned *calls = ud;
    return ++*calls > 32u;
}

int main(int argc, char **argv) {
    if (argc != 4 && (argc != 5 || (strcmp(argv[4], "stream") && strcmp(argv[4], "cache")))) return 2;
    const bool streaming = argc == 5;
    const bool cache = streaming && !strcmp(argv[4], "cache");
    ds4_model m = {.fd = -1};
    ds4_weights w = {0};
    ds4_session *session = calloc(1, sizeof(*session));
    ds4_engine *engine = calloc(1, sizeof(*engine));
    if (!session || !engine) return 1;
    session->engine = engine;
    engine->backend = DS4_BACKEND_METAL;
    ds41_gpu_graph *g = &session->ds41_graph;
    g_ds4_shape = DS4_SHAPE_FLASH41;
    g_ds4_shape.n_embd = 256; g_ds4_shape.n_vocab = 64;
    g_ds4_shape.n_head = 64; g_ds4_shape.n_out_group = 8;
    g_ds4_shape.n_lora_q = 64; g_ds4_shape.n_lora_o = 32;
    g_ds4_shape.n_expert = 8; g_ds4_shape.n_ff_exp = 256;
    g_ds4_shape.n_indexer_head = 32;
    for (uint32_t i = 0; i < 40; i++) g_ds4_compress_ratios[i] = i < 2 ? 0 : i < 20 ? 2 : 1;
    setenv("DS4_METAL_DISABLE_V41_PREFILL_ALIAS", "1", 1);
    setenv("DS4_METAL_V41_PREFILL_ROWS", "8", 1);
    model_open(&m, argv[1], true, false);
    weights_bind(&w, &m, false, 0, UINT32_MAX, true, false);
    weights_validate_layout(&w, 0, UINT32_MAX, true, true);
    ds4_gpu_model_residency_skip(1);
    if (!ds4_gpu_dsv41_require_native_math() || !ds4_gpu_init()) return 1;
    ds4_gpu_set_ssd_streaming(streaming);
    if (cache) {
        ds4_gpu_set_model_fd(m.fd);
        ds4_gpu_set_streaming_expert_cache_budget(256);
    }
    FILE *tokens = fopen(argv[2], "rb"), *out = fopen(argv[3], "wb");
    if (!tokens || !out) return 1;
    int ids[16384];
    const size_t count = fread(ids, sizeof(*ids), 16384, tokens);
    if (count < 132 || count >= 16384 || ferror(tokens)) return 1;
    uint32_t ctx = 1024;
    while (ctx <= count) ctx *= 2u;
    if (!(streaming ? metal_graph_stream_map_decode_static_all(&m, &w) :
                       ds4_gpu_set_model_map(m.map, m.size)) ||
        !ds41_graph_alloc(g, &m, &w, argv[1], ctx, streaming)) return 1;
    float *expected = calloc(count * 64, sizeof(float));
    float logits[64];
    char err[256] = {0};
    FILE *snapshot = tmpfile();
    if (!expected || !snapshot) return 1;
    session->ds41_graph_ready = true;
    session->logits = logits;
    for (unsigned pos = 0; pos < count; pos++) {
        if (!ds41_graph_step(g, &m, &w, ids[pos], logits)) return 1;
        for (unsigned i=0; i<64; i++) if (!isfinite(logits[i])) return 1;
        if (fwrite(logits, sizeof(float), 64, out) != 64) return 1;
        memcpy(expected + pos * 64, logits, sizeof(logits));
        token_vec_push(&session->checkpoint, ids[pos]);
        session->checkpoint_valid = true;
        if (pos == 128 && ds4_session_save_payload(session, snapshot, err, sizeof(err))) {
            fprintf(stderr, "save: %s\n", err); return 1;
        }
        if (!(pos % 16)) fprintf(stderr, "native graph token %u complete\n", pos);

    }
    fclose(out); fclose(tokens);
    if (cache && (!ds4_gpu_stream_expert_cache_current_count() ||
                  ds4_gpu_stream_expert_cache_current_count() > 256)) {
        fprintf(stderr, "native expert cache is empty or exceeds its budget\n"); return 1;
    }
    /* Restore into fresh allocations after the raw ring wrapped, with a
     * compression pair unfinished. Future logits must be byte-identical. */
    const uint64_t snapshot_bytes = (uint64_t)ftello(snapshot);
    /* Caches from the older native reductions have obsolete tags. They
     * must fail before changing any live state, even with identical weights. */
    if (fseeko(snapshot, 12u * sizeof(uint32_t), SEEK_SET)) return 1;
    uint32_t saved_tag;
    if (fread(&saved_tag, sizeof(saved_tag), 1, snapshot) != 1) return 1;
    const uint32_t obsolete_tags[] = {0x413431u, 0x4e3432u};
    for (unsigned i = 0; i < sizeof(obsolete_tags) / sizeof(obsolete_tags[0]); i++) {
        if (fseeko(snapshot, 12u * sizeof(uint32_t), SEEK_SET) ||
            payload_write_u32(snapshot, obsolete_tags[i], err, sizeof(err))) return 1;
        rewind(snapshot);
        if (!ds4_session_load_payload(session, snapshot, snapshot_bytes, err, sizeof(err)) ||
            !session->checkpoint_valid || g->pos != count || session->checkpoint.len != (int)count) {
            fprintf(stderr, "obsolete native arithmetic checkpoint accepted or live state changed\n"); return 1;
        }
    }
    if (fseeko(snapshot, 12u * sizeof(uint32_t), SEEK_SET) ||
        payload_write_u32(snapshot, saved_tag, err, sizeof(err))) return 1;
    rewind(snapshot);
    ds41_graph_free(g);
    if (!ds41_graph_alloc(g, &m, &w, argv[1], ctx, streaming) ||
        ds4_session_load_payload(session, snapshot, snapshot_bytes, err, sizeof(err))) {
        fprintf(stderr, "restore: %s\n", err); return 1;
    }
    for (unsigned pos = 129; pos < count; pos++) {
        if (!ds41_graph_step(g, &m, &w, ids[pos], logits) ||
            memcmp(logits, expected + pos * 64, sizeof(logits))) {
            fprintf(stderr, "restored logits differ at %u\n", pos); return 1;
        }
    }
    fclose(snapshot);
    ds41_graph_reset(g);
    /* Layer-major prefill must preserve the serial graph's public logits. */
    prefill_progress progress = {.valid = true};
    if (!ds41_graph_prefill(g, &m, &w, ids, 129, check_prefill_progress, &progress, 129, NULL, NULL) ||
        !progress.valid || !progress.seen || progress.last != 129 ||
        !ds41_graph_logits(g, &m, &w, logits) ||
        memcmp(logits, expected + 128 * 64, sizeof(logits))) {
        fprintf(stderr, "prefill/decode boundary differs\n"); return 1;
    }
    /* A checkpoint made by batched prefill must resume serial decoding after
     * a ring wrap and a half-filled compression pair, just like a serial one. */
    ds4_tokens_free(&session->checkpoint);
    for (unsigned pos = 0; pos < 129; pos++) token_vec_push(&session->checkpoint, ids[pos]);
    session->checkpoint_valid = true;
    snapshot = tmpfile();
    if (!snapshot || ds4_session_save_payload(session, snapshot, err, sizeof(err))) {
        fprintf(stderr, "prefill save: %s\n", err); return 1;
    }
    const uint64_t prefill_snapshot_bytes = (uint64_t)ftello(snapshot);
    rewind(snapshot);
    ds41_graph_free(g);
    if (!ds41_graph_alloc(g, &m, &w, argv[1], ctx, streaming) ||
        ds4_session_load_payload(session, snapshot, prefill_snapshot_bytes, err, sizeof(err)) ||
        !ds41_graph_step(g, &m, &w, ids[129], logits) ||
        memcmp(logits, expected + 129 * 64, sizeof(logits))) {
        fprintf(stderr, "prefill snapshot/decode boundary differs: %s\n", err); return 1;
    }
    fclose(snapshot);
    token_vec_push(&session->checkpoint, ids[129]);
    snapshot = tmpfile();
    if (!snapshot || ds4_session_save_payload(session, snapshot, err, sizeof(err))) return 1;
    const uint64_t recovery_bytes = (uint64_t)ftello(snapshot);
    unsigned calls = 0;
    FILE *partial = tmpfile();
    if (!partial || ds41_graph_prefill(g, &m, &w, ids + 130, (uint32_t)count - 130,
            NULL, NULL, (int)count, cancel_prefill, &calls) ||
        !ds4_session_save_payload(session, partial, err, sizeof(err))) {
        fprintf(stderr, "canceled prefill published a partial checkpoint\n"); return 1;
    }
    fclose(partial);
    rewind(snapshot);
    ds41_graph_free(g);
    if (!ds41_graph_alloc(g, &m, &w, argv[1], ctx, streaming) ||
        ds4_session_load_payload(session, snapshot, recovery_bytes, err, sizeof(err)) ||
        !ds41_graph_step(g, &m, &w, ids[130], logits) ||
        memcmp(logits, expected + 130 * 64, sizeof(logits)) ||
        (cache && !ds4_gpu_stream_expert_cache_current_count())) {
        fprintf(stderr, "canceled prefill recovery differs: %s\n", err); return 1;
    }
    fclose(snapshot);
    /* Continue across appends and the indexed-attention threshold. */
    if (!ds41_graph_prefill(g, &m, &w, ids + 131, (uint32_t)count - 131,
                           NULL, NULL, (int)count, NULL, NULL) ||
        !ds41_graph_logits(g, &m, &w, logits) ||
        memcmp(logits, expected + (count - 1) * 64, sizeof(logits))) {
        fprintf(stderr, "prefill append continuation differs\n"); return 1;
    }
    /* A single wide batch must agree with the differently partitioned appends
     * and serial steps, including reuse of compressed keys from earlier layers. */
    ds41_graph_reset(g);
    if (!ds41_graph_prefill(g, &m, &w, ids, (uint32_t)count,
                           NULL, NULL, (int)count, NULL, NULL) ||
        !ds41_graph_logits(g, &m, &w, logits) ||
        memcmp(logits, expected + (count - 1) * 64, sizeof(logits))) {
        fprintf(stderr, "wide prefill logits differ\n"); return 1;
    }
    fprintf(stderr, "PASS: %zu tokens, serial and batched snapshots, cancellation recovery, wide prefill/append/decode boundaries\n", count);
    free(expected); ds4_tokens_free(&session->checkpoint);
    ds41_graph_free(g); free(session); free(engine);
    ds4_gpu_cleanup(); model_close(&m);
    return 0;
}
