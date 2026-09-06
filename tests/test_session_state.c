/* Exercise private session bookkeeping without loading a model. The GPU build
 * also runs speculative rollback through the real pooled-indexer kernel. */
#include "../ds4.c"
#include <assert.h>

static void test_rewind(void) {
    ds4_engine e = { .backend = DS4_BACKEND_CPU };
    ds4_session *s = calloc(1, sizeof(*s));
    assert(s);
    s->engine = &e;
    s->ctx_size = 1024;
    for (int i = 0; i < 260; i++) ds4_tokens_push(&s->checkpoint, i);
    s->checkpoint_valid = true;
    s->mtp_draft_valid = true;
    s->checkpoint_images = calloc(1, sizeof(*s->checkpoint_images));
    assert(s->checkpoint_images);
    s->checkpoint_image_count = 1;
    ds4_vision_identity *images = s->checkpoint_images;

    ds4_session_rewind(s, 260);
    assert(s->checkpoint_valid && s->mtp_draft_valid);
    ds4_session_rewind(s, 300);
    assert(s->checkpoint.len == 260 && s->checkpoint_valid);
    const int boundaries[] = {259, 256, 255, 128, 127, 4, 3, 0};
    for (size_t i = 0; i < sizeof(boundaries) / sizeof(*boundaries); i++) {
        s->checkpoint_valid = true;
        ds4_session_rewind(s, boundaries[i]);
        assert(s->checkpoint.len == boundaries[i]);
        assert(!s->checkpoint_valid && !s->mtp_draft_valid);
        assert(s->checkpoint_images == images && s->checkpoint_image_count == 1);
        assert(ds4_session_common_prefix(s, &s->checkpoint) == 0);
        assert(ds4_session_argmax(s) == -1);
        assert(ds4_session_argmax_excluding(s, 1) == -1);
        assert(ds4_session_argmax_ignoring_eos(s, DS4_THINK_NONE) == -1);
        uint64_t rng = 1;
        assert(ds4_session_sample(s, 1, 0, 1, 0, &rng) == -1);
        char err[256] = "";
        int accepted[2];
        assert(ds4_session_eval(s, 1, err, sizeof(err)) != 0);
        assert(strstr(err, "synchronized checkpoint"));
        assert(ds4_session_eval_speculative(s, 1, 2, -1, 1, 0, 1, 0,
                    &rng, accepted, 2, err, sizeof(err)) == -1);
    }
    ds4_session_rewind(s, -1);
    ds4_session_rewind(NULL, 0);
    ds4_session_free(s);
}

static void test_session_memory(void) {
    const uint64_t gib = UINT64_C(1) << 30;
    ds4_engine e = { .backend = DS4_BACKEND_METAL,
                     .placement_session_count_hint = 4 };
    assert(ds4_engine_glm_graph_budget(&e, 2*gib) == 8*gib);
    e.glm_session_count = 1;
    e.glm_session_graph_bytes = 3*gib;
    assert(ds4_engine_glm_graph_budget(&e, 2*gib) == 9*gib);
    e.glm_session_count = 4;
    e.glm_session_graph_bytes = 8*gib;
    assert(ds4_engine_glm_graph_budget(&e, 3*gib) == 11*gib);
    e.placement_session_count_hint = 0;
    assert(ds4_engine_glm_graph_budget(&e, 3*gib) == 11*gib);
    assert(ds4_engine_glm_graph_budget(&e, UINT64_MAX) == UINT64_MAX);
    e.backend = DS4_BACKEND_CUDA;
    assert(ds4_engine_glm_graph_budget(&e, 3*gib) == 3*gib);

    ds4_session *s = calloc(1, sizeof(*s));
    assert(s);
    e.backend = DS4_BACKEND_CPU;
    s->engine = &e;
    s->glm_reserved_graph_bytes = 2*gib;
    ds4_session_free(s);
    assert(e.glm_session_count == 3 && e.glm_session_graph_bytes == 6*gib);
    e.backend = DS4_BACKEND_METAL;
    e.placement_session_count_hint = 4;
    assert(ds4_engine_glm_graph_budget(&e, 2*gib) == 8*gib);
}

static void test_payload_tokens(void) {
    FILE *fp = tmpfile();
    assert(fp);
    char err[256] = "";
    const uint32_t ids[] = {71, 19, 900, 4};
    for (size_t i = 0; i < 4; i++)
        assert(payload_write_u32(fp, ids[i], err, sizeof(err)) == 0);
    /* Opaque cache bytes are skipped, but the caller's trailer stays unread. */
    for (int i = 0; i < 65539; i++) assert(fputc('x', fp) != EOF);
    const long end = ftell(fp);
    assert(fputs("TRAILER", fp) >= 0);
    rewind(fp);
    ds4_tokens tokens = {0};
    assert(payload_read_tokens_for_rebuild(fp, 4, end,
                                           &tokens, err, sizeof(err)) == 0);
    assert(tokens.len == 4);
    for (int i = 0; i < 4; i++) assert(tokens.v[i] == (int)ids[i]);
    assert(ftell(fp) == end && fgetc(fp) == 'T');
    ds4_tokens_free(&tokens);
    rewind(fp);
    assert(payload_read_tokens_for_rebuild(fp, 4, 15,
                                           &tokens, err, sizeof(err)) != 0);
    assert(ftell(fp) == 0 && tokens.len == 0);
    assert(payload_read_tokens_for_rebuild(fp, 0, end,
                                           &tokens, err, sizeof(err)) != 0);
    rewind(fp);
    assert(payload_write_u32(fp, DS4_N_VOCAB, err, sizeof(err)) == 0);
    rewind(fp);
    assert(payload_read_tokens_for_rebuild(fp, 4, end,
                                           &tokens, err, sizeof(err)) != 0);
    assert(tokens.len == 0);
    rewind(fp);
    assert(payload_write_u32(fp, ids[0], err, sizeof(err)) == 0);
    rewind(fp);
    assert(payload_read_tokens_for_rebuild(fp, 4, end + 8,
                                           &tokens, err, sizeof(err)) != 0);
    ds4_tokens_free(&tokens);
    fclose(fp);
}

static void test_snapshot_bytes(void) {
    const ds4_shape saved_shape = g_ds4_shape;
    const uint32_t saved_ratio = g_ds4_compress_ratios[0];
    g_ds4_shape = DS4_SHAPE_FLASH;
    g_ds4_shape.n_layer = 1;
    g_ds4_shape.n_head_dim = 4;
    g_ds4_shape.n_vocab = 8;
    g_ds4_compress_ratios[0] = 0;
    ds4_engine e = {.backend = DS4_BACKEND_CPU};
    ds4_session *s = calloc(1, sizeof(*s));
    assert(s);
    s->engine = &e;
    s->ctx_size = 8;
    s->prefill_cap = 1;
    s->checkpoint_valid = true;
    s->logits = calloc(DS4_N_VOCAB, sizeof(float));
    assert(s->logits);
    kv_cache_init(&s->cpu_cache, 8, 0);
    for (int i = 0; i < 3; i++) ds4_tokens_push(&s->checkpoint, i);
    /* Every byte of the final float is nonzero, including the last byte. */
    const uint32_t bits = UINT32_C(0x3f9e0652);
    for (int i = 0; i < 3 * 4; i++)
        memcpy(s->cpu_cache.layer[0].raw_kv + i, &bits, sizeof(bits));
    ds4_session_snapshot snapshot = {0};
    const int lengths[] = {1, 3, 2, 3};
    for (size_t i = 0; i < sizeof(lengths) / sizeof(*lengths); i++) {
        s->checkpoint.len = lengths[i];
        s->cpu_cache.layer[0].n_raw = lengths[i];
        char err[192] = {0};
        FILE *fp = tmpfile();
        assert(fp);
        assert(ds4_session_save_payload(s, fp, err, sizeof(err)) == 0);
        const long bytes = ftell(fp);
        assert(bytes > 0);
        assert(ds4_session_save_snapshot(s, &snapshot, err, sizeof(err)) == 0);
        assert(snapshot.len == (uint64_t)bytes);
        rewind(fp);
        for (long j = 0; j < bytes; j++) {
            const int expected = fgetc(fp);
            if (expected != snapshot.ptr[j])
                fprintf(stderr, "snapshot byte %ld/%ld: expected=%d actual=%d\n",
                        j, bytes, expected, snapshot.ptr[j]);
            assert(expected == snapshot.ptr[j]);
        }
        assert(fgetc(fp) == EOF);
        fclose(fp);
    }
    ds4_session_snapshot_free(&snapshot);
    ds4_session_free(s);
    g_ds4_shape = saved_shape;
    g_ds4_compress_ratios[0] = saved_ratio;
}

static void test_text_observations(void) {
    ds4_engine e = {0};
    ds4_vocab *v = &e.vocab;
    char bytes[256][5] = {{0}};
    v->n_vocab = 260;
    v->token = calloc((size_t)v->n_vocab, sizeof(*v->token));
    assert(v->token);
    table_init(&v->token_to_id, v->n_vocab);
    table_init(&v->merge_rank, 0);
    for (int i = 0; i < 256; i++) {
        char *p = bytes[i];
        utf8_put(&p, gpt2_byte_to_codepoint((uint8_t)i));
        v->token[i] = (ds4_str){bytes[i], (uint64_t)(p - bytes[i])};
        table_put(&v->token_to_id, v->token[i], i);
    }
    const char *special[] = {
        "<|user|>", "<|observation|>", "<tool_response>", "</tool_response>"
    };
    for (int i = 0; i < 4; i++) {
        v->token[256+i] = (ds4_str){special[i], strlen(special[i])};
        table_put(&v->token_to_id, v->token[256+i], 256+i);
    }
    v->user_id = 256;
    v->observation_id = 257;
    v->tool_response_start_id = 258;
    v->tool_response_end_id = 259;
    const ds4_shape saved = g_ds4_shape;
    const ds4_shape shapes[] = {DS4_SHAPE_FLASH, DS4_SHAPE_PRO, DS4_SHAPE_GLM53};
    const char *roles[] = {"user", "tool", "function"};
    const char *parts[] = {"ok <x> & </tool_result> </tool_response>"};
    for (size_t family = 0; family < 3; family++) {
        g_ds4_shape = shapes[family];
        for (size_t role = 0; role < 3; role++) {
            ds4_tokens expected = {0}, actual = {0};
            ds4_tokens_push(&expected, 7);
            ds4_tokens_push(&actual, 7);
            ds4_chat_append_message(&e, &expected, roles[role], parts[0]);
            char err[160] = {0};
            assert(ds4_chat_append_multimodal_message(&e, &actual, roles[role],
                        parts, NULL, 0, NULL, err, sizeof(err)));
            assert(actual.len == expected.len);
            assert(!memcmp(actual.v, expected.v, (size_t)actual.len * sizeof(int)));
            const int len = actual.len;
            assert(!ds4_chat_append_multimodal_message(&e, &actual, "assistant",
                        parts, NULL, 0, NULL, err, sizeof(err)));
            assert(actual.len == len);
            float pixel = 1;
            ds4_vision_embedding image = {.data = &pixel, .token_count = 1};
            ds4_vision_span span = {0};
            const char *image_parts[] = {"before", "after"};
            assert(!ds4_chat_append_multimodal_message(&e, &actual, roles[role],
                        image_parts, &image, 1, &span, err, sizeof(err)));
            assert(actual.len == len && image.data == &pixel && !span.embedding.data);
            ds4_tokens_free(&expected);
            ds4_tokens_free(&actual);
        }
    }
    g_ds4_shape = saved;
    vocab_free(v);
}

#ifndef DS4_NO_GPU
static void test_glm_attention_budget(void) {
    const ds4_shape saved_shape = g_ds4_shape;
    g_ds4_shape = DS4_SHAPE_GLM53;
    ds4_glm_gpu_graph *g = calloc(1, sizeof(*g));
    assert(g);
    g->glm53 = true;
    g->compact_cache_cap = 16384;
    const uint32_t capacities[] = {1024, 4096, 8192, 16384};
    for (size_t i = 0; i < sizeof(capacities) / sizeof(*capacities); i++) {
        g->ctx_cap = capacities[i];
        assert(glm_graph_dense_compact_attention_limit(g) == 2051);
        g->full_kv_cache = true;
        assert(glm_graph_dense_compact_attention_limit(g) == 2051);
        g->full_kv_cache = false;
    }
    free(g);
    g_ds4_shape = saved_shape;
}

enum { POOL_DIM = 128, POOL_ROWS = 12, MODEL_BYTES = 16384 };
static float pool_raw[POOL_ROWS * POOL_DIM], pool_gates[POOL_ROWS * POOL_DIM];

static void pool_update(ds4_glm_gpu_graph *g, ds4_gpu_tensor *x,
                         ds4_gpu_tensor *gate, void *model, int pos, int rows) {
    assert(ds4_gpu_tensor_write(x, 0, pool_raw + pos * POOL_DIM,
                                rows * POOL_DIM * sizeof(float)));
    assert(ds4_gpu_tensor_write(gate, 0, pool_gates + pos * POOL_DIM,
                                rows * POOL_DIM * sizeof(float)));
    assert(ds4_gpu_glm53_indexer_pool_update_tensor(
        g->layer_indexer_key_cache[3], g->layer_indexer_tail_k[3],
        g->layer_indexer_tail_gate[3], x, gate, model, MODEL_BYTES,
        0, 4096, 8192, pos, rows, 16, POOL_DIM, 4, 1e-6f, false));
    assert(ds4_gpu_synchronize());
}

static void test_glm_spec_rollback(void) {
    const ds4_shape saved_shape = g_ds4_shape;
    g_ds4_shape = DS4_SHAPE_GLM53;
    void *model = mmap(NULL, MODEL_BYTES, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANON, -1, 0);
    assert(model != MAP_FAILED);
    for (int d = 0; d < POOL_DIM; d++) ((float *)model)[d] = 1;
    for (int t = 0; t < POOL_ROWS; t++) for (int d = 0; d < POOL_DIM; d++) {
        pool_raw[t * POOL_DIM + d] = sinf((t + 1) * (d + 1) * 0.1f);
        pool_gates[t * POOL_DIM + d] = cosf((t + 2) * (d + 1) * 0.03f);
    }
    assert(ds4_gpu_init());
    assert(ds4_gpu_set_model_map(model, MODEL_BYTES));
    ds4_glm_gpu_graph *g = calloc(1, sizeof(*g));
    assert(g);
    g->glm53 = true;
    g->layer_end = 3;
    for (int il = 0; il < 3; il++) {
        g->layer_kda_conv_state[il] = ds4_gpu_tensor_alloc(64*sizeof(float));
        g->layer_kda_recurrent_state[il] = ds4_gpu_tensor_alloc(128*sizeof(float));
        assert(g->layer_kda_conv_state[il] && g->layer_kda_recurrent_state[il]);
        assert(ds4_gpu_tensor_fill_f32(g->layer_kda_conv_state[il], 7, 64));
        assert(ds4_gpu_tensor_fill_f32(g->layer_kda_recurrent_state[il], 11, 128));
    }
    const uint64_t tail_bytes = 4 * POOL_DIM * sizeof(float);
    g->layer_indexer_key_cache[3] = ds4_gpu_tensor_alloc(tail_bytes);
    g->layer_indexer_tail_k[3] = ds4_gpu_tensor_alloc(2 * tail_bytes);
    g->layer_indexer_tail_gate[3] = ds4_gpu_tensor_view(
        g->layer_indexer_tail_k[3], tail_bytes, tail_bytes);
    const uint64_t bytes = glm53_graph_spec_state_bytes(g);
    assert(bytes == 3*(64+128)*sizeof(float) + 2*tail_bytes);
    g->mtp_state_backup = ds4_gpu_tensor_alloc(bytes);
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(POOL_ROWS*POOL_DIM*sizeof(float));
    ds4_gpu_tensor *gate = ds4_gpu_tensor_alloc(POOL_ROWS*POOL_DIM*sizeof(float));
    assert(g->layer_indexer_key_cache[3] && g->layer_indexer_tail_gate[3] &&
           g->mtp_state_backup && x && gate);

    for (int start = 3; start < 7; start++) {
        assert(ds4_gpu_tensor_fill_f32(g->layer_indexer_tail_k[3], 0, 8*POOL_DIM));
        for (int p = 0; p < start; p++) pool_update(g, x, gate, model, p, 1);
        assert(glm53_graph_copy_spec_state(g, true));
        for (int p = start; p < POOL_ROWS; p++) pool_update(g, x, gate, model, p, 1);
        float expected[3*POOL_DIM], replay[3*POOL_DIM];
        assert(ds4_gpu_tensor_read(g->layer_indexer_key_cache[3], 0,
                                   expected, sizeof(expected)));
        assert(glm53_graph_copy_spec_state(g, false));
        pool_update(g, x, gate, model, start, 2);
        for (int il = 0; il < 3; il++) {
            assert(ds4_gpu_tensor_fill_f32(g->layer_kda_conv_state[il], -1, 64));
            assert(ds4_gpu_tensor_fill_f32(g->layer_kda_recurrent_state[il], -1, 128));
        }
        assert(glm53_graph_copy_spec_state(g, false));
        for (int p = start; p < POOL_ROWS; p++) pool_update(g, x, gate, model, p, 1);
        assert(ds4_gpu_tensor_read(g->layer_indexer_key_cache[3], 0,
                                   replay, sizeof(replay)));
        assert(memcmp(expected, replay, sizeof(expected)) == 0);
        for (int il = 0; il < 3; il++) {
            float state[128];
            assert(ds4_gpu_tensor_read(g->layer_kda_conv_state[il], 0,
                                       state, 64*sizeof(float)));
            for (int j = 0; j < 64; j++) assert(state[j] == 7);
            assert(ds4_gpu_tensor_read(g->layer_kda_recurrent_state[il], 0,
                                       state, sizeof(state)));
            for (int j = 0; j < 128; j++) assert(state[j] == 11);
        }
    }
    ds4_gpu_tensor_free(x);
    ds4_gpu_tensor_free(gate);
    glm_graph_free(g);
    free(g);
    ds4_gpu_cleanup();
    munmap(model, MODEL_BYTES);
    g_ds4_shape = saved_shape;
}
#endif

int main(void) {
    test_rewind();
    test_session_memory();
    test_payload_tokens();
    test_snapshot_bytes();
    test_text_observations();
#ifndef DS4_NO_GPU
    test_glm_attention_budget();
    test_glm_spec_rollback();
#endif
    puts("session state tests: ok");
    return 0;
}
