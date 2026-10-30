/* Private, model-free checks for labels and persisted CPU frontiers. */
#include "../ds4.c"
#include <assert.h>
#include <sys/wait.h>

static ds4_str str(const char *s) { return (ds4_str){s, strlen(s)}; }

/* Feed real metadata accessors with a tiny synthetic mapped header. */
struct metadata_fixture { uint8_t data[4096]; ds4_kv kv[32]; char keys[32][128]; size_t count, used; };
static void metadata_value(struct metadata_fixture *f, const char *key, uint32_t type,
                           const void *data, size_t bytes) {
    assert(f->count < 32 && f->used + bytes <= sizeof(f->data));
    size_t i = f->count++;
    snprintf(f->keys[i], sizeof(f->keys[i]), "%s", key);
    f->kv[i] = (ds4_kv){str(f->keys[i]), type, f->used};
    memcpy(f->data + f->used, data, bytes);
    f->used += bytes;
}
static void metadata_string(struct metadata_fixture *f, const char *key, const char *value) {
    uint8_t data[128];
    uint64_t length = strlen(value);
    assert(length + 8 <= sizeof(data));
    memcpy(data, &length, 8);
    memcpy(data + 8, value, (size_t)length);
    metadata_value(f, key, GGUF_VALUE_STRING, data, (size_t)length + 8);
}
static void test_metadata(void) {
    const struct { const char *key; uint32_t value; } fields[] = {
        {"block_count", 41}, {"nextn_predict_layers", 1}, {"embedding_length", 2048},
        {"attention.head_count", 16}, {"attention.head_count_kv", 2},
        {"attention.key_length", 256}, {"attention.value_length", 256},
        {"rope.dimension_count", 64}, {"expert_count", 256}, {"expert_used_count", 8},
        {"expert_feed_forward_length", 512}, {"expert_shared_feed_forward_length", 512},
        {"ssm.conv_kernel", 4}, {"ssm.state_size", 128}, {"ssm.group_count", 16},
        {"ssm.time_step_rank", 32}, {"ssm.inner_size", 4096}, {"full_attention_interval", 4}
    };
    const char *basenames[] = {"Qwen3.6", "Ornith-1.5", "Custom"};
    const char *names[] = {"Qwen3.6 35B A3B", "Ornith 1.5 35B A3B", "Custom checkpoint"};
    ds4_shape old = g_ds4_shape;
    for (int label = 0; label < 3; label++) {
        struct metadata_fixture f = {0};
        metadata_string(&f, "general.architecture", "qwen35moe");
        metadata_string(&f, "general.basename", basenames[label]);
        metadata_string(&f, "general.name", names[label]);
        metadata_string(&f, "tokenizer.ggml.pre", "qwen35");
        for (size_t i = 0; i < sizeof(fields) / sizeof(fields[0]); i++) {
            char key[128];
            snprintf(key, sizeof(key), "qwen35moe.%s", fields[i].key);
            metadata_value(&f, key, GGUF_VALUE_UINT32, &fields[i].value, 4);
        }
        uint8_t array[12];
        uint32_t element_type = GGUF_VALUE_STRING;
        uint64_t length = 248320;
        memcpy(array, &element_type, 4); memcpy(array + 4, &length, 8);
        metadata_value(&f, "tokenizer.ggml.tokens", GGUF_VALUE_ARRAY, array, sizeof(array));
        float epsilon = 1e-6f, base = 10000000.0f;
        metadata_value(&f, "qwen35moe.attention.layer_norm_rms_epsilon", GGUF_VALUE_FLOAT32, &epsilon, 4);
        metadata_value(&f, "qwen35moe.rope.freq_base", GGUF_VALUE_FLOAT32, &base, 4);
        ds4_model m = {.map = f.data, .size = f.used, .kv = f.kv, .n_kv = f.count};
        config_validate_model(&m);
        assert(g_ds4_shape.family == DS4_MODEL_FAMILY_QWEN35MOE && g_ds4_shape.variant == (ds4_variant)(label + 7));
        assert(g_ds4_shape.n_layer == 40 && g_ds4_shape.n_vocab == 248320);
        /* A recognizable brand does not bypass shape validation. */
        pid_t child = fork();
        assert(child >= 0);
        if (!child) {
            ds4_kv *blocks = model_find_kv(&m, "qwen35moe.block_count");
            uint32_t wrong = 40;
            memcpy(f.data + blocks->value_pos, &wrong, 4);
            config_validate_model(&m);
            _exit(0);
        }
        int status;
        assert(waitpid(child, &status, 0) == child);
        assert(WIFEXITED(status) && WEXITSTATUS(status) != 0);
    }
    g_ds4_shape = old;
}

static void test_labels(void) {
    assert(DS4_VARIANT_FLASH == 0 && DS4_VARIANT_QWEN4_MINI == 6);
    assert(DS4_VARIANT_QWEN36 == 7 && DS4_VARIANT_ORNITH == 8 && DS4_VARIANT_QWEN35MOE == 9);
    qwen35moe_model_label q = qwen35moe_classify(str("Qwen3.6"), str("Qwen3.6 35B A3B"));
    qwen35moe_model_label o = qwen35moe_classify(str("Ornith-1.5"), str("Ornith 1.5 35B A3B"));
    qwen35moe_model_label other = qwen35moe_classify(str("Custom"), str("Custom model"));
    assert(q.variant == 7 && o.variant == 8 && other.variant == 9);
    assert(qwen35moe_classify(str("Qwen3.6"), str("Ornith 1.5 35B A3B")).variant == 9);
    assert(qwen35moe_classify((ds4_str){0}, str("Qwen3.6 35B A3B")).variant == 7);
    assert(qwen35moe_classify((ds4_str){0}, (ds4_str){0}).variant == 9);
    ds4_engine e = {.qwen35moe_model_name = q.name, .qwen35moe_model_variant = q.variant};
    ds4_shape old = g_ds4_shape;
    g_ds4_shape.name = o.name;
    g_ds4_shape.variant = o.variant;
    assert(!strcmp(ds4_engine_model_name(&e), "Qwen3.6 35B A3B"));
    assert(ds4_engine_model_id(&e) == 7);
    g_ds4_shape = old;
    unsetenv("DS4_QWEN35MOE_PREFILL_BATCH");
    assert(qwen35moe_env("PREFILL_BATCH") == NULL);
    assert(setenv("DS4_QWEN35MOE_PREFILL_BATCH", "8", 1) == 0);
    assert(!strcmp(qwen35moe_env("PREFILL_BATCH"), "8"));
    assert(setenv("DS4_QWEN35MOE_PREFILL_BATCH", "", 1) == 0);
    assert(!strcmp(qwen35moe_env("PREFILL_BATCH"), ""));
    unsetenv("DS4_QWEN35MOE_PREFILL_BATCH");
}

static void test_tensor_validation(void) {
    ds4_tensor tensor = {.name = {"fixture", 7}, .ndim = 2, .dim = {2048, 512}, .type = 16};
    ds4_model model = {.tensors = &tensor, .n_tensors = 1};
    assert(qwen35moe_tensor(&model, "fixture", 2, 2048, 512, 0) == &tensor);
    for (int bad = 0; bad < 2; bad++) {
        pid_t child = fork();
        assert(child >= 0);
        if (!child) {
            if (bad) tensor.type = DS4_TENSOR_Q4_0;
            else tensor.dim[1] = 511;
            qwen35moe_tensor(&model, "fixture", 2, 2048, 512, 0);
            _exit(0);
        }
        int status;
        assert(waitpid(child, &status, 0) == child);
        assert(WIFEXITED(status) && WEXITSTATUS(status) != 0);
    }
}

static void put_u32_at(FILE *fp, off_t offset, uint32_t value) {
    char err[128];
    assert(fseeko(fp, offset, SEEK_SET) == 0);
    assert(payload_write_u32(fp, value, err, sizeof(err)) == 0);
    assert(fflush(fp) == 0);
}
static int load(ds4_session *s, FILE *fp, uint64_t bytes) {
    char err[128];
    assert(fseeko(fp, 0, SEEK_SET) == 0);
    return ds4_session_load_payload(s, fp, bytes, err, sizeof(err));
}
static void test_stale_model_rebuild(ds4_session *saved) {
    assert(saved->engine->qwen35moe_model_invalid);
    /* Deliberately absent weights and mapping make any accidental inference
     * access fail. Every token, batch and context below otherwise fits. */
    assert(saved->engine->qwen35moe_weights == NULL && saved->engine->model.map == NULL);
    ds4_session probe = {.engine = saved->engine, .ctx_size = 4,
                         .qwen35moe_state = saved->qwen35moe_state,
                         .logits = saved->logits, .checkpoint_valid = true};
    ds4_tokens_push(&probe.checkpoint, 42);
    int ids[] = {42, 198};
    ds4_tokens unchanged = {ids, 1, 2}, rebuild = {ids, 2, 2}, empty = {0};
    char err[128];
    uint32_t old_pos = probe.qwen35moe_state->pos;
    assert(ds4_session_sync(&probe, &unchanged, err, sizeof(err)) == 1);
    assert(strstr(err, "reopen") && probe.checkpoint_valid && probe.checkpoint.len == 1);
    assert(ds4_session_sync(&probe, &empty, err, sizeof(err)) == 1);
    assert(strstr(err, "reopen"));
    assert(ds4_session_eval(&probe, 198, err, sizeof(err)) == 1);
    assert(strstr(err, "reopen") && probe.checkpoint.len == 1);
    assert(qwen35moe_forward(&probe, probe.qwen35moe_state, 42, 0,
                             probe.logits, err, sizeof(err)) == 1);
    assert(strstr(err, "reopen"));
    assert(qwen35moe_forward_batch(&probe, probe.qwen35moe_state, ids, 2, 0,
                                   probe.logits, err, sizeof(err)) == 1);
    assert(strstr(err, "reopen"));
    ds4_session_invalidate(&probe); /* The sysprompt cache-miss fallback. */
    assert(ds4_session_sync(&probe, &rebuild, err, sizeof(err)) == 1);
    assert(strstr(err, "reopen") && !probe.checkpoint_valid && probe.checkpoint.len == 0);
    assert(probe.qwen35moe_state->pos == old_pos);
    assert(probe.qwen35moe_state->recurrent[77] == -0.25f && probe.logits[123] == 0.75f);
    ds4_tokens_free(&probe.checkpoint);
}
static void test_payload(void) {
    char path[] = "/tmp/ds4-qwen35moe-model.XXXXXX";
    int fd = mkstemp(path);
    assert(fd >= 0 && write(fd, "fixture", 7) == 7);
    ds4_engine engine = {.backend = DS4_BACKEND_CPU, .model = {.fd = fd, .file_size = 7}};
    assert(qwen35moe_file_identity_read(&engine.model, &engine.qwen35moe_model_identity) == 0);
    /* Fixed-width encoding has no native struct padding or host byte order. */
    FILE *encoded = tmpfile();
    qwen35moe_file_identity pattern = {{UINT64_C(0x123456789abcdef0)}};
    char identity_err[128];
    assert(encoded && qwen35moe_identity_write(encoded, &pattern, identity_err, sizeof(identity_err)) == 0);
    assert(fflush(encoded) == 0 && ftello(encoded) == QWEN35MOE_IDENTITY_BYTES);
    rewind(encoded);
    uint8_t prefix[12], expected_prefix[] = {1, 0, 0, 0, 0xf0, 0xde, 0xbc, 0x9a, 0x78, 0x56, 0x34, 0x12};
    assert(fread(prefix, 1, sizeof(prefix), encoded) == sizeof(prefix));
    assert(!memcmp(prefix, expected_prefix, sizeof(prefix)));
    fclose(encoded);
    ds4_session *s = calloc(1, sizeof(*s));
    assert(s);
    s->engine = &engine;
    s->ctx_size = 2;
    s->qwen35moe_state = qwen35moe_state_new(2);
    s->logits = calloc(248320, sizeof(float));
    assert(s->logits);
    ds4_tokens_push(&s->checkpoint, 42);
    ds4_tokens_push(&s->checkpoint, 198);
    s->checkpoint_valid = true;
    s->logits[123] = 0.75f;
    s->qwen35moe_state->recurrent[77] = -0.25f;
    s->qwen35moe_state->history[99] = 1.25f;
    s->qwen35moe_state->keys[512 + 1] = 0x3555;
    s->qwen35moe_state->values[512 + 1] = 0x3999;
    uint64_t bytes = ds4_session_payload_bytes(s);
    FILE *fp = tmpfile();
    char err[128];
    assert(fp && ds4_session_save_payload(s, fp, err, sizeof(err)) == 0);
    assert(fflush(fp) == 0 && (uint64_t)ftello(fp) == bytes);
    s->logits[123] = 0;
    s->qwen35moe_state->recurrent[77] = 0;
    assert(load(s, fp, bytes) == 0); /* A full context can be restored. */
    assert(s->checkpoint_valid && s->checkpoint.len == 2 && s->checkpoint.v[0] == 42);
    assert(s->logits[123] == 0.75f && s->qwen35moe_state->recurrent[77] == -0.25f);
    assert(s->qwen35moe_state->history[99] == 1.25f);
    assert(s->qwen35moe_state->keys[513] == 0x3555 && s->qwen35moe_state->values[513] == 0x3999);

    /* Every identity field participates. Incompatibility must not destroy a
     * currently valid session or load any state from the foreign checkpoint. */
    off_t extension = DS4_SESSION_PAYLOAD_U32_FIELDS * 4;
    for (size_t i = 0; i < 7; i++) {
        uint64_t value = engine.qwen35moe_model_identity.field[i];
        put_u32_at(fp, extension + 4 + i * 8, (uint32_t)value ^ 1u);
        assert(load(s, fp, bytes) == DS4_SESSION_PAYLOAD_INCOMPATIBLE);
        assert(s->checkpoint_valid && s->checkpoint.len == 2 && s->logits[123] == 0.75f);
        put_u32_at(fp, extension + 4 + i * 8, (uint32_t)value);
    }
    put_u32_at(fp, extension, 2);
    assert(load(s, fp, bytes) == DS4_SESSION_PAYLOAD_INCOMPATIBLE);
    put_u32_at(fp, extension, QWEN35MOE_IDENTITY_VERSION);
    assert(!engine.qwen35moe_model_invalid); /* Foreign caches never poison a valid engine. */
    put_u32_at(fp, 0, 0);
    assert(load(s, fp, bytes) == 1);
    put_u32_at(fp, 0, DS4_SESSION_PAYLOAD_MAGIC);
    assert(load(s, fp, bytes - 1) == 1);

    /* Unknown layouts are errors, not recoverable identity mismatches. */
    put_u32_at(fp, 12 * 4, 0);
    assert(load(s, fp, bytes) == 1);
    assert(s->checkpoint_valid && s->checkpoint.len == 2);
    put_u32_at(fp, 12 * 4, QWEN35MOE_PAYLOAD_TAG);

    /* Identical file bytes under another inode are conservatively incompatible. */
    FILE *copy = tmpfile();
    assert(copy && fwrite("fixture", 1, 7, copy) == 7 && fflush(copy) == 0);
    ds4_model other = {.fd = fileno(copy), .file_size = 7};
    qwen35moe_file_identity copied;
    assert(qwen35moe_file_identity_read(&other, &copied) == 0);
    assert(memcmp(&copied, &engine.qwen35moe_model_identity, sizeof(copied)) != 0);
    fclose(copy);
    /* Force a timestamp change, without changing size, and retain the opened
     * snapshot. Save and load both reject; no scan of model bytes is needed. */
    struct timespec stamp[2] = {{1, 0}, {2, 3}};
    assert(futimens(fd, stamp) == 0);
    assert(load(s, fp, bytes) == 1);
    test_stale_model_rebuild(s);
    put_u32_at(fp, extension + 4, (uint32_t)engine.qwen35moe_model_identity.field[0] ^ 1u);
    assert(load(s, fp, bytes) == 1); /* Foreign saved identity cannot mask a stale engine. */
    assert(s->checkpoint_valid && s->checkpoint.len == 2 && s->logits[123] == 0.75f);
    assert(s->qwen35moe_state->recurrent[77] == -0.25f);
    FILE *out = tmpfile();
    assert(out && ds4_session_save_payload(s, out, err, sizeof(err)) == 1);
    assert(ftello(out) == 0);
    fclose(out);
    assert(ftruncate(fd, 8) == 0);
    assert(qwen35moe_file_identity_read(&engine.model, &copied) == 1);
    assert(load(s, fp, bytes) == 1);
    assert(s->checkpoint_valid && s->checkpoint.len == 2 && s->logits[123] == 0.75f);
    /* Actual truncation of the opened model has the same hard-error policy. */
    assert(ftruncate(fd, 0) == 0);
    assert(load(s, fp, bytes) == 1);
    test_stale_model_rebuild(s);
    assert(s->checkpoint_valid && s->checkpoint.len == 2);
    assert(s->qwen35moe_state->recurrent[77] == -0.25f);

    fclose(fp);
    qwen35moe_state_free(s->qwen35moe_state);
    ds4_tokens_free(&s->checkpoint);
    free(s->logits);
    free(s);
    close(fd);
    unlink(path);
}

int main(void) {
    test_labels();
    test_metadata();
    test_tensor_validation();
    test_payload();
    puts("qwen35moe labels, tensor contracts, and versioned local-file payload identity: PASS");
    return 0;
}
