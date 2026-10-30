/* qwen35moe-35B-A3B CPU execution. Included by ds4.c to keep tensor internals
 * behind the engine boundary. Architecture reference: pinned llama.cpp,
 * src/models/qwen35moe.cpp; MIT, copyright 2023-2026 The ggml authors. */
#include "cpu/ggml/qwen35moe-quants.h"
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC push_options
#pragma GCC optimize("no-fast-math", "fp-contract=off")
#elif defined(__clang__)
#pragma float_control(precise, on, push)
#pragma clang fp contract(off)
#endif

struct qwen35moe_layer {
    ds4_tensor *norm, *post_norm;
    ds4_tensor *q, *k, *v, *out, *qn, *kn;
    ds4_tensor *qkv, *z, *alpha, *beta, *conv, *a, *dt, *ssm_norm;
    ds4_tensor *router, *gate, *up, *down;
    ds4_tensor *shared_gate, *shared_up, *shared_down, *shared_router;
};
struct qwen35moe_weights {
    ds4_tensor *embedding, *output, *norm;
    struct qwen35moe_layer layer[40];
};
static int qwen35moe_routed_quant_bits(const qwen35moe_weights *w) {
    const ds4_tensor *gate = w->layer[0].gate;
    if (!gate) return 0;
    return (gate->type == DS4_TENSOR_Q4_K || gate->type == DS4_TENSOR_MXFP4) ? 4 : 2;
}
struct qwen35moe_state {
    uint32_t cap, pos;
    bool timing;
    uint64_t timed_tokens;
    double attention_seconds, feedforward_seconds, output_seconds;
    double linear_finish_seconds, full_attention_finish_seconds;
    double batch_norm_seconds, batch_post_rows_seconds;
    double batch_router_seconds, batch_conv_prepare_seconds;
    float *recurrent, *history;
    uint16_t *keys, *values;
    /* Reused token scratch; model weights are never expanded to float. */
    float x[2048], norm[2048], attn[2048], ffn[2048];
    float qkv[8192], mixed[8192], z[4096], heads[4096];
    float alpha[32], beta[32], router[256], gate[512], up[512], down[2048];
    uint64_t activation[32 * 292 / 8];
    float route_gate[8 * 512], route_up[8 * 512], route_down[8 * 2048];
    uint8_t route_activation[8][sizeof(uint64_t) * (32 * 292 / 8)];
};

/* Payloads are tied conservatively to the exact local GGUF, without reading
 * its weights: copying or touching the file invalidates cached state. fstat
 * identifies the mapped fd rather than whatever a pathname later resolves to. */
#define QWEN35MOE_PAYLOAD_TAG 0x51333502u
#define QWEN35MOE_IDENTITY_VERSION 1u
#define QWEN35MOE_IDENTITY_BYTES (4u + 7u * 8u)
#define QWEN35MOE_RECURRENT_BYTES (30ull * 32 * 128 * 128 * sizeof(float))
#define QWEN35MOE_HISTORY_BYTES (30ull * 3 * 8192 * sizeof(float))
#define QWEN35MOE_KV_ROW_BYTES (512ull * sizeof(uint16_t))

static int qwen35moe_file_identity_read(const ds4_model *m, qwen35moe_file_identity *id) {
    struct stat st;
    if (m->fd < 0 || fstat(m->fd, &st) || st.st_size < 0 ||
        (uint64_t)st.st_size != m->file_size) return 1;
    id->field[0] = (uint64_t)st.st_dev;
    id->field[1] = (uint64_t)st.st_ino;
    id->field[2] = (uint64_t)st.st_size;
#if defined(__APPLE__)
    id->field[3] = (uint64_t)(int64_t)st.st_mtimespec.tv_sec;
    id->field[4] = (uint64_t)st.st_mtimespec.tv_nsec;
    id->field[5] = (uint64_t)(int64_t)st.st_ctimespec.tv_sec;
    id->field[6] = (uint64_t)st.st_ctimespec.tv_nsec;
#else
    id->field[3] = (uint64_t)(int64_t)st.st_mtim.tv_sec;
    id->field[4] = (uint64_t)st.st_mtim.tv_nsec;
    id->field[5] = (uint64_t)(int64_t)st.st_ctim.tv_sec;
    id->field[6] = (uint64_t)st.st_ctim.tv_nsec;
#endif
    return 0;
}
static int qwen35moe_require_model_valid(const ds4_engine *e, char *err, size_t errlen) {
    if (!e->qwen35moe_model_invalid) return 0;
    payload_set_err(err, errlen, "qwen35moe model file changed since engine open; reopen and rebuild");
    return 1;
}
static int qwen35moe_file_identity_check(ds4_engine *e, char *err, size_t errlen) {
    if (qwen35moe_require_model_valid(e, err, errlen)) return 1;
    qwen35moe_file_identity current;
    if (qwen35moe_file_identity_read(&e->model, &current) ||
        memcmp(&current, &e->qwen35moe_model_identity, sizeof(current))) {
        /* A stale mapping cannot safely rebuild text. Only a foreign saved
         * checkpoint with a still-valid open engine is recoverable. Latch the
         * failure so cache-miss retries cannot subsequently touch its weights;
         * forward execution checks this bit without any per-token fstat. */
        e->qwen35moe_model_invalid = true;
        return qwen35moe_require_model_valid(e, err, errlen);
    }
    return 0;
}
static int qwen35moe_identity_write(FILE *fp, const qwen35moe_file_identity *id,
                                   char *err, size_t errlen) {
    if (payload_write_u32(fp, QWEN35MOE_IDENTITY_VERSION, err, errlen)) return 1;
    for (size_t i = 0; i < 7; i++) {
        if (payload_write_u32(fp, (uint32_t)id->field[i], err, errlen) ||
            payload_write_u32(fp, (uint32_t)(id->field[i] >> 32), err, errlen)) return 1;
    }
    return 0;
}
static int qwen35moe_identity_read(FILE *fp, qwen35moe_file_identity *id,
                                  uint32_t *version, uint64_t *remaining,
                                  char *err, size_t errlen) {
    if (payload_read_u32(fp, version, remaining, err, errlen)) return 1;
    for (size_t i = 0; i < 7; i++) {
        uint32_t lo, hi;
        if (payload_read_u32(fp, &lo, remaining, err, errlen) ||
            payload_read_u32(fp, &hi, remaining, err, errlen)) return 1;
        id->field[i] = (uint64_t)lo | (uint64_t)hi << 32;
    }
    return 0;
}
static uint64_t qwen35moe_payload_state_bytes(uint32_t rows) {
    return DS4_SESSION_PAYLOAD_U32_FIELDS * sizeof(uint32_t) +
           (uint64_t)rows * sizeof(uint32_t) + 248320ull * sizeof(float) +
           QWEN35MOE_RECURRENT_BYTES + QWEN35MOE_HISTORY_BYTES + 20ull * rows * QWEN35MOE_KV_ROW_BYTES;
}
static uint64_t qwen35moe_payload_bytes(const ds4_session *s) {
    if (!s->checkpoint_valid || s->checkpoint.len <= 0) return 0;
    return qwen35moe_payload_state_bytes((uint32_t)s->checkpoint.len) + QWEN35MOE_IDENTITY_BYTES;
}
static int qwen35moe_save_payload(ds4_session *s, FILE *fp, char *err, size_t errlen) {
    if (!s || !fp || !qwen35moe_payload_bytes(s)) {
        payload_set_err(err, errlen, "qwen35moe session has no valid checkpoint to save");
        return 1;
    }
    int rc = qwen35moe_file_identity_check(s->engine, err, errlen);
    if (rc) return rc;
    qwen35moe_state *st = s->qwen35moe_state;
    uint32_t rows = (uint32_t)s->checkpoint.len;
    uint32_t h[DS4_SESSION_PAYLOAD_U32_FIELDS] = {
        DS4_SESSION_PAYLOAD_MAGIC, DS4_SESSION_PAYLOAD_VERSION,
        (uint32_t)s->ctx_size, 1, (uint32_t)s->ctx_size, 0, 0, rows,
        40, 512, 512, 248320, QWEN35MOE_PAYLOAD_TAG,
    };
    for (uint32_t i = 0; i < DS4_SESSION_PAYLOAD_U32_FIELDS; i++)
        if (payload_write_u32(fp, h[i], err, errlen)) return 1;
    if (qwen35moe_identity_write(fp, &s->engine->qwen35moe_model_identity, err, errlen)) return 1;
    for (uint32_t i = 0; i < rows; i++)
        if (payload_write_u32(fp, (uint32_t)s->checkpoint.v[i], err, errlen)) return 1;
    if (payload_write_bytes(fp, s->logits, 248320ull * sizeof(float), err, errlen) ||
        payload_write_bytes(fp, st->recurrent, QWEN35MOE_RECURRENT_BYTES, err, errlen) ||
        payload_write_bytes(fp, st->history, QWEN35MOE_HISTORY_BYTES, err, errlen)) return 1;
    for (uint32_t il = 0; il < 10; il++) {
        const uint64_t base = (uint64_t)il * st->cap * 512;
        if (payload_write_bytes(fp, st->keys + base, (uint64_t)rows * QWEN35MOE_KV_ROW_BYTES, err, errlen) ||
            payload_write_bytes(fp, st->values + base, (uint64_t)rows * QWEN35MOE_KV_ROW_BYTES, err, errlen)) return 1;
    }
    return qwen35moe_file_identity_check(s->engine, err, errlen);
}
static int qwen35moe_load_payload(ds4_session *s, FILE *fp, uint64_t bytes, char *err, size_t errlen) {
    if (!s || !fp) {
        payload_set_err(err, errlen, "invalid qwen35moe checkpoint load");
        return 1;
    }
    /* The caller supplies a payload segment length, which must also exist in
     * the underlying seekable file before an identity rejection is classified
     * as a well-formed incompatibility. Checking extent avoids reading state. */
    off_t start = ftello(fp);
    if (start < 0 || fseeko(fp, 0, SEEK_END)) {
        payload_set_err(err, errlen, "qwen35moe checkpoint needs a seekable payload file");
        return 1;
    }
    off_t end = ftello(fp);
    if (fseeko(fp, start, SEEK_SET) || end < start || bytes > (uint64_t)(end - start)) {
        payload_set_err(err, errlen, "truncated qwen35moe checkpoint");
        return 1;
    }
    uint64_t remaining = bytes;
    uint32_t h[DS4_SESSION_PAYLOAD_U32_FIELDS];
    for (uint32_t i = 0; i < DS4_SESSION_PAYLOAD_U32_FIELDS; i++)
        if (payload_read_u32(fp, &h[i], &remaining, err, errlen)) return 1;
    uint32_t rows = h[7];
    uint64_t expected = qwen35moe_payload_state_bytes(rows) + QWEN35MOE_IDENTITY_BYTES;
    if (h[0] != DS4_SESSION_PAYLOAD_MAGIC || h[1] != DS4_SESSION_PAYLOAD_VERSION ||
        h[12] != QWEN35MOE_PAYLOAD_TAG ||
        h[3] != 1 || h[4] != h[2] || h[5] || h[6] ||
        h[8] != 40 || h[9] != 512 || h[10] != 512 || h[11] != 248320 ||
        !rows || rows > (uint32_t)s->ctx_size || h[2] < rows || expected != bytes) {
        payload_set_err(err, errlen, "qwen35moe checkpoint layout or context mismatch");
        return 1;
    }
    int rc = qwen35moe_file_identity_check(s->engine, err, errlen);
    if (rc) return rc;
    qwen35moe_file_identity saved;
    uint32_t identity_version;
    if (qwen35moe_identity_read(fp, &saved, &identity_version, &remaining, err, errlen)) return 1;
    if (identity_version != QWEN35MOE_IDENTITY_VERSION ||
        memcmp(&saved, &s->engine->qwen35moe_model_identity, sizeof(saved))) {
        payload_set_err(err, errlen, "qwen35moe checkpoint model identity/version mismatch; rebuild from saved text");
        return DS4_SESSION_PAYLOAD_INCOMPATIBLE;
    }
    ds4_session_invalidate(s);
    s->checkpoint.len = 0;
    for (uint32_t i = 0; i < rows; i++) {
        uint32_t token;
        if (payload_read_u32(fp, &token, &remaining, err, errlen)) return 1;
        if (token >= 248320) {
            payload_set_err(err, errlen, "invalid qwen35moe checkpoint token");
            return 1;
        }
        token_vec_push(&s->checkpoint, (int)token);
    }
    qwen35moe_state *st = s->qwen35moe_state;
    if (payload_read_bytes(fp, s->logits, 248320ull * sizeof(float), &remaining, err, errlen) ||
        payload_read_bytes(fp, st->recurrent, QWEN35MOE_RECURRENT_BYTES, &remaining, err, errlen) ||
        payload_read_bytes(fp, st->history, QWEN35MOE_HISTORY_BYTES, &remaining, err, errlen)) return 1;
    for (uint32_t il = 0; il < 10; il++) {
        const uint64_t base = (uint64_t)il * st->cap * 512;
        if (payload_read_bytes(fp, st->keys + base, (uint64_t)rows * QWEN35MOE_KV_ROW_BYTES, &remaining, err, errlen) ||
            payload_read_bytes(fp, st->values + base, (uint64_t)rows * QWEN35MOE_KV_ROW_BYTES, &remaining, err, errlen)) return 1;
    }
    if (remaining) {
        payload_set_err(err, errlen, "qwen35moe checkpoint has trailing data");
        return 1;
    }
    rc = qwen35moe_file_identity_check(s->engine, err, errlen);
    if (rc) return rc;
    st->pos = rows - 1;
    s->checkpoint_valid = true;
    return 0;
}

static const char *qwen35moe_env(const char *suffix) {
    char name[96];
    snprintf(name, sizeof(name), "DS4_QWEN35MOE_%s", suffix);
    return getenv(name);
}

static float qwen35moe_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}
static float qwen35moe_softplus(float x) {
    return x > 20 ? x : logf(1 + expf(x));
}
static void qwen35moe_rms(float *out, const float *x, const float *g, uint32_t n, float eps) {
    double sum = 0;
    for (uint32_t i = 0; i < n; i++)
        sum += (double)(x[i] * x[i]);
    float mean = (float)(sum / n), scale = 1 / sqrtf(mean + eps);
    for (uint32_t i = 0; i < n; i++)
        out[i] = (x[i] * scale) * (g ? g[i] : 1);
}
static ds4_tensor *qwen35moe_tensor(const ds4_model *m, const char *name, uint32_t rank, uint64_t d0,
                              uint64_t d1, uint64_t d2) {
    ds4_tensor *t = required_tensor(m, name);
    if (t->ndim != rank || t->dim[0] != d0 || (rank > 1 && t->dim[1] != d1) ||
        (rank > 2 && t->dim[2] != d2)) {
        fprintf(stderr, "ds4: qwen35moe invalid tensor dimensions: %s\n", name);
        exit(1);
    }
    switch (t->type) {
    case 0:
    case 1:
    case 8:
    case 10:
    case 12:
    case 13:
    case 14:
    case 16:
    case 17:
    case 30:
        break;
    default:
        ds4_die("qwen35moe tensor has an unsupported quantization format");
    }
    if (t->type != 0 && t->type != 1 && t->type != 30 && d0 % gguf_types[t->type].block_elems)
        ds4_die("qwen35moe unaligned quantized row");
    return t;
}

static ds4_tensor *qwen35moe_layer_tensor(const ds4_model *m, uint32_t il, const char *suffix,
                                    uint32_t rank, uint64_t d0, uint64_t d1, uint64_t d2) {
    char name[128];
    snprintf(name, sizeof(name), "blk.%u.%s", il, suffix);
    return qwen35moe_tensor(m, name, rank, d0, d1, d2);
}

typedef struct { const char *name; ds4_variant variant; } qwen35moe_model_label;
static qwen35moe_model_label qwen35moe_classify(ds4_str basename, ds4_str name) {
    if ((ds4_streq(basename, "Qwen3.6") || ds4_streq(name, "Qwen3.6 35B A3B")) &&
        (!basename.len || ds4_streq(basename, "Qwen3.6")) &&
        (!name.len || ds4_streq(name, "Qwen3.6 35B A3B")))
        return (qwen35moe_model_label){"Qwen3.6 35B A3B", DS4_VARIANT_QWEN36};
    if ((ds4_streq(basename, "Ornith-1.5") || ds4_streq(name, "Ornith 1.5 35B A3B")) &&
        (!basename.len || ds4_streq(basename, "Ornith-1.5")) &&
        (!name.len || ds4_streq(name, "Ornith 1.5 35B A3B")))
        return (qwen35moe_model_label){"Ornith 1.5 35B A3B", DS4_VARIANT_ORNITH};
    return (qwen35moe_model_label){"qwen35moe 35B A3B", DS4_VARIANT_QWEN35MOE};
}

static void qwen35moe_validate(const ds4_model *m) {
    const struct {
        const char *key;
        uint32_t value;
    } expected[] = {{"block_count", 41},
                    {"nextn_predict_layers", 1},
                    {"embedding_length", 2048},
                    {"attention.head_count", 16},
                    {"attention.head_count_kv", 2},
                    {"attention.key_length", 256},
                    {"attention.value_length", 256},
                    {"rope.dimension_count", 64},
                    {"expert_count", 256},
                    {"expert_used_count", 8},
                    {"expert_feed_forward_length", 512},
                    {"expert_shared_feed_forward_length", 512},
                    {"ssm.conv_kernel", 4},
                    {"ssm.state_size", 128},
                    {"ssm.group_count", 16},
                    {"ssm.time_step_rank", 32},
                    {"ssm.inner_size", 4096},
                    {"full_attention_interval", 4}};
    for (size_t i = 0; i < sizeof(expected) / sizeof(expected[0]); i++) {
        char key[128];
        snprintf(key, sizeof(key), "qwen35moe.%s", expected[i].key);
        config_expect_u32(key, required_u32(m, key), expected[i].value);
    }
    ds4_array_ref vocab;
    if (!model_get_array(m, "tokenizer.ggml.tokens", &vocab) || vocab.len != 248320)
        ds4_die("qwen35moe unexpected vocabulary size");
    ds4_str pre;
    if (!model_get_string(m, "tokenizer.ggml.pre", &pre) || !ds4_streq(pre, "qwen35"))
        ds4_die("qwen35moe requires the qwen35 tokenizer");
    ds4_str basename = {0}, name = {0};
    model_get_string(m, "general.basename", &basename);
    model_get_string(m, "general.name", &name);
    qwen35moe_model_label label = qwen35moe_classify(basename, name);
    g_ds4_shape = (ds4_shape){.name = label.name,
                              .family = DS4_MODEL_FAMILY_QWEN35MOE,
                              .variant = label.variant,
                              .n_layer = 40,
                              .n_embd = 2048,
                              .n_vocab = 248320,
                              .n_head = 16,
                              .n_head_kv = 2,
                              .n_head_dim = 256,
                              .n_value_dim = 256,
                              .n_rot = 64,
                              .n_expert = 256,
                              .n_expert_used = 8,
                              .n_expert_shared = 1,
                              .n_ff_exp = 512,
                              .n_hc = 1,
                              .n_lin_k_head = 16,
                              .n_lin_v_head = 32,
                              .n_lin_head_dim = 128,
                              .n_lin_conv = 4,
                              .n_full_attn_interval = 4,
                              .rms_eps = 1e-6f,
                              .rope_freq_base = 10000000.0f};
    config_expect_f32("qwen35moe RMS epsilon",
                      required_f32(m, "qwen35moe.attention.layer_norm_rms_epsilon"), 1e-6f);
    config_expect_f32("qwen35moe rope base", required_f32(m, "qwen35moe.rope.freq_base"),
                      10000000.0f);
}

static qwen35moe_weights *qwen35moe_bind(const ds4_model *m) {
    ds4_qwen35moe_init();
    qwen35moe_weights *w = xcalloc(1, sizeof(*w));
    w->embedding = qwen35moe_tensor(m, "token_embd.weight", 2, 2048, 248320, 0);
    w->output = qwen35moe_tensor(m, "output.weight", 2, 2048, 248320, 0);
    w->norm = qwen35moe_tensor(m, "output_norm.weight", 1, 2048, 0, 0);
    if (w->norm->type != 0)
        ds4_die("qwen35moe output normalization must be F32");
    for (uint32_t il = 0; il < 40; il++) {
        struct qwen35moe_layer *l = &w->layer[il];
#define QWEN35MOE_T(field, suffix, rank, d0, d1, d2)                                                     \
    l->field = qwen35moe_layer_tensor(m, il, suffix, rank, d0, d1, d2)
        QWEN35MOE_T(norm, "attn_norm.weight", 1, 2048, 0, 0);
        QWEN35MOE_T(post_norm, "post_attention_norm.weight", 1, 2048, 0, 0);
        QWEN35MOE_T(router, "ffn_gate_inp.weight", 2, 2048, 256, 0);
        QWEN35MOE_T(gate, "ffn_gate_exps.weight", 3, 2048, 512, 256);
        QWEN35MOE_T(up, "ffn_up_exps.weight", 3, 2048, 512, 256);
        QWEN35MOE_T(down, "ffn_down_exps.weight", 3, 512, 2048, 256);
        QWEN35MOE_T(shared_gate, "ffn_gate_shexp.weight", 2, 2048, 512, 0);
        QWEN35MOE_T(shared_up, "ffn_up_shexp.weight", 2, 2048, 512, 0);
        QWEN35MOE_T(shared_down, "ffn_down_shexp.weight", 2, 512, 2048, 0);
        QWEN35MOE_T(shared_router, "ffn_gate_inp_shexp.weight", 1, 2048, 0, 0);
        if (il % 4 == 3) {
            QWEN35MOE_T(q, "attn_q.weight", 2, 2048, 8192, 0);
            QWEN35MOE_T(k, "attn_k.weight", 2, 2048, 512, 0);
            QWEN35MOE_T(v, "attn_v.weight", 2, 2048, 512, 0);
            QWEN35MOE_T(out, "attn_output.weight", 2, 4096, 2048, 0);
            QWEN35MOE_T(qn, "attn_q_norm.weight", 1, 256, 0, 0);
            QWEN35MOE_T(kn, "attn_k_norm.weight", 1, 256, 0, 0);
        } else {
            QWEN35MOE_T(qkv, "attn_qkv.weight", 2, 2048, 8192, 0);
            QWEN35MOE_T(z, "attn_gate.weight", 2, 2048, 4096, 0);
            QWEN35MOE_T(out, "ssm_out.weight", 2, 4096, 2048, 0);
            QWEN35MOE_T(alpha, "ssm_alpha.weight", 2, 2048, 32, 0);
            QWEN35MOE_T(beta, "ssm_beta.weight", 2, 2048, 32, 0);
            QWEN35MOE_T(conv, "ssm_conv1d.weight", 2, 4, 8192, 0);
            QWEN35MOE_T(a, "ssm_a", 1, 32, 0, 0);
            QWEN35MOE_T(dt, "ssm_dt.bias", 1, 32, 0, 0);
            QWEN35MOE_T(ssm_norm, "ssm_norm.weight", 1, 128, 0, 0);
        }
#undef QWEN35MOE_T
        ds4_tensor *controls[] = {l->norm, l->post_norm, l->qn, l->kn,
                                  l->conv, l->a,         l->dt, l->ssm_norm};
        for (size_t j = 0; j < sizeof(controls) / sizeof(controls[0]); j++)
            if (controls[j] && controls[j]->type != 0)
                ds4_die("qwen35moe normalization/recurrent controls must be F32");
    }
    return w;
}

struct qwen35moe_weight_pages { uint64_t first, end; };
static int qwen35moe_weight_pages_compare(const void *a, const void *b) {
    const struct qwen35moe_weight_pages *x = a, *y = b;
    return x->first < y->first ? -1 : x->first > y->first ? 1 : 0;
}
static void qwen35moe_weight_pages_add(struct qwen35moe_weight_pages *pages, size_t *count,
                                 size_t capacity, const ds4_tensor *t, uint64_t page_size) {
    if (!t) return;
    if (*count >= capacity) ds4_die("qwen35moe bound tensor count exceeds GGUF tensor count");
    pages[(*count)++] = (struct qwen35moe_weight_pages){
        .first = t->abs_offset / page_size,
        .end = (t->abs_offset + t->bytes - 1) / page_size + 1,
    };
}
/* Include metadata and only tensors bound by the text path; block-40 MTP
 * weights are present in the GGUF but are not accessed by inference. */
static struct qwen35moe_weight_pages *qwen35moe_bound_pages(const ds4_model *m, const qwen35moe_weights *w,
                                                uint64_t *page_size_out, size_t *count_out) {
    long system_page = sysconf(_SC_PAGESIZE);
    uint64_t page_size = system_page > 0 ? (uint64_t)system_page : 4096;
    size_t capacity = (size_t)m->n_tensors + 1, count = 0;
    struct qwen35moe_weight_pages *pages = xmalloc(capacity * sizeof(*pages));
    pages[count++] = (struct qwen35moe_weight_pages){0,
        (m->tensor_data_pos + page_size - 1) / page_size};
#define QWEN35MOE_PAGE(t) qwen35moe_weight_pages_add(pages, &count, capacity, (t), page_size)
    QWEN35MOE_PAGE(w->embedding); QWEN35MOE_PAGE(w->output); QWEN35MOE_PAGE(w->norm);
    for (uint32_t il = 0; il < 40; il++) {
        const struct qwen35moe_layer *l = &w->layer[il];
        ds4_tensor *bound[] = {
            l->norm, l->post_norm, l->q, l->k, l->v, l->out, l->qn, l->kn,
            l->qkv, l->z, l->alpha, l->beta, l->conv, l->a, l->dt, l->ssm_norm,
            l->router, l->gate, l->up, l->down, l->shared_gate, l->shared_up,
            l->shared_down, l->shared_router,
        };
        for (size_t i = 0; i < sizeof(bound) / sizeof(bound[0]); i++) QWEN35MOE_PAGE(bound[i]);
    }
#undef QWEN35MOE_PAGE
    qsort(pages, count, sizeof(*pages), qwen35moe_weight_pages_compare);
    size_t merged = 0;
    for (size_t i = 0; i < count; i++) {
        if (merged && pages[i].first <= pages[merged - 1].end) {
            if (pages[i].end > pages[merged - 1].end)
                pages[merged - 1].end = pages[i].end;
        } else pages[merged++] = pages[i];
    }
    *page_size_out = page_size;
    *count_out = merged;
    return pages;
}
static uint64_t qwen35moe_text_model_bytes(const ds4_model *m, const qwen35moe_weights *w) {
    uint64_t page_size, total_pages = 0;
    size_t count;
    struct qwen35moe_weight_pages *pages = qwen35moe_bound_pages(m, w, &page_size, &count);
    for (size_t i = 0; i < count; i++) total_pages += pages[i].end - pages[i].first;
    free(pages);
    return total_pages * page_size;
}
static void qwen35moe_warm_text_weights(const ds4_model *m, const qwen35moe_weights *w) {
    uint64_t page_size, checksum = 0;
    size_t count;
    struct qwen35moe_weight_pages *pages = qwen35moe_bound_pages(m, w, &page_size, &count);
    const double start = now_sec();
    for (size_t i = 0; i < count; i++)
        for (uint64_t page = pages[i].first; page < pages[i].end; page++) {
            uint64_t off = page * page_size;
            if (off < m->size) checksum += m->map[off];
        }
    fprintf(stderr, "ds4: warmed qwen35moe bound text pages in %.3fs (checksum=%llu)\n",
            now_sec() - start, (unsigned long long)checksum);
    free(pages);
}

static size_t qwen35moe_row_bytes(const ds4_tensor *t) {
    return t->dim[0] / gguf_types[t->type].block_elems * gguf_types[t->type].block_bytes;
}
static void qwen35moe_row(const ds4_model *m, const ds4_tensor *t, uint64_t row, float *out) {
    const uint8_t *p = (const uint8_t *)tensor_data(m, t) + row * qwen35moe_row_bytes(t);
    if (t->type == 0)
        memcpy(out, p, t->dim[0] * sizeof(float));
    else if (t->type == 1 || t->type == 30) {
        for (uint64_t i = 0; i < t->dim[0]; i++) {
            uint16_t h;
            memcpy(&h, p + 2 * i, 2);
            if (t->type == 1)
                out[i] = f16_to_f32(h);
            else {
                uint32_t u = (uint32_t)h << 16;
                memcpy(out + i, &u, 4);
            }
        }
    } else
        ds4_qwen35moe_dequant(t->type, p, out, (int)t->dim[0]);
}
struct qwen35moe_matvec_job {
    const ds4_model *model;
    const ds4_tensor *tensor;
    const float *x;
    const void *activation;
    float *out;
    uint64_t row0;
    const int *chosen;
    uint32_t rows_per_expert;
    size_t xstride, astride;
    ds4_qwen35moe_dot_kernel dot;
};
static void qwen35moe_matvec_worker(void *arg, uint64_t begin, uint64_t end) {
    struct qwen35moe_matvec_job *job = arg;
    const ds4_tensor *t = job->tensor;
    const uint8_t *base = tensor_data(job->model, t);
    const size_t stride = qwen35moe_row_bytes(t);
    for (uint64_t r = begin; r < end; r++) {
        uint64_t expert = job->chosen ? r / job->rows_per_expert : 0;
        uint64_t source_row = job->chosen ?
            (uint64_t)job->chosen[expert] * job->rows_per_expert + r % job->rows_per_expert : r;
        const void *row = base + (job->row0 + source_row) * stride;
        const float *x = job->x + expert * job->xstride;
        const uint8_t *activation = (const uint8_t *)job->activation + expert * job->astride;
        if (t->type == 0) {
            const float *p = row;
            job->out[r] = ds4_qwen35moe_float_dot((int)t->dim[0], p, x);
        } else if (t->type == 1 || t->type == 30) {
            float buf[4096];
            qwen35moe_row(job->model, t, job->row0 + source_row, buf);
            double sum = 0;
            for (uint64_t i = 0; i < t->dim[0]; i++)
                sum += (double)buf[i] * x[i];
            job->out[r] = (float)sum;
        } else {
            float value = 0;
            job->dot((int)t->dim[0], &value, 0, row, 0, activation, 0, 1);
            job->out[r] = value;
        }
    }
}
static void qwen35moe_matvec(const ds4_model *m, const ds4_tensor *t, uint64_t row0, uint64_t rows,
                       const float *x, float *out, qwen35moe_state *st) {
    if (t->type != 0 && t->type != 1 && t->type != 30)
        ds4_qwen35moe_quantize(t->type, x, st->activation, (int)t->dim[0]);
    ds4_qwen35moe_dot_kernel dot = t->type != 0 && t->type != 1 && t->type != 30 ?
        ds4_qwen35moe_select_dot(t->type) : NULL;
#if defined(__AVX2__) && defined(__FMA__)
    if (t->type == 12 && rows % 8 == 0 && row0 % 8 == 0)
        dot = ds4_qwen35moe_q4k_gemv_dot;
#endif
    struct qwen35moe_matvec_job job = {.model = m, .tensor = t, .x = x,
        .activation = st->activation, .out = out, .row0 = row0,
        .dot = dot};
    ds4_parallel_for_min_rows(rows, qwen35moe_matvec_worker, &job, 256);
}
static void qwen35moe_matvec_selected(const ds4_model *m, const ds4_tensor *t,
                                const int chosen[8], uint32_t rows_per_expert,
                                const float *x, size_t xstride, float *out,
                                qwen35moe_state *st, bool separate_inputs) {
    uint8_t *activation = separate_inputs ? st->route_activation[0] : (uint8_t *)st->activation;
    size_t astride = separate_inputs ? sizeof(st->route_activation[0]) : 0;
    if (t->type != 0 && t->type != 1 && t->type != 30)
        for (uint32_t rank = 0; rank < (separate_inputs ? 8u : 1u); rank++)
            ds4_qwen35moe_quantize(t->type, x + rank * xstride,
                             activation + rank * astride, (int)t->dim[0]);
    ds4_qwen35moe_dot_kernel dot = t->type != 0 && t->type != 1 && t->type != 30 ?
        ds4_qwen35moe_select_dot(t->type) : NULL;
#if defined(__AVX2__) && defined(__FMA__)
    if (t->type == 12 && rows_per_expert % 8 == 0)
        dot = ds4_qwen35moe_q4k_gemv_dot;
#endif
    struct qwen35moe_matvec_job job = {.model = m, .tensor = t, .x = x,
        .activation = activation, .out = out, .chosen = chosen,
        .rows_per_expert = rows_per_expert, .xstride = xstride, .astride = astride,
        .dot = dot};
    ds4_parallel_for_min_rows(8ull * rows_per_expert, qwen35moe_matvec_worker, &job, 256);
}
/* GGML's quantized dot kernels accept one packed activation at a time. Keep a
 * bounded set of packed activations beside each weight row so prefill streams
 * the large model matrices once per chunk, rather than once per token. */
struct qwen35moe_matbatch_job {
    const ds4_model *model;
    const ds4_tensor *tensor;
    const float *x;
    const uint8_t *activation;
    float *out;
    size_t xstride, ostride, astride;
    uint32_t count;
    uint64_t row0;
    bool sgemm;
    ds4_qwen35moe_dot_kernel dot;
};
static void qwen35moe_matbatch_worker(void *arg, uint64_t begin, uint64_t end) {
    const struct qwen35moe_matbatch_job *job = arg;
    const ds4_tensor *t = job->tensor;
    const uint8_t *base = tensor_data(job->model, t);
    const size_t stride = qwen35moe_row_bytes(t);
    if (t->type == 0 && job->sgemm) {
        for (uint32_t b = 0; b < job->count; b += 2)
            for (uint64_t r = begin; r < end; r += 4)
                ds4_qwen35moe_sgemm_float_tile4x2((int)t->dim[0],
                    (const float *)(base + (job->row0 + r) * stride), t->dim[0],
                    job->x + b * job->xstride, job->xstride,
                    job->out + b * job->ostride + r, job->ostride,
                    (int)(end - r < 4 ? end - r : 4),
                    (int)(job->count - b < 2 ? job->count - b : 2));
        return;
    }
    if (t->type == 1 || t->type == 30) {
        /* Decode F16/BF16 rows once even when many activations share them. */
        for (uint64_t r = begin; r < end; r++) {
            float decoded[4096];
            qwen35moe_row(job->model, t, job->row0 + r, decoded);
            for (uint32_t b = 0; b < job->count; b++) {
                const float *x = job->x + b * job->xstride;
                double sum = 0;
                for (uint64_t i = 0; i < t->dim[0]; i++) sum += (double)decoded[i] * x[i];
                job->out[b * job->ostride + r] = (float)sum;
            }
        }
        return;
    }
    /* GGML's plain MUL_MAT uses 16-by-16 row/activation tiles. The dot
     * kernels and each output's accumulation order remain unchanged. */
    for (uint32_t b0 = 0; b0 < job->count; b0 += 16) {
        uint32_t b1 = b0 + 16 < job->count ? b0 + 16 : job->count;
        for (uint64_t r0 = begin; r0 < end; r0 += 16) {
            uint64_t r1 = r0 + 16 < end ? r0 + 16 : end;
            for (uint32_t b = b0; b < b1; b++) {
                const float *x = job->x + b * job->xstride;
                const uint8_t *activation = job->activation + b * job->astride;
                for (uint64_t r = r0; r < r1; r++) {
                    const void *row = base + (job->row0 + r) * stride;
                    float value;
                    if (t->type == 0)
                        value = job->sgemm ? ds4_qwen35moe_sgemm_float_dot((int)t->dim[0], row, x) :
                                             ds4_qwen35moe_float_dot((int)t->dim[0], row, x);
                    else {
                        value = 0;
                        job->dot((int)t->dim[0], &value, 0, row, 0, activation, 0, 1);
                    }
                    job->out[b * job->ostride + r] = value;
                }
            }
        }
    }
}
static void qwen35moe_panel_batch_worker(void *arg, uint64_t begin, uint64_t end) {
    const struct qwen35moe_matbatch_job *job = arg;
    const ds4_tensor *t = job->tensor;
    const uint8_t *weights = tensor_data(job->model, t) +
        (job->row0 + begin * 8) * qwen35moe_row_bytes(t);
    ds4_qwen35moe_iqp_matbatch(t->type, (int)t->dim[0], weights, qwen35moe_row_bytes(t),
                          job->activation, job->astride, job->out + begin * 8,
                          job->ostride, (int)((end - begin) * 8), (int)job->count);
}
static void qwen35moe_q4k_batch_worker(void *arg, uint64_t begin, uint64_t end) {
    const struct qwen35moe_matbatch_job *job = arg;
    const ds4_tensor *t = job->tensor;
    const uint8_t *weights = tensor_data(job->model, t) +
        (job->row0 + begin * 8) * qwen35moe_row_bytes(t);
    ds4_qwen35moe_q4k_matbatch((int)t->dim[0], weights, qwen35moe_row_bytes(t),
                           job->activation, job->astride, job->out + begin * 8,
                           job->ostride, (int)((end - begin) * 8), (int)job->count);
}
static void qwen35moe_matbatch(const ds4_model *m, const ds4_tensor *t, uint64_t row0, uint64_t rows,
                         const float *x, size_t xstride, float *out, size_t ostride,
                         uint32_t count, uint8_t *activation, size_t astride) {
    if (!count) return;
    if (t->type != 0 && t->type != 1 && t->type != 30)
        for (uint32_t b = 0; b < count; b++)
            ds4_qwen35moe_quantize(t->type, x + b * xstride, activation + b * astride, (int)t->dim[0]);
    struct qwen35moe_matbatch_job job = {m, t, x, activation, out, xstride, ostride,
                                   astride, count, row0,
                                   t->type == 0 && count >= 2 && rows % 4 == 0 &&
                                   t->dim[0] % 8 == 0, NULL};
    if (t->type != 0 && t->type != 1 && t->type != 30)
        job.dot = ds4_qwen35moe_select_dot(t->type);
#if defined(__AVX2__) && defined(__FMA__)
    if (t->type == 12 && rows % 8 == 0 && row0 % 8 == 0)
        job.dot = ds4_qwen35moe_q4k_gemv_dot;
    /* GGML reserves an IQ panel for an outer MUL_MAT_ID batch of eight, but
     * dispatches it only when this individual expert has eight selections. */
    if ((t->type == 16 || t->type == 17) && rows % 8 == 0 && row0 % 8 == 0 &&
        count >= 8) {
        ds4_parallel_for_min_rows(rows / 8, qwen35moe_panel_batch_worker, &job, 8);
        return;
    }
    if (t->type == 12 && rows % 8 == 0 && row0 % 8 == 0 && count >= 4) {
        ds4_parallel_for_min_rows(rows / 8, qwen35moe_q4k_batch_worker, &job, 8);
        return;
    }
#endif
    ds4_parallel_for_min_rows(rows, qwen35moe_matbatch_worker, &job, 128);
}
#define QWEN35MOE_MV(t, x, out) qwen35moe_matvec(m, t, 0, (t)->ndim > 1 ? (t)->dim[1] : 1, x, out, st)

static qwen35moe_state *qwen35moe_state_new(uint32_t cap) {
    qwen35moe_state *st = xcalloc(1, sizeof(*st));
    st->cap = cap;
    st->timing = qwen35moe_env("TIMING") != NULL;
    st->recurrent = xcalloc(30ull * 32 * 128 * 128, sizeof(float));
    st->history = xcalloc(30ull * 3 * 8192, sizeof(float));
    st->keys = xcalloc(10ull * cap * 512, sizeof(uint16_t));
    st->values = xcalloc(10ull * cap * 512, sizeof(uint16_t));
    return st;
}
static void qwen35moe_state_reset(qwen35moe_state *st) {
    memset(st->recurrent, 0, 30ull * 32 * 128 * 128 * sizeof(float));
    memset(st->history, 0, 30ull * 3 * 8192 * sizeof(float));
}
static void qwen35moe_state_free(qwen35moe_state *st) {
    if (!st)
        return;
    if (st->timing && st->timed_tokens)
        fprintf(stderr, "ds4: qwen35moe stage timing: %llu tokens, attention %.3fs (recurrent %.3fs, full attention %.3fs), FFN %.3fs, output %.3fs\n",
                (unsigned long long)st->timed_tokens, st->attention_seconds,
                st->linear_finish_seconds, st->full_attention_finish_seconds,
                st->feedforward_seconds, st->output_seconds);
    if (st->timing && st->timed_tokens)
        fprintf(stderr, "ds4: qwen35moe batched row timing: first norm %.3fs, post rows %.3fs, router %.3fs, convolution prepare %.3fs\n",
                st->batch_norm_seconds, st->batch_post_rows_seconds,
                st->batch_router_seconds, st->batch_conv_prepare_seconds);
    free(st->recurrent);
    free(st->history);
    free(st->keys);
    free(st->values);
    free(st);
}
static const float *qwen35moe_control(const ds4_model *m, const ds4_tensor *t) {
    return tensor_data(m, t);
}
static void qwen35moe_rope(float *x, uint32_t pos) {
    float theta = (float)pos, step = powf(10000000.0f, -2.0f / 64.0f);
    for (uint32_t i = 0; i < 32; i++) {
        float c = cosf(theta), s = sinf(theta), a = x[i], b = x[i + 32];
        x[i] = fmaf(a, c, -b * s);
        x[i + 32] = fmaf(a, s, b * c);
        theta *= step;
    }
}
static void qwen35moe_dump(qwen35moe_state *st, const char *name, uint32_t il, const float *x, uint32_t n) {
    const char *prefix = qwen35moe_env("STAGE_DUMP");
    if (!prefix || !prefix[0])
        return;
    const char *only_pos = qwen35moe_env("STAGE_POS");
    const char *min_pos = qwen35moe_env("STAGE_MIN_POS");
    const char *only_layer = qwen35moe_env("STAGE_LAYER");
    if ((only_pos && strtoul(only_pos, NULL, 10) != st->pos) ||
        (min_pos && st->pos < strtoul(min_pos, NULL, 10)) ||
        (only_layer && strtoul(only_layer, NULL, 10) != il)) return;
    char path[1024];
    snprintf(path, sizeof(path), "%s.%u.%s-%u.bin", prefix, st->pos, name, il);
    FILE *fp = fopen(path, "wb");
    if (fp) {
        fwrite(x, sizeof(float), n, fp);
        fclose(fp);
    }
}
static double qwen35moe_clock_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}
struct qwen35moe_recurrent_job {
    const ds4_model *m;
    const struct qwen35moe_layer *layer;
    qwen35moe_state *state;
    const float *q, *k, *v;
    uint32_t index;
};
static inline void qwen35moe_recurrent_step(float *S, const float *qh, const float *kh,
                                       const float *vh, const float *z, float decay,
                                       float beta, const float *weight, float *out) {
    float delta[128];
    for (uint32_t dv = 0; dv < 128; dv++) {
        for (uint32_t dk = 0; dk < 128; dk++) S[dv * 128 + dk] *= decay;
        float sum = ds4_qwen35moe_float_dot(128, S + dv * 128, kh);
        delta[dv] = (vh[dv] - sum) * beta;
    }
    for (uint32_t dv = 0; dv < 128; dv++) {
        for (uint32_t dk = 0; dk < 128; dk++)
            S[dv * 128 + dk] = fmaf(kh[dk], delta[dv], S[dv * 128 + dk]);
        out[dv] = ds4_qwen35moe_float_dot(128, S + dv * 128, qh) * (1 / sqrtf(128));
    }
    float norm[128];
    qwen35moe_rms(norm, out, weight, 128, 1e-6f);
    for (uint32_t d = 0; d < 128; d++) out[d] = norm[d] * z[d];
}
static void qwen35moe_recurrent_worker(void *arg, uint64_t begin, uint64_t end) {
    const struct qwen35moe_recurrent_job *job = arg;
    qwen35moe_state *st = job->state;
    const float *A = qwen35moe_control(job->m, job->layer->a);
    const float *dt = qwen35moe_control(job->m, job->layer->dt);
    const float *weight = qwen35moe_control(job->m, job->layer->ssm_norm);
    for (uint32_t h = (uint32_t)begin; h < (uint32_t)end; h++) {
        float decay = expf(A[h] * qwen35moe_softplus(st->alpha[h] + dt[h]));
        float beta = qwen35moe_sigmoid(st->beta[h]);
        float *S = st->recurrent + ((uint64_t)job->index * 32 + h) * 128 * 128;
        const float *qh = job->q + (h % 16) * 128;
        const float *kh = job->k + (h % 16) * 128;
        const float *vh = job->v + h * 128;
        qwen35moe_recurrent_step(S, qh, kh, vh, st->z + h * 128, decay, beta,
                            weight, st->heads + h * 128);
    }
}
struct qwen35moe_recurrent_batch_job {
    const ds4_model *m;
    const struct qwen35moe_layer *layer;
    qwen35moe_state *state;
    const float *qkv, *z, *alpha, *beta;
    float *heads;
    uint32_t count, index;
};
static void qwen35moe_recurrent_batch_worker(void *arg, uint64_t begin, uint64_t end) {
    const struct qwen35moe_recurrent_batch_job *job = arg;
    const float *A = qwen35moe_control(job->m, job->layer->a);
    const float *dt = qwen35moe_control(job->m, job->layer->dt);
    const float *weight = qwen35moe_control(job->m, job->layer->ssm_norm);
    for (uint32_t h = (uint32_t)begin; h < (uint32_t)end; h++) {
        float *S = job->state->recurrent + ((uint64_t)job->index * 32 + h) * 128 * 128;
        for (uint32_t b = 0; b < job->count; b++) {
            const float *mixed = job->qkv + b * 8192;
            const float *qh = mixed + (h % 16) * 128;
            const float *kh = mixed + 2048 + (h % 16) * 128;
            const float *vh = mixed + 4096 + h * 128;
            float decay = expf(A[h] * qwen35moe_softplus(job->alpha[b * 32 + h] + dt[h]));
            float beta = qwen35moe_sigmoid(job->beta[b * 32 + h]);
            qwen35moe_recurrent_step(S, qh, kh, vh, job->z + b * 4096 + h * 128,
                                decay, beta, weight, job->heads + b * 4096 + h * 128);
        }
    }
}
static void qwen35moe_linear_prepare(const ds4_model *m, const struct qwen35moe_layer *l, uint32_t il,
                               qwen35moe_state *st) {
    qwen35moe_dump(st, "linear_attn_qkv_mixed", il, st->qkv, 8192);
    qwen35moe_dump(st, "z", il, st->z, 4096);
    uint32_t index = il - il / 4;
    float *history = st->history + (uint64_t)index * 3 * 8192;
    const float *cw = qwen35moe_control(m, l->conv);
    for (uint32_t c = 0; c < 8192; c++) {
        float sum = 0;
        for (uint32_t k = 0; k < 3; k++)
            sum += cw[c * 4 + k] * history[k * 8192 + c];
        sum += cw[c * 4 + 3] * st->qkv[c];
        st->mixed[c] = sum;
    }
    qwen35moe_dump(st, "conv_output_raw", il, st->mixed, 8192);
    ds4_qwen35moe_silu(8192, st->mixed, st->mixed);
    qwen35moe_dump(st, "conv_output_silu", il, st->mixed, 8192);
    ds4_qwen35moe_silu(4096, st->z, st->z);
    memmove(history, history + 8192, 2 * 8192 * sizeof(float));
    memcpy(history + 2 * 8192, st->qkv, 8192 * sizeof(float));
    float *q = st->mixed, *k = q + 2048;
    for (uint32_t h = 0; h < 16; h++) {
        for (uint32_t which = 0; which < 2; which++) {
            float *row = (which ? k : q) + h * 128;
            double sum = 0;
            for (uint32_t d = 0; d < 128; d++)
                sum += (double)(row[d] * row[d]);
            float mean = (float)(sum / 128), scale = 1 / sqrtf(mean + 1e-6f / 128);
            for (uint32_t d = 0; d < 128; d++)
                row[d] = (row[d] * scale) * (1 / sqrtf(128));
        }
    }
    qwen35moe_dump(st, "q_conv_predelta", il, q, 2048);
    qwen35moe_dump(st, "k_conv_predelta", il, k, 2048);
}
struct qwen35moe_linear_prepare_job {
    const float *raw, *history, *conv;
    float *prepared, *z;
};
static void qwen35moe_linear_prepare_worker(void *arg, uint64_t begin, uint64_t end) {
    const struct qwen35moe_linear_prepare_job *job = arg;
    for (uint64_t b = begin; b < end; b++) {
        float *mixed = job->prepared + b * 8192;
        const float *current = job->raw + b * 8192;
        for (uint32_t c = 0; c < 8192; c++) {
            float sum = 0;
            for (uint32_t k = 0; k < 3; k++) {
                int previous = (int)b + (int)k - 3;
                const float *row = previous >= 0 ? job->raw + (size_t)previous * 8192 :
                    job->history + (b + k) * 8192;
                sum += job->conv[c * 4 + k] * row[c];
            }
            sum += job->conv[c * 4 + 3] * current[c];
            mixed[c] = sum;
        }
        ds4_qwen35moe_silu(8192, mixed, mixed);
        ds4_qwen35moe_silu(4096, job->z + b * 4096, job->z + b * 4096);
        float *q = mixed, *key = mixed + 2048;
        for (uint32_t h = 0; h < 16; h++) {
            for (uint32_t which = 0; which < 2; which++) {
                float *row = (which ? key : q) + h * 128;
                double sum = 0;
                for (uint32_t d = 0; d < 128; d++)
                    sum += (double)(row[d] * row[d]);
                float mean = (float)(sum / 128), scale = 1 / sqrtf(mean + 1e-6f / 128);
                for (uint32_t d = 0; d < 128; d++)
                    row[d] = (row[d] * scale) * (1 / sqrtf(128));
            }
        }
    }
}
static void qwen35moe_linear_finish(const ds4_model *m, const struct qwen35moe_layer *l, uint32_t il,
                              qwen35moe_state *st) {
    qwen35moe_linear_prepare(m, l, il, st);
    uint32_t index = il - il / 4;
    struct qwen35moe_recurrent_job job = {m, l, st, st->mixed,
                                     st->mixed + 2048, st->mixed + 4096, index};
    ds4_parallel_for_min_rows(32, qwen35moe_recurrent_worker, &job, 1);
    qwen35moe_dump(st, "final_output", il, st->heads, 4096);
}
static void qwen35moe_linear(const ds4_model *m, const struct qwen35moe_layer *l, uint32_t il, qwen35moe_state *st) {
    QWEN35MOE_MV(l->qkv, st->norm, st->qkv);
    QWEN35MOE_MV(l->z, st->norm, st->z);
    QWEN35MOE_MV(l->alpha, st->norm, st->alpha);
    QWEN35MOE_MV(l->beta, st->norm, st->beta);
    double begin = st->timing ? qwen35moe_clock_seconds() : 0;
    qwen35moe_linear_finish(m, l, il, st);
    if (st->timing) st->linear_finish_seconds += qwen35moe_clock_seconds() - begin;
    QWEN35MOE_MV(l->out, st->heads, st->attn);
}
struct qwen35moe_attention_head_job {
    const ds4_model *m;
    const struct qwen35moe_layer *l;
    qwen35moe_state *st;
    const uint16_t *keys, *values;
    uint32_t pos;
    bool single_query;
};
static void qwen35moe_attention_split_head(const uint16_t *keys, const uint16_t *values,
                                     const float *query, uint32_t pos, uint32_t kvh,
                                     uint32_t n_threads, float *out) {
    const uint32_t padded = (pos + 256u) & ~255u;
    const uint32_t chunk_size = (padded + n_threads - 1) / n_threads;
    float partial[DS4_MAX_THREADS][258];
    for (uint32_t c = 0; c < n_threads; c++) {
        const uint32_t first = c * chunk_size;
        const uint32_t last = first + chunk_size < padded ? first + chunk_size : padded;
        float *part = partial[c];
        float max = -INFINITY, sum = 0;
        uint16_t accum[256] = {0};
        for (uint32_t t = first; t < last && t <= pos; t++) {
            float key[256];
            ds4_qwen35moe_f16_to_f32(256, keys + (uint64_t)t * 512 + kvh * 256, key);
            float score = ds4_qwen35moe_float_dot(256, key, query) * (1.0f / 16);
            float ms = 1, vs = 1;
            if (score > max) {
                ms = expf(max - score);
                max = score;
                ds4_qwen35moe_f16_scale(256, accum, ms);
            } else vs = expf(score - max);
            ds4_qwen35moe_f16_mad(256, accum, values + (uint64_t)t * 512 + kvh * 256, vs);
            sum = fmaf(sum, ms, vs);
        }
        part[0] = max;
        part[1] = sum;
        ds4_qwen35moe_f16_to_f32(256, accum, part + 2);
    }
    float final_max = -INFINITY, final_sum = 0;
    memset(out, 0, 256 * sizeof(float));
    for (uint32_t c = 0; c < n_threads; c++) {
        const float *part = partial[c];
        if (part[1] == 0) continue;
        const float next_max = fmaxf(final_max, part[0]);
        const float old_scale = expf(final_max - next_max);
        const float new_scale = expf(part[0] - next_max);
        for (uint32_t d = 0; d < 256; d++)
            out[d] = fmaf(out[d], old_scale, part[2 + d] * new_scale);
        final_sum = fmaf(final_sum, old_scale, part[1] * new_scale);
        final_max = next_max;
    }
    if (final_sum != 0) {
        float inv = 1 / final_sum;
        for (uint32_t d = 0; d < 256; d++) out[d] *= inv;
    }
}
static void qwen35moe_attention_head_worker(void *arg, uint64_t begin, uint64_t end) {
    const struct qwen35moe_attention_head_job *job = arg;
    const ds4_model *m = job->m;
    const struct qwen35moe_layer *l = job->l;
    qwen35moe_state *st = job->st;
    const uint16_t *keys = job->keys, *values = job->values;
    const uint32_t pos = job->pos;
    for (uint32_t h = (uint32_t)begin; h < (uint32_t)end; h++) {
        float query[256];
        qwen35moe_rms(query, st->qkv + h * 512, qwen35moe_control(m, l->qn), 256, 1e-6f);
        qwen35moe_rope(query, pos);
        memcpy(st->z + h * 256, query, 256 * sizeof(float));
        uint32_t kvh = h / 8;
        float max = -INFINITY, sum_weights = 0;
        uint16_t accum[256] = {0};
        for (uint32_t d = 0; d < 256; d++)
            query[d] = f16_to_f32(f32_to_f16(query[d]));
        if (job->single_query && ((pos + 256u) & ~255u) >= 512u) {
            qwen35moe_attention_split_head(keys, values, query, pos, kvh,
                                     g_pool.n_threads ? g_pool.n_threads : 1,
                                     st->heads + h * 256);
            continue;
        }
        /* Match GGML's one-query F16 attention: online softmax with an
         * F16 value accumulator. Reusing the reference's precision policy
         * avoids quantization discontinuities in subsequent projections. */
        for (uint32_t t = 0; t <= pos; t++) {
            float key[256];
            ds4_qwen35moe_f16_to_f32(256, keys + (uint64_t)t * 512 + kvh * 256, key);
            float score = ds4_qwen35moe_float_dot(256, key, query) * (1.0f / 16);
            float ms = 1, vs = 1;
            if (score > max) {
                ms = expf(max - score);
                max = score;
                ds4_qwen35moe_f16_scale(256, accum, ms);
            } else
                vs = expf(score - max);
            ds4_qwen35moe_f16_mad(256, accum, values + (uint64_t)t * 512 + kvh * 256, vs);
            sum_weights = fmaf(sum_weights, ms, vs);
        }
        float inv = 1 / sum_weights;
        ds4_qwen35moe_f16_to_f32(256, accum, st->heads + h * 256);
        for (uint32_t d = 0; d < 256; d++)
            st->heads[h * 256 + d] *= inv;
    }
}
static void qwen35moe_attention_prepare_kv(const ds4_model *m, const struct qwen35moe_layer *l, uint32_t il,
                                     uint32_t pos, qwen35moe_state *st) {
    uint16_t *keys = st->keys + (uint64_t)(il / 4) * st->cap * 512;
    uint16_t *values = st->values + (uint64_t)(il / 4) * st->cap * 512;
    bool dump_kv = qwen35moe_env("STAGE_DUMP") != NULL;
    if (dump_kv) qwen35moe_dump(st, "Vcur", il, st->mixed + 512, 512);
    for (uint32_t h = 0; h < 2; h++) {
        float key[256];
        qwen35moe_rms(key, st->mixed + h * 256, qwen35moe_control(m, l->kn), 256, 1e-6f);
        qwen35moe_rope(key, pos);
        if (dump_kv) memcpy(st->mixed + 1024 + h * 256, key, 256 * sizeof(float));
        for (uint32_t d = 0; d < 256; d++) {
            keys[pos * 512 + h * 256 + d] = f32_to_f16(key[d]);
            values[pos * 512 + h * 256 + d] = f32_to_f16(st->mixed[512 + h * 256 + d]);
        }
    }
    if (dump_kv) qwen35moe_dump(st, "Kcur", il, st->mixed + 1024, 512);
}
static void qwen35moe_attention_finish(const ds4_model *m, const struct qwen35moe_layer *l, uint32_t il,
                                 uint32_t pos, qwen35moe_state *st, bool single_query) {
    qwen35moe_attention_prepare_kv(m, l, il, pos, st);
    uint16_t *keys = st->keys + (uint64_t)(il / 4) * st->cap * 512;
    uint16_t *values = st->values + (uint64_t)(il / 4) * st->cap * 512;
    struct qwen35moe_attention_head_job job = {m, l, st, keys, values, pos, single_query};
    if (pos >= 32) ds4_parallel_for_min_rows(16, qwen35moe_attention_head_worker, &job, 1);
    else qwen35moe_attention_head_worker(&job, 0, 16);
    qwen35moe_dump(st, "Qcur", il, st->z, 4096);
    qwen35moe_dump(st, "attn_pregate", il, st->heads, 4096);
    for (uint32_t h = 0; h < 16; h++)
        for (uint32_t d = 0; d < 256; d++)
            st->heads[h * 256 + d] *= qwen35moe_sigmoid(st->qkv[h * 512 + 256 + d]);
    qwen35moe_dump(st, "final_output", il, st->heads, 4096);
}
struct qwen35moe_attn_tile_job {
    const uint16_t *keys, *values;
    const float *queries;
    float *heads;
    uint32_t start, count;
};
static void qwen35moe_attention_tile_worker(void *arg, uint64_t begin, uint64_t end) {
    const struct qwen35moe_attn_tile_job *job = arg;
    const uint32_t kv_end = job->start + job->count;
    /* Pinned GGML CPU flash attention uses Q64/KV64 and an F32 accumulator.
     * These arrays are reused per head and never scale with context length. */
    float q[64 * 256], kt[256 * 64], v[64 * 256];
    float scores[64 * 64], accum[64 * 256], sums[64], maxima[64];
    float half_row[256];
    for (uint32_t h = (uint32_t)begin; h < (uint32_t)end; h++) {
        const uint32_t kvh = h / 8;
        for (uint32_t q0 = 0; q0 < job->count; q0 += 64) {
            const uint32_t rows = job->count - q0 < 64 ? job->count - q0 : 64;
            for (uint32_t r = 0; r < rows; r++)
                memcpy(q + r * 256, job->queries + (q0 + r) * 4096 + h * 256,
                       256 * sizeof(float));
            if (rows < 64) memset(q + rows * 256, 0, (64 - rows) * 256 * sizeof(float));
            memset(accum, 0, sizeof(accum));
            for (uint32_t r = 0; r < 64; r++) { sums[r] = 0; maxima[r] = -INFINITY; }
            for (uint32_t tile = 0; tile < kv_end; tile += 64) {
                const uint32_t cols = kv_end - tile < 64 ? kv_end - tile : 64;
                if (tile > job->start + q0 + rows - 1) break;
                memset(kt, 0, sizeof(kt));
                memset(v, 0, sizeof(v));
                for (uint32_t k = 0; k < cols; k++) {
                    const uint16_t *key = job->keys + (uint64_t)(tile + k) * 512 + kvh * 256;
                    const uint16_t *value = job->values + (uint64_t)(tile + k) * 512 + kvh * 256;
                    ds4_qwen35moe_f16_to_f32(256, key, half_row);
                    for (uint32_t d = 0; d < 256; d++) kt[d * 64 + k] = half_row[d];
                    ds4_qwen35moe_f16_to_f32(256, value, v + k * 256);
                }
                memset(scores, 0, sizeof(scores));
                ds4_qwen35moe_gemm_f32(scores, q, kt, 64, 256, 64);
                for (uint32_t r = 0; r < rows; r++) {
                    float *row = scores + r * 64;
                    uint32_t pos = job->start + q0 + r;
                    for (uint32_t k = 0; k < 64; k++) {
                        row[k] *= 1.0f / 16;
                        if (k >= cols || tile + k > pos) row[k] = -INFINITY;
                    }
                    float tile_max = -INFINITY;
                    for (uint32_t k = 0; k < 64; k++)
                        if (row[k] > tile_max) tile_max = row[k];
                    if (tile_max == -INFINITY) {
                        memset(row, 0, 64 * sizeof(float));
                        continue;
                    }
                    const float next_max = fmaxf(maxima[r], tile_max);
                    if (next_max > maxima[r]) {
                        const float scale = expf(maxima[r] - next_max);
                        for (uint32_t d = 0; d < 256; d++) accum[r * 256 + d] *= scale;
                        sums[r] *= scale;
                    }
                    maxima[r] = next_max;
                    sums[r] += ds4_qwen35moe_softmax(64, row, row, next_max);
                }
                /* Masked query rows are zero so this GEMM preserves the
                 * reference's K-order FMA accumulation for every live row. */
                ds4_qwen35moe_gemm_f32(accum, scores, v, 64, 64, 256);
            }
            for (uint32_t r = 0; r < rows; r++) {
                float inv = sums[r] == 0 ? 0 : 1 / sums[r];
                float *out = job->heads + (q0 + r) * 4096 + h * 256;
                for (uint32_t d = 0; d < 256; d++) out[d] = accum[r * 256 + d] * inv;
            }
        }
    }
}
static void qwen35moe_attention_tiled(const ds4_model *m, const struct qwen35moe_layer *l,
                                uint32_t il, uint32_t start, uint32_t count,
                                const float *qkv, const float *mixed, float *queries,
                                float *heads, qwen35moe_state *st) {
    for (uint32_t b = 0; b < count; b++) {
        st->pos = start + b;
        memcpy(st->qkv, qkv + b * 8192, 8192 * sizeof(float));
        memcpy(st->mixed, mixed + b * 1024, 1024 * sizeof(float));
        qwen35moe_attention_prepare_kv(m, l, il, start + b, st);
        for (uint32_t h = 0; h < 16; h++) {
            float *query = queries + b * 4096 + h * 256;
            qwen35moe_rms(query, st->qkv + h * 512, qwen35moe_control(m, l->qn), 256, 1e-6f);
            qwen35moe_rope(query, start + b);
        }
        qwen35moe_dump(st, "Qcur", il, queries + b * 4096, 4096);
    }
    const uint16_t *keys = st->keys + (uint64_t)(il / 4) * st->cap * 512;
    const uint16_t *values = st->values + (uint64_t)(il / 4) * st->cap * 512;
    struct qwen35moe_attn_tile_job job = {keys, values, queries, heads, start, count};
    ds4_parallel_for_min_rows(16, qwen35moe_attention_tile_worker, &job, 1);
    for (uint32_t b = 0; b < count; b++) {
        st->pos = start + b;
        qwen35moe_dump(st, "attn_pregate", il, heads + b * 4096, 4096);
        for (uint32_t h = 0; h < 16; h++)
            for (uint32_t d = 0; d < 256; d++)
                heads[b * 4096 + h * 256 + d] *=
                    qwen35moe_sigmoid(qkv[b * 8192 + h * 512 + 256 + d]);
        qwen35moe_dump(st, "final_output", il, heads + b * 4096, 4096);
    }
}
static void qwen35moe_attention(const ds4_model *m, const struct qwen35moe_layer *l, uint32_t il, uint32_t pos,
                          qwen35moe_state *st) {
    QWEN35MOE_MV(l->q, st->norm, st->qkv);
    QWEN35MOE_MV(l->k, st->norm, st->mixed);
    QWEN35MOE_MV(l->v, st->norm, st->mixed + 512);
    double begin = st->timing ? qwen35moe_clock_seconds() : 0;
    qwen35moe_attention_finish(m, l, il, pos, st, true);
    if (st->timing) st->full_attention_finish_seconds += qwen35moe_clock_seconds() - begin;
    QWEN35MOE_MV(l->out, st->heads, st->attn);
}
static void qwen35moe_moe(const ds4_model *m, const struct qwen35moe_layer *l, qwen35moe_state *st) {
    QWEN35MOE_MV(l->router, st->norm, st->router);
    int chosen[8];
    float weights[8], max = -FLT_MAX;
    for (uint32_t i = 0; i < 256; i++)
        if (st->router[i] > max)
            max = st->router[i];
    double total = ds4_qwen35moe_softmax(256, st->router, st->router, max);
    float inv_total = (float)(1.0 / total);
    for (uint32_t i = 0; i < 256; i++)
        st->router[i] *= inv_total;
    double weight_sum = 0;
    ds4_qwen35moe_router_top8(st->router, chosen);
    for (uint32_t i = 0; i < 8; i++) {
        weights[i] = st->router[chosen[i]];
        weight_sum += weights[i];
    }
    float denom = (float)weight_sum;
    memset(st->ffn, 0, sizeof(st->ffn));
    /* Quantize the common input once per matrix and dispatch all selected
     * rows together. Each expert still accumulates in its original rank. */
    qwen35moe_matvec_selected(m, l->gate, chosen, 512, st->norm, 0,
                        st->route_gate, st, false);
    qwen35moe_matvec_selected(m, l->up, chosen, 512, st->norm, 0,
                        st->route_up, st, false);
    for (uint32_t i = 0; i < 8; i++) {
        float *gate = st->route_gate + i * 512, *up = st->route_up + i * 512;
        ds4_qwen35moe_silu(512, gate, gate);
        for (uint32_t d = 0; d < 512; d++)
            gate[d] *= up[d];
    }
    qwen35moe_matvec_selected(m, l->down, chosen, 2048, st->route_gate, 512,
                        st->route_down, st, true);
    for (uint32_t i = 0; i < 8; i++) {
        for (uint32_t d = 0; d < 2048; d++)
            st->ffn[d] += st->route_down[i * 2048 + d] * (weights[i] / denom);
    }
    float shared[1];
    QWEN35MOE_MV(l->shared_router, st->norm, shared);
    QWEN35MOE_MV(l->shared_gate, st->norm, st->gate);
    QWEN35MOE_MV(l->shared_up, st->norm, st->up);
    ds4_qwen35moe_silu(512, st->gate, st->gate);
    for (uint32_t d = 0; d < 512; d++)
        st->gate[d] *= st->up[d];
    QWEN35MOE_MV(l->shared_down, st->gate, st->down);
    float g = qwen35moe_sigmoid(shared[0]);
    for (uint32_t d = 0; d < 2048; d++)
        st->ffn[d] += st->down[d] * g;
}
static int qwen35moe_forward(ds4_session *s, qwen35moe_state *st, int token, uint32_t pos, float *logits,
                       char *err, size_t errlen) {
    if (qwen35moe_require_model_valid(s->engine, err, errlen)) return 1;
    if (token < 0 || token >= 248320 || pos >= st->cap) {
        snprintf(err, errlen, "qwen35moe token or context position out of range");
        return 1;
    }
    const ds4_model *m = &s->engine->model;
    const qwen35moe_weights *w = s->engine->qwen35moe_weights;
    st->pos = pos;
    qwen35moe_row(m, w->embedding, (uint32_t)token, st->x);
    for (uint32_t il = 0; il < 40; il++) {
        if (ds4_session_cancelled(s)) {
            snprintf(err, errlen, "interrupted");
            return DS4_SESSION_SYNC_INTERRUPTED;
        }
        const struct qwen35moe_layer *l = &w->layer[il];
        double stage_start = st->timing ? qwen35moe_clock_seconds() : 0;
        qwen35moe_rms(st->norm, st->x, qwen35moe_control(m, l->norm), 2048, 1e-6f);
        qwen35moe_dump(st, "attn_norm", il, st->norm, 2048);
        if (il % 4 == 3)
            qwen35moe_attention(m, l, il, pos, st);
        else
            qwen35moe_linear(m, l, il, st);
        qwen35moe_dump(st, "attn_output", il, st->attn, 2048);
        for (uint32_t d = 0; d < 2048; d++)
            st->x[d] += st->attn[d];
        if (st->timing) {
            st->attention_seconds += qwen35moe_clock_seconds() - stage_start;
            stage_start = qwen35moe_clock_seconds();
        }
        qwen35moe_rms(st->norm, st->x, qwen35moe_control(m, l->post_norm), 2048, 1e-6f);
        qwen35moe_dump(st, "attn_post_norm", il, st->norm, 2048);
        qwen35moe_moe(m, l, st);
        qwen35moe_dump(st, "ffn_out", il, st->ffn, 2048);
        for (uint32_t d = 0; d < 2048; d++)
            st->x[d] += st->ffn[d];
        if (st->timing) st->feedforward_seconds += qwen35moe_clock_seconds() - stage_start;
        const char *dump = qwen35moe_env("LAYER_DUMP");
        if (dump && dump[0]) {
            char path[1024];
            snprintf(path, sizeof(path), "%s.%u.%u.bin", dump, pos, il);
            FILE *fp = fopen(path, "wb");
            if (fp) {
                fwrite(st->x, sizeof(float), 2048, fp);
                fclose(fp);
            }
        }
    }
    /* Prefill only needs the final token's logits. The output matrix has
     * 248320 rows, so projecting every earlier token wastes most of prefill. */
    if (logits) {
        double stage_start = st->timing ? qwen35moe_clock_seconds() : 0;
        qwen35moe_rms(st->norm, st->x, qwen35moe_control(m, w->norm), 2048, 1e-6f);
        QWEN35MOE_MV(w->output, st->norm, logits);
        if (st->timing) st->output_seconds += qwen35moe_clock_seconds() - stage_start;
    }
    if (st->timing) st->timed_tokens++;
    return 0;
}

#define QWEN35MOE_BATCH_MAX 1024
/* Bounded layerwise prefill is the normal Qwen CPU path. The override retains
 * a one-token diagnostic for reference comparisons and chunk-boundary tests. */
static uint32_t qwen35moe_prefill_chunk(void) {
    const char *value = qwen35moe_env("PREFILL_BATCH");
    if (!value || !value[0]) return QWEN35MOE_BATCH_MAX;
    char *end;
    unsigned long count = strtoul(value, &end, 10);
    return *end || count < 1 || count > QWEN35MOE_BATCH_MAX ? 1 : (uint32_t)count;
}
struct qwen35moe_expert_batch_job {
    const ds4_model *model;
    const struct qwen35moe_layer *layer;
    const float *norm;
    float *route_output;
    const int (*chosen)[8];
    uint32_t count;
    size_t activation_stride;
    const uint16_t *active_experts;
    uint32_t active_count;
    uint32_t next_expert;
};
struct qwen35moe_router_job {
    float *router;
    int (*chosen)[8];
    float (*weights)[8];
    float *denominators;
};
static void qwen35moe_router_worker(void *arg, uint64_t begin, uint64_t end) {
    struct qwen35moe_router_job *job = arg;
    for (uint64_t b = begin; b < end; b++) {
        float *p = job->router + b * 256, max = -FLT_MAX;
        for (uint32_t e = 0; e < 256; e++) if (p[e] > max) max = p[e];
        double total = ds4_qwen35moe_softmax(256, p, p, max);
        float inv = (float)(1.0 / total);
        for (uint32_t e = 0; e < 256; e++) p[e] *= inv;
        double weight_sum = 0;
        ds4_qwen35moe_router_top8(p, job->chosen[b]);
        for (uint32_t rank = 0; rank < 8; rank++) {
            job->weights[b][rank] = p[job->chosen[b][rank]];
            weight_sum += job->weights[b][rank];
        }
        job->denominators[b] = (float)weight_sum;
    }
}
static void qwen35moe_expert_batch_worker(void *arg, uint64_t begin, uint64_t end) {
    struct qwen35moe_expert_batch_job *job = arg;
    (void)begin;
    (void)end;
    const uint32_t count = job->count;
    float *work = xmalloc((size_t)count * (2048 + 512 + 512 + 2048) * sizeof(float));
    float *work_x = work;
    float *work_gate = work_x + (size_t)count * 2048;
    float *work_up = work_gate + (size_t)count * 512;
    float *work_down = work_up + (size_t)count * 512;
    uint8_t *activation = xmalloc((size_t)count * job->activation_stride);
    uint32_t token_at[QWEN35MOE_BATCH_MAX], rank_at[QWEN35MOE_BATCH_MAX];
    for (;;) {
        uint32_t index = __atomic_fetch_add(&job->next_expert, 1u, __ATOMIC_RELAXED);
        if (index >= job->active_count) break;
        uint32_t expert = job->active_experts[index];
        uint32_t used = 0;
        for (uint32_t b = 0; b < count; b++)
            for (uint32_t rank = 0; rank < 8; rank++)
                if (job->chosen[b][rank] == (int)expert) {
                    token_at[used] = b;
                    rank_at[used] = rank;
                    memcpy(work_x + used * 2048, job->norm + b * 2048, 2048 * sizeof(float));
                    used++;
                }
        if (!used) continue;
        const ds4_model *m = job->model;
        const struct qwen35moe_layer *l = job->layer;
        qwen35moe_matbatch(m, l->gate, (uint64_t)expert * 512, 512,
                     work_x, 2048, work_gate, 512, used,
                     activation, job->activation_stride);
        qwen35moe_matbatch(m, l->up, (uint64_t)expert * 512, 512,
                     work_x, 2048, work_up, 512, used,
                     activation, job->activation_stride);
        for (uint32_t k = 0; k < used; k++) {
            ds4_qwen35moe_silu(512, work_gate + k * 512, work_gate + k * 512);
            for (uint32_t d = 0; d < 512; d++)
                work_gate[k * 512 + d] *= work_up[k * 512 + d];
        }
        qwen35moe_matbatch(m, l->down, (uint64_t)expert * 2048, 2048,
                     work_gate, 512, work_down, 2048, used,
                     activation, job->activation_stride);
        for (uint32_t k = 0; k < used; k++)
            memcpy(job->route_output + ((size_t)token_at[k] * 8 + rank_at[k]) * 2048,
                   work_down + k * 2048, 2048 * sizeof(float));
    }
    free(activation);
    free(work);
}
static int qwen35moe_forward_batch(ds4_session *s, qwen35moe_state *st, const int *tokens,
                             uint32_t count, uint32_t start, float *logits,
                             char *err, size_t errlen) {
    if (qwen35moe_require_model_valid(s->engine, err, errlen)) return 1;
    if (!count || count > QWEN35MOE_BATCH_MAX || start + count > st->cap) {
        snprintf(err, errlen, "qwen35moe batch or context position out of range");
        return 1;
    }
    if (count == 1) return qwen35moe_forward(s, st, tokens[0], start, logits, err, errlen);
    const ds4_model *m = &s->engine->model;
    const qwen35moe_weights *w = s->engine->qwen35moe_weights;
    for (uint32_t b = 0; b < count; b++)
        if (tokens[b] < 0 || tokens[b] >= 248320) {
            snprintf(err, errlen, "qwen35moe token out of range");
            return 1;
        }
    /* Scratch is bounded independently of context length and is never part
     * of a persisted frontier. The 8 route outputs retain expert rank order. */
    const size_t floats = (size_t)count * (2048 + 2048 + 8192 + 4096 + 1024 +
        32 + 32 + 4096 + 2048 + 256 + 512 + 512 + 2048 + 2048 +
        8 * 2048);
    float *arena = xcalloc(floats, sizeof(float));
    float *next = arena;
#define QWEN35MOE_AREA(name, width) float *name = next; next += (size_t)count * (width)
    QWEN35MOE_AREA(x, 2048); QWEN35MOE_AREA(norm, 2048); QWEN35MOE_AREA(qkv, 8192);
    QWEN35MOE_AREA(z, 4096); QWEN35MOE_AREA(mixed, 1024); QWEN35MOE_AREA(alpha, 32);
    QWEN35MOE_AREA(beta, 32); QWEN35MOE_AREA(heads, 4096); QWEN35MOE_AREA(attn, 2048);
    QWEN35MOE_AREA(router, 256); QWEN35MOE_AREA(gate, 512); QWEN35MOE_AREA(up, 512);
    QWEN35MOE_AREA(down, 2048); QWEN35MOE_AREA(ffn, 2048); QWEN35MOE_AREA(route_output, 8 * 2048);
#undef QWEN35MOE_AREA
    uint8_t *activation = xcalloc(count, sizeof(st->activation));
    const size_t astride = sizeof(st->activation);
#define QWEN35MOE_BATCH_MV(t, xbuf, xs, ybuf, ys, n) \
    qwen35moe_matbatch(m, (t), 0, (t)->ndim > 1 ? (t)->dim[1] : 1, \
                 (xbuf), (xs), (ybuf), (ys), (n), activation, astride)
    int chosen[QWEN35MOE_BATCH_MAX][8];
    float route_weights[QWEN35MOE_BATCH_MAX][8];
    float denominators[QWEN35MOE_BATCH_MAX];
    int rc = 0;
    for (uint32_t b = 0; b < count; b++) qwen35moe_row(m, w->embedding, tokens[b], x + b * 2048);
    for (uint32_t il = 0; il < 40; il++) {
        if (ds4_session_cancelled(s)) {
            snprintf(err, errlen, "interrupted");
            rc = DS4_SESSION_SYNC_INTERRUPTED;
            break;
        }
        const struct qwen35moe_layer *l = &w->layer[il];
        double stage_start = st->timing ? qwen35moe_clock_seconds() : 0;
        for (uint32_t b = 0; b < count; b++)
            qwen35moe_rms(norm + b * 2048, x + b * 2048, qwen35moe_control(m, l->norm), 2048, 1e-6f);
        if (st->timing) st->batch_norm_seconds += qwen35moe_clock_seconds() - stage_start;
        if (qwen35moe_env("STAGE_DUMP")) for (uint32_t b = 0; b < count; b++) {
            st->pos = start + b;
            qwen35moe_dump(st, "attn_norm", il, norm + b * 2048, 2048);
        }
        if (il % 4 == 3) {
            QWEN35MOE_BATCH_MV(l->q, norm, 2048, qkv, 8192, count);
            QWEN35MOE_BATCH_MV(l->k, norm, 2048, mixed, 1024, count);
            qwen35moe_matbatch(m, l->v, 0, 512, norm, 2048, mixed + 512, 1024,
                         count, activation, astride);
            double finish_start = st->timing ? qwen35moe_clock_seconds() : 0;
            if (count >= 64)
                qwen35moe_attention_tiled(m, l, il, start, count, qkv, mixed, z, heads, st);
            else for (uint32_t b = 0; b < count; b++) {
                st->pos = start + b;
                memcpy(st->qkv, qkv + b * 8192, 8192 * sizeof(float));
                memcpy(st->mixed, mixed + b * 1024, 1024 * sizeof(float));
                qwen35moe_attention_finish(m, l, il, start + b, st, false);
                memcpy(heads + b * 4096, st->heads, 4096 * sizeof(float));
            }
            if (st->timing)
                st->full_attention_finish_seconds += qwen35moe_clock_seconds() - finish_start;
        } else {
            QWEN35MOE_BATCH_MV(l->qkv, norm, 2048, qkv, 8192, count);
            QWEN35MOE_BATCH_MV(l->z, norm, 2048, z, 4096, count);
            QWEN35MOE_BATCH_MV(l->alpha, norm, 2048, alpha, 32, count);
            QWEN35MOE_BATCH_MV(l->beta, norm, 2048, beta, 32, count);
            if (qwen35moe_env("STAGE_DUMP")) for (uint32_t b = 0; b < count; b++) {
                st->pos = start + b;
                qwen35moe_dump(st, "alpha", il, alpha + b * 32, 32);
                qwen35moe_dump(st, "beta", il, beta + b * 32, 32);
                float gate[32], beta_sigmoid[32];
                const float *A = qwen35moe_control(m, l->a), *dt = qwen35moe_control(m, l->dt);
                for (uint32_t h = 0; h < 32; h++) {
                    gate[h] = A[h] * qwen35moe_softplus(alpha[b * 32 + h] + dt[h]);
                    beta_sigmoid[h] = qwen35moe_sigmoid(beta[b * 32 + h]);
                }
                qwen35moe_dump(st, "gate", il, gate, 32);
                qwen35moe_dump(st, "beta_sigmoid", il, beta_sigmoid, 32);
            }
            double finish_start = st->timing ? qwen35moe_clock_seconds() : 0;
            if (qwen35moe_env("STAGE_DUMP")) {
                for (uint32_t b = 0; b < count; b++) {
                    st->pos = start + b;
                    memcpy(st->qkv, qkv + b * 8192, 8192 * sizeof(float));
                    memcpy(st->z, z + b * 4096, 4096 * sizeof(float));
                    qwen35moe_linear_prepare(m, l, il, st);
                    memcpy(qkv + b * 8192, st->mixed, 8192 * sizeof(float));
                    memcpy(z + b * 4096, st->z, 4096 * sizeof(float));
                }
            } else {
                /* Every convolution row reads three immutable raw-QKV rows.
                 * Commit the rolling history only after all rows finish. */
                float *history = st->history + (uint64_t)(il - il / 4) * 3 * 8192;
                memcpy(route_output, qkv, (size_t)count * 8192 * sizeof(float));
                struct qwen35moe_linear_prepare_job prepare = {
                    route_output, history, qwen35moe_control(m, l->conv), qkv, z};
                ds4_parallel_for_min_rows(count, qwen35moe_linear_prepare_worker, &prepare, 8);
                if (count >= 3)
                    memcpy(history, route_output + (size_t)(count - 3) * 8192,
                           3 * 8192 * sizeof(float));
                else {
                    memmove(history, history + (size_t)count * 8192,
                            (3 - count) * 8192 * sizeof(float));
                    memcpy(history + (size_t)(3 - count) * 8192, route_output,
                           (size_t)count * 8192 * sizeof(float));
                }
            }
            if (st->timing) st->batch_conv_prepare_seconds += qwen35moe_clock_seconds() - finish_start;
            struct qwen35moe_recurrent_batch_job job = {m, l, st, qkv, z, alpha, beta,
                                                   heads, count, il - il / 4};
            ds4_parallel_for_min_rows(32, qwen35moe_recurrent_batch_worker, &job, 1);
            if (st->timing)
                st->linear_finish_seconds += qwen35moe_clock_seconds() - finish_start;
            if (qwen35moe_env("STAGE_DUMP")) for (uint32_t b = 0; b < count; b++) {
                st->pos = start + b;
                qwen35moe_dump(st, "final_output", il, heads + b * 4096, 4096);
            }
        }
        QWEN35MOE_BATCH_MV(l->out, heads, 4096, attn, 2048, count);
        double post_start = st->timing ? qwen35moe_clock_seconds() : 0;
        for (uint32_t b = 0; b < count; b++) {
            for (uint32_t d = 0; d < 2048; d++) x[b * 2048 + d] += attn[b * 2048 + d];
            st->pos = start + b;
            qwen35moe_dump(st, "attn_residual", il, x + b * 2048, 2048);
            qwen35moe_rms(norm + b * 2048, x + b * 2048,
                    qwen35moe_control(m, l->post_norm), 2048, 1e-6f);
            st->pos = start + b;
            qwen35moe_dump(st, "attn_output", il, attn + b * 2048, 2048);
            qwen35moe_dump(st, "attn_post_norm", il, norm + b * 2048, 2048);
        }
        if (st->timing) st->batch_post_rows_seconds += qwen35moe_clock_seconds() - post_start;
        if (st->timing) {
            st->attention_seconds += qwen35moe_clock_seconds() - stage_start;
            stage_start = qwen35moe_clock_seconds();
        }
        QWEN35MOE_BATCH_MV(l->router, norm, 2048, router, 256, count);
        double router_start = st->timing ? qwen35moe_clock_seconds() : 0;
        if (qwen35moe_env("STAGE_DUMP")) for (uint32_t b = 0; b < count; b++) {
            st->pos = start + b;
            qwen35moe_dump(st, "ffn_moe_logits", il, router + b * 256, 256);
        }
        struct qwen35moe_router_job router_job = {router, chosen, route_weights, denominators};
        ds4_parallel_for_min_rows(count, qwen35moe_router_worker, &router_job, 8);
        if (st->timing) st->batch_router_seconds += qwen35moe_clock_seconds() - router_start;
        uint16_t expert_counts[256] = {0}, active_experts[256];
        uint32_t active_count = 0;
        for (uint32_t b = 0; b < count; b++)
            for (uint32_t rank = 0; rank < 8; rank++) expert_counts[chosen[b][rank]]++;
        for (uint32_t expert = 0; expert < 256; expert++) if (expert_counts[expert]) {
            uint32_t at = active_count++;
            while (at && expert_counts[active_experts[at - 1]] < expert_counts[expert]) {
                active_experts[at] = active_experts[at - 1];
                at--;
            }
            active_experts[at] = (uint16_t)expert;
        }
        struct qwen35moe_expert_batch_job expert_job = {m, l, norm, route_output,
            chosen, count, astride, active_experts, active_count, 0};
        ds4_parallel_for_min_rows(active_count, qwen35moe_expert_batch_worker, &expert_job, 1);
        QWEN35MOE_BATCH_MV(l->shared_router, norm, 2048, router, 256, count);
        QWEN35MOE_BATCH_MV(l->shared_gate, norm, 2048, gate, 512, count);
        QWEN35MOE_BATCH_MV(l->shared_up, norm, 2048, up, 512, count);
        for (uint32_t b = 0; b < count; b++) {
            ds4_qwen35moe_silu(512, gate + b * 512, gate + b * 512);
            for (uint32_t d = 0; d < 512; d++) gate[b * 512 + d] *= up[b * 512 + d];
        }
        QWEN35MOE_BATCH_MV(l->shared_down, gate, 512, down, 2048, count);
        for (uint32_t b = 0; b < count; b++) {
            float *out = ffn + b * 2048;
            memset(out, 0, 2048 * sizeof(float));
            for (uint32_t rank = 0; rank < 8; rank++) {
                const float *expert = route_output + ((size_t)b * 8 + rank) * 2048;
                float scale = route_weights[b][rank] / denominators[b];
                for (uint32_t d = 0; d < 2048; d++) out[d] += expert[d] * scale;
            }
            float scale = qwen35moe_sigmoid(router[b * 256]);
            st->pos = start + b;
            for (uint32_t d = 0; d < 2048; d++) {
                out[d] += down[b * 2048 + d] * scale;
                x[b * 2048 + d] += out[d];
            }
            qwen35moe_dump(st, "ffn_out", il, out, 2048);
        }
        if (st->timing) st->feedforward_seconds += qwen35moe_clock_seconds() - stage_start;
        const char *dump = qwen35moe_env("LAYER_DUMP");
        if (dump && dump[0]) for (uint32_t b = 0; b < count; b++) {
            const char *only_pos = qwen35moe_env("LAYER_DUMP_POS");
            const char *min_pos = qwen35moe_env("LAYER_DUMP_MIN_POS");
            const char *min_layer = qwen35moe_env("LAYER_DUMP_MIN_LAYER");
            const char *max_layer = qwen35moe_env("LAYER_DUMP_MAX_LAYER");
            if (only_pos && strtoul(only_pos, NULL, 10) != start + b) continue;
            if (min_pos && start + b < strtoul(min_pos, NULL, 10)) continue;
            if (min_layer && il < strtoul(min_layer, NULL, 10)) continue;
            if (max_layer && il > strtoul(max_layer, NULL, 10)) continue;
            char path[1024];
            snprintf(path, sizeof(path), "%s.%u.%u.bin", dump, start + b, il);
            FILE *fp = fopen(path, "wb");
            if (fp) { fwrite(x + b * 2048, sizeof(float), 2048, fp); fclose(fp); }
        }
    }
    if (!rc && logits) {
        double stage_start = st->timing ? qwen35moe_clock_seconds() : 0;
        qwen35moe_rms(st->norm, x + (count - 1) * 2048, qwen35moe_control(m, w->norm), 2048, 1e-6f);
        QWEN35MOE_MV(w->output, st->norm, logits);
        if (st->timing) st->output_seconds += qwen35moe_clock_seconds() - stage_start;
    }
    if (!rc) {
        st->pos = start + count - 1;
        if (st->timing) st->timed_tokens += count;
    }
#undef QWEN35MOE_BATCH_MV
    free(activation);
    free(arena);
    return rc;
}
#undef QWEN35MOE_BATCH_MAX
#undef QWEN35MOE_MV

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC pop_options
#elif defined(__clang__)
#pragma float_control(pop)
#endif
