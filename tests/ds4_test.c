#define DS4_SERVER_TEST
#define DS4_SERVER_TEST_NO_MAIN
#include "../ds4_server.c"
#ifndef DS4_NO_GPU
#include "../ds4_gpu.h"
#include <math.h>

static ds4_engine *test_engine_fast;
static ds4_engine *test_engine_quality;

static const char *test_model_path(void) {
    const char *model_path = getenv("DS4_TEST_MODEL");
    return (model_path && model_path[0]) ? model_path : "ds4flash.gguf";
}

static ds4_engine *test_get_engine(bool quality) {
    ds4_engine **slot = quality ? &test_engine_quality : &test_engine_fast;
    if (*slot) return *slot;

    ds4_engine_options opt = {
        .model_path = test_model_path(),
#ifdef __APPLE__
        .backend = DS4_BACKEND_METAL,
#else
        .backend = DS4_BACKEND_CUDA,
#endif
        .quality = quality,
    };
    TEST_ASSERT(ds4_engine_open(slot, &opt) == 0);
    return *slot;
}

static void test_close_engines(void) {
    ds4_engine_close(test_engine_fast);
    ds4_engine_close(test_engine_quality);
    test_engine_fast = NULL;
    test_engine_quality = NULL;
}

static void test_close_engine(bool quality) {
    ds4_engine **slot = quality ? &test_engine_quality : &test_engine_fast;
    ds4_engine_close(*slot);
    *slot = NULL;
}

static uint64_t test_round_up_u64(uint64_t n, uint64_t align) {
    return (n + align - 1) & ~(align - 1);
}

static uint16_t test_float_to_f16(float f) {
    union {
        float f;
        uint32_t u;
    } v = { .f = f };

    uint32_t sign = (v.u >> 16) & 0x8000u;
    int32_t exp = (int32_t)((v.u >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = v.u & 0x7fffffu;

    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x800000u;
        uint32_t shift = (uint32_t)(14 - exp);
        uint32_t half_mant = mant >> shift;
        if ((mant >> (shift - 1)) & 1u) half_mant++;
        return (uint16_t)(sign | half_mant);
    }
    if (exp >= 31) return (uint16_t)(sign | 0x7c00u);

    uint32_t half = sign | ((uint32_t)exp << 10) | (mant >> 13);
    if (mant & 0x1000u) half++;
    return (uint16_t)half;
}

static void test_metal_f16_matvec_fast_nr0_4(void) {
    /*
     * This is the short regression for the long-context repetition failure.
     * Decode uses one-token F16 matvecs for several DS4 projections; the fast
     * nr0=4 variant must be numerically equivalent to the plain kernel.
     */
    const uint32_t in_dim = 4096;
    const uint32_t out_dim = 512;
    const uint64_t weight_bytes = (uint64_t)in_dim * out_dim * sizeof(uint16_t);
    const uint64_t weight_alloc = test_round_up_u64(weight_bytes, (uint64_t)getpagesize());

    void *weights_raw = NULL;
    TEST_ASSERT(posix_memalign(&weights_raw, (size_t)getpagesize(), (size_t)weight_alloc) == 0);
    if (!weights_raw) return;

    uint16_t *weights = weights_raw;
    memset(weights, 0, (size_t)weight_alloc);
    for (uint32_t o = 0; o < out_dim; o++) {
        for (uint32_t i = 0; i < in_dim; i++) {
            float w = (float)((int)((o * 3u + i * 5u) % 23u) - 11) / 64.0f;
            weights[(uint64_t)o * in_dim + i] = test_float_to_f16(w);
        }
    }

    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc((uint64_t)in_dim * sizeof(float));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc((uint64_t)out_dim * sizeof(float));
    TEST_ASSERT(x != NULL);
    TEST_ASSERT(out != NULL);
    if (!x || !out) {
        ds4_gpu_tensor_free(x);
        ds4_gpu_tensor_free(out);
        free(weights_raw);
        return;
    }

    float *x_host = malloc((size_t)in_dim * sizeof(float));
    float *out_host = malloc((size_t)out_dim * sizeof(float));
    TEST_ASSERT(x_host != NULL);
    TEST_ASSERT(out_host != NULL);
    if (!x_host || !out_host) {
        free(x_host);
        free(out_host);
        ds4_gpu_tensor_free(x);
        ds4_gpu_tensor_free(out);
        free(weights_raw);
        return;
    }

    for (uint32_t i = 0; i < in_dim; i++) {
        x_host[i] = (float)((int)(i % 31u) - 15) / 32.0f;
    }

    TEST_ASSERT(ds4_gpu_tensor_write(x, 0, x_host, (uint64_t)in_dim * sizeof(float)) != 0);
    TEST_ASSERT(ds4_gpu_set_model_map(weights_raw, weight_alloc) != 0);
    ds4_gpu_set_quality(false);
    TEST_ASSERT(ds4_gpu_matmul_f16_tensor(out, weights_raw, weight_alloc, 0,
                                            in_dim, out_dim, x, 1) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(out, 0, out_host, (uint64_t)out_dim * sizeof(float)) != 0);

    float max_abs = 0.0f;
    for (uint32_t o = 0; o < out_dim; o++) {
        float ref = 0.0f;
        for (uint32_t i = 0; i < in_dim; i++) {
            float w = (float)((int)((o * 3u + i * 5u) % 23u) - 11) / 64.0f;
            ref += w * x_host[i];
        }
        float err = fabsf(out_host[o] - ref);
        if (err > max_abs) max_abs = err;
    }
    TEST_ASSERT(max_abs < 0.02f);

    free(x_host);
    free(out_host);
    ds4_gpu_tensor_free(x);
    ds4_gpu_tensor_free(out);
    free(weights_raw);
}

static void test_metal_turbo_quant_roundtrip_one(bool turbo4) {
    const uint32_t n_tok = 3;
    const uint32_t n_head = 2;
    const uint32_t n_rows = n_tok * n_head;
    const uint32_t head_dim = 96;
    const uint32_t n_rot = 32;
    const uint32_t n_nope = head_dim - n_rot;
    const uint32_t n_blocks = n_nope / 32;
    const uint64_t block_bytes = turbo4 ? DS4_TURBO4_BLOCK_BYTES : DS4_TURBO3_BLOCK_BYTES;
    const uint64_t row_bytes = (uint64_t)n_blocks * block_bytes + (uint64_t)n_rot * sizeof(float);
    const uint64_t src_bytes = (uint64_t)n_rows * head_dim * sizeof(float);
    const uint64_t q_bytes = (uint64_t)n_rows * row_bytes;

    ds4_gpu_tensor *src = ds4_gpu_tensor_alloc(src_bytes);
    ds4_gpu_tensor *q = ds4_gpu_tensor_alloc(q_bytes);
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(src_bytes);
    TEST_ASSERT(src != NULL);
    TEST_ASSERT(q != NULL);
    TEST_ASSERT(out != NULL);
    if (!src || !q || !out) {
        ds4_gpu_tensor_free(out);
        ds4_gpu_tensor_free(q);
        ds4_gpu_tensor_free(src);
        return;
    }

    float *src_host = malloc((size_t)src_bytes);
    float *out_host = malloc((size_t)src_bytes);
    TEST_ASSERT(src_host != NULL);
    TEST_ASSERT(out_host != NULL);
    if (!src_host || !out_host) {
        free(out_host);
        free(src_host);
        ds4_gpu_tensor_free(out);
        ds4_gpu_tensor_free(q);
        ds4_gpu_tensor_free(src);
        return;
    }

    for (uint32_t r = 0; r < n_rows; r++) {
        for (uint32_t i = 0; i < head_dim; i++) {
            float v = (float)((int)((r * 17u + i * 13u) % 41u) - 20) / 21.0f;
            if (i >= n_nope) v = 1000.0f + (float)(r * 100u + i);
            src_host[(uint64_t)r * head_dim + i] = v;
        }
    }

    TEST_ASSERT(ds4_gpu_tensor_write(src, 0, src_host, src_bytes) != 0);
    if (turbo4) {
        TEST_ASSERT(ds4_gpu_turbo4_kv_quantize_tensor(q, src, n_tok, n_head, head_dim, n_rot) != 0);
        TEST_ASSERT(ds4_gpu_turbo4_dequant_f32_tensor(out, q, n_blocks, n_rot, n_rows) != 0);
    } else {
        TEST_ASSERT(ds4_gpu_turbo3_kv_quantize_tensor(q, src, n_tok, n_head, head_dim, n_rot) != 0);
        TEST_ASSERT(ds4_gpu_turbo3_dequant_f32_tensor(out, q, n_blocks, n_rot, n_rows) != 0);
    }
    TEST_ASSERT(ds4_gpu_tensor_read(out, 0, out_host, src_bytes) != 0);

    double prefix_ss = 0.0;
    double err_ss = 0.0;
    float max_tail = 0.0f;
    for (uint32_t r = 0; r < n_rows; r++) {
        for (uint32_t i = 0; i < head_dim; i++) {
            const uint64_t off = (uint64_t)r * head_dim + i;
            TEST_ASSERT(isfinite(out_host[off]));
            if (i < n_nope) {
                const double d = (double)out_host[off] - (double)src_host[off];
                err_ss += d * d;
                prefix_ss += (double)src_host[off] * (double)src_host[off];
            } else {
                const float err = fabsf(out_host[off] - src_host[off]);
                if (err > max_tail) max_tail = err;
            }
        }
    }
    const double rel_rms = sqrt(err_ss / prefix_ss);
    TEST_ASSERT(max_tail == 0.0f);
    TEST_ASSERT(rel_rms < (turbo4 ? 0.55 : 0.75));

    free(out_host);
    free(src_host);
    ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(q);
    ds4_gpu_tensor_free(src);
}

static void test_metal_turbo_quant_roundtrip(void) {
    test_metal_turbo_quant_roundtrip_one(false);
    test_metal_turbo_quant_roundtrip_one(true);
}

static void test_metal_turbo_indexed_attention_one(bool turbo4) {
    const uint32_t n_tokens = 1;
    const uint32_t n_head = 8;
    const uint32_t head_dim = 512;
    const uint32_t n_rot = 64;
    const uint32_t n_blocks = (head_dim - n_rot) / 32;
    const uint32_t n_raw = 4;
    const uint32_t raw_cap = 4;
    const uint32_t n_comp = 8;
    const uint32_t top_k = 4;
    const uint64_t row_bytes = (uint64_t)head_dim * sizeof(float);
    const uint64_t q_bytes = (uint64_t)n_tokens * n_head * row_bytes;
    const uint64_t raw_bytes = (uint64_t)raw_cap * row_bytes;
    const uint64_t comp_bytes = (uint64_t)n_comp * row_bytes;
    const uint64_t turbo_block_bytes = turbo4 ? DS4_TURBO4_BLOCK_BYTES : DS4_TURBO3_BLOCK_BYTES;
    const uint64_t turbo_row_bytes = (uint64_t)n_blocks * turbo_block_bytes + (uint64_t)n_rot * sizeof(float);
    const uint64_t turbo_bytes = (uint64_t)n_comp * turbo_row_bytes;
    const uint64_t heads_bytes = q_bytes;
    const uint64_t sinks_alloc = test_round_up_u64((uint64_t)n_head * sizeof(float),
                                                   (uint64_t)getpagesize());

    ds4_gpu_tensor *q = ds4_gpu_tensor_alloc(q_bytes);
    ds4_gpu_tensor *raw = ds4_gpu_tensor_alloc(raw_bytes);
    ds4_gpu_tensor *comp_src = ds4_gpu_tensor_alloc(comp_bytes);
    ds4_gpu_tensor *comp_deq = ds4_gpu_tensor_alloc(comp_bytes);
    ds4_gpu_tensor *comp_turbo = ds4_gpu_tensor_alloc(turbo_bytes);
    ds4_gpu_tensor *comp_selected_deq = ds4_gpu_tensor_alloc((uint64_t)top_k * row_bytes);
    ds4_gpu_tensor *topk = ds4_gpu_tensor_alloc((uint64_t)top_k * sizeof(int32_t));
    ds4_gpu_tensor *identity_topk = ds4_gpu_tensor_alloc((uint64_t)top_k * sizeof(int32_t));
    ds4_gpu_tensor *heads_ref = ds4_gpu_tensor_alloc(heads_bytes);
    ds4_gpu_tensor *heads_turbo = ds4_gpu_tensor_alloc(heads_bytes);
    ds4_gpu_tensor *heads_selected = ds4_gpu_tensor_alloc(heads_bytes);
    TEST_ASSERT(q && raw && comp_src && comp_deq && comp_turbo && comp_selected_deq &&
                topk && identity_topk && heads_ref && heads_turbo && heads_selected);

    float *q_host = malloc((size_t)q_bytes);
    float *raw_host = malloc((size_t)raw_bytes);
    float *comp_host = malloc((size_t)comp_bytes);
    float *ref_host = malloc((size_t)heads_bytes);
    float *turbo_host = malloc((size_t)heads_bytes);
    void *sinks_raw = NULL;
    TEST_ASSERT(q_host && raw_host && comp_host && ref_host && turbo_host);
    TEST_ASSERT(posix_memalign(&sinks_raw, (size_t)getpagesize(), (size_t)sinks_alloc) == 0);
    TEST_ASSERT(sinks_raw != NULL);

    if (q && raw && comp_src && comp_deq && comp_turbo && comp_selected_deq &&
        topk && identity_topk && heads_ref && heads_turbo && heads_selected &&
        q_host && raw_host && comp_host && ref_host && turbo_host && sinks_raw) {
        for (uint32_t h = 0; h < n_head; h++) {
            for (uint32_t i = 0; i < head_dim; i++) {
                q_host[(uint64_t)h * head_dim + i] =
                    0.25f * sinf((float)((h + 1u) * (i + 3u)) * 0.013f);
            }
        }
        for (uint32_t r = 0; r < raw_cap; r++) {
            for (uint32_t i = 0; i < head_dim; i++) {
                raw_host[(uint64_t)r * head_dim + i] =
                    0.20f * cosf((float)((r + 2u) * (i + 5u)) * 0.011f);
            }
        }
        for (uint32_t r = 0; r < n_comp; r++) {
            for (uint32_t i = 0; i < head_dim; i++) {
                comp_host[(uint64_t)r * head_dim + i] =
                    0.18f * sinf((float)((r + 4u) * (i + 7u)) * 0.009f);
            }
        }
        float *sinks = (float *)sinks_raw;
        memset(sinks, 0, (size_t)sinks_alloc);
        for (uint32_t h = 0; h < n_head; h++) {
            sinks[h] = -0.35f + 0.01f * (float)h;
        }
        const int32_t topk_host[4] = { 7, 3, 1, 5 };

        TEST_ASSERT(ds4_gpu_tensor_write(q, 0, q_host, q_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(raw, 0, raw_host, raw_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(comp_src, 0, comp_host, comp_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(topk, 0, topk_host, sizeof(topk_host)) != 0);
        TEST_ASSERT(ds4_gpu_set_model_map(sinks_raw, sinks_alloc) != 0);
        ds4_gpu_set_quality(false);

        if (turbo4) {
            TEST_ASSERT(ds4_gpu_turbo4_kv_quantize_tensor(comp_turbo, comp_src,
                                                           n_comp, 1, head_dim, n_rot) != 0);
            TEST_ASSERT(ds4_gpu_turbo4_dequant_f32_tensor(comp_deq, comp_turbo,
                                                          n_blocks, n_rot, n_comp) != 0);
            TEST_ASSERT(ds4_gpu_turbo4_dequant_selected_f32_tensor(comp_selected_deq,
                                                                    identity_topk,
                                                                    comp_turbo,
                                                                    topk,
                                                                    n_blocks,
                                                                    n_rot,
                                                                    n_comp,
                                                                    top_k,
                                                                    n_tokens) != 0);
        } else {
            TEST_ASSERT(ds4_gpu_turbo3_kv_quantize_tensor(comp_turbo, comp_src,
                                                           n_comp, 1, head_dim, n_rot) != 0);
            TEST_ASSERT(ds4_gpu_turbo3_dequant_f32_tensor(comp_deq, comp_turbo,
                                                          n_blocks, n_rot, n_comp) != 0);
            TEST_ASSERT(ds4_gpu_turbo3_dequant_selected_f32_tensor(comp_selected_deq,
                                                                    identity_topk,
                                                                    comp_turbo,
                                                                    topk,
                                                                    n_blocks,
                                                                    n_rot,
                                                                    n_comp,
                                                                    top_k,
                                                                    n_tokens) != 0);
        }

        TEST_ASSERT(ds4_gpu_attention_indexed_mixed_batch_heads_tensor(
                        heads_ref, sinks_raw, sinks_alloc, 0,
                        q, raw, comp_deq, topk,
                        n_tokens, 31, n_raw, raw_cap, 0, n_comp, top_k, 128, 4,
                        n_head, head_dim) != 0);
        TEST_ASSERT(ds4_gpu_attention_indexed_mixed_turbo_batch_heads_tensor(
                        heads_turbo, sinks_raw, sinks_alloc, 0,
                        q, raw, comp_turbo, topk,
                        turbo4 ? DS4_KV_QUANT_TURBO4 : DS4_KV_QUANT_TURBO3,
                        n_blocks, n_rot,
                        n_tokens, 31, n_raw, raw_cap, 0, n_comp, top_k, 128, 4,
                        n_head, head_dim) != 0);
        TEST_ASSERT(ds4_gpu_attention_indexed_mixed_batch_heads_tensor(
                        heads_selected, sinks_raw, sinks_alloc, 0,
                        q, raw, comp_selected_deq, identity_topk,
                        n_tokens, 31, n_raw, raw_cap, 0, top_k, top_k, 128, 4,
                        n_head, head_dim) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(heads_ref, 0, ref_host, heads_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(heads_turbo, 0, turbo_host, heads_bytes) != 0);

        float max_abs = 0.0f;
        for (uint64_t i = 0; i < heads_bytes / sizeof(float); i++) {
            TEST_ASSERT(isfinite(ref_host[i]));
            TEST_ASSERT(isfinite(turbo_host[i]));
            const float err = fabsf(ref_host[i] - turbo_host[i]);
            if (err > max_abs) max_abs = err;
        }
        TEST_ASSERT(max_abs < 2.0e-4f);

        TEST_ASSERT(ds4_gpu_tensor_read(heads_selected, 0, turbo_host, heads_bytes) != 0);
        max_abs = 0.0f;
        for (uint64_t i = 0; i < heads_bytes / sizeof(float); i++) {
            TEST_ASSERT(isfinite(turbo_host[i]));
            const float err = fabsf(ref_host[i] - turbo_host[i]);
            if (err > max_abs) max_abs = err;
        }
        TEST_ASSERT(max_abs < 2.0e-4f);
    }

    free(sinks_raw);
    free(turbo_host);
    free(ref_host);
    free(comp_host);
    free(raw_host);
    free(q_host);
    ds4_gpu_tensor_free(heads_turbo);
    ds4_gpu_tensor_free(heads_selected);
    ds4_gpu_tensor_free(heads_ref);
    ds4_gpu_tensor_free(identity_topk);
    ds4_gpu_tensor_free(topk);
    ds4_gpu_tensor_free(comp_selected_deq);
    ds4_gpu_tensor_free(comp_turbo);
    ds4_gpu_tensor_free(comp_deq);
    ds4_gpu_tensor_free(comp_src);
    ds4_gpu_tensor_free(raw);
    ds4_gpu_tensor_free(q);
}

static void test_metal_turbo_indexed_attention(void) {
    test_metal_turbo_indexed_attention_one(false);
    test_metal_turbo_indexed_attention_one(true);
}

static void test_metal_kernels(void) {
    test_metal_f16_matvec_fast_nr0_4();
    test_metal_turbo_quant_roundtrip();
    test_metal_turbo_indexed_attention();
}

static char *test_read_file(const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) return NULL;
    if (fseek(fp, 0, SEEK_END) != 0) {
        fclose(fp);
        return NULL;
    }
    long len = ftell(fp);
    if (len < 0) {
        fclose(fp);
        return NULL;
    }
    rewind(fp);
    char *s = malloc((size_t)len + 1);
    if (!s) {
        fclose(fp);
        return NULL;
    }
    size_t nread = fread(s, 1, (size_t)len, fp);
    fclose(fp);
    if (nread != (size_t)len) {
        free(s);
        return NULL;
    }
    s[len] = '\0';
    return s;
}

typedef struct {
    const char *name;
    int number;
} test_long_fact;

static const test_long_fact test_long_facts[] = {
    {"Bob", 34},
    {"Alice", 52},
    {"Clara", 71},
    {"Diego", 93},
    {"Elena", 16},
    {"Felix", 88},
    {"Greta", 47},
    {"Hugo", 29},
    {"Iris", 64},
    {"Jonas", 12},
    {"Kira", 81},
    {"Leo", 39},
    {"Marta", 76},
    {"Nadia", 23},
    {"Owen", 58},
    {"Priya", 97},
};

static bool test_is_name_boundary(char c) {
    unsigned char uc = (unsigned char)c;
    return c == '\0' || !(isalnum(uc) || c == '_');
}

static bool test_parse_assignment_value(const char *p, int *value) {
    while (*p == ' ' || *p == '\t') p++;
    if (*p != '=') return false;
    p++;
    while (*p == ' ' || *p == '\t') p++;
    if (!isdigit((unsigned char)*p)) return false;

    int v = 0;
    while (isdigit((unsigned char)*p)) {
        v = v * 10 + (*p - '0');
        p++;
    }
    *value = v;
    return true;
}

static bool test_output_has_fact(const char *text, const test_long_fact *fact) {
    const size_t name_len = strlen(fact->name);
    const char *p = text;
    bool saw_wrong_assignment = false;
    int wrong_value = -1;

    while ((p = strstr(p, fact->name)) != NULL) {
        const bool before_ok = p == text || test_is_name_boundary(p[-1]);
        const bool after_ok = test_is_name_boundary(p[name_len]) ||
                              p[name_len] == ' ' ||
                              p[name_len] == '\t' ||
                              p[name_len] == '=';
        if (before_ok && after_ok) {
            int value = 0;
            if (test_parse_assignment_value(p + name_len, &value)) {
                if (value == fact->number) return true;
                saw_wrong_assignment = true;
                wrong_value = value;
            }
        }
        p += name_len;
    }

    if (saw_wrong_assignment) {
        fprintf(stderr,
                "ds4-test: long-context wrong assignment for %s: got %d expected %d\n",
                fact->name, wrong_value, fact->number);
    } else {
        fprintf(stderr,
                "ds4-test: long-context missing assignment for %s=%d\n",
                fact->name, fact->number);
    }
    return false;
}

static int test_hex_digit(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return 10 + c - 'a';
    if (c >= 'A' && c <= 'F') return 10 + c - 'A';
    return -1;
}

static bool test_hex_to_bytes(const char *hex, unsigned char *out, int cap, int *len) {
    int n = 0;
    while (*hex && !isspace((unsigned char)*hex)) {
        int hi = test_hex_digit(hex[0]);
        int lo = test_hex_digit(hex[1]);
        if (hi < 0 || lo < 0 || n >= cap) return false;
        out[n++] = (unsigned char)((hi << 4) | lo);
        hex += 2;
    }
    *len = n;
    return true;
}

static bool test_token_bytes_equal(ds4_engine *engine, int token,
                                   const unsigned char *want, int want_len) {
    size_t got_len = 0;
    char *got = ds4_token_text(engine, token, &got_len);
    bool eq = got && got_len == (size_t)want_len &&
              memcmp(got, want, (size_t)want_len) == 0;
    free(got);
    return eq;
}

static void test_long_prefill_progress(void *ud, const char *event, int current, int total) {
    (void)ud;
    if (strcmp(event, "prefill_chunk")) return;
    if (current == 0 || current == total || current % 8192 == 0) {
        fprintf(stderr, "ds4-test: long-context prefill %d/%d\n", current, total);
    }
}

static void test_long_story_fact_recall(void) {
    const char *prompt_path = getenv("DS4_TEST_LONG_PROMPT");
    if (!prompt_path || !prompt_path[0]) {
        prompt_path = "tests/long_context_story_prompt.txt";
    }
    char *prompt_text = test_read_file(prompt_path);
    TEST_ASSERT(prompt_text != NULL);
    if (!prompt_text) return;

    ds4_engine *engine = test_get_engine(false);
    if (!engine) {
        free(prompt_text);
        return;
    }

    ds4_tokens prompt = {0};
    ds4_tokenize_rendered_chat(engine, prompt_text, &prompt);
    TEST_ASSERT(prompt.len > 30000);

    ds4_session *session = NULL;
    TEST_ASSERT(ds4_session_create(&session, engine, 100000) == 0);
    if (!session) {
        ds4_tokens_free(&prompt);
        free(prompt_text);
        return;
    }

    char err[160];
    ds4_session_set_progress(session, test_long_prefill_progress, NULL);
    TEST_ASSERT(ds4_session_sync(session, &prompt, err, sizeof(err)) == 0);
    ds4_session_set_progress(session, NULL, NULL);

    buf out = {0};
    uint64_t rng = 12345;
    int generated = 0;
    bool decode_ok = true;
    for (; generated < 350; generated++) {
        int token = ds4_session_sample(session, 0.0f, 0, 1.0f, 0.0f, &rng);
        if (token == ds4_token_eos(engine)) break;

        size_t piece_len = 0;
        char *piece = ds4_token_text(engine, token, &piece_len);
        buf_append(&out, piece, piece_len);
        free(piece);

        if (ds4_session_eval(session, token, err, sizeof(err)) != 0) {
            decode_ok = false;
            break;
        }
    }

    const char *text = out.ptr ? out.ptr : "";
    TEST_ASSERT(decode_ok);
    TEST_ASSERT(generated > 0);
    for (size_t i = 0; i < sizeof(test_long_facts) / sizeof(test_long_facts[0]); i++) {
        TEST_ASSERT(test_output_has_fact(text, &test_long_facts[i]));
    }

    buf_free(&out);
    ds4_session_free(session);
    ds4_tokens_free(&prompt);
    free(prompt_text);
}

#define TEST_VEC_MAX_STEPS 16
#define TEST_VEC_MAX_TOP 32
#define TEST_VEC_MAX_TOKEN_BYTES 128

typedef struct {
    unsigned char bytes[TEST_VEC_MAX_TOKEN_BYTES];
    int len;
    float logprob;
} test_vec_top;

typedef struct {
    unsigned char selected[TEST_VEC_MAX_TOKEN_BYTES];
    int selected_len;
    int ntop;
    test_vec_top top[TEST_VEC_MAX_TOP];
} test_vec_step;

typedef struct {
    char id[96];
    char prompt_path[512];
    int ctx;
    int nsteps;
    test_vec_step steps[TEST_VEC_MAX_STEPS];
} test_vec_case;

static char *test_trim_line(char *line) {
    while (*line && isspace((unsigned char)*line)) line++;
    size_t n = strlen(line);
    while (n && isspace((unsigned char)line[n - 1])) line[--n] = '\0';
    return line;
}

static bool test_read_vector_case(FILE *fp, test_vec_case *vc) {
    char line[2048];
    memset(vc, 0, sizeof(*vc));
    while (fgets(line, sizeof(line), fp)) {
        char *p = test_trim_line(line);
        if (!p[0] || p[0] == '#') continue;
        if (sscanf(p, "case %95s %d %d %511s",
                   vc->id, &vc->ctx, &vc->nsteps, vc->prompt_path) == 4) {
            TEST_ASSERT(vc->nsteps > 0 && vc->nsteps <= TEST_VEC_MAX_STEPS);
            return true;
        }
        TEST_ASSERT(!"unexpected line before vector case");
    }
    return false;
}

static bool test_fill_vector_case(FILE *fp, test_vec_case *vc) {
    char line[2048];
    int step_index = -1;
    int top_index = 0;

    while (fgets(line, sizeof(line), fp)) {
        char *p = test_trim_line(line);
        if (!p[0] || p[0] == '#') continue;
        if (!strcmp(p, "end")) return true;

        if (!strncmp(p, "step ", 5)) {
            char hex[TEST_VEC_MAX_TOKEN_BYTES * 2 + 2];
            int ntop = 0;
            if (sscanf(p, "step %d %257s %d", &step_index, hex, &ntop) != 3) {
                TEST_ASSERT(!"bad vector step line");
                return false;
            }
            TEST_ASSERT(step_index >= 0 && step_index < vc->nsteps);
            TEST_ASSERT(ntop >= 0 && ntop <= TEST_VEC_MAX_TOP);
            vc->steps[step_index].ntop = ntop;
            TEST_ASSERT(test_hex_to_bytes(hex,
                                          vc->steps[step_index].selected,
                                          TEST_VEC_MAX_TOKEN_BYTES,
                                          &vc->steps[step_index].selected_len));
            top_index = 0;
            continue;
        }

        if (!strncmp(p, "top ", 4)) {
            char hex[TEST_VEC_MAX_TOKEN_BYTES * 2 + 2];
            float lp = 0.0f;
            TEST_ASSERT(step_index >= 0 && step_index < vc->nsteps);
            TEST_ASSERT(top_index < vc->steps[step_index].ntop);
            if (sscanf(p, "top %257s %f", hex, &lp) != 2) {
                TEST_ASSERT(!"bad vector top line");
                return false;
            }
            test_vec_top *top = &vc->steps[step_index].top[top_index++];
            top->logprob = lp;
            TEST_ASSERT(test_hex_to_bytes(hex, top->bytes,
                                          TEST_VEC_MAX_TOKEN_BYTES, &top->len));
            continue;
        }

        TEST_ASSERT(!"unexpected vector line");
        return false;
    }

    TEST_ASSERT(!"unterminated vector case");
    return false;
}

static void test_logprob_vector_case(ds4_engine *engine, const test_vec_case *vc) {
    char *prompt_text = test_read_file(vc->prompt_path);
    TEST_ASSERT(prompt_text != NULL);
    if (!prompt_text) return;

    ds4_tokens prompt = {0};
    ds4_encode_chat_prompt(engine, "", prompt_text, DS4_THINK_NONE, &prompt);
    free(prompt_text);

    ds4_session *session = NULL;
    TEST_ASSERT(ds4_session_create(&session, engine, vc->ctx) == 0);
    if (!session) {
        ds4_tokens_free(&prompt);
        return;
    }

    char err[160];
    TEST_ASSERT(ds4_session_sync(session, &prompt, err, sizeof(err)) == 0);

    ds4_token_score scores[20];
    for (int i = 0; i < vc->nsteps; i++) {
        const test_vec_step *step = &vc->steps[i];
        int nscore = ds4_session_top_logprobs(session, scores, 20);
        int token = ds4_session_argmax(session);
        if (!test_token_bytes_equal(engine, token, step->selected, step->selected_len)) {
            fprintf(stderr, "ds4-test: vector %s step %d selected token mismatch\n",
                    vc->id, i);
            TEST_ASSERT(false);
        }

        for (int t = 0; t < step->ntop; t++) {
            bool found = false;
            float local_lp = 0.0f;
            for (int j = 0; j < nscore; j++) {
                if (scores[j].id < 0) continue;
                if (test_token_bytes_equal(engine, scores[j].id,
                                           step->top[t].bytes,
                                           step->top[t].len)) {
                    found = true;
                    local_lp = scores[j].logprob;
                    break;
                }
            }
            if (!found) {
                fprintf(stderr, "ds4-test: vector %s step %d official top token missing locally\n",
                        vc->id, i);
                TEST_ASSERT(false);
            } else if (fabsf(local_lp - step->top[t].logprob) > 4.0f) {
                fprintf(stderr,
                        "ds4-test: vector %s step %d logprob delta too high: local=%g official=%g\n",
                        vc->id, i, local_lp, step->top[t].logprob);
                TEST_ASSERT(false);
            }
        }

        if (i + 1 < vc->nsteps) {
            TEST_ASSERT(ds4_session_eval(session, token, err, sizeof(err)) == 0);
        }
    }

    ds4_session_free(session);
    ds4_tokens_free(&prompt);
}

static void test_official_logprob_vectors(void) {
    const char *path = getenv("DS4_TEST_VECTOR_FILE");
    if (!path || !path[0]) path = "tests/test-vectors/official.vec";
    FILE *fp = fopen(path, "rb");
    TEST_ASSERT(fp != NULL);
    if (!fp) return;

    ds4_engine *engine = test_get_engine(false);
    if (!engine) {
        fclose(fp);
        return;
    }

    test_vec_case vc;
    while (test_read_vector_case(fp, &vc)) {
        if (!test_fill_vector_case(fp, &vc)) break;
        fprintf(stderr, "ds4-test: vector %s\n", vc.id);
        test_logprob_vector_case(engine, &vc);
    }
    fclose(fp);
}

static const char *test_tool_call_request_json(void) {
    return
        "{"
        "\"model\":\"deepseek-v4-flash\","
        "\"messages\":[{\"role\":\"user\",\"content\":\"List the files in the current directory. Use the provided tool; do not answer in prose.\"}],"
        "\"tools\":[{\"type\":\"function\",\"function\":{"
            "\"name\":\"list_files\","
            "\"description\":\"List files in a directory.\","
            "\"parameters\":{\"type\":\"object\",\"properties\":{"
                "\"path\":{\"type\":\"string\",\"description\":\"Directory path to list.\"}"
            "},\"required\":[\"path\"]}"
        "}}],"
        "\"tool_choice\":\"auto\","
        "\"think\":false,"
        "\"temperature\":0,"
        "\"max_tokens\":256,"
        "\"stream\":false"
        "}";
}

static void test_tool_call_quality_one(bool quality) {
    ds4_engine *engine = test_get_engine(quality);
    if (!engine) return;

    request r;
    char err[160];
    TEST_ASSERT(parse_chat_request(engine, NULL, test_tool_call_request_json(),
                                   512, 32768, &r, err, sizeof(err)));

    ds4_session *session = NULL;
    TEST_ASSERT(ds4_session_create(&session, engine, 32768) == 0);
    if (!session) {
        request_free(&r);
        return;
    }
    TEST_ASSERT(ds4_session_sync(session, &r.prompt, err, sizeof(err)) == 0);

    buf text = {0};
    uint64_t rng = 123;
    bool decode_ok = true;
    bool saw_tool_start = false;
    bool saw_tool_end = false;
    for (int i = 0; i < r.max_tokens; i++) {
        int token = ds4_session_sample(session, r.temperature, r.top_k,
                                       r.top_p, r.min_p, &rng);
        size_t piece_len = 0;
        char *piece = ds4_token_text(engine, token, &piece_len);
        buf_append(&text, piece, piece_len);
        free(piece);
        observe_tool_markers(text.ptr ? text.ptr : "", &saw_tool_start, &saw_tool_end, NULL);
        if (saw_tool_end) break;
        if (ds4_session_eval(session, token, err, sizeof(err)) != 0) {
            decode_ok = false;
            break;
        }
    }

    char *content = NULL;
    char *reasoning = NULL;
    tool_calls calls = {0};
    bool parsed = parse_generated_message(text.ptr ? text.ptr : "",
                                          &content, &reasoning, &calls);
    TEST_ASSERT(decode_ok);
    TEST_ASSERT(parsed);
    TEST_ASSERT(calls.len > 0);
    TEST_ASSERT(calls.len > 0 && !strcmp(calls.v[0].name, "list_files"));

    free(content);
    free(reasoning);
    tool_calls_free(&calls);
    buf_free(&text);
    ds4_session_free(session);
    request_free(&r);
}

static void test_tool_call_quality(void) {
    fprintf(stderr, "ds4-test: tool-call quality fast path\n");
    test_tool_call_quality_one(false);
    test_close_engine(false);
    fprintf(stderr, "ds4-test: tool-call quality exact path\n");
    test_tool_call_quality_one(true);
    test_close_engine(true);
}

#endif

static void test_server_unit_group(void) {
    ds4_server_unit_tests_run();
}

typedef void (*test_fn)(void);

typedef struct {
    const char *flag;
    const char *name;
    const char *desc;
    test_fn fn;
} ds4_test_entry;

static const ds4_test_entry test_entries[] = {
#ifndef DS4_NO_GPU
    {"--long-context", "long-context", "long-context story fact-recall regression", test_long_story_fact_recall},
    {"--tool-call-quality", "tool-call-quality", "model emits valid DSML tool calls", test_tool_call_quality},
    {"--logprob-vectors", "logprob-vectors", "official API top-logprob vector comparison", test_official_logprob_vectors},
    {"--metal-kernels", "metal-kernels", "isolated Metal kernel numeric regressions", test_metal_kernels},
#endif
    {"--server", "server", "server parser/rendering/cache unit tests", test_server_unit_group},
};

static void test_print_help(const char *prog) {
    printf("Usage: %s [--all | TEST...]\n\n", prog);
    puts("Tests:");
    puts("  --all");
    puts("      Run every test. This is the default, ordered from slower to faster.");
    for (size_t i = 0; i < sizeof(test_entries) / sizeof(test_entries[0]); i++) {
        printf("  %-20s %s\n", test_entries[i].flag, test_entries[i].desc);
    }
    puts("  --list");
    puts("      Print test names only.");
    puts("  -h, --help");
    puts("      Show this help.");
    puts("\nEnvironment:");
    puts("  DS4_TEST_MODEL=FILE        Model path. Default: ds4flash.gguf");
    puts("  DS4_TEST_LONG_PROMPT=FILE  Rendered long-context story fact prompt.");
    puts("  DS4_TEST_VECTOR_FILE=FILE  Simple official-vector fixture.");
}

static const ds4_test_entry *test_find_entry(const char *arg) {
    for (size_t i = 0; i < sizeof(test_entries) / sizeof(test_entries[0]); i++) {
        if (!strcmp(arg, test_entries[i].flag)) return &test_entries[i];
    }
    return NULL;
}

static void test_run_entry(const ds4_test_entry *entry) {
    int before = test_failures;
    fprintf(stderr, "%s:\n", entry->name);
    entry->fn();
    fprintf(stderr, "%s: ", entry->name);
    ds4_log(stderr,
            test_failures == before ? DS4_LOG_OK : DS4_LOG_ERROR,
            "%s",
            test_failures == before ? "OK" : "ERR");
    fputc('\n', stderr);
}

int main(int argc, char **argv) {
    bool run_all = argc == 1;
    bool selected[sizeof(test_entries) / sizeof(test_entries[0])] = {0};

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--all")) {
            run_all = true;
        } else if (!strcmp(argv[i], "--list")) {
            for (size_t j = 0; j < sizeof(test_entries) / sizeof(test_entries[0]); j++) {
                puts(test_entries[j].flag);
            }
            return 0;
        } else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
            test_print_help(argv[0]);
            return 0;
        } else {
            const ds4_test_entry *entry = test_find_entry(argv[i]);
            if (!entry) {
                fprintf(stderr, "ds4-test: unknown test switch: %s\n", argv[i]);
                test_print_help(argv[0]);
                return 2;
            }
            selected[(size_t)(entry - test_entries)] = true;
        }
    }

    if (run_all) {
        for (size_t i = 0; i < sizeof(test_entries) / sizeof(test_entries[0]); i++) {
            test_run_entry(&test_entries[i]);
        }
    } else {
        for (size_t i = 0; i < sizeof(test_entries) / sizeof(test_entries[0]); i++) {
            if (selected[i]) test_run_entry(&test_entries[i]);
        }
    }

#ifndef DS4_NO_GPU
    test_close_engines();
#endif

    if (test_failures) {
        fprintf(stderr, "ds4 tests: %d failure(s)\n", test_failures);
        return 1;
    }
    puts("ds4 tests: ok");
    return 0;
}
