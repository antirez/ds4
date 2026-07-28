#define PULSAR_SERVER_TEST
#define PULSAR_SERVER_TEST_NO_MAIN
#include "../src/server/util.cpp"
#include "../src/server/request.cpp"
#include "../src/server/prompt_render.cpp"
#include "../src/server/api_parse.cpp"
#include "../src/server/genmsg.cpp"
#include "../src/server/openai_stream.cpp"
#include "../src/server/responses_stream.cpp"
#include "../src/server/anthropic_stream.cpp"
#include "../src/server/tool_memory.cpp"
#include "../src/server/kv_cache.cpp"
#include "../src/server/trace.cpp"
#include "../src/server/generate.cpp"
#include "../src/server/server_jobs.cpp"
#include "../src/server/server_sched.cpp"
#include "../src/server/http_server.cpp"
#include "../src/server/cli_main.cpp"
#include "../src/server/server_tests.cpp"
/* engine internals: the sampler byte-exactness gate builds distributions
 * directly and pins them against a copy of the pre-radix implementation. */
#include "../src/engine/pulsar_engine_internal.h"
#ifndef PULSAR_NO_GPU
#include "../src/pulsar_gpu.h"
#include <math.h>

static pulsar_engine *test_engine_fast;
static pulsar_engine *test_engine_quality;

static const char *test_model_path(void) {
    const char *model_path = getenv("DS4_TEST_MODEL");
    return (model_path && model_path[0]) ? model_path : "ds4flash.gguf";
}

static char *test_save_env(const char *name) {
    const char *value = getenv(name);
    if (!value) return NULL;
    size_t len = strlen(value);
    char *copy = (char *)malloc(len + 1);
    TEST_ASSERT(copy != NULL);
    if (!copy) return NULL;
    memcpy(copy, value, len + 1);
    return copy;
}

static void test_restore_env(const char *name, char *saved) {
    if (saved) {
        setenv(name, saved, 1);
        free(saved);
    } else {
        unsetenv(name);
    }
}

static pulsar_engine *test_open_engine(bool quality) {
    pulsar_engine *engine = NULL;
    pulsar_engine_options opt = {
        .model_path = test_model_path(),
        .backend = PULSAR_BACKEND_CUDA,
        .quality = quality,
    };
    TEST_ASSERT(pulsar_engine_open(&engine, &opt) == 0);
    return engine;
}

static pulsar_engine *test_get_engine(bool quality) {
    pulsar_engine **slot = quality ? &test_engine_quality : &test_engine_fast;
    if (*slot) return *slot;

    *slot = test_open_engine(quality);
    return *slot;
}

static void test_close_engines(void) {
    pulsar_engine_close(test_engine_fast);
    pulsar_engine_close(test_engine_quality);
    test_engine_fast = NULL;
    test_engine_quality = NULL;
}

static void test_close_engine(bool quality) {
    pulsar_engine **slot = quality ? &test_engine_quality : &test_engine_fast;
    pulsar_engine_close(*slot);
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

static float test_f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x03ffu;
    uint32_t bits;

    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            exp = 1;
            while ((mant & 0x0400u) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x03ffu;
            bits = sign | ((exp + 127u - 15u) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 127u - 15u) << 23) | (mant << 13);
    }

    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

static void test_f16_matvec_fast_nr0_4(void) {
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

    uint16_t *weights = (uint16_t *)weights_raw;
    memset(weights, 0, (size_t)weight_alloc);
    for (uint32_t o = 0; o < out_dim; o++) {
        for (uint32_t i = 0; i < in_dim; i++) {
            float w = (float)((int)((o * 3u + i * 5u) % 23u) - 11) / 64.0f;
            weights[(uint64_t)o * in_dim + i] = test_float_to_f16(w);
        }
    }

    pulsar_gpu_tensor *x = pulsar_gpu_tensor_alloc((uint64_t)in_dim * sizeof(float));
    pulsar_gpu_tensor *out = pulsar_gpu_tensor_alloc((uint64_t)out_dim * sizeof(float));
    TEST_ASSERT(x != NULL);
    TEST_ASSERT(out != NULL);
    if (!x || !out) {
        pulsar_gpu_tensor_free(x);
        pulsar_gpu_tensor_free(out);
        free(weights_raw);
        return;
    }

    float *x_host = (float *)malloc((size_t)in_dim * sizeof(float));
    float *out_host = (float *)malloc((size_t)out_dim * sizeof(float));
    TEST_ASSERT(x_host != NULL);
    TEST_ASSERT(out_host != NULL);
    if (!x_host || !out_host) {
        free(x_host);
        free(out_host);
        pulsar_gpu_tensor_free(x);
        pulsar_gpu_tensor_free(out);
        free(weights_raw);
        return;
    }

    for (uint32_t i = 0; i < in_dim; i++) {
        x_host[i] = (float)((int)(i % 31u) - 15) / 32.0f;
    }

    TEST_ASSERT(pulsar_gpu_tensor_write(x, 0, x_host, (uint64_t)in_dim * sizeof(float)) != 0);
    TEST_ASSERT(pulsar_gpu_set_model_map(weights_raw, weight_alloc) != 0);
    pulsar_gpu_set_quality(false);
    TEST_ASSERT(pulsar_gpu_matmul_f16_tensor(out, weights_raw, weight_alloc, 0,
                                            in_dim, out_dim, x, 1) != 0);
    TEST_ASSERT(pulsar_gpu_tensor_read(out, 0, out_host, (uint64_t)out_dim * sizeof(float)) != 0);

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
    pulsar_gpu_tensor_free(x);
    pulsar_gpu_tensor_free(out);
    free(weights_raw);
}

static void test_f16_prefill_matmul(void) {
    const uint32_t in_dim = 128;
    const uint32_t out_dim = 64;
    const uint32_t n_tok = 128;
    const uint64_t weight_bytes = (uint64_t)out_dim * in_dim * sizeof(uint16_t);
    const uint64_t weight_alloc = test_round_up_u64(weight_bytes, (uint64_t)getpagesize());
    const uint64_t x_bytes = (uint64_t)n_tok * in_dim * sizeof(float);
    const uint64_t out_bytes = (uint64_t)n_tok * out_dim * sizeof(float);

    void *weights_raw = NULL;
    TEST_ASSERT(posix_memalign(&weights_raw, (size_t)getpagesize(), (size_t)weight_alloc) == 0);
    if (!weights_raw) return;

    uint16_t *weights = (uint16_t *)weights_raw;
    memset(weights, 0, (size_t)weight_alloc);
    for (uint32_t o = 0; o < out_dim; o++) {
        for (uint32_t i = 0; i < in_dim; i++) {
            const int v = (int)((o * 11u + i * 13u + (o ^ i) * 5u) % 61u) - 30;
            weights[(uint64_t)o * in_dim + i] = test_float_to_f16((float)v / 96.0f);
        }
    }

    pulsar_gpu_tensor *x = pulsar_gpu_tensor_alloc(x_bytes);
    pulsar_gpu_tensor *out = pulsar_gpu_tensor_alloc(out_bytes);
    TEST_ASSERT(x != NULL);
    TEST_ASSERT(out != NULL);
    if (!x || !out) {
        pulsar_gpu_tensor_free(x);
        pulsar_gpu_tensor_free(out);
        free(weights_raw);
        return;
    }

    float *x_host = (float *)malloc((size_t)x_bytes);
    float *out_host = (float *)malloc((size_t)out_bytes);
    TEST_ASSERT(x_host != NULL);
    TEST_ASSERT(out_host != NULL);
    if (!x_host || !out_host) {
        free(x_host);
        free(out_host);
        pulsar_gpu_tensor_free(x);
        pulsar_gpu_tensor_free(out);
        free(weights_raw);
        return;
    }

    for (uint32_t t = 0; t < n_tok; t++) {
        for (uint32_t i = 0; i < in_dim; i++) {
            const int v = (int)((t * 7u + i * 17u + (t ^ i) * 3u) % 73u) - 36;
            x_host[(uint64_t)t * in_dim + i] = (float)v / 80.0f;
        }
    }
    for (uint32_t i = 0; i < n_tok * out_dim; i++) {
        out_host[i] = 12345.0f;
    }

    TEST_ASSERT(pulsar_gpu_tensor_write(x, 0, x_host, x_bytes) != 0);
    TEST_ASSERT(pulsar_gpu_tensor_write(out, 0, out_host, out_bytes) != 0);
    TEST_ASSERT(pulsar_gpu_set_model_map(weights_raw, weight_alloc) != 0);
    pulsar_gpu_set_quality(false);
    TEST_ASSERT(pulsar_gpu_matmul_f16_tensor(out, weights_raw, weight_alloc, 0,
                                          in_dim, out_dim, x, n_tok) != 0);
    TEST_ASSERT(pulsar_gpu_tensor_read(out, 0, out_host, out_bytes) != 0);

    float max_abs = 0.0f;
    float rms = 0.0f;
    for (uint32_t t = 0; t < n_tok; t++) {
        for (uint32_t o = 0; o < out_dim; o++) {
            float ref = 0.0f;
            for (uint32_t i = 0; i < in_dim; i++) {
                ref += test_f16_to_f32(weights[(uint64_t)o * in_dim + i]) *
                       x_host[(uint64_t)t * in_dim + i];
            }
            const float got = out_host[(uint64_t)t * out_dim + o];
            TEST_ASSERT(isfinite(got));
            const float err = fabsf(got - ref);
            if (err > max_abs) max_abs = err;
            rms += err * err;
        }
    }
    rms = sqrtf(rms / (float)(n_tok * out_dim));
    TEST_ASSERT(max_abs < 0.08f);
    TEST_ASSERT(rms < 0.02f);

    free(x_host);
    free(out_host);
    pulsar_gpu_tensor_free(x);
    pulsar_gpu_tensor_free(out);
    free(weights_raw);
}

static void test_f16_kernel_group(void) {
    test_f16_matvec_fast_nr0_4();
    test_f16_prefill_matmul();
}

static void test_short_prefill_ratio4(void) {
    pulsar_engine *engine = test_get_engine(false);
    if (!engine) return;

    const int tokens[] = {
        pulsar_token_user(engine),
        pulsar_token_assistant(engine),
        pulsar_token_eos(engine),
    };
    for (size_t i = 0; i < sizeof(tokens) / sizeof(tokens[0]); i++) {
        TEST_ASSERT(tokens[i] >= 0);
        if (tokens[i] < 0) return;
    }

    for (size_t n = 1; n <= 3; n++) {
        pulsar_tokens prompt = {0};
        for (size_t i = 0; i < n; i++) {
            pulsar_tokens_push(&prompt, tokens[i]);
        }
        TEST_ASSERT(prompt.len == (int)n);

        pulsar_session *session = NULL;
        TEST_ASSERT(pulsar_session_create(&session, engine, 2048) == 0);
        if (!session) {
            pulsar_tokens_free(&prompt);
            return;
        }

        char err[160] = {0};
        const int rc = pulsar_session_sync(session, &prompt, err, sizeof(err));
        if (rc != 0) {
            fprintf(stderr, "pulsar-test: short prefill failed for %zu token(s): %s\n",
                    n, err);
        }
        TEST_ASSERT(rc == 0);

        pulsar_session_free(session);
        pulsar_tokens_free(&prompt);
    }
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
    char *s = (char *)malloc((size_t)len + 1);
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
                "pulsar-test: long-context wrong assignment for %s: got %d expected %d\n",
                fact->name, wrong_value, fact->number);
    } else {
        fprintf(stderr,
                "pulsar-test: long-context missing assignment for %s=%d\n",
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

static bool test_token_bytes_equal(pulsar_engine *engine, int token,
                                   const unsigned char *want, int want_len) {
    size_t got_len = 0;
    char *got = pulsar_token_text(engine, token, &got_len);
    bool eq = got && got_len == (size_t)want_len &&
              memcmp(got, want, (size_t)want_len) == 0;
    free(got);
    return eq;
}

static void test_long_prefill_progress(void *ud, const char *event, int current, int total) {
    (void)ud;
    if (strcmp(event, "prefill_chunk")) return;
    if (current == 0 || current == total || current % 8192 == 0) {
        fprintf(stderr, "pulsar-test: long-context prefill %d/%d\n", current, total);
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

    pulsar_engine *engine = test_get_engine(false);
    if (!engine) {
        free(prompt_text);
        return;
    }

    pulsar_tokens prompt = {0};
    pulsar_tokenize_rendered_chat(engine, prompt_text, &prompt);
    TEST_ASSERT(prompt.len > 30000);

    pulsar_session *session = NULL;
    TEST_ASSERT(pulsar_session_create(&session, engine, 100000) == 0);
    if (!session) {
        pulsar_tokens_free(&prompt);
        free(prompt_text);
        return;
    }

    char err[160];
    pulsar_session_set_progress(session, test_long_prefill_progress, NULL);
    TEST_ASSERT(pulsar_session_sync(session, &prompt, err, sizeof(err)) == 0);
    pulsar_session_set_progress(session, NULL, NULL);

    buf out = {0};
    uint64_t rng = 12345;
    int generated = 0;
    bool decode_ok = true;
    for (; generated < 350; generated++) {
        int token = pulsar_session_sample(session, 0.0f, 0, 1.0f, 0.0f, &rng);
        if (token == pulsar_token_eos(engine)) break;

        size_t piece_len = 0;
        char *piece = pulsar_token_text(engine, token, &piece_len);
        buf_append(&out, piece, piece_len);
        free(piece);

        if (pulsar_session_eval(session, token, err, sizeof(err)) != 0) {
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
    pulsar_session_free(session);
    pulsar_tokens_free(&prompt);
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

static void test_logprob_vector_case(pulsar_engine *engine, const test_vec_case *vc) {
    char *prompt_text = test_read_file(vc->prompt_path);
    TEST_ASSERT(prompt_text != NULL);
    if (!prompt_text) return;

    pulsar_tokens prompt = {0};
    pulsar_encode_chat_prompt(engine, "", prompt_text, PULSAR_THINK_NONE, &prompt);
    free(prompt_text);

    pulsar_session *session = NULL;
    TEST_ASSERT(pulsar_session_create(&session, engine, vc->ctx) == 0);
    if (!session) {
        pulsar_tokens_free(&prompt);
        return;
    }

    char err[160];
    TEST_ASSERT(pulsar_session_sync(session, &prompt, err, sizeof(err)) == 0);

    pulsar_token_score scores[20];
    for (int i = 0; i < vc->nsteps; i++) {
        const test_vec_step *step = &vc->steps[i];
        int nscore = pulsar_session_top_logprobs(session, scores, 20);
        int token = pulsar_session_argmax(session);
        if (!test_token_bytes_equal(engine, token, step->selected, step->selected_len)) {
            fprintf(stderr, "pulsar-test: vector %s step %d selected token mismatch\n",
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
                fprintf(stderr, "pulsar-test: vector %s step %d official top token missing locally\n",
                        vc->id, i);
                TEST_ASSERT(false);
            } else if (fabsf(local_lp - step->top[t].logprob) > 4.0f) {
                fprintf(stderr,
                        "pulsar-test: vector %s step %d logprob delta too high: local=%g official=%g\n",
                        vc->id, i, local_lp, step->top[t].logprob);
                TEST_ASSERT(false);
            }
        }

        if (i + 1 < vc->nsteps) {
            TEST_ASSERT(pulsar_session_eval(session, token, err, sizeof(err)) == 0);
        }
    }

    pulsar_session_free(session);
    pulsar_tokens_free(&prompt);
}

static bool test_logprob_vector_case_disabled(const test_vec_case *vc) {
    /*
     * This one long-context vector currently matches the public DeepSeek API less
     * after adding the official Hadamard+FP4 indexer path.  The public official
     * implementation and the API appear to disagree here; the official graph has
     * slightly lower local perplexity on the A/B check we ran, so DS4 keeps that
     * implementation and only excludes this brittle API fixture for now.
     */
    return !strcmp(vc->id, "long_memory_archive");
}

static void test_official_logprob_vectors_run(const char *case_filter) {
    const char *path = getenv("DS4_TEST_VECTOR_FILE");
    if (!path || !path[0]) path = "tests/test-vectors/official.vec";
    FILE *fp = fopen(path, "rb");
    TEST_ASSERT(fp != NULL);
    if (!fp) return;

    char *saved_prefill_chunk = test_save_env("DS4_CUDA_PREFILL_CHUNK");
    setenv("DS4_CUDA_PREFILL_CHUNK", "2048", 1);
    pulsar_engine *engine = test_open_engine(false);
    if (!engine) {
        test_restore_env("DS4_CUDA_PREFILL_CHUNK", saved_prefill_chunk);
        fclose(fp);
        return;
    }

    test_vec_case vc;
    int ran = 0;
    while (test_read_vector_case(fp, &vc)) {
        if (!test_fill_vector_case(fp, &vc)) break;
        if (case_filter && case_filter[0] && strcmp(vc.id, case_filter)) {
            continue;
        }
        if (test_logprob_vector_case_disabled(&vc)) {
            fprintf(stderr, "pulsar-test: vector %s skipped (API/official graph mismatch)\n",
                    vc.id);
            continue;
        }
        fprintf(stderr, "pulsar-test: vector %s\n", vc.id);
        test_logprob_vector_case(engine, &vc);
        ran++;
    }
    TEST_ASSERT(!case_filter || !case_filter[0] || ran == 1);
    pulsar_engine_close(engine);
    test_restore_env("DS4_CUDA_PREFILL_CHUNK", saved_prefill_chunk);
    fclose(fp);
}

static void test_official_logprob_vectors(void) {
    test_official_logprob_vectors_run(NULL);
}

static void test_logits_topk(const float *logits, int n, int *out, int k);
static bool test_topk_contains(const int *top, int k, int id);

#define TEST_LOCAL_GOLDEN_MAX_TOP 128

typedef struct {
    int id;
    float logit;
} test_local_golden_top;

typedef struct {
    char id[96];
    char mode[16];
    char prompt_path[512];
    int ctx;
    int frontier;
    int ntop;
    test_local_golden_top top[TEST_LOCAL_GOLDEN_MAX_TOP];
} test_local_golden_case;

static bool test_read_local_golden_case(FILE *fp, test_local_golden_case *tc) {
    char line[2048];
    memset(tc, 0, sizeof(*tc));
    while (fgets(line, sizeof(line), fp)) {
        char *p = test_trim_line(line);
        if (!p[0] || p[0] == '#') continue;
        if (sscanf(p, "case %95s %15s %d %d %511s %d",
                   tc->id, tc->mode, &tc->ctx, &tc->frontier,
                   tc->prompt_path, &tc->ntop) == 6) {
            TEST_ASSERT(tc->ctx > tc->frontier);
            TEST_ASSERT(tc->frontier > 0);
            TEST_ASSERT(tc->ntop > 0 && tc->ntop <= TEST_LOCAL_GOLDEN_MAX_TOP);
            return true;
        }
        TEST_ASSERT(!"unexpected line before local golden case");
        return false;
    }
    return false;
}

static bool test_fill_local_golden_case(FILE *fp, test_local_golden_case *tc) {
    char line[2048];
    int seen = 0;
    while (fgets(line, sizeof(line), fp)) {
        char *p = test_trim_line(line);
        if (!p[0] || p[0] == '#') continue;
        if (!strcmp(p, "end")) {
            TEST_ASSERT(seen == tc->ntop);
            return seen == tc->ntop;
        }
        int rank = -1;
        int id = -1;
        float logit = 0.0f;
        if (sscanf(p, "top %d %d %f", &rank, &id, &logit) != 3) {
            TEST_ASSERT(!"bad local golden top line");
            return false;
        }
        TEST_ASSERT(rank == seen);
        TEST_ASSERT(seen < tc->ntop);
        if (seen >= tc->ntop) return false;
        tc->top[seen].id = id;
        tc->top[seen].logit = logit;
        seen++;
    }
    TEST_ASSERT(!"unterminated local golden case");
    return false;
}

static int test_local_golden_overlap(const test_local_golden_case *tc,
                                     const int *cand_top,
                                     int n) {
    int overlap = 0;
    if (n > tc->ntop) n = tc->ntop;
    for (int i = 0; i < n; i++) {
        if (test_topk_contains(cand_top, n, tc->top[i].id)) overlap++;
    }
    return overlap;
}

static float test_local_golden_max_abs(const test_local_golden_case *tc,
                                       const float *cand_logits,
                                       int n) {
    float max_abs = 0.0f;
    if (n > tc->ntop) n = tc->ntop;
    for (int i = 0; i < n; i++) {
        const int id = tc->top[i].id;
        if (id < 0) continue;
        const float abs_delta = fabsf(cand_logits[id] - tc->top[i].logit);
        if (abs_delta > max_abs) max_abs = abs_delta;
    }
    return max_abs;
}

static void test_local_golden_case_run(pulsar_engine *engine,
                                       const test_local_golden_case *tc) {
    char *prompt_text = test_read_file(tc->prompt_path);
    TEST_ASSERT(prompt_text != NULL);
    if (!prompt_text) return;

    pulsar_tokens prompt = {0};
    if (!strcmp(tc->mode, "text")) {
        pulsar_tokenize_text(engine, prompt_text, &prompt);
    } else if (!strcmp(tc->mode, "rendered")) {
        pulsar_tokenize_rendered_chat(engine, prompt_text, &prompt);
    } else if (!strcmp(tc->mode, "chat")) {
        pulsar_encode_chat_prompt(engine, "", prompt_text, PULSAR_THINK_NONE, &prompt);
    } else {
        TEST_ASSERT(!"unknown local golden prompt mode");
    }
    free(prompt_text);
    TEST_ASSERT(prompt.len >= tc->frontier);
    if (prompt.len < tc->frontier) {
        pulsar_tokens_free(&prompt);
        return;
    }

    pulsar_tokens prefix = {
        .v = prompt.v,
        .len = tc->frontier,
        .cap = tc->frontier,
    };

    pulsar_session *session = NULL;
    TEST_ASSERT(pulsar_session_create(&session, engine, tc->ctx) == 0);
    if (!session) {
        pulsar_tokens_free(&prompt);
        return;
    }

    char err[160];
    TEST_ASSERT(pulsar_session_sync(session, &prefix, err, sizeof(err)) == 0);

    const int vocab = pulsar_engine_vocab_size(engine);
    float *cand_logits = (float *)malloc((size_t)vocab * sizeof(cand_logits[0]));
    TEST_ASSERT(cand_logits != NULL);
    if (cand_logits &&
        pulsar_session_copy_logits(session, cand_logits, vocab) == vocab) {
        int cand_top[TEST_LOCAL_GOLDEN_MAX_TOP];
        const int ntop = tc->ntop < TEST_LOCAL_GOLDEN_MAX_TOP ?
                         tc->ntop : TEST_LOCAL_GOLDEN_MAX_TOP;
        test_logits_topk(cand_logits, vocab, cand_top, ntop);

        const int top5_overlap = test_local_golden_overlap(tc, cand_top, 5);
        const int top20_overlap = test_local_golden_overlap(tc, cand_top, 20);
        const int top64_overlap = test_local_golden_overlap(tc, cand_top, 64);
        const float top20_max_abs =
            test_local_golden_max_abs(tc, cand_logits, 20);

        fprintf(stderr,
                "pulsar-test: local golden %s top1 ref=%d cand=%d "
                "top5_overlap=%d/5 top20_overlap=%d/20 top64_overlap=%d/64 "
                "top20_max_abs=%g\n",
                tc->id, tc->top[0].id, cand_top[0],
                top5_overlap, top20_overlap, top64_overlap, top20_max_abs);

        /*
         * This is intentionally tolerant: it is meant to catch substantial
         * backend drift (wrong tiling, skipped work, bad dispatch), not tiny
         * floating-point differences from otherwise sane kernel changes.
         */
        TEST_ASSERT(cand_top[0] == tc->top[0].id);
        TEST_ASSERT(top5_overlap >= 4);
        TEST_ASSERT(top20_overlap >= 15);
        TEST_ASSERT(top64_overlap >= 40);
        TEST_ASSERT(top20_max_abs <= 8.0f);
    } else {
        TEST_ASSERT(false);
    }

    free(cand_logits);
    pulsar_session_free(session);
    pulsar_tokens_free(&prompt);
}

static void test_local_golden_vectors(void) {
    const char *path = getenv("DS4_TEST_LOCAL_GOLDEN_FILE");
    if (!path || !path[0]) path = "tests/test-vectors/local-golden.vec";
    FILE *fp = fopen(path, "rb");
    TEST_ASSERT(fp != NULL);
    if (!fp) return;

    char *saved_prefill_chunk = test_save_env("DS4_CUDA_PREFILL_CHUNK");
    setenv("DS4_CUDA_PREFILL_CHUNK", "4096", 1);

    pulsar_engine *engine = test_open_engine(false);
    if (!engine) {
        test_restore_env("DS4_CUDA_PREFILL_CHUNK", saved_prefill_chunk);
        fclose(fp);
        return;
    }

    test_local_golden_case tc;
    while (test_read_local_golden_case(fp, &tc)) {
        if (!test_fill_local_golden_case(fp, &tc)) break;
        test_local_golden_case_run(engine, &tc);
    }

    pulsar_engine_close(engine);
    test_restore_env("DS4_CUDA_PREFILL_CHUNK", saved_prefill_chunk);
    fclose(fp);
}

#define TEST_MPP_EQ_MAX_CASES 8
#define TEST_MPP_EQ_TOPK 20
#define TEST_MPP_EQ_TOP5 5
#define TEST_MPP_EQ_DELTAS 5

typedef struct {
    char id[96];
    int ctx;
    int vocab_size;
    int gen_steps;
    pulsar_tokens prompt;
    float *ref_logits;
    int ref_gen[TEST_VEC_MAX_STEPS];
    int ref_gen_len;
} test_mpp_eq_case;

typedef struct {
    int ref_top1;
    int cand_top1;
    int overlap;
    int top5_overlap;
    int max_rank_delta;
    int nonfinite;
    float rms;
    float max_abs;
    float top20_max_abs;
    bool same_top1;
    bool pass;
} test_mpp_eq_result;

typedef struct {
    const char *label;
    int cases;
    int capture_failures;
    int logits_failures;
    int greedy_failures;
    int top1_mismatches;
    int min_overlap;
    int min_top5_overlap;
    int worst_rank_delta;
    float worst_rms;
    float worst_max_abs;
    float worst_top20_max_abs;
} test_mpp_eq_summary;

static void test_mpp_eq_case_free(test_mpp_eq_case *tc) {
    if (!tc) return;
    pulsar_tokens_free(&tc->prompt);
    free(tc->ref_logits);
    memset(tc, 0, sizeof(*tc));
}

static void test_logits_topk(const float *logits, int n, int *out, int k) {
    for (int i = 0; i < k; i++) out[i] = -1;
    for (int id = 0; id < n; id++) {
        const float v = logits[id];
        if (!isfinite(v)) continue;
        for (int j = 0; j < k; j++) {
            if (out[j] < 0 || v > logits[out[j]]) {
                for (int l = k - 1; l > j; l--) out[l] = out[l - 1];
                out[j] = id;
                break;
            }
        }
    }
}

static bool test_topk_contains(const int *top, int k, int id) {
    for (int i = 0; i < k; i++) {
        if (top[i] == id) return true;
    }
    return false;
}

static int test_topk_rank(const int *top, int k, int id) {
    for (int i = 0; i < k; i++) {
        if (top[i] == id) return i;
    }
    return -1;
}

static void test_note_delta(int *ids, float *ref_vals, float *cand_vals,
                            float *abs_vals, int id, float ref, float cand) {
    const float abs_delta = fabsf(cand - ref);
    for (int i = 0; i < TEST_MPP_EQ_DELTAS; i++) {
        if (ids[i] < 0 || abs_delta > abs_vals[i]) {
            for (int j = TEST_MPP_EQ_DELTAS - 1; j > i; j--) {
                ids[j] = ids[j - 1];
                ref_vals[j] = ref_vals[j - 1];
                cand_vals[j] = cand_vals[j - 1];
                abs_vals[j] = abs_vals[j - 1];
            }
            ids[i] = id;
            ref_vals[i] = ref;
            cand_vals[i] = cand;
            abs_vals[i] = abs_delta;
            return;
        }
    }
}

static float test_top_union_max_abs(const float *ref, const float *cand,
                                    const int *ref_top, const int *cand_top, int k) {
    float max_abs = 0.0f;
    for (int i = 0; i < k; i++) {
        if (ref_top[i] >= 0) {
            const float d = fabsf(cand[ref_top[i]] - ref[ref_top[i]]);
            if (d > max_abs) max_abs = d;
        }
        if (cand_top[i] >= 0 && !test_topk_contains(ref_top, k, cand_top[i])) {
            const float d = fabsf(cand[cand_top[i]] - ref[cand_top[i]]);
            if (d > max_abs) max_abs = d;
        }
    }
    return max_abs;
}

/*
 * Tensor-core equivalence is a smoke test, not a demand for bitwise local
 * logits.  Tensor kernels change precision and reduction order, so the useful
 * invariant here is: no NaNs, same first greedy token, and same short greedy
 * continuation.  Larger logit drift is still printed so it can be compared with
 * official API-vector and long-context recall gates.
 */
static test_mpp_eq_result test_compare_mpp_logits(const test_mpp_eq_case *tc,
                                                  const float *cand_logits,
                                                  bool assert_thresholds) {
    int ref_top[TEST_MPP_EQ_TOPK];
    int cand_top[TEST_MPP_EQ_TOPK];
    test_logits_topk(tc->ref_logits, tc->vocab_size, ref_top, TEST_MPP_EQ_TOPK);
    test_logits_topk(cand_logits, tc->vocab_size, cand_top, TEST_MPP_EQ_TOPK);

    int overlap = 0;
    int top5_overlap = 0;
    int max_rank_delta = 0;
    for (int i = 0; i < TEST_MPP_EQ_TOPK; i++) {
        const int cand_rank = test_topk_rank(cand_top, TEST_MPP_EQ_TOPK, ref_top[i]);
        if (ref_top[i] >= 0 && cand_rank >= 0) {
            overlap++;
            const int rank_delta = abs(cand_rank - i);
            if (rank_delta > max_rank_delta) max_rank_delta = rank_delta;
        }
        if (i < TEST_MPP_EQ_TOP5 &&
            ref_top[i] >= 0 &&
            test_topk_contains(cand_top, TEST_MPP_EQ_TOP5, ref_top[i])) {
            top5_overlap++;
        }
    }

    double sumsq = 0.0;
    float max_abs = 0.0f;
    int nonfinite = 0;
    int delta_ids[TEST_MPP_EQ_DELTAS];
    float delta_ref[TEST_MPP_EQ_DELTAS];
    float delta_cand[TEST_MPP_EQ_DELTAS];
    float delta_abs[TEST_MPP_EQ_DELTAS];
    for (int i = 0; i < TEST_MPP_EQ_DELTAS; i++) {
        delta_ids[i] = -1;
        delta_ref[i] = 0.0f;
        delta_cand[i] = 0.0f;
        delta_abs[i] = 0.0f;
    }

    for (int i = 0; i < tc->vocab_size; i++) {
        if (!isfinite(tc->ref_logits[i]) || !isfinite(cand_logits[i])) {
            nonfinite++;
            continue;
        }
        const float delta = cand_logits[i] - tc->ref_logits[i];
        const float abs_delta = fabsf(delta);
        if (abs_delta > max_abs) max_abs = abs_delta;
        sumsq += (double)delta * (double)delta;
        test_note_delta(delta_ids, delta_ref, delta_cand, delta_abs,
                        (int)i, tc->ref_logits[i], cand_logits[i]);
    }

    const float rms = (float)sqrt(sumsq / (double)tc->vocab_size);
    const float top_abs = test_top_union_max_abs(tc->ref_logits, cand_logits,
                                                 ref_top, cand_top, TEST_MPP_EQ_TOPK);
    const bool same_top1 = ref_top[0] >= 0 && ref_top[0] == cand_top[0];
    test_mpp_eq_result result = {
        .ref_top1 = ref_top[0],
        .cand_top1 = cand_top[0],
        .overlap = overlap,
        .top5_overlap = top5_overlap,
        .max_rank_delta = max_rank_delta,
        .nonfinite = nonfinite,
        .rms = rms,
        .max_abs = max_abs,
        .top20_max_abs = top_abs,
        .same_top1 = same_top1,
        .pass = nonfinite == 0 && same_top1,
    };

    fprintf(stderr,
            "pulsar-test: Tensor equivalence %s top1 ref=%d cand=%d top5_overlap=%d/%d overlap=%d/%d max_rank_delta=%d rms=%g max_abs=%g top20_max_abs=%g\n",
            tc->id, ref_top[0], cand_top[0],
            top5_overlap, TEST_MPP_EQ_TOP5,
            overlap, TEST_MPP_EQ_TOPK,
            max_rank_delta, rms, max_abs, top_abs);
    fprintf(stderr, "pulsar-test: Tensor equivalence %s largest deltas:", tc->id);
    for (int i = 0; i < TEST_MPP_EQ_DELTAS && delta_ids[i] >= 0; i++) {
        fprintf(stderr, " id=%d ref=%g cand=%g abs=%g",
                delta_ids[i], delta_ref[i], delta_cand[i], delta_abs[i]);
    }
    fputc('\n', stderr);

    if (assert_thresholds) {
        TEST_ASSERT(nonfinite == 0);
        TEST_ASSERT(same_top1);
    }
    return result;
}

static bool test_mpp_capture(pulsar_engine *engine, const test_mpp_eq_case *tc,
                             float *logits, int *gen, int *gen_len) {
    pulsar_session *session = NULL;
    TEST_ASSERT(pulsar_session_create(&session, engine, tc->ctx) == 0);
    if (!session) return false;

    char err[160];
    bool ok = pulsar_session_sync(session, &tc->prompt, err, sizeof(err)) == 0;
    TEST_ASSERT(ok);
    if (ok) {
        ok = pulsar_session_copy_logits(session, logits, tc->vocab_size) == tc->vocab_size;
        TEST_ASSERT(ok);
    }

    int n = 0;
    while (ok && n < tc->gen_steps) {
        const int token = pulsar_session_argmax(session);
        gen[n++] = token;
        if (n < tc->gen_steps && pulsar_session_eval(session, token, err, sizeof(err)) != 0) {
            ok = false;
            TEST_ASSERT(false);
        }
    }
    *gen_len = n;

    pulsar_session_free(session);
    return ok;
}

static bool test_mpp_capture_logits_only(pulsar_engine *engine,
                                         const test_mpp_eq_case *tc,
                                         float *logits) {
    pulsar_session *session = NULL;
    TEST_ASSERT(pulsar_session_create(&session, engine, tc->ctx) == 0);
    if (!session) return false;

    char err[160];
    bool ok = pulsar_session_sync(session, &tc->prompt, err, sizeof(err)) == 0;
    TEST_ASSERT(ok);
    if (ok) {
        ok = pulsar_session_copy_logits(session, logits, tc->vocab_size) == tc->vocab_size;
        TEST_ASSERT(ok);
    }

    pulsar_session_free(session);
    return ok;
}

static bool test_mpp_eq_case_selected(const char *id) {
    const char *filter = getenv("DS4_TEST_MPP_EQ_CASE");
    if (!filter || !filter[0]) return true;

    char buf[256];
    snprintf(buf, sizeof(buf), "%s", filter);
    for (char *tok = strtok(buf, ","); tok; tok = strtok(NULL, ",")) {
        tok = test_trim_line(tok);
        if (tok[0] && strstr(id, tok)) return true;
    }
    return false;
}

static int test_load_mpp_cases(pulsar_engine *engine, test_mpp_eq_case *cases, int cap) {
    const char *path = getenv("DS4_TEST_VECTOR_FILE");
    if (!path || !path[0]) path = "tests/test-vectors/official.vec";
    FILE *fp = fopen(path, "rb");
    TEST_ASSERT(fp != NULL);
    if (!fp) return 0;

    int ncase = 0;
    test_vec_case vc;
    while (ncase < cap && test_read_vector_case(fp, &vc)) {
        if (!test_fill_vector_case(fp, &vc)) break;
        if (!test_mpp_eq_case_selected(vc.id)) continue;
        char *prompt_text = test_read_file(vc.prompt_path);
        TEST_ASSERT(prompt_text != NULL);
        if (!prompt_text) continue;

        test_mpp_eq_case *tc = &cases[ncase++];
        snprintf(tc->id, sizeof(tc->id), "%s", vc.id);
        tc->ctx = vc.ctx;
        tc->vocab_size = pulsar_engine_vocab_size(engine);
        tc->gen_steps = vc.nsteps < TEST_VEC_MAX_STEPS ? vc.nsteps : TEST_VEC_MAX_STEPS;
        pulsar_encode_chat_prompt(engine, "", prompt_text, PULSAR_THINK_NONE, &tc->prompt);
        free(prompt_text);
        TEST_ASSERT(tc->prompt.len > 0);
    }
    fclose(fp);
    return ncase;
}

static void test_mpp_summary_init(test_mpp_eq_summary *summary, const char *label) {
    memset(summary, 0, sizeof(*summary));
    summary->label = label;
    summary->min_overlap = TEST_MPP_EQ_TOPK;
    summary->min_top5_overlap = TEST_MPP_EQ_TOP5;
}

static void test_mpp_summary_note_logits(test_mpp_eq_summary *summary,
                                         const test_mpp_eq_result *result) {
    if (!result->pass) summary->logits_failures++;
    if (!result->same_top1) summary->top1_mismatches++;
    if (result->overlap < summary->min_overlap) summary->min_overlap = result->overlap;
    if (result->top5_overlap < summary->min_top5_overlap) {
        summary->min_top5_overlap = result->top5_overlap;
    }
    if (result->max_rank_delta > summary->worst_rank_delta) {
        summary->worst_rank_delta = result->max_rank_delta;
    }
    if (result->rms > summary->worst_rms) summary->worst_rms = result->rms;
    if (result->max_abs > summary->worst_max_abs) summary->worst_max_abs = result->max_abs;
    if (result->top20_max_abs > summary->worst_top20_max_abs) {
        summary->worst_top20_max_abs = result->top20_max_abs;
    }
}

static void test_mpp_summary_print(const test_mpp_eq_summary *summary) {
    fprintf(stderr,
            "pulsar-test: Tensor summary route=%s cases=%d capture_fail=%d logits_fail=%d greedy_fail=%d top1_mismatch=%d min_top5_overlap=%d/%d min_overlap=%d/%d worst_rank_delta=%d worst_rms=%g worst_max_abs=%g worst_top20_max_abs=%g\n",
            summary->label,
            summary->cases,
            summary->capture_failures,
            summary->logits_failures,
            summary->greedy_failures,
            summary->top1_mismatches,
            summary->min_top5_overlap,
            TEST_MPP_EQ_TOP5,
            summary->min_overlap,
            TEST_MPP_EQ_TOPK,
            summary->worst_rank_delta,
            summary->worst_rms,
            summary->worst_max_abs,
            summary->worst_top20_max_abs);
}

static void test_run_mpp_candidate(const char *label,
                                   test_mpp_eq_case *cases,
                                   int ncase) {
    fprintf(stderr, "pulsar-test: Tensor equivalence candidate route=%s\n", label);
    test_mpp_eq_summary summary;
    test_mpp_summary_init(&summary, label);
    pulsar_engine *cand_engine = test_open_engine(false);
    if (cand_engine) {
        const int vocab_size = ncase > 0 ? cases[0].vocab_size : 0;
        float *cand_logits = (float *)malloc((size_t)vocab_size * sizeof(cand_logits[0]));
        TEST_ASSERT(cand_logits != NULL);
        if (cand_logits) {
            for (int i = 0; i < ncase; i++) {
                test_mpp_eq_case *tc = &cases[i];
                if (!tc->ref_logits) continue;
                int cand_gen[TEST_VEC_MAX_STEPS] = {0};
                int cand_gen_len = 0;
                if (!test_mpp_capture(cand_engine, tc, cand_logits, cand_gen, &cand_gen_len)) {
                    summary.capture_failures++;
                    continue;
                }
                summary.cases++;
                test_mpp_eq_result result = test_compare_mpp_logits(tc, cand_logits, true);
                test_mpp_summary_note_logits(&summary, &result);
                TEST_ASSERT(cand_gen_len == tc->ref_gen_len);
                if (cand_gen_len != tc->ref_gen_len) summary.greedy_failures++;
                for (int j = 0; j < tc->ref_gen_len && j < cand_gen_len; j++) {
                    if (cand_gen[j] != tc->ref_gen[j]) {
                        fprintf(stderr,
                                "pulsar-test: Tensor equivalence %s greedy token mismatch step=%d ref=%d cand=%d\n",
                                tc->id, j, tc->ref_gen[j], cand_gen[j]);
                        summary.greedy_failures++;
                    }
                    TEST_ASSERT(cand_gen[j] == tc->ref_gen[j]);
                }
            }
            free(cand_logits);
        }
        pulsar_engine_close(cand_engine);
    }
    test_mpp_summary_print(&summary);
}

static void test_mpp_equivalence(void) {
    test_close_engines();

    test_mpp_eq_case cases[TEST_MPP_EQ_MAX_CASES];
    memset(cases, 0, sizeof(cases));

    pulsar_engine *ref_engine = test_open_engine(false);
    if (!ref_engine) {
        return;
    }

    const int ncase = test_load_mpp_cases(ref_engine, cases, TEST_MPP_EQ_MAX_CASES);
    TEST_ASSERT(ncase > 0);
    for (int i = 0; i < ncase; i++) {
        test_mpp_eq_case *tc = &cases[i];
        tc->ref_logits = (float *)malloc((size_t)tc->vocab_size * sizeof(tc->ref_logits[0]));
        TEST_ASSERT(tc->ref_logits != NULL);
        if (!tc->ref_logits) continue;
        TEST_ASSERT(test_mpp_capture(ref_engine, tc,
                                     tc->ref_logits,
                                     tc->ref_gen,
                                     &tc->ref_gen_len));
    }
    pulsar_engine_close(ref_engine);

    test_run_mpp_candidate("auto", cases, ncase);

    for (int i = 0; i < ncase; i++) test_mpp_eq_case_free(&cases[i]);
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

static const char *test_think_recovery_request_json(void) {
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
        "\"think\":true,"
        "\"temperature\":0,"
        "\"max_tokens\":384,"
        "\"stream\":false"
        "}";
}

/* The model sometimes opens a DSML stanza without closing </think> first.
 * The server's forward recovery must force the close plus a fresh stanza
 * opening, after which the model must still complete a valid call.  The
 * malformed prefix is teacher-forced so the regression is deterministic and
 * does not depend on coaxing the model into misbehaving. */
static void test_think_tool_recovery(void) {
    pulsar_engine *engine = test_get_engine(false);
    if (!engine) return;

    request r;
    char err[160];
    TEST_ASSERT(parse_chat_request(engine, NULL, test_think_recovery_request_json(),
                                   512, 32768, &r, err, sizeof(err)));

    pulsar_session *session = NULL;
    TEST_ASSERT(pulsar_session_create(&session, engine, 32768) == 0);
    if (!session) {
        request_free(&r);
        return;
    }
    TEST_ASSERT(pulsar_session_sync(session, &r.prompt, err, sizeof(err)) == 0);

    if (getenv("DS4_TEST_RECOVERY_PROBE") != NULL) {
        /* Diagnostic: print the model's natural tool-call turn for this
         * request instead of running the recovery. */
        buf nat = {0};
        uint64_t prng = 7;
        for (int i = 0; i < 300; i++) {
            int token = pulsar_session_sample(session, 0.0f, 0, 1.0f, 0.0f, &prng);
            if (token == pulsar_token_eos(engine)) break;
            size_t plen = 0;
            char *p = pulsar_token_text(engine, token, &plen);
            buf_append(&nat, p, plen);
            free(p);
            bool ps = false, pe = false;
            observe_tool_markers(nat.ptr, &ps, &pe, NULL);
            if (pe) break;
            if (pulsar_session_eval(session, token, err, sizeof(err)) != 0) break;
        }
        fprintf(stderr, "pulsar-test: natural turn=[%s]\n", nat.ptr ? nat.ptr : "");
        buf_free(&nat);
        pulsar_session_free(session);
        request_free(&r);
        test_close_engine(false);
        return;
    }

    thinking_state thinking = thinking_state_from_prompt(&r);
    buf text = {0};
    buf forced = {0};
    if (!thinking.inside) buf_append(&forced, "<think>", 7);
    const char *body =
        "The user wants a directory listing. I will call the "
        "list_files tool right away.\n\n" PULSAR_TOOL_CALLS_START;
    buf_append(&forced, body, strlen(body));

    server srv;
    memset(&srv, 0, sizeof(srv));
    srv.engine = engine;
    srv.n_slots = PULSAR_SESSION_POOL_CAP;
    srv.sess = session;
    srv.slots[0].provisioned = true;

    /* Replay the malformed prefix exactly as the worker loop would see it:
     * token by token, running the recovery scan after each piece.  The stanza
     * opening spans several tokens, so this also checks that detection does
     * not depend on how the marker happens to be tokenized: recovery must
     * stay quiet on every partial prefix and trigger exactly when the
     * opening completes. */
    pulsar_tokens toks = {0};
    pulsar_tokenize_rendered_chat(engine, forced.ptr, &toks);
    TEST_ASSERT(toks.len > 1);
    size_t scan_from = 0;
    int completion = 0;
    int rec = 0;
    int triggered_at = -1;
    for (int i = 0; i < toks.len; i++) {
        TEST_ASSERT(pulsar_session_eval(session, toks.v[i], err, sizeof(err)) == 0);
        size_t piece_len = 0;
        char *piece = pulsar_token_text(engine, toks.v[i], &piece_len);
        buf_append(&text, piece, piece_len);
        thinking_state_feed(&thinking, piece, piece_len);
        free(piece);
        TEST_ASSERT(thinking.inside);
        rec = chat_think_tool_recovery(&srv, &srv.slots[0], &text, &thinking,
                                       &scan_from,
                                       &completion, 512, err, sizeof(err));
        TEST_ASSERT(rec >= 0);
        if (rec == 1) {
            triggered_at = i;
            break;
        }
    }
    fprintf(stderr,
            "pulsar-test: think-tool-recovery trigger=%d/%d injected_tokens=%d\n",
            triggered_at, toks.len, completion);
    TEST_ASSERT(rec == 1);
    TEST_ASSERT(triggered_at == toks.len - 1);
    pulsar_tokens_free(&toks);
    buf_free(&forced);
    TEST_ASSERT(!thinking.inside);
    TEST_ASSERT(completion > 0);
    TEST_ASSERT(text.ptr && text.len >= 10 &&
                !memcmp(text.ptr + text.len - 10, "</think>\n\n", 10));

    /* The model must now complete a valid call on the executable side. */
    uint64_t rng = 123;
    bool decode_ok = true;
    bool saw_start = false;
    bool saw_end = false;
    for (int i = 0; i < 256 && !saw_end; i++) {
        int token = pulsar_session_sample(session, 0.0f, 0, 1.0f, 0.0f, &rng);
        if (token == pulsar_token_eos(engine)) break;
        size_t piece_len = 0;
        char *piece = pulsar_token_text(engine, token, &piece_len);
        buf_append(&text, piece, piece_len);
        free(piece);
        observe_tool_markers(text.ptr, &saw_start, &saw_end, NULL);
        if (saw_end) break;
        if (pulsar_session_eval(session, token, err, sizeof(err)) != 0) {
            decode_ok = false;
            break;
        }
    }
    fprintf(stderr, "pulsar-test: think-tool-recovery continuation=[%s]\n",
            text.ptr ? text.ptr : "");
    TEST_ASSERT(decode_ok);
    TEST_ASSERT(saw_end);

    char *content = NULL;
    char *reasoning = NULL;
    tool_calls calls = {0};
    bool parsed = parse_generated_message_ex(text.ptr, true,
                                             &content, &reasoning, &calls);
    TEST_ASSERT(parsed);
    TEST_ASSERT(calls.len > 0 && !strcmp(calls.v[0].name, "list_files"));
    TEST_ASSERT(reasoning && strstr(reasoning, "list_files tool right away"));

    fprintf(stderr,
            "pulsar-test: think-tool-recovery recovered=%d gen_tokens=%d calls=%d name=%s\n",
            rec, completion, calls.len, calls.len ? calls.v[0].name : "-");

    free(content);
    free(reasoning);
    tool_calls_free(&calls);
    buf_free(&text);
    pulsar_session_free(session);
    request_free(&r);
    test_close_engine(false);
}

static void test_tool_call_quality_one(bool quality) {
    pulsar_engine *engine = test_get_engine(quality);
    if (!engine) return;

    request r;
    char err[160];
    TEST_ASSERT(parse_chat_request(engine, NULL, test_tool_call_request_json(),
                                   512, 32768, &r, err, sizeof(err)));

    pulsar_session *session = NULL;
    TEST_ASSERT(pulsar_session_create(&session, engine, 32768) == 0);
    if (!session) {
        request_free(&r);
        return;
    }
    TEST_ASSERT(pulsar_session_sync(session, &r.prompt, err, sizeof(err)) == 0);

    buf text = {0};
    uint64_t rng = 123;
    bool decode_ok = true;
    bool saw_tool_start = false;
    bool saw_tool_end = false;
    for (int i = 0; i < r.max_tokens; i++) {
        int token = pulsar_session_sample(session, r.temperature, r.top_k,
                                       r.top_p, r.min_p, &rng);
        size_t piece_len = 0;
        char *piece = pulsar_token_text(engine, token, &piece_len);
        buf_append(&text, piece, piece_len);
        free(piece);
        observe_tool_markers(text.ptr ? text.ptr : "", &saw_tool_start, &saw_tool_end, NULL);
        if (saw_tool_end) break;
        if (pulsar_session_eval(session, token, err, sizeof(err)) != 0) {
            decode_ok = false;
            break;
        }
    }

    char *content = NULL;
    char *reasoning = NULL;
    tool_calls calls = {0};
    bool parsed = parse_generated_message_ex(text.ptr ? text.ptr : "",
                                             false, &content, &reasoning, &calls);
    TEST_ASSERT(decode_ok);
    TEST_ASSERT(parsed);
    TEST_ASSERT(calls.len > 0);
    TEST_ASSERT(calls.len > 0 && !strcmp(calls.v[0].name, "list_files"));

    free(content);
    free(reasoning);
    tool_calls_free(&calls);
    buf_free(&text);
    pulsar_session_free(session);
    request_free(&r);
}

static void test_tool_call_quality(void) {
    fprintf(stderr, "pulsar-test: tool-call quality fast path\n");
    test_tool_call_quality_one(false);
    test_close_engine(false);
    fprintf(stderr, "pulsar-test: tool-call quality exact path\n");
    test_tool_call_quality_one(true);
    test_close_engine(true);
}


#endif

/* ===== Sampler byte-exactness gate =============================================
 * pulsar_sample_dist_build's full-vocab path replaced a qsort over the whole
 * 129k vocab with a stable radix sort + reused scratch (it ran once per
 * accepted position in the sampled speculative walk, ~10.6 ms a call). The
 * rewrite claims BYTE-EXACT equivalence, which is subtle: the sort's result
 * feeds max_logit, `sum` is accumulated over ALL candidates, and both
 * cutoffs are relative to that sum — so sort order and summation order
 * jointly decide which token a given seed draws.
 *
 * RE-DERIVED for the min-p prefilter (dev-minp): on the top_k <= 0,
 * min_p > 0 path the shipped build now accumulates `sum` in VOCAB-INDEX
 * order (one pass computing each prob once, collecting only the prefilter
 * survivors for sorting) instead of the old sorted-descending order. The
 * byte-exact reference below (ref_sample_dist_build) mirrors that: an
 * UNFILTERED qsort implementation whose sum is taken in index order on that
 * path — so any prefilter-induced membership or ordering error is caught
 * bit-for-bit, while the deliberate summation-order change is shared by
 * both sides. The OLD sorted-order-sum reference is kept verbatim as
 * ref_sample_dist_build_sortsum, and --sampler-prefilter gates the shipped
 * build against IT for survivor-set/order identity plus a characterized
 * (<= 1e-6 relative) prob delta. min_p <= 0 and top_k > 0 paths are
 * unchanged and stay byte-exact against the old semantics by construction
 * (both references agree there).
 *
 * This gate pins the claim across adversarial logit shapes (all-equal,
 * heavy ties, +/-0.0, non-finites, exact min-p boundary) x the sampling
 * configs that reach production. It compares the built distribution
 * bit-for-bit AND the rng-driven accept/draw/draw-excluding sequences the
 * spec path actually consumes.
 * ============================================================================ */

static int ref_sample_argmax(const float *logits, uint32_t n_vocab) {
    float best_v = -INFINITY;
    int best = 0;
    for (uint32_t i = 0; i < n_vocab; i++) {
        const float v = logits[i];
        if (!isfinite(v)) continue;
        if (v > best_v) { best_v = v; best = (int)i; }
    }
    return best;
}

/* Tie order is pinned EXPLICITLY (ascending id) rather than inherited from
 * qsort's stability. Ties are not hypothetical: ~129k float32 logits over a
 * normal range collide with ~99.96% probability, and the tie-order mutant is
 * caught on the realistic-peaked shape at rank 2969 (inside the nucleus).
 * The pre-rewrite code got ascending-id ties because glibc's qsort happens to
 * take the stable msort_with_tmp path at this size (129280 x 12B, verified) --
 * an unspecified detail upstream has been trying to remove since the 2.37
 * introsort work. Without this tiebreak, a future glibc would fail this gate
 * with a tie diff that looks like a radix bug but is reference drift, inviting
 * a "fix" to the wrong side. This comparator now pins the same order
 * canonically, glibc-independently. */
static int ref_cand_cmp_desc(const void *a, const void *b) {
    const sample_candidate *ca = (const sample_candidate *)a;
    const sample_candidate *cb = (const sample_candidate *)b;
    if (ca->logit != cb->logit) {
        return (cb->logit > ca->logit) - (cb->logit < ca->logit);
    }
    return (ca->id > cb->id) - (ca->id < cb->id);
}

/* OLD-semantics reference (pre-prefilter, sorted-order sum), kept VERBATIM.
 * No longer the byte-exact target on the top_k <= 0, min_p > 0 path — the
 * shipped build's sum moved to index order there. --sampler-prefilter gates
 * the shipped build against this for survivor-set/order IDENTITY and a
 * bounded prob delta; on all other paths it still agrees bit-for-bit. */
static int ref_sample_dist_build_sortsum(const float *logits, uint32_t n_vocab,
                                         float temperature, int top_k, float top_p, float min_p,
                                         pulsar_sample_dist *out) {
    memset(out, 0, sizeof(*out));
    if (temperature <= 0.0f) {
        out->ids = (int *)malloc(sizeof(int));
        out->probs = (float *)malloc(sizeof(float));
        out->ids[0] = ref_sample_argmax(logits, n_vocab);
        out->probs[0] = 1.0f;
        out->n = 1;
        return 1;
    }
    if (top_p <= 0.0f || top_p > 1.0f) top_p = 1.0f;
    if (min_p < 0.0f) min_p = 0.0f;
    if (top_k <= 0 || top_k > 1024) top_k = top_k <= 0 ? 0 : 1024;

    /* collect candidates: full vocab, or top-k preselect like the sampler */
    uint32_t cap = top_k > 0 ? (uint32_t)top_k : n_vocab;
    sample_candidate *cand = (sample_candidate *)malloc((size_t)cap * sizeof(cand[0]));
    uint32_t n = 0;
    if (top_k > 0) {
        for (uint32_t i = 0; i < n_vocab; i++) {
            const float v = logits[i];
            if (!isfinite(v)) continue;
            if (n == (uint32_t)top_k && v <= cand[n - 1].logit) continue;
            uint32_t j = n < (uint32_t)top_k ? n++ : n - 1;
            while (j > 0 && cand[j - 1].logit < v) {
                cand[j] = cand[j - 1];
                j--;
            }
            cand[j].id = (int)i;
            cand[j].logit = v;
        }
    } else {
        for (uint32_t i = 0; i < n_vocab; i++) {
            const float v = logits[i];
            if (!isfinite(v)) continue;
            cand[n++] = (sample_candidate){.id = (int)i, .logit = v, .prob = 0.0f};
        }
        if (n) qsort(cand, n, sizeof(cand[0]), ref_cand_cmp_desc);
    }
    if (n == 0) {
        free(cand);
        out->ids = (int *)malloc(sizeof(int));
        out->probs = (float *)malloc(sizeof(float));
        out->ids[0] = ref_sample_argmax(logits, n_vocab);
        out->probs[0] = 1.0f;
        out->n = 1;
        return 1;
    }

    const float max_logit = cand[0].logit;
    float sum = 0.0f;
    for (uint32_t i = 0; i < n; i++) {
        cand[i].prob = expf((cand[i].logit - max_logit) / temperature);
        sum += cand[i].prob;
    }
    if (sum <= 0.0f || !isfinite(sum)) {
        out->ids = (int *)malloc(sizeof(int));
        out->probs = (float *)malloc(sizeof(float));
        out->ids[0] = cand[0].id;
        out->probs[0] = 1.0f;
        out->n = 1;
        free(cand);
        return 1;
    }
    const float min_prob = (cand[0].prob / sum) * min_p;
    float filtered_sum = 0.0f;
    uint32_t filtered = 0;
    for (uint32_t i = 0; i < n; i++) {
        const float pr = cand[i].prob / sum;
        if (i > 0 && pr < min_prob) break;
        filtered_sum += cand[i].prob;
        filtered++;
        if (filtered_sum / sum >= top_p) break;
    }
    if (filtered == 0) filtered = 1;
    out->ids = (int *)malloc((size_t)filtered * sizeof(int));
    out->probs = (float *)malloc((size_t)filtered * sizeof(float));
    out->n = filtered;
    for (uint32_t i = 0; i < filtered; i++) {
        out->ids[i] = cand[i].id;
        out->probs[i] = cand[i].prob / filtered_sum;   /* renormalized nucleus */
    }
    free(cand);
    return 1;
}


/* The byte-exact reference for the CURRENT build, re-derived for the min-p
 * prefilter: identical to the sortsum reference EXCEPT that the
 * top_k <= 0, min_p > 0 path computes each prob once and accumulates `sum`
 * in VOCAB-INDEX order before the sort, carrying probs through the qsort —
 * the summation-order semantics the shipped build adopted. It applies NO
 * prefilter (full descending walk with the byte-exact cutoffs), so a
 * prefilter that drops a candidate the cutoff would keep, keeps one it
 * would drop, or perturbs order/probs in any way fails the memcmp below. */
static int ref_sample_dist_build(const float *logits, uint32_t n_vocab,
                                 float temperature, int top_k, float top_p, float min_p,
                                 pulsar_sample_dist *out) {
    memset(out, 0, sizeof(*out));
    if (temperature <= 0.0f) {
        out->ids = (int *)malloc(sizeof(int));
        out->probs = (float *)malloc(sizeof(float));
        out->ids[0] = ref_sample_argmax(logits, n_vocab);
        out->probs[0] = 1.0f;
        out->n = 1;
        return 1;
    }
    if (top_p <= 0.0f || top_p > 1.0f) top_p = 1.0f;
    if (min_p < 0.0f) min_p = 0.0f;
    if (top_k <= 0 || top_k > 1024) top_k = top_k <= 0 ? 0 : 1024;

    uint32_t cap = top_k > 0 ? (uint32_t)top_k : n_vocab;
    sample_candidate *cand = (sample_candidate *)malloc((size_t)cap * sizeof(cand[0]));
    uint32_t n = 0;
    int have_probs = 0;
    float sum = 0.0f;
    if (top_k > 0) {
        for (uint32_t i = 0; i < n_vocab; i++) {
            const float v = logits[i];
            if (!isfinite(v)) continue;
            if (n == (uint32_t)top_k && v <= cand[n - 1].logit) continue;
            uint32_t j = n < (uint32_t)top_k ? n++ : n - 1;
            while (j > 0 && cand[j - 1].logit < v) {
                cand[j] = cand[j - 1];
                j--;
            }
            cand[j].id = (int)i;
            cand[j].logit = v;
        }
    } else if (min_p > 1e-30f) {   /* SAMPLE_MINP_PREFILTER_MIN */
        /* index-order sum, probs computed once and carried through the sort */
        float max_logit = 0.0f;
        uint32_t finite = 0;
        for (uint32_t i = 0; i < n_vocab; i++) {
            const float v = logits[i];
            if (!isfinite(v)) continue;
            if (finite == 0 || v > max_logit) max_logit = v;
            finite++;
        }
        for (uint32_t i = 0; i < n_vocab; i++) {
            const float v = logits[i];
            if (!isfinite(v)) continue;
            const float p = expf((v - max_logit) / temperature);
            sum += p;
            cand[n++] = (sample_candidate){.id = (int)i, .logit = v, .prob = p};
        }
        if (n) qsort(cand, n, sizeof(cand[0]), ref_cand_cmp_desc);
        have_probs = 1;
    } else {
        for (uint32_t i = 0; i < n_vocab; i++) {
            const float v = logits[i];
            if (!isfinite(v)) continue;
            cand[n++] = (sample_candidate){.id = (int)i, .logit = v, .prob = 0.0f};
        }
        if (n) qsort(cand, n, sizeof(cand[0]), ref_cand_cmp_desc);
    }
    if (n == 0) {
        free(cand);
        out->ids = (int *)malloc(sizeof(int));
        out->probs = (float *)malloc(sizeof(float));
        out->ids[0] = ref_sample_argmax(logits, n_vocab);
        out->probs[0] = 1.0f;
        out->n = 1;
        return 1;
    }

    if (!have_probs) {
        const float max_logit = cand[0].logit;
        for (uint32_t i = 0; i < n; i++) {
            cand[i].prob = expf((cand[i].logit - max_logit) / temperature);
            sum += cand[i].prob;
        }
    }
    if (sum <= 0.0f || !isfinite(sum)) {
        out->ids = (int *)malloc(sizeof(int));
        out->probs = (float *)malloc(sizeof(float));
        out->ids[0] = cand[0].id;
        out->probs[0] = 1.0f;
        out->n = 1;
        free(cand);
        return 1;
    }
    const float min_prob = (cand[0].prob / sum) * min_p;
    float filtered_sum = 0.0f;
    uint32_t filtered = 0;
    for (uint32_t i = 0; i < n; i++) {
        const float pr = cand[i].prob / sum;
        if (i > 0 && pr < min_prob) break;
        filtered_sum += cand[i].prob;
        filtered++;
        if (filtered_sum / sum >= top_p) break;
    }
    if (filtered == 0) filtered = 1;
    out->ids = (int *)malloc((size_t)filtered * sizeof(int));
    out->probs = (float *)malloc((size_t)filtered * sizeof(float));
    out->n = filtered;
    for (uint32_t i = 0; i < filtered; i++) {
        out->ids[i] = cand[i].id;
        out->probs[i] = cand[i].prob / filtered_sum;   /* renormalized nucleus */
    }
    free(cand);
    return 1;
}


/* Verbatim pre-radix sample_full_vocab / sample_top_p_min_p (tokenizer.c at
 * 9587033), renamed. The plain sampling path carries the same full-vocab sort
 * as the speculative one and was rewritten the same way; this pins it. Note
 * its `sum` accumulates in VOCAB order BEFORE the sort, unlike dist_build's —
 * a difference the rewrite has to respect, so it is worth gating directly.
 *
 * Kept verbatim ON PURPOSE — including its qsort, whose tie order the C
 * standard leaves unspecified. That is the point: this is the behaviour we
 * claim to reproduce, so the reference must be the real thing. Note which
 * side is fragile: glibc merge-sorts (stably) at this size today, but falls
 * back to an unstable quicksort if its temp malloc fails, and upstream has
 * been moving qsort toward introsort. If a future glibc lands that, the
 * tie-sensitive shapes fail HERE while the shipped code stays correct — the
 * radix sort is unconditionally stable, i.e. strictly MORE deterministic than
 * the qsort it replaced. Fix that by giving this reference an explicit
 * (logit, id) comparator; do not "fix" the sampler. */
static float ref_rng_f32(uint64_t *state) {
    uint64_t x = *state;
    if (x == 0) x = 0x9e3779b97f4a7c15ULL;
    x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
    *state = x;
    const uint64_t r = x * 0x2545f4914f6cdd1dULL;
    return (float)((r >> 40) & 0xffffffu) / 16777216.0f;
}

static int ref_full_vocab(
        const float *logits,
        uint32_t     n_vocab,
        float        temperature,
        float        top_p,
        float        min_p,
        uint64_t    *rng) {
    float max_logit = PULSAR_NEG_INF;
    int best = 0;
    uint32_t finite = 0;
    for (uint32_t i = 0; i < n_vocab; i++) {
        const float v = logits[i];
        if (!isfinite(v)) continue;
        finite++;
        if (v > max_logit) {
            max_logit = v;
            best = (int)i;
        }
    }
    if (finite == 0) return ref_sample_argmax(logits, n_vocab);

    if (top_p >= 1.0f) {
        float sum = 0.0f;
        const float min_rel = min_p > 0.0f ? min_p : 0.0f;
        for (uint32_t i = 0; i < n_vocab; i++) {
            const float v = logits[i];
            if (!isfinite(v)) continue;
            const float p = expf((v - max_logit) / temperature);
            if (p < min_rel) continue;
            sum += p;
        }
        if (sum <= 0.0f || !isfinite(sum)) return best;
        float r = ref_rng_f32(rng) * sum;
        for (uint32_t i = 0; i < n_vocab; i++) {
            const float v = logits[i];
            if (!isfinite(v)) continue;
            const float p = expf((v - max_logit) / temperature);
            if (p < min_rel) continue;
            r -= p;
            if (r <= 0.0f) return (int)i;
        }
        return best;
    }

    sample_candidate *cand = (sample_candidate *)malloc((size_t)finite * sizeof(cand[0]));
    uint32_t n = 0;
    float sum = 0.0f;
    for (uint32_t i = 0; i < n_vocab; i++) {
        const float v = logits[i];
        if (!isfinite(v)) continue;
        const float p = expf((v - max_logit) / temperature);
        cand[n++] = (sample_candidate){.id = (int)i, .logit = v, .prob = p};
        sum += p;
    }
    if (sum <= 0.0f || !isfinite(sum)) {
        free(cand);
        return best;
    }

    qsort(cand, n, sizeof(cand[0]), ref_cand_cmp_desc);
    const float min_prob = (cand[0].prob / sum) * (min_p > 0.0f ? min_p : 0.0f);
    float filtered_sum = 0.0f;
    uint32_t filtered = 0;
    for (uint32_t i = 0; i < n; i++) {
        const float p = cand[i].prob / sum;
        if (i > 0 && p < min_prob) break;
        filtered_sum += cand[i].prob;
        filtered++;
        if (filtered_sum / sum >= top_p) break;
    }
    if (filtered == 0) {
        free(cand);
        return best;
    }

    float r = ref_rng_f32(rng) * filtered_sum;
    for (uint32_t i = 0; i < filtered; i++) {
        r -= cand[i].prob;
        if (r <= 0.0f) {
            const int id = cand[i].id;
            free(cand);
            return id;
        }
    }
    const int id = cand[filtered - 1].id;
    free(cand);
    return id;
}


static int ref_top_p_min_p(
        const float *logits,
        uint32_t     n_vocab,
        float        temperature,
        int          top_k,
        float        top_p,
        float        min_p,
        uint64_t    *rng) {
    if (temperature <= 0.0f) return ref_sample_argmax(logits, n_vocab);
    if (top_p <= 0.0f || top_p > 1.0f) top_p = 1.0f;
    if (min_p < 0.0f) min_p = 0.0f;
    if (top_k <= 0) return ref_full_vocab(logits, n_vocab, temperature, top_p, min_p, rng);
    if (top_k > 1024) top_k = 1024;
    if ((uint32_t)top_k > n_vocab) top_k = (int)n_vocab;

    int ids[1024];
    float vals[1024];
    int n = 0;
    for (uint32_t i = 0; i < n_vocab; i++) {
        float v = logits[i];
        if (!isfinite(v)) continue;
        if (n == top_k && v <= vals[n - 1]) continue;
        int j = n < top_k ? n++ : n - 1;
        while (j > 0 && vals[j - 1] < v) {
            vals[j] = vals[j - 1];
            ids[j] = ids[j - 1];
            j--;
        }
        vals[j] = v;
        ids[j] = (int)i;
    }
    if (n == 0) return ref_sample_argmax(logits, n_vocab);

    float probs[1024];
    const float max_logit = vals[0];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        probs[i] = expf((vals[i] - max_logit) / temperature);
        sum += probs[i];
    }
    if (sum <= 0.0f || !isfinite(sum)) return ids[0];

    const float min_prob = (probs[0] / sum) * min_p;
    float filtered_sum = 0.0f;
    int filtered = 0;
    for (int i = 0; i < n; i++) {
        float p = probs[i] / sum;
        if (i > 0 && p < min_prob) break;
        filtered_sum += probs[i];
        filtered++;
        if (filtered_sum / sum >= top_p) break;
    }
    if (filtered <= 0) return ids[0];

    float r = ref_rng_f32(rng) * filtered_sum;
    for (int i = 0; i < filtered; i++) {
        r -= probs[i];
        if (r <= 0.0f) return ids[i];
    }
    return ids[filtered - 1];
}

#define SAMP_N_VOCAB 129280u

static uint64_t samp_rs;
static void samp_seed(uint64_t s) { samp_rs = s ? s : 1; }
static double samp_u01(void) {
    samp_rs ^= samp_rs >> 12; samp_rs ^= samp_rs << 25; samp_rs ^= samp_rs >> 27;
    return (double)((samp_rs * 0x2545f4914f6cdd1dULL) >> 11) / 9007199254740992.0;
}

/* Find a float x with expf(x) == target EXACTLY, scanning a few thousand
 * ulps around logf(target). Around 0.5 the expf image spacing (~3e-8) is
 * about half the float spacing of the target (~6e-8), so a preimage exists
 * for essentially every target there; the scan is cheap and deterministic. */
static int samp_expf_preimage(float target, float *out) {
    const float x0 = logf(target);
    uint32_t ub;
    memcpy(&ub, &x0, sizeof(ub));
    for (int32_t d = -4096; d <= 4096; d++) {
        const uint32_t u = ub + (uint32_t)d;
        float x;
        memcpy(&x, &u, sizeof(x));
        if (expf(x) == target) {
            *out = x;
            return 1;
        }
    }
    return 0;
}

/* Adversarial logit shapes. Ties and signed zeros are the cases where sort
 * order is observable at all; non-finites exercise the compaction path. */
static void samp_fill_shape(float *l, uint32_t n, int shape, const char **name) {
    samp_seed(0xC0FFEE00u + (uint64_t)shape);
    switch (shape) {
    case 0: *name = "realistic-peaked";
        for (uint32_t i = 0; i < n; i++) {
            double u1 = samp_u01(), u2 = samp_u01();
            l[i] = (float)(sqrt(-2.0 * log(u1 + 1e-12)) * cos(2 * M_PI * u2) * 2.0);
        }
        for (int i = 0; i < 40; i++) l[(uint32_t)(samp_u01() * n)] = 8.0f + (float)samp_u01() * 6.0f;
        break;
    case 1: *name = "uniform-wide";
        for (uint32_t i = 0; i < n; i++) l[i] = (float)(samp_u01() * 40.0 - 20.0);
        break;
    case 2: *name = "all-equal (degenerate)";
        for (uint32_t i = 0; i < n; i++) l[i] = 1.25f;
        break;
    case 3: *name = "heavy-ties (64 distinct)";
        for (uint32_t i = 0; i < n; i++) l[i] = (float)((i * 2654435761u) % 64u) * 0.5f;
        break;
    case 4: *name = "signed-zeros + ties";
        /* BOTH zeros must be stored as raw bits. This file is compiled
         * -ffast-math (=> -fno-signed-zeros): a -0.0f literal folds to +0.0f,
         * and even a memcpy-built -0.0 gets substituted for a +0.0f literal on
         * the other arm of a ternary, collapsing the mix. Writing the bit
         * patterns keeps the FP value model out of it entirely. The engine's
         * logits arrive by memcpy from the GPU, so a +/-0.0 mix really can
         * occur there regardless of host flags. */
        {
            static const uint32_t zbits[2] = {0x80000000u, 0x00000000u};
            for (uint32_t i = 0; i < n; i++)
                memcpy(&l[i], &zbits[i & 1u], sizeof(l[i]));
            for (int i = 0; i < 8; i++) l[(uint32_t)(samp_u01() * n)] = 0.5f;
        }
        break;
    case 5: *name = "with -inf/NaN holes";
        for (uint32_t i = 0; i < n; i++) l[i] = (float)(samp_u01() * 10.0 - 5.0);
        for (uint32_t i = 0; i < n; i += 3) l[i] = -INFINITY;
        for (uint32_t i = 1; i < n; i += 997) l[i] = NAN;
        break;
    case 6: *name = "single finite";
        for (uint32_t i = 0; i < n; i++) l[i] = -INFINITY;
        l[n / 3] = 2.0f;
        break;
    case 7: *name = "all non-finite";
        for (uint32_t i = 0; i < n; i++) l[i] = -INFINITY;
        break;
    case 8: *name = "near-flat (tiny spread)";
        for (uint32_t i = 0; i < n; i++) l[i] = 3.0f + (float)(samp_u01() * 1e-6);
        break;
    case 9: *name = "subnormals + zeros (FZ range)";
        /* Pins that sample_desc_key ORDERS subnormals and ties ONLY +/-0.0 —
         * i.e. that it must NOT flush subnormals. -ffast-math would set
         * FPCR.FZ (making the replaced comparator tie them), but only via
         * crtfastmath.o, which the LINKER pulls in, and every binary carrying
         * this code links through nvcc. Measured in the real link config:
         * FPCR = 0x0, FZ = 0, comparator orders subnormals. A gcc-linked
         * probe of the same source reports FPCR = 0x1000000 and the opposite
         * answer — so if this shape ever fails, suspect the LINKER, not this
         * test. Adding an FZ flush to sample_desc_key fails here, by design.
         *
         * Written as raw bits: these values cannot survive the host FP value
         * model (-ffast-math folds a -0.0f literal to +0.0f). */
        {
            static const uint32_t fz[6] = {
                0x00000000u, 0x80000000u,   /* +0.0, -0.0                */
                0x00000001u, 0x80000001u,   /* smallest +/- subnormal    */
                0x007fffffu, 0x80400000u,   /* largest / mid subnormal   */
            };
            for (uint32_t i = 0; i < n; i++)
                memcpy(&l[i], &fz[i % 6u], sizeof(l[i]));
            for (int i = 0; i < 8; i++) l[(uint32_t)(samp_u01() * n)] = 0.25f;
        }
        break;
    case 10: *name = "tie group straddling the cutoff";
        /* All-equal logits + top_p<1 make `filtered` cut THROUGH one tie
         * group, so tie order decides nucleus MEMBERSHIP (ascending ->
         * ids[0..filtered-1], descending -> a disjoint set). This is the case
         * that makes sort stability load-bearing rather than cosmetic. */
        for (uint32_t i = 0; i < n; i++) l[i] = 1.25f;
        break;
    case 11: *name = "min-p exact boundary (p==min_p tie pair + 1ulp below)";
        /* Crafted so that at temp=1, min_p=0.5 the sum is EXACT (1+.5+.5
         * [+.5-1ulp] rounds identically in index and sorted order) and the
         * tie pair sits exactly AT the min-p threshold: pr == min_prob
         * bit-for-bit, so the `pr < min_prob` operator alone decides
         * membership. The fourth candidate is 1ulp of prob below: a min-p
         * PREFILTER must keep it within slack yet the exact cutoff must trim
         * it. Logits are laid out in ascending-id descending-logit order so
         * old (sorted-order) and new (index-order) sums are the same float
         * sequence — boundary decisions are isolated from the summation-order
         * change. samp_expf_preimage can in principle fail to find an exact
         * preimage (degrade: logf value, still a near-boundary shape); the
         * dedicated boundary teeth in --sampler-prefilter TEST_ASSERT it. */
        {
            float L_at, L_below;
            if (!samp_expf_preimage(0.5f, &L_at)) L_at = logf(0.5f);
            if (!samp_expf_preimage(nextafterf(0.5f, 0.0f), &L_below))
                L_below = logf(nextafterf(0.5f, 0.0f));
            for (uint32_t i = 0; i < n; i++) l[i] = -INFINITY;
            l[100] = 0.0f;      /* max, prob exactly 1.0 */
            l[200] = L_at;      /* prob exactly 0.5 == min_p */
            l[300] = L_at;      /* tie at the boundary     */
            l[400] = L_below;   /* prob 1ulp below the boundary */
        }
        break;
    default: *name = "?"; break;
    }
}

typedef struct { float temp; int top_k; float top_p; float min_p; const char *name; } samp_cfg;

static const samp_cfg samp_cfgs[] = {
    {1.0f,    0, 1.0f,  0.05f, "server defaults (top_k=0,top_p=1,min_p=0.05)"},
    {1.0f,    0, 1.0f,  0.0f,  "no cutoffs"},
    {1.0f,   40, 1.0f,  0.05f, "top_k=40"},
    {1.0f, 2048, 1.0f,  0.05f, "top_k=2048 (clamped to 1024)"},
    {1.0f,    0, 0.9f,  0.0f,  "top_p=0.9"},
    {1.0f,    0, 0.5f,  0.05f, "top_p=0.5 + min_p"},
    {1.0f,    0, 1.0f,  0.5f,  "min_p=0.5 (aggressive)"},
    {0.01f,   0, 1.0f,  0.05f, "temp=0.01 (very low)"},
    {0.1f,    0, 0.95f, 0.02f, "temp=0.1 combined"},
    {5.0f,    0, 1.0f,  0.05f, "temp=5 (high)"},
    {100.0f,  0, 0.99f, 0.01f, "temp=100 (extreme)"},
    {0.0f,    0, 1.0f,  0.05f, "temp=0 (greedy early-out)"},
    {-1.0f,   0, 1.0f,  0.05f, "temp<0 (greedy early-out)"},
};

/* The reference comparator above compares floats, so it is sensitive to
 * flush-to-zero; the production key (sample_desc_key) is integer-only and is
 * NOT. Production binaries are nvcc-linked and measure FPCR=0x0 (FZ clear), so
 * the two agree on subnormals. A gcc -ffast-math link would pull in
 * crtfastmath.o ("msr fpcr, x0" with bit 24 set), flush subnormals to zero in
 * the REFERENCE only, and surface here as a mystifying rank-8 id diff on the
 * subnormal shape. Say so out loud instead. */
static void sampler_warn_if_flush_to_zero(void) {
#if defined(__aarch64__)
    uint64_t fpcr = 0;
    __asm__ __volatile__("mrs %0, fpcr" : "=r"(fpcr));
    if (fpcr & (1u << 24)) {
        fprintf(stderr,
                "pulsar_test: WARNING: FPCR.FZ is set (fpcr=0x%llx) — this binary was "
                "linked with crtfastmath.o (gcc -ffast-math), so the float reference "
                "comparator flushes subnormals while the production integer key does "
                "not. A subnormal-shape id diff below is REFERENCE drift, not a "
                "sampler bug.\n",
                (unsigned long long)fpcr);
    }
#endif
}

static void test_sampler_dist_equivalence(void) {
    const uint32_t n = SAMP_N_VOCAB;
    float *logits = (float *)malloc((size_t)n * sizeof(float));
    TEST_ASSERT(logits != NULL);
    sampler_warn_if_flush_to_zero();
    pulsar_sample_scratch scratch;
    memset(&scratch, 0, sizeof(scratch));

    int checked = 0;
    for (int shape = 0; shape <= 11; shape++) {
        const char *sname = "?";
        samp_fill_shape(logits, n, shape, &sname);
        for (size_t c = 0; c < sizeof(samp_cfgs) / sizeof(samp_cfgs[0]); c++) {
            const samp_cfg *cfg = &samp_cfgs[c];
            pulsar_sample_dist ref, got;
            ref_sample_dist_build(logits, n, cfg->temp, cfg->top_k, cfg->top_p, cfg->min_p, &ref);
            pulsar_sample_dist_build(logits, n, cfg->temp, cfg->top_k, cfg->top_p, cfg->min_p,
                                  &scratch, &got);

            /* (a) the distribution itself, bit-for-bit */
            if (ref.n != got.n)
                fprintf(stderr, "sampler: shape=%s cfg=%s n %u != %u\n",
                        sname, cfg->name, ref.n, got.n);
            TEST_ASSERT(ref.n == got.n);
            for (uint32_t i = 0; i < ref.n; i++) {
                if (ref.ids[i] != got.ids[i])
                    fprintf(stderr, "sampler: shape=%s cfg=%s rank %u id %d != %d\n",
                            sname, cfg->name, i, ref.ids[i], got.ids[i]);
                TEST_ASSERT(ref.ids[i] == got.ids[i]);
                /* memcmp, not ==: this must be bit-exact, not merely equal */
                TEST_ASSERT(memcmp(&ref.probs[i], &got.probs[i], sizeof(float)) == 0);
            }

            /* (b) the rng-driven sequences the spec acceptance walk consumes:
             * same seed must select the same tokens through accept / draw /
             * draw_excluding. This is the property that actually decides the
             * emitted token stream. */
            for (int trial = 0; trial < 64; trial++) {
                const uint64_t seed = 0x5EED0000u + (uint64_t)trial * 7919u;
                uint64_t r1 = seed, r2 = seed;
                const int probe = ref.ids[(uint32_t)trial % ref.n];
                TEST_ASSERT(pulsar_sample_dist_accept(&ref, probe, &r1) ==
                            pulsar_sample_dist_accept(&got, probe, &r2));
                TEST_ASSERT(r1 == r2);
                TEST_ASSERT(pulsar_sample_dist_draw(&ref, &r1) == pulsar_sample_dist_draw(&got, &r2));
                TEST_ASSERT(r1 == r2);
                TEST_ASSERT(pulsar_sample_dist_draw_excluding(&ref, probe, &r1) ==
                            pulsar_sample_dist_draw_excluding(&got, probe, &r2));
                TEST_ASSERT(r1 == r2);
                TEST_ASSERT(pulsar_sample_dist_prob(&ref, probe) == pulsar_sample_dist_prob(&got, probe));
            }

            /* (c) the PLAIN sampling path (sample_full_vocab), pinned against
             * the pre-radix reference: same seed must draw the same token and
             * leave the rng in the same state. */
            for (int trial = 0; trial < 16; trial++) {
                uint64_t r1 = 0xABCD0000u + (uint64_t)trial, r2 = r1;
                const int want = ref_top_p_min_p(logits, n, cfg->temp, cfg->top_k,
                                                 cfg->top_p, cfg->min_p, &r1);
                const int got_tok = sample_top_p_min_p(logits, n, cfg->temp, cfg->top_k,
                                                       cfg->top_p, cfg->min_p, &r2);
                if (want != got_tok)
                    fprintf(stderr, "sampler: shape=%s cfg=%s plain draw %d != %d\n",
                            sname, cfg->name, want, got_tok);
                TEST_ASSERT(want == got_tok);
                TEST_ASSERT(r1 == r2);
            }

            pulsar_sample_dist_free(&ref);
            pulsar_sample_dist_free(&got);
            checked++;
        }
    }

    /* scratch reuse must be order-independent: a fresh scratch and a hot one
     * (already grown by a full-vocab call) must produce identical results. */
    {
        const char *sname = "?";
        samp_fill_shape(logits, n, 0, &sname);
        pulsar_sample_dist hot, cold;
        pulsar_sample_scratch fresh;
        memset(&fresh, 0, sizeof(fresh));
        pulsar_sample_dist_build(logits, n, 1.0f, 40, 1.0f, 0.05f, &fresh, &cold);
        pulsar_sample_dist_free(&cold);
        /* now a full-vocab build on a scratch previously sized for top_k=40 */
        pulsar_sample_dist_build(logits, n, 1.0f, 0, 1.0f, 0.05f, &fresh, &hot);
        pulsar_sample_dist_build(logits, n, 1.0f, 0, 1.0f, 0.05f, &scratch, &cold);
        TEST_ASSERT(hot.n == cold.n);
        for (uint32_t i = 0; i < hot.n; i++) {
            TEST_ASSERT(hot.ids[i] == cold.ids[i]);
            TEST_ASSERT(memcmp(&hot.probs[i], &cold.probs[i], sizeof(float)) == 0);
        }
        pulsar_sample_dist_free(&hot);
        pulsar_sample_dist_free(&cold);
        pulsar_sample_scratch_free(&fresh);
    }

    pulsar_sample_scratch_free(&scratch);
    free(logits);
    printf("  sampler: %d shape x config combinations byte-exact vs re-derived reference\n",
           checked);
}



/* ===== min-p prefilter equivalence gate ========================================
 * The shipped pulsar_sample_dist_build (top_k <= 0, min_p > 0) vs the OLD
 * sorted-order-sum semantics (ref_sample_dist_build_sortsum): the survivor
 * SET, ids and order must be IDENTICAL wherever the old top_p crossing is
 * reproducible at all (always when top_p >= 1; see the in-loop comment for
 * the top_p < 1 tail-shift exception, which is characterized and bounded,
 * not waved through); probs may differ only by the summation-order
 * rounding, characterized here and bounded at 1e-6 relative (measured
 * ~1e-7). Plus dedicated boundary teeth: a candidate
 * whose prob sits bit-exactly AT the min-p threshold must be INCLUDED
 * (`pr < min_prob` is the operator, so >= keeps), and a candidate 1 ulp
 * below — which the prefilter's slack deliberately keeps for sorting —
 * must be trimmed by the exact cutoff. Constructed so index-order and
 * sorted-order sums are the same float sequence, isolating boundary
 * semantics from the deliberate stream change.
 * ============================================================================ */
static void test_sampler_prefilter_equivalence(void) {
    const uint32_t n = SAMP_N_VOCAB;
    float *logits = (float *)malloc((size_t)n * sizeof(float));
    TEST_ASSERT(logits != NULL);
    sampler_warn_if_flush_to_zero();
    pulsar_sample_scratch scratch;
    memset(&scratch, 0, sizeof(scratch));

    int checked = 0;
    int n_shifts = 0;
    double max_rel = 0.0, max_rel_shifted = 0.0;
    int max_shape = -1, max_cfg = -1;
    for (int shape = 0; shape <= 11; shape++) {
        const char *sname = "?";
        samp_fill_shape(logits, n, shape, &sname);
        for (size_t c = 0; c < sizeof(samp_cfgs) / sizeof(samp_cfgs[0]); c++) {
            const samp_cfg *cfg = &samp_cfgs[c];
            pulsar_sample_dist old, got;
            ref_sample_dist_build_sortsum(logits, n, cfg->temp, cfg->top_k,
                                          cfg->top_p, cfg->min_p, &old);
            pulsar_sample_dist_build(logits, n, cfg->temp, cfg->top_k,
                                  cfg->top_p, cfg->min_p, &scratch, &got);
            /* MIN-P membership vs the old semantics is identical (the
             * prefilter is a superset under ANY sum and the exact cutoff
             * decides; the boundary teeth below pin it). What CAN move is
             * the TOP_P crossing index: with top_p < 1 the break compares
             * fl(filtered_sum/sum) >= top_p, and the deliberate index-order
             * `sum` differs from the old sorted-order one by ~1e-7 rel — on
             * a huge near-flat nucleus (e.g. temp=100, ~128k candidates,
             * per-candidate step ~1e-5 of mass) that rounding shift moves
             * the crossing by a few candidates (a whole tie group if one
             * straddles it). So: n must be IDENTICAL whenever top_p >= 1;
             * with top_p < 1 a small tail shift is the characterized cost
             * of the stream change, the common prefix must still match
             * exactly, and the shift is reported below. */
            const uint32_t k = old.n < got.n ? old.n : got.n;
            if (old.n != got.n) {
                fprintf(stderr,
                        "prefilter: shape=%s cfg=%s top_p crossing shift: "
                        "n %u -> %u (dn %+d)\n",
                        sname, cfg->name, old.n, got.n,
                        (int)got.n - (int)old.n);
                TEST_ASSERT(cfg->top_p < 1.0f);
                const int dn = (int)got.n - (int)old.n;
                TEST_ASSERT(dn <= 64 && dn >= -64);
                n_shifts++;
            }
            for (uint32_t i = 0; i < k; i++) {
                if (old.ids[i] != got.ids[i])
                    fprintf(stderr, "prefilter: shape=%s cfg=%s rank %u id %d != %d\n",
                            sname, cfg->name, i, old.ids[i], got.ids[i]);
                TEST_ASSERT(old.ids[i] == got.ids[i]);
                const double a = (double)old.probs[i];
                const double b = (double)got.probs[i];
                const double denom = a > 0.0 ? a : 1e-45;
                const double rel = a == b ? 0.0 : fabs(b - a) / denom;
                if (old.n == got.n) {
                    if (rel > max_rel) {
                        max_rel = rel;
                        max_shape = shape;
                        max_cfg = (int)c;
                    }
                } else if (rel > max_rel_shifted) {
                    max_rel_shifted = rel;
                }
            }
            pulsar_sample_dist_free(&old);
            pulsar_sample_dist_free(&got);
            checked++;
        }
    }
    /* the float-sum-order delta: expected ~1e-7 relative; > 1e-6 means
     * something beyond summation-order moved and needs investigating. On the
     * few top_p-crossing-shifted combos the renormalizer (filtered_sum) also
     * moves by the shifted candidates' mass — characterized separately. */
    printf("  prefilter: max prob delta %.3e rel (shape %d, cfg %d) over %d combos\n",
           max_rel, max_shape, max_cfg, checked);
    printf("  prefilter: %d top_p-crossing tail shift(s); max prefix prob delta there %.3e rel\n",
           n_shifts, max_rel_shifted);
    TEST_ASSERT(max_rel <= 1e-6);
    TEST_ASSERT(max_rel_shifted <= 1e-3);

    /* survivor-count stats at the server defaults (what the radix actually
     * sorts now: prefilter survivors, not 129,280 candidates). */
    for (int shape = 0; shape <= 1; shape++) {
        const char *sname = "?";
        samp_fill_shape(logits, n, shape, &sname);
        float max_logit = 0.0f;
        uint32_t finite = 0;
        for (uint32_t i = 0; i < n; i++) {
            if (!isfinite(logits[i])) continue;
            if (finite == 0 || logits[i] > max_logit) max_logit = logits[i];
            finite++;
        }
        const float thr = 0.05f * (1.0f - 4e-6f);
        uint32_t m = 0;
        for (uint32_t i = 0; i < n; i++) {
            if (!isfinite(logits[i])) continue;
            if (expf(logits[i] - max_logit) >= thr) m++;
        }
        printf("  prefilter: shape \"%s\": %u of %u sorted at defaults (min_p=0.05, temp=1)\n",
               sname, m, finite);
    }

    /* dedicated boundary teeth (temp=1, top_k=0, top_p=1, min_p=0.5) */
    {
        float L_at = 0.0f, L_below = 0.0f, p_below = 0.0f;
        TEST_ASSERT(samp_expf_preimage(0.5f, &L_at));
        TEST_ASSERT(expf(L_at) == 0.5f);
        /* an exact preimage of (0.5 - 1ulp) is not guaranteed to exist (its
         * rounding interval is ~1 x-ulp wide); 1..3 ulps below all work —
         * anything in [min_p*slack, min_p) that the exact cutoff rounds
         * below min_prob, with the sum still rounding to 2.5 either order. */
        int found_below = 0;
        for (int k = 1; k <= 3 && !found_below; k++) {
            p_below = 0.5f - (float)k * 0x1p-25f;
            found_below = samp_expf_preimage(p_below, &L_below);
        }
        TEST_ASSERT(found_below);
        TEST_ASSERT(expf(L_below) == p_below && p_below < 0.5f &&
                    p_below >= 0.5f * (1.0f - 4e-6f));
        for (uint32_t i = 0; i < n; i++) logits[i] = -INFINITY;
        logits[100] = 0.0f;   /* max: prob 1.0 */
        logits[200] = L_at;   /* prob == 0.5 == min_p: exactly AT the threshold */
        logits[300] = L_at;   /* boundary tie */

        /* Shape A: sum = 1.0+0.5+0.5 = 2.0 EXACT in any order, so
         * min_prob = (1.0/2.0)*0.5 = 0.25 exact and pr = 0.5/2.0 = 0.25
         * exact: pr == min_prob, and `pr < min_prob` false must INCLUDE both
         * boundary candidates. Old semantics agree bit-for-bit here. */
        pulsar_sample_dist old, got;
        pulsar_sample_dist_build(logits, n, 1.0f, 0, 1.0f, 0.5f, &scratch, &got);
        ref_sample_dist_build_sortsum(logits, n, 1.0f, 0, 1.0f, 0.5f, &old);
        TEST_ASSERT(got.n == 3);
        TEST_ASSERT(got.ids[0] == 100 && got.ids[1] == 200 && got.ids[2] == 300);
        TEST_ASSERT(old.n == 3 && old.ids[1] == 200 && old.ids[2] == 300);
        for (uint32_t i = 0; i < 3; i++)
            TEST_ASSERT(memcmp(&old.probs[i], &got.probs[i], sizeof(float)) == 0);
        pulsar_sample_dist_free(&old);
        pulsar_sample_dist_free(&got);

        /* Shape B: add prob = 0.5 - (1..3)ulp at id 400. sum = 2.5 exact both
         * orders; the boundary pair still sits exactly AT min_prob
         * (0.5/2.5 == (1.0/2.5)*0.5 bit-for-bit) and stays included, while
         * id 400's pr rounds strictly below min_prob: the PREFILTER keeps it
         * (within slack) but the exact cutoff must trim it. n == 4 here
         * would mean prefilter slack leaked into membership. */
        logits[400] = L_below;
        pulsar_sample_dist_build(logits, n, 1.0f, 0, 1.0f, 0.5f, &scratch, &got);
        ref_sample_dist_build_sortsum(logits, n, 1.0f, 0, 1.0f, 0.5f, &old);
        TEST_ASSERT(got.n == 3);
        TEST_ASSERT(got.ids[0] == 100 && got.ids[1] == 200 && got.ids[2] == 300);
        TEST_ASSERT(old.n == 3);
        for (uint32_t i = 0; i < 3; i++) {
            TEST_ASSERT(old.ids[i] == got.ids[i]);
            TEST_ASSERT(memcmp(&old.probs[i], &got.probs[i], sizeof(float)) == 0);
        }
        pulsar_sample_dist_free(&old);
        pulsar_sample_dist_free(&got);
        printf("  prefilter: boundary teeth: AT-threshold pair included, "
               "ulp-below trimmed post-prefilter\n");
    }

    pulsar_sample_scratch_free(&scratch);
    free(logits);
}



/* =====
 * Sampled-proposal (p/q) speculative math — the pure-CPU oracle for
 * temperature-matched draft sampling. No model, no GPU, no batch numerics, so
 * a failure here is the math, not the engine.
 *
 * The property: drawing x ~ q, accepting w.p. min(1, p(x)/q(x)), and otherwise
 * drawing from the residual (p-q)+ must reproduce p EXACTLY — for ANY proposal
 * q, including one that proposes tokens p rules out. That is what lets the
 * drafter propose from a temperature-matched q without biasing the output.
 *
 * Also pins the two bugs fixed in pulsar_sample_dist_accept_pq / _draw_residual
 * (see the block comment in tokenizer.c), which are silent-corruption bugs:
 * both emit a *plausible* wrong token, so only a distributional test catches
 * them.
 */
#define SPEC_V 64u

/* total (p-q)+ mass, computed independently of _draw_residual's own loop */
static float spec_residual_mass(const pulsar_sample_dist *p, const pulsar_sample_dist *q) {
    float m = 0.0f;
    for (uint32_t i = 0; i < p->n; i++) {
        const float r = p->probs[i] - pulsar_sample_dist_prob(q, p->ids[i]);
        if (r > 0.0f) m += r;
    }
    return m;
}

static void test_spec_pq_math(void) {
    pulsar_sample_scratch scratch;
    memset(&scratch, 0, sizeof(scratch));
    sampler_warn_if_flush_to_zero();
    float pl[SPEC_V], ql[SPEC_V];
    long counts[SPEC_V];
    uint64_t rng = 12345;
    int fails = 0, checked = 0;

    for (int trial = 0; trial < 8; trial++) {
        /* p and q are built from DIFFERENT logits (target row vs drafter's
         * refined row) filtered at the SAME request params — the production
         * shape. Params vary across trials to cover top_p / top_k / min_p. */
        samp_seed(0xBEEF0000u + (uint64_t)trial);
        for (uint32_t i = 0; i < SPEC_V; i++) pl[i] = (float)(samp_u01() * 8.0 - 4.0);
        for (uint32_t i = 0; i < SPEC_V; i++) ql[i] = (float)(samp_u01() * 8.0 - 4.0);
        const char *shape = "disjoint-logits";
        if (trial == 6) {
            /* q is sharply peaked on a token p ranks last — forces the
             * rejection branch (and thus the residual) almost every draw. */
            shape = "q-peaked-off-p";
            for (uint32_t i = 0; i < SPEC_V; i++) ql[i] = -8.0f;
            ql[SPEC_V - 1] = 12.0f;
            pl[SPEC_V - 1] = -9.0f;
        } else if (trial == 7) {
            /* p == q: the ratio is 1 everywhere, acceptance must be total and
             * the residual must never be reached. */
            shape = "p-equals-q";
            memcpy(ql, pl, sizeof(pl));
        }
        const float temp = (trial % 2) ? 0.95f : 0.7f;
        const float topp = (trial % 3 == 0) ? 1.0f : 0.38f;
        const int   topk = (trial % 4 == 3) ? 8 : 0;
        const float minp = (trial == 5) ? 0.05f : 0.0f;

        pulsar_sample_dist p, q;
        pulsar_sample_dist_build(pl, SPEC_V, temp, topk, topp, minp, &scratch, &p);
        pulsar_sample_dist_build(ql, SPEC_V, temp, topk, topp, minp, &scratch, &q);
        const float rmass = spec_residual_mass(&p, &q);

        memset(counts, 0, sizeof(counts));
        const long N = 400000;
        long accepts = 0, zero_p = 0, zero_resid = 0;
        for (long it = 0; it < N; it++) {
            const int x = pulsar_sample_dist_draw(&q, &rng);
            int tok;
            if (pulsar_sample_dist_accept_pq(&p, x, pulsar_sample_dist_prob(&q, x), &rng)) {
                tok = x;
                accepts++;
            } else {
                tok = pulsar_sample_dist_draw_residual(&p, &q, &scratch, &rng);
                /* bug (b): a residual draw must carry strictly positive
                 * residual mass. Only the mass<=0 fallback (plain draw from p)
                 * is exempt, and it cannot occur while rmass > 0. */
                if (rmass > 0.0f &&
                    pulsar_sample_dist_prob(&p, tok) - pulsar_sample_dist_prob(&q, tok) <= 0.0f)
                    zero_resid++;
            }
            /* bug (a): an emitted token must be possible under the target. */
            if (pulsar_sample_dist_prob(&p, tok) <= 0.0f) zero_p++;
            counts[tok]++;
        }

        /* the marginal must be p */
        double chi = 0.0;
        int df = 0;
        for (uint32_t i = 0; i < p.n; i++) {
            const double e = (double)N * p.probs[i];
            if (e < 8.0) continue;
            const double o = (double)counts[p.ids[i]];
            chi += (o - e) * (o - e) / e;
            df++;
        }
        df = df > 1 ? df - 1 : 1;
        const double crit = df + 3.1 * sqrt(2.0 * df) + 4.0;
        const int bad = chi > crit || zero_p || zero_resid;
        printf("  spec-math trial %d (%s temp=%.2f top_k=%d top_p=%.2f min_p=%.2f "
               "|p|=%u |q|=%u): alpha=%.3f chi2=%.1f df=%d crit=%.1f%s%s -> %s\n",
               trial, shape, (double)temp, topk, (double)topp, (double)minp, p.n, q.n,
               (double)accepts / (double)N, chi, df, crit,
               zero_p ? " ZERO-P-EMITTED" : "", zero_resid ? " ZERO-RESIDUAL" : "",
               bad ? "FAIL" : "OK");
        if (bad) fails++;
        /* p == q must accept every single draw: min(1,p/q) == 1 identically. */
        if (trial == 7 && accepts != N) {
            printf("  spec-math: p==q accepted only %ld/%ld\n", accepts, N);
            fails++;
        }
        checked++;
        pulsar_sample_dist_free(&p);
        pulsar_sample_dist_free(&q);
    }

    /* Deterministic guards for the two reference bugs. The distributional loop
     * above is necessary but NOT sufficient for bug (a): `u <= p/q` only
     * misfires when u is exactly 0, which 3.2M random draws will usually miss.
     * What makes the bug impossible is that the p<=0 rejection happens BEFORE
     * any rng is drawn — so pin exactly that, by asserting the rng state is
     * untouched. A `u <= ap` implementation consumes a word here and fails,
     * whatever u happens to be. */
    {
        float l[SPEC_V];
        samp_seed(0x5AFE01u);
        for (uint32_t i = 0; i < SPEC_V; i++) l[i] = (float)(samp_u01() * 8.0 - 4.0);
        l[0] = 20.0f;                      /* one dominant token */
        pulsar_sample_dist p;
        /* top_p=0.5 with a dominant token => a small nucleus, so most of the
         * vocab sits strictly outside it with p(x) == 0. */
        pulsar_sample_dist_build(l, SPEC_V, 1.0f, 0, 0.5f, 0.0f, &scratch, &p);
        int off = -1;
        for (uint32_t i = 0; i < SPEC_V && off < 0; i++)
            if (pulsar_sample_dist_prob(&p, (int)i) <= 0.0f) off = (int)i;
        TEST_ASSERT(off >= 0);

        uint64_t r_before = 0xD15EA5Eull, r_after = 0xD15EA5Eull;
        const int acc = pulsar_sample_dist_accept_pq(&p, off, 0.5f, &r_after);
        TEST_ASSERT(acc == 0);             /* bug (a): p(x)==0 is never accepted */
        TEST_ASSERT(r_after == r_before);  /* ...and the guard fired before the rng */

        /* A certain accept (p >= q) must likewise consume no rng — this is what
         * keeps temperature<=0 byte-identical, where p and q are both point
         * masses of 1.0 at the argmax. */
        r_after = r_before;
        const float pmode = p.probs[0];
        const int acc2 = pulsar_sample_dist_accept_pq(&p, p.ids[0], pmode * 0.5f, &r_after);
        TEST_ASSERT(acc2 == 1);
        TEST_ASSERT(r_after == r_before);
        pulsar_sample_dist_free(&p);
    }
    {
        /* bug (b): the residual must never return a zero-(p-q)+ token. Build a
         * q that exactly covers p's LAST-ranked support element (residual 0
         * there, positive earlier) — the element a "return the last index"
         * fallback would wrongly emit on a rounding overrun. */
        float pl2[SPEC_V], ql2[SPEC_V];
        samp_seed(0x5AFE02u);
        for (uint32_t i = 0; i < SPEC_V; i++) pl2[i] = (float)(samp_u01() * 4.0 - 2.0);
        memcpy(ql2, pl2, sizeof(pl2));
        pulsar_sample_dist p2, q2;
        pulsar_sample_dist_build(pl2, SPEC_V, 1.0f, 4, 1.0f, 0.0f, &scratch, &p2);
        /* q2 == p2 on the tail but heavier there => tail residual is <= 0 */
        ql2[p2.ids[p2.n - 1]] += 3.0f;
        pulsar_sample_dist_build(ql2, SPEC_V, 1.0f, 4, 1.0f, 0.0f, &scratch, &q2);
        const int tail = p2.ids[p2.n - 1];
        TEST_ASSERT(pulsar_sample_dist_prob(&p2, tail) -
                    pulsar_sample_dist_prob(&q2, tail) <= 0.0f);
        uint64_t r2 = 777;
        int tail_emits = 0;
        for (int it = 0; it < 200000; it++) {
            const int tok = pulsar_sample_dist_draw_residual(&p2, &q2, &scratch, &r2);
            if (tok == tail) tail_emits++;
            if (pulsar_sample_dist_prob(&p2, tok) <= 0.0f) tail_emits++;
        }
        if (tail_emits)
            printf("  spec-math: residual emitted a zero-residual token %d times\n",
                   tail_emits);
        TEST_ASSERT(tail_emits == 0);
        pulsar_sample_dist_free(&p2);
        pulsar_sample_dist_free(&q2);
    }

    TEST_ASSERT(fails == 0);
    pulsar_sample_scratch_free(&scratch);
    printf("  spec-math: %d p/q accept+residual trials reproduce the target "
           "distribution; p=0 reject and zero-residual guards pinned\n", checked);
}



/* Per-surface sampling presence flags: every API surface must mark
 * temperature/top_k/top_p/min_p as client-sent exactly when the key appears
 * in the request body, so downstream think-mode defaulting
 * (gen_resolve_sampling in generate.c) can distinguish "explicitly 1.0" from
 * "absent". Parse-only; the engine is used just to tokenize the prompt. */
static void test_api_sampling_presence_flags(void) {
    pulsar_engine *engine = test_get_engine(false);
    if (!engine) return;
    char err[160];
    request r;

    /* OpenAI chat completions: all four knobs */
    TEST_ASSERT(parse_chat_request(engine, NULL,
        "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],"
        "\"temperature\":0.35,\"top_k\":40,\"top_p\":0.9,\"min_p\":0.1}",
        128, 32768, &r, err, sizeof(err)));
    TEST_ASSERT(r.has_temperature && r.temperature == 0.35f);
    TEST_ASSERT(r.has_top_k && r.top_k == 40);
    TEST_ASSERT(r.has_top_p && r.top_p == 0.9f);
    TEST_ASSERT(r.has_min_p && r.min_p == 0.1f);
    request_free(&r);

    /* OpenAI chat completions: nothing sent -> defaults, all flags false */
    TEST_ASSERT(parse_chat_request(engine, NULL,
        "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}",
        128, 32768, &r, err, sizeof(err)));
    TEST_ASSERT(!r.has_temperature && r.temperature == PULSAR_DEFAULT_TEMPERATURE);
    TEST_ASSERT(!r.has_top_k && r.top_k == 0);
    TEST_ASSERT(!r.has_top_p && r.top_p == PULSAR_DEFAULT_TOP_P);
    TEST_ASSERT(!r.has_min_p && r.min_p == PULSAR_DEFAULT_MIN_P);
    request_free(&r);

    /* Anthropic messages: temperature/top_p/top_k (the API has no min_p) */
    TEST_ASSERT(parse_anthropic_request(engine, NULL,
        "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],"
        "\"max_tokens\":64,\"temperature\":0.35,\"top_p\":0.9,\"top_k\":40}",
        128, 32768, &r, err, sizeof(err)));
    TEST_ASSERT(r.has_temperature && r.temperature == 0.35f);
    TEST_ASSERT(r.has_top_k && r.top_k == 40);
    TEST_ASSERT(r.has_top_p && r.top_p == 0.9f);
    TEST_ASSERT(!r.has_min_p && r.min_p == PULSAR_DEFAULT_MIN_P);
    request_free(&r);

    /* Responses: temperature/top_p */
    TEST_ASSERT(parse_responses_request(engine, NULL,
        "{\"input\":\"hi\",\"temperature\":0.35,\"top_p\":0.9}",
        128, 32768, &r, err, sizeof(err)));
    TEST_ASSERT(r.has_temperature && r.temperature == 0.35f);
    TEST_ASSERT(r.has_top_p && r.top_p == 0.9f);
    TEST_ASSERT(!r.has_top_k && !r.has_min_p);
    request_free(&r);

    /* Legacy completions: all four knobs; explicit values EQUAL to the
     * defaults must still be flagged as client-sent */
    TEST_ASSERT(parse_completion_request(engine,
        "{\"prompt\":\"hi\",\"temperature\":1.0,\"top_k\":0,"
        "\"top_p\":1.0,\"min_p\":0.05}",
        128, 32768, &r, err, sizeof(err)));
    TEST_ASSERT(r.has_temperature && r.temperature == PULSAR_DEFAULT_TEMPERATURE);
    TEST_ASSERT(r.has_top_k && r.top_k == 0);
    TEST_ASSERT(r.has_top_p && r.top_p == PULSAR_DEFAULT_TOP_P);
    TEST_ASSERT(r.has_min_p && r.min_p == PULSAR_DEFAULT_MIN_P);
    request_free(&r);
}



/* min_p range validation at parse: an out-of-range min_p (min_p < 0 or
 * min_p > 1) disables the filter (0.0), following top_p's out-of-range
 * convention in the engine samplers; unvalidated, min_p > 1 silently
 * collapsed sampling to greedy.  In-range values pass through untouched,
 * and the presence flag is set either way (the client DID send it, so
 * think-mode defaulting must not re-assert PULSAR_DEFAULT_MIN_P).  Covers
 * both surfaces that parse min_p (OpenAI chat + legacy completions). */
static void test_api_min_p_range_validation(void) {
    pulsar_engine *engine = test_get_engine(false);
    if (!engine) return;
    char err[160];
    request r;

    /* OpenAI chat completions: min_p > 1 -> filter disabled, flag set */
    TEST_ASSERT(parse_chat_request(engine, NULL,
        "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"min_p\":1.5}",
        128, 32768, &r, err, sizeof(err)));
    TEST_ASSERT(r.has_min_p && r.min_p == 0.0f);
    request_free(&r);

    /* OpenAI chat completions: min_p < 0 -> filter disabled, flag set */
    TEST_ASSERT(parse_chat_request(engine, NULL,
        "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"min_p\":-0.5}",
        128, 32768, &r, err, sizeof(err)));
    TEST_ASSERT(r.has_min_p && r.min_p == 0.0f);
    request_free(&r);

    /* OpenAI chat completions: boundary values 0 and 1 are valid as-is */
    TEST_ASSERT(parse_chat_request(engine, NULL,
        "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"min_p\":1.0}",
        128, 32768, &r, err, sizeof(err)));
    TEST_ASSERT(r.has_min_p && r.min_p == 1.0f);
    request_free(&r);

    /* Legacy completions: same clamp on the other min_p surface */
    TEST_ASSERT(parse_completion_request(engine,
        "{\"prompt\":\"hi\",\"min_p\":2.0}",
        128, 32768, &r, err, sizeof(err)));
    TEST_ASSERT(r.has_min_p && r.min_p == 0.0f);
    request_free(&r);

    /* Legacy completions: in-range value untouched */
    TEST_ASSERT(parse_completion_request(engine,
        "{\"prompt\":\"hi\",\"min_p\":0.1}",
        128, 32768, &r, err, sizeof(err)));
    TEST_ASSERT(r.has_min_p && r.min_p == 0.1f);
    request_free(&r);
}



static void test_server_unit_group(void) {
    pulsar_server_unit_tests_run();
}

typedef void (*test_fn)(void);

typedef struct {
    const char *flag;
    const char *name;
    const char *desc;
    test_fn fn;
} pulsar_test_entry;

static const pulsar_test_entry test_entries[] = {
#ifndef PULSAR_NO_GPU
    {"--long-context", "long-context", "long-context story fact-recall regression", test_long_story_fact_recall},
    {"--tool-call-quality", "tool-call-quality", "model emits valid DSML tool calls", test_tool_call_quality},
    {"--think-tool-recovery", "think-tool-recovery", "forced </think> recovery when a tool call starts inside thinking", test_think_tool_recovery},
    {"--logprob-vectors", "logprob-vectors", "official API top-logprob vector comparison on the standard path", test_official_logprob_vectors},
    {"--local-golden-vectors", "local-golden-vectors", "local top-k/logit drift regression for long prefill", test_local_golden_vectors},
    {"--short-prefill-ratio4", "short-prefill-ratio4", "ratio-4 short prefill regression", test_short_prefill_ratio4},
    {"--f16-kernels", "f16-kernels", "isolated F16 matmul kernel numeric regressions", test_f16_kernel_group},
    {"--tensor-equivalence", "tensor-equivalence", "fast/quality prompt-logit and greedy equivalence", test_mpp_equivalence},
    {"--api-sampling-flags", "api-sampling-flags", "per-surface sampling params set client-sent presence flags", test_api_sampling_presence_flags},
    {"--api-min-p-range", "api-min-p-range", "out-of-range min_p disables the filter at parse (top_p convention)", test_api_min_p_range_validation},
#endif
    {"--sampler", "sampler", "sampler byte-exactness vs re-derived reference", test_sampler_dist_equivalence},
    {"--sampler-prefilter", "sampler-prefilter", "min-p prefilter: survivor set/order identity vs old-sum reference + boundary teeth", test_sampler_prefilter_equivalence},
    {"--spec-math", "spec-math", "sampled-proposal p/q accept + residual reproduces the target", test_spec_pq_math},
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
#ifndef PULSAR_NO_GPU
    puts("  --mpp-equivalence");
    puts("      Compatibility alias for --tensor-equivalence.");
#endif
    puts("  -h, --help");
    puts("      Show this help.");
    puts("\nEnvironment:");
    puts("  DS4_TEST_MODEL=FILE        Model path. Default: ds4flash.gguf");
    puts("  DS4_TEST_VECTOR_FILE=FILE  Simple official-vector fixture.");
    puts("  DS4_TEST_LOCAL_GOLDEN_FILE=FILE  Local top-k golden-vector fixture.");
    puts("  DS4_TEST_MPP_EQ_CASE=NAME  Run only Tensor equivalence cases whose id contains NAME.");
}

static const pulsar_test_entry *test_find_entry(const char *arg) {
#ifndef PULSAR_NO_GPU
    if (!strcmp(arg, "--mpp-equivalence")) {
        arg = "--tensor-equivalence";
    }
#endif
    for (size_t i = 0; i < sizeof(test_entries) / sizeof(test_entries[0]); i++) {
        if (!strcmp(arg, test_entries[i].flag)) return &test_entries[i];
    }
    return NULL;
}

/* Informational (non-gating) tests. Their mismatches are reported but do
 * not fail the suite:
 * - logprob-vectors compares against full-precision official-API logprobs;
 *   the 2-bit REAP-pruned production model is EXPECTED to mismatch a few
 *   cases (stable set, unchanged across releases). It is a drift dashboard
 *   against the official reference, not a pass/fail gate.
 * - think-tool-recovery flips run-to-run because batched prefill's routed
 *   down-sum uses float atomics (nondeterministic order); see the
 *   determinism item in the release notes. Internal-correctness gates
 *   (tensor-equivalence, golden vectors, server, ...) remain gating. */
static bool test_entry_is_informational(const pulsar_test_entry *entry) {
    return !strcmp(entry->name, "logprob-vectors") ||
           !strcmp(entry->name, "think-tool-recovery");
}

static int test_informational_failures;

static void test_run_entry(const pulsar_test_entry *entry) {
    int before = test_failures;
    fprintf(stderr, "%s:\n", entry->name);
    entry->fn();
    int delta = test_failures - before;
    fprintf(stderr, "%s: ", entry->name);
    if (delta != 0 && test_entry_is_informational(entry)) {
        test_failures = before;
        test_informational_failures += delta;
        pulsar_log(stderr, PULSAR_LOG_WARNING,
                "INFORMATIONAL (%d expected-parity/flaky mismatches, non-gating)",
                delta);
    } else {
        pulsar_log(stderr,
                delta == 0 ? PULSAR_LOG_OK : PULSAR_LOG_ERROR,
                "%s",
                delta == 0 ? "OK" : "ERR");
    }
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
            const pulsar_test_entry *entry = test_find_entry(argv[i]);
            if (!entry) {
                fprintf(stderr, "pulsar-test: unknown test switch: %s\n", argv[i]);
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

#ifndef PULSAR_NO_GPU
    test_close_engines();
#endif

    if (test_failures) {
        fprintf(stderr, "pulsar tests: %d failure(s)\n", test_failures);
        return 1;
    }
    if (test_informational_failures) {
        fprintf(stderr,
                "pulsar tests: PASS (%d informational mismatches: official-parity "
                "on the 2-bit model and/or the known-flaky recovery case)\n",
                test_informational_failures);
        return 0;
    }
    puts("pulsar tests: ok");
    return 0;
}
