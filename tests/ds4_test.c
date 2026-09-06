#define DS4_SERVER_TEST
#define DS4_SERVER_TEST_NO_MAIN
#include "../ds4_server.c"
#ifndef DS4_NO_GPU
#include "../ds4_gpu.h"
#include <math.h>

bool ds4_test_dspark_cache_window_crop(void);
bool ds4_test_dspark_prefix_capture(ds4_engine *engine, const ds4_tokens *prompt);

static ds4_engine *test_engine_fast;
static ds4_engine *test_engine_quality;

static char *test_read_file(const char *path);

static const char *test_model_path(void) {
    const char *model_path = getenv("DS4_TEST_MODEL");
    return (model_path && model_path[0]) ? model_path : "ds4flash.gguf";
}

static bool test_env_bool(const char *name) {
    const char *v = getenv(name);
    return v && v[0] && strcmp(v, "0") != 0;
}

static uint32_t test_env_u32(const char *name) {
    const char *v = getenv(name);
    if (!v || !v[0]) return 0;
    char *end = NULL;
    unsigned long n = strtoul(v, &end, 10);
    if (end == v) return 0;
    return n > UINT32_MAX ? UINT32_MAX : (uint32_t)n;
}

static uint64_t test_env_gib(const char *name) {
    const char *v = getenv(name);
    if (!v || !v[0]) return 0;
    char *end = NULL;
    unsigned long long n = strtoull(v, &end, 10);
    if (end == v || n == 0) return 0;
    const uint64_t one_gib = 1024ull * 1024ull * 1024ull;
    if (n > UINT64_MAX / one_gib) return UINT64_MAX;
    return (uint64_t)n * one_gib;
}

static char *test_save_env(const char *name) {
    const char *value = getenv(name);
    if (!value) return NULL;
    size_t len = strlen(value);
    char *copy = malloc(len + 1);
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

typedef struct {
    char *cold_decode;
    char *batch_selected_addr;
} test_streaming_prefill_env;

static test_streaming_prefill_env test_force_canonical_streaming_prefill(void) {
    test_streaming_prefill_env saved = {
        .cold_decode =
            test_save_env("DS4_METAL_DISABLE_STREAMING_COLD_DECODE_PREFILL"),
        .batch_selected_addr =
            test_save_env("DS4_METAL_DISABLE_STREAMING_PREFILL_BATCH_SELECTED_ADDR"),
    };
    if (test_env_bool("DS4_TEST_SSD_STREAMING")) {
        setenv("DS4_METAL_DISABLE_STREAMING_COLD_DECODE_PREFILL", "1", 1);
        setenv("DS4_METAL_DISABLE_STREAMING_PREFILL_BATCH_SELECTED_ADDR", "1", 1);
    }
    return saved;
}

static void test_restore_canonical_streaming_prefill(
        test_streaming_prefill_env saved) {
    test_restore_env("DS4_METAL_DISABLE_STREAMING_COLD_DECODE_PREFILL",
                     saved.cold_decode);
    test_restore_env("DS4_METAL_DISABLE_STREAMING_PREFILL_BATCH_SELECTED_ADDR",
                     saved.batch_selected_addr);
}

static ds4_backend test_model_backend(void) {
    const char *backend = getenv("DS4_TEST_BACKEND");
    if (backend && !strcmp(backend, "cpu")) return DS4_BACKEND_CPU;
#ifdef __APPLE__
    return DS4_BACKEND_METAL;
#else
    return DS4_BACKEND_CUDA;
#endif
}

static ds4_engine *test_open_engine(bool quality) {
    ds4_engine *engine = NULL;
    /* DS4_TEST_MTP loads the MTP head on the fast engine so the speculative
     * verify regression can reuse it; draft=4 hits the multi-row verify path. */
    const char *mtp = getenv("DS4_TEST_MTP");
    ds4_engine_options opt = {
        .model_path = test_model_path(),
        .backend = test_model_backend(),
        .quality = quality,
        .ssd_streaming = test_env_bool("DS4_TEST_SSD_STREAMING"),
        .ssd_streaming_cold = test_env_bool("DS4_TEST_SSD_STREAMING_COLD"),
        .ssd_streaming_cache_experts =
            test_env_u32("DS4_TEST_SSD_STREAMING_CACHE_EXPERTS"),
        .ssd_streaming_cache_bytes =
            test_env_gib("DS4_TEST_SSD_STREAMING_CACHE_GB"),
        .ssd_streaming_preload_experts =
            test_env_u32("DS4_TEST_SSD_STREAMING_PRELOAD_EXPERTS"),
        .mtp_path = (mtp && mtp[0] && !quality) ? mtp : NULL,
        .mtp_draft_tokens = (mtp && mtp[0] && !quality) ? 4 : 0,
        .glm_mtp = test_env_bool("DS4_TEST_GLM_MTP"),
    };
    TEST_ASSERT(ds4_engine_open(&engine, &opt) == 0);
    return engine;
}

static ds4_engine *test_get_engine(bool quality) {
    ds4_engine **slot = quality ? &test_engine_quality : &test_engine_fast;
    if (*slot) return *slot;

    *slot = test_open_engine(quality);
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

static void test_session_snapshot_roundtrip(void) {
    ds4_engine *engine = test_get_engine(false);
    if (!engine) return;

    ds4_session *reference = NULL;
    ds4_session *restored = NULL;
    ds4_session_snapshot snapshot = {0};
    ds4_tokens prompt = {0};
    char err[192] = {0};
    ds4_token_score before[8];
    ds4_token_score reference_after[8];
    ds4_token_score restored_before[8];
    ds4_token_score restored_after[8];
    enum { GLM_MTP_SNAPSHOT_CYCLES = 16 };
    int reference_accepted[GLM_MTP_SNAPSHOT_CYCLES * 2] = {0};
    int reference_counts[GLM_MTP_SNAPSHOT_CYCLES] = {0};
    int reference_total = 0;
    const bool test_glm_mtp = test_env_bool("DS4_TEST_GLM_MTP");
#ifdef DS4_ROCM_BUILD
    const float continued_logit_tolerance =
        ds4_engine_is_glm53(engine) ? 1e-5f : 1e-6f;
#else
    const float continued_logit_tolerance = 1e-6f;
#endif

    uint32_t ctx = test_env_u32("DS4_TEST_SNAPSHOT_CTX");
    if (ctx == 0) ctx = 1024;
    const char *prompt_path = getenv("DS4_TEST_SNAPSHOT_PROMPT");
    char *prompt_text = prompt_path && prompt_path[0] ?
        test_read_file(prompt_path) : NULL;
    if (prompt_path && prompt_path[0]) {
        TEST_ASSERT(prompt_text != NULL);
        if (!prompt_text) goto cleanup;
    }

    TEST_ASSERT(ds4_session_create(&reference, engine, ctx) == 0);
    if (!reference) goto cleanup;

    ds4_chat_begin(engine, &prompt);
    ds4_chat_append_message(engine, &prompt, "user",
                            prompt_text ? prompt_text :
                            "Give one concise reason to test session restore.");
    ds4_chat_append_assistant_prefix(engine, &prompt, DS4_THINK_NONE);
    TEST_ASSERT(prompt.len > 0);
    fprintf(stderr, "ds4-test: snapshot prompt=%d ctx=%u glm_mtp=%d\n",
            prompt.len, ctx, test_glm_mtp);
    TEST_ASSERT(ds4_session_sync(reference, &prompt, err, sizeof(err)) == 0);
    TEST_ASSERT(ds4_session_top_logprobs(reference, before, 8) == 8);
    TEST_ASSERT(ds4_session_save_snapshot(reference, &snapshot,
                                          err, sizeof(err)) == 0);
    TEST_ASSERT(snapshot.ptr != NULL && snapshot.len > 0);
    if (!snapshot.ptr || snapshot.len == 0) goto cleanup;

    if (test_glm_mtp) {
        for (int cycle = 0; cycle < GLM_MTP_SNAPSHOT_CYCLES; cycle++) {
            const int first = ds4_session_argmax(reference);
            const int n = ds4_session_eval_speculative_argmax(
                    reference, first, 2, -1,
                    reference_accepted + reference_total, 2,
                    err, sizeof(err));
            TEST_ASSERT(n > 0 && n <= 2);
            if (n <= 0 || n > 2) goto cleanup;
            reference_counts[cycle] = n;
            reference_total += n;
        }
    } else {
        TEST_ASSERT(ds4_session_eval(reference, before[0].id,
                                     err, sizeof(err)) == 0);
    }
    TEST_ASSERT(ds4_session_top_logprobs(reference, reference_after, 8) == 8);
    ds4_session_free(reference);
    reference = NULL;

    TEST_ASSERT(ds4_session_create(&restored, engine, ctx) == 0);
    if (!restored) goto cleanup;
    TEST_ASSERT(ds4_session_load_snapshot(restored, &snapshot,
                                          err, sizeof(err)) == 0);
    TEST_ASSERT(ds4_session_top_logprobs(restored, restored_before, 8) == 8);
    for (int i = 0; i < 8; i++) {
        if (restored_before[i].id != before[i].id ||
            fabsf(restored_before[i].logit - before[i].logit) > 1e-6f) {
            fprintf(stderr,
                    "ds4-test: snapshot before[%d] reference=(%d,%.9g) "
                    "restored=(%d,%.9g) delta=%.9g\n",
                    i, before[i].id, before[i].logit,
                    restored_before[i].id, restored_before[i].logit,
                    restored_before[i].logit - before[i].logit);
        }
        TEST_ASSERT(restored_before[i].id == before[i].id);
        TEST_ASSERT(fabsf(restored_before[i].logit - before[i].logit) <= 1e-6f);
    }

    if (test_glm_mtp) {
        int restored_total = 0;
        int single_cycles = 0;
        int double_cycles = 0;
        for (int cycle = 0; cycle < GLM_MTP_SNAPSHOT_CYCLES; cycle++) {
            int restored_accepted[2] = {0};
            const int first = ds4_session_argmax(restored);
            TEST_ASSERT(first == reference_accepted[restored_total]);
            const int n = ds4_session_eval_speculative_argmax(
                    restored, first, 2, -1,
                    restored_accepted, 2, err, sizeof(err));
            TEST_ASSERT(n == reference_counts[cycle]);
            if (n != reference_counts[cycle]) goto cleanup;
            for (int i = 0; i < n; i++) {
                TEST_ASSERT(restored_accepted[i] ==
                            reference_accepted[restored_total + i]);
            }
            restored_total += n;
            single_cycles += n == 1;
            double_cycles += n == 2;
        }
        TEST_ASSERT(restored_total == reference_total);
        fprintf(stderr,
                "ds4-test: GLM MTP snapshot cycles=%d single=%d double=%d tokens=%d\n",
                GLM_MTP_SNAPSHOT_CYCLES,
                single_cycles,
                double_cycles,
                restored_total);
    } else {
        TEST_ASSERT(ds4_session_eval(restored, before[0].id,
                                     err, sizeof(err)) == 0);
    }
    TEST_ASSERT(ds4_session_top_logprobs(restored, restored_after, 8) == 8);
    for (int i = 0; i < 8; i++) {
        if (restored_after[i].id != reference_after[i].id ||
            fabsf(restored_after[i].logit - reference_after[i].logit) >
                continued_logit_tolerance) {
            fprintf(stderr,
                    "ds4-test: snapshot after[%d] reference=(%d,%.9g) "
                    "restored=(%d,%.9g) delta=%.9g\n",
                    i, reference_after[i].id, reference_after[i].logit,
                    restored_after[i].id, restored_after[i].logit,
                    restored_after[i].logit - reference_after[i].logit);
        }
        TEST_ASSERT(restored_after[i].id == reference_after[i].id);
        TEST_ASSERT(fabsf(restored_after[i].logit -
                          reference_after[i].logit) <=
                    continued_logit_tolerance);
    }
    if (test_glm_mtp) {
        TEST_ASSERT(ds4_session_sync(restored, &prompt,
                                     err, sizeof(err)) == 0);
        TEST_ASSERT(ds4_session_top_logprobs(restored,
                                             restored_before, 8) == 8);
        for (int i = 0; i < 8; i++) {
            TEST_ASSERT(restored_before[i].id == before[i].id);
            TEST_ASSERT(fabsf(restored_before[i].logit - before[i].logit) <=
                        1e-6f);
        }
        int reuse_single = 0;
        int reuse_double = 0;
        for (int cycle = 0; cycle < 4; cycle++) {
            int cycle_accepted[2] = {0};
            const int first = ds4_session_argmax(restored);
            const int n = ds4_session_eval_speculative_argmax(
                    restored, first, 2, -1,
                    cycle_accepted, 2, err, sizeof(err));
            TEST_ASSERT(n > 0 && n <= 2);
            if (n <= 0 || n > 2) goto cleanup;
            reuse_single += n == 1;
            reuse_double += n == 2;
        }
        fprintf(stderr,
                "ds4-test: GLM MTP context reuse single=%d double=%d\n",
                reuse_single,
                reuse_double);
    }

cleanup:
    free(prompt_text);
    ds4_tokens_free(&prompt);
    ds4_session_snapshot_free(&snapshot);
    ds4_session_free(restored);
    ds4_session_free(reference);
}

static uint64_t test_round_up_u64(uint64_t n, uint64_t align) {
    return (n + align - 1) & ~(align - 1);
}

static uint32_t test_float_ordered_u32(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return (bits & 0x80000000u) != 0 ? ~bits : bits | 0x80000000u;
}

typedef struct {
    size_t mismatch_count;
    uint32_t max_ulp;
    float max_abs;
} test_float_compare_stats;

static test_float_compare_stats test_compare_float_bits(
        const float *reference,
        const float *actual,
        size_t count) {
    test_float_compare_stats stats = {0};
    for (size_t i = 0; i < count; i++) {
        if (memcmp(&reference[i], &actual[i], sizeof(float)) != 0) {
            stats.mismatch_count++;
        }

        const uint32_t ref_ordered = test_float_ordered_u32(reference[i]);
        const uint32_t actual_ordered = test_float_ordered_u32(actual[i]);
        const uint32_t ulp = ref_ordered > actual_ordered
            ? ref_ordered - actual_ordered
            : actual_ordered - ref_ordered;
        if (ulp > stats.max_ulp) stats.max_ulp = ulp;

        const float abs_error = fabsf(reference[i] - actual[i]);
        if (abs_error > stats.max_abs) stats.max_abs = abs_error;
    }
    return stats;
}

#if defined(__APPLE__)
static const uint32_t test_copy_f32_patterns[] = {
    0x00000000u, 0x80000000u, /* signed zero */
    0x7f800000u, 0xff800000u, /* infinities */
    0x7fc00000u, 0x7fc12345u, 0xffc54321u, 0x7fa00001u, /* NaNs */
    0x00000001u, 0x007fffffu, 0x80000001u, 0x807fffffu, /* F32 subnormals */
    0x32ffffffu, 0x33000000u, 0x33000001u,
    0x337fffffu, 0x33800000u, 0x33800001u, /* minimum F16 subnormal boundary */
    0x387fbfffu, 0x387fc000u, 0x387fffffu,
    0x38800000u, 0x38800001u, /* maximum subnormal/minimum normal boundary */
    0x3f800fffu, 0x3f801000u, 0x3f801001u,
    0x3f802fffu, 0x3f803000u, 0x3f803001u, /* round-to-nearest ties */
    0x477fdfffu, 0x477fe000u, 0x477fefffu,
    0x477ff000u, 0x477ff001u, 0x47800000u, /* maximum/overflow boundary */
    0xc77fe000u, 0xbf801000u, 0x3eaaaaabu, 0xbeaaaaabu,
};

static void test_fill_copy_f32_patterns(void *dst, uint32_t n, uint32_t salt) {
    uint8_t *bytes = dst;
    const uint32_t n_patterns =
        (uint32_t)(sizeof(test_copy_f32_patterns) / sizeof(test_copy_f32_patterns[0]));
    for (uint32_t i = 0; i < n; i++) {
        const uint32_t bits = test_copy_f32_patterns[(i + salt) % n_patterns];
        memcpy(bytes + (uint64_t)i * sizeof(bits), &bits, sizeof(bits));
    }
}
#endif

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

static void test_fill_q8_0_weights(uint8_t *weights,
                                   uint32_t in_dim,
                                   uint32_t out_dim,
                                   uint32_t seed) {
    const uint32_t blocks = in_dim / 32u;
    const uint64_t row_bytes = (uint64_t)blocks * 34u;
    for (uint32_t o = 0; o < out_dim; o++) {
        uint8_t *row = weights + (uint64_t)o * row_bytes;
        for (uint32_t b = 0; b < blocks; b++) {
            float vals[32];
            float amax = 0.0f;
            for (uint32_t i = 0; i < 32; i++) {
                const uint32_t k = b * 32u + i;
                const int v = (int)((o * 17u + k * 23u + (o ^ k) * 3u +
                                     seed * 29u + ((o + seed) ^ k) * 5u) % 67u) - 33;
                vals[i] = (float)v / 96.0f;
                float av = fabsf(vals[i]);
                if (av > amax) amax = av;
            }
            const uint16_t scale_bits = test_float_to_f16(amax / 127.0f);
            const float scale = test_f16_to_f32(scale_bits);
            memcpy(row + b * 34u, &scale_bits, sizeof(scale_bits));
            int8_t *qs = (int8_t *)(row + b * 34u + 2u);
            for (uint32_t i = 0; i < 32; i++) {
                int q = scale != 0.0f ? (int)lrintf(vals[i] / scale) : 0;
                if (q > 127) q = 127;
                if (q < -128) q = -128;
                qs[i] = (int8_t)q;
            }
        }
    }
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

static void test_metal_f16_prefill_matmul(void) {
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

    uint16_t *weights = weights_raw;
    memset(weights, 0, (size_t)weight_alloc);
    for (uint32_t o = 0; o < out_dim; o++) {
        for (uint32_t i = 0; i < in_dim; i++) {
            const int v = (int)((o * 11u + i * 13u + (o ^ i) * 5u) % 61u) - 30;
            weights[(uint64_t)o * in_dim + i] = test_float_to_f16((float)v / 96.0f);
        }
    }

    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(x_bytes);
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(out_bytes);
    TEST_ASSERT(x != NULL);
    TEST_ASSERT(out != NULL);
    if (!x || !out) {
        ds4_gpu_tensor_free(x);
        ds4_gpu_tensor_free(out);
        free(weights_raw);
        return;
    }

    float *x_host = malloc((size_t)x_bytes);
    float *out_host = malloc((size_t)out_bytes);
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

    for (uint32_t t = 0; t < n_tok; t++) {
        for (uint32_t i = 0; i < in_dim; i++) {
            const int v = (int)((t * 7u + i * 17u + (t ^ i) * 3u) % 73u) - 36;
            x_host[(uint64_t)t * in_dim + i] = (float)v / 80.0f;
        }
    }
    for (uint32_t i = 0; i < n_tok * out_dim; i++) {
        out_host[i] = 12345.0f;
    }

    TEST_ASSERT(ds4_gpu_tensor_write(x, 0, x_host, x_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(out, 0, out_host, out_bytes) != 0);
    TEST_ASSERT(ds4_gpu_set_model_map(weights_raw, weight_alloc) != 0);
    ds4_gpu_set_quality(false);
    TEST_ASSERT(ds4_gpu_matmul_f16_tensor(out, weights_raw, weight_alloc, 0,
                                          in_dim, out_dim, x, n_tok) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(out, 0, out_host, out_bytes) != 0);

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
    ds4_gpu_tensor_free(x);
    ds4_gpu_tensor_free(out);
    free(weights_raw);
}

static void test_metal_q8_0_prefill_matmul(void) {
    const uint32_t in_dim = 128;
    const uint32_t out_dim = 64;
    const uint32_t n_tok = 128;
    const uint64_t row_bytes = (uint64_t)(in_dim / 32u) * 34u;
    const uint64_t weight_bytes = (uint64_t)out_dim * row_bytes;
    const uint64_t weight_alloc = test_round_up_u64(weight_bytes, (uint64_t)getpagesize());
    const uint64_t x_bytes = (uint64_t)n_tok * in_dim * sizeof(float);
    const uint64_t out_bytes = (uint64_t)n_tok * out_dim * sizeof(float);

    void *weights_raw = NULL;
    TEST_ASSERT(posix_memalign(&weights_raw, (size_t)getpagesize(), (size_t)weight_alloc) == 0);
    if (!weights_raw) return;

    uint8_t *weights = weights_raw;
    memset(weights, 0, (size_t)weight_alloc);
    test_fill_q8_0_weights(weights, in_dim, out_dim, 0);

    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(x_bytes);
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(out_bytes);
    TEST_ASSERT(x != NULL);
    TEST_ASSERT(out != NULL);
    if (!x || !out) {
        ds4_gpu_tensor_free(x);
        ds4_gpu_tensor_free(out);
        free(weights_raw);
        return;
    }

    float *x_host = malloc((size_t)x_bytes);
    float *out_host = malloc((size_t)out_bytes);
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

    for (uint32_t t = 0; t < n_tok; t++) {
        for (uint32_t i = 0; i < in_dim; i++) {
            const int v = (int)((t * 19u + i * 7u + (t ^ i)) % 71u) - 35;
            x_host[(uint64_t)t * in_dim + i] = (float)v / 80.0f;
        }
    }
    for (uint32_t i = 0; i < n_tok * out_dim; i++) {
        out_host[i] = 12345.0f;
    }

    TEST_ASSERT(ds4_gpu_tensor_write(x, 0, x_host, x_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(out, 0, out_host, out_bytes) != 0);
    TEST_ASSERT(ds4_gpu_set_model_map(weights_raw, weight_alloc) != 0);
    ds4_gpu_set_quality(false);
    TEST_ASSERT(ds4_gpu_matmul_q8_0_tensor(out, weights_raw, weight_alloc, 0,
                                           in_dim, out_dim, x, n_tok) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(out, 0, out_host, out_bytes) != 0);

    float max_abs = 0.0f;
    float rms = 0.0f;
    for (uint32_t t = 0; t < n_tok; t++) {
        for (uint32_t o = 0; o < out_dim; o++) {
            const uint8_t *row = weights + (uint64_t)o * row_bytes;
            float ref = 0.0f;
            for (uint32_t b = 0; b < in_dim / 32u; b++) {
                uint16_t scale_bits;
                memcpy(&scale_bits, row + b * 34u, sizeof(scale_bits));
                const float scale = test_f16_to_f32(scale_bits);
                const int8_t *qs = (const int8_t *)(row + b * 34u + 2u);
                for (uint32_t i = 0; i < 32; i++) {
                    ref += scale * (float)qs[i] *
                           x_host[(uint64_t)t * in_dim + b * 32u + i];
                }
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
    ds4_gpu_tensor_free(x);
    ds4_gpu_tensor_free(out);
    free(weights_raw);
}

static void test_metal_pack_slot_rows_f32(void) {
    const uint32_t n_rows = 3;
    const uint32_t width = 5;
    const uint32_t n_slots = 4;
    const uint32_t slot_cap = 6;
    const uint64_t slot_count = (uint64_t)n_slots * slot_cap * width;
    const uint64_t out_count = (uint64_t)n_rows * n_slots * width;
    const uint64_t slot_bytes = slot_count * sizeof(float);
    const uint64_t out_bytes = out_count * sizeof(float);

    ds4_gpu_tensor *slots = ds4_gpu_tensor_alloc(slot_bytes);
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(out_bytes);
    TEST_ASSERT(slots != NULL);
    TEST_ASSERT(out != NULL);
    if (!slots || !out) {
        ds4_gpu_tensor_free(slots);
        ds4_gpu_tensor_free(out);
        return;
    }

    float *slot_host = malloc((size_t)slot_bytes);
    float *out_host = malloc((size_t)out_bytes);
    TEST_ASSERT(slot_host != NULL);
    TEST_ASSERT(out_host != NULL);
    if (!slot_host || !out_host) {
        free(slot_host);
        free(out_host);
        ds4_gpu_tensor_free(slots);
        ds4_gpu_tensor_free(out);
        return;
    }

    for (uint32_t slot = 0; slot < n_slots; slot++) {
        for (uint32_t row = 0; row < slot_cap; row++) {
            for (uint32_t col = 0; col < width; col++) {
                slot_host[((uint64_t)slot * slot_cap + row) * width + col] =
                    (float)(slot * 1000u + row * 100u + col);
            }
        }
    }
    for (uint64_t i = 0; i < out_count; i++) out_host[i] = -1.0f;

    TEST_ASSERT(ds4_gpu_tensor_write(slots, 0, slot_host, slot_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(out, 0, out_host, out_bytes) != 0);
    TEST_ASSERT(ds4_gpu_pack_slot_rows_f32_tensor(out,
                                                  slots,
                                                  n_rows,
                                                  width,
                                                  n_slots,
                                                  slot_cap) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(out, 0, out_host, out_bytes) != 0);

    for (uint32_t row = 0; row < n_rows; row++) {
        for (uint32_t slot = 0; slot < n_slots; slot++) {
            for (uint32_t col = 0; col < width; col++) {
                const float ref =
                    slot_host[((uint64_t)slot * slot_cap + row) * width + col];
                const float got =
                    out_host[((uint64_t)row * n_slots + slot) * width + col];
                TEST_ASSERT(got == ref);
            }
        }
    }

    free(slot_host);
    free(out_host);
    ds4_gpu_tensor_free(slots);
    ds4_gpu_tensor_free(out);
}

static void test_metal_store_raw_kv_batch_wrap(void) {
    const uint32_t raw_cap = 5;
    const uint32_t head_dim = 3;
    const uint32_t n_tokens = 4;
    const uint32_t pos0 = 3;
    const uint64_t kv_count = (uint64_t)n_tokens * head_dim;
    const uint64_t raw_count = (uint64_t)raw_cap * head_dim;
    const uint64_t kv_bytes = kv_count * sizeof(float);
    const uint64_t raw_bytes = raw_count * sizeof(float);

    ds4_gpu_tensor *kv = ds4_gpu_tensor_alloc(kv_bytes);
    ds4_gpu_tensor *raw = ds4_gpu_tensor_alloc(raw_bytes);
    TEST_ASSERT(kv != NULL);
    TEST_ASSERT(raw != NULL);
    if (!kv || !raw) {
        ds4_gpu_tensor_free(kv);
        ds4_gpu_tensor_free(raw);
        return;
    }

    float kv_host[12];
    float raw_host[15];
    for (uint32_t t = 0; t < n_tokens; t++) {
        for (uint32_t d = 0; d < head_dim; d++) {
            kv_host[(uint64_t)t * head_dim + d] = (float)(100u * t + d);
        }
    }
    for (uint64_t i = 0; i < raw_count; i++) raw_host[i] = -1.0f;

    TEST_ASSERT(ds4_gpu_tensor_write(kv, 0, kv_host, kv_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(raw, 0, raw_host, raw_bytes) != 0);
    TEST_ASSERT(ds4_gpu_store_raw_kv_batch_tensor(raw,
                                                  kv,
                                                  raw_cap,
                                                  pos0,
                                                  n_tokens,
                                                  head_dim) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(raw, 0, raw_host, raw_bytes) != 0);

    for (uint32_t t = 0; t < n_tokens; t++) {
        const uint32_t row = (pos0 + t) % raw_cap;
        for (uint32_t d = 0; d < head_dim; d++) {
            const float ref = kv_host[(uint64_t)t * head_dim + d];
            const float got = raw_host[(uint64_t)row * head_dim + d];
            TEST_ASSERT(got == ref);
        }
    }
    for (uint32_t d = 0; d < head_dim; d++) {
        TEST_ASSERT(raw_host[(uint64_t)2u * head_dim + d] == -1.0f);
    }

    ds4_gpu_tensor_free(kv);
    ds4_gpu_tensor_free(raw);
}

#if defined(__APPLE__)
typedef struct {
    volatile uint32_t *sequence_count;
    volatile uint32_t *sequence;
    volatile uint32_t  calls;
    uint32_t           marker;
} test_metal_progress_callback_ctx;

static void test_metal_progress_callback(void *opaque) {
    test_metal_progress_callback_ctx *ctx = opaque;
    const uint32_t slot = __atomic_fetch_add(
        ctx->sequence_count, 1u, __ATOMIC_ACQ_REL);
    if (slot < 2u) {
        __atomic_store_n(
            ctx->sequence + slot, ctx->marker, __ATOMIC_RELEASE);
    }
    (void)__atomic_fetch_add(&ctx->calls, 1u, __ATOMIC_ACQ_REL);
}

static void test_metal_flush_commands_progress_exact(void) {
    enum {
        n = 257,
        guard = 19,
        alloc_n = n + guard,
    };
    const uint64_t active_bytes = (uint64_t)n * sizeof(float);
    const uint64_t alloc_bytes = (uint64_t)alloc_n * sizeof(float);
    const uint32_t tmp_poison = 0x7fc1a500u;
    const uint32_t out_poison = 0x7fc25a00u;
    float a_host[alloc_n];
    float b_host[alloc_n];
    float c_host[alloc_n];
    float tmp_host[alloc_n];
    float out_host[alloc_n];
    float expected_tmp[n];
    float expected_out[n];

    for (uint32_t i = 0; i < alloc_n; i++) {
        a_host[i] = (float)((int)(i % 17u) - 8) * 0.5f;
        b_host[i] = (float)((int)((i * 5u) % 23u) - 11) * 0.25f;
        c_host[i] = (float)((int)((i * 7u) % 29u) - 14) * 0.125f;
        const uint32_t tmp_bits = tmp_poison + (i & 0xffu);
        const uint32_t out_bits = out_poison + (i & 0xffu);
        memcpy(tmp_host + i, &tmp_bits, sizeof(tmp_bits));
        memcpy(out_host + i, &out_bits, sizeof(out_bits));
        if (i < n) {
            expected_tmp[i] = a_host[i] + b_host[i];
            expected_out[i] = expected_tmp[i] + c_host[i];
        }
    }

    ds4_gpu_tensor *a = ds4_gpu_tensor_alloc(alloc_bytes);
    ds4_gpu_tensor *b = ds4_gpu_tensor_alloc(alloc_bytes);
    ds4_gpu_tensor *c = ds4_gpu_tensor_alloc(alloc_bytes);
    ds4_gpu_tensor *tmp_base = ds4_gpu_tensor_alloc(alloc_bytes);
    ds4_gpu_tensor *out_base = ds4_gpu_tensor_alloc(alloc_bytes);
    ds4_gpu_tensor *tmp = tmp_base
        ? ds4_gpu_tensor_view(tmp_base, 0u, active_bytes) : NULL;
    ds4_gpu_tensor *out = out_base
        ? ds4_gpu_tensor_view(out_base, 0u, active_bytes) : NULL;
    TEST_ASSERT(a && b && c && tmp_base && out_base && tmp && out);

    volatile uint32_t sequence_count = 0u;
    volatile uint32_t sequence[2] = {0u, 0u};
    test_metal_progress_callback_ctx callback0 = {
        .sequence_count = &sequence_count,
        .sequence = sequence,
        .calls = 0u,
        .marker = 1u,
    };
    test_metal_progress_callback_ctx callback1 = {
        .sequence_count = &sequence_count,
        .sequence = sequence,
        .calls = 0u,
        .marker = 2u,
    };

    bool submitted = false;
    if (a && b && c && tmp_base && out_base && tmp && out) {
        TEST_ASSERT(ds4_gpu_tensor_write(a, 0u, a_host, alloc_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(b, 0u, b_host, alloc_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(c, 0u, c_host, alloc_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        tmp_base, 0u, tmp_host, alloc_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        out_base, 0u, out_host, alloc_bytes) != 0);
        TEST_ASSERT(ds4_gpu_commands_active() == 0);

        const int begun = ds4_gpu_begin_commands();
        TEST_ASSERT(begun != 0);
        const int first = begun
            ? ds4_gpu_add_tensor(tmp, a, b, n) : 0;
        TEST_ASSERT(first != 0);
        const int flushed0 = first
            ? ds4_gpu_flush_commands_progress(
                  test_metal_progress_callback, &callback0)
            : 0;
        TEST_ASSERT(flushed0 != 0);
        const int second = flushed0
            ? ds4_gpu_add_tensor(out, tmp, c, n) : 0;
        TEST_ASSERT(second != 0);
        const int flushed1 = second
            ? ds4_gpu_flush_commands_progress(
                  test_metal_progress_callback, &callback1)
            : 0;
        TEST_ASSERT(flushed1 != 0);
        submitted = flushed1 != 0;

        const int drained = ds4_gpu_commands_active()
            ? ds4_gpu_end_commands() : 0;
        TEST_ASSERT(drained != 0);
        TEST_ASSERT(ds4_gpu_commands_active() == 0);
    }

    test_float_compare_stats tmp_stats = {0};
    test_float_compare_stats out_stats = {0};
    size_t guard_mismatches = 0u;
    if (submitted && tmp_base && out_base) {
        TEST_ASSERT(ds4_gpu_tensor_read(
                        tmp_base, 0u, tmp_host, alloc_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        out_base, 0u, out_host, alloc_bytes) != 0);
        tmp_stats = test_compare_float_bits(expected_tmp, tmp_host, n);
        out_stats = test_compare_float_bits(expected_out, out_host, n);
        for (uint32_t i = n; i < alloc_n; i++) {
            uint32_t tmp_bits = 0u;
            uint32_t out_bits = 0u;
            memcpy(&tmp_bits, tmp_host + i, sizeof(tmp_bits));
            memcpy(&out_bits, out_host + i, sizeof(out_bits));
            if (tmp_bits != tmp_poison + (i & 0xffu) ||
                out_bits != out_poison + (i & 0xffu)) {
                guard_mismatches++;
            }
        }
    }

    const uint32_t calls0 =
        __atomic_load_n(&callback0.calls, __ATOMIC_ACQUIRE);
    const uint32_t calls1 =
        __atomic_load_n(&callback1.calls, __ATOMIC_ACQUIRE);
    const uint32_t completed =
        __atomic_load_n(&sequence_count, __ATOMIC_ACQUIRE);
    const uint32_t observed0 =
        __atomic_load_n(sequence + 0u, __ATOMIC_ACQUIRE);
    const uint32_t observed1 =
        __atomic_load_n(sequence + 1u, __ATOMIC_ACQUIRE);
    const bool distinct_contexts =
        (observed0 == 1u && observed1 == 2u) ||
        (observed0 == 2u && observed1 == 1u);
    fprintf(stderr,
            "ds4-test: Metal progress flush exact callbacks=%u/%u "
            "completed=%u order=%u,%u tmp=%zu out=%zu guard=%zu\n",
            calls0, calls1, completed, observed0, observed1,
            tmp_stats.mismatch_count, out_stats.mismatch_count,
            guard_mismatches);
    TEST_ASSERT(calls0 == 1u);
    TEST_ASSERT(calls1 == 1u);
    TEST_ASSERT(completed == 2u);
    /* Completion-handler ordering across different command buffers is not a
     * public contract.  The dependent tensor result proves GPU submission
     * order; here only prove that both distinct contexts were delivered. */
    TEST_ASSERT(distinct_contexts);
    TEST_ASSERT(tmp_stats.mismatch_count == 0u && tmp_stats.max_ulp == 0u);
    TEST_ASSERT(out_stats.mismatch_count == 0u && out_stats.max_ulp == 0u);
    TEST_ASSERT(guard_mismatches == 0u);

    if (ds4_gpu_commands_active()) {
        TEST_ASSERT(ds4_gpu_end_commands() != 0);
    }
    ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(tmp);
    ds4_gpu_tensor_free(out_base);
    ds4_gpu_tensor_free(tmp_base);
    ds4_gpu_tensor_free(c);
    ds4_gpu_tensor_free(b);
    ds4_gpu_tensor_free(a);
}
#endif

static void test_dspark_cache_window_crop(void) {
    TEST_ASSERT(ds4_test_dspark_cache_window_crop());
}

static void test_metal_q8_0_decode_pair_exact_case(
        uint32_t out0_dim,
        uint32_t out1_dim,
        uint32_t seed0,
        uint32_t seed1) {
    /* Exercise the Q-A/KV contract with unequal, odd output extents and
     * independently page-aligned model ranges. The paired kernel must be
     * bit-identical to two standalone decode matvec dispatches. */
    const uint32_t in_dim = 4096;
    const uint64_t page = (uint64_t)getpagesize();
    const uint64_t row_bytes = (uint64_t)(in_dim / 32u) * 34u;
    const uint64_t weight0_bytes = (uint64_t)out0_dim * row_bytes;
    const uint64_t weight1_bytes = (uint64_t)out1_dim * row_bytes;
    const uint64_t weight1_offset = test_round_up_u64(weight0_bytes, page);
    const uint64_t weight_alloc =
        test_round_up_u64(weight1_offset + weight1_bytes, page);

    void *weights_raw = NULL;
    TEST_ASSERT(posix_memalign(&weights_raw, (size_t)page, (size_t)weight_alloc) == 0);
    if (!weights_raw) return;
    memset(weights_raw, 0, (size_t)weight_alloc);
    test_fill_q8_0_weights((uint8_t *)weights_raw, in_dim, out0_dim, seed0);
    test_fill_q8_0_weights((uint8_t *)weights_raw + weight1_offset,
                           in_dim, out1_dim, seed1);

    const uint64_t x_bytes = (uint64_t)in_dim * sizeof(float);
    const uint64_t out0_bytes = (uint64_t)out0_dim * sizeof(float);
    const uint64_t out1_bytes = (uint64_t)out1_dim * sizeof(float);
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(x_bytes);
    ds4_gpu_tensor *ref0 = ds4_gpu_tensor_alloc(out0_bytes);
    ds4_gpu_tensor *ref1 = ds4_gpu_tensor_alloc(out1_bytes);
    ds4_gpu_tensor *pair0 = ds4_gpu_tensor_alloc(out0_bytes);
    ds4_gpu_tensor *pair1 = ds4_gpu_tensor_alloc(out1_bytes);
    TEST_ASSERT(x != NULL);
    TEST_ASSERT(ref0 != NULL);
    TEST_ASSERT(ref1 != NULL);
    TEST_ASSERT(pair0 != NULL);
    TEST_ASSERT(pair1 != NULL);
    if (!x || !ref0 || !ref1 || !pair0 || !pair1) {
        ds4_gpu_tensor_free(x);
        ds4_gpu_tensor_free(ref0);
        ds4_gpu_tensor_free(ref1);
        ds4_gpu_tensor_free(pair0);
        ds4_gpu_tensor_free(pair1);
        free(weights_raw);
        return;
    }

    float *x_host = malloc((size_t)x_bytes);
    float *ref0_host = malloc((size_t)out0_bytes);
    float *ref1_host = malloc((size_t)out1_bytes);
    float *pair0_host = malloc((size_t)out0_bytes);
    float *pair1_host = malloc((size_t)out1_bytes);
    TEST_ASSERT(x_host != NULL);
    TEST_ASSERT(ref0_host != NULL);
    TEST_ASSERT(ref1_host != NULL);
    TEST_ASSERT(pair0_host != NULL);
    TEST_ASSERT(pair1_host != NULL);
    if (!x_host || !ref0_host || !ref1_host || !pair0_host || !pair1_host) {
        free(x_host);
        free(ref0_host);
        free(ref1_host);
        free(pair0_host);
        free(pair1_host);
        ds4_gpu_tensor_free(x);
        ds4_gpu_tensor_free(ref0);
        ds4_gpu_tensor_free(ref1);
        ds4_gpu_tensor_free(pair0);
        ds4_gpu_tensor_free(pair1);
        free(weights_raw);
        return;
    }

    for (uint32_t i = 0; i < in_dim; i++) {
        const int v = (int)((i * 29u + (i ^ (i >> 3u)) * 7u) % 127u) - 63;
        x_host[i] = (float)v / 72.0f;
    }

    TEST_ASSERT(ds4_gpu_tensor_write(x, 0, x_host, x_bytes) != 0);
    TEST_ASSERT(ds4_gpu_set_model_map(weights_raw, weight_alloc) != 0);
    ds4_gpu_set_quality(false);
    TEST_ASSERT(ds4_gpu_matmul_q8_0_tensor(ref0, weights_raw, weight_alloc, 0,
                                           in_dim, out0_dim, x, 1) != 0);
    TEST_ASSERT(ds4_gpu_matmul_q8_0_tensor(ref1, weights_raw, weight_alloc,
                                           weight1_offset,
                                           in_dim, out1_dim, x, 1) != 0);
    TEST_ASSERT(ds4_gpu_matmul_q8_0_pair_tensor(pair0, pair1,
                                                weights_raw, weight_alloc,
                                                0, weight1_offset,
                                                in_dim, out0_dim, out1_dim,
                                                x, 1) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(ref0, 0, ref0_host, out0_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(ref1, 0, ref1_host, out1_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(pair0, 0, pair0_host, out0_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(pair1, 0, pair1_host, out1_bytes) != 0);

    uint32_t mismatch0 = 0;
    uint32_t mismatch1 = 0;
    float max_abs0 = 0.0f;
    float max_abs1 = 0.0f;
    for (uint32_t i = 0; i < out0_dim; i++) {
        if (memcmp(&ref0_host[i], &pair0_host[i], sizeof(float)) != 0) mismatch0++;
        const float err = fabsf(ref0_host[i] - pair0_host[i]);
        if (err > max_abs0) max_abs0 = err;
    }
    for (uint32_t i = 0; i < out1_dim; i++) {
        if (memcmp(&ref1_host[i], &pair1_host[i], sizeof(float)) != 0) mismatch1++;
        const float err = fabsf(ref1_host[i] - pair1_host[i]);
        if (err > max_abs1) max_abs1 = err;
    }
    if (mismatch0 != 0 || mismatch1 != 0) {
        fprintf(stderr,
                "ds4-test: paired Q8_0 exactness mismatches=%u/%u max_abs=%g, %u/%u max_abs=%g\n",
                mismatch0, out0_dim, max_abs0,
                mismatch1, out1_dim, max_abs1);
    }
    TEST_ASSERT(mismatch0 == 0);
    TEST_ASSERT(mismatch1 == 0);

    free(x_host);
    free(ref0_host);
    free(ref1_host);
    free(pair0_host);
    free(pair1_host);
    ds4_gpu_tensor_free(x);
    ds4_gpu_tensor_free(ref0);
    ds4_gpu_tensor_free(ref1);
    ds4_gpu_tensor_free(pair0);
    ds4_gpu_tensor_free(pair1);
    free(weights_raw);
}

static void test_metal_q8_0_decode_pair_exact(void) {
    /* Cover both possible one-bank tail directions. Distinct seeds ensure a
     * mistaken A-for-B weight binding cannot compare equal by construction. */
    test_metal_q8_0_decode_pair_exact_case(77, 19, 11, 97);
    test_metal_q8_0_decode_pair_exact_case(19, 77, 23, 131);
}

#if defined(__APPLE__)
static void test_fill_q4_K_weights(uint8_t *weights,
                                   uint32_t in_dim,
                                   uint32_t out_dim,
                                   uint32_t seed) {
    /* GGUF Q4_K block: f16 d, f16 dmin, 12 packed scales, 128 quants. */
    const uint32_t block_elems = 256u;
    const uint32_t block_bytes = 144u;
    TEST_ASSERT(weights != NULL);
    TEST_ASSERT((in_dim % block_elems) == 0u);
    if (!weights || (in_dim % block_elems) != 0u) return;

    const uint32_t blocks_per_row = in_dim / block_elems;
    for (uint32_t row = 0; row < out_dim; row++) {
        for (uint32_t block = 0; block < blocks_per_row; block++) {
            uint8_t *q = weights +
                ((uint64_t)row * blocks_per_row + block) * block_bytes;
            const uint32_t key =
                seed + row * 1009u + block * 313u + (row ^ (block * 17u));
            const uint16_t d = test_float_to_f16(
                0.0025f + (float)(key % 13u) / 4096.0f);
            const uint16_t dmin = test_float_to_f16(
                0.0010f + (float)((key >> 3u) % 7u) / 8192.0f);
            memcpy(q + 0u, &d, sizeof(d));
            memcpy(q + 2u, &dmin, sizeof(dmin));
            for (uint32_t i = 0; i < 12u; i++) {
                q[4u + i] = (uint8_t)(
                    1u + ((key + i * 23u + (i ^ row) * 5u) % 0xfeu));
            }
            for (uint32_t i = 0; i < 128u; i++) {
                q[16u + i] = (uint8_t)(
                    key + i * 37u + (i >> 2u) * 11u + row * 3u);
            }
        }
    }
}

static void test_metal_q8_0_decode_rows_exact(void) {
    /* The exact-N output head must preserve the canonical one-row Q8_0
     * arithmetic for every speculative row, including the output tail. */
    const uint32_t in_dim = 4096u;
    const uint32_t out_dim = 259u;
    const uint32_t max_rows = 5u;
    const uint64_t page = (uint64_t)getpagesize();
    const uint64_t row_bytes = (uint64_t)(in_dim / 32u) * 34u;
    const uint64_t weight_bytes = (uint64_t)out_dim * row_bytes;
    const uint64_t weight_alloc = test_round_up_u64(weight_bytes, page);
    const uint64_t x_row_bytes = (uint64_t)in_dim * sizeof(float);
    const uint64_t out_row_bytes = (uint64_t)out_dim * sizeof(float);
    const uint64_t x_bytes = (uint64_t)max_rows * x_row_bytes;
    const uint64_t out_bytes = (uint64_t)max_rows * out_row_bytes;

    void *weights_raw = NULL;
    TEST_ASSERT(posix_memalign(&weights_raw,
                               (size_t)page,
                               (size_t)weight_alloc) == 0);
    if (!weights_raw) return;
    memset(weights_raw, 0, (size_t)weight_alloc);
    test_fill_q8_0_weights(weights_raw, in_dim, out_dim, 173u);

    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(x_bytes);
    ds4_gpu_tensor *reference = ds4_gpu_tensor_alloc(out_bytes);
    ds4_gpu_tensor *batched = ds4_gpu_tensor_alloc(out_bytes);
    TEST_ASSERT(x != NULL);
    TEST_ASSERT(reference != NULL);
    TEST_ASSERT(batched != NULL);
    if (!x || !reference || !batched) {
        ds4_gpu_tensor_free(x);
        ds4_gpu_tensor_free(reference);
        ds4_gpu_tensor_free(batched);
        free(weights_raw);
        return;
    }

    float *x_host = malloc((size_t)x_bytes);
    float *reference_host = malloc((size_t)out_bytes);
    float *batched_host = malloc((size_t)out_bytes);
    TEST_ASSERT(x_host != NULL);
    TEST_ASSERT(reference_host != NULL);
    TEST_ASSERT(batched_host != NULL);
    if (!x_host || !reference_host || !batched_host) {
        free(x_host);
        free(reference_host);
        free(batched_host);
        ds4_gpu_tensor_free(x);
        ds4_gpu_tensor_free(reference);
        ds4_gpu_tensor_free(batched);
        free(weights_raw);
        return;
    }

    for (uint32_t row = 0; row < max_rows; row++) {
        for (uint32_t i = 0; i < in_dim; i++) {
            const uint32_t mix =
                i * 29u + row * 101u + ((i + row * 17u) ^ (i >> 3u)) * 7u;
            x_host[(uint64_t)row * in_dim + i] =
                (float)((int)(mix % 191u) - 95) / 113.0f;
        }
    }

    TEST_ASSERT(ds4_gpu_tensor_write(x, 0, x_host, x_bytes) != 0);
    TEST_ASSERT(ds4_gpu_set_model_map(weights_raw, weight_alloc) != 0);
    ds4_gpu_set_quality(false);

    for (uint32_t row = 0; row < max_rows; row++) {
        ds4_gpu_tensor *x_row = ds4_gpu_tensor_view(
                x, (uint64_t)row * x_row_bytes, x_row_bytes);
        ds4_gpu_tensor *out_row = ds4_gpu_tensor_view(
                reference, (uint64_t)row * out_row_bytes, out_row_bytes);
        TEST_ASSERT(x_row != NULL);
        TEST_ASSERT(out_row != NULL);
        if (x_row && out_row) {
            TEST_ASSERT(ds4_gpu_matmul_q8_0_tensor(
                            out_row, weights_raw, weight_alloc, 0,
                            in_dim, out_dim, x_row, 1) != 0);
        }
        ds4_gpu_tensor_free(x_row);
        ds4_gpu_tensor_free(out_row);
    }
    TEST_ASSERT(ds4_gpu_tensor_read(
                    reference, 0, reference_host, out_bytes) != 0);

    for (uint32_t n_rows = 2u; n_rows <= max_rows; n_rows++) {
        memset(batched_host, 0xa5, (size_t)out_bytes);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        batched, 0, batched_host, out_bytes) != 0);
        TEST_ASSERT(ds4_gpu_matmul_q8_0_decode_rows_exact_tensor(
                        batched, weights_raw, weight_alloc, 0,
                        in_dim, out_dim, x, n_rows) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        batched, 0, batched_host, out_bytes) != 0);

        const size_t compared = (size_t)n_rows * out_dim;
        const test_float_compare_stats stats = test_compare_float_bits(
                reference_host, batched_host, compared);
        if (stats.mismatch_count != 0) {
            fprintf(stderr,
                    "ds4-test: Metal Q8_0 exact-%u rows mismatches=%zu/%zu "
                    "max_ulp=%u max_abs=%g\n",
                    n_rows,
                    stats.mismatch_count,
                    compared,
                    stats.max_ulp,
                    stats.max_abs);
        }
        TEST_ASSERT(memcmp(reference_host,
                           batched_host,
                           compared * sizeof(float)) == 0);
    }

    free(x_host);
    free(reference_host);
    free(batched_host);
    ds4_gpu_tensor_free(x);
    ds4_gpu_tensor_free(reference);
    ds4_gpu_tensor_free(batched);
    free(weights_raw);
}

static void test_fill_q8_0_constant_weights(uint8_t *weights,
                                             uint32_t in_dim,
                                             uint32_t out_dim,
                                             int8_t quant) {
    const uint32_t blocks = in_dim / 32u;
    const uint64_t row_bytes = (uint64_t)blocks * 34u;
    const uint16_t scale_bits = test_float_to_f16(1.0f / 256.0f);
    for (uint32_t row = 0; row < out_dim; row++) {
        uint8_t *dst = weights + (uint64_t)row * row_bytes;
        for (uint32_t block = 0; block < blocks; block++) {
            memcpy(dst + (uint64_t)block * 34u,
                   &scale_bits,
                   sizeof(scale_bits));
            memset(dst + (uint64_t)block * 34u + 2u,
                   (unsigned char)quant,
                   32u);
        }
    }
}

static bool test_metal_q8_attention_output_static_batch_exact_case(
        uint32_t n_tokens) {
    const int failures_before = test_failures;
    /*
     * The production AProjQ8 static kernel receives a flattened z coordinate:
     *
     *     pair = token * n_groups + group
     *
     * Only pair % n_groups may select Woa.  Keep a second, sign-inverted Woa
     * immediately after the real one so the old pair-as-group bug is a safe,
     * deterministic wrong read for token 1 instead of an out-of-bounds Metal
     * access.  The public batch API also runs the small Q8 output projection;
     * both its low intermediate and final output must match the generic direct
     * kernel bit for bit.
     */
    const uint32_t group_dim = 4096u;
    const uint32_t rank = 1024u;
    const uint32_t n_groups = 8u;
    const uint32_t low_dim = n_groups * rank;
    const uint32_t out_dim = 32u;
    const uint32_t alloc_tokens = n_tokens + 1u;
    const uint64_t page = (uint64_t)getpagesize();
    const uint64_t row_a_bytes = (uint64_t)(group_dim / 32u) * 34u;
    const uint64_t group_a_bytes = (uint64_t)rank * row_a_bytes;
    const uint64_t out_a_bytes = (uint64_t)n_groups * group_a_bytes;
    const uint64_t shadow_a_offset = out_a_bytes;
    const uint64_t out_b_offset = 2u * out_a_bytes;
    const uint64_t row_b_bytes = (uint64_t)(low_dim / 32u) * 34u;
    const uint64_t out_b_bytes = (uint64_t)out_dim * row_b_bytes;
    const uint64_t model_bytes =
        test_round_up_u64(out_b_offset + out_b_bytes, page);
    const uint64_t heads_bytes =
        (uint64_t)alloc_tokens * n_groups * group_dim * sizeof(float);
    const uint64_t low_bytes =
        (uint64_t)alloc_tokens * low_dim * sizeof(float);
    const uint64_t out_bytes =
        (uint64_t)alloc_tokens * out_dim * sizeof(float);
    const uint64_t active_low_bytes =
        (uint64_t)n_tokens * low_dim * sizeof(float);
    const uint64_t active_out_bytes =
        (uint64_t)n_tokens * out_dim * sizeof(float);
    const char *disable_direct_env =
        "DS4_METAL_DISABLE_ATTN_OUT_LOW_DIRECT";
    const char *disable_ports_env =
        "DS4_METAL_DISABLE_PRE_M5_DECODE_PORTS";
    const char *disable_static_env =
        "DS4_METAL_DISABLE_PRE_M5_ATTN_OUT_LOW_Q8_STATIC";

    _Static_assert(4096u / 32u * 34u == 4352u,
                   "production AProjQ8 row size changed");
    _Static_assert(8u * 1024u * 4352u == 35651584u,
                   "production AProjQ8 Woa size changed");

    char *saved_disable_direct = test_save_env(disable_direct_env);
    char *saved_disable_ports = test_save_env(disable_ports_env);
    char *saved_disable_static = test_save_env(disable_static_env);
    void *model_raw = NULL;
    float *heads_host = NULL;
    float *reference_low_host = NULL;
    float *candidate_low_host = NULL;
    float *reference_out_host = NULL;
    float *candidate_out_host = NULL;
    ds4_gpu_tensor *heads = NULL;
    ds4_gpu_tensor *reference_low = NULL;
    ds4_gpu_tensor *candidate_low = NULL;
    ds4_gpu_tensor *reference_out = NULL;
    ds4_gpu_tensor *candidate_out = NULL;
    ds4_gpu_tensor *group_tmp = NULL;
    ds4_gpu_tensor *low_tmp = NULL;

    TEST_ASSERT(row_a_bytes == 4352u);
    TEST_ASSERT(out_a_bytes == 35651584u);
    TEST_ASSERT(posix_memalign(
                    &model_raw, (size_t)page, (size_t)model_bytes) == 0);
    if (!model_raw) goto cleanup;
    memset(model_raw, 0, (size_t)model_bytes);
    for (uint32_t group = 0; group < n_groups; group++) {
        test_fill_q8_0_constant_weights(
            (uint8_t *)model_raw + (uint64_t)group * group_a_bytes,
            group_dim,
            rank,
            (int8_t)(group + 1u));
        test_fill_q8_0_constant_weights(
            (uint8_t *)model_raw + shadow_a_offset +
                (uint64_t)group * group_a_bytes,
            group_dim,
            rank,
            (int8_t)-(int8_t)(group + 1u));
    }
    test_fill_q8_0_constant_weights(
        (uint8_t *)model_raw + out_b_offset,
        low_dim,
        out_dim,
        1);

    heads_host = malloc((size_t)heads_bytes);
    reference_low_host = malloc((size_t)low_bytes);
    candidate_low_host = malloc((size_t)low_bytes);
    reference_out_host = malloc((size_t)out_bytes);
    candidate_out_host = malloc((size_t)out_bytes);
    heads = ds4_gpu_tensor_alloc(heads_bytes);
    reference_low = ds4_gpu_tensor_alloc(low_bytes);
    candidate_low = ds4_gpu_tensor_alloc(low_bytes);
    reference_out = ds4_gpu_tensor_alloc(out_bytes);
    candidate_out = ds4_gpu_tensor_alloc(out_bytes);
    group_tmp = ds4_gpu_tensor_alloc(
        (uint64_t)n_tokens * group_dim * sizeof(float));
    low_tmp = ds4_gpu_tensor_alloc(
        (uint64_t)n_tokens * rank * sizeof(float));
    TEST_ASSERT(heads_host && reference_low_host && candidate_low_host &&
                reference_out_host && candidate_out_host && heads &&
                reference_low && candidate_low && reference_out &&
                candidate_out && group_tmp && low_tmp);
    if (!heads_host || !reference_low_host || !candidate_low_host ||
        !reference_out_host || !candidate_out_host || !heads ||
        !reference_low || !candidate_low || !reference_out ||
        !candidate_out || !group_tmp || !low_tmp) {
        goto cleanup;
    }

    for (uint64_t i = 0; i < heads_bytes / sizeof(float); i++) {
        const uint32_t token = (uint32_t)(i / ((uint64_t)n_groups * group_dim));
        const uint32_t key =
            (uint32_t)i * 17u + token * 131u + ((uint32_t)i >> 4u);
        heads_host[i] = (float)(1u + key % 13u) / 128.0f;
    }
    memset(reference_low_host, 0xa5, (size_t)low_bytes);
    memset(candidate_low_host, 0xa5, (size_t)low_bytes);
    memset(reference_out_host, 0xa5, (size_t)out_bytes);
    memset(candidate_out_host, 0xa5, (size_t)out_bytes);
    TEST_ASSERT(ds4_gpu_tensor_write(
                    heads, 0, heads_host, heads_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(
                    reference_low, 0, reference_low_host, low_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(
                    candidate_low, 0, candidate_low_host, low_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(
                    reference_out, 0, reference_out_host, out_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(
                    candidate_out, 0, candidate_out_host, out_bytes) != 0);
    TEST_ASSERT(ds4_gpu_set_model_map(model_raw, model_bytes) != 0);
    ds4_gpu_set_quality(false);

    TEST_ASSERT(unsetenv(disable_direct_env) == 0);
    TEST_ASSERT(unsetenv(disable_ports_env) == 0);
    TEST_ASSERT(unsetenv(disable_static_env) == 0);
    ds4_gpu_test_set_flags(DS4_GPU_TEST_ATTN_OUT_LOW_Q8_STATIC);
    TEST_ASSERT(ds4_gpu_attention_output_q8_batch_tensor(
                    candidate_out,
                    candidate_low,
                    group_tmp,
                    low_tmp,
                    model_raw,
                    model_bytes,
                    0,
                    out_b_offset,
                    group_dim,
                    rank,
                    n_groups,
                    out_dim,
                    heads,
                    n_tokens) == 1);

    ds4_gpu_test_set_flags(0);
    TEST_ASSERT(setenv(disable_static_env, "1", 1) == 0);
    TEST_ASSERT(ds4_gpu_attention_output_q8_batch_tensor(
                    reference_out,
                    reference_low,
                    group_tmp,
                    low_tmp,
                    model_raw,
                    model_bytes,
                    0,
                    out_b_offset,
                    group_dim,
                    rank,
                    n_groups,
                    out_dim,
                    heads,
                    n_tokens) == 1);

    TEST_ASSERT(ds4_gpu_tensor_read(
                    reference_low, 0, reference_low_host, low_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(
                    candidate_low, 0, candidate_low_host, low_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(
                    reference_out, 0, reference_out_host, out_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(
                    candidate_out, 0, candidate_out_host, out_bytes) != 0);

    const size_t low_count = (size_t)n_tokens * low_dim;
    const size_t out_count = (size_t)n_tokens * out_dim;
    const test_float_compare_stats low_stats = test_compare_float_bits(
        reference_low_host, candidate_low_host, low_count);
    const test_float_compare_stats out_stats = test_compare_float_bits(
        reference_out_host, candidate_out_host, out_count);
    size_t tail_byte_mismatch = 0;
    for (uint64_t i = active_low_bytes; i < low_bytes; i++) {
        if (((const uint8_t *)reference_low_host)[i] != 0xa5u) {
            tail_byte_mismatch++;
        }
        if (((const uint8_t *)candidate_low_host)[i] != 0xa5u) {
            tail_byte_mismatch++;
        }
    }
    for (uint64_t i = active_out_bytes; i < out_bytes; i++) {
        if (((const uint8_t *)reference_out_host)[i] != 0xa5u) {
            tail_byte_mismatch++;
        }
        if (((const uint8_t *)candidate_out_host)[i] != 0xa5u) {
            tail_byte_mismatch++;
        }
    }
    fprintf(stderr,
            "ds4-test: Metal Q8 static attention-output exact-%u "
            "low=%zu/%zu max_ulp=%u max_abs=%g "
            "out=%zu/%zu max_ulp=%u max_abs=%g tail_bytes=%zu\n",
            n_tokens,
            low_stats.mismatch_count,
            low_count,
            low_stats.max_ulp,
            low_stats.max_abs,
            out_stats.mismatch_count,
            out_count,
            out_stats.max_ulp,
            out_stats.max_abs,
            tail_byte_mismatch);
    TEST_ASSERT(low_stats.mismatch_count == 0 && low_stats.max_ulp == 0);
    TEST_ASSERT(out_stats.mismatch_count == 0 && out_stats.max_ulp == 0);
    TEST_ASSERT(tail_byte_mismatch == 0);

cleanup:
    ds4_gpu_test_set_flags(0);
    ds4_gpu_tensor_free(low_tmp);
    ds4_gpu_tensor_free(group_tmp);
    ds4_gpu_tensor_free(candidate_out);
    ds4_gpu_tensor_free(reference_out);
    ds4_gpu_tensor_free(candidate_low);
    ds4_gpu_tensor_free(reference_low);
    ds4_gpu_tensor_free(heads);
    free(candidate_out_host);
    free(reference_out_host);
    free(candidate_low_host);
    free(reference_low_host);
    free(heads_host);
    free(model_raw);
    test_restore_env(disable_static_env, saved_disable_static);
    test_restore_env(disable_ports_env, saved_disable_ports);
    test_restore_env(disable_direct_env, saved_disable_direct);
    return test_failures == failures_before;
}

static void test_metal_q8_attention_output_static_batch_exact(void) {
    /* The shadow Woa makes the old bug safe at N=2; stop there on failure. */
    if (test_metal_q8_attention_output_static_batch_exact_case(2u)) {
        (void)test_metal_q8_attention_output_static_batch_exact_case(31u);
    }
}

static void test_metal_q4_attention_output_tiny_batch_exact_case(
        uint32_t out_b_type) {
    /*
     * The opt-in AProjQ4 tiny batch must be the exact composition of the two
     * canonical one-row projections for both deployed output quantizations.
     * Keep a sixth poisoned row in every destination so each N=2..5 case also
     * checks dispatch bounds.
     */
    TEST_ASSERT(out_b_type == 8u || out_b_type == 12u);
    if (out_b_type != 8u && out_b_type != 12u) return;
    const uint32_t group_dim = 256u;
    const uint32_t rank = 128u;
    const uint32_t n_groups = 2u;
    const uint32_t low_dim = n_groups * rank;
    const uint32_t out_dim = 67u;
    const uint32_t max_rows = 5u;
    const uint32_t alloc_rows = max_rows + 1u;
    const uint64_t page = (uint64_t)getpagesize();
    const uint64_t row_a_bytes = (uint64_t)(group_dim / 256u) * 144u;
    const uint64_t out_a_bytes =
        (uint64_t)n_groups * rank * row_a_bytes;
    const uint64_t out_b_offset = test_round_up_u64(out_a_bytes, page);
    const uint64_t row_b_bytes = out_b_type == 8u
        ? (uint64_t)(low_dim / 32u) * 34u
        : (uint64_t)(low_dim / 256u) * 144u;
    const uint64_t out_b_bytes = (uint64_t)out_dim * row_b_bytes;
    const uint64_t model_bytes =
        test_round_up_u64(out_b_offset + out_b_bytes, page);
    const uint64_t heads_row_bytes =
        (uint64_t)n_groups * group_dim * sizeof(float);
    const uint64_t low_row_bytes = (uint64_t)low_dim * sizeof(float);
    const uint64_t out_row_bytes = (uint64_t)out_dim * sizeof(float);
    const uint64_t heads_bytes = (uint64_t)alloc_rows * heads_row_bytes;
    const uint64_t low_bytes = (uint64_t)alloc_rows * low_row_bytes;
    const uint64_t out_bytes = (uint64_t)alloc_rows * out_row_bytes;
    const char *enable_env =
        "DS4_METAL_ENABLE_Q4_ATTN_OUT_TINY_BATCH";
    const char *disable_env =
        "DS4_METAL_DISABLE_Q4_ATTN_OUT_TINY_BATCH";
    const char *require_env =
        "DS4_METAL_REQUIRE_Q4_ATTN_OUT_TINY_BATCH";
    const char *disable_classic_env =
        "DS4_METAL_DISABLE_Q4_MV_CLASSIC";

    char *saved_enable = test_save_env(enable_env);
    char *saved_disable = test_save_env(disable_env);
    char *saved_require = test_save_env(require_env);
    char *saved_disable_classic = test_save_env(disable_classic_env);
    void *model_raw = NULL;
    float *heads_host = NULL;
    float *reference_low_host = NULL;
    float *reference_out_host = NULL;
    float *candidate_low_host = NULL;
    float *candidate_out_host = NULL;
    ds4_gpu_tensor *heads = NULL;
    ds4_gpu_tensor *reference_low = NULL;
    ds4_gpu_tensor *reference_out = NULL;
    ds4_gpu_tensor *candidate_low = NULL;
    ds4_gpu_tensor *candidate_out = NULL;
    ds4_gpu_tensor *group_tmp = NULL;
    ds4_gpu_tensor *low_tmp = NULL;

    TEST_ASSERT(posix_memalign(
                    &model_raw, (size_t)page, (size_t)model_bytes) == 0);
    if (!model_raw) goto cleanup;
    memset(model_raw, 0, (size_t)model_bytes);
    test_fill_q4_K_weights((uint8_t *)model_raw,
                           group_dim,
                           n_groups * rank,
                           211u);
    if (out_b_type == 8u) {
        test_fill_q8_0_weights(
            (uint8_t *)model_raw + out_b_offset, low_dim, out_dim, 307u);
    } else {
        test_fill_q4_K_weights(
            (uint8_t *)model_raw + out_b_offset, low_dim, out_dim, 307u);
    }

    heads_host = malloc((size_t)heads_bytes);
    reference_low_host = malloc((size_t)low_bytes);
    reference_out_host = malloc((size_t)out_bytes);
    candidate_low_host = malloc((size_t)low_bytes);
    candidate_out_host = malloc((size_t)out_bytes);
    heads = ds4_gpu_tensor_alloc(heads_bytes);
    reference_low = ds4_gpu_tensor_alloc(low_bytes);
    reference_out = ds4_gpu_tensor_alloc(out_bytes);
    candidate_low = ds4_gpu_tensor_alloc(low_bytes);
    candidate_out = ds4_gpu_tensor_alloc(out_bytes);
    group_tmp = ds4_gpu_tensor_alloc(
        (uint64_t)alloc_rows * group_dim * sizeof(float));
    low_tmp = ds4_gpu_tensor_alloc(
        (uint64_t)alloc_rows * rank * sizeof(float));
    TEST_ASSERT(heads_host && reference_low_host && reference_out_host &&
                candidate_low_host && candidate_out_host && heads &&
                reference_low && reference_out && candidate_low &&
                candidate_out && group_tmp && low_tmp);
    if (!heads_host || !reference_low_host || !reference_out_host ||
        !candidate_low_host || !candidate_out_host || !heads ||
        !reference_low || !reference_out || !candidate_low ||
        !candidate_out || !group_tmp || !low_tmp) {
        goto cleanup;
    }

    for (uint32_t row = 0; row < alloc_rows; row++) {
        for (uint32_t i = 0; i < n_groups * group_dim; i++) {
            const uint32_t key =
                i * 41u + row * 271u + ((i >> 2u) ^ (row * 19u));
            heads_host[(uint64_t)row * n_groups * group_dim + i] =
                (float)((int)(key % 257u) - 128) / 137.0f;
        }
    }
    memset(reference_low_host, 0, (size_t)low_bytes);
    memset(reference_out_host, 0, (size_t)out_bytes);
    TEST_ASSERT(ds4_gpu_tensor_write(
                    heads, 0, heads_host, heads_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(
                    reference_low, 0, reference_low_host, low_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(
                    reference_out, 0, reference_out_host, out_bytes) != 0);
    TEST_ASSERT(ds4_gpu_set_model_map(model_raw, model_bytes) != 0);
    ds4_gpu_set_quality(false);
    TEST_ASSERT(unsetenv(disable_classic_env) == 0);

    for (uint32_t row = 0; row < max_rows; row++) {
        ds4_gpu_tensor *heads_row = ds4_gpu_tensor_view(
            heads, (uint64_t)row * heads_row_bytes, heads_row_bytes);
        ds4_gpu_tensor *low_row = ds4_gpu_tensor_view(
            reference_low, (uint64_t)row * low_row_bytes, low_row_bytes);
        ds4_gpu_tensor *out_row = ds4_gpu_tensor_view(
            reference_out, (uint64_t)row * out_row_bytes, out_row_bytes);
        TEST_ASSERT(heads_row && low_row && out_row);
        if (heads_row && low_row && out_row) {
            TEST_ASSERT(ds4_gpu_attention_output_low_q4_K_slice_tensor(
                            low_row,
                            model_raw,
                            model_bytes,
                            0,
                            group_dim,
                            rank,
                            0,
                            n_groups,
                            heads_row,
                            0) != 0);
            if (out_b_type == 8u) {
                TEST_ASSERT(ds4_gpu_matmul_q8_0_tensor(
                                out_row,
                                model_raw,
                                model_bytes,
                                out_b_offset,
                                low_dim,
                                out_dim,
                                low_row,
                                1) != 0);
            } else {
                TEST_ASSERT(ds4_gpu_matmul_quant_tensor(
                                out_row,
                                model_raw,
                                model_bytes,
                                out_b_offset,
                                out_b_type,
                                low_dim,
                                out_dim,
                                low_row,
                                1) != 0);
            }
        }
        ds4_gpu_tensor_free(heads_row);
        ds4_gpu_tensor_free(low_row);
        ds4_gpu_tensor_free(out_row);
    }
    TEST_ASSERT(ds4_gpu_tensor_read(
                    reference_low, 0, reference_low_host, low_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(
                    reference_out, 0, reference_out_host, out_bytes) != 0);

    /* The API itself is fail-closed; the graph caller owns the row fallback. */
    TEST_ASSERT(unsetenv(enable_env) == 0);
    TEST_ASSERT(unsetenv(disable_env) == 0);
    TEST_ASSERT(unsetenv(require_env) == 0);
    TEST_ASSERT(ds4_gpu_attention_output_q4_K_batch_tensor(
                    candidate_out,
                    candidate_low,
                    group_tmp,
                    low_tmp,
                    model_raw,
                    model_bytes,
                    0,
                    out_b_offset,
                    out_b_type,
                    group_dim,
                    rank,
                    n_groups,
                    out_dim,
                    heads,
                    2u) == 0);

    TEST_ASSERT(setenv(enable_env, "1", 1) == 0);
    for (uint32_t n_rows = 2u; n_rows <= max_rows; n_rows++) {
        for (uint64_t i = 0; i < (uint64_t)alloc_rows * low_dim; i++) {
            const uint32_t bits = 0x7fc10000u + (uint32_t)(i & 0xffffu);
            memcpy(&candidate_low_host[i], &bits, sizeof(bits));
        }
        for (uint64_t i = 0; i < (uint64_t)alloc_rows * out_dim; i++) {
            const uint32_t bits = 0x7fc20000u + (uint32_t)(i & 0xffffu);
            memcpy(&candidate_out_host[i], &bits, sizeof(bits));
        }
        TEST_ASSERT(ds4_gpu_tensor_write(
                        candidate_low, 0, candidate_low_host, low_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        candidate_out, 0, candidate_out_host, out_bytes) != 0);
        TEST_ASSERT(ds4_gpu_attention_output_q4_K_batch_tensor(
                        candidate_out,
                        candidate_low,
                        group_tmp,
                        low_tmp,
                        model_raw,
                        model_bytes,
                        0,
                        out_b_offset,
                        out_b_type,
                        group_dim,
                        rank,
                        n_groups,
                        out_dim,
                        heads,
                        n_rows) == 1);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        candidate_low, 0, candidate_low_host, low_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        candidate_out, 0, candidate_out_host, out_bytes) != 0);

        const size_t low_count = (size_t)n_rows * low_dim;
        const size_t out_count = (size_t)n_rows * out_dim;
        const test_float_compare_stats low_stats = test_compare_float_bits(
            reference_low_host, candidate_low_host, low_count);
        const test_float_compare_stats out_stats = test_compare_float_bits(
            reference_out_host, candidate_out_host, out_count);
        size_t low_oob_mismatch = 0;
        size_t out_oob_mismatch = 0;
        for (uint64_t i = (uint64_t)n_rows * low_dim;
             i < (uint64_t)alloc_rows * low_dim;
             i++) {
            uint32_t bits = 0;
            memcpy(&bits, &candidate_low_host[i], sizeof(bits));
            if (bits != 0x7fc10000u + (uint32_t)(i & 0xffffu)) {
                low_oob_mismatch++;
            }
        }
        for (uint64_t i = (uint64_t)n_rows * out_dim;
             i < (uint64_t)alloc_rows * out_dim;
             i++) {
            uint32_t bits = 0;
            memcpy(&bits, &candidate_out_host[i], sizeof(bits));
            if (bits != 0x7fc20000u + (uint32_t)(i & 0xffffu)) {
                out_oob_mismatch++;
            }
        }
        fprintf(stderr,
                "ds4-test: Metal Q4 attention-output out_b=%s exact-%u "
                "low=%zu/%zu out=%zu/%zu low_oob=%zu out_oob=%zu\n",
                out_b_type == 8u ? "Q8_0" : "Q4_K",
                n_rows,
                low_stats.mismatch_count,
                low_count,
                out_stats.mismatch_count,
                out_count,
                low_oob_mismatch,
                out_oob_mismatch);
        TEST_ASSERT(low_stats.mismatch_count == 0 && low_stats.max_ulp == 0);
        TEST_ASSERT(out_stats.mismatch_count == 0 && out_stats.max_ulp == 0);
        TEST_ASSERT(low_oob_mismatch == 0);
        TEST_ASSERT(out_oob_mismatch == 0);
    }

    /* REQUIRE is a fail-closed model-oracle gate and implies enable for the
     * exact N=2..5 scope. */
    TEST_ASSERT(unsetenv(enable_env) == 0);
    TEST_ASSERT(setenv(require_env, "1", 1) == 0);
    TEST_ASSERT(ds4_gpu_attention_output_q4_K_batch_tensor(
                    candidate_out,
                    candidate_low,
                    group_tmp,
                    low_tmp,
                    model_raw,
                    model_bytes,
                    0,
                    out_b_offset,
                    out_b_type,
                    group_dim,
                    rank,
                    n_groups,
                    out_dim,
                    heads,
                    2u) == 1);

    if (out_b_type == 12u) {
        /* A required candidate must hard-fail rather than bypass an explicit
         * request for the alternate Q4 schedule. */
        TEST_ASSERT(unsetenv(disable_env) == 0);
        TEST_ASSERT(setenv(disable_classic_env, "1", 1) == 0);
        TEST_ASSERT(ds4_gpu_attention_output_q4_K_batch_tensor(
                        candidate_out,
                        candidate_low,
                        group_tmp,
                        low_tmp,
                        model_raw,
                        model_bytes,
                        0,
                        out_b_offset,
                        out_b_type,
                        group_dim,
                        rank,
                        n_groups,
                        out_dim,
                        heads,
                        2u) == -1);
        TEST_ASSERT(unsetenv(disable_classic_env) == 0);
    }

    /* The unconditional kill switch wins over REQUIRE and makes the
     * model-oracle run fail instead of silently selecting the fallback. */
    TEST_ASSERT(setenv(disable_env, "1", 1) == 0);
    TEST_ASSERT(ds4_gpu_attention_output_q4_K_batch_tensor(
                    candidate_out,
                    candidate_low,
                    group_tmp,
                    low_tmp,
                    model_raw,
                    model_bytes,
                    0,
                    out_b_offset,
                    out_b_type,
                    group_dim,
                    rank,
                    n_groups,
                    out_dim,
                    heads,
                    2u) == -1);

    /* Without REQUIRE the same kill switch retains the ordinary fallback. */
    TEST_ASSERT(unsetenv(require_env) == 0);
    TEST_ASSERT(setenv(enable_env, "1", 1) == 0);
    TEST_ASSERT(ds4_gpu_attention_output_q4_K_batch_tensor(
                    candidate_out,
                    candidate_low,
                    group_tmp,
                    low_tmp,
                    model_raw,
                    model_bytes,
                    0,
                    out_b_offset,
                    out_b_type,
                    group_dim,
                    rank,
                    n_groups,
                    out_dim,
                    heads,
                    2u) == 0);

cleanup:
    ds4_gpu_tensor_free(low_tmp);
    ds4_gpu_tensor_free(group_tmp);
    ds4_gpu_tensor_free(candidate_out);
    ds4_gpu_tensor_free(candidate_low);
    ds4_gpu_tensor_free(reference_out);
    ds4_gpu_tensor_free(reference_low);
    ds4_gpu_tensor_free(heads);
    free(candidate_out_host);
    free(candidate_low_host);
    free(reference_out_host);
    free(reference_low_host);
    free(heads_host);
    free(model_raw);
    test_restore_env(disable_classic_env, saved_disable_classic);
    test_restore_env(require_env, saved_require);
    test_restore_env(disable_env, saved_disable);
    test_restore_env(enable_env, saved_enable);
}

static void test_metal_q4_attention_output_tiny_batch_exact(void) {
    test_metal_q4_attention_output_tiny_batch_exact_case(8u);
    test_metal_q4_attention_output_tiny_batch_exact_case(12u);
}

static void test_metal_dspark_device_proposer_q8(void) {
    /* Keep this fixture small while exercising the complete public contract:
     * six on-device draft steps, a confidence stop after a verified prefix,
     * and stable lowest-token tie breaking in the Markov argmax. */
    _Static_assert(sizeof(ds4_gpu_dspark_device_proposal) == 64u,
                   "DSpark device proposal ABI must stay 64 bytes");
    _Static_assert(DS4_GPU_DSPARK_MAX_DRAFTS == 6u,
                   "DSpark device proposer test covers the six-token limit");
    _Static_assert(offsetof(ds4_gpu_dspark_device_proposal, tokens) == 0u &&
                   offsetof(ds4_gpu_dspark_device_proposal,
                            confidence_logits) == 24u &&
                   offsetof(ds4_gpu_dspark_device_proposal,
                            proposal_len) == 48u &&
                   offsetof(ds4_gpu_dspark_device_proposal,
                            confidence_len) == 52u &&
                   offsetof(ds4_gpu_dspark_device_proposal, status) == 56u &&
                   offsetof(ds4_gpu_dspark_device_proposal, reserved) == 60u,
                   "DSpark device proposal field offsets changed");

    const uint32_t vocab = 64u;
    const uint32_t rank = 32u;
    const uint32_t hidden_dim = 32u;
    const uint32_t max_drafts = DS4_GPU_DSPARK_MAX_DRAFTS;
    const uint32_t first_prev_token = 17u;
    const uint32_t tied_token_lo = 3u;
    const uint32_t tied_token_hi = 11u;
    const float reused_confidence = 8.0f;
    const uint64_t page = (uint64_t)getpagesize();
    const uint64_t markov_row_bytes = (uint64_t)(rank / 32u) * 34u;
    const uint64_t markov_bytes = (uint64_t)vocab * markov_row_bytes;
    const uint64_t confidence_bytes =
        (uint64_t)((hidden_dim + rank) / 32u) * 34u;
    const uint64_t w1_offset = 0u;
    const uint64_t w2_offset = test_round_up_u64(markov_bytes, page);
    const uint64_t confidence_offset =
        test_round_up_u64(w2_offset + markov_bytes, page);
    const uint64_t model_bytes = test_round_up_u64(
        confidence_offset + confidence_bytes, page);
    const uint64_t logits_bytes =
        (uint64_t)max_drafts * vocab * sizeof(float);
    const uint64_t hidden_bytes =
        (uint64_t)max_drafts * hidden_dim * sizeof(float);

    void *model_raw = NULL;
    TEST_ASSERT(posix_memalign(&model_raw,
                               (size_t)page,
                               (size_t)model_bytes) == 0);
    if (!model_raw) return;
    /* Zero is a valid Q8_0 block (zero half scale and zero quants).  It makes
     * every Markov correction and computed confidence exactly zero while the
     * kernels still traverse all Q8 blocks. */
    memset(model_raw, 0, (size_t)model_bytes);

    ds4_gpu_tensor *logits = ds4_gpu_tensor_alloc(logits_bytes);
    ds4_gpu_tensor *hidden = ds4_gpu_tensor_alloc(hidden_bytes);
    ds4_gpu_tensor *result =
        ds4_gpu_tensor_alloc(DS4_GPU_DSPARK_DEVICE_PROPOSAL_BYTES);
    TEST_ASSERT(logits != NULL);
    TEST_ASSERT(hidden != NULL);
    TEST_ASSERT(result != NULL);
    if (!logits || !hidden || !result) {
        ds4_gpu_tensor_free(logits);
        ds4_gpu_tensor_free(hidden);
        ds4_gpu_tensor_free(result);
        free(model_raw);
        return;
    }

    float *logits_host = malloc((size_t)logits_bytes);
    float *hidden_host = malloc((size_t)hidden_bytes);
    TEST_ASSERT(logits_host != NULL);
    TEST_ASSERT(hidden_host != NULL);
    if (!logits_host || !hidden_host) {
        free(logits_host);
        free(hidden_host);
        ds4_gpu_tensor_free(logits);
        ds4_gpu_tensor_free(hidden);
        ds4_gpu_tensor_free(result);
        free(model_raw);
        return;
    }

    for (uint32_t draft = 0u; draft < max_drafts; draft++) {
        for (uint32_t token = 0u; token < vocab; token++) {
            logits_host[(uint64_t)draft * vocab + token] =
                -4.0f - (float)(token % 7u) * 0.03125f;
        }
        logits_host[(uint64_t)draft * vocab + tied_token_lo] = 5.0f;
        logits_host[(uint64_t)draft * vocab + tied_token_hi] = 5.0f;
        for (uint32_t i = 0u; i < hidden_dim; i++) {
            hidden_host[(uint64_t)draft * hidden_dim + i] =
                (float)((int)((draft * 19u + i * 13u) % 37u) - 18) / 23.0f;
        }
    }

    TEST_ASSERT(ds4_gpu_tensor_write(
                    logits, 0, logits_host, logits_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(
                    hidden, 0, hidden_host, hidden_bytes) != 0);
    TEST_ASSERT(ds4_gpu_set_model_map(model_raw, model_bytes) != 0);
    ds4_gpu_set_quality(false);

    char *saved_enable =
        test_save_env("DS4_METAL_DSPARK_DEVICE_PROPOSER");
    char *saved_kill =
        test_save_env("DS4_METAL_DSPARK_NO_DEVICE_PROPOSER");
    setenv("DS4_METAL_DSPARK_DEVICE_PROPOSER", "1", 1);
    unsetenv("DS4_METAL_DSPARK_NO_DEVICE_PROPOSER");

    ds4_gpu_dspark_device_proposal public_result;
    bool full_six_ok = false;
    for (uint32_t n_drafts = 1u; n_drafts <= max_drafts; n_drafts++) {
        memset(&public_result, 0xa5, sizeof(public_result));
        TEST_ASSERT(ds4_gpu_tensor_write(result,
                                          0,
                                          &public_result,
                                          sizeof(public_result)) != 0);
        const int ok = ds4_gpu_dspark_markov_confidence_q8_tensor(
                result,
                logits,
                hidden,
                model_raw,
                model_bytes,
                w1_offset,
                w2_offset,
                confidence_offset,
                first_prev_token,
                vocab,
                rank,
                hidden_dim,
                n_drafts,
                0.25f,
                1,
                reused_confidence);
        TEST_ASSERT(ok != 0);
        if (!ok) break;
        TEST_ASSERT(ds4_gpu_tensor_read(result,
                                         0,
                                         &public_result,
                                         sizeof(public_result)) != 0);
        TEST_ASSERT(public_result.status == 1u);
        TEST_ASSERT(public_result.reserved == 0u);
        TEST_ASSERT(public_result.proposal_len == n_drafts);
        TEST_ASSERT(public_result.confidence_len == n_drafts);
        for (uint32_t i = 0u; i < max_drafts; i++) {
            if (i < n_drafts) {
                TEST_ASSERT(public_result.tokens[i] ==
                            (int32_t)tied_token_lo);
                TEST_ASSERT(public_result.confidence_logits[i] ==
                            (i == 0u ? reused_confidence : 0.0f));
            } else {
                TEST_ASSERT(public_result.tokens[i] == -1);
                TEST_ASSERT(public_result.confidence_logits[i] == 0.0f);
            }
        }
        if (n_drafts == max_drafts) full_six_ok = true;
    }
    TEST_ASSERT(full_six_ok);

    /* Draft zero passes via the supplied confidence.  Draft one computes a
     * zero logit (p=0.5), fails the 0.75 threshold, and therefore leaves a
     * one-token proposal with two evaluated confidence rows. */
    memset(&public_result, 0xa5, sizeof(public_result));
    TEST_ASSERT(ds4_gpu_tensor_write(result,
                                      0,
                                      &public_result,
                                      sizeof(public_result)) != 0);
    const int stop_ok = ds4_gpu_dspark_markov_confidence_q8_tensor(
            result,
            logits,
            hidden,
            model_raw,
            model_bytes,
            w1_offset,
            w2_offset,
            confidence_offset,
            first_prev_token,
            vocab,
            rank,
            hidden_dim,
            max_drafts,
            0.75f,
            1,
            reused_confidence);
    TEST_ASSERT(stop_ok != 0);
    if (stop_ok) {
        TEST_ASSERT(ds4_gpu_tensor_read(result,
                                         0,
                                         &public_result,
                                         sizeof(public_result)) != 0);
        TEST_ASSERT(public_result.status == 1u);
        TEST_ASSERT(public_result.reserved == 0u);
        TEST_ASSERT(public_result.proposal_len == 1u);
        TEST_ASSERT(public_result.confidence_len == 2u);
        TEST_ASSERT(public_result.tokens[0] == (int32_t)tied_token_lo);
        TEST_ASSERT(public_result.confidence_logits[0] == reused_confidence);
        TEST_ASSERT(public_result.confidence_logits[1] == 0.0f);
        for (uint32_t i = 1u; i < max_drafts; i++) {
            TEST_ASSERT(public_result.tokens[i] == -1);
            if (i > 1u) {
                TEST_ASSERT(public_result.confidence_logits[i] == 0.0f);
            }
        }
    }

    setenv("DS4_METAL_DSPARK_NO_DEVICE_PROPOSER", "1", 1);
    TEST_ASSERT(ds4_gpu_dspark_markov_confidence_q8_tensor(
                    result,
                    logits,
                    hidden,
                    model_raw,
                    model_bytes,
                    w1_offset,
                    w2_offset,
                    confidence_offset,
                    first_prev_token,
                    vocab,
                    rank,
                    hidden_dim,
                    1u,
                    0.25f,
                    1,
                    reused_confidence) == 0);
    test_restore_env("DS4_METAL_DSPARK_NO_DEVICE_PROPOSER", saved_kill);
    test_restore_env("DS4_METAL_DSPARK_DEVICE_PROPOSER", saved_enable);

    fprintf(stderr,
            "ds4-test: Metal DSpark device proposer full6=%d stop_prefix=%u "
            "confidence_rows=%u tie=%u<%u\n",
            full_six_ok ? 1 : 0,
            stop_ok ? public_result.proposal_len : 0u,
            stop_ok ? public_result.confidence_len : 0u,
            tied_token_lo,
            tied_token_hi);

    free(logits_host);
    free(hidden_host);
    ds4_gpu_tensor_free(logits);
    ds4_gpu_tensor_free(hidden);
    ds4_gpu_tensor_free(result);
    free(model_raw);
}

static void test_metal_f16_compressor_pair_state_store_exact_case(
        uint32_t width,
        uint32_t ratio,
        uint32_t pos,
        uint32_t ape_type,
        uint32_t seed,
        bool test_decode_pack) {
    const uint32_t in_dim = 4096u;
    const uint32_t coff = ratio == 4u ? 2u : 1u;
    const uint32_t head_dim = width / coff;
    const uint32_t state_rows = coff * ratio;
    const bool emit = ((pos + 1u) % ratio) == 0u;
    TEST_ASSERT(!test_decode_pack ||
                (ratio == 4u && emit &&
                 (head_dim == 128u || head_dim == 512u)));
    const uint64_t page = (uint64_t)getpagesize();
    const uint64_t weight_bytes =
        (uint64_t)width * in_dim * sizeof(uint16_t);
    const uint64_t score_weight_offset =
        test_round_up_u64(weight_bytes, page);
    const uint64_t ape_offset = test_round_up_u64(
        score_weight_offset + weight_bytes, page);
    const uint64_t ape_elem_bytes = ape_type == 1u ? 2u : 4u;
    const uint64_t ape_bytes = (uint64_t)ratio * width * ape_elem_bytes;
    const uint64_t norm_offset =
        test_round_up_u64(ape_offset + ape_bytes, page);
    const uint64_t model_bytes = test_round_up_u64(
        norm_offset + (uint64_t)head_dim * sizeof(float), page);
    const uint64_t x_bytes = (uint64_t)in_dim * sizeof(float);
    const uint64_t out_bytes = (uint64_t)width * sizeof(float);
    const uint64_t state_count = (uint64_t)state_rows * width;
    const uint64_t state_bytes = state_count * sizeof(float);
    const uint64_t comp_bytes = (uint64_t)head_dim * sizeof(float);

    void *model_raw = NULL;
    TEST_ASSERT(posix_memalign(&model_raw, (size_t)page,
                               (size_t)model_bytes) == 0);
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(x_bytes);
    ds4_gpu_tensor *ref_kv = ds4_gpu_tensor_alloc(out_bytes);
    ds4_gpu_tensor *ref_score = ds4_gpu_tensor_alloc(out_bytes);
    ds4_gpu_tensor *fused_kv = ds4_gpu_tensor_alloc(out_bytes);
    ds4_gpu_tensor *fused_score = ds4_gpu_tensor_alloc(out_bytes);
    ds4_gpu_tensor *ref_state_kv = ds4_gpu_tensor_alloc(state_bytes);
    ds4_gpu_tensor *ref_state_score = ds4_gpu_tensor_alloc(state_bytes);
    ds4_gpu_tensor *fused_state_kv = ds4_gpu_tensor_alloc(state_bytes);
    ds4_gpu_tensor *fused_state_score = ds4_gpu_tensor_alloc(state_bytes);
    ds4_gpu_tensor *ref_comp = ds4_gpu_tensor_alloc(comp_bytes);
    ds4_gpu_tensor *fused_comp = ds4_gpu_tensor_alloc(comp_bytes);

    float *x_host = malloc((size_t)x_bytes);
    float *ref_kv_host = malloc((size_t)out_bytes);
    float *ref_score_host = malloc((size_t)out_bytes);
    float *fused_kv_host = malloc((size_t)out_bytes);
    float *fused_score_host = malloc((size_t)out_bytes);
    float *ref_state_kv_host = malloc((size_t)state_bytes);
    float *ref_state_score_host = malloc((size_t)state_bytes);
    float *fused_state_kv_host = malloc((size_t)state_bytes);
    float *fused_state_score_host = malloc((size_t)state_bytes);
    float *ref_comp_host = malloc((size_t)comp_bytes);
    float *fused_comp_host = malloc((size_t)comp_bytes);

    TEST_ASSERT(model_raw != NULL);
    TEST_ASSERT(x != NULL);
    TEST_ASSERT(ref_kv != NULL);
    TEST_ASSERT(ref_score != NULL);
    TEST_ASSERT(fused_kv != NULL);
    TEST_ASSERT(fused_score != NULL);
    TEST_ASSERT(ref_state_kv != NULL);
    TEST_ASSERT(ref_state_score != NULL);
    TEST_ASSERT(fused_state_kv != NULL);
    TEST_ASSERT(fused_state_score != NULL);
    TEST_ASSERT(ref_comp != NULL);
    TEST_ASSERT(fused_comp != NULL);
    TEST_ASSERT(x_host != NULL);
    TEST_ASSERT(ref_kv_host != NULL);
    TEST_ASSERT(ref_score_host != NULL);
    TEST_ASSERT(fused_kv_host != NULL);
    TEST_ASSERT(fused_score_host != NULL);
    TEST_ASSERT(ref_state_kv_host != NULL);
    TEST_ASSERT(ref_state_score_host != NULL);
    TEST_ASSERT(fused_state_kv_host != NULL);
    TEST_ASSERT(fused_state_score_host != NULL);
    TEST_ASSERT(ref_comp_host != NULL);
    TEST_ASSERT(fused_comp_host != NULL);

    const bool allocated = model_raw && x && ref_kv && ref_score && fused_kv &&
        fused_score && ref_state_kv && ref_state_score && fused_state_kv &&
        fused_state_score && ref_comp && fused_comp && x_host && ref_kv_host &&
        ref_score_host && fused_kv_host && fused_score_host &&
        ref_state_kv_host && ref_state_score_host && fused_state_kv_host &&
        fused_state_score_host && ref_comp_host && fused_comp_host;

    const char *pair_disable_env =
        "DS4_METAL_DISABLE_COMPRESSOR_PAIR_PROJ";
    const char *store_disable_env =
        "DS4_METAL_DISABLE_COMPRESSOR_STORE_ONE";
    const char *decode_pack_disable_env =
        "DS4_METAL_DISABLE_PRE_M5_COMPRESSOR_RATIO4_DECODE_PACK_FUSION";
    const char *exact_reduction_disable_env =
        "DS4_METAL_DISABLE_PRE_M5_COMPRESSOR_EXACT_REDUCTION_FUSION";
    const char *exact_reduction_poison_env =
        "DS4_METAL_TEST_POISON_COMPRESSOR_EXACT_REDUCTION_SCRATCH";
    char *saved_pair_disable = test_save_env(pair_disable_env);
    char *saved_store_disable = test_save_env(store_disable_env);
    char *saved_decode_pack_disable =
        test_save_env(decode_pack_disable_env);
    char *saved_exact_reduction_disable =
        test_save_env(exact_reduction_disable_env);
    char *saved_exact_reduction_poison =
        test_save_env(exact_reduction_poison_env);

    test_float_compare_stats kv_stats = {0};
    test_float_compare_stats score_stats = {0};
    test_float_compare_stats state_kv_stats = {0};
    test_float_compare_stats state_score_stats = {0};
    test_float_compare_stats comp_stats = {0};

    if (allocated) {
        memset(model_raw, 0, (size_t)model_bytes);
        uint16_t *kv_weights = model_raw;
        uint16_t *score_weights =
            (uint16_t *)((uint8_t *)model_raw + score_weight_offset);
        for (uint32_t o = 0; o < width; o++) {
            for (uint32_t i = 0; i < in_dim; i++) {
                const int kv_value =
                    (int)((o * 17u + i * 23u + (o ^ i) * 3u +
                           seed * 29u) % 67u) - 33;
                const int score_value =
                    (int)((o * 31u + i * 11u + (o ^ (i >> 2u)) * 5u +
                           seed * 19u) % 71u) - 35;
                const uint64_t wi = (uint64_t)o * in_dim + i;
                kv_weights[wi] = test_float_to_f16(
                    (float)kv_value / 96.0f);
                score_weights[wi] = test_float_to_f16(
                    (float)score_value / 104.0f);
            }
        }

        if (ape_type == 1u) {
            uint16_t *ape =
                (uint16_t *)((uint8_t *)model_raw + ape_offset);
            for (uint64_t i = 0; i < (uint64_t)ratio * width; i++) {
                const int value =
                    (int)((i * 13u + (i ^ (i >> 3u)) * 7u +
                           seed * 17u) % 61u) - 30;
                ape[i] = test_float_to_f16((float)value / 80.0f);
            }
        } else {
            float *ape = (float *)((uint8_t *)model_raw + ape_offset);
            for (uint64_t i = 0; i < (uint64_t)ratio * width; i++) {
                const int value =
                    (int)((i * 13u + (i ^ (i >> 3u)) * 7u +
                           seed * 17u) % 61u) - 30;
                ape[i] = (float)value / 80.0f;
            }
        }
        float *norm = (float *)((uint8_t *)model_raw + norm_offset);
        for (uint32_t i = 0; i < head_dim; i++) {
            norm[i] = 0.75f +
                (float)((i * 7u + seed * 3u) % 23u) / 64.0f;
        }

        for (uint32_t i = 0; i < in_dim; i++) {
            const int value =
                (int)((i * 29u + (i ^ (i >> 4u)) * 9u +
                       seed * 11u) % 127u) - 63;
            x_host[i] = (float)value / 88.0f;
        }
        for (uint32_t i = 0; i < width; i++) {
            const uint32_t poison = 0x7fc00001u + (i & 0x3ffu);
            memcpy(ref_kv_host + i, &poison, sizeof(poison));
            memcpy(ref_score_host + i, &poison, sizeof(poison));
            memcpy(fused_kv_host + i, &poison, sizeof(poison));
            memcpy(fused_score_host + i, &poison, sizeof(poison));
        }
        for (uint64_t i = 0; i < state_count; i++) {
            const int kv_value =
                (int)((i * 5u + seed * 13u) % 97u) - 48;
            const int score_value =
                (int)((i * 7u + seed * 5u) % 101u) - 50;
            ref_state_kv_host[i] = (float)kv_value / 64.0f;
            fused_state_kv_host[i] = ref_state_kv_host[i];
            ref_state_score_host[i] = (float)score_value / 72.0f;
            fused_state_score_host[i] = ref_state_score_host[i];
        }
        if (test_decode_pack) {
            static const uint32_t edge_bits[8] = {
                0x00000000u, 0x80000000u, 0x00000001u, 0x80000001u,
                0x3f800000u, 0xbf800000u, 0x42a00000u, 0xc2a00000u,
            };
            for (uint32_t col = 0; col < 4u; col++) {
                for (uint32_t row = 0; row < 8u; row++) {
                    const uint64_t state_col =
                        (row >= 4u ? head_dim : 0u) + col;
                    const uint64_t state_index =
                        (uint64_t)row * width + state_col;
                    const uint32_t score_bits =
                        edge_bits[(row + col) & 7u];
                    const uint32_t kv_bits =
                        edge_bits[(7u - row + col) & 7u];
                    memcpy(ref_state_score_host + state_index,
                           &score_bits, sizeof(score_bits));
                    memcpy(fused_state_score_host + state_index,
                           &score_bits, sizeof(score_bits));
                    memcpy(ref_state_kv_host + state_index,
                           &kv_bits, sizeof(kv_bits));
                    memcpy(fused_state_kv_host + state_index,
                           &kv_bits, sizeof(kv_bits));
                }
            }
        }
        for (uint32_t i = 0; i < head_dim; i++) {
            const uint32_t poison = 0x7fc01001u + (i & 0x3ffu);
            memcpy(ref_comp_host + i, &poison, sizeof(poison));
            memcpy(fused_comp_host + i, &poison, sizeof(poison));
        }

        TEST_ASSERT(ds4_gpu_tensor_write(x, 0, x_host, x_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        ref_kv, 0, ref_kv_host, out_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        ref_score, 0, ref_score_host, out_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        fused_kv, 0, fused_kv_host, out_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        fused_score, 0, fused_score_host, out_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        ref_state_kv, 0, ref_state_kv_host, state_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        ref_state_score, 0, ref_state_score_host,
                        state_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        fused_state_kv, 0, fused_state_kv_host,
                        state_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        fused_state_score, 0, fused_state_score_host,
                        state_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        ref_comp, 0, ref_comp_host, comp_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        fused_comp, 0, fused_comp_host, comp_bytes) != 0);
        TEST_ASSERT(ds4_gpu_set_model_map(model_raw, model_bytes) != 0);
        ds4_gpu_set_quality(false);

        TEST_ASSERT(unsetenv(pair_disable_env) == 0);
        TEST_ASSERT(unsetenv(store_disable_env) == 0);
        TEST_ASSERT(setenv(exact_reduction_disable_env, "1", 1) == 0);
        TEST_ASSERT(unsetenv(exact_reduction_poison_env) == 0);
        if (test_decode_pack) {
            TEST_ASSERT(setenv(decode_pack_disable_env, "1", 1) == 0);
        }

        TEST_ASSERT(ds4_gpu_matmul_f16_pair_tensor(
                        ref_kv, ref_score, model_raw, model_bytes,
                        0, score_weight_offset, in_dim, width, x, 1) != 0);
        TEST_ASSERT(ds4_gpu_compressor_update_tensor(
                        ref_kv, ref_score, ref_state_kv, ref_state_score,
                        ref_comp, model_raw, model_bytes, ape_offset, ape_type,
                        norm_offset, 0, head_dim, ratio, pos, 0, 0, 0,
                        10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f,
                        1.0e-6f, false, test_decode_pack, false) != 0);

        TEST_ASSERT(ds4_gpu_matmul_f16_pair_compressor_store_tensor(
                        fused_kv, fused_score,
                        fused_state_kv, fused_state_score,
                        model_raw, model_bytes, 0, score_weight_offset,
                        ape_offset, ape_type, in_dim, width, x,
                        ratio, pos) == 1);
        if (test_decode_pack) {
            TEST_ASSERT(unsetenv(decode_pack_disable_env) == 0);
            TEST_ASSERT(unsetenv(exact_reduction_disable_env) == 0);
            TEST_ASSERT(setenv(exact_reduction_poison_env, "1", 1) == 0);
        }
        TEST_ASSERT(ds4_gpu_compressor_update_tensor(
                        fused_kv, fused_score,
                        fused_state_kv, fused_state_score,
                        fused_comp, model_raw, model_bytes,
                        ape_offset, ape_type, norm_offset, 0,
                        head_dim, ratio, pos, 0, 0, 0,
                        10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f,
                        1.0e-6f, true, test_decode_pack, false) != 0);

        TEST_ASSERT(ds4_gpu_tensor_read(
                        ref_kv, 0, ref_kv_host, out_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        ref_score, 0, ref_score_host, out_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        fused_kv, 0, fused_kv_host, out_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        fused_score, 0, fused_score_host, out_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        ref_state_kv, 0, ref_state_kv_host, state_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        ref_state_score, 0, ref_state_score_host,
                        state_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        fused_state_kv, 0, fused_state_kv_host,
                        state_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        fused_state_score, 0, fused_state_score_host,
                        state_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        ref_comp, 0, ref_comp_host, comp_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        fused_comp, 0, fused_comp_host, comp_bytes) != 0);

        kv_stats = test_compare_float_bits(
            ref_kv_host, fused_kv_host, width);
        score_stats = test_compare_float_bits(
            ref_score_host, fused_score_host, width);
        state_kv_stats = test_compare_float_bits(
            ref_state_kv_host, fused_state_kv_host, (size_t)state_count);
        state_score_stats = test_compare_float_bits(
            ref_state_score_host, fused_state_score_host,
            (size_t)state_count);
        comp_stats = test_compare_float_bits(
            ref_comp_host, fused_comp_host, head_dim);
    }

    test_restore_env(pair_disable_env, saved_pair_disable);
    test_restore_env(store_disable_env, saved_store_disable);
    test_restore_env(decode_pack_disable_env, saved_decode_pack_disable);
    test_restore_env(exact_reduction_disable_env,
                     saved_exact_reduction_disable);
    test_restore_env(
        exact_reduction_poison_env, saved_exact_reduction_poison);

    fprintf(stderr,
            "ds4-test: compressor pair state-store exact width=%u ratio=%u "
            "pos=%u emit=%u ape=%s decode_pack=%u exact_reduce=%u "
            "proj=%zu/%zu state=%zu/%zu "
            "comp=%zu max_ulp=%u/%u/%u/%u/%u\n",
            width, ratio, pos, emit ? 1u : 0u,
            ape_type == 1u ? "f16" : "f32",
            test_decode_pack ? 1u : 0u,
            test_decode_pack ? 1u : 0u,
            kv_stats.mismatch_count, score_stats.mismatch_count,
            state_kv_stats.mismatch_count,
            state_score_stats.mismatch_count,
            comp_stats.mismatch_count,
            kv_stats.max_ulp, score_stats.max_ulp,
            state_kv_stats.max_ulp, state_score_stats.max_ulp,
            comp_stats.max_ulp);
    TEST_ASSERT(kv_stats.mismatch_count == 0);
    TEST_ASSERT(score_stats.mismatch_count == 0);
    TEST_ASSERT(state_kv_stats.mismatch_count == 0);
    TEST_ASSERT(state_score_stats.mismatch_count == 0);
    TEST_ASSERT(comp_stats.mismatch_count == 0);

    free(fused_comp_host);
    free(ref_comp_host);
    free(fused_state_score_host);
    free(fused_state_kv_host);
    free(ref_state_score_host);
    free(ref_state_kv_host);
    free(fused_score_host);
    free(fused_kv_host);
    free(ref_score_host);
    free(ref_kv_host);
    free(x_host);
    ds4_gpu_tensor_free(fused_comp);
    ds4_gpu_tensor_free(ref_comp);
    ds4_gpu_tensor_free(fused_state_score);
    ds4_gpu_tensor_free(fused_state_kv);
    ds4_gpu_tensor_free(ref_state_score);
    ds4_gpu_tensor_free(ref_state_kv);
    ds4_gpu_tensor_free(fused_score);
    ds4_gpu_tensor_free(fused_kv);
    ds4_gpu_tensor_free(ref_score);
    ds4_gpu_tensor_free(ref_kv);
    ds4_gpu_tensor_free(x);
    free(model_raw);
}

static void test_metal_f16_compressor_pair_state_store_exact(void) {
    test_metal_f16_compressor_pair_state_store_exact_case(
        256, 4, 8, 0, 17, false);
    test_metal_f16_compressor_pair_state_store_exact_case(
        256, 4, 11, 1, 23, true);
    test_metal_f16_compressor_pair_state_store_exact_case(
        1024, 4, 11, 1, 29, true);
    test_metal_f16_compressor_pair_state_store_exact_case(
        512, 128, 255, 1, 43, false);
}

static void test_metal_f16_compressor_quad_state_store_exact_case(
        uint32_t width0,
        uint32_t pos,
        uint32_t ape0_type,
        uint32_t ape1_type,
        uint32_t seed) {
    const uint32_t in_dim = 4096u;
    const uint32_t ratio = 4u;
    const uint32_t width1 = 256u;
    const uint32_t state_rows = 2u * ratio;
    const uint32_t widths[4] = {width0, width0, width1, width1};
    const uint32_t ape_types[2] = {ape0_type, ape1_type};
    const uint64_t page = (uint64_t)getpagesize();
    const uint64_t weight_bytes[4] = {
        (uint64_t)width0 * in_dim * sizeof(uint16_t),
        (uint64_t)width0 * in_dim * sizeof(uint16_t),
        (uint64_t)width1 * in_dim * sizeof(uint16_t),
        (uint64_t)width1 * in_dim * sizeof(uint16_t),
    };
    uint64_t weight_offsets[4] = {0};
    for (uint32_t i = 1; i < 4u; i++) {
        weight_offsets[i] = test_round_up_u64(
            weight_offsets[i - 1u] + weight_bytes[i - 1u], page);
    }
    const uint64_t ape0_offset = test_round_up_u64(
        weight_offsets[3] + weight_bytes[3], page);
    const uint64_t ape0_bytes = (uint64_t)ratio * width0 *
        (ape0_type == 1u ? sizeof(uint16_t) : sizeof(float));
    const uint64_t ape1_offset = test_round_up_u64(
        ape0_offset + ape0_bytes, page);
    const uint64_t ape1_bytes = (uint64_t)ratio * width1 *
        (ape1_type == 1u ? sizeof(uint16_t) : sizeof(float));
    const uint64_t ape_offsets[2] = {ape0_offset, ape1_offset};
    const uint32_t ape_widths[2] = {width0, width1};
    const uint64_t model_bytes = test_round_up_u64(
        ape1_offset + ape1_bytes, page);
    const uint64_t x_bytes = (uint64_t)in_dim * sizeof(float);
    const uint64_t max_state_count = (uint64_t)state_rows * width0;

    void *model_raw = NULL;
    TEST_ASSERT(posix_memalign(
                    &model_raw, (size_t)page, (size_t)model_bytes) == 0);
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(x_bytes);
    ds4_gpu_tensor *ref_out[4] = {0};
    ds4_gpu_tensor *quad_out[4] = {0};
    ds4_gpu_tensor *ref_state[4] = {0};
    ds4_gpu_tensor *quad_state[4] = {0};
    bool allocated = model_raw != NULL && x != NULL;
    for (uint32_t i = 0; i < 4u; i++) {
        const uint64_t out_bytes =
            (uint64_t)widths[i] * sizeof(float);
        const uint64_t state_bytes =
            (uint64_t)state_rows * widths[i] * sizeof(float);
        ref_out[i] = ds4_gpu_tensor_alloc(out_bytes);
        quad_out[i] = ds4_gpu_tensor_alloc(out_bytes);
        ref_state[i] = ds4_gpu_tensor_alloc(state_bytes);
        quad_state[i] = ds4_gpu_tensor_alloc(state_bytes);
        TEST_ASSERT(ref_out[i] != NULL);
        TEST_ASSERT(quad_out[i] != NULL);
        TEST_ASSERT(ref_state[i] != NULL);
        TEST_ASSERT(quad_state[i] != NULL);
        allocated = allocated && ref_out[i] && quad_out[i] &&
                    ref_state[i] && quad_state[i];
    }

    float *x_host = malloc((size_t)x_bytes);
    float *ref_host = malloc((size_t)max_state_count * sizeof(float));
    float *quad_host = malloc((size_t)max_state_count * sizeof(float));
    TEST_ASSERT(x_host != NULL);
    TEST_ASSERT(ref_host != NULL);
    TEST_ASSERT(quad_host != NULL);
    allocated = allocated && x_host && ref_host && quad_host;

    const char *force_pair_env =
        "DS4_METAL_ENABLE_COMPRESSOR_PAIR_STATE_STORE";
    const char *disable_pair_state_env =
        "DS4_METAL_DISABLE_M3_COMPRESSOR_PAIR_STATE_STORE";
    const char *disable_pair_proj_env =
        "DS4_METAL_DISABLE_COMPRESSOR_PAIR_PROJ";
    const char *disable_store_env =
        "DS4_METAL_DISABLE_COMPRESSOR_STORE_ONE";
    char *saved_force_pair = test_save_env(force_pair_env);
    char *saved_disable_pair_state = test_save_env(disable_pair_state_env);
    char *saved_disable_pair_proj = test_save_env(disable_pair_proj_env);
    char *saved_disable_store = test_save_env(disable_store_env);

    test_float_compare_stats out_stats[4] = {{0}};
    test_float_compare_stats state_stats[4] = {{0}};
    if (allocated) {
        memset(model_raw, 0, (size_t)model_bytes);
        for (uint32_t matrix = 0; matrix < 4u; matrix++) {
            uint16_t *weights = (uint16_t *)(
                (uint8_t *)model_raw + weight_offsets[matrix]);
            for (uint32_t o = 0; o < widths[matrix]; o++) {
                for (uint32_t i = 0; i < in_dim; i++) {
                    const int value =
                        (int)((o * (17u + 2u * matrix) +
                               i * (23u + 4u * matrix) +
                               (o ^ (i >> (matrix & 3u))) *
                                   (3u + 2u * matrix) +
                               seed * (29u + matrix)) % 127u) - 63;
                    weights[(uint64_t)o * in_dim + i] =
                        test_float_to_f16(
                            (float)value / (96.0f + 8.0f * matrix));
                }
            }
        }

        for (uint32_t which = 0; which < 2u; which++) {
            const uint64_t count =
                (uint64_t)ratio * ape_widths[which];
            if (ape_types[which] == 1u) {
                uint16_t *ape = (uint16_t *)(
                    (uint8_t *)model_raw + ape_offsets[which]);
                for (uint64_t i = 0; i < count; i++) {
                    const int value =
                        (int)((i * (13u + 2u * which) +
                               (i ^ (i >> 3u)) * (7u + 2u * which) +
                               seed * (17u + which)) % 61u) - 30;
                    ape[i] = test_float_to_f16(
                        (float)value / (80.0f + 8.0f * which));
                }
            } else {
                float *ape = (float *)(
                    (uint8_t *)model_raw + ape_offsets[which]);
                for (uint64_t i = 0; i < count; i++) {
                    const int value =
                        (int)((i * (13u + 2u * which) +
                               (i ^ (i >> 3u)) * (7u + 2u * which) +
                               seed * (17u + which)) % 61u) - 30;
                    ape[i] =
                        (float)value / (80.0f + 8.0f * which);
                }
            }
        }

        for (uint32_t i = 0; i < in_dim; i++) {
            const int value =
                (int)((i * 29u + (i ^ (i >> 4u)) * 9u +
                       seed * 11u) % 127u) - 63;
            x_host[i] = (float)value / 88.0f;
        }
        TEST_ASSERT(ds4_gpu_tensor_write(x, 0, x_host, x_bytes) != 0);

        for (uint32_t buffer = 0; buffer < 4u; buffer++) {
            const uint64_t out_bytes =
                (uint64_t)widths[buffer] * sizeof(float);
            const uint64_t state_count =
                (uint64_t)state_rows * widths[buffer];
            const uint64_t state_bytes = state_count * sizeof(float);
            for (uint32_t i = 0; i < widths[buffer]; i++) {
                const uint32_t poison =
                    0x7fc00001u + ((buffer * 1024u + i) & 0x3fffu);
                memcpy(ref_host + i, &poison, sizeof(poison));
            }
            TEST_ASSERT(ds4_gpu_tensor_write(
                            ref_out[buffer], 0, ref_host,
                            out_bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_write(
                            quad_out[buffer], 0, ref_host,
                            out_bytes) != 0);

            for (uint64_t i = 0; i < state_count; i++) {
                const int value =
                    (int)((i * (5u + 2u * buffer) +
                           seed * (13u + buffer)) % 193u) - 96;
                ref_host[i] =
                    (float)value / (64.0f + 8.0f * buffer);
            }
            TEST_ASSERT(ds4_gpu_tensor_write(
                            ref_state[buffer], 0, ref_host,
                            state_bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_write(
                            quad_state[buffer], 0, ref_host,
                            state_bytes) != 0);
        }

        TEST_ASSERT(ds4_gpu_set_model_map(model_raw, model_bytes) != 0);
        ds4_gpu_set_quality(false);
        TEST_ASSERT(setenv(force_pair_env, "1", 1) == 0);
        TEST_ASSERT(unsetenv(disable_pair_state_env) == 0);
        TEST_ASSERT(unsetenv(disable_pair_proj_env) == 0);
        TEST_ASSERT(unsetenv(disable_store_env) == 0);

        TEST_ASSERT(ds4_gpu_matmul_f16_pair_compressor_store_tensor(
                        ref_out[0], ref_out[1],
                        ref_state[0], ref_state[1],
                        model_raw, model_bytes,
                        weight_offsets[0], weight_offsets[1],
                        ape_offsets[0], ape_types[0],
                        in_dim, width0, x, ratio, pos) == 1);
        TEST_ASSERT(ds4_gpu_matmul_f16_pair_compressor_store_tensor(
                        ref_out[2], ref_out[3],
                        ref_state[2], ref_state[3],
                        model_raw, model_bytes,
                        weight_offsets[2], weight_offsets[3],
                        ape_offsets[1], ape_types[1],
                        in_dim, width1, x, ratio, pos) == 1);
        TEST_ASSERT(ds4_gpu_matmul_f16_quad_compressor_store_tensor(
                        quad_out[0], quad_out[1],
                        quad_out[2], quad_out[3],
                        quad_state[0], quad_state[1],
                        quad_state[2], quad_state[3],
                        model_raw, model_bytes,
                        weight_offsets[0], weight_offsets[1],
                        weight_offsets[2], weight_offsets[3],
                        ape_offsets[0], ape_types[0],
                        ape_offsets[1], ape_types[1],
                        in_dim, width0, width1, x, ratio, pos) == 1);

        for (uint32_t buffer = 0; buffer < 4u; buffer++) {
            const uint64_t out_bytes =
                (uint64_t)widths[buffer] * sizeof(float);
            const uint64_t state_count =
                (uint64_t)state_rows * widths[buffer];
            const uint64_t state_bytes = state_count * sizeof(float);
            TEST_ASSERT(ds4_gpu_tensor_read(
                            ref_out[buffer], 0, ref_host,
                            out_bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_read(
                            quad_out[buffer], 0, quad_host,
                            out_bytes) != 0);
            out_stats[buffer] = test_compare_float_bits(
                ref_host, quad_host, widths[buffer]);
            TEST_ASSERT(ds4_gpu_tensor_read(
                            ref_state[buffer], 0, ref_host,
                            state_bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_read(
                            quad_state[buffer], 0, quad_host,
                            state_bytes) != 0);
            state_stats[buffer] = test_compare_float_bits(
                ref_host, quad_host, (size_t)state_count);
        }
    }

    test_restore_env(force_pair_env, saved_force_pair);
    test_restore_env(disable_pair_state_env, saved_disable_pair_state);
    test_restore_env(disable_pair_proj_env, saved_disable_pair_proj);
    test_restore_env(disable_store_env, saved_disable_store);

    size_t out_mismatches = 0;
    size_t state_mismatches = 0;
    uint32_t max_out_ulp = 0;
    uint32_t max_state_ulp = 0;
    for (uint32_t i = 0; i < 4u; i++) {
        out_mismatches += out_stats[i].mismatch_count;
        state_mismatches += state_stats[i].mismatch_count;
        if (out_stats[i].max_ulp > max_out_ulp) {
            max_out_ulp = out_stats[i].max_ulp;
        }
        if (state_stats[i].max_ulp > max_state_ulp) {
            max_state_ulp = state_stats[i].max_ulp;
        }
        TEST_ASSERT(out_stats[i].mismatch_count == 0);
        TEST_ASSERT(state_stats[i].mismatch_count == 0);
    }
    fprintf(stderr,
            "ds4-test: compressor quad state-store exact width=%u/256 "
            "pos_mod4=%u ape=%s/%s outputs=%zu states=%zu "
            "max_ulp=%u/%u\n",
            width0, pos % ratio,
            ape0_type == 1u ? "f16" : "f32",
            ape1_type == 1u ? "f16" : "f32",
            out_mismatches, state_mismatches,
            max_out_ulp, max_state_ulp);

    free(quad_host);
    free(ref_host);
    free(x_host);
    for (uint32_t i = 0; i < 4u; i++) {
        ds4_gpu_tensor_free(quad_state[i]);
        ds4_gpu_tensor_free(ref_state[i]);
        ds4_gpu_tensor_free(quad_out[i]);
        ds4_gpu_tensor_free(ref_out[i]);
    }
    ds4_gpu_tensor_free(x);
    free(model_raw);
}

static void test_metal_f16_compressor_quad_state_store_exact(void) {
    test_metal_f16_compressor_quad_state_store_exact_case(
        1024u, 8u, 1u, 0u, 59u);
    test_metal_f16_compressor_quad_state_store_exact_case(
        512u, 11u, 0u, 1u, 71u);
}

static void test_metal_q8_qkv_compressor_compound_exact_case(
        uint32_t ratio,
        uint32_t pos,
        uint32_t ape0_type,
        uint32_t ape1_type,
        uint32_t seed,
        bool     batched_commands) {
    enum {
        Q_A_WEIGHT = 0,
        KV_WEIGHT,
        COMP0_KV_WEIGHT,
        COMP0_SCORE_WEIGHT,
        COMP1_KV_WEIGHT,
        COMP1_SCORE_WEIGHT,
        COMP0_APE,
        COMP1_APE,
        MODEL_RANGE_COUNT,
    };
    const uint32_t in_dim = 4096u;
    const uint32_t q_rank = 77u;
    const uint32_t kv_dim = 19u;
    const uint32_t width0 = ratio == 4u ? 1024u : 512u;
    const uint32_t width1 = ratio == 4u ? 256u : 0u;
    const uint32_t state_rows = ratio == 4u ? 2u * ratio : ratio;
    const uint32_t n_comp_outputs = width1 != 0u ? 4u : 2u;
    const uint32_t comp_widths[4] = {
        width0, width0, width1, width1,
    };
    const uint64_t page = (uint64_t)getpagesize();
    const uint64_t q8_row_bytes = (uint64_t)(in_dim / 32u) * 34u;
    const uint64_t f16_row_bytes =
        (uint64_t)in_dim * sizeof(uint16_t);
    uint64_t range_bytes[MODEL_RANGE_COUNT] = {
        (uint64_t)q_rank * q8_row_bytes,
        (uint64_t)kv_dim * q8_row_bytes,
        (uint64_t)width0 * f16_row_bytes,
        (uint64_t)width0 * f16_row_bytes,
        (uint64_t)width1 * f16_row_bytes,
        (uint64_t)width1 * f16_row_bytes,
        (uint64_t)ratio * width0 *
            (ape0_type == 1u ? sizeof(uint16_t) : sizeof(float)),
        (uint64_t)ratio * width1 *
            (ape1_type == 1u ? sizeof(uint16_t) : sizeof(float)),
    };
    uint64_t range_offsets[MODEL_RANGE_COUNT] = {0};
    uint64_t cursor = 0;
    for (uint32_t i = 0; i < MODEL_RANGE_COUNT; i++) {
        if (range_bytes[i] == 0u) continue;
        range_offsets[i] = test_round_up_u64(cursor, page);
        cursor = range_offsets[i] + range_bytes[i];
    }
    const uint64_t model_bytes = test_round_up_u64(cursor, page);
    const uint64_t x_bytes = (uint64_t)in_dim * sizeof(float);

    void *model_raw = NULL;
    TEST_ASSERT(posix_memalign(
                    &model_raw, (size_t)page, (size_t)model_bytes) == 0);
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(x_bytes);
    ds4_gpu_tensor *ref_q =
        ds4_gpu_tensor_alloc((uint64_t)q_rank * sizeof(float));
    ds4_gpu_tensor *ref_kv =
        ds4_gpu_tensor_alloc((uint64_t)kv_dim * sizeof(float));
    ds4_gpu_tensor *fused_q =
        ds4_gpu_tensor_alloc((uint64_t)q_rank * sizeof(float));
    ds4_gpu_tensor *fused_kv =
        ds4_gpu_tensor_alloc((uint64_t)kv_dim * sizeof(float));
    ds4_gpu_tensor *ref_out[4] = {0};
    ds4_gpu_tensor *fused_out[4] = {0};
    ds4_gpu_tensor *ref_state[4] = {0};
    ds4_gpu_tensor *fused_state[4] = {0};
    bool allocated = model_raw && x && ref_q && ref_kv && fused_q && fused_kv;
    for (uint32_t i = 0; i < n_comp_outputs; i++) {
        const uint64_t out_bytes =
            (uint64_t)comp_widths[i] * sizeof(float);
        const uint64_t state_bytes =
            (uint64_t)state_rows * comp_widths[i] * sizeof(float);
        ref_out[i] = ds4_gpu_tensor_alloc(out_bytes);
        fused_out[i] = ds4_gpu_tensor_alloc(out_bytes);
        ref_state[i] = ds4_gpu_tensor_alloc(state_bytes);
        fused_state[i] = ds4_gpu_tensor_alloc(state_bytes);
        TEST_ASSERT(ref_out[i] && fused_out[i] &&
                    ref_state[i] && fused_state[i]);
        allocated = allocated && ref_out[i] && fused_out[i] &&
                    ref_state[i] && fused_state[i];
    }

    uint64_t host_count = (uint64_t)state_rows * width0;
    if (host_count < q_rank) host_count = q_rank;
    if (host_count < kv_dim) host_count = kv_dim;
    float *x_host = malloc((size_t)x_bytes);
    float *ref_host = malloc((size_t)host_count * sizeof(float));
    float *fused_host = malloc((size_t)host_count * sizeof(float));
    TEST_ASSERT(x_host && ref_host && fused_host);
    allocated = allocated && x_host && ref_host && fused_host;

    const char *force_pair_env =
        "DS4_METAL_ENABLE_COMPRESSOR_PAIR_STATE_STORE";
    const char *disable_pair_state_env =
        "DS4_METAL_DISABLE_M3_COMPRESSOR_PAIR_STATE_STORE";
    const char *disable_pair_proj_env =
        "DS4_METAL_DISABLE_COMPRESSOR_PAIR_PROJ";
    const char *disable_store_env =
        "DS4_METAL_DISABLE_COMPRESSOR_STORE_ONE";
    const char *q8_nsg_env = "DS4_METAL_Q8_MV_NSG";
    char *saved_force_pair = test_save_env(force_pair_env);
    char *saved_disable_pair_state = test_save_env(disable_pair_state_env);
    char *saved_disable_pair_proj = test_save_env(disable_pair_proj_env);
    char *saved_disable_store = test_save_env(disable_store_env);
    char *saved_q8_nsg = test_save_env(q8_nsg_env);

    size_t q_mismatches = 0;
    size_t kv_mismatches = 0;
    size_t out_mismatches = 0;
    size_t state_mismatches = 0;
    uint32_t max_ulp = 0;
    if (allocated) {
        memset(model_raw, 0, (size_t)model_bytes);
        test_fill_q8_0_weights(
            (uint8_t *)model_raw + range_offsets[Q_A_WEIGHT],
            in_dim, q_rank, seed + 1u);
        test_fill_q8_0_weights(
            (uint8_t *)model_raw + range_offsets[KV_WEIGHT],
            in_dim, kv_dim, seed + 3u);

        for (uint32_t matrix = 0; matrix < n_comp_outputs; matrix++) {
            uint16_t *weights = (uint16_t *)(
                (uint8_t *)model_raw +
                range_offsets[COMP0_KV_WEIGHT + matrix]);
            const uint32_t width = comp_widths[matrix];
            for (uint32_t o = 0; o < width; o++) {
                for (uint32_t i = 0; i < in_dim; i++) {
                    const int value =
                        (int)((o * (13u + 2u * matrix) +
                               i * (19u + 4u * matrix) +
                               (o ^ (i >> (matrix & 3u))) *
                                   (5u + 2u * matrix) +
                               seed * (23u + matrix)) % 127u) - 63;
                    weights[(uint64_t)o * in_dim + i] =
                        test_float_to_f16(
                            (float)value / (88.0f + 8.0f * matrix));
                }
            }
        }

        const uint32_t ape_types[2] = {ape0_type, ape1_type};
        const uint32_t ape_widths[2] = {width0, width1};
        const uint32_t ape_ranges[2] = {COMP0_APE, COMP1_APE};
        for (uint32_t which = 0; which < 2u; which++) {
            if (ape_widths[which] == 0u) continue;
            const uint64_t count =
                (uint64_t)ratio * ape_widths[which];
            if (ape_types[which] == 1u) {
                uint16_t *ape = (uint16_t *)(
                    (uint8_t *)model_raw + range_offsets[ape_ranges[which]]);
                for (uint64_t i = 0; i < count; i++) {
                    const int value =
                        (int)((i * (11u + 2u * which) +
                               (i ^ (i >> 3u)) * (7u + 2u * which) +
                               seed * (17u + which)) % 61u) - 30;
                    ape[i] = test_float_to_f16(
                        (float)value / (72.0f + 8.0f * which));
                }
            } else {
                float *ape = (float *)(
                    (uint8_t *)model_raw + range_offsets[ape_ranges[which]]);
                for (uint64_t i = 0; i < count; i++) {
                    const int value =
                        (int)((i * (11u + 2u * which) +
                               (i ^ (i >> 3u)) * (7u + 2u * which) +
                               seed * (17u + which)) % 61u) - 30;
                    ape[i] =
                        (float)value / (72.0f + 8.0f * which);
                }
            }
        }

        for (uint32_t i = 0; i < in_dim; i++) {
            const int value =
                (int)((i * 29u + (i ^ (i >> 4u)) * 9u +
                       seed * 11u) % 127u) - 63;
            x_host[i] = (float)value / 84.0f;
        }
        TEST_ASSERT(ds4_gpu_tensor_write(x, 0, x_host, x_bytes) != 0);

        for (uint32_t buffer = 0; buffer < n_comp_outputs; buffer++) {
            const uint32_t width = comp_widths[buffer];
            const uint64_t out_bytes =
                (uint64_t)width * sizeof(float);
            const uint64_t state_count =
                (uint64_t)state_rows * width;
            const uint64_t state_bytes = state_count * sizeof(float);
            for (uint32_t i = 0; i < width; i++) {
                const uint32_t poison =
                    0x7fc10001u + ((buffer * 2048u + i) & 0x7fffu);
                memcpy(ref_host + i, &poison, sizeof(poison));
            }
            TEST_ASSERT(ds4_gpu_tensor_write(
                            ref_out[buffer], 0, ref_host,
                            out_bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_write(
                            fused_out[buffer], 0, ref_host,
                            out_bytes) != 0);
            for (uint64_t i = 0; i < state_count; i++) {
                const int value =
                    (int)((i * (5u + 2u * buffer) +
                           seed * (13u + buffer)) % 193u) - 96;
                ref_host[i] =
                    (float)value / (64.0f + 8.0f * buffer);
            }
            TEST_ASSERT(ds4_gpu_tensor_write(
                            ref_state[buffer], 0, ref_host,
                            state_bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_write(
                            fused_state[buffer], 0, ref_host,
                            state_bytes) != 0);
        }

        TEST_ASSERT(ds4_gpu_set_model_map(model_raw, model_bytes) != 0);
        ds4_gpu_set_quality(false);
        TEST_ASSERT(setenv(force_pair_env, "1", 1) == 0);
        TEST_ASSERT(unsetenv(disable_pair_state_env) == 0);
        TEST_ASSERT(unsetenv(disable_pair_proj_env) == 0);
        TEST_ASSERT(unsetenv(disable_store_env) == 0);
        TEST_ASSERT(unsetenv(q8_nsg_env) == 0);

        TEST_ASSERT(ds4_gpu_matmul_q8_0_pair_tensor(
                        ref_q, ref_kv,
                        model_raw, model_bytes,
                        range_offsets[Q_A_WEIGHT],
                        range_offsets[KV_WEIGHT],
                        in_dim, q_rank, kv_dim, x, 1u) == 1);
        TEST_ASSERT(ds4_gpu_matmul_f16_pair_compressor_store_tensor(
                        ref_out[0], ref_out[1],
                        ref_state[0], ref_state[1],
                        model_raw, model_bytes,
                        range_offsets[COMP0_KV_WEIGHT],
                        range_offsets[COMP0_SCORE_WEIGHT],
                        range_offsets[COMP0_APE], ape0_type,
                        in_dim, width0, x, ratio, pos) == 1);
        if (width1 != 0u) {
            TEST_ASSERT(ds4_gpu_matmul_f16_pair_compressor_store_tensor(
                            ref_out[2], ref_out[3],
                            ref_state[2], ref_state[3],
                            model_raw, model_bytes,
                            range_offsets[COMP1_KV_WEIGHT],
                            range_offsets[COMP1_SCORE_WEIGHT],
                            range_offsets[COMP1_APE], ape1_type,
                            in_dim, width1, x, ratio, pos) == 1);
        }

        bool commands_open = false;
        if (batched_commands) {
            commands_open = ds4_gpu_begin_commands() != 0;
            TEST_ASSERT(commands_open);
        }
        const int fused =
            ds4_gpu_qkv_pair_quad_compressor_store_tensor(
                fused_q, fused_kv,
                fused_out[0], fused_out[1],
                width1 != 0u ? fused_out[2] : NULL,
                width1 != 0u ? fused_out[3] : NULL,
                fused_state[0], fused_state[1],
                width1 != 0u ? fused_state[2] : NULL,
                width1 != 0u ? fused_state[3] : NULL,
                model_raw, model_bytes,
                range_offsets[Q_A_WEIGHT],
                range_offsets[KV_WEIGHT],
                range_offsets[COMP0_KV_WEIGHT],
                range_offsets[COMP0_SCORE_WEIGHT],
                range_offsets[COMP1_KV_WEIGHT],
                range_offsets[COMP1_SCORE_WEIGHT],
                range_offsets[COMP0_APE], ape0_type,
                range_offsets[COMP1_APE], ape1_type,
                in_dim, q_rank, kv_dim, width0, width1,
                x, ratio, pos);
        if (commands_open) {
            TEST_ASSERT(ds4_gpu_end_commands() != 0);
        }
        TEST_ASSERT(fused == 1);

        TEST_ASSERT(ds4_gpu_tensor_read(
                        ref_q, 0, ref_host,
                        (uint64_t)q_rank * sizeof(float)) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        fused_q, 0, fused_host,
                        (uint64_t)q_rank * sizeof(float)) != 0);
        test_float_compare_stats stats =
            test_compare_float_bits(ref_host, fused_host, q_rank);
        q_mismatches = stats.mismatch_count;
        if (stats.max_ulp > max_ulp) max_ulp = stats.max_ulp;

        TEST_ASSERT(ds4_gpu_tensor_read(
                        ref_kv, 0, ref_host,
                        (uint64_t)kv_dim * sizeof(float)) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        fused_kv, 0, fused_host,
                        (uint64_t)kv_dim * sizeof(float)) != 0);
        stats = test_compare_float_bits(ref_host, fused_host, kv_dim);
        kv_mismatches = stats.mismatch_count;
        if (stats.max_ulp > max_ulp) max_ulp = stats.max_ulp;

        for (uint32_t buffer = 0; buffer < n_comp_outputs; buffer++) {
            const uint32_t width = comp_widths[buffer];
            const uint64_t out_bytes =
                (uint64_t)width * sizeof(float);
            const uint64_t state_count =
                (uint64_t)state_rows * width;
            const uint64_t state_bytes = state_count * sizeof(float);
            TEST_ASSERT(ds4_gpu_tensor_read(
                            ref_out[buffer], 0, ref_host,
                            out_bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_read(
                            fused_out[buffer], 0, fused_host,
                            out_bytes) != 0);
            stats = test_compare_float_bits(ref_host, fused_host, width);
            out_mismatches += stats.mismatch_count;
            if (stats.max_ulp > max_ulp) max_ulp = stats.max_ulp;

            TEST_ASSERT(ds4_gpu_tensor_read(
                            ref_state[buffer], 0, ref_host,
                            state_bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_read(
                            fused_state[buffer], 0, fused_host,
                            state_bytes) != 0);
            stats = test_compare_float_bits(
                ref_host, fused_host, (size_t)state_count);
            state_mismatches += stats.mismatch_count;
            if (stats.max_ulp > max_ulp) max_ulp = stats.max_ulp;
        }
    }

    fprintf(stderr,
            "ds4-test: Q8 QKV/compressor compound exact ratio=%u "
            "pos_mod=%u batch=%u q=%zu kv=%zu outputs=%zu states=%zu "
            "max_ulp=%u\n",
            ratio, pos % ratio, batched_commands ? 1u : 0u,
            q_mismatches, kv_mismatches,
            out_mismatches, state_mismatches, max_ulp);
    TEST_ASSERT(q_mismatches == 0);
    TEST_ASSERT(kv_mismatches == 0);
    TEST_ASSERT(out_mismatches == 0);
    TEST_ASSERT(state_mismatches == 0);
    TEST_ASSERT(max_ulp == 0);

    test_restore_env(q8_nsg_env, saved_q8_nsg);
    test_restore_env(disable_store_env, saved_disable_store);
    test_restore_env(disable_pair_proj_env, saved_disable_pair_proj);
    test_restore_env(disable_pair_state_env, saved_disable_pair_state);
    test_restore_env(force_pair_env, saved_force_pair);
    free(fused_host);
    free(ref_host);
    free(x_host);
    for (uint32_t i = 0; i < 4u; i++) {
        ds4_gpu_tensor_free(fused_state[i]);
        ds4_gpu_tensor_free(ref_state[i]);
        ds4_gpu_tensor_free(fused_out[i]);
        ds4_gpu_tensor_free(ref_out[i]);
    }
    ds4_gpu_tensor_free(fused_kv);
    ds4_gpu_tensor_free(fused_q);
    ds4_gpu_tensor_free(ref_kv);
    ds4_gpu_tensor_free(ref_q);
    ds4_gpu_tensor_free(x);
    free(model_raw);
}

static void test_metal_q8_qkv_compressor_compound_exact(void) {
    test_metal_q8_qkv_compressor_compound_exact_case(
        4u, 11u, 1u, 0u, 79u, false);
    test_metal_q8_qkv_compressor_compound_exact_case(
        4u, 11u, 1u, 0u, 79u, true);
    test_metal_q8_qkv_compressor_compound_exact_case(
        128u, 255u, 0u, 1u, 83u, false);
    test_metal_q8_qkv_compressor_compound_exact_case(
        128u, 255u, 0u, 1u, 83u, true);
}

static void test_metal_compressor_ape_add_exact_case(
        uint32_t head_dim,
        uint32_t ratio,
        uint32_t pos0,
        uint32_t n_tokens,
        uint32_t ape_type,
        uint32_t seed,
        bool test_pack_fusion) {
    const uint32_t coff = ratio == 4u ? 2u : 1u;
    const uint32_t width = coff * head_dim;
    const uint32_t state_rows = coff * ratio;
    const uint32_t n_comp = n_tokens / ratio;
    const uint64_t input_count = (uint64_t)n_tokens * width;
    const uint64_t state_count = (uint64_t)state_rows * width;
    const uint64_t comp_count = (uint64_t)n_comp * head_dim;
    const uint64_t input_bytes = input_count * sizeof(float);
    const uint64_t state_bytes = state_count * sizeof(float);
    const uint64_t comp_bytes = comp_count * sizeof(float);
    const uint64_t ape_elem_bytes = ape_type == 1u ? 2u : 4u;
    const uint64_t ape_bytes = (uint64_t)width * ratio * ape_elem_bytes;
    const uint64_t page = (uint64_t)getpagesize();
    const uint64_t norm_offset = test_round_up_u64(ape_bytes, page);
    const uint64_t model_bytes = test_round_up_u64(
        norm_offset + (uint64_t)head_dim * sizeof(float), page);

    ds4_gpu_tensor *kv = ds4_gpu_tensor_alloc(input_bytes);
    ds4_gpu_tensor *sc = ds4_gpu_tensor_alloc(input_bytes);
    ds4_gpu_tensor *ref_comp = ds4_gpu_tensor_alloc(comp_bytes);
    ds4_gpu_tensor *fused_comp = ds4_gpu_tensor_alloc(comp_bytes);
    ds4_gpu_tensor *ref_state_kv = ds4_gpu_tensor_alloc(state_bytes);
    ds4_gpu_tensor *ref_state_score = ds4_gpu_tensor_alloc(state_bytes);
    ds4_gpu_tensor *fused_state_kv = ds4_gpu_tensor_alloc(state_bytes);
    ds4_gpu_tensor *fused_state_score = ds4_gpu_tensor_alloc(state_bytes);
    TEST_ASSERT(kv != NULL);
    TEST_ASSERT(sc != NULL);
    TEST_ASSERT(ref_comp != NULL);
    TEST_ASSERT(fused_comp != NULL);
    TEST_ASSERT(ref_state_kv != NULL);
    TEST_ASSERT(ref_state_score != NULL);
    TEST_ASSERT(fused_state_kv != NULL);
    TEST_ASSERT(fused_state_score != NULL);

    float *kv_host = malloc((size_t)input_bytes);
    float *sc_host = malloc((size_t)input_bytes);
    float *ref_comp_host = malloc((size_t)comp_bytes);
    float *fused_comp_host = malloc((size_t)comp_bytes);
    float *ref_state_kv_host = malloc((size_t)state_bytes);
    float *ref_state_score_host = malloc((size_t)state_bytes);
    float *fused_state_kv_host = malloc((size_t)state_bytes);
    float *fused_state_score_host = malloc((size_t)state_bytes);
    const uint64_t poison_count = input_count > state_count ?
        input_count : state_count;
    float *poison_host = test_pack_fusion ?
        malloc((size_t)(poison_count * sizeof(float))) : NULL;
    TEST_ASSERT(kv_host != NULL);
    TEST_ASSERT(sc_host != NULL);
    TEST_ASSERT(ref_comp_host != NULL);
    TEST_ASSERT(fused_comp_host != NULL);
    TEST_ASSERT(ref_state_kv_host != NULL);
    TEST_ASSERT(ref_state_score_host != NULL);
    TEST_ASSERT(fused_state_kv_host != NULL);
    TEST_ASSERT(fused_state_score_host != NULL);
    TEST_ASSERT(!test_pack_fusion || poison_host != NULL);

    void *model_raw = NULL;
    TEST_ASSERT(posix_memalign(&model_raw, (size_t)page,
                               (size_t)model_bytes) == 0);
    const bool allocated = kv && sc && ref_comp && fused_comp && ref_state_kv &&
        ref_state_score && fused_state_kv && fused_state_score && kv_host &&
        sc_host && ref_comp_host && fused_comp_host && ref_state_kv_host &&
        ref_state_score_host && fused_state_kv_host && fused_state_score_host &&
        (!test_pack_fusion || poison_host) && model_raw;
    const char *disable_env = "DS4_METAL_DISABLE_COMPRESSOR_APE_ADD";
    const char *pack_disable_env =
        "DS4_METAL_DISABLE_COMPRESSOR_RATIO4_PACK_FUSION";
    char *saved_disable = test_save_env(disable_env);
    char *saved_pack_disable = test_save_env(pack_disable_env);
    test_float_compare_stats comp_stats = {0};
    test_float_compare_stats state_kv_stats = {0};
    test_float_compare_stats state_score_stats = {0};

    if (allocated) {
        memset(model_raw, 0, (size_t)model_bytes);
        if (ape_type == 1u) {
            uint16_t *ape = model_raw;
            for (uint64_t i = 0; i < (uint64_t)width * ratio; i++) {
                const int value = (int)((i * 17u + (i ^ (i >> 4u)) * 5u +
                                         seed * 13u) % 127u) - 63;
                ape[i] = test_float_to_f16((float)value / 80.0f);
            }
        } else {
            float *ape = model_raw;
            for (uint64_t i = 0; i < (uint64_t)width * ratio; i++) {
                const int value = (int)((i * 17u + (i ^ (i >> 4u)) * 5u +
                                         seed * 13u) % 127u) - 63;
                ape[i] = (float)value / 80.0f;
            }
        }
        float *norm = (float *)((uint8_t *)model_raw + norm_offset);
        for (uint32_t i = 0; i < head_dim; i++) {
            norm[i] = 0.75f + (float)((i * 7u + seed) % 19u) / 64.0f;
        }
        for (uint64_t i = 0; i < input_count; i++) {
            const int kv_value = (int)((i * 29u + (i ^ (i >> 5u)) * 3u +
                                        seed * 23u) % 181u) - 90;
            const int sc_value = (int)((i * 31u + (i ^ (i >> 3u)) * 11u +
                                        seed * 17u) % 173u) - 86;
            kv_host[i] = (float)kv_value / 112.0f;
            sc_host[i] = (float)sc_value / 96.0f;
        }
        kv_host[0] = -0.0f;
        sc_host[0] = -0.0f;

        // Exercise exact-add edge values in the first active APE row. Using
        // bit patterns avoids host fast-math rewriting signed zeros or
        // subnormals before the legacy and fused Metal paths see them.
        const uint32_t cutoff = (n_tokens / ratio) * ratio;
        const uint32_t edge_token = cutoff < n_tokens ? cutoff : cutoff - ratio;
        const uint64_t edge_ape =
            (uint64_t)((pos0 + edge_token) % ratio) * width;
        float *edge_score = sc_host + (uint64_t)edge_token * width;
        static const uint16_t edge_f16[] = {
            0x0000u, 0x8000u, 0x0001u, 0x8001u,
            0x3c00u, 0xbc00u, 0x7bffu, 0xfbffu,
        };
        static const uint32_t edge_f32[] = {
            0x00000000u, 0x80000000u, 0x00000001u, 0x80000001u,
            0x3f800000u, 0xbf800000u, 0x7f7fffffu, 0xff7fffffu,
        };
        static const uint32_t edge_score_f16[] = {
            0x80000000u, 0x00000000u, 0x00000000u, 0x80000000u,
            0xbf800000u, 0x3f800000u, 0xc77fe000u, 0x477fe000u,
        };
        static const uint32_t edge_score_f32[] = {
            0x80000000u, 0x00000000u, 0x00000000u, 0x80000000u,
            0xbf800000u, 0x3f800000u, 0xff7fffffu, 0x7f7fffffu,
        };
        if (ape_type == 1u) {
            uint16_t *ape = model_raw;
            memcpy(ape + edge_ape, edge_f16, sizeof(edge_f16));
            for (uint32_t i = 0; i < 8u; i++) {
                memcpy(edge_score + i, edge_score_f16 + i, sizeof(uint32_t));
            }
        } else {
            uint32_t *ape = model_raw;
            memcpy(ape + edge_ape, edge_f32, sizeof(edge_f32));
            for (uint32_t i = 0; i < 8u; i++) {
                memcpy(edge_score + i, edge_score_f32 + i, sizeof(uint32_t));
            }
        }
        for (uint64_t i = 0; i < state_count; i++) {
            ref_state_kv_host[i] = 1234.0f;
            fused_state_kv_host[i] = 1234.0f;
            ref_state_score_host[i] = -1234.0f;
            fused_state_score_host[i] = -1234.0f;
        }

        TEST_ASSERT(ds4_gpu_tensor_write(kv, 0, kv_host, input_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(sc, 0, sc_host, input_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        ref_state_kv, 0, ref_state_kv_host, state_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        fused_state_kv, 0, fused_state_kv_host, state_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        ref_state_score, 0, ref_state_score_host, state_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        fused_state_score, 0, fused_state_score_host, state_bytes) != 0);
        TEST_ASSERT(ds4_gpu_set_model_map(model_raw, model_bytes) != 0);
        ds4_gpu_set_quality(false);

        if (test_pack_fusion) {
            TEST_ASSERT(unsetenv(disable_env) == 0);
        } else {
            TEST_ASSERT(setenv(disable_env, "1", 1) == 0);
        }
        TEST_ASSERT(setenv(pack_disable_env, "1", 1) == 0);
        TEST_ASSERT(ds4_gpu_compressor_prefill_tensor(
            ref_comp, ref_state_kv, ref_state_score, kv, sc,
            model_raw, model_bytes, 0, ape_type, norm_offset, 0,
            head_dim, ratio, pos0, n_tokens, 0, 0, false,
            10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f, 1.0e-6f) != 0);

        if (test_pack_fusion) {
            // Overwrite every persistent pack cell with qNaN payloads through
            // the legacy replay path. This makes an omitted candidate write
            // observable even for plane-zero padding, which normal legacy
            // packing would otherwise leave at the correct 0/-inf values.
            for (uint64_t i = 0; i < poison_count; i++) {
                const uint32_t bits =
                    0x7fc00001u + (uint32_t)(i & 0x3ffu);
                memcpy(poison_host + i, &bits, sizeof(bits));
            }
            TEST_ASSERT(ds4_gpu_tensor_write(
                            kv, 0, poison_host, input_bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_write(
                            sc, 0, poison_host, input_bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_write(
                            fused_state_kv, 0, poison_host,
                            state_bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_write(
                            fused_state_score, 0, poison_host,
                            state_bytes) != 0);
            TEST_ASSERT(ds4_gpu_compressor_prefill_ratio4_replay_tensor(
                fused_comp, fused_state_kv, fused_state_score, kv, sc,
                model_raw, model_bytes, 0, ape_type, norm_offset, 0,
                head_dim, 0, n_comp * ratio, 0, 0, false,
                10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f, 1.0e-6f) != 0);
            TEST_ASSERT(ds4_gpu_tensor_write(
                            kv, 0, kv_host, input_bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_write(
                            sc, 0, sc_host, input_bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_write(
                            fused_state_kv, 0, fused_state_kv_host,
                            state_bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_write(
                            fused_state_score, 0, fused_state_score_host,
                            state_bytes) != 0);
            TEST_ASSERT(unsetenv(pack_disable_env) == 0);
        } else {
            TEST_ASSERT(unsetenv(disable_env) == 0);
        }
        TEST_ASSERT(ds4_gpu_compressor_prefill_tensor(
            fused_comp, fused_state_kv, fused_state_score, kv, sc,
            model_raw, model_bytes, 0, ape_type, norm_offset, 0,
            head_dim, ratio, pos0, n_tokens, 0, 0, false,
            10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f, 1.0e-6f) != 0);

        if (test_pack_fusion) {
            TEST_ASSERT(ds4_gpu_tensor_read(
                            kv, 0, poison_host, input_bytes) != 0);
            TEST_ASSERT(memcmp(kv_host, poison_host,
                               (size_t)input_bytes) == 0);
            TEST_ASSERT(ds4_gpu_tensor_read(
                            sc, 0, poison_host, input_bytes) != 0);
            TEST_ASSERT(memcmp(sc_host, poison_host,
                               (size_t)input_bytes) == 0);
        }
        TEST_ASSERT(ds4_gpu_tensor_read(
                        ref_comp, 0, ref_comp_host, comp_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        fused_comp, 0, fused_comp_host, comp_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        ref_state_kv, 0, ref_state_kv_host, state_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        fused_state_kv, 0, fused_state_kv_host, state_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        ref_state_score, 0, ref_state_score_host, state_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        fused_state_score, 0, fused_state_score_host,
                        state_bytes) != 0);

        comp_stats = test_compare_float_bits(
            ref_comp_host, fused_comp_host, (size_t)comp_count);
        state_kv_stats = test_compare_float_bits(
            ref_state_kv_host, fused_state_kv_host, (size_t)state_count);
        state_score_stats = test_compare_float_bits(
            ref_state_score_host, fused_state_score_host, (size_t)state_count);
    }

    test_restore_env(disable_env, saved_disable);
    test_restore_env(pack_disable_env, saved_pack_disable);
    fprintf(stderr,
            "ds4-test: compressor %s exact head=%u ratio=%u pos=%u "
            "tokens=%u ape=%s comp=%zu/%llu state_kv=%zu/%llu "
            "state_score=%zu/%llu max_ulp=%u/%u/%u\n",
            test_pack_fusion ? "ratio4 pack" : "APE add",
            head_dim, ratio, pos0, n_tokens, ape_type == 1u ? "f16" : "f32",
            comp_stats.mismatch_count, (unsigned long long)comp_count,
            state_kv_stats.mismatch_count, (unsigned long long)state_count,
            state_score_stats.mismatch_count, (unsigned long long)state_count,
            comp_stats.max_ulp, state_kv_stats.max_ulp,
            state_score_stats.max_ulp);
    TEST_ASSERT(comp_stats.mismatch_count == 0);
    TEST_ASSERT(state_kv_stats.mismatch_count == 0);
    TEST_ASSERT(state_score_stats.mismatch_count == 0);

    free(model_raw);
    free(poison_host);
    free(fused_state_score_host);
    free(fused_state_kv_host);
    free(ref_state_score_host);
    free(ref_state_kv_host);
    free(fused_comp_host);
    free(ref_comp_host);
    free(sc_host);
    free(kv_host);
    ds4_gpu_tensor_free(fused_state_score);
    ds4_gpu_tensor_free(fused_state_kv);
    ds4_gpu_tensor_free(ref_state_score);
    ds4_gpu_tensor_free(ref_state_kv);
    ds4_gpu_tensor_free(fused_comp);
    ds4_gpu_tensor_free(ref_comp);
    ds4_gpu_tensor_free(sc);
    ds4_gpu_tensor_free(kv);
}

static void test_metal_compressor_ape_add_exact(void) {
    test_metal_compressor_ape_add_exact_case(128, 4, 3, 16, 1, 7, false);
    test_metal_compressor_ape_add_exact_case(512, 4, 0, 16, 1, 13, false);
    test_metal_compressor_ape_add_exact_case(512, 4, 1, 14, 0, 19, false);
    test_metal_compressor_ape_add_exact_case(512, 128, 127, 257, 1, 31, false);
}

static void test_metal_compressor_ratio4_pack_exact(void) {
    test_metal_compressor_ape_add_exact_case(128, 4, 0, 4, 1, 37, true);
    test_metal_compressor_ape_add_exact_case(512, 4, 0, 8, 1, 41, true);
    test_metal_compressor_ape_add_exact_case(512, 4, 1, 14, 0, 43, true);
}

static void test_metal_compressor_ratio4_replay_pack_exact_case(
        uint32_t head_dim,
        uint32_t n_tokens,
        uint32_t seed) {
    const uint32_t width = 2u * head_dim;
    const uint32_t n_comp = n_tokens / 4u;
    const uint64_t input_count = (uint64_t)n_tokens * width;
    const uint64_t state_count = (uint64_t)8u * width;
    const uint64_t comp_count = (uint64_t)n_comp * head_dim;
    const uint64_t input_bytes = input_count * sizeof(float);
    const uint64_t state_bytes = state_count * sizeof(float);
    const uint64_t comp_bytes = comp_count * sizeof(float);
    const uint64_t ape_bytes = (uint64_t)4u * width * sizeof(uint16_t);
    const uint64_t page = (uint64_t)getpagesize();
    const uint64_t norm_offset = test_round_up_u64(ape_bytes, page);
    const uint64_t model_bytes = test_round_up_u64(
        norm_offset + (uint64_t)head_dim * sizeof(float), page);

    ds4_gpu_tensor *kv = ds4_gpu_tensor_alloc(input_bytes);
    ds4_gpu_tensor *sc = ds4_gpu_tensor_alloc(input_bytes);
    ds4_gpu_tensor *ref_comp = ds4_gpu_tensor_alloc(comp_bytes);
    ds4_gpu_tensor *fused_comp = ds4_gpu_tensor_alloc(comp_bytes);
    ds4_gpu_tensor *ref_state_kv = ds4_gpu_tensor_alloc(state_bytes);
    ds4_gpu_tensor *ref_state_score = ds4_gpu_tensor_alloc(state_bytes);
    ds4_gpu_tensor *fused_state_kv = ds4_gpu_tensor_alloc(state_bytes);
    ds4_gpu_tensor *fused_state_score = ds4_gpu_tensor_alloc(state_bytes);
    float *kv_host = malloc((size_t)input_bytes);
    float *sc_host = malloc((size_t)input_bytes);
    float *state_kv_host = malloc((size_t)state_bytes);
    float *state_score_host = malloc((size_t)state_bytes);
    float *source_after_host = malloc((size_t)input_bytes);
    float *ref_comp_host = malloc((size_t)comp_bytes);
    float *fused_comp_host = malloc((size_t)comp_bytes);
    float *ref_state_kv_host = malloc((size_t)state_bytes);
    float *fused_state_kv_host = malloc((size_t)state_bytes);
    float *ref_state_score_host = malloc((size_t)state_bytes);
    float *fused_state_score_host = malloc((size_t)state_bytes);
    void *model_raw = NULL;
    TEST_ASSERT(posix_memalign(
                    &model_raw, (size_t)page, (size_t)model_bytes) == 0);

    TEST_ASSERT(kv != NULL);
    TEST_ASSERT(sc != NULL);
    TEST_ASSERT(ref_comp != NULL);
    TEST_ASSERT(fused_comp != NULL);
    TEST_ASSERT(ref_state_kv != NULL);
    TEST_ASSERT(ref_state_score != NULL);
    TEST_ASSERT(fused_state_kv != NULL);
    TEST_ASSERT(fused_state_score != NULL);
    TEST_ASSERT(kv_host != NULL);
    TEST_ASSERT(sc_host != NULL);
    TEST_ASSERT(state_kv_host != NULL);
    TEST_ASSERT(state_score_host != NULL);
    TEST_ASSERT(source_after_host != NULL);
    TEST_ASSERT(ref_comp_host != NULL);
    TEST_ASSERT(fused_comp_host != NULL);
    TEST_ASSERT(ref_state_kv_host != NULL);
    TEST_ASSERT(fused_state_kv_host != NULL);
    TEST_ASSERT(ref_state_score_host != NULL);
    TEST_ASSERT(fused_state_score_host != NULL);
    TEST_ASSERT(model_raw != NULL);

    const char *ape_disable_env = "DS4_METAL_DISABLE_COMPRESSOR_APE_ADD";
    const char *pack_disable_env =
        "DS4_METAL_DISABLE_COMPRESSOR_RATIO4_PACK_FUSION";
    char *saved_ape_disable = test_save_env(ape_disable_env);
    char *saved_pack_disable = test_save_env(pack_disable_env);
    test_float_compare_stats comp_stats = {0};
    test_float_compare_stats state_kv_stats = {0};
    test_float_compare_stats state_score_stats = {0};

    const bool allocated = kv && sc && ref_comp && fused_comp &&
        ref_state_kv && ref_state_score && fused_state_kv &&
        fused_state_score && kv_host && sc_host && state_kv_host &&
        state_score_host && source_after_host && ref_comp_host && fused_comp_host &&
        ref_state_kv_host && fused_state_kv_host && ref_state_score_host &&
        fused_state_score_host && model_raw;
    if (allocated) {
        memset(model_raw, 0, (size_t)model_bytes);
        uint16_t *ape = model_raw;
        for (uint64_t i = 0; i < (uint64_t)4u * width; i++) {
            const int value =
                (int)((i * 19u + (i ^ (i >> 3u)) * 7u + seed * 11u) %
                      113u) - 56;
            ape[i] = test_float_to_f16((float)value / 72.0f);
        }
        float *norm = (float *)((uint8_t *)model_raw + norm_offset);
        for (uint32_t i = 0; i < head_dim; i++) {
            norm[i] = 0.875f + (float)((i * 5u + seed) % 17u) / 64.0f;
        }
        for (uint64_t i = 0; i < input_count; i++) {
            const int kv_value =
                (int)((i * 31u + (i ^ (i >> 4u)) * 5u + seed * 13u) %
                      193u) - 96;
            const int sc_value =
                (int)((i * 37u + (i ^ (i >> 5u)) * 9u + seed * 17u) %
                      181u) - 90;
            kv_host[i] = (float)kv_value / 104.0f;
            sc_host[i] = (float)sc_value / 88.0f;
        }
        for (uint64_t i = 0; i < state_count; i++) {
            const int kv_value =
                (int)((i * 23u + (i ^ (i >> 2u)) * 3u + seed * 29u) %
                      167u) - 83;
            const int sc_value =
                (int)((i * 41u + (i ^ (i >> 6u)) * 11u + seed * 7u) %
                      157u) - 78;
            state_kv_host[i] = (float)kv_value / 80.0f;
            state_score_host[i] = (float)sc_value / 92.0f;
        }
        const uint32_t negative_zero = 0x80000000u;
        memcpy(kv_host, &negative_zero, sizeof(negative_zero));
        memcpy(state_kv_host, &negative_zero, sizeof(negative_zero));

        TEST_ASSERT(ds4_gpu_tensor_write(kv, 0, kv_host, input_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(sc, 0, sc_host, input_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        ref_state_kv, 0, state_kv_host, state_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        fused_state_kv, 0, state_kv_host, state_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        ref_state_score, 0, state_score_host, state_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        fused_state_score, 0, state_score_host, state_bytes) != 0);
        TEST_ASSERT(ds4_gpu_set_model_map(model_raw, model_bytes) != 0);
        ds4_gpu_set_quality(false);
        TEST_ASSERT(unsetenv(ape_disable_env) == 0);
        TEST_ASSERT(setenv(pack_disable_env, "1", 1) == 0);

        TEST_ASSERT(ds4_gpu_compressor_prefill_ratio4_replay_tensor(
            ref_comp, ref_state_kv, ref_state_score, kv, sc,
            model_raw, model_bytes, 0, 1, norm_offset, 0,
            head_dim, 0, n_tokens, 0, 0, false,
            10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f, 1.0e-6f) != 0);

        // Poison every persistent pack cell through the legacy full-fill path
        // so a missing candidate write cannot inherit the reference value.
        for (uint64_t i = 0; i < input_count; i++) {
            const uint32_t bits = 0x7fc00001u + (uint32_t)(i & 0x3ffu);
            memcpy(kv_host + i, &bits, sizeof(bits));
            memcpy(sc_host + i, &bits, sizeof(bits));
        }
        for (uint64_t i = 0; i < state_count; i++) {
            const uint32_t bits = 0x7fc00401u + (uint32_t)(i & 0x3ffu);
            memcpy(ref_state_kv_host + i, &bits, sizeof(bits));
            memcpy(ref_state_score_host + i, &bits, sizeof(bits));
        }
        TEST_ASSERT(ds4_gpu_tensor_write(kv, 0, kv_host, input_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(sc, 0, sc_host, input_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        fused_state_kv, 0, ref_state_kv_host, state_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        fused_state_score, 0, ref_state_score_host,
                        state_bytes) != 0);
        TEST_ASSERT(ds4_gpu_compressor_prefill_ratio4_replay_tensor(
            fused_comp, fused_state_kv, fused_state_score, kv, sc,
            model_raw, model_bytes, 0, 1, norm_offset, 0,
            head_dim, 0, n_tokens, 0, 0, false,
            10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f, 1.0e-6f) != 0);

        // Restore finite sources and the candidate's original replay state.
        for (uint64_t i = 0; i < input_count; i++) {
            const int kv_value =
                (int)((i * 31u + (i ^ (i >> 4u)) * 5u + seed * 13u) %
                      193u) - 96;
            const int sc_value =
                (int)((i * 37u + (i ^ (i >> 5u)) * 9u + seed * 17u) %
                      181u) - 90;
            kv_host[i] = (float)kv_value / 104.0f;
            sc_host[i] = (float)sc_value / 88.0f;
        }
        memcpy(kv_host, &negative_zero, sizeof(negative_zero));
        TEST_ASSERT(ds4_gpu_tensor_write(kv, 0, kv_host, input_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(sc, 0, sc_host, input_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        fused_state_kv, 0, state_kv_host, state_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        fused_state_score, 0, state_score_host,
                        state_bytes) != 0);
        TEST_ASSERT(unsetenv(pack_disable_env) == 0);
        TEST_ASSERT(ds4_gpu_compressor_prefill_ratio4_replay_tensor(
            fused_comp, fused_state_kv, fused_state_score, kv, sc,
            model_raw, model_bytes, 0, 1, norm_offset, 0,
            head_dim, 0, n_tokens, 0, 0, false,
            10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f, 1.0e-6f) != 0);

        TEST_ASSERT(ds4_gpu_tensor_read(
                        kv, 0, source_after_host, input_bytes) != 0);
        TEST_ASSERT(memcmp(kv_host, source_after_host,
                           (size_t)input_bytes) == 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        sc, 0, source_after_host, input_bytes) != 0);
        TEST_ASSERT(memcmp(sc_host, source_after_host,
                           (size_t)input_bytes) == 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        ref_comp, 0, ref_comp_host, comp_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        fused_comp, 0, fused_comp_host, comp_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        ref_state_kv, 0, ref_state_kv_host, state_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        fused_state_kv, 0, fused_state_kv_host,
                        state_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        ref_state_score, 0, ref_state_score_host,
                        state_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        fused_state_score, 0, fused_state_score_host,
                        state_bytes) != 0);

        comp_stats = test_compare_float_bits(
            ref_comp_host, fused_comp_host, (size_t)comp_count);
        state_kv_stats = test_compare_float_bits(
            ref_state_kv_host, fused_state_kv_host, (size_t)state_count);
        state_score_stats = test_compare_float_bits(
            ref_state_score_host, fused_state_score_host,
            (size_t)state_count);
    }

    test_restore_env(ape_disable_env, saved_ape_disable);
    test_restore_env(pack_disable_env, saved_pack_disable);
    fprintf(stderr,
            "ds4-test: compressor ratio4 replay pack exact head=%u "
            "tokens=%u comp=%zu/%llu state_kv=%zu/%llu "
            "state_score=%zu/%llu max_ulp=%u/%u/%u\n",
            head_dim, n_tokens,
            comp_stats.mismatch_count, (unsigned long long)comp_count,
            state_kv_stats.mismatch_count, (unsigned long long)state_count,
            state_score_stats.mismatch_count, (unsigned long long)state_count,
            comp_stats.max_ulp, state_kv_stats.max_ulp,
            state_score_stats.max_ulp);
    TEST_ASSERT(comp_stats.mismatch_count == 0);
    TEST_ASSERT(state_kv_stats.mismatch_count == 0);
    TEST_ASSERT(state_score_stats.mismatch_count == 0);

    free(model_raw);
    free(fused_state_score_host);
    free(ref_state_score_host);
    free(fused_state_kv_host);
    free(ref_state_kv_host);
    free(fused_comp_host);
    free(ref_comp_host);
    free(source_after_host);
    free(state_score_host);
    free(state_kv_host);
    free(sc_host);
    free(kv_host);
    ds4_gpu_tensor_free(fused_state_score);
    ds4_gpu_tensor_free(fused_state_kv);
    ds4_gpu_tensor_free(ref_state_score);
    ds4_gpu_tensor_free(ref_state_kv);
    ds4_gpu_tensor_free(fused_comp);
    ds4_gpu_tensor_free(ref_comp);
    ds4_gpu_tensor_free(sc);
    ds4_gpu_tensor_free(kv);
}

static void test_metal_compressor_ratio4_replay_pack_exact(void) {
    test_metal_compressor_ratio4_replay_pack_exact_case(128, 4, 47);
    test_metal_compressor_ratio4_replay_pack_exact_case(512, 8, 53);
}

static void test_metal_compressor_ratio4_direct_pool_exact_case(
        uint32_t head_dim,
        uint32_t pos0,
        uint32_t n_tokens,
        uint32_t ape_type,
        bool replay,
        uint32_t seed) {
    const uint32_t ratio = 4u;
    const uint32_t width = 2u * head_dim;
    const uint32_t state_rows = 8u;
    const uint32_t n_comp = n_tokens / ratio;
    const uint64_t input_count = (uint64_t)n_tokens * width;
    const uint64_t state_count = (uint64_t)state_rows * width;
    const uint64_t comp_count = (uint64_t)n_comp * head_dim;
    const uint64_t input_bytes = input_count * sizeof(float);
    const uint64_t state_bytes = state_count * sizeof(float);
    const uint64_t comp_bytes = comp_count * sizeof(float);
    const uint64_t ape_elem_bytes = ape_type == 1u ? 2u : 4u;
    const uint64_t ape_bytes = (uint64_t)ratio * width * ape_elem_bytes;
    const uint64_t page = (uint64_t)getpagesize();
    const uint64_t norm_offset = test_round_up_u64(ape_bytes, page);
    const uint64_t model_bytes = test_round_up_u64(
        norm_offset + (uint64_t)head_dim * sizeof(float), page);

    TEST_ASSERT(n_comp != 0);
    TEST_ASSERT(!replay || ((pos0 & 3u) == 0u && (n_tokens & 3u) == 0u));

    ds4_gpu_tensor *kv = ds4_gpu_tensor_alloc(input_bytes);
    ds4_gpu_tensor *sc = ds4_gpu_tensor_alloc(input_bytes);
    ds4_gpu_tensor *ref_comp = ds4_gpu_tensor_alloc(comp_bytes);
    ds4_gpu_tensor *direct_comp = ds4_gpu_tensor_alloc(comp_bytes);
    ds4_gpu_tensor *ref_state_kv = ds4_gpu_tensor_alloc(state_bytes);
    ds4_gpu_tensor *ref_state_score = ds4_gpu_tensor_alloc(state_bytes);
    ds4_gpu_tensor *direct_state_kv = ds4_gpu_tensor_alloc(state_bytes);
    ds4_gpu_tensor *direct_state_score = ds4_gpu_tensor_alloc(state_bytes);
    float *kv_host = malloc((size_t)input_bytes);
    float *sc_host = malloc((size_t)input_bytes);
    float *source_after_host = malloc((size_t)input_bytes);
    float *state_kv_host = malloc((size_t)state_bytes);
    float *state_score_host = malloc((size_t)state_bytes);
    float *ref_comp_host = malloc((size_t)comp_bytes);
    float *direct_comp_host = malloc((size_t)comp_bytes);
    float *ref_state_kv_host = malloc((size_t)state_bytes);
    float *direct_state_kv_host = malloc((size_t)state_bytes);
    float *ref_state_score_host = malloc((size_t)state_bytes);
    float *direct_state_score_host = malloc((size_t)state_bytes);
    void *model_raw = NULL;
    TEST_ASSERT(posix_memalign(
                    &model_raw, (size_t)page, (size_t)model_bytes) == 0);

    TEST_ASSERT(kv != NULL);
    TEST_ASSERT(sc != NULL);
    TEST_ASSERT(ref_comp != NULL);
    TEST_ASSERT(direct_comp != NULL);
    TEST_ASSERT(ref_state_kv != NULL);
    TEST_ASSERT(ref_state_score != NULL);
    TEST_ASSERT(direct_state_kv != NULL);
    TEST_ASSERT(direct_state_score != NULL);
    TEST_ASSERT(kv_host != NULL);
    TEST_ASSERT(sc_host != NULL);
    TEST_ASSERT(source_after_host != NULL);
    TEST_ASSERT(state_kv_host != NULL);
    TEST_ASSERT(state_score_host != NULL);
    TEST_ASSERT(ref_comp_host != NULL);
    TEST_ASSERT(direct_comp_host != NULL);
    TEST_ASSERT(ref_state_kv_host != NULL);
    TEST_ASSERT(direct_state_kv_host != NULL);
    TEST_ASSERT(ref_state_score_host != NULL);
    TEST_ASSERT(direct_state_score_host != NULL);
    TEST_ASSERT(model_raw != NULL);

    const char *ape_disable_env = "DS4_METAL_DISABLE_COMPRESSOR_APE_ADD";
    const char *pack_disable_env =
        "DS4_METAL_DISABLE_COMPRESSOR_RATIO4_PACK_FUSION";
    const char *direct_disable_env =
        "DS4_METAL_DISABLE_COMPRESSOR_RATIO4_DIRECT_POOL";
    char *saved_ape_disable = test_save_env(ape_disable_env);
    char *saved_pack_disable = test_save_env(pack_disable_env);
    char *saved_direct_disable = test_save_env(direct_disable_env);
    test_float_compare_stats comp_stats = {0};
    test_float_compare_stats state_kv_stats = {0};
    test_float_compare_stats state_score_stats = {0};

    const bool allocated = kv && sc && ref_comp && direct_comp &&
        ref_state_kv && ref_state_score && direct_state_kv &&
        direct_state_score && kv_host && sc_host && source_after_host &&
        state_kv_host && state_score_host && ref_comp_host &&
        direct_comp_host && ref_state_kv_host && direct_state_kv_host &&
        ref_state_score_host && direct_state_score_host && model_raw;
    if (allocated) {
        memset(model_raw, 0, (size_t)model_bytes);
        if (ape_type == 1u) {
            uint16_t *ape = model_raw;
            for (uint64_t i = 0; i < (uint64_t)ratio * width; i++) {
                const int value =
                    (int)((i * 17u + (i ^ (i >> 4u)) * 5u + seed * 13u) %
                          127u) - 63;
                ape[i] = test_float_to_f16((float)value / 80.0f);
            }
        } else {
            float *ape = model_raw;
            for (uint64_t i = 0; i < (uint64_t)ratio * width; i++) {
                const int value =
                    (int)((i * 17u + (i ^ (i >> 4u)) * 5u + seed * 13u) %
                          127u) - 63;
                ape[i] = (float)value / 80.0f;
            }
        }
        float *norm = (float *)((uint8_t *)model_raw + norm_offset);
        for (uint32_t i = 0; i < head_dim; i++) {
            norm[i] = 0.75f + (float)((i * 7u + seed) % 19u) / 64.0f;
        }

        for (uint64_t i = 0; i < input_count; i++) {
            const uint32_t token = (uint32_t)(i / width);
            const uint32_t col = (uint32_t)(i - (uint64_t)token * width);
            const int kv_value =
                (int)(((uint64_t)token * 37u + (uint64_t)col * 29u +
                       (col >= head_dim ? 71u : 3u) + seed * 23u) %
                      193u) - 96;
            const int sc_value =
                (int)(((uint64_t)token * 41u + (uint64_t)col * 31u +
                       (col >= head_dim ? 17u : 83u) + seed * 11u) %
                      181u) - 90;
            kv_host[i] = (float)kv_value / 104.0f;
            sc_host[i] = (float)sc_value / 32.0f;
        }
        const uint32_t negative_zero = 0x80000000u;
        memcpy(kv_host, &negative_zero, sizeof(negative_zero));
        memcpy(sc_host + width + head_dim, &negative_zero,
               sizeof(negative_zero));

        for (uint32_t row = 0; row < state_rows; row++) {
            for (uint32_t col = 0; col < width; col++) {
                const uint64_t i = (uint64_t)row * width + col;
                if (replay && row < ratio && col < head_dim) {
                    const int kv_value =
                        (int)(((uint64_t)row * 43u + (uint64_t)col * 19u +
                               seed * 29u) % 167u) - 83;
                    const int sc_value =
                        (int)(((uint64_t)row * 47u + (uint64_t)col * 23u +
                               seed * 7u) % 157u) - 78;
                    state_kv_host[i] = (float)kv_value / 80.0f;
                    state_score_host[i] = (float)sc_value / 28.0f;
                } else {
                    const uint32_t kv_bits =
                        0x7fc00001u + (uint32_t)(i & 0x3ffu);
                    const uint32_t score_bits =
                        0x7fc00401u + (uint32_t)(i & 0x3ffu);
                    memcpy(state_kv_host + i, &kv_bits, sizeof(kv_bits));
                    memcpy(state_score_host + i, &score_bits,
                           sizeof(score_bits));
                }
            }
        }

        for (uint64_t i = 0; i < comp_count; i++) {
            const uint32_t bits = 0x7fc00801u + (uint32_t)(i & 0x3ffu);
            memcpy(ref_comp_host + i, &bits, sizeof(bits));
            memcpy(direct_comp_host + i, &bits, sizeof(bits));
        }

        TEST_ASSERT(ds4_gpu_tensor_write(kv, 0, kv_host, input_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(sc, 0, sc_host, input_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        ref_comp, 0, ref_comp_host, comp_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        direct_comp, 0, direct_comp_host, comp_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        ref_state_kv, 0, state_kv_host, state_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        direct_state_kv, 0, state_kv_host, state_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        ref_state_score, 0, state_score_host,
                        state_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        direct_state_score, 0, state_score_host,
                        state_bytes) != 0);
        TEST_ASSERT(ds4_gpu_set_model_map(model_raw, model_bytes) != 0);
        ds4_gpu_set_quality(false);

        TEST_ASSERT(unsetenv(ape_disable_env) == 0);
        TEST_ASSERT(setenv(direct_disable_env, "1", 1) == 0);
        TEST_ASSERT(unsetenv(pack_disable_env) == 0);
        int ref_ok;
        if (replay) {
            ref_ok = ds4_gpu_compressor_prefill_ratio4_replay_tensor(
                ref_comp, ref_state_kv, ref_state_score, kv, sc,
                model_raw, model_bytes, 0, ape_type, norm_offset, 0,
                head_dim, pos0, n_tokens, 0, 0, false,
                10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f, 1.0e-6f);
        } else {
            ref_ok = ds4_gpu_compressor_prefill_tensor(
                ref_comp, ref_state_kv, ref_state_score, kv, sc,
                model_raw, model_bytes, 0, ape_type, norm_offset, 0,
                head_dim, ratio, pos0, n_tokens, 0, 0, false,
                10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f, 1.0e-6f);
        }
        TEST_ASSERT(ref_ok != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        kv, 0, source_after_host, input_bytes) != 0);
        TEST_ASSERT(memcmp(kv_host, source_after_host,
                           (size_t)input_bytes) == 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        sc, 0, source_after_host, input_bytes) != 0);
        TEST_ASSERT(memcmp(sc_host, source_after_host,
                           (size_t)input_bytes) == 0);

        TEST_ASSERT(unsetenv(direct_disable_env) == 0);
        TEST_ASSERT(setenv(pack_disable_env, "1", 1) == 0);
        int direct_ok;
        if (replay) {
            direct_ok = ds4_gpu_compressor_prefill_ratio4_replay_tensor(
                direct_comp, direct_state_kv, direct_state_score, kv, sc,
                model_raw, model_bytes, 0, ape_type, norm_offset, 0,
                head_dim, pos0, n_tokens, 0, 0, false,
                10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f, 1.0e-6f);
        } else {
            direct_ok = ds4_gpu_compressor_prefill_tensor(
                direct_comp, direct_state_kv, direct_state_score, kv, sc,
                model_raw, model_bytes, 0, ape_type, norm_offset, 0,
                head_dim, ratio, pos0, n_tokens, 0, 0, false,
                10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f, 1.0e-6f);
        }
        TEST_ASSERT(direct_ok != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        kv, 0, source_after_host, input_bytes) != 0);
        TEST_ASSERT(memcmp(kv_host, source_after_host,
                           (size_t)input_bytes) == 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        sc, 0, source_after_host, input_bytes) != 0);
        TEST_ASSERT(memcmp(sc_host, source_after_host,
                           (size_t)input_bytes) == 0);

        TEST_ASSERT(ds4_gpu_tensor_read(
                        ref_comp, 0, ref_comp_host, comp_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        direct_comp, 0, direct_comp_host, comp_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        ref_state_kv, 0, ref_state_kv_host,
                        state_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        direct_state_kv, 0, direct_state_kv_host,
                        state_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        ref_state_score, 0, ref_state_score_host,
                        state_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        direct_state_score, 0, direct_state_score_host,
                        state_bytes) != 0);

        comp_stats = test_compare_float_bits(
            ref_comp_host, direct_comp_host, (size_t)comp_count);
        state_kv_stats = test_compare_float_bits(
            ref_state_kv_host, direct_state_kv_host, (size_t)state_count);
        state_score_stats = test_compare_float_bits(
            ref_state_score_host, direct_state_score_host,
            (size_t)state_count);
    }

    test_restore_env(ape_disable_env, saved_ape_disable);
    test_restore_env(pack_disable_env, saved_pack_disable);
    test_restore_env(direct_disable_env, saved_direct_disable);
    fprintf(stderr,
            "ds4-test: compressor ratio4 direct pool exact mode=%s "
            "head=%u pos=%u tokens=%u comp_rows=%u ape=%s "
            "comp=%zu/%llu state_kv=%zu/%llu state_score=%zu/%llu "
            "max_ulp=%u/%u/%u\n",
            replay ? "replay" : "prefill",
            head_dim, pos0, n_tokens, n_comp,
            ape_type == 1u ? "f16" : "f32",
            comp_stats.mismatch_count, (unsigned long long)comp_count,
            state_kv_stats.mismatch_count, (unsigned long long)state_count,
            state_score_stats.mismatch_count,
            (unsigned long long)state_count,
            comp_stats.max_ulp, state_kv_stats.max_ulp,
            state_score_stats.max_ulp);
    TEST_ASSERT(comp_stats.mismatch_count == 0);
    TEST_ASSERT(state_kv_stats.mismatch_count == 0);
    TEST_ASSERT(state_score_stats.mismatch_count == 0);

    free(model_raw);
    free(direct_state_score_host);
    free(ref_state_score_host);
    free(direct_state_kv_host);
    free(ref_state_kv_host);
    free(direct_comp_host);
    free(ref_comp_host);
    free(state_score_host);
    free(state_kv_host);
    free(source_after_host);
    free(sc_host);
    free(kv_host);
    ds4_gpu_tensor_free(direct_state_score);
    ds4_gpu_tensor_free(direct_state_kv);
    ds4_gpu_tensor_free(ref_state_score);
    ds4_gpu_tensor_free(ref_state_kv);
    ds4_gpu_tensor_free(direct_comp);
    ds4_gpu_tensor_free(ref_comp);
    ds4_gpu_tensor_free(sc);
    ds4_gpu_tensor_free(kv);
}

static void test_metal_compressor_ratio4_direct_pool_exact(void) {
    /* Prefill direct-pool coverage, including its legacy n_comp == 1 case. */
    test_metal_compressor_ratio4_direct_pool_exact_case(
        512, 0, 4, 1, false, 59);
    test_metal_compressor_ratio4_direct_pool_exact_case(
        512, 1, 16, 0, false, 61);
    test_metal_compressor_ratio4_direct_pool_exact_case(
        128, 3, 14, 1, false, 67);
    test_metal_compressor_ratio4_direct_pool_exact_case(
        512, 2048, 8, 1, true, 71);
    test_metal_compressor_ratio4_direct_pool_exact_case(
        128, 12, 12, 0, true, 73);
    test_metal_compressor_ratio4_direct_pool_exact_case(
        128, 8, 4, 1, true, 79);
}

static void test_metal_compressor_ratio4_exact_pool_decode_case(
        uint32_t head_dim,
        uint32_t ape_type,
        uint32_t seed) {
    const uint32_t ratio = 4u;
    const uint32_t width = 2u * head_dim;
    const uint32_t state_rows = 8u;
    const uint32_t pos = 3u;
    const uint64_t cur_count = width;
    const uint64_t state_count = (uint64_t)state_rows * width;
    const uint64_t cur_bytes = cur_count * sizeof(float);
    const uint64_t state_bytes = state_count * sizeof(float);
    const uint64_t comp_bytes = (uint64_t)head_dim * sizeof(float);
    const uint64_t ape_elem_bytes = ape_type == 1u ? 2u : 4u;
    const uint64_t ape_bytes = (uint64_t)ratio * width * ape_elem_bytes;
    const uint64_t page = (uint64_t)getpagesize();
    const uint64_t norm_offset = test_round_up_u64(ape_bytes, page);
    const uint64_t model_bytes = test_round_up_u64(
        norm_offset + (uint64_t)head_dim * sizeof(float), page);

    ds4_gpu_tensor *kv_cur = ds4_gpu_tensor_alloc(cur_bytes);
    ds4_gpu_tensor *sc_cur = ds4_gpu_tensor_alloc(cur_bytes);
    ds4_gpu_tensor *ref_state_kv = ds4_gpu_tensor_alloc(state_bytes);
    ds4_gpu_tensor *ref_state_score = ds4_gpu_tensor_alloc(state_bytes);
    ds4_gpu_tensor *exact_state_kv = ds4_gpu_tensor_alloc(state_bytes);
    ds4_gpu_tensor *exact_state_score = ds4_gpu_tensor_alloc(state_bytes);
    ds4_gpu_tensor *ref_comp = ds4_gpu_tensor_alloc(comp_bytes);
    ds4_gpu_tensor *exact_comp = ds4_gpu_tensor_alloc(comp_bytes);
    float *kv_cur_host = malloc((size_t)cur_bytes);
    float *sc_cur_host = malloc((size_t)cur_bytes);
    float *state_kv_host = malloc((size_t)state_bytes);
    float *state_score_host = malloc((size_t)state_bytes);
    float *ref_state_kv_host = malloc((size_t)state_bytes);
    float *ref_state_score_host = malloc((size_t)state_bytes);
    float *exact_state_kv_host = malloc((size_t)state_bytes);
    float *exact_state_score_host = malloc((size_t)state_bytes);
    float *ref_comp_host = malloc((size_t)comp_bytes);
    float *exact_comp_host = malloc((size_t)comp_bytes);
    void *model_raw = NULL;
    const int model_alloc_ok = posix_memalign(
        &model_raw, (size_t)page, (size_t)model_bytes) == 0;

    TEST_ASSERT(kv_cur != NULL);
    TEST_ASSERT(sc_cur != NULL);
    TEST_ASSERT(ref_state_kv != NULL);
    TEST_ASSERT(ref_state_score != NULL);
    TEST_ASSERT(exact_state_kv != NULL);
    TEST_ASSERT(exact_state_score != NULL);
    TEST_ASSERT(ref_comp != NULL);
    TEST_ASSERT(exact_comp != NULL);
    TEST_ASSERT(kv_cur_host != NULL);
    TEST_ASSERT(sc_cur_host != NULL);
    TEST_ASSERT(state_kv_host != NULL);
    TEST_ASSERT(state_score_host != NULL);
    TEST_ASSERT(ref_state_kv_host != NULL);
    TEST_ASSERT(ref_state_score_host != NULL);
    TEST_ASSERT(exact_state_kv_host != NULL);
    TEST_ASSERT(exact_state_score_host != NULL);
    TEST_ASSERT(ref_comp_host != NULL);
    TEST_ASSERT(exact_comp_host != NULL);
    TEST_ASSERT(model_alloc_ok);

    const char *enable_env =
        "DS4_METAL_ENABLE_COMPRESSOR_EXACT_POOL_RATIO4";
    const char *disable_env =
        "DS4_METAL_DISABLE_COMPRESSOR_EXACT_POOL_RATIO4";
    const char *pre_m5_disable_env =
        "DS4_METAL_DISABLE_PRE_M5_COMPRESSOR_EXACT_POOL_RATIO4";
    const char *m3_disable_env =
        "DS4_METAL_DISABLE_M3_COMPRESSOR_EXACT_POOL_RATIO4";
    const char *m5_disable_env =
        "DS4_METAL_DISABLE_M5_COMPRESSOR_EXACT_POOL_RATIO4";
    const char *all_pre_m5_disable_env =
        "DS4_METAL_DISABLE_PRE_M5_DECODE_PORTS";
    const char *require_env =
        "DS4_METAL_REQUIRE_COMPRESSOR_EXACT_POOL_RATIO4";
    const char *poison_env =
        "DS4_METAL_TEST_POISON_COMPRESSOR_EXACT_REDUCTION_SCRATCH";
    char *saved_enable = test_save_env(enable_env);
    char *saved_disable = test_save_env(disable_env);
    char *saved_pre_m5_disable = test_save_env(pre_m5_disable_env);
    char *saved_m3_disable = test_save_env(m3_disable_env);
    char *saved_m5_disable = test_save_env(m5_disable_env);
    char *saved_all_pre_m5_disable = test_save_env(all_pre_m5_disable_env);
    char *saved_require = test_save_env(require_env);
    char *saved_poison = test_save_env(poison_env);
    test_float_compare_stats comp_stats = {0};
    test_float_compare_stats state_kv_stats = {0};
    test_float_compare_stats state_score_stats = {0};

    const bool allocated = kv_cur && sc_cur && ref_state_kv &&
        ref_state_score && exact_state_kv && exact_state_score && ref_comp &&
        exact_comp && kv_cur_host && sc_cur_host && state_kv_host &&
        state_score_host && ref_state_kv_host && ref_state_score_host &&
        exact_state_kv_host && exact_state_score_host && ref_comp_host &&
        exact_comp_host && model_alloc_ok;
    if (allocated) {
        memset(model_raw, 0, (size_t)model_bytes);
        if (ape_type == 1u) {
            uint16_t *ape = model_raw;
            for (uint64_t i = 0; i < (uint64_t)ratio * width; i++) {
                const int value =
                    (int)((i * 17u + (i >> 3u) * 13u + seed * 11u) %
                          101u) - 50;
                ape[i] = test_float_to_f16((float)value / 96.0f);
            }
        } else {
            float *ape = model_raw;
            for (uint64_t i = 0; i < (uint64_t)ratio * width; i++) {
                const int value =
                    (int)((i * 17u + (i >> 3u) * 13u + seed * 11u) %
                          101u) - 50;
                ape[i] = (float)value / 96.0f;
            }
        }
        float *norm =
            (float *)((uint8_t *)model_raw + norm_offset);
        for (uint32_t i = 0; i < head_dim; i++) {
            norm[i] = 0.75f +
                (float)((i * 7u + seed * 3u) % 29u) / 128.0f;
        }

        for (uint32_t i = 0; i < width; i++) {
            const int kv_value =
                (int)(((uint64_t)i * 29u + seed * 37u) % 211u) - 105;
            const int score_value =
                (int)(((uint64_t)i * 31u + seed * 19u) % 181u) - 90;
            kv_cur_host[i] = (float)kv_value / 112.0f;
            sc_cur_host[i] = (float)score_value / 48.0f;
        }
        for (uint32_t row = 0; row < state_rows; row++) {
            for (uint32_t col = 0; col < width; col++) {
                const uint64_t i = (uint64_t)row * width + col;
                const int kv_value =
                    (int)(((uint64_t)row * 43u + (uint64_t)col * 23u +
                           seed * 41u) % 257u) - 128;
                const int score_value =
                    (int)(((uint64_t)row * 47u + (uint64_t)col * 17u +
                           seed * 31u) % 193u) - 96;
                state_kv_host[i] = (float)kv_value / 128.0f;
                state_score_host[i] = (float)score_value / 56.0f;
            }
        }

        TEST_ASSERT(ds4_gpu_tensor_write(
                        kv_cur, 0, kv_cur_host, cur_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        sc_cur, 0, sc_cur_host, cur_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        ref_state_kv, 0, state_kv_host, state_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        ref_state_score, 0, state_score_host,
                        state_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        exact_state_kv, 0, state_kv_host, state_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        exact_state_score, 0, state_score_host,
                        state_bytes) != 0);
        TEST_ASSERT(ds4_gpu_set_model_map(model_raw, model_bytes) != 0);
        ds4_gpu_set_quality(false);

        /* Force-enable and force-disable together: the kill switch must win. */
        TEST_ASSERT(setenv(enable_env, "1", 1) == 0);
        TEST_ASSERT(setenv(disable_env, "1", 1) == 0);
        TEST_ASSERT(unsetenv(require_env) == 0);
        TEST_ASSERT(unsetenv(poison_env) == 0);
        const int ref_ok = ds4_gpu_compressor_update_tensor(
            kv_cur, sc_cur, ref_state_kv, ref_state_score, ref_comp,
            model_raw, model_bytes, 0, ape_type, norm_offset, 0,
            head_dim, ratio, pos, 0, 0, 0,
            10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f, 1.0e-6f,
            false, false, false);
        TEST_ASSERT(ref_ok != 0);

        TEST_ASSERT(unsetenv(disable_env) == 0);
        TEST_ASSERT(unsetenv(pre_m5_disable_env) == 0);
        TEST_ASSERT(unsetenv(m3_disable_env) == 0);
        TEST_ASSERT(unsetenv(m5_disable_env) == 0);
        TEST_ASSERT(unsetenv(all_pre_m5_disable_env) == 0);
        TEST_ASSERT(setenv(require_env, "1", 1) == 0);
        TEST_ASSERT(setenv(poison_env, "1", 1) == 0);
        const int exact_ok = ds4_gpu_compressor_update_tensor(
            kv_cur, sc_cur, exact_state_kv, exact_state_score, exact_comp,
            model_raw, model_bytes, 0, ape_type, norm_offset, 0,
            head_dim, ratio, pos, 0, 0, 0,
            10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f, 1.0e-6f,
            false, false, false);
        TEST_ASSERT(exact_ok != 0);

        TEST_ASSERT(ds4_gpu_tensor_read(
                        ref_comp, 0, ref_comp_host, comp_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        exact_comp, 0, exact_comp_host, comp_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        ref_state_kv, 0, ref_state_kv_host,
                        state_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        exact_state_kv, 0, exact_state_kv_host,
                        state_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        ref_state_score, 0, ref_state_score_host,
                        state_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        exact_state_score, 0, exact_state_score_host,
                        state_bytes) != 0);

        comp_stats = test_compare_float_bits(
            ref_comp_host, exact_comp_host, head_dim);
        state_kv_stats = test_compare_float_bits(
            ref_state_kv_host, exact_state_kv_host, (size_t)state_count);
        state_score_stats = test_compare_float_bits(
            ref_state_score_host, exact_state_score_host,
            (size_t)state_count);
    }

    test_restore_env(enable_env, saved_enable);
    test_restore_env(disable_env, saved_disable);
    test_restore_env(pre_m5_disable_env, saved_pre_m5_disable);
    test_restore_env(m3_disable_env, saved_m3_disable);
    test_restore_env(m5_disable_env, saved_m5_disable);
    test_restore_env(all_pre_m5_disable_env, saved_all_pre_m5_disable);
    test_restore_env(require_env, saved_require);
    test_restore_env(poison_env, saved_poison);
    fprintf(stderr,
            "ds4-test: compressor ratio4 exact decode pool head=%u ape=%s "
            "comp=%zu/%u state_kv=%zu/%llu state_score=%zu/%llu "
            "max_ulp=%u/%u/%u\n",
            head_dim, ape_type == 1u ? "f16" : "f32",
            comp_stats.mismatch_count, head_dim,
            state_kv_stats.mismatch_count,
            (unsigned long long)state_count,
            state_score_stats.mismatch_count,
            (unsigned long long)state_count,
            comp_stats.max_ulp, state_kv_stats.max_ulp,
            state_score_stats.max_ulp);
    TEST_ASSERT(comp_stats.mismatch_count == 0);
    TEST_ASSERT(state_kv_stats.mismatch_count == 0);
    TEST_ASSERT(state_score_stats.mismatch_count == 0);

    free(model_raw);
    free(exact_comp_host);
    free(ref_comp_host);
    free(exact_state_score_host);
    free(exact_state_kv_host);
    free(ref_state_score_host);
    free(ref_state_kv_host);
    free(state_score_host);
    free(state_kv_host);
    free(sc_cur_host);
    free(kv_cur_host);
    ds4_gpu_tensor_free(exact_comp);
    ds4_gpu_tensor_free(ref_comp);
    ds4_gpu_tensor_free(exact_state_score);
    ds4_gpu_tensor_free(exact_state_kv);
    ds4_gpu_tensor_free(ref_state_score);
    ds4_gpu_tensor_free(ref_state_kv);
    ds4_gpu_tensor_free(sc_cur);
    ds4_gpu_tensor_free(kv_cur);
}

static void test_metal_compressor_ratio4_exact_pool_decode(void) {
    test_metal_compressor_ratio4_exact_pool_decode_case(128u, 1u, 83u);
    test_metal_compressor_ratio4_exact_pool_decode_case(512u, 0u, 89u);
}

static void test_metal_inplace_rope_pair_exact(void) {
    typedef struct {
        uint32_t head_dim;
        uint32_t n_rot;
        uint32_t n_head;
        uint32_t n_tok;
        uint32_t pos0;
        bool inverse;
        float ext_factor;
    } rope_case;
    static const rope_case cases[] = {
        { 512, 64,  64,  1, UINT32_MAX, false, 1.0f },
        { 512, 64,   1,  1,  2047, false, 1.0f },
        { 512, 64,  64,  1, 65533,  true, 1.0f },
        { 128, 64,  64,  1,    37, false, 0.0f },
        { 512, 64,   4, 32,     0, false, 0.0f },
        { 512, 64,   7, 33,  2047,  true, 1.0f },
        { 128, 64,  64, 34, 65533, false, 1.0f },
        { 128, 64,  64, 35,    37,  true, 0.0f },
        { 128, 64,   4, 35, UINT32_MAX - 16u, true, 1.0f },
    };
    const char *disable_env = "DS4_METAL_DISABLE_INPLACE_ROPE_PAIR";
    const char *shared_disable_env =
        "DS4_METAL_DISABLE_SHARED_ROPE_COEFF";
    const char *affine_disable_env =
        "DS4_METAL_DISABLE_AFFINE_ROPE_PAIR";
    char *saved_disable = test_save_env(disable_env);
    char *saved_shared_disable = test_save_env(shared_disable_env);
    char *saved_affine_disable = test_save_env(affine_disable_env);
    size_t total_pair_mismatch = 0;
    size_t total_shared_mismatch = 0;
    size_t total_affine_mismatch = 0;
    size_t total_pair_prefix_mismatch = 0;
    size_t total_pair_tail_mismatch = 0;
    size_t total_shared_prefix_mismatch = 0;
    size_t total_shared_tail_mismatch = 0;
    size_t total_affine_prefix_mismatch = 0;
    size_t total_affine_tail_mismatch = 0;
    size_t total_elements = 0;

    for (size_t ci = 0; ci < sizeof(cases) / sizeof(cases[0]); ci++) {
        const rope_case *c = &cases[ci];
        const size_t elements =
            (size_t)c->n_tok * c->n_head * c->head_dim;
        const uint64_t bytes = (uint64_t)elements * sizeof(float);
        const uint32_t n_nope = c->head_dim - c->n_rot;
        const float freq_base = c->ext_factor != 0.0f ? 160000.0f : 10000.0f;
        const float freq_scale = c->ext_factor != 0.0f ? 1.0f / 16.0f : 1.0f;
        const uint32_t n_ctx_orig = c->ext_factor != 0.0f ? 65536u : 0u;
        float attn_factor = 1.0f;
        if (c->ext_factor != 0.0f) {
            attn_factor /= 1.0f + 0.1f * logf(1.0f / freq_scale);
        }

        ds4_gpu_tensor *reference = ds4_gpu_tensor_alloc(bytes);
        ds4_gpu_tensor *pair_candidate = ds4_gpu_tensor_alloc(bytes);
        ds4_gpu_tensor *shared_candidate = ds4_gpu_tensor_alloc(bytes);
        ds4_gpu_tensor *affine_candidate = ds4_gpu_tensor_alloc(bytes);
        float *input = malloc((size_t)bytes);
        float *reference_host = malloc((size_t)bytes);
        float *pair_host = malloc((size_t)bytes);
        float *shared_host = malloc((size_t)bytes);
        float *affine_host = malloc((size_t)bytes);
        TEST_ASSERT(reference != NULL);
        TEST_ASSERT(pair_candidate != NULL);
        TEST_ASSERT(shared_candidate != NULL);
        TEST_ASSERT(affine_candidate != NULL);
        TEST_ASSERT(input != NULL);
        TEST_ASSERT(reference_host != NULL);
        TEST_ASSERT(pair_host != NULL);
        TEST_ASSERT(shared_host != NULL);
        TEST_ASSERT(affine_host != NULL);

        const bool allocated = reference && pair_candidate &&
            shared_candidate && affine_candidate && input && reference_host &&
            pair_host && shared_host && affine_host;
        if (allocated) {
            for (size_t i = 0; i < elements; i++) {
                const uint32_t key =
                    (uint32_t)(i * 37u + (i ^ (i >> 5u)) * 11u + ci * 101u);
                const int value = (int)(key % 4093u) - 2046;
                input[i] = (float)value / 1024.0f;
                if ((i + ci * 17u) % 257u == 0u) {
                    const uint32_t negative_zero = 0x80000000u;
                    memcpy(&input[i], &negative_zero, sizeof(negative_zero));
                }
            }

            TEST_ASSERT(ds4_gpu_tensor_write(reference, 0, input, bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_write(
                pair_candidate, 0, input, bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_write(
                shared_candidate, 0, input, bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_write(
                affine_candidate, 0, input, bytes) != 0);
            ds4_gpu_set_quality(false);

            TEST_ASSERT(setenv(affine_disable_env, "1", 1) == 0);
            TEST_ASSERT(setenv(disable_env, "1", 1) == 0);
            TEST_ASSERT(setenv(shared_disable_env, "1", 1) == 0);
            TEST_ASSERT(ds4_gpu_rope_tail_tensor(
                reference,
                c->n_tok,
                c->n_head,
                c->head_dim,
                c->n_rot,
                c->pos0,
                n_ctx_orig,
                c->inverse,
                freq_base,
                freq_scale,
                c->ext_factor,
                attn_factor,
                32.0f,
                1.0f) != 0);

            TEST_ASSERT(unsetenv(disable_env) == 0);
            TEST_ASSERT(ds4_gpu_rope_tail_tensor(
                pair_candidate,
                c->n_tok,
                c->n_head,
                c->head_dim,
                c->n_rot,
                c->pos0,
                n_ctx_orig,
                c->inverse,
                freq_base,
                freq_scale,
                c->ext_factor,
                attn_factor,
                32.0f,
                1.0f) != 0);

            TEST_ASSERT(unsetenv(shared_disable_env) == 0);
            TEST_ASSERT(ds4_gpu_rope_tail_tensor(
                shared_candidate,
                c->n_tok,
                c->n_head,
                c->head_dim,
                c->n_rot,
                c->pos0,
                n_ctx_orig,
                c->inverse,
                freq_base,
                freq_scale,
                c->ext_factor,
                attn_factor,
                32.0f,
                1.0f) != 0);

            TEST_ASSERT(setenv(shared_disable_env, "1", 1) == 0);
            TEST_ASSERT(unsetenv(affine_disable_env) == 0);
            TEST_ASSERT(ds4_gpu_rope_tail_tensor(
                affine_candidate,
                c->n_tok,
                c->n_head,
                c->head_dim,
                c->n_rot,
                c->pos0,
                n_ctx_orig,
                c->inverse,
                freq_base,
                freq_scale,
                c->ext_factor,
                attn_factor,
                32.0f,
                1.0f) != 0);

            TEST_ASSERT(ds4_gpu_tensor_read(
                reference, 0, reference_host, bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_read(
                pair_candidate, 0, pair_host, bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_read(
                shared_candidate, 0, shared_host, bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_read(
                affine_candidate, 0, affine_host, bytes) != 0);

            const test_float_compare_stats pair_stats =
                test_compare_float_bits(
                    reference_host, pair_host, elements);
            const test_float_compare_stats shared_stats =
                test_compare_float_bits(
                    pair_host, shared_host, elements);
            const test_float_compare_stats affine_stats =
                test_compare_float_bits(
                    pair_host, affine_host, elements);
            size_t pair_prefix_mismatch = 0;
            size_t pair_tail_mismatch = 0;
            size_t shared_prefix_mismatch = 0;
            size_t shared_tail_mismatch = 0;
            size_t affine_prefix_mismatch = 0;
            size_t affine_tail_mismatch = 0;
            for (uint32_t t = 0; t < c->n_tok; t++) {
                for (uint32_t h = 0; h < c->n_head; h++) {
                    const size_t row =
                        ((size_t)t * c->n_head + h) * c->head_dim;
                    for (uint32_t d = 0; d < n_nope; d++) {
                        if (memcmp(&input[row + d],
                                   &pair_host[row + d],
                                   sizeof(float)) != 0) {
                            pair_prefix_mismatch++;
                        }
                        if (memcmp(&input[row + d],
                                   &shared_host[row + d],
                                   sizeof(float)) != 0) {
                            shared_prefix_mismatch++;
                        }
                        if (memcmp(&input[row + d],
                                   &affine_host[row + d],
                                   sizeof(float)) != 0) {
                            affine_prefix_mismatch++;
                        }
                    }
                    for (uint32_t d = n_nope; d < c->head_dim; d++) {
                        if (memcmp(&reference_host[row + d],
                                   &pair_host[row + d],
                                   sizeof(float)) != 0) {
                            pair_tail_mismatch++;
                        }
                        if (memcmp(&pair_host[row + d],
                                   &shared_host[row + d],
                                   sizeof(float)) != 0) {
                            shared_tail_mismatch++;
                        }
                        if (memcmp(&pair_host[row + d],
                                   &affine_host[row + d],
                                   sizeof(float)) != 0) {
                            affine_tail_mismatch++;
                        }
                    }
                }
            }

            fprintf(stderr,
                    "ds4-test: in-place RoPE exactness case=%zu "
                    "shape=%ux%ux%u pos=%u inverse=%d ext=%g "
                    "pair=%zu/%zu shared=%zu/%zu affine=%zu/%zu "
                    "pair_prefix=%zu pair_tail=%zu "
                    "shared_prefix=%zu shared_tail=%zu "
                    "affine_prefix=%zu affine_tail=%zu "
                    "pair_max_ulp=%u shared_max_ulp=%u affine_max_ulp=%u "
                    "pair_max_abs=%g shared_max_abs=%g affine_max_abs=%g\n",
                    ci,
                    c->n_tok,
                    c->n_head,
                    c->head_dim,
                    c->pos0,
                    c->inverse ? 1 : 0,
                    c->ext_factor,
                    pair_stats.mismatch_count,
                    elements,
                    shared_stats.mismatch_count,
                    elements,
                    affine_stats.mismatch_count,
                    elements,
                    pair_prefix_mismatch,
                    pair_tail_mismatch,
                    shared_prefix_mismatch,
                    shared_tail_mismatch,
                    affine_prefix_mismatch,
                    affine_tail_mismatch,
                    pair_stats.max_ulp,
                    shared_stats.max_ulp,
                    affine_stats.max_ulp,
                    pair_stats.max_abs,
                    shared_stats.max_abs,
                    affine_stats.max_abs);
            TEST_ASSERT(pair_stats.mismatch_count == 0);
            TEST_ASSERT(shared_stats.mismatch_count == 0);
            TEST_ASSERT(affine_stats.mismatch_count == 0);
            TEST_ASSERT(pair_prefix_mismatch == 0);
            TEST_ASSERT(pair_tail_mismatch == 0);
            TEST_ASSERT(shared_prefix_mismatch == 0);
            TEST_ASSERT(shared_tail_mismatch == 0);
            TEST_ASSERT(affine_prefix_mismatch == 0);
            TEST_ASSERT(affine_tail_mismatch == 0);
            total_pair_mismatch += pair_stats.mismatch_count;
            total_shared_mismatch += shared_stats.mismatch_count;
            total_affine_mismatch += affine_stats.mismatch_count;
            total_pair_prefix_mismatch += pair_prefix_mismatch;
            total_pair_tail_mismatch += pair_tail_mismatch;
            total_shared_prefix_mismatch += shared_prefix_mismatch;
            total_shared_tail_mismatch += shared_tail_mismatch;
            total_affine_prefix_mismatch += affine_prefix_mismatch;
            total_affine_tail_mismatch += affine_tail_mismatch;
            total_elements += elements;
        }

        free(affine_host);
        free(shared_host);
        free(pair_host);
        free(reference_host);
        free(input);
        ds4_gpu_tensor_free(affine_candidate);
        ds4_gpu_tensor_free(shared_candidate);
        ds4_gpu_tensor_free(pair_candidate);
        ds4_gpu_tensor_free(reference);
    }

    test_restore_env(disable_env, saved_disable);
    test_restore_env(shared_disable_env, saved_shared_disable);
    test_restore_env(affine_disable_env, saved_affine_disable);
    fprintf(stderr,
            "ds4-test: in-place RoPE total pair=%zu/%zu shared=%zu/%zu "
            "affine=%zu/%zu "
            "pair_prefix=%zu pair_tail=%zu "
            "shared_prefix=%zu shared_tail=%zu "
            "affine_prefix=%zu affine_tail=%zu\n",
            total_pair_mismatch,
            total_elements,
            total_shared_mismatch,
            total_elements,
            total_affine_mismatch,
            total_elements,
            total_pair_prefix_mismatch,
            total_pair_tail_mismatch,
            total_shared_prefix_mismatch,
            total_shared_tail_mismatch,
            total_affine_prefix_mismatch,
            total_affine_tail_mismatch);
    TEST_ASSERT(total_pair_mismatch == 0);
    TEST_ASSERT(total_shared_mismatch == 0);
    TEST_ASSERT(total_affine_mismatch == 0);
    TEST_ASSERT(total_pair_prefix_mismatch == 0);
    TEST_ASSERT(total_pair_tail_mismatch == 0);
    TEST_ASSERT(total_shared_prefix_mismatch == 0);
    TEST_ASSERT(total_shared_tail_mismatch == 0);
    TEST_ASSERT(total_affine_prefix_mismatch == 0);
    TEST_ASSERT(total_affine_tail_mismatch == 0);
}

static void test_metal_contiguous_f32_f16_roundtrip_exact(void) {
    typedef struct {
        uint32_t n;
        uint32_t src_offset;
        uint32_t dst_offset;
    } copy_case;
    static const copy_case cases[] = {
        { 1,  0,  0 },
        { 3,  4,  2 },
        { 4, 16,  8 },
        { 5, 12,  6 },
        { 17, 20, 10 },
        { 65,  4,  2 },
    };
    const char *env_name = "DS4_METAL_DISABLE_CONTIG_F32_F16_COPY";
    char *saved_env = test_save_env(env_name);
    size_t half_mismatch = 0;
    size_t half_guard_mismatch = 0;
    size_t roundtrip_mismatch = 0;
    size_t roundtrip_guard_mismatch = 0;
    size_t half_total = 0;
    size_t roundtrip_total = 0;

    for (size_t ci = 0; ci < sizeof(cases) / sizeof(cases[0]); ci++) {
        const uint32_t n = cases[ci].n;
        const uint64_t src_bytes = cases[ci].src_offset +
                                   (uint64_t)n * sizeof(float) + 16u;
        const uint64_t half_bytes = cases[ci].dst_offset +
                                    (uint64_t)n * sizeof(uint16_t) + 16u;
        const uint32_t raw_cap = 3;
        const uint32_t raw_row = 1;
        const uint64_t raw_bytes =
            (uint64_t)raw_cap * n * sizeof(float);

        ds4_gpu_tensor *src_base = ds4_gpu_tensor_alloc(src_bytes);
        ds4_gpu_tensor *src_view = src_base
            ? ds4_gpu_tensor_view(src_base,
                                  cases[ci].src_offset,
                                  (uint64_t)n * sizeof(float))
            : NULL;
        ds4_gpu_tensor *half_ref = ds4_gpu_tensor_alloc(half_bytes);
        ds4_gpu_tensor *half_vec = ds4_gpu_tensor_alloc(half_bytes);
        ds4_gpu_tensor *raw_ref = ds4_gpu_tensor_alloc(raw_bytes);
        ds4_gpu_tensor *raw_vec = ds4_gpu_tensor_alloc(raw_bytes);
        TEST_ASSERT(src_base != NULL);
        TEST_ASSERT(src_view != NULL);
        TEST_ASSERT(half_ref != NULL);
        TEST_ASSERT(half_vec != NULL);
        TEST_ASSERT(raw_ref != NULL);
        TEST_ASSERT(raw_vec != NULL);

        uint8_t *src_host = malloc((size_t)src_bytes);
        uint16_t *half_init = malloc((size_t)half_bytes);
        uint16_t *half_ref_host = malloc((size_t)half_bytes);
        uint16_t *half_vec_host = malloc((size_t)half_bytes);
        uint32_t *raw_init = malloc((size_t)raw_bytes);
        uint32_t *raw_ref_host = malloc((size_t)raw_bytes);
        uint32_t *raw_vec_host = malloc((size_t)raw_bytes);
        TEST_ASSERT(src_host != NULL);
        TEST_ASSERT(half_init != NULL);
        TEST_ASSERT(half_ref_host != NULL);
        TEST_ASSERT(half_vec_host != NULL);
        TEST_ASSERT(raw_init != NULL);
        TEST_ASSERT(raw_ref_host != NULL);
        TEST_ASSERT(raw_vec_host != NULL);

        const bool allocated = src_base && src_view && half_ref && half_vec &&
            raw_ref && raw_vec && src_host && half_init && half_ref_host &&
            half_vec_host && raw_init && raw_ref_host && raw_vec_host;
        if (allocated) {
            memset(src_host, 0x6d, (size_t)src_bytes);
            test_fill_copy_f32_patterns(src_host + cases[ci].src_offset,
                                        n,
                                        (uint32_t)(ci * 7u));
            const size_t half_words = (size_t)(half_bytes / sizeof(uint16_t));
            for (size_t i = 0; i < half_words; i++) {
                half_init[i] = (uint16_t)(0xa55au ^ (uint16_t)(i * 73u));
            }
            const size_t raw_words = (size_t)raw_cap * n;
            for (size_t i = 0; i < raw_words; i++) {
                raw_init[i] = 0x4a000000u + (uint32_t)i;
            }

            TEST_ASSERT(ds4_gpu_tensor_write(src_base, 0, src_host, src_bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_write(half_ref, 0, half_init, half_bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_write(half_vec, 0, half_init, half_bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_write(raw_ref, 0, raw_init, raw_bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_write(raw_vec, 0, raw_init, raw_bytes) != 0);

            TEST_ASSERT(setenv(env_name, "1", 1) == 0);
            TEST_ASSERT(ds4_gpu_tensor_copy_f32_to_f16(
                half_ref,
                cases[ci].dst_offset,
                src_base,
                cases[ci].src_offset,
                n) != 0);
            TEST_ASSERT(ds4_gpu_store_raw_kv_tensor(
                raw_ref, src_view, raw_cap, raw_row, n) != 0);

            TEST_ASSERT(setenv(env_name, "0", 1) == 0);
            TEST_ASSERT(ds4_gpu_tensor_copy_f32_to_f16(
                half_vec,
                cases[ci].dst_offset,
                src_base,
                cases[ci].src_offset,
                n) != 0);
            TEST_ASSERT(ds4_gpu_store_raw_kv_tensor(
                raw_vec, src_view, raw_cap, raw_row, n) != 0);

            TEST_ASSERT(ds4_gpu_tensor_read(
                half_ref, 0, half_ref_host, half_bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_read(
                half_vec, 0, half_vec_host, half_bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_read(
                raw_ref, 0, raw_ref_host, raw_bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_read(
                raw_vec, 0, raw_vec_host, raw_bytes) != 0);

            const size_t half_first = cases[ci].dst_offset / sizeof(uint16_t);
            const size_t half_last = half_first + n;
            for (size_t i = 0; i < half_words; i++) {
                if (half_ref_host[i] == half_vec_host[i]) continue;
                if (i >= half_first && i < half_last) {
                    half_mismatch++;
                } else {
                    half_guard_mismatch++;
                }
            }

            const size_t raw_first = (size_t)raw_row * n;
            const size_t raw_last = raw_first + n;
            for (size_t i = 0; i < raw_words; i++) {
                if (raw_ref_host[i] == raw_vec_host[i]) continue;
                if (i >= raw_first && i < raw_last) {
                    roundtrip_mismatch++;
                } else {
                    roundtrip_guard_mismatch++;
                }
            }
            half_total += n;
            roundtrip_total += n;
        }

        free(raw_vec_host);
        free(raw_ref_host);
        free(raw_init);
        free(half_vec_host);
        free(half_ref_host);
        free(half_init);
        free(src_host);
        ds4_gpu_tensor_free(raw_vec);
        ds4_gpu_tensor_free(raw_ref);
        ds4_gpu_tensor_free(half_vec);
        ds4_gpu_tensor_free(half_ref);
        ds4_gpu_tensor_free(src_view);
        ds4_gpu_tensor_free(src_base);
    }

    test_restore_env(env_name, saved_env);
    fprintf(stderr,
            "ds4-test: contiguous conversion exactness "
            "f32_f16=%zu/%zu guard=%zu, f16_f32_roundtrip=%zu/%zu guard=%zu\n",
            half_mismatch,
            half_total,
            half_guard_mismatch,
            roundtrip_mismatch,
            roundtrip_total,
            roundtrip_guard_mismatch);
    TEST_ASSERT(half_mismatch == 0);
    TEST_ASSERT(half_guard_mismatch == 0);
    TEST_ASSERT(roundtrip_mismatch == 0);
    TEST_ASSERT(roundtrip_guard_mismatch == 0);
}
#endif

#if defined(__APPLE__)
static void test_metal_gathered_kv_stage_exact(void) {
    const uint32_t head_dim = 512;
    const uint32_t raw_cap = 7;
    const uint32_t n_raw = 5;
    const uint32_t n_comp = 3;
    const uint32_t raw_starts[] = {0, 2, 5, 6};
    const uint64_t raw_bytes =
        (uint64_t)raw_cap * head_dim * sizeof(float);
    const uint64_t comp_bytes =
        (uint64_t)n_comp * head_dim * sizeof(uint16_t);
    const uint64_t payload_bytes =
        ((uint64_t)n_raw + n_comp) * head_dim * sizeof(uint16_t);
    const uint64_t raw_out_bytes =
        (uint64_t)n_raw * head_dim * sizeof(uint16_t);
    const uint64_t raw_view_offset = 4;
    const uint64_t comp_view_offset = 2;
    const uint64_t dst_view_offset = 6;
    const uint64_t raw_base_bytes = raw_view_offset + raw_bytes + 12;
    const uint64_t comp_base_bytes = comp_view_offset + comp_bytes + 14;
    const uint64_t dst_base_bytes = dst_view_offset + payload_bytes + 10;
    const char *envs[] = {
        "DS4_METAL_REQUIRE_GATHERED_KV_STAGE",
        "DS4_METAL_DISABLE_CONTIG_F32_F16_COPY",
        "DS4_METAL_DISABLE_CONTIG_F16_F16_COPY",
    };
    char *saved[sizeof(envs)/sizeof(envs[0])];
    for (size_t i = 0; i < sizeof(envs)/sizeof(envs[0]); i++) {
        saved[i] = test_save_env(envs[i]);
    }

    ds4_gpu_tensor *raw_base = ds4_gpu_tensor_alloc(raw_base_bytes);
    ds4_gpu_tensor *raw = raw_base
        ? ds4_gpu_tensor_view(raw_base, raw_view_offset, raw_bytes)
        : NULL;
    ds4_gpu_tensor *comp_base = ds4_gpu_tensor_alloc(comp_base_bytes);
    ds4_gpu_tensor *comp = comp_base
        ? ds4_gpu_tensor_view(comp_base, comp_view_offset, comp_bytes)
        : NULL;
    ds4_gpu_tensor *ref_base = ds4_gpu_tensor_alloc(dst_base_bytes);
    ds4_gpu_tensor *ref = ref_base
        ? ds4_gpu_tensor_view(ref_base, dst_view_offset, payload_bytes)
        : NULL;
    ds4_gpu_tensor *fused_base = ds4_gpu_tensor_alloc(dst_base_bytes);
    ds4_gpu_tensor *fused = fused_base
        ? ds4_gpu_tensor_view(fused_base, dst_view_offset, payload_bytes)
        : NULL;
    TEST_ASSERT(raw_base != NULL);
    TEST_ASSERT(raw != NULL);
    TEST_ASSERT(comp_base != NULL);
    TEST_ASSERT(comp != NULL);
    TEST_ASSERT(ref_base != NULL);
    TEST_ASSERT(ref != NULL);
    TEST_ASSERT(fused_base != NULL);
    TEST_ASSERT(fused != NULL);

    float *raw_host = malloc((size_t)raw_bytes);
    uint16_t *comp_host = malloc((size_t)comp_bytes);
    uint8_t *dst_init = malloc((size_t)dst_base_bytes);
    uint8_t *ref_host = malloc((size_t)dst_base_bytes);
    uint8_t *fused_host = malloc((size_t)dst_base_bytes);
    TEST_ASSERT(raw_host != NULL);
    TEST_ASSERT(comp_host != NULL);
    TEST_ASSERT(dst_init != NULL);
    TEST_ASSERT(ref_host != NULL);
    TEST_ASSERT(fused_host != NULL);

    static const uint16_t half_patterns[] = {
        0x0000u, 0x8000u, 0x0001u, 0x03ffu, 0x0400u,
        0x3555u, 0x3c00u, 0x3c01u, 0x7bffu, 0xfbffu,
        0x7c00u, 0xfc00u, 0x7e00u, 0x7e01u, 0xfe55u,
    };
    const bool allocated = raw_base && raw && comp_base && comp &&
        ref_base && ref && fused_base && fused && raw_host && comp_host &&
        dst_init && ref_host && fused_host;
    size_t raw_mismatch = 0;
    size_t comp_mismatch = 0;
    size_t guard_mismatch = 0;
    if (allocated) {
        for (uint32_t row = 0; row < raw_cap; row++) {
            test_fill_copy_f32_patterns(
                raw_host + (uint64_t)row * head_dim,
                head_dim,
                row * 17u + 3u);
        }
        for (uint64_t i = 0; i < (uint64_t)n_comp * head_dim; i++) {
            comp_host[i] = half_patterns[(i * 7u + (i >> 3u)) %
                (sizeof(half_patterns)/sizeof(half_patterns[0]))];
        }
        for (uint64_t i = 0; i < dst_base_bytes; i++) {
            dst_init[i] = (uint8_t)(0xa5u ^ (uint8_t)(i * 37u));
        }
        TEST_ASSERT(ds4_gpu_tensor_write(
                        raw, 0, raw_host, raw_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        comp, 0, comp_host, comp_bytes) != 0);
        for (size_t ci = 0;
             ci < sizeof(raw_starts)/sizeof(raw_starts[0]);
             ci++) {
            TEST_ASSERT(ds4_gpu_tensor_write(
                            ref_base, 0, dst_init, dst_base_bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_write(
                            fused_base, 0, dst_init, dst_base_bytes) != 0);

            TEST_ASSERT(unsetenv(envs[0]) == 0);
            TEST_ASSERT(unsetenv(envs[1]) == 0);
            TEST_ASSERT(unsetenv(envs[2]) == 0);
            ds4_gpu_set_quality(true);
            TEST_ASSERT(ds4_gpu_flash_kv_stage_f16_tensor(
                            ref, raw, raw_cap, raw_starts[ci], n_raw,
                            comp, 1, n_comp, head_dim) != 0);

            ds4_gpu_set_quality(false);
            TEST_ASSERT(setenv(envs[0], "1", 1) == 0);
            TEST_ASSERT(ds4_gpu_flash_kv_stage_f16_tensor(
                            fused, raw, raw_cap, raw_starts[ci], n_raw,
                            comp, 1, n_comp, head_dim) != 0);

            TEST_ASSERT(ds4_gpu_tensor_read(
                            ref_base, 0, ref_host, dst_base_bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_read(
                            fused_base, 0, fused_host,
                            dst_base_bytes) != 0);
            for (uint64_t i = 0; i < dst_base_bytes; i++) {
                if (i < dst_view_offset ||
                    i >= dst_view_offset + payload_bytes) {
                    if (ref_host[i] != dst_init[i]) guard_mismatch++;
                    if (fused_host[i] != dst_init[i]) guard_mismatch++;
                } else if (ref_host[i] == fused_host[i]) {
                    continue;
                } else if (i < dst_view_offset + raw_out_bytes) {
                    raw_mismatch++;
                } else {
                    comp_mismatch++;
                }
            }
        }

        /* Component-copy diagnostics and quality mode prevent strict
         * selection of the gathered kernel. */
        TEST_ASSERT(setenv(envs[0], "1", 1) == 0);
        TEST_ASSERT(setenv(envs[1], "1", 1) == 0);
        TEST_ASSERT(ds4_gpu_flash_kv_stage_f16_tensor(
                        fused, raw, raw_cap, 5, n_raw,
                        comp, 1, n_comp, head_dim) == 0);
        TEST_ASSERT(unsetenv(envs[1]) == 0);
        TEST_ASSERT(setenv(envs[2], "1", 1) == 0);
        TEST_ASSERT(ds4_gpu_flash_kv_stage_f16_tensor(
                        fused, raw, raw_cap, 5, n_raw,
                        comp, 1, n_comp, head_dim) == 0);
        TEST_ASSERT(unsetenv(envs[2]) == 0);
        ds4_gpu_set_quality(true);
        TEST_ASSERT(ds4_gpu_flash_kv_stage_f16_tensor(
                        fused, raw, raw_cap, 5, n_raw,
                        comp, 1, n_comp, head_dim) == 0);
        ds4_gpu_set_quality(false);
    }

    for (size_t i = 0; i < sizeof(envs)/sizeof(envs[0]); i++) {
        test_restore_env(envs[i], saved[i]);
    }
    fprintf(stderr,
            "ds4-test: gathered KV staging exact cases=%zu "
            "raw_bytes=%zu comp_bytes=%zu guard_bytes=%zu\n",
            sizeof(raw_starts)/sizeof(raw_starts[0]),
            raw_mismatch, comp_mismatch, guard_mismatch);
    TEST_ASSERT(raw_mismatch == 0);
    TEST_ASSERT(comp_mismatch == 0);
    TEST_ASSERT(guard_mismatch == 0);

    free(fused_host);
    free(ref_host);
    free(dst_init);
    free(comp_host);
    free(raw_host);
    ds4_gpu_tensor_free(fused);
    ds4_gpu_tensor_free(fused_base);
    ds4_gpu_tensor_free(ref);
    ds4_gpu_tensor_free(ref_base);
    ds4_gpu_tensor_free(comp);
    ds4_gpu_tensor_free(comp_base);
    ds4_gpu_tensor_free(raw);
    ds4_gpu_tensor_free(raw_base);
}

static void test_metal_contiguous_compressed_f16_attention_exact(void) {
    const uint32_t head_dim = 512;
    const uint32_t n_head = 2;
    const uint32_t raw_cap = 7;
    const uint32_t n_raw = 5;
    const uint32_t raw_start = 5;
    const uint32_t n_comp = 3;
    const uint64_t raw_bytes =
        (uint64_t)raw_cap * head_dim * sizeof(float);
    const uint64_t comp_bytes =
        (uint64_t)n_comp * head_dim * sizeof(uint16_t);
    const uint64_t q_bytes =
        (uint64_t)n_head * head_dim * sizeof(float);
    const uint64_t page = (uint64_t)getpagesize();
    const char *env_name = "DS4_METAL_DISABLE_CONTIG_F16_F16_COPY";
    static const uint16_t half_patterns[] = {
        0x0000u, 0x8000u, 0x0001u, 0x03ffu, 0x0400u,
        0x1001u, 0x3555u, 0x3c00u, 0x3c01u, 0x4000u,
        0xbc00u, 0xc000u, 0x7bffu, 0xfbffu,
    };

    ds4_gpu_tensor *raw = ds4_gpu_tensor_alloc(raw_bytes);
    ds4_gpu_tensor *comp_base = ds4_gpu_tensor_alloc(comp_bytes + 18u);
    ds4_gpu_tensor *comp = comp_base
        ? ds4_gpu_tensor_view(comp_base, 2u, comp_bytes)
        : NULL;
    ds4_gpu_tensor *q = ds4_gpu_tensor_alloc(q_bytes);
    ds4_gpu_tensor *heads_blit = ds4_gpu_tensor_alloc(q_bytes);
    ds4_gpu_tensor *heads_compute = ds4_gpu_tensor_alloc(q_bytes);
    TEST_ASSERT(raw != NULL);
    TEST_ASSERT(comp_base != NULL);
    TEST_ASSERT(comp != NULL);
    TEST_ASSERT(q != NULL);
    TEST_ASSERT(heads_blit != NULL);
    TEST_ASSERT(heads_compute != NULL);

    float *raw_host = malloc((size_t)raw_bytes);
    uint16_t *comp_host = malloc((size_t)comp_bytes);
    float *q_host = malloc((size_t)q_bytes);
    float *blit_host = malloc((size_t)q_bytes);
    float *compute_host = malloc((size_t)q_bytes);
    TEST_ASSERT(raw_host != NULL);
    TEST_ASSERT(comp_host != NULL);
    TEST_ASSERT(q_host != NULL);
    TEST_ASSERT(blit_host != NULL);
    TEST_ASSERT(compute_host != NULL);

    void *model_raw = NULL;
    TEST_ASSERT(posix_memalign(&model_raw, (size_t)page, (size_t)page) == 0);
    const bool allocated = raw && comp_base && comp && q && heads_blit &&
        heads_compute && raw_host && comp_host && q_host && blit_host &&
        compute_host && model_raw;
    char *saved_env = test_save_env(env_name);
    test_float_compare_stats stats = {0};
    if (allocated) {
        memset(model_raw, 0, (size_t)page);
        float *sinks = model_raw;
        sinks[0] = -0.375f;
        sinks[1] = 0.1875f;
        for (uint64_t i = 0; i < (uint64_t)raw_cap * head_dim; i++) {
            const int value =
                (int)((i * 19u + (i ^ (i >> 5u)) * 7u) % 193u) - 96;
            raw_host[i] = (float)value / 128.0f;
        }
        for (uint64_t i = 0; i < (uint64_t)n_comp * head_dim; i++) {
            comp_host[i] = half_patterns[(i * 5u + (i >> 4u)) %
                (sizeof(half_patterns) / sizeof(half_patterns[0]))];
        }
        for (uint32_t i = 0; i < n_head * head_dim; i++) {
            const int value = (int)((i * 37u + (i ^ (i >> 3u)) * 11u) % 251u) - 125;
            q_host[i] = (float)value / 96.0f;
        }

        TEST_ASSERT(ds4_gpu_tensor_write(raw, 0, raw_host, raw_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(comp, 0, comp_host, comp_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(q, 0, q_host, q_bytes) != 0);
        TEST_ASSERT(ds4_gpu_set_model_map(model_raw, page) != 0);
        ds4_gpu_set_quality(false);

        TEST_ASSERT(setenv(env_name, "1", 1) == 0);
        TEST_ASSERT(ds4_gpu_attention_decode_heads_tensor(
            heads_blit,
            model_raw,
            page,
            0,
            q,
            raw,
            n_raw,
            raw_cap,
            raw_start,
            comp,
            1,
            n_comp,
            NULL,
            0,
            n_head,
            head_dim) != 0);

        TEST_ASSERT(setenv(env_name, "0", 1) == 0);
        TEST_ASSERT(ds4_gpu_attention_decode_heads_tensor(
            heads_compute,
            model_raw,
            page,
            0,
            q,
            raw,
            n_raw,
            raw_cap,
            raw_start,
            comp,
            1,
            n_comp,
            NULL,
            0,
            n_head,
            head_dim) != 0);

        TEST_ASSERT(ds4_gpu_tensor_read(
            heads_blit, 0, blit_host, q_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
            heads_compute, 0, compute_host, q_bytes) != 0);
        stats = test_compare_float_bits(
            blit_host, compute_host, (size_t)n_head * head_dim);
    }
    test_restore_env(env_name, saved_env);
    fprintf(stderr,
            "ds4-test: contiguous compressed-F16 staging exactness "
            "mismatches=%zu/%u max_ulp=%u max_abs=%g\n",
            stats.mismatch_count,
            n_head * head_dim,
            stats.max_ulp,
            stats.max_abs);
    TEST_ASSERT(stats.mismatch_count == 0);

    free(model_raw);
    free(compute_host);
    free(blit_host);
    free(q_host);
    free(comp_host);
    free(raw_host);
    ds4_gpu_tensor_free(heads_compute);
    ds4_gpu_tensor_free(heads_blit);
    ds4_gpu_tensor_free(q);
    ds4_gpu_tensor_free(comp);
    ds4_gpu_tensor_free(comp_base);
    ds4_gpu_tensor_free(raw);
}

static void test_metal_persistent_zero_attention_mask_exact_case(
        uint32_t raw_cap,
        uint32_t n_raw,
        uint32_t raw_start,
        uint32_t n_comp,
        uint32_t seed) {
    const uint32_t head_dim = 512;
    const uint32_t n_head = 2;
    const uint64_t raw_bytes =
        (uint64_t)raw_cap * head_dim * sizeof(float);
    const uint64_t comp_bytes =
        (uint64_t)n_comp * head_dim * sizeof(uint16_t);
    const uint64_t q_bytes =
        (uint64_t)n_head * head_dim * sizeof(float);
    const uint64_t mask_bytes = (uint64_t)n_comp * sizeof(float);
    const uint64_t page = (uint64_t)getpagesize();

    ds4_gpu_tensor *raw = ds4_gpu_tensor_alloc(raw_bytes);
    ds4_gpu_tensor *comp = ds4_gpu_tensor_alloc(comp_bytes);
    ds4_gpu_tensor *q = ds4_gpu_tensor_alloc(q_bytes);
    ds4_gpu_tensor *comp_mask = ds4_gpu_tensor_alloc(mask_bytes);
    ds4_gpu_tensor *legacy = ds4_gpu_tensor_alloc(q_bytes);
    ds4_gpu_tensor *persistent = ds4_gpu_tensor_alloc(q_bytes);
    ds4_gpu_tensor *masked = ds4_gpu_tensor_alloc(q_bytes);
    ds4_gpu_tensor *pad_legacy = ds4_gpu_tensor_alloc(q_bytes);
    ds4_gpu_tensor *after_mask = ds4_gpu_tensor_alloc(q_bytes);
    TEST_ASSERT(raw != NULL);
    TEST_ASSERT(comp != NULL);
    TEST_ASSERT(q != NULL);
    TEST_ASSERT(comp_mask != NULL);
    TEST_ASSERT(legacy != NULL);
    TEST_ASSERT(persistent != NULL);
    TEST_ASSERT(masked != NULL);
    TEST_ASSERT(pad_legacy != NULL);
    TEST_ASSERT(after_mask != NULL);

    float *raw_host = malloc((size_t)raw_bytes);
    uint16_t *comp_host = malloc((size_t)comp_bytes);
    float *q_host = malloc((size_t)q_bytes);
    float *mask_host = malloc((size_t)mask_bytes);
    float *legacy_host = malloc((size_t)q_bytes);
    float *persistent_host = malloc((size_t)q_bytes);
    float *masked_host = malloc((size_t)q_bytes);
    float *pad_legacy_host = malloc((size_t)q_bytes);
    float *after_mask_host = malloc((size_t)q_bytes);
    TEST_ASSERT(raw_host != NULL);
    TEST_ASSERT(comp_host != NULL);
    TEST_ASSERT(q_host != NULL);
    TEST_ASSERT(mask_host != NULL);
    TEST_ASSERT(legacy_host != NULL);
    TEST_ASSERT(persistent_host != NULL);
    TEST_ASSERT(masked_host != NULL);
    TEST_ASSERT(pad_legacy_host != NULL);
    TEST_ASSERT(after_mask_host != NULL);

    void *model_raw = NULL;
    TEST_ASSERT(posix_memalign(&model_raw, (size_t)page, (size_t)page) == 0);
    const bool allocated = raw && comp && q && comp_mask && legacy &&
        persistent && masked && pad_legacy && after_mask && raw_host &&
        comp_host && q_host && mask_host && legacy_host && persistent_host &&
        masked_host && pad_legacy_host && after_mask_host &&
        model_raw;
    const char *disable_env =
        "DS4_METAL_DISABLE_PERSISTENT_ZERO_ATTN_MASK";
    const char *pad_disable_env =
        "DS4_METAL_DISABLE_GATHERED_KV_PAD_FUSION";
    const char *shared_pad_disable_env =
        "DS4_METAL_DISABLE_SHARED_KV_PAD";
    char *saved_disable = test_save_env(disable_env);
    char *saved_pad_disable = test_save_env(pad_disable_env);
    char *saved_shared_pad_disable = test_save_env(shared_pad_disable_env);
    test_float_compare_stats persistent_stats = {0};
    test_float_compare_stats masked_stats = {0};
    test_float_compare_stats pad_stats = {0};
    test_float_compare_stats after_mask_stats = {0};

    if (allocated) {
        memset(model_raw, 0, (size_t)page);
        float *sinks = model_raw;
        sinks[0] = -0.3125f;
        sinks[1] = 0.21875f;
        for (uint64_t i = 0; i < (uint64_t)raw_cap * head_dim; i++) {
            const int value = (int)((i * 17u + (i ^ (i >> 5u)) * 11u +
                                     seed * 13u) % 211u) - 105;
            raw_host[i] = (float)value / 128.0f;
        }
        for (uint64_t i = 0; i < (uint64_t)n_comp * head_dim; i++) {
            const int value = (int)((i * 23u + (i ^ (i >> 4u)) * 7u +
                                     seed * 19u) % 193u) - 96;
            comp_host[i] = test_float_to_f16((float)value / 112.0f);
        }
        for (uint32_t i = 0; i < n_head * head_dim; i++) {
            const int value = (int)((i * 31u + (i ^ (i >> 3u)) * 5u +
                                     seed * 29u) % 227u) - 113;
            q_host[i] = (float)value / 104.0f;
        }
        for (uint32_t i = 0; i < n_comp; i++) {
            mask_host[i] = i == 0 ? -8.0f : -(float)(i + 1u) / 8.0f;
        }

        TEST_ASSERT(ds4_gpu_tensor_write(raw, 0, raw_host, raw_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(comp, 0, comp_host, comp_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(q, 0, q_host, q_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        comp_mask, 0, mask_host, mask_bytes) != 0);
        TEST_ASSERT(ds4_gpu_set_model_map(model_raw, page) != 0);
        ds4_gpu_set_quality(false);

        TEST_ASSERT(setenv(disable_env, "1", 1) == 0);
        TEST_ASSERT(ds4_gpu_attention_decode_heads_tensor(
            legacy, model_raw, page, 0, q, raw,
            n_raw, raw_cap, raw_start, comp, 1, n_comp,
            NULL, 0, n_head, head_dim) != 0);

        unsetenv(disable_env);
        unsetenv(pad_disable_env);
        unsetenv(shared_pad_disable_env);
        TEST_ASSERT(ds4_gpu_attention_decode_heads_tensor(
            persistent, model_raw, page, 0, q, raw,
            n_raw, raw_cap, raw_start, comp, 1, n_comp,
            NULL, 0, n_head, head_dim) != 0);

        TEST_ASSERT(ds4_gpu_attention_decode_heads_tensor(
            masked, model_raw, page, 0, q, raw,
            n_raw, raw_cap, raw_start, comp, 1, n_comp,
            comp_mask, 1, n_head, head_dim) != 0);

        TEST_ASSERT(setenv(pad_disable_env, "1", 1) == 0);
        TEST_ASSERT(setenv(shared_pad_disable_env, "1", 1) == 0);
        TEST_ASSERT(ds4_gpu_attention_decode_heads_tensor(
            pad_legacy, model_raw, page, 0, q, raw,
            n_raw, raw_cap, raw_start, comp, 1, n_comp,
            comp_mask, 1, n_head, head_dim) != 0);
        unsetenv(pad_disable_env);
        unsetenv(shared_pad_disable_env);

        TEST_ASSERT(ds4_gpu_attention_decode_heads_tensor(
            after_mask, model_raw, page, 0, q, raw,
            n_raw, raw_cap, raw_start, comp, 1, n_comp,
            NULL, 0, n_head, head_dim) != 0);

        TEST_ASSERT(ds4_gpu_tensor_read(
                        legacy, 0, legacy_host, q_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        persistent, 0, persistent_host, q_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        masked, 0, masked_host, q_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        pad_legacy, 0, pad_legacy_host, q_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        after_mask, 0, after_mask_host, q_bytes) != 0);
        persistent_stats = test_compare_float_bits(
            legacy_host, persistent_host, (size_t)n_head * head_dim);
        masked_stats = test_compare_float_bits(
            legacy_host, masked_host, (size_t)n_head * head_dim);
        pad_stats = test_compare_float_bits(
            pad_legacy_host, masked_host, (size_t)n_head * head_dim);
        after_mask_stats = test_compare_float_bits(
            legacy_host, after_mask_host, (size_t)n_head * head_dim);
    }

    test_restore_env(disable_env, saved_disable);
    test_restore_env(pad_disable_env, saved_pad_disable);
    test_restore_env(shared_pad_disable_env, saved_shared_pad_disable);
    fprintf(stderr,
            "ds4-test: persistent zero attention mask exact keys=%u "
            "candidate=%zu/%u max_ulp=%u masked_diff=%zu/%u "
            "pad_fusion=%zu/%u max_ulp=%u "
            "after_mask=%zu/%u max_ulp=%u\n",
            n_raw + n_comp,
            persistent_stats.mismatch_count, n_head * head_dim,
            persistent_stats.max_ulp,
            masked_stats.mismatch_count, n_head * head_dim,
            pad_stats.mismatch_count, n_head * head_dim,
            pad_stats.max_ulp,
            after_mask_stats.mismatch_count, n_head * head_dim,
            after_mask_stats.max_ulp);
    TEST_ASSERT(persistent_stats.mismatch_count == 0);
    TEST_ASSERT(masked_stats.mismatch_count != 0);
    TEST_ASSERT(pad_stats.mismatch_count == 0);
    TEST_ASSERT(after_mask_stats.mismatch_count == 0);

    free(model_raw);
    free(after_mask_host);
    free(pad_legacy_host);
    free(masked_host);
    free(persistent_host);
    free(legacy_host);
    free(mask_host);
    free(q_host);
    free(comp_host);
    free(raw_host);
    ds4_gpu_tensor_free(after_mask);
    ds4_gpu_tensor_free(pad_legacy);
    ds4_gpu_tensor_free(masked);
    ds4_gpu_tensor_free(persistent);
    ds4_gpu_tensor_free(legacy);
    ds4_gpu_tensor_free(comp_mask);
    ds4_gpu_tensor_free(q);
    ds4_gpu_tensor_free(comp);
    ds4_gpu_tensor_free(raw);
}

static void test_metal_persistent_zero_attention_mask_exact(void) {
    test_metal_persistent_zero_attention_mask_exact_case(7, 5, 5, 3, 11);
    test_metal_persistent_zero_attention_mask_exact_case(37, 29, 35, 3, 23);
}

typedef enum {
    TEST_METAL_PREFILL_MASK_CACHE_RAW = 1,
    TEST_METAL_PREFILL_MASK_CACHE_RATIO4 = 2,
    TEST_METAL_PREFILL_MASK_CACHE_RATIO128 = 3,
} test_metal_prefill_mask_cache_kind;

typedef struct {
    uint32_t n_tokens;
    uint32_t n_comp;
    uint32_t window;
    uint32_t ratio;
} test_metal_prefill_mask_cache_shape;

static int test_metal_zero_prefix_prefill_mask_cache_call(
        test_metal_prefill_mask_cache_kind  kind,
        ds4_gpu_tensor                    *heads,
        const void                        *model_map,
        uint64_t                           model_size,
        const ds4_gpu_tensor              *q,
        const ds4_gpu_tensor              *raw,
        const ds4_gpu_tensor              *comp,
        const ds4_gpu_tensor              *comp_mask,
        const test_metal_prefill_mask_cache_shape *shape,
        bool                               masked,
        uint32_t                           n_head,
        uint32_t                           head_dim) {
    if (kind == TEST_METAL_PREFILL_MASK_CACHE_RAW) {
        if (masked) return 0;
        return ds4_gpu_attention_prefill_raw_heads_tensor(
            heads, model_map, model_size, 0, q, raw,
            shape->n_tokens, shape->window, n_head, head_dim);
    }

    if (masked) {
        return ds4_gpu_attention_prefill_masked_mixed_heads_tensor(
            heads, model_map, model_size, 0, q, raw, comp, 1, comp_mask,
            shape->n_tokens, shape->n_comp, shape->window, shape->ratio,
            n_head, head_dim);
    }

    return ds4_gpu_attention_prefill_static_mixed_heads_tensor(
        heads, model_map, model_size, 0, q, raw, comp, 1,
        shape->n_tokens, shape->n_comp, shape->window, shape->ratio,
        n_head, head_dim);
}

static bool test_metal_zero_prefix_prefill_mask_cache_run(
        test_metal_prefill_mask_cache_kind  kind,
        ds4_gpu_tensor                    *heads,
        const void                        *model_map,
        uint64_t                           model_size,
        const ds4_gpu_tensor              *q,
        const ds4_gpu_tensor              *raw,
        const ds4_gpu_tensor              *comp,
        const ds4_gpu_tensor              *comp_mask,
        const test_metal_prefill_mask_cache_shape *shape,
        bool                               masked,
        uint32_t                           n_head,
        uint32_t                           head_dim,
        float                             *host) {
    const uint64_t bytes =
        (uint64_t)shape->n_tokens * n_head * head_dim * sizeof(float);
    const int call_ok = test_metal_zero_prefix_prefill_mask_cache_call(
        kind, heads, model_map, model_size, q, raw, comp, comp_mask,
        shape, masked, n_head, head_dim);
    TEST_ASSERT(call_ok != 0);
    if (!call_ok) return false;

    const int read_ok = ds4_gpu_tensor_read(heads, 0, host, bytes);
    TEST_ASSERT(read_ok != 0);
    return read_ok != 0;
}

static void test_metal_zero_prefix_prefill_mask_cache_compare(
        const float *expected,
        const float *actual,
        size_t       count,
        size_t      *total_mismatches,
        uint32_t    *max_ulp) {
    const test_float_compare_stats stats =
        test_compare_float_bits(expected, actual, count);
    *total_mismatches += stats.mismatch_count;
    if (stats.max_ulp > *max_ulp) *max_ulp = stats.max_ulp;
}

static void test_metal_zero_prefix_prefill_mask_cache_exact_kind(
        test_metal_prefill_mask_cache_kind kind,
        uint32_t                           seed) {
    const uint32_t head_dim = 512;
    const uint32_t n_head = 1;
    const uint32_t max_tokens = 129;
    const uint32_t max_comp = 32;
    const uint64_t raw_count = (uint64_t)max_tokens * head_dim;
    const uint64_t comp_count = (uint64_t)max_comp * head_dim;
    const uint64_t q_count = (uint64_t)max_tokens * n_head * head_dim;
    const uint64_t mask_count = (uint64_t)max_tokens * max_comp;
    const uint64_t raw_bytes = raw_count * sizeof(float);
    const uint64_t comp_bytes = comp_count * sizeof(uint16_t);
    const uint64_t q_bytes = q_count * sizeof(float);
    const uint64_t mask_bytes = mask_count * sizeof(float);
    const uint64_t page = (uint64_t)getpagesize();

    const test_metal_prefill_mask_cache_shape shape_a = {
        .n_tokens = 128,
        .n_comp = kind == TEST_METAL_PREFILL_MASK_CACHE_RAW ? 0u :
                  (kind == TEST_METAL_PREFILL_MASK_CACHE_RATIO4 ? 32u : 1u),
        .window = 128,
        .ratio = kind == TEST_METAL_PREFILL_MASK_CACHE_RATIO4 ? 4u :
                 (kind == TEST_METAL_PREFILL_MASK_CACHE_RATIO128 ? 128u : 0u),
    };
    const test_metal_prefill_mask_cache_shape shape_b = {
        .n_tokens = 129,
        .n_comp = kind == TEST_METAL_PREFILL_MASK_CACHE_RAW ? 0u :
                  (kind == TEST_METAL_PREFILL_MASK_CACHE_RATIO4 ? 31u : 2u),
        .window = 63,
        .ratio = shape_a.ratio,
    };
    const size_t count_a =
        (size_t)shape_a.n_tokens * n_head * head_dim;
    const size_t count_b =
        (size_t)shape_b.n_tokens * n_head * head_dim;

    ds4_gpu_tensor *raw = ds4_gpu_tensor_alloc(raw_bytes);
    ds4_gpu_tensor *comp = ds4_gpu_tensor_alloc(comp_bytes);
    ds4_gpu_tensor *q = ds4_gpu_tensor_alloc(q_bytes);
    ds4_gpu_tensor *comp_mask = ds4_gpu_tensor_alloc(mask_bytes);
    ds4_gpu_tensor *heads = ds4_gpu_tensor_alloc(q_bytes);
    float *raw_host = malloc((size_t)raw_bytes);
    uint16_t *comp_host = malloc((size_t)comp_bytes);
    float *q_host = calloc((size_t)q_count, sizeof(float));
    float *mask_host = malloc((size_t)mask_bytes);
    float *ref_a = malloc((size_t)q_bytes);
    float *ref_b = malloc((size_t)q_bytes);
    float *actual = malloc((size_t)q_bytes);
    float *masked_actual = malloc((size_t)q_bytes);
    void *model_raw = NULL;
    TEST_ASSERT(posix_memalign(&model_raw, (size_t)page, (size_t)page) == 0);

    TEST_ASSERT(raw != NULL);
    TEST_ASSERT(comp != NULL);
    TEST_ASSERT(q != NULL);
    TEST_ASSERT(comp_mask != NULL);
    TEST_ASSERT(heads != NULL);
    TEST_ASSERT(raw_host != NULL);
    TEST_ASSERT(comp_host != NULL);
    TEST_ASSERT(q_host != NULL);
    TEST_ASSERT(mask_host != NULL);
    TEST_ASSERT(ref_a != NULL);
    TEST_ASSERT(ref_b != NULL);
    TEST_ASSERT(actual != NULL);
    TEST_ASSERT(masked_actual != NULL);
    TEST_ASSERT(model_raw != NULL);

    const char *disable_env =
        "DS4_METAL_DISABLE_ZERO_PREFIX_PREFILL_MASK_CACHE";
    char *saved_disable = test_save_env(disable_env);
    size_t total_mismatches = 0;
    uint32_t max_ulp = 0;
    size_t key_difference = 0;
    size_t masked_difference = 0;

    const bool allocated = raw && comp && q && comp_mask && heads &&
        raw_host && comp_host && q_host && mask_host && ref_a && ref_b &&
        actual && masked_actual && model_raw;
    if (allocated) {
        memset(model_raw, 0, (size_t)page);
        ((float *)model_raw)[0] = -1.0f;

        for (uint32_t row = 0; row < max_tokens; row++) {
            for (uint32_t col = 0; col < head_dim; col++) {
                const int value = (int)((row * 37u + col * 17u +
                                         (col ^ (col >> 3u)) * 5u +
                                         seed * 13u) % 257u) - 128;
                raw_host[(uint64_t)row * head_dim + col] =
                    (float)value / 256.0f;
            }
        }
        for (uint32_t row = 0; row < max_comp; row++) {
            for (uint32_t col = 0; col < head_dim; col++) {
                const int value = (int)((row * 29u + col * 11u +
                                         (col ^ (col >> 4u)) * 7u +
                                         seed * 19u) % 193u) - 96;
                comp_host[(uint64_t)row * head_dim + col] =
                    test_float_to_f16(0.375f + (float)value / 384.0f);
            }
        }
        for (uint64_t i = 0; i < mask_count; i++) {
            mask_host[i] = -65504.0f;
        }
        if (shape_a.n_comp != 0) {
            for (uint32_t row = 0; row < shape_a.n_tokens; row++) {
                const uint32_t visible = (row + 1u) / shape_a.ratio;
                for (uint32_t col = 0; col < shape_a.n_comp; col++) {
                    if (col < visible) {
                        mask_host[(uint64_t)row * shape_a.n_comp + col] =
                            (col & 1u) != 0u ? -2.0f : -65504.0f;
                    }
                }
            }
        }

        TEST_ASSERT(ds4_gpu_tensor_write(raw, 0, raw_host, raw_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(comp, 0, comp_host, comp_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(q, 0, q_host, q_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        comp_mask, 0, mask_host, mask_bytes) != 0);
        TEST_ASSERT(ds4_gpu_set_model_map(model_raw, page) != 0);
        ds4_gpu_set_quality(false);

        TEST_ASSERT(setenv(disable_env, "1", 1) == 0);
        const bool have_ref_a =
            test_metal_zero_prefix_prefill_mask_cache_run(
                kind, heads, model_raw, page, q, raw, comp, comp_mask,
                &shape_a, false, n_head, head_dim, ref_a);
        const bool have_ref_b =
            test_metal_zero_prefix_prefill_mask_cache_run(
                kind, heads, model_raw, page, q, raw, comp, comp_mask,
                &shape_b, false, n_head, head_dim, ref_b);

        if (have_ref_a && have_ref_b) {
            const test_float_compare_stats key_stats =
                test_compare_float_bits(ref_a, ref_b, count_a);
            key_difference = key_stats.mismatch_count;
            TEST_ASSERT(key_difference != 0);
        }

        TEST_ASSERT(unsetenv(disable_env) == 0);

        if (test_metal_zero_prefix_prefill_mask_cache_run(
                kind, heads, model_raw, page, q, raw, comp, comp_mask,
                &shape_a, false, n_head, head_dim, actual) && have_ref_a) {
            test_metal_zero_prefix_prefill_mask_cache_compare(
                ref_a, actual, count_a, &total_mismatches, &max_ulp);
        }
        if (test_metal_zero_prefix_prefill_mask_cache_run(
                kind, heads, model_raw, page, q, raw, comp, comp_mask,
                &shape_a, false, n_head, head_dim, actual) && have_ref_a) {
            test_metal_zero_prefix_prefill_mask_cache_compare(
                ref_a, actual, count_a, &total_mismatches, &max_ulp);
        }

        if (kind != TEST_METAL_PREFILL_MASK_CACHE_RAW) {
            if (test_metal_zero_prefix_prefill_mask_cache_run(
                    kind, heads, model_raw, page, q, raw, comp, comp_mask,
                    &shape_a, true, n_head, head_dim, masked_actual) &&
                have_ref_a) {
                const test_float_compare_stats masked_stats =
                    test_compare_float_bits(ref_a, masked_actual, count_a);
                masked_difference = masked_stats.mismatch_count;
                TEST_ASSERT(masked_difference != 0);
            }
            if (test_metal_zero_prefix_prefill_mask_cache_run(
                    kind, heads, model_raw, page, q, raw, comp, comp_mask,
                    &shape_a, false, n_head, head_dim, actual) && have_ref_a) {
                test_metal_zero_prefix_prefill_mask_cache_compare(
                    ref_a, actual, count_a,
                    &total_mismatches, &max_ulp);
            }
        }

        if (test_metal_zero_prefix_prefill_mask_cache_run(
                kind, heads, model_raw, page, q, raw, comp, comp_mask,
                &shape_b, false, n_head, head_dim, actual) && have_ref_b) {
            test_metal_zero_prefix_prefill_mask_cache_compare(
                ref_b, actual, count_b, &total_mismatches, &max_ulp);
        }
        if (test_metal_zero_prefix_prefill_mask_cache_run(
                kind, heads, model_raw, page, q, raw, comp, comp_mask,
                &shape_a, false, n_head, head_dim, actual) && have_ref_a) {
            test_metal_zero_prefix_prefill_mask_cache_compare(
                ref_a, actual, count_a, &total_mismatches, &max_ulp);
        }
    }

    test_restore_env(disable_env, saved_disable);
    const char *kind_name = kind == TEST_METAL_PREFILL_MASK_CACHE_RAW ? "raw" :
        (kind == TEST_METAL_PREFILL_MASK_CACHE_RATIO4 ? "ratio4" : "ratio128");
    fprintf(stderr,
            "ds4-test: zero-prefix prefill mask cache %s exact "
            "mismatches=%zu max_ulp=%u key_diff=%zu masked_diff=%zu\n",
            kind_name, total_mismatches, max_ulp,
            key_difference, masked_difference);
    TEST_ASSERT(total_mismatches == 0);
    TEST_ASSERT(max_ulp == 0);

    free(model_raw);
    free(masked_actual);
    free(actual);
    free(ref_b);
    free(ref_a);
    free(mask_host);
    free(q_host);
    free(comp_host);
    free(raw_host);
    ds4_gpu_tensor_free(heads);
    ds4_gpu_tensor_free(comp_mask);
    ds4_gpu_tensor_free(q);
    ds4_gpu_tensor_free(comp);
    ds4_gpu_tensor_free(raw);
}

static void test_metal_zero_prefix_prefill_mask_cache_exact(void) {
    test_metal_zero_prefix_prefill_mask_cache_exact_kind(
        TEST_METAL_PREFILL_MASK_CACHE_RAW, 41);
    test_metal_zero_prefix_prefill_mask_cache_exact_kind(
        TEST_METAL_PREFILL_MASK_CACHE_RATIO4, 43);
    test_metal_zero_prefix_prefill_mask_cache_exact_kind(
        TEST_METAL_PREFILL_MASK_CACHE_RATIO128, 47);
}

static double test_metal_small_prefill_gpu_batch_ms(
        test_metal_prefill_mask_cache_kind kind,
        bool masked,
        const test_metal_prefill_mask_cache_shape *shape,
        ds4_gpu_tensor *heads,
        const void *model_map,
        uint64_t model_size,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw,
        const ds4_gpu_tensor *comp,
        const ds4_gpu_tensor *comp_mask,
        uint32_t n_head,
        uint32_t head_dim,
        uint32_t flags,
        uint32_t repeats,
        uint32_t expected_nwg) {
    ds4_gpu_test_set_flags(flags);
    const int begin_ok = ds4_gpu_begin_commands();
    TEST_ASSERT(begin_ok != 0);
    if (!begin_ok) return -1.0;
    for (uint32_t rep = 0u; rep < repeats; rep++) {
        const int call_ok = test_metal_zero_prefix_prefill_mask_cache_call(
            kind, heads, model_map, model_size, q, raw, comp, comp_mask,
            shape, masked, n_head, head_dim);
        TEST_ASSERT(call_ok != 0);
        if (!call_ok) {
            (void)ds4_gpu_end_commands();
            return -1.0;
        }
    }
    const int end_ok = ds4_gpu_end_commands();
    TEST_ASSERT(end_ok != 0);
    if (!end_ok) return -1.0;
    const uint32_t selected_nwg =
        ds4_gpu_test_last_flash_attn_prefill_nwg();
    TEST_ASSERT(selected_nwg == expected_nwg);
    if (selected_nwg != expected_nwg) return -1.0;
    const double gpu_ms = ds4_gpu_test_last_completed_gpu_ms();
    TEST_ASSERT(gpu_ms > 0.0);
    return gpu_ms > 0.0 ? gpu_ms / (double)repeats : -1.0;
}

static void test_metal_small_prefill_direct_exact(void) {
    const uint32_t head_dim = 512u;
    const uint32_t n_head = 2u;
    const uint32_t max_tokens = 19u;
    const uint32_t max_comp = 14u;
    const uint64_t guard_bytes = 256u;
    const uint64_t raw_count = (uint64_t)max_tokens * head_dim;
    const uint64_t comp_count = (uint64_t)max_comp * head_dim;
    const uint64_t q_count =
        (uint64_t)max_tokens * n_head * head_dim;
    const uint64_t mask_count = (uint64_t)max_tokens * max_comp;
    const uint64_t raw_bytes = raw_count * sizeof(float);
    const uint64_t comp_bytes = comp_count * sizeof(uint16_t);
    const uint64_t q_bytes = q_count * sizeof(float);
    const uint64_t mask_bytes = mask_count * sizeof(float);
    const uint64_t heads_base_bytes = guard_bytes + q_bytes + guard_bytes;
    const uint64_t page = (uint64_t)getpagesize();
    const char *disable_env = "DS4_METAL_DISABLE_SMALL_PREFILL_DIRECT";
    char *saved_disable = test_save_env(disable_env);
    const char *stage_profile_env = "DS4_METAL_FLASH_ATTN_STAGE_PROFILE";
    char *saved_stage_profile = test_save_env(stage_profile_env);

    typedef struct {
        const char *name;
        test_metal_prefill_mask_cache_kind kind;
        bool masked;
        test_metal_prefill_mask_cache_shape shape;
    } test_metal_small_prefill_case;
    static const test_metal_small_prefill_case cases[] = {
        {"raw-1", TEST_METAL_PREFILL_MASK_CACHE_RAW, false,
         {1u, 0u, 0u, 0u}},
        {"raw-7", TEST_METAL_PREFILL_MASK_CACHE_RAW, false,
         {7u, 0u, 5u, 0u}},
        {"raw-19", TEST_METAL_PREFILL_MASK_CACHE_RAW, false,
         {19u, 0u, 11u, 0u}},
        {"static-12", TEST_METAL_PREFILL_MASK_CACHE_RATIO4, false,
         {7u, 5u, 5u, 4u}},
        {"static-32", TEST_METAL_PREFILL_MASK_CACHE_RATIO4, false,
         {19u, 13u, 11u, 4u}},
        {"masked-12", TEST_METAL_PREFILL_MASK_CACHE_RATIO4, true,
         {7u, 5u, 5u, 4u}},
        {"masked-32", TEST_METAL_PREFILL_MASK_CACHE_RATIO4, true,
         {19u, 13u, 11u, 4u}},
        {"static-33-fallback", TEST_METAL_PREFILL_MASK_CACHE_RATIO4, false,
         {19u, 14u, 11u, 4u}},
    };
    enum {
        TEST_SMALL_PREFILL_DIRECT_COLD = 0,
        TEST_SMALL_PREFILL_PSO_FALLBACK,
        TEST_SMALL_PREFILL_ENV_ROLLBACK,
        TEST_SMALL_PREFILL_DIRECT_REPLAY,
        TEST_SMALL_PREFILL_ARM_COUNT,
    };

    ds4_gpu_tensor *raw = ds4_gpu_tensor_alloc(raw_bytes);
    ds4_gpu_tensor *comp = ds4_gpu_tensor_alloc(comp_bytes);
    ds4_gpu_tensor *q = ds4_gpu_tensor_alloc(q_bytes);
    ds4_gpu_tensor *comp_mask = ds4_gpu_tensor_alloc(mask_bytes);
    ds4_gpu_tensor *heads_base = ds4_gpu_tensor_alloc(heads_base_bytes);
    float *raw_host = malloc((size_t)raw_bytes);
    uint16_t *comp_host = malloc((size_t)comp_bytes);
    float *q_host = malloc((size_t)q_bytes);
    float *mask_host = malloc((size_t)mask_bytes);
    uint8_t *initial = malloc((size_t)heads_base_bytes);
    uint8_t *snapshots = malloc(
        (size_t)(TEST_SMALL_PREFILL_ARM_COUNT * heads_base_bytes));
    /* Metal retains a no-copy model view after this oracle returns.  Static
     * page-aligned storage keeps that backing valid until backend cleanup. */
    static uint8_t model_storage[16384]
        __attribute__((aligned(16384)));
    void *model_raw = model_storage;
    const bool model_storage_ok = page <= sizeof(model_storage);
    TEST_ASSERT(model_storage_ok);
    TEST_ASSERT(raw != NULL);
    TEST_ASSERT(comp != NULL);
    TEST_ASSERT(q != NULL);
    TEST_ASSERT(comp_mask != NULL);
    TEST_ASSERT(heads_base != NULL);
    TEST_ASSERT(raw_host != NULL);
    TEST_ASSERT(comp_host != NULL);
    TEST_ASSERT(q_host != NULL);
    TEST_ASSERT(mask_host != NULL);
    TEST_ASSERT(initial != NULL);
    TEST_ASSERT(snapshots != NULL);

    size_t direct_mismatches = 0u;
    size_t replay_mismatches = 0u;
    size_t fallback_mismatches = 0u;
    size_t alias_mismatches = 0u;
    size_t alias_guard_mismatches = 0u;
    size_t guard_mismatches = 0u;
    size_t nonfinite = 0u;
    size_t total_words = 0u;
    size_t completed_cases = 0u;
    const bool allocated = raw && comp && q && comp_mask && heads_base &&
        raw_host && comp_host && q_host && mask_host && initial &&
        snapshots && model_storage_ok;
    if (allocated) {
        memset(model_raw, 0, (size_t)page);
        ((float *)model_raw)[0] = -0.375f;
        ((float *)model_raw)[1] = 0.21875f;
        for (uint64_t i = 0; i < raw_count; i++) {
            const int value = (int)((i * 17u +
                (i ^ (i >> 5u)) * 11u + 23u) % 211u) - 105;
            raw_host[i] = (float)value / 192.0f;
        }
        for (uint64_t i = 0; i < comp_count; i++) {
            const int value = (int)((i * 23u +
                (i ^ (i >> 4u)) * 7u + 29u) % 193u) - 96;
            comp_host[i] = test_float_to_f16(
                0.125f + (float)value / 224.0f);
        }
        for (uint64_t i = 0; i < q_count; i++) {
            const int value = (int)((i * 31u +
                (i ^ (i >> 3u)) * 5u + 37u) % 227u) - 113;
            q_host[i] = (float)value / 208.0f;
        }
        for (uint64_t i = 0; i < mask_count; i++) {
            mask_host[i] = (i % 5u) == 0u
                ? -65504.0f : -(float)((i % 13u) + 1u) / 16.0f;
        }
        for (uint64_t i = 0; i < heads_base_bytes; i++) {
            initial[i] = (uint8_t)(0xa5u ^ (uint8_t)(i * 37u));
        }
        const uint32_t poison = 0x7fc12345u;
        for (uint64_t i = 0; i < q_count; i++) {
            memcpy(initial + guard_bytes + i * sizeof(poison),
                   &poison, sizeof(poison));
        }

        TEST_ASSERT(ds4_gpu_tensor_write(
            raw, 0u, raw_host, raw_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
            comp, 0u, comp_host, comp_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
            q, 0u, q_host, q_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
            comp_mask, 0u, mask_host, mask_bytes) != 0);
        TEST_ASSERT(ds4_gpu_set_model_map(model_raw, page) != 0);
        ds4_gpu_set_quality(false);
        TEST_ASSERT(unsetenv(disable_env) == 0);
        TEST_ASSERT(unsetenv(stage_profile_env) == 0);

        for (size_t case_i = 0;
             case_i < sizeof(cases) / sizeof(cases[0]); case_i++) {
            const test_metal_small_prefill_case *c = &cases[case_i];
            const uint64_t output_count =
                (uint64_t)c->shape.n_tokens * n_head * head_dim;
            const uint64_t output_bytes = output_count * sizeof(float);
            ds4_gpu_tensor *heads = ds4_gpu_tensor_view(
                heads_base, guard_bytes, output_bytes);
            TEST_ASSERT(heads != NULL);
            bool case_ok = heads != NULL;
            for (uint32_t arm = 0u;
                 arm < TEST_SMALL_PREFILL_ARM_COUNT && case_ok; arm++) {
                const bool cold_scratch_case =
                    case_i == 0u || case_i == 3u;
                if (cold_scratch_case &&
                    (arm == TEST_SMALL_PREFILL_DIRECT_COLD ||
                     arm == TEST_SMALL_PREFILL_PSO_FALLBACK)) {
                    TEST_ASSERT(ds4_gpu_test_reset_flash_attn_tmp() != 0);
                    TEST_ASSERT(ds4_gpu_test_flash_attn_tmp_bytes() == 0u);
                }
                TEST_ASSERT(ds4_gpu_tensor_write(
                    heads_base, 0u, initial, heads_base_bytes) != 0);
                const bool env_rollback =
                    arm == TEST_SMALL_PREFILL_ENV_ROLLBACK;
                const int env_ok = env_rollback
                    ? setenv(disable_env, "0", 1)
                    : unsetenv(disable_env);
                TEST_ASSERT(env_ok == 0);
                ds4_gpu_test_set_flags(
                    arm == TEST_SMALL_PREFILL_PSO_FALLBACK
                        ? DS4_GPU_TEST_FLASH_ATTN_SMALL_PREFILL_NWG1_FAILURE
                        : 0u);
                const int call_ok =
                    test_metal_zero_prefix_prefill_mask_cache_call(
                        c->kind, heads, model_raw, page, q, raw, comp,
                        comp_mask, &c->shape, c->masked,
                        n_head, head_dim);
                TEST_ASSERT(call_ok != 0);
                const uint32_t n_keys =
                    c->shape.n_tokens + c->shape.n_comp;
                const bool expect_direct =
                    (arm == TEST_SMALL_PREFILL_DIRECT_COLD ||
                     arm == TEST_SMALL_PREFILL_DIRECT_REPLAY) &&
                    n_keys <= 32u;
                const uint32_t expected_nwg = expect_direct ? 1u : 32u;
                const uint32_t selected_nwg =
                    ds4_gpu_test_last_flash_attn_prefill_nwg();
                TEST_ASSERT(selected_nwg == expected_nwg);
                if (cold_scratch_case &&
                    arm == TEST_SMALL_PREFILL_DIRECT_COLD) {
                    TEST_ASSERT(
                        ds4_gpu_test_flash_attn_tmp_bytes() == 0u);
                }
                if (cold_scratch_case &&
                    arm == TEST_SMALL_PREFILL_PSO_FALLBACK) {
                    TEST_ASSERT(
                        ds4_gpu_test_flash_attn_tmp_bytes() != 0u);
                }
                const int read_ok = call_ok && ds4_gpu_tensor_read(
                    heads_base, 0u,
                    snapshots + (uint64_t)arm * heads_base_bytes,
                    heads_base_bytes) != 0;
                TEST_ASSERT(read_ok != 0);
                case_ok = read_ok != 0 && selected_nwg == expected_nwg;
            }
            ds4_gpu_test_set_flags(0u);
            ds4_gpu_tensor_free(heads);
            if (!case_ok) continue;

            const uint8_t *direct = snapshots +
                TEST_SMALL_PREFILL_DIRECT_COLD * heads_base_bytes;
            const uint8_t *baseline = snapshots +
                TEST_SMALL_PREFILL_ENV_ROLLBACK * heads_base_bytes;
            const uint8_t *replay = snapshots +
                TEST_SMALL_PREFILL_DIRECT_REPLAY * heads_base_bytes;
            const uint8_t *fallback = snapshots +
                TEST_SMALL_PREFILL_PSO_FALLBACK * heads_base_bytes;
            const test_float_compare_stats direct_stats =
                test_compare_float_bits(
                    (const float *)(baseline + guard_bytes),
                    (const float *)(direct + guard_bytes),
                    (size_t)output_count);
            const test_float_compare_stats replay_stats =
                test_compare_float_bits(
                    (const float *)(baseline + guard_bytes),
                    (const float *)(replay + guard_bytes),
                    (size_t)output_count);
            const test_float_compare_stats fallback_stats =
                test_compare_float_bits(
                    (const float *)(baseline + guard_bytes),
                    (const float *)(fallback + guard_bytes),
                    (size_t)output_count);
            direct_mismatches += direct_stats.mismatch_count;
            replay_mismatches += replay_stats.mismatch_count;
            fallback_mismatches += fallback_stats.mismatch_count;
            total_words += (size_t)output_count;
            completed_cases++;

            for (uint32_t arm = 0u;
                 arm < TEST_SMALL_PREFILL_ARM_COUNT; arm++) {
                const uint8_t *snapshot =
                    snapshots + (uint64_t)arm * heads_base_bytes;
                for (uint64_t i = 0; i < heads_base_bytes; i++) {
                    const bool in_output = i >= guard_bytes &&
                        i < guard_bytes + output_bytes;
                    if (!in_output && snapshot[i] != initial[i]) {
                        guard_mismatches++;
                    }
                }
                const float *values =
                    (const float *)(snapshot + guard_bytes);
                for (uint64_t i = 0; i < output_count; i++) {
                    if (!isfinite(values[i])) nonfinite++;
                }
            }
            fprintf(stderr,
                    "ds4-test: small-prefill direct %s keys=%u "
                    "direct=%zu/%llu replay=%zu/%llu fallback=%zu/%llu\n",
                    c->name,
                    c->shape.n_tokens + c->shape.n_comp,
                    direct_stats.mismatch_count,
                    (unsigned long long)output_count,
                    replay_stats.mismatch_count,
                    (unsigned long long)output_count,
                    fallback_stats.mismatch_count,
                    (unsigned long long)output_count);
        }

        /* The legacy split path permits heads to alias q because its first
         * dispatch finishes reading q before the reducer writes heads.  The
         * direct kernel must preserve that API behavior by falling back. */
        {
            static const size_t alias_case_indices[] = {1u, 3u, 5u};
            for (size_t alias_i = 0u;
                 alias_i < sizeof(alias_case_indices) /
                     sizeof(alias_case_indices[0]);
                 alias_i++) {
                const test_metal_small_prefill_case *c =
                    &cases[alias_case_indices[alias_i]];
                const uint64_t output_count =
                    (uint64_t)c->shape.n_tokens * n_head * head_dim;
                const uint64_t output_bytes = output_count * sizeof(float);
                ds4_gpu_tensor *heads_ref = ds4_gpu_tensor_view(
                    heads_base, guard_bytes, output_bytes);
                ds4_gpu_tensor *q_alias = ds4_gpu_tensor_view(
                    heads_base, guard_bytes, q_bytes);
                ds4_gpu_tensor *heads_alias = ds4_gpu_tensor_view(
                    heads_base, guard_bytes, output_bytes);
                TEST_ASSERT(heads_ref != NULL);
                TEST_ASSERT(q_alias != NULL);
                TEST_ASSERT(heads_alias != NULL);
                if (heads_ref && q_alias && heads_alias) {
                    TEST_ASSERT(ds4_gpu_tensor_write(
                        q, 0u, q_host, q_bytes) != 0);
                    TEST_ASSERT(ds4_gpu_tensor_write(
                        heads_base, 0u, initial, heads_base_bytes) != 0);
                    TEST_ASSERT(setenv(disable_env, "0", 1) == 0);
                    ds4_gpu_test_set_flags(0u);
                    const int ref_ok =
                        test_metal_zero_prefix_prefill_mask_cache_call(
                            c->kind, heads_ref, model_raw, page, q, raw,
                            comp, comp_mask, &c->shape, c->masked,
                            n_head, head_dim);
                    TEST_ASSERT(ref_ok != 0);
                    TEST_ASSERT(
                        ds4_gpu_test_last_flash_attn_prefill_nwg() == 32u);
                    const int ref_read_ok = ref_ok && ds4_gpu_tensor_read(
                        heads_base, 0u, snapshots, heads_base_bytes) != 0;
                    TEST_ASSERT(ref_read_ok != 0);

                    TEST_ASSERT(ds4_gpu_tensor_write(
                        heads_base, 0u, initial, heads_base_bytes) != 0);
                    TEST_ASSERT(ds4_gpu_tensor_write(
                        q_alias, 0u, q_host, q_bytes) != 0);
                    const int before_read_ok = ds4_gpu_tensor_read(
                        heads_base, 0u, snapshots + heads_base_bytes,
                        heads_base_bytes) != 0;
                    TEST_ASSERT(before_read_ok != 0);
                    TEST_ASSERT(unsetenv(disable_env) == 0);
                    ds4_gpu_test_set_flags(0u);
                    const int alias_ok =
                        test_metal_zero_prefix_prefill_mask_cache_call(
                            c->kind, heads_alias, model_raw, page, q_alias,
                            raw, comp, comp_mask, &c->shape, c->masked,
                            n_head, head_dim);
                    TEST_ASSERT(alias_ok != 0);
                    const uint32_t alias_nwg =
                        ds4_gpu_test_last_flash_attn_prefill_nwg();
                    TEST_ASSERT(alias_nwg == 32u);
                    const int alias_read_ok = alias_ok &&
                        ds4_gpu_tensor_read(
                            heads_base, 0u,
                            snapshots + 2u * heads_base_bytes,
                            heads_base_bytes) != 0;
                    TEST_ASSERT(alias_read_ok != 0);
                    if (ref_read_ok && before_read_ok && alias_read_ok) {
                        const test_float_compare_stats alias_stats =
                            test_compare_float_bits(
                                (const float *)(snapshots + guard_bytes),
                                (const float *)(snapshots +
                                    2u * heads_base_bytes + guard_bytes),
                                (size_t)output_count);
                        alias_mismatches += alias_stats.mismatch_count;
                        size_t case_guard_mismatches = 0u;
                        const uint8_t *before =
                            snapshots + heads_base_bytes;
                        const uint8_t *after =
                            snapshots + 2u * heads_base_bytes;
                        for (uint64_t i = 0u;
                             i < heads_base_bytes; i++) {
                            const bool in_output = i >= guard_bytes &&
                                i < guard_bytes + output_bytes;
                            if (!in_output && before[i] != after[i]) {
                                case_guard_mismatches++;
                            }
                        }
                        alias_guard_mismatches += case_guard_mismatches;
                        fprintf(stderr,
                                "ds4-test: small-prefill q/heads alias "
                                "%s nwg=%u mismatches=%zu/%llu guards=%zu\n",
                                c->name, alias_nwg,
                                alias_stats.mismatch_count,
                                (unsigned long long)output_count,
                                case_guard_mismatches);
                    }
                }
                ds4_gpu_test_set_flags(0u);
                TEST_ASSERT(unsetenv(disable_env) == 0);
                ds4_gpu_tensor_free(heads_alias);
                ds4_gpu_tensor_free(q_alias);
                ds4_gpu_tensor_free(heads_ref);
            }
        }

        if (test_env_bool("DS4_TEST_METAL_SMALL_PREFILL_TIMING")) {
            /* GPUStartTime/GPUEndTime excludes CPU mask construction,
             * encoding, submit, and wait time.  Both arms run the same
             * copy/pad kernels; their only topology difference is NWG=32 +
             * reduce versus direct NWG=1 output.  Alternate pair order to
             * limit thermal/order bias.  Keep this opt-in so the exactness
             * oracle remains suitable for routine test runs. */
            static const size_t timing_case_indices[] = {2u, 4u};
            const uint32_t timing_repeats = 16u;
            const uint32_t timing_samples = 4u;
            for (size_t timing_i = 0u;
                 timing_i < sizeof(timing_case_indices) /
                     sizeof(timing_case_indices[0]);
                 timing_i++) {
                const test_metal_small_prefill_case *c =
                    &cases[timing_case_indices[timing_i]];
                const uint64_t output_bytes =
                    (uint64_t)c->shape.n_tokens * n_head * head_dim *
                    sizeof(float);
                ds4_gpu_tensor *heads = ds4_gpu_tensor_view(
                    heads_base, guard_bytes, output_bytes);
                TEST_ASSERT(heads != NULL);
                if (!heads) continue;

                (void)test_metal_small_prefill_gpu_batch_ms(
                    c->kind, c->masked, &c->shape, heads, model_raw, page,
                    q, raw, comp, comp_mask, n_head, head_dim,
                    DS4_GPU_TEST_FLASH_ATTN_SMALL_PREFILL_NWG32, 4u, 32u);
                (void)test_metal_small_prefill_gpu_batch_ms(
                    c->kind, c->masked, &c->shape, heads, model_raw, page,
                    q, raw, comp, comp_mask, n_head, head_dim,
                    0u, 4u, 1u);

                double baseline_sum = 0.0;
                double direct_sum = 0.0;
                double log_speedup_sum = 0.0;
                uint32_t valid_samples = 0u;
                for (uint32_t sample = 0u;
                     sample < timing_samples; sample++) {
                    double pair_ms[2] = {-1.0, -1.0};
                    for (uint32_t order_i = 0u; order_i < 2u; order_i++) {
                        const uint32_t arm = (sample & 1u) != 0u
                            ? 1u - order_i : order_i;
                        pair_ms[arm] =
                            test_metal_small_prefill_gpu_batch_ms(
                                c->kind, c->masked, &c->shape,
                                heads, model_raw, page, q, raw, comp,
                                comp_mask, n_head, head_dim,
                                arm == 0u
                                    ? DS4_GPU_TEST_FLASH_ATTN_SMALL_PREFILL_NWG32
                                    : 0u,
                                timing_repeats,
                                arm == 0u ? 32u : 1u);
                    }
                    if (pair_ms[0] > 0.0 && pair_ms[1] > 0.0) {
                        baseline_sum += pair_ms[0];
                        direct_sum += pair_ms[1];
                        log_speedup_sum += log(pair_ms[0] / pair_ms[1]);
                        valid_samples++;
                    }
                }
                TEST_ASSERT(valid_samples == timing_samples);
                if (valid_samples != 0u) {
                    const double baseline_ms =
                        baseline_sum / (double)valid_samples;
                    const double direct_ms =
                        direct_sum / (double)valid_samples;
                    const double speedup =
                        exp(log_speedup_sum / (double)valid_samples);
                    fprintf(stderr,
                            "ds4-test: small-prefill GPU-only kernel-chain %s "
                            "legacy=%.6f ms direct=%.6f ms speedup=%.3fx "
                            "throughput=%.1f%%\n",
                            c->name, baseline_ms, direct_ms, speedup,
                            (speedup - 1.0) * 100.0);
                }
                ds4_gpu_tensor_free(heads);
            }
        }
    }

    ds4_gpu_test_set_flags(0u);
    test_restore_env(stage_profile_env, saved_stage_profile);
    test_restore_env(disable_env, saved_disable);
    fprintf(stderr,
            "ds4-test: small-prefill direct exact cases=%zu/%zu "
            "words=%zu direct=%zu replay=%zu fallback=%zu alias=%zu "
            "guards=%zu alias_guards=%zu nonfinite=%zu\n",
            completed_cases, sizeof(cases) / sizeof(cases[0]), total_words,
            direct_mismatches, replay_mismatches, fallback_mismatches,
            alias_mismatches, guard_mismatches, alias_guard_mismatches,
            nonfinite);
    TEST_ASSERT(completed_cases == sizeof(cases) / sizeof(cases[0]));
    TEST_ASSERT(direct_mismatches == 0u);
    TEST_ASSERT(replay_mismatches == 0u);
    TEST_ASSERT(fallback_mismatches == 0u);
    TEST_ASSERT(alias_mismatches == 0u);
    TEST_ASSERT(guard_mismatches == 0u);
    TEST_ASSERT(alias_guard_mismatches == 0u);
    TEST_ASSERT(nonfinite == 0u);

    free(snapshots);
    free(initial);
    free(mask_host);
    free(q_host);
    free(comp_host);
    free(raw_host);
    ds4_gpu_tensor_free(heads_base);
    ds4_gpu_tensor_free(comp_mask);
    ds4_gpu_tensor_free(q);
    ds4_gpu_tensor_free(comp);
    ds4_gpu_tensor_free(raw);
}
#endif

#if defined(__APPLE__)
typedef enum {
    TEST_METAL_ATTN_OUT_HC_Q8,
    TEST_METAL_ATTN_OUT_HC_Q4,
} test_metal_attn_out_hc_kind;

static uint32_t test_metal_attn_out_hc_poison_bits(
        uint32_t tag,
        uint64_t index) {
    return 0x7fc00000u |
        ((tag & 0x1fu) << 16u) |
        (uint32_t)((index + 1u) & 0xffffu);
}

static void test_metal_attn_out_hc_fill_poison(
        float    *dst,
        uint64_t  count,
        uint32_t  tag) {
    for (uint64_t i = 0; i < count; i++) {
        const uint32_t bits = test_metal_attn_out_hc_poison_bits(tag, i);
        memcpy(dst + i, &bits, sizeof(bits));
    }
}

static size_t test_metal_attn_out_hc_canary_mismatches(
        const float *values,
        uint64_t     begin,
        uint64_t     end,
        uint32_t     tag) {
    size_t mismatches = 0u;
    for (uint64_t i = begin; i < end; i++) {
        uint32_t bits = 0u;
        memcpy(&bits, values + i, sizeof(bits));
        if (bits != test_metal_attn_out_hc_poison_bits(tag, i)) {
            mismatches++;
        }
    }
    return mismatches;
}

static size_t test_metal_attn_out_hc_poisoned_values(
        const float *values,
        uint64_t     count,
        uint32_t     tag) {
    size_t poisoned = 0u;
    for (uint64_t i = 0; i < count; i++) {
        uint32_t bits = 0u;
        memcpy(&bits, values + i, sizeof(bits));
        if (bits == test_metal_attn_out_hc_poison_bits(tag, i)) {
            poisoned++;
        }
    }
    return poisoned;
}

static int test_metal_batch_attn_out_hc_fused_call(
        test_metal_attn_out_hc_kind kind,
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *out_hc,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        ds4_gpu_tensor       *low,
        ds4_gpu_tensor       *group_tmp,
        ds4_gpu_tensor       *low_tmp,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              out_b_offset,
        const ds4_gpu_tensor *heads,
        uint32_t              n_tokens) {
    const uint32_t group_dim = 4096u;
    const uint32_t rank = 1024u;
    const uint32_t n_groups = 8u;
    const uint32_t out_dim = 4096u;
    const uint32_t n_hc = 4u;
    if (kind == TEST_METAL_ATTN_OUT_HC_Q4) {
        return ds4_gpu_attention_output_q4_K_batch_hc_tensor(
            out, out_hc, residual_hc, split, low, group_tmp, low_tmp,
            model_map, model_size, 0u, out_b_offset, 12u,
            group_dim, rank, n_groups, out_dim, heads, n_tokens, n_hc);
    }
    return ds4_gpu_attention_output_q8_batch_hc_tensor(
        out, out_hc, residual_hc, split, low, group_tmp, low_tmp,
        model_map, model_size, 0u, out_b_offset,
        group_dim, rank, n_groups, out_dim, heads, n_tokens, n_hc);
}

static void test_metal_batch_attn_out_hc_fusion_exact_case(
        test_metal_attn_out_hc_kind kind) {
    const bool q4 = kind == TEST_METAL_ATTN_OUT_HC_Q4;
    const uint32_t n_tokens = 32u;
    const uint32_t alloc_tokens = n_tokens + 1u;
    const uint32_t group_dim = 4096u;
    const uint32_t rank = 1024u;
    const uint32_t n_groups = 8u;
    const uint32_t low_dim = n_groups * rank;
    const uint32_t out_dim = 4096u;
    const uint32_t n_hc = 4u;
    const uint64_t mix_hc = 2ull * n_hc + (uint64_t)n_hc * n_hc;
    const uint64_t page = (uint64_t)getpagesize();
    const uint64_t row_a_bytes = q4
        ? (uint64_t)(group_dim / 256u) * 144u
        : (uint64_t)(group_dim / 32u) * 34u;
    const uint64_t row_b_bytes = q4
        ? (uint64_t)(low_dim / 256u) * 144u
        : (uint64_t)(low_dim / 32u) * 34u;
    const uint64_t out_a_bytes = (uint64_t)low_dim * row_a_bytes;
    const uint64_t out_b_offset = test_round_up_u64(out_a_bytes, page);
    const uint64_t out_b_bytes = (uint64_t)out_dim * row_b_bytes;
    const uint64_t model_bytes =
        test_round_up_u64(out_b_offset + out_b_bytes, page);
    const uint64_t heads_count =
        (uint64_t)alloc_tokens * n_groups * group_dim;
    const uint64_t low_count = (uint64_t)alloc_tokens * low_dim;
    const uint64_t out_count = (uint64_t)alloc_tokens * out_dim;
    const uint64_t hc_count =
        (uint64_t)alloc_tokens * n_hc * out_dim;
    const uint64_t split_count = (uint64_t)alloc_tokens * mix_hc;
    const uint64_t active_heads_count =
        (uint64_t)n_tokens * n_groups * group_dim;
    const uint64_t active_low_count = (uint64_t)n_tokens * low_dim;
    const uint64_t active_out_count = (uint64_t)n_tokens * out_dim;
    const uint64_t active_hc_count =
        (uint64_t)n_tokens * n_hc * out_dim;
    const uint64_t active_split_count = (uint64_t)n_tokens * mix_hc;
    const uint64_t heads_bytes = heads_count * sizeof(float);
    const uint64_t low_bytes = low_count * sizeof(float);
    const uint64_t out_bytes = out_count * sizeof(float);
    const uint64_t hc_bytes = hc_count * sizeof(float);
    const uint64_t split_bytes = split_count * sizeof(float);
    const uint64_t active_heads_bytes = active_heads_count * sizeof(float);
    const uint64_t active_low_bytes = active_low_count * sizeof(float);
    const uint64_t active_out_bytes = active_out_count * sizeof(float);
    const uint64_t active_hc_bytes = active_hc_count * sizeof(float);
    const uint64_t active_split_bytes = active_split_count * sizeof(float);
    const uint64_t scratch_bytes =
        (uint64_t)alloc_tokens * low_dim * sizeof(uint16_t);
    enum {
        ref_low_tag = 1u,
        fused_low_tag = 2u,
        ref_out_tag = 3u,
        fused_out_tag = 4u,
        ref_hc_tag = 5u,
        fused_hc_tag = 6u,
    };

    const char *disable_fusion_env =
        "DS4_METAL_DISABLE_PRE_M5_BATCH_ATTN_OUT_HC_FUSION";
    const char *require_q8_env =
        "DS4_METAL_REQUIRE_PRE_M5_BATCH_ATTN_OUT_HC_FUSION";
    const char *require_q4_env =
        "DS4_METAL_REQUIRE_Q4_BATCH_ATTN_OUT_HC_FUSION";
    const char *require_q4_rhs_env =
        "DS4_METAL_REQUIRE_Q4_ATTN_OUT_B_F16_RHS";
    const char *disable_q4_rhs_env =
        "DS4_METAL_DISABLE_Q4_ATTN_OUT_B_F16_RHS";
    const char *stage_profile_env = "DS4_METAL_ATTN_OUT_STAGE_PROFILE";
    const char *q8_profile_env = "DS4_METAL_Q8_PREFILL_PROFILE";
    char *saved_disable_fusion = test_save_env(disable_fusion_env);
    char *saved_require_q8 = test_save_env(require_q8_env);
    char *saved_require_q4 = test_save_env(require_q4_env);
    char *saved_require_q4_rhs = test_save_env(require_q4_rhs_env);
    char *saved_disable_q4_rhs = test_save_env(disable_q4_rhs_env);
    char *saved_stage_profile = test_save_env(stage_profile_env);
    char *saved_q8_profile = test_save_env(q8_profile_env);

    void *model_raw = NULL;
    float *heads_host = NULL;
    float *residual_host = NULL;
    float *split_host = NULL;
    float *ref_low_host = NULL;
    float *fused_low_host = NULL;
    float *ref_out_host = NULL;
    float *fused_out_host = NULL;
    float *ref_hc_host = NULL;
    float *fused_hc_host = NULL;
    ds4_gpu_tensor *heads_base = NULL;
    ds4_gpu_tensor *residual_base = NULL;
    ds4_gpu_tensor *split_base = NULL;
    ds4_gpu_tensor *ref_low_base = NULL;
    ds4_gpu_tensor *fused_low_base = NULL;
    ds4_gpu_tensor *ref_out_base = NULL;
    ds4_gpu_tensor *fused_out_base = NULL;
    ds4_gpu_tensor *ref_hc_base = NULL;
    ds4_gpu_tensor *fused_hc_base = NULL;
    ds4_gpu_tensor *group_tmp = NULL;
    ds4_gpu_tensor *low_tmp = NULL;
    ds4_gpu_tensor *heads = NULL;
    ds4_gpu_tensor *residual = NULL;
    ds4_gpu_tensor *split = NULL;
    ds4_gpu_tensor *ref_low = NULL;
    ds4_gpu_tensor *fused_low = NULL;
    ds4_gpu_tensor *ref_out = NULL;
    ds4_gpu_tensor *fused_out = NULL;
    ds4_gpu_tensor *ref_hc = NULL;
    ds4_gpu_tensor *fused_hc = NULL;

    TEST_ASSERT(posix_memalign(
                    &model_raw, (size_t)page, (size_t)model_bytes) == 0);
    if (model_raw) {
        memset(model_raw, 0, (size_t)model_bytes);
        if (q4) {
            test_fill_q4_K_weights(
                model_raw, group_dim, low_dim, 211u);
            test_fill_q4_K_weights(
                (uint8_t *)model_raw + out_b_offset,
                low_dim, out_dim, 307u);
        } else {
            test_fill_q8_0_weights(
                model_raw, group_dim, low_dim, 211u);
            test_fill_q8_0_weights(
                (uint8_t *)model_raw + out_b_offset,
                low_dim, out_dim, 307u);
        }
    }

    heads_host = malloc((size_t)heads_bytes);
    residual_host = malloc((size_t)hc_bytes);
    split_host = malloc((size_t)split_bytes);
    ref_low_host = malloc((size_t)low_bytes);
    fused_low_host = malloc((size_t)low_bytes);
    ref_out_host = malloc((size_t)out_bytes);
    fused_out_host = malloc((size_t)out_bytes);
    ref_hc_host = malloc((size_t)hc_bytes);
    fused_hc_host = malloc((size_t)hc_bytes);
    heads_base = ds4_gpu_tensor_alloc(heads_bytes);
    residual_base = ds4_gpu_tensor_alloc(hc_bytes);
    split_base = ds4_gpu_tensor_alloc(split_bytes);
    ref_low_base = ds4_gpu_tensor_alloc(low_bytes);
    fused_low_base = ds4_gpu_tensor_alloc(low_bytes);
    ref_out_base = ds4_gpu_tensor_alloc(out_bytes);
    fused_out_base = ds4_gpu_tensor_alloc(out_bytes);
    ref_hc_base = ds4_gpu_tensor_alloc(hc_bytes);
    fused_hc_base = ds4_gpu_tensor_alloc(hc_bytes);
    group_tmp = ds4_gpu_tensor_alloc(scratch_bytes);
    low_tmp = ds4_gpu_tensor_alloc(sizeof(float));
    heads = heads_base
        ? ds4_gpu_tensor_view(heads_base, 0u, active_heads_bytes) : NULL;
    residual = residual_base
        ? ds4_gpu_tensor_view(residual_base, 0u, active_hc_bytes) : NULL;
    split = split_base
        ? ds4_gpu_tensor_view(split_base, 0u, active_split_bytes) : NULL;
    ref_low = ref_low_base
        ? ds4_gpu_tensor_view(ref_low_base, 0u, active_low_bytes) : NULL;
    fused_low = fused_low_base
        ? ds4_gpu_tensor_view(fused_low_base, 0u, active_low_bytes) : NULL;
    ref_out = ref_out_base
        ? ds4_gpu_tensor_view(ref_out_base, 0u, active_out_bytes) : NULL;
    fused_out = fused_out_base
        ? ds4_gpu_tensor_view(fused_out_base, 0u, active_out_bytes) : NULL;
    ref_hc = ref_hc_base
        ? ds4_gpu_tensor_view(ref_hc_base, 0u, active_hc_bytes) : NULL;
    fused_hc = fused_hc_base
        ? ds4_gpu_tensor_view(fused_hc_base, 0u, active_hc_bytes) : NULL;

    const bool allocated = model_raw && heads_host && residual_host &&
        split_host && ref_low_host && fused_low_host && ref_out_host &&
        fused_out_host && ref_hc_host && fused_hc_host && heads_base &&
        residual_base && split_base && ref_low_base && fused_low_base &&
        ref_out_base && fused_out_base && ref_hc_base && fused_hc_base &&
        group_tmp && low_tmp && heads && residual && split && ref_low &&
        fused_low && ref_out && fused_out && ref_hc && fused_hc;
    TEST_ASSERT(allocated);

    test_float_compare_stats low_stats = {0};
    test_float_compare_stats hc_stats = {0};
    size_t reject_writes = 0u;
    size_t active_poisoned = 0u;
    size_t guard_mismatches = 0u;
    size_t fused_out_writes = 0u;
    if (allocated) {
        for (uint64_t i = 0; i < heads_count; i++) {
            const int value =
                (int)((i * 17u + (i ^ (i >> 7u)) * 3u +
                       (q4 ? 19u : 7u)) % 127u) - 63;
            heads_host[i] = (float)value / 96.0f;
        }
        for (uint64_t i = 0; i < hc_count; i++) {
            const int value =
                (int)((i * 19u + (i ^ (i >> 5u)) * 7u +
                       (q4 ? 23u : 11u)) % 149u) - 74;
            residual_host[i] = (float)value / 80.0f;
        }
        for (uint32_t t = 0; t < alloc_tokens; t++) {
            float *row = split_host + (uint64_t)t * mix_hc;
            for (uint32_t h = 0; h < n_hc; h++) {
                row[h] = 0.0f;
                row[n_hc + h] =
                    0.55f + (float)((t + h * 3u) % 11u) / 32.0f;
            }
            for (uint32_t dst_hc = 0; dst_hc < n_hc; dst_hc++) {
                for (uint32_t src_hc = 0; src_hc < n_hc; src_hc++) {
                    const int value =
                        (int)((t * 5u + dst_hc * 7u + src_hc * 11u +
                               (q4 ? 3u : 0u)) % 17u) - 8;
                    row[2u * n_hc + dst_hc * n_hc + src_hc] =
                        (float)value / 24.0f;
                }
            }
        }
        test_metal_attn_out_hc_fill_poison(
            ref_low_host, low_count, ref_low_tag);
        test_metal_attn_out_hc_fill_poison(
            fused_low_host, low_count, fused_low_tag);
        test_metal_attn_out_hc_fill_poison(
            ref_out_host, out_count, ref_out_tag);
        test_metal_attn_out_hc_fill_poison(
            fused_out_host, out_count, fused_out_tag);
        test_metal_attn_out_hc_fill_poison(
            ref_hc_host, hc_count, ref_hc_tag);
        test_metal_attn_out_hc_fill_poison(
            fused_hc_host, hc_count, fused_hc_tag);

        TEST_ASSERT(ds4_gpu_tensor_write(
                        heads_base, 0u, heads_host, heads_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        residual_base, 0u, residual_host, hc_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        split_base, 0u, split_host, split_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        ref_low_base, 0u, ref_low_host, low_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        fused_low_base, 0u, fused_low_host, low_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        ref_out_base, 0u, ref_out_host, out_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        fused_out_base, 0u, fused_out_host, out_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        ref_hc_base, 0u, ref_hc_host, hc_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        fused_hc_base, 0u, fused_hc_host, hc_bytes) != 0);
        TEST_ASSERT(ds4_gpu_set_model_map(model_raw, model_bytes) != 0);

        TEST_ASSERT(unsetenv(disable_fusion_env) == 0);
        TEST_ASSERT(unsetenv(require_q8_env) == 0);
        TEST_ASSERT(unsetenv(require_q4_env) == 0);
        TEST_ASSERT(unsetenv(require_q4_rhs_env) == 0);
        TEST_ASSERT(unsetenv(disable_q4_rhs_env) == 0);
        TEST_ASSERT(unsetenv(stage_profile_env) == 0);
        TEST_ASSERT(unsetenv(q8_profile_env) == 0);
        TEST_ASSERT(setenv(
                        q4 ? require_q4_env : require_q8_env,
                        "1", 1) == 0);
        if (q4) {
            TEST_ASSERT(setenv(require_q4_rhs_env, "1", 1) == 0);
        }
        ds4_gpu_set_quality(false);
        ds4_gpu_test_set_flags(q4
            ? DS4_GPU_TEST_BATCH_ATTN_OUT_Q4_HC_FUSION
            : DS4_GPU_TEST_BATCH_ATTN_OUT_Q8_HC_FUSION);

        /* The force hook removes only the production minimum.  Neighbors of
         * the 32-row tile must still fail closed and leave every destination
         * byte untouched. */
        TEST_ASSERT(test_metal_batch_attn_out_hc_fused_call(
                        kind, fused_out_base, fused_hc_base, residual_base,
                        split_base, fused_low_base, group_tmp, low_tmp,
                        model_raw, model_bytes, out_b_offset, heads_base,
                        n_tokens - 1u) == -1);
        TEST_ASSERT(test_metal_batch_attn_out_hc_fused_call(
                        kind, fused_out_base, fused_hc_base, residual_base,
                        split_base, fused_low_base, group_tmp, low_tmp,
                        model_raw, model_bytes, out_b_offset, heads_base,
                        n_tokens + 1u) == -1);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        fused_low_base, 0u, fused_low_host, low_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        fused_out_base, 0u, fused_out_host, out_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        fused_hc_base, 0u, fused_hc_host, hc_bytes) != 0);
        reject_writes += test_metal_attn_out_hc_canary_mismatches(
            fused_low_host, 0u, low_count, fused_low_tag);
        reject_writes += test_metal_attn_out_hc_canary_mismatches(
            fused_out_host, 0u, out_count, fused_out_tag);
        reject_writes += test_metal_attn_out_hc_canary_mismatches(
            fused_hc_host, 0u, hc_count, fused_hc_tag);

        int reference_ok = 0;
        if (q4) {
            reference_ok = ds4_gpu_attention_output_q4_K_batch_tensor(
                ref_out, ref_low, group_tmp, low_tmp,
                model_raw, model_bytes, 0u, out_b_offset, 12u,
                group_dim, rank, n_groups, out_dim, heads, n_tokens);
        } else {
            reference_ok = ds4_gpu_attention_output_q8_batch_tensor(
                ref_out, ref_low, group_tmp, low_tmp,
                model_raw, model_bytes, 0u, out_b_offset,
                group_dim, rank, n_groups, out_dim, heads, n_tokens);
        }
        TEST_ASSERT(reference_ok == 1);
        TEST_ASSERT(ds4_gpu_hc_expand_split_tensor(
                        ref_hc, ref_out, residual, split,
                        out_dim, n_hc) != 0);
        TEST_ASSERT(test_metal_batch_attn_out_hc_fused_call(
                        kind, fused_out, fused_hc, residual, split,
                        fused_low, group_tmp, low_tmp,
                        model_raw, model_bytes, out_b_offset, heads,
                        n_tokens) == 1);

        TEST_ASSERT(ds4_gpu_tensor_read(
                        ref_low_base, 0u, ref_low_host, low_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        fused_low_base, 0u, fused_low_host, low_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        ref_out_base, 0u, ref_out_host, out_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        fused_out_base, 0u, fused_out_host, out_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        ref_hc_base, 0u, ref_hc_host, hc_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        fused_hc_base, 0u, fused_hc_host, hc_bytes) != 0);

        low_stats = test_compare_float_bits(
            ref_low_host, fused_low_host, (size_t)active_low_count);
        hc_stats = test_compare_float_bits(
            ref_hc_host, fused_hc_host, (size_t)active_hc_count);
        active_poisoned += test_metal_attn_out_hc_poisoned_values(
            ref_low_host, active_low_count, ref_low_tag);
        active_poisoned += test_metal_attn_out_hc_poisoned_values(
            fused_low_host, active_low_count, fused_low_tag);
        active_poisoned += test_metal_attn_out_hc_poisoned_values(
            ref_out_host, active_out_count, ref_out_tag);
        active_poisoned += test_metal_attn_out_hc_poisoned_values(
            ref_hc_host, active_hc_count, ref_hc_tag);
        active_poisoned += test_metal_attn_out_hc_poisoned_values(
            fused_hc_host, active_hc_count, fused_hc_tag);
        fused_out_writes = test_metal_attn_out_hc_canary_mismatches(
            fused_out_host, 0u, out_count, fused_out_tag);
        guard_mismatches += test_metal_attn_out_hc_canary_mismatches(
            ref_low_host, active_low_count, low_count, ref_low_tag);
        guard_mismatches += test_metal_attn_out_hc_canary_mismatches(
            fused_low_host, active_low_count, low_count, fused_low_tag);
        guard_mismatches += test_metal_attn_out_hc_canary_mismatches(
            ref_out_host, active_out_count, out_count, ref_out_tag);
        guard_mismatches += test_metal_attn_out_hc_canary_mismatches(
            ref_hc_host, active_hc_count, hc_count, ref_hc_tag);
        guard_mismatches += test_metal_attn_out_hc_canary_mismatches(
            fused_hc_host, active_hc_count, hc_count, fused_hc_tag);
    }

    fprintf(stderr,
            "ds4-test: Metal %s batch attention-output HC exact N=%u "
            "low=%zu/%llu hc=%zu/%llu max_ulp=%u/%u "
            "active_poison=%zu guard=%zu fused_out_writes=%zu "
            "reject_writes=%zu\n",
            q4 ? "Q4_K/F16-RHS" : "Q8_0",
            n_tokens,
            low_stats.mismatch_count,
            (unsigned long long)active_low_count,
            hc_stats.mismatch_count,
            (unsigned long long)active_hc_count,
            low_stats.max_ulp,
            hc_stats.max_ulp,
            active_poisoned,
            guard_mismatches,
            fused_out_writes,
            reject_writes);
    TEST_ASSERT(low_stats.mismatch_count == 0u && low_stats.max_ulp == 0u);
    TEST_ASSERT(hc_stats.mismatch_count == 0u && hc_stats.max_ulp == 0u);
    TEST_ASSERT(active_poisoned == 0u);
    TEST_ASSERT(guard_mismatches == 0u);
    TEST_ASSERT(fused_out_writes == 0u);
    TEST_ASSERT(reject_writes == 0u);

    ds4_gpu_test_set_flags(0u);
    ds4_gpu_set_quality(false);
    test_restore_env(q8_profile_env, saved_q8_profile);
    test_restore_env(stage_profile_env, saved_stage_profile);
    test_restore_env(disable_q4_rhs_env, saved_disable_q4_rhs);
    test_restore_env(require_q4_rhs_env, saved_require_q4_rhs);
    test_restore_env(require_q4_env, saved_require_q4);
    test_restore_env(require_q8_env, saved_require_q8);
    test_restore_env(disable_fusion_env, saved_disable_fusion);
    ds4_gpu_tensor_free(fused_hc);
    ds4_gpu_tensor_free(ref_hc);
    ds4_gpu_tensor_free(fused_out);
    ds4_gpu_tensor_free(ref_out);
    ds4_gpu_tensor_free(fused_low);
    ds4_gpu_tensor_free(ref_low);
    ds4_gpu_tensor_free(split);
    ds4_gpu_tensor_free(residual);
    ds4_gpu_tensor_free(heads);
    ds4_gpu_tensor_free(low_tmp);
    ds4_gpu_tensor_free(group_tmp);
    ds4_gpu_tensor_free(fused_hc_base);
    ds4_gpu_tensor_free(ref_hc_base);
    ds4_gpu_tensor_free(fused_out_base);
    ds4_gpu_tensor_free(ref_out_base);
    ds4_gpu_tensor_free(fused_low_base);
    ds4_gpu_tensor_free(ref_low_base);
    ds4_gpu_tensor_free(split_base);
    ds4_gpu_tensor_free(residual_base);
    ds4_gpu_tensor_free(heads_base);
    free(fused_hc_host);
    free(ref_hc_host);
    free(fused_out_host);
    free(ref_out_host);
    free(fused_low_host);
    free(ref_low_host);
    free(split_host);
    free(residual_host);
    free(heads_host);
    free(model_raw);
}

static void test_metal_batch_attn_out_hc_fusion_exact(void) {
    test_metal_batch_attn_out_hc_fusion_exact_case(
        TEST_METAL_ATTN_OUT_HC_Q8);
    test_metal_batch_attn_out_hc_fusion_exact_case(
        TEST_METAL_ATTN_OUT_HC_Q4);
}

static void test_metal_hc_split_weighted_sum_norm_batch_exact(void) {
    /* Compare the batched HC+RMSNorm fusion against the exact two-dispatch
     * sequence used by the reference path at DS4's production dimensions. */
    const uint32_t n_embd = 7168;
    const uint32_t n_hc = 4;
    const uint32_t n_rows = 3;
    const uint32_t sinkhorn_iters = 20;
    const float hc_eps = 1.0e-6f;
    const float norm_eps = 1.0e-6f;
    const uint64_t mix_hc = 2ull * n_hc + (uint64_t)n_hc * n_hc;
    const uint64_t page = (uint64_t)getpagesize();
    const uint64_t scale_offset = 0;
    const uint64_t base_offset = test_round_up_u64(3u * sizeof(float), page);
    const uint64_t norm_weight_offset =
        test_round_up_u64(base_offset + mix_hc * sizeof(float), page);
    const uint64_t model_alloc = test_round_up_u64(
        norm_weight_offset + (uint64_t)n_embd * sizeof(float), page);
    const uint64_t mix_bytes = (uint64_t)n_rows * mix_hc * sizeof(float);
    const uint64_t residual_bytes =
        (uint64_t)n_rows * n_hc * n_embd * sizeof(float);
    const uint64_t out_bytes = (uint64_t)n_rows * n_embd * sizeof(float);

    void *model_raw = NULL;
    TEST_ASSERT(posix_memalign(&model_raw, (size_t)page, (size_t)model_alloc) == 0);
    if (!model_raw) return;
    memset(model_raw, 0, (size_t)model_alloc);

    float *scale = (float *)((uint8_t *)model_raw + scale_offset);
    float *base = (float *)((uint8_t *)model_raw + base_offset);
    float *norm_weight = (float *)((uint8_t *)model_raw + norm_weight_offset);
    scale[0] = 0.625f;
    scale[1] = -0.75f;
    scale[2] = 0.4375f;
    for (uint32_t i = 0; i < mix_hc; i++) {
        const int value = (int)((i * 17u + 5u) % 29u) - 14;
        base[i] = (float)value / 16.0f;
    }
    for (uint32_t i = 0; i < n_embd; i++) {
        norm_weight[i] = 0.5f + (float)((i * 13u + 7u) % 31u) / 32.0f;
    }

    ds4_gpu_tensor *mix = ds4_gpu_tensor_alloc(mix_bytes);
    ds4_gpu_tensor *residual = ds4_gpu_tensor_alloc(residual_bytes);
    ds4_gpu_tensor *ref_split = ds4_gpu_tensor_alloc(mix_bytes);
    ds4_gpu_tensor *fused_split = ds4_gpu_tensor_alloc(mix_bytes);
    ds4_gpu_tensor *ref_out = ds4_gpu_tensor_alloc(out_bytes);
    ds4_gpu_tensor *fused_out = ds4_gpu_tensor_alloc(out_bytes);
    ds4_gpu_tensor *ref_norm = ds4_gpu_tensor_alloc(out_bytes);
    ds4_gpu_tensor *fused_norm = ds4_gpu_tensor_alloc(out_bytes);
    TEST_ASSERT(mix != NULL);
    TEST_ASSERT(residual != NULL);
    TEST_ASSERT(ref_split != NULL);
    TEST_ASSERT(fused_split != NULL);
    TEST_ASSERT(ref_out != NULL);
    TEST_ASSERT(fused_out != NULL);
    TEST_ASSERT(ref_norm != NULL);
    TEST_ASSERT(fused_norm != NULL);
    if (!mix || !residual || !ref_split || !fused_split ||
        !ref_out || !fused_out || !ref_norm || !fused_norm) {
        ds4_gpu_tensor_free(mix);
        ds4_gpu_tensor_free(residual);
        ds4_gpu_tensor_free(ref_split);
        ds4_gpu_tensor_free(fused_split);
        ds4_gpu_tensor_free(ref_out);
        ds4_gpu_tensor_free(fused_out);
        ds4_gpu_tensor_free(ref_norm);
        ds4_gpu_tensor_free(fused_norm);
        free(model_raw);
        return;
    }

    float *mix_host = malloc((size_t)mix_bytes);
    float *residual_host = malloc((size_t)residual_bytes);
    float *ref_split_host = malloc((size_t)mix_bytes);
    float *fused_split_host = malloc((size_t)mix_bytes);
    float *ref_out_host = malloc((size_t)out_bytes);
    float *fused_out_host = malloc((size_t)out_bytes);
    float *ref_norm_host = malloc((size_t)out_bytes);
    float *fused_norm_host = malloc((size_t)out_bytes);
    TEST_ASSERT(mix_host != NULL);
    TEST_ASSERT(residual_host != NULL);
    TEST_ASSERT(ref_split_host != NULL);
    TEST_ASSERT(fused_split_host != NULL);
    TEST_ASSERT(ref_out_host != NULL);
    TEST_ASSERT(fused_out_host != NULL);
    TEST_ASSERT(ref_norm_host != NULL);
    TEST_ASSERT(fused_norm_host != NULL);
    if (!mix_host || !residual_host || !ref_split_host || !fused_split_host ||
        !ref_out_host || !fused_out_host || !ref_norm_host || !fused_norm_host) {
        free(mix_host);
        free(residual_host);
        free(ref_split_host);
        free(fused_split_host);
        free(ref_out_host);
        free(fused_out_host);
        free(ref_norm_host);
        free(fused_norm_host);
        ds4_gpu_tensor_free(mix);
        ds4_gpu_tensor_free(residual);
        ds4_gpu_tensor_free(ref_split);
        ds4_gpu_tensor_free(fused_split);
        ds4_gpu_tensor_free(ref_out);
        ds4_gpu_tensor_free(fused_out);
        ds4_gpu_tensor_free(ref_norm);
        ds4_gpu_tensor_free(fused_norm);
        free(model_raw);
        return;
    }

    for (uint32_t row = 0; row < n_rows; row++) {
        for (uint32_t i = 0; i < mix_hc; i++) {
            const int value =
                (int)(((row + 1u) * 19u + i * 11u + (i ^ row) * 3u) % 47u) - 23;
            mix_host[(uint64_t)row * mix_hc + i] = (float)value / 9.0f;
        }
        for (uint32_t hc = 0; hc < n_hc; hc++) {
            for (uint32_t d = 0; d < n_embd; d++) {
                const uint32_t key =
                    d * 37u + hc * 173u + row * 997u + ((d >> 3u) ^ (d * 7u));
                const int value = (int)(key % 2047u) - 1023;
                residual_host[((uint64_t)row * n_hc + hc) * n_embd + d] =
                    (float)value / 512.0f;
            }
        }
    }

    TEST_ASSERT(ds4_gpu_tensor_write(mix, 0, mix_host, mix_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(residual, 0, residual_host, residual_bytes) != 0);
    TEST_ASSERT(ds4_gpu_set_model_map(model_raw, model_alloc) != 0);
    TEST_ASSERT(ds4_gpu_hc_split_weighted_sum_tensor(
        ref_out, ref_split, mix, residual,
        model_raw, model_alloc, scale_offset, base_offset,
        n_embd, n_hc, sinkhorn_iters, hc_eps) != 0);
    TEST_ASSERT(ds4_gpu_rms_norm_weight_rows_tensor(
        ref_norm, ref_out, model_raw, model_alloc, norm_weight_offset,
        n_embd, n_rows, norm_eps) != 0);
    TEST_ASSERT(ds4_gpu_hc_split_weighted_sum_norm_tensor(
        fused_out, fused_norm, fused_split, mix, residual,
        model_raw, model_alloc, scale_offset, base_offset, norm_weight_offset,
        n_embd, n_hc, sinkhorn_iters, hc_eps, norm_eps) != 0);

    TEST_ASSERT(ds4_gpu_tensor_read(ref_split, 0, ref_split_host, mix_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(fused_split, 0, fused_split_host, mix_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(ref_out, 0, ref_out_host, out_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(fused_out, 0, fused_out_host, out_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(ref_norm, 0, ref_norm_host, out_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(fused_norm, 0, fused_norm_host, out_bytes) != 0);

    const test_float_compare_stats split_stats = test_compare_float_bits(
        ref_split_host, fused_split_host, (size_t)(n_rows * mix_hc));
    const test_float_compare_stats out_stats = test_compare_float_bits(
        ref_out_host, fused_out_host, (size_t)n_rows * n_embd);
    const test_float_compare_stats norm_stats = test_compare_float_bits(
        ref_norm_host, fused_norm_host, (size_t)n_rows * n_embd);
    fprintf(stderr,
            "ds4-test: batch HC+RMSNorm exactness rows=%u "
            "split=%zu/%llu max_ulp=%u max_abs=%g, "
            "collapse=%zu/%llu max_ulp=%u max_abs=%g, "
            "norm=%zu/%llu max_ulp=%u max_abs=%g\n",
            n_rows,
            split_stats.mismatch_count,
            (unsigned long long)(n_rows * mix_hc),
            split_stats.max_ulp,
            split_stats.max_abs,
            out_stats.mismatch_count,
            (unsigned long long)((uint64_t)n_rows * n_embd),
            out_stats.max_ulp,
            out_stats.max_abs,
            norm_stats.mismatch_count,
            (unsigned long long)((uint64_t)n_rows * n_embd),
            norm_stats.max_ulp,
            norm_stats.max_abs);
    TEST_ASSERT(split_stats.mismatch_count == 0);
    TEST_ASSERT(out_stats.mismatch_count == 0);
    TEST_ASSERT(norm_stats.mismatch_count == 0);

    free(mix_host);
    free(residual_host);
    free(ref_split_host);
    free(fused_split_host);
    free(ref_out_host);
    free(fused_out_host);
    free(ref_norm_host);
    free(fused_norm_host);
    ds4_gpu_tensor_free(mix);
    ds4_gpu_tensor_free(residual);
    ds4_gpu_tensor_free(ref_split);
    ds4_gpu_tensor_free(fused_split);
    ds4_gpu_tensor_free(ref_out);
    ds4_gpu_tensor_free(fused_out);
    ds4_gpu_tensor_free(ref_norm);
    ds4_gpu_tensor_free(fused_norm);
    free(model_raw);
}

static void test_metal_hc_producer_pre_norm_compound_exact(void) {
    const uint32_t n = 16384u;
    const uint32_t mix_dim = 24u;
    const uint32_t n_embd = 4096u;
    const uint32_t n_hc = 4u;
    const uint32_t sinkhorn_iters = 20u;
    const float eps = 1.0e-6f;
    const uint64_t page = (uint64_t)getpagesize();
    const uint64_t weight_offset = page;
    const uint64_t weight_bytes =
        (uint64_t)n * mix_dim * sizeof(uint16_t);
    const uint64_t scale_offset =
        test_round_up_u64(weight_offset + weight_bytes, page);
    const uint64_t base_offset =
        test_round_up_u64(scale_offset + 3u*sizeof(float), page);
    const uint64_t norm_offset =
        test_round_up_u64(base_offset + mix_dim*sizeof(float), page);
    const uint64_t model_bytes =
        test_round_up_u64(norm_offset + n_embd*sizeof(float), page);
    const uint64_t residual_bytes = (uint64_t)n*sizeof(float);
    const uint64_t mix_bytes = (uint64_t)mix_dim*sizeof(float);
    const uint64_t out_bytes = (uint64_t)n_embd*sizeof(float);

    const char *force_env =
        "DS4_METAL_ENABLE_HC_PRODUCER_PRE_NORM_FUSE";
    const char *disable_env =
        "DS4_METAL_DISABLE_HC_PRODUCER_PRE_NORM_FUSE";
    const char *disable_pre_env =
        "DS4_METAL_DISABLE_PRE_M5_HC_PRODUCER_PRE_NORM_FUSE";
    const char *disable_m5_env =
        "DS4_METAL_DISABLE_M5_HC_PRODUCER_PRE_NORM_FUSE";
    const char *disable_ports_env =
        "DS4_METAL_DISABLE_PRE_M5_DECODE_PORTS";
    char *saved_force = test_save_env(force_env);
    char *saved_disable = test_save_env(disable_env);
    char *saved_disable_pre = test_save_env(disable_pre_env);
    char *saved_disable_m5 = test_save_env(disable_m5_env);
    char *saved_disable_ports = test_save_env(disable_ports_env);

    void *model_raw = NULL;
    float *residual_host = NULL;
    float *ref_host = NULL;
    float *fused_host = NULL;
    ds4_gpu_tensor *residual = NULL;
    ds4_gpu_tensor *ref_mix = NULL;
    ds4_gpu_tensor *fused_mix = NULL;
    ds4_gpu_tensor *ref_split = NULL;
    ds4_gpu_tensor *fused_split = NULL;
    ds4_gpu_tensor *ref_out = NULL;
    ds4_gpu_tensor *fused_out = NULL;
    ds4_gpu_tensor *ref_norm = NULL;
    ds4_gpu_tensor *fused_norm = NULL;

    TEST_ASSERT(setenv(force_env, "1", 1) == 0);
    TEST_ASSERT(unsetenv(disable_env) == 0);
    TEST_ASSERT(unsetenv(disable_pre_env) == 0);
    TEST_ASSERT(unsetenv(disable_m5_env) == 0);
    TEST_ASSERT(unsetenv(disable_ports_env) == 0);
    TEST_ASSERT(posix_memalign(
                    &model_raw, (size_t)page, (size_t)model_bytes) == 0);
    if (!model_raw) goto cleanup;
    memset(model_raw, 0, (size_t)model_bytes);

    uint16_t *weight =
        (uint16_t *)((uint8_t *)model_raw + weight_offset);
    float *hc_scale = (float *)((uint8_t *)model_raw + scale_offset);
    float *hc_base = (float *)((uint8_t *)model_raw + base_offset);
    float *norm_weight = (float *)((uint8_t *)model_raw + norm_offset);
    for (uint32_t o = 0; o < mix_dim; o++) {
        for (uint32_t i = 0; i < n; i++) {
            const uint32_t key = i*37u + o*1009u + ((i >> 3u) ^ (o*19u));
            const int value = (int)(key % 127u) - 63;
            weight[(uint64_t)o*n + i] =
                test_float_to_f16((float)value/256.0f);
        }
    }
    hc_scale[0] = 0.625f;
    hc_scale[1] = -0.75f;
    hc_scale[2] = 0.4375f;
    for (uint32_t i = 0; i < mix_dim; i++) {
        const int value = (int)((i*17u + 5u) % 29u) - 14;
        hc_base[i] = (float)value/32.0f;
    }
    for (uint32_t i = 0; i < n_embd; i++) {
        norm_weight[i] = 0.5f + (float)((i*13u + 7u) % 31u)/32.0f;
    }

    residual_host = malloc((size_t)residual_bytes);
    ref_host = malloc((size_t)out_bytes);
    fused_host = malloc((size_t)out_bytes);
    TEST_ASSERT(residual_host != NULL);
    TEST_ASSERT(ref_host != NULL);
    TEST_ASSERT(fused_host != NULL);
    if (!residual_host || !ref_host || !fused_host) goto cleanup;
    for (uint32_t i = 0; i < n; i++) {
        const uint32_t key = i*131u + ((i >> 4u) ^ (i*7u));
        const int value = (int)(key % 4093u) - 2046;
        residual_host[i] = (float)value/1024.0f;
    }

    residual = ds4_gpu_tensor_alloc(residual_bytes);
    ref_mix = ds4_gpu_tensor_alloc(mix_bytes);
    fused_mix = ds4_gpu_tensor_alloc(mix_bytes);
    ref_split = ds4_gpu_tensor_alloc(mix_bytes);
    fused_split = ds4_gpu_tensor_alloc(mix_bytes);
    ref_out = ds4_gpu_tensor_alloc(out_bytes);
    fused_out = ds4_gpu_tensor_alloc(out_bytes);
    ref_norm = ds4_gpu_tensor_alloc(out_bytes);
    fused_norm = ds4_gpu_tensor_alloc(out_bytes);
    TEST_ASSERT(residual && ref_mix && fused_mix && ref_split && fused_split &&
                ref_out && fused_out && ref_norm && fused_norm);
    if (!residual || !ref_mix || !fused_mix || !ref_split || !fused_split ||
        !ref_out || !fused_out || !ref_norm || !fused_norm) {
        goto cleanup;
    }

    TEST_ASSERT(ds4_gpu_tensor_write(
                    residual, 0, residual_host, residual_bytes) != 0);
    TEST_ASSERT(ds4_gpu_set_model_map(model_raw, model_bytes) != 0);
    TEST_ASSERT(ds4_gpu_hc_rms_norm_mix_f16_tensor(
                    ref_mix, residual, model_raw, model_bytes,
                    weight_offset, n, mix_dim, eps) != 0);
    TEST_ASSERT(ds4_gpu_hc_split_weighted_sum_norm_tensor(
                    ref_out, ref_norm, ref_split, ref_mix, residual,
                    model_raw, model_bytes, scale_offset, base_offset,
                    norm_offset, n_embd, n_hc, sinkhorn_iters,
                    eps, eps) != 0);
    TEST_ASSERT(ds4_gpu_hc_rms_norm_mix_split_norm_f16_available() != 0);
    TEST_ASSERT(ds4_gpu_hc_rms_norm_mix_split_norm_f16_tensor(
                    fused_mix, fused_out, fused_norm, fused_split, residual,
                    model_raw, model_bytes, weight_offset, scale_offset,
                    base_offset, norm_offset, n, mix_dim, n_embd, n_hc,
                    sinkhorn_iters, eps, eps, eps) > 0);

    test_float_compare_stats mix_stats = {0};
    test_float_compare_stats split_stats = {0};
    test_float_compare_stats out_stats = {0};
    test_float_compare_stats norm_stats = {0};
#define TEST_HC_COMPOUND_COMPARE(ref_, fused_, bytes_, count_, stats_) do { \
        TEST_ASSERT(ds4_gpu_tensor_read((ref_), 0, ref_host, (bytes_)) != 0); \
        TEST_ASSERT(ds4_gpu_tensor_read((fused_), 0, fused_host, (bytes_)) != 0); \
        (stats_) = test_compare_float_bits(ref_host, fused_host, (count_)); \
    } while (0)
    TEST_HC_COMPOUND_COMPARE(
        ref_mix, fused_mix, mix_bytes, mix_dim, mix_stats);
    TEST_HC_COMPOUND_COMPARE(
        ref_split, fused_split, mix_bytes, mix_dim, split_stats);
    TEST_HC_COMPOUND_COMPARE(
        ref_out, fused_out, out_bytes, n_embd, out_stats);
    TEST_HC_COMPOUND_COMPARE(
        ref_norm, fused_norm, out_bytes, n_embd, norm_stats);
#undef TEST_HC_COMPOUND_COMPARE

    fprintf(stderr,
            "ds4-test: HC producer/pre-norm compound exact "
            "mix=%zu/%u split=%zu/%u collapse=%zu/%u norm=%zu/%u\n",
            mix_stats.mismatch_count, mix_dim,
            split_stats.mismatch_count, mix_dim,
            out_stats.mismatch_count, n_embd,
            norm_stats.mismatch_count, n_embd);
    TEST_ASSERT(mix_stats.mismatch_count == 0 && mix_stats.max_ulp == 0);
    TEST_ASSERT(split_stats.mismatch_count == 0 && split_stats.max_ulp == 0);
    TEST_ASSERT(out_stats.mismatch_count == 0 && out_stats.max_ulp == 0);
    TEST_ASSERT(norm_stats.mismatch_count == 0 && norm_stats.max_ulp == 0);
    TEST_ASSERT(setenv(disable_env, "1", 1) == 0);
    TEST_ASSERT(ds4_gpu_hc_rms_norm_mix_split_norm_f16_available() == 0);

cleanup:
    ds4_gpu_tensor_free(fused_norm);
    ds4_gpu_tensor_free(ref_norm);
    ds4_gpu_tensor_free(fused_out);
    ds4_gpu_tensor_free(ref_out);
    ds4_gpu_tensor_free(fused_split);
    ds4_gpu_tensor_free(ref_split);
    ds4_gpu_tensor_free(fused_mix);
    ds4_gpu_tensor_free(ref_mix);
    ds4_gpu_tensor_free(residual);
    free(fused_host);
    free(ref_host);
    free(residual_host);
    free(model_raw);
    test_restore_env(disable_ports_env, saved_disable_ports);
    test_restore_env(disable_m5_env, saved_disable_m5);
    test_restore_env(disable_pre_env, saved_disable_pre);
    test_restore_env(disable_env, saved_disable);
    test_restore_env(force_env, saved_force);
}

static void test_metal_output_hc_weights4_exact(void) {
    const uint32_t n_hc = 4;
    const float eps = 1.0e-6f;
    const uint64_t page = (uint64_t)getpagesize();
    const uint64_t scale_offset = 0;
    const uint64_t base_offset = page;
    const uint64_t model_alloc = 2u * page;
    const uint64_t bytes = n_hc * sizeof(float);
    const char *require_env = "DS4_METAL_REQUIRE_OUTPUT_HC_WEIGHTS4";
    char *saved_require = test_save_env(require_env);

    void *model_raw = NULL;
    TEST_ASSERT(posix_memalign(
                    &model_raw, (size_t)page, (size_t)model_alloc) == 0);
    ds4_gpu_tensor *pre = ds4_gpu_tensor_alloc(bytes);
    ds4_gpu_tensor *reference = ds4_gpu_tensor_alloc(bytes);
    ds4_gpu_tensor *candidate = ds4_gpu_tensor_alloc(bytes);
    TEST_ASSERT(model_raw != NULL);
    TEST_ASSERT(pre != NULL);
    TEST_ASSERT(reference != NULL);
    TEST_ASSERT(candidate != NULL);

    size_t total_mismatch = 0;
    uint32_t max_ulp = 0;
    const bool allocated = model_raw && pre && reference && candidate;
    if (allocated) {
        memset(model_raw, 0, (size_t)model_alloc);
        float *scale = (float *)((uint8_t *)model_raw + scale_offset);
        float *base = (float *)((uint8_t *)model_raw + base_offset);

        for (uint32_t ci = 0; ci < 4; ci++) {
            float pre_host[4];
            float ref_host[4];
            float candidate_host[4];
            for (uint32_t i = 0; i < 4; i++) {
                pre_host[i] =
                    ((float)((int)(ci * 17u + i * 11u) - 23)) / 8.0f;
                base[i] =
                    ((float)((int)(ci * 13u + i * 7u) - 19)) / 16.0f;
            }
            scale[0] = 0.62500012f + (float)ci * 0.125f;

            if (ci == 1) {
                /* The first lane distinguishes the required two-rounding
                 * sequence from an illegally contracted multiply-add. */
                const uint32_t pre_bits = 0x4620a541u;
                const uint32_t scale_bits = 0x462483bau;
                const uint32_t base_bits = 0xccce790eu;
                memcpy(&pre_host[0], &pre_bits, sizeof(pre_bits));
                memcpy(&scale[0], &scale_bits, sizeof(scale_bits));
                memcpy(&base[0], &base_bits, sizeof(base_bits));
            } else if (ci == 2) {
                const uint32_t pre_bits[4] = {
                    0x00000000u, 0x80000000u,
                    0x00000001u, 0x80000001u,
                };
                const uint32_t base_bits[4] = {
                    0x80000000u, 0x00000000u,
                    0x00800000u, 0x80800000u,
                };
                for (uint32_t i = 0; i < 4; i++) {
                    memcpy(&pre_host[i], &pre_bits[i], sizeof(uint32_t));
                    memcpy(&base[i], &base_bits[i], sizeof(uint32_t));
                }
                scale[0] = -1.00000012f;
            } else if (ci == 3) {
                pre_host[0] = 100.0f;
                pre_host[1] = -100.0f;
                pre_host[2] = 88.0f;
                pre_host[3] = -88.0f;
                scale[0] = 1.0f;
                for (uint32_t i = 0; i < 4; i++) base[i] = 0.0f;
            }

            memset(ref_host, 0xa5, sizeof(ref_host));
            memset(candidate_host, 0xa5, sizeof(candidate_host));
            TEST_ASSERT(ds4_gpu_tensor_write(pre, 0, pre_host, bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_write(
                            reference, 0, ref_host, bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_write(
                            candidate, 0, candidate_host, bytes) != 0);
            TEST_ASSERT(ds4_gpu_set_model_map(model_raw, model_alloc) != 0);
            TEST_ASSERT(unsetenv(require_env) == 0);
            ds4_gpu_test_set_flags(0);
            ds4_gpu_set_quality(true);
            TEST_ASSERT(ds4_gpu_output_hc_weights_tensor(
                            reference, pre, model_raw, model_alloc,
                            scale_offset, base_offset, n_hc, eps) != 0);

            ds4_gpu_set_quality(false);
            ds4_gpu_test_set_flags(DS4_GPU_TEST_OUTPUT_HC_WEIGHTS4);
            TEST_ASSERT(setenv(require_env, "1", 1) == 0);
            TEST_ASSERT(ds4_gpu_output_hc_weights_tensor(
                            candidate, pre, model_raw, model_alloc,
                            scale_offset, base_offset, n_hc, eps) != 0);

            TEST_ASSERT(ds4_gpu_tensor_read(
                            reference, 0, ref_host, bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_read(
                            candidate, 0, candidate_host, bytes) != 0);
            const test_float_compare_stats stats =
                test_compare_float_bits(ref_host, candidate_host, n_hc);
            fprintf(stderr,
                    "ds4-test: output HC weights4 exact case=%u "
                    "mismatch=%zu/%u max_ulp=%u max_abs=%g\n",
                    ci, stats.mismatch_count, n_hc,
                    stats.max_ulp, stats.max_abs);
            TEST_ASSERT(stats.mismatch_count == 0);
            total_mismatch += stats.mismatch_count;
            if (stats.max_ulp > max_ulp) max_ulp = stats.max_ulp;
        }

        /* Quality mode prevents strict selection of the fast path. */
        TEST_ASSERT(setenv(require_env, "1", 1) == 0);
        ds4_gpu_set_quality(true);
        TEST_ASSERT(ds4_gpu_output_hc_weights_tensor(
                        candidate, pre, model_raw, model_alloc,
                        scale_offset, base_offset, n_hc, eps) == 0);
        ds4_gpu_set_quality(false);
        ds4_gpu_test_set_flags(0);
    }

    test_restore_env(require_env, saved_require);
    fprintf(stderr,
            "ds4-test: output HC weights4 total mismatch=%zu/16 max_ulp=%u\n",
            total_mismatch, max_ulp);
    TEST_ASSERT(total_mismatch == 0);
    TEST_ASSERT(max_ulp == 0);

    ds4_gpu_tensor_free(candidate);
    ds4_gpu_tensor_free(reference);
    ds4_gpu_tensor_free(pre);
    free(model_raw);
}

static void test_metal_hc_rms_scale_project_f16_exact_shape(
        uint32_t in_dim,
        uint32_t seed) {
    const uint32_t out_dim = 24;
    /* One full 32-row matmul tile plus a tail row covers both load paths. */
    const uint32_t n_rows = 33;
    const float eps = 1.0e-6f;
    const uint64_t page = (uint64_t)getpagesize();
    const uint64_t weight_offset = page;
    const uint64_t weight_bytes =
        (uint64_t)in_dim * out_dim * sizeof(uint16_t);
    const uint64_t model_alloc = test_round_up_u64(
        weight_offset + weight_bytes, page);
    const uint64_t x_count = (uint64_t)in_dim * n_rows;
    const uint64_t out_count = (uint64_t)out_dim * n_rows;
    const uint64_t x_bytes = x_count * sizeof(float);
    const uint64_t out_bytes = out_count * sizeof(float);
    const uint64_t scale_bytes = (uint64_t)n_rows * sizeof(float);

    void *model_raw = NULL;
    TEST_ASSERT(posix_memalign(
                    &model_raw, (size_t)page, (size_t)model_alloc) == 0);
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(x_bytes);
    ds4_gpu_tensor *ref_norm = ds4_gpu_tensor_alloc(x_bytes);
    ds4_gpu_tensor *ref_out = ds4_gpu_tensor_alloc(out_bytes);
    ds4_gpu_tensor *scaled_out = ds4_gpu_tensor_alloc(out_bytes);
    /* Deliberately too small for the full-RMS fallback. */
    ds4_gpu_tensor *scale_scratch = ds4_gpu_tensor_alloc(scale_bytes);
    float *x_host = malloc((size_t)x_bytes);
    float *ref_norm_host = malloc((size_t)x_bytes);
    float *ref_out_host = malloc((size_t)out_bytes);
    float *scaled_out_host = malloc((size_t)out_bytes);
    float *scale_host = malloc((size_t)scale_bytes);
    float *expected_scale = malloc((size_t)scale_bytes);

    TEST_ASSERT(model_raw != NULL);
    TEST_ASSERT(x != NULL);
    TEST_ASSERT(ref_norm != NULL);
    TEST_ASSERT(ref_out != NULL);
    TEST_ASSERT(scaled_out != NULL);
    TEST_ASSERT(scale_scratch != NULL);
    TEST_ASSERT(x_host != NULL);
    TEST_ASSERT(ref_norm_host != NULL);
    TEST_ASSERT(ref_out_host != NULL);
    TEST_ASSERT(scaled_out_host != NULL);
    TEST_ASSERT(scale_host != NULL);
    TEST_ASSERT(expected_scale != NULL);

    const bool allocated = model_raw && x && ref_norm && ref_out &&
        scaled_out && scale_scratch && x_host && ref_norm_host &&
        ref_out_host && scaled_out_host && scale_host && expected_scale;
    test_float_compare_stats scale_stats = {0};
    test_float_compare_stats out_stats = {0};
    if (allocated) {
        memset(model_raw, 0, (size_t)model_alloc);
        uint16_t *weights = (uint16_t *)((uint8_t *)model_raw + weight_offset);
        for (uint32_t o = 0; o < out_dim; o++) {
            for (uint32_t i = 0; i < in_dim; i++) {
                const uint32_t key = i * 37u + o * 1009u + seed * 53u +
                    ((i >> 4u) ^ (o * 17u));
                uint16_t bits;
                if (key % 257u == 0u) {
                    bits = (key & 1u) ? 0x8000u : 0x0000u;
                } else if (key % 263u == 0u) {
                    bits = (key & 1u) ? 0x8001u : 0x0001u;
                } else if (key % 269u == 0u) {
                    bits = (key & 1u) ? 0x8400u : 0x0400u;
                } else {
                    const int value = (int)(key % 127u) - 63;
                    bits = test_float_to_f16((float)value / 128.0f);
                }
                weights[(uint64_t)o * in_dim + i] = bits;
            }
        }

        static const uint32_t rounding_bits[] = {
            0x3f800fffu, 0x3f801000u, 0x3f801001u,
            0x3f802fffu, 0x3f803000u, 0x3f803001u,
            0xbf800fffu, 0xbf801000u, 0xbf801001u,
            0x3eaaaaabu, 0xbeaaaaabu,
        };
        static const int sentinel_exp[] = { -2, -10, 0, -4, -6, -1 };
        for (uint32_t row = 0; row < n_rows; row++) {
            for (uint32_t i = 0; i < in_dim; i++) {
                const uint32_t key = i * 131u + row * 977u + seed * 71u +
                    ((i >> 3u) ^ (row * 29u));
                const float sign = (key & 1u) ? -1.0f : 1.0f;
                float value;
                switch (row % 6u) {
                    case 0:
                        value = (float)((int)(key % 4093u) - 2046) / 512.0f;
                        break;
                    case 1:
                        value = sign * ldexpf(
                            (float)(1u + (key & 7u)) / 8.0f, -19);
                        break;
                    case 2:
                        value = i % 257u == row % 257u
                            ? sign * (16.0f + (float)(key & 7u))
                            : sign * (float)(1u + (key & 31u)) / 8192.0f;
                        break;
                    case 3: {
                        const uint32_t bits = rounding_bits[
                            key % (sizeof(rounding_bits) /
                                   sizeof(rounding_bits[0]))];
                        memcpy(&value, &bits, sizeof(value));
                        break;
                    }
                    case 4:
                        if ((key & 7u) == 0u) {
                            const uint32_t bits = (key & 8u) ? 0x80000000u : 0u;
                            memcpy(&value, &bits, sizeof(value));
                        } else {
                            value = sign * ldexpf(
                                1.0f + (float)(key & 3u) * 0.25f,
                                (int)((key >> 4u) % 14u) - 10);
                        }
                        break;
                    default:
                        value = sign * (float)(1u + (key % 251u)) / 256.0f;
                        break;
                }
                x_host[(uint64_t)row * in_dim + i] = value;
            }
            x_host[(uint64_t)row * in_dim] =
                ldexpf(1.0f, sentinel_exp[row % 6u]);
        }

        for (uint64_t i = 0; i < out_count; i++) {
            const uint32_t bits = 0x7fc01000u + (uint32_t)(i & 0xfffu);
            memcpy(scaled_out_host + i, &bits, sizeof(bits));
        }
        for (uint32_t row = 0; row < n_rows; row++) {
            const uint32_t bits = 0x7fc02000u + row;
            memcpy(scale_host + row, &bits, sizeof(bits));
        }

        TEST_ASSERT(ds4_gpu_tensor_write(x, 0, x_host, x_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        scaled_out, 0, scaled_out_host, out_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        scale_scratch, 0, scale_host, scale_bytes) != 0);
        TEST_ASSERT(ds4_gpu_set_model_map(model_raw, model_alloc) != 0);
        ds4_gpu_set_quality(false);

        int ref_begun = ds4_gpu_begin_commands();
        int ref_ok = ref_begun;
        if (ref_ok) ref_ok = ds4_gpu_rms_norm_plain_rows_tensor(
            ref_norm, x, in_dim, n_rows, eps);
        if (ref_ok) ref_ok = ds4_gpu_matmul_f16_tensor(
            ref_out, model_raw, model_alloc, weight_offset,
            in_dim, out_dim, ref_norm, n_rows);
        const int ref_end = ref_begun ? ds4_gpu_end_commands() : 0;
        TEST_ASSERT(ref_ok != 0);
        TEST_ASSERT(ref_end != 0);

        int scaled_begun = ds4_gpu_begin_commands();
        int scaled_ok = scaled_begun;
        if (scaled_ok) scaled_ok = ds4_gpu_hc_rms_scale_project_f16_tensor(
            scaled_out, scale_scratch, model_raw, model_alloc, weight_offset,
            in_dim, out_dim, x, n_rows, eps);
        const int scaled_end = scaled_begun ? ds4_gpu_end_commands() : 0;
        TEST_ASSERT(scaled_ok != 0);
        TEST_ASSERT(scaled_end != 0);

        TEST_ASSERT(ds4_gpu_tensor_read(
                        ref_norm, 0, ref_norm_host, x_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        ref_out, 0, ref_out_host, out_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        scaled_out, 0, scaled_out_host, out_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        scale_scratch, 0, scale_host, scale_bytes) != 0);

        for (uint32_t row = 0; row < n_rows; row++) {
            expected_scale[row] = ldexpf(
                ref_norm_host[(uint64_t)row * in_dim],
                -sentinel_exp[row % 6u]);
            TEST_ASSERT(isfinite(expected_scale[row]));
            TEST_ASSERT(isfinite(scale_host[row]));
        }
        scale_stats = test_compare_float_bits(
            expected_scale, scale_host, n_rows);
        out_stats = test_compare_float_bits(
            ref_out_host, scaled_out_host, (size_t)out_count);
    }

    fprintf(stderr,
            "ds4-test: HC RMS scale F16 projection exact K=%u rows=%u "
            "scale=%zu/%u max_ulp=%u projection=%zu/%llu max_ulp=%u\n",
            in_dim, n_rows,
            scale_stats.mismatch_count, n_rows, scale_stats.max_ulp,
            out_stats.mismatch_count, (unsigned long long)out_count,
            out_stats.max_ulp);
    TEST_ASSERT(scale_stats.mismatch_count == 0);
    TEST_ASSERT(scale_stats.max_ulp == 0);
    TEST_ASSERT(out_stats.mismatch_count == 0);
    TEST_ASSERT(out_stats.max_ulp == 0);

    free(expected_scale);
    free(scale_host);
    free(scaled_out_host);
    free(ref_out_host);
    free(ref_norm_host);
    free(x_host);
    ds4_gpu_tensor_free(scale_scratch);
    ds4_gpu_tensor_free(scaled_out);
    ds4_gpu_tensor_free(ref_out);
    ds4_gpu_tensor_free(ref_norm);
    ds4_gpu_tensor_free(x);
    free(model_raw);
}

static void test_metal_hc_rms_scale_project_f16_exact(void) {
    const char *disable_env = "DS4_METAL_DISABLE_HC_RMS_SCALE_PROJ";
    char *saved_disable = test_save_env(disable_env);

    TEST_ASSERT(unsetenv(disable_env) == 0);
    ds4_gpu_test_set_flags(DS4_GPU_TEST_HC_RMS_SCALE_PROJ);
    test_metal_hc_rms_scale_project_f16_exact_shape(16384u, 59u);
    test_metal_hc_rms_scale_project_f16_exact_shape(28672u, 61u);
    ds4_gpu_test_set_flags(0);

    test_restore_env(disable_env, saved_disable);
}

static void test_metal_router_simd_finalize_exact(void) {
    typedef struct {
        const char *name;
        bool has_bias;
        uint32_t pattern;
    } router_case;
    static const router_case cases[] = {
        { "unique", false, 0 },
        { "bias", true, 0 },
        { "top6-ties", false, 1 },
        { "signed-zero-extremes", false, 2 },
        { "clamp-underflow", false, 3 },
        { "sum-rounding", true, 4 },
    };
    const uint32_t n_expert = 256;
    const uint32_t n_used = 6;
    const uint32_t modes = 4;
    const uint64_t probs_bytes = (uint64_t)n_expert * sizeof(float);
    const uint64_t selected_bytes = (uint64_t)n_used * sizeof(int32_t);
    const uint64_t weights_bytes = (uint64_t)n_used * sizeof(float);
    const uint64_t page = (uint64_t)getpagesize();
    const char *disable_env =
        "DS4_METAL_DISABLE_PRE_M5_ROUTER_SIMD_FINALIZE";
    const char *weights_disable_env =
        "DS4_METAL_DISABLE_PRE_M5_ROUTER_SIMD_WEIGHTS_FUSION";
    const char *transform_finalize_disable_env =
        "DS4_METAL_DISABLE_PRE_M5_ROUTER_TRANSFORM_FINALIZE_FUSION";
    const char *select_disable_env =
        "DS4_METAL_DISABLE_ROUTER_SELECT_FUSION";

    void *model_raw = NULL;
    TEST_ASSERT(posix_memalign(&model_raw, (size_t)page, (size_t)page) == 0);
    ds4_gpu_tensor *logits = ds4_gpu_tensor_alloc(probs_bytes);
    ds4_gpu_tensor *ref_selected = ds4_gpu_tensor_alloc(selected_bytes);
    ds4_gpu_tensor *simd_selected = ds4_gpu_tensor_alloc(selected_bytes);
    ds4_gpu_tensor *ref_weights = ds4_gpu_tensor_alloc(weights_bytes);
    ds4_gpu_tensor *simd_weights = ds4_gpu_tensor_alloc(weights_bytes);
    ds4_gpu_tensor *ref_probs = ds4_gpu_tensor_alloc(probs_bytes);
    ds4_gpu_tensor *simd_probs = ds4_gpu_tensor_alloc(probs_bytes);
    float *logits_host = malloc((size_t)probs_bytes);
    float *ref_probs_host = malloc((size_t)probs_bytes);
    float *simd_probs_host = malloc((size_t)probs_bytes);
    int32_t ref_selected_host[6];
    int32_t simd_selected_host[6];
    int32_t unique_selected_host[6];
    float ref_weights_host[6];
    float simd_weights_host[6];
    TEST_ASSERT(model_raw != NULL);
    TEST_ASSERT(logits != NULL);
    TEST_ASSERT(ref_selected != NULL);
    TEST_ASSERT(simd_selected != NULL);
    TEST_ASSERT(ref_weights != NULL);
    TEST_ASSERT(simd_weights != NULL);
    TEST_ASSERT(ref_probs != NULL);
    TEST_ASSERT(simd_probs != NULL);
    TEST_ASSERT(logits_host != NULL);
    TEST_ASSERT(ref_probs_host != NULL);
    TEST_ASSERT(simd_probs_host != NULL);

    char *saved_disable = test_save_env(disable_env);
    char *saved_weights_disable = test_save_env(weights_disable_env);
    char *saved_transform_finalize_disable =
        test_save_env(transform_finalize_disable_env);
    char *saved_select_disable = test_save_env(select_disable_env);
    size_t total_selected_mismatch = 0;
    size_t total_weights_mismatch = 0;
    size_t total_probs_mismatch = 0;
    const bool allocated = model_raw && logits && ref_selected &&
        simd_selected && ref_weights && simd_weights && ref_probs &&
        simd_probs && logits_host && ref_probs_host && simd_probs_host;
    if (allocated) {
        memset(model_raw, 0, (size_t)page);
        float *bias = model_raw;
        for (uint32_t i = 0; i < n_used; i++) {
            bias[200u + i] = 16.0f - (float)i;
        }
        TEST_ASSERT(ds4_gpu_set_model_map(model_raw, page) != 0);
        ds4_gpu_set_quality(false);
        TEST_ASSERT(setenv(transform_finalize_disable_env, "1", 1) == 0);
        TEST_ASSERT(unsetenv(select_disable_env) == 0);

        for (size_t ci = 0; ci < sizeof(cases) / sizeof(cases[0]); ci++) {
            const router_case *c = &cases[ci];
            for (uint32_t i = 0; i < n_expert; i++) {
                const int value =
                    (int)((i * 47u + (i ^ (i >> 3u)) * 13u) % 257u) - 128;
                logits_host[i] = (float)value / 32.0f;
            }
            if (c->pattern == 1) {
                static const uint32_t tied[] = {
                    7u, 19u, 43u, 71u, 103u, 149u, 211u, 239u,
                };
                for (uint32_t i = 0; i < n_expert; i++) {
                    logits_host[i] = -7.0f - (float)(i % 17u) / 64.0f;
                }
                for (size_t i = 0; i < sizeof(tied) / sizeof(tied[0]); i++) {
                    logits_host[tied[i]] = 1.0f;
                }
            } else if (c->pattern == 2) {
                for (uint32_t i = 0; i < n_expert; i++) {
                    const uint32_t zero_bits = (i & 1u) ? 0x80000000u : 0u;
                    memcpy(&logits_host[i], &zero_bits, sizeof(zero_bits));
                }
                logits_host[3] = 80.0f;
                logits_host[17] = 40.0f;
                logits_host[61] = 20.0f;
                logits_host[127] = -20.0f;
                logits_host[193] = -40.0f;
                logits_host[251] = -80.0f;
            } else if (c->pattern == 3) {
                for (uint32_t i = 0; i < n_expert; i++) {
                    logits_host[i] = -30.0f - (float)(i % 11u);
                }
            } else if (c->pattern == 4) {
                static const float rounding_logits[6] = {
                    -0.6356699467f,
                    -0.8182631135f,
                    -2.7906901836f,
                    -3.4414808750f,
                    -3.4991359711f,
                    -3.1251864433f,
                };
                for (uint32_t i = 0; i < n_expert; i++) {
                    logits_host[i] = -20.0f;
                }
                for (uint32_t i = 0; i < n_used; i++) {
                    logits_host[200u + i] = rounding_logits[i];
                }
            }
            TEST_ASSERT(ds4_gpu_tensor_write(
                logits, 0, logits_host, probs_bytes) != 0);

            TEST_ASSERT(setenv(disable_env, "1", 1) == 0);
            TEST_ASSERT(ds4_gpu_router_select_tensor(
                ref_selected, ref_weights, ref_probs,
                model_raw, page, 0, 0, 1, 0,
                n_expert, n_used, 1.5f, 1, 0,
                c->has_bias, false, logits) != 0);
            TEST_ASSERT(ds4_gpu_tensor_read(
                ref_selected, 0, ref_selected_host, selected_bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_read(
                ref_weights, 0, ref_weights_host, weights_bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_read(
                ref_probs, 0, ref_probs_host, probs_bytes) != 0);

            if (ci == 0) {
                memcpy(unique_selected_host,
                       ref_selected_host,
                       sizeof(unique_selected_host));
            } else if (c->has_bias) {
                TEST_ASSERT(memcmp(unique_selected_host,
                                   ref_selected_host,
                                   sizeof(unique_selected_host)) != 0);
            }

            TEST_ASSERT(unsetenv(disable_env) == 0);
            for (uint32_t mode = 0; mode < modes; mode++) {
                const bool fused_weights = mode != 0;
                const bool fused_transform = mode >= 2;
                if (fused_weights) {
                    TEST_ASSERT(unsetenv(weights_disable_env) == 0);
                } else {
                    TEST_ASSERT(setenv(weights_disable_env, "1", 1) == 0);
                }
                if (fused_transform) {
                    TEST_ASSERT(unsetenv(transform_finalize_disable_env) == 0);
                } else {
                    TEST_ASSERT(setenv(
                        transform_finalize_disable_env, "1", 1) == 0);
                }
                for (uint32_t i = 0; i < n_expert; i++) {
                    const uint32_t poison_bits = 0x7fc00001u + i;
                    memcpy(&simd_probs_host[i],
                           &poison_bits,
                           sizeof(poison_bits));
                }
                TEST_ASSERT(ds4_gpu_tensor_write(
                    simd_probs, 0, simd_probs_host, probs_bytes) != 0);
                TEST_ASSERT(ds4_gpu_router_select_tensor(
                    simd_selected, simd_weights, simd_probs,
                    model_raw, page, 0, 0, 1, 0,
                    n_expert, n_used, 1.5f, 1, 0,
                    c->has_bias, false, logits) != 0);
                TEST_ASSERT(ds4_gpu_tensor_read(
                    simd_selected, 0, simd_selected_host,
                    selected_bytes) != 0);
                TEST_ASSERT(ds4_gpu_tensor_read(
                    simd_weights, 0, simd_weights_host,
                    weights_bytes) != 0);
                TEST_ASSERT(ds4_gpu_tensor_read(
                    simd_probs, 0, simd_probs_host, probs_bytes) != 0);

                size_t selected_mismatch = 0;
                for (uint32_t i = 0; i < n_used; i++) {
                    if (ref_selected_host[i] != simd_selected_host[i]) {
                        selected_mismatch++;
                    }
                }
                const test_float_compare_stats weights_stats =
                    test_compare_float_bits(
                        ref_weights_host, simd_weights_host, n_used);
                const test_float_compare_stats probs_stats =
                    test_compare_float_bits(
                        ref_probs_host, simd_probs_host, n_expert);
                fprintf(stderr,
                        "ds4-test: router SIMD finalize exactness "
                        "case=%s mode=%s rep=%u selected=%zu/%u weights=%zu/%u "
                        "probs=%zu/%u max_weight_ulp=%u max_prob_ulp=%u\n",
                        c->name,
                        fused_transform ? "fused-transform" :
                        fused_weights ? "fused-weights" : "split-weights",
                        fused_transform ? mode - 2u : 0u,
                        selected_mismatch,
                        n_used,
                        weights_stats.mismatch_count,
                        n_used,
                        probs_stats.mismatch_count,
                        n_expert,
                        weights_stats.max_ulp,
                        probs_stats.max_ulp);
                TEST_ASSERT(selected_mismatch == 0);
                TEST_ASSERT(weights_stats.mismatch_count == 0);
                TEST_ASSERT(probs_stats.mismatch_count == 0);
                total_selected_mismatch += selected_mismatch;
                total_weights_mismatch += weights_stats.mismatch_count;
                total_probs_mismatch += probs_stats.mismatch_count;
            }
        }
    }

    test_restore_env(disable_env, saved_disable);
    test_restore_env(weights_disable_env, saved_weights_disable);
    test_restore_env(transform_finalize_disable_env,
                     saved_transform_finalize_disable);
    test_restore_env(select_disable_env, saved_select_disable);
    fprintf(stderr,
            "ds4-test: router SIMD finalize total selected=%zu/%zu "
            "weights=%zu/%zu probs=%zu/%zu\n",
            total_selected_mismatch,
            (sizeof(cases) / sizeof(cases[0])) * modes * (size_t)n_used,
            total_weights_mismatch,
            (sizeof(cases) / sizeof(cases[0])) * modes * (size_t)n_used,
            total_probs_mismatch,
            (sizeof(cases) / sizeof(cases[0])) * modes * (size_t)n_expert);
    TEST_ASSERT(total_selected_mismatch == 0);
    TEST_ASSERT(total_weights_mismatch == 0);
    TEST_ASSERT(total_probs_mismatch == 0);

    free(simd_probs_host);
    free(ref_probs_host);
    free(logits_host);
    ds4_gpu_tensor_free(simd_probs);
    ds4_gpu_tensor_free(ref_probs);
    ds4_gpu_tensor_free(simd_weights);
    ds4_gpu_tensor_free(ref_weights);
    ds4_gpu_tensor_free(simd_selected);
    ds4_gpu_tensor_free(ref_selected);
    ds4_gpu_tensor_free(logits);
    free(model_raw);
}

static void test_metal_router_weights_batch_exact(void) {
    typedef struct {
        const char *name;
        uint32_t n_tokens;
        bool has_bias;
        bool hash_mode;
        uint32_t pattern;
    } router_batch_case;
    static const router_batch_case cases[] = {
        { "rows2-unique", 2, false, false, 0 },
        { "rows17-bias-ties", 17, true, false, 1 },
        { "rows129-hash-duplicates", 129, false, true, 2 },
        { "rows2048-typical", 2048, false, false, 0 },
        { "rows3-clamp-underflow", 3, false, false, 3 },
    };
    const uint32_t n_expert = 256;
    const uint32_t n_used = 6;
    const uint32_t max_tokens = 2048;
    const uint32_t hash_rows = 64;
    const uint32_t repeats = 2;
    const uint64_t logits_bytes =
        (uint64_t)max_tokens * n_expert * sizeof(float);
    const uint64_t selected_bytes =
        (uint64_t)max_tokens * n_used * sizeof(int32_t);
    const uint64_t weights_bytes =
        (uint64_t)max_tokens * n_used * sizeof(float);
    const uint64_t tokens_bytes = (uint64_t)max_tokens * sizeof(int32_t);
    const uint64_t page = (uint64_t)getpagesize();
    const uint64_t bias_offset = 0;
    const uint64_t hash_offset = 2048;
    const char *disable_env =
        "DS4_METAL_DISABLE_ROUTER_WEIGHTS_BATCH_FUSION";
    const char *select_disable_env =
        "DS4_METAL_DISABLE_ROUTER_SELECT_FUSION";

    void *model_raw = NULL;
    TEST_ASSERT(posix_memalign(&model_raw, (size_t)page, (size_t)page) == 0);
    ds4_gpu_tensor *logits = ds4_gpu_tensor_alloc(logits_bytes);
    ds4_gpu_tensor *tokens = ds4_gpu_tensor_alloc(tokens_bytes);
    ds4_gpu_tensor *ref_selected = ds4_gpu_tensor_alloc(selected_bytes);
    ds4_gpu_tensor *batch_selected = ds4_gpu_tensor_alloc(selected_bytes);
    ds4_gpu_tensor *ref_weights = ds4_gpu_tensor_alloc(weights_bytes);
    ds4_gpu_tensor *batch_weights = ds4_gpu_tensor_alloc(weights_bytes);
    ds4_gpu_tensor *ref_probs = ds4_gpu_tensor_alloc(logits_bytes);
    ds4_gpu_tensor *batch_probs = ds4_gpu_tensor_alloc(logits_bytes);
    float *logits_host = malloc((size_t)logits_bytes);
    int32_t *tokens_host = malloc((size_t)tokens_bytes);
    int32_t *ref_selected_host = malloc((size_t)selected_bytes);
    int32_t *batch_selected_host = malloc((size_t)selected_bytes);
    float *ref_weights_host = malloc((size_t)weights_bytes);
    float *batch_weights_host = malloc((size_t)weights_bytes);
    float *ref_probs_host = malloc((size_t)logits_bytes);
    float *batch_probs_host = malloc((size_t)logits_bytes);
    TEST_ASSERT(model_raw != NULL);
    TEST_ASSERT(logits != NULL);
    TEST_ASSERT(tokens != NULL);
    TEST_ASSERT(ref_selected != NULL);
    TEST_ASSERT(batch_selected != NULL);
    TEST_ASSERT(ref_weights != NULL);
    TEST_ASSERT(batch_weights != NULL);
    TEST_ASSERT(ref_probs != NULL);
    TEST_ASSERT(batch_probs != NULL);
    TEST_ASSERT(logits_host != NULL);
    TEST_ASSERT(tokens_host != NULL);
    TEST_ASSERT(ref_selected_host != NULL);
    TEST_ASSERT(batch_selected_host != NULL);
    TEST_ASSERT(ref_weights_host != NULL);
    TEST_ASSERT(batch_weights_host != NULL);
    TEST_ASSERT(ref_probs_host != NULL);
    TEST_ASSERT(batch_probs_host != NULL);

    char *saved_disable = test_save_env(disable_env);
    char *saved_select_disable = test_save_env(select_disable_env);
    size_t total_selected_mismatch = 0;
    size_t total_weights_mismatch = 0;
    size_t total_probs_mismatch = 0;
    const bool allocated = model_raw && logits && tokens && ref_selected &&
        batch_selected && ref_weights && batch_weights && ref_probs &&
        batch_probs && logits_host && tokens_host && ref_selected_host &&
        batch_selected_host && ref_weights_host && batch_weights_host &&
        ref_probs_host && batch_probs_host;
    if (allocated) {
        memset(model_raw, 0, (size_t)page);
        float *bias = (float *)((uint8_t *)model_raw + bias_offset);
        int32_t *hash = (int32_t *)((uint8_t *)model_raw + hash_offset);
        for (uint32_t i = 0; i < n_expert; i++) {
            bias[i] = (float)((int)((i * 29u) % 67u) - 33) / 16.0f;
        }
        for (uint32_t row = 0; row < hash_rows; row++) {
            const int32_t base = (int32_t)((row * 37u) % n_expert);
            hash[(uint64_t)row * n_used + 0u] = base;
            hash[(uint64_t)row * n_used + 1u] = base;
            hash[(uint64_t)row * n_used + 2u] = (base + 19) % (int32_t)n_expert;
            hash[(uint64_t)row * n_used + 3u] = (base + 43) % (int32_t)n_expert;
            hash[(uint64_t)row * n_used + 4u] = (base + 43) % (int32_t)n_expert;
            hash[(uint64_t)row * n_used + 5u] = (base + 101) % (int32_t)n_expert;
        }
        for (uint32_t row = 0; row < max_tokens; row++) {
            tokens_host[row] = (int32_t)((row * 13u + 7u) % hash_rows);
        }

        TEST_ASSERT(ds4_gpu_set_model_map(model_raw, page) != 0);
        ds4_gpu_set_quality(false);
        TEST_ASSERT(ds4_gpu_tensor_write(
            tokens, 0, tokens_host, tokens_bytes) != 0);
        TEST_ASSERT(unsetenv(select_disable_env) == 0);

        for (size_t ci = 0; ci < sizeof(cases) / sizeof(cases[0]); ci++) {
            const router_batch_case *c = &cases[ci];
            const size_t prob_count = (size_t)c->n_tokens * n_expert;
            const size_t route_count = (size_t)c->n_tokens * n_used;
            const uint64_t case_probs_bytes = prob_count * sizeof(float);
            const uint64_t case_selected_bytes = route_count * sizeof(int32_t);
            const uint64_t case_weights_bytes = route_count * sizeof(float);
            for (uint32_t row = 0; row < c->n_tokens; row++) {
                for (uint32_t expert = 0; expert < n_expert; expert++) {
                    const size_t index = (size_t)row * n_expert + expert;
                    const int value =
                        (int)((row * 131u + expert * 47u +
                               (expert ^ (expert >> 3u)) * 13u) % 513u) -
                        256;
                    logits_host[index] = (float)value / 64.0f;
                    if (c->pattern == 1) {
                        logits_host[index] =
                            -7.0f - (float)((row + expert) % 17u) / 64.0f;
                    } else if (c->pattern == 3) {
                        logits_host[index] =
                            -30.0f - (float)((row * 17u + expert * 29u) % 11u);
                    }
                }
                if (c->pattern == 1) {
                    static const uint32_t tied[] = {
                        7u, 19u, 43u, 71u, 103u, 149u, 211u, 239u,
                    };
                    for (size_t i = 0; i < sizeof(tied) / sizeof(tied[0]); i++) {
                        logits_host[(size_t)row * n_expert + tied[i]] = 1.0f;
                    }
                }
            }
            TEST_ASSERT(ds4_gpu_tensor_write(
                logits, 0, logits_host, case_probs_bytes) != 0);

            TEST_ASSERT(setenv(disable_env, "1", 1) == 0);
            TEST_ASSERT(ds4_gpu_router_select_batch_tensor(
                ref_selected, ref_weights, ref_probs,
                model_raw, page, bias_offset, hash_offset, hash_rows,
                1, 0, c->has_bias, c->hash_mode, logits, tokens,
                n_expert, n_used, 1.5f, c->n_tokens) != 0);
            TEST_ASSERT(ds4_gpu_tensor_read(
                ref_selected, 0, ref_selected_host, case_selected_bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_read(
                ref_weights, 0, ref_weights_host, case_weights_bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_read(
                ref_probs, 0, ref_probs_host, case_probs_bytes) != 0);

            if (c->hash_mode) {
                for (uint32_t row = 0; row < c->n_tokens; row++) {
                    const uint32_t hash_row = (uint32_t)tokens_host[row];
                    for (uint32_t lane = 0; lane < n_used; lane++) {
                        TEST_ASSERT(
                            ref_selected_host[(size_t)row * n_used + lane] ==
                            hash[(uint64_t)hash_row * n_used + lane]);
                    }
                }
            }
            if (c->pattern == 3) {
                for (uint32_t row = 0; row < c->n_tokens; row++) {
                    float sum = 0.0f;
                    for (uint32_t lane = 0; lane < n_used; lane++) {
                        const int32_t expert =
                            ref_selected_host[(size_t)row * n_used + lane];
                        sum += ref_probs_host[(size_t)row * n_expert +
                                              (uint32_t)expert];
                    }
                    TEST_ASSERT(sum < 6.103515625e-5f);
                }
            }

            TEST_ASSERT(unsetenv(disable_env) == 0);
            for (uint32_t rep = 0; rep < repeats; rep++) {
                TEST_ASSERT(ds4_gpu_router_select_batch_tensor(
                    batch_selected, batch_weights, batch_probs,
                    model_raw, page, bias_offset, hash_offset, hash_rows,
                    1, 0, c->has_bias, c->hash_mode, logits, tokens,
                    n_expert, n_used, 1.5f, c->n_tokens) != 0);
                TEST_ASSERT(ds4_gpu_tensor_read(
                    batch_selected, 0, batch_selected_host,
                    case_selected_bytes) != 0);
                TEST_ASSERT(ds4_gpu_tensor_read(
                    batch_weights, 0, batch_weights_host,
                    case_weights_bytes) != 0);
                TEST_ASSERT(ds4_gpu_tensor_read(
                    batch_probs, 0, batch_probs_host,
                    case_probs_bytes) != 0);

                size_t selected_mismatch = 0;
                for (size_t i = 0; i < route_count; i++) {
                    if (ref_selected_host[i] != batch_selected_host[i]) {
                        selected_mismatch++;
                    }
                }
                const test_float_compare_stats weights_stats =
                    test_compare_float_bits(
                        ref_weights_host, batch_weights_host, route_count);
                const test_float_compare_stats probs_stats =
                    test_compare_float_bits(
                        ref_probs_host, batch_probs_host, prob_count);
                fprintf(stderr,
                        "ds4-test: router batch weights exactness "
                        "case=%s rep=%u selected=%zu/%zu weights=%zu/%zu "
                        "probs=%zu/%zu max_weight_ulp=%u max_prob_ulp=%u\n",
                        c->name,
                        rep,
                        selected_mismatch,
                        route_count,
                        weights_stats.mismatch_count,
                        route_count,
                        probs_stats.mismatch_count,
                        prob_count,
                        weights_stats.max_ulp,
                        probs_stats.max_ulp);
                TEST_ASSERT(selected_mismatch == 0);
                TEST_ASSERT(weights_stats.mismatch_count == 0);
                TEST_ASSERT(probs_stats.mismatch_count == 0);
                total_selected_mismatch += selected_mismatch;
                total_weights_mismatch += weights_stats.mismatch_count;
                total_probs_mismatch += probs_stats.mismatch_count;
            }
        }
    }

    test_restore_env(disable_env, saved_disable);
    test_restore_env(select_disable_env, saved_select_disable);
    fprintf(stderr,
            "ds4-test: router batch weights total selected=%zu "
            "weights=%zu probs=%zu\n",
            total_selected_mismatch,
            total_weights_mismatch,
            total_probs_mismatch);
    TEST_ASSERT(total_selected_mismatch == 0);
    TEST_ASSERT(total_weights_mismatch == 0);
    TEST_ASSERT(total_probs_mismatch == 0);

    free(batch_probs_host);
    free(ref_probs_host);
    free(batch_weights_host);
    free(ref_weights_host);
    free(batch_selected_host);
    free(ref_selected_host);
    free(tokens_host);
    free(logits_host);
    ds4_gpu_tensor_free(batch_probs);
    ds4_gpu_tensor_free(ref_probs);
    ds4_gpu_tensor_free(batch_weights);
    ds4_gpu_tensor_free(ref_weights);
    ds4_gpu_tensor_free(batch_selected);
    ds4_gpu_tensor_free(ref_selected);
    ds4_gpu_tensor_free(tokens);
    ds4_gpu_tensor_free(logits);
    free(model_raw);
}
#endif

static void test_metal_kernel_group(void) {
#if defined(__APPLE__)
    if (test_env_bool("DS4_TEST_METAL_RESIDENT_ORACLES_ONLY")) {
        test_metal_flush_commands_progress_exact();
        test_metal_small_prefill_direct_exact();
        test_metal_batch_attn_out_hc_fusion_exact();
        return;
    }
#endif
    test_metal_f16_matvec_fast_nr0_4();
    test_metal_f16_prefill_matmul();
    test_metal_q8_0_prefill_matmul();
    test_metal_pack_slot_rows_f32();
    test_metal_store_raw_kv_batch_wrap();
    test_dspark_cache_window_crop();
    test_metal_q8_0_decode_pair_exact();
#if defined(__APPLE__)
    test_metal_flush_commands_progress_exact();
    test_metal_q8_0_decode_rows_exact();
    test_metal_q8_attention_output_static_batch_exact();
    test_metal_q4_attention_output_tiny_batch_exact();
    test_metal_dspark_device_proposer_q8();
    test_metal_f16_compressor_pair_state_store_exact();
    test_metal_f16_compressor_quad_state_store_exact();
    test_metal_q8_qkv_compressor_compound_exact();
    test_metal_compressor_ape_add_exact();
    test_metal_compressor_ratio4_pack_exact();
    test_metal_compressor_ratio4_replay_pack_exact();
    test_metal_compressor_ratio4_direct_pool_exact();
    test_metal_compressor_ratio4_exact_pool_decode();
    test_metal_inplace_rope_pair_exact();
    test_metal_contiguous_f32_f16_roundtrip_exact();
    test_metal_gathered_kv_stage_exact();
    test_metal_contiguous_compressed_f16_attention_exact();
    test_metal_persistent_zero_attention_mask_exact();
    test_metal_zero_prefix_prefill_mask_cache_exact();
    test_metal_small_prefill_direct_exact();
    test_metal_batch_attn_out_hc_fusion_exact();
    test_metal_hc_split_weighted_sum_norm_batch_exact();
    test_metal_hc_producer_pre_norm_compound_exact();
    test_metal_output_hc_weights4_exact();
    test_metal_hc_rms_scale_project_f16_exact();
    test_metal_router_simd_finalize_exact();
    test_metal_router_weights_batch_exact();
#endif
}

static void test_metal_short_prefill_ratio4(void) {
    ds4_engine *engine = test_get_engine(false);
    if (!engine) return;

    const int tokens[] = {
        ds4_token_user(engine),
        ds4_token_assistant(engine),
        ds4_token_eos(engine),
    };
    for (size_t i = 0; i < sizeof(tokens) / sizeof(tokens[0]); i++) {
        TEST_ASSERT(tokens[i] >= 0);
        if (tokens[i] < 0) return;
    }

    for (size_t n = 1; n <= 3; n++) {
        ds4_tokens prompt = {0};
        for (size_t i = 0; i < n; i++) {
            ds4_tokens_push(&prompt, tokens[i]);
        }
        TEST_ASSERT(prompt.len == (int)n);

        ds4_session *session = NULL;
        TEST_ASSERT(ds4_session_create(&session, engine, 2048) == 0);
        if (!session) {
            ds4_tokens_free(&prompt);
            return;
        }

        char err[160] = {0};
        const int rc = ds4_session_sync(session, &prompt, err, sizeof(err));
        if (rc != 0) {
            fprintf(stderr, "ds4-test: short prefill failed for %zu token(s): %s\n",
                    n, err);
        }
        TEST_ASSERT(rc == 0);

        ds4_session_free(session);
        ds4_tokens_free(&prompt);
    }
}

typedef struct {
    int expected_start;
    int expected_total;
    int last_current;
    int chunk_events;
    int display_events;
    int intermediate_events;
    bool invalid;
} test_continued_prefill_progress;

static void test_continued_prefill_progress_cb(
        void       *ud,
        const char *event,
        int         current,
        int         total) {
    test_continued_prefill_progress *p = ud;
    const bool chunk = !strcmp(event, "prefill_chunk");
    const bool display = !strcmp(event, "prefill_display");
    if (!chunk && !display) return;

    if (total != p->expected_total || current < p->expected_start ||
        current > total || current < p->last_current) {
        p->invalid = true;
    }
    if (current > p->expected_start && current < total) {
        p->intermediate_events++;
    }
    if (current > p->last_current) p->last_current = current;
    if (chunk) p->chunk_events++;
    if (display) p->display_events++;
}

static double test_monotonic_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1000000000.0;
}

static bool test_sync_continued_prefill_stage(
        ds4_session *session,
        ds4_tokens  *prompt,
        int          previous_len,
        int          next_len,
        double      *elapsed_out) {
    test_continued_prefill_progress progress = {
        .expected_start = previous_len,
        .expected_total = next_len,
        .last_current = previous_len,
    };
    char err[160] = {0};
    prompt->len = next_len;
    ds4_session_set_progress(session, test_continued_prefill_progress_cb,
                             &progress);
    ds4_session_set_display_progress(session,
                                     test_continued_prefill_progress_cb,
                                     &progress);
    const double started = test_monotonic_seconds();
    const int rc = ds4_session_sync(session, prompt, err, sizeof(err));
    const double elapsed = test_monotonic_seconds() - started;
    ds4_session_set_progress(session, NULL, NULL);
    ds4_session_set_display_progress(session, NULL, NULL);
    if (elapsed_out) *elapsed_out = elapsed;

    if (rc != 0) {
        fprintf(stderr,
                "ds4-test: continued prefill %d -> %d failed: %s\n",
                previous_len, next_len, err);
    }
    const int added = next_len - previous_len;
    fprintf(stderr,
            "ds4-test: continued prefill +%d: %.2f ms, %.1f t/s, "
            "chunk=%d display=%d intermediate=%d\n",
            added, elapsed * 1000.0,
            elapsed > 0.0 ? (double)added / elapsed : 0.0,
            progress.chunk_events, progress.display_events,
            progress.intermediate_events);

    TEST_ASSERT(rc == 0);
    TEST_ASSERT(!progress.invalid);
    TEST_ASSERT(progress.chunk_events > 0);
    TEST_ASSERT(progress.last_current == next_len);
    if (added > 1 &&
        (added >= 32 ||
         !test_env_bool("DS4_TEST_CONTINUED_PREFILL_ALLOW_COARSE"))) {
        TEST_ASSERT(progress.intermediate_events > 0);
    }
    if (added >= 32) TEST_ASSERT(progress.display_events > 0);
    return rc == 0 && !progress.invalid;
}

static bool test_top_tokens_overlap(
        const ds4_token_score *a,
        const ds4_token_score *b,
        int                    count) {
    bool a0_in_b = false;
    bool b0_in_a = false;
    for (int i = 0; i < count; i++) {
        if (a[0].id == b[i].id) a0_in_b = true;
        if (b[0].id == a[i].id) b0_in_a = true;
    }
    return a0_in_b && b0_in_a;
}

static void test_glm53_continued_prefill(void) {
    ds4_engine *engine = test_get_engine(false);
    if (!engine || !ds4_engine_is_glm53(engine)) {
        fprintf(stderr,
                "ds4-test: glm53-continued-prefill skipped (GLM 5.3 model required)\n");
        return;
    }

    const int base_len = 64;
    uint32_t large_add = test_env_u32("DS4_TEST_CONTINUED_PREFILL_TOKENS");
    if (large_add == 0) large_add = 256;
    uint32_t large_steps = test_env_u32("DS4_TEST_CONTINUED_PREFILL_STEPS");
    if (large_steps == 0) large_steps = 1;
    const uint64_t final_len64 = (uint64_t)base_len + 4u +
                                 (uint64_t)large_add * large_steps;
    TEST_ASSERT(final_len64 < INT_MAX - 128);
    if (final_len64 >= INT_MAX - 128) return;
    const int final_len = (int)final_len64;
    int ctx_size = final_len + 128;
    if (ctx_size < 4096) ctx_size = 4096;

    ds4_tokens pattern = {0};
    ds4_tokens prompt = {0};
    ds4_session *resumed = NULL;
    ds4_session *cold = NULL;
    ds4_token_score resumed_top[8] = {0};
    ds4_token_score cold_top[8] = {0};
    char err[160] = {0};
    ds4_tokenize_text(engine,
                      " continued prefill checks latency throughput and progress",
                      &pattern);
    TEST_ASSERT(pattern.len > 0);
    if (pattern.len == 0) goto cleanup;
    ds4_chat_begin(engine, &prompt);
    while (prompt.len < final_len) {
        ds4_tokens_push(&prompt, pattern.v[prompt.len % pattern.len]);
    }

    TEST_ASSERT(ds4_session_create(&resumed, engine, ctx_size) == 0);
    if (!resumed) goto cleanup;

    prompt.len = base_len;
    TEST_ASSERT(ds4_session_sync(resumed, &prompt, err, sizeof(err)) == 0);
    if (!test_sync_continued_prefill_stage(resumed, &prompt,
                                           base_len, base_len + 1, NULL)) {
        goto cleanup;
    }
    if (!test_sync_continued_prefill_stage(resumed, &prompt,
                                           base_len + 1, base_len + 4, NULL)) {
        goto cleanup;
    }
    int previous_len = base_len + 4;
    for (uint32_t step = 0; step < large_steps; step++) {
        const int next_len = previous_len + (int)large_add;
        if (!test_sync_continued_prefill_stage(resumed, &prompt,
                                               previous_len, next_len, NULL)) {
            goto cleanup;
        }
        previous_len = next_len;
    }
    TEST_ASSERT(ds4_session_top_logprobs(resumed, resumed_top, 8) == 8);
    ds4_session_free(resumed);
    resumed = NULL;

    TEST_ASSERT(ds4_session_create(&cold, engine, ctx_size) == 0);
    if (!cold) goto cleanup;
    prompt.len = final_len;
    TEST_ASSERT(ds4_session_sync(cold, &prompt, err, sizeof(err)) == 0);
    TEST_ASSERT(ds4_session_top_logprobs(cold, cold_top, 8) == 8);
    int same_rank = 0;
    float max_same_rank_delta = 0.0f;
    for (int i = 0; i < 8; i++) {
        if (resumed_top[i].id != cold_top[i].id) continue;
        same_rank++;
        const float delta = fabsf(resumed_top[i].logit - cold_top[i].logit);
        if (delta > max_same_rank_delta) max_same_rank_delta = delta;
    }
    fprintf(stderr,
            "ds4-test: continued/cold final top token %d/%d, "
            "same-rank top8=%d, max same-rank logit delta=%.6g\n",
            resumed_top[0].id, cold_top[0].id,
            same_rank, max_same_rank_delta);
    TEST_ASSERT(resumed_top[0].id == cold_top[0].id);
    TEST_ASSERT(test_top_tokens_overlap(resumed_top, cold_top, 8));

cleanup:
    ds4_session_free(cold);
    ds4_session_free(resumed);
    ds4_tokens_free(&prompt);
    ds4_tokens_free(&pattern);
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
    if (ds4_session_sync(session, &prompt, err, sizeof(err)) != 0) {
        fprintf(stderr, "ds4-test: vector %s prefill failed: %s\n", vc->id, err);
        TEST_ASSERT(false);
        ds4_session_free(session);
        ds4_tokens_free(&prompt);
        return;
    }

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
            if (ds4_session_eval(session, token, err, sizeof(err)) != 0) {
                fprintf(stderr, "ds4-test: vector %s step %d eval failed: %s\n",
                        vc->id, i, err);
                TEST_ASSERT(false);
                break;
            }
        }
    }

    ds4_session_free(session);
    ds4_tokens_free(&prompt);
}

static bool test_logprob_vector_case_disabled(const char *path,
                                              const test_vec_case *vc) {
    /*
     * This one long-context vector currently matches the public DeepSeek API less
     * after adding the official Hadamard+FP4 indexer path.  The public official
     * implementation and the API appear to disagree here; the official graph has
     * slightly lower local perplexity on the A/B check we ran, so DS4 keeps that
     * implementation and only excludes this brittle API fixture for now.
     */
    return !strcmp(path, "tests/test-vectors/flash-pre-0731/official.vec") &&
           !strcmp(vc->id, "long_memory_archive");
}

static void test_official_logprob_vectors_run(const char *case_filter) {
    const char *path = getenv("DS4_TEST_VECTOR_FILE");
    if (!path || !path[0]) {
        path = "tests/test-vectors/flash-0731/official.vec";
    }
    FILE *fp = fopen(path, "rb");
    TEST_ASSERT(fp != NULL);
    if (!fp) return;

    char *saved_prefill_chunk = test_save_env("DS4_METAL_PREFILL_CHUNK");
    char *saved_disable_metal4 = test_save_env("DS4_METAL_DISABLE_METAL4");
    test_streaming_prefill_env saved_canonical_streaming_prefill =
        test_force_canonical_streaming_prefill();
    setenv("DS4_METAL_PREFILL_CHUNK", "2048", 1);
    if (getenv("DS4_TEST_LOGPROB_AUTO_METAL") == NULL) {
        setenv("DS4_METAL_DISABLE_METAL4", "1", 1);
    } else {
        unsetenv("DS4_METAL_DISABLE_METAL4");
    }
    ds4_engine *engine = test_open_engine(false);
    if (!engine) {
        test_restore_canonical_streaming_prefill(saved_canonical_streaming_prefill);
        test_restore_env("DS4_METAL_DISABLE_METAL4", saved_disable_metal4);
        test_restore_env("DS4_METAL_PREFILL_CHUNK", saved_prefill_chunk);
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
        if (test_logprob_vector_case_disabled(path, &vc)) {
            fprintf(stderr, "ds4-test: vector %s skipped (API/official graph mismatch)\n",
                    vc.id);
            continue;
        }
        fprintf(stderr, "ds4-test: vector %s\n", vc.id);
        test_logprob_vector_case(engine, &vc);
        ran++;
    }
    TEST_ASSERT(!case_filter || !case_filter[0] || ran == 1);
    ds4_engine_close(engine);
    test_restore_canonical_streaming_prefill(saved_canonical_streaming_prefill);
    test_restore_env("DS4_METAL_DISABLE_METAL4", saved_disable_metal4);
    test_restore_env("DS4_METAL_PREFILL_CHUNK", saved_prefill_chunk);
    fclose(fp);
}

static void test_official_logprob_vectors(void) {
    test_official_logprob_vectors_run(NULL);
}

static void test_metal_ssd_streaming_cache_pressure(void) {
#ifndef __APPLE__
    fprintf(stderr,
            "ds4-test: Metal SSD streaming cache-pressure repro skipped "
            "(Metal-only)\n");
#else
    /*
     * Regression repro for GitHub issue #384.
     *
     * The bug needs the Metal SSD-streaming decode layer-batch path and a small
     * routed-expert cache. Under pressure, a cache entry referenced by an
     * already-encoded-but-not-yet-executed layer can be reused for a later
     * layer in the same command buffer, producing deterministic wrong logits.
     */
    char *saved_streaming = test_save_env("DS4_TEST_SSD_STREAMING");
    char *saved_cache_gb = test_save_env("DS4_TEST_SSD_STREAMING_CACHE_GB");
    char *saved_cache_experts =
        test_save_env("DS4_TEST_SSD_STREAMING_CACHE_EXPERTS");
    char *saved_disable_layer_batch =
        test_save_env("DS4_METAL_DISABLE_STREAMING_LAYER_BATCH");
    char *saved_disable_static_decode =
        test_save_env("DS4_METAL_DISABLE_STREAMING_STATIC_DECODE_MAP");
    char *saved_one_stage =
        test_save_env("DS4_METAL_MOE_ONE_STAGE_PROFILE");

    setenv("DS4_TEST_SSD_STREAMING", "1", 1);
    setenv("DS4_TEST_SSD_STREAMING_CACHE_GB", "16", 1);
    unsetenv("DS4_TEST_SSD_STREAMING_CACHE_EXPERTS");
    unsetenv("DS4_METAL_DISABLE_STREAMING_LAYER_BATCH");
    unsetenv("DS4_METAL_DISABLE_STREAMING_STATIC_DECODE_MAP");
    unsetenv("DS4_METAL_MOE_ONE_STAGE_PROFILE");

    fprintf(stderr,
            "ds4-test: Metal SSD streaming cache-pressure repro "
            "(16GiB cache, layer-batched decode, short_code_completion)\n");
    test_official_logprob_vectors_run("short_code_completion");

    test_restore_env("DS4_METAL_MOE_ONE_STAGE_PROFILE", saved_one_stage);
    test_restore_env("DS4_METAL_DISABLE_STREAMING_STATIC_DECODE_MAP",
                     saved_disable_static_decode);
    test_restore_env("DS4_METAL_DISABLE_STREAMING_LAYER_BATCH",
                     saved_disable_layer_batch);
    test_restore_env("DS4_TEST_SSD_STREAMING_CACHE_EXPERTS",
                     saved_cache_experts);
    test_restore_env("DS4_TEST_SSD_STREAMING_CACHE_GB", saved_cache_gb);
    test_restore_env("DS4_TEST_SSD_STREAMING", saved_streaming);
#endif
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

static void test_local_golden_case_run(ds4_engine *engine,
                                       const test_local_golden_case *tc) {
    char *prompt_text = test_read_file(tc->prompt_path);
    TEST_ASSERT(prompt_text != NULL);
    if (!prompt_text) return;

    ds4_tokens prompt = {0};
    if (!strcmp(tc->mode, "text")) {
        ds4_tokenize_text(engine, prompt_text, &prompt);
    } else if (!strcmp(tc->mode, "rendered")) {
        ds4_tokenize_rendered_chat(engine, prompt_text, &prompt);
    } else if (!strcmp(tc->mode, "chat")) {
        ds4_encode_chat_prompt(engine, "", prompt_text, DS4_THINK_NONE, &prompt);
    } else {
        TEST_ASSERT(!"unknown local golden prompt mode");
    }
    free(prompt_text);
    TEST_ASSERT(prompt.len >= tc->frontier);
    if (prompt.len < tc->frontier) {
        ds4_tokens_free(&prompt);
        return;
    }

    ds4_tokens prefix = {
        .v = prompt.v,
        .len = tc->frontier,
        .cap = tc->frontier,
    };

    ds4_session *session = NULL;
    TEST_ASSERT(ds4_session_create(&session, engine, tc->ctx) == 0);
    if (!session) {
        ds4_tokens_free(&prompt);
        return;
    }

    char err[160];
    TEST_ASSERT(ds4_session_sync(session, &prefix, err, sizeof(err)) == 0);

    const int vocab = ds4_engine_vocab_size(engine);
    float *cand_logits = malloc((size_t)vocab * sizeof(cand_logits[0]));
    TEST_ASSERT(cand_logits != NULL);
    if (cand_logits &&
        ds4_session_copy_logits(session, cand_logits, vocab) == vocab) {
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
                "ds4-test: local golden %s top1 ref=%d cand=%d "
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
    ds4_session_free(session);
    ds4_tokens_free(&prompt);
}

static void test_local_golden_vectors(void) {
    const char *path = getenv("DS4_TEST_LOCAL_GOLDEN_FILE");
    if (!path || !path[0]) {
        path = "tests/test-vectors/flash-0731/local-golden.vec";
    }
    FILE *fp = fopen(path, "rb");
    TEST_ASSERT(fp != NULL);
    if (!fp) return;

    char *saved_prefill_chunk = test_save_env("DS4_METAL_PREFILL_CHUNK");
    char *saved_disable_metal4 = test_save_env("DS4_METAL_DISABLE_METAL4");
    char *saved_moe_tile_max = test_save_env("DS4_METAL_MOE_TILE_MAX");
    test_streaming_prefill_env saved_canonical_streaming_prefill =
        test_force_canonical_streaming_prefill();
    setenv("DS4_METAL_PREFILL_CHUNK", "4096", 1);
    setenv("DS4_METAL_DISABLE_METAL4", "1", 1);
    unsetenv("DS4_METAL_MOE_TILE_MAX");

    ds4_engine *engine = test_open_engine(false);
    if (!engine) {
        test_restore_canonical_streaming_prefill(saved_canonical_streaming_prefill);
        test_restore_env("DS4_METAL_MOE_TILE_MAX", saved_moe_tile_max);
        test_restore_env("DS4_METAL_DISABLE_METAL4", saved_disable_metal4);
        test_restore_env("DS4_METAL_PREFILL_CHUNK", saved_prefill_chunk);
        fclose(fp);
        return;
    }

    test_local_golden_case tc;
    while (test_read_local_golden_case(fp, &tc)) {
        if (!test_fill_local_golden_case(fp, &tc)) break;
        test_local_golden_case_run(engine, &tc);
    }

    ds4_engine_close(engine);
    test_restore_canonical_streaming_prefill(saved_canonical_streaming_prefill);
    test_restore_env("DS4_METAL_MOE_TILE_MAX", saved_moe_tile_max);
    test_restore_env("DS4_METAL_DISABLE_METAL4", saved_disable_metal4);
    test_restore_env("DS4_METAL_PREFILL_CHUNK", saved_prefill_chunk);
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
    ds4_tokens prompt;
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
    ds4_tokens_free(&tc->prompt);
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
 * Metal4/TensorOps equivalence is a smoke test, not a demand for bitwise local
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
            "ds4-test: Tensor equivalence %s top1 ref=%d cand=%d top5_overlap=%d/%d overlap=%d/%d max_rank_delta=%d rms=%g max_abs=%g top20_max_abs=%g\n",
            tc->id, ref_top[0], cand_top[0],
            top5_overlap, TEST_MPP_EQ_TOP5,
            overlap, TEST_MPP_EQ_TOPK,
            max_rank_delta, rms, max_abs, top_abs);
    fprintf(stderr, "ds4-test: Tensor equivalence %s largest deltas:", tc->id);
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

static bool test_mpp_capture(ds4_engine *engine, const test_mpp_eq_case *tc,
                             float *logits, int *gen, int *gen_len) {
    ds4_session *session = NULL;
    TEST_ASSERT(ds4_session_create(&session, engine, tc->ctx) == 0);
    if (!session) return false;

    char err[160];
    bool ok = ds4_session_sync(session, &tc->prompt, err, sizeof(err)) == 0;
    TEST_ASSERT(ok);
    if (ok) {
        ok = ds4_session_copy_logits(session, logits, tc->vocab_size) == tc->vocab_size;
        TEST_ASSERT(ok);
    }

    int n = 0;
    while (ok && n < tc->gen_steps) {
        const int token = ds4_session_argmax(session);
        gen[n++] = token;
        if (n < tc->gen_steps && ds4_session_eval(session, token, err, sizeof(err)) != 0) {
            ok = false;
            TEST_ASSERT(false);
        }
    }
    *gen_len = n;

    ds4_session_free(session);
    return ok;
}

static bool test_mpp_capture_logits_only(ds4_engine *engine,
                                         const test_mpp_eq_case *tc,
                                         float *logits) {
    ds4_session *session = NULL;
    TEST_ASSERT(ds4_session_create(&session, engine, tc->ctx) == 0);
    if (!session) return false;

    char err[160];
    bool ok = ds4_session_sync(session, &tc->prompt, err, sizeof(err)) == 0;
    TEST_ASSERT(ok);
    if (ok) {
        ok = ds4_session_copy_logits(session, logits, tc->vocab_size) == tc->vocab_size;
        TEST_ASSERT(ok);
    }

    ds4_session_free(session);
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

static int test_load_mpp_cases(ds4_engine *engine, test_mpp_eq_case *cases, int cap) {
    const char *path = getenv("DS4_TEST_VECTOR_FILE");
    if (!path || !path[0]) {
        path = "tests/test-vectors/flash-0731/official.vec";
    }
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
        tc->vocab_size = ds4_engine_vocab_size(engine);
        tc->gen_steps = vc.nsteps < TEST_VEC_MAX_STEPS ? vc.nsteps : TEST_VEC_MAX_STEPS;
        ds4_encode_chat_prompt(engine, "", prompt_text, DS4_THINK_NONE, &tc->prompt);
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
            "ds4-test: Tensor summary route=%s cases=%d capture_fail=%d logits_fail=%d greedy_fail=%d top1_mismatch=%d min_top5_overlap=%d/%d min_overlap=%d/%d worst_rank_delta=%d worst_rms=%g worst_max_abs=%g worst_top20_max_abs=%g\n",
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
    fprintf(stderr, "ds4-test: Tensor equivalence candidate route=%s\n", label);
    test_mpp_eq_summary summary;
    test_mpp_summary_init(&summary, label);
    ds4_engine *cand_engine = test_open_engine(false);
    if (cand_engine) {
        const int vocab_size = ncase > 0 ? cases[0].vocab_size : 0;
        float *cand_logits = malloc((size_t)vocab_size * sizeof(cand_logits[0]));
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
                                "ds4-test: Tensor equivalence %s greedy token mismatch step=%d ref=%d cand=%d\n",
                                tc->id, j, tc->ref_gen[j], cand_gen[j]);
                        summary.greedy_failures++;
                    }
                    TEST_ASSERT(cand_gen[j] == tc->ref_gen[j]);
                }
            }
            free(cand_logits);
        }
        ds4_engine_close(cand_engine);
    }
    test_mpp_summary_print(&summary);
}

static void test_metal_mpp_equivalence(void) {
    test_close_engines();

    test_mpp_eq_case cases[TEST_MPP_EQ_MAX_CASES];
    memset(cases, 0, sizeof(cases));

    char *saved_disable_metal4 = test_save_env("DS4_METAL_DISABLE_METAL4");
    setenv("DS4_METAL_DISABLE_METAL4", "1", 1);
    ds4_engine *ref_engine = test_open_engine(false);
    if (!ref_engine) {
        test_restore_env("DS4_METAL_DISABLE_METAL4", saved_disable_metal4);
        return;
    }

    const int ncase = test_load_mpp_cases(ref_engine, cases, TEST_MPP_EQ_MAX_CASES);
    TEST_ASSERT(ncase > 0);
    for (int i = 0; i < ncase; i++) {
        test_mpp_eq_case *tc = &cases[i];
        tc->ref_logits = malloc((size_t)tc->vocab_size * sizeof(tc->ref_logits[0]));
        TEST_ASSERT(tc->ref_logits != NULL);
        if (!tc->ref_logits) continue;
        TEST_ASSERT(test_mpp_capture(ref_engine, tc,
                                     tc->ref_logits,
                                     tc->ref_gen,
                                     &tc->ref_gen_len));
    }
    ds4_engine_close(ref_engine);
    test_restore_env("DS4_METAL_DISABLE_METAL4", saved_disable_metal4);

    test_run_mpp_candidate("auto", cases, ncase);

    for (int i = 0; i < ncase; i++) test_mpp_eq_case_free(&cases[i]);
}

static void test_streaming_decode_prefill_correctness(void) {
    test_close_engines();
    if (!test_env_bool("DS4_TEST_SSD_STREAMING")) {
        fprintf(stderr,
                "ds4-test: streaming decode-prefill correctness skipped "
                "(set DS4_TEST_SSD_STREAMING=1 to enable)\n");
        return;
    }

    test_mpp_eq_case cases[TEST_MPP_EQ_MAX_CASES];
    memset(cases, 0, sizeof(cases));

    test_streaming_prefill_env saved_canonical_streaming_prefill =
        test_force_canonical_streaming_prefill();

    ds4_engine *ref_engine = test_open_engine(false);
    if (!ref_engine) {
        test_restore_canonical_streaming_prefill(saved_canonical_streaming_prefill);
        return;
    }

    const int ncase = test_load_mpp_cases(ref_engine, cases, TEST_MPP_EQ_MAX_CASES);
    TEST_ASSERT(ncase > 0);
    for (int i = 0; i < ncase; i++) {
        test_mpp_eq_case *tc = &cases[i];
        tc->ref_logits = malloc((size_t)tc->vocab_size * sizeof(tc->ref_logits[0]));
        TEST_ASSERT(tc->ref_logits != NULL);
        if (!tc->ref_logits) continue;
        TEST_ASSERT(test_mpp_capture(ref_engine, tc,
                                     tc->ref_logits,
                                     tc->ref_gen,
                                     &tc->ref_gen_len));
    }
    ds4_engine_close(ref_engine);

    unsetenv("DS4_METAL_DISABLE_STREAMING_COLD_DECODE_PREFILL");
    unsetenv("DS4_METAL_DISABLE_STREAMING_PREFILL_BATCH_SELECTED_ADDR");

    ds4_engine *cand_engine = test_open_engine(false);
    if (cand_engine) {
        for (int i = 0; i < ncase; i++) {
            test_mpp_eq_case *tc = &cases[i];
            if (!tc->ref_logits) continue;

            float *cand_cold = malloc((size_t)tc->vocab_size * sizeof(cand_cold[0]));
            float *cand_warm_a = malloc((size_t)tc->vocab_size * sizeof(cand_warm_a[0]));
            float *cand_warm_b = malloc((size_t)tc->vocab_size * sizeof(cand_warm_b[0]));
            TEST_ASSERT(cand_cold != NULL);
            TEST_ASSERT(cand_warm_a != NULL);
            TEST_ASSERT(cand_warm_b != NULL);
            if (!cand_cold || !cand_warm_a || !cand_warm_b) {
                free(cand_cold);
                free(cand_warm_a);
                free(cand_warm_b);
                continue;
            }

            TEST_ASSERT(test_mpp_capture_logits_only(cand_engine, tc, cand_cold));
            TEST_ASSERT(test_mpp_capture_logits_only(cand_engine, tc, cand_warm_a));
            TEST_ASSERT(test_mpp_capture_logits_only(cand_engine, tc, cand_warm_b));

            test_mpp_eq_result result = test_compare_mpp_logits(tc, cand_cold, false);
            TEST_ASSERT(result.nonfinite == 0);
            TEST_ASSERT(result.top5_overlap >= 2);
            TEST_ASSERT(result.overlap >= 10);
            TEST_ASSERT(result.rms <= 4.0f);
            TEST_ASSERT(result.top20_max_abs <= 12.0f);

            int cold_warm_neq = 0;
            int warm_repeat_neq = 0;
            int repeat_nonfinite = 0;
            float cold_warm_max_abs = 0.0f;
            float warm_repeat_max_abs = 0.0f;
            for (int j = 0; j < tc->vocab_size; j++) {
                if (!isfinite(cand_cold[j]) ||
                    !isfinite(cand_warm_a[j]) ||
                    !isfinite(cand_warm_b[j])) {
                    repeat_nonfinite++;
                    continue;
                }
                const float cold_warm_d = fabsf(cand_cold[j] - cand_warm_a[j]);
                if (cold_warm_d != 0.0f) cold_warm_neq++;
                if (cold_warm_d > cold_warm_max_abs) cold_warm_max_abs = cold_warm_d;
                const float warm_repeat_d = fabsf(cand_warm_a[j] - cand_warm_b[j]);
                if (warm_repeat_d != 0.0f) warm_repeat_neq++;
                if (warm_repeat_d > warm_repeat_max_abs) {
                    warm_repeat_max_abs = warm_repeat_d;
                }
            }
            TEST_ASSERT(repeat_nonfinite == 0);
            TEST_ASSERT(cold_warm_neq == 0);
            TEST_ASSERT(warm_repeat_neq == 0);
            fprintf(stderr,
                    "ds4-test: streaming decode-prefill %s cold_warm_neq=%d "
                    "cold_warm_max_abs=%g warm_repeat_neq=%d "
                    "warm_repeat_max_abs=%g top1 canonical=%d decode=%d\n",
                    tc->id,
                    cold_warm_neq,
                    cold_warm_max_abs,
                    warm_repeat_neq,
                    warm_repeat_max_abs,
                    result.ref_top1,
                    result.cand_top1);

            free(cand_cold);
            free(cand_warm_a);
            free(cand_warm_b);
        }
        ds4_engine_close(cand_engine);
    }

    test_restore_canonical_streaming_prefill(saved_canonical_streaming_prefill);
    for (int i = 0; i < ncase; i++) test_mpp_eq_case_free(&cases[i]);
}

#define TEST_LIST_FILES_USER_PROMPT \
    "Use the list_files tool to list the current directory exactly once, " \
    "then report the listed files and stop."

#define TEST_LIST_FILES_TOOL_JSON \
    "{\"type\":\"function\",\"function\":{" \
        "\"name\":\"list_files\"," \
        "\"description\":\"List files in a directory.\"," \
        "\"parameters\":{\"type\":\"object\",\"properties\":{" \
            "\"path\":{\"type\":\"string\",\"description\":\"Directory path to list.\"}" \
        "},\"required\":[\"path\"]}" \
    "}}"

#define TEST_LIST_FILES_RESULT "[\"README.md\",\"Makefile\",\"ds4.c\",\"metal\"]"

static const char *test_tool_call_request_json(void) {
    return
        "{"
        "\"model\":\"deepseek-v4-flash\","
        "\"messages\":[{\"role\":\"user\",\"content\":\""
            TEST_LIST_FILES_USER_PROMPT
        "\"}],"
        "\"tools\":[" TEST_LIST_FILES_TOOL_JSON "],"
        "\"tool_choice\":\"auto\","
        "\"think\":false,"
        "\"temperature\":0,"
        "\"max_tokens\":256,"
        "\"stream\":false"
        "}";
}

static char *test_tool_result_request_json(const char *assistant_content,
                                           const tool_call *call) {
    if (!call || !call->id || !call->id[0]) return NULL;

    buf b = {0};
    buf_puts(&b,
        "{\"model\":\"deepseek-v4-flash\",\"messages\":["
        "{\"role\":\"user\",\"content\":");
    json_escape(&b, TEST_LIST_FILES_USER_PROMPT);
    buf_puts(&b, "},{\"role\":\"assistant\",\"content\":");
    json_escape(&b, assistant_content ? assistant_content : "");
    buf_puts(&b, ",\"tool_calls\":[{\"id\":");
    json_escape(&b, call->id);
    buf_puts(&b, ",\"type\":\"function\",\"function\":{\"name\":");
    json_escape(&b, call->name ? call->name : "");
    buf_puts(&b, ",\"arguments\":");
    json_escape(&b, call->arguments ? call->arguments : "{}");
    buf_puts(&b, "}}]},{\"role\":\"tool\",\"tool_call_id\":");
    json_escape(&b, call->id);
    buf_puts(&b, ",\"content\":");
    json_escape(&b, TEST_LIST_FILES_RESULT);
    buf_puts(&b,
        "}],\"tools\":[" TEST_LIST_FILES_TOOL_JSON "],"
        "\"tool_choice\":\"auto\",\"think\":false,"
        "\"temperature\":0,\"max_tokens\":256,\"stream\":false}");
    return buf_take(&b);
}

/* A complete tool call inside unclosed reasoning is recovered directly. The
 * detector must wait for the complete block, and the parser must keep only the
 * preceding prose in reasoning_content. */
static void test_think_tool_recovery(void) {
    const char *generated =
        "The user wants a directory listing.\n\n"
        DS4_TOOL_CALLS_START "\n"
        DS4_INVOKE_START " name=\"list_files\">\n"
        DS4_PARAM_START " name=\"path\" string=\"true\">." DS4_PARAM_END "\n"
        DS4_INVOKE_END "\n"
        DS4_TOOL_CALLS_END;

    buf text = {0};
    size_t scan_from = 0;
    bool complete = false;
    for (size_t i = 0; generated[i]; i++) {
        buf_append(&text, generated + i, 1);
        complete = complete_tool_call_inside_thinking(text.ptr, text.len,
                                                      &scan_from);
        TEST_ASSERT(complete == (generated[i + 1] == '\0'));
    }
    TEST_ASSERT(complete);

    char *content = NULL;
    char *reasoning = NULL;
    tool_calls calls = {0};
    bool parsed = parse_generated_message_ex(text.ptr, true,
                                             &content, &reasoning, &calls);
    TEST_ASSERT(parsed);
    TEST_ASSERT(calls.len > 0 && !strcmp(calls.v[0].name, "list_files"));
    TEST_ASSERT(calls.v[0].arguments && strstr(calls.v[0].arguments, "\"path\": \".\""));
    TEST_ASSERT(content && content[0] == '\0');
    TEST_ASSERT(reasoning && !strcmp(reasoning, "The user wants a directory listing."));

    fprintf(stderr,
            "ds4-test: think-tool-recovery complete=%d calls=%d name=%s\n",
            complete ? 1 : 0, calls.len, calls.len ? calls.v[0].name : "-");

    free(content);
    free(reasoning);
    tool_calls_free(&calls);
    buf_free(&text);
}

typedef struct {
    char *raw;
    char *content;
    char *reasoning;
    tool_calls calls;
    const char *finish;
} test_chat_turn;

static void test_chat_turn_free(test_chat_turn *turn) {
    if (!turn) return;
    free(turn->raw);
    free(turn->content);
    free(turn->reasoning);
    tool_calls_free(&turn->calls);
    memset(turn, 0, sizeof(*turn));
}

/* Run the same greedy stop/tool-marker/response parse path needed by the server,
 * while keeping one session alive so the next request genuinely continues from
 * this sampled turn. */
static bool test_generate_chat_turn(ds4_engine *engine, ds4_session *session,
                                    const request *r, test_chat_turn *turn) {
    memset(turn, 0, sizeof(*turn));
    char err[160] = {0};
    if (ds4_session_sync(session, &r->prompt, err, sizeof(err)) != 0) {
        fprintf(stderr, "ds4-test: tool-call sync failed: %s\n", err);
        turn->finish = "error";
        return false;
    }

    buf text = {0};
    uint64_t rng = 123;
    const char *finish = "length";
    bool saw_tool_start = false;
    bool saw_tool_end = false;
    bool decode_ok = true;

    for (int i = 0; i < r->max_tokens; i++) {
        int token = ds4_session_sample(session, r->temperature, r->top_k,
                                       r->top_p, r->min_p, &rng);
        if (ds4_token_is_stop_for_think_mode(engine, token, r->think_mode)) {
            finish = "stop";
            break;
        }
        if (ds4_session_eval(session, token, err, sizeof(err)) != 0) {
            finish = "error";
            decode_ok = false;
            break;
        }

        size_t piece_len = 0;
        char *piece = ds4_token_text(engine, token, &piece_len);
        buf_append(&text, piece, piece_len);
        free(piece);
        if (r->has_tools) {
            observe_tool_markers(text.ptr ? text.ptr : "",
                                 &saw_tool_start, &saw_tool_end, NULL);
            if (saw_tool_end) {
                finish = "tool_calls";
                break;
            }
        }
    }

    turn->raw = buf_take(&text);
    turn->finish = finish;
    if (!decode_ok) {
        fprintf(stderr, "ds4-test: tool-call decode failed: %s\n", err);
        return false;
    }

    bool recovered = false;
    bool parsed = parse_generated_message_for_response_for_syntax(
        r->model_syntax,
        turn->raw ? turn->raw : "",
        r->has_tools,
        saw_tool_start,
        ds4_think_mode_enabled(r->think_mode),
        &turn->finish,
        err,
        sizeof(err),
        &turn->content,
        &turn->reasoning,
        &turn->calls,
        &recovered);
    if (turn->calls.len > 0) turn->finish = "tool_calls";
    if (!parsed) {
        fprintf(stderr,
                "ds4-test: generated message parse failed: %s recovered=%d raw=%s\n",
                err, recovered ? 1 : 0, turn->raw ? turn->raw : "");
    }
    return parsed;
}

static void test_tool_call_quality_one(bool quality) {
    ds4_engine *engine = test_get_engine(quality);
    if (!engine) return;

    server s = {0};
    s.engine = engine;
    pthread_mutex_init(&s.tool_mu, NULL);

    request first_request = {0};
    request second_request = {0};
    ds4_session *session = NULL;
    test_chat_turn first = {0};
    test_chat_turn second = {0};
    char *second_body = NULL;
    char err[160] = {0};

    bool request_ok = parse_chat_request(engine, &s,
                                         test_tool_call_request_json(),
                                         512, 32768, &first_request,
                                         err, sizeof(err));
    TEST_ASSERT(request_ok);
    if (!request_ok) goto done;

    bool session_ok = ds4_session_create(&session, engine, 32768) == 0;
    TEST_ASSERT(session_ok);
    if (!session_ok) goto done;

    bool first_ok = test_generate_chat_turn(engine, session,
                                            &first_request, &first);
    TEST_ASSERT(first_ok);
    TEST_ASSERT(first.finish && !strcmp(first.finish, "tool_calls"));
    TEST_ASSERT(first.calls.len == 1);
    TEST_ASSERT(first.calls.len == 1 && first.calls.v[0].name &&
                !strcmp(first.calls.v[0].name, "list_files"));
    TEST_ASSERT(first.calls.raw_tool_text && first.calls.raw_tool_text[0]);
    if (!first_ok || first.calls.len != 1 ||
        !first.calls.v[0].name ||
        strcmp(first.calls.v[0].name, "list_files") ||
        !first.calls.raw_tool_text || !first.calls.raw_tool_text[0]) {
        goto done;
    }

    /* Use the real response-side id assignment and exact sampled-DSML memory.
     * The same id is serialized on both the assistant call and tool result. */
    assign_tool_call_ids(&s, &first.calls, API_OPENAI);
    TEST_ASSERT(first.calls.v[0].id && first.calls.v[0].id[0]);
    if (!first.calls.v[0].id || !first.calls.v[0].id[0]) goto done;
    tool_memory_remember(&s, &first.calls);

    second_body = test_tool_result_request_json(first.content,
                                                &first.calls.v[0]);
    TEST_ASSERT(second_body != NULL);
    if (!second_body) goto done;

    err[0] = '\0';
    request_ok = parse_chat_request(engine, &s, second_body,
                                    512, 32768, &second_request,
                                    err, sizeof(err));
    TEST_ASSERT(request_ok);
    if (!request_ok) goto done;
    TEST_ASSERT(second_request.tool_replay.mem == 1);
    TEST_ASSERT(second_request.tool_replay.disk == 0);
    TEST_ASSERT(second_request.tool_replay.canonical == 0);
    TEST_ASSERT(second_request.tool_replay.missing_ids == 0);

    bool second_ok = test_generate_chat_turn(engine, session,
                                             &second_request, &second);
    TEST_ASSERT(second_ok);
    TEST_ASSERT(second.finish && !strcmp(second.finish, "stop"));
    TEST_ASSERT(second.calls.len == 0);
    TEST_ASSERT(second.content && second.content[0]);

    fprintf(stderr,
            "ds4-test: post-tool-result turn1 finish_reason=%s tool_calls=%d "
            "turn2 finish_reason=%s tool_calls=%d replay_mem=%d\n",
            first.finish ? first.finish : "-", first.calls.len,
            second.finish ? second.finish : "-", second.calls.len,
            second_request.tool_replay.mem);

done:
    free(second_body);
    test_chat_turn_free(&second);
    test_chat_turn_free(&first);
    ds4_session_free(session);
    request_free(&second_request);
    request_free(&first_request);
    tool_memory_free(&s.tool_mem);
    pthread_mutex_destroy(&s.tool_mu);
}

static void test_tool_call_quality(void) {
    fprintf(stderr, "ds4-test: tool-call quality fast path\n");
    test_tool_call_quality_one(false);
    test_close_engine(false);
    fprintf(stderr, "ds4-test: tool-call quality exact path\n");
    test_tool_call_quality_one(true);
    test_close_engine(true);
}

/* Greedy speculative decode: capture committed tokens and the largest accepted
 * chunk, so the caller can confirm the multi-row verify path actually ran. */
static bool test_mtp_capture_speculative(ds4_engine *engine, const ds4_tokens *prompt,
                                         int max_tokens, int *out, int *out_len,
                                         int *max_chunk) {
    *out_len = 0;
    *max_chunk = 0;
    ds4_session *session = NULL;
    TEST_ASSERT(ds4_session_create(&session, engine, 32768) == 0);
    if (!session) return false;

    char err[160];
    bool ok = ds4_session_sync(session, prompt, err, sizeof(err)) == 0;
    TEST_ASSERT(ok);

    const int eos = ds4_token_eos(engine);
    int n = 0;
    bool stop = false;
    while (ok && !stop && n < max_tokens) {
        const int token = ds4_session_argmax(session);
        if (token == eos) break;

        int toks[17]; /* base token + draft depth, which the engine clamps to 16 */
        const int ntok = ds4_session_eval_speculative_argmax(
            session, token, max_tokens - n, eos, toks,
            (int)(sizeof(toks) / sizeof(toks[0])), err, sizeof(err));
        if (ntok < 0) { ok = false; TEST_ASSERT(false); break; }
        if (ntok > *max_chunk) *max_chunk = ntok;

        for (int j = 0; j < ntok; j++) {
            if (toks[j] == eos) { stop = true; break; }
            out[n++] = toks[j];
            if (n >= max_tokens) { stop = true; break; }
        }
    }

    *out_len = n;
    ds4_session_free(session);
    return ok;
}

/* Replay toks[] through plain decode and return the largest gap between a
 * position's argmax logit and the committed token's logit.  Correct speculation
 * commits (near-)argmax tokens (gap ~0); a mis-committed token gives a big gap. */
static bool test_mtp_worst_argmax_gap(ds4_engine *engine, const ds4_tokens *prompt,
                                      const int *toks, int n,
                                      float *worst_gap, int *worst_at) {
    *worst_gap = 0.0f;
    *worst_at = -1;
    ds4_session *session = NULL;
    TEST_ASSERT(ds4_session_create(&session, engine, 32768) == 0);
    if (!session) return false;

    char err[160];
    bool ok = ds4_session_sync(session, prompt, err, sizeof(err)) == 0;
    TEST_ASSERT(ok);

    for (int i = 0; ok && i < n; i++) {
        ds4_token_score best, cur;
        ok = ds4_session_top_logprobs(session, &best, 1) >= 1 &&
             ds4_session_token_logprob(session, toks[i], &cur) == 1;
        TEST_ASSERT(ok);
        if (!ok) break;

        const float gap = best.logit - cur.logit;
        if (gap > *worst_gap) { *worst_gap = gap; *worst_at = i; }
        if (ds4_session_eval(session, toks[i], err, sizeof(err)) != 0) { ok = false; TEST_ASSERT(false); break; }
    }

    ds4_session_free(session);
    return ok;
}

/* Verbatim-copy task: keeps the model confident (a mis-committed token shows as
 * a large argmax gap) and draft acceptance high (so the multi-row verify path is
 * exercised across the generation). */
static const char *test_mtp_copy_prompt(void) {
    return
        "Reproduce the following C code EXACTLY, character for character, "
        "inside a single code block and output nothing else:\n\n"
        "```c\n"
        "static uint32_t clamp_u32(uint32_t v, uint32_t lo, uint32_t hi) {\n"
        "    if (v < lo) return lo;\n"
        "    if (v > hi) return hi;\n"
        "    return v;\n"
        "}\n"
        "\n"
        "static uint32_t ring_advance(uint32_t pos, uint32_t cap) {\n"
        "    uint32_t next = pos + 1u;\n"
        "    return next >= cap ? 0u : next;\n"
        "}\n"
        "\n"
        "static int scratch_init(scratch *s, uint32_t ctx_size) {\n"
        "    if (ctx_size == 0u) ctx_size = 1u;\n"
        "    s->ctx_size = ctx_size;\n"
        "    s->comp_cap = ctx_size / 4u + 2u;\n"
        "    s->rows = clamp_u32(s->comp_cap, 1u, 4096u);\n"
        "    s->head = 0u;\n"
        "    return s->rows > 0u ? 0 : -1;\n"
        "}\n"
        "```\n";
}

#define TEST_MTP_MAXGEN 256
#define TEST_DSPARK_MAXGEN 128

static ds4_engine *test_open_dspark_engine(const char *support_path) {
    ds4_engine *engine = NULL;
    ds4_engine_options opt = {
        .model_path = test_model_path(),
#ifdef __APPLE__
        .backend = DS4_BACKEND_METAL,
#else
        .backend = DS4_BACKEND_CUDA,
#endif
        .quality = false,
        .ssd_streaming = test_env_bool("DS4_TEST_SSD_STREAMING"),
        .ssd_streaming_cold = test_env_bool("DS4_TEST_SSD_STREAMING_COLD"),
        .ssd_streaming_cache_experts =
            test_env_u32("DS4_TEST_SSD_STREAMING_CACHE_EXPERTS"),
        .ssd_streaming_cache_bytes =
            test_env_gib("DS4_TEST_SSD_STREAMING_CACHE_GB"),
        .ssd_streaming_preload_experts =
            test_env_u32("DS4_TEST_SSD_STREAMING_PRELOAD_EXPERTS"),
        .mtp_path = support_path,
        .mtp_draft_tokens = 0,
        .dspark = true,
        .dspark_confidence_threshold = 0.9f,
        .dspark_confidence_threshold_set = true,
    };
    const int rc = ds4_engine_open(&engine, &opt);
    TEST_ASSERT(rc == 0);
    return rc == 0 ? engine : NULL;
}

/* Regression for the swapped top-k arguments in metal_graph_verify_suffix_tops
 * at draft depth > 2.  Replays the committed speculative tokens through plain
 * decode and requires each to be a (near-)argmax: that is the verify invariant,
 * and unlike comparing token streams it tolerates the near-greedy tie
 * divergences.  Needs an MTP head, so it self-skips without DS4_TEST_MTP. */
static void test_mtp_verify_depth(void) {
    ds4_engine *engine = test_get_engine(false);
    if (!engine || !ds4_engine_has_mtp(engine)) {
        fprintf(stderr, "ds4-test: mtp-verify-depth skipped (set DS4_TEST_MTP to an MTP GGUF)\n");
        return;
    }
    TEST_ASSERT(ds4_engine_mtp_draft_tokens(engine) > 2);

    ds4_tokens prompt = {0};
    ds4_chat_begin(engine, &prompt);
    ds4_chat_append_message(engine, &prompt, "user", test_mtp_copy_prompt());
    ds4_chat_append_assistant_prefix(engine, &prompt, DS4_THINK_NONE);
    TEST_ASSERT(prompt.len > 0);

    int *spec = malloc((size_t)TEST_MTP_MAXGEN * sizeof(*spec));
    TEST_ASSERT(spec != NULL);
    if (spec && prompt.len > 0) {
        int nspec = 0, max_chunk = 0;
        const bool ok_spec = test_mtp_capture_speculative(engine, &prompt, TEST_MTP_MAXGEN,
                                                          spec, &nspec, &max_chunk);
        TEST_ASSERT(ok_spec);
        TEST_ASSERT(max_chunk > 1);  /* multi-token chunks committed: the multi-row path ran */
        TEST_ASSERT(nspec > 128);    /* enough output to surface the bug, incl. a spurious-EOS truncation */

        float worst_gap = 0.0f;
        int worst_at = -1;
        const bool ok_check = test_mtp_worst_argmax_gap(engine, &prompt, spec, nspec,
                                                        &worst_gap, &worst_at);
        TEST_ASSERT(ok_check);
        fprintf(stderr, "ds4-test: mtp-verify-depth nspec=%d max_chunk=%d worst_argmax_gap=%.3f at=%d\n",
                nspec, max_chunk, worst_gap, worst_at);
        TEST_ASSERT(worst_gap <= 2.0f);  /* correct: ~0; bug: ~21 on the reference model */
    }

    free(spec);
    ds4_tokens_free(&prompt);
}

/* Same invariant as the MTP depth smoke, but for the DSpark support model.  This
 * is separate from the fixture because it teacher-forces every committed token
 * through normal decode and directly checks that DSpark never commits a token
 * that was not near the target argmax. */
static void test_dspark_verify_depth(void) {
    const char *support = getenv("DS4_TEST_DSPARK");
    if (!support || !support[0]) {
        fprintf(stderr, "ds4-test: dspark-verify-depth skipped (set DS4_TEST_DSPARK to a DSpark support GGUF)\n");
        return;
    }

    ds4_engine *engine = test_open_dspark_engine(support);
    ds4_tokens prompt = {0};
    int *spec = NULL;

    if (engine) {
        const int draft_depth = ds4_engine_mtp_draft_tokens(engine);
        TEST_ASSERT(draft_depth > 2);

        ds4_chat_begin(engine, &prompt);
        ds4_chat_append_message(engine, &prompt, "user", test_mtp_copy_prompt());
        ds4_chat_append_assistant_prefix(engine, &prompt, DS4_THINK_NONE);
        TEST_ASSERT(prompt.len > 0);

        TEST_ASSERT(ds4_test_dspark_prefix_capture(engine, &prompt));

        spec = malloc((size_t)TEST_DSPARK_MAXGEN * sizeof(*spec));
        TEST_ASSERT(spec != NULL);
        if (draft_depth > 2 && spec && prompt.len > 0) {
            int nspec = 0, max_chunk = 0;
            const bool ok_spec = test_mtp_capture_speculative(engine, &prompt,
                                                              TEST_DSPARK_MAXGEN,
                                                              spec, &nspec,
                                                              &max_chunk);
            TEST_ASSERT(ok_spec);
            TEST_ASSERT(max_chunk > 1);
            TEST_ASSERT(nspec > 64);

            float worst_gap = 0.0f;
            int worst_at = -1;
            const bool ok_check = test_mtp_worst_argmax_gap(engine, &prompt,
                                                            spec, nspec,
                                                            &worst_gap,
                                                            &worst_at);
            TEST_ASSERT(ok_check);
            fprintf(stderr,
                    "ds4-test: dspark-verify-depth nspec=%d max_chunk=%d draft_depth=%d worst_argmax_gap=%.3f at=%d\n",
                    nspec, max_chunk, draft_depth, worst_gap, worst_at);
            TEST_ASSERT(worst_gap <= 2.0f);
        }
    }

    free(spec);
    ds4_tokens_free(&prompt);
    ds4_engine_close(engine);
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
    {"--session-snapshot", "session-snapshot", "session snapshot and recurrent-state round trip", test_session_snapshot_roundtrip},
    {"--long-context", "long-context", "long-context story fact-recall regression", test_long_story_fact_recall},
    {"--tool-call-quality", "tool-call-quality", "model tool call and post-result stop regression", test_tool_call_quality},
    {"--think-tool-recovery", "think-tool-recovery", "recover a complete tool call emitted inside unclosed reasoning", test_think_tool_recovery},
    {"--logprob-vectors", "logprob-vectors", "official API top-logprob vector comparison on the standard Metal path", test_official_logprob_vectors},
    {"--metal-ssd-streaming-cache-pressure", "metal-ssd-streaming-cache-pressure", "Metal SSD-streaming layer-batched decode cache-pressure repro for issue #384", test_metal_ssd_streaming_cache_pressure},
    {"--local-golden-vectors", "local-golden-vectors", "local top-k/logit drift regression for long Metal prefill", test_local_golden_vectors},
    {"--metal-short-prefill", "metal-short-prefill", "Metal ratio-4 short prefill regression", test_metal_short_prefill_ratio4},
    {"--glm53-continued-prefill", "glm53-continued-prefill", "GLM 5.3 resumed prefill latency, throughput, progress, and cold-path agreement", test_glm53_continued_prefill},
    {"--metal-kernels", "metal-kernels", "isolated Metal kernel numeric regressions", test_metal_kernel_group},
    {"--metal-tensor-equivalence", "metal-tensor-equivalence", "fast/quality Metal prompt-logit and greedy equivalence", test_metal_mpp_equivalence},
    {"--streaming-decode-prefill-correctness", "streaming-decode-prefill-correctness", "streaming decode-style cold prefill drift and repeatability", test_streaming_decode_prefill_correctness},
    {"--mtp-verify-depth", "mtp-verify-depth", "MTP speculative verify commits autoregressive-identical tokens at draft depth > 2", test_mtp_verify_depth},
    {"--dspark-verify-depth", "dspark-verify-depth", "DSpark speculative verify commits autoregressive-identical tokens at draft depth > 2", test_dspark_verify_depth},
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
#ifndef DS4_NO_GPU
    puts("  --metal-mpp-equivalence");
    puts("      Compatibility alias for --metal-tensor-equivalence.");
#endif
    puts("  -h, --help");
    puts("      Show this help.");
    puts("\nEnvironment:");
    puts("  DS4_TEST_MODEL=FILE        Model path. Default: ds4flash.gguf");
    puts("  DS4_TEST_BACKEND=cpu       Run model tests on CPU instead of Metal/CUDA.");
    puts("  DS4_TEST_SSD_STREAMING=1   Run model tests through backend SSD streaming.");
    puts("  DS4_TEST_SSD_STREAMING_CACHE_GB=N  Streaming routed expert cache in GiB.");
    puts("  DS4_TEST_SSD_STREAMING_CACHE_EXPERTS=N  Streaming routed expert cache count.");
    puts("  DS4_TEST_SSD_STREAMING_COLD=1  Skip streaming hot expert preload.");
    puts("  DS4_METAL_DISABLE_STREAMING_COLD_DECODE_PREFILL=1  Force canonical streamed cold prefill.");
    puts("  DS4_TEST_SNAPSHOT_PROMPT=FILE  Prompt for the session snapshot round trip.");
    puts("  DS4_TEST_SNAPSHOT_CTX=N        Context for the session snapshot round trip.");
    puts("  DS4_TEST_GLM_MTP=1             Include embedded GLM MTP in snapshot verification.");
    puts("  DS4_TEST_LONG_PROMPT=FILE  Rendered long-context story fact prompt.");
    puts("  DS4_TEST_VECTOR_FILE=FILE  Official fixture. Default: flash-0731/official.vec.");
    puts("  DS4_TEST_LOCAL_GOLDEN_FILE=FILE  Local fixture. Default: flash-0731/local-golden.vec.");
    puts("  DS4_TEST_MPP_EQ_CASE=NAME  Run only Tensor equivalence cases whose id contains NAME.");
    puts("  DS4_TEST_METAL_RESIDENT_ORACLES_ONLY=1  Restrict --metal-kernels to resident optimization oracles.");
    puts("  DS4_TEST_METAL_SMALL_PREFILL_TIMING=1   Include the small-prefill GPU-only microbenchmark.");
    puts("  DS4_TEST_MTP=FILE         Legacy MTP support GGUF for --mtp-verify-depth.");
    puts("  DS4_TEST_DSPARK=FILE      DSpark support GGUF for --dspark-verify-depth.");
    puts("  DS4_TEST_CONTINUED_PREFILL_TOKENS=N  Large suffix size for --glm53-continued-prefill.");
    puts("  DS4_TEST_CONTINUED_PREFILL_STEPS=N   Number of consecutive large suffixes to test.");
    puts("  DS4_TEST_CONTINUED_PREFILL_ALLOW_COARSE=1  Permit coarse short-suffix progress for baseline timing.");
}

static const ds4_test_entry *test_find_entry(const char *arg) {
#ifndef DS4_NO_GPU
    if (!strcmp(arg, "--metal-mpp-equivalence")) {
        arg = "--metal-tensor-equivalence";
    }
#endif
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
