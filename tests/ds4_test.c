#define DS4_SERVER_TEST
#define DS4_SERVER_TEST_NO_MAIN
#include "../ds4_server.c"
#ifndef DS4_NO_GPU
#include "../ds4_gpu.h"
#include <math.h>
#include <sys/wait.h>

static ds4_engine *test_engine_fast;
static ds4_engine *test_engine_quality;

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

static ds4_engine *test_open_engine(bool quality) {
    ds4_engine *engine = NULL;
    /* DS4_TEST_MTP loads the MTP head on the fast engine so the speculative
     * verify regression can reuse it; draft=4 hits the multi-row verify path. */
    const char *mtp = getenv("DS4_TEST_MTP");
    ds4_engine_options opt = {
        .model_path = test_model_path(),
#ifdef __APPLE__
        .backend = DS4_BACKEND_METAL,
#else
        .backend = DS4_BACKEND_CUDA,
#endif
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

static void test_fill_q8_0_weights(uint8_t *weights,
                                   uint32_t in_dim,
                                   uint32_t out_dim) {
    const uint32_t blocks = in_dim / 32u;
    const uint64_t row_bytes = (uint64_t)blocks * 34u;
    for (uint32_t o = 0; o < out_dim; o++) {
        uint8_t *row = weights + (uint64_t)o * row_bytes;
        for (uint32_t b = 0; b < blocks; b++) {
            float vals[32];
            float amax = 0.0f;
            for (uint32_t i = 0; i < 32; i++) {
                const uint32_t k = b * 32u + i;
                const int v = (int)((o * 17u + k * 23u + (o ^ k) * 3u) % 67u) - 33;
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

static void test_metal_glm_router_select_batch(void) {
    const uint32_t n_expert = 16;
    const uint32_t n_expert_used = 4;
    const uint32_t n_tokens = 2;
    const float scale = 1.25f;
    const uint64_t logits_bytes = (uint64_t)n_tokens * n_expert * sizeof(float);
    const uint64_t topk_bytes = (uint64_t)n_tokens * n_expert_used * sizeof(int32_t);
    const uint64_t weight_bytes = (uint64_t)n_tokens * n_expert_used * sizeof(float);
    const uint64_t bias_bytes = (uint64_t)n_expert * sizeof(float);
    const uint64_t model_alloc = test_round_up_u64(bias_bytes, (uint64_t)getpagesize());

    void *model_raw = NULL;
    TEST_ASSERT(posix_memalign(&model_raw, (size_t)getpagesize(), (size_t)model_alloc) == 0);
    if (!model_raw) return;

    float *bias = model_raw;
    memset(model_raw, 0, (size_t)model_alloc);
    for (uint32_t i = 0; i < n_expert; i++) {
        bias[i] = (float)((int)((i * 11u) % 9u) - 4) / 20.0f;
    }
    bias[3] += 1.0f;
    bias[10] += 0.9f;

    ds4_gpu_tensor *logits = ds4_gpu_tensor_alloc(logits_bytes);
    ds4_gpu_tensor *probs = ds4_gpu_tensor_alloc(logits_bytes);
    ds4_gpu_tensor *selected = ds4_gpu_tensor_alloc(topk_bytes);
    ds4_gpu_tensor *weights = ds4_gpu_tensor_alloc(weight_bytes);
    ds4_gpu_tensor *tokens = ds4_gpu_tensor_alloc((uint64_t)n_tokens * sizeof(int32_t));
    TEST_ASSERT(logits != NULL);
    TEST_ASSERT(probs != NULL);
    TEST_ASSERT(selected != NULL);
    TEST_ASSERT(weights != NULL);
    TEST_ASSERT(tokens != NULL);
    if (!logits || !probs || !selected || !weights || !tokens) {
        ds4_gpu_tensor_free(logits);
        ds4_gpu_tensor_free(probs);
        ds4_gpu_tensor_free(selected);
        ds4_gpu_tensor_free(weights);
        ds4_gpu_tensor_free(tokens);
        free(model_raw);
        return;
    }

    float h_logits[32];
    int32_t h_tokens[2] = { 7, 8 };
    for (uint32_t t = 0; t < n_tokens; t++) {
        for (uint32_t i = 0; i < n_expert; i++) {
            h_logits[(uint64_t)t * n_expert + i] =
                (float)((int)((t * 29u + i * 17u + (t ^ i) * 5u) % 43u) - 21) / 7.0f;
        }
    }

    TEST_ASSERT(ds4_gpu_tensor_write(logits, 0, h_logits, logits_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(tokens, 0, h_tokens, sizeof(h_tokens)) != 0);
    TEST_ASSERT(ds4_gpu_set_model_map(model_raw, model_alloc) != 0);
    ds4_gpu_set_quality(false);
    TEST_ASSERT(ds4_gpu_glm_router_select_batch_tensor(selected,
                                                       weights,
                                                       probs,
                                                       model_raw,
                                                       model_alloc,
                                                       0,
                                                       true,
                                                       logits,
                                                       tokens,
                                                       n_expert,
                                                       n_expert_used,
                                                       scale,
                                                       n_tokens) != 0);

    float h_probs[32];
    float h_weights[8];
    int32_t h_selected[8];
    TEST_ASSERT(ds4_gpu_tensor_read(probs, 0, h_probs, logits_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(selected, 0, h_selected, topk_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(weights, 0, h_weights, weight_bytes) != 0);

    for (uint32_t t = 0; t < n_tokens; t++) {
        float ref_probs[16];
        bool used[16] = { false };
        for (uint32_t i = 0; i < n_expert; i++) {
            const float logit = h_logits[(uint64_t)t * n_expert + i];
            ref_probs[i] = 1.0f / (1.0f + expf(-logit));
            TEST_ASSERT(fabsf(h_probs[(uint64_t)t * n_expert + i] - ref_probs[i]) < 1e-5f);
        }

        int32_t ref_selected[4];
        for (uint32_t k = 0; k < n_expert_used; k++) {
            float best = -FLT_MAX;
            int32_t best_i = -1;
            for (uint32_t i = 0; i < n_expert; i++) {
                const float score = ref_probs[i] + bias[i];
                if (!used[i] && score > best) {
                    best = score;
                    best_i = (int32_t)i;
                }
            }
            TEST_ASSERT(best_i >= 0);
            ref_selected[k] = best_i;
            used[best_i] = true;
        }

        float sum = 0.0f;
        for (uint32_t k = 0; k < n_expert_used; k++) {
            sum += ref_probs[ref_selected[k]];
        }
        if (sum < 6.103515625e-5f) sum = 6.103515625e-5f;

        for (uint32_t k = 0; k < n_expert_used; k++) {
            const uint64_t ix = (uint64_t)t * n_expert_used + k;
            TEST_ASSERT(h_selected[ix] == ref_selected[k]);
            const float ref_w = ref_probs[ref_selected[k]] / sum * scale;
            TEST_ASSERT(fabsf(h_weights[ix] - ref_w) < 1e-5f);
        }
    }

    ds4_gpu_tensor_free(logits);
    ds4_gpu_tensor_free(probs);
    ds4_gpu_tensor_free(selected);
    ds4_gpu_tensor_free(weights);
    ds4_gpu_tensor_free(tokens);
    free(model_raw);
}

static void test_metal_glm_kv_norm_copy(void) {
    const uint32_t kv_lora = 512;
    const uint32_t qk_rope = 64;
    const uint32_t rows = 2;
    const uint64_t row_elems = (uint64_t)kv_lora + qk_rope;
    const uint64_t elems = (uint64_t)rows * row_elems;
    const uint64_t x_bytes = elems * sizeof(float);
    const uint64_t weight_bytes = (uint64_t)kv_lora * sizeof(float);
    const uint64_t weight_alloc = test_round_up_u64(weight_bytes, (uint64_t)getpagesize());

    void *weights_raw = NULL;
    TEST_ASSERT(posix_memalign(&weights_raw, (size_t)getpagesize(), (size_t)weight_alloc) == 0);
    if (!weights_raw) return;
    float *weights = weights_raw;
    memset(weights_raw, 0, (size_t)weight_alloc);
    for (uint32_t i = 0; i < kv_lora; i++) {
        weights[i] = (float)((int)((i * 17u + 5u) % 31u) - 15) / 1024.0f;
    }

    float *x_host = malloc((size_t)x_bytes);
    float *out_host = malloc((size_t)x_bytes);
    TEST_ASSERT(x_host != NULL);
    TEST_ASSERT(out_host != NULL);
    if (!x_host || !out_host) {
        free(x_host);
        free(out_host);
        free(weights_raw);
        return;
    }
    for (uint64_t i = 0; i < elems; i++) {
        x_host[i] = (float)((int)((i * 13u + (i >> 3) * 7u) % 47u) - 23) / 64.0f;
    }
    memset(out_host, 0xA5, (size_t)x_bytes);

    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(x_bytes);
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(x_bytes);
    TEST_ASSERT(x != NULL);
    TEST_ASSERT(out != NULL);
    if (!x || !out) {
        ds4_gpu_tensor_free(x);
        ds4_gpu_tensor_free(out);
        free(x_host);
        free(out_host);
        free(weights_raw);
        return;
    }

    TEST_ASSERT(ds4_gpu_tensor_write(x, 0, x_host, x_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(out, 0, out_host, x_bytes) != 0);
    TEST_ASSERT(ds4_gpu_set_model_map(weights_raw, weight_alloc) != 0);
    ds4_gpu_set_quality(false);
    TEST_ASSERT(ds4_gpu_glm_kv_norm_copy_tensor(out,
                                                x,
                                                weights_raw,
                                                weight_alloc,
                                                0,
                                                kv_lora,
                                                qk_rope,
                                                rows,
                                                1.0e-5f) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(out, 0, out_host, x_bytes) != 0);

    float max_abs = 0.0f;
    float rms = 0.0f;
    for (uint32_t row = 0; row < rows; row++) {
        const float *src = x_host + (uint64_t)row * row_elems;
        const float *got_row = out_host + (uint64_t)row * row_elems;
        float sum = 0.0f;
        for (uint32_t i = 0; i < kv_lora; i++) sum += src[i] * src[i];
        const float scale = 1.0f / sqrtf(sum / (float)kv_lora + 1.0e-5f);
        for (uint32_t i = 0; i < kv_lora; i++) {
            const float ref = src[i] * scale * weights[i];
            const float got = got_row[i];
            TEST_ASSERT(isfinite(got));
            const float err = fabsf(got - ref);
            if (err > max_abs) max_abs = err;
            rms += err * err;
        }
        for (uint32_t i = 0; i < qk_rope; i++) {
            const float ref = src[kv_lora + i];
            const float got = got_row[kv_lora + i];
            TEST_ASSERT(isfinite(got));
            const float err = fabsf(got - ref);
            if (err > max_abs) max_abs = err;
            rms += err * err;
        }
    }
    rms = sqrtf(rms / (float)elems);
    TEST_ASSERT(max_abs < 2e-5f);
    TEST_ASSERT(rms < 2e-6f);

    ds4_gpu_tensor_free(x);
    ds4_gpu_tensor_free(out);
    free(x_host);
    free(out_host);
    free(weights_raw);
}


static void test_metal_glm_mla_attention(void) {
    const uint32_t n_tokens = 3;
    const uint32_t pos0 = 3;
    const uint32_t raw_cap = 8;
    const uint32_t n_head = 2;
    const uint32_t qk_nope = 16;
    const uint32_t qk_rope = 8;
    const uint32_t kv_lora = 12;
    const uint32_t v_head = 256;
    const uint64_t q_head_dim = (uint64_t)qk_nope + qk_rope;
    const uint64_t kv_width = (uint64_t)kv_lora + qk_rope;
    const uint64_t kvb_head_dim = (uint64_t)qk_nope + v_head;
    const uint64_t q_elems = (uint64_t)n_tokens * n_head * q_head_dim;
    const uint64_t raw_elems = (uint64_t)raw_cap * kv_width;
    const uint64_t kvb_elems = (uint64_t)raw_cap * n_head * kvb_head_dim;
    const uint64_t out_elems = (uint64_t)n_tokens * n_head * v_head;

    float *q_host = calloc((size_t)q_elems, sizeof(q_host[0]));
    float *raw_host = calloc((size_t)raw_elems, sizeof(raw_host[0]));
    uint16_t *kvb_host = calloc((size_t)kvb_elems, sizeof(kvb_host[0]));
    float *out_host = calloc((size_t)out_elems, sizeof(out_host[0]));
    TEST_ASSERT(q_host != NULL);
    TEST_ASSERT(raw_host != NULL);
    TEST_ASSERT(kvb_host != NULL);
    TEST_ASSERT(out_host != NULL);
    if (!q_host || !raw_host || !kvb_host || !out_host) {
        free(q_host);
        free(raw_host);
        free(kvb_host);
        free(out_host);
        return;
    }

    for (uint64_t i = 0; i < q_elems; i++) {
        q_host[i] = (float)((int)((i * 17u + 11u) % 29u) - 14) / 37.0f;
    }
    for (uint64_t i = 0; i < raw_elems; i++) {
        raw_host[i] = (float)((int)((i * 13u + 7u) % 31u) - 15) / 41.0f;
    }
    for (uint64_t i = 0; i < kvb_elems; i++) {
        float v = (float)((int)((i * 19u + 5u) % 43u) - 21) / 53.0f;
        kvb_host[i] = test_float_to_f16(v);
    }

    ds4_gpu_tensor *q = ds4_gpu_tensor_alloc(q_elems * sizeof(float));
    ds4_gpu_tensor *raw = ds4_gpu_tensor_alloc(raw_elems * sizeof(float));
    ds4_gpu_tensor *kvb = ds4_gpu_tensor_alloc(kvb_elems * sizeof(uint16_t));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(out_elems * sizeof(float));
    TEST_ASSERT(q != NULL);
    TEST_ASSERT(raw != NULL);
    TEST_ASSERT(kvb != NULL);
    TEST_ASSERT(out != NULL);
    if (!q || !raw || !kvb || !out) {
        ds4_gpu_tensor_free(q);
        ds4_gpu_tensor_free(raw);
        ds4_gpu_tensor_free(kvb);
        ds4_gpu_tensor_free(out);
        free(q_host);
        free(raw_host);
        free(kvb_host);
        free(out_host);
        return;
    }

    TEST_ASSERT(ds4_gpu_tensor_write(q, 0, q_host, q_elems * sizeof(float)) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(raw, 0, raw_host, raw_elems * sizeof(float)) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(kvb, 0, kvb_host, kvb_elems * sizeof(uint16_t)) != 0);
    ds4_gpu_set_quality(false);
    TEST_ASSERT(ds4_gpu_glm_mla_attention_tensor(out,
                                                 q,
                                                 raw,
                                                 kvb,
                                                 n_tokens,
                                                 pos0,
                                                 raw_cap,
                                                 n_head,
                                                 qk_nope,
                                                 qk_rope,
                                                 kv_lora,
                                                 v_head) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(out, 0, out_host, out_elems * sizeof(float)) != 0);

    const float scale = 1.0f / sqrtf((float)q_head_dim);
    for (uint32_t token = 0; token < n_tokens; token++) {
        const uint32_t pos = pos0 + token;
        const uint32_t n_ctx = pos + 1u < raw_cap ? pos + 1u : raw_cap;
        const uint32_t first = pos + 1u - n_ctx;
        for (uint32_t head = 0; head < n_head; head++) {
            float scores[8];
            float max_score = -FLT_MAX;
            const float *qh = q_host +
                    ((uint64_t)token * n_head + head) * q_head_dim;
            for (uint32_t j = 0; j < n_ctx; j++) {
                const uint32_t phys = (first + j) % raw_cap;
                const float *rope = raw_host + (uint64_t)phys * kv_width + kv_lora;
                const uint16_t *kvh = kvb_host +
                        ((uint64_t)phys * n_head + head) * kvb_head_dim;
                float dot = 0.0f;
                for (uint32_t i = 0; i < qk_nope; i++) {
                    dot += qh[i] * test_f16_to_f32(kvh[i]);
                }
                for (uint32_t i = 0; i < qk_rope; i++) {
                    dot += qh[qk_nope + i] * rope[i];
                }
                scores[j] = dot * scale;
                if (scores[j] > max_score) max_score = scores[j];
            }

            float denom = 0.0f;
            for (uint32_t j = 0; j < n_ctx; j++) denom += expf(scores[j] - max_score);
            TEST_ASSERT(denom > 0.0f);

            for (uint32_t i = 0; i < v_head; i++) {
                float ref = 0.0f;
                for (uint32_t j = 0; j < n_ctx; j++) {
                    const uint32_t phys = (first + j) % raw_cap;
                    const uint16_t *kvh = kvb_host +
                            ((uint64_t)phys * n_head + head) * kvb_head_dim;
                    ref += expf(scores[j] - max_score) / denom *
                           test_f16_to_f32(kvh[qk_nope + i]);
                }
                const float got = out_host[((uint64_t)token * n_head + head) * v_head + i];
                TEST_ASSERT(isfinite(got));
                TEST_ASSERT(fabsf(got - ref) < 2e-5f);
            }
        }
    }

    ds4_gpu_tensor_free(q);
    ds4_gpu_tensor_free(raw);
    ds4_gpu_tensor_free(kvb);
    ds4_gpu_tensor_free(out);
    free(q_host);
    free(raw_host);
    free(kvb_host);
    free(out_host);
}

static void test_metal_hy3_gqa_attention_case(uint32_t n_tokens,
                                              uint32_t pos0,
                                              uint32_t raw_cap,
                                              uint32_t n_raw,
                                              uint32_t raw_start,
                                              uint32_t n_head,
                                              uint32_t n_kv_head,
                                              uint32_t head_dim,
                                              int assert_wrap_on_last_token) {
    TEST_ASSERT(n_head > 0 && n_kv_head > 0 && head_dim > 0);
    TEST_ASSERT(n_head % n_kv_head == 0);
    const uint64_t kv_pack = (uint64_t)n_kv_head * head_dim;
    const uint64_t row_floats = 2u * kv_pack;
    const uint64_t q_elems = (uint64_t)n_tokens * n_head * head_dim;
    const uint64_t raw_elems = (uint64_t)raw_cap * row_floats;
    const uint64_t out_elems = q_elems;

    if (assert_wrap_on_last_token) {
        TEST_ASSERT(raw_start + n_raw > raw_cap);
    }

    float *q_host = calloc((size_t)q_elems, sizeof(q_host[0]));
    float *raw_host = calloc((size_t)raw_elems, sizeof(raw_host[0]));
    float *out_host = calloc((size_t)out_elems, sizeof(out_host[0]));
    float *scores = calloc((size_t)n_raw, sizeof(scores[0]));
    TEST_ASSERT(q_host != NULL);
    TEST_ASSERT(raw_host != NULL);
    TEST_ASSERT(out_host != NULL);
    TEST_ASSERT(scores != NULL);
    if (!q_host || !raw_host || !out_host || !scores) {
        free(q_host);
        free(raw_host);
        free(out_host);
        free(scores);
        return;
    }

    for (uint64_t i = 0; i < q_elems; i++) {
        q_host[i] = (float)((int)((i * 17u + 11u) % 29u) - 14) / 97.0f;
    }
    for (uint32_t phys = 0; phys < raw_cap; phys++) {
        for (uint32_t kv_head = 0; kv_head < n_kv_head; kv_head++) {
            float *kh = raw_host + (uint64_t)phys * row_floats +
                        (uint64_t)kv_head * head_dim;
            float *vh = raw_host + (uint64_t)phys * row_floats + kv_pack +
                        (uint64_t)kv_head * head_dim;
            for (uint32_t i = 0; i < head_dim; i++) {
                kh[i] = 0.25f * (float)(phys * 100u + kv_head * 10u + i) - 3.0f;
                vh[i] = 0.17f * (float)(phys * 70u + kv_head * 13u + i + 500u) + 1.5f;
            }
        }
    }

    ds4_gpu_tensor *q = ds4_gpu_tensor_alloc(q_elems * sizeof(float));
    ds4_gpu_tensor *raw = ds4_gpu_tensor_alloc(raw_elems * sizeof(float));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(out_elems * sizeof(float));
    TEST_ASSERT(q != NULL);
    TEST_ASSERT(raw != NULL);
    TEST_ASSERT(out != NULL);
    if (!q || !raw || !out) {
        ds4_gpu_tensor_free(q);
        ds4_gpu_tensor_free(raw);
        ds4_gpu_tensor_free(out);
        free(q_host);
        free(raw_host);
        free(out_host);
        free(scores);
        return;
    }

    TEST_ASSERT(ds4_gpu_tensor_write(q, 0, q_host, q_elems * sizeof(float)) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(raw, 0, raw_host, raw_elems * sizeof(float)) != 0);
    ds4_gpu_set_quality(false);
    TEST_ASSERT(ds4_gpu_hy3_gqa_attention_tensor(out,
                                                 q,
                                                 raw,
                                                 n_tokens,
                                                 pos0,
                                                 n_raw,
                                                 raw_cap,
                                                 raw_start,
                                                 n_head,
                                                 n_kv_head,
                                                 head_dim) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(out, 0, out_host, out_elems * sizeof(float)) != 0);

    const float scale = 1.0f / sqrtf((float)head_dim);
    const uint32_t q_per_kv = n_head / n_kv_head;
    const uint32_t last_pos = pos0 + n_tokens - 1u;
    const uint32_t first_raw_pos = last_pos + 1u - n_raw;
    for (uint32_t token = 0; token < n_tokens; token++) {
        const uint32_t qpos = pos0 + token;
        for (uint32_t head = 0; head < n_head; head++) {
            const uint32_t kv_head = head / q_per_kv;
            const float *qh = q_host + ((uint64_t)token * n_head + head) * head_dim;
            float max_score = -FLT_MAX;
            uint32_t n_ctx = 0;
            for (uint32_t pos = first_raw_pos; pos <= qpos; pos++) {
                const uint32_t logical = pos - first_raw_pos;
                const uint32_t phys = (raw_start + logical) % raw_cap;
                TEST_ASSERT(logical < n_raw);
                if (assert_wrap_on_last_token && token == n_tokens - 1u &&
                    pos >= first_raw_pos + 3u) {
                    TEST_ASSERT(phys < raw_start);
                }
                const float *kh = raw_host + (uint64_t)phys * row_floats +
                                  (uint64_t)kv_head * head_dim;
                float dot = 0.0f;
                for (uint32_t i = 0; i < head_dim; i++) dot += qh[i] * kh[i];
                scores[n_ctx] = dot * scale;
                if (scores[n_ctx] > max_score) max_score = scores[n_ctx];
                n_ctx++;
            }
            TEST_ASSERT(n_ctx > 0 && n_ctx <= n_raw);
            float denom = 0.0f;
            for (uint32_t j = 0; j < n_ctx; j++) denom += expf(scores[j] - max_score);
            TEST_ASSERT(denom > 0.0f);
            for (uint32_t i = 0; i < head_dim; i++) {
                float ref = 0.0f;
                for (uint32_t j = 0; j < n_ctx; j++) {
                    const uint32_t pos = first_raw_pos + j;
                    const uint32_t logical = pos - first_raw_pos;
                    const uint32_t phys = (raw_start + logical) % raw_cap;
                    const float *vh = raw_host + (uint64_t)phys * row_floats + kv_pack +
                                      (uint64_t)kv_head * head_dim;
                    ref += expf(scores[j] - max_score) / denom * vh[i];
                }
                const float got = out_host[((uint64_t)token * n_head + head) * head_dim + i];
                TEST_ASSERT(isfinite(got));
                TEST_ASSERT(fabsf(got - ref) < 1e-4f);
            }
        }
    }

    ds4_gpu_tensor_free(q);
    ds4_gpu_tensor_free(raw);
    ds4_gpu_tensor_free(out);
    free(q_host);
    free(raw_host);
    free(out_host);
    free(scores);
}

static void test_metal_hy3_gqa_attention(void) {
    const uint32_t n_tokens = 2;
    const uint32_t pos0 = 7;
    const uint32_t raw_cap = 8;
    const uint32_t n_raw = 6;
    const uint32_t raw_start = 5;

    /* Ring must wrap: raw_start + n_raw > raw_cap so logical rows map past the end. */
    test_metal_hy3_gqa_attention_case(n_tokens,
                                      pos0,
                                      raw_cap,
                                      n_raw,
                                      raw_start,
                                      4,
                                      2,
                                      32,
                                      1);

    /* Production HY3 GQA: 64 query heads, 8 KV heads, 128-wide rows (GQA broadcast). */
    test_metal_hy3_gqa_attention_case(n_tokens,
                                      pos0,
                                      raw_cap,
                                      n_raw,
                                      raw_start,
                                      64,
                                      8,
                                      128,
                                      1);
}




static void test_ref_rope_tail_split_half_batch(
        float *x,
        uint32_t n_tok,
        uint32_t n_head,
        uint32_t head_dim,
        uint32_t n_rot,
        uint32_t pos0,
        float freq_base,
        float freq_scale,
        bool inverse) {
    const uint32_t n_nope = head_dim - n_rot;
    const uint32_t half = n_rot / 2u;
    const float theta_scale = powf(freq_base, -2.0f / (float)n_rot);
    const float sin_sign = inverse ? -1.0f : 1.0f;
    const uint64_t tok_stride = (uint64_t)n_head * head_dim;

    for (uint32_t t = 0; t < n_tok; t++) {
        const uint32_t pos = pos0 + t;
        float *tok = x + (uint64_t)t * tok_stride;
        for (uint32_t h = 0; h < n_head; h++) {
            float *tail = tok + (uint64_t)h * head_dim + n_nope;
            float theta_extrap = (float)pos;
            for (uint32_t i = 0; i < half; i++) {
                const float theta = freq_scale * theta_extrap;
                const float c = cosf(theta);
                const float s = sin_sign * sinf(theta);
                const float x0 = tail[i];
                const float x1 = tail[i + half];
                tail[i]        = x0 * c - x1 * s;
                tail[i + half] = x0 * s + x1 * c;
                theta_extrap *= theta_scale;
            }
        }
    }
}

static void test_metal_rope_split_half(void) {
    const uint32_t n_tokens = 3;
    const uint32_t n_head = 4;
    const uint32_t n_head_kv = 2;
    const uint32_t head_dim = 32;
    const uint32_t n_rot = 16;
    const uint32_t pos0 = 17;
    const float freq_base = 10000.0f;
    const float freq_scale = 1.0f;
    const float beta_fast = 32.0f;
    const float beta_slow = 1.0f;
    const uint64_t q_elems = (uint64_t)n_tokens * n_head * head_dim;
    const uint64_t kv_elems = (uint64_t)n_tokens * n_head_kv * head_dim;
    const uint64_t q_bytes = q_elems * sizeof(float);
    const uint64_t kv_bytes = kv_elems * sizeof(float);

    float *q_host = malloc((size_t)q_bytes);
    float *kv_host = malloc((size_t)kv_bytes);
    float *q_ref = malloc((size_t)q_bytes);
    float *kv_ref = malloc((size_t)kv_bytes);
    float *q_gpu = malloc((size_t)q_bytes);
    float *kv_gpu = malloc((size_t)kv_bytes);
    TEST_ASSERT(q_host && kv_host && q_ref && kv_ref && q_gpu && kv_gpu);
    if (!q_host || !kv_host || !q_ref || !kv_ref || !q_gpu || !kv_gpu) {
        free(q_host);
        free(kv_host);
        free(q_ref);
        free(kv_ref);
        free(q_gpu);
        free(kv_gpu);
        return;
    }

    for (uint64_t i = 0; i < q_elems; i++) {
        q_host[i] = (float)((int)((i * 19u + (i >> 2) * 11u) % 53u) - 26) / 37.0f;
    }
    for (uint64_t i = 0; i < kv_elems; i++) {
        kv_host[i] = (float)((int)((i * 23u + (i >> 1) * 7u) % 41u) - 20) / 29.0f;
    }

    memcpy(q_ref, q_host, (size_t)q_bytes);
    memcpy(kv_ref, kv_host, (size_t)kv_bytes);
    test_ref_rope_tail_split_half_batch(q_ref,
                                        n_tokens,
                                        n_head,
                                        head_dim,
                                        n_rot,
                                        pos0,
                                        freq_base,
                                        freq_scale,
                                        false);
    test_ref_rope_tail_split_half_batch(kv_ref,
                                        n_tokens,
                                        n_head_kv,
                                        head_dim,
                                        n_rot,
                                        pos0,
                                        freq_base,
                                        freq_scale,
                                        false);

    ds4_gpu_tensor *q = ds4_gpu_tensor_alloc(q_bytes);
    ds4_gpu_tensor *kv = ds4_gpu_tensor_alloc(kv_bytes);
    TEST_ASSERT(q != NULL);
    TEST_ASSERT(kv != NULL);
    if (!q || !kv) {
        ds4_gpu_tensor_free(q);
        ds4_gpu_tensor_free(kv);
        free(q_host);
        free(kv_host);
        free(q_ref);
        free(kv_ref);
        free(q_gpu);
        free(kv_gpu);
        return;
    }

    TEST_ASSERT(ds4_gpu_tensor_write(q, 0, q_host, q_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(kv, 0, kv_host, kv_bytes) != 0);
    ds4_gpu_set_quality(false);
    TEST_ASSERT(ds4_gpu_rope_tail_split_half_tensor(q,
                                                    n_tokens,
                                                    n_head,
                                                    head_dim,
                                                    n_rot,
                                                    pos0,
                                                    0,
                                                    false,
                                                    freq_base,
                                                    freq_scale,
                                                    0.0f,
                                                    1.0f,
                                                    beta_fast,
                                                    beta_slow) != 0);
    TEST_ASSERT(ds4_gpu_rope_tail_split_half_tensor(kv,
                                                    n_tokens,
                                                    n_head_kv,
                                                    head_dim,
                                                    n_rot,
                                                    pos0,
                                                    0,
                                                    false,
                                                    freq_base,
                                                    freq_scale,
                                                    0.0f,
                                                    1.0f,
                                                    beta_fast,
                                                    beta_slow) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(q, 0, q_gpu, q_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(kv, 0, kv_gpu, kv_bytes) != 0);

    float max_abs = 0.0f;
    float rms = 0.0f;
    uint64_t rot_count = 0;
    for (uint64_t i = 0; i < q_elems; i++) {
        const uint32_t d = (uint32_t)(i % head_dim);
        const float ref = q_ref[i];
        const float got = q_gpu[i];
        TEST_ASSERT(isfinite(got));
        if (d < head_dim - n_rot) {
            TEST_ASSERT(fabsf(got - ref) < 1.0e-6f);
        } else {
            const float err = fabsf(got - ref);
            if (err > max_abs) max_abs = err;
            rms += err * err;
            rot_count++;
        }
    }
    TEST_ASSERT(rot_count > 0);
    rms = sqrtf(rms / (float)rot_count);
    TEST_ASSERT(max_abs < 2.0e-4f);
    TEST_ASSERT(rms < 5.0e-5f);

    max_abs = 0.0f;
    rms = 0.0f;
    rot_count = 0;
    for (uint64_t i = 0; i < kv_elems; i++) {
        const uint32_t d = (uint32_t)(i % head_dim);
        const float ref = kv_ref[i];
        const float got = kv_gpu[i];
        TEST_ASSERT(isfinite(got));
        if (d < head_dim - n_rot) {
            TEST_ASSERT(fabsf(got - ref) < 1.0e-6f);
        } else {
            const float err = fabsf(got - ref);
            if (err > max_abs) max_abs = err;
            rms += err * err;
            rot_count++;
        }
    }
    TEST_ASSERT(rot_count > 0);
    rms = sqrtf(rms / (float)rot_count);
    TEST_ASSERT(max_abs < 2.0e-4f);
    TEST_ASSERT(rms < 5.0e-5f);

    ds4_gpu_tensor_free(q);
    ds4_gpu_tensor_free(kv);
    free(q_host);
    free(kv_host);
    free(q_ref);
    free(kv_ref);
    free(q_gpu);
    free(kv_gpu);
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

static void test_metal_f16_prefill_matmul_short_batch(void) {
    const uint32_t in_dim = 6144;
    const uint32_t out_dim = 256;
    const uint32_t n_tok = 16;
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
            const int v = (int)((o * 7u + i * 19u + (o ^ (i * 5u))) % 67u) - 33;
            weights[(uint64_t)o * in_dim + i] = test_float_to_f16((float)v / 128.0f);
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
            const int v = (int)((t * 17u + i * 11u + (t ^ i) * 7u) % 83u) - 41;
            x_host[(uint64_t)t * in_dim + i] = (float)v / 96.0f;
        }
    }
    memset(out_host, 0xA5, (size_t)out_bytes);

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
    test_fill_q8_0_weights(weights, in_dim, out_dim);

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

static void test_metal_q8_0_prefill_matmul_short_batch(void) {
    const uint32_t in_dim = 6144;
    const uint32_t out_dim = 2048;
    const uint32_t n_tok = 16;
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
    test_fill_q8_0_weights(weights, in_dim, out_dim);

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
            const int v = (int)((t * 23u + i * 5u + (t ^ (i * 3u))) % 79u) - 39;
            x_host[(uint64_t)t * in_dim + i] = (float)v / 96.0f;
        }
    }
    memset(out_host, 0xA5, (size_t)out_bytes);

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

static void test_metal_q8_0_large_output_matvec_nsg8(void) {
    const uint32_t in_dim = 512;
    const uint32_t out_dim = 65568;
    const uint32_t n_tok = 1;
    const uint64_t row_bytes = (uint64_t)(in_dim / 32u) * 34u;
    const uint64_t weight_bytes = (uint64_t)out_dim * row_bytes;
    const uint64_t weight_alloc = test_round_up_u64(weight_bytes, (uint64_t)getpagesize());
    const uint64_t x_bytes = (uint64_t)in_dim * sizeof(float);
    const uint64_t out_bytes = (uint64_t)out_dim * sizeof(float);

    void *weights_raw = NULL;
    TEST_ASSERT(posix_memalign(&weights_raw, (size_t)getpagesize(), (size_t)weight_alloc) == 0);
    if (!weights_raw) return;

    uint8_t *weights = weights_raw;
    memset(weights, 0, (size_t)weight_alloc);
    test_fill_q8_0_weights(weights, in_dim, out_dim);

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

    for (uint32_t i = 0; i < in_dim; i++) {
        const int v = (int)((i * 37u + (i >> 1) * 11u) % 97u) - 48;
        x_host[i] = (float)v / 96.0f;
    }
    memset(out_host, 0xA5, (size_t)out_bytes);

    TEST_ASSERT(ds4_gpu_tensor_write(x, 0, x_host, x_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(out, 0, out_host, out_bytes) != 0);
    TEST_ASSERT(ds4_gpu_set_model_map(weights_raw, weight_alloc) != 0);
    ds4_gpu_set_quality(false);
    TEST_ASSERT(ds4_gpu_matmul_q8_0_tensor(out, weights_raw, weight_alloc, 0,
                                           in_dim, out_dim, x, n_tok) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(out, 0, out_host, out_bytes) != 0);

    float max_abs = 0.0f;
    float rms = 0.0f;
    for (uint32_t o = 0; o < out_dim; o++) {
        const uint8_t *row = weights + (uint64_t)o * row_bytes;
        float ref = 0.0f;
        for (uint32_t b = 0; b < in_dim / 32u; b++) {
            uint16_t scale_bits;
            memcpy(&scale_bits, row + b * 34u, sizeof(scale_bits));
            const float scale = test_f16_to_f32(scale_bits);
            const int8_t *qs = (const int8_t *)(row + b * 34u + 2u);
            for (uint32_t i = 0; i < 32; i++) {
                ref += scale * (float)qs[i] * x_host[b * 32u + i];
            }
        }
        const float got = out_host[o];
        TEST_ASSERT(isfinite(got));
        const float err = fabsf(got - ref);
        if (err > max_abs) max_abs = err;
        rms += err * err;
    }
    rms = sqrtf(rms / (float)out_dim);
    TEST_ASSERT(max_abs < 0.08f);
    TEST_ASSERT(rms < 0.02f);

    free(x_host);
    free(out_host);
    ds4_gpu_tensor_free(x);
    ds4_gpu_tensor_free(out);
    free(weights_raw);
}

enum {
    TEST_QK_K = 256,
    TEST_MAX_EXPERT_USED = 8,
    TEST_TENSOR_Q2_K = 10,
    TEST_TENSOR_IQ2_XXS = 16,
};

typedef struct {
    uint8_t scales[TEST_QK_K / 16];
    uint8_t qs[TEST_QK_K / 4];
    uint16_t d;
    uint16_t dmin;
} test_block_q2_K;

typedef struct {
    uint16_t d;
    uint16_t qs[TEST_QK_K / 8];
} test_block_iq2_xxs;

typedef char test_block_q2_k_size[(sizeof(test_block_q2_K) == 84) ? 1 : -1];
typedef char test_block_iq2_xxs_size[(sizeof(test_block_iq2_xxs) == 66) ? 1 : -1];


static void test_fill_iq2_xxs_blocks(test_block_iq2_xxs *blocks,
                                     uint32_t n_blocks,
                                     bool nonzero) {
    for (uint32_t i = 0; i < n_blocks; i++) {
        blocks[i].d = nonzero ? 0x3c00u : 0u; /* half(1.0) */
        for (uint32_t q = 0; q < TEST_QK_K / 8; q++) {
            blocks[i].qs[q] = nonzero ? (uint16_t)(0x1111u + (q & 3u)) : 0u;
        }
    }
}

static void test_fill_q2_k_blocks(test_block_q2_K *blocks,
                                  uint32_t n_blocks,
                                  bool nonzero) {
    for (uint32_t i = 0; i < n_blocks; i++) {
        memset(blocks[i].scales, nonzero ? 0x11 : 0x00, sizeof(blocks[i].scales));
        memset(blocks[i].qs, nonzero ? 0x55 : 0x00, sizeof(blocks[i].qs));
        blocks[i].d = nonzero ? 0x3c00u : 0u; /* half(1.0) */
        blocks[i].dmin = 0u;
    }
}

static void test_metal_iq2_slots8_direct_sum_does_not_resum_scratch(void) {
    const uint32_t n_total_expert = 128;
    const uint32_t n_expert_used = TEST_MAX_EXPERT_USED;
    const uint32_t in_dim = TEST_QK_K;
    const uint32_t mid_dim = TEST_QK_K;
    const uint32_t out_dim = TEST_QK_K;
    const uint64_t gate_row_bytes = sizeof(test_block_iq2_xxs);
    const uint64_t down_row_bytes = sizeof(test_block_q2_K);
    const uint64_t gate_expert_bytes = (uint64_t)mid_dim * gate_row_bytes;
    const uint64_t down_expert_bytes = (uint64_t)out_dim * down_row_bytes;
    const uint64_t gate_tensor_bytes = (uint64_t)n_total_expert * gate_expert_bytes;
    const uint64_t down_tensor_bytes = (uint64_t)n_total_expert * down_expert_bytes;
    const uint64_t model_bytes = gate_tensor_bytes * 2u + down_tensor_bytes;
    const uint64_t model_alloc = test_round_up_u64(model_bytes, (uint64_t)getpagesize());
    const uint64_t pair_elems = (uint64_t)n_expert_used * mid_dim;
    const uint64_t out_bytes = (uint64_t)out_dim * sizeof(float);

    void *model_raw = NULL;
    TEST_ASSERT(posix_memalign(&model_raw, (size_t)getpagesize(), (size_t)model_alloc) == 0);
    if (!model_raw) return;
    memset(model_raw, 0, (size_t)model_alloc);

    float *x_host = malloc((size_t)in_dim * sizeof(float));
    float *out_host = malloc((size_t)out_bytes);
    int32_t selected_host[TEST_MAX_EXPERT_USED];
    float weights_host[TEST_MAX_EXPERT_USED];
    TEST_ASSERT(x_host != NULL);
    TEST_ASSERT(out_host != NULL);
    if (!x_host || !out_host) {
        free(x_host);
        free(out_host);
        free(model_raw);
        return;
    }
    for (uint32_t i = 0; i < in_dim; i++) {
        x_host[i] = (float)((int)(i % 17u) - 8) / 8.0f;
    }
    for (uint32_t i = 0; i < n_expert_used; i++) {
        selected_host[i] = (int32_t)i;
        weights_host[i] = 1.0f;
    }

    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(out_bytes);
    ds4_gpu_tensor *gate = ds4_gpu_tensor_alloc(pair_elems * sizeof(float));
    ds4_gpu_tensor *up = ds4_gpu_tensor_alloc(pair_elems * sizeof(float));
    ds4_gpu_tensor *mid = ds4_gpu_tensor_alloc(pair_elems * sizeof(float));
    ds4_gpu_tensor *experts = ds4_gpu_tensor_alloc((uint64_t)n_expert_used * out_dim * sizeof(float));
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc((uint64_t)in_dim * sizeof(float));
    ds4_gpu_tensor *selected = ds4_gpu_tensor_alloc((uint64_t)n_expert_used * sizeof(int32_t));
    ds4_gpu_tensor *weights = ds4_gpu_tensor_alloc((uint64_t)n_expert_used * sizeof(float));
    TEST_ASSERT(out && gate && up && mid && experts && x && selected && weights);
    if (!out || !gate || !up || !mid || !experts || !x || !selected || !weights) {
        ds4_gpu_tensor_free(out);
        ds4_gpu_tensor_free(gate);
        ds4_gpu_tensor_free(up);
        ds4_gpu_tensor_free(mid);
        ds4_gpu_tensor_free(experts);
        ds4_gpu_tensor_free(x);
        ds4_gpu_tensor_free(selected);
        ds4_gpu_tensor_free(weights);
        free(x_host);
        free(out_host);
        free(model_raw);
        return;
    }

    TEST_ASSERT(ds4_gpu_tensor_write(x, 0, x_host, (uint64_t)in_dim * sizeof(float)) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(selected, 0, selected_host, sizeof(selected_host)) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(weights, 0, weights_host, sizeof(weights_host)) != 0);
    TEST_ASSERT(ds4_gpu_set_model_map(model_raw, model_alloc) != 0);
    ds4_gpu_set_quality(false);
    ds4_gpu_set_ssd_streaming(true);

    test_block_iq2_xxs *gate_blocks = (test_block_iq2_xxs *)model_raw;
    test_block_iq2_xxs *up_blocks = (test_block_iq2_xxs *)((uint8_t *)model_raw + gate_tensor_bytes);
    test_block_q2_K *down_blocks = (test_block_q2_K *)((uint8_t *)model_raw + gate_tensor_bytes * 2u);
    const uint32_t gate_blocks_total = (uint32_t)(gate_tensor_bytes / sizeof(test_block_iq2_xxs));
    const uint32_t down_blocks_total = (uint32_t)(down_tensor_bytes / sizeof(test_block_q2_K));
    test_fill_iq2_xxs_blocks(gate_blocks, gate_blocks_total, true);
    test_fill_iq2_xxs_blocks(up_blocks, gate_blocks_total, true);
    test_fill_q2_k_blocks(down_blocks, down_blocks_total, true);

    char *saved_disable = test_save_env("DS4_METAL_DISABLE_IQ2_SELECTED_EXPERT_VIEWS");
    setenv("DS4_METAL_DISABLE_IQ2_SELECTED_EXPERT_VIEWS", "1", 1);
    bool mid_is_f16 = false;
    TEST_ASSERT(ds4_gpu_routed_moe_batch_tensor(out,
                                                gate,
                                                up,
                                                mid,
                                                experts,
                                                model_raw,
                                                model_alloc,
                                                0,
                                                gate_tensor_bytes,
                                                gate_tensor_bytes * 2u,
                                                TEST_TENSOR_IQ2_XXS,
                                                TEST_TENSOR_Q2_K,
                                                gate_expert_bytes,
                                                gate_row_bytes,
                                                down_expert_bytes,
                                                down_row_bytes,
                                                in_dim,
                                                mid_dim,
                                                out_dim,
                                                selected,
                                                weights,
                                                n_total_expert,
                                                n_expert_used,
                                                10.0f,
                                                x,
                                                3,
                                                1,
                                                &mid_is_f16) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(out, 0, out_host, out_bytes) != 0);
    float warm_max = 0.0f;
    for (uint32_t i = 0; i < out_dim; i++) {
        const float v = fabsf(out_host[i]);
        if (v > warm_max) warm_max = v;
    }
    TEST_ASSERT(warm_max > 1.0e-5f);

    test_restore_env("DS4_METAL_DISABLE_IQ2_SELECTED_EXPERT_VIEWS", saved_disable);
    test_fill_iq2_xxs_blocks(gate_blocks, gate_blocks_total, false);
    test_fill_iq2_xxs_blocks(up_blocks, gate_blocks_total, false);
    test_fill_q2_k_blocks(down_blocks, down_blocks_total, false);
    memset(out_host, 0xA5, (size_t)out_bytes);
    TEST_ASSERT(ds4_gpu_tensor_write(out, 0, out_host, out_bytes) != 0);
    TEST_ASSERT(ds4_gpu_routed_moe_batch_tensor(out,
                                                gate,
                                                up,
                                                mid,
                                                experts,
                                                model_raw,
                                                model_alloc,
                                                0,
                                                gate_tensor_bytes,
                                                gate_tensor_bytes * 2u,
                                                TEST_TENSOR_IQ2_XXS,
                                                TEST_TENSOR_Q2_K,
                                                gate_expert_bytes,
                                                gate_row_bytes,
                                                down_expert_bytes,
                                                down_row_bytes,
                                                in_dim,
                                                mid_dim,
                                                out_dim,
                                                selected,
                                                weights,
                                                n_total_expert,
                                                n_expert_used,
                                                10.0f,
                                                x,
                                                3,
                                                1,
                                                &mid_is_f16) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(out, 0, out_host, out_bytes) != 0);
    float zero_max = 0.0f;
    for (uint32_t i = 0; i < out_dim; i++) {
        TEST_ASSERT(isfinite(out_host[i]));
        const float v = fabsf(out_host[i]);
        if (v > zero_max) zero_max = v;
    }
    TEST_ASSERT(zero_max < 1.0e-7f);

    ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(gate);
    ds4_gpu_tensor_free(up);
    ds4_gpu_tensor_free(mid);
    ds4_gpu_tensor_free(experts);
    ds4_gpu_tensor_free(x);
    ds4_gpu_tensor_free(selected);
    ds4_gpu_tensor_free(weights);
    free(x_host);
    free(out_host);
    free(model_raw);
}




static void test_metal_kernel_group(void) {
    test_metal_f16_matvec_fast_nr0_4();
    test_metal_f16_prefill_matmul();
    test_metal_f16_prefill_matmul_short_batch();
    test_metal_q8_0_prefill_matmul();
    test_metal_q8_0_prefill_matmul_short_batch();
    test_metal_q8_0_large_output_matvec_nsg8();
    test_metal_glm_router_select_batch();
    test_metal_iq2_slots8_direct_sum_does_not_resum_scratch();
    test_metal_glm_kv_norm_copy();
    test_metal_glm_mla_attention();
    test_metal_hy3_gqa_attention();
    test_metal_rope_split_half();
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
        if (ds4_is_stop_token(engine, token)) break;

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
        if (test_logprob_vector_case_disabled(&vc)) {
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
    if (!path || !path[0]) path = "tests/test-vectors/local-golden.vec";
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
    ds4_engine *engine = test_get_engine(false);
    if (!engine) return;

    request r;
    char err[160];
    TEST_ASSERT(parse_chat_request(engine, NULL, test_think_recovery_request_json(),
                                   512, 32768, &r, err, sizeof(err)));

    ds4_session *session = NULL;
    TEST_ASSERT(ds4_session_create(&session, engine, 32768) == 0);
    if (!session) {
        request_free(&r);
        return;
    }
    TEST_ASSERT(ds4_session_sync(session, &r.prompt, err, sizeof(err)) == 0);

    if (getenv("DS4_TEST_RECOVERY_PROBE") != NULL) {
        /* Diagnostic: print the model's natural tool-call turn for this
         * request instead of running the recovery. */
        buf nat = {0};
        uint64_t prng = 7;
        for (int i = 0; i < 300; i++) {
            int token = ds4_session_sample(session, 0.0f, 0, 1.0f, 0.0f, &prng);
            if (ds4_is_stop_token(engine, token)) break;
            size_t plen = 0;
            char *p = ds4_token_text(engine, token, &plen);
            buf_append(&nat, p, plen);
            free(p);
            bool ps = false, pe = false;
            observe_tool_markers(nat.ptr, &ps, &pe, NULL);
            if (pe) break;
            if (ds4_session_eval(session, token, err, sizeof(err)) != 0) break;
        }
        fprintf(stderr, "ds4-test: natural turn=[%s]\n", nat.ptr ? nat.ptr : "");
        buf_free(&nat);
        ds4_session_free(session);
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
        "list_files tool right away.\n\n" DS4_TOOL_CALLS_START;
    buf_append(&forced, body, strlen(body));

    server srv;
    memset(&srv, 0, sizeof(srv));
    srv.engine = engine;
    srv.session = session;

    /* Replay the malformed prefix exactly as the worker loop would see it:
     * token by token, running the recovery scan after each piece.  The stanza
     * opening spans several tokens, so this also checks that detection does
     * not depend on how the marker happens to be tokenized: recovery must
     * stay quiet on every partial prefix and trigger exactly when the
     * opening completes. */
    ds4_tokens toks = {0};
    ds4_tokenize_rendered_chat(engine, forced.ptr, &toks);
    TEST_ASSERT(toks.len > 1);
    size_t scan_from = 0;
    int completion = 0;
    int rec = 0;
    int triggered_at = -1;
    for (int i = 0; i < toks.len; i++) {
        TEST_ASSERT(ds4_session_eval(session, toks.v[i], err, sizeof(err)) == 0);
        size_t piece_len = 0;
        char *piece = ds4_token_text(engine, toks.v[i], &piece_len);
        buf_append(&text, piece, piece_len);
        thinking_state_feed(&thinking, piece, piece_len);
        free(piece);
        TEST_ASSERT(thinking.inside);
        rec = chat_think_tool_recovery(&srv, &text, &thinking, &scan_from,
                                       &completion, 512, err, sizeof(err));
        TEST_ASSERT(rec >= 0);
        if (rec == 1) {
            triggered_at = i;
            break;
        }
    }
    fprintf(stderr,
            "ds4-test: think-tool-recovery trigger=%d/%d injected_tokens=%d\n",
            triggered_at, toks.len, completion);
    TEST_ASSERT(rec == 1);
    TEST_ASSERT(triggered_at == toks.len - 1);
    ds4_tokens_free(&toks);
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
        int token = ds4_session_sample(session, 0.0f, 0, 1.0f, 0.0f, &rng);
        if (ds4_is_stop_token(engine, token)) break;
        size_t piece_len = 0;
        char *piece = ds4_token_text(engine, token, &piece_len);
        buf_append(&text, piece, piece_len);
        free(piece);
        observe_tool_markers(text.ptr, &saw_start, &saw_end, NULL);
        if (saw_end) break;
        if (ds4_session_eval(session, token, err, sizeof(err)) != 0) {
            decode_ok = false;
            break;
        }
    }
    fprintf(stderr, "ds4-test: think-tool-recovery continuation=[%s]\n",
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
            "ds4-test: think-tool-recovery recovered=%d gen_tokens=%d calls=%d name=%s\n",
            rec, completion, calls.len, calls.len ? calls.v[0].name : "-");

    free(content);
    free(reasoning);
    tool_calls_free(&calls);
    buf_free(&text);
    ds4_session_free(session);
    request_free(&r);
    test_close_engine(false);
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

    int n = 0;
    bool stop = false;
    while (ok && !stop && n < max_tokens) {
        const int token = ds4_session_argmax(session);
        if (ds4_is_stop_token(engine, token)) break;

        int toks[17]; /* base token + draft depth, which the engine clamps to 16 */
        const int ntok = ds4_session_eval_speculative_argmax(
            session, token, max_tokens - n, toks,
            (int)(sizeof(toks) / sizeof(toks[0])), err, sizeof(err));
        if (ntok < 0) { ok = false; TEST_ASSERT(false); break; }
        if (ntok > *max_chunk) *max_chunk = ntok;

        for (int j = 0; j < ntok; j++) {
            if (ds4_is_stop_token(engine, toks[j])) { stop = true; break; }
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
#endif

static void test_cli_refuses_oversized_metal_diagnostic(void) {
    char refused_tmpl[] = "/tmp/ds4-huge-metal-diagnostic.XXXXXX";
    int refused_fd = mkstemp(refused_tmpl);
    TEST_ASSERT(refused_fd >= 0);

    const uint64_t one_gib = 1024ull * 1024ull * 1024ull;
    TEST_ASSERT(ftruncate(refused_fd, (off_t)(40ull * one_gib)) == 0);
    close(refused_fd);

    char allowed_tmpl[] = "/tmp/ds4-small-metal-diagnostic.XXXXXX";
    int allowed_fd = mkstemp(allowed_tmpl);
    TEST_ASSERT(allowed_fd >= 0);
    TEST_ASSERT(ftruncate(allowed_fd, (off_t)(8ull * one_gib)) == 0);
    close(allowed_fd);

    char refused_out_tmpl[] = "/tmp/ds4-huge-metal-diagnostic.out.XXXXXX";
    int refused_out_fd = mkstemp(refused_out_tmpl);
    TEST_ASSERT(refused_out_fd >= 0);
    close(refused_out_fd);

    char allowed_out_tmpl[] = "/tmp/ds4-small-metal-diagnostic.out.XXXXXX";
    int allowed_out_fd = mkstemp(allowed_out_tmpl);
    TEST_ASSERT(allowed_out_fd >= 0);
    close(allowed_out_fd);

    char command[1200];
    int n = snprintf(command, sizeof(command),
                     "DS4_METAL_DIAGNOSTIC_RAM_BYTES=68719476736 "
                     "./ds4 -m %s --metal-graph-prompt-test -p x > %s 2>&1",
                     refused_tmpl, refused_out_tmpl);
    TEST_ASSERT(n > 0 && (size_t)n < sizeof(command));

    int status = system(command);
    TEST_ASSERT(status != -1);
    TEST_ASSERT(WIFEXITED(status));
    TEST_ASSERT(WEXITSTATUS(status) == 2);

    FILE *fp = fopen(refused_out_tmpl, "rb");
    TEST_ASSERT(fp != NULL);
    char output[512];
    size_t got = fp ? fread(output, 1, sizeof(output) - 1u, fp) : 0;
    output[got] = '\0';
    if (fp) fclose(fp);
    TEST_ASSERT(strstr(output, "refusing oversized Metal diagnostic") != NULL);

    n = snprintf(command, sizeof(command),
                 "DS4_METAL_DIAGNOSTIC_RAM_BYTES=68719476736 "
                 "./ds4 -m %s --metal-graph-prompt-test -p x > %s 2>&1",
                 allowed_tmpl, allowed_out_tmpl);
    TEST_ASSERT(n > 0 && (size_t)n < sizeof(command));

    status = system(command);
    TEST_ASSERT(status != -1);
    TEST_ASSERT(WIFEXITED(status));

    fp = fopen(allowed_out_tmpl, "rb");
    TEST_ASSERT(fp != NULL);
    got = fp ? fread(output, 1, sizeof(output) - 1u, fp) : 0;
    output[got] = '\0';
    if (fp) fclose(fp);
    TEST_ASSERT(strstr(output, "refusing oversized Metal diagnostic") == NULL);

    unlink(refused_tmpl);
    unlink(allowed_tmpl);
    unlink(refused_out_tmpl);
    unlink(allowed_out_tmpl);
}

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
    {"--think-tool-recovery", "think-tool-recovery", "forced </think> recovery when a tool call starts inside thinking", test_think_tool_recovery},
    {"--logprob-vectors", "logprob-vectors", "official API top-logprob vector comparison on the standard Metal path", test_official_logprob_vectors},
    {"--metal-ssd-streaming-cache-pressure", "metal-ssd-streaming-cache-pressure", "Metal SSD-streaming layer-batched decode cache-pressure repro for issue #384", test_metal_ssd_streaming_cache_pressure},
    {"--local-golden-vectors", "local-golden-vectors", "local top-k/logit drift regression for long Metal prefill", test_local_golden_vectors},
    {"--metal-short-prefill", "metal-short-prefill", "Metal ratio-4 short prefill regression", test_metal_short_prefill_ratio4},
    {"--metal-rope-split-half", "metal-rope-split-half", "Metal HY3 split-half RoPE batch kernel vs CPU reference", test_metal_rope_split_half},
    {"--metal-kernels", "metal-kernels", "isolated Metal kernel numeric regressions", test_metal_kernel_group},
    {"--metal-tensor-equivalence", "metal-tensor-equivalence", "fast/quality Metal prompt-logit and greedy equivalence", test_metal_mpp_equivalence},
    {"--streaming-decode-prefill-correctness", "streaming-decode-prefill-correctness", "streaming decode-style cold prefill drift and repeatability", test_streaming_decode_prefill_correctness},
    {"--mtp-verify-depth", "mtp-verify-depth", "MTP speculative verify commits autoregressive-identical tokens at draft depth > 2", test_mtp_verify_depth},
#endif
    {"--cli-safety", "cli-safety", "CLI refuses oversized local Metal diagnostics before model open", test_cli_refuses_oversized_metal_diagnostic},
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
    puts("  DS4_TEST_SSD_STREAMING=1   Run model tests through Metal SSD streaming.");
    puts("  DS4_TEST_SSD_STREAMING_CACHE_GB=N  Streaming routed expert cache in GiB.");
    puts("  DS4_TEST_SSD_STREAMING_CACHE_EXPERTS=N  Streaming routed expert cache count.");
    puts("  DS4_TEST_SSD_STREAMING_COLD=1  Skip streaming hot expert preload.");
    puts("  DS4_METAL_DISABLE_STREAMING_COLD_DECODE_PREFILL=1  Force canonical streamed cold prefill.");
    puts("  DS4_TEST_LONG_PROMPT=FILE  Rendered long-context story fact prompt.");
    puts("  DS4_TEST_VECTOR_FILE=FILE  Simple official-vector fixture.");
    puts("  DS4_TEST_LOCAL_GOLDEN_FILE=FILE  Local top-k golden-vector fixture.");
    puts("  DS4_TEST_MPP_EQ_CASE=NAME  Run only Tensor equivalence cases whose id contains NAME.");
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
