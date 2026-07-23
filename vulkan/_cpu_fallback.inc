
/* CPU FALLBACKS for inference — replace with Vulkan compute shaders incrementally */
#include <stdint.h>
#include <string.h>
#include <math.h>

/* ---- ds4_gpu_embed_tokens_hc_tensor (CPU implementation) ---- */
int ds4_gpu_embed_tokens_hc_tensor(
    ds4_gpu_tensor *out_hc, const ds4_gpu_tensor *tokens,
    const void *model_map, uint64_t model_size,
    uint64_t weight_offset, uint32_t n_vocab, uint32_t n_tokens,
    uint32_t n_embd, uint32_t n_hc)
{
    (void)model_size; (void)n_hc;
    if (!out_hc || !tokens || !model_map) return 0;
    const int32_t *tok = (const int32_t*)tokens->ptr;
    float *out = (float*)out_hc->ptr;
    const float *w = (const float*)((const char*)model_map + weight_offset);
    for (uint32_t t = 0; t < n_tokens; t++) {
        int32_t id = tok[t]; if (id < 0) id = 0; if ((uint32_t)id >= n_vocab) id = 0;
        memcpy(out + t * n_embd, w + (uint64_t)id * n_embd, n_embd * sizeof(float));
    }
    return 1;
}

/* ---- ds4_gpu_embed_token_hc_tensor (CPU, single token) ---- */
int ds4_gpu_embed_token_hc_tensor(
    ds4_gpu_tensor *out_hc,
    const void *model_map, uint64_t model_size,
    uint64_t weight_offset, uint32_t n_vocab, uint32_t token,
    uint32_t n_embd, uint32_t n_hc)
{
    (void)model_size; (void)n_hc;
    if (!out_hc || !model_map) return 0;
    int32_t id = (int32_t)token; if (id < 0) id = 0; if ((uint32_t)id >= n_vocab) id = 0;
    const float *w = (const float*)((const char*)model_map + weight_offset);
    memcpy(out_hc->ptr, w + (uint64_t)id * n_embd, n_embd * sizeof(float));
    return 1;
}


/* ---- ds4_gpu_matmul_f32_tensor (CPU) ---- */
int ds4_gpu_matmul_f32_tensor(
    ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
    uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim,
    const ds4_gpu_tensor *x, uint64_t n_tok)
{
    (void)model_size; if (!out || !model_map || !x) return 0;
    float *outp = (float*)out->ptr;
    const float *xp = (const float*)x->ptr;
    const float *wp = (const float*)((const char*)model_map + weight_offset);
    for (uint64_t t = 0; t < n_tok; t++)
        for (uint64_t o = 0; o < out_dim; o++) {
            double sum = 0.0;
            for (uint64_t i = 0; i < in_dim; i++) sum += (double)xp[t * in_dim + i] * (double)wp[o * in_dim + i];
            outp[t * out_dim + o] = (float)sum;
        }
    return 1;
}

/* ---- ds4_gpu_matmul_f16_tensor (CPU) ---- */
int ds4_gpu_matmul_f16_tensor(
    ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
    uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim,
    const ds4_gpu_tensor *x, uint64_t n_tok)
{
    (void)model_size; if (!out || !model_map || !x) return 0;
    float *outp = (float*)out->ptr;
    const float *xp = (const float*)x->ptr;
    const uint16_t *wp = (const uint16_t*)((const char*)model_map + weight_offset);
    for (uint64_t t = 0; t < n_tok; t++)
        for (uint64_t o = 0; o < out_dim; o++) {
            double sum = 0.0;
            for (uint64_t i = 0; i < in_dim; i++) {
                uint16_t h = wp[o * in_dim + i];
                uint32_t sign = (uint32_t)(h >> 15);
                uint32_t exp = (uint32_t)((h >> 10) & 0x1f);
                uint32_t mant = (uint32_t)(h & 0x3ff);
                float w;
                if (exp == 0) w = (sign ? -1.0f : 1.0f) * ldexpf((float)mant, -24);
                else if (exp == 31) w = mant ? NAN : (sign ? -INFINITY : INFINITY);
                else w = (sign ? -1.0f : 1.0f) * ldexpf((float)mant + 1024.0f, (int)exp - 25);
                sum += (double)xp[t * in_dim + i] * (double)w;
            }
            outp[t * out_dim + o] = (float)sum;
        }
    return 1;
}

/* ---- ds4_gpu_rms_norm_weight_rows_tensor (CPU) ---- */
int ds4_gpu_rms_norm_weight_rows_tensor(
    ds4_gpu_tensor *out, const ds4_gpu_tensor *x,
    const void *model_map, uint64_t model_size,
    uint64_t weight_offset, uint32_t n, uint32_t rows, float eps)
{
    (void)model_size; if (!out || !x) return 0;
    float *outp = (float*)out->ptr; const float *xp = (const float*)x->ptr;
    const float *wp = (const float*)((const char*)model_map + weight_offset);
    for (uint32_t r = 0; r < rows; r++) {
        double sum = 0.0; for (uint32_t i = 0; i < n; i++) sum += (double)xp[r * n + i] * xp[r * n + i];
        float rcp = 1.0f / sqrtf((float)(sum / n) + eps);
        for (uint32_t i = 0; i < n; i++) outp[r * n + i] = xp[r * n + i] * rcp * wp[i];
    }
    return 1;
}

/* ---- ds4_gpu_dsv4_qkv_rms_norm_rows_tensor (CPU) ---- */
int ds4_gpu_dsv4_qkv_rms_norm_rows_tensor(
    ds4_gpu_tensor *q_out, const ds4_gpu_tensor *q,
    const void *mm, uint64_t ms, uint64_t qwo, uint32_t qn,
    ds4_gpu_tensor *kv_out, const ds4_gpu_tensor *kv,
    uint64_t kvwo, uint32_t kvn, uint32_t rows, float eps)
{
    (void)ms; if (!q_out || !q || !kv_out || !kv) return 0;
    float *qop = (float*)q_out->ptr; const float *qp = (const float*)q->ptr;
    const float *qw = (const float*)((const char*)mm + qwo);
    float *kvop = (float*)kv_out->ptr; const float *kvp = (const float*)kv->ptr;
    const float *kvw = (const float*)((const char*)mm + kvwo);
    for (uint32_t r = 0; r < rows; r++) {
        double qs = 0.0, kvs = 0.0;
        for (uint32_t i = 0; i < qn; i++) qs += (double)qp[r*qn+i] * qp[r*qn+i];
        for (uint32_t i = 0; i < kvn; i++) kvs += (double)kvp[r*kvn+i] * kvp[r*kvn+i];
        float qr = 1.0f / sqrtf((float)(qs / qn) + eps);
        float kvr = 1.0f / sqrtf((float)(kvs / kvn) + eps);
        for (uint32_t i = 0; i < qn; i++) qop[r*qn+i] = qp[r*qn+i] * qr * qw[i];
        for (uint32_t i = 0; i < kvn; i++) kvop[r*kvn+i] = kvp[r*kvn+i] * kvr * kvw[i];
    }
    return 1;
}

/* ---- ds4_gpu_repeat_hc_tensor (CPU) ---- */
int ds4_gpu_repeat_hc_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *row,
                              uint32_t n_embd, uint32_t n_hc)
{
    if (!out || !row) return 0;
    memcpy(out->ptr, row->ptr, (uint64_t)n_hc * n_embd * sizeof(float));
    return 1;
}

/* ---- ds4_gpu_output_hc_weights_tensor (CPU) ---- */
int ds4_gpu_output_hc_weights_tensor(
    ds4_gpu_tensor *out, const ds4_gpu_tensor *pre,
    const void *mm, uint64_t ms, uint64_t scale_off, uint64_t base_off,
    uint32_t n_hc, float eps)
{
    (void)ms; (void)scale_off; (void)base_off; (void)eps;
    if (!out || !pre || !mm) return 0;
    memcpy(out->ptr, pre->ptr, (uint64_t)n_hc * sizeof(float));
    return 1;
}
