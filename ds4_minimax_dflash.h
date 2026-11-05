/* MiniMax-M3.1 DFlash2 drafter.
 * Included once from ds4.c, after mm3_graph_forward_tokens, inside
 * DS4_HAS_QWEN4_METAL. Draft GEMMs are Q8_0 so the existing batch matmul
 * reads each weight once. Capture blits the six target layer inputs; the
 * permanent draft cache receives only an accepted prefix. Greedy verify
 * keeps the target stream: a rejected draft is rewound before it is emitted.
 */
#ifndef DS4_MINIMAX_DFLASH_H
#define DS4_MINIMAX_DFLASH_H

#define DFLASH_LAYERS 5
#define DFLASH_TAPS 6
#define DFLASH_BLOCK 8
/* Prompt notes go into the draft cache in wider bites than the propose block.
 * The Q8 batch matmul already accepts 32 rows; the block path stays at 8. */
#define DFLASH_INJECT 32u
#define DFLASH_DRAFT_MAX 7
#define DFLASH_H 6144u
#define DFLASH_FF 12288u
#define DFLASH_Q 8192u
#define DFLASH_KV 1024u
#define DFLASH_HEAD 128u
#define DFLASH_NQ 64u
#define DFLASH_NKV 8u
#define DFLASH_FUSED (DFLASH_TAPS * DFLASH_H)
#define DFLASH_RANK 256u
#define DFLASH_TOPK 32u
#define DFLASH_SWA 2048u
#define DFLASH_GROUPS 384u
#define DFLASH_CONV 1536u
#define DFLASH_MASK 200061
#define DFLASH_VOCAB 200064u
#define DFLASH_KEYS (DFLASH_SWA + DFLASH_BLOCK)

static const uint32_t dflash_tap_layer[DFLASH_TAPS] = { 6u, 17u, 27u, 38u, 48u, 59u };
static const char dflash_tensor_name[] = "dflash";

typedef struct {
    ds4_tensor q, k, v, o, gate, up, down, attn_proj, ffn_proj;
    float *attn_norm, *ffn_norm, *q_norm, *k_norm;
    float *attn_base, *ffn_base;
} mm3_dflash_layer;

struct mm3_dflash {
    ds4_model model;
    bool registered;
    bool use_gpu;
    bool broken;
    bool noted;
    bool log;
    uint32_t n_max;
    uint32_t ctx_cap;
    uint32_t cap;
    float p_min;
    float rope_freq[64];
    ds4_tensor fc, hidden_proj;
    float *hidden_norm, *final_norm;
    float *pred_code, *succ_code;
    mm3_dflash_layer layer[DFLASH_LAYERS];
    uint16_t *k_cache[DFLASH_LAYERS];
    uint16_t *v_cache[DFLASH_LAYERS];
    ds4_gpu_tensor *tap;
    ds4_gpu_tensor *act_in;
    ds4_gpu_tensor *act_out;
    ds4_gpu_tensor *logits;
    ds4_gpu_tensor *q_gpu, *k_gpu, *v_gpu, *gate_gpu, *up_gpu;
    float *tap_f, *in_f, *out_f, *logit_f;
    float *q_gpu_f, *k_gpu_f, *v_gpu_f, *gate_gpu_f, *up_gpu_f;
    float *hid, *scratch, *scratch2, *qbuf, *kbuf, *vbuf, *dyn;
    float *attn_k, *attn_v;
    uint64_t cycles, drafted, accepted;
    double draft_ms, verify_ms;
    uint64_t alloc_bytes;
};

static int dflash_tap_index(uint32_t il) {
    for (uint32_t i = 0; i < DFLASH_TAPS; i++) if (dflash_tap_layer[i] == il) return (int)i;
    return -1;
}

static float dflash_bf16(uint16_t h) {
    uint32_t u = (uint32_t)h << 16;
    float f;
    memcpy(&f, &u, sizeof(f));
    return f;
}

static void dflash_quant_row(const uint16_t *src, uint8_t *dst, uint32_t n) {
    const uint32_t nb = n / 32u;
    for (uint32_t b = 0; b < nb; b++) {
        float v[32];
        float amax = 0.0f;
        for (uint32_t j = 0; j < 32u; j++) {
            v[j] = dflash_bf16(src[b * 32u + j]);
            float a = fabsf(v[j]);
            if (a > amax) amax = a;
        }
        const float d = amax / 127.0f;
        const float id = d > 0.0f ? 1.0f / d : 0.0f;
        uint16_t dh = f32_to_f16(d);
        memcpy(dst + b * 34u, &dh, sizeof(dh));
        int8_t *qs = (int8_t *)(dst + b * 34u + 2u);
        for (uint32_t j = 0; j < 32u; j++) {
            long q = lrintf(v[j] * id);
            if (q > 127) q = 127;
            if (q < -128) q = -128;
            qs[j] = (int8_t)q;
        }
    }
}

typedef struct {
    const uint16_t *src;
    uint8_t *dst;
    uint32_t in;
    uint32_t row_bytes;
} dflash_quant_ctx;

static void dflash_quant_worker(void *ctx, uint64_t row0, uint64_t row1) {
    dflash_quant_ctx *c = ctx;
    for (uint64_t r = row0; r < row1; r++) {
        dflash_quant_row(c->src + r * c->in, c->dst + r * c->row_bytes, c->in);
    }
}

static void dflash_set_q8(ds4_tensor *t, uint64_t off, uint32_t in, uint32_t out) {
    memset(t, 0, sizeof(*t));
    t->name.ptr = dflash_tensor_name;
    t->name.len = sizeof(dflash_tensor_name) - 1u;
    t->ndim = 2;
    t->dim[0] = in;
    t->dim[1] = out;
    t->type = DS4_TENSOR_Q8_0;
    t->abs_offset = off;
    t->elements = (uint64_t)in * out;
    t->bytes = (uint64_t)out * (in / 32u) * 34u;
}

static const uint8_t *dflash_st_find(const char *json, const uint8_t *data, uint64_t data_n,
                                    const char *name, uint64_t expect) {
    char key[192];
    snprintf(key, sizeof(key), "\"%s\"", name);
    const char *p = strstr(json, key);
    if (!p) return NULL;
    const char *offs = strstr(p, "\"data_offsets\"");
    if (!offs) return NULL;
    const char *lb = strchr(offs, '[');
    if (!lb) return NULL;
    char *end = NULL;
    unsigned long long a = strtoull(lb + 1, &end, 10);
    if (!end || *end != ',') return NULL;
    unsigned long long b = strtoull(end + 1, NULL, 10);
    if (b < a || b - a != expect || b > data_n) return NULL;
    return data + a;
}

static float *dflash_f32_tensor(const char *json, const uint8_t *data, uint64_t data_n,
                               const char *name, uint64_t n) {
    const uint8_t *src = dflash_st_find(json, data, data_n, name, n * 2u);
    if (!src) {
        fprintf(stderr, "ds4: MiniMax DFlash2 is missing %s\n", name);
        return NULL;
    }
    float *dst = malloc(n * sizeof(float));
    if (!dst) return NULL;
    const uint16_t *h = (const uint16_t *)src;
    for (uint64_t i = 0; i < n; i++) dst[i] = dflash_bf16(h[i]);
    return dst;
}

static bool dflash_quant_tensor(struct mm3_dflash *d, uint8_t **cursor, uint8_t *end,
                               const char *json, const uint8_t *data, uint64_t data_n,
                               const char *name, uint32_t out, uint32_t in, ds4_tensor *dst) {
    const uint64_t row_bytes = (uint64_t)(in / 32u) * 34u;
    const uint64_t bytes = (uint64_t)out * row_bytes;
    const uint64_t aligned = (bytes + 255u) & ~255ull;
    if ((uint64_t)(end - *cursor) < aligned) return false;
    const uint8_t *src = dflash_st_find(json, data, data_n, name, (uint64_t)out * in * 2u);
    if (!src) {
        fprintf(stderr, "ds4: MiniMax DFlash2 is missing %s\n", name);
        return false;
    }
    dflash_quant_ctx ctx = {
        .src = (const uint16_t *)src,
        .dst = *cursor,
        .in = in,
        .row_bytes = (uint32_t)row_bytes,
    };
    ds4_parallel_for(out, dflash_quant_worker, &ctx);
    dflash_set_q8(dst, (uint64_t)(*cursor - (uint8_t *)d->model.map), in, out);
    *cursor += aligned;
    return true;
}

static void dflash_rms(float *x, const float *w, uint32_t n) {
    double ss = 0.0;
    for (uint32_t i = 0; i < n; i++) ss += (double)x[i] * x[i];
    const float scale = 1.0f / sqrtf((float)(ss / (double)n) + 1.0e-6f);
    if (w) {
        for (uint32_t i = 0; i < n; i++) x[i] *= scale * w[i];
    } else {
        for (uint32_t i = 0; i < n; i++) x[i] *= scale;
    }
}

static void dflash_rope(float *h, uint32_t pos, const float *freq) {
    for (uint32_t i = 0; i < 64u; i++) {
        const float ang = (float)pos * freq[i];
        const float c = cosf(ang), s = sinf(ang);
        const float x0 = h[i], x1 = h[i + 64u];
        h[i] = x0 * c - x1 * s;
        h[i + 64u] = x0 * s + x1 * c;
    }
}

static void dflash_norm_rope(float *x, uint32_t n_heads, const float *w, uint32_t pos, const float *freq) {
    for (uint32_t h = 0; h < n_heads; h++) {
        float *head = x + h * DFLASH_HEAD;
        dflash_rms(head, w, DFLASH_HEAD);
        dflash_rope(head, pos, freq);
    }
}

static float dflash_silu(float x) {
    if (x >= 0.0f) return x / (1.0f + expf(-x));
    const float e = expf(x);
    return x * e / (1.0f + e);
}

static uint32_t dflash_n_call(uint32_t n) {
    if (n >= 5u && n <= 8u) return 8u;
    if (n >= 9u && n <= 16u) return 16u;
    if (n >= 17u && n <= DFLASH_INJECT) return DFLASH_INJECT;
    return n;
}

static void dflash_note(struct mm3_dflash *d, const char *what) {
    if (d->noted) return;
    d->noted = true;
    d->broken = true;
    fprintf(stderr, "ds4: MiniMax DFlash2 disabled after %s\n", what);
}

static bool dflash_linear(struct mm3_dflash *d, float *out, const ds4_tensor *w,
                         const float *in, uint32_t n) {
    if (n == 0) return true;
    const uint32_t in_dim = (uint32_t)w->dim[0], out_dim = (uint32_t)w->dim[1];
    if (!d->use_gpu || ds4_gpu_commands_active() || n > DFLASH_INJECT) {
        matmul_q8_0_batch(out, &d->model, w, in, n);
        return true;
    }
    const uint32_t n_call = dflash_n_call(n);
    if (in != d->in_f) memcpy(d->in_f, in, (size_t)n * in_dim * sizeof(float));
    if (n_call > n) {
        memset(d->in_f + (size_t)n * in_dim, 0, (size_t)(n_call - n) * in_dim * sizeof(float));
    }
    if (!qwen4_gemv(d->act_out, &d->model, w, d->act_in, n_call)) return false;
    if (out != d->out_f) memcpy(out, d->out_f, (size_t)n * out_dim * sizeof(float));
    return true;
}

/* Independent GEMMs share one command buffer. Each needs its own output
 * because the batch matmul reads weights once per dispatch and the outputs
 * are not ready until the buffer completes. */
static bool dflash_linear_group(struct mm3_dflash *d, const float *in, uint32_t n,
                               const ds4_tensor *w0, ds4_gpu_tensor *g0, float *c0, float *h0,
                               const ds4_tensor *w1, ds4_gpu_tensor *g1, float *c1, float *h1,
                               const ds4_tensor *w2, ds4_gpu_tensor *g2, float *c2, float *h2) {
    if (!w0 || !w1 || !g0 || !g1 || !c0 || !c1 || !h0 || !h1) return false;
    if (w2 && (!g2 || !c2 || !h2)) return false;
    if (!d->use_gpu || n > DFLASH_INJECT || ds4_gpu_commands_active()) {
        if (!dflash_linear(d, h0, w0, in, n) || !dflash_linear(d, h1, w1, in, n)) return false;
        return !w2 || dflash_linear(d, h2, w2, in, n);
    }
    const uint32_t in_dim = (uint32_t)w0->dim[0];
    const uint32_t n_call = dflash_n_call(n);
    const uint32_t n_gemm = n >= 5u ? n_call : n;
    if (in != d->in_f) memcpy(d->in_f, in, (size_t)n * in_dim * sizeof(float));
    if (n_gemm > n) memset(d->in_f + (size_t)n * in_dim, 0, (size_t)(n_gemm - n) * in_dim * sizeof(float));
    if (!ds4_gpu_begin_commands()) return false;
    bool ok = qwen4_gemv(g0, &d->model, w0, d->act_in, n_gemm) &&
              qwen4_gemv(g1, &d->model, w1, d->act_in, n_gemm) &&
              (!w2 || qwen4_gemv(g2, &d->model, w2, d->act_in, n_gemm));
    if (!ds4_gpu_end_commands()) ok = false;
    if (!ok) return false;
    memcpy(h0, c0, (size_t)n * (uint32_t)w0->dim[1] * sizeof(float));
    memcpy(h1, c1, (size_t)n * (uint32_t)w1->dim[1] * sizeof(float));
    if (w2) memcpy(h2, c2, (size_t)n * (uint32_t)w2->dim[1] * sizeof(float));
    return true;
}

static void dflash_conv(float *y, const float *x, const float *dynamic, const float *base,
                       int side, uint32_t T) {
    for (uint32_t t = 0; t < T; t++) {
        const float *dyn = dynamic + (uint64_t)t * DFLASH_CONV;
        for (uint32_t h = 0; h < DFLASH_H; h++) {
            const uint32_t group = h / 16u;
            float acc = 0.0f;
            for (uint32_t tap = 0; tap < 2u; tap++) {
                const float xv = t >= tap ? x[((uint64_t)t - tap) * DFLASH_H + h] : 0.0f;
                const float coeff = dyn[group + tap * DFLASH_GROUPS + (uint32_t)side * 2u * DFLASH_GROUPS];
                const float b = base[(((uint32_t)side * 2u + tap) * DFLASH_H) + h];
                acc += xv * (coeff + b);
            }
            y[(uint64_t)t * DFLASH_H + h] = acc;
        }
    }
}

typedef struct {
    const float *q, *k, *v;
    float *out;
    const uint32_t *kpos;
    uint32_t n_keys, T, qpos0;
} dflash_attn_job;

static void dflash_attn_worker(void *ctx, uint64_t kv0, uint64_t kv1) {
    dflash_attn_job *a = ctx;
    const float scale = 1.0f / sqrtf((float)DFLASH_HEAD);
    for (uint64_t kv = kv0; kv < kv1; kv++) {
        for (uint32_t t = 0; t < a->T; t++) {
            const uint32_t qpos = a->qpos0 + t;
            const uint32_t lo = qpos > DFLASH_SWA ? qpos - DFLASH_SWA : 0u;
            for (uint32_t qh = 0; qh < DFLASH_NQ / DFLASH_NKV; qh++) {
                const uint32_t head = (uint32_t)kv * (DFLASH_NQ / DFLASH_NKV) + qh;
                const float *qq = a->q + ((uint64_t)t * DFLASH_NQ + head) * DFLASH_HEAD;
                float scores[DFLASH_KEYS];
                uint8_t use[DFLASH_KEYS];
                float maxs = -1.0e30f;
                uint32_t n_use = 0;
                for (uint32_t i = 0; i < a->n_keys; i++) {
                    use[i] = a->kpos[i] >= lo;
                    if (!use[i]) continue;
                    const float *kk = a->k + ((uint64_t)i * DFLASH_NKV + kv) * DFLASH_HEAD;
                    float dot = 0.0f;
                    for (uint32_t d0 = 0; d0 < DFLASH_HEAD; d0++) dot += qq[d0] * kk[d0];
                    scores[i] = dot * scale;
                    if (scores[i] > maxs) maxs = scores[i];
                    n_use++;
                }
                float sum = 0.0f;
                for (uint32_t i = 0; i < a->n_keys; i++) {
                    if (!use[i]) continue;
                    scores[i] = expf(scores[i] - maxs);
                    sum += scores[i];
                }
                const float inv = sum > 0.0f && n_use ? 1.0f / sum : 0.0f;
                float *oo = a->out + ((uint64_t)t * DFLASH_NQ + head) * DFLASH_HEAD;
                for (uint32_t d0 = 0; d0 < DFLASH_HEAD; d0++) oo[d0] = 0.0f;
                for (uint32_t i = 0; i < a->n_keys; i++) {
                    if (!use[i]) continue;
                    const float w = scores[i] * inv;
                    const float *vv = a->v + ((uint64_t)i * DFLASH_NKV + kv) * DFLASH_HEAD;
                    for (uint32_t d0 = 0; d0 < DFLASH_HEAD; d0++) oo[d0] += w * vv[d0];
                }
            }
        }
    }
}

static void dflash_attend(struct mm3_dflash *d, const float *q, const float *k, const float *v,
                         float *out, const uint32_t *kpos, uint32_t n_keys, uint32_t T, uint32_t qpos0) {
    (void)d;
    dflash_attn_job job = { q, k, v, out, kpos, n_keys, T, qpos0 };
    ds4_parallel_for_min_rows(DFLASH_NKV, dflash_attn_worker, &job, 1);
}

static void dflash_store_f16(uint16_t *dst, const float *src, uint32_t n) {
    for (uint32_t i = 0; i < n; i++) dst[i] = f32_to_f16(src[i]);
}

static bool dflash_inject(struct mm3_dflash *d, uint32_t pos0, uint32_t T) {
    if (!d || d->broken || T == 0) return !d || !d->broken;
    for (uint32_t base = 0; base < T; ) {
        const uint32_t n = T - base > DFLASH_INJECT ? DFLASH_INJECT : T - base;
        for (uint32_t t = 0; t < n; t++) {
            float *dst = d->in_f + (uint64_t)t * DFLASH_FUSED;
            for (uint32_t tap = 0; tap < DFLASH_TAPS; tap++) {
                const float *src = d->tap_f + ((uint64_t)tap * d->cap + base + t) * DFLASH_H;
                memcpy(dst + (uint64_t)tap * DFLASH_H, src, DFLASH_H * sizeof(float));
            }
        }
        if (!dflash_linear(d, d->hid, &d->fc, d->in_f, n)) return false;
        for (uint32_t t = 0; t < n; t++) dflash_rms(d->hid + (uint64_t)t * DFLASH_H, d->hidden_norm, DFLASH_H);
        for (uint32_t il = 0; il < DFLASH_LAYERS; il++) {
            mm3_dflash_layer *L = &d->layer[il];
            if (!dflash_linear_group(d, d->hid, n,
                                    &L->k, d->k_gpu, d->k_gpu_f, d->kbuf,
                                    &L->v, d->v_gpu, d->v_gpu_f, d->vbuf,
                                    NULL, NULL, NULL, NULL)) return false;
            for (uint32_t t = 0; t < n; t++) {
                const uint32_t pos = pos0 + base + t;
                if (pos >= d->ctx_cap) return false;
                float *kk = d->kbuf + (uint64_t)t * DFLASH_KV;
                dflash_norm_rope(kk, DFLASH_NKV, L->k_norm, pos, d->rope_freq);
                dflash_store_f16(d->k_cache[il] + (uint64_t)pos * DFLASH_KV, kk, DFLASH_KV);
                dflash_store_f16(d->v_cache[il] + (uint64_t)pos * DFLASH_KV,
                                 d->vbuf + (uint64_t)t * DFLASH_KV, DFLASH_KV);
            }
        }
        base += n;
    }
    return true;
}

static bool dflash_block(struct mm3_dflash *d, const ds4_model *tm, const ds4_weights *tw,
                        int anchor, const int *ids, uint32_t T, uint32_t pos0) {
    for (uint32_t t = 0; t < T; t++) {
        qwen4_ref_row(tm, tw->token_embd, (uint64_t)ids[t], d->hid + (uint64_t)t * DFLASH_H);
    }
    for (uint32_t il = 0; il < DFLASH_LAYERS; il++) {
        mm3_dflash_layer *L = &d->layer[il];
        memcpy(d->scratch, d->hid, (size_t)T * DFLASH_H * sizeof(float));
        for (uint32_t t = 0; t < T; t++) dflash_rms(d->scratch + (uint64_t)t * DFLASH_H, L->attn_norm, DFLASH_H);
        if (!dflash_linear(d, d->dyn, &L->attn_proj, d->scratch, T)) return false;
        dflash_conv(d->scratch2, d->scratch, d->dyn, L->attn_base, 0, T);
        if (!dflash_linear_group(d, d->scratch2, T,
                                &L->q, d->q_gpu, d->q_gpu_f, d->qbuf,
                                &L->k, d->k_gpu, d->k_gpu_f, d->kbuf,
                                &L->v, d->v_gpu, d->v_gpu_f, d->vbuf)) return false;
        for (uint32_t t = 0; t < T; t++) {
            const uint32_t pos = pos0 + t;
            dflash_norm_rope(d->qbuf + (uint64_t)t * DFLASH_Q, DFLASH_NQ, L->q_norm, pos, d->rope_freq);
            dflash_norm_rope(d->kbuf + (uint64_t)t * DFLASH_KV, DFLASH_NKV, L->k_norm, pos, d->rope_freq);
        }
        const uint32_t ctx_lo = pos0 > DFLASH_SWA ? pos0 - DFLASH_SWA : 0u;
        const uint32_t n_ctx = pos0 - ctx_lo;
        const uint32_t n_keys = n_ctx + T;
        uint32_t kpos[DFLASH_KEYS];
        for (uint32_t i = 0; i < n_ctx; i++) {
            const uint32_t pos = ctx_lo + i;
            kpos[i] = pos;
            for (uint32_t e = 0; e < DFLASH_KV; e++) {
                d->attn_k[(uint64_t)i * DFLASH_KV + e] = f16_to_f32(d->k_cache[il][(uint64_t)pos * DFLASH_KV + e]);
                d->attn_v[(uint64_t)i * DFLASH_KV + e] = f16_to_f32(d->v_cache[il][(uint64_t)pos * DFLASH_KV + e]);
            }
        }
        for (uint32_t t = 0; t < T; t++) {
            kpos[n_ctx + t] = pos0 + t;
            memcpy(d->attn_k + (uint64_t)(n_ctx + t) * DFLASH_KV, d->kbuf + (uint64_t)t * DFLASH_KV, DFLASH_KV * sizeof(float));
            memcpy(d->attn_v + (uint64_t)(n_ctx + t) * DFLASH_KV, d->vbuf + (uint64_t)t * DFLASH_KV, DFLASH_KV * sizeof(float));
        }
        /* scratch holds the 8192-wide attention output; it is larger than qbuf's live Q. */
        dflash_attend(d, d->qbuf, d->attn_k, d->attn_v, d->scratch, kpos, n_keys, T, pos0);
        if (!dflash_linear(d, d->scratch2, &L->o, d->scratch, T)) return false;
        dflash_conv(d->scratch, d->scratch2, d->dyn, L->attn_base, 1, T);
        for (uint32_t i = 0; i < T * DFLASH_H; i++) d->hid[i] += d->scratch[i];
        memcpy(d->scratch2, d->hid, (size_t)T * DFLASH_H * sizeof(float));
        for (uint32_t t = 0; t < T; t++) dflash_rms(d->hid + (uint64_t)t * DFLASH_H, L->ffn_norm, DFLASH_H);
        if (!dflash_linear(d, d->dyn, &L->ffn_proj, d->hid, T)) return false;
        dflash_conv(d->scratch, d->hid, d->dyn, L->ffn_base, 0, T);
        if (!dflash_linear_group(d, d->scratch, T,
                                &L->gate, d->gate_gpu, d->gate_gpu_f, d->qbuf,
                                &L->up, d->up_gpu, d->up_gpu_f, d->kbuf,
                                NULL, NULL, NULL, NULL)) return false;
        for (uint32_t i = 0; i < T * DFLASH_FF; i++) d->qbuf[i] = dflash_silu(d->qbuf[i]) * d->kbuf[i];
        if (!dflash_linear(d, d->scratch, &L->down, d->qbuf, T)) return false;
        dflash_conv(d->hid, d->scratch, d->dyn, L->ffn_base, 1, T);
        for (uint32_t i = 0; i < T * DFLASH_H; i++) d->hid[i] += d->scratch2[i];
    }
    for (uint32_t t = 0; t < T; t++) dflash_rms(d->hid + (uint64_t)t * DFLASH_H, d->final_norm, DFLASH_H);
    (void)anchor;
    return true;
}

static void dflash_topk(const float *logits, int32_t *ids, float *vals) {
    for (uint32_t i = 0; i < DFLASH_TOPK; i++) { ids[i] = -1; vals[i] = -1.0e30f; }
    for (uint32_t i = 0; i < DFLASH_VOCAB; i++) {
        const float v = logits[i];
        if (!(v > vals[DFLASH_TOPK - 1u])) continue;
        uint32_t j = DFLASH_TOPK - 1u;
        while (j > 0 && v > vals[j - 1u]) {
            vals[j] = vals[j - 1u];
            ids[j] = ids[j - 1u];
            j--;
        }
        vals[j] = v;
        ids[j] = (int32_t)i;
    }
}

static int dflash_walk(struct mm3_dflash *d, int anchor, uint32_t T, int *drafts) {
    int32_t cand[DFLASH_BLOCK][DFLASH_TOPK];
    float unary[DFLASH_BLOCK][DFLASH_TOPK];
    float gate[DFLASH_BLOCK][DFLASH_RANK];
    if (!dflash_linear(d, &gate[0][0], &d->hidden_proj, d->hid, T)) return -1;
    for (uint32_t t = 0; t < T; t++) {
        dflash_topk(d->logit_f + (uint64_t)t * DFLASH_VOCAB, cand[t], unary[t]);
    }
    int n = 0;
    int pred = anchor;
    for (uint32_t t = 1; t < T; t++) {
        float best = -1.0e30f;
        int best_k = -1;
        const float *pc = d->pred_code + (uint64_t)pred * DFLASH_RANK;
        float scored[DFLASH_TOPK];
        for (uint32_t k = 0; k < DFLASH_TOPK; k++) {
            const int id = cand[t][k];
            if (id < 0) { scored[k] = -1.0e30f; continue; }
            const float *sc = d->succ_code + (uint64_t)id * DFLASH_RANK;
            float s = unary[t][k];
            for (uint32_t r = 0; r < DFLASH_RANK; r++) s += sc[r] * pc[r] * gate[t][r];
            scored[k] = s;
            if (s > best) { best = s; best_k = (int)k; }
        }
        if (best_k < 0) break;
        if (d->p_min > 0.0f) {
            float sum = 0.0f;
            for (uint32_t k = 0; k < DFLASH_TOPK; k++) sum += expf(scored[k] - best);
            if (sum <= 0.0f || 1.0f / sum < d->p_min) break;
        }
        drafts[n++] = cand[t][best_k];
        pred = cand[t][best_k];
    }
    return n;
}

static int dflash_propose(struct mm3_dflash *d, const ds4_model *tm, const ds4_weights *tw,
                         int anchor, uint32_t pos0, uint32_t n_draft, int *drafts) {
    if (n_draft > DFLASH_DRAFT_MAX) n_draft = DFLASH_DRAFT_MAX;
    const uint32_t T = n_draft + 1u;
    int ids[DFLASH_BLOCK];
    ids[0] = anchor;
    for (uint32_t t = 1; t < T; t++) ids[t] = DFLASH_MASK;
    if (!dflash_block(d, tm, tw, anchor, ids, T, pos0)) return -1;
    const uint32_t n_call = dflash_n_call(T);
    if (T < n_call && T >= 5u) {
        memset(d->hid + (uint64_t)T * DFLASH_H, 0, (size_t)(n_call - T) * DFLASH_H * sizeof(float));
    }
    memcpy(d->in_f, d->hid, (size_t)(T >= 5u ? n_call : T) * DFLASH_H * sizeof(float));
    if (!ds4_gpu_commands_active()) {
        if (!qwen4_gemv(d->logits, tm, tw->output, d->act_in, T >= 5u ? n_call : T)) return -1;
    } else {
        return -1;
    }
    return dflash_walk(d, anchor, T, drafts);
}

static void mm3_dflash_free(struct mm3_dflash *d) {
    if (!d) return;
    if (d->registered && d->model.map) ds4_gpu_unregister_host_model(d->model.map);
    if (d->model.map && d->alloc_bytes) munmap((void *)d->model.map, d->alloc_bytes);
    free(d->hidden_norm);
    free(d->final_norm);
    free(d->pred_code);
    free(d->succ_code);
    for (uint32_t il = 0; il < DFLASH_LAYERS; il++) {
        free(d->layer[il].attn_norm);
        free(d->layer[il].ffn_norm);
        free(d->layer[il].q_norm);
        free(d->layer[il].k_norm);
        free(d->layer[il].attn_base);
        free(d->layer[il].ffn_base);
        free(d->k_cache[il]);
        free(d->v_cache[il]);
    }
    ds4_gpu_tensor_free(d->tap);
    ds4_gpu_tensor_free(d->act_in);
    ds4_gpu_tensor_free(d->act_out);
    ds4_gpu_tensor_free(d->logits);
    ds4_gpu_tensor_free(d->q_gpu);
    ds4_gpu_tensor_free(d->k_gpu);
    ds4_gpu_tensor_free(d->v_gpu);
    ds4_gpu_tensor_free(d->gate_gpu);
    ds4_gpu_tensor_free(d->up_gpu);
    free(d->hid);
    free(d->scratch);
    free(d->scratch2);
    free(d->qbuf);
    free(d->kbuf);
    free(d->vbuf);
    free(d->dyn);
    free(d->attn_k);
    free(d->attn_v);
    free(d);
}

static void mm3_dflash_capture(ds4_mm3_graph *g, uint32_t il, uint32_t T) {
    struct mm3_dflash *d = g ? g->draft : NULL;
    if (!d || d->broken || T == 0 || T > d->cap) return;
    const int tap = dflash_tap_index(il);
    if (tap < 0) return;
    if (!ds4_gpu_commands_active() ||
        !ds4_gpu_tensor_copy(d->tap, (uint64_t)tap * d->cap * DFLASH_H * sizeof(float),
                             g->x, 0, (uint64_t)T * DFLASH_H * sizeof(float))) {
        dflash_note(d, "a layer-input capture");
    }
}

static void mm3_dflash_commit_capture(ds4_mm3_graph *g, const ds4_model *m, const ds4_weights *w,
                                     uint32_t pos0, uint32_t T) {
    (void)m;
    (void)w;
    struct mm3_dflash *d = g ? g->draft : NULL;
    if (!d || d->broken || T == 0) return;
    if (g->draft_defer) return;
    if (!dflash_inject(d, pos0, T)) dflash_note(d, "draft KV injection");
}

static bool mm3_dflash_verify(ds4_mm3_graph *g, const ds4_model *m, const ds4_weights *w,
                             int anchor, int *extra, int *n_extra, int *bonus, int room) {
    struct mm3_dflash *d = g ? g->draft : NULL;
    if (n_extra) *n_extra = 0;
    if (!d || d->broken || !extra || !n_extra || !bonus || room < 1 || anchor < 0) return false;
    uint32_t n_draft = d->n_max;
    if ((int)n_draft > room) n_draft = (uint32_t)room;
    if (g->pos + 1u + n_draft > g->ctx_cap) {
        if (g->ctx_cap <= g->pos + 1u) return false;
        n_draft = g->ctx_cap - g->pos - 1u;
    }
    if (n_draft < 1u) return false;
    const uint32_t pos0 = g->pos;
    const double t0 = now_sec();
    int drafts[DFLASH_DRAFT_MAX];
    const int n_prop = dflash_propose(d, m, w, anchor, pos0, n_draft, drafts);
    d->draft_ms += (now_sec() - t0) * 1000.0;
    if (n_prop < 0) {
        dflash_note(d, "a draft forward");
        return false;
    }
    if (n_prop == 0) return false;
    int tokens[DFLASH_BLOCK];
    tokens[0] = anchor;
    for (int i = 0; i < n_prop; i++) tokens[i + 1] = drafts[i];
    const uint32_t T = (uint32_t)n_prop + 1u;
    g->draft_defer = true;
    const double tv = now_sec();
    const bool fwd = mm3_graph_forward_tokens(g, m, w, tokens, T, NULL);
    g->draft_defer = false;
    d->verify_ms += (now_sec() - tv) * 1000.0;
    if (!fwd) return false;
    /* The Q8 batch matmul reads 8 or 16 rows. Extra rows are zeros so they
     * cannot change the accepted prefix. */
    uint32_t n_head = T;
    if (T >= 5u && T <= 7u) n_head = 8u;
    else if (T >= 9u && T <= 15u) n_head = 16u;
    if (n_head > T) {
        float *xf = ds4_gpu_tensor_contents(g->x);
        if (!xf) {
            g->pos = pos0;
            dflash_note(d, "verification logits");
            return false;
        }
        memset(xf + (uint64_t)T * DFLASH_H, 0, (size_t)(n_head - T) * DFLASH_H * sizeof(float));
    }
    bool head = ds4_gpu_begin_commands() != 0;
    if (head) {
        head = ds4_gpu_rms_norm_weight_rows_tensor(g->xn, g->x, m->map, m->size,
                                                   w->output_norm->abs_offset,
                                                   DFLASH_H, n_head, DS4_RMS_EPS) != 0 &&
               qwen4_gemv(d->logits, m, w->output, g->xn, n_head);
    }
    if (!ds4_gpu_end_commands()) head = false;
    if (!head) {
        g->pos = pos0;
        dflash_note(d, "verification logits");
        return false;
    }
    int A = 0;
    for (int i = 0; i < n_prop; i++) {
        if (sample_argmax(d->logit_f + (uint64_t)i * DS4_N_VOCAB, DS4_N_VOCAB) != drafts[i]) break;
        extra[A++] = drafts[i];
    }
    *bonus = sample_argmax(d->logit_f + (uint64_t)A * DS4_N_VOCAB, DS4_N_VOCAB);
    *n_extra = A;
    g->pos = pos0 + 1u + (uint32_t)A;
    d->cycles++;
    d->drafted += (uint64_t)n_prop;
    d->accepted += (uint64_t)A;
    if (d->log) {
        fprintf(stderr, "ds4: DFlash2 cycle pos %u drafted %d accepted %d bonus %d\n",
                pos0, n_prop, A, *bonus);
    }
    if (!dflash_inject(d, pos0, 1u + (uint32_t)A)) dflash_note(d, "accepted-prefix injection");
    return true;
}

static void mm3_dflash_report(const struct mm3_dflash *d) {
    if (!d) return;
    const double rate = d->drafted ? 100.0 * (double)d->accepted / (double)d->drafted : 0.0;
    fprintf(stderr,
            "ds4: MiniMax DFlash2: %" PRIu64 " cycles, %" PRIu64 " drafted, %" PRIu64
            " accepted (%.1f%%), draft %.1f ms, verify %.1f ms\n",
            d->cycles, d->drafted, d->accepted, rate, d->draft_ms, d->verify_ms);
}

static bool dflash_load_layer(struct mm3_dflash *d, uint8_t **cursor, uint8_t *end,
                             const char *json, const uint8_t *data, uint64_t data_n, uint32_t il) {
    mm3_dflash_layer *L = &d->layer[il];
    char name[128];
    snprintf(name, sizeof(name), "layers.%u.input_layernorm.weight", il);
    L->attn_norm = dflash_f32_tensor(json, data, data_n, name, DFLASH_H);
    snprintf(name, sizeof(name), "layers.%u.post_attention_layernorm.weight", il);
    L->ffn_norm = dflash_f32_tensor(json, data, data_n, name, DFLASH_H);
    snprintf(name, sizeof(name), "layers.%u.self_attn.q_norm.weight", il);
    L->q_norm = dflash_f32_tensor(json, data, data_n, name, DFLASH_HEAD);
    snprintf(name, sizeof(name), "layers.%u.self_attn.k_norm.weight", il);
    L->k_norm = dflash_f32_tensor(json, data, data_n, name, DFLASH_HEAD);
    snprintf(name, sizeof(name), "layers.%u.attention_conv.base_kernel", il);
    L->attn_base = dflash_f32_tensor(json, data, data_n, name, 2u * 2u * DFLASH_H);
    snprintf(name, sizeof(name), "layers.%u.mlp_conv.base_kernel", il);
    L->ffn_base = dflash_f32_tensor(json, data, data_n, name, 2u * 2u * DFLASH_H);
    if (!L->attn_norm || !L->ffn_norm || !L->q_norm || !L->k_norm || !L->attn_base || !L->ffn_base) return false;
#define DFLASH_Q8(field, fmt, out, in) do { \
        snprintf(name, sizeof(name), fmt, il); \
        if (!dflash_quant_tensor(d, cursor, end, json, data, data_n, name, (out), (in), &(field))) return false; \
    } while (0)
    DFLASH_Q8(L->q, "layers.%u.self_attn.q_proj.weight", DFLASH_Q, DFLASH_H);
    DFLASH_Q8(L->k, "layers.%u.self_attn.k_proj.weight", DFLASH_KV, DFLASH_H);
    DFLASH_Q8(L->v, "layers.%u.self_attn.v_proj.weight", DFLASH_KV, DFLASH_H);
    DFLASH_Q8(L->o, "layers.%u.self_attn.o_proj.weight", DFLASH_H, DFLASH_Q);
    DFLASH_Q8(L->gate, "layers.%u.mlp.gate_proj.weight", DFLASH_FF, DFLASH_H);
    DFLASH_Q8(L->up, "layers.%u.mlp.up_proj.weight", DFLASH_FF, DFLASH_H);
    DFLASH_Q8(L->down, "layers.%u.mlp.down_proj.weight", DFLASH_H, DFLASH_FF);
    DFLASH_Q8(L->attn_proj, "layers.%u.attention_conv.kernel_projection.weight", DFLASH_CONV, DFLASH_H);
    DFLASH_Q8(L->ffn_proj, "layers.%u.mlp_conv.kernel_projection.weight", DFLASH_CONV, DFLASH_H);
#undef DFLASH_Q8
    return true;
}

/* DS4_MM3_DFLASH is a safetensors path, or "0" to leave the drafter off.
 * With the variable unset, look beside the process and under ~/ds4/gguf. */
static int mm3_dflash_resolve_path(char *out, size_t n) {
    const char *env = getenv("DS4_MM3_DFLASH");
    if (env && strcmp(env, "0") == 0) return 0;
    if (env && env[0]) {
        snprintf(out, n, "%s", env);
        return 1;
    }
    const char *rels[] = {
        "minimax-m31-dflash2/model.safetensors",
        "gguf/minimax-m31-dflash2/model.safetensors",
    };
    for (size_t i = 0; i < sizeof(rels) / sizeof(rels[0]); i++) {
        if (access(rels[i], R_OK) == 0) {
            snprintf(out, n, "%s", rels[i]);
            return 1;
        }
    }
    const char *home = getenv("HOME");
    if (home && home[0]) {
        snprintf(out, n, "%s/ds4/gguf/minimax-m31-dflash2/model.safetensors", home);
        if (access(out, R_OK) == 0) return 1;
    }
    return 0;
}

static void mm3_dflash_open(ds4_mm3_graph *g) {
    if (!g || g->draft) return;
    char path[1024];
    if (!mm3_dflash_resolve_path(path, sizeof(path))) return;
    if (DS4_N_EMBD != DFLASH_H || DS4_N_VOCAB != DFLASH_VOCAB || DS4_N_LAYER < 60u) {
        fprintf(stderr, "ds4: MiniMax DFlash2 does not match this model shape\n");
        return;
    }
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "ds4: MiniMax DFlash2 weights not found at %s\n", path);
        return;
    }
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size < 16) {
        fprintf(stderr, "ds4: MiniMax DFlash2 weights at %s are unreadable\n", path);
        close(fd);
        return;
    }
    uint8_t *file = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (file == MAP_FAILED) {
        fprintf(stderr, "ds4: MiniMax DFlash2 weights could not be mapped\n");
        return;
    }
    uint64_t header_len = 0;
    memcpy(&header_len, file, sizeof(header_len));
    if (header_len < 2 || header_len > 64u * 1024u * 1024u ||
        (uint64_t)st.st_size < 8u + header_len) {
        fprintf(stderr, "ds4: MiniMax DFlash2 safetensors header is invalid\n");
        munmap(file, (size_t)st.st_size);
        return;
    }
    char *json = malloc(header_len + 1u);
    if (!json) {
        munmap(file, (size_t)st.st_size);
        return;
    }
    memcpy(json, file + 8, header_len);
    json[header_len] = 0;
    const uint8_t *data = file + 8 + header_len;
    const uint64_t data_n = (uint64_t)st.st_size - 8u - header_len;
    struct mm3_dflash *d = calloc(1, sizeof(*d));
    if (!d) {
        free(json);
        munmap(file, (size_t)st.st_size);
        return;
    }
    d->ctx_cap = g->ctx_cap;
    d->cap = g->cap_tokens;
    d->n_max = DFLASH_DRAFT_MAX;
    d->log = metal_graph_tp_env_flag("DS4_MM3_DFLASH_LOG", false);
    const char *ns = getenv("DS4_MM3_DFLASH_N");
    if (ns && ns[0]) {
        int n = atoi(ns);
        if (n < 1) n = 1;
        if (n > DFLASH_DRAFT_MAX) n = DFLASH_DRAFT_MAX;
        d->n_max = (uint32_t)n;
    }
    const char *ps = getenv("DS4_MM3_DFLASH_PMIN");
    if (ps && ps[0]) d->p_min = strtof(ps, NULL);
    for (uint32_t i = 0; i < 64u; i++) {
        d->rope_freq[i] = (float)pow(5000000.0, -(double)i / 64.0);
    }
    fprintf(stderr, "ds4: MiniMax DFlash2 quantizing %s\n", path);
    const double tq = now_sec();
    /* fc plus five layers of nine GEMMs, each rounded up to a 256-byte pad. */
    const uint64_t q8_bound = (uint64_t)2200 * 1024u * 1024u;
    const uint64_t page = (uint64_t)getpagesize();
    uint64_t bytes = (q8_bound + page - 1u) & ~(page - 1u);
    /* mmap, not malloc: Metal wraps this pointer with newBufferWithBytesNoCopy. */
    void *blob = mmap(NULL, (size_t)bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    if (blob == MAP_FAILED) {
        fprintf(stderr, "ds4: MiniMax DFlash2 weight buffer allocation failed\n");
        free(json);
        munmap(file, (size_t)st.st_size);
        mm3_dflash_free(d);
        return;
    }
    d->model.map = blob;
    d->alloc_bytes = bytes;
    d->model.size = bytes;
    uint8_t *cursor = blob;
    uint8_t *end = cursor + q8_bound;
    bool ok = dflash_quant_tensor(d, &cursor, end, json, data, data_n,
                                 "fc.weight", DFLASH_H, DFLASH_FUSED, &d->fc) &&
              dflash_quant_tensor(d, &cursor, end, json, data, data_n,
                                 "candidate_selector.hidden_projection.weight",
                                 DFLASH_RANK, DFLASH_H, &d->hidden_proj);
    d->hidden_norm = dflash_f32_tensor(json, data, data_n, "hidden_norm.weight", DFLASH_H);
    d->final_norm = dflash_f32_tensor(json, data, data_n, "norm.weight", DFLASH_H);
    d->pred_code = dflash_f32_tensor(json, data, data_n, "candidate_selector.predecessor_codebook",
                                    (uint64_t)DFLASH_VOCAB * DFLASH_RANK);
    d->succ_code = dflash_f32_tensor(json, data, data_n, "candidate_selector.successor_codebook",
                                    (uint64_t)DFLASH_VOCAB * DFLASH_RANK);
    ok = ok && d->hidden_norm && d->final_norm && d->pred_code && d->succ_code;
    for (uint32_t il = 0; ok && il < DFLASH_LAYERS; il++) {
        ok = dflash_load_layer(d, &cursor, end, json, data, data_n, il);
    }
    free(json);
    munmap(file, (size_t)st.st_size);
    if (!ok) {
        fprintf(stderr, "ds4: MiniMax DFlash2 weight load failed\n");
        mm3_dflash_free(d);
        return;
    }
    const uint64_t used = (uint64_t)(cursor - (uint8_t *)blob);
    const uint64_t used_pages = (used + page - 1u) & ~(page - 1u);
    d->model.size = used_pages;
    d->use_gpu = !metal_graph_tp_env_flag("DS4_MM3_DFLASH_CPU", false);
    if (d->use_gpu) {
        d->registered = ds4_gpu_register_host_model(d->model.map, d->model.size) != 0;
        if (!d->registered) {
            fprintf(stderr, "ds4: MiniMax DFlash2 draft GEMMs are on the CPU\n");
            d->use_gpu = false;
            d->model.size = bytes;
        }
    } else {
        d->model.size = bytes;
    }
    const uint64_t kv_n = (uint64_t)d->ctx_cap * DFLASH_KV;
    for (uint32_t il = 0; il < DFLASH_LAYERS; il++) {
        d->k_cache[il] = calloc(kv_n, sizeof(uint16_t));
        d->v_cache[il] = calloc(kv_n, sizeof(uint16_t));
        if (!d->k_cache[il] || !d->v_cache[il]) ok = false;
    }
    d->tap = ds4_gpu_tensor_alloc((uint64_t)d->cap * DFLASH_TAPS * DFLASH_H * sizeof(float));
    d->act_in = ds4_gpu_tensor_alloc(32ull * DFLASH_FUSED * sizeof(float));
    d->act_out = ds4_gpu_tensor_alloc(32ull * DFLASH_FF * sizeof(float));
    d->logits = ds4_gpu_tensor_alloc(32ull * DFLASH_VOCAB * sizeof(float));
    d->q_gpu = ds4_gpu_tensor_alloc(32ull * DFLASH_Q * sizeof(float));
    d->k_gpu = ds4_gpu_tensor_alloc(32ull * DFLASH_KV * sizeof(float));
    d->v_gpu = ds4_gpu_tensor_alloc(32ull * DFLASH_KV * sizeof(float));
    d->gate_gpu = ds4_gpu_tensor_alloc(32ull * DFLASH_FF * sizeof(float));
    d->up_gpu = ds4_gpu_tensor_alloc(32ull * DFLASH_FF * sizeof(float));
    /* hid and vbuf also hold a 32-token inject. kbuf is BLOCK*FF, which is
     * already wider than 32 KV rows. The block path still uses at most 8. */
    d->hid = calloc((size_t)DFLASH_INJECT * DFLASH_H, sizeof(float));
    d->scratch = calloc((size_t)DFLASH_BLOCK * DFLASH_FF, sizeof(float));
    d->scratch2 = calloc((size_t)DFLASH_BLOCK * DFLASH_H, sizeof(float));
    d->qbuf = calloc((size_t)DFLASH_BLOCK * DFLASH_FF, sizeof(float));
    d->kbuf = calloc((size_t)DFLASH_BLOCK * DFLASH_FF, sizeof(float));
    d->vbuf = calloc((size_t)DFLASH_INJECT * DFLASH_KV, sizeof(float));
    d->dyn = calloc((size_t)DFLASH_BLOCK * DFLASH_CONV, sizeof(float));
    d->attn_k = calloc((size_t)DFLASH_KEYS * DFLASH_KV, sizeof(float));
    d->attn_v = calloc((size_t)DFLASH_KEYS * DFLASH_KV, sizeof(float));
    if (!ok || !d->tap || !d->act_in || !d->act_out || !d->logits || !d->q_gpu || !d->k_gpu ||
        !d->v_gpu || !d->gate_gpu || !d->up_gpu || !d->hid || !d->scratch ||
        !d->scratch2 || !d->qbuf || !d->kbuf || !d->vbuf || !d->dyn || !d->attn_k || !d->attn_v) {
        fprintf(stderr, "ds4: MiniMax DFlash2 activation allocation failed; speculation is off\n");
        mm3_dflash_free(d);
        return;
    }
    d->tap_f = ds4_gpu_tensor_contents(d->tap);
    d->in_f = ds4_gpu_tensor_contents(d->act_in);
    d->out_f = ds4_gpu_tensor_contents(d->act_out);
    d->logit_f = ds4_gpu_tensor_contents(d->logits);
    d->q_gpu_f = ds4_gpu_tensor_contents(d->q_gpu);
    d->k_gpu_f = ds4_gpu_tensor_contents(d->k_gpu);
    d->v_gpu_f = ds4_gpu_tensor_contents(d->v_gpu);
    d->gate_gpu_f = ds4_gpu_tensor_contents(d->gate_gpu);
    d->up_gpu_f = ds4_gpu_tensor_contents(d->up_gpu);
    if (!d->tap_f || !d->in_f || !d->out_f || !d->logit_f || !d->q_gpu_f || !d->k_gpu_f ||
        !d->v_gpu_f || !d->gate_gpu_f || !d->up_gpu_f) {
        fprintf(stderr, "ds4: MiniMax DFlash2 GPU buffers are not host-visible\n");
        mm3_dflash_free(d);
        return;
    }
    g->draft = d;
    fprintf(stderr,
            "ds4: MiniMax DFlash2 ready in %.1fs: Q8 %.2f GiB, draft KV %.2f GiB, width %u, %s\n",
            now_sec() - tq,
            (double)used / (1024.0 * 1024.0 * 1024.0),
            (double)(DFLASH_LAYERS * 2ull * kv_n * sizeof(uint16_t)) / (1024.0 * 1024.0 * 1024.0),
            d->n_max, d->use_gpu ? "GPU GEMM" : "CPU GEMM");
}

#endif
