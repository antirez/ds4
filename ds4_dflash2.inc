/* Classic DFlash and DFlash2 draft engine for dense Qwen targets.
 * Contracts: z-lab DFlashDraftModel and DFlash2DraftModel. */

typedef struct {
    float *out;
    const uint8_t *data;
    const float *x;
    uint64_t in_dim;
    uint64_t row_bytes;
    uint32_t type;
} dflash2_matvec_kquant_ctx;

static void dflash2_matvec_kquant_worker(void *vctx, uint64_t r0, uint64_t r1) {
    dflash2_matvec_kquant_ctx *c = vctx;
    for (uint64_t r = r0; r < r1; r++) {
        const uint8_t *row = c->data + r * c->row_bytes;
        if (c->type == DS4_TENSOR_Q4_K) {
            c->out[r] = ds4_vec_dot_q4_K_f32((int)c->in_dim, (const block_q4_K *)row, c->x);
        } else {
            c->out[r] = ds4_vec_dot_q5_q6_K_f32(c->type, (int)c->in_dim, row, c->x);
        }
    }
}

static void dflash2_matvec_kquant(float *out, const ds4_model *m, const ds4_tensor *w, const float *x) {
    if (!w || w->ndim < 2 || w->dim[0] == 0) ds4_die("expected a 2D k-quant tensor");
    const uint64_t in_dim = w->dim[0];
    const uint64_t n_rows = w->elements / in_dim;
    if (in_dim % QK_K != 0) ds4_die("k-quant row is not QK_K aligned");
    uint64_t row_bytes;
    if (w->type == DS4_TENSOR_Q4_K) {
        row_bytes = (in_dim / QK_K) * sizeof(block_q4_K);
    } else if (w->type == DS4_TENSOR_Q6_K) {
        row_bytes = (in_dim / QK_K) * sizeof(block_q6_K);
    } else if (w->type == DS4_TENSOR_Q5_K) {
        row_bytes = (in_dim / QK_K) * sizeof(block_q5_K);
    } else {
        ds4_die("unsupported k-quant type");
        return;
    }
    dflash2_matvec_kquant_ctx ctx = {
        .out = out,
        .data = tensor_data(m, w),
        .x = x,
        .in_dim = in_dim,
        .row_bytes = row_bytes,
        .type = w->type,
    };
    ds4_parallel_for(n_rows, dflash2_matvec_kquant_worker, &ctx);
}

static int dflash2_matvecs_gpu(float *out, const ds4_model *m, const ds4_tensor *w,
                               const float *x, uint32_t n_tok);
static int dflash2_matvec_gpu(float *out, const ds4_model *m, const ds4_tensor *w, const float *x) {
#ifndef DS4_NO_GPU
    return dflash2_matvecs_gpu(out, m, w, x, 1);
#else
    (void)out; (void)m; (void)w; (void)x;
    return 0;
#endif
}


static int dflash2_matvecs_gpu(float *out, const ds4_model *m, const ds4_tensor *w,
                               const float *x, uint32_t n_tok) {
#ifdef DS4_NO_GPU
    (void)out; (void)m; (void)w; (void)x; (void)n_tok;
    return 0;
#else
    static int nfail;
    if (!g_qwen_pool.inited || !w || !m || !m->map || n_tok == 0) {
        if (nfail++ < 3) fprintf(stderr, "ds4: dflash gpu skip pool/w map n=%u inited=%d\n",
                                n_tok, g_qwen_pool.inited);
        return 0;
    }
    if (w->type == DS4_TENSOR_Q5_K) return 0;
    if (w->type == DS4_TENSOR_Q6_K && n_tok != 1) return 0;
    if (w->ndim < 2 || w->dim[0] == 0) return 0;
    const uint64_t in_dim = w->dim[0];
    const uint64_t out_dim = w->elements / in_dim;
    static ds4_gpu_tensor *gx, *gy;
    static uint64_t gx_bytes, gy_bytes;
    const uint64_t xb = (uint64_t)n_tok * in_dim * sizeof(float);
    const uint64_t yb = (uint64_t)n_tok * out_dim * sizeof(float);
    if (!gx || gx_bytes < xb) {
        if (gx) ds4_gpu_tensor_free(gx);
        gx = ds4_gpu_tensor_alloc(xb);
        gx_bytes = xb;
    }
    if (!gy || gy_bytes < yb) {
        if (gy) ds4_gpu_tensor_free(gy);
        gy = ds4_gpu_tensor_alloc(yb);
        gy_bytes = yb;
    }
    if (!gx || !gy) {
        if (nfail++ < 3) fprintf(stderr, "ds4: dflash gpu alloc fail in=%llu out=%llu\n",
                                (unsigned long long)in_dim, (unsigned long long)out_dim);
        return 0;
    }
    if (!ds4_gpu_begin_commands()) {
        if (nfail++ < 3) fprintf(stderr, "ds4: dflash gpu begin fail\n");
        return 0;
    }
    int ok = ds4_gpu_tensor_write(gx, 0, x, xb);
    if (ok) {
        ok = ds4_gpu_matmul_quant_tensor(gy, m->map, m->size, w->abs_offset,
                                         w->type, in_dim, out_dim, gx, n_tok);
    }
    if (!ds4_gpu_end_commands()) ok = 0;
    if (ok && !ds4_gpu_tensor_read(gy, 0, out, yb)) ok = 0;
    if (!ok && nfail++ < 4) {
        fprintf(stderr, "ds4: dflash gpu gemm fail type=%u in=%llu out=%llu n=%u\n",
                w->type, (unsigned long long)in_dim, (unsigned long long)out_dim, n_tok);
    }
    return ok;
#endif
}

static void dflash2_matvecs(float *out, const ds4_model *m, const ds4_tensor *w,
                            const float *x, uint32_t n_tok);

static int dflash2_matvecs_n_gpu(float **outs, const ds4_model *m, const ds4_tensor **ws,
                                 const float *x, uint32_t n_tok, int n_w) {
#ifdef DS4_NO_GPU
    (void)outs; (void)m; (void)ws; (void)x; (void)n_tok; (void)n_w;
    return 0;
#else
    static int nfail;
    static ds4_gpu_tensor *gx;
    static ds4_gpu_tensor *gy[4];
    static uint64_t gx_bytes, gy_bytes[4];
    if (!g_qwen_pool.inited || !m || !m->map || !x || n_tok == 0 || n_w <= 0 || n_w > 4) return 0;
    uint64_t in_dim0 = 0;
    uint64_t xb = 0;
    for (int i = 0; i < n_w; i++) {
        const ds4_tensor *w = ws[i];
        if (!w || !outs[i] || w->ndim < 2 || w->dim[0] == 0) return 0;
        if (w->type == DS4_TENSOR_Q5_K) return 0;
        if (w->type == DS4_TENSOR_Q6_K && n_tok != 1) return 0;
        if (i == 0) {
            in_dim0 = w->dim[0];
            xb = (uint64_t)n_tok * in_dim0 * sizeof(float);
        } else if (w->dim[0] != in_dim0) {
            return 0;
        }
    }
    if (!gx || gx_bytes < xb) {
        if (gx) ds4_gpu_tensor_free(gx);
        gx = ds4_gpu_tensor_alloc(xb);
        gx_bytes = xb;
    }
    if (!gx) return 0;
    for (int i = 0; i < n_w; i++) {
        const uint64_t out_dim = ws[i]->elements / in_dim0;
        const uint64_t yb = (uint64_t)n_tok * out_dim * sizeof(float);
        if (!gy[i] || gy_bytes[i] < yb) {
            if (gy[i]) ds4_gpu_tensor_free(gy[i]);
            gy[i] = ds4_gpu_tensor_alloc(yb);
            gy_bytes[i] = yb;
        }
        if (!gy[i]) return 0;
    }
    if (!ds4_gpu_begin_commands()) {
        if (nfail++ < 3) fprintf(stderr, "ds4: dflash gpu n-begin fail\n");
        return 0;
    }
    int ok = ds4_gpu_tensor_write(gx, 0, x, xb);
    for (int i = 0; ok && i < n_w; i++) {
        const uint64_t out_dim = ws[i]->elements / in_dim0;
        ok = ds4_gpu_matmul_quant_tensor(gy[i], m->map, m->size, ws[i]->abs_offset,
                                         ws[i]->type, in_dim0, out_dim, gx, n_tok);
    }
    if (!ds4_gpu_end_commands()) ok = 0;
    for (int i = 0; ok && i < n_w; i++) {
        const uint64_t yb = (uint64_t)n_tok * (ws[i]->elements / in_dim0) * sizeof(float);
        if (!ds4_gpu_tensor_read(gy[i], 0, outs[i], yb)) ok = 0;
    }
    if (!ok && nfail++ < 4) fprintf(stderr, "ds4: dflash gpu n-gemm fail n_w=%d n=%u\n", n_w, n_tok);
    return ok;
#endif
}

static void dflash2_matvecs_n(float **outs, const ds4_model *m, const ds4_tensor **ws,
                              const float *x, uint32_t n_tok, int n_w) {
    if (dflash2_matvecs_n_gpu(outs, m, ws, x, n_tok, n_w)) return;
    for (int i = 0; i < n_w; i++) dflash2_matvecs(outs[i], m, ws[i], x, n_tok);
}


static void dflash2_matvec(float *out, const ds4_model *m, const ds4_tensor *w, const float *x) {
    if (!w) ds4_die("missing DFlash2 weight");
    if (dflash2_matvec_gpu(out, m, w, x)) return;
    if (w->type == DS4_TENSOR_Q4_K || w->type == DS4_TENSOR_Q5_K || w->type == DS4_TENSOR_Q6_K) {
        dflash2_matvec_kquant(out, m, w, x);
        return;
    }
    matvec_any(out, m, w, x);
}

static void dflash2_matvecs(float *out, const ds4_model *m, const ds4_tensor *w,
                            const float *x, uint32_t n_tok) {
    if (!w) ds4_die("missing DFlash2 weight");
    if (n_tok == 0) return;
    if (dflash2_matvecs_gpu(out, m, w, x, n_tok)) return;
    for (uint32_t t = 0; t < n_tok; t++) {
        const uint64_t in_dim = w->dim[0];
        const uint64_t out_dim = w->elements / in_dim;
        dflash2_matvec(out + (size_t)t * out_dim, m, w, x + (size_t)t * in_dim);
    }
}



static void dflash2_q4k_dequant_row(float *out, const block_q4_K *x, int n) {
    const int nb = n / QK_K;
    for (int i = 0; i < nb; i++) {
        const float d = f16_to_f32(x[i].d);
        const float dmin = f16_to_f32(x[i].dmin);
        const uint8_t *qs = x[i].qs;
        const uint8_t *scales = x[i].scales;
        float *ob = out + (uint64_t)i * QK_K;
        for (int j = 0; j < QK_K / 32; j++) {
            uint8_t sc_val, m_val;
            q4_k_get_scale_min(j, scales, &sc_val, &m_val);
            const int byte_off = (j >> 1) * 32;
            const int shift = (j & 1) * 4;
            const float scale = d * (float)sc_val;
            const float minv = dmin * (float)m_val;
            for (int l = 0; l < 32; l++) {
                const int q = (qs[byte_off + l] >> shift) & 0x0F;
                ob[j * 32 + l] = scale * (float)q - minv;
            }
        }
    }
}

/* Q5_K/Q6_K row dequantizers, mirroring ds4_vec_dot_q5_K_f32 and
 * ds4_vec_dot_q6_K_f32 element for element so a gathered embedding row agrees
 * with what the matvec paths would compute from the same bytes. APEX-style
 * mixed-precision files put token_embd at Q5_K or Q6_K. */
static void dflash2_q5k_dequant_row(float *out, const block_q5_K *x, int n) {
    const int nb = n / QK_K;
    for (int i = 0; i < nb; i++) {
        const float d = f16_to_f32(x[i].d);
        const float dmin = f16_to_f32(x[i].dmin);
        const uint8_t *ql = x[i].qs;
        const uint8_t *qh = x[i].qh;
        const uint8_t *scales = x[i].scales;
        float *ob = out + (uint64_t)i * QK_K;
        for (int group = 0; group < QK_K / 32; group++) {
            uint8_t sc_val, m_val;
            q4_k_get_scale_min(group, scales, &sc_val, &m_val);
            const int ql_base = (group >> 1) * 32;
            const int shift = (group & 1) * 4;
            const uint8_t hmask = (uint8_t)(1u << group);
            const float scale = d * (float)sc_val;
            const float minv = dmin * (float)m_val;
            for (int l = 0; l < 32; l++) {
                const int q = ((ql[ql_base + l] >> shift) & 0x0F) +
                              ((qh[l] & hmask) ? 16 : 0);
                ob[group * 32 + l] = scale * (float)q - minv;
            }
        }
    }
}

static void dflash2_q6k_dequant_row(float *out, const block_q6_K *x, int n) {
    const int nb = n / QK_K;
    for (int i = 0; i < nb; i++) {
        const float d = f16_to_f32(x[i].d);
        const uint8_t *ql = x[i].ql;
        const uint8_t *qh = x[i].qh;
        const int8_t *scales = x[i].scales;
        float *ob = out + (uint64_t)i * QK_K;
        for (int n128 = 0; n128 < QK_K; n128 += 128) {
            for (int l = 0; l < 32; l++) {
                const int is = l / 16;
                const int q1 = ((int)(ql[l +  0] & 0x0F) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int q2 = ((int)(ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int q3 = ((int)(ql[l +  0] >>   4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int q4 = ((int)(ql[l + 32] >>   4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                ob[n128 + l +  0] = d * (float)scales[is + 0] * (float)q1;
                ob[n128 + l + 32] = d * (float)scales[is + 2] * (float)q2;
                ob[n128 + l + 64] = d * (float)scales[is + 4] * (float)q3;
                ob[n128 + l + 96] = d * (float)scales[is + 6] * (float)q4;
            }
            ql += 64;
            qh += 32;
            scales += 8;
        }
    }
}

static void dflash2_embed_row(float *out, const ds4_model *m, const ds4_tensor *w, int token) {
    if (!w || w->ndim != 2) ds4_die("expected a 2D embedding tensor");
    const uint64_t dim = w->dim[0];
    if (token < 0 || (uint64_t)token >= w->dim[1]) ds4_die("DFlash2 embed token out of range");
    if (w->type == DS4_TENSOR_F32) {
        memcpy(out, (const float *)tensor_data(m, w) + (uint64_t)token * dim, (size_t)dim * sizeof(float));
        return;
    }
    if (w->type == DS4_TENSOR_Q4_K) {
        if (dim % QK_K != 0) ds4_die("Q4_K embed row is not aligned");
        const uint64_t row_bytes = (dim / QK_K) * sizeof(block_q4_K);
        const block_q4_K *row = (const block_q4_K *)((const uint8_t *)tensor_data(m, w) + (uint64_t)token * row_bytes);
        dflash2_q4k_dequant_row(out, row, (int)dim);
        return;
    }
    if (w->type == DS4_TENSOR_Q5_K) {
        if (dim % QK_K != 0) ds4_die("Q5_K embed row is not aligned");
        const uint64_t row_bytes = (dim / QK_K) * sizeof(block_q5_K);
        const block_q5_K *row = (const block_q5_K *)((const uint8_t *)tensor_data(m, w) + (uint64_t)token * row_bytes);
        dflash2_q5k_dequant_row(out, row, (int)dim);
        return;
    }
    if (w->type == DS4_TENSOR_Q6_K) {
        if (dim % QK_K != 0) ds4_die("Q6_K embed row is not aligned");
        const uint64_t row_bytes = (dim / QK_K) * sizeof(block_q6_K);
        const block_q6_K *row = (const block_q6_K *)((const uint8_t *)tensor_data(m, w) + (uint64_t)token * row_bytes);
        dflash2_q6k_dequant_row(out, row, (int)dim);
        return;
    }
    ds4_die("unsupported DFlash2 embedding type");
}

static void dflash2_grouped_conv(
        float *out,
        const float *hidden,
        const float *dynamic,
        const float *base,
        int tap,
        uint32_t L,
        uint32_t hidden_size,
        uint32_t kernel,
        uint32_t group) {
    const uint32_t groups = hidden_size / group;
    memset(out, 0, (size_t)L * hidden_size * sizeof(float));
    for (uint32_t t = 0; t < L; t++) {
        for (uint32_t off = 0; off < kernel; off++) {
            const int src = (int)t - (int)off;
            const float *src_h = src >= 0 ? hidden + (uint32_t)src * hidden_size : NULL;
            const float *kb = base + (size_t)hidden_size * (off + kernel * (uint32_t)tap);
            for (uint32_t g = 0; g < groups; g++) {
                const float dyn = dynamic[(size_t)t * kernel * groups + (size_t)off * groups + g];
                for (uint32_t j = 0; j < group; j++) {
                    const float v = src_h ? src_h[g * group + j] : 0.0f;
                    out[(size_t)t * hidden_size + g * group + j] += (kb[g * group + j] + dyn) * v;
                }
            }
        }
    }
}

static void dflash2_rms_head(float *x, const float *w, uint32_t n_head, uint32_t head_dim, float eps) {
    for (uint32_t h = 0; h < n_head; h++) {
        rms_norm_weight(x + (uint64_t)h * head_dim, x + (uint64_t)h * head_dim, w, head_dim, eps);
    }
}

static void dflash2_rope_rows(float *x, uint32_t n_tok, uint32_t n_head,
                              uint32_t head_dim, uint32_t pos0,
                              float freq_base) {
    for (uint32_t t = 0; t < n_tok; t++) {
        qwen_rope_rotate_half_inplace(
            x + (uint64_t)t * n_head * head_dim,
            n_head, head_dim, head_dim, pos0 + t, freq_base);
    }
}

static void dflash2_sdpa(
        float *out,
        const float *q,
        const float *k_ctx,
        const float *v_ctx,
        const float *k_prop,
        const float *v_prop,
        uint32_t L,
        uint32_t n_head,
        uint32_t n_kv,
        uint32_t head_dim) {
    const uint32_t heads_per_kv = n_head / n_kv;
    const float scale = 1.0f / sqrtf((float)head_dim);
    for (uint32_t t = 0; t < L; t++) {
        for (uint32_t h = 0; h < n_head; h++) {
            const uint32_t kv_h = h / heads_per_kv;
            const float *qh = q + ((uint64_t)t * n_head + h) * head_dim;
            float *oh = out + ((uint64_t)t * n_head + h) * head_dim;
            float m_prev = -1e30f;
            float l_prev = 0.0f;
            float acc[128];
            memset(acc, 0, (size_t)head_dim * sizeof(float));
            const uint32_t n_key = 1u + L;
            for (uint32_t s = 0; s < n_key; s++) {
                const float *kh;
                const float *vh;
                if (s == 0) {
                    kh = k_ctx + (uint64_t)kv_h * head_dim;
                    vh = v_ctx + (uint64_t)kv_h * head_dim;
                } else {
                    kh = k_prop + ((uint64_t)(s - 1) * n_kv + kv_h) * head_dim;
                    vh = v_prop + ((uint64_t)(s - 1) * n_kv + kv_h) * head_dim;
                }
                float dot = 0.0f;
                for (uint32_t d = 0; d < head_dim; d++) dot += qh[d] * kh[d];
                dot *= scale;
                const float m_curr = m_prev > dot ? m_prev : dot;
                const float alpha = expf(m_prev - m_curr);
                const float beta = expf(dot - m_curr);
                l_prev = l_prev * alpha + beta;
                for (uint32_t d = 0; d < head_dim; d++) acc[d] = acc[d] * alpha + beta * vh[d];
                m_prev = m_curr;
            }
            const float inv_l = 1.0f / (l_prev > 1e-8f ? l_prev : 1.0f);
            for (uint32_t d = 0; d < head_dim; d++) oh[d] = acc[d] * inv_l;
        }
    }
}

static ds4_tensor *dflash2_tensorf(const ds4_model *m, const char *fmt, uint32_t layer) {
    char name[128];
    const int n = snprintf(name, sizeof(name), fmt, layer);
    if (n < 0 || (size_t)n >= sizeof(name)) ds4_die("DFlash tensor name is too long");
    return model_find_tensor(m, name);
}

static bool dflash2_shape(const ds4_tensor *t, uint32_t ndim,
                          uint64_t d0, uint64_t d1) {
    return t && t->ndim == ndim && t->dim[0] == d0 &&
           (ndim < 2 || t->dim[1] == d1);
}

static bool dflash2_shape3(const ds4_tensor *t, uint64_t d0,
                           uint64_t d1, uint64_t d2) {
    return t && t->ndim == 3 && t->dim[0] == d0 &&
           t->dim[1] == d1 && t->dim[2] == d2;
}

static bool dflash2_f32_vector(const ds4_tensor *t, uint64_t n) {
    return dflash2_shape(t, 1, n, 0) && t->type == DS4_TENSOR_F32;
}

static bool dflash2_attention_shape_valid(
        uint32_t n_head, uint32_t n_kv, uint32_t head_dim) {
    if (n_head == 0 || n_kv == 0 || n_kv > n_head ||
        (n_head % n_kv) != 0 || head_dim == 0 ||
        head_dim > DS4_DFLASH2_MAX_HEAD_DIM || (head_dim & 1u) != 0u) {
        return false;
    }
    /* q_dim/kv_dim are stored in uint32_t by the runtime. */
    return (uint64_t)n_head * head_dim <= UINT32_MAX &&
           (uint64_t)n_kv * head_dim <= UINT32_MAX;
}

static int dflash2_bind(ds4_dflash2_weights *w, const ds4_model *m) {
    memset(w, 0, sizeof(*w));
    if (!m) return 0;
    ds4_str arch = {0};
    if (!model_get_string(m, "general.architecture", &arch) || !ds4_streq(arch, "dflash")) {
        return 0;
    }
    uint32_t version = 0, n_layer = 0, n_embd = 0, n_ff = 0;
    uint32_t n_head = 0, n_kv = 0, head_dim = 0, block = 0, swa = 0;
    uint32_t conv_k = 0, conv_g = 0, rank = 0, topk = 0;
    uint32_t full_layer = UINT32_MAX;
    (void)model_get_u32(m, "dflash.version", &version);
    if (version > 2u) return 0;
    bool classic = version == 1u;
    if (!model_get_u32(m, "dflash.block_count", &n_layer)) return 0;
    if (!model_get_u32(m, "dflash.embedding_length", &n_embd)) return 0;
    if (!model_get_u32(m, "dflash.feed_forward_length", &n_ff)) return 0;
    if (!model_get_u32(m, "dflash.attention.head_count", &n_head)) return 0;
    if (!model_get_u32(m, "dflash.attention.head_count_kv", &n_kv)) return 0;
    if (!model_get_u32(m, "dflash.attention.key_length", &head_dim)) return 0;
    if (!model_get_u32(m, "dflash.block_size", &block)) return 0;
    (void)model_get_u32(m, "dflash.attention.sliding_window", &swa);
    (void)model_get_u32(m, "dflash.attention.full_layer", &full_layer);
    const bool has_conv_metadata =
        model_get_u32(m, "dflash.conv_kernel_size", &conv_k);
    if (version == 0u && !has_conv_metadata) classic = true;
    if (!classic) {
        if (!has_conv_metadata) return 0;
        if (!model_get_u32(m, "dflash.conv_group_size", &conv_g)) return 0;
        if (!model_get_u32(m, "dflash.selector_rank", &rank)) return 0;
        if (!model_get_u32(m, "dflash.selector_top_k", &topk)) return 0;
    }
    if (n_layer == 0 || n_layer > DS4_DFLASH2_MAX_LAYERS) return 0;
    if (block == 0 || block > DS4_DFLASH2_MAX_BLOCK) return 0;
    if (n_embd == 0 || n_embd != DS4_N_EMBD || n_ff == 0 ||
        !dflash2_attention_shape_valid(n_head, n_kv, head_dim) ||
        (classic && (uint64_t)n_head * head_dim != n_embd)) {
        return 0;
    }
    uint64_t conv_dyn_dim = 0;
    if (!classic) {
        if (conv_k == 0 || conv_g == 0 || (n_embd % conv_g) != 0 ||
            rank == 0 || topk == 0 || topk > DS4_N_VOCAB) {
            return 0;
        }
        conv_dyn_dim = 2ull * conv_k * (n_embd / conv_g);
        if (conv_dyn_dim > UINT32_MAX) return 0;
    }
    if (classic && full_layer == UINT32_MAX) full_layer = n_layer - 1u;
    if (classic && full_layer >= n_layer) return 0;

    static const char *const target_keys[] = { "dflash.target_layers" };
    uint32_t n_target = 0;
    if (!model_get_u32_array_any(m, target_keys, 1, w->target_layers,
                                 DS4_DFLASH2_MAX_TARGET, &n_target) || n_target == 0) {
        return 0;
    }
    if (classic && version == 0u) {
        for (uint32_t i = 0; i < n_target; i++) {
            if (w->target_layers[i] == 0u) return 0;
            w->target_layers[i]--;
        }
    }
    for (uint32_t i = 0; i < n_target; i++) {
        if (w->target_layers[i] >= DS4_N_LAYER) return 0;
    }

    w->n_layer = n_layer;
    w->n_embd = n_embd;
    w->n_ff = n_ff;
    w->n_head = n_head;
    w->n_head_kv = n_kv;
    w->head_dim = head_dim;
    w->block_size = block;
    w->conv_kernel = conv_k;
    w->conv_group = conv_g;
    w->selector_rank = rank;
    w->selector_top_k = topk;
    w->n_target = n_target;
    w->sliding_window = swa;
    w->full_attention_layer = full_layer;
    w->classic = classic;
    if (!model_get_token_id(m, "tokenizer.ggml.mask_token_id", &w->mask_token_id) ||
        w->mask_token_id < 0 || (uint32_t)w->mask_token_id >= DS4_N_VOCAB) {
        return 0;
    }

    w->enc_output_norm = model_find_tensor(m, "enc.output_norm.weight");
    w->fc = model_find_tensor(m, "fc.weight");
    w->output_norm = model_find_tensor(m, "output_norm.weight");
    w->selector_hidden = model_find_tensor(m, "selector_hidden.weight");
    w->selector_predecessor = model_find_tensor(m, "selector_predecessor.weight");
    w->selector_successor = model_find_tensor(m, "selector_successor.weight");
    if (!dflash2_f32_vector(w->enc_output_norm, n_embd) ||
        !dflash2_shape(w->fc, 2, (uint64_t)n_target * n_embd, n_embd) ||
        !dflash2_f32_vector(w->output_norm, n_embd)) {
        return 0;
    }
    if (!classic &&
        (!dflash2_shape(w->selector_hidden, 2, n_embd, rank) ||
         !dflash2_shape(w->selector_predecessor, 2, rank, DS4_N_VOCAB) ||
         !dflash2_shape(w->selector_successor, 2, rank, DS4_N_VOCAB))) {
        return 0;
    }

    const uint64_t q_dim = (uint64_t)n_head * head_dim;
    const uint64_t kv_dim = (uint64_t)n_kv * head_dim;
    for (uint32_t il = 0; il < n_layer; il++) {
        ds4_dflash2_layer *l = &w->layer[il];
        /* Binding a support model is a recoverable operation: malformed or
         * incomplete draft GGUFs must make dflash2_bind() return false, not
         * terminate the process through required_tensorf(). Shape checks below
         * provide the single failure path for both absent and malformed tensors. */
        l->attn_norm = dflash2_tensorf(m, "blk.%u.attn_norm.weight", il);
        l->attn_q = dflash2_tensorf(m, "blk.%u.attn_q.weight", il);
        l->attn_k = dflash2_tensorf(m, "blk.%u.attn_k.weight", il);
        l->attn_v = dflash2_tensorf(m, "blk.%u.attn_v.weight", il);
        l->attn_output = dflash2_tensorf(m, "blk.%u.attn_output.weight", il);
        l->attn_q_norm = dflash2_tensorf(m, "blk.%u.attn_q_norm.weight", il);
        l->attn_k_norm = dflash2_tensorf(m, "blk.%u.attn_k_norm.weight", il);
        l->ffn_norm = dflash2_tensorf(m, "blk.%u.ffn_norm.weight", il);
        l->ffn_gate = dflash2_tensorf(m, "blk.%u.ffn_gate.weight", il);
        l->ffn_up = dflash2_tensorf(m, "blk.%u.ffn_up.weight", il);
        l->ffn_down = dflash2_tensorf(m, "blk.%u.ffn_down.weight", il);
        if (!classic) {
            l->attn_conv_base = dflash2_tensorf(m, "blk.%u.attn_conv_base", il);
            l->attn_conv_proj = dflash2_tensorf(m, "blk.%u.attn_conv_proj.weight", il);
            l->ffn_conv_base = dflash2_tensorf(m, "blk.%u.ffn_conv_base", il);
            l->ffn_conv_proj = dflash2_tensorf(m, "blk.%u.ffn_conv_proj.weight", il);
        } else {
            l->attn_conv_base = dflash2_tensorf(m, "blk.%u.attn_conv_base", il);
            l->attn_conv_proj = dflash2_tensorf(m, "blk.%u.attn_conv_proj.weight", il);
            l->ffn_conv_base = dflash2_tensorf(m, "blk.%u.ffn_conv_base", il);
            l->ffn_conv_proj = dflash2_tensorf(m, "blk.%u.ffn_conv_proj.weight", il);
        }
        if (!dflash2_f32_vector(l->attn_norm, n_embd) ||
            !dflash2_shape(l->attn_q, 2, n_embd, q_dim) ||
            !dflash2_shape(l->attn_k, 2, n_embd, kv_dim) ||
            !dflash2_shape(l->attn_v, 2, n_embd, kv_dim) ||
            !dflash2_shape(l->attn_output, 2, q_dim, n_embd) ||
            !dflash2_f32_vector(l->attn_q_norm, head_dim) ||
            !dflash2_f32_vector(l->attn_k_norm, head_dim) ||
            !dflash2_f32_vector(l->ffn_norm, n_embd) ||
            !dflash2_shape(l->ffn_gate, 2, n_embd, n_ff) ||
            !dflash2_shape(l->ffn_up, 2, n_embd, n_ff) ||
            !dflash2_shape(l->ffn_down, 2, n_ff, n_embd)) {
            return 0;
        }
        if (!classic &&
            (!dflash2_shape3(l->attn_conv_base, n_embd, conv_k, 2) ||
             l->attn_conv_base->type != DS4_TENSOR_F32 ||
             !dflash2_shape(l->attn_conv_proj, 2, n_embd, conv_dyn_dim) ||
             !dflash2_shape3(l->ffn_conv_base, n_embd, conv_k, 2) ||
             l->ffn_conv_base->type != DS4_TENSOR_F32 ||
             !dflash2_shape(l->ffn_conv_proj, 2, n_embd, conv_dyn_dim))) {
            return 0;
        }
    }
    w->ready = true;
    return 1;
}
static bool dflash_pending_reserve(
        ds4_qwen_session_state *qs,
        uint32_t rows,
        uint32_t n_target,
        uint32_t hidden) {
    if (!qs) return false;
    if (rows <= qs->dflash_pending_cap) return true;
    uint32_t cap = qs->dflash_pending_cap ? qs->dflash_pending_cap : 8u;
    while (cap < rows) {
        if (cap > UINT32_MAX / 2u) return false;
        cap *= 2u;
    }
    const uint64_t elems = (uint64_t)cap * n_target * hidden;
    if (elems > SIZE_MAX / sizeof(float)) return false;
    qs->dflash_pending = xrealloc(qs->dflash_pending, (size_t)elems * sizeof(float));
    qs->dflash_pending_cap = cap;
    return true;
}

static bool dflash_pending_set(
        ds4_qwen_session_state *qs,
        const float *rows,
        uint32_t n_rows,
        uint32_t n_target,
        uint32_t hidden) {
    if (!dflash_pending_reserve(qs, n_rows, n_target, hidden)) return false;
    if (n_rows != 0) {
        memcpy(qs->dflash_pending, rows,
               (size_t)n_rows * n_target * hidden * sizeof(float));
    }
    qs->dflash_pending_len = n_rows;
    return true;
}


static int dflash_cache_init(
        ds4_dflash_cache *cache,
        const ds4_dflash2_weights *cfg,
        uint32_t max_ctx) {
    if (!cache || !cfg || !cfg->classic || max_ctx == 0) return 0;
#if defined(__APPLE__) && !defined(DS4_NO_GPU)
    if (cache->n_layer == cfg->n_layer) {
        bool reusable = true;
        for (uint32_t il = 0; il < cache->n_layer; il++) {
            if (cache->layer[il].cap < max_ctx) reusable = false;
        }
        if (reusable) {
            for (uint32_t il = 0; il < cache->n_layer; il++) cache->layer[il].len = 0;
            return 1;
        }
    }
    dflash_cache_free(cache);
    cache->n_layer = cfg->n_layer;
    const uint64_t row_bytes =
        (uint64_t)cfg->n_head_kv * cfg->head_dim * sizeof(float);
    for (uint32_t il = 0; il < cache->n_layer; il++) {
        ds4_dflash_cache_layer *layer = &cache->layer[il];
        layer->cap = max_ctx;
        layer->k = ds4_gpu_tensor_alloc((uint64_t)max_ctx * row_bytes);
        layer->v = ds4_gpu_tensor_alloc((uint64_t)max_ctx * row_bytes);
        if (!layer->k || !layer->v) {
            dflash_cache_free(cache);
            return 0;
        }
    }
    return 1;
#else
    (void)max_ctx;
    return 0;
#endif
}



static void dflash2_swiglu(float *mid, const float *gate, const float *up, uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        const float g = gate[i];
        mid[i] = g / (1.0f + expf(-g)) * up[i];
    }
}


static void dflash2_conv_finish(
        float *out,
        const float *attn_or_ffn,
        const float *saved_dyn,
        const ds4_model *m,
        const ds4_tensor *base,
        uint32_t L,
        uint32_t hidden,
        uint32_t kernel,
        uint32_t group) {
    const uint32_t groups = hidden / group;
    /* finish uses the second tap's dynamic slice: dyn layout [L, 2, K, G] */
    float *dyn1 = xmalloc((size_t)L * kernel * groups * sizeof(float));
    const uint32_t half = kernel * groups;
    for (uint32_t t = 0; t < L; t++) {
        memcpy(dyn1 + (size_t)t * half,
               saved_dyn + (size_t)t * 2u * half + half,
               (size_t)half * sizeof(float));
    }
    dflash2_grouped_conv(out, attn_or_ffn, dyn1, tensor_data(m, base),
                         1, L, hidden, kernel, group);
    free(dyn1);
}

#if defined(__APPLE__) && !defined(DS4_NO_GPU)
int ds4_gpu_dflash2_grouped_conv(
        ds4_gpu_tensor *out, const ds4_gpu_tensor *hidden, const ds4_gpu_tensor *dynamic,
        const void *model_map, uint64_t model_size, uint64_t base_off,
        uint32_t L, uint32_t hidden_size, uint32_t kernel, uint32_t group, uint32_t tap);
int ds4_gpu_dflash2_sdpa(
        ds4_gpu_tensor *out, const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *k_ctx, const ds4_gpu_tensor *v_ctx,
        const ds4_gpu_tensor *k_prop, const ds4_gpu_tensor *v_prop,
        uint32_t L, uint32_t n_ctx, uint32_t n_head, uint32_t n_kv,
        uint32_t head_dim, bool causal, uint32_t window);
int ds4_gpu_qwen_rope_rotate_half_rows_tensor(
        ds4_gpu_tensor *x, uint32_t n_tok, uint32_t n_head,
        uint32_t head_dim, uint32_t pos0, float freq_base);

static int dflash2_tensure(ds4_gpu_tensor **t, uint64_t *cap, uint64_t bytes) {
    if (*t && *cap >= bytes) return 1;
    if (*t) ds4_gpu_tensor_free(*t);
    *t = ds4_gpu_tensor_alloc(bytes);
    *cap = bytes;
    return *t != NULL;
}

static int dflash2_gpu_gemm(ds4_gpu_tensor *out, const ds4_model *m, const ds4_tensor *w,
                            const ds4_gpu_tensor *x, uint32_t n_tok) {
    if (!w || w->ndim < 2 || w->dim[0] == 0) return 0;
    if (w->type == DS4_TENSOR_Q5_K) return 0;
    const uint64_t in_dim = w->dim[0];
    const uint64_t out_dim = w->elements / in_dim;
    if (ds4_gpu_matmul_quant_tensor(out, m->map, m->size, w->abs_offset, w->type,
                                    in_dim, out_dim, x, n_tok)) return 1;
    if (n_tok <= 1) return 0;
    for (uint32_t t = 0; t < n_tok; t++) {
        ds4_gpu_tensor *xr = ds4_gpu_tensor_view(x, (uint64_t)t * in_dim * sizeof(float),
                                                 in_dim * sizeof(float));
        ds4_gpu_tensor *yr = ds4_gpu_tensor_view(out, (uint64_t)t * out_dim * sizeof(float),
                                                 out_dim * sizeof(float));
        int row_ok = xr && yr &&
            ds4_gpu_matmul_quant_tensor(yr, m->map, m->size, w->abs_offset, w->type,
                                        in_dim, out_dim, xr, 1);
        ds4_gpu_tensor_free(xr);
        ds4_gpu_tensor_free(yr);
        if (!row_ok) return 0;
    }
    return 1;
}

static ds4_gpu_tensor *th;
static int h_live;

static int dflash2_layer_forward_gpu(
        float *h,
        const float *h_ctx,
        uint32_t n_ctx_new,
        ds4_dflash_cache_layer *cache,
        uint32_t layer_index,
        const ds4_model *m,
        const ds4_dflash2_layer *l,
        const ds4_dflash2_weights *cfg,
        uint32_t L,
        uint32_t noise_pos) {
    if (!g_qwen_pool.inited || !m || !m->map || !l || !cfg || L == 0 || L > 8) return 0;
    if (cfg->head_dim == 0 || cfg->head_dim > 128) return 0;
    if (n_ctx_new == 0 || noise_pos < n_ctx_new) return 0;
    if (cfg->classic &&
        (!cache || !cache->k || !cache->v ||
         cache->len > cache->cap || n_ctx_new > cache->cap - cache->len)) {
        return 0;
    }
    if (!cfg->classic &&
        (!l->attn_conv_base || l->attn_conv_base->type != DS4_TENSOR_F32 ||
         !l->ffn_conv_base || l->ffn_conv_base->type != DS4_TENSOR_F32)) {
        static int ntype;
        if (ntype++ < 3) fprintf(stderr, "ds4: dflash gpu skip conv type attn=%u ffn=%u\n",
                                 l->attn_conv_base ? l->attn_conv_base->type : 99u,
                                 l->ffn_conv_base ? l->ffn_conv_base->type : 99u);
        return 0;
    }
    const uint32_t H = cfg->n_embd;
    const uint32_t ff = cfg->n_ff;
    const uint32_t n_head = cfg->n_head;
    const uint32_t n_kv = cfg->n_head_kv;
    const uint32_t hd = cfg->head_dim;
    const uint32_t q_dim = n_head * hd;
    const uint32_t kv_dim = n_kv * hd;
    const uint32_t groups = cfg->classic ? 1u : H / cfg->conv_group;
    const uint32_t dyn_dim = cfg->classic ? H : 2u * cfg->conv_kernel * groups;
    static ds4_gpu_tensor *thctx, *txn, *tprep, *tdyn, *tq, *tkp, *tvp, *tkc, *tvc, *theads, *tattn, *tfin, *tgate, *tup, *tmid, *tffn;
    static uint64_t ch, cctx, cxn, cprep, cdyn, cq, ckp, cvp, ckc, cvc, chd, catt, cfin, cg, cu, cm, cf;
    const uint64_t bH = (uint64_t)L * H * sizeof(float);
    if (!dflash2_tensure(&th, &ch, bH) ||
        !dflash2_tensure(&thctx, &cctx, (uint64_t)n_ctx_new * H * sizeof(float)) ||
        !dflash2_tensure(&txn, &cxn, bH) ||
        !dflash2_tensure(&tprep, &cprep, bH) ||
        !dflash2_tensure(&tdyn, &cdyn, (uint64_t)L * dyn_dim * sizeof(float)) ||
        !dflash2_tensure(&tq, &cq, (uint64_t)L * q_dim * sizeof(float)) ||
        !dflash2_tensure(&tkp, &ckp, (uint64_t)L * kv_dim * sizeof(float)) ||
        !dflash2_tensure(&tvp, &cvp, (uint64_t)L * kv_dim * sizeof(float)) ||
        !dflash2_tensure(&tkc, &ckc, (uint64_t)n_ctx_new * kv_dim * sizeof(float)) ||
        !dflash2_tensure(&tvc, &cvc, (uint64_t)n_ctx_new * kv_dim * sizeof(float)) ||
        !dflash2_tensure(&theads, &chd, (uint64_t)L * q_dim * sizeof(float)) ||
        !dflash2_tensure(&tattn, &catt, bH) ||
        !dflash2_tensure(&tfin, &cfin, bH) ||
        !dflash2_tensure(&tgate, &cg, (uint64_t)L * ff * sizeof(float)) ||
        !dflash2_tensure(&tup, &cu, (uint64_t)L * ff * sizeof(float)) ||
        !dflash2_tensure(&tmid, &cm, (uint64_t)L * ff * sizeof(float)) ||
        !dflash2_tensure(&tffn, &cf, bH)) return 0;

    int started = 0;
    if (!ds4_gpu_commands_active()) {
        if (!ds4_gpu_begin_commands()) return 0;
        started = 1;
        h_live = 0;
    }
    int ok = 1;
    if (!h_live) {
        ok = ds4_gpu_tensor_write(th, 0, h, bH) &&
             ds4_gpu_tensor_write(thctx, 0, h_ctx,
                                  (uint64_t)n_ctx_new * H * sizeof(float));
        h_live = ok;
    }
    const char *fail = NULL;
    #define DFLASH2_STEP(name, expr) do { if (ok && !(expr)) { fail = (name); ok = 0; } } while (0)
    DFLASH2_STEP("attn_rms", ds4_gpu_rms_norm_weight_rows_tensor(txn, th, m->map, m->size, l->attn_norm->abs_offset, H, L, 1e-6f));
    if (!cfg->classic) {
        DFLASH2_STEP("attn_dyn", dflash2_gpu_gemm(tdyn, m, l->attn_conv_proj, txn, L));
        DFLASH2_STEP("attn_conv0", ds4_gpu_dflash2_grouped_conv(tprep, txn, tdyn, m->map, m->size, l->attn_conv_base->abs_offset, L, H, cfg->conv_kernel, cfg->conv_group, 0));
    }
    const ds4_gpu_tensor *attn_input = cfg->classic ? txn : tprep;
    DFLASH2_STEP("attn_q", dflash2_gpu_gemm(
        tq, m, l->attn_q, attn_input, L));
    int fused_prop_kv = 0;
    if (ok && l->attn_k->type == DS4_TENSOR_Q4_K &&
        l->attn_v->type == DS4_TENSOR_Q4_K &&
        ds4_gpu_matmul_q4_k_pair_tensor(
            tkp, tvp, m->map, m->size,
            l->attn_k->abs_offset, l->attn_v->abs_offset,
            DS4_TENSOR_Q4_K, H, kv_dim, attn_input, L)) {
        fused_prop_kv = 1;
    }
    if (!fused_prop_kv) {
        DFLASH2_STEP("attn_k", dflash2_gpu_gemm(
            tkp, m, l->attn_k, attn_input, L));
        DFLASH2_STEP("attn_v", dflash2_gpu_gemm(
            tvp, m, l->attn_v, attn_input, L));
    }
    int fused_ctx_kv = 0;
    if (ok && l->attn_k->type == DS4_TENSOR_Q4_K &&
        l->attn_v->type == DS4_TENSOR_Q4_K &&
        ds4_gpu_matmul_q4_k_pair_tensor(
            tkc, tvc, m->map, m->size,
            l->attn_k->abs_offset, l->attn_v->abs_offset,
            DS4_TENSOR_Q4_K, H, kv_dim, thctx, n_ctx_new)) {
        fused_ctx_kv = 1;
    }
    if (!fused_ctx_kv) {
        DFLASH2_STEP("ctx_k", dflash2_gpu_gemm(
            tkc, m, l->attn_k, thctx, n_ctx_new));
        DFLASH2_STEP("ctx_v", dflash2_gpu_gemm(
            tvc, m, l->attn_v, thctx, n_ctx_new));
    }
    DFLASH2_STEP("q_rms", l->attn_q_norm && ds4_gpu_rms_norm_weight_rows_tensor(tq, tq, m->map, m->size, l->attn_q_norm->abs_offset, hd, L * n_head, 1e-6f));
    DFLASH2_STEP("k_rms", l->attn_k_norm && ds4_gpu_rms_norm_weight_rows_tensor(tkp, tkp, m->map, m->size, l->attn_k_norm->abs_offset, hd, L * n_kv, 1e-6f));
    DFLASH2_STEP("kc_rms", l->attn_k_norm && ds4_gpu_rms_norm_weight_rows_tensor(
        tkc, tkc, m->map, m->size, l->attn_k_norm->abs_offset,
        hd, n_ctx_new * n_kv, 1e-6f));
    DFLASH2_STEP("q_rope", ds4_gpu_qwen_rope_rotate_half_rows_tensor(
        tq, L, n_head, hd, noise_pos, 10000000.0f));
    DFLASH2_STEP("k_rope", ds4_gpu_qwen_rope_rotate_half_rows_tensor(
        tkp, L, n_kv, hd, noise_pos, 10000000.0f));
    DFLASH2_STEP("kc_rope", ds4_gpu_qwen_rope_rotate_half_rows_tensor(
        tkc, n_ctx_new, n_kv, hd, noise_pos - n_ctx_new, 10000000.0f));
    ds4_gpu_tensor *attn_k_ctx = tkc;
    ds4_gpu_tensor *attn_v_ctx = tvc;
    uint32_t n_ctx = n_ctx_new;
    if (cfg->classic && ok) {
        const uint64_t row_bytes = (uint64_t)kv_dim * sizeof(float);
        DFLASH2_STEP("cache_k", ds4_gpu_tensor_copy(
            cache->k, (uint64_t)cache->len * row_bytes,
            tkc, 0, (uint64_t)n_ctx_new * row_bytes));
        DFLASH2_STEP("cache_v", ds4_gpu_tensor_copy(
            cache->v, (uint64_t)cache->len * row_bytes,
            tvc, 0, (uint64_t)n_ctx_new * row_bytes));
        if (ok) {
            cache->len += n_ctx_new;
            n_ctx = cache->len;
            attn_k_ctx = cache->k;
            attn_v_ctx = cache->v;
        }
    }
    const bool causal = cfg->classic && layer_index != cfg->full_attention_layer;
    const uint32_t window = causal ? cfg->sliding_window : 0u;
    DFLASH2_STEP("sdpa", ds4_gpu_dflash2_sdpa(
        theads, tq, attn_k_ctx, attn_v_ctx, tkp, tvp,
        L, n_ctx, n_head, n_kv, hd, causal, window));
    DFLASH2_STEP("attn_o", dflash2_gpu_gemm(tattn, m, l->attn_output, theads, L));
    if (!cfg->classic) {
        DFLASH2_STEP("attn_conv1", ds4_gpu_dflash2_grouped_conv(tfin, tattn, tdyn, m->map, m->size, l->attn_conv_base->abs_offset, L, H, cfg->conv_kernel, cfg->conv_group, 1));
    }
    DFLASH2_STEP("attn_add", ds4_gpu_add_tensor(th, th, cfg->classic ? tattn : tfin, L * H));
    DFLASH2_STEP("ffn_rms", ds4_gpu_rms_norm_weight_rows_tensor(txn, th, m->map, m->size, l->ffn_norm->abs_offset, H, L, 1e-6f));
    if (!cfg->classic) {
        DFLASH2_STEP("ffn_dyn", dflash2_gpu_gemm(tdyn, m, l->ffn_conv_proj, txn, L));
        DFLASH2_STEP("ffn_conv0", ds4_gpu_dflash2_grouped_conv(tprep, txn, tdyn, m->map, m->size, l->ffn_conv_base->abs_offset, L, H, cfg->conv_kernel, cfg->conv_group, 0));
    }
    const ds4_gpu_tensor *ffn_input = cfg->classic ? txn : tprep;
    int fused_ffn = 0;
    if (ok && l->ffn_gate->type == DS4_TENSOR_Q4_K &&
        l->ffn_up->type == DS4_TENSOR_Q4_K &&
        ds4_gpu_matmul_q4_k_pair_swiglu_rows_tensor(
            tmid, m->map, m->size,
            l->ffn_gate->abs_offset, l->ffn_up->abs_offset,
            DS4_TENSOR_Q4_K, H, ff, ffn_input, L)) {
        fused_ffn = 1;
    }
    if (!fused_ffn) {
        DFLASH2_STEP("ffn_gate", dflash2_gpu_gemm(
            tgate, m, l->ffn_gate, ffn_input, L));
        DFLASH2_STEP("ffn_up", dflash2_gpu_gemm(
            tup, m, l->ffn_up, ffn_input, L));
        DFLASH2_STEP("swiglu", ds4_gpu_swiglu_tensor(
            tmid, tgate, tup, L * ff, 0.0f, 1.0f));
    }
    DFLASH2_STEP("ffn_down", dflash2_gpu_gemm(
        tffn, m, l->ffn_down, tmid, L));
    if (!cfg->classic) {
        DFLASH2_STEP("ffn_conv1", ds4_gpu_dflash2_grouped_conv(tfin, tffn, tdyn, m->map, m->size, l->ffn_conv_base->abs_offset, L, H, cfg->conv_kernel, cfg->conv_group, 1));
    }
    DFLASH2_STEP("ffn_add", ds4_gpu_add_tensor(th, th, cfg->classic ? tffn : tfin, L * H));
    #undef DFLASH2_STEP
    if (!ok) {
        static int nfail;
        if (nfail++ < 6) fprintf(stderr, "ds4: dflash gpu step fail %s L=%u\n", fail ? fail : "?", L);
    }
    if (started) {
        if (!ds4_gpu_end_commands()) ok = 0;
        if (ok && !ds4_gpu_tensor_read(th, 0, h, bH)) ok = 0;
        else if (!ok) (void)ds4_gpu_end_commands();
        h_live = 0;
    }
    return ok;
}

static int dflash2_gpu_finish_h(float *h, uint32_t L, uint32_t H) {
    h_live = 0;
    if (!ds4_gpu_end_commands()) return 0;
    return ds4_gpu_tensor_read(th, 0, h, (uint64_t)L * H * sizeof(float));
}
#endif


static void dflash2_layer_forward(
        float *h,
        const float *h_ctx,
        const ds4_model *m,
        const ds4_dflash2_layer *l,
        const ds4_dflash2_weights *cfg,
        uint32_t L,
        uint32_t ctx_pos) {
    const uint32_t H = cfg->n_embd;
#if defined(__APPLE__) && !defined(DS4_NO_GPU)
    {
        const int gpu_ok = dflash2_layer_forward_gpu(
            h, h_ctx, 1u, NULL, 0u, m, l, cfg, L, ctx_pos + 1u);
        static int nlog;
        if (nlog < 2) {
            fprintf(stderr, "ds4: dflash layer gpu=%d L=%u\n", gpu_ok, L);
            nlog++;
        }
        if (gpu_ok) return;
    }
#endif

    const uint32_t ff = cfg->n_ff;
    const uint32_t n_head = cfg->n_head;
    const uint32_t n_kv = cfg->n_head_kv;
    const uint32_t hd = cfg->head_dim;
    const uint32_t q_dim = n_head * hd;
    const uint32_t kv_dim = n_kv * hd;
    const uint32_t groups = cfg->classic ? 1u : H / cfg->conv_group;
    const uint32_t dyn_dim = cfg->classic ? H : 2u * cfg->conv_kernel * groups;
    const float eps = 1e-6f;
    float *xn = xmalloc((size_t)L * H * sizeof(float));
    float *prep = xmalloc((size_t)L * H * sizeof(float));
    float *dyn = xmalloc((size_t)L * dyn_dim * sizeof(float));
    float *q = xmalloc((size_t)L * q_dim * sizeof(float));
    float *k_prop = xmalloc((size_t)L * kv_dim * sizeof(float));
    float *v_prop = xmalloc((size_t)L * kv_dim * sizeof(float));
    float *k_ctx = xmalloc((size_t)kv_dim * sizeof(float));
    float *v_ctx = xmalloc((size_t)kv_dim * sizeof(float));
    float *heads = xmalloc((size_t)L * q_dim * sizeof(float));
    float *attn = xmalloc((size_t)L * H * sizeof(float));
    float *fin = xmalloc((size_t)L * H * sizeof(float));

    for (uint32_t t = 0; t < L; t++) {
        rms_norm_weight(xn + (size_t)t * H, h + (size_t)t * H,
                        tensor_data(m, l->attn_norm), H, eps);
    }
    if (!cfg->classic) {
        dflash2_matvecs(dyn, m, l->attn_conv_proj, xn, L);
        float *dyn0 = xmalloc((size_t)L * cfg->conv_kernel * groups * sizeof(float));
        const uint32_t half = cfg->conv_kernel * groups;
        for (uint32_t t = 0; t < L; t++) {
            memcpy(dyn0 + (size_t)t * half, dyn + (size_t)t * 2u * half, (size_t)half * sizeof(float));
        }
        dflash2_grouped_conv(prep, xn, dyn0, tensor_data(m, l->attn_conv_base),
                             0, L, H, cfg->conv_kernel, cfg->conv_group);
        free(dyn0);
    }
    {
        const ds4_tensor *qkv_w[3] = { l->attn_q, l->attn_k, l->attn_v };
        float *qkv_o[3] = { q, k_prop, v_prop };
        dflash2_matvecs_n(qkv_o, m, qkv_w, cfg->classic ? xn : prep, L, 3);
        const ds4_tensor *ctx_w[2] = { l->attn_k, l->attn_v };
        float *ctx_o[2] = { k_ctx, v_ctx };
        dflash2_matvecs_n(ctx_o, m, ctx_w, h_ctx, 1, 2);
    }

    for (uint32_t t = 0; t < L; t++) {
        dflash2_rms_head(q + (size_t)t * q_dim, tensor_data(m, l->attn_q_norm), n_head, hd, eps);
        dflash2_rms_head(k_prop + (size_t)t * kv_dim, tensor_data(m, l->attn_k_norm), n_kv, hd, eps);
    }
    dflash2_rms_head(k_ctx, tensor_data(m, l->attn_k_norm), n_kv, hd, eps);
    dflash2_rope_rows(q, L, n_head, hd, ctx_pos + 1u, 10000000.0f);
    dflash2_rope_rows(k_prop, L, n_kv, hd, ctx_pos + 1u, 10000000.0f);
    qwen_rope_rotate_half_inplace(
        k_ctx, n_kv, hd, hd, ctx_pos, 10000000.0f);
    dflash2_sdpa(heads, q, k_ctx, v_ctx, k_prop, v_prop, L, n_head, n_kv, hd);
    dflash2_matvecs(attn, m, l->attn_output, heads, L);
    if (cfg->classic) {
        for (uint32_t i = 0; i < L * H; i++) h[i] += attn[i];
    } else {
        dflash2_conv_finish(fin, attn, dyn, m, l->attn_conv_base, L, H, cfg->conv_kernel, cfg->conv_group);
        for (uint32_t i = 0; i < L * H; i++) h[i] += fin[i];
    }

    for (uint32_t t = 0; t < L; t++) {
        rms_norm_weight(xn + (size_t)t * H, h + (size_t)t * H,
                        tensor_data(m, l->ffn_norm), H, eps);
    }
    if (!cfg->classic) {
        dflash2_matvecs(dyn, m, l->ffn_conv_proj, xn, L);
        float *dyn0 = xmalloc((size_t)L * cfg->conv_kernel * groups * sizeof(float));
        const uint32_t half = cfg->conv_kernel * groups;
        for (uint32_t t = 0; t < L; t++) {
            memcpy(dyn0 + (size_t)t * half, dyn + (size_t)t * 2u * half, (size_t)half * sizeof(float));
        }
        dflash2_grouped_conv(prep, xn, dyn0, tensor_data(m, l->ffn_conv_base),
                             0, L, H, cfg->conv_kernel, cfg->conv_group);
        free(dyn0);
    }
    float *gate = xmalloc((size_t)L * ff * sizeof(float));
    float *up = xmalloc((size_t)L * ff * sizeof(float));
    float *mid = xmalloc((size_t)L * ff * sizeof(float));
    float *ffn = xmalloc((size_t)L * H * sizeof(float));
    {
        const ds4_tensor *gu_w[2] = { l->ffn_gate, l->ffn_up };
        float *gu_o[2] = { gate, up };
        dflash2_matvecs_n(gu_o, m, gu_w, cfg->classic ? xn : prep, L, 2);
    }
    for (uint32_t t = 0; t < L; t++) {
        dflash2_swiglu(mid + (size_t)t * ff, gate + (size_t)t * ff, up + (size_t)t * ff, ff);
    }
    dflash2_matvecs(ffn, m, l->ffn_down, mid, L);

    if (cfg->classic) {
        for (uint32_t i = 0; i < L * H; i++) h[i] += ffn[i];
    } else {
        dflash2_conv_finish(fin, ffn, dyn, m, l->ffn_conv_base, L, H, cfg->conv_kernel, cfg->conv_group);
        for (uint32_t i = 0; i < L * H; i++) h[i] += fin[i];
    }

    free(ffn); free(mid); free(up); free(gate);
    free(fin); free(attn); free(heads);
    free(v_ctx); free(k_ctx); free(v_prop); free(k_prop); free(q);
    free(dyn); free(prep); free(xn);
}

/* Opt-in draft-confidence floor: drop trailing drafts whose top1-top2 logit
 * margin falls below DS4_DFLASH_MIN_MARGIN. Absent or negative = disabled
 * (default), so no logits readback or behavior change unless opted in. */
static float dflash2_min_margin(void) {
    static int cached;
    static float margin = -1.0f;
    if (!cached) {
        cached = 1;
        const char *value = getenv("DS4_DFLASH_MIN_MARGIN");
        if (value && value[0]) {
            char *end = NULL;
            const double parsed = strtod(value, &end);
            if (end != value && parsed >= 0.0) margin = (float)parsed;
        }
    }
    return margin;
}

static int dflash2_propose(
        int *drafts,
        int n_draft,
        const ds4_model *dm,
        const ds4_dflash2_weights *dw,
        const ds4_model *tm,
        const ds4_weights *tw,
        const float *target_concat,
        uint32_t target_rows,
        ds4_dflash_cache *cache,
        int anchor,
        uint32_t noise_pos) {
    if (n_draft <= 0 || target_rows == 0) return 0;
    if (dw->classic && (!cache || noise_pos < target_rows)) return 0;
    const uint32_t H = dw->n_embd;
    const uint32_t L = (uint32_t)n_draft + 1u;
    /* Per-row top1-top2 logit margin; INFINITY = unknown (no truncation). */
    float margins[DS4_DFLASH2_MAX_BLOCK];
    for (int t = 0; t < DS4_DFLASH2_MAX_BLOCK; t++) margins[t] = 1e30f;
    float *h_ctx = xmalloc((size_t)target_rows * H * sizeof(float));
    float *fc_out = xmalloc((size_t)target_rows * H * sizeof(float));
    dflash2_matvecs(fc_out, dm, dw->fc, target_concat, target_rows);
    for (uint32_t t = 0; t < target_rows; t++) {
        rms_norm_weight(h_ctx + (size_t)t * H, fc_out + (size_t)t * H,
                        tensor_data(dm, dw->enc_output_norm), H, 1e-6f);
    }

    float *h = xmalloc((size_t)L * H * sizeof(float));
    embed_token_any(tm, tw, anchor, h);
    for (uint32_t t = 1; t < L; t++) {
        embed_token_any(tm, tw, dw->mask_token_id, h + (size_t)t * H);
    }
    int gpu_all = 0;
#if defined(__APPLE__) && !defined(DS4_NO_GPU)
    h_live = 0;
    if (g_qwen_pool.inited && ds4_gpu_begin_commands()) {
        int ok = 1;
        uint32_t fail_il = 0;
        for (uint32_t il = 0; ok && il < dw->n_layer; il++) {
            ds4_dflash_cache_layer *layer_cache =
                dw->classic ? &cache->layer[il] : NULL;
            ok = dflash2_layer_forward_gpu(
                h, h_ctx, target_rows, layer_cache, il,
                dm, &dw->layer[il], dw, L, noise_pos);
            if (!ok) fail_il = il;
            else (void)ds4_gpu_flush_encoder();
        }
        if (ok) ok = dflash2_gpu_finish_h(h, L, H);
        else (void)ds4_gpu_end_commands();
        gpu_all = ok;
        static int nbat;
        if (nbat++ < 3) fprintf(stderr, "ds4: dflash batched gpu_all=%d fail_il=%u n_layer=%u\n",
                                gpu_all, fail_il, dw->n_layer);
    } else if (g_qwen_pool.inited) {
        static int nbeg;
        if (nbeg++ < 2) fprintf(stderr, "ds4: dflash batched begin fail inited=%d\n", g_qwen_pool.inited);
    }
#endif
    if (!gpu_all) {
        if (dw->classic) ds4_die("classic DFlash Metal forward failed");
        for (uint32_t il = 0; il < dw->n_layer; il++) {
            dflash2_layer_forward(h, h_ctx, dm, &dw->layer[il], dw, L, noise_pos - 1u);
        }
    }
    float *normed = xmalloc((size_t)L * H * sizeof(float));
    for (uint32_t t = 0; t < L; t++) {
        rms_norm_weight(normed + (size_t)t * H, h + (size_t)t * H,
                        tensor_data(dm, dw->output_norm), H, 1e-6f);
    }

    int n_out = 0;
#ifndef DS4_NO_GPU
    if (g_qwen_pool.inited && g_qwen_pool.batch_norm &&
        g_qwen_pool.batch_logits && g_qwen_pool.batch_argmax && tw->output &&
        (uint32_t)n_draft <= g_qwen_pool.spec_cap &&
        ds4_gpu_tensor_write(
            g_qwen_pool.batch_norm, 0, normed + H,
            (uint64_t)n_draft * H * sizeof(float)) &&
        ds4_gpu_begin_commands()) {
        int ok = ds4_gpu_matmul_quant_tensor(
            g_qwen_pool.batch_logits, tm->map, tm->size,
            tw->output->abs_offset, tw->output->type,
            H, DS4_N_VOCAB, g_qwen_pool.batch_norm, (uint32_t)n_draft);
        if (!ok) {
            ok = 1;
            for (int t = 0; ok && t < n_draft; t++) {
                ds4_gpu_tensor *row_n = ds4_gpu_tensor_view(
                    g_qwen_pool.batch_norm, (uint64_t)t * H * sizeof(float),
                    (uint64_t)H * sizeof(float));
                ds4_gpu_tensor *row_l = ds4_gpu_tensor_view(
                    g_qwen_pool.batch_logits,
                    (uint64_t)t * DS4_N_VOCAB * sizeof(float),
                    (uint64_t)DS4_N_VOCAB * sizeof(float));
                ok = row_n && row_l && ds4_gpu_matmul_quant_tensor(
                    row_l, tm->map, tm->size,
                    tw->output->abs_offset, tw->output->type,
                    H, DS4_N_VOCAB, row_n, 1);
                ds4_gpu_tensor_free(row_n);
                ds4_gpu_tensor_free(row_l);
            }
        }
        if (ok) {
            ok = ds4_gpu_indexer_topk_tensor(
                g_qwen_pool.batch_argmax, g_qwen_pool.batch_logits,
                DS4_N_VOCAB, (uint32_t)n_draft, 1) != 0;
        }
        if (!ds4_gpu_end_commands()) ok = 0;
        int32_t idx[DS4_DFLASH2_MAX_BLOCK];
        if (ok && ds4_gpu_tensor_read(
                g_qwen_pool.batch_argmax, 0, idx,
                (uint64_t)n_draft * sizeof(int32_t))) {
            for (int t = 0; t < n_draft; t++) {
                if (idx[t] < 0 || idx[t] >= (int32_t)DS4_N_VOCAB) {
                    ok = 0;
                    break;
                }
                drafts[t] = (int)idx[t];
            }
            if (ok) n_out = n_draft;
        }
        if (ok && n_out == n_draft && dflash2_min_margin() >= 0.0f) {
            const size_t nrows = (size_t)n_draft * DS4_N_VOCAB;
            float *lrows = xmalloc(nrows * sizeof(float));
            if (ds4_gpu_tensor_read(
                    g_qwen_pool.batch_logits, 0, lrows,
                    (uint64_t)nrows * sizeof(float))) {
                for (int t = 0; t < n_draft; t++) {
                    float best = -1e30f, second = -1e30f;
                    const float *row = lrows + (size_t)t * DS4_N_VOCAB;
                    for (uint32_t i = 0; i < DS4_N_VOCAB; i++) {
                        if (row[i] > best) {
                            second = best;
                            best = row[i];
                        } else if (row[i] > second) {
                            second = row[i];
                        }
                    }
                    margins[t] = best - second;
                }
            }
            free(lrows);
        }
    } else {
        (void)ds4_gpu_end_commands();
    }
#endif
    if (n_out == 0) {
        float *logits = xmalloc((size_t)DS4_N_VOCAB * sizeof(float));
        for (uint32_t t = 1; t < L && n_out < n_draft; t++) {
            dflash2_matvec(logits, tm, tw->output, normed + (size_t)t * H);
            int best = 0;
            float best_v = logits[0], second_v = -1e30f;
            for (uint32_t i = 1; i < DS4_N_VOCAB; i++) {
                if (logits[i] > best_v) {
                    second_v = best_v;
                    best_v = logits[i];
                    best = (int)i;
                } else if (logits[i] > second_v) {
                    second_v = logits[i];
                }
            }
            drafts[n_out] = best;
            margins[n_out] = best_v - second_v;
            n_out++;
        }
        free(logits);
    }
    const float min_margin = dflash2_min_margin();
    if (min_margin >= 0.0f) {
        while (n_out > 0 && margins[n_out - 1] < min_margin) n_out--;
    }
    free(normed); free(h); free(fc_out); free(h_ctx);
    return n_out;
}

static void qwen_forward_token_cpu_layers(
        float *logits,
        float *hidden_out,
        float *layer_out,
        const uint32_t *layer_ids,
        uint32_t n_ids,
        const ds4_model *model,
        const ds4_weights *weights,
        int token,
        uint32_t pos) {
    const uint32_t n_embd = 5120;
    float *cur = xmalloc((size_t)n_embd * sizeof(float));
    float *next = xmalloc((size_t)n_embd * sizeof(float));
    embed_token_any(model, weights, token, cur);
    for (uint32_t il = 0; il < 64; il++) {
        qwen_layer_forward_cpu(next, model, &weights->layer[il], cur, pos, il);
        float *tmp = cur; cur = next; next = tmp;
        if (layer_out && layer_ids) {
            for (uint32_t j = 0; j < n_ids; j++) {
                if (layer_ids[j] == il) {
                    memcpy(layer_out + (size_t)j * n_embd, cur, (size_t)n_embd * sizeof(float));
                }
            }
        }
    }
    if (hidden_out) memcpy(hidden_out, cur, (size_t)n_embd * sizeof(float));
    if (logits) {
        float *norm = xmalloc((size_t)n_embd * sizeof(float));
        rms_norm_weight(norm, cur, tensor_data(model, weights->output_norm), n_embd, 1e-6f);
        matvec_any(logits, model, weights->output, norm);
        free(norm);
    }
    free(next);
    free(cur);
}
