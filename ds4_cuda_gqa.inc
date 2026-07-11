/* GQA full attention for Hy3-class models (hy_v3: 64 q heads / 8 kv heads,
 * head_dim 128, per-head QK-RMSNorm with weights, rotate-half RoPE).
 *
 * ds4's existing attention paths are all MLA-family (shared latent KV, no
 * per-kv-head cache); this file adds the grouped-query path: a standard
 * per-head KV cache [n_kv_head][cap][head_dim] and a decode/prefill
 * attention kernel where q head h attends kv head h / (n_head / n_kv_head).
 *
 * Everything here is validated by ds4_gpu_gqa_selftest(): a CPU reference
 * implementation runs the identical pipeline on random data and the kernels
 * must match it numerically. Run with `ds4 --gqa-selftest` (no model file
 * needed), including a multi-chunk phase for the online-softmax rescale.
 *
 * RoPE convention: HF Llama rotate-half. Pair (i, i + d/2), angle
 * pos * theta^(-2i/d). Hy3 rope_type is "default" (no yarn/ntk scaling). */

/* ---- kernels ---- */

/* Per-head RMSNorm with a [head_dim] weight vector, in place.
 * x: [rows][head_dim] where rows = n_tok * n_head (or n_kv_head). */
__global__ static void gqa_head_rms_norm_weight_kernel(
        float *x,
        const float *w,
        uint32_t rows,
        uint32_t head_dim,
        float eps) {
    uint32_t row = blockIdx.x;
    if (row >= rows) return;
    float *v = x + (uint64_t)row * head_dim;
    __shared__ float red[256];
    float acc = 0.0f;
    for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        acc += v[d] * v[d];
    }
    red[threadIdx.x] = acc;
    __syncthreads();
    for (uint32_t s = blockDim.x / 2u; s != 0; s >>= 1u) {
        if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    float inv = rsqrtf(red[0] / (float)head_dim + eps);
    for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        v[d] = v[d] * inv * w[d];
    }
}

/* Rotate-half RoPE over the full head, in place.
 * x: [n_tok][n_head][head_dim], token t is at position pos0 + t. */
__global__ static void gqa_rope_neox_kernel(
        float *x,
        uint32_t n_tok,
        uint32_t n_head,
        uint32_t head_dim,
        uint32_t pos0,
        float theta) {
    uint32_t half = head_dim / 2u;
    uint64_t total = (uint64_t)n_tok * n_head * half;
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    uint32_t d = (uint32_t)(i % half);
    uint32_t rem = (uint32_t)(i / half);
    uint32_t h = rem % n_head;
    uint32_t t = rem / n_head;
    float *v = x + ((uint64_t)t * n_head + h) * head_dim;
    float freq = powf(theta, -2.0f * (float)d / (float)head_dim);
    float ang = (float)(pos0 + t) * freq;
    float c = cosf(ang), s = sinf(ang);
    float a = v[d], b = v[d + half];
    v[d] = a * c - b * s;
    v[d + half] = a * s + b * c;
}

/* Append n_tok tokens of K and V into the per-head cache.
 * kv_in: [n_tok][n_kv_head][head_dim]; cache: [n_kv_head][cap][head_dim]. */
__global__ static void gqa_kv_cache_append_kernel(
        float *cache,
        const float *kv_in,
        uint32_t n_tok,
        uint32_t n_kv_head,
        uint32_t head_dim,
        uint32_t cap,
        uint32_t pos0) {
    uint64_t total = (uint64_t)n_tok * n_kv_head * head_dim;
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    uint32_t d = (uint32_t)(i % head_dim);
    uint32_t rem = (uint32_t)(i / head_dim);
    uint32_t h = rem % n_kv_head;
    uint32_t t = rem / n_kv_head;
    cache[((uint64_t)h * cap + pos0 + t) * head_dim + d] =
        kv_in[((uint64_t)t * n_kv_head + h) * head_dim + d];
}

/* Causal GQA attention over the cache, single-pass online softmax.
 * q: [n_tok][n_head][head_dim] (post norm+rope); token t sees cache rows
 * [0, pos0 + t]. out: [n_tok][n_head][head_dim]. One block per (t, h). */
__global__ static void gqa_attention_kernel(
        float *out,
        const float *q,
        const float *k_cache,
        const float *v_cache,
        uint32_t n_tok,
        uint32_t n_head,
        uint32_t n_kv_head,
        uint32_t head_dim,
        uint32_t cap,
        uint32_t pos0) {
    uint32_t t = blockIdx.x;
    uint32_t h = blockIdx.y;
    if (t >= n_tok || h >= n_head) return;
    uint32_t kvh = h / (n_head / n_kv_head);
    uint32_t n_rows = pos0 + t + 1u;
    const float *qh = q + ((uint64_t)t * n_head + h) * head_dim;
    const float *kh = k_cache + (uint64_t)kvh * cap * head_dim;
    const float *vh = v_cache + (uint64_t)kvh * cap * head_dim;
    float scale = rsqrtf((float)head_dim);
    /* Online softmax, single pass over K and V.  128 threads: thread d owns
     * output dim d; rows are processed in chunks of 128 with each thread
     * computing one row's score, then everyone accumulates V against the
     * shared per-chunk weights.  Scores are computed exactly once (the old
     * two-pass version recomputed them per output dim: ~130x the FLOPs and,
     * past L1, ~head_dim re-reads of K per row at long context). */
    const uint32_t tid = threadIdx.x;
    __shared__ float sq[128];
    __shared__ float s_w[128];
    __shared__ float red[128];
    if (tid < head_dim) sq[tid] = qh[tid];
    __syncthreads();

    float acc = 0.0f;   /* V accumulator for dim = tid */
    float den = 0.0f;   /* running softmax denominator (same in all threads) */
    float m = -INFINITY;/* running max */
    for (uint32_t base = 0; base < n_rows; base += 128u) {
        uint32_t chunk = n_rows - base < 128u ? n_rows - base : 128u;
        /* one score per thread */
        float s = -INFINITY;
        if (tid < chunk) {
            const float *krow = kh + (uint64_t)(base + tid) * head_dim;
            float dot = 0.0f;
            for (uint32_t d = 0; d < head_dim; d++) dot += sq[d] * krow[d];
            s = dot * scale;
        }
        red[tid] = s;
        __syncthreads();
        for (uint32_t st = 64u; st != 0; st >>= 1u) {
            if (tid < st) red[tid] = fmaxf(red[tid], red[tid + st]);
            __syncthreads();
        }
        float m_new = fmaxf(m, red[0]);
        float corr = expf(m - m_new); /* expf(-inf) == 0, first chunk safe */
        acc *= corr;
        den *= corr;
        if (tid < chunk) s_w[tid] = expf(s - m_new);
        __syncthreads();
        for (uint32_t r = 0; r < chunk; r++) {
            float w = s_w[r];
            den += w;
            if (tid < head_dim) {
                acc += w * vh[(uint64_t)(base + r) * head_dim + tid];
            }
        }
        m = m_new;
        __syncthreads();
    }
    if (tid < head_dim) {
        out[((uint64_t)t * n_head + h) * head_dim + tid] = acc / den;
    }
}

/* ---- host wrappers ---- */

extern "C" int ds4_gpu_gqa_head_rms_norm_weight(
        ds4_gpu_tensor *x, const ds4_gpu_tensor *w,
        uint32_t rows, uint32_t head_dim, float eps) {
    if (!x || !w || rows == 0 || head_dim == 0 || head_dim > 128u) return 0;
    gqa_head_rms_norm_weight_kernel<<<rows, 256>>>(
            (float *)x->ptr, (const float *)w->ptr, rows, head_dim, eps);
    return cuda_ok(cudaGetLastError(), "gqa head rms norm launch");
}

extern "C" int ds4_gpu_gqa_rope(
        ds4_gpu_tensor *x, uint32_t n_tok, uint32_t n_head,
        uint32_t head_dim, uint32_t pos0, float theta) {
    if (!x || n_tok == 0 || n_head == 0 || head_dim == 0 || head_dim % 2u) return 0;
    uint64_t total = (uint64_t)n_tok * n_head * (head_dim / 2u);
    gqa_rope_neox_kernel<<<(uint32_t)((total + 255u) / 256u), 256>>>(
            (float *)x->ptr, n_tok, n_head, head_dim, pos0, theta);
    return cuda_ok(cudaGetLastError(), "gqa rope launch");
}

extern "C" int ds4_gpu_gqa_kv_cache_append(
        ds4_gpu_tensor *cache, const ds4_gpu_tensor *kv_in,
        uint32_t n_tok, uint32_t n_kv_head, uint32_t head_dim,
        uint32_t cap, uint32_t pos0) {
    if (!cache || !kv_in || n_tok == 0 || pos0 + n_tok > cap) return 0;
    uint64_t total = (uint64_t)n_tok * n_kv_head * head_dim;
    gqa_kv_cache_append_kernel<<<(uint32_t)((total + 255u) / 256u), 256>>>(
            (float *)cache->ptr, (const float *)kv_in->ptr,
            n_tok, n_kv_head, head_dim, cap, pos0);
    return cuda_ok(cudaGetLastError(), "gqa kv append launch");
}

extern "C" int ds4_gpu_gqa_attention(
        ds4_gpu_tensor *out, const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *k_cache, const ds4_gpu_tensor *v_cache,
        uint32_t n_tok, uint32_t n_head, uint32_t n_kv_head,
        uint32_t head_dim, uint32_t cap, uint32_t pos0) {
    if (!out || !q || !k_cache || !v_cache ||
        n_head == 0 || n_kv_head == 0 || n_head % n_kv_head != 0 ||
        head_dim == 0 || head_dim > 128u || pos0 + n_tok > cap) {
        return 0;
    }
    dim3 grid(n_tok, n_head, 1);
    gqa_attention_kernel<<<grid, 128>>>(
            (float *)out->ptr, (const float *)q->ptr,
            (const float *)k_cache->ptr, (const float *)v_cache->ptr,
            n_tok, n_head, n_kv_head, head_dim, cap, pos0);
    return cuda_ok(cudaGetLastError(), "gqa attention launch");
}

/* ---- CPU reference + self-test ---- */

static void gqa_ref_head_rms_norm(float *x, const float *w,
                                  uint32_t rows, uint32_t hd, float eps) {
    for (uint32_t r = 0; r < rows; r++) {
        float *v = x + (uint64_t)r * hd;
        double ss = 0.0;
        for (uint32_t d = 0; d < hd; d++) ss += (double)v[d] * v[d];
        float inv = 1.0f / sqrtf((float)(ss / hd) + eps);
        for (uint32_t d = 0; d < hd; d++) v[d] = v[d] * inv * w[d];
    }
}

static void gqa_ref_rope(float *x, uint32_t n_tok, uint32_t n_head,
                         uint32_t hd, uint32_t pos0, float theta) {
    uint32_t half = hd / 2u;
    for (uint32_t t = 0; t < n_tok; t++)
    for (uint32_t h = 0; h < n_head; h++) {
        float *v = x + ((uint64_t)t * n_head + h) * hd;
        for (uint32_t d = 0; d < half; d++) {
            float freq = powf(theta, -2.0f * (float)d / (float)hd);
            float ang = (float)(pos0 + t) * freq;
            float c = cosf(ang), s = sinf(ang);
            float a = v[d], b = v[d + half];
            v[d] = a * c - b * s;
            v[d + half] = a * s + b * c;
        }
    }
}

static void gqa_ref_attention(float *out, const float *q,
                              const float *kc, const float *vc,
                              uint32_t n_tok, uint32_t n_head,
                              uint32_t n_kv_head, uint32_t hd,
                              uint32_t cap, uint32_t pos0) {
    float scale = 1.0f / sqrtf((float)hd);
    uint32_t group = n_head / n_kv_head;
    float *scores = (float *)malloc((size_t)(pos0 + n_tok) * sizeof(float));
    for (uint32_t t = 0; t < n_tok; t++)
    for (uint32_t h = 0; h < n_head; h++) {
        uint32_t kvh = h / group;
        uint32_t n_rows = pos0 + t + 1u;
        const float *qh = q + ((uint64_t)t * n_head + h) * hd;
        const float *kh = kc + (uint64_t)kvh * cap * hd;
        const float *vh = vc + (uint64_t)kvh * cap * hd;
        float m = -INFINITY;
        for (uint32_t r = 0; r < n_rows; r++) {
            float dot = 0.0f;
            for (uint32_t d = 0; d < hd; d++) dot += qh[d] * kh[(uint64_t)r * hd + d];
            scores[r] = dot * scale;
            if (scores[r] > m) m = scores[r];
        }
        float den = 0.0f;
        for (uint32_t r = 0; r < n_rows; r++) den += expf(scores[r] - m);
        float *oh = out + ((uint64_t)t * n_head + h) * hd;
        for (uint32_t d = 0; d < hd; d++) {
            float acc = 0.0f;
            for (uint32_t r = 0; r < n_rows; r++) {
                acc += expf(scores[r] - m) / den * vh[(uint64_t)r * hd + d];
            }
            oh[d] = acc;
        }
    }
    free(scores);
}

static float gqa_max_abs_diff(const float *a, const float *b, uint64_t n) {
    float m = 0.0f;
    for (uint64_t i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

static uint32_t gqa_test_rng_state = 0x1234567u;
static float gqa_test_randf(void) {
    gqa_test_rng_state = gqa_test_rng_state * 1664525u + 1013904223u;
    return ((float)(gqa_test_rng_state >> 8) / (float)(1u << 24)) * 2.0f - 1.0f;
}

/* Full-pipeline self-test on random data: qk-norm -> rope -> cache append
 * -> causal GQA attention, CPU reference vs kernels. Hy3 dims: 64/8 heads,
 * head_dim 128. Covers a 13-token prefill at pos0=0 and a 1-token decode at
 * pos0=13 against the populated cache. Returns 1 on pass. */
extern "C" int ds4_gpu_gqa_selftest(void) {
    const uint32_t n_head = 64, n_kv_head = 8, hd = 128, cap = 256;
    const float eps = 1e-5f, theta = 11158840.0f;
    /* {n_tok, pos0}: short prefill, decode, then a 200-token prefill and a
     * decode at row 215 to exercise the multi-chunk online-softmax path
     * (rows > 128 crosses a chunk boundary and rescales acc/den). */
    const uint32_t phases[4][2] = { {13u, 0u}, {1u, 13u}, {200u, 14u}, {1u, 214u} };
    int all_ok = 1;

    float *qn_w = (float *)malloc(hd * sizeof(float));
    float *kn_w = (float *)malloc(hd * sizeof(float));
    for (uint32_t d = 0; d < hd; d++) { qn_w[d] = 0.5f + 0.5f * fabsf(gqa_test_randf()); kn_w[d] = 0.5f + 0.5f * fabsf(gqa_test_randf()); }

    uint64_t kcache_elems = (uint64_t)n_kv_head * cap * hd;
    float *ref_kc = (float *)calloc(kcache_elems, sizeof(float));
    float *ref_vc = (float *)calloc(kcache_elems, sizeof(float));

    ds4_gpu_tensor *g_qnw = ds4_gpu_tensor_alloc(hd * sizeof(float));
    ds4_gpu_tensor *g_knw = ds4_gpu_tensor_alloc(hd * sizeof(float));
    ds4_gpu_tensor *g_kc = ds4_gpu_tensor_alloc(kcache_elems * sizeof(float));
    ds4_gpu_tensor *g_vc = ds4_gpu_tensor_alloc(kcache_elems * sizeof(float));
    if (!g_qnw || !g_knw || !g_kc || !g_vc) { fprintf(stderr, "gqa-selftest: alloc failed\n"); return 0; }
    ds4_gpu_tensor_write(g_qnw, 0, qn_w, hd * sizeof(float));
    ds4_gpu_tensor_write(g_knw, 0, kn_w, hd * sizeof(float));
    { float *z = (float *)calloc(kcache_elems, sizeof(float));
      ds4_gpu_tensor_write(g_kc, 0, z, kcache_elems * sizeof(float));
      ds4_gpu_tensor_write(g_vc, 0, z, kcache_elems * sizeof(float));
      free(z); }

    for (int phase = 0; phase < 4; phase++) {
        uint32_t n_tok = phases[phase][0], pos0 = phases[phase][1];
        uint64_t q_elems = (uint64_t)n_tok * n_head * hd;
        uint64_t kv_elems = (uint64_t)n_tok * n_kv_head * hd;
        float *q = (float *)malloc(q_elems * sizeof(float));
        float *k = (float *)malloc(kv_elems * sizeof(float));
        float *v = (float *)malloc(kv_elems * sizeof(float));
        for (uint64_t i = 0; i < q_elems; i++) q[i] = gqa_test_randf();
        for (uint64_t i = 0; i < kv_elems; i++) { k[i] = gqa_test_randf(); v[i] = gqa_test_randf(); }

        ds4_gpu_tensor *g_q = ds4_gpu_tensor_alloc(q_elems * sizeof(float));
        ds4_gpu_tensor *g_k = ds4_gpu_tensor_alloc(kv_elems * sizeof(float));
        ds4_gpu_tensor *g_v = ds4_gpu_tensor_alloc(kv_elems * sizeof(float));
        ds4_gpu_tensor *g_out = ds4_gpu_tensor_alloc(q_elems * sizeof(float));
        ds4_gpu_tensor_write(g_q, 0, q, q_elems * sizeof(float));
        ds4_gpu_tensor_write(g_k, 0, k, kv_elems * sizeof(float));
        ds4_gpu_tensor_write(g_v, 0, v, kv_elems * sizeof(float));

        /* reference pipeline */
        gqa_ref_head_rms_norm(q, qn_w, n_tok * n_head, hd, eps);
        gqa_ref_head_rms_norm(k, kn_w, n_tok * n_kv_head, hd, eps);
        gqa_ref_rope(q, n_tok, n_head, hd, pos0, theta);
        gqa_ref_rope(k, n_tok, n_kv_head, hd, pos0, theta);
        for (uint32_t t = 0; t < n_tok; t++)
        for (uint32_t h = 0; h < n_kv_head; h++)
        for (uint32_t d = 0; d < hd; d++) {
            ref_kc[((uint64_t)h * cap + pos0 + t) * hd + d] = k[((uint64_t)t * n_kv_head + h) * hd + d];
            ref_vc[((uint64_t)h * cap + pos0 + t) * hd + d] = v[((uint64_t)t * n_kv_head + h) * hd + d];
        }
        float *ref_out = (float *)malloc(q_elems * sizeof(float));
        gqa_ref_attention(ref_out, q, ref_kc, ref_vc, n_tok, n_head, n_kv_head, hd, cap, pos0);

        /* kernel pipeline */
        int ok = ds4_gpu_gqa_head_rms_norm_weight(g_q, g_qnw, n_tok * n_head, hd, eps) &&
                 ds4_gpu_gqa_head_rms_norm_weight(g_k, g_knw, n_tok * n_kv_head, hd, eps) &&
                 ds4_gpu_gqa_rope(g_q, n_tok, n_head, hd, pos0, theta) &&
                 ds4_gpu_gqa_rope(g_k, n_tok, n_kv_head, hd, pos0, theta) &&
                 ds4_gpu_gqa_kv_cache_append(g_kc, g_k, n_tok, n_kv_head, hd, cap, pos0) &&
                 ds4_gpu_gqa_kv_cache_append(g_vc, g_v, n_tok, n_kv_head, hd, cap, pos0) &&
                 ds4_gpu_gqa_attention(g_out, g_q, g_kc, g_vc, n_tok, n_head, n_kv_head, hd, cap, pos0) &&
                 cuda_ok(cudaDeviceSynchronize(), "gqa selftest sync");
        float *gpu_out = (float *)malloc(q_elems * sizeof(float));
        if (ok) ok = ds4_gpu_tensor_read(g_out, 0, gpu_out, q_elems * sizeof(float)) != 0;
        float diff = ok ? gqa_max_abs_diff(ref_out, gpu_out, q_elems) : INFINITY;
        int pass = ok && diff < 2e-4f;
        fprintf(stderr, "gqa-selftest phase %d (n_tok=%u pos0=%u): %s (max diff %.2e)\n",
                phase, n_tok, pos0, pass ? "PASS" : "FAIL", diff);
        if (!pass) all_ok = 0;

        free(q); free(k); free(v); free(ref_out); free(gpu_out);
        ds4_gpu_tensor_free(g_q); ds4_gpu_tensor_free(g_k);
        ds4_gpu_tensor_free(g_v); ds4_gpu_tensor_free(g_out);
    }

    free(qn_w); free(kn_w); free(ref_kc); free(ref_vc);
    ds4_gpu_tensor_free(g_qnw); ds4_gpu_tensor_free(g_knw);
    ds4_gpu_tensor_free(g_kc); ds4_gpu_tensor_free(g_vc);
    return all_ok;
}
