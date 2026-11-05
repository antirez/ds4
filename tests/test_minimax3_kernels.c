/* GPU kernel tests for MiniMax-M3.1-Flash (metal/minimax3.metal) against a
 * double-precision reference at the release shapes.
 * Build: make test-minimax3-kernels */

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

#include "ds4.h"
#include "ds4_gpu.h"

bool ds4_log_is_tty(FILE *fp) {
    (void)fp;
    return false;
}

enum { H = 64, HKV = 4, D = 128, BLOCK = 128, TOPK = 16, NE = 128, NUSED = 4 };

static uint32_t g_rng = 0x2545f491u;

static float frand(void) {
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 17;
    g_rng ^= g_rng << 5;
    return ((float)(g_rng & 0xffffffu) / 8388608.0f) - 1.0f;
}

static void require_ok(int ok, const char *what) {
    if (!ok) {
        fprintf(stderr, "%s failed\n", what);
        exit(1);
    }
}

static void check_close(const char *what, const float *got, const double *ref, uint64_t n, double tol) {
    double worst = 0.0, scale = 1e-6;
    uint64_t worst_i = 0;
    for (uint64_t i = 0; i < n; i++) {
        if (!isfinite(got[i])) {
            fprintf(stderr, "%s: non-finite value at %llu\n", what, (unsigned long long)i);
            exit(1);
        }
        const double d = fabs((double)got[i] - ref[i]);
        if (d > worst) { worst = d; worst_i = i; }
        if (fabs(ref[i]) > scale) scale = fabs(ref[i]);
    }
    if (worst > tol * scale) {
        fprintf(stderr, "%s: max|d| %.3e (rel %.3e) at %llu: got %.6f ref %.6f\n",
                what, worst, worst / scale, (unsigned long long)worst_i, got[worst_i], ref[worst_i]);
        exit(1);
    }
    printf("  %-48s ok  max|d|=%.2e (scale %.2e)\n", what, worst, scale);
}

static ds4_gpu_tensor *upload(const float *data, uint64_t n) {
    ds4_gpu_tensor *t = ds4_gpu_tensor_alloc(n * sizeof(float));
    require_ok(t != NULL, "tensor alloc");
    if (data) require_ok(ds4_gpu_tensor_write(t, 0, data, n * sizeof(float)), "tensor write");
    else require_ok(ds4_gpu_tensor_fill_f32(t, 0.0f, n), "tensor fill");
    return t;
}

static void *download_bytes(const ds4_gpu_tensor *t, uint64_t bytes) {
    void *out = malloc(bytes);
    require_ok(ds4_gpu_tensor_read(t, 0, out, bytes), "tensor read");
    return out;
}

static float *rand_vec(uint64_t n, float scale) {
    float *v = malloc(n * sizeof(float));
    for (uint64_t i = 0; i < n; i++) v[i] = scale * frand();
    return v;
}

static float f16_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu, mant = h & 0x3ffu;
    union { uint32_t u; float f; } v;
    if (exp == 0) {
        if (mant == 0) { v.u = sign; return v.f; }
        while (!(mant & 0x400u)) { mant <<= 1; exp--; }
        exp++;
        mant &= 0x3ffu;
    } else if (exp == 31) {
        v.u = sign | 0x7f800000u | (mant << 13);
        return v.f;
    }
    v.u = sign | ((exp + 112u) << 23) | (mant << 13);
    return v.f;
}

static void rope_freq(float *freq) {
    for (uint32_t i = 0; i < 32; i++) freq[i] = (float)pow(1.0e7, -2.0 * i / 64.0);
}

/* gemma-folded RMSNorm then NeoX RoPE over the first 64 dims */
static void norm_rope_ref(double *out, const float *x, const float *g, uint32_t pos, const float *freq) {
    double ss = 0.0;
    for (uint32_t i = 0; i < D; i++) ss += (double)x[i] * x[i];
    const double r = 1.0 / sqrt(ss / D + 1e-6);
    for (uint32_t i = 0; i < D; i++) out[i] = x[i] * r * g[i];
    for (uint32_t i = 0; i < 32; i++) {
        const double th = (float)pos * freq[i], c = cos(th), s = sin(th);   /* float32 angle, as vLLM */
        const double a = out[i], b = out[i + 32];
        out[i] = a * c - b * s;
        out[i + 32] = a * s + b * c;
    }
}

/* prep writes normed/roped q, f16 k/v cache rows, indexer q and the indexer key cache */
static void test_attn_prep(uint64_t *map_off, uint8_t *map, uint32_t T, uint32_t pos0) {
    const uint32_t cap = pos0 + T, HI = HKV;
    float freq[32];
    rope_freq(freq);
    float *qp = rand_vec((uint64_t)T * H * D, 2.0f), *kp = rand_vec((uint64_t)T * HKV * D, 2.0f);
    float *vp = rand_vec((uint64_t)T * HKV * D, 2.0f), *iqp = rand_vec((uint64_t)T * HI * D, 2.0f);
    float *ikp = rand_vec((uint64_t)T * D, 2.0f);
    float *g = (float *)(map + *map_off);
    for (uint32_t i = 0; i < 4 * D; i++) g[i] = 1.0f + 0.3f * frand();
    const uint64_t goff = *map_off;
    *map_off += 4 * D * sizeof(float);
    ds4_gpu_tensor *tq = upload(qp, (uint64_t)T * H * D), *tk = upload(kp, (uint64_t)T * HKV * D);
    ds4_gpu_tensor *tv = upload(vp, (uint64_t)T * HKV * D), *tiq = upload(iqp, (uint64_t)T * HI * D);
    ds4_gpu_tensor *tik = upload(ikp, (uint64_t)T * D);
    ds4_gpu_tensor *q = upload(NULL, (uint64_t)T * H * D), *iq = upload(NULL, (uint64_t)T * HI * D);
    ds4_gpu_tensor *kc = upload(NULL, (uint64_t)cap * HKV * D / 2), *vc = upload(NULL, (uint64_t)cap * HKV * D / 2);
    ds4_gpu_tensor *ikc = upload(NULL, (uint64_t)cap * D);
    require_ok(ds4_gpu_mm3_attn_prep_tensor(q, kc, vc, iq, ikc, tq, tk, tv, tiq, tik, map, (uint64_t)1 << 28,
                                            goff, goff + D * 4, goff + 2 * D * 4, goff + 3 * D * 4,
                                            T, H, HKV, HI, pos0, cap, freq, 1e-6f), "mm3 attn prep");
    require_ok(ds4_gpu_synchronize(), "sync");
    double *rq = malloc((uint64_t)T * H * D * sizeof(double)), *rk = malloc((uint64_t)T * HKV * D * sizeof(double));
    double *rv = malloc((uint64_t)T * HKV * D * sizeof(double)), *riq = malloc((uint64_t)T * HI * D * sizeof(double));
    double *rik = malloc((uint64_t)T * D * sizeof(double));
    for (uint32_t t = 0; t < T; t++) {
        for (uint32_t h = 0; h < H; h++) norm_rope_ref(rq + ((uint64_t)t * H + h) * D, qp + ((uint64_t)t * H + h) * D, g, pos0 + t, freq);
        for (uint32_t h = 0; h < HKV; h++) {
            norm_rope_ref(rk + ((uint64_t)t * HKV + h) * D, kp + ((uint64_t)t * HKV + h) * D, g + D, pos0 + t, freq);
            for (uint32_t i = 0; i < D; i++) rv[((uint64_t)t * HKV + h) * D + i] = vp[((uint64_t)t * HKV + h) * D + i];
        }
        for (uint32_t h = 0; h < HI; h++) norm_rope_ref(riq + ((uint64_t)t * HI + h) * D, iqp + ((uint64_t)t * HI + h) * D, g + 2 * D, pos0 + t, freq);
        norm_rope_ref(rik + (uint64_t)t * D, ikp + (uint64_t)t * D, g + 3 * D, pos0 + t, freq);
    }
    float *gq = download_bytes(q, (uint64_t)T * H * D * 4), *giq = download_bytes(iq, (uint64_t)T * HI * D * 4);
    float *gik = download_bytes(ikc, (uint64_t)cap * D * 4);
    uint16_t *gk16 = download_bytes(kc, (uint64_t)cap * HKV * D * 2), *gv16 = download_bytes(vc, (uint64_t)cap * HKV * D * 2);
    float *gk = malloc((uint64_t)T * HKV * D * 4), *gv = malloc((uint64_t)T * HKV * D * 4);
    for (uint64_t i = 0; i < (uint64_t)T * HKV * D; i++) {
        gk[i] = f16_to_f32(gk16[(uint64_t)pos0 * HKV * D + i]);
        gv[i] = f16_to_f32(gv16[(uint64_t)pos0 * HKV * D + i]);
    }
    char name[96];
    snprintf(name, sizeof(name), "attn prep q T=%u pos0=%u", T, pos0);
    check_close(name, gq, rq, (uint64_t)T * H * D, 1e-4);
    check_close("attn prep k cache (f16)", gk, rk, (uint64_t)T * HKV * D, 2e-3);
    check_close("attn prep v cache (f16)", gv, rv, (uint64_t)T * HKV * D, 2e-3);
    check_close("attn prep indexer q", giq, riq, (uint64_t)T * HI * D, 1e-4);
    check_close("attn prep indexer k cache", gik + (uint64_t)pos0 * D, rik, (uint64_t)T * D, 1e-4);
    free(qp); free(kp); free(vp); free(iqp); free(ikp); free(rq); free(rk); free(rv); free(riq); free(rik);
    free(gq); free(giq); free(gik); free(gk16); free(gv16); free(gk); free(gv);
    ds4_gpu_tensor *all[] = { tq, tk, tv, tiq, tik, q, iq, kc, vc, ikc };
    for (size_t i = 0; i < sizeof(all) / sizeof(all[0]); i++) ds4_gpu_tensor_free(all[i]);
}

static uint16_t f32_to_f16_bits(float f) {
    __fp16 h = (__fp16)f;
    uint16_t u;
    memcpy(&u, &h, 2);
    return u;
}

/* selection + attention over a cache of n_keys, T queries ending at n_keys - 1 */
static void test_sparse_attention(uint32_t n_keys, uint32_t T) {
    const uint32_t pos0 = n_keys - T, HI = HKV;
    const uint32_t stride = (n_keys + BLOCK - 1) / BLOCK;
    float *q = rand_vec((uint64_t)T * H * D, 1.0f), *iq = rand_vec((uint64_t)T * HI * D, 1.0f);
    float *ik = rand_vec((uint64_t)n_keys * D, 1.0f);
    float *k = rand_vec((uint64_t)n_keys * HKV * D, 1.0f), *v = rand_vec((uint64_t)n_keys * HKV * D, 1.0f);
    uint16_t *k16 = malloc((uint64_t)n_keys * HKV * D * 2), *v16 = malloc((uint64_t)n_keys * HKV * D * 2);
    for (uint64_t i = 0; i < (uint64_t)n_keys * HKV * D; i++) {
        k16[i] = f32_to_f16_bits(k[i]); k[i] = f16_to_f32(k16[i]);
        v16[i] = f32_to_f16_bits(v[i]); v[i] = f16_to_f32(v16[i]);
    }
    ds4_gpu_tensor *tq = upload(q, (uint64_t)T * H * D), *tiq = upload(iq, (uint64_t)T * HI * D);
    ds4_gpu_tensor *tik = upload(ik, (uint64_t)n_keys * D);
    ds4_gpu_tensor *tk = ds4_gpu_tensor_alloc((uint64_t)n_keys * HKV * D * 2), *tv = ds4_gpu_tensor_alloc((uint64_t)n_keys * HKV * D * 2);
    require_ok(tk && tv && ds4_gpu_tensor_write(tk, 0, k16, (uint64_t)n_keys * HKV * D * 2) &&
               ds4_gpu_tensor_write(tv, 0, v16, (uint64_t)n_keys * HKV * D * 2), "kv upload");
    ds4_gpu_tensor *score = upload(NULL, (uint64_t)T * HI * stride), *sel = upload(NULL, (uint64_t)T * HI * TOPK);
    ds4_gpu_tensor *out = upload(NULL, (uint64_t)T * H * D);
    const float scale = 1.0f / sqrtf((float)D);
    require_ok(ds4_gpu_mm3_idx_select_tensor(sel, score, tiq, tik, T, pos0, HI, BLOCK, TOPK, stride), "mm3 select");
    /* decode-sized batches take the split path, as in the graph */
    ds4_gpu_tensor *part = T <= 4 ? upload(NULL, (uint64_t)T * H * TOPK * 132) : NULL;
    require_ok(ds4_gpu_mm3_attn_tensor(out, tq, tk, tv, sel, part, T, H, HKV, pos0, BLOCK, TOPK, scale), "mm3 attention");
    require_ok(ds4_gpu_synchronize(), "sync");
    int32_t *gsel = download_bytes(sel, (uint64_t)T * HI * TOPK * 4);
    float *gout = download_bytes(out, (uint64_t)T * H * D * 4);

    double *ref = calloc((uint64_t)T * H * D, sizeof(double));
    uint32_t sel_mismatch = 0;
    for (uint32_t t = 0; t < T; t++) {
        const uint32_t pos = pos0 + t, nb = pos / BLOCK + 1;
        for (uint32_t h = 0; h < HI; h++) {
            bool chosen[8192] = { false };
            if (nb <= TOPK) {
                for (uint32_t b = 0; b < nb; b++) chosen[b] = true;
            } else {
                double bs[8192];
                for (uint32_t b = 0; b < nb; b++) {
                    double m = -INFINITY;
                    for (uint32_t j = b * BLOCK; j < (b + 1) * BLOCK && j <= pos; j++) {
                        double s = 0.0;
                        for (uint32_t d = 0; d < D; d++) s += (double)iq[((uint64_t)t * HI + h) * D + d] * ik[(uint64_t)j * D + d];
                        if (s > m) m = s;
                    }
                    bs[b] = m;
                }
                chosen[nb - 1] = true;
                bs[nb - 1] = -INFINITY;
                for (uint32_t k2 = 1; k2 < TOPK; k2++) {
                    uint32_t best = 0;
                    for (uint32_t b = 1; b < nb; b++) if (bs[b] > bs[best]) best = b;
                    chosen[best] = true;
                    bs[best] = -INFINITY;
                }
            }
            /* the GPU list must name exactly the reference blocks */
            bool got[8192] = { false };
            for (uint32_t i = 0; i < TOPK; i++) {
                const int32_t b = gsel[((uint64_t)t * HI + h) * TOPK + i];
                if (b >= 0) got[b] = true;
            }
            for (uint32_t b = 0; b < nb; b++) sel_mismatch += got[b] != chosen[b];
            for (uint32_t g = 0; g < H / HKV; g++) {
                const uint32_t head = h * (H / HKV) + g;
                const float *qh = q + ((uint64_t)t * H + head) * D;
                double mx = -INFINITY, sum = 0.0;
                double *o = ref + ((uint64_t)t * H + head) * D;
                for (int pass = 0; pass < 2; pass++) {
                    for (uint32_t j = 0; j <= pos; j++) {
                        if (!got[j / BLOCK]) continue;   /* attend what the GPU chose */
                        double s = 0.0;
                        for (uint32_t d = 0; d < D; d++) s += (double)qh[d] * scale * k[((uint64_t)j * HKV + h) * D + d];
                        if (pass == 0) { if (s > mx) mx = s; continue; }
                        const double p = exp(s - mx);
                        sum += p;
                        for (uint32_t d = 0; d < D; d++) o[d] += p * v[((uint64_t)j * HKV + h) * D + d];
                    }
                }
                for (uint32_t d = 0; d < D; d++) o[d] /= sum;
            }
        }
    }
    if (sel_mismatch) {
        fprintf(stderr, "sparse selection n_keys=%u: %u block membership mismatches\n", n_keys, sel_mismatch);
        exit(1);
    }
    char name[96];
    snprintf(name, sizeof(name), "sparse attention n_keys=%u T=%u", n_keys, T);
    check_close(name, gout, ref, (uint64_t)T * H * D, 2e-4);
    free(q); free(iq); free(ik); free(k); free(v); free(k16); free(v16); free(gsel); free(gout); free(ref);
    ds4_gpu_tensor_free(part);
    ds4_gpu_tensor *all[] = { tq, tiq, tik, tk, tv, score, sel, out };
    for (size_t i = 0; i < sizeof(all) / sizeof(all[0]); i++) ds4_gpu_tensor_free(all[i]);
}

static void test_router(uint64_t *map_off, uint8_t *map, uint32_t T) {
    float *logits = rand_vec((uint64_t)T * NE, 4.0f);
    float *bias = (float *)(map + *map_off);
    for (uint32_t e = 0; e < NE; e++) bias[e] = 0.2f * frand();
    const uint64_t boff = *map_off;
    *map_off += NE * sizeof(float);
    ds4_gpu_tensor *tl = upload(logits, (uint64_t)T * NE), *sel = upload(NULL, (uint64_t)T * NUSED);
    ds4_gpu_tensor *wt = upload(NULL, (uint64_t)T * NUSED);
    require_ok(ds4_gpu_mm3_router_tensor(sel, wt, tl, map, (uint64_t)1 << 28, boff, T, NE, NUSED, 2.0f), "mm3 router");
    require_ok(ds4_gpu_synchronize(), "sync");
    int32_t *gs = download_bytes(sel, (uint64_t)T * NUSED * 4);
    float *gw = download_bytes(wt, (uint64_t)T * NUSED * 4);
    double *rw = malloc((uint64_t)T * NUSED * sizeof(double));
    for (uint32_t t = 0; t < T; t++) {
        double sc[NE], rank[NE], sum = 0.0;
        for (uint32_t e = 0; e < NE; e++) {
            sc[e] = 1.0 / (1.0 + exp(-(double)logits[(uint64_t)t * NE + e]));
            rank[e] = sc[e] + bias[e];
        }
        int32_t ch[NUSED];
        for (uint32_t k = 0; k < NUSED; k++) {
            uint32_t best = 0;
            for (uint32_t e = 1; e < NE; e++) if (rank[e] > rank[best]) best = e;
            ch[k] = (int32_t)best;
            rank[best] = -INFINITY;
            sum += sc[best];
        }
        for (uint32_t k = 0; k < NUSED; k++) {
            if (gs[(uint64_t)t * NUSED + k] != ch[k]) {
                fprintf(stderr, "router token %u slot %u: got expert %d, want %d\n", t, k, gs[(uint64_t)t * NUSED + k], ch[k]);
                exit(1);
            }
            rw[(uint64_t)t * NUSED + k] = sc[ch[k]] / sum * 2.0;
        }
    }
    check_close("router sigmoid+bias top-4 weights", gw, rw, (uint64_t)T * NUSED, 1e-5);
    free(logits); free(gs); free(gw); free(rw);
    ds4_gpu_tensor_free(tl); ds4_gpu_tensor_free(sel); ds4_gpu_tensor_free(wt);
}

static void test_reduce_glu(uint32_t T) {
    const uint32_t E = 6144, stride = NUSED + 1;
    float *x = rand_vec((uint64_t)T * E, 1.0f), *part = rand_vec((uint64_t)T * stride * E, 1.0f);
    float *w = rand_vec((uint64_t)T * NUSED, 1.0f);
    ds4_gpu_tensor *tx = upload(x, (uint64_t)T * E), *tp = upload(part, (uint64_t)T * stride * E);
    ds4_gpu_tensor *tw = upload(w, (uint64_t)T * NUSED);
    require_ok(ds4_gpu_mm3_moe_reduce_tensor(tx, tp, tw, NULL, T, NUSED, stride, E, 1), "mm3 reduce");
    float *g = rand_vec((uint64_t)T * E, 12.0f), *u = rand_vec((uint64_t)T * E, 12.0f);
    ds4_gpu_tensor *tg = upload(g, (uint64_t)T * E), *tu = upload(u, (uint64_t)T * E), *to = upload(NULL, (uint64_t)T * E);
    require_ok(ds4_gpu_mm3_glu_tensor(to, tg, tu, T * E), "mm3 glu");
    require_ok(ds4_gpu_synchronize(), "sync");
    double *rx = malloc((uint64_t)T * E * sizeof(double)), *ro = malloc((uint64_t)T * E * sizeof(double));
    for (uint32_t t = 0; t < T; t++) {
        for (uint32_t d = 0; d < E; d++) {
            double acc = 0.0;
            for (uint32_t s = 0; s < NUSED; s++) acc += (double)w[t * NUSED + s] * part[((uint64_t)t * stride + s) * E + d];
            acc += part[((uint64_t)t * stride + NUSED) * E + d];
            rx[(uint64_t)t * E + d] = x[(uint64_t)t * E + d] + acc;
            double gg = fmin(g[(uint64_t)t * E + d], 7.0), uu = fmax(-7.0, fmin(7.0, u[(uint64_t)t * E + d]));
            ro[(uint64_t)t * E + d] = gg / (1.0 + exp(-1.702 * gg)) * (uu + 1.0);
        }
    }
    float *gx = download_bytes(tx, (uint64_t)T * E * 4), *go = download_bytes(to, (uint64_t)T * E * 4);
    check_close("moe reduce (+shared slot, residual)", gx, rx, (uint64_t)T * E, 1e-5);
    check_close("SwiGLU-OAI (clamped)", go, ro, (uint64_t)T * E, 1e-5);
    free(x); free(part); free(w); free(g); free(u); free(rx); free(ro); free(gx); free(go);
    ds4_gpu_tensor *all[] = { tx, tp, tw, tg, tu, to };
    for (size_t i = 0; i < sizeof(all) / sizeof(all[0]); i++) ds4_gpu_tensor_free(all[i]);
}

static double swiglu_oai(double g, double u) {
    g = fmin(g, 7.0);
    u = fmax(-7.0, fmin(7.0, u));
    return g / (1.0 + exp(-1.702 * g)) * (u + 1.0);
}

static void deq_q3(const uint8_t *b, double *v) {
    const double d = f16_to_f32(*(const uint16_t *)(b + 108));
    uint32_t aux[4] = {0};
    memcpy(aux, b + 96, 12);
    const uint32_t km1 = 0x03030303u, km2 = 0x0f0f0f0fu, tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & km2) | (((tmp >> 4) & km1) << 4);
    aux[3] = ((aux[1] >> 4) & km2) | (((tmp >> 6) & km1) << 4);
    aux[0] = (aux[0] & km2) | (((tmp >> 0) & km1) << 4);
    aux[1] = (aux[1] & km2) | (((tmp >> 2) & km1) << 4);
    const int8_t *sc = (const int8_t *)aux;
    const uint8_t *hm = b, *q = b + 32;
    uint32_t m = 1, out = 0;
    for (uint32_t n = 0, is = 0; n < 256; n += 128) {
        for (uint32_t j = 0, shift = 0; j < 4; j++, shift += 2, m <<= 1) {
            const double d0 = d * (sc[is++] - 32);
            for (uint32_t l = 0; l < 16; l++) v[out++] = d0 * ((int)((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4));
            const double d1 = d * (sc[is++] - 32);
            for (uint32_t l = 0; l < 16; l++) v[out++] = d1 * ((int)((q[l + 16] >> shift) & 3) - ((hm[l + 16] & m) ? 0 : 4));
        }
        q += 32;
    }
}

static void deq_q6(const uint8_t *b, double *v) {
    const uint8_t *ql = b, *qh = b + 128;
    const int8_t *sc = (const int8_t *)(b + 192);
    const double d = f16_to_f32(*(const uint16_t *)(b + 208));
    uint32_t out = 0;
    for (uint32_t n = 0; n < 256; n += 128) {
        for (uint32_t l = 0; l < 32; l++) {
            const uint32_t is = l / 16;
            v[out + l] = d * sc[is] * ((int)((ql[l] & 15) | (((qh[l] >> 0) & 3) << 4)) - 32);
            v[out + l + 32] = d * sc[is + 2] * ((int)((ql[l + 32] & 15) | (((qh[l] >> 2) & 3) << 4)) - 32);
            v[out + l + 64] = d * sc[is + 4] * ((int)((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32);
            v[out + l + 96] = d * sc[is + 6] * ((int)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32);
        }
        out += 128; ql += 64; qh += 32; sc += 8;
    }
}

static double ue4m3(uint8_t x) {
    if (x == 0 || x == 0x7f) return 0.0;
    const int e = (x >> 3) & 15, m = x & 7;
    return (e == 0 ? ldexp((double)m, -9) : ldexp(1.0 + (double)m / 8.0, e - 7)) * 0.5;
}

static const double fp4[16] = { 0,.5,1,1.5,2,3,4,6, -0,-.5,-1,-1.5,-2,-3,-4,-6 };

static void deq_nvfp4(const uint8_t *b, double *v) {
    for (uint32_t s = 0; s < 4; s++) for (uint32_t j = 0; j < 8; j++) {
        const uint8_t q = b[4 + s * 8 + j];
        v[s * 16 + j] = ue4m3(b[s]) * fp4[q & 15];
        v[s * 16 + j + 8] = ue4m3(b[s]) * fp4[q >> 4];
    }
}

static void fill_quant(uint8_t *p, uint32_t type, uint32_t rows) {
    const uint32_t bytes = type == 11 ? 110 : type == 14 ? 210 : 36;
    for (uint32_t r = 0; r < rows; r++, p += bytes) {
        for (uint32_t i = 0; i < bytes; i++) p[i] = (uint8_t)(g_rng = g_rng * 1664525u + 1013904223u);
        if (type == 11) *(uint16_t *)(p + 108) = f32_to_f16_bits(0.0078125f);
        if (type == 14) {
            for (uint32_t i = 0; i < 16; i++) p[192 + i] = (uint8_t)((int)(i * 13u % 31u) - 15);
            *(uint16_t *)(p + 208) = f32_to_f16_bits(0.00390625f);
        }
        if (type == 40) for (uint32_t i = 0; i < 4; i++) p[i] = (uint8_t)(0x30 + ((r + i) & 15));
    }
}

static void test_scaled_experts(uint64_t *map_off, uint8_t *map) {
    enum { NX = 256, NR = 8, NEX = 2, T = 2, NS = 1 };
    float *x = rand_vec(T * NX, .25f), scales[3][NEX] = {{.75f, 1.25f}, {1.5f, .5f}, {.625f, 1.125f}};
    int32_t ids[T] = {0, 1};
    const uint64_t go = *map_off, expert_q3 = (uint64_t)110 * NR;
    fill_quant(map + go, 11, 2 * NEX * NR);
    const uint64_t uo = go + expert_q3 * NEX;
    const uint64_t gso = uo + expert_q3 * NEX, uso = gso + sizeof(scales[0]);
    memcpy(map + gso, scales[0], sizeof(scales[0])); memcpy(map + uso, scales[1], sizeof(scales[1]));
    *map_off = uso + sizeof(scales[1]);
    ds4_gpu_tensor *tx = upload(x, T * NX), *ti = ds4_gpu_tensor_alloc(sizeof(ids)), *mid = upload(NULL, T * NR);
    require_ok(ds4_gpu_tensor_write(ti, 0, ids, sizeof(ids)), "expert ids upload");
    require_ok(ds4_gpu_mm3_moe_mid_scaled_tensor(mid, tx, ti, map, (uint64_t)1 << 28, go, uo, gso, uso,
                                                  11, NEX, T, NS, NX, NR), "scaled q3 mid");
    require_ok(ds4_gpu_synchronize(), "scaled q3 sync");
    float *gm = download_bytes(mid, T * NR * 4); double rm[T * NR], w[256];
    for (uint32_t t = 0; t < T; t++) for (uint32_t r = 0; r < NR; r++) {
        deq_q3(map + go + (uint64_t)ids[t] * expert_q3 + r * 110, w); double g = 0, u = 0;
        for (uint32_t k = 0; k < NX; k++) g += w[k] * x[t * NX + k];
        deq_q3(map + uo + (uint64_t)ids[t] * expert_q3 + r * 110, w);
        for (uint32_t k = 0; k < NX; k++) u += w[k] * x[t * NX + k];
        rm[t * NR + r] = swiglu_oai(g * scales[0][ids[t]], u * scales[1][ids[t]]);
    }
    check_close("scaled Q3_K gate/up experts", gm, rm, T * NR, 2e-4);
    free(gm); ds4_gpu_tensor_free(mid);

    for (uint32_t pass = 0; pass < 2; pass++) {
        const uint32_t type = pass ? 40 : 14, rb = pass ? 144 : 210, block = pass ? 36 : 210;
        const uint64_t down = *map_off, expert = (uint64_t)rb * NR;
        fill_quant(map + down, type, NEX * NR * (pass ? 4 : 1));
        const uint64_t so = down + expert * NEX; memcpy(map + so, scales[2], sizeof(scales[2]));
        *map_off = so + sizeof(scales[2]);
        ds4_gpu_tensor *part = upload(NULL, T * NR);
        require_ok(ds4_gpu_mm3_moe_down_scaled_tensor(part, tx, ti, map, (uint64_t)1 << 28, down, so,
                                                       type, NEX, T, NS, NX, NR), "scaled down");
        require_ok(ds4_gpu_synchronize(), "scaled down sync");
        float *gp = download_bytes(part, T * NR * 4); double rp[T * NR];
        for (uint32_t t = 0; t < T; t++) for (uint32_t r = 0; r < NR; r++) {
            double sum = 0; const uint8_t *row = map + down + (uint64_t)ids[t] * expert + r * rb;
            for (uint32_t b = 0; b < (pass ? 4u : 1u); b++) {
                if (pass) deq_nvfp4(row + b * block, w); else deq_q6(row, w);
                for (uint32_t k = 0; k < (pass ? 64u : 256u); k++) sum += w[k] * x[t * NX + b * 64u + k];
            }
            rp[t * NR + r] = sum * scales[2][ids[t]];
        }
        check_close(pass ? "scaled NVFP4 down experts" : "scaled Q6_K down experts", gp, rp, T * NR, 2e-4);
        free(gp); ds4_gpu_tensor_free(part);
    }
    free(x); ds4_gpu_tensor_free(tx); ds4_gpu_tensor_free(ti);
}

int main(void) {
    const uint64_t map_size = (uint64_t)1 << 28;
    uint8_t *map = mmap(NULL, map_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    if (map == MAP_FAILED) { perror("mmap"); return 1; }
    uint64_t off = 0;
    require_ok(ds4_gpu_init(), "GPU initialization");
    require_ok(ds4_gpu_set_model_map(map, map_size), "model map registration");
    printf("MiniMax-M3 kernels:\n");
    test_attn_prep(&off, map, 1, 0);
    test_attn_prep(&off, map, 5, 1000000);
    test_sparse_attention(7, 7);          /* one partial block, prefill */
    test_sparse_attention(2048, 1);       /* exactly top_k blocks: dense */
    test_sparse_attention(2049, 1);       /* first sparse decode */
    test_sparse_attention(5000, 3);       /* sparse, partial own block */
    test_router(&off, map, 9);
    test_reduce_glu(3);
    test_scaled_experts(&off, map);
    printf("all minimax3 kernel tests passed\n");
    return 0;
}
