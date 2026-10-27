/* The fused Spark HC pre boundary evaluates the same graph as RMSNorm + F16
 * mixer + split/sum/norm: check it against an FP64 evaluation and against
 * the separate chain, row independence, overflow and nonfinite behaviour,
 * aliasing, tails, rejection and the disable switch.  Its split, weighted sum
 * and norm must equal, bit for bit, the unchanged split+norm entry fed the
 * fused mixer rows. */
#include "ds4_gpu.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); exit(1); } } while (0)
static uint32_t rng = 783;
static float random_value() {
    rng = rng * 1664525u + 1013904223u;
    return ((int)(rng >> 16) - 32768) / 8192.0f;
}

static const uint32_t EMBD = 4096, K = 4 * EMBD, MIX = 24, ITERS = 20, MAX_ROWS = 8;
static const float EPS = 1e-6f, SENT = -9182.0f;
struct Set { uint64_t w, scale, base, norm; };
static std::vector<uint8_t> model;
static std::vector<Set> sets;

static uint64_t reserve(uint64_t bytes) {
    const uint64_t off = (model.size() + 255u) & ~255ull;
    model.resize(off + bytes);
    return off;
}
static void build_model(uint32_t n_sets) {
    for (uint32_t s = 0; s < n_sets; s++) {
        Set st = {reserve((uint64_t)MIX * K * 2), reserve(12), reserve(MIX * 4), reserve(EMBD * 4)};
        sets.push_back(st);
    }
    for (const Set &st : sets) {
        __half *w = (__half *)&model[st.w];
        for (uint64_t i = 0; i < (uint64_t)MIX * K; i++) w[i] = __float2half(random_value() * 0.01f);
        float *sc = (float *)&model[st.scale];
        sc[0] = 0.25f; sc[1] = 0.75f; sc[2] = 1.5f;
        for (uint32_t j = 0; j < MIX; j++) ((float *)&model[st.base])[j] = random_value() * 0.25f;
        for (uint32_t d = 0; d < EMBD; d++) ((float *)&model[st.norm])[d] = 1.0f + random_value() * 0.05f;
    }
}

/* FP64 evaluation of one row: mix, split, weighted sum, output norm. */
static void reference(const float *x, const Set &st, double *out4[4]) {
    const __half *w = (const __half *)&model[st.w];
    const float *sc = (const float *)&model[st.scale], *b = (const float *)&model[st.base];
    const float *nw = (const float *)&model[st.norm];
    double ss = 0, *mix = out4[0], *sp = out4[1];
    for (uint32_t k = 0; k < K; k++) ss += (double)x[k] * x[k];
    const double inv = 1.0 / std::sqrt(ss / K + EPS);
    for (uint32_t j = 0; j < MIX; j++) {
        double acc = 0;
        for (uint32_t k = 0; k < K; k++) acc += (double)__half2float(w[(uint64_t)j * K + k]) * x[k];
        mix[j] = acc * inv;
    }
    for (int i = 0; i < 4; i++) sp[i] = 1.0 / (1.0 + std::exp(-(mix[i] * sc[0] + b[i]))) + EPS;
    for (int i = 0; i < 4; i++) sp[4 + i] = 2.0 / (1.0 + std::exp(-(mix[4 + i] * sc[1] + b[4 + i])));
    double c[16];
    for (int r = 0; r < 4; r++) {
        double m = -INFINITY, s = 0;
        for (int q = 0; q < 4; q++) m = std::max(m, c[r * 4 + q] = mix[8 + r * 4 + q] * sc[2] + b[8 + r * 4 + q]);
        for (int q = 0; q < 4; q++) s += (c[r * 4 + q] = std::exp(c[r * 4 + q] - m));
        for (int q = 0; q < 4; q++) c[r * 4 + q] = c[r * 4 + q] / s + EPS;
    }
    for (uint32_t it = 0; it < ITERS; it++) {
        if (it) {
            for (int r = 0; r < 4; r++) {
                double s = EPS;
                for (int q = 0; q < 4; q++) s += c[r * 4 + q];
                for (int q = 0; q < 4; q++) c[r * 4 + q] /= s;
            }
        }
        for (int q = 0; q < 4; q++) {
            double s = EPS;
            for (int r = 0; r < 4; r++) s += c[r * 4 + q];
            for (int r = 0; r < 4; r++) c[r * 4 + q] /= s;
        }
    }
    for (int i = 0; i < 16; i++) sp[8 + i] = c[i];
    double ns = 0;
    for (uint32_t d = 0; d < EMBD; d++) {
        double acc = 0;
        for (uint32_t h = 0; h < 4; h++) acc += sp[h] * x[h * EMBD + d];
        out4[2][d] = acc;
        ns += acc * acc;
    }
    const double nscale = 1.0 / std::sqrt(ns / EMBD + EPS);
    for (uint32_t d = 0; d < EMBD; d++) out4[3][d] = out4[2][d] * nscale * nw[d];
}

struct Out {
    uint32_t rows;
    ds4_gpu_tensor *t[4], *v[4], *scr, *vscr;   /* mix, split, out, norm */
    Out(uint32_t n) : rows(n) {
        const uint64_t f[4] = {n * MIX, n * MIX, (uint64_t)n * EMBD, (uint64_t)n * EMBD};
        for (int i = 0; i < 4; i++) {
            CHECK((t[i] = ds4_gpu_tensor_alloc(f[i] * 4 + 32)) && (v[i] = ds4_gpu_tensor_view(t[i], 0, f[i] * 4)));
        }
        CHECK((scr = ds4_gpu_tensor_alloc((uint64_t)n * K * 4 + 32)) &&
              (vscr = ds4_gpu_tensor_view(scr, 0, (uint64_t)n * K * 4)));
    }
    ~Out() { for (int i = 0; i < 4; i++) { ds4_gpu_tensor_free(v[i]); ds4_gpu_tensor_free(t[i]); }
             ds4_gpu_tensor_free(vscr); ds4_gpu_tensor_free(scr); }
    void sentinel() {
        for (ds4_gpu_tensor *x : {t[0], t[1], t[2], t[3], scr})
            CHECK(ds4_gpu_tensor_fill_f32(x, SENT, ds4_gpu_tensor_bytes(x) / 4));
    }
    std::vector<float> read(int i) const {
        std::vector<float> h(ds4_gpu_tensor_bytes(t[i]) / 4);
        CHECK(ds4_gpu_tensor_read(t[i], 0, h.data(), h.size() * 4));
        return h;
    }
    bool untouched() const {
        for (int i = 0; i < 4; i++) for (float x : read(i)) if (memcmp(&x, &SENT, 4)) return false;
        std::vector<float> h(ds4_gpu_tensor_bytes(scr) / 4);
        CHECK(ds4_gpu_tensor_read(scr, 0, h.data(), h.size() * 4));
        for (float x : h) if (memcmp(&x, &SENT, 4)) return false;
        return true;
    }
};

static int fused(Out &o, ds4_gpu_tensor *x, const Set &st, uint32_t n) {
    return ds4_gpu_hc_pre_f32_mix_split_norm_tensor(o.v[0], o.v[2], o.v[3], o.v[1], o.vscr, x,
        model.data(), model.size(), st.w, st.scale, st.base, st.norm, EMBD, 4, n, ITERS, EPS, EPS, EPS);
}
static void chain(Out &o, ds4_gpu_tensor *x, const Set &st) {
    if (o.rows == 1) {
        CHECK(ds4_gpu_rms_norm_plain_tensor(o.vscr, x, K, EPS));
        CHECK(ds4_gpu_matmul_f16_tensor(o.v[0], model.data(), model.size(), st.w, K, MIX, o.vscr, 1));
    } else {
        CHECK(ds4_gpu_matmul_f16_rms_fold_tensor(o.v[0], model.data(), model.size(), st.w, K, MIX, x, o.rows, EPS));
    }
    CHECK(ds4_gpu_hc_split_weighted_sum_norm_tensor(o.v[2], o.v[3], o.v[1], o.v[0], x, model.data(), model.size(),
                                                     st.scale, st.base, st.norm, EMBD, 4, ITERS, EPS, EPS));
}

static void fill_rows(std::vector<float> &h, uint32_t n, unsigned pattern) {
    h.resize((size_t)n * K);
    for (size_t i = 0; i < h.size(); i++) {
        const float v = random_value();
        h[i] = pattern == 0 ? v : pattern == 1 ? v * (1u << (i / EMBD % 4u * 2u)) : ldexpf(v, (int)(i % 41u) - 20);
    }
    if (pattern == 1) {
        for (uint32_t o = 0; o < 16u * n; o++) {
            random_value();
            h[(rng >> 3) % h.size()] *= 100.0f;
        }
    }
}

static void check_graphs(uint32_t n) {
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc((uint64_t)n * K * 4);
    CHECK(x);
    Out eager(n), captured(n);
    ds4_decode_graph_key key = {};
    key.il = 0; key.island = 1; key.variant = n; key.cur_hc = x;
    unsigned captures = 0, replays = 0;
    std::vector<float> h;
    for (unsigned epoch = 0; epoch < 2; epoch++) {
        ds4_gpu_decode_graphs_invalidate();
        for (unsigned round = 0; round < 6; round++) {
            fill_rows(h, n, round % 3);
            CHECK(ds4_gpu_tensor_write(x, 0, h.data(), h.size() * 4));
            eager.sentinel(); captured.sentinel();
            CHECK(fused(eager, x, sets[0], n) == 1);
            const int graph = ds4_gpu_decode_graph_begin(&key);
            CHECK(graph == (round == 0 ? -1 : round == 1 ? 0 : 1));
            if (graph == 1) {
                replays++;
            } else {
                CHECK(fused(captured, x, sets[0], n) == 1);
                if (graph == 0) {
                    CHECK(ds4_gpu_decode_graph_end(&key) == 0);
                    captures++;
                }
            }
            for (int i = 0; i < 4; i++) {
                const std::vector<float> expected = eager.read(i), actual = captured.read(i);
                CHECK(!memcmp(expected.data(), actual.data(), expected.size() * 4));
            }
        }
    }
    CHECK(captures == 2 && replays == 8);
    CHECK(ds4_gpu_synchronize());
    ds4_gpu_decode_graphs_invalidate();
    ds4_gpu_tensor_free(x);
}

int main(int argc, char **argv) {
    CHECK(argc == 1);
    (void)argv;
    build_model(2);
    CHECK(setenv("DS4_CUDA_COPY_MODEL", "1", 0) == 0);
    CHECK(setenv("DS4_CUDA_DECODE_GRAPHS", "1", 1) == 0);
    CHECK(ds4_gpu_init() && ds4_gpu_set_model_map(model.data(), model.size()));
    if (!ds4_gpu_device_is_spark()) {
        {
            ds4_gpu_tensor *x = ds4_gpu_tensor_alloc((uint64_t)K * 4);
            Out o(1);
            o.sentinel();
            CHECK(x && fused(o, x, sets[0], 1) == 0 && o.untouched());
            ds4_gpu_tensor_free(x);
        }
        ds4_gpu_cleanup();
        printf("HC pre fused: not a Spark device, path off and outputs untouched: SKIP\n");
        return 0;
    }
    /* Accuracy: aggregate relative L2 error against FP64 for the fused path
     * and the separate chain; per-element errors are not compared because
     * cancellation makes their relative size meaningless. */
    double e2[2][4] = {{0}}, r2[4] = {0};
    std::vector<double> ref[4] = {std::vector<double>(MIX), std::vector<double>(MIX),
                                  std::vector<double>(EMBD), std::vector<double>(EMBD)};
    double *rp[4] = {ref[0].data(), ref[1].data(), ref[2].data(), ref[3].data()};
    unsigned cases = 0;
    for (uint32_t n = 1; n <= MAX_ROWS; n++) {
        ds4_gpu_tensor *x = ds4_gpu_tensor_alloc((uint64_t)n * K * 4);
        CHECK(x);
        Out a(n), b(n), c(n), one(1), anchor(n);
        /* The split+norm entry runs its one-block-per-row kernel for n = 1 and
         * Spark n <= 6; larger batches use separate kernels with their own
         * reduction, so the bitwise anchor covers n <= 6. */
        auto check_anchor = [&](const Set &st, ds4_gpu_tensor *x) {
            if (n > 6) return;
            anchor.sentinel();
            CHECK(ds4_gpu_hc_split_weighted_sum_norm_tensor(anchor.v[2], anchor.v[3], anchor.v[1], a.v[0], x,
                  model.data(), model.size(), st.scale, st.base, st.norm, EMBD, 4, ITERS, EPS, EPS));
            for (int i = 1; i < 4; i++) {
                const std::vector<float> fused_out = a.read(i), entry_out = anchor.read(i);
                CHECK(!memcmp(fused_out.data(), entry_out.data(), fused_out.size() * 4));
            }
        };
        std::vector<float> h;
        for (unsigned pattern = 0; pattern < 3; pattern++) {
            const Set &st = sets[(n + pattern) % sets.size()];
            fill_rows(h, n, pattern);
            CHECK(ds4_gpu_tensor_write(x, 0, h.data(), h.size() * 4));
            a.sentinel(); b.sentinel(); c.sentinel();
            CHECK(fused(a, x, st, n) == 1);
            chain(b, x, st);
            CHECK(fused(c, x, st, n) == 1);
            const std::vector<float> fa[4] = {a.read(0), a.read(1), a.read(2), a.read(3)};
            const std::vector<float> fb[4] = {b.read(0), b.read(1), b.read(2), b.read(3)};
            check_anchor(st, x);
            for (int i = 0; i < 4; i++) {
                const size_t live = (size_t)n * (i < 2 ? MIX : EMBD);
                CHECK(!memcmp(fa[i].data(), c.read(i).data(), fa[i].size() * 4));                    /* deterministic */
                for (size_t j = live; j < fa[i].size(); j++) CHECK(!memcmp(&fa[i][j], &SENT, 4));      /* tails */
            }
            {   /* The partials use n * 26 * (K / 256) floats of the scratch. */
                std::vector<float> s(ds4_gpu_tensor_bytes(a.scr) / 4);
                CHECK(ds4_gpu_tensor_read(a.scr, 0, s.data(), s.size() * 4));
                for (size_t j = (size_t)n * 26 * (K / 256); j < s.size(); j++) CHECK(!memcmp(&s[j], &SENT, 4));
            }
            for (uint32_t r = 0; r < n; r++) {
                reference(&h[(size_t)r * K], st, rp);
                for (int i = 0; i < 4; i++) {
                    const uint32_t w = i < 2 ? MIX : EMBD;
                    for (uint32_t j = 0; j < w; j++) {
                        const double d0 = fa[i][(size_t)r * w + j] - ref[i][j], d1 = fb[i][(size_t)r * w + j] - ref[i][j];
                        e2[0][i] += d0 * d0; e2[1][i] += d1 * d1; r2[i] += ref[i][j] * ref[i][j];
                    }
                }
                /* Only the final norm uses a different reduction tree for n=1. */
                ds4_gpu_tensor *xr = ds4_gpu_tensor_view(x, (uint64_t)r * K * 4, (uint64_t)K * 4);
                CHECK(xr);
                one.sentinel();
                CHECK(fused(one, xr, st, 1) == 1);
                for (int i = 0; i < 4; i++) {
                    const uint32_t w = i < 2 ? MIX : EMBD;
                    const std::vector<float> single = one.read(i);
                    const float *batch = &fa[i][(size_t)r * w];
                    if (i < 3) {
                        CHECK(!memcmp(single.data(), batch, (size_t)w * 4));
                    } else {
                        for (uint32_t j = 0; j < w; j++)
                            CHECK(std::fabs(single[j] - batch[j]) <= 5e-7f * std::max(1.0f, std::fabs(single[j])));
                    }
                }
                ds4_gpu_tensor_free(xr);
            }
            /* In place: norm_out == out. */
            c.sentinel();
            CHECK(ds4_gpu_hc_pre_f32_mix_split_norm_tensor(c.v[0], c.v[2], c.v[2], c.v[1], c.vscr, x,
                  model.data(), model.size(), st.w, st.scale, st.base, st.norm, EMBD, 4, n, ITERS, EPS, EPS, EPS) == 1);
            CHECK(!memcmp(c.read(2).data(), fa[3].data(), fa[3].size() * 4));
            ++cases;
        }
        /* Overflowing F32 sum of squares with finite input normalizes to zero in
         * both paths; Inf and NaN input stay nonfinite in both. */
        for (unsigned kind = 0; kind < 3; kind++) {
            for (size_t i = 0; i < h.size(); i++) h[i] = kind == 0 ? random_value() * 3e17f : random_value();
            if (kind == 1) h[123] = INFINITY;
            if (kind == 2) h[456] = NAN;
            CHECK(ds4_gpu_tensor_write(x, 0, h.data(), h.size() * 4));
            a.sentinel(); b.sentinel();
            CHECK(fused(a, x, sets[0], n) == 1);
            chain(b, x, sets[0]);
            check_anchor(sets[0], x);
            for (int i = 0; i < 4; i++) {
                const std::vector<float> fa = a.read(i), fb = b.read(i);
                for (size_t j = 0; j < (size_t)n * (i < 2 ? MIX : EMBD); j++) {
                    CHECK(std::isnan(fa[j]) == std::isnan(fb[j]) && std::isfinite(fa[j]) == std::isfinite(fb[j]));
                    if (kind == 0) CHECK(fa[j] == fb[j]);
                }
            }
        }
        ds4_gpu_tensor_free(x);
    }
    const char *name[4] = {"mix", "split", "weighted", "norm"};
    for (int i = 0; i < 4; i++) {
        const double fused_rel = std::sqrt(e2[0][i] / r2[i]), chain_rel = std::sqrt(e2[1][i] / r2[i]);
        printf("%-8s relL2 vs FP64: fused %.3e  separate %.3e\n", name[i], fused_rel, chain_rel);
        CHECK(fused_rel < 1e-5);
        CHECK(i >= 2 ? fused_rel <= 1.05 * chain_rel : fused_rel <= chain_rel);
    }
    /* Unsupported calls leave every output and the scratch untouched. */
    {
        const uint32_t n = 4;
        ds4_gpu_tensor *x = ds4_gpu_tensor_alloc((uint64_t)n * K * 4 + 64);
        CHECK(x && ds4_gpu_tensor_fill_f32(x, 1.0f, ((uint64_t)n * K * 4 + 64) / 4));
        ds4_gpu_tensor *vx = ds4_gpu_tensor_view(x, 0, (uint64_t)n * K * 4);
        ds4_gpu_tensor *mis = ds4_gpu_tensor_view(x, 4, (uint64_t)n * K * 4);
        Out o(n);
        ds4_gpu_tensor *short_scr = ds4_gpu_tensor_view(o.scr, 0, (uint64_t)n * 26 * 64 * 4 - 4);
        ds4_gpu_tensor *shift = ds4_gpu_tensor_view(o.t[2], 4, (uint64_t)n * EMBD * 4);
        CHECK(vx && mis && short_scr && shift);
        const uint64_t size = model.size();
        auto call = [&](ds4_gpu_tensor *mx, ds4_gpu_tensor *ou, ds4_gpu_tensor *nm, ds4_gpu_tensor *sp,
                        ds4_gpu_tensor *sc, ds4_gpu_tensor *xx, Set st, uint32_t rows, uint32_t embd) {
            o.sentinel();
            const int rc = ds4_gpu_hc_pre_f32_mix_split_norm_tensor(mx, ou, nm, sp, sc, xx, model.data(), size,
                                                                    st.w, st.scale, st.base, st.norm, embd, 4, rows,
                                                                    ITERS, EPS, EPS, EPS);
            CHECK(rc == 0 && o.untouched());
        };
        const Set ok = sets[0];
        Set bad;
        call(o.v[0], o.v[2], o.v[3], o.v[1], o.vscr, vx, ok, 0, EMBD);            /* zero rows */
        call(o.v[0], o.v[2], o.v[3], o.v[1], o.vscr, vx, ok, 9, EMBD);            /* too many rows */
        call(o.v[0], o.v[2], o.v[3], o.v[1], o.vscr, vx, ok, n, 100);             /* K % 256 */
        call(o.v[0], o.v[2], o.v[3], o.v[1], o.vscr, mis, ok, n, EMBD);           /* state alignment */
        bad = ok; bad.w += 2;
        call(o.v[0], o.v[2], o.v[3], o.v[1], o.vscr, vx, bad, n, EMBD);           /* weight alignment */
        bad = ok; bad.w = size - 256;
        call(o.v[0], o.v[2], o.v[3], o.v[1], o.vscr, vx, bad, n, EMBD);           /* weight past the map */
        bad = ok; bad.w = size + 256;
        call(o.v[0], o.v[2], o.v[3], o.v[1], o.vscr, vx, bad, n, EMBD);           /* weight offset outside */
        bad = ok; bad.scale = size - 8;
        call(o.v[0], o.v[2], o.v[3], o.v[1], o.vscr, vx, bad, n, EMBD);           /* scale range */
        bad = ok; bad.base = size - 64;
        call(o.v[0], o.v[2], o.v[3], o.v[1], o.vscr, vx, bad, n, EMBD);           /* base range */
        bad = ok; bad.norm = size - 4;
        call(o.v[0], o.v[2], o.v[3], o.v[1], o.vscr, vx, bad, n, EMBD);           /* norm range */
        bad = ok; bad.scale += 1;
        call(o.v[0], o.v[2], o.v[3], o.v[1], o.vscr, vx, bad, n, EMBD);           /* scale alignment */
        bad = ok; bad.base += 1;
        call(o.v[0], o.v[2], o.v[3], o.v[1], o.vscr, vx, bad, n, EMBD);           /* base alignment */
        bad = ok; bad.norm += 1;
        call(o.v[0], o.v[2], o.v[3], o.v[1], o.vscr, vx, bad, n, EMBD);           /* norm alignment */
        call(o.v[0], o.v[2], o.v[3], o.v[1], short_scr, vx, ok, n, EMBD);         /* scratch size */
        call(o.v[0], o.v[2], shift, o.v[1], o.vscr, vx, ok, n, EMBD);             /* out/norm overlap */
        call(o.v[0], o.v[2], o.v[3], o.v[0], o.vscr, vx, ok, n, EMBD);            /* mix/split alias */
        call(o.v[0], o.v[2], o.v[3], o.v[1], o.v[2], vx, ok, n, EMBD);            /* scratch/out alias */
        for (ds4_gpu_tensor *t : {short_scr, shift, vx, mis, x}) ds4_gpu_tensor_free(t);
    }
    for (uint32_t n : {1u, 6u, 8u}) check_graphs(n);
    ds4_gpu_cleanup();
    /* The disable switch selects the separate chain. */
    CHECK(setenv("DS4_CUDA_DISABLE_HC_PRE_FUSED", "1", 1) == 0);
    CHECK(ds4_gpu_init() && ds4_gpu_set_model_map(model.data(), model.size()));
    {
        ds4_gpu_tensor *x = ds4_gpu_tensor_alloc((uint64_t)2 * K * 4);
        CHECK(x && ds4_gpu_tensor_fill_f32(x, 1.0f, (uint64_t)2 * K));
        Out o(2);
        o.sentinel();
        CHECK(fused(o, x, sets[0], 2) == 0 && o.untouched());
        ds4_gpu_tensor_free(x);
    }
    ds4_gpu_cleanup();
    printf("HC pre fused: %u cases, split+norm anchor, row independence, overflow, aliasing, tails, graphs and switch PASS\n", cases);
    return 0;
}
