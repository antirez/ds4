#include "ds4_gpu.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); exit(1); } } while (0)

/* One-row indexed attention on Spark runs in three passes by default and
 * must equal the per-head kernel, which DS4_CUDA_NO_INDEXED_HEADS8 selects,
 * bit for bit.  Shapes: 1..64 heads, and 65 and 128 (per-head kernel either
 * way); positions that wrap the raw ring; windows 128 (the model's), 3, 1
 * and 0 (no limit, at most 200 rows kept); compression ratios 4, 1 and 128;
 * top-k 512, 5 and 1; top-k entries that are causal, beyond the visible
 * rows, masked as -1 or INT32_MAX, or all masked (raw rows only). Every
 * output must be finite; sampled coordinates also match a double-precision
 * reference. Invalid arguments are refused, the 16 floats after the output
 * stay untouched, decode graphs replay the default, opted-out, 65- and
 * 128-head paths on new queries, and two- and six-row calls stay within the
 * reference. */

static uint32_t state = 4711;
static uint32_t next_u32() {
    state = state * 1664525u + 1013904223u;
    return state;
}
static float random_value() { return ((int)(next_u32() >> 16) - 32768) / 8192.0f; }

static const uint32_t D = 512, RAW_CAP = 512, MAXH = 128, MAXN = 6, TOP_K = 512;
static const uint32_t COMP_CAP = 16384u + 8u;
static const float SENT = -9182.0f;
static std::vector<float> sinks, hq, hraw, hcomp;
static std::vector<int32_t> hids;
static ds4_gpu_tensor *q, *raw, *comp, *ids, *out;

struct Call { uint32_t n, n_head, pos0, window, top_k, ratio; };

/* Rows the caller's ring still holds: the engine keeps n + window (128 in
 * the model); window 0 (no limit) keeps 200 here, below the kernels'
 * 256-row bound, which no model shape reaches. */
static uint32_t kept_raw(const Call &c) {
    const uint32_t want = c.window ? c.n + c.window : 200u;
    return std::min<uint32_t>(std::min<uint32_t>(want, c.pos0 + c.n), RAW_CAP);
}
static uint32_t n_comp(const Call &c) { return (c.pos0 + c.n) / c.ratio; }
static uint32_t visible(const Call &c, uint32_t t) { return std::min((c.pos0 + t + 1) / c.ratio, n_comp(c)); }
static int call(const Call &c, ds4_gpu_tensor *heads) {
    const uint32_t n_raw = kept_raw(c);
    return ds4_gpu_attention_indexed_mixed_batch_heads_tensor(heads, sinks.data(), sinks.size() * 4, 0, q, raw, comp,
        0, ids, c.n, c.pos0, n_raw, RAW_CAP, (c.pos0 + c.n - n_raw) % RAW_CAP, n_comp(c), c.top_k,
        c.window, c.ratio, c.n_head, D);
}
static std::vector<float> read_out(const Call &c) {
    const uint64_t live = (uint64_t)c.n * c.n_head * D;
    std::vector<float> r(live + 16);
    CHECK(ds4_gpu_tensor_read(out, 0, r.data(), r.size() * 4));
    for (uint64_t i = 0; i < live; i++) CHECK(std::isfinite(r[i]));
    for (uint64_t i = live; i < r.size(); i++) CHECK(r[i] == SENT);
    r.resize(live);
    return r;
}
static void per_head_kernel(bool on) {
    if (on) CHECK(setenv("DS4_CUDA_NO_INDEXED_HEADS8", "1", 1) == 0);
    else CHECK(unsetenv("DS4_CUDA_NO_INDEXED_HEADS8") == 0);
}
static std::vector<float> run(const Call &c, bool per_head) {
    per_head_kernel(per_head);
    CHECK(ds4_gpu_tensor_fill_f32(out, SENT, (uint64_t)MAXN * MAXH * D + 16));
    ds4_gpu_tensor *heads = ds4_gpu_tensor_view(out, 0, (uint64_t)c.n * c.n_head * D * 4);
    CHECK(heads && call(c, heads));
    ds4_gpu_tensor_free(heads);
    CHECK(ds4_gpu_synchronize());
    per_head_kernel(false);
    return read_out(c);
}

/* Double-precision attention from the definition: the raw rows from
 * max(first kept row, qpos + 1 - window) (window 0: no limit) to qpos,
 * every visible top-k entry (repeats count again), scale 1/sqrt(512), and
 * the sink as one more logit with a zero value. */
static void check_reference(const Call &c, const std::vector<float> &got) {
    const uint32_t first_raw = c.pos0 + c.n - kept_raw(c);
    for (uint32_t t = 0; t < c.n; t++) {
        const uint32_t qpos = c.pos0 + t;
        std::vector<const float *> rows;
        uint32_t lo = first_raw;
        if (c.window && qpos + 1 > c.window) lo = std::max(lo, qpos + 1 - c.window);
        for (uint32_t p = lo; p <= qpos; p++) rows.push_back(&hraw[(size_t)(p % RAW_CAP) * D]);
        CHECK(rows.size() <= 256);
        for (uint32_t j = 0; j < c.top_k; j++) {
            const int32_t id = hids[(size_t)t * c.top_k + j];
            if (id >= 0 && (uint32_t)id < visible(c, t)) rows.push_back(&hcomp[(size_t)id * D]);
        }
        for (uint32_t h = 0; h < c.n_head; h += 7) {
            const float *qh = &hq[((size_t)t * c.n_head + h) * D];
            std::vector<double> logit(rows.size());
            double mx = sinks[h];
            for (size_t j = 0; j < rows.size(); j++) {
                double dot = 0;
                for (uint32_t d = 0; d < D; d++) dot += (double)qh[d] * rows[j][d];
                logit[j] = dot / sqrt((double)D);
                mx = std::max(mx, logit[j]);
            }
            double den = exp((double)sinks[h] - mx);
            for (size_t j = 0; j < rows.size(); j++) den += exp(logit[j] - mx);
            for (uint32_t d = 0; d < D; d += 37) {
                double num = 0;
                for (size_t j = 0; j < rows.size(); j++) num += exp(logit[j] - mx) * rows[j][d];
                const double ref = num / den, v = got[((size_t)t * c.n_head + h) * D + d];
                CHECK(std::isfinite(v) && fabs(v - ref) < 3e-5 * (1 + fabs(ref)));
            }
        }
    }
}

/* 0 causal, 1 any compressed row (later ones masked by position), 2 V = 40
 * visible and -1 elsewhere, 3 one visible and INT32_MAX elsewhere, 4 all -1. */
static void fill_ids(const Call &c, unsigned mode) {
    hids.assign((size_t)c.n * c.top_k, 0);
    for (uint32_t t = 0; t < c.n; t++) {
        for (uint32_t j = 0; j < c.top_k; j++) {
            int32_t &id = hids[(size_t)t * c.top_k + j];
            const uint32_t range = mode == 1 ? n_comp(c) : visible(c, t);
            id = range ? (int32_t)(next_u32() % range) : -1;
            if ((mode == 2 && j >= 40) || mode == 4) id = -1;
            if (mode == 3 && j >= 1) id = INT32_MAX;
        }
        for (uint32_t j = c.top_k - 1; mode >= 2 && j > 0; j--)
            std::swap(hids[(size_t)t * c.top_k + j], hids[(size_t)t * c.top_k + next_u32() % (j + 1)]);
    }
    CHECK(ds4_gpu_tensor_write(ids, 0, hids.data(), hids.size() * 4));
}
static void fill(ds4_gpu_tensor *t, std::vector<float> &h, size_t n) {
    h.resize(n);
    for (float &v : h) v = random_value() / 4;
    CHECK(ds4_gpu_tensor_write(t, 0, h.data(), n * 4));
}
static bool same(const std::vector<float> &a, const std::vector<float> &b) {
    return a.size() == b.size() && !memcmp(a.data(), b.data(), a.size() * 4);
}

static unsigned compare(const Call &c, unsigned mode) {
    if (n_comp(c) == 0 || (c.top_k < 40 && mode == 2)) return 0;
    fill_ids(c, mode);
    const std::vector<float> fast = run(c, false);
    CHECK(same(fast, run(c, true)));
    if (c.pos0 % 2 == 0) check_reference(c, fast);
    return 1;
}

/* Warm pass, capture, two replays, then a replay on new queries, which must
 * equal the same path run eagerly and stay within the reference. */
static void graph_case(uint32_t il, uint32_t n_head, bool per_head) {
    const Call c = {1, n_head, 4096, 128, TOP_K, 4};
    fill_ids(c, 0);
    const std::vector<float> first = run(c, per_head);
    ds4_gpu_tensor *heads = ds4_gpu_tensor_view(out, 0, (uint64_t)n_head * D * 4);
    CHECK(heads);
    ds4_decode_graph_key key = {};
    key.il = il;
    key.island = 2;
    key.cur_hc = q;
    std::vector<float> got;
    per_head_kernel(per_head);
    for (int round = 0; round < 4; round++) {
        if (round == 3) fill(q, hq, (size_t)MAXN * MAXH * D);
        CHECK(ds4_gpu_tensor_fill_f32(out, SENT, (uint64_t)MAXN * MAXH * D + 16));
        CHECK(ds4_gpu_synchronize());
        const int captured = ds4_gpu_decode_graph_begin(&key);
        CHECK(captured == (round == 0 ? -1 : round == 1 ? 0 : 1));
        if (captured != 1) {
            CHECK(call(c, heads));
            if (captured == 0) CHECK(ds4_gpu_decode_graph_end(&key) == 0);
        }
        CHECK(ds4_gpu_synchronize());
        got = read_out(c);
        if (round < 3) CHECK(same(got, first));
    }
    per_head_kernel(false);
    ds4_gpu_decode_graphs_invalidate();
    ds4_gpu_tensor_free(heads);
    CHECK(!same(got, first));
    CHECK(same(got, run(c, per_head)));
    check_reference(c, got);
}

int main(void) {
    for (const char *name : {"DS4_CUDA_NO_INDEXED_HEADS8", "DS4_CUDA_INDEXED_TWOPASS", "DS4_CUDA_DECODE_GRAPHS"})
        CHECK(unsetenv(name) == 0);
    CHECK(setenv("DS4_CUDA_COPY_MODEL", "1", 0) == 0);
    CHECK(ds4_gpu_init());
    if (!ds4_gpu_device_is_spark()) {
        ds4_gpu_cleanup();
        puts("CUDA indexed one-row attention: Spark required, SKIP");
        return 0;
    }
    sinks.resize(MAXH);
    for (float &v : sinks) v = random_value();
    CHECK(ds4_gpu_set_model_map(sinks.data(), sinks.size() * 4));
    q = ds4_gpu_tensor_alloc((uint64_t)MAXN * MAXH * D * 4);
    raw = ds4_gpu_tensor_alloc((uint64_t)RAW_CAP * D * 4);
    comp = ds4_gpu_tensor_alloc((uint64_t)COMP_CAP * D * 4);
    ids = ds4_gpu_tensor_alloc((uint64_t)MAXN * TOP_K * 4);
    out = ds4_gpu_tensor_alloc(((uint64_t)MAXN * MAXH * D + 16) * 4);
    CHECK(q && raw && comp && ids && out);
    fill(q, hq, (size_t)MAXN * MAXH * D);
    fill(raw, hraw, (size_t)RAW_CAP * D);
    fill(comp, hcomp, (size_t)COMP_CAP * D);

    unsigned cases = 0;
    for (uint32_t n_head : {1u, 31u, 32u, 33u, 64u, 65u, 128u})
    for (uint32_t pos0 : {6u, 300u, 2051u, 4096u, 16384u})
    for (uint32_t window : {128u, 3u, 1u, 0u})
    for (uint32_t top_k : {512u, 5u, 1u})
    for (unsigned mode = 0; mode < 5; mode++)
        cases += compare({1, n_head, pos0, window, top_k, 4}, mode);
    for (uint32_t ratio : {1u, 128u})
    for (uint32_t n_head : {1u, 32u, 64u})
    for (uint32_t pos0 : {300u, 4096u, 16384u})
    for (uint32_t window : {128u, 0u})
    for (uint32_t top_k : {512u, 1u})
    for (unsigned mode : {0u, 4u})
        cases += compare({1, n_head, pos0, window, top_k, ratio}, mode);

    /* Two and six rows keep their own paths and stay within the reference. */
    for (uint32_t n : {2u, MAXN}) {
        const Call c = {n, 32, 4096, 128, TOP_K, 4};
        fill_ids(c, 0);
        check_reference(c, run(c, false));
    }

    /* Invalid arguments are refused. */
    const Call base = {1, 32, 4096, 128, TOP_K, 4};
    fill_ids(base, 0);
    ds4_gpu_tensor *heads = ds4_gpu_tensor_view(out, 0, (uint64_t)32 * D * 4);
    CHECK(heads);
    CHECK(!ds4_gpu_attention_indexed_mixed_batch_heads_tensor(heads, sinks.data(), sinks.size() * 4, 0, q, raw, comp,
        0, ids, 1, 4096, 0, RAW_CAP, 0, 1024, TOP_K, 128, 4, 32, D));             /* no raw rows */
    CHECK(!ds4_gpu_attention_indexed_mixed_batch_heads_tensor(heads, sinks.data(), sinks.size() * 4, 0, q, raw, comp,
        0, ids, 1, 4096, 129, RAW_CAP, RAW_CAP, 1024, TOP_K, 128, 4, 32, D));     /* ring start */
    CHECK(!ds4_gpu_attention_indexed_mixed_batch_heads_tensor(heads, sinks.data(), sinks.size() * 4, 0, q, raw, comp,
        0, ids, 1, 4096, 129, RAW_CAP, 0, 1024, 513, 128, 4, 32, D));             /* top-k */
    ds4_gpu_tensor_free(heads);

    CHECK(ds4_gpu_decode_graphs_supported());
    graph_case(60, 32, false);     /* three-pass default */
    graph_case(61, 32, true);      /* DS4_CUDA_NO_INDEXED_HEADS8 */
    graph_case(62, 65, false);     /* beyond 64 heads: per-head kernel */
    graph_case(59, 128, false);

    ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(ids);
    ds4_gpu_tensor_free(comp);
    ds4_gpu_tensor_free(raw);
    ds4_gpu_tensor_free(q);
    ds4_gpu_cleanup();
    printf("CUDA indexed one-row attention: %u cases equal the per-head kernel and the reference; "
           "invalid arguments, tails, graph replays and multi-row calls PASS\n", cases);
    return 0;
}
