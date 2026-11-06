/* Verify-head vocab split primitives (no TP, one GPU).
 *
 * 1. The Q8_0 output head (V = 129280, K = 4096) computed as two half-vocab
 *    calls, rows [0, V/2) and [V/2, V) at a weight row offset, must equal
 *    the full-vocab call bit for bit for every verify width the split uses
 *    (1, the 8-row padded 2..7, and 8..17).  This is the call
 *    metal_graph_output_logits_head_matmul makes on each rank.  Mismatches
 *    are counted, not assumed away; the split is only valid at 0.  Median
 *    times of the full and half calls are printed.
 * 2. Per-row top-1 over each half (ds4_gpu_indexer_top1_value_tensor with a
 *    global index offset) combined on the host with the engine's rule
 *    (larger value, then lower id) must equal the unsplit selection
 *    (ds4_gpu_indexer_topk_tensor k = 1 and ds4_gpu_argmax_tensor) on rows
 *    with cross-half and boundary ties, signed zeros, infinities, all -inf,
 *    NaN halves and all-NaN rows. */
#include "ds4_gpu.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <limits>
#include <unistd.h>
#include <vector>

#include <cuda_fp16.h>

static uint32_t random_state = 0x7e5a11u;
static uint32_t random_bits(void) {
    random_state = random_state * 1664525u + 1013904223u;
    return random_state;
}
static float random_unit(void) { return ((int)(random_bits() >> 8) - (1 << 23)) / (float)(1 << 23); }

static double now_sec(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

namespace {

constexpr uint32_t V = 129280, HALF = V / 2, K = 4096, MAX_ROWS = 17;
constexpr uint64_t ROW_BYTES = (uint64_t)(K / 32) * 34;

struct dev_tensor {
    ds4_gpu_tensor *t = nullptr;
    explicit dev_tensor(uint64_t bytes) : t(ds4_gpu_tensor_alloc(bytes)) {}
    ~dev_tensor() { if (t) ds4_gpu_tensor_free(t); }
    dev_tensor(const dev_tensor &) = delete;
    dev_tensor &operator=(const dev_tensor &) = delete;
};

/* Head rows as the engine runs them: n_tokens 2..7 are padded to 8 rows. */
uint32_t head_rows(uint32_t n) { return n > 1 && n < 8 ? 8u : n; }

bool head_call(ds4_gpu_tensor *out, const void *map, uint64_t size, uint64_t row0,
               uint64_t rows, ds4_gpu_tensor *x, uint32_t n) {
    return ds4_gpu_matmul_q8_0_tensor(out, map, size, row0 * ROW_BYTES, K, rows, x, n) != 0;
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

int check_head(const std::vector<unsigned char> &model) {
    const void *map = model.data();
    const uint64_t size = model.size();
    dev_tensor x((uint64_t)MAX_ROWS * K * 4);
    dev_tensor full((uint64_t)MAX_ROWS * V * 4);
    dev_tensor lo((uint64_t)MAX_ROWS * HALF * 4);
    dev_tensor hi((uint64_t)MAX_ROWS * HALF * 4);
    assert(x.t && full.t && lo.t && hi.t);
    std::vector<float> hx((size_t)MAX_ROWS * K), hf((size_t)MAX_ROWS * V),
                       hl((size_t)MAX_ROWS * HALF), hh((size_t)MAX_ROWS * HALF);
    int failures = 0;
    const uint32_t widths[] = {1, 2, 6, 8, 9, 12, 16, 17};
    for (uint32_t n : widths) {
        const uint32_t rows = head_rows(n);
        for (float &v : hx) v = random_unit();
        /* Padding rows are zero in the engine (the norm tail is filled). */
        std::fill(hx.begin() + (size_t)n * K, hx.end(), 0.0f);
        assert(ds4_gpu_tensor_write(x.t, 0, hx.data(), (uint64_t)rows * K * 4));
        assert(ds4_gpu_tensor_fill_f32(full.t, -1.0f, (uint64_t)MAX_ROWS * V));
        assert(ds4_gpu_tensor_fill_f32(lo.t, -2.0f, (uint64_t)MAX_ROWS * HALF));
        assert(ds4_gpu_tensor_fill_f32(hi.t, -3.0f, (uint64_t)MAX_ROWS * HALF));
        if (!head_call(full.t, map, size, 0, V, x.t, rows) ||
            !head_call(lo.t, map, size, 0, HALF, x.t, rows) ||
            !head_call(hi.t, map, size, HALF, HALF, x.t, rows) ||
            !ds4_gpu_synchronize()) {
            fprintf(stderr, "head n=%u: matmul failed\n", n);
            failures++;
            continue;
        }
        assert(ds4_gpu_tensor_read(full.t, 0, hf.data(), (uint64_t)rows * V * 4));
        assert(ds4_gpu_tensor_read(lo.t, 0, hl.data(), (uint64_t)rows * HALF * 4));
        assert(ds4_gpu_tensor_read(hi.t, 0, hh.data(), (uint64_t)rows * HALF * 4));
        uint64_t mismatches = 0, rows_hit = 0;
        double max_diff = 0.0;
        for (uint32_t r = 0; r < n; r++) {
            uint64_t row_bad = 0;
            for (uint32_t j = 0; j < V; j++) {
                const float a = hf[(size_t)r * V + j];
                const float b = j < HALF ? hl[(size_t)r * HALF + j]
                                         : hh[(size_t)r * HALF + (j - HALF)];
                if (memcmp(&a, &b, sizeof(a)) != 0) {
                    row_bad++;
                    max_diff = std::max(max_diff, (double)fabsf(a - b));
                }
            }
            mismatches += row_bad;
            rows_hit += row_bad != 0;
        }
        /* Timing: 20 calls each after the correctness pass. */
        std::vector<double> tf, th;
        for (int it = 0; it < 20; it++) {
            double t0 = now_sec();
            assert(head_call(full.t, map, size, 0, V, x.t, rows) && ds4_gpu_synchronize());
            tf.push_back(now_sec() - t0);
            t0 = now_sec();
            assert(head_call(hi.t, map, size, HALF, HALF, x.t, rows) && ds4_gpu_synchronize());
            th.push_back(now_sec() - t0);
        }
        fprintf(stderr, "head n=%2u (rows %2u): %llu mismatching logits in %llu rows, max |diff| %.3g; "
                "full %.3f ms, half %.3f ms: %s\n",
                n, rows, (unsigned long long)mismatches, (unsigned long long)rows_hit, max_diff,
                median(tf) * 1e3, median(th) * 1e3, mismatches ? "FAIL" : "PASS");
        failures += mismatches != 0;
    }
    return failures;
}

/* The engine's combine (metal_graph_verify_head_split_combine), including
 * its flush of subnormals to match the kernel's FTZ compares. */
float ftz(float v) {
    uint32_t u;
    memcpy(&u, &v, sizeof(u));
    if ((u & 0x7f800000u) == 0) u &= 0x80000000u;
    memcpy(&v, &u, sizeof(v));
    return v;
}

uint32_t combine(float mv, uint32_t mi, float pv, uint32_t pi) {
    mv = ftz(mv);
    pv = ftz(pv);
    const bool take_peer = pv > mv || (pv == mv && pi < mi);
    return take_peer ? pi : mi;
}

int check_top1(void) {
    constexpr uint32_t ROWS = 17;
    const float inf = std::numeric_limits<float>::infinity();
    const float nan = std::numeric_limits<float>::quiet_NaN();
    std::vector<float> s((size_t)ROWS * V);
    for (float &v : s) v = random_unit() * 10.0f;
    auto row = [&](uint32_t r) { return &s[(size_t)r * V]; };
    /* 0, 1: random rows.  2: the same max in both halves -> lower half wins. */
    row(2)[777] = 50.0f; row(2)[HALF + 5] = 50.0f;
    /* 3: the same max at the split boundary. */
    row(3)[HALF - 1] = 50.0f; row(3)[HALF] = 50.0f;
    /* 4: ties inside the upper half only. */
    row(4)[HALF + 900] = 60.0f; row(4)[HALF + 100] = 60.0f;
    /* 5: all -inf -> index 0. */
    std::fill(row(5), row(5) + V, -inf);
    /* 6: lower half NaN, upper half finite. */
    std::fill(row(6), row(6) + HALF, nan);
    /* 7: lower half NaN, upper half -inf. */
    std::fill(row(7), row(7) + HALF, nan);
    std::fill(row(7) + HALF, row(7) + V, -inf);
    /* 8: all NaN. */
    std::fill(row(8), row(8) + V, nan);
    /* 9: signed zeros are the max (others negative): -0 at a lower index. */
    for (uint32_t j = 0; j < V; j++) row(9)[j] = -1.0f - fabsf(random_unit());
    row(9)[HALF + 3] = 0.0f; row(9)[40] = -0.0f;
    /* 10: +inf in both halves. */
    row(10)[HALF + 1] = inf; row(10)[12345] = inf;
    /* 11: max at index 0 and at V - 1. */
    row(11)[0] = 99.0f; row(11)[V - 1] = 99.0f;
    /* 12: max only at V - 1.  13: upper half all -inf, lower finite. */
    row(12)[V - 1] = 99.0f;
    std::fill(row(13) + HALF, row(13) + V, -inf);
    /* 14: denormal max over a negative row; 15: NaNs sprinkled in. */
    for (uint32_t j = 0; j < V; j++) row(14)[j] = -1.0f;
    row(14)[HALF + 77] = 1e-42f; row(14)[HALF + 76] = 1e-42f;
    for (uint32_t j = 0; j < V; j += 3) row(15)[j] = nan;
    /* 16: distinct subnormal maxima in the two halves (FTZ: a tie). */
    for (uint32_t j = 0; j < V; j++) row(16)[j] = -1.0f;
    row(16)[HALF + 9] = 3e-42f; row(16)[9] = 2e-42f;

    std::vector<float> lo((size_t)ROWS * HALF), hi((size_t)ROWS * HALF);
    for (uint32_t r = 0; r < ROWS; r++) {
        memcpy(&lo[(size_t)r * HALF], row(r), HALF * 4);
        memcpy(&hi[(size_t)r * HALF], row(r) + HALF, HALF * 4);
    }
    dev_tensor ds((uint64_t)ROWS * V * 4), dlo((uint64_t)ROWS * HALF * 4), dhi((uint64_t)ROWS * HALF * 4);
    dev_tensor ref(ROWS * 4), ref1(4), slo(ROWS * 4), shi(ROWS * 4), vlo(ROWS * 4), vhi(ROWS * 4);
    assert(ds.t && dlo.t && dhi.t && ref.t && ref1.t && slo.t && shi.t && vlo.t && vhi.t);
    assert(ds4_gpu_tensor_write(ds.t, 0, s.data(), s.size() * 4));
    assert(ds4_gpu_tensor_write(dlo.t, 0, lo.data(), lo.size() * 4));
    assert(ds4_gpu_tensor_write(dhi.t, 0, hi.data(), hi.size() * 4));
    assert(ds4_gpu_indexer_topk_tensor(ref.t, ds.t, V, ROWS, 1));
    assert(ds4_gpu_indexer_top1_value_tensor(slo.t, vlo.t, dlo.t, HALF, ROWS, 0));
    assert(ds4_gpu_indexer_top1_value_tensor(shi.t, vhi.t, dhi.t, HALF, ROWS, HALF));
    assert(ds4_gpu_synchronize());
    uint32_t r_ids[ROWS], lo_ids[ROWS], hi_ids[ROWS];
    float lo_v[ROWS], hi_v[ROWS];
    assert(ds4_gpu_tensor_read(ref.t, 0, r_ids, sizeof(r_ids)));
    assert(ds4_gpu_tensor_read(slo.t, 0, lo_ids, sizeof(lo_ids)));
    assert(ds4_gpu_tensor_read(shi.t, 0, hi_ids, sizeof(hi_ids)));
    assert(ds4_gpu_tensor_read(vlo.t, 0, lo_v, sizeof(lo_v)));
    assert(ds4_gpu_tensor_read(vhi.t, 0, hi_v, sizeof(hi_v)));
    int failures = 0;
    for (uint32_t r = 0; r < ROWS; r++) {
        /* Both ranks combine with their own half as "mine". */
        const uint32_t leader = combine(lo_v[r], lo_ids[r], hi_v[r], hi_ids[r]);
        const uint32_t worker = combine(hi_v[r], hi_ids[r], lo_v[r], lo_ids[r]);
        /* The one-row unsplit path (top_rows == 1) goes through argmax. */
        uint32_t a1 = 0;
        dev_tensor one((uint64_t)V * 4);
        assert(one.t && ds4_gpu_tensor_write(one.t, 0, row(r), (uint64_t)V * 4) &&
               ds4_gpu_argmax_tensor(ref1.t, one.t, V) && ds4_gpu_synchronize() &&
               ds4_gpu_tensor_read(ref1.t, 0, &a1, 4));
        const bool ok = leader == r_ids[r] && worker == r_ids[r] && a1 == r_ids[r];
        if (!ok) {
            fprintf(stderr, "top1 row %2u: unsplit %u argmax %u, split leader %u worker %u "
                    "(lo %u %g, hi %u %g): FAIL\n", r, r_ids[r], a1, leader, worker,
                    lo_ids[r], lo_v[r], hi_ids[r], hi_v[r]);
        }
        failures += !ok;
    }
    fprintf(stderr, "top1 combine: %u rows (ties across/at the split, +-0, inf, -inf, NaN, subnormal): %s\n",
            ROWS, failures ? "FAIL" : "PASS");
    return failures;
}

} /* namespace */

int main() {
    if (getenv("DS4_CUDA_NO_TOP1")) {
        fprintf(stderr, "unset DS4_CUDA_NO_TOP1: the engine split requires the top-1 kernel\n");
        return 2;
    }
    std::vector<unsigned char> model((size_t)V * ROW_BYTES);
    for (uint64_t blk = 0; blk < (uint64_t)V * (K / 32); blk++) {
        unsigned char *p = &model[blk * 34];
        const __half d = __float2half(0.002f + 0.002f * fabsf(random_unit()));
        memcpy(p, &d, 2);
        for (int j = 0; j < 32; j++) p[2 + j] = (unsigned char)(random_bits() >> 24);
    }
    char path[] = "/tmp/ds4-head-split-XXXXXX";
    const int fd = mkstemp(path);
    assert(fd >= 0);
    FILE *file = fdopen(dup(fd), "wb");
    assert(file && fwrite(model.data(), 1, model.size(), file) == model.size());
    assert(fclose(file) == 0 && unlink(path) == 0);
    const uint64_t span_off = 0, span_size = model.size();
    assert(ds4_gpu_init());
    if (!ds4_gpu_device_is_spark())
        fprintf(stderr, "note: not a DGX Spark; the engine enables the split on Spark only\n");
    assert(ds4_gpu_set_model_map_spans(model.data(), model.size(), &span_off, &span_size, 1, 0));
    assert(ds4_gpu_set_model_fd_for_map(fd, model.data()));
    const int failures = check_head(model) + check_top1();
    close(fd);
    fprintf(stderr, "%s\n", failures ? "FAIL" : "PASS");
    return failures ? 1 : 0;
}
