/* Side-stream region (ds4_gpu_side_begin/end/join) must not change any bit.
 * A reduced compressed-layer state chain (F16 compressor projections, the
 * row-batched compressor update with prefix snapshots, RMS/rope/FP8 on the
 * emitted rows, a tensor copy of the updated state, indexer weights and
 * indexer QAT) runs once inline on the main
 * stream (reference) and then repeatedly inside a side region while large Q8
 * projections keep the main stream busy.  Every output buffer must match the
 * reference byte for byte.  Also checks the region guards and that a side
 * F16 projection larger than the side scratch fails cleanly. */
#include "ds4_gpu.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <vector>

#include <cuda_fp16.h>

static uint32_t random_state = 0x51de5u;
static uint32_t random_bits(void) {
    random_state = random_state * 1664525u + 1013904223u;
    return random_state;
}
static float random_unit(void) { return ((int)(random_bits() >> 8) - (1 << 23)) / (float)(1 << 23); }

namespace {

constexpr uint32_t EMBD = 4096, HEAD_DIM = 512, RATIO = 4, COFF = 2, WIDTH = COFF * HEAD_DIM;
constexpr uint32_t IDX_HEADS = 64, IDX_DIM = 128, N_ROT = 64, ROWS = 6, POS0 = 10;
constexpr uint32_t QB_IN = 1024, QB_OUT = 16384, SNAPS = 5;
constexpr uint64_t STATE_FLOATS = (uint64_t)COFF * RATIO * WIDTH;

struct layout {
    uint64_t w_kv, w_sc, w_iw, norm, ape, w_q, size;
};

struct bufs {
    ds4_gpu_tensor *x, *y, *x9, *comp_kv, *comp_sc, *state_kv, *state_sc, *cache;
    ds4_gpu_tensor *snap_kv, *snap_sc, *idx_w, *idx_rows, *q, *out9, *copy_dst;
};

std::vector<float> g_state_kv, g_state_sc, g_idx_rows;

bool reset_state(const bufs &b) {
    return ds4_gpu_tensor_write(b.state_kv, 0, g_state_kv.data(), g_state_kv.size() * 4) &&
           ds4_gpu_tensor_write(b.state_sc, 0, g_state_sc.data(), g_state_sc.size() * 4) &&
           ds4_gpu_tensor_write(b.idx_rows, 0, g_idx_rows.data(), g_idx_rows.size() * 4) &&
           ds4_gpu_tensor_fill_f32(b.cache, 0.0f, 4u * HEAD_DIM) &&
           ds4_gpu_tensor_fill_f32(b.snap_kv, 0.0f, SNAPS * STATE_FLOATS) &&
           ds4_gpu_tensor_fill_f32(b.snap_sc, 0.0f, SNAPS * STATE_FLOATS) &&
           ds4_gpu_tensor_fill_f32(b.comp_kv, 0.0f, ROWS * WIDTH) &&
           ds4_gpu_tensor_fill_f32(b.comp_sc, 0.0f, ROWS * WIDTH) &&
           ds4_gpu_tensor_fill_f32(b.idx_w, 0.0f, ROWS * IDX_HEADS) &&
           ds4_gpu_tensor_fill_f32(b.q, 0.0f, (uint64_t)ROWS * QB_OUT) &&
           ds4_gpu_tensor_fill_f32(b.copy_dst, 0.0f, STATE_FLOATS) &&
           ds4_gpu_synchronize();
}

/* The work the engine moves to the side stream, in engine order. */
bool side_chain(const void *map, const layout &L, const bufs &b, bool pair = false) {
    const bool projected = pair
        ? ds4_gpu_matmul_f16_pair_tensor(b.comp_kv, b.comp_sc, map, L.size,
            L.w_kv, L.w_sc, EMBD, WIDTH, b.x, ROWS) != 0
        : ds4_gpu_matmul_f16_tensor(b.comp_kv, map, L.size, L.w_kv, EMBD, WIDTH, b.x, ROWS) &&
          ds4_gpu_matmul_f16_tensor(b.comp_sc, map, L.size, L.w_sc, EMBD, WIDTH, b.x, ROWS);
    if (!projected || !ds4_gpu_compressor_verify_rows_tensor(b.comp_kv, b.comp_sc, b.state_kv, b.state_sc,
                                               b.cache, 0, b.snap_kv, b.snap_sc, SNAPS,
                                               map, L.size, L.ape, 0, HEAD_DIM, RATIO,
                                               POS0, ROWS) ||
        /* Not a TP process: without the side-region guard this copy would
         * take the synchronous legacy-stream path and race the update. */
        !ds4_gpu_tensor_copy(b.copy_dst, 0, b.state_kv, 0, STATE_FLOATS * 4)) return false;
    /* Positions 10..15 emit two compressed rows (11 and 15). */
    ds4_gpu_tensor *rows = ds4_gpu_tensor_view(b.cache, 0, 2ull * HEAD_DIM * 4);
    const bool ok = rows &&
        ds4_gpu_rms_norm_weight_rows_tensor(rows, rows, map, L.size, L.norm, HEAD_DIM, 2, 1e-6f) &&
        ds4_gpu_rope_tail_tensor(rows, 2, 1, HEAD_DIM, N_ROT, POS0, 0, false,
                                 10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f) &&
        ds4_gpu_dsv4_fp8_kv_quantize_tensor(rows, 2, HEAD_DIM, N_ROT) &&
        ds4_gpu_matmul_f16_tensor(b.idx_w, map, L.size, L.w_iw, EMBD, IDX_HEADS, b.x, ROWS) &&
        ds4_gpu_dsv4_indexer_qat_tensor(b.idx_rows, ROWS * 2, IDX_DIM);
    ds4_gpu_tensor_free(rows);
    return ok;
}

/* Main-stream work running beside it: three 17.8 MB Q8 projections. */
bool main_chain(const void *map, const layout &L, const bufs &b) {
    for (int i = 0; i < 3; i++) {
        if (!ds4_gpu_matmul_q8_0_tensor(b.q, map, L.size, L.w_q, QB_IN, QB_OUT, b.y, ROWS)) return false;
    }
    return true;
}

struct snapshot {
    std::vector<std::vector<unsigned char>> bytes;
};

snapshot capture(const bufs &b) {
    const ds4_gpu_tensor *ts[] = {b.comp_kv, b.comp_sc, b.state_kv, b.state_sc, b.cache,
                                  b.snap_kv, b.snap_sc, b.idx_w, b.idx_rows, b.q, b.copy_dst};
    snapshot s;
    for (const ds4_gpu_tensor *t : ts) {
        std::vector<unsigned char> v(ds4_gpu_tensor_bytes(t));
        assert(ds4_gpu_tensor_read(t, 0, v.data(), v.size()));
        s.bytes.push_back(std::move(v));
    }
    return s;
}

const char *names[] = {"comp_kv", "comp_sc", "state_kv", "state_sc", "cache",
                       "snap_kv", "snap_sc", "idx_w", "idx_rows", "q", "copy_dst"};

int compare(const snapshot &ref, const snapshot &got, const char *what, int iter) {
    int bad = 0;
    for (size_t i = 0; i < ref.bytes.size(); i++) {
        if (ref.bytes[i] != got.bytes[i]) {
            fprintf(stderr, "%s iteration %d: %s differs from the inline reference\n",
                    what, iter, names[i]);
            bad++;
        }
    }
    return bad;
}

/* Deliberately queue the input copy behind other side-stream work. The
 * paired conversion and GEMMs must all wait for it on that same stream. */
int check_pair_stream(const void *map, const layout &L, const bufs &b) {
    constexpr uint64_t delay_bytes = 64ull << 20;
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(8ull * EMBD * 4);
    ds4_gpu_tensor *delay = ds4_gpu_tensor_alloc(2 * delay_bytes);
    ds4_gpu_tensor *out0 = ds4_gpu_tensor_alloc(8ull * WIDTH * 4);
    ds4_gpu_tensor *out1 = ds4_gpu_tensor_alloc(8ull * WIDTH * 4);
    assert(x && delay && out0 && out1);
    assert(ds4_gpu_tensor_fill_f32(delay, 0.0f, 2 * delay_bytes / sizeof(float)) &&
           ds4_gpu_synchronize());
    int failures = 0;
    for (uint32_t rows = 1; rows <= 8; rows++) {
        const uint64_t bytes = (uint64_t)rows * WIDTH * 4;
        assert(ds4_gpu_matmul_f16_pair_tensor(out0, out1, map, L.size,
            L.w_kv, L.w_sc, EMBD, WIDTH, b.x9, rows));
        std::vector<unsigned char> expected0(bytes), expected1(bytes), got0(bytes), got1(bytes);
        assert(ds4_gpu_tensor_read(out0, 0, expected0.data(), bytes));
        assert(ds4_gpu_tensor_read(out1, 0, expected1.data(), bytes));
        assert(ds4_gpu_tensor_fill_f32(x, 0.0f, 8ull * EMBD) && ds4_gpu_synchronize());
        assert(ds4_gpu_side_begin() == 1);
        assert(ds4_gpu_tensor_copy(delay, 0, delay, delay_bytes, delay_bytes));
        assert(ds4_gpu_tensor_copy(x, 0, b.x9, 0, (uint64_t)rows * EMBD * 4));
        assert(ds4_gpu_matmul_f16_pair_tensor(out0, out1, map, L.size,
            L.w_kv, L.w_sc, EMBD, WIDTH, x, rows));
        assert(ds4_gpu_side_end() == 1);
        assert(main_chain(map, L, b));
        assert(ds4_gpu_side_join() == 1 && ds4_gpu_synchronize());
        assert(ds4_gpu_tensor_read(out0, 0, got0.data(), bytes));
        assert(ds4_gpu_tensor_read(out1, 0, got1.data(), bytes));
        if (got0 != expected0 || got1 != expected1) {
            fprintf(stderr, "paired F16 side-stream ordering failed for %u rows\n", rows);
            failures++;
        }
        if (failures == 0 && ds4_gpu_decode_graphs_supported()) {
            ds4_decode_graph_key key{};
            key.il = 62;
            key.island = 2;
            key.variant = rows;
            key.cur_hc = x;
            for (int round = 0; round < 4; round++) {
                const int graph = ds4_gpu_decode_graph_begin(&key);
                assert(graph == (round == 0 ? -1 : round == 1 ? 0 : 1));
                if (graph != 1) {
                    assert(ds4_gpu_matmul_f16_pair_tensor(out0, out1, map, L.size,
                        L.w_kv, L.w_sc, EMBD, WIDTH, x, rows));
                    if (graph == 0) assert(ds4_gpu_decode_graph_end(&key) == 0);
                }
                assert(ds4_gpu_synchronize());
                assert(ds4_gpu_tensor_read(out0, 0, got0.data(), bytes));
                assert(ds4_gpu_tensor_read(out1, 0, got1.data(), bytes));
                assert(got0 == expected0 && got1 == expected1);
            }
            ds4_gpu_decode_graphs_invalidate();
        }
    }
    ds4_gpu_tensor_free(x);
    ds4_gpu_tensor_free(delay);
    ds4_gpu_tensor_free(out0);
    ds4_gpu_tensor_free(out1);
    return failures;
}

}  // namespace

int main() {
    assert(setenv("DS4_CUDA_DECODE_GRAPHS", "1", 1) == 0);
    layout L{};
    uint64_t off = 0;
    auto place = [&](uint64_t bytes) { const uint64_t at = off; off = (off + bytes + 255) & ~255ull; return at; };
    L.w_kv = place((uint64_t)WIDTH * EMBD * 2);
    L.w_sc = place((uint64_t)WIDTH * EMBD * 2);
    L.w_iw = place((uint64_t)IDX_HEADS * EMBD * 2);
    L.norm = place((uint64_t)HEAD_DIM * 4);
    L.ape = place((uint64_t)WIDTH * RATIO * 4);
    L.w_q = place((uint64_t)QB_OUT * (QB_IN / 32) * 34);
    L.size = off;
    std::vector<unsigned char> model(L.size);
    for (uint64_t i = L.w_kv; i < L.norm; i += 2) {
        const __half h = __float2half(0.02f * random_unit());
        memcpy(&model[i], &h, 2);
    }
    for (uint64_t i = L.norm; i < L.ape; i += 4) { const float v = 1.0f + 0.1f * random_unit(); memcpy(&model[i], &v, 4); }
    for (uint64_t i = L.ape; i < L.w_q; i += 4) { const float v = 0.5f * random_unit(); memcpy(&model[i], &v, 4); }
    for (uint64_t blk = 0; blk < (uint64_t)QB_OUT * (QB_IN / 32); blk++) {
        unsigned char *p = &model[L.w_q + blk * 34];
        const __half d = __float2half(0.002f + 0.002f * fabsf(random_unit()));
        memcpy(p, &d, 2);
        for (int j = 0; j < 32; j++) p[2 + j] = (unsigned char)(random_bits() >> 24);
    }
    char path[] = "/tmp/ds4-side-stream-XXXXXX";
    const int fd = mkstemp(path);
    assert(fd >= 0);
    FILE *file = fdopen(dup(fd), "wb");
    assert(file && fwrite(model.data(), 1, L.size, file) == L.size);
    assert(fclose(file) == 0 && unlink(path) == 0);
    const uint64_t span_off = 0, span_size = L.size;
    assert(ds4_gpu_init());
    assert(ds4_gpu_set_model_map_spans(model.data(), L.size, &span_off, &span_size, 1, 0));
    assert(ds4_gpu_set_model_fd_for_map(fd, model.data()));

    bufs b{};
    b.x = ds4_gpu_tensor_alloc((uint64_t)ROWS * EMBD * 4);
    b.x9 = ds4_gpu_tensor_alloc(9ull * EMBD * 4);
    b.y = ds4_gpu_tensor_alloc((uint64_t)ROWS * QB_IN * 4);
    b.comp_kv = ds4_gpu_tensor_alloc((uint64_t)ROWS * WIDTH * 4);
    b.comp_sc = ds4_gpu_tensor_alloc((uint64_t)ROWS * WIDTH * 4);
    b.state_kv = ds4_gpu_tensor_alloc(STATE_FLOATS * 4);
    b.state_sc = ds4_gpu_tensor_alloc(STATE_FLOATS * 4);
    b.cache = ds4_gpu_tensor_alloc(4ull * HEAD_DIM * 4);
    b.snap_kv = ds4_gpu_tensor_alloc(SNAPS * STATE_FLOATS * 4);
    b.snap_sc = ds4_gpu_tensor_alloc(SNAPS * STATE_FLOATS * 4);
    b.idx_w = ds4_gpu_tensor_alloc((uint64_t)ROWS * IDX_HEADS * 4);
    b.idx_rows = ds4_gpu_tensor_alloc((uint64_t)ROWS * 2 * IDX_DIM * 4);
    b.q = ds4_gpu_tensor_alloc((uint64_t)ROWS * QB_OUT * 4);
    b.out9 = ds4_gpu_tensor_alloc(9ull * WIDTH * 4);
    b.copy_dst = ds4_gpu_tensor_alloc(STATE_FLOATS * 4);
    assert(b.x && b.x9 && b.y && b.comp_kv && b.comp_sc && b.state_kv && b.state_sc && b.cache &&
           b.snap_kv && b.snap_sc && b.idx_w && b.idx_rows && b.q && b.out9 && b.copy_dst);
    {
        std::vector<float> v((size_t)9 * EMBD);
        for (float &f : v) f = random_unit();
        assert(ds4_gpu_tensor_write(b.x, 0, v.data(), (uint64_t)ROWS * EMBD * 4));
        assert(ds4_gpu_tensor_write(b.x9, 0, v.data(), v.size() * 4));
        std::vector<float> y((size_t)ROWS * QB_IN);
        for (float &f : y) f = random_unit();
        assert(ds4_gpu_tensor_write(b.y, 0, y.data(), y.size() * 4));
    }
    g_state_kv.resize(STATE_FLOATS);
    g_state_sc.resize(STATE_FLOATS);
    g_idx_rows.resize((size_t)ROWS * 2 * IDX_DIM);
    for (float &f : g_state_kv) f = random_unit();
    for (float &f : g_state_sc) f = -1.0f + random_unit();
    for (float &f : g_idx_rows) f = random_unit();

    /* Reference: everything inline on the main stream. */
    assert(reset_state(b));
    assert(side_chain(model.data(), L, b) && main_chain(model.data(), L, b));
    assert(ds4_gpu_synchronize());
    const snapshot ref = capture(b);

    /* Guards. */
    int fails = 0;
    if (ds4_gpu_side_end() != 0) { fprintf(stderr, "end without begin must return 0\n"); fails++; }
    if (ds4_gpu_side_join() != 1) { fprintf(stderr, "idle join must return 1\n"); fails++; }
    const int first = ds4_gpu_side_begin();
    if (first == 0) {
        printf("SKIP: side stream unavailable on this device configuration\n");
        ds4_gpu_cleanup();
        return 0;
    }
    if (first != 1) { fprintf(stderr, "begin failed (%d)\n", first); return 1; }
    if (ds4_gpu_side_begin() != -1) { fprintf(stderr, "nested begin must return -1\n"); fails++; }
    if (ds4_gpu_side_end() != 1 || ds4_gpu_side_join() != 1) { fprintf(stderr, "empty region failed\n"); fails++; }

    /* A previously created side stream must still stay out of graph capture. */
    ds4_decode_graph_key key{};
    key.il = 63;
    key.island = 2;
    key.variant = 0x51de;
    key.cur_hc = b.x;
    if (ds4_gpu_decode_graphs_supported()) {
        assert(ds4_gpu_decode_graph_begin(&key) == -1);
        assert(ds4_gpu_decode_graph_begin(&key) == 0);
        assert(ds4_gpu_side_begin() == 0);
        assert(ds4_gpu_rms_norm_weight_rows_tensor(b.cache, b.cache,
            model.data(), L.size, L.norm, HEAD_DIM, 2, 1e-6f));
        assert(ds4_gpu_decode_graph_end(&key) == 0);
        assert(ds4_gpu_decode_graph_begin(&key) == 1);
        assert(ds4_gpu_synchronize());
        ds4_gpu_decode_graphs_invalidate();
    }

    /* Side runs: both enqueue orders, repeated to give races a chance. */
    for (int iter = 0; iter < 40; iter++) {
        assert(reset_state(b));
        bool ok = true;
        if (iter & 1) ok = main_chain(model.data(), L, b);
        const int rc = ds4_gpu_side_begin();
        if (rc != 1) { fprintf(stderr, "iteration %d: begin returned %d\n", iter, rc); fails++; break; }
        ok = side_chain(model.data(), L, b, iter >= 20) && ok;
        if (ds4_gpu_side_end() != 1) { fprintf(stderr, "iteration %d: end failed\n", iter); fails++; }
        if (!(iter & 1)) ok = main_chain(model.data(), L, b) && ok;
        if (ds4_gpu_side_join() != 1) { fprintf(stderr, "iteration %d: join failed\n", iter); fails++; }
        if (!ok || !ds4_gpu_synchronize()) { fprintf(stderr, "iteration %d: launch failed\n", iter); fails++; break; }
        fails += compare(ref, capture(b), "side region", iter);
    }

    /* A side F16 projection beyond the side scratch (9 x 4096 F16 > 64 KiB)
     * must fail without side effects; the region still closes, and the same
     * call then succeeds on the main stream. */
    {
        const int rc = ds4_gpu_side_begin();
        const int big = rc == 1 ? ds4_gpu_matmul_f16_tensor(b.out9, model.data(), L.size, L.w_kv,
                                                            EMBD, WIDTH, b.x9, 9) : -1;
        const int end = ds4_gpu_side_end(), join = ds4_gpu_side_join();
        const int inl = ds4_gpu_matmul_f16_tensor(b.out9, model.data(), L.size, L.w_kv, EMBD, WIDTH, b.x9, 9);
        if (rc != 1 || big != 0 || end != 1 || join != 1 || inl != 1 || !ds4_gpu_synchronize()) {
            fprintf(stderr, "oversized side projection: begin %d side %d end %d join %d inline %d\n",
                    rc, big, end, join, inl);
            fails++;
        }
    }
    fails += check_pair_stream(model.data(), L, b);
    ds4_gpu_tensor *allocated[] = {b.x, b.y, b.x9, b.comp_kv, b.comp_sc,
        b.state_kv, b.state_sc, b.cache, b.snap_kv, b.snap_sc, b.idx_w,
        b.idx_rows, b.q, b.out9, b.copy_dst};
    for (ds4_gpu_tensor *t : allocated) ds4_gpu_tensor_free(t);
    ds4_gpu_cleanup();
    close(fd);
    printf("side stream region: %s (%d failures)\n", fails ? "FAIL" : "PASS", fails);
    return fails ? 1 : 0;
}
