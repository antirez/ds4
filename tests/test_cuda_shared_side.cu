/* Shared-expert slice in a side region beside main-stream Q8 work must not
 * change any bit.  This is the CUDA network-TP verify FFN: each rank's
 * shared-expert slice (gate/up row halves through dense Q8 MMQ, SwiGLU,
 * K-half down through the Q8 K-slice kernel) runs on the side stream while
 * the main stream runs the routed work, and after the join the shared output
 * is added into the exchanged partial.
 *
 * The main stream here first runs the F16 router projection, then dense Q8
 * MMQ (pool memory on the main stream while the side MMQ uses pool memory
 * on the side stream) and a Q8 K-slice
 * (main tmp scratch while the side K-slice uses the side scratch), so both
 * scratch paths are exercised concurrently.  For both rank lanes, the outputs
 * of 20 side-region runs (alternating enqueue orders) must match an
 * all-inline reference byte for byte. */
#include "ds4_gpu.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <vector>

#include <cuda_fp16.h>

static uint32_t random_state = 0x5ade51u;
static uint32_t random_bits(void) {
    random_state = random_state * 1664525u + 1013904223u;
    return random_state;
}
static float random_unit(void) { return ((int)(random_bits() >> 8) - (1 << 23)) / (float)(1 << 23); }

namespace {

constexpr uint32_t EMBD = 4096, SHARED = 2048, HALF = SHARED / 2, MAX_ROWS = 6;
constexpr uint32_t EXPERTS = 256;
/* Rows of the current case: verify blocks use 2..6, the TP drafter 5. */
uint32_t ROWS = MAX_ROWS;
constexpr uint32_t MAIN_IN = 1024, MAIN_OUT = 16384;
constexpr float CLAMP = 10.0f;
constexpr uint64_t ROW_EMBD = (uint64_t)(EMBD / 32) * 34;      /* Q8_0 row of 4096 */
constexpr uint64_t ROW_SHARED = (uint64_t)(SHARED / 32) * 34;  /* Q8_0 row of 2048 */
constexpr uint64_t ROW_MAIN = (uint64_t)(MAIN_IN / 32) * 34;

struct layout {
    uint64_t gate, up, down, main_w, main_k, router, size;
};

struct bufs {
    ds4_gpu_tensor *x, *gate, *up, *mid, *shared_out, *partial;
    ds4_gpu_tensor *main_x, *main_mid, *main_out, *router_out;
};

void fill_q8(unsigned char *p, uint64_t blocks) {
    for (uint64_t b = 0; b < blocks; b++, p += 34) {
        const __half d = __float2half(0.002f + 0.002f * fabsf(random_unit()));
        memcpy(p, &d, 2);
        for (int j = 0; j < 32; j++) p[2 + j] = (unsigned char)(random_bits() >> 24);
    }
}

/* The engine's shared-expert slice for one rank, in engine order
 * (DS4_METAL_ENCODE_PREFILL_SHARED_EXPERT, tp_shared_slice branch). */
bool shared_chain(const void *map, const layout &L, const bufs &b, uint32_t rank) {
    const uint64_t lane = (uint64_t)rank * HALF * ROW_EMBD;
    return ds4_gpu_matmul_q8_0_tensor(b.gate, map, L.size, L.gate + lane, EMBD, HALF, b.x, ROWS) &&
           ds4_gpu_matmul_q8_0_tensor(b.up, map, L.size, L.up + lane, EMBD, HALF, b.x, ROWS) &&
           ds4_gpu_swiglu_tensor(b.mid, b.gate, b.up, ROWS * HALF, CLAMP, 1.0f) &&
           ds4_gpu_matmul_q8_0_kslice_rows_tensor(b.shared_out, map, L.size, L.down, SHARED, EMBD,
                                                  (uint64_t)rank * HALF, HALF, b.mid, ROWS);
}

/* Main-stream work beside it: F16 router, three dense Q8 MMQs (pool memory)
 * and a Q8 K-slice (main tmp scratch) into the partial. */
bool main_chain(const void *map, const layout &L, const bufs &b) {
    if (!ds4_gpu_matmul_f16_tensor(b.router_out, map, L.size, L.router,
                                   EMBD, EXPERTS, b.x, ROWS)) return false;
    for (int i = 0; i < 3; i++)
        if (!ds4_gpu_matmul_q8_0_tensor(b.main_out, map, L.size, L.main_w, MAIN_IN, MAIN_OUT,
                                        b.main_x, ROWS)) return false;
    return ds4_gpu_matmul_q8_0_kslice_rows_tensor(b.partial, map, L.size, L.main_k, SHARED, EMBD,
                                                  0, HALF, b.main_mid, ROWS);
}

/* After the join: the shared rows are folded into the exchanged partial. */
bool fold(const bufs &b) {
    return ds4_gpu_add_tensor(b.partial, b.partial, b.shared_out, ROWS * EMBD);
}

std::vector<std::vector<unsigned char>> snapshot(const bufs &b) {
    const ds4_gpu_tensor *ts[] = {b.gate, b.up, b.mid, b.shared_out, b.partial, b.main_out, b.router_out};
    std::vector<std::vector<unsigned char>> s;
    for (const ds4_gpu_tensor *t : ts) {
        std::vector<unsigned char> v(ds4_gpu_tensor_bytes(t));
        assert(ds4_gpu_tensor_read(t, 0, v.data(), v.size()));
        s.push_back(std::move(v));
    }
    return s;
}

const char *names[] = {"shared_gate", "shared_up", "shared_mid", "shared_out", "partial", "main_out", "router_out"};

bool clear(const bufs &b) {
    return ds4_gpu_tensor_fill_f32(b.gate, -1.0f, (uint64_t)MAX_ROWS * HALF) &&
           ds4_gpu_tensor_fill_f32(b.up, -1.0f, (uint64_t)MAX_ROWS * HALF) &&
           ds4_gpu_tensor_fill_f32(b.mid, -1.0f, (uint64_t)MAX_ROWS * HALF) &&
           ds4_gpu_tensor_fill_f32(b.shared_out, -1.0f, (uint64_t)MAX_ROWS * EMBD) &&
           ds4_gpu_tensor_fill_f32(b.partial, -1.0f, (uint64_t)MAX_ROWS * EMBD) &&
           ds4_gpu_tensor_fill_f32(b.main_out, -1.0f, (uint64_t)MAX_ROWS * MAIN_OUT) &&
           ds4_gpu_tensor_fill_f32(b.router_out, -1.0f, (uint64_t)MAX_ROWS * EXPERTS) &&
           ds4_gpu_synchronize();
}

} // namespace

int main() {
    layout L{};
    uint64_t off = 0;
    auto place = [&](uint64_t bytes) { const uint64_t at = off; off = (off + bytes + 255) & ~255ull; return at; };
    L.gate = place((uint64_t)SHARED * ROW_EMBD);
    L.up = place((uint64_t)SHARED * ROW_EMBD);
    L.down = place((uint64_t)EMBD * ROW_SHARED);
    L.main_w = place((uint64_t)MAIN_OUT * ROW_MAIN);
    L.main_k = place((uint64_t)EMBD * ROW_SHARED);
    L.router = place((uint64_t)EXPERTS * EMBD * sizeof(__half));
    L.size = off;
    std::vector<unsigned char> model(L.size);
    fill_q8(&model[L.gate], (uint64_t)SHARED * (EMBD / 32));
    fill_q8(&model[L.up], (uint64_t)SHARED * (EMBD / 32));
    fill_q8(&model[L.down], (uint64_t)EMBD * (SHARED / 32));
    fill_q8(&model[L.main_w], (uint64_t)MAIN_OUT * (MAIN_IN / 32));
    fill_q8(&model[L.main_k], (uint64_t)EMBD * (SHARED / 32));
    for (uint64_t i = 0; i < (uint64_t)EXPERTS * EMBD; i++) {
        const __half w = __float2half(random_unit() * 0.02f);
        memcpy(&model[L.router + i * sizeof(w)], &w, sizeof(w));
    }
    char path[] = "/tmp/ds4-shared-side-XXXXXX";
    const int fd = mkstemp(path);
    assert(fd >= 0);
    FILE *file = fdopen(dup(fd), "wb");
    assert(file && fwrite(model.data(), 1, L.size, file) == L.size);
    assert(fclose(file) == 0 && unlink(path) == 0);
    const uint64_t span_off = 0, span_size = L.size;
    assert(ds4_gpu_init());
    assert(ds4_gpu_set_model_map_spans(model.data(), L.size, &span_off, &span_size, 1, 0));
    assert(ds4_gpu_set_model_fd_for_map(fd, model.data()));
    if (!ds4_gpu_side_dense_q8_ok()) {
        printf("SKIP: a persistent Q8_1 scratch is active (the engine keeps the shared expert inline)\n");
        return 0;
    }

    bufs b{};
    b.x = ds4_gpu_tensor_alloc((uint64_t)MAX_ROWS * EMBD * 4);
    b.gate = ds4_gpu_tensor_alloc((uint64_t)MAX_ROWS * HALF * 4);
    b.up = ds4_gpu_tensor_alloc((uint64_t)MAX_ROWS * HALF * 4);
    b.mid = ds4_gpu_tensor_alloc((uint64_t)MAX_ROWS * HALF * 4);
    b.shared_out = ds4_gpu_tensor_alloc((uint64_t)MAX_ROWS * EMBD * 4);
    b.partial = ds4_gpu_tensor_alloc((uint64_t)MAX_ROWS * EMBD * 4);
    b.main_x = ds4_gpu_tensor_alloc((uint64_t)MAX_ROWS * MAIN_IN * 4);
    b.main_mid = ds4_gpu_tensor_alloc((uint64_t)MAX_ROWS * HALF * 4);
    b.main_out = ds4_gpu_tensor_alloc((uint64_t)MAX_ROWS * MAIN_OUT * 4);
    b.router_out = ds4_gpu_tensor_alloc((uint64_t)MAX_ROWS * EXPERTS * 4);
    assert(b.x && b.gate && b.up && b.mid && b.shared_out && b.partial &&
           b.main_x && b.main_mid && b.main_out && b.router_out);
    {
        std::vector<float> v((size_t)MAX_ROWS * EMBD);
        for (float &f : v) f = random_unit();
        assert(ds4_gpu_tensor_write(b.x, 0, v.data(), v.size() * 4));
        std::vector<float> m((size_t)MAX_ROWS * MAIN_IN);
        for (float &f : m) f = random_unit();
        assert(ds4_gpu_tensor_write(b.main_x, 0, m.data(), m.size() * 4));
        std::vector<float> h((size_t)MAX_ROWS * HALF);
        for (float &f : h) f = random_unit();
        assert(ds4_gpu_tensor_write(b.main_mid, 0, h.data(), h.size() * 4));
    }

    int fails = 0;
    for (const uint32_t rows : {2u, 3u, 4u, 5u, 6u})
    for (uint32_t rank = 0; rank < 2; rank++) {
        ROWS = rows;
        /* Reference: everything inline, in the engine's eager order. */
        assert(clear(b));
        assert(main_chain(model.data(), L, b) && shared_chain(model.data(), L, b, rank) && fold(b));
        assert(ds4_gpu_synchronize());
        const auto ref = snapshot(b);

        for (int iter = 0; iter < 20; iter++) {
            assert(clear(b));
            bool ok = true;
            /* Odd iterations queue main work before the region as well. */
            if (iter & 1) ok = ds4_gpu_matmul_q8_0_tensor(b.main_out, model.data(), L.size, L.main_w,
                                                          MAIN_IN, MAIN_OUT, b.main_x, ROWS) != 0;
            const int rc = ds4_gpu_side_begin();
            if (rc == 0) {
                printf("SKIP: side stream unavailable on this device configuration\n");
                ds4_gpu_cleanup();
                return 0;
            }
            if (rc != 1) { fprintf(stderr, "rank %u iteration %d: begin returned %d\n", rank, iter, rc); fails++; break; }
            ok = shared_chain(model.data(), L, b, rank) && ok;
            if (ds4_gpu_side_end() != 1) { fprintf(stderr, "rank %u iteration %d: end failed\n", rank, iter); fails++; }
            ok = main_chain(model.data(), L, b) && ok;
            if (ds4_gpu_side_join() != 1) { fprintf(stderr, "rank %u iteration %d: join failed\n", rank, iter); fails++; }
            ok = fold(b) && ok;
            if (!ok || !ds4_gpu_synchronize()) {
                fprintf(stderr, "rank %u iteration %d: launch failed\n", rank, iter);
                fails++;
                break;
            }
            const auto got = snapshot(b);
            for (size_t i = 0; i < ref.size(); i++) {
                if (ref[i] != got[i]) {
                    fprintf(stderr, "rows %u rank %u iteration %d: %s differs from the inline reference\n",
                            rows, rank, iter, names[i]);
                    fails++;
                }
            }
        }
    }
    ds4_gpu_tensor *allocated[] = {b.x, b.gate, b.up, b.mid, b.shared_out, b.partial,
                                   b.main_x, b.main_mid, b.main_out, b.router_out};
    for (ds4_gpu_tensor *t : allocated) ds4_gpu_tensor_free(t);
    ds4_gpu_cleanup();
    close(fd);
    printf("shared expert side region (rows 2..6 x 2 ranks x 20 runs): %s (%d failures)\n",
           fails ? "FAIL" : "PASS", fails);
    return fails ? 1 : 0;
}
