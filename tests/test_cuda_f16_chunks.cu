#include "ds4_gpu.h"
#include <cuda_fp16.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); exit(1); } } while (0)

/* One-row F16 pairs on Spark read chunk-interleaved copies of the compressor
 * weights, built at startup.  Every call here must return exactly what the
 * raw weights give before any copy exists.
 *
 * The fixture GGUF holds compressor pairs of the real widths (1024, 512, 256
 * outputs) and an odd one (641), a same-named pair whose K is not a multiple
 * of 256 (never copied), a candidate paired with a non-candidate (partial
 * pair), a same-named F32 tensor (never copied) and a Q8_0 projection that
 * single-GPU builds repack.  A second file adds an MXFP4 tensor, whose
 * checkpoints must keep building no aligned repacks.
 *
 * The copies are read from the file, while DS4_CUDA_COPY_MODEL takes the raw
 * weights from the mapping.  Changing the mapped F16 weights before the copy
 * therefore makes the two sources differ, which shows which one each call
 * used: the copies for aligned one-row pairs, raw weights everywhere else. */

static uint32_t state = 4711;
static uint32_t next_u32() {
    state = state * 1664525u + 1013904223u;
    return state;
}
static float random_value() { return ((int)(next_u32() >> 16) - 32768) / 8192.0f; }

struct Tensor { std::string name; uint32_t type; uint64_t in, out; std::vector<uint8_t> data; uint64_t off; };

static std::vector<uint8_t> f16_data(uint64_t in, uint64_t out) {
    std::vector<uint8_t> d(in * out * 2);
    for (size_t i = 0; i < d.size(); i += 2) {
        const __half h = __float2half(random_value() / 16.0f);
        memcpy(&d[i], &h, 2);
    }
    return d;
}
static std::vector<uint8_t> q8_data(uint64_t in, uint64_t out) {
    std::vector<uint8_t> d(out * (in / 32) * 34);
    for (size_t i = 0; i < d.size(); i += 34) {
        const __half s = __float2half(0.01f);
        memcpy(&d[i], &s, 2);
        for (int j = 2; j < 34; j++) d[i + j] = (uint8_t)next_u32();
    }
    return d;
}

static void put(std::vector<uint8_t> &b, const void *p, size_t n) { b.insert(b.end(), (const uint8_t *)p, (const uint8_t *)p + n); }
static void put_u32(std::vector<uint8_t> &b, uint32_t v) { put(b, &v, 4); }
static void put_u64(std::vector<uint8_t> &b, uint64_t v) { put(b, &v, 8); }
static void put_str(std::vector<uint8_t> &b, const std::string &s) { put_u64(b, s.size()); put(b, s.data(), s.size()); }

/* GGUF v3, 32-byte alignment; sets each tensor's absolute file offset. */
static void write_gguf(const std::string &path, std::vector<Tensor> &ts) {
    std::vector<uint8_t> h;
    put_u32(h, 0x46554747u); put_u32(h, 3); put_u64(h, ts.size()); put_u64(h, 1);
    put_str(h, "general.alignment"); put_u32(h, 4); put_u32(h, 32);
    uint64_t rel = 0;
    for (Tensor &t : ts) {
        put_str(h, t.name);
        if (t.type == 39) {                      /* [32, 64, 1]: 64 MXFP4 blocks */
            put_u32(h, 3); put_u64(h, 32); put_u64(h, 64); put_u64(h, 1);
        } else {
            put_u32(h, 2); put_u64(h, t.in); put_u64(h, t.out);
        }
        put_u32(h, t.type); put_u64(h, rel);
        t.off = rel;
        rel = (rel + t.data.size() + 31) & ~31ull;
    }
    const uint64_t data0 = (h.size() + 31) & ~31ull;
    h.resize(data0);
    for (Tensor &t : ts) {
        h.resize(data0 + t.off);
        put(h, t.data.data(), t.data.size());
        t.off += data0;
    }
    FILE *f = fopen(path.c_str(), "wb");
    CHECK(f && fwrite(h.data(), 1, h.size(), f) == h.size() && fclose(f) == 0);
}

enum { KV0, GATE0, KV1, GATE1, KV2, GATE2, KV3, GATE3, ODD_KV, ODD_GATE, LONE_KV, OTHER, F32_KV, Q8, MXFP4 };
static const uint32_t K = 4096, MAXM = 1024, MAXN = 6;

struct Pair { int a, b; uint64_t in, out; bool copied; };
static const Pair pairs[] = {
    {KV0, GATE0, K, 1024, true}, {KV1, GATE1, K, 512, true}, {KV2, GATE2, K, 256, true},
    {KV3, GATE3, K, 641, true},
    {ODD_KV, ODD_GATE, 1000, 64, false},          /* same names, K % 256 != 0 */
    {LONE_KV, OTHER, K, 128, false},              /* only one side copied */
};

struct Call { size_t pair; uint32_t n; uint32_t shift; };   /* shift: activation byte offset */
static std::vector<Call> calls() {
    std::vector<Call> c;
    for (size_t p = 0; p < sizeof(pairs) / sizeof(pairs[0]); p++)
        for (uint32_t n : {1u, 2u, MAXN})
            for (uint32_t shift : {0u, 4u}) c.push_back({p, n, shift});
    return c;
}
/* The one call shape that uses the copies. */
static bool uses_copy(const Call &c) { return pairs[c.pair].copied && c.n == 1 && c.shift == 0; }

struct Gpu {
    ds4_gpu_tensor *xbase, *o0, *o1;
};
static Gpu gpu_open(const void *map, uint64_t size, const std::vector<float> &x) {
    CHECK(ds4_gpu_init());
    CHECK(ds4_gpu_set_model_map(map, size));
    Gpu g = {ds4_gpu_tensor_alloc((MAXN * K + 4) * 4), ds4_gpu_tensor_alloc((MAXN * MAXM + 16) * 4),
             ds4_gpu_tensor_alloc((MAXN * MAXM + 16) * 4)};
    CHECK(g.xbase && g.o0 && g.o1);
    CHECK(ds4_gpu_tensor_write(g.xbase, 0, x.data(), x.size() * 4));
    return g;
}
static void gpu_close(Gpu &g) {
    ds4_gpu_tensor_free(g.o1);
    ds4_gpu_tensor_free(g.o0);
    ds4_gpu_tensor_free(g.xbase);
    ds4_gpu_cleanup();
}

/* Outputs of one call, both halves of the pair back to back. */
static std::vector<float> run(Gpu &g, const void *map, uint64_t size, const std::vector<Tensor> &ts, const Call &c) {
    const Pair &p = pairs[c.pair];
    const uint64_t live = (uint64_t)c.n * p.out;
    ds4_gpu_tensor *x = ds4_gpu_tensor_view(g.xbase, c.shift, (uint64_t)c.n * p.in * 4);
    CHECK(x);
    CHECK(ds4_gpu_tensor_fill_f32(g.o0, -9182.0f, MAXN * MAXM + 16));
    CHECK(ds4_gpu_tensor_fill_f32(g.o1, -9182.0f, MAXN * MAXM + 16));
    CHECK(ds4_gpu_matmul_f16_pair_tensor(g.o0, g.o1, map, size, ts[p.a].off, ts[p.b].off, p.in, p.out, x, c.n));
    ds4_gpu_tensor_free(x);
    std::vector<float> out(2 * live), tail(16);
    CHECK(ds4_gpu_tensor_read(g.o0, 0, out.data(), live * 4));
    CHECK(ds4_gpu_tensor_read(g.o1, 0, out.data() + live, live * 4));
    for (ds4_gpu_tensor *o : {g.o0, g.o1}) {
        CHECK(ds4_gpu_tensor_read(o, live * 4, tail.data(), tail.size() * 4));
        for (float v : tail) CHECK(v == -9182.0f);
    }
    return out;
}
static std::vector<std::vector<float>> run_all(Gpu &g, const void *map, uint64_t size, const std::vector<Tensor> &ts) {
    std::vector<std::vector<float>> r;
    for (const Call &c : calls()) r.push_back(run(g, map, size, ts, c));
    return r;
}
static bool same(const std::vector<float> &a, const std::vector<float> &b) {
    return a.size() == b.size() && !memcmp(a.data(), b.data(), a.size() * 4);
}

struct Map { void *p; uint64_t size; };
static Map map_file(const std::string &path) {
    const int fd = open(path.c_str(), O_RDONLY);
    CHECK(fd >= 0);
    struct stat st;
    CHECK(fstat(fd, &st) == 0);
    void *p = mmap(NULL, (size_t)st.st_size, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0);
    CHECK(p != MAP_FAILED);
    close(fd);
    return {p, (uint64_t)st.st_size};
}

static int build(const Map &m, const std::string &path, int rank) {
    return rank < 0 ? ds4_gpu_build_derived_artifacts(m.p, m.size, path.c_str())
                    : ds4_gpu_build_derived_artifacts_shard(m.p, m.size, m.size, path.c_str(), (uint32_t)rank);
}

static std::string fixture;
static void remove_fixture(void) {
    if (!fixture.empty()) unlink(fixture.c_str());
}

int main(int argc, char **argv) {
    const char *tmp = getenv("TMPDIR");
    const std::string dir = argc > 1 ? argv[1] : tmp && tmp[0] ? tmp : "/tmp";
    CHECK(ds4_gpu_init());
    const bool spark = ds4_gpu_device_is_spark();
    ds4_gpu_cleanup();
    if (!spark) {
        puts("CUDA F16 pair chunks: Spark required, SKIP");
        return 0;
    }
    atexit(remove_fixture);
    CHECK(setenv("DS4_CUDA_COPY_MODEL", "1", 1) == 0);
    for (const char *name : {"DS4_CUDA_NO_DERIVED_WEIGHTS", "DS4_CUDA_BUILD_ARTIFACTS", "DS4_CUDA_DECODE_GRAPHS",
                             "DS4_CUDA_NO_F16_PAIR_MATMUL", "DS4_CUDA_SERIAL_F16_MATMUL",
                             "DS4_CUDA_NO_ORDERED_F16_MATMUL"})
        CHECK(unsetenv(name) == 0);

    std::vector<Tensor> ts = {
        {"blk.0.attn_compressor_kv.weight", 1, K, 1024, {}, 0}, {"blk.0.attn_compressor_gate.weight", 1, K, 1024, {}, 0},
        {"blk.1.attn_compressor_kv.weight", 1, K, 512, {}, 0}, {"blk.1.attn_compressor_gate.weight", 1, K, 512, {}, 0},
        {"blk.2.indexer_compressor_kv.weight", 1, K, 256, {}, 0}, {"blk.2.indexer_compressor_gate.weight", 1, K, 256, {}, 0},
        {"blk.3.indexer_compressor_kv.weight", 1, K, 641, {}, 0}, {"blk.3.indexer_compressor_gate.weight", 1, K, 641, {}, 0},
        {"blk.4.attn_compressor_kv.weight", 1, 1000, 64, {}, 0}, {"blk.4.attn_compressor_gate.weight", 1, 1000, 64, {}, 0},
        {"blk.5.attn_compressor_kv.weight", 1, K, 128, {}, 0}, {"blk.5.attn_kv.weight", 1, K, 128, {}, 0},
        {"blk.6.attn_compressor_kv.weight", 0, K, 64, {}, 0}, {"blk.7.attn_q_a.weight", 8, 1024, 2048, {}, 0},
        {"blk.8.ffn_gate_exps.weight", 39, 0, 0, std::vector<uint8_t>(64 * 17, 0x11), 0},
    };
    for (Tensor &t : ts) {
        if (t.type == 1) t.data = f16_data(t.in, t.out);
        if (t.type == 0) t.data.assign(t.in * t.out * 4, 0);
        if (t.type == 8) t.data = q8_data(t.in, t.out);
    }
    const uint32_t copies = 9;                      /* four pairs and LONE_KV */
    std::vector<float> x(MAXN * K + 1);
    for (float &v : x) v = random_value();

    unsigned checked = 0;
    for (int mx = 0; mx < 2; mx++) {
        std::vector<Tensor> file(ts.begin(), ts.end() - (mx ? 0 : 1));
        std::string name = dir + "/ds4-f16-chunks-XXXXXX";
        const int fd = mkstemp(&name[0]);
        CHECK(fd >= 0 && close(fd) == 0);
        fixture = name;
        const std::string path = fixture;
        write_gguf(path, file);
        Map m = map_file(path);

        /* Raw weights, no copies. */
        Gpu g = gpu_open(m.p, m.size, x);
        const std::vector<std::vector<float>> raw = run_all(g, m.p, m.size, file);
        gpu_close(g);

        /* Builds: single GPU and both network-TP ranks, each built twice
         * (the second call keeps what exists), torn down and built again.
         * Q8_0 repacks only on single-GPU builds without MXFP4. */
        for (int rank = -1; rank < 2; rank++) {
            for (int again = 0; again < 2; again++) {
                Gpu b = gpu_open(m.p, m.size, x);
                const int expect = (int)copies + (rank < 0 && !mx ? 1 : 0);
                CHECK(build(m, path, rank) == expect);
                CHECK(build(m, path, rank) == expect);
                const std::vector<std::vector<float>> got = run_all(b, m.p, m.size, file);
                for (size_t i = 0; i < got.size(); i++) CHECK(same(got[i], raw[i]));
                checked += (unsigned)got.size();
                gpu_close(b);
            }
            printf("f16 pair chunks: %s %s: %d derived artifacts, outputs equal raw\n", mx ? "mxfp4" : "plain",
                   rank < 0 ? "single GPU" : rank == 0 ? "rank 0" : "rank 1", (int)copies + (rank < 0 && !mx ? 1 : 0));
        }

        /* Disabled: no copies are built. */
        CHECK(setenv("DS4_CUDA_NO_DERIVED_WEIGHTS", "1", 1) == 0);
        g = gpu_open(m.p, m.size, x);
        CHECK(build(m, path, -1) == 0);
        const std::vector<std::vector<float>> off = run_all(g, m.p, m.size, file);
        for (size_t i = 0; i < off.size(); i++) CHECK(same(off[i], raw[i]));
        gpu_close(g);
        CHECK(unsetenv("DS4_CUDA_NO_DERIVED_WEIGHTS") == 0);

        /* Which weights each call read: the mapped F16 weights change, the
         * file does not.  Raw results for the changed mapping first. */
        for (Tensor &t : file) {
            if (t.type != 1) continue;
            uint8_t *w = (uint8_t *)m.p + t.off;
            const std::vector<uint8_t> other = f16_data(t.in, t.out);
            memcpy(w, other.data(), other.size());
        }
        g = gpu_open(m.p, m.size, x);
        const std::vector<std::vector<float>> changed = run_all(g, m.p, m.size, file);
        gpu_close(g);
        g = gpu_open(m.p, m.size, x);
        CHECK(build(m, path, -1) == (int)copies + (mx ? 0 : 1));
        const std::vector<Call> cs = calls();
        const std::vector<std::vector<float>> got = run_all(g, m.p, m.size, file);
        for (size_t i = 0; i < cs.size(); i++) {
            CHECK(!same(raw[i], changed[i]));
            CHECK(same(got[i], uses_copy(cs[i]) ? raw[i] : changed[i]));
        }
        /* DS4_CUDA_NO_DERIVED_WEIGHTS also stops built copies being used. */
        CHECK(setenv("DS4_CUDA_NO_DERIVED_WEIGHTS", "1", 1) == 0);
        const std::vector<std::vector<float>> bypass = run_all(g, m.p, m.size, file);
        for (size_t i = 0; i < bypass.size(); i++) CHECK(same(bypass[i], changed[i]));
        CHECK(unsetenv("DS4_CUDA_NO_DERIVED_WEIGHTS") == 0);

        /* A decode graph captures the copy path: warm eager pass, capture,
         * then replays with new activations written into the same buffer. */
        CHECK(ds4_gpu_decode_graphs_supported());
        const Call first = {0, 1, 0};
        ds4_decode_graph_key key = {};
        key.il = 0;
        key.island = 0;
        key.cur_hc = g.xbase;
        std::vector<float> x2(x.size());
        for (float &v : x2) v = random_value();
        for (int round = 0; round < 4; round++) {
            if (round == 3) CHECK(ds4_gpu_tensor_write(g.xbase, 0, x2.data(), x2.size() * 4));
            CHECK(ds4_gpu_tensor_fill_f32(g.o0, -9182.0f, MAXN * MAXM + 16));
            CHECK(ds4_gpu_tensor_fill_f32(g.o1, -9182.0f, MAXN * MAXM + 16));
            CHECK(ds4_gpu_synchronize());
            const int state = ds4_gpu_decode_graph_begin(&key);
            CHECK(state == (round == 0 ? -1 : round == 1 ? 0 : 1));
            if (state != 1) {
                ds4_gpu_tensor *xv = ds4_gpu_tensor_view(g.xbase, 0, K * 4);
                CHECK(xv);
                CHECK(ds4_gpu_matmul_f16_pair_tensor(g.o0, g.o1, m.p, m.size, file[KV0].off, file[GATE0].off,
                                                     K, 1024, xv, 1));
                ds4_gpu_tensor_free(xv);
                if (state == 0) CHECK(ds4_gpu_decode_graph_end(&key) == 0);
            }
            CHECK(ds4_gpu_synchronize());
            std::vector<float> out(2 * 1024);
            CHECK(ds4_gpu_tensor_read(g.o0, 0, out.data(), 1024 * 4));
            CHECK(ds4_gpu_tensor_read(g.o1, 0, out.data() + 1024, 1024 * 4));
            if (round < 3) {
                CHECK(same(out, raw[0]));
            } else {
                /* The replay read the new input and equals the eager copy path. */
                CHECK(!same(out, raw[0]));
                ds4_gpu_decode_graphs_invalidate();
                CHECK(same(out, run(g, m.p, m.size, file, first)));
            }
        }
        ds4_gpu_decode_graphs_invalidate();
        gpu_close(g);
        CHECK(munmap(m.p, m.size) == 0);
        CHECK(unlink(path.c_str()) == 0);
        fixture.clear();
    }
    printf("CUDA F16 pair chunks: %u calls equal raw weights; copy dispatch, disable switch, "
           "fallbacks and graph replay PASS\n", checked);
    return 0;
}
