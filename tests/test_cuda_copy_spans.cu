/* ds4_gpu_tensor_copy_spans must leave every byte exactly as the same spans
 * copied one by one with ds4_gpu_tensor_copy, and must decline (copying
 * nothing) whatever it cannot batch.  Covers the DSpark frontier shapes of
 * DeepSeek V4 Flash (snapshot, restore and every prefix slot: 128 spans),
 * arbitrary sizes and unaligned offsets, empty spans, more spans than one
 * launch carries, untouched guard bytes around every destination, decline on
 * bad bounds, overlap, too many spans and a span over 4 GiB (disjoint views
 * of one 8 GiB allocation, only when that much memory is free), and
 * decode-graph capture/replay. */
#include "ds4_gpu.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); exit(1); } } while (0)
static uint32_t rng = 2213;
static const uint64_t GUARD = 256;

/* A tensor view with GUARD sentinel bytes on each side inside its base. */
struct guarded {
    ds4_gpu_tensor *base, *view;
    uint64_t bytes;
};
static std::vector<guarded> g_tensors;
static guarded make(uint64_t bytes) {
    guarded g;
    g.bytes = bytes;
    CHECK((g.base = ds4_gpu_tensor_alloc(bytes + 2 * GUARD)) && (g.view = ds4_gpu_tensor_view(g.base, GUARD, bytes)));
    g_tensors.push_back(g);
    return g;
}
static void fill_all(uint32_t seed) {
    rng = seed;
    for (const guarded &g : g_tensors) {
        std::vector<uint8_t> h(g.bytes + 2 * GUARD);
        for (uint8_t &x : h) { rng = rng * 1664525u + 1013904223u; x = (uint8_t)(rng >> 24); }
        CHECK(ds4_gpu_tensor_write(g.base, 0, h.data(), h.size()));
    }
}
static std::vector<std::vector<uint8_t>> read_all() {
    CHECK(ds4_gpu_synchronize());
    std::vector<std::vector<uint8_t>> out;
    for (const guarded &g : g_tensors) {
        std::vector<uint8_t> h(g.bytes + 2 * GUARD);
        CHECK(ds4_gpu_tensor_read(g.base, 0, h.data(), h.size()));
        out.push_back(h);
    }
    return out;
}
static void copy_one_by_one(const std::vector<ds4_gpu_copy_span> &v) {
    for (const ds4_gpu_copy_span &s : v)
        CHECK(ds4_gpu_tensor_copy(s.dst, s.dst_offset, s.src, s.src_offset, s.bytes));
}
/* The batched copy of v equals the one-by-one copy, every byte of every
 * tensor (guards included). */
static void same_as_one_by_one(const std::vector<ds4_gpu_copy_span> &v, const char *what) {
    fill_all(77);
    copy_one_by_one(v);
    const auto want = read_all();
    fill_all(77);
    CHECK(ds4_gpu_tensor_copy_spans(v.data(), (uint32_t)v.size()) == 1);
    CHECK(read_all() == want);
    printf("copy spans: %-28s %3zu spans: every byte equal to one-by-one copies\n", what, v.size());
}
/* Declined calls return 0 and copy nothing. */
static void declined(const std::vector<ds4_gpu_copy_span> &v, uint32_t n, const char *what) {
    fill_all(91);
    const auto before = read_all();
    CHECK(ds4_gpu_tensor_copy_spans(v.data(), n) == 0);
    CHECK(read_all() == before);
    printf("copy spans: declined %-28s nothing copied\n", what);
}

int main(int argc, char **argv) {
    CHECK(argc == 1);
    (void)argv;
    CHECK(setenv("DS4_CUDA_DECODE_GRAPHS", "1", 1) == 0);
    CHECK(ds4_gpu_init());
    guarded a = make(4096), b = make(4096);
    if (!ds4_gpu_device_is_spark()) {
        declined({{b.view, 0, a.view, 0, 4096}}, 1, "(not a Spark device)");
        for (const guarded &g : g_tensors) { ds4_gpu_tensor_free(g.view); ds4_gpu_tensor_free(g.base); }
        ds4_gpu_cleanup();
        printf("copy spans: not a Spark device, batched copies off: SKIP\n");
        return 0;
    }
    /* DeepSeek V4 Flash frontier: 21 ratio-4 layers (attention kv/score
     * 32 KiB, index kv/score 8 KiB), 22 ratio-128 layers (attention kv/score
     * 256 KiB); live, snapshot and five prefix slots. */
    std::vector<guarded> live, spec, slot;
    for (uint32_t il = 0; il < 43; il++) {
        const bool r4 = il % 2 == 0 && il / 2 < 21;
        std::vector<uint64_t> sizes = {r4 ? 32768u : 262144u, r4 ? 32768u : 262144u};
        if (r4) { sizes.push_back(8192); sizes.push_back(8192); }
        for (uint64_t bytes : sizes) { live.push_back(make(bytes)); spec.push_back(make(bytes)); slot.push_back(make(5 * bytes)); }
    }
    CHECK(live.size() == 128);
    std::vector<ds4_gpu_copy_span> snapshot, restore;
    for (size_t i = 0; i < live.size(); i++) {
        snapshot.push_back({spec[i].view, 0, live[i].view, 0, live[i].bytes});
        restore.push_back({live[i].view, 0, spec[i].view, 0, live[i].bytes});
    }
    same_as_one_by_one(snapshot, "frontier snapshot");
    same_as_one_by_one(restore, "frontier restore");
    for (uint32_t k = 0; k < 5; k++) {
        std::vector<ds4_gpu_copy_span> commit;
        for (size_t i = 0; i < live.size(); i++) commit.push_back({live[i].view, 0, slot[i].view, k * live[i].bytes, live[i].bytes});
        char what[32];
        snprintf(what, sizeof(what), "frontier commit slot %u", k);
        same_as_one_by_one(commit, what);
    }

    /* Arbitrary sizes and offsets (unaligned, tails), with empty spans. */
    guarded src = make(1u << 20), dst = make(1u << 20);
    const uint64_t sizes[] = {0, 1, 3, 15, 16, 17, 31, 32, 33, 4095, 4096, 4097, 32771, 65536, 65537};
    for (uint64_t so : {0u, 1u, 4u, 8u, 12u, 16u, 48u})
    for (uint64_t dof : {0u, 1u, 4u, 8u, 12u, 16u, 48u}) {
        std::vector<ds4_gpu_copy_span> v;
        uint64_t s0 = 0, d0 = 0;
        for (uint64_t sz : sizes) {
            v.push_back({dst.view, d0 + dof, src.view, s0 + so, sz});
            s0 += sz + 64 + so;
            d0 += sz + 64 + dof;
        }
        fill_all(5);
        copy_one_by_one(v);
        const auto want = read_all();
        fill_all(5);
        CHECK(ds4_gpu_tensor_copy_spans(v.data(), (uint32_t)v.size()) == 1);
        CHECK(read_all() == want);
    }
    printf("copy spans: sizes 0..65537 B at src/dst offsets 0..48: every byte equal (49 sets)\n");

    /* More spans than one launch carries: DS4_GPU_COPY_SPANS_MAX in 3. */
    std::vector<ds4_gpu_copy_span> many;
    for (uint32_t i = 0; i < DS4_GPU_COPY_SPANS_MAX; i++) {
        const uint64_t bytes = 256 + 8 * i + i % 5;          /* <= 2780: destinations 3200 apart stay disjoint */
        many.push_back({dst.view, (uint64_t)i * 3200, src.view, (uint64_t)i * 3100 + i % 7, bytes});
    }
    same_as_one_by_one(many, "maximum span count");

    /* Declines. */
    std::vector<ds4_gpu_copy_span> bad = snapshot;
    bad[37].bytes = live[37].bytes + 1;                          /* past the source */
    declined(bad, (uint32_t)bad.size(), "span past its tensor");
    bad = snapshot;
    bad[12].dst_offset = spec[12].bytes + 16;
    declined(bad, (uint32_t)bad.size(), "offset past its tensor");
    bad = snapshot;
    bad[5].dst = NULL;
    declined(bad, (uint32_t)bad.size(), "null tensor");
    bad = snapshot;
    bad.push_back({live[3].view, 0, live[4].view, 0, live[3].bytes < live[4].bytes ? live[3].bytes : live[4].bytes});
    bad.push_back({live[4].view, 0, live[5].view, 0, 64});    /* writes the source of the span before */
    declined(bad, (uint32_t)bad.size(), "destination over a source");
    bad = {{dst.view, 0, src.view, 0, 4096}, {dst.view, 2048, src.view, 8192, 4096}};
    declined(bad, 2, "overlapping destinations");
    declined(many, DS4_GPU_COPY_SPANS_MAX + 1u, "too many spans");
    declined(snapshot, 0, "no spans");
    {
        /* A span over 4 GiB between two disjoint views of one allocation, so
         * only the size rule can decline it.  Not part of the filled set;
         * exercised only when the memory is free. */
        const uint64_t span = (4ull << 30) + 16, half = span + 64;
        size_t free_bytes = 0, total_bytes = 0;
        ds4_gpu_tensor *big = NULL;
        if (cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess && free_bytes >= (16ull << 30))
            big = ds4_gpu_tensor_alloc(2 * half);
        if (big) {
            ds4_gpu_tensor *lo = ds4_gpu_tensor_view(big, 0, half), *hi = ds4_gpu_tensor_view(big, half, half);
            CHECK(lo && hi);
            const ds4_gpu_copy_span huge = {lo, 0, hi, 0, span};
            CHECK(ds4_gpu_tensor_copy_spans(&huge, 1) == 0);
            ds4_gpu_tensor_free(hi);
            ds4_gpu_tensor_free(lo);
            ds4_gpu_tensor_free(big);
            printf("copy spans: declined span over 4 GiB (disjoint views)\n");
        } else {
            (void)cudaGetLastError();
            printf("copy spans: SKIP over-4-GiB decline: less than 16 GiB free or 8 GiB allocation failed\n");
        }
    }

    /* Decode-graph capture and replay equal eager copies. */
    ds4_decode_graph_key key = {};
    key.il = 63; key.island = 2; key.variant = 128; key.cur_hc = live[0].view;   /* valid, unused island */
    unsigned captures = 0, replays = 0;
    for (unsigned round = 0; round < 6; round++) {
        fill_all(300 + round);
        copy_one_by_one(snapshot);
        const auto want = read_all();
        fill_all(300 + round);
        const int graph = ds4_gpu_decode_graph_begin(&key);
        CHECK(graph == (round == 0 ? -1 : round == 1 ? 0 : 1));
        if (graph == 1) {
            replays++;
        } else {
            CHECK(ds4_gpu_tensor_copy_spans(snapshot.data(), (uint32_t)snapshot.size()) == 1);
            if (graph == 0) { CHECK(ds4_gpu_decode_graph_end(&key) == 0); captures++; }
        }
        CHECK(read_all() == want);
    }
    CHECK(captures == 1 && replays == 4);
    ds4_gpu_decode_graphs_invalidate();
    printf("copy spans: frontier snapshot captured in a decode graph, 4 replays equal\n");

    for (const guarded &g : g_tensors) { ds4_gpu_tensor_free(g.view); ds4_gpu_tensor_free(g.base); }
    ds4_gpu_cleanup();
    printf("copy spans: PASS\n");
    return 0;
}
