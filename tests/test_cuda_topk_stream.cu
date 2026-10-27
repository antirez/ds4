#include "ds4_gpu.h"
#include "ds4_deepseek41_gpu.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); exit(1); } } while (0)

static uint32_t state = 4711;
static uint32_t next_u32() {
    state = state * 1664525u + 1013904223u;
    return state;
}

static const uint32_t top_k = 512;
static const float sentinel = -9182.0f;

static float score(unsigned pattern, uint32_t t, uint32_t j, uint32_t visible) {
    float v = ((int)(next_u32() >> 8) - (1 << 23)) / 65536.0f;
    if (pattern == 1) v = (float)((int)(next_u32() >> 28) - 8);          /* heavy ties */
    if (pattern == 2 && j >= visible - visible / 3u + t) v = -INFINITY;  /* masked tail */
    if (pattern == 3 && t % 4u == 0u) v = -INFINITY;                    /* whole rows masked */
    if (pattern == 4) v = (next_u32() & 1u) ? 0.0f : -0.0f;             /* signed zeros */
    if (pattern == 5 && (next_u32() & 7u) == 0u) v = INFINITY;          /* +INF ties */
    return v;
}

/* Wide rows take the streaming top-512 kernel: Flash prefill (>= 32 rows,
 * more than 8192 keys, no causal bound) and V4.1 causal batches (rows past
 * 4096 visible keys).  Its threads must all agree on the candidate count when
 * they decide to compact; run under compute-sanitizer --tool synccheck to
 * catch a divergent barrier.  Every row is checked against the host order
 * (score descending, then index ascending) over its visible keys, a repeat
 * must match bit for bit, and nothing past the output may be written.
 * Causal rows see (start + row + 1) / ratio keys; the keys after that are
 * +INF so a row that reads past its bound picks them. */
static void check(bool causal, uint32_t rows, uint32_t width, uint32_t start, uint32_t ratio,
                  unsigned pattern) {
    std::vector<float> scores((size_t)rows * width);
    std::vector<uint32_t> visible(rows);
    for (uint32_t t = 0; t < rows; t++) {
        visible[t] = causal ? std::min(width, (start + t + 1u) / ratio) : width;
        for (uint32_t j = 0; j < width; j++)
            scores[(size_t)t * width + j] = j < visible[t] ? score(pattern, t, j, visible[t]) : INFINITY;
    }
    ds4_gpu_tensor *st = ds4_gpu_tensor_alloc(scores.size() * sizeof(float));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(((size_t)rows * top_k + 16) * sizeof(uint32_t));
    CHECK(st && out);
    CHECK(ds4_gpu_tensor_write(st, 0, scores.data(), scores.size() * sizeof(float)));
    ds4_gpu_tensor *view = ds4_gpu_tensor_view(out, 0, (size_t)rows * top_k * sizeof(uint32_t));
    CHECK(view);
    std::vector<uint32_t> first((size_t)rows * top_k + 16), again(first.size());
    for (std::vector<uint32_t> *result : {&first, &again}) {
        CHECK(ds4_gpu_tensor_fill_f32(out, sentinel, first.size()));
        CHECK(causal ? ds4_gpu_dsv41_indexer_topk_batch(view, st, width, rows, start, ratio)
                     : ds4_gpu_indexer_topk_tensor(view, st, width, rows, top_k));
        CHECK(ds4_gpu_synchronize());
        CHECK(ds4_gpu_tensor_read(out, 0, result->data(), result->size() * sizeof(uint32_t)));
    }
    CHECK(!memcmp(first.data(), again.data(), first.size() * sizeof(uint32_t)));
    if (!causal && rows == 1u && pattern == 0u &&
        (width == 513u || width == 2049u) && ds4_gpu_device_is_spark() &&
        getenv("DS4_CUDA_NO_TOPK2048") == NULL) {
        /* Exercise both compact sorts on the non-default capture stream. */
        CHECK(ds4_gpu_decode_graphs_supported());
        ds4_decode_graph_key key{};
        key.il = 61;
        key.island = 2;
        key.cur_hc = st;
        for (int round = 0; round < 4; round++) {
            CHECK(ds4_gpu_tensor_fill_f32(out, sentinel, first.size()));
            CHECK(ds4_gpu_synchronize());
            const int capture = ds4_gpu_decode_graph_begin(&key);
            CHECK(capture == (round == 0 ? -1 : round == 1 ? 0 : 1));
            if (capture != 1) {
                CHECK(ds4_gpu_indexer_topk_tensor(view, st, width, rows, top_k));
                if (capture == 0) CHECK(ds4_gpu_decode_graph_end(&key) == 0);
            }
            CHECK(ds4_gpu_synchronize());
            CHECK(ds4_gpu_tensor_read(out, 0, again.data(), again.size() * sizeof(uint32_t)));
            CHECK(!memcmp(first.data(), again.data(), first.size() * sizeof(uint32_t)));
        }
        ds4_gpu_decode_graphs_invalidate();
    }
    for (size_t i = (size_t)rows * top_k; i < first.size(); i++) CHECK(!memcmp(&first[i], &sentinel, 4));
    std::vector<uint32_t> order(width);
    for (uint32_t t = 0; t < rows; t++) {
        const float *row = &scores[(size_t)t * width];
        for (uint32_t j = 0; j < visible[t]; j++) order[j] = j;
        std::partial_sort(order.begin(), order.begin() + top_k, order.begin() + visible[t],
                          [&](uint32_t a, uint32_t b) {
            return row[a] > row[b] || (row[a] == row[b] && a < b);
        });
        if (memcmp(order.data(), &first[(size_t)t * top_k], top_k * sizeof(uint32_t))) {
            fprintf(stderr, "top-k mismatch causal=%d rows=%u width=%u start=%u ratio=%u pattern=%u row=%u\n",
                    causal, rows, width, start, ratio, pattern, t);
            CHECK(0);
        }
    }
    ds4_gpu_tensor_free(view);
    ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(st);
}

/* The V4.1 batch requires 1024 visible keys in its first row; the caller
 * picks rows one by one below that, so no row can be empty.  A refused
 * batch must leave the output untouched. */
static void check_refused(uint32_t start, uint32_t ratio) {
    const uint32_t rows = 4, width = 4096;
    ds4_gpu_tensor *st = ds4_gpu_tensor_alloc((size_t)rows * width * sizeof(float));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc((size_t)rows * top_k * sizeof(uint32_t));
    CHECK(st && out);
    CHECK(ds4_gpu_tensor_fill_f32(st, 1.0f, (uint64_t)rows * width));
    CHECK(ds4_gpu_tensor_fill_f32(out, sentinel, (uint64_t)rows * top_k));
    CHECK(!ds4_gpu_dsv41_indexer_topk_batch(out, st, width, rows, start, ratio));
    CHECK(ds4_gpu_synchronize());
    std::vector<float> got((size_t)rows * top_k);
    CHECK(ds4_gpu_tensor_read(out, 0, got.data(), got.size() * sizeof(float)));
    for (float v : got) CHECK(!memcmp(&v, &sentinel, 4));
    ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(st);
}

int main(int argc, char **argv) {
    /* --quick bounds racecheck; --compact skips the wide streaming cases. */
    bool quick = false, compact = false;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--quick")) quick = true;
        else if (!strcmp(argv[i], "--compact")) compact = true;
        else CHECK(0);
    }
    for (const char *name : {"DS4_CUDA_NO_TOPK1024", "DS4_CUDA_NO_TOPK2048",
                             "DS4_CUDA_NO_TOPK8192", "DS4_CUDA_NO_TOPK_STREAM",
                             "DS4_CUDA_DECODE_GRAPHS"})
        CHECK(unsetenv(name) == 0);
    CHECK(ds4_gpu_init());
    ds4_gpu_set_quality(false);
    /* First visible keys and rows: the gate minimum and 1024/2048 sort split,
     * sort-4096 batches followed by one row (single-row path), two rows or
     * many rows past 4096, and fully streamed batches. */
    const struct { uint32_t ratio, first, rows; } causal[] = {
        {1, 1024, 2}, {2, 1024, 3}, {1, 4066, 32}, {1, 4065, 64}, {2, 4081, 34}, {2, 4096, 33},
        {1, 4097, 2}, {1, 8193, 32}, {1, 8193, 64}, {1, 16385, 32}, {1, 16385, 64},
        {2, 8193, 64}, {2, 16385, 32},
    };
    unsigned cases = 0;
    for (unsigned pattern = 0; pattern < (quick ? 2u : 6u); pattern++) {
        /* Include both sides of every compact-sort dispatch boundary and
         * the unchanged wider sort, for decode, verification and prefill. */
        for (uint32_t keys : {512u, 513u, 700u, 1023u, 1024u, 1025u, 2047u,
                              2048u, 2049u, 3000u, 4095u, 4096u, 4097u, 8192u}) {
            for (uint32_t rows : {1u, 6u, 33u, 64u}) {
                if (quick && (rows > 6u || (keys != 513u && keys != 2049u))) continue;
                check(false, rows, keys, 0, 1, pattern);
                cases++;
            }
        }
        if (compact) continue;
        for (uint32_t keys : {8193u, 16385u}) {
            for (uint32_t rows : {32u, 64u}) {
                if (quick && (keys != 8193u || rows != 32u)) continue;
                check(false, rows, keys, 0, 1, pattern);
                cases++;
            }
        }
        for (const auto &c : causal) {
            if (quick && c.rows != 64u) continue;
            const uint32_t start = c.first * c.ratio - 1u;
            check(true, c.rows, (start + c.rows) / c.ratio + 512u, start, c.ratio, pattern);
            cases++;
        }
    }
    check_refused(1022, 1);
    check_refused(0, 2);
    check_refused(2046, 2);
    CHECK(setenv("DS4_CUDA_NO_TOPK2048", "1", 1) == 0);
    for (unsigned pattern : {0u, 1u}) {
        for (uint32_t keys : {513u, 2049u}) {
            for (uint32_t rows : {1u, 6u}) {
                check(false, rows, keys, 0, 1, pattern);
                cases++;
            }
        }
    }
    CHECK(unsetenv("DS4_CUDA_NO_TOPK2048") == 0);
    ds4_gpu_cleanup();
    printf("CUDA top-k: %u cases, host order, repeat, capture and tails PASS\n", cases);
    return 0;
}
