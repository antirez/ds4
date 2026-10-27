#include "ds4_gpu.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

static uint32_t seed = 419;
static float random_value() {
    seed = seed * 1664525u + 1013904223u;
    return ((int)(seed >> 16) - 32768) / 8192.0f;
}

static void check(unsigned width, unsigned rows) {
    const size_t n = (size_t)width * rows;
    std::vector<float> weights(width), input(n * 4), block(n), split(rows * 24);
    for (float &v : weights) v = random_value();
    for (float &v : input) v = random_value();
    for (float &v : block) v = random_value();
    for (float &v : split) v = random_value();
    assert(ds4_gpu_init());
    assert(ds4_gpu_set_model_map(weights.data(), weights.size() * 4));
    auto *x = ds4_gpu_tensor_alloc(input.size() * 4);
    auto *b = ds4_gpu_tensor_alloc(block.size() * 4);
    auto *s = ds4_gpu_tensor_alloc(split.size() * 4);
    auto *old = ds4_gpu_tensor_alloc((n * 4 + 8) * 4);
    auto *fast = ds4_gpu_tensor_alloc((n * 4 + 8) * 4);
    assert(x && b && s && old && fast);
    assert(ds4_gpu_tensor_write(x, 0, input.data(), input.size() * 4));
    assert(ds4_gpu_tensor_write(b, 0, block.data(), block.size() * 4));
    assert(ds4_gpu_tensor_write(s, 0, split.data(), split.size() * 4));
    auto compare = [&](size_t count) {
        std::vector<float> a(n * 4 + 8), c(a.size());
        assert(ds4_gpu_tensor_read(old, 0, a.data(), a.size() * 4));
        assert(ds4_gpu_tensor_read(fast, 0, c.data(), c.size() * 4));
        if (memcmp(a.data(), c.data(), count * 4)) {
            for (size_t i = 0; i < count; i++) if (memcmp(&a[i], &c[i], 4)) {
                fprintf(stderr, "width=%u rows=%u index=%zu: %.9g != %.9g\n",
                    width, rows, i, a[i], c[i]);
                break;
            }
            assert(false);
        }
        for (size_t i = count; i < c.size(); i++) assert(std::isnan(c[i]));
    };
    auto clear = [&]() {
        assert(ds4_gpu_tensor_fill_f32(old, NAN, n * 4 + 8));
        assert(ds4_gpu_tensor_fill_f32(fast, NAN, n * 4 + 8));
    };
    for (float eps : {1e-6f, 1e-20f}) {
        clear();
        assert(ds4_gpu_rms_norm_weight_rows_tensor(old, x, weights.data(), weights.size() * 4,
            0, width, rows, eps));
        assert(ds4_gpu_dsv41_quantize(old, width, rows, DS4_V41_BF16));
        assert(ds4_gpu_dsv41_norm(fast, x, weights.data(), weights.size() * 4,
            0, width, rows, eps));
        compare(n);
        assert(ds4_gpu_tensor_write(fast, 0, input.data(), n * 4));
        assert(ds4_gpu_dsv41_norm(fast, fast, weights.data(), weights.size() * 4,
            0, width, rows, eps));
        compare(n);
    }
    for (unsigned stride : {4u, 24u}) {
        clear();
        auto *bounded = ds4_gpu_tensor_view(old, 0, n * 4);
        assert(bounded);
        assert(stride == 4 ? ds4_gpu_hc_weighted_sum_tensor(bounded, x, s, width, 4) :
            ds4_gpu_hc_weighted_sum_split_tensor(bounded, x, s, width, 4));
        ds4_gpu_tensor_free(bounded);
        assert(ds4_gpu_dsv41_quantize(old, width, rows, DS4_V41_BF16));
        assert(ds4_gpu_dsv41_hc_sum(fast, x, s, width, rows, stride));
        compare(n);
    }
    clear();
    auto *bounded = ds4_gpu_tensor_view(old, 0, n * 4 * 4);
    assert(bounded);
    assert(ds4_gpu_hc_expand_split_tensor(bounded, b, x, s, width, 4));
    ds4_gpu_tensor_free(bounded);
    assert(ds4_gpu_dsv41_quantize(old, width * 4, rows, DS4_V41_BF16));
    assert(ds4_gpu_dsv41_hc_expand(fast, b, x, s, width, rows));
    compare(n * 4);
    assert(!ds4_gpu_dsv41_norm(fast, x, weights.data(), weights.size() * 4,
        weights.size() * 4, width, rows, 1e-6f));
    assert(!ds4_gpu_dsv41_hc_sum(fast, x, s, width, rows, 5));
    assert(!ds4_gpu_dsv41_hc_expand(fast, b, x, s, width, rows + 1));
    ds4_gpu_tensor_free(fast); ds4_gpu_tensor_free(old); ds4_gpu_tensor_free(s);
    ds4_gpu_tensor_free(b); ds4_gpu_tensor_free(x);
    ds4_gpu_cleanup();
    printf("V4.1 BF16 fusion width=%u rows=%u: exact norm/HC, aliases and guards PASS\n", width, rows);
}

int main() {
    for (unsigned width : {31u, 32u, 127u, 512u, 1280u, 5120u, 20480u})
        for (unsigned rows : {1u, 3u, 129u}) check(width, rows);
}
