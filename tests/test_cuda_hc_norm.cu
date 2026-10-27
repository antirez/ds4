/* Short-row HC fusion must preserve both intermediates and normalized output. */
#include "ds4_gpu.h"
#include <cuda_runtime.h>
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

static void compare(ds4_gpu_tensor *a, ds4_gpu_tensor *b, const char *name,
                    uint32_t width, uint32_t rows, unsigned pattern) {
    CHECK(ds4_gpu_tensor_bytes(a) == ds4_gpu_tensor_bytes(b));
    std::vector<float> x(ds4_gpu_tensor_bytes(a) / sizeof(float)), y(x.size());
    CHECK(ds4_gpu_tensor_read(a, 0, x.data(), x.size() * sizeof(float)));
    CHECK(ds4_gpu_tensor_read(b, 0, y.data(), y.size() * sizeof(float)));
    for (size_t i = 0; i < x.size(); ++i) {
        if (memcmp(&x[i], &y[i], sizeof(float))) {
            fprintf(stderr, "%s width=%u rows=%u pattern=%u index=%zu: %.9g != %.9g\n",
                    name, width, rows, pattern, i, x[i], y[i]);
            exit(1);
        }
        const size_t live = (size_t)rows * (!strcmp(name, "split") ? 24u : width);
        if (i >= live) CHECK(x[i] == -9182.0f);
    }
}

int main(int argc, char **argv) {
    CHECK(argc == 1 || (argc == 2 && !strcmp(argv[1], "--bench")));
    const bool bench = argc == 2;
    const uint32_t max_width = 16384;
    std::vector<float> weights(3 + 24 + max_width);
    weights[0] = 0.25f; weights[1] = 0.75f; weights[2] = 1.5f;
    for (size_t i = 3; i < weights.size(); ++i) weights[i] = random_value();
    const uint64_t size = weights.size() * sizeof(float), norm_off = 27 * sizeof(float);
    /* This fixture's model is only 66 KiB. Avoid unsupported host-registration
     * probes on Spark, so sanitizer API-error reporting remains useful. */
    CHECK(setenv("DS4_CUDA_COPY_MODEL", "1", 0) == 0);
    CHECK(ds4_gpu_init() && ds4_gpu_set_model_map(weights.data(), size));
    unsigned cases = 0;
    for (uint32_t width : {1u, 31u, 32u, 33u, 255u, 256u, 257u, 4095u, 4096u, 5120u, max_width}) {
        for (uint32_t rows : {1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 16u}) {
            if (bench && (width != 4096 || rows > 6 || rows < 2)) continue;
            const uint64_t bytes = (uint64_t)width * rows * sizeof(float);
            ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(4 * bytes);
            ds4_gpu_tensor *mix = ds4_gpu_tensor_alloc(rows * 24 * sizeof(float));
            ds4_gpu_tensor *a = ds4_gpu_tensor_alloc(bytes + 32);
            ds4_gpu_tensor *b = ds4_gpu_tensor_alloc(bytes + 32);
            ds4_gpu_tensor *an = ds4_gpu_tensor_alloc(bytes + 32);
            ds4_gpu_tensor *bn = ds4_gpu_tensor_alloc(bytes + 32);
            ds4_gpu_tensor *as = ds4_gpu_tensor_alloc(rows * 24 * sizeof(float) + 32);
            ds4_gpu_tensor *bs = ds4_gpu_tensor_alloc(rows * 24 * sizeof(float) + 32);
            CHECK(x && mix && a && b && an && bn && as && bs);
            ds4_gpu_tensor *av = ds4_gpu_tensor_view(a, 0, bytes);
            ds4_gpu_tensor *bv = ds4_gpu_tensor_view(b, 0, bytes);
            ds4_gpu_tensor *anv = ds4_gpu_tensor_view(an, 0, bytes);
            ds4_gpu_tensor *bnv = ds4_gpu_tensor_view(bn, 0, bytes);
            CHECK(av && bv && anv && bnv);
            std::vector<float> input(4 * bytes / sizeof(float)), mixing(rows * 24);
            for (unsigned pattern = 0; pattern < (bench ? 1u : 3u); ++pattern) {
                for (size_t i = 0; i < input.size(); ++i)
                    input[i] = pattern == 0 ? random_value() : pattern == 1 ? 0.0f :
                        ldexpf(random_value(), (int)(i % 41u) - 20);
                for (float &v : mixing) v = random_value() * (pattern == 2 ? 15.0f : 1.0f);
                CHECK(ds4_gpu_tensor_write(x, 0, input.data(), 4 * bytes));
                CHECK(ds4_gpu_tensor_write(mix, 0, mixing.data(), mixing.size() * sizeof(float)));
                for (ds4_gpu_tensor *t : {a, b, an, bn, as, bs})
                    CHECK(ds4_gpu_tensor_fill_f32(t, -9182.0f, ds4_gpu_tensor_bytes(t) / sizeof(float)));
                CHECK(ds4_gpu_hc_split_weighted_sum_tensor(av, as, mix, x,
                    weights.data(), size, 0, 3 * sizeof(float), width, 4, 20, 1e-6f));
                CHECK(ds4_gpu_rms_norm_weight_rows_tensor(anv, av, weights.data(),
                    size, norm_off, width, rows, 1e-6f));
                CHECK(ds4_gpu_hc_split_weighted_sum_norm_tensor(bv, bnv, bs, mix, x,
                    weights.data(), size, 0, 3 * sizeof(float), norm_off, width, 4, 20, 1e-6f, 1e-6f));
                compare(a, b, "weighted", width, rows, pattern);
                compare(an, bn, "norm", width, rows, pattern);
                compare(as, bs, "split", width, rows, pattern);
                CHECK(ds4_gpu_hc_split_weighted_sum_norm_tensor(bv, bv, bs, mix, x,
                    weights.data(), size, 0, 3 * sizeof(float), norm_off, width, 4, 20, 1e-6f, 1e-6f));
                compare(an, b, "inplace", width, rows, pattern);
                ++cases;
            }
            if (bench) {
                cudaEvent_t start, stop;
                CHECK(cudaEventCreate(&start) == cudaSuccess && cudaEventCreate(&stop) == cudaSuccess);
                for (unsigned rep = 0; rep < 8; ++rep) {
                    const bool fused = rep % 4 == 1 || rep % 4 == 2;
                    CHECK(cudaEventRecord(start) == cudaSuccess);
                    for (unsigned i = 0; i < 1000; ++i) {
                        if (fused) {
                            CHECK(ds4_gpu_hc_split_weighted_sum_norm_tensor(bv, bnv, bs, mix, x,
                                weights.data(), size, 0, 3 * sizeof(float), norm_off, width, 4, 20, 1e-6f, 1e-6f));
                        } else {
                            CHECK(ds4_gpu_hc_split_weighted_sum_tensor(bv, bs, mix, x,
                                weights.data(), size, 0, 3 * sizeof(float), width, 4, 20, 1e-6f));
                            CHECK(ds4_gpu_rms_norm_weight_rows_tensor(bnv, bv, weights.data(),
                                size, norm_off, width, rows, 1e-6f));
                        }
                    }
                    CHECK(cudaEventRecord(stop) == cudaSuccess && cudaEventSynchronize(stop) == cudaSuccess);
                    float ms = 0;
                    CHECK(cudaEventElapsedTime(&ms, start, stop) == cudaSuccess);
                    printf("rows=%u fused=%d us=%.3f\n", rows, fused, ms);
                }
                CHECK(cudaEventDestroy(start) == cudaSuccess && cudaEventDestroy(stop) == cudaSuccess);
            }
            for (ds4_gpu_tensor *t : {av, bv, anv, bnv, x, mix, a, b, an, bn, as, bs}) ds4_gpu_tensor_free(t);
        }
    }
    ds4_gpu_cleanup();
    printf("HC batch exactness, tails and inplace: %u cases PASS\n", cases);
    return 0;
}
