#include "ds4_gpu.h"
#include <cuda_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); exit(1); } } while (0)

static uint32_t state = 4711;
static float random_value() {
    state = state * 1664525u + 1013904223u;
    return ((int)(state >> 16) - 32768) / 8192.0f;
}

/* Compare the default scorer with the established 64-key WMMA path. These
 * shapes cross both dispatch limits, warp tails and the causal frontier. */
static void check(uint32_t rows, uint32_t keys, uint32_t start, unsigned pattern) {
    const uint32_t heads = 64, dim = 128, ratio = 4;
    const float scale = 0.011048543f, sentinel = -9182.0f;
    const size_t count = (size_t)rows * keys;
    std::vector<float> q((size_t)rows * heads * dim), w((size_t)rows * heads);
    std::vector<float> k((size_t)keys * dim), result(count + 16), ref(count + 16);
    for (float &v : q) v = random_value() * (pattern == 1 ? 8.0f : 1.0f);
    for (float &v : w) v = random_value() * 0.25f;
    for (float &v : k) v = random_value() * (pattern == 2 ? 64.0f : 1.0f);
    ds4_gpu_tensor *qt = ds4_gpu_tensor_alloc(q.size() * sizeof(float));
    ds4_gpu_tensor *wt = ds4_gpu_tensor_alloc(w.size() * sizeof(float));
    ds4_gpu_tensor *kt = ds4_gpu_tensor_alloc(k.size() * sizeof(float));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(result.size() * sizeof(float));
    CHECK(qt && wt && kt && out);
    CHECK(ds4_gpu_tensor_write(qt, 0, q.data(), q.size() * sizeof(float)));
    CHECK(ds4_gpu_tensor_write(wt, 0, w.data(), w.size() * sizeof(float)));
    CHECK(ds4_gpu_tensor_write(kt, 0, k.data(), k.size() * sizeof(float)));
    for (unsigned arm = 0; arm < 3; arm++) {
        if (arm == 0) CHECK(setenv("DS4_CUDA_NO_INDEXER_WMMA128", "1", 1) == 0);
        else CHECK(unsetenv("DS4_CUDA_NO_INDEXER_WMMA128") == 0);
        CHECK(ds4_gpu_tensor_fill_f32(out, sentinel, result.size()));
        CHECK(ds4_gpu_indexer_scores_decode_batch_tensor(out, qt, wt, kt,
            keys, rows, start, heads, dim, ratio, scale));
        CHECK(ds4_gpu_tensor_read(out, 0, result.data(), result.size() * sizeof(float)));
        for (size_t i = count; i < result.size(); i++) CHECK(result[i] == sentinel);
        for (uint32_t t = 0; t < rows; t++) {
            const uint32_t visible = std::min((start + t + 1) / ratio, keys);
            for (uint32_t j = 0; j < keys; j++) {
                const float v = result[(size_t)t * keys + j];
                CHECK(!std::isnan(v));
                CHECK((j >= visible) == (std::isinf(v) && v < 0));
            }
        }
        if (arm == 0) ref = result;
        else CHECK(!memcmp(ref.data(), result.data(), result.size() * sizeof(float)));
    }
    /* Sparse double-precision oracle, with the same half-rounded operands
     * used by WMMA. The separate one-row kernel uses full-float operands. */
    for (uint32_t t : {0u, rows / 2, rows - 1}) {
        const uint32_t visible = std::min((start + t + 1) / ratio, keys);
        if (!visible) continue;
        for (uint32_t j : {0u, visible / 2, visible - 1}) {
            double sum = 0, magnitude = 0;
            for (uint32_t h = 0; h < heads; h++) {
                double dot = 0;
                for (uint32_t d = 0; d < dim; d++) {
                    float a = q[((size_t)t * heads + h) * dim + d];
                    float b = k[(size_t)j * dim + d];
                    if (rows != 1) {
                        a = __half2float(__float2half(a));
                        b = __half2float(__float2half(b));
                    }
                    dot += (double)a * b;
                }
                const double term = std::max(dot, 0.0) * w[(size_t)t * heads + h];
                sum += term;
                magnitude += std::fabs(term);
            }
            const double error = std::fabs(result[(size_t)t * keys + j] - sum * scale);
            CHECK(error < 2e-5 * (1.0 + magnitude * scale));
        }
    }
    ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(kt);
    ds4_gpu_tensor_free(wt);
    ds4_gpu_tensor_free(qt);
}

int main() {
    CHECK(ds4_gpu_init());
    ds4_gpu_set_quality(false);
    for (const char *name : {"DS4_CUDA_NO_INDEXER_WMMA", "DS4_CUDA_NO_INDEXER_WMMA64",
                            "DS4_CUDA_NO_INDEXER_DIRECT_ONE"}) CHECK(unsetenv(name) == 0);
    unsigned cases = 0;
    for (unsigned pattern = 0; pattern < 3; pattern++) {
        for (uint32_t keys : {1u, 15u, 16u, 17u, 513u, 1025u, 4096u, 4097u, 8195u}) {
            for (uint32_t rows : {1u, 2u, 6u, 8u, 16u, 17u}) {
                check(rows, keys, keys * 4 > rows ? keys * 4 - rows : 0, pattern);
                cases++;
            }
        }
        check(6, 1025, 0, pattern); /* Most warps contain only masked keys. */
        check(2, 1025, 0, pattern); /* Entire output is masked. */
        cases += 2;
    }
    ds4_gpu_cleanup();
    printf("CUDA indexer: %u cases, exact default/control, masks, tails and double oracle PASS\n", cases);
    return 0;
}
