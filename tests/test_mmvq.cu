#include "ds4_mmq.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#ifndef GGML_USE_HIP
extern "C" int ds4_cuda_q8_fold_take_q81(
        const void *, uint64_t, const void **q81) {
    if (q81) *q81 = nullptr;
    return 0;
}
#endif

#define CHECK_CUDA(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        std::fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(err)); \
        std::exit(1); \
    } \
} while (0)

struct q8_block {
    uint16_t d;
    int8_t qs[32];
};
static_assert(sizeof(q8_block) == 34, "Q8_0 block size");

static bool test_shape(int rows, int cols, int tokens, bool routed) {
    const int experts = routed ? 8 : 1, used = routed ? 6 : 1;
    const size_t count = (size_t)tokens * used * rows;
    constexpr size_t guard = 32;
    constexpr float sentinel = -1234567.0f;
    std::vector<q8_block> weights((size_t)experts * rows * cols / 32);
    for (int e = 0; e < experts; ++e) {
        for (int r = 0; r < rows; ++r) {
            for (int b = 0; b < cols / 32; ++b) {
                auto &w = weights[((size_t)e * rows + r) * (cols / 32) + b];
                w.d = 0x3c00; // FP16 one: weights and activations are exact.
                for (int k = 0; k < 32; ++k) w.qs[k] = (e + r + b + k) % 7 - 3;
            }
        }
    }
    std::vector<float> x((size_t)tokens * cols);
    for (int t = 0; t < tokens; ++t)
        for (int k = 0; k < cols; ++k) x[(size_t)t * cols + k] = (k + t) % 2 ? 1 : -1;
    std::vector<int32_t> ids((size_t)tokens * used);
    for (int t = 0; t < tokens; ++t)
        for (int s = 0; s < used; ++s)
            ids[(size_t)t * used + s] = routed ? (s == 2 ? -1 : (7 + t + s) % experts) : 0;
    std::vector<float> out(count + 2 * guard, sentinel), expected(count, 0);
    for (int t = 0; t < tokens; ++t) {
        for (int s = 0; s < used; ++s) {
            int e = ids[(size_t)t * used + s];
            if (e < 0) continue;
            for (int r = 0; r < rows; ++r) {
                for (int k = 0; k < cols; ++k) {
                    expected[((size_t)t * used + s) * rows + r] +=
                        weights[((size_t)e * rows + r) * (cols / 32) + k / 32].qs[k % 32] *
                        x[(size_t)t * cols + k];
                }
            }
        }
    }
    q8_block *dw;
    float *dx, *dy;
    int32_t *di;
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));
    CHECK_CUDA(cudaMalloc(&dw, weights.size() * sizeof(q8_block)));
    CHECK_CUDA(cudaMalloc(&dx, x.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dy, out.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&di, ids.size() * sizeof(int32_t)));
    CHECK_CUDA(cudaMemcpyAsync(dw, weights.data(), weights.size() * sizeof(q8_block), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(dx, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(dy, out.data(), out.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(di, ids.data(), ids.size() * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    int rc = routed ? ds4_mmq_q8_0_moe_vec(dw, dx, di, dy + guard, rows, cols, tokens, experts, used, stream)
                    : ds4_mmq_q8_0_dense_vec(dw, dx, dy + guard, rows, tokens, cols, stream);
    CHECK_CUDA(cudaMemcpyAsync(out.data(), dy, out.size() * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    bool ok = rc == 0;
    for (size_t i = 0; i < guard; ++i)
        ok = ok && out[i] == sentinel && out[guard + count + i] == sentinel;
    for (size_t i = 0; i < count; ++i)
        ok = ok && std::isfinite(out[guard + i]) && std::fabs(out[guard + i] - expected[i]) < 0.05f;
    CHECK_CUDA(cudaFree(dw));
    CHECK_CUDA(cudaFree(dx));
    CHECK_CUDA(cudaFree(dy));
    CHECK_CUDA(cudaFree(di));
    CHECK_CUDA(cudaStreamDestroy(stream));
    std::fprintf(stderr, "MMVQ %s M=%d K=%d N=%d bounds/parity: %s\n",
                 routed ? "routed" : "dense", rows, cols, tokens, ok ? "PASS" : "FAIL");
    return ok;
}

int main() {
    if (ds4_mmq_init(0) != 0) return 1;
    bool ok = true;
    for (int rows : {1, 3, 17, 63, 64, 65})
        for (int cols : {256, 4096})
            for (int tokens : {1, 3, 8, 9})
                for (bool routed : {false, true}) {
                    if (!routed && tokens > 8) continue;
                    ok = test_shape(rows, cols, tokens, routed) && ok;
                }
    return ok ? 0 : 1;
}
