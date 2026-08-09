// Regression for the aligned Q8_0 output-row slice entry used by CUDA
// network tensor parallelism.  A sliced launch must be bit-identical to the
// corresponding rows of a complete launch for both decode and small verify
// widths.

#include "ds4_mmq.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

// ds4_mmq.o optionally consumes producer-folded Q8_1 activations supplied by
// ds4_cuda.cu.  This standalone kernel test intentionally exercises the
// ordinary quantizer, so report that no folded activation is available.
extern "C" int ds4_cuda_q8_fold_take_q81(
        const void *src, uint64_t in_dim, const void **q81) {
    (void)src;
    (void)in_dim;
    (void)q81;
    return 0;
}

#define CUDA_CHECK(expr) do {                                             \
    const cudaError_t err_ = (expr);                                      \
    if (err_ != cudaSuccess) {                                            \
        fprintf(stderr, "%s:%d: %s: %s\n", __FILE__, __LINE__, #expr,   \
                cudaGetErrorString(err_));                                \
        return 1;                                                         \
    }                                                                     \
} while (0)

static int check_width(
        const void *artifact,
        const float *x,
        float *full,
        float *slice,
        int m_total,
        int row0,
        int rows,
        int n,
        int k,
        cudaStream_t stream) {
    if (ds4_mmq_q8_0_aligned_dense_vec(
                artifact, x, full, m_total, n, k, stream) != 0 ||
        ds4_mmq_q8_0_aligned_dense_vec_rows(
                artifact, x, slice, m_total, row0, rows, n, k, stream) != 0) {
        fprintf(stderr, "aligned Q8 row launch rejected N=%d\n", n);
        return 0;
    }
    if (cudaStreamSynchronize(stream) != cudaSuccess) return 0;

    std::vector<float> got_full((size_t)n * m_total);
    std::vector<float> got_slice((size_t)n * rows);
    if (cudaMemcpy(got_full.data(), full,
                   got_full.size() * sizeof(float),
                   cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(got_slice.data(), slice,
                   got_slice.size() * sizeof(float),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        return 0;
    }
    for (int col = 0; col < n; col++) {
        if (memcmp(got_full.data() + (size_t)col * m_total + row0,
                   got_slice.data() + (size_t)col * rows,
                   (size_t)rows * sizeof(float)) != 0) {
            fprintf(stderr,
                    "aligned Q8 row mismatch N=%d column=%d rows=%d:%d\n",
                    n, col, row0, row0 + rows - 1);
            return 0;
        }
    }
    return 1;
}

int main(void) {
    constexpr int m_total = 256;
    constexpr int row0 = 64;
    constexpr int rows = 64;
    constexpr int k = 1024;
    constexpr int max_n = 3;
    constexpr int blocks = k / 32;
    const uint64_t n_blocks = (uint64_t)m_total * blocks;
    const uint64_t dq_bytes = (n_blocks * 2u + 63u) & ~63ull;
    const uint64_t artifact_bytes =
        ds4_mmq_q8_0_aligned_bytes(m_total, k);
    if (artifact_bytes != dq_bytes + n_blocks * 32u) {
        fprintf(stderr, "aligned Q8 byte geometry mismatch\n");
        return 1;
    }

    std::mt19937 rng(0x5148524fu);
    std::vector<uint8_t> artifact((size_t)artifact_bytes, 0u);
    for (uint64_t block = 0; block < n_blocks; block++) {
        const __half scale = __float2half_rn(
            0.0025f + (float)(rng() % 2000u) / 100000.0f);
        memcpy(artifact.data() + block * 2u, &scale, sizeof(scale));
        int8_t *codes = reinterpret_cast<int8_t *>(
            artifact.data() + dq_bytes + block * 32u);
        for (int i = 0; i < 32; i++) {
            codes[i] = (int8_t)((int)(rng() % 255u) - 127);
        }
    }
    std::vector<float> input((size_t)max_n * k);
    for (float &value : input) {
        value = ((float)(rng() % 20001u) - 10000.0f) / 5000.0f;
    }

    if (ds4_mmq_init(0) != 0) {
        fprintf(stderr, "ds4_mmq_init failed\n");
        return 1;
    }
    void *device_artifact = nullptr;
    float *device_input = nullptr;
    float *device_full = nullptr;
    float *device_slice = nullptr;
    CUDA_CHECK(cudaMalloc(&device_artifact, artifact.size()));
    CUDA_CHECK(cudaMalloc(&device_input, input.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&device_full,
                          (size_t)max_n * m_total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&device_slice,
                          (size_t)max_n * rows * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(device_artifact, artifact.data(), artifact.size(),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_input, input.data(),
                          input.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));

    const int ok =
        check_width(device_artifact, device_input,
                    device_full, device_slice,
                    m_total, row0, rows, 1, k, stream) &&
        check_width(device_artifact, device_input,
                    device_full, device_slice,
                    m_total, row0, rows, max_n, k, stream) &&
        ds4_mmq_q8_0_aligned_dense_vec_rows(
            device_artifact, device_input, device_slice,
            m_total, m_total - rows + 1, rows, 1, k, stream) != 0;

    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(device_slice));
    CUDA_CHECK(cudaFree(device_full));
    CUDA_CHECK(cudaFree(device_input));
    CUDA_CHECK(cudaFree(device_artifact));
    if (!ok) return 1;
    puts("aligned Q8 output-row slices: bit-identical PASS (N=1,3)");
    return 0;
}
