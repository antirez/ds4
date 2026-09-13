#include "ds4_mmq.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CHECK(call) do { \
    cudaError_t error = (call); \
    if (error != cudaSuccess) { \
        fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(error)); \
        exit(1); \
    } \
} while (0)

static void *device_copy(const void *host, size_t bytes) {
    void *device;
    CHECK(cudaMalloc(&device, bytes));
    CHECK(cudaMemcpy(device, host, bytes, cudaMemcpyHostToDevice));
    return device;
}

static void run(int rows, int columns, int tokens, bool pair, cudaStream_t stream) {
    const size_t blocks = (size_t)rows * columns / 32;
    const size_t scales_bytes = (blocks * 2 + 63) & ~size_t(63);
    const size_t bytes = ds4_mmq_q8_0_aligned_bytes(rows, columns);
    assert(bytes == scales_bytes + blocks * 32);
    std::vector<unsigned char> weights(bytes);
    for (size_t i = 0; i < blocks; i++) {
        const __half scale = __float2half(0.001f * (1 + i % 13));
        memcpy(weights.data() + 2*i, &scale, 2);
    }
    for (size_t i = scales_bytes; i < bytes; i++) weights[i] = (unsigned char)(i * 37);
    std::vector<float> input((size_t)columns * tokens);
    for (size_t i = 0; i < input.size(); i++) input[i] = (int(i % 107) - 53) * 0.031f;
    void *w0 = device_copy(weights.data(), bytes);
    for (size_t i = scales_bytes; i < bytes; i++) weights[i] ^= 0x35;
    void *w1 = device_copy(weights.data(), bytes);
    float *x = (float *)device_copy(input.data(), input.size() * sizeof(float));
    const size_t count = (size_t)rows * tokens * (pair ? 2 : 1);
    float *out;
    CHECK(cudaMalloc(&out, count * sizeof(float)));
    void *scratch;
    CHECK(cudaMalloc(&scratch, 64 * 1024));
    auto launch = [&]() {
        const int rc = pair
            ? ds4_mmq_q8_0_aligned_dense_vec_pair(w0, w1, x, out, out + rows,
                                                 rows, rows, columns, stream)
            : ds4_mmq_q8_0_aligned_dense_vec(w0, x, out, rows, tokens, columns, stream);
        assert(rc == 0);
    };
    auto read = [&]() {
        CHECK(cudaStreamSynchronize(stream));
        std::vector<float> result(count);
        CHECK(cudaMemcpy(result.data(), out, count * sizeof(float), cudaMemcpyDeviceToHost));
        return result;
    };
    ds4_mmq_set_aligned_q81_scratch(nullptr, 0);
    launch();
    const auto reference = read();
    if (!pair && tokens > 1) {
        for (int token = 0; token < tokens; token++) {
            assert(ds4_mmq_q8_0_aligned_dense_vec(
                w0, x + token * columns, out + token * rows,
                rows, 1, columns, stream) == 0);
        }
        const auto separate = read();
        assert(memcmp(reference.data(), separate.data(), count * sizeof(float)) == 0);
    }
    for (size_t capacity : {size_t(1), size_t(64 * 1024)}) {
        ds4_mmq_set_aligned_q81_scratch(scratch, capacity);
        CHECK(cudaMemsetAsync(out, 0x7f, count * sizeof(float), stream));
        launch();
        const auto result = read();
        assert(memcmp(reference.data(), result.data(), count * sizeof(float)) == 0);
    }
    cudaGraph_t graph;
    cudaGraphExec_t executable;
    CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    launch();
    CHECK(cudaStreamEndCapture(stream, &graph));
    CHECK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
    for (int replay = 0; replay < 3; replay++) {
        for (float &value : input) value += 0.013f * (replay + 1);
        CHECK(cudaMemcpyAsync(x, input.data(), input.size() * sizeof(float),
                              cudaMemcpyHostToDevice, stream));
        CHECK(cudaGraphLaunch(executable, stream));
        const auto captured = read();
        ds4_mmq_set_aligned_q81_scratch(nullptr, 0);
        launch();
        const auto eager = read();
        assert(memcmp(captured.data(), eager.data(), count * sizeof(float)) == 0);
        ds4_mmq_set_aligned_q81_scratch(scratch, 64 * 1024);
    }
    CHECK(cudaGraphExecDestroy(executable));
    CHECK(cudaGraphDestroy(graph));
    ds4_mmq_set_aligned_q81_scratch(nullptr, 0);
    CHECK(cudaFree(scratch));
    CHECK(cudaFree(out));
    CHECK(cudaFree(x));
    CHECK(cudaFree(w1));
    CHECK(cudaFree(w0));
    printf("PASS Q8 scratch rows=%d K=%d tokens=%d pair=%d\n", rows, columns, tokens, pair);
}

int main() {
    assert(ds4_mmq_init(0) == 0);
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));
    for (int rows : {128, 129, 2048}) {
        for (int columns : {1024, 4096}) {
            for (int tokens = 1; tokens <= 8; tokens++)
                run(rows, columns, tokens, false, stream);
            run(rows, columns, 1, true, stream);
        }
    }
    CHECK(cudaStreamDestroy(stream));
    puts("CUDA Q8 scratch: PASS");
    return 0;
}
