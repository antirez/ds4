/* Dense Q8_0 through ds4_mmq_q8_0_dense must equal mul_mat_q on the same
 * quantized activations (ds4_mmq_q8_0_dense_preq) in every bit: random,
 * NaN/Inf and huge activations, zero/Inf/NaN weight scales; the exact GEMV
 * shapes (q_b: K 1024, M 16384; q_a and kv: K 4096, M 1024 and 512; 2..8
 * columns; vocabulary head: K 4096, M 64640, 5/6/8 columns) and shapes
 * they decline (1 and 9 columns, other and partial row
 * counts); on a non-blocking stream behind the activation producer; and
 * captured in a CUDA graph replayed with new activations.  On Spark the
 * graph of an exact shape must hold two kernels (quantize and the GEMV), not
 * the mul_mat_q chain. */
#include "ds4_gpu.h"
#include "cuda/mmq/ds4_mmq.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); exit(1); } } while (0)
#define CUDA_CHECK(x) CHECK((x) == cudaSuccess)

static uint32_t rng = 2654435761u;
static uint32_t rnd() { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng; }
static float uniform(float lo, float hi) { return lo + (hi - lo) * (float)(rnd() >> 8) / 16777216.0f; }

static const int MAX_N = 9;
static const size_t Y_MAX = (size_t)MAX_N * 4096 / 128 * 144 + 128 * 144;

/* Q8_0 weights: random int8, realistic scales, some zero / Inf / NaN scales. */
static uint8_t *make_weights(int M, int K) {
    const size_t row = (size_t)K / 32 * 34, bytes = (size_t)M * row;
    std::vector<uint8_t> h(bytes);
    for (size_t r = 0; r < (size_t)M; r++)
        for (int b = 0; b < K / 32; b++) {
            uint8_t *blk = &h[r * row + (size_t)b * 34];
            float d = uniform(1e-4f, 1e-2f);
            if (r % 4099 == 7 && b == 3) d = 0.0f;
            if (r % 8191 == 11 && b == 5) d = INFINITY;
            if (r % 8193 == 13 && b == K / 32 - 2) d = NAN;
            const __half hd = __float2half(d);
            memcpy(blk, &hd, 2);
            for (int i = 0; i < 32; i++) blk[2 + i] = (uint8_t)(int8_t)((int)(rnd() % 255) - 127);
        }
    uint8_t *w = NULL;
    CUDA_CHECK(cudaMalloc(&w, bytes));
    CUDA_CHECK(cudaMemcpy(w, h.data(), bytes, cudaMemcpyHostToDevice));
    return w;
}
static std::vector<float> activations(int N, int K, int round) {
    std::vector<float> x((size_t)N * K);
    for (float &v : x) v = uniform(-3.0f, 3.0f) * (round == 2 ? 1e3f : 1.0f);
    if (round == 1) { x[17] = NAN; x[(size_t)(N - 1) * K + K / 2] = INFINITY; }
    return x;
}

struct bufs { float *x, *ref, *out; uint8_t *y; };

/* mul_mat_q on the quantized activations of the current x. */
static std::vector<float> reference(const bufs &b, const uint8_t *w, int M, int N, int K) {
    CUDA_CHECK(cudaDeviceSynchronize());
    CHECK(ds4_mmq_q8_0_quantize_ref(b.x, b.y, Y_MAX, N, K, 0) == 0);
    CHECK(ds4_mmq_q8_0_dense_preq(w, b.y, Y_MAX, b.ref, M, N, K, 0) == 0);
    std::vector<float> r((size_t)M * N);
    CUDA_CHECK(cudaMemcpy(r.data(), b.ref, r.size() * sizeof(float), cudaMemcpyDeviceToHost));
    return r;
}
static std::vector<float> result(const bufs &b, int M, int N) {
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> r((size_t)M * N);
    CUDA_CHECK(cudaMemcpy(r.data(), b.out, r.size() * sizeof(float), cudaMemcpyDeviceToHost));
    return r;
}
static void same(const std::vector<float> &want, const std::vector<float> &got, const char *what, int M, int N, int K) {
    size_t diff = 0, first = 0;
    for (size_t i = 0; i < want.size(); i++)
        if (memcmp(&want[i], &got[i], 4)) { if (!diff++) first = i; }
    if (diff) {
        fprintf(stderr, "q8_0 gemv: %s M %d N %d K %d: %zu of %zu differ, first col %zu row %zu: %.9g vs %.9g\n",
                what, M, N, K, diff, want.size(), first / M, first % M, want[first], got[first]);
        exit(1);
    }
}

int main(int argc, char **argv) {
    (void)argv;
    CHECK(argc == 1);
    CHECK(ds4_gpu_init());
    int dev = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    const bool spark = prop.major * 100 + prop.minor * 10 == 1210;
    bufs b;
    CUDA_CHECK(cudaMalloc(&b.x, (size_t)MAX_N * 4096 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b.ref, (size_t)MAX_N * 64640 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b.out, (size_t)MAX_N * 64640 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b.y, Y_MAX));

    /* The exact shapes (q_b, q_a, kv, head) and shapes they decline, eager on the
     * legacy stream. */
    struct shape { int M, K; };
    const shape admitted[] = {{16384, 1024}, {1024, 4096}, {512, 4096}, {64640, 4096}};
    const shape shapes[] = {{16384, 1024}, {1024, 4096}, {512, 4096},
                            {16256, 1024}, {16320, 1024}, {32768, 1024}, {2048, 4096}, {16384, 4096},
                            {64640, 4096}, {64512, 4096}, {64639, 4096}};
    unsigned checked = 0;
    for (const shape &s : shapes) {
        uint8_t *w = make_weights(s.M, s.K);
        const bool exact_shape = (s.M == 16384 && s.K == 1024) ||
            (s.K == 4096 && (s.M == 1024 || s.M == 512 || s.M == 64640));
        for (int N = 1; N <= MAX_N; N++) {
            if (!exact_shape && N != 6) continue;
            for (int round = 0; round < 3; round++) {
                const std::vector<float> x = activations(N, s.K, round);
                CUDA_CHECK(cudaMemcpy(b.x, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice));
                const std::vector<float> want = reference(b, w, s.M, N, s.K);
                CUDA_CHECK(cudaMemset(b.out, 0xa5, (size_t)s.M * N * sizeof(float)));
                CHECK(ds4_mmq_q8_0_dense(w, b.x, b.out, s.M, N, s.K, 0) == 0);
                same(want, result(b, s.M, N), "eager", s.M, N, s.K);
                checked++;
            }
        }
        CUDA_CHECK(cudaFree(w));
    }
    printf("q8_0 gemv: %u eager cases equal to mul_mat_q in every bit, "
           "including vocabulary head and declined shapes\n", checked);

    cudaStream_t s;
    CUDA_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    for (const shape &a : admitted) {
        uint8_t *w = make_weights(a.M, a.K);
        const int M = a.M, K = a.K;
        /* Non-blocking stream: the activations a copy on the same stream writes
         * behind a long copy. */
        {
            const size_t big = 256u << 20;
            char *src = NULL, *dst = NULL;
            float *stage = NULL;
            CUDA_CHECK(cudaMalloc(&src, big));
            CUDA_CHECK(cudaMalloc(&dst, big));
            CUDA_CHECK(cudaMemset(src, 0x3c, big));
            CUDA_CHECK(cudaMalloc(&stage, (size_t)MAX_N * K * sizeof(float)));
            for (int N = 2; N <= 8; N++) {
                const std::vector<float> fresh = activations(N, K, N % 3), old((size_t)N * K, 1.0f);
                CUDA_CHECK(cudaMemcpy(b.x, fresh.data(), fresh.size() * sizeof(float), cudaMemcpyHostToDevice));
                const std::vector<float> want = reference(b, w, M, N, K);
                CUDA_CHECK(cudaMemcpy(stage, fresh.data(), fresh.size() * sizeof(float), cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(b.x, old.data(), old.size() * sizeof(float), cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemset(b.out, 0xa5, (size_t)M * N * sizeof(float)));
                CUDA_CHECK(cudaDeviceSynchronize());
                CUDA_CHECK(cudaMemcpyAsync(dst, src, big, cudaMemcpyDeviceToDevice, s));
                CUDA_CHECK(cudaMemcpyAsync(b.x, stage, fresh.size() * sizeof(float), cudaMemcpyDeviceToDevice, s));
                CHECK(ds4_mmq_q8_0_dense(w, b.x, b.out, M, N, K, s) == 0);
                CUDA_CHECK(cudaStreamSynchronize(s));
                same(want, result(b, M, N), "non-blocking stream", M, N, K);
            }
            CUDA_CHECK(cudaFree(stage)); CUDA_CHECK(cudaFree(dst)); CUDA_CHECK(cudaFree(src));
        }
        printf("q8_0 gemv: M %d K %d, N 2..8 on a non-blocking stream behind the activation producer: equal\n", M, K);

        /* CUDA graph: captured once, replayed with new activations. */
        for (int N = 2; N <= 8; N++) {
            const std::vector<float> warm = activations(N, K, 0);
            CUDA_CHECK(cudaMemcpy(b.x, warm.data(), warm.size() * sizeof(float), cudaMemcpyHostToDevice));
            CHECK(ds4_mmq_q8_0_dense(w, b.x, b.out, M, N, K, s) == 0);      /* warm: pools, occupancy */
            CUDA_CHECK(cudaStreamSynchronize(s));
            cudaGraph_t graph;
            cudaGraphExec_t exec;
            CUDA_CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
            CHECK(ds4_mmq_q8_0_dense(w, b.x, b.out, M, N, K, s) == 0);
            CUDA_CHECK(cudaStreamEndCapture(s, &graph));
            size_t nodes = 0;
            CUDA_CHECK(cudaGraphGetNodes(graph, NULL, &nodes));
            std::vector<cudaGraphNode_t> list(nodes);
            CUDA_CHECK(cudaGraphGetNodes(graph, list.data(), &nodes));
            unsigned kernels = 0;
            for (cudaGraphNode_t n : list) {
                cudaGraphNodeType t;
                CUDA_CHECK(cudaGraphNodeGetType(n, &t));
                kernels += t == cudaGraphNodeTypeKernel;
            }
            const bool head_on = M == 64640 && (N == 5 || N == 6 || N == 8);
            if (spark && (M != 64640 || head_on)) CHECK(kernels == 2);
            else CHECK(kernels >= 3);
            CUDA_CHECK(cudaGraphInstantiate(&exec, graph, 0));
            for (int round = 0; round < 3; round++) {
                const std::vector<float> x = activations(N, K, round);
                CUDA_CHECK(cudaMemcpy(b.x, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice));
                const std::vector<float> want = reference(b, w, M, N, K);
                CUDA_CHECK(cudaMemset(b.out, 0xa5, (size_t)M * N * sizeof(float)));
                CUDA_CHECK(cudaDeviceSynchronize());
                CUDA_CHECK(cudaGraphLaunch(exec, s));
                CUDA_CHECK(cudaStreamSynchronize(s));
                same(want, result(b, M, N), "graph replay", M, N, K);
            }
            CUDA_CHECK(cudaGraphExecDestroy(exec));
            CUDA_CHECK(cudaGraphDestroy(graph));
        }
        printf("q8_0 gemv: M %d K %d, N 2..8 captured in a CUDA graph, "
               "3 replays with new activations equal\n", M, K);
        CUDA_CHECK(cudaFree(w));
    }

    CUDA_CHECK(cudaStreamDestroy(s));
    cudaFree(b.x); cudaFree(b.ref); cudaFree(b.out); cudaFree(b.y);
    ds4_gpu_cleanup();
    printf("q8_0 gemv: PASS%s\n", spark ? "" : " (not a Spark device: exact GEMV off, mul_mat_q only)");
    return 0;
}
