// SPDX-License-Identifier: MIT
// test_mmq_soa_tiles.cu - P4 Inc3 decision gate: SoA-direct mmq tile loaders
// vs the raw-block path, at the production V4-Flash MoE shapes.
//
// Method: generate a RANDOM aligned-SoA artifact (the weight-server format),
// derive the raw block stream from it with the production derepack kernel
// (ds4_mmq_*_aligned_derepack -- byte-exactly what the scratch path serves),
// then run the same MoE matmul through both entries and require bit-identical
// outputs:
//
//   Q2_K down leg:      ds4_mmq_q2_K_moe(raw)          vs ds4_mmq_q2_K_moe_soa(soa)
//   IQ2_XXS gate/up leg: ds4_mmq_iq2_xxs_moe_pair(raw)  vs ds4_mmq_iq2_xxs_moe_pair_soa(soa)
//
// Shapes are the V4-Flash production calls at a W4096 prefill chunk:
//   down:    M=4096 (n_embd), K=2048 (n_ff_exp), 24576 assignments x 1 expert
//   gate/up: M=2048,          K=4096,            4096 tokens x 6 experts
// with 256 experts and uniform routing (~96 columns/expert, the trace shape).
//
// Timing: cudaEvent over REPS calls of each entry (identical quantize +
// mm_ids overhead on both sides, so the delta is the tile-load effect), plus
// the standalone derepack kernel cost (what the SoA path deletes per layer
// switch).
//
// Build (on the box, after `make -j6 cuda-spark` built the mmq objects):
//   ar rcs libds4mmq.a cuda/mmq/ds4_ggml_stubs.o cuda/mmq/ds4_mmq.o \
//          cuda/mmq/quantize.o cuda/mmq/mmid.o cuda/mmq/mmvq.o
//   /usr/local/cuda/bin/nvcc -O3 --use_fast_math -std=c++17 -arch=sm_121 \
//        -Icuda/mmq cuda/mmq/test/test_mmq_soa_tiles.cu libds4mmq.a \
//        -lcudart -lcublas -lcuda -o test_mmq_soa_tiles

#include "ds4_mmq.h"

#include <cuda_runtime.h>

// libds4mmq.a references this ds4_cuda.cu symbol from the q8-fold vec paths
// (C3 Inc4); the entries under test never reach it, so a "no fold available"
// stub satisfies the link.
extern "C" int ds4_cuda_q8_fold_take_q81(const void *src, uint64_t in_dim, void *out) {
    (void)src; (void)in_dim; (void)out;
    return 0;
}

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define CHECK(call) do { \
    cudaError_t err_ = (call); \
    if (err_ != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__, __LINE__, \
                cudaGetErrorString(err_)); \
        exit(1); \
    } \
} while (0)

namespace {

int g_failures = 0;

// Random bytes with half-precision words nudged into a sane exponent range
// (bit-parity would hold for any bits -- both paths read the same words --
// but normal-range scales keep the timing legs representative).
uint16_t sane_half(std::mt19937 &rng) {
    const uint16_t mant = (uint16_t)(rng() & 0x03FFu);
    const uint16_t sign = (uint16_t)((rng() & 1u) << 15);
    return (uint16_t)(sign | (14u << 10) | mant);   // +/- [0.25, 0.5)
}

void fill_random(std::mt19937 &rng, void *dst, size_t bytes) {
    uint32_t *p = (uint32_t *)dst;
    for (size_t i = 0; i < bytes / 4; ++i) p[i] = rng();
}

float *device_random_floats(std::mt19937 &rng, size_t n) {
    std::vector<float> h(n);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < n; ++i) h[i] = dist(rng);
    float *d = nullptr;
    CHECK(cudaMalloc(&d, n * sizeof(float)));
    CHECK(cudaMemcpy(d, h.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    return d;
}

bool compare_outputs(const char *what, const float *d_a, const float *d_b, size_t n) {
    std::vector<float> a(n), b(n);
    CHECK(cudaMemcpy(a.data(), d_a, n * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(b.data(), d_b, n * sizeof(float), cudaMemcpyDeviceToHost));
    if (memcmp(a.data(), b.data(), n * sizeof(float)) == 0) {
        printf("  %-28s PARITY-OK (bit-identical, %zu floats)\n", what, n);
        return true;
    }
    size_t first = 0, count = 0;
    for (size_t i = 0; i < n; ++i) {
        if (memcmp(&a[i], &b[i], 4) != 0) { if (!count) first = i; ++count; }
    }
    printf("  %-28s PARITY-FAIL: %zu/%zu differ, first at %zu (raw=%.9g soa=%.9g)\n",
           what, count, n, first, a[first], b[first]);
    ++g_failures;
    return false;
}

template <typename F>
float time_ms(F &&fn, int reps) {
    cudaEvent_t e0, e1;
    CHECK(cudaEventCreate(&e0));
    CHECK(cudaEventCreate(&e1));
    fn();                                   // warmup
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaEventRecord(e0));
    for (int r = 0; r < reps; ++r) fn();
    CHECK(cudaEventRecord(e1));
    CHECK(cudaEventSynchronize(e1));
    float ms = 0.0f;
    CHECK(cudaEventElapsedTime(&ms, e0, e1));
    CHECK(cudaEventDestroy(e0));
    CHECK(cudaEventDestroy(e1));
    return ms / reps;
}

// ---------------------------------------------------------------------------
// Q2_K down leg.
// ---------------------------------------------------------------------------
void run_q2k_leg(std::mt19937 &rng, int reps) {
    const int E = 256, M = 4096, K = 2048;
    const int n_tokens = 4096 * 6;          // assignments, n_expert_used == 1
    const uint64_t nb_row = (uint64_t)K / 256;
    const uint64_t nblk   = (uint64_t)E * M * nb_row;
    const uint64_t npair  = nblk / 2;
    const uint64_t soa_bytes = ds4_mmq_q2_k_aligned_bytes(M, K, E);
    const uint64_t raw_bytes = nblk * 84ull;
    printf("Q2_K down leg: E=%d M=%d K=%d cols=%d (soa %.1f MiB, raw %.1f MiB)\n",
           E, M, K, n_tokens, soa_bytes / 1048576.0, raw_bytes / 1048576.0);

    // Random SoA artifact; dm halves get sane exponents.
    std::vector<uint8_t> h_soa(soa_bytes);
    fill_random(rng, h_soa.data(), soa_bytes);
    uint16_t *dm = (uint16_t *)h_soa.data();
    for (uint64_t i = 0; i < npair * 4; ++i) dm[i] = sane_half(rng);

    void *d_soa = nullptr, *d_raw = nullptr;
    CHECK(cudaMalloc(&d_soa, soa_bytes));
    CHECK(cudaMalloc(&d_raw, raw_bytes));
    CHECK(cudaMemcpy(d_soa, h_soa.data(), soa_bytes, cudaMemcpyHostToDevice));

    // Raw twin = production derepack of the artifact.
    if (ds4_mmq_q2_K_aligned_derepack(d_soa, d_raw, M, K, E, 0) != 0) {
        fprintf(stderr, "q2k derepack failed\n"); exit(1);
    }
    CHECK(cudaDeviceSynchronize());

    float *d_x = device_random_floats(rng, (size_t)n_tokens * K);
    std::vector<int32_t> h_ids(n_tokens);
    for (int i = 0; i < n_tokens; ++i) h_ids[i] = (int32_t)(rng() % E);
    int32_t *d_ids = nullptr;
    CHECK(cudaMalloc(&d_ids, n_tokens * sizeof(int32_t)));
    CHECK(cudaMemcpy(d_ids, h_ids.data(), n_tokens * sizeof(int32_t), cudaMemcpyHostToDevice));

    const size_t out_n = (size_t)M * n_tokens;
    float *d_out_raw = nullptr, *d_out_soa = nullptr;
    CHECK(cudaMalloc(&d_out_raw, out_n * sizeof(float)));
    CHECK(cudaMalloc(&d_out_soa, out_n * sizeof(float)));

    if (ds4_mmq_q2_K_moe(d_raw, d_x, d_ids, d_out_raw, M, K, n_tokens, E, 1, 0) != 0) {
        fprintf(stderr, "q2k raw moe failed\n"); exit(1);
    }
    if (ds4_mmq_q2_K_moe_soa(d_soa, d_x, d_ids, d_out_soa, M, K, n_tokens, E, 1, 0) != 0) {
        fprintf(stderr, "q2k soa moe failed\n"); exit(1);
    }
    CHECK(cudaDeviceSynchronize());
    compare_outputs("q2k down W4096", d_out_raw, d_out_soa, out_n);

    const float ms_raw = time_ms([&] {
        ds4_mmq_q2_K_moe(d_raw, d_x, d_ids, d_out_raw, M, K, n_tokens, E, 1, 0);
    }, reps);
    const float ms_soa = time_ms([&] {
        ds4_mmq_q2_K_moe_soa(d_soa, d_x, d_ids, d_out_soa, M, K, n_tokens, E, 1, 0);
    }, reps);
    const float ms_dr = time_ms([&] {
        ds4_mmq_q2_K_aligned_derepack(d_soa, d_raw, M, K, E, 0);
    }, reps);
    printf("  q2k down: raw %.3f ms  soa %.3f ms (%+.1f%%)  [derepack alone %.3f ms/launch]\n",
           ms_raw, ms_soa, 100.0 * (ms_soa - ms_raw) / ms_raw, ms_dr);

    CHECK(cudaFree(d_soa));  CHECK(cudaFree(d_raw));
    CHECK(cudaFree(d_x));    CHECK(cudaFree(d_ids));
    CHECK(cudaFree(d_out_raw)); CHECK(cudaFree(d_out_soa));
}

// ---------------------------------------------------------------------------
// IQ2_XXS gate/up pair leg.
// ---------------------------------------------------------------------------
void run_iq2_leg(std::mt19937 &rng, int reps) {
    const int E = 256, M = 2048, K = 4096;
    const int n_tokens = 4096, n_used = 6;
    const uint64_t nb_row = (uint64_t)K / 256;
    const uint64_t nblk   = (uint64_t)E * M * nb_row;
    const uint64_t soa_bytes = ds4_mmq_iq2_xxs_aligned_bytes(M, K, E);
    const uint64_t raw_bytes = nblk * 66ull;
    printf("IQ2_XXS gate/up leg: E=%d M=%d K=%d tokens=%dx%d (soa %.1f MiB x2, raw %.1f MiB x2)\n",
           E, M, K, n_tokens, n_used, soa_bytes / 1048576.0, raw_bytes / 1048576.0);

    void *d_soa[2] = {nullptr, nullptr}, *d_raw[2] = {nullptr, nullptr};
    for (int t = 0; t < 2; ++t) {
        std::vector<uint8_t> h_soa(soa_bytes);
        fill_random(rng, h_soa.data(), soa_bytes);
        uint16_t *dq = (uint16_t *)h_soa.data();
        for (uint64_t i = 0; i < nblk; ++i) dq[i] = sane_half(rng);
        CHECK(cudaMalloc(&d_soa[t], soa_bytes));
        CHECK(cudaMalloc(&d_raw[t], raw_bytes));
        CHECK(cudaMemcpy(d_soa[t], h_soa.data(), soa_bytes, cudaMemcpyHostToDevice));
        if (ds4_mmq_iq2_xxs_aligned_derepack(d_soa[t], d_raw[t], M, K, E, 0) != 0) {
            fprintf(stderr, "iq2 derepack failed\n"); exit(1);
        }
    }
    CHECK(cudaDeviceSynchronize());

    float *d_x = device_random_floats(rng, (size_t)n_tokens * K);
    // Production routing: n_used DISTINCT experts per token.
    std::vector<int32_t> h_ids((size_t)n_tokens * n_used);
    for (int i = 0; i < n_tokens; ++i) {
        int picked[6];
        for (int j = 0; j < n_used; ++j) {
            int e;
            bool dup;
            do {
                e = (int)(rng() % E);
                dup = false;
                for (int q = 0; q < j; ++q) dup |= (picked[q] == e);
            } while (dup);
            picked[j] = e;
            h_ids[(size_t)i * n_used + j] = e;
        }
    }
    int32_t *d_ids = nullptr;
    CHECK(cudaMalloc(&d_ids, h_ids.size() * sizeof(int32_t)));
    CHECK(cudaMemcpy(d_ids, h_ids.data(), h_ids.size() * sizeof(int32_t), cudaMemcpyHostToDevice));

    const size_t out_n = (size_t)M * n_tokens * n_used;
    float *d_out[4] = {nullptr, nullptr, nullptr, nullptr};   // raw a/b, soa a/b
    for (int i = 0; i < 4; ++i) CHECK(cudaMalloc(&d_out[i], out_n * sizeof(float)));

    if (ds4_mmq_iq2_xxs_moe_pair(d_raw[0], d_raw[1], d_x, d_ids, d_out[0], d_out[1],
                                 M, K, n_tokens, E, n_used, 0) != 0) {
        fprintf(stderr, "iq2 raw pair failed\n"); exit(1);
    }
    if (ds4_mmq_iq2_xxs_moe_pair_soa(d_soa[0], d_soa[1], d_x, d_ids, d_out[2], d_out[3],
                                     M, K, n_tokens, E, n_used, 0) != 0) {
        fprintf(stderr, "iq2 soa pair failed\n"); exit(1);
    }
    CHECK(cudaDeviceSynchronize());
    compare_outputs("iq2 gate W4096", d_out[0], d_out[2], out_n);
    compare_outputs("iq2 up   W4096", d_out[1], d_out[3], out_n);

    const float ms_raw = time_ms([&] {
        ds4_mmq_iq2_xxs_moe_pair(d_raw[0], d_raw[1], d_x, d_ids, d_out[0], d_out[1],
                                 M, K, n_tokens, E, n_used, 0);
    }, reps);
    const float ms_soa = time_ms([&] {
        ds4_mmq_iq2_xxs_moe_pair_soa(d_soa[0], d_soa[1], d_x, d_ids, d_out[2], d_out[3],
                                     M, K, n_tokens, E, n_used, 0);
    }, reps);
    const float ms_dr = time_ms([&] {
        ds4_mmq_iq2_xxs_aligned_derepack(d_soa[0], d_raw[0], M, K, E, 0);
    }, reps);
    printf("  iq2 pair: raw %.3f ms  soa %.3f ms (%+.1f%%)  [derepack alone %.3f ms/launch, x2 tensors]\n",
           ms_raw, ms_soa, 100.0 * (ms_soa - ms_raw) / ms_raw, ms_dr);

    for (int t = 0; t < 2; ++t) { CHECK(cudaFree(d_soa[t])); CHECK(cudaFree(d_raw[t])); }
    CHECK(cudaFree(d_x)); CHECK(cudaFree(d_ids));
    for (int i = 0; i < 4; ++i) CHECK(cudaFree(d_out[i]));
}

} // namespace

int main(int argc, char **argv) {
    const int reps = (argc > 1) ? atoi(argv[1]) : 20;
    std::mt19937 rng(0x5A0Au);
    run_q2k_leg(rng, reps);
    run_iq2_leg(rng, reps);
    printf(g_failures ? "SOA-TILES FAIL (%d)\n" : "SOA-TILES PASS\n", g_failures);
    return g_failures ? 1 : 0;
}
