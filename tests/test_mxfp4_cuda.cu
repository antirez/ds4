#include "ds4_mmq.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

extern "C" int ds4_cuda_q8_fold_take_q81(
        const void *src, uint64_t in_dim, const void **q81) {
    (void)src;
    (void)in_dim;
    if (q81) *q81 = nullptr;
    return 0;
}

namespace {

constexpr int QK = 32;

struct block_mxfp4_test {
    uint8_t e;
    uint8_t qs[QK / 2];
};

static_assert(sizeof(block_mxfp4_test) == 17, "unexpected MXFP4 layout");

const float kMxfp4Values[16] = {
     0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

float e8m0_to_float(uint8_t e) {
    const uint32_t bits = e == 0 ? 0x00400000u : (uint32_t)e << 23;
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

void fill_blocks(std::vector<block_mxfp4_test> &blocks, std::mt19937 &rng) {
    std::uniform_int_distribution<int> exponent(123, 127);
    std::uniform_int_distribution<int> byte(0, 255);
    for (auto &block : blocks) {
        block.e = (uint8_t)exponent(rng);
        for (uint8_t &q : block.qs) q = (uint8_t)byte(rng);
    }
}

void dequantize_row(const block_mxfp4_test *blocks, float *out, int n) {
    for (int ib = 0; ib < n / QK; ib++) {
        const float d = e8m0_to_float(blocks[ib].e);
        for (int j = 0; j < QK / 2; j++) {
            const uint8_t q = blocks[ib].qs[j];
            out[ib * QK + j] = d * kMxfp4Values[q & 0x0f];
            out[ib * QK + j + QK / 2] = d * kMxfp4Values[q >> 4];
        }
    }
}

bool cuda_ok(cudaError_t err, const char *what) {
    if (err == cudaSuccess) return true;
    std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(err));
    return false;
}

bool close_enough(const std::vector<float> &got,
                  const std::vector<float> &expected,
                  float abs_tol,
                  float rel_tol,
                  const char *label) {
    float worst = 0.0f;
    size_t worst_i = 0;
    int failures = 0;
    for (size_t i = 0; i < got.size(); i++) {
        const float diff = std::fabs(got[i] - expected[i]);
        const float limit = abs_tol + rel_tol * std::fabs(expected[i]);
        if (diff > worst) {
            worst = diff;
            worst_i = i;
        }
        if (!std::isfinite(got[i]) || diff > limit) failures++;
    }
    std::fprintf(stderr, "%s: max_abs=%g at=%zu failures=%d/%zu: %s\n",
                 label, worst, worst_i, failures, got.size(),
                 failures == 0 ? "PASS" : "FAIL");
    if (failures != 0) {
        const size_t shown = std::min<size_t>(got.size(), 8);
        for (size_t i = 0; i < shown; i++) {
            std::fprintf(stderr, "  [%zu] got=%g expected=%g diff=%g\n",
                         i, got[i], expected[i], got[i] - expected[i]);
        }
    }
    return failures == 0;
}

bool test_dense_and_moe() {
    constexpr int M = 64;
    constexpr int N = 4;
    constexpr int K = 512;
    constexpr int n_tokens = 4;
    constexpr int n_experts = 8;
    constexpr int n_used = 6;
    std::mt19937 rng(0x4d584650u);
    std::uniform_real_distribution<float> activation(-1.0f, 1.0f);
    cudaDeviceProp device = {};
    if (!cuda_ok(cudaGetDeviceProperties(&device, 0),
                 "read CUDA device properties")) {
        return false;
    }
    const bool native_fp4 = device.major >= 12;
    const float mmq_abs_tol = native_fp4 ? 8.0f : 1.0f;
    const float mmq_rel_tol = native_fp4 ? 0.08f : 0.02f;

    std::vector<block_mxfp4_test> dense_blocks((size_t)M * K / QK);
    fill_blocks(dense_blocks, rng);
    std::vector<float> dense_weights((size_t)M * K);
    for (int row = 0; row < M; row++) {
        dequantize_row(dense_blocks.data() + (size_t)row * K / QK,
                       dense_weights.data() + (size_t)row * K, K);
    }
    std::vector<float> dense_x((size_t)N * K);
    for (float &v : dense_x) v = activation(rng);
    std::vector<float> dense_ref((size_t)N * M, 0.0f);
    for (int col = 0; col < N; col++) {
        for (int row = 0; row < M; row++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += dense_weights[(size_t)row * K + k] *
                       dense_x[(size_t)col * K + k];
            }
            dense_ref[(size_t)col * M + row] = sum;
        }
    }

    void *d_weights = nullptr;
    float *d_x = nullptr;
    float *d_out = nullptr;
    cudaStream_t stream = nullptr;
    if (!cuda_ok(cudaStreamCreate(&stream), "create stream") ||
        !cuda_ok(cudaMalloc(&d_weights,
                            dense_blocks.size() * sizeof(dense_blocks[0])),
                 "allocate dense weights") ||
        !cuda_ok(cudaMalloc(&d_x, dense_x.size() * sizeof(float)),
                 "allocate dense input") ||
        !cuda_ok(cudaMalloc(&d_out, dense_ref.size() * sizeof(float)),
                 "allocate dense output")) {
        return false;
    }
    cudaMemcpyAsync(d_weights, dense_blocks.data(),
                    dense_blocks.size() * sizeof(dense_blocks[0]),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_x, dense_x.data(), dense_x.size() * sizeof(float),
                    cudaMemcpyHostToDevice, stream);
    int rc = ds4_mmq_mxfp4_dense(d_weights, d_x, d_out, M, N, K, stream);
    std::vector<float> dense_got(dense_ref.size());
    cudaMemcpyAsync(dense_got.data(), d_out,
                    dense_got.size() * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    const bool dense_ok = rc == 0 &&
        close_enough(dense_got, dense_ref, mmq_abs_tol, mmq_rel_tol,
                     "MXFP4 dense MMQ");
    bool ok = dense_ok;
    if (native_fp4) {
        const int guard_rc = ds4_mmq_mxfp4_dense(
            d_weights, d_x, d_out, M, N, K / 2, stream);
        const bool guard_ok = guard_rc != 0;
        std::fprintf(stderr, "MXFP4 Blackwell K-tile guard: %s\n",
                     guard_ok ? "PASS" : "FAIL");
        ok = ok && guard_ok;
    }
    cudaFree(d_weights);
    cudaFree(d_x);
    cudaFree(d_out);

    const size_t blocks_per_expert = (size_t)M * K / QK;
    std::vector<block_mxfp4_test> moe_blocks(
        (size_t)n_experts * blocks_per_expert);
    fill_blocks(moe_blocks, rng);
    std::vector<float> moe_weights((size_t)n_experts * M * K);
    for (int expert = 0; expert < n_experts; expert++) {
        for (int row = 0; row < M; row++) {
            dequantize_row(
                moe_blocks.data() + (size_t)expert * blocks_per_expert +
                    (size_t)row * K / QK,
                moe_weights.data() + ((size_t)expert * M + row) * K, K);
        }
    }
    std::vector<float> moe_x((size_t)n_tokens * K);
    for (float &v : moe_x) v = activation(rng);
    std::vector<int32_t> ids((size_t)n_tokens * n_used);
    for (int token = 0; token < n_tokens; token++) {
        for (int slot = 0; slot < n_used; slot++) {
            ids[(size_t)token * n_used + slot] = (token + slot) % n_experts;
        }
    }
    std::vector<float> moe_ref((size_t)n_tokens * n_used * M, 0.0f);
    for (int token = 0; token < n_tokens; token++) {
        for (int slot = 0; slot < n_used; slot++) {
            const int expert = ids[(size_t)token * n_used + slot];
            for (int row = 0; row < M; row++) {
                float sum = 0.0f;
                for (int k = 0; k < K; k++) {
                    sum += moe_weights[((size_t)expert * M + row) * K + k] *
                           moe_x[(size_t)token * K + k];
                }
                moe_ref[((size_t)token * n_used + slot) * M + row] = sum;
            }
        }
    }

    int32_t *d_ids = nullptr;
    cudaMalloc(&d_weights, moe_blocks.size() * sizeof(moe_blocks[0]));
    cudaMalloc(&d_x, moe_x.size() * sizeof(float));
    cudaMalloc(&d_ids, ids.size() * sizeof(int32_t));
    cudaMalloc(&d_out, moe_ref.size() * sizeof(float));
    cudaMemcpyAsync(d_weights, moe_blocks.data(),
                    moe_blocks.size() * sizeof(moe_blocks[0]),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_x, moe_x.data(), moe_x.size() * sizeof(float),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_ids, ids.data(), ids.size() * sizeof(int32_t),
                    cudaMemcpyHostToDevice, stream);
    rc = ds4_mmq_mxfp4_moe(d_weights, d_x, d_ids, d_out,
                            M, K, n_tokens, n_experts, n_used, stream);
    std::vector<float> moe_got(moe_ref.size());
    cudaMemcpyAsync(moe_got.data(), d_out, moe_got.size() * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    const bool moe_mmq_ok = rc == 0 &&
        close_enough(moe_got, moe_ref, mmq_abs_tol, mmq_rel_tol,
                     "MXFP4 routed MMQ");
    ok = ok && moe_mmq_ok;

    rc = ds4_mmq_mxfp4_moe_vec(d_weights, d_x, d_ids, d_out,
                                M, K, n_tokens, n_experts, n_used, stream);
    std::vector<float> moe_vec_got(moe_ref.size());
    cudaMemcpyAsync(moe_vec_got.data(), d_out,
                    moe_vec_got.size() * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    const bool moe_vec_ok = rc == 0 &&
        close_enough(moe_vec_got, moe_ref, 1.0f, 0.02f,
                     "MXFP4 routed MMVQ");
    ok = ok && moe_vec_ok;
    cudaFree(d_weights);
    cudaFree(d_x);
    cudaFree(d_ids);
    cudaFree(d_out);
    cudaStreamDestroy(stream);
    return ok;
}

bool test_fused_decode() {
    constexpr int input_dim = 256;
    constexpr int mid_dim = 256;
    constexpr int out_dim = 64;
    constexpr int n_experts = 8;
    constexpr int n_used = 6;
    std::mt19937 rng(0x4445434fu);
    std::uniform_real_distribution<float> activation(-1.0f, 1.0f);

    const size_t gu_blocks = (size_t)n_experts * mid_dim * input_dim / QK;
    const size_t down_blocks = (size_t)n_experts * out_dim * mid_dim / QK;
    std::vector<block_mxfp4_test> gate(gu_blocks), up(gu_blocks),
                                   down(down_blocks);
    fill_blocks(gate, rng);
    fill_blocks(up, rng);
    fill_blocks(down, rng);
    std::vector<float> x(input_dim);
    for (float &v : x) v = activation(rng);
    std::vector<int32_t> ids = {7, 1, 5, 0, 3, 6};
    std::vector<float> router = {0.25f, 0.20f, 0.18f, 0.15f, 0.12f, 0.10f};

    void *d_gate = nullptr, *d_up = nullptr, *d_down = nullptr;
    float *d_x = nullptr, *d_gate_out = nullptr, *d_up_out = nullptr;
    float *d_mid = nullptr, *d_slots = nullptr, *d_out = nullptr;
    int32_t *d_ids = nullptr;
    float *d_router = nullptr;
    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);
    cudaMalloc(&d_gate, gate.size() * sizeof(gate[0]));
    cudaMalloc(&d_up, up.size() * sizeof(up[0]));
    cudaMalloc(&d_down, down.size() * sizeof(down[0]));
    cudaMalloc(&d_x, x.size() * sizeof(float));
    cudaMalloc(&d_ids, ids.size() * sizeof(int32_t));
    cudaMalloc(&d_router, router.size() * sizeof(float));
    cudaMalloc(&d_gate_out, (size_t)n_used * mid_dim * sizeof(float));
    cudaMalloc(&d_up_out, (size_t)n_used * mid_dim * sizeof(float));
    cudaMalloc(&d_mid, (size_t)n_used * mid_dim * sizeof(float));
    cudaMalloc(&d_slots, (size_t)n_used * out_dim * sizeof(float));
    cudaMalloc(&d_out, (size_t)out_dim * sizeof(float));
    cudaMemcpyAsync(d_gate, gate.data(), gate.size() * sizeof(gate[0]),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_up, up.data(), up.size() * sizeof(up[0]),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_down, down.data(), down.size() * sizeof(down[0]),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_x, x.data(), x.size() * sizeof(float),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_ids, ids.data(), ids.size() * sizeof(int32_t),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_router, router.data(), router.size() * sizeof(float),
                    cudaMemcpyHostToDevice, stream);

    int rc_gate = ds4_mmq_mxfp4_moe_vec(
        d_gate, d_x, d_ids, d_gate_out, mid_dim, input_dim,
        1, n_experts, n_used, stream);
    int rc_up = ds4_mmq_mxfp4_moe_vec(
        d_up, d_x, d_ids, d_up_out, mid_dim, input_dim,
        1, n_experts, n_used, stream);
    int rc_mid = ds4_mmq_mxfp4_moe_gate_up_mid_vec(
        d_gate, d_up, d_x, d_ids, d_router, d_mid,
        mid_dim, input_dim, 1, n_experts, n_used, 7.0f, stream);

    std::vector<float> gate_out((size_t)n_used * mid_dim);
    std::vector<float> up_out(gate_out.size()), mid(gate_out.size());
    cudaMemcpyAsync(gate_out.data(), d_gate_out,
                    gate_out.size() * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(up_out.data(), d_up_out,
                    up_out.size() * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(mid.data(), d_mid,
                    mid.size() * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    std::vector<float> mid_ref(mid.size());
    for (int slot = 0; slot < n_used; slot++) {
        for (int row = 0; row < mid_dim; row++) {
            const size_t i = (size_t)slot * mid_dim + row;
            const float g = std::min(gate_out[i], 7.0f);
            const float u = std::max(-7.0f, std::min(up_out[i], 7.0f));
            mid_ref[i] = g / (1.0f + std::exp(-g)) * u * router[slot];
        }
    }
    bool ok = rc_gate == 0 && rc_up == 0 && rc_mid == 0 &&
        close_enough(mid, mid_ref, 0.02f, 0.002f,
                     "MXFP4 fused gate/up decode");

    int rc_slots = ds4_mmq_mxfp4_moe_vec(
        d_down, d_mid, d_ids, d_slots, out_dim, mid_dim,
        n_used, n_experts, 1, stream);
    int rc_sum = ds4_mmq_mxfp4_moe_down_sum6_vec(
        d_down, d_mid, d_ids, d_out, out_dim, mid_dim,
        1, n_experts, n_used, stream);
    std::vector<float> slots((size_t)n_used * out_dim), out(out_dim);
    cudaMemcpyAsync(slots.data(), d_slots, slots.size() * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(out.data(), d_out, out.size() * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    std::vector<float> out_ref(out_dim, 0.0f);
    for (int slot = 0; slot < n_used; slot++) {
        for (int row = 0; row < out_dim; row++) {
            out_ref[row] += slots[(size_t)slot * out_dim + row];
        }
    }
    ok = ok && rc_slots == 0 && rc_sum == 0 &&
        close_enough(out, out_ref, 0.01f, 0.001f,
                     "MXFP4 fused down decode");

    cudaFree(d_gate);
    cudaFree(d_up);
    cudaFree(d_down);
    cudaFree(d_x);
    cudaFree(d_ids);
    cudaFree(d_router);
    cudaFree(d_gate_out);
    cudaFree(d_up_out);
    cudaFree(d_mid);
    cudaFree(d_slots);
    cudaFree(d_out);
    cudaStreamDestroy(stream);
    return ok;
}

} // namespace

int main() {
    if (ds4_mmq_init(0) != 0) {
        std::fprintf(stderr, "ds4_mmq_init failed\n");
        return 1;
    }
    const bool matrix_ok = test_dense_and_moe();
    const bool decode_ok = test_fused_decode();
    const bool ok = matrix_ok && decode_ok;
    std::fprintf(stderr, "MXFP4 CUDA parity: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
