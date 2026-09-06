// SPDX-License-Identifier: MIT
// Actual Q4 RMS reduction helper versus an independent 256-leaf tree.
// Optional nvcc build verifies both barriers, all consumers, and GPU rounding.
// This does not compile or benchmark the full Q-B GEMM/RMS/RoPE operation.
#if defined(__CUDACC__)
#include <cuda_runtime.h>
#endif
#include "../cuda/ds4_q4_prefill_reduce.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static void require(bool ok, const char *what) {
    if (!ok) {
        std::fprintf(stderr, "Q4 prefill reduction: FAIL %s\n", what);
        std::exit(1);
    }
}
static uint32_t bits(float value) {
    uint32_t word; std::memcpy(&word, &value, sizeof(word)); return word;
}
static float from_bits(uint32_t word) {
    float value; std::memcpy(&value, &word, sizeof(value)); return value;
}
static bool is_nan_bits(uint32_t word) {
    return (word & 0x7f800000u) == 0x7f800000u && (word & 0x007fffffu);
}
static bool same(float a, float b) {
    return bits(a) == bits(b) || (is_nan_bits(bits(a)) && is_nan_bits(bits(b)));
}
static uint32_t random_word(uint32_t &seed) {
    seed ^= seed << 13u; seed ^= seed >> 17u; seed ^= seed << 5u;
    return seed;
}
static float reference_add(float a, float b) {
    volatile float rounded = a + b;
    return rounded;
}
static float reference(std::array<float, 256> leaves) {
    for (unsigned stride = 128u; stride; stride >>= 1u)
        for (unsigned lane = 0; lane < stride; ++lane)
            leaves[lane] = reference_add(leaves[lane], leaves[lane + stride]);
    return leaves[0];
}
static float candidate(const std::array<float, 256> &leaves) {
    std::array<float, 32> warp;
    for (unsigned lane = 0; lane < 32u; ++lane)
        warp[lane] = ds4_q4_prefill_reduce_lane(leaves.data(), lane);
    for (unsigned stride = 16u; stride; stride >>= 1u) {
        // A shuffle reads the values from before this warp instruction.
        const auto before = warp;
        for (unsigned lane = 0; lane < stride; ++lane)
            warp[lane] = ds4_q4_prefill_add(before[lane], before[lane + stride]);
    }
    return warp[0];
}
static void fill_leaves(std::array<float, 256> &leaves, unsigned row) {
    uint32_t seed = 0xafe41cb9u ^ (row + 1u) * 7919u;
    constexpr uint32_t special[] = {
        0u, 0x80000000u, 1u, 0x80000001u, 0x007fffffu, 0x00800000u,
        0x3f800000u, 0xbf800000u, 0x7f7fffffu, 0xff7fffffu,
        0x7f800000u, 0xff800000u, 0x7fc00001u,
    };
    for (unsigned lane = 0; lane < 256u; ++lane) {
        const uint32_t word = random_word(seed);
        if (row < 13u) leaves[lane] = from_bits(special[row]);
        else if (row < 26u) leaves[lane] = from_bits(special[(row + lane) % 13u]);
        else if (row % 5u == 0u) {
            // Non-negative, finite partials resemble sum-of-squares inputs.
            leaves[lane] = from_bits((word & 0x007fffffu) | ((word % 230u) << 23u));
        } else if (row % 5u == 1u) {
            // Cancellation makes a reassociated, superficially valid tree fail.
            leaves[lane] = from_bits((word & 0x807fffffu) | ((word % 254u) << 23u));
        } else if (row % 5u == 2u) {
            leaves[lane] = float(int(word % 1025u) - 512) / 256.0f;
        } else if (row % 5u == 3u) {
            leaves[lane] = from_bits((word & 0x807fffffu) | 0x00800000u);
        } else {
            leaves[lane] = lane % 3u == 0u ? 16777216.0f :
                           lane % 3u == 1u ? -16777216.0f : 1.0f;
        }
    }
}
static void host_test() {
    std::array<float, 256> leaves{};
    unsigned cases = 0;
    auto check = [&] {
        const auto before = leaves;
        require(same(candidate(leaves), reference(leaves)), "host operand tree / rounding");
        require(std::memcmp(leaves.data(), before.data(), sizeof(leaves)) == 0,
                "host helper modified shared leaves");
        ++cases;
    };
    for (unsigned row = 0; row < 32768u; ++row) {
        fill_leaves(leaves, row); check();
    }
    for (unsigned lane = 0; lane < 256u; ++lane) {
        leaves.fill(0.0f); leaves[lane] = 1.0f; check();
        leaves.fill(-0.0f); leaves[lane] = -1.0f; check();
    }
    // A head may give each worker zero, one, or multiple products. Feed the
    // same original per-thread partials to both reductions, without imposing
    // CPU multiply/FMA semantics on the CUDA kernel's accumulation loop.
    for (unsigned dim : {32u, 64u, 128u, 256u, 512u, 768u, 1024u, 1536u, 4096u}) {
        for (unsigned row = 0; row < 128u; ++row) {
            uint32_t seed = 0x901239c5u + row * 97u;
            leaves.fill(0.0f);
            for (unsigned i = 0; i < dim; ++i) {
                const float x = float(int(random_word(seed) % 65537u) - 32768) / 512.0f;
                leaves[i % 256u] = reference_add(leaves[i % 256u], x * x);
            }
            check();
        }
    }
    // Attest that the fixture can distinguish the forbidden reassociation.
    leaves.fill(0.0f); leaves[0] = 16777216.0f;
    leaves[128] = 1.0f; leaves[64] = -16777216.0f; leaves[192] = 1.0f;
    require(bits(reference(leaves)) != bits(2.0f), "cancellation witness");
    check();
    std::printf("PASS host: %u trees, finite results bitwise, non-finite classes, "
                "signed zeros, subnormals, lane ownership, varied head partials.\n", cases);
}

#if defined(__CUDACC__)
static void cuda_check(cudaError_t result, const char *what) {
    if (result != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(result));
        std::exit(1);
    }
}
template<bool optimized>
__global__ static void reduction_kernel(float *out, const float *leaves, unsigned rows) {
    if (blockIdx.x >= rows) return;
    const unsigned tid = threadIdx.x;
    __shared__ float partial[256];
    const float sum = leaves[(size_t)blockIdx.x * 256u + tid];
    float total;
    if (optimized) total = ds4_q4_prefill_reduce_256(sum, partial);
    else {
        partial[tid] = sum;
        __syncthreads();
        for (unsigned stride = blockDim.x >> 1u; stride; stride >>= 1u) {
            if (tid < stride) partial[tid] += partial[tid + stride];
            __syncthreads();
        }
        total = partial[0];
    }
    // Every warp consumes the publication after the second CTA barrier.
    out[(size_t)blockIdx.x * 256u + tid] = total;
}
static void gpu_test(bool bench) {
    constexpr unsigned rows = 4099u, guard = 64u, repeats = 16u;
    constexpr uint32_t sentinel = 0x4f123456u;
    const size_t values = (size_t)rows * 256u, bytes = values * sizeof(float);
    std::vector<float> input(values), initial(values + 2u * guard, from_bits(sentinel));
    std::array<float, 256> leaves;
    for (unsigned row = 0; row < rows; ++row) {
        fill_leaves(leaves, row);
        std::copy(leaves.begin(), leaves.end(), input.begin() + (size_t)row * 256u);
    }
    float *d_in = nullptr, *d_ref = nullptr, *d_got = nullptr;
    cuda_check(cudaMalloc((void **)&d_in, bytes), "allocate input");
    cuda_check(cudaMalloc((void **)&d_ref, initial.size() * sizeof(float)), "allocate reference");
    cuda_check(cudaMalloc((void **)&d_got, initial.size() * sizeof(float)), "allocate candidate");
    cuda_check(cudaMemcpy(d_in, input.data(), bytes, cudaMemcpyHostToDevice), "copy input");
    cuda_check(cudaMemcpy(d_ref, initial.data(), initial.size() * sizeof(float), cudaMemcpyHostToDevice), "guard reference");
    cuda_check(cudaMemcpy(d_got, initial.data(), initial.size() * sizeof(float), cudaMemcpyHostToDevice), "guard candidate");
    auto launch = [&](bool candidate) {
        if (candidate) reduction_kernel<true><<<rows + 3u, 256>>>(d_got + guard, d_in, rows);
        else reduction_kernel<false><<<rows + 3u, 256>>>(d_ref + guard, d_in, rows);
        cuda_check(cudaGetLastError(), "reduction kernel launch");
    };
    for (bool order : {false, true, true, false}) launch(order);
    cuda_check(cudaDeviceSynchronize(), "reduction completion");
    std::vector<float> expected(initial.size()), got(initial.size()), input_after(values);
    cuda_check(cudaMemcpy(expected.data(), d_ref, initial.size() * sizeof(float), cudaMemcpyDeviceToHost), "read reference");
    cuda_check(cudaMemcpy(got.data(), d_got, initial.size() * sizeof(float), cudaMemcpyDeviceToHost), "read candidate");
    cuda_check(cudaMemcpy(input_after.data(), d_in, bytes, cudaMemcpyDeviceToHost), "read input");
    require(std::memcmp(input.data(), input_after.data(), bytes) == 0, "GPU input modified");
    for (size_t i = 0; i < initial.size(); ++i) {
        if (i < guard || i >= values + guard) {
            require(bits(got[i]) == sentinel && bits(expected[i]) == sentinel, "GPU guard / row tail");
        } else {
            require(same(expected[i], got[i]), "GPU bitwise tree parity / shared publication");
            const size_t row0 = guard + ((i - guard) / 256u) * 256u;
            require(same(got[i], got[row0]), "GPU all-warp broadcast");
            require(bits(got[i]) != sentinel, "GPU unwritten consumer");
        }
    }
    std::printf("PASS CUDA: %u rows, all 256 consumers, guards, immutable input; "
                "finite output bitwise versus canonical GPU tree.\n", rows);
    if (bench) {
        cudaEvent_t start, end;
        cuda_check(cudaEventCreate(&start), "event create");
        cuda_check(cudaEventCreate(&end), "event create");
        std::vector<float> timings[2];
        for (unsigned sample = 0; sample < 8u; ++sample) {
            const bool order[4] = {bool(sample & 1u), !bool(sample & 1u),
                                   !bool(sample & 1u), bool(sample & 1u)};
            for (bool candidate : order) {
                cuda_check(cudaEventRecord(start), "event record");
                for (unsigned repeat = 0; repeat < repeats; ++repeat) launch(candidate);
                cuda_check(cudaEventRecord(end), "event record");
                cuda_check(cudaEventSynchronize(end), "event completion");
                float elapsed = 0.0f;
                cuda_check(cudaEventElapsedTime(&elapsed, start, end), "event time");
                timings[candidate].push_back(elapsed * 1000.0f / repeats);
            }
        }
        for (auto &samples : timings) std::sort(samples.begin(), samples.end());
        const float old_us = (timings[0][7] + timings[0][8]) * 0.5f;
        const float new_us = (timings[1][7] + timings[1][8]) * 0.5f;
        std::printf("CUDA reduction-only ABBA/BAAB median: baseline %.3f us, candidate %.3f us, "
                    "speedup %.4fx. Includes all-consumer stores; not full prefill timing.\n",
                    old_us, new_us, old_us / new_us);
        cuda_check(cudaEventDestroy(start), "event destroy");
        cuda_check(cudaEventDestroy(end), "event destroy");
    }
    cuda_check(cudaFree(d_got), "free candidate");
    cuda_check(cudaFree(d_ref), "free reference");
    cuda_check(cudaFree(d_in), "free input");
}
#endif

int main(int argc, char **argv) {
    const bool bench = argc == 2 && std::strcmp(argv[1], "--bench") == 0;
    require(argc == 1 || bench, "usage: test_cuda_q4_prefill_reduce [--bench]");
    host_test();
#if defined(__CUDACC__)
    gpu_test(bench);
#else
    require(!bench, "--bench requires an nvcc build and CUDA device");
    std::puts("Host only: GPU compilation, FTZ, barriers and full Q-B/RMS/RoPE parity remain unverified.");
#endif
    return 0;
}
