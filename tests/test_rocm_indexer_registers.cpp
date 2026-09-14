#include "rocm/ds4_rocm_indexer_registers.cuh"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

static void require(bool ok, const char *what) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what); std::exit(1); }
}

#if !defined(TEST_NATIVE)
// Model the public coordinate-label scheme under arbitrary bijective fragment
// layouts. This proves packing/scattering independently of any AMD lane formula;
// it does NOT establish the mapping implemented by an installed rocWMMA build.
static float value(std::mt19937 &rng) {
    uint32_t bits = (rng() & 0x807fffffu) | ((120u + rng() % 12u) << 23u);
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}
static __attribute__((noinline)) float weighted(float acc, float dot, float weight) {
    return acc + std::fmax(dot, 0.0f) * weight;
}
int main() {
    std::mt19937 rng(0x1151u);
    std::array<uint32_t, 256> permutation;
    std::iota(permutation.begin(), permutation.end(), 0u);
    std::array<float, 64 * 256> dot;
    std::array<float, 64 * 16> weight;
    for (float &x : dot) x = value(rng);
    for (float &x : weight) x = value(rng);
    unsigned checks = 0;
    for (unsigned trial = 0; trial < 128; ++trial) {
        std::shuffle(permutation.begin(), permutation.end(), rng);
        std::array<float, 256> actual{}, expected{};
        std::array<unsigned, 256> written{};
        for (unsigned h = 0; h < 64; ++h)
            for (unsigned rc = 0; rc < 256; ++rc)
                expected[rc] = weighted(expected[rc], dot[h * 256 + rc], weight[h * 16 + rc / 16]);
        for (unsigned lane = 0; lane < 32; ++lane) {
            const uint32_t *p = permutation.data() + lane * 8;
            const uint32_t lo = ds4_indexer_pack_coordinates(p[0], p[1], p[2], p[3]);
            const uint32_t hi = ds4_indexer_pack_coordinates(p[4], p[5], p[6], p[7]);
            float acc[8] = {};
            for (unsigned h = 0; h < 64; ++h)
                for (unsigned slot = 0; slot < 8; ++slot) {
                    const uint32_t rc = ds4_indexer_unpack_coordinate(lo, hi, slot);
                    require(rc == p[slot], "packed coordinate roundtrip");
                    acc[slot] = weighted(acc[slot], dot[h * 256 + rc], weight[h * 16 + rc / 16]);
                    ++checks;
                }
            for (unsigned slot = 0; slot < 8; ++slot) {
                const uint32_t rc = ds4_indexer_unpack_coordinate(lo, hi, slot);
                actual[rc] = acc[slot];
                ++written[rc];
            }
        }
        for (unsigned rc = 0; rc < 256; ++rc) {
            require(written[rc] == 1, "unique output ownership");
            require(std::memcmp(&actual[rc], &expected[rc], sizeof(float)) == 0,
                    "ordered head accumulation under permuted layout");
        }
    }
    std::printf("PASS host: %u coordinate checks, 128 arbitrary layouts; native mapping untested\n", checks);
}
#else
static void hip_check(hipError_t err, const char *what) {
    if (err != hipSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, hipGetErrorString(err));
        std::exit(1);
    }
}
#define HIP(call) hip_check((call), #call)

// No assumed lane mapping: compare load_matrix_sync labels to the result of
// identity * coordinate-matrix and to repeated MMA with a loaded accumulator.
// Also store via rocWMMA so the host checks an independent matrix-space view.
__global__ static void diagnose(uint32_t *mapping, float *roundtrip, uint32_t *failures) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx1151__)
    using fa = rocwmma::fragment<rocwmma::matrix_a, 16, 16, 16, __half, rocwmma::row_major>;
    using fb = rocwmma::fragment<rocwmma::matrix_b, 16, 16, 16, __half, rocwmma::col_major>;
    using fc = rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, float>;
    static_assert(fc::num_elements == 8, "diagnostic requires wave32");
    __shared__ float labels[256];
    __shared__ __half identity[256], matrix[256];
    const uint32_t tid = threadIdx.x;
    const uint32_t wave = tid / 32u, lane = tid % 32u;
    labels[tid] = (float)tid;
    identity[tid] = __float2half(tid / 16u == tid % 16u ? 1.0f : 0.0f);
    matrix[tid] = __float2half((float)((tid % 16u) * 16u + tid / 16u));
    __syncthreads();
    fc coords;
    rocwmma::load_matrix_sync(coords, labels, 16, rocwmma::mem_row_major);
    const uint32_t lo = ds4_indexer_pack_coordinates(
        (uint32_t)coords.x[0], (uint32_t)coords.x[1], (uint32_t)coords.x[2], (uint32_t)coords.x[3]);
    const uint32_t hi = ds4_indexer_pack_coordinates(
        (uint32_t)coords.x[4], (uint32_t)coords.x[5], (uint32_t)coords.x[6], (uint32_t)coords.x[7]);
    fa a;
    fb b;
    fc c;
    rocwmma::load_matrix_sync(a, identity, 16);
    rocwmma::load_matrix_sync(b, matrix, 16);
    rocwmma::fill_fragment(c, 0.0f);
    rocwmma::mma_sync(c, a, b, c);
    for (uint32_t slot = 0; slot < 8; ++slot) {
        const uint32_t rc = ds4_indexer_unpack_coordinate(lo, hi, slot);
        mapping[wave * 256u + lane * 8u + slot] = rc;
        if (coords.x[slot] != (float)rc || c.x[slot] != (float)rc) atomicAdd(failures, 1u);
    }
    rocwmma::store_matrix_sync(roundtrip + wave * 256u, c, 16, rocwmma::mem_row_major);
    rocwmma::load_matrix_sync(c, labels, 16, rocwmma::mem_row_major);
    for (uint32_t step = 0; step < 8; ++step) rocwmma::mma_sync(c, a, b, c);
    for (uint32_t slot = 0; slot < 8; ++slot)
        if (c.x[slot] != 9.0f * coords.x[slot]) atomicAdd(failures, 1u);
    rocwmma::store_matrix_sync(roundtrip + 2048u + wave * 256u, c, 16, rocwmma::mem_row_major);
#endif
}

int main() {
    int dev = 0, version = 0;
    HIP(hipGetDevice(&dev));
    hipDeviceProp_t prop{};
    HIP(hipGetDeviceProperties(&prop, dev));
    HIP(hipRuntimeGetVersion(&version));
    require(std::strncmp(prop.gcnArchName, "gfx1151", 7) == 0 &&
            (prop.gcnArchName[7] == '\0' || prop.gcnArchName[7] == ':') && prop.warpSize == 32,
            "native diagnostic needs gfx1151 / wave32");
    constexpr uint32_t guard = 64, poison = 0x7fc12345u;
    std::vector<uint32_t> map(2048 + 2 * guard, poison), output(4096 + 2 * guard, poison);
    uint32_t *dm = nullptr, *dout = nullptr, *dfail = nullptr;
    HIP(hipMalloc(&dm, map.size() * 4));
    HIP(hipMalloc(&dout, output.size() * 4));
    HIP(hipMalloc(&dfail, 4));
    HIP(hipMemcpy(dm, map.data(), map.size() * 4, hipMemcpyHostToDevice));
    HIP(hipMemcpy(dout, output.data(), output.size() * 4, hipMemcpyHostToDevice));
    HIP(hipMemset(dfail, 0, 4));
    diagnose<<<1, 256>>>(dm + guard, reinterpret_cast<float *>(dout + guard), dfail);
    HIP(hipGetLastError());
    HIP(hipDeviceSynchronize());
    uint32_t failures = 0;
    HIP(hipMemcpy(&failures, dfail, 4, hipMemcpyDeviceToHost));
    HIP(hipMemcpy(map.data(), dm, map.size() * 4, hipMemcpyDeviceToHost));
    HIP(hipMemcpy(output.data(), dout, output.size() * 4, hipMemcpyDeviceToHost));
    require(failures == 0, "load/MMA register mapping disagrees");
    for (unsigned i = 0; i < guard; ++i) {
        require(map[i] == poison && map[map.size() - 1 - i] == poison, "mapping guards");
        require(output[i] == poison && output[output.size() - 1 - i] == poison, "output guards");
    }
    for (unsigned wave = 0; wave < 8; ++wave) {
        std::array<unsigned, 256> seen{};
        for (unsigned i = 0; i < 256; ++i) {
            const uint32_t rc = map[guard + wave * 256 + i];
            require(rc < 256, "coordinate range / all lanes written");
            ++seen[rc];
            for (unsigned mode = 0; mode < 2; ++mode) {
                const float expected = (float)i * (mode ? 9.0f : 1.0f);
                uint32_t bits;
                std::memcpy(&bits, &expected, 4);
                require(output[guard + mode * 2048 + wave * 256 + i] == bits,
                        "matrix-space roundtrip after MMA");
            }
        }
        for (unsigned n : seen) require(n == 1, "one register owner per matrix element");
    }
    HIP(hipFree(dm)); HIP(hipFree(dout)); HIP(hipFree(dfail));
    std::printf("PASS native mapping: %s HIP runtime %d; 8 waves, 4096 MMA/store cells\n",
                prop.gcnArchName, version);
}
#endif
