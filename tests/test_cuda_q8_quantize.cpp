// SPDX-License-Identifier: MIT
// Build with a host C++ compiler for the reduction-topology check, or nvcc
// for a bitwise oracle that compiles the production CUDA kernels themselves.
#include <array>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static void require(bool ok, const char *what) {
    if (!ok) { std::fprintf(stderr, "q8 quantize test: %s\n", what); std::exit(1); }
}
static uint32_t rng(uint32_t &seed) { return seed = seed * 1664525u + 1013904223u; }
static uint32_t bits(float v) { uint32_t b; std::memcpy(&b, &v, 4); return b; }
static float sample(uint32_t &seed, size_t i, int pattern) {
    if (pattern == 0) return i & 1 ? -0.f : 0.f;
    if (pattern == 1) return (int(rng(seed) % 8193u) - 4096) / 1024.f;
    if (pattern == 2) return (i % 7 == 0 ? -8.f : (i % 5 == 0 ? 8.f : 0.125f));
    return std::ldexp((int(rng(seed) % 2049u) - 1024) / 1024.f,
                      int(rng(seed) % 161u) - 80);
}
static float cpu_amax(const float *x, unsigned n, bool warp) {
#ifdef DS4_Q8_HOST_AMAX_EXTRACTED
    if (warp) return production_host_amax(x, n);
#endif
    std::array<float, 32> v{};
    for (unsigned i = 0; i < n; ++i) v[i] = std::fabs(x[i]);
    for (unsigned stride = 16; stride; stride >>= 1) {
        // Warp shuffles read the pre-instruction state of every lane; shared
        // memory updates only the active left subtree, followed by a barrier.
        const auto before = v;
        for (unsigned lane = 0; lane < stride; ++lane)
            v[lane] = std::fmax(v[lane], warp ? before[lane + stride] : v[lane + stride]);
    }
    return v[0];
}
static void cpu_topology_test() {
    uint32_t seed = 0x1281ab21;
    size_t count = 0;
    for (unsigned n = 1; n <= 32; ++n) for (int pattern = 0; pattern < 4; ++pattern)
        for (unsigned repeat = 0; repeat < 512; ++repeat) {
            float x[32];
            for (unsigned i = 0; i < 32; ++i) x[i] = sample(seed, i + repeat, pattern);
            const float a = cpu_amax(x, n, false), b = cpu_amax(x, n, true);
            require(bits(a) == bits(b), "host max/reduction tree mismatch");
            const float da = a / 127.f, db = b / 127.f;
            const float ia = da != 0.f ? 1.f / da : 0.f, ib = db != 0.f ? 1.f / db : 0.f;
            require(bits(da) == bits(db), "host scale mismatch");
            for (unsigned i = 0; i < n; ++i)
                require(std::lrintf(x[i] * ia) == std::lrintf(x[i] * ib), "host quant mismatch");
            ++count;
        }
    std::printf("PASS: %zu host reduction/scale/rounding cases (all 1..32 tails).\n", count);
}

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include "../cuda/ds4_q8_quantize.cuh"

struct BlockQ8K { float d; int8_t qs[256]; int16_t bsums[16]; };
static_assert(sizeof(BlockQ8K) == 292, "Q8_K ABI");
static void check(cudaError_t e) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s\n", cudaGetErrorString(e)); std::exit(1); }
}
struct Buffer {
    static constexpr size_t guard = 256;
    unsigned char *base = nullptr;
    size_t bytes, prefix;
    explicit Buffer(size_t n, size_t skew = 0) : bytes(n), prefix(guard + skew) {
        check(cudaMalloc(reinterpret_cast<void **>(&base), prefix + bytes + guard));
        check(cudaMemset(base, 0xa5, prefix + bytes + guard));
    }
    ~Buffer() { cudaFree(base); }
    Buffer(const Buffer &) = delete;
    Buffer &operator=(const Buffer &) = delete;
    template<typename T> T *ptr() { return reinterpret_cast<T *>(base + prefix); }
    std::vector<unsigned char> read() {
        std::vector<unsigned char> all(prefix + bytes + guard);
        check(cudaMemcpy(all.data(), base, all.size(), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < prefix; ++i) require(all[i] == 0xa5, "prefix overwritten");
        for (size_t i = 0; i < guard; ++i)
            require(all[prefix + bytes + i] == 0xa5, "suffix overwritten");
        return {all.begin() + prefix, all.end() - guard};
    }
};
struct Result { std::vector<unsigned char> q, scale, k; };

// mode 0: ordinary rows; 1: packed group slice; 2: dual Q8_K + Q8_0.
static Result run_gpu(const std::vector<float> &input, uint32_t dim,
                      uint32_t rows, bool fast, int mode, bool graph,
                      uint32_t total_groups = 1, uint32_t group0 = 0,
                      uint32_t group_count = 1, uint32_t skew = 0) {
    const uint32_t blocks = (dim + 31u) / 32u;
    const uint32_t packed_rows = mode == 1 ? rows * group_count : rows;
    Buffer x(input.size() * sizeof(float), skew * sizeof(float)),
           q(size_t(packed_rows) * blocks * 32u, skew),
           scale(size_t(packed_rows) * blocks * 4u, skew * sizeof(float)),
           k(mode == 2 ? size_t(rows) * (dim/256u) * sizeof(BlockQ8K) : 4u);
    check(cudaMemcpy(x.ptr<float>(), input.data(), x.bytes, cudaMemcpyHostToDevice));
    cudaStream_t stream;
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    auto launch = [&] {
        if (mode == 0) ds4_cuda_launch_q8_0_quantize(fast, dim3(blocks, rows), stream,
            q.ptr<int8_t>(), scale.ptr<float>(), x.ptr<float>(), dim, blocks);
        else if (mode == 1) ds4_cuda_launch_q8_0_group_slice_quantize(fast,
            dim3(blocks, packed_rows), stream, q.ptr<int8_t>(), scale.ptr<float>(),
            x.ptr<float>(), dim, blocks, total_groups, group0, group_count);
        else ds4_cuda_launch_q8_dual_quantize(fast, dim3(dim/256u, rows), stream,
            k.ptr<BlockQ8K>(), q.ptr<int8_t>(), scale.ptr<float>(), x.ptr<float>(), dim, rows);
        check(cudaGetLastError());
    };
    if (graph) {
        cudaGraph_t captured;
        cudaGraphExec_t executable;
        // Resolve lazy CUDA module loading before entering stream capture.
        launch();
        check(cudaStreamSynchronize(stream));
        check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        launch();
        check(cudaStreamEndCapture(stream, &captured));
        check(cudaGraphInstantiate(&executable, captured, nullptr, nullptr, 0));
        for (int replay = 0; replay < 3; ++replay) {
            check(cudaMemsetAsync(q.ptr<int8_t>(), 0xa5, q.bytes, stream));
            check(cudaMemsetAsync(scale.ptr<float>(), 0xa5, scale.bytes, stream));
            if (mode == 2) check(cudaMemsetAsync(k.ptr<BlockQ8K>(), 0xa5, k.bytes, stream));
            check(cudaGraphLaunch(executable, stream));
        }
        check(cudaStreamSynchronize(stream));
        check(cudaGraphExecDestroy(executable));
        check(cudaGraphDestroy(captured));
    } else { launch(); check(cudaStreamSynchronize(stream)); }
    check(cudaStreamDestroy(stream));
    const auto after = x.read();
    require(!std::memcmp(after.data(), input.data(), x.bytes), "input overwritten");
    Result result{q.read(), scale.read(), mode == 2 ? k.read() : std::vector<unsigned char>{}};
    for (size_t i = 0; i < result.scale.size(); i += 4) {
        uint32_t word; std::memcpy(&word, result.scale.data() + i, 4);
        require(word != 0xa5a5a5a5u && (word & 0x7f800000u) != 0x7f800000u,
                "nonfinite/unwritten scale");
    }
    // The contract always writes zero to every padded tail byte.
    for (uint32_t r = 0; r < packed_rows; ++r)
        for (uint32_t i = dim; i < blocks * 32u; ++i)
            require(result.q[size_t(r) * blocks * 32u + i] == 0, "tail not zero padded");
    return result;
}
static void equal(const Result &a, const Result &b) {
    require(a.q == b.q, "Q8_0 bytes differ");
    require(a.scale == b.scale, "Q8_0 scale bits differ");
    require(a.k == b.k, "dual Q8_K output differs (signed ties/bsums)");
}
static void gpu_test(int device) {
    int devices = 0;
    check(cudaGetDeviceCount(&devices));
    require(devices > 0, "no CUDA GPU (GPU oracle must not silently skip)");
    require(device >= 0 && device < devices, "device ordinal");
    check(cudaSetDevice(device));
    uint32_t seed = 0x827121ab;
    size_t cases = 0;
    for (uint32_t dim : {1u, 17u, 31u, 32u, 33u, 65u, 97u, 129u,
                         161u, 193u, 225u, 255u, 256u, 257u, 4096u, 7168u})
        for (uint32_t rows : {1u, 3u}) for (int pattern = 0; pattern < 4; ++pattern) {
            std::vector<float> x(size_t(dim) * rows);
            for (size_t i = 0; i < x.size(); ++i) x[i] = sample(seed, i, pattern);
            const auto reference = run_gpu(x, dim, rows, false, 0, false);
            equal(reference, run_gpu(x, dim, rows, true, 0, false));
            equal(reference, run_gpu(x, dim, rows, true, 0, true, 1, 0, 1, pattern));
            ++cases;
        }
    for (uint32_t tail = 1; tail <= 32; ++tail) {
        const uint32_t dim = 256 + tail, rows = 3;
        std::vector<float> x(size_t(dim) * rows);
        for (size_t i = 0; i < x.size(); ++i) x[i] = sample(seed, i, tail % 4);
        equal(run_gpu(x, dim, rows, false, 0, false),
              run_gpu(x, dim, rows, true, 0, false, 1, 0, 1, tail % 4));
        ++cases;
    }
    for (uint32_t dim : {17u, 256u, 4096u}) for (uint32_t rows : {1u, 3u})
        for (uint32_t group0 : {0u, 2u}) {
            const uint32_t groups = 8, count = group0 ? 3 : groups;
            std::vector<float> x(size_t(dim) * rows * groups), packed(size_t(dim) * rows * count);
            for (size_t i = 0; i < x.size(); ++i) x[i] = sample(seed, i, 1);
            for (uint32_t r = 0; r < rows; ++r) for (uint32_t g = 0; g < count; ++g)
                std::memcpy(packed.data() + (size_t(r)*count + g)*dim,
                            x.data() + (size_t(r)*groups + group0 + g)*dim, dim * sizeof(float));
            auto reference = run_gpu(packed, dim, rows*count, false, 0, false);
            equal(reference, run_gpu(x, dim, rows, false, 1, false, groups, group0, count));
            equal(reference, run_gpu(x, dim, rows, true, 1, true, groups, group0, count));
            ++cases;
        }
    for (uint32_t dim : {256u, 512u, 4096u, 7168u}) for (uint32_t rows : {1u, 3u})
        for (int pattern = 0; pattern < 4; ++pattern) {
            std::vector<float> x(size_t(dim) * rows);
            for (size_t i = 0; i < x.size(); ++i) x[i] = sample(seed, i, pattern);
            auto old = run_gpu(x, dim, rows, false, 2, false);
            auto candidate = run_gpu(x, dim, rows, true, 2, true);
            equal(old, candidate);
            candidate.k.clear();
            equal(run_gpu(x, dim, rows, false, 0, false), candidate);
            ++cases;
        }
    std::vector<float> x(4096);
    for (size_t i = 0; i < x.size(); ++i) x[i] = sample(seed, i, 2);
    equal(run_gpu(x, 4096, 1, false, 0, false), run_gpu(x, 4096, 1, true, 0, true));
    std::printf("PASS: %zu GPU cases, bitwise legacy/candidate; guards, group slices, dual and graph replay.\n", ++cases);
}

#endif

int main(int argc, char **argv) {
    require(argc <= 2, "usage: test_cuda_q8_quantize [device-ordinal]");
    cpu_topology_test();
#ifdef __CUDACC__
    gpu_test(argc == 2 ? std::atoi(argv[1]) : 0);
#else
    (void)argv;
#ifdef DS4_Q8_HOST_AMAX_EXTRACTED
    production_policy_test();
    std::puts("Host production shuffle simulation; CUDA scheduling/device math remain unverified.");
#else
    std::puts("Topology-only host check. Use tests/test_cuda_q8_quantize.py to test the production helper.");
#endif
#endif
}
