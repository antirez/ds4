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
#include "../cuda/ds4_q8_prefill_layout.h"

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

static void cpu_prefill_test() {
    namespace layout = ds4_cuda_q8_prefill;
    size_t policies = 0, tiles = 0;
    for (uint64_t n : std::array<uint64_t, 13>{0,1,2,8,255,256,257,2048,
                                             4096,8192,8193,65535,UINT64_MAX})
    for (uint64_t k : std::array<uint64_t, 11>{0,1,32,255,256,257,4096,
                                             7168,16384,16385,UINT64_MAX})
    for (unsigned flags = 0; flags < 16; ++flags) {
        const bool device = flags & 1, quality = flags & 2;
        const bool disable_warp = flags & 4, disable_prefill = flags & 8;
        const bool expected = n >= 256 && n <= 8192 && k >= 256 && k <= 16384 &&
            device && !quality && !disable_warp && !disable_prefill;
        require(layout::select(n, k, device, quality, disable_warp, disable_prefill)
                == expected, "prefill admission/rollback");
        ++policies;
    }
    // Output-A packs groups into rows: a single token with 256 groups is
    // still decode, not prefill. The caller must pass tokens, not grid.y.
    require(!layout::select(1, 4096, true, false, false, false), "packed decode");
    uint32_t seed = 0xc82372;
    for (unsigned dim = 1; dim <= 16384; ++dim) {
        // Every 1..32 byte tail and 1..8 warp tail, plus realistic large K.
        if (dim > 1025 && dim != 4096 && dim != 7168 && dim != 16384) continue;
        const uint64_t blocks = (dim + 31u) / 32u;
        std::vector<float> x(dim);
        for (unsigned i = 0; i < dim; ++i) x[i] = sample(seed, i, dim % 4);
        std::vector<unsigned> reads(dim), writes(blocks * 32u), scales(blocks);
        for (uint64_t tile = 0; tile < layout::tiles(blocks); ++tile) {
            float partial[layout::threads] = {};
            for (unsigned tid = 0; tid < layout::threads; ++tid) {
                const uint64_t b = layout::quant_block(tile, tid), i = b * 32 + tid % 32;
                if (b < blocks && i < dim) { partial[tid] = std::fabs(x[i]); ++reads[i]; }
            }
            for (unsigned stride = 16; stride; stride >>= 1)
                for (unsigned tid = 0; tid < layout::threads; ++tid)
                    if (tid % 32 < stride) {
                        require(tid / 32 == (tid + stride) / 32, "cross-warp reduction");
                        partial[tid] = std::fmax(partial[tid], partial[tid + stride]);
                    }
            for (unsigned tid = 0; tid < layout::threads; ++tid) {
                const uint64_t b = layout::quant_block(tile, tid), i = b * 32 + tid % 32;
                if (b >= blocks) continue;
                const auto valid = static_cast<unsigned>(std::min<uint64_t>(32, dim - b*32));
                require(bits(partial[tid - tid % 32]) == bits(cpu_amax(x.data()+b*32, valid, false)),
                        "tiled reduction differs from legacy");
                ++writes[i];
                if (tid % 32 == 0) ++scales[b];
            }
            ++tiles;
        }
        for (unsigned c : reads) require(c == 1, "input ownership");
        for (unsigned c : writes) require(c == 1, "output/padding ownership");
        for (unsigned c : scales) require(c == 1, "scale ownership");
    }
    std::printf("PASS: %zu prefill policies, %zu tiled mappings/reductions; no cross-warp reads.\n",
                policies, tiles);
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
                      uint32_t group_count = 1, bool tiled = false,
                      uint32_t skew = 0) {
    const uint32_t blocks = (dim + 31u) / 32u;
    const uint32_t packed_rows = mode == 1 ? rows * group_count : rows;
    require(!tiled || mode == 0, "tiled mode is ordinary/contiguous packed rows only");
    Buffer x(input.size() * sizeof(float), skew * sizeof(float)),
           q(size_t(packed_rows) * blocks * 32u, skew),
           scale(size_t(packed_rows) * blocks * 4u, skew * sizeof(float)),
           k(mode == 2 ? size_t(rows) * (dim/256u) * sizeof(BlockQ8K) : 4u);
    check(cudaMemcpy(x.ptr<float>(), input.data(), x.bytes, cudaMemcpyHostToDevice));
    cudaStream_t stream;
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    auto launch = [&] {
        if (mode == 0) ds4_cuda_launch_q8_0_quantize(fast, dim3(blocks, rows), stream,
            q.ptr<int8_t>(), scale.ptr<float>(), x.ptr<float>(), dim, blocks, tiled);
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
static void gpu_test() {
    int devices = 0;
    check(cudaGetDeviceCount(&devices));
    require(devices > 0, "no CUDA GPU (GPU oracle must not silently skip)");
    uint32_t seed = 0x827121ab;
    size_t cases = 0;
    for (uint32_t dim : {1u, 17u, 31u, 32u, 33u, 65u, 97u, 129u,
                         161u, 193u, 225u, 255u, 256u, 257u, 4096u, 7168u})
        for (uint32_t rows : {1u, 3u}) for (int pattern = 0; pattern < 4; ++pattern) {
            std::vector<float> x(size_t(dim) * rows);
            for (size_t i = 0; i < x.size(); ++i) x[i] = sample(seed, i, pattern);
            const auto reference = run_gpu(x, dim, rows, false, 0, false);
            equal(reference, run_gpu(x, dim, rows, true, 0, false));
            equal(reference, run_gpu(x, dim, rows, true, 0, true, 1, 0, 1, true, pattern));
            ++cases;
        }
    for (uint32_t tail = 1; tail <= 32; ++tail) {
        const uint32_t dim = 256 + tail, rows = 3;
        std::vector<float> x(size_t(dim) * rows);
        for (size_t i = 0; i < x.size(); ++i) x[i] = sample(seed, i, tail % 4);
        equal(run_gpu(x, dim, rows, false, 0, false),
              run_gpu(x, dim, rows, true, 0, false, 1, 0, 1, true, tail % 4));
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

static void gpu_prefill_test() {
    uint32_t seed = 0x382192ab;
    size_t cases = 0;
    // Test production batch sizes plus tails and contiguous group-packed rows.
    for (const auto shape : {std::array<uint32_t, 3>{257, 256, 1},
                            {4096, 256, 8}, {4096, 2048, 1}, {7168, 4096, 1},
                            {16384, 256, 1}, {512, 8192, 1}}) {
        const uint32_t dim = shape[0], tokens = shape[1], groups = shape[2];
        const uint32_t rows = tokens * groups;
        require(ds4_cuda_q8_prefill::select(tokens, dim, true, false, false, false),
                "prefill GPU fixture must be admitted");
        std::vector<float> x(size_t(rows) * dim);
        for (size_t i = 0; i < x.size(); ++i) x[i] = sample(seed, i, 1);
        // Explicit ties at +/- 0.5, 1.5, 2.5 after scaling, and subnormals.
        for (size_t i = 0; i + 32 <= x.size(); i += 32 * 17) {
            x[i] = 127.f; x[i+1] = -127.f;
            x[i+2] = 0.5f; x[i+3] = -0.5f; x[i+4] = 1.5f; x[i+5] = 2.5f;
            x[i+6] = -0.f; x[i+7] = std::ldexp(1.f, -140);
        }
        const auto reference = run_gpu(x, dim, rows, false, 0, false);
        equal(reference, run_gpu(x, dim, rows, true, 0, false));
        equal(reference, run_gpu(x, dim, rows, true, 0, false, 1, 0, 1, true));
        equal(reference, run_gpu(x, dim, rows, true, 0, true, 1, 0, 1, true, 1));
        ++cases;
    }
    std::printf("PASS: %zu large prefill shapes (256..8192 logical tokens), bitwise three-way parity.\n", cases);
}

static void gpu_prefill_bench() {
    int device = 0;
    cudaDeviceProp prop{};
    check(cudaGetDevice(&device));
    check(cudaGetDeviceProperties(&prop, device));
    require(prop.major == 12 && prop.minor == 1 && prop.integrated,
            "benchmark requires a GB10: other GPUs are outside production admission");
    constexpr int repeats = 16, samples = 12;
    const char *names[] = {"legacy-shared-32", "decode-shuffle-32", "prefill-shared-256"};
    const int orders[6][3] = {{0,1,2}, {2,1,0}, {1,2,0}, {0,2,1}, {2,0,1}, {1,0,2}};
    std::printf("GPU %s: CUDA-event graph replay, %d samples x %d independent quantizations;\n"
                "input is immutable, resets/readbacks excluded, three-way bitwise verification.\n",
                prop.name, samples, repeats);
    std::puts("kind,tokens,groups,K,variant,ctas,median_us,min_us,max_us");
    uint32_t seed = 0xf81a937b;
    for (const auto shape : {std::array<uint32_t, 3>{4096, 256, 1},
                            {4096, 2048, 1}, {7168, 4096, 1}, {4096, 256, 8}}) {
        const uint32_t dim = shape[0], tokens = shape[1], groups = shape[2];
        const uint32_t rows = tokens * groups, blocks = (dim + 31u) / 32u;
        require(ds4_cuda_q8_prefill::select(tokens, dim, true, false, false, false),
                "benchmark fixture must be admitted");
        std::vector<float> input(size_t(rows) * dim);
        for (size_t i = 0; i < input.size(); ++i) input[i] = sample(seed, i, 1);
        Buffer x(input.size() * sizeof(float)), q(size_t(rows) * blocks * 32u),
               scale(size_t(rows) * blocks * sizeof(float));
        check(cudaMemcpy(x.ptr<float>(), input.data(), x.bytes, cudaMemcpyHostToDevice));
        cudaStream_t stream;
        cudaEvent_t begin, end;
        check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        check(cudaEventCreate(&begin));
        check(cudaEventCreate(&end));
        auto launch = [&](int variant) {
            ds4_cuda_launch_q8_0_quantize(variant != 0, dim3(blocks, rows), stream,
                q.ptr<int8_t>(), scale.ptr<float>(), x.ptr<float>(), dim, blocks, variant == 2);
            check(cudaGetLastError());
        };
        launch(0);
        check(cudaStreamSynchronize(stream));
        const auto ref_q = q.read(), ref_scale = scale.read();
        cudaGraph_t graphs[3]{};
        cudaGraphExec_t execs[3]{};
        for (int variant = 0; variant < 3; ++variant) {
            for (int warmup = 0; warmup < 8; ++warmup) launch(variant);
            check(cudaStreamSynchronize(stream));
            require(q.read() == ref_q && scale.read() == ref_scale, "benchmark warmup parity");
            check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
            for (int i = 0; i < repeats; ++i) launch(variant);
            check(cudaStreamEndCapture(stream, &graphs[variant]));
            check(cudaGraphInstantiate(&execs[variant], graphs[variant], nullptr, nullptr, 0));
            // Exclude initial graph upload from the timed samples, too.
            check(cudaGraphLaunch(execs[variant], stream));
            check(cudaStreamSynchronize(stream));
        }
        std::array<std::vector<float>, 3> times;
        for (int sample_index = 0; sample_index < samples; ++sample_index)
            for (int order = 0; order < 3; ++order) {
                const int variant = orders[sample_index % 6][order];
                check(cudaMemsetAsync(q.ptr<int8_t>(), 0xa5, q.bytes, stream));
                check(cudaMemsetAsync(scale.ptr<float>(), 0xa5, scale.bytes, stream));
                check(cudaEventRecord(begin, stream));
                check(cudaGraphLaunch(execs[variant], stream));
                check(cudaEventRecord(end, stream));
                check(cudaEventSynchronize(end));
                float ms = 0;
                check(cudaEventElapsedTime(&ms, begin, end));
                require(ms > 0.f && std::isfinite(ms), "invalid benchmark timing");
                times[variant].push_back(ms * 1000.f / repeats);
                require(q.read() == ref_q && scale.read() == ref_scale, "benchmark sample parity/guards");
            }
        const auto after = x.read();
        require(!std::memcmp(after.data(), input.data(), x.bytes), "benchmark input overwritten");
        for (int variant = 0; variant < 3; ++variant) {
            auto &t = times[variant];
            std::sort(t.begin(), t.end());
            const uint64_t ctas = uint64_t(rows) *
                (variant == 2 ? ds4_cuda_q8_prefill::tiles(blocks) : blocks);
            std::printf("quantize,%u,%u,%u,%s,%llu,%.3f,%.3f,%.3f\n", tokens, groups, dim,
                        names[variant], (unsigned long long)ctas,
                        (t[samples/2-1] + t[samples/2]) * 0.5f, t.front(), t.back());
            check(cudaGraphExecDestroy(execs[variant]));
            check(cudaGraphDestroy(graphs[variant]));
        }
        check(cudaEventDestroy(begin));
        check(cudaEventDestroy(end));
        check(cudaStreamDestroy(stream));
    }
    std::puts("Kernel-only measurements: not model prefill/decode TPS or proof of closing the Q4/Q8 gap.");
}
#endif

int main(int argc, char **argv) {
    const bool bench = argc == 2 && std::strcmp(argv[1], "--bench") == 0;
    require(argc == 1 || bench, "usage: test_cuda_q8_quantize [--bench]");
#ifndef __CUDACC__
    require(!bench, "--bench requires the CUDA build and a GPU, not a host-only check");
#endif
    cpu_topology_test();
    cpu_prefill_test();
#ifdef __CUDACC__
    gpu_test();
    gpu_prefill_test();
    if (bench) gpu_prefill_bench();
#else
    std::puts("Host-only check: CUDA kernels NOT compiled or executed. Run make test-cuda-q8-quantize on a CUDA host.");
#endif
}
