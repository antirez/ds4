// SPDX-License-Identifier: MIT
// Host: shared admission/sanitizer checks. nvcc: production MMVQ/API/graph oracle.
#ifdef __CUDACC__
#include <cuda_runtime.h>
#include "../cuda/mmq/ds4_mmq.h"
#endif
#include "../cuda/mmq/ds4_q4_mmvq_epilogue.h"
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static void require(bool ok, const char *message) {
    if (!ok) { std::fprintf(stderr, "Q4 epilogue: %s\n", message); std::exit(1); }
}
static uint32_t random_u32(uint32_t &state) { return state = state*1664525u + 1013904223u; }
static void check_bits(uint32_t bits) {
    float v; std::memcpy(&v, &bits, sizeof(v));
    const uint32_t expected = std::isfinite(v) ? bits : 0u;
    require(ds4_q4_mmvq_sanitize_bits(bits) == expected, "sanitizer bit mismatch");
}
static void host_tests() {
    const uint32_t special[] = {0u, 0x80000000u, 1u, 0x80000001u,
        0x007fffffu, 0x807fffffu, 0x00800000u, 0x80800000u,
        0x7f7fffffu, 0xff7fffffu, 0x7f800000u, 0xff800000u,
        0x7fc00000u, 0xffc00000u, 0x7f800001u, 0xffffffffu};
    for (uint32_t bits : special) check_bits(bits);
    uint32_t seed = 127;
    constexpr unsigned random_patterns = 1u << 22;
    for (unsigned i = 0; i < random_patterns; ++i) check_bits(random_u32(seed));

    size_t shapes = 0;
    const int ks[] = {0, 1, 255, 256, 512, 1024, 1536, 1792, 2048, 4096, 8192};
    for (int m = 0; m <= 1025; ++m) for (int n = 0; n <= 9; ++n) for (int k : ks) {
        const bool eligible = ds4_q4_mmvq_epilogue_shape_ok(m, n, k);
        const bool expected = m > 0 && m % 4 == 0 && n >= 1 && n <= 8 && k > 0 && k % 256 == 0;
        require(eligible == expected, "shape admission mismatch");
        if (eligible) {
            // Canonical NVIDIA: N=1 has four rows for K<2048, otherwise one;
            // every N=2..8 variant has two. Check all token columns as well.
            const int rows_per_block = n == 1 ? (k < 2048 ? 4 : 1) : 2;
            std::vector<unsigned> writes(size_t(m)*n, 0);
            for (int block = 0; block < m / rows_per_block; ++block)
                for (int j = 0; j < n; ++j)
                    for (int lane = 0; lane < rows_per_block; ++lane)
                        ++writes[size_t(j)*m + block*rows_per_block + lane];
            for (unsigned count : writes) require(count == 1, "token/row coverage mismatch");
        }
        ++shapes;
    }
    require(!ds4_q4_mmvq_epilogue_shape_ok(-4, 1, 256), "negative M admitted");
    require(!ds4_q4_mmvq_epilogue_shape_ok(4, 1, -256), "negative K admitted");
    require(!ds4_q4_mmvq_epilogue_shape_ok(INT_MAX-3, 1, 512), "block index overflow admitted");
    require(ds4_q4_mmvq_epilogue_shape_ok(INT_MAX-3, 1, 256), "valid index bound rejected");
    require(!ds4_q4_mmvq_epilogue_shape_ok(4, -1, 256), "negative N admitted");
    for (int n = 1; n <= 8; ++n) {
        const uint64_t output_bound = uint64_t(UINT32_MAX) / n;
        const uint64_t bound = output_bound < uint64_t(INT_MAX) ? output_bound : uint64_t(INT_MAX);
        const int aligned = int(bound / 4u * 4u);
        require(ds4_q4_mmvq_epilogue_shape_ok(aligned, n, 256), "valid batch output bound rejected");
        if (aligned <= INT_MAX - 4)
            require(!ds4_q4_mmvq_epilogue_shape_ok(aligned+4, n, 256), "batch output offset overflow admitted");
    }
    std::printf("PASS: %zu sanitizer patterns, %zu shape/coverage cases, index bounds.\n",
        random_patterns + sizeof(special) / sizeof(special[0]), shapes);

    size_t grouped_shapes = 0;
    for (int m = 0; m <= 65; ++m) for (int k : ks)
    for (int tokens = 0; tokens <= 9; ++tokens) for (int groups = 0; groups <= 17; ++groups) {
        const int64_t stride = ((int64_t(k) + 511) / 512) * 16;
        const bool expected = m > 0 && m % 4 == 0 && k > 0 && k % 256 == 0 &&
                              tokens >= 1 && tokens <= 8 && groups >= 1 && groups <= 16;
        require(ds4_q4_mmvq_grouped_shape_ok(m, k, tokens, groups, stride) == expected,
                "grouped shape admission mismatch");
        if (expected) {
            for (int t = 0; t < tokens; ++t) for (int g = 0; g < groups; ++g) {
                const int channel = t * groups + g;
                require(channel % groups == g, "cyclic weight group mismatch");
                require(channel % (tokens * groups) == channel, "activation channel mismatch");
            }
        }
        ++grouped_shapes;
    }
    require(!ds4_q4_mmvq_grouped_shape_ok(4, 256, -1, 8, 16), "negative tokens admitted");
    require(!ds4_q4_mmvq_grouped_shape_ok(4, 256, 1, -1, 16), "negative groups admitted");
    require(!ds4_q4_mmvq_grouped_shape_ok(INT_MAX-3, 256, 1, 2, 16), "grouped weight index overflow admitted");
    require(!ds4_q4_mmvq_grouped_shape_ok(1<<29, 256, 8, 1, 16), "grouped output offset overflow admitted");
    require(ds4_q4_mmvq_grouped_shape_ok((1<<29)-4, 256, 8, 1, 16), "valid output offset bound rejected");
    require(!ds4_q4_mmvq_grouped_shape_ok(4, 256, 8, 16, int64_t(UINT32_MAX)/128+1),
            "grouped Q8 offset overflow admitted");
    require(ds4_q4_mmvq_grouped_shape_ok(4, 256, 8, 16, int64_t(UINT32_MAX)/128),
            "valid Q8 offset bound rejected");
    require(!ds4_q4_mmvq_grouped_shape_ok(4, 256, 1, 1, 7), "short Q8 stride admitted");
    require(!ds4_q4_mmvq_grouped_shape_ok(4, 256, 1, 1, INT64_MAX), "Q8 stride overflow admitted");
    std::printf("PASS: %zu grouped shape/channel cases and offset bounds.\n", grouped_shapes);
}

#ifdef __CUDACC__
static void check(cudaError_t e) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s\n", cudaGetErrorString(e)); std::exit(1); }
}
struct ScopedEnv {
    const char *name;
    bool present;
    std::string original;
    explicit ScopedEnv(const char *n) : name(n), present(std::getenv(n) != nullptr),
        original(present ? std::getenv(n) : "") {}
    ScopedEnv(const ScopedEnv &) = delete;
    ScopedEnv &operator=(const ScopedEnv &) = delete;
    void set(const char *value) {
        require((value ? setenv(name, value, 1) : unsetenv(name)) == 0, "environment update failed");
    }
    ~ScopedEnv() { if (present) setenv(name, original.c_str(), 1); else unsetenv(name); }
};
struct Buffer {
    static constexpr size_t guard = 256;
    unsigned char *base = nullptr;
    size_t bytes;
    explicit Buffer(size_t n) : bytes(n) {
        check(cudaMalloc(reinterpret_cast<void **>(&base), bytes + 2*guard));
        check(cudaMemset(base, 0xa5, bytes + 2*guard));
    }
    ~Buffer() { cudaFree(base); }
    Buffer(const Buffer &) = delete;
    Buffer &operator=(const Buffer &) = delete;
    template<typename T> T *ptr() { return reinterpret_cast<T *>(base + guard); }
    std::vector<unsigned char> read() {
        std::vector<unsigned char> all(bytes + 2*guard);
        check(cudaMemcpy(all.data(), base, all.size(), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < guard; ++i)
            require(all[i] == 0xa5 && all[guard + bytes + i] == 0xa5, "guard overwritten");
        return {all.begin() + guard, all.end() - guard};
    }
};
struct Q4Block { uint16_t d, dmin; uint8_t scales[12], qs[128]; };
static_assert(sizeof(Q4Block) == 144, "Q4_K ABI");
struct GraphCounts { size_t kernels = 0, memsets = 0; };
struct EpilogueArm { const char *rollback_value; bool candidate; };
static const EpilogueArm epilogue_arms[] = {{"1", false}, {nullptr, true}, {"0", false}, {"", false}};
static GraphCounts graph_counts(cudaGraph_t graph) {
    size_t count = 0;
    check(cudaGraphGetNodes(graph, nullptr, &count));
    std::vector<cudaGraphNode_t> nodes(count);
    check(cudaGraphGetNodes(graph, nodes.data(), &count));
    GraphCounts result;
    for (cudaGraphNode_t node : nodes) {
        cudaGraphNodeType type;
        check(cudaGraphNodeGetType(node, &type));
        result.kernels += type == cudaGraphNodeTypeKernel;
        result.memsets += type == cudaGraphNodeTypeMemset;
    }
    return result;
}

static void gpu_case(int m0, int m1, int n, int k, bool nonfinite, bool pair) {
    uint32_t seed = 65537u + m0 + m1 + n + k;
    // Legacy small-K MMVQ reads a complete cohort even for odd M. Pad only
    // physical weights; true output dimensions and their guards remain exact.
    std::vector<Q4Block> weights[2] = {
        std::vector<Q4Block>(size_t((m0+3)/4*4)*(k/256)),
        std::vector<Q4Block>(size_t((m1+3)/4*4)*(k/256))};
    for (auto &matrix : weights) for (size_t i = 0; i < matrix.size(); ++i) {
        auto &b = matrix[i];
        b.d = 0x2400u + (random_u32(seed) & 0x3ffu);
        b.dmin = i % 7 ? 0x1c00u + (random_u32(seed) & 0x3ffu) : 0u;
        if (nonfinite) {
            const uint16_t values[] = {0x7c00, 0xfc00, 0x7e00, 0x7c01};
            b.d = values[(i/(k/256)) % 4];
        }
        for (auto &v : b.scales) v = random_u32(seed) >> 24;
        for (auto &v : b.qs) v = random_u32(seed) >> 24;
    }
    Buffer w0(weights[0].size()*sizeof(Q4Block)), w1(weights[1].size()*sizeof(Q4Block)),
        x(size_t(n)*k*sizeof(float)), y0(size_t(m0)*n*sizeof(float)), y1(size_t(m1)*n*sizeof(float));
    check(cudaMemcpy(w0.ptr<void>(), weights[0].data(), w0.bytes, cudaMemcpyHostToDevice));
    check(cudaMemcpy(w1.ptr<void>(), weights[1].data(), w1.bytes, cudaMemcpyHostToDevice));
    cudaStream_t stream;
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    ScopedEnv rollback("DS4_CUDA_DISABLE_Q4_MMVQ_EPILOGUE");
    auto launch = [&] {
        if (pair) {
            require(ds4_mmq_q4_K_dense_pair_vec(w0.ptr<void>(), w1.ptr<void>(), x.ptr<float>(),
                y0.ptr<float>(), y1.ptr<float>(), m0, m1, n, k, stream) == 0, "pair enqueue failed");
        } else {
            require(ds4_mmq_q4_K_dense_vec(w0.ptr<void>(), x.ptr<float>(), y0.ptr<float>(),
                m0, n, k, stream) == 0, "first single enqueue failed");
            require(ds4_mmq_q4_K_dense_vec(w1.ptr<void>(), x.ptr<float>(), y1.ptr<float>(),
                m1, n, k, stream) == 0, "second single enqueue failed");
        }
        check(cudaGetLastError());
    };
    auto poison = [&] {
        check(cudaMemsetAsync(y0.ptr<void>(), 0xff, y0.bytes, stream));
        check(cudaMemsetAsync(y1.ptr<void>(), 0xff, y1.bytes, stream));
    };
    // Start with the legacy reference. Every defined value must disable the
    // candidate, including "0" and an empty string (presence-based opt-out).
    const size_t eligible_legs = size_t(ds4_q4_mmvq_epilogue_shape_ok(m0, n, k)) +
                                 size_t(ds4_q4_mmvq_epilogue_shape_ok(m1, n, k));
    for (int repeat = 0; repeat < 3; ++repeat) {
        std::vector<float> input(size_t(n)*k);
        for (size_t i = 0; i < input.size(); ++i) {
            input[i] = repeat == 0 ? (i & 1 ? -0.f : 0.f) :
                (int(random_u32(seed) % 8193u) - 4096) / 1024.f;
        }
        check(cudaMemcpyAsync(x.ptr<void>(), input.data(), x.bytes, cudaMemcpyHostToDevice, stream));
        std::array<std::vector<unsigned char>, 2> reference;
        GraphCounts reference_counts;
        bool first_arm = true;
        for (const EpilogueArm &arm : epilogue_arms) {
            rollback.set(arm.rollback_value);
            poison(); launch(); check(cudaStreamSynchronize(stream));
            const std::array<std::vector<unsigned char>, 2> output = {y0.read(), y1.read()};
            if (first_arm) reference = output;
            else require(output == reference, "direct output not bitwise equal");
            for (const auto &v : output) for (size_t i = 0; i < v.size(); i += sizeof(uint32_t)) {
                uint32_t bits; std::memcpy(&bits, v.data()+i, sizeof(bits));
                require((bits & 0x7f800000u) != 0x7f800000u, "nonfinite/unwritten result");
                if (nonfinite) require(bits == 0u, "nonfinite result was not sanitized to +0");
            }

            // Warmup above resolves lazy module loading and pool allocation.
            // Capture contains production work only, not the output poison.
            cudaGraph_t graph;
            cudaGraphExec_t executable;
            check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
            launch();
            check(cudaStreamEndCapture(stream, &graph));
            const GraphCounts counts = graph_counts(graph);
            if (first_arm) reference_counts = counts;
            const size_t removed = arm.candidate ? eligible_legs : 0;
            require(reference_counts.kernels == counts.kernels + removed &&
                    reference_counts.memsets == counts.memsets + removed,
                    arm.candidate ? "candidate graph did not remove the expected sanitizer/pre-clear nodes" :
                                    "rollback did not restore the graph");
            check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
            // Opt-out changes must not rewrite an already captured batch.
            rollback.set(arm.candidate ? "1" : nullptr);
            for (int replay = 0; replay < 3; ++replay) {
                poison(); check(cudaGraphLaunch(executable, stream));
            }
            check(cudaStreamSynchronize(stream));
            require(y0.read() == reference[0] && y1.read() == reference[1], "graph output not bitwise equal");
            check(cudaGraphExecDestroy(executable));
            check(cudaGraphDestroy(graph));
            first_arm = false;
        }
        const auto after_x = x.read();
        require(!std::memcmp(after_x.data(), input.data(), x.bytes), "input mutated");
    }
    const auto after_w0 = w0.read(), after_w1 = w1.read();
    require(!std::memcmp(after_w0.data(), weights[0].data(), w0.bytes) &&
            !std::memcmp(after_w1.data(), weights[1].data(), w1.bytes), "weights mutated");
    check(cudaStreamDestroy(stream));
    std::printf("PASS: %s M=%d+%d N=%d K=%d %s (direct/graph/rollback/canaries).\n",
        pair ? "pair" : "single", m0, m1, n, k, nonfinite ? "NaN/Inf" : "finite");
}
static void gpu_grouped_case(int m, int k, int tokens, int groups, bool nonfinite) {
    uint32_t seed = 9137u + m + k + tokens + groups;
    const size_t row_blocks = size_t(k) / 256;
    // Extra physical rows only protect the legacy small-K tail loader. Each
    // logical group still has exactly M rows; there is no inter-group padding.
    std::vector<Q4Block> weights((size_t(groups)*m + 3)*row_blocks);
    for (size_t i = 0; i < weights.size(); ++i) {
        auto &b = weights[i];
        b.d = 0x2000u + (random_u32(seed) & 0x7ffu);
        b.dmin = i % 5 ? 0x1800u + (random_u32(seed) & 0x3ffu) : 0u;
        if (nonfinite) {
            const uint16_t values[] = {0x7c00, 0xfc00, 0x7e00, 0x7c01};
            b.d = values[(i / row_blocks) % 4];
        }
        for (auto &v : b.scales) v = random_u32(seed) >> 24;
        for (auto &v : b.qs) v = random_u32(seed) >> 24;
    }
    const size_t channels = size_t(tokens)*groups;
    const size_t padded_k = (size_t(k) + 511) / 512 * 512;
    const size_t q8_bytes = channels * (padded_k / 32) * 36; // block_q8_1 ABI
    const size_t ids_offset = (q8_bytes + 15) / 16 * 16;
    const size_t scratch_bytes = ids_offset + channels*sizeof(int32_t);
    Buffer w(weights.size()*sizeof(Q4Block)), x(channels*k*sizeof(float)),
        ref(channels*m*sizeof(float)), out(ref.bytes), scratch(scratch_bytes);
    check(cudaMemcpy(w.ptr<void>(), weights.data(), w.bytes, cudaMemcpyHostToDevice));
    cudaStream_t stream;
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    ds4_mmq_set_aligned_q81_scratch(scratch.ptr<void>(), scratch.bytes);
    ScopedEnv rollback("DS4_CUDA_DISABLE_Q4_GROUPED_MMVQ_FUSION");
    auto enqueue = [&] {
        return tokens == 1
            ? ds4_mmq_q4_K_grouped_vec(w.ptr<void>(), x.ptr<float>(), out.ptr<float>(), m, k, groups, stream)
            : ds4_mmq_q4_K_grouped_batch_vec(w.ptr<void>(), x.ptr<float>(), out.ptr<float>(), m, k, tokens, groups, stream);
    };
    auto launch = [&] { require(enqueue() == 0, "grouped enqueue failed"); check(cudaGetLastError()); };
    auto poison = [&] {
        check(cudaMemsetAsync(out.ptr<void>(), 0xff, out.bytes, stream));
        check(cudaMemsetAsync(scratch.ptr<void>(), 0xa5, scratch.bytes, stream));
    };
    const bool eligible = ds4_q4_mmvq_grouped_shape_ok(m, k, tokens, groups, padded_k / 32);
    for (int repeat = 0; repeat < 3; ++repeat) {
        std::vector<float> input(channels*k);
        for (size_t i = 0; i < input.size(); ++i)
            input[i] = repeat == 0 ? (i & 1 ? -0.f : 0.f) :
                (int(random_u32(seed) % 8193u) - 4096) / 1024.f;
        check(cudaMemcpyAsync(x.ptr<void>(), input.data(), x.bytes, cudaMemcpyHostToDevice, stream));
        // Independent per-(token, group) reference, with dense epilogue fusion
        // disabled by the suite. Different rows/groups/tokens carry distinct data.
        for (int t = 0; t < tokens; ++t) for (int g = 0; g < groups; ++g) {
            const size_t channel = size_t(t)*groups + g;
            require(ds4_mmq_q4_K_dense_vec(
                w.ptr<Q4Block>() + size_t(g)*m*row_blocks, x.ptr<float>() + channel*k,
                ref.ptr<float>() + channel*m, m, 1, k, stream) == 0, "grouped dense reference failed");
        }
        check(cudaStreamSynchronize(stream));
        const auto reference = ref.read();
        std::vector<unsigned char> reference_q8;
        GraphCounts reference_counts;
        bool first_arm = true;
        for (const EpilogueArm &arm : epilogue_arms) {
            rollback.set(arm.rollback_value);
            // Both arms keep the original scratch-capacity/pre-enqueue contract.
            ds4_mmq_set_aligned_q81_scratch(scratch.ptr<void>(), scratch.bytes - 1);
            require(enqueue() == DS4_MMQ_NOT_APPLICABLE, "short grouped scratch accepted");
            ds4_mmq_set_aligned_q81_scratch(scratch.ptr<void>(), scratch.bytes);
            poison(); launch(); check(cudaStreamSynchronize(stream));
            const auto result = out.read();
            require(result == reference, "grouped direct output not bitwise equal");
            for (size_t i = 0; i < result.size(); i += sizeof(uint32_t)) {
                uint32_t bits; std::memcpy(&bits, result.data()+i, sizeof(bits));
                require((bits & 0x7f800000u) != 0x7f800000u, "grouped nonfinite/unwritten output");
                if (nonfinite) require(bits == 0u, "grouped nonfinite result not sanitized to +0");
            }
            const auto arena = scratch.read();
            const std::vector<unsigned char> q8(arena.begin(), arena.begin() + q8_bytes);
            if (first_arm) reference_q8 = q8;
            else require(q8 == reference_q8, "grouped Q8_1 bytes changed");
            if (arm.candidate && eligible)
                for (size_t i = ids_offset; i < arena.size(); ++i)
                    require(arena[i] == 0xa5, "fused grouped path wrote the unused ids table");

            cudaGraph_t graph;
            cudaGraphExec_t executable;
            check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
            launch();
            check(cudaStreamEndCapture(stream, &graph));
            const GraphCounts counts = graph_counts(graph);
            if (first_arm) reference_counts = counts;
            const size_t removed = arm.candidate && eligible ? 1 : 0;
            require(reference_counts.kernels == counts.kernels + 2*removed &&
                    reference_counts.memsets == counts.memsets + removed,
                    "grouped graph did not remove/restore ids, sanitizer and output clear");
            check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
            // A changed environment must not change an already captured graph.
            rollback.set(arm.candidate ? "1" : nullptr);
            for (int replay = 0; replay < 3; ++replay) {
                poison(); check(cudaGraphLaunch(executable, stream));
                check(cudaStreamSynchronize(stream));
                require(out.read() == reference, "grouped graph output not bitwise equal");
            }
            check(cudaGraphExecDestroy(executable));
            check(cudaGraphDestroy(graph));
            first_arm = false;
        }
        const auto after_x = x.read();
        require(!std::memcmp(after_x.data(), input.data(), x.bytes), "grouped input mutated");
    }
    const auto after_w = w.read();
    require(!std::memcmp(after_w.data(), weights.data(), w.bytes), "grouped weights mutated");
    (void)scratch.read();
    ds4_mmq_set_aligned_q81_scratch(nullptr, 0);
    check(cudaStreamDestroy(stream));
    std::printf("PASS: grouped M=%d K=%d tokens=%d groups=%d %s (bits/Q8/graph/rollback/guards).\n",
        m, k, tokens, groups, nonfinite ? "NaN/Inf" : "finite");
}

static void gpu_grouped_tests() {
    ScopedEnv global_off("DS4_CUDA_NO_Q4_GB10_FAST"); global_off.set(nullptr);
    ScopedEnv grouped_off("DS4_CUDA_NO_Q4_GROUPED_ATTN_A"); grouped_off.set(nullptr);
    ScopedEnv batch_off("DS4_CUDA_NO_Q4_GROUPED_ATTN_A_BATCH"); batch_off.set(nullptr);
    ScopedEnv batch_on("DS4_CUDA_ENABLE_Q4_GROUPED_ATTN_A_BATCH"); batch_on.set("1");
    ScopedEnv dense_epilogue("DS4_CUDA_DISABLE_Q4_MMVQ_EPILOGUE"); dense_epilogue.set("1");
    ds4_mmq_set_gb10_optimizations(1);
    const int shapes[][4] = {{4,256,1,1}, {12,512,2,3}, {32,1024,3,8},
        {12,1792,8,16}, {36,2048,2,5}, {1024,4096,1,8}, {1024,4096,8,8},
        {64,8192,3,16}, {33,512,3,3}, {33,4096,1,8}};
    unsigned cases = 0;
    for (const auto &s : shapes) for (bool nonfinite : {false, true}) {
        gpu_grouped_case(s[0], s[1], s[2], s[3], nonfinite);
        ++cases;
    }
    ds4_mmq_set_gb10_optimizations(0);
    std::printf("PASS: %u CUDA grouped Q4 cases. No throughput claim.\n", cases);
}

static void gpu_tests() {
    cudaDeviceProp props{};
    check(cudaGetDeviceProperties(&props, 0));
    require(props.major > 7 || (props.major == 7 && props.minor >= 5), "oracle requires NVIDIA Turing or newer");
    require(ds4_mmq_init(0) == 0, "MMQ initialization failed");
    ScopedEnv persistent("DS4_CUDA_NO_Q4_K1024_PERSISTENT"); persistent.set("1");
    ScopedEnv require_persistent("DS4_CUDA_REQUIRE_Q4_K1024_PERSISTENT"); require_persistent.set(nullptr);
    ScopedEnv persistent_oracle("DS4_CUDA_Q4_K1024_PERSISTENT_ORACLE"); persistent_oracle.set(nullptr);
    const int shapes[][4] = {{4,8,1,256}, {32,16,1,1024}, {32,12,1,1792},
        {64,36,1,2048}, {1024,512,1,4096}, {512,1024,1,4096},
        {128,64,1,8192}, {32768,4,1,1024}, {33,12,1,512}};
    unsigned cases = 0;
    for (const auto &s : shapes) for (bool nonfinite : {false, true}) for (bool pair : {false, true}) {
        gpu_case(s[0], s[1], s[2], s[3], nonfinite, pair);
        ++cases;
    }
    const int batch_shapes[][3] = {{16,32,1024}, {1024,512,4096},
                                   {33,12,512}, {128,64,8192}};
    for (int n = 2; n <= 8; ++n) for (const auto &s : batch_shapes)
        for (bool nonfinite : {false, true}) for (bool pair : {false, true}) {
            gpu_case(s[0], s[1], n, s[2], nonfinite, pair);
            ++cases;
        }
    for (int n : {2,4,8}) for (bool nonfinite : {false, true}) for (bool pair : {false, true}) {
        gpu_case(32768,4,n,1024,nonfinite,pair);
        ++cases;
    }
    std::printf("PASS: %u CUDA production Q4 epilogue cases. No throughput claim.\n", cases);
    gpu_grouped_tests();
}
#endif

int main() {
    host_tests();
#ifdef __CUDACC__
    gpu_tests();
#else
    std::puts("Host-only check: CUDA kernel compilation and GPU parity still require nvcc/device.");
#endif
    return 0;
}
