// SPDX-License-Identifier: MIT
// Generated with the actual production functions by test_cuda_q8_hc_aligned.py.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static void require(bool ok, const char *message) {
    if (!ok) { std::fprintf(stderr, "Q8 aligned HC: %s\n", message); std::exit(1); }
}
static uint32_t random_word(uint32_t &s) { return s = 1664525u * s + 1013904223u; }

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "cuda/ds4_q8_quantize.cuh"
static __half half_from_float(float f) { return __float2half_rn(f); }
#else
#include <setjmp.h>
using __half = _Float16;
struct alignas(16) int4 { int32_t x, y, z, w; };
static __half half_from_float(float f) { return (__half)f; }
static float __half2float(__half f) { return (float)f; }
static float __fadd_rn(float a, float b) { volatile float v = a + b; return v; }
static int32_t __dp4a(int32_t a, int32_t b, int32_t acc) {
    for (unsigned i = 0; i < 4; ++i) {
        int av = int((uint32_t(a) >> (i * 8)) & 255u);
        int bv = int((uint32_t(b) >> (i * 8)) & 255u);
        if (av >= 128) av -= 256;
        if (bv >= 128) bv -= 256;
        acc += av * bv;
    }
    return acc;
}
static struct { unsigned x; } blockIdx, threadIdx;
static float leaves[32], total;
static bool collecting;
static jmp_buf collected;
static float host_warp_sum(float v) {
    if (collecting) { leaves[threadIdx.x & 31u] = v; longjmp(collected, 1); }
    return total;
}
#endif

/* DS4_PRODUCTION_FUNCTIONS */

struct Case {
    unsigned k, m, hc, mode, blocks;
    std::vector<unsigned char> raw;
    std::vector<int4> codes;
    std::vector<__half> scales;
    std::vector<float> input, xscale, add, add2, residual, split, home, peer;
    std::vector<int8_t> xq;
    std::vector<int32_t> selected;
    Case(unsigned kin, unsigned mout, unsigned hc_count, unsigned mode_in, unsigned seed)
        : k(kin), m(mout), hc(hc_count), mode(mode_in), blocks(k / 32),
          raw(size_t(m) * blocks * 34), codes(size_t(m) * blocks * 2),
          scales(size_t(m) * blocks), input(k), xscale(blocks), add(m), add2(m),
          residual(size_t(hc) * m), split(2 * hc + hc * hc), home(6 * m),
          peer(4 * m), xq(k), selected{0, 257, 258, -1, 1, 259} {
        uint32_t s = seed + 8317;
        for (size_t b = 0; b < scales.size(); ++b) {
            const float f = std::ldexp(float(int(random_word(s) % 255) - 127), -12);
            scales[b] = half_from_float(f);
            std::memcpy(raw.data() + b * 34, &scales[b], 2);
            for (unsigned j = 0; j < 32; ++j) {
                const int8_t q = int8_t(random_word(s) >> 24);
                raw[b * 34 + 2 + j] = (unsigned char)q;
            }
            std::memcpy(codes.data() + b * 2, raw.data() + b * 34 + 2, 32);
        }
        for (auto *v : {&add, &add2, &residual, &split, &home, &peer})
            for (float &f : *v) f = std::ldexp(float(int(random_word(s) % 1025) - 512), -7);
        refresh(seed);
    }
    void refresh(unsigned seed) {
        uint32_t s = seed + 47;
        for (unsigned b = 0; b < blocks; ++b) {
            float a = 0;
            for (unsigned j = 0; j < 32; ++j) {
                // Non-half-representable scales expose accidental Q8_1 use.
                float f = (int(random_word(s) % 32749) - 16374) / 997.f;
                if (seed % 5 == 0) f = 0.f;
                input[b * 32 + j] = f;
                a = std::fmax(a, std::fabs(f));
            }
            xscale[b] = a / 127.f;
            const float inv = a != 0 ? 1.f / xscale[b] : 0;
            for (unsigned j = 0; j < 32; ++j) {
                const long q = std::lrintf(input[b * 32 + j] * inv);
                xq[b * 32 + j] = int8_t(std::max(-128l, std::min(127l, q)));
            }
        }
    }
};
static constexpr unsigned guard = 16;
static constexpr float sentinel = 543210.f;
struct Result {
    std::vector<float> out, block;
    explicit Result(const Case &c) : out(size_t(c.m) * c.hc + 2 * guard, sentinel),
                                   block(c.m + 2 * guard, sentinel) {}
};
static void compare(const Result &a, const Result &b) {
    for (const auto *v : {&a.out, &a.block, &b.out, &b.block})
        for (unsigned i = 0; i < guard; ++i)
            require((*v)[i] == sentinel && (*v)[v->size() - 1 - i] == sentinel, "output canary");
    require(a.out.size() == b.out.size() && !std::memcmp(a.out.data(), b.out.data(), a.out.size() * 4), "HC output bits");
    require(a.block.size() == b.block.size() && !std::memcmp(a.block.data(), b.block.data(), a.block.size() * 4), "block output bits");
}

#ifndef __CUDACC__
static Result run_host(Case &c, bool aligned) {
    Result r(c);
    auto launch = [&] {
        const int owned = c.mode == 3;
        const int add = c.mode == 1 || c.mode == 2;
        if (aligned)
            matmul_q8_0_hc_expand_aligned_preq_warp8_kernel(
                r.out.data() + guard, r.block.data() + guard, c.add.data(), c.add2.data(),
                c.home.data(), c.peer.data(), c.selected.data(), c.residual.data(), c.split.data(),
                c.codes.data(), c.scales.data(), c.xq.data(), c.xscale.data(), c.m, c.m, c.hc,
                c.blocks, add, c.mode == 2, owned, 256);
        else
            matmul_q8_0_hc_expand_preq_warp8_kernel(
                r.out.data() + guard, r.block.data() + guard, c.add.data(), c.add2.data(),
                c.home.data(), c.peer.data(), c.selected.data(), c.residual.data(), c.split.data(),
                c.raw.data(), c.xq.data(), c.xscale.data(), c.k, c.m, c.m, c.hc, c.blocks,
                add, c.mode == 2, owned, 256, 1);
    };
    for (unsigned row = 0; row < ((c.m + 7) / 8) * 8; ++row) {
        blockIdx.x = row / 8;
        collecting = true;
        for (unsigned lane = 0; lane < 32; ++lane) {
            threadIdx.x = (row % 8) * 32 + lane;
            if (!setjmp(collected)) launch();
        }
        if (row >= c.m) continue;
        for (unsigned stride = 16; stride; stride >>= 1)
            for (unsigned lane = 0; lane < stride; ++lane) {
                volatile float v = leaves[lane] + leaves[lane + stride];
                leaves[lane] = v;
            }
        total = leaves[0]; collecting = false;
        threadIdx.x = (row % 8) * 32;
        launch();
    }
    return r;
}
static void dot_oracle() {
    uint32_t s = 781;
    alignas(16) int8_t a[32], b[32];
    for (unsigned pattern = 0; pattern < 8192; ++pattern) {
        int32_t scalar = 0;
        for (unsigned j = 0; j < 32; ++j) {
            a[j] = pattern == 0 ? -128 : int8_t(random_word(s) >> 24);
            b[j] = pattern == 0 ? -128 : int8_t(random_word(s) >> 24);
            scalar += int32_t(a[j]) * int32_t(b[j]);
        }
        int4 packed[2]; std::memcpy(packed, a, 32);
        require(dot_i8x32_aligned_int4(packed[0], packed[1], b) == scalar, "aligned DP4A scalar reference");
        require(dot_i8_block(a, b, 32, 1) == scalar, "GGUF DP4A scalar reference");
    }
}
#else
static void cuda_check(cudaError_t e) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s\n", cudaGetErrorString(e)); std::exit(1); }
}
struct Device {
    unsigned char *p = nullptr;
    size_t bytes;
    explicit Device(size_t n) : bytes(n) { cuda_check(cudaMalloc((void **)&p, n)); }
    ~Device() { cudaFree(p); }
    template<class T> T *ptr() { return reinterpret_cast<T *>(p); }
    template<class T> void upload(const std::vector<T> &v) {
        require(v.size() * sizeof(T) == bytes, "upload shape");
        cuda_check(cudaMemcpy(p, v.data(), bytes, cudaMemcpyHostToDevice));
    }
    template<class T> void download(std::vector<T> &v) {
        require(v.size() * sizeof(T) == bytes, "download shape");
        cuda_check(cudaMemcpy(v.data(), p, bytes, cudaMemcpyDeviceToHost));
    }
};
static Result run_cuda(Case &c, bool aligned, bool graph, bool warp_quant) {
    Result result(c);
    Device raw(c.raw.size()), codes(c.codes.size() * 16), scales(c.scales.size() * 2),
        x(c.input.size() * 4), q(c.xq.size()), xs(c.xscale.size() * 4),
        add(c.add.size() * 4), add2(c.add2.size() * 4), home(c.home.size() * 4),
        peer(c.peer.size() * 4), selected(c.selected.size() * 4),
        residual(c.residual.size() * 4), split(c.split.size() * 4),
        out(result.out.size() * 4), block(result.block.size() * 4);
    raw.upload(c.raw); codes.upload(c.codes); scales.upload(c.scales); x.upload(c.input);
    add.upload(c.add); add2.upload(c.add2); home.upload(c.home); peer.upload(c.peer);
    selected.upload(c.selected); residual.upload(c.residual); split.upload(c.split);
    out.upload(result.out); block.upload(result.block);
    cudaStream_t stream;
    cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    // Complete default-stream initialization before using the nonblocking stream.
    cuda_check(cudaDeviceSynchronize());
    auto launch = [&] {
        ds4_cuda_launch_q8_0_quantize(warp_quant, dim3(c.blocks), stream,
            q.ptr<int8_t>(), xs.ptr<float>(), x.ptr<float>(), c.k, c.blocks);
        const int has_add = c.mode == 1 || c.mode == 2;
        if (aligned)
            matmul_q8_0_hc_expand_aligned_preq_warp8_kernel<<<(c.m + 7) / 8, 256, 0, stream>>>(
                out.ptr<float>() + guard, block.ptr<float>() + guard, add.ptr<float>(), add2.ptr<float>(),
                home.ptr<float>(), peer.ptr<float>(), selected.ptr<int32_t>(), residual.ptr<float>(), split.ptr<float>(),
                codes.ptr<int4>(), scales.ptr<__half>(), q.ptr<int8_t>(), xs.ptr<float>(), c.m, c.m, c.hc,
                c.blocks, has_add, c.mode == 2, c.mode == 3, 256);
        else
            matmul_q8_0_hc_expand_preq_warp8_kernel<<<(c.m + 7) / 8, 256, 0, stream>>>(
                out.ptr<float>() + guard, block.ptr<float>() + guard, add.ptr<float>(), add2.ptr<float>(),
                home.ptr<float>(), peer.ptr<float>(), selected.ptr<int32_t>(), residual.ptr<float>(), split.ptr<float>(),
                raw.ptr<unsigned char>(), q.ptr<int8_t>(), xs.ptr<float>(), c.k, c.m, c.m, c.hc, c.blocks,
                has_add, c.mode == 2, c.mode == 3, 256, 1);
        cuda_check(cudaGetLastError());
    };
    launch(); cuda_check(cudaStreamSynchronize(stream));
    if (graph) {
        cudaGraph_t captured; cudaGraphExec_t executable;
        cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        launch();
        cuda_check(cudaStreamEndCapture(stream, &captured));
        cuda_check(cudaGraphInstantiate(&executable, captured, nullptr, nullptr, 0));
        for (unsigned replay = 1; replay <= 3; ++replay) {
            c.refresh(113 + replay);
            cuda_check(cudaMemcpyAsync(x.p, c.input.data(), x.bytes, cudaMemcpyHostToDevice, stream));
            cuda_check(cudaGraphLaunch(executable, stream));
            cuda_check(cudaStreamSynchronize(stream));
            // Compare each replay with a fresh eager run on the same new input.
            Result got(c); out.download(got.out); block.download(got.block);
            const Result reference = run_cuda(c, false, false, warp_quant);
            compare(reference, got);
        }
        cuda_check(cudaGraphExecDestroy(executable)); cuda_check(cudaGraphDestroy(captured));
    }
    std::vector<int8_t> qcopy(c.xq.size()); q.download(qcopy);
    std::vector<float> scale_copy(c.xscale.size()); xs.download(scale_copy);
    // Compare canonical GPU quantization to the separate production quantizer
    // using the same input; CPU division/FTZ is not a device bitwise oracle.
    Device qref(c.xq.size()), sref(c.xscale.size() * 4);
    ds4_cuda_launch_q8_0_quantize(false, dim3(c.blocks), stream,
        qref.ptr<int8_t>(), sref.ptr<float>(), x.ptr<float>(), c.k, c.blocks);
    cuda_check(cudaGetLastError()); cuda_check(cudaStreamSynchronize(stream));
    std::vector<int8_t> expected(c.xq.size()); qref.download(expected);
    std::vector<float> expected_scale(c.xscale.size()); sref.download(expected_scale);
    require(qcopy == expected, "canonical Q8_0 activation bytes");
    require(!std::memcmp(scale_copy.data(), expected_scale.data(), xs.bytes), "canonical F32 activation scales");
    out.download(result.out); block.download(result.block);
    std::vector<unsigned char> raw_after(c.raw.size()); raw.download(raw_after);
    std::vector<int4> codes_after(c.codes.size()); codes.download(codes_after);
    std::vector<__half> scales_after(c.scales.size()); scales.download(scales_after);
    require(raw_after == c.raw && !std::memcmp(codes_after.data(), c.codes.data(), codes.bytes) &&
            !std::memcmp(scales_after.data(), c.scales.data(), scales.bytes), "weight inputs modified");
    cuda_check(cudaStreamDestroy(stream));
    return result;
}
#endif

int main(int argc, char **argv) {
#ifdef __CUDACC__
    cuda_check(cudaSetDevice(argc > 1 ? std::atoi(argv[1]) : 0));
#else
    (void)argc; (void)argv;
    dot_oracle();
#endif
    unsigned count = 0;
    const unsigned shapes[][2] = {{32, 1}, {512, 7}, {512, 9}, {1024, 127},
        {1024, 128}, {1024, 129}, {2048, 4095}, {512, 4096}, {2048, 4096}, {4096, 4096}};
    for (const auto &shape : shapes) for (unsigned mode = 0; mode < 4; ++mode) {
        Case c(shape[0], shape[1], mode == 0 ? 1 : 4, mode, count);
#ifdef __CUDACC__
        for (bool warp_quant : {false, true}) {
            const Result a = run_cuda(c, false, false, warp_quant);
            const Result b = run_cuda(c, true, false, warp_quant);
            compare(a, b);
            if (shape[1] == 129 || shape[1] == 4096) {
                (void)run_cuda(c, true, true, warp_quant);
            }
        }
#else
        const auto q_before = c.xq;
        const auto scale_before = c.xscale;
        const Result a = run_host(c, false), b = run_host(c, true);
        compare(a, b);
        require(c.xq == q_before && !std::memcmp(c.xscale.data(), scale_before.data(), c.xscale.size() * 4), "activation inputs modified");
#endif
        ++count;
    }
    std::printf("PASS: %u production GGUF/aligned Q8 HC cases, row tails, add/add2/owned epilogues\n", count);
    return 0;
}
