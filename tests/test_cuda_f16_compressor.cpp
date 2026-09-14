// Used by test_cuda_f16_compressor.py; production bodies are inserted below.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static void require(bool ok, const char *what) {
    if (!ok) { fprintf(stderr, "FAIL F16 compressor: %s\n", what); exit(1); }
}
#ifdef NATIVE_CUDA
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#define CHECK(call) require((call) == cudaSuccess, #call)
static cudaStream_t stream;
#define LAUNCH(name, grid, block, ...) do { \
    name<<<grid, block, 0, stream>>>(__VA_ARGS__); \
    CHECK(cudaGetLastError()); \
} while (0)
#else
using __half = _Float16;
struct __half2 { __half lo, hi; };
static float __half2float(__half v) { return float(v); }
static __half __float2half(float v) { return __half(v); }
static __half2 __halves2half2(__half a, __half b) { return {a,b}; }
static __half __low2half(__half2 v) { return v.lo; }
static __half __high2half(__half2 v) { return v.hi; }
#define __device__
#define __global__
#define __shared__ static
#define __syncthreads() ((void)0)
static struct { unsigned x; } threadIdx, blockIdx, blockDim;
#define LAUNCH(name, grid, block, ...) do { \
    blockDim.x = (block); \
    for (blockIdx.x = 0; blockIdx.x < (grid); ++blockIdx.x) \
        for (int lane = int(block)-1; lane >= 0; --lane) { \
            threadIdx.x = unsigned(lane); name(__VA_ARGS__); \
        } \
} while (0)
#endif

// PRODUCTION_KERNELS

template<class T> struct buffer {
    std::vector<T> host;
    T *ptr;
    explicit buffer(size_t n) : host(n) {
#ifdef NATIVE_CUDA
        CHECK(cudaMalloc(&ptr, n*sizeof(T)));
#else
        ptr = host.data();
#endif
    }
    ~buffer() {
#ifdef NATIVE_CUDA
        CHECK(cudaFree(ptr));
#endif
    }
    void upload() {
#ifdef NATIVE_CUDA
        CHECK(cudaMemcpyAsync(ptr, host.data(), host.size()*sizeof(T),
                              cudaMemcpyHostToDevice, stream));
#endif
    }
    std::vector<T> read() {
#ifdef NATIVE_CUDA
        CHECK(cudaStreamSynchronize(stream));
        CHECK(cudaMemcpy(host.data(), ptr, host.size()*sizeof(T), cudaMemcpyDeviceToHost));
#endif
        return host;
    }
};
constexpr size_t guard = 16;
constexpr float sentinel = 123456.0f;

static void run(unsigned width, unsigned ratio, unsigned ape_type, bool bench) {
    constexpr unsigned k = 4096;
    const size_t state_rows = ratio == 4 ? 8 : ratio;
    buffer<__half> w0(size_t(width)*k), w1(size_t(width)*k);
    buffer<__half2> packed(size_t(width)*k + 2*guard);
    buffer<float> x(k), ape32(size_t(width)*ratio);
    buffer<__half> ape16(size_t(width)*ratio);
    buffer<float> kv(width+2*guard), score(width+2*guard);
    buffer<float> skv(state_rows*width+2*guard), sscore(state_rows*width+2*guard);
    uint32_t rng = 719;
    auto word = [&]() { rng ^= rng<<13; rng ^= rng>>17; rng ^= rng<<5; return rng; };
    for (size_t i=0; i<w0.host.size(); ++i) {
        w0.host[i] = __float2half((int(word()%509)-254)*0.00071f);
        w1.host[i] = __float2half((int(word()%509)-254)*0.00053f);
    }
    for (size_t i=0; i<ape32.host.size(); ++i) {
        ape32.host[i] = (int(word()%257)-128)*0.013f;
        ape16.host[i] = __float2half(ape32.host[i]);
    }
    const __half2 half_guard = __halves2half2(__float2half(7), __float2half(-9));
    std::fill(packed.host.begin(), packed.host.end(), half_guard);
    w0.upload(); w1.upload(); ape32.upload(); ape16.upload(); packed.upload();
    LAUNCH(f16_pair_chunk32_repack_kernel, (width*k+255)/256, 256,
           packed.ptr+guard, w0.ptr, w1.ptr, k, width);
    auto packed_host = packed.read();
    for (size_t i=0; i<guard; ++i) {
        require(!memcmp(&packed_host[i], &half_guard, sizeof(half_guard)), "repack leading guard");
        require(!memcmp(&packed_host[packed_host.size()-1-i], &half_guard, sizeof(half_guard)), "repack trailing guard");
    }
    for (unsigned row=0; row<width; ++row) for (unsigned col=0; col<k; ++col) {
        const unsigned lane=col/(k/32), iteration=col%(k/32);
        const __half2 want = __halves2half2(w0.host[size_t(row)*k+col], w1.host[size_t(row)*k+col]);
        require(!memcmp(&packed_host[guard+(size_t(row)*(k/32)+iteration)*32+lane],
                        &want, sizeof(want)), "repacked weight bits");
    }
    const void *ape = ape_type == 1 ? (const void *)ape16.ptr : (const void *)ape32.ptr;
    auto reset = [&]() {
        for (auto *b : {&kv,&score,&skv,&sscore}) {
            std::fill(b->host.begin(), b->host.end(), sentinel); b->upload();
        }
    };
    auto launch = [&](int variant, unsigned pos) {
        if (variant == 0) {
            LAUNCH(matmul_f16_pair_ordered_chunks_kernel, width+1, 32,
                   kv.ptr+guard, score.ptr+guard, w0.ptr, w1.ptr, x.ptr, k, width, width);
            LAUNCH(compressor_store_kernel, (width+255)/256+1, 256,
                   kv.ptr+guard, score.ptr+guard, skv.ptr+guard, sscore.ptr+guard,
                   ape, 0, ape_type, width/(ratio==4 ? 2 : 1), ratio, pos, 1);
        } else if (variant == 1) {
            LAUNCH(matmul_f16_pair_compressor_store_ordered_chunks_kernel, width+1, 32,
                   kv.ptr+guard, score.ptr+guard, skv.ptr+guard, sscore.ptr+guard,
                   w0.ptr, w1.ptr, x.ptr, ape, ape_type, k, width, ratio, pos);
        } else {
            LAUNCH(matmul_f16_pair_compressor_store_chunk32_prefetch8_kernel, width+1, 32,
                   kv.ptr+guard, score.ptr+guard, skv.ptr+guard, sscore.ptr+guard,
                   packed.ptr+guard, x.ptr, ape, ape_type, k, width, ratio, pos);
        }
    };
    auto read = [&]() { return std::vector<std::vector<float>>{
        kv.read(), score.read(), skv.read(), sscore.read()}; };
    auto compare = [&](const std::vector<std::vector<float>> &ref) {
        const auto got=read();
        for (unsigned j=0; j<got.size(); ++j) {
            require(!memcmp(ref[j].data(), got[j].data(), got[j].size()*sizeof(float)),
                    "output/state bits and untouched rows");
            for (size_t i=0; i<guard; ++i)
                require(got[j][i]==sentinel && got[j][got[j].size()-1-i]==sentinel, "output/state guard");
        }
    };
    for (unsigned pos : {0u, ratio-1, ratio, UINT32_MAX}) {
#ifdef NATIVE_CUDA
        cudaGraph_t graph;
        cudaGraphExec_t executable;
        CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        launch(2, pos);
        CHECK(cudaStreamEndCapture(stream, &graph));
        CHECK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
#endif
        for (unsigned replay=0; replay<3; ++replay) {
            for (size_t i=0; i<x.host.size(); ++i)
                x.host[i] = replay==0 ? 0.0f : (int(word()%1023)-511)*0.0127f;
            x.upload(); reset(); launch(0, pos); const auto reference=read();
            for (int variant : {1,2}) { reset(); launch(variant, pos); compare(reference); }
#ifdef NATIVE_CUDA
            reset(); CHECK(cudaGraphLaunch(executable, stream)); compare(reference);
#endif
        }
#ifdef NATIVE_CUDA
        CHECK(cudaGraphExecDestroy(executable)); CHECK(cudaGraphDestroy(graph));
#endif
    }
#ifdef NATIVE_CUDA
    if (bench) {
        std::vector<float> samples[3];
        cudaEvent_t start, end; CHECK(cudaEventCreate(&start)); CHECK(cudaEventCreate(&end));
        for (int round=0; round<8; ++round) for (int arm=0; arm<3; ++arm) {
            const int variant = round%2 ? 2-arm : arm;
            for (int i=0; i<10; ++i) launch(variant, 0);
            CHECK(cudaEventRecord(start, stream));
            for (int i=0; i<100; ++i) launch(variant, 0);
            CHECK(cudaEventRecord(end, stream)); CHECK(cudaEventSynchronize(end));
            float ms; CHECK(cudaEventElapsedTime(&ms, start, end));
            samples[variant].push_back(ms*10.0f);
        }
        float medians[3];
        for (int i=0; i<3; ++i) { std::sort(samples[i].begin(),samples[i].end()); medians[i]=(samples[i][3]+samples[i][4])*0.5f; }
        printf("BENCH K=%u width=%u ratio=%u ape=%u separate_us=%.3f fused_us=%.3f packed_us=%.3f\n",
               k,width,ratio,ape_type,medians[0],medians[1],medians[2]);
        CHECK(cudaEventDestroy(start)); CHECK(cudaEventDestroy(end));
    }
#else
    (void)bench;
#endif
    printf("PASS F16 compressor K=%u width=%u ratio=%u ape=%u (12 input/position cases)\n",k,width,ratio,ape_type);
}

int main(int argc, char **argv) {
    const bool bench = argc==2 && !strcmp(argv[1], "--bench");
#ifdef NATIVE_CUDA
    CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
#endif
    for (unsigned type : {0u,1u}) {
        run(256,4,type,bench); run(1024,4,type,bench); run(512,128,type,bench);
    }
#ifdef NATIVE_CUDA
    CHECK(cudaStreamDestroy(stream));
#endif
    puts("F16 compressor production oracle: PASS");
}
