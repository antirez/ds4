// Production source is inserted by test_rocm_f16_compressor.py.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
static void require(bool ok, const char *what) {
    if (!ok) { fprintf(stderr, "FAIL ROCm F16 compressor: %s\n", what); exit(1); }
}
#ifdef NATIVE_HIP
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#define CHECK(call) require((call) == hipSuccess, #call)
static hipStream_t stream;
#define LAUNCH(name, grid, block, shared, ...) do { \
    name<<<grid, block, shared, stream>>>(__VA_ARGS__); \
    CHECK(hipGetLastError()); \
} while (0)
#else
using __half = _Float16;
static float __half2float(__half v) { return float(v); }
static __half __float2half(float v) { return __half(v); }
static __half __ushort_as_half(unsigned short v) { __half h; memcpy(&h, &v, 2); return h; }
#define __device__
#define __global__
#define __shared__
#define __syncthreads() ((void)0)
static struct { unsigned x; } threadIdx, blockIdx, blockDim;
float shx[8192];
static float lane_partials[2][32];
static unsigned reduction_call;
static float warp_sum_f32(float v) {
    const unsigned lane = threadIdx.x & 31u;
    float *values = lane_partials[reduction_call++];
    values[lane] = v;
    if (lane != 0) return v;
    for (unsigned offset = 16; offset; offset >>= 1)
        for (unsigned i = 0; i + offset < 32; ++i) values[i] += values[i + offset];
    return values[0];
}
#define LAUNCH(name, grid, block, shared, ...) do { \
    blockDim.x = (block); \
    for (blockIdx.x = 0; blockIdx.x < (grid); ++blockIdx.x) \
        for (int lane = int(block)-1; lane >= 0; --lane) { \
            threadIdx.x = unsigned(lane); reduction_call = 0; name(__VA_ARGS__); \
        } \
} while (0)
#endif

// PRODUCTION_KERNELS

template<class T> struct buffer {
    std::vector<T> host;
    T *ptr;
    explicit buffer(size_t n) : host(n) {
#ifdef NATIVE_HIP
        CHECK(hipMalloc(&ptr, n*sizeof(T)));
#else
        ptr = host.data();
#endif
    }
    ~buffer() {
#ifdef NATIVE_HIP
        CHECK(hipFree(ptr));
#endif
    }
    void upload() {
#ifdef NATIVE_HIP
        CHECK(hipMemcpyAsync(ptr, host.data(), host.size()*sizeof(T), hipMemcpyHostToDevice, stream));
#endif
    }
    std::vector<T> read() {
#ifdef NATIVE_HIP
        CHECK(hipStreamSynchronize(stream));
        CHECK(hipMemcpy(host.data(), ptr, host.size()*sizeof(T), hipMemcpyDeviceToHost));
#endif
        return host;
    }
};
constexpr size_t guard = 16;
constexpr float sentinel = 123456.0f;
static unsigned cases;

static void run(unsigned k, unsigned width, unsigned ratio, unsigned ape_type, bool bench) {
    const size_t state_rows = ratio == 4 ? 8 : ratio;
    buffer<__half> w0(size_t(width)*k), w1(size_t(width)*k);
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
    w0.upload(); w1.upload(); ape32.upload(); ape16.upload();
    const void *ape = ape_type == 1 ? (const void *)ape16.ptr : (const void *)ape32.ptr;
    auto reset = [&]() {
        for (auto *b : {&kv,&score,&skv,&sscore}) {
            std::fill(b->host.begin(), b->host.end(), sentinel); b->upload();
        }
    };
    auto launch = [&](bool fused, unsigned pos, bool overlaunch=true) {
        const unsigned blocks = (width+31)/32 + (overlaunch ? 1 : 0);
        if (fused) {
            LAUNCH(matmul_f16_pair_compressor_store_sharedx_w32_kernel, blocks, 1024, size_t(k)*4,
                kv.ptr+guard, score.ptr+guard, w0.ptr, w1.ptr, x.ptr, k, width,
                skv.ptr+guard, sscore.ptr+guard, ape, ape_type, ratio, pos);
        } else {
            LAUNCH(matmul_f16_pair_f32_sharedx_warp_rows_w32_kernel, blocks, 1024, size_t(k)*4,
                kv.ptr+guard, score.ptr+guard, w0.ptr, w1.ptr, x.ptr, k, width);
            LAUNCH(compressor_store_kernel, (width+255)/256+(overlaunch ? 1 : 0), 256, 0,
                kv.ptr+guard, score.ptr+guard, skv.ptr+guard, sscore.ptr+guard,
                ape, 0, ape_type, width/(ratio==4 ? 2 : 1), ratio, pos, 1);
        }
    };
    auto read = [&]() { return std::vector<std::vector<float>>{kv.read(), score.read(), skv.read(), sscore.read()}; };
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
#ifdef NATIVE_HIP
        CHECK(hipStreamSynchronize(stream));
        hipGraph_t graph; hipGraphExec_t executable;
        CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
        launch(true, pos);
        CHECK(hipStreamEndCapture(stream, &graph));
        CHECK(hipGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
#endif
        for (unsigned replay=0; replay<3; ++replay) {
            for (size_t i=0; i<x.host.size(); ++i)
                x.host[i] = replay==0 ? 0.0f : (int(word()%1023)-511)*0.0127f;
            x.upload();
#ifndef NATIVE_HIP
            std::copy(x.host.begin(), x.host.end(), shx);
#endif
            reset(); launch(false, pos); const auto reference=read();
            reset(); launch(true, pos); compare(reference);
#ifdef NATIVE_HIP
            reset(); CHECK(hipGraphLaunch(executable, stream)); compare(reference);
#endif
            ++cases;
        }
#ifdef NATIVE_HIP
        CHECK(hipGraphExecDestroy(executable)); CHECK(hipGraphDestroy(graph));
#endif
    }
#ifdef NATIVE_HIP
    if (bench && k == 4096 && ape_type == 1) {
        constexpr unsigned repeats=100, samples=8;
        for (unsigned mode=0; mode<2; ++mode) {
            hipGraph_t graphs[2]; hipGraphExec_t executables[2];
            for (unsigned variant=0; variant<2; ++variant) {
                for (unsigned i=0;i<10;++i) launch(variant, 1, false);
                CHECK(hipStreamSynchronize(stream));
                CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
                launch(variant,1,false);
                CHECK(hipStreamEndCapture(stream,&graphs[variant]));
                CHECK(hipGraphInstantiate(&executables[variant],graphs[variant],nullptr,nullptr,0));
            }
            hipEvent_t begin,end; CHECK(hipEventCreate(&begin)); CHECK(hipEventCreate(&end));
            double totals[2]={};
            for(unsigned sample=0;sample<samples;++sample) for(unsigned order=0;order<2;++order) {
                const unsigned variant=order^(sample&1u);
                CHECK(hipEventRecord(begin,stream));
                for(unsigned i=0;i<repeats;++i) {
                    if(mode) CHECK(hipGraphLaunch(executables[variant],stream));
                    else launch(variant,1,false);
                }
                CHECK(hipEventRecord(end,stream)); CHECK(hipEventSynchronize(end));
                float ms; CHECK(hipEventElapsedTime(&ms,begin,end)); totals[variant]+=ms;
            }
            printf("ROCm F16 K=%u width=%u ratio=%u %s pair+store=%.3f us fused=%.3f us ratio=%.5f\n",
                k,width,ratio,mode?"graph":"eager",totals[0]*1000/(samples*repeats),
                totals[1]*1000/(samples*repeats),totals[1]/totals[0]);
            CHECK(hipEventDestroy(begin)); CHECK(hipEventDestroy(end));
            for(unsigned i=0;i<2;++i) { CHECK(hipGraphExecDestroy(executables[i])); CHECK(hipGraphDestroy(graphs[i])); }
        }
    }
#else
    (void)bench;
#endif
}
int main(int argc, char **argv) {
    bool bench=argc==2 && !strcmp(argv[1],"--bench");
#ifdef NATIVE_HIP
    CHECK(hipSetDevice(0)); CHECK(hipStreamCreateWithFlags(&stream,hipStreamNonBlocking));
    hipDeviceProp_t prop; CHECK(hipGetDeviceProperties(&prop,0));
    printf("ROCm native device: %s (%s)\n",prop.name,prop.gcnArchName);
#endif
    for(unsigned ape_type:{0u,1u}) {
        run(4096,256,4,ape_type,bench); run(4096,1024,4,ape_type,bench);
        run(4096,512,128,ape_type,bench);
        // Generic kernel tails, including incomplete waves and unroll tails.
        for(unsigned k:{1u,31u,32u,33u,255u,257u}) run(k,33,128,ape_type,false);
    }
#ifdef NATIVE_HIP
    CHECK(hipStreamDestroy(stream));
#endif
    printf("PASS ROCm F16 compressor: %u arithmetic/state cases\n",cases);
}
