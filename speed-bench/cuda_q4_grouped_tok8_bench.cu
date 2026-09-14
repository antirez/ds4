// SPDX-License-Identifier: MIT
// Included by tests/test_cuda_q4_grouped_tok8.py after actual production helpers.
// Candidate benchmark only: no model/default dispatch is changed.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "cuda/ds4_q4_grouped_tok8_candidate.cuh"
using ds4_q4_tok8_candidate::Plan;
static void require(bool ok, const char *why) {
    if (!ok) { std::fprintf(stderr, "Q4 tok8: %s\n", why); std::exit(1); }
}
static uint32_t random_u32(uint32_t &state) { return state = state*1664525u + 1013904223u; }
[[maybe_unused]] static uint32_t float_bits(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static void fill_weights(std::vector<cuda_block_q4_K> &w, uint32_t &seed) {
    for (auto &b : w) {
        b.d = uint16_t(0x1800u + (random_u32(seed) & 0x3ffu));
        b.dmin = uint16_t(0x1400u + (random_u32(seed) & 0x3ffu));
        for (auto &v : b.scales) v = uint8_t(random_u32(seed) >> 24);
        for (auto &v : b.qs) v = uint8_t(random_u32(seed) >> 24);
    }
}
#ifndef __CUDACC__
static void host_tests() {
    size_t covered = 0;
    for (uint32_t m : {1u, 7u, 31u, 32u, 33u})
    for (uint32_t g : {1u, 3u, 8u}) for (uint32_t n = 1; n <= 35; ++n) {
        Plan p{}; require(ds4_q4_tok8_candidate::plan(m, g, n, 512, &p), "small admission");
        std::vector<unsigned> writes(size_t(n)*m*g);
        for (uint32_t by = 0; by < p.grid_y; ++by)
        for (uint32_t bx = 0; bx < p.grid_x; ++bx)
        for (uint32_t row_lane = 0; row_lane < 32; ++row_lane) {
            const uint32_t row = bx*32u + row_lane;
            if (row >= p.low_dim) continue;
            const uint32_t group = row / p.rank;
            for (uint32_t t = by*8u; t < std::min(n, by*8u + 8u); ++t) {
                const uint64_t qrow = uint64_t(t)*p.groups + group;
                require(qrow < p.rows && qrow*p.blocks + p.blocks <= p.q8_bytes/292u,
                        "grouped input bounds");
                ++writes[size_t(t)*p.low_dim + row];
            }
        }
        for (unsigned v : writes) require(v == 1, "output ownership/tail coverage");
        covered += writes.size();
    }
    for (uint32_t n : {8191u, 8192u, 8193u, 65535u, 65536u, 524280u}) {
        Plan p{}; require(ds4_q4_tok8_candidate::plan(33, 8, n, 256, &p), "large admission");
        uint64_t rows = 0; unsigned launches = 0;
        while (rows < p.rows) {
            uint32_t batch = ds4_q4_tok8_candidate::quant_batch_rows(p, rows);
            require(batch && batch <= 65535u, "CUDA quantizer grid.y bound");
            rows += batch; ++launches;
        }
        require(rows == p.rows && launches == p.quant_launches && p.grid_y <= 65535u,
                "large row batching/coverage");
        require(ds4_q4_tok8_candidate::quant_batch_rows(p, rows) == 0, "batch end");
    }
    Plan p{};
    require(!ds4_q4_tok8_candidate::plan(1, 8, 524281, 256, &p), "tok8 grid overflow");
    require(!ds4_q4_tok8_candidate::plan(UINT32_MAX, 2, 9, 256, &p), "low_dim overflow");
    require(!ds4_q4_tok8_candidate::plan(1, UINT32_MAX, 9, 256, &p), "row overflow");
    require(!ds4_q4_tok8_candidate::plan(0, 8, 9, 256, &p), "zero rank");
    require(!ds4_q4_tok8_candidate::plan(1, 8, 9, 257, &p), "partial K block");
    require(!ds4_q4_tok8_candidate::plan(1, 8, 9, 256, nullptr), "null plan");
    require(!ds4_q4_tok8_candidate::plan(1, 8192, 524280, UINT32_MAX-255u, &p),
            "allocation byte overflow");

    // Actual production dot8 versus independent scalar nibble/scale unpack.
    // Integer sums are exact; half-to-float has a host implementation here.
    uint32_t seed = 47; size_t dots = 0;
    for (unsigned repeat = 0; repeat < 1024; ++repeat) {
        std::vector<cuda_block_q4_K> weights(1); fill_weights(weights, seed);
        const auto &w = weights[0];
        std::array<cuda_block_q8_K, 8> q{};
        for (auto &b : q) {
            b.d = (int(random_u32(seed) % 33u) - 16) / 1024.f;
            for (auto &v : b.qs) v = int8_t(random_u32(seed) >> 24);
            for (unsigned j = 0; j < 16; ++j)
                for (unsigned i = 0; i < 16; ++i) b.bsums[j] += b.qs[j*16+i];
        }
        for (unsigned n = 1; n <= 8; ++n) {
            float acc[8]{};
            dev_dot_q4_K_q8_K_block8(&w, &q[0], n>1?&q[1]:nullptr,
                n>2?&q[2]:nullptr, n>3?&q[3]:nullptr, n>4?&q[4]:nullptr,
                n>5?&q[5]:nullptr, n>6?&q[6]:nullptr, n>7?&q[7]:nullptr, n, acc);
            for (unsigned t = 0; t < n; ++t) {
                int isum = 0, msum = 0;
                for (unsigned j = 0; j < 8; ++j) {
                    const unsigned scale = j<4 ? w.scales[j]&63u :
                        (w.scales[j+4]&15u) | ((w.scales[j-4]>>6u)<<4u);
                    const unsigned minimum = j<4 ? w.scales[j+4]&63u :
                        (w.scales[j+4]>>4u) | ((w.scales[j]>>6u)<<4u);
                    for (unsigned i = 0; i < 32; ++i) {
                        unsigned packed = w.qs[(j/2)*32+i];
                        unsigned nibble = (j&1) ? packed>>4 : packed&15u;
                        isum += int(scale*nibble)*q[t].qs[j*32+i];
                        msum += int(minimum)*q[t].qs[j*32+i];
                    }
                }
                const float a = q[t].d*dev_f16_to_f32(w.d)*float(isum);
                const float b = q[t].d*dev_f16_to_f32(w.dmin)*float(msum);
                const float expected = a-b;
                require(std::fabs(acc[t]-expected) <= 8e-7f*(std::fabs(a)+std::fabs(b)+1.f),
                        "dot8 scalar reference (host rounding tolerance)");
                ++dots;
            }
            for (unsigned t = n; t < 8; ++t) require(acc[t] == 0.f, "inactive token touched");
        }
    }
    std::printf("PASS host: %zu output ownerships; grid/size boundaries; %zu production dot8 comparisons.\n",
                covered, dots);
}
int main() { host_tests(); }
#else
#include <cuda_runtime.h>
#include "cuda/mmq/ds4_mmq.h"
// Standalone MMQ link: no backend producer-fold state exists in this fixture.
extern "C" int ds4_cuda_q8_fold_take_q81(
        const void *, uint64_t, const void **q81) {
    if (q81) *q81 = nullptr;
    return 0;
}
static void check(cudaError_t e) {
    if (e != cudaSuccess) { std::fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(e)); std::exit(1); }
}
struct Buffer {
    static constexpr size_t guard = 256;
    unsigned char *base = nullptr; size_t size;
    explicit Buffer(size_t n) : size(n) {
        require(n <= SIZE_MAX - 2*guard, "buffer overflow");
        check(cudaMalloc(reinterpret_cast<void **>(&base), n+2*guard));
        check(cudaMemset(base, 0xa5, n+2*guard));
    }
    ~Buffer() { cudaFree(base); }
    Buffer(const Buffer &) = delete;
    Buffer &operator=(const Buffer &) = delete;
    template<typename T> T *ptr() { return reinterpret_cast<T *>(base+guard); }
    void verify() {
        std::array<unsigned char, guard> before{}, after{};
        check(cudaMemcpy(before.data(), base, guard, cudaMemcpyDeviceToHost));
        check(cudaMemcpy(after.data(), base+guard+size, guard, cudaMemcpyDeviceToHost));
        for (size_t i=0; i<guard; ++i) require(before[i]==0xa5 && after[i]==0xa5, "buffer guard");
    }
    std::vector<float> read() {
        verify(); std::vector<float> out(size/4);
        check(cudaMemcpy(out.data(), ptr<float>(), size, cudaMemcpyDeviceToHost)); return out;
    }
};
__global__ static void q4_grouped_scalar_reference(
        float *out, const cuda_block_q4_K *w, const cuda_block_q8_K *q,
        uint32_t blocks, uint32_t rank, uint32_t groups, uint32_t tokens,
        uint32_t token0) {
    const uint32_t lane = threadIdx.x&7u;
    const uint32_t row = blockIdx.x*32u+(threadIdx.x>>3u);
    const uint32_t t = token0+blockIdx.y;
    const uint32_t low_dim = groups*rank;
    if (row>=low_dim || t>=tokens) return;
    const auto *wr = w+uint64_t(row)*blocks;
    const auto *qr = q+(uint64_t(t)*groups+row/rank)*blocks;
    float acc=0.f;
    for (uint32_t b=lane; b<blocks; b+=8u) acc += dev_dot_q4_K_q8_K_block(wr+b, qr+b);
    acc=quarter_warp_sum_f32(acc,lane);
    if (lane==0) out[uint64_t(t)*low_dim+row]=acc;
}
static void compare(const char *name, const std::vector<float> &got,
                    const std::vector<float> &ref, bool same_quant) {
    require(got.size()==ref.size(), "comparison size");
    size_t different=0; double maximum=0, squared=0, ref_squared=0;
    for (size_t i=0;i<got.size();++i) {
        require(std::isfinite(got[i]) && std::isfinite(ref[i]), "nonfinite output");
        different += float_bits(got[i]) != float_bits(ref[i]);
        const double error=std::fabs(double(got[i])-ref[i]);
        maximum=std::max(maximum,error); squared+=error*error; ref_squared+=double(ref[i])*ref[i];
        if (same_quant) require(error<=1e-4*(1.+std::fabs(double(ref[i]))),
                                "Q8_K scalar error exceeded 1e-4*(1+abs(reference))");
    }
    std::printf("  %s: differing=%zu/%zu max_abs=%.8g normalized_RMSE=%.8g%s\n", name,
        different, got.size(), maximum, std::sqrt(squared/std::max(ref_squared,1e-30)),
        same_quant ? " (bounded Q8_K check)" : " (Q8_K/Q8_1 differences, no quality claim)");
}
static void gpu_case(uint32_t m, uint32_t g, uint32_t n, uint32_t k,
                     cudaStream_t stream, bool gb10, bool timing) {
    Plan p{}; require(ds4_q4_tok8_candidate::plan(m,g,n,k,&p), "fixture shape admission");
    require(m<=INT_MAX && g<=INT_MAX && n<=INT_MAX && k<=INT_MAX &&
            uint64_t(p.rows)*k<=INT_MAX, "native MMQ input indexing bound");
    uint32_t seed=19u+n+k+m;
    std::vector<cuda_block_q4_K> weights(p.weight_bytes/144u); fill_weights(weights,seed);
    std::vector<float> input(p.input_bytes/4u);
    for (size_t i=0;i<input.size();++i)
        input[i]=(int(random_u32(seed)%8193u)-4096)/2048.f;
    // Preserve an exact zero block and opposite-sign maximum ties.
    std::fill(input.begin(), input.begin()+256, 0.f);
    if (input.size()>512) { input[256]=8.f; input[257]=-8.f; }
    Buffer w(p.weight_bytes), x(p.input_bytes), q(p.q8_bytes), candidate(p.output_bytes),
           oracle(p.output_bytes), canonical(p.output_bytes), packed_out(p.output_bytes),
           pack_x(size_t(n)*k*4u), pack_low(size_t(n)*m*4u),
           quant_probe(size_t(4u)*p.blocks*sizeof(cuda_block_q8_K));
    // Buffer poison uses the legacy stream; join it before nonblocking work.
    check(cudaDeviceSynchronize());
    check(cudaMemcpyAsync(w.ptr<void>(),weights.data(),p.weight_bytes,cudaMemcpyHostToDevice,stream));
    check(cudaMemcpyAsync(x.ptr<void>(),input.data(),p.input_bytes,cudaMemcpyHostToDevice,stream));
    auto tok8 = [&] {
        check(ds4_q4_tok8_quantize(p,q.ptr<cuda_block_q8_K>(),x.ptr<float>(),stream));
        check(ds4_q4_tok8_matmul(p,candidate.ptr<float>(),w.ptr<cuda_block_q4_K>(),q.ptr<cuda_block_q8_K>(),stream));
    };
    auto mmq = [&] {
        require(ds4_mmq_q4_K_grouped_dense(w.ptr<void>(),x.ptr<float>(),canonical.ptr<float>(),
            int(m),int(n),int(k),int(g),stream)==0, "canonical grouped MMQ failed");
    };
    auto packed = [&] {
        for (uint32_t group=0;group<g;++group) {
            check(cudaMemcpy2DAsync(pack_x.ptr<void>(),size_t(k)*4,
                x.ptr<float>()+uint64_t(group)*k,uint64_t(g)*k*4,
                size_t(k)*4,n,cudaMemcpyDeviceToDevice,stream));
            require(ds4_mmq_q4_K_dense(w.ptr<cuda_block_q4_K>()+uint64_t(group)*m*p.blocks,
                pack_x.ptr<float>(),pack_low.ptr<float>(),int(m),int(n),int(k),stream)==0,
                "packed dense MMQ failed");
            check(cudaMemcpy2DAsync(packed_out.ptr<float>()+uint64_t(group)*m,uint64_t(g)*m*4,
                pack_low.ptr<void>(),size_t(m)*4,size_t(m)*4,n,cudaMemcpyDeviceToDevice,stream));
        }
    };
    tok8(); mmq(); if (!gb10) packed();
    for (uint32_t t0=0;t0<n;) {
        const uint32_t count=std::min(n-t0,65535u);
        q4_grouped_scalar_reference<<<dim3(p.grid_x,count),256,0,stream>>>(
            oracle.ptr<float>(),w.ptr<cuda_block_q4_K>(),q.ptr<cuda_block_q8_K>(),
            p.blocks,m,g,n,t0);
        check(cudaGetLastError()); t0+=count;
    }
    // Check quantized rows independently around the split, including a split
    // inside a token's group range (N8193/G8: first launch ends at group6).
    std::vector<uint32_t> probe_rows={0u,p.rows-1u};
    if(p.rows>65535u) { probe_rows.push_back(65534u); probe_rows.push_back(65535u); }
    for(size_t i=0;i<probe_rows.size();++i) {
        q8_K_quantize_kernel<<<p.blocks,256,0,stream>>>(
            quant_probe.ptr<cuda_block_q8_K>()+i*p.blocks,
            x.ptr<float>()+uint64_t(probe_rows[i])*k,k,1u);
        check(cudaGetLastError());
    }
    check(cudaStreamSynchronize(stream));
    const size_t quant_row_bytes=size_t(p.blocks)*sizeof(cuda_block_q8_K);
    std::vector<unsigned char> quant_actual(quant_row_bytes),quant_expected(quant_row_bytes);
    for(size_t i=0;i<probe_rows.size();++i) {
        check(cudaMemcpy(quant_actual.data(),q.ptr<cuda_block_q8_K>()+uint64_t(probe_rows[i])*p.blocks,
                         quant_row_bytes,cudaMemcpyDeviceToHost));
        check(cudaMemcpy(quant_expected.data(),quant_probe.ptr<cuda_block_q8_K>()+i*p.blocks,
                         quant_row_bytes,cudaMemcpyDeviceToHost));
        require(quant_actual==quant_expected,"batched Q8_K row differs from original one-row quantizer");
    }
    const auto actual=candidate.read();
    std::printf("M=%u G=%u N=%u K=%u quant_launches=%u standalone_default_family=%s\n",m,g,n,k,
                p.quant_launches,gb10?"grouped-MMQ":"pack-MMQ-scatter");
    compare("tok8 vs scalar Q8_K",actual,oracle.read(),true);
    compare("tok8 vs grouped MMQ",actual,canonical.read(),false);
    if (!gb10) compare("tok8 vs current packed MMQ",actual,packed_out.read(),false);

    // Same stream, preallocated scratch, warmed kernels: verify graph replay.
    check(cudaMemsetAsync(candidate.ptr<void>(),0xa5,p.output_bytes,stream));
    check(cudaMemsetAsync(q.ptr<void>(),0xa5,p.q8_bytes,stream));
    check(cudaStreamSynchronize(stream));
    cudaGraph_t graph=nullptr; cudaGraphExec_t executable=nullptr;
    check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal)); tok8();
    check(cudaStreamEndCapture(stream,&graph));
    check(cudaGraphInstantiate(&executable,graph,nullptr,nullptr,0));
    for (int replay=0;replay<3;++replay) check(cudaGraphLaunch(executable,stream));
    check(cudaStreamSynchronize(stream));
    const auto replay=candidate.read();
    require(std::memcmp(actual.data(),replay.data(),p.output_bytes)==0,"graph replay bytes");
    check(cudaGraphExecDestroy(executable)); check(cudaGraphDestroy(graph));

    if (timing) {
        std::puts("  eager CUDA-event timing includes submit gaps; synthetic resident inputs, warm caches. "
                  "tok8 scratch preallocated; MMQ scratch from pool; no aligned model arena installed.");
        cudaEvent_t begin,end; check(cudaEventCreate(&begin)); check(cudaEventCreate(&end));
        std::array<std::vector<float>,3> samples;
        const unsigned arms=gb10?2:3;
        auto launch=[&](unsigned a) { if(a==0) tok8(); else if(a==1) mmq(); else packed(); };
        for (unsigned warm=0;warm<2;++warm) for(unsigned a=0;a<arms;++a) launch(a);
        check(cudaStreamSynchronize(stream));
        // Rotate/reverse arm order so every arm is measured at each position.
        for(unsigned repeat=0;repeat<12;++repeat) for(unsigned j=0;j<arms;++j) {
            unsigned a=(repeat/2+(repeat%2?arms-1-j:j))%arms;
            check(cudaEventRecord(begin,stream)); launch(a); check(cudaEventRecord(end,stream));
            check(cudaEventSynchronize(end)); float ms=0;
            check(cudaEventElapsedTime(&ms,begin,end)); samples[a].push_back(ms);
        }
        const char *names[]={"tok8 quant+matmul","grouped MMQ quant+matmul","packed MMQ full pipeline"};
        for(unsigned a=0;a<arms;++a) {
            auto &s=samples[a]; std::sort(s.begin(),s.end());
            std::printf("  time %-26s median=%.6f ms min=%.6f ms samples=%zu\n",names[a],
                (s[s.size()/2-1]+s[s.size()/2])*0.5f,s.front(),s.size());
        }
        check(cudaEventDestroy(begin)); check(cudaEventDestroy(end));
    }
    for (Buffer *b : {&w,&x,&q,&candidate,&oracle,&canonical,&packed_out,&pack_x,&pack_low,&quant_probe}) b->verify();
}
static unsigned parse_uint(const char *s) {
    char *end=nullptr; unsigned long n=std::strtoul(s,&end,10);
    require(*s && end && !*end && n<=UINT32_MAX,"invalid unsigned argument"); return unsigned(n);
}
int main(int argc,char **argv) {
    int device=0; bool bench=false; uint32_t tokens=512;
    for(int i=1;i<argc;++i) {
        if(std::strcmp(argv[i],"--bench")==0) bench=true;
        else if(std::strcmp(argv[i],"--device")==0 && i+1<argc) {
            const unsigned selected=parse_uint(argv[++i]);
            require(selected<=INT_MAX,"device overflow"); device=int(selected);
        }
        else if(std::strcmp(argv[i],"--tokens")==0 && i+1<argc) tokens=parse_uint(argv[++i]);
        else require(false,"usage: [--device N] [--bench] [--tokens N]");
    }
    require(device>=0,"device overflow"); check(cudaSetDevice(device));
    cudaDeviceProp prop{}; check(cudaGetDeviceProperties(&prop,device));
    const bool gb10=prop.major==12 && prop.minor==1 && prop.integrated;
    require(ds4_mmq_init(device)==0,"MMQ initialization failed");
    ds4_mmq_set_gb10_optimizations(gb10);
    std::printf("CUDA device=%d %s sm_%d%d integrated=%d; isolated canonical APIs; production dispatch has additional runtime gates\n",
                device,prop.name,prop.major,prop.minor,prop.integrated);
    cudaStream_t stream=nullptr; check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    for(uint32_t n=9;n<=17;++n) gpu_case(33,3,n,256,stream,gb10,false);
    gpu_case(32,8,8193,256,stream,gb10,false);
    gpu_case(1024,8,33,4096,stream,gb10,false);
    if(bench) {
        require(tokens>8,"benchmark must exercise prefill N>8");
        gpu_case(1024,8,tokens,4096,stream,gb10,true);
    }
    check(cudaStreamDestroy(stream));
    std::puts("PASS CUDA candidate guards/tails/Q8_K bounded oracle/graph. MMQ deltas above are diagnostic, not a quality gate.");
}
#endif
