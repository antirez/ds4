// Extracted production kernels are inserted by test_rocm_moe_prefill.py.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <type_traits>
#include <vector>

static void check(bool ok, const char *what) {
    if (!ok) { std::fprintf(stderr, "ROCm MoE prefill FAIL: %s\n", what); std::exit(1); }
}
#ifdef TEST_NATIVE
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <rocwmma/rocwmma.hpp>
#define FULL_WARP_MASK 0xffffffffffffffffull
#else
using half = _Float16;
using __half = half;
struct half2 { half x, y; };
using __half2 = half2;
static float __half2float(half x) { return float(x); }
static half __float2half(float x) { return half(x); }
static half2 __floats2half2_rn(float x, float y) { return {half(x), half(y)}; }
static uint32_t __popcll(uint64_t x) { return uint32_t(__builtin_popcountll(x)); }
struct HostDim { uint32_t x=0, y=0, z=0; };
static HostDim blockIdx, threadIdx, blockDim;
static uint32_t warpSize;
#define FULL_WARP_MASK UINT64_MAX
static const std::vector<int32_t> *ballot_ids;
static uint64_t ballot_first;
static uint64_t __ballot_sync(uint64_t, bool predicate) {
    uint64_t mask=0;
    for (uint32_t lane=0; lane<warpSize; ++lane) {
        const uint64_t pair=ballot_first+lane;
        if (pair<ballot_ids->size() && uint32_t(std::max((*ballot_ids)[pair], 0))==blockIdx.x)
            mask |= uint64_t(1)<<lane;
    }
    check(predicate==bool(mask & (uint64_t(1)<<threadIdx.x)), "production ballot predicate");
    ballot_first += warpSize;
    return mask;
}
// These mocks verify the actual production load pointers, leading dimensions
// and the order of calls; they deliberately do not emulate WMMA arithmetic.
struct MockFragment { int tag; };
static half *mma_a,*mma_b0,*mma_b1;
static uint32_t mma_wave,mma_stride,mma_phase,mma_next_acc;
namespace rocwmma {
static void load_matrix_sync(MockFragment &f,half *ptr,uint32_t leading) {
    check(leading==mma_stride,"actual MMA leading dimension");
    half *expected=f.tag==0 ? mma_a+mma_wave*16*mma_stride+mma_phase :
                   f.tag==1 ? mma_b0+mma_phase : mma_b1+mma_phase;
    check(ptr==expected,"actual MMA A row-major/B column-major K-phase pointer");
}
static void mma_sync(MockFragment &out,const MockFragment &a,const MockFragment &b,const MockFragment &acc) {
    check(&out==&acc && a.tag==0 && b.tag==int(mma_next_acc+1) && out.tag==int(mma_next_acc+3),
          "actual MMA accumulators and unchanged acc0/acc1 K order");
    if(++mma_next_acc==2) {mma_next_acc=0;mma_phase+=16;}
}
}
#endif

// PRODUCTION_SOURCE

static constexpr size_t guard=16;
static constexpr uint32_t poison=0xfbcaf12du;
static uint16_t half_bits(half x) { uint16_t u; std::memcpy(&u,&x,2); return u; }
static half half_from_bits(uint16_t u) { half x; std::memcpy(&x,&u,2); return x; }
static half host_half(float x) { return half(x); }
static float host_float(half x) { return float(x); }

static std::vector<uint32_t> offsets_for(const std::vector<int32_t>& ids, uint32_t experts) {
    std::vector<uint32_t> offsets(experts+1,0);
    for (int32_t id: ids) {
        const uint32_t e=uint32_t(std::max(id,0));
        if (e<experts) ++offsets[e+1];
    }
    for (uint32_t e=0;e<experts;++e) offsets[e+1]+=offsets[e];
    return offsets;
}
static std::vector<uint32_t> sorted_reference(const std::vector<int32_t>& ids,
                                            const std::vector<uint32_t>& offsets) {
    std::vector<uint32_t> result(guard+ids.size()+guard,poison);
    for (uint32_t e=0;e+1<offsets.size();++e) {
        uint32_t pos=offsets[e];
        for (size_t i=0;i<ids.size();++i)
            if (uint32_t(std::max(ids[i],0))==e) result[guard+pos++]=uint32_t(i);
    }
    return result;
}

#ifndef TEST_NATIVE
static void test_routing() {
    std::mt19937 random(5903);
    size_t cases=0;
    for (uint32_t width: {32u,64u})
    for (uint32_t experts: {1u,6u,256u})
    for (uint32_t n: {0u,1u,31u,32u,33u,63u,64u,65u,127u,128u,512u,2048u,4096u})
    for (uint32_t pattern=0;pattern<4;++pattern) {
        std::vector<int32_t> ids(size_t(n)*6u);
        for (size_t i=0;i<ids.size();++i) {
            if (pattern==0) ids[i]=int32_t(i%experts);
            if (pattern==1) ids[i]=0;
            if (pattern==2) ids[i]=int32_t(random()%experts);
            if (pattern==3) ids[i]= i%7==0 ? -7 : i%11==0 ? int32_t(experts) : int32_t(random()%experts);
        }
        const auto offsets=offsets_for(ids,experts);
        const auto expected=sorted_reference(ids,offsets);
        std::vector<uint32_t> actual(expected.size(),poison);
        ballot_ids=&ids; warpSize=width; blockDim.x=64;
        for (blockIdx.x=0;blockIdx.x<=experts;++blockIdx.x)
        for (threadIdx.x=0;threadIdx.x<64;++threadIdx.x) {
            ballot_first=0;
            moe_scatter_sorted_pairs_deterministic_kernel(actual.data()+guard,offsets.data(),ids.data(),uint32_t(ids.size()),experts);
        }
        check(actual==expected,"stable routing order/guards/invalid IDs/wave tails");
        ++cases;
    }
    // Validate the last window of large pair counts without allocating them.
    for (uint32_t width: {32u,64u})
    for (uint64_t n: {uint64_t(UINT32_MAX)-1, uint64_t(UINT32_MAX)}) {
        uint64_t first=((n-1)/width)*width;
        uint32_t valid=0;
        for (uint32_t lane=0;lane<width;++lane) valid+=(first+lane<n);
        check(valid==n-first && first+width>=n && first+width>first,"64-bit routing window termination");
    }
    std::printf("PASS %zu extracted routing cases: wave32/64, ordering, guards and UINT32_MAX windows\n",cases);
}

static half reference_weight(const unsigned char *blk,uint32_t k) {
    uint16_t db,mb;
    std::memcpy(&db,blk+80,2); std::memcpy(&mb,blk+82,2);
    const float d=host_float(half_from_bits(db)), dm=host_float(half_from_bits(mb));
    const uint32_t group=k/16;
    const uint32_t byte=16+(k/128)*32+(k%32);
    const uint32_t q=(blk[byte]>>(2*((k%128)/32)))&3u;
    const float scale=float(blk[group]&15u), min=float(blk[group]>>4);
    return host_half((d*scale)*float(q)-dm*min);
}

template<int STAGE_K> static size_t test_staging(std::mt19937& random) {
    constexpr uint32_t rows=32, words=21;
    size_t cases=0;
    for(uint32_t threads: {128u,256u})
    for(uint32_t seed=0;seed<8;++seed) {
        std::vector<uint32_t> raw(rows*words);
        auto bytes=reinterpret_cast<unsigned char*>(raw.data());
        for(size_t i=0;i<raw.size()*4;++i) bytes[i]=uint8_t(random());
        for(uint32_t r=0;r<rows;++r) {
            const uint16_t db=half_bits(host_half(seed==0 ? 0.0f : float((r%7)+1)*0.0625f));
            const uint16_t mb=half_bits(host_half(seed==1 ? 0.0f : float((r%5)+1)*0.03125f));
            std::memcpy(bytes+r*84+80,&db,2); std::memcpy(bytes+r*84+82,&mb,2);
            if(seed==7 && r>=17) std::memset(bytes+r*84,0,84); // output-tail rows
        }
        for(uint32_t k0=0;k0<256;k0+=STAGE_K) {
            std::vector<uint16_t> actual0(guard+16*STAGE_K+guard,0x5ad5), actual1=actual0;
            auto expect0=actual0,expect1=actual1;
            for(uint32_t r=0;r<32;++r)
            for(uint32_t k=0;k<STAGE_K;++k)
                (r<16 ? expect0 : expect1)[guard+(r%16)*STAGE_K+k]=half_bits(reference_weight(bytes+r*84,k0+k));
            blockDim.x=threads;
            for(uint32_t tid=0;tid<threads;++tid)
                q2_K_dequant_pair_tile_half_rowwise_staged<16,STAGE_K>(
                    reinterpret_cast<half*>(actual0.data()+guard),reinterpret_cast<half*>(actual1.data()+guard),raw.data(),k0,tid);
            check(actual0==expect0 && actual1==expect1,"extracted Q2 dequant/scale groups/LDS output guards");
            ++cases;
        }
    }
    for(uint32_t kdim: {256u,512u,2048u})
    for(uint32_t valid: {0u,1u,7u,8u,15u,16u,63u,64u})
    for(bool f16: {false,true}) {
        // Offset the half input by4bytes: new K32 must retain old half2 ABI,
        // with no implicit16-byte alignment requirement.
        std::vector<half> input_h(size_t(67)*kdim+2);
        std::vector<float> input_f(size_t(67)*kdim);
        for(size_t i=0;i<input_f.size();++i) {input_f[i]=float(int(i%101)-50)*0.03125f;input_h[i+2]=host_half(input_f[i]);}
        uint32_t pairs[64];
        for(uint32_t r=0;r<64;++r) pairs[r]=r<valid ? 66-r : UINT32_MAX;
        for(uint32_t k0=0;k0<kdim;k0+=STAGE_K) {
            std::vector<uint16_t> actual(guard+64*STAGE_K+guard,0x5ad5),expected=actual;
            for(uint32_t r=0;r<64;++r)
            for(uint32_t k=0;k<STAGE_K;++k)
                expected[guard+r*STAGE_K+k]= r<valid ? half_bits(f16 ? input_h[2+size_t(pairs[r])*kdim+k0+k] : host_half(input_f[size_t(pairs[r])*kdim+k0+k])) : 0;
            blockDim.x=128;
            for(uint32_t tid=0;tid<128;++tid) {
                if(f16) stage_a<true,STAGE_K>(reinterpret_cast<half*>(actual.data()+guard),pairs,input_f.data(),input_h.data()+2,kdim,k0,tid);
                else stage_a<false,STAGE_K>(reinterpret_cast<half*>(actual.data()+guard),pairs,input_f.data(),input_h.data()+2,kdim,k0,tid);
            }
            check(actual==expected,"extracted A staging pair map/half2 alignment/guards");
            ++cases;
        }
    }
    // Extracted production LDS pointer declarations with exact requested
    // bytes. C's K32 alias may write only A, and no trailing guard is touched.
    constexpr size_t scratch=(64+32)*STAGE_K*2+32*84+(STAGE_K==32 ? 0 : 64*16*4);
    std::vector<uint32_t> lds(guard+scratch/4+guard,poison);
    check_layout<STAGE_K>(reinterpret_cast<unsigned char*>(lds.data()+guard),scratch);
    for(size_t i=0;i<guard;++i) check(lds[i]==poison && lds[guard+scratch/4+i]==poison,"LDS epilogue alias canaries");
    const size_t c_first=STAGE_K==32 ? 0 : ((64+32)*STAGE_K*2+32*84)/4;
    for(size_t i=0;i<scratch/4;++i)
        if(i<c_first || i>=c_first+64*16) check(lds[guard+i]==poison,"C epilogue preserves B/raw weights and other LDS");
    // Exercise the original MMA source block with pointer/order mocks.
    std::vector<half> a(64*STAGE_K), b0(16*STAGE_K), b1(16*STAGE_K);
    for(uint32_t wave=0;wave<5;++wave) {
        mma_a=a.data();mma_b0=b0.data();mma_b1=b1.data();
        mma_wave=wave;mma_stride=STAGE_K;mma_phase=0;mma_next_acc=0;
        mma_stage<STAGE_K>(a.data(),b0.data(),b1.data(),wave);
        check(mma_phase==(wave<4 ? STAGE_K : 0) && mma_next_acc==0,"complete MMA K-stage sequence");
        ++cases;
    }
    return cases;
}

int main() {
    test_routing();
    std::mt19937 random(891);
    const size_t cases=test_staging<16>(random)+test_staging<32>(random);
    std::printf("PASS %zu extracted Q2/A staging cases: K16/K32, all scale groups, tails, half2 alignment and guards\n",cases);
    std::puts("Host results do not validate HIP compilation, WMMA arithmetic or speed.");
}
#else
static void hip_check(hipError_t rc,const char *what) {
    if(rc!=hipSuccess) {std::fprintf(stderr,"%s: %s\n",what,hipGetErrorString(rc));std::exit(1);}
}
template<class T> struct Device {
    T *p=nullptr;
    size_t count;
    explicit Device(const std::vector<T>& values):count(values.size()) {
        hip_check(hipMalloc(reinterpret_cast<void**>(&p),count*sizeof(T)),"hipMalloc");
        hip_check(hipMemcpy(p,values.data(),count*sizeof(T),hipMemcpyHostToDevice),"upload");
    }
    ~Device(){hipFree(p);}
    Device(const Device&)=delete;
    std::vector<T> read() const {
        std::vector<T> values(count);
        hip_check(hipMemcpy(values.data(),p,count*sizeof(T),hipMemcpyDeviceToHost),"readback");return values;
    }
};

__global__ static void scatter_scalar_reference(uint32_t *out,const uint32_t *offsets,
                                               const int32_t *ids,uint32_t count,uint32_t experts) {
    const uint32_t expert=blockIdx.x;
    if(expert>=experts || threadIdx.x!=0) return;
    uint32_t pos=offsets[expert];
    for(uint32_t pair=0;pair<count;++pair) {
        int32_t id=ids[pair];if(id<0) id=0;
        if(uint32_t(id)==expert) out[pos++]=pair;
    }
}

template<class Launch> static double elapsed(Launch launch,int repeats) {
    hipEvent_t begin,end;
    hip_check(hipEventCreate(&begin),"event create");hip_check(hipEventCreate(&end),"event create");
    hip_check(hipEventRecord(begin),"event record");
    for(int i=0;i<repeats;++i) launch();
    hip_check(hipEventRecord(end),"event record");hip_check(hipEventSynchronize(end),"event sync");
    float ms=0;hip_check(hipEventElapsedTime(&ms,begin,end),"event elapsed");
    hipEventDestroy(begin);hipEventDestroy(end);return double(ms)*1000.0/repeats;
}
template<class Before,class After> static void time_pair(const char *tag,uint32_t tokens,
                                                        Before before,After after,int repeats) {
    for(int i=0;i<4;++i) {before();after();}
    hip_check(hipDeviceSynchronize(),"warmup");
    std::vector<double> a,b;
    for(int sample=0;sample<10;++sample) {
        double x,y;
        if(sample%2) {y=elapsed(after,repeats);x=elapsed(before,repeats);}
        else {x=elapsed(before,repeats);y=elapsed(after,repeats);}
        a.push_back(x);b.push_back(y);
        std::printf("sample,%s,N%u,%d,baseline_us,%.6f,candidate_us,%.6f\n",tag,tokens,sample,x,y);
    }
    std::sort(a.begin(),a.end());std::sort(b.begin(),b.end());
    const double am=(a[4]+a[5])*0.5,bm=(b[4]+b[5])*0.5;
    std::printf("median,%s,N%u,baseline_us,%.6f,candidate_us,%.6f,speedup_percent,%.3f\n",tag,tokens,am,bm,100*(am/bm-1));
}

static void test_routing_native(bool bench) {
    std::mt19937 random(7280);size_t cases=0;
    for(uint32_t experts: {1u,6u,256u})
    for(uint32_t tokens: {0u,1u,5u,11u,31u,32u,33u,128u,512u,2048u,4096u})
    for(uint32_t pattern=0;pattern<3;++pattern) {
        std::vector<int32_t> ids(size_t(tokens)*6u);
        for(size_t i=0;i<ids.size();++i) ids[i]=pattern==0 ? int32_t(i%experts) :
            pattern==1 ? 0 : i%7==0 ? -9 : i%11==0 ? int32_t(experts) : int32_t(random()%experts);
        const auto offsets=offsets_for(ids,experts);
        const auto expected=sorted_reference(ids,offsets);
        // Empty input still has a valid device allocation; candidate must not read it.
        auto storage=ids;if(storage.empty())storage.push_back(0x12345678);
        Device<int32_t> di(storage);Device<uint32_t> off(offsets);
        Device<uint32_t> a(std::vector<uint32_t>(expected.size(),poison)),b(std::vector<uint32_t>(expected.size(),poison));
        auto baseline=[&]{scatter_scalar_reference<<<experts+1,1>>>(a.p+guard,off.p,di.p,uint32_t(ids.size()),experts);};
        auto candidate=[&]{moe_scatter_sorted_pairs_deterministic_kernel<<<experts+1,64>>>(b.p+guard,off.p,di.p,uint32_t(ids.size()),experts);};
        baseline();candidate();hip_check(hipGetLastError(),"routing launch");
        check(a.read()==expected && b.read()==expected,"native stable routing/guards/invalid IDs");
        check(di.read()==storage && off.read()==offsets,"immutable routing inputs");
        if(bench && experts==256 && pattern==0 && tokens>=128)
            time_pair("stable-routing",tokens,baseline,candidate,32);
        ++cases;
    }
    std::printf("PASS %zu native routing cases\n",cases);
}

struct DownFixture {
    uint32_t n,experts,m,k,total_pairs,hot_max=0;
    std::vector<uint32_t> counts,offsets,pairs,hot;
    std::vector<uint32_t> weight_words;
    std::vector<half> mid_h;
    std::vector<float> mid;
    static constexpr uint32_t slots=6;
    static constexpr size_t mid_guard=guard+2; // half2 aligned, deliberately not16 aligned
    DownFixture(uint32_t tokens,uint32_t outputs,uint32_t inner,std::vector<uint32_t> c):
        n(tokens),experts(uint32_t(c.size())),m(outputs),k(inner),total_pairs(tokens*slots),counts(std::move(c)) {
        offsets.resize(experts+1,0);
        for(uint32_t e=0;e<experts;++e) {
            offsets[e+1]=offsets[e]+counts[e];
            if(counts[e]>=8) {hot.push_back(e);hot_max=std::max(hot_max,counts[e]);}
        }
        check(offsets.back()<=total_pairs,"fixture counts fit pair capacity");
        pairs.resize(offsets.back());
        std::vector<uint32_t> permutation(total_pairs);
        for(uint32_t i=0;i<total_pairs;++i)permutation[i]=i;
        std::mt19937 random(412);std::shuffle(permutation.begin(),permutation.end(),random);
        for(size_t i=0;i<pairs.size();++i)pairs[i]=permutation[i];
        const size_t words=size_t(experts)*m*(k/256)*21;
        weight_words.assign(guard+words+guard,poison);
        for(size_t i=0;i<words;++i) weight_words[guard+i]=uint32_t(i)*2654435761u+0x2ae5091du;
        for(size_t block=0;block<words/21;++block) {
            const uint16_t d=half_bits(host_half(block%13==0 ? 0.0f : 0.00390625f));
            const uint16_t dm=half_bits(host_half(block%17==0 ? 0.0f : 0.001953125f));
            weight_words[guard+block*21+20]=uint32_t(d)|(uint32_t(dm)<<16);
        }
        mid.assign(guard+size_t(total_pairs)*k+guard,-1234.0f);
        mid_h.assign(mid_guard+size_t(total_pairs)*k+guard,host_half(-1234.0f));
        for(size_t i=0;i<size_t(total_pairs)*k;++i) {
            const float value=float(int(i%197)-98)*0.015625f;
            mid[guard+i]=value;mid_h[mid_guard+i]=host_half(value);
        }
    }
};

template<bool MID_F16,bool OUT_F16,bool SLOT_MAJOR>
static void down_case(DownFixture& f,bool timing) {
    Device<uint32_t> dw(f.weight_words),dc(f.counts),doff(f.offsets),dp(f.pairs),dh(f.hot);
    Device<half> dxh(f.mid_h);Device<float> dx(f.mid);
    using Out=typename std::conditional<OUT_F16,uint16_t,uint32_t>::type;
    const Out sentinel=OUT_F16 ? Out(0x7e55u) : Out(0x7fcaf12du);
    const std::vector<Out> initial(guard+size_t(f.total_pairs)*f.m+guard,sentinel);
    Device<Out> a(initial),b(initial);
    const dim3 block(128),grid((f.m+31)/32,(f.hot_max+63)/64,uint32_t(f.hot.size()));
    const uint64_t row_bytes=uint64_t(f.k/256)*84,expert_bytes=uint64_t(f.m)*row_bytes;
    auto launch=[&](bool candidate) {
        auto target=candidate ? b.p : a.p;
        float *out=OUT_F16 ? nullptr : reinterpret_cast<float*>(target+guard);
        half *outh=OUT_F16 ? reinterpret_cast<half*>(target+guard) : nullptr;
        const size_t stage=candidate ? 32 : 16;
        const size_t scratch=(64*stage+2*stage*16)*sizeof(half)+(candidate ? 0 : 64*16*sizeof(float))+32*84;
        if(candidate)
            moe_down_q2K_hotlist_wmma_n2_kernel<4,16,16,16,MID_F16,OUT_F16,SLOT_MAJOR,32><<<grid,block,scratch>>>(
                out,outh,reinterpret_cast<const char*>(dw.p+guard),dx.p+guard,dxh.p+DownFixture::mid_guard,
                dc.p,doff.p,dp.p,dh.p,uint32_t(f.hot.size()),f.k,f.m,expert_bytes,row_bytes,6,f.n);
        else
            moe_down_q2K_hotlist_wmma_n2_kernel<4,16,16,16,MID_F16,OUT_F16,SLOT_MAJOR,16><<<grid,block,scratch>>>(
                out,outh,reinterpret_cast<const char*>(dw.p+guard),dx.p+guard,dxh.p+DownFixture::mid_guard,
                dc.p,doff.p,dp.p,dh.p,uint32_t(f.hot.size()),f.k,f.m,expert_bytes,row_bytes,6,f.n);
    };
    check(!f.hot.empty(),"fixture needs a hot expert");
    launch(false);launch(true);hip_check(hipGetLastError(),"Q2 WMMA launch");
    auto ar=a.read(),br=b.read();
    check(ar==br,"native K16/K32 exact output bits including guards");
    std::vector<bool> written(initial.size(),false);
    for(uint32_t e:f.hot)
    for(uint32_t p=f.offsets[e];p<f.offsets[e+1];++p)
    for(uint32_t row=0;row<f.m;++row) {
        const uint32_t pair=f.pairs[p];
        const size_t dst=SLOT_MAJOR ? (size_t(pair%6)*f.n+pair/6)*f.m+row : size_t(pair)*f.m+row;
        written[guard+dst]=true;
    }
    size_t writes=0;
    for(size_t i=0;i<ar.size();++i) {
        if(!written[i]) check(ar[i]==initial[i],"native WMMA inactive outputs/canaries");
        else {
            // Integer check remains effective even when HIPCC host flags
            // assume finite arithmetic and simplify std::isfinite below.
            check(ar[i]!=sentinel,"every active native output was written");
            float value;
            if(OUT_F16) value=host_float(half_from_bits(uint16_t(ar[i])));
            else {uint32_t bits=uint32_t(ar[i]);std::memcpy(&value,&bits,4);}
            check(std::isfinite(value),"native WMMA finite output");++writes;
        }
    }
    check(writes>0 && ar!=initial,"native WMMA produced output");
    check(dc.read()==f.counts && doff.read()==f.offsets && dp.read()==f.pairs && dh.read()==f.hot,"immutable routing maps");
    check(dw.read()==f.weight_words && dxh.read().size()==f.mid_h.size(),"immutable Q2 weights");
    const auto got_h=dxh.read();const auto got_x=dx.read();
    check(std::memcmp(got_h.data(),f.mid_h.data(),got_h.size()*sizeof(half))==0 && got_x==f.mid,"immutable activations and input guards");
    std::printf("PASS Q2 K16/K32 M%u K%u N%u MID_F16%d OUT_F16%d SLOT_MAJOR%d hot%zu\n",f.m,f.k,f.n,int(MID_F16),int(OUT_F16),int(SLOT_MAJOR),f.hot.size());
    if(timing) {
        std::printf("benchmark,q2-down-hot-wmma,N%u,M%u,K%u,E%u,hot_count%zu,hot_max%u,fixed_input_warm_weights,cold_scalar_excluded\n",
                    f.n,f.m,f.k,f.experts,f.hot.size(),f.hot_max);
        time_pair("q2-down-hot-wmma",f.n,[&]{launch(false);},[&]{launch(true);},8);
    }
}

int main(int argc,char **argv) {
    const int device=argc>1 ? std::atoi(argv[1]) : 0;
    const bool bench=argc>2 && std::strcmp(argv[2],"--bench")==0;
    hip_check(hipSetDevice(device),"device");hipDeviceProp_t prop{};
    hip_check(hipGetDeviceProperties(&prop,device),"device properties");
    std::printf("GPU %s %s wave%d\n",prop.name,prop.gcnArchName,prop.warpSize);
    test_routing_native(bench);
    check(std::strncmp(prop.gcnArchName,"gfx1151",7)==0 && prop.warpSize==32,
          "K32 production timing/correctness requires gfx1151 wave32");
    size_t cases=0;
    for(uint32_t k: {256u,512u,2048u})
    for(uint32_t m: {16u,31u,48u,65u}) {
        DownFixture f(40,m,k,{0,1,7,8,15,16,63,64,65});
        down_case<true,true,false>(f,false);
        down_case<false,true,false>(f,false);
        down_case<true,false,false>(f,false);
        down_case<false,false,false>(f,false);
        down_case<true,true,true>(f,false);
        cases+=5;
    }
    std::printf("PASS %zu native Q2 down cases\n",cases);
    if(bench) {
        for(uint32_t n: {128u,512u,2048u,4096u}) {
            const uint32_t active=std::min(256u,n*6u/12u);
            std::vector<uint32_t> counts(256,0);
            for(uint32_t pair=0;pair<n*6;++pair)++counts[pair%active];
            std::printf("routing_fixture,N%u,top_k6,active_experts%u,E256,uniform_within_active_set\n",n,active);
            DownFixture f(n,4096,2048,std::move(counts));
            down_case<true,true,false>(f,true);
        }
    }
}

#endif
