// SPDX-License-Identifier: MIT
// Native ROCm oracle and HIP-event A/B for the production Q4 F32 epilogue.
// No model, Python or remote connection required. Both arms start from the
// same input; never time repeated in-place normalizations of changing data.
#include "ds4_gpu.h"
#include "rocm/ds4_rocm_q4_qb_epilogue_layout.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <vector>
#if defined(__has_include)
#if __has_include(<hip/hip_runtime.h>)
#include <hip/hip_runtime.h>
#define DS4_EP_HAS_HIP 1
#endif
#endif
#ifndef DS4_EP_HAS_HIP
#define DS4_EP_HAS_HIP 0
#endif

extern "C" int ds4_rocm_test_q4_qb_f32_epilogue(
    ds4_gpu_tensor *, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    uint32_t, bool, float, float, float, float, float, float, float);
extern "C" void ds4_rocm_test_q4_qb_f32_epilogue_reset(void);
extern "C" uint64_t ds4_rocm_test_q4_qb_f32_epilogue_get_calls(void);

namespace {
namespace ep = ds4_rocm_q4_qb_epilogue;
constexpr const char *rollback = "DS4_ROCM_DISABLE_Q4_QB_F32_EPILOGUE";
constexpr size_t guard = 257;
void check(bool ok, const char *what) {
    if (!ok) throw std::runtime_error(what);
}
void set(const char *key, const char *value) {
    check((value ? setenv(key,value,1) : unsetenv(key)) == 0, "environment update");
}
struct saved_env {
    const char *key; bool present; std::string value;
    explicit saved_env(const char *k) : key(k), present(getenv(k) != nullptr) {
        if (present) value = getenv(k);
    }
    ~saved_env() {
        if (present) (void)setenv(key,value.c_str(),1); else (void)unsetenv(key);
    }
};
struct tensor {
    ds4_gpu_tensor *p;
    explicit tensor(uint64_t bytes) : p(ds4_gpu_tensor_alloc(bytes)) {
        check(p != nullptr,"allocation");
    }
    tensor(const tensor &base, uint64_t offset, uint64_t bytes)
        : p(ds4_gpu_tensor_view(base.p,offset,bytes)) { check(p != nullptr,"view"); }
    ~tensor() { ds4_gpu_tensor_free(p); }
    tensor(const tensor &) = delete;
};
struct params {
    uint32_t n = 256, heads = 64, dim = 512, rot = 64, pos = 8192, orig = 65536;
    bool inverse = false;
    float base = 10000.0f, scale = 0.25f, ext = 1.0f, attn = 1.0f;
    float fast = 32.0f, slow = 1.0f, eps = 1e-6f;
};
int invoke(ds4_gpu_tensor *x, const params &p, bool generic = false) {
    auto fn = generic ? ds4_gpu_head_rms_norm_rope_tail_tensor
                      : ds4_rocm_test_q4_qb_f32_epilogue;
    return fn(x,p.n,p.heads,p.dim,p.rot,p.pos,p.orig,p.inverse,
              p.base,p.scale,p.ext,p.attn,p.fast,p.slow,p.eps);
}
struct fixture {
    params p;
    size_t offset, count;
    std::vector<float> input;
    tensor storage, view;
    explicit fixture(params args, unsigned skew = 0, bool exceptional = false)
        : p(args), offset(guard + skew), count((size_t)p.n*p.heads*p.dim),
          input(offset + count + guard,123456.0f),
          storage(input.size()*sizeof(float)), view(storage,offset*sizeof(float),count*sizeof(float)) {
        uint32_t rng = 0x612aa759u;
        for (size_t i = 0; i < count; ++i) {
            rng = rng * 1664525u + 1013904223u;
            float v = (float)((int)(rng >> 16) - 32768) / 4096.0f;
            switch ((i / p.dim) % 13u) {
                case 0: v = (i & 1u) ? -0.0f : 0.0f; break;
                case 1: v = std::ldexp(v,-130); break;
                case 2: v = std::ldexp(v,(i & 1u) ? 20 : -20); break;
                case 3: v = i % p.dim == p.dim-1u ? 1.0f : 0.0f; break;
                case 4: v = i % p.dim == 0u ? 4096.0f : 1.0f; break;
                default: break;
            }
            input[offset+i] = v;
        }
        if (exceptional) {
            const uint32_t special[] = {0x7f800000u,0xff800000u,0x7fc01234u};
            for (unsigned i = 0; i < 3; ++i)
                std::memcpy(&input[offset+(size_t)i*p.dim+511u], &special[i],sizeof(float));
        }
    }
    void prepare() {
        check(ds4_gpu_tensor_write(storage.p,0,input.data(),input.size()*sizeof(float)) != 0,"write");
        check(ds4_gpu_synchronize() != 0,"prepare sync");
    }
    std::vector<float> read() {
        std::vector<float> out(input.size());
        check(ds4_gpu_tensor_read(storage.p,0,out.data(),out.size()*sizeof(float)) != 0,"read");
        check(std::memcmp(out.data(),input.data(),offset*sizeof(float)) == 0,"prefix guard");
        check(std::memcmp(out.data()+offset+count,input.data()+offset+count,guard*sizeof(float)) == 0,"suffix guard");
        return out;
    }
    void compare(const std::vector<float> &ref, bool exceptional = false) {
        const auto got = read();
        if (std::memcmp(got.data(),ref.data(),got.size()*sizeof(float)) == 0) return;
        for (size_t i = 0; i < got.size(); ++i) {
            if (std::memcmp(&got[i],&ref[i],sizeof(float)) == 0) continue;
            // NaN payloads are not an output contract. All finite values and
            // infinities, including signed zero, must still match bitwise.
            if (exceptional && i >= offset && i < offset+count &&
                std::isnan(got[i]) && std::isnan(ref[i])) continue;
            std::fprintf(stderr,"mismatch at %zu: %.9g != %.9g\n",i,got[i],ref[i]);
            throw std::runtime_error("epilogue bitwise parity");
        }
    }
};
std::vector<float> reference(fixture &f) {
    set(rollback,"1"); f.prepare();
    ds4_rocm_test_q4_qb_f32_epilogue_reset();
    check(invoke(f.view.p,f.p) != 0,"rollback launch");
    check(ds4_gpu_synchronize() != 0,"rollback completion");
    check(ds4_rocm_test_q4_qb_f32_epilogue_get_calls() == 0,"rollback dispatch");
    return f.read();
}
void compare_arm(fixture &f, const std::vector<float> &ref,
                 const char *disable_value, bool expected, bool exceptional = false,
                 bool generic = false) {
    set(rollback,disable_value); f.prepare();
    ds4_rocm_test_q4_qb_f32_epilogue_reset();
    check(invoke(f.view.p,f.p,generic) != 0,"launch");
    check(ds4_gpu_synchronize() != 0,"completion");
    check(ds4_rocm_test_q4_qb_f32_epilogue_get_calls() == (expected ? 1u : 0u),"dispatch coverage");
    f.compare(ref,exceptional);
}
void oracle() {
    unsigned cases = 0;
    for (uint32_t n : {256u,257u,512u,2048u,4096u}) {
        params p; p.n = n;
        fixture f(p); const auto ref = reference(f);
        compare_arm(f,ref,nullptr,true);
        ++cases;
    }
    for (unsigned variant = 0; variant < 12; ++variant) {
        params p; p.n = 256 + (variant & 1u);
        p.pos = variant % 3 == 0 ? 0u : (variant % 3 == 1 ? 32768u : 1048576u);
        p.inverse = (variant & 1u) != 0;
        p.ext = variant % 3 == 0 ? 0.0f : (variant % 3 == 1 ? 0.5f : 1.0f);
        p.scale = variant & 2u ? 1.0f : 0.0625f;
        p.base = variant & 4u ? 1000000.0f : 10000.0f;
        p.attn = variant & 8u ? 1.25f : 1.0f;
        p.eps = variant & 2u ? 1e-5f : 1e-6f;
        fixture f(p,variant % 4); const auto ref = reference(f);
        compare_arm(f,ref,nullptr,true);
        ++cases;
    }
    for (unsigned exclusion = 0; exclusion < 8; ++exclusion) {
        params p;
        const uint32_t excluded_tokens[] = {1u,8u,255u,4097u};
        if (exclusion < 4) p.n = excluded_tokens[exclusion];
        if (exclusion == 4) p.heads = 32;
        if (exclusion == 5) p.heads = 128;
        if (exclusion == 6) p.dim = 576;
        if (exclusion == 7) p.rot = 128;
        fixture f(p); const auto ref = reference(f);
        compare_arm(f,ref,nullptr,false); ++cases;
    }
    params p;
    fixture f(p); const auto ref = reference(f);
    for (const char *value : {"0","false","off"}) compare_arm(f,ref,value,true);
    for (const char *value : {"1","","yes"}) compare_arm(f,ref,value,false);
    compare_arm(f,ref,nullptr,false,false,true); // Untyped norm API stays legacy.
    ds4_gpu_set_quality(true);
    compare_arm(f,ref,nullptr,false);
    ds4_gpu_set_quality(false);
    ds4_gpu_set_ssd_streaming(true);
    compare_arm(f,ref,nullptr,false);
    ds4_gpu_set_ssd_streaming(false);
    set(rollback,nullptr);
    ds4_rocm_test_q4_qb_f32_epilogue_reset();
    tensor short_view(f.storage,guard*sizeof(float),f.count*sizeof(float)-sizeof(float));
    check(invoke(nullptr,p) == 0 && invoke(short_view.p,p) == 0,"invalid span rejection");
    check(ds4_rocm_test_q4_qb_f32_epilogue_get_calls() == 0,"invalid span did not launch");
    fixture nf(p,0,true); const auto nf_ref = reference(nf);
    compare_arm(nf,nf_ref,nullptr,true,true);
    std::printf("PASS Q4 F32 epilogue: %u shape/YaRN cases plus controls, guards, "
                "generic/quality/SSD exclusions, invalid spans and non-finite classification.\n",cases);
}
#if DS4_EP_HAS_HIP
struct events {
    hipEvent_t start{}, stop{};
    events() {
        check(hipEventCreate(&start) == hipSuccess,"start event");
        if (hipEventCreate(&stop) != hipSuccess) {
            (void)hipEventDestroy(start); throw std::runtime_error("stop event");
        }
    }
    ~events() { (void)hipEventDestroy(stop); (void)hipEventDestroy(start); }
    void begin() { check(hipEventRecord(start,0) == hipSuccess,"event start"); }
    double end() {
        check(hipEventRecord(stop,0) == hipSuccess,"event stop");
        check(hipEventSynchronize(stop) == hipSuccess,"event completion");
        float ms = 0;
        check(hipEventElapsedTime(&ms,start,stop) == hipSuccess,"event elapsed");
        return ms;
    }
};
#endif
void benchmark(uint32_t tokens, unsigned samples) {
#if DS4_EP_HAS_HIP
    params p; p.n = tokens;
    fixture f(p); const auto ref = reference(f);
    compare_arm(f,ref,nullptr,true);
    for (unsigned w = 0; w < 4; ++w) {
        compare_arm(f,ref,w & 1u ? nullptr : "1",(w & 1u) != 0);
        compare_arm(f,ref,w & 1u ? "1" : nullptr,(w & 1u) == 0);
    }
    events timer;
    double a = 0, b = 0;
    std::puts("HIP events: epilogue dispatch/kernel only; reset/readback outside timing. "
              "Not GEMM, dequantization or model t/s.");
    std::puts("sample,tokens,order,slot,arm,ms");
    for (unsigned sample = 0; sample < samples; ++sample) {
        const char *order = sample & 1u ? "BAAB" : "ABBA";
        for (unsigned slot = 0; slot < 4; ++slot) {
            const bool candidate = order[slot] == 'B';
            set(rollback,candidate ? nullptr : "1");
            f.prepare();
            ds4_rocm_test_q4_qb_f32_epilogue_reset();
            timer.begin();
            const int ok = invoke(f.view.p,p);
            const double ms = timer.end();
            check(ok != 0 && ms > 0,"timed launch");
            check(ds4_rocm_test_q4_qb_f32_epilogue_get_calls() == (candidate ? 1u : 0u),"timed dispatch");
            f.compare(ref);
            (candidate ? b : a) += ms;
            std::printf("%u,%u,%s,%u,%c,%.6f\n",sample,tokens,order,slot,order[slot],ms);
        }
    }
    std::printf("SUMMARY tokens=%u A_mean_ms=%.6f B_mean_ms=%.6f speedup_pct=%.3f\n",
                tokens,a/(samples*2),b/(samples*2),(a/b-1)*100);
    std::puts("PASS timed parity/guards/dispatch. Model throughput and quality still require full-model A/B.");
#else
    (void)tokens; (void)samples;
    throw std::runtime_error("HIP events unavailable");
#endif
}
bool device() {
#if DS4_EP_HAS_HIP
    int count = 0, current = -1;
    hipDeviceProp_t prop{};
    if (hipGetDeviceCount(&count) != hipSuccess || count == 0 ||
        hipGetDevice(&current) != hipSuccess ||
        hipGetDeviceProperties(&prop,current) != hipSuccess) return false;
    std::printf("device=%d name=%s arch=%s wave=%d\n",current,prop.name,prop.gcnArchName,prop.warpSize);
    check(prop.warpSize == 32 && std::strncmp(prop.gcnArchName,"gfx1151",7) == 0 &&
          (prop.gcnArchName[7] == '\0' || prop.gcnArchName[7] == ':'),"oracle requires gfx1151 wave32");
    return true;
#else
    return false;
#endif
}
struct backend {
    backend() { check(ds4_gpu_init() != 0,"GPU init"); }
    ~backend() { ds4_gpu_cleanup(); }
};
unsigned number(const char *s, unsigned lo, unsigned hi) {
    char *end = nullptr;
    const unsigned long value = std::strtoul(s,&end,10);
    check(*s && *s != '-' && end && !*end && value >= lo && value <= hi,"numeric argument");
    return (unsigned)value;
}
} // namespace
int main(int argc, char **argv) {
    try {
        bool bench = false;
        unsigned tokens = 4096, samples = 12;
        for (int i = 1; i < argc; ++i) {
            if (!std::strcmp(argv[i],"--bench")) bench = true;
            else if (!std::strcmp(argv[i],"--tokens") && i+1 < argc) tokens = number(argv[++i],256,4096);
            else if (!std::strcmp(argv[i],"--samples") && i+1 < argc) samples = number(argv[++i],2,1000);
            else if (!std::strcmp(argv[i],"--help")) {
                std::puts("usage: test_rocm_q4_qb_epilogue [--bench --tokens 4096 --samples 12]"); return 0;
            } else throw std::runtime_error("unknown/missing argument; use --help");
        }
        if (!device()) {
            const char *required = getenv("DS4_TEST_REQUIRE_ROCM_DEVICE");
            if (required && *required && std::strcmp(required,"0")) throw std::runtime_error("HIP device required but unavailable");
            std::puts("SKIP Q4 F32 epilogue: no HIP device"); return 77;
        }
        backend gpu;
        saved_env saved(rollback), stats("DS4_ROCM_Q4_QB_F32_EPILOGUE_STATS");
        set("DS4_ROCM_Q4_QB_F32_EPILOGUE_STATS",nullptr);
        ds4_gpu_set_quality(false); ds4_gpu_set_ssd_streaming(false);
        if (bench) benchmark(tokens,samples); else oracle();
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr,"FAIL Q4 F32 epilogue: %s\n",e.what()); return 1;
    }
}
