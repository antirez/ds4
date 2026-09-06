// SPDX-License-Identifier: MIT
// Native gfx1151 parity/benchmark for output-B Q4_K TILE8 aligned LDS.
// Both public-API arms include the unchanged F32 -> Q8_K quantizer.
#include "ds4_gpu.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <stdexcept>
#include <sys/mman.h>
#include <utility>
#include <vector>
#if defined(__has_include)
#if __has_include(<hip/hip_runtime.h>)
#include <hip/hip_runtime.h>
#define DS4_LDS_HAS_HIP 1
#endif
#endif
#ifndef DS4_LDS_HAS_HIP
#define DS4_LDS_HAS_HIP 0
#endif

extern "C" void ds4_rocm_test_q4_prefill_lds_aligned_reset(void);
extern "C" uint64_t ds4_rocm_test_q4_prefill_lds_aligned_get_calls(void);
extern "C" void ds4_rocm_test_q4_prefill_wmma_reset(void);
extern "C" uint64_t ds4_rocm_test_q4_prefill_wmma_get_calls(void);
extern "C" const void *ds4_rocm_bench_q4_K_resident_weight_ptr(
    const void *, uint64_t, uint64_t, uint64_t);

namespace {
constexpr uint32_t K = 8192u, M = 4096u, q4_type = 12u;
constexpr size_t guard = 260u; // 16-byte aligned views for attention-A WMMA.
constexpr const char *rollback = "DS4_ROCM_DISABLE_Q4_PREFILL_LDS_ALIGNED";
constexpr const char *wmma_disable = "DS4_ROCM_DISABLE_Q4_PREFILL_WMMA";
void check(bool ok, const char *what) {
    if (!ok) throw std::runtime_error(what);
}
void env(const char *key, const char *value) {
    check((value ? setenv(key,value,1) : unsetenv(key)) == 0,"environment update");
}
void controls() {
    // This dedicated process must exercise production defaults independently
    // of diagnostic assertions inherited from the invoking benchmark shell.
    const char *keys[] = {rollback,
        "DS4_ROCM_DISABLE_Q4_PREFILL_TILE8", "DS4_ROCM_REQUIRE_Q4_PREFILL_TILE8",
        "DS4_ROCM_REQUIRE_Q4_PREFILL_K1024_TILE4",
        "DS4_ROCM_REQUIRE_Q4_DECODE_LANE4",
        "DS4_ROCM_ENABLE_Q4_DECODE_LANE4", "DS4_ROCM_DISABLE_Q4_DECODE_LANE4",
        "DS4_ROCM_ENABLE_Q4_PREFILL_WMMA", "DS4_ROCM_ENABLE_Q4_PREFILL_WMMA_SSD",
        "DS4_ROCM_REQUIRE_Q4_PREFILL_WMMA", "DS4_ROCM_Q4_PREFILL_WMMA_ROW_TILE",
        "DS4_ROCM_ENABLE_Q4_PREFILL_WMMA_K64",
        "DS4_ROCM_DISABLE_Q4_PREFILL_WMMA_K128",
        "DS4_ROCM_DISABLE_Q4_PREFILL_WMMA_K64_LOAD4",
        "DS4_ROCM_ENABLE_Q4_PREFILL_Q8_K_WAVE32",
        "DS4_ROCM_DISABLE_Q4_PREFILL_Q8_K_WAVE32",
        "DS4_ROCM_REQUIRE_Q4_PREFILL_Q8_K_WAVE32",
        "DS4_ROCM_DISABLE_Q4_PREFILL_LDS_STREAM",
        "DS4_ROCM_DISABLE_Q4_PREFILL_LDS_VECTOR"};
    for (const char *key : keys) env(key,nullptr);
    env(wmma_disable,"1");
}
struct tensor {
    ds4_gpu_tensor *p;
    explicit tensor(uint64_t bytes) : p(ds4_gpu_tensor_alloc(bytes)) {
        check(p != nullptr,"tensor allocation");
    }
    tensor(const tensor &base, uint64_t offset, uint64_t bytes)
        : p(ds4_gpu_tensor_view(base.p,offset,bytes)) { check(p != nullptr,"tensor view"); }
    ~tensor() { ds4_gpu_tensor_free(p); }
    tensor(const tensor &) = delete;
};
struct guarded {
    size_t count;
    std::vector<float> initial;
    tensor storage, view;
    guarded(uint32_t n, uint32_t columns, bool input = false, bool zeros = false)
        : count((size_t)n*columns), initial(guard+count+63u*columns+guard,123456.0f),
          storage(initial.size()*sizeof(float)), view(storage,guard*sizeof(float),count*sizeof(float)) {
        for (size_t i = 0; i < count; ++i) {
            float value = -987654.0f;
            if (input) {
                value = zeros ? ((i & 1u) ? -0.0f : 0.0f)
                    : (float)((int)((i*73u+i/256u*19u)%241u)-120)/32.0f;
                if (!zeros && i%256u == 0u) value = (i/256u & 1u) ? 4.0f : -4.0f;
                if (!zeros && i/256u%17u == 0u) value = 0.0f;
            }
            initial[guard+i] = value;
        }
        prepare();
    }
    void prepare() {
        check(ds4_gpu_tensor_write(storage.p,0,initial.data(),initial.size()*sizeof(float)) != 0,"write");
    }
    std::vector<float> read(bool output) {
        std::vector<float> got(initial.size());
        check(ds4_gpu_tensor_read(storage.p,0,got.data(),got.size()*sizeof(float)) != 0,"read");
        check(std::memcmp(got.data(),initial.data(),guard*sizeof(float)) == 0,"prefix guard");
        check(std::memcmp(got.data()+guard+count,initial.data()+guard+count,
                          (got.size()-guard-count)*sizeof(float)) == 0,"suffix/N-tail guard");
        if (output) {
            for (size_t i = guard; i < guard+count; ++i)
                check(std::isfinite(got[i]) && got[i] != -987654.0f,"non-finite or unwritten output");
        } else check(std::memcmp(got.data(),initial.data(),got.size()*sizeof(float)) == 0,"input modified");
        return got;
    }
};
void equal(const std::vector<float> &got, const std::vector<float> &expected) {
    check(got.size() == expected.size(),"comparison size");
    if (std::memcmp(got.data(),expected.data(),got.size()*sizeof(float)) == 0) return;
    for (size_t i = 0; i < got.size(); ++i) {
        if (std::memcmp(&got[i],&expected[i],sizeof(float)) == 0) continue;
        std::fprintf(stderr,"first mismatch at %zu: %.9g vs %.9g\n",i,got[i],expected[i]);
        throw std::runtime_error("bitwise parity");
    }
}
struct q4_block { uint16_t d, dmin; uint8_t scales[12], qs[128]; };
static_assert(sizeof(q4_block) == 144u,"raw GGUF Q4_K layout");
struct model {
    // Both projections have 32M weights; keep page-aligned, nonzero offsets.
    uint64_t a = 4096u, b = a+(uint64_t)M*(K/256u)*sizeof(q4_block);
    uint64_t size = b+(uint64_t)M*(K/256u)*sizeof(q4_block);
    uint8_t *data = nullptr;
    FILE *file = nullptr;
    model() {
        std::vector<uint8_t> staging((size_t)size,0xa5);
        uint32_t rng = 0x31af0759u;
        auto random = [&]() { rng = rng*1664525u+1013904223u; return rng; };
        for (uint64_t offset : {a,b}) {
            auto *w = reinterpret_cast<q4_block *>(staging.data()+offset);
            for (size_t i = 0; i < (size_t)M*K/256u; ++i) {
                w[i].d = (uint16_t)(0x1800u+(random() & 1023u));
                w[i].dmin = (uint16_t)(0x1000u+(random() & 1023u));
                for (auto &v : w[i].scales) v = (uint8_t)(random() >> 24u);
                for (auto &v : w[i].qs) v = (uint8_t)(random() >> 24u);
            }
        }
        // Resident upload may discard source pages. A read-only file mapping
        // retains their backing; malloc-owned source pages cannot be used.
        FILE *opened = std::tmpfile();
        check(opened != nullptr,"model temporary file");
        if (std::fwrite(staging.data(),1u,(size_t)size,opened) != (size_t)size ||
            std::fflush(opened) != 0) {
            std::fclose(opened); throw std::runtime_error("model file write");
        }
        void *mapped = mmap(nullptr,(size_t)size,PROT_READ,MAP_PRIVATE,fileno(opened),0);
        if (mapped == MAP_FAILED) {
            std::fclose(opened); throw std::runtime_error("model file mapping");
        }
        data = static_cast<uint8_t *>(mapped); file = opened;
    }
    ~model() {
        if (data) (void)munmap(data,(size_t)size);
        if (file) std::fclose(file);
    }
    model(const model &) = delete;
};
void reset_counts() {
    ds4_rocm_test_q4_prefill_lds_aligned_reset();
    ds4_rocm_test_q4_prefill_wmma_reset();
}
void counts(bool candidate, bool attention) {
    const auto got = ds4_rocm_test_q4_prefill_lds_aligned_get_calls();
    if (got != (candidate ? 1u : 0u)) {
        std::fprintf(stderr,"aligned LDS calls=%llu expected=%u\n",
                     (unsigned long long)got,candidate ? 1u : 0u);
        throw std::runtime_error("aligned kernel dispatch");
    }
    check(ds4_rocm_test_q4_prefill_wmma_get_calls() == (attention ? 1u : 0u),"WMMA dispatch");
}
int dense(const model &m, guarded &out, const guarded &x, uint32_t n,
          uint32_t k = K, uint32_t rows = M) {
    return ds4_gpu_matmul_quant_tensor(out.view.p,m.data,m.size,m.b,q4_type,k,rows,x.view.p,n);
}
void dense_case(const model &m, uint32_t n, uint32_t k = K, uint32_t rows = M,
                bool admitted = true, bool zeros = false) {
    guarded x(n,k,true,zeros), out(n,rows);
    std::vector<float> ref;
    for (bool candidate : {false,true}) {
        env(rollback,candidate ? nullptr : "1"); out.prepare(); reset_counts();
        check(dense(m,out,x,n,k,rows) > 0,"dense API launch");
        check(ds4_gpu_synchronize() != 0,"dense completion");
        counts(candidate && admitted,false);
        auto got = out.read(true); x.read(false);
        if (candidate) equal(got,ref); else ref = std::move(got);
    }
    std::printf("PASS K=%u M=%u N=%u %s: bitwise, guards, dispatch\n",
                k,rows,n,zeros ? "zeros" : "finite");
}
void attention_case(const model &m) {
    constexpr uint32_t n = 257u;
    guarded heads(n,8u*4096u,true), low(n,K), out(n,M);
    std::vector<float> ref_low, ref_out;
    env(wmma_disable,nullptr);
    for (bool candidate : {false,true}) {
        env(rollback,candidate ? nullptr : "1"); low.prepare(); out.prepare(); reset_counts();
        check(ds4_gpu_attention_output_q4_K_batch_tensor(out.view.p,low.view.p,nullptr,nullptr,
              m.data,m.size,m.a,m.b,q4_type,4096u,1024u,8u,M,heads.view.p,n) == 1,"attention API launch");
        check(ds4_gpu_synchronize() != 0,"attention completion"); counts(candidate,true);
        auto got_low = low.read(true), got_out = out.read(true); heads.read(false);
        if (candidate) { equal(got_low,ref_low); equal(got_out,ref_out); }
        else { ref_low = std::move(got_low); ref_out = std::move(got_out); }
    }
    env(wmma_disable,"1");
    std::puts("PASS attention A-WMMA/B-TILE8 N257: both intermediates bitwise, guards, dispatch");
}
void oracle(const model &m) {
    for (uint32_t n : {256u,257u,511u,512u,1024u,2048u,4096u}) dense_case(m,n);
    dense_case(m,257u,K,M,true,true);
    for (uint32_t n : {8u,255u}) dense_case(m,n,K,M,false);
    dense_case(m,256u,4096u,M,false); dense_case(m,256u,K,M-1u,false);
    ds4_gpu_set_quality(true); dense_case(m,256u,K,M,false); ds4_gpu_set_quality(false);
    ds4_gpu_set_ssd_streaming(true); dense_case(m,256u,K,M,false); ds4_gpu_set_ssd_streaming(false);
    // The rollback switch uses presence semantics, including the literal "0".
    guarded x(256u,K,true), out(256u,M);
    for (const char *value : {"0",""}) {
        env(rollback,value); reset_counts(); out.prepare();
        check(dense(m,out,x,256u) > 0 && ds4_gpu_synchronize() != 0,"presence rollback");
        counts(false,false); out.read(true); x.read(false);
    }
    const auto ref = out.read(true);
    env(rollback,nullptr);
    for (const char *key : {"DS4_ROCM_DISABLE_Q4_PREFILL_LDS_STREAM",
                           "DS4_ROCM_DISABLE_Q4_PREFILL_LDS_VECTOR"}) {
        for (const char *value : {"1","0",""}) {
            env(key,value); reset_counts(); out.prepare();
            check(dense(m,out,x,256u) > 0 && ds4_gpu_synchronize() != 0,"older LDS rollback");
            counts(false,false); equal(out.read(true),ref); x.read(false);
        }
        env(key,nullptr);
    }
    attention_case(m);
}
#if DS4_LDS_HAS_HIP
struct events {
    hipEvent_t start{}, stop{};
    events() {
        check(hipEventCreate(&start) == hipSuccess,"start event");
        if (hipEventCreate(&stop) != hipSuccess) {
            (void)hipEventDestroy(start); throw std::runtime_error("stop event");
        }
    }
    ~events() { (void)hipEventDestroy(stop); (void)hipEventDestroy(start); }
    void begin() { check(hipEventRecord(start,0) == hipSuccess,"record start"); }
    double end() {
        check(hipEventRecord(stop,0) == hipSuccess,"record stop");
        check(hipEventSynchronize(stop) == hipSuccess,"event synchronize");
        float ms = 0;
        check(hipEventElapsedTime(&ms,start,stop) == hipSuccess,"elapsed time");
        check(ms > 0,"nonpositive elapsed time"); return ms;
    }
};
#endif
void benchmark(const model &m, uint32_t n, unsigned samples) {
#if DS4_LDS_HAS_HIP
    guarded x(n,K,true), out(n,M);
    std::vector<float> ref;
    for (unsigned warmup = 0; warmup < 4; ++warmup) {
        const bool candidate = (warmup & 1u) != 0;
        env(rollback,candidate ? nullptr : "1"); out.prepare(); reset_counts();
        check(dense(m,out,x,n) > 0 && ds4_gpu_synchronize() != 0,"warmup");
        counts(candidate,false);
        auto got = out.read(true); if (warmup) equal(got,ref); else ref = std::move(got);
    }
    events timer;
    double a = 0, b = 0;
    std::puts("sample,tokens,order,slot,arm,ms");
    for (unsigned sample = 0; sample < samples; ++sample) {
        const char *order = (sample & 1u) ? "BAAB" : "ABBA";
        for (unsigned slot = 0; slot < 4; ++slot) {
            const bool candidate = order[slot] == 'B';
            env(rollback,candidate ? nullptr : "1"); out.prepare(); reset_counts();
            check(ds4_gpu_synchronize() != 0,"pre-timing sync");
            timer.begin(); const int ok = dense(m,out,x,n); const double ms = timer.end();
            check(ok > 0,"timed launch"); counts(candidate,false);
            equal(out.read(true),ref); x.read(false);
            (candidate ? b : a) += ms;
            std::printf("%u,%u,%s,%u,%c,%.6f\n",sample,n,order,slot,order[slot],ms);
        }
    }
    std::printf("SUMMARY K=%u M=%u N=%u A_mean_ms=%.6f B_mean_ms=%.6f speedup_pct=%.3f\n",
                K,M,n,a/(2*samples),b/(2*samples),(a/b-1)*100);
#else
    (void)m; (void)n; (void)samples; throw std::runtime_error("HIP events unavailable");
#endif
}
void device() {
#if DS4_LDS_HAS_HIP
    int count = 0, current = -1; hipDeviceProp_t p{};
    check(hipGetDeviceCount(&count) == hipSuccess && count > 0 &&
          hipGetDevice(&current) == hipSuccess && hipGetDeviceProperties(&p,current) == hipSuccess,"HIP device required");
    std::printf("device=%d name=%s arch=%s wave=%d\n",current,p.name,p.gcnArchName,p.warpSize);
    check(p.warpSize == 32 && !std::strncmp(p.gcnArchName,"gfx1151",7) &&
          (p.gcnArchName[7] == '\0' || p.gcnArchName[7] == ':'),"gfx1151 wave32 required");
#else
    throw std::runtime_error("HIP runtime and gfx1151 device required");
#endif
}
struct backend {
    backend() { check(ds4_gpu_init() != 0,"GPU init"); }
    ~backend() { ds4_gpu_cleanup(); }
};
unsigned number(const char *s, unsigned lo, unsigned hi) {
    char *end = nullptr; const unsigned long n = std::strtoul(s,&end,10);
    check(*s && *s != '-' && end && !*end && n >= lo && n <= hi,"numeric argument");
    return (unsigned)n;
}
} // namespace
int main(int argc, char **argv) {
    try {
        bool bench = false; unsigned tokens = 0, samples = 8;
        for (int i = 1; i < argc; ++i) {
            if (!std::strcmp(argv[i],"--bench")) bench = true;
            else if (!std::strcmp(argv[i],"--tokens") && i+1 < argc) tokens = number(argv[++i],256,4096);
            else if (!std::strcmp(argv[i],"--samples") && i+1 < argc) samples = number(argv[++i],2,1000);
            else if (!std::strcmp(argv[i],"--help")) {
                std::puts("usage: test_rocm_q4_lds_aligned [--bench [--tokens N (256..4096)] [--samples 8]]"); return 0;
            } else throw std::runtime_error("unknown/missing argument; use --help");
        }
        check(!tokens || bench,"--tokens requires --bench");
        device(); controls(); model m; backend gpu;
        ds4_gpu_set_quality(false); ds4_gpu_set_ssd_streaming(false);
        const uint64_t offsets[] = {m.a,m.b}, sizes[] = {m.b-m.a,m.size-m.b};
        check(ds4_gpu_set_model_fd(fileno(m.file)) != 0 &&
              ds4_gpu_set_model_map_spans(m.data,m.size,offsets,sizes,2u,
                                          std::max(sizes[0],sizes[1])) != 0 &&
              ds4_gpu_synchronize() != 0,"resident model upload");
        for (unsigned i = 0; i < 2; ++i)
            check(ds4_rocm_bench_q4_K_resident_weight_ptr(m.data,m.size,offsets[i],sizes[i]) != nullptr,
                  "projection weights are not device-resident");
        if (bench) {
            std::puts("HIP events: public output-B Q8_K quantizer + Q4_K TILE8; uploads/readbacks excluded.");
            if (tokens) benchmark(m,tokens,samples);
            else for (uint32_t n : {2048u,4096u}) benchmark(m,n,samples);
            std::puts("PASS timed bitwise parity/guards/dispatch. Full-model prefill t/s needs a separate A/B.");
        } else oracle(m);
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr,"FAIL Q4 aligned LDS: %s\n",e.what()); return 1;
    }
}
