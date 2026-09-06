// SPDX-License-Identifier: MIT
// Focused public-API Q4_K K1024/M32768/N1..8 oracle and native HIP-event A/B.
// No model download, Python, or full dense/pair/prefill suite is needed.
#include "ds4_gpu.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
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
#define DS4_LANE4_HAS_HIP 1
#endif
#endif
#ifndef DS4_LANE4_HAS_HIP
#define DS4_LANE4_HAS_HIP 0
#endif

extern "C" void ds4_rocm_test_q4_decode_lane4_reset(void);
extern "C" uint64_t ds4_rocm_test_q4_decode_lane4_get_calls(void);

namespace {
constexpr uint32_t K = 1024u, M = 32768u, type_q4 = 12u;
constexpr size_t guard = 257u;
constexpr const char *enable = "DS4_ROCM_ENABLE_Q4_DECODE_LANE4";
constexpr const char *disable = "DS4_ROCM_DISABLE_Q4_DECODE_LANE4";
constexpr const char *require = "DS4_ROCM_REQUIRE_Q4_DECODE_LANE4";

void check(bool ok, const char *what) {
    if (!ok) throw std::runtime_error(what);
}
void env(const char *key, const char *value) {
    check((value ? setenv(key, value, 1) : unsetenv(key)) == 0, "environment update");
}
struct saved_env {
    std::string key, value;
    bool present;
    explicit saved_env(const char *k) : key(k), present(std::getenv(k) != nullptr) {
        if (present) value = std::getenv(k);
    }
    ~saved_env() {
        if (present) (void)setenv(key.c_str(), value.c_str(), 1);
        else (void)unsetenv(key.c_str());
    }
};
struct controls {
    // The standalone API has no F16 sidecar; neutralize only competing
    // prefill assertions so the N=9 exclusion check is independent of shell state.
    std::vector<saved_env> saved;
    controls() {
        const char *keys[] = {enable, disable, require,
            "DS4_ROCM_REQUIRE_Q4_PREFILL_TILE8",
            "DS4_ROCM_DISABLE_Q4_PREFILL_TILE8",
            "DS4_ROCM_REQUIRE_Q4_PREFILL_K1024_TILE4",
            "DS4_ROCM_ENABLE_Q4_PREFILL_Q8_K_WAVE32",
            "DS4_ROCM_REQUIRE_Q4_PREFILL_Q8_K_WAVE32",
            "DS4_ROCM_REQUIRE_Q4_PREFILL_WMMA",
            "DS4_ROCM_DISABLE_Q4_PREFILL_WMMA"};
        saved.reserve(sizeof(keys) / sizeof(keys[0]));
        for (const char *key : keys) { saved.emplace_back(key); env(key, nullptr); }
        env("DS4_ROCM_DISABLE_Q4_PREFILL_WMMA", "1");
    }
};
void arm(bool candidate) {
    // A forcibly rolls back even with ENABLE set, so default promotion
    // cannot silently turn the experiment into candidate vs candidate.
    env(enable, "1");
    env(disable, candidate ? nullptr : "1");
    env(require, candidate ? "1" : nullptr);
}
struct tensor {
    ds4_gpu_tensor *p;
    explicit tensor(uint64_t bytes) : p(ds4_gpu_tensor_alloc(bytes)) {
        check(p != nullptr, "tensor allocation");
    }
    tensor(const tensor &parent, uint64_t offset, uint64_t bytes)
        : p(ds4_gpu_tensor_view(parent.p, offset, bytes)) {
        check(p != nullptr, "tensor view");
    }
    ~tensor() { ds4_gpu_tensor_free(p); }
    tensor(const tensor &) = delete;
    tensor &operator=(const tensor &) = delete;
};
void write(const tensor &t, const std::vector<float> &v) {
    check(ds4_gpu_tensor_write(t.p, 0u, v.data(), v.size() * sizeof(float)) != 0,
          "tensor write");
}
std::vector<float> read(const tensor &t, size_t count) {
    std::vector<float> v(count);
    check(ds4_gpu_tensor_read(t.p, 0u, v.data(), count * sizeof(float)) != 0,
          "tensor read");
    return v;
}
void equal(const std::vector<float> &a, const std::vector<float> &b, const char *label) {
    check(a.size() == b.size(), "comparison size");
    for (size_t i = 0u; i < a.size(); ++i) {
        if (std::memcmp(&a[i], &b[i], sizeof(float)) == 0) continue;
        uint32_t ab, bb;
        std::memcpy(&ab, &a[i], sizeof(ab)); std::memcpy(&bb, &b[i], sizeof(bb));
        std::fprintf(stderr, "%s: first mismatch at %zu: %08x vs %08x\n", label, i, ab, bb);
        check(false, label);
    }
}
uint32_t random_bits(uint32_t &state) {
    state = state * 1664525u + 1013904223u; return state;
}
struct q4_block {
    uint16_t d, dmin;
    uint8_t scales[12], qs[128];
};
static_assert(sizeof(q4_block) == 144u, "raw GGUF Q4_K ABI");
struct fixture {
    uint8_t *data = nullptr;
    uint64_t size = 0u;
    uint32_t sets;
    // A guard page before each matrix also exercises nonzero weight offsets.
    const uint64_t stride = 4096u + (uint64_t)M * 4u * sizeof(q4_block);
    fixture(uint32_t count, bool exceptional) : sets(count) {
        size = stride * (count + (exceptional ? 1u : 0u));
        void *p = nullptr;
        check(posix_memalign(&p, 4096u, (size_t)size) == 0, "model allocation");
        data = static_cast<uint8_t *>(p);
        std::memset(data, 0xa5, (size_t)size);
        for (uint32_t s = 0u; s < count + (exceptional ? 1u : 0u); ++s) {
            uint32_t rng = 0x12345678u ^ (s * 0x9e3779b9u);
            auto *w = reinterpret_cast<q4_block *>(data + offset(s));
            for (uint32_t r = 0u; r < M; ++r)
            for (uint32_t b = 0u; b < 4u; ++b) {
                q4_block &q = w[r * 4u + b];
                q.d = (uint16_t)(0x2000u + (random_bits(rng) & 0x3ffu));
                q.dmin = (uint16_t)(0x1400u + (random_bits(rng) & 0x3ffu));
                for (auto &v : q.scales) v = (uint8_t)(random_bits(rng) >> 24u);
                for (auto &v : q.qs) v = (uint8_t)(random_bits(rng) >> 24u);
            }
            if (exceptional && s == count) {
                // Non-finites enter via weights, not undefined host casts of
                // NaN/Inf activations in the unchanged Q8_K quantizer.
                const uint16_t special[] = {0u, 0x8000u, 1u, 0x8001u,
                    0x7c00u, 0xfc00u, 0x7e11u, 0xfe22u, 0x7c01u};
                for (uint32_t r = 0u; r < M; ++r) {
                    if ((r % 64u) >= 9u && r != M - 1u) continue;
                    for (uint32_t b = 0u; b < 4u; ++b) {
                        w[r * 4u + b].d = special[(r + b) % 9u];
                        w[r * 4u + b].dmin = special[(r + 2u * b) % 9u];
                    }
                }
            }
        }
    }
    uint64_t offset(uint32_t s) const { return s * stride + 4096u; }
    ~fixture() { std::free(data); }
    fixture(const fixture &) = delete;
    fixture &operator=(const fixture &) = delete;
};
struct io {
    uint32_t n, k, m;
    std::vector<float> input, sentinel;
    tensor x_storage, out_storage, x, out;
    io(uint32_t nt, uint32_t in = K, uint32_t rows = M, bool zeros = false)
        : n(nt), k(in), m(rows), input(2u * guard + (size_t)nt * in),
          sentinel(2u * guard + (size_t)nt * rows),
          x_storage(input.size() * sizeof(float)),
          out_storage(sentinel.size() * sizeof(float)),
          x(x_storage, guard * sizeof(float), (uint64_t)nt * in * sizeof(float)),
          out(out_storage, guard * sizeof(float), (uint64_t)nt * rows * sizeof(float)) {
        std::fill(input.begin(), input.end(), 123456.0f);
        for (size_t i = guard; i < input.size() - guard; ++i) {
            const size_t j = i - guard;
            float v = zeros ? ((j & 1u) ? -0.0f : 0.0f)
                : (float)((int)((j * 73u + j / 256u * 19u) % 241u) - 120) / 32.0f;
            if (!zeros && j % 256u == 0u) v = (j / 256u & 1u) ? 4.0f : -4.0f;
            input[i] = v;
        }
        for (size_t i = 0; i < sentinel.size(); ++i)
            sentinel[i] = 100000.0f + (float)(i % 8192u);
        write(x_storage, input);
    }
    void prepare() { write(out_storage, sentinel); }
    int invoke(const fixture &f, uint32_t s, uint64_t bytes = 0u, uint64_t off = UINT64_MAX) {
        return ds4_gpu_matmul_quant_tensor(out.p, f.data, bytes ? bytes : f.size,
            off == UINT64_MAX ? f.offset(s) : off, type_q4, k, m, x.p, n);
    }
    std::vector<float> verify(const std::vector<float> *expected, bool finite = true) {
        auto v = read(out_storage, sentinel.size());
        for (size_t i = 0u; i < v.size(); ++i) {
            const bool is_guard = i < guard || i >= v.size() - guard;
            const bool untouched = std::memcmp(&v[i], &sentinel[i], sizeof(float)) == 0;
            check(is_guard ? untouched : !untouched, "output guard or unwritten body");
            if (!is_guard && finite) check(std::isfinite(v[i]), "unexpected non-finite output");
        }
        if (expected) equal(v, *expected, "bitwise output including prefix/suffix guards");
        equal(read(x_storage, input.size()), input, "input and guards modified");
        return v;
    }
};

std::vector<float> checked_call(io &v, const fixture &f, uint32_t s,
                                uint64_t calls, const std::vector<float> *reference,
                                bool finite = true) {
    v.prepare();
    ds4_rocm_test_q4_decode_lane4_reset();
    check(v.invoke(f, s) != 0, "public API rejected valid call");
    check(ds4_rocm_test_q4_decode_lane4_get_calls() == calls, "wrong kernel launch count");
    return v.verify(reference, finite);
}
void rejected_call(io &v, const fixture &f, uint64_t bytes = 0u,
                   uint64_t offset = UINT64_MAX) {
    v.prepare();
    ds4_rocm_test_q4_decode_lane4_reset();
    check(v.invoke(f, 0u, bytes, offset) == 0, "REQUIRE/validation failed open");
    check(ds4_rocm_test_q4_decode_lane4_get_calls() == 0u, "rejected call launched candidate");
    equal(read(v.out_storage, v.sentinel.size()), v.sentinel, "rejected output modified");
    equal(read(v.x_storage, v.input.size()), v.input, "rejected input modified");
}
void oracle(const fixture &f) {
    for (uint32_t n = 1u; n <= 8u; ++n)
    for (bool zeros : {false, true}) {
        io v(n, K, M, zeros);
        arm(false);
        const auto a = checked_call(v, f, 0u, 0u, nullptr);
        arm(true);
        checked_call(v, f, 0u, 1u, &a);
        std::printf("PASS N=%u %s: rollback/candidate bitwise, guards, selected kernel\n",
                    n, zeros ? "signed-zero input" : "finite input");
    }
    for (uint32_t n : {1u, 8u}) {
        io v(n);
        arm(false);
        const auto a = checked_call(v, f, f.sets, 0u, nullptr, false);
        arm(true);
        checked_call(v, f, f.sets, 1u, &a, false);
        std::printf("PASS N=%u exceptional weights: bitwise NaN/Inf/zero/subnormal, guards\n", n);
    }
    io v(1u);
    arm(false);
    const auto a = checked_call(v, f, 0u, 0u, nullptr);
    env(enable, nullptr); env(disable, nullptr); env(require, nullptr);
    checked_call(v, f, 0u, 0u, &a); // actual default, not only forced rollback
    env(require, "1");
    checked_call(v, f, 0u, 1u, &a); // REQUIRE alone opts in
    for (const char *value : {"1", "0", ""}) {
        env(enable, value); env(disable, nullptr); env(require, nullptr);
        checked_call(v, f, 0u, 1u, &a); // all controls use presence semantics
        env(disable, value);
        checked_call(v, f, 0u, 0u, &a);
        env(require, value);
        rejected_call(v, f); // DISABLE beats ENABLE+REQUIRE, no fallback write
    }
    arm(true);
    rejected_call(v, f, f.offset(0u) + (uint64_t)M * 4u * sizeof(q4_block) - 1u);
    rejected_call(v, f, 0u, f.size);
    // Eligible geometry but one float short: existing validation must reject
    // before quantization, and neither the output nor its guards may change.
    tensor short_x(v.x_storage, guard * sizeof(float), (K - 1u) * sizeof(float));
    tensor short_out(v.out_storage, guard * sizeof(float), (M - 1u) * sizeof(float));
    for (bool short_input : {false, true}) {
        v.prepare();
        ds4_rocm_test_q4_decode_lane4_reset();
        const int rc = ds4_gpu_matmul_quant_tensor(
            short_input ? v.out.p : short_out.p, f.data, f.size, f.offset(0u),
            type_q4, K, M, short_input ? short_x.p : v.x.p, 1u);
        check(rc == 0 && ds4_rocm_test_q4_decode_lane4_get_calls() == 0u,
              "short tensor failed open");
        equal(read(v.out_storage, v.sentinel.size()), v.sentinel, "short-tensor output modified");
        equal(read(v.x_storage, v.input.size()), v.input, "short-tensor input modified");
    }
    ds4_gpu_set_quality(true);
    env(require, nullptr);
    checked_call(v, f, 0u, 0u, &a); // opt-in cannot bypass quality rollback
    env(require, "1"); rejected_call(v, f);
    ds4_gpu_set_quality(false);
    // Valid but excluded shapes must roll back with ENABLE, fail with REQUIRE.
    for (uint32_t scenario = 0u; scenario < 3u; ++scenario) {
        io other(scenario == 0u ? 9u : 1u, scenario == 1u ? 768u : K,
                 scenario == 2u ? M - 1u : M);
        arm(false);
        const auto baseline = checked_call(other, f, 0u, 0u, nullptr);
        env(disable, nullptr); env(require, nullptr);
        checked_call(other, f, 0u, 0u, &baseline);
        env(require, "1"); rejected_call(other, f);
    }
    std::puts("PASS default, controls, quality, model/tensor ranges and N/K/M exclusions.");
}

struct events {
#if DS4_LANE4_HAS_HIP
    hipEvent_t start = nullptr, stop = nullptr;
    events() {
        check(hipEventCreate(&start) == hipSuccess, "create start event");
        if (hipEventCreate(&stop) != hipSuccess) {
            (void)hipEventDestroy(start); check(false, "create stop event");
        }
    }
    ~events() { (void)hipEventDestroy(stop); (void)hipEventDestroy(start); }
    void begin() { check(hipEventRecord(start, nullptr) == hipSuccess, "record start"); }
    double end() {
        check(hipEventRecord(stop, nullptr) == hipSuccess, "record stop");
        check(hipEventSynchronize(stop) == hipSuccess, "synchronize stop");
        float ms = 0.0f;
        check(hipEventElapsedTime(&ms, start, stop) == hipSuccess, "event elapsed time");
        check(std::isfinite(ms) && ms > 0.0f, "invalid HIP event interval");
        return ms;
    }
#else
    void begin() { check(false, "HIP events unavailable"); }
    double end() { check(false, "HIP events unavailable"); return 0.0; }
#endif
};
struct options {
    bool bench = false;
    uint32_t samples = 8u, warmup = 4u, sets = 4u, iterations = 1u;
    std::vector<uint32_t> tokens = {1u, 2u, 4u, 8u};
};
uint32_t number(const char *text, uint32_t max) {
    check(text && text[0] >= '0' && text[0] <= '9', "expected positive integer");
    char *end = nullptr;
    const unsigned long n = std::strtoul(text, &end, 10);
    check(*end == '\0' && n >= 1u && n <= max, "integer out of range");
    return (uint32_t)n;
}
options parse(int argc, char **argv) {
    options o;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--bench") { o.bench = true; continue; }
        check(i + 1 < argc, "missing option argument");
        const char *value = argv[++i];
        if (key == "--samples") o.samples = number(value, 10000u);
        else if (key == "--warmup") o.warmup = number(value, 1000u);
        else if (key == "--sets") o.sets = number(value, 32u);
        else if (key == "--iterations") o.iterations = number(value, 10000u);
        else if (key == "--tokens") {
            o.tokens.clear();
            const std::string list(value);
            size_t pos = 0u;
            do {
                const size_t comma = list.find(',', pos);
                const auto token = list.substr(pos, comma == std::string::npos ? comma : comma - pos);
                const uint32_t n = number(token.c_str(), 8u);
                check(std::find(o.tokens.begin(), o.tokens.end(), n) == o.tokens.end(), "duplicate N");
                o.tokens.push_back(n);
                if (comma == std::string::npos) break;
                pos = comma + 1u;
            } while (true);
        } else check(false, "unknown option");
    }
    if (!o.bench) check(argc == 1, "benchmark parameters need --bench");
    if (o.bench) check(o.samples % (2u * o.sets) == 0u,
                       "samples must be a multiple of 2*sets (both orders per weight set)");
    return o;
}
double mean(const std::vector<double> &v) {
    double sum = 0.0; for (double x : v) sum += x; return sum / v.size();
}
double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return (v[(v.size() - 1u) / 2u] + v[v.size() / 2u]) * 0.5;
}
void benchmark(const fixture &f, const options &o) {
    std::printf("HIP events: public API quantizer+matmul, not matmul-only or end-to-end t/s.\n"
                "K=%u M=%u sets=%u samples=%u warmup=%u iterations=%u; "
                "readback/guards/control/counter queries outside timing.\n",
                K, M, o.sets, o.samples, o.warmup, o.iterations);
    std::puts("sample,N,set,order,arm,slot,ms_per_call,selected_launches");
    events timer;
    for (uint32_t n : o.tokens) {
        io v(n);
        std::vector<std::vector<float>> reference;
        for (uint32_t s = 0u; s < o.sets; ++s) {
            arm(false);
            reference.push_back(checked_call(v, f, s, 0u, nullptr));
            arm(true);
            checked_call(v, f, s, 1u, &reference.back());
        }
        // Warm every weight set on both paths; no domain suite or exceptional
        // matrices in --bench. Every measured arm is re-poisoned independently.
        for (uint32_t w = 0u; w < o.warmup; ++w)
        for (uint32_t s = 0u; s < o.sets; ++s)
        for (uint32_t j = 0u; j < 2u; ++j) {
            const bool b = ((w + j) & 1u) != 0u;
            arm(b); checked_call(v, f, s, b ? 1u : 0u, &reference[s]);
        }
        std::vector<double> a_ms, b_ms, ratios;
        uint32_t b_wins = 0u;
        for (uint32_t sample = 0u; sample < o.samples; ++sample) {
            const uint32_t s = sample % o.sets;
            const char *order = ((sample / o.sets + s) & 1u) ? "BAAB" : "ABBA";
            double a_sum = 0.0, b_sum = 0.0;
            for (uint32_t j = 0u; j < 4u; ++j) {
                const bool b = order[j] == 'B';
                arm(b); v.prepare();
                ds4_rocm_test_q4_decode_lane4_reset();
                check(ds4_rocm_test_q4_decode_lane4_get_calls() == 0u, "counter reset");
                timer.begin();
                int ok = 1;
                for (uint32_t it = 0u; it < o.iterations; ++it) ok &= v.invoke(f, s);
                const double ms = timer.end() / o.iterations;
                const uint64_t calls = ds4_rocm_test_q4_decode_lane4_get_calls();
                check(ok != 0 && calls == (b ? o.iterations : 0u), "timed dispatch/counter");
                v.verify(&reference[s]);
                (b ? b_ms : a_ms).push_back(ms);
                (b ? b_sum : a_sum) += ms;
                std::printf("%u,%u,%u,%s,%c,%u,%.9f,%llu\n", sample, n, s,
                            order, order[j], j, ms, (unsigned long long)calls);
            }
            ratios.push_back(a_sum / b_sum); // speed ratio: >1 favors B
            b_wins += b_sum < a_sum;
        }
        double log_sum = 0.0;
        for (double r : ratios) log_sum += std::log(r);
        const auto bounds = std::minmax_element(ratios.begin(), ratios.end());
        std::printf("SUMMARY N=%u A_mean_ms=%.9f B_mean_ms=%.9f "
                    "A_median_ms=%.9f B_median_ms=%.9f "
                    "paired_geomean_A_over_B=%.6f pair_min=%.6f pair_max=%.6f "
                    "B_wins=%u/%u\n", n, mean(a_ms), mean(b_ms), median(a_ms),
                    median(b_ms), std::exp(log_sum / ratios.size()),
                    *bounds.first, *bounds.second, b_wins, o.samples);
    }
    std::puts("PASS benchmark parity/guards/dispatch. Timings are measurements, not proof of model speedup.");
}
int device() {
#if DS4_LANE4_HAS_HIP
    int count = 0, current = -1;
    hipDeviceProp_t prop{};
    if (hipGetDeviceCount(&count) != hipSuccess || count == 0 ||
        hipGetDevice(&current) != hipSuccess ||
        hipGetDeviceProperties(&prop, current) != hipSuccess) return 0;
    std::printf("device=%d name=%s arch=%s wave=%d\n", current, prop.name,
                prop.gcnArchName, prop.warpSize);
    check(prop.warpSize == 32 || prop.warpSize == 64, "unsupported wave size");
    return 1;
#else
    return 0;
#endif
}
struct backend {
    backend() { check(ds4_gpu_init() != 0, "GPU init"); }
    ~backend() { ds4_gpu_cleanup(); }
};
} // namespace

int main(int argc, char **argv) {
    if (argc == 2 && std::strcmp(argv[1], "--help") == 0) {
        std::printf("usage: %s [--bench [--tokens 1,2,4,8] [--samples 8] "
                    "[--warmup 4] [--sets 4] [--iterations 1]]\n", argv[0]);
        return 0;
    }
    try {
        const options o = parse(argc, argv);
        if (!device()) {
            const char *r = std::getenv("DS4_TEST_REQUIRE_ROCM_DEVICE");
            const bool required = r && r[0] != '\0' && std::strcmp(r, "0") != 0;
            std::fprintf(stderr, "%s: Q4 decode lane4 (HIP/device unavailable)\n",
                         required ? "FAIL" : "SKIP");
            return required ? 1 : 77;
        }
        controls settings;
        fixture f(o.bench ? o.sets : 1u, !o.bench);
        backend gpu; // destroyed before the registered model's backing memory
        ds4_gpu_set_quality(false);
        check(ds4_gpu_set_model_map(f.data, f.size) != 0, "model registration");
        if (o.bench) benchmark(f, o); else oracle(f);
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "FAIL Q4 decode lane4: %s\n", e.what());
        return 1;
    }
}
