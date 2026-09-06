// SPDX-License-Identifier: MIT
// Resident, GPU-event-only ROCm Q4_K prefill microbenchmark.

#include "ds4_gpu.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <string>
#include <sys/mman.h>
#include <utility>
#include <vector>

/* Deliberately local to this harness: the backend entry performs only the
 * selected kernel enqueue after this process has completed all validation. */
extern "C" void ds4_rocm_bench_q8_K_quantize_enqueue(
    void *out, const void *x, uint32_t in_dim, uint32_t n_rows,
    int use_wave32);
extern "C" const void *ds4_rocm_bench_q4_K_resident_weight_ptr(
    const void *model_map, uint64_t model_size, uint64_t weight_offset,
    uint64_t weight_bytes);
extern "C" int ds4_rocm_bench_q4_K_wmma_enqueue(
    void *out, const void *w, const void *x, uint32_t n_tok,
    uint32_t n_groups, uint32_t in_dim, uint32_t out_dim,
    uint64_t row_bytes, uint64_t x_token_stride,
    uint64_t x_group_stride, uint64_t out_token_stride,
    uint32_t row_tile, int load2);
extern "C" int ds4_rocm_bench_q4_K_wmma_variant_enqueue(
    void *out, const void *w, const void *x, uint32_t n_tok,
    uint32_t n_groups, uint32_t in_dim, uint32_t out_dim,
    uint64_t row_bytes, uint64_t x_token_stride,
    uint64_t x_group_stride, uint64_t out_token_stride,
    uint32_t row_tile, uint32_t k_tile, int load2);
extern "C" int ds4_rocm_bench_q4_K_wmma_k128_enqueue(
    void *out, const void *w, const void *x, uint32_t n_tok,
    uint32_t n_groups, uint32_t in_dim, uint32_t out_dim,
    uint64_t row_bytes, uint64_t x_token_stride,
    uint64_t x_group_stride, uint64_t out_token_stride);
extern "C" int ds4_rocm_bench_q4_K_wmma_k64_load4_enqueue(
    void *out, const void *w, const void *x, uint32_t n_tok,
    uint32_t n_groups, uint32_t in_dim, uint32_t out_dim,
    uint64_t row_bytes, uint64_t x_token_stride,
    uint64_t x_group_stride, uint64_t out_token_stride, uint32_t row_tile);
extern "C" void ds4_rocm_test_q4_prefill_wmma_reset(void);
extern "C" uint64_t ds4_rocm_test_q4_prefill_wmma_get_calls(void);
extern "C" uint64_t ds4_rocm_test_q4_prefill_wmma_k64_get_calls(void);
extern "C" void ds4_rocm_test_q4_prefill_lds_stream_reset(void);
extern "C" uint64_t ds4_rocm_test_q4_prefill_lds_stream_get_calls(void);
extern "C" void ds4_rocm_test_q4_prefill_lds_vector_reset(void);
extern "C" uint64_t ds4_rocm_test_q4_prefill_lds_vector_get_calls(void);

namespace {

constexpr uint32_t kQ4Type = 12u;
constexpr uint32_t kQkK = 256u;
constexpr uint32_t kDenseK = 4096u;
constexpr uint32_t kDenseM = 1024u;
constexpr uint32_t kKvM = 512u;
constexpr uint32_t kQbK = 1024u;
constexpr uint32_t kQbM = 32768u;
constexpr uint32_t kOutputGroups = 8u;
constexpr uint32_t kOutputRank = 1024u;
constexpr uint32_t kOutputLowDim = kOutputGroups * kOutputRank;
constexpr uint32_t kOutputM = 4096u;
constexpr uint32_t kDefaultSets = 4u;
constexpr uint32_t kDefaultSamples = 8u;
constexpr uint32_t kDefaultWarmup = 2u;
constexpr uint32_t kRawQ8GuardWords = 64u;
/* Catch a bad N-tail predicate anywhere in the final 64-token WMMA tile,
 * including the widest q_b output used by this harness. */
constexpr uint32_t kGuardWords = (64u - 1u) * kQbM;
constexpr uint64_t kCompareChunk = 4u * 1024u * 1024u;

constexpr const char *kPrefillEnable =
    "DS4_ROCM_ENABLE_Q4_PREFILL_TILE8";
constexpr const char *kPrefillDisable =
    "DS4_ROCM_DISABLE_Q4_PREFILL_TILE8";
constexpr const char *kPrefillRequire =
    "DS4_ROCM_REQUIRE_Q4_PREFILL_TILE8";
constexpr const char *kLdsDisable =
    "DS4_ROCM_DISABLE_Q4_PREFILL_LDS_STREAM";
constexpr const char *kLdsVectorDisable =
    "DS4_ROCM_DISABLE_Q4_PREFILL_LDS_VECTOR";
constexpr const char *kK1024Tile4Disable =
    "DS4_ROCM_DISABLE_Q4_PREFILL_K1024_TILE4";
constexpr const char *kK1024Tile4SsdEnable =
    "DS4_ROCM_ENABLE_Q4_PREFILL_K1024_TILE4_SSD";
constexpr const char *kK1024Tile4Require =
    "DS4_ROCM_REQUIRE_Q4_PREFILL_K1024_TILE4";
constexpr const char *kWmmaEnable =
    "DS4_ROCM_ENABLE_Q4_PREFILL_WMMA";
constexpr const char *kWmmaSsdEnable =
    "DS4_ROCM_ENABLE_Q4_PREFILL_WMMA_SSD";
constexpr const char *kWmmaDisable =
    "DS4_ROCM_DISABLE_Q4_PREFILL_WMMA";
constexpr const char *kWmmaRequire =
    "DS4_ROCM_REQUIRE_Q4_PREFILL_WMMA";
constexpr const char *kWmmaRowTile =
    "DS4_ROCM_Q4_PREFILL_WMMA_ROW_TILE";
constexpr const char *kWmmaK64 =
    "DS4_ROCM_ENABLE_Q4_PREFILL_WMMA_K64";
constexpr const char *kWmmaK128Disable =
    "DS4_ROCM_DISABLE_Q4_PREFILL_WMMA_K128";
constexpr const char *kQ8Wave32Enable =
    "DS4_ROCM_ENABLE_Q4_PREFILL_Q8_K_WAVE32";
constexpr const char *kQ8Wave32Disable =
    "DS4_ROCM_DISABLE_Q4_PREFILL_Q8_K_WAVE32";
constexpr const char *kQ8Wave32Require =
    "DS4_ROCM_REQUIRE_Q4_PREFILL_Q8_K_WAVE32";

struct block_q4_K_host {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[12];
    uint8_t qs[kQkK / 2u];
};

static_assert(sizeof(block_q4_K_host) == 144u,
              "Q4_K fixture must match the raw GGUF layout");

struct block_q8_K_host {
    float d;
    int8_t qs[kQkK];
    int16_t bsums[kQkK / 16u];
};

static_assert(sizeof(block_q8_K_host) == 292u,
              "Q8_K fixture must match the ROCm activation layout");

enum class bench_case {
    all,
    lds,
    lds_vector,
    wmma_load4,
    dense,
    pair,
    qb,
    outb,
    output,
};

struct config {
    bench_case selected = bench_case::all;
    std::vector<uint32_t> tokens = {
        9u, 17u, 33u, 128u, 256u, 257u, 512u};
    uint32_t sets = kDefaultSets;
    uint32_t samples = kDefaultSamples;
    uint32_t warmup = kDefaultWarmup;
    bool wmma_supported = false;
};

struct weight_set {
    uint64_t dense_offset = 0;
    uint64_t kv_offset = 0;
    uint64_t qb_offset = 0;
    uint64_t output_a_offset = 0;
    uint64_t output_b_offset = 0;
};

struct model_fixture {
    uint8_t *data = nullptr;
    uint64_t size = 0;
    uint64_t resident_bytes = 0;
    FILE *file = nullptr;
    std::vector<weight_set> weights;
    std::vector<uint64_t> span_offsets;
    std::vector<uint64_t> span_sizes;

    ~model_fixture() {
        if (data && size != 0u) (void)munmap(data, static_cast<size_t>(size));
        if (file) std::fclose(file);
    }
    model_fixture() = default;
    model_fixture(const model_fixture &) = delete;
    model_fixture &operator=(const model_fixture &) = delete;
};

struct tensor_owner {
    ds4_gpu_tensor *ptr = nullptr;

    explicit tensor_owner(uint64_t bytes) : ptr(ds4_gpu_tensor_alloc(bytes)) {}
    ~tensor_owner() { ds4_gpu_tensor_free(ptr); }
    tensor_owner(const tensor_owner &) = delete;
    tensor_owner &operator=(const tensor_owner &) = delete;
};

struct env_snapshot {
    const char *name;
    bool existed;
    std::string value;

    explicit env_snapshot(const char *key)
        : name(key), existed(std::getenv(key) != nullptr),
          value(existed ? std::getenv(key) : "") {}
    ~env_snapshot() {
        if (existed) {
            (void)setenv(name, value.c_str(), 1);
        } else {
            (void)unsetenv(name);
        }
    }
    env_snapshot(const env_snapshot &) = delete;
    env_snapshot &operator=(const env_snapshot &) = delete;
};

struct event_timer {
    hipEvent_t begin = nullptr;
    hipEvent_t end = nullptr;

    event_timer() {
        if (hipEventCreate(&begin) != hipSuccess ||
            hipEventCreate(&end) != hipSuccess) {
            std::fprintf(stderr,
                         "rocm-q4-prefill-bench: HIP event allocation failed\n");
            std::exit(1);
        }
    }
    ~event_timer() {
        if (begin) (void)hipEventDestroy(begin);
        if (end) (void)hipEventDestroy(end);
    }

    bool measure(const std::function<bool()> &dispatch, float *milliseconds) {
        if (hipEventRecord(begin, 0) != hipSuccess) return false;
        if (!dispatch()) return false;
        if (hipEventRecord(end, 0) != hipSuccess ||
            hipEventSynchronize(end) != hipSuccess ||
            hipEventElapsedTime(milliseconds, begin, end) != hipSuccess) {
            return false;
        }
        /* Outside the measured interval: surface an immediate launch/config
         * error from enqueue-only benchmark hooks instead of false-greening. */
        return hipGetLastError() == hipSuccess;
    }
};

struct arm {
    const char *name;
    std::function<void()> prepare;
    std::function<bool(uint32_t)> dispatch;
};

struct stats {
    double minimum = 0.0;
    double median = 0.0;
    double p95 = 0.0;
    double mean = 0.0;
};

enum class benchmark_rate {
    macs,
    quantized_values,
};

uint64_t align_up(uint64_t value, uint64_t alignment) {
    return (value + alignment - 1u) / alignment * alignment;
}

bool checked_mul(uint64_t a, uint64_t b, uint64_t *out) {
    if (a != 0u && b > std::numeric_limits<uint64_t>::max() / a) return false;
    *out = a * b;
    return true;
}

uint32_t lcg_next(uint32_t *state) {
    *state = *state * 1664525u + 1013904223u;
    return *state;
}

void fill_q4(void *storage, uint64_t bytes, uint32_t seed) {
    auto *blocks = static_cast<block_q4_K_host *>(storage);
    const uint64_t count = bytes / sizeof(*blocks);
    uint32_t state = seed;
    for (uint64_t i = 0; i < count; i++) {
        // Positive, finite FP16 scales. The payload is deterministic but does
        // not need a CPU oracle: the benchmark compares production GPU paths.
        blocks[i].d = static_cast<uint16_t>(0x2400u + (lcg_next(&state) & 0xffu));
        blocks[i].dmin =
            static_cast<uint16_t>(0x2000u + (lcg_next(&state) & 0xffu));
        for (uint8_t &v : blocks[i].scales) {
            v = static_cast<uint8_t>(lcg_next(&state) >> 24u);
        }
        for (uint8_t &v : blocks[i].qs) {
            v = static_cast<uint8_t>(lcg_next(&state) >> 24u);
        }
    }
}

uint64_t q4_weight_bytes(uint32_t in_dim, uint32_t out_dim) {
    return static_cast<uint64_t>(out_dim) * (in_dim / kQkK) *
           sizeof(block_q4_K_host);
}

uint32_t shape_wmma_row_tile(uint32_t out_dim) {
    return out_dim >= 8192u ? 256u : (out_dim >= 1024u ? 128u : 64u);
}

bool resolve_resident_weights(
        const model_fixture &model,
        uint64_t weight_set::*offset_member,
        uint64_t weight_bytes,
        std::vector<const void *> *resolved) {
    resolved->resize(model.weights.size());
    for (size_t i = 0; i < model.weights.size(); i++) {
        const uint64_t offset = model.weights[i].*offset_member;
        (*resolved)[i] = ds4_rocm_bench_q4_K_resident_weight_ptr(
            model.data, model.size, offset, weight_bytes);
        if (!(*resolved)[i]) {
            std::fprintf(stderr,
                         "rocm-q4-prefill-bench: weight set %zu is not "
                         "physically device-resident\n",
                         i);
            return false;
        }
    }
    return true;
}

bool make_model(model_fixture *model, uint32_t sets) {
    constexpr uint64_t page = 4096u;
    const uint64_t dense_bytes = q4_weight_bytes(kDenseK, kDenseM);
    const uint64_t kv_bytes = q4_weight_bytes(kDenseK, kKvM);
    const uint64_t qb_bytes = q4_weight_bytes(kQbK, kQbM);
    const uint64_t output_a_bytes =
        q4_weight_bytes(kDenseK, kOutputLowDim);
    const uint64_t output_b_bytes =
        q4_weight_bytes(kOutputLowDim, kOutputM);

    model->weights.resize(sets);
    model->span_offsets.reserve(static_cast<size_t>(sets) * 5u);
    model->span_sizes.reserve(static_cast<size_t>(sets) * 5u);
    uint64_t cursor = 0;
    auto append = [&](uint64_t bytes) {
        const uint64_t offset = align_up(cursor, page);
        cursor = offset + bytes;
        model->span_offsets.push_back(offset);
        model->span_sizes.push_back(bytes);
        model->resident_bytes += bytes;
        return offset;
    };
    for (uint32_t i = 0; i < sets; i++) {
        model->weights[i].dense_offset = append(dense_bytes);
        model->weights[i].kv_offset = append(kv_bytes);
        model->weights[i].qb_offset = append(qb_bytes);
        model->weights[i].output_a_offset = append(output_a_bytes);
        model->weights[i].output_b_offset = append(output_b_bytes);
    }
    model->size = align_up(cursor, page);
    void *storage = nullptr;
    if (posix_memalign(&storage, static_cast<size_t>(page),
                       static_cast<size_t>(model->size)) != 0) {
        return false;
    }
    auto *staging = static_cast<uint8_t *>(storage);
    std::memset(staging, 0, static_cast<size_t>(model->size));
    for (uint32_t i = 0; i < sets; i++) {
        fill_q4(staging + model->weights[i].dense_offset, dense_bytes,
                0x243f6a88u ^ (i * 0x9e3779b9u));
        fill_q4(staging + model->weights[i].kv_offset, kv_bytes,
                0x85a308d3u ^ (i * 0x7f4a7c15u));
        fill_q4(staging + model->weights[i].qb_offset, qb_bytes,
                0x13198a2eu ^ (i * 0x94d049bbu));
        fill_q4(staging + model->weights[i].output_a_offset, output_a_bytes,
                0x03707344u ^ (i * 0x369dea0fu));
        fill_q4(staging + model->weights[i].output_b_offset, output_b_bytes,
                0xa4093822u ^ (i * 0xdb4f0b91u));
    }

    FILE *file = std::tmpfile();
    if (!file ||
        std::fwrite(staging, 1u, static_cast<size_t>(model->size), file) !=
            static_cast<size_t>(model->size) ||
        std::fflush(file) != 0) {
        std::free(staging);
        if (file) std::fclose(file);
        return false;
    }
    void *mapping = mmap(nullptr, static_cast<size_t>(model->size), PROT_READ,
                         MAP_PRIVATE, fileno(file), 0);
    std::free(staging);
    if (mapping == MAP_FAILED) {
        std::fclose(file);
        return false;
    }
    model->data = static_cast<uint8_t *>(mapping);
    model->file = file;
    return true;
}

void fill_activation(std::vector<float> *values, uint32_t n_tokens,
                     uint32_t in_dim) {
    values->resize(static_cast<uint64_t>(n_tokens) * in_dim);
    for (uint32_t token = 0; token < n_tokens; token++) {
        for (uint32_t block = 0; block < in_dim / kQkK; block++) {
            float *dst = values->data() +
                         static_cast<uint64_t>(token) * in_dim + block * kQkK;
            for (uint32_t i = 0; i < kQkK; i++) {
                const int q = static_cast<int>((i * 73u + token * 37u +
                                                block * 19u) % 241u) - 120;
                /* Exercise real F32->F16 rounding in the direct-WMMA
                 * scalar/load2 oracle; division by 32 made every fixture
                 * value exactly representable in F16. */
                dst[i] = static_cast<float>(q) / 37.0f;
            }
            dst[0] = ((token + block) & 1u) ? 127.0f / 32.0f
                                             : -127.0f / 32.0f;
        }
    }
}

std::vector<uint32_t> guard_pattern(uint32_t words = kGuardWords) {
    std::vector<uint32_t> guard(words);
    for (uint32_t i = 0; i < words; i++) guard[i] = 0x7fc12000u + i;
    return guard;
}

bool prepare_guard(ds4_gpu_tensor *tensor, uint64_t logical_bytes,
                   uint32_t guard_words = kGuardWords) {
    const std::vector<uint32_t> guard = guard_pattern(guard_words);
    return ds4_gpu_tensor_write(tensor, logical_bytes, guard.data(),
                                guard.size() * sizeof(guard[0])) != 0;
}

bool poison_output(ds4_gpu_tensor *tensor, uint64_t logical_bytes,
                   uint32_t pattern,
                   uint32_t guard_words = kGuardWords) {
    if (!tensor || logical_bytes == 0u ||
        logical_bytes % sizeof(uint32_t) != 0u) {
        return false;
    }
    const uint64_t chunk_bytes = std::min(kCompareChunk, logical_bytes);
    std::vector<uint32_t> poison(
        static_cast<size_t>(chunk_bytes / sizeof(uint32_t)), pattern);
    for (uint64_t offset = 0; offset < logical_bytes; offset += chunk_bytes) {
        const uint64_t count = std::min(chunk_bytes, logical_bytes - offset);
        if (!ds4_gpu_tensor_write(tensor, offset, poison.data(), count)) {
            return false;
        }
    }
    return prepare_guard(tensor, logical_bytes, guard_words);
}

bool check_guard(const ds4_gpu_tensor *tensor, uint64_t logical_bytes,
                 const char *label,
                 uint32_t guard_words = kGuardWords) {
    const std::vector<uint32_t> expected = guard_pattern(guard_words);
    std::vector<uint32_t> got(expected.size());
    if (!ds4_gpu_tensor_read(tensor, logical_bytes, got.data(),
                             got.size() * sizeof(got[0]))) {
        std::fprintf(stderr, "%s: guard read failed\n", label);
        return false;
    }
    if (got != expected) {
        const auto mismatch = std::mismatch(got.begin(), got.end(),
                                            expected.begin());
        std::fprintf(stderr, "%s: output guard overwritten at word %zu\n",
                     label,
                     static_cast<size_t>(mismatch.first - got.begin()));
        return false;
    }
    return true;
}

bool bitwise_equal(const ds4_gpu_tensor *a, const ds4_gpu_tensor *b,
                   uint64_t bytes, const char *label) {
    const uint64_t chunk = std::min(kCompareChunk, bytes);
    std::vector<uint8_t> lhs(static_cast<size_t>(chunk));
    std::vector<uint8_t> rhs(static_cast<size_t>(chunk));
    for (uint64_t offset = 0; offset < bytes; offset += chunk) {
        const uint64_t count = std::min(chunk, bytes - offset);
        if (!ds4_gpu_tensor_read(a, offset, lhs.data(), count) ||
            !ds4_gpu_tensor_read(b, offset, rhs.data(), count)) {
            std::fprintf(stderr, "%s: oracle read failed\n", label);
            return false;
        }
        if (std::memcmp(lhs.data(), rhs.data(), static_cast<size_t>(count)) != 0) {
            uint64_t first = 0;
            while (first < count && lhs[static_cast<size_t>(first)] ==
                                         rhs[static_cast<size_t>(first)]) {
                first++;
            }
            std::fprintf(stderr,
                         "%s: bitwise mismatch at output byte %llu\n", label,
                         static_cast<unsigned long long>(offset + first));
            return false;
        }
    }
    return true;
}

bool numerically_close(const ds4_gpu_tensor *got,
                       const ds4_gpu_tensor *reference,
                       uint64_t bytes,
                       const char *label,
                       float abs_tolerance = 2.0f,
                       float rel_tolerance = 3.0e-2f,
                       bool gate = true) {
    if ((bytes % sizeof(float)) != 0u) return false;
    const uint64_t chunk_bytes =
        std::min(kCompareChunk, bytes) & ~(uint64_t)(sizeof(float) - 1u);
    std::vector<float> lhs(static_cast<size_t>(chunk_bytes / sizeof(float)));
    std::vector<float> rhs(static_cast<size_t>(chunk_bytes / sizeof(float)));
    uint64_t failures = 0u;
    uint64_t nonfinite = 0u;
    uint64_t compared = 0u;
    float max_abs = 0.0f;
    float max_rel = 0.0f;
    uint64_t worst = 0u;
    for (uint64_t offset = 0u; offset < bytes; offset += chunk_bytes) {
        const uint64_t count = std::min(chunk_bytes, bytes - offset);
        if (!ds4_gpu_tensor_read(got, offset, lhs.data(), count) ||
            !ds4_gpu_tensor_read(reference, offset, rhs.data(), count)) {
            std::fprintf(stderr, "%s: oracle read failed\n", label);
            return false;
        }
        const uint64_t values = count / sizeof(float);
        for (uint64_t i = 0u; i < values; i++) {
            if (!std::isfinite(lhs[(size_t)i]) ||
                !std::isfinite(rhs[(size_t)i])) {
                failures++;
                nonfinite++;
                continue;
            }
            const float diff = std::fabs(lhs[(size_t)i] - rhs[(size_t)i]);
            const float rel = diff /
                std::max(1.0f, std::fabs(rhs[(size_t)i]));
            if (diff > max_abs) {
                max_abs = diff;
                worst = compared + i;
            }
            max_rel = std::max(max_rel, rel);
            if (diff > abs_tolerance +
                       rel_tolerance * std::fabs(rhs[(size_t)i])) {
                failures++;
            }
        }
        compared += values;
    }
    std::fprintf(stderr,
                 "%s: failures=%llu/%llu nonfinite=%llu max_abs=%g "
                 "max_rel=%g worst=%llu tolerance(abs=%g rel=%g) %s\n",
                 label, (unsigned long long)failures,
                 (unsigned long long)compared,
                 (unsigned long long)nonfinite, max_abs, max_rel,
                 (unsigned long long)worst, abs_tolerance, rel_tolerance,
                 failures == 0u ? "PASS" :
                 (gate || nonfinite != 0u ? "FAIL" : "DIAGNOSTIC"));
    return nonfinite == 0u && (failures == 0u || !gate);
}

void select_legacy() {
    (void)unsetenv(kPrefillEnable);
    (void)setenv(kPrefillDisable, "1", 1);
    (void)unsetenv(kPrefillRequire);
    (void)unsetenv(kK1024Tile4Disable);
    (void)unsetenv(kK1024Tile4SsdEnable);
    (void)unsetenv(kK1024Tile4Require);
    (void)unsetenv(kWmmaEnable);
    (void)unsetenv(kWmmaSsdEnable);
    (void)setenv(kWmmaDisable, "1", 1);
    (void)unsetenv(kWmmaRequire);
    (void)unsetenv(kWmmaRowTile);
    (void)unsetenv(kWmmaK64);
    (void)unsetenv(kWmmaK128Disable);
    (void)unsetenv(kQ8Wave32Enable);
    (void)unsetenv(kQ8Wave32Disable);
    (void)unsetenv(kQ8Wave32Require);
}

void select_tile8(bool disable_k1024_tile4) {
    (void)unsetenv(kPrefillEnable);
    (void)unsetenv(kPrefillDisable);
    (void)setenv(kPrefillRequire, "1", 1);
    (void)unsetenv(kK1024Tile4SsdEnable);
    (void)unsetenv(kK1024Tile4Require);
    (void)unsetenv(kWmmaEnable);
    (void)unsetenv(kWmmaSsdEnable);
    (void)setenv(kWmmaDisable, "1", 1);
    (void)unsetenv(kWmmaRequire);
    (void)unsetenv(kWmmaRowTile);
    (void)unsetenv(kWmmaK64);
    (void)unsetenv(kWmmaK128Disable);
    (void)unsetenv(kQ8Wave32Enable);
    (void)unsetenv(kQ8Wave32Disable);
    (void)unsetenv(kQ8Wave32Require);
    if (disable_k1024_tile4) {
        (void)setenv(kK1024Tile4Disable, "1", 1);
    } else {
        (void)unsetenv(kK1024Tile4Disable);
    }
}

void select_k1024_tile4() {
    select_tile8(false);
    (void)setenv(kK1024Tile4Require, "1", 1);
}

void select_wmma_shape() {
    select_tile8(false);
    /* WMMA and TILE8 are separate strict contracts.  The candidate must not
     * inherit REQUIRE_TILE8 from the baseline selector. */
    (void)unsetenv(kPrefillRequire);
    (void)setenv(kWmmaEnable, "1", 1);
    (void)unsetenv(kWmmaSsdEnable);
    (void)unsetenv(kWmmaDisable);
    (void)setenv(kWmmaRequire, "1", 1);
    (void)unsetenv(kK1024Tile4Require);
    (void)unsetenv(kWmmaRowTile);
    (void)unsetenv(kWmmaK64);
    (void)unsetenv(kWmmaK128Disable);
}

void select_wmma_attention_a_tile8_b() {
    select_tile8(false);
    /* ENABLE is the production A-only request.  Do not use REQUIRE here:
     * REQUIRE deliberately promotes the numerically compounded output-B
     * direct-WMMA path for diagnostics. */
    (void)unsetenv(kPrefillRequire);
    (void)setenv(kWmmaEnable, "1", 1);
    (void)unsetenv(kWmmaSsdEnable);
    (void)unsetenv(kWmmaDisable);
    (void)unsetenv(kWmmaRequire);
    (void)unsetenv(kK1024Tile4Require);
    (void)unsetenv(kWmmaRowTile);
    (void)unsetenv(kWmmaK64);
    (void)unsetenv(kWmmaK128Disable);
    ds4_rocm_test_q4_prefill_wmma_reset();
}

double percentile(std::vector<double> sorted, double fraction) {
    std::sort(sorted.begin(), sorted.end());
    if (sorted.empty()) return 0.0;
    const double position = fraction * static_cast<double>(sorted.size() - 1u);
    const size_t lo = static_cast<size_t>(std::floor(position));
    const size_t hi = static_cast<size_t>(std::ceil(position));
    const double alpha = position - static_cast<double>(lo);
    return sorted[lo] + (sorted[hi] - sorted[lo]) * alpha;
}

stats summarize(const std::vector<double> &samples) {
    stats out;
    out.minimum = *std::min_element(samples.begin(), samples.end());
    out.median = percentile(samples, 0.5);
    out.p95 = percentile(samples, 0.95);
    for (double sample : samples) out.mean += sample;
    out.mean /= static_cast<double>(samples.size());
    return out;
}

bool benchmark_arms(const char *case_name, uint32_t n_tokens, uint32_t in_dim,
                    uint32_t out_dim, const config &cfg, const arm &baseline,
                    const arm &candidate,
                    const std::function<bool()> &oracle_prepare,
                    const std::function<bool()> &oracle,
                    benchmark_rate rate = benchmark_rate::macs,
                    bool balance_weight_order = false) {
    // Validate every rotating weight set and prime reusable Q8_K scratch
    // before any timed event.  This catches data-dependent path errors without
    // admitting readback or comparison work into the HIP-event interval.
    for (uint32_t set = 0; set < cfg.sets; set++) {
        if (!oracle_prepare()) {
            std::fprintf(stderr,
                         "rocm-q4-prefill-bench: %s oracle poison failed "
                         "for weight set %u\n",
                         case_name, set);
            return false;
        }
        baseline.prepare();
        if (!baseline.dispatch(set) || !ds4_gpu_synchronize()) return false;
        candidate.prepare();
        if (!candidate.dispatch(set) || !ds4_gpu_synchronize()) return false;
        if (!oracle()) {
            std::fprintf(stderr,
                         "rocm-q4-prefill-bench: %s oracle failed for "
                         "weight set %u\n",
                         case_name, set);
            return false;
        }
    }

    for (uint32_t i = 0; i < cfg.warmup; i++) {
        const uint32_t set = i % cfg.sets;
        baseline.prepare();
        if (!baseline.dispatch(set) || !ds4_gpu_synchronize()) return false;
        candidate.prepare();
        if (!candidate.dispatch(set) || !ds4_gpu_synchronize()) return false;
    }

    event_timer timer;
    std::vector<double> a_samples;
    std::vector<double> b_samples;
    a_samples.reserve(cfg.samples);
    b_samples.reserve(cfg.samples);

    auto take = [&](const arm &which, uint32_t set,
                    std::vector<double> *samples) {
        which.prepare();
        float elapsed = 0.0f;
        const bool ok = timer.measure(
            [&]() { return which.dispatch(set); }, &elapsed);
        if (!ok) {
            std::fprintf(stderr,
                         "rocm-q4-prefill-bench: %s/%s timed dispatch failed\n",
                         case_name, which.name);
            return false;
        }
        samples->push_back(static_cast<double>(elapsed));
        return true;
    };

    // Each cycle contributes two samples per arm. Alternating ABBA/BAAB
    // balances first/last position, while both arms see identical weight sets.
    for (uint32_t cycle = 0; a_samples.size() < cfg.samples; cycle++) {
        const uint32_t set0 = (cycle * 2u) % cfg.sets;
        const uint32_t set1 = (cycle * 2u + 1u) % cfg.sets;
        // LOAD4: reverse each rotating set's order on the next traversal.
        // Simple cycle parity pins even/odd sets to AB/BA for four sets.
        const uint32_t period = cfg.sets % 2u ? cfg.sets : cfg.sets / 2u;
        const uint32_t order = balance_weight_order
            ? (cycle % period + cycle / period) : cycle;
        if ((order & 1u) == 0u) {
            if (!take(baseline, set0, &a_samples) ||
                !take(candidate, set0, &b_samples) ||
                !take(candidate, set1, &b_samples) ||
                !take(baseline, set1, &a_samples)) return false;
        } else {
            if (!take(candidate, set0, &b_samples) ||
                !take(baseline, set0, &a_samples) ||
                !take(baseline, set1, &a_samples) ||
                !take(candidate, set1, &b_samples)) return false;
        }
    }

    /* Re-run one set after the timed samples.  Event synchronization catches
     * asynchronous launch failures, while this final readback also catches a
     * geometry-dependent overwrite or wrong result that appears only after
     * repeated launches. */
    const uint32_t post_set = cfg.sets - 1u;
    if (!oracle_prepare()) return false;
    baseline.prepare();
    if (!baseline.dispatch(post_set) || !ds4_gpu_synchronize()) return false;
    candidate.prepare();
    if (!candidate.dispatch(post_set) || !ds4_gpu_synchronize()) return false;
    if (!oracle()) {
        std::fprintf(stderr,
                     "rocm-q4-prefill-bench: %s post-timing oracle failed\n",
                     case_name);
        return false;
    }

    const stats a = summarize(a_samples);
    const stats b = summarize(b_samples);
    std::vector<double> paired_delta;
    paired_delta.reserve(a_samples.size());
    for (size_t i = 0; i < a_samples.size(); i++) {
        paired_delta.push_back((b_samples[i] / a_samples[i] - 1.0) * 100.0);
    }
    const double paired_median = percentile(paired_delta, 0.5);
    const double median_delta = (b.median / a.median - 1.0) * 100.0;
    const double speedup = (a.median / b.median - 1.0) * 100.0;
    const double work = static_cast<double>(n_tokens) * in_dim *
        (rate == benchmark_rate::macs ? out_dim : 1u);
    const double a_rate = work / (a.median * 1.0e6);
    const double b_rate = work / (b.median * 1.0e6);

    if (rate == benchmark_rate::quantized_values) {
        std::printf(
            "DS4_ROCM_Q4_PREFILL_BENCH case=%s N=%u K=%u "
            "baseline=%s candidate=%s samples=%u sets=%u "
            "baseline_ms_p50=%.6f candidate_ms_p50=%.6f "
            "baseline_ms_min=%.6f candidate_ms_min=%.6f "
            "baseline_ms_p95=%.6f candidate_ms_p95=%.6f "
            "baseline_gvalue_s=%.3f candidate_gvalue_s=%.3f "
            "candidate_delta_pct=%.3f paired_delta_pct_p50=%.3f "
            "speedup_pct=%.3f\n",
            case_name, n_tokens, in_dim, baseline.name, candidate.name,
            cfg.samples, cfg.sets, a.median, b.median, a.minimum, b.minimum,
            a.p95, b.p95, a_rate, b_rate, median_delta, paired_median,
            speedup);
    } else {
        std::printf(
            "DS4_ROCM_Q4_PREFILL_BENCH case=%s N=%u K=%u M=%u "
            "baseline=%s candidate=%s samples=%u sets=%u "
            "baseline_ms_p50=%.6f candidate_ms_p50=%.6f "
            "baseline_ms_min=%.6f candidate_ms_min=%.6f "
            "baseline_ms_p95=%.6f candidate_ms_p95=%.6f "
            "baseline_gmac_s=%.3f candidate_gmac_s=%.3f "
            "candidate_delta_pct=%.3f paired_delta_pct_p50=%.3f "
            "speedup_pct=%.3f\n",
            case_name, n_tokens, in_dim, out_dim, baseline.name,
            candidate.name, cfg.samples, cfg.sets, a.median, b.median,
            a.minimum, b.minimum, a.p95, b.p95, a_rate, b_rate,
            median_delta, paired_median, speedup);
    }
    std::fflush(stdout);
    return true;
}

bool allocate_io(uint32_t n_tokens, uint32_t in_dim, uint64_t out_elements,
                 tensor_owner *x, tensor_owner *out_a, tensor_owner *out_b) {
    std::vector<float> activation;
    fill_activation(&activation, n_tokens, in_dim);
    if (!x->ptr || !out_a->ptr || !out_b->ptr ||
        !ds4_gpu_tensor_write(x->ptr, 0, activation.data(),
                              activation.size() * sizeof(float))) {
        return false;
    }
    const uint64_t logical_bytes = out_elements * sizeof(float);
    return poison_output(out_a->ptr, logical_bytes, 0x7fc10001u) &&
           poison_output(out_b->ptr, logical_bytes, 0x7fc20002u);
}

bool run_q8_quantizer(const config &cfg, ds4_gpu_tensor *x,
                      uint32_t n_tokens) {
    int active_device = -1;
    hipDeviceProp_t properties{};
    if (hipGetDevice(&active_device) != hipSuccess || active_device < 0 ||
        hipGetDeviceProperties(&properties, active_device) != hipSuccess ||
        properties.warpSize != 32 ||
        std::strncmp(properties.gcnArchName, "gfx1151", 7u) != 0) {
        std::fprintf(stderr,
                     "dense_q8_wave32 N=%u: active device is not "
                     "gfx1151 wave32\n",
                     n_tokens);
        return false;
    }

    uint64_t block_count = 0;
    uint64_t logical_bytes = 0;
    uint64_t x_bytes = 0;
    if (!x ||
        !checked_mul(n_tokens, kDenseK / kQkK, &block_count) ||
        !checked_mul(block_count, sizeof(block_q8_K_host), &logical_bytes) ||
        !checked_mul(static_cast<uint64_t>(n_tokens) * kDenseK,
                     sizeof(float), &x_bytes) ||
        logical_bytes > std::numeric_limits<uint64_t>::max() -
                            kRawQ8GuardWords * sizeof(uint32_t) ||
        ds4_gpu_tensor_bytes(x) < x_bytes) {
        std::fprintf(stderr,
                     "dense_q8_wave32 N=%u: raw tensor size overflow\n",
                     n_tokens);
        return false;
    }

    const uint64_t allocation_bytes = logical_bytes +
        kRawQ8GuardWords * sizeof(uint32_t);
    tensor_owner canonical(allocation_bytes);
    tensor_owner wave32(allocation_bytes);
    if (!canonical.ptr || !wave32.ptr) {
        std::fprintf(stderr,
                     "dense_q8_wave32 N=%u: raw tensor allocation failed\n",
                     n_tokens);
        return false;
    }

    /* contents() synchronizes; resolve all raw device pointers before any
     * event is recorded so the timed callbacks contain one launch only. */
    const void *x_device = ds4_gpu_tensor_contents(x);
    void *canonical_device = ds4_gpu_tensor_contents(canonical.ptr);
    void *wave32_device = ds4_gpu_tensor_contents(wave32.ptr);
    if (!x_device || !canonical_device || !wave32_device) {
        std::fprintf(stderr,
                     "dense_q8_wave32 N=%u: device pointer resolution failed\n",
                     n_tokens);
        return false;
    }

    const arm q8_canonical = {
        "q8_canonical_raw", []() {},
        [&](uint32_t) {
            ds4_rocm_bench_q8_K_quantize_enqueue(
                canonical_device, x_device, kDenseK, n_tokens, 0);
            return true;
        }};
    const arm q8_wave32 = {
        "q8_wave32_raw", []() {},
        [&](uint32_t) {
            ds4_rocm_bench_q8_K_quantize_enqueue(
                wave32_device, x_device, kDenseK, n_tokens, 1);
            return true;
        }};

    config q8_cfg = cfg;
    q8_cfg.sets = 1u;  // Raw quantization has no rotating weight set.
    return benchmark_arms(
        "dense_q8_wave32", n_tokens, kDenseK, 1u, q8_cfg,
        q8_canonical, q8_wave32,
        [&]() {
            return poison_output(canonical.ptr, logical_bytes, 0x5a5a5a5au,
                                 kRawQ8GuardWords) &&
                   poison_output(wave32.ptr, logical_bytes, 0xa5a5a5a5u,
                                 kRawQ8GuardWords);
        },
        [&]() {
            return bitwise_equal(canonical.ptr, wave32.ptr, logical_bytes,
                                 "raw canonical vs wave32 Q8_K") &&
                   check_guard(canonical.ptr, logical_bytes,
                               "raw canonical Q8_K", kRawQ8GuardWords) &&
                   check_guard(wave32.ptr, logical_bytes,
                               "raw wave32 Q8_K", kRawQ8GuardWords);
        },
        benchmark_rate::quantized_values);
}

// Isolate this change: same quantizer, tile shape, grid and launch count.
// Public-API HIP-event timing includes Q8_K quantization in both arms.
bool run_lds(const model_fixture &model, const config &cfg,
             uint32_t n_tokens, bool qb, bool vector_only = false) {
    env_snapshot lds_guard(kLdsDisable);
    env_snapshot vector_guard(kLdsVectorDisable);
    const uint32_t k = qb ? kQbK : kDenseK;
    const uint32_t m = qb ? kQbM : kDenseM;
    const uint64_t elements = (uint64_t)n_tokens * m;
    const uint64_t bytes = elements * sizeof(float);
    const uint64_t allocation = bytes + kGuardWords * sizeof(uint32_t);
    tensor_owner x((uint64_t)n_tokens * k * sizeof(float));
    tensor_owner reference(allocation), candidate(allocation);
    if (!allocate_io(n_tokens, k, elements, &x, &reference, &candidate)) return false;
    auto enqueue = [&](uint32_t set, ds4_gpu_tensor *out, bool optimized) {
        const uint64_t offset = qb ? model.weights[set].qb_offset :
                                     model.weights[set].dense_offset;
        ds4_rocm_test_q4_prefill_lds_stream_reset();
        ds4_rocm_test_q4_prefill_lds_vector_reset();
        const int rc = ds4_gpu_matmul_quant_tensor(
            out, model.data, model.size, offset, kQ4Type, k, m, x.ptr, n_tokens);
        const uint64_t calls = ds4_rocm_test_q4_prefill_lds_stream_get_calls();
        const uint64_t vector_calls = ds4_rocm_test_q4_prefill_lds_vector_get_calls();
        if (rc == 0 || calls != (optimized || vector_only ? 1u : 0u) ||
            vector_calls != (optimized && vector_only ? 1u : 0u)) {
            std::fprintf(stderr, "Q4 LDS benchmark wrong path: optimized=%d vector_only=%d rc=%d calls=%llu vector_calls=%llu\n",
                         optimized, vector_only, rc, (unsigned long long)calls,
                         (unsigned long long)vector_calls);
            return false;
        }
        return true;
    };
    const arm baseline = {
        vector_only ? "lds_stream_scalar" : "lds_legacy",
        [&]() {
            if (qb) select_k1024_tile4(); else select_tile8(false);
            if (vector_only) (void)unsetenv(kLdsDisable);
            else (void)setenv(kLdsDisable, "1", 1);
            (void)setenv(kLdsVectorDisable, "1", 1);
        },
        [&](uint32_t set) { return enqueue(set, reference.ptr, false); }};
    const arm optimized = {
        vector_only ? "lds_stream_vector" : "lds_stream",
        [&]() {
            if (qb) select_k1024_tile4(); else select_tile8(false);
            (void)unsetenv(kLdsDisable);
            if (vector_only) (void)unsetenv(kLdsVectorDisable);
            else (void)setenv(kLdsVectorDisable, "1", 1);
        },
        [&](uint32_t set) { return enqueue(set, candidate.ptr, true); }};
    return benchmark_arms(
        vector_only ? (qb ? "q_b_lds_vector" : "dense_lds_vector") :
                      (qb ? "q_b_lds" : "dense_lds"),
        n_tokens, k, m, cfg, baseline, optimized,
        [&]() {
            return poison_output(reference.ptr, bytes, 0x7fc10001u) &&
                   poison_output(candidate.ptr, bytes, 0x7fc20002u);
        },
        [&]() {
            return bitwise_equal(reference.ptr, candidate.ptr, bytes, "Q4 LDS rollback vs default") &&
                   check_guard(reference.ptr, bytes, "Q4 LDS reference") &&
                   check_guard(candidate.ptr, bytes, "Q4 LDS candidate");
        });
}

// Enqueue-only A/B: the old hook is explicitly LOAD2, the new strict hook
// explicitly LOAD4. No mutable environment or production selector in timing.
bool run_wmma_load4(const model_fixture &model, const config &cfg,
                    uint32_t n, uint32_t k, uint32_t m, uint32_t groups,
                    uint64_t weight_set::*offset, const char *label) {
    if (!cfg.wmma_supported || n < 256u) return true;
    const uint64_t elements = (uint64_t)n * groups * m;
    const uint64_t bytes = elements * sizeof(float);
    tensor_owner x((uint64_t)n * groups * k * sizeof(float));
    tensor_owner a(bytes + kGuardWords * sizeof(uint32_t));
    tensor_owner b(bytes + kGuardWords * sizeof(uint32_t));
    if (!allocate_io(n, groups * k, elements, &x, &a, &b)) return false;
    const void *xp = ds4_gpu_tensor_contents(x.ptr);
    void *ap = ds4_gpu_tensor_contents(a.ptr);
    void *bp = ds4_gpu_tensor_contents(b.ptr);
    std::vector<const void *> weights;
    if (!xp || !ap || !bp || !resolve_resident_weights(
            model, offset, q4_weight_bytes(k, groups * m), &weights)) return false;
    const uint32_t rows = shape_wmma_row_tile(m);
    const uint64_t row_bytes = q4_weight_bytes(k, 1u);
    const arm baseline = {"wmma_k64p80_load2", []() {}, [&](uint32_t set) {
        return ds4_rocm_bench_q4_K_wmma_variant_enqueue(
            ap, weights[set], xp, n, groups, k, m, row_bytes,
            (uint64_t)groups * k, k, (uint64_t)groups * m, rows, 64u, 1) != 0;
    }};
    const arm candidate = {"wmma_k64p80_load4", []() {}, [&](uint32_t set) {
        return ds4_rocm_bench_q4_K_wmma_k64_load4_enqueue(
            bp, weights[set], xp, n, groups, k, m, row_bytes,
            (uint64_t)groups * k, k, (uint64_t)groups * m, rows) != 0;
    }};
    return benchmark_arms(label, n, k, groups * m, cfg, baseline, candidate,
        [&]() {
            return poison_output(a.ptr, bytes, 0x7fc10001u) &&
                   poison_output(b.ptr, bytes, 0x7fc20002u);
        }, [&]() {
            return bitwise_equal(a.ptr, b.ptr, bytes, label) &&
                   check_guard(a.ptr, bytes, "K64 LOAD2 guard") &&
                   check_guard(b.ptr, bytes, "K64 LOAD4 guard");
        }, benchmark_rate::macs, true);
}

bool run_dense(const model_fixture &model, const config &cfg,
               uint32_t n_tokens) {
    uint64_t out_elements = 0;
    if (!checked_mul(n_tokens, kDenseM, &out_elements)) return false;
    const uint64_t logical_bytes = out_elements * sizeof(float);
    const uint64_t allocation_bytes = logical_bytes +
                                      kGuardWords * sizeof(uint32_t);
    tensor_owner x(static_cast<uint64_t>(n_tokens) * kDenseK * sizeof(float));
    tensor_owner legacy(allocation_bytes);
    tensor_owner tiled(allocation_bytes);
    if (!allocate_io(n_tokens, kDenseK, out_elements, &x, &legacy, &tiled)) {
        std::fprintf(stderr, "dense N=%u: tensor setup failed\n", n_tokens);
        return false;
    }
    const arm baseline = {
        "legacy", select_legacy,
        [&](uint32_t set) {
            return ds4_gpu_matmul_quant_tensor(
                       legacy.ptr, model.data, model.size,
                       model.weights[set].dense_offset, kQ4Type, kDenseK,
                       kDenseM, x.ptr, n_tokens) != 0;
        }};
    const arm candidate = {
        "tile8", []() { select_tile8(false); },
        [&](uint32_t set) {
            return ds4_gpu_matmul_quant_tensor(
                       tiled.ptr, model.data, model.size,
                       model.weights[set].dense_offset, kQ4Type, kDenseK,
                       kDenseM, x.ptr, n_tokens) != 0;
        }};
    if (!benchmark_arms(
            "dense", n_tokens, kDenseK, kDenseM, cfg, baseline, candidate,
            [&]() {
                return poison_output(legacy.ptr, logical_bytes, 0x7fc10001u) &&
                       poison_output(tiled.ptr, logical_bytes, 0x7fc20002u);
            },
            [&]() {
                return bitwise_equal(legacy.ptr, tiled.ptr, logical_bytes,
                                     "dense legacy vs tile8") &&
                       check_guard(legacy.ptr, logical_bytes,
                                   "dense legacy oracle") &&
                       check_guard(tiled.ptr, logical_bytes,
                                   "dense tile8 oracle");
            })) return false;
    if (!bitwise_equal(legacy.ptr, tiled.ptr, logical_bytes,
                       "dense legacy vs tile8") ||
        !check_guard(legacy.ptr, logical_bytes, "dense legacy") ||
        !check_guard(tiled.ptr, logical_bytes, "dense tile8")) {
        return false;
    }
    if (!cfg.wmma_supported) return true;
    if (!run_q8_quantizer(cfg, x.ptr, n_tokens)) return false;

    if (n_tokens < 256u) return true;

    const arm wmma_baseline = {
        "tile8", []() { select_tile8(false); },
        [&](uint32_t set) {
            return ds4_gpu_matmul_quant_tensor(
                       legacy.ptr, model.data, model.size,
                       model.weights[set].dense_offset, kQ4Type, kDenseK,
                       kDenseM, x.ptr, n_tokens) != 0;
        }};
    const arm wmma_candidate = {
        "wmma_shape", select_wmma_shape,
        [&](uint32_t set) {
            return ds4_gpu_matmul_quant_tensor(
                       tiled.ptr, model.data, model.size,
                       model.weights[set].dense_offset, kQ4Type, kDenseK,
                       kDenseM, x.ptr, n_tokens) != 0;
        }};
    if (!benchmark_arms(
        "dense_wmma", n_tokens, kDenseK, kDenseM, cfg, wmma_baseline,
        wmma_candidate,
        [&]() {
            return poison_output(legacy.ptr, logical_bytes, 0x7fc10001u) &&
                   poison_output(tiled.ptr, logical_bytes, 0x7fc20002u);
        },
        [&]() {
            return numerically_close(tiled.ptr, legacy.ptr, logical_bytes,
                                     "dense WMMA shape vs TILE8") &&
                   check_guard(legacy.ptr, logical_bytes,
                               "dense WMMA baseline oracle") &&
                   check_guard(tiled.ptr, logical_bytes,
                               "dense WMMA shape oracle");
        })) return false;

    void *const x_device = ds4_gpu_tensor_contents(x.ptr);
    void *const rows64_device = ds4_gpu_tensor_contents(legacy.ptr);
    void *const shape_device = ds4_gpu_tensor_contents(tiled.ptr);
    std::vector<const void *> dense_weights;
    if (!x_device || !rows64_device || !shape_device ||
        !resolve_resident_weights(
            model, &weight_set::dense_offset,
            q4_weight_bytes(kDenseK, kDenseM), &dense_weights)) {
        std::fprintf(stderr,
                     "dense_wmma_rows N=%u: direct setup failed\n", n_tokens);
        return false;
    }
    const uint64_t row_bytes = q4_weight_bytes(kDenseK, 1u);
    const arm geometry_baseline = {
        "wmma_rows64_scalar", []() {},
        [&](uint32_t set) {
            return ds4_rocm_bench_q4_K_wmma_enqueue(
                       rows64_device, dense_weights[set], x_device,
                       n_tokens, 1u, kDenseK, kDenseM, row_bytes,
                       kDenseK, 0u, kDenseM, 64u, 0) != 0;
        }};
    const arm geometry_candidate = {
        "wmma_shape_scalar", []() {},
        [&](uint32_t set) {
            return ds4_rocm_bench_q4_K_wmma_enqueue(
                       shape_device, dense_weights[set], x_device,
                       n_tokens, 1u, kDenseK, kDenseM, row_bytes,
                       kDenseK, 0u, kDenseM,
                       shape_wmma_row_tile(kDenseM), 0) != 0;
        }};
    if (!benchmark_arms(
        "dense_wmma_rows", n_tokens, kDenseK, kDenseM, cfg,
        geometry_baseline, geometry_candidate,
        [&]() {
            return poison_output(legacy.ptr, logical_bytes, 0x7fc10001u) &&
                   poison_output(tiled.ptr, logical_bytes, 0x7fc20002u);
        },
        [&]() {
            return bitwise_equal(legacy.ptr, tiled.ptr, logical_bytes,
                                 "dense WMMA rows64 vs shape") &&
                   check_guard(legacy.ptr, logical_bytes,
                               "dense WMMA rows64 oracle") &&
                   check_guard(tiled.ptr, logical_bytes,
                               "dense WMMA shape oracle");
        })) return false;

    const uint32_t fixed_row_tile = shape_wmma_row_tile(kDenseM);
    const arm k32_baseline = {
        "wmma_k32_rows128_load2", []() {},
        [&](uint32_t set) {
            return ds4_rocm_bench_q4_K_wmma_variant_enqueue(
                       rows64_device, dense_weights[set], x_device,
                       n_tokens, 1u, kDenseK, kDenseM, row_bytes,
                       kDenseK, 0u, kDenseM, fixed_row_tile, 32u, 1) != 0;
        }};
    const arm k64_candidate = {
        "wmma_k64p80_rows128_load2", []() {},
        [&](uint32_t set) {
            return ds4_rocm_bench_q4_K_wmma_variant_enqueue(
                       shape_device, dense_weights[set], x_device,
                       n_tokens, 1u, kDenseK, kDenseM, row_bytes,
                       kDenseK, 0u, kDenseM, fixed_row_tile, 64u, 1) != 0;
        }};
    return benchmark_arms(
        "dense_wmma_k32_k64", n_tokens, kDenseK, kDenseM, cfg,
        k32_baseline, k64_candidate,
        [&]() {
            return poison_output(legacy.ptr, logical_bytes, 0x7fc10001u) &&
                   poison_output(tiled.ptr, logical_bytes, 0x7fc20002u);
        },
        [&]() {
            return bitwise_equal(legacy.ptr, tiled.ptr, logical_bytes,
                                 "dense WMMA K32 vs K64/P80") &&
                   check_guard(legacy.ptr, logical_bytes,
                               "dense WMMA K32 oracle") &&
                   check_guard(tiled.ptr, logical_bytes,
                               "dense WMMA K64/P80 oracle");
        });
}

bool run_pair(const model_fixture &model, const config &cfg,
              uint32_t n_tokens) {
    uint64_t out0_elements = 0, out1_elements = 0;
    if (!checked_mul(n_tokens, kDenseM, &out0_elements) ||
        !checked_mul(n_tokens, kKvM, &out1_elements)) return false;
    const uint64_t out0_bytes = out0_elements * sizeof(float);
    const uint64_t out1_bytes = out1_elements * sizeof(float);
    const uint64_t guard_bytes = kGuardWords * sizeof(uint32_t);
    tensor_owner x(static_cast<uint64_t>(n_tokens) * kDenseK * sizeof(float));
    tensor_owner separate0(out0_bytes + guard_bytes);
    tensor_owner separate1(out1_bytes + guard_bytes);
    tensor_owner pair0(out0_bytes + guard_bytes);
    tensor_owner pair1(out1_bytes + guard_bytes);
    std::vector<float> activation;
    fill_activation(&activation, n_tokens, kDenseK);
    if (!x.ptr || !separate0.ptr || !separate1.ptr || !pair0.ptr || !pair1.ptr ||
        !ds4_gpu_tensor_write(x.ptr, 0, activation.data(),
                              activation.size() * sizeof(float)) ||
        !prepare_guard(separate0.ptr, out0_bytes) ||
        !prepare_guard(separate1.ptr, out1_bytes) ||
        !prepare_guard(pair0.ptr, out0_bytes) ||
        !prepare_guard(pair1.ptr, out1_bytes)) {
        std::fprintf(stderr, "pair N=%u: tensor setup failed\n", n_tokens);
        return false;
    }
    const arm baseline = {
        "two_dense_tile8", []() { select_tile8(false); },
        [&](uint32_t set) {
            return ds4_gpu_matmul_quant_tensor(
                       separate0.ptr, model.data, model.size,
                       model.weights[set].dense_offset, kQ4Type, kDenseK,
                       kDenseM, x.ptr, n_tokens) != 0 &&
                   ds4_gpu_matmul_quant_tensor(
                       separate1.ptr, model.data, model.size,
                       model.weights[set].kv_offset, kQ4Type, kDenseK, kKvM,
                       x.ptr, n_tokens) != 0;
        }};
    const arm candidate = {
        "pair_tile8", []() { select_tile8(false); },
        [&](uint32_t set) {
            return ds4_gpu_matmul_q4_K_pair_tensor(
                       pair0.ptr, pair1.ptr, model.data, model.size,
                       model.weights[set].dense_offset,
                       model.weights[set].kv_offset, kDenseK, kDenseM, kKvM,
                       x.ptr, n_tokens) == 1;
        }};
    if (!benchmark_arms(
            "pair", n_tokens, kDenseK, kDenseM + kKvM, cfg, baseline,
            candidate,
            [&]() {
                return poison_output(separate0.ptr, out0_bytes, 0x7fc10001u) &&
                       poison_output(separate1.ptr, out1_bytes, 0x7fc20002u) &&
                       poison_output(pair0.ptr, out0_bytes, 0x7fc30003u) &&
                       poison_output(pair1.ptr, out1_bytes, 0x7fc40004u);
            },
            [&]() {
                return bitwise_equal(separate0.ptr, pair0.ptr, out0_bytes,
                                     "pair q_a output") &&
                       bitwise_equal(separate1.ptr, pair1.ptr, out1_bytes,
                                     "pair kv output") &&
                       check_guard(separate0.ptr, out0_bytes,
                                   "pair separate q_a oracle") &&
                       check_guard(separate1.ptr, out1_bytes,
                                   "pair separate kv oracle") &&
                       check_guard(pair0.ptr, out0_bytes,
                                   "pair fused q_a oracle") &&
                       check_guard(pair1.ptr, out1_bytes,
                                   "pair fused kv oracle");
            })) return false;
    return bitwise_equal(separate0.ptr, pair0.ptr, out0_bytes,
                         "pair q_a output") &&
           bitwise_equal(separate1.ptr, pair1.ptr, out1_bytes,
                         "pair kv output") &&
           check_guard(separate0.ptr, out0_bytes, "pair separate q_a") &&
           check_guard(separate1.ptr, out1_bytes, "pair separate kv") &&
           check_guard(pair0.ptr, out0_bytes, "pair fused q_a") &&
           check_guard(pair1.ptr, out1_bytes, "pair fused kv");
}

bool run_qb(const model_fixture &model, const config &cfg,
            uint32_t n_tokens) {
    uint64_t out_elements = 0;
    if (!checked_mul(n_tokens, kQbM, &out_elements)) return false;
    const uint64_t logical_bytes = out_elements * sizeof(float);
    const uint64_t allocation_bytes = logical_bytes +
                                      kGuardWords * sizeof(uint32_t);
    tensor_owner x(static_cast<uint64_t>(n_tokens) * kQbK * sizeof(float));
    tensor_owner tile8(allocation_bytes);
    tensor_owner tile4(allocation_bytes);
    if (!allocate_io(n_tokens, kQbK, out_elements, &x, &tile8, &tile4)) {
        std::fprintf(stderr, "q_b N=%u: tensor setup failed\n", n_tokens);
        return false;
    }
    const arm baseline = {
        "tile8", []() { select_tile8(true); },
        [&](uint32_t set) {
            return ds4_gpu_matmul_quant_tensor(
                       tile8.ptr, model.data, model.size,
                       model.weights[set].qb_offset, kQ4Type, kQbK, kQbM,
                       x.ptr, n_tokens) != 0;
        }};
    const arm candidate = {
        "tile4", select_k1024_tile4,
        [&](uint32_t set) {
            return ds4_gpu_matmul_quant_tensor(
                       tile4.ptr, model.data, model.size,
                       model.weights[set].qb_offset, kQ4Type, kQbK, kQbM,
                       x.ptr, n_tokens) != 0;
        }};
    if (!benchmark_arms(
            "q_b", n_tokens, kQbK, kQbM, cfg, baseline, candidate,
            [&]() {
                return poison_output(tile8.ptr, logical_bytes, 0x7fc10001u) &&
                       poison_output(tile4.ptr, logical_bytes, 0x7fc20002u);
            },
            [&]() {
                return bitwise_equal(tile8.ptr, tile4.ptr, logical_bytes,
                                     "q_b tile8 vs tile4") &&
                       check_guard(tile8.ptr, logical_bytes,
                                   "q_b tile8 oracle") &&
                       check_guard(tile4.ptr, logical_bytes,
                                   "q_b tile4 oracle");
            })) return false;
    if (!bitwise_equal(tile8.ptr, tile4.ptr, logical_bytes,
                       "q_b tile8 vs tile4") ||
        !check_guard(tile8.ptr, logical_bytes, "q_b tile8") ||
        !check_guard(tile4.ptr, logical_bytes, "q_b tile4")) {
        return false;
    }
    if (!cfg.wmma_supported || n_tokens < 256u) return true;

    const arm wmma_baseline = {
        "tile4", select_k1024_tile4,
        [&](uint32_t set) {
            return ds4_gpu_matmul_quant_tensor(
                       tile8.ptr, model.data, model.size,
                       model.weights[set].qb_offset, kQ4Type, kQbK, kQbM,
                       x.ptr, n_tokens) != 0;
        }};
    const arm wmma_candidate = {
        "wmma_shape", select_wmma_shape,
        [&](uint32_t set) {
            return ds4_gpu_matmul_quant_tensor(
                       tile4.ptr, model.data, model.size,
                       model.weights[set].qb_offset, kQ4Type, kQbK, kQbM,
                       x.ptr, n_tokens) != 0;
        }};
    if (!benchmark_arms(
        "q_b_wmma", n_tokens, kQbK, kQbM, cfg, wmma_baseline,
        wmma_candidate,
        [&]() {
            return poison_output(tile8.ptr, logical_bytes, 0x7fc10001u) &&
                   poison_output(tile4.ptr, logical_bytes, 0x7fc20002u);
        },
        [&]() {
            return numerically_close(tile4.ptr, tile8.ptr, logical_bytes,
                                     "q_b WMMA shape vs TILE4") &&
                   check_guard(tile8.ptr, logical_bytes,
                               "q_b WMMA baseline oracle") &&
                   check_guard(tile4.ptr, logical_bytes,
                               "q_b WMMA shape oracle");
        })) return false;

    void *const x_device = ds4_gpu_tensor_contents(x.ptr);
    void *const rows64_device = ds4_gpu_tensor_contents(tile8.ptr);
    void *const shape_device = ds4_gpu_tensor_contents(tile4.ptr);
    std::vector<const void *> qb_weights;
    if (!x_device || !rows64_device || !shape_device ||
        !resolve_resident_weights(
            model, &weight_set::qb_offset,
            q4_weight_bytes(kQbK, kQbM), &qb_weights)) {
        std::fprintf(stderr,
                     "q_b_wmma_rows N=%u: direct setup failed\n", n_tokens);
        return false;
    }
    const uint64_t row_bytes = q4_weight_bytes(kQbK, 1u);
    const arm geometry_baseline = {
        "wmma_rows64_scalar", []() {},
        [&](uint32_t set) {
            return ds4_rocm_bench_q4_K_wmma_enqueue(
                       rows64_device, qb_weights[set], x_device,
                       n_tokens, 1u, kQbK, kQbM, row_bytes,
                       kQbK, 0u, kQbM, 64u, 0) != 0;
        }};
    const arm rows128_candidate = {
        "wmma_rows128_scalar", []() {},
        [&](uint32_t set) {
            return ds4_rocm_bench_q4_K_wmma_enqueue(
                       shape_device, qb_weights[set], x_device,
                       n_tokens, 1u, kQbK, kQbM, row_bytes,
                       kQbK, 0u, kQbM, 128u, 0) != 0;
        }};
    if (!benchmark_arms(
        "q_b_wmma_rows64_128", n_tokens, kQbK, kQbM, cfg,
        geometry_baseline, rows128_candidate,
        [&]() {
            return poison_output(tile8.ptr, logical_bytes, 0x7fc10001u) &&
                   poison_output(tile4.ptr, logical_bytes, 0x7fc20002u);
        },
        [&]() {
            return bitwise_equal(tile8.ptr, tile4.ptr, logical_bytes,
                                 "q_b WMMA rows64 vs rows128") &&
                   check_guard(tile8.ptr, logical_bytes,
                               "q_b WMMA rows64 oracle") &&
                   check_guard(tile4.ptr, logical_bytes,
                               "q_b WMMA rows128 oracle");
        })) return false;

    const arm rows128_baseline = {
        "wmma_rows128_scalar", []() {},
        [&](uint32_t set) {
            return ds4_rocm_bench_q4_K_wmma_enqueue(
                       rows64_device, qb_weights[set], x_device,
                       n_tokens, 1u, kQbK, kQbM, row_bytes,
                       kQbK, 0u, kQbM, 128u, 0) != 0;
        }};
    const arm rows256_candidate = {
        "wmma_rows256_scalar", []() {},
        [&](uint32_t set) {
            return ds4_rocm_bench_q4_K_wmma_enqueue(
                       shape_device, qb_weights[set], x_device,
                       n_tokens, 1u, kQbK, kQbM, row_bytes,
                       kQbK, 0u, kQbM, 256u, 0) != 0;
        }};
    if (!benchmark_arms(
        "q_b_wmma_rows128_256", n_tokens, kQbK, kQbM, cfg,
        rows128_baseline, rows256_candidate,
        [&]() {
            return poison_output(tile8.ptr, logical_bytes, 0x7fc10001u) &&
                   poison_output(tile4.ptr, logical_bytes, 0x7fc20002u);
        },
        [&]() {
            return bitwise_equal(tile8.ptr, tile4.ptr, logical_bytes,
                                 "q_b WMMA rows128 vs rows256") &&
                   check_guard(tile8.ptr, logical_bytes,
                               "q_b WMMA rows128 oracle") &&
                   check_guard(tile4.ptr, logical_bytes,
                               "q_b WMMA rows256 oracle");
        })) return false;

    const arm rows128_scalar = {
        "wmma_rows128_scalar", []() {},
        [&](uint32_t set) {
            return ds4_rocm_bench_q4_K_wmma_enqueue(
                       rows64_device, qb_weights[set], x_device,
                       n_tokens, 1u, kQbK, kQbM, row_bytes,
                       kQbK, 0u, kQbM, 128u, 0) != 0;
        }};
    const arm rows128_load2 = {
        "wmma_rows128_load2", []() {},
        [&](uint32_t set) {
            return ds4_rocm_bench_q4_K_wmma_enqueue(
                       shape_device, qb_weights[set], x_device,
                       n_tokens, 1u, kQbK, kQbM, row_bytes,
                       kQbK, 0u, kQbM, 128u, 1) != 0;
        }};
    if (!benchmark_arms(
        "q_b_wmma_load2_128", n_tokens, kQbK, kQbM, cfg,
        rows128_scalar, rows128_load2,
        [&]() {
            return poison_output(tile8.ptr, logical_bytes, 0x7fc10001u) &&
                   poison_output(tile4.ptr, logical_bytes, 0x7fc20002u);
        },
        [&]() {
            return bitwise_equal(tile8.ptr, tile4.ptr, logical_bytes,
                                 "q_b WMMA rows128 scalar vs load2") &&
                   check_guard(tile8.ptr, logical_bytes,
                               "q_b WMMA rows128 scalar oracle") &&
                   check_guard(tile4.ptr, logical_bytes,
                               "q_b WMMA rows128 load2 oracle");
        })) return false;

    const arm rows256_scalar = {
        "wmma_rows256_scalar", []() {},
        [&](uint32_t set) {
            return ds4_rocm_bench_q4_K_wmma_enqueue(
                       rows64_device, qb_weights[set], x_device,
                       n_tokens, 1u, kQbK, kQbM, row_bytes,
                       kQbK, 0u, kQbM, 256u, 0) != 0;
        }};
    const arm rows256_load2 = {
        "wmma_rows256_load2", []() {},
        [&](uint32_t set) {
            return ds4_rocm_bench_q4_K_wmma_enqueue(
                       shape_device, qb_weights[set], x_device,
                       n_tokens, 1u, kQbK, kQbM, row_bytes,
                       kQbK, 0u, kQbM, 256u, 1) != 0;
        }};
    if (!benchmark_arms(
        "q_b_wmma_load2_256", n_tokens, kQbK, kQbM, cfg,
        rows256_scalar, rows256_load2,
        [&]() {
            return poison_output(tile8.ptr, logical_bytes, 0x7fc10001u) &&
                   poison_output(tile4.ptr, logical_bytes, 0x7fc20002u);
        },
        [&]() {
            return bitwise_equal(tile8.ptr, tile4.ptr, logical_bytes,
                                 "q_b WMMA rows256 scalar vs load2") &&
                   check_guard(tile8.ptr, logical_bytes,
                               "q_b WMMA rows256 scalar oracle") &&
                   check_guard(tile4.ptr, logical_bytes,
                               "q_b WMMA rows256 load2 oracle");
        })) return false;

    const arm k32_baseline = {
        "wmma_k32_rows256_load2", []() {},
        [&](uint32_t set) {
            return ds4_rocm_bench_q4_K_wmma_variant_enqueue(
                       rows64_device, qb_weights[set], x_device,
                       n_tokens, 1u, kQbK, kQbM, row_bytes,
                       kQbK, 0u, kQbM, 256u, 32u, 1) != 0;
        }};
    const arm k64_candidate = {
        "wmma_k64p80_rows256_load2", []() {},
        [&](uint32_t set) {
            return ds4_rocm_bench_q4_K_wmma_variant_enqueue(
                       shape_device, qb_weights[set], x_device,
                       n_tokens, 1u, kQbK, kQbM, row_bytes,
                       kQbK, 0u, kQbM, 256u, 64u, 1) != 0;
        }};
    if (!benchmark_arms(
        "q_b_wmma_k32_k64", n_tokens, kQbK, kQbM, cfg,
        k32_baseline, k64_candidate,
        [&]() {
            return poison_output(tile8.ptr, logical_bytes, 0x7fc10001u) &&
                   poison_output(tile4.ptr, logical_bytes, 0x7fc20002u);
        },
        [&]() {
            return bitwise_equal(tile8.ptr, tile4.ptr, logical_bytes,
                                 "q_b WMMA K32 vs K64/P80") &&
                   check_guard(tile8.ptr, logical_bytes,
                               "q_b WMMA K32 oracle") &&
                   check_guard(tile4.ptr, logical_bytes,
                               "q_b WMMA K64/P80 oracle");
        })) return false;

    const arm k64_baseline = {
        "wmma_k64p80_rows256_load2", []() {},
        [&](uint32_t set) {
            return ds4_rocm_bench_q4_K_wmma_variant_enqueue(
                       rows64_device, qb_weights[set], x_device,
                       n_tokens, 1u, kQbK, kQbM, row_bytes,
                       kQbK, 0u, kQbM, 256u, 64u, 1) != 0;
        }};
    const arm k128_candidate = {
        "wmma_k128p144_rows256_load4", []() {},
        [&](uint32_t set) {
            return ds4_rocm_bench_q4_K_wmma_k128_enqueue(
                       shape_device, qb_weights[set], x_device,
                       n_tokens, 1u, kQbK, kQbM, row_bytes,
                       kQbK, 0u, kQbM) != 0;
        }};
    return benchmark_arms(
        "q_b_wmma_k64_k128", n_tokens, kQbK, kQbM, cfg,
        k64_baseline, k128_candidate,
        [&]() {
            return poison_output(tile8.ptr, logical_bytes, 0x7fc10001u) &&
                   poison_output(tile4.ptr, logical_bytes, 0x7fc20002u);
        },
        [&]() {
            return bitwise_equal(tile8.ptr, tile4.ptr, logical_bytes,
                                 "q_b WMMA K64/P80 vs K128/P144") &&
                   check_guard(tile8.ptr, logical_bytes,
                               "q_b WMMA K64/P80 oracle") &&
                   check_guard(tile4.ptr, logical_bytes,
                               "q_b WMMA K128/P144 oracle");
        });
}

bool run_output_b(const model_fixture &model, const config &cfg,
                  uint32_t n_tokens) {
    if (!cfg.wmma_supported || n_tokens < 256u) return true;
    uint64_t out_elements = 0;
    if (!checked_mul(n_tokens, kOutputM, &out_elements)) return false;
    const uint64_t logical_bytes = out_elements * sizeof(float);
    const uint64_t allocation_bytes = logical_bytes +
                                      kGuardWords * sizeof(uint32_t);
    tensor_owner x(
        static_cast<uint64_t>(n_tokens) * kOutputLowDim * sizeof(float));
    tensor_owner tile8(allocation_bytes);
    tensor_owner wmma(allocation_bytes);
    if (!allocate_io(n_tokens, kOutputLowDim, out_elements,
                     &x, &tile8, &wmma)) {
        std::fprintf(stderr, "output_b N=%u: tensor setup failed\n", n_tokens);
        return false;
    }

    const arm baseline = {
        "tile8", []() { select_tile8(false); },
        [&](uint32_t set) {
            return ds4_gpu_matmul_quant_tensor(
                       tile8.ptr, model.data, model.size,
                       model.weights[set].output_b_offset, kQ4Type,
                       kOutputLowDim, kOutputM, x.ptr, n_tokens) != 0;
        }};
    const arm candidate = {
        "wmma_shape", select_wmma_shape,
        [&](uint32_t set) {
            return ds4_gpu_matmul_quant_tensor(
                       wmma.ptr, model.data, model.size,
                       model.weights[set].output_b_offset, kQ4Type,
                       kOutputLowDim, kOutputM, x.ptr, n_tokens) != 0;
        }};
    if (!benchmark_arms(
        "output_b_wmma", n_tokens, kOutputLowDim, kOutputM, cfg,
        baseline, candidate,
        [&]() {
            return poison_output(tile8.ptr, logical_bytes, 0x7fc10001u) &&
                   poison_output(wmma.ptr, logical_bytes, 0x7fc20002u);
        },
        [&]() {
            return numerically_close(wmma.ptr, tile8.ptr, logical_bytes,
                                     "output_b WMMA shape vs TILE8") &&
                   check_guard(tile8.ptr, logical_bytes,
                               "output_b TILE8 oracle") &&
                   check_guard(wmma.ptr, logical_bytes,
                               "output_b WMMA shape oracle");
        })) return false;

    void *const x_device = ds4_gpu_tensor_contents(x.ptr);
    void *const rows64_device = ds4_gpu_tensor_contents(tile8.ptr);
    void *const shape_device = ds4_gpu_tensor_contents(wmma.ptr);
    std::vector<const void *> output_b_weights;
    if (!x_device || !rows64_device || !shape_device ||
        !resolve_resident_weights(
            model, &weight_set::output_b_offset,
            q4_weight_bytes(kOutputLowDim, kOutputM), &output_b_weights)) {
        std::fprintf(stderr,
                     "output_b_wmma_rows N=%u: direct setup failed\n",
                     n_tokens);
        return false;
    }
    const uint64_t row_bytes = q4_weight_bytes(kOutputLowDim, 1u);
    const arm geometry_baseline = {
        "wmma_rows64_scalar", []() {},
        [&](uint32_t set) {
            return ds4_rocm_bench_q4_K_wmma_enqueue(
                       rows64_device, output_b_weights[set], x_device,
                       n_tokens, 1u, kOutputLowDim, kOutputM, row_bytes,
                       kOutputLowDim, 0u, kOutputM, 64u, 0) != 0;
        }};
    const arm geometry_candidate = {
        "wmma_shape_scalar", []() {},
        [&](uint32_t set) {
            return ds4_rocm_bench_q4_K_wmma_enqueue(
                       shape_device, output_b_weights[set], x_device,
                       n_tokens, 1u, kOutputLowDim, kOutputM, row_bytes,
                       kOutputLowDim, 0u, kOutputM,
                       shape_wmma_row_tile(kOutputM), 0) != 0;
        }};
    if (!benchmark_arms(
        "output_b_wmma_rows", n_tokens, kOutputLowDim, kOutputM, cfg,
        geometry_baseline, geometry_candidate,
        [&]() {
            return poison_output(tile8.ptr, logical_bytes, 0x7fc10001u) &&
                   poison_output(wmma.ptr, logical_bytes, 0x7fc20002u);
        },
        [&]() {
            return bitwise_equal(tile8.ptr, wmma.ptr, logical_bytes,
                                 "output_b WMMA rows64 vs shape") &&
                   check_guard(tile8.ptr, logical_bytes,
                               "output_b WMMA rows64 oracle") &&
                   check_guard(wmma.ptr, logical_bytes,
                               "output_b WMMA shape oracle");
        })) return false;

    const uint32_t fixed_row_tile = shape_wmma_row_tile(kOutputM);
    const arm k32_baseline = {
        "wmma_k32_rows128_load2", []() {},
        [&](uint32_t set) {
            return ds4_rocm_bench_q4_K_wmma_variant_enqueue(
                       rows64_device, output_b_weights[set], x_device,
                       n_tokens, 1u, kOutputLowDim, kOutputM, row_bytes,
                       kOutputLowDim, 0u, kOutputM, fixed_row_tile,
                       32u, 1) != 0;
        }};
    const arm k64_candidate = {
        "wmma_k64p80_rows128_load2", []() {},
        [&](uint32_t set) {
            return ds4_rocm_bench_q4_K_wmma_variant_enqueue(
                       shape_device, output_b_weights[set], x_device,
                       n_tokens, 1u, kOutputLowDim, kOutputM, row_bytes,
                       kOutputLowDim, 0u, kOutputM, fixed_row_tile,
                       64u, 1) != 0;
        }};
    return benchmark_arms(
        "output_b_wmma_k32_k64", n_tokens, kOutputLowDim, kOutputM, cfg,
        k32_baseline, k64_candidate,
        [&]() {
            return poison_output(tile8.ptr, logical_bytes, 0x7fc10001u) &&
                   poison_output(wmma.ptr, logical_bytes, 0x7fc20002u);
        },
        [&]() {
            return bitwise_equal(tile8.ptr, wmma.ptr, logical_bytes,
                                 "output_b WMMA K32 vs K64/P80") &&
                   check_guard(tile8.ptr, logical_bytes,
                               "output_b WMMA K32 oracle") &&
                   check_guard(wmma.ptr, logical_bytes,
                               "output_b WMMA K64/P80 oracle");
        });
}

bool run_attention_output(const model_fixture &model, const config &cfg,
                          uint32_t n_tokens) {
    if (!cfg.wmma_supported || n_tokens < 256u) return true;
    uint64_t heads_elements = 0;
    uint64_t low_elements = 0;
    uint64_t out_elements = 0;
    if (!checked_mul(n_tokens,
                     static_cast<uint64_t>(kOutputGroups) * kDenseK,
                     &heads_elements) ||
        !checked_mul(n_tokens, kOutputLowDim, &low_elements) ||
        !checked_mul(n_tokens, kOutputM, &out_elements)) {
        return false;
    }
    const uint64_t low_bytes = low_elements * sizeof(float);
    const uint64_t out_bytes = out_elements * sizeof(float);
    const uint64_t guard_bytes = kGuardWords * sizeof(uint32_t);
    tensor_owner heads(heads_elements * sizeof(float));
    tensor_owner tile8_low(low_bytes + guard_bytes);
    tensor_owner tile8_out(out_bytes + guard_bytes);
    tensor_owner wmma_low(low_bytes + guard_bytes);
    tensor_owner wmma_out(out_bytes + guard_bytes);
    tensor_owner replay_out(out_bytes + guard_bytes);
    std::vector<float> activation;
    fill_activation(&activation, n_tokens, kOutputGroups * kDenseK);
    if (!heads.ptr || !tile8_low.ptr || !tile8_out.ptr || !wmma_low.ptr ||
        !wmma_out.ptr || !replay_out.ptr ||
        !ds4_gpu_tensor_write(heads.ptr, 0, activation.data(),
                              activation.size() * sizeof(float))) {
        std::fprintf(stderr,
                     "attention_output N=%u: tensor setup failed\n", n_tokens);
        return false;
    }

    auto poison_outputs = [&]() {
        return poison_output(tile8_low.ptr, low_bytes, 0x7fc10001u) &&
               poison_output(tile8_out.ptr, out_bytes, 0x7fc20002u) &&
               poison_output(wmma_low.ptr, low_bytes, 0x7fc30003u) &&
               poison_output(wmma_out.ptr, out_bytes, 0x7fc40004u);
    };
    auto poison_production = [&]() {
        return poison_outputs() &&
               poison_output(replay_out.ptr, out_bytes, 0x7fc50005u);
    };
    if (!poison_production()) return false;

    uint32_t production_candidate_set = 0u;

    const arm baseline = {
        "tile8_ab", []() { select_tile8(false); },
        [&](uint32_t set) {
            return ds4_gpu_attention_output_q4_K_batch_tensor(
                       tile8_out.ptr, tile8_low.ptr, nullptr, nullptr,
                       model.data, model.size,
                       model.weights[set].output_a_offset,
                       model.weights[set].output_b_offset, kQ4Type,
                       kDenseK, kOutputRank, kOutputGroups, kOutputM,
                       heads.ptr, n_tokens) > 0;
        }};
    const arm candidate = {
        "wmma_a_k64p80_b_tile8", select_wmma_attention_a_tile8_b,
        [&](uint32_t set) {
            production_candidate_set = set;
            return ds4_gpu_attention_output_q4_K_batch_tensor(
                       wmma_out.ptr, wmma_low.ptr, nullptr, nullptr,
                       model.data, model.size,
                       model.weights[set].output_a_offset,
                       model.weights[set].output_b_offset, kQ4Type,
                       kDenseK, kOutputRank, kOutputGroups, kOutputM,
                       heads.ptr, n_tokens) > 0;
        }};
    if (!benchmark_arms(
        "attention_output_a_wmma_b_tile8", n_tokens, kOutputLowDim,
        kDenseK + kOutputM, cfg, baseline, candidate, poison_production,
        [&]() {
            const uint64_t wmma_calls =
                ds4_rocm_test_q4_prefill_wmma_get_calls();
            const uint64_t k64_calls =
                ds4_rocm_test_q4_prefill_wmma_k64_get_calls();
            bool oracle_ok = wmma_calls == 1u && k64_calls == 1u;
            std::fprintf(stderr,
                         "output production dispatch: WMMA=%llu/1 "
                         "K64=%llu/1 %s\n",
                         (unsigned long long)wmma_calls,
                         (unsigned long long)k64_calls,
                         oracle_ok ? "PASS" : "FAIL");
            oracle_ok = numerically_close(
                wmma_low.ptr, tile8_low.ptr, low_bytes,
                "output_a grouped WMMA shape vs TILE8") && oracle_ok;

            /* A changes the values presented to B, so comparing the composed
             * candidate against an all-TILE8 run conflates A's deliberate F16
             * boundary with B correctness.  Keep the delta visible, but do
             * not make it the production-path gate. */
            oracle_ok = numerically_close(
                wmma_out.ptr, tile8_out.ptr, out_bytes,
                "output A-WMMA/B-TILE8 vs all-TILE8 (diagnostic only)",
                16.0f, 8.0e-2f, false) && oracle_ok;

            /* Hard B oracle: replay exact TILE8 with the very same WMMA-low
             * intermediate consumed by the production candidate. */
            select_tile8(false);
            const bool replay_ok =
                ds4_gpu_matmul_quant_tensor(
                    replay_out.ptr, model.data, model.size,
                    model.weights[production_candidate_set].output_b_offset,
                    kQ4Type, kOutputLowDim, kOutputM, wmma_low.ptr,
                    n_tokens) != 0 &&
                ds4_gpu_synchronize();
            if (!replay_ok) {
                std::fprintf(stderr,
                             "output_b same-low TILE8 replay dispatch FAIL\n");
                oracle_ok = false;
            } else {
                const uint64_t replay_wmma_calls =
                    ds4_rocm_test_q4_prefill_wmma_get_calls();
                const uint64_t replay_k64_calls =
                    ds4_rocm_test_q4_prefill_wmma_k64_get_calls();
                if (replay_wmma_calls != wmma_calls ||
                    replay_k64_calls != k64_calls) {
                    std::fprintf(stderr,
                                 "output_b same-low TILE8 replay used WMMA: "
                                 "WMMA=%llu/%llu K64=%llu/%llu FAIL\n",
                                 (unsigned long long)replay_wmma_calls,
                                 (unsigned long long)wmma_calls,
                                 (unsigned long long)replay_k64_calls,
                                 (unsigned long long)k64_calls);
                    oracle_ok = false;
                }
                oracle_ok = bitwise_equal(
                    wmma_out.ptr, replay_out.ptr, out_bytes,
                    "output_b production TILE8 vs same-low TILE8 replay") &&
                    oracle_ok;
            }
            oracle_ok = check_guard(
                tile8_low.ptr, low_bytes, "output_a TILE8 oracle") &&
                oracle_ok;
            oracle_ok = check_guard(
                wmma_low.ptr, low_bytes, "output_a WMMA shape oracle") &&
                oracle_ok;
            oracle_ok = check_guard(
                tile8_out.ptr, out_bytes, "output_b all-TILE8 oracle") &&
                oracle_ok;
            oracle_ok = check_guard(
                wmma_out.ptr, out_bytes,
                "output_b production A-WMMA/B-TILE8 oracle") && oracle_ok;
            oracle_ok = check_guard(
                replay_out.ptr, out_bytes,
                "output_b same-low TILE8 replay oracle") && oracle_ok;
            return oracle_ok;
        })) return false;

    void *const heads_device = ds4_gpu_tensor_contents(heads.ptr);
    void *const rows64_low_device = ds4_gpu_tensor_contents(tile8_low.ptr);
    void *const rows64_out_device = ds4_gpu_tensor_contents(tile8_out.ptr);
    void *const shape_low_device = ds4_gpu_tensor_contents(wmma_low.ptr);
    void *const shape_out_device = ds4_gpu_tensor_contents(wmma_out.ptr);
    std::vector<const void *> output_a_weights;
    std::vector<const void *> output_b_weights;
    if (!heads_device || !rows64_low_device || !rows64_out_device ||
        !shape_low_device || !shape_out_device ||
        !resolve_resident_weights(
            model, &weight_set::output_a_offset,
            q4_weight_bytes(kDenseK, kOutputLowDim), &output_a_weights) ||
        !resolve_resident_weights(
            model, &weight_set::output_b_offset,
            q4_weight_bytes(kOutputLowDim, kOutputM), &output_b_weights)) {
        std::fprintf(stderr,
                     "attention_output_ab_wmma_rows N=%u: direct setup failed\n",
                     n_tokens);
        return false;
    }
    const uint64_t row_a_bytes = q4_weight_bytes(kDenseK, 1u);
    const uint64_t row_b_bytes = q4_weight_bytes(kOutputLowDim, 1u);
    const arm geometry_baseline = {
        "wmma_rows64_ab_scalar", []() {},
        [&](uint32_t set) {
            return ds4_rocm_bench_q4_K_wmma_enqueue(
                       rows64_low_device, output_a_weights[set], heads_device,
                       n_tokens, kOutputGroups, kDenseK, kOutputRank,
                       row_a_bytes, kOutputGroups * kDenseK, kDenseK,
                       kOutputLowDim, 64u, 0) != 0 &&
                   ds4_rocm_bench_q4_K_wmma_enqueue(
                       rows64_out_device, output_b_weights[set],
                       rows64_low_device, n_tokens, 1u, kOutputLowDim,
                       kOutputM, row_b_bytes, kOutputLowDim, 0u, kOutputM,
                       64u, 0) != 0;
        }};
    const arm geometry_candidate = {
        "wmma_shape_ab_scalar", []() {},
        [&](uint32_t set) {
            return ds4_rocm_bench_q4_K_wmma_enqueue(
                       shape_low_device, output_a_weights[set], heads_device,
                       n_tokens, kOutputGroups, kDenseK, kOutputRank,
                       row_a_bytes, kOutputGroups * kDenseK, kDenseK,
                       kOutputLowDim,
                       shape_wmma_row_tile(kOutputRank), 0) != 0 &&
                   ds4_rocm_bench_q4_K_wmma_enqueue(
                       shape_out_device, output_b_weights[set],
                       shape_low_device, n_tokens, 1u, kOutputLowDim,
                       kOutputM, row_b_bytes, kOutputLowDim, 0u, kOutputM,
                       shape_wmma_row_tile(kOutputM), 0) != 0;
        }};
    if (!benchmark_arms(
        "attention_output_ab_wmma_rows", n_tokens, kOutputLowDim,
        kDenseK + kOutputM, cfg, geometry_baseline, geometry_candidate,
        poison_outputs,
        [&]() {
            return bitwise_equal(tile8_low.ptr, wmma_low.ptr, low_bytes,
                                 "output_a WMMA rows64 vs shape") &&
                   bitwise_equal(tile8_out.ptr, wmma_out.ptr, out_bytes,
                                 "output_a+b WMMA rows64 vs shape") &&
                   check_guard(tile8_low.ptr, low_bytes,
                               "output_a WMMA rows64 oracle") &&
                   check_guard(wmma_low.ptr, low_bytes,
                               "output_a WMMA shape oracle") &&
                   check_guard(tile8_out.ptr, out_bytes,
                               "output_b WMMA rows64 oracle") &&
                   check_guard(wmma_out.ptr, out_bytes,
                               "output_b WMMA shape oracle");
        })) return false;

    const uint32_t fixed_row_a_tile = shape_wmma_row_tile(kOutputRank);
    const uint32_t fixed_row_b_tile = shape_wmma_row_tile(kOutputM);
    const arm k32_baseline = {
        "wmma_k32_ab_rows128_load2", []() {},
        [&](uint32_t set) {
            return ds4_rocm_bench_q4_K_wmma_variant_enqueue(
                       rows64_low_device, output_a_weights[set], heads_device,
                       n_tokens, kOutputGroups, kDenseK, kOutputRank,
                       row_a_bytes, kOutputGroups * kDenseK, kDenseK,
                       kOutputLowDim, fixed_row_a_tile, 32u, 1) != 0 &&
                   ds4_rocm_bench_q4_K_wmma_variant_enqueue(
                       rows64_out_device, output_b_weights[set],
                       rows64_low_device, n_tokens, 1u, kOutputLowDim,
                       kOutputM, row_b_bytes, kOutputLowDim, 0u, kOutputM,
                       fixed_row_b_tile, 32u, 1) != 0;
        }};
    const arm k64_candidate = {
        "wmma_k64p80_ab_rows128_load2", []() {},
        [&](uint32_t set) {
            return ds4_rocm_bench_q4_K_wmma_variant_enqueue(
                       shape_low_device, output_a_weights[set], heads_device,
                       n_tokens, kOutputGroups, kDenseK, kOutputRank,
                       row_a_bytes, kOutputGroups * kDenseK, kDenseK,
                       kOutputLowDim, fixed_row_a_tile, 64u, 1) != 0 &&
                   ds4_rocm_bench_q4_K_wmma_variant_enqueue(
                       shape_out_device, output_b_weights[set],
                       shape_low_device, n_tokens, 1u, kOutputLowDim,
                       kOutputM, row_b_bytes, kOutputLowDim, 0u, kOutputM,
                       fixed_row_b_tile, 64u, 1) != 0;
        }};
    return benchmark_arms(
        "attention_output_ab_wmma_k32_k64", n_tokens, kOutputLowDim,
        kDenseK + kOutputM, cfg, k32_baseline, k64_candidate, poison_outputs,
        [&]() {
            return bitwise_equal(tile8_low.ptr, wmma_low.ptr, low_bytes,
                                 "output_a WMMA K32 vs K64/P80") &&
                   bitwise_equal(tile8_out.ptr, wmma_out.ptr, out_bytes,
                                 "output_a+b WMMA K32 vs K64/P80") &&
                   check_guard(tile8_low.ptr, low_bytes,
                               "output_a WMMA K32 oracle") &&
                   check_guard(wmma_low.ptr, low_bytes,
                               "output_a WMMA K64/P80 oracle") &&
                   check_guard(tile8_out.ptr, out_bytes,
                               "output_b WMMA K32 oracle") &&
                   check_guard(wmma_out.ptr, out_bytes,
                               "output_b WMMA K64/P80 oracle");
        });
}

void usage(FILE *stream, const char *argv0) {
    std::fprintf(
        stream,
        "usage: %s [options]\n\n"
        "Resident ROCm Q4_K prefill kernel A/B (HIP event timing only).\n\n"
        "  --case all|lds|lds_vector|wmma_load4|dense|pair|qb|outb|output\n"
        "                              comparison to run (default: all)\n"
        "  --tokens N[,N...]          token counts, each 9..4096\n"
        "  --full                     use 9,16,17,31,32,33,128,256,257,512,4096\n"
        "  --sets N                   rotating resident weight sets (default: %u)\n"
        "  --samples N                samples/arm, multiple of 4 (default: %u)\n"
        "  --warmup N                 untimed dispatches/arm (default: %u)\n"
        "  -h, --help                 show this help\n\n"
        "lds isolates the Q4 LDS loader/final-fence rollback at K4096 and\n"
        "Q-b K1024, with bitwise checks, launch evidence and identical tiling.\n"
        "lds_vector isolates four-word vs scalar LDS copies with the same\n"
        "streaming fence schedule; lds keeps the previous scalar-only A/B.\n"
        "wmma_load4 isolates K64 float2/float4 loads at identical geometry\n"
        "for dense, grouped output-A, output-B and q_b (N>=256, gfx1151).\n"
        "The q_b comparison forces K64 on both arms, not production K128.\n"
        "dense compares legacy/TILE8, times the raw canonical/wave32 Q8_K\n"
        "quantizer kernels on gfx1151, and TILE8/direct WMMA there for N>=256\n"
        "at K=4096,M=1024. pair compares\n"
        "two TILE8 projections with\n"
        "the fused K=4096,M=(1024+512) path. qb adds TILE4/direct WMMA at the\n"
        "K=1024,M=32768 shape. outb isolates K=8192,M=4096; output measures\n"
        "the production grouped output_a plus output_b API as all-TILE8 vs\n"
        "A-WMMA/B-TILE8. Its hard B oracle replays TILE8 on the same low; the\n"
        "composed all-TILE8 delta is diagnostic only. Other WMMA comparisons\n"
        "use a finite/toleranced oracle because their F16 boundary is not\n"
        "bit-identical to the Q8_K activation path. At N>=256 the raw direct\n"
        "arms also compare K32 with K64/P80 at fixed production geometry and\n"
        "load2; q_b additionally compares K64/P80 with the default K128/P144\n"
        "load4 stage. Both staged-kernel checks require bitwise output.\n",
        argv0, kDefaultSets, kDefaultSamples, kDefaultWarmup);
}

uint32_t parse_u32(const char *text, const char *option, uint32_t minimum,
                   uint32_t maximum) {
    char *end = nullptr;
    errno = 0;
    const unsigned long value = std::strtoul(text, &end, 10);
    if (errno != 0 || !text[0] || !end || *end || value < minimum ||
        value > maximum) {
        std::fprintf(stderr, "invalid %s: %s\n", option, text);
        std::exit(2);
    }
    return static_cast<uint32_t>(value);
}

const char *need_value(int *index, int argc, char **argv) {
    if (*index + 1 >= argc) {
        std::fprintf(stderr, "%s requires a value\n", argv[*index]);
        std::exit(2);
    }
    return argv[++*index];
}

std::vector<uint32_t> parse_tokens(const char *text) {
    std::vector<uint32_t> result;
    const char *cursor = text;
    while (*cursor) {
        const char *comma = std::strchr(cursor, ',');
        const std::string item(cursor,
                               comma ? static_cast<size_t>(comma - cursor)
                                     : std::strlen(cursor));
        result.push_back(parse_u32(item.c_str(), "--tokens", 9u, 4096u));
        if (!comma) break;
        cursor = comma + 1;
        if (!*cursor) {
            std::fprintf(stderr, "invalid --tokens: trailing comma\n");
            std::exit(2);
        }
    }
    if (result.empty()) {
        std::fprintf(stderr, "--tokens cannot be empty\n");
        std::exit(2);
    }
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
}

config parse_options(int argc, char **argv) {
    config cfg;
    for (int i = 1; i < argc; i++) {
        if (!std::strcmp(argv[i], "-h") || !std::strcmp(argv[i], "--help")) {
            usage(stdout, argv[0]);
            std::exit(0);
        } else if (!std::strcmp(argv[i], "--case")) {
            const char *value = need_value(&i, argc, argv);
            if (!std::strcmp(value, "all")) cfg.selected = bench_case::all;
            else if (!std::strcmp(value, "lds")) cfg.selected = bench_case::lds;
            else if (!std::strcmp(value, "lds_vector")) cfg.selected = bench_case::lds_vector;
            else if (!std::strcmp(value, "wmma_load4")) cfg.selected = bench_case::wmma_load4;
            else if (!std::strcmp(value, "dense")) cfg.selected = bench_case::dense;
            else if (!std::strcmp(value, "pair")) cfg.selected = bench_case::pair;
            else if (!std::strcmp(value, "qb")) cfg.selected = bench_case::qb;
            else if (!std::strcmp(value, "outb")) cfg.selected = bench_case::outb;
            else if (!std::strcmp(value, "output")) cfg.selected = bench_case::output;
            else {
                std::fprintf(stderr, "invalid --case: %s\n", value);
                std::exit(2);
            }
        } else if (!std::strcmp(argv[i], "--tokens")) {
            cfg.tokens = parse_tokens(need_value(&i, argc, argv));
        } else if (!std::strcmp(argv[i], "--full")) {
            cfg.tokens = {
                9u, 16u, 17u, 31u, 32u, 33u, 128u, 256u, 257u, 512u,
                4096u};
        } else if (!std::strcmp(argv[i], "--sets")) {
            cfg.sets = parse_u32(need_value(&i, argc, argv), "--sets", 1u, 32u);
        } else if (!std::strcmp(argv[i], "--samples")) {
            cfg.samples =
                parse_u32(need_value(&i, argc, argv), "--samples", 4u, 1000u);
        } else if (!std::strcmp(argv[i], "--warmup")) {
            cfg.warmup =
                parse_u32(need_value(&i, argc, argv), "--warmup", 0u, 100u);
        } else {
            std::fprintf(stderr, "unknown option: %s\n", argv[i]);
            usage(stderr, argv[0]);
            std::exit(2);
        }
    }
    if ((cfg.samples % 4u) != 0u) {
        std::fprintf(stderr,
                     "--samples must be a multiple of 4 for ABBA/BAAB balance\n");
        std::exit(2);
    }
    return cfg;
}

bool includes(bench_case selected, bench_case wanted) {
    return selected == bench_case::all || selected == wanted;
}

}  // namespace

int main(int argc, char **argv) {
    config cfg = parse_options(argc, argv);
    env_snapshot enable_guard(kPrefillEnable);
    env_snapshot disable_guard(kPrefillDisable);
    env_snapshot require_guard(kPrefillRequire);
    env_snapshot tile4_guard(kK1024Tile4Disable);
    env_snapshot tile4_ssd_guard(kK1024Tile4SsdEnable);
    env_snapshot tile4_require_guard(kK1024Tile4Require);
    env_snapshot wmma_enable_guard(kWmmaEnable);
    env_snapshot wmma_ssd_enable_guard(kWmmaSsdEnable);
    env_snapshot wmma_disable_guard(kWmmaDisable);
    env_snapshot wmma_require_guard(kWmmaRequire);
    env_snapshot wmma_row_tile_guard(kWmmaRowTile);
    env_snapshot wmma_k64_guard(kWmmaK64);
    env_snapshot wmma_k128_disable_guard(kWmmaK128Disable);
    env_snapshot q8_wave32_enable_guard(kQ8Wave32Enable);
    env_snapshot q8_wave32_disable_guard(kQ8Wave32Disable);
    env_snapshot q8_wave32_require_guard(kQ8Wave32Require);
    (void)unsetenv(kQ8Wave32Enable);
    (void)unsetenv(kQ8Wave32Disable);
    (void)unsetenv(kQ8Wave32Require);
    (void)unsetenv(kWmmaK128Disable);

    int device_count = 0;
    hipError_t hip_rc = hipGetDeviceCount(&device_count);
    if (hip_rc != hipSuccess || device_count <= 0) {
        std::fprintf(stderr,
                     "rocm-q4-prefill-bench: no visible HIP device (%s)\n",
                     hip_rc == hipSuccess ? "device count is zero"
                                          : hipGetErrorString(hip_rc));
        return 77;
    }
    hipDeviceProp_t properties{};
    if (hipGetDeviceProperties(&properties, 0) != hipSuccess) {
        std::fprintf(stderr,
                     "rocm-q4-prefill-bench: cannot query device properties\n");
        return 1;
    }
    cfg.wmma_supported = properties.warpSize == 32 &&
        std::strncmp(properties.gcnArchName, "gfx1151", 7u) == 0;
    const bool wmma_only_case =
        cfg.selected == bench_case::outb || cfg.selected == bench_case::output ||
        cfg.selected == bench_case::wmma_load4;
    const bool has_wmma_tokens = std::any_of(
        cfg.tokens.begin(), cfg.tokens.end(),
        [](uint32_t n_tokens) { return n_tokens >= 256u; });
    if (wmma_only_case && (!cfg.wmma_supported || !has_wmma_tokens)) {
        std::fprintf(stderr,
                     "rocm-q4-prefill-bench: SKIP (%s requires gfx1151 "
                     "wave32 and at least one N>=256 sample)\n",
                     cfg.selected == bench_case::outb ? "outb" :
                     (cfg.selected == bench_case::output ? "output" : "wmma_load4"));
        return 77;
    }
    if (!ds4_gpu_init()) {
        std::fprintf(stderr, "rocm-q4-prefill-bench: ds4_gpu_init failed\n");
        return 1;
    }

    bool ok = true;
    model_fixture model;
    if (!make_model(&model, cfg.sets)) {
        std::fprintf(stderr, "rocm-q4-prefill-bench: model fixture allocation failed\n");
        ok = false;
    }
    if (ok) {
        ds4_gpu_set_ssd_streaming(false);
        const uint64_t max_tensor = std::max(
            q4_weight_bytes(kOutputLowDim, kOutputM),
            q4_weight_bytes(kDenseK, kOutputLowDim));
        if (!ds4_gpu_set_model_fd(fileno(model.file)) ||
            !ds4_gpu_set_model_map_spans(
                model.data, model.size, model.span_offsets.data(),
                model.span_sizes.data(),
                static_cast<uint32_t>(model.span_offsets.size()), max_tensor) ||
            !ds4_gpu_synchronize()) {
            std::fprintf(stderr,
                         "rocm-q4-prefill-bench: device-resident weight copy failed\n");
            ok = false;
        }
    }

    if (ok) {
        std::printf(
            "DS4_ROCM_Q4_PREFILL_SETUP device=%s arch=%s warp=%d sets=%u "
            "resident_mib=%.2f timing=hip_events ssd_streaming=off "
            "wmma_rowtiles=%s wmma_loaders=%s wmma_k_stages=%s "
            "wmma_k_default=%s q8_wave32=%s\n",
            properties.name, properties.gcnArchName, properties.warpSize,
            cfg.sets, static_cast<double>(model.resident_bytes) / 1048576.0,
            cfg.wmma_supported ? "64,128,256" : "skipped",
            cfg.wmma_supported ? "scalar,load2,load4" : "skipped",
            cfg.wmma_supported ? "32,64p80,128p144" : "skipped",
            cfg.wmma_supported ? "128p144@rows256,64p80@rows64/128"
                               : "skipped",
            cfg.wmma_supported ? "available" : "skipped");
        std::fflush(stdout);
        for (uint32_t n_tokens : cfg.tokens) {
            if (includes(cfg.selected, bench_case::wmma_load4)) {
                ok = run_wmma_load4(model, cfg, n_tokens, kDenseK, kDenseM, 1u,
                                   &weight_set::dense_offset, "dense_k64_load4") &&
                     run_wmma_load4(model, cfg, n_tokens, kDenseK, kOutputRank,
                                   kOutputGroups, &weight_set::output_a_offset,
                                   "output_a_k64_load4") &&
                     run_wmma_load4(model, cfg, n_tokens, kOutputLowDim, kOutputM,
                                   1u, &weight_set::output_b_offset, "output_b_k64_load4") &&
                     run_wmma_load4(model, cfg, n_tokens, kQbK, kQbM, 1u,
                                   &weight_set::qb_offset, "q_b_k64_load4") && ok;
                if (!ok) break;
            }
            if (includes(cfg.selected, bench_case::lds)) {
                ok = run_lds(model, cfg, n_tokens, false) &&
                     run_lds(model, cfg, n_tokens, true) && ok;
                if (!ok) break;
            }
            if (includes(cfg.selected, bench_case::lds_vector)) {
                ok = run_lds(model, cfg, n_tokens, false, true) &&
                     run_lds(model, cfg, n_tokens, true, true) && ok;
                if (!ok) break;
            }
            if (includes(cfg.selected, bench_case::dense)) {
                ok = run_dense(model, cfg, n_tokens) && ok;
            }
            if (ok && includes(cfg.selected, bench_case::pair)) {
                ok = run_pair(model, cfg, n_tokens) && ok;
            }
            if (ok && includes(cfg.selected, bench_case::qb)) {
                ok = run_qb(model, cfg, n_tokens) && ok;
            }
            if (ok && includes(cfg.selected, bench_case::outb)) {
                ok = run_output_b(model, cfg, n_tokens) && ok;
            }
            if (ok && includes(cfg.selected, bench_case::output)) {
                ok = run_attention_output(model, cfg, n_tokens) && ok;
            }
            if (!ok) break;
        }
    }

    (void)ds4_gpu_set_model_fd(-1);
    ds4_gpu_cleanup();
    std::fprintf(stderr, "rocm-q4-prefill-bench: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
