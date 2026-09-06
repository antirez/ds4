// SPDX-License-Identifier: MIT
// Resident, CUDA-event-only Q4_K prefill microbenchmark.

#include "ds4_gpu.h"
#include "cuda/mmq/ds4_mmq.h"
#include "cuda/mmq/ds4_mmq_q4_16warp.cuh"

#include <cuda_runtime.h>

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
#include <utility>
#include <vector>

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
constexpr uint32_t kOutputMinB = 256u;
constexpr uint32_t kOutputM = 4096u;
constexpr uint32_t kDefaultSets = 4u;
constexpr uint32_t kDefaultSamples = 8u;
constexpr uint32_t kDefaultWarmup = 2u;
constexpr uint32_t kGuardWords = 64u;
constexpr uint64_t kCompareChunk = 4u * 1024u * 1024u;

constexpr const char *kGroupedPrefillEnable =
    "DS4_CUDA_ENABLE_Q4_GROUPED_ATTN_A_PREFILL";
constexpr const char *kGroupedPrefillDisable =
    "DS4_CUDA_NO_Q4_GROUPED_ATTN_A_PREFILL";
constexpr const char *kGroupedPrefillRequire =
    "DS4_CUDA_REQUIRE_Q4_GROUPED_ATTN_A_PREFILL";
constexpr const char *kGroupedSingleGridEnable =
    "DS4_CUDA_ENABLE_Q4_GROUPED_ATTN_A_SINGLE_GRID";
constexpr const char *kGroupedSingleGridDisable =
    "DS4_CUDA_DISABLE_Q4_GROUPED_ATTN_A_SINGLE_GRID";
constexpr const char *kGroupedSingleGridRequire =
    "DS4_CUDA_REQUIRE_Q4_GROUPED_ATTN_A_SINGLE_GRID";
constexpr const char *kGroupedQ81Disable =
    "DS4_CUDA_NO_Q4_GROUPED_ATTN_A_Q81";
constexpr const char *kGroupedQ81Require =
    "DS4_CUDA_REQUIRE_Q4_GROUPED_ATTN_A_Q81";
constexpr const char *kGroupedGlobalDisable =
    "DS4_CUDA_NO_Q4_GROUPED_ATTN_A";
constexpr const char *kGb10GlobalDisable = "DS4_CUDA_NO_Q4_GB10_FAST";

struct block_q4_K_host {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[12];
    uint8_t qs[kQkK / 2u];
};

static_assert(sizeof(block_q4_K_host) == 144u,
              "Q4_K fixture must match the raw GGUF layout");

enum class bench_case {
    all,
    dense,
    pair,
    qb,
    outa,
    outb,
};

enum class cuda_path {
    mmq,
    legacy,
};

struct config {
    bench_case selected = bench_case::all;
    cuda_path path = cuda_path::mmq;
    std::vector<uint32_t> tokens = {9u, 17u, 33u, 127u, 128u, 129u, 257u,
                                    512u};
    uint32_t sets = kDefaultSets;
    uint32_t samples = kDefaultSamples;
    uint32_t warmup = kDefaultWarmup;
    bool kernel_16warp = false;
    bool grouped_single_grid = false;
    bool grouped_q81_kernel = false;
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
    uint64_t payload_bytes = 0;
    uint32_t output_b_rows = kOutputMinB;
    std::vector<weight_set> weights;

    ~model_fixture() { std::free(data); }
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

struct cuda_buffer {
    void *ptr = nullptr;
    cudaError_t status = cudaSuccess;

    explicit cuda_buffer(size_t bytes) {
        if (bytes == 0u) return;
        status = cudaMalloc(&ptr, bytes);
    }
    ~cuda_buffer() {
        if (ptr) (void)cudaFree(ptr);
    }
    cuda_buffer(const cuda_buffer &) = delete;
    cuda_buffer &operator=(const cuda_buffer &) = delete;
};

struct guarded_cuda_buffer {
    void *storage = nullptr;
    void *ptr = nullptr;
    size_t logical_bytes = 0;
    uint32_t salt = 0;
    cudaError_t status = cudaSuccess;

    guarded_cuda_buffer(size_t bytes, uint32_t guard_salt)
        : logical_bytes(bytes), salt(guard_salt) {
        if (bytes == 0u) return;
        constexpr size_t guard_bytes = kGuardWords * sizeof(uint32_t);
        if (bytes > std::numeric_limits<size_t>::max() - 2u * guard_bytes) {
            status = static_cast<cudaError_t>(1);
            return;
        }
        status = cudaMalloc(&storage, bytes + 2u * guard_bytes);
        if (status != cudaSuccess) return;
        ptr = static_cast<uint8_t *>(storage) + guard_bytes;
        std::vector<uint32_t> prefix(kGuardWords);
        std::vector<uint32_t> suffix(kGuardWords);
        for (uint32_t i = 0; i < kGuardWords; ++i) {
            prefix[i] = 0x6b8b4567u ^ salt ^ (i * 0x00010101u);
            suffix[i] = 0x327b23c6u ^ salt ^ (i * 0x01000101u);
        }
        status = cudaMemcpy(storage, prefix.data(), guard_bytes,
                            cudaMemcpyHostToDevice);
        if (status == cudaSuccess) {
            status = cudaMemcpy(
                static_cast<uint8_t *>(ptr) + bytes, suffix.data(),
                guard_bytes, cudaMemcpyHostToDevice);
        }
    }

    ~guarded_cuda_buffer() {
        if (storage) (void)cudaFree(storage);
    }

    bool intact(const char *label) const {
        if (logical_bytes == 0u) return true;
        constexpr size_t guard_bytes = kGuardWords * sizeof(uint32_t);
        std::vector<uint32_t> prefix(kGuardWords);
        std::vector<uint32_t> suffix(kGuardWords);
        if (cudaMemcpy(prefix.data(), storage, guard_bytes,
                       cudaMemcpyDeviceToHost) != cudaSuccess ||
            cudaMemcpy(suffix.data(),
                       static_cast<const uint8_t *>(ptr) + logical_bytes,
                       guard_bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
            std::fprintf(stderr, "%s: scratch guard read failed\n", label);
            return false;
        }
        for (uint32_t i = 0; i < kGuardWords; ++i) {
            const uint32_t expected_prefix =
                0x6b8b4567u ^ salt ^ (i * 0x00010101u);
            const uint32_t expected_suffix =
                0x327b23c6u ^ salt ^ (i * 0x01000101u);
            if (prefix[i] != expected_prefix || suffix[i] != expected_suffix) {
                std::fprintf(stderr,
                             "%s: scratch guard overwritten at word %u\n",
                             label, i);
                return false;
            }
        }
        return true;
    }

    guarded_cuda_buffer(const guarded_cuda_buffer &) = delete;
    guarded_cuda_buffer &operator=(const guarded_cuda_buffer &) = delete;
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
    cudaEvent_t begin = nullptr;
    cudaEvent_t end = nullptr;

    event_timer() {
        if (cudaEventCreate(&begin) != cudaSuccess ||
            cudaEventCreate(&end) != cudaSuccess) {
            std::fprintf(stderr,
                         "cuda-q4-prefill-bench: CUDA event allocation failed\n");
            std::exit(1);
        }
    }
    ~event_timer() {
        if (begin) (void)cudaEventDestroy(begin);
        if (end) (void)cudaEventDestroy(end);
    }

    bool measure(const std::function<bool()> &dispatch, float *milliseconds) {
        // Every harness arm uses stream 0.  Production paths enqueue their
        // complete backend work there; kernel-only arms enqueue exactly one
        // already-prepared GEMM there.
        if (cudaEventRecord(begin, nullptr) != cudaSuccess) return false;
        if (!dispatch()) return false;
        if (cudaEventRecord(end, nullptr) != cudaSuccess ||
            cudaEventSynchronize(end) != cudaSuccess ||
            cudaEventElapsedTime(milliseconds, begin, end) != cudaSuccess) {
            return false;
        }
        return true;
    }
};

struct arm {
    const char *name;
    std::function<bool(uint32_t)> dispatch;
    // Host-only path selection must happen before the start event.  Otherwise
    // an idle device can execute that event while setenv()/unsetenv() is still
    // running, folding host-side gate switching into the reported GPU time.
    std::function<bool()> select;
    // Optional checked boundary for oracle passes. Timed samples continue to
    // use dispatch so an enqueue-only experimental arm can keep its safety
    // preflight outside the CUDA-event interval.
    std::function<bool(uint32_t)> oracle_dispatch = {};
};

struct stats {
    double minimum = 0.0;
    double median = 0.0;
    double p95 = 0.0;
    double mean = 0.0;
};

uint64_t align_up(uint64_t value, uint64_t alignment) {
    return (value + alignment - 1u) / alignment * alignment;
}

bool checked_mul(uint64_t a, uint64_t b, uint64_t *out) {
    if (a != 0u && b > std::numeric_limits<uint64_t>::max() / a) return false;
    *out = a * b;
    return true;
}

bool current_device_nsm(int *nsm) {
    if (!nsm) return false;
    int device = -1;
    cudaDeviceProp prop = {};
    const cudaError_t device_err = cudaGetDevice(&device);
    const cudaError_t prop_err = device_err == cudaSuccess
        ? cudaGetDeviceProperties(&prop, device) : device_err;
    if (prop_err != cudaSuccess || prop.multiProcessorCount <= 0) {
        std::fprintf(stderr,
                     "cuda-q4-prefill-bench: CUDA geometry query failed: %s\n",
                     cudaGetErrorString(prop_err));
        return false;
    }
    *nsm = prop.multiProcessorCount;
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
        // Positive, finite FP16 scales.  Payload values are deterministic.
        blocks[i].d = static_cast<uint16_t>(0x2400u +
                                            (lcg_next(&state) & 0xffu));
        blocks[i].dmin = static_cast<uint16_t>(0x2000u +
                                               (lcg_next(&state) & 0xffu));
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

bool make_model(model_fixture *model, uint32_t sets,
                uint32_t output_b_rows) {
    if (output_b_rows < kOutputMinB) return false;
    constexpr uint64_t page = 4096u;
    const uint64_t dense_bytes = q4_weight_bytes(kDenseK, kDenseM);
    const uint64_t kv_bytes = q4_weight_bytes(kDenseK, kKvM);
    const uint64_t qb_bytes = q4_weight_bytes(kQbK, kQbM);
    const uint64_t output_a_bytes =
        q4_weight_bytes(kDenseK, kOutputLowDim);
    const uint64_t output_b_bytes =
        q4_weight_bytes(kOutputLowDim, output_b_rows);

    model->output_b_rows = output_b_rows;
    model->weights.resize(sets);
    uint64_t cursor = 0;
    auto append = [&](uint64_t bytes) {
        const uint64_t offset = align_up(cursor, page);
        cursor = offset + bytes;
        model->payload_bytes += bytes;
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
    model->data = static_cast<uint8_t *>(storage);
    std::memset(model->data, 0, static_cast<size_t>(model->size));
    for (uint32_t i = 0; i < sets; i++) {
        fill_q4(model->data + model->weights[i].dense_offset, dense_bytes,
                0x243f6a88u ^ (i * 0x9e3779b9u));
        fill_q4(model->data + model->weights[i].kv_offset, kv_bytes,
                0x85a308d3u ^ (i * 0x7f4a7c15u));
        fill_q4(model->data + model->weights[i].qb_offset, qb_bytes,
                0x13198a2eu ^ (i * 0x94d049bbu));
        fill_q4(model->data + model->weights[i].output_a_offset,
                output_a_bytes, 0xa4093822u ^ (i * 0x2545f491u));
        fill_q4(model->data + model->weights[i].output_b_offset,
                output_b_bytes, 0x299f31d0u ^ (i * 0x369dea0fu));
    }
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
                dst[i] = static_cast<float>(q) / 32.0f;
            }
            dst[0] = ((token + block) & 1u) ? 127.0f / 32.0f
                                             : -127.0f / 32.0f;
        }
    }
}

std::vector<uint32_t> guard_pattern(uint32_t salt) {
    std::vector<uint32_t> guard(kGuardWords);
    for (uint32_t i = 0; i < kGuardWords; i++) {
        guard[i] = 0x7fc12000u ^ salt ^ (i * 0x00010101u);
    }
    return guard;
}

bool prepare_guard(ds4_gpu_tensor *tensor, uint64_t logical_bytes,
                   uint32_t salt) {
    const std::vector<uint32_t> guard = guard_pattern(salt);
    return ds4_gpu_tensor_write(tensor, logical_bytes, guard.data(),
                                guard.size() * sizeof(guard[0])) != 0;
}

bool poison_output(ds4_gpu_tensor *tensor, uint64_t logical_bytes,
                   uint32_t pattern, uint32_t guard_salt) {
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
    return prepare_guard(tensor, logical_bytes, guard_salt);
}

bool check_guard(const ds4_gpu_tensor *tensor, uint64_t logical_bytes,
                 uint32_t salt, const char *label) {
    const std::vector<uint32_t> expected = guard_pattern(salt);
    std::vector<uint32_t> got(expected.size());
    if (!ds4_gpu_tensor_read(tensor, logical_bytes, got.data(),
                             got.size() * sizeof(got[0]))) {
        std::fprintf(stderr, "%s: guard read failed\n", label);
        return false;
    }
    if (got != expected) {
        const auto mismatch = std::mismatch(got.begin(), got.end(),
                                            expected.begin());
        std::fprintf(stderr, "%s: guard overwritten at word %zu\n", label,
                     static_cast<size_t>(mismatch.first - got.begin()));
        return false;
    }
    return true;
}

bool output_is_finite(const ds4_gpu_tensor *tensor, uint64_t logical_bytes,
                      const char *label) {
    const uint64_t chunk_bytes = std::min(kCompareChunk, logical_bytes);
    std::vector<float> values(
        static_cast<size_t>(chunk_bytes / sizeof(float)));
    for (uint64_t offset = 0; offset < logical_bytes; offset += chunk_bytes) {
        const uint64_t count = std::min(chunk_bytes, logical_bytes - offset);
        if (!ds4_gpu_tensor_read(tensor, offset, values.data(), count)) {
            std::fprintf(stderr, "%s: output read failed\n", label);
            return false;
        }
        const size_t n = static_cast<size_t>(count / sizeof(float));
        for (size_t i = 0; i < n; i++) {
            if (!std::isfinite(values[i])) {
                std::fprintf(stderr,
                             "%s: non-finite/unwritten output at element %llu\n",
                             label,
                             static_cast<unsigned long long>(
                                 offset / sizeof(float) + i));
                return false;
            }
        }
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
        if (std::memcmp(lhs.data(), rhs.data(), static_cast<size_t>(count)) !=
            0) {
            uint64_t first = 0;
            while (first < count && lhs[static_cast<size_t>(first)] ==
                                        rhs[static_cast<size_t>(first)]) {
                first++;
            }
            std::fprintf(stderr, "%s: bitwise mismatch at output byte %llu\n",
                         label,
                         static_cast<unsigned long long>(offset + first));
            return false;
        }
    }
    return true;
}

float fp16_to_float(uint16_t h) {
    const float sign = (h & 0x8000u) ? -1.0f : 1.0f;
    const uint32_t exponent = (h >> 10u) & 0x1fu;
    const uint32_t mantissa = h & 0x3ffu;
    if (exponent == 0u) {
        return mantissa == 0u
            ? std::copysign(0.0f, sign)
            : sign * std::ldexp(static_cast<float>(mantissa), -24);
    }
    if (exponent == 31u) {
        return mantissa == 0u
            ? std::copysign(std::numeric_limits<float>::infinity(), sign)
            : std::numeric_limits<float>::quiet_NaN();
    }
    return sign * std::ldexp(1.0f + static_cast<float>(mantissa) / 1024.0f,
                             static_cast<int>(exponent) - 15);
}

void get_scale_min_k4(uint32_t index, const uint8_t *packed,
                      uint8_t *scale, uint8_t *minimum) {
    if (index < 4u) {
        *scale = packed[index] & 63u;
        *minimum = packed[index + 4u] & 63u;
    } else {
        *scale = static_cast<uint8_t>((packed[index + 4u] & 0x0fu) |
                                      ((packed[index - 4u] >> 6u) << 4u));
        *minimum = static_cast<uint8_t>((packed[index + 4u] >> 4u) |
                                        ((packed[index] >> 6u) << 4u));
    }
}

void dequantize_q4_row(const block_q4_K_host *blocks, uint32_t in_dim,
                       std::vector<float> *row) {
    row->resize(in_dim);
    float *dst = row->data();
    const uint32_t n_blocks = in_dim / kQkK;
    for (uint32_t block = 0; block < n_blocks; block++) {
        const float d = fp16_to_float(blocks[block].d);
        const float dmin = fp16_to_float(blocks[block].dmin);
        const uint8_t *q = blocks[block].qs;
        uint32_t scale_index = 0;
        for (uint32_t group = 0; group < kQkK; group += 64u) {
            uint8_t sc0 = 0, min0 = 0, sc1 = 0, min1 = 0;
            get_scale_min_k4(scale_index, blocks[block].scales,
                             &sc0, &min0);
            get_scale_min_k4(scale_index + 1u, blocks[block].scales,
                             &sc1, &min1);
            const float ds0 = d * sc0;
            const float dm0 = dmin * min0;
            const float ds1 = d * sc1;
            const float dm1 = dmin * min1;
            for (uint32_t i = 0; i < 32u; i++) {
                *dst++ = ds0 * static_cast<float>(q[i] & 0x0fu) - dm0;
            }
            for (uint32_t i = 0; i < 32u; i++) {
                *dst++ = ds1 * static_cast<float>(q[i] >> 4u) - dm1;
            }
            q += 32u;
            scale_index += 2u;
            (void)group;
        }
    }
}

std::vector<uint32_t> sample_indices(uint32_t extent) {
    std::vector<uint32_t> values = {
        0u, extent / 3u, extent / 2u, extent - 1u,
    };
    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end()), values.end());
    return values;
}

bool sampled_cpu_oracle(const ds4_gpu_tensor *output,
                        const model_fixture &model, uint64_t weight_offset,
                        const std::vector<float> &activation,
                        uint32_t n_tokens, uint32_t in_dim,
                        uint32_t out_dim, const char *label) {
    const std::vector<uint32_t> tokens = sample_indices(n_tokens);
    const std::vector<uint32_t> rows = sample_indices(out_dim);
    const uint64_t blocks_per_row = in_dim / kQkK;
    const uint64_t row_bytes = blocks_per_row * sizeof(block_q4_K_host);
    std::vector<float> weight_row;
    // cuda/mmq/test/test_mmq_parity.cu validates Q4_K MMQ against this same
    // dequantized-weight x original-F32 reference at 0.20*sqrt(K) absolute
    // and 5% relative error.  Use a small shared envelope for both that Q8_1
    // path and the legacy 256-value Q8_K activation quantizer.  Every fixture
    // block pins |x|max to 127/32, keeping the Q8_K step bounded and stable.
    const double abs_tol = 0.25 * std::sqrt(static_cast<double>(in_dim));
    constexpr double rel_tol = 0.06;

    for (uint32_t row : rows) {
        const auto *blocks = reinterpret_cast<const block_q4_K_host *>(
            model.data + weight_offset + static_cast<uint64_t>(row) * row_bytes);
        dequantize_q4_row(blocks, in_dim, &weight_row);
        for (uint32_t token : tokens) {
            const float *x = activation.data() +
                             static_cast<uint64_t>(token) * in_dim;
            float reference = 0.0f;
            for (uint32_t k = 0; k < in_dim; k++) {
                reference += weight_row[k] * x[k];
            }
            float got = 0.0f;
            const uint64_t element = static_cast<uint64_t>(token) * out_dim + row;
            if (!ds4_gpu_tensor_read(output, element * sizeof(float), &got,
                                     sizeof(got))) {
                std::fprintf(stderr, "%s: sampled output read failed\n", label);
                return false;
            }
            const double abs_error = std::fabs(static_cast<double>(got) -
                                               reference);
            const double rel_error = reference != 0.0f
                ? abs_error / std::fabs(static_cast<double>(reference))
                : (abs_error == 0.0 ? 0.0
                                    : std::numeric_limits<double>::infinity());
            if (!std::isfinite(got) ||
                (abs_error > abs_tol && rel_error > rel_tol)) {
                std::fprintf(
                    stderr,
                    "%s: CPU oracle mismatch token=%u row=%u got=%.7g "
                    "reference=%.7g abs=%.5g rel=%.5g limits=%.5g/%.3g\n",
                    label, token, row, got, reference, abs_error, rel_error,
                    abs_tol, rel_tol);
                return false;
            }
        }
    }
    return true;
}

bool sampled_grouped_cpu_oracle(
        const ds4_gpu_tensor *output, const model_fixture &model,
        uint64_t weight_offset, const std::vector<float> &activation,
        uint32_t n_tokens, const char *label) {
    const std::vector<uint32_t> tokens = sample_indices(n_tokens);
    const std::vector<uint32_t> rows = sample_indices(kOutputRank);
    const uint64_t blocks_per_row = kDenseK / kQkK;
    const uint64_t row_bytes = blocks_per_row * sizeof(block_q4_K_host);
    std::vector<float> weight_row;
    const double abs_tol = 0.25 * std::sqrt(static_cast<double>(kDenseK));
    constexpr double rel_tol = 0.06;

    for (uint32_t group = 0; group < kOutputGroups; group++) {
        for (uint32_t row : rows) {
            const uint64_t weight_row_index =
                static_cast<uint64_t>(group) * kOutputRank + row;
            const auto *blocks =
                reinterpret_cast<const block_q4_K_host *>(
                    model.data + weight_offset + weight_row_index * row_bytes);
            dequantize_q4_row(blocks, kDenseK, &weight_row);
            for (uint32_t token : tokens) {
                const float *x = activation.data() +
                    (static_cast<uint64_t>(token) * kOutputGroups + group) *
                        kDenseK;
                float reference = 0.0f;
                for (uint32_t k = 0; k < kDenseK; k++) {
                    reference += weight_row[k] * x[k];
                }
                const uint64_t element =
                    static_cast<uint64_t>(token) * kOutputLowDim +
                    static_cast<uint64_t>(group) * kOutputRank + row;
                float got = 0.0f;
                if (!ds4_gpu_tensor_read(output, element * sizeof(float),
                                         &got, sizeof(got))) {
                    std::fprintf(stderr,
                                 "%s: grouped output read failed\n", label);
                    return false;
                }
                const double abs_error = std::fabs(
                    static_cast<double>(got) - reference);
                const double rel_error = reference != 0.0f
                    ? abs_error / std::fabs(static_cast<double>(reference))
                    : (abs_error == 0.0
                           ? 0.0
                           : std::numeric_limits<double>::infinity());
                if (!std::isfinite(got) ||
                    (abs_error > abs_tol && rel_error > rel_tol)) {
                    std::fprintf(
                        stderr,
                        "%s: grouped CPU oracle mismatch token=%u group=%u "
                        "row=%u got=%.7g reference=%.7g abs=%.5g rel=%.5g "
                        "limits=%.5g/%.3g\n",
                        label, token, group, row, got, reference, abs_error,
                        rel_error, abs_tol, rel_tol);
                    return false;
                }
            }
        }
    }
    return true;
}

bool select_grouped_prefill_legacy() {
    return unsetenv(kGroupedPrefillEnable) == 0 &&
           setenv(kGroupedPrefillDisable, "1", 1) == 0 &&
           unsetenv(kGroupedPrefillRequire) == 0 &&
           unsetenv(kGroupedSingleGridEnable) == 0 &&
           setenv(kGroupedSingleGridDisable, "1", 1) == 0 &&
           unsetenv(kGroupedSingleGridRequire) == 0 &&
           unsetenv(kGroupedQ81Disable) == 0 &&
           unsetenv(kGroupedQ81Require) == 0;
}

bool select_grouped_prefill_grid8() {
    return setenv(kGroupedPrefillEnable, "1", 1) == 0 &&
           unsetenv(kGroupedPrefillDisable) == 0 &&
           setenv(kGroupedPrefillRequire, "1", 1) == 0 &&
           unsetenv(kGroupedSingleGridEnable) == 0 &&
           setenv(kGroupedSingleGridDisable, "1", 1) == 0 &&
           unsetenv(kGroupedSingleGridRequire) == 0 &&
           unsetenv(kGroupedQ81Disable) == 0 &&
           unsetenv(kGroupedQ81Require) == 0;
}

bool select_grouped_prefill_single_grid() {
    return setenv(kGroupedPrefillEnable, "1", 1) == 0 &&
           unsetenv(kGroupedPrefillDisable) == 0 &&
           setenv(kGroupedPrefillRequire, "1", 1) == 0 &&
           setenv(kGroupedSingleGridEnable, "1", 1) == 0 &&
           unsetenv(kGroupedSingleGridDisable) == 0 &&
           setenv(kGroupedSingleGridRequire, "1", 1) == 0 &&
           unsetenv(kGroupedQ81Disable) == 0 &&
           unsetenv(kGroupedQ81Require) == 0;
}

bool select_grouped_q81_reference() {
    return select_grouped_prefill_grid8() &&
           setenv(kGroupedQ81Disable, "1", 1) == 0 &&
           unsetenv(kGroupedQ81Require) == 0;
}

bool select_grouped_q81_candidate() {
    return select_grouped_prefill_grid8() &&
           unsetenv(kGroupedQ81Disable) == 0 &&
           setenv(kGroupedQ81Require, "1", 1) == 0;
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

const char *path_name(cuda_path path) {
    return path == cuda_path::mmq ? "mmq" : "legacy";
}

const char *case_scope(const config &cfg) {
    switch (cfg.selected) {
        case bench_case::all:
            return cfg.kernel_16warp ? "dense,pair,q_b,output_b"
                                     : "dense,pair,q_b,outa";
        case bench_case::dense: return "dense";
        case bench_case::pair: return "pair";
        case bench_case::qb: return "q_b";
        case bench_case::outa: return "outa";
        case bench_case::outb: return "output_b";
    }
    return "unknown";
}

bool select_arm(const arm &which) {
    return !which.select || which.select();
}

bool dispatch_oracle_arm(const arm &which, uint32_t set) {
    return which.oracle_dispatch ? which.oracle_dispatch(set)
                                 : which.dispatch(set);
}

bool benchmark_single_path(
        const char *case_name, uint32_t n_tokens, uint32_t in_dim,
        uint32_t out_dim, const config &cfg, const arm &which,
        const std::function<bool()> &oracle_prepare,
        const std::function<bool(uint32_t)> &oracle) {
    auto validate = [&](const char *phase) {
        for (uint32_t set = 0; set < cfg.sets; set++) {
            if (!oracle_prepare() || !select_arm(which) ||
                !dispatch_oracle_arm(which, set) || !ds4_gpu_synchronize() ||
                !oracle(set)) {
                std::fprintf(stderr,
                             "cuda-q4-prefill-bench: %s %s oracle failed "
                             "for weight set %u\n",
                             case_name, phase, set);
                return false;
            }
        }
        return true;
    };

    // Validate every rotating weight set and prime lazy Q8 scratch before
    // recording a CUDA event.  Upload, poison, readback, and CPU work remain
    // outside the measured interval.
    if (!validate("pre-timing")) return false;

    for (uint32_t i = 0; i < cfg.warmup; i++) {
        if (!select_arm(which) || !which.dispatch(i % cfg.sets) ||
            !ds4_gpu_synchronize()) {
            return false;
        }
    }

    event_timer timer;
    std::vector<double> samples;
    samples.reserve(cfg.samples);
    for (uint32_t i = 0; i < cfg.samples; i++) {
        float elapsed = 0.0f;
        if (!select_arm(which) ||
            !timer.measure([&]() { return which.dispatch(i % cfg.sets); },
                           &elapsed)) {
            std::fprintf(stderr,
                         "cuda-q4-prefill-bench: %s/%s timed dispatch failed\n",
                         case_name, which.name);
            return false;
        }
        samples.push_back(static_cast<double>(elapsed));
    }

    const uint32_t timed_set = (cfg.samples - 1u) % cfg.sets;
    if (!oracle(timed_set)) {
        std::fprintf(stderr,
                     "cuda-q4-prefill-bench: %s timed-output oracle failed "
                     "for weight set %u\n",
                     case_name, timed_set);
        return false;
    }

    // Timed launches overwrite the validated buffers.  Re-run the complete
    // finite/CPU/guard oracle for every resident weight set after timing so a
    // late or state-dependent corruption cannot survive as a benchmark result.
    if (!validate("post-timing")) return false;

    const stats result = summarize(samples);
    const double macs = static_cast<double>(n_tokens) * in_dim * out_dim;
    const double gmac_s = macs / (result.median * 1.0e6);
    std::printf(
        "DS4_CUDA_Q4_PREFILL_BENCH case=%s path=%s N=%u K=%u M=%u "
        "variant=%s samples=%u sets=%u ms_p50=%.6f ms_min=%.6f "
        "ms_p95=%.6f ms_mean=%.6f gmac_s=%.3f\n",
        case_name, path_name(cfg.path), n_tokens, in_dim, out_dim, which.name,
        cfg.samples, cfg.sets, result.median, result.minimum, result.p95,
        result.mean, gmac_s);
    std::fflush(stdout);
    return true;
}

bool benchmark_pair_arms(
        const char *case_name, uint32_t n_tokens, uint32_t in_dim,
        uint32_t out_dim, uint64_t focus_macs_per_token,
        uint64_t common_macs_per_token, const config &cfg,
        const arm &baseline, const arm &candidate,
        const std::function<bool()> &oracle_prepare,
        const std::function<bool(uint32_t)> &oracle,
        const char *timing_scope = "production_api_cuda_events") {
    auto validate = [&](const char *phase) {
        for (uint32_t set = 0; set < cfg.sets; set++) {
            if (!oracle_prepare() || !select_arm(baseline) ||
                !dispatch_oracle_arm(baseline, set) ||
                !ds4_gpu_synchronize() || !select_arm(candidate) ||
                !dispatch_oracle_arm(candidate, set) ||
                !ds4_gpu_synchronize() || !oracle(set)) {
                std::fprintf(stderr,
                             "cuda-q4-prefill-bench: %s %s oracle failed "
                             "for weight set %u\n",
                             case_name, phase, set);
                return false;
            }
        }
        return true;
    };

    if (!validate("pre-timing")) return false;

    for (uint32_t i = 0; i < cfg.warmup; i++) {
        const uint32_t set = i % cfg.sets;
        if (!select_arm(baseline) || !baseline.dispatch(set) ||
            !ds4_gpu_synchronize() || !select_arm(candidate) ||
            !candidate.dispatch(set) || !ds4_gpu_synchronize()) {
            return false;
        }
    }

    event_timer timer;
    std::vector<double> a_samples;
    std::vector<double> b_samples;
    a_samples.reserve(cfg.samples);
    b_samples.reserve(cfg.samples);
    auto take = [&](const arm &which, uint32_t set,
                    std::vector<double> *samples) {
        float elapsed = 0.0f;
        if (!select_arm(which) ||
            !timer.measure([&]() { return which.dispatch(set); }, &elapsed)) {
            std::fprintf(stderr,
                         "cuda-q4-prefill-bench: %s/%s timed dispatch "
                         "failed\n",
                         case_name, which.name);
            return false;
        }
        samples->push_back(static_cast<double>(elapsed));
        return true;
    };

    // Two samples per arm per cycle. Alternating ABBA/BAAB balances order
    // while both arms see identical rotating resident weights.
    for (uint32_t cycle = 0; a_samples.size() < cfg.samples; cycle++) {
        const uint32_t set0 = (cycle * 2u) % cfg.sets;
        const uint32_t set1 = (cycle * 2u + 1u) % cfg.sets;
        if ((cycle & 1u) == 0u) {
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

    // samples is a multiple of four, so the final ABBA/BAAB cycle leaves both
    // output buffers holding weight set samples-1. Check those exact timed
    // results before oracle_prepare is allowed to poison or reset any guard.
    const uint32_t timed_set = (cfg.samples - 1u) % cfg.sets;
    if (!oracle(timed_set)) {
        std::fprintf(stderr,
                     "cuda-q4-prefill-bench: %s timed-output oracle failed "
                     "for weight set %u\n",
                     case_name, timed_set);
        return false;
    }

    // Both arms must still agree bit-for-bit (and with their CPU/guard
    // oracle) after the measured ABBA/BAAB sequence, for every weight set.
    if (!validate("post-timing")) return false;

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
    const double macs = static_cast<double>(n_tokens) *
                        (focus_macs_per_token + common_macs_per_token);
    std::printf(
        "DS4_CUDA_Q4_PREFILL_BENCH case=%s path=%s N=%u K=%u M=%u "
        "baseline=%s candidate=%s samples=%u sets=%u "
        "timing=%s "
        "focus_macs_per_token=%llu common_macs_per_token=%llu "
        "baseline_ms_p50=%.6f candidate_ms_p50=%.6f "
        "baseline_ms_min=%.6f candidate_ms_min=%.6f "
        "baseline_ms_p95=%.6f candidate_ms_p95=%.6f "
        "baseline_gmac_s=%.3f candidate_gmac_s=%.3f "
        "candidate_delta_pct=%.3f paired_delta_pct_p50=%.3f "
        "speedup_pct=%.3f\n",
        case_name, path_name(cfg.path), n_tokens, in_dim, out_dim,
        baseline.name, candidate.name, cfg.samples, cfg.sets,
        timing_scope,
        static_cast<unsigned long long>(focus_macs_per_token),
        static_cast<unsigned long long>(common_macs_per_token),
        a.median, b.median, a.minimum, b.minimum, a.p95, b.p95,
        macs / (a.median * 1.0e6), macs / (b.median * 1.0e6),
        median_delta, paired_median, speedup);
    std::fflush(stdout);
    return true;
}

bool run_q4_16warp_kernel(
        const model_fixture &model, const config &cfg, uint32_t n_tokens,
        uint32_t in_dim, uint32_t out_dim,
        uint64_t weight_set::*weight_offset_member, const char *case_name) {
    uint64_t x_elements = 0;
    uint64_t out_elements = 0;
    if (!checked_mul(n_tokens, in_dim, &x_elements) ||
        !checked_mul(n_tokens, out_dim, &out_elements)) {
        return false;
    }
    const uint64_t x_bytes = x_elements * sizeof(float);
    const uint64_t out_bytes = out_elements * sizeof(float);
    const uint64_t guard_bytes = kGuardWords * sizeof(uint32_t);
    const uint64_t weight_bytes = q4_weight_bytes(in_dim, out_dim);
    const size_t q8_bytes = ds4_mmq_q4_K_q8_1_scratch_bytes(
        static_cast<int>(n_tokens), static_cast<int>(in_dim));
    if (q8_bytes == 0u) {
        std::fprintf(stderr,
                     "cuda-q4-prefill-bench: %s kernel-only Q8_1 scratch "
                     "shape rejected for N=%u K=%u\n",
                     case_name, n_tokens, in_dim);
        return false;
    }
    int nsm = 0;
    if (!current_device_nsm(&nsm)) return false;
    const size_t fixup_bytes =
        ds4_mmq_q4_K_dense_16warp_streamk_scratch_bytes(
            static_cast<int>(out_dim), static_cast<int>(n_tokens), nsm);

    tensor_owner x(x_bytes + guard_bytes);
    tensor_owner reference(out_bytes + guard_bytes);
    tensor_owner candidate(out_bytes + guard_bytes);
    cuda_buffer prequant(q8_bytes);
    guarded_cuda_buffer fixup(fixup_bytes, 0x1a000u);
    std::vector<float> activation;
    fill_activation(&activation, n_tokens, in_dim);
    if (!x.ptr || !reference.ptr || !candidate.ptr || !prequant.ptr ||
        prequant.status != cudaSuccess || fixup.status != cudaSuccess ||
        (fixup_bytes != 0u && !fixup.ptr) ||
        !ds4_gpu_tensor_write(x.ptr, 0, activation.data(), x_bytes) ||
        !prepare_guard(x.ptr, x_bytes, 0x12000u)) {
        std::fprintf(stderr,
                     "cuda-q4-prefill-bench: %s N=%u kernel-only tensor/"
                     "scratch setup failed (%s)\n",
                     case_name, n_tokens,
                     prequant.status != cudaSuccess
                         ? cudaGetErrorString(prequant.status)
                         : fixup.status != cudaSuccess
                               ? cudaGetErrorString(fixup.status)
                               : "tensor setup");
        return false;
    }

    const auto *x_device = static_cast<const float *>(
        ds4_gpu_tensor_contents(x.ptr));
    auto *reference_device = static_cast<float *>(
        ds4_gpu_tensor_contents(reference.ptr));
    auto *candidate_device = static_cast<float *>(
        ds4_gpu_tensor_contents(candidate.ptr));
    if (!x_device || !reference_device || !candidate_device) {
        std::fprintf(stderr,
                     "cuda-q4-prefill-bench: %s N=%u device tensor pointer "
                     "lookup failed\n",
                     case_name, n_tokens);
        return false;
    }

    std::vector<const void *> weight_device(cfg.sets, nullptr);
    for (uint32_t set = 0; set < cfg.sets; set++) {
        const uint64_t offset = model.weights[set].*weight_offset_member;
        if (!ds4_cuda_test_model_range_device_ptr(
                model.data, model.size, offset, weight_bytes,
                /*logical_tier=*/0, &weight_device[set]) ||
            !weight_device[set]) {
            std::fprintf(stderr,
                         "cuda-q4-prefill-bench: %s N=%u resident weight "
                         "pointer lookup failed for set %u\n",
                         case_name, n_tokens, set);
            return false;
        }
    }

    // Quantize exactly once. Both A/B arms consume this immutable canonical
    // DS4 Q8_1 buffer. Timed work is restricted to each GEMM's scheduling,
    // required stream-K scratch clear, producer, and fixup kernels.
    const int quant_rc = ds4_mmq_q4_K_quantize_q8_1_for_test(
        x_device, prequant.ptr, q8_bytes, static_cast<int>(n_tokens),
        static_cast<int>(in_dim), /*stream=*/nullptr);
    if (quant_rc != 0 || !ds4_gpu_synchronize() ||
        !check_guard(x.ptr, x_bytes, 0x12000u,
                     "kernel-only prequant input")) {
        std::fprintf(stderr,
                     "cuda-q4-prefill-bench: %s N=%u prequantization "
                     "failed rc=%d\n",
                     case_name, n_tokens, quant_rc);
        return false;
    }

    const arm baseline = {
        "canonical_preq_stream_k",
        [&](uint32_t set) {
            return ds4_mmq_q4_K_dense_preq_reference_for_test(
                       weight_device[set], prequant.ptr, q8_bytes,
                       reference_device, static_cast<int>(out_dim),
                       static_cast<int>(n_tokens), static_cast<int>(in_dim),
                       /*use_stream_k=*/1, fixup.ptr, fixup_bytes,
                       /*stream=*/nullptr) == 0;
        },
        {}};
    const arm candidate_arm = {
        "q4_16warp_m128n128_stream_k",
        [&](uint32_t set) {
            return ds4_mmq_q4_K_dense_16warp_streamk_enqueue(
                       weight_device[set], prequant.ptr, candidate_device,
                       fixup.ptr, fixup_bytes,
                       static_cast<int>(out_dim),
                       static_cast<int>(n_tokens), static_cast<int>(in_dim),
                       nsm, /*stream=*/nullptr) == 0;
        },
        {},
        [&](uint32_t set) {
            return ds4_mmq_q4_K_dense_16warp_streamk_enqueue(
                       weight_device[set], prequant.ptr, candidate_device,
                       fixup.ptr, fixup_bytes, static_cast<int>(out_dim),
                       static_cast<int>(n_tokens), static_cast<int>(in_dim),
                       nsm, /*stream=*/nullptr) == 0;
        }};
    const bool ok = benchmark_pair_arms(
        case_name, n_tokens, in_dim, out_dim,
        static_cast<uint64_t>(in_dim) * out_dim, 0u, cfg,
        baseline, candidate_arm,
        [&]() {
            return poison_output(reference.ptr, out_bytes, 0x7fc50005u,
                                 0x13000u) &&
                   poison_output(candidate.ptr, out_bytes, 0x7fc60006u,
                                 0x14000u);
        },
        [&](uint32_t set) {
            const uint64_t offset =
                model.weights[set].*weight_offset_member;
            return bitwise_equal(reference.ptr, candidate.ptr, out_bytes,
                                 "16-warp vs canonical stream-K") &&
                   output_is_finite(reference.ptr, out_bytes,
                                    "16-warp canonical output") &&
                   output_is_finite(candidate.ptr, out_bytes,
                                    "16-warp candidate output") &&
                   sampled_cpu_oracle(
                       reference.ptr, model, offset, activation, n_tokens,
                       in_dim, out_dim, "16-warp canonical") &&
                   check_guard(x.ptr, x_bytes, 0x12000u,
                               "16-warp input") &&
                   check_guard(reference.ptr, out_bytes, 0x13000u,
                               "16-warp canonical output") &&
                   check_guard(candidate.ptr, out_bytes, 0x14000u,
                               "16-warp candidate output") &&
                   fixup.intact("16-warp Stream-K scratch");
        },
        "kernel_only_prequant");
    return ok &&
           check_guard(x.ptr, x_bytes, 0x12000u,
                       "16-warp input final") &&
           check_guard(reference.ptr, out_bytes, 0x13000u,
                       "16-warp canonical output final") &&
           check_guard(candidate.ptr, out_bytes, 0x14000u,
                       "16-warp candidate output final") &&
           fixup.intact("16-warp Stream-K scratch final");
}

bool run_q4_16warp_pair_kernel(
        const model_fixture &model, const config &cfg, uint32_t n_tokens) {
    uint64_t x_elements = 0;
    uint64_t out0_elements = 0;
    uint64_t out1_elements = 0;
    if (!checked_mul(n_tokens, kDenseK, &x_elements) ||
        !checked_mul(n_tokens, kDenseM, &out0_elements) ||
        !checked_mul(n_tokens, kKvM, &out1_elements)) {
        return false;
    }
    const uint64_t x_bytes = x_elements * sizeof(float);
    const uint64_t out0_bytes = out0_elements * sizeof(float);
    const uint64_t out1_bytes = out1_elements * sizeof(float);
    const uint64_t guard_bytes = kGuardWords * sizeof(uint32_t);
    const uint64_t weight0_bytes = q4_weight_bytes(kDenseK, kDenseM);
    const uint64_t weight1_bytes = q4_weight_bytes(kDenseK, kKvM);
    const size_t q8_bytes = ds4_mmq_q4_K_q8_1_scratch_bytes(
        static_cast<int>(n_tokens), static_cast<int>(kDenseK));
    if (q8_bytes == 0u) {
        std::fprintf(stderr,
                     "cuda-q4-prefill-bench: pair kernel-only Q8_1 scratch "
                     "shape rejected for N=%u K=%u\n",
                     n_tokens, kDenseK);
        return false;
    }
    int nsm = 0;
    if (!current_device_nsm(&nsm)) return false;
    const size_t fixup0_bytes =
        ds4_mmq_q4_K_dense_16warp_streamk_scratch_bytes(
            static_cast<int>(kDenseM), static_cast<int>(n_tokens), nsm);
    const size_t fixup1_bytes =
        ds4_mmq_q4_K_dense_16warp_streamk_scratch_bytes(
            static_cast<int>(kKvM), static_cast<int>(n_tokens), nsm);
    const size_t fixup_bytes = fixup0_bytes > fixup1_bytes
        ? fixup0_bytes : fixup1_bytes;

    tensor_owner x(x_bytes + guard_bytes);
    tensor_owner reference0(out0_bytes + guard_bytes);
    tensor_owner reference1(out1_bytes + guard_bytes);
    tensor_owner candidate0(out0_bytes + guard_bytes);
    tensor_owner candidate1(out1_bytes + guard_bytes);
    cuda_buffer prequant(q8_bytes);
    guarded_cuda_buffer fixup(fixup_bytes, 0x1b000u);
    std::vector<float> activation;
    fill_activation(&activation, n_tokens, kDenseK);
    if (!x.ptr || !reference0.ptr || !reference1.ptr || !candidate0.ptr ||
        !candidate1.ptr || !prequant.ptr || prequant.status != cudaSuccess ||
        fixup.status != cudaSuccess ||
        (fixup_bytes != 0u && !fixup.ptr) ||
        !ds4_gpu_tensor_write(x.ptr, 0, activation.data(), x_bytes) ||
        !prepare_guard(x.ptr, x_bytes, 0x15000u)) {
        std::fprintf(stderr,
                     "cuda-q4-prefill-bench: pair N=%u kernel-only tensor/"
                     "scratch setup failed (%s)\n",
                     n_tokens,
                     prequant.status != cudaSuccess
                         ? cudaGetErrorString(prequant.status)
                         : fixup.status != cudaSuccess
                               ? cudaGetErrorString(fixup.status)
                               : "tensor setup");
        return false;
    }

    const auto *x_device = static_cast<const float *>(
        ds4_gpu_tensor_contents(x.ptr));
    auto *reference0_device = static_cast<float *>(
        ds4_gpu_tensor_contents(reference0.ptr));
    auto *reference1_device = static_cast<float *>(
        ds4_gpu_tensor_contents(reference1.ptr));
    auto *candidate0_device = static_cast<float *>(
        ds4_gpu_tensor_contents(candidate0.ptr));
    auto *candidate1_device = static_cast<float *>(
        ds4_gpu_tensor_contents(candidate1.ptr));
    if (!x_device || !reference0_device || !reference1_device ||
        !candidate0_device || !candidate1_device) {
        std::fprintf(stderr,
                     "cuda-q4-prefill-bench: pair N=%u device tensor pointer "
                     "lookup failed\n",
                     n_tokens);
        return false;
    }

    std::vector<const void *> weight0_device(cfg.sets, nullptr);
    std::vector<const void *> weight1_device(cfg.sets, nullptr);
    for (uint32_t set = 0; set < cfg.sets; set++) {
        if (!ds4_cuda_test_model_range_device_ptr(
                model.data, model.size, model.weights[set].dense_offset,
                weight0_bytes, /*logical_tier=*/0, &weight0_device[set]) ||
            !weight0_device[set] ||
            !ds4_cuda_test_model_range_device_ptr(
                model.data, model.size, model.weights[set].kv_offset,
                weight1_bytes, /*logical_tier=*/0, &weight1_device[set]) ||
            !weight1_device[set]) {
            std::fprintf(stderr,
                         "cuda-q4-prefill-bench: pair N=%u resident weight "
                         "pointer lookup failed for set %u\n",
                         n_tokens, set);
            return false;
        }
    }

    // One immutable activation quantization is shared by both Q-A/KV legs and
    // is deliberately outside every event interval.
    const int quant_rc = ds4_mmq_q4_K_quantize_q8_1_for_test(
        x_device, prequant.ptr, q8_bytes, static_cast<int>(n_tokens),
        static_cast<int>(kDenseK), /*stream=*/nullptr);
    if (quant_rc != 0 || !ds4_gpu_synchronize() ||
        !check_guard(x.ptr, x_bytes, 0x15000u,
                     "kernel-only pair prequant input")) {
        std::fprintf(stderr,
                     "cuda-q4-prefill-bench: pair N=%u prequantization "
                     "failed rc=%d\n",
                     n_tokens, quant_rc);
        return false;
    }

    const arm baseline = {
        "two_canonical_preq_stream_k",
        [&](uint32_t set) {
            return ds4_mmq_q4_K_dense_preq_reference_for_test(
                       weight0_device[set], prequant.ptr, q8_bytes,
                       reference0_device, static_cast<int>(kDenseM),
                       static_cast<int>(n_tokens), static_cast<int>(kDenseK),
                       /*use_stream_k=*/1, fixup.ptr, fixup_bytes,
                       /*stream=*/nullptr) == 0 &&
                   ds4_mmq_q4_K_dense_preq_reference_for_test(
                       weight1_device[set], prequant.ptr, q8_bytes,
                       reference1_device, static_cast<int>(kKvM),
                       static_cast<int>(n_tokens), static_cast<int>(kDenseK),
                       /*use_stream_k=*/1, fixup.ptr, fixup_bytes,
                       /*stream=*/nullptr) == 0;
        },
        {}};
    const arm candidate = {
        "two_q4_16warp_m128n128_stream_k",
        [&](uint32_t set) {
            return ds4_mmq_q4_K_dense_16warp_streamk_enqueue(
                       weight0_device[set], prequant.ptr, candidate0_device,
                       fixup.ptr, fixup_bytes,
                       static_cast<int>(kDenseM),
                       static_cast<int>(n_tokens), static_cast<int>(kDenseK),
                       nsm, /*stream=*/nullptr) == 0 &&
                   ds4_mmq_q4_K_dense_16warp_streamk_enqueue(
                       weight1_device[set], prequant.ptr, candidate1_device,
                       fixup.ptr, fixup_bytes,
                       static_cast<int>(kKvM),
                       static_cast<int>(n_tokens), static_cast<int>(kDenseK),
                       nsm, /*stream=*/nullptr) == 0;
        },
        {},
        [&](uint32_t set) {
            return ds4_mmq_q4_K_dense_16warp_streamk_enqueue(
                       weight0_device[set], prequant.ptr, candidate0_device,
                       fixup.ptr, fixup_bytes, static_cast<int>(kDenseM),
                       static_cast<int>(n_tokens), static_cast<int>(kDenseK),
                       nsm, /*stream=*/nullptr) == 0 &&
                   ds4_mmq_q4_K_dense_16warp_streamk_enqueue(
                       weight1_device[set], prequant.ptr, candidate1_device,
                       fixup.ptr, fixup_bytes, static_cast<int>(kKvM),
                       static_cast<int>(n_tokens), static_cast<int>(kDenseK),
                       nsm, /*stream=*/nullptr) == 0;
        }};
    const bool ok = benchmark_pair_arms(
        "pair", n_tokens, kDenseK, kDenseM + kKvM,
        static_cast<uint64_t>(kDenseK) * (kDenseM + kKvM), 0u, cfg,
        baseline, candidate,
        [&]() {
            return poison_output(reference0.ptr, out0_bytes, 0x7fc70007u,
                                 0x16000u) &&
                   poison_output(reference1.ptr, out1_bytes, 0x7fc80008u,
                                 0x17000u) &&
                   poison_output(candidate0.ptr, out0_bytes, 0x7fc90009u,
                                 0x18000u) &&
                   poison_output(candidate1.ptr, out1_bytes, 0x7fca000au,
                                 0x19000u);
        },
        [&](uint32_t set) {
            return bitwise_equal(reference0.ptr, candidate0.ptr, out0_bytes,
                                 "16-warp pair q_a vs canonical") &&
                   bitwise_equal(reference1.ptr, candidate1.ptr, out1_bytes,
                                 "16-warp pair kv vs canonical") &&
                   output_is_finite(reference0.ptr, out0_bytes,
                                    "16-warp pair q_a canonical") &&
                   output_is_finite(reference1.ptr, out1_bytes,
                                    "16-warp pair kv canonical") &&
                   output_is_finite(candidate0.ptr, out0_bytes,
                                    "16-warp pair q_a candidate") &&
                   output_is_finite(candidate1.ptr, out1_bytes,
                                    "16-warp pair kv candidate") &&
                   sampled_cpu_oracle(
                       reference0.ptr, model,
                       model.weights[set].dense_offset, activation, n_tokens,
                       kDenseK, kDenseM, "16-warp pair q_a canonical") &&
                   sampled_cpu_oracle(
                       reference1.ptr, model, model.weights[set].kv_offset,
                       activation, n_tokens, kDenseK, kKvM,
                       "16-warp pair kv canonical") &&
                   check_guard(x.ptr, x_bytes, 0x15000u,
                               "16-warp pair input") &&
                   check_guard(reference0.ptr, out0_bytes, 0x16000u,
                               "16-warp pair q_a canonical") &&
                   check_guard(reference1.ptr, out1_bytes, 0x17000u,
                               "16-warp pair kv canonical") &&
                   check_guard(candidate0.ptr, out0_bytes, 0x18000u,
                               "16-warp pair q_a candidate") &&
                   check_guard(candidate1.ptr, out1_bytes, 0x19000u,
                               "16-warp pair kv candidate") &&
                   fixup.intact("16-warp pair Stream-K scratch");
        },
        "kernel_only_prequant");
    return ok &&
           check_guard(x.ptr, x_bytes, 0x15000u,
                       "16-warp pair input final") &&
           check_guard(reference0.ptr, out0_bytes, 0x16000u,
                       "16-warp pair q_a canonical final") &&
           check_guard(reference1.ptr, out1_bytes, 0x17000u,
                       "16-warp pair kv canonical final") &&
           check_guard(candidate0.ptr, out0_bytes, 0x18000u,
                       "16-warp pair q_a candidate final") &&
           check_guard(candidate1.ptr, out1_bytes, 0x19000u,
                       "16-warp pair kv candidate final") &&
           fixup.intact("16-warp pair Stream-K scratch final");
}

bool run_dense(const model_fixture &model, const config &cfg,
               uint32_t n_tokens) {
    uint64_t out_elements = 0;
    if (!checked_mul(n_tokens, kDenseM, &out_elements)) return false;
    const uint64_t x_bytes = static_cast<uint64_t>(n_tokens) * kDenseK *
                             sizeof(float);
    const uint64_t out_bytes = out_elements * sizeof(float);
    const uint64_t guard_bytes = kGuardWords * sizeof(uint32_t);
    tensor_owner x(x_bytes + guard_bytes);
    tensor_owner output(out_bytes + guard_bytes);
    std::vector<float> activation;
    fill_activation(&activation, n_tokens, kDenseK);
    if (!x.ptr || !output.ptr ||
        !ds4_gpu_tensor_write(x.ptr, 0, activation.data(), x_bytes) ||
        !prepare_guard(x.ptr, x_bytes, 0x1000u)) {
        std::fprintf(stderr, "dense N=%u: tensor setup failed\n", n_tokens);
        return false;
    }
    const arm path = {
        cfg.path == cuda_path::mmq ? "mmq" : "legacy_q8k",
        [&](uint32_t set) {
            return ds4_gpu_matmul_quant_tensor(
                       output.ptr, model.data, model.size,
                       model.weights[set].dense_offset, kQ4Type, kDenseK,
                       kDenseM, x.ptr, n_tokens) != 0;
        },
        {}};
    return benchmark_single_path(
        "dense", n_tokens, kDenseK, kDenseM, cfg, path,
        [&]() {
            return poison_output(output.ptr, out_bytes, 0x7fc10001u,
                                 0x2000u);
        },
        [&](uint32_t set) {
            return output_is_finite(output.ptr, out_bytes, "dense output") &&
                   sampled_cpu_oracle(output.ptr, model,
                                      model.weights[set].dense_offset,
                                      activation, n_tokens, kDenseK, kDenseM,
                                      "dense") &&
                   check_guard(x.ptr, x_bytes, 0x1000u, "dense input") &&
                   check_guard(output.ptr, out_bytes, 0x2000u,
                               "dense output");
        }) &&
        check_guard(x.ptr, x_bytes, 0x1000u, "dense input final") &&
        check_guard(output.ptr, out_bytes, 0x2000u, "dense output final");
}

bool run_pair(const model_fixture &model, const config &cfg,
              uint32_t n_tokens) {
    if (cfg.path == cuda_path::legacy) {
        std::printf(
            "DS4_CUDA_Q4_PREFILL_SKIP case=pair path=legacy N=%u "
            "reason=fused_prefill_api_requires_mmq\n",
            n_tokens);
        std::fflush(stdout);
        return true;
    }

    const uint64_t x_bytes = static_cast<uint64_t>(n_tokens) * kDenseK *
                             sizeof(float);
    const uint64_t out0_bytes = static_cast<uint64_t>(n_tokens) * kDenseM *
                                sizeof(float);
    const uint64_t out1_bytes = static_cast<uint64_t>(n_tokens) * kKvM *
                                sizeof(float);
    const uint64_t guard_bytes = kGuardWords * sizeof(uint32_t);
    tensor_owner x(x_bytes + guard_bytes);
    tensor_owner separate0(out0_bytes + guard_bytes);
    tensor_owner separate1(out1_bytes + guard_bytes);
    tensor_owner pair0(out0_bytes + guard_bytes);
    tensor_owner pair1(out1_bytes + guard_bytes);
    std::vector<float> activation;
    fill_activation(&activation, n_tokens, kDenseK);
    if (!x.ptr || !separate0.ptr || !separate1.ptr || !pair0.ptr ||
        !pair1.ptr ||
        !ds4_gpu_tensor_write(x.ptr, 0, activation.data(), x_bytes) ||
        !prepare_guard(x.ptr, x_bytes, 0x3000u)) {
        std::fprintf(stderr, "pair N=%u: tensor setup failed\n", n_tokens);
        return false;
    }
    const arm baseline = {
        "two_dense_mmq",
        [&](uint32_t set) {
            return ds4_gpu_matmul_quant_tensor(
                       separate0.ptr, model.data, model.size,
                       model.weights[set].dense_offset, kQ4Type, kDenseK,
                       kDenseM, x.ptr, n_tokens) != 0 &&
                   ds4_gpu_matmul_quant_tensor(
                       separate1.ptr, model.data, model.size,
                       model.weights[set].kv_offset, kQ4Type, kDenseK,
                       kKvM, x.ptr, n_tokens) != 0;
        },
        {}};
    const arm candidate = {
        "pair_mmq",
        [&](uint32_t set) {
            return ds4_gpu_matmul_q4_K_pair_tensor(
                       pair0.ptr, pair1.ptr, model.data, model.size,
                       model.weights[set].dense_offset,
                       model.weights[set].kv_offset, kDenseK, kDenseM, kKvM,
                       x.ptr, n_tokens) == 1;
        },
        {}};
    return benchmark_pair_arms(
        "pair", n_tokens, kDenseK, kDenseM + kKvM,
        static_cast<uint64_t>(kDenseK) * (kDenseM + kKvM), 0u,
        cfg, baseline, candidate,
        [&]() {
            return poison_output(separate0.ptr, out0_bytes, 0x7fc10001u,
                                 0x4000u) &&
                   poison_output(separate1.ptr, out1_bytes, 0x7fc20002u,
                                 0x5000u) &&
                   poison_output(pair0.ptr, out0_bytes, 0x7fc30003u,
                                 0x6000u) &&
                   poison_output(pair1.ptr, out1_bytes, 0x7fc40004u,
                                 0x7000u);
        },
        [&](uint32_t set) {
            return bitwise_equal(separate0.ptr, pair0.ptr, out0_bytes,
                                 "pair q_a output") &&
                   bitwise_equal(separate1.ptr, pair1.ptr, out1_bytes,
                                 "pair kv output") &&
                   output_is_finite(separate0.ptr, out0_bytes,
                                    "pair q_a output") &&
                   output_is_finite(separate1.ptr, out1_bytes,
                                    "pair kv output") &&
                   sampled_cpu_oracle(separate0.ptr, model,
                                      model.weights[set].dense_offset,
                                      activation, n_tokens, kDenseK, kDenseM,
                                      "pair q_a") &&
                   sampled_cpu_oracle(separate1.ptr, model,
                                      model.weights[set].kv_offset,
                                      activation, n_tokens, kDenseK, kKvM,
                                      "pair kv") &&
                   check_guard(x.ptr, x_bytes, 0x3000u, "pair input") &&
                   check_guard(separate0.ptr, out0_bytes, 0x4000u,
                               "pair separate q_a") &&
                   check_guard(separate1.ptr, out1_bytes, 0x5000u,
                               "pair separate kv") &&
                   check_guard(pair0.ptr, out0_bytes, 0x6000u,
                               "pair fused q_a") &&
                   check_guard(pair1.ptr, out1_bytes, 0x7000u,
                               "pair fused kv");
        }) &&
        check_guard(x.ptr, x_bytes, 0x3000u, "pair input final") &&
        check_guard(separate0.ptr, out0_bytes, 0x4000u,
                    "pair separate q_a final") &&
        check_guard(separate1.ptr, out1_bytes, 0x5000u,
                    "pair separate kv final") &&
        check_guard(pair0.ptr, out0_bytes, 0x6000u,
                    "pair fused q_a final") &&
        check_guard(pair1.ptr, out1_bytes, 0x7000u,
                    "pair fused kv final");
}

bool run_qb(const model_fixture &model, const config &cfg,
            uint32_t n_tokens) {
    uint64_t out_elements = 0;
    if (!checked_mul(n_tokens, kQbM, &out_elements)) return false;
    const uint64_t x_bytes = static_cast<uint64_t>(n_tokens) * kQbK *
                             sizeof(float);
    const uint64_t out_bytes = out_elements * sizeof(float);
    const uint64_t guard_bytes = kGuardWords * sizeof(uint32_t);
    tensor_owner x(x_bytes + guard_bytes);
    tensor_owner output(out_bytes + guard_bytes);
    std::vector<float> activation;
    fill_activation(&activation, n_tokens, kQbK);
    if (!x.ptr || !output.ptr ||
        !ds4_gpu_tensor_write(x.ptr, 0, activation.data(), x_bytes) ||
        !prepare_guard(x.ptr, x_bytes, 0x8000u)) {
        std::fprintf(stderr, "q_b N=%u: tensor setup failed\n", n_tokens);
        return false;
    }
    const arm path = {
        cfg.path == cuda_path::mmq ? "mmq" : "legacy_q8k",
        [&](uint32_t set) {
            return ds4_gpu_matmul_quant_tensor(
                       output.ptr, model.data, model.size,
                       model.weights[set].qb_offset, kQ4Type, kQbK, kQbM,
                       x.ptr, n_tokens) != 0;
        },
        {}};
    return benchmark_single_path(
        "q_b", n_tokens, kQbK, kQbM, cfg, path,
        [&]() {
            return poison_output(output.ptr, out_bytes, 0x7fc10001u,
                                 0x9000u);
        },
        [&](uint32_t set) {
            return output_is_finite(output.ptr, out_bytes, "q_b output") &&
                   sampled_cpu_oracle(output.ptr, model,
                                      model.weights[set].qb_offset,
                                      activation, n_tokens, kQbK, kQbM,
                                      "q_b") &&
                   check_guard(x.ptr, x_bytes, 0x8000u, "q_b input") &&
                   check_guard(output.ptr, out_bytes, 0x9000u,
                               "q_b output");
        }) &&
        check_guard(x.ptr, x_bytes, 0x8000u, "q_b input final") &&
        check_guard(output.ptr, out_bytes, 0x9000u, "q_b output final");
}

bool run_output_a(const model_fixture &model, const config &cfg,
                  uint32_t n_tokens) {
    if (cfg.path == cuda_path::legacy) {
        std::printf(
            "DS4_CUDA_Q4_PREFILL_SKIP case=outa_plus_min_b path=legacy "
            "N=%u reason=grouped_prefill_requires_mmq\n",
            n_tokens);
        std::fflush(stdout);
        return true;
    }

    uint64_t heads_elements = 0;
    uint64_t low_elements = 0;
    uint64_t out_elements = 0;
    if (!checked_mul(n_tokens,
                     static_cast<uint64_t>(kOutputGroups) * kDenseK,
                     &heads_elements) ||
        !checked_mul(n_tokens, kOutputLowDim, &low_elements) ||
        !checked_mul(n_tokens, kOutputMinB, &out_elements)) {
        return false;
    }
    const uint64_t heads_bytes = heads_elements * sizeof(float);
    const uint64_t low_bytes = low_elements * sizeof(float);
    const uint64_t out_bytes = out_elements * sizeof(float);
    const uint64_t group_tmp_bytes =
        static_cast<uint64_t>(n_tokens) * kDenseK * sizeof(float);
    const uint64_t low_tmp_bytes =
        static_cast<uint64_t>(n_tokens) * kOutputRank * sizeof(float);
    const uint64_t guard_bytes = kGuardWords * sizeof(uint32_t);
    tensor_owner heads(heads_bytes + guard_bytes);
    tensor_owner baseline_low(low_bytes + guard_bytes);
    tensor_owner baseline_out(out_bytes + guard_bytes);
    tensor_owner candidate_low(low_bytes + guard_bytes);
    tensor_owner candidate_out(out_bytes + guard_bytes);
    tensor_owner group_tmp(group_tmp_bytes + guard_bytes);
    tensor_owner low_tmp(low_tmp_bytes + guard_bytes);
    std::vector<float> activation;
    fill_activation(&activation, n_tokens, kOutputGroups * kDenseK);
    if (!heads.ptr || !baseline_low.ptr || !baseline_out.ptr ||
        !candidate_low.ptr || !candidate_out.ptr || !group_tmp.ptr ||
        !low_tmp.ptr ||
        !ds4_gpu_tensor_write(heads.ptr, 0, activation.data(), heads_bytes) ||
        !prepare_guard(heads.ptr, heads_bytes, 0xa000u)) {
        std::fprintf(stderr,
                     "outa_plus_min_b N=%u: tensor setup failed\n",
                     n_tokens);
        return false;
    }

    const bool compare_single_grid = cfg.grouped_single_grid;
    const bool compare_q81_kernel = cfg.grouped_q81_kernel;
    const arm baseline = {
        compare_q81_kernel
            ? "grouped_generic_q81"
            : (compare_single_grid
                   ? "grouped_8_grids" : "pack8_mmq_unpack"),
        [&](uint32_t set) {
            return ds4_gpu_attention_output_q4_K_batch_tensor(
                       baseline_out.ptr, baseline_low.ptr, group_tmp.ptr,
                       low_tmp.ptr, model.data, model.size,
                       model.weights[set].output_a_offset,
                       model.weights[set].output_b_offset, kQ4Type,
                       kDenseK, kOutputRank, kOutputGroups, kOutputMinB,
                       heads.ptr, n_tokens) > 0;
        },
        [=]() {
            return compare_q81_kernel
                ? select_grouped_q81_reference()
                : (compare_single_grid
                       ? select_grouped_prefill_grid8()
                       : select_grouped_prefill_legacy());
        }};
    const arm candidate = {
        compare_q81_kernel
            ? "grouped_k4096_g8x2_q81"
            : (compare_single_grid
                   ? "grouped_single_grid" : "grouped_8_grids"),
        [&](uint32_t set) {
            return ds4_gpu_attention_output_q4_K_batch_tensor(
                       candidate_out.ptr, candidate_low.ptr, group_tmp.ptr,
                       low_tmp.ptr, model.data, model.size,
                       model.weights[set].output_a_offset,
                       model.weights[set].output_b_offset, kQ4Type,
                       kDenseK, kOutputRank, kOutputGroups, kOutputMinB,
                       heads.ptr, n_tokens) > 0;
        },
        [=]() {
            return compare_q81_kernel
                ? select_grouped_q81_candidate()
                : (compare_single_grid
                       ? select_grouped_prefill_single_grid()
                       : select_grouped_prefill_grid8());
        }};
    const uint64_t output_a_macs_per_token =
        static_cast<uint64_t>(kOutputGroups) * kDenseK * kOutputRank;
    const uint64_t output_b_macs_per_token =
        static_cast<uint64_t>(kOutputLowDim) * kOutputMinB;
    const bool ok = benchmark_pair_arms(
        "outa_plus_min_b", n_tokens, kDenseK, kOutputLowDim,
        output_a_macs_per_token, output_b_macs_per_token, cfg,
        baseline, candidate,
        [&]() {
            return poison_output(baseline_low.ptr, low_bytes, 0x7fc10001u,
                                 0xb000u) &&
                   poison_output(baseline_out.ptr, out_bytes, 0x7fc20002u,
                                 0xc000u) &&
                   poison_output(candidate_low.ptr, low_bytes, 0x7fc30003u,
                                 0xd000u) &&
                   poison_output(candidate_out.ptr, out_bytes, 0x7fc40004u,
                                 0xe000u) &&
                   prepare_guard(group_tmp.ptr, group_tmp_bytes, 0xf000u) &&
                   prepare_guard(low_tmp.ptr, low_tmp_bytes, 0x11000u);
        },
        [&](uint32_t set) {
            return bitwise_equal(baseline_low.ptr, candidate_low.ptr,
                                 low_bytes,
                                 compare_q81_kernel
                                     ? "output_a specialized vs generic Q8_1"
                                     : (compare_single_grid
                                            ? "output_a single-grid vs eight-grid"
                                            : "output_a grouped vs pack/unpack")) &&
                   bitwise_equal(baseline_out.ptr, candidate_out.ptr,
                                 out_bytes,
                                 "output_a minimal-B final output") &&
                   output_is_finite(baseline_low.ptr, low_bytes,
                                    "output_a low") &&
                   output_is_finite(baseline_out.ptr, out_bytes,
                                    "output_a minimal-B output") &&
                   sampled_grouped_cpu_oracle(
                       baseline_low.ptr, model,
                       model.weights[set].output_a_offset, activation,
                       n_tokens, "output_a") &&
                   check_guard(heads.ptr, heads_bytes, 0xa000u,
                               "output_a heads") &&
                   check_guard(baseline_low.ptr, low_bytes, 0xb000u,
                               "output_a baseline low") &&
                   check_guard(baseline_out.ptr, out_bytes, 0xc000u,
                               "output_a baseline out") &&
                   check_guard(candidate_low.ptr, low_bytes, 0xd000u,
                               "output_a candidate low") &&
                   check_guard(candidate_out.ptr, out_bytes, 0xe000u,
                               "output_a candidate out") &&
                   check_guard(group_tmp.ptr, group_tmp_bytes, 0xf000u,
                               "output_a group scratch") &&
                   check_guard(low_tmp.ptr, low_tmp_bytes, 0x11000u,
                               "output_a low scratch");
        });
    return ok &&
           check_guard(heads.ptr, heads_bytes, 0xa000u,
                       "output_a heads final") &&
           check_guard(baseline_low.ptr, low_bytes, 0xb000u,
                       "output_a baseline low final") &&
           check_guard(baseline_out.ptr, out_bytes, 0xc000u,
                       "output_a baseline out final") &&
           check_guard(candidate_low.ptr, low_bytes, 0xd000u,
                       "output_a candidate low final") &&
           check_guard(candidate_out.ptr, out_bytes, 0xe000u,
                       "output_a candidate out final") &&
           check_guard(group_tmp.ptr, group_tmp_bytes, 0xf000u,
                       "output_a group scratch final") &&
           check_guard(low_tmp.ptr, low_tmp_bytes, 0x11000u,
                       "output_a low scratch final");
}

void usage(FILE *stream, const char *argv0) {
    std::fprintf(
        stream,
        "usage: %s [options]\n\n"
        "Resident CUDA Q4_K prefill kernel benchmark (CUDA event timing).\n\n"
        "  --path mmq|legacy         process-wide path (default: mmq)\n"
        "  --case all|dense|pair|qb|outa|outb\n"
        "                             case to run (default: all)\n"
        "  --tokens N[,N...]         token counts, each 9..8192\n"
        "  --full                    use 9,16,17,31,32,33,127,128,129,256,"
        "257,512,1024,2048,2049,4096,6144,8192\n"
        "  --sets N                  rotating resident weight sets (default: %u)\n"
        "  --samples N               samples/arm, multiple of 4 (default: %u)\n"
        "  --warmup N                untimed dispatches/arm (default: %u)\n"
        "  --grouped-single-grid     compare grouped 8-grid vs grid.z outa\n"
        "  --grouped-q81-kernel      compare generic vs K4096/G8x2 Q8_1 outa\n"
        "  --kernel-16warp           prequantized canonical-vs-16-warp A/B\n"
        "  -h, --help                show this help\n\n"
        "Dense and q_b measure one immutable process path. Run separate "
        "legacy/MMQ\nprocesses (preferably ABBA/BAAB) to compare them because "
        "the CUDA backend\ncaches DS4_CUDA_MMQ on its first dispatch. Pair is "
        "an in-process ABBA/BAAB\ncomparison of two MMQ projections against "
        "the fused public pair API. outa\ncompares the rollback eight-group "
        "pack/MMQ/unpack sequence against the default\ndirect-strided grouped "
        "path with one canonical MMQ grid per group. Add\n"
        "--grouped-single-grid to compare that default with the experimental "
        "grid.z\nsubmission instead. Add\n"
        "--grouped-q81-kernel to isolate the default fixed-shape Q8_1 "
        "front-end against\nthe canonical strided quantizer while retaining "
        "the same eight MMQ grids. Both outa comparisons include a common minimal "
        "Q4 output-B (M=256) whose MACs are reported separately. "
        "DS4_CUDA_MMQ_X_MAX\nmay explicitly select an 8..128 multiple-of-8 "
        "sweep point; the setup line\nattests it, or prints auto when the "
        "variable is unset. --kernel-16warp requires\n--path mmq and "
        "--case dense/pair/qb/outb; with --case all it also runs the real "
        "K=8192,M=4096 output-B. Its "
        "pair arm\nshares one prequantized activation across M=1024+512. Token "
        "counts must be\n>=512 and select canonical m128n128 on the active "
        "device; without --tokens/--full it uses "
        "512,1024,2048,2049,4096,6144,8192.\n",
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

bool env_value_enabled(const char *name) {
    const char *value = std::getenv(name);
    return value && value[0] && !(value[0] == '0' && value[1] == '\0');
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
        result.push_back(parse_u32(item.c_str(), "--tokens", 9u, 8192u));
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
    bool tokens_explicit = false;
    for (int i = 1; i < argc; i++) {
        if (!std::strcmp(argv[i], "-h") || !std::strcmp(argv[i], "--help")) {
            usage(stdout, argv[0]);
            std::exit(0);
        } else if (!std::strcmp(argv[i], "--path")) {
            const char *value = need_value(&i, argc, argv);
            if (!std::strcmp(value, "mmq")) cfg.path = cuda_path::mmq;
            else if (!std::strcmp(value, "legacy")) {
                cfg.path = cuda_path::legacy;
            } else {
                std::fprintf(stderr, "invalid --path: %s\n", value);
                std::exit(2);
            }
        } else if (!std::strcmp(argv[i], "--case")) {
            const char *value = need_value(&i, argc, argv);
            if (!std::strcmp(value, "all")) cfg.selected = bench_case::all;
            else if (!std::strcmp(value, "dense")) {
                cfg.selected = bench_case::dense;
            } else if (!std::strcmp(value, "pair")) {
                cfg.selected = bench_case::pair;
            } else if (!std::strcmp(value, "qb")) {
                cfg.selected = bench_case::qb;
            } else if (!std::strcmp(value, "outa")) {
                cfg.selected = bench_case::outa;
            } else if (!std::strcmp(value, "outb")) {
                cfg.selected = bench_case::outb;
            } else {
                std::fprintf(stderr, "invalid --case: %s\n", value);
                std::exit(2);
            }
        } else if (!std::strcmp(argv[i], "--tokens")) {
            cfg.tokens = parse_tokens(need_value(&i, argc, argv));
            tokens_explicit = true;
        } else if (!std::strcmp(argv[i], "--full")) {
            cfg.tokens = {9u,   16u,  17u,  31u,  32u,  33u, 127u,
                          128u, 129u, 256u, 257u, 512u, 1024u, 2048u,
                          2049u, 4096u, 6144u, 8192u};
            tokens_explicit = true;
        } else if (!std::strcmp(argv[i], "--sets")) {
            cfg.sets = parse_u32(need_value(&i, argc, argv), "--sets", 1u,
                                 32u);
        } else if (!std::strcmp(argv[i], "--samples")) {
            cfg.samples = parse_u32(need_value(&i, argc, argv), "--samples",
                                    4u, 1000u);
        } else if (!std::strcmp(argv[i], "--warmup")) {
            cfg.warmup = parse_u32(need_value(&i, argc, argv), "--warmup",
                                   0u, 100u);
        } else if (!std::strcmp(argv[i], "--kernel-16warp")) {
            cfg.kernel_16warp = true;
        } else if (!std::strcmp(argv[i], "--grouped-single-grid")) {
            cfg.grouped_single_grid = true;
        } else if (!std::strcmp(argv[i], "--grouped-q81-kernel")) {
            cfg.grouped_q81_kernel = true;
        } else {
            std::fprintf(stderr, "unknown option: %s\n", argv[i]);
            usage(stderr, argv[0]);
            std::exit(2);
        }
    }
    if ((cfg.samples % 4u) != 0u) {
        std::fprintf(stderr,
                     "--samples must be a multiple of 4 for balanced runs\n");
        std::exit(2);
    }
    if (cfg.kernel_16warp && !tokens_explicit) {
        cfg.tokens = {512u, 1024u, 2048u, 2049u, 4096u, 6144u, 8192u};
    }
    if (cfg.grouped_q81_kernel && !tokens_explicit) {
        cfg.tokens = {512u, 1024u, 2048u, 4096u, 6144u, 8192u};
    }
    if (cfg.kernel_16warp) {
        const auto below_minimum = std::find_if(
            cfg.tokens.begin(), cfg.tokens.end(),
            [](uint32_t value) { return value < 512u; });
        if (below_minimum != cfg.tokens.end()) {
            std::fprintf(stderr,
                         "--kernel-16warp requires every token count to be "
                         ">=512 (got %u)\n",
                         *below_minimum);
            std::exit(2);
        }
    }
    if (cfg.kernel_16warp && cfg.path != cuda_path::mmq) {
        std::fprintf(stderr,
                     "--kernel-16warp requires --path mmq\n");
        std::exit(2);
    }
    if (cfg.grouped_single_grid && cfg.path != cuda_path::mmq) {
        std::fprintf(stderr,
                     "--grouped-single-grid requires --path mmq\n");
        std::exit(2);
    }
    if (cfg.grouped_q81_kernel && cfg.path != cuda_path::mmq) {
        std::fprintf(stderr,
                     "--grouped-q81-kernel requires --path mmq\n");
        std::exit(2);
    }
    if (cfg.grouped_single_grid && cfg.kernel_16warp) {
        std::fprintf(stderr,
                     "--grouped-single-grid cannot be combined with "
                     "--kernel-16warp\n");
        std::exit(2);
    }
    if (cfg.grouped_q81_kernel &&
        (cfg.grouped_single_grid || cfg.kernel_16warp)) {
        std::fprintf(stderr,
                     "--grouped-q81-kernel cannot be combined with "
                     "--grouped-single-grid or --kernel-16warp\n");
        std::exit(2);
    }
    if (cfg.grouped_single_grid && cfg.selected != bench_case::all &&
        cfg.selected != bench_case::outa) {
        std::fprintf(stderr,
                     "--grouped-single-grid requires --case outa or all\n");
        std::exit(2);
    }
    if (cfg.grouped_q81_kernel && cfg.selected != bench_case::all &&
        cfg.selected != bench_case::outa) {
        std::fprintf(stderr,
                     "--grouped-q81-kernel requires --case outa or all\n");
        std::exit(2);
    }
    if (cfg.kernel_16warp && cfg.selected == bench_case::outa) {
        std::fprintf(stderr,
                     "--kernel-16warp supports only --case dense, pair, qb, "
                     "outb, or all\n");
        std::exit(2);
    }
    if (!cfg.kernel_16warp && cfg.selected == bench_case::outb) {
        std::fprintf(stderr,
                     "--case outb requires --kernel-16warp\n");
        std::exit(2);
    }
    if (cfg.path == cuda_path::legacy &&
        (cfg.selected == bench_case::pair ||
         cfg.selected == bench_case::outa)) {
        std::fprintf(stderr,
                     "--case pair/outa requires --path mmq for prefill "
                     "N > 8\n");
        std::exit(2);
    }
    return cfg;
}

bool includes(bench_case selected, bench_case wanted) {
    return selected == bench_case::all || selected == wanted;
}

std::string mmq_x_max_attestation(bool require_m128n128) {
    const char *value = std::getenv("DS4_CUDA_MMQ_X_MAX");
    if (!value || !value[0]) return "auto";
    const uint32_t parsed =
        parse_u32(value, "DS4_CUDA_MMQ_X_MAX", 8u, 128u);
    if ((parsed % 8u) != 0u) {
        std::fprintf(stderr,
                     "invalid DS4_CUDA_MMQ_X_MAX: %s (must be a multiple "
                     "of 8)\n",
                     value);
        std::exit(2);
    }
    if (require_m128n128 && parsed != 128u) {
        std::fprintf(stderr,
                     "--kernel-16warp requires DS4_CUDA_MMQ_X_MAX=128 "
                     "when the variable is set (got %s)\n",
                     value);
        std::exit(2);
    }
    return std::to_string(parsed);
}

bool install_resident_model(const model_fixture &model,
                            size_t *resident_delta,
                            bool *resident_delta_valid) {
    size_t free_before = 0, total_before = 0;
    size_t free_after = 0, total_after = 0;
    const bool have_before =
        cudaMemGetInfo(&free_before, &total_before) == cudaSuccess;
    if (!have_before) (void)cudaGetLastError();
    if (!ds4_gpu_set_model_map(model.data, model.size) ||
        !ds4_gpu_synchronize()) {
        return false;
    }
    const bool have_after =
        cudaMemGetInfo(&free_after, &total_after) == cudaSuccess;
    if (!have_after) (void)cudaGetLastError();
    (void)total_before;
    (void)total_after;
    *resident_delta_valid = have_before && have_after;
    *resident_delta = *resident_delta_valid && free_before >= free_after
        ? free_before - free_after : 0u;
    return true;
}

bool verify_resident_weight_ranges(const model_fixture &model) {
    const uint64_t dense_bytes = q4_weight_bytes(kDenseK, kDenseM);
    const uint64_t kv_bytes = q4_weight_bytes(kDenseK, kKvM);
    const uint64_t qb_bytes = q4_weight_bytes(kQbK, kQbM);
    const uint64_t output_a_bytes =
        q4_weight_bytes(kDenseK, kOutputLowDim);
    const uint64_t output_b_bytes =
        q4_weight_bytes(kOutputLowDim, model.output_b_rows);
    for (uint32_t set = 0; set < model.weights.size(); set++) {
        struct range_desc {
            const char *name;
            uint64_t offset;
            uint64_t bytes;
        };
        const range_desc ranges[] = {
            {"dense", model.weights[set].dense_offset, dense_bytes},
            {"kv", model.weights[set].kv_offset, kv_bytes},
            {"q_b", model.weights[set].qb_offset, qb_bytes},
            {"output_a", model.weights[set].output_a_offset, output_a_bytes},
            {model.output_b_rows == kOutputMinB
                 ? "output_b_min" : "output_b",
             model.weights[set].output_b_offset,
             output_b_bytes},
        };
        for (const range_desc &range : ranges) {
            if (!ds4_cuda_test_model_range_is_device_resident(
                    model.data, model.size, range.offset, range.bytes, 0)) {
                std::fprintf(
                    stderr,
                    "cuda-q4-prefill-bench: nonresident %s weight range "
                    "set=%u offset=%llu bytes=%llu\n",
                    range.name, set,
                    static_cast<unsigned long long>(range.offset),
                    static_cast<unsigned long long>(range.bytes));
                return false;
            }
        }
    }
    return true;
}

bool verify_mmq_prefill_dispatch(const model_fixture &model) {
    // For N > 8 the public pair API succeeds only through MMQ. Non-required
    // production runs use this small probe to initialize and attest the
    // process-wide decision. The caller skips it for raw-kernel and required
    // 16-warp runs, whose measured dispatches provide their own fail-closed
    // proof at an eligible shape.
    constexpr uint32_t n_tokens = 9u;
    const uint64_t x_bytes = static_cast<uint64_t>(n_tokens) * kDenseK *
                             sizeof(float);
    const uint64_t out0_bytes = static_cast<uint64_t>(n_tokens) * kDenseM *
                                sizeof(float);
    const uint64_t out1_bytes = static_cast<uint64_t>(n_tokens) * kKvM *
                                sizeof(float);
    tensor_owner x(x_bytes);
    tensor_owner out0(out0_bytes);
    tensor_owner out1(out1_bytes);
    std::vector<float> activation;
    fill_activation(&activation, n_tokens, kDenseK);
    return x.ptr && out0.ptr && out1.ptr &&
           ds4_gpu_tensor_write(x.ptr, 0, activation.data(), x_bytes) &&
           ds4_gpu_matmul_q4_K_pair_tensor(
               out0.ptr, out1.ptr, model.data, model.size,
               model.weights[0].dense_offset, model.weights[0].kv_offset,
               kDenseK, kDenseM, kKvM, x.ptr, n_tokens) == 1 &&
           ds4_gpu_synchronize();
}

}  // namespace

int main(int argc, char **argv) {
    const config cfg = parse_options(argc, argv);
    // get_mmq_x_max_host() caches this process-wide on its first call.  Read
    // and validate the inherited sweep request before backend initialization,
    // then attest it in the setup record instead of silently contaminating a
    // supposedly default run.
    const std::string mmq_x_max =
        mmq_x_max_attestation(cfg.kernel_16warp);
    env_snapshot mmq_guard("DS4_CUDA_MMQ");
    env_snapshot copy_guard("DS4_CUDA_COPY_MODEL");
    env_snapshot pair_guard("DS4_CUDA_DISABLE_Q4_DENSE_PAIR");
    env_snapshot graph_guard("DS4_CUDA_DECODE_GRAPHS");
    env_snapshot grouped_enable_guard(kGroupedPrefillEnable);
    env_snapshot grouped_disable_guard(kGroupedPrefillDisable);
    env_snapshot grouped_require_guard(kGroupedPrefillRequire);
    env_snapshot grouped_single_grid_enable_guard(kGroupedSingleGridEnable);
    env_snapshot grouped_single_grid_disable_guard(kGroupedSingleGridDisable);
    env_snapshot grouped_single_grid_require_guard(kGroupedSingleGridRequire);
    env_snapshot grouped_q81_disable_guard(kGroupedQ81Disable);
    env_snapshot grouped_q81_require_guard(kGroupedQ81Require);
    env_snapshot grouped_global_disable_guard(kGroupedGlobalDisable);
    env_snapshot gb10_global_disable_guard(kGb10GlobalDisable);
    if (setenv("DS4_CUDA_MMQ",
               cfg.path == cuda_path::mmq ? "1" : "0", 1) != 0 ||
        setenv("DS4_CUDA_COPY_MODEL", "1", 1) != 0 ||
        setenv("DS4_CUDA_DECODE_GRAPHS", "0", 1) != 0 ||
        unsetenv("DS4_CUDA_DISABLE_Q4_DENSE_PAIR") != 0 ||
        unsetenv(kGroupedPrefillEnable) != 0 ||
        unsetenv(kGroupedPrefillDisable) != 0 ||
        unsetenv(kGroupedPrefillRequire) != 0 ||
        unsetenv(kGroupedSingleGridEnable) != 0 ||
        unsetenv(kGroupedSingleGridDisable) != 0 ||
        unsetenv(kGroupedSingleGridRequire) != 0 ||
        unsetenv(kGroupedQ81Disable) != 0 ||
        unsetenv(kGroupedQ81Require) != 0 ||
        unsetenv(kGroupedGlobalDisable) != 0 ||
        unsetenv(kGb10GlobalDisable) != 0) {
        std::fprintf(stderr,
                     "cuda-q4-prefill-bench: environment setup failed\n");
        return 1;
    }

    int device_count = 0;
    const cudaError_t count_rc = cudaGetDeviceCount(&device_count);
    if (count_rc != cudaSuccess || device_count <= 0) {
        std::fprintf(stderr,
                     "cuda-q4-prefill-bench: no visible CUDA device (%s)\n",
                     count_rc == cudaSuccess ? "device count is zero"
                                             : cudaGetErrorString(count_rc));
        return 77;
    }
    cudaDeviceProp properties{};
    if (cudaGetDeviceProperties(&properties, 0) != cudaSuccess) {
        std::fprintf(stderr,
                     "cuda-q4-prefill-bench: cannot query device properties\n");
        return 1;
    }
    if (cfg.kernel_16warp && device_count != 1) {
        std::fprintf(stderr,
                     "cuda-q4-prefill-bench: SKIP (--kernel-16warp "
                     "requires exactly one visible CUDA device)\n");
        return 77;
    }
    const bool grouped_prefill_supported =
        device_count == 1 && properties.major == 12 &&
        properties.minor == 1 && properties.warpSize == 32;
    if (cfg.selected == bench_case::outa &&
        !grouped_prefill_supported) {
        std::fprintf(stderr,
                     "cuda-q4-prefill-bench: SKIP (outa requires one "
                     "GB10/sm_121 device)\n");
        return 77;
    }
    if (!ds4_gpu_init()) {
        std::fprintf(stderr, "cuda-q4-prefill-bench: ds4_gpu_init failed\n");
        return 1;
    }
    if (cfg.kernel_16warp) {
        const int prepare_rc = ds4_mmq_q4_K_dense_16warp_prepare();
        if (prepare_rc != 0) {
            std::fprintf(
                stderr,
                "cuda-q4-prefill-bench: --kernel-16warp prepare failed "
                "rc=%d\n",
                prepare_rc);
            ds4_gpu_cleanup();
            return 1;
        }
        for (uint32_t tokens : cfg.tokens) {
            if (!ds4_mmq_q4_K_dense_preq_reference_m128n128_for_test(
                    static_cast<int>(tokens))) {
                std::fprintf(
                    stderr,
                    "cuda-q4-prefill-bench: SKIP (--kernel-16warp N=%u "
                    "does not select canonical m128n128 on this device)\n",
                    tokens);
                ds4_gpu_cleanup();
                return 77;
            }
        }
    }
    ds4_cuda_test_set_q4_mmq_strict(
        cfg.path == cuda_path::mmq ? 1 : 0);

    bool ok = true;
    model_fixture model;
    const uint32_t output_b_rows =
        cfg.kernel_16warp && includes(cfg.selected, bench_case::outb)
            ? kOutputM : kOutputMinB;
    if (!make_model(&model, cfg.sets, output_b_rows)) {
        std::fprintf(stderr,
                     "cuda-q4-prefill-bench: model fixture allocation failed\n");
        ok = false;
    }
    size_t resident_delta = 0;
    bool resident_delta_valid = false;
    if (ok) {
        ds4_gpu_set_quality(false);
        ds4_gpu_set_ssd_streaming(false);
        if (!install_resident_model(
                model, &resident_delta, &resident_delta_valid)) {
            std::fprintf(
                stderr,
                "cuda-q4-prefill-bench: model-map installation failed\n");
            ok = false;
        } else if (!verify_resident_weight_ranges(model)) {
            std::fprintf(
                stderr,
                "cuda-q4-prefill-bench: explicit CUDA model provenance "
                "check failed; refusing PCIe/HMM-contaminated timings\n");
            ok = false;
        }
    }
    // The N=9 probe is intentionally outside the 16-warp admission envelope.
    // Kernel-only runs prove their raw path directly; required production runs
    // fail closed at each measured dispatch, so probing here would be a false
    // failure before the requested shape is reached.
    const bool skip_mmq_probe = cfg.kernel_16warp || env_value_enabled(
        "DS4_CUDA_REQUIRE_Q4_MMQ_16WARP");
    if (ok && cfg.path == cuda_path::mmq && !skip_mmq_probe &&
        !verify_mmq_prefill_dispatch(model)) {
        std::fprintf(stderr,
                     "cuda-q4-prefill-bench: MMQ prefill proof probe failed; "
                     "refusing to label fallback timings as MMQ\n");
        ok = false;
    }

    if (ok) {
        std::printf(
            "DS4_CUDA_Q4_PREFILL_SETUP device=%s cc=%d.%d warp=%d path=%s "
            "mmq_x_max=%s sets=%u resident_payload_mib=%.2f "
            "device_free_delta_mib=%.2f "
            "device_free_delta_valid=%d timing=%s kernel_16warp=%d "
            "grouped_q81_kernel=%d "
            "cases=%s "
            "ssd_streaming=off model_storage=cudaMalloc "
            "residency=backend_provenance strict_mmq=%d "
            "grouped_attn_a_prefill=%s grouped_attn_a_ab=%s "
            "dispatch_stream=legacy_default\n",
            properties.name, properties.major, properties.minor,
            properties.warpSize, path_name(cfg.path), mmq_x_max.c_str(),
            cfg.sets,
            static_cast<double>(model.payload_bytes) / 1048576.0,
            static_cast<double>(resident_delta) / 1048576.0,
            resident_delta_valid ? 1 : 0,
            cfg.kernel_16warp ? "kernel_only_prequant" : "cuda_events",
            cfg.kernel_16warp ? 1 : 0,
            cfg.grouped_q81_kernel ? 1 : 0,
            case_scope(cfg),
            cfg.path == cuda_path::mmq ? 1 : 0,
            grouped_prefill_supported ? "available" : "skipped",
            cfg.grouped_q81_kernel
                ? "q81_generic_vs_k4096_g8x2"
                : (cfg.grouped_single_grid
                       ? "grid8_vs_single_grid" : "pack8_vs_grid8"));
        std::fflush(stdout);
        for (uint32_t n_tokens : cfg.tokens) {
            if (includes(cfg.selected, bench_case::dense)) {
                ok = (cfg.kernel_16warp
                          ? run_q4_16warp_kernel(
                                model, cfg, n_tokens, kDenseK, kDenseM,
                                &weight_set::dense_offset, "dense")
                          : run_dense(model, cfg, n_tokens)) && ok;
            }
            if (ok && includes(cfg.selected, bench_case::pair)) {
                ok = (cfg.kernel_16warp
                          ? run_q4_16warp_pair_kernel(
                                model, cfg, n_tokens)
                          : run_pair(model, cfg, n_tokens)) && ok;
            }
            if (ok && includes(cfg.selected, bench_case::qb)) {
                ok = (cfg.kernel_16warp
                          ? run_q4_16warp_kernel(
                                model, cfg, n_tokens, kQbK, kQbM,
                                &weight_set::qb_offset, "q_b")
                          : run_qb(model, cfg, n_tokens)) && ok;
            }
            if (ok && cfg.kernel_16warp &&
                includes(cfg.selected, bench_case::outb)) {
                ok = run_q4_16warp_kernel(
                         model, cfg, n_tokens, kOutputLowDim, kOutputM,
                         &weight_set::output_b_offset, "output_b") && ok;
            }
            if (ok && !cfg.kernel_16warp && grouped_prefill_supported &&
                includes(cfg.selected, bench_case::outa)) {
                ok = run_output_a(model, cfg, n_tokens) && ok;
            }
            if (!ok) break;
        }
    }

    ds4_gpu_cleanup();
    std::fprintf(stderr, "cuda-q4-prefill-bench: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
