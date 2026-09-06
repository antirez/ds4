// SPDX-License-Identifier: MIT
// Deterministic ROCm Q4_K dense/pair/tiled-prefill oracle.
//
// The test deliberately goes through the public tensor/model-map API.  Weight
// rows use the raw 144-byte GGUF Q4_K layout, while the CPU reference mirrors
// the backend's F32 -> Q8_K quantizer and Q4_K x Q8_K integer dot product.
// Prefill controls are forced through the rollback path before the TILE8
// REQUIRE path so a future default promotion cannot turn parity into a
// candidate-vs-candidate false green.

#include "ds4_gpu.h"

#if defined(__has_include)
#  if __has_include(<hip/hip_runtime.h>)
#    include <hip/hip_runtime.h>
#    define DS4_TEST_HAS_HIP_RUNTIME 1
#  endif
#endif
#ifndef DS4_TEST_HAS_HIP_RUNTIME
#  define DS4_TEST_HAS_HIP_RUNTIME 0
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <string>
#include <sys/mman.h>
#include <vector>

extern "C" int ds4_rocm_test_q4_prefill_k1024_tile4_policy(
    int ssd_streaming, int weight_device_resident, int ssd_enabled,
    int disabled, int required);
extern "C" void ds4_rocm_test_q4_prefill_wmma_reset(void);
extern "C" uint64_t ds4_rocm_test_q4_prefill_wmma_get_calls(void);
extern "C" uint64_t ds4_rocm_test_q4_prefill_wmma_k64_get_calls(void);
extern "C" uint64_t ds4_rocm_test_q4_prefill_wmma_k128_get_calls(void);
extern "C" uint64_t ds4_rocm_test_q4_prefill_wmma_k64_load4_get_calls(void);
extern "C" int ds4_rocm_test_q4_prefill_wmma_k64_control_policy(
    int control);
extern "C" int ds4_rocm_test_q4_prefill_wmma_k128_policy(
    int disabled, int k64_enabled, uint32_t row_tile,
    int load4_compatible);
extern "C" int ds4_rocm_test_q4_prefill_wmma_requested_policy(
    int ssd_streaming, int enabled, int ssd_enabled, int disabled,
    int required);
extern "C" int
ds4_rocm_test_q4_prefill_wmma_attention_a_requested_policy(
    int ssd_streaming, int enabled, int ssd_enabled, int disabled,
    int required);
extern "C" int
ds4_rocm_test_q4_prefill_wmma_attention_b_requested_policy(
    int ssd_streaming, int enabled, int ssd_enabled, int disabled,
    int required);
extern "C" int ds4_rocm_test_q4_prefill_wmma_yields_to_q8_wave32(
    int q8_selected, int wmma_required);
extern "C" uint32_t ds4_rocm_test_q4_prefill_wmma_row_tile(
    uint32_t out_dim);
extern "C" int ds4_rocm_test_q4_prefill_q8_wave32_policy(
    int prefill_scope, int runtime_compatible, int enabled, int disabled,
    int required);
extern "C" void ds4_rocm_test_q4_prefill_q8_wave32_reset(void);
extern "C" uint64_t ds4_rocm_test_q4_prefill_q8_wave32_get_calls(void);
extern "C" int ds4_rocm_test_q8_K_quantize_tensor(
    ds4_gpu_tensor *out, const ds4_gpu_tensor *x, uint32_t in_dim,
    uint32_t n_rows, int use_wave32);
extern "C" int ds4_rocm_test_q4_attn_q_b_yield_to_q8_wave32_policy(
    uint32_t weight_type, uint32_t n_tok, int q8_wave32_required,
    int f16_cache_required);
extern "C" int ds4_rocm_test_q4_pair_pre_enqueue_failure_policy(
    int prefill_scope, int tile8_required, int q8_wave32_required);
extern "C" void ds4_rocm_test_q4_prefill_lds_stream_reset(void);
extern "C" uint64_t ds4_rocm_test_q4_prefill_lds_stream_get_calls(void);
extern "C" void ds4_rocm_test_q4_prefill_lds_vector_reset(void);
extern "C" uint64_t ds4_rocm_test_q4_prefill_lds_vector_get_calls(void);

namespace {

constexpr uint32_t kQkK = 256u;
constexpr uint32_t kK = 4096u;
constexpr uint32_t kM0 = 65u;
constexpr uint32_t kM1 = 33u;
constexpr uint32_t kQ4Type = 12u;
constexpr uint32_t kQ8Type = 8u;
constexpr uint32_t kTailK = 1024u;
constexpr uint32_t kQbOutDim = 32768u;
constexpr uint32_t kQbHeads = 64u;
constexpr uint32_t kQbHeadDim = 512u;
constexpr uint32_t kQbRot = 64u;
constexpr uint32_t kAttnGroupDim = 4096u;
constexpr uint32_t kAttnRank = 32u;
constexpr uint32_t kAttnGroups = 8u;
constexpr uint32_t kAttnLowDim = kAttnGroups * kAttnRank;
constexpr uint32_t kAttnOutDim = 65u;
constexpr uint32_t kDecodeAttnGroupDim = 4096u;
constexpr uint32_t kDecodeAttnRank = 1024u;
constexpr uint32_t kDecodeAttnGroups = 8u;
constexpr uint32_t kDecodeAttnLowDim =
    kDecodeAttnGroups * kDecodeAttnRank;
constexpr uint32_t kDecodeAttnOutDim = 4096u;
constexpr size_t kOutputGuardFloats = 257u;
constexpr float kCpuAbsTolerance = 2.0e-3f;
constexpr float kCpuRelTolerance = 3.0e-5f;
constexpr int kSkip = 77;

constexpr const char *kPrefillEnable =
    "DS4_ROCM_ENABLE_Q4_PREFILL_TILE8";
constexpr const char *kPrefillDisable =
    "DS4_ROCM_DISABLE_Q4_PREFILL_TILE8";
constexpr const char *kPrefillRequire =
    "DS4_ROCM_REQUIRE_Q4_PREFILL_TILE8";
constexpr const char *kPrefillLdsDisable =
    "DS4_ROCM_DISABLE_Q4_PREFILL_LDS_STREAM";
constexpr const char *kPrefillLdsVectorDisable =
    "DS4_ROCM_DISABLE_Q4_PREFILL_LDS_VECTOR";
constexpr const char *kPrefillK1024Tile4Disable =
    "DS4_ROCM_DISABLE_Q4_PREFILL_K1024_TILE4";
constexpr const char *kPrefillK1024Tile4SsdEnable =
    "DS4_ROCM_ENABLE_Q4_PREFILL_K1024_TILE4_SSD";
constexpr const char *kPrefillK1024Tile4Require =
    "DS4_ROCM_REQUIRE_Q4_PREFILL_K1024_TILE4";
constexpr const char *kPrefillWmmaEnable =
    "DS4_ROCM_ENABLE_Q4_PREFILL_WMMA";
constexpr const char *kPrefillWmmaSsdEnable =
    "DS4_ROCM_ENABLE_Q4_PREFILL_WMMA_SSD";
constexpr const char *kPrefillWmmaDisable =
    "DS4_ROCM_DISABLE_Q4_PREFILL_WMMA";
constexpr const char *kPrefillWmmaRequire =
    "DS4_ROCM_REQUIRE_Q4_PREFILL_WMMA";
constexpr const char *kPrefillWmmaRowTile =
    "DS4_ROCM_Q4_PREFILL_WMMA_ROW_TILE";
constexpr const char *kPrefillWmmaK64 =
    "DS4_ROCM_ENABLE_Q4_PREFILL_WMMA_K64";
constexpr const char *kPrefillWmmaK128Disable =
    "DS4_ROCM_DISABLE_Q4_PREFILL_WMMA_K128";
constexpr const char *kPrefillWmmaK64Load4Disable =
    "DS4_ROCM_DISABLE_Q4_PREFILL_WMMA_K64_LOAD4";
constexpr const char *kPrefillQ8Wave32Enable =
    "DS4_ROCM_ENABLE_Q4_PREFILL_Q8_K_WAVE32";
constexpr const char *kPrefillQ8Wave32Disable =
    "DS4_ROCM_DISABLE_Q4_PREFILL_Q8_K_WAVE32";
constexpr const char *kPrefillQ8Wave32Require =
    "DS4_ROCM_REQUIRE_Q4_PREFILL_Q8_K_WAVE32";
constexpr const char *kQbF16Enable =
    "DS4_ROCM_ENABLE_Q4_ATTN_Q_B_F16_CACHE";
constexpr const char *kQbF16Disable =
    "DS4_ROCM_DISABLE_Q4_ATTN_Q_B_F16_CACHE";
constexpr const char *kQbF16Require =
    "DS4_ROCM_REQUIRE_Q4_ATTN_Q_B_F16_CACHE";
constexpr const char *kQbF16MinTokens =
    "DS4_ROCM_Q4_ATTN_Q_B_F16_CACHE_MIN_TOKENS";
constexpr const char *kQbF16OutputEnable =
    "DS4_ROCM_ENABLE_Q4_ATTN_Q_B_F16_OUTPUT";
constexpr const char *kGroupedDecodeEnable =
    "DS4_ROCM_ENABLE_Q4_GROUPED_ATTN_A";
constexpr const char *kGroupedDecodeDisable =
    "DS4_ROCM_DISABLE_Q4_GROUPED_ATTN_A";
constexpr const char *kGroupedDecodeRequire =
    "DS4_ROCM_REQUIRE_Q4_GROUPED_ATTN_A";
constexpr const char *kGroupedDecodeStats =
    "DS4_ROCM_Q4_GROUPED_ATTN_A_STATS";

struct block_q4_K_test {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[12];
    uint8_t qs[kQkK / 2u];
};

struct block_q8_K_test {
    float d;
    int8_t qs[kQkK];
    int16_t bsums[kQkK / 16u];
};

struct block_q8_0_test {
    uint16_t d;
    int8_t qs[32];
};

static_assert(sizeof(block_q4_K_test) == 144u,
              "Q4_K fixture must match the raw GGUF layout");
static_assert(sizeof(block_q8_K_test) == 292u,
              "Q8_K oracle must match the ROCm activation layout");
static_assert(sizeof(block_q8_0_test) == 34u,
              "Q8_0 fixture must match the raw GGUF layout");

struct tensor_owner {
    ds4_gpu_tensor *ptr = nullptr;

    explicit tensor_owner(uint64_t bytes) : ptr(ds4_gpu_tensor_alloc(bytes)) {}
    explicit tensor_owner(ds4_gpu_tensor *owned) : ptr(owned) {}
    ~tensor_owner() { ds4_gpu_tensor_free(ptr); }

    tensor_owner(const tensor_owner &) = delete;
    tensor_owner &operator=(const tensor_owner &) = delete;
};

struct aligned_model {
    uint8_t *data = nullptr;
    uint64_t size = 0;
    uint64_t weight0_offset = 0;
    uint64_t weight1_offset = 0;
    uint64_t attn_a_offset = 0;
    uint64_t decode_attn_a_offset = 0;
    uint64_t attn_b_offset = 0;
    uint64_t attn_b_q8_offset = 0;
    uint64_t tail_k1024_offset = 0;
    uint64_t tail_k1024_pair_offset = 0;
    uint64_t q_b_k1024_offset = 0;
    uint64_t decode_attn_b_offset = 0;

    ~aligned_model() { std::free(data); }

    aligned_model(const aligned_model &) = delete;
    aligned_model &operator=(const aligned_model &) = delete;
    aligned_model() = default;
};

struct env_snapshot {
    const char *name;
    bool was_set;
    std::string value;

    explicit env_snapshot(const char *key)
        : name(key), was_set(std::getenv(key) != nullptr),
          value(was_set ? std::getenv(key) : "") {}
    ~env_snapshot() {
        if (was_set) {
            (void)setenv(name, value.c_str(), 1);
        } else {
            (void)unsetenv(name);
        }
    }

    env_snapshot(const env_snapshot &) = delete;
    env_snapshot &operator=(const env_snapshot &) = delete;
};

uint64_t round_up(uint64_t value, uint64_t alignment) {
    return (value + alignment - 1u) / alignment * alignment;
}

uint32_t lcg_next(uint32_t &state) {
    state = state * 1664525u + 1013904223u;
    return state;
}

float fp16_to_float(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16u;
    uint32_t exp = (h >> 10u) & 0x1fu;
    uint32_t mant = h & 0x3ffu;
    uint32_t bits;
    if (exp == 0u) {
        if (mant == 0u) {
            bits = sign;
        } else {
            int shift = 0;
            while ((mant & 0x400u) == 0u) {
                mant <<= 1u;
                shift++;
            }
            mant &= 0x3ffu;
            bits = sign | (uint32_t)(127 - 14 - shift) << 23u | mant << 13u;
        }
    } else if (exp == 31u) {
        bits = sign | 0x7f800000u | mant << 13u;
    } else {
        bits = sign | (exp + 112u) << 23u | mant << 13u;
    }
    float out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

uint16_t float_to_fp16(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = bits >> 31u;
    int32_t exp = (int32_t)((bits >> 23u) & 0xffu) - 127 + 15;
    uint32_t mant = bits & 0x7fffffu;
    if (exp >= 31) return (uint16_t)((sign << 15u) | 0x7c00u);
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)(sign << 15u);
        mant |= 0x800000u;
        const uint32_t shift = (uint32_t)(14 - exp);
        uint32_t rounded = mant >> shift;
        const uint32_t halfway = 1u << (shift - 1u);
        if ((mant & halfway) &&
            ((mant & (halfway - 1u)) || (rounded & 1u))) {
            rounded++;
        }
        return (uint16_t)((sign << 15u) | rounded);
    }
    uint32_t rounded = mant + 0x0fffu + ((mant >> 13u) & 1u);
    if (rounded & 0x800000u) {
        rounded = 0u;
        exp++;
        if (exp >= 31) return (uint16_t)((sign << 15u) | 0x7c00u);
    }
    return (uint16_t)((sign << 15u) | (uint32_t)exp << 10u |
                      rounded >> 13u);
}

void q4_scale_min(uint32_t j, const uint8_t *scales,
                  uint8_t *scale, uint8_t *minimum) {
    if (j < 4u) {
        *scale = scales[j] & 63u;
        *minimum = scales[j + 4u] & 63u;
    } else {
        *scale = (scales[j + 4u] & 0x0fu) |
                 (uint8_t)((scales[j - 4u] >> 6u) << 4u);
        *minimum = (scales[j + 4u] >> 4u) |
                   (uint8_t)((scales[j] >> 6u) << 4u);
    }
}

void fill_q4_rows(block_q4_K_test *rows, uint32_t n_rows,
                  uint32_t in_dim, uint32_t seed) {
    uint32_t state = seed;
    const uint32_t blocks_per_row = in_dim / kQkK;
    for (uint32_t row = 0; row < n_rows; row++) {
        for (uint32_t b = 0; b < blocks_per_row; b++) {
            block_q4_K_test &block = rows[(uint64_t)row * blocks_per_row + b];
            const float d = 0.0025f +
                0.00025f * (float)(1u + (lcg_next(state) % 23u));
            const float dmin = 0.0010f +
                0.00020f * (float)(1u + (lcg_next(state) % 19u));
            block.d = float_to_fp16(d);
            block.dmin = float_to_fp16(dmin);
            for (uint8_t &v : block.scales) v = (uint8_t)(lcg_next(state) >> 24u);
            for (uint8_t &v : block.qs) v = (uint8_t)(lcg_next(state) >> 24u);
        }
    }
}

void fill_q8_0_rows(block_q8_0_test *rows, uint32_t n_rows,
                    uint32_t in_dim, uint32_t seed) {
    uint32_t state = seed;
    const uint32_t blocks_per_row = in_dim / 32u;
    for (uint32_t row = 0; row < n_rows; row++) {
        for (uint32_t b = 0; b < blocks_per_row; b++) {
            block_q8_0_test &block =
                rows[(uint64_t)row * blocks_per_row + b];
            const float scale = 0.0015f +
                0.000125f * (float)(1u + (lcg_next(state) % 29u));
            block.d = float_to_fp16(scale);
            for (int8_t &q : block.qs) {
                q = (int8_t)((int)(lcg_next(state) % 255u) - 127);
            }
        }
    }
}

bool make_model(aligned_model *model) {
    constexpr uint64_t page = 4096u;
    const uint64_t row_bytes = (kK / kQkK) * sizeof(block_q4_K_test);
    const uint64_t weight0_bytes = kM0 * row_bytes;
    const uint64_t weight1_bytes = kM1 * row_bytes;
    const uint64_t attn_a_row_bytes =
        (kAttnGroupDim / kQkK) * sizeof(block_q4_K_test);
    const uint64_t attn_a_bytes =
        (uint64_t)kAttnGroups * kAttnRank * attn_a_row_bytes;
    const uint64_t decode_attn_a_row_bytes =
        (kDecodeAttnGroupDim / kQkK) * sizeof(block_q4_K_test);
    const uint64_t decode_attn_a_group_bytes =
        (uint64_t)kDecodeAttnRank * decode_attn_a_row_bytes;
    const uint64_t decode_attn_a_bytes =
        (uint64_t)kDecodeAttnGroups * decode_attn_a_group_bytes;
    const uint64_t attn_b_row_bytes =
        (kAttnLowDim / kQkK) * sizeof(block_q4_K_test);
    const uint64_t attn_b_bytes =
        (uint64_t)kAttnOutDim * attn_b_row_bytes;
    const uint64_t tail_row_bytes =
        (kTailK / kQkK) * sizeof(block_q4_K_test);
    const uint64_t tail0_bytes = (uint64_t)kM0 * tail_row_bytes;
    const uint64_t tail1_bytes = (uint64_t)kM1 * tail_row_bytes;
    const uint64_t q_b_k1024_bytes =
        (uint64_t)kQbOutDim * tail_row_bytes;
    const uint64_t decode_attn_b_row_bytes =
        (kDecodeAttnLowDim / kQkK) * sizeof(block_q4_K_test);
    const uint64_t decode_attn_b_bytes =
        (uint64_t)kDecodeAttnOutDim * decode_attn_b_row_bytes;
    const uint64_t attn_b_q8_row_bytes =
        (kAttnLowDim / 32u) * sizeof(block_q8_0_test);
    const uint64_t attn_b_q8_bytes =
        (uint64_t)kAttnOutDim * attn_b_q8_row_bytes;
    model->weight0_offset = 0u;
    model->weight1_offset = round_up(weight0_bytes, page);
    model->attn_a_offset = round_up(
        model->weight1_offset + weight1_bytes, page);
    model->decode_attn_a_offset = round_up(
        model->attn_a_offset + attn_a_bytes, page);
    model->attn_b_offset = round_up(
        model->decode_attn_a_offset + decode_attn_a_bytes, page);
    model->tail_k1024_offset = round_up(
        model->attn_b_offset + attn_b_bytes, page);
    model->tail_k1024_pair_offset = round_up(
        model->tail_k1024_offset + tail0_bytes, page);
    model->attn_b_q8_offset = round_up(
        model->tail_k1024_pair_offset + tail1_bytes, page);
    model->q_b_k1024_offset = round_up(
        model->attn_b_q8_offset + attn_b_q8_bytes, page);
    model->decode_attn_b_offset = round_up(
        model->q_b_k1024_offset + q_b_k1024_bytes, page);
    model->size = round_up(
        model->decode_attn_b_offset + decode_attn_b_bytes, page);
    void *storage = nullptr;
    if (posix_memalign(&storage, (size_t)page, (size_t)model->size) != 0) {
        return false;
    }
    model->data = static_cast<uint8_t *>(storage);
    std::memset(model->data, 0xa5, (size_t)model->size);
    fill_q4_rows(reinterpret_cast<block_q4_K_test *>(
                     model->data + model->weight0_offset),
                 kM0, kK, 0x41c64e6du);
    fill_q4_rows(reinterpret_cast<block_q4_K_test *>(
                     model->data + model->weight1_offset),
                 kM1, kK, 0x9e3779b9u);
    fill_q4_rows(reinterpret_cast<block_q4_K_test *>(
                     model->data + model->attn_a_offset),
                 kAttnGroups * kAttnRank, kAttnGroupDim, 0x243f6a88u);
    for (uint32_t group = 0; group < kDecodeAttnGroups; group++) {
        fill_q4_rows(reinterpret_cast<block_q4_K_test *>(
                         model->data + model->decode_attn_a_offset +
                         (uint64_t)group * decode_attn_a_group_bytes),
                     kDecodeAttnRank, kDecodeAttnGroupDim,
                     0xd1b54a35u ^ (group * 0x9e3779b9u));
    }
    fill_q4_rows(reinterpret_cast<block_q4_K_test *>(
                     model->data + model->attn_b_offset),
                 kAttnOutDim, kAttnLowDim, 0x85a308d3u);
    fill_q4_rows(reinterpret_cast<block_q4_K_test *>(
                     model->data + model->tail_k1024_offset),
                 kM0, kTailK, 0x13198a2eu);
    fill_q4_rows(reinterpret_cast<block_q4_K_test *>(
                     model->data + model->tail_k1024_pair_offset),
                 kM1, kTailK, 0xa4093822u);
    fill_q4_rows(reinterpret_cast<block_q4_K_test *>(
                     model->data + model->q_b_k1024_offset),
                 kQbOutDim, kTailK, 0x082efa98u);
    fill_q4_rows(reinterpret_cast<block_q4_K_test *>(
                     model->data + model->decode_attn_b_offset),
                 kDecodeAttnOutDim, kDecodeAttnLowDim, 0x452821e6u);
    fill_q8_0_rows(reinterpret_cast<block_q8_0_test *>(
                       model->data + model->attn_b_q8_offset),
                   kAttnOutDim, kAttnLowDim, 0x03707344u);
    return true;
}

void fill_activation(std::vector<float> *x, uint32_t n_tokens,
                     uint32_t in_dim = kK) {
    x->resize((uint64_t)n_tokens * in_dim);
    for (uint32_t token = 0; token < n_tokens; token++) {
        for (uint32_t b = 0; b < in_dim / kQkK; b++) {
            float *block =
                x->data() + (uint64_t)token * in_dim + b * kQkK;
            for (uint32_t i = 0; i < kQkK; i++) {
                const int q = (int)((i * 73u + token * 37u + b * 19u) % 241u) - 120;
                block[i] = (float)q / 32.0f;
            }
            // A unique, exactly representable maximum makes the CPU and GPU
            // quantizers select the same signed scale without tie ambiguity.
            block[0] = ((token + b) & 1u) ? 127.0f / 32.0f
                                          : -127.0f / 32.0f;
        }
    }
}

/* Exercise the canonical max-selection edge cases: all-zero blocks and equal
 * opposite-sign maxima spanning both the per-lane and cross-lane tree. */
void fill_q8_wave32_activation(std::vector<float> *x, uint32_t n_tokens,
                               uint32_t in_dim) {
    x->resize((uint64_t)n_tokens * in_dim);
    for (uint32_t token = 0; token < n_tokens; token++) {
        for (uint32_t b = 0; b < in_dim / kQkK; b++) {
            float *block = x->data() + (uint64_t)token * in_dim + b * kQkK;
            if (((token + b) % 7u) == 0u) {
                std::fill(block, block + kQkK, 0.0f);
                continue;
            }
            for (uint32_t i = 0; i < kQkK; i++) {
                const int value =
                    (int)((i * 29u + token * 17u + b * 11u) % 97u) - 48;
                block[i] = (float)value / 16.0f;
            }
            const float first = ((token + b) & 1u) ? 4.0f : -4.0f;
            uint32_t first_index = 5u;
            uint32_t second_index = 224u;
            switch ((token + b) % 3u) {
                case 1u:
                    first_index = 0u;
                    second_index = 128u;
                    break;
                case 2u:
                    first_index = 1u;
                    second_index = 16u;
                    break;
                default:
                    break;
            }
            block[first_index] = first;
            block[second_index] = -first;
        }
    }
}

void quantize_q8_K_cpu(const float *x, block_q8_K_test *out) {
    float abs_part[kQkK];
    float val_part[kQkK];
    for (uint32_t i = 0; i < kQkK; i++) {
        abs_part[i] = std::fabs(x[i]);
        val_part[i] = x[i];
    }
    /* Mirror the canonical GPU reduction literally.  A linear scan chooses a
     * different signed maximum when equal magnitudes occur, which changes the
     * raw Q8_K bytes even though the resulting dot product is equivalent. */
    for (uint32_t stride = kQkK >> 1u; stride != 0u; stride >>= 1u) {
        for (uint32_t i = 0; i < stride; i++) {
            if (abs_part[i + stride] > abs_part[i]) {
                abs_part[i] = abs_part[i + stride];
                val_part[i] = val_part[i + stride];
            }
        }
    }
    const float amax = abs_part[0];
    const float maxv = val_part[0];
    if (amax == 0.0f) {
        std::memset(out, 0, sizeof(*out));
        return;
    }
    const float iscale = -127.0f / maxv;
    for (uint32_t i = 0; i < kQkK; i++) {
        int q = (int)std::lrint(iscale * x[i]);
        q = std::max(-128, std::min(127, q));
        out->qs[i] = (int8_t)q;
    }
    for (uint32_t group = 0; group < kQkK / 16u; group++) {
        int sum = 0;
        for (uint32_t i = 0; i < 16u; i++) {
            sum += out->qs[group * 16u + i];
        }
        out->bsums[group] = (int16_t)sum;
    }
    out->d = 1.0f / iscale;
}

float dot_q4_q8_raw(const block_q4_K_test &weight,
                    const block_q8_K_test &activation) {
    int isum = 0;
    int summs = 0;
    for (uint32_t j = 0; j < 8u; j++) {
        uint8_t scale = 0;
        uint8_t minimum = 0;
        q4_scale_min(j, weight.scales, &scale, &minimum);
        summs += (int)minimum *
                 ((int)activation.bsums[2u * j] +
                  (int)activation.bsums[2u * j + 1u]);
        const uint32_t byte_offset = (j >> 1u) * 32u;
        const uint32_t shift = (j & 1u) ? 4u : 0u;
        int group_dot = 0;
        for (uint32_t i = 0; i < 32u; i++) {
            const int q4 = (weight.qs[byte_offset + i] >> shift) & 0x0f;
            group_dot += q4 * (int)activation.qs[j * 32u + i];
        }
        isum += (int)scale * group_dot;
    }
    const float d = fp16_to_float(weight.d);
    const float dmin = fp16_to_float(weight.dmin);
    return activation.d * d * (float)isum -
           activation.d * dmin * (float)summs;
}

std::vector<float> dense_reference(const uint8_t *weight_base,
                                   const std::vector<float> &x,
                                   uint32_t out_dim,
                                   uint32_t n_tokens,
                                   uint32_t in_dim = kK) {
    const auto *weights = reinterpret_cast<const block_q4_K_test *>(weight_base);
    const uint32_t blocks_per_row = in_dim / kQkK;
    std::vector<block_q8_K_test> xq((uint64_t)n_tokens * blocks_per_row);
    for (uint32_t token = 0; token < n_tokens; token++) {
        for (uint32_t b = 0; b < blocks_per_row; b++) {
            quantize_q8_K_cpu(
                x.data() + (uint64_t)token * in_dim + b * kQkK,
                &xq[(uint64_t)token * blocks_per_row + b]);
        }
    }
    std::vector<float> result((uint64_t)n_tokens * out_dim, 0.0f);
    for (uint32_t token = 0; token < n_tokens; token++) {
        for (uint32_t row = 0; row < out_dim; row++) {
            // Mirror the kernel's b=lane; b+=8 walk and width-8 shuffle tree.
            // This makes raw-bit diagnostics meaningful even at the real
            // K=4096 shape while the tolerance remains the promotion gate.
            float lane_sum[8] = {};
            for (uint32_t lane = 0; lane < 8u; lane++) {
                for (uint32_t b = lane; b < blocks_per_row; b += 8u) {
                    lane_sum[lane] += dot_q4_q8_raw(
                        weights[(uint64_t)row * blocks_per_row + b],
                        xq[(uint64_t)token * blocks_per_row + b]);
                }
            }
            for (uint32_t offset = 4u; offset > 0u; offset >>= 1u) {
                for (uint32_t lane = 0; lane + offset < 8u; lane++) {
                    lane_sum[lane] += lane_sum[lane + offset];
                }
            }
            result[(uint64_t)token * out_dim + row] = lane_sum[0];
        }
    }
    return result;
}

bool close_to_cpu(const std::vector<float> &got,
                  const std::vector<float> &expected,
                  const char *label) {
    uint64_t raw_mismatches = 0;
    uint64_t tolerance_failures = 0;
    float max_abs = 0.0f;
    float max_rel = 0.0f;
    size_t worst = 0;
    for (size_t i = 0; i < got.size(); i++) {
        if (std::memcmp(&got[i], &expected[i], sizeof(float)) != 0) {
            raw_mismatches++;
        }
        const float diff = std::fabs(got[i] - expected[i]);
        const float rel = diff / std::max(1.0f, std::fabs(expected[i]));
        if (diff > max_abs) {
            max_abs = diff;
            worst = i;
        }
        max_rel = std::max(max_rel, rel);
        const float limit = kCpuAbsTolerance +
                            kCpuRelTolerance * std::fabs(expected[i]);
        if (!std::isfinite(got[i]) || diff > limit) tolerance_failures++;
    }
    std::fprintf(stderr,
                 "%s: raw_mismatches=%llu/%zu max_abs=%g max_rel=%g "
                 "worst=%zu tolerance(abs=%g rel=%g) %s\n",
                 label, (unsigned long long)raw_mismatches, got.size(),
                 max_abs, max_rel, worst, kCpuAbsTolerance,
                 kCpuRelTolerance, tolerance_failures == 0 ? "PASS" : "FAIL");
    if (tolerance_failures != 0 && worst < got.size()) {
        std::fprintf(stderr, "  worst got=%g cpu=%g delta=%g\n",
                     got[worst], expected[worst], got[worst] - expected[worst]);
    }
    return tolerance_failures == 0;
}

bool close_with_tolerance(const std::vector<float> &got,
                          const std::vector<float> &expected,
                          float abs_tolerance,
                          float rel_tolerance,
                          const char *label,
                          bool gate = true) {
    if (got.size() != expected.size()) return false;
    uint64_t failures = 0;
    uint64_t nonfinite = 0;
    float max_abs = 0.0f;
    float max_rel = 0.0f;
    size_t worst = 0;
    for (size_t i = 0; i < got.size(); i++) {
        if (!std::isfinite(got[i]) || !std::isfinite(expected[i])) {
            failures++;
            nonfinite++;
            continue;
        }
        const float diff = std::fabs(got[i] - expected[i]);
        const float rel = diff / std::max(1.0f, std::fabs(expected[i]));
        if (diff > max_abs) {
            max_abs = diff;
            worst = i;
        }
        max_rel = std::max(max_rel, rel);
        if (diff > abs_tolerance + rel_tolerance * std::fabs(expected[i])) {
            failures++;
        }
    }
    std::fprintf(stderr,
                 "%s: failures=%llu/%zu nonfinite=%llu max_abs=%g "
                 "max_rel=%g worst=%zu tolerance(abs=%g rel=%g) %s\n",
                 label, (unsigned long long)failures, got.size(),
                 (unsigned long long)nonfinite, max_abs, max_rel, worst,
                 abs_tolerance, rel_tolerance,
                 failures == 0u ? "PASS" :
                 (gate || nonfinite != 0u ? "FAIL" : "DIAGNOSTIC"));
    return nonfinite == 0u && (failures == 0u || !gate);
}

bool bitwise_equal(const std::vector<float> &got,
                   const std::vector<float> &expected,
                   const char *label) {
    uint64_t mismatches = 0;
    size_t first = 0;
    for (size_t i = 0; i < got.size(); i++) {
        if (std::memcmp(&got[i], &expected[i], sizeof(float)) != 0) {
            if (mismatches == 0) first = i;
            mismatches++;
        }
    }
    std::fprintf(stderr, "%s: raw_mismatches=%llu/%zu %s\n",
                 label, (unsigned long long)mismatches, got.size(),
                 mismatches == 0 ? "PASS" : "FAIL");
    if (mismatches != 0) {
        uint32_t got_bits = 0;
        uint32_t expected_bits = 0;
        std::memcpy(&got_bits, &got[first], sizeof(got_bits));
        std::memcpy(&expected_bits, &expected[first], sizeof(expected_bits));
        std::fprintf(stderr,
                     "  first=%zu got=%g/0x%08x expected=%g/0x%08x\n",
                     first, got[first], got_bits, expected[first], expected_bits);
    }
    return mismatches == 0;
}

bool write_tensor(ds4_gpu_tensor *tensor, const std::vector<float> &values) {
    return tensor && ds4_gpu_tensor_write(
        tensor, 0, values.data(), values.size() * sizeof(float)) != 0;
}

bool read_tensor(const ds4_gpu_tensor *tensor, std::vector<float> *values) {
    return tensor && ds4_gpu_tensor_read(
        tensor, 0, values->data(), values->size() * sizeof(float)) != 0;
}

struct lds_output_check {
    ds4_gpu_tensor *tensor;
    const std::vector<float> *expected;
    const std::vector<float> *sentinel;
};

template<typename Enqueue>
bool run_prefill_lds_rollback_oracle(
        Enqueue enqueue, std::initializer_list<lds_output_check> outputs) {
    // Isolate the packed-layout copy/fence controls. The aligned output-B
    // layout has a separate native oracle and successful-enqueue counter.
    env_snapshot aligned_disable("DS4_ROCM_DISABLE_Q4_PREFILL_LDS_ALIGNED");
    (void)setenv("DS4_ROCM_DISABLE_Q4_PREFILL_LDS_ALIGNED", "1", 1);
    env_snapshot disable(kPrefillLdsDisable);
    env_snapshot vector_disable(kPrefillLdsVectorDisable);
    // Both presence-based controls, including their precedence. The new
    // rollback must keep scalar streaming; the older one disables both.
    for (const char *value : {static_cast<const char *>(nullptr), "1", "0", ""})
    for (const char *vector_value : {static_cast<const char *>(nullptr), "1", "0", ""}) {
        if ((value ? setenv(kPrefillLdsDisable, value, 1) :
                     unsetenv(kPrefillLdsDisable)) != 0) return false;
        if ((vector_value ? setenv(kPrefillLdsVectorDisable, vector_value, 1) :
                            unsetenv(kPrefillLdsVectorDisable)) != 0) return false;
        for (const auto &out : outputs)
            if (!write_tensor(out.tensor, *out.sentinel)) return false;
        ds4_rocm_test_q4_prefill_lds_stream_reset();
        ds4_rocm_test_q4_prefill_lds_vector_reset();
        if (enqueue() <= 0) return false;
        const uint64_t calls = ds4_rocm_test_q4_prefill_lds_stream_get_calls();
        const uint64_t vector_calls = ds4_rocm_test_q4_prefill_lds_vector_get_calls();
        if ((value == nullptr && calls == 0u) || (value != nullptr && calls != 0u)) {
            std::fprintf(stderr, "Q4 LDS opt-out '%s': unexpected launches=%llu FAIL\n",
                         value ? value : "unset", (unsigned long long)calls);
            return false;
        }
        if (vector_calls != (!value && !vector_value ? calls : 0u)) {
            std::fprintf(stderr, "Q4 LDS vector opt-out '%s' (stream '%s'): launches=%llu FAIL\n",
                         vector_value ? vector_value : "unset", value ? value : "unset",
                         (unsigned long long)vector_calls);
            return false;
        }
        for (const auto &out : outputs) {
            std::vector<float> got(out.expected->size());
            if (!read_tensor(out.tensor, &got) ||
                !bitwise_equal(got, *out.expected, "Q4 LDS default/rollback and guards"))
                return false;
        }
    }
    return true;
}

std::vector<float> sentinel_values(size_t count) {
    std::vector<float> result(count);
    for (size_t i = 0; i < count; i++) {
        const uint32_t bits = 0x4b000000u + (uint32_t)i;
        std::memcpy(&result[i], &bits, sizeof(bits));
    }
    return result;
}

bool run_dense_case(const aligned_model &model, uint32_t n_tokens,
                    uint64_t offset, uint32_t out_dim, const char *label) {
    std::vector<float> x;
    fill_activation(&x, n_tokens);
    tensor_owner x_gpu(x.size() * sizeof(float));
    tensor_owner out_gpu((uint64_t)n_tokens * out_dim * sizeof(float));
    if (!x_gpu.ptr || !out_gpu.ptr || !write_tensor(x_gpu.ptr, x)) {
        std::fprintf(stderr, "%s: tensor allocation/write FAIL\n", label);
        return false;
    }
    const int rc = ds4_gpu_matmul_quant_tensor(
        out_gpu.ptr, model.data, model.size, offset, kQ4Type,
        kK, out_dim, x_gpu.ptr, n_tokens);
    std::vector<float> got((uint64_t)n_tokens * out_dim);
    if (rc == 0 || !read_tensor(out_gpu.ptr, &got)) {
        std::fprintf(stderr, "%s: dense dispatch rc=%d FAIL\n", label, rc);
        return false;
    }
    const std::vector<float> cpu = dense_reference(
        model.data + offset, x, out_dim, n_tokens);
    return close_to_cpu(got, cpu, label);
}

bool output_guard_unchanged(const std::vector<float> &values,
                            const std::vector<float> &sentinel,
                            size_t logical_count,
                            const char *label);

bool run_pair_case(const aligned_model &model, uint32_t n_tokens,
                   const char *label, bool reverse_outputs = false,
                   uint32_t in_dim = kK) {
    const uint32_t out0_dim = reverse_outputs ? kM1 : kM0;
    const uint32_t out1_dim = reverse_outputs ? kM0 : kM1;
    const uint64_t base0_offset = in_dim == kTailK
                                ? model.tail_k1024_offset
                                : model.weight0_offset;
    const uint64_t base1_offset = in_dim == kTailK
                                ? model.tail_k1024_pair_offset
                                : model.weight1_offset;
    const uint64_t weight0_offset = reverse_outputs
                                  ? base1_offset : base0_offset;
    const uint64_t weight1_offset = reverse_outputs
                                  ? base0_offset : base1_offset;
    const size_t count0 = (size_t)n_tokens * out0_dim;
    const size_t count1 = (size_t)n_tokens * out1_dim;
    const std::vector<float> sentinel0 =
        sentinel_values(count0 + kOutputGuardFloats);
    const std::vector<float> sentinel1 =
        sentinel_values(count1 + kOutputGuardFloats);
    std::vector<float> x;
    fill_activation(&x, n_tokens, in_dim);
    tensor_owner x_gpu(x.size() * sizeof(float));
    tensor_owner dense0(count0 * sizeof(float));
    tensor_owner dense1(count1 * sizeof(float));
    tensor_owner pair0(sentinel0.size() * sizeof(float));
    tensor_owner pair1(sentinel1.size() * sizeof(float));
    if (!x_gpu.ptr || !dense0.ptr || !dense1.ptr || !pair0.ptr || !pair1.ptr ||
        !write_tensor(x_gpu.ptr, x) ||
        !write_tensor(pair0.ptr, sentinel0) ||
        !write_tensor(pair1.ptr, sentinel1)) {
        std::fprintf(stderr, "%s: tensor allocation/write FAIL\n", label);
        return false;
    }
    const int dense_rc0 = ds4_gpu_matmul_quant_tensor(
        dense0.ptr, model.data, model.size, weight0_offset, kQ4Type,
        in_dim, out0_dim, x_gpu.ptr, n_tokens);
    const int dense_rc1 = ds4_gpu_matmul_quant_tensor(
        dense1.ptr, model.data, model.size, weight1_offset, kQ4Type,
        in_dim, out1_dim, x_gpu.ptr, n_tokens);
    const int pair_rc = ds4_gpu_matmul_q4_K_pair_tensor(
        pair0.ptr, pair1.ptr, model.data, model.size,
        weight0_offset, weight1_offset,
        in_dim, out0_dim, out1_dim, x_gpu.ptr, n_tokens);
    std::vector<float> dense0_host(count0);
    std::vector<float> dense1_host(count1);
    std::vector<float> pair0_host(sentinel0.size());
    std::vector<float> pair1_host(sentinel1.size());
    if (dense_rc0 == 0 || dense_rc1 == 0 || pair_rc <= 0 ||
        !read_tensor(dense0.ptr, &dense0_host) ||
        !read_tensor(dense1.ptr, &dense1_host) ||
        !read_tensor(pair0.ptr, &pair0_host) ||
        !read_tensor(pair1.ptr, &pair1_host)) {
        std::fprintf(stderr,
                     "%s: dispatch/read dense=(%d,%d) pair=%d FAIL\n",
                     label, dense_rc0, dense_rc1, pair_rc);
        return false;
    }
    const std::vector<float> cpu0 = dense_reference(
        model.data + weight0_offset, x, out0_dim, n_tokens, in_dim);
    const std::vector<float> cpu1 = dense_reference(
        model.data + weight1_offset, x, out1_dim, n_tokens, in_dim);
    bool ok = close_to_cpu(dense0_host, cpu0, "pair control dense0 vs CPU");
    ok = close_to_cpu(dense1_host, cpu1, "pair control dense1 vs CPU") && ok;
    ok = output_guard_unchanged(pair0_host, sentinel0, count0,
                                "pair0 output canary") && ok;
    ok = output_guard_unchanged(pair1_host, sentinel1, count1,
                                "pair1 output canary") && ok;
    pair0_host.resize(count0);
    pair1_host.resize(count1);
    ok = bitwise_equal(pair0_host, dense0_host, "pair0 vs standalone dense0") && ok;
    ok = bitwise_equal(pair1_host, dense1_host, "pair1 vs standalone dense1") && ok;
    std::fprintf(stderr, "%s: %s\n", label, ok ? "PASS" : "FAIL");
    return ok;
}

bool unchanged_after_rejected_call(ds4_gpu_tensor *tensor,
                                   const std::vector<float> &sentinel,
                                   const char *label) {
    std::vector<float> after(sentinel.size());
    if (!read_tensor(tensor, &after)) {
        std::fprintf(stderr, "%s: readback FAIL\n", label);
        return false;
    }
    return bitwise_equal(after, sentinel, label);
}

bool output_guard_unchanged(const std::vector<float> &values,
                            const std::vector<float> &sentinel,
                            size_t logical_count,
                            const char *label) {
    if (values.size() != sentinel.size() ||
        logical_count > values.size()) {
        std::fprintf(stderr, "%s: invalid guard geometry FAIL\n", label);
        return false;
    }
    uint64_t mismatches = 0;
    size_t first = logical_count;
    for (size_t i = logical_count; i < values.size(); i++) {
        if (std::memcmp(&values[i], &sentinel[i], sizeof(float)) != 0) {
            if (mismatches == 0) first = i;
            mismatches++;
        }
    }
    std::fprintf(stderr, "%s: mismatches=%llu/%zu %s\n",
                 label, (unsigned long long)mismatches,
                 values.size() - logical_count,
                 mismatches == 0 ? "PASS" : "FAIL");
    if (mismatches != 0) {
        std::fprintf(stderr, "  first guard overwrite at float %zu\n", first);
    }
    return mismatches == 0;
}

bool output_body_overwritten(const std::vector<float> &values,
                             const std::vector<float> &sentinel,
                             size_t logical_count,
                             const char *label) {
    if (values.size() != sentinel.size() || logical_count > values.size()) {
        std::fprintf(stderr, "%s: invalid body geometry FAIL\n", label);
        return false;
    }
    uint64_t unchanged = 0;
    size_t first = logical_count;
    for (size_t i = 0; i < logical_count; i++) {
        if (std::memcmp(&values[i], &sentinel[i], sizeof(float)) == 0) {
            if (unchanged == 0) first = i;
            unchanged++;
        }
    }
    std::fprintf(stderr, "%s: unchanged=%llu/%zu %s\n",
                 label, (unsigned long long)unchanged, logical_count,
                 unchanged == 0 ? "PASS" : "FAIL");
    if (unchanged != 0) {
        std::fprintf(stderr, "  first unwritten output at float %zu\n", first);
    }
    return unchanged == 0;
}

bool output_body_finite(const std::vector<float> &values,
                        size_t logical_count,
                        const char *label) {
    if (logical_count > values.size()) {
        std::fprintf(stderr, "%s: invalid body geometry FAIL\n", label);
        return false;
    }
    uint64_t nonfinite = 0u;
    size_t first = logical_count;
    for (size_t i = 0; i < logical_count; i++) {
        if (!std::isfinite(values[i])) {
            if (nonfinite == 0u) first = i;
            nonfinite++;
        }
    }
    std::fprintf(stderr, "%s: nonfinite=%llu/%zu %s\n",
                 label, (unsigned long long)nonfinite, logical_count,
                 nonfinite == 0u ? "PASS" : "FAIL");
    if (nonfinite != 0u) {
        std::fprintf(stderr, "  first non-finite output at float %zu\n", first);
    }
    return nonfinite == 0u;
}

bool run_prefill_parity_case(const aligned_model &model, uint32_t n_tokens,
                             uint64_t offset, uint32_t out_dim,
                             bool compare_cpu, const char *label,
                             uint32_t in_dim = kK) {
    std::vector<float> x;
    fill_activation(&x, n_tokens, in_dim);
    const size_t logical_count = (size_t)n_tokens * out_dim;
    const size_t allocation_count = logical_count + kOutputGuardFloats;
    const std::vector<float> sentinel = sentinel_values(allocation_count);

    tensor_owner x_gpu(x.size() * sizeof(float));
    tensor_owner legacy_gpu(allocation_count * sizeof(float));
    tensor_owner candidate_gpu(allocation_count * sizeof(float));
    if (!x_gpu.ptr || !legacy_gpu.ptr || !candidate_gpu.ptr ||
        !write_tensor(x_gpu.ptr, x) ||
        !write_tensor(legacy_gpu.ptr, sentinel) ||
        !write_tensor(candidate_gpu.ptr, sentinel)) {
        std::fprintf(stderr, "%s: tensor allocation/write FAIL\n", label);
        return false;
    }

    env_snapshot enable(kPrefillEnable);
    env_snapshot disable(kPrefillDisable);
    env_snapshot require(kPrefillRequire);
    env_snapshot k1024_tile4_disable(kPrefillK1024Tile4Disable);
    env_snapshot k1024_tile4_ssd_enable(kPrefillK1024Tile4SsdEnable);
    env_snapshot k1024_tile4_require(kPrefillK1024Tile4Require);
    env_snapshot wmma_disable(kPrefillWmmaDisable);
    const bool require_k1024_tile4 =
        in_dim == kTailK && out_dim == kQbOutDim;

    // The authoritative rollback remains the reference now that the tiled
    // path is default-on.
    (void)unsetenv(kPrefillEnable);
    (void)setenv(kPrefillDisable, "1", 1);
    (void)unsetenv(kPrefillRequire);
    (void)unsetenv(kPrefillK1024Tile4SsdEnable);
    (void)unsetenv(kPrefillK1024Tile4Require);
    (void)setenv(kPrefillWmmaDisable, "1", 1);
    const int legacy_rc = ds4_gpu_matmul_quant_tensor(
        legacy_gpu.ptr, model.data, model.size, offset, kQ4Type,
        in_dim, out_dim, x_gpu.ptr, n_tokens);

    // TILE8 is default-on.  Leave the legacy ENABLE unset and use REQUIRE so
    // a silently ineligible default cannot compare the legacy kernel with
    // itself.
    (void)unsetenv(kPrefillEnable);
    (void)unsetenv(kPrefillDisable);
    (void)setenv(kPrefillRequire, "1", 1);
    (void)unsetenv(kPrefillK1024Tile4Disable);
    (void)unsetenv(kPrefillK1024Tile4SsdEnable);
    (void)setenv(kPrefillWmmaDisable, "1", 1);
    if (require_k1024_tile4) {
        (void)setenv(kPrefillK1024Tile4Require, "1", 1);
    } else {
        (void)unsetenv(kPrefillK1024Tile4Require);
    }
    const int candidate_rc = ds4_gpu_matmul_quant_tensor(
        candidate_gpu.ptr, model.data, model.size, offset, kQ4Type,
        in_dim, out_dim, x_gpu.ptr, n_tokens);

    std::vector<float> legacy_all(allocation_count);
    std::vector<float> candidate_all(allocation_count);
    if (legacy_rc == 0 || candidate_rc == 0 ||
        !read_tensor(legacy_gpu.ptr, &legacy_all) ||
        !read_tensor(candidate_gpu.ptr, &candidate_all)) {
        std::fprintf(stderr,
                     "%s: dispatch/read legacy=%d candidate=%d FAIL\n",
                     label, legacy_rc, candidate_rc);
        return false;
    }

    bool ok = output_guard_unchanged(
        legacy_all, sentinel, logical_count, "prefill legacy output canary");
    ok = run_prefill_lds_rollback_oracle([&]() {
        return ds4_gpu_matmul_quant_tensor(
            candidate_gpu.ptr, model.data, model.size, offset, kQ4Type,
            in_dim, out_dim, x_gpu.ptr, n_tokens);
    }, {{candidate_gpu.ptr, &candidate_all, &sentinel}}) && ok;
    ok = output_guard_unchanged(
             candidate_all, sentinel, logical_count,
             "prefill candidate output canary") && ok;

    legacy_all.resize(logical_count);
    candidate_all.resize(logical_count);
    ok = bitwise_equal(candidate_all, legacy_all,
                       "prefill candidate vs forced legacy") && ok;
    if (compare_cpu) {
        const std::vector<float> cpu = dense_reference(
            model.data + offset, x, out_dim, n_tokens, in_dim);
        ok = close_to_cpu(legacy_all, cpu,
                          "prefill forced legacy vs CPU") && ok;
        ok = close_to_cpu(candidate_all, cpu,
                          "prefill candidate vs CPU") && ok;
    }
    std::fprintf(stderr,
                 "%s: legacy_rc=%d candidate_rc=%d logical=%zu guard=%zu %s\n",
                 label, legacy_rc, candidate_rc, logical_count,
                 kOutputGuardFloats, ok ? "PASS" : "FAIL");
    return ok;
}

bool run_q_b_f16_null_qhalf_case(const aligned_model &model) {
    constexpr uint32_t n_tokens = 32u;
    static_assert(kQbHeads * kQbHeadDim == kQbOutDim,
                  "q_b test head geometry must cover the projection");
    const size_t logical_count = (size_t)n_tokens * kQbOutDim;
    const size_t allocation_count = logical_count + kOutputGuardFloats;
    const size_t q_half_guard = kOutputGuardFloats;
    const size_t q_half_count = logical_count + q_half_guard;
    std::vector<float> x;
    fill_activation(&x, n_tokens, kTailK);
    const std::vector<float> sentinel = sentinel_values(allocation_count);
    const std::vector<uint16_t> q_half_sentinel(q_half_count, 0x7e55u);
    tensor_owner x_gpu(x.size() * sizeof(float));
    tensor_owner provided_out(allocation_count * sizeof(float));
    tensor_owner scratch_out(allocation_count * sizeof(float));
    tensor_owner reference_out(logical_count * sizeof(float));
    tensor_owner q_half_gpu(q_half_count * sizeof(uint16_t));
    if (!x_gpu.ptr || !provided_out.ptr || !scratch_out.ptr ||
        !reference_out.ptr || !q_half_gpu.ptr ||
        !write_tensor(x_gpu.ptr, x) ||
        !write_tensor(provided_out.ptr, sentinel) ||
        !write_tensor(scratch_out.ptr, sentinel) ||
        !ds4_gpu_tensor_write(q_half_gpu.ptr, 0, q_half_sentinel.data(),
                              q_half_sentinel.size() * sizeof(uint16_t))) {
        std::fprintf(stderr,
                     "q_b F16 null-q_half: tensor allocation/write FAIL\n");
        return false;
    }

    env_snapshot enable(kQbF16Enable);
    env_snapshot disable(kQbF16Disable);
    env_snapshot require(kQbF16Require);
    env_snapshot min_tokens(kQbF16MinTokens);
    env_snapshot f16_output(kQbF16OutputEnable);
    (void)setenv(kQbF16Enable, "1", 1);
    (void)unsetenv(kQbF16Disable);
    (void)setenv(kQbF16Require, "1", 1);
    (void)setenv(kQbF16MinTokens, "32", 1);
    (void)setenv(kQbF16OutputEnable, "1", 1);

    const uint64_t weight_bytes =
        (uint64_t)kQbOutDim * (kTailK / kQkK) *
        sizeof(block_q4_K_test);
    const ds4_gpu_q4_attn_q_b_f16_sidecar_desc desc = {
        model.q_b_k1024_offset,
        weight_bytes,
        kTailK,
        kQbOutDim,
        kQ4Type,
        0u,
    };
    uint64_t prepared_bytes = 0;
    const int prepare_rc = ds4_gpu_prepare_q4_attn_q_b_f16_sidecars(
        model.data, model.size, &desc, 1u, n_tokens, 0u,
        &prepared_bytes);

    /* The release default must keep writing C=F32 and must not touch q_half,
     * even when a large staging tensor is supplied by a test caller. */
    (void)setenv(kQbF16OutputEnable, "0", 1);
    int default_rc = 0;
    if (prepare_rc > 0) {
        default_rc = ds4_gpu_attn_q_b_f16_head_rms_rope_tail_tensor(
            provided_out.ptr, q_half_gpu.ptr, model.data, model.size,
            model.q_b_k1024_offset, kQ4Type, kTailK, kQbOutDim,
            x_gpu.ptr, n_tokens, kQbHeads, kQbHeadDim, kQbRot,
            17u, 0u, false, 10000.0f, 1.0f, 0.0f, 1.0f,
            32.0f, 1.0f, 1.0e-6f);
    }
    std::vector<float> default_host(allocation_count);
    std::vector<uint16_t> default_half_host(q_half_count);
    const bool default_read_ok = default_rc > 0 &&
        read_tensor(provided_out.ptr, &default_host) &&
        ds4_gpu_tensor_read(q_half_gpu.ptr, 0, default_half_host.data(),
                            default_half_host.size() * sizeof(uint16_t));
    bool ok = prepare_rc > 0 && default_read_ok;
    if (default_read_ok) {
        ok = output_guard_unchanged(
                 default_host, sentinel, logical_count,
                 "q_b default-F32 output canary") && ok;
        uint64_t untouched_or_nonfinite = 0;
        for (size_t i = 0; i < logical_count; i++) {
            if (!std::isfinite(default_host[i]) ||
                std::memcmp(&default_host[i], &sentinel[i],
                            sizeof(float)) == 0) {
                untouched_or_nonfinite++;
            }
        }
        uint64_t half_mismatches = 0;
        for (size_t i = 0; i < q_half_count; i++) {
            if (default_half_host[i] != q_half_sentinel[i]) {
                half_mismatches++;
            }
        }
        std::fprintf(stderr,
                     "q_b default-F32 writes finite output: failures=%llu/%zu; "
                     "q_half mismatches=%llu/%zu %s\n",
                     (unsigned long long)untouched_or_nonfinite,
                     logical_count,
                     (unsigned long long)half_mismatches, q_half_count,
                     untouched_or_nonfinite == 0 && half_mismatches == 0
                         ? "PASS" : "FAIL");
        ok = untouched_or_nonfinite == 0 && half_mismatches == 0 && ok;
    }

    (void)setenv(kQbF16OutputEnable, "1", 1);
    if (!write_tensor(provided_out.ptr, sentinel) ||
        !ds4_gpu_tensor_write(q_half_gpu.ptr, 0, q_half_sentinel.data(),
                              q_half_sentinel.size() * sizeof(uint16_t))) {
        ok = false;
    }
    int provided_rc = 0;
    if (prepare_rc > 0) {
        provided_rc = ds4_gpu_attn_q_b_f16_head_rms_rope_tail_tensor(
            provided_out.ptr, q_half_gpu.ptr, model.data, model.size,
            model.q_b_k1024_offset, kQ4Type, kTailK, kQbOutDim,
            x_gpu.ptr, n_tokens, kQbHeads, kQbHeadDim, kQbRot,
            17u, 0u, false, 10000.0f, 1.0f, 0.0f, 1.0f,
            32.0f, 1.0f, 1.0e-6f);
    }

    std::vector<float> provided_host(allocation_count);
    std::vector<uint16_t> q_half_host(q_half_count);
    const bool provided_read_ok = provided_rc > 0 &&
        read_tensor(provided_out.ptr, &provided_host) &&
        ds4_gpu_tensor_read(q_half_gpu.ptr, 0, q_half_host.data(),
                            q_half_host.size() * sizeof(uint16_t));
    ok = provided_read_ok && ok;
    if (provided_read_ok) {
        ok = output_guard_unchanged(
                 provided_host, sentinel, logical_count,
                 "q_b F16 provided-q_half output canary") && ok;
        uint64_t nonfinite = 0;
        for (size_t i = 0; i < logical_count; i++) {
            if (!std::isfinite(provided_host[i])) nonfinite++;
        }
        uint64_t half_guard_mismatches = 0;
        for (size_t i = logical_count; i < q_half_count; i++) {
            if (q_half_host[i] != q_half_sentinel[i]) {
                half_guard_mismatches++;
            }
        }
        std::fprintf(stderr,
                     "q_b F16 provided-q_half: nonfinite=%llu/%zu; "
                     "canary mismatches=%llu/%zu %s\n",
                     (unsigned long long)nonfinite, logical_count,
                     (unsigned long long)half_guard_mismatches, q_half_guard,
                     nonfinite == 0 && half_guard_mismatches == 0
                         ? "PASS" : "FAIL");
        ok = nonfinite == 0 && half_guard_mismatches == 0 && ok;
    }

    /* Re-expand the accepted F16 projection exactly and feed the established
     * F32 epilogue. This is a bitwise oracle for the fused half-input tail,
     * with a nonzero production-shape projection rather than a zero smoke test. */
    int reference_rc = 0;
    std::vector<float> reference_input(logical_count);
    if (provided_read_ok) {
        for (size_t i = 0; i < logical_count; i++) {
            reference_input[i] = fp16_to_float(q_half_host[i]);
        }
        if (write_tensor(reference_out.ptr, reference_input)) {
            reference_rc = ds4_gpu_head_rms_norm_rope_tail_tensor(
                reference_out.ptr, n_tokens, kQbHeads, kQbHeadDim, kQbRot,
                17u, 0u, false, 10000.0f, 1.0f, 0.0f, 1.0f,
                32.0f, 1.0f, 1.0e-6f);
        }
    }
    std::vector<float> reference_host(logical_count);
    if (reference_rc > 0 && read_tensor(reference_out.ptr, &reference_host)) {
        provided_host.resize(logical_count);
        ok = bitwise_equal(provided_host, reference_host,
                           "q_b F16 fused tail vs expanded-F16 F32 tail") && ok;
    } else {
        std::fprintf(stderr,
                     "q_b F16 expanded-F16 epilogue reference rc=%d FAIL\n",
                     reference_rc);
        ok = false;
    }

    /* Non-Apple graphs pass NULL q_half. The backend-owned Q_F16 region must
     * produce the exact same fused result as an explicit staging tensor. */
    int scratch_rc = 0;
    if (prepare_rc > 0) {
        scratch_rc = ds4_gpu_attn_q_b_f16_head_rms_rope_tail_tensor(
            scratch_out.ptr, nullptr, model.data, model.size,
            model.q_b_k1024_offset, kQ4Type, kTailK, kQbOutDim,
            x_gpu.ptr, n_tokens, kQbHeads, kQbHeadDim, kQbRot,
            17u, 0u, false, 10000.0f, 1.0f, 0.0f, 1.0f,
            32.0f, 1.0f, 1.0e-6f);
    }
    std::vector<float> scratch_host(allocation_count);
    if (scratch_rc > 0 && read_tensor(scratch_out.ptr, &scratch_host)) {
        ok = output_guard_unchanged(
                 scratch_host, sentinel, logical_count,
                 "q_b F16 null-q_half output canary") && ok;
        scratch_host.resize(logical_count);
        if (provided_read_ok) {
            ok = bitwise_equal(scratch_host, provided_host,
                               "q_b F16 null vs provided q_half") && ok;
        } else {
            ok = false;
        }
    } else {
        std::fprintf(stderr,
                     "q_b F16 null-q_half dispatch/read rc=%d FAIL\n",
                     scratch_rc);
        ok = false;
    }

    const int release_rc = ds4_gpu_release_q4_attn_q_b_f16_sidecars();
    ok = release_rc != 0 && ok;
    std::fprintf(stderr,
                 "q_b F16 q_half staging: prepare=%d default=%d provided=%d "
                 "scratch=%d "
                 "prepared=%.2f MiB release=%d %s\n",
                 prepare_rc, default_rc, provided_rc, scratch_rc,
                 (double)prepared_bytes / 1048576.0, release_rc,
                 ok ? "PASS" : "FAIL");
    return ok;
}

bool run_prefill_wmma_row_tile_policy_oracle() {
    struct row_tile_case {
        const char *label;
        const char *value;
        uint32_t out_dim;
        uint32_t expected;
    };
    const row_tile_case cases[] = {
        {"automatic small", nullptr, 1023u, 64u},
        {"automatic normal lower boundary", nullptr, 1024u, 128u},
        {"automatic normal upper boundary", nullptr, 8191u, 128u},
        {"automatic large", nullptr, 8192u, 256u},
        {"explicit 64", "64", 32768u, 64u},
        {"explicit 128", "128", 32768u, 128u},
        {"explicit 256", "256", 1024u, 256u},
        {"non-whitelisted integer", "257", 1024u, 128u},
        {"negative integer", "-1", 1024u, 128u},
        {"negative wrapped whitelist", "-18446744073709551552", 1024u,
         128u},
        {"malformed", "128x", 1024u, 128u},
        {"empty", "", 8192u, 256u},
    };

    env_snapshot row_tile(kPrefillWmmaRowTile);
    bool ok = true;
    for (const row_tile_case &test : cases) {
        const int env_rc = test.value
            ? setenv(kPrefillWmmaRowTile, test.value, 1)
            : unsetenv(kPrefillWmmaRowTile);
        const uint32_t got = env_rc == 0
            ? ds4_rocm_test_q4_prefill_wmma_row_tile(test.out_dim)
            : 0u;
        if (env_rc != 0 || got != test.expected) {
            std::fprintf(stderr,
                         "Q4 WMMA row-tile policy %s: expected=%u got=%u "
                         "env_rc=%d FAIL\n",
                         test.label, test.expected, got, env_rc);
            ok = false;
        }
    }
    std::fprintf(stderr, "Q4 WMMA row-tile policy: %s\n",
                 ok ? "PASS" : "FAIL");
    return ok;
}

bool run_prefill_wmma_requested_policy_oracle() {
    struct policy_case {
        const char *label;
        int ssd_streaming;
        int enabled;
        int ssd_enabled;
        int disabled;
        int required;
        int standalone_expected;
        int attention_a_expected;
        int attention_b_expected;
    };
    const policy_case cases[] = {
        {"resident unset keeps A automatic", 0, -1, 0, 0, 0, 1, 1, 0},
        {"resident ENABLE=0 compatibility opt-out", 0, 0, 0, 0, 0,
         0, 0, 0},
        {"resident ENABLE=1 opts attention A only", 0, 1, 0, 0, 0,
         1, 1, 0},
        {"resident DISABLE opts out", 0, -1, 0, 1, 0, 0, 0, 0},
        {"resident REQUIRE requests both attention stages", 0, 0, 0, 0, 1,
         1, 1, 1},
        {"DISABLE+REQUIRE reaches strict rejection", 0, -1, 0, 1, 1,
         1, 1, 1},
        {"SSD default stays conservative", 1, -1, 0, 0, 0, 0, 0, 0},
        {"generic ENABLE does not bypass SSD gate", 1, 1, 0, 0, 0,
         0, 0, 0},
        {"SSD gate opts attention A only", 1, -1, 1, 0, 0, 1, 1, 0},
        {"SSD DISABLE opts out", 1, -1, 1, 1, 0, 0, 0, 0},
        {"SSD REQUIRE requests both attention stages", 1, -1, 0, 0, 1,
         1, 1, 1},
    };

    bool ok = true;
    for (const policy_case &test : cases) {
        const int got = ds4_rocm_test_q4_prefill_wmma_requested_policy(
            test.ssd_streaming, test.enabled, test.ssd_enabled,
            test.disabled, test.required);
        if (got != test.standalone_expected) {
            std::fprintf(stderr,
                         "Q4 standalone WMMA request policy %s: "
                         "expected=%d got=%d FAIL\n",
                         test.label, test.standalone_expected, got);
            ok = false;
        }
        const int a_got =
            ds4_rocm_test_q4_prefill_wmma_attention_a_requested_policy(
                test.ssd_streaming, test.enabled, test.ssd_enabled,
                test.disabled, test.required);
        if (a_got != test.attention_a_expected) {
            std::fprintf(stderr,
                         "Q4 attention-output A WMMA policy %s: "
                         "expected=%d got=%d FAIL\n",
                         test.label, test.attention_a_expected, a_got);
            ok = false;
        }
        const int b_got =
            ds4_rocm_test_q4_prefill_wmma_attention_b_requested_policy(
                test.ssd_streaming, test.enabled, test.ssd_enabled,
                test.disabled, test.required);
        if (b_got != test.attention_b_expected) {
            std::fprintf(stderr,
                         "Q4 attention-output B WMMA policy %s: "
                         "expected=%d got=%d FAIL\n",
                         test.label, test.attention_b_expected, b_got);
            ok = false;
        }
    }
    const int optional_q8_yield =
        ds4_rocm_test_q4_prefill_wmma_yields_to_q8_wave32(1, 0);
    const int absent_q8_yield =
        ds4_rocm_test_q4_prefill_wmma_yields_to_q8_wave32(0, 0);
    const int strict_wmma_yield =
        ds4_rocm_test_q4_prefill_wmma_yields_to_q8_wave32(1, 1);
    if (optional_q8_yield != 1 || absent_q8_yield != 0 ||
        strict_wmma_yield != 0) {
        std::fprintf(stderr,
                     "Q4 WMMA/Q8 precedence: optional=%d absent=%d "
                     "strict=%d FAIL\n",
                     optional_q8_yield, absent_q8_yield,
                     strict_wmma_yield);
        ok = false;
    }
    const int k64_unset =
        ds4_rocm_test_q4_prefill_wmma_k64_control_policy(-1);
    const int k64_false =
        ds4_rocm_test_q4_prefill_wmma_k64_control_policy(0);
    const int k64_true =
        ds4_rocm_test_q4_prefill_wmma_k64_control_policy(1);
    if (k64_unset != 1 || k64_false != 0 || k64_true != 1) {
        std::fprintf(stderr,
                     "Q4 WMMA K64 default policy: unset=%d false=%d "
                     "true=%d FAIL\n",
                     k64_unset, k64_false, k64_true);
        ok = false;
    }
    const int k128_unset =
        ds4_rocm_test_q4_prefill_wmma_k128_policy(
            -1, 1, 256u, 1);
    const int k128_disabled =
        ds4_rocm_test_q4_prefill_wmma_k128_policy(
            1, 1, 256u, 1);
    const int k128_false =
        ds4_rocm_test_q4_prefill_wmma_k128_policy(
            0, 1, 256u, 1);
    const int k128_k32_rollback =
        ds4_rocm_test_q4_prefill_wmma_k128_policy(
            -1, 0, 256u, 1);
    const int k128_rows128 =
        ds4_rocm_test_q4_prefill_wmma_k128_policy(
            -1, 1, 128u, 1);
    const int k128_unaligned =
        ds4_rocm_test_q4_prefill_wmma_k128_policy(
            -1, 1, 256u, 0);
    if (k128_unset != 1 || k128_disabled != 0 || k128_false != 1 ||
        k128_k32_rollback != 0 || k128_rows128 != 0 ||
        k128_unaligned != 0) {
        std::fprintf(stderr,
                     "Q4 WMMA K128 opt-out policy: unset=%d disabled=%d "
                     "false=%d k32=%d rows128=%d unaligned=%d FAIL\n",
                     k128_unset, k128_disabled, k128_false,
                     k128_k32_rollback, k128_rows128, k128_unaligned);
        ok = false;
    }
    std::fprintf(stderr,
                 "ROCm Q4 WMMA request policy oracle: cases=%zu "
                 "q8_yield=%d/%d/%d k64=%d/%d/%d "
                 "k128=%d/%d/%d/%d/%d/%d %s\n",
                 sizeof(cases) / sizeof(cases[0]), optional_q8_yield,
                 absent_q8_yield, strict_wmma_yield,
                 k64_unset, k64_false, k64_true,
                 k128_unset, k128_disabled, k128_false,
                 k128_k32_rollback, k128_rows128, k128_unaligned,
                 ok ? "PASS" : "FAIL");
    return ok;
}

bool run_prefill_k1024_tile4_policy_oracle() {
    struct policy_case {
        const char *label;
        int ssd_streaming;
        int device_resident;
        int ssd_enabled;
        int disabled;
        int required;
        int expected;
    };
    const policy_case cases[] = {
        {"resident default remains automatic", 0, 0, 0, 0, 0, 1},
        {"resident rollback", 0, 1, 0, 1, 0, 0},
        {"resident DISABLE dominates REQUIRE", 0, 1, 1, 1, 1, -1},
        {"SSD default stays conservative", 1, 1, 0, 0, 0, 0},
        {"SSD opt-in rejects a nonresident range", 1, 0, 1, 0, 0, 0},
        {"SSD opt-in accepts a resident range", 1, 1, 1, 0, 0, 1},
        {"SSD REQUIRE rejects a nonresident range", 1, 0, 0, 0, 1, -1},
        {"SSD REQUIRE requests a resident range", 1, 1, 0, 0, 1, 1},
        {"SSD DISABLE dominates ENABLE", 1, 1, 1, 1, 0, 0},
        {"SSD DISABLE dominates REQUIRE", 1, 1, 1, 1, 1, -1},
    };

    bool ok = true;
    for (const policy_case &test : cases) {
        const int got = ds4_rocm_test_q4_prefill_k1024_tile4_policy(
            test.ssd_streaming, test.device_resident, test.ssd_enabled,
            test.disabled, test.required);
        if (got != test.expected) {
            std::fprintf(stderr,
                         "K1024 TILE4 policy %s: expected=%d got=%d FAIL\n",
                         test.label, test.expected, got);
            ok = false;
        }
    }
    std::fprintf(stderr,
                 "ROCm Q4 K1024 TILE4 policy oracle: cases=%zu %s\n",
                 sizeof(cases) / sizeof(cases[0]), ok ? "PASS" : "FAIL");
    return ok;
}

bool run_prefill_q8_wave32_policy_oracle() {
    struct policy_case {
        const char *label;
        int scope;
        int compatible;
        int enabled;
        int disabled;
        int required;
        int expected;
    };
    const policy_case cases[] = {
        {"default remains canonical", 1, 1, 0, 0, 0, 0},
        {"ENABLE selects compatible prefill", 1, 1, 1, 0, 0, 1},
        {"ENABLE falls back outside prefill", 0, 1, 1, 0, 0, 0},
        {"ENABLE falls back on incompatible runtime", 1, 0, 1, 0, 0, 0},
        {"DISABLE dominates ENABLE", 1, 1, 1, 1, 0, 0},
        {"REQUIRE is an opt-in", 1, 1, 0, 0, 1, 1},
        {"REQUIRE rejects non-prefill", 0, 1, 0, 0, 1, -1},
        {"REQUIRE rejects incompatible runtime", 1, 0, 0, 0, 1, -1},
        {"DISABLE dominates REQUIRE", 1, 1, 1, 1, 1, -1},
    };
    bool ok = true;
    for (const policy_case &test : cases) {
        const int got = ds4_rocm_test_q4_prefill_q8_wave32_policy(
            test.scope, test.compatible, test.enabled, test.disabled,
            test.required);
        if (got != test.expected) {
            std::fprintf(stderr,
                         "Q8_K wave32 policy %s: expected=%d got=%d FAIL\n",
                         test.label, test.expected, got);
            ok = false;
        }
    }
    struct q_b_policy_case {
        const char *label;
        uint32_t weight_type;
        uint32_t n_tok;
        int required;
        int f16_required;
        int expected;
    };
    const q_b_policy_case q_b_cases[] = {
        {"Q4 prefill REQUIRE yields", kQ4Type, 32u, 1, 0, 1},
        {"Q4 long prefill REQUIRE yields", kQ4Type, 4097u, 1, 0, 1},
        {"dual REQUIRE conflicts", kQ4Type, 32u, 1, 1, -1},
        {"Q4 decode does not claim prefill", kQ4Type, 8u, 1, 1, 0},
        {"Q8 prefill is unrelated", kQ8Type, 32u, 1, 1, 0},
        {"Q4 prefill without Q8 REQUIRE keeps F16 policy",
         kQ4Type, 32u, 0, 1, 0},
    };
    for (const q_b_policy_case &test : q_b_cases) {
        const int got =
            ds4_rocm_test_q4_attn_q_b_yield_to_q8_wave32_policy(
                test.weight_type, test.n_tok, test.required,
                test.f16_required);
        if (got != test.expected) {
            std::fprintf(stderr,
                         "Q8_K wave32 q_b yield policy %s: "
                         "expected=%d got=%d FAIL\n",
                         test.label, test.expected, got);
            ok = false;
        }
    }
    std::fprintf(stderr,
                 "ROCm Q4 Q8_K wave32 policy oracle: selector=%zu q_b=%zu "
                 "%s\n",
                 sizeof(cases) / sizeof(cases[0]),
                 sizeof(q_b_cases) / sizeof(q_b_cases[0]),
                 ok ? "PASS" : "FAIL");
    return ok;
}

bool run_pair_pre_enqueue_policy_oracle() {
    struct policy_case {
        const char *label;
        int prefill_scope;
        int tile8_required;
        int q8_wave32_required;
        int expected;
    };
    const policy_case cases[] = {
        {"optional prefill rejection falls back", 1, 0, 0, 0},
        {"TILE8 REQUIRE rejection fails closed", 1, 1, 0, -1},
        {"Q8 wave32 REQUIRE can use dense fallback", 1, 0, 1, 0},
        {"TILE8 remains strict with dual REQUIRE", 1, 1, 1, -1},
        {"decode ignores prefill TILE8 REQUIRE", 0, 1, 0, 0},
        {"optional decode rejection falls back", 0, 0, 0, 0},
    };

    bool ok = true;
    for (const policy_case &test : cases) {
        const int got = ds4_rocm_test_q4_pair_pre_enqueue_failure_policy(
            test.prefill_scope, test.tile8_required,
            test.q8_wave32_required);
        if (got != test.expected) {
            std::fprintf(stderr,
                         "Q4 pair pre-enqueue policy %s: "
                         "expected=%d got=%d FAIL\n",
                         test.label, test.expected, got);
            ok = false;
        }
    }

    /* Exercise the public selector without a GPU: null tensors reach the
     * validation rejection only when the pair is selected. */
    env_snapshot tile8_disable(kPrefillDisable);
    env_snapshot tile8_require(kPrefillRequire);
    env_snapshot dense_pair_enable("DS4_ROCM_ENABLE_Q4_DENSE_PAIR");
    env_snapshot dense_pair_disable("DS4_ROCM_DISABLE_Q4_DENSE_PAIR");
    env_snapshot wmma_enable(kPrefillWmmaEnable);
    env_snapshot wmma_disable(kPrefillWmmaDisable);
    env_snapshot wmma_require(kPrefillWmmaRequire);
    env_snapshot wmma_row_tile(kPrefillWmmaRowTile);
    env_snapshot wmma_k64(kPrefillWmmaK64);
    env_snapshot q8_enable(kPrefillQ8Wave32Enable);
    env_snapshot q8_disable(kPrefillQ8Wave32Disable);
    env_snapshot q8_require(kPrefillQ8Wave32Require);
    (void)unsetenv(kPrefillDisable);
    (void)setenv(kPrefillRequire, "1", 1);
    (void)unsetenv("DS4_ROCM_ENABLE_Q4_DENSE_PAIR");
    (void)unsetenv("DS4_ROCM_DISABLE_Q4_DENSE_PAIR");
    (void)unsetenv(kPrefillWmmaEnable);
    (void)unsetenv(kPrefillWmmaDisable);
    (void)unsetenv(kPrefillWmmaRequire);
    (void)unsetenv(kPrefillWmmaRowTile);
    (void)setenv(kPrefillWmmaK64, "0", 1);
    (void)unsetenv(kPrefillQ8Wave32Enable);
    (void)unsetenv(kPrefillQ8Wave32Disable);
    (void)unsetenv(kPrefillQ8Wave32Require);
    const int required_validation = ds4_gpu_matmul_q4_K_pair_tensor(
        nullptr, nullptr, nullptr, 0u, 0u, 0u,
        kK, kM0, kM1, nullptr, 9u);

    (void)unsetenv(kPrefillRequire);
    (void)setenv(kPrefillDisable, "1", 1);
    const int opt_out = ds4_gpu_matmul_q4_K_pair_tensor(
        nullptr, nullptr, nullptr, 0u, 0u, 0u,
        kK, kM0, kM1, nullptr, 9u);

    (void)unsetenv(kPrefillDisable);
    (void)setenv(kPrefillRequire, "1", 1);
    (void)setenv("DS4_ROCM_ENABLE_Q4_DENSE_PAIR", "1", 1);
    const int decode = ds4_gpu_matmul_q4_K_pair_tensor(
        nullptr, nullptr, nullptr, 0u, 0u, 0u,
        kK, kM0, kM1, nullptr, 1u);
    if (required_validation != -1 || opt_out != 0 || decode != 0) {
        std::fprintf(stderr,
                     "Q4 pair public pre-enqueue policy: "
                     "required=%d opt_out=%d decode=%d FAIL\n",
                     required_validation, opt_out, decode);
        ok = false;
    }
    std::fprintf(stderr,
                 "ROCm Q4 pair pre-enqueue policy oracle: cases=%zu "
                 "required=%d opt_out=%d decode=%d %s\n",
                 sizeof(cases) / sizeof(cases[0]), required_validation,
                 opt_out, decode, ok ? "PASS" : "FAIL");
    return ok;
}

bool run_prefill_q8_quantizer_oracle(const aligned_model &model) {
#if DS4_TEST_HAS_HIP_RUNTIME
    int device = 0;
    hipDeviceProp_t properties = {};
    if (hipGetDevice(&device) != hipSuccess ||
        hipGetDeviceProperties(&properties, device) != hipSuccess) {
        std::fprintf(stderr,
                     "ROCm Q4 Q8_K quantizer oracle: device query FAIL\n");
        return false;
    }
    const bool has_wave32 = properties.warpSize == 32 &&
        std::strncmp(properties.gcnArchName, "gfx1151", 7u) == 0;

    bool ok = true;
    constexpr uint32_t n_tokens = 9u;
    // The canonical quantizer also runs on wave64. Exercise complete and
    // partial eight-block batches even when the optional wave32 path cannot.
    const uint32_t dimensions[] = {256u, 768u, 1024u, 1792u, 2048u, 2304u, 4096u, 8192u};
    for (uint32_t in_dim : dimensions) {
        std::vector<float> x;
        fill_q8_wave32_activation(&x, n_tokens, in_dim);
        const size_t block_count =
            (size_t)n_tokens * (in_dim / kQkK);
        const size_t bytes = block_count * sizeof(block_q8_K_test);
        tensor_owner x_gpu(x.size() * sizeof(float));
        tensor_owner legacy_gpu(bytes);
        tensor_owner wave_gpu(bytes);
        if (!x_gpu.ptr || !legacy_gpu.ptr || !wave_gpu.ptr ||
            !write_tensor(x_gpu.ptr, x)) {
            std::fprintf(stderr,
                         "Q8_K wave32 raw K=%u allocation/write FAIL\n",
                         in_dim);
            ok = false;
            continue;
        }
        const int legacy_rc = ds4_rocm_test_q8_K_quantize_tensor(
            legacy_gpu.ptr, x_gpu.ptr, in_dim, n_tokens, 0);
        const int wave_rc = has_wave32 ? ds4_rocm_test_q8_K_quantize_tensor(
            wave_gpu.ptr, x_gpu.ptr, in_dim, n_tokens, 1) : 0;
        std::vector<block_q8_K_test> legacy(block_count);
        std::vector<block_q8_K_test> wave(block_count);
        std::vector<block_q8_K_test> cpu(block_count);
        const bool read_ok = legacy_rc != 0 &&
            ds4_gpu_tensor_read(legacy_gpu.ptr, 0, legacy.data(), bytes) != 0 &&
            (!has_wave32 || (wave_rc != 0 &&
             ds4_gpu_tensor_read(wave_gpu.ptr, 0, wave.data(), bytes) != 0));
        for (size_t i = 0; i < block_count; i++) {
            quantize_q8_K_cpu(x.data() + i * kQkK, &cpu[i]);
        }
        const bool pair_equal = read_ok && (!has_wave32 ||
            std::memcmp(legacy.data(), wave.data(), bytes) == 0);
        const bool cpu_equal = read_ok &&
            std::memcmp(legacy.data(), cpu.data(), bytes) == 0;
        std::fprintf(stderr,
                     "Q8_K raw K=%u blocks=%zu legacy=%d wave=%d "
                     "pair=%s cpu=%s %s\n",
                     in_dim, block_count, legacy_rc, wave_rc,
                     !has_wave32 ? "SKIP" : pair_equal ? "bitwise" : "MISMATCH",
                     cpu_equal ? "bitwise" : "MISMATCH",
                     pair_equal && cpu_equal ? "PASS" : "FAIL");
        ok = pair_equal && cpu_equal && ok;
    }

    if (!has_wave32) {
        std::fprintf(stderr,
                     "ROCm Q4 Q8_K wave32 dispatch oracle: SKIP "
                     "(requires gfx1151 wave32; canonical raw bytes tested)\n");
        return ok;
    }

    env_snapshot tile8_disable(kPrefillDisable);
    env_snapshot tile8_require(kPrefillRequire);
    env_snapshot wmma_disable(kPrefillWmmaDisable);
    env_snapshot q8_enable(kPrefillQ8Wave32Enable);
    env_snapshot q8_disable(kPrefillQ8Wave32Disable);
    env_snapshot q8_require(kPrefillQ8Wave32Require);
    (void)unsetenv(kPrefillDisable);
    (void)setenv(kPrefillRequire, "1", 1);
    (void)setenv(kPrefillWmmaDisable, "1", 1);
    (void)unsetenv(kPrefillQ8Wave32Enable);
    (void)unsetenv(kPrefillQ8Wave32Disable);
    (void)setenv(kPrefillQ8Wave32Require, "1", 1);

    std::vector<float> x;
    fill_q8_wave32_activation(&x, n_tokens, kK);
    tensor_owner x_gpu(x.size() * sizeof(float));
    tensor_owner out_gpu((uint64_t)n_tokens * kM0 * sizeof(float));
    bool dispatch_ok = x_gpu.ptr && out_gpu.ptr && write_tensor(x_gpu.ptr, x);
    ds4_rocm_test_q4_prefill_q8_wave32_reset();
    const int rc = dispatch_ok ? ds4_gpu_matmul_quant_tensor(
        out_gpu.ptr, model.data, model.size, model.weight0_offset, kQ4Type,
        kK, kM0, x_gpu.ptr, n_tokens) : 0;
    const uint64_t calls =
        ds4_rocm_test_q4_prefill_q8_wave32_get_calls();
    std::vector<float> got((uint64_t)n_tokens * kM0);
    dispatch_ok = dispatch_ok && rc != 0 && calls == 1u &&
                  read_tensor(out_gpu.ptr, &got);
    if (dispatch_ok) {
        const std::vector<float> cpu = dense_reference(
            model.data + model.weight0_offset, x, kM0, n_tokens);
        dispatch_ok = close_to_cpu(
            got, cpu, "Q8_K wave32 REQUIRE public dispatch vs CPU");
    }

    /* Q8 REQUIRE owns only activation quantization.  With TILE8 opted out,
     * the pair must yield so the graph can issue two dense calls, each of
     * which still attests the wave32 quantizer. */
    tensor_owner fallback0((uint64_t)n_tokens * kM0 * sizeof(float));
    tensor_owner fallback1((uint64_t)n_tokens * kM1 * sizeof(float));
    (void)setenv(kPrefillDisable, "1", 1);
    (void)unsetenv(kPrefillRequire);
    (void)unsetenv(kPrefillQ8Wave32Disable);
    ds4_rocm_test_q4_prefill_q8_wave32_reset();
    const int pair_fallback_rc = fallback0.ptr && fallback1.ptr
        ? ds4_gpu_matmul_q4_K_pair_tensor(
              fallback0.ptr, fallback1.ptr, model.data, model.size,
              model.weight0_offset, model.weight1_offset,
              kK, kM0, kM1, x_gpu.ptr, n_tokens)
        : -1;
    const uint64_t pair_fallback_calls =
        ds4_rocm_test_q4_prefill_q8_wave32_get_calls();
    const int fallback_rc0 = pair_fallback_rc == 0
        ? ds4_gpu_matmul_quant_tensor(
              fallback0.ptr, model.data, model.size, model.weight0_offset,
              kQ4Type, kK, kM0, x_gpu.ptr, n_tokens)
        : 0;
    const int fallback_rc1 = fallback_rc0 != 0
        ? ds4_gpu_matmul_quant_tensor(
              fallback1.ptr, model.data, model.size, model.weight1_offset,
              kQ4Type, kK, kM1, x_gpu.ptr, n_tokens)
        : 0;
    const uint64_t dense_fallback_calls =
        ds4_rocm_test_q4_prefill_q8_wave32_get_calls();
    const bool pair_fallback = pair_fallback_rc == 0 &&
        pair_fallback_calls == 0u && fallback_rc0 != 0 &&
        fallback_rc1 != 0 && dense_fallback_calls == 2u;

    (void)unsetenv(kPrefillDisable);
    (void)setenv(kPrefillRequire, "1", 1);
    (void)setenv(kPrefillQ8Wave32Disable, "1", 1);
    ds4_rocm_test_q4_prefill_q8_wave32_reset();
    const int rejected_rc = ds4_gpu_matmul_quant_tensor(
        out_gpu.ptr, model.data, model.size, model.weight0_offset, kQ4Type,
        kK, kM0, x_gpu.ptr, n_tokens);
    const uint64_t rejected_calls =
        ds4_rocm_test_q4_prefill_q8_wave32_get_calls();
    const bool rejected = rejected_rc == 0 && rejected_calls == 0u;
    std::fprintf(stderr,
                 "ROCm Q4 Q8_K wave32 dispatch: rc=%d calls=%llu "
                 "pair_fallback=%d/%llu dense_fallback=%d,%d/%llu "
                 "disable+require=%d/%llu %s\n",
                 rc, (unsigned long long)calls, pair_fallback_rc,
                 (unsigned long long)pair_fallback_calls,
                 fallback_rc0, fallback_rc1,
                 (unsigned long long)dense_fallback_calls,
                 rejected_rc,
                 (unsigned long long)rejected_calls,
                 dispatch_ok && pair_fallback && rejected ? "PASS" : "FAIL");
    return dispatch_ok && pair_fallback && rejected && ok;
#else
    (void)model;
    return true;
#endif
}

bool run_prefill_k1024_tile4_ssd_case(const aligned_model &model) {
    constexpr uint32_t n_tokens = 9u;
    const size_t logical_count = (size_t)n_tokens * kQbOutDim;
    const size_t allocation_count = logical_count + kOutputGuardFloats;
    const std::vector<float> sentinel = sentinel_values(allocation_count);
    std::vector<float> x;
    fill_activation(&x, n_tokens, kTailK);
    tensor_owner x_gpu(x.size() * sizeof(float));
    tensor_owner tile8_gpu(allocation_count * sizeof(float));
    tensor_owner tile4_gpu(allocation_count * sizeof(float));
    if (!x_gpu.ptr || !tile8_gpu.ptr || !tile4_gpu.ptr ||
        !write_tensor(x_gpu.ptr, x) ||
        !write_tensor(tile8_gpu.ptr, sentinel) ||
        !write_tensor(tile4_gpu.ptr, sentinel)) {
        std::fprintf(stderr, "prefill K1024 TILE4 SSD: setup FAIL\n");
        return false;
    }

    env_snapshot tile8_disable(kPrefillDisable);
    env_snapshot tile8_require(kPrefillRequire);
    env_snapshot tile4_enable(kPrefillK1024Tile4SsdEnable);
    env_snapshot tile4_disable(kPrefillK1024Tile4Disable);
    env_snapshot tile4_require(kPrefillK1024Tile4Require);
    env_snapshot cache_limit("DS4_ROCM_STREAM_MODEL_CACHE_GB");
    (void)unsetenv(kPrefillDisable);
    (void)setenv(kPrefillRequire, "1", 1);
    (void)unsetenv(kPrefillK1024Tile4SsdEnable);
    (void)unsetenv(kPrefillK1024Tile4Disable);
    (void)unsetenv(kPrefillK1024Tile4Require);
    (void)setenv("DS4_ROCM_STREAM_MODEL_CACHE_GB", "1", 1);

    FILE *model_file = std::tmpfile();
    void *ssd_model_map = MAP_FAILED;
    bool model_map_switched = false;
    bool ok = model_file != nullptr && model.size <= (uint64_t)SIZE_MAX;
    if (ok) {
        ok = std::fwrite(model.data, 1u, (size_t)model.size, model_file) ==
                 (size_t)model.size &&
             std::fflush(model_file) == 0;
    }
    const int model_fd = ok ? fileno(model_file) : -1;
    if (ok && model_fd >= 0) {
        ssd_model_map = mmap(nullptr, (size_t)model.size, PROT_READ,
                             MAP_PRIVATE, model_fd, 0);
        ok = ssd_model_map != MAP_FAILED;
    } else {
        ok = false;
    }
    if (ok) {
        ok = ds4_gpu_synchronize() != 0 &&
             ds4_gpu_set_model_map(ssd_model_map, model.size) != 0;
        model_map_switched = ok;
    }
    if (ok) ok = ds4_gpu_set_model_fd(model_fd) != 0;

    int tile8_rc = 0;
    int tile4_rc = 0;
    int rejected_rc = 1;
    if (ok) {
        /* Switching modes releases prior synthetic resident ranges.  The
         * default call reloads q_b from the file into the normal SSD device
         * cache but must retain TILE8 because the new SSD candidate is off. */
        ds4_gpu_set_ssd_streaming(true);
        tile8_rc = ds4_gpu_matmul_quant_tensor(
            tile8_gpu.ptr, ssd_model_map, model.size,
            model.q_b_k1024_offset,
            kQ4Type, kTailK, kQbOutDim, x_gpu.ptr, n_tokens);

        /* REQUIRE is also an opt-in.  The second call reuses the already
         * cached device range and cannot false-green through TILE8. */
        (void)setenv(kPrefillK1024Tile4SsdEnable, "1", 1);
        (void)setenv(kPrefillK1024Tile4Require, "1", 1);
        tile4_rc = ds4_gpu_matmul_quant_tensor(
            tile4_gpu.ptr, ssd_model_map, model.size,
            model.q_b_k1024_offset,
            kQ4Type, kTailK, kQbOutDim, x_gpu.ptr, n_tokens);
    }

    std::vector<float> tile8_host(allocation_count);
    std::vector<float> tile4_host(allocation_count);
    const bool read_ok = tile8_rc != 0 && tile4_rc != 0 &&
        read_tensor(tile8_gpu.ptr, &tile8_host) &&
        read_tensor(tile4_gpu.ptr, &tile4_host);
    bool parity_ok = read_ok;
    if (read_ok) {
        parity_ok = output_body_overwritten(
            tile8_host, sentinel, logical_count,
            "prefill K1024 TILE8 SSD output body");
        parity_ok = output_body_overwritten(
                        tile4_host, sentinel, logical_count,
                        "prefill K1024 TILE4 SSD output body") &&
                    parity_ok;
        parity_ok = output_guard_unchanged(
            tile8_host, sentinel, logical_count,
            "prefill K1024 TILE8 SSD output canary");
        parity_ok = output_guard_unchanged(
                        tile4_host, sentinel, logical_count,
                        "prefill K1024 TILE4 SSD output canary") &&
                    parity_ok;
        tile8_host.resize(logical_count);
        tile4_host.resize(logical_count);
        parity_ok = bitwise_equal(
                        tile4_host, tile8_host,
                        "prefill K1024 TILE4 SSD vs TILE8 SSD") &&
                    parity_ok;
    }

    if (ok) {
        (void)setenv(kPrefillK1024Tile4Disable, "1", 1);
        if (write_tensor(tile4_gpu.ptr, sentinel)) {
            rejected_rc = ds4_gpu_matmul_quant_tensor(
                tile4_gpu.ptr, ssd_model_map, model.size,
                model.q_b_k1024_offset, kQ4Type, kTailK, kQbOutDim,
                x_gpu.ptr, n_tokens);
        }
    }
    const bool rejected_ok = rejected_rc == 0 &&
        unchanged_after_rejected_call(
            tile4_gpu.ptr, sentinel,
            "prefill K1024 TILE4 SSD DISABLE+REQUIRE preserves output");

    /* Even a failed candidate may follow an accepted baseline launch.  Drain
     * it before the mode transition releases the backing range cache. */
    (void)ds4_gpu_synchronize();
    ds4_gpu_set_ssd_streaming(false);
    (void)ds4_gpu_set_model_fd(-1);
    bool model_map_restored = !model_map_switched;
    if (model_map_switched &&
        !ds4_gpu_set_model_map(model.data, model.size)) {
        std::fprintf(stderr,
                     "prefill K1024 TILE4 SSD: model-map restore FAIL\n");
        ok = false;
    } else if (model_map_switched) {
        model_map_restored = true;
    }
    if (ssd_model_map != MAP_FAILED && model_map_restored) {
        (void)munmap(ssd_model_map, (size_t)model.size);
    }
    if (model_file) std::fclose(model_file);
    ok = ok && parity_ok && rejected_ok;
    std::fprintf(stderr,
                 "prefill q_b K1024 TILE4 SSD: tile8=%d tile4=%d "
                 "rejected=%d %s\n",
                 tile8_rc, tile4_rc, rejected_rc, ok ? "PASS" : "FAIL");
    return ok;
}

bool run_prefill_gate_guards(const aligned_model &model) {
    constexpr uint32_t n_tokens = 9u;
    std::vector<float> x;
    fill_activation(&x, n_tokens);
    const size_t output_count = (size_t)n_tokens * kM1 + kOutputGuardFloats;
    const std::vector<float> sentinel = sentinel_values(output_count);
    tensor_owner x_gpu(x.size() * sizeof(float));
    tensor_owner out_gpu(output_count * sizeof(float));
    if (!x_gpu.ptr || !out_gpu.ptr || !write_tensor(x_gpu.ptr, x) ||
        !write_tensor(out_gpu.ptr, sentinel)) {
        std::fprintf(stderr, "prefill gate guards: setup FAIL\n");
        return false;
    }

    env_snapshot enable(kPrefillEnable);
    env_snapshot disable(kPrefillDisable);
    env_snapshot require(kPrefillRequire);

    /* REQUIRE with neither ENABLE nor DISABLE proves that TILE8 is selected
     * by the default policy. */
    (void)unsetenv(kPrefillEnable);
    (void)unsetenv(kPrefillDisable);
    (void)setenv(kPrefillRequire, "1", 1);
    const int default_rc = ds4_gpu_matmul_quant_tensor(
        out_gpu.ptr, model.data, model.size, model.weight1_offset, kQ4Type,
        kK, kM1, x_gpu.ptr, n_tokens);
    bool ok = default_rc != 0;
    if (default_rc == 0) {
        std::fprintf(stderr,
                     "prefill default-on REQUIRE: expected success got=%d FAIL\n",
                     default_rc);
    }
    std::vector<float> default_out(output_count);
    if (!read_tensor(out_gpu.ptr, &default_out)) return false;
    ok = output_guard_unchanged(
             default_out, sentinel, (size_t)n_tokens * kM1,
             "prefill default-on output canary") && ok;
    if (!write_tensor(out_gpu.ptr, sentinel)) return false;

    (void)setenv(kPrefillEnable, "1", 1);
    (void)setenv(kPrefillDisable, "1", 1);
    (void)setenv(kPrefillRequire, "1", 1);
    const int rc = ds4_gpu_matmul_quant_tensor(
        out_gpu.ptr, model.data, model.size, model.weight1_offset, kQ4Type,
        kK, kM1, x_gpu.ptr, n_tokens);
    ok = rc == 0 && ok;
    if (rc != 0) {
        std::fprintf(stderr,
                     "prefill DISABLE+REQUIRE: expected rc=0 got=%d FAIL\n",
                     rc);
    }
    ok = unchanged_after_rejected_call(
             out_gpu.ptr, sentinel,
             "prefill DISABLE dominates REQUIRE and preserves output") && ok;
    return ok;
}

bool run_prefill_pair_case(const aligned_model &model, uint32_t n_tokens,
                           bool reverse_outputs, uint32_t in_dim = kK) {
    const uint32_t out0_dim = reverse_outputs ? kM1 : kM0;
    const uint32_t out1_dim = reverse_outputs ? kM0 : kM1;
    const uint64_t base0_offset = in_dim == kTailK
                                ? model.tail_k1024_offset
                                : model.weight0_offset;
    const uint64_t base1_offset = in_dim == kTailK
                                ? model.tail_k1024_pair_offset
                                : model.weight1_offset;
    const uint64_t weight0_offset = reverse_outputs
                                  ? base1_offset : base0_offset;
    const uint64_t weight1_offset = reverse_outputs
                                  ? base0_offset : base1_offset;
    std::vector<float> x;
    fill_activation(&x, n_tokens, in_dim);
    const size_t count0 = (size_t)n_tokens * out0_dim;
    const size_t count1 = (size_t)n_tokens * out1_dim;
    const std::vector<float> sentinel0 =
        sentinel_values(count0 + kOutputGuardFloats);
    const std::vector<float> sentinel1 =
        sentinel_values(count1 + kOutputGuardFloats);

    tensor_owner x_gpu(x.size() * sizeof(float));
    tensor_owner legacy0(sentinel0.size() * sizeof(float));
    tensor_owner legacy1(sentinel1.size() * sizeof(float));
    tensor_owner pair0(sentinel0.size() * sizeof(float));
    tensor_owner pair1(sentinel1.size() * sizeof(float));
    if (!x_gpu.ptr || !legacy0.ptr || !legacy1.ptr || !pair0.ptr ||
        !pair1.ptr || !write_tensor(x_gpu.ptr, x) ||
        !write_tensor(legacy0.ptr, sentinel0) ||
        !write_tensor(legacy1.ptr, sentinel1) ||
        !write_tensor(pair0.ptr, sentinel0) ||
        !write_tensor(pair1.ptr, sentinel1)) {
        std::fprintf(stderr, "prefill pair n_tok=%u reverse=%d: setup FAIL\n",
                     n_tokens, reverse_outputs ? 1 : 0);
        return false;
    }

    env_snapshot prefill_enable(kPrefillEnable);
    env_snapshot prefill_disable(kPrefillDisable);
    env_snapshot prefill_require(kPrefillRequire);
    env_snapshot pair_enable("DS4_ROCM_ENABLE_Q4_DENSE_PAIR");
    env_snapshot pair_disable("DS4_ROCM_DISABLE_Q4_DENSE_PAIR");

    (void)unsetenv(kPrefillEnable);
    (void)setenv(kPrefillDisable, "1", 1);
    (void)unsetenv(kPrefillRequire);
    const int legacy_rc0 = ds4_gpu_matmul_quant_tensor(
        legacy0.ptr, model.data, model.size, weight0_offset, kQ4Type,
        in_dim, out0_dim, x_gpu.ptr, n_tokens);
    const int legacy_rc1 = ds4_gpu_matmul_quant_tensor(
        legacy1.ptr, model.data, model.size, weight1_offset, kQ4Type,
        in_dim, out1_dim, x_gpu.ptr, n_tokens);

    // The prefill pair is a distinct path: it must not depend on the legacy
    // decode-pair opt-in, whose <=8-token behavior is tested separately.
    (void)unsetenv(kPrefillEnable);
    (void)unsetenv(kPrefillDisable);
    (void)setenv(kPrefillRequire, "1", 1);
    (void)unsetenv("DS4_ROCM_ENABLE_Q4_DENSE_PAIR");
    (void)unsetenv("DS4_ROCM_DISABLE_Q4_DENSE_PAIR");
    const int pair_rc = ds4_gpu_matmul_q4_K_pair_tensor(
        pair0.ptr, pair1.ptr, model.data, model.size,
        weight0_offset, weight1_offset,
        in_dim, out0_dim, out1_dim, x_gpu.ptr, n_tokens);

    std::vector<float> legacy0_host(sentinel0.size());
    std::vector<float> legacy1_host(sentinel1.size());
    std::vector<float> pair0_host(sentinel0.size());
    std::vector<float> pair1_host(sentinel1.size());
    if (legacy_rc0 == 0 || legacy_rc1 == 0 || pair_rc <= 0 ||
        !read_tensor(legacy0.ptr, &legacy0_host) ||
        !read_tensor(legacy1.ptr, &legacy1_host) ||
        !read_tensor(pair0.ptr, &pair0_host) ||
        !read_tensor(pair1.ptr, &pair1_host)) {
        std::fprintf(stderr,
                     "prefill pair n_tok=%u reverse=%d: dispatch/read "
                     "legacy=(%d,%d) pair=%d FAIL\n",
                     n_tokens, reverse_outputs ? 1 : 0,
                     legacy_rc0, legacy_rc1, pair_rc);
        return false;
    }

    bool ok = output_guard_unchanged(
        pair0_host, sentinel0, count0, "prefill pair0 output canary");
    ok = run_prefill_lds_rollback_oracle([&]() {
        return ds4_gpu_matmul_q4_K_pair_tensor(
            pair0.ptr, pair1.ptr, model.data, model.size,
            weight0_offset, weight1_offset,
            in_dim, out0_dim, out1_dim, x_gpu.ptr, n_tokens);
    }, {{pair0.ptr, &pair0_host, &sentinel0},
        {pair1.ptr, &pair1_host, &sentinel1}}) && ok;
    ok = output_guard_unchanged(
             pair1_host, sentinel1, count1,
             "prefill pair1 output canary") && ok;
    pair0_host.resize(count0);
    pair1_host.resize(count1);
    legacy0_host.resize(count0);
    legacy1_host.resize(count1);
    ok = bitwise_equal(pair0_host, legacy0_host,
                       "prefill pair0 vs forced legacy dense0") && ok;
    ok = bitwise_equal(pair1_host, legacy1_host,
                       "prefill pair1 vs forced legacy dense1") && ok;
    std::fprintf(stderr,
                 "prefill pair K=%u M=(%u,%u) n_tok=%u "
                 "legacy=(%d,%d) pair=%d %s\n",
                 in_dim, out0_dim, out1_dim, n_tokens,
                 legacy_rc0, legacy_rc1, pair_rc, ok ? "PASS" : "FAIL");
    return ok;
}

bool run_attention_rowwise_reference(const aligned_model &model,
                                     const ds4_gpu_tensor *heads,
                                     ds4_gpu_tensor *low,
                                     ds4_gpu_tensor *out,
                                     uint32_t n_tokens,
                                     uint64_t out_b_offset,
                                     uint32_t out_b_type) {
    const uint64_t heads_group_bytes =
        (uint64_t)kAttnGroupDim * sizeof(float);
    const uint64_t heads_token_bytes =
        (uint64_t)kAttnGroups * heads_group_bytes;
    const uint64_t low_group_bytes =
        (uint64_t)kAttnRank * sizeof(float);
    const uint64_t low_token_bytes =
        (uint64_t)kAttnLowDim * sizeof(float);
    const uint64_t out_token_bytes =
        (uint64_t)kAttnOutDim * sizeof(float);
    const uint64_t row_a_bytes =
        (kAttnGroupDim / kQkK) * sizeof(block_q4_K_test);
    const uint64_t group_a_bytes = (uint64_t)kAttnRank * row_a_bytes;

    for (uint32_t token = 0; token < n_tokens; token++) {
        for (uint32_t group = 0; group < kAttnGroups; group++) {
            tensor_owner heads_group(ds4_gpu_tensor_view(
                heads,
                (uint64_t)token * heads_token_bytes +
                    (uint64_t)group * heads_group_bytes,
                heads_group_bytes));
            tensor_owner low_group(ds4_gpu_tensor_view(
                low,
                (uint64_t)token * low_token_bytes +
                    (uint64_t)group * low_group_bytes,
                low_group_bytes));
            if (!heads_group.ptr || !low_group.ptr ||
                ds4_gpu_matmul_quant_tensor(
                    low_group.ptr, model.data, model.size,
                    model.attn_a_offset + (uint64_t)group * group_a_bytes,
                    kQ4Type, kAttnGroupDim, kAttnRank,
                    heads_group.ptr, 1u) == 0) {
                std::fprintf(stderr,
                             "attention row reference A token=%u group=%u FAIL\n",
                             token, group);
                return false;
            }
        }
        tensor_owner low_row(ds4_gpu_tensor_view(
            low, (uint64_t)token * low_token_bytes, low_token_bytes));
        tensor_owner out_row(ds4_gpu_tensor_view(
            out, (uint64_t)token * out_token_bytes, out_token_bytes));
        if (!low_row.ptr || !out_row.ptr ||
            ds4_gpu_matmul_quant_tensor(
                out_row.ptr, model.data, model.size, out_b_offset,
                out_b_type, kAttnLowDim, kAttnOutDim,
                low_row.ptr, 1u) == 0) {
            std::fprintf(stderr,
                         "attention row reference B token=%u FAIL\n", token);
            return false;
        }
    }
    return true;
}

bool run_attention_prefill_case(const aligned_model &model,
                                uint32_t n_tokens,
                                const char *label,
                                uint32_t out_b_type = kQ4Type) {
    if (out_b_type != kQ4Type && out_b_type != kQ8Type) {
        std::fprintf(stderr, "%s: unsupported output-B type %u FAIL\n",
                     label, out_b_type);
        return false;
    }
    const uint64_t out_b_offset = out_b_type == kQ8Type
        ? model.attn_b_q8_offset : model.attn_b_offset;
    const size_t heads_count =
        (size_t)n_tokens * kAttnGroups * kAttnGroupDim;
    const size_t low_count = (size_t)n_tokens * kAttnLowDim;
    const size_t out_count = (size_t)n_tokens * kAttnOutDim;
    const size_t group_tmp_count = (size_t)n_tokens * kAttnGroupDim;
    const size_t low_tmp_count = (size_t)n_tokens * kAttnRank;
    std::vector<float> heads_host;
    fill_activation(&heads_host, n_tokens * kAttnGroups);
    const std::vector<float> low_sentinel =
        sentinel_values(low_count + kOutputGuardFloats);
    const std::vector<float> out_sentinel =
        sentinel_values(out_count + kOutputGuardFloats);
    const std::vector<float> group_tmp_sentinel =
        sentinel_values(group_tmp_count + kOutputGuardFloats);
    const std::vector<float> low_tmp_sentinel =
        sentinel_values(low_tmp_count + kOutputGuardFloats);

    tensor_owner heads_gpu(heads_count * sizeof(float));
    tensor_owner reference_low(low_sentinel.size() * sizeof(float));
    tensor_owner reference_out(out_sentinel.size() * sizeof(float));
    tensor_owner candidate_low(low_sentinel.size() * sizeof(float));
    tensor_owner candidate_out(out_sentinel.size() * sizeof(float));
    tensor_owner group_tmp(group_tmp_sentinel.size() * sizeof(float));
    tensor_owner low_tmp(low_tmp_sentinel.size() * sizeof(float));
    if (!heads_gpu.ptr || !reference_low.ptr || !reference_out.ptr ||
        !candidate_low.ptr || !candidate_out.ptr || !group_tmp.ptr ||
        !low_tmp.ptr || !write_tensor(heads_gpu.ptr, heads_host) ||
        !write_tensor(reference_low.ptr, low_sentinel) ||
        !write_tensor(reference_out.ptr, out_sentinel) ||
        !write_tensor(candidate_low.ptr, low_sentinel) ||
        !write_tensor(candidate_out.ptr, out_sentinel) ||
        !write_tensor(group_tmp.ptr, group_tmp_sentinel) ||
        !write_tensor(low_tmp.ptr, low_tmp_sentinel)) {
        std::fprintf(stderr, "%s: setup FAIL\n", label);
        return false;
    }

    /* Register/copy A as one contiguous tensor before the row-wise oracle
     * requests eight contained group views.  Reversing that order creates
     * overlapping cache registrations (eight subranges followed by their
     * superset) and can fail before either candidate kernel is launched. */
    const uint64_t out_a_row_bytes =
        (kAttnGroupDim / kQkK) * sizeof(block_q4_K_test);
    const uint64_t out_a_bytes =
        (uint64_t)kAttnGroups * kAttnRank * out_a_row_bytes;
    if (!ds4_gpu_cache_model_range(
            model.data, model.size, model.attn_a_offset, out_a_bytes,
            "ROCm Q4 attention-prefill fixture A")) {
        std::fprintf(stderr, "%s: attention-output-A preload FAIL\n", label);
        return false;
    }

    env_snapshot enable(kPrefillEnable);
    env_snapshot disable(kPrefillDisable);
    env_snapshot require(kPrefillRequire);
    (void)unsetenv(kPrefillEnable);
    (void)setenv(kPrefillDisable, "1", 1);
    (void)unsetenv(kPrefillRequire);
    if (!run_attention_rowwise_reference(
            model, heads_gpu.ptr, reference_low.ptr, reference_out.ptr,
            n_tokens, out_b_offset, out_b_type)) {
        std::fprintf(stderr, "%s: row-wise reference FAIL\n", label);
        return false;
    }

    (void)unsetenv(kPrefillEnable);
    (void)unsetenv(kPrefillDisable);
    (void)setenv(kPrefillRequire, "1", 1);
    const int candidate_rc = ds4_gpu_attention_output_q4_K_batch_tensor(
        candidate_out.ptr, candidate_low.ptr, group_tmp.ptr, low_tmp.ptr,
        model.data, model.size, model.attn_a_offset, out_b_offset,
        out_b_type, kAttnGroupDim, kAttnRank, kAttnGroups, kAttnOutDim,
        heads_gpu.ptr, n_tokens);

    std::vector<float> reference_low_host(low_sentinel.size());
    std::vector<float> reference_out_host(out_sentinel.size());
    std::vector<float> candidate_low_host(low_sentinel.size());
    std::vector<float> candidate_out_host(out_sentinel.size());
    std::vector<float> group_tmp_host(group_tmp_sentinel.size());
    std::vector<float> low_tmp_host(low_tmp_sentinel.size());
    if (candidate_rc != 1 ||
        !read_tensor(reference_low.ptr, &reference_low_host) ||
        !read_tensor(reference_out.ptr, &reference_out_host) ||
        !read_tensor(candidate_low.ptr, &candidate_low_host) ||
        !read_tensor(candidate_out.ptr, &candidate_out_host) ||
        !read_tensor(group_tmp.ptr, &group_tmp_host) ||
        !read_tensor(low_tmp.ptr, &low_tmp_host)) {
        std::fprintf(stderr, "%s: candidate dispatch/read rc=%d FAIL\n",
                     label, candidate_rc);
        return false;
    }

    bool ok = output_guard_unchanged(
        candidate_low_host, low_sentinel, low_count,
        "attention candidate low canary");
    ok = run_prefill_lds_rollback_oracle([&]() {
        return ds4_gpu_attention_output_q4_K_batch_tensor(
            candidate_out.ptr, candidate_low.ptr, group_tmp.ptr, low_tmp.ptr,
            model.data, model.size, model.attn_a_offset, out_b_offset,
            out_b_type, kAttnGroupDim, kAttnRank, kAttnGroups, kAttnOutDim,
            heads_gpu.ptr, n_tokens);
    }, {{candidate_low.ptr, &candidate_low_host, &low_sentinel},
        {candidate_out.ptr, &candidate_out_host, &out_sentinel}}) && ok;
    ok = output_guard_unchanged(
             candidate_out_host, out_sentinel, out_count,
             "attention candidate out canary") && ok;
    ok = output_guard_unchanged(
             group_tmp_host, group_tmp_sentinel, group_tmp_count,
             "attention group scratch canary") && ok;
    ok = output_guard_unchanged(
             low_tmp_host, low_tmp_sentinel, low_tmp_count,
             "attention low scratch canary") && ok;
    reference_low_host.resize(low_count);
    reference_out_host.resize(out_count);
    candidate_low_host.resize(low_count);
    candidate_out_host.resize(out_count);
    ok = bitwise_equal(candidate_low_host, reference_low_host,
                       "attention candidate low vs 8x row-wise A") && ok;
    ok = bitwise_equal(candidate_out_host, reference_out_host,
                       "attention candidate out vs row-wise B") && ok;

    // The batch API promises -1 for a REQUIRE diagnostic so the graph does
    // not replay its row fallback after a forced-candidate failure.
    if (!write_tensor(candidate_low.ptr, low_sentinel) ||
        !write_tensor(candidate_out.ptr, out_sentinel) ||
        !write_tensor(group_tmp.ptr, group_tmp_sentinel) ||
        !write_tensor(low_tmp.ptr, low_tmp_sentinel)) {
        return false;
    }
    (void)setenv(kPrefillDisable, "1", 1);
    const int rejected_rc = ds4_gpu_attention_output_q4_K_batch_tensor(
        candidate_out.ptr, candidate_low.ptr, group_tmp.ptr, low_tmp.ptr,
        model.data, model.size, model.attn_a_offset, out_b_offset,
        out_b_type, kAttnGroupDim, kAttnRank, kAttnGroups, kAttnOutDim,
        heads_gpu.ptr, n_tokens);
    if (rejected_rc != -1) {
        std::fprintf(stderr,
                     "%s: DISABLE+REQUIRE expected rc=-1 got=%d FAIL\n",
                     label, rejected_rc);
        ok = false;
    }
    ok = unchanged_after_rejected_call(
             candidate_low.ptr, low_sentinel,
             "attention rejected call preserves low") && ok;
    ok = unchanged_after_rejected_call(
             candidate_out.ptr, out_sentinel,
             "attention rejected call preserves out") && ok;
    ok = unchanged_after_rejected_call(
             group_tmp.ptr, group_tmp_sentinel,
             "attention rejected call preserves group scratch") && ok;
    ok = unchanged_after_rejected_call(
             low_tmp.ptr, low_tmp_sentinel,
             "attention rejected call preserves low scratch") && ok;
    std::fprintf(stderr,
                 "%s: candidate_rc=%d rejected_rc=%d %s\n",
                 label, candidate_rc, rejected_rc, ok ? "PASS" : "FAIL");
    return ok;
}

bool run_grouped_attention_decode_case(const aligned_model &model) {
    const uint64_t row_bytes =
        (kDecodeAttnGroupDim / kQkK) * sizeof(block_q4_K_test);
    const uint64_t group_weight_bytes =
        (uint64_t)kDecodeAttnRank * row_bytes;
    const size_t heads_count =
        (size_t)kDecodeAttnGroups * kDecodeAttnGroupDim;
    const size_t logical_count = kDecodeAttnLowDim;
    const size_t allocation_count = logical_count + kOutputGuardFloats;
    std::vector<float> heads_host;
    fill_activation(&heads_host, kDecodeAttnGroups, kDecodeAttnGroupDim);
    const std::vector<float> sentinel = sentinel_values(allocation_count);

    tensor_owner heads_gpu(heads_count * sizeof(float));
    tensor_owner legacy_gpu(allocation_count * sizeof(float));
    tensor_owner candidate_gpu(allocation_count * sizeof(float));
    if (!heads_gpu.ptr || !legacy_gpu.ptr || !candidate_gpu.ptr ||
        !write_tensor(heads_gpu.ptr, heads_host) ||
        !write_tensor(legacy_gpu.ptr, sentinel) ||
        !write_tensor(candidate_gpu.ptr, sentinel)) {
        std::fprintf(stderr, "grouped attention-A decode: setup FAIL\n");
        return false;
    }

    /* Eight standalone decode calls are the bitwise oracle.  Per-group seeds
     * and activation rows make a wrong weight/input group immediately visible. */
    for (uint32_t group = 0; group < kDecodeAttnGroups; group++) {
        tensor_owner head_group(ds4_gpu_tensor_view(
            heads_gpu.ptr,
            (uint64_t)group * kDecodeAttnGroupDim * sizeof(float),
            (uint64_t)kDecodeAttnGroupDim * sizeof(float)));
        tensor_owner low_group(ds4_gpu_tensor_view(
            legacy_gpu.ptr,
            (uint64_t)group * kDecodeAttnRank * sizeof(float),
            (uint64_t)kDecodeAttnRank * sizeof(float)));
        if (!head_group.ptr || !low_group.ptr ||
            ds4_gpu_matmul_quant_tensor(
                low_group.ptr, model.data, model.size,
                model.decode_attn_a_offset +
                    (uint64_t)group * group_weight_bytes,
                kQ4Type, kDecodeAttnGroupDim, kDecodeAttnRank,
                head_group.ptr, 1u) == 0) {
            std::fprintf(stderr,
                         "grouped attention-A legacy group=%u FAIL\n", group);
            return false;
        }
    }

    env_snapshot enable(kGroupedDecodeEnable);
    env_snapshot disable(kGroupedDecodeDisable);
    env_snapshot require(kGroupedDecodeRequire);
    env_snapshot stats(kGroupedDecodeStats);
    (void)setenv(kGroupedDecodeStats, "1", 1);
    (void)unsetenv(kGroupedDecodeEnable);
    (void)unsetenv(kGroupedDecodeDisable);
    (void)unsetenv(kGroupedDecodeRequire);
    ds4_gpu_set_ssd_streaming(false);

    const int default_rc = ds4_gpu_attention_output_low_q4_K_slice_tensor(
        candidate_gpu.ptr, model.data, model.size,
        model.decode_attn_a_offset, kDecodeAttnGroupDim, kDecodeAttnRank,
        0u, kDecodeAttnGroups, heads_gpu.ptr, 1);
    std::vector<float> resident_default_host(allocation_count);
    std::vector<float> resident_legacy_host(allocation_count);
    bool ok = default_rc == 1 &&
              read_tensor(candidate_gpu.ptr, &resident_default_host) &&
              read_tensor(legacy_gpu.ptr, &resident_legacy_host);
    if (!ok) {
        std::fprintf(stderr,
                     "grouped attention-A resident production default: "
                     "expected rc=1 got=%d/readback FAIL\n",
                     default_rc);
    }
    if (default_rc == 1) {
        ok = output_guard_unchanged(
                 resident_default_host, sentinel, logical_count,
                 "grouped attention-A resident default output canary") && ok;
        ok = bitwise_equal(
                 resident_default_host, resident_legacy_host,
                 "grouped attention-A resident default vs 8 legacy calls") && ok;
    }
    if (!write_tensor(candidate_gpu.ptr, sentinel)) {
        std::fprintf(stderr,
                     "grouped attention-A gate reset: tensor write FAIL\n");
        return false;
    }

    /* The batch fallback passes one row at a time through the same low-level
     * API.  It must not inherit the automatic one-token decode policy. */
    const int batch_row_default_rc =
        ds4_gpu_attention_output_low_q4_K_slice_tensor(
            candidate_gpu.ptr, model.data, model.size,
            model.decode_attn_a_offset, kDecodeAttnGroupDim, kDecodeAttnRank,
            0u, kDecodeAttnGroups, heads_gpu.ptr, 0);
    if (batch_row_default_rc != 0) {
        std::fprintf(stderr,
                     "grouped attention-A batch-row default: "
                     "expected rc=0 got=%d FAIL\n",
                     batch_row_default_rc);
        ok = false;
    }
    ok = unchanged_after_rejected_call(
             candidate_gpu.ptr, sentinel,
             "grouped attention-A batch-row context preserves output") && ok;

    /* The production shape is implicit only for a fully resident model.  This
     * toggles policy state without opening or reading an SSD-backed model. */
    ds4_gpu_set_ssd_streaming(true);
    const int streaming_default_rc =
        ds4_gpu_attention_output_low_q4_K_slice_tensor(
            candidate_gpu.ptr, model.data, model.size,
            model.decode_attn_a_offset, kDecodeAttnGroupDim, kDecodeAttnRank,
            0u, kDecodeAttnGroups, heads_gpu.ptr, 1);
    ds4_gpu_set_ssd_streaming(false);
    if (streaming_default_rc != 0) {
        std::fprintf(stderr,
                     "grouped attention-A streaming default: "
                     "expected rc=0 got=%d FAIL\n",
                     streaming_default_rc);
        ok = false;
    }
    ok = unchanged_after_rejected_call(
             candidate_gpu.ptr, sentinel,
             "grouped attention-A streaming mode preserves output") && ok;

    (void)setenv(kGroupedDecodeDisable, "1", 1);
    const int rollback_rc =
        ds4_gpu_attention_output_low_q4_K_slice_tensor(
            candidate_gpu.ptr, model.data, model.size,
            model.decode_attn_a_offset, kDecodeAttnGroupDim, kDecodeAttnRank,
            0u, kDecodeAttnGroups, heads_gpu.ptr, 1);
    if (rollback_rc != 0) {
        std::fprintf(stderr,
                     "grouped attention-A resident rollback: "
                     "expected rc=0 got=%d FAIL\n",
                     rollback_rc);
        ok = false;
    }
    ok = unchanged_after_rejected_call(
             candidate_gpu.ptr, sentinel,
             "grouped attention-A DISABLE rolls back resident default") && ok;

    (void)setenv(kGroupedDecodeEnable, "1", 1);
    const int disabled_enabled_rc =
        ds4_gpu_attention_output_low_q4_K_slice_tensor(
            candidate_gpu.ptr, model.data, model.size,
            model.decode_attn_a_offset, kDecodeAttnGroupDim, kDecodeAttnRank,
            0u, kDecodeAttnGroups, heads_gpu.ptr, 1);
    if (disabled_enabled_rc != 0) {
        std::fprintf(stderr,
                     "grouped attention-A ENABLE+DISABLE: expected rc=0 got=%d FAIL\n",
                     disabled_enabled_rc);
        ok = false;
    }
    ok = unchanged_after_rejected_call(
             candidate_gpu.ptr, sentinel,
             "grouped attention-A DISABLE dominates ENABLE") && ok;

    (void)setenv(kGroupedDecodeRequire, "1", 1);
    const int disabled_rc =
        ds4_gpu_attention_output_low_q4_K_slice_tensor(
            candidate_gpu.ptr, model.data, model.size,
            model.decode_attn_a_offset, kDecodeAttnGroupDim, kDecodeAttnRank,
            0u, kDecodeAttnGroups, heads_gpu.ptr, 1);
    if (disabled_rc != -1) {
        std::fprintf(stderr,
                     "grouped attention-A DISABLE+REQUIRE: expected rc=-1 got=%d FAIL\n",
                     disabled_rc);
        ok = false;
    }
    ok = unchanged_after_rejected_call(
             candidate_gpu.ptr, sentinel,
             "grouped attention-A DISABLE dominates REQUIRE") && ok;

    (void)unsetenv(kGroupedDecodeDisable);
    const int invalid_rc = ds4_gpu_attention_output_low_q4_K_slice_tensor(
        candidate_gpu.ptr, model.data, model.size, model.size - 16u,
        kDecodeAttnGroupDim, kDecodeAttnRank, 0u, kDecodeAttnGroups,
        heads_gpu.ptr, 1);
    if (invalid_rc != -1) {
        std::fprintf(stderr,
                     "grouped attention-A REQUIRE range guard: expected rc=-1 got=%d FAIL\n",
                     invalid_rc);
        ok = false;
    }
    ok = unchanged_after_rejected_call(
             candidate_gpu.ptr, sentinel,
             "grouped attention-A rejected range preserves output") && ok;

    const int candidate_rc = ds4_gpu_attention_output_low_q4_K_slice_tensor(
        candidate_gpu.ptr, model.data, model.size,
        model.decode_attn_a_offset, kDecodeAttnGroupDim, kDecodeAttnRank,
        0u, kDecodeAttnGroups, heads_gpu.ptr, 1);
    std::vector<float> legacy_host(allocation_count);
    std::vector<float> candidate_host(allocation_count);
    if (candidate_rc != 1 || !read_tensor(legacy_gpu.ptr, &legacy_host) ||
        !read_tensor(candidate_gpu.ptr, &candidate_host)) {
        std::fprintf(stderr,
                     "grouped attention-A candidate dispatch/read rc=%d FAIL\n",
                     candidate_rc);
        return false;
    }
    ok = output_guard_unchanged(
             legacy_host, sentinel, logical_count,
             "grouped attention-A legacy output canary") && ok;
    ok = output_guard_unchanged(
             candidate_host, sentinel, logical_count,
             "grouped attention-A candidate output canary") && ok;
    legacy_host.resize(logical_count);
    candidate_host.resize(logical_count);
    ok = bitwise_equal(candidate_host, legacy_host,
                       "grouped attention-A candidate vs 8 legacy calls") && ok;

    /* A non-zero weight-group origin consumes a compact input/output slice.
     * Reuse groups 3 and 4 from the full fixture to verify both the weight
     * skip and local grouped layout. */
    constexpr uint32_t subset_group0 = 3u;
    constexpr uint32_t subset_group_cnt = 2u;
    const size_t subset_logical_count =
        (size_t)subset_group_cnt * kDecodeAttnRank;
    const std::vector<float> subset_sentinel =
        sentinel_values(subset_logical_count + kOutputGuardFloats);
    tensor_owner subset_heads(ds4_gpu_tensor_view(
        heads_gpu.ptr,
        (uint64_t)subset_group0 * kDecodeAttnGroupDim * sizeof(float),
        (uint64_t)subset_group_cnt * kDecodeAttnGroupDim * sizeof(float)));
    tensor_owner subset_legacy(subset_sentinel.size() * sizeof(float));
    tensor_owner subset_candidate(subset_sentinel.size() * sizeof(float));
    if (!subset_heads.ptr || !subset_legacy.ptr || !subset_candidate.ptr ||
        !write_tensor(subset_legacy.ptr, subset_sentinel) ||
        !write_tensor(subset_candidate.ptr, subset_sentinel)) {
        std::fprintf(stderr, "grouped attention-A subset: setup FAIL\n");
        return false;
    }
    for (uint32_t i = 0; i < subset_group_cnt; i++) {
        tensor_owner head_group(ds4_gpu_tensor_view(
            subset_heads.ptr,
            (uint64_t)i * kDecodeAttnGroupDim * sizeof(float),
            (uint64_t)kDecodeAttnGroupDim * sizeof(float)));
        tensor_owner low_group(ds4_gpu_tensor_view(
            subset_legacy.ptr,
            (uint64_t)i * kDecodeAttnRank * sizeof(float),
            (uint64_t)kDecodeAttnRank * sizeof(float)));
        if (!head_group.ptr || !low_group.ptr ||
            ds4_gpu_matmul_quant_tensor(
                low_group.ptr, model.data, model.size,
                model.decode_attn_a_offset +
                    (uint64_t)(subset_group0 + i) * group_weight_bytes,
                kQ4Type, kDecodeAttnGroupDim, kDecodeAttnRank,
                head_group.ptr, 1u) == 0) {
            std::fprintf(stderr,
                         "grouped attention-A subset legacy group=%u FAIL\n",
                         subset_group0 + i);
            return false;
        }
    }

    /* A slice is deliberately outside the implicit production scope.  It
     * must fall back while the environment is clean, then dispatch when the
     * existing explicit ENABLE override is restored. */
    (void)unsetenv(kGroupedDecodeEnable);
    (void)unsetenv(kGroupedDecodeRequire);
    const int subset_default_rc =
        ds4_gpu_attention_output_low_q4_K_slice_tensor(
            subset_candidate.ptr, model.data, model.size,
            model.decode_attn_a_offset, kDecodeAttnGroupDim, kDecodeAttnRank,
            subset_group0, subset_group_cnt, subset_heads.ptr, 1);
    if (subset_default_rc != 0) {
        std::fprintf(stderr,
                     "grouped attention-A non-standard default: "
                     "expected rc=0 got=%d FAIL\n",
                     subset_default_rc);
        ok = false;
    }
    ok = unchanged_after_rejected_call(
             subset_candidate.ptr, subset_sentinel,
             "grouped attention-A non-standard default preserves output") && ok;
    (void)setenv(kGroupedDecodeEnable, "1", 1);
    const int subset_rc = ds4_gpu_attention_output_low_q4_K_slice_tensor(
        subset_candidate.ptr, model.data, model.size,
        model.decode_attn_a_offset, kDecodeAttnGroupDim, kDecodeAttnRank,
        subset_group0, subset_group_cnt, subset_heads.ptr, 1);
    std::vector<float> subset_legacy_host(subset_sentinel.size());
    std::vector<float> subset_candidate_host(subset_sentinel.size());
    if (subset_rc != 1 ||
        !read_tensor(subset_legacy.ptr, &subset_legacy_host) ||
        !read_tensor(subset_candidate.ptr, &subset_candidate_host)) {
        std::fprintf(stderr,
                     "grouped attention-A subset dispatch/read rc=%d FAIL\n",
                     subset_rc);
        return false;
    }
    ok = output_guard_unchanged(
             subset_legacy_host, subset_sentinel, subset_logical_count,
             "grouped attention-A subset legacy canary") && ok;
    ok = output_guard_unchanged(
             subset_candidate_host, subset_sentinel, subset_logical_count,
             "grouped attention-A subset candidate canary") && ok;
    subset_legacy_host.resize(subset_logical_count);
    subset_candidate_host.resize(subset_logical_count);
    ok = bitwise_equal(
             subset_candidate_host, subset_legacy_host,
             "grouped attention-A subset group0=3 count=2 vs legacy") && ok;
    std::fprintf(stderr,
                 "grouped attention-A decode groups=8 K=4096 rank=1024: "
                 "resident_default=%d batch_row_default=%d "
                 "streaming_default=%d rollback=%d "
                 "disabled_enabled=%d disabled_required=%d invalid=%d "
                 "candidate=%d subset_default=%d subset_enabled=%d "
                 "stats_expected=calls:10,dispatches:3,groups:18,"
                 "fallbacks:5,failures:2 %s\n",
                 default_rc, batch_row_default_rc,
                 streaming_default_rc, rollback_rc,
                 disabled_enabled_rc, disabled_rc, invalid_rc,
                 candidate_rc, subset_default_rc, subset_rc,
                 ok ? "PASS" : "FAIL");
    return ok;
}

bool run_dense_guards(const aligned_model &model) {
    std::vector<float> x;
    fill_activation(&x, 1u);
    tensor_owner x_gpu(x.size() * sizeof(float));
    tensor_owner out_gpu((uint64_t)kM0 * sizeof(float));
    const std::vector<float> sentinel = sentinel_values(kM0);
    if (!x_gpu.ptr || !out_gpu.ptr || !write_tensor(x_gpu.ptr, x) ||
        !write_tensor(out_gpu.ptr, sentinel)) {
        std::fprintf(stderr, "dense guards: setup FAIL\n");
        return false;
    }
    const int bad_k_rc = ds4_gpu_matmul_quant_tensor(
        out_gpu.ptr, model.data, model.size, model.weight0_offset, kQ4Type,
        kK - 1u, kM0, x_gpu.ptr, 1u);
    bool ok = bad_k_rc == 0 && unchanged_after_rejected_call(
        out_gpu.ptr, sentinel, "dense K%256 guard preserves output");
    if (bad_k_rc != 0) {
        std::fprintf(stderr, "dense K%%256 guard: expected rc=0 got=%d FAIL\n",
                     bad_k_rc);
    }
    if (!write_tensor(out_gpu.ptr, sentinel)) return false;
    const int range_rc = ds4_gpu_matmul_quant_tensor(
        out_gpu.ptr, model.data, model.size, model.size - 16u, kQ4Type,
        kK, kM0, x_gpu.ptr, 1u);
    ok = (range_rc == 0) && unchanged_after_rejected_call(
        out_gpu.ptr, sentinel, "dense model-range guard preserves output") && ok;
    if (range_rc != 0) {
        std::fprintf(stderr, "dense model-range guard: expected rc=0 got=%d FAIL\n",
                     range_rc);
    }
    return ok;
}

bool run_pair_guards(const aligned_model &model) {
    constexpr uint32_t n_tokens = 9u;
    std::vector<float> x;
    fill_activation(&x, n_tokens);
    tensor_owner x_gpu(x.size() * sizeof(float));
    tensor_owner out0((uint64_t)n_tokens * kM0 * sizeof(float));
    tensor_owner out1((uint64_t)n_tokens * kM1 * sizeof(float));
    const std::vector<float> sentinel0 = sentinel_values((uint64_t)n_tokens * kM0);
    const std::vector<float> sentinel1 = sentinel_values((uint64_t)n_tokens * kM1);
    if (!x_gpu.ptr || !out0.ptr || !out1.ptr || !write_tensor(x_gpu.ptr, x) ||
        !write_tensor(out0.ptr, sentinel0) || !write_tensor(out1.ptr, sentinel1)) {
        std::fprintf(stderr, "pair guards: setup FAIL\n");
        return false;
    }
    env_snapshot prefill_enable(kPrefillEnable);
    env_snapshot prefill_disable(kPrefillDisable);
    env_snapshot prefill_require(kPrefillRequire);
    env_snapshot wmma_enable(kPrefillWmmaEnable);
    env_snapshot wmma_disable(kPrefillWmmaDisable);
    env_snapshot wmma_require(kPrefillWmmaRequire);
    env_snapshot q8_enable(kPrefillQ8Wave32Enable);
    env_snapshot q8_disable(kPrefillQ8Wave32Disable);
    env_snapshot q8_require(kPrefillQ8Wave32Require);
    (void)unsetenv(kPrefillEnable);
    (void)setenv(kPrefillDisable, "1", 1);
    (void)unsetenv(kPrefillRequire);
    (void)unsetenv(kPrefillWmmaEnable);
    (void)unsetenv(kPrefillWmmaDisable);
    (void)unsetenv(kPrefillWmmaRequire);
    (void)unsetenv(kPrefillQ8Wave32Enable);
    (void)unsetenv(kPrefillQ8Wave32Disable);
    (void)unsetenv(kPrefillQ8Wave32Require);
    const int rc = ds4_gpu_matmul_q4_K_pair_tensor(
        out0.ptr, out1.ptr, model.data, model.size,
        model.weight0_offset, model.weight1_offset,
        kK, kM0, kM1, x_gpu.ptr, n_tokens);
    bool ok = rc == 0;
    if (rc != 0) {
        std::fprintf(stderr, "pair n_tok=9 guard: expected rc=0 got=%d FAIL\n", rc);
    }
    ok = unchanged_after_rejected_call(
             out0.ptr, sentinel0, "pair n_tok=9 preserves out0") && ok;
    ok = unchanged_after_rejected_call(
             out1.ptr, sentinel1, "pair n_tok=9 preserves out1") && ok;

    (void)unsetenv(kPrefillDisable);
    (void)setenv(kPrefillRequire, "1", 1);
    if (!write_tensor(out0.ptr, sentinel0) ||
        !write_tensor(out1.ptr, sentinel1)) {
        return false;
    }
    const int validation_rc = ds4_gpu_matmul_q4_K_pair_tensor(
        out0.ptr, out1.ptr, model.data, model.size,
        model.weight0_offset, model.weight1_offset,
        kK - 1u, kM0, kM1, x_gpu.ptr, n_tokens);
    ok = validation_rc == -1 && unchanged_after_rejected_call(
        out0.ptr, sentinel0,
        "required pair validation preserves out0") && ok;
    ok = unchanged_after_rejected_call(
             out1.ptr, sentinel1,
             "required pair validation preserves out1") && ok;
    if (validation_rc != -1) {
        std::fprintf(stderr,
                     "required pair validation: expected rc=-1 got=%d FAIL\n",
                     validation_rc);
    }

    if (!write_tensor(out0.ptr, sentinel0) ||
        !write_tensor(out1.ptr, sentinel1)) {
        return false;
    }
    const int range_rc = ds4_gpu_matmul_q4_K_pair_tensor(
        out0.ptr, out1.ptr, model.data, model.size,
        model.size - 16u, model.weight1_offset,
        kK, kM0, kM1, x_gpu.ptr, n_tokens);
    ok = range_rc == -1 && unchanged_after_rejected_call(
        out0.ptr, sentinel0, "required pair range preserves out0") && ok;
    ok = unchanged_after_rejected_call(
             out1.ptr, sentinel1,
             "required pair range preserves out1") && ok;
    if (range_rc != -1) {
        std::fprintf(stderr,
                     "required pair range: expected rc=-1 got=%d FAIL\n",
                     range_rc);
    }

    const size_t strict_shared_count =
        (size_t)n_tokens * ((size_t)kM0 + kM1);
    const std::vector<float> strict_shared_sentinel =
        sentinel_values(strict_shared_count);
    tensor_owner strict_shared(strict_shared_count * sizeof(float));
    tensor_owner strict_overlap0(ds4_gpu_tensor_view(
        strict_shared.ptr, 0u,
        (uint64_t)n_tokens * kM0 * sizeof(float)));
    tensor_owner strict_overlap1(ds4_gpu_tensor_view(
        strict_shared.ptr,
        ((uint64_t)n_tokens * kM0 - 1u) * sizeof(float),
        (uint64_t)n_tokens * kM1 * sizeof(float)));
    if (!strict_shared.ptr || !strict_overlap0.ptr || !strict_overlap1.ptr ||
        !write_tensor(strict_shared.ptr, strict_shared_sentinel)) {
        std::fprintf(stderr, "required pair overlap guard: setup FAIL\n");
        return false;
    }
    const int strict_overlap_rc = ds4_gpu_matmul_q4_K_pair_tensor(
        strict_overlap0.ptr, strict_overlap1.ptr, model.data, model.size,
        model.weight0_offset, model.weight1_offset,
        kK, kM0, kM1, x_gpu.ptr, n_tokens);
    ok = strict_overlap_rc == -1 && unchanged_after_rejected_call(
        strict_shared.ptr, strict_shared_sentinel,
        "required pair partial-overlap preserves storage") && ok;
    if (strict_overlap_rc != -1) {
        std::fprintf(stderr,
                     "required pair partial-overlap: "
                     "expected rc=-1 got=%d FAIL\n",
                     strict_overlap_rc);
    }

    (void)setenv(kPrefillDisable, "1", 1);
    (void)unsetenv(kPrefillRequire);
    const size_t shared_count = (size_t)kM0 + kM1;
    const std::vector<float> shared_sentinel = sentinel_values(shared_count);
    tensor_owner shared(shared_count * sizeof(float));
    tensor_owner overlap0(ds4_gpu_tensor_view(
        shared.ptr, 0u, (uint64_t)kM0 * sizeof(float)));
    tensor_owner overlap1(ds4_gpu_tensor_view(
        shared.ptr, (uint64_t)(kM0 - 1u) * sizeof(float),
        (uint64_t)kM1 * sizeof(float)));
    if (!shared.ptr || !overlap0.ptr || !overlap1.ptr ||
        !write_tensor(shared.ptr, shared_sentinel)) {
        std::fprintf(stderr, "pair overlap guard: setup FAIL\n");
        return false;
    }
    const int overlap_rc = ds4_gpu_matmul_q4_K_pair_tensor(
        overlap0.ptr, overlap1.ptr, model.data, model.size,
        model.weight0_offset, model.weight1_offset,
        kK, kM0, kM1, x_gpu.ptr, 1u);
    ok = overlap_rc == 0 && unchanged_after_rejected_call(
        shared.ptr, shared_sentinel,
        "pair partial-overlap guard preserves storage") && ok;
    if (overlap_rc != 0) {
        std::fprintf(stderr,
                     "pair partial-overlap guard: expected rc=0 got=%d FAIL\n",
                     overlap_rc);
    }
    return ok;
}

bool run_pair_opt_in_guards(const aligned_model &model) {
    constexpr uint32_t n_tokens = 1u;
    std::vector<float> x;
    fill_activation(&x, n_tokens);
    tensor_owner x_gpu(x.size() * sizeof(float));
    tensor_owner out0((uint64_t)n_tokens * kM0 * sizeof(float));
    tensor_owner out1((uint64_t)n_tokens * kM1 * sizeof(float));
    const std::vector<float> sentinel0 = sentinel_values(kM0);
    const std::vector<float> sentinel1 = sentinel_values(kM1);
    if (!x_gpu.ptr || !out0.ptr || !out1.ptr || !write_tensor(x_gpu.ptr, x)) {
        std::fprintf(stderr, "pair opt-in guards: setup FAIL\n");
        return false;
    }

    auto rejected_call = [&](const char *label) {
        if (!write_tensor(out0.ptr, sentinel0) ||
            !write_tensor(out1.ptr, sentinel1)) {
            return false;
        }
        const int rc = ds4_gpu_matmul_q4_K_pair_tensor(
            out0.ptr, out1.ptr, model.data, model.size,
            model.weight0_offset, model.weight1_offset,
            kK, kM0, kM1, x_gpu.ptr, n_tokens);
        bool guard_ok = rc == 0;
        if (rc != 0) {
            std::fprintf(stderr, "%s: expected rc=0 got=%d FAIL\n", label, rc);
        }
        guard_ok = unchanged_after_rejected_call(out0.ptr, sentinel0, label) &&
                   guard_ok;
        guard_ok = unchanged_after_rejected_call(out1.ptr, sentinel1, label) &&
                   guard_ok;
        return guard_ok;
    };

    env_snapshot enable("DS4_ROCM_ENABLE_Q4_DENSE_PAIR");
    env_snapshot disable("DS4_ROCM_DISABLE_Q4_DENSE_PAIR");
    (void)unsetenv("DS4_ROCM_ENABLE_Q4_DENSE_PAIR");
    (void)unsetenv("DS4_ROCM_DISABLE_Q4_DENSE_PAIR");
    bool ok = rejected_call("pair disabled-by-default preserves outputs");
    (void)setenv("DS4_ROCM_ENABLE_Q4_DENSE_PAIR", "1", 1);
    (void)setenv("DS4_ROCM_DISABLE_Q4_DENSE_PAIR", "1", 1);
    ok = rejected_call("pair DISABLE dominates ENABLE") && ok;
    return ok;
}

bool run_prefill_wmma_smoke(const aligned_model &model) {
#if DS4_TEST_HAS_HIP_RUNTIME
    hipDeviceProp_t properties{};
    if (hipGetDeviceProperties(&properties, 0) != hipSuccess ||
        properties.warpSize != 32 ||
        std::strncmp(properties.gcnArchName, "gfx1151", 7u) != 0) {
        std::fprintf(stderr,
                     "ROCm Q4 direct-WMMA prefill: SKIP "
                     "(requires gfx1151 wave32)\n");
        return true;
    }
#else
    (void)model;
    return true;
#endif

    constexpr uint32_t n_tokens = 257u;
    const size_t logical_count = (size_t)n_tokens * kM0;
    /* A broken N-tail store could write the remaining 63 tokens of the final
     * tile, while a simultaneous M-tail failure in the forced K128 rowtile
     * could address the remaining 255 rows.  Cover both predicates failing
     * together, not just the normal API canary. */
    constexpr size_t wmma_guard_floats =
        (64u - 1u) * kM0 + (256u - 1u);
    const size_t allocation_count = logical_count + wmma_guard_floats;
    const std::vector<float> sentinel = sentinel_values(allocation_count);
    std::vector<float> x;
    fill_activation(&x, n_tokens, kK);
    tensor_owner x_gpu(x.size() * sizeof(float));
    tensor_owner tile8_gpu(allocation_count * sizeof(float));
    tensor_owner wmma_gpu(allocation_count * sizeof(float));
    tensor_owner k64_gpu(allocation_count * sizeof(float));
    tensor_owner k128_gpu(allocation_count * sizeof(float));
    tensor_owner k128_rollback_gpu(allocation_count * sizeof(float));
    if (!x_gpu.ptr || !tile8_gpu.ptr || !wmma_gpu.ptr || !k64_gpu.ptr ||
        !k128_gpu.ptr || !k128_rollback_gpu.ptr ||
        !write_tensor(x_gpu.ptr, x) || !write_tensor(tile8_gpu.ptr, sentinel) ||
        !write_tensor(wmma_gpu.ptr, sentinel) ||
        !write_tensor(k64_gpu.ptr, sentinel) ||
        !write_tensor(k128_gpu.ptr, sentinel) ||
        !write_tensor(k128_rollback_gpu.ptr, sentinel)) {
        std::fprintf(stderr, "ROCm Q4 direct-WMMA prefill: setup FAIL\n");
        return false;
    }

    env_snapshot tile8_enable(kPrefillEnable);
    env_snapshot tile8_disable(kPrefillDisable);
    env_snapshot tile8_require(kPrefillRequire);
    env_snapshot tile4_require(kPrefillK1024Tile4Require);
    env_snapshot wmma_enable(kPrefillWmmaEnable);
    env_snapshot wmma_ssd_enable(kPrefillWmmaSsdEnable);
    env_snapshot wmma_disable(kPrefillWmmaDisable);
    env_snapshot wmma_require(kPrefillWmmaRequire);
    env_snapshot wmma_row_tile(kPrefillWmmaRowTile);
    env_snapshot wmma_k64(kPrefillWmmaK64);
    env_snapshot wmma_k128_disable(kPrefillWmmaK128Disable);
    env_snapshot q8_wave32_enable(kPrefillQ8Wave32Enable);
    env_snapshot q8_wave32_disable(kPrefillQ8Wave32Disable);
    env_snapshot q8_wave32_require(kPrefillQ8Wave32Require);

    (void)unsetenv(kPrefillEnable);
    (void)unsetenv(kPrefillDisable);
    (void)setenv(kPrefillRequire, "1", 1);
    (void)unsetenv(kPrefillK1024Tile4Require);
    (void)unsetenv(kPrefillWmmaEnable);
    (void)unsetenv(kPrefillWmmaSsdEnable);
    (void)setenv(kPrefillWmmaDisable, "1", 1);
    (void)unsetenv(kPrefillWmmaRequire);
    (void)unsetenv(kPrefillWmmaRowTile);
    (void)setenv(kPrefillWmmaK64, "0", 1);
    (void)unsetenv(kPrefillWmmaK128Disable);
    (void)unsetenv(kPrefillQ8Wave32Enable);
    (void)unsetenv(kPrefillQ8Wave32Disable);
    (void)unsetenv(kPrefillQ8Wave32Require);
    ds4_rocm_test_q4_prefill_wmma_reset();
    const int tile8_rc = ds4_gpu_matmul_quant_tensor(
        tile8_gpu.ptr, model.data, model.size, model.weight0_offset, kQ4Type,
        kK, kM0, x_gpu.ptr, n_tokens);
    const uint64_t tile8_wmma_calls =
        ds4_rocm_test_q4_prefill_wmma_get_calls();
    const uint64_t tile8_k64_calls =
        ds4_rocm_test_q4_prefill_wmma_k64_get_calls();

    (void)unsetenv(kPrefillRequire);
    (void)unsetenv(kPrefillWmmaEnable);
    (void)unsetenv(kPrefillWmmaDisable);
    (void)unsetenv(kPrefillWmmaRequire);
    (void)setenv(kPrefillWmmaK64, "0", 1);
    ds4_rocm_test_q4_prefill_wmma_reset();
    const int wmma_rc = ds4_gpu_matmul_quant_tensor(
        wmma_gpu.ptr, model.data, model.size, model.weight0_offset, kQ4Type,
        kK, kM0, x_gpu.ptr, n_tokens);
    const uint64_t wmma_calls =
        ds4_rocm_test_q4_prefill_wmma_get_calls();
    const uint64_t wmma_k64_calls =
        ds4_rocm_test_q4_prefill_wmma_k64_get_calls();

    /* K64 remains the automatic fallback for row geometries outside K128's
     * 256-row scope. */
    (void)unsetenv(kPrefillWmmaK64);
    ds4_rocm_test_q4_prefill_wmma_reset();
    const int k64_fallback_rc = ds4_gpu_matmul_quant_tensor(
        k64_gpu.ptr, model.data, model.size, model.weight0_offset, kQ4Type,
        kK, kM0, x_gpu.ptr, n_tokens);
    const uint64_t k64_fallback_wmma_calls =
        ds4_rocm_test_q4_prefill_wmma_get_calls();
    const uint64_t k64_fallback_calls =
        ds4_rocm_test_q4_prefill_wmma_k64_get_calls();

    /* Force the 256-row geometry on this compact M-tail fixture so default-on
     * K128 is covered without allocating the full q_b output.  The direct q_b
     * benchmark exercises the natural shape. */
    (void)setenv(kPrefillWmmaRowTile, "256", 1);
    (void)unsetenv(kPrefillWmmaK128Disable);
    ds4_rocm_test_q4_prefill_wmma_reset();
    const int k128_rc = ds4_gpu_matmul_quant_tensor(
        k128_gpu.ptr, model.data, model.size, model.weight0_offset, kQ4Type,
        kK, kM0, x_gpu.ptr, n_tokens);
    const uint64_t k128_wmma_calls =
        ds4_rocm_test_q4_prefill_wmma_get_calls();
    const uint64_t k128_k64_calls =
        ds4_rocm_test_q4_prefill_wmma_k64_get_calls();
    const uint64_t k128_calls =
        ds4_rocm_test_q4_prefill_wmma_k128_get_calls();

    /* A single opt-out must restore K64 on the same otherwise-eligible
     * production geometry. */
    (void)setenv(kPrefillWmmaK128Disable, "1", 1);
    ds4_rocm_test_q4_prefill_wmma_reset();
    const int k128_rollback_rc = ds4_gpu_matmul_quant_tensor(
        k128_rollback_gpu.ptr, model.data, model.size, model.weight0_offset,
        kQ4Type, kK, kM0, x_gpu.ptr, n_tokens);
    const uint64_t k128_rollback_wmma_calls =
        ds4_rocm_test_q4_prefill_wmma_get_calls();
    const uint64_t k128_rollback_k64_calls =
        ds4_rocm_test_q4_prefill_wmma_k64_get_calls();
    const uint64_t k128_rollback_k128_calls =
        ds4_rocm_test_q4_prefill_wmma_k128_get_calls();

    std::vector<float> tile8(allocation_count);
    std::vector<float> wmma(allocation_count);
    std::vector<float> k64_fallback(allocation_count);
    std::vector<float> k128(allocation_count);
    std::vector<float> k128_rollback(allocation_count);
    bool ok = tile8_rc != 0 && wmma_rc != 0 && k64_fallback_rc != 0 &&
        k128_rc != 0 && k128_rollback_rc != 0 &&
        tile8_wmma_calls == 0u && tile8_k64_calls == 0u &&
        wmma_calls == 1u && wmma_k64_calls == 0u &&
        k64_fallback_wmma_calls == 1u && k64_fallback_calls == 1u &&
        k128_wmma_calls == 1u && k128_k64_calls == 0u &&
        k128_calls == 1u && k128_rollback_wmma_calls == 1u &&
        k128_rollback_k64_calls == 1u &&
        k128_rollback_k128_calls == 0u &&
        read_tensor(tile8_gpu.ptr, &tile8) &&
        read_tensor(wmma_gpu.ptr, &wmma) &&
        read_tensor(k64_gpu.ptr, &k64_fallback) &&
        read_tensor(k128_gpu.ptr, &k128) &&
        read_tensor(k128_rollback_gpu.ptr, &k128_rollback);
    if (ok) {
        ok = output_body_overwritten(tile8, sentinel, logical_count,
                                     "direct-WMMA TILE8 output body") && ok;
        ok = output_body_overwritten(wmma, sentinel, logical_count,
                                     "direct-WMMA candidate output body") && ok;
        ok = output_body_overwritten(
                 k64_fallback, sentinel, logical_count,
                 "direct-WMMA K64 fallback output body") && ok;
        ok = output_body_overwritten(
                 k128, sentinel, logical_count,
                 "direct-WMMA default K128 output body") && ok;
        ok = output_body_overwritten(
                 k128_rollback, sentinel, logical_count,
                 "direct-WMMA K128 opt-out output body") && ok;
        ok = output_guard_unchanged(tile8, sentinel, logical_count,
                                    "direct-WMMA TILE8 output canary") && ok;
        ok = output_guard_unchanged(wmma, sentinel, logical_count,
                                    "direct-WMMA candidate output canary") && ok;
        ok = output_guard_unchanged(
                 k64_fallback, sentinel, logical_count,
                 "direct-WMMA K64 fallback output canary") && ok;
        ok = output_guard_unchanged(
                 k128, sentinel, logical_count,
                 "direct-WMMA default K128 output canary") && ok;
        ok = output_guard_unchanged(
                 k128_rollback, sentinel, logical_count,
                 "direct-WMMA K128 opt-out output canary") && ok;
        tile8.resize(logical_count);
        wmma.resize(logical_count);
        k64_fallback.resize(logical_count);
        k128.resize(logical_count);
        k128_rollback.resize(logical_count);
        ok = close_with_tolerance(wmma, tile8, 2.0f, 3.0e-2f,
                                  "direct-WMMA vs TILE8 N/M tail") && ok;
        ok = bitwise_equal(k64_fallback, wmma,
                           "direct-WMMA K64 fallback vs K32 N/M tail") && ok;
        ok = bitwise_equal(k128, k64_fallback,
                           "direct-WMMA K128 vs K64 N/M tail") && ok;
        ok = bitwise_equal(k128_rollback, k128,
                           "direct-WMMA K128 opt-out vs default") && ok;
    }

    /* Return to the neutral K32 setting for the remaining policy cases. */
    (void)setenv(kPrefillWmmaK64, "0", 1);
    (void)unsetenv(kPrefillWmmaK128Disable);
    (void)unsetenv(kPrefillWmmaRowTile);
    (void)setenv(kPrefillWmmaDisable, "1", 1);
    (void)unsetenv(kPrefillWmmaRequire);
    if (!write_tensor(wmma_gpu.ptr, sentinel)) return false;
    ds4_rocm_test_q4_prefill_wmma_reset();
    const int opt_out_rc = ds4_gpu_matmul_quant_tensor(
        wmma_gpu.ptr, model.data, model.size, model.weight0_offset, kQ4Type,
        kK, kM0, x_gpu.ptr, n_tokens);
    const uint64_t opt_out_wmma_calls =
        ds4_rocm_test_q4_prefill_wmma_get_calls();
    const uint64_t opt_out_k64_calls =
        ds4_rocm_test_q4_prefill_wmma_k64_get_calls();
    std::vector<float> opt_out(allocation_count);
    const bool opt_out_read = opt_out_rc != 0 && opt_out_wmma_calls == 0u &&
        opt_out_k64_calls == 0u &&
        read_tensor(wmma_gpu.ptr, &opt_out);
    ok = opt_out_read && ok;
    if (opt_out_read) {
        ok = output_guard_unchanged(
                 opt_out, sentinel, logical_count,
                 "direct-WMMA opt-out output canary") && ok;
        opt_out.resize(logical_count);
        ok = bitwise_equal(opt_out, tile8,
                           "direct-WMMA opt-out vs TILE8") && ok;
    }

    (void)setenv(kPrefillWmmaRequire, "1", 1);
    if (!write_tensor(wmma_gpu.ptr, sentinel)) return false;
    ds4_rocm_test_q4_prefill_wmma_reset();
    const int rejected_rc = ds4_gpu_matmul_quant_tensor(
        wmma_gpu.ptr, model.data, model.size, model.weight0_offset, kQ4Type,
        kK, kM0, x_gpu.ptr, n_tokens);
    const uint64_t rejected_wmma_calls =
        ds4_rocm_test_q4_prefill_wmma_get_calls();
    const uint64_t rejected_k64_calls =
        ds4_rocm_test_q4_prefill_wmma_k64_get_calls();
    ok = rejected_rc == 0 && rejected_wmma_calls == 0u &&
         rejected_k64_calls == 0u &&
         unchanged_after_rejected_call(
             wmma_gpu.ptr, sentinel,
             "direct-WMMA DISABLE+REQUIRE preserves output") && ok;
    std::fprintf(stderr,
                 "ROCm Q4 direct-WMMA prefill: tile8=%d/%llu/%llu "
                 "K32=%d/%llu/%llu K64-fallback=%d/%llu/%llu "
                 "K128-default=%d/%llu/%llu/%llu "
                 "K128-optout=%d/%llu/%llu/%llu "
                 "opt_out=%d/%llu/%llu "
                 "rejected=%d/%llu/%llu %s\n",
                 tile8_rc, (unsigned long long)tile8_wmma_calls,
                 (unsigned long long)tile8_k64_calls,
                 wmma_rc, (unsigned long long)wmma_calls,
                 (unsigned long long)wmma_k64_calls,
                 k64_fallback_rc,
                 (unsigned long long)k64_fallback_wmma_calls,
                 (unsigned long long)k64_fallback_calls,
                 k128_rc, (unsigned long long)k128_wmma_calls,
                 (unsigned long long)k128_k64_calls,
                 (unsigned long long)k128_calls,
                 k128_rollback_rc,
                 (unsigned long long)k128_rollback_wmma_calls,
                 (unsigned long long)k128_rollback_k64_calls,
                 (unsigned long long)k128_rollback_k128_calls,
                 opt_out_rc, (unsigned long long)opt_out_wmma_calls,
                 (unsigned long long)opt_out_k64_calls,
                 rejected_rc, (unsigned long long)rejected_wmma_calls,
                 (unsigned long long)rejected_k64_calls,
                 ok ? "PASS" : "FAIL");
    return ok;
}

bool run_prefill_wmma_load4_oracle(const aligned_model &model, bool required = false) {
#if DS4_TEST_HAS_HIP_RUNTIME
    hipDeviceProp_t properties{};
    if (hipGetDeviceProperties(&properties, 0) != hipSuccess ||
        properties.warpSize != 32 ||
        std::strncmp(properties.gcnArchName, "gfx1151", 7u) != 0) {
        std::fprintf(stderr, "Q4 K64 LOAD4: %s (requires gfx1151 wave32)\n",
                     required ? "FAIL" : "SKIP");
        return !required;
    }
#else
    (void)model;
    return !required;
#endif
    env_snapshot enabled(kPrefillWmmaEnable), disabled(kPrefillWmmaDisable);
    env_snapshot wmma_required(kPrefillWmmaRequire), tile_required(kPrefillRequire);
    env_snapshot row_tile(kPrefillWmmaRowTile), k64(kPrefillWmmaK64);
    env_snapshot k128(kPrefillWmmaK128Disable), load4(kPrefillWmmaK64Load4Disable);
    (void)setenv(kPrefillWmmaEnable, "1", 1);
    (void)unsetenv(kPrefillWmmaDisable);
    (void)unsetenv(kPrefillWmmaRequire);
    (void)unsetenv(kPrefillRequire);
    // This suite, like the surrounding oracle, starts/ends outside quality.
    struct quality_reset { ~quality_reset() { ds4_gpu_set_quality(false); } } reset;
    unsigned cases = 0;
    for (uint32_t n : {9u, 255u, 256u, 257u, 319u})
    for (uint32_t offset : {0u, 1u, 2u, 3u, 4u})
    for (unsigned mode = 0; mode < 6; ++mode) {
        // modes: K64 rows128/256, scalar rows64, K32, K128, quality fallback.
        const uint32_t rows = mode == 1 || mode == 4 ? 256 : (mode == 2 ? 64 : 128);
        (void)setenv(kPrefillWmmaRowTile, std::to_string(rows).c_str(), 1);
        (void)setenv(kPrefillWmmaK64, mode == 3 ? "0" : "1", 1);
        (void)setenv(kPrefillWmmaK128Disable, mode == 4 ? "0" : "1", 1);
        ds4_gpu_set_quality(mode == 5);
        std::vector<float> activation;
        fill_activation(&activation, n, kK);
        // Exercise pair conversion at signed-zero/subnormal/halfway limits.
        const float limits[] = {0.0f, -0.0f, 0x1p-25f, -0x1p-24f,
                                1.00048828125f, 1.00146484375f, 65504.0f};
        for (size_t i = 0; i < sizeof(limits) / sizeof(limits[0]); ++i)
            activation[i] = limits[i];
        std::vector<float> input(offset + activation.size() + 4, -123.0f);
        std::copy(activation.begin(), activation.end(), input.begin() + offset);
        tensor_owner backing(input.size() * sizeof(float));
        if (!backing.ptr || !write_tensor(backing.ptr, input)) return false;
        tensor_owner x(ds4_gpu_tensor_view(backing.ptr, offset * sizeof(float),
                                           activation.size() * sizeof(float)));
        const size_t logical = (size_t)n * kM0;
        const std::vector<float> sentinel =
            sentinel_values(logical + 63u * kM0 + 255u);
        tensor_owner out(sentinel.size() * sizeof(float));
        if (!x.ptr || !out.ptr) return false;
        std::vector<float> reference(sentinel.size()), actual(sentinel.size());
        auto run = [&](const char *flag, std::vector<float> *result) {
            if (flag) (void)setenv(kPrefillWmmaK64Load4Disable, flag, 1);
            else (void)unsetenv(kPrefillWmmaK64Load4Disable);
            if (!write_tensor(out.ptr, sentinel)) return false;
            ds4_rocm_test_q4_prefill_wmma_reset();
            const bool rc = ds4_gpu_matmul_quant_tensor(
                out.ptr, model.data, model.size, model.weight0_offset,
                kQ4Type, kK, kM0, x.ptr, n) != 0;
            const uint64_t count = ds4_rocm_test_q4_prefill_wmma_k64_load4_get_calls();
            const uint64_t expected =
                !flag && n >= 256u && offset % 4 == 0 && mode < 2 ? 1 : 0;
            if (!rc || count != expected || !read_tensor(out.ptr, result) ||
                !output_body_overwritten(*result, sentinel, logical, "K64 LOAD4 body") ||
                !output_guard_unchanged(*result, sentinel, logical, "K64 LOAD4 guard")) {
                std::fprintf(stderr,
                    "Q4 K64 LOAD4 N=%u offset=%u mode=%u flag=%s calls=%llu/%llu FAIL\n",
                    n, offset, mode, flag ? flag : "unset",
                    (unsigned long long)count, (unsigned long long)expected);
                return false;
            }
            return true;
        };
        if (!run("1", &reference)) return false;
        for (const char *flag : {static_cast<const char *>(nullptr), "1", "0", ""}) {
            if (!run(flag, &actual) ||
                !bitwise_equal(actual, reference, "K64 LOAD4 rollback/output guards")) return false;
            ++cases;
        }
        std::vector<float> unchanged(input.size());
        if (!read_tensor(backing.ptr, &unchanged) ||
            !bitwise_equal(unchanged, input, "K64 LOAD4 input unchanged")) return false;
    }
    std::fprintf(stderr, "Q4 K64 LOAD4 runtime: PASS %u cases\n", cases);
    return true;
}

bool run_attention_output_wmma_smoke(const aligned_model &model) {
#if DS4_TEST_HAS_HIP_RUNTIME
    hipDeviceProp_t properties{};
    if (hipGetDeviceProperties(&properties, 0) != hipSuccess ||
        properties.warpSize != 32 ||
        std::strncmp(properties.gcnArchName, "gfx1151", 7u) != 0) {
        std::fprintf(stderr,
                     "ROCm Q4 production output direct-WMMA: SKIP "
                     "(requires gfx1151 wave32)\n");
        return true;
    }
#else
    (void)model;
    return true;
#endif

    constexpr uint32_t n_tokens = 257u;
    const size_t heads_count =
        (size_t)n_tokens * kDecodeAttnGroups * kDecodeAttnGroupDim;
    const size_t low_count = (size_t)n_tokens * kDecodeAttnLowDim;
    const size_t out_count = (size_t)n_tokens * kDecodeAttnOutDim;
    constexpr size_t low_guard = (64u - 1u) * kDecodeAttnLowDim;
    constexpr size_t out_guard = (64u - 1u) * kDecodeAttnOutDim;
    std::vector<float> heads_host;
    fill_activation(
        &heads_host, n_tokens * kDecodeAttnGroups, kDecodeAttnGroupDim);
    const std::vector<float> low_sentinel =
        sentinel_values(low_count + low_guard);
    const std::vector<float> out_sentinel =
        sentinel_values(out_count + out_guard);

    tensor_owner heads_gpu(heads_count * sizeof(float));
    tensor_owner tile8_low(low_sentinel.size() * sizeof(float));
    tensor_owner tile8_out(out_sentinel.size() * sizeof(float));
    tensor_owner candidate_low(low_sentinel.size() * sizeof(float));
    tensor_owner candidate_out(out_sentinel.size() * sizeof(float));
    tensor_owner replay_out(out_sentinel.size() * sizeof(float));
    if (!heads_gpu.ptr || !tile8_low.ptr || !tile8_out.ptr ||
        !candidate_low.ptr || !candidate_out.ptr || !replay_out.ptr ||
        !write_tensor(heads_gpu.ptr, heads_host) ||
        !write_tensor(tile8_low.ptr, low_sentinel) ||
        !write_tensor(tile8_out.ptr, out_sentinel)) {
        std::fprintf(stderr,
                     "ROCm Q4 production output direct-WMMA: setup FAIL\n");
        return false;
    }

    env_snapshot tile8_enable(kPrefillEnable);
    env_snapshot tile8_disable(kPrefillDisable);
    env_snapshot tile8_require(kPrefillRequire);
    env_snapshot tile4_require(kPrefillK1024Tile4Require);
    env_snapshot wmma_enable(kPrefillWmmaEnable);
    env_snapshot wmma_ssd_enable(kPrefillWmmaSsdEnable);
    env_snapshot wmma_disable(kPrefillWmmaDisable);
    env_snapshot wmma_require(kPrefillWmmaRequire);
    env_snapshot wmma_row_tile(kPrefillWmmaRowTile);
    env_snapshot wmma_k64(kPrefillWmmaK64);
    env_snapshot q8_wave32_enable(kPrefillQ8Wave32Enable);
    env_snapshot q8_wave32_disable(kPrefillQ8Wave32Disable);
    env_snapshot q8_wave32_require(kPrefillQ8Wave32Require);

    struct batch_result {
        int rc = 0;
        uint64_t wmma_calls = 0u;
        uint64_t k64_calls = 0u;
        bool write_ok = false;
        bool low_read = false;
        bool out_read = false;
    };

    struct replay_result {
        int rc = 0;
        uint64_t wmma_calls = 0u;
        uint64_t k64_calls = 0u;
        bool write_ok = false;
        bool out_read = false;
    };

    auto clear_controls = [&]() {
        (void)unsetenv(kPrefillEnable);
        (void)unsetenv(kPrefillDisable);
        (void)unsetenv(kPrefillRequire);
        (void)unsetenv(kPrefillK1024Tile4Require);
        (void)unsetenv(kPrefillWmmaEnable);
        (void)unsetenv(kPrefillWmmaSsdEnable);
        (void)unsetenv(kPrefillWmmaDisable);
        (void)unsetenv(kPrefillWmmaRequire);
        (void)unsetenv(kPrefillWmmaRowTile);
        (void)unsetenv(kPrefillWmmaK64);
        (void)unsetenv(kPrefillQ8Wave32Enable);
        (void)unsetenv(kPrefillQ8Wave32Disable);
        (void)unsetenv(kPrefillQ8Wave32Require);
    };

    auto run_batch = [&](ds4_gpu_tensor *low, ds4_gpu_tensor *out,
                         std::vector<float> *low_host,
                         std::vector<float> *out_host) {
        batch_result result;
        ds4_rocm_test_q4_prefill_wmma_reset();
        result.write_ok = write_tensor(low, low_sentinel) &&
                          write_tensor(out, out_sentinel);
        if (result.write_ok) {
            result.rc = ds4_gpu_attention_output_q4_K_batch_tensor(
                out, low, nullptr, nullptr,
                model.data, model.size, model.decode_attn_a_offset,
                model.decode_attn_b_offset, kQ4Type, kDecodeAttnGroupDim,
                kDecodeAttnRank, kDecodeAttnGroups, kDecodeAttnOutDim,
                heads_gpu.ptr, n_tokens);
        }
        result.wmma_calls = ds4_rocm_test_q4_prefill_wmma_get_calls();
        result.k64_calls = ds4_rocm_test_q4_prefill_wmma_k64_get_calls();
        if (result.write_ok) {
            result.low_read = read_tensor(low, low_host);
            result.out_read = read_tensor(out, out_host);
        }
        return result;
    };

    auto batch_ready = [&](const batch_result &result,
                           uint64_t expected_wmma,
                           uint64_t expected_k64,
                           const char *label) {
        const bool ready = result.write_ok && result.rc == 1 &&
            result.wmma_calls == expected_wmma &&
            result.k64_calls == expected_k64 &&
            result.low_read && result.out_read;
        std::fprintf(stderr,
                     "%s: rc=%d WMMA=%llu/%llu K64=%llu/%llu "
                     "write=%d read=%d/%d %s\n",
                     label, result.rc,
                     (unsigned long long)result.wmma_calls,
                     (unsigned long long)expected_wmma,
                     (unsigned long long)result.k64_calls,
                     (unsigned long long)expected_k64,
                     result.write_ok ? 1 : 0,
                     result.low_read ? 1 : 0,
                     result.out_read ? 1 : 0,
                     ready ? "PASS" : "FAIL");
        return ready;
    };

    auto batch_memory_ok = [&](const batch_result &result,
                               const std::vector<float> &low_host,
                               const std::vector<float> &out_host,
                               const char *label) {
        if (!result.low_read || !result.out_read) return false;
        const std::string low_body = std::string(label) + " low body";
        const std::string out_body = std::string(label) + " out body";
        const std::string low_finite = std::string(label) + " low finite";
        const std::string out_finite = std::string(label) + " out finite";
        const std::string low_guard = std::string(label) + " low N-tail";
        const std::string out_guard = std::string(label) + " out N-tail";
        bool memory_ok = output_body_overwritten(
            low_host, low_sentinel, low_count, low_body.c_str());
        memory_ok = output_body_overwritten(
            out_host, out_sentinel, out_count, out_body.c_str()) && memory_ok;
        memory_ok = output_body_finite(
            low_host, low_count, low_finite.c_str()) && memory_ok;
        memory_ok = output_body_finite(
            out_host, out_count, out_finite.c_str()) && memory_ok;
        memory_ok = output_guard_unchanged(
            low_host, low_sentinel, low_count, low_guard.c_str()) && memory_ok;
        memory_ok = output_guard_unchanged(
            out_host, out_sentinel, out_count, out_guard.c_str()) && memory_ok;
        return memory_ok;
    };

    auto run_tile8_replay = [&](const ds4_gpu_tensor *low,
                                std::vector<float> *out_host) {
        replay_result result;
        clear_controls();
        (void)setenv(kPrefillRequire, "1", 1);
        (void)setenv(kPrefillWmmaDisable, "1", 1);
        ds4_rocm_test_q4_prefill_wmma_reset();
        result.write_ok = write_tensor(replay_out.ptr, out_sentinel);
        if (result.write_ok) {
            result.rc = ds4_gpu_matmul_quant_tensor(
                replay_out.ptr, model.data, model.size,
                model.decode_attn_b_offset, kQ4Type,
                kDecodeAttnLowDim, kDecodeAttnOutDim, low, n_tokens);
        }
        result.wmma_calls = ds4_rocm_test_q4_prefill_wmma_get_calls();
        result.k64_calls = ds4_rocm_test_q4_prefill_wmma_k64_get_calls();
        if (result.write_ok) {
            result.out_read = read_tensor(replay_out.ptr, out_host);
        }
        return result;
    };

    auto replay_ready = [&](const replay_result &result,
                            const std::vector<float> &out_host,
                            const char *label) {
        bool ready = result.write_ok && result.rc != 0 &&
            result.wmma_calls == 0u && result.k64_calls == 0u &&
            result.out_read;
        std::fprintf(stderr,
                     "%s: rc=%d WMMA=%llu K64=%llu write=%d read=%d %s\n",
                     label, result.rc,
                     (unsigned long long)result.wmma_calls,
                     (unsigned long long)result.k64_calls,
                     result.write_ok ? 1 : 0,
                     result.out_read ? 1 : 0,
                     ready ? "PASS" : "FAIL");
        if (result.out_read) {
            const std::string body = std::string(label) + " body";
            const std::string finite = std::string(label) + " finite";
            const std::string guard = std::string(label) + " N-tail";
            ready = output_body_overwritten(
                out_host, out_sentinel, out_count, body.c_str()) && ready;
            ready = output_body_finite(
                out_host, out_count, finite.c_str()) && ready;
            ready = output_guard_unchanged(
                out_host, out_sentinel, out_count, guard.c_str()) && ready;
        }
        return ready;
    };

    ds4_gpu_set_ssd_streaming(false);

    /* Forced TILE8 is the exact baseline. */
    clear_controls();
    (void)setenv(kPrefillRequire, "1", 1);
    (void)setenv(kPrefillWmmaDisable, "1", 1);
    std::vector<float> tile8_low_host(low_sentinel.size());
    std::vector<float> tile8_out_host(out_sentinel.size());
    const batch_result tile8_result = run_batch(
        tile8_low.ptr, tile8_out.ptr, &tile8_low_host, &tile8_out_host);

    /* The production default keeps validated WMMA on A and exact TILE8 on B. */
    clear_controls();
    std::vector<float> default_low_host(low_sentinel.size());
    std::vector<float> default_out_host(out_sentinel.size());
    const batch_result default_result = run_batch(
        candidate_low.ptr, candidate_out.ptr,
        &default_low_host, &default_out_host);

    /* ENABLE opts only A into direct WMMA.  Pin K32 as the rollback arm. */
    clear_controls();
    (void)setenv(kPrefillWmmaEnable, "1", 1);
    (void)setenv(kPrefillWmmaK64, "0", 1);
    std::vector<float> k32_low_host(low_sentinel.size());
    std::vector<float> k32_out_host(out_sentinel.size());
    const batch_result k32_result = run_batch(
        candidate_low.ptr, candidate_out.ptr, &k32_low_host, &k32_out_host);

    std::vector<float> k32_replay_host(out_sentinel.size());
    replay_result k32_replay_result;
    if (k32_result.rc == 1 && k32_result.low_read) {
        k32_replay_result = run_tile8_replay(
            candidate_low.ptr, &k32_replay_host);
    }

    /* Run K64 regardless of every earlier comparison result. */
    clear_controls();
    (void)setenv(kPrefillWmmaEnable, "1", 1);
    std::vector<float> k64_low_host(low_sentinel.size());
    std::vector<float> k64_out_host(out_sentinel.size());
    const batch_result k64_result = run_batch(
        candidate_low.ptr, candidate_out.ptr, &k64_low_host, &k64_out_host);

    /* REQUIRE is the only policy that opts both A and B into direct WMMA. */
    clear_controls();
    (void)setenv(kPrefillWmmaRequire, "1", 1);
    (void)setenv(kPrefillWmmaK64, "0", 1);
    std::vector<float> strict_k32_low_host(low_sentinel.size());
    std::vector<float> strict_k32_out_host(out_sentinel.size());
    const batch_result strict_k32_result = run_batch(
        candidate_low.ptr, candidate_out.ptr,
        &strict_k32_low_host, &strict_k32_out_host);

    clear_controls();
    (void)setenv(kPrefillWmmaRequire, "1", 1);
    std::vector<float> strict_k64_low_host(low_sentinel.size());
    std::vector<float> strict_k64_out_host(out_sentinel.size());
    const batch_result strict_k64_result = run_batch(
        candidate_low.ptr, candidate_out.ptr,
        &strict_k64_low_host, &strict_k64_out_host);

    const bool tile8_ready = batch_ready(
        tile8_result, 0u, 0u, "production forced TILE8");
    const bool default_ready = batch_ready(
        default_result, 1u, 1u,
        "production clean-default A-K64/B-TILE8");
    const bool k32_ready = batch_ready(
        k32_result, 1u, 0u, "production ENABLE A-K32/B-TILE8");
    const bool k64_ready = batch_ready(
        k64_result, 1u, 1u, "production ENABLE A-K64/B-TILE8");
    const bool strict_k32_ready = batch_ready(
        strict_k32_result, 2u, 0u, "production REQUIRE A+B K32");
    const bool strict_k64_ready = batch_ready(
        strict_k64_result, 2u, 2u, "production REQUIRE A+B K64");

    const bool tile8_memory = batch_memory_ok(
        tile8_result, tile8_low_host, tile8_out_host,
        "production forced TILE8");
    const bool default_memory = batch_memory_ok(
        default_result, default_low_host, default_out_host,
        "production clean-default A-K64/B-TILE8");
    const bool k32_memory = batch_memory_ok(
        k32_result, k32_low_host, k32_out_host,
        "production ENABLE A-K32/B-TILE8");
    const bool k64_memory = batch_memory_ok(
        k64_result, k64_low_host, k64_out_host,
        "production ENABLE A-K64/B-TILE8");
    const bool strict_k32_memory = batch_memory_ok(
        strict_k32_result, strict_k32_low_host, strict_k32_out_host,
        "production REQUIRE A+B K32");
    const bool strict_k64_memory = batch_memory_ok(
        strict_k64_result, strict_k64_low_host, strict_k64_out_host,
        "production REQUIRE A+B K64");
    const bool k32_replay_ready = replay_ready(
        k32_replay_result, k32_replay_host,
        "production K32-low standalone B-TILE8 replay");

    tile8_low_host.resize(low_count);
    tile8_out_host.resize(out_count);
    default_low_host.resize(low_count);
    default_out_host.resize(out_count);
    k32_low_host.resize(low_count);
    k32_out_host.resize(out_count);
    k32_replay_host.resize(out_count);
    k64_low_host.resize(low_count);
    k64_out_host.resize(out_count);
    strict_k32_low_host.resize(low_count);
    strict_k32_out_host.resize(out_count);
    strict_k64_low_host.resize(low_count);
    strict_k64_out_host.resize(out_count);

    bool comparisons_ok = true;
    if (tile8_result.low_read && default_result.low_read) {
        comparisons_ok = close_with_tolerance(
            default_low_host, tile8_low_host,
            2.0f, 3.0e-2f,
            "production clean-default A-WMMA vs forced TILE8") &&
            comparisons_ok;
    } else {
        comparisons_ok = false;
    }
    if (tile8_result.out_read && default_result.out_read) {
        comparisons_ok = close_with_tolerance(
            default_out_host, tile8_out_host,
            16.0f, 8.0e-2f,
            "production clean-default A-WMMA/B-TILE8 vs all-TILE8",
            false) && comparisons_ok;
    } else {
        comparisons_ok = false;
    }
    if (tile8_result.low_read && k32_result.low_read) {
        comparisons_ok = close_with_tolerance(
            k32_low_host, tile8_low_host, 2.0f, 3.0e-2f,
            "production ENABLE output-A K32 vs TILE8") && comparisons_ok;
    } else {
        comparisons_ok = false;
    }
    if (k32_result.out_read && k32_replay_result.out_read) {
        comparisons_ok = bitwise_equal(
            k32_out_host, k32_replay_host,
            "production ENABLE K32 B vs standalone TILE8 replay") &&
            comparisons_ok;
    } else {
        comparisons_ok = false;
    }
    if (k32_result.low_read && k64_result.low_read) {
        comparisons_ok = bitwise_equal(
            k64_low_host, k32_low_host,
            "production ENABLE output-A K64 vs K32") && comparisons_ok;
    } else {
        comparisons_ok = false;
    }
    if (k32_result.out_read && k64_result.out_read) {
        comparisons_ok = bitwise_equal(
            k64_out_host, k32_out_host,
            "production ENABLE A-K64/B-TILE8 vs A-K32/B-TILE8") &&
            comparisons_ok;
    } else {
        comparisons_ok = false;
    }
    if (default_result.low_read && k64_result.low_read) {
        comparisons_ok = bitwise_equal(
            default_low_host, k64_low_host,
            "production clean-default A vs explicit ENABLE A-K64") &&
            comparisons_ok;
    } else {
        comparisons_ok = false;
    }
    if (default_result.out_read && k64_result.out_read) {
        comparisons_ok = bitwise_equal(
            default_out_host, k64_out_host,
            "production clean-default B vs explicit ENABLE B-TILE8") &&
            comparisons_ok;
    } else {
        comparisons_ok = false;
    }
    if (strict_k32_result.low_read && strict_k64_result.low_read) {
        comparisons_ok = bitwise_equal(
            strict_k64_low_host, strict_k32_low_host,
            "production REQUIRE output-A K64 vs K32") && comparisons_ok;
    } else {
        comparisons_ok = false;
    }
    if (strict_k32_result.out_read && strict_k64_result.out_read) {
        comparisons_ok = bitwise_equal(
            strict_k64_out_host, strict_k32_out_host,
            "production REQUIRE output A+B K64 vs K32") && comparisons_ok;
    } else {
        comparisons_ok = false;
    }

    const bool ok = tile8_ready && default_ready && k32_ready && k64_ready &&
        strict_k32_ready && strict_k64_ready && tile8_memory &&
        default_memory && k32_memory && k64_memory && strict_k32_memory &&
        strict_k64_memory && k32_replay_ready && comparisons_ok;
    std::fprintf(stderr,
                 "ROCm Q4 production output WMMA policy: "
                 "tile8=%d/%llu/%llu default=%d/%llu/%llu "
                 "enable-k32=%d/%llu/%llu enable-k64=%d/%llu/%llu "
                 "require-k32=%d/%llu/%llu require-k64=%d/%llu/%llu %s\n",
                 tile8_result.rc,
                 (unsigned long long)tile8_result.wmma_calls,
                 (unsigned long long)tile8_result.k64_calls,
                 default_result.rc,
                 (unsigned long long)default_result.wmma_calls,
                 (unsigned long long)default_result.k64_calls,
                 k32_result.rc,
                 (unsigned long long)k32_result.wmma_calls,
                 (unsigned long long)k32_result.k64_calls,
                 k64_result.rc,
                 (unsigned long long)k64_result.wmma_calls,
                 (unsigned long long)k64_result.k64_calls,
                 strict_k32_result.rc,
                 (unsigned long long)strict_k32_result.wmma_calls,
                 (unsigned long long)strict_k32_result.k64_calls,
                 strict_k64_result.rc,
                 (unsigned long long)strict_k64_result.wmma_calls,
                 (unsigned long long)strict_k64_result.k64_calls,
                 ok ? "PASS" : "FAIL");
    return ok;
}

int detect_rocm_device() {
#if DS4_TEST_HAS_HIP_RUNTIME
    int count = 0;
    const hipError_t err = hipGetDeviceCount(&count);
    if (err != hipSuccess || count <= 0) {
        std::fprintf(stderr,
                     "ROCm Q4 dense/pair/prefill: SKIP "
                     "(HIP runtime has no visible device: %s)\n",
                     err == hipSuccess ? "device count is zero" : hipGetErrorString(err));
        return 0;
    }
    return count;
#else
    std::fprintf(stderr,
                 "ROCm Q4 dense/pair/prefill: SKIP "
                 "(compiled without HIP runtime headers)\n");
    return 0;
#endif
}

}  // namespace

int main(int argc, char **argv) {
    bool run_dense = true;
    bool run_pair = true;
    bool run_prefill = true;
    bool run_grouped_decode = true;
    bool run_prefill_long = false;
    bool run_policy_only = false;
    bool run_load4_only = false;
    if (argc == 2 && std::strcmp(argv[1], "--dense") == 0) {
        run_pair = false;
        run_prefill = false;
        run_grouped_decode = false;
    } else if (argc == 2 && std::strcmp(argv[1], "--pair") == 0) {
        run_dense = false;
        run_prefill = false;
        run_grouped_decode = false;
    } else if (argc == 2 && std::strcmp(argv[1], "--prefill-wmma-load4") == 0) {
        run_dense = run_pair = run_prefill = run_grouped_decode = false;
        run_load4_only = true;
    } else if (argc == 2 && std::strcmp(argv[1], "--prefill") == 0) {
        run_dense = false;
        run_pair = false;
        run_grouped_decode = false;
    } else if (argc == 2 &&
               std::strcmp(argv[1], "--grouped-decode") == 0) {
        run_dense = false;
        run_pair = false;
        run_prefill = false;
    } else if (argc == 2 &&
               std::strcmp(argv[1], "--prefill-long") == 0) {
        run_dense = false;
        run_pair = false;
        run_grouped_decode = false;
        run_prefill_long = true;
    } else if (argc == 2 && std::strcmp(argv[1], "--policy") == 0) {
        run_dense = false;
        run_pair = false;
        run_prefill = false;
        run_grouped_decode = false;
        run_policy_only = true;
    } else if (argc > 1 &&
               !(argc == 2 && std::strcmp(argv[1], "--all") == 0)) {
        std::fprintf(stderr,
                     "usage: %s [--all|--dense|--pair|--grouped-decode|"
                     "--prefill|--prefill-wmma-load4|--prefill-long|--policy]\n",
                     argv[0]);
        return 2;
    }

    env_snapshot prefill_enable(kPrefillEnable);
    env_snapshot prefill_disable(kPrefillDisable);
    env_snapshot prefill_require(kPrefillRequire);
    env_snapshot tile4_ssd_enable(kPrefillK1024Tile4SsdEnable);
    env_snapshot tile4_disable(kPrefillK1024Tile4Disable);
    env_snapshot tile4_require(kPrefillK1024Tile4Require);
    env_snapshot wmma_enable(kPrefillWmmaEnable);
    env_snapshot wmma_ssd_enable(kPrefillWmmaSsdEnable);
    env_snapshot wmma_disable(kPrefillWmmaDisable);
    env_snapshot wmma_require(kPrefillWmmaRequire);
    env_snapshot wmma_row_tile(kPrefillWmmaRowTile);
    env_snapshot wmma_k64_global(kPrefillWmmaK64);
    env_snapshot wmma_k128_disable_global(kPrefillWmmaK128Disable);
    env_snapshot wmma_k64_load4_disable_global(kPrefillWmmaK64Load4Disable);
    (void)unsetenv(kPrefillWmmaK64Load4Disable);
    env_snapshot q8_wave32_enable(kPrefillQ8Wave32Enable);
    env_snapshot q8_wave32_disable(kPrefillQ8Wave32Disable);
    env_snapshot q8_wave32_require(kPrefillQ8Wave32Require);
    env_snapshot grouped_enable(kGroupedDecodeEnable);
    env_snapshot grouped_disable(kGroupedDecodeDisable);
    env_snapshot grouped_require(kGroupedDecodeRequire);
    env_snapshot grouped_stats(kGroupedDecodeStats);
    (void)unsetenv(kPrefillEnable);
    (void)unsetenv(kPrefillDisable);
    (void)unsetenv(kPrefillRequire);
    (void)unsetenv(kPrefillK1024Tile4SsdEnable);
    (void)unsetenv(kPrefillK1024Tile4Disable);
    (void)unsetenv(kPrefillK1024Tile4Require);
    (void)unsetenv(kPrefillWmmaEnable);
    (void)unsetenv(kPrefillWmmaSsdEnable);
    (void)unsetenv(kPrefillWmmaDisable);
    (void)unsetenv(kPrefillWmmaRequire);
    (void)unsetenv(kPrefillWmmaRowTile);
    /* Neutralize wider K64/K128 staging for every non-staging oracle. */
    (void)setenv(kPrefillWmmaK64, "0", 1);
    (void)unsetenv(kPrefillWmmaK128Disable);
    (void)unsetenv(kPrefillQ8Wave32Enable);
    (void)unsetenv(kPrefillQ8Wave32Disable);
    (void)unsetenv(kPrefillQ8Wave32Require);
    (void)unsetenv(kGroupedDecodeEnable);
    (void)unsetenv(kGroupedDecodeDisable);
    (void)unsetenv(kGroupedDecodeRequire);
    (void)unsetenv(kGroupedDecodeStats);

    const bool policy_ok = run_prefill_wmma_requested_policy_oracle() &&
                           run_prefill_wmma_row_tile_policy_oracle() &&
                           run_prefill_k1024_tile4_policy_oracle() &&
                           run_prefill_q8_wave32_policy_oracle() &&
                           run_pair_pre_enqueue_policy_oracle();
    if (run_policy_only || !policy_ok) return policy_ok ? 0 : 1;

    if (detect_rocm_device() <= 0) {
        const char *require_device =
            std::getenv("DS4_TEST_REQUIRE_ROCM_DEVICE");
        const bool required = require_device && require_device[0] != '\0' &&
                              std::strcmp(require_device, "0") != 0;
        return required ? 1 : kSkip;
    }
    if (!ds4_gpu_init()) {
        std::fprintf(stderr,
                     "ROCm Q4 dense/pair/prefill: FAIL "
                     "(device is visible but ds4_gpu_init failed)\n");
        return 1;
    }

    aligned_model model;
    bool ok = policy_ok && make_model(&model);
    if (!ok) {
        std::fprintf(stderr,
                     "ROCm Q4 dense/pair/prefill: fixture allocation FAIL\n");
    } else if (!ds4_gpu_set_model_map(model.data, model.size)) {
        std::fprintf(stderr,
                     "ROCm Q4 dense/pair/prefill: model-map registration FAIL\n");
        ok = false;
    }

    const bool model_ready = ok;
    if (model_ready && run_load4_only) {
        ok = run_prefill_wmma_load4_oracle(model, true) && ok;
    }
    if (model_ready && run_dense) {
        std::fprintf(stderr, "ROCm Q4 dense oracle (raw GGUF Q4_K x Q8_K):\n");
        ok = run_dense_case(model, 1u, model.weight0_offset, kM0,
                            "dense n_tok=1") && ok;
        ok = run_dense_case(model, 3u, model.weight1_offset, kM1,
                            "dense n_tok=3") && ok;
        ok = run_dense_case(model, 9u, model.weight1_offset, kM1,
                            "dense n_tok=9") && ok;
        ok = run_dense_case(model, 128u, model.weight1_offset, kM1,
                            "dense n_tok=128") && ok;
        ok = run_dense_guards(model) && ok;
    }
    if (model_ready && run_pair) {
        std::fprintf(stderr, "ROCm Q4 pair parity (pair vs two dense):\n");
        env_snapshot enable("DS4_ROCM_ENABLE_Q4_DENSE_PAIR");
        env_snapshot disable("DS4_ROCM_DISABLE_Q4_DENSE_PAIR");
        (void)setenv("DS4_ROCM_ENABLE_Q4_DENSE_PAIR", "1", 1);
        (void)unsetenv("DS4_ROCM_DISABLE_Q4_DENSE_PAIR");
        const bool pair1_ok = run_pair_case(model, 1u, "pair n_tok=1");
        const bool pair1_tail_ok = run_pair_case(
            model, 1u, "pair K=1024 n_tok=1", false, kTailK);
        const bool pair3_ok = run_pair_case(
            model, 3u, "pair n_tok=3 reverse M=(33,65)", true);
        const bool pair8_ok = run_pair_case(model, 8u, "pair n_tok=8");
        const bool pair_guard_ok = run_pair_guards(model);
        const bool pair_opt_in_ok = run_pair_opt_in_guards(model);
        ok = pair1_ok && pair1_tail_ok && pair3_ok && pair8_ok && pair_guard_ok &&
             pair_opt_in_ok && ok;
    }
    if (model_ready && run_grouped_decode) {
        std::fprintf(stderr,
                     "ROCm Q4 grouped attention-A decode parity "
                     "(one grouped dispatch vs eight legacy calls):\n");
        ok = run_grouped_attention_decode_case(model) && ok;
    }
    if (model_ready && run_prefill) {
        std::fprintf(stderr,
                     "ROCm Q4 tiled prefill parity "
                     "(forced DISABLE vs default+REQUIRE):\n");
        const bool q8_wave32_ok = run_prefill_q8_quantizer_oracle(model);
        const bool prefill9_ok = run_prefill_parity_case(
            model, 9u, model.weight0_offset, kM0, true,
            "prefill K=4096 M=65 n_tok=9");
        const bool prefill30_ok = run_prefill_parity_case(
            model, 30u, model.weight0_offset, kM0, true,
            "prefill K=4096 M=65 n_tok=30 (token-tail nt=6)");
        const bool prefill128_ok = run_prefill_parity_case(
            model, 128u, model.weight1_offset, kM1, true,
            "prefill K=4096 M=33 n_tok=128");
        const bool prefill_tail9_ok = run_prefill_parity_case(
            model, 9u, model.tail_k1024_offset, kM0, true,
            "prefill K=1024 M=65 n_tok=9 (K-tail nb=4)", kTailK);
        const bool prefill_tail128_ok = run_prefill_parity_case(
            model, 128u, model.tail_k1024_offset, kM0, true,
            "prefill K=1024 M=65 n_tok=128 (K-tail nb=4)", kTailK);
        const bool prefill_q_b_tile4_ok = run_prefill_parity_case(
            model, 9u, model.q_b_k1024_offset, kQbOutDim, false,
            "prefill q_b K=1024 M=32768 n_tok=9 (tile4)", kTailK);
        const bool q_b_f16_null_qhalf_ok =
            run_q_b_f16_null_qhalf_case(model);
        const bool prefill_single9_ok = run_prefill_parity_case(
            model, 9u, model.attn_b_offset, kAttnOutDim, true,
            "prefill K=256 M=65 n_tok=9 (K-tail nb=1)", kAttnLowDim);
        const bool prefill_single128_ok = run_prefill_parity_case(
            model, 128u, model.attn_b_offset, kAttnOutDim, true,
            "prefill K=256 M=65 n_tok=128 (K-tail nb=1)", kAttnLowDim);
        const bool prefill_pair9_ok =
            run_prefill_pair_case(model, 9u, false, kTailK);
        const bool prefill_pair30_reverse_ok =
            run_prefill_pair_case(model, 30u, true);
        const bool prefill_pair128_ok =
            run_prefill_pair_case(model, 128u, false);
        const bool attention9_ok = run_attention_prefill_case(
            model, 9u,
            "attention prefill groups=8 K=4096 rank=32 M=65 n_tok=9");
        const bool attention30_ok = run_attention_prefill_case(
            model, 30u,
            "attention prefill groups=8 K=4096 rank=32 M=65 "
            "n_tok=30 (token-tail nt=6)");
        const bool attention128_ok = run_attention_prefill_case(
            model, 128u,
            "attention prefill groups=8 K=4096 rank=32 M=65 n_tok=128");
        const bool attention_q8_9_ok = run_attention_prefill_case(
            model, 9u,
            "attention prefill Q4-A/Q8-B groups=8 K=4096 rank=32 M=65 "
            "n_tok=9",
            kQ8Type);
        const bool attention_q8_30_ok = run_attention_prefill_case(
            model, 30u,
            "attention prefill Q4-A/Q8-B groups=8 K=4096 rank=32 M=65 "
            "n_tok=30 (token-tail nt=6)",
            kQ8Type);
        const bool prefill_wmma_ok = run_prefill_wmma_smoke(model);
        const bool prefill_load4_ok = run_prefill_wmma_load4_oracle(model);
        const bool output_wmma_ok =
            run_attention_output_wmma_smoke(model);
        const bool gate_ok = run_prefill_gate_guards(model);
        ok = q8_wave32_ok && prefill9_ok && prefill30_ok && prefill128_ok &&
             prefill_tail9_ok &&
             prefill_tail128_ok && prefill_single9_ok &&
             prefill_q_b_tile4_ok && q_b_f16_null_qhalf_ok &&
             prefill_single128_ok && prefill_pair9_ok &&
             prefill_pair30_reverse_ok && prefill_pair128_ok && attention9_ok &&
             attention30_ok && attention128_ok && attention_q8_9_ok &&
             attention_q8_30_ok && prefill_wmma_ok && prefill_load4_ok && output_wmma_ok &&
             gate_ok && ok;
        if (run_prefill_long) {
            // A 64 MiB activation and a roughly 0.5 Gi-op projection stress
            // arbitrary token-grid tails without the much slower CPU oracle.
            const bool long_ok = run_prefill_parity_case(
                model, 4096u, model.weight1_offset, kM1, false,
                "prefill stress K=4096 M=33 n_tok=4096");
            ok = long_ok && ok;
        }
        /* Run the file-backed SSD oracle last.  Switching model maps marks the
         * process multi-model and intentionally disables optional caches; no
         * later parity case should inherit that conservative policy. */
        const bool prefill_q_b_tile4_ssd_ok =
            run_prefill_k1024_tile4_ssd_case(model);
        ok = prefill_q_b_tile4_ssd_ok && ok;
    }

    // Registered host ranges must be released before their aligned backing
    // allocation is destroyed.
    ds4_gpu_cleanup();
    std::fprintf(stderr, "ROCm Q4 dense/pair/prefill oracle: %s\n",
                 ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
