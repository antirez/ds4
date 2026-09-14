#define _DARWIN_C_SOURCE

/* GGUF-free production-shape oracle for the DeepSeek Flash Q4_K indexer
 * query projection.  The fixture deliberately uses the real 1024 -> 8192
 * geometry and checks the dispatch boundaries around the Metal matvec,
 * generic matmul, and 32-token matrix paths. */

#include "ds4_gpu.h"

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

bool ds4_log_is_tty(FILE *fp) {
    (void)fp;
    return false;
}

#ifdef __APPLE__

enum {
    Q4_K_TYPE = 12u,
    QK_K = 256u,
    INDEXER_IN_DIM = 1024u,
    INDEXER_OUT_DIM = 8192u,
    BLOCKS_PER_ROW = INDEXER_IN_DIM / QK_K,
    GROUPS_PER_BLOCK = 8u,
    GROUP_SIZE = 32u,
    Q_PATTERNS = 16u,
    MAX_TOKENS = 33u,
    GUARD_FLOATS = 64u,
};

typedef struct {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[12];
    uint8_t qs[QK_K / 2u];
} block_q4_K;

static const float k_abs_tolerance = 2.0e-3f;
static const float k_rel_tolerance = 3.0e-5f;
static const uint32_t k_poison_base = 0x7fc10000u;

static void fail(const char *what) {
    fprintf(stderr, "Metal indexer Q4_K oracle FAIL: %s\n", what);
    exit(1);
}

#define CHECK(expr, what) do { if (!(expr)) fail(what); } while (0)

static uint64_t align_up(uint64_t value, uint64_t alignment) {
    return (value + alignment - 1u) / alignment * alignment;
}

static float f16_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16u;
    uint32_t exp = (h >> 10u) & 0x1fu;
    uint32_t mant = h & 0x03ffu;
    uint32_t bits;

    if (exp == 0u) {
        if (mant == 0u) {
            bits = sign;
        } else {
            exp = 1u;
            while ((mant & 0x0400u) == 0u) {
                mant <<= 1u;
                exp--;
            }
            mant &= 0x03ffu;
            bits = sign | ((exp + 127u - 15u) << 23u) | (mant << 13u);
        }
    } else if (exp == 31u) {
        bits = sign | 0x7f800000u | (mant << 13u);
    } else {
        bits = sign | ((exp + 127u - 15u) << 23u) | (mant << 13u);
    }

    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static void q4_pack_scales(uint8_t packed[12],
                           const uint8_t scales[GROUPS_PER_BLOCK],
                           const uint8_t minima[GROUPS_PER_BLOCK]) {
    memset(packed, 0, 12u);
    for (uint32_t group = 0; group < 4u; group++) {
        packed[group] = scales[group] & 63u;
        packed[group + 4u] = minima[group] & 63u;
    }
    for (uint32_t group = 4u; group < GROUPS_PER_BLOCK; group++) {
        packed[group + 4u] = (scales[group] & 15u) |
                             ((minima[group] & 15u) << 4u);
        packed[group - 4u] |= (scales[group] >> 4u) << 6u;
        packed[group] |= (minima[group] >> 4u) << 6u;
    }
}

static void q4_scale_min(const uint8_t packed[12], uint32_t group,
                         uint8_t *scale, uint8_t *minimum) {
    if (group < 4u) {
        *scale = packed[group] & 63u;
        *minimum = packed[group + 4u] & 63u;
    } else {
        *scale = (packed[group + 4u] & 15u) |
                 ((packed[group - 4u] >> 6u) << 4u);
        *minimum = (packed[group + 4u] >> 4u) |
                   ((packed[group] >> 6u) << 4u);
    }
}

static uint32_t q4_value(const block_q4_K *block,
                         uint32_t group, uint32_t lane) {
    const uint32_t byte_offset = (group >> 1u) * GROUP_SIZE + lane;
    const uint32_t shift = (group & 1u) * 4u;
    return (block->qs[byte_offset] >> shift) & 15u;
}

static void fill_q4_indexer(block_q4_K *matrix) {
    CHECK(sizeof(block_q4_K) == 144u, "unexpected Q4_K block size");

    for (uint32_t row = 0; row < INDEXER_OUT_DIM; row++) {
        const uint32_t pattern = row & (Q_PATTERNS - 1u);
        for (uint32_t block = 0; block < BLOCKS_PER_ROW; block++) {
            block_q4_K *b = matrix +
                (uint64_t)row * BLOCKS_PER_ROW + block;
            uint8_t scales[GROUPS_PER_BLOCK];
            uint8_t minima[GROUPS_PER_BLOCK];

            for (uint32_t group = 0; group < GROUPS_PER_BLOCK; group++) {
                scales[group] = (uint8_t)(1u +
                    (row * 13u + block * 7u + group * 5u + 3u) % 31u);
                minima[group] = (uint8_t)(
                    (row * 11u + block * 3u + group * 7u + 1u) % 17u);
            }
            q4_pack_scales(b->scales, scales, minima);
            memset(b->qs, 0, sizeof(b->qs));

            for (uint32_t group = 0; group < GROUPS_PER_BLOCK; group++) {
                for (uint32_t lane = 0; lane < GROUP_SIZE; lane++) {
                    const uint32_t q =
                        (lane * 5u + pattern * 3u + block * 7u +
                         group * 11u) & 15u;
                    const uint32_t byte_offset =
                        (group >> 1u) * GROUP_SIZE + lane;
                    const uint32_t shift = (group & 1u) * 4u;
                    b->qs[byte_offset] |= (uint8_t)(q << shift);
                }
            }

            /* Exact binary scales: d=2^-8 and dmin=2^-9. */
            b->d = 0x1c00u;
            b->dmin = 0x1800u;
        }
    }
}

static void fill_inputs(float *inputs) {
    for (uint32_t token = 0; token < MAX_TOKENS; token++) {
        for (uint32_t col = 0; col < INDEXER_IN_DIM; col++) {
            const uint32_t key = token * 131u + col * 17u +
                                 ((col >> 3u) ^ (token * 29u));
            /* Multiples of 2^-6 are exactly representable by the half RHS
             * used in the Metal matrix kernels. */
            inputs[(uint64_t)token * INDEXER_IN_DIM + col] =
                (float)((int)(key % 129u) - 64) / 64.0f;
        }
    }
}

static void build_cpu_dequant_oracle(const block_q4_K *matrix,
                                     const float *inputs,
                                     float *reference) {
    float group_sum[MAX_TOKENS][BLOCKS_PER_ROW][GROUPS_PER_BLOCK];
    float q_dot[MAX_TOKENS][Q_PATTERNS]
               [BLOCKS_PER_ROW][GROUPS_PER_BLOCK];

    memset(group_sum, 0, sizeof(group_sum));
    memset(q_dot, 0, sizeof(q_dot));

    /* q nibbles repeat every Q_PATTERNS rows, while scales/minima remain
     * row-specific.  Factoring these sums preserves the exact dequantized
     * dot-product algebra and keeps the production-size oracle inexpensive. */
    for (uint32_t token = 0; token < MAX_TOKENS; token++) {
        const float *x = inputs + (uint64_t)token * INDEXER_IN_DIM;
        for (uint32_t block = 0; block < BLOCKS_PER_ROW; block++) {
            for (uint32_t group = 0; group < GROUPS_PER_BLOCK; group++) {
                const uint32_t col0 = block * QK_K + group * GROUP_SIZE;
                float x_sum = 0.0f;
                for (uint32_t lane = 0; lane < GROUP_SIZE; lane++) {
                    x_sum += x[col0 + lane];
                }
                group_sum[token][block][group] = x_sum;

                for (uint32_t pattern = 0; pattern < Q_PATTERNS; pattern++) {
                    const block_q4_K *b = matrix +
                        (uint64_t)pattern * BLOCKS_PER_ROW + block;
                    float sum = 0.0f;
                    for (uint32_t lane = 0; lane < GROUP_SIZE; lane++) {
                        sum += (float)q4_value(b, group, lane) *
                               x[col0 + lane];
                    }
                    q_dot[token][pattern][block][group] = sum;
                }
            }
        }
    }

    for (uint32_t token = 0; token < MAX_TOKENS; token++) {
        for (uint32_t row = 0; row < INDEXER_OUT_DIM; row++) {
            const uint32_t pattern = row & (Q_PATTERNS - 1u);
            const block_q4_K *row_blocks = matrix +
                (uint64_t)row * BLOCKS_PER_ROW;
            float acc = 0.0f;

            for (uint32_t block = 0; block < BLOCKS_PER_ROW; block++) {
                const block_q4_K *b = row_blocks + block;
                const float d = f16_to_f32(b->d);
                const float dmin = f16_to_f32(b->dmin);
                for (uint32_t group = 0; group < GROUPS_PER_BLOCK; group++) {
                    uint8_t scale = 0;
                    uint8_t minimum = 0;
                    q4_scale_min(b->scales, group, &scale, &minimum);
                    acc += d * (float)scale *
                               q_dot[token][pattern][block][group] -
                           dmin * (float)minimum *
                               group_sum[token][block][group];
                }
            }
            reference[(uint64_t)token * INDEXER_OUT_DIM + row] = acc;
        }
    }
}

static uint32_t poison_bits(uint64_t index) {
    return k_poison_base + (uint32_t)(index & 0xffffu);
}

static void poison(float *values, uint64_t count) {
    for (uint64_t i = 0; i < count; i++) {
        const uint32_t bits = poison_bits(i);
        memcpy(&values[i], &bits, sizeof(bits));
    }
}

static uint64_t count_canary_failures(const float *values,
                                      uint64_t begin, uint64_t end,
                                      uint64_t *first) {
    uint64_t failures = 0;
    *first = UINT64_MAX;
    for (uint64_t i = begin; i < end; i++) {
        uint32_t actual = 0;
        memcpy(&actual, &values[i], sizeof(actual));
        if (actual != poison_bits(i)) {
            if (*first == UINT64_MAX) *first = i;
            failures++;
        }
    }
    return failures;
}

static bool compare_case(const float *actual, const float *reference,
                         uint32_t n_tokens) {
    const uint64_t count = (uint64_t)n_tokens * INDEXER_OUT_DIM;
    uint64_t raw_mismatches = 0;
    uint64_t tolerance_failures = 0;
    uint64_t worst = 0;
    float max_abs = 0.0f;
    float max_rel = 0.0f;

    for (uint64_t i = 0; i < count; i++) {
        if (memcmp(&actual[i], &reference[i], sizeof(float)) != 0) {
            raw_mismatches++;
        }
        const float diff = fabsf(actual[i] - reference[i]);
        const float rel = diff / fmaxf(1.0f, fabsf(reference[i]));
        const float limit = k_abs_tolerance +
                            k_rel_tolerance * fabsf(reference[i]);
        if (diff > max_abs) {
            max_abs = diff;
            worst = i;
        }
        if (rel > max_rel) max_rel = rel;
        if (!isfinite(actual[i]) || diff > limit) tolerance_failures++;
    }

    fprintf(stderr,
            "Metal indexer Q4_K n_tok=%u: raw=%llu/%llu "
            "max_abs=%g max_rel=%g tol(abs=%g rel=%g) %s\n",
            n_tokens,
            (unsigned long long)raw_mismatches,
            (unsigned long long)count,
            max_abs, max_rel, k_abs_tolerance, k_rel_tolerance,
            tolerance_failures == 0 ? "PASS" : "FAIL");
    if (tolerance_failures != 0) {
        fprintf(stderr,
                "  worst token=%llu row=%llu gpu=%g cpu=%g delta=%g "
                "failures=%llu\n",
                (unsigned long long)(worst / INDEXER_OUT_DIM),
                (unsigned long long)(worst % INDEXER_OUT_DIM),
                actual[worst], reference[worst],
                actual[worst] - reference[worst],
                (unsigned long long)tolerance_failures);
    }
    return tolerance_failures == 0;
}

int main(void) {
    static const uint32_t token_cases[] = {
        1u, 8u, 9u, 16u, 17u, 31u, 32u, 33u,
    };
    const uint64_t row_bytes =
        (INDEXER_IN_DIM / QK_K) * sizeof(block_q4_K);
    const uint64_t weight_bytes = (uint64_t)INDEXER_OUT_DIM * row_bytes;
    const uint64_t page = (uint64_t)getpagesize();
    const uint64_t model_bytes = align_up(weight_bytes, page);
    const uint64_t input_count = (uint64_t)MAX_TOKENS * INDEXER_IN_DIM;
    const uint64_t max_output_count =
        (uint64_t)MAX_TOKENS * INDEXER_OUT_DIM;
    const uint64_t storage_count =
        GUARD_FLOATS + max_output_count + GUARD_FLOATS;

    CHECK(Q4_K_TYPE == 12u, "Q4_K GGUF type must be 12");
    CHECK(row_bytes == 576u, "unexpected production Q4_K row size");
    CHECK(weight_bytes == 4718592u,
          "unexpected production Q4_K indexer size");
    CHECK(unsetenv("DS4_METAL_DISABLE_Q4_MV_CLASSIC") == 0,
          "clear classic Q4_K kill switch");
    CHECK(ds4_gpu_init() != 0, "Metal init");

    void *model = NULL;
    CHECK(posix_memalign(&model, (size_t)page, (size_t)model_bytes) == 0,
          "page-aligned model allocation");
    memset(model, 0, (size_t)model_bytes);
    fill_q4_indexer(model);

    float *inputs = malloc((size_t)input_count * sizeof(float));
    float *reference = malloc((size_t)max_output_count * sizeof(float));
    float *storage = malloc((size_t)storage_count * sizeof(float));
    CHECK(inputs && reference && storage, "host tensor allocation");
    fill_inputs(inputs);
    build_cpu_dequant_oracle(model, inputs, reference);

    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(input_count * sizeof(float));
    ds4_gpu_tensor *out_base =
        ds4_gpu_tensor_alloc(storage_count * sizeof(float));
    CHECK(x && out_base, "Metal tensor allocation");
    CHECK(ds4_gpu_tensor_write(x, 0, inputs,
                               input_count * sizeof(float)) != 0,
          "input upload");
    CHECK(ds4_gpu_set_model_map(model, model_bytes) != 0, "model map");
    ds4_gpu_set_quality(false);
    ds4_gpu_set_ssd_streaming(false);

    bool passed = true;
    for (uint32_t case_i = 0;
         case_i < sizeof(token_cases) / sizeof(token_cases[0]);
         case_i++) {
        const uint32_t n_tokens = token_cases[case_i];
        const uint64_t output_count =
            (uint64_t)n_tokens * INDEXER_OUT_DIM;
        const uint64_t output_bytes = output_count * sizeof(float);

        poison(storage, storage_count);
        CHECK(ds4_gpu_tensor_write(out_base, 0, storage,
                                   storage_count * sizeof(float)) != 0,
              "output poison upload");
        ds4_gpu_tensor *out = ds4_gpu_tensor_view(
            out_base, GUARD_FLOATS * sizeof(float), output_bytes);
        CHECK(out != NULL, "exact-size output view");

        /* This is the production entry point and Q4_K's GGUF type is 12. */
        CHECK(ds4_gpu_matmul_quant_tensor(
                  out, model, model_bytes, 0, Q4_K_TYPE,
                  INDEXER_IN_DIM, INDEXER_OUT_DIM, x, n_tokens) != 0,
              "ds4_gpu_matmul_quant_tensor(type=12)");
        ds4_gpu_tensor_free(out);

        CHECK(ds4_gpu_tensor_read(out_base, 0, storage,
                                  storage_count * sizeof(float)) != 0,
              "output readback");

        const bool values_ok = compare_case(
            storage + GUARD_FLOATS, reference, n_tokens);
        uint64_t first_prefix = UINT64_MAX;
        uint64_t first_suffix = UINT64_MAX;
        const uint64_t prefix_failures = count_canary_failures(
            storage, 0, GUARD_FLOATS, &first_prefix);
        const uint64_t suffix_begin = GUARD_FLOATS + output_count;
        const uint64_t suffix_failures = count_canary_failures(
            storage, suffix_begin, storage_count, &first_suffix);
        const bool canary_ok = prefix_failures == 0 && suffix_failures == 0;

        fprintf(stderr,
                "Metal indexer Q4_K n_tok=%u canary: prefix=%llu "
                "suffix=%llu %s\n",
                n_tokens,
                (unsigned long long)prefix_failures,
                (unsigned long long)suffix_failures,
                canary_ok ? "PASS" : "FAIL");
        if (!canary_ok) {
            fprintf(stderr, "  first prefix=%llu first suffix=%llu\n",
                    (unsigned long long)first_prefix,
                    (unsigned long long)first_suffix);
        }
        if (!values_ok || !canary_ok) passed = false;
    }

    ds4_gpu_tensor_free(out_base);
    ds4_gpu_tensor_free(x);
    ds4_gpu_cleanup();
    free(storage);
    free(reference);
    free(inputs);
    free(model);

    fprintf(stderr,
            "Metal indexer Q4_K production geometry 1024x8192: %s\n",
            passed ? "PASS" : "FAIL");
    return passed ? 0 : 1;
}

#else

int main(void) {
    fprintf(stderr, "Metal indexer Q4_K oracle SKIP: requires macOS\n");
    return 0;
}

#endif
