#define _DARWIN_C_SOURCE

/* GGUF-free production-shape oracle for the resident pre-M5 Metal Q4_K
 * attn_q_b F16 weight sidecar.  The candidate is compared bit-for-bit with
 * the established Q4_K matmul followed by the exact same head norm/RoPE
 * entry point used by the production fallback. */

#include "ds4_gpu.h"

#include <errno.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

bool ds4_log_is_tty(FILE *fp) {
    (void)fp;
    return false;
}

#ifdef __APPLE__

enum {
    Q4_K_TYPE = 12u,
    QK_K = 256u,
    IN_DIM = 1024u,
    OUT_DIM = 32768u,
    N_HEAD = 64u,
    HEAD_DIM = 512u,
    N_ROT = 64u,
    MAX_TOKENS = 64u,
    BLOCKS_PER_ROW = IN_DIM / QK_K,
    GROUPS_PER_BLOCK = 8u,
    GROUP_SIZE = 32u,
    GUARD_FLOATS = 256u,
    GUARD_HALFS = 256u,
    TIMING_SAMPLES = 8u,
    SSD_SOURCE_LEADING = 128u,
};

typedef struct {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[12];
    uint8_t qs[QK_K / 2u];
} block_q4_K;

static const char *k_disable =
    "DS4_METAL_DISABLE_Q4_ATTN_Q_B_F16_CACHE";
static const char *k_enable_ssd_streaming =
    "DS4_METAL_ENABLE_Q4_ATTN_Q_B_F16_CACHE_WITH_SSD_STREAMING";
static const char *k_disable_f16_rhs =
    "DS4_METAL_DISABLE_Q4_ATTN_Q_B_F16_RHS";
static const char *k_disable_transient_f16 =
    "DS4_METAL_DISABLE_Q4_ATTN_Q_B_TRANSIENT_F16";
static const char *k_require =
    "DS4_METAL_REQUIRE_Q4_ATTN_Q_B_F16_CACHE";
static const char *k_min_tokens =
    "DS4_METAL_Q4_ATTN_Q_B_F16_CACHE_MIN_TOKENS";
static const char *k_transient_min_tokens =
    "DS4_METAL_Q4_ATTN_Q_B_TRANSIENT_F16_MIN_TOKENS";
static const char *k_cache_mb =
    "DS4_METAL_Q4_ATTN_Q_B_F16_CACHE_MB";
static const char *k_timing =
    "DS4_TEST_METAL_Q4_QB_F16_CACHE_TIMING";

static const uint32_t k_reference_poison = 0x7fc10000u;
static const uint32_t k_candidate_poison = 0x7fc30000u;

static void fail(const char *what) {
    fprintf(stderr, "Metal Q4 attn_q_b F16 cache oracle FAIL: %s\n", what);
    exit(1);
}

#define CHECK(expr, what) do { if (!(expr)) fail(what); } while (0)

static uint64_t align_up(uint64_t value, uint64_t alignment) {
    return (value + alignment - 1u) / alignment * alignment;
}

static void pack_scales(uint8_t packed[12],
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

static void fill_q4_matrix(block_q4_K *matrix) {
    CHECK(sizeof(block_q4_K) == 144u, "unexpected Q4_K block size");

    for (uint32_t row = 0; row < OUT_DIM; row++) {
        for (uint32_t block = 0; block < BLOCKS_PER_ROW; block++) {
            block_q4_K *b = matrix +
                (uint64_t)row * BLOCKS_PER_ROW + block;
            const uint32_t key =
                row * 1009u + block * 313u +
                (row ^ (block * 17u)) + 29u;
            uint8_t scales[GROUPS_PER_BLOCK];
            uint8_t minima[GROUPS_PER_BLOCK];

            for (uint32_t group = 0; group < GROUPS_PER_BLOCK; group++) {
                scales[group] =
                    (uint8_t)((key + group * 37u) & 63u);
                minima[group] =
                    (uint8_t)((key / 3u + group * 29u) & 63u);
            }
            pack_scales(b->scales, scales, minima);
            for (uint32_t i = 0; i < QK_K / 2u; i++) {
                b->qs[i] = (uint8_t)(
                    key + i * 37u + (i >> 2u) * 11u);
            }

            /* Exercise non-power-of-two half scales and all packed 6-bit
             * scale/minimum lanes, including the high bits of groups 4--7. */
            b->d = (uint16_t)(0x1801u + (key & 0x01ffu));
            b->dmin = (uint16_t)(0x1403u + ((key >> 3u) & 0x01ffu));
        }
    }
}

static uint32_t poison_bits(uint32_t base, uint64_t index) {
    return base + (uint32_t)(index & 0xffffu);
}

static void poison_f32(float *values, uint64_t count, uint32_t base) {
    for (uint64_t i = 0; i < count; i++) {
        const uint32_t bits = poison_bits(base, i);
        memcpy(&values[i], &bits, sizeof(bits));
    }
}

static uint64_t count_poison_f32_mismatches(const float *values,
                                            uint64_t begin,
                                            uint64_t end,
                                            uint32_t base) {
    uint64_t mismatches = 0;
    for (uint64_t i = begin; i < end; i++) {
        uint32_t bits = 0;
        memcpy(&bits, &values[i], sizeof(bits));
        if (bits != poison_bits(base, i)) mismatches++;
    }
    return mismatches;
}

static uint16_t half_poison_bits(uint64_t index) {
    return (uint16_t)(0x7e00u | (uint16_t)(index & 0x01ffu));
}

static void poison_f16(uint16_t *values, uint64_t count) {
    for (uint64_t i = 0; i < count; i++) {
        values[i] = half_poison_bits(i);
    }
}

static uint64_t count_poison_f16_mismatches_range(
        const uint16_t *values,
        uint64_t        begin,
        uint64_t        end) {
    uint64_t mismatches = 0;
    for (uint64_t i = begin; i < end; i++) {
        if (values[i] != half_poison_bits(i)) mismatches++;
    }
    return mismatches;
}

static uint64_t count_poison_f16_mismatches(const uint16_t *values,
                                            uint64_t count) {
    return count_poison_f16_mismatches_range(values, 0, count);
}

static void fill_inputs(float *values) {
    for (uint32_t token = 0; token < MAX_TOKENS; token++) {
        for (uint32_t col = 0; col < IN_DIM; col++) {
            const uint32_t key = token * 131u + col * 17u +
                                 ((col >> 3u) ^ (token * 29u));
            /* Exercise values that are not exactly representable as half;
             * the F16-RHS path must match the legacy per-tile narrowing. */
            values[(uint64_t)token * IN_DIM + col] =
                (float)((int)(key % 129u) - 64) / 509.0f;
        }
    }
}

static uint64_t count_bit_mismatches(const float *reference,
                                     const float *candidate,
                                     uint64_t count,
                                     uint64_t *first) {
    uint64_t mismatches = 0;
    *first = UINT64_MAX;
    for (uint64_t i = 0; i < count; i++) {
        if (memcmp(&reference[i], &candidate[i], sizeof(float)) != 0) {
            if (*first == UINT64_MAX) *first = i;
            mismatches++;
        }
    }
    return mismatches;
}

static int run_reference_at(ds4_gpu_tensor *out,
                            const void *model,
                            uint64_t model_bytes,
                            uint64_t weight_offset,
                            const ds4_gpu_tensor *x,
                            uint32_t n_tok) {
    if (!ds4_gpu_matmul_quant_tensor(
            out, model, model_bytes, weight_offset, Q4_K_TYPE,
            IN_DIM, OUT_DIM, x, n_tok)) {
        return 0;
    }
    return ds4_gpu_head_rms_norm_rope_tail_tensor(
        out, n_tok, N_HEAD, HEAD_DIM, N_ROT,
        17u, 0u, false,
        10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f, 1.0e-6f);
}

static int run_reference(ds4_gpu_tensor *out,
                         const void *model,
                         uint64_t model_bytes,
                         const ds4_gpu_tensor *x,
                         uint32_t n_tok) {
    return run_reference_at(
        out, model, model_bytes, 0u, x, n_tok);
}

static int run_candidate_at(ds4_gpu_tensor *out,
                            ds4_gpu_tensor *q_half,
                            const void *model,
                            uint64_t model_bytes,
                            uint64_t weight_offset,
                            const ds4_gpu_tensor *x,
                            uint32_t n_tok) {
    return ds4_gpu_attn_q_b_f16_head_rms_rope_tail_tensor(
        out, q_half, model, model_bytes, weight_offset, Q4_K_TYPE,
        IN_DIM, OUT_DIM, x, n_tok, N_HEAD, HEAD_DIM, N_ROT,
        17u, 0u, false,
        10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f, 1.0e-6f);
}

static int run_candidate(ds4_gpu_tensor *out,
                         ds4_gpu_tensor *q_half,
                         const void *model,
                         uint64_t model_bytes,
                         const ds4_gpu_tensor *x,
                         uint32_t n_tok) {
    return run_candidate_at(
        out, q_half, model, model_bytes, 0u, x, n_tok);
}

static const char *mm_arm_name(ds4_gpu_test_q4_qb_mm_arm arm) {
    switch (arm) {
    case DS4_GPU_TEST_Q4_QB_MM_Q4_F32: return "Q4/F32";
    case DS4_GPU_TEST_Q4_QB_MM_Q4_F16: return "Q4/F16";
    case DS4_GPU_TEST_Q4_QB_MM_F16_F32: return "F16/F32";
    case DS4_GPU_TEST_Q4_QB_MM_F16_F16: return "F16/F16";
    case DS4_GPU_TEST_Q4_QB_MM_Q4_TRANSIENT_F16_F16:
        return "Q4 transient/F16/F16";
    default: return "invalid";
    }
}

static bool mm_arm_uses_f16_rhs(ds4_gpu_test_q4_qb_mm_arm arm) {
    switch (arm) {
    case DS4_GPU_TEST_Q4_QB_MM_Q4_F16:
    case DS4_GPU_TEST_Q4_QB_MM_F16_F16:
    case DS4_GPU_TEST_Q4_QB_MM_Q4_TRANSIENT_F16_F16:
        return true;
    default:
        return false;
    }
}

static bool mm_arm_is_transient(ds4_gpu_test_q4_qb_mm_arm arm) {
    return arm == DS4_GPU_TEST_Q4_QB_MM_Q4_TRANSIENT_F16_F16;
}

static bool mm_arm_is_prepacked_experiment(
        ds4_gpu_test_q4_qb_mm_arm arm) {
    return mm_arm_is_transient(arm);
}

static int run_mm_arm_projection(
        ds4_gpu_tensor             *out,
        ds4_gpu_tensor             *rhs_f16,
        const void                 *model,
        uint64_t                    model_bytes,
        const ds4_gpu_tensor       *x,
        uint32_t                    n_tok,
        ds4_gpu_test_q4_qb_mm_arm   arm,
        bool                        materialize_rhs) {
    return ds4_gpu_test_q4_attn_q_b_mm_variant_tensor(
        out, rhs_f16, model, model_bytes, 0u, IN_DIM, OUT_DIM,
        x, n_tok, arm, materialize_rhs);
}

static int run_mm_arm_with_tail(
        ds4_gpu_tensor             *out,
        ds4_gpu_tensor             *rhs_f16,
        const void                 *model,
        uint64_t                    model_bytes,
        const ds4_gpu_tensor       *x,
        uint32_t                    n_tok,
        ds4_gpu_test_q4_qb_mm_arm   arm,
        bool                        materialize_rhs) {
    if (!run_mm_arm_projection(
            out, rhs_f16, model, model_bytes, x, n_tok,
            arm, materialize_rhs)) {
        return 0;
    }
    return ds4_gpu_head_rms_norm_rope_tail_tensor(
        out, n_tok, N_HEAD, HEAD_DIM, N_ROT,
        17u, 0u, false,
        10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f, 1.0e-6f);
}

static void check_cache_storage_unchanged(
        const ds4_gpu_q4_attn_q_b_f16_cache_report *before,
        const ds4_gpu_q4_attn_q_b_f16_cache_report *after,
        const char *what) {
    if (after->entries != before->entries ||
        after->bytes != before->bytes ||
        after->lookups != before->lookups ||
        after->hits != before->hits ||
        after->misses != before->misses ||
        after->builds != before->builds ||
        after->build_failures != before->build_failures ||
        after->build_circuit_open != before->build_circuit_open) {
        fail(what);
    }
}

static double monotonic_ms(void) {
    struct timespec ts;
    CHECK(clock_gettime(CLOCK_MONOTONIC, &ts) == 0, "monotonic clock");
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1.0e6;
}

static int compare_double(const void *lhs, const void *rhs) {
    const double a = *(const double *)lhs;
    const double b = *(const double *)rhs;
    return (a > b) - (a < b);
}

static double timing_quantile(const double samples[TIMING_SAMPLES],
                              double q) {
    double sorted[TIMING_SAMPLES];
    memcpy(sorted, samples, sizeof(sorted));
    qsort(sorted, TIMING_SAMPLES, sizeof(double), compare_double);
    const double position = q * (double)(TIMING_SAMPLES - 1u);
    const uint32_t lower = (uint32_t)position;
    const uint32_t upper = lower + 1u < TIMING_SAMPLES ? lower + 1u : lower;
    const double fraction = position - (double)lower;
    return sorted[lower] + (sorted[upper] - sorted[lower]) * fraction;
}

static double timing_paired_geomean_speedup(
        const double baseline[TIMING_SAMPLES],
        const double candidate[TIMING_SAMPLES]) {
    double log_sum = 0.0;
    for (uint32_t i = 0; i < TIMING_SAMPLES; i++) {
        log_sum += log(baseline[i] / candidate[i]);
    }
    /* TIMING_SAMPLES is two complete Williams cycles.  The geometric mean
     * preserves their multiplicative position balancing. */
    return exp(log_sum / (double)TIMING_SAMPLES);
}

int main(void) {
    static const uint32_t token_cases[] = {32u, 33u, 64u};
    const uint64_t page = (uint64_t)getpagesize();
    const uint64_t row_bytes =
        (uint64_t)BLOCKS_PER_ROW * sizeof(block_q4_K);
    const uint64_t weight_bytes = (uint64_t)OUT_DIM * row_bytes;
    const uint64_t ssd_weight_offset =
        2u * weight_bytes + SSD_SOURCE_LEADING;
    const uint64_t model_bytes = align_up(
        ssd_weight_offset + weight_bytes, page);
    const uint64_t support_model_bytes = align_up(weight_bytes, page);
    const uint64_t f16_cache_bytes =
        (uint64_t)OUT_DIM * IN_DIM * sizeof(uint16_t);
    const uint64_t input_count = (uint64_t)MAX_TOKENS * IN_DIM;
    const uint64_t input_storage_count =
        GUARD_FLOATS + input_count + GUARD_FLOATS;
    const uint64_t max_output_count = (uint64_t)MAX_TOKENS * OUT_DIM;
    const uint64_t output_storage_count =
        GUARD_FLOATS + max_output_count + GUARD_FLOATS;
    const uint64_t q_half_storage_count =
        GUARD_HALFS + max_output_count + GUARD_HALFS;

    CHECK(Q4_K_TYPE == 12u, "Q4_K GGUF type must be 12");
    CHECK(row_bytes == 576u, "unexpected Q4_K row size");
    CHECK(weight_bytes == 18u * 1024u * 1024u,
          "unexpected production q_b Q4_K size");
    CHECK(ssd_weight_offset % page == SSD_SOURCE_LEADING,
          "SSD source offset must exercise a non-page-aligned exact view");
    CHECK(f16_cache_bytes == 64u * 1024u * 1024u,
          "unexpected production q_b F16 sidecar size");
    CHECK(ds4_gpu_test_q4_attn_q_b_f16_working_set_policy(
              0u, 0u, 0u) == 0,
          "unknown working set must reject");
    CHECK(ds4_gpu_test_q4_attn_q_b_f16_working_set_policy(
              800u, 600u, 100u) == 1,
          "working-set equality boundary");
    CHECK(ds4_gpu_test_q4_attn_q_b_f16_working_set_policy(
              800u, 600u, 101u) == 0,
          "working-set one-byte overflow");
    CHECK(ds4_gpu_test_q4_attn_q_b_f16_working_set_policy(
              800u, 701u, 0u) == 0,
          "allocated working set beyond safety limit");
    CHECK(ds4_gpu_test_q4_attn_q_b_f16_working_set_policy(
              UINT64_MAX, 0u, UINT64_MAX) == 0,
          "working-set overflow-sized request");

    CHECK(unsetenv(k_disable) == 0, "clear cache disable env");
    CHECK(unsetenv(k_enable_ssd_streaming) == 0,
          "clear SSD-streaming cache opt-in env");
    CHECK(unsetenv(k_disable_f16_rhs) == 0,
          "clear compact F16 RHS disable env");
    CHECK(unsetenv(k_disable_transient_f16) == 0,
          "clear transient F16 disable env");
    CHECK(unsetenv(k_require) == 0, "clear cache require env");
    CHECK(unsetenv(k_min_tokens) == 0, "clear cache minimum env");
    CHECK(unsetenv(k_transient_min_tokens) == 0,
          "clear transient F16 minimum env");
    CHECK(unsetenv(k_cache_mb) == 0, "clear cache budget env");
    CHECK(setenv(k_min_tokens, "32", 1) == 0,
          "set 32-token cache minimum");
    CHECK(setenv(k_require, "1", 1) == 0,
          "require Q4 attn_q_b F16 cache");

    CHECK(ds4_gpu_init() != 0, "Metal init");
    if (!ds4_gpu_device_is_pre_m5_apple_silicon()) {
        fprintf(stderr,
                "Metal Q4 attn_q_b F16 cache oracle SKIP: "
                "requires Apple M1--M4\n");
        ds4_gpu_cleanup();
        return 0;
    }

    void *model = NULL;
    CHECK(posix_memalign(&model, (size_t)page, (size_t)model_bytes) == 0,
          "page-aligned model allocation");
    memset(model, 0, (size_t)model_bytes);
    fill_q4_matrix(model);
    fill_q4_matrix((block_q4_K *)((uint8_t *)model + weight_bytes));
    ((block_q4_K *)((uint8_t *)model + weight_bytes))[0].d ^= 0x001fu;
    fill_q4_matrix((block_q4_K *)((uint8_t *)model + ssd_weight_offset));
    ((block_q4_K *)((uint8_t *)model + ssd_weight_offset))[0].d ^= 0x005bu;

    void *support_model = NULL;
    CHECK(posix_memalign(&support_model, (size_t)page,
                         (size_t)support_model_bytes) == 0,
          "page-aligned support model allocation");
    memset(support_model, 0, (size_t)support_model_bytes);
    fill_q4_matrix(support_model);
    ((block_q4_K *)support_model)[0].dmin ^= 0x003du;

    float *input_host = malloc(
        (size_t)input_storage_count * sizeof(float));
    float *input_readback = malloc(
        (size_t)input_storage_count * sizeof(float));
    float *reference_host = malloc(
        (size_t)output_storage_count * sizeof(float));
    float *candidate_host = malloc(
        (size_t)output_storage_count * sizeof(float));
    uint16_t *q_half_host = malloc(
        (size_t)q_half_storage_count * sizeof(uint16_t));
    CHECK(input_host && input_readback && reference_host &&
          candidate_host && q_half_host, "host tensor allocation");

    poison_f32(input_host, input_storage_count, 0x7fc50000u);
    fill_inputs(input_host + GUARD_FLOATS);
    poison_f16(q_half_host, q_half_storage_count);

    ds4_gpu_tensor *x_base = ds4_gpu_tensor_alloc(
        input_storage_count * sizeof(float));
    ds4_gpu_tensor *reference_base = ds4_gpu_tensor_alloc(
        output_storage_count * sizeof(float));
    ds4_gpu_tensor *candidate_base = ds4_gpu_tensor_alloc(
        output_storage_count * sizeof(float));
    ds4_gpu_tensor *q_half_base = ds4_gpu_tensor_alloc(
        q_half_storage_count * sizeof(uint16_t));
    CHECK(x_base && reference_base && candidate_base && q_half_base,
          "Metal tensor allocation");

    ds4_gpu_tensor *x = ds4_gpu_tensor_view(
        x_base, GUARD_FLOATS * sizeof(float),
        input_count * sizeof(float));
    CHECK(x != NULL, "input tensor view");
    CHECK(ds4_gpu_tensor_write(
              x_base, 0, input_host,
              input_storage_count * sizeof(float)) != 0,
          "input upload");
    CHECK(ds4_gpu_tensor_write(
              q_half_base, 0, q_half_host,
              q_half_storage_count * sizeof(uint16_t)) != 0,
          "q_half poison upload");

    /* Residency must be selected before installing the synthetic model. */
    ds4_gpu_set_quality(false);
    ds4_gpu_set_ssd_streaming(false);
    CHECK(ds4_gpu_set_model_map(model, model_bytes) != 0,
          "resident model map");
    CHECK(ds4_gpu_prepare_support_model(
              support_model, support_model_bytes, 0,
              support_model_bytes, 0) != 0,
          "resident support model map");
    ds4_gpu_test_q4_attn_q_b_f16_cache_reset();

    /* REQUIRE applies only at or above MIN_TOKENS.  A short tail must remain
     * a non-candidate rather than failing after a successful session prewarm. */
    {
        CHECK(setenv(k_min_tokens, "512", 1) == 0,
              "set short-tail cache minimum");
        ds4_gpu_tensor *short_out = ds4_gpu_tensor_view(
            candidate_base, GUARD_FLOATS * sizeof(float),
            (uint64_t)32u * OUT_DIM * sizeof(float));
        ds4_gpu_tensor *short_half = ds4_gpu_tensor_view(
            q_half_base, GUARD_HALFS * sizeof(uint16_t),
            (uint64_t)32u * OUT_DIM * sizeof(uint16_t));
        CHECK(short_out && short_half, "short-tail tensor views");
        CHECK(run_candidate(short_out, short_half, model, model_bytes,
                            x, 32u) == 0,
              "below-min REQUIRE batch must remain a non-candidate");
        ds4_gpu_tensor_free(short_half);
        ds4_gpu_tensor_free(short_out);

        ds4_gpu_q4_attn_q_b_f16_cache_report short_report;
        ds4_gpu_test_q4_attn_q_b_f16_cache_report(&short_report);
        CHECK(short_report.candidate_calls == 0u &&
                  short_report.lookups == 0u &&
                  short_report.fallbacks == 0u &&
                  short_report.rejects == 0u &&
                  short_report.build_circuit_open == 0u,
              "below-min REQUIRE candidate accounting");
        CHECK(setenv(k_min_tokens, "32", 1) == 0,
              "restore 32-token cache minimum");
    }

    /* A stable admission failure opens the build circuit.  Raising the
     * logical budget alone must not trigger repeated allocations; an
     * explicit cache reset is required before builds may resume. */
    {
        const uint64_t gate_output_count = (uint64_t)32u * OUT_DIM;
        const uint64_t gate_output_bytes =
            gate_output_count * sizeof(float);
        const uint64_t gate_half_bytes =
            gate_output_count * sizeof(uint16_t);
        CHECK(setenv(k_cache_mb, "63", 1) == 0,
              "set undersized sidecar budget");
        poison_f32(candidate_host, output_storage_count,
                   k_candidate_poison);
        poison_f16(q_half_host, q_half_storage_count);
        CHECK(ds4_gpu_tensor_write(
                  candidate_base, 0, candidate_host,
                  output_storage_count * sizeof(float)) != 0,
              "budget output poison upload");
        CHECK(ds4_gpu_tensor_write(
                  q_half_base, 0, q_half_host,
                  q_half_storage_count * sizeof(uint16_t)) != 0,
              "budget q_half poison upload");

        ds4_gpu_tensor *budget_out = ds4_gpu_tensor_view(
            candidate_base, GUARD_FLOATS * sizeof(float),
            gate_output_bytes);
        ds4_gpu_tensor *budget_half = ds4_gpu_tensor_view(
            q_half_base, GUARD_HALFS * sizeof(uint16_t),
            gate_half_bytes);
        CHECK(budget_out && budget_half, "budget tensor views");
        CHECK(run_candidate(budget_out, budget_half, model, model_bytes,
                            x, 32u) == -1,
              "undersized budget must fail required candidate");
        ds4_gpu_tensor_free(budget_half);
        ds4_gpu_tensor_free(budget_out);

        ds4_gpu_q4_attn_q_b_f16_cache_report budget_report;
        ds4_gpu_test_q4_attn_q_b_f16_cache_report(&budget_report);
        CHECK(budget_report.entries == 0u && budget_report.bytes == 0u &&
                  budget_report.builds == 0u &&
                  budget_report.build_failures == 0u &&
                  budget_report.rejects == 1u &&
                  budget_report.build_circuit_open == 1u,
              "undersized budget circuit accounting");

        CHECK(setenv(k_cache_mb, "3072", 1) == 0,
              "raise sidecar budget without reset");
        budget_out = ds4_gpu_tensor_view(
            candidate_base, GUARD_FLOATS * sizeof(float),
            gate_output_bytes);
        budget_half = ds4_gpu_tensor_view(
            q_half_base, GUARD_HALFS * sizeof(uint16_t),
            gate_half_bytes);
        CHECK(budget_out && budget_half,
              "open-circuit tensor views");
        CHECK(run_candidate(budget_out, budget_half, model, model_bytes,
                            x, 32u) == -1,
              "open circuit must suppress budget-only retry");
        ds4_gpu_tensor_free(budget_half);
        ds4_gpu_tensor_free(budget_out);
        ds4_gpu_test_q4_attn_q_b_f16_cache_report(&budget_report);
        CHECK(budget_report.entries == 0u && budget_report.builds == 0u &&
                  budget_report.build_failures == 0u &&
                  budget_report.rejects == 2u &&
                  budget_report.build_circuit_open == 1u,
              "open circuit retry accounting");

        CHECK(ds4_gpu_tensor_read(
                  candidate_base, 0, candidate_host,
                  output_storage_count * sizeof(float)) != 0,
              "budget output readback");
        CHECK(ds4_gpu_tensor_read(
                  q_half_base, 0, q_half_host,
                  q_half_storage_count * sizeof(uint16_t)) != 0,
              "budget q_half readback");
        CHECK(count_poison_f32_mismatches(
                  candidate_host, 0, output_storage_count,
                  k_candidate_poison) == 0,
              "budget rejection touched output");
        CHECK(count_poison_f16_mismatches(
                  q_half_host, q_half_storage_count) == 0,
              "budget rejection touched q_half");

        CHECK(ds4_gpu_release_q4_attn_q_b_f16_sidecars() != 0,
              "production lifecycle release after open circuit");
        ds4_gpu_test_q4_attn_q_b_f16_cache_report(&budget_report);
        CHECK(budget_report.entries == 0u && budget_report.bytes == 0u &&
                  budget_report.build_circuit_open == 0u,
              "lifecycle release did not close empty build circuit");
        ds4_gpu_test_q4_attn_q_b_f16_cache_reset();
        CHECK(unsetenv(k_cache_mb) == 0,
              "clear sidecar budget override");
    }

    /* Session creation uses the batch-prewarm API before prefill timing.
     * Exercise its transactional publication independently from lazy use. */
    {
        const ds4_gpu_q4_attn_q_b_f16_sidecar_desc descs[2] = {
            {
                .weight_offset = 0,
                .weight_bytes = weight_bytes,
                .in_dim = IN_DIM,
                .out_dim = OUT_DIM,
                .weight_type = Q4_K_TYPE,
                .layer = 0,
            },
            {
                .weight_offset = weight_bytes,
                .weight_bytes = weight_bytes,
                .in_dim = IN_DIM,
                .out_dim = OUT_DIM,
                .weight_type = Q4_K_TYPE,
                .layer = 1,
            },
        };
        const ds4_gpu_q4_attn_q_b_f16_sidecar_desc support_desc = {
            .weight_offset = 0,
            .weight_bytes = weight_bytes,
            .in_dim = IN_DIM,
            .out_dim = OUT_DIM,
            .weight_type = Q4_K_TYPE,
            .layer = 0,
        };
        uint64_t prepared_bytes = 0;
        CHECK(setenv(k_cache_mb, "127", 1) == 0,
              "set undersized transactional prewarm budget");
        CHECK(ds4_gpu_prepare_q4_attn_q_b_f16_sidecars(
                  model, model_bytes, descs, 2u, 64u, 0u,
                  &prepared_bytes) == -1,
              "transactional prewarm budget rejection");
        ds4_gpu_q4_attn_q_b_f16_cache_report prewarm_report;
        ds4_gpu_test_q4_attn_q_b_f16_cache_report(&prewarm_report);
        CHECK(prepared_bytes == 0u && prewarm_report.entries == 0u &&
                  prewarm_report.bytes == 0u &&
                  prewarm_report.builds == 0u &&
                  prewarm_report.build_circuit_open == 1u,
              "transactional prewarm published a partial cache");
        ds4_gpu_test_q4_attn_q_b_f16_cache_reset();
        CHECK(unsetenv(k_cache_mb) == 0,
              "clear transactional prewarm budget");

        CHECK(ds4_gpu_prepare_q4_attn_q_b_f16_sidecars(
                  model, model_bytes, descs, 2u, 64u, 0u,
                  &prepared_bytes) == 1,
              "transactional two-offset prewarm");
        ds4_gpu_test_q4_attn_q_b_f16_cache_report(&prewarm_report);
        CHECK(prepared_bytes == 2u * f16_cache_bytes &&
                  prewarm_report.entries == 2u &&
                  prewarm_report.bytes == 2u * f16_cache_bytes &&
                  prewarm_report.builds == 2u &&
                  prewarm_report.build_circuit_open == 0u,
              "two-offset prewarm publication accounting");

        prepared_bytes = 0;
        CHECK(ds4_gpu_prepare_q4_attn_q_b_f16_sidecars(
                  support_model, support_model_bytes, &support_desc,
                  1u, 64u, 0u, &prepared_bytes) == 1,
              "same-offset second-model prewarm");
        ds4_gpu_test_q4_attn_q_b_f16_cache_report(&prewarm_report);
        CHECK(prepared_bytes == f16_cache_bytes &&
                  prewarm_report.entries == 3u &&
                  prewarm_report.bytes == 3u * f16_cache_bytes &&
                  prewarm_report.builds == 3u &&
                  prewarm_report.build_circuit_open == 0u,
              "model-map cache-key isolation");
        const uint64_t generation_before_eviction =
            ds4_gpu_q4_attn_q_b_f16_cache_generation();
        CHECK(ds4_gpu_make_room_for_q4_attn_q_b_f16_session() != 0,
              "session admission cache eviction");
        CHECK(ds4_gpu_q4_attn_q_b_f16_cache_generation() !=
                  generation_before_eviction,
              "session admission did not advance cache generation");
        ds4_gpu_test_q4_attn_q_b_f16_cache_report(&prewarm_report);
        CHECK(prewarm_report.entries == 0u &&
                  prewarm_report.bytes == 0u &&
                  prewarm_report.build_circuit_open == 0u,
              "session admission retained resident sidecars");
        ds4_gpu_test_q4_attn_q_b_f16_cache_reset();
    }

    for (uint32_t case_i = 0;
         case_i < sizeof(token_cases) / sizeof(token_cases[0]);
         case_i++) {
        const uint32_t n_tok = token_cases[case_i];
        const uint64_t output_count = (uint64_t)n_tok * OUT_DIM;
        const uint64_t output_bytes = output_count * sizeof(float);
        const uint64_t q_half_count = (uint64_t)n_tok * OUT_DIM;
        const uint64_t q_half_bytes = q_half_count * sizeof(uint16_t);

        poison_f32(reference_host, output_storage_count,
                   k_reference_poison);
        poison_f32(candidate_host, output_storage_count,
                   k_candidate_poison);
        poison_f16(q_half_host, q_half_storage_count);
        CHECK(ds4_gpu_tensor_write(
                  reference_base, 0, reference_host,
                  output_storage_count * sizeof(float)) != 0,
              "reference poison upload");
        CHECK(ds4_gpu_tensor_write(
                  candidate_base, 0, candidate_host,
                  output_storage_count * sizeof(float)) != 0,
              "candidate poison upload");
        CHECK(ds4_gpu_tensor_write(
                  q_half_base, 0, q_half_host,
                  q_half_storage_count * sizeof(uint16_t)) != 0,
              "q_half poison refresh");

        ds4_gpu_tensor *reference = ds4_gpu_tensor_view(
            reference_base, GUARD_FLOATS * sizeof(float), output_bytes);
        ds4_gpu_tensor *candidate = ds4_gpu_tensor_view(
            candidate_base, GUARD_FLOATS * sizeof(float), output_bytes);
        ds4_gpu_tensor *q_half = ds4_gpu_tensor_view(
            q_half_base, GUARD_HALFS * sizeof(uint16_t), q_half_bytes);
        CHECK(reference && candidate && q_half, "case tensor views");

        CHECK(run_reference(reference, model, model_bytes, x, n_tok) == 1,
              "baseline Q4 projection plus head norm/RoPE");
        CHECK(run_candidate(
                  candidate, q_half, model, model_bytes, x, n_tok) == 1,
              "required cached-F16 candidate");

        ds4_gpu_tensor_free(q_half);
        ds4_gpu_tensor_free(candidate);
        ds4_gpu_tensor_free(reference);

        CHECK(ds4_gpu_tensor_read(
                  reference_base, 0, reference_host,
                  output_storage_count * sizeof(float)) != 0,
              "reference readback");
        CHECK(ds4_gpu_tensor_read(
                  candidate_base, 0, candidate_host,
                  output_storage_count * sizeof(float)) != 0,
              "candidate readback");
        CHECK(ds4_gpu_tensor_read(
                  q_half_base, 0, q_half_host,
                  q_half_storage_count * sizeof(uint16_t)) != 0,
              "q_half readback");

        uint64_t first = UINT64_MAX;
        const uint64_t bit_mismatches = count_bit_mismatches(
            reference_host + GUARD_FLOATS,
            candidate_host + GUARD_FLOATS,
            output_count, &first);
        const uint64_t reference_prefix = count_poison_f32_mismatches(
            reference_host, 0, GUARD_FLOATS, k_reference_poison);
        const uint64_t reference_suffix = count_poison_f32_mismatches(
            reference_host, GUARD_FLOATS + output_count,
            output_storage_count, k_reference_poison);
        const uint64_t candidate_prefix = count_poison_f32_mismatches(
            candidate_host, 0, GUARD_FLOATS, k_candidate_poison);
        const uint64_t candidate_suffix = count_poison_f32_mismatches(
            candidate_host, GUARD_FLOATS + output_count,
            output_storage_count, k_candidate_poison);
        const uint64_t q_half_rhs_count = (uint64_t)n_tok * IN_DIM;
        const uint64_t q_half_written =
            count_poison_f16_mismatches_range(
                q_half_host,
                GUARD_HALFS,
                GUARD_HALFS + q_half_rhs_count);
        const uint64_t q_half_prefix =
            count_poison_f16_mismatches_range(
                q_half_host, 0, GUARD_HALFS);
        const uint64_t q_half_tail =
            count_poison_f16_mismatches_range(
                q_half_host,
                GUARD_HALFS + q_half_rhs_count,
                q_half_storage_count);

        fprintf(stderr,
                "Metal Q4 attn_q_b F16 cache N=%u bitwise=%llu "
                "ref_guard=%llu/%llu candidate_guard=%llu/%llu "
                "q_half_written=%llu guard=%llu/%llu\n",
                n_tok,
                (unsigned long long)bit_mismatches,
                (unsigned long long)reference_prefix,
                (unsigned long long)reference_suffix,
                (unsigned long long)candidate_prefix,
                (unsigned long long)candidate_suffix,
                (unsigned long long)q_half_written,
                (unsigned long long)q_half_prefix,
                (unsigned long long)q_half_tail);
        if (first != UINT64_MAX) {
            fprintf(stderr,
                    "  first bitwise mismatch token=%llu row=%llu\n",
                    (unsigned long long)(first / OUT_DIM),
                    (unsigned long long)(first % OUT_DIM));
        }
        CHECK(bit_mismatches == 0, "candidate bitwise mismatch");
        CHECK(reference_prefix == 0 && reference_suffix == 0,
              "reference output canary");
        CHECK(candidate_prefix == 0 && candidate_suffix == 0,
              "candidate output canary");
        CHECK(q_half_written == q_half_rhs_count,
              "Q4 candidate did not materialize the complete F16 RHS");
        CHECK(q_half_prefix == 0 && q_half_tail == 0,
              "Q4 candidate F16 RHS scratch canary");

        ds4_gpu_q4_attn_q_b_f16_cache_report report;
        ds4_gpu_test_q4_attn_q_b_f16_cache_report(&report);
        CHECK(report.entries == 1u, "one cache entry");
        CHECK(report.bytes == f16_cache_bytes, "64 MiB cache entry");
        CHECK(report.lookups == (uint64_t)case_i + 1u,
              "cache lookup count");
        CHECK(report.hits == (uint64_t)case_i,
              "cache hit count");
        CHECK(report.misses == 1u, "single cold cache miss");
        CHECK(report.builds == 1u, "single cache build");
        CHECK(report.build_failures == 0u, "no cache build failure");
        CHECK(report.candidate_calls == (uint64_t)case_i + 1u,
              "candidate call count");
        CHECK(report.fallbacks == 0u && report.rejects == 0u,
              "no correctness-case fallback");
    }

    /* Compact RHS staging is subordinate to the resident sidecar.  Invalid
     * scratch must use F16 weights with the original F32 RHS, even when the
     * sidecar itself is required, rather than rejecting or replaying Q4. */
    {
        const uint32_t n_tok = 33u;
        const uint64_t output_count = (uint64_t)n_tok * OUT_DIM;
        const uint64_t output_bytes = output_count * sizeof(float);
        const uint64_t compact_rhs_bytes =
            (uint64_t)n_tok * IN_DIM * sizeof(uint16_t);

        poison_f32(reference_host, output_storage_count,
                   k_reference_poison);
        poison_f16(q_half_host, q_half_storage_count);
        CHECK(ds4_gpu_tensor_write(
                  reference_base, 0, reference_host,
                  output_storage_count * sizeof(float)) != 0,
              "nested fallback reference poison upload");
        CHECK(ds4_gpu_tensor_write(
                  q_half_base, 0, q_half_host,
                  q_half_storage_count * sizeof(uint16_t)) != 0,
              "nested fallback scratch poison upload");

        ds4_gpu_tensor *reference = ds4_gpu_tensor_view(
            reference_base, GUARD_FLOATS * sizeof(float), output_bytes);
        ds4_gpu_tensor *candidate = ds4_gpu_tensor_view(
            candidate_base, GUARD_FLOATS * sizeof(float), output_bytes);
        ds4_gpu_tensor *short_half = ds4_gpu_tensor_view(
            q_half_base, GUARD_HALFS * sizeof(uint16_t),
            compact_rhs_bytes - sizeof(uint16_t));
        CHECK(reference && candidate && short_half,
              "nested fallback tensor views");
        CHECK(run_reference(reference, model, model_bytes, x, n_tok) == 1,
              "nested fallback reference");
        CHECK(ds4_gpu_tensor_read(
                  reference_base, 0, reference_host,
                  output_storage_count * sizeof(float)) != 0,
              "nested fallback reference readback");

        ds4_gpu_tensor *scratch_cases[] = {x, short_half};
        const char *scratch_names[] = {"alias", "undersized"};
        for (uint32_t scratch_i = 0; scratch_i < 2u; scratch_i++) {
            poison_f32(candidate_host, output_storage_count,
                       k_candidate_poison);
            CHECK(ds4_gpu_tensor_write(
                      candidate_base, 0, candidate_host,
                      output_storage_count * sizeof(float)) != 0,
                  "nested fallback candidate poison upload");
            CHECK(run_candidate(
                      candidate, scratch_cases[scratch_i],
                      model, model_bytes, x, n_tok) == 1,
                  "nested F16/F32 fallback candidate");
            CHECK(ds4_gpu_tensor_read(
                      candidate_base, 0, candidate_host,
                      output_storage_count * sizeof(float)) != 0,
                  "nested fallback candidate readback");
            uint64_t first = UINT64_MAX;
            CHECK(count_bit_mismatches(
                      reference_host + GUARD_FLOATS,
                      candidate_host + GUARD_FLOATS,
                      output_count, &first) == 0,
                  "nested F16/F32 fallback bitwise mismatch");
            CHECK(count_poison_f32_mismatches(
                      candidate_host, 0, GUARD_FLOATS,
                      k_candidate_poison) == 0 &&
                  count_poison_f32_mismatches(
                      candidate_host, GUARD_FLOATS + output_count,
                      output_storage_count, k_candidate_poison) == 0,
                  "nested fallback output canary");
            fprintf(stderr,
                    "Metal Q4 attn_q_b compact RHS %s fallback: PASS\n",
                    scratch_names[scratch_i]);
        }
        CHECK(ds4_gpu_tensor_read(
                  q_half_base, 0, q_half_host,
                  q_half_storage_count * sizeof(uint16_t)) != 0,
              "nested fallback scratch readback");
        CHECK(count_poison_f16_mismatches(
                  q_half_host, q_half_storage_count) == 0,
              "undersized compact RHS fallback touched scratch");

        ds4_gpu_tensor_free(short_half);
        ds4_gpu_tensor_free(candidate);
        ds4_gpu_tensor_free(reference);
    }

    /* Compare the raw projection before RMSNorm/RoPE so normalization cannot
     * hide a uniform scale or dequantization error. */
    {
        const uint32_t n_tok = 33u;
        const uint64_t output_count = (uint64_t)n_tok * OUT_DIM;
        const uint64_t output_bytes = output_count * sizeof(float);
        poison_f32(reference_host, output_storage_count,
                   k_reference_poison);
        poison_f32(candidate_host, output_storage_count,
                   k_candidate_poison);
        CHECK(ds4_gpu_tensor_write(
                  reference_base, 0, reference_host,
                  output_storage_count * sizeof(float)) != 0,
              "raw reference poison upload");
        CHECK(ds4_gpu_tensor_write(
                  candidate_base, 0, candidate_host,
                  output_storage_count * sizeof(float)) != 0,
              "raw candidate poison upload");
        ds4_gpu_tensor *reference = ds4_gpu_tensor_view(
            reference_base, GUARD_FLOATS * sizeof(float), output_bytes);
        ds4_gpu_tensor *candidate = ds4_gpu_tensor_view(
            candidate_base, GUARD_FLOATS * sizeof(float), output_bytes);
        CHECK(reference && candidate, "raw projection tensor views");
        CHECK(ds4_gpu_matmul_quant_tensor(
                  reference, model, model_bytes, 0, Q4_K_TYPE,
                  IN_DIM, OUT_DIM, x, n_tok) == 1,
              "raw native Q4 projection");
        CHECK(ds4_gpu_test_q4_attn_q_b_f16_projection_tensor(
                  candidate, model, model_bytes, 0,
                  IN_DIM, OUT_DIM, x, n_tok) == 1,
              "raw cached-F16 projection");
        ds4_gpu_tensor_free(candidate);
        ds4_gpu_tensor_free(reference);
        CHECK(ds4_gpu_tensor_read(
                  reference_base, 0, reference_host,
                  output_storage_count * sizeof(float)) != 0,
              "raw reference readback");
        CHECK(ds4_gpu_tensor_read(
                  candidate_base, 0, candidate_host,
                  output_storage_count * sizeof(float)) != 0,
              "raw candidate readback");
        uint64_t first = UINT64_MAX;
        CHECK(count_bit_mismatches(
                  reference_host + GUARD_FLOATS,
                  candidate_host + GUARD_FLOATS,
                  output_count, &first) == 0,
              "raw projection bitwise mismatch");
        CHECK(count_poison_f32_mismatches(
                  reference_host, 0, GUARD_FLOATS,
                  k_reference_poison) == 0 &&
              count_poison_f32_mismatches(
                  reference_host, GUARD_FLOATS + output_count,
                  output_storage_count, k_reference_poison) == 0,
              "raw reference output canary");
        CHECK(count_poison_f32_mismatches(
                  candidate_host, 0, GUARD_FLOATS,
                  k_candidate_poison) == 0 &&
              count_poison_f32_mismatches(
                  candidate_host, GUARD_FLOATS + output_count,
                  output_storage_count, k_candidate_poison) == 0,
              "raw candidate output canary");
        fprintf(stderr,
                "Metal Q4 attn_q_b F16 cache raw projection N=33: PASS\n");

        /* Keep the boundary raw oracle in the default (non-timing) target as
         * well: the tail must not be able to mask a projection discrepancy. */
        static const ds4_gpu_test_q4_qb_mm_arm raw_arms[] = {
            DS4_GPU_TEST_Q4_QB_MM_Q4_F16,
            DS4_GPU_TEST_Q4_QB_MM_F16_F32,
            DS4_GPU_TEST_Q4_QB_MM_F16_F16,
        };
        ds4_gpu_tensor *raw_rhs_f16 = ds4_gpu_tensor_view(
            q_half_base, GUARD_HALFS * sizeof(uint16_t),
            (uint64_t)n_tok * IN_DIM * sizeof(uint16_t));
        CHECK(raw_rhs_f16 != NULL, "raw four-way F16 RHS view");
        const uint64_t raw_rhs_count = (uint64_t)n_tok * IN_DIM;
        for (uint32_t arm_i = 0;
             arm_i < sizeof(raw_arms) / sizeof(raw_arms[0]);
             arm_i++) {
            const bool rhs_is_f16 =
                mm_arm_uses_f16_rhs(raw_arms[arm_i]);
            const uint32_t rhs_modes = rhs_is_f16 ? 2u : 1u;
            for (uint32_t rhs_mode = 0; rhs_mode < rhs_modes;
                 rhs_mode++) {
                const bool materialize_rhs = rhs_mode == 0u;
                if (rhs_is_f16) {
                    poison_f16(q_half_host, q_half_storage_count);
                    CHECK(ds4_gpu_tensor_write(
                              q_half_base, 0, q_half_host,
                              q_half_storage_count * sizeof(uint16_t)) != 0,
                          "raw four-way RHS poison upload");
                    if (!materialize_rhs) {
                        CHECK(ds4_gpu_tensor_copy_f32_to_f16(
                                  raw_rhs_f16, 0u, x, 0u,
                                  raw_rhs_count) != 0,
                              "raw four-way explicit RHS prepack");
                    }
                }

                poison_f32(candidate_host, output_storage_count,
                           k_candidate_poison);
                CHECK(ds4_gpu_tensor_write(
                          candidate_base, 0, candidate_host,
                          output_storage_count * sizeof(float)) != 0,
                      "raw four-way candidate poison upload");
                candidate = ds4_gpu_tensor_view(
                    candidate_base, GUARD_FLOATS * sizeof(float),
                    output_bytes);
                CHECK(candidate != NULL, "raw four-way candidate view");
                CHECK(run_mm_arm_projection(
                          candidate, raw_rhs_f16,
                          model, model_bytes, x, n_tok,
                          raw_arms[arm_i], materialize_rhs) == 1,
                      "raw four-way projection");
                ds4_gpu_tensor_free(candidate);
                CHECK(ds4_gpu_tensor_read(
                          candidate_base, 0, candidate_host,
                          output_storage_count * sizeof(float)) != 0,
                      "raw four-way candidate readback");
                first = UINT64_MAX;
                CHECK(count_bit_mismatches(
                          reference_host + GUARD_FLOATS,
                          candidate_host + GUARD_FLOATS,
                          output_count, &first) == 0,
                      "raw four-way bitwise mismatch");
                CHECK(count_poison_f32_mismatches(
                          candidate_host, 0, GUARD_FLOATS,
                          k_candidate_poison) == 0 &&
                      count_poison_f32_mismatches(
                          candidate_host, GUARD_FLOATS + output_count,
                          output_storage_count, k_candidate_poison) == 0,
                      "raw four-way candidate canary");
                if (rhs_is_f16) {
                    CHECK(ds4_gpu_tensor_read(
                              q_half_base, 0, q_half_host,
                              q_half_storage_count * sizeof(uint16_t)) != 0,
                          "raw four-way RHS readback");
                    CHECK(count_poison_f16_mismatches_range(
                              q_half_host, GUARD_HALFS,
                              GUARD_HALFS + raw_rhs_count) ==
                              raw_rhs_count,
                          "raw four-way RHS payload was not materialized");
                    CHECK(count_poison_f16_mismatches_range(
                              q_half_host, 0, GUARD_HALFS) == 0 &&
                          count_poison_f16_mismatches_range(
                              q_half_host,
                              GUARD_HALFS + raw_rhs_count,
                              q_half_storage_count) == 0,
                          "raw four-way RHS canary");
                }
                fprintf(stderr,
                        "Metal Q4 attn_q_b raw N=33 %s %s: PASS\n",
                        mm_arm_name(raw_arms[arm_i]),
                        rhs_is_f16
                            ? (materialize_rhs
                                ? "with-pack" : "prepacked")
                            : "control");
            }
        }
        ds4_gpu_tensor_free(raw_rhs_f16);
    }

    /* Exercise the production command-batch lifecycle from a cold cache.
     * The sidecar build must complete before publication even though the
     * projection and norm/RoPE remain encoded in the caller's batch.  A
     * second batch then proves that the published entry is a real hot hit. */
    {
        const uint32_t n_tok = 64u;
        const uint64_t output_count = (uint64_t)n_tok * OUT_DIM;
        const uint64_t output_bytes = output_count * sizeof(float);
        const uint64_t q_half_bytes =
            output_count * sizeof(uint16_t);

        ds4_gpu_test_q4_attn_q_b_f16_cache_reset();
        poison_f32(reference_host, output_storage_count,
                   k_reference_poison);
        poison_f32(candidate_host, output_storage_count,
                   k_candidate_poison);
        CHECK(ds4_gpu_tensor_write(
                  reference_base, 0, reference_host,
                  output_storage_count * sizeof(float)) != 0,
              "batch reference poison upload");
        CHECK(ds4_gpu_tensor_write(
                  candidate_base, 0, candidate_host,
                  output_storage_count * sizeof(float)) != 0,
              "batch candidate poison upload");

        ds4_gpu_tensor *reference = ds4_gpu_tensor_view(
            reference_base, GUARD_FLOATS * sizeof(float), output_bytes);
        ds4_gpu_tensor *candidate = ds4_gpu_tensor_view(
            candidate_base, GUARD_FLOATS * sizeof(float), output_bytes);
        ds4_gpu_tensor *q_half = ds4_gpu_tensor_view(
            q_half_base, GUARD_HALFS * sizeof(uint16_t), q_half_bytes);
        CHECK(reference && candidate && q_half,
              "cold batch tensor views");
        CHECK(run_reference(reference, model, model_bytes, x, n_tok) == 1,
              "cold batch reference");
        CHECK(ds4_gpu_begin_commands() != 0,
              "begin cold candidate batch");
        CHECK(run_candidate(candidate, q_half, model, model_bytes,
                            x, n_tok) == 1,
              "encode cold batched candidate");
        CHECK(ds4_gpu_end_commands() != 0,
              "finish cold candidate batch");
        ds4_gpu_tensor_free(q_half);
        ds4_gpu_tensor_free(candidate);
        ds4_gpu_tensor_free(reference);

        CHECK(ds4_gpu_tensor_read(
                  reference_base, 0, reference_host,
                  output_storage_count * sizeof(float)) != 0,
              "cold batch reference readback");
        CHECK(ds4_gpu_tensor_read(
                  candidate_base, 0, candidate_host,
                  output_storage_count * sizeof(float)) != 0,
              "cold batch candidate readback");
        uint64_t first = UINT64_MAX;
        CHECK(count_bit_mismatches(
                  reference_host + GUARD_FLOATS,
                  candidate_host + GUARD_FLOATS,
                  output_count, &first) == 0,
              "cold batched candidate bitwise mismatch");

        ds4_gpu_q4_attn_q_b_f16_cache_report report;
        ds4_gpu_test_q4_attn_q_b_f16_cache_report(&report);
        CHECK(report.entries == 1u && report.builds == 1u &&
                  report.misses == 1u && report.hits == 0u,
              "cold batch cache publication");

        poison_f32(candidate_host, output_storage_count,
                   k_candidate_poison);
        CHECK(ds4_gpu_tensor_write(
                  candidate_base, 0, candidate_host,
                  output_storage_count * sizeof(float)) != 0,
              "hot batch candidate poison upload");
        candidate = ds4_gpu_tensor_view(
            candidate_base, GUARD_FLOATS * sizeof(float), output_bytes);
        q_half = ds4_gpu_tensor_view(
            q_half_base, GUARD_HALFS * sizeof(uint16_t), q_half_bytes);
        CHECK(candidate && q_half, "hot batch tensor views");
        CHECK(ds4_gpu_begin_commands() != 0,
              "begin hot candidate batch");
        CHECK(run_candidate(candidate, q_half, model, model_bytes,
                            x, n_tok) == 1,
              "encode hot batched candidate");
        CHECK(ds4_gpu_end_commands() != 0,
              "finish hot candidate batch");
        ds4_gpu_tensor_free(q_half);
        ds4_gpu_tensor_free(candidate);
        CHECK(ds4_gpu_tensor_read(
                  candidate_base, 0, candidate_host,
                  output_storage_count * sizeof(float)) != 0,
              "hot batch candidate readback");
        first = UINT64_MAX;
        CHECK(count_bit_mismatches(
                  reference_host + GUARD_FLOATS,
                  candidate_host + GUARD_FLOATS,
                  output_count, &first) == 0,
              "hot batched candidate bitwise mismatch");
        ds4_gpu_test_q4_attn_q_b_f16_cache_report(&report);
        CHECK(report.entries == 1u && report.builds == 1u &&
                  report.misses == 1u && report.hits == 1u &&
                  report.lookups == 2u,
              "hot batch cache hit");
        fprintf(stderr,
                "Metal Q4 attn_q_b F16 cache cold/hot command batch: PASS\n");
    }

    /* DISABLE wins over REQUIRE, must reject before cache lookup, and must
     * leave the entire output allocation untouched. */
    ds4_gpu_q4_attn_q_b_f16_cache_report gate_before;
    ds4_gpu_q4_attn_q_b_f16_cache_report gate_after;
    const ds4_gpu_q4_attn_q_b_f16_sidecar_desc ssd_desc = {
        .weight_offset = ssd_weight_offset,
        .weight_bytes = weight_bytes,
        .in_dim = IN_DIM,
        .out_dim = OUT_DIM,
        .weight_type = Q4_K_TYPE,
        .layer = 0,
    };
    uint64_t ssd_prepared_bytes = 0;
    ds4_gpu_test_q4_attn_q_b_f16_cache_report(&gate_before);
    poison_f32(candidate_host, output_storage_count, k_candidate_poison);
    CHECK(ds4_gpu_tensor_write(
              candidate_base, 0, candidate_host,
              output_storage_count * sizeof(float)) != 0,
          "DISABLE output poison");
    CHECK(setenv(k_disable, "1", 1) == 0, "set cache disable env");
    ds4_gpu_tensor *gate_out = ds4_gpu_tensor_view(
        candidate_base, GUARD_FLOATS * sizeof(float),
        (uint64_t)32u * OUT_DIM * sizeof(float));
    ds4_gpu_tensor *gate_half = ds4_gpu_tensor_view(
        q_half_base, GUARD_HALFS * sizeof(uint16_t),
        (uint64_t)32u * OUT_DIM * sizeof(uint16_t));
    CHECK(gate_out && gate_half, "DISABLE tensor views");
    CHECK(run_candidate(
              gate_out, gate_half, model, model_bytes, x, 32u) == -1,
          "DISABLE must win over REQUIRE");
    ds4_gpu_tensor_free(gate_half);
    ds4_gpu_tensor_free(gate_out);
    CHECK(ds4_gpu_tensor_read(
              candidate_base, 0, candidate_host,
              output_storage_count * sizeof(float)) != 0,
          "DISABLE output readback");
    CHECK(count_poison_f32_mismatches(
              candidate_host, 0, output_storage_count,
              k_candidate_poison) == 0,
          "DISABLE touched output");
    ds4_gpu_test_q4_attn_q_b_f16_cache_report(&gate_after);
    check_cache_storage_unchanged(
        &gate_before, &gate_after, "DISABLE touched cache storage/state");
    CHECK(gate_after.candidate_calls == gate_before.candidate_calls + 1u &&
          gate_after.fallbacks == gate_before.fallbacks + 1u &&
          gate_after.rejects == gate_before.rejects + 1u,
          "DISABLE accounting");
    CHECK(unsetenv(k_disable) == 0, "clear cache disable env");

    /* The persistent-sidecar SSD extension is opt-in: its default must
     * continue to drop resident-only sidecars and, under REQUIRE, reject the
     * persistent cache path without rebuilding it.  The independent transient
     * path is exercised below with REQUIRE cleared. */
    gate_before = gate_after;
    CHECK(gate_before.entries != 0u,
          "default SSD transition lacks a resident sidecar to release");
    const uint64_t default_ssd_generation =
        ds4_gpu_q4_attn_q_b_f16_cache_generation();
    const uint64_t ssd_output_count = (uint64_t)32u * OUT_DIM;
    poison_f32(reference_host, output_storage_count,
               k_reference_poison);
    CHECK(ds4_gpu_tensor_write(
              reference_base, 0, reference_host,
              output_storage_count * sizeof(float)) != 0,
          "SSD exact-view reference poison");
    ds4_gpu_tensor *ssd_reference = ds4_gpu_tensor_view(
        reference_base, GUARD_FLOATS * sizeof(float),
        ssd_output_count * sizeof(float));
    CHECK(ssd_reference != NULL, "SSD exact-view reference tensor view");
    CHECK(run_reference_at(
              ssd_reference, model, model_bytes, ssd_weight_offset,
              x, 32u) == 1,
          "SSD exact-view native Q4 reference");
    ds4_gpu_tensor_free(ssd_reference);
    CHECK(ds4_gpu_tensor_read(
              reference_base, 0, reference_host,
              output_storage_count * sizeof(float)) != 0,
          "SSD exact-view reference readback");
    poison_f32(candidate_host, output_storage_count, k_candidate_poison);
    CHECK(ds4_gpu_tensor_write(
              candidate_base, 0, candidate_host,
              output_storage_count * sizeof(float)) != 0,
          "SSD output poison");
    ds4_gpu_set_ssd_streaming(true);
    ds4_gpu_test_q4_attn_q_b_f16_cache_report(&gate_after);
    CHECK(gate_after.entries == 0u && gate_after.bytes == 0u,
          "enabling SSD mode retained resident sidecars");
    CHECK(ds4_gpu_q4_attn_q_b_f16_cache_generation() !=
              default_ssd_generation,
          "default SSD transition did not advance cache generation");
    CHECK(gate_after.build_circuit_open == 0u,
          "enabling SSD mode retained the build circuit state");
    CHECK(gate_after.candidate_calls == gate_before.candidate_calls &&
          gate_after.fallbacks == gate_before.fallbacks &&
          gate_after.rejects == gate_before.rejects,
          "SSD transition reset cache accounting");

    gate_before = gate_after;
    CHECK(ds4_gpu_prepare_q4_attn_q_b_f16_sidecars(
              model, model_bytes, &ssd_desc, 1u, 32u, 0u,
              &ssd_prepared_bytes) == -1,
          "REQUIRE alone must not enable SSD-streaming prewarm");
    CHECK(ssd_prepared_bytes == 0u,
          "default SSD prewarm reported prepared bytes");
    ds4_gpu_test_q4_attn_q_b_f16_cache_report(&gate_after);
    check_cache_storage_unchanged(
        &gate_before, &gate_after,
        "default SSD prewarm touched cache storage/state");
    gate_before = gate_after;
    gate_out = ds4_gpu_tensor_view(
        candidate_base, GUARD_FLOATS * sizeof(float),
        (uint64_t)32u * OUT_DIM * sizeof(float));
    gate_half = ds4_gpu_tensor_view(
        q_half_base, GUARD_HALFS * sizeof(uint16_t),
        (uint64_t)32u * OUT_DIM * sizeof(uint16_t));
    CHECK(gate_out && gate_half, "SSD tensor views");
    CHECK(run_candidate_at(
              gate_out, gate_half, model, model_bytes,
              ssd_weight_offset, x, 32u) == -1,
          "SSD mode must reject required resident cache");
    ds4_gpu_tensor_free(gate_half);
    ds4_gpu_tensor_free(gate_out);
    CHECK(ds4_gpu_tensor_read(
              candidate_base, 0, candidate_host,
              output_storage_count * sizeof(float)) != 0,
          "SSD output readback");
    CHECK(count_poison_f32_mismatches(
              candidate_host, 0, output_storage_count,
              k_candidate_poison) == 0,
          "SSD rejection touched output");
    ds4_gpu_test_q4_attn_q_b_f16_cache_report(&gate_after);
    CHECK(gate_after.entries == 0u && gate_after.bytes == 0u &&
          gate_after.build_circuit_open == 0u,
          "SSD rejection rebuilt resident cache");
    CHECK(gate_after.candidate_calls == gate_before.candidate_calls + 1u &&
          gate_after.fallbacks == gate_before.fallbacks + 1u &&
          gate_after.rejects == gate_before.rejects + 1u,
          "SSD rejection accounting");

    /* Explicit opt-in permits the production prewarm to build the sidecar and
     * the prefill dispatch to hit it while experts remain SSD-streamed.
     * REQUIRE is deliberately not the opt-in: it only makes failure strict. */
    CHECK(unsetenv(k_require) == 0,
          "clear REQUIRE before standalone SSD opt-in");
    CHECK(setenv(k_enable_ssd_streaming, "1", 1) == 0,
          "enable Q4 attn_q_b F16 cache with SSD streaming");
    gate_before = gate_after;
    ssd_prepared_bytes = 0;
    CHECK(ds4_gpu_prepare_q4_attn_q_b_f16_sidecars(
              model, model_bytes, &ssd_desc, 1u, 32u, 0u,
              &ssd_prepared_bytes) == 1,
          "SSD opt-in prewarm build");
    ds4_gpu_test_q4_attn_q_b_f16_cache_report(&gate_after);
    CHECK(ssd_prepared_bytes == f16_cache_bytes &&
              gate_after.entries == 1u &&
              gate_after.bytes == f16_cache_bytes &&
              gate_after.builds == gate_before.builds + 1u &&
              gate_after.misses == gate_before.misses &&
              gate_after.hits == gate_before.hits &&
              gate_after.lookups == gate_before.lookups &&
              gate_after.candidate_calls == gate_before.candidate_calls &&
              gate_after.fallbacks == gate_before.fallbacks &&
              gate_after.rejects == gate_before.rejects &&
              gate_after.build_circuit_open == 0u,
          "SSD opt-in prewarm build accounting");

    gate_before = gate_after;
    poison_f32(candidate_host, output_storage_count, k_candidate_poison);
    CHECK(ds4_gpu_tensor_write(
              candidate_base, 0, candidate_host,
              output_storage_count * sizeof(float)) != 0,
          "SSD opt-in first-hit output poison");
    gate_out = ds4_gpu_tensor_view(
        candidate_base, GUARD_FLOATS * sizeof(float),
        (uint64_t)32u * OUT_DIM * sizeof(float));
    gate_half = ds4_gpu_tensor_view(
        q_half_base, GUARD_HALFS * sizeof(uint16_t),
        (uint64_t)32u * OUT_DIM * sizeof(uint16_t));
    CHECK(gate_out && gate_half, "SSD opt-in first-hit tensor views");
    CHECK(run_candidate_at(
              gate_out, gate_half, model, model_bytes,
              ssd_weight_offset, x, 32u) == 1,
          "SSD opt-in first prefill cache hit");
    ds4_gpu_tensor_free(gate_half);
    ds4_gpu_tensor_free(gate_out);
    ds4_gpu_test_q4_attn_q_b_f16_cache_report(&gate_after);
    CHECK(gate_after.entries == gate_before.entries &&
              gate_after.bytes == gate_before.bytes &&
              gate_after.builds == gate_before.builds &&
              gate_after.misses == gate_before.misses &&
              gate_after.hits == gate_before.hits + 1u &&
              gate_after.lookups == gate_before.lookups + 1u &&
              gate_after.candidate_calls ==
                  gate_before.candidate_calls + 1u &&
              gate_after.fallbacks == gate_before.fallbacks &&
              gate_after.rejects == gate_before.rejects &&
              gate_after.build_circuit_open == 0u,
          "SSD opt-in first-hit accounting");
    CHECK(ds4_gpu_tensor_read(
              candidate_base, 0, candidate_host,
              output_storage_count * sizeof(float)) != 0,
          "SSD opt-in first-hit output readback");
    CHECK(count_poison_f32_mismatches(
              candidate_host, GUARD_FLOATS,
              GUARD_FLOATS + (uint64_t)32u * OUT_DIM,
              k_candidate_poison) != 0u,
          "SSD opt-in first hit did not write output");
    uint64_t ssd_first_mismatch = UINT64_MAX;
    CHECK(count_bit_mismatches(
              reference_host + GUARD_FLOATS,
              candidate_host + GUARD_FLOATS,
              ssd_output_count,
              &ssd_first_mismatch) == 0u,
          "SSD non-page-aligned exact-view candidate bitwise mismatch");

    gate_before = gate_after;
    gate_out = ds4_gpu_tensor_view(
        candidate_base, GUARD_FLOATS * sizeof(float),
        (uint64_t)32u * OUT_DIM * sizeof(float));
    gate_half = ds4_gpu_tensor_view(
        q_half_base, GUARD_HALFS * sizeof(uint16_t),
        (uint64_t)32u * OUT_DIM * sizeof(uint16_t));
    CHECK(gate_out && gate_half, "SSD opt-in hot tensor views");
    CHECK(run_candidate_at(
              gate_out, gate_half, model, model_bytes,
              ssd_weight_offset, x, 32u) == 1,
          "SSD opt-in cache hit");
    ds4_gpu_tensor_free(gate_half);
    ds4_gpu_tensor_free(gate_out);
    ds4_gpu_test_q4_attn_q_b_f16_cache_report(&gate_after);
    CHECK(gate_after.entries == gate_before.entries &&
              gate_after.bytes == gate_before.bytes &&
              gate_after.builds == gate_before.builds &&
              gate_after.misses == gate_before.misses &&
              gate_after.hits == gate_before.hits + 1u &&
              gate_after.lookups == gate_before.lookups + 1u &&
              gate_after.candidate_calls ==
                  gate_before.candidate_calls + 1u &&
              gate_after.fallbacks == gate_before.fallbacks &&
              gate_after.rejects == gate_before.rejects,
          "SSD opt-in hot-hit accounting");

    /* DISABLE remains the highest-priority policy even when SSD opt-in and
     * REQUIRE are both active, and it must not evict a ready sidecar. */
    CHECK(setenv(k_require, "1", 1) == 0,
          "restore REQUIRE for SSD DISABLE priority test");
    CHECK(setenv(k_disable, "1", 1) == 0,
          "disable SSD opt-in Q4 attn_q_b F16 cache");
    gate_before = gate_after;
    ssd_prepared_bytes = 0;
    CHECK(ds4_gpu_prepare_q4_attn_q_b_f16_sidecars(
              model, model_bytes, &ssd_desc, 1u, 32u, 0u,
              &ssd_prepared_bytes) == -1,
          "DISABLE must win over SSD opt-in prewarm and REQUIRE");
    CHECK(ssd_prepared_bytes == 0u,
          "disabled SSD opt-in prewarm reported prepared bytes");
    ds4_gpu_test_q4_attn_q_b_f16_cache_report(&gate_after);
    check_cache_storage_unchanged(
        &gate_before, &gate_after,
        "SSD opt-in prewarm DISABLE touched cache storage/state");
    gate_before = gate_after;
    gate_out = ds4_gpu_tensor_view(
        candidate_base, GUARD_FLOATS * sizeof(float),
        (uint64_t)32u * OUT_DIM * sizeof(float));
    gate_half = ds4_gpu_tensor_view(
        q_half_base, GUARD_HALFS * sizeof(uint16_t),
        (uint64_t)32u * OUT_DIM * sizeof(uint16_t));
    CHECK(gate_out && gate_half, "SSD opt-in DISABLE tensor views");
    CHECK(run_candidate_at(
              gate_out, gate_half, model, model_bytes,
              ssd_weight_offset, x, 32u) == -1,
          "DISABLE must win over SSD opt-in and REQUIRE");
    ds4_gpu_tensor_free(gate_half);
    ds4_gpu_tensor_free(gate_out);
    ds4_gpu_test_q4_attn_q_b_f16_cache_report(&gate_after);
    check_cache_storage_unchanged(
        &gate_before, &gate_after,
        "SSD opt-in DISABLE touched cache storage/state");
    CHECK(gate_after.candidate_calls == gate_before.candidate_calls + 1u &&
              gate_after.fallbacks == gate_before.fallbacks + 1u &&
              gate_after.rejects == gate_before.rejects + 1u,
          "SSD opt-in DISABLE accounting");
    CHECK(unsetenv(k_disable) == 0,
          "clear SSD opt-in cache disable env");

    /* Reasserting SSD mode with opt-in preserves the sidecar.  Explicit
     * lifecycle release remains authoritative and advances the generation. */
    const uint64_t opt_in_ssd_generation =
        ds4_gpu_q4_attn_q_b_f16_cache_generation();
    gate_before = gate_after;
    ds4_gpu_set_ssd_streaming(true);
    ds4_gpu_test_q4_attn_q_b_f16_cache_report(&gate_after);
    check_cache_storage_unchanged(
        &gate_before, &gate_after,
        "SSD opt-in transition evicted ready sidecar");
    CHECK(ds4_gpu_q4_attn_q_b_f16_cache_generation() ==
              opt_in_ssd_generation,
          "SSD opt-in transition advanced cache generation");

    CHECK(ds4_gpu_release_q4_attn_q_b_f16_sidecars() != 0,
          "SSD opt-in lifecycle release");
    ds4_gpu_test_q4_attn_q_b_f16_cache_report(&gate_after);
    CHECK(gate_after.entries == 0u && gate_after.bytes == 0u &&
              gate_after.build_circuit_open == 0u,
          "SSD opt-in lifecycle release retained cache state");
    CHECK(ds4_gpu_q4_attn_q_b_f16_cache_generation() !=
              opt_in_ssd_generation,
          "SSD opt-in lifecycle release did not advance generation");
    fprintf(stderr,
            "Metal Q4 attn_q_b F16 cache SSD opt-in policy/lifecycle: "
            "PASS\n");

    CHECK(unsetenv(k_enable_ssd_streaming) == 0,
          "clear SSD-streaming cache opt-in env");
    ds4_gpu_set_ssd_streaming(false);

    /* Exercise the public production selector, not only its benchmark hook.
     * Lower the threshold for this bounded oracle so aligned and boundary
     * geometries fit in the guarded allocations.  Resident and SSD-streamed
     * modes must use one transient scratch without publishing persistent
     * sidecars. */
    ds4_gpu_test_q4_attn_q_b_f16_cache_reset();
    CHECK(unsetenv(k_require) == 0,
          "clear REQUIRE for transient production oracle");
    const ds4_gpu_q4_attn_q_b_f16_sidecar_desc transient_desc = {
        .weight_offset = 0u,
        .weight_bytes = weight_bytes,
        .in_dim = IN_DIM,
        .out_dim = OUT_DIM,
        .weight_type = Q4_K_TYPE,
        .layer = 0u,
    };
    uint64_t transient_prepared_bytes = 0u;
    CHECK(ds4_gpu_prepare_q4_attn_q_b_f16_sidecars(
              model, model_bytes, &transient_desc, 1u, 4096u, 0u,
              &transient_prepared_bytes) == 1,
          "transient production preflight");
    CHECK(transient_prepared_bytes == f16_cache_bytes,
          "transient production preflight did not allocate one scratch");
    ds4_gpu_q4_attn_q_b_f16_cache_report transient_preflight_report;
    ds4_gpu_test_q4_attn_q_b_f16_cache_report(
        &transient_preflight_report);
    CHECK(transient_preflight_report.entries == 0u &&
              transient_preflight_report.bytes == 0u &&
              transient_preflight_report.lookups == 0u &&
              transient_preflight_report.builds == 0u,
          "transient production preflight published a sidecar");
    CHECK(setenv(k_transient_min_tokens, "32", 1) == 0,
          "lower transient production threshold for oracle");
    for (uint32_t case_i = 0;
         case_i < sizeof(token_cases) / sizeof(token_cases[0]);
         case_i++) {
        const uint32_t n_tok = token_cases[case_i];
        const uint64_t output_count = (uint64_t)n_tok * OUT_DIM;
        const uint64_t output_bytes = output_count * sizeof(float);
        const uint64_t q_half_count = (uint64_t)n_tok * IN_DIM;
        const uint64_t q_half_bytes = q_half_count * sizeof(uint16_t);

        poison_f32(reference_host, output_storage_count,
                   k_reference_poison);
        poison_f32(candidate_host, output_storage_count,
                   k_candidate_poison);
        poison_f16(q_half_host, q_half_storage_count);
        CHECK(ds4_gpu_tensor_write(
                  reference_base, 0, reference_host,
                  output_storage_count * sizeof(float)) != 0,
              "transient production reference poison upload");
        CHECK(ds4_gpu_tensor_write(
                  candidate_base, 0, candidate_host,
                  output_storage_count * sizeof(float)) != 0,
              "transient production candidate poison upload");
        CHECK(ds4_gpu_tensor_write(
                  q_half_base, 0, q_half_host,
                  q_half_storage_count * sizeof(uint16_t)) != 0,
              "transient production q_half poison upload");

        ds4_gpu_tensor *transient_reference = ds4_gpu_tensor_view(
            reference_base, GUARD_FLOATS * sizeof(float), output_bytes);
        ds4_gpu_tensor *transient_candidate = ds4_gpu_tensor_view(
            candidate_base, GUARD_FLOATS * sizeof(float), output_bytes);
        ds4_gpu_tensor *transient_half = ds4_gpu_tensor_view(
            q_half_base, GUARD_HALFS * sizeof(uint16_t), q_half_bytes);
        CHECK(transient_reference && transient_candidate && transient_half,
              "transient production tensor views");
        CHECK(run_reference(
                  transient_reference, model, model_bytes, x, n_tok) == 1,
              "transient production native Q4 reference");
        CHECK(run_candidate(
                  transient_candidate, transient_half,
                  model, model_bytes, x, n_tok) == 1,
              "transient production public entry point");
        ds4_gpu_tensor_free(transient_half);
        ds4_gpu_tensor_free(transient_candidate);
        ds4_gpu_tensor_free(transient_reference);

        CHECK(ds4_gpu_tensor_read(
                  reference_base, 0, reference_host,
                  output_storage_count * sizeof(float)) != 0,
              "transient production reference readback");
        CHECK(ds4_gpu_tensor_read(
                  candidate_base, 0, candidate_host,
                  output_storage_count * sizeof(float)) != 0,
              "transient production candidate readback");
        CHECK(ds4_gpu_tensor_read(
                  q_half_base, 0, q_half_host,
                  q_half_storage_count * sizeof(uint16_t)) != 0,
              "transient production q_half readback");
        uint64_t transient_first = UINT64_MAX;
        CHECK(count_bit_mismatches(
                  reference_host + GUARD_FLOATS,
                  candidate_host + GUARD_FLOATS,
                  output_count, &transient_first) == 0u,
              "transient production bitwise mismatch");
        CHECK(count_poison_f32_mismatches(
                  candidate_host, 0u, GUARD_FLOATS,
                  k_candidate_poison) == 0u &&
              count_poison_f32_mismatches(
                  candidate_host, GUARD_FLOATS + output_count,
                  output_storage_count, k_candidate_poison) == 0u,
              "transient production touched output guards");
        CHECK(count_poison_f16_mismatches_range(
                  q_half_host, 0u, GUARD_HALFS) == 0u &&
              count_poison_f16_mismatches_range(
                  q_half_host, GUARD_HALFS,
                  GUARD_HALFS + q_half_count) == q_half_count &&
              count_poison_f16_mismatches_range(
                  q_half_host, GUARD_HALFS + q_half_count,
                  q_half_storage_count) == 0u,
              "transient production q_half payload/canary mismatch");
    }

    /* Two different layer weights in one real command batch exercise the
     * production encoder boundaries and prove that stream-local scratch is
     * not overwritten before the preceding F16/F16 consumer and tail. */
    {
        const uint32_t n_tok = 33u;
        const uint64_t output_count = (uint64_t)n_tok * OUT_DIM;
        const uint64_t output_bytes = output_count * sizeof(float);
        const uint64_t q_half_count = (uint64_t)n_tok * IN_DIM;
        const uint64_t q_half_bytes =
            q_half_count * sizeof(uint16_t);
        poison_f32(reference_host, output_storage_count,
                   k_reference_poison);
        poison_f32(candidate_host, output_storage_count,
                   k_candidate_poison);
        poison_f16(q_half_host, q_half_storage_count);
        CHECK(ds4_gpu_tensor_write(
                  reference_base, 0, reference_host,
                  output_storage_count * sizeof(float)) != 0,
              "transient batch second-output poison upload");
        CHECK(ds4_gpu_tensor_write(
                  candidate_base, 0, candidate_host,
                  output_storage_count * sizeof(float)) != 0,
              "transient batch first-output poison upload");
        CHECK(ds4_gpu_tensor_write(
                  q_half_base, 0, q_half_host,
                  q_half_storage_count * sizeof(uint16_t)) != 0,
              "transient batch q_half poison upload");

        ds4_gpu_tensor *batch_first = ds4_gpu_tensor_view(
            candidate_base, GUARD_FLOATS * sizeof(float), output_bytes);
        ds4_gpu_tensor *batch_second = ds4_gpu_tensor_view(
            reference_base, GUARD_FLOATS * sizeof(float), output_bytes);
        ds4_gpu_tensor *batch_half = ds4_gpu_tensor_view(
            q_half_base, GUARD_HALFS * sizeof(uint16_t), q_half_bytes);
        CHECK(batch_first && batch_second && batch_half,
              "transient production batch tensor views");
        CHECK(ds4_gpu_begin_commands() != 0,
              "begin transient production command batch");
        CHECK(run_candidate_at(
                  batch_first, batch_half, model, model_bytes, 0u,
                  x, n_tok) == 1,
              "encode first transient production batch layer");
        CHECK(run_candidate_at(
                  batch_second, batch_half, model, model_bytes,
                  weight_bytes, x, n_tok) == 1,
              "encode second transient production batch layer");
        CHECK(ds4_gpu_end_commands() != 0,
              "finish transient production command batch");
        ds4_gpu_tensor_free(batch_half);
        ds4_gpu_tensor_free(batch_second);
        ds4_gpu_tensor_free(batch_first);

        CHECK(ds4_gpu_tensor_read(
                  candidate_base, 0, candidate_host,
                  output_storage_count * sizeof(float)) != 0,
              "transient batch first candidate readback");
        ds4_gpu_tensor *batch_reference = ds4_gpu_tensor_view(
            candidate_base, GUARD_FLOATS * sizeof(float), output_bytes);
        CHECK(batch_reference != NULL,
              "transient batch first reference view");
        CHECK(run_reference_at(
                  batch_reference, model, model_bytes, 0u,
                  x, n_tok) == 1,
              "transient batch first native reference");
        ds4_gpu_tensor_free(batch_reference);
        CHECK(ds4_gpu_tensor_read(
                  candidate_base, 0, reference_host,
                  output_storage_count * sizeof(float)) != 0,
              "transient batch first reference readback");
        uint64_t batch_first_mismatch = UINT64_MAX;
        CHECK(count_bit_mismatches(
                  reference_host + GUARD_FLOATS,
                  candidate_host + GUARD_FLOATS,
                  output_count, &batch_first_mismatch) == 0u,
              "transient batch first layer mismatch");

        CHECK(ds4_gpu_tensor_read(
                  reference_base, 0, candidate_host,
                  output_storage_count * sizeof(float)) != 0,
              "transient batch second candidate readback");
        batch_reference = ds4_gpu_tensor_view(
            candidate_base, GUARD_FLOATS * sizeof(float), output_bytes);
        CHECK(batch_reference != NULL,
              "transient batch second reference view");
        CHECK(run_reference_at(
                  batch_reference, model, model_bytes, weight_bytes,
                  x, n_tok) == 1,
              "transient batch second native reference");
        ds4_gpu_tensor_free(batch_reference);
        CHECK(ds4_gpu_tensor_read(
                  candidate_base, 0, reference_host,
                  output_storage_count * sizeof(float)) != 0,
              "transient batch second reference readback");
        uint64_t batch_second_mismatch = UINT64_MAX;
        CHECK(count_bit_mismatches(
                  reference_host + GUARD_FLOATS,
                  candidate_host + GUARD_FLOATS,
                  output_count, &batch_second_mismatch) == 0u,
              "transient batch second layer mismatch");

        CHECK(ds4_gpu_tensor_read(
                  q_half_base, 0, q_half_host,
                  q_half_storage_count * sizeof(uint16_t)) != 0,
              "transient batch q_half readback");
        CHECK(count_poison_f16_mismatches_range(
                  q_half_host, 0u, GUARD_HALFS) == 0u &&
              count_poison_f16_mismatches_range(
                  q_half_host, GUARD_HALFS,
                  GUARD_HALFS + q_half_count) == q_half_count &&
              count_poison_f16_mismatches_range(
                  q_half_host, GUARD_HALFS + q_half_count,
                  q_half_storage_count) == 0u,
              "transient batch q_half payload/canary mismatch");
        fprintf(stderr,
                "Metal Q4 attn_q_b transient F16 command-batch "
                "two-layer scratch reuse N=33: PASS\n");
    }

    ds4_gpu_q4_attn_q_b_f16_cache_report transient_report;
    ds4_gpu_test_q4_attn_q_b_f16_cache_report(&transient_report);
    CHECK(transient_report.entries == 0u &&
              transient_report.bytes == 0u &&
              transient_report.lookups == 0u &&
              transient_report.builds == 0u,
          "transient production published a persistent sidecar");

    /* Install a second model identity whose registered view covers only a
     * disjoint prefix.  Its byte-identical Q4 matrix starts at a deliberately
     * non-page-aligned, uncovered offset, so this succeeds only when the SSD
     * path creates an exact owned source view.  Running inside a real command
     * batch also exercises the completion-handler lifetime under
     * DS4_METAL_UNRETAINED_COMMAND_BUFFERS. */
    const uint64_t ssd_transient_weight_offset = page + SSD_SOURCE_LEADING;
    const uint64_t ssd_transient_model_bytes = align_up(
        ssd_transient_weight_offset + weight_bytes, page);
    void *ssd_transient_model = NULL;
    CHECK(posix_memalign(&ssd_transient_model, (size_t)page,
                         (size_t)ssd_transient_model_bytes) == 0,
          "SSD transient exact-view model allocation");
    memset(ssd_transient_model, 0, (size_t)ssd_transient_model_bytes);
    fill_q4_matrix((block_q4_K *)((uint8_t *)ssd_transient_model +
                                  ssd_transient_weight_offset));

    const uint32_t ssd_transient_tokens = 64u;
    const uint64_t ssd_transient_output_count =
        (uint64_t)ssd_transient_tokens * OUT_DIM;
    const uint64_t ssd_transient_output_bytes =
        ssd_transient_output_count * sizeof(float);
    const uint64_t ssd_transient_half_count =
        (uint64_t)ssd_transient_tokens * IN_DIM;
    const uint64_t ssd_transient_half_bytes =
        ssd_transient_half_count * sizeof(uint16_t);
    poison_f32(reference_host, output_storage_count, k_reference_poison);
    poison_f32(candidate_host, output_storage_count, k_candidate_poison);
    poison_f16(q_half_host, q_half_storage_count);
    CHECK(ds4_gpu_tensor_write(
              reference_base, 0, reference_host,
              output_storage_count * sizeof(float)) != 0,
          "transient SSD reference poison upload");
    CHECK(ds4_gpu_tensor_write(
              candidate_base, 0, candidate_host,
              output_storage_count * sizeof(float)) != 0,
          "transient SSD candidate poison upload");
    CHECK(ds4_gpu_tensor_write(
              q_half_base, 0, q_half_host,
              q_half_storage_count * sizeof(uint16_t)) != 0,
          "transient SSD q_half poison upload");
    ds4_gpu_tensor *transient_ssd_reference = ds4_gpu_tensor_view(
        reference_base, GUARD_FLOATS * sizeof(float),
        ssd_transient_output_bytes);
    ds4_gpu_tensor *transient_ssd_out = ds4_gpu_tensor_view(
        candidate_base, GUARD_FLOATS * sizeof(float),
        ssd_transient_output_bytes);
    ds4_gpu_tensor *transient_ssd_half = ds4_gpu_tensor_view(
        q_half_base, GUARD_HALFS * sizeof(uint16_t),
        ssd_transient_half_bytes);
    CHECK(transient_ssd_reference && transient_ssd_out && transient_ssd_half,
          "transient SSD exact-view tensor views");
    CHECK(run_reference(
              transient_ssd_reference, model, model_bytes, x,
              ssd_transient_tokens) == 1,
          "transient SSD native Q4 reference");
    ds4_gpu_tensor_free(transient_ssd_reference);
    CHECK(ds4_gpu_tensor_read(
              reference_base, 0, reference_host,
              output_storage_count * sizeof(float)) != 0,
          "transient SSD reference readback");

    /* Stream 0 already owns the resident oracle's 64 MiB scratch.  Move the
     * SSD transient case to a fresh stream so its successful preflight also
     * covers cold scratch + exact-source admission and allocation. */
    ds4_gpu_set_stream(1);
    ds4_gpu_set_ssd_streaming(true);
    CHECK(ds4_gpu_set_model_map_range(
              ssd_transient_model, ssd_transient_model_bytes,
              0u, page, page) != 0,
          "install disjoint SSD transient model prefix");
    const ds4_gpu_q4_attn_q_b_f16_sidecar_desc ssd_transient_desc = {
        .weight_offset = ssd_transient_weight_offset,
        .weight_bytes = weight_bytes,
        .in_dim = IN_DIM,
        .out_dim = OUT_DIM,
        .weight_type = Q4_K_TYPE,
        .layer = 0u,
    };
    ds4_gpu_q4_attn_q_b_f16_cache_report ssd_lifetime_before;
    ds4_gpu_q4_attn_q_b_f16_cache_report ssd_lifetime_mid;
    ds4_gpu_q4_attn_q_b_f16_cache_report ssd_lifetime_after;
    ds4_gpu_stream_test_stats ssd_transients_before;
    ds4_gpu_stream_test_stats ssd_transients_mid;
    ds4_gpu_stream_test_stats ssd_transients_after;
    ds4_gpu_test_q4_attn_q_b_f16_cache_report(&ssd_lifetime_before);

    /* A rejected session reserve must latch the transient selector off.  The
     * public call then returns the native-Q4 fallback sentinel without
     * touching either output, even though its smaller runtime-only gate would
     * otherwise fit.  A later successful preflight re-arms the same slot. */
    uint64_t ssd_transient_prepared_bytes = UINT64_MAX;
    CHECK(ds4_gpu_prepare_q4_attn_q_b_f16_sidecars(
              ssd_transient_model, ssd_transient_model_bytes,
              &ssd_transient_desc, 1u, 4096u, UINT64_MAX,
              &ssd_transient_prepared_bytes) == 0,
          "transient SSD oversized-reserve preflight rejection");
    CHECK(ssd_transient_prepared_bytes == 0u,
          "transient SSD rejected preflight allocated scratch");
    CHECK(ds4_gpu_begin_commands() != 0,
          "begin transient SSD rejected command batch");
    CHECK(run_candidate_at(
              transient_ssd_out, transient_ssd_half,
              ssd_transient_model, ssd_transient_model_bytes,
              ssd_transient_weight_offset, x,
              ssd_transient_tokens) == 0,
          "transient SSD rejected preflight escaped admission latch");
    CHECK(ds4_gpu_end_commands() != 0,
          "finish transient SSD rejected command batch");
    CHECK(ds4_gpu_tensor_read(
              candidate_base, 0, candidate_host,
              output_storage_count * sizeof(float)) != 0,
          "transient SSD rejected output readback");
    CHECK(ds4_gpu_tensor_read(
              q_half_base, 0, q_half_host,
              q_half_storage_count * sizeof(uint16_t)) != 0,
          "transient SSD rejected q_half readback");
    CHECK(count_poison_f32_mismatches(
              candidate_host, 0u, output_storage_count,
              k_candidate_poison) == 0u,
          "transient SSD rejected preflight touched output");
    CHECK(count_poison_f16_mismatches(
              q_half_host, q_half_storage_count) == 0u,
          "transient SSD rejected preflight touched q_half");

    ssd_transient_prepared_bytes = UINT64_MAX;
    CHECK(ds4_gpu_prepare_q4_attn_q_b_f16_sidecars(
              ssd_transient_model, ssd_transient_model_bytes,
              &ssd_transient_desc, 1u, 4096u, 0u,
              &ssd_transient_prepared_bytes) == 1,
          "transient SSD successful preflight re-arm");
    CHECK(ssd_transient_prepared_bytes == f16_cache_bytes,
          "transient SSD cold preflight did not allocate one scratch");
    ds4_gpu_test_stream_stats(&ssd_transients_before);
    CHECK(ds4_gpu_begin_commands() != 0,
          "begin transient SSD exact-view command batch");
    CHECK(run_candidate_at(
              transient_ssd_out, transient_ssd_half,
              ssd_transient_model, ssd_transient_model_bytes,
              ssd_transient_weight_offset, x,
              ssd_transient_tokens) == 1,
          "encode transient SSD exact-view candidate");
    ds4_gpu_test_q4_attn_q_b_f16_cache_report(&ssd_lifetime_mid);
    ds4_gpu_test_stream_stats(&ssd_transients_mid);
    CHECK(ssd_lifetime_mid.transient_exact_views_created ==
              ssd_lifetime_before.transient_exact_views_created + 1u &&
          ssd_lifetime_mid.transient_exact_views_live ==
              ssd_lifetime_before.transient_exact_views_live + 1u &&
          ssd_lifetime_mid.model_exact_cache_entries ==
              ssd_lifetime_before.model_exact_cache_entries &&
          ssd_lifetime_mid.model_exact_cache_bytes ==
              ssd_lifetime_before.model_exact_cache_bytes &&
          ssd_transients_mid.transient_references ==
              ssd_transients_before.transient_references,
          "transient SSD exact source was not command-buffer-owned");
    CHECK(ds4_gpu_end_commands() != 0,
          "finish transient SSD exact-view command batch");
    ds4_gpu_test_q4_attn_q_b_f16_cache_report(&ssd_lifetime_after);
    ds4_gpu_test_stream_stats(&ssd_transients_after);
    CHECK(ssd_lifetime_after.transient_exact_views_created ==
              ssd_lifetime_before.transient_exact_views_created + 1u &&
          ssd_lifetime_after.transient_exact_views_live ==
              ssd_lifetime_before.transient_exact_views_live &&
          ssd_lifetime_after.model_exact_cache_entries ==
              ssd_lifetime_before.model_exact_cache_entries &&
          ssd_lifetime_after.model_exact_cache_bytes ==
              ssd_lifetime_before.model_exact_cache_bytes &&
          ssd_transients_after.transient_references ==
              ssd_transients_before.transient_references,
          "transient SSD exact source leaked past command completion");
    ds4_gpu_tensor_free(transient_ssd_half);
    ds4_gpu_tensor_free(transient_ssd_out);
    CHECK(ds4_gpu_tensor_read(
              candidate_base, 0, candidate_host,
              output_storage_count * sizeof(float)) != 0,
          "transient SSD candidate readback");
    uint64_t transient_ssd_first = UINT64_MAX;
    CHECK(count_bit_mismatches(
              reference_host + GUARD_FLOATS,
              candidate_host + GUARD_FLOATS,
              ssd_transient_output_count,
              &transient_ssd_first) == 0u,
          "transient SSD exact-view bitwise mismatch");
    CHECK(count_poison_f32_mismatches(
              candidate_host, 0u, GUARD_FLOATS,
              k_candidate_poison) == 0u &&
          count_poison_f32_mismatches(
              candidate_host,
              GUARD_FLOATS + ssd_transient_output_count,
              output_storage_count,
              k_candidate_poison) == 0u,
          "transient SSD exact-view touched output guards");
    CHECK(ds4_gpu_tensor_read(
              q_half_base, 0, q_half_host,
              q_half_storage_count * sizeof(uint16_t)) != 0,
          "transient SSD q_half readback");
    CHECK(count_poison_f16_mismatches_range(
              q_half_host, 0u, GUARD_HALFS) == 0u &&
          count_poison_f16_mismatches_range(
              q_half_host, GUARD_HALFS,
              GUARD_HALFS + ssd_transient_half_count) ==
                  ssd_transient_half_count &&
          count_poison_f16_mismatches_range(
              q_half_host,
              GUARD_HALFS + ssd_transient_half_count,
              q_half_storage_count) == 0u,
          "transient SSD q_half payload/canary mismatch");
    ds4_gpu_test_q4_attn_q_b_f16_cache_report(&transient_report);
    CHECK(transient_report.entries == 0u &&
              transient_report.bytes == 0u &&
              transient_report.lookups == 0u &&
              transient_report.builds == 0u,
          "transient SSD exact-view published a persistent sidecar");
    ds4_gpu_set_ssd_streaming(false);
    ds4_gpu_set_stream(0);
    CHECK(setenv(k_require, "1", 1) == 0,
          "restore REQUIRE after transient production oracle");
    CHECK(unsetenv(k_transient_min_tokens) == 0,
          "restore transient production threshold");
    fprintf(stderr,
            "Metal Q4 attn_q_b transient F16 production selector "
            "N=32/33/64 and SSD exact-view: PASS\n");

    /* The input, including both guards, is immutable across every path. */
    CHECK(ds4_gpu_tensor_read(
              x_base, 0, input_readback,
              input_storage_count * sizeof(float)) != 0,
          "input readback");
    CHECK(memcmp(input_host, input_readback,
                 (size_t)input_storage_count * sizeof(float)) == 0,
          "input payload/canary modified");

    if (getenv(k_timing) != NULL) {
        const char *timing_tokens_env = getenv(
            "DS4_TEST_METAL_Q4_QB_F16_CACHE_TIMING_TOKENS");
        char *timing_tokens_end = NULL;
        errno = 0;
        const unsigned long timing_tokens_value = timing_tokens_env
            ? strtoul(timing_tokens_env, &timing_tokens_end, 10)
            : 4096ul;
        CHECK(!timing_tokens_env ||
                  (timing_tokens_env[0] >= '0' &&
                   timing_tokens_env[0] <= '9' &&
                   errno == 0 &&
                   timing_tokens_end != timing_tokens_env &&
                   *timing_tokens_end == '\0'),
              "invalid timing token count");
        CHECK(timing_tokens_value >= 32ul &&
                  timing_tokens_value <= 4096ul,
              "timing tokens must be in [32, 4096]");
        const uint32_t timing_tokens = (uint32_t)timing_tokens_value;
        const uint64_t timing_input_count =
            (uint64_t)timing_tokens * IN_DIM;
        const uint64_t timing_output_count =
            (uint64_t)timing_tokens * OUT_DIM;
        static const ds4_gpu_test_q4_qb_mm_arm base_arms[] = {
            DS4_GPU_TEST_Q4_QB_MM_Q4_F32,
            DS4_GPU_TEST_Q4_QB_MM_Q4_F16,
            DS4_GPU_TEST_Q4_QB_MM_F16_F32,
            DS4_GPU_TEST_Q4_QB_MM_F16_F16,
        };
        static const ds4_gpu_test_q4_qb_mm_arm transient_panel_arms[] = {
            DS4_GPU_TEST_Q4_QB_MM_Q4_F16,
            DS4_GPU_TEST_Q4_QB_MM_Q4_TRANSIENT_F16_F16,
            DS4_GPU_TEST_Q4_QB_MM_F16_F16,
        };
        static const ds4_gpu_test_q4_qb_mm_arm oracle_arms[] = {
            DS4_GPU_TEST_Q4_QB_MM_Q4_F32,
            DS4_GPU_TEST_Q4_QB_MM_Q4_F16,
            DS4_GPU_TEST_Q4_QB_MM_F16_F32,
            DS4_GPU_TEST_Q4_QB_MM_F16_F16,
            DS4_GPU_TEST_Q4_QB_MM_Q4_TRANSIENT_F16_F16,
        };
        static const uint8_t williams_order[4][4] = {
            {0u, 1u, 3u, 2u},
            {1u, 2u, 0u, 3u},
            {2u, 3u, 1u, 0u},
            {3u, 0u, 2u, 1u},
        };
        static const uint8_t transient_order[2][3] = {
            {0u, 1u, 2u},
            {2u, 1u, 0u},
        };
        double with_pack_ms[DS4_GPU_TEST_Q4_QB_MM_ARM_COUNT]
                           [TIMING_SAMPLES];
        double prepacked_ms[DS4_GPU_TEST_Q4_QB_MM_ARM_COUNT]
                           [TIMING_SAMPLES];
        double transient_panel_ms[3][TIMING_SAMPLES];
        double production_reference_ms[TIMING_SAMPLES];
        double production_transient_ms[TIMING_SAMPLES];
        float *timing_input = malloc(
            (size_t)timing_input_count * sizeof(float));
        CHECK(timing_input != NULL, "timing input host allocation");
        for (uint32_t token = 0; token < timing_tokens; token++) {
            for (uint32_t col = 0; col < IN_DIM; col++) {
                const uint32_t key = token * 131u + col * 17u +
                                     ((col >> 3u) ^ (token * 29u));
                timing_input[(uint64_t)token * IN_DIM + col] =
                    (float)((int)(key % 129u) - 64) / 509.0f;
            }
        }

        /* Timing allocations are independent from the guarded correctness
         * tensors.  A single output is sufficient because each projection
         * overwrites it completely before the in-place norm/RoPE stage. */
        ds4_gpu_tensor *timing_x = ds4_gpu_tensor_alloc(
            timing_input_count * sizeof(float));
        ds4_gpu_tensor *timing_out = ds4_gpu_tensor_alloc(
            timing_output_count * sizeof(float));
        ds4_gpu_tensor *timing_rhs_f16 = ds4_gpu_tensor_alloc(
            timing_input_count * sizeof(uint16_t));
        CHECK(timing_x && timing_out && timing_rhs_f16,
              "timing Metal tensor allocation");
        CHECK(ds4_gpu_tensor_write(
                  timing_x, 0, timing_input,
                  timing_input_count * sizeof(float)) != 0,
              "timing input upload");
        free(timing_input);

        ds4_gpu_test_q4_attn_q_b_f16_cache_reset();
        CHECK(ds4_gpu_synchronize() != 0, "pre-cold synchronize");
        const double cold_t0 = monotonic_ms();
        CHECK(run_candidate(timing_out, timing_rhs_f16, model, model_bytes,
                            timing_x, timing_tokens) == 1,
              "timing cold candidate");
        const double cold_ms = monotonic_ms() - cold_t0;
        CHECK(ds4_gpu_synchronize() != 0, "post-cold synchronize");

        CHECK(ds4_gpu_tensor_copy_f32_to_f16(
                  timing_rhs_f16, 0u, timing_x, 0u,
                  timing_input_count) != 0,
              "timing prepacked F16 RHS");
        for (uint32_t arm_i = 0;
             arm_i < sizeof(base_arms) / sizeof(base_arms[0]);
             arm_i++) {
            CHECK(run_mm_arm_projection(
                      timing_out, timing_rhs_f16, model, model_bytes,
                      timing_x, timing_tokens, base_arms[arm_i], true) == 1,
                  "four-way with-pack warmup");
            CHECK(run_mm_arm_projection(
                      timing_out, timing_rhs_f16, model, model_bytes,
                      timing_x, timing_tokens, base_arms[arm_i], false) == 1,
                  "four-way prepacked warmup");
        }

        for (uint32_t sample = 0; sample < TIMING_SAMPLES; sample++) {
            const uint8_t *order = williams_order[sample & 3u];
            for (uint32_t pos = 0; pos < 4u; pos++) {
                const uint32_t arm_i = order[pos];
                const ds4_gpu_test_q4_qb_mm_arm arm = base_arms[arm_i];
                if ((sample & 1u) == 0u) {
                    double t0 = monotonic_ms();
                    CHECK(run_mm_arm_projection(
                              timing_out, timing_rhs_f16,
                              model, model_bytes, timing_x, timing_tokens,
                              arm, true) == 1,
                          "four-way with-pack timing");
                    with_pack_ms[arm_i][sample] = monotonic_ms() - t0;
                    t0 = monotonic_ms();
                    CHECK(run_mm_arm_projection(
                              timing_out, timing_rhs_f16,
                              model, model_bytes, timing_x, timing_tokens,
                              arm, false) == 1,
                          "four-way prepacked timing");
                    prepacked_ms[arm_i][sample] = monotonic_ms() - t0;
                } else {
                    double t0 = monotonic_ms();
                    CHECK(run_mm_arm_projection(
                              timing_out, timing_rhs_f16,
                              model, model_bytes, timing_x, timing_tokens,
                              arm, false) == 1,
                          "four-way prepacked timing");
                    prepacked_ms[arm_i][sample] = monotonic_ms() - t0;
                    t0 = monotonic_ms();
                    CHECK(run_mm_arm_projection(
                              timing_out, timing_rhs_f16,
                              model, model_bytes, timing_x, timing_tokens,
                              arm, true) == 1,
                          "four-way with-pack timing");
                    with_pack_ms[arm_i][sample] = monotonic_ms() - t0;
                }
            }
        }
        fprintf(stderr,
                "Metal Q4 attn_q_b four-way resident timing N=%u "
                "cold-sidecar=%.3f ms\n",
                timing_tokens, cold_ms);
        for (uint32_t arm_i = 0;
             arm_i < sizeof(base_arms) / sizeof(base_arms[0]);
             arm_i++) {
            const bool rhs_is_f16 =
                mm_arm_uses_f16_rhs(base_arms[arm_i]);
            const double with_pack =
                timing_quantile(with_pack_ms[arm_i], 0.5);
            const double with_pack_p25 =
                timing_quantile(with_pack_ms[arm_i], 0.25);
            const double with_pack_p75 =
                timing_quantile(with_pack_ms[arm_i], 0.75);
            const double prepacked =
                timing_quantile(prepacked_ms[arm_i], 0.5);
            const double prepacked_p25 =
                timing_quantile(prepacked_ms[arm_i], 0.25);
            const double prepacked_p75 =
                timing_quantile(prepacked_ms[arm_i], 0.75);
            const double with_pack_speedup =
                timing_paired_geomean_speedup(
                with_pack_ms[DS4_GPU_TEST_Q4_QB_MM_Q4_F32],
                with_pack_ms[arm_i]);
            const double prepacked_speedup =
                timing_paired_geomean_speedup(
                prepacked_ms[DS4_GPU_TEST_Q4_QB_MM_Q4_F32],
                prepacked_ms[arm_i]);
            fprintf(stderr,
                    "  %-8s %s=%.3f ms [%.3f, %.3f] %.3fx paired-gmean "
                    "%s=%.3f ms [%.3f, %.3f] %.3fx paired-gmean\n",
                    mm_arm_name(base_arms[arm_i]),
                    rhs_is_f16 ? "with-pack" : "control-a",
                    with_pack, with_pack_p25, with_pack_p75,
                    with_pack_speedup,
                    rhs_is_f16 ? "prepacked" : "control-b",
                    prepacked, prepacked_p25, prepacked_p75,
                    prepacked_speedup);
        }

        /* This arm deliberately rebuilds a transient F16 weight matrix for
         * every projection, then dispatches the same F16/F16 multiply as the
         * resident sidecar control.  Keep it at the production-prefill
         * N=4096 geometry and time only with the already-packed RHS.  The two
         * controls swap first/last position every sample while the transient
         * arm remains between them, yielding eight directly paired samples. */
        if (timing_tokens != 4096u) {
            fprintf(stderr,
                    "Metal Q4 attn_q_b transient prepacked timing N=%u: "
                    "SKIP (dedicated geometry is N=4096)\n",
                    timing_tokens);
        } else if (!ds4_gpu_test_q4_attn_q_b_mm_arm_supported(
                       DS4_GPU_TEST_Q4_QB_MM_Q4_TRANSIENT_F16_F16)) {
            fprintf(stderr,
                    "Metal Q4 attn_q_b transient prepacked timing N=4096: "
                    "SKIP (pipeline unsupported)\n");
        } else {
            for (uint32_t arm_i = 0;
                 arm_i < sizeof(transient_panel_arms) /
                             sizeof(transient_panel_arms[0]);
                 arm_i++) {
                CHECK(run_mm_arm_projection(
                          timing_out, timing_rhs_f16, model, model_bytes,
                          timing_x, timing_tokens,
                          transient_panel_arms[arm_i], false) == 1,
                      "transient panel prepacked warmup");
            }

            for (uint32_t sample = 0; sample < TIMING_SAMPLES; sample++) {
                const uint8_t *order = transient_order[sample & 1u];
                for (uint32_t pos = 0; pos < 3u; pos++) {
                    const uint32_t arm_i = order[pos];
                    const double t0 = monotonic_ms();
                    CHECK(run_mm_arm_projection(
                              timing_out, timing_rhs_f16,
                              model, model_bytes, timing_x, timing_tokens,
                              transient_panel_arms[arm_i], false) == 1,
                          "transient panel prepacked timing");
                    transient_panel_ms[arm_i][sample] =
                        monotonic_ms() - t0;
                }
            }

            const double legacy_median =
                timing_quantile(transient_panel_ms[0], 0.5);
            const double legacy_p25 =
                timing_quantile(transient_panel_ms[0], 0.25);
            const double legacy_p75 =
                timing_quantile(transient_panel_ms[0], 0.75);
            const double transient_median =
                timing_quantile(transient_panel_ms[1], 0.5);
            const double transient_p25 =
                timing_quantile(transient_panel_ms[1], 0.25);
            const double transient_p75 =
                timing_quantile(transient_panel_ms[1], 0.75);
            const double sidecar_median =
                timing_quantile(transient_panel_ms[2], 0.5);
            const double sidecar_p25 =
                timing_quantile(transient_panel_ms[2], 0.25);
            const double sidecar_p75 =
                timing_quantile(transient_panel_ms[2], 0.75);
            const double sidecar_vs_legacy =
                timing_paired_geomean_speedup(
                    transient_panel_ms[0], transient_panel_ms[2]);
            const double transient_vs_legacy =
                timing_paired_geomean_speedup(
                    transient_panel_ms[0], transient_panel_ms[1]);
            const double transient_vs_sidecar =
                timing_paired_geomean_speedup(
                    transient_panel_ms[2], transient_panel_ms[1]);
            fprintf(stderr,
                    "Metal Q4 attn_q_b transient prepacked timing N=4096\n"
                    "  %-22s %.3f ms [%.3f, %.3f] control\n"
                    "  %-22s %.3f ms [%.3f, %.3f] "
                    "%.3fx vs legacy paired-gmean\n"
                    "  %-22s %.3f ms [%.3f, %.3f] "
                    "%.3fx vs legacy, %.3fx vs sidecar paired-gmean\n",
                    mm_arm_name(transient_panel_arms[0]),
                    legacy_median, legacy_p25, legacy_p75,
                    mm_arm_name(transient_panel_arms[2]),
                    sidecar_median, sidecar_p25, sidecar_p75,
                    sidecar_vs_legacy,
                    mm_arm_name(transient_panel_arms[1]),
                    transient_median, transient_p25, transient_p75,
                    transient_vs_legacy, transient_vs_sidecar);
        }

        /* Verify the exact timing geometry after sampling so readback and the
         * second output allocation cannot perturb the measured resident path.
         * Chunked reads bound host memory even for N=4096. */
        ds4_gpu_tensor *timing_reference = ds4_gpu_tensor_alloc(
            timing_output_count * sizeof(float));
        CHECK(timing_reference != NULL,
              "timing verification output allocation");
        CHECK(run_mm_arm_projection(
                  timing_reference, timing_rhs_f16, model, model_bytes,
                  timing_x, timing_tokens,
                  DS4_GPU_TEST_Q4_QB_MM_Q4_F32, true) == 1,
              "timing raw verification reference");

        const uint64_t verify_bytes =
            timing_output_count * sizeof(float);
        const uint64_t verify_chunk_bytes = 4u * 1024u * 1024u;
        float *verify_reference = malloc((size_t)verify_chunk_bytes);
        float *verify_candidate = malloc((size_t)verify_chunk_bytes);
        CHECK(verify_reference && verify_candidate,
              "timing verification host chunks");
        for (uint32_t pass = 0; pass < 2u; pass++) {
            if (pass == 1u) {
                CHECK(ds4_gpu_head_rms_norm_rope_tail_tensor(
                          timing_reference, timing_tokens,
                          N_HEAD, HEAD_DIM, N_ROT,
                          17u, 0u, false,
                          10000.0f, 1.0f, 0.0f, 1.0f,
                          32.0f, 1.0f, 1.0e-6f) != 0,
                      "timing tail verification reference");
            }
            for (uint32_t arm_i = 1;
                 arm_i < sizeof(oracle_arms) / sizeof(oracle_arms[0]);
                 arm_i++) {
                const ds4_gpu_test_q4_qb_mm_arm arm =
                    oracle_arms[arm_i];
                if (mm_arm_is_transient(arm) &&
                    timing_tokens != 4096u) {
                    continue;
                }
                const bool prepacked_experiment =
                    mm_arm_is_prepacked_experiment(arm);
                if (prepacked_experiment &&
                    !ds4_gpu_test_q4_attn_q_b_mm_arm_supported(arm)) {
                    continue;
                }
                const bool rhs_is_f16 = mm_arm_uses_f16_rhs(arm);
                /* The transient arm uses only the prepacked RHS; the compact
                 * copy oracle already covers the same producer above. */
                const uint32_t rhs_modes =
                    prepacked_experiment ? 1u : (rhs_is_f16 ? 2u : 1u);
                for (uint32_t rhs_mode = 0; rhs_mode < rhs_modes;
                     rhs_mode++) {
                    const bool materialize_rhs =
                        prepacked_experiment ? false : rhs_mode == 0u;
                    const int ok = pass == 0u
                        ? run_mm_arm_projection(
                            timing_out, timing_rhs_f16,
                            model, model_bytes, timing_x, timing_tokens,
                            arm, materialize_rhs)
                        : run_mm_arm_with_tail(
                            timing_out, timing_rhs_f16,
                            model, model_bytes, timing_x, timing_tokens,
                            arm, materialize_rhs);
                    CHECK(ok == 1, "timing projection verification arm");

                    uint64_t verify_mismatches = 0;
                    uint64_t verify_first = UINT64_MAX;
                    for (uint64_t offset = 0; offset < verify_bytes;
                         offset += verify_chunk_bytes) {
                        const uint64_t chunk_bytes =
                            verify_bytes - offset < verify_chunk_bytes
                                ? verify_bytes - offset
                                : verify_chunk_bytes;
                        CHECK(ds4_gpu_tensor_read(
                                  timing_reference, offset,
                                  verify_reference, chunk_bytes) != 0,
                              "timing verification reference readback");
                        CHECK(ds4_gpu_tensor_read(
                                  timing_out, offset,
                                  verify_candidate, chunk_bytes) != 0,
                              "timing verification candidate readback");
                        uint64_t chunk_first = UINT64_MAX;
                        verify_mismatches += count_bit_mismatches(
                            verify_reference, verify_candidate,
                            chunk_bytes / sizeof(float), &chunk_first);
                        if (verify_first == UINT64_MAX &&
                            chunk_first != UINT64_MAX) {
                            verify_first =
                                offset / sizeof(float) + chunk_first;
                        }
                    }
                    fprintf(stderr,
                            "Metal Q4 attn_q_b projection oracle N=%u "
                            "%s %s %s bitwise=%llu\n",
                            timing_tokens,
                            pass == 0u ? "raw" : "tail",
                            mm_arm_name(arm),
                            rhs_is_f16
                                ? (materialize_rhs
                                    ? "with-pack" : "prepacked")
                                : "control",
                            (unsigned long long)verify_mismatches);
                    if (verify_first != UINT64_MAX) {
                        fprintf(stderr,
                                "  first mismatch token=%llu row=%llu\n",
                                (unsigned long long)(verify_first / OUT_DIM),
                                (unsigned long long)(verify_first % OUT_DIM));
                    }
                    CHECK(verify_mismatches == 0,
                          "timing projection bitwise mismatch");
                }
            }
        }

        /* Time the complete public production selector after the kernel-only
         * panels: F32->F16 RHS copy, Q4->F16 transient expansion, F16/F16 MM,
         * and head norm/RoPE are all included.  The control is the native
         * Q4/F32 projection with the identical tail.  Alternate first place
         * to keep command-order and thermal bias paired. */
        if (timing_tokens == 4096u) {
            CHECK(ds4_gpu_release_q4_attn_q_b_f16_sidecars() != 0,
                  "release sidecar before production timing");
            ds4_gpu_test_q4_attn_q_b_f16_cache_reset();
            CHECK(unsetenv(k_require) == 0,
                  "clear REQUIRE for production timing");
            CHECK(unsetenv(k_transient_min_tokens) == 0,
                  "use default production transient threshold");

            CHECK(run_reference(
                      timing_reference, model, model_bytes,
                      timing_x, timing_tokens) == 1,
                  "production timing native warmup");
            CHECK(run_candidate(
                      timing_out, timing_rhs_f16, model, model_bytes,
                      timing_x, timing_tokens) == 1,
                  "production timing transient warmup");

            for (uint32_t sample = 0; sample < TIMING_SAMPLES; sample++) {
                for (uint32_t pos = 0; pos < 2u; pos++) {
                    const bool run_transient =
                        ((sample & 1u) == 0u) ? pos == 1u : pos == 0u;
                    const double t0 = monotonic_ms();
                    const int ok = run_transient
                        ? run_candidate(
                            timing_out, timing_rhs_f16,
                            model, model_bytes, timing_x, timing_tokens)
                        : run_reference(
                            timing_reference, model, model_bytes,
                            timing_x, timing_tokens);
                    CHECK(ok == 1, "production timing dispatch");
                    const double elapsed = monotonic_ms() - t0;
                    if (run_transient) {
                        production_transient_ms[sample] = elapsed;
                    } else {
                        production_reference_ms[sample] = elapsed;
                    }
                }
            }

            const double production_reference_median =
                timing_quantile(production_reference_ms, 0.5);
            const double production_reference_p25 =
                timing_quantile(production_reference_ms, 0.25);
            const double production_reference_p75 =
                timing_quantile(production_reference_ms, 0.75);
            const double production_transient_median =
                timing_quantile(production_transient_ms, 0.5);
            const double production_transient_p25 =
                timing_quantile(production_transient_ms, 0.25);
            const double production_transient_p75 =
                timing_quantile(production_transient_ms, 0.75);
            const double production_speedup =
                timing_paired_geomean_speedup(
                    production_reference_ms, production_transient_ms);
            fprintf(stderr,
                    "Metal Q4 attn_q_b public production timing N=4096\n"
                    "  native Q4/F32 + tail  %.3f ms [%.3f, %.3f] control\n"
                    "  transient full + tail %.3f ms [%.3f, %.3f] "
                    "%.3fx paired-gmean\n",
                    production_reference_median,
                    production_reference_p25,
                    production_reference_p75,
                    production_transient_median,
                    production_transient_p25,
                    production_transient_p75,
                    production_speedup);

            /* Re-run both arms immediately before readback, then prove that
             * the complete public entry point is exact at production N. */
            CHECK(run_reference(
                      timing_reference, model, model_bytes,
                      timing_x, timing_tokens) == 1,
                  "production timing final native reference");
            CHECK(run_candidate(
                      timing_out, timing_rhs_f16, model, model_bytes,
                      timing_x, timing_tokens) == 1,
                  "production timing final transient candidate");
            uint64_t production_mismatches = 0u;
            for (uint64_t offset = 0; offset < verify_bytes;
                 offset += verify_chunk_bytes) {
                const uint64_t chunk_bytes =
                    verify_bytes - offset < verify_chunk_bytes
                        ? verify_bytes - offset
                        : verify_chunk_bytes;
                CHECK(ds4_gpu_tensor_read(
                          timing_reference, offset,
                          verify_reference, chunk_bytes) != 0,
                      "production timing reference readback");
                CHECK(ds4_gpu_tensor_read(
                          timing_out, offset,
                          verify_candidate, chunk_bytes) != 0,
                      "production timing transient readback");
                uint64_t production_chunk_first = UINT64_MAX;
                production_mismatches += count_bit_mismatches(
                    verify_reference, verify_candidate,
                    chunk_bytes / sizeof(float),
                    &production_chunk_first);
            }
            CHECK(production_mismatches == 0u,
                  "production timing public entry point mismatch");
            ds4_gpu_q4_attn_q_b_f16_cache_report production_report;
            ds4_gpu_test_q4_attn_q_b_f16_cache_report(
                &production_report);
            CHECK(production_report.entries == 0u &&
                      production_report.bytes == 0u &&
                      production_report.lookups == 0u &&
                      production_report.builds == 0u,
                  "production timing retained a sidecar");
            CHECK(setenv(k_require, "1", 1) == 0,
                  "restore REQUIRE after production timing");
        }
        free(verify_candidate);
        free(verify_reference);
        ds4_gpu_tensor_free(timing_reference);

        ds4_gpu_tensor_free(timing_rhs_f16);
        ds4_gpu_tensor_free(timing_out);
        ds4_gpu_tensor_free(timing_x);
    }

    ds4_gpu_tensor_free(x);
    ds4_gpu_tensor_free(q_half_base);
    ds4_gpu_tensor_free(candidate_base);
    ds4_gpu_tensor_free(reference_base);
    ds4_gpu_tensor_free(x_base);
    ds4_gpu_cleanup();

    free(q_half_host);
    free(candidate_host);
    free(reference_host);
    free(input_readback);
    free(input_host);
    free(support_model);
    free(ssd_transient_model);
    free(model);

    CHECK(unsetenv(k_require) == 0, "clear cache require env at exit");
    CHECK(unsetenv(k_min_tokens) == 0,
          "clear cache minimum env at exit");
    CHECK(unsetenv(k_disable_transient_f16) == 0,
          "clear transient F16 disable env at exit");
    CHECK(unsetenv(k_transient_min_tokens) == 0,
          "clear transient F16 minimum env at exit");
    fprintf(stderr,
            "Metal Q4 attn_q_b F16 cache production geometry "
            "1024x32768 N=32/33/64: PASS\n");
    return 0;
}

#else

int main(void) {
    fprintf(stderr,
            "Metal Q4 attn_q_b F16 cache oracle SKIP: requires macOS\n");
    return 0;
}

#endif
