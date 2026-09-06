#define _DARWIN_C_SOURCE

#include "ds4_gpu.h"

#include <fcntl.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#ifdef __APPLE__

#define IQ2_XXS_TYPE 16u
#define Q2_K_TYPE 10u
#define QK_K 256u
#define IN_DIM 256u
#define MID_DIM 256u
#define OUT_DIM 64u
#define N_TOTAL_EXPERT 8u
#define N_EXPERT 6u
#define MAX_TOKENS 33u
#define N_TOTAL_EXPERT_256 256u
#define HIGH_EXPERT_ID 255u
#define CLAMP 4.0f
#define SENTINEL 1234567.0f
#define TAIL_CULL_ENV \
    "DS4_METAL_ENABLE_IQ2_XXS_SSD_PREFILL_MM_ADDR_TAIL_CULL"
#define TAIL_CULL_DISABLE_ENV \
    "DS4_METAL_DISABLE_IQ2_XXS_SSD_PREFILL_MM_ADDR_TAIL_CULL"

/* Production Flash routed-expert geometry.  Eight physical experts keep the
 * standalone fixture bounded while retaining the production top-6 routing,
 * 32-token grouped tile, and exact quantized row strides. */
#define FULL_IN_DIM 4096u
#define FULL_MID_DIM 2048u
#define FULL_OUT_DIM 4096u
#define FULL_TOKENS 32u
#define FULL_GUARD_WORDS 64u
#define FULL_GUARD_BITS 0x51a7c3e9u

typedef struct {
    uint16_t d;
    uint16_t qs[QK_K / 8u];
} block_iq2_xxs;

typedef struct {
    uint8_t scales[QK_K / 16u];
    uint8_t qs[QK_K / 4u];
    uint16_t d;
    uint16_t dmin;
} block_q2_K;

typedef struct {
    uint32_t tokens;
    uint64_t pair_count;
    uint64_t out_count;
    float *gate;
    float *up;
    float *mid;
    float *out;
    uint64_t guard_mismatches;
} run_result;

typedef struct {
    uint64_t pair_count;
    uint64_t out_count;
    float *gate;
    float *up;
    float *mid;
    float *out;
    uint64_t guard_mismatches;
} full_run_result;

typedef struct {
    uint64_t candidate_calls;
    uint64_t calls;
    uint64_t tokens;
    uint64_t rows;
    uint64_t require_failures;
    uint32_t min_tokens;
    uint32_t max_tokens;
} mm_stats_snapshot;

static uint32_t *full_guarded_payload(uint64_t payload_words);
static int full_check_guard(const char *run_name,
                            const char *tensor_name,
                            const ds4_gpu_tensor *tensor,
                            uint64_t payload_words,
                            uint64_t *mismatches);

/* Test-only counter reader implemented by the Metal backend. */
int ds4_gpu_test_iq2_stream_addr_mm_stats(
    uint64_t *candidate_calls, uint64_t *calls, uint64_t *tokens,
    uint64_t *rows, uint64_t *require_failures, uint32_t *min_tokens,
    uint32_t *max_tokens);
int ds4_gpu_test_iq2_stream_addr_mm_policy(
    int enable, int require, int disable, int material_ready,
    int *requested_out, int *required_out);

bool ds4_log_is_tty(FILE *fp) {
    (void)fp;
    return false;
}

static uint64_t align_up(uint64_t value, uint64_t alignment) {
    return (value + alignment - 1u) / alignment * alignment;
}

static int check_grouped_mm_policy(void) {
    int ok = 1;
#define CHECK_POLICY(label, valid_expected, request_expected,                \
                     require_expected, enable, require, disable, ready) do { \
    int requested = -1;                                                      \
    int required = -1;                                                       \
    const int valid = ds4_gpu_test_iq2_stream_addr_mm_policy(                \
        (enable), (require), (disable), (ready), &requested, &required);     \
    const int case_ok = valid == (valid_expected) &&                         \
        requested == (request_expected) && required == (require_expected);  \
    fprintf(stderr,                                                          \
            "IQ2 grouped-MM policy %-28s %s valid=%d request=%d "           \
            "require=%d\n",                                                \
            (label), case_ok ? "PASS" : "FAIL", valid, requested, required);\
    ok = case_ok && ok;                                                      \
} while (0)

    /* Unset defaults: automatic selection, with fail-closed coverage only
     * after the complete selected-address domain is materially ready. */
    CHECK_POLICY("default-ready",       1, 1, 1, -1, -1, -1, 1);
    CHECK_POLICY("default-small-cache", 1, 1, 0, -1, -1, -1, 0);

    /* Value-aware rollback and explicit fallback controls. */
    CHECK_POLICY("enable-zero",         1, 0, 0,  0, -1, -1, 1);
    CHECK_POLICY("enable-one",          1, 1, 1,  1, -1, -1, 1);
    CHECK_POLICY("require-zero",        1, 1, 0, -1,  0, -1, 1);
    CHECK_POLICY("require-one-not-ready", 1, 1, 1, -1, 1, -1, 0);
    CHECK_POLICY("require-over-enable-zero", 1, 1, 1, 0, 1, -1, 0);
    CHECK_POLICY("disable-one",         1, 0, 0, -1, -1,  1, 1);
    CHECK_POLICY("disable-zero",        1, 1, 1, -1, -1,  0, 1);
    /* The explicit REQUIRE is retained so an eligible IQ2 prefill fails at
     * the candidate boundary; unrelated shapes and short tails stay valid. */
    CHECK_POLICY("require-disable-conflict", 1, 0, 1, -1, 1, 1, 1);
#undef CHECK_POLICY
    return ok;
}

static void fill_iq2(block_iq2_xxs *matrix, uint32_t salt,
                     uint32_t n_total_expert) {
    for (uint32_t expert = 0; expert < n_total_expert; expert++) {
        for (uint32_t row = 0; row < MID_DIM; row++) {
            block_iq2_xxs *b = matrix + (uint64_t)expert * MID_DIM + row;
            const uint32_t key = salt * 977u + expert * 431u + row * 37u;
            b->d = (uint16_t)(0x1800u + ((key & 1u) ? 0x0200u : 0u));
            for (uint32_t i = 0; i < QK_K / 8u; i++) {
                b->qs[i] = (uint16_t)(key + i * 509u + (i >> 2u) * 131u);
            }
        }
    }
}

static void fill_q2(block_q2_K *matrix, uint32_t n_total_expert) {
    for (uint32_t expert = 0; expert < n_total_expert; expert++) {
        for (uint32_t row = 0; row < OUT_DIM; row++) {
            block_q2_K *b = matrix + (uint64_t)expert * OUT_DIM + row;
            const uint32_t key = expert * 617u + row * 73u;
            for (uint32_t group = 0; group < QK_K / 16u; group++) {
                const uint8_t scale =
                    (uint8_t)(1u + (key + 3u * group) % 7u);
                const uint8_t min = (uint8_t)((key / 5u + group) % 4u);
                b->scales[group] = (uint8_t)(scale | (min << 4u));
            }
            for (uint32_t i = 0; i < QK_K / 4u; i++) {
                b->qs[i] = (uint8_t)(key + 29u * i + (i >> 1u) * 7u);
            }
            b->d = 0x1800u;
            b->dmin = 0x1400u;
        }
    }
}

static void fill_iq2_full(block_iq2_xxs *matrix, uint32_t salt) {
    const uint32_t blocks_per_row = FULL_IN_DIM / QK_K;
    for (uint32_t expert = 0; expert < N_TOTAL_EXPERT; expert++) {
        for (uint32_t row = 0; row < FULL_MID_DIM; row++) {
            for (uint32_t block = 0; block < blocks_per_row; block++) {
                block_iq2_xxs *b = matrix +
                    ((uint64_t)expert * FULL_MID_DIM + row) *
                        blocks_per_row + block;
                const uint32_t key = salt * 977u + expert * 431u +
                                     row * 37u + block * 811u;
                b->d = (uint16_t)(0x1800u +
                                  ((key & 1u) ? 0x0200u : 0u));
                for (uint32_t i = 0; i < QK_K / 8u; i++) {
                    b->qs[i] = (uint16_t)(key + i * 509u +
                                          (i >> 2u) * 131u);
                }
            }
        }
    }
}

static void fill_q2_full(block_q2_K *matrix) {
    const uint32_t blocks_per_row = FULL_MID_DIM / QK_K;
    for (uint32_t expert = 0; expert < N_TOTAL_EXPERT; expert++) {
        for (uint32_t row = 0; row < FULL_OUT_DIM; row++) {
            for (uint32_t block = 0; block < blocks_per_row; block++) {
                block_q2_K *b = matrix +
                    ((uint64_t)expert * FULL_OUT_DIM + row) *
                        blocks_per_row + block;
                const uint32_t key = expert * 617u + row * 73u +
                                     block * 991u;
                for (uint32_t group = 0; group < QK_K / 16u; group++) {
                    const uint8_t scale =
                        (uint8_t)(1u + (key + 3u * group) % 7u);
                    const uint8_t min =
                        (uint8_t)((key / 5u + group) % 4u);
                    b->scales[group] =
                        (uint8_t)(scale | (uint8_t)(min << 4u));
                }
                for (uint32_t i = 0; i < QK_K / 4u; i++) {
                    b->qs[i] =
                        (uint8_t)(key + 29u * i + (i >> 1u) * 7u);
                }
                b->d = 0x1800u;
                b->dmin = 0x1400u;
            }
        }
    }
}

static int result_alloc(run_result *r, uint32_t tokens) {
    memset(r, 0, sizeof(*r));
    r->tokens = tokens;
    r->pair_count = (uint64_t)tokens * N_EXPERT * MID_DIM;
    r->out_count = (uint64_t)tokens * OUT_DIM;
    r->gate = calloc((size_t)r->pair_count, sizeof(float));
    r->up = calloc((size_t)r->pair_count, sizeof(float));
    r->mid = calloc((size_t)r->pair_count, sizeof(float));
    r->out = calloc((size_t)r->out_count, sizeof(float));
    return r->gate && r->up && r->mid && r->out;
}

static void result_free(run_result *r) {
    free(r->gate);
    free(r->up);
    free(r->mid);
    free(r->out);
    memset(r, 0, sizeof(*r));
}

static int run_once(
        const char *run_name,
        run_result *result,
        const void *model,
        uint64_t model_size,
        uint64_t gate_offset,
        uint64_t up_offset,
        uint64_t down_offset,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t n_total_expert,
        const float *x,
        const int32_t *selected,
        const float *weights,
        bool allow_mid_f16) {
    const uint32_t tokens = result->tokens;
    const uint64_t x_count = (uint64_t)tokens * IN_DIM;
    const uint64_t route_count = (uint64_t)tokens * N_EXPERT;
    const uint64_t expert_count = (uint64_t)tokens * N_EXPERT * OUT_DIM;
    const uint64_t guard_bytes =
        (uint64_t)FULL_GUARD_WORDS * sizeof(uint32_t);
    uint32_t *pair_init = full_guarded_payload(result->pair_count);
    uint32_t *expert_init = full_guarded_payload(expert_count);
    uint32_t *out_init = full_guarded_payload(result->out_count);
    if (!pair_init || !expert_init || !out_init) {
        free(pair_init);
        free(expert_init);
        free(out_init);
        return 0;
    }

    ds4_gpu_tensor *x_t = ds4_gpu_tensor_alloc(x_count * sizeof(float));
    ds4_gpu_tensor *selected_t =
        ds4_gpu_tensor_alloc(route_count * sizeof(int32_t));
    ds4_gpu_tensor *weights_t =
        ds4_gpu_tensor_alloc(route_count * sizeof(float));
    ds4_gpu_tensor *gate_t =
        ds4_gpu_tensor_alloc(result->pair_count * sizeof(float) + guard_bytes);
    ds4_gpu_tensor *up_t =
        ds4_gpu_tensor_alloc(result->pair_count * sizeof(float) + guard_bytes);
    ds4_gpu_tensor *mid_t =
        ds4_gpu_tensor_alloc(result->pair_count * sizeof(float) + guard_bytes);
    ds4_gpu_tensor *experts_t =
        ds4_gpu_tensor_alloc(expert_count * sizeof(float) + guard_bytes);
    ds4_gpu_tensor *out_t =
        ds4_gpu_tensor_alloc(result->out_count * sizeof(float) + guard_bytes);
    int ok = x_t && selected_t && weights_t && gate_t && up_t && mid_t &&
             experts_t && out_t;
    ok = ok && ds4_gpu_tensor_write(x_t, 0, x, x_count * sizeof(float));
    ok = ok && ds4_gpu_tensor_write(selected_t, 0, selected,
                                     route_count * sizeof(int32_t));
    ok = ok && ds4_gpu_tensor_write(weights_t, 0, weights,
                                     route_count * sizeof(float));
    ok = ok && ds4_gpu_tensor_write(
        gate_t, 0, pair_init,
        result->pair_count * sizeof(float) + guard_bytes);
    ok = ok && ds4_gpu_tensor_write(
        up_t, 0, pair_init,
        result->pair_count * sizeof(float) + guard_bytes);
    ok = ok && ds4_gpu_tensor_write(
        mid_t, 0, pair_init,
        result->pair_count * sizeof(float) + guard_bytes);
    ok = ok && ds4_gpu_tensor_write(
        experts_t, 0, expert_init,
        expert_count * sizeof(float) + guard_bytes);
    ok = ok && ds4_gpu_tensor_write(
        out_t, 0, out_init,
        result->out_count * sizeof(float) + guard_bytes);

    bool mid_is_f16 = true;
    if (ok) {
        ok = ds4_gpu_routed_moe_batch_tensor(
            out_t, gate_t, up_t, mid_t, experts_t, model, model_size,
            gate_offset, up_offset, down_offset, IQ2_XXS_TYPE, Q2_K_TYPE,
            gate_expert_bytes, gate_row_bytes, down_expert_bytes,
            down_row_bytes, IN_DIM, MID_DIM, OUT_DIM, selected_t, weights_t,
            n_total_expert, N_EXPERT, CLAMP, x_t, 0u, tokens,
            &mid_is_f16, false);
    }
    if (ok && mid_is_f16 && !allow_mid_f16) {
        fprintf(stderr,
                "IQ2_XXS SSD grouped-MM oracle unexpectedly selected f16 mid\n");
        ok = 0;
    }
    ok = ok && ds4_gpu_tensor_read(gate_t, 0, result->gate,
                                    result->pair_count * sizeof(float));
    ok = ok && ds4_gpu_tensor_read(up_t, 0, result->up,
                                    result->pair_count * sizeof(float));
    ok = ok && ds4_gpu_tensor_read(mid_t, 0, result->mid,
                                    result->pair_count * sizeof(float));
    ok = ok && ds4_gpu_tensor_read(out_t, 0, result->out,
                                    result->out_count * sizeof(float));

    result->guard_mismatches = 0;
    if (gate_t) {
        const int guard_ok = full_check_guard(
            run_name, "gate", gate_t, result->pair_count,
            &result->guard_mismatches);
        ok = guard_ok && ok;
    }
    if (up_t) {
        const int guard_ok = full_check_guard(
            run_name, "up", up_t, result->pair_count,
            &result->guard_mismatches);
        ok = guard_ok && ok;
    }
    if (mid_t) {
        const int guard_ok = full_check_guard(
            run_name, "mid", mid_t, result->pair_count,
            &result->guard_mismatches);
        ok = guard_ok && ok;
    }
    if (experts_t) {
        const int guard_ok = full_check_guard(
            run_name, "experts", experts_t, expert_count,
            &result->guard_mismatches);
        ok = guard_ok && ok;
    }
    if (out_t) {
        const int guard_ok = full_check_guard(
            run_name, "out", out_t, result->out_count,
            &result->guard_mismatches);
        ok = guard_ok && ok;
    }

    ds4_gpu_tensor_free(x_t);
    ds4_gpu_tensor_free(selected_t);
    ds4_gpu_tensor_free(weights_t);
    ds4_gpu_tensor_free(gate_t);
    ds4_gpu_tensor_free(up_t);
    ds4_gpu_tensor_free(mid_t);
    ds4_gpu_tensor_free(experts_t);
    ds4_gpu_tensor_free(out_t);
    free(pair_init);
    free(expert_init);
    free(out_init);
    return ok;
}

static int compare_array(const char *case_name, const char *tensor_name,
                         const float *candidate, const float *control,
                         uint64_t count, double max_limit,
                         double rms_limit) {
    double max_abs = 0.0;
    long double sum_sq = 0.0;
    uint64_t nonfinite = 0;
    uint64_t unwritten = 0;
    uint32_t sentinel_bits = 0;
    memcpy(&sentinel_bits, &(float){ SENTINEL }, sizeof(sentinel_bits));
    for (uint64_t i = 0; i < count; i++) {
        uint32_t candidate_bits = 0;
        uint32_t control_bits = 0;
        memcpy(&candidate_bits, candidate + i, sizeof(candidate_bits));
        memcpy(&control_bits, control + i, sizeof(control_bits));
        if (candidate_bits == sentinel_bits || control_bits == sentinel_bits) {
            unwritten++;
            continue;
        }
        if ((candidate_bits & 0x7f800000u) == 0x7f800000u ||
            (control_bits & 0x7f800000u) == 0x7f800000u) {
            nonfinite++;
            continue;
        }
        const double diff = fabs((double)candidate[i] - (double)control[i]);
        if (diff > max_abs) max_abs = diff;
        sum_sq += (long double)diff * (long double)diff;
    }
    const double rms = count ? sqrt((double)(sum_sq / count)) : 0.0;
    const int pass = nonfinite == 0 && unwritten == 0 &&
                     max_abs <= max_limit && rms <= rms_limit;
    fprintf(stderr,
            "IQ2_XXS SSD grouped-MM %-12s %-4s %s count=%llu "
            "max_abs=%.9g rms=%.9g limits=%.9g/%.9g nonfinite=%llu "
            "unwritten=%llu\n",
            case_name, tensor_name, pass ? "PASS" : "FAIL",
            (unsigned long long)count, max_abs, rms, max_limit, rms_limit,
            (unsigned long long)nonfinite,
            (unsigned long long)unwritten);
    return pass;
}

static int compare_results(const char *name, const run_result *candidate,
                           const run_result *control) {
    if (candidate->pair_count != control->pair_count ||
        candidate->out_count != control->out_count ||
        candidate->guard_mismatches != 0 ||
        control->guard_mismatches != 0) {
        return 0;
    }
    int ok = compare_array(name, "gate", candidate->gate, control->gate,
                           candidate->pair_count, 0.025, 0.004);
    ok = compare_array(name, "up", candidate->up, control->up,
                       candidate->pair_count, 0.025, 0.004) && ok;
    ok = compare_array(name, "mid", candidate->mid, control->mid,
                       candidate->pair_count, 0.08, 0.015) && ok;
    ok = compare_array(name, "out", candidate->out, control->out,
                       candidate->out_count, 0.08, 0.015) && ok;
    return ok;
}

static void clear_tail_cull_test_env(void) {
    unsetenv(TAIL_CULL_ENV);
    unsetenv(TAIL_CULL_DISABLE_ENV);
    unsetenv("DS4_METAL_ENABLE_IQ2_XXS_SSD_PREFILL_MM");
    unsetenv("DS4_METAL_REQUIRE_IQ2_XXS_SSD_PREFILL_MM");
    unsetenv("DS4_METAL_DISABLE_IQ2_XXS_SSD_PREFILL_MM");
}

static void configure_tail_cull_test_env(bool enable_tail_cull) {
    unsetenv(TAIL_CULL_DISABLE_ENV);
    unsetenv("DS4_METAL_DISABLE_IQ2_XXS_SSD_PREFILL_MM");
    setenv("DS4_METAL_ENABLE_IQ2_XXS_SSD_PREFILL_MM", "1", 1);
    setenv("DS4_METAL_REQUIRE_IQ2_XXS_SSD_PREFILL_MM", "1", 1);
    if (enable_tail_cull) {
        setenv(TAIL_CULL_ENV, "1", 1);
    } else {
        unsetenv(TAIL_CULL_ENV);
    }
}

static int compare_array_bit_exact(const char *case_name,
                                   const char *tensor_name,
                                   const float *candidate,
                                   const float *control,
                                   uint64_t count) {
    uint32_t sentinel_bits = 0;
    memcpy(&sentinel_bits, &(float){ SENTINEL }, sizeof(sentinel_bits));
    uint64_t mismatches = 0;
    uint64_t nonfinite = 0;
    uint64_t unwritten = 0;
    uint64_t first_mismatch = UINT64_MAX;
    uint32_t first_candidate = 0;
    uint32_t first_control = 0;
    for (uint64_t i = 0; i < count; i++) {
        uint32_t candidate_bits = 0;
        uint32_t control_bits = 0;
        memcpy(&candidate_bits, candidate + i, sizeof(candidate_bits));
        memcpy(&control_bits, control + i, sizeof(control_bits));
        if (candidate_bits == sentinel_bits || control_bits == sentinel_bits) {
            unwritten++;
        }
        if ((candidate_bits & 0x7f800000u) == 0x7f800000u ||
            (control_bits & 0x7f800000u) == 0x7f800000u) {
            nonfinite++;
        }
        if (candidate_bits != control_bits) {
            if (first_mismatch == UINT64_MAX) {
                first_mismatch = i;
                first_candidate = candidate_bits;
                first_control = control_bits;
            }
            mismatches++;
        }
    }
    const int pass = mismatches == 0 && nonfinite == 0 && unwritten == 0;
    fprintf(stderr,
            "IQ2_XXS SSD tail-cull %-20s %-4s %s count=%llu "
            "bit_mismatches=%llu nonfinite=%llu unwritten=%llu",
            case_name, tensor_name, pass ? "PASS" : "FAIL",
            (unsigned long long)count,
            (unsigned long long)mismatches,
            (unsigned long long)nonfinite,
            (unsigned long long)unwritten);
    if (first_mismatch != UINT64_MAX) {
        fprintf(stderr, " first=%llu candidate=0x%08x control=0x%08x",
                (unsigned long long)first_mismatch,
                first_candidate, first_control);
    }
    fputc('\n', stderr);
    return pass;
}

static int compare_results_bit_exact(const char *name,
                                     const run_result *candidate,
                                     const run_result *control) {
    const int shape_ok =
        candidate->pair_count == control->pair_count &&
        candidate->out_count == control->out_count &&
        candidate->guard_mismatches == 0 &&
        control->guard_mismatches == 0;
    if (!shape_ok) {
        fprintf(stderr,
                "IQ2_XXS SSD tail-cull %-20s shape/guard FAIL\n", name);
        return 0;
    }
    int ok = compare_array_bit_exact(
        name, "gate", candidate->gate, control->gate,
        candidate->pair_count);
    ok = compare_array_bit_exact(
        name, "up", candidate->up, control->up,
        candidate->pair_count) && ok;
    ok = compare_array_bit_exact(
        name, "mid", candidate->mid, control->mid,
        candidate->pair_count) && ok;
    ok = compare_array_bit_exact(
        name, "out", candidate->out, control->out,
        candidate->out_count) && ok;
    return ok;
}

static int run_tail_cull_bit_exact(
        const void *model,
        uint64_t model_size,
        uint64_t gate_offset,
        uint64_t up_offset,
        uint64_t down_offset,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t n_total_expert,
        uint32_t cache_budget,
        const float *x,
        const int32_t *selected,
        const float *weights) {
    run_result control;
    run_result candidate;
    run_result repeat;
    memset(&control, 0, sizeof(control));
    memset(&candidate, 0, sizeof(candidate));
    memset(&repeat, 0, sizeof(repeat));
    int ok = 0;

    clear_tail_cull_test_env();
    if (!result_alloc(&control, MAX_TOKENS) ||
        !result_alloc(&candidate, MAX_TOKENS) ||
        !result_alloc(&repeat, MAX_TOKENS)) {
        goto cleanup;
    }

    configure_tail_cull_test_env(false);
    ds4_gpu_set_streaming_expert_cache_budget(cache_budget);
    const int control_ok = run_once(
        "tail-cull-control", &control, model, model_size,
        gate_offset, up_offset, down_offset, gate_expert_bytes,
        gate_row_bytes, down_expert_bytes, down_row_bytes,
        n_total_expert, x, selected, weights, false);

    configure_tail_cull_test_env(true);
    ds4_gpu_set_streaming_expert_cache_budget(cache_budget);
    const int candidate_ok = run_once(
        "tail-cull-candidate", &candidate, model, model_size,
        gate_offset, up_offset, down_offset, gate_expert_bytes,
        gate_row_bytes, down_expert_bytes, down_row_bytes,
        n_total_expert, x, selected, weights, false);

    configure_tail_cull_test_env(true);
    ds4_gpu_set_streaming_expert_cache_budget(cache_budget);
    const int repeat_ok = run_once(
        "tail-cull-repeat", &repeat, model, model_size,
        gate_offset, up_offset, down_offset, gate_expert_bytes,
        gate_row_bytes, down_expert_bytes, down_row_bytes,
        n_total_expert, x, selected, weights, false);

    ok = control_ok && candidate_ok && repeat_ok;
    if (ok) {
        const int candidate_exact = compare_results_bit_exact(
            "candidate-control", &candidate, &control);
        const int repeat_exact = compare_results_bit_exact(
            "repeat-control", &repeat, &control);
        ok = candidate_exact && repeat_exact;
    }
    fprintf(stderr,
            "IQ2_XXS/Q2_K Metal SSD address-MM tail-cull bit-exact %s\n",
            ok ? "PASS" : "FAIL");

cleanup:
    clear_tail_cull_test_env();
    result_free(&control);
    result_free(&candidate);
    result_free(&repeat);
    return ok;
}

static int run_pair(
        const char *name,
        uint32_t tokens,
        const void *model,
        uint64_t model_size,
        uint64_t gate_offset,
        uint64_t up_offset,
        uint64_t down_offset,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t n_total_expert,
        uint32_t cache_budget,
        const float *x,
        const int32_t *selected,
        const float *weights) {
    run_result control;
    run_result candidate;
    memset(&control, 0, sizeof(control));
    memset(&candidate, 0, sizeof(candidate));
    if (!result_alloc(&control, tokens) || !result_alloc(&candidate, tokens)) {
        result_free(&control);
        result_free(&candidate);
        return 0;
    }

    unsetenv("DS4_METAL_ENABLE_IQ2_XXS_SSD_PREFILL_MM");
    unsetenv("DS4_METAL_REQUIRE_IQ2_XXS_SSD_PREFILL_MM");
    setenv("DS4_METAL_DISABLE_IQ2_XXS_SSD_PREFILL_MM", "1", 1);
    ds4_gpu_set_streaming_expert_cache_budget(cache_budget);
    char control_name[64];
    char candidate_name[64];
    snprintf(control_name, sizeof(control_name), "%s-control", name);
    snprintf(candidate_name, sizeof(candidate_name), "%s-candidate", name);
    int ok = run_once(control_name, &control,
                      model, model_size, gate_offset, up_offset,
                      down_offset, gate_expert_bytes, gate_row_bytes,
                      down_expert_bytes, down_row_bytes, n_total_expert,
                      x, selected, weights, false);

    unsetenv("DS4_METAL_DISABLE_IQ2_XXS_SSD_PREFILL_MM");
    unsetenv("DS4_METAL_ENABLE_IQ2_XXS_SSD_PREFILL_MM");
    unsetenv("DS4_METAL_REQUIRE_IQ2_XXS_SSD_PREFILL_MM");
    ds4_gpu_set_streaming_expert_cache_budget(cache_budget);
    ok = ok && run_once(candidate_name, &candidate,
                        model, model_size, gate_offset, up_offset,
                        down_offset, gate_expert_bytes, gate_row_bytes,
                        down_expert_bytes, down_row_bytes, n_total_expert,
                        x, selected, weights, false);
    if (ok) ok = compare_results(name, &candidate, &control);

    result_free(&control);
    result_free(&candidate);
    return ok;
}

static int read_mm_stats(mm_stats_snapshot *s) {
    memset(s, 0, sizeof(*s));
    return ds4_gpu_test_iq2_stream_addr_mm_stats(
        &s->candidate_calls, &s->calls, &s->tokens, &s->rows,
        &s->require_failures, &s->min_tokens, &s->max_tokens);
}

static int check_mm_stats_delta(const char *name,
                                const mm_stats_snapshot *before,
                                const mm_stats_snapshot *after,
                                uint64_t expected_tokens) {
    const int monotonic =
        after->candidate_calls >= before->candidate_calls &&
        after->calls >= before->calls &&
        after->tokens >= before->tokens &&
        after->rows >= before->rows &&
        after->require_failures >= before->require_failures;
    const uint64_t candidates = monotonic ?
        after->candidate_calls - before->candidate_calls : UINT64_MAX;
    const uint64_t calls = monotonic ?
        after->calls - before->calls : UINT64_MAX;
    const uint64_t tokens = monotonic ?
        after->tokens - before->tokens : UINT64_MAX;
    const uint64_t rows = monotonic ?
        after->rows - before->rows : UINT64_MAX;
    const uint64_t failures = monotonic ?
        after->require_failures - before->require_failures : UINT64_MAX;
    const int ok = monotonic && candidates == 2u && calls == 1u &&
        tokens == expected_tokens &&
        rows == expected_tokens * N_EXPERT && failures == 0u;
    fprintf(stderr,
            "IQ2_XXS SSD grouped-MM %-15s coverage %s candidates=%llu "
            "calls=%llu tokens=%llu rows=%llu require_failures=%llu\n",
            name, ok ? "PASS" : "FAIL",
            (unsigned long long)candidates,
            (unsigned long long)calls,
            (unsigned long long)tokens,
            (unsigned long long)rows,
            (unsigned long long)failures);
    return ok;
}

static int check_implicit_require_fault(
        const void *model,
        uint64_t model_size,
        uint64_t gate_offset,
        uint64_t up_offset,
        uint64_t down_offset,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        const float *x,
        const int32_t *selected,
        const float *weights) {
    run_result result;
    run_result tail_result;
    memset(&result, 0, sizeof(result));
    memset(&tail_result, 0, sizeof(tail_result));
    if (!result_alloc(&result, 32u) || !result_alloc(&tail_result, 31u)) {
        result_free(&result);
        result_free(&tail_result);
        return 0;
    }

    mm_stats_snapshot before;
    mm_stats_snapshot after;
    int ok = read_mm_stats(&before);
    ds4_gpu_test_set_flags(
        DS4_GPU_TEST_IQ2_SSD_GROUPED_PIPELINE_FAILURE);

    unsetenv("DS4_METAL_ENABLE_IQ2_XXS_SSD_PREFILL_MM");
    unsetenv("DS4_METAL_REQUIRE_IQ2_XXS_SSD_PREFILL_MM");
    unsetenv("DS4_METAL_DISABLE_IQ2_XXS_SSD_PREFILL_MM");
    ds4_gpu_set_streaming_expert_cache_budget(N_TOTAL_EXPERT);
    const int default_failed = !run_once(
        "default-fail-closed", &result, model, model_size,
        gate_offset, up_offset, down_offset, gate_expert_bytes,
        gate_row_bytes, down_expert_bytes, down_row_bytes,
        N_TOTAL_EXPERT, x, selected, weights, false);

    setenv("DS4_METAL_REQUIRE_IQ2_XXS_SSD_PREFILL_MM", "0", 1);
    ds4_gpu_set_streaming_expert_cache_budget(N_TOTAL_EXPERT);
    const int require_zero_fallback = run_once(
        "require-zero-fallback", &result, model, model_size,
        gate_offset, up_offset, down_offset, gate_expert_bytes,
        gate_row_bytes, down_expert_bytes, down_row_bytes,
        N_TOTAL_EXPERT, x, selected, weights, false);

    unsetenv("DS4_METAL_REQUIRE_IQ2_XXS_SSD_PREFILL_MM");
    setenv("DS4_METAL_DISABLE_IQ2_XXS_SSD_PREFILL_MM", "1", 1);
    ds4_gpu_set_streaming_expert_cache_budget(N_TOTAL_EXPERT);
    const int disable_fallback = run_once(
        "disable-fallback", &result, model, model_size,
        gate_offset, up_offset, down_offset, gate_expert_bytes,
        gate_row_bytes, down_expert_bytes, down_row_bytes,
        N_TOTAL_EXPERT, x, selected, weights, false);

    setenv("DS4_METAL_REQUIRE_IQ2_XXS_SSD_PREFILL_MM", "1", 1);
    ds4_gpu_set_streaming_expert_cache_budget(N_TOTAL_EXPERT);
    const int require_disable_failed = !run_once(
        "require-disable-fail", &result, model, model_size,
        gate_offset, up_offset, down_offset, gate_expert_bytes,
        gate_row_bytes, down_expert_bytes, down_row_bytes,
        N_TOTAL_EXPERT, x, selected, weights, false);

    /* The automatic fail-closed arm must not turn a cache that cannot hold
     * the complete expert domain into an error. Explicit REQUIRE remains
     * strong for exactly that same materially-ineligible configuration. */
    unsetenv("DS4_METAL_REQUIRE_IQ2_XXS_SSD_PREFILL_MM");
    unsetenv("DS4_METAL_DISABLE_IQ2_XXS_SSD_PREFILL_MM");
    ds4_gpu_set_streaming_expert_cache_budget(N_TOTAL_EXPERT - 1u);
    const int small_cache_fallback = run_once(
        "small-cache-fallback", &result, model, model_size,
        gate_offset, up_offset, down_offset, gate_expert_bytes,
        gate_row_bytes, down_expert_bytes, down_row_bytes,
        N_TOTAL_EXPERT, x, selected, weights, true);

    setenv("DS4_METAL_REQUIRE_IQ2_XXS_SSD_PREFILL_MM", "1", 1);
    ds4_gpu_set_streaming_expert_cache_budget(N_TOTAL_EXPERT - 1u);
    const int small_cache_require_failed = !run_once(
        "small-cache-require-fail", &result, model, model_size,
        gate_offset, up_offset, down_offset, gate_expert_bytes,
        gate_row_bytes, down_expert_bytes, down_row_bytes,
        N_TOTAL_EXPERT, x, selected, weights, false);

    /* A short final chunk is outside the grouped-MM candidate contract even
     * with a full cache and the injected missing pipeline. It must retain the
     * established path, including for the contradictory explicit controls. */
    unsetenv("DS4_METAL_REQUIRE_IQ2_XXS_SSD_PREFILL_MM");
    ds4_gpu_set_streaming_expert_cache_budget(N_TOTAL_EXPERT);
    const int short_tail_fallback = run_once(
        "short-tail-fallback", &tail_result, model, model_size,
        gate_offset, up_offset, down_offset, gate_expert_bytes,
        gate_row_bytes, down_expert_bytes, down_row_bytes,
        N_TOTAL_EXPERT, x, selected, weights, false);

    setenv("DS4_METAL_REQUIRE_IQ2_XXS_SSD_PREFILL_MM", "1", 1);
    setenv("DS4_METAL_DISABLE_IQ2_XXS_SSD_PREFILL_MM", "1", 1);
    ds4_gpu_set_streaming_expert_cache_budget(N_TOTAL_EXPERT);
    const int short_tail_conflict_fallback = run_once(
        "short-tail-conflict-fallback", &tail_result, model, model_size,
        gate_offset, up_offset, down_offset, gate_expert_bytes,
        gate_row_bytes, down_expert_bytes, down_row_bytes,
        N_TOTAL_EXPERT, x, selected, weights, false);

    ds4_gpu_test_set_flags(0);
    unsetenv("DS4_METAL_ENABLE_IQ2_XXS_SSD_PREFILL_MM");
    unsetenv("DS4_METAL_REQUIRE_IQ2_XXS_SSD_PREFILL_MM");
    unsetenv("DS4_METAL_DISABLE_IQ2_XXS_SSD_PREFILL_MM");
    ok = read_mm_stats(&after) && ok;

    const int monotonic =
        after.candidate_calls >= before.candidate_calls &&
        after.calls >= before.calls &&
        after.require_failures >= before.require_failures;
    const uint64_t candidate_delta = monotonic ?
        after.candidate_calls - before.candidate_calls : UINT64_MAX;
    const uint64_t call_delta = monotonic ?
        after.calls - before.calls : UINT64_MAX;
    const uint64_t failure_delta = monotonic ?
        after.require_failures - before.require_failures : UINT64_MAX;
    const int coverage_ok = monotonic && candidate_delta == 6u &&
        call_delta == 0u && failure_delta == 3u;
    ok = default_failed && require_zero_fallback && disable_fallback &&
        require_disable_failed && small_cache_fallback &&
        small_cache_require_failed && short_tail_fallback &&
        short_tail_conflict_fallback && coverage_ok && ok;
    fprintf(stderr,
            "IQ2 grouped-MM implicit REQUIRE integration %s "
            "default_fail=%d require0_fallback=%d disable_fallback=%d "
            "require_disable_fail=%d small_fallback=%d "
            "small_require_fail=%d tail_fallback=%d tail_conflict=%d "
            "candidates=%llu calls=%llu require_failures=%llu\n",
            ok ? "PASS" : "FAIL", default_failed, require_zero_fallback,
            disable_fallback, require_disable_failed, small_cache_fallback,
            small_cache_require_failed, short_tail_fallback,
            short_tail_conflict_fallback,
            (unsigned long long)candidate_delta,
            (unsigned long long)call_delta,
            (unsigned long long)failure_delta);
    result_free(&result);
    result_free(&tail_result);
    return ok;
}

static int full_result_alloc(full_run_result *r) {
    memset(r, 0, sizeof(*r));
    r->pair_count =
        (uint64_t)FULL_TOKENS * N_EXPERT * FULL_MID_DIM;
    r->out_count = (uint64_t)FULL_TOKENS * FULL_OUT_DIM;
    r->gate = calloc((size_t)r->pair_count, sizeof(float));
    r->up = calloc((size_t)r->pair_count, sizeof(float));
    r->mid = calloc((size_t)r->pair_count, sizeof(float));
    r->out = calloc((size_t)r->out_count, sizeof(float));
    return r->gate && r->up && r->mid && r->out;
}

static void full_result_free(full_run_result *r) {
    free(r->gate);
    free(r->up);
    free(r->mid);
    free(r->out);
    memset(r, 0, sizeof(*r));
}

static uint32_t *full_guarded_payload(uint64_t payload_words) {
    if (payload_words > (SIZE_MAX / sizeof(uint32_t)) - FULL_GUARD_WORDS) {
        return NULL;
    }
    uint32_t *words = malloc(
        (size_t)(payload_words + FULL_GUARD_WORDS) * sizeof(uint32_t));
    if (!words) return NULL;

    uint32_t sentinel_bits = 0;
    const float sentinel = SENTINEL;
    memcpy(&sentinel_bits, &sentinel, sizeof(sentinel_bits));
    for (uint64_t i = 0; i < payload_words; i++) words[i] = sentinel_bits;
    for (uint32_t i = 0; i < FULL_GUARD_WORDS; i++) {
        words[payload_words + i] = FULL_GUARD_BITS;
    }
    return words;
}

static int full_check_guard(const char *run_name,
                            const char *tensor_name,
                            const ds4_gpu_tensor *tensor,
                            uint64_t payload_words,
                            uint64_t *mismatches) {
    uint32_t words[FULL_GUARD_WORDS];
    if (!ds4_gpu_tensor_read(tensor,
                             payload_words * sizeof(uint32_t),
                             words,
                             sizeof(words))) {
        fprintf(stderr,
                "IQ2_XXS SSD grouped-MM %s %s guard readback FAIL\n",
                run_name, tensor_name);
        return 0;
    }
    uint64_t local = 0;
    for (uint32_t i = 0; i < FULL_GUARD_WORDS; i++) {
        if (words[i] != FULL_GUARD_BITS) local++;
    }
    *mismatches += local;
    fprintf(stderr,
            "IQ2_XXS SSD grouped-MM %-24s %-7s guard_%s=%llu\n",
            run_name, tensor_name, local == 0 ? "PASS" : "FAIL",
            (unsigned long long)local);
    return local == 0;
}

static int run_full_once(
        const char *run_name,
        full_run_result *result,
        const void *model,
        uint64_t model_size,
        uint64_t gate_offset,
        uint64_t up_offset,
        uint64_t down_offset,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        const float *x,
        const int32_t *selected,
        const float *weights) {
    const uint64_t x_count = (uint64_t)FULL_TOKENS * FULL_IN_DIM;
    const uint64_t route_count = (uint64_t)FULL_TOKENS * N_EXPERT;
    const uint64_t expert_count =
        (uint64_t)FULL_TOKENS * N_EXPERT * FULL_OUT_DIM;
    const uint64_t pair_bytes = result->pair_count * sizeof(float);
    const uint64_t expert_bytes = expert_count * sizeof(float);
    const uint64_t out_bytes = result->out_count * sizeof(float);
    const uint64_t guard_bytes =
        (uint64_t)FULL_GUARD_WORDS * sizeof(uint32_t);

    uint32_t *pair_init = full_guarded_payload(result->pair_count);
    uint32_t *expert_init = full_guarded_payload(expert_count);
    uint32_t *out_init = full_guarded_payload(result->out_count);
    ds4_gpu_tensor *x_t = NULL;
    ds4_gpu_tensor *selected_t = NULL;
    ds4_gpu_tensor *weights_t = NULL;
    ds4_gpu_tensor *gate_t = NULL;
    ds4_gpu_tensor *up_t = NULL;
    ds4_gpu_tensor *mid_t = NULL;
    ds4_gpu_tensor *experts_t = NULL;
    ds4_gpu_tensor *out_t = NULL;
    int ok = pair_init && expert_init && out_init;

    if (ok) x_t = ds4_gpu_tensor_alloc(x_count * sizeof(float));
    if (ok) selected_t =
        ds4_gpu_tensor_alloc(route_count * sizeof(int32_t));
    if (ok) weights_t = ds4_gpu_tensor_alloc(route_count * sizeof(float));
    if (ok) gate_t = ds4_gpu_tensor_alloc(pair_bytes + guard_bytes);
    if (ok) up_t = ds4_gpu_tensor_alloc(pair_bytes + guard_bytes);
    if (ok) mid_t = ds4_gpu_tensor_alloc(pair_bytes + guard_bytes);
    if (ok) experts_t = ds4_gpu_tensor_alloc(expert_bytes + guard_bytes);
    if (ok) out_t = ds4_gpu_tensor_alloc(out_bytes + guard_bytes);
    ok = ok && x_t && selected_t && weights_t && gate_t && up_t && mid_t &&
         experts_t && out_t;
    ok = ok && ds4_gpu_tensor_write(x_t, 0, x,
                                     x_count * sizeof(float));
    ok = ok && ds4_gpu_tensor_write(selected_t, 0, selected,
                                     route_count * sizeof(int32_t));
    ok = ok && ds4_gpu_tensor_write(weights_t, 0, weights,
                                     route_count * sizeof(float));
    ok = ok && ds4_gpu_tensor_write(gate_t, 0, pair_init,
                                     pair_bytes + guard_bytes);
    ok = ok && ds4_gpu_tensor_write(up_t, 0, pair_init,
                                     pair_bytes + guard_bytes);
    ok = ok && ds4_gpu_tensor_write(mid_t, 0, pair_init,
                                     pair_bytes + guard_bytes);
    ok = ok && ds4_gpu_tensor_write(experts_t, 0, expert_init,
                                     expert_bytes + guard_bytes);
    ok = ok && ds4_gpu_tensor_write(out_t, 0, out_init,
                                     out_bytes + guard_bytes);

    bool mid_is_f16 = true;
    if (ok) {
        ok = ds4_gpu_routed_moe_batch_tensor(
            out_t, gate_t, up_t, mid_t, experts_t, model, model_size,
            gate_offset, up_offset, down_offset, IQ2_XXS_TYPE, Q2_K_TYPE,
            gate_expert_bytes, gate_row_bytes, down_expert_bytes,
            down_row_bytes, FULL_IN_DIM, FULL_MID_DIM, FULL_OUT_DIM,
            selected_t, weights_t, N_TOTAL_EXPERT, N_EXPERT, CLAMP, x_t,
            0u, FULL_TOKENS, &mid_is_f16, false);
    }
    if (ok && mid_is_f16) {
        fprintf(stderr,
                "IQ2_XXS SSD full-shape %s unexpectedly selected f16 mid\n",
                run_name);
        ok = 0;
    }
    if (ok) {
        ok = ds4_gpu_tensor_read(gate_t, 0, result->gate, pair_bytes) &&
             ds4_gpu_tensor_read(up_t, 0, result->up, pair_bytes) &&
             ds4_gpu_tensor_read(mid_t, 0, result->mid, pair_bytes) &&
             ds4_gpu_tensor_read(out_t, 0, result->out, out_bytes);
    }

    result->guard_mismatches = 0;
    if (gate_t) {
        const int guard_ok = full_check_guard(
            run_name, "gate", gate_t, result->pair_count,
            &result->guard_mismatches);
        ok = guard_ok && ok;
    }
    if (up_t) {
        const int guard_ok = full_check_guard(
            run_name, "up", up_t, result->pair_count,
            &result->guard_mismatches);
        ok = guard_ok && ok;
    }
    if (mid_t) {
        const int guard_ok = full_check_guard(
            run_name, "mid", mid_t, result->pair_count,
            &result->guard_mismatches);
        ok = guard_ok && ok;
    }
    if (experts_t) {
        const int guard_ok = full_check_guard(
            run_name, "experts", experts_t, expert_count,
            &result->guard_mismatches);
        ok = guard_ok && ok;
    }
    if (out_t) {
        const int guard_ok = full_check_guard(
            run_name, "out", out_t, result->out_count,
            &result->guard_mismatches);
        ok = guard_ok && ok;
    }

    ds4_gpu_tensor_free(x_t);
    ds4_gpu_tensor_free(selected_t);
    ds4_gpu_tensor_free(weights_t);
    ds4_gpu_tensor_free(gate_t);
    ds4_gpu_tensor_free(up_t);
    ds4_gpu_tensor_free(mid_t);
    ds4_gpu_tensor_free(experts_t);
    ds4_gpu_tensor_free(out_t);
    free(pair_init);
    free(expert_init);
    free(out_init);
    return ok;
}

static int compare_full_results(const full_run_result *candidate,
                                const full_run_result *control) {
    if (candidate->pair_count != control->pair_count ||
        candidate->out_count != control->out_count ||
        candidate->guard_mismatches != 0 ||
        control->guard_mismatches != 0) {
        return 0;
    }

    /* The first M1 production-shape run measured max errors below 0.0036 for
     * every tensor (gate/up RMS below 0.00057, mid below 0.00018, out below
     * 0.0016).  The absolute bounds below retain roughly 3x max headroom and
     * 2.5x-or-better RMS headroom for compiler/device variation while still
     * rejecting a material change in accumulation or intermediate precision. */
    int ok = compare_array("full-4096", "gate", candidate->gate,
                           control->gate, candidate->pair_count,
                           0.012, 0.002);
    ok = compare_array("full-4096", "up", candidate->up,
                       control->up, candidate->pair_count,
                       0.012, 0.002) && ok;
    ok = compare_array("full-4096", "mid", candidate->mid,
                       control->mid, candidate->pair_count,
                       0.012, 0.001) && ok;
    ok = compare_array("full-4096", "out", candidate->out,
                       control->out, candidate->out_count,
                       0.012, 0.004) && ok;
    return ok;
}

static int run_full_pair(
        const void *model,
        uint64_t model_size,
        uint64_t gate_offset,
        uint64_t up_offset,
        uint64_t down_offset,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        const float *x,
        const int32_t *selected,
        const float *weights) {
    full_run_result control;
    full_run_result candidate;
    memset(&control, 0, sizeof(control));
    memset(&candidate, 0, sizeof(candidate));
    if (!full_result_alloc(&control) || !full_result_alloc(&candidate)) {
        full_result_free(&control);
        full_result_free(&candidate);
        return 0;
    }

    unsetenv("DS4_METAL_ENABLE_IQ2_XXS_SSD_PREFILL_MM");
    unsetenv("DS4_METAL_REQUIRE_IQ2_XXS_SSD_PREFILL_MM");
    setenv("DS4_METAL_DISABLE_IQ2_XXS_SSD_PREFILL_MM", "1", 1);
    ds4_gpu_set_streaming_expert_cache_budget(N_TOTAL_EXPERT);
    const int control_ok = run_full_once(
        "control", &control, model, model_size, gate_offset, up_offset,
        down_offset, gate_expert_bytes, gate_row_bytes, down_expert_bytes,
        down_row_bytes, x, selected, weights);

    unsetenv("DS4_METAL_DISABLE_IQ2_XXS_SSD_PREFILL_MM");
    setenv("DS4_METAL_ENABLE_IQ2_XXS_SSD_PREFILL_MM", "1", 1);
    setenv("DS4_METAL_REQUIRE_IQ2_XXS_SSD_PREFILL_MM", "1", 1);
    ds4_gpu_set_streaming_expert_cache_budget(N_TOTAL_EXPERT);
    const int candidate_ok = run_full_once(
        "candidate", &candidate, model, model_size, gate_offset, up_offset,
        down_offset, gate_expert_bytes, gate_row_bytes, down_expert_bytes,
        down_row_bytes, x, selected, weights);

    int ok = control_ok && candidate_ok;
    if (control_ok && candidate_ok) {
        ok = compare_full_results(&candidate, &control);
    }
    full_result_free(&control);
    full_result_free(&candidate);
    return ok;
}

static int run_full_shape_oracle(const void *restore_model,
                                 uint64_t restore_model_size,
                                 int restore_model_fd,
                                 uint64_t restore_cache_expert_bytes) {
    const uint64_t page = (uint64_t)getpagesize();
    const uint64_t gate_row_bytes =
        (FULL_IN_DIM / QK_K) * sizeof(block_iq2_xxs);
    const uint64_t gate_expert_bytes =
        (uint64_t)FULL_MID_DIM * gate_row_bytes;
    const uint64_t gate_tensor_bytes =
        (uint64_t)N_TOTAL_EXPERT * gate_expert_bytes;
    const uint64_t down_row_bytes =
        (FULL_MID_DIM / QK_K) * sizeof(block_q2_K);
    const uint64_t down_expert_bytes =
        (uint64_t)FULL_OUT_DIM * down_row_bytes;
    const uint64_t down_tensor_bytes =
        (uint64_t)N_TOTAL_EXPERT * down_expert_bytes;
    const uint64_t gate_offset = 0;
    const uint64_t up_offset = align_up(gate_tensor_bytes, page);
    const uint64_t down_offset =
        align_up(up_offset + gate_tensor_bytes, page);
    const uint64_t full_model_size =
        align_up(down_offset + down_tensor_bytes, page);

    if (gate_row_bytes != 1056u || gate_expert_bytes != 2162688u ||
        down_row_bytes != 672u || down_expert_bytes != 2752512u) {
        fprintf(stderr,
                "IQ2_XXS SSD full-shape layout FAIL gate_row=%llu "
                "gate_expert=%llu down_row=%llu down_expert=%llu\n",
                (unsigned long long)gate_row_bytes,
                (unsigned long long)gate_expert_bytes,
                (unsigned long long)down_row_bytes,
                (unsigned long long)down_expert_bytes);
        return 0;
    }
    fprintf(stderr,
            "IQ2_XXS SSD full-shape layout PASS in=%u mid=%u out=%u "
            "tokens=%u topk=%u experts=%u gate_row=%llu down_row=%llu "
            "model_bytes=%llu\n",
            FULL_IN_DIM, FULL_MID_DIM, FULL_OUT_DIM, FULL_TOKENS,
            N_EXPERT, N_TOTAL_EXPERT,
            (unsigned long long)gate_row_bytes,
            (unsigned long long)down_row_bytes,
            (unsigned long long)full_model_size);

    int ok = 1;
    int backend_switched = 0;
    void *full_model = NULL;
    float *x = NULL;
    int32_t *selected = NULL;
    float *weights = NULL;
    char tmp_path[] = "/tmp/ds4-iq2-ssd-full-mm.XXXXXX";
    int full_fd = -1;

    if (posix_memalign(&full_model, (size_t)page,
                       (size_t)full_model_size) != 0) {
        return 0;
    }
    memset(full_model, 0, (size_t)full_model_size);
    block_iq2_xxs *gate =
        (block_iq2_xxs *)((uint8_t *)full_model + gate_offset);
    block_iq2_xxs *up =
        (block_iq2_xxs *)((uint8_t *)full_model + up_offset);
    block_q2_K *down =
        (block_q2_K *)((uint8_t *)full_model + down_offset);
    fill_iq2_full(gate, 19u);
    fill_iq2_full(up, 47u);
    fill_q2_full(down);

    const uint64_t x_count = (uint64_t)FULL_TOKENS * FULL_IN_DIM;
    const uint64_t route_count = (uint64_t)FULL_TOKENS * N_EXPERT;
    x = calloc((size_t)x_count, sizeof(float));
    selected = calloc((size_t)route_count, sizeof(int32_t));
    weights = calloc((size_t)route_count, sizeof(float));
    if (!x || !selected || !weights) {
        ok = 0;
        goto cleanup;
    }
    for (uint32_t token = 0; token < FULL_TOKENS; token++) {
        for (uint32_t k = 0; k < FULL_IN_DIM; k++) {
            const int32_t v =
                (int32_t)((token * 43u + k * 29u +
                           (k >> 3u) * 17u) % 251u) - 125;
            x[(uint64_t)token * FULL_IN_DIM + k] = (float)v / 256.0f;
        }
        for (uint32_t slot = 0; slot < N_EXPERT; slot++) {
            const uint64_t route = (uint64_t)token * N_EXPERT + slot;
            selected[route] =
                (int32_t)((token * 3u + slot) % N_TOTAL_EXPERT);
            weights[route] = (float)(slot + 1u) / 21.0f;
        }
    }

    full_fd = mkstemp(tmp_path);
    if (full_fd < 0 || ftruncate(full_fd, (off_t)full_model_size) != 0) {
        fprintf(stderr, "IQ2_XXS SSD full-shape fixture creation FAIL\n");
        ok = 0;
        goto cleanup;
    }
    uint64_t written = 0;
    while (written < full_model_size) {
        const size_t chunk = full_model_size - written > (1u << 20) ?
            (1u << 20) : (size_t)(full_model_size - written);
        const ssize_t n = pwrite(full_fd,
                                 (const uint8_t *)full_model + written,
                                 chunk,
                                 (off_t)written);
        if (n <= 0) {
            fprintf(stderr, "IQ2_XXS SSD full-shape fixture write FAIL\n");
            ok = 0;
            goto cleanup;
        }
        written += (uint64_t)n;
    }

    /* The streaming cache deliberately freezes one expert-size slab class.
     * Re-seed it for the production 6.75 MiB expert after clearing the compact
     * fixture, then restore the compact class before returning. */
    ds4_gpu_set_streaming_expert_cache_budget(N_TOTAL_EXPERT);
    ds4_gpu_set_streaming_expert_cache_expert_bytes(
        2u * gate_expert_bytes + down_expert_bytes);
    if (!ds4_gpu_set_model_fd(full_fd)) {
        ok = 0;
        goto cleanup;
    }
    backend_switched = 1;
    if (!ds4_gpu_set_model_map(full_model, full_model_size)) {
        ok = 0;
        goto cleanup;
    }

    uint64_t before_candidates = 0;
    uint64_t before_calls = 0;
    uint64_t before_tokens = 0;
    uint64_t before_rows = 0;
    uint64_t before_failures = 0;
    uint32_t before_min = 0;
    uint32_t before_max = 0;
    if (!ds4_gpu_test_iq2_stream_addr_mm_stats(
            &before_candidates, &before_calls, &before_tokens, &before_rows,
            &before_failures, &before_min, &before_max)) {
        ok = 0;
        goto cleanup;
    }

    const int pair_ok = run_full_pair(
        full_model, full_model_size, gate_offset, up_offset, down_offset,
        gate_expert_bytes, gate_row_bytes, down_expert_bytes, down_row_bytes,
        x, selected, weights);

    uint64_t after_candidates = 0;
    uint64_t after_calls = 0;
    uint64_t after_tokens = 0;
    uint64_t after_rows = 0;
    uint64_t after_failures = 0;
    uint32_t after_min = 0;
    uint32_t after_max = 0;
    const int stats_ok = ds4_gpu_test_iq2_stream_addr_mm_stats(
        &after_candidates, &after_calls, &after_tokens, &after_rows,
        &after_failures, &after_min, &after_max);
    const int monotonic = stats_ok &&
        after_candidates >= before_candidates &&
        after_calls >= before_calls &&
        after_tokens >= before_tokens &&
        after_rows >= before_rows &&
        after_failures >= before_failures;
    const uint64_t delta_candidates = monotonic ?
        after_candidates - before_candidates : UINT64_MAX;
    const uint64_t delta_calls = monotonic ?
        after_calls - before_calls : UINT64_MAX;
    const uint64_t delta_tokens = monotonic ?
        after_tokens - before_tokens : UINT64_MAX;
    const uint64_t delta_rows = monotonic ?
        after_rows - before_rows : UINT64_MAX;
    const uint64_t delta_failures = monotonic ?
        after_failures - before_failures : UINT64_MAX;
    const int coverage_ok = monotonic &&
        delta_candidates == 2u &&
        delta_calls == 1u &&
        delta_tokens == FULL_TOKENS &&
        delta_rows == (uint64_t)FULL_TOKENS * N_EXPERT &&
        delta_failures == 0u;
    fprintf(stderr,
            "IQ2_XXS SSD full-shape coverage %s candidates=%llu calls=%llu "
            "tokens=%llu rows=%llu require_failures=%llu "
            "global_min=%u->%u global_max=%u->%u\n",
            coverage_ok ? "PASS" : "FAIL",
            (unsigned long long)delta_candidates,
            (unsigned long long)delta_calls,
            (unsigned long long)delta_tokens,
            (unsigned long long)delta_rows,
            (unsigned long long)delta_failures,
            before_min, after_min, before_max, after_max);
    ok = pair_ok && coverage_ok && ok;

cleanup:
    unsetenv("DS4_METAL_ENABLE_IQ2_XXS_SSD_PREFILL_MM");
    unsetenv("DS4_METAL_REQUIRE_IQ2_XXS_SSD_PREFILL_MM");
    unsetenv("DS4_METAL_DISABLE_IQ2_XXS_SSD_PREFILL_MM");
    if (backend_switched) {
        ds4_gpu_set_streaming_expert_cache_budget(N_TOTAL_EXPERT);
        ds4_gpu_set_streaming_expert_cache_expert_bytes(
            restore_cache_expert_bytes);
        const int fd_ok = ds4_gpu_set_model_fd(restore_model_fd);
        const int map_ok = ds4_gpu_set_model_map(
            restore_model, restore_model_size);
        if (!fd_ok || !map_ok) {
            fprintf(stderr,
                    "IQ2_XXS SSD full-shape backend restoration FAIL\n");
            ok = 0;
        }
    }
    if (full_fd >= 0) close(full_fd);
    if (full_fd >= 0) unlink(tmp_path);
    free(x);
    free(selected);
    free(weights);
    free(full_model);
    fprintf(stderr,
            "IQ2_XXS/Q2_K Metal SSD full production-shape oracle %s\n",
            ok ? "PASS" : "FAIL");
    return ok;
}

static int run_256_expert_oracle(
        const void *restore_model,
        uint64_t restore_model_size,
        int restore_model_fd,
        uint64_t restore_gate_expert_bytes,
        uint64_t restore_gate_row_bytes,
        uint64_t restore_down_expert_bytes,
        uint64_t restore_down_row_bytes) {
    const uint64_t page = (uint64_t)getpagesize();
    const uint64_t gate_row_bytes = sizeof(block_iq2_xxs);
    const uint64_t gate_expert_bytes = MID_DIM * gate_row_bytes;
    const uint64_t gate_tensor_bytes =
        (uint64_t)N_TOTAL_EXPERT_256 * gate_expert_bytes;
    const uint64_t down_row_bytes = sizeof(block_q2_K);
    const uint64_t down_expert_bytes = OUT_DIM * down_row_bytes;
    const uint64_t down_tensor_bytes =
        (uint64_t)N_TOTAL_EXPERT_256 * down_expert_bytes;
    const uint64_t gate_offset = 0;
    const uint64_t up_offset = align_up(gate_tensor_bytes, page);
    const uint64_t down_offset =
        align_up(up_offset + gate_tensor_bytes, page);
    const uint64_t model_size =
        align_up(down_offset + down_tensor_bytes, page);

    int ok = 1;
    int backend_switched = 0;
    void *model = NULL;
    float *x = NULL;
    int32_t *selected = NULL;
    float *weights = NULL;
    char tmp_path[] = "/tmp/ds4-iq2-ssd-256-mm.XXXXXX";
    int model_fd = -1;

    if (posix_memalign(&model, (size_t)page, (size_t)model_size) != 0) {
        return 0;
    }
    memset(model, 0, (size_t)model_size);
    block_iq2_xxs *gate =
        (block_iq2_xxs *)((uint8_t *)model + gate_offset);
    block_iq2_xxs *up =
        (block_iq2_xxs *)((uint8_t *)model + up_offset);
    block_q2_K *down =
        (block_q2_K *)((uint8_t *)model + down_offset);
    fill_iq2(gate, 23u, N_TOTAL_EXPERT_256);
    fill_iq2(up, 53u, N_TOTAL_EXPERT_256);
    fill_q2(down, N_TOTAL_EXPERT_256);

    const uint32_t tokens = 33u;
    const uint64_t x_count = (uint64_t)tokens * IN_DIM;
    const uint64_t route_count = (uint64_t)tokens * N_EXPERT;
    x = calloc((size_t)x_count, sizeof(float));
    selected = calloc((size_t)route_count, sizeof(int32_t));
    weights = calloc((size_t)route_count, sizeof(float));
    if (!x || !selected || !weights) {
        ok = 0;
        goto cleanup;
    }
    for (uint32_t token = 0; token < tokens; token++) {
        for (uint32_t k = 0; k < IN_DIM; k++) {
            const int32_t v =
                (int32_t)((token * 31u + k * 17u + (k >> 2u) * 7u) %
                          127u) - 63;
            x[(uint64_t)token * IN_DIM + k] = (float)v / 96.0f;
        }
        for (uint32_t slot = 0; slot < N_EXPERT; slot++) {
            const uint64_t route = (uint64_t)token * N_EXPERT + slot;
            if (slot == 0u) {
                /* Exactly 33 routes: one full tile plus a one-row tail. */
                selected[route] = (int32_t)HIGH_EXPERT_ID;
            } else if (slot == 1u && token < 15u) {
                selected[route] = 0;
            } else if (slot == 2u && token == 0u) {
                /* Duplicate expert zero within token zero. Together with the
                 * 15 routes above this makes the critical nr1 == 16 tile. */
                selected[route] = 0;
            } else if (slot == 1u && token < 32u) {
                /* Exactly 17 routes exercise the first non-culled row half. */
                selected[route] = 1;
            } else {
                /* Keep all remaining routes away from 0, 1, and 255 so the
                 * three boundary counts remain exact. */
                selected[route] =
                    (int32_t)(2u + (token * 17u + slot * 29u) % 252u);
            }
            weights[route] = (float)(slot + 1u) / 21.0f;
        }
    }
    uint32_t routes_16 = 0;
    uint32_t routes_17 = 0;
    uint32_t high_id_routes = 0;
    uint32_t duplicate_routes = 0;
    for (uint32_t token = 0; token < tokens; token++) {
        for (uint32_t slot = 0; slot < N_EXPERT; slot++) {
            const int32_t id = selected[(uint64_t)token * N_EXPERT + slot];
            if (id == 0) routes_16++;
            if (id == 1) routes_17++;
            if (id == (int32_t)HIGH_EXPERT_ID) high_id_routes++;
            for (uint32_t prior = 0; prior < slot; prior++) {
                if (id == selected[(uint64_t)token * N_EXPERT + prior]) {
                    duplicate_routes++;
                    break;
                }
            }
        }
    }
    const uint32_t high_id_work_items = (high_id_routes + 31u) / 32u;
    if (routes_16 != 16u || routes_17 != 17u || high_id_routes != 33u ||
        duplicate_routes == 0u || high_id_work_items != 2u) {
        fprintf(stderr,
                "IQ2_XXS SSD tail-cull route construction FAIL "
                "rows16=%u rows17=%u id255_rows=%u duplicates=%u "
                "work_items=%u\n",
                routes_16, routes_17, high_id_routes, duplicate_routes,
                high_id_work_items);
        ok = 0;
        goto cleanup;
    }
    fprintf(stderr,
            "IQ2_XXS SSD tail-cull routes PASS model_bytes=%llu "
            "rows16=%u rows17=%u id255_rows=%u duplicates=%u "
            "work_items=%u second_tile_r1=32\n",
            (unsigned long long)model_size, routes_16, routes_17,
            high_id_routes, duplicate_routes,
            high_id_work_items);

    model_fd = mkstemp(tmp_path);
    if (model_fd < 0 || ftruncate(model_fd, (off_t)model_size) != 0) {
        ok = 0;
        goto cleanup;
    }
    uint64_t written = 0;
    while (written < model_size) {
        const size_t chunk = model_size - written > (1u << 20) ?
            (1u << 20) : (size_t)(model_size - written);
        const ssize_t n = pwrite(model_fd,
                                 (const uint8_t *)model + written,
                                 chunk,
                                 (off_t)written);
        if (n <= 0) {
            ok = 0;
            goto cleanup;
        }
        written += (uint64_t)n;
    }

    ds4_gpu_set_streaming_expert_cache_budget(N_TOTAL_EXPERT_256);
    ds4_gpu_set_streaming_expert_cache_expert_bytes(
        2u * gate_expert_bytes + down_expert_bytes);
    if (!ds4_gpu_set_model_fd(model_fd)) {
        ok = 0;
        goto cleanup;
    }
    backend_switched = 1;
    if (!ds4_gpu_set_model_map(model, model_size)) {
        ok = 0;
        goto cleanup;
    }

    mm_stats_snapshot before;
    mm_stats_snapshot after;
    if (!read_mm_stats(&before)) {
        ok = 0;
        goto cleanup;
    }
    const int pair_ok = run_pair(
        "id255-hot-33", tokens, model, model_size,
        gate_offset, up_offset, down_offset,
        gate_expert_bytes, gate_row_bytes,
        down_expert_bytes, down_row_bytes,
        N_TOTAL_EXPERT_256, N_TOTAL_EXPERT_256,
        x, selected, weights);
    const int stats_ok = read_mm_stats(&after) &&
        check_mm_stats_delta("id255-hot-33", &before, &after, tokens);
    const int tail_cull_ok = run_tail_cull_bit_exact(
        model, model_size, gate_offset, up_offset, down_offset,
        gate_expert_bytes, gate_row_bytes, down_expert_bytes,
        down_row_bytes, N_TOTAL_EXPERT_256, N_TOTAL_EXPERT_256,
        x, selected, weights);
    ok = pair_ok && stats_ok && tail_cull_ok && ok;

cleanup:
    clear_tail_cull_test_env();
    if (backend_switched) {
        ds4_gpu_set_streaming_expert_cache_budget(N_TOTAL_EXPERT);
        ds4_gpu_set_streaming_expert_cache_expert_bytes(
            2u * restore_gate_expert_bytes + restore_down_expert_bytes);
        const int fd_ok = ds4_gpu_set_model_fd(restore_model_fd);
        const int map_ok = ds4_gpu_set_model_map(
            restore_model, restore_model_size);
        if (!fd_ok || !map_ok ||
            restore_gate_row_bytes != sizeof(block_iq2_xxs) ||
            restore_down_row_bytes != sizeof(block_q2_K)) {
            fprintf(stderr,
                    "IQ2_XXS SSD 256-expert backend restoration FAIL\n");
            ok = 0;
        }
    }
    if (model_fd >= 0) close(model_fd);
    if (model_fd >= 0) unlink(tmp_path);
    free(x);
    free(selected);
    free(weights);
    free(model);
    fprintf(stderr,
            "IQ2_XXS/Q2_K Metal compact 256-expert ID255 oracle %s\n",
            ok ? "PASS" : "FAIL");
    return ok;
}

int main(void) {
    if (sizeof(block_iq2_xxs) != 66u || sizeof(block_q2_K) != 84u) {
        fprintf(stderr,
                "IQ2_XXS SSD grouped-MM unexpected block sizes iq2=%zu q2=%zu\n",
                sizeof(block_iq2_xxs), sizeof(block_q2_K));
        return 1;
    }

    int ok = check_grouped_mm_policy();

    const uint64_t gate_row_bytes = sizeof(block_iq2_xxs);
    const uint64_t gate_expert_bytes = MID_DIM * gate_row_bytes;
    const uint64_t gate_tensor_bytes = N_TOTAL_EXPERT * gate_expert_bytes;
    const uint64_t down_row_bytes = sizeof(block_q2_K);
    const uint64_t down_expert_bytes = OUT_DIM * down_row_bytes;
    const uint64_t down_tensor_bytes = N_TOTAL_EXPERT * down_expert_bytes;
    const uint64_t page = (uint64_t)getpagesize();
    const uint64_t gate_offset = 0;
    const uint64_t up_offset = align_up(gate_tensor_bytes, page);
    const uint64_t down_offset = align_up(up_offset + gate_tensor_bytes, page);
    const uint64_t model_size = align_up(down_offset + down_tensor_bytes, page);

    void *model = NULL;
    if (posix_memalign(&model, (size_t)page, (size_t)model_size) != 0) {
        return 1;
    }
    memset(model, 0, (size_t)model_size);
    block_iq2_xxs *gate =
        (block_iq2_xxs *)((uint8_t *)model + gate_offset);
    block_iq2_xxs *up =
        (block_iq2_xxs *)((uint8_t *)model + up_offset);
    block_q2_K *down = (block_q2_K *)((uint8_t *)model + down_offset);
    fill_iq2(gate, 3u, N_TOTAL_EXPERT);
    fill_iq2(up, 11u, N_TOTAL_EXPERT);
    fill_q2(down, N_TOTAL_EXPERT);

    float *x = calloc((size_t)MAX_TOKENS * IN_DIM, sizeof(float));
    int32_t *selected =
        calloc((size_t)MAX_TOKENS * N_EXPERT, sizeof(int32_t));
    int32_t *selected_duplicate =
        calloc((size_t)MAX_TOKENS * N_EXPERT, sizeof(int32_t));
    int32_t *selected_hot =
        calloc((size_t)MAX_TOKENS * N_EXPERT, sizeof(int32_t));
    float *weights =
        calloc((size_t)MAX_TOKENS * N_EXPERT, sizeof(float));
    ok = x && selected && selected_duplicate && selected_hot && weights && ok;
    for (uint32_t token = 0; ok && token < MAX_TOKENS; token++) {
        for (uint32_t k = 0; k < IN_DIM; k++) {
            const int32_t v = (int32_t)((token * 17u + k * 11u +
                                         (token ^ k) * 3u) % 97u) - 48;
            x[(uint64_t)token * IN_DIM + k] = (float)v / 64.0f;
        }
        for (uint32_t slot = 0; slot < N_EXPERT; slot++) {
            const uint64_t route = (uint64_t)token * N_EXPERT + slot;
            selected[route] = (int32_t)((token + slot) % N_TOTAL_EXPERT);
            selected_duplicate[route] = selected[route];
            /* Expert zero receives exactly one route per token. At 33 tokens
             * this forces work items at r1=0 and r1=32 without relying on
             * duplicate IDs inside one token. */
            selected_hot[route] = slot == 0u ? 0 :
                (int32_t)(1u + ((token + slot - 1u) %
                                (N_TOTAL_EXPERT - 1u)));
            weights[route] = 1.0f / N_EXPERT;
        }
        selected_duplicate[(uint64_t)token * N_EXPERT + N_EXPERT - 1u] =
            selected_duplicate[(uint64_t)token * N_EXPERT];
    }

    char tmp_path[] = "/tmp/ds4-iq2-ssd-grouped-mm.XXXXXX";
    int model_fd = ok ? mkstemp(tmp_path) : -1;
    if (model_fd < 0 || ftruncate(model_fd, (off_t)model_size) != 0) {
        fprintf(stderr, "IQ2_XXS SSD grouped-MM could not create fixture\n");
        ok = 0;
    }
    uint64_t written = 0;
    while (ok && written < model_size) {
        const size_t chunk = model_size - written > (1u << 20) ?
            (1u << 20) : (size_t)(model_size - written);
        const ssize_t n = pwrite(model_fd, (const uint8_t *)model + written,
                                 chunk, (off_t)written);
        if (n <= 0) {
            fprintf(stderr, "IQ2_XXS SSD grouped-MM fixture write failed\n");
            ok = 0;
        } else {
            written += (uint64_t)n;
        }
    }

    unsetenv("DS4_METAL_DISABLE_STREAMING_PREFILL_BATCH_SELECTED_ADDR");
    unsetenv("DS4_METAL_DISABLE_STREAMING_EXPERT_ADDR_TABLE");
    unsetenv("DS4_METAL_DISABLE_ROUTED_PAIR_SWIGLU_FUSION");
    unsetenv("DS4_METAL_MOE_WRITE_CLAMPED_ACT");
    unsetenv("DS4_METAL_GRAPH_DUMP_PREFIX");
    clear_tail_cull_test_env();
    setenv("DS4_METAL_ENABLE_STREAMING_PREFILL_BATCH_SELECTED_ADDR", "1", 1);
    setenv("DS4_METAL_ENABLE_STREAMING_EXPERT_ADDR_TABLE", "1", 1);
    setenv("DS4_METAL_IQ2_XXS_SSD_PREFILL_MM_STATS", "1", 1);

    ok = ok && ds4_gpu_init() && ds4_gpu_set_model_map(model, model_size);
    ds4_gpu_set_quality(false);
    ds4_gpu_set_ssd_streaming(true);
    ds4_gpu_set_streaming_expert_cache_budget(N_TOTAL_EXPERT);
    ok = ok && ds4_gpu_set_model_fd(model_fd);
    const int backend_ready = ok;

    if (backend_ready) {
        const int case_ok = run_pair("tile-32", 32u, model, model_size,
                                     gate_offset, up_offset, down_offset,
                                     gate_expert_bytes, gate_row_bytes,
                                     down_expert_bytes, down_row_bytes,
                                     N_TOTAL_EXPERT, N_TOTAL_EXPERT,
                                     x, selected, weights);
        ok = case_ok && ok;
    }
    if (backend_ready) {
        const int case_ok = run_pair("tail-33", 33u, model, model_size,
                                     gate_offset, up_offset, down_offset,
                                     gate_expert_bytes, gate_row_bytes,
                                     down_expert_bytes, down_row_bytes,
                                     N_TOTAL_EXPERT, N_TOTAL_EXPERT,
                                     x, selected, weights);
        ok = case_ok && ok;
    }
    if (backend_ready) {
        const int case_ok = run_pair("duplicate-32", 32u, model, model_size,
                                     gate_offset, up_offset, down_offset,
                                     gate_expert_bytes, gate_row_bytes,
                                     down_expert_bytes, down_row_bytes,
                                     N_TOTAL_EXPERT, N_TOTAL_EXPERT,
                                     x, selected_duplicate, weights);
        ok = case_ok && ok;
    }

    uint64_t candidate_calls = 0;
    uint64_t calls = 0;
    uint64_t tokens = 0;
    uint64_t rows = 0;
    uint64_t require_failures = 0;
    uint32_t min_tokens = 0;
    uint32_t max_tokens = 0;
    const int stats_ok = ds4_gpu_test_iq2_stream_addr_mm_stats(
        &candidate_calls, &calls, &tokens, &rows, &require_failures,
        &min_tokens, &max_tokens);
    const int counters_ok =
        candidate_calls == 6u && calls == 3u && tokens == 97u &&
        rows == 582u && require_failures == 0u &&
        min_tokens == 32u && max_tokens == 33u;
    fprintf(stderr,
            "IQ2_XXS SSD grouped-MM coverage %s candidates=%llu calls=%llu "
            "tokens=%llu rows=%llu require_failures=%llu min=%u max=%u\n",
            counters_ok ? "PASS" : "FAIL",
            (unsigned long long)candidate_calls,
            (unsigned long long)calls,
            (unsigned long long)tokens,
            (unsigned long long)rows,
            (unsigned long long)require_failures,
            min_tokens, max_tokens);
    ok = stats_ok && counters_ok && ok;

    if (backend_ready) {
        const int policy_fault_ok = check_implicit_require_fault(
            model, model_size, gate_offset, up_offset, down_offset,
            gate_expert_bytes, gate_row_bytes, down_expert_bytes,
            down_row_bytes, x, selected, weights);
        ok = policy_fault_ok && ok;
    }

    if (backend_ready) {
        uint32_t hot_rows = 0;
        for (uint32_t token = 0; token < 33u; token++) {
            for (uint32_t slot = 0; slot < N_EXPERT; slot++) {
                if (selected_hot[(uint64_t)token * N_EXPERT + slot] == 0) {
                    hot_rows++;
                }
            }
        }
        const uint32_t hot_work_items = (hot_rows + 31u) / 32u;
        const int route_shape_ok = hot_rows >= 33u && hot_work_items >= 2u;
        fprintf(stderr,
                "IQ2_XXS SSD compact hot-route %s expert=0 rows=%u "
                "work_items=%u second_tile_r1=32\n",
                route_shape_ok ? "PASS" : "FAIL",
                hot_rows, hot_work_items);
        mm_stats_snapshot before;
        mm_stats_snapshot after;
        const int before_ok = read_mm_stats(&before);
        const int pair_ok = route_shape_ok && run_pair(
            "hot-r1-32", 33u, model, model_size,
            gate_offset, up_offset, down_offset,
            gate_expert_bytes, gate_row_bytes,
            down_expert_bytes, down_row_bytes,
            N_TOTAL_EXPERT, N_TOTAL_EXPERT,
            x, selected_hot, weights);
        const int delta_ok = before_ok && read_mm_stats(&after) &&
            check_mm_stats_delta("hot-r1-32", &before, &after, 33u);
        ok = pair_ok && delta_ok && ok;
    }

    if (backend_ready) {
        const int id255_ok = run_256_expert_oracle(
            model, model_size, model_fd,
            gate_expert_bytes, gate_row_bytes,
            down_expert_bytes, down_row_bytes);
        ok = id255_ok && ok;
    }

    /* Keep the full production geometry as a distinct oracle and report its
     * counter deltas independently from the compact 32/33 smoke cases. */
    if (backend_ready) {
        const int full_ok =
            run_full_shape_oracle(
                model, model_size, model_fd,
                2u * gate_expert_bytes + down_expert_bytes);
        ok = full_ok && ok;
    }

    clear_tail_cull_test_env();
    ds4_gpu_set_model_fd(-1);
    ds4_gpu_set_ssd_streaming(false);
    ds4_gpu_cleanup();
    if (model_fd >= 0) close(model_fd);
    if (tmp_path[0]) unlink(tmp_path);
    free(x);
    free(selected);
    free(selected_duplicate);
    free(selected_hot);
    free(weights);
    free(model);

    fprintf(stderr, "IQ2_XXS/Q2_K Metal SSD grouped address-MM %s\n",
            ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

#else

int main(void) {
    fprintf(stderr,
            "test_metal_iq2_ssd_grouped_mm: skipped (Metal requires macOS)\n");
    return 0;
}

#endif
