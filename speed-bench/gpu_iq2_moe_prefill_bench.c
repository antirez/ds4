#define _POSIX_C_SOURCE 200809L

#include "ds4_gpu.h"

#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#if defined(DS4_BENCH_ROCM) && defined(DS4_BENCH_CUDA)
#error "define exactly one of DS4_BENCH_ROCM or DS4_BENCH_CUDA"
#elif !defined(DS4_BENCH_ROCM) && !defined(DS4_BENCH_CUDA)
#error "define exactly one of DS4_BENCH_ROCM or DS4_BENCH_CUDA"
#endif

/*
 * This source intentionally has no implicit backend selection.  Compile it
 * once with DS4_BENCH_ROCM and link the ROCm implementation, or once with
 * DS4_BENCH_CUDA and link the CUDA implementation.
 *
 * ROCm exposes a real in-process A/B policy switch.  CUDA currently has no
 * independent tail-cull candidate selected by this harness; its build is a
 * measurement-only run of the current production path.  In particular, the
 * CUDA markers never label two executions of the same path as an A/B result.
 */
#if defined(DS4_BENCH_ROCM)
#define BENCH_BACKEND "rocm"
#define TAIL_ENABLE_ENV  "DS4_ROCM_ENABLE_IQ2_MOE_WMMA_TAIL_CULL"
#define TAIL_DISABLE_ENV "DS4_ROCM_DISABLE_IQ2_MOE_WMMA_TAIL_CULL"
#define ROCM_PROFILE_ENV "DS4_ROCM_IQ2_MOE_WMMA_PROFILE"
#else
#define BENCH_BACKEND "cuda"
#define CUDA_PROFILE_ENV "DS4_CUDA_MOE_PROFILE"
#endif

#define IQ2_XXS_TYPE 16u
#define Q2_K_TYPE 10u
#define QK_K 256u
#define IN_DIM 4096u
#define MID_DIM 2048u
#define OUT_DIM 4096u
#define N_TOKENS 4096u
#define N_TOTAL_EXPERT 256u
#define N_EXPERT 6u
#define CLAMP 4.0f
#define GUARD_WORDS 64u
#define GUARD_BYTES ((uint64_t)GUARD_WORDS * sizeof(uint32_t))
#define GUARD_BITS 0x51a7c3e9u
#define DEFAULT_SAMPLES 8u
#define DEFAULT_WARMUPS 2u
#define IO_CHUNK_BYTES (8u * 1024u * 1024u)
#define GIB (1024ull * 1024ull * 1024ull)

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

_Static_assert(sizeof(block_iq2_xxs) == 66u,
               "IQ2_XXS block layout changed");
_Static_assert(sizeof(block_q2_K) == 84u,
               "Q2_K block layout changed");

typedef struct {
    uint32_t samples;
    uint32_t warmups;
} bench_config;

typedef enum {
    ARM_BASELINE,
    ARM_CANDIDATE,
    ARM_CURRENT,
} bench_arm;

typedef struct {
    void *model;
    uint64_t model_size;
    uint64_t gate_offset;
    uint64_t up_offset;
    uint64_t down_offset;
    uint64_t gate_row_bytes;
    uint64_t gate_expert_bytes;
    uint64_t down_row_bytes;
    uint64_t down_expert_bytes;

    uint64_t x_bytes;
    uint64_t route_count;
    uint64_t route_i32_bytes;
    uint64_t route_f32_bytes;
    uint64_t pair_count;
    uint64_t pair_f32_bytes;
    uint64_t expert_count;
    uint64_t expert_bytes;
    uint64_t out_count;
    uint64_t out_bytes;

    ds4_gpu_tensor *x;
    ds4_gpu_tensor *selected;
    ds4_gpu_tensor *weights;
    ds4_gpu_tensor *gate;
    ds4_gpu_tensor *up;
    ds4_gpu_tensor *mid;
    ds4_gpu_tensor *experts;
    ds4_gpu_tensor *out;
} fixture;

#if defined(DS4_BENCH_ROCM)
typedef struct {
    uint8_t *storage;
    uint8_t *gate;
    uint8_t *up;
    uint8_t *mid;
    uint8_t *experts;
    uint8_t *out;
    uint64_t bytes;
} oracle_snapshot;
#endif

bool ds4_log_is_tty(FILE *fp) {
    (void)fp;
    return false;
}

#if defined(DS4_BENCH_CUDA)
extern uint64_t ds4_cuda_test_moe_fast_profile_report_count(void);
#endif

static void usage(FILE *fp, const char *argv0) {
    fprintf(fp,
            "usage: %s [options]\n"
            "\n"
            "Resident production-geometry IQ2_XXS/Q2_K routed-MoE prefill "
            "benchmark (%s build).\n",
            argv0, BENCH_BACKEND);
#if defined(DS4_BENCH_ROCM)
    fprintf(fp,
            "ROCm performs a real balanced A/B: DISABLE=1 is baseline and "
            "ENABLE=1 is candidate. The harness enables the GPU-only "
            "IQ2/Q2 WMMA profiler.\n"
            "  --samples N    samples per arm, even (default: %u)\n"
            "  --warmups N    warmups per arm, even (default: %u)\n",
            DEFAULT_SAMPLES, DEFAULT_WARMUPS);
#else
    fprintf(fp,
            "CUDA is measurement-only: every marker is variant=current and "
            "DS4_CUDA_MOE_PROFILE is enabled.\n"
            "No reliable baseline/candidate selector exists in this process; "
            "this harness does not report a false A/B.\n"
            "  --samples N    current-path measured runs (default: %u)\n"
            "  --warmups N    current-path warmup runs (default: %u)\n",
            DEFAULT_SAMPLES, DEFAULT_WARMUPS);
#endif
    fprintf(fp, "  -h, --help     show this help\n");
}

static uint32_t parse_u32(const char *text, const char *option) {
    char *end = NULL;
    errno = 0;
    const unsigned long long value = strtoull(text, &end, 10);
    if (errno != 0 || !text[0] || !end || *end || value > UINT32_MAX) {
        fprintf(stderr, "gpu-iq2-moe-prefill-bench: invalid %s: %s\n",
                option, text);
        exit(2);
    }
    return (uint32_t)value;
}

static const char *need_arg(int *index, int argc, char **argv) {
    if (*index + 1 >= argc) {
        fprintf(stderr, "gpu-iq2-moe-prefill-bench: %s needs a value\n",
                argv[*index]);
        exit(2);
    }
    return argv[++*index];
}

static bench_config parse_options(int argc, char **argv) {
    bench_config config = {
        .samples = DEFAULT_SAMPLES,
        .warmups = DEFAULT_WARMUPS,
    };
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
            usage(stdout, argv[0]);
            exit(0);
        } else if (!strcmp(argv[i], "--samples")) {
            const char *option = argv[i];
            config.samples = parse_u32(need_arg(&i, argc, argv), option);
        } else if (!strcmp(argv[i], "--warmups")) {
            const char *option = argv[i];
            config.warmups = parse_u32(need_arg(&i, argc, argv), option);
        } else {
            fprintf(stderr, "gpu-iq2-moe-prefill-bench: unknown option: %s\n",
                    argv[i]);
            usage(stderr, argv[0]);
            exit(2);
        }
    }
    if (config.samples == 0u) {
        fprintf(stderr, "gpu-iq2-moe-prefill-bench: --samples must be nonzero\n");
        exit(2);
    }
#if defined(DS4_BENCH_ROCM)
    if ((config.samples & 1u) != 0u || (config.warmups & 1u) != 0u) {
        fprintf(stderr,
                "gpu-iq2-moe-prefill-bench: ROCm --samples and --warmups "
                "must be even\n");
        exit(2);
    }
#endif
    return config;
}

static uint64_t align_up(uint64_t value, uint64_t alignment) {
    return (value + alignment - 1u) / alignment * alignment;
}

static uint32_t mix32(uint32_t value) {
    value ^= value >> 16u;
    value *= 0x7feb352du;
    value ^= value >> 15u;
    value *= 0x846ca68bu;
    value ^= value >> 16u;
    return value;
}

static void fill_iq2(block_iq2_xxs *matrix, uint32_t salt) {
    const uint32_t blocks_per_row = IN_DIM / QK_K;
    for (uint32_t expert = 0; expert < N_TOTAL_EXPERT; expert++) {
        for (uint32_t row = 0; row < MID_DIM; row++) {
            for (uint32_t block = 0; block < blocks_per_row; block++) {
                block_iq2_xxs *b = matrix +
                    ((uint64_t)expert * MID_DIM + row) * blocks_per_row + block;
                const uint32_t key = salt * 977u + expert * 431u +
                                     row * 37u + block * 811u;
                b->d = (uint16_t)(0x1800u + ((key & 1u) ? 0x0200u : 0u));
                for (uint32_t i = 0; i < QK_K / 8u; i++) {
                    b->qs[i] = (uint16_t)(key + i * 509u +
                                          (i >> 2u) * 131u);
                }
            }
        }
    }
}

static void fill_q2(block_q2_K *matrix) {
    const uint32_t blocks_per_row = MID_DIM / QK_K;
    for (uint32_t expert = 0; expert < N_TOTAL_EXPERT; expert++) {
        for (uint32_t row = 0; row < OUT_DIM; row++) {
            for (uint32_t block = 0; block < blocks_per_row; block++) {
                block_q2_K *b = matrix +
                    ((uint64_t)expert * OUT_DIM + row) * blocks_per_row + block;
                const uint32_t key = expert * 617u + row * 73u + block * 991u;
                for (uint32_t group = 0; group < QK_K / 16u; group++) {
                    const uint8_t scale =
                        (uint8_t)(1u + (key + 3u * group) % 7u);
                    const uint8_t min =
                        (uint8_t)((key / 5u + group) % 4u);
                    b->scales[group] =
                        (uint8_t)(scale | (uint8_t)(min << 4u));
                }
                for (uint32_t i = 0; i < QK_K / 4u; i++) {
                    b->qs[i] = (uint8_t)(key + 29u * i + (i >> 1u) * 7u);
                }
                b->d = 0x1800u;
                b->dmin = 0x1400u;
            }
        }
    }
}

static uint64_t touch_model_pages(const void *model, uint64_t bytes,
                                  uint64_t page) {
    const volatile uint8_t *data = model;
    uint64_t checksum = 0xcbf29ce484222325ull;
    for (uint64_t offset = 0; offset < bytes; offset += page) {
        checksum ^= data[offset];
        checksum *= 0x100000001b3ull;
    }
    checksum ^= data[bytes - 1u];
    return checksum;
}

/* Every 32-row tail size 1..31 is represented at production N=4096/top-6. */
static int build_routes(int32_t *selected, float *weights) {
    static const uint8_t final_remainders[8] = {1, 2, 3, 4, 5, 6, 7, 4};
    uint32_t target[N_TOTAL_EXPERT];
    uint32_t remaining[N_TOTAL_EXPERT];
    uint32_t actual[N_TOTAL_EXPERT] = {0};
    uint64_t target_sum = 0;

    for (uint32_t expert = 0; expert < N_TOTAL_EXPERT; expert++) {
        const uint32_t tail = expert < 248u ?
            1u + (expert * 17u) % 31u : final_remainders[expert - 248u];
        const uint32_t full_tiles =
            ((expert * 73u) & 255u) < 131u ? 3u : 2u;
        target[expert] = full_tiles * 32u + tail;
        remaining[expert] = target[expert];
        target_sum += target[expert];
    }
    if (target_sum != (uint64_t)N_TOKENS * N_EXPERT) {
        fprintf(stderr, "gpu-iq2-moe-prefill-bench: route target sum=%llu\n",
                (unsigned long long)target_sum);
        return 0;
    }

    for (uint32_t token = 0; token < N_TOKENS; token++) {
        uint8_t used[N_TOTAL_EXPERT] = {0};
        float raw_weight[N_EXPERT];
        float weight_sum = 0.0f;
        for (uint32_t slot = 0; slot < N_EXPERT; slot++) {
            uint32_t best = UINT32_MAX;
            uint32_t best_remaining = 0;
            uint32_t best_hash = 0;
            for (uint32_t expert = 0; expert < N_TOTAL_EXPERT; expert++) {
                if (used[expert] || remaining[expert] == 0u) continue;
                const uint32_t hash = mix32(
                    token * 0x9e3779b9u ^ slot * 0x85ebca6bu ^
                    expert * 0xc2b2ae35u);
                if (best == UINT32_MAX || remaining[expert] > best_remaining ||
                    (remaining[expert] == best_remaining && hash > best_hash)) {
                    best = expert;
                    best_remaining = remaining[expert];
                    best_hash = hash;
                }
            }
            if (best == UINT32_MAX) {
                fprintf(stderr,
                        "gpu-iq2-moe-prefill-bench: route scheduler exhausted "
                        "at token=%u slot=%u\n", token, slot);
                return 0;
            }
            const uint64_t route = (uint64_t)token * N_EXPERT + slot;
            selected[route] = (int32_t)best;
            used[best] = 1u;
            remaining[best]--;
            actual[best]++;
            raw_weight[slot] = 1.0f + (float)(mix32(
                token * 0x27d4eb2du ^ slot * 0x165667b1u) % 17u);
            weight_sum += raw_weight[slot];
        }
        for (uint32_t slot = 0; slot < N_EXPERT; slot++) {
            weights[(uint64_t)token * N_EXPERT + slot] =
                raw_weight[slot] / weight_sum;
        }
    }

    bool tails_seen[32] = {false};
    uint32_t min_count = UINT32_MAX;
    uint32_t max_count = 0;
    for (uint32_t expert = 0; expert < N_TOTAL_EXPERT; expert++) {
        if (remaining[expert] != 0u || actual[expert] != target[expert]) {
            fprintf(stderr,
                    "gpu-iq2-moe-prefill-bench: expert=%u target=%u "
                    "actual=%u remaining=%u\n",
                    expert, target[expert], actual[expert], remaining[expert]);
            return 0;
        }
        tails_seen[actual[expert] & 31u] = true;
        if (actual[expert] < min_count) min_count = actual[expert];
        if (actual[expert] > max_count) max_count = actual[expert];
    }
    for (uint32_t tail = 1; tail < 32u; tail++) {
        if (!tails_seen[tail]) {
            fprintf(stderr,
                    "gpu-iq2-moe-prefill-bench: missing tail=%u\n", tail);
            return 0;
        }
    }
    if (tails_seen[0]) {
        fprintf(stderr,
                "gpu-iq2-moe-prefill-bench: unexpected full-only expert\n");
        return 0;
    }
    fprintf(stderr,
            "DS4_GPU_IQ2_MOE_PREFILL_SETUP backend=%s routes=%u experts=%u "
            "topk=%u count_min=%u count_max=%u tail_coverage=1..31 "
            "unique_per_token=yes\n",
            BENCH_BACKEND, N_TOKENS * N_EXPERT, N_TOTAL_EXPERT, N_EXPERT,
            min_count, max_count);
    return 1;
}

static void fill_input(float *x) {
    for (uint32_t token = 0; token < N_TOKENS; token++) {
        for (uint32_t column = 0; column < IN_DIM; column++) {
            const uint32_t bits = mix32(
                token * 0x9e3779b9u ^ column * 0x85ebca6bu);
            const int32_t centered = (int32_t)(bits & 511u) - 256;
            x[(uint64_t)token * IN_DIM + column] =
                (float)centered / 1024.0f;
        }
    }
}

static void make_guard(uint32_t guard[GUARD_WORDS]) {
    for (uint32_t i = 0; i < GUARD_WORDS; i++) {
        guard[i] = GUARD_BITS ^ (i * 0x9e3779b9u);
    }
}

static int write_guard(ds4_gpu_tensor *tensor, uint64_t offset) {
    uint32_t guard[GUARD_WORDS];
    make_guard(guard);
    return ds4_gpu_tensor_write(tensor, offset, guard, sizeof(guard));
}

static int check_guard(const char *name, const ds4_gpu_tensor *tensor,
                       uint64_t offset) {
    uint32_t expected[GUARD_WORDS];
    uint32_t actual[GUARD_WORDS];
    make_guard(expected);
    if (!ds4_gpu_tensor_read(tensor, offset, actual, sizeof(actual))) {
        fprintf(stderr,
                "DS4_GPU_IQ2_MOE_PREFILL_CANARY backend=%s name=%s "
                "result=READ_FAIL\n", BENCH_BACKEND, name);
        return 0;
    }
    for (uint32_t i = 0; i < GUARD_WORDS; i++) {
        if (actual[i] != expected[i]) {
            fprintf(stderr,
                    "DS4_GPU_IQ2_MOE_PREFILL_CANARY backend=%s name=%s "
                    "result=FAIL word=%u expected=0x%08x actual=0x%08x\n",
                    BENCH_BACKEND, name, i, expected[i], actual[i]);
            return 0;
        }
    }
    return 1;
}

static int poison_outputs(fixture *f) {
    int ok = ds4_gpu_tensor_fill_f32(
        f->gate, -101.0f, (f->pair_f32_bytes + GUARD_BYTES) / sizeof(float));
    ok = ds4_gpu_tensor_fill_f32(
        f->up, -102.0f,
        (f->pair_f32_bytes + GUARD_BYTES) / sizeof(float)) && ok;
    ok = ds4_gpu_tensor_fill_f32(
        f->mid, -103.0f,
        (f->pair_f32_bytes + GUARD_BYTES) / sizeof(float)) && ok;
    ok = ds4_gpu_tensor_fill_f32(
        f->experts, -104.0f,
        (f->expert_bytes + GUARD_BYTES) / sizeof(float)) && ok;
    ok = ds4_gpu_tensor_fill_f32(
        f->out, -105.0f,
        (f->out_bytes + GUARD_BYTES) / sizeof(float)) && ok;
    ok = write_guard(f->gate, f->pair_f32_bytes) && ok;
    ok = write_guard(f->up, f->pair_f32_bytes) && ok;
    ok = write_guard(f->mid, f->pair_f32_bytes) && ok;
    ok = write_guard(f->experts, f->expert_bytes) && ok;
    ok = write_guard(f->out, f->out_bytes) && ok;
    return ok;
}

static int check_all_canaries(const fixture *f) {
    int ok = check_guard("x", f->x, f->x_bytes);
    ok = check_guard("selected", f->selected, f->route_i32_bytes) && ok;
    ok = check_guard("weights", f->weights, f->route_f32_bytes) && ok;
    ok = check_guard("gate", f->gate, f->pair_f32_bytes) && ok;
    ok = check_guard("up", f->up, f->pair_f32_bytes) && ok;
    ok = check_guard("mid-f32", f->mid, f->pair_f32_bytes) && ok;
    ok = check_guard("experts-f32", f->experts, f->expert_bytes) && ok;
    ok = check_guard("out-f32", f->out, f->out_bytes) && ok;
    return ok;
}

static const char *variant_name(bench_arm arm) {
    switch (arm) {
        case ARM_BASELINE: return "baseline";
        case ARM_CANDIDATE: return "candidate";
        case ARM_CURRENT: return "current";
    }
    return "invalid";
}

static int select_variant(bench_arm arm) {
#if defined(DS4_BENCH_ROCM)
    if (arm != ARM_BASELINE && arm != ARM_CANDIDATE) return 0;
    if (unsetenv(TAIL_ENABLE_ENV) != 0 || unsetenv(TAIL_DISABLE_ENV) != 0) {
        return 0;
    }
    return setenv(arm == ARM_BASELINE ? TAIL_DISABLE_ENV : TAIL_ENABLE_ENV,
                  "1", 1) == 0;
#else
    return arm == ARM_CURRENT;
#endif
}

static int run_once(fixture *f, bench_arm arm,
                    const char *phase, const char *order,
                    uint32_t sample, uint32_t cycle, uint32_t position,
                    bool poison, bool check_canaries) {
    if (!select_variant(arm)) {
        fprintf(stderr,
                "gpu-iq2-moe-prefill-bench: backend policy setup failed\n");
        return 0;
    }
    if (poison && !poison_outputs(f)) {
        fprintf(stderr, "gpu-iq2-moe-prefill-bench: output poison failed\n");
        return 0;
    }

    fprintf(stderr,
            "DS4_GPU_IQ2_MOE_PREFILL_BENCH backend=%s phase=%s "
            "variant=%s sample=%u cycle=%u position=%u order=%s "
            "force_resident=1 ssd_streaming=0\n",
            BENCH_BACKEND, phase, variant_name(arm), sample,
            cycle, position, order);
    fflush(stderr);

    if (!ds4_gpu_begin_commands()) {
        fprintf(stderr, "gpu-iq2-moe-prefill-bench: begin commands failed\n");
        return 0;
    }
#if defined(DS4_BENCH_CUDA)
    const uint64_t profile_reports_before =
        ds4_cuda_test_moe_fast_profile_report_count();
#endif
    bool mid_is_f16 = false;
    const int call_ok = ds4_gpu_routed_moe_batch_tensor(
        f->out, f->gate, f->up, f->mid, f->experts,
        f->model, f->model_size,
        f->gate_offset, f->up_offset, f->down_offset,
        IQ2_XXS_TYPE, Q2_K_TYPE,
        f->gate_expert_bytes, f->gate_row_bytes,
        f->down_expert_bytes, f->down_row_bytes,
        IN_DIM, MID_DIM, OUT_DIM,
        f->selected, f->weights, N_TOTAL_EXPERT, N_EXPERT, CLAMP, f->x,
        0u, N_TOKENS, &mid_is_f16, true);
    const int end_ok = ds4_gpu_end_commands();
#if defined(DS4_BENCH_CUDA)
    const uint64_t profile_reports_after =
        ds4_cuda_test_moe_fast_profile_report_count();
    const int profile_ok =
        profile_reports_after == profile_reports_before + 1u;
    if (!profile_ok) {
        fprintf(stderr,
                "DS4_GPU_IQ2_MOE_PREFILL_BENCH backend=cuda result=FAIL "
                "variant=%s fast_profile_reports_before=%llu "
                "fast_profile_reports_after=%llu expected_delta=1\n",
                variant_name(arm),
                (unsigned long long)profile_reports_before,
                (unsigned long long)profile_reports_after);
    }
#else
    const int profile_ok = 1;
#endif
    int ok = call_ok && end_ok && profile_ok;
    if (!ok) {
        fprintf(stderr,
                "DS4_GPU_IQ2_MOE_PREFILL_BENCH backend=%s result=FAIL "
                "variant=%s call=%d end=%d reported_mid_format=%s\n",
                BENCH_BACKEND, variant_name(arm), call_ok, end_ok,
                mid_is_f16 ? "f16" : "f32-or-opaque");
    }
    if (check_canaries) ok = check_all_canaries(f) && ok;
    return ok;
}

#if defined(DS4_BENCH_ROCM)
static int snapshot_alloc(oracle_snapshot *snapshot, const fixture *f) {
    memset(snapshot, 0, sizeof(*snapshot));
    snapshot->bytes = 3u * f->pair_f32_bytes +
                      f->expert_bytes + f->out_bytes;
    if (snapshot->bytes > SIZE_MAX) return 0;
    snapshot->storage = malloc((size_t)snapshot->bytes);
    if (!snapshot->storage) return 0;
    snapshot->gate = snapshot->storage;
    snapshot->up = snapshot->gate + f->pair_f32_bytes;
    snapshot->mid = snapshot->up + f->pair_f32_bytes;
    snapshot->experts = snapshot->mid + f->pair_f32_bytes;
    snapshot->out = snapshot->experts + f->expert_bytes;
    return 1;
}

static int capture_snapshot(oracle_snapshot *snapshot, const fixture *f) {
    return ds4_gpu_tensor_read(
               f->gate, 0, snapshot->gate, f->pair_f32_bytes) &&
           ds4_gpu_tensor_read(
               f->up, 0, snapshot->up, f->pair_f32_bytes) &&
           ds4_gpu_tensor_read(
               f->mid, 0, snapshot->mid, f->pair_f32_bytes) &&
           ds4_gpu_tensor_read(
               f->experts, 0, snapshot->experts, f->expert_bytes) &&
           ds4_gpu_tensor_read(
               f->out, 0, snapshot->out, f->out_bytes);
}

static int tensor_matches(const char *name, const ds4_gpu_tensor *tensor,
                          const uint8_t *expected, uint64_t bytes,
                          uint8_t *scratch) {
    uint64_t offset = 0;
    while (offset < bytes) {
        const size_t chunk = bytes - offset > IO_CHUNK_BYTES ?
            IO_CHUNK_BYTES : (size_t)(bytes - offset);
        if (!ds4_gpu_tensor_read(tensor, offset, scratch, chunk)) {
            fprintf(stderr,
                    "DS4_GPU_IQ2_MOE_PREFILL_ORACLE backend=rocm "
                    "candidate=tail-cull tensor=%s result=READ_FAIL "
                    "offset=%llu\n", name, (unsigned long long)offset);
            return 0;
        }
        if (memcmp(scratch, expected + offset, chunk) != 0) {
            size_t mismatch = 0;
            while (mismatch < chunk &&
                   scratch[mismatch] == expected[offset + mismatch]) {
                mismatch++;
            }
            fprintf(stderr,
                    "DS4_GPU_IQ2_MOE_PREFILL_ORACLE backend=rocm "
                    "candidate=tail-cull tensor=%s result=MISMATCH byte=%llu "
                    "expected=0x%02x actual=0x%02x\n",
                    name, (unsigned long long)(offset + mismatch),
                    expected[offset + mismatch], scratch[mismatch]);
            return 0;
        }
        offset += chunk;
    }
    return 1;
}

static int run_correctness(fixture *f) {
    oracle_snapshot baseline;
    if (!snapshot_alloc(&baseline, f)) {
        fprintf(stderr,
                "gpu-iq2-moe-prefill-bench: oracle snapshot allocation failed\n");
        return 0;
    }
    uint8_t *scratch = malloc(IO_CHUNK_BYTES);
    int ok = scratch != NULL;
    if (ok) {
        ok = run_once(f, ARM_BASELINE, "oracle", "BASELINE",
                      0u, 0u, 0u, true, true);
    }
    if (ok) ok = capture_snapshot(&baseline, f);
    if (ok) {
        ok = run_once(f, ARM_CANDIDATE, "oracle", "CANDIDATE",
                      0u, 0u, 0u, true, true);
    }
    const int gate_ok = ok && tensor_matches(
        "gate_scratch", f->gate, baseline.gate, f->pair_f32_bytes, scratch);
    const int up_ok = ok && tensor_matches(
        "up_scratch", f->up, baseline.up, f->pair_f32_bytes, scratch);
    const int mid_ok = ok && tensor_matches(
        "mid_scratch", f->mid, baseline.mid, f->pair_f32_bytes, scratch);
    const int experts_ok = ok && tensor_matches(
        "down_scratch", f->experts, baseline.experts, f->expert_bytes, scratch);
    const int out_ok = ok && tensor_matches(
        "out_f32", f->out, baseline.out, f->out_bytes, scratch);
    ok = ok && gate_ok && up_ok && mid_ok && experts_ok && out_ok;
    fprintf(stderr,
            "DS4_GPU_IQ2_MOE_PREFILL_ORACLE backend=rocm "
            "candidate=tail-cull result=%s gate_scratch=%s up_scratch=%s "
            "mid_scratch=%s down_scratch=%s out_f32=%s canaries=%s\n",
            ok ? "PASS" : "FAIL",
            gate_ok ? "exact" : "mismatch",
            up_ok ? "exact" : "mismatch",
            mid_ok ? "exact" : "mismatch",
            experts_ok ? "exact" : "mismatch",
            out_ok ? "exact" : "mismatch",
            ok ? "PASS" : "FAIL");
    free(scratch);
    free(baseline.storage);
    return ok;
}

static int run_balanced_block(fixture *f, const char *phase,
                              uint32_t samples_per_arm) {
    uint32_t arm_samples[2] = {0, 0};
    const uint32_t cycles = samples_per_arm / 2u;
    for (uint32_t cycle = 0; cycle < cycles; cycle++) {
        static const bench_arm abba[4] = {
            ARM_BASELINE, ARM_CANDIDATE, ARM_CANDIDATE, ARM_BASELINE,
        };
        static const bench_arm baab[4] = {
            ARM_CANDIDATE, ARM_BASELINE, ARM_BASELINE, ARM_CANDIDATE,
        };
        const bench_arm *order = (cycle & 1u) ? baab : abba;
        const char *order_name = (cycle & 1u) ? "BAAB" : "ABBA";
        for (uint32_t position = 0; position < 4u; position++) {
            const bench_arm arm = order[position];
            if (!run_once(f, arm, phase, order_name,
                          arm_samples[arm]++, cycle, position,
                          false, false)) {
                return 0;
            }
        }
    }
    return arm_samples[ARM_BASELINE] == samples_per_arm &&
           arm_samples[ARM_CANDIDATE] == samples_per_arm;
}

static int run_experiment(fixture *f, const bench_config *config) {
    if (config->warmups != 0u &&
        !run_balanced_block(f, "warmup", config->warmups)) {
        return 0;
    }
    if (!run_balanced_block(f, "sample", config->samples)) return 0;
    fprintf(stderr,
            "DS4_GPU_IQ2_MOE_PREFILL_BENCH backend=rocm phase=complete "
            "mode=real-ab samples_per_variant=%u warmups_per_variant=%u "
            "result=PASS\n", config->samples, config->warmups);
    return 1;
}
#else
static int validate_current_tensor(const char *name,
                                   const ds4_gpu_tensor *tensor,
                                   uint64_t bytes, float poison,
                                   bool dense_f32) {
    uint8_t *scratch = malloc(IO_CHUNK_BYTES);
    if (!scratch) return 0;
    uint64_t hash = 0xcbf29ce484222325ull;
    uint64_t changed = 0;
    uint64_t unchanged = 0;
    uint64_t offset = 0;
    int ok = 1;
    uint8_t poison_bytes[sizeof(float)];
    memcpy(poison_bytes, &poison, sizeof(poison_bytes));
    while (offset < bytes && ok) {
        const size_t chunk = bytes - offset > IO_CHUNK_BYTES ?
            IO_CHUNK_BYTES : (size_t)(bytes - offset);
        if (!ds4_gpu_tensor_read(tensor, offset, scratch, chunk)) {
            ok = 0;
            break;
        }
        if (dense_f32) {
            const float *values = (const float *)scratch;
            const size_t count = chunk / sizeof(float);
            for (size_t i = 0; i < count; i++) {
                if (!isfinite(values[i])) {
                    fprintf(stderr,
                            "DS4_GPU_IQ2_MOE_PREFILL_ORACLE backend=cuda "
                            "mode=current-single-run tensor=%s "
                            "result=NONFINITE element=%llu\n", name,
                            (unsigned long long)(offset / sizeof(float) + i));
                    ok = 0;
                    break;
                }
                if (values[i] != poison) {
                    changed++;
                } else {
                    unchanged++;
                }
            }
        } else {
            for (size_t i = 0; i < chunk; i++) {
                const uint8_t expected =
                    poison_bytes[(size_t)((offset + i) % sizeof(float))];
                if (scratch[i] != expected) {
                    changed++;
                } else {
                    unchanged++;
                }
            }
        }
        for (size_t i = 0; i < chunk; i++) {
            hash ^= scratch[i];
            hash *= 0x100000001b3ull;
        }
        offset += chunk;
    }
    if (ok && (dense_f32 ? unchanged != 0u : changed == 0u)) ok = 0;
    fprintf(stderr,
            "DS4_GPU_IQ2_MOE_PREFILL_ORACLE backend=cuda "
            "mode=current-single-run scope=structural tensor=%s layout=%s "
            "result=%s "
            "written=%llu unchanged_poison=%llu units=%s hash=0x%016llx\n",
            name, dense_f32 ? "dense_f32" : "opaque_scratch",
            ok ? "PASS" : "FAIL",
            (unsigned long long)changed, (unsigned long long)unchanged,
            dense_f32 ? "elements" : "bytes",
            (unsigned long long)hash);
    free(scratch);
    return ok;
}

static int run_correctness(fixture *f) {
    int ok = run_once(f, ARM_CURRENT, "oracle", "CURRENT",
                      0u, 0u, 0u, true, true);
    const int gate_ok = ok && validate_current_tensor(
        "gate_scratch", f->gate, f->pair_f32_bytes, -101.0f, false);
    const int up_ok = ok && validate_current_tensor(
        "up_scratch", f->up, f->pair_f32_bytes, -102.0f, false);
    const int mid_ok = ok && validate_current_tensor(
        "mid_scratch", f->mid, f->pair_f32_bytes, -103.0f, false);
    const int experts_ok = ok && validate_current_tensor(
        "down_f32", f->experts, f->expert_bytes, -104.0f, true);
    const int out_ok = ok && validate_current_tensor(
        "out_f32", f->out, f->out_bytes, -105.0f, true);
    ok = ok && gate_ok && up_ok && mid_ok && experts_ok && out_ok;
    fprintf(stderr,
            "DS4_GPU_IQ2_MOE_PREFILL_ORACLE backend=cuda "
            "mode=current-single-run scope=structural result=%s "
            "canaries=%s numerical_ab=NOT_AVAILABLE\n",
            ok ? "PASS" : "FAIL", ok ? "PASS" : "FAIL");
    return ok;
}

static int run_experiment(fixture *f, const bench_config *config) {
    for (uint32_t i = 0; i < config->warmups; i++) {
        if (!run_once(f, ARM_CURRENT, "warmup", "CURRENT",
                      i, 0u, 0u, false, false)) {
            return 0;
        }
    }
    for (uint32_t i = 0; i < config->samples; i++) {
        if (!run_once(f, ARM_CURRENT, "sample", "CURRENT",
                      i, 0u, 0u, false, false)) {
            return 0;
        }
    }
    fprintf(stderr,
            "DS4_GPU_IQ2_MOE_PREFILL_BENCH backend=cuda phase=complete "
            "mode=measurement-only variant=current samples=%u warmups=%u "
            "profiler=%s numerical_ab=NOT_AVAILABLE result=PASS\n",
            config->samples, config->warmups, CUDA_PROFILE_ENV);
    return 1;
}
#endif

static void free_tensors(fixture *f) {
    ds4_gpu_tensor_free(f->out);
    ds4_gpu_tensor_free(f->experts);
    ds4_gpu_tensor_free(f->mid);
    ds4_gpu_tensor_free(f->up);
    ds4_gpu_tensor_free(f->gate);
    ds4_gpu_tensor_free(f->weights);
    ds4_gpu_tensor_free(f->selected);
    ds4_gpu_tensor_free(f->x);
    f->out = NULL;
    f->experts = NULL;
    f->mid = NULL;
    f->up = NULL;
    f->gate = NULL;
    f->weights = NULL;
    f->selected = NULL;
    f->x = NULL;
}

static int init_fixture(fixture *f) {
    memset(f, 0, sizeof(*f));
    const long page_long = sysconf(_SC_PAGESIZE);
    if (page_long <= 0) {
        fprintf(stderr, "gpu-iq2-moe-prefill-bench: page size unavailable\n");
        return 0;
    }
    const uint64_t page = (uint64_t)page_long;
    f->gate_row_bytes =
        (uint64_t)(IN_DIM / QK_K) * sizeof(block_iq2_xxs);
    f->gate_expert_bytes = (uint64_t)MID_DIM * f->gate_row_bytes;
    const uint64_t gate_tensor_bytes =
        (uint64_t)N_TOTAL_EXPERT * f->gate_expert_bytes;
    f->down_row_bytes =
        (uint64_t)(MID_DIM / QK_K) * sizeof(block_q2_K);
    f->down_expert_bytes = (uint64_t)OUT_DIM * f->down_row_bytes;
    const uint64_t down_tensor_bytes =
        (uint64_t)N_TOTAL_EXPERT * f->down_expert_bytes;
    f->gate_offset = 0;
    f->up_offset = align_up(gate_tensor_bytes, page);
    f->down_offset = align_up(f->up_offset + gate_tensor_bytes, page);
    f->model_size = align_up(f->down_offset + down_tensor_bytes, page);

    f->x_bytes = (uint64_t)N_TOKENS * IN_DIM * sizeof(float);
    f->route_count = (uint64_t)N_TOKENS * N_EXPERT;
    f->route_i32_bytes = f->route_count * sizeof(int32_t);
    f->route_f32_bytes = f->route_count * sizeof(float);
    f->pair_count = f->route_count * MID_DIM;
    f->pair_f32_bytes = f->pair_count * sizeof(float);
    f->expert_count = (uint64_t)N_TOKENS * N_EXPERT * OUT_DIM;
    f->expert_bytes = f->expert_count * sizeof(float);
    f->out_count = (uint64_t)N_TOKENS * OUT_DIM;
    f->out_bytes = f->out_count * sizeof(float);

    if (f->gate_row_bytes != 1056u ||
        f->gate_expert_bytes != 2162688u ||
        f->down_row_bytes != 672u ||
        f->down_expert_bytes != 2752512u) {
        fprintf(stderr,
                "gpu-iq2-moe-prefill-bench: production layout mismatch "
                "gate_row=%llu gate_expert=%llu down_row=%llu "
                "down_expert=%llu\n",
                (unsigned long long)f->gate_row_bytes,
                (unsigned long long)f->gate_expert_bytes,
                (unsigned long long)f->down_row_bytes,
                (unsigned long long)f->down_expert_bytes);
        return 0;
    }

    const uint64_t tensor_bytes =
        f->x_bytes + f->route_i32_bytes + f->route_f32_bytes +
        3u * (f->pair_f32_bytes + GUARD_BYTES) +
        f->expert_bytes + GUARD_BYTES + f->out_bytes + GUARD_BYTES +
        3u * GUARD_BYTES;
#if defined(DS4_BENCH_ROCM)
    const uint64_t oracle_bytes =
        3u * f->pair_f32_bytes + f->expert_bytes +
        f->out_bytes + IO_CHUNK_BYTES;
#else
    const uint64_t oracle_bytes = IO_CHUNK_BYTES;
#endif
    const uint64_t setup_host_bytes =
        f->x_bytes + f->route_i32_bytes + f->route_f32_bytes;
    const uint64_t explicit_peak =
        f->model_size + tensor_bytes + oracle_bytes + setup_host_bytes;
    fprintf(stderr,
            "DS4_GPU_IQ2_MOE_PREFILL_SETUP backend=%s "
            "geometry=N%u,d%u,mid%u,out%u,experts%u,top%u "
            "model=%.3f_GiB tensors=%.3f_GiB oracle=%.3f_GiB "
            "explicit_peak=%.3f_GiB resident=1\n",
            BENCH_BACKEND, N_TOKENS, IN_DIM, MID_DIM, OUT_DIM,
            N_TOTAL_EXPERT, N_EXPERT,
            (double)f->model_size / (double)GIB,
            (double)tensor_bytes / (double)GIB,
            (double)oracle_bytes / (double)GIB,
            (double)explicit_peak / (double)GIB);
    if (explicit_peak >= 5u * GIB) {
        fprintf(stderr,
                "gpu-iq2-moe-prefill-bench: explicit peak exceeds 5 GiB\n");
        return 0;
    }

    if (posix_memalign(&f->model, (size_t)page,
                       (size_t)f->model_size) != 0) {
        fprintf(stderr,
                "gpu-iq2-moe-prefill-bench: model allocation failed\n");
        return 0;
    }
    fprintf(stderr,
            "DS4_GPU_IQ2_MOE_PREFILL_SETUP backend=%s phase=fill_weights "
            "tensor=gate\n", BENCH_BACKEND);
    fill_iq2((block_iq2_xxs *)((uint8_t *)f->model + f->gate_offset), 19u);
    fprintf(stderr,
            "DS4_GPU_IQ2_MOE_PREFILL_SETUP backend=%s phase=fill_weights "
            "tensor=up\n", BENCH_BACKEND);
    fill_iq2((block_iq2_xxs *)((uint8_t *)f->model + f->up_offset), 47u);
    fprintf(stderr,
            "DS4_GPU_IQ2_MOE_PREFILL_SETUP backend=%s phase=fill_weights "
            "tensor=down\n", BENCH_BACKEND);
    fill_q2((block_q2_K *)((uint8_t *)f->model + f->down_offset));
    const uint64_t checksum = touch_model_pages(f->model, f->model_size, page);
    fprintf(stderr,
            "DS4_GPU_IQ2_MOE_PREFILL_SETUP backend=%s phase=touch_weights "
            "pages=%llu checksum=0x%016llx resident=1\n",
            BENCH_BACKEND,
            (unsigned long long)((f->model_size + page - 1u) / page),
            (unsigned long long)checksum);

    float *x_host = malloc((size_t)f->x_bytes);
    int32_t *selected_host = malloc((size_t)f->route_i32_bytes);
    float *weights_host = malloc((size_t)f->route_f32_bytes);
    int ok = x_host && selected_host && weights_host;
    if (ok) fill_input(x_host);
    if (ok) ok = build_routes(selected_host, weights_host);
    if (ok) ok = ds4_gpu_set_model_map(f->model, f->model_size);

    if (ok) f->x = ds4_gpu_tensor_alloc(f->x_bytes + GUARD_BYTES);
    if (ok) f->selected =
        ds4_gpu_tensor_alloc(f->route_i32_bytes + GUARD_BYTES);
    if (ok) f->weights =
        ds4_gpu_tensor_alloc(f->route_f32_bytes + GUARD_BYTES);
    if (ok) f->gate =
        ds4_gpu_tensor_alloc(f->pair_f32_bytes + GUARD_BYTES);
    if (ok) f->up =
        ds4_gpu_tensor_alloc(f->pair_f32_bytes + GUARD_BYTES);
    if (ok) f->mid =
        ds4_gpu_tensor_alloc(f->pair_f32_bytes + GUARD_BYTES);
    if (ok) f->experts =
        ds4_gpu_tensor_alloc(f->expert_bytes + GUARD_BYTES);
    if (ok) f->out = ds4_gpu_tensor_alloc(f->out_bytes + GUARD_BYTES);
    ok = ok && f->x && f->selected && f->weights && f->gate && f->up &&
         f->mid && f->experts && f->out;

    if (ok) ok = ds4_gpu_tensor_write(f->x, 0, x_host, f->x_bytes);
    if (ok) ok = ds4_gpu_tensor_write(
        f->selected, 0, selected_host, f->route_i32_bytes);
    if (ok) ok = ds4_gpu_tensor_write(
        f->weights, 0, weights_host, f->route_f32_bytes);
    if (ok) ok = write_guard(f->x, f->x_bytes);
    if (ok) ok = write_guard(f->selected, f->route_i32_bytes);
    if (ok) ok = write_guard(f->weights, f->route_f32_bytes);
    if (ok) ok = poison_outputs(f);

    free(weights_host);
    free(selected_host);
    free(x_host);
    if (!ok) {
        fprintf(stderr,
                "gpu-iq2-moe-prefill-bench: fixture initialization failed\n");
    }
    return ok;
}

static int configure_backend(void) {
#if defined(DS4_BENCH_ROCM)
    return unsetenv(TAIL_ENABLE_ENV) == 0 &&
           unsetenv(TAIL_DISABLE_ENV) == 0 &&
           setenv(ROCM_PROFILE_ENV, "1", 1) == 0;
#else
    /* Presence, including value zero, enables the existing CUDA profiler. */
    return setenv(CUDA_PROFILE_ENV, "1", 1) == 0;
#endif
}

int main(int argc, char **argv) {
    const bench_config config = parse_options(argc, argv);
    if (!configure_backend()) {
        fprintf(stderr,
                "gpu-iq2-moe-prefill-bench: backend environment setup failed\n");
        return 1;
    }

    if (!ds4_gpu_init()) {
        fprintf(stderr,
                "gpu-iq2-moe-prefill-bench: %s initialization failed\n",
                BENCH_BACKEND);
        /* Both GPU backends make cleanup idempotent for partially initialized
         * state; do not strand a stream, handle, or allocation on init error. */
        ds4_gpu_cleanup();
        return 1;
    }
    ds4_gpu_set_quality(false);
    ds4_gpu_set_ssd_streaming(false);

    fixture f;
    const int fixture_ready = init_fixture(&f);
    int ok = fixture_ready;
    if (ok) ok = run_correctness(&f);
    if (ok) ok = run_experiment(&f, &config);
    /* A failing dispatch must not suppress the final overrun check. */
    if (fixture_ready) {
        const int canaries_ok = check_all_canaries(&f);
        ok = canaries_ok && ok;
    }

#if defined(DS4_BENCH_ROCM)
    (void)unsetenv(TAIL_ENABLE_ENV);
    (void)unsetenv(TAIL_DISABLE_ENV);
    (void)unsetenv(ROCM_PROFILE_ENV);
#endif
    free_tensors(&f);
    ds4_gpu_cleanup();
    free(f.model);
    fprintf(stderr,
            "DS4_GPU_IQ2_MOE_PREFILL_BENCH backend=%s result=%s\n",
            BENCH_BACKEND, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
