#define _DARWIN_C_SOURCE

#include "ds4_gpu.h"

#include <errno.h>
#include <limits.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define IQ2_XXS_TYPE 16u
#define Q2_K_TYPE 10u
#define QK_K 256u
#define IN_DIM 4096u
#define MID_DIM 2048u
#define OUT_DIM 4096u
#define N_TOKENS 4096u
#ifdef DS4_METAL_IQ2_MOE_TOP8_PAIR_BENCH
#define N_TOTAL_EXPERT 288u
#define N_EXPERT 8u
#define CLAMP 0.0f
#define BENCH_DESCRIPTION \
    "Resident GLM-geometry IQ2_XXS routed-MoE top-8 pair-fusion benchmark."
#define EXPERIMENT_NAME "top8-pair-fusion"
#else
#define N_TOTAL_EXPERT 256u
#define N_EXPERT 6u
#define CLAMP 4.0f
#define BENCH_DESCRIPTION \
    "Resident production-geometry IQ2_XXS pair routed-MoE tail-cull benchmark."
#define EXPERIMENT_NAME "pair-tail-cull"
#endif
#define GUARD_WORDS 64u
#define GUARD_BYTES ((uint64_t)GUARD_WORDS * sizeof(uint32_t))
#define GUARD_BITS 0x51a7c3e9u
#define DEFAULT_SAMPLES 8u
#define DEFAULT_WARMUP_CYCLES 1u
#define COMPARE_CHUNK_BYTES (8u * 1024u * 1024u)
#define GIB (1024ull * 1024ull * 1024ull)

#define PAIR_TAIL_ENABLE_ENV \
    "DS4_METAL_ENABLE_IQ2_XXS_MOE_MM_ID_PAIR_TAIL_SIMDGROUP_CULL"
#define PAIR_TAIL_DISABLE_ENV \
    "DS4_METAL_DISABLE_IQ2_XXS_MOE_MM_ID_PAIR_TAIL_SIMDGROUP_CULL"
#define PAIR_FUSION_DISABLE_ENV \
    "DS4_METAL_DISABLE_MOE_MM_ID_PAIR_SWIGLU"

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
    uint32_t warmup_cycles;
} bench_config;

typedef enum {
    ARM_BASELINE,
    ARM_CANDIDATE,
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
    uint64_t pair_f16_bytes;
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

typedef struct {
    uint8_t *storage;
    uint8_t *mid;
    uint8_t *experts;
    uint8_t *out;
    uint64_t bytes;
} oracle_snapshot;

bool ds4_log_is_tty(FILE *fp) {
    (void)fp;
    return false;
}

static void usage(FILE *fp, const char *argv0) {
    fprintf(fp,
            "usage: %s [options]\n"
            "\n"
            BENCH_DESCRIPTION "\n"
            "Metal stage profiling prints the kernel GPU timestamps; marker "
            "lines identify each arm.\n"
            "\n"
            "  --samples N         samples per arm, even (default: %u)\n"
            "  --warmup-cycles N   four-run balanced warmup cycles "
            "(default: %u)\n"
            "  -h, --help          show this help\n",
            argv0, DEFAULT_SAMPLES, DEFAULT_WARMUP_CYCLES);
}

static uint32_t parse_u32(const char *text, const char *option,
                          uint32_t minimum) {
    char *end = NULL;
    errno = 0;
    const unsigned long long value = strtoull(text, &end, 10);
    if (errno != 0 || !text[0] || !end || *end ||
        value < minimum || value > UINT32_MAX) {
        fprintf(stderr,
                "metal-iq2-moe-tail-cull-bench: invalid %s: %s\n",
                option, text);
        exit(2);
    }
    return (uint32_t)value;
}

static const char *need_arg(int *index, int argc, char **argv) {
    if (*index + 1 >= argc) {
        fprintf(stderr,
                "metal-iq2-moe-tail-cull-bench: %s needs a value\n",
                argv[*index]);
        exit(2);
    }
    return argv[++*index];
}

static bench_config parse_options(int argc, char **argv) {
    bench_config config = {
        .samples = DEFAULT_SAMPLES,
        .warmup_cycles = DEFAULT_WARMUP_CYCLES,
    };
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
            usage(stdout, argv[0]);
            exit(0);
        } else if (!strcmp(argv[i], "--samples")) {
            const char *option = argv[i];
            const char *value = need_arg(&i, argc, argv);
            config.samples = parse_u32(value, option, 2u);
        } else if (!strcmp(argv[i], "--warmup-cycles")) {
            const char *option = argv[i];
            const char *value = need_arg(&i, argc, argv);
            config.warmup_cycles = parse_u32(value, option, 0u);
        } else {
            fprintf(stderr,
                    "metal-iq2-moe-tail-cull-bench: unknown option: %s\n",
                    argv[i]);
            usage(stderr, argv[0]);
            exit(2);
        }
    }
    if ((config.samples & 1u) != 0u) {
        fprintf(stderr,
                "metal-iq2-moe-tail-cull-bench: --samples must be even\n");
        exit(2);
    }
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
                    ((uint64_t)expert * MID_DIM + row) * blocks_per_row +
                    block;
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

static void fill_q2(block_q2_K *matrix) {
    const uint32_t blocks_per_row = MID_DIM / QK_K;
    for (uint32_t expert = 0; expert < N_TOTAL_EXPERT; expert++) {
        for (uint32_t row = 0; row < OUT_DIM; row++) {
            for (uint32_t block = 0; block < blocks_per_row; block++) {
                block_q2_K *b = matrix +
                    ((uint64_t)expert * OUT_DIM + row) * blocks_per_row +
                    block;
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

/* Construct non-uniform expert counts with every final tile size 1..31.
 * A per-token hash breaks ties in the remaining-count scheduler and keeps all
 * routed experts unique without changing the target counts. */
static int build_routes(int32_t *selected, float *weights) {
#ifndef DS4_METAL_IQ2_MOE_TOP8_PAIR_BENCH
    static const uint8_t final_remainders[8] = {1, 2, 3, 4, 5, 6, 7, 4};
#endif
    uint32_t target[N_TOTAL_EXPERT];
    uint32_t remaining[N_TOTAL_EXPERT];
    uint32_t actual[N_TOTAL_EXPERT] = {0};
    uint64_t target_sum = 0;

    for (uint32_t expert = 0; expert < N_TOTAL_EXPERT; expert++) {
#ifdef DS4_METAL_IQ2_MOE_TOP8_PAIR_BENCH
        const uint32_t tail = 1u + (expert * 17u) % 31u;
        target[expert] = 3u * 32u + tail;
#else
        const uint32_t tail = expert < 248u ?
            1u + (expert * 17u) % 31u :
            final_remainders[expert - 248u];
        const uint32_t full_tiles =
            ((expert * 73u) & 255u) < 131u ? 3u : 2u;
        target[expert] = full_tiles * 32u + tail;
#endif
        target_sum += target[expert];
    }
#ifdef DS4_METAL_IQ2_MOE_TOP8_PAIR_BENCH
    uint64_t deficit = (uint64_t)N_TOKENS * N_EXPERT - target_sum;
    while (deficit != 0u) {
        bool progressed = false;
        /* Preserve experts 0..30 as one complete permutation of tails. */
        for (uint32_t expert = 31u;
             expert < N_TOTAL_EXPERT && deficit != 0u;
             expert++) {
            if ((target[expert] & 31u) == 31u) continue;
            target[expert]++;
            target_sum++;
            deficit--;
            progressed = true;
        }
        if (!progressed) {
            fprintf(stderr,
                    "metal-iq2-moe-tail-cull-bench: route target "
                    "distribution exhausted\n");
            return 0;
        }
    }
#endif
    for (uint32_t expert = 0; expert < N_TOTAL_EXPERT; expert++) {
        remaining[expert] = target[expert];
    }
    if (target_sum != (uint64_t)N_TOKENS * N_EXPERT) {
        fprintf(stderr,
                "metal-iq2-moe-tail-cull-bench: route target sum=%llu\n",
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
                if (best == UINT32_MAX ||
                    remaining[expert] > best_remaining ||
                    (remaining[expert] == best_remaining &&
                     hash > best_hash)) {
                    best = expert;
                    best_remaining = remaining[expert];
                    best_hash = hash;
                }
            }
            if (best == UINT32_MAX) {
                fprintf(stderr,
                        "metal-iq2-moe-tail-cull-bench: route scheduler "
                        "exhausted at token=%u slot=%u\n",
                        token, slot);
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
                    "metal-iq2-moe-tail-cull-bench: expert=%u target=%u "
                    "actual=%u remaining=%u\n",
                    expert, target[expert], actual[expert],
                    remaining[expert]);
            return 0;
        }
        tails_seen[actual[expert] & 31u] = true;
        if (actual[expert] < min_count) min_count = actual[expert];
        if (actual[expert] > max_count) max_count = actual[expert];
    }
    for (uint32_t tail = 1; tail < 32u; tail++) {
        if (!tails_seen[tail]) {
            fprintf(stderr,
                    "metal-iq2-moe-tail-cull-bench: missing tail=%u\n",
                    tail);
            return 0;
        }
    }
    if (tails_seen[0]) {
        fprintf(stderr,
                "metal-iq2-moe-tail-cull-bench: unexpected full-only expert\n");
        return 0;
    }
    fprintf(stderr,
            "DS4_IQ2_MOE_TAIL_SETUP routes=%u experts=%u topk=%u "
            "count_min=%u count_max=%u tail_coverage=1..31 "
            "unique_per_token=yes\n",
            N_TOKENS * N_EXPERT, N_TOTAL_EXPERT, N_EXPERT,
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
                "DS4_IQ2_MOE_TAIL_CANARY name=%s result=READ_FAIL\n",
                name);
        return 0;
    }
    for (uint32_t i = 0; i < GUARD_WORDS; i++) {
        if (actual[i] != expected[i]) {
            fprintf(stderr,
                    "DS4_IQ2_MOE_TAIL_CANARY name=%s result=FAIL "
                    "word=%u expected=0x%08x actual=0x%08x\n",
                    name, i, expected[i], actual[i]);
            return 0;
        }
    }
    return 1;
}

static int poison_outputs(fixture *f) {
    int ok = ds4_gpu_tensor_fill_f32(
        f->gate, -101.0f, (f->pair_f32_bytes + GUARD_BYTES) / sizeof(float));
    ok = ds4_gpu_tensor_fill_f32(
        f->up, -102.0f, (f->pair_f32_bytes + GUARD_BYTES) / sizeof(float)) && ok;
    ok = ds4_gpu_tensor_fill_f32(
        f->mid, -103.0f, (f->pair_f32_bytes + GUARD_BYTES) / sizeof(float)) && ok;
    ok = ds4_gpu_tensor_fill_f32(
        f->experts, -104.0f,
        (f->expert_bytes + GUARD_BYTES) / sizeof(float)) && ok;
    ok = ds4_gpu_tensor_fill_f32(
        f->out, -105.0f,
        (f->out_bytes + GUARD_BYTES) / sizeof(float)) && ok;
    ok = write_guard(f->gate, f->pair_f32_bytes) && ok;
    ok = write_guard(f->up, f->pair_f32_bytes) && ok;
    /* The first guard proves the production F16 mid contract. Full F32
     * capacity remains allocated so a wrong fallback is caught safely. */
    ok = write_guard(f->mid, f->pair_f16_bytes) && ok;
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
    ok = check_guard("mid-f16-boundary", f->mid, f->pair_f16_bytes) && ok;
    ok = check_guard("mid-allocation-end", f->mid, f->pair_f32_bytes) && ok;
    ok = check_guard("experts", f->experts, f->expert_bytes) && ok;
    ok = check_guard("out", f->out, f->out_bytes) && ok;
    return ok;
}

static const char *variant_name(bench_arm arm) {
#ifdef DS4_METAL_IQ2_MOE_TOP8_PAIR_BENCH
    return arm == ARM_BASELINE ? "separate" : "fused";
#else
    return arm == ARM_BASELINE ? "baseline" : "candidate";
#endif
}

static int select_variant(bench_arm arm) {
#ifdef DS4_METAL_IQ2_MOE_TOP8_PAIR_BENCH
    ds4_gpu_test_set_flags(arm == ARM_CANDIDATE
        ? DS4_GPU_TEST_REQUIRE_IQ2_TOP8_PAIR_SWIGLU : 0u);
    return arm == ARM_BASELINE
        ? setenv(PAIR_FUSION_DISABLE_ENV, "1", 1) == 0
        : unsetenv(PAIR_FUSION_DISABLE_ENV) == 0;
#else
    if (unsetenv(PAIR_TAIL_ENABLE_ENV) != 0 ||
        unsetenv(PAIR_TAIL_DISABLE_ENV) != 0) {
        return 0;
    }
    return setenv(arm == ARM_BASELINE ? PAIR_TAIL_DISABLE_ENV :
                  PAIR_TAIL_ENABLE_ENV, "1", 1) == 0;
#endif
}

static int run_once(fixture *f, bench_arm arm,
                    const char *phase, const char *order,
                    uint32_t sample, uint32_t cycle, uint32_t position,
                    bool poison, bool check_canaries) {
    if (!select_variant(arm)) {
        fprintf(stderr,
                "metal-iq2-moe-tail-cull-bench: environment setup failed\n");
        return 0;
    }
    if (poison && !poison_outputs(f)) {
        fprintf(stderr,
                "metal-iq2-moe-tail-cull-bench: output poison failed\n");
        return 0;
    }

    fprintf(stderr,
            "DS4_IQ2_MOE_TAIL_BENCH phase=%s experiment=%s variant=%s "
            "sample=%u cycle=%u position=%u order=%s force_resident=1\n",
            phase, EXPERIMENT_NAME, variant_name(arm), sample,
            cycle, position, order);
    fflush(stderr);

    if (!ds4_gpu_begin_commands()) {
        fprintf(stderr,
                "metal-iq2-moe-tail-cull-bench: begin commands failed\n");
        return 0;
    }
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
    int ok = call_ok && end_ok && mid_is_f16;
    if (!ok) {
        fprintf(stderr,
                "DS4_IQ2_MOE_TAIL_BENCH result=FAIL experiment=%s "
                "variant=%s call=%d end=%d mid_f16=%d\n",
                EXPERIMENT_NAME, variant_name(arm),
                call_ok, end_ok, mid_is_f16 ? 1 : 0);
    }
    if (check_canaries) ok = check_all_canaries(f) && ok;
    return ok;
}

static int snapshot_alloc(oracle_snapshot *snapshot, const fixture *f) {
    memset(snapshot, 0, sizeof(*snapshot));
    snapshot->bytes = f->pair_f16_bytes + f->expert_bytes + f->out_bytes;
    if (snapshot->bytes > SIZE_MAX) return 0;
    snapshot->storage = malloc((size_t)snapshot->bytes);
    if (!snapshot->storage) return 0;
    snapshot->mid = snapshot->storage;
    snapshot->experts = snapshot->mid + f->pair_f16_bytes;
    snapshot->out = snapshot->experts + f->expert_bytes;
    return 1;
}

static int capture_snapshot(oracle_snapshot *snapshot, const fixture *f) {
    return ds4_gpu_tensor_read(
               f->mid, 0, snapshot->mid, f->pair_f16_bytes) &&
           ds4_gpu_tensor_read(
               f->experts, 0, snapshot->experts, f->expert_bytes) &&
           ds4_gpu_tensor_read(
               f->out, 0, snapshot->out, f->out_bytes);
}

static int tensor_matches(const char *candidate, const char *name,
                          const ds4_gpu_tensor *tensor,
                          const uint8_t *expected, uint64_t bytes,
                          uint8_t *scratch, size_t scratch_bytes) {
    uint64_t offset = 0;
    while (offset < bytes) {
        const size_t chunk = bytes - offset > scratch_bytes ?
            scratch_bytes : (size_t)(bytes - offset);
        if (!ds4_gpu_tensor_read(tensor, offset, scratch, chunk)) {
            fprintf(stderr,
                    "DS4_IQ2_MOE_TAIL_ORACLE candidate=%s tensor=%s "
                    "result=READ_FAIL offset=%llu\n",
                    candidate, name, (unsigned long long)offset);
            return 0;
        }
        if (memcmp(scratch, expected + offset, chunk) != 0) {
            size_t mismatch = 0;
            while (mismatch < chunk &&
                   scratch[mismatch] == expected[offset + mismatch]) {
                mismatch++;
            }
            fprintf(stderr,
                    "DS4_IQ2_MOE_TAIL_ORACLE candidate=%s tensor=%s "
                    "result=MISMATCH byte=%llu expected=0x%02x actual=0x%02x\n",
                    candidate, name,
                    (unsigned long long)(offset + mismatch),
                    expected[offset + mismatch], scratch[mismatch]);
            return 0;
        }
        offset += chunk;
    }
    return 1;
}

static int compare_candidate(const char *candidate, const fixture *f,
                             const oracle_snapshot *baseline,
                             uint8_t *scratch) {
    const int mid_ok = tensor_matches(
        candidate, "mid_f16", f->mid, baseline->mid,
        f->pair_f16_bytes, scratch, COMPARE_CHUNK_BYTES);
    const int experts_ok = tensor_matches(
        candidate, "experts_f32", f->experts, baseline->experts,
        f->expert_bytes, scratch, COMPARE_CHUNK_BYTES);
    const int out_ok = tensor_matches(
        candidate, "out_f32", f->out, baseline->out,
        f->out_bytes, scratch, COMPARE_CHUNK_BYTES);
    const int ok = mid_ok && experts_ok && out_ok;
    fprintf(stderr,
            "DS4_IQ2_MOE_TAIL_ORACLE candidate=%s result=%s "
            "mid_f16=%s experts_f32=%s out_f32=%s canaries=PASS\n",
            candidate, ok ? "PASS" : "FAIL",
            mid_ok ? "exact" : "mismatch",
            experts_ok ? "exact" : "mismatch",
            out_ok ? "exact" : "mismatch");
    return ok;
}

static int run_oracle(fixture *f) {
    oracle_snapshot baseline;
    if (!snapshot_alloc(&baseline, f)) {
        fprintf(stderr,
                "metal-iq2-moe-tail-cull-bench: oracle snapshot allocation "
                "failed\n");
        return 0;
    }
    uint8_t *scratch = malloc(COMPARE_CHUNK_BYTES);
    int ok = scratch != NULL;

    if (ok) {
        ok = run_once(f, ARM_BASELINE,
                      "oracle", "baseline", 0u, 0u, 0u, true, true);
    }
    if (ok) ok = capture_snapshot(&baseline, f);

    if (ok) {
        ok = run_once(f, ARM_CANDIDATE,
                      "oracle", "pair", 0u, 0u, 0u, true, true);
    }
    if (ok) ok = compare_candidate(EXPERIMENT_NAME, f, &baseline, scratch);

    free(scratch);
    free(baseline.storage);
    return ok;
}

static int run_balanced_block(fixture *f, const char *phase, uint32_t cycles,
                              uint32_t sample_limit) {
    uint32_t arm_samples[2] = {0, 0};
    for (uint32_t cycle = 0; cycle < cycles; cycle++) {
        static const bench_arm abba[4] = {
            ARM_BASELINE, ARM_CANDIDATE,
            ARM_CANDIDATE, ARM_BASELINE,
        };
        static const bench_arm baab[4] = {
            ARM_CANDIDATE, ARM_BASELINE,
            ARM_BASELINE, ARM_CANDIDATE,
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
    if (arm_samples[ARM_BASELINE] != sample_limit ||
        arm_samples[ARM_CANDIDATE] != sample_limit) {
        fprintf(stderr,
                "metal-iq2-moe-tail-cull-bench: %s %s arm count "
                "baseline=%u candidate=%u expected=%u\n",
                EXPERIMENT_NAME, phase,
                arm_samples[ARM_BASELINE], arm_samples[ARM_CANDIDATE],
                sample_limit);
        return 0;
    }
    return 1;
}

static int run_experiment(fixture *f, const bench_config *config) {
    if (config->warmup_cycles != 0u &&
        !run_balanced_block(f, "warmup",
                            config->warmup_cycles,
                            config->warmup_cycles * 2u)) {
        return 0;
    }
    if (!run_balanced_block(f, "sample",
                            config->samples / 2u, config->samples)) {
        return 0;
    }
    fprintf(stderr,
            "DS4_IQ2_MOE_TAIL_BENCH phase=complete experiment=%s "
            "samples_per_variant=%u warmup_per_variant=%u result=PASS\n",
            EXPERIMENT_NAME, config->samples,
            config->warmup_cycles * 2u);
    return 1;
}

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
    const uint64_t page = (uint64_t)getpagesize();
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
    f->pair_f16_bytes = f->pair_count * sizeof(_Float16);
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
                "metal-iq2-moe-tail-cull-bench: production layout mismatch "
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
    const uint64_t oracle_bytes =
        f->pair_f16_bytes + f->expert_bytes + f->out_bytes +
        COMPARE_CHUNK_BYTES;
    const uint64_t setup_host_bytes =
        f->x_bytes + f->route_i32_bytes + f->route_f32_bytes;
    const uint64_t explicit_peak =
        f->model_size + tensor_bytes + oracle_bytes + setup_host_bytes;
    fprintf(stderr,
            "DS4_IQ2_MOE_TAIL_SETUP geometry=N%u,d%u,mid%u,out%u,"
            "experts%u,top%u model=%.3f_GiB tensors=%.3f_GiB "
            "oracle=%.3f_GiB explicit_peak=%.3f_GiB\n",
            N_TOKENS, IN_DIM, MID_DIM, OUT_DIM,
            N_TOTAL_EXPERT, N_EXPERT,
            (double)f->model_size / (double)GIB,
            (double)tensor_bytes / (double)GIB,
            (double)oracle_bytes / (double)GIB,
            (double)explicit_peak / (double)GIB);
    if (explicit_peak >= 5u * GIB) {
        fprintf(stderr,
                "metal-iq2-moe-tail-cull-bench: explicit peak exceeds "
                "5 GiB\n");
        return 0;
    }

    if (posix_memalign(&f->model, (size_t)page,
                       (size_t)f->model_size) != 0) {
        fprintf(stderr,
                "metal-iq2-moe-tail-cull-bench: model allocation failed\n");
        return 0;
    }
    fprintf(stderr,
            "DS4_IQ2_MOE_TAIL_SETUP phase=fill_weights tensor=gate\n");
    fill_iq2((block_iq2_xxs *)((uint8_t *)f->model + f->gate_offset),
             19u);
    fprintf(stderr,
            "DS4_IQ2_MOE_TAIL_SETUP phase=fill_weights tensor=up\n");
    fill_iq2((block_iq2_xxs *)((uint8_t *)f->model + f->up_offset),
             47u);
    fprintf(stderr,
            "DS4_IQ2_MOE_TAIL_SETUP phase=fill_weights tensor=down\n");
    fill_q2((block_q2_K *)((uint8_t *)f->model + f->down_offset));
    const uint64_t checksum =
        touch_model_pages(f->model, f->model_size, page);
    fprintf(stderr,
            "DS4_IQ2_MOE_TAIL_SETUP phase=touch_weights pages=%llu "
            "checksum=0x%016llx\n",
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
                "metal-iq2-moe-tail-cull-bench: fixture initialization "
                "failed\n");
    }
    return ok;
}

int main(int argc, char **argv) {
    const bench_config config = parse_options(argc, argv);

    /* This benchmark compares the legacy grouped pair kernels directly.
     * Prevent Metal 4 TensorOps/MPP from bypassing either A/B variant on
     * M5+ hosts; production defaults remain independently hardware-gated. */
    setenv("DS4_METAL_DISABLE_METAL4", "1", 1);
    setenv("DS4_METAL_MOE_STAGE_PROFILE", "1", 1);
    setenv("DS4_METAL_MOE_STAGE_PROFILE_LAYER", "0", 1);
    unsetenv("DS4_METAL_MOE_STAGE_PROFILE_FILTER");
    unsetenv(PAIR_FUSION_DISABLE_ENV);
    unsetenv("DS4_METAL_MOE_WRITE_CLAMPED_ACT");
    unsetenv("DS4_METAL_GRAPH_DUMP_PREFIX");
    unsetenv(PAIR_TAIL_ENABLE_ENV);
    unsetenv(PAIR_TAIL_DISABLE_ENV);

    if (!ds4_gpu_init()) {
        fprintf(stderr,
                "metal-iq2-moe-tail-cull-bench: Metal initialization failed\n");
        return 1;
    }
    ds4_gpu_set_quality(false);
    ds4_gpu_set_ssd_streaming(false);

    fixture f;
    int ok = init_fixture(&f);
    if (ok) ok = run_oracle(&f);
    if (ok) ok = run_experiment(&f, &config);
    if (ok) ok = check_all_canaries(&f);
    (void)select_variant(ARM_BASELINE);

    free_tensors(&f);
    ds4_gpu_cleanup();
    free(f.model);
    fprintf(stderr,
            "DS4_IQ2_MOE_TAIL_BENCH result=%s\n",
            ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
