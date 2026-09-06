#define _DARWIN_C_SOURCE

/* Resident Q4_K oracle for the per-stream Metal command queues.
 *
 * This deliberately uses a tiny synthetic mmap-shaped model instead of a
 * production GGUF: the local Q4 target is much larger than unified memory.
 * Each row is evaluated three ways: synchronous FIFO, one native row batch,
 * and one command buffer per stream.  All three retain the same Q4_K kernel
 * reduction order and therefore must be bit-identical.
 *
 * The default is a short correctness/leak smoke.  Set
 * DS4_TEST_Q4_STREAM_SOAK=N for a bounded longer overlap soak, and
 * DS4_TEST_Q4_STREAM_TIMING=1 to report wall-clock A/B numbers.
 */

#include "ds4.h"
#include "ds4_gpu.h"

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#ifdef __APPLE__

#include <mach/mach.h>
#include <sys/resource.h>
#include <time.h>

#define Q4_K_TYPE 12u
#define QK_K 256u
/* DeepSeek-V4 Flash AProjQ4 q_a/attn_kv projection geometry. */
#define IN_DIM 4096u
#define OUT0_DIM 1024u
#define OUT1_DIM 512u
#define MAX_STREAMS 8u
#define GUARD_FLOATS 64u
#define MAX_TIMING_BLOCKS 51u

typedef struct {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[12];
    uint8_t qs[QK_K / 2u];
} block_q4_K;

typedef enum {
    ARM_FIFO,
    ARM_NATIVE,
    ARM_OVERLAP,
} test_arm;

typedef struct {
    uint64_t footprint;
    uint64_t resident;
    uint64_t virtual_size;
    uint64_t max_rss;
} task_memory;

typedef struct {
    void *model;
    uint64_t model_size;
    uint64_t weight0_offset;
    uint64_t weight1_offset;
    uint64_t row_bytes;
    ds4_gpu_tensor *x;
    ds4_gpu_tensor *x_row[MAX_STREAMS];
    ds4_gpu_tensor *out[3][2];
    ds4_gpu_tensor *out_row[3][2][MAX_STREAMS];
    float *host[3][2];
    uint64_t out_count[2];
    float *x_host;
} fixture;

static void fail(const char *what) {
    fprintf(stderr, "Q4 stream oracle FAIL: %s\n", what);
    exit(1);
}

typedef struct {
    const char *name;
    char *value;
    bool present;
} saved_env;

static saved_env save_env(const char *name) {
    const char *value = getenv(name);
    saved_env saved = {
        .name = name,
        .value = value ? strdup(value) : NULL,
        .present = value != NULL,
    };
    if (value && !saved.value) fail("environment snapshot");
    return saved;
}

static void restore_env(saved_env *saved) {
    const int rc = saved->present
        ? setenv(saved->name, saved->value, 1)
        : unsetenv(saved->name);
    free(saved->value);
    saved->value = NULL;
    if (rc != 0) fail("environment restore");
}

static void expect_overlap_policy(const char *label, int expected,
                                  int count, bool resident,
                                  bool ssd_streaming, bool quality) {
    const int actual = ds4_test_q4_stream_overlap_policy(
        count, resident, ssd_streaming, quality);
    if (actual != expected) {
        fprintf(stderr,
                "Q4 stream policy %s: got=%d expected=%d "
                "count=%d resident=%d ssd=%d quality=%d\n",
                label, actual, expected, count,
                resident ? 1 : 0, ssd_streaming ? 1 : 0,
                quality ? 1 : 0);
        fail("scheduler admission policy");
    }
}

static void test_overlap_policy(void) {
    saved_env enabled = save_env("DS4_METAL_ENABLE_Q4_STREAM_OVERLAP");
    saved_env disabled = save_env("DS4_METAL_DISABLE_Q4_STREAM_OVERLAP");

    if (unsetenv(enabled.name) != 0 || unsetenv(disabled.name) != 0) {
        fail("environment clear");
    }
    expect_overlap_policy("default-off", 0, 2, true, false, false);

    if (setenv(enabled.name, "1", 1) != 0) fail("enable policy");
    expect_overlap_policy("minimum-count", 1, 2, true, false, false);
    expect_overlap_policy("maximum-count", 1, 8, true, false, false);
    expect_overlap_policy("count-one", 0, 1, true, false, false);
    expect_overlap_policy("count-nine", 0, 9, true, false, false);
    expect_overlap_policy("nonresident", 0, 2, false, false, false);
    expect_overlap_policy("ssd", 0, 2, true, true, false);
    expect_overlap_policy("quality", 0, 2, true, false, true);

    if (setenv(disabled.name, "1", 1) != 0) fail("disable policy");
    expect_overlap_policy("disable-precedence", 0, 2, true, false, false);
    if (setenv(disabled.name, "0", 1) != 0) fail("clear disable policy");
    expect_overlap_policy("disable-zero", 1, 2, true, false, false);
    if (setenv(enabled.name, "0", 1) != 0) fail("clear enable policy");
    expect_overlap_policy("enable-zero", 0, 2, true, false, false);

    restore_env(&disabled);
    restore_env(&enabled);
    fprintf(stderr,
            "Q4 stream policy PASS default-off=1 disable-precedence=1 "
            "count-bounds=1 resident-only=1 ssd-fallback=1 quality-fallback=1\n");
}

static void test_indexer_q_type_policy(void) {
    if (!ds4_test_indexer_q_type_supported(1u) ||
        !ds4_test_indexer_q_type_supported(8u) ||
        !ds4_test_indexer_q_type_supported(Q4_K_TYPE) ||
        ds4_test_indexer_q_type_supported(0u) ||
        ds4_test_indexer_q_type_supported(2u)) {
        fail("indexer query projection type policy");
    }
}

static uint64_t env_u64(const char *name, uint64_t fallback,
                        uint64_t minimum, uint64_t maximum) {
    const char *value = getenv(name);
    if (!value || !value[0]) return fallback;
    char *end = NULL;
    unsigned long long parsed = strtoull(value, &end, 10);
    if (end == value || *end != '\0' || parsed < minimum || parsed > maximum) {
        fprintf(stderr, "Q4 stream oracle invalid %s=%s (range %llu..%llu)\n",
                name, value,
                (unsigned long long)minimum,
                (unsigned long long)maximum);
        exit(1);
    }
    return (uint64_t)parsed;
}

static uint64_t align_up(uint64_t value, uint64_t alignment) {
    return (value + alignment - 1u) / alignment * alignment;
}

static double now_seconds(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) fail("clock_gettime");
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static task_memory read_task_memory(void) {
    task_memory result = {0};
    task_vm_info_data_t info;
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO,
                  (task_info_t)&info, &count) == KERN_SUCCESS) {
        result.footprint = (uint64_t)info.phys_footprint;
        result.resident = (uint64_t)info.resident_size;
        result.virtual_size = (uint64_t)info.virtual_size;
    }
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
        /* ru_maxrss is bytes on Darwin. */
        result.max_rss = (uint64_t)usage.ru_maxrss;
    }
    return result;
}

static void print_task_memory(const char *label, task_memory memory) {
    const double mib = 1024.0 * 1024.0;
    fprintf(stderr,
            "Q4 stream memory %-12s footprint=%.2f MiB resident=%.2f MiB "
            "virtual=%.2f MiB peak_rss=%.2f MiB\n",
            label,
            (double)memory.footprint / mib,
            (double)memory.resident / mib,
            (double)memory.virtual_size / mib,
            (double)memory.max_rss / mib);
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

static void q4_pack_scales(uint8_t packed[12], const uint8_t scale[8],
                           const uint8_t minimum[8]) {
    memset(packed, 0, 12u);
    for (uint32_t group = 0; group < 4u; group++) {
        packed[group] = scale[group] & 63u;
        packed[group + 4u] = minimum[group] & 63u;
    }
    for (uint32_t group = 4u; group < 8u; group++) {
        packed[group + 4u] = (scale[group] & 15u) |
                             ((minimum[group] & 15u) << 4u);
        packed[group - 4u] |= (scale[group] >> 4u) << 6u;
        packed[group] |= (minimum[group] >> 4u) << 6u;
    }
}

static void fill_q4_matrix(block_q4_K *matrix, uint32_t rows, uint32_t salt) {
    const uint32_t blocks_per_row = IN_DIM / QK_K;
    for (uint32_t row = 0; row < rows; row++) {
        for (uint32_t block = 0; block < blocks_per_row; block++) {
            block_q4_K *b = matrix + (uint64_t)row * blocks_per_row + block;
            const uint32_t key = salt + row * 1009u + block * 313u;
            uint8_t scale[8];
            uint8_t minimum[8];
            for (uint32_t group = 0; group < 8u; group++) {
                scale[group] = (uint8_t)(1u + (key + group * 7u) % 31u);
                minimum[group] = (uint8_t)((key / 3u + group * 5u) % 17u);
            }
            q4_pack_scales(b->scales, scale, minimum);
            for (uint32_t i = 0; i < QK_K / 2u; i++) {
                b->qs[i] = (uint8_t)(key + i * 37u + (i >> 2u) * 11u);
            }
            /* Exact binary scales: 2^-5 and 2^-7. */
            b->d = 0x2800u;
            b->dmin = 0x2000u;
        }
    }
}

static float q4_dot(const block_q4_K *row, const float *x) {
    float sum = 0.0f;
    for (uint32_t k = 0; k < IN_DIM; k++) {
        const block_q4_K *b = row + k / QK_K;
        const uint32_t in_block = k % QK_K;
        const uint32_t group = in_block / 32u;
        const uint32_t lane = in_block % 32u;
        uint8_t scale, minimum;
        q4_scale_min(b->scales, group, &scale, &minimum);
        const uint32_t byte_offset = (group >> 1u) * 32u + lane;
        const uint32_t shift = (group & 1u) * 4u;
        const uint32_t q = (b->qs[byte_offset] >> shift) & 15u;
        const float w = f16_to_f32(b->d) * (float)scale * (float)q -
                        f16_to_f32(b->dmin) * (float)minimum;
        sum += w * x[k];
    }
    return sum;
}

static uint64_t checksum(const float *values, uint64_t count) {
    const uint8_t *bytes = (const uint8_t *)values;
    const uint64_t byte_count = count * sizeof(float);
    uint64_t hash = UINT64_C(1469598103934665603);
    for (uint64_t i = 0; i < byte_count; i++) {
        hash ^= bytes[i];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static const char *arm_name(test_arm arm) {
    switch (arm) {
    case ARM_FIFO: return "fifo";
    case ARM_NATIVE: return "native";
    case ARM_OVERLAP: return "overlap";
    }
    return "unknown";
}

static int encode_pair(fixture *f, ds4_gpu_tensor *out0,
                       ds4_gpu_tensor *out1, const ds4_gpu_tensor *x,
                       uint32_t rows) {
    return ds4_gpu_matmul_q4_K_pair_tensor(
        out0, out1, f->model, f->model_size,
        f->weight0_offset, f->weight1_offset,
        IN_DIM, OUT0_DIM, OUT1_DIM, x, rows) > 0;
}

static int run_arm_once_with_options(fixture *f, test_arm arm,
                                     uint32_t streams,
                                     bool hold_test_transient) {
    if (arm == ARM_NATIVE) {
        ds4_gpu_set_stream(0);
        return encode_pair(f, f->out[arm][0], f->out[arm][1], f->x, streams);
    }
    if (arm == ARM_FIFO) {
        ds4_gpu_set_stream(0);
        for (uint32_t i = 0; i < streams; i++) {
            if (!encode_pair(f, f->out_row[arm][0][i],
                             f->out_row[arm][1][i], f->x_row[i], 1u)) {
                return 0;
            }
        }
        return 1;
    }

    uint32_t submitted = 0;
    for (uint32_t i = 0; i < streams; i++) {
        ds4_gpu_set_stream((int)i);
        if (!ds4_gpu_begin_commands() ||
            (hold_test_transient &&
             !ds4_gpu_test_hold_stream_transient(4096u)) ||
            !encode_pair(f, f->out_row[arm][0][i],
                         f->out_row[arm][1][i], f->x_row[i], 1u) ||
            !ds4_gpu_end_commands_async()) {
            goto fail_overlap;
        }
        submitted++;
    }
    for (uint32_t i = 0; i < submitted; i++) {
        if (!ds4_gpu_wait_stream((int)i)) goto fail_overlap;
    }
    ds4_gpu_set_stream(0);
    return 1;

fail_overlap:
    for (uint32_t i = 0; i < submitted; i++) {
        (void)ds4_gpu_wait_stream((int)i);
    }
    ds4_gpu_set_stream(0);
    return 0;
}

static int run_arm_once(fixture *f, test_arm arm, uint32_t streams) {
    return run_arm_once_with_options(f, arm, streams, true);
}

/* Exercise the intermediate-command-buffer lifetime too: the first pair and
 * its test resource must survive flush, while wait_stream must retire both
 * the pending and final command buffers before releasing transients. */
static int run_overlap_flush_once(fixture *f, uint32_t streams) {
    uint32_t submitted = 0;
    for (uint32_t i = 0; i < streams; i++) {
        ds4_gpu_set_stream((int)i);
        if (!ds4_gpu_begin_commands() ||
            !ds4_gpu_test_hold_stream_transient(4096u) ||
            !encode_pair(f, f->out_row[ARM_OVERLAP][0][i],
                         f->out_row[ARM_OVERLAP][1][i], f->x_row[i], 1u) ||
            !ds4_gpu_flush_commands() ||
            !encode_pair(f, f->out_row[ARM_OVERLAP][0][i],
                         f->out_row[ARM_OVERLAP][1][i], f->x_row[i], 1u) ||
            !ds4_gpu_end_commands_async()) {
            goto fail_overlap;
        }
        submitted++;
    }
    for (uint32_t i = 0; i < submitted; i++) {
        if (!ds4_gpu_wait_stream((int)i)) goto fail_overlap;
    }
    ds4_gpu_set_stream(0);
    return 1;

fail_overlap:
    for (uint32_t i = 0; i < submitted; i++) {
        (void)ds4_gpu_wait_stream((int)i);
    }
    ds4_gpu_set_stream(0);
    return 0;
}

static void poison_arm(fixture *f, test_arm arm, float poison) {
    for (uint32_t output = 0; output < 2u; output++) {
        for (uint64_t i = 0; i < f->out_count[output]; i++) {
            f->host[arm][output][i] = poison;
        }
        if (!ds4_gpu_tensor_write(f->out[arm][output], 0,
                                  f->host[arm][output],
                                  f->out_count[output] * sizeof(float))) {
            fail("output poison write");
        }
    }
}

static void read_arm(fixture *f, test_arm arm) {
    for (uint32_t output = 0; output < 2u; output++) {
        if (!ds4_gpu_tensor_read(f->out[arm][output], 0,
                                 f->host[arm][output],
                                 f->out_count[output] * sizeof(float))) {
            fail("output read");
        }
    }
}

static void check_outputs(fixture *f, uint32_t streams, float poison) {
    const uint32_t dims[2] = {OUT0_DIM, OUT1_DIM};
    const uint64_t offsets[2] = {f->weight0_offset, f->weight1_offset};
    for (test_arm arm = ARM_FIFO; arm <= ARM_OVERLAP; arm++) read_arm(f, arm);

    for (uint32_t output = 0; output < 2u; output++) {
        const uint64_t active = (uint64_t)streams * dims[output];
        const block_q4_K *matrix = (const block_q4_K *)
            ((const uint8_t *)f->model + offsets[output]);
        float max_abs = 0.0f;
        float max_rel = 0.0f;
        for (uint32_t stream = 0; stream < streams; stream++) {
            for (uint32_t row = 0; row < dims[output]; row++) {
                const uint64_t index = (uint64_t)stream * dims[output] + row;
                const float expected = q4_dot(
                    matrix + (uint64_t)row * (IN_DIM / QK_K),
                    f->x_host + (uint64_t)stream * IN_DIM);
                const float actual = f->host[ARM_FIFO][output][index];
                const float absolute = fabsf(actual - expected);
                const float relative = absolute / fmaxf(1.0f, fabsf(expected));
                if (absolute > max_abs) max_abs = absolute;
                if (relative > max_rel) max_rel = relative;
                if (!isfinite(actual) || (absolute > 0.004f && relative > 2e-5f)) {
                    fprintf(stderr,
                            "Q4 stream CPU mismatch output=%u stream=%u row=%u "
                            "expected=%g actual=%g abs=%g rel=%g\n",
                            output, stream, row, expected, actual,
                            absolute, relative);
                    fail("CPU tolerance");
                }
            }
        }
        for (uint64_t i = active; i < f->out_count[output]; i++) {
            for (test_arm arm = ARM_FIFO; arm <= ARM_OVERLAP; arm++) {
                if (memcmp(&f->host[arm][output][i], &poison,
                           sizeof(poison)) != 0) {
                    fprintf(stderr,
                            "Q4 stream canary mismatch arm=%s output=%u index=%llu\n",
                            arm_name(arm), output, (unsigned long long)i);
                    fail("output canary");
                }
            }
        }
        for (test_arm arm = ARM_NATIVE; arm <= ARM_OVERLAP; arm++) {
            if (memcmp(f->host[ARM_FIFO][output], f->host[arm][output],
                       active * sizeof(float)) != 0) {
                uint64_t first = 0;
                while (first < active &&
                       memcmp(&f->host[ARM_FIFO][output][first],
                              &f->host[arm][output][first], sizeof(float)) == 0) {
                    first++;
                }
                fprintf(stderr,
                        "Q4 stream exact mismatch arm=%s output=%u first=%llu "
                        "fifo=%g actual=%g\n",
                        arm_name(arm), output, (unsigned long long)first,
                        first < active ? f->host[ARM_FIFO][output][first] : 0.0f,
                        first < active ? f->host[arm][output][first] : 0.0f);
                fail("bitwise parity");
            }
        }
        fprintf(stderr,
                "Q4 stream N=%u output=%u CPU max_abs=%g max_rel=%g "
                "checksum=%016llx bitwise=1 canary=1\n",
                streams, output, max_abs, max_rel,
                (unsigned long long)checksum(f->host[ARM_FIFO][output], active));
    }
}

static void check_stats_equal(const ds4_gpu_stream_test_stats *before,
                              const ds4_gpu_stream_test_stats *after,
                              const char *scope) {
    if (before->tensor_live_bytes != after->tensor_live_bytes ||
        before->tensor_live_count != after->tensor_live_count ||
        after->transient_references != 0u ||
        after->pending_command_buffers != 0u ||
        after->last_command_buffers != 0u) {
        fprintf(stderr,
                "Q4 stream stats leak scope=%s live_bytes=%llu->%llu "
                "live_count=%u->%u transient=%llu->%llu pending=%u last=%u\n",
                scope,
                (unsigned long long)before->tensor_live_bytes,
                (unsigned long long)after->tensor_live_bytes,
                before->tensor_live_count, after->tensor_live_count,
                (unsigned long long)before->transient_references,
                (unsigned long long)after->transient_references,
                after->pending_command_buffers,
                after->last_command_buffers);
        fail("Metal allocation/transient counters");
    }
}

static int compare_double(const void *a, const void *b) {
    const double av = *(const double *)a;
    const double bv = *(const double *)b;
    return (av > bv) - (av < bv);
}

static double percentile(const double *samples, uint32_t count, double p) {
    double sorted[MAX_TIMING_BLOCKS];
    memcpy(sorted, samples, (size_t)count * sizeof(sorted[0]));
    qsort(sorted, count, sizeof(sorted[0]), compare_double);
    const double rank = p * (double)(count - 1u);
    const uint32_t lo = (uint32_t)rank;
    const uint32_t hi = lo + 1u < count ? lo + 1u : lo;
    const double fraction = rank - (double)lo;
    return sorted[lo] + (sorted[hi] - sorted[lo]) * fraction;
}

static double run_timing_group(fixture *f, test_arm arm, uint32_t streams,
                               uint64_t iterations) {
    const double start = now_seconds();
    for (uint64_t i = 0; i < iterations; i++) {
        /* The transient hook is a lifecycle oracle, not production work.  Do
         * not charge its allocation/retain overhead to the overlap arm. */
        if (!run_arm_once_with_options(f, arm, streams, false)) {
            fail("timing arm");
        }
    }
    return (now_seconds() - start) / (double)iterations;
}

static void run_timing_pair(fixture *f, test_arm baseline, test_arm candidate,
                            uint32_t streams, uint32_t blocks,
                            uint64_t group_iterations) {
    double baseline_samples[MAX_TIMING_BLOCKS];
    double candidate_samples[MAX_TIMING_BLOCKS];
    for (uint32_t block = 0; block < blocks; block++) {
        double base_total = 0.0;
        double candidate_total = 0.0;
        if ((block & 1u) == 0u) {
            base_total += run_timing_group(
                f, baseline, streams, group_iterations);
            candidate_total += run_timing_group(
                f, candidate, streams, group_iterations);
            candidate_total += run_timing_group(
                f, candidate, streams, group_iterations);
            base_total += run_timing_group(
                f, baseline, streams, group_iterations);
        } else {
            candidate_total += run_timing_group(
                f, candidate, streams, group_iterations);
            base_total += run_timing_group(
                f, baseline, streams, group_iterations);
            base_total += run_timing_group(
                f, baseline, streams, group_iterations);
            candidate_total += run_timing_group(
                f, candidate, streams, group_iterations);
        }
        baseline_samples[block] = base_total * 0.5;
        candidate_samples[block] = candidate_total * 0.5;
    }

    const double baseline_median = percentile(baseline_samples, blocks, 0.50);
    const double candidate_median = percentile(candidate_samples, blocks, 0.50);
    const double speedup = candidate_median > 0.0 ?
        baseline_median / candidate_median : 0.0;
    fprintf(stderr,
            "Q4 stream timing microkernel=%s/%s N=%u blocks=%u group_iters=%llu "
            "baseline_ms[p25/med/p75]=%.3f/%.3f/%.3f "
            "candidate_ms[p25/med/p75]=%.3f/%.3f/%.3f speedup=%.3fx "
            "aggregate_candidate=%.2f rows/s wall=encode+GPU full_logits=0\n",
            arm_name(baseline), arm_name(candidate), streams, blocks,
            (unsigned long long)group_iterations,
            percentile(baseline_samples, blocks, 0.25) * 1000.0,
            baseline_median * 1000.0,
            percentile(baseline_samples, blocks, 0.75) * 1000.0,
            percentile(candidate_samples, blocks, 0.25) * 1000.0,
            candidate_median * 1000.0,
            percentile(candidate_samples, blocks, 0.75) * 1000.0,
            speedup,
            candidate_median > 0.0 ? (double)streams / candidate_median : 0.0);
}

static void fixture_init(fixture *f) {
    memset(f, 0, sizeof(*f));
    if (sizeof(block_q4_K) != 144u) fail("unexpected Q4_K block size");

    f->row_bytes = (IN_DIM / QK_K) * sizeof(block_q4_K);
    f->weight0_offset = 0;
    f->weight1_offset = OUT0_DIM * f->row_bytes;
    const uint64_t weights_end = f->weight1_offset + OUT1_DIM * f->row_bytes;
    const uint64_t page = (uint64_t)getpagesize();
    f->model_size = align_up(weights_end, page);
    if (posix_memalign(&f->model, (size_t)page, (size_t)f->model_size) != 0) {
        fail("model allocation");
    }
    memset(f->model, 0, (size_t)f->model_size);
    fill_q4_matrix((block_q4_K *)((uint8_t *)f->model + f->weight0_offset),
                   OUT0_DIM, 17u);
    fill_q4_matrix((block_q4_K *)((uint8_t *)f->model + f->weight1_offset),
                   OUT1_DIM, 7919u);

    const uint64_t x_count = (uint64_t)MAX_STREAMS * IN_DIM;
    f->x_host = malloc((size_t)x_count * sizeof(float));
    if (!f->x_host) fail("host activation allocation");
    for (uint32_t stream = 0; stream < MAX_STREAMS; stream++) {
        for (uint32_t k = 0; k < IN_DIM; k++) {
            const int32_t value = (int32_t)(
                (stream * 19u + k * 7u + (stream ^ k) * 3u) % 127u) - 63;
            f->x_host[(uint64_t)stream * IN_DIM + k] = (float)value / 128.0f;
        }
    }

    if (!ds4_gpu_init() || !ds4_gpu_set_model_map(f->model, f->model_size)) {
        fail("Metal init/model map");
    }
    ds4_gpu_set_quality(false);
    f->x = ds4_gpu_tensor_alloc(x_count * sizeof(float));
    if (!f->x || !ds4_gpu_tensor_write(f->x, 0, f->x_host,
                                        x_count * sizeof(float))) {
        fail("activation tensor");
    }
    for (uint32_t stream = 0; stream < MAX_STREAMS; stream++) {
        f->x_row[stream] = ds4_gpu_tensor_view(
            f->x, (uint64_t)stream * IN_DIM * sizeof(float),
            IN_DIM * sizeof(float));
        if (!f->x_row[stream]) fail("activation row view");
    }

    const uint32_t dims[2] = {OUT0_DIM, OUT1_DIM};
    for (uint32_t output = 0; output < 2u; output++) {
        f->out_count[output] = (uint64_t)MAX_STREAMS * dims[output] + GUARD_FLOATS;
        for (test_arm arm = ARM_FIFO; arm <= ARM_OVERLAP; arm++) {
            f->host[arm][output] = malloc(
                (size_t)f->out_count[output] * sizeof(float));
            f->out[arm][output] = ds4_gpu_tensor_alloc(
                f->out_count[output] * sizeof(float));
            if (!f->host[arm][output] || !f->out[arm][output]) {
                fail("output allocation");
            }
            for (uint32_t stream = 0; stream < MAX_STREAMS; stream++) {
                f->out_row[arm][output][stream] = ds4_gpu_tensor_view(
                    f->out[arm][output],
                    (uint64_t)stream * dims[output] * sizeof(float),
                    (uint64_t)dims[output] * sizeof(float));
                if (!f->out_row[arm][output][stream]) {
                    fail("output row view");
                }
            }
        }
    }
}

static void fixture_release_tensors(fixture *f) {
    for (uint32_t output = 0; output < 2u; output++) {
        for (test_arm arm = ARM_FIFO; arm <= ARM_OVERLAP; arm++) {
            for (uint32_t stream = 0; stream < MAX_STREAMS; stream++) {
                ds4_gpu_tensor_free(f->out_row[arm][output][stream]);
            }
            ds4_gpu_tensor_free(f->out[arm][output]);
            free(f->host[arm][output]);
        }
    }
    for (uint32_t stream = 0; stream < MAX_STREAMS; stream++) {
        ds4_gpu_tensor_free(f->x_row[stream]);
    }
    ds4_gpu_tensor_free(f->x);
    f->x = NULL;
}

static void fixture_destroy(fixture *f) {
    fixture_release_tensors(f);

    /* Cleanup itself is responsible for every stream.  Leave one empty async
     * command buffer and one retained test resource on the last stream so a
     * cleanup implementation that only drains stream zero cannot false-pass. */
    ds4_gpu_set_stream((int)MAX_STREAMS - 1);
    if (!ds4_gpu_begin_commands() ||
        !ds4_gpu_test_hold_stream_transient(4096u) ||
        !ds4_gpu_end_commands_async()) {
        fail("cleanup in-flight setup");
    }
    ds4_gpu_stream_test_stats inflight;
    ds4_gpu_test_stream_stats(&inflight);
    if (inflight.last_command_buffers == 0u ||
        inflight.transient_references == 0u) {
        fail("cleanup in-flight state was not observable");
    }
    ds4_gpu_cleanup();
    free(f->x_host);
    free(f->model);
    memset(f, 0, sizeof(*f));
}

int main(void) {
    static const uint32_t stream_counts[] = {2u, 4u, 8u};
    const float poison = -12345.25f;
    const uint64_t warmup = env_u64("DS4_TEST_Q4_STREAM_WARMUP", 1u, 1u, 64u);
    const uint64_t iterations = env_u64(
        "DS4_TEST_Q4_STREAM_ITERS", 2u, 1u, 10000u);
    const uint64_t soak = env_u64(
        "DS4_TEST_Q4_STREAM_SOAK", 8u, 1u, 100000u);
    const bool timing = getenv("DS4_TEST_Q4_STREAM_TIMING") != NULL;
    const uint32_t timing_blocks = (uint32_t)env_u64(
        "DS4_TEST_Q4_STREAM_TIMING_BLOCKS", 5u, 5u, MAX_TIMING_BLOCKS);
    const uint64_t timing_iterations = env_u64(
        "DS4_TEST_Q4_STREAM_TIMING_ITERS", 20u, 1u, 10000u);

    test_overlap_policy();
    test_indexer_q_type_policy();

    fprintf(stderr,
            "Q4 stream oracle model_untracked=%s warmup=%llu iterations=%llu "
            "soak=%llu timing=%d footprint=synth-resident\n",
            getenv("DS4_METAL_MODEL_UNTRACKED") ? "on" : "off",
            (unsigned long long)warmup,
            (unsigned long long)iterations,
            (unsigned long long)soak,
            timing ? 1 : 0);
    const task_memory process_before = read_task_memory();
    print_task_memory("process-start", process_before);

    fixture f;
    fixture_init(&f);
    ds4_gpu_stream_test_stats initial;
    ds4_gpu_test_stream_stats(&initial);
    if (initial.active_queue_mask != 1u ||
        initial.model_residency_queue_mask != 1u) {
        fail("initial model residency");
    }
    for (uint32_t c = 0; c < sizeof(stream_counts) / sizeof(stream_counts[0]); c++) {
        const uint32_t streams = stream_counts[c];
        for (test_arm arm = ARM_FIFO; arm <= ARM_OVERLAP; arm++) {
            poison_arm(&f, arm, poison);
            for (uint64_t i = 0; i < warmup; i++) {
                if (!run_arm_once(&f, arm, streams)) fail("warmup arm");
            }
        }

        ds4_gpu_stream_test_stats stats_before;
        ds4_gpu_test_stream_stats(&stats_before);
        if (stats_before.transient_references != 0u) {
            fail("transients retained after warmup");
        }

        for (test_arm arm = ARM_FIFO; arm <= ARM_OVERLAP; arm++) {
            for (uint64_t i = 0; i < iterations; i++) {
                if (!run_arm_once(&f, arm, streams)) fail("test arm");
            }
        }
        check_outputs(&f, streams, poison);
        if (timing) {
            run_timing_pair(&f, ARM_FIFO, ARM_OVERLAP, streams,
                            timing_blocks, timing_iterations);
            run_timing_pair(&f, ARM_NATIVE, ARM_OVERLAP, streams,
                            timing_blocks, timing_iterations);
        }

        ds4_gpu_stream_test_stats stats_after;
        ds4_gpu_test_stream_stats(&stats_after);
        check_stats_equal(&stats_before, &stats_after, "A/B");

        if (!run_overlap_flush_once(&f, streams)) {
            fail("overlap flush lifecycle");
        }
        ds4_gpu_stream_test_stats flush_after;
        ds4_gpu_test_stream_stats(&flush_after);
        check_stats_equal(&stats_before, &flush_after, "flush");
        check_outputs(&f, streams, poison);
        const uint32_t expected_mask = (1u << streams) - 1u;
        if ((stats_after.active_queue_mask & expected_mask) != expected_mask ||
            (stats_after.model_residency_queue_mask & expected_mask) != expected_mask) {
            fprintf(stderr,
                    "Q4 stream queue residency N=%u active=0x%02x "
                    "model=0x%02x expected=0x%02x\n",
                    streams, stats_after.active_queue_mask,
                    stats_after.model_residency_queue_mask, expected_mask);
            fail("residency missing from an active queue");
        }

        ds4_gpu_stream_test_stats soak_before;
        ds4_gpu_test_stream_stats(&soak_before);
        for (uint64_t i = 0; i < soak; i++) {
            if (!run_arm_once(&f, ARM_OVERLAP, streams)) fail("overlap soak");
        }
        ds4_gpu_stream_test_stats soak_after;
        ds4_gpu_test_stream_stats(&soak_after);
        check_stats_equal(&soak_before, &soak_after, "soak");
        fprintf(stderr,
                "Q4 stream N=%u soak=%llu active_queues=0x%02x "
                "model_residency=0x%02x transient=%llu counters=stable\n",
                streams, (unsigned long long)soak,
                soak_after.active_queue_mask,
                soak_after.model_residency_queue_mask,
                (unsigned long long)soak_after.transient_references);
    }

    /* Rebuild residency after all queues already exist.  Model replacement in
     * a long-lived engine must attach the new set to every existing queue. */
    if (!ds4_gpu_set_model_map(f.model, f.model_size)) {
        fail("model residency rebuild");
    }
    ds4_gpu_stream_test_stats rebuilt;
    ds4_gpu_test_stream_stats(&rebuilt);
    if (rebuilt.active_queue_mask != 0xffu ||
        rebuilt.model_residency_queue_mask != 0xffu) {
        fprintf(stderr,
                "Q4 residency rebuild active=0x%02x model=0x%02x\n",
                rebuilt.active_queue_mask,
                rebuilt.model_residency_queue_mask);
        fail("model residency rebuild queue coverage");
    }

    print_task_memory("before-cleanup", read_task_memory());
    fixture_destroy(&f);
    ds4_gpu_stream_test_stats cleanup_stats;
    ds4_gpu_test_stream_stats(&cleanup_stats);
    if (cleanup_stats.tensor_live_bytes != 0u ||
        cleanup_stats.tensor_live_count != 0u ||
        cleanup_stats.transient_references != 0u ||
        cleanup_stats.pending_command_buffers != 0u ||
        cleanup_stats.last_command_buffers != 0u ||
        cleanup_stats.active_queue_mask != 0u) {
        fail("cleanup counters did not return to zero");
    }
    print_task_memory("after-cleanup", read_task_memory());
    fprintf(stderr,
            "test_metal_q4_streams PASS counts=2,4,8 bitwise=1 canary=1 "
            "model_residency=1 counters=stable model_untracked=%s\n",
            getenv("DS4_METAL_MODEL_UNTRACKED") ? "on" : "off");
    return 0;
}

#else

int main(void) {
    fprintf(stderr, "test_metal_q4_streams skipped (Metal requires macOS)\n");
    return 0;
}

#endif
