#define _DARWIN_C_SOURCE

#include "ds4_gpu.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef __APPLE__

enum {
    PROD_VOCAB = 129280u,
    GUARD_WORDS = 8u,
    OVERLAP_CALLS = 17u,
    BENCH_CALLS = 512u,
    TAIL_BENCH_CALLS = 128u,
    BENCH_SAMPLES = 8u,
};

static const uint32_t k_guard = 0x7fc12345u;
static const char *k_disable = "DS4_METAL_DISABLE_DECODE_ARGMAX_TOP1";
static const char *k_require = "DS4_METAL_REQUIRE_DECODE_ARGMAX_TOP1";

bool ds4_log_is_tty(FILE *fp) {
    (void)fp;
    return false;
}

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1.0e9;
}

static float f32_from_bits(uint32_t bits) {
    float value = 0.0f;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static int cpu_argmax(const float *v, uint32_t n) {
    int best = 0;
    float best_v = f32_from_bits(0xff800000u);
    for (uint32_t i = 0; i < n; i++) {
        if (v[i] > best_v) {
            best_v = v[i];
            best = (int)i;
        }
    }
    return best;
}

/* Mirrors the eight-way ILP used by production sample_argmax for the finite
 * benchmark row.  The correctness oracle above intentionally stays scalar so
 * its edge-case behavior remains obvious. */
static int cpu_argmax_decode(const float *v, uint32_t n) {
    int best = 0;
    float best_v = f32_from_bits(0xff800000u);
    int bi[8] = {0};
    float bv[8] = {
        best_v, best_v, best_v, best_v,
        best_v, best_v, best_v, best_v,
    };
    uint32_t i = 0;
    for (; n - i >= 8u; i += 8u) {
        for (uint32_t lane = 0; lane < 8u; lane++) {
            const float x = v[i + lane];
            if (x > bv[lane]) {
                bv[lane] = x;
                bi[lane] = (int)(i + lane);
            }
        }
    }
    for (uint32_t lane = 0; lane < 8u; lane++) {
        if (bv[lane] > best_v ||
            (bv[lane] == best_v && bi[lane] < best)) {
            best_v = bv[lane];
            best = bi[lane];
        }
    }
    for (; i < n; i++) {
        if (v[i] > best_v) {
            best_v = v[i];
            best = (int)i;
        }
    }
    return best;
}

static void fill_case(float *v, uint32_t n, uint32_t kind) {
    for (uint32_t i = 0; i < n; i++) {
        const int32_t centered = (int32_t)((i * 2654435761u + 17u) % 8191u) - 4095;
        v[i] = (float)centered / 257.0f;
    }
    switch (kind) {
        case 0:
            v[91357u] = 1000.0f;
            break;
        case 1:
            v[17u] = 2000.0f;
            v[80003u] = 2000.0f;
            break;
        case 2:
            v[9u] = f32_from_bits(0x7f800000u);
            v[42000u] = f32_from_bits(0x7f800000u);
            break;
        case 3:
            for (uint32_t i = 0; i < n; i++) {
                v[i] = f32_from_bits(0xff800000u);
            }
            break;
        case 4:
            for (uint32_t i = 0; i < n; i++) {
                v[i] = f32_from_bits(0x7fc00001u);
            }
            break;
        case 5:
            for (uint32_t i = 0; i < n; i++) {
                v[i] = f32_from_bits((i & 1u) ? 0x7fc00001u : 0xff800000u);
            }
            v[777u] = -100.0f;
            break;
        default:
            abort();
    }
}

static int set_candidate(bool candidate) {
    if (candidate) {
        return unsetenv(k_disable) == 0 &&
               setenv(k_require, "1", 1) == 0;
    }
    return unsetenv(k_require) == 0 &&
           setenv(k_disable, "1", 1) == 0;
}

static int run_one(ds4_gpu_tensor *out, const ds4_gpu_tensor *logits,
                   uint32_t n_vocab, bool candidate, int32_t *result) {
    return set_candidate(candidate) &&
           ds4_gpu_argmax_tensor(out, logits, n_vocab) &&
           ds4_gpu_tensor_read(out, 0, result, sizeof(*result));
}

static int check_guarded_cases(void) {
    const size_t total = (size_t)PROD_VOCAB + 2u * GUARD_WORDS;
    uint32_t *host = malloc(total * sizeof(*host));
    ds4_gpu_tensor *input_base = ds4_gpu_tensor_alloc(total * sizeof(float));
    ds4_gpu_tensor *input = input_base ? ds4_gpu_tensor_view(
        input_base, (uint64_t)GUARD_WORDS * sizeof(float),
        (uint64_t)PROD_VOCAB * sizeof(float)) : NULL;
    ds4_gpu_tensor *out_base = ds4_gpu_tensor_alloc(3u * sizeof(uint32_t));
    ds4_gpu_tensor *out = out_base ? ds4_gpu_tensor_view(
        out_base, sizeof(uint32_t), sizeof(uint32_t)) : NULL;
    int ok = host && input_base && input && out_base && out;

    for (uint32_t kind = 0; ok && kind < 6u; kind++) {
        for (size_t i = 0; i < total; i++) host[i] = k_guard;
        float *values = (float *)(host + GUARD_WORDS);
        fill_case(values, PROD_VOCAB, kind);
        const int expected = cpu_argmax(values, PROD_VOCAB);
        uint32_t out_words[3] = {k_guard, k_guard, k_guard};
        int32_t control = -1;
        int32_t candidate = -1;
        ok = ds4_gpu_tensor_write(input_base, 0, host,
                                  total * sizeof(*host)) &&
             ds4_gpu_tensor_write(out_base, 0, out_words,
                                  sizeof(out_words)) &&
             run_one(out, input, PROD_VOCAB, false, &control) &&
             ds4_gpu_tensor_write(out_base, 0, out_words,
                                  sizeof(out_words)) &&
             run_one(out, input, PROD_VOCAB, true, &candidate) &&
             ds4_gpu_tensor_read(out_base, 0, out_words,
                                 sizeof(out_words)) &&
             ds4_gpu_tensor_read(input_base, 0, host,
                                 total * sizeof(*host));
        if (!ok) break;
        if (candidate != expected ||
            (kind < 3u && control != candidate) ||
            out_words[0] != k_guard || out_words[2] != k_guard) {
            fprintf(stderr,
                    "argmax case %u failed: expected=%d control=%d candidate=%d "
                    "guards=%08x/%08x\n",
                    kind, expected, control, candidate,
                    out_words[0], out_words[2]);
            ok = 0;
        }
        for (uint32_t i = 0; ok && i < GUARD_WORDS; i++) {
            if (host[i] != k_guard ||
                host[GUARD_WORDS + PROD_VOCAB + i] != k_guard) {
                fprintf(stderr, "argmax input guard changed in case %u\n", kind);
                ok = 0;
            }
        }
    }

    ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(out_base);
    ds4_gpu_tensor_free(input);
    ds4_gpu_tensor_free(input_base);
    free(host);
    if (ok) fprintf(stderr, "Metal top-1 production-shape oracle: PASS\n");
    return ok;
}

static int check_vocab_shapes(void) {
    static const uint32_t sizes[] = {4096u, 4097u, 129279u, 129281u};
    const uint32_t max_n = sizes[sizeof(sizes) / sizeof(sizes[0]) - 1u];
    float *host = malloc((size_t)max_n * sizeof(*host));
    ds4_gpu_tensor *logits = ds4_gpu_tensor_alloc(
        (uint64_t)max_n * sizeof(float));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(sizeof(int32_t));
    int ok = host && logits && out;
    for (size_t shape = 0;
         ok && shape < sizeof(sizes) / sizeof(sizes[0]);
         shape++) {
        const uint32_t n = sizes[shape];
        for (uint32_t i = 0; i < n; i++) {
            host[i] = -(float)((i * 17u + 3u) % 8191u);
        }
        host[n - 1u] = 2000.0f;
        const int expected = cpu_argmax(host, n);
        int32_t control = -1;
        int32_t candidate = -1;
        ok = ds4_gpu_tensor_write(logits, 0, host,
                                  (uint64_t)n * sizeof(*host)) &&
             run_one(out, logits, n, false, &control) &&
             run_one(out, logits, n, true, &candidate);
        if (ok && (control != expected || candidate != expected)) {
            fprintf(stderr,
                    "Metal top-1 shape %u failed: expected=%d control=%d candidate=%d\n",
                    n, expected, control, candidate);
            ok = 0;
        }
    }
    ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(logits);
    free(host);
    if (ok) fprintf(stderr, "Metal top-1 tail/empty-group shapes: PASS\n");
    return ok;
}

static int check_scratch_wrap(void) {
    const size_t values = (size_t)OVERLAP_CALLS * PROD_VOCAB;
    float *host = malloc(values * sizeof(*host));
    int32_t expected[OVERLAP_CALLS];
    int32_t results[OVERLAP_CALLS + 2u];
    ds4_gpu_tensor *logits = ds4_gpu_tensor_alloc(values * sizeof(float));
    ds4_gpu_tensor *out_base = ds4_gpu_tensor_alloc(sizeof(results));
    ds4_gpu_tensor *rows[OVERLAP_CALLS] = {0};
    ds4_gpu_tensor *outs[OVERLAP_CALLS] = {0};
    int ok = host && logits && out_base;

    for (uint32_t row = 0; ok && row < OVERLAP_CALLS; row++) {
        float *v = host + (size_t)row * PROD_VOCAB;
        for (uint32_t i = 0; i < PROD_VOCAB; i++) {
            v[i] = -(float)((i + row * 13u) % 4093u);
        }
        const uint32_t winner = (row * 7919u + 23u) % PROD_VOCAB;
        v[winner] = 1000.0f + (float)row;
        expected[row] = (int32_t)winner;
        rows[row] = ds4_gpu_tensor_view(
            logits, (uint64_t)row * PROD_VOCAB * sizeof(float),
            (uint64_t)PROD_VOCAB * sizeof(float));
        outs[row] = ds4_gpu_tensor_view(
            out_base, (uint64_t)(row + 1u) * sizeof(int32_t),
            sizeof(int32_t));
        ok = rows[row] && outs[row];
    }
    for (uint32_t i = 0; i < OVERLAP_CALLS + 2u; i++) {
        results[i] = (int32_t)k_guard;
    }
    if (ok) {
        ok = ds4_gpu_tensor_write(logits, 0, host,
                                  values * sizeof(float)) &&
             ds4_gpu_tensor_write(out_base, 0, results, sizeof(results)) &&
             set_candidate(true) && ds4_gpu_begin_commands();
    }
    for (uint32_t row = 0; ok && row < OVERLAP_CALLS; row++) {
        ok = ds4_gpu_argmax_tensor(outs[row], rows[row], PROD_VOCAB);
        if (ok && row + 1u < OVERLAP_CALLS) {
            ok = ds4_gpu_flush_commands();
        }
    }
    if (ok) ok = ds4_gpu_end_commands();
    else (void)ds4_gpu_synchronize();
    if (ok) ok = ds4_gpu_tensor_read(out_base, 0, results, sizeof(results));
    if (ok && ((uint32_t)results[0] != k_guard ||
               (uint32_t)results[OVERLAP_CALLS + 1u] != k_guard)) {
        fprintf(stderr, "argmax overlap output guard changed\n");
        ok = 0;
    }
    for (uint32_t row = 0; ok && row < OVERLAP_CALLS; row++) {
        if (results[row + 1u] != expected[row]) {
            fprintf(stderr,
                    "argmax scratch wrap failed at call %u: expected=%d got=%d\n",
                    row, expected[row], results[row + 1u]);
            ok = 0;
        }
    }

    for (uint32_t row = 0; row < OVERLAP_CALLS; row++) {
        ds4_gpu_tensor_free(outs[row]);
        ds4_gpu_tensor_free(rows[row]);
    }
    ds4_gpu_tensor_free(out_base);
    ds4_gpu_tensor_free(logits);
    free(host);
    if (ok) {
        fprintf(stderr,
                "Metal top-1 scratch-ring wrap (%u command-buffer submissions "
                "without host waits): PASS\n",
                OVERLAP_CALLS);
    }
    return ok;
}

static int timed_batch(ds4_gpu_tensor *out,
                       const ds4_gpu_tensor *logits,
                       bool candidate,
                       uint32_t calls,
                       double *elapsed) {
    if (!set_candidate(candidate) || !ds4_gpu_begin_commands()) return 0;
    const double t0 = now_sec();
    int ok = 1;
    for (uint32_t i = 0; ok && i < calls; i++) {
        ok = ds4_gpu_argmax_tensor(out, logits, PROD_VOCAB);
    }
    if (ok) ok = ds4_gpu_end_commands();
    else (void)ds4_gpu_synchronize();
    const double t1 = now_sec();
    if (ok && elapsed) *elapsed = t1 - t0;
    return ok;
}

static int cmp_double(const void *a, const void *b) {
    const double x = *(const double *)a;
    const double y = *(const double *)b;
    return (x > y) - (x < y);
}

static int timed_selection_tail(
        ds4_gpu_tensor *out,
        ds4_gpu_tensor *logits,
        const ds4_gpu_tensor *producer,
        float *readback,
        int expected,
        bool candidate,
        uint32_t calls,
        double *elapsed) {
    if (!set_candidate(true)) return 0;
    const double t0 = now_sec();
    int ok = 1;
    for (uint32_t i = 0; ok && i < calls; i++) {
        int32_t got = -1;
        ok = ds4_gpu_begin_commands();
        if (ok) {
            ok = ds4_gpu_tensor_copy(
                logits, 0, producer, 0,
                (uint64_t)PROD_VOCAB * sizeof(float));
        }
        if (ok && candidate) {
            ok = ds4_gpu_argmax_tensor(out, logits, PROD_VOCAB);
        }
        if (ok) ok = ds4_gpu_end_commands();
        else (void)ds4_gpu_synchronize();
        if (ok && candidate) {
            ok = ds4_gpu_tensor_read(out, 0, &got, sizeof(got));
        } else if (ok) {
            ok = ds4_gpu_tensor_read(logits, 0, readback,
                                     (uint64_t)PROD_VOCAB * sizeof(*readback));
            if (ok) got = cpu_argmax_decode(readback, PROD_VOCAB);
        }
        if (ok && got != expected) {
            fprintf(stderr,
                    "Metal selection-tail mismatch: candidate=%d expected=%d got=%d\n",
                    candidate, expected, got);
            ok = 0;
        }
    }
    const double t1 = now_sec();
    if (ok && elapsed) *elapsed = t1 - t0;
    return ok;
}

static int run_selection_tail_benchmark(
        ds4_gpu_tensor *out,
        ds4_gpu_tensor *logits,
        const ds4_gpu_tensor *producer,
        const float *values) {
    float *readback = malloc((size_t)PROD_VOCAB * sizeof(*readback));
    int ok = readback != NULL;
    const int expected = cpu_argmax_decode(values, PROD_VOCAB);
    double ignored = 0.0;
    if (ok) {
        ok = timed_selection_tail(out, logits, producer, readback, expected,
                                  false, 16u, &ignored);
    }
    if (ok) {
        ok = timed_selection_tail(out, logits, producer, readback, expected,
                                  true, 16u, &ignored);
    }

    double control[BENCH_SAMPLES] = {0};
    double candidate[BENCH_SAMPLES] = {0};
    uint32_t nc = 0;
    uint32_t nn = 0;
    for (uint32_t cycle = 0; ok && cycle < BENCH_SAMPLES / 2u; cycle++) {
        const bool order[4] = {false, true, true, false};
        for (uint32_t j = 0; ok && j < 4u; j++) {
            double elapsed = 0.0;
            ok = timed_selection_tail(out, logits, producer,
                                      readback, expected,
                                      order[j], TAIL_BENCH_CALLS, &elapsed);
            if (!ok) break;
            if (order[j]) candidate[nn++] = elapsed;
            else control[nc++] = elapsed;
        }
    }
    if (ok && (nc != BENCH_SAMPLES || nn != BENCH_SAMPLES)) ok = 0;
    if (ok) {
        qsort(control, BENCH_SAMPLES, sizeof(control[0]), cmp_double);
        qsort(candidate, BENCH_SAMPLES, sizeof(candidate[0]), cmp_double);
        const double control_median =
            0.5 * (control[BENCH_SAMPLES / 2u - 1u] +
                   control[BENCH_SAMPLES / 2u]);
        const double candidate_median =
            0.5 * (candidate[BENCH_SAMPLES / 2u - 1u] +
                   candidate[BENCH_SAMPLES / 2u]);
        const double control_us =
            control_median * 1.0e6 / TAIL_BENCH_CALLS;
        const double candidate_us =
            candidate_median * 1.0e6 / TAIL_BENCH_CALLS;
        const double saved_us = control_us - candidate_us;
        fprintf(stderr,
                "Metal resident greedy-selection tail A/B (%u logits, "
                "%u calls/sample, %u samples): full-read+CPU=%.3f us "
                "p95=%.3f us device-top1+4B=%.3f us p95=%.3f us "
                "saved=%.3f us speedup=%.2fx reduction=%.1f%%\n",
                PROD_VOCAB, TAIL_BENCH_CALLS, BENCH_SAMPLES,
                control_us,
                control[BENCH_SAMPLES - 1u] * 1.0e6 / TAIL_BENCH_CALLS,
                candidate_us,
                candidate[BENCH_SAMPLES - 1u] * 1.0e6 / TAIL_BENCH_CALLS,
                saved_us, control_us / candidate_us,
                saved_us * 100.0 / control_us);
        fprintf(stderr,
                "Metal resident greedy-selection tail excludes GGUF/model access and SSD I/O.\n");
    }
    free(readback);
    return ok;
}

static int run_benchmark(void) {
    float *host = malloc((size_t)PROD_VOCAB * sizeof(*host));
    ds4_gpu_tensor *logits = ds4_gpu_tensor_alloc(
        (uint64_t)PROD_VOCAB * sizeof(float));
    ds4_gpu_tensor *producer = ds4_gpu_tensor_alloc(
        (uint64_t)PROD_VOCAB * sizeof(float));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(sizeof(int32_t));
    int ok = host && logits && producer && out;
    if (ok) {
        fill_case(host, PROD_VOCAB, 0u);
        ok = ds4_gpu_tensor_write(logits, 0, host,
                                  (uint64_t)PROD_VOCAB * sizeof(float)) &&
             ds4_gpu_tensor_write(producer, 0, host,
                                  (uint64_t)PROD_VOCAB * sizeof(float));
    }

    double ignored = 0.0;
    if (ok) ok = timed_batch(out, logits, false, 64u, &ignored);
    if (ok) ok = timed_batch(out, logits, true, 64u, &ignored);

    double control[BENCH_SAMPLES] = {0};
    double candidate[BENCH_SAMPLES] = {0};
    uint32_t nc = 0;
    uint32_t nn = 0;
    for (uint32_t cycle = 0; ok && cycle < BENCH_SAMPLES / 2u; cycle++) {
        const bool order[4] = {false, true, true, false};
        for (uint32_t j = 0; ok && j < 4u; j++) {
            double elapsed = 0.0;
            ok = timed_batch(out, logits, order[j], BENCH_CALLS, &elapsed);
            if (!ok) break;
            int32_t got = -1;
            ok = ds4_gpu_tensor_read(out, 0, &got, sizeof(got));
            if (!ok || got != 91357) {
                fprintf(stderr,
                        "Metal timed top-1 output mismatch: candidate=%d expected=91357 got=%d\n",
                        order[j], got);
                ok = 0;
                break;
            }
            if (order[j]) candidate[nn++] = elapsed;
            else control[nc++] = elapsed;
        }
    }
    if (ok && (nc != BENCH_SAMPLES || nn != BENCH_SAMPLES)) ok = 0;
    if (ok) {
        qsort(control, BENCH_SAMPLES, sizeof(control[0]), cmp_double);
        qsort(candidate, BENCH_SAMPLES, sizeof(candidate[0]), cmp_double);
        const double control_median =
            0.5 * (control[BENCH_SAMPLES / 2u - 1u] +
                   control[BENCH_SAMPLES / 2u]);
        const double candidate_median =
            0.5 * (candidate[BENCH_SAMPLES / 2u - 1u] +
                   candidate[BENCH_SAMPLES / 2u]);
        const double control_us = control_median * 1.0e6 / BENCH_CALLS;
        const double candidate_us = candidate_median * 1.0e6 / BENCH_CALLS;
        const double saved_us = control_us - candidate_us;
        const double speedup = control_us / candidate_us;
        const double gain = saved_us * 100.0 / control_us;
        fprintf(stderr,
                "Metal resident warm top-1 A/B (%u logits, %u calls/sample, "
                "%u samples): control=%.3f us p95=%.3f us "
                "candidate=%.3f us p95=%.3f us saved=%.3f us "
                "speedup=%.2fx reduction=%.1f%%\n",
                PROD_VOCAB, BENCH_CALLS, BENCH_SAMPLES,
                control_us,
                control[BENCH_SAMPLES - 1u] * 1.0e6 / BENCH_CALLS,
                candidate_us,
                candidate[BENCH_SAMPLES - 1u] * 1.0e6 / BENCH_CALLS,
                saved_us, speedup, gain);
        fprintf(stderr,
                "Metal resident top-1 A/B excludes GGUF/model access and SSD I/O.\n");
    }

    if (ok) {
        ok = run_selection_tail_benchmark(out, logits, producer, host);
    }

    ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(producer);
    ds4_gpu_tensor_free(logits);
    free(host);
    return ok;
}

int main(void) {
    int ok = ds4_gpu_init();
    if (ok) ok = check_guarded_cases();
    if (ok) ok = check_vocab_shapes();
    if (ok) ok = check_scratch_wrap();
    if (ok && getenv("DS4_TEST_METAL_ARGMAX_TOP1_TIMING") != NULL) {
        ok = run_benchmark();
    }
    (void)unsetenv(k_disable);
    (void)unsetenv(k_require);
    ds4_gpu_cleanup();
    return ok ? 0 : 1;
}

#else

int main(void) {
    fprintf(stderr, "test_metal_argmax_top1: skipped (Metal requires macOS)\n");
    return 0;
}

#endif
