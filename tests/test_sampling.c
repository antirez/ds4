#include "../ds4.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
    int id;
    float logit;
    float prob;
} reference_candidate;

static int failures;

#define CHECK(cond, ...) do {                                                 \
    if (!(cond)) {                                                            \
        fprintf(stderr, "FAIL: " __VA_ARGS__);                               \
        fputc('\n', stderr);                                                  \
        failures++;                                                           \
    }                                                                         \
} while (0)

static uint64_t reference_rng_next(uint64_t *state) {
    uint64_t x = *state;
    if (x == 0) x = 0x9e3779b97f4a7c15ULL;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    return x * 0x2545f4914f6cdd1dULL;
}

static float reference_rng_f32(uint64_t *state) {
    const uint64_t x = reference_rng_next(state);
    return (float)((x >> 40) & 0xffffffu) / 16777216.0f;
}

static int reference_argmax(const float *logits, uint32_t n_vocab) {
    int best = 0;
    float best_v = -INFINITY;
    for (uint32_t i = 0; i < n_vocab; i++) {
        if (logits[i] > best_v) {
            best_v = logits[i];
            best = (int)i;
        }
    }
    return best;
}

static int reference_candidate_cmp_desc(const void *a, const void *b) {
    const reference_candidate *ca = a;
    const reference_candidate *cb = b;
    const int logit_order =
        (cb->logit > ca->logit) - (cb->logit < ca->logit);
    if (logit_order != 0) return logit_order;
    return (ca->id > cb->id) - (ca->id < cb->id);
}

/* This is the sampler immediately before the optimized implementation. */
static int reference_sample(const float *logits, uint32_t n_vocab,
                            float temperature, int top_k,
                            float top_p, float min_p, uint64_t *rng) {
    if (temperature <= 0.0f) return reference_argmax(logits, n_vocab);
    if (top_p <= 0.0f || top_p > 1.0f) top_p = 1.0f;
    if (min_p < 0.0f) min_p = 0.0f;

    if (top_k > 0) {
        if (top_k > 1024) top_k = 1024;
        if ((uint32_t)top_k > n_vocab) top_k = (int)n_vocab;
        int ids[1024];
        float vals[1024];
        int n = 0;
        for (uint32_t i = 0; i < n_vocab; i++) {
            const float v = logits[i];
            if (!isfinite(v)) continue;
            if (n == top_k && v <= vals[n - 1]) continue;
            int j = n < top_k ? n++ : n - 1;
            while (j > 0 && vals[j - 1] < v) {
                vals[j] = vals[j - 1];
                ids[j] = ids[j - 1];
                j--;
            }
            vals[j] = v;
            ids[j] = (int)i;
        }
        if (n == 0) return reference_argmax(logits, n_vocab);

        float probs[1024];
        const float max_logit = vals[0];
        float sum = 0.0f;
        for (int i = 0; i < n; i++) {
            probs[i] = expf((vals[i] - max_logit) / temperature);
            sum += probs[i];
        }
        if (sum <= 0.0f || !isfinite(sum)) return ids[0];
        const float min_prob = (probs[0] / sum) * min_p;
        float filtered_sum = 0.0f;
        int filtered = 0;
        for (int i = 0; i < n; i++) {
            const float p = probs[i] / sum;
            if (i > 0 && p < min_prob) break;
            filtered_sum += probs[i];
            filtered++;
            if (filtered_sum / sum >= top_p) break;
        }
        float r = reference_rng_f32(rng) * filtered_sum;
        for (int i = 0; i < filtered; i++) {
            r -= probs[i];
            if (r <= 0.0f) return ids[i];
        }
        return ids[filtered - 1];
    }

    float max_logit = -INFINITY;
    int best = 0;
    uint32_t finite = 0;
    for (uint32_t i = 0; i < n_vocab; i++) {
        const float v = logits[i];
        if (!isfinite(v)) continue;
        finite++;
        if (v > max_logit) {
            max_logit = v;
            best = (int)i;
        }
    }
    if (finite == 0) return reference_argmax(logits, n_vocab);

    if (top_p >= 1.0f) {
        float sum = 0.0f;
        const float min_rel = min_p > 0.0f ? min_p : 0.0f;
        for (uint32_t i = 0; i < n_vocab; i++) {
            const float v = logits[i];
            if (!isfinite(v)) continue;
            const float p = expf((v - max_logit) / temperature);
            if (p < min_rel) continue;
            sum += p;
        }
        if (sum <= 0.0f || !isfinite(sum)) return best;
        float r = reference_rng_f32(rng) * sum;
        for (uint32_t i = 0; i < n_vocab; i++) {
            const float v = logits[i];
            if (!isfinite(v)) continue;
            const float p = expf((v - max_logit) / temperature);
            if (p < min_rel) continue;
            r -= p;
            if (r <= 0.0f) return (int)i;
        }
        return best;
    }

    reference_candidate *cand = malloc((size_t)finite * sizeof(*cand));
    CHECK(cand != NULL, "reference candidate allocation");
    if (!cand) return best;
    uint32_t n = 0;
    float sum = 0.0f;
    for (uint32_t i = 0; i < n_vocab; i++) {
        const float v = logits[i];
        if (!isfinite(v)) continue;
        const float p = expf((v - max_logit) / temperature);
        cand[n++] = (reference_candidate){(int)i, v, p};
        sum += p;
    }
    if (sum <= 0.0f || !isfinite(sum)) {
        free(cand);
        return best;
    }
    qsort(cand, n, sizeof(*cand), reference_candidate_cmp_desc);
    const float min_prob = (cand[0].prob / sum) * (min_p > 0.0f ? min_p : 0.0f);
    float filtered_sum = 0.0f;
    uint32_t filtered = 0;
    for (uint32_t i = 0; i < n; i++) {
        const float p = cand[i].prob / sum;
        if (i > 0 && p < min_prob) break;
        filtered_sum += cand[i].prob;
        filtered++;
        if (filtered_sum / sum >= top_p) break;
    }
    float r = reference_rng_f32(rng) * filtered_sum;
    for (uint32_t i = 0; i < filtered; i++) {
        r -= cand[i].prob;
        if (r <= 0.0f) {
            const int id = cand[i].id;
            free(cand);
            return id;
        }
    }
    const int id = cand[filtered - 1].id;
    free(cand);
    return id;
}

static uint64_t data_rng_next(uint64_t *state) {
    *state = *state * 6364136223846793005ULL + 1442695040888963407ULL;
    return *state;
}

static void fill_logits(float *logits, uint32_t n, uint64_t seed) {
    for (uint32_t i = 0; i < n; i++) {
        const uint64_t x = data_rng_next(&seed);
        const int32_t q = (int32_t)(x >> 32) % 250000;
        logits[i] = (float)q / 10000.0f + (float)i * 1.0e-7f;
    }
    if (n > 17) logits[17] = -INFINITY;
    if (n > 113) logits[113] = NAN;
}

static void compare_case(const float *logits, float *scratch, uint32_t n,
                         float temperature, int top_k,
                         float top_p, float min_p, const char *label) {
    for (uint64_t seed = 0; seed < 256; seed++) {
        uint64_t ref_rng = seed;
        uint64_t opt_rng = seed;
        const int ref = reference_sample(logits, n, temperature, top_k,
                                         top_p, min_p, &ref_rng);
        const int opt = ds4_test_sample_logits(logits, n, temperature, top_k,
                                               top_p, min_p, &opt_rng, scratch);
        CHECK(ref == opt,
              "%s seed=%llu token reference=%d optimized=%d",
              label, (unsigned long long)seed, ref, opt);
        CHECK(ref_rng == opt_rng,
              "%s seed=%llu RNG reference=%llu optimized=%llu",
              label, (unsigned long long)seed,
              (unsigned long long)ref_rng, (unsigned long long)opt_rng);
    }
}

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1.0e-9;
}

int main(void) {
    const uint32_t semantic_n = 4096;
    float *logits = malloc((size_t)semantic_n * sizeof(*logits));
    float *scratch = malloc((size_t)semantic_n * sizeof(*scratch));
    CHECK(logits && scratch, "semantic scratch allocation");
    if (!logits || !scratch) return 1;
    fill_logits(logits, semantic_n, 0x123456789abcdef0ULL);

    compare_case(logits, scratch, semantic_n, 1.0f, 0, 1.0f, 0.05f,
                 "default-min-p");
    compare_case(logits, scratch, semantic_n, 0.7f, 0, 1.0f, 0.0f,
                 "full-softmax");
    compare_case(logits, scratch, semantic_n, 1.3f, 0, 0.9f, 0.05f,
                 "top-p-min-p");
    compare_case(logits, scratch, semantic_n, 0.9f, 0, 0.95f, 0.0f,
                 "top-p");
    compare_case(logits, scratch, semantic_n, 0.8f, 64, 0.9f, 0.05f,
                 "top-k");
    compare_case(logits, scratch, semantic_n, 0.0f, 0, 1.0f, 0.05f,
                 "greedy");

    /* Exercise min-p values immediately around expf's cutoff. */
    const float cutoff = logf(0.05f);
    const float boundary_logits[] = {
        0.0f,
        cutoff,
        nextafterf(cutoff, -INFINITY),
        nextafterf(cutoff, INFINITY),
        -1.0f,
        -5.0f,
        -INFINITY,
        NAN,
    };
    float boundary_scratch[sizeof(boundary_logits) / sizeof(boundary_logits[0])];
    compare_case(boundary_logits, boundary_scratch,
                 (uint32_t)(sizeof(boundary_logits) / sizeof(boundary_logits[0])),
                 1.0f, 0, 1.0f, 0.05f, "min-p-boundary");

    const float tied_logits[] = {
        2.0f, 2.0f, 2.0f, 1.0f, 1.0f, 0.0f, -INFINITY, NAN,
    };
    float tied_scratch[sizeof(tied_logits) / sizeof(tied_logits[0])];
    compare_case(tied_logits, tied_scratch,
                 (uint32_t)(sizeof(tied_logits) / sizeof(tied_logits[0])),
                 1.0f, 0, 1.0f, 0.05f, "equal-logits-default");
    compare_case(tied_logits, tied_scratch,
                 (uint32_t)(sizeof(tied_logits) / sizeof(tied_logits[0])),
                 1.0f, 0, 0.8f, 0.05f, "equal-logits-top-p");
    compare_case(tied_logits, tied_scratch,
                 (uint32_t)(sizeof(tied_logits) / sizeof(tied_logits[0])),
                 0.01f, 0, 1.0f, 0.05f, "low-temperature");
    compare_case(tied_logits, tied_scratch,
                 (uint32_t)(sizeof(tied_logits) / sizeof(tied_logits[0])),
                 100.0f, 0, 1.0f, 0.05f, "high-temperature");
    compare_case(tied_logits, tied_scratch,
                 (uint32_t)(sizeof(tied_logits) / sizeof(tied_logits[0])),
                 1.0f, 0, 1.0f, 1.0f, "min-p-one");

    uint64_t null_ref_rng = 42;
    uint64_t null_opt_rng = 42;
    const int null_ref = reference_sample(
            tied_logits,
            (uint32_t)(sizeof(tied_logits) / sizeof(tied_logits[0])),
            1.0f, 0, 1.0f, 0.05f, &null_ref_rng);
    const int null_opt = ds4_test_sample_logits(
            tied_logits,
            (uint32_t)(sizeof(tied_logits) / sizeof(tied_logits[0])),
            1.0f, 0, 1.0f, 0.05f, &null_opt_rng, NULL);
    CHECK(null_ref == null_opt && null_ref_rng == null_opt_rng,
          "null probability scratch fallback mismatch");

    const float nonfinite_logits[] = {-INFINITY, NAN, INFINITY, NAN};
    float nonfinite_scratch[sizeof(nonfinite_logits) /
                            sizeof(nonfinite_logits[0])];
    compare_case(nonfinite_logits, nonfinite_scratch,
                 (uint32_t)(sizeof(nonfinite_logits) /
                            sizeof(nonfinite_logits[0])),
                 1.0f, 0, 1.0f, 0.05f, "all-nonfinite");

    free(logits);
    free(scratch);

    const uint32_t perf_n = 163840;
    const int iterations = 100;
    logits = malloc((size_t)perf_n * sizeof(*logits));
    scratch = malloc((size_t)perf_n * sizeof(*scratch));
    CHECK(logits && scratch, "performance scratch allocation");
    if (!logits || !scratch) return 1;
    fill_logits(logits, perf_n, 0xfeedfacecafebeefULL);

    uint64_t ref_rng = 1234;
    uint64_t checksum = 0;
    double start = now_sec();
    for (int i = 0; i < iterations; i++) {
        checksum += (uint64_t)reference_sample(logits, perf_n, 1.0f, 0,
                                               1.0f, 0.05f, &ref_rng);
    }
    const double reference_ms = (now_sec() - start) * 1000.0;

    uint64_t opt_rng = 1234;
    start = now_sec();
    for (int i = 0; i < iterations; i++) {
        checksum += (uint64_t)ds4_test_sample_logits(logits, perf_n, 1.0f, 0,
                                                     1.0f, 0.05f,
                                                     &opt_rng, scratch);
    }
    const double optimized_ms = (now_sec() - start) * 1000.0;
    CHECK(ref_rng == opt_rng, "performance RNG state mismatch");
    printf("sampling default: reference %.3f ms, optimized %.3f ms, %.2fx, checksum=%llu\n",
           reference_ms, optimized_ms,
           optimized_ms > 0.0 ? reference_ms / optimized_ms : 0.0,
           (unsigned long long)checksum);

    ref_rng = 5678;
    start = now_sec();
    for (int i = 0; i < iterations; i++) {
        checksum += (uint64_t)reference_sample(logits, perf_n, 1.0f, 0,
                                               0.9f, 0.05f, &ref_rng);
    }
    const double reference_top_p_ms = (now_sec() - start) * 1000.0;

    opt_rng = 5678;
    start = now_sec();
    for (int i = 0; i < iterations; i++) {
        checksum += (uint64_t)ds4_test_sample_logits(logits, perf_n, 1.0f, 0,
                                                     0.9f, 0.05f,
                                                     &opt_rng, scratch);
    }
    const double optimized_top_p_ms = (now_sec() - start) * 1000.0;
    CHECK(ref_rng == opt_rng, "top-p performance RNG state mismatch");
    printf("sampling top-p+min-p: reference %.3f ms, optimized %.3f ms, %.2fx, checksum=%llu\n",
           reference_top_p_ms, optimized_top_p_ms,
           optimized_top_p_ms > 0.0 ? reference_top_p_ms / optimized_top_p_ms : 0.0,
           (unsigned long long)checksum);

    free(logits);
    free(scratch);
    if (failures) {
        fprintf(stderr, "%d sampling test(s) failed\n", failures);
        return 1;
    }
    puts("sampling tests: OK");
    return 0;
}
