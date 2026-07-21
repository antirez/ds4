#include "ds4_bench_metrics.h"

#include <math.h>
#include <stdio.h>

static int failures;

#define CHECK(cond, message) do {                                           \
    if (!(cond)) {                                                          \
        fprintf(stderr, "FAIL: %s (line %d)\n", (message), __LINE__);      \
        failures++;                                                         \
    }                                                                       \
} while (0)

static int close_enough(double a, double b) {
    return fabs(a - b) < 1e-9;
}

static void test_no_warmup_preserves_total_window(void) {
    ds4_bench_decode_metrics m =
        ds4_bench_decode_metrics_make(512, 0, 10.0, 99.0, 50.0);
    CHECK(m.warmup_tokens == 0, "zero warmup token count");
    CHECK(close_enough(m.warmup_sec, 0.0), "zero warmup duration");
    CHECK(m.measured_tokens == 512, "all tokens measured");
    CHECK(close_enough(m.measured_sec, 40.0), "total window measured");
    CHECK(close_enough(m.measured_tps, 12.8), "total throughput preserved");
}

static void test_32_token_warmup(void) {
    ds4_bench_decode_metrics m =
        ds4_bench_decode_metrics_make(512, 32, 10.0, 14.0, 50.0);
    CHECK(m.warmup_tokens == 32, "32 warmup tokens");
    CHECK(close_enough(m.warmup_sec, 4.0), "warmup window");
    CHECK(close_enough(m.warmup_tps, 8.0), "warmup throughput");
    CHECK(m.measured_tokens == 480, "480 measured tokens");
    CHECK(close_enough(m.measured_sec, 36.0), "measured window");
    CHECK(close_enough(m.measured_tps, 480.0 / 36.0), "measured throughput");
}

static void test_warmup_consumes_short_run(void) {
    ds4_bench_decode_metrics m =
        ds4_bench_decode_metrics_make(16, 32, 5.0, 6.0, 7.0);
    CHECK(m.warmup_tokens == 16, "warmup clamps to completed tokens");
    CHECK(m.measured_tokens == 0, "no measured tokens remain");
    CHECK(close_enough(m.warmup_sec, 2.0), "whole short run is warmup");
    CHECK(close_enough(m.measured_sec, 0.0), "zero measured duration");
    CHECK(close_enough(m.measured_tps, 0.0), "zero measured throughput");
}

static void test_zero_token_run(void) {
    ds4_bench_decode_metrics m =
        ds4_bench_decode_metrics_make(0, 0, 3.0, 3.0, 3.0);
    CHECK(m.warmup_tokens == 0, "zero-run warmup tokens");
    CHECK(m.measured_tokens == 0, "zero-run measured tokens");
    CHECK(close_enough(m.warmup_tps, 0.0), "zero-run warmup throughput");
    CHECK(close_enough(m.measured_tps, 0.0), "zero-run measured throughput");
}

int main(void) {
    test_no_warmup_preserves_total_window();
    test_32_token_warmup();
    test_warmup_consumes_short_run();
    test_zero_token_run();
    if (failures) {
        fprintf(stderr, "test_bench_metrics: %d failure(s)\n", failures);
        return 1;
    }
    fprintf(stderr, "test_bench_metrics: PASS\n");
    return 0;
}
