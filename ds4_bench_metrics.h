#ifndef DS4_BENCH_METRICS_H
#define DS4_BENCH_METRICS_H

typedef struct {
    int warmup_tokens;
    double warmup_sec;
    double warmup_tps;
    int measured_tokens;
    double measured_sec;
    double measured_tps;
} ds4_bench_decode_metrics;

ds4_bench_decode_metrics ds4_bench_decode_metrics_make(
    int gen_tokens,
    int requested_warmup_tokens,
    double total_start_sec,
    double measured_start_sec,
    double total_end_sec);

#endif
