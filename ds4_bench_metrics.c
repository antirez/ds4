#include "ds4_bench_metrics.h"

ds4_bench_decode_metrics ds4_bench_decode_metrics_make(
        int gen_tokens,
        int requested_warmup_tokens,
        double total_start_sec,
        double measured_start_sec,
        double total_end_sec) {
    ds4_bench_decode_metrics m = {0};
    if (gen_tokens < 0) gen_tokens = 0;
    if (requested_warmup_tokens < 0) requested_warmup_tokens = 0;

    m.warmup_tokens = requested_warmup_tokens < gen_tokens
        ? requested_warmup_tokens : gen_tokens;
    m.measured_tokens = gen_tokens - m.warmup_tokens;

    double boundary = measured_start_sec;
    if (m.warmup_tokens == 0) boundary = total_start_sec;
    else if (m.measured_tokens == 0) boundary = total_end_sec;
    if (boundary < total_start_sec) boundary = total_start_sec;
    if (boundary > total_end_sec) boundary = total_end_sec;

    if (m.warmup_tokens > 0) m.warmup_sec = boundary - total_start_sec;
    if (m.measured_tokens > 0) m.measured_sec = total_end_sec - boundary;
    if (m.warmup_sec > 0.0)
        m.warmup_tps = (double)m.warmup_tokens / m.warmup_sec;
    if (m.measured_sec > 0.0)
        m.measured_tps = (double)m.measured_tokens / m.measured_sec;
    return m;
}
