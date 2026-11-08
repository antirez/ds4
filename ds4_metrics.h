#ifndef DS4_METRICS_H
#define DS4_METRICS_H

#include <stddef.h>
#include <stdint.h>

/* Process-wide inference phase counters.
 *
 * Every counter is a monotonically increasing uint64: a call count, a token
 * count, a byte count, or nanoseconds.  Rates, averages and time shares are
 * derived by whoever reads them (ds4-server's /metrics, the CLI's /stats), so
 * the hot path only does one relaxed atomic add per phase and never a lock.
 *
 * Phases are accounted at the public engine API: ds4_session_sync() is
 * prefill, the ds4_session_eval*() family is decode.  Backend time spent
 * waiting for another node is also counted under network, so for a
 * distributed coordinator prefill_ns and decode_ns include net_recv_ns.
 *
 * The X-macro keeps the enum, the exported names and the help strings in
 * one place.  Nanosecond counters carry the _ns suffix and are exported as
 * _seconds_total by the Prometheus renderer. */
#define DS4_METRICS(X) \
    X(PREFILL_CALLS, "prefill_calls_total", "Prompt syncs executed.") \
    X(PREFILL_TOKENS, "prefill_tokens_total", "Prompt tokens evaluated by prefill.") \
    X(PREFILL_REUSED_TOKENS, "prefill_reused_tokens_total", "Prompt tokens served by the live KV prefix instead of prefill.") \
    X(PREFILL_NS, "prefill_ns", "Wall time spent in prefill, including mixed prefill/decode batches.") \
    X(DECODE_PASSES, "decode_passes_total", "Decode forward passes; tokens/passes is the speculative gain.") \
    X(DECODE_ROWS, "decode_rows_total", "Session rows across decode passes; rows/passes is the batch size.") \
    X(DECODE_TOKENS, "decode_tokens_total", "Tokens committed by decode.") \
    X(DECODE_NS, "decode_ns", "Wall time spent in decode passes.") \
    X(VISION_IMAGES, "vision_images_total", "Images run through the vision encoder.") \
    X(VISION_NS, "vision_ns", "Wall time spent encoding images.") \
    X(KV_SAVE_CALLS, "kv_save_calls_total", "Session KV payloads written.") \
    X(KV_SAVE_BYTES, "kv_save_bytes_total", "Session KV payload bytes written.") \
    X(KV_SAVE_NS, "kv_save_ns", "Wall time spent staging and writing KV payloads.") \
    X(KV_LOAD_CALLS, "kv_load_calls_total", "Session KV payloads restored.") \
    X(KV_LOAD_BYTES, "kv_load_bytes_total", "Session KV payload bytes restored.") \
    X(KV_LOAD_NS, "kv_load_ns", "Wall time spent restoring KV payloads.") \
    X(NET_SEND_BYTES, "net_send_bytes_total", "Bytes sent to other nodes (pipeline and TP).") \
    X(NET_SEND_NS, "net_send_ns", "Wall time blocked sending to other nodes.") \
    X(NET_RECV_BYTES, "net_recv_bytes_total", "Bytes received from other nodes.") \
    X(NET_RECV_NS, "net_recv_ns", "Wall time blocked waiting for other nodes; includes their compute.") \
    X(NET_GATES, "net_gates_total", "Tensor-parallel gate exchanges with the peer rank.") \
    X(NET_REMOTE_EVAL_NS, "net_remote_eval_ns", "Compute time reported by pipeline workers; recv minus this is transport.") \
    X(SSD_READ_BYTES, "ssd_read_bytes_total", "Routed-expert bytes streamed from disk.") \
    X(SSD_READ_NS, "ssd_read_ns", "Wall time spent streaming experts from disk.")

typedef enum {
#define DS4_METRIC_ENUM(id, name, help) DS4_METRIC_##id,
    DS4_METRICS(DS4_METRIC_ENUM)
#undef DS4_METRIC_ENUM
    DS4_METRIC_COUNT
} ds4_metric;

#ifdef __cplusplus
extern "C" {
#endif

void ds4_metric_add(ds4_metric m, uint64_t v);
uint64_t ds4_metric_now_ns(void);
const char *ds4_metric_name(ds4_metric m);
const char *ds4_metric_help(ds4_metric m);
/* True for counters measured in nanoseconds. */
int ds4_metric_is_ns(ds4_metric m);
void ds4_metrics_snapshot(uint64_t out[DS4_METRIC_COUNT]);
/* One-line human summary of the counters in `now`, or of the delta since
 * `prev` when it is not NULL: throughput per phase and the share of
 * `wall_seconds` each phase took. */
void ds4_metrics_summary(const uint64_t now[DS4_METRIC_COUNT],
                         const uint64_t prev[DS4_METRIC_COUNT],
                         double wall_seconds, char *buf, size_t cap);

#ifdef __cplusplus
}
#endif

#endif
